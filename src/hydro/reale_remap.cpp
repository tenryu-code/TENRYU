#include "hydro/reale_remap.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "core/error.hpp"
#include "mesh/reale_contracts.hpp"

namespace tenryu::hydro::reale {

namespace detail {

inline constexpr std::size_t kDvclpNewtonDofCap = 256;
inline constexpr int kDvclpDykstraSweepCap = 256;
inline constexpr int kDvclpQuotientSweepCap = 64;

struct DvclpTestCaps {
  std::size_t dof_cap = kDvclpNewtonDofCap;
  int sweep_cap = kDvclpDykstraSweepCap;
  int quotient_sweep_cap = kDvclpQuotientSweepCap;
  int translation_sweep_cap = kDvclpDykstraSweepCap;
};

struct DvclpEdgeIncidence {
  int cell = -1;
  std::size_t corner0 = 0;
  std::size_t corner1 = 0;
};

struct DvclpEdgeBudget {
  double length = 0.0;
  double raw_jump = 0.0;
  double budget = 0.0;
  int node0 = -1;
  int node1 = -1;
  int walk_segments = 0;
  bool walk_failed = false;
  std::array<DvclpEdgeIncidence, 2> incident_corners{};
  int incident_count = 0;
};

struct DvclpProjectionResult {
  bool ok = true;
  std::string reason;
  int events = 0;
  int sweeps = 0;
  int alternations = 0;
  int consensus_ms = 0;
  int sweep_ms = 0;
  int solve_ms = 0;
  int certify_ms = 0;
  double q_total = 0.0;
  long double axis_projection_r = 0.0L;
  double max_R_before = 0.0;
  double max_R_after = 0.0;
  std::size_t unconstrained = 0;
  std::size_t representation_limited = 0;
  std::size_t consensus_edges = 0;
  std::size_t consensus_components = 0;
  std::size_t consensus_nodes = 0;
  std::size_t homothety_components = 0;
  std::size_t homothety_expansions = 0;
  std::size_t newton_capped = 0;
  double alpha_min = 1.0;
  std::size_t quotient_certified_edges = 0;
  std::size_t resolution_certified_edges = 0;
  std::size_t lattice_certified_edges = 0;
  int nonstar_fallback_cells = 0;
  int degenerate_skipped_triangles = 0;
  std::size_t kkt_certified_components = 0;
  std::size_t kkt_fail_eq = 0;
  std::size_t kkt_fail_stationarity = 0;
  std::size_t kkt_fail_cone = 0;
  double kkt_lam_max = 0.0;
  std::size_t qr_dropped_rows = 0;
};

RemapResult build_dvclp_edge_budgets_for_test(
    const RemapMesh& old_mesh,
    const RemapMesh& target_mesh,
    const RemapFields& old_fields,
    const double* target_node_vr,
    const double* target_node_vz,
    std::vector<DvclpEdgeBudget>& budgets,
    int* nonstar_fallback_cells,
    int* degenerate_skipped_triangles);

bool build_dvclp_edge_budgets_core(
    const RemapMesh& old_mesh,
    const RemapMesh& target_mesh,
    const RemapFields& old_fields,
    const double* target_node_vr,
    const double* target_node_vz,
    const std::vector<std::pair<int, int>>& ring_edges,
    const std::vector<double>& edge_lengths,
    const std::vector<std::vector<DvclpEdgeIncidence>>& edge_incidences,
    const std::vector<std::vector<int>>& candidate_cells,
    std::vector<DvclpEdgeBudget>& budgets,
    std::string& reason,
    int& nonstar_fallback_cells,
    int& degenerate_skipped_triangles);

DvclpProjectionResult project_dvclp_velocities_core(
    const RemapMesh& target_mesh,
    const std::vector<DvclpEdgeBudget>& budgets,
    const double* nodal_mass,
    const double* corner_mass,
    double* corner_momentum_r,
    double* corner_momentum_z,
    double* node_vr,
    double* node_vz,
    double* cell_q_projection,
    const DvclpTestCaps& test_caps = {},
    const int solver_rev = 0);

}  // namespace detail

namespace {

constexpr double kEps64 =
    64.0 * std::numeric_limits<double>::epsilon();
constexpr long double kPiLocalL =
    3.14159265358979323846264338328L;
constexpr long double kTwoPiLocalL =
    6.283185307179586476925286766559L;

using Moments = mesh::reale::RZMoments;

// Host-only long double is deliberate: it lowers the fragment-sum rounding
// floor below the 64*eps additivity identity gate.
struct LocalMoments {
  long double area = 0.0L;
  long double ir = 0.0L;
  long double irr = 0.0L;
  long double iz = 0.0L;
  long double irz = 0.0L;
};

LocalMoments local_planar_moments(const double* r,
                                  const double* z,
                                  const int n) {
  LocalMoments sums{};
  for (int k = 0; k < n; ++k) {
    const int next = (k + 1 == n) ? 0 : k + 1;
    const long double cross =
        static_cast<long double>(r[k]) * z[next] -
        static_cast<long double>(r[next]) * z[k];
    sums.area += cross;
    sums.ir += (static_cast<long double>(r[k]) + r[next]) * cross;
    sums.irr +=
        (static_cast<long double>(r[k]) * r[k] +
         static_cast<long double>(r[k]) * r[next] +
         static_cast<long double>(r[next]) * r[next]) *
        cross;
    sums.iz += (static_cast<long double>(z[k]) + z[next]) * cross;
    sums.irz +=
        (2.0L * static_cast<long double>(r[k]) * z[k] +
         static_cast<long double>(r[k]) * z[next] +
         static_cast<long double>(r[next]) * z[k] +
         2.0L * static_cast<long double>(r[next]) * z[next]) *
        cross;
  }
  sums.area *= 0.5L;
  sums.ir *= 1.0L / 6.0L;
  sums.irr *= 1.0L / 12.0L;
  sums.iz *= 1.0L / 6.0L;
  sums.irz *= 1.0L / 24.0L;
  return sums;
}

Moments reconstruct_rz(const LocalMoments& moments,
                       const double r0,
                       const double z0) {
  const long double r0_ld = static_cast<long double>(r0);
  const long double z0_ld = static_cast<long double>(z0);
  const long double volume =
      kTwoPiLocalL * (moments.ir + r0_ld * moments.area);
  const long double moment_r =
      kTwoPiLocalL *
      (moments.irr + 2.0L * r0_ld * moments.ir +
       r0_ld * r0_ld * moments.area);
  const long double moment_z =
      kTwoPiLocalL *
      (moments.irz + r0_ld * moments.iz + z0_ld * moments.ir +
       r0_ld * z0_ld * moments.area);
  Moments result{};
  result.volume = static_cast<double>(volume);
  result.moment_r = static_cast<double>(moment_r);
  result.moment_z = static_cast<double>(moment_z);
  return result;
}

int clip_by_convex(const double* subj_r,
                   const double* subj_z,
                   const int subj_n,
                   const double* clip_r,
                   const double* clip_z,
                   const int clip_n,
                   double* out_r,
                   double* out_z,
                   const int cap) {
  constexpr int kWorkCapacity = 32;
  if (subj_n > cap || cap > kWorkCapacity) {
    return -1;
  }

  double doubled = 0.0;
  for (int k = 0; k < clip_n; ++k) {
    const int next = (k + 1 == clip_n) ? 0 : k + 1;
    doubled += clip_r[k] * clip_z[next] - clip_r[next] * clip_z[k];
  }
  if (doubled == 0.0) {
    return 0;
  }
  const double orient = doubled > 0.0 ? 1.0 : -1.0;

  std::array<double, kWorkCapacity> scratch_r{};
  std::array<double, kWorkCapacity> scratch_z{};
  const double* current_r = subj_r;
  const double* current_z = subj_z;
  int current_n = subj_n;
  double* next_r = out_r;
  double* next_z = out_z;

  for (int edge = 0; edge < clip_n; ++edge) {
    if (current_n == 0) {
      return 0;
    }
    const int edge_next = (edge + 1 == clip_n) ? 0 : edge + 1;
    const double a_r = clip_r[edge];
    const double a_z = clip_z[edge];
    const double edge_dr = clip_r[edge_next] - a_r;
    const double edge_dz = clip_z[edge_next] - a_z;

    int next_n = 0;
    int p = current_n - 1;
    double side_p = orient *
                    (edge_dr * (current_z[p] - a_z) -
                     edge_dz * (current_r[p] - a_r));
    bool p_inside = side_p >= 0.0;
    for (int q = 0; q < current_n; ++q) {
      const double side_q = orient *
                            (edge_dr * (current_z[q] - a_z) -
                             edge_dz * (current_r[q] - a_r));
      const bool q_inside = side_q >= 0.0;
      if (p_inside != q_inside) {
        if (next_n >= cap) {
          return -1;
        }
        const double t = side_p / (side_p - side_q);
        next_r[next_n] =
            current_r[p] + t * (current_r[q] - current_r[p]);
        next_z[next_n] =
            current_z[p] + t * (current_z[q] - current_z[p]);
        ++next_n;
      }
      if (q_inside) {
        if (next_n >= cap) {
          return -1;
        }
        next_r[next_n] = current_r[q];
        next_z[next_n] = current_z[q];
        ++next_n;
      }
      p = q;
      side_p = side_q;
      p_inside = q_inside;
    }

    current_r = next_r;
    current_z = next_z;
    current_n = next_n;
    if (next_r == out_r) {
      next_r = scratch_r.data();
      next_z = scratch_z.data();
    } else {
      next_r = out_r;
      next_z = out_z;
    }
  }

  if (current_r != out_r) {
    for (int k = 0; k < current_n; ++k) {
      out_r[k] = current_r[k];
      out_z[k] = current_z[k];
    }
  }
  return current_n;
}

struct ClipHalfPlane {
  double a_r = 0.0;
  double a_z = 0.0;
  double b_r = 0.0;
  double b_z = 0.0;
  double orient = 1.0;
};

int clip_by_half_planes(const double* subj_r,
                        const double* subj_z,
                        const int subj_n,
                        const std::vector<ClipHalfPlane>& half_planes,
                        const double origin_r,
                        const double origin_z,
                        double* out_r,
                        double* out_z,
                        const int cap) {
  constexpr int kWorkCapacity = 32;
  if (subj_n > cap || cap > kWorkCapacity) {
    return -1;
  }

  std::array<double, kWorkCapacity> scratch_r{};
  std::array<double, kWorkCapacity> scratch_z{};
  const double* current_r = subj_r;
  const double* current_z = subj_z;
  int current_n = subj_n;
  double* next_r = out_r;
  double* next_z = out_z;

  for (const ClipHalfPlane& half_plane : half_planes) {
    if (current_n == 0) {
      return 0;
    }
    const double a_r = half_plane.a_r - origin_r;
    const double a_z = half_plane.a_z - origin_z;
    const double edge_dr =
        (half_plane.b_r - origin_r) - a_r;
    const double edge_dz =
        (half_plane.b_z - origin_z) - a_z;

    int next_n = 0;
    int p = current_n - 1;
    double side_p =
        half_plane.orient *
        (edge_dr * (current_z[p] - a_z) -
         edge_dz * (current_r[p] - a_r));
    bool p_inside = side_p >= 0.0;
    for (int q = 0; q < current_n; ++q) {
      const double side_q =
          half_plane.orient *
          (edge_dr * (current_z[q] - a_z) -
           edge_dz * (current_r[q] - a_r));
      const bool q_inside = side_q >= 0.0;
      if (p_inside != q_inside) {
        if (next_n >= cap) {
          return -1;
        }
        const double t = side_p / (side_p - side_q);
        next_r[next_n] =
            current_r[p] + t * (current_r[q] - current_r[p]);
        next_z[next_n] =
            current_z[p] + t * (current_z[q] - current_z[p]);
        ++next_n;
      }
      if (q_inside) {
        if (next_n >= cap) {
          return -1;
        }
        next_r[next_n] = current_r[q];
        next_z[next_n] = current_z[q];
        ++next_n;
      }
      p = q;
      side_p = side_q;
      p_inside = q_inside;
    }

    current_r = next_r;
    current_z = next_z;
    current_n = next_n;
    if (next_r == out_r) {
      next_r = scratch_r.data();
      next_z = scratch_z.data();
    } else {
      next_r = out_r;
      next_z = out_z;
    }
  }

  if (current_r != out_r) {
    for (int k = 0; k < current_n; ++k) {
      out_r[k] = current_r[k];
      out_z[k] = current_z[k];
    }
  }
  return current_n;
}

struct Bounds {
  double r_min = std::numeric_limits<double>::infinity();
  double r_max = -std::numeric_limits<double>::infinity();
  double z_min = std::numeric_limits<double>::infinity();
  double z_max = -std::numeric_limits<double>::infinity();
};

struct CellGeometry {
  std::vector<double> r;
  std::vector<double> z;
  Moments moments{};
  Bounds bounds{};
  double coord_scale = 0.0;
  double perimeter = 0.0;
};

// each snapped vertex carries <= 1 ulp(s) positional noise; a tiling cut of
// length l between two pieces contributes at most ~ulp(s) * l of planar area
// mismatch counted once per side (factor 2); total cut length across a cell's
// pieces is bounded by 2x its perimeter; the shoelace 1/2 cancels one factor 2;
// RZ conversion multiplies by 2*pi*r_bar <= 2*pi*s. Collapsed constants:
// 2*pi*s * ulp(s) * 2*perimeter = 4*pi*eps*s^2*perimeter. No fitted constants.
long double snap_additivity_bound(const CellGeometry& cell) {
  return 4.0L * kPiLocalL *
         (long double)std::numeric_limits<double>::epsilon() *
         cell.coord_scale * cell.coord_scale * cell.perimeter;
}

struct PairMoments {
  Moments moments;
  long double volume_ld;
};

struct TargetFan {
  double center_r = 0.0;
  double center_z = 0.0;
};

bool build_target_fans(const std::vector<CellGeometry>& target_cells,
                       std::vector<TargetFan>& target_fans,
                       std::string& reason) {
  target_fans.resize(target_cells.size());
  for (std::size_t cell = 0; cell < target_cells.size(); ++cell) {
    const CellGeometry& target = target_cells[cell];
    TargetFan& fan = target_fans[cell];
    for (std::size_t vertex = 0; vertex < target.r.size(); ++vertex) {
      fan.center_r += target.r[vertex];
      fan.center_z += target.z[vertex];
    }
    fan.center_r /= static_cast<double>(target.r.size());
    fan.center_z /= static_cast<double>(target.z.size());

    // build_geometry accepts only positive signed RZ volumes, so target rings
    // use the CCW orientation and every valid fan triangle has positive area.
    for (std::size_t triangle = 0; triangle < target.r.size(); ++triangle) {
      const std::size_t next =
          triangle + 1 == target.r.size() ? 0U : triangle + 1;
      const double area =
          0.5 * ((target.r[triangle] - fan.center_r) *
                     (target.z[next] - fan.center_z) -
                 (target.r[next] - fan.center_r) *
                     (target.z[triangle] - fan.center_z));
      if (!(area > 0.0)) {
        char buffer[192];
        std::snprintf(
            buffer, sizeof(buffer),
            "target_fan_nonstar: cell=%zu tri=%zu area=%.17e",
            cell, triangle, area);
        reason = buffer;
        return false;
      }
    }
  }
  return true;
}

void print_audit_pair_polygon(const int new_cell,
                              const double* clipped_r,
                              const double* clipped_z,
                              const int clipped_n,
                              const double origin_r,
                              const double origin_z) {
  if (new_cell < 0 || clipped_n <= 0) {
    return;
  }
  std::fprintf(stderr, "[remap_audit] pair_poly new_cell=%d n=%d coords=",
               new_cell, clipped_n);
  for (int vertex = 0; vertex < clipped_n; ++vertex) {
    std::fprintf(stderr, "%s%.17e,%.17e", vertex == 0 ? "" : ";",
                 clipped_r[vertex] + origin_r,
                 clipped_z[vertex] + origin_z);
  }
  std::fputc('\n', stderr);
}

// Local-frame polygon integrals stay cell-scale, while the common r0/z0
// reconstruction terms factor over additive areas and telescope exactly.
PairMoments intersect_pair_local(const CellGeometry& source,
                                 const CellGeometry& target,
                                 const TargetFan& target_fan,
                                 const int audit_new_cell) {
  const double r0 = source.r[0];
  const double z0 = source.z[0];
  std::array<double, 16> source_r{};
  std::array<double, 16> source_z{};
  for (std::size_t k = 0; k < source.r.size(); ++k) {
    source_r[k] = source.r[k] - r0;
    source_z[k] = source.z[k] - z0;
  }

  // Fan triangles tile a validated star-shaped target ring with matching CCW
  // orientation, so each Sutherland-Hodgman clip polygon remains convex even
  // when welding introduces a micro-concave target edge. This performs about
  // target.r.size() times more host-side clip calls; acceptable because remap
  // runs once per rezone. No exact arithmetic is added.
  std::array<double, 32> clipped_r{};
  std::array<double, 32> clipped_z{};
  LocalMoments sum{};
  for (std::size_t triangle = 0; triangle < target.r.size(); ++triangle) {
    const std::size_t next =
        triangle + 1 == target.r.size() ? 0U : triangle + 1;
    const double triangle_r[3] = {
        target_fan.center_r - r0, target.r[triangle] - r0,
        target.r[next] - r0};
    const double triangle_z[3] = {
        target_fan.center_z - z0, target.z[triangle] - z0,
        target.z[next] - z0};
    const int clipped_n = clip_by_convex(
        source_r.data(), source_z.data(),
        static_cast<int>(source.r.size()), triangle_r, triangle_z, 3,
        clipped_r.data(), clipped_z.data(), 32);
    print_audit_pair_polygon(audit_new_cell, clipped_r.data(),
                             clipped_z.data(), clipped_n, r0, z0);
    if (clipped_n == -1) {
      const double nan = std::numeric_limits<double>::quiet_NaN();
      Moments invalid{};
      invalid.volume = nan;
      invalid.moment_r = nan;
      invalid.moment_z = nan;
      return {invalid, std::numeric_limits<long double>::quiet_NaN()};
    }
    if (clipped_n < 3) {
      continue;
    }

    const LocalMoments piece = local_planar_moments(
        clipped_r.data(), clipped_z.data(), clipped_n);
    sum.area += piece.area;
    sum.ir += piece.ir;
    sum.irr += piece.irr;
    sum.iz += piece.iz;
    sum.irz += piece.irz;
  }

  return {reconstruct_rz(sum, r0, z0),
          kTwoPiLocalL *
              (sum.ir + static_cast<long double>(r0) * sum.area)};
}

void print_audit_half_planes(const RemapMesh& new_mesh,
                             const int new_cell,
                             const std::vector<int>& targets) {
  std::size_t k = 0;
  for (const int u : targets) {
    if (u == new_cell) {
      continue;
    }
    std::fprintf(
        stderr,
        "[remap_audit] hp new_cell=%d k=%zu kind=bisector "
        "gen_t=%.17e,%.17e gen_n=%.17e,%.17e\n",
        new_cell, k, new_mesh.generator_r[new_cell],
        new_mesh.generator_z[new_cell], new_mesh.generator_r[u],
        new_mesh.generator_z[u]);
    ++k;
  }
}

PairMoments intersect_pair_local(
    const CellGeometry& source,
    const std::vector<ClipHalfPlane>& target_half_planes,
    const int audit_new_cell) {
  const double r0 = source.r[0];
  const double z0 = source.z[0];
  std::array<double, 16> source_r{};
  std::array<double, 16> source_z{};
  for (std::size_t k = 0; k < source.r.size(); ++k) {
    source_r[k] = source.r[k] - r0;
    source_z[k] = source.z[k] - z0;
  }

  std::array<double, 32> clipped_r{};
  std::array<double, 32> clipped_z{};
  const int clipped_n = clip_by_half_planes(
      source_r.data(), source_z.data(), static_cast<int>(source.r.size()),
      target_half_planes, r0, z0, clipped_r.data(), clipped_z.data(), 32);
  print_audit_pair_polygon(audit_new_cell, clipped_r.data(), clipped_z.data(),
                           clipped_n, r0, z0);
  if (clipped_n == -1) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Moments invalid{};
    invalid.volume = nan;
    invalid.moment_r = nan;
    invalid.moment_z = nan;
    return {invalid, std::numeric_limits<long double>::quiet_NaN()};
  }
  if (clipped_n < 3) {
    return {{}, 0.0L};
  }

  const LocalMoments sum = local_planar_moments(
      clipped_r.data(), clipped_z.data(), clipped_n);
  return {reconstruct_rz(sum, r0, z0),
          kTwoPiLocalL *
              (sum.ir + static_cast<long double>(r0) * sum.area)};
}

struct OverlayEntry {
  int old_cell = -1;
  int new_cell = -1;
  Moments moments{};
  double coefficient = 0.0;
};

struct SourceContribution {
  int source_node = -1;
  double mass = 0.0;
  double momentum_r = 0.0;
  double momentum_z = 0.0;
  double electron_internal = 0.0;
  double ion_internal = 0.0;
  double kinetic_cons = 0.0;
};

struct CornerPacket {
  double mass = 0.0;
  double electron_internal = 0.0;
  double ion_internal = 0.0;
  double kinetic_cons = 0.0;
};

struct AffineDensity {
  bool active = false;
  long double constant = 0.0L;
  long double radial = 0.0L;
  long double axial = 0.0L;
};

RemapResult reject(const std::string& reason) {
  RemapResult result;
  result.reject_reason = reason;
  return result;
}

bool relative_identity(const long double lhs, const long double rhs) {
  const long double scale = std::max(
      {std::abs(lhs), std::abs(rhs),
       static_cast<long double>(std::numeric_limits<double>::min())});
  return std::abs(lhs - rhs) <=
         static_cast<long double>(kEps64) * scale;
}

bool bounds_overlap(const Bounds& lhs, const Bounds& rhs) {
  return lhs.r_min <= rhs.r_max && rhs.r_min <= lhs.r_max &&
         lhs.z_min <= rhs.z_max && rhs.z_min <= lhs.z_max;
}

bool valid_fields(const RemapFields& fields) {
  return fields.mass != nullptr && fields.ee != nullptr &&
         fields.ei != nullptr && fields.corner_mass != nullptr &&
         fields.node_vr != nullptr && fields.node_vz != nullptr;
}

bool valid_outputs(double* new_mass,
                   double* new_ee,
                   double* new_ei,
                   double* new_corner_mass,
                   double* new_node_vr,
                   double* new_node_vz) {
  return new_mass != nullptr && new_ee != nullptr && new_ei != nullptr &&
         new_corner_mass != nullptr && new_node_vr != nullptr &&
         new_node_vz != nullptr;
}

bool build_geometry(const RemapMesh& mesh_view,
                    std::vector<CellGeometry>& cells,
                    int& node_count,
                    std::string& reason) {
  if (mesh_view.n_cells <= 0 || mesh_view.corner_stride < 3 ||
      mesh_view.csr_offsets == nullptr ||
      mesh_view.csr_indices == nullptr || mesh_view.nverts == nullptr ||
      mesh_view.node_r == nullptr || mesh_view.node_z == nullptr) {
    reason = "invalid_mesh";
    return false;
  }
  if (mesh_view.csr_offsets[0] < 0) {
    reason = "invalid_csr";
    return false;
  }

  cells.resize(static_cast<std::size_t>(mesh_view.n_cells));
  node_count = 0;
  for (int cell_id = 0; cell_id < mesh_view.n_cells; ++cell_id) {
    const int begin = mesh_view.csr_offsets[cell_id];
    const int end = mesh_view.csr_offsets[cell_id + 1];
    const int nverts = static_cast<int>(mesh_view.nverts[cell_id]);
    if (begin < 0 || end < begin || nverts < 3 ||
        nverts > mesh_view.corner_stride ||
        end - begin < nverts) {
      reason = "invalid_csr";
      return false;
    }

    CellGeometry& cell = cells[static_cast<std::size_t>(cell_id)];
    cell.r.reserve(static_cast<std::size_t>(nverts));
    cell.z.reserve(static_cast<std::size_t>(nverts));
    for (int corner = 0; corner < nverts; ++corner) {
      const int node = mesh_view.csr_indices[begin + corner];
      if (node < 0) {
        reason = "invalid_node";
        return false;
      }
      node_count = std::max(node_count, node + 1);
      const double r = mesh_view.node_r[node];
      const double z = mesh_view.node_z[node];
      if (!std::isfinite(r) || !std::isfinite(z) || r < 0.0) {
        reason = "invalid_coordinate";
        return false;
      }
      cell.r.push_back(r);
      cell.z.push_back(z);
      cell.bounds.r_min = std::min(cell.bounds.r_min, r);
      cell.bounds.r_max = std::max(cell.bounds.r_max, r);
      cell.bounds.z_min = std::min(cell.bounds.z_min, z);
      cell.bounds.z_max = std::max(cell.bounds.z_max, z);
      cell.coord_scale =
          std::max(cell.coord_scale, std::max(std::abs(r), std::abs(z)));
    }
    for (int k = 0; k < nverts; ++k) {
      const int next = (k + 1 == nverts) ? 0 : k + 1;
      cell.perimeter +=
          std::hypot(cell.r[next] - cell.r[k], cell.z[next] - cell.z[k]);
    }
    const double r0 = cell.r[0];
    const double z0 = cell.z[0];
    std::array<double, 16> local_r{};
    std::array<double, 16> local_z{};
    for (int k = 0; k < nverts; ++k) {
      local_r[static_cast<std::size_t>(k)] = cell.r[k] - r0;
      local_z[static_cast<std::size_t>(k)] = cell.z[k] - z0;
    }
    cell.moments = reconstruct_rz(
        local_planar_moments(local_r.data(), local_z.data(), nverts),
        r0, z0);
    if (!(cell.moments.volume > 0.0) ||
        !std::isfinite(cell.moments.volume) ||
        !std::isfinite(cell.moments.moment_r) ||
        !std::isfinite(cell.moments.moment_z)) {
      reason = "nonpositive_cell_volume";
      return false;
    }
  }
  return true;
}

class UniformSpatialHash {
 public:
  explicit UniformSpatialHash(const std::vector<CellGeometry>& cells) {
    for (const CellGeometry& cell : cells) {
      domain_.r_min = std::min(domain_.r_min, cell.bounds.r_min);
      domain_.r_max = std::max(domain_.r_max, cell.bounds.r_max);
      domain_.z_min = std::min(domain_.z_min, cell.bounds.z_min);
      domain_.z_max = std::max(domain_.z_max, cell.bounds.z_max);
    }
    bins_per_axis_ = std::max(
        1, static_cast<int>(std::ceil(std::sqrt(
               static_cast<double>(cells.size())))));
    buckets_.resize(static_cast<std::size_t>(bins_per_axis_ *
                                             bins_per_axis_));
    for (std::size_t cell_id = 0; cell_id < cells.size(); ++cell_id) {
      const Bounds& bounds = cells[cell_id].bounds;
      const int r_begin = r_bin(bounds.r_min);
      const int r_end = r_bin(bounds.r_max);
      const int z_begin = z_bin(bounds.z_min);
      const int z_end = z_bin(bounds.z_max);
      for (int z = z_begin; z <= z_end; ++z) {
        for (int r = r_begin; r <= r_end; ++r) {
          buckets_[bucket_index(r, z)].push_back(
              static_cast<int>(cell_id));
        }
      }
    }
  }

  std::vector<int> candidates(const Bounds& bounds) const {
    std::vector<int> result;
    const int r_begin = r_bin(bounds.r_min);
    const int r_end = r_bin(bounds.r_max);
    const int z_begin = z_bin(bounds.z_min);
    const int z_end = z_bin(bounds.z_max);
    for (int z = z_begin; z <= z_end; ++z) {
      for (int r = r_begin; r <= r_end; ++r) {
        for (const int cell : buckets_[bucket_index(r, z)]) {
          result.push_back(cell);
        }
      }
    }

    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
  }

 private:
  int coordinate_bin(const double coordinate,
                     const double minimum,
                     const double maximum) const {
    if (bins_per_axis_ == 1 || !(maximum > minimum)) {
      return 0;
    }
    const double scaled =
        (coordinate - minimum) / (maximum - minimum) * bins_per_axis_;
    return std::clamp(static_cast<int>(std::floor(scaled)),
                      0, bins_per_axis_ - 1);
  }

  int r_bin(const double r) const {
    return coordinate_bin(r, domain_.r_min, domain_.r_max);
  }

  int z_bin(const double z) const {
    return coordinate_bin(z, domain_.z_min, domain_.z_max);
  }

  std::size_t bucket_index(const int r, const int z) const {
    return static_cast<std::size_t>(z * bins_per_axis_ + r);
  }

  Bounds domain_{};
  int bins_per_axis_ = 1;
  std::vector<std::vector<int>> buckets_;
};

struct RingEdge {
  double length = 0.0;
  int node0 = -1;
  int node1 = -1;
  std::vector<detail::DvclpEdgeIncidence> incident_corners;
};

std::vector<RingEdge> enumerate_ring_edges(const RemapMesh& mesh) {
  std::map<std::pair<int, int>, std::size_t> seen;
  std::vector<RingEdge> edges;
  for (int cell = 0; cell < mesh.n_cells; ++cell) {
    const int begin = mesh.csr_offsets[cell];
    const int nverts = static_cast<int>(mesh.nverts[cell]);
    for (int corner = 0; corner < nverts; ++corner) {
      const int next = corner + 1 == nverts ? 0 : corner + 1;
      int node0 = mesh.csr_indices[begin + corner];
      int node1 = mesh.csr_indices[begin + next];
      std::size_t corner0 =
          static_cast<std::size_t>(cell) * mesh.corner_stride +
          static_cast<std::size_t>(corner);
      std::size_t corner1 =
          static_cast<std::size_t>(cell) * mesh.corner_stride +
          static_cast<std::size_t>(next);
      if (node1 < node0) {
        std::swap(node0, node1);
        std::swap(corner0, corner1);
      }
      const std::pair<int, int> key{node0, node1};
      const detail::DvclpEdgeIncidence incidence{
          cell, corner0, corner1};
      const auto found = seen.find(key);
      if (found != seen.end()) {
        edges[found->second].incident_corners.push_back(incidence);
        continue;
      }
      RingEdge edge;
      edge.node0 = node0;
      edge.node1 = node1;
      edge.length = std::hypot(mesh.node_r[node1] - mesh.node_r[node0],
                               mesh.node_z[node1] - mesh.node_z[node0]);
      edge.incident_corners.push_back(incidence);
      seen.emplace(key, edges.size());
      edges.push_back(edge);
    }
  }
  return edges;
}

bool build_dvclp_edge_budgets(
    const RemapMesh& old_mesh,
    const RemapMesh& target_mesh,
    const std::vector<CellGeometry>& old_cells,
    const RemapFields& old_fields,
    const double* target_node_vr,
    const double* target_node_vz,
    std::vector<detail::DvclpEdgeBudget>& budgets,
    std::string& reason,
    int& nonstar_fallback_cells,
    int& degenerate_skipped_triangles) {
  const UniformSpatialHash spatial_hash(old_cells);
  const std::vector<RingEdge> ring_edges =
      enumerate_ring_edges(target_mesh);
  std::vector<std::pair<int, int>> edge_nodes;
  std::vector<double> edge_lengths;
  std::vector<std::vector<detail::DvclpEdgeIncidence>> edge_incidences;
  std::vector<std::vector<int>> candidate_cells;
  edge_nodes.reserve(ring_edges.size());
  edge_lengths.reserve(ring_edges.size());
  edge_incidences.reserve(ring_edges.size());
  candidate_cells.reserve(ring_edges.size());
  for (const RingEdge& ring_edge : ring_edges) {
    edge_nodes.emplace_back(ring_edge.node0, ring_edge.node1);
    edge_lengths.push_back(ring_edge.length);
    edge_incidences.push_back(ring_edge.incident_corners);
    Bounds segment_bounds;
    segment_bounds.r_min = std::min(target_mesh.node_r[ring_edge.node0],
                                    target_mesh.node_r[ring_edge.node1]);
    segment_bounds.r_max = std::max(target_mesh.node_r[ring_edge.node0],
                                    target_mesh.node_r[ring_edge.node1]);
    segment_bounds.z_min = std::min(target_mesh.node_z[ring_edge.node0],
                                    target_mesh.node_z[ring_edge.node1]);
    segment_bounds.z_max = std::max(target_mesh.node_z[ring_edge.node0],
                                    target_mesh.node_z[ring_edge.node1]);
    candidate_cells.push_back(spatial_hash.candidates(segment_bounds));
  }
  return detail::build_dvclp_edge_budgets_core(
      old_mesh, target_mesh, old_fields, target_node_vr, target_node_vz,
      edge_nodes, edge_lengths, edge_incidences, candidate_cells, budgets,
      reason, nonstar_fallback_cells, degenerate_skipped_triangles);
}

double dvclp_ratio(const detail::DvclpEdgeBudget& edge) {
  if (edge.budget == 0.0) {
    return edge.raw_jump == 0.0
               ? 0.0
               : std::numeric_limits<double>::infinity();
  }
  return edge.raw_jump / edge.budget;
}

void print_dvclp_diagnostic(
    const std::vector<detail::DvclpEdgeBudget>& budgets) {
  std::vector<std::size_t> successful;
  successful.reserve(budgets.size());
  std::size_t failed_count = 0;
  std::size_t ratio_gt_one = 0;
  double max_ratio = 0.0;
  for (std::size_t edge_index = 0; edge_index < budgets.size();
       ++edge_index) {
    const detail::DvclpEdgeBudget& edge = budgets[edge_index];
    if (edge.walk_failed) {
      ++failed_count;
      std::fprintf(stderr,
                   "[dvclp] walk_failed L=%.3e D=%.3e n0=%d n1=%d "
                   "walk_segs=%d\n",
                   edge.length, edge.raw_jump, edge.node0, edge.node1,
                   edge.walk_segments);
      continue;
    }
    successful.push_back(edge_index);
    const double ratio = dvclp_ratio(edge);
    if (ratio > 1.0) {
      ++ratio_gt_one;
    }
    max_ratio = std::max(max_ratio, ratio);
  }
  const std::size_t selected_count =
      std::min<std::size_t>(16, successful.size());
  std::partial_sort(
      successful.begin(), successful.begin() + selected_count,
      successful.end(),
      [&budgets](const std::size_t left_index,
                 const std::size_t right_index) {
        const detail::DvclpEdgeBudget& left = budgets[left_index];
        const detail::DvclpEdgeBudget& right = budgets[right_index];
        const double left_ratio = dvclp_ratio(left);
        const double right_ratio = dvclp_ratio(right);
        if (left_ratio != right_ratio) {
          return left_ratio > right_ratio;
        }
        if (left.node0 != right.node0) {
          return left.node0 < right.node0;
        }
        return left.node1 < right.node1;
      });
  for (std::size_t selected = 0; selected < selected_count; ++selected) {
    const detail::DvclpEdgeBudget& edge = budgets[successful[selected]];
    std::fprintf(stderr,
                 "[dvclp] L=%.3e D=%.3e B=%.3e R=%.3e n0=%d n1=%d "
                 "walk_segs=%d\n",
                 edge.length, edge.raw_jump, edge.budget,
                 dvclp_ratio(edge), edge.node0, edge.node1,
                 edge.walk_segments);
  }
  std::fprintf(stderr,
               "[dvclp] edges=%zu budgets_ok=%zu walk_failed=%zu "
               "R_gt_1=%zu max_R=%.3e\n",
               budgets.size(), successful.size(), failed_count,
               ratio_gt_one, max_ratio);
}

bool remap_timing_enabled() {
  static const bool enabled = [] {
    const char* const value = std::getenv("TENRYU_TESS_TIMING");
    return value != nullptr && std::string(value) == std::string("1");
  }();
  return enabled;
}

double remap_ms_between(
    const std::chrono::steady_clock::time_point& begin,
    const std::chrono::steady_clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool dvclp_diagnostic_enabled() {
  static const char* dvclp_env = std::getenv("TENRYU_DVCLP_DIAG");
  return dvclp_env != nullptr && std::atoi(dvclp_env) == 1;
}

bool run_dvclp_projection(
    const RemapMesh& old_mesh,
    const RemapMesh& target_mesh,
    const std::vector<CellGeometry>& old_cells,
    const RemapFields& old_fields,
    const std::vector<double>& nodal_mass,
    const std::vector<double>& corner_mass,
    std::vector<double>& corner_momentum_r,
    std::vector<double>& corner_momentum_z,
    std::vector<double>& target_node_vr,
    std::vector<double>& target_node_vz,
    std::vector<double>& cell_q_projection,
    std::vector<detail::DvclpEdgeBudget>& budgets,
    detail::DvclpProjectionResult& projection,
    std::string& reason,
    const int dvclp_solver_rev) {
  int nonstar_fallback_cells = 0;
  int degenerate_skipped_triangles = 0;
  if (!build_dvclp_edge_budgets(
          old_mesh, target_mesh, old_cells, old_fields,
          target_node_vr.data(), target_node_vz.data(), budgets, reason,
          nonstar_fallback_cells, degenerate_skipped_triangles)) {
    return false;
  }
  if (dvclp_diagnostic_enabled()) {
    print_dvclp_diagnostic(budgets);
  }
  projection = detail::project_dvclp_velocities_core(
      target_mesh, budgets, nodal_mass.data(), corner_mass.data(),
      corner_momentum_r.data(), corner_momentum_z.data(),
      target_node_vr.data(), target_node_vz.data(),
      cell_q_projection.data(), detail::DvclpTestCaps{}, dvclp_solver_rev);
  projection.nonstar_fallback_cells = nonstar_fallback_cells;
  projection.degenerate_skipped_triangles =
      degenerate_skipped_triangles;
  if (dvclp_diagnostic_enabled()) {
    std::fprintf(
        stderr,
        "[dvclp_proj] events=%d sweeps=%d q_total=%.3e "
        "maxR_before=%.3e maxR_after=%.3e unconstrained=%zu "
        "representation_limited=%zu consensus_edges=%zu "
        "consensus_components=%zu consensus_nodes=%zu alternations=%d "
        "homothety_components=%zu homothety_expansions=%zu "
        "alpha_min=%.3e consensus_ms=%d sweep_ms=%d solve_ms=%d "
        "certify_ms=%d newton_capped=%zu quotient=%zu resolution=%zu "
        "lattice=%zu nonstar_fb=%d degen_skip=%d\n",
        projection.events, projection.sweeps, projection.q_total,
        projection.max_R_before, projection.max_R_after,
        projection.unconstrained, projection.representation_limited,
        projection.consensus_edges, projection.consensus_components,
        projection.consensus_nodes, projection.alternations,
        projection.homothety_components, projection.homothety_expansions,
        projection.alpha_min, projection.consensus_ms, projection.sweep_ms,
        projection.solve_ms, projection.certify_ms, projection.newton_capped,
        projection.quotient_certified_edges,
        projection.resolution_certified_edges,
        projection.lattice_certified_edges,
        projection.nonstar_fallback_cells,
        projection.degenerate_skipped_triangles);
  }
  if (!projection.ok) {
    reason = projection.reason;
    return false;
  }
  return true;
}

bool meshes_identical(const RemapMesh& old_mesh,
                      const RemapMesh& new_mesh,
                      const int old_node_count,
                      const int new_node_count) {
  if (old_mesh.n_cells != new_mesh.n_cells ||
      old_node_count != new_node_count) {
    return false;
  }
  for (int cell = 0; cell < old_mesh.n_cells; ++cell) {
    const int nverts = static_cast<int>(old_mesh.nverts[cell]);
    if (nverts != static_cast<int>(new_mesh.nverts[cell])) {
      return false;
    }
    for (int corner = 0; corner < nverts; ++corner) {
      if (old_mesh.csr_indices[old_mesh.csr_offsets[cell] + corner] !=
          new_mesh.csr_indices[new_mesh.csr_offsets[cell] + corner]) {
        return false;
      }
    }
  }
  for (int node = 0; node < old_node_count; ++node) {
    if (old_mesh.node_r[node] != new_mesh.node_r[node] ||
        old_mesh.node_z[node] != new_mesh.node_z[node]) {
      return false;
    }
  }
  return true;
}

AffineDensity detect_affine_density(
    const std::vector<CellGeometry>& cells,
    const RemapFields& fields) {
  AffineDensity result;
  // Three cell averages always define a plane.  Require at least one
  // independent verification cell before selecting the affine-exact path.
  if (cells.size() < 4U) {
    return result;
  }

  std::vector<long double> centroid_r(cells.size(), 0.0L);
  std::vector<long double> centroid_z(cells.size(), 0.0L);
  std::vector<long double> density(cells.size(), 0.0L);
  bool uniform = true;
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    const long double volume = cells[cell].moments.volume;
    centroid_r[cell] = cells[cell].moments.moment_r / volume;
    centroid_z[cell] = cells[cell].moments.moment_z / volume;
    density[cell] = fields.mass[cell] / volume;
    if (cell != 0U && !relative_identity(density[cell], density[0])) {
      uniform = false;
    }
  }
  if (uniform) {
    return result;
  }

  // The exhaustive best-triple scan was O(N^3); any non-degenerate triple reconstructs the same affine field exactly, conditioning only affects rounding, and the full-mesh verification gates acceptance.
  std::size_t first_best = 0U;
  std::size_t second_best = 1U;
  long double distance_best = -1.0L;
  for (std::size_t cell = 1U; cell < cells.size(); ++cell) {
    const long double delta_r = centroid_r[cell] - centroid_r[first_best];
    const long double delta_z = centroid_z[cell] - centroid_z[first_best];
    const long double distance = delta_r * delta_r + delta_z * delta_z;
    if (distance > distance_best) {
      distance_best = distance;
      second_best = cell;
    }
  }

  std::size_t third_best = 1U;
  long double determinant_magnitude_best = -1.0L;
  for (std::size_t cell = 1U; cell < cells.size(); ++cell) {
    if (cell == second_best) {
      continue;
    }
    const long double determinant_magnitude = std::abs(
        (centroid_r[second_best] - centroid_r[first_best]) *
            (centroid_z[cell] - centroid_z[first_best]) -
        (centroid_r[cell] - centroid_r[first_best]) *
            (centroid_z[second_best] - centroid_z[first_best]));
    if (determinant_magnitude > determinant_magnitude_best) {
      determinant_magnitude_best = determinant_magnitude;
      third_best = cell;
    }
  }
  const long double determinant_best =
      (centroid_r[second_best] - centroid_r[first_best]) *
          (centroid_z[third_best] - centroid_z[first_best]) -
      (centroid_r[third_best] - centroid_r[first_best]) *
          (centroid_z[second_best] - centroid_z[first_best]);
  if (determinant_best == 0.0L) {
    return result;
  }

  const long double delta_r_second =
      centroid_r[second_best] - centroid_r[first_best];
  const long double delta_z_second =
      centroid_z[second_best] - centroid_z[first_best];
  const long double delta_r_third =
      centroid_r[third_best] - centroid_r[first_best];
  const long double delta_z_third =
      centroid_z[third_best] - centroid_z[first_best];
  const long double delta_density_second =
      density[second_best] - density[first_best];
  const long double delta_density_third =
      density[third_best] - density[first_best];
  result.radial =
      (delta_density_second * delta_z_third -
       delta_density_third * delta_z_second) /
      determinant_best;
  result.axial =
      (delta_r_second * delta_density_third -
       delta_r_third * delta_density_second) /
      determinant_best;
  result.constant =
      density[first_best] - result.radial * centroid_r[first_best] -
      result.axial * centroid_z[first_best];

  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    const long double reconstructed =
        result.constant + result.radial * centroid_r[cell] +
        result.axial * centroid_z[cell];
    if (!relative_identity(reconstructed, density[cell])) {
      return AffineDensity{};
    }
  }
  result.active = true;
  return result;
}

}  // namespace

namespace detail {

RemapResult build_dvclp_edge_budgets_for_test(
    const RemapMesh& old_mesh,
    const RemapMesh& target_mesh,
    const RemapFields& old_fields,
    const double* target_node_vr,
    const double* target_node_vz,
    std::vector<DvclpEdgeBudget>& budgets,
    int* nonstar_fallback_cells,
    int* degenerate_skipped_triangles) {
  if (!valid_fields(old_fields) || target_node_vr == nullptr ||
      target_node_vz == nullptr) {
    return reject("null_field");
  }
  std::vector<CellGeometry> old_cells;
  std::vector<CellGeometry> target_cells;
  int old_node_count = 0;
  int target_node_count = 0;
  int nonstar_count = 0;
  int degenerate_count = 0;
  std::string reason;
  if (!build_geometry(old_mesh, old_cells, old_node_count, reason) ||
      !build_geometry(target_mesh, target_cells, target_node_count,
                      reason)) {
    return reject(reason);
  }
  if (!build_dvclp_edge_budgets(
          old_mesh, target_mesh, old_cells, old_fields, target_node_vr,
          target_node_vz, budgets, reason, nonstar_count,
          degenerate_count)) {
    return reject(reason);
  }
  if (nonstar_fallback_cells != nullptr) {
    *nonstar_fallback_cells = nonstar_count;
  }
  if (degenerate_skipped_triangles != nullptr) {
    *degenerate_skipped_triangles = degenerate_count;
  }
  RemapResult result;
  result.ok = true;
  return result;
}

}  // namespace detail

RemapResult remap_conservative(
    const RemapMesh& old_mesh,
    const RemapMesh& new_mesh,
    const RemapFields& old_fields,
    double* new_mass,
    double* new_ee,
    double* new_ei,
    double* new_corner_mass,
    double* new_node_vr,
    double* new_node_vz,
    const bool target_is_exact_voronoi,
    const int dvclp_solver_rev,
    const double reale_overlay_additivity_tol) {
  const auto tp_entry = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point tp_hash0 = tp_entry;
  std::chrono::steady_clock::time_point tp_hash1 = tp_entry;
  std::chrono::steady_clock::time_point tp_enum1 = tp_entry;
  std::chrono::steady_clock::time_point tp_overlay1 = tp_entry;
  std::chrono::steady_clock::time_point tp_dvclp0 = tp_entry;
  std::chrono::steady_clock::time_point tp_dvclp1 = tp_entry;
  const char* audit_env = std::getenv("TENRYU_REMAP_AUDIT_CELL");
  int audit_cell = audit_env ? std::atoi(audit_env) : -1;
  if (!valid_fields(old_fields) ||
      !valid_outputs(new_mass, new_ee, new_ei, new_corner_mass,
                     new_node_vr, new_node_vz)) {
    return reject("null_field");
  }

  std::vector<CellGeometry> old_cells;
  std::vector<CellGeometry> target_cells;
  int old_node_count = 0;
  int new_node_count = 0;
  std::string geometry_reason;
  const auto tp_geom0 = std::chrono::steady_clock::now();
  if (!build_geometry(old_mesh, old_cells, old_node_count,
                      geometry_reason) ||
      !build_geometry(new_mesh, target_cells, new_node_count,
                      geometry_reason)) {
    return reject(geometry_reason);
  }
  const auto tp_geom1 = std::chrono::steady_clock::now();

  long double old_mass_total = 0.0L;
  long double old_momentum_r_total = 0.0L;
  long double old_momentum_z_total = 0.0L;
  long double old_electron_internal_total = 0.0L;
  long double old_ion_internal_total = 0.0L;
  long double old_kinetic_total = 0.0L;
  long double momentum_r_scale =
      static_cast<long double>(std::numeric_limits<double>::min());
  long double momentum_z_scale =
      static_cast<long double>(std::numeric_limits<double>::min());
  for (int cell = 0; cell < old_mesh.n_cells; ++cell) {
    const double cell_mass = old_fields.mass[cell];
    const double ee = old_fields.ee[cell];
    const double ei = old_fields.ei[cell];
    if (!(cell_mass > 0.0) || !std::isfinite(cell_mass) ||
        !std::isfinite(ee) || ee < 0.0 ||
        !std::isfinite(ei) || ei < 0.0) {
      return reject("input_positivity");
    }

    long double corner_mass_sum = 0.0L;
    const int nverts = static_cast<int>(old_mesh.nverts[cell]);
    const int begin = old_mesh.csr_offsets[cell];
    for (int corner = 0; corner < nverts; ++corner) {
      const std::size_t packet_index =
          static_cast<std::size_t>(cell) * old_mesh.corner_stride +
          static_cast<std::size_t>(corner);
      const double mass = old_fields.corner_mass[packet_index];
      const int node = old_mesh.csr_indices[begin + corner];
      const double vr = old_fields.node_vr[node];
      const double vz = old_fields.node_vz[node];
      if (!std::isfinite(mass) || mass < 0.0 ||
          !std::isfinite(vr) || !std::isfinite(vz)) {
        return reject("input_positivity");
      }
      const double pr = mass * vr;
      const double pz = mass * vz;
      const double kinetic = 0.5 * mass * (vr * vr + vz * vz);
      if (!std::isfinite(pr) || !std::isfinite(pz) ||
          !std::isfinite(kinetic) || kinetic < 0.0) {
        return reject("input_positivity");
      }
      corner_mass_sum += mass;
      old_mass_total += mass;
      old_momentum_r_total += pr;
      old_momentum_z_total += pz;
      old_electron_internal_total += mass * ee;
      old_ion_internal_total += mass * ei;
      old_kinetic_total += kinetic;
      momentum_r_scale += std::abs(pr);
      momentum_z_scale += std::abs(pz);
    }
    if (!relative_identity(corner_mass_sum, cell_mass)) {
      return reject("corner_mass_defect");
    }
  }

  const AffineDensity affine_density =
      detect_affine_density(old_cells, old_fields);
  const auto tp_scan1 = std::chrono::steady_clock::now();

  std::vector<OverlayEntry> overlay;
  std::vector<long double> old_piece_volume(old_cells.size(), 0.0L);
  std::vector<long double> new_piece_volume(target_cells.size(), 0.0L);
  std::vector<long double> old_coefficient_sum(old_cells.size(), 0.0L);
  // D11: bisector overlay requires the target partition to be the exact
  // Voronoi partition of its generators; welded targets use ring-polygon fans.
  const bool use_target_bisectors =
      new_mesh.generator_r != nullptr && new_mesh.generator_z != nullptr &&
      new_mesh.neighbor_index != nullptr && target_is_exact_voronoi;
  const bool identity_meshes =
      !use_target_bisectors && meshes_identical(
                                  old_mesh, new_mesh, old_node_count,
                                  new_node_count);
  if (audit_cell >= 0 && audit_cell < old_mesh.n_cells) {
    const CellGeometry& audit =
        old_cells[static_cast<std::size_t>(audit_cell)];
    std::fprintf(
        stderr,
        "[remap_audit] old_cell=%d nverts=%zu volume=%.17e "
        "bbox=%.17e,%.17e,%.17e,%.17e coords=",
        audit_cell, audit.r.size(), audit.moments.volume,
        audit.bounds.r_min, audit.bounds.r_max, audit.bounds.z_min,
        audit.bounds.z_max);
    for (std::size_t vertex = 0; vertex < audit.r.size(); ++vertex) {
      std::fprintf(stderr, "%s%.17e,%.17e", vertex == 0 ? "" : ";",
                   audit.r[vertex], audit.z[vertex]);
    }
    std::fputc('\n', stderr);

    std::size_t overlapping_new_cells = 0;
    for (std::size_t new_cell = 0; new_cell < target_cells.size();
         ++new_cell) {
      const CellGeometry& target = target_cells[new_cell];
      if (!bounds_overlap(audit.bounds, target.bounds)) {
        continue;
      }
      if (overlapping_new_cells < 40U) {
        std::fprintf(
            stderr,
            "[remap_audit] overlapping_new_cell=%zu nverts=%zu "
            "volume=%.17e coords=",
            new_cell, target.r.size(), target.moments.volume);
        for (std::size_t vertex = 0; vertex < target.r.size(); ++vertex) {
          std::fprintf(stderr, "%s%.17e,%.17e",
                       vertex == 0 ? "" : ";", target.r[vertex],
                       target.z[vertex]);
        }
        std::fputc('\n', stderr);
      }
      ++overlapping_new_cells;
    }
    if (overlapping_new_cells > 40U) {
      std::fprintf(stderr, "[remap_audit] ...and %zu more\n",
                   overlapping_new_cells - 40U);
    }
  }
  std::size_t split_pair_count = 0;
  std::size_t split_plane_count = 0;
  if (identity_meshes) {
    split_pair_count = static_cast<std::size_t>(old_mesh.n_cells);
    for (int cell = 0; cell < old_mesh.n_cells; ++cell) {
      const Moments moments =
          old_cells[static_cast<std::size_t>(cell)].moments;
      overlay.push_back({cell, cell, moments, 1.0});
      old_piece_volume[static_cast<std::size_t>(cell)] = moments.volume;
      new_piece_volume[static_cast<std::size_t>(cell)] = moments.volume;
      old_coefficient_sum[static_cast<std::size_t>(cell)] = 1.0L;
    }
  } else {
    tp_hash0 = std::chrono::steady_clock::now();
    const UniformSpatialHash spatial_hash(old_cells);
    tp_hash1 = std::chrono::steady_clock::now();
    if (use_target_bisectors) {
      std::vector<std::pair<int, int>> pair_list;
      std::vector<std::vector<int>> old_to_targets(old_cells.size());
      for (int new_cell = 0; new_cell < new_mesh.n_cells; ++new_cell) {
        const CellGeometry& target =
            target_cells[static_cast<std::size_t>(new_cell)];
        const std::vector<int> candidates =
            spatial_hash.candidates(target.bounds);
        for (const int old_cell : candidates) {
          const CellGeometry& source =
              old_cells[static_cast<std::size_t>(old_cell)];
          if (!bounds_overlap(source.bounds, target.bounds)) {
            continue;
          }
          pair_list.push_back({new_cell, old_cell});
          old_to_targets[static_cast<std::size_t>(old_cell)].push_back(
              new_cell);
        }
      }
      tp_enum1 = std::chrono::steady_clock::now();
      split_pair_count = pair_list.size();

      std::vector<ClipHalfPlane> pair_planes;
      for (const std::pair<int, int>& cell_pair : pair_list) {
        const int new_cell = cell_pair.first;
        const int old_cell = cell_pair.second;
        const CellGeometry& source =
            old_cells[static_cast<std::size_t>(old_cell)];
        pair_planes.clear();
        // All candidate generators for an old cell define its per-pair
        // bisector partition. Boundary planes are excluded because old cells
        // lie inside the domain tiled by targets; snapped sliver edges can give
        // rounding-noise plane directions that carve real area.
        for (const int u :
             old_to_targets[static_cast<std::size_t>(old_cell)]) {
          if (u == new_cell) {
            continue;
          }
          const double target_r = new_mesh.generator_r[new_cell];
          const double target_z = new_mesh.generator_z[new_cell];
          const double dr = new_mesh.generator_r[u] - target_r;
          const double dz = new_mesh.generator_z[u] - target_z;
          const double midpoint_r =
              0.5 * (target_r + new_mesh.generator_r[u]);
          const double midpoint_z =
              0.5 * (target_z + new_mesh.generator_z[u]);
          pair_planes.push_back(
              {midpoint_r, midpoint_z, midpoint_r + dz,
               midpoint_z - dr, -1.0});
        }
        split_plane_count += pair_planes.size();
        const int audit_new_cell = old_cell == audit_cell ? new_cell : -1;
        if (audit_new_cell >= 0) {
          print_audit_half_planes(
              new_mesh, new_cell,
              old_to_targets[static_cast<std::size_t>(old_cell)]);
        }
        const PairMoments pair =
            intersect_pair_local(source, pair_planes, audit_new_cell);
        // Touching candidate pairs produce +/-machine-residue signed fan sums;
        // classify non-positive pieces as no-overlap by sign alone. The
        // 64*eps-relative additivity gates remain the sole correctness authority,
        // so a genuinely missing or inverted overlap still fails loudly.
        if (!std::isfinite(pair.moments.volume) ||
            !std::isfinite(pair.moments.moment_r) ||
            !std::isfinite(pair.moments.moment_z)) {
          if (old_cell == audit_cell) {
            std::fprintf(
                stderr,
                "[remap_audit] pair new_cell=%d piece_volume=%.17e "
                "coefficient=skipped-nonfinite_overlap\n",
                new_cell, pair.moments.volume);
          }
          return reject("nonfinite_overlap");
        }
        if (pair.moments.volume <= 0.0) {
          if (old_cell == audit_cell) {
            std::fprintf(
                stderr,
                "[remap_audit] pair new_cell=%d piece_volume=%.17e "
                "classified=no_overlap\n",
                new_cell, pair.moments.volume);
          }
          continue;
        }
        const long double overlap_measure =
            affine_density.active
                ? affine_density.constant * pair.moments.volume +
                      affine_density.radial * pair.moments.moment_r +
                      affine_density.axial * pair.moments.moment_z
                : pair.volume_ld;
        const long double source_measure =
            affine_density.active
                ? static_cast<long double>(old_fields.mass[old_cell])
                : static_cast<long double>(source.moments.volume);
        const double coefficient =
            static_cast<double>(overlap_measure / source_measure);
        if (!std::isfinite(coefficient)) {
          if (old_cell == audit_cell) {
            std::fprintf(
                stderr,
                "[remap_audit] pair new_cell=%d piece_volume=%.17e "
                "coefficient=skipped-nonfinite_coefficient\n",
                new_cell, pair.moments.volume);
          }
          return reject("nonfinite_coefficient");
        }
        if (old_cell == audit_cell) {
          std::fprintf(
              stderr,
              "[remap_audit] pair new_cell=%d piece_volume=%.17e "
              "coefficient=%.17e\n",
              new_cell, pair.moments.volume, coefficient);
        }
        if (coefficient <= 0.0) {
          continue;
        }
        overlay.push_back(
            {old_cell, new_cell, pair.moments, coefficient});
        old_piece_volume[static_cast<std::size_t>(old_cell)] +=
            pair.volume_ld;
        new_piece_volume[static_cast<std::size_t>(new_cell)] +=
            pair.volume_ld;
        old_coefficient_sum[static_cast<std::size_t>(old_cell)] +=
            coefficient;
      }
    } else {
      std::vector<TargetFan> target_fans;
      std::string fan_reason;
      if (!build_target_fans(target_cells, target_fans, fan_reason)) {
        return reject(fan_reason);
      }
      for (int new_cell = 0; new_cell < new_mesh.n_cells; ++new_cell) {
        const CellGeometry& target =
            target_cells[static_cast<std::size_t>(new_cell)];
        const std::vector<int> candidates =
            spatial_hash.candidates(target.bounds);
        for (const int old_cell : candidates) {
          const CellGeometry& source =
              old_cells[static_cast<std::size_t>(old_cell)];
          if (!bounds_overlap(source.bounds, target.bounds)) {
            continue;
          }
          ++split_pair_count;
          const int audit_new_cell = old_cell == audit_cell ? new_cell : -1;
          const PairMoments pair = intersect_pair_local(
              source, target,
              target_fans[static_cast<std::size_t>(new_cell)],
              audit_new_cell);
          // Touching candidate pairs produce +/-machine-residue signed fan sums;
          // classify non-positive pieces as no-overlap by sign alone. The
          // 64*eps-relative additivity gates remain the sole correctness authority,
          // so a genuinely missing or inverted overlap still fails loudly.
          if (!std::isfinite(pair.moments.volume) ||
              !std::isfinite(pair.moments.moment_r) ||
              !std::isfinite(pair.moments.moment_z)) {
            if (old_cell == audit_cell) {
              std::fprintf(
                  stderr,
                  "[remap_audit] pair new_cell=%d piece_volume=%.17e "
                  "coefficient=skipped-nonfinite_overlap\n",
                  new_cell, pair.moments.volume);
            }
            return reject("nonfinite_overlap");
          }
          if (pair.moments.volume <= 0.0) {
            if (old_cell == audit_cell) {
              std::fprintf(
                  stderr,
                  "[remap_audit] pair new_cell=%d piece_volume=%.17e "
                  "classified=no_overlap\n",
                  new_cell, pair.moments.volume);
            }
            continue;
          }
          const long double overlap_measure =
              affine_density.active
                  ? affine_density.constant * pair.moments.volume +
                        affine_density.radial * pair.moments.moment_r +
                        affine_density.axial * pair.moments.moment_z
                  : pair.volume_ld;
          const long double source_measure =
              affine_density.active
                  ? static_cast<long double>(old_fields.mass[old_cell])
                  : static_cast<long double>(source.moments.volume);
          const double coefficient =
              static_cast<double>(overlap_measure / source_measure);
          if (!std::isfinite(coefficient)) {
            if (old_cell == audit_cell) {
              std::fprintf(
                  stderr,
                  "[remap_audit] pair new_cell=%d piece_volume=%.17e "
                  "coefficient=skipped-nonfinite_coefficient\n",
                  new_cell, pair.moments.volume);
            }
            return reject("nonfinite_coefficient");
          }
          if (old_cell == audit_cell) {
            std::fprintf(
                stderr,
                "[remap_audit] pair new_cell=%d piece_volume=%.17e "
                "coefficient=%.17e\n",
                new_cell, pair.moments.volume, coefficient);
          }
          if (coefficient <= 0.0) {
            continue;
          }
          overlay.push_back(
              {old_cell, new_cell, pair.moments, coefficient});
          old_piece_volume[static_cast<std::size_t>(old_cell)] +=
              pair.volume_ld;
          new_piece_volume[static_cast<std::size_t>(new_cell)] +=
              pair.volume_ld;
          old_coefficient_sum[static_cast<std::size_t>(old_cell)] +=
              coefficient;
        }
      }
    }
  }
  tp_overlay1 = std::chrono::steady_clock::now();
  if (audit_cell >= 0 && audit_cell < old_mesh.n_cells) {
    std::fprintf(
        stderr,
        "[remap_audit] total old_piece_volume=%.17e ref=%.17e\n",
        static_cast<double>(
            old_piece_volume[static_cast<std::size_t>(audit_cell)]),
        old_cells[static_cast<std::size_t>(audit_cell)].moments.volume);
  }

  std::size_t worst_cell = 0;
  long double worst_residual = 0.0L;
  long double worst_bound = 0.0L;
  std::size_t fail_count = 0;
  std::size_t coeff_worst_cell = 0;
  long double coeff_worst_residual = 0.0L;
  long double coeff_worst_bound = 0.0L;
  std::size_t coeff_fail_count = 0;
  for (std::size_t cell = 0; cell < old_cells.size(); ++cell) {
    const long double ref = old_cells[cell].moments.volume;
    const long double scale =
        std::max({std::abs(ref), std::abs(old_piece_volume[cell]),
                  (long double)std::numeric_limits<double>::min()});
    const long double residual =
        std::abs(old_piece_volume[cell] - ref) / scale;
    const long double bound = std::max(
        (long double)kEps64 * scale,
        snap_additivity_bound(old_cells[cell]));
    if (std::abs(old_piece_volume[cell] - ref) > bound) {
      ++fail_count;
      if (fail_count == 1 || residual > worst_residual) {
        worst_cell = cell;
        worst_residual = residual;
        worst_bound = bound;
      }
    }
    const long double coeff_scale =
        std::max({std::abs(1.0L), std::abs(old_coefficient_sum[cell]),
                  (long double)std::numeric_limits<double>::min()});
    const long double coeff_residual =
        std::abs(old_coefficient_sum[cell] - 1.0L) / coeff_scale;
    const long double coeff_bound = std::max(
        (long double)kEps64 * coeff_scale,
        snap_additivity_bound(old_cells[cell]) /
            std::max(std::abs((long double)old_cells[cell].moments.volume),
                     (long double)std::numeric_limits<double>::min()));
    if (std::abs(old_coefficient_sum[cell] - 1.0L) > coeff_bound) {
      ++coeff_fail_count;
      if (coeff_fail_count == 1 ||
          coeff_residual > coeff_worst_residual) {
        coeff_worst_cell = cell;
        coeff_worst_residual = coeff_residual;
        coeff_worst_bound = coeff_bound;
      }
    }
  }
  if (fail_count > 0) {
    const bool tolerated =
        reale_overlay_additivity_tol > 0.0 &&
        worst_residual <= reale_overlay_additivity_tol &&
        fail_count <= old_cells.size() / 1000;
    if (tolerated) {
      std::fprintf(
          stderr,
          "[reale_v2] overlay_additivity tolerated: stage=old "
          "worst_rel=%.3e fails=%d/%d\n",
          static_cast<double>(worst_residual), static_cast<int>(fail_count),
          static_cast<int>(old_cells.size()));
    } else {
      char reason[192];
      std::snprintf(
          reason, sizeof(reason),
          "old_overlay_additivity: worst_cell=%zu rel=%.3Le "
          "pieces=%.17Lg ref=%.17g fails=%zu/%zu bound=%.3Le",
          worst_cell, worst_residual, old_piece_volume[worst_cell],
          old_cells[worst_cell].moments.volume, fail_count,
          old_cells.size(), worst_bound);
      return reject(reason);
    }
  }
  if (coeff_fail_count > 0) {
    char reason[192];
    std::snprintf(
        reason, sizeof(reason),
        "coefficient_additivity: worst_cell=%zu rel=%.3Le "
        "fails=%zu/%zu bound=%.3Le",
        coeff_worst_cell, coeff_worst_residual, coeff_fail_count,
        old_cells.size(), coeff_worst_bound);
    return reject(reason);
  }

  worst_cell = 0;
  worst_residual = 0.0L;
  worst_bound = 0.0L;
  fail_count = 0;
  for (std::size_t cell = 0; cell < target_cells.size(); ++cell) {
    const long double ref = target_cells[cell].moments.volume;
    const long double scale =
        std::max({std::abs(ref), std::abs(new_piece_volume[cell]),
                  (long double)std::numeric_limits<double>::min()});
    const long double residual =
        std::abs(new_piece_volume[cell] - ref) / scale;
    const long double bound = std::max(
        (long double)kEps64 * scale,
        snap_additivity_bound(target_cells[cell]));
    if (std::abs(new_piece_volume[cell] - ref) > bound) {
      ++fail_count;
      if (fail_count == 1 || residual > worst_residual) {
        worst_cell = cell;
        worst_residual = residual;
        worst_bound = bound;
      }
    }
  }
  if (fail_count > 0) {
    const bool tolerated =
        reale_overlay_additivity_tol > 0.0 &&
        worst_residual <= reale_overlay_additivity_tol &&
        fail_count <= target_cells.size() / 1000;
    if (tolerated) {
      std::fprintf(
          stderr,
          "[reale_v2] overlay_additivity tolerated: stage=new "
          "worst_rel=%.3e fails=%d/%d\n",
          static_cast<double>(worst_residual), static_cast<int>(fail_count),
          static_cast<int>(target_cells.size()));
    } else {
      char reason[192];
      std::snprintf(
          reason, sizeof(reason),
          "new_overlay_additivity: worst_cell=%zu rel=%.3Le "
          "pieces=%.17Lg ref=%.17g fails=%zu/%zu bound=%.3Le",
          worst_cell, worst_residual, new_piece_volume[worst_cell],
          target_cells[worst_cell].moments.volume, fail_count,
          target_cells.size(), worst_bound);
      return reject(reason);
    }
  }
  const auto tp_addgate1 = std::chrono::steady_clock::now();

  std::vector<double> mass_stage(
      static_cast<std::size_t>(new_mesh.n_cells), 0.0);
  std::vector<double> ee_stage(
      static_cast<std::size_t>(new_mesh.n_cells), 0.0);
  std::vector<double> ei_stage(
      static_cast<std::size_t>(new_mesh.n_cells), 0.0);
  std::vector<double> corner_mass_stage(
      static_cast<std::size_t>(new_mesh.n_cells * new_mesh.corner_stride),
      0.0);
  std::vector<double> node_vr_stage(
      static_cast<std::size_t>(new_node_count), 0.0);
  std::vector<double> node_vz_stage(
      static_cast<std::size_t>(new_node_count), 0.0);

  // Preserve the exact fixed point after the overlay itself has passed every
  // geometric gate.  A general overlap path changes rounding association and
  // cannot reproduce arbitrary corner packets bit for bit.
  if (identity_meshes) {
    for (int cell = 0; cell < old_mesh.n_cells; ++cell) {
      mass_stage[static_cast<std::size_t>(cell)] = old_fields.mass[cell];
      ee_stage[static_cast<std::size_t>(cell)] = old_fields.ee[cell];
      ei_stage[static_cast<std::size_t>(cell)] = old_fields.ei[cell];
      const int copy_stride =
          std::min(old_mesh.corner_stride, new_mesh.corner_stride);
      for (int corner = 0; corner < copy_stride; ++corner) {
        const std::size_t old_index =
            static_cast<std::size_t>(cell) * old_mesh.corner_stride +
            static_cast<std::size_t>(corner);
        const std::size_t new_index =
            static_cast<std::size_t>(cell) * new_mesh.corner_stride +
            static_cast<std::size_t>(corner);
        corner_mass_stage[new_index] = old_fields.corner_mass[old_index];
      }
    }
    for (int node = 0; node < old_node_count; ++node) {
      node_vr_stage[static_cast<std::size_t>(node)] =
          old_fields.node_vr[node];
      node_vz_stage[static_cast<std::size_t>(node)] =
          old_fields.node_vz[node];
    }

    std::vector<double> nodal_mass(
        static_cast<std::size_t>(new_node_count), 0.0);
    std::vector<double> corner_momentum_r(corner_mass_stage.size(), 0.0);
    std::vector<double> corner_momentum_z(corner_mass_stage.size(), 0.0);
    for (int cell = 0; cell < new_mesh.n_cells; ++cell) {
      const int begin = new_mesh.csr_offsets[cell];
      const int nverts = static_cast<int>(new_mesh.nverts[cell]);
      for (int corner = 0; corner < nverts; ++corner) {
        const std::size_t index =
            static_cast<std::size_t>(cell) * new_mesh.corner_stride +
            static_cast<std::size_t>(corner);
        const int node = new_mesh.csr_indices[begin + corner];
        const double corner_mass = corner_mass_stage[index];
        nodal_mass[static_cast<std::size_t>(node)] += corner_mass;
        corner_momentum_r[index] =
            corner_mass * node_vr_stage[static_cast<std::size_t>(node)];
        corner_momentum_z[index] =
            corner_mass * node_vz_stage[static_cast<std::size_t>(node)];
      }
    }
    std::vector<double> cell_q_projection(
        static_cast<std::size_t>(new_mesh.n_cells), 0.0);
    std::vector<detail::DvclpEdgeBudget> dvclp_budgets;
    detail::DvclpProjectionResult dvclp_projection;
    std::string dvclp_reason;
    const auto tp_ident_dvclp0 = std::chrono::steady_clock::now();
    if (!run_dvclp_projection(
            old_mesh, new_mesh, old_cells, old_fields, nodal_mass,
            corner_mass_stage, corner_momentum_r, corner_momentum_z,
            node_vr_stage, node_vz_stage, cell_q_projection,
            dvclp_budgets, dvclp_projection, dvclp_reason,
            dvclp_solver_rev)) {
      return reject(dvclp_reason);
    }
    const auto tp_ident_dvclp1 = std::chrono::steady_clock::now();
    for (int cell = 0; cell < new_mesh.n_cells; ++cell) {
      ei_stage[static_cast<std::size_t>(cell)] +=
          cell_q_projection[static_cast<std::size_t>(cell)] /
          mass_stage[static_cast<std::size_t>(cell)];
    }

    std::copy(mass_stage.begin(), mass_stage.end(), new_mass);
    std::copy(ee_stage.begin(), ee_stage.end(), new_ee);
    std::copy(ei_stage.begin(), ei_stage.end(), new_ei);
    std::copy(corner_mass_stage.begin(), corner_mass_stage.end(),
              new_corner_mass);
    std::copy(node_vr_stage.begin(), node_vr_stage.end(), new_node_vr);
    std::copy(node_vz_stage.begin(), node_vz_stage.end(), new_node_vz);
    RemapResult result;
    result.ok = true;
    result.q_total = dvclp_projection.q_total;
    if (remap_timing_enabled()) {
      std::fprintf(
          stderr,
          "[remap_split] cells_old=%d cells_new=%d nodes_new=%d ident=1 "
          "bisect=0 pairs=%zu planes=0 overlay=%zu geom_ms=%.1f "
          "scan_ms=%.1f overlay_ms=%.1f addgate_ms=%.1f dvclp_ms=%.1f "
          "total_ms=%.1f\n",
          old_mesh.n_cells, new_mesh.n_cells, new_node_count,
          split_pair_count, overlay.size(),
          remap_ms_between(tp_geom0, tp_geom1),
          remap_ms_between(tp_geom1, tp_scan1),
          remap_ms_between(tp_scan1, tp_overlay1),
          remap_ms_between(tp_overlay1, tp_addgate1),
          remap_ms_between(tp_ident_dvclp0, tp_ident_dvclp1),
          remap_ms_between(tp_entry, tp_ident_dvclp1));
    }
    return result;
  }

  std::vector<std::vector<SourceContribution>> received(
      static_cast<std::size_t>(new_mesh.n_cells));
  // Consult-24/row_merge discipline: the same non-negative overlap
  // coefficient transports mass, momentum, both internal energies, and the
  // conservative kinetic energy of every source corner.
  for (const OverlayEntry& entry : overlay) {
    const int old_cell = entry.old_cell;
    const int nverts = static_cast<int>(old_mesh.nverts[old_cell]);
    const int begin = old_mesh.csr_offsets[old_cell];
    for (int corner = 0; corner < nverts; ++corner) {
      const std::size_t source_corner =
          static_cast<std::size_t>(old_cell) * old_mesh.corner_stride +
          static_cast<std::size_t>(corner);
      const int source_node = old_mesh.csr_indices[begin + corner];
      const double source_mass = old_fields.corner_mass[source_corner];
      const double transfer_mass = source_mass * entry.coefficient;
      const double vr = old_fields.node_vr[source_node];
      const double vz = old_fields.node_vz[source_node];
      SourceContribution contribution;
      contribution.source_node = source_node;
      contribution.mass = transfer_mass;
      contribution.momentum_r = transfer_mass * vr;
      contribution.momentum_z = transfer_mass * vz;
      contribution.electron_internal =
          transfer_mass * old_fields.ee[old_cell];
      contribution.ion_internal =
          transfer_mass * old_fields.ei[old_cell];
      contribution.kinetic_cons =
          0.5 * transfer_mass * (vr * vr + vz * vz);
      if (!std::isfinite(contribution.mass) || contribution.mass < 0.0 ||
          !std::isfinite(contribution.momentum_r) ||
          !std::isfinite(contribution.momentum_z) ||
          !std::isfinite(contribution.electron_internal) ||
          contribution.electron_internal < 0.0 ||
          !std::isfinite(contribution.ion_internal) ||
          contribution.ion_internal < 0.0 ||
          !std::isfinite(contribution.kinetic_cons) ||
          contribution.kinetic_cons < 0.0) {
        return reject("transport_positivity");
      }
      received[static_cast<std::size_t>(entry.new_cell)].push_back(
          contribution);
    }
  }
  const auto tp_transport1 = std::chrono::steady_clock::now();

  std::vector<CornerPacket> corner_packets(corner_mass_stage.size());
  std::vector<double> corner_momentum_r_stage(
      corner_mass_stage.size(), 0.0);
  std::vector<double> corner_momentum_z_stage(
      corner_mass_stage.size(), 0.0);
  std::vector<int> node_source_count(
      static_cast<std::size_t>(new_node_count), 0);
  std::vector<std::pair<int, double>> node_source_entries;
  std::vector<std::size_t> node_source_offsets;
  std::vector<std::pair<int, double>> node_source_runs;
  std::vector<std::size_t> node_source_run_offsets;
  std::vector<unsigned char> node_used(
      static_cast<std::size_t>(new_node_count), 0U);
  for (int cell = 0; cell < new_mesh.n_cells; ++cell) {
    const int nverts = static_cast<int>(new_mesh.nverts[cell]);
    const int begin = new_mesh.csr_offsets[cell];
    const std::size_t cell_sources =
        received[static_cast<std::size_t>(cell)].size();
    for (int corner = 0; corner < nverts; ++corner) {
      const int target_node = new_mesh.csr_indices[begin + corner];
      node_source_count[static_cast<std::size_t>(target_node)] +=
          static_cast<int>(cell_sources);
    }
  }
  node_source_offsets.assign(
      static_cast<std::size_t>(new_node_count) + 1, 0);
  for (int node = 0; node < new_node_count; ++node) {
    node_source_offsets[static_cast<std::size_t>(node) + 1] =
        node_source_offsets[static_cast<std::size_t>(node)] +
        static_cast<std::size_t>(
            node_source_count[static_cast<std::size_t>(node)]);
  }
  node_source_entries.resize(
      node_source_offsets[static_cast<std::size_t>(new_node_count)]);
  std::vector<std::size_t> node_source_cursor(
      node_source_offsets.begin(), node_source_offsets.end() - 1);
  for (int cell = 0; cell < new_mesh.n_cells; ++cell) {
    const int nverts = static_cast<int>(new_mesh.nverts[cell]);
    const int begin = new_mesh.csr_offsets[cell];
    const CellGeometry& geometry =
        target_cells[static_cast<std::size_t>(cell)];
    const double r0 = geometry.r[0];
    const double z0 = geometry.z[0];
    std::array<double, 16> local_r{};
    std::array<double, 16> local_z{};
    double center_r = 0.0;
    double center_z = 0.0;
    for (int corner = 0; corner < nverts; ++corner) {
      local_r[static_cast<std::size_t>(corner)] = geometry.r[corner] - r0;
      local_z[static_cast<std::size_t>(corner)] = geometry.z[corner] - z0;
      center_r += local_r[static_cast<std::size_t>(corner)];
      center_z += local_z[static_cast<std::size_t>(corner)];
    }
    center_r /= static_cast<double>(nverts);
    center_z /= static_cast<double>(nverts);
    std::array<double, 16> midpoint_r{};
    std::array<double, 16> midpoint_z{};
    for (int corner = 0; corner < nverts; ++corner) {
      const int next = (corner + 1 == nverts) ? 0 : corner + 1;
      midpoint_r[static_cast<std::size_t>(corner)] =
          0.5 * (local_r[static_cast<std::size_t>(corner)] +
                 local_r[static_cast<std::size_t>(next)]);
      midpoint_z[static_cast<std::size_t>(corner)] =
          0.5 * (local_z[static_cast<std::size_t>(corner)] +
                 local_z[static_cast<std::size_t>(next)]);
    }

    // Preserve polygon_corner_rz_volumes_n's mean-center midpoint-quad decomposition; only evaluate it in the local frame so the partition gate compares one pipeline against itself.
    std::vector<double> corner_volumes(
        static_cast<std::size_t>(new_mesh.corner_stride), 0.0);
    long double corner_volume_sum = 0.0L;
    for (int corner = 0; corner < nverts; ++corner) {
      const int previous = corner == 0 ? nverts - 1 : corner - 1;
      const double quad_r[4] = {
          local_r[static_cast<std::size_t>(corner)],
          midpoint_r[static_cast<std::size_t>(corner)], center_r,
          midpoint_r[static_cast<std::size_t>(previous)]};
      const double quad_z[4] = {
          local_z[static_cast<std::size_t>(corner)],
          midpoint_z[static_cast<std::size_t>(corner)], center_z,
          midpoint_z[static_cast<std::size_t>(previous)]};
      const LocalMoments m = local_planar_moments(quad_r, quad_z, 4);
      const long double vol_k =
          kTwoPiLocalL *
          (m.ir + static_cast<long double>(r0) * m.area);
      corner_volumes[corner] = static_cast<double>(vol_k);
      corner_volume_sum += vol_k;
    }
    for (int corner = 0; corner < nverts; ++corner) {
      if (!std::isfinite(corner_volumes[corner]) ||
          corner_volumes[corner] < 0.0) {
        return reject("corner_partition_positivity");
      }
    }
    if (!(corner_volume_sum > 0.0) ||
        !std::isfinite(corner_volume_sum) ||
        !relative_identity(corner_volume_sum,
                           target_cells[static_cast<std::size_t>(cell)]
                               .moments.volume)) {
      return reject("corner_partition_additivity");
    }

    // V1 subzonal attribution: split every packet by the target polygon's
    // deterministic n-generic reference corner-volume partition.
    std::vector<double> partition(
        static_cast<std::size_t>(new_mesh.corner_stride), 0.0);
    double partition_prefix = 0.0;
    for (int corner = 0; corner < nverts; ++corner) {
      partition[corner] =
          corner + 1 == nverts
              ? 1.0 - partition_prefix
              : corner_volumes[corner] / corner_volume_sum;
      if (!std::isfinite(partition[corner]) || partition[corner] < 0.0) {
        return reject("negative_coefficient");
      }
      partition_prefix += partition[corner];
    }

    for (const SourceContribution& contribution :
         received[static_cast<std::size_t>(cell)]) {
      for (int corner = 0; corner < nverts; ++corner) {
        const double coefficient = partition[corner];
        const std::size_t target_corner =
            static_cast<std::size_t>(cell) * new_mesh.corner_stride +
            static_cast<std::size_t>(corner);
        const int target_node = new_mesh.csr_indices[begin + corner];
        CornerPacket& packet = corner_packets[target_corner];
        const double mass_piece = contribution.mass * coefficient;
        packet.mass += mass_piece;
        corner_momentum_r_stage[target_corner] +=
            contribution.momentum_r * coefficient;
        corner_momentum_z_stage[target_corner] +=
            contribution.momentum_z * coefficient;
        packet.electron_internal +=
            contribution.electron_internal * coefficient;
        packet.ion_internal += contribution.ion_internal * coefficient;
        packet.kinetic_cons += contribution.kinetic_cons * coefficient;
        node_source_entries
            [node_source_cursor[static_cast<std::size_t>(target_node)]++] =
                {contribution.source_node, mass_piece};
      }
    }

    long double cell_mass = 0.0L;
    long double cell_electron_internal = 0.0L;
    long double cell_ion_internal = 0.0L;
    for (int corner = 0; corner < nverts; ++corner) {
      const std::size_t target_corner =
          static_cast<std::size_t>(cell) * new_mesh.corner_stride +
          static_cast<std::size_t>(corner);
      const CornerPacket& packet = corner_packets[target_corner];
      if (!std::isfinite(packet.mass) || packet.mass < 0.0 ||
          !std::isfinite(corner_momentum_r_stage[target_corner]) ||
          !std::isfinite(corner_momentum_z_stage[target_corner]) ||
          !std::isfinite(packet.electron_internal) ||
          packet.electron_internal < 0.0 ||
          !std::isfinite(packet.ion_internal) ||
          packet.ion_internal < 0.0 ||
          !std::isfinite(packet.kinetic_cons) ||
          packet.kinetic_cons < 0.0) {
        return reject("transport_positivity");
      }
      corner_mass_stage[target_corner] = packet.mass;
      cell_mass += packet.mass;
      cell_electron_internal += packet.electron_internal;
      cell_ion_internal += packet.ion_internal;
      const int node = new_mesh.csr_indices[begin + corner];
      node_used[static_cast<std::size_t>(node)] = 1U;
    }
    if (!(cell_mass > 0.0L) ||
        !std::isfinite(static_cast<double>(cell_mass)) ||
        cell_electron_internal < 0.0L || cell_ion_internal < 0.0L) {
      return reject("cell_positivity");
    }
    mass_stage[static_cast<std::size_t>(cell)] =
        static_cast<double>(cell_mass);
    ee_stage[static_cast<std::size_t>(cell)] =
        static_cast<double>(cell_electron_internal / cell_mass);
    ei_stage[static_cast<std::size_t>(cell)] =
        static_cast<double>(cell_ion_internal / cell_mass);
  }

  // The map summed each key in arrival order and iterated keys in ascending
  // order; stable sorting and left-to-right run sums reproduce both orders.
  node_source_run_offsets.assign(
      static_cast<std::size_t>(new_node_count) + 1, 0);
  node_source_runs.clear();
  node_source_runs.reserve(node_source_entries.size());
  for (int node = 0; node < new_node_count; ++node) {
    const std::size_t begin_entry =
        node_source_offsets[static_cast<std::size_t>(node)];
    const std::size_t end_entry =
        node_source_offsets[static_cast<std::size_t>(node) + 1];
    std::stable_sort(
        node_source_entries.begin() + begin_entry,
        node_source_entries.begin() + end_entry,
        [](const std::pair<int, double>& lhs,
           const std::pair<int, double>& rhs) {
          return lhs.first < rhs.first;
        });
    for (std::size_t entry = begin_entry; entry < end_entry;) {
      const int key = node_source_entries[entry].first;
      double sum = 0.0;
      while (entry < end_entry &&
             node_source_entries[entry].first == key) {
        sum += node_source_entries[entry].second;
        ++entry;
      }
      node_source_runs.push_back({key, sum});
    }
    node_source_run_offsets[static_cast<std::size_t>(node) + 1] =
        node_source_runs.size();
  }
  const auto tp_scatter1 = std::chrono::steady_clock::now();

  std::vector<double> nodal_mass(
      static_cast<std::size_t>(new_node_count), 0.0);
  std::vector<double> nodal_momentum_r(
      static_cast<std::size_t>(new_node_count), 0.0);
  std::vector<double> nodal_momentum_z(
      static_cast<std::size_t>(new_node_count), 0.0);
  std::vector<double> nodal_kinetic_cons(
      static_cast<std::size_t>(new_node_count), 0.0);
  std::vector<std::vector<std::size_t>> node_corners(
      static_cast<std::size_t>(new_node_count));
  for (int cell = 0; cell < new_mesh.n_cells; ++cell) {
    const int begin = new_mesh.csr_offsets[cell];
    const int nverts = static_cast<int>(new_mesh.nverts[cell]);
    for (int corner = 0; corner < nverts; ++corner) {
      const std::size_t index =
          static_cast<std::size_t>(cell) * new_mesh.corner_stride +
          static_cast<std::size_t>(corner);
      const int node = new_mesh.csr_indices[begin + corner];
      const CornerPacket& packet = corner_packets[index];
      nodal_mass[static_cast<std::size_t>(node)] += packet.mass;
      nodal_momentum_r[static_cast<std::size_t>(node)] +=
          corner_momentum_r_stage[index];
      nodal_momentum_z[static_cast<std::size_t>(node)] +=
          corner_momentum_z_stage[index];
      nodal_kinetic_cons[static_cast<std::size_t>(node)] +=
          packet.kinetic_cons;
      node_corners[static_cast<std::size_t>(node)].push_back(index);
    }
  }
  const auto tp_gather1 = std::chrono::steady_clock::now();

  // As in row_merge, gather the complete target-node star, recover p/m, and
  // evaluate Q from the explicitly non-negative weighted-variance identity.
  std::vector<double> nodal_q(
      static_cast<std::size_t>(new_node_count), 0.0);
  long double axis_projection_r = 0.0L;
  long double q_total_accum = 0.0L;
  for (int node = 0; node < new_node_count; ++node) {
    if (node_used[static_cast<std::size_t>(node)] == 0U) {
      continue;
    }
    const double mass = nodal_mass[static_cast<std::size_t>(node)];
    double momentum_r =
        nodal_momentum_r[static_cast<std::size_t>(node)];
    const double momentum_z =
        nodal_momentum_z[static_cast<std::size_t>(node)];
    const double kinetic_cons =
        nodal_kinetic_cons[static_cast<std::size_t>(node)];
    if (!(mass > 0.0) || !std::isfinite(mass) ||
        !std::isfinite(momentum_r) || !std::isfinite(momentum_z) ||
        !std::isfinite(kinetic_cons) || kinetic_cons < 0.0) {
      return reject("nodal_positivity");
    }

    const double source_mass_sum = [&]() {
      double sum = 0.0;
      const std::size_t source_begin =
          node_source_run_offsets[static_cast<std::size_t>(node)];
      const std::size_t source_end =
          node_source_run_offsets[static_cast<std::size_t>(node) + 1];
      for (std::size_t source = source_begin; source < source_end;
           ++source) {
        sum += node_source_runs[source].second;
      }
      return sum;
    }();
    if (!relative_identity(source_mass_sum, mass)) {
      return reject("nodal_mass_defect");
    }

    const bool axis_node = new_mesh.node_r[node] == 0.0;
    const double axis_momentum_r = axis_node ? momentum_r : 0.0;
    if (axis_node) {
      axis_projection_r += momentum_r;
      momentum_r = 0.0;
    }
    node_vr_stage[static_cast<std::size_t>(node)] = momentum_r / mass;
    node_vz_stage[static_cast<std::size_t>(node)] = momentum_z / mass;
    const double speed_squared =
        node_vr_stage[static_cast<std::size_t>(node)] *
            node_vr_stage[static_cast<std::size_t>(node)] +
        node_vz_stage[static_cast<std::size_t>(node)] *
            node_vz_stage[static_cast<std::size_t>(node)];
    const double kinetic_act = 0.5 * mass * speed_squared;
    const double q_difference = kinetic_cons - kinetic_act;
    const double q_scale = std::max(
        {std::abs(kinetic_cons), std::abs(kinetic_act),
         std::numeric_limits<double>::min()});
    if (!std::isfinite(q_difference) ||
        q_difference < -kEps64 * q_scale) {
      return reject("variance_negative");
    }

    long double q_variance = 0.0L;
    const std::size_t source_begin =
        node_source_run_offsets[static_cast<std::size_t>(node)];
    const std::size_t source_end =
        node_source_run_offsets[static_cast<std::size_t>(node) + 1];
    for (std::size_t first = source_begin; first < source_end; ++first) {
      for (std::size_t second = first + 1; second < source_end; ++second) {
        const double delta_vr =
            old_fields.node_vr[node_source_runs[first].first] -
            old_fields.node_vr[node_source_runs[second].first];
        const double delta_vz =
            old_fields.node_vz[node_source_runs[first].first] -
            old_fields.node_vz[node_source_runs[second].first];
        q_variance +=
            0.5L *
            static_cast<long double>(node_source_runs[first].second) *
            static_cast<long double>(node_source_runs[second].second) /
            static_cast<long double>(mass) *
            static_cast<long double>(delta_vr * delta_vr +
                                     delta_vz * delta_vz);
      }
    }
    const long double axis_q =
        0.5L * static_cast<long double>(axis_momentum_r) *
        static_cast<long double>(axis_momentum_r) /
        static_cast<long double>(mass);
    const double q = static_cast<double>(q_variance + axis_q);
    if (!std::isfinite(q) || q < 0.0 ||
        std::abs(q - q_difference) > kEps64 * q_scale) {
      return reject("energy_defect");
    }
    nodal_q[static_cast<std::size_t>(node)] = q;
    q_total_accum += q;
  }
  const auto tp_qvar1 = std::chrono::steady_clock::now();

  std::vector<double> cell_q_projection(
      static_cast<std::size_t>(new_mesh.n_cells), 0.0);
  std::vector<detail::DvclpEdgeBudget> dvclp_budgets;
  detail::DvclpProjectionResult dvclp_projection;
  std::string dvclp_reason;
  tp_dvclp0 = std::chrono::steady_clock::now();
  if (!run_dvclp_projection(
          old_mesh, new_mesh, old_cells, old_fields, nodal_mass,
          corner_mass_stage, corner_momentum_r_stage,
          corner_momentum_z_stage, node_vr_stage, node_vz_stage,
          cell_q_projection, dvclp_budgets, dvclp_projection,
          dvclp_reason, dvclp_solver_rev)) {
    return reject(dvclp_reason);
  }
  axis_projection_r += dvclp_projection.axis_projection_r;
  q_total_accum += dvclp_projection.q_total;
  tp_dvclp1 = std::chrono::steady_clock::now();

  static const char* vdiag_env = std::getenv("TENRYU_REMAP_VDIAG");
  if (vdiag_env != nullptr && std::atoi(vdiag_env) == 1) {
    struct VdiagEdge {
      double proxy_dt = 0.0;
      double length = 0.0;
      double delta_velocity_raw = 0.0;
      double delta_velocity_projected = 0.0;
      int node0 = -1;
      int node1 = -1;
    };
    const std::vector<RingEdge> ring_edges =
        enumerate_ring_edges(new_mesh);
    std::vector<VdiagEdge> candidates;
    for (std::size_t ring_edge_index = 0;
         ring_edge_index < ring_edges.size(); ++ring_edge_index) {
      const RingEdge& ring_edge = ring_edges[ring_edge_index];
      const double delta_vr =
          node_vr_stage[static_cast<std::size_t>(ring_edge.node1)] -
          node_vr_stage[static_cast<std::size_t>(ring_edge.node0)];
      const double delta_vz =
          node_vz_stage[static_cast<std::size_t>(ring_edge.node1)] -
          node_vz_stage[static_cast<std::size_t>(ring_edge.node0)];
      const double delta_velocity_projected =
          std::hypot(delta_vr, delta_vz);
      VdiagEdge edge;
      edge.length = ring_edge.length;
      edge.delta_velocity_raw =
          dvclp_budgets[ring_edge_index].raw_jump;
      edge.delta_velocity_projected = delta_velocity_projected;
      edge.node0 = ring_edge.node0;
      edge.node1 = ring_edge.node1;
      if (delta_velocity_projected > 0.0) {
        edge.proxy_dt =
            0.25 * ring_edge.length / delta_velocity_projected;
        candidates.push_back(edge);
      }
    }

    const std::size_t selected_count =
        std::min<std::size_t>(16, candidates.size());
    std::partial_sort(
        candidates.begin(), candidates.begin() + selected_count,
        candidates.end(),
        [](const VdiagEdge& left, const VdiagEdge& right) {
          return left.proxy_dt < right.proxy_dt;
        });
    for (std::size_t edge_index = 0; edge_index < selected_count;
         ++edge_index) {
      const VdiagEdge& edge = candidates[edge_index];
      const std::size_t node0 = static_cast<std::size_t>(edge.node0);
      const std::size_t node1 = static_cast<std::size_t>(edge.node1);
      const std::size_t source0_begin = node_source_run_offsets[node0];
      const std::size_t source0_end = node_source_run_offsets[node0 + 1];
      const std::size_t source1_begin = node_source_run_offsets[node1];
      const std::size_t source1_end = node_source_run_offsets[node1 + 1];
      const std::map<int, double> source0_map(
          node_source_runs.begin() + source0_begin,
          node_source_runs.begin() + source0_end);
      double shared_mass = 0.0;
      for (std::size_t source = source1_begin; source < source1_end;
           ++source) {
        const auto found = source0_map.find(node_source_runs[source].first);
        if (found != source0_map.end()) {
          shared_mass +=
              std::min(found->second, node_source_runs[source].second);
        }
      }
      const double shared_fraction =
          shared_mass / std::max(nodal_mass[node0], nodal_mass[node1]);
      std::fprintf(
          stderr,
          "[remap_vdiag] L=%.3e du_raw=%.3e du_proj=%.3e n0=%d "
          "n1=%d m0=%.3e m1=%.3e src0=%zu src1=%zu "
          "shared_frac=%.3f vr0=%.3e vz0=%.3e vr1=%.3e vz1=%.3e\n",
          edge.length, edge.delta_velocity_raw,
          edge.delta_velocity_projected, edge.node0, edge.node1,
          nodal_mass[node0], nodal_mass[node1],
          source0_end - source0_begin, source1_end - source1_begin,
          shared_fraction, node_vr_stage[node0],
          node_vz_stage[node0], node_vr_stage[node1],
          node_vz_stage[node1]);
    }
    const double min_proxy_dt =
        selected_count > 0 ? candidates[0].proxy_dt
                           : std::numeric_limits<double>::infinity();
    std::fprintf(stderr,
                 "[remap_vdiag] edges_scanned=%zu selected=%zu "
                 "min_proxy_dt=%.3e\n",
                 ring_edges.size(), selected_count, min_proxy_dt);
  }

  const auto tp_qscat0 = std::chrono::steady_clock::now();
  std::vector<double> corner_q(corner_packets.size(), 0.0);
  for (int node = 0; node < new_node_count; ++node) {
    if (node_used[static_cast<std::size_t>(node)] == 0U) {
      continue;
    }
    const auto& incident = node_corners[static_cast<std::size_t>(node)];
    double scattered = 0.0;
    for (std::size_t position = 0; position < incident.size(); ++position) {
      const std::size_t corner = incident[position];
      const double deposit =
          position + 1 == incident.size()
              ? nodal_q[static_cast<std::size_t>(node)] - scattered
              : nodal_q[static_cast<std::size_t>(node)] *
                    corner_packets[corner].mass /
                    nodal_mass[static_cast<std::size_t>(node)];
      if (!std::isfinite(deposit) || deposit < 0.0) {
        return reject("variance_negative");
      }
      corner_q[corner] = deposit;
      scattered += deposit;
    }
    if (!relative_identity(scattered,
                           nodal_q[static_cast<std::size_t>(node)])) {
      return reject("energy_defect");
    }
  }
  const auto tp_qscat1 = std::chrono::steady_clock::now();

  for (int cell = 0; cell < new_mesh.n_cells; ++cell) {
    long double electron_internal = 0.0L;
    long double ion_internal = 0.0L;
    long double cell_mass = 0.0L;
    const int nverts = static_cast<int>(new_mesh.nverts[cell]);
    for (int corner = 0; corner < nverts; ++corner) {
      const std::size_t index =
          static_cast<std::size_t>(cell) * new_mesh.corner_stride +
          static_cast<std::size_t>(corner);
      cell_mass += corner_packets[index].mass;
      electron_internal += corner_packets[index].electron_internal;
      ion_internal += corner_packets[index].ion_internal + corner_q[index];
    }
    ion_internal +=
        cell_q_projection[static_cast<std::size_t>(cell)];
    if (!(cell_mass > 0.0L) || electron_internal < 0.0L ||
        ion_internal < 0.0L) {
      return reject("cell_positivity");
    }
    mass_stage[static_cast<std::size_t>(cell)] =
        static_cast<double>(cell_mass);
    ee_stage[static_cast<std::size_t>(cell)] =
        static_cast<double>(electron_internal / cell_mass);
    ei_stage[static_cast<std::size_t>(cell)] =
        static_cast<double>(ion_internal / cell_mass);
    if (!std::isfinite(mass_stage[static_cast<std::size_t>(cell)]) ||
        !std::isfinite(ee_stage[static_cast<std::size_t>(cell)]) ||
        ee_stage[static_cast<std::size_t>(cell)] < 0.0 ||
        !std::isfinite(ei_stage[static_cast<std::size_t>(cell)]) ||
        ei_stage[static_cast<std::size_t>(cell)] < 0.0) {
      return reject("cell_positivity");
    }
  }
  const auto tp_asm1 = std::chrono::steady_clock::now();

  long double new_mass_total = 0.0L;
  long double new_momentum_r_total = 0.0L;
  long double new_momentum_z_total = 0.0L;
  long double new_electron_internal_total = 0.0L;
  long double new_ion_internal_total = 0.0L;
  long double new_kinetic_total = 0.0L;
  for (int cell = 0; cell < new_mesh.n_cells; ++cell) {
    const int begin = new_mesh.csr_offsets[cell];
    const int nverts = static_cast<int>(new_mesh.nverts[cell]);
    for (int corner = 0; corner < nverts; ++corner) {
      const std::size_t index =
          static_cast<std::size_t>(cell) * new_mesh.corner_stride +
          static_cast<std::size_t>(corner);
      const int node = new_mesh.csr_indices[begin + corner];
      const double mass = corner_mass_stage[index];
      const double vr = node_vr_stage[static_cast<std::size_t>(node)];
      const double vz = node_vz_stage[static_cast<std::size_t>(node)];
      const double pr = mass * vr;
      const double pz = mass * vz;
      new_mass_total += mass;
      new_momentum_r_total += pr;
      new_momentum_z_total += pz;
      new_electron_internal_total +=
          corner_packets[index].electron_internal;
      new_ion_internal_total +=
          corner_packets[index].ion_internal + corner_q[index];
      new_kinetic_total += 0.5L * mass * (vr * vr + vz * vz);
      momentum_r_scale += std::abs(pr);
      momentum_z_scale += std::abs(pz);
    }
    new_ion_internal_total +=
        cell_q_projection[static_cast<std::size_t>(cell)];
  }

  const long double mass_residual = new_mass_total - old_mass_total;
  const long double momentum_r_residual =
      new_momentum_r_total + axis_projection_r - old_momentum_r_total;
  const long double momentum_z_residual =
      new_momentum_z_total - old_momentum_z_total;
  const long double old_energy_total =
      old_electron_internal_total + old_ion_internal_total +
      old_kinetic_total;
  const long double new_energy_total =
      new_electron_internal_total + new_ion_internal_total +
      new_kinetic_total;
  const long double energy_residual =
      new_energy_total - old_energy_total;
  const long double mass_scale = std::max(
      {std::abs(old_mass_total), std::abs(new_mass_total),
       static_cast<long double>(std::numeric_limits<double>::min())});
  const long double energy_scale = std::max(
      {std::abs(old_energy_total), std::abs(new_energy_total),
       static_cast<long double>(std::numeric_limits<double>::min())});
  if (std::abs(mass_residual) > kEps64 * mass_scale) {
    return reject("mass_defect");
  }
  if (std::abs(momentum_r_residual) > kEps64 * momentum_r_scale ||
      std::abs(momentum_z_residual) > kEps64 * momentum_z_scale) {
    return reject("momentum_defect");
  }
  if (std::abs(energy_residual) > kEps64 * energy_scale) {
    return reject("energy_defect");
  }
  const auto tp_fingate1 = std::chrono::steady_clock::now();

  std::copy(mass_stage.begin(), mass_stage.end(), new_mass);
  std::copy(ee_stage.begin(), ee_stage.end(), new_ee);
  std::copy(ei_stage.begin(), ei_stage.end(), new_ei);
  std::copy(corner_mass_stage.begin(), corner_mass_stage.end(),
            new_corner_mass);
  std::copy(node_vr_stage.begin(), node_vr_stage.end(), new_node_vr);
  std::copy(node_vz_stage.begin(), node_vz_stage.end(), new_node_vz);
  const auto tp_copy1 = std::chrono::steady_clock::now();

  if (remap_timing_enabled()) {
    std::fprintf(
        stderr,
        "[remap_split] cells_old=%d cells_new=%d nodes_new=%d ident=0 "
        "bisect=%d pairs=%zu planes=%zu overlay=%zu geom_ms=%.1f "
        "scan_ms=%.1f hash_ms=%.1f enum_ms=%.1f clip_ms=%.1f "
        "overlay_ms=%.1f addgate_ms=%.1f transport_ms=%.1f "
        "scatter_ms=%.1f gather_ms=%.1f qvar_ms=%.1f dvclp_ms=%.1f "
        "qscat_ms=%.1f asm_ms=%.1f fingate_ms=%.1f copy_ms=%.1f "
        "total_ms=%.1f\n",
        old_mesh.n_cells, new_mesh.n_cells, new_node_count,
        use_target_bisectors ? 1 : 0, split_pair_count, split_plane_count,
        overlay.size(), remap_ms_between(tp_geom0, tp_geom1),
        remap_ms_between(tp_geom1, tp_scan1),
        remap_ms_between(tp_hash0, tp_hash1),
        use_target_bisectors ? remap_ms_between(tp_hash1, tp_enum1) : 0.0,
        use_target_bisectors ? remap_ms_between(tp_enum1, tp_overlay1)
                             : remap_ms_between(tp_hash1, tp_overlay1),
        remap_ms_between(tp_scan1, tp_overlay1),
        remap_ms_between(tp_overlay1, tp_addgate1),
        remap_ms_between(tp_addgate1, tp_transport1),
        remap_ms_between(tp_transport1, tp_scatter1),
        remap_ms_between(tp_scatter1, tp_gather1),
        remap_ms_between(tp_gather1, tp_qvar1),
        remap_ms_between(tp_dvclp0, tp_dvclp1),
        remap_ms_between(tp_qscat0, tp_qscat1),
        remap_ms_between(tp_qscat1, tp_asm1),
        remap_ms_between(tp_asm1, tp_fingate1),
        remap_ms_between(tp_fingate1, tp_copy1),
        remap_ms_between(tp_entry, tp_copy1));
  }

  RemapResult result;
  result.ok = true;
  result.mass_residual = static_cast<double>(mass_residual);
  result.momentum_residual_r =
      static_cast<double>(momentum_r_residual);
  result.momentum_residual_z =
      static_cast<double>(momentum_z_residual);
  result.energy_residual = static_cast<double>(energy_residual);
  result.q_total = static_cast<double>(q_total_accum);
  return result;
}

}  // namespace tenryu::hydro::reale
