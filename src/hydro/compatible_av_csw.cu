#include "hydro/compatible_av_csw.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "hydro/central_pseudo_core.cuh"
#include "hydro/hydro_2d.hpp"
#include "hydro/pole_angular_derefine.cuh"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::compatible {
namespace {

constexpr double kPi = 3.1415926535897932384626433832795028841971693993751;
constexpr double kTiny = 1.0e-300;
constexpr double kRatioTiny = 1.0e-30;
constexpr double kPolarSlavingCompressionEpsilon = 1.0e-12;
constexpr int kPolarSlavingMinActive = 2;

#ifdef TENRYU_CSW98_DIAG
// Free-stream defect forensics (2026-08-17): print every csw98 side force
// whose magnitude exceeds this floor. Diagnostic builds only.
__device__ constexpr double kCsw98DiagForceFloor = 1.0e2;
#endif

struct CswPolarSlavingDeviceView {
  const int* instance_lookup = nullptr;
  const int* instance_cohort = nullptr;
  const double* instance_weight = nullptr;
  const double* psi_raw = nullptr;
  const double* cohort_gate = nullptr;
  const double* cohort_psi_star = nullptr;
  double strength = 1.0;
};

struct CswPolarSlavingPreparation {
  CswPolarSlavingDeviceView view;
  double stiffness_dt = std::numeric_limits<double>::infinity();
  double stiffness_lambda = 0.0;
  double stiffness_sigma = 0.0;
  int stiffness_winner_node = -1;
};

struct CswKernelParams {
  double c1 = 1.0;
  double c2 = 1.0;
  double gamma = 5.0 / 3.0;
  double degenerate_side_floor_rel = 1.0e-2;
  double damper_impulse_beta = 0.0;
  double dt = 0.0;
  int limiter_enabled = 1;
  int axis_mirror_limiter = 0;
  int edge_diag_enabled = 0;
  int edge_diag_nodes[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  long long edge_diag_call = 0;
  int pole_floor_enabled = 0;
  double pole_floor_sigma0 = 1.0;
  double pole_floor_theta0 = 0.033;
  double pole_floor_thetaf = 0.033;
  int pole_desens_enabled = 0;
  double pole_desens_alpha = 0.25;
  double pole_desens_theta0 = 0.033;
  double pole_desens_thetaf = 0.033;
  // Multiblock axis-line AV exclusion (Wave D1, csw98 path only).
  // axisline_av_enabled mirrors the TENRYU_AW_AXISLINE_AV=1 kill switch:
  // non-zero RESTORES the unprojected pre-D1 behaviour; axis_eps_cm tests r=0.
  // TENRYU_AW_AXISLINE_AV=1 takes precedence over D1'.
  int axisline_av_enabled = 0;
  int axisline_d1prime = 0;
  int axisline_d1prime_cfl = 1;
  int limiter_shock_floor = 0;
  double shock_limiter_floor = 0.65;
  int axistouch_av_off = 0;
  double axis_eps_cm = 0.0;
  double av_cfl_diag_below = 0.0;
};

struct Csw98EdgeDiagConfig {
  int enabled = 0;
  int nodes[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
};

static long long g_csw98_edge_diag_calls = 0;

inline void configure_csw98_edge_diag(CswKernelParams* params) {
  static const Csw98EdgeDiagConfig config = [] {
    Csw98EdgeDiagConfig parsed;
    const char* cursor = std::getenv("TENRYU_CSW98_EDGE_DIAG");
    if (cursor == nullptr || cursor[0] == '\0') {
      return parsed;
    }
    parsed.enabled = 1;
    for (int i = 0; i < 8 && cursor[0] != '\0'; ++i) {
      char* end = nullptr;
      const long node = std::strtol(cursor, &end, 10);
      if (end == cursor) {
        break;
      }
      parsed.nodes[i] = static_cast<int>(node);
      if (end[0] != ',') {
        break;
      }
      cursor = end + 1;
    }
    return parsed;
  }();
  if (config.enabled == 0) {
    return;
  }
  params->edge_diag_enabled = 1;
  for (int i = 0; i < 8; ++i) {
    params->edge_diag_nodes[i] = config.nodes[i];
  }
  params->edge_diag_call = ++g_csw98_edge_diag_calls;
}

struct CswEdgeAvCflWinner {
  int edge_id = -1;
  int n0 = -1;
  int n1 = -1;
  int cell_a = -1;
  int local = -1;
  int block_id = -1;
  double r0 = 0.0;
  double z0 = 0.0;
  double r1 = 0.0;
  double z1 = 0.0;
  double dx = 0.0;
  double du = 0.0;
  double raw_dt = std::numeric_limits<double>::infinity();
};

struct EdgeAccelDisplacementWinner {
  int edge_id = -1;
  int cell = -1;
  int node0 = -1;
  int node1 = -1;
  double length = 0.0;
  double c_e = 0.0;
  double a_e = 0.0;
  double dt = std::numeric_limits<double>::infinity();
};

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
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
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(
                        val + __longlong_as_double(static_cast<long long>(assumed)))));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
#endif
}

__device__ inline double atomic_min_double(double* address, const double val) {
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  while (val < __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(val)));
    if (assumed == old) {
      break;
    }
  }
  return __longlong_as_double(static_cast<long long>(old));
}

__device__ inline bool finite_device(const double x) {
  return isfinite(x);
}

__device__ inline double csw98_polar_slaving_effective_psi(
    const CswPolarSlavingDeviceView slaving,
    const int instance,
    double* out_a_raw = nullptr,
    double* out_a_eff = nullptr) {
  const double psi_raw = slaving.psi_raw[instance];
  const int cohort = slaving.instance_cohort[instance];
  const double gate = slaving.cohort_gate[cohort];
  const double a_raw = 1.0 - psi_raw;
  const double a_star = 1.0 - slaving.cohort_psi_star[cohort];
  const double a_target = fmax(a_raw, a_star);
  const double lambda =
      slaving.strength * slaving.instance_weight[instance] * gate;
  if (lambda == 0.0) {
    if (out_a_raw != nullptr) {
      *out_a_raw = a_raw;
    }
    if (out_a_eff != nullptr) {
      *out_a_eff = a_raw;
    }
    return psi_raw;
  }
  double a_eff = a_raw;
  if (lambda == 1.0) {
    a_eff = a_target;
  } else {
    a_eff = a_raw + lambda * (a_target - a_raw);
  }
#ifndef NDEBUG
  if (!(a_eff >= a_raw)) {
    __trap();
  }
#endif
  if (out_a_raw != nullptr) {
    *out_a_raw = a_raw;
  }
  if (out_a_eff != nullptr) {
    *out_a_eff = a_eff;
  }
  return 1.0 - a_eff;
}

__device__ inline bool edge_dt_matches(const double edge_dt,
                                       const double target_dt) {
  return edge_dt == target_dt;
}

__device__ inline bool cell_active(const std::int8_t* hydro_active,
                                   const int c) {
  return c >= 0 && (hydro_active == nullptr || hydro_active[c] != 0);
}

__device__ inline int structured_node_index(const int i,
                                            const int j,
                                            const int nz) {
  return i * (nz + 1) + j;
}

__device__ inline bool structured_aw_axis_line_edge(
    const int e,
    const int n_radial,
    const int nz,
    const int aw_axis_slave_first_i,
    const bool aw_planar,
    const bool aw_axisline_av_enabled,
    const bool aw_axis_slave_theta0_active,
    const bool aw_axis_slave_theta_pi_active) {
  if (aw_axisline_av_enabled || !aw_planar ||
      (!aw_axis_slave_theta0_active &&
       !aw_axis_slave_theta_pi_active) ||
      e < 0 || e >= n_radial) {
    return false;
  }
  const int i = e / (nz + 1);
  if (i < aw_axis_slave_first_i) {
    return false;
  }
  const int j = e % (nz + 1);
  return (aw_axis_slave_theta0_active && j == 0) ||
         (aw_axis_slave_theta_pi_active && j == nz);
}

// Multiblock/CSR mirror of structured_aw_axis_line_edge, for the csw98 path.
// Multiblock cells carry no (i,j) indexing, so the axis-line test is geometric:
// after the axis snap every slaved axis node sits at r == 0 exactly, and an
// axis-line edge is one with BOTH endpoints on the axis. Same gating as the
// structured guard: AW (planar) mode only, disabled by TENRYU_AW_AXISLINE_AV=1.
// The legacy csw_edge multiblock family is deliberately NOT covered: it is a
// retired path kept bit-frozen, and its CFL kernel computes the edge
// contribution inline without access to these parameters.
__device__ inline bool multiblock_aw_axis_line_edge(
    const double* __restrict__ x_r,
    const int n0,
    const int n1,
    const bool aw_planar,
    const CswKernelParams params) {
  if (params.axisline_av_enabled != 0 || !aw_planar) {
    return false;
  }
  return x_r[n0] <= params.axis_eps_cm && x_r[n1] <= params.axis_eps_cm;
}

// A80 diagnostic: axis-touching edges keep the legacy mixed-metric pairing
// under the RZ lift guard (A79 obstruction); this switch removes their AV force
// AND work entirely to discharge whether they source the A68b pole entropy
// anomaly.
__device__ inline bool multiblock_axis_touch_edge(
    const double* __restrict__ x_r,
    const int n0,
    const int n1,
    const bool aw_planar,
    const CswKernelParams params) {
  return params.axistouch_av_off != 0 && aw_planar &&
         (x_r[n0] <= params.axis_eps_cm || x_r[n1] <= params.axis_eps_cm);
}

__device__ inline void local_face_corners(const int local,
                                          int* corner0,
                                          int* corner1);

__device__ inline void outward_face_corners(const int local,
                                            int* corner0,
                                            int* corner1);

__device__ inline bool local_face_corners(const int active_nverts,
                                          const int local,
                                          int* corner0,
                                          int* corner1) {
  if (active_nverts >= 5) {
    return mesh::mesh_topo_active_local_face_corners(active_nverts, local,
                                                     corner0, corner1);
  }
  if (active_nverts != 3) {
    local_face_corners(local, corner0, corner1);
    return true;
  }
  return mesh::mesh_topo_active_local_face_corners(active_nverts, local,
                                                   corner0, corner1);
}

__device__ inline void local_face_corners(const int local,
                                          int* corner0,
                                          int* corner1) {
  if (local == 0) {
    *corner0 = 0;
    *corner1 = 3;
  } else if (local == 1) {
    *corner0 = 1;
    *corner1 = 2;
  } else if (local == 2) {
    *corner0 = 0;
    *corner1 = 1;
  } else {
    *corner0 = 3;
    *corner1 = 2;
  }
}

__device__ inline bool outward_face_corners(const int active_nverts,
                                            const int local,
                                            int* corner0,
                                            int* corner1) {
  if (active_nverts >= 5) {
    return mesh::mesh_topo_active_local_face_corners(active_nverts, local,
                                                     corner0, corner1);
  }
  if (active_nverts != 3) {
    outward_face_corners(local, corner0, corner1);
    return true;
  }
  return mesh::mesh_topo_active_local_face_corners(active_nverts, local,
                                                   corner0, corner1);
}

__device__ inline void outward_face_corners(const int local,
                                            int* corner0,
                                            int* corner1) {
  if (local == 0) {
    *corner0 = 3;
    *corner1 = 0;
  } else if (local == 1) {
    *corner0 = 1;
    *corner1 = 2;
  } else if (local == 2) {
    *corner0 = 0;
    *corner1 = 1;
  } else {
    *corner0 = 2;
    *corner1 = 3;
  }
}

__device__ inline int opposite_face(const int local) {
  return (local == 0) ? 1 : (local == 1) ? 0 : (local == 2) ? 3 : 2;
}

__device__ inline int opposite_face(const int active_nverts, const int local) {
  // Geometric continuation selection for polygons is a deferred refinement.
  return (active_nverts == 4) ? opposite_face(local) : -1;
}

__device__ inline void structured_cell_nodes(const int c,
                                             const int nz,
                                             int nodes[4]) {
  const int i = c / nz;
  const int j = c - i * nz;
  nodes[0] = structured_node_index(i, j, nz);
  nodes[1] = structured_node_index(i + 1, j, nz);
  nodes[2] = structured_node_index(i + 1, j + 1, nz);
  nodes[3] = structured_node_index(i, j + 1, nz);
}

__device__ inline void csr_cell_nodes(const int c,
                                      const int* __restrict__ offsets,
                                      const int* __restrict__ indices,
                                      int nodes[4]) {
  const int off = offsets[c];
  nodes[0] = indices[off + 0];
  nodes[1] = indices[off + 1];
  nodes[2] = indices[off + 2];
  nodes[3] = indices[off + 3];
}

__device__ inline void csr_cell_active_nodes(
    const int c,
    const int active_nverts,
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral]) {
  if (active_nverts == 3) {
    const int off = offsets[c];
    nodes[0] = indices[off + 0];
    nodes[1] = indices[off + 1];
    nodes[2] = indices[off + 2];
    return;
  }
  if (active_nverts == 4) {
    csr_cell_nodes(c, offsets, indices, nodes);
    return;
  }
  const int off = offsets[c];
  for (int k = 0; k < active_nverts; ++k) {
    nodes[k] = indices[off + k];
  }
}

__device__ inline int structured_face_neighbor(const int c,
                                               const int local,
                                               const int nr,
                                               const int nz) {
  const int i = c / nz;
  const int j = c - i * nz;
  if (local == 0) {
    return (i > 0) ? ((i - 1) * nz + j) : -1;
  }
  if (local == 1) {
    return (i + 1 < nr) ? ((i + 1) * nz + j) : -1;
  }
  if (local == 2) {
    return (j > 0) ? (i * nz + (j - 1)) : -1;
  }
  return (j + 1 < nz) ? (i * nz + (j + 1)) : -1;
}

__device__ inline int reciprocal_face_lookup(
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int cell,
    const int neighbor) {
  if (cell < 0 || neighbor < 0) {
    return -1;
  }
  const int off = face_adj_offsets[neighbor];
  const int end = face_adj_offsets[neighbor + 1];
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, neighbor);
  for (int p = off; p < end; ++p) {
    if (!mesh::mesh_topo_local_face_is_active(active_nverts, p - off)) {
      continue;
    }
    if (face_adj_indices[p] == cell) {
      return p - off;
    }
  }
  return -1;
}

__device__ inline double harmonic_positive(const double a, const double b) {
  const bool good_a = a > 0.0 && finite_device(a);
  const bool good_b = b > 0.0 && finite_device(b);
  if (good_a && good_b && (a + b) > 0.0) {
    return 2.0 * a * b / (a + b);
  }
  if (good_a) {
    return a;
  }
  if (good_b) {
    return b;
  }
  return 0.0;
}

__device__ inline bool csw98_axisline_d1prime_cfl_dt(
    double* __restrict__ dt_cand,
    const double* __restrict__ node_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const int n0,
    const int n1,
    const int c,
    const int neighbor,
    const double w,
    const double s_z,
    const double limiter_scale,
    const CswKernelParams params) {
  if (node_mass == nullptr) {
    return false;
  }
  const double m0 = node_mass[n0];
  const double m1 = node_mass[n1];
  if (!(m0 > 0.0) || !(m1 > 0.0)) {
    return false;
  }
  const double rho_e =
      harmonic_positive(rho[c], (neighbor >= 0) ? rho[neighbor] : rho[c]);
  const double cs_e =
      harmonic_positive(cs[c], (neighbor >= 0) ? cs[neighbor] : cs[c]);
  if (!(rho_e > 0.0) || !(cs_e >= 0.0)) {
    return false;
  }
  const double aw = fabs(w);
  const double s_along = fabs(s_z);
  const double c2g = params.c2 * (params.gamma + 1.0) * 0.25;
  const double root = sqrt(c2g * c2g * aw * aw +
                           params.c1 * params.c1 * cs_e * cs_e);
  const double wave = c2g * aw + root;
  double wave_jacobian = wave;
  if (aw > 0.0) {
    if (!(root > 0.0)) {
      return false;
    }
    const double dwave_dw = c2g + (c2g * c2g * aw) / root;
    wave_jacobian += aw * dwave_dw;
  }
  const double Z = rho_e * limiter_scale * s_along * wave_jacobian;
  const double lambda = Z * (1.0 / m0 + 1.0 / m1);
  if (!(lambda > 0.0) || !finite_device(lambda)) {
    return false;
  }
  // R_time = 2 from Eq. (30), the leapfrog/velocity-update stability radius.
  *dt_cand = 2.0 / lambda;
  return true;
}

__device__ inline void edge_svec_from_cell(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int nodes[4],
    const int local,
    double* s_r,
    double* s_z,
    const bool aw_planar = false) {
  int c0 = 0;
  int c1 = 0;
  outward_face_corners(local, &c0, &c1);
  const int n0 = nodes[c0];
  const int n1 = nodes[c1];
  const double r0 = x_r[n0];
  const double z0 = x_z[n0];
  const double r1 = x_r[n1];
  const double z1 = x_z[n1];
  const double coeff = aw_planar ? 1.0 : kPi * (r0 + r1);
  *s_r = coeff * (z1 - z0);
  *s_z = -coeff * (r1 - r0);
}

__device__ inline bool edge_svec_from_cell(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int nodes[4],
    const int active_nverts,
    const int local,
    double* s_r,
    double* s_z,
    const bool aw_planar = false) {
  int c0 = 0;
  int c1 = 0;
  if (!outward_face_corners(active_nverts, local, &c0, &c1)) {
    return false;
  }
  const int n0 = nodes[c0];
  const int n1 = nodes[c1];
  const double r0 = x_r[n0];
  const double z0 = x_z[n0];
  const double r1 = x_r[n1];
  const double z1 = x_z[n1];
  const double coeff = aw_planar ? 1.0 : kPi * (r0 + r1);
  *s_r = coeff * (z1 - z0);
  *s_z = -coeff * (r1 - r0);
  return true;
}

__device__ inline double projected_edge_ratio(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int nodes[4],
    const int local,
    const double xhat_r,
    const double xhat_z,
    const double uhat_r,
    const double uhat_z,
    const double current_grad) {
  if (!(current_grad > kTiny) || !finite_device(current_grad)) {
    return 1.0;
  }
  int c0 = 0;
  int c1 = 0;
  local_face_corners(local, &c0, &c1);
  const int n0 = nodes[c0];
  const int n1 = nodes[c1];
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const double proj_dx = dx_r * xhat_r + dx_z * xhat_z;
  const double denom = proj_dx * current_grad;
  const double scale = fmax(fabs(proj_dx * current_grad), 1.0);
  if (!(fabs(denom) > kRatioTiny * scale) || !finite_device(denom)) {
    return 1.0;
  }
  const double ratio = (du_r * uhat_r + du_z * uhat_z) / denom;
  return finite_device(ratio) ? ratio : 1.0;
}

__device__ inline double projected_edge_ratio(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int nodes[4],
    const int active_nverts,
    const int local,
    const double xhat_r,
    const double xhat_z,
    const double uhat_r,
    const double uhat_z,
    const double current_grad) {
  if (!(current_grad > kTiny) || !finite_device(current_grad)) {
    return 1.0;
  }
  int c0 = 0;
  int c1 = 0;
  if (!local_face_corners(active_nverts, local, &c0, &c1)) {
    return 1.0;
  }
  const int n0 = nodes[c0];
  const int n1 = nodes[c1];
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const double proj_dx = dx_r * xhat_r + dx_z * xhat_z;
  const double denom = proj_dx * current_grad;
  const double scale = fmax(fabs(proj_dx * current_grad), 1.0);
  if (!(fabs(denom) > kRatioTiny * scale) || !finite_device(denom)) {
    return 1.0;
  }
  const double ratio = (du_r * uhat_r + du_z * uhat_z) / denom;
  return finite_device(ratio) ? ratio : 1.0;
}

__device__ inline double limiter_from_ratios(const double r0,
                                             const double r1) {
  const double limited = fmin(fmin(1.0, 2.0 * r0),
                             fmin(2.0 * r1, 0.5 * (r0 + r1)));
  return fmax(0.0, limited);
}

__device__ inline double structured_limiter(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int c,
    const int local,
    const int nr,
    const int nz,
    const double xhat_r,
    const double xhat_z,
    const double uhat_r,
    const double uhat_z,
    const double current_grad) {
  int nodes[4] = {0, 0, 0, 0};
  structured_cell_nodes(c, nz, nodes);
  const double r_inside =
      projected_edge_ratio(x_r, x_z, v_r, v_z, nodes, opposite_face(local),
                           xhat_r, xhat_z, uhat_r, uhat_z, current_grad);

  double r_outside = 1.0;
  const int n = structured_face_neighbor(c, local, nr, nz);
  if (cell_active(hydro_active, n)) {
    int neighbor_nodes[4] = {0, 0, 0, 0};
    structured_cell_nodes(n, nz, neighbor_nodes);
    r_outside = projected_edge_ratio(
        x_r, x_z, v_r, v_z, neighbor_nodes, opposite_face(local), xhat_r,
        xhat_z, uhat_r, uhat_z, current_grad);
  }
  return limiter_from_ratios(r_inside, r_outside);
}

__device__ inline double multiblock_limiter(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int c,
    const int local,
    const double xhat_r,
    const double xhat_z,
    const double uhat_r,
    const double uhat_z,
    const double current_grad) {
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {0};
  csr_cell_active_nodes(c, active_nverts, cell_node_offsets, cell_node_indices,
                        nodes);
  const int inside_local = opposite_face(active_nverts, local);
  const double r_inside =
      projected_edge_ratio(x_r, x_z, v_r, v_z, nodes, active_nverts,
                           inside_local,
                           xhat_r, xhat_z, uhat_r, uhat_z, current_grad);

  double r_outside = 1.0;
  const int neighbor = face_adj_indices[face_adj_offsets[c] + local];
  const int reciprocal =
      reciprocal_face_lookup(face_adj_offsets, face_adj_indices, cell_nverts, c,
                             neighbor);
  if (cell_active(hydro_active, neighbor) && reciprocal >= 0) {
    const int neighbor_active_nverts =
        mesh::mesh_topo_cell_active_nverts(cell_nverts, neighbor);
    int neighbor_nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {0};
    csr_cell_active_nodes(neighbor, neighbor_active_nverts, cell_node_offsets,
                          cell_node_indices, neighbor_nodes);
    r_outside = projected_edge_ratio(
        x_r, x_z, v_r, v_z, neighbor_nodes, neighbor_active_nverts,
        reciprocal, xhat_r, xhat_z, uhat_r, uhat_z, current_grad);
  }
  return limiter_from_ratios(r_inside, r_outside);
}

template <bool AtomicWork>
__device__ inline void apply_csw_contribution(
    double* __restrict__ work_slot,
    int* __restrict__ compressive_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const int n0,
    const int n1,
    const int c,
    const int neighbor,
    const int nodes[4],
    const int active_nverts,
    const int local,
    const double psi,
    const CswKernelParams params,
    const bool aw_planar,
    double* edge_force_r,
    double* edge_force_z) {
  double s_r = 0.0;
  double s_z = 0.0;
  if (active_nverts == 3) {
    if (!edge_svec_from_cell(x_r, x_z, nodes, active_nverts, local, &s_r,
                             &s_z, aw_planar)) {
      return;
    }
  } else {
    edge_svec_from_cell(x_r, x_z, nodes, local, &s_r, &s_z, aw_planar);
  }

  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const double du_mag = hypot(du_r, du_z);
  const double compression = du_r * s_r + du_z * s_z;
  if (compression <= 0.0 && finite_device(compression)) {
    atomicAdd(compressive_count, 1);
  }
  if (!(compression <= 0.0) || !(du_mag > kTiny) ||
      !finite_device(du_mag)) {
    return;
  }

  const double rho_e =
      harmonic_positive(rho[c], (neighbor >= 0) ? rho[neighbor] : rho[c]);
  const double cs_e =
      harmonic_positive(cs[c], (neighbor >= 0) ? cs[neighbor] : cs[c]);
  if (!(rho_e > 0.0) || !(cs_e >= 0.0)) {
    return;
  }

  double limiter_scale = fmax(0.0, 1.0 - fmin(1.0, fmax(0.0, psi)));
  if (params.limiter_shock_floor != 0) {
    limiter_scale = fmax(limiter_scale, params.shock_limiter_floor);
  }
  if (!(limiter_scale > 0.0)) {
    return;
  }
  const double c2g = params.c2 * (params.gamma + 1.0) * 0.25;
  const double wave =
      c2g * du_mag +
      sqrt(c2g * c2g * du_mag * du_mag + params.c1 * params.c1 * cs_e * cs_e);
  const double force_scale = rho_e * wave * limiter_scale * compression / du_mag;
  if (!finite_device(force_scale)) {
    return;
  }
  const double fr = force_scale * du_r;
  const double fz = force_scale * du_z;
  const double work = -(fr * du_r + fz * du_z);
  if (!(work >= 0.0) || !finite_device(work)) {
    atomicAdd(negative_work_count, 1);
  } else {
    if constexpr (AtomicWork) {
      atomic_add_double(work_slot, work);
    } else {
      *work_slot += work;
    }
  }
  *edge_force_r += fr;
  *edge_force_z += fz;
}

__device__ inline void structured_cell_face_contribution(
    double* __restrict__ work_av,
    int* __restrict__ compressive_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int c,
    const int local,
    const int nr,
    const int nz,
    const CswKernelParams params,
    const bool aw_planar,
    double* edge_force_r,
    double* edge_force_z) {
  if (!cell_active(hydro_active, c)) {
    return;
  }

  int nodes[4] = {0, 0, 0, 0};
  structured_cell_nodes(c, nz, nodes);
  int corner0 = 0;
  int corner1 = 0;
  local_face_corners(local, &corner0, &corner1);
  const int n0 = nodes[corner0];
  const int n1 = nodes[corner1];
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const double dx_mag = hypot(dx_r, dx_z);
  const double du_mag = hypot(du_r, du_z);
  if (!(dx_mag > kTiny) || !(du_mag > kTiny)) {
    double s_r = 0.0;
    double s_z = 0.0;
    edge_svec_from_cell(x_r, x_z, nodes, local, &s_r, &s_z, aw_planar);
    if (du_r * s_r + du_z * s_z <= 0.0) {
      atomicAdd(compressive_count, 1);
    }
    return;
  }

  const double xhat_r = dx_r / dx_mag;
  const double xhat_z = dx_z / dx_mag;
  const double uhat_r = du_r / du_mag;
  const double uhat_z = du_z / du_mag;
  const double current_grad = du_mag / dx_mag;
  const double psi = params.limiter_enabled != 0
                         ? structured_limiter(x_r, x_z, v_r, v_z, hydro_active,
                                              c, local, nr, nz, xhat_r, xhat_z,
                                              uhat_r, uhat_z, current_grad)
                         : 0.0;
  const int neighbor = structured_face_neighbor(c, local, nr, nz);
  apply_csw_contribution<true>(
      work_av + c, compressive_count, negative_work_count, x_r, x_z, v_r, v_z,
      rho, cs, n0, n1, c,
      cell_active(hydro_active, neighbor) ? neighbor : -1, nodes,
      mesh::kMeshTopoCellStorageSlots, local, psi, params, aw_planar,
      edge_force_r, edge_force_z);
}

__device__ inline void multiblock_cell_face_contribution(
    double* __restrict__ work_slot,
    int* __restrict__ compressive_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int c,
    const int local,
    const CswKernelParams params,
    const bool aw_planar,
    double* edge_force_r,
    double* edge_force_z) {
  if (!cell_active(hydro_active, c)) {
    return;
  }

  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {0};
  csr_cell_active_nodes(c, active_nverts, cell_node_offsets, cell_node_indices,
                        nodes);
  int corner0 = 0;
  int corner1 = 0;
  if (!local_face_corners(active_nverts, local, &corner0, &corner1)) {
    return;
  }
  const int n0 = nodes[corner0];
  const int n1 = nodes[corner1];
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const double dx_mag = hypot(dx_r, dx_z);
  const double du_mag = hypot(du_r, du_z);
  if (!(dx_mag > kTiny) || !(du_mag > kTiny)) {
    double s_r = 0.0;
    double s_z = 0.0;
    if (!edge_svec_from_cell(x_r, x_z, nodes, active_nverts, local, &s_r,
                             &s_z, aw_planar)) {
      return;
    }
    if (du_r * s_r + du_z * s_z <= 0.0) {
      atomicAdd(compressive_count, 1);
    }
    return;
  }

  const double xhat_r = dx_r / dx_mag;
  const double xhat_z = dx_z / dx_mag;
  const double uhat_r = du_r / du_mag;
  const double uhat_z = du_z / du_mag;
  const double current_grad = du_mag / dx_mag;
  const double psi =
      params.limiter_enabled != 0
          ? multiblock_limiter(x_r, x_z, v_r, v_z, hydro_active,
                               cell_node_offsets, cell_node_indices,
                               face_adj_offsets, face_adj_indices, cell_nverts,
                               c, local, xhat_r, xhat_z, uhat_r, uhat_z,
                               current_grad)
          : 0.0;
  const int neighbor = face_adj_indices[face_adj_offsets[c] + local];
  apply_csw_contribution<false>(
      work_slot, compressive_count, negative_work_count, x_r, x_z, v_r, v_z,
      rho, cs, n0, n1, c,
      cell_active(hydro_active, neighbor) ? neighbor : -1, nodes,
      active_nverts, local, psi, params, aw_planar, edge_force_r,
      edge_force_z);
}

// ==== csw_edge_csw98: CSW98-faithful median-mesh edge AV (I1-B Stage-G W1).
// Force per CSW98 Eq. 20, f = q_Kur (1-psi) (dv_hat.S) dv_hat, with the
// median-mesh S of compatible_av_csw.cuh (RZ pi*(r_c+r_m) weighting) and
// the Eq. 18 logical-line continuation-edge limiter (van Leer psi, Eq. 12).
// q_Kur = rho_e * wave * |dv| with the same Kuropatenko wave speed and
// harmonic rho_e/cs_e as the old mode, so the force assembly below is the
// old expression with the projection vector replaced. Deposit convention
// inherited: buffer force F is applied -F at n0 / +F at n1 by
// sum_edge_forces_*; compression = dv.S < 0 makes force_scale negative, so
// the pair force opposes dv and the assembly work -F.dv is positive.

__device__ inline double csw98_continuation_ratio_values(
    const double xa_r,
    const double xa_z,
    const double va_r,
    const double va_z,
    const double xb_r,
    const double xb_z,
    const double vb_r,
    const double vb_z,
    const double xhat_r,
    const double xhat_z,
    const double uhat_r,
    const double uhat_z,
    const double du_mag,
    const double dx_mag,
    int* miss) {
  const double dxn_r = xb_r - xa_r;
  const double dxn_z = xb_z - xa_z;
  const double dxn_mag = hypot(dxn_r, dxn_z);
  if (!(dxn_mag > kTiny) || !finite_device(dxn_mag)) {
    if (miss != nullptr) {
      atomicAdd(miss, 1);
    }
    return 1.0;
  }
  const double den_geom = dxn_r * xhat_r + dxn_z * xhat_z;
  if (!(fabs(den_geom) > kRatioTiny * dxn_mag)) {
    if (miss != nullptr) {
      atomicAdd(miss, 1);
    }
    return 1.0;
  }
  const double dun_r = vb_r - va_r;
  const double dun_z = vb_z - va_z;
  const double ratio =
      ((dun_r * uhat_r + dun_z * uhat_z) * dx_mag) / (den_geom * du_mag);
  return finite_device(ratio) ? ratio : 1.0;
}

__device__ inline double csw98_continuation_ratio(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int na,
    const int nb,
    const double xhat_r,
    const double xhat_z,
    const double uhat_r,
    const double uhat_z,
    const double du_mag,
    const double dx_mag,
    int* miss) {
  const double dxn_r = x_r[nb] - x_r[na];
  const double dxn_z = x_z[nb] - x_z[na];
  const double dxn_mag = hypot(dxn_r, dxn_z);
  if (!(dxn_mag > kTiny) || !finite_device(dxn_mag)) {
    if (miss != nullptr) {
      atomicAdd(miss, 1);
    }
    return 1.0;
  }
  const double den_geom = dxn_r * xhat_r + dxn_z * xhat_z;
  if (!(fabs(den_geom) > kRatioTiny * dxn_mag)) {
    if (miss != nullptr) {
      atomicAdd(miss, 1);
    }
    return 1.0;
  }
  const double dun_r = v_r[nb] - v_r[na];
  const double dun_z = v_z[nb] - v_z[na];
  const double ratio =
      ((dun_r * uhat_r + dun_z * uhat_z) * dx_mag) / (den_geom * du_mag);
  return finite_device(ratio) ? ratio : 1.0;
}

__device__ inline double csw98_axis_mirror_continuation_ratio(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int n,
    const int n_other,
    const bool prev,
    const double xhat_r,
    const double xhat_z,
    const double uhat_r,
    const double uhat_z,
    const double du_mag,
    const double dx_mag) {
  if (prev) {
    return csw98_continuation_ratio_values(
        -x_r[n_other], x_z[n_other], -v_r[n_other], v_z[n_other], x_r[n],
        x_z[n], v_r[n], v_z[n], xhat_r, xhat_z, uhat_r, uhat_z, du_mag,
        dx_mag, nullptr);
  }
  return csw98_continuation_ratio_values(
      x_r[n], x_z[n], v_r[n], v_z[n], -x_r[n_other], x_z[n_other],
      -v_r[n_other], v_z[n_other], xhat_r, xhat_z, uhat_r, uhat_z, du_mag,
      dx_mag, nullptr);
}

__device__ inline double csw98_structured_limiter(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int c,
    const int local,
    const int nr,
    const int nz,
    const double xhat_r,
    const double xhat_z,
    const double uhat_r,
    const double uhat_z,
    const double du_mag,
    const double dx_mag,
    const bool aw_planar,
    const bool aw_axis_slave_theta0_active,
    const bool aw_axis_slave_theta_pi_active) {
  const int i = c / nz;
  const int j = c - i * nz;
  int c_prev = -1;
  int c_next = -1;
  if (local == 2 || local == 3) {
    c_prev = (i > 0) ? (c - nz) : -1;
    c_next = (i + 1 < nr) ? (c + nz) : -1;
  } else {
    c_prev = (j > 0) ? (c - 1) : -1;
    c_next = (j + 1 < nz) ? (c + 1) : -1;
  }
  double r_prev = 1.0;
  double r_next = 1.0;
  int corner0 = 0;
  int corner1 = 0;
  local_face_corners(local, &corner0, &corner1);
  if (aw_planar && (local == 0 || local == 1) &&
      ((aw_axis_slave_theta0_active && j == 0) ||
       (aw_axis_slave_theta_pi_active && j == nz - 1))) {
    int nodes[4] = {0, 0, 0, 0};
    structured_cell_nodes(c, nz, nodes);
    // Complete the missing boundary-side continuation with H=diag(-1, 1).
    if (aw_axis_slave_theta0_active && j == 0) {
      const int axis = nodes[corner0];
      const int off_axis = nodes[corner1];
      r_prev = csw98_continuation_ratio_values(
          -x_r[off_axis], x_z[off_axis], -v_r[off_axis], v_z[off_axis],
          x_r[axis], x_z[axis], v_r[axis], v_z[axis], xhat_r, xhat_z,
          uhat_r, uhat_z, du_mag, dx_mag, nullptr);
    }
    if (aw_axis_slave_theta_pi_active && j == nz - 1) {
      const int off_axis = nodes[corner0];
      const int axis = nodes[corner1];
      r_next = csw98_continuation_ratio_values(
          x_r[axis], x_z[axis], v_r[axis], v_z[axis], -x_r[off_axis],
          x_z[off_axis], -v_r[off_axis], v_z[off_axis], xhat_r, xhat_z,
          uhat_r, uhat_z, du_mag, dx_mag, nullptr);
    }
  }
  if (cell_active(hydro_active, c_prev)) {
    int nodes_nb[4] = {0, 0, 0, 0};
    structured_cell_nodes(c_prev, nz, nodes_nb);
    r_prev = csw98_continuation_ratio(
        x_r, x_z, v_r, v_z, nodes_nb[corner0], nodes_nb[corner1], xhat_r,
        xhat_z, uhat_r, uhat_z, du_mag, dx_mag, nullptr);
  }
  if (cell_active(hydro_active, c_next)) {
    int nodes_nb[4] = {0, 0, 0, 0};
    structured_cell_nodes(c_next, nz, nodes_nb);
    r_next = csw98_continuation_ratio(
        x_r, x_z, v_r, v_z, nodes_nb[corner0], nodes_nb[corner1], xhat_r,
        xhat_z, uhat_r, uhat_z, du_mag, dx_mag, nullptr);
  }
  return limiter_from_ratios(r_prev, r_next);
}

__device__ inline double csw98_edge_prepass_ratio(
    const double dxk_r,
    const double dxk_z,
    const double duk_r,
    const double duk_z,
    const double dx_r,
    const double dx_z,
    const double du_r,
    const double du_z,
    const double ell,
    const double d) {
  const double num = duk_r * du_r + duk_z * du_z;
  const double den = dxk_r * dx_r + dxk_z * dx_z;
  if (!(fabs(den) > 1.0e-300) || !(d > 0.0)) {
    return 1.0;
  }
  const double ratio = (num * ell * ell) / (den * d * d);
  return finite_device(ratio) ? ratio : 1.0;
}

__global__ void csw98_edge_psi_prepass_kernel(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ cs,
    const int* __restrict__ line_edge_n0,
    const int* __restrict__ line_edge_n1,
    const int* __restrict__ line_prev_edge,
    const int* __restrict__ line_next_edge,
    const std::int8_t* __restrict__ line_prev_sign,
    const std::int8_t* __restrict__ line_next_sign,
    const int* __restrict__ line_cand_offsets,
    const int* __restrict__ line_cand_edges,
    const std::int8_t* __restrict__ line_cand_signs,
    const int* __restrict__ unique_face_cell_a,
    const int* __restrict__ unique_face_cell_b,
    const int* __restrict__ boundary_face_cell,
    const int n_internal,
    const int n_edges,
    double* __restrict__ edge_psi) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_edges) {
    return;
  }
  (void)line_prev_sign;
  (void)line_next_sign;

  const int a = line_edge_n0[e];
  const int b = line_edge_n1[e];
  const double dx_r = x_r[b] - x_r[a];
  const double dx_z = x_z[b] - x_z[a];
  const double du_r = v_r[b] - v_r[a];
  const double du_z = v_z[b] - v_z[a];
  const double ell = hypot(dx_r, dx_z);
  const double d = hypot(du_r, du_z);
  const double cs_ref =
      e < n_internal
          ? fmax(cs[unique_face_cell_a[e]], cs[unique_face_cell_b[e]])
          : cs[boundary_face_cell[e - n_internal]];

  double ratios[2] = {1.0, 1.0};
  double continuation_d[2] = {0.0, 0.0};
  for (int side = 0; side < 2; ++side) {
    const int continuation =
        side == 0 ? line_prev_edge[e] : line_next_edge[e];
    if (continuation >= 0) {
      const int ka = line_edge_n0[continuation];
      const int kb = line_edge_n1[continuation];
      const double dxk_r = x_r[kb] - x_r[ka];
      const double dxk_z = x_z[kb] - x_z[ka];
      const double duk_r = v_r[kb] - v_r[ka];
      const double duk_z = v_z[kb] - v_z[ka];
      continuation_d[side] = hypot(duk_r, duk_z);
      ratios[side] = csw98_edge_prepass_ratio(
          dxk_r, dxk_z, duk_r, duk_z, dx_r, dx_z, du_r, du_z, ell, d);
      continue;
    }
    if (continuation == -2) {
      double dxk_r = 0.0;
      double dxk_z = 0.0;
      double duk_r = 0.0;
      double duk_z = 0.0;
      if (side == 0) {
        dxk_r = x_r[a] + x_r[b];
        dxk_z = x_z[a] - x_z[b];
        duk_r = v_r[a] + v_r[b];
        duk_z = v_z[a] - v_z[b];
      } else {
        dxk_r = -x_r[a] - x_r[b];
        dxk_z = x_z[a] - x_z[b];
        duk_r = -v_r[a] - v_r[b];
        duk_z = v_z[a] - v_z[b];
      }
      continuation_d[side] = hypot(duk_r, duk_z);
      ratios[side] = csw98_edge_prepass_ratio(
          dxk_r, dxk_z, duk_r, duk_z, dx_r, dx_z, du_r, du_z, ell, d);
      continue;
    }
    if (continuation == -3) {
      const int row = 2 * e + side;
      bool accepted = false;
      double selected_ratio = -DBL_MAX;
      double selected_d = 0.0;
      for (int p = line_cand_offsets[row]; p < line_cand_offsets[row + 1];
           ++p) {
        const int candidate = line_cand_edges[p];
        const int ka = line_edge_n0[candidate];
        const int kb = line_edge_n1[candidate];
        const double dxk_r = x_r[kb] - x_r[ka];
        const double dxk_z = x_z[kb] - x_z[ka];
        const double sign = static_cast<double>(line_cand_signs[p]);
        if (!(sign * (dxk_r * dx_r + dxk_z * dx_z) > 0.0)) {
          continue;
        }
        const double duk_r = v_r[kb] - v_r[ka];
        const double duk_z = v_z[kb] - v_z[ka];
        const double candidate_ratio = csw98_edge_prepass_ratio(
            dxk_r, dxk_z, duk_r, duk_z, dx_r, dx_z, du_r, du_z, ell, d);
        if (!accepted || candidate_ratio > selected_ratio) {
          accepted = true;
          selected_ratio = candidate_ratio;
          selected_d = hypot(duk_r, duk_z);
        }
      }
      if (accepted) {
        ratios[side] = selected_ratio;
        continuation_d[side] = selected_d;
      }
    }
  }

  const double d_ref =
      fmax(fmax(cs_ref, d), fmax(continuation_d[0], continuation_d[1]));
  if (!(d > 64.0 * DBL_EPSILON * d_ref)) {
    edge_psi[e] = 1.0;
    return;
  }
  for (int side = 0; side < 2; ++side) {
    if (fabs(ratios[side] - 1.0) <=
        64.0 * DBL_EPSILON * fmax(1.0, fabs(ratios[side]))) {
      ratios[side] = 1.0;
    }
  }
  edge_psi[e] =
      fmax(0.0, fmin(fmin(1.0, 2.0 * ratios[0]),
                     fmin(2.0 * ratios[1], 0.5 * (ratios[0] + ratios[1]))));
}

__device__ inline int csw98_multiblock_cell_edge_id(
    const int* __restrict__ cell_edge_offsets,
    const int* __restrict__ cell_edge_edges,
    const int* __restrict__ line_edge_n0,
    const int* __restrict__ line_edge_n1,
    const int c,
    const int n0,
    const int n1) {
  for (int p = cell_edge_offsets[c]; p < cell_edge_offsets[c + 1]; ++p) {
    const int edge_id = cell_edge_edges[p];
    const int a = line_edge_n0[edge_id];
    const int b = line_edge_n1[edge_id];
    if ((a == n0 && b == n1) || (a == n1 && b == n0)) {
      return edge_id;
    }
  }
  return -1;
}

__device__ inline double csw98_multiblock_limiter(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int c,
    const int local,
    const int nodes[4],
    const int active_nverts,
    const int n0,
    const int n1,
    const double xhat_r,
    const double xhat_z,
    const double uhat_r,
    const double uhat_z,
    const double du_mag,
    const double dx_mag,
    const CswKernelParams params,
    int* miss,
    double* out_r0 = nullptr,
    double* out_r1 = nullptr) {
  double ratios[2] = {1.0, 1.0};
  for (int e = 0; e < 2; ++e) {
    const int n = (e == 0) ? n0 : n1;
    int l_perp = -1;
    const int face_scan_count =
        (active_nverts <= 4) ? 4 : active_nverts;
    for (int lf = 0; lf < face_scan_count; ++lf) {
      if (lf == local) {
        continue;
      }
      int a0 = 0;
      int a1 = 0;
      if (!local_face_corners(active_nverts, lf, &a0, &a1)) {
        continue;
      }
      if (nodes[a0] == n || nodes[a1] == n) {
        l_perp = lf;
        break;
      }
    }
    if (l_perp < 0) {
      if (params.axis_mirror_limiter != 0 &&
          x_r[n] <= params.axis_eps_cm) {
        const int n_other = (e == 0) ? n1 : n0;
        ratios[e] = csw98_axis_mirror_continuation_ratio(
            x_r, x_z, v_r, v_z, n, n_other, e == 0, xhat_r, xhat_z, uhat_r,
            uhat_z, du_mag, dx_mag);
        continue;
      }
      if (miss != nullptr) {
        atomicAdd(miss, 1);
      }
      continue;
    }
    const int nb = face_adj_indices[face_adj_offsets[c] + l_perp];
    if (!cell_active(hydro_active, nb)) {
      if (params.axis_mirror_limiter != 0 &&
          x_r[n] <= params.axis_eps_cm) {
        const int n_other = (e == 0) ? n1 : n0;
        ratios[e] = csw98_axis_mirror_continuation_ratio(
            x_r, x_z, v_r, v_z, n, n_other, e == 0, xhat_r, xhat_z, uhat_r,
            uhat_z, du_mag, dx_mag);
        continue;
      }
      if (miss != nullptr) {
        atomicAdd(miss, 1);
      }
      continue;
    }
    const int nb_nverts =
        mesh::mesh_topo_cell_active_nverts(cell_nverts, nb);
    int nb_nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {0};
    csr_cell_active_nodes(nb, nb_nverts, cell_node_offsets, cell_node_indices,
                          nb_nodes);
    int cont_a = -1;
    int cont_b = -1;
    const int nb_face_scan_count = (nb_nverts <= 4) ? 4 : nb_nverts;
    for (int lf = 0; lf < nb_face_scan_count; ++lf) {
      int b0 = 0;
      int b1 = 0;
      if (!local_face_corners(nb_nverts, lf, &b0, &b1)) {
        continue;
      }
      const int m0 = nb_nodes[b0];
      const int m1 = nb_nodes[b1];
      if (m0 != n && m1 != n) {
        continue;
      }
      if (face_adj_indices[face_adj_offsets[nb] + lf] == c) {
        continue;
      }
      cont_a = m0;
      cont_b = m1;
      break;
    }
    if (cont_a < 0) {
      if (params.axis_mirror_limiter != 0 &&
          x_r[n] <= params.axis_eps_cm) {
        const int n_other = (e == 0) ? n1 : n0;
        ratios[e] = csw98_axis_mirror_continuation_ratio(
            x_r, x_z, v_r, v_z, n, n_other, e == 0, xhat_r, xhat_z, uhat_r,
            uhat_z, du_mag, dx_mag);
        continue;
      }
      if (miss != nullptr) {
        atomicAdd(miss, 1);
      }
      continue;
    }
    ratios[e] = csw98_continuation_ratio(
        x_r, x_z, v_r, v_z, cont_a, cont_b, xhat_r, xhat_z, uhat_r, uhat_z,
        du_mag, dx_mag, miss);
  }
  if (out_r0) {
    *out_r0 = ratios[0];
  }
  if (out_r1) {
    *out_r1 = ratios[1];
  }
  return limiter_from_ratios(ratios[0], ratios[1]);
}

__device__ inline double csw98_multiblock_limiter_defect(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ edge_psi,
    const int* __restrict__ cell_edge_offsets,
    const int* __restrict__ cell_edge_edges,
    const int* __restrict__ line_edge_n0,
    const int* __restrict__ line_edge_n1,
    const int c,
    const int local,
    const CswKernelParams params,
    const bool aw_planar,
    int* limiter_miss) {
  if (!cell_active(hydro_active, c)) {
    return 0.0;
  }

  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {0};
  csr_cell_active_nodes(c, active_nverts, cell_node_offsets, cell_node_indices,
                        nodes);
  int corner0 = 0;
  int corner1 = 0;
  if (!local_face_corners(active_nverts, local, &corner0, &corner1)) {
    return 0.0;
  }
  const int n0 = nodes[corner0];
  const int n1 = nodes[corner1];
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const bool axis_line_edge =
      multiblock_aw_axis_line_edge(x_r, n0, n1, aw_planar, params);
  if (axis_line_edge && params.axisline_d1prime == 0) {
    return 0.0;
  }
  if (axis_line_edge) {
    du_r = 0.0;
  }
  const double dx_mag = hypot(dx_r, dx_z);
  const double du_mag = hypot(du_r, du_z);
  if (!(dx_mag > kTiny) || !(du_mag > kTiny)) {
    return 0.0;
  }
  if (params.limiter_enabled == 0) {
    return 1.0;
  }

  (void)limiter_miss;
  const int edge_id = csw98_multiblock_cell_edge_id(
      cell_edge_offsets, cell_edge_edges, line_edge_n0, line_edge_n1, c, n0,
      n1);
  const double psi = edge_psi[edge_id];
  return fmax(0.0, 1.0 - fmin(1.0, fmax(0.0, psi)));
}

__device__ inline double csw98_multiblock_accumulate_cell_limiter_defect(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ edge_psi,
    const int* __restrict__ cell_edge_offsets,
    const int* __restrict__ cell_edge_edges,
    const int* __restrict__ line_edge_n0,
    const int* __restrict__ line_edge_n1,
    const int c,
    const CswKernelParams params,
    const bool aw_planar,
    double max_defect,
    int* limiter_miss) {
  if (!cell_active(hydro_active, c)) {
    return max_defect;
  }
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  const int face_scan_count = (active_nverts <= 4) ? 4 : active_nverts;
  for (int local = 0; local < face_scan_count; ++local) {
    int corner0 = 0;
    int corner1 = 0;
    if (!local_face_corners(active_nverts, local, &corner0, &corner1)) {
      continue;
    }
    max_defect = fmax(
        max_defect,
        csw98_multiblock_limiter_defect(
            x_r, x_z, v_r, v_z, hydro_active, cell_node_offsets,
            cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts,
            edge_psi, cell_edge_offsets, cell_edge_edges, line_edge_n0,
            line_edge_n1, c, local, params, aw_planar, limiter_miss));
  }
  return max_defect;
}

struct Csw98PoleLimiterSensor {
  double pole_weight = 0.0;
  double max_defect = 0.0;
};

__device__ inline Csw98PoleLimiterSensor
csw98_multiblock_pole_limiter_sensor(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ edge_psi,
    const int* __restrict__ cell_edge_offsets,
    const int* __restrict__ cell_edge_edges,
    const int* __restrict__ line_edge_n0,
    const int* __restrict__ line_edge_n1,
    const int c,
    const int neighbor,
    const int n0,
    const int n1,
    const double theta0,
    const double thetaf,
    const CswKernelParams params,
    const bool aw_planar,
    int* limiter_miss) {
  Csw98PoleLimiterSensor sensor;
  const double r_mid = 0.5 * (x_r[n0] + x_r[n1]);
  const double z_mid = 0.5 * (x_z[n0] + x_z[n1]);
  const double theta = atan2(r_mid, z_mid);
  const double d = fmin(theta, kPi - theta);
  double pole_weight = 0.0;
  if (d < theta0) {
    pole_weight = 1.0;
  } else if (thetaf > 0.0 && d < theta0 + thetaf) {
    pole_weight =
        0.5 *
        (1.0 + cos(kPi * (d - theta0) / thetaf));
  }
  if (!(pole_weight > 0.0)) {
    return sensor;
  }

  double max_defect = 0.0;
  max_defect = csw98_multiblock_accumulate_cell_limiter_defect(
      x_r, x_z, v_r, v_z, hydro_active, cell_node_offsets, cell_node_indices,
      face_adj_offsets, face_adj_indices, cell_nverts, edge_psi,
      cell_edge_offsets, cell_edge_edges, line_edge_n0, line_edge_n1, c,
      params, aw_planar, max_defect, limiter_miss);
  if (cell_active(hydro_active, neighbor)) {
    max_defect = csw98_multiblock_accumulate_cell_limiter_defect(
        x_r, x_z, v_r, v_z, hydro_active, cell_node_offsets,
        cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts,
        edge_psi, cell_edge_offsets, cell_edge_edges, line_edge_n0,
        line_edge_n1, neighbor, params, aw_planar, max_defect, limiter_miss);
  }
  sensor.pole_weight = pole_weight;
  sensor.max_defect = max_defect;
  return sensor;
}

__device__ inline double csw98_multiblock_pole_limiter_scale(
    const double s_side,
    const Csw98PoleLimiterSensor sensor,
    const CswKernelParams params) {
  if (!(sensor.pole_weight > 0.0) || !(sensor.max_defect > 0.0)) {
    return s_side;
  }
  if (params.pole_floor_enabled != 0) {
    const double limiter_floor = params.pole_floor_sigma0 *
                                 sensor.pole_weight * sensor.max_defect;
    return fmax(s_side, limiter_floor);
  }
  if (params.pole_desens_enabled != 0) {
    const double desens_e =
        1.0 - (params.pole_desens_alpha * sensor.pole_weight *
               sensor.max_defect);
    return s_side * desens_e;
  }
  return s_side;
}

// W1b (2026-08-17, free-stream defect dossier §6.2): a side whose length
// is degenerate relative to its cell transmits no viscous momentum flux.
// Fire only when l^2 >= eta^2 * A_cell (shoelace area, absolute value).
__device__ inline bool csw98_side_degenerate(
    const double* cell_r, const double* cell_z, const int nv,
    const int n0, const int n1,
    const double* __restrict__ x_r, const double* __restrict__ x_z,
    const double floor_rel) {
  if (!(floor_rel > 0.0)) {
    return false;
  }
  const double dr = x_r[n1] - x_r[n0];
  const double dz = x_z[n1] - x_z[n0];
  const double l2 = dr * dr + dz * dz;
  double twice_area = 0.0;
  for (int k = 0; k < nv; ++k) {
    const int j = (k + 1 == nv) ? 0 : k + 1;
    twice_area += cell_r[k] * cell_z[j] - cell_r[j] * cell_z[k];
  }
  const double area = 0.5 * fabs(twice_area);
  return l2 < floor_rel * floor_rel * area;
}

template <bool AtomicWork>
__device__ inline void csw98_side_force(
    double* __restrict__ work_slot,
    int* __restrict__ fire_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const int n0,
    const int n1,
    const int corner0,
    const int corner1,
    const int c,
    const int neighbor,
    const int nodes[4],
    const int active_nverts,
    const double psi,
    const CswKernelParams params,
    const bool aw_planar,
    const bool axisline_d1prime_edge,
    double* edge_force_r,
    double* edge_force_z) {
  double limiter_scale =
      fmax(0.0, 1.0 - fmin(1.0, fmax(0.0, psi)));
  const int nv = active_nverts;
  double cell_r[kCsw98MaxSideVecs];
  double cell_z[kCsw98MaxSideVecs];
  for (int k = 0; k < nv; ++k) {
    cell_r[k] = x_r[nodes[k]];
    cell_z[k] = x_z[nodes[k]];
  }
  double s_r = 0.0;
  double s_z = 0.0;
  if (!csw98_c2_svec_for_side(cell_r, cell_z, nv, corner0, corner1, &s_r,
                              &s_z, aw_planar)) {
    return;
  }
  if (csw98_side_degenerate(cell_r, cell_z, nv, n0, n1, x_r, x_z,
                            params.degenerate_side_floor_rel)) {
    return;
  }
  double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  if (axisline_d1prime_edge) {
    du_r = 0.0;
  }
  const double du_mag = hypot(du_r, du_z);
  const double compression = du_r * s_r + du_z * s_z;
  if (!(compression < 0.0) || !(du_mag > kTiny) ||
      !finite_device(compression) || !finite_device(du_mag)) {
    return;
  }
  atomicAdd(fire_count, 1);
  if (limiter_scale == 0.0 && params.limiter_shock_floor == 0) {
    return;
  }
  const double rho_e =
      harmonic_positive(rho[c], (neighbor >= 0) ? rho[neighbor] : rho[c]);
  const double cs_e =
      harmonic_positive(cs[c], (neighbor >= 0) ? cs[neighbor] : cs[c]);
  if (!(rho_e > 0.0) || !(cs_e >= 0.0)) {
    return;
  }
  if (params.limiter_shock_floor != 0) {
    limiter_scale = fmax(limiter_scale, params.shock_limiter_floor);
  }
  if (!(limiter_scale > 0.0)) {
    return;
  }
  const double c2g = params.c2 * (params.gamma + 1.0) * 0.25;
  const double wave =
      c2g * du_mag +
      sqrt(c2g * c2g * du_mag * du_mag + params.c1 * params.c1 * cs_e * cs_e);
  const double force_scale = rho_e * wave * limiter_scale * compression / du_mag;
  if (!finite_device(force_scale)) {
    return;
  }
  const double fr = force_scale * du_r;
  const double fz = force_scale * du_z;
  double clamp_scale = 1.0;
  if (params.damper_impulse_beta > 0.0 && params.dt > 0.0 &&
      node_mass != nullptr) {
    const double m0 = node_mass[n0];
    const double m1 = node_mass[n1];
    if (m0 > 0.0 && m1 > 0.0) {
      const double mu = m0 * m1 / (m0 + m1);
      const double f_mag = hypot(fr, fz);
      const double instances = (neighbor >= 0) ? 2.0 : 1.0;
      const double cap =
          params.damper_impulse_beta * mu * du_mag / instances;
      const double impulse = f_mag * params.dt;
      if (impulse > cap) {
        clamp_scale = cap / impulse;
      }
    }
  }
  const double fr_c = axisline_d1prime_edge ? 0.0 : fr * clamp_scale;
  const double fz_c = fz * clamp_scale;
#ifdef TENRYU_CSW98_DIAG
  if (fabs(fr_c) + fabs(fz_c) > kCsw98DiagForceFloor) {
    const double l_side = hypot(x_r[n1] - x_r[n0], x_z[n1] - x_z[n0]);
    printf("[csw98diag] side c=%d nb=%d n0=%d n1=%d l=%.3e S=(%.3e,%.3e) "
           "du=(%.3e,%.3e) dumag=%.3e comp=%.3e psi=%.3e rho=%.3e cs=%.3e "
           "wave=%.3e fscale=%.3e f=(%.3e,%.3e) clamp=%.3e\n",
           c, neighbor, n0, n1, l_side, s_r, s_z, du_r, du_z, du_mag,
           compression, psi, rho_e, cs_e, wave, force_scale, fr_c, fz_c,
           clamp_scale);
  }
#endif
  const double work = -(fr_c * du_r + fz_c * du_z);
  if (!(work >= 0.0) || !finite_device(work)) {
    atomicAdd(negative_work_count, 1);
  } else {
    if constexpr (AtomicWork) {
      atomic_add_double(work_slot, work);
    } else {
      *work_slot += work;
    }
  }
  *edge_force_r += fr_c;
  *edge_force_z += fz_c;
}

__device__ inline void csw98_multiblock_side_force(
    double* __restrict__ work_slot,
    int* __restrict__ fire_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const int n0,
    const int n1,
    const int corner0,
    const int corner1,
    const int c,
    const int neighbor,
    const int nodes[4],
    const int active_nverts,
    const double psi,
    const Csw98PoleLimiterSensor pole_sensor,
    const CswKernelParams params,
    const bool aw_planar,
    const bool axisline_d1prime_edge,
    double* edge_force_r,
    double* edge_force_z) {
  double s_side = fmax(0.0, 1.0 - fmin(1.0, fmax(0.0, psi)));
  const int nv = active_nverts;
  double cell_r[kCsw98MaxSideVecs];
  double cell_z[kCsw98MaxSideVecs];
  for (int k = 0; k < nv; ++k) {
    cell_r[k] = x_r[nodes[k]];
    cell_z[k] = x_z[nodes[k]];
  }
  double s_r = 0.0;
  double s_z = 0.0;
  if (!csw98_c2_svec_for_side(cell_r, cell_z, nv, corner0, corner1, &s_r,
                              &s_z, aw_planar)) {
    return;
  }
  if (csw98_side_degenerate(cell_r, cell_z, nv, n0, n1, x_r, x_z,
                            params.degenerate_side_floor_rel)) {
    return;
  }
  double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  if (axisline_d1prime_edge) {
    du_r = 0.0;
  }
  const double du_mag = hypot(du_r, du_z);
  const double compression = du_r * s_r + du_z * s_z;
  if (!(compression < 0.0) || !(du_mag > kTiny) ||
      !finite_device(compression) || !finite_device(du_mag)) {
    return;
  }
  atomicAdd(fire_count, 1);
  if (s_side == 0.0 && params.limiter_shock_floor == 0) {
    return;
  }
  const double rho_e =
      harmonic_positive(rho[c], (neighbor >= 0) ? rho[neighbor] : rho[c]);
  const double cs_e =
      harmonic_positive(cs[c], (neighbor >= 0) ? cs[neighbor] : cs[c]);
  if (!(rho_e > 0.0) || !(cs_e >= 0.0)) {
    return;
  }
  if (params.limiter_shock_floor != 0) {
    s_side = fmax(s_side, params.shock_limiter_floor);
  }
  const double lim_scale = csw98_multiblock_pole_limiter_scale(
      s_side, pole_sensor, params);
  if (!(lim_scale > 0.0)) {
    return;
  }
  const double c2g = params.c2 * (params.gamma + 1.0) * 0.25;
  const double wave =
      c2g * du_mag +
      sqrt(c2g * c2g * du_mag * du_mag + params.c1 * params.c1 * cs_e * cs_e);
  const double force_scale =
      rho_e * wave * lim_scale * compression / du_mag;
  if (!finite_device(force_scale)) {
    return;
  }
  const double fr = force_scale * du_r;
  const double fz = force_scale * du_z;
  double clamp_scale = 1.0;
  if (params.damper_impulse_beta > 0.0 && params.dt > 0.0 &&
      node_mass != nullptr) {
    const double m0 = node_mass[n0];
    const double m1 = node_mass[n1];
    if (m0 > 0.0 && m1 > 0.0) {
      const double mu = m0 * m1 / (m0 + m1);
      const double f_mag = hypot(fr, fz);
      const double instances = (neighbor >= 0) ? 2.0 : 1.0;
      const double cap =
          params.damper_impulse_beta * mu * du_mag / instances;
      const double impulse = f_mag * params.dt;
      if (impulse > cap) {
        clamp_scale = cap / impulse;
      }
    }
  }
  const double fr_c = axisline_d1prime_edge ? 0.0 : fr * clamp_scale;
  const double fz_c = fz * clamp_scale;
#ifdef TENRYU_CSW98_DIAG
  if (fabs(fr_c) + fabs(fz_c) > kCsw98DiagForceFloor) {
    const double l_side = hypot(x_r[n1] - x_r[n0], x_z[n1] - x_z[n0]);
    printf("[csw98diag] side c=%d nb=%d n0=%d n1=%d l=%.3e S=(%.3e,%.3e) "
           "du=(%.3e,%.3e) dumag=%.3e comp=%.3e psi=%.3e rho=%.3e cs=%.3e "
           "wave=%.3e fscale=%.3e f=(%.3e,%.3e) lim=%.3e clamp=%.3e\n",
           c, neighbor, n0, n1, l_side, s_r, s_z, du_r, du_z, du_mag,
           compression, psi, rho_e, cs_e, wave, force_scale, fr_c, fz_c,
           lim_scale, clamp_scale);
  }
#endif
  const double work = -(fr_c * du_r + fz_c * du_z);
  if (!(work >= 0.0) || !finite_device(work)) {
    atomicAdd(negative_work_count, 1);
  } else {
    *work_slot += work;
  }
  *edge_force_r += fr_c;
  *edge_force_z += fz_c;
}

__device__ inline void csw98_structured_cell_face_contribution(
    double* __restrict__ work_av,
    int* __restrict__ fire_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int c,
    const int local,
    const int nr,
    const int nz,
    const CswKernelParams params,
    const bool aw_planar,
    const bool aw_axis_slave_theta0_active,
    const bool aw_axis_slave_theta_pi_active,
    const bool axisline_d1prime_edge,
    double* edge_force_r,
    double* edge_force_z) {
  if (!cell_active(hydro_active, c)) {
    return;
  }

  int nodes[4] = {0, 0, 0, 0};
  structured_cell_nodes(c, nz, nodes);
  int corner0 = 0;
  int corner1 = 0;
  local_face_corners(local, &corner0, &corner1);
  const int n0 = nodes[corner0];
  const int n1 = nodes[corner1];
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  if (axisline_d1prime_edge) {
    du_r = 0.0;
  }
  const double dx_mag = hypot(dx_r, dx_z);
  const double du_mag = hypot(du_r, du_z);
  if (!(dx_mag > kTiny) || !(du_mag > kTiny)) {
    return;
  }

  const double xhat_r = dx_r / dx_mag;
  const double xhat_z = dx_z / dx_mag;
  const double uhat_r = du_r / du_mag;
  const double uhat_z = du_z / du_mag;
  const double psi =
      params.limiter_enabled != 0
          ? csw98_structured_limiter(x_r, x_z, v_r, v_z, hydro_active, c,
                                     local, nr, nz, xhat_r, xhat_z, uhat_r,
                                     uhat_z, du_mag, dx_mag, aw_planar,
                                     aw_axis_slave_theta0_active,
                                     aw_axis_slave_theta_pi_active)
          : 0.0;
  const int neighbor = structured_face_neighbor(c, local, nr, nz);
  csw98_side_force<true>(
      work_av + c, fire_count, negative_work_count, x_r, x_z, v_r, v_z,
      node_mass, rho, cs, n0, n1, corner0, corner1, c,
      cell_active(hydro_active, neighbor) ? neighbor : -1, nodes,
      mesh::kMeshTopoCellStorageSlots, psi, params, aw_planar,
      axisline_d1prime_edge, edge_force_r, edge_force_z);
}

__device__ inline void csw98_multiblock_cell_face_contribution(
    double* __restrict__ work_slot,
    int* __restrict__ fire_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ edge_psi,
    const int* __restrict__ cell_edge_offsets,
    const int* __restrict__ cell_edge_edges,
    const int* __restrict__ line_edge_n0,
    const int* __restrict__ line_edge_n1,
    const int c,
    const int local,
    const CswKernelParams params,
    const bool aw_planar,
    double* edge_force_r,
    double* edge_force_z,
    int* limiter_miss) {
  if (!cell_active(hydro_active, c)) {
    return;
  }

  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {0};
  csr_cell_active_nodes(c, active_nverts, cell_node_offsets, cell_node_indices,
                        nodes);
  int corner0 = 0;
  int corner1 = 0;
  if (!local_face_corners(active_nverts, local, &corner0, &corner1)) {
    return;
  }
  const int n0 = nodes[corner0];
  const int n1 = nodes[corner1];
  const int edge_id = csw98_multiblock_cell_edge_id(
      cell_edge_offsets, cell_edge_edges, line_edge_n0, line_edge_n1, c, n0,
      n1);
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const bool axis_line_edge =
      multiblock_aw_axis_line_edge(x_r, n0, n1, aw_planar, params);
  if (axis_line_edge && params.axisline_d1prime == 0) {
    return;
  }
  if (axis_line_edge) {
    du_r = 0.0;
  }
  const double dx_mag = hypot(dx_r, dx_z);
  const double du_mag = hypot(du_r, du_z);
  if (!(dx_mag > kTiny) || !(du_mag > kTiny)) {
    return;
  }

  if (multiblock_axis_touch_edge(x_r, n0, n1, aw_planar, params)) {
    return;
  }

  const double xhat_r = dx_r / dx_mag;
  const double xhat_z = dx_z / dx_mag;
  const double uhat_r = du_r / du_mag;
  const double uhat_z = du_z / du_mag;
  if (params.limiter_enabled != 0 && limiter_miss != nullptr) {
    // limiter evaluation counter (diag buffer slot [1]); reached only
    // when dx_mag/du_mag > tiny, i.e. the limiter actually runs.
    atomicAdd(limiter_miss + 1, 1);
  }
  double diag_r0 = 0.0;
  double diag_r1 = 0.0;
  double psi_legacy = 0.0;
  if (params.limiter_enabled != 0 && params.edge_diag_enabled != 0) {
    psi_legacy = csw98_multiblock_limiter(
        x_r, x_z, v_r, v_z, hydro_active, cell_node_offsets,
        cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts, c,
        local, nodes, active_nverts, n0, n1, xhat_r, xhat_z, uhat_r, uhat_z,
        du_mag, dx_mag, params, limiter_miss, &diag_r0, &diag_r1);
  }
  const double psi =
      params.limiter_enabled != 0 ? edge_psi[edge_id] : 0.0;
  if (params.edge_diag_enabled != 0) {
    bool watch = false;
    for (int w = 0; w < 8; ++w) {
      const int id = params.edge_diag_nodes[w];
      if (id >= 0 && (id == n0 || id == n1)) {
        watch = true;
        break;
      }
    }
    if (watch) {
      double diag_s_r = 0.0;
      double diag_s_z = 0.0;
      if (active_nverts == 3) {
        edge_svec_from_cell(x_r, x_z, nodes, active_nverts, local, &diag_s_r,
                            &diag_s_z, aw_planar);
      } else {
        edge_svec_from_cell(x_r, x_z, nodes, local, &diag_s_r, &diag_s_z,
                            aw_planar);
      }
      const double diag_compression =
          du_r * diag_s_r + du_z * diag_s_z;
      printf("[csw98_edge] call=%lld c=%d local=%d n0=%d n1=%d "
             "duR=%.10e duZ=%.10e sR=%.10e sZ=%.10e chi=%.10e "
             "psi_legacy=%.10e psi_new=%.10e r0=%.10e r1=%.10e "
             "axis_line=%d\n",
             params.edge_diag_call, c, local, n0, n1, du_r, du_z, diag_s_r,
             diag_s_z, diag_compression, psi_legacy, edge_psi[edge_id], diag_r0,
             diag_r1,
             axis_line_edge ? 1 : 0);
    }
  }
  const int neighbor = face_adj_indices[face_adj_offsets[c] + local];
  const int active_neighbor =
      cell_active(hydro_active, neighbor) ? neighbor : -1;
  if (params.pole_floor_enabled == 0 && params.pole_desens_enabled == 0) {
    csw98_side_force<false>(
        work_slot, fire_count, negative_work_count, x_r, x_z, v_r, v_z,
        node_mass, rho, cs, n0, n1, corner0, corner1, c, active_neighbor,
        nodes, active_nverts, psi, params, aw_planar, axis_line_edge,
        edge_force_r, edge_force_z);
    return;
  }
  const bool pole_floor_enabled = params.pole_floor_enabled != 0;
  const Csw98PoleLimiterSensor pole_sensor =
      csw98_multiblock_pole_limiter_sensor(
          x_r, x_z, v_r, v_z, hydro_active, cell_node_offsets,
          cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts,
          edge_psi, cell_edge_offsets, cell_edge_edges, line_edge_n0,
          line_edge_n1, c, neighbor, n0, n1,
          pole_floor_enabled ? params.pole_floor_theta0
                             : params.pole_desens_theta0,
          pole_floor_enabled ? params.pole_floor_thetaf
                             : params.pole_desens_thetaf,
          params, aw_planar, limiter_miss);
  csw98_multiblock_side_force(
      work_slot, fire_count, negative_work_count, x_r, x_z, v_r, v_z,
      node_mass, rho, cs, n0, n1, corner0, corner1, c, active_neighbor, nodes,
      active_nverts, psi, pole_sensor, params, aw_planar, axis_line_edge,
      edge_force_r, edge_force_z);
}

__device__ inline void csw98_multiblock_cell_face_contribution_polar_slaving(
    double* __restrict__ work_slot,
    int* __restrict__ fire_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ edge_psi,
    const int* __restrict__ cell_edge_offsets,
    const int* __restrict__ cell_edge_edges,
    const int* __restrict__ line_edge_n0,
    const int* __restrict__ line_edge_n1,
    const int c,
    const int local,
    const CswKernelParams params,
    const bool aw_planar,
    const CswPolarSlavingDeviceView slaving,
    double* edge_force_r,
    double* edge_force_z,
    int* limiter_miss) {
  if (!cell_active(hydro_active, c)) {
    return;
  }

  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {0};
  csr_cell_active_nodes(c, active_nverts, cell_node_offsets, cell_node_indices,
                        nodes);
  int corner0 = 0;
  int corner1 = 0;
  if (!local_face_corners(active_nverts, local, &corner0, &corner1)) {
    return;
  }
  const int n0 = nodes[corner0];
  const int n1 = nodes[corner1];
  const int edge_id = csw98_multiblock_cell_edge_id(
      cell_edge_offsets, cell_edge_edges, line_edge_n0, line_edge_n1, c, n0,
      n1);
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const bool axis_line_edge =
      multiblock_aw_axis_line_edge(x_r, n0, n1, aw_planar, params);
  if (axis_line_edge && params.axisline_d1prime == 0) {
    return;
  }
  if (axis_line_edge) {
    du_r = 0.0;
  }
  const double dx_mag = hypot(dx_r, dx_z);
  const double du_mag = hypot(du_r, du_z);
  if (!(dx_mag > kTiny) || !(du_mag > kTiny)) {
    return;
  }

  if (multiblock_axis_touch_edge(x_r, n0, n1, aw_planar, params)) {
    return;
  }

  const int lookup = face_adj_offsets[c] + local;
  const int instance = slaving.instance_lookup[lookup];
  double psi = 0.0;
  if (instance >= 0) {
    psi = csw98_polar_slaving_effective_psi(slaving, instance);
  } else {
    if (params.limiter_enabled != 0 && limiter_miss != nullptr) {
      atomicAdd(limiter_miss + 1, 1);
    }
    psi = params.limiter_enabled != 0 ? edge_psi[edge_id] : 0.0;
  }
  const int neighbor = face_adj_indices[lookup];
  const int active_neighbor =
      cell_active(hydro_active, neighbor) ? neighbor : -1;
  csw98_side_force<false>(
      work_slot, fire_count, negative_work_count, x_r, x_z, v_r, v_z,
      node_mass, rho, cs, n0, n1, corner0, corner1, c, active_neighbor, nodes,
      active_nverts, psi, params, aw_planar, axis_line_edge, edge_force_r,
      edge_force_z);
}

__device__ inline void csw98_structured_side_cfl(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int c,
    const int local,
    const int nr,
    const int nz,
    const CswKernelParams params,
    const double coefficient,
    const bool aw_planar,
    const bool aw_axis_slave_theta0_active,
    const bool aw_axis_slave_theta_pi_active,
    const bool axisline_d1prime_edge,
    int* __restrict__ winner_index,
    const int candidate_index,
    const double target_dt) {
  if (!cell_active(hydro_active, c)) {
    return;
  }

  int nodes[4] = {0, 0, 0, 0};
  structured_cell_nodes(c, nz, nodes);
  int corner0 = 0;
  int corner1 = 0;
  local_face_corners(local, &corner0, &corner1);
  const int n0 = nodes[corner0];
  const int n1 = nodes[corner1];
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  if (axisline_d1prime_edge) {
    du_r = 0.0;
  }
  const double dx_mag = hypot(dx_r, dx_z);
  const double du_mag = hypot(du_r, du_z);
  if (!(dx_mag > kTiny) || !(du_mag > kTiny)) {
    return;
  }

  double cell_r[4];
  double cell_z[4];
  for (int k = 0; k < mesh::kMeshTopoCellStorageSlots; ++k) {
    cell_r[k] = x_r[nodes[k]];
    cell_z[k] = x_z[nodes[k]];
  }
  double s_r = 0.0;
  double s_z = 0.0;
  if (!csw98_c2_svec_for_side(cell_r, cell_z,
                              mesh::kMeshTopoCellStorageSlots, corner0,
                              corner1, &s_r, &s_z, aw_planar)) {
    return;
  }
  const double compression = du_r * s_r + du_z * s_z;
  if (!(compression < 0.0) || !finite_device(compression)) {
    return;
  }

  const double xhat_r = dx_r / dx_mag;
  const double xhat_z = dx_z / dx_mag;
  const double uhat_r = du_r / du_mag;
  const double uhat_z = du_z / du_mag;
  const double psi =
      params.limiter_enabled != 0
          ? csw98_structured_limiter(x_r, x_z, v_r, v_z, hydro_active, c,
                                     local, nr, nz, xhat_r, xhat_z, uhat_r,
                                     uhat_z, du_mag, dx_mag, aw_planar,
                                     aw_axis_slave_theta0_active,
                                     aw_axis_slave_theta_pi_active)
          : 0.0;
  const double s_mag = hypot(s_r, s_z);
  if (!(s_mag > 0.0)) {
    return;
  }
  const double proj = fabs(compression) / (du_mag * s_mag);
  const double psi_clamped = fmin(1.0, fmax(0.0, psi));
  if (axisline_d1prime_edge) {
    if (params.axisline_d1prime_cfl != 0) {
      const int neighbor = structured_face_neighbor(c, local, nr, nz);
      const int active_neighbor =
          cell_active(hydro_active, neighbor) ? neighbor : -1;
      double dt_cand = 0.0;
      if (csw98_axisline_d1prime_cfl_dt(
              &dt_cand, node_mass, rho, cs, n0, n1, c, active_neighbor,
              du_z, s_z, fmax(0.0, 1.0 - psi_clamped), params)) {
        atomic_min_double(min_dt, dt_cand);
        if (winner_index != nullptr && edge_dt_matches(dt_cand, target_dt)) {
          atomicMin(winner_index, candidate_index);
        }
      }
    }
    return;
  }
  const double du_eff = du_mag * fmax(0.0, 1.0 - psi_clamped) * proj;
  if (du_eff > kTiny && isfinite(du_eff) && dx_mag > 0.0) {
    const double dt = coefficient * dx_mag / du_eff;
    atomic_min_double(min_dt, dt);
    if (winner_index != nullptr && edge_dt_matches(dt, target_dt)) {
      atomicMin(winner_index, candidate_index);
    }
  }
}

__device__ inline void csw98_multiblock_side_cfl(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ edge_psi,
    const int* __restrict__ cell_edge_offsets,
    const int* __restrict__ cell_edge_edges,
    const int* __restrict__ line_edge_n0,
    const int* __restrict__ line_edge_n1,
    const int edge_id,
    const int c,
    const int local,
    const CswKernelParams params,
    const double coefficient,
    const bool aw_planar,
    int* __restrict__ winner_index,
    const int candidate_index,
    const double target_dt) {
  if (!cell_active(hydro_active, c)) {
    return;
  }

  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {0};
  csr_cell_active_nodes(c, active_nverts, cell_node_offsets, cell_node_indices,
                        nodes);
  int corner0 = 0;
  int corner1 = 0;
  if (!local_face_corners(active_nverts, local, &corner0, &corner1)) {
    return;
  }
  const int n0 = nodes[corner0];
  const int n1 = nodes[corner1];
  const bool axis_line_edge =
      multiblock_aw_axis_line_edge(x_r, n0, n1, aw_planar, params);
  if (axis_line_edge && params.axisline_d1prime == 0) {
    return;
  }
  if (multiblock_axis_touch_edge(x_r, n0, n1, aw_planar, params)) {
    return;
  }
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  if (axis_line_edge) {
    du_r = 0.0;
  }
  const double dx_mag = hypot(dx_r, dx_z);
  const double du_mag = hypot(du_r, du_z);
  if (!(dx_mag > kTiny) || !(du_mag > kTiny)) {
    return;
  }

  const int nv = active_nverts;
  double cell_r[kCsw98MaxSideVecs];
  double cell_z[kCsw98MaxSideVecs];
  for (int k = 0; k < nv; ++k) {
    cell_r[k] = x_r[nodes[k]];
    cell_z[k] = x_z[nodes[k]];
  }
  double s_r = 0.0;
  double s_z = 0.0;
  if (!csw98_c2_svec_for_side(cell_r, cell_z, nv, corner0, corner1, &s_r,
                              &s_z, aw_planar)) {
    return;
  }
  const double compression = du_r * s_r + du_z * s_z;
  if (!(compression < 0.0) || !finite_device(compression)) {
    return;
  }

  const double psi =
      params.limiter_enabled != 0 ? edge_psi[edge_id] : 0.0;
  const double s_mag = hypot(s_r, s_z);
  if (!(s_mag > 0.0)) {
    return;
  }
  const double proj = fabs(compression) / (du_mag * s_mag);
  const double psi_clamped = fmin(1.0, fmax(0.0, psi));
  if (params.pole_floor_enabled == 0 && params.pole_desens_enabled == 0) {
    if (axis_line_edge) {
      if (params.axisline_d1prime_cfl != 0) {
        const int neighbor = face_adj_indices[face_adj_offsets[c] + local];
        const int active_neighbor =
            cell_active(hydro_active, neighbor) ? neighbor : -1;
        double dt_cand = 0.0;
        if (csw98_axisline_d1prime_cfl_dt(
                &dt_cand, node_mass, rho, cs, n0, n1, c, active_neighbor,
                du_z, s_z, fmax(0.0, 1.0 - psi_clamped), params)) {
          atomic_min_double(min_dt, dt_cand);
          if (winner_index != nullptr && edge_dt_matches(dt_cand, target_dt)) {
            atomicMin(winner_index, candidate_index);
          }
        }
      }
      return;
    }
    const double du_eff = du_mag * fmax(0.0, 1.0 - psi_clamped) * proj;
    if (du_eff > kTiny && isfinite(du_eff) && dx_mag > 0.0) {
      const double dt = coefficient * dx_mag / du_eff;
      if (dt < params.av_cfl_diag_below) {
        printf("[av_cfl_diag] c=%d local=%d dt=%.3e dx=%.3e du=%.3e "
               "du_eff=%.3e s_mag=%.3e proj=%.3f psi=%.3f n0=%d "
               "n1=%d r0=%.6e z0=%.6e r1=%.6e z1=%.6e\n",
               c, local, dt, dx_mag, du_mag, du_eff, s_mag, proj, psi, n0,
               n1, x_r[n0], x_z[n0], x_r[n1], x_z[n1]);
      }
      atomic_min_double(min_dt, dt);
      if (winner_index != nullptr && edge_dt_matches(dt, target_dt)) {
        atomicMin(winner_index, candidate_index);
      }
    }
    return;
  }
  const int neighbor = face_adj_indices[face_adj_offsets[c] + local];
  const bool pole_floor_enabled = params.pole_floor_enabled != 0;
  const Csw98PoleLimiterSensor pole_sensor =
      csw98_multiblock_pole_limiter_sensor(
          x_r, x_z, v_r, v_z, hydro_active, cell_node_offsets,
          cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts,
          edge_psi, cell_edge_offsets, cell_edge_edges, line_edge_n0,
          line_edge_n1, c, neighbor, n0, n1,
          pole_floor_enabled ? params.pole_floor_theta0
                             : params.pole_desens_theta0,
          pole_floor_enabled ? params.pole_floor_thetaf
                             : params.pole_desens_thetaf,
          params, aw_planar, nullptr);
  const double s_side = fmax(0.0, 1.0 - psi_clamped);
  const double lim_scale = csw98_multiblock_pole_limiter_scale(
      s_side, pole_sensor, params);
  if (axis_line_edge) {
    if (params.axisline_d1prime_cfl != 0) {
      const int active_neighbor =
          cell_active(hydro_active, neighbor) ? neighbor : -1;
      double dt_cand = 0.0;
      if (csw98_axisline_d1prime_cfl_dt(
              &dt_cand, node_mass, rho, cs, n0, n1, c, active_neighbor,
              du_z, s_z, lim_scale, params)) {
        atomic_min_double(min_dt, dt_cand);
        if (winner_index != nullptr && edge_dt_matches(dt_cand, target_dt)) {
          atomicMin(winner_index, candidate_index);
        }
      }
    }
    return;
  }
  const double du_eff = du_mag * lim_scale * proj;
  if (du_eff > kTiny && isfinite(du_eff) && dx_mag > 0.0) {
    const double dt = coefficient * dx_mag / du_eff;
    if (dt < params.av_cfl_diag_below) {
      printf("[av_cfl_diag] c=%d local=%d dt=%.3e dx=%.3e du=%.3e "
             "du_eff=%.3e s_mag=%.3e proj=%.3f psi=%.3f n0=%d "
             "n1=%d r0=%.6e z0=%.6e r1=%.6e z1=%.6e\n",
             c, local, dt, dx_mag, du_mag, du_eff, s_mag, proj, psi, n0, n1,
             x_r[n0], x_z[n0], x_r[n1], x_z[n1]);
    }
    atomic_min_double(min_dt, dt);
    if (winner_index != nullptr && edge_dt_matches(dt, target_dt)) {
      atomicMin(winner_index, candidate_index);
    }
  }
}

__global__ void csw98_polar_slaving_raw_kernel(
    double* __restrict__ psi_raw,
    double* __restrict__ chi,
    std::uint8_t* __restrict__ active,
    std::uint8_t* __restrict__ valid,
    int* __restrict__ diagnostics,
    const int* __restrict__ instance_cell,
    const int* __restrict__ instance_local,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ edge_psi,
    const int* __restrict__ cell_edge_offsets,
    const int* __restrict__ cell_edge_edges,
    const int* __restrict__ line_edge_n0,
    const int* __restrict__ line_edge_n1,
    const int n_instances,
    const CswKernelParams params,
    const bool aw_planar,
    int* limiter_miss) {
  const int instance = blockIdx.x * blockDim.x + threadIdx.x;
  if (instance >= n_instances) {
    return;
  }
  psi_raw[instance] = 0.0;
  chi[instance] = 0.0;
  active[instance] = 0U;
  valid[instance] = 0U;

  const int c = instance_cell[instance];
  const int local = instance_local[instance];
  if (!cell_active(hydro_active, c)) {
    return;
  }
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {0};
  csr_cell_active_nodes(c, active_nverts, cell_node_offsets, cell_node_indices,
                        nodes);
  int corner0 = 0;
  int corner1 = 0;
  if (!local_face_corners(active_nverts, local, &corner0, &corner1)) {
    return;
  }
  const int n0 = nodes[corner0];
  const int n1 = nodes[corner1];
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const bool axis_line_edge =
      multiblock_aw_axis_line_edge(x_r, n0, n1, aw_planar, params);
  if (axis_line_edge && params.axisline_d1prime == 0) {
    return;
  }
  if (axis_line_edge) {
    du_r = 0.0;
  }
  const double dx_mag = hypot(dx_r, dx_z);
  const double du_mag = hypot(du_r, du_z);
  if (!(dx_mag > kTiny) || !(du_mag > kTiny)) {
    return;
  }
  if (multiblock_axis_touch_edge(x_r, n0, n1, aw_planar, params)) {
    return;
  }

  if (params.limiter_enabled != 0 && limiter_miss != nullptr) {
    atomicAdd(limiter_miss + 1, 1);
  }
  const int edge_id = csw98_multiblock_cell_edge_id(
      cell_edge_offsets, cell_edge_edges, line_edge_n0, line_edge_n1, c, n0,
      n1);
  const double psi =
      params.limiter_enabled != 0 ? edge_psi[edge_id] : 0.0;
  psi_raw[instance] = psi;
  valid[instance] = 1U;
  if (!isfinite(psi) || psi < 0.0 || psi > 1.0) {
    atomicAdd(diagnostics + 2, 1);
    return;
  }

  double cell_r[kCsw98MaxSideVecs];
  double cell_z[kCsw98MaxSideVecs];
  for (int k = 0; k < active_nverts; ++k) {
    cell_r[k] = x_r[nodes[k]];
    cell_z[k] = x_z[nodes[k]];
  }
  double s_r = 0.0;
  double s_z = 0.0;
  if (!csw98_c2_svec_for_side(cell_r, cell_z, active_nverts, corner0, corner1,
                              &s_r, &s_z, aw_planar)) {
    atomicAdd(diagnostics + 2, 1);
    return;
  }
  const double compression = du_r * s_r + du_z * s_z;
  const double s_mag = hypot(s_r, s_z);
  const int neighbor = face_adj_indices[face_adj_offsets[c] + local];
  const int active_neighbor =
      cell_active(hydro_active, neighbor) ? neighbor : -1;
  const double rho_e =
      harmonic_positive(rho[c],
                        (active_neighbor >= 0) ? rho[active_neighbor] : rho[c]);
  const double cs_e =
      harmonic_positive(cs[c],
                        (active_neighbor >= 0) ? cs[active_neighbor] : cs[c]);
  (void)rho_e;
  const double denom = cs_e * s_mag;
  if (!(denom > 0.0) || !isfinite(denom) || !isfinite(compression)) {
    atomicAdd(diagnostics + 2, 1);
    return;
  }
  const double chi_value = fmax(-compression, 0.0) / denom;
  if (!isfinite(chi_value)) {
    atomicAdd(diagnostics + 2, 1);
    return;
  }
  chi[instance] = chi_value;
  active[instance] =
      compression < -kPolarSlavingCompressionEpsilon * denom ? 1U : 0U;
}

__global__ void csw98_polar_slaving_cohort_kernel(
    double* __restrict__ cohort_gate,
    double* __restrict__ cohort_psi_star,
    const int* __restrict__ cohort_members,
    const double* __restrict__ psi_raw,
    const double* __restrict__ chi,
    const std::uint8_t* __restrict__ active,
    const int n_cohorts,
    const int k_core,
    const double chi_on,
    const double chi_full) {
  const int cohort = blockIdx.x * blockDim.x + threadIdx.x;
  if (cohort >= n_cohorts) {
    return;
  }
  double gate = 0.0;
  double psi_star = 1.0;
  int n_active = 0;
  const int base = cohort * k_core;
  for (int q = 0; q < k_core; ++q) {
    const int instance = cohort_members[base + q];
    if (instance < 0) {
      continue;
    }
    const double chi_member = chi[instance];
    double member_gate = 0.0;
    if (chi_member >= chi_full) {
      member_gate = 1.0;
    } else if (chi_member > chi_on) {
      const double xi = (chi_member - chi_on) / (chi_full - chi_on);
      member_gate = xi * xi * (3.0 - 2.0 * xi);
    }
    gate = fmax(gate, member_gate);
    if (active[instance] != 0U) {
      const double normalized = psi_raw[instance] == 0.0
                                    ? 0.0
                                    : psi_raw[instance];
      psi_star = normalized < psi_star ? normalized : psi_star;
      ++n_active;
    }
  }
  if (!(gate > 0.0) || n_active < kPolarSlavingMinActive) {
    gate = 0.0;
  }
  cohort_gate[cohort] = gate;
  cohort_psi_star[cohort] = psi_star;
}

__global__ void csw98_polar_slaving_diagnostics_kernel(
    int* __restrict__ diagnostics,
    const std::uint8_t* __restrict__ valid,
    const int n_instances,
    const int n_cohorts,
    const CswPolarSlavingDeviceView slaving) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int modified = 0;
  for (int instance = 0; instance < n_instances; ++instance) {
    double a_raw = 0.0;
    double a_eff = 0.0;
    csw98_polar_slaving_effective_psi(slaving, instance, &a_raw, &a_eff);
    if (valid[instance] != 0U && a_eff > a_raw) {
      ++modified;
    }
  }
  int cohorts = 0;
  for (int cohort = 0; cohort < n_cohorts; ++cohort) {
    if (slaving.cohort_gate[cohort] > 0.0) {
      ++cohorts;
    }
  }
  diagnostics[0] = modified;
  diagnostics[1] = cohorts;
}

__global__ void csw98_polar_slaving_stiffness_instance_kernel(
    double* __restrict__ instance_tangent,
    const int* __restrict__ instance_cell,
    const int* __restrict__ instance_local,
    const int* __restrict__ instance_face,
    const int* __restrict__ face_node0,
    const int* __restrict__ face_node1,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_instances,
    const CswKernelParams params,
    const bool aw_planar,
    const CswPolarSlavingDeviceView slaving) {
  const int instance = blockIdx.x * blockDim.x + threadIdx.x;
  if (instance >= n_instances) {
    return;
  }
  double* const tangent = instance_tangent + 4 * instance;
  tangent[0] = 0.0;
  tangent[1] = 0.0;
  tangent[2] = 0.0;
  tangent[3] = 0.0;

  const int cohort = slaving.instance_cohort[instance];
  if (!(slaving.cohort_gate[cohort] > 0.0)) {
    return;
  }
  const int c = instance_cell[instance];
  if (!cell_active(hydro_active, c)) {
    return;
  }
  const int local = instance_local[instance];
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {0};
  csr_cell_active_nodes(c, active_nverts, cell_node_offsets, cell_node_indices,
                        nodes);
  int corner0 = 0;
  int corner1 = 0;
  if (!local_face_corners(active_nverts, local, &corner0, &corner1)) {
    return;
  }

  const int face = instance_face[instance];
  const int canonical_node0 = face_node0[face];
  const int canonical_node1 = face_node1[face];
  const int local_node0 = nodes[corner0];
  const int local_node1 = nodes[corner1];
  if (!((local_node0 == canonical_node0 && local_node1 == canonical_node1) ||
        (local_node0 == canonical_node1 && local_node1 == canonical_node0))) {
    return;
  }

  double cell_r[kCsw98MaxSideVecs];
  double cell_z[kCsw98MaxSideVecs];
  for (int k = 0; k < active_nverts; ++k) {
    cell_r[k] = x_r[nodes[k]];
    cell_z[k] = x_z[nodes[k]];
  }
  double s_r = 0.0;
  double s_z = 0.0;
  if (!csw98_c2_svec_for_side(cell_r, cell_z, active_nverts, corner0, corner1,
                              &s_r, &s_z, aw_planar)) {
    return;
  }
  if (local_node0 != canonical_node0) {
    s_r = -s_r;
    s_z = -s_z;
  }

  const double du_r = v_r[canonical_node1] - v_r[canonical_node0];
  const double du_z = v_z[canonical_node1] - v_z[canonical_node0];
  const double compression = du_r * s_r + du_z * s_z;
  if (!(compression < 0.0) || !finite_device(compression)) {
    return;
  }
  const double du_mag = hypot(du_r, du_z);
  if (!finite_device(du_mag)) {
    return;
  }

  const int neighbor = face_adj_indices[face_adj_offsets[c] + local];
  const int active_neighbor =
      cell_active(hydro_active, neighbor) ? neighbor : -1;
  const double rho_e =
      harmonic_positive(rho[c],
                        (active_neighbor >= 0) ? rho[active_neighbor] : rho[c]);
  const double cs_e =
      harmonic_positive(cs[c],
                        (active_neighbor >= 0) ? cs[active_neighbor] : cs[c]);
  if (!(rho_e > 0.0) || !(cs_e >= 0.0)) {
    return;
  }

  double a_eff = 0.0;
  csw98_polar_slaving_effective_psi(slaving, instance, nullptr, &a_eff);
  if (!(a_eff > 0.0)) {
    return;
  }
  const double alpha = params.c2 * (params.gamma + 1.0) * 0.25;
  const double beta = params.c1 * cs_e;
  if (du_mag <= kTiny) {
    const double bound =
        2.0 * rho_e * a_eff * beta * hypot(s_r, s_z);
    tangent[0] = bound;
    tangent[3] = bound;
    return;
  }

  const double n_r = du_r / du_mag;
  const double n_z = du_z / du_mag;
  const double alpha_s = alpha * du_mag;
  const double radical = sqrt(alpha_s * alpha_s + beta * beta);
  const double g = alpha_s + radical;
  const double g_prime =
      radical > 0.0 ? alpha + (alpha * alpha * du_mag) / radical : 0.0;
  const double q = -compression;
  const double transverse = g * q / du_mag;
  const double scale = rho_e * a_eff;
  tangent[0] = scale *
               (g_prime * q * n_r * n_r - g * n_r * s_r +
                transverse * (1.0 - n_r * n_r));
  tangent[1] = scale *
               (g_prime * q * n_r * n_z - g * n_r * s_z -
                transverse * n_r * n_z);
  tangent[2] = scale *
               (g_prime * q * n_z * n_r - g * n_z * s_r -
                transverse * n_z * n_r);
  tangent[3] = scale *
               (g_prime * q * n_z * n_z - g * n_z * s_z +
                transverse * (1.0 - n_z * n_z));
}

__global__ void csw98_polar_slaving_stiffness_face_kernel(
    double* __restrict__ face_kappa,
    const double* __restrict__ instance_tangent,
    const int* __restrict__ face_instance0,
    const int* __restrict__ face_instance1,
    const int n_faces) {
  const int face = blockIdx.x * blockDim.x + threadIdx.x;
  if (face >= n_faces) {
    return;
  }
  const int instance0 = face_instance0[face];
  const int instance1 = face_instance1[face];
  const double* const d0 = instance_tangent + 4 * instance0;
  const double d1_0 = instance1 >= 0 ? instance_tangent[4 * instance1] : 0.0;
  const double d1_1 =
      instance1 >= 0 ? instance_tangent[4 * instance1 + 1] : 0.0;
  const double d1_2 =
      instance1 >= 0 ? instance_tangent[4 * instance1 + 2] : 0.0;
  const double d1_3 =
      instance1 >= 0 ? instance_tangent[4 * instance1 + 3] : 0.0;
  const double a = d0[0] + d1_0;
  const double b = d0[1] + d1_1;
  const double c = d0[2] + d1_2;
  const double d = d0[3] + d1_3;
  const double tau = a * a + b * b + c * c + d * d;
  const double det = a * d - b * c;
  const double discriminant = fmax(tau * tau - 4.0 * det * det, 0.0);
  face_kappa[face] = sqrt(0.5 * (tau + sqrt(discriminant)));
}

__global__ void csw98_polar_slaving_stiffness_node_kernel(
    double* __restrict__ node_lambda,
    int* __restrict__ failure,
    const int* __restrict__ node_ids,
    const int* __restrict__ node_face_offsets,
    const int* __restrict__ node_face_indices,
    const double* __restrict__ face_kappa,
    const double* __restrict__ node_planar_mass,
    const int n_nodes) {
  const int compact_node = blockIdx.x * blockDim.x + threadIdx.x;
  if (compact_node >= n_nodes) {
    return;
  }
  double sum = 0.0;
  for (int p = node_face_offsets[compact_node];
       p < node_face_offsets[compact_node + 1]; ++p) {
    const double kappa = face_kappa[node_face_indices[p]];
    if (!isfinite(kappa)) {
      atomicExch(failure, 1);
      node_lambda[compact_node] = 0.0;
      return;
    }
    sum += kappa;
  }
  const double mass = node_planar_mass[node_ids[compact_node]];
  if (!(mass > 0.0) || !isfinite(sum)) {
    atomicExch(failure, 1);
    node_lambda[compact_node] = 0.0;
    return;
  }
  const double lambda = 2.0 * sum / fmax(mass, kTiny);
  if (!isfinite(lambda)) {
    atomicExch(failure, 1);
    node_lambda[compact_node] = 0.0;
    return;
  }
  node_lambda[compact_node] = lambda;
}

__global__ void csw98_polar_slaving_stiffness_max_kernel(
    double* __restrict__ max_lambda,
    int* __restrict__ winner_node,
    int* __restrict__ failure,
    const double* __restrict__ node_lambda,
    const int* __restrict__ node_ids,
    const int n_nodes) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  double maximum = 0.0;
  int winner = -1;
  for (int compact_node = 0; compact_node < n_nodes; ++compact_node) {
    const double lambda = node_lambda[compact_node];
    if (!isfinite(lambda)) {
      *failure = 1;
      continue;
    }
    const int node = node_ids[compact_node];
    if (lambda > maximum ||
        (lambda == maximum && lambda > 0.0 &&
         (winner < 0 || node < winner))) {
      maximum = lambda;
      winner = node;
    }
  }
  *max_lambda = maximum;
  *winner_node = winner;
}

__device__ inline void csw98_multiblock_side_cfl_polar_slaving(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ edge_psi,
    const int edge_id,
    const int c,
    const int local,
    const CswKernelParams params,
    const double coefficient,
    const bool aw_planar,
    const CswPolarSlavingDeviceView slaving,
    int* __restrict__ winner_index,
    const int candidate_index,
    const double target_dt) {
  if (!cell_active(hydro_active, c)) {
    return;
  }

  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {0};
  csr_cell_active_nodes(c, active_nverts, cell_node_offsets, cell_node_indices,
                        nodes);
  int corner0 = 0;
  int corner1 = 0;
  if (!local_face_corners(active_nverts, local, &corner0, &corner1)) {
    return;
  }
  const int n0 = nodes[corner0];
  const int n1 = nodes[corner1];
  const bool axis_line_edge =
      multiblock_aw_axis_line_edge(x_r, n0, n1, aw_planar, params);
  if (axis_line_edge && params.axisline_d1prime == 0) {
    return;
  }
  if (multiblock_axis_touch_edge(x_r, n0, n1, aw_planar, params)) {
    return;
  }
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  if (axis_line_edge) {
    du_r = 0.0;
  }
  const double dx_mag = hypot(dx_r, dx_z);
  const double du_mag = hypot(du_r, du_z);
  if (!(dx_mag > kTiny) || !(du_mag > kTiny)) {
    return;
  }

  double cell_r[kCsw98MaxSideVecs];
  double cell_z[kCsw98MaxSideVecs];
  for (int k = 0; k < active_nverts; ++k) {
    cell_r[k] = x_r[nodes[k]];
    cell_z[k] = x_z[nodes[k]];
  }
  double s_r = 0.0;
  double s_z = 0.0;
  if (!csw98_c2_svec_for_side(cell_r, cell_z, active_nverts, corner0, corner1,
                              &s_r, &s_z, aw_planar)) {
    return;
  }
  const double compression = du_r * s_r + du_z * s_z;
  if (!(compression < 0.0) || !finite_device(compression)) {
    return;
  }

  const int lookup = face_adj_offsets[c] + local;
  const int instance = slaving.instance_lookup[lookup];
  double psi = 0.0;
  if (instance >= 0) {
    psi = csw98_polar_slaving_effective_psi(slaving, instance);
  } else {
    psi = params.limiter_enabled != 0 ? edge_psi[edge_id] : 0.0;
  }
  const double s_mag = hypot(s_r, s_z);
  if (!(s_mag > 0.0)) {
    return;
  }
  const double proj = fabs(compression) / (du_mag * s_mag);
  const double psi_clamped = fmin(1.0, fmax(0.0, psi));
  if (axis_line_edge) {
    if (params.axisline_d1prime_cfl != 0) {
      const int neighbor = face_adj_indices[face_adj_offsets[c] + local];
      const int active_neighbor =
          cell_active(hydro_active, neighbor) ? neighbor : -1;
      double dt_cand = 0.0;
      if (csw98_axisline_d1prime_cfl_dt(
              &dt_cand, node_mass, rho, cs, n0, n1, c, active_neighbor,
              du_z, s_z, fmax(0.0, 1.0 - psi_clamped), params)) {
        atomic_min_double(min_dt, dt_cand);
        if (winner_index != nullptr && edge_dt_matches(dt_cand, target_dt)) {
          atomicMin(winner_index, candidate_index);
        }
      }
    }
    return;
  }
  const double du_eff = du_mag * fmax(0.0, 1.0 - psi_clamped) * proj;
  if (du_eff > kTiny && isfinite(du_eff) && dx_mag > 0.0) {
    const double dt = coefficient * dx_mag / du_eff;
    if (dt < params.av_cfl_diag_below) {
      printf("[av_cfl_diag] c=%d local=%d dt=%.3e dx=%.3e du=%.3e "
             "du_eff=%.3e s_mag=%.3e proj=%.3f psi=%.3f n0=%d "
             "n1=%d r0=%.6e z0=%.6e r1=%.6e z1=%.6e\n",
             c, local, dt, dx_mag, du_mag, du_eff, s_mag, proj, psi, n0, n1,
             x_r[n0], x_z[n0], x_r[n1], x_z[n1]);
    }
    atomic_min_double(min_dt, dt);
    if (winner_index != nullptr && edge_dt_matches(dt, target_dt)) {
      atomicMin(winner_index, candidate_index);
    }
  }
}

__global__ void csw98_structured_force_kernel(
    double* __restrict__ edge_force_r,
    double* __restrict__ edge_force_z,
    double* __restrict__ work_av,
    int* __restrict__ fire_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const CswKernelParams params,
    const bool aw_planar,
    const bool aw_axisline_av_enabled,
    const int aw_axis_slave_first_i,
    const bool aw_axis_slave_theta0_active,
    const bool aw_axis_slave_theta_pi_active) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_radial = nr * (nz + 1);
  const int n_edges = n_radial + (nr + 1) * nz;
  if (e >= n_edges) {
    return;
  }
  const bool axis_line_edge = structured_aw_axis_line_edge(
      e, n_radial, nz, aw_axis_slave_first_i, aw_planar,
      aw_axisline_av_enabled, aw_axis_slave_theta0_active,
      aw_axis_slave_theta_pi_active);
  if (axis_line_edge && params.axisline_d1prime == 0) {
    return;
  }

  double fr = 0.0;
  double fz = 0.0;
  if (e < n_radial) {
    const int i = e / (nz + 1);
    const int j = e - i * (nz + 1);
    if (j > 0) {
      csw98_structured_cell_face_contribution(
          work_av, fire_count, negative_work_count, x_r, x_z, v_r, v_z,
          node_mass, rho, cs, hydro_active, i * nz + (j - 1), 3, nr, nz,
          params, aw_planar, aw_axis_slave_theta0_active,
          aw_axis_slave_theta_pi_active, axis_line_edge, &fr, &fz);
    }
    if (j < nz) {
      csw98_structured_cell_face_contribution(
          work_av, fire_count, negative_work_count, x_r, x_z, v_r, v_z,
          node_mass, rho, cs, hydro_active, i * nz + j, 2, nr, nz, params,
          aw_planar, aw_axis_slave_theta0_active,
          aw_axis_slave_theta_pi_active, axis_line_edge, &fr, &fz);
    }
  } else {
    const int a = e - n_radial;
    const int i = a / nz;
    const int j = a - i * nz;
    if (i > 0) {
      csw98_structured_cell_face_contribution(
          work_av, fire_count, negative_work_count, x_r, x_z, v_r, v_z,
          node_mass, rho, cs, hydro_active, (i - 1) * nz + j, 1, nr, nz,
          params, aw_planar, aw_axis_slave_theta0_active,
          aw_axis_slave_theta_pi_active, axis_line_edge, &fr, &fz);
    }
    if (i < nr) {
      csw98_structured_cell_face_contribution(
          work_av, fire_count, negative_work_count, x_r, x_z, v_r, v_z,
          node_mass, rho, cs, hydro_active, i * nz + j, 0, nr, nz, params,
          aw_planar, aw_axis_slave_theta0_active,
          aw_axis_slave_theta_pi_active, axis_line_edge, &fr, &fz);
    }
  }
  edge_force_r[e] = axis_line_edge ? 0.0 : fr;
  edge_force_z[e] = fz;
}

__global__ void csw98_multiblock_internal_force_kernel(
    double* __restrict__ edge_force_r,
    double* __restrict__ edge_force_z,
    double* __restrict__ work_av_edge,
    int* __restrict__ fire_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ face_local_b,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ edge_psi,
    const int* __restrict__ cell_edge_offsets,
    const int* __restrict__ cell_edge_edges,
    const int* __restrict__ line_edge_n0,
    const int* __restrict__ line_edge_n1,
    const int n_faces,
    const CswKernelParams params,
    const bool aw_planar,
    int* __restrict__ limiter_miss_count) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  double fr = 0.0;
  double fz = 0.0;
  double work_a = 0.0;
  double work_b = 0.0;
  csw98_multiblock_cell_face_contribution(
      &work_a, fire_count, negative_work_count, x_r, x_z, v_r, v_z,
      node_mass, rho, cs, hydro_active, cell_node_offsets, cell_node_indices,
      face_adj_offsets, face_adj_indices, cell_nverts, edge_psi,
      cell_edge_offsets, cell_edge_edges, line_edge_n0, line_edge_n1,
      face_cell_a[f], face_local_a[f], params, aw_planar, &fr, &fz,
      limiter_miss_count);
  csw98_multiblock_cell_face_contribution(
      &work_b, fire_count, negative_work_count, x_r, x_z, v_r, v_z,
      node_mass, rho, cs, hydro_active, cell_node_offsets, cell_node_indices,
      face_adj_offsets, face_adj_indices, cell_nverts, edge_psi,
      cell_edge_offsets, cell_edge_edges, line_edge_n0, line_edge_n1,
      face_cell_b[f], face_local_b[f], params, aw_planar, &fr, &fz,
      limiter_miss_count);
  work_av_edge[2 * f] += work_a;
  work_av_edge[2 * f + 1] += work_b;
  edge_force_r[f] = fr;
  edge_force_z[f] = fz;
}

__global__ void csw98_multiblock_internal_force_polar_slaving_kernel(
    double* __restrict__ edge_force_r,
    double* __restrict__ edge_force_z,
    double* __restrict__ work_av_edge,
    int* __restrict__ fire_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ face_local_b,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ edge_psi,
    const int* __restrict__ cell_edge_offsets,
    const int* __restrict__ cell_edge_edges,
    const int* __restrict__ line_edge_n0,
    const int* __restrict__ line_edge_n1,
    const int n_faces,
    const CswKernelParams params,
    const bool aw_planar,
    const CswPolarSlavingDeviceView slaving,
    int* __restrict__ limiter_miss_count) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  double fr = 0.0;
  double fz = 0.0;
  double work_a = 0.0;
  double work_b = 0.0;
  csw98_multiblock_cell_face_contribution_polar_slaving(
      &work_a, fire_count, negative_work_count, x_r, x_z, v_r, v_z,
      node_mass, rho, cs, hydro_active, cell_node_offsets, cell_node_indices,
      face_adj_offsets, face_adj_indices, cell_nverts, edge_psi,
      cell_edge_offsets, cell_edge_edges, line_edge_n0, line_edge_n1,
      face_cell_a[f], face_local_a[f], params, aw_planar, slaving, &fr, &fz,
      limiter_miss_count);
  csw98_multiblock_cell_face_contribution_polar_slaving(
      &work_b, fire_count, negative_work_count, x_r, x_z, v_r, v_z,
      node_mass, rho, cs, hydro_active, cell_node_offsets, cell_node_indices,
      face_adj_offsets, face_adj_indices, cell_nverts, edge_psi,
      cell_edge_offsets, cell_edge_edges, line_edge_n0, line_edge_n1,
      face_cell_b[f], face_local_b[f], params, aw_planar, slaving, &fr, &fz,
      limiter_miss_count);
  work_av_edge[2 * f] += work_a;
  work_av_edge[2 * f + 1] += work_b;
  edge_force_r[f] = fr;
  edge_force_z[f] = fz;
}

__global__ void csw98_multiblock_boundary_force_kernel(
    double* __restrict__ edge_force_r,
    double* __restrict__ edge_force_z,
    double* __restrict__ work_av_edge,
    int* __restrict__ fire_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell,
    const int* __restrict__ face_local,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ edge_psi,
    const int* __restrict__ cell_edge_offsets,
    const int* __restrict__ cell_edge_edges,
    const int* __restrict__ line_edge_n0,
    const int* __restrict__ line_edge_n1,
    const int edge_offset,
    const int n_faces,
    const CswKernelParams params,
    const bool aw_planar,
    int* __restrict__ limiter_miss_count) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  double fr = 0.0;
  double fz = 0.0;
  double work = 0.0;
  const int e = edge_offset + f;
  csw98_multiblock_cell_face_contribution(
      &work, fire_count, negative_work_count, x_r, x_z, v_r, v_z, node_mass,
      rho, cs, hydro_active, cell_node_offsets, cell_node_indices,
      face_adj_offsets, face_adj_indices, cell_nverts, edge_psi,
      cell_edge_offsets, cell_edge_edges, line_edge_n0, line_edge_n1,
      face_cell[f], face_local[f], params, aw_planar, &fr, &fz,
      limiter_miss_count);
  work_av_edge[2 * e] += work;
  edge_force_r[e] = fr;
  edge_force_z[e] = fz;
}

__global__ void csw_structured_force_kernel(
    double* __restrict__ edge_force_r,
    double* __restrict__ edge_force_z,
    double* __restrict__ work_av,
    int* __restrict__ compressive_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const CswKernelParams params,
    const bool aw_planar,
    const bool aw_axisline_av_enabled,
    const int aw_axis_slave_first_i,
    const bool aw_axis_slave_theta0_active,
    const bool aw_axis_slave_theta_pi_active) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_radial = nr * (nz + 1);
  const int n_edges = n_radial + (nr + 1) * nz;
  if (e >= n_edges) {
    return;
  }
  if (structured_aw_axis_line_edge(
          e, n_radial, nz, aw_axis_slave_first_i, aw_planar,
          aw_axisline_av_enabled,
          aw_axis_slave_theta0_active,
          aw_axis_slave_theta_pi_active)) {
    return;
  }

  double fr = 0.0;
  double fz = 0.0;
  if (e < n_radial) {
    const int i = e / (nz + 1);
    const int j = e - i * (nz + 1);
    if (j > 0) {
      structured_cell_face_contribution(
          work_av, compressive_count, negative_work_count, x_r, x_z, v_r, v_z,
          rho, cs, hydro_active, i * nz + (j - 1), 3, nr, nz, params,
          aw_planar, &fr, &fz);
    }
    if (j < nz) {
      structured_cell_face_contribution(
          work_av, compressive_count, negative_work_count, x_r, x_z, v_r, v_z,
          rho, cs, hydro_active, i * nz + j, 2, nr, nz, params, aw_planar,
          &fr, &fz);
    }
  } else {
    const int a = e - n_radial;
    const int i = a / nz;
    const int j = a - i * nz;
    if (i > 0) {
      structured_cell_face_contribution(
          work_av, compressive_count, negative_work_count, x_r, x_z, v_r, v_z,
          rho, cs, hydro_active, (i - 1) * nz + j, 1, nr, nz, params,
          aw_planar, &fr, &fz);
    }
    if (i < nr) {
      structured_cell_face_contribution(
          work_av, compressive_count, negative_work_count, x_r, x_z, v_r, v_z,
          rho, cs, hydro_active, i * nz + j, 0, nr, nz, params, aw_planar,
          &fr, &fz);
    }
  }
  edge_force_r[e] = fr;
  edge_force_z[e] = fz;
}

__global__ void csw_multiblock_internal_force_kernel(
    double* __restrict__ edge_force_r,
    double* __restrict__ edge_force_z,
    double* __restrict__ work_av_edge,
    int* __restrict__ compressive_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ face_local_b,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_faces,
    const CswKernelParams params,
    const bool aw_planar) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  double fr = 0.0;
  double fz = 0.0;
  double work_a = 0.0;
  double work_b = 0.0;
  multiblock_cell_face_contribution(
      &work_a, compressive_count, negative_work_count, x_r, x_z, v_r, v_z,
      rho, cs, hydro_active, cell_node_offsets, cell_node_indices,
      face_adj_offsets, face_adj_indices, cell_nverts, face_cell_a[f],
      face_local_a[f], params, aw_planar, &fr, &fz);
  multiblock_cell_face_contribution(
      &work_b, compressive_count, negative_work_count, x_r, x_z, v_r, v_z,
      rho, cs, hydro_active, cell_node_offsets, cell_node_indices,
      face_adj_offsets, face_adj_indices, cell_nverts, face_cell_b[f],
      face_local_b[f], params, aw_planar, &fr, &fz);
  work_av_edge[2 * f] += work_a;
  work_av_edge[2 * f + 1] += work_b;
  edge_force_r[f] = fr;
  edge_force_z[f] = fz;
}

__global__ void csw_multiblock_boundary_force_kernel(
    double* __restrict__ edge_force_r,
    double* __restrict__ edge_force_z,
    double* __restrict__ work_av_edge,
    int* __restrict__ compressive_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell,
    const int* __restrict__ face_local,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int edge_offset,
    const int n_faces,
    const CswKernelParams params,
    const bool aw_planar) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  double fr = 0.0;
  double fz = 0.0;
  double work = 0.0;
  multiblock_cell_face_contribution(
      &work, compressive_count, negative_work_count, x_r, x_z, v_r, v_z, rho,
      cs, hydro_active, cell_node_offsets, cell_node_indices, face_adj_offsets,
      face_adj_indices, cell_nverts, face_cell[f], face_local[f], params,
      aw_planar, &fr, &fz);
  const int e = edge_offset + f;
  work_av_edge[2 * e] += work;
  edge_force_r[e] = fr;
  edge_force_z[e] = fz;
}

__global__ void gather_multiblock_work_av_kernel(
    double* __restrict__ work_av,
    const double* __restrict__ work_av_edge,
    const int* __restrict__ cell_edge_offsets,
    const int* __restrict__ cell_edge_edges,
    const std::int8_t* __restrict__ cell_edge_side,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  double work = 0.0;
  for (int k = cell_edge_offsets[c]; k < cell_edge_offsets[c + 1]; ++k) {
    const int e = cell_edge_edges[k];
    const int side = static_cast<int>(cell_edge_side[k]);
    work += work_av_edge[2 * e + side];
  }
  work_av[c] += work;
}

__global__ void clip_negative_work_kernel(double* __restrict__ work_av,
                                          int* __restrict__ negative_count,
                                          const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const double w = work_av[c];
  if (!(w >= 0.0) || !isfinite(w)) {
    work_av[c] = 0.0;
    atomicAdd(negative_count, 1);
  }
}

// I1-B pole tangential damper (pole-shear verdict step 7, experimental,
// env-gated default-off). The pole-column angular-shear mode is a radially
// alternating tangential slip of adjacent polar-shell node rows in the same
// near-pole angular column; it is AV-blind (div(u) ~ 0 along the slip).
// This damper applies, on every radial pole-column node pair (q,k)-(q+1,k)
// with column a in [1, M) of either pole, the pairwise tangential drag
//   F = -C_theta * rho_e * cs_e * |S_edge| * (du . t_hat) t_hat
// (du = v[n1]-v[n0], t_hat = in-plane unit normal of the edge, |S_edge| the
// same RZ edge area convention as the CSW AV svec). The force is accumulated
// into the edge AV force buffers in the canonical cell_a orientation, so the
// momentum kick (-F at n0, +F at n1 via sum_edge_forces) and the corrector's
// time-centered SIGNED work recompute inherit the exact compatible
// total-energy closure unchanged. Momentum is conserved pairwise by
// construction. Explicit-stability margin is ~1/C_theta acoustic steps
// (no extra dt limiter at the small default C_theta).
bool pole_damper_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_DAMPER");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  return enabled;
}

double pole_damper_ctheta() {
  static const double c_theta = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_DAMPER_CTHETA");
    const double v = raw != nullptr ? std::atof(raw) : 0.0;
    return (std::isfinite(v) && v > 0.0 && v <= 10.0) ? v : 0.05;
  }();
  return c_theta;
}

int pole_damper_m() {
  static const int m = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_DAMPER_M");
    const int v = raw != nullptr ? std::atoi(raw) : 0;
    return v > 1 ? v : 4;
  }();
  return m;
}

// Optional node-row band gate [q_lo, q_hi] for the damped pairs (both pair
// rows must lie inside). Default unrestricted. Motivation (v1 empirical):
// all-rows damping at C_theta=0.05 left the dense-band sawtooth essentially
// unchanged while its dissipated heat, deposited half/half into the
// near-pole LOW-MASS GAS sliver cells, produced a ~50 keV hot cell whose
// axis-margin dt collapse killed the run EARLIER (2.51 ns) than the no-op
// baseline (2.864 ns). A dense-shell-only band deposits the slip heat into
// rho ~ 0.25 cells where it is thermally negligible.
int pole_damper_q_lo() {
  static const int q_lo = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_DAMPER_Q_LO");
    const int v = raw != nullptr ? std::atoi(raw) : 0;
    return v > 0 ? v : 0;
  }();
  return q_lo;
}

int pole_damper_q_hi() {
  static const int q_hi = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_DAMPER_Q_HI");
    const int v = raw != nullptr ? std::atoi(raw) : 0;
    return v > 0 ? v : (1 << 28);
  }();
  return q_hi;
}

__global__ void pole_damper_internal_force_kernel(
    double* __restrict__ edge_force_r,
    double* __restrict__ edge_force_z,
    double* __restrict__ work_av_edge,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_faces,
    const int shell_node_begin,
    const int ntheta,
    const int n_node_rows,
    const int q_begin_active,
    const int m_columns,
    const int q_lo,
    const int q_hi,
    const double c_theta,
    int* __restrict__ damped_pair_count) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int cell_a = face_cell_a[f];
  const int cell_b = face_cell_b[f];
  // Canonical endpoint orientation: cell_a's corner pair, identical to
  // sum_edge_forces_multiblock_kernel / av_work_multiblock_kernel. The pair
  // formula is orientation-covariant, so consistency with the downstream
  // kernels is the only requirement.
  const int active_nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, cell_a);
  int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {0};
  csr_cell_active_nodes(cell_a, active_nverts, cell_node_offsets,
                        cell_node_indices, nodes);
  int corner0 = 0;
  int corner1 = 0;
  if (!local_face_corners(active_nverts, face_local_a[f], &corner0, &corner1)) {
    return;
  }
  const int n0 = nodes[corner0];
  const int n1 = nodes[corner1];
  // Shell membership + same angular column + radially adjacent node rows.
  const int npc = ntheta + 1;
  const int s0 = n0 - shell_node_begin;
  const int s1 = n1 - shell_node_begin;
  if (s0 < 0 || s1 < 0) {
    return;
  }
  const int q0 = s0 / npc;
  const int k0 = s0 - q0 * npc;
  const int q1 = s1 / npc;
  const int k1 = s1 - q1 * npc;
  if (q0 >= n_node_rows || q1 >= n_node_rows || k0 != k1) {
    return;
  }
  const int dq = q1 - q0;
  if (dq != 1 && dq != -1) {
    return;
  }
  // Rows interior to the central macro cell are virtual; never touch them.
  if (q0 < q_begin_active || q1 < q_begin_active) {
    return;
  }
  // Optional row-band gate: both pair rows inside [q_lo, q_hi].
  const int q_pair_min = q0 < q1 ? q0 : q1;
  const int q_pair_max = q0 < q1 ? q1 : q0;
  if (q_pair_min < q_lo || q_pair_max > q_hi) {
    return;
  }
  // Near-pole columns a in [1, m_columns): north a = k, south a = ntheta - k.
  // a = 0 is the axis column itself (PAVA/axis-constrained), excluded.
  const int a_north = k0;
  const int a_south = ntheta - k0;
  const int a = a_north < a_south ? a_north : a_south;
  if (a < 1 || a >= m_columns) {
    return;
  }
  if (!cell_active(hydro_active, cell_a) || !cell_active(hydro_active, cell_b)) {
    return;
  }
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  const double h = hypot(dx_r, dx_z);
  if (!(h > kTiny)) {
    return;
  }
  const double that_r = -dx_z / h;
  const double that_z = dx_r / h;
  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const double du_t = du_r * that_r + du_z * that_z;
  if (!finite_device(du_t) || du_t == 0.0) {
    return;
  }
  const double rho_e = harmonic_positive(rho[cell_a], rho[cell_b]);
  const double cs_e = harmonic_positive(cs[cell_a], cs[cell_b]);
  // Same RZ edge-area convention as edge_svec_from_cell: |S| = pi*(r0+r1)*h.
  const double area = kPi * (x_r[n0] + x_r[n1]) * h;
  const double coef = c_theta * rho_e * cs_e * area;
  if (!(coef > 0.0) || !finite_device(coef)) {
    return;
  }
  const double f_t = -coef * du_t;
  edge_force_r[f] += f_t * that_r;
  edge_force_z[f] += f_t * that_z;
  // Assembly-stage work estimate, same convention as apply_csw_contribution
  // (w = -F.du >= 0 here by construction); the corrector recomputes the
  // exact signed time-centered work from the summed edge forces.
  const double w = coef * du_t * du_t;
  if (finite_device(w) && w > 0.0) {
    work_av_edge[2 * f] += 0.5 * w;
    work_av_edge[2 * f + 1] += 0.5 * w;
  }
  if (damped_pair_count != nullptr) {
    atomicAdd(damped_pair_count, 1);
  }
}

__device__ inline bool structured_edge_is_compressive(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int c,
    const int local,
    const int nz) {
  if (!cell_active(hydro_active, c)) {
    return false;
  }
  int nodes[4] = {0, 0, 0, 0};
  structured_cell_nodes(c, nz, nodes);
  int c0 = 0;
  int c1 = 0;
  local_face_corners(local, &c0, &c1);
  const int n0 = nodes[c0];
  const int n1 = nodes[c1];
  double s_r = 0.0;
  double s_z = 0.0;
  edge_svec_from_cell(x_r, x_z, nodes, local, &s_r, &s_z);
  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  return hypot(du_r, du_z) > kTiny && (du_r * s_r + du_z * s_z) <= 0.0;
}

__device__ inline bool multiblock_edge_is_compressive(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int c,
    const int local) {
  if (!cell_active(hydro_active, c)) {
    return false;
  }
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {0};
  csr_cell_active_nodes(c, active_nverts, cell_node_offsets, cell_node_indices,
                        nodes);
  int c0 = 0;
  int c1 = 0;
  if (!local_face_corners(active_nverts, local, &c0, &c1)) {
    return false;
  }
  const int n0 = nodes[c0];
  const int n1 = nodes[c1];
  double s_r = 0.0;
  double s_z = 0.0;
  if (active_nverts == 3) {
    if (!edge_svec_from_cell(x_r, x_z, nodes, active_nverts, local, &s_r,
                             &s_z)) {
      return false;
    }
  } else {
    edge_svec_from_cell(x_r, x_z, nodes, local, &s_r, &s_z);
  }
  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  return hypot(du_r, du_z) > kTiny && (du_r * s_r + du_z * s_z) <= 0.0;
}

__global__ void csw_structured_cfl_kernel(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const double coefficient,
    const bool aw_planar,
    const bool aw_axisline_av_enabled,
    const int aw_axis_slave_first_i,
    const bool aw_axis_slave_theta0_active,
    const bool aw_axis_slave_theta_pi_active) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_radial = nr * (nz + 1);
  const int n_edges = n_radial + (nr + 1) * nz;
  if (e >= n_edges) {
    return;
  }
  if (structured_aw_axis_line_edge(
          e, n_radial, nz, aw_axis_slave_first_i, aw_planar,
          aw_axisline_av_enabled,
          aw_axis_slave_theta0_active,
          aw_axis_slave_theta_pi_active)) {
    return;
  }

  int n0 = -1;
  int n1 = -1;
  bool compressive = false;
  if (e < n_radial) {
    const int i = e / (nz + 1);
    const int j = e - i * (nz + 1);
    n0 = structured_node_index(i, j, nz);
    n1 = structured_node_index(i + 1, j, nz);
    if (j > 0) {
      compressive = compressive ||
                    structured_edge_is_compressive(
                        x_r, x_z, v_r, v_z, hydro_active, i * nz + (j - 1), 3,
                        nz);
    }
    if (j < nz) {
      compressive = compressive ||
                    structured_edge_is_compressive(x_r, x_z, v_r, v_z,
                                                   hydro_active, i * nz + j, 2,
                                                   nz);
    }
  } else {
    const int a = e - n_radial;
    const int i = a / nz;
    const int j = a - i * nz;
    n0 = structured_node_index(i, j, nz);
    n1 = structured_node_index(i, j + 1, nz);
    if (i > 0) {
      compressive = compressive ||
                    structured_edge_is_compressive(
                        x_r, x_z, v_r, v_z, hydro_active, (i - 1) * nz + j, 1,
                        nz);
    }
    if (i < nr) {
      compressive = compressive ||
                    structured_edge_is_compressive(x_r, x_z, v_r, v_z,
                                                   hydro_active, i * nz + j, 0,
                                                   nz);
    }
  }
  if (!compressive) {
    return;
  }
  const double dx = hypot(x_r[n1] - x_r[n0], x_z[n1] - x_z[n0]);
  const double du = hypot(v_r[n1] - v_r[n0], v_z[n1] - v_z[n0]);
  if (dx > 0.0 && du > kTiny && isfinite(dx) && isfinite(du)) {
    atomic_min_double(min_dt, coefficient * dx / du);
  }
}

__global__ void csw98_structured_cfl_kernel(
    double* __restrict__ min_dt,
    int* __restrict__ winner_index,
    const double target_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const CswKernelParams params,
    const double coefficient,
    const bool aw_planar,
    const bool aw_axisline_av_enabled,
    const int aw_axis_slave_first_i,
    const bool aw_axis_slave_theta0_active,
    const bool aw_axis_slave_theta_pi_active) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_radial = nr * (nz + 1);
  const int n_edges = n_radial + (nr + 1) * nz;
  if (e >= n_edges) {
    return;
  }
  const bool axis_line_edge = structured_aw_axis_line_edge(
      e, n_radial, nz, aw_axis_slave_first_i, aw_planar,
      aw_axisline_av_enabled, aw_axis_slave_theta0_active,
      aw_axis_slave_theta_pi_active);
  if (axis_line_edge && params.axisline_d1prime == 0) {
    return;
  }

  if (e < n_radial) {
    const int i = e / (nz + 1);
    const int j = e - i * (nz + 1);
    if (j > 0) {
      csw98_structured_side_cfl(min_dt, x_r, x_z, v_r, v_z, node_mass, rho,
                                cs, hydro_active, i * nz + (j - 1), 3, nr, nz,
                                params, coefficient, aw_planar,
                                aw_axis_slave_theta0_active,
                                aw_axis_slave_theta_pi_active, axis_line_edge,
                                winner_index, 2 * e, target_dt);
    }
    if (j < nz) {
      csw98_structured_side_cfl(min_dt, x_r, x_z, v_r, v_z, node_mass, rho,
                                cs, hydro_active, i * nz + j, 2, nr, nz,
                                params, coefficient, aw_planar,
                                aw_axis_slave_theta0_active,
                                aw_axis_slave_theta_pi_active, axis_line_edge,
                                winner_index, 2 * e + 1, target_dt);
    }
  } else {
    const int a = e - n_radial;
    const int i = a / nz;
    const int j = a - i * nz;
    if (i > 0) {
      csw98_structured_side_cfl(min_dt, x_r, x_z, v_r, v_z, node_mass, rho,
                                cs, hydro_active, (i - 1) * nz + j, 1, nr, nz,
                                params, coefficient, aw_planar,
                                aw_axis_slave_theta0_active,
                                aw_axis_slave_theta_pi_active, axis_line_edge,
                                winner_index, 2 * e, target_dt);
    }
    if (i < nr) {
      csw98_structured_side_cfl(min_dt, x_r, x_z, v_r, v_z, node_mass, rho,
                                cs, hydro_active, i * nz + j, 0, nr, nz,
                                params, coefficient, aw_planar,
                                aw_axis_slave_theta0_active,
                                aw_axis_slave_theta_pi_active, axis_line_edge,
                                winner_index, 2 * e + 1, target_dt);
    }
  }
}

__device__ inline bool structured_cfl_edge_info(
    int* __restrict__ n0,
    int* __restrict__ n1,
    int* __restrict__ cell_a,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const int e) {
  const int n_radial = nr * (nz + 1);
  bool compressive = false;
  *cell_a = -1;
  if (e < n_radial) {
    const int i = e / (nz + 1);
    const int j = e - i * (nz + 1);
    *n0 = structured_node_index(i, j, nz);
    *n1 = structured_node_index(i + 1, j, nz);
    if (j > 0 &&
        structured_edge_is_compressive(x_r, x_z, v_r, v_z, hydro_active,
                                       i * nz + (j - 1), 3, nz)) {
      compressive = true;
      *cell_a = i * nz + (j - 1);
    }
    if (j < nz &&
        structured_edge_is_compressive(x_r, x_z, v_r, v_z, hydro_active,
                                       i * nz + j, 2, nz)) {
      compressive = true;
      if (*cell_a < 0) {
        *cell_a = i * nz + j;
      }
    }
  } else {
    const int a = e - n_radial;
    const int i = a / nz;
    const int j = a - i * nz;
    *n0 = structured_node_index(i, j, nz);
    *n1 = structured_node_index(i, j + 1, nz);
    if (i > 0 &&
        structured_edge_is_compressive(x_r, x_z, v_r, v_z, hydro_active,
                                       (i - 1) * nz + j, 1, nz)) {
      compressive = true;
      *cell_a = (i - 1) * nz + j;
    }
    if (i < nr &&
        structured_edge_is_compressive(x_r, x_z, v_r, v_z, hydro_active,
                                       i * nz + j, 0, nz)) {
      compressive = true;
      if (*cell_a < 0) {
        *cell_a = i * nz + j;
      }
    }
  }
  return compressive;
}

__device__ inline void fill_csw_edge_av_cfl_winner(
    CswEdgeAvCflWinner* __restrict__ winner,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int* __restrict__ cell_block_id,
    const int n0,
    const int n1,
    const int cell_a) {
  winner->n0 = n0;
  winner->n1 = n1;
  winner->cell_a = cell_a;
  winner->block_id =
      (cell_block_id != nullptr && cell_a >= 0) ? cell_block_id[cell_a] : -1;
  winner->r0 = x_r[n0];
  winner->z0 = x_z[n0];
  winner->r1 = x_r[n1];
  winner->z1 = x_z[n1];
  winner->dx = hypot(x_r[n1] - x_r[n0], x_z[n1] - x_z[n0]);
  winner->du = hypot(v_r[n1] - v_r[n0], v_z[n1] - v_z[n0]);
}

__global__ void csw_structured_cfl_winner_index_kernel(
    int* __restrict__ winner_index,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const double coefficient,
    const double target_dt,
    const bool aw_planar,
    const bool aw_axisline_av_enabled,
    const int aw_axis_slave_first_i,
    const bool aw_axis_slave_theta0_active,
    const bool aw_axis_slave_theta_pi_active) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_radial = nr * (nz + 1);
  const int n_edges = n_radial + (nr + 1) * nz;
  if (e >= n_edges) {
    return;
  }
  if (structured_aw_axis_line_edge(
          e, n_radial, nz, aw_axis_slave_first_i, aw_planar,
          aw_axisline_av_enabled,
          aw_axis_slave_theta0_active,
          aw_axis_slave_theta_pi_active)) {
    return;
  }
  int n0 = -1;
  int n1 = -1;
  int cell_a = -1;
  if (!structured_cfl_edge_info(&n0, &n1, &cell_a, x_r, x_z, v_r, v_z,
                                hydro_active, nr, nz, e)) {
    return;
  }
  const double dx = hypot(x_r[n1] - x_r[n0], x_z[n1] - x_z[n0]);
  const double du = hypot(v_r[n1] - v_r[n0], v_z[n1] - v_z[n0]);
  if (dx > 0.0 && du > kTiny && isfinite(dx) && isfinite(du) &&
      edge_dt_matches(coefficient * dx / du, target_dt)) {
    atomicMin(winner_index, e);
  }
}

__global__ void csw_structured_cfl_winner_values_kernel(
    CswEdgeAvCflWinner* __restrict__ winner,
    const int* __restrict__ winner_index,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const bool aw_planar,
    const bool aw_axisline_av_enabled,
    const int aw_axis_slave_first_i,
    const bool aw_axis_slave_theta0_active,
    const bool aw_axis_slave_theta_pi_active) {
  const int e = *winner_index;
  const int n_radial = nr * (nz + 1);
  const int n_edges = n_radial + (nr + 1) * nz;
  if (e < 0 || e >= n_edges) {
    return;
  }
  if (structured_aw_axis_line_edge(
          e, n_radial, nz, aw_axis_slave_first_i, aw_planar,
          aw_axisline_av_enabled,
          aw_axis_slave_theta0_active,
          aw_axis_slave_theta_pi_active)) {
    return;
  }
  int n0 = -1;
  int n1 = -1;
  int cell_a = -1;
  if (structured_cfl_edge_info(&n0, &n1, &cell_a, x_r, x_z, v_r, v_z,
                               hydro_active, nr, nz, e)) {
    fill_csw_edge_av_cfl_winner(winner, x_r, x_z, v_r, v_z, nullptr, n0, n1,
                                cell_a);
    winner->edge_id = e;
  }
}

__global__ void csw98_structured_cfl_winner_values_kernel(
    CswEdgeAvCflWinner* __restrict__ winner,
    const int* __restrict__ winner_index,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int nr,
    const int nz,
    const CswKernelParams params,
    const bool aw_planar,
    const bool aw_axisline_av_enabled,
    const int aw_axis_slave_first_i,
    const bool aw_axis_slave_theta0_active,
    const bool aw_axis_slave_theta_pi_active,
    const double raw_dt) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int candidate = *winner_index;
  if (candidate < 0) {
    return;
  }
  const int e = candidate / 2;
  const int side = candidate & 1;
  const int n_radial = nr * (nz + 1);
  const int n_edges = n_radial + (nr + 1) * nz;
  if (e >= n_edges) {
    return;
  }
  int cell = -1;
  int local = -1;
  int n0 = -1;
  int n1 = -1;
  if (e < n_radial) {
    const int i = e / (nz + 1);
    const int j = e - i * (nz + 1);
    n0 = structured_node_index(i, j, nz);
    n1 = structured_node_index(i + 1, j, nz);
    if (side == 0 && j > 0) {
      cell = i * nz + (j - 1);
      local = 3;
    } else if (side == 1 && j < nz) {
      cell = i * nz + j;
      local = 2;
    }
  } else {
    const int a = e - n_radial;
    const int i = a / nz;
    const int j = a - i * nz;
    n0 = structured_node_index(i, j, nz);
    n1 = structured_node_index(i, j + 1, nz);
    if (side == 0 && i > 0) {
      cell = (i - 1) * nz + j;
      local = 1;
    } else if (side == 1 && i < nr) {
      cell = i * nz + j;
      local = 0;
    }
  }
  if (cell < 0) {
    return;
  }
  fill_csw_edge_av_cfl_winner(winner, x_r, x_z, v_r, v_z, nullptr, n0, n1,
                              cell);
  if (params.axisline_d1prime != 0 &&
      structured_aw_axis_line_edge(
          e, n_radial, nz, aw_axis_slave_first_i, aw_planar,
          aw_axisline_av_enabled, aw_axis_slave_theta0_active,
          aw_axis_slave_theta_pi_active)) {
    winner->du = fabs(v_z[n1] - v_z[n0]);
    winner->raw_dt = raw_dt;
  }
  winner->edge_id = e;
  winner->local = local;
}

__global__ void csw_multiblock_internal_cfl_kernel(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ face_local_b,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_faces,
    const double coefficient) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int ca = face_cell_a[f];
  const int la = face_local_a[f];
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, ca);
  int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {0};
  csr_cell_active_nodes(ca, active_nverts, cell_node_offsets,
                        cell_node_indices, nodes);
  int c0 = 0;
  int c1 = 0;
  if (!local_face_corners(active_nverts, la, &c0, &c1)) {
    return;
  }
  const int n0 = nodes[c0];
  const int n1 = nodes[c1];
  const bool compressive =
      multiblock_edge_is_compressive(x_r, x_z, v_r, v_z, hydro_active,
                                     cell_node_offsets, cell_node_indices,
                                     cell_nverts, ca, la) ||
      multiblock_edge_is_compressive(x_r, x_z, v_r, v_z, hydro_active,
                                     cell_node_offsets, cell_node_indices,
                                     cell_nverts, face_cell_b[f],
                                     face_local_b[f]);
  const double dx = hypot(x_r[n1] - x_r[n0], x_z[n1] - x_z[n0]);
  const double du = hypot(v_r[n1] - v_r[n0], v_z[n1] - v_z[n0]);
  if (compressive && dx > 0.0 && du > kTiny && isfinite(dx) && isfinite(du)) {
    atomic_min_double(min_dt, coefficient * dx / du);
  }
}

__global__ void csw98_multiblock_internal_cfl_kernel(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ face_local_b,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ edge_psi,
    const int* __restrict__ cell_edge_offsets,
    const int* __restrict__ cell_edge_edges,
    const int* __restrict__ line_edge_n0,
    const int* __restrict__ line_edge_n1,
    const int n_faces,
    const CswKernelParams params,
    const double coefficient,
    const bool aw_planar) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  csw98_multiblock_side_cfl(
      min_dt, x_r, x_z, v_r, v_z, node_mass, rho, cs, hydro_active,
      cell_node_offsets,
      cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts,
      edge_psi, cell_edge_offsets, cell_edge_edges, line_edge_n0, line_edge_n1,
      f, face_cell_a[f], face_local_a[f], params, coefficient, aw_planar,
      nullptr, -1, 0.0);
  csw98_multiblock_side_cfl(
      min_dt, x_r, x_z, v_r, v_z, node_mass, rho, cs, hydro_active,
      cell_node_offsets,
      cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts,
      edge_psi, cell_edge_offsets, cell_edge_edges, line_edge_n0, line_edge_n1,
      f, face_cell_b[f], face_local_b[f], params, coefficient, aw_planar,
      nullptr, -1, 0.0);
}

__global__ void csw98_multiblock_internal_cfl_polar_slaving_kernel(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ face_local_b,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ edge_psi,
    const int n_faces,
    const CswKernelParams params,
    const double coefficient,
    const bool aw_planar,
    const CswPolarSlavingDeviceView slaving) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  csw98_multiblock_side_cfl_polar_slaving(
      min_dt, x_r, x_z, v_r, v_z, node_mass, rho, cs, hydro_active,
      cell_node_offsets,
      cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts,
      edge_psi, f, face_cell_a[f], face_local_a[f], params, coefficient,
      aw_planar, slaving, nullptr, -1, 0.0);
  csw98_multiblock_side_cfl_polar_slaving(
      min_dt, x_r, x_z, v_r, v_z, node_mass, rho, cs, hydro_active,
      cell_node_offsets,
      cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts,
      edge_psi, f, face_cell_b[f], face_local_b[f], params, coefficient,
      aw_planar, slaving, nullptr, -1, 0.0);
}

__device__ inline bool multiblock_internal_cfl_edge_info(
    int* __restrict__ n0,
    int* __restrict__ n1,
    int* __restrict__ cell_a,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ face_local_b,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int f) {
  const int ca = face_cell_a[f];
  const int la = face_local_a[f];
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, ca);
  int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {0};
  csr_cell_active_nodes(ca, active_nverts, cell_node_offsets,
                        cell_node_indices, nodes);
  int c0 = 0;
  int c1 = 0;
  if (!local_face_corners(active_nverts, la, &c0, &c1)) {
    return false;
  }
  *n0 = nodes[c0];
  *n1 = nodes[c1];
  *cell_a = ca;
  return multiblock_edge_is_compressive(x_r, x_z, v_r, v_z, hydro_active,
                                        cell_node_offsets, cell_node_indices,
                                        cell_nverts, ca, la) ||
         multiblock_edge_is_compressive(x_r, x_z, v_r, v_z, hydro_active,
                                        cell_node_offsets, cell_node_indices,
                                        cell_nverts, face_cell_b[f],
                                        face_local_b[f]);
}

__global__ void csw_multiblock_internal_cfl_winner_index_kernel(
    int* __restrict__ winner_index,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ face_local_b,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_faces,
    const double coefficient,
    const double target_dt) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  int n0 = -1;
  int n1 = -1;
  int cell_a = -1;
  if (!multiblock_internal_cfl_edge_info(
          &n0, &n1, &cell_a, x_r, x_z, v_r, v_z, hydro_active, face_cell_a,
          face_cell_b, face_local_a, face_local_b, cell_node_offsets,
          cell_node_indices, cell_nverts, f)) {
    return;
  }
  const double dx = hypot(x_r[n1] - x_r[n0], x_z[n1] - x_z[n0]);
  const double du = hypot(v_r[n1] - v_r[n0], v_z[n1] - v_z[n0]);
  if (dx > 0.0 && du > kTiny && isfinite(dx) && isfinite(du) &&
      edge_dt_matches(coefficient * dx / du, target_dt)) {
    atomicMin(winner_index, f);
  }
}

__global__ void csw_multiblock_boundary_cfl_kernel(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell,
    const int* __restrict__ face_local,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_faces,
    const double coefficient) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int c = face_cell[f];
  const int local = face_local[f];
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {0};
  csr_cell_active_nodes(c, active_nverts, cell_node_offsets, cell_node_indices,
                        nodes);
  int c0 = 0;
  int c1 = 0;
  if (!local_face_corners(active_nverts, local, &c0, &c1)) {
    return;
  }
  const int n0 = nodes[c0];
  const int n1 = nodes[c1];
  const bool compressive =
      multiblock_edge_is_compressive(x_r, x_z, v_r, v_z, hydro_active,
                                     cell_node_offsets, cell_node_indices,
                                     cell_nverts, c, local);
  const double dx = hypot(x_r[n1] - x_r[n0], x_z[n1] - x_z[n0]);
  const double du = hypot(v_r[n1] - v_r[n0], v_z[n1] - v_z[n0]);
  if (compressive && dx > 0.0 && du > kTiny && isfinite(dx) && isfinite(du)) {
    atomic_min_double(min_dt, coefficient * dx / du);
  }
}

__global__ void csw98_multiblock_boundary_cfl_kernel(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell,
    const int* __restrict__ face_local,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ edge_psi,
    const int* __restrict__ cell_edge_offsets,
    const int* __restrict__ cell_edge_edges,
    const int* __restrict__ line_edge_n0,
    const int* __restrict__ line_edge_n1,
    const int edge_offset,
    const int n_faces,
    const CswKernelParams params,
    const double coefficient,
    const bool aw_planar) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  csw98_multiblock_side_cfl(
      min_dt, x_r, x_z, v_r, v_z, node_mass, rho, cs, hydro_active,
      cell_node_offsets,
      cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts,
      edge_psi, cell_edge_offsets, cell_edge_edges, line_edge_n0, line_edge_n1,
      edge_offset + f, face_cell[f], face_local[f], params, coefficient,
      aw_planar, nullptr, -1, 0.0);
}

__global__ void csw98_multiblock_internal_cfl_winner_index_kernel(
    int* __restrict__ winner_index,
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ face_local_b,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ edge_psi,
    const int* __restrict__ cell_edge_offsets,
    const int* __restrict__ cell_edge_edges,
    const int* __restrict__ line_edge_n0,
    const int* __restrict__ line_edge_n1,
    const int n_faces,
    const CswKernelParams params,
    const double coefficient,
    const bool aw_planar,
    const double target_dt) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  csw98_multiblock_side_cfl(
      min_dt, x_r, x_z, v_r, v_z, node_mass, rho, cs, hydro_active,
      cell_node_offsets,
      cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts,
      edge_psi, cell_edge_offsets, cell_edge_edges, line_edge_n0, line_edge_n1,
      f, face_cell_a[f], face_local_a[f], params, coefficient, aw_planar,
      winner_index, 2 * f, target_dt);
  csw98_multiblock_side_cfl(
      min_dt, x_r, x_z, v_r, v_z, node_mass, rho, cs, hydro_active,
      cell_node_offsets,
      cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts,
      edge_psi, cell_edge_offsets, cell_edge_edges, line_edge_n0, line_edge_n1,
      f, face_cell_b[f], face_local_b[f], params, coefficient, aw_planar,
      winner_index, 2 * f + 1, target_dt);
}

__global__ void
csw98_multiblock_internal_cfl_winner_index_polar_slaving_kernel(
    int* __restrict__ winner_index,
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ face_local_b,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ edge_psi,
    const int n_faces,
    const CswKernelParams params,
    const double coefficient,
    const bool aw_planar,
    const double target_dt,
    const CswPolarSlavingDeviceView slaving) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  csw98_multiblock_side_cfl_polar_slaving(
      min_dt, x_r, x_z, v_r, v_z, node_mass, rho, cs, hydro_active,
      cell_node_offsets,
      cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts,
      edge_psi, f, face_cell_a[f], face_local_a[f], params, coefficient,
      aw_planar, slaving, winner_index, 2 * f, target_dt);
  csw98_multiblock_side_cfl_polar_slaving(
      min_dt, x_r, x_z, v_r, v_z, node_mass, rho, cs, hydro_active,
      cell_node_offsets,
      cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts,
      edge_psi, f, face_cell_b[f], face_local_b[f], params, coefficient,
      aw_planar, slaving, winner_index, 2 * f + 1, target_dt);
}

__global__ void csw98_multiblock_boundary_cfl_winner_index_kernel(
    int* __restrict__ winner_index,
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell,
    const int* __restrict__ face_local,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ edge_psi,
    const int* __restrict__ cell_edge_offsets,
    const int* __restrict__ cell_edge_edges,
    const int* __restrict__ line_edge_n0,
    const int* __restrict__ line_edge_n1,
    const int n_internal_faces,
    const int n_boundary_faces,
    const CswKernelParams params,
    const double coefficient,
    const bool aw_planar,
    const double target_dt) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_boundary_faces) {
    return;
  }
  csw98_multiblock_side_cfl(
      min_dt, x_r, x_z, v_r, v_z, node_mass, rho, cs, hydro_active,
      cell_node_offsets,
      cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts,
      edge_psi, cell_edge_offsets, cell_edge_edges, line_edge_n0, line_edge_n1,
      n_internal_faces + f, face_cell[f], face_local[f], params, coefficient,
      aw_planar, winner_index, 2 * n_internal_faces + f, target_dt);
}

__global__ void csw98_multiblock_cfl_winner_values_kernel(
    CswEdgeAvCflWinner* __restrict__ winner,
    const int* __restrict__ winner_index,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ face_local_b,
    const int* __restrict__ boundary_face_cell,
    const int* __restrict__ boundary_face_local,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int* __restrict__ cell_block_id,
    const int n_internal_faces,
    const int n_boundary_faces,
    const CswKernelParams params,
    const bool aw_planar,
    const double raw_dt) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int candidate = *winner_index;
  const int n_internal_candidates = 2 * n_internal_faces;
  int cell = -1;
  int local = -1;
  if (candidate >= 0 && candidate < n_internal_candidates) {
    const int f = candidate / 2;
    if ((candidate & 1) == 0) {
      cell = face_cell_a[f];
      local = face_local_a[f];
    } else {
      cell = face_cell_b[f];
      local = face_local_b[f];
    }
  } else {
    const int f = candidate - n_internal_candidates;
    if (f < 0 || f >= n_boundary_faces) {
      return;
    }
    cell = boundary_face_cell[f];
    local = boundary_face_local[f];
  }

  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, cell);
  int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {0};
  csr_cell_active_nodes(cell, active_nverts, cell_node_offsets,
                        cell_node_indices, nodes);
  int corner0 = 0;
  int corner1 = 0;
  if (!local_face_corners(active_nverts, local, &corner0, &corner1)) {
    return;
  }
  fill_csw_edge_av_cfl_winner(
      winner, x_r, x_z, v_r, v_z, cell_block_id, nodes[corner0],
      nodes[corner1], cell);
  if (params.axisline_d1prime != 0 &&
      multiblock_aw_axis_line_edge(x_r, nodes[corner0], nodes[corner1],
                                   aw_planar, params)) {
    winner->du = fabs(v_z[nodes[corner1]] - v_z[nodes[corner0]]);
    winner->raw_dt = raw_dt;
  }
  winner->edge_id = candidate < n_internal_candidates
                        ? candidate / 2
                        : n_internal_faces +
                              (candidate - n_internal_candidates);
  winner->local = local;
}

__device__ inline bool multiblock_boundary_cfl_edge_info(
    int* __restrict__ n0,
    int* __restrict__ n1,
    int* __restrict__ cell_a,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell,
    const int* __restrict__ face_local,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int f) {
  const int c = face_cell[f];
  const int local = face_local[f];
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {0};
  csr_cell_active_nodes(c, active_nverts, cell_node_offsets, cell_node_indices,
                        nodes);
  int c0 = 0;
  int c1 = 0;
  if (!local_face_corners(active_nverts, local, &c0, &c1)) {
    return false;
  }
  *n0 = nodes[c0];
  *n1 = nodes[c1];
  *cell_a = c;
  return multiblock_edge_is_compressive(x_r, x_z, v_r, v_z, hydro_active,
                                        cell_node_offsets, cell_node_indices,
                                        cell_nverts, c, local);
}

__global__ void csw_multiblock_boundary_cfl_winner_index_kernel(
    int* __restrict__ winner_index,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell,
    const int* __restrict__ face_local,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_internal,
    const int n_faces,
    const double coefficient,
    const double target_dt) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  int n0 = -1;
  int n1 = -1;
  int cell_a = -1;
  if (!multiblock_boundary_cfl_edge_info(
          &n0, &n1, &cell_a, x_r, x_z, v_r, v_z, hydro_active, face_cell,
          face_local, cell_node_offsets, cell_node_indices, cell_nverts, f)) {
    return;
  }
  const double dx = hypot(x_r[n1] - x_r[n0], x_z[n1] - x_z[n0]);
  const double du = hypot(v_r[n1] - v_r[n0], v_z[n1] - v_z[n0]);
  if (dx > 0.0 && du > kTiny && isfinite(dx) && isfinite(du) &&
      edge_dt_matches(coefficient * dx / du, target_dt)) {
    atomicMin(winner_index, n_internal + f);
  }
}

__global__ void csw_multiblock_cfl_winner_values_kernel(
    CswEdgeAvCflWinner* __restrict__ winner,
    const int* __restrict__ winner_index,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ unique_face_cell_a,
    const int* __restrict__ unique_face_cell_b,
    const int* __restrict__ unique_face_local_a,
    const int* __restrict__ unique_face_local_b,
    const int* __restrict__ boundary_face_cell,
    const int* __restrict__ boundary_face_local,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int* __restrict__ cell_block_id,
    const int n_internal,
    const int n_boundary) {
  const int idx = *winner_index;
  int n0 = -1;
  int n1 = -1;
  int cell_a = -1;
  bool compressive = false;
  if (idx >= 0 && idx < n_internal) {
    compressive = multiblock_internal_cfl_edge_info(
        &n0, &n1, &cell_a, x_r, x_z, v_r, v_z, hydro_active, unique_face_cell_a,
        unique_face_cell_b, unique_face_local_a, unique_face_local_b,
        cell_node_offsets, cell_node_indices, cell_nverts, idx);
  } else if (idx >= n_internal && idx < n_internal + n_boundary) {
    compressive = multiblock_boundary_cfl_edge_info(
        &n0, &n1, &cell_a, x_r, x_z, v_r, v_z, hydro_active, boundary_face_cell,
        boundary_face_local, cell_node_offsets, cell_node_indices, cell_nverts,
        idx - n_internal);
  }
  if (compressive) {
    fill_csw_edge_av_cfl_winner(winner, x_r, x_z, v_r, v_z, cell_block_id, n0,
                                n1, cell_a);
    winner->edge_id = idx;
  }
}

__device__ inline bool edge_accel_displacement_candidate(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ accel_r,
    const double* __restrict__ accel_z,
    const int n0,
    const int n1,
    const double coefficient,
    double* __restrict__ length,
    double* __restrict__ c_e,
    double* __restrict__ a_e,
    double* __restrict__ dt_e) {
  const double dx = x_r[n1] - x_r[n0];
  const double dz = x_z[n1] - x_z[n0];
  const double edge_length = hypot(dx, dz);
  if (!(edge_length > 0.0) || !finite_device(edge_length)) {
    return false;
  }
  const double e_r = dx / edge_length;
  const double e_z = dz / edge_length;
  const double dv_dot_e =
      (v_r[n1] - v_r[n0]) * e_r + (v_z[n1] - v_z[n0]) * e_z;
  const double da_dot_e = (accel_r[n1] - accel_r[n0]) * e_r +
                          (accel_z[n1] - accel_z[n0]) * e_z;
  const double closing_speed = fmax(0.0, -dv_dot_e);
  const double closing_accel = fmax(0.0, -da_dot_e);
  if (!(closing_speed > 0.0) && !(closing_accel > 0.0)) {
    return false;
  }
  double candidate = 0.0;
  if (closing_accel > 0.0) {
    candidate =
        (-closing_speed +
         sqrt(closing_speed * closing_speed +
              2.0 * closing_accel * coefficient * edge_length)) /
        closing_accel;
  } else {
    candidate = coefficient * edge_length / closing_speed;
  }
  if (!(candidate > 0.0) || !finite_device(candidate)) {
    return false;
  }
  if (length != nullptr) {
    *length = edge_length;
  }
  if (c_e != nullptr) {
    *c_e = closing_speed;
  }
  if (a_e != nullptr) {
    *a_e = closing_accel;
  }
  if (dt_e != nullptr) {
    *dt_e = candidate;
  }
  return true;
}

__device__ inline bool edge_accel_displacement_cell_face_nodes(
    int* __restrict__ n0,
    int* __restrict__ n1,
    const int cell,
    const int local,
    const int nz,
    const int corner_stride,
    const bool multiblock,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts) {
  const int active_nverts =
      multiblock
          ? mesh::mesh_topo_cell_active_nverts(cell_nverts, cell)
          : mesh::kMeshTopoCellStorageSlots;
  if (!mesh::mesh_topo_local_face_is_active(active_nverts, local)) {
    return false;
  }
  int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {0};
  if (multiblock) {
    csr_cell_active_nodes(cell, active_nverts, cell_node_offsets,
                          cell_node_indices, nodes);
  } else {
    structured_cell_nodes(cell, nz, nodes);
  }
  int corner0 = 0;
  int corner1 = 0;
  if (!local_face_corners(active_nverts, local, &corner0, &corner1) ||
      corner0 >= corner_stride || corner1 >= corner_stride) {
    return false;
  }
  *n0 = nodes[corner0];
  *n1 = nodes[corner1];
  return true;
}

__global__ void edge_accel_displacement_cfl_kernel(
    double* __restrict__ min_dt,
    int* __restrict__ winner_index,
    const double target_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ accel_r,
    const double* __restrict__ accel_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells,
    const int nz,
    const int corner_stride,
    const bool multiblock,
    const double coefficient) {
  const int candidate_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int cell = candidate_index / corner_stride;
  if (cell >= n_cells || !cell_active(hydro_active, cell)) {
    return;
  }
  const int local = candidate_index - cell * corner_stride;
  int n0 = -1;
  int n1 = -1;
  if (!edge_accel_displacement_cell_face_nodes(
          &n0, &n1, cell, local, nz, corner_stride, multiblock,
          cell_node_offsets, cell_node_indices, cell_nverts)) {
    return;
  }
  double candidate = 0.0;
  if (!edge_accel_displacement_candidate(
          x_r, x_z, v_r, v_z, accel_r, accel_z, n0, n1, coefficient,
          nullptr, nullptr, nullptr, &candidate)) {
    return;
  }
  atomic_min_double(min_dt, candidate);
  if (winner_index != nullptr &&
      edge_dt_matches(candidate, target_dt)) {
    atomicMin(winner_index, candidate_index);
  }
}

__device__ inline void fill_edge_accel_displacement_winner(
    EdgeAccelDisplacementWinner* __restrict__ winner,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ accel_r,
    const double* __restrict__ accel_z,
    const int edge_id,
    const int cell,
    const int n0,
    const int n1,
    const double coefficient) {
  double length = 0.0;
  double c_e = 0.0;
  double a_e = 0.0;
  double dt = 0.0;
  if (!edge_accel_displacement_candidate(
          x_r, x_z, v_r, v_z, accel_r, accel_z, n0, n1, coefficient,
          &length, &c_e, &a_e, &dt)) {
    return;
  }
  winner->edge_id = edge_id;
  winner->cell = cell;
  winner->node0 = n0;
  winner->node1 = n1;
  winner->length = length;
  winner->c_e = c_e;
  winner->a_e = a_e;
  winner->dt = dt;
}

__global__ void edge_accel_displacement_winner_values_kernel(
    EdgeAccelDisplacementWinner* __restrict__ winner,
    const int* __restrict__ winner_index,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ accel_r,
    const double* __restrict__ accel_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells,
    const int nz,
    const int corner_stride,
    const bool multiblock,
    const double coefficient) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int candidate_index = *winner_index;
  const int cell = candidate_index / corner_stride;
  if (candidate_index < 0 || cell >= n_cells ||
      !cell_active(hydro_active, cell)) {
    return;
  }
  const int local = candidate_index - cell * corner_stride;
  int n0 = -1;
  int n1 = -1;
  if (!edge_accel_displacement_cell_face_nodes(
          &n0, &n1, cell, local, nz, corner_stride, multiblock,
          cell_node_offsets, cell_node_indices, cell_nverts)) {
    return;
  }
  fill_edge_accel_displacement_winner(
      winner, x_r, x_z, v_r, v_z, accel_r, accel_z, candidate_index, cell,
      n0, n1, coefficient);
}

template <typename T>
const T* raw_or_null(const thrust::device_vector<T>& values) {
  return values.empty() ? nullptr : thrust::raw_pointer_cast(values.data());
}

__global__ void csw98_d1prime_node_mass_scatter_kernel(
    double* node_mass,
    const double* corner_mass,
    const int* csr_offsets,
    const int* csr_indices,
    const std::uint8_t* cell_nverts,
    const int corner_stride,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const int nv = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  for (int k = 0; k < nv; ++k) {
    atomicAdd(&node_mass[csr_indices[csr_offsets[c] + k]],
              corner_mass[c * corner_stride + k]);
  }
}

std::int8_t* upload_hydro_active_or_null(const core::State& state,
                                         const core::Config& cfg) {
  const int n_cells = static_cast<int>(state.rho.size());
  std::vector<std::int8_t> active;
  if (!state.hydro_active.empty()) {
    TENRYU_ASSERT(state.hydro_active.size() == static_cast<std::size_t>(n_cells),
                  "CSW edge AV hydro_active size mismatch");
    active = state.hydro_active;
  }
  core::State& mutable_state = const_cast<core::State&>(state);
  if (central_pseudo_core::configured(cfg)) {
    central_pseudo_core::ensure_built(mutable_state, cfg);
  }
  pole_angular_derefine::ensure_built(mutable_state, cfg);
  const auto apply_inactive = [&](const std::vector<std::uint8_t>& inactive) {
    if (inactive.size() != static_cast<std::size_t>(n_cells)) {
      return;
    }
    if (active.empty()) {
      active.assign(static_cast<std::size_t>(n_cells), 1);
    }
    for (int c = 0; c < n_cells; ++c) {
      if (inactive[static_cast<std::size_t>(c)] != 0U) {
        active[static_cast<std::size_t>(c)] = 0;
      }
    }
  };
  apply_inactive(mutable_state.central_pseudo_core.inactive_member_mask);
  apply_inactive(mutable_state.pole_angular_derefine.inactive_member_mask);
  if (active.empty()) {
    return nullptr;
  }
  std::int8_t* d_active = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_active),
                        active.size() * sizeof(std::int8_t)),
             "CSW edge AV: cudaMalloc hydro_active failed");
  cuda_check(cudaMemcpy(d_active, active.data(),
                        active.size() * sizeof(std::int8_t),
                        cudaMemcpyHostToDevice),
             "CSW edge AV: cudaMemcpy hydro_active failed");
  return d_active;
}

bool has_nonquad_cell_nverts(const std::vector<std::uint8_t>& cell_nverts,
                             const int n_cells) {
  if (cell_nverts.size() != static_cast<std::size_t>(n_cells)) {
    return false;
  }
  return std::any_of(cell_nverts.begin(), cell_nverts.end(),
                     [](const std::uint8_t nverts) {
                       return nverts != 4U;
                     });
}

const std::uint8_t* upload_cell_nverts_if_nonquad(
    core::DeviceArray<std::uint8_t>& d_cell_nverts,
    const core::State& state) {
  const int n_cells = static_cast<int>(state.rho.size());
  if (!has_nonquad_cell_nverts(state.mesh.cell_nverts, n_cells)) {
    return nullptr;
  }
  d_cell_nverts.reset(static_cast<std::size_t>(n_cells));
  d_cell_nverts.copy_from_host(state.mesh.cell_nverts);
  return d_cell_nverts.data();
}

std::string format_csw_scientific(const double value) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(16) << value;
  return oss.str();
}

int detect_structured_aw_axis_slave_first_i(const core::State& state) {
  if (state.mesh.topo.multiblock.has_value() || state.x_r.empty() ||
      state.mesh.topo.node_flags.size() != state.x_r.size()) {
    return 0;
  }
  const int n = state.mesh.topo.node_index(0, 0);
  return (state.mesh.topo.node_flags[static_cast<std::size_t>(n)] &
          mesh::NODE_CENTER) != 0U ? 1 : 0;
}

bool aw_axisline_av_enabled() {
  static const bool enabled = [] {
    const char* const raw = std::getenv("TENRYU_AW_AXISLINE_AV");
    return raw != nullptr && raw[0] == '1' && raw[1] == '\0';
  }();
  return enabled;
}

bool axistouch_av_off_enabled() {
  static const bool enabled = [] {
    const char* const raw = std::getenv("TENRYU_I1B_AXISTOUCH_AV_OFF");
    return raw != nullptr && raw[0] == '1' && raw[1] == '\0';
  }();
  return enabled;
}

struct CswPolarSlavingRuntime {
  const mesh::MultiBlockTopology* topology = nullptr;
  int n_cells = 0;
  int min_columns = 0;
  int k_core = 0;
  int k_outer = 0;
  int n_instances = 0;
  int n_cohorts = 0;
  int n_stiffness_faces = 0;
  int n_stiffness_nodes = 0;
  double chi_on = 0.0;
  double chi_full = 0.0;
  double strength = 0.0;
  bool stiffness_cfl_enabled = false;
  double stiffness_sigma = 0.8;
  long long diagnostic_step = -1;
  long long diagnostic_modified = 0;
  long long diagnostic_cohorts = 0;
  bool stiffness_diagnostics_valid = false;
  long long diagnostic_stiffness_step = -1;
  double diagnostic_stiffness_lambda = 0.0;
  double diagnostic_stiffness_theta = 0.0;
  int diagnostic_stiffness_winner_node = -1;
  bool previous_stiffness_diagnostics_valid = false;
  long long previous_diagnostic_stiffness_step = -1;
  double previous_diagnostic_stiffness_lambda = 0.0;
  double previous_diagnostic_stiffness_theta = 0.0;
  int previous_diagnostic_stiffness_winner_node = -1;

  core::DeviceArray<int> instance_cell;
  core::DeviceArray<int> instance_local;
  core::DeviceArray<int> instance_cohort;
  core::DeviceArray<int> instance_lookup;
  core::DeviceArray<int> cohort_members;
  core::DeviceArray<double> instance_weight;
  core::DeviceArray<double> psi_raw;
  core::DeviceArray<double> chi;
  core::DeviceArray<std::uint8_t> active;
  core::DeviceArray<std::uint8_t> valid;
  core::DeviceArray<double> cohort_gate;
  core::DeviceArray<double> cohort_psi_star;
  core::DeviceArray<int> diagnostics;
  core::DeviceArray<int> instance_stiffness_face;
  core::DeviceArray<int> stiffness_face_id;
  core::DeviceArray<int> stiffness_face_instance0;
  core::DeviceArray<int> stiffness_face_instance1;
  core::DeviceArray<int> stiffness_face_node0;
  core::DeviceArray<int> stiffness_face_node1;
  core::DeviceArray<int> stiffness_node_ids;
  core::DeviceArray<int> stiffness_node_face_offsets;
  core::DeviceArray<int> stiffness_node_face_indices;
  core::DeviceArray<double> stiffness_instance_tangent;
  core::DeviceArray<double> stiffness_face_kappa;
  core::DeviceArray<double> stiffness_node_lambda;
  core::DeviceArray<int> stiffness_failure;
  core::DeviceArray<double> stiffness_max_lambda;
  core::DeviceArray<int> stiffness_winner_node;

  CswPolarSlavingRuntime(const core::State& state, const core::Config& cfg) {
    TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                  "polar slaving requires multiblock topology metadata");
    topology = &*state.mesh.topo.multiblock;
    const auto& mb = *topology;
    n_cells = state.mesh.topo.n_cells;
    min_columns = cfg.numerics.hydro.csw_polar_slaving_min_columns;
    k_core = cfg.numerics.hydro.csw_polar_slaving_full_columns;
    k_outer = cfg.numerics.hydro.csw_polar_slaving_outer_columns;
    chi_on = cfg.numerics.hydro.csw_polar_slaving_chi_on;
    chi_full = cfg.numerics.hydro.csw_polar_slaving_chi_full;
    strength = cfg.numerics.hydro.csw_polar_slaving_strength;
    stiffness_cfl_enabled =
        cfg.numerics.hydro.csw_polar_slaving_av_stiffness_cfl_enabled;
    stiffness_sigma =
        cfg.numerics.hydro.csw_polar_slaving_av_stiffness_sigma;

    struct EligibleBlock {
      int block_id = -1;
      int cohort_base = 0;
    };
    std::vector<EligibleBlock> eligible_blocks;
    for (int block_id = 0; block_id < static_cast<int>(mb.blocks.size());
         ++block_id) {
      const auto& block = mb.blocks[static_cast<std::size_t>(block_id)];
      if (block.role != mesh::BlockRole::POLAR_SHELL &&
          block.role != mesh::BlockRole::POLAR_TIER) {
        continue;
      }
      if (block.n_i_cells <= 0 || block.n_j_cells < min_columns ||
          block.cell_count != block.n_i_cells * block.n_j_cells) {
        continue;
      }
      bool quad_only = true;
      if (!state.mesh.cell_nverts.empty()) {
        for (int cell = block.cell_begin;
             cell < block.cell_begin + block.cell_count; ++cell) {
          if (state.mesh.cell_nverts[static_cast<std::size_t>(cell)] != 4U) {
            core::log_info(
                "[polar-slaving] skipping block=" +
                std::to_string(block_id) + " non-quad cell=" +
                std::to_string(cell));
            quad_only = false;
            break;
          }
        }
      }
      if (!quad_only) {
        continue;
      }
      const int cohort_base = n_cohorts;
      n_cohorts += 2 * block.n_i_cells * 2;
      eligible_blocks.push_back({block_id, cohort_base});
    }
    TENRYU_ASSERT(!eligible_blocks.empty(),
                  "polar slaving requires at least one eligible block");

    std::vector<int> host_cell;
    std::vector<int> host_local;
    std::vector<int> host_cohort;
    std::vector<double> host_weight;
    std::vector<int> host_lookup(mb.face_adj_csr_indices.size(), -1);
    std::vector<int> host_members(
        static_cast<std::size_t>(n_cohorts * k_core), -1);
    struct PendingStiffnessFace {
      int face_id = -1;
      int instance0 = -1;
      int instance1 = -1;
    };
    std::vector<PendingStiffnessFace> pending_stiffness_faces;
    host_cell.reserve(mb.unique_internal_faces.size());
    host_local.reserve(mb.unique_internal_faces.size());
    host_cohort.reserve(mb.unique_internal_faces.size());
    host_weight.reserve(mb.unique_internal_faces.size());

    for (const EligibleBlock& eligible : eligible_blocks) {
      const auto& block =
          mb.blocks[static_cast<std::size_t>(eligible.block_id)];
      const auto append_instance =
          [&](const int c, const int local, const int neighbor) {
            const int local_cell = c - block.cell_begin;
            const int local_neighbor = neighbor - block.cell_begin;
            const int i = local_cell / block.n_j_cells;
            const int j = local_cell - i * block.n_j_cells;
            const int neighbor_i = local_neighbor / block.n_j_cells;
            const int neighbor_j =
                local_neighbor - neighbor_i * block.n_j_cells;
            const bool north = j < block.n_j_cells / 2;
            const int q = north ? j : block.n_j_cells - 1 - j;
            if (q < 0 || q >= k_outer) {
              return;
            }
            const bool neighbor_north = neighbor_j < block.n_j_cells / 2;
            const int neighbor_q =
                neighbor_north ? neighbor_j
                               : block.n_j_cells - 1 - neighbor_j;
            if (neighbor_q == q) {
              return;
            }
            const int role = neighbor_q < q ? 0 : 1;
            const int pole = north ? 0 : 1;
            const int cohort =
                eligible.cohort_base +
                (pole * block.n_i_cells + i) * 2 + role;
            double weight = 1.0;
            if (q >= k_core) {
              const double eta =
                  (static_cast<double>(q) + 0.5 -
                   static_cast<double>(k_core)) /
                  static_cast<double>(k_outer - k_core);
              const double one_minus_eta = 1.0 - eta;
              weight = one_minus_eta * one_minus_eta * (1.0 + 2.0 * eta);
            }
            const int lookup =
                mb.face_adj_csr_offsets[static_cast<std::size_t>(c)] + local;
            TENRYU_ASSERT(
                lookup >= 0 &&
                    lookup < static_cast<int>(host_lookup.size()) &&
                    host_lookup[static_cast<std::size_t>(lookup)] < 0,
                "polar slaving instance lookup is not one-to-one");
            const int instance = static_cast<int>(host_cell.size());
            host_lookup[static_cast<std::size_t>(lookup)] = instance;
            host_cell.push_back(c);
            host_local.push_back(local);
            host_cohort.push_back(cohort);
            host_weight.push_back(weight);
            if (q < k_core) {
              const std::size_t member =
                  static_cast<std::size_t>(cohort * k_core + q);
              TENRYU_ASSERT(host_members[member] < 0,
                            "polar slaving cohort member is not unique");
              host_members[member] = instance;
            }
          };

      for (int face_id = 0;
           face_id < static_cast<int>(mb.unique_internal_faces.size());
           ++face_id) {
        const auto& face =
            mb.unique_internal_faces[static_cast<std::size_t>(face_id)];
        if (face.cell_a < block.cell_begin ||
            face.cell_a >= block.cell_begin + block.cell_count ||
            face.cell_b < block.cell_begin ||
            face.cell_b >= block.cell_begin + block.cell_count) {
          continue;
        }
        const int local_a = face.cell_a - block.cell_begin;
        const int local_b = face.cell_b - block.cell_begin;
        if (local_a / block.n_j_cells != local_b / block.n_j_cells) {
          continue;
        }
        const int instance_begin = static_cast<int>(host_cell.size());
        append_instance(face.cell_a, face.local_a, face.cell_b);
        append_instance(face.cell_b, face.local_b, face.cell_a);
        const int appended = static_cast<int>(host_cell.size()) - instance_begin;
        if (appended != 0) {
          TENRYU_ASSERT(appended == 1 || appended == 2,
                        "polar slaving face must append one or two instances");
          if (stiffness_cfl_enabled) {
            pending_stiffness_faces.push_back(
                {face_id, instance_begin,
                 appended == 2 ? instance_begin + 1 : -1});
          }
        }
      }
    }
    n_instances = static_cast<int>(host_cell.size());
    TENRYU_ASSERT(n_instances > 0,
                  "polar slaving found no supported block instances");

    instance_cell.reset(host_cell.size());
    instance_cell.copy_from_host(host_cell);
    instance_local.reset(host_local.size());
    instance_local.copy_from_host(host_local);
    instance_cohort.reset(host_cohort.size());
    instance_cohort.copy_from_host(host_cohort);
    instance_lookup.reset(host_lookup.size());
    instance_lookup.copy_from_host(host_lookup);
    cohort_members.reset(host_members.size());
    cohort_members.copy_from_host(host_members);
    instance_weight.reset(host_weight.size());
    instance_weight.copy_from_host(host_weight);
    psi_raw.reset(host_cell.size());
    chi.reset(host_cell.size());
    active.reset(host_cell.size());
    valid.reset(host_cell.size());
    cohort_gate.reset(static_cast<std::size_t>(n_cohorts));
    cohort_psi_star.reset(static_cast<std::size_t>(n_cohorts));
    diagnostics.reset(3U);
    if (stiffness_cfl_enabled) {
      std::sort(pending_stiffness_faces.begin(),
                pending_stiffness_faces.end(),
                [](const PendingStiffnessFace& lhs,
                   const PendingStiffnessFace& rhs) {
                  return lhs.face_id < rhs.face_id;
                });
      n_stiffness_faces =
          static_cast<int>(pending_stiffness_faces.size());
      TENRYU_ASSERT(n_stiffness_faces > 0,
                    "polar slaving stiffness CFL found no slaved faces");
      TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                        static_cast<std::size_t>(n_cells) + 1U,
                    "polar slaving stiffness CFL requires host cell-node CSR "
                    "offsets");

      std::vector<int> host_instance_face(
          static_cast<std::size_t>(n_instances), -1);
      std::vector<int> host_face_id;
      std::vector<int> host_face_instance0;
      std::vector<int> host_face_instance1;
      std::vector<int> host_face_node0;
      std::vector<int> host_face_node1;
      host_face_id.reserve(pending_stiffness_faces.size());
      host_face_instance0.reserve(pending_stiffness_faces.size());
      host_face_instance1.reserve(pending_stiffness_faces.size());
      host_face_node0.reserve(pending_stiffness_faces.size());
      host_face_node1.reserve(pending_stiffness_faces.size());
      for (int stiffness_face = 0; stiffness_face < n_stiffness_faces;
           ++stiffness_face) {
        const PendingStiffnessFace& pending =
            pending_stiffness_faces[static_cast<std::size_t>(stiffness_face)];
        const auto& face =
            mb.unique_internal_faces[static_cast<std::size_t>(pending.face_id)];
        const int active_nverts = mesh::mesh_topo_cell_active_nverts(
            state.mesh.cell_nverts, face.cell_a);
        int corner0 = 0;
        int corner1 = 0;
        TENRYU_ASSERT(mesh::mesh_topo_active_local_face_corners(
                          active_nverts, face.local_a, &corner0, &corner1),
                      "polar slaving stiffness CFL face is not active");
        const int cell_offset = mb.cell_node_csr_offsets[static_cast<std::size_t>(
            face.cell_a)];
        TENRYU_ASSERT(
            cell_offset >= 0 &&
                cell_offset + corner0 <
                    static_cast<int>(mb.cell_node_csr_indices.size()) &&
                cell_offset + corner1 <
                    static_cast<int>(mb.cell_node_csr_indices.size()),
            "polar slaving stiffness CFL face-node lookup is out of range");
        const int local_node0 = mb.cell_node_csr_indices[
            static_cast<std::size_t>(cell_offset + corner0)];
        const int local_node1 = mb.cell_node_csr_indices[
            static_cast<std::size_t>(cell_offset + corner1)];
        const int canonical_node0 = std::min(local_node0, local_node1);
        const int canonical_node1 = std::max(local_node0, local_node1);
        TENRYU_ASSERT(canonical_node0 >= 0 &&
                          canonical_node1 < state.mesh.topo.n_nodes &&
                          canonical_node0 < canonical_node1,
                      "polar slaving stiffness CFL face endpoints are invalid");
        if (pending.instance1 >= 0) {
          const int active_nverts_b = mesh::mesh_topo_cell_active_nverts(
              state.mesh.cell_nverts, face.cell_b);
          int corner0_b = 0;
          int corner1_b = 0;
          TENRYU_ASSERT(mesh::mesh_topo_active_local_face_corners(
                            active_nverts_b, face.local_b, &corner0_b,
                            &corner1_b),
                        "polar slaving stiffness CFL reciprocal face is not "
                        "active");
          const int cell_offset_b = mb.cell_node_csr_offsets[
              static_cast<std::size_t>(face.cell_b)];
          TENRYU_ASSERT(
              cell_offset_b >= 0 &&
                  cell_offset_b + corner0_b <
                      static_cast<int>(mb.cell_node_csr_indices.size()) &&
                  cell_offset_b + corner1_b <
                      static_cast<int>(mb.cell_node_csr_indices.size()),
              "polar slaving stiffness CFL reciprocal face-node lookup is "
              "out of range");
          const int reciprocal_node0 = mb.cell_node_csr_indices[
              static_cast<std::size_t>(cell_offset_b + corner0_b)];
          const int reciprocal_node1 = mb.cell_node_csr_indices[
              static_cast<std::size_t>(cell_offset_b + corner1_b)];
          TENRYU_ASSERT(std::min(reciprocal_node0, reciprocal_node1) ==
                                canonical_node0 &&
                            std::max(reciprocal_node0, reciprocal_node1) ==
                                canonical_node1,
                        "polar slaving stiffness CFL reciprocal face "
                        "endpoints do not match");
        }

        host_instance_face[static_cast<std::size_t>(pending.instance0)] =
            stiffness_face;
        if (pending.instance1 >= 0) {
          host_instance_face[static_cast<std::size_t>(pending.instance1)] =
              stiffness_face;
        }
        host_face_id.push_back(pending.face_id);
        host_face_instance0.push_back(pending.instance0);
        host_face_instance1.push_back(pending.instance1);
        host_face_node0.push_back(canonical_node0);
        host_face_node1.push_back(canonical_node1);
      }
      TENRYU_ASSERT(
          std::all_of(host_instance_face.begin(), host_instance_face.end(),
                      [](const int face) { return face >= 0; }),
          "polar slaving stiffness CFL instance-face map is incomplete");

      std::vector<std::pair<int, int>> incidence;
      incidence.reserve(2U * pending_stiffness_faces.size());
      for (int face = 0; face < n_stiffness_faces; ++face) {
        incidence.emplace_back(
            host_face_node0[static_cast<std::size_t>(face)], face);
        incidence.emplace_back(
            host_face_node1[static_cast<std::size_t>(face)], face);
      }
      std::sort(incidence.begin(), incidence.end());
      std::vector<int> host_node_ids;
      std::vector<int> host_node_face_offsets(1U, 0);
      std::vector<int> host_node_face_indices;
      host_node_face_indices.reserve(incidence.size());
      int current_node = -1;
      for (const auto& item : incidence) {
        if (item.first != current_node) {
          if (current_node >= 0) {
            host_node_face_offsets.push_back(
                static_cast<int>(host_node_face_indices.size()));
          }
          current_node = item.first;
          host_node_ids.push_back(current_node);
        }
        host_node_face_indices.push_back(item.second);
      }
      host_node_face_offsets.push_back(
          static_cast<int>(host_node_face_indices.size()));
      n_stiffness_nodes = static_cast<int>(host_node_ids.size());
      TENRYU_ASSERT(n_stiffness_nodes > 0 &&
                        host_node_face_offsets.size() ==
                            host_node_ids.size() + 1U,
                    "polar slaving stiffness CFL node CSR is invalid");

      instance_stiffness_face.reset(host_instance_face.size());
      instance_stiffness_face.copy_from_host(host_instance_face);
      stiffness_face_id.reset(host_face_id.size());
      stiffness_face_id.copy_from_host(host_face_id);
      stiffness_face_instance0.reset(host_face_instance0.size());
      stiffness_face_instance0.copy_from_host(host_face_instance0);
      stiffness_face_instance1.reset(host_face_instance1.size());
      stiffness_face_instance1.copy_from_host(host_face_instance1);
      stiffness_face_node0.reset(host_face_node0.size());
      stiffness_face_node0.copy_from_host(host_face_node0);
      stiffness_face_node1.reset(host_face_node1.size());
      stiffness_face_node1.copy_from_host(host_face_node1);
      stiffness_node_ids.reset(host_node_ids.size());
      stiffness_node_ids.copy_from_host(host_node_ids);
      stiffness_node_face_offsets.reset(host_node_face_offsets.size());
      stiffness_node_face_offsets.copy_from_host(host_node_face_offsets);
      stiffness_node_face_indices.reset(host_node_face_indices.size());
      stiffness_node_face_indices.copy_from_host(host_node_face_indices);
      stiffness_instance_tangent.reset(4U * host_cell.size());
      stiffness_face_kappa.reset(host_face_id.size());
      stiffness_node_lambda.reset(host_node_ids.size());
      stiffness_failure.reset(1U);
      stiffness_max_lambda.reset(1U);
      stiffness_winner_node.reset(1U);
    }
    core::log_info(
        "[polar-slaving] runtime built blocks=" +
        std::to_string(eligible_blocks.size()));
  }

  bool matches(const core::State& state, const core::Config& cfg) const {
    return state.mesh.topo.multiblock.has_value() &&
           topology == &*state.mesh.topo.multiblock &&
           n_cells == state.mesh.topo.n_cells &&
           min_columns ==
               cfg.numerics.hydro.csw_polar_slaving_min_columns &&
           k_core == cfg.numerics.hydro.csw_polar_slaving_full_columns &&
           k_outer == cfg.numerics.hydro.csw_polar_slaving_outer_columns &&
           chi_on == cfg.numerics.hydro.csw_polar_slaving_chi_on &&
           chi_full == cfg.numerics.hydro.csw_polar_slaving_chi_full &&
           strength == cfg.numerics.hydro.csw_polar_slaving_strength &&
           stiffness_cfl_enabled ==
               cfg.numerics.hydro
                   .csw_polar_slaving_av_stiffness_cfl_enabled &&
           stiffness_sigma ==
               cfg.numerics.hydro.csw_polar_slaving_av_stiffness_sigma;
  }

  CswPolarSlavingDeviceView view() const {
    CswPolarSlavingDeviceView result;
    result.instance_lookup = instance_lookup.data();
    result.instance_cohort = instance_cohort.data();
    result.instance_weight = instance_weight.data();
    result.psi_raw = psi_raw.data();
    result.cohort_gate = cohort_gate.data();
    result.cohort_psi_star = cohort_psi_star.data();
    result.strength = strength;
    return result;
  }
};

std::unique_ptr<CswPolarSlavingRuntime> g_csw_polar_slaving_runtime;

CswPolarSlavingRuntime& ensure_csw_polar_slaving_runtime(
    const core::State& state,
    const core::Config& cfg) {
  if (g_csw_polar_slaving_runtime == nullptr ||
      !g_csw_polar_slaving_runtime->matches(state, cfg)) {
    g_csw_polar_slaving_runtime =
        std::make_unique<CswPolarSlavingRuntime>(state, cfg);
  }
  return *g_csw_polar_slaving_runtime;
}

CswPolarSlavingPreparation prepare_csw_polar_slaving(
    CswPolarSlavingRuntime& runtime,
    const double* x_r,
    const double* x_z,
    const double* v_r,
    const double* v_z,
    const double* rho,
    const double* cs,
    const std::int8_t* hydro_active,
    const int* cell_node_offsets,
    const int* cell_node_indices,
    const int* face_adj_offsets,
    const int* face_adj_indices,
    const std::uint8_t* cell_nverts,
    const double* edge_psi,
    const int* cell_edge_offsets,
    const int* cell_edge_edges,
    const int* line_edge_n0,
    const int* line_edge_n1,
    const CswKernelParams params,
    const bool aw_planar,
    int* limiter_miss,
    const bool prepare_stiffness_cfl,
    const core::NodeField1D* node_planar_mass,
    const int n_nodes,
    const double dt_accepted,
    const long long stiffness_step) {
  cuda_check(cudaMemset(runtime.diagnostics.data(), 0, 3U * sizeof(int)),
             "polar slaving: zero diagnostics failed");
  const int instance_blocks = (runtime.n_instances + 255) / 256;
  csw98_polar_slaving_raw_kernel<<<instance_blocks, 256>>>(
      runtime.psi_raw.data(), runtime.chi.data(), runtime.active.data(),
      runtime.valid.data(), runtime.diagnostics.data(),
      runtime.instance_cell.data(), runtime.instance_local.data(), x_r, x_z,
      v_r, v_z, rho, cs, hydro_active, cell_node_offsets, cell_node_indices,
      face_adj_offsets, face_adj_indices, cell_nverts, edge_psi,
      cell_edge_offsets, cell_edge_edges, line_edge_n0, line_edge_n1,
      runtime.n_instances, params, aw_planar, limiter_miss);
  cuda_check(cudaGetLastError(), "polar slaving: pass A launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "polar slaving: pass A execution failed");
  int nonfinite_count = 0;
  cuda_check(cudaMemcpy(&nonfinite_count, runtime.diagnostics.data() + 2,
                        sizeof(int), cudaMemcpyDeviceToHost),
             "polar slaving: copy nonfinite count failed");
  if (nonfinite_count != 0) {
    throw std::runtime_error(
        "polar slaving: non-finite/out-of-range limiter or compression "
        "diagnostic count=" +
        std::to_string(nonfinite_count));
  }
  const int cohort_blocks = (runtime.n_cohorts + 255) / 256;
  csw98_polar_slaving_cohort_kernel<<<cohort_blocks, 256>>>(
      runtime.cohort_gate.data(), runtime.cohort_psi_star.data(),
      runtime.cohort_members.data(), runtime.psi_raw.data(), runtime.chi.data(),
      runtime.active.data(), runtime.n_cohorts, runtime.k_core, runtime.chi_on,
      runtime.chi_full);
  cuda_check(cudaGetLastError(), "polar slaving: pass B launch failed");
  const CswPolarSlavingDeviceView view = runtime.view();
  csw98_polar_slaving_diagnostics_kernel<<<1, 1>>>(
      runtime.diagnostics.data(), runtime.valid.data(), runtime.n_instances,
      runtime.n_cohorts, view);
  cuda_check(cudaGetLastError(),
             "polar slaving: diagnostic counter launch failed");
  CswPolarSlavingPreparation preparation;
  preparation.view = view;
  if (!prepare_stiffness_cfl) {
    return preparation;
  }

  TENRYU_ASSERT(runtime.stiffness_cfl_enabled,
                "polar slaving stiffness CFL runtime is not enabled");
  if (node_planar_mass == nullptr || node_planar_mass->empty()) {
    static bool warned_node_planar_mass_unavailable = false;
    if (!warned_node_planar_mass_unavailable) {
      core::log_warning(
          "[polar-slaving] stiffness CFL: node_planar_mass not yet "
          "available; bound skipped this step");
      warned_node_planar_mass_unavailable = true;
    }
    return preparation;
  }
  TENRYU_ASSERT(node_planar_mass->size() ==
                    static_cast<std::size_t>(n_nodes),
                "polar slaving stiffness CFL state.node_planar_mass size "
                "mismatch");
  TENRYU_ASSERT(runtime.n_stiffness_faces > 0 &&
                    runtime.n_stiffness_nodes > 0,
                "polar slaving stiffness CFL runtime topology is empty");
  cuda_check(cudaMemset(runtime.stiffness_failure.data(), 0, sizeof(int)),
             "polar slaving stiffness CFL: zero failure flag failed");
  const int stiffness_instance_blocks = (runtime.n_instances + 255) / 256;
  csw98_polar_slaving_stiffness_instance_kernel
      <<<stiffness_instance_blocks, 256>>>(
      runtime.stiffness_instance_tangent.data(), runtime.instance_cell.data(),
      runtime.instance_local.data(), runtime.instance_stiffness_face.data(),
      runtime.stiffness_face_node0.data(),
      runtime.stiffness_face_node1.data(), x_r, x_z, v_r, v_z, rho, cs,
      hydro_active, cell_node_offsets, cell_node_indices, face_adj_offsets,
      face_adj_indices, cell_nverts, runtime.n_instances, params, aw_planar,
      view);
  cuda_check(cudaGetLastError(),
             "polar slaving stiffness CFL: pass S1 launch failed");
  const int face_blocks = (runtime.n_stiffness_faces + 255) / 256;
  csw98_polar_slaving_stiffness_face_kernel<<<face_blocks, 256>>>(
      runtime.stiffness_face_kappa.data(),
      runtime.stiffness_instance_tangent.data(),
      runtime.stiffness_face_instance0.data(),
      runtime.stiffness_face_instance1.data(), runtime.n_stiffness_faces);
  cuda_check(cudaGetLastError(),
             "polar slaving stiffness CFL: pass S2 launch failed");
  const int node_blocks = (runtime.n_stiffness_nodes + 255) / 256;
  csw98_polar_slaving_stiffness_node_kernel<<<node_blocks, 256>>>(
      runtime.stiffness_node_lambda.data(), runtime.stiffness_failure.data(),
      runtime.stiffness_node_ids.data(),
      runtime.stiffness_node_face_offsets.data(),
      runtime.stiffness_node_face_indices.data(),
      runtime.stiffness_face_kappa.data(), node_planar_mass->data(),
      runtime.n_stiffness_nodes);
  cuda_check(cudaGetLastError(),
             "polar slaving stiffness CFL: pass S3 launch failed");
  csw98_polar_slaving_stiffness_max_kernel<<<1, 1>>>(
      runtime.stiffness_max_lambda.data(),
      runtime.stiffness_winner_node.data(), runtime.stiffness_failure.data(),
      runtime.stiffness_node_lambda.data(), runtime.stiffness_node_ids.data(),
      runtime.n_stiffness_nodes);
  cuda_check(cudaGetLastError(),
             "polar slaving stiffness CFL: pass S4 launch failed");

  int failure = 0;
  double max_lambda = 0.0;
  int winner_node = -1;
  cuda_check(cudaMemcpy(&failure, runtime.stiffness_failure.data(), sizeof(int),
                        cudaMemcpyDeviceToHost),
             "polar slaving stiffness CFL: copy failure flag failed");
  cuda_check(cudaMemcpy(&max_lambda, runtime.stiffness_max_lambda.data(),
                        sizeof(double), cudaMemcpyDeviceToHost),
             "polar slaving stiffness CFL: copy max lambda failed");
  cuda_check(cudaMemcpy(&winner_node, runtime.stiffness_winner_node.data(),
                        sizeof(int), cudaMemcpyDeviceToHost),
             "polar slaving stiffness CFL: copy winner node failed");
  if (failure != 0 || !std::isfinite(max_lambda) || max_lambda < 0.0) {
    throw std::runtime_error(
        "polar slaving stiffness CFL: non-finite stiffness or non-positive "
        "node planar mass");
  }
  if (max_lambda > 0.0) {
    preparation.stiffness_dt = runtime.stiffness_sigma / max_lambda;
  }
  preparation.stiffness_lambda = max_lambda;
  preparation.stiffness_sigma = runtime.stiffness_sigma;
  preparation.stiffness_winner_node = winner_node;
  if (runtime.diagnostic_stiffness_step != stiffness_step) {
    runtime.previous_stiffness_diagnostics_valid =
        runtime.stiffness_diagnostics_valid;
    runtime.previous_diagnostic_stiffness_step =
        runtime.diagnostic_stiffness_step;
    runtime.previous_diagnostic_stiffness_lambda =
        runtime.diagnostic_stiffness_lambda;
    runtime.previous_diagnostic_stiffness_theta =
        runtime.diagnostic_stiffness_theta;
    runtime.previous_diagnostic_stiffness_winner_node =
        runtime.diagnostic_stiffness_winner_node;
  }
  runtime.stiffness_diagnostics_valid = true;
  runtime.diagnostic_stiffness_step = stiffness_step;
  runtime.diagnostic_stiffness_lambda = max_lambda;
  runtime.diagnostic_stiffness_theta = max_lambda * dt_accepted;
  runtime.diagnostic_stiffness_winner_node = winner_node;
  return preparation;
}

void record_csw_polar_slaving_diagnostics(CswPolarSlavingRuntime& runtime,
                                          const long long step) {
  int counters[2] = {0, 0};
  cuda_check(cudaMemcpy(counters, runtime.diagnostics.data(),
                        2U * sizeof(int), cudaMemcpyDeviceToHost),
             "polar slaving: copy diagnostic counters failed");
  if (runtime.diagnostic_step != step) {
    if (runtime.diagnostic_step >= 0 &&
        runtime.diagnostic_step % 200 == 0) {
      std::string message =
          "[polar-slaving] step=" +
          std::to_string(runtime.diagnostic_step) +
          " modified=" + std::to_string(runtime.diagnostic_modified) +
          " cohorts=" + std::to_string(runtime.diagnostic_cohorts);
      const bool use_current_stiffness_sample =
          runtime.stiffness_diagnostics_valid &&
          runtime.diagnostic_stiffness_step == runtime.diagnostic_step;
      const bool use_previous_stiffness_sample =
          runtime.previous_stiffness_diagnostics_valid &&
          runtime.previous_diagnostic_stiffness_step ==
              runtime.diagnostic_step;
      if (runtime.stiffness_cfl_enabled &&
          (use_current_stiffness_sample || use_previous_stiffness_sample)) {
        const double theta = use_current_stiffness_sample
                                 ? runtime.diagnostic_stiffness_theta
                                 : runtime.previous_diagnostic_stiffness_theta;
        const double lambda = use_current_stiffness_sample
                                  ? runtime.diagnostic_stiffness_lambda
                                  : runtime.previous_diagnostic_stiffness_lambda;
        const int winner_node =
            use_current_stiffness_sample
                ? runtime.diagnostic_stiffness_winner_node
                : runtime.previous_diagnostic_stiffness_winner_node;
        message +=
            " Theta=" + format_csw_scientific(theta) +
            " lamMax=" + format_csw_scientific(lambda) +
            " winnerNode=" + std::to_string(winner_node);
      }
      core::log_info(message);
    }
    runtime.diagnostic_step = step;
    runtime.diagnostic_modified = 0;
    runtime.diagnostic_cohorts = 0;
  }
  runtime.diagnostic_modified += counters[0];
  runtime.diagnostic_cohorts += counters[1];
}

}  // namespace

void destroy_csw_polar_slaving_runtime() {
  g_csw_polar_slaving_runtime.reset();
}

void launch_compute_csw_edge_av_2d(core::State& state,
                                   const core::Config& cfg,
                                   const core::CellField1D& cell_cs,
                                   const double* v_r,
                                   const double* v_z,
                                   const std::int8_t* hydro_active,
                                   const bool aw_axis_slave_theta0_active,
                                   const bool aw_axis_slave_theta_pi_active,
                                   const double* node_mass,
                                   const double dt) {
  if (pole_damper_enabled() && cfg.mesh.shell_polar_cap_dendrite) {
    throw core::namelist::ConfigError(
        "compatible-AV pole damper is not supported with "
        "Mesh.shell_polar_cap_dendrite=true (pending shell-chain generalization)");
  }
  if (cfg.main.dim != 2 ||
      (cfg.numerics.hydro.av_model != core::AvModel::CswEdge &&
       cfg.numerics.hydro.av_model != core::AvModel::CswEdgeCsw98)) {
    return;
  }
  const bool use_csw98 =
      cfg.numerics.hydro.av_model == core::AvModel::CswEdgeCsw98;
  const bool use_polar_slaving =
      use_csw98 && cfg.numerics.hydro.csw_polar_slaving_enabled;
  const bool aw_planar = cfg.numerics.hydro.aw_compatible_force_work;
  const bool aw_axisline_av = aw_axisline_av_enabled();
  const int aw_axis_slave_first_i =
      (aw_axis_slave_theta0_active || aw_axis_slave_theta_pi_active)
          ? detect_structured_aw_axis_slave_first_i(state)
          : 0;
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_nodes = static_cast<int>(state.x_r.size());
  const bool is_multiblock = mesh::mesh_topo_is_multiblock(cfg.mesh);
  TENRYU_ASSERT(cell_cs.size() == state.rho.size(),
                "CSW edge AV requires cell sound speed per cell");
  TENRYU_ASSERT(state.edge_force_av_r.size() == state.edge_force_av_z.size(),
                "CSW edge AV force buffer size mismatch");
  TENRYU_ASSERT(state.work_av_per_cell.size() == state.rho.size(),
                "CSW edge AV work buffer is not allocated");
  TENRYU_ASSERT(state.mesh.topo.n_cells == n_cells,
                "CSW edge AV topo/state cell-size mismatch");
  TENRYU_ASSERT(state.mesh.topo.n_nodes == n_nodes,
                "CSW edge AV topo/state node-size mismatch");

  CswKernelParams params;
  params.c1 = cfg.numerics.hydro.av_linear;
  params.c2 = cfg.numerics.hydro.av_quadratic;
  params.degenerate_side_floor_rel =
      cfg.numerics.hydro.csw98_degenerate_side_floor_rel;
  params.damper_impulse_beta =
      cfg.numerics.hydro.csw98_damper_impulse_beta;
  params.dt = dt;
  params.gamma = cfg.materials.materials.empty()
                     ? (5.0 / 3.0)
                     : cfg.materials.materials.front().ideal_gas_gamma;
  params.limiter_enabled = cfg.numerics.hydro.csw_limiter_enabled ? 1 : 0;
  params.axis_mirror_limiter =
      cfg.numerics.hydro.csw_axis_mirror_limiter ? 1 : 0;
  configure_csw98_edge_diag(&params);
  params.pole_floor_enabled =
      cfg.numerics.hydro.csw_pole_floor_enabled ? 1 : 0;
  params.pole_floor_sigma0 = cfg.numerics.hydro.csw_pole_floor_sigma0;
  params.pole_floor_theta0 = cfg.numerics.hydro.csw_pole_floor_theta0_rad;
  params.pole_floor_thetaf = cfg.numerics.hydro.csw_pole_floor_thetaf_rad;
  params.pole_desens_enabled =
      cfg.numerics.hydro.csw_pole_desens_enabled ? 1 : 0;
  params.pole_desens_alpha = cfg.numerics.hydro.csw_pole_desens_alpha;
  params.pole_desens_theta0 = cfg.numerics.hydro.csw_pole_desens_theta0_rad;
  params.pole_desens_thetaf = cfg.numerics.hydro.csw_pole_desens_thetaf_rad;
  params.axisline_av_enabled = aw_axisline_av_enabled() ? 1 : 0;
  params.axisline_d1prime =
      cfg.numerics.hydro.csw98_axisline_av_mode == "d1prime" ? 1 : 0;
  params.axisline_d1prime_cfl =
      cfg.numerics.hydro.csw98_axisline_d1prime_cfl_enabled ? 1 : 0;
  params.limiter_shock_floor =
      cfg.numerics.hydro.csw98_limiter_shock_floor_enabled ? 1 : 0;
  params.shock_limiter_floor = cfg.numerics.hydro.csw_shock_limiter_floor;
  params.axistouch_av_off = axistouch_av_off_enabled() ? 1 : 0;
  params.axis_eps_cm = cfg.numerics.axis_eps_cm;
  CswPolarSlavingRuntime* polar_slaving_runtime = nullptr;
  CswPolarSlavingDeviceView polar_slaving_view;
  double* d_work_av_edge = nullptr;
  if (is_multiblock && !state.edge_force_av_r.empty()) {
    d_work_av_edge = static_cast<double*>(core::device_scratch_acquire(
        "compatible_av_csw:work_av_edge",
        2U * state.edge_force_av_r.size() * sizeof(double)));
  }

  if (!state.edge_force_av_r.empty()) {
    cuda_check(cudaMemset(state.edge_force_av_r.data(), 0,
                          state.edge_force_av_r.size() * sizeof(double)),
               "CSW edge AV: zero edge_force_av_r failed");
    cuda_check(cudaMemset(state.edge_force_av_z.data(), 0,
                          state.edge_force_av_z.size() * sizeof(double)),
               "CSW edge AV: zero edge_force_av_z failed");
    if (is_multiblock) {
      cuda_check(cudaMemset(d_work_av_edge, 0,
                            2U * state.edge_force_av_r.size() *
                                sizeof(double)),
                 "CSW edge AV: zero work_av_edge failed");
    }
  }
  if (!state.work_av_per_cell.empty()) {
    cuda_check(cudaMemset(state.work_av_per_cell.data(), 0,
                          state.work_av_per_cell.size() * sizeof(double)),
               "CSW edge AV: zero work_av_per_cell failed");
  }

  int* d_compressive_count = nullptr;
  int* d_negative_work_count = nullptr;
  d_compressive_count = static_cast<int*>(core::device_scratch_acquire(
      "compatible_av_csw:edge_force_compressive_count", sizeof(int)));
  d_negative_work_count = static_cast<int*>(core::device_scratch_acquire(
      "compatible_av_csw:edge_force_negative_work_count", sizeof(int)));
  cuda_check(cudaMemset(d_compressive_count, 0, sizeof(int)),
             "CSW edge AV: zero compressive count failed");
  cuda_check(cudaMemset(d_negative_work_count, 0, sizeof(int)),
             "CSW edge AV: zero negative work count failed");

  int* d_limiter_miss = nullptr;
  static bool csw98_limiter_diag_logged = false;
  const bool csw98_limiter_diag =
      use_csw98 && !csw98_limiter_diag_logged &&
      std::getenv("TENRYU_CSW98_LIMITER_DIAG") != nullptr;
  if (csw98_limiter_diag) {
    d_limiter_miss = static_cast<int*>(core::device_scratch_acquire(
        "compatible_av_csw:edge_force_limiter_miss", 2 * sizeof(int)));
    cuda_check(cudaMemset(d_limiter_miss, 0, 2 * sizeof(int)),
               "csw98 AV: zero limiter diag counters failed");
  }

  if (is_multiblock) {
    TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                  "CSW edge AV multiblock topology metadata missing");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "CSW edge AV requires multiblock cell-node CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                      static_cast<std::size_t>(n_cells) *
                          static_cast<std::size_t>(state.mesh.corner_stride),
                  "CSW edge AV requires multiblock cell-node CSR indices");
    const auto& mb = *state.mesh.topo.multiblock;
    TENRYU_ASSERT(mb.face_adj_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "CSW edge AV requires face adjacency CSR offsets");
    TENRYU_ASSERT(mb.face_adj_csr_indices.size() >=
                      static_cast<std::size_t>(n_cells) *
                          static_cast<std::size_t>(state.mesh.corner_stride),
                  "CSW edge AV requires face adjacency CSR indices");
    const int n_internal = static_cast<int>(mb.unique_internal_faces.size());
    const int n_boundary = static_cast<int>(mb.boundary_faces.size());
    const int n_edges = n_internal + n_boundary;
    TENRYU_ASSERT(state.edge_force_av_r.size() ==
                      static_cast<std::size_t>(n_internal + n_boundary),
                  "CSW edge AV multiblock edge buffer size mismatch");
    const std::size_t cell_edge_incidence_count =
        2U * static_cast<std::size_t>(n_internal) +
        static_cast<std::size_t>(n_boundary);
    TENRYU_ASSERT(state.mesh.multiblock_cell_edge_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "CSW edge AV requires multiblock cell-edge CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_cell_edge_csr_edges.size() ==
                          cell_edge_incidence_count &&
                      state.mesh.multiblock_cell_edge_csr_side.size() ==
                          cell_edge_incidence_count,
                  "CSW edge AV requires multiblock cell-edge CSR entries");
    double* d_edge_psi = nullptr;
    if (use_csw98) {
      TENRYU_ASSERT(
          n_edges > 0 &&
              mb.d_csw_line_edge_n0.size() ==
                  static_cast<std::size_t>(n_edges) &&
              mb.d_csw_line_edge_n1.size() ==
                  static_cast<std::size_t>(n_edges) &&
              mb.d_csw_line_prev_edge.size() ==
                  static_cast<std::size_t>(n_edges) &&
              mb.d_csw_line_next_edge.size() ==
                  static_cast<std::size_t>(n_edges) &&
              mb.d_csw_line_prev_sign.size() ==
                  static_cast<std::size_t>(n_edges) &&
              mb.d_csw_line_next_sign.size() ==
                  static_cast<std::size_t>(n_edges) &&
              mb.d_csw_line_cand_offsets.size() ==
                  2U * static_cast<std::size_t>(n_edges) + 1U &&
              !mb.d_csw_line_cand_edges.empty() &&
              mb.d_csw_line_cand_edges.size() ==
                  mb.d_csw_line_cand_signs.size(),
          "csw98 edge AV requires non-empty phase-line device topology");
      TENRYU_ASSERT(
          mb.d_unique_face_cell_a.size() ==
                  static_cast<std::size_t>(n_internal) &&
              mb.d_unique_face_cell_b.size() ==
                  static_cast<std::size_t>(n_internal) &&
              mb.d_boundary_face_cell.size() ==
                  static_cast<std::size_t>(n_boundary),
          "csw98 edge AV requires face-cell device topology");
      d_edge_psi = static_cast<double*>(core::device_scratch_acquire(
          "compatible_av_csw:edge_psi",
          static_cast<std::size_t>(n_edges) * sizeof(double)));
      csw98_edge_psi_prepass_kernel<<<(n_edges + 255) / 256, 256>>>(
          state.x_r.data(), state.x_z.data(), v_r, v_z, cell_cs.data(),
          raw_or_null(mb.d_csw_line_edge_n0),
          raw_or_null(mb.d_csw_line_edge_n1),
          raw_or_null(mb.d_csw_line_prev_edge),
          raw_or_null(mb.d_csw_line_next_edge),
          raw_or_null(mb.d_csw_line_prev_sign),
          raw_or_null(mb.d_csw_line_next_sign),
          raw_or_null(mb.d_csw_line_cand_offsets),
          raw_or_null(mb.d_csw_line_cand_edges),
          raw_or_null(mb.d_csw_line_cand_signs),
          raw_or_null(mb.d_unique_face_cell_a),
          raw_or_null(mb.d_unique_face_cell_b),
          raw_or_null(mb.d_boundary_face_cell), n_internal, n_edges,
          d_edge_psi);
      cuda_check(cudaGetLastError(),
                 "csw98 edge AV psi pre-pass launch failed");
    }
    const bool face_adj_mirror_ok =
        mb.d_face_adj_csr_offsets.size() == mb.face_adj_csr_offsets.size() &&
        mb.d_face_adj_csr_indices.size() == mb.face_adj_csr_indices.size();
    // Hand-built topologies (tests) may lack the device mirrors — fall back
    // to the per-call staging with identical bytes.
    thrust::device_vector<int> face_adj_offsets_fallback;
    thrust::device_vector<int> face_adj_indices_fallback;
    if (!face_adj_mirror_ok) {
      face_adj_offsets_fallback.assign(mb.face_adj_csr_offsets.begin(),
                                       mb.face_adj_csr_offsets.end());
      face_adj_indices_fallback.assign(mb.face_adj_csr_indices.begin(),
                                       mb.face_adj_csr_indices.end());
    }
    const thrust::device_vector<int>& face_adj_offsets_use =
        face_adj_mirror_ok ? mb.d_face_adj_csr_offsets
                           : face_adj_offsets_fallback;
    const thrust::device_vector<int>& face_adj_indices_use =
        face_adj_mirror_ok ? mb.d_face_adj_csr_indices
                           : face_adj_indices_fallback;
    core::DeviceArray<std::uint8_t> d_cell_nverts{
        "compatible_av_csw:edge_force_cell_nverts"};
    const std::uint8_t* d_cell_nverts_ptr =
        upload_cell_nverts_if_nonquad(d_cell_nverts, state);
    if (use_polar_slaving) {
      TENRYU_ASSERT(params.pole_floor_enabled == 0 &&
                        params.pole_desens_enabled == 0,
                    "polar slaving cannot be combined with pole floor/desens");
      polar_slaving_runtime =
          &ensure_csw_polar_slaving_runtime(state, cfg);
      polar_slaving_view =
          prepare_csw_polar_slaving(
              *polar_slaving_runtime, state.x_r.data(), state.x_z.data(), v_r,
              v_z, state.rho.data(), cell_cs.data(), hydro_active,
              state.mesh.multiblock_cell_node_csr_offsets.data(),
              state.mesh.multiblock_cell_node_csr_indices.data(),
              raw_or_null(face_adj_offsets_use),
              raw_or_null(face_adj_indices_use),
              d_cell_nverts_ptr, d_edge_psi,
              state.mesh.multiblock_cell_edge_csr_offsets.data(),
              state.mesh.multiblock_cell_edge_csr_edges.data(),
              raw_or_null(mb.d_csw_line_edge_n0),
              raw_or_null(mb.d_csw_line_edge_n1), params, aw_planar,
              d_limiter_miss, false, nullptr, n_nodes, state.dt, state.step)
              .view;
    }
    if (n_internal > 0) {
      const int blocks = (n_internal + 255) / 256;
      if (use_csw98) {
        if (use_polar_slaving) {
          csw98_multiblock_internal_force_polar_slaving_kernel<<<blocks, 256>>>(
              state.edge_force_av_r.data(), state.edge_force_av_z.data(),
              d_work_av_edge, d_compressive_count,
              d_negative_work_count, state.x_r.data(), state.x_z.data(), v_r,
              v_z, node_mass, state.rho.data(), cell_cs.data(), hydro_active,
              raw_or_null(mb.d_unique_face_cell_a),
              raw_or_null(mb.d_unique_face_cell_b),
              raw_or_null(mb.d_unique_face_local_a),
              raw_or_null(mb.d_unique_face_local_b),
              state.mesh.multiblock_cell_node_csr_offsets.data(),
              state.mesh.multiblock_cell_node_csr_indices.data(),
              raw_or_null(face_adj_offsets_use),
              raw_or_null(face_adj_indices_use), d_cell_nverts_ptr,
              d_edge_psi,
              state.mesh.multiblock_cell_edge_csr_offsets.data(),
              state.mesh.multiblock_cell_edge_csr_edges.data(),
              raw_or_null(mb.d_csw_line_edge_n0),
              raw_or_null(mb.d_csw_line_edge_n1), n_internal,
              params, aw_planar, polar_slaving_view, d_limiter_miss);
        } else {
          csw98_multiblock_internal_force_kernel<<<blocks, 256>>>(
              state.edge_force_av_r.data(),
              state.edge_force_av_z.data(),
              d_work_av_edge,
              d_compressive_count,
              d_negative_work_count,
              state.x_r.data(),
              state.x_z.data(),
              v_r,
              v_z,
              node_mass,
              state.rho.data(),
              cell_cs.data(),
              hydro_active,
              raw_or_null(mb.d_unique_face_cell_a),
              raw_or_null(mb.d_unique_face_cell_b),
              raw_or_null(mb.d_unique_face_local_a),
              raw_or_null(mb.d_unique_face_local_b),
              state.mesh.multiblock_cell_node_csr_offsets.data(),
              state.mesh.multiblock_cell_node_csr_indices.data(),
              raw_or_null(face_adj_offsets_use),
              raw_or_null(face_adj_indices_use),
              d_cell_nverts_ptr,
              d_edge_psi,
              state.mesh.multiblock_cell_edge_csr_offsets.data(),
              state.mesh.multiblock_cell_edge_csr_edges.data(),
              raw_or_null(mb.d_csw_line_edge_n0),
              raw_or_null(mb.d_csw_line_edge_n1),
              n_internal,
              params,
              aw_planar,
              d_limiter_miss);
        }
      } else {
        csw_multiblock_internal_force_kernel<<<blocks, 256>>>(
            state.edge_force_av_r.data(),
            state.edge_force_av_z.data(),
            d_work_av_edge,
            d_compressive_count,
            d_negative_work_count,
            state.x_r.data(),
            state.x_z.data(),
            v_r,
            v_z,
            state.rho.data(),
            cell_cs.data(),
            hydro_active,
            raw_or_null(mb.d_unique_face_cell_a),
            raw_or_null(mb.d_unique_face_cell_b),
            raw_or_null(mb.d_unique_face_local_a),
            raw_or_null(mb.d_unique_face_local_b),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            raw_or_null(face_adj_offsets_use),
            raw_or_null(face_adj_indices_use),
            d_cell_nverts_ptr,
            n_internal,
            params,
            aw_planar);
      }
    }
    if (n_boundary > 0) {
      const int blocks = (n_boundary + 255) / 256;
      if (use_csw98) {
        csw98_multiblock_boundary_force_kernel<<<blocks, 256>>>(
            state.edge_force_av_r.data(),
            state.edge_force_av_z.data(),
            d_work_av_edge,
            d_compressive_count,
            d_negative_work_count,
            state.x_r.data(),
            state.x_z.data(),
            v_r,
            v_z,
            node_mass,
            state.rho.data(),
            cell_cs.data(),
            hydro_active,
            raw_or_null(mb.d_boundary_face_cell),
            raw_or_null(mb.d_boundary_face_local),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            raw_or_null(face_adj_offsets_use),
            raw_or_null(face_adj_indices_use),
            d_cell_nverts_ptr,
            d_edge_psi,
            state.mesh.multiblock_cell_edge_csr_offsets.data(),
            state.mesh.multiblock_cell_edge_csr_edges.data(),
            raw_or_null(mb.d_csw_line_edge_n0),
            raw_or_null(mb.d_csw_line_edge_n1),
            n_internal,
            n_boundary,
            params,
            aw_planar,
            d_limiter_miss);
      } else {
        csw_multiblock_boundary_force_kernel<<<blocks, 256>>>(
            state.edge_force_av_r.data(),
            state.edge_force_av_z.data(),
            d_work_av_edge,
            d_compressive_count,
            d_negative_work_count,
            state.x_r.data(),
            state.x_z.data(),
            v_r,
            v_z,
            state.rho.data(),
            cell_cs.data(),
            hydro_active,
            raw_or_null(mb.d_boundary_face_cell),
            raw_or_null(mb.d_boundary_face_local),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            raw_or_null(face_adj_offsets_use),
            raw_or_null(face_adj_indices_use),
            d_cell_nverts_ptr,
            n_internal,
            n_boundary,
            params,
            aw_planar);
      }
    }
    // I1-B pole tangential damper: extra pairwise tangential edge drag on
    // near-pole shell columns, accumulated into the same edge AV buffers so
    // momentum kick and corrector time-centered work closure are inherited.
    // Env-gated default-off (TENRYU_I1B_POLE_DAMPER).
    if (pole_damper_enabled() && n_internal > 0) {
      const mesh::BlockInfo* shell_block = nullptr;
      int north_fan_rows = 0;
      for (const auto& block : mb.blocks) {
        if (block.role == mesh::BlockRole::POLAR_SHELL) {
          shell_block = &block;
        } else if (block.role == mesh::BlockRole::NORTH_FAN) {
          north_fan_rows = block.n_i_cells;
        }
      }
      if (shell_block != nullptr && shell_block->n_j_cells >= 4 &&
          shell_block->n_i_cells >= 1) {
        int q_begin_active = 0;
        if (state.central_pseudo_core.built && mb.has_trifan_cap) {
          q_begin_active =
              std::max(0, state.central_pseudo_core.member_ring_count -
                              mb.n_cap - north_fan_rows);
        }
        static bool pole_damper_logged = false;
        int* d_pair_count = nullptr;
        if (!pole_damper_logged) {
          d_pair_count = static_cast<int*>(core::device_scratch_acquire(
              "compatible_av_csw:edge_force_pole_damper_pair_count",
              sizeof(int)));
          cuda_check(cudaMemset(d_pair_count, 0, sizeof(int)),
                     "pole damper: zero pair count failed");
        }
        const int blocks_damper = (n_internal + 255) / 256;
        pole_damper_internal_force_kernel<<<blocks_damper, 256>>>(
            state.edge_force_av_r.data(),
            state.edge_force_av_z.data(),
            d_work_av_edge,
            state.x_r.data(),
            state.x_z.data(),
            v_r,
            v_z,
            state.rho.data(),
            cell_cs.data(),
            hydro_active,
            raw_or_null(mb.d_unique_face_cell_a),
            raw_or_null(mb.d_unique_face_cell_b),
            raw_or_null(mb.d_unique_face_local_a),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            d_cell_nverts_ptr,
            n_internal,
            shell_block->owned_node_begin,
            shell_block->n_j_cells,
            shell_block->n_i_cells + 1,
            q_begin_active,
            pole_damper_m(),
            pole_damper_q_lo(),
            pole_damper_q_hi(),
            pole_damper_ctheta(),
            d_pair_count);
        if (!pole_damper_logged) {
          int pair_count = 0;
          cuda_check(cudaMemcpy(&pair_count, d_pair_count, sizeof(int),
                                cudaMemcpyDeviceToHost),
                     "pole damper: copy pair count failed");
          core::log_info(
              "[pole_damper] first fire: C_theta=" +
              format_csw_scientific(pole_damper_ctheta()) +
              " m_columns=" + std::to_string(pole_damper_m()) +
              " q_band=[" + std::to_string(pole_damper_q_lo()) + "," +
              std::to_string(pole_damper_q_hi()) + "]" +
              " q_begin_active=" + std::to_string(q_begin_active) +
              " damped_pairs=" + std::to_string(pair_count));
          pole_damper_logged = true;
        }
      }
    }
  } else {
    const int n_edges = state.mesh.topo.n_edges();
    TENRYU_ASSERT(state.edge_force_av_r.size() == static_cast<std::size_t>(n_edges),
                  "CSW edge AV structured edge buffer size mismatch");
    const int blocks = (n_edges + 255) / 256;
    if (use_csw98) {
      csw98_structured_force_kernel<<<blocks, 256>>>(
          state.edge_force_av_r.data(),
          state.edge_force_av_z.data(),
          state.work_av_per_cell.data(),
          d_compressive_count,
          d_negative_work_count,
          state.x_r.data(),
          state.x_z.data(),
          v_r,
          v_z,
          node_mass,
          state.rho.data(),
          cell_cs.data(),
          hydro_active,
          state.mesh.topo.nr,
          state.mesh.topo.nz,
          params,
          aw_planar,
          aw_axisline_av,
          aw_axis_slave_first_i,
          aw_axis_slave_theta0_active,
          aw_axis_slave_theta_pi_active);
    } else {
      csw_structured_force_kernel<<<blocks, 256>>>(
          state.edge_force_av_r.data(),
          state.edge_force_av_z.data(),
          state.work_av_per_cell.data(),
          d_compressive_count,
          d_negative_work_count,
          state.x_r.data(),
          state.x_z.data(),
          v_r,
          v_z,
          state.rho.data(),
          cell_cs.data(),
          hydro_active,
          state.mesh.topo.nr,
          state.mesh.topo.nz,
          params,
          aw_planar,
          aw_axisline_av,
          aw_axis_slave_first_i,
          aw_axis_slave_theta0_active,
          aw_axis_slave_theta_pi_active);
    }
  }

  if (csw98_limiter_diag) {
    int counters[2] = {0, 0};
    cuda_check(cudaMemcpy(counters, d_limiter_miss, 2 * sizeof(int),
                          cudaMemcpyDeviceToHost),
               "csw98 AV: copy limiter diag counters failed");
    if (counters[1] > 0) {
      // report the first launch where the limiter actually ran
      // (evals > 0); stay armed through vacuous t~0 launches.
      core::log_info(
          "[csw98-limiter-diag] first active launch: evals=" +
          std::to_string(counters[1]) +
          " unresolved continuation members: " +
          std::to_string(counters[0]));
      csw98_limiter_diag_logged = true;
    }
  }

  const int blocks_cells = (n_cells + 255) / 256;
  if (is_multiblock) {
    gather_multiblock_work_av_kernel<<<blocks_cells, 256>>>(
        state.work_av_per_cell.data(), d_work_av_edge,
        state.mesh.multiblock_cell_edge_csr_offsets.data(),
        state.mesh.multiblock_cell_edge_csr_edges.data(),
        state.mesh.multiblock_cell_edge_csr_side.data(), n_cells);
  }
  clip_negative_work_kernel<<<blocks_cells, 256>>>(
      state.work_av_per_cell.data(), d_negative_work_count, n_cells);
  cuda_check(cudaGetLastError(), "CSW edge AV kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "CSW edge AV kernel execution failed");

  int compressive_count = 0;
  int negative_count = 0;
  cuda_check(cudaMemcpy(&compressive_count, d_compressive_count, sizeof(int),
                        cudaMemcpyDeviceToHost),
             "CSW edge AV: copy compressive count failed");
  cuda_check(cudaMemcpy(&negative_count, d_negative_work_count, sizeof(int),
                        cudaMemcpyDeviceToHost),
             "CSW edge AV: copy negative work count failed");

  if (polar_slaving_runtime != nullptr) {
    record_csw_polar_slaving_diagnostics(*polar_slaving_runtime, state.step);
  }

  state.count_edge_compressive_edges_step = compressive_count;
  if (negative_count > 0) {
    core::log_warning("CSW edge AV clipped " + std::to_string(negative_count) +
                      " negative/non-finite AV work contribution(s)");
  }
  // [av_activity] read-only export (env TENRYU_I1B_AV_ACTIVITY_LOG_EVERY=N,
  // default off): per-call sums of the per-call-zeroed work_av buffer and
  // the fire/negative counters, aggregated per hydro STEP (the launcher
  // runs k>=2 times per step: predictor + corrector accelerations, plus
  // any retry/subcycle re-evaluations -- calls= reports the completed
  // step's call count). A completed step is detected on the first call
  // with a different state.step; its totals print with one-step latency
  // when completed_step % N == 0.
  static const int activity_every = [] {
    const char* raw = std::getenv("TENRYU_I1B_AV_ACTIVITY_LOG_EVERY");
    const int v = raw != nullptr && raw[0] != '\0' ? std::atoi(raw) : 0;
    return v > 0 ? v : 0;
  }();
  if (activity_every > 0) {
    static long long act_prev_step = -1;
    static double act_prev_t = 0.0;
    static int act_calls = 0;
    static long long act_fire = 0;
    static long long act_neg = 0;
    static long double act_work_step = 0.0L;
    static long double act_work_cum = 0.0L;
    const double call_work = thrust::reduce(
        thrust::device,
        thrust::device_pointer_cast(state.work_av_per_cell.data()),
        thrust::device_pointer_cast(state.work_av_per_cell.data() + n_cells),
        0.0,
        thrust::plus<double>());
    if (act_prev_step >= 0 && state.step != act_prev_step) {
      if (act_prev_step % activity_every == 0) {
        std::fprintf(stderr,
                     "[av_activity] step=%lld t=%.6e calls=%d "
                     "fire_count=%lld work_av_step=%.6e work_av_cum=%.10e "
                     "negative_work_count=%lld\n",
                     act_prev_step,
                     act_prev_t,
                     act_calls,
                     act_fire,
                     static_cast<double>(act_work_step),
                     static_cast<double>(act_work_cum),
                     act_neg);
      }
      act_calls = 0;
      act_fire = 0;
      act_neg = 0;
      act_work_step = 0.0L;
    }
    act_prev_step = state.step;
    act_prev_t = state.t;
    ++act_calls;
    act_fire += compressive_count;
    act_neg += negative_count;
    act_work_step += call_work;
    act_work_cum += call_work;
  }
}

double compute_edge_accel_displacement_dt(
    const core::State& state,
    const core::Config& cfg,
    EdgeAccelDisplacementArgmin* argmin) {
  if (argmin != nullptr) {
    *argmin = EdgeAccelDisplacementArgmin{};
  }
  if (cfg.main.dim != 2 ||
      !cfg.numerics.dt.edge_accel_displacement_cfl_enabled ||
      state.rho.empty() ||
      state.node_accel_r.size() != state.v_r.size() ||
      state.node_accel_z.size() != state.v_z.size()) {
    return std::numeric_limits<double>::infinity();
  }
  TENRYU_ASSERT(state.x_r.size() == state.v_r.size() &&
                    state.x_z.size() == state.v_z.size() &&
                    state.x_r.size() == state.x_z.size(),
                "edge acceleration displacement CFL requires matching node "
                "arrays");
  const double coefficient = cfg.numerics.dt.cfl_hydro;
  if (!(coefficient > 0.0)) {
    return std::numeric_limits<double>::infinity();
  }

  const bool multiblock = mesh::mesh_topo_is_multiblock(cfg.mesh);
  if (multiblock) {
    TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                  "edge acceleration displacement CFL multiblock topology "
                  "metadata missing");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                      state.rho.size() + 1U,
                  "edge acceleration displacement CFL requires multiblock "
                  "cell-node CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                      state.rho.size() *
                          static_cast<std::size_t>(state.mesh.corner_stride),
                  "edge acceleration displacement CFL requires multiblock "
                  "cell-node CSR indices");
  }
  const int n_cells = static_cast<int>(state.rho.size());
  const int corner_stride =
      multiblock ? state.mesh.corner_stride : mesh::kMeshTopoCellStorageSlots;
  const int n_candidates = n_cells * corner_stride;
  const int* cell_node_offsets =
      multiblock ? state.mesh.multiblock_cell_node_csr_offsets.data() : nullptr;
  const int* cell_node_indices =
      multiblock ? state.mesh.multiblock_cell_node_csr_indices.data() : nullptr;

  double* d_min_dt = static_cast<double*>(core::device_scratch_acquire(
      "compatible_av_csw:edge_accel_displacement_min_dt", sizeof(double)));
  const double inf = std::numeric_limits<double>::infinity();
  cuda_check(cudaMemcpy(d_min_dt, &inf, sizeof(double), cudaMemcpyHostToDevice),
             "edge acceleration displacement CFL: init min_dt failed");
  std::int8_t* d_active = upload_hydro_active_or_null(state, cfg);
  core::DeviceArray<std::uint8_t> d_cell_nverts{
      "compatible_av_csw:edge_accel_displacement_cell_nverts"};
  const std::uint8_t* d_cell_nverts_ptr =
      upload_cell_nverts_if_nonquad(d_cell_nverts, state);

  edge_accel_displacement_cfl_kernel
      <<<(n_candidates + 255) / 256, 256>>>(
          d_min_dt, nullptr, 0.0, state.x_r.data(), state.x_z.data(),
          state.v_r.data(), state.v_z.data(), state.node_accel_r.data(),
          state.node_accel_z.data(), d_active, cell_node_offsets,
          cell_node_indices, d_cell_nverts_ptr, n_cells, state.mesh.topo.nz,
          corner_stride, multiblock, coefficient);
  cuda_check(cudaGetLastError(),
             "edge acceleration displacement CFL kernel launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "edge acceleration displacement CFL kernel execution failed");

  double min_dt = std::numeric_limits<double>::infinity();
  cuda_check(cudaMemcpy(&min_dt, d_min_dt, sizeof(double),
                        cudaMemcpyDeviceToHost),
             "edge acceleration displacement CFL: copy min_dt failed");
  if (argmin != nullptr && std::isfinite(min_dt)) {
    int* d_winner_index = static_cast<int*>(core::device_scratch_acquire(
        "compatible_av_csw:edge_accel_displacement_winner_index",
        sizeof(int)));
    EdgeAccelDisplacementWinner* d_winner =
        static_cast<EdgeAccelDisplacementWinner*>(core::device_scratch_acquire(
            "compatible_av_csw:edge_accel_displacement_winner",
            sizeof(EdgeAccelDisplacementWinner)));
    const int no_winner = std::numeric_limits<int>::max();
    const EdgeAccelDisplacementWinner empty_winner;
    cuda_check(cudaMemcpy(d_winner_index, &no_winner, sizeof(int),
                          cudaMemcpyHostToDevice),
               "edge acceleration displacement CFL: init winner index failed");
    cuda_check(cudaMemcpy(d_winner, &empty_winner,
                          sizeof(EdgeAccelDisplacementWinner),
                          cudaMemcpyHostToDevice),
               "edge acceleration displacement CFL: init winner failed");

    edge_accel_displacement_cfl_kernel
        <<<(n_candidates + 255) / 256, 256>>>(
            d_min_dt, d_winner_index, min_dt, state.x_r.data(),
            state.x_z.data(), state.v_r.data(), state.v_z.data(),
            state.node_accel_r.data(), state.node_accel_z.data(), d_active,
            cell_node_offsets, cell_node_indices, d_cell_nverts_ptr, n_cells,
            state.mesh.topo.nz, corner_stride, multiblock, coefficient);
    cuda_check(
        cudaGetLastError(),
        "edge acceleration displacement CFL winner kernel launch failed");
    cuda_check(
        cudaDeviceSynchronize(),
        "edge acceleration displacement CFL winner kernel execution failed");

    edge_accel_displacement_winner_values_kernel<<<1, 1>>>(
        d_winner, d_winner_index, state.x_r.data(), state.x_z.data(),
        state.v_r.data(), state.v_z.data(), state.node_accel_r.data(),
        state.node_accel_z.data(), d_active, cell_node_offsets,
        cell_node_indices, d_cell_nverts_ptr, n_cells, state.mesh.topo.nz,
        corner_stride, multiblock, coefficient);
    cuda_check(cudaGetLastError(),
               "edge acceleration displacement CFL winner values kernel "
               "launch failed");
    cuda_check(cudaDeviceSynchronize(),
               "edge acceleration displacement CFL winner values kernel "
               "execution failed");

    EdgeAccelDisplacementWinner winner;
    cuda_check(cudaMemcpy(&winner, d_winner,
                          sizeof(EdgeAccelDisplacementWinner),
                          cudaMemcpyDeviceToHost),
               "edge acceleration displacement CFL: copy winner failed");
    if (winner.cell >= 0) {
      argmin->dt = min_dt;
      argmin->edge_id = winner.edge_id;
      argmin->cell_id = winner.cell;
      if (state.mesh.topo.multiblock.has_value()) {
        const auto& mb = *state.mesh.topo.multiblock;
        if (mb.cell_id_stable.size() == state.rho.size()) {
          argmin->cell_id =
              mb.cell_id_stable[static_cast<std::size_t>(winner.cell)];
        }
      }
      argmin->node0 = winner.node0;
      argmin->node1 = winner.node1;
      argmin->length = winner.length;
      argmin->c_e = winner.c_e;
      argmin->a_e = winner.a_e;
      argmin->coefficient = coefficient;
    }
  }
  if (d_active != nullptr) {
    cuda_check(cudaFree(d_active),
               "edge acceleration displacement CFL: cudaFree hydro_active "
               "failed");
  }
  return min_dt;
}

double compute_csw_edge_av_cfl_dt(const core::State& state,
                                  const core::Config& cfg,
                                  const bool aw_axis_slave_theta0_active,
                                  const bool aw_axis_slave_theta_pi_active,
                                  CswEdgeAvCflArgmin* argmin) {
  if (argmin != nullptr) {
    *argmin = CswEdgeAvCflArgmin{};
  }
  if (cfg.main.dim != 2 ||
      (cfg.numerics.hydro.av_model != core::AvModel::CswEdge &&
       cfg.numerics.hydro.av_model != core::AvModel::CswEdgeCsw98)) {
    return std::numeric_limits<double>::infinity();
  }
  const bool use_csw98 =
      cfg.numerics.hydro.av_model == core::AvModel::CswEdgeCsw98;
  const bool use_polar_slaving =
      use_csw98 && cfg.numerics.hydro.csw_polar_slaving_enabled;
  const bool use_polar_slaving_stiffness_cfl =
      use_polar_slaving &&
      cfg.numerics.hydro.csw_polar_slaving_av_stiffness_cfl_enabled;
  const bool aw_planar = cfg.numerics.hydro.aw_compatible_force_work;
  const bool aw_axisline_av = aw_axisline_av_enabled();
  const int aw_axis_slave_first_i =
      (aw_axis_slave_theta0_active || aw_axis_slave_theta_pi_active)
          ? detect_structured_aw_axis_slave_first_i(state)
          : 0;
  const double coefficient = cfg.numerics.hydro.av_cfl_coefficient;
  if (!(coefficient > 0.0) || state.rho.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  TENRYU_ASSERT(state.x_r.size() == state.v_r.size() &&
                    state.x_z.size() == state.v_z.size() &&
                    state.x_r.size() == state.x_z.size(),
                "CSW edge AV CFL requires matching node arrays");

  double* d_min_dt = nullptr;
  d_min_dt = static_cast<double*>(core::device_scratch_acquire(
      "compatible_av_csw:cfl_min_dt", sizeof(double)));
  const double inf = std::numeric_limits<double>::infinity();
  cuda_check(cudaMemcpy(d_min_dt, &inf, sizeof(double), cudaMemcpyHostToDevice),
             "CSW edge AV CFL: init min_dt failed");
  std::int8_t* d_active = upload_hydro_active_or_null(state, cfg);
  core::DeviceArray<std::uint8_t> d_cell_nverts{
      "compatible_av_csw:cfl_cell_nverts"};
  const std::uint8_t* d_cell_nverts_ptr =
      upload_cell_nverts_if_nonquad(d_cell_nverts, state);
  CswKernelParams params;
  params.c1 = cfg.numerics.hydro.av_linear;
  params.c2 = cfg.numerics.hydro.av_quadratic;
  params.degenerate_side_floor_rel =
      cfg.numerics.hydro.csw98_degenerate_side_floor_rel;
  params.damper_impulse_beta =
      cfg.numerics.hydro.csw98_damper_impulse_beta;
  params.dt = 0.0;
  params.gamma = cfg.materials.materials.empty()
                     ? (5.0 / 3.0)
                     : cfg.materials.materials.front().ideal_gas_gamma;
  params.limiter_enabled = cfg.numerics.hydro.csw_limiter_enabled ? 1 : 0;
  params.axis_mirror_limiter =
      cfg.numerics.hydro.csw_axis_mirror_limiter ? 1 : 0;
  configure_csw98_edge_diag(&params);
  params.pole_floor_enabled =
      cfg.numerics.hydro.csw_pole_floor_enabled ? 1 : 0;
  params.pole_floor_sigma0 = cfg.numerics.hydro.csw_pole_floor_sigma0;
  params.pole_floor_theta0 = cfg.numerics.hydro.csw_pole_floor_theta0_rad;
  params.pole_floor_thetaf = cfg.numerics.hydro.csw_pole_floor_thetaf_rad;
  params.pole_desens_enabled =
      cfg.numerics.hydro.csw_pole_desens_enabled ? 1 : 0;
  params.pole_desens_alpha = cfg.numerics.hydro.csw_pole_desens_alpha;
  params.pole_desens_theta0 = cfg.numerics.hydro.csw_pole_desens_theta0_rad;
  params.pole_desens_thetaf = cfg.numerics.hydro.csw_pole_desens_thetaf_rad;
  params.axisline_av_enabled = aw_axisline_av_enabled() ? 1 : 0;
  params.axisline_d1prime =
      cfg.numerics.hydro.csw98_axisline_av_mode == "d1prime" ? 1 : 0;
  params.axisline_d1prime_cfl =
      cfg.numerics.hydro.csw98_axisline_d1prime_cfl_enabled ? 1 : 0;
  params.limiter_shock_floor =
      cfg.numerics.hydro.csw98_limiter_shock_floor_enabled ? 1 : 0;
  params.shock_limiter_floor = cfg.numerics.hydro.csw_shock_limiter_floor;
  params.axistouch_av_off = axistouch_av_off_enabled() ? 1 : 0;
  params.axis_eps_cm = cfg.numerics.axis_eps_cm;
  static const double av_cfl_diag_below = [] {
    const char* const raw = std::getenv("TENRYU_AV_CFL_DIAG_BELOW");
    return raw != nullptr && raw[0] != '\0' ? std::strtod(raw, nullptr) : 0.0;
  }();
  params.av_cfl_diag_below = av_cfl_diag_below;
  core::NodeField1D cfl_node_mass{"compatible_av_csw:cfl_node_mass"};
  const double* node_mass = nullptr;
  if (use_csw98 && params.axisline_d1prime != 0 &&
      params.axisline_av_enabled == 0 && aw_planar &&
      state.corner_mass_initialized &&
      state.corner_mass.size() ==
          state.rho.size() *
              static_cast<std::size_t>(state.corner_stride)) {
    TENRYU_ASSERT(state.cs.size() == state.rho.size(),
                  "D1-prime AV CFL requires cell sound speed");
    const int n_nodes = static_cast<int>(state.x_r.size());
    const int n_cells = static_cast<int>(state.rho.size());
    cfl_node_mass.reset(n_nodes);
    cuda_check(cudaMemset(cfl_node_mass.data(), 0,
                          static_cast<std::size_t>(n_nodes) * sizeof(double)),
               "D1-prime AV CFL: zero node mass failed");
    if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
      TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                        state.rho.size() + 1U,
                    "D1-prime AV CFL requires multiblock cell-node CSR "
                    "offsets");
      TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                        state.rho.size() *
                            static_cast<std::size_t>(state.corner_stride),
                    "D1-prime AV CFL requires multiblock cell-node CSR "
                    "indices");
      // atomicAdd arrival order makes these node masses LSB-nondeterministic;
      // they feed only this dt bound, consistent with the documented 2D RZ
      // noise-band policy.
      csw98_d1prime_node_mass_scatter_kernel
          <<<(n_cells + 255) / 256, 256>>>(
              cfl_node_mass.data(), state.corner_mass.data(),
              state.mesh.multiblock_cell_node_csr_offsets.data(),
              state.mesh.multiblock_cell_node_csr_indices.data(),
              d_cell_nverts_ptr, state.corner_stride, n_cells);
      cuda_check(cudaGetLastError(),
                 "D1-prime AV CFL: node mass scatter launch failed");
    } else {
      tenryu::hydro::launch_compute_node_mass_for_cfl_2d(
          cfl_node_mass.data(), state, cfg, d_cell_nverts_ptr, d_active);
    }
    node_mass = cfl_node_mass.data();
  }
  CswPolarSlavingDeviceView polar_slaving_view;
  double* d_edge_psi = nullptr;
  double polar_slaving_stiffness_dt =
      std::numeric_limits<double>::infinity();
  double polar_slaving_stiffness_lambda = 0.0;
  double polar_slaving_stiffness_sigma = 0.0;
  int polar_slaving_stiffness_winner_node = -1;

  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                  "CSW edge AV CFL multiblock topology metadata missing");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                      state.rho.size() + 1U,
                  "CSW edge AV CFL requires multiblock cell-node CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                      state.rho.size() *
                          static_cast<std::size_t>(state.mesh.corner_stride),
                  "CSW edge AV CFL requires multiblock cell-node CSR indices");
    const auto& mb = *state.mesh.topo.multiblock;
    const int n_internal = static_cast<int>(mb.unique_internal_faces.size());
    const int n_boundary = static_cast<int>(mb.boundary_faces.size());
    const int n_edges = n_internal + n_boundary;
    if (use_csw98) {
      const std::size_t cell_edge_incidence_count =
          2U * static_cast<std::size_t>(n_internal) +
          static_cast<std::size_t>(n_boundary);
      TENRYU_ASSERT(
          state.mesh.multiblock_cell_edge_csr_offsets.size() ==
              state.rho.size() + 1U,
          "CSW edge AV CFL requires multiblock cell-edge CSR offsets");
      TENRYU_ASSERT(state.mesh.multiblock_cell_edge_csr_edges.size() ==
                        cell_edge_incidence_count,
                    "CSW edge AV CFL requires multiblock cell-edge CSR entries");
      TENRYU_ASSERT(state.cs.size() == state.rho.size(),
                    "csw98 edge AV CFL requires cell sound speed");
      TENRYU_ASSERT(
          n_edges > 0 &&
              mb.d_csw_line_edge_n0.size() ==
                  static_cast<std::size_t>(n_edges) &&
              mb.d_csw_line_edge_n1.size() ==
                  static_cast<std::size_t>(n_edges) &&
              mb.d_csw_line_prev_edge.size() ==
                  static_cast<std::size_t>(n_edges) &&
              mb.d_csw_line_next_edge.size() ==
                  static_cast<std::size_t>(n_edges) &&
              mb.d_csw_line_prev_sign.size() ==
                  static_cast<std::size_t>(n_edges) &&
              mb.d_csw_line_next_sign.size() ==
                  static_cast<std::size_t>(n_edges) &&
              mb.d_csw_line_cand_offsets.size() ==
                  2U * static_cast<std::size_t>(n_edges) + 1U &&
              !mb.d_csw_line_cand_edges.empty() &&
              mb.d_csw_line_cand_edges.size() ==
                  mb.d_csw_line_cand_signs.size(),
          "csw98 edge AV CFL requires non-empty phase-line device topology");
      TENRYU_ASSERT(
          mb.d_unique_face_cell_a.size() ==
                  static_cast<std::size_t>(n_internal) &&
              mb.d_unique_face_cell_b.size() ==
                  static_cast<std::size_t>(n_internal) &&
              mb.d_boundary_face_cell.size() ==
                  static_cast<std::size_t>(n_boundary),
          "csw98 edge AV CFL requires face-cell device topology");
      d_edge_psi = static_cast<double*>(core::device_scratch_acquire(
          "compatible_av_csw:edge_psi",
          static_cast<std::size_t>(n_edges) * sizeof(double)));
      csw98_edge_psi_prepass_kernel<<<(n_edges + 255) / 256, 256>>>(
          state.x_r.data(), state.x_z.data(), state.v_r.data(), state.v_z.data(),
          state.cs.data(), raw_or_null(mb.d_csw_line_edge_n0),
          raw_or_null(mb.d_csw_line_edge_n1),
          raw_or_null(mb.d_csw_line_prev_edge),
          raw_or_null(mb.d_csw_line_next_edge),
          raw_or_null(mb.d_csw_line_prev_sign),
          raw_or_null(mb.d_csw_line_next_sign),
          raw_or_null(mb.d_csw_line_cand_offsets),
          raw_or_null(mb.d_csw_line_cand_edges),
          raw_or_null(mb.d_csw_line_cand_signs),
          raw_or_null(mb.d_unique_face_cell_a),
          raw_or_null(mb.d_unique_face_cell_b),
          raw_or_null(mb.d_boundary_face_cell), n_internal, n_edges,
          d_edge_psi);
      cuda_check(cudaGetLastError(),
                 "csw98 edge AV CFL psi pre-pass launch failed");
    }
    const bool face_adj_mirror_ok =
        mb.d_face_adj_csr_offsets.size() == mb.face_adj_csr_offsets.size() &&
        mb.d_face_adj_csr_indices.size() == mb.face_adj_csr_indices.size();
    // Hand-built topologies (tests) may lack the device mirrors — fall back
    // to the per-call staging with identical bytes.
    thrust::device_vector<int> face_adj_offsets_fallback;
    thrust::device_vector<int> face_adj_indices_fallback;
    if (!face_adj_mirror_ok) {
      face_adj_offsets_fallback.assign(mb.face_adj_csr_offsets.begin(),
                                       mb.face_adj_csr_offsets.end());
      face_adj_indices_fallback.assign(mb.face_adj_csr_indices.begin(),
                                       mb.face_adj_csr_indices.end());
    }
    const thrust::device_vector<int>& face_adj_offsets_use =
        face_adj_mirror_ok ? mb.d_face_adj_csr_offsets
                           : face_adj_offsets_fallback;
    const thrust::device_vector<int>& face_adj_indices_use =
        face_adj_mirror_ok ? mb.d_face_adj_csr_indices
                           : face_adj_indices_fallback;
    if (use_polar_slaving) {
      TENRYU_ASSERT(state.cs.size() == state.rho.size(),
                    "polar-slaved AV CFL requires cell sound speed");
      CswPolarSlavingRuntime& runtime =
          ensure_csw_polar_slaving_runtime(state, cfg);
      const CswPolarSlavingPreparation preparation =
          prepare_csw_polar_slaving(
              runtime, state.x_r.data(), state.x_z.data(), state.v_r.data(),
              state.v_z.data(), state.rho.data(), state.cs.data(), d_active,
              state.mesh.multiblock_cell_node_csr_offsets.data(),
              state.mesh.multiblock_cell_node_csr_indices.data(),
              raw_or_null(face_adj_offsets_use),
              raw_or_null(face_adj_indices_use),
              d_cell_nverts_ptr, d_edge_psi,
              state.mesh.multiblock_cell_edge_csr_offsets.data(),
              state.mesh.multiblock_cell_edge_csr_edges.data(),
              raw_or_null(mb.d_csw_line_edge_n0),
              raw_or_null(mb.d_csw_line_edge_n1), params, aw_planar, nullptr,
              use_polar_slaving_stiffness_cfl,
              use_polar_slaving_stiffness_cfl ? &state.node_planar_mass
                                               : nullptr,
              static_cast<int>(state.x_r.size()), state.dt, state.step);
      polar_slaving_view = preparation.view;
      polar_slaving_stiffness_dt = preparation.stiffness_dt;
      polar_slaving_stiffness_lambda = preparation.stiffness_lambda;
      polar_slaving_stiffness_sigma = preparation.stiffness_sigma;
      polar_slaving_stiffness_winner_node =
          preparation.stiffness_winner_node;
    }
    if (n_internal > 0) {
      const int blocks = (n_internal + 255) / 256;
      if (use_csw98) {
        if (use_polar_slaving) {
          csw98_multiblock_internal_cfl_polar_slaving_kernel<<<blocks, 256>>>(
              d_min_dt, state.x_r.data(), state.x_z.data(), state.v_r.data(),
              state.v_z.data(), node_mass, state.rho.data(), state.cs.data(),
              d_active,
              raw_or_null(mb.d_unique_face_cell_a),
              raw_or_null(mb.d_unique_face_cell_b),
              raw_or_null(mb.d_unique_face_local_a),
              raw_or_null(mb.d_unique_face_local_b),
              state.mesh.multiblock_cell_node_csr_offsets.data(),
              state.mesh.multiblock_cell_node_csr_indices.data(),
              raw_or_null(face_adj_offsets_use),
              raw_or_null(face_adj_indices_use), d_cell_nverts_ptr,
              d_edge_psi, n_internal,
              params, coefficient, aw_planar, polar_slaving_view);
        } else {
          csw98_multiblock_internal_cfl_kernel<<<blocks, 256>>>(
              d_min_dt,
              state.x_r.data(),
              state.x_z.data(),
              state.v_r.data(),
              state.v_z.data(),
              node_mass,
              state.rho.data(),
              state.cs.data(),
              d_active,
              raw_or_null(mb.d_unique_face_cell_a),
              raw_or_null(mb.d_unique_face_cell_b),
              raw_or_null(mb.d_unique_face_local_a),
              raw_or_null(mb.d_unique_face_local_b),
              state.mesh.multiblock_cell_node_csr_offsets.data(),
              state.mesh.multiblock_cell_node_csr_indices.data(),
              raw_or_null(face_adj_offsets_use),
              raw_or_null(face_adj_indices_use),
              d_cell_nverts_ptr,
              d_edge_psi,
              state.mesh.multiblock_cell_edge_csr_offsets.data(),
              state.mesh.multiblock_cell_edge_csr_edges.data(),
              raw_or_null(mb.d_csw_line_edge_n0),
              raw_or_null(mb.d_csw_line_edge_n1),
              n_internal,
              params,
              coefficient,
              aw_planar);
        }
      } else {
        csw_multiblock_internal_cfl_kernel<<<blocks, 256>>>(
            d_min_dt,
            state.x_r.data(),
            state.x_z.data(),
            state.v_r.data(),
            state.v_z.data(),
            d_active,
            raw_or_null(mb.d_unique_face_cell_a),
            raw_or_null(mb.d_unique_face_cell_b),
            raw_or_null(mb.d_unique_face_local_a),
            raw_or_null(mb.d_unique_face_local_b),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            d_cell_nverts_ptr,
            n_internal,
            coefficient);
      }
    }
    if (n_boundary > 0) {
      const int blocks = (n_boundary + 255) / 256;
      if (use_csw98) {
        csw98_multiblock_boundary_cfl_kernel<<<blocks, 256>>>(
            d_min_dt,
            state.x_r.data(),
            state.x_z.data(),
            state.v_r.data(),
            state.v_z.data(),
            node_mass,
            state.rho.data(),
            state.cs.data(),
            d_active,
            raw_or_null(mb.d_boundary_face_cell),
            raw_or_null(mb.d_boundary_face_local),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            raw_or_null(face_adj_offsets_use),
            raw_or_null(face_adj_indices_use),
            d_cell_nverts_ptr,
            d_edge_psi,
            state.mesh.multiblock_cell_edge_csr_offsets.data(),
            state.mesh.multiblock_cell_edge_csr_edges.data(),
            raw_or_null(mb.d_csw_line_edge_n0),
            raw_or_null(mb.d_csw_line_edge_n1),
            n_internal,
            n_boundary,
            params,
            coefficient,
            aw_planar);
      } else {
        csw_multiblock_boundary_cfl_kernel<<<blocks, 256>>>(
            d_min_dt,
            state.x_r.data(),
            state.x_z.data(),
            state.v_r.data(),
            state.v_z.data(),
            d_active,
            raw_or_null(mb.d_boundary_face_cell),
            raw_or_null(mb.d_boundary_face_local),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            d_cell_nverts_ptr,
            n_boundary,
            coefficient);
      }
    }
  } else {
    const int n_edges = state.mesh.topo.n_edges();
    const int blocks = (n_edges + 255) / 256;
    if (use_csw98) {
      csw98_structured_cfl_kernel<<<blocks, 256>>>(
          d_min_dt,
          nullptr,
          0.0,
          state.x_r.data(),
          state.x_z.data(),
          state.v_r.data(),
          state.v_z.data(),
          node_mass,
          state.rho.data(),
          state.cs.data(),
          d_active,
          state.mesh.topo.nr,
          state.mesh.topo.nz,
          params,
          coefficient,
          aw_planar,
          aw_axisline_av,
          aw_axis_slave_first_i,
          aw_axis_slave_theta0_active,
          aw_axis_slave_theta_pi_active);
    } else {
      csw_structured_cfl_kernel<<<blocks, 256>>>(
          d_min_dt,
          state.x_r.data(),
          state.x_z.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_active,
          state.mesh.topo.nr,
          state.mesh.topo.nz,
          coefficient,
          aw_planar,
          aw_axisline_av,
          aw_axis_slave_first_i,
          aw_axis_slave_theta0_active,
          aw_axis_slave_theta_pi_active);
    }
  }
  cuda_check(cudaGetLastError(), "CSW edge AV CFL kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "CSW edge AV CFL kernel execution failed");

  double min_dt = std::numeric_limits<double>::infinity();
  cuda_check(cudaMemcpy(&min_dt, d_min_dt, sizeof(double), cudaMemcpyDeviceToHost),
             "CSW edge AV CFL: copy min_dt failed");
  const double edge_min_dt = min_dt;
  if (use_polar_slaving_stiffness_cfl) {
    min_dt = std::min(min_dt, polar_slaving_stiffness_dt);
  }
  if (argmin != nullptr && min_dt == polar_slaving_stiffness_dt &&
      min_dt < edge_min_dt) {
    argmin->dt = min_dt;
    argmin->polar_slaving_stiffness = true;
    argmin->polar_slaving_node = polar_slaving_stiffness_winner_node;
    argmin->polar_slaving_lambda = polar_slaving_stiffness_lambda;
    argmin->polar_slaving_sigma = polar_slaving_stiffness_sigma;
  }
  if (argmin != nullptr && std::isfinite(edge_min_dt) &&
      min_dt == edge_min_dt) {
    int* d_winner_index = nullptr;
    CswEdgeAvCflWinner* d_winner = nullptr;
    d_winner_index = static_cast<int*>(core::device_scratch_acquire(
        "compatible_av_csw:cfl_lineage_winner_index", sizeof(int)));
    d_winner = static_cast<CswEdgeAvCflWinner*>(core::device_scratch_acquire(
        "compatible_av_csw:cfl_lineage_winner",
        sizeof(CswEdgeAvCflWinner)));
    const int no_winner = std::numeric_limits<int>::max();
    const CswEdgeAvCflWinner empty_winner;
    cuda_check(cudaMemcpy(d_winner_index, &no_winner, sizeof(int),
                          cudaMemcpyHostToDevice),
               "CSW edge AV CFL lineage: init winner_index failed");
    cuda_check(cudaMemcpy(d_winner, &empty_winner, sizeof(CswEdgeAvCflWinner),
                          cudaMemcpyHostToDevice),
               "CSW edge AV CFL lineage: init winner failed");

    if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
      const auto& mb = *state.mesh.topo.multiblock;
      const int n_internal = static_cast<int>(mb.unique_internal_faces.size());
      const int n_boundary = static_cast<int>(mb.boundary_faces.size());
      const bool face_adj_mirror_ok =
          mb.d_face_adj_csr_offsets.size() == mb.face_adj_csr_offsets.size() &&
          mb.d_face_adj_csr_indices.size() == mb.face_adj_csr_indices.size();
      // Hand-built topologies (tests) may lack the device mirrors — fall back
      // to the per-call staging with identical bytes.
      thrust::device_vector<int> face_adj_offsets_fallback;
      thrust::device_vector<int> face_adj_indices_fallback;
      if (!face_adj_mirror_ok) {
        face_adj_offsets_fallback.assign(mb.face_adj_csr_offsets.begin(),
                                         mb.face_adj_csr_offsets.end());
        face_adj_indices_fallback.assign(mb.face_adj_csr_indices.begin(),
                                         mb.face_adj_csr_indices.end());
      }
      const thrust::device_vector<int>& face_adj_offsets_use =
          face_adj_mirror_ok ? mb.d_face_adj_csr_offsets
                             : face_adj_offsets_fallback;
      const thrust::device_vector<int>& face_adj_indices_use =
          face_adj_mirror_ok ? mb.d_face_adj_csr_indices
                             : face_adj_indices_fallback;
      if (n_internal > 0) {
        const int blocks = (n_internal + 255) / 256;
        if (use_csw98) {
          if (use_polar_slaving) {
            csw98_multiblock_internal_cfl_winner_index_polar_slaving_kernel
                <<<blocks, 256>>>(
                    d_winner_index, d_min_dt, state.x_r.data(),
                    state.x_z.data(), state.v_r.data(), state.v_z.data(),
                    node_mass, state.rho.data(), state.cs.data(), d_active,
                    raw_or_null(mb.d_unique_face_cell_a),
                    raw_or_null(mb.d_unique_face_cell_b),
                    raw_or_null(mb.d_unique_face_local_a),
                    raw_or_null(mb.d_unique_face_local_b),
                    state.mesh.multiblock_cell_node_csr_offsets.data(),
                    state.mesh.multiblock_cell_node_csr_indices.data(),
                    raw_or_null(face_adj_offsets_use),
                    raw_or_null(face_adj_indices_use), d_cell_nverts_ptr,
                    d_edge_psi, n_internal, params, coefficient, aw_planar,
                    edge_min_dt, polar_slaving_view);
          } else {
            csw98_multiblock_internal_cfl_winner_index_kernel<<<blocks, 256>>>(
                d_winner_index, d_min_dt, state.x_r.data(), state.x_z.data(),
                state.v_r.data(), state.v_z.data(), node_mass,
                state.rho.data(), state.cs.data(), d_active,
                raw_or_null(mb.d_unique_face_cell_a),
                raw_or_null(mb.d_unique_face_cell_b),
                raw_or_null(mb.d_unique_face_local_a),
                raw_or_null(mb.d_unique_face_local_b),
                state.mesh.multiblock_cell_node_csr_offsets.data(),
                state.mesh.multiblock_cell_node_csr_indices.data(),
                raw_or_null(face_adj_offsets_use),
                raw_or_null(face_adj_indices_use),
                d_cell_nverts_ptr, d_edge_psi,
                state.mesh.multiblock_cell_edge_csr_offsets.data(),
                state.mesh.multiblock_cell_edge_csr_edges.data(),
                raw_or_null(mb.d_csw_line_edge_n0),
                raw_or_null(mb.d_csw_line_edge_n1), n_internal, params,
                coefficient, aw_planar, edge_min_dt);
          }
        } else {
          csw_multiblock_internal_cfl_winner_index_kernel<<<blocks, 256>>>(
              d_winner_index, state.x_r.data(), state.x_z.data(),
              state.v_r.data(), state.v_z.data(), d_active,
              raw_or_null(mb.d_unique_face_cell_a),
              raw_or_null(mb.d_unique_face_cell_b),
              raw_or_null(mb.d_unique_face_local_a),
              raw_or_null(mb.d_unique_face_local_b),
              state.mesh.multiblock_cell_node_csr_offsets.data(),
              state.mesh.multiblock_cell_node_csr_indices.data(),
              d_cell_nverts_ptr, n_internal, coefficient, edge_min_dt);
        }
      }
      if (n_boundary > 0) {
        const int blocks = (n_boundary + 255) / 256;
        if (use_csw98) {
          csw98_multiblock_boundary_cfl_winner_index_kernel<<<blocks, 256>>>(
              d_winner_index, d_min_dt, state.x_r.data(), state.x_z.data(),
              state.v_r.data(), state.v_z.data(), node_mass,
              state.rho.data(), state.cs.data(), d_active,
              raw_or_null(mb.d_boundary_face_cell),
              raw_or_null(mb.d_boundary_face_local),
              state.mesh.multiblock_cell_node_csr_offsets.data(),
              state.mesh.multiblock_cell_node_csr_indices.data(),
              raw_or_null(face_adj_offsets_use),
              raw_or_null(face_adj_indices_use),
              d_cell_nverts_ptr, d_edge_psi,
              state.mesh.multiblock_cell_edge_csr_offsets.data(),
              state.mesh.multiblock_cell_edge_csr_edges.data(),
              raw_or_null(mb.d_csw_line_edge_n0),
              raw_or_null(mb.d_csw_line_edge_n1), n_internal, n_boundary,
              params, coefficient, aw_planar, edge_min_dt);
        } else {
          csw_multiblock_boundary_cfl_winner_index_kernel<<<blocks, 256>>>(
              d_winner_index, state.x_r.data(), state.x_z.data(),
              state.v_r.data(), state.v_z.data(), d_active,
              raw_or_null(mb.d_boundary_face_cell),
              raw_or_null(mb.d_boundary_face_local),
              state.mesh.multiblock_cell_node_csr_offsets.data(),
              state.mesh.multiblock_cell_node_csr_indices.data(),
              d_cell_nverts_ptr, n_internal, n_boundary, coefficient,
              edge_min_dt);
        }
      }
      cuda_check(cudaGetLastError(),
                 "CSW edge AV CFL lineage winner index kernel launch failed");
      cuda_check(cudaDeviceSynchronize(),
                 "CSW edge AV CFL lineage winner index kernel execution failed");
      int* d_cell_block_id = nullptr;
      if (mb.cell_block_id.size() == state.rho.size()) {
        d_cell_block_id = static_cast<int*>(core::device_scratch_acquire(
            "compatible_av_csw:cfl_lineage_cell_block_id",
            mb.cell_block_id.size() * sizeof(int)));
        cuda_check(cudaMemcpy(d_cell_block_id, mb.cell_block_id.data(),
                              mb.cell_block_id.size() * sizeof(int),
                              cudaMemcpyHostToDevice),
                   "CSW edge AV CFL lineage: copy cell_block_id failed");
      }
      if (use_csw98) {
        csw98_multiblock_cfl_winner_values_kernel<<<1, 1>>>(
            d_winner, d_winner_index, state.x_r.data(), state.x_z.data(),
            state.v_r.data(), state.v_z.data(),
            raw_or_null(mb.d_unique_face_cell_a),
            raw_or_null(mb.d_unique_face_cell_b),
            raw_or_null(mb.d_unique_face_local_a),
            raw_or_null(mb.d_unique_face_local_b),
            raw_or_null(mb.d_boundary_face_cell),
            raw_or_null(mb.d_boundary_face_local),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            d_cell_nverts_ptr, d_cell_block_id, n_internal, n_boundary,
            params, aw_planar, edge_min_dt);
      } else {
        csw_multiblock_cfl_winner_values_kernel<<<1, 1>>>(
            d_winner, d_winner_index, state.x_r.data(), state.x_z.data(),
            state.v_r.data(), state.v_z.data(), d_active,
            raw_or_null(mb.d_unique_face_cell_a),
            raw_or_null(mb.d_unique_face_cell_b),
            raw_or_null(mb.d_unique_face_local_a),
            raw_or_null(mb.d_unique_face_local_b),
            raw_or_null(mb.d_boundary_face_cell),
            raw_or_null(mb.d_boundary_face_local),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            d_cell_nverts_ptr, d_cell_block_id, n_internal, n_boundary);
      }
      cuda_check(cudaGetLastError(),
                 "CSW edge AV CFL lineage winner values kernel launch failed");
      cuda_check(cudaDeviceSynchronize(),
                 "CSW edge AV CFL lineage winner values kernel execution failed");
    } else {
      const int n_edges = state.mesh.topo.n_edges();
      const int blocks = (n_edges + 255) / 256;
      if (use_csw98) {
        csw98_structured_cfl_kernel<<<blocks, 256>>>(
            d_min_dt, d_winner_index, edge_min_dt, state.x_r.data(),
            state.x_z.data(), state.v_r.data(), state.v_z.data(), node_mass,
            state.rho.data(), state.cs.data(), d_active,
            state.mesh.topo.nr, state.mesh.topo.nz, params, coefficient,
            aw_planar, aw_axisline_av, aw_axis_slave_first_i,
            aw_axis_slave_theta0_active, aw_axis_slave_theta_pi_active);
        cuda_check(cudaGetLastError(),
                   "CSW98 edge AV CFL lineage winner index kernel launch failed");
        cuda_check(cudaDeviceSynchronize(),
                   "CSW98 edge AV CFL lineage winner index kernel execution failed");
        csw98_structured_cfl_winner_values_kernel<<<1, 1>>>(
            d_winner, d_winner_index, state.x_r.data(), state.x_z.data(),
            state.v_r.data(), state.v_z.data(), state.mesh.topo.nr,
            state.mesh.topo.nz, params, aw_planar, aw_axisline_av,
            aw_axis_slave_first_i, aw_axis_slave_theta0_active,
            aw_axis_slave_theta_pi_active, edge_min_dt);
      } else {
        csw_structured_cfl_winner_index_kernel<<<blocks, 256>>>(
            d_winner_index, state.x_r.data(), state.x_z.data(),
            state.v_r.data(), state.v_z.data(), d_active, state.mesh.topo.nr,
            state.mesh.topo.nz, coefficient, edge_min_dt, aw_planar,
            aw_axisline_av, aw_axis_slave_first_i,
            aw_axis_slave_theta0_active, aw_axis_slave_theta_pi_active);
        cuda_check(cudaGetLastError(),
                   "CSW edge AV CFL lineage winner index kernel launch failed");
        cuda_check(cudaDeviceSynchronize(),
                   "CSW edge AV CFL lineage winner index kernel execution failed");
        csw_structured_cfl_winner_values_kernel<<<1, 1>>>(
            d_winner, d_winner_index, state.x_r.data(), state.x_z.data(),
            state.v_r.data(), state.v_z.data(), d_active, state.mesh.topo.nr,
            state.mesh.topo.nz, aw_planar, aw_axisline_av,
            aw_axis_slave_first_i, aw_axis_slave_theta0_active,
            aw_axis_slave_theta_pi_active);
      }
      cuda_check(cudaGetLastError(),
                 "CSW edge AV CFL lineage winner values kernel launch failed");
      cuda_check(cudaDeviceSynchronize(),
                 "CSW edge AV CFL lineage winner values kernel execution failed");
    }
    CswEdgeAvCflWinner winner;
    cuda_check(cudaMemcpy(&winner, d_winner, sizeof(CswEdgeAvCflWinner),
                          cudaMemcpyDeviceToHost),
               "CSW edge AV CFL lineage: copy winner failed");
    if (winner.cell_a >= 0) {
      const bool damping_eigenvalue_winner =
          use_csw98 && std::isfinite(winner.raw_dt);
      argmin->dt =
          damping_eigenvalue_winner ? winner.raw_dt : edge_min_dt;
      argmin->edge_id = winner.edge_id;
      argmin->cell_id = winner.cell_a;
      if (state.mesh.topo.multiblock.has_value()) {
        const auto& mb = *state.mesh.topo.multiblock;
        if (mb.cell_id_stable.size() == state.rho.size()) {
          argmin->cell_id =
              mb.cell_id_stable[static_cast<std::size_t>(winner.cell_a)];
        }
      }
      argmin->node0 = winner.n0;
      argmin->node1 = winner.n1;
      argmin->length = winner.dx;
      argmin->du = damping_eigenvalue_winner
                       ? winner.du
                       : (use_csw98 ? coefficient * winner.dx / edge_min_dt
                                    : winner.du);
      argmin->coefficient = coefficient;
    }
  }
  if (use_csw98 && mesh::mesh_topo_is_multiblock(cfg.mesh) &&
      std::isfinite(min_dt) && min_dt < 1.0e-15 &&
      cfg.numerics.debug.trace_mesh_motion) {
    int* d_winner_index = nullptr;
    CswEdgeAvCflWinner* d_winner = nullptr;
    d_winner_index = static_cast<int*>(core::device_scratch_acquire(
        "compatible_av_csw:cfl_csw98_diag_winner_index", sizeof(int)));
    d_winner = static_cast<CswEdgeAvCflWinner*>(core::device_scratch_acquire(
        "compatible_av_csw:cfl_csw98_diag_winner",
        sizeof(CswEdgeAvCflWinner)));
    const int no_winner = std::numeric_limits<int>::max();
    const CswEdgeAvCflWinner empty_winner;
    cuda_check(cudaMemcpy(d_winner_index, &no_winner, sizeof(int),
                          cudaMemcpyHostToDevice),
               "CSW98 edge AV CFL: init winner_index failed");
    cuda_check(cudaMemcpy(d_winner, &empty_winner, sizeof(CswEdgeAvCflWinner),
                          cudaMemcpyHostToDevice),
               "CSW98 edge AV CFL: init winner failed");

    const auto& mb = *state.mesh.topo.multiblock;
    const int n_internal = static_cast<int>(mb.unique_internal_faces.size());
    const int n_boundary = static_cast<int>(mb.boundary_faces.size());
    const bool face_adj_mirror_ok =
        mb.d_face_adj_csr_offsets.size() == mb.face_adj_csr_offsets.size() &&
        mb.d_face_adj_csr_indices.size() == mb.face_adj_csr_indices.size();
    // Hand-built topologies (tests) may lack the device mirrors — fall back
    // to the per-call staging with identical bytes.
    thrust::device_vector<int> face_adj_offsets_fallback;
    thrust::device_vector<int> face_adj_indices_fallback;
    if (!face_adj_mirror_ok) {
      face_adj_offsets_fallback.assign(mb.face_adj_csr_offsets.begin(),
                                       mb.face_adj_csr_offsets.end());
      face_adj_indices_fallback.assign(mb.face_adj_csr_indices.begin(),
                                       mb.face_adj_csr_indices.end());
    }
    const thrust::device_vector<int>& face_adj_offsets_use =
        face_adj_mirror_ok ? mb.d_face_adj_csr_offsets
                           : face_adj_offsets_fallback;
    const thrust::device_vector<int>& face_adj_indices_use =
        face_adj_mirror_ok ? mb.d_face_adj_csr_indices
                           : face_adj_indices_fallback;
    if (n_internal > 0) {
      const int blocks = (n_internal + 255) / 256;
      if (use_polar_slaving) {
        csw98_multiblock_internal_cfl_winner_index_polar_slaving_kernel
            <<<blocks, 256>>>(
                d_winner_index, d_min_dt, state.x_r.data(), state.x_z.data(),
                state.v_r.data(), state.v_z.data(), node_mass,
                state.rho.data(), state.cs.data(), d_active,
                raw_or_null(mb.d_unique_face_cell_a),
                raw_or_null(mb.d_unique_face_cell_b),
                raw_or_null(mb.d_unique_face_local_a),
                raw_or_null(mb.d_unique_face_local_b),
                state.mesh.multiblock_cell_node_csr_offsets.data(),
                state.mesh.multiblock_cell_node_csr_indices.data(),
                raw_or_null(face_adj_offsets_use),
                raw_or_null(face_adj_indices_use), d_cell_nverts_ptr,
                d_edge_psi, n_internal, params, coefficient, aw_planar, min_dt,
                polar_slaving_view);
      } else {
        csw98_multiblock_internal_cfl_winner_index_kernel<<<blocks, 256>>>(
            d_winner_index,
            d_min_dt,
            state.x_r.data(),
            state.x_z.data(),
            state.v_r.data(),
            state.v_z.data(),
            node_mass,
            state.rho.data(),
            state.cs.data(),
            d_active,
            raw_or_null(mb.d_unique_face_cell_a),
            raw_or_null(mb.d_unique_face_cell_b),
            raw_or_null(mb.d_unique_face_local_a),
            raw_or_null(mb.d_unique_face_local_b),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            raw_or_null(face_adj_offsets_use),
            raw_or_null(face_adj_indices_use),
            d_cell_nverts_ptr,
            d_edge_psi,
            state.mesh.multiblock_cell_edge_csr_offsets.data(),
            state.mesh.multiblock_cell_edge_csr_edges.data(),
            raw_or_null(mb.d_csw_line_edge_n0),
            raw_or_null(mb.d_csw_line_edge_n1),
            n_internal,
            params,
            coefficient,
            aw_planar,
            min_dt);
      }
    }
    if (n_boundary > 0) {
      const int blocks = (n_boundary + 255) / 256;
      csw98_multiblock_boundary_cfl_winner_index_kernel<<<blocks, 256>>>(
          d_winner_index,
          d_min_dt,
          state.x_r.data(),
          state.x_z.data(),
          state.v_r.data(),
          state.v_z.data(),
          node_mass,
          state.rho.data(),
          state.cs.data(),
          d_active,
          raw_or_null(mb.d_boundary_face_cell),
          raw_or_null(mb.d_boundary_face_local),
          state.mesh.multiblock_cell_node_csr_offsets.data(),
          state.mesh.multiblock_cell_node_csr_indices.data(),
          raw_or_null(face_adj_offsets_use),
          raw_or_null(face_adj_indices_use),
          d_cell_nverts_ptr,
          d_edge_psi,
          state.mesh.multiblock_cell_edge_csr_offsets.data(),
          state.mesh.multiblock_cell_edge_csr_edges.data(),
          raw_or_null(mb.d_csw_line_edge_n0),
          raw_or_null(mb.d_csw_line_edge_n1),
          n_internal,
          n_boundary,
          params,
          coefficient,
          aw_planar,
          min_dt);
    }
    cuda_check(cudaGetLastError(),
               "CSW98 edge AV CFL winner index kernel launch failed");
    cuda_check(cudaDeviceSynchronize(),
               "CSW98 edge AV CFL winner index kernel execution failed");

    int* d_cell_block_id = nullptr;
    if (mb.cell_block_id.size() == state.rho.size()) {
      d_cell_block_id = static_cast<int*>(core::device_scratch_acquire(
          "compatible_av_csw:cfl_csw98_diag_cell_block_id",
          mb.cell_block_id.size() * sizeof(int)));
      cuda_check(cudaMemcpy(d_cell_block_id, mb.cell_block_id.data(),
                            mb.cell_block_id.size() * sizeof(int),
                            cudaMemcpyHostToDevice),
                 "CSW98 edge AV CFL: copy cell_block_id failed");
    }
    csw98_multiblock_cfl_winner_values_kernel<<<1, 1>>>(
        d_winner,
        d_winner_index,
        state.x_r.data(),
        state.x_z.data(),
        state.v_r.data(),
        state.v_z.data(),
        raw_or_null(mb.d_unique_face_cell_a),
        raw_or_null(mb.d_unique_face_cell_b),
        raw_or_null(mb.d_unique_face_local_a),
        raw_or_null(mb.d_unique_face_local_b),
        raw_or_null(mb.d_boundary_face_cell),
        raw_or_null(mb.d_boundary_face_local),
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        d_cell_nverts_ptr,
        d_cell_block_id,
        n_internal,
        n_boundary,
        params,
        aw_planar,
        min_dt);
    cuda_check(cudaGetLastError(),
               "CSW98 edge AV CFL winner values kernel launch failed");
    cuda_check(cudaDeviceSynchronize(),
               "CSW98 edge AV CFL winner values kernel execution failed");

    CswEdgeAvCflWinner winner;
    cuda_check(cudaMemcpy(&winner, d_winner, sizeof(CswEdgeAvCflWinner),
                          cudaMemcpyDeviceToHost),
               "CSW98 edge AV CFL: copy winner failed");
    if (winner.cell_a >= 0) {
      pole_angular_derefine::assert_active_cell(
          state, winner.cell_a, "CSW98 edge AV CFL winner");
      int stable_cell = winner.cell_a;
      if (mb.cell_id_stable.size() == state.rho.size()) {
        stable_cell =
            mb.cell_id_stable[static_cast<std::size_t>(winner.cell_a)];
      }
      core::log_warning("[csw98-edge-av-cfl-winner] min_dt=" +
                        format_csw_scientific(min_dt) +
                        " n0=" + std::to_string(winner.n0) +
                        " n1=" + std::to_string(winner.n1) +
                        " r0=" + format_csw_scientific(winner.r0) +
                        " z0=" + format_csw_scientific(winner.z0) +
                        " r1=" + format_csw_scientific(winner.r1) +
                        " z1=" + format_csw_scientific(winner.z1) +
                        " dx=" + format_csw_scientific(winner.dx) +
                        " du=" + format_csw_scientific(winner.du) +
                        " stable_cell=" + std::to_string(stable_cell) +
                        " block_id=" + std::to_string(winner.block_id));
    } else {
      core::log_warning(
          "[csw98-edge-av-cfl-winner] UNAVAILABLE min_dt=" +
          format_csw_scientific(min_dt));
    }
  }
  if (!use_csw98 && std::isfinite(min_dt) && min_dt < 1.0e-15 &&
      cfg.numerics.debug.trace_mesh_motion) {
    int* d_winner_index = nullptr;
    CswEdgeAvCflWinner* d_winner = nullptr;
    d_winner_index = static_cast<int*>(core::device_scratch_acquire(
        "compatible_av_csw:cfl_csw_diag_winner_index", sizeof(int)));
    d_winner = static_cast<CswEdgeAvCflWinner*>(core::device_scratch_acquire(
        "compatible_av_csw:cfl_csw_diag_winner",
        sizeof(CswEdgeAvCflWinner)));
    const int no_winner = std::numeric_limits<int>::max();
    const CswEdgeAvCflWinner empty_winner;
    cuda_check(cudaMemcpy(d_winner_index, &no_winner, sizeof(int),
                          cudaMemcpyHostToDevice),
               "CSW edge AV CFL: init winner_index failed");
    cuda_check(cudaMemcpy(d_winner, &empty_winner, sizeof(CswEdgeAvCflWinner),
                          cudaMemcpyHostToDevice),
               "CSW edge AV CFL: init winner failed");
    if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
      const auto& mb = *state.mesh.topo.multiblock;
      const int n_internal = static_cast<int>(mb.unique_internal_faces.size());
      const int n_boundary = static_cast<int>(mb.boundary_faces.size());
      if (n_internal > 0) {
        const int blocks = (n_internal + 255) / 256;
        csw_multiblock_internal_cfl_winner_index_kernel<<<blocks, 256>>>(
            d_winner_index,
            state.x_r.data(),
            state.x_z.data(),
            state.v_r.data(),
            state.v_z.data(),
            d_active,
            raw_or_null(mb.d_unique_face_cell_a),
            raw_or_null(mb.d_unique_face_cell_b),
            raw_or_null(mb.d_unique_face_local_a),
            raw_or_null(mb.d_unique_face_local_b),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            d_cell_nverts_ptr,
            n_internal,
            coefficient,
            min_dt);
      }
      if (n_boundary > 0) {
        const int blocks = (n_boundary + 255) / 256;
        csw_multiblock_boundary_cfl_winner_index_kernel<<<blocks, 256>>>(
            d_winner_index,
            state.x_r.data(),
            state.x_z.data(),
            state.v_r.data(),
            state.v_z.data(),
            d_active,
            raw_or_null(mb.d_boundary_face_cell),
            raw_or_null(mb.d_boundary_face_local),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            d_cell_nverts_ptr,
            n_internal,
            n_boundary,
            coefficient,
            min_dt);
      }
      cuda_check(cudaGetLastError(),
                 "CSW edge AV CFL winner index kernel launch failed");
      cuda_check(cudaDeviceSynchronize(),
                 "CSW edge AV CFL winner index kernel execution failed");
      int* d_cell_block_id = nullptr;
      if (mb.cell_block_id.size() == state.rho.size()) {
        d_cell_block_id = static_cast<int*>(core::device_scratch_acquire(
            "compatible_av_csw:cfl_csw_diag_cell_block_id",
            mb.cell_block_id.size() * sizeof(int)));
        cuda_check(cudaMemcpy(d_cell_block_id, mb.cell_block_id.data(),
                              mb.cell_block_id.size() * sizeof(int),
                              cudaMemcpyHostToDevice),
                   "CSW edge AV CFL: copy cell_block_id failed");
      }
      csw_multiblock_cfl_winner_values_kernel<<<1, 1>>>(
          d_winner,
          d_winner_index,
          state.x_r.data(),
          state.x_z.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_active,
          raw_or_null(mb.d_unique_face_cell_a),
          raw_or_null(mb.d_unique_face_cell_b),
          raw_or_null(mb.d_unique_face_local_a),
          raw_or_null(mb.d_unique_face_local_b),
          raw_or_null(mb.d_boundary_face_cell),
          raw_or_null(mb.d_boundary_face_local),
          state.mesh.multiblock_cell_node_csr_offsets.data(),
          state.mesh.multiblock_cell_node_csr_indices.data(),
          d_cell_nverts_ptr,
          d_cell_block_id,
          n_internal,
          n_boundary);
      cuda_check(cudaGetLastError(),
                 "CSW edge AV CFL winner values kernel launch failed");
      cuda_check(cudaDeviceSynchronize(),
                 "CSW edge AV CFL winner values kernel execution failed");
    } else {
      const int n_edges = state.mesh.topo.n_edges();
      const int blocks = (n_edges + 255) / 256;
      csw_structured_cfl_winner_index_kernel<<<blocks, 256>>>(
          d_winner_index,
          state.x_r.data(),
          state.x_z.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_active,
          state.mesh.topo.nr,
          state.mesh.topo.nz,
          coefficient,
          min_dt,
          aw_planar,
          aw_axisline_av,
          aw_axis_slave_first_i,
          aw_axis_slave_theta0_active,
          aw_axis_slave_theta_pi_active);
      cuda_check(cudaGetLastError(),
                 "CSW edge AV CFL winner index kernel launch failed");
      cuda_check(cudaDeviceSynchronize(),
                 "CSW edge AV CFL winner index kernel execution failed");
      csw_structured_cfl_winner_values_kernel<<<1, 1>>>(
          d_winner,
          d_winner_index,
          state.x_r.data(),
          state.x_z.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_active,
          state.mesh.topo.nr,
          state.mesh.topo.nz,
          aw_planar,
          aw_axisline_av,
          aw_axis_slave_first_i,
          aw_axis_slave_theta0_active,
          aw_axis_slave_theta_pi_active);
      cuda_check(cudaGetLastError(),
                 "CSW edge AV CFL winner values kernel launch failed");
      cuda_check(cudaDeviceSynchronize(),
                 "CSW edge AV CFL winner values kernel execution failed");
    }
    CswEdgeAvCflWinner winner;
    cuda_check(cudaMemcpy(&winner, d_winner, sizeof(CswEdgeAvCflWinner),
                          cudaMemcpyDeviceToHost),
               "CSW edge AV CFL: copy winner failed");
    pole_angular_derefine::assert_active_cell(
        state, winner.cell_a, "CSW edge AV CFL winner");
    core::log_warning("[csw-edge-av-cfl-winner] min_dt=" +
                      format_csw_scientific(min_dt) +
                      " n0=" + std::to_string(winner.n0) +
                      " n1=" + std::to_string(winner.n1) +
                      " r0=" + format_csw_scientific(winner.r0) +
                      " z0=" + format_csw_scientific(winner.z0) +
                      " r1=" + format_csw_scientific(winner.r1) +
                      " z1=" + format_csw_scientific(winner.z1) +
                      " dx=" + format_csw_scientific(winner.dx) +
                      " du=" + format_csw_scientific(winner.du) +
                      " cell_a=" + std::to_string(winner.cell_a) +
                      " block_id=" + std::to_string(winner.block_id));
  }
  if (d_active != nullptr) {
    cuda_check(cudaFree(d_active), "CSW edge AV CFL: cudaFree hydro_active failed");
  }
  return min_dt;
}

}  // namespace tenryu::hydro::compatible
