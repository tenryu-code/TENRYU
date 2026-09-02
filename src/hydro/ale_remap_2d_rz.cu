#include "hydro/ale_remap_2d_rz.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>

#include "burn/burn_constants.hpp"
#include "core/constants.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/kernel_guard.hpp"
#include "diagnostics/energy_budget.hpp"
#include "hydro/ale_driver.cuh"
#include "hydro/ale_identity_diag.hpp"
#include "hydro/ale_reference_diagnostics.cuh"
#include "hydro/ale_remap.cuh"
#include "hydro/ale_scaled_reference.cuh"
#include "hydro/ale_tracking_reference.cuh"
#include "hydro/anti_hourglass.cuh"
#include "hydro/boundary_2d.hpp"
#include "hydro/cap_energy_audit.hpp"
#include "hydro/cfl.hpp"
#include "hydro/compatible_force_work_2d.cuh"
#include "hydro/conservation_audit.hpp"
#include "hydro/corner_mass_remap_audit.hpp"
#include "hydro/ke_fixup_deposit.hpp"
#include "hydro/central_pseudo_core.cuh"
#include "hydro/eos_context.hpp"
#include "hydro/mesh_motion_trace.hpp"
#include "hydro/optionb_velocity_remap.cuh"
#include "hydro/oriented_swept_volume.cuh"
#include "hydro/pentagon_geometry.cuh"
#include "hydro/pole_axis_constraints.cuh"
#include "hydro/pole_angular_derefine.cuh"
#include "hydro/remap_eos_closure.cuh"
#include "hydro/rz_corner_mass.cuh"
#include "hydro/remap_corner_distribution.hpp"
#include "mesh/mesh.hpp"
#include "mesh/rz_moments.cuh"
#include "parallel/reduction.hpp"

namespace tenryu::hydro::ale {

namespace {

constexpr int kRemapDispatchAuditCounterCount =
    static_cast<int>(RemapDispatchAuditCounter::Count);
constexpr double kRemapDispatchAuditFeatherRadiusMin = 1.0e-3;
constexpr double kRemapDispatchAuditFeatherRadiusMax = 2.5e-3;
constexpr double kAwTwoPi =
    6.283185307179586476925286766559005768394338798750211641949;

bool csr_corner_distribution_same_bits(const double& a, const double& b) {
  return std::memcmp(&a, &b, sizeof(double)) == 0;
}

void csr_corner_distribution_subpolygon(
    const double* const r,
    const double* const z,
    const int nverts,
    const int corner,
    double* const sub_r,
    double* const sub_z) {
  double center_r = 0.0;
  double center_z = 0.0;
  for (int k = 0; k < nverts; ++k) {
    center_r += r[k];
    center_z += z[k];
  }
  center_r /= static_cast<double>(nverts);
  center_z /= static_cast<double>(nverts);
  const int next = (corner + 1 == nverts) ? 0 : corner + 1;
  const int previous = (corner == 0) ? nverts - 1 : corner - 1;
  sub_r[0] = r[corner];
  sub_z[0] = z[corner];
  sub_r[1] = 0.5 * (r[corner] + r[next]);
  sub_z[1] = 0.5 * (z[corner] + z[next]);
  sub_r[2] = center_r;
  sub_z[2] = center_z;
  sub_r[3] = 0.5 * (r[previous] + r[corner]);
  sub_z[3] = 0.5 * (z[previous] + z[corner]);
}

void csr_corner_distribution_cell_geometry(
    const double* const r,
    const double* const z,
    const int nverts,
    double* const corner_volume,
    double* const corner_centroid_r,
    double* const corner_centroid_z) {
  TENRYU_ASSERT(nverts == 3 || nverts == 4 || nverts == 5,
                "CSR corner distribution requires triangle, quad, or "
                "pentagon cells");
  if (nverts == 3) {
    for (int k = 0; k < 3; ++k) {
      double sub_r[4] = {};
      double sub_z[4] = {};
      csr_corner_distribution_subpolygon(
          r, z, nverts, k, sub_r, sub_z);
      corner_volume[k] =
          std::fabs(rz::rz_polygon_volume_exact(sub_r, sub_z, 4));
    }
  } else if (nverts == 4) {
    rz::compute_quad_corner_volumes_exact_subpolygon(
        r[0], z[0], r[1], z[1], r[2], z[2], r[3], z[3], corner_volume);
  } else {
    PentagonPoint points[5];
    for (int k = 0; k < 5; ++k) {
      points[k] = {r[k], z[k]};
    }
    pentagon_corner_rz_volumes(points, corner_volume);
    for (int k = 0; k < 5; ++k) {
      corner_volume[k] = std::fabs(corner_volume[k]);
    }
  }
  for (int k = 0; k < nverts; ++k) {
    double sub_r[4] = {};
    double sub_z[4] = {};
    csr_corner_distribution_subpolygon(
        r, z, nverts, k, sub_r, sub_z);
    rz::rz_polygon_area_centroid_exact(
        sub_r, sub_z, 4, corner_centroid_r + k, corner_centroid_z + k);
  }
}

void csr_aw_barlow_corner_area_partition(
    const double* const r,
    const double* const z,
    const int nverts,
    double* const corner_area) {
  TENRYU_ASSERT(nverts >= 3 &&
                    nverts <= mesh::kMeshTopoCellStorageSlotsMaxGeneral,
                "AW Barlow corner-area partition requires a supported "
                "general-N cell");
  const double cell_area =
      0.5 * std::abs(rz::rz_polygon_area2_exact(r, z, nverts));
  TENRYU_ASSERT(cell_area > 0.0 && std::isfinite(cell_area),
                "AW planar cell area must be positive and finite");

  double centroid_r = 0.0;
  double centroid_z = 0.0;
  for (int k = 0; k < nverts; ++k) {
    centroid_r += r[k];
    centroid_z += z[k];
  }
  const double inv_nverts = 1.0 / static_cast<double>(nverts);
  centroid_r *= inv_nverts;
  centroid_z *= inv_nverts;

  double edge_triangle_area[
      mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  double edge_area_sum = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int next = (k + 1) % nverts;
    const double triangle_r[3] = {r[k], r[next], centroid_r};
    const double triangle_z[3] = {z[k], z[next], centroid_z};
    edge_triangle_area[k] =
        0.5 * std::abs(
                  rz::rz_polygon_area2_exact(triangle_r, triangle_z, 3));
    edge_area_sum += edge_triangle_area[k];
  }
  TENRYU_ASSERT(
      std::abs(edge_area_sum - cell_area) <= 1.0e-12 * cell_area,
      "AW Barlow edge-triangle areas must sum to the cell area");

  const double shared_area = edge_area_sum * inv_nverts / 3.0;
  for (int k = 0; k < nverts; ++k) {
    const int prev = (k + nverts - 1) % nverts;
    corner_area[k] =
        (edge_triangle_area[prev] + edge_triangle_area[k]) / 3.0 +
        shared_area;
  }
}

struct RemapDispatchAuditRuntime {
  bool initialized = false;
  bool flushed = false;
  int rank = 0;
  int n_cells = 0;
  const parallel::Reduction* reduction = nullptr;
  core::DeviceArray<int> counters{"remap_dispatch_audit:counters"};
  core::DeviceArray<std::uint8_t> feather_mask{
      "remap_dispatch_audit:feather_mask"};
};

struct GclAuditTransactionSummary {
  int step = -1;
  double max_abs_R = 0.0;
  double max_rel_R = 0.0;
  double sum_R = 0.0;
  double sum_rel = 0.0;
  int n_cells_active = 0;
};

struct GclAuditRuntime {
  bool initialized = false;
  bool flushed = false;
  bool manifest_emitted = false;
  bool has_transaction = false;
  int rank = 0;
  int transaction_count = 0;
  const parallel::Reduction* reduction = nullptr;
  GclAuditTransactionSummary worst;
};

RemapDispatchAuditRuntime& remap_dispatch_audit_runtime() {
  static RemapDispatchAuditRuntime runtime;
  return runtime;
}

GclAuditRuntime& gcl_audit_runtime() {
  static GclAuditRuntime runtime;
  return runtime;
}

bool gcl_audit_env_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_GCL_AUDIT");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  return enabled;
}

const char* remap_dispatch_audit_counter_name(const int index) {
  static constexpr std::array<const char*, kRemapDispatchAuditCounterCount>
      names = {"exact_swept_moment",
               "legacy_swept_volume",
               "first_order_donor_fallback",
               "limiter_activation",
               "momentum_packet_fallback",
               "boundary_one_sided",
               "swept_centroid_average_fallback",
               "ms2_degenerate_gradient_fallback",
               "csr_gradient_zero_fallback",
               "reconstruction_nonfinite_fallback",
               "momentum_expanded_failure_fallback",
               "momentum_invalid_input_fallback",
               "projection_gradient_condition_fallback"};
  return names[static_cast<std::size_t>(index)];
}

}  // namespace

bool remap_dispatch_audit_env_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_REMAP_DISPATCH_AUDIT");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  return enabled;
}

void remap_dispatch_audit_run_start(
    const core::State& state,
    const parallel::Reduction* reduction,
    const int rank) {
  if (!remap_dispatch_audit_env_enabled()) {
    return;
  }
  RemapDispatchAuditRuntime& runtime = remap_dispatch_audit_runtime();
  runtime.initialized = true;
  runtime.flushed = false;
  runtime.rank = rank;
  runtime.reduction = reduction;
  runtime.n_cells = state.mesh.topo.n_cells;
  runtime.counters.reset(
      static_cast<std::size_t>(2 * kRemapDispatchAuditCounterCount));
  CUDA_CHECK(cudaMemset(runtime.counters.data(),
                        0,
                        runtime.counters.size() * sizeof(int)));

  runtime.feather_mask.reset(static_cast<std::size_t>(runtime.n_cells));
  std::vector<std::uint8_t> feather_mask(
      static_cast<std::size_t>(runtime.n_cells), 0U);
  if (state.mesh.dim == 2 && runtime.n_cells > 0) {
    TENRYU_ASSERT(state.x_r_initial.size() == state.x_r.size() &&
                      state.x_z_initial.size() == state.x_z.size(),
                  "remap dispatch audit requires initial node coordinates");
    std::vector<double> initial_r;
    std::vector<double> initial_z;
    state.x_r_initial.copy_to_host(initial_r);
    state.x_z_initial.copy_to_host(initial_z);
    const auto& cell_nverts = state.mesh.cell_nverts;
    const auto& mb = state.mesh.topo.multiblock;
    for (int c = 0; c < runtime.n_cells; ++c) {
      double center_r = 0.0;
      double center_z = 0.0;
      const int nverts =
          mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
      if (mb.has_value()) {
        const int off = mb->cell_node_csr_offsets[static_cast<std::size_t>(c)];
        for (int k = 0; k < nverts; ++k) {
          const int node = mb->cell_node_csr_indices[
              static_cast<std::size_t>(off + k)];
          center_r += initial_r[static_cast<std::size_t>(node)];
          center_z += initial_z[static_cast<std::size_t>(node)];
        }
      } else {
        TENRYU_ASSERT(nverts == 4,
                      "structured remap dispatch audit cells must be quads");
        const int nz = state.mesh.topo.nz;
        const int i = c / nz;
        const int j = c - i * nz;
        const int node[4] = {i * (nz + 1) + j,
                             (i + 1) * (nz + 1) + j,
                             (i + 1) * (nz + 1) + j + 1,
                             i * (nz + 1) + j + 1};
        for (int k = 0; k < nverts; ++k) {
          center_r += initial_r[static_cast<std::size_t>(node[k])];
          center_z += initial_z[static_cast<std::size_t>(node[k])];
        }
      }
      center_r /= static_cast<double>(nverts);
      center_z /= static_cast<double>(nverts);
      const double radius = std::hypot(center_r, center_z);
      feather_mask[static_cast<std::size_t>(c)] =
          (radius >= kRemapDispatchAuditFeatherRadiusMin &&
           radius <= kRemapDispatchAuditFeatherRadiusMax)
              ? 1U
              : 0U;
    }
  }
  runtime.feather_mask.copy_from_host(feather_mask);
}

RemapDispatchAuditDeviceView remap_dispatch_audit_device_view() {
  RemapDispatchAuditRuntime& runtime = remap_dispatch_audit_runtime();
  if (!remap_dispatch_audit_env_enabled() || !runtime.initialized ||
      runtime.flushed) {
    return {};
  }
  return {runtime.counters.data(),
          runtime.feather_mask.data(),
          runtime.n_cells};
}

void remap_dispatch_audit_flush() {
  if (!remap_dispatch_audit_env_enabled()) {
    return;
  }
  RemapDispatchAuditRuntime& runtime = remap_dispatch_audit_runtime();
  if (!runtime.initialized || runtime.flushed) {
    return;
  }
  runtime.flushed = true;
  std::vector<int> local_counts;
  runtime.counters.copy_to_host(local_counts);
  std::vector<double> counts(local_counts.size(), 0.0);
  for (std::size_t i = 0; i < local_counts.size(); ++i) {
    counts[i] = static_cast<double>(local_counts[i]);
  }
  if (runtime.reduction != nullptr) {
    runtime.reduction->allreduce_sum(counts.data(),
                                     static_cast<int>(counts.size()));
  }
  if (runtime.rank != 0) {
    return;
  }
  std::ostringstream oss;
  oss << "[remap-audit]";
  for (int i = 0; i < kRemapDispatchAuditCounterCount; ++i) {
    oss << " " << remap_dispatch_audit_counter_name(i) << "="
        << static_cast<long long>(counts[static_cast<std::size_t>(i)]);
  }
  // Centroid-outside is only an expanded-stencil reason classification;
  // there is no distinct centroid-outside-to-donor-fallback branch.
  oss << " donor_centroid_outside_fallback=not_distinct";
  for (int i = 0; i < kRemapDispatchAuditCounterCount; ++i) {
    oss << " feather_" << remap_dispatch_audit_counter_name(i) << "="
        << static_cast<long long>(counts[static_cast<std::size_t>(
               kRemapDispatchAuditCounterCount + i)]);
  }
  oss << " feather_donor_centroid_outside_fallback=not_distinct";
  core::log_info(oss.str());
}

void gcl_audit_run_start(
    const parallel::Reduction* const reduction,
    const int rank) {
  if (!gcl_audit_env_enabled()) {
    return;
  }
  GclAuditRuntime& runtime = gcl_audit_runtime();
  runtime = {};
  runtime.initialized = true;
  runtime.rank = rank;
  runtime.reduction = reduction;
}

void gcl_audit_flush() {
  if (!gcl_audit_env_enabled()) {
    return;
  }
  GclAuditRuntime& runtime = gcl_audit_runtime();
  if (!runtime.initialized || runtime.flushed) {
    return;
  }
  runtime.flushed = true;
  if (runtime.rank != 0) {
    return;
  }
  const GclAuditTransactionSummary& worst = runtime.worst;
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(17)
      << "[gcl-audit] FINAL worst_step=" << worst.step
      << " max_abs_R=" << worst.max_abs_R
      << " max_rel_R=" << worst.max_rel_R
      << " sum_R=" << worst.sum_R
      << " sum_rel=" << worst.sum_rel
      << " n_cells_active=" << worst.n_cells_active
      << " n_transactions=" << runtime.transaction_count;
  core::log_info(oss.str());
}

__host__ __device__ inline void csr_compute_pentagon_qk_corner_masses(
    const double m_cell,
    const double* const r,
    const double* const z,
    double* const m_corner) {
  PentagonPoint x[5];
  for (int k = 0; k < 5; ++k) {
    x[k] = {r[k], z[k]};
  }
  double v_sub[5] = {};
  pentagon_corner_rz_volumes(x, v_sub);
  double v_sum = 0.0;
  for (int k = 0; k < 5; ++k) {
    const bool valid = v_sub[k] > 0.0 &&
                       pentagon_geometry_detail::finite_double(v_sub[k]);
#if defined(__CUDA_ARCH__)
    if (!valid) {
      printf("CSR remap pentagon Q_k corner volume invalid\n");
      __trap();
    }
#else
    TENRYU_ASSERT(
        valid,
        "CSR remap pentagon Q_k corner volume must be positive and finite");
#endif
    v_sum += v_sub[k];
  }
  const bool valid_sum =
      v_sum > 0.0 && pentagon_geometry_detail::finite_double(v_sum);
#if defined(__CUDA_ARCH__)
  if (!valid_sum) {
    printf("CSR remap pentagon Q_k corner volume sum invalid\n");
    __trap();
  }
#else
  TENRYU_ASSERT(
      valid_sum,
      "CSR remap pentagon Q_k corner volume sum must be positive and finite");
#endif
  // Pentagon corner masses follow the Q_k subzonal convention (NUMERICS §3.2.4).
  for (int k = 0; k < 5; ++k) {
    m_corner[k] = m_cell * (v_sub[k] / v_sum);
  }
  m_corner[2] =
      m_cell - ((m_corner[0] + m_corner[1]) + (m_corner[3] + m_corner[4]));
}

__global__ void compute_cell_velocity_from_nodes_kernel(
    double* __restrict__ v_r_cell,
    double* __restrict__ v_z_cell,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const std::uint8_t* __restrict__ cell_nverts,
    int nr,
    int nz);

__global__ void project_cell_velocity_to_nodes_kernel(
    double* __restrict__ v_r_node,
    double* __restrict__ v_z_node,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const std::uint8_t* __restrict__ node_flags,
    int nr,
    int nz,
    int r_outer_bc_mode,
    int z_bottom_bc_mode,
    int z_top_bc_mode);

__device__ inline double rz_van_leer_slope_pair(const double dl,
                                                const double dr) {
  if (!isfinite(dl) || !isfinite(dr) || dl * dr <= 0.0) {
    return 0.0;
  }
  const double denom = dl + dr;
  if (fabs(denom) <= 1.0e-300) {
    return 0.0;
  }
  return 2.0 * dl * dr / denom;
}

__device__ inline double rz_cell_center_coord(const double* __restrict__ x,
                                              const int i,
                                              const int j,
                                              const int nz,
                                              const std::uint8_t* __restrict__ cell_nverts) {
  const int n00 = detail::node_index(i, j, nz);
  const int n10 = detail::node_index(i + 1, j, nz);
  const int n11 = detail::node_index(i + 1, j + 1, nz);
  const int n01 = detail::node_index(i, j + 1, nz);
  const int c = detail::cell_index(i, j, nz);
  if (cell_nverts != nullptr && cell_nverts[c] == 3U) {
    return (x[n00] + x[n10] + x[n11]) / 3.0;
  }
  return 0.25 * (x[n00] + x[n10] + x[n11] + x[n01]);
}

__device__ inline double rz_axis_limited_slope(const double qm,
                                               const double q,
                                               const double qp,
                                               const double xm,
                                               const double x,
                                               const double xp,
                                               const bool has_m,
                                               const bool has_p) {
  if (has_m && has_p) {
    const double dl = (q - qm) / fmax(fabs(x - xm), 1.0e-300);
    const double dr = (qp - q) / fmax(fabs(xp - x), 1.0e-300);
    return rz_van_leer_slope_pair(dl, dr);
  }
  if (has_m) {
    const double slope = (q - qm) / fmax(fabs(x - xm), 1.0e-300);
    return isfinite(slope) ? slope : 0.0;
  }
  if (has_p) {
    const double slope = (qp - q) / fmax(fabs(xp - x), 1.0e-300);
    return isfinite(slope) ? slope : 0.0;
  }
  return 0.0;
}

__device__ double rz_limited_slope_r(const int i,
                                     const int j,
                                     const int nr,
                                     const int nz,
                                     const double* __restrict__ Q,
                                     const double* __restrict__ x_r,
                                     const std::uint8_t* __restrict__ cell_nverts) {
  const int c = detail::cell_index(i, j, nz);
  const double q = Q[c];
  const bool has_m = i > 0;
  const bool has_p = i + 1 < nr;
  const double qm = has_m ? Q[detail::cell_index(i - 1, j, nz)] : q;
  const double qp = has_p ? Q[detail::cell_index(i + 1, j, nz)] : q;
  const double x = rz_cell_center_coord(x_r, i, j, nz, cell_nverts);
  const double xm =
      has_m ? rz_cell_center_coord(x_r, i - 1, j, nz, cell_nverts) : x;
  const double xp =
      has_p ? rz_cell_center_coord(x_r, i + 1, j, nz, cell_nverts) : x;
  return rz_axis_limited_slope(qm, q, qp, xm, x, xp, has_m, has_p);
}

__device__ double rz_limited_slope_z(const int i,
                                     const int j,
                                     const int nr,
                                     const int nz,
                                     const double* __restrict__ Q,
                                     const double* __restrict__ x_z,
                                     const std::uint8_t* __restrict__ cell_nverts) {
  const int c = detail::cell_index(i, j, nz);
  const double q = Q[c];
  const bool has_m = j > 0;
  const bool has_p = j + 1 < nz;
  const double qm = has_m ? Q[detail::cell_index(i, j - 1, nz)] : q;
  const double qp = has_p ? Q[detail::cell_index(i, j + 1, nz)] : q;
  const double x = rz_cell_center_coord(x_z, i, j, nz, cell_nverts);
  const double xm =
      has_m ? rz_cell_center_coord(x_z, i, j - 1, nz, cell_nverts) : x;
  const double xp =
      has_p ? rz_cell_center_coord(x_z, i, j + 1, nz, cell_nverts) : x;
  return rz_axis_limited_slope(qm, q, qp, xm, x, xp, has_m, has_p);
}

__device__ double rz_barth_jespersen_limiter(const double slope,
                                             const double Q_K,
                                             const double Q_min,
                                             const double Q_max,
                                             const double dx) {
  const double dq = slope * dx;
  if (!isfinite(dq) || fabs(dq) <= 1.0e-300) {
    return 1.0;
  }
  if (dq > 0.0) {
    return fmin(1.0, fmax(0.0, (Q_max - Q_K) / dq));
  }
  return fmin(1.0, fmax(0.0, (Q_min - Q_K) / dq));
}

__device__ inline void rz_neighbor_bounds(const int i,
                                          const int j,
                                          const int nr,
                                          const int nz,
                                          const double* __restrict__ Q,
                                          double& q_min,
                                          double& q_max) {
  const int c = detail::cell_index(i, j, nz);
  q_min = Q[c];
  q_max = Q[c];
  if (i > 0) {
    const double q = Q[detail::cell_index(i - 1, j, nz)];
    q_min = fmin(q_min, q);
    q_max = fmax(q_max, q);
  }
  if (i + 1 < nr) {
    const double q = Q[detail::cell_index(i + 1, j, nz)];
    q_min = fmin(q_min, q);
    q_max = fmax(q_max, q);
  }
  if (j > 0) {
    const double q = Q[detail::cell_index(i, j - 1, nz)];
    q_min = fmin(q_min, q);
    q_max = fmax(q_max, q);
  }
  if (j + 1 < nz) {
    const double q = Q[detail::cell_index(i, j + 1, nz)];
    q_min = fmin(q_min, q);
    q_max = fmax(q_max, q);
  }
}

__device__ inline double rz_reconstructed_value_at(const int donor,
                                                   const double* __restrict__ Q,
                                                   const double* __restrict__ x_r_lag,
                                                   const double* __restrict__ x_z_lag,
                                                   const double r_face,
                                                   const double z_face,
	                                                const int nr,
                                                const int nz,
                                                const double Q_floor,
                                                const std::uint8_t* __restrict__ cell_nverts) {
  const int i = donor / nz;
  const int j = donor - i * nz;
  const double q = Q[donor];
  const double r_c = detail::cell_center_r(x_r_lag, i, j, nz, cell_nverts);
  const double z_c = detail::cell_center_z(x_z_lag, i, j, nz, cell_nverts);
  const double slope_r =
      rz_limited_slope_r(i, j, nr, nz, Q, x_r_lag, cell_nverts);
  const double slope_z =
      rz_limited_slope_z(i, j, nr, nz, Q, x_z_lag, cell_nverts);
  double q_min = q;
  double q_max = q;
  rz_neighbor_bounds(i, j, nr, nz, Q, q_min, q_max);
  const double dq = slope_r * (r_face - r_c) + slope_z * (z_face - z_c);
  const double psi = rz_barth_jespersen_limiter(1.0, q, q_min, q_max, dq);
  double q_face = q + psi * dq;
  q_face = fmin(q_max, fmax(q_min, q_face));
  if (Q_floor >= 0.0) {
    q_face = fmax(q_face, Q_floor);
  }
  return isfinite(q_face) ? q_face : fmax(q, Q_floor);
}

__device__ inline double rz_reconstruction_delta_at(
    const int donor,
    const double* __restrict__ Q,
    const double* __restrict__ x_r_lag,
    const double* __restrict__ x_z_lag,
    const double r_face,
    const double z_face,
    const int nr,
    const int nz,
    double& q,
    double& q_min,
    double& q_max,
    const std::uint8_t* __restrict__ cell_nverts) {
  const int i = donor / nz;
  const int j = donor - i * nz;
  q = Q[donor];
  const double r_c = detail::cell_center_r(x_r_lag, i, j, nz, cell_nverts);
  const double z_c = detail::cell_center_z(x_z_lag, i, j, nz, cell_nverts);
  const double slope_r =
      rz_limited_slope_r(i, j, nr, nz, Q, x_r_lag, cell_nverts);
  const double slope_z =
      rz_limited_slope_z(i, j, nr, nz, Q, x_z_lag, cell_nverts);
  q_min = q;
  q_max = q;
  rz_neighbor_bounds(i, j, nr, nz, Q, q_min, q_max);
  return slope_r * (r_face - r_c) + slope_z * (z_face - z_c);
}

__device__ inline double rz_limiter_at_face(const int donor,
                                            const double* __restrict__ Q,
                                            const double* __restrict__ x_r_lag,
                                            const double* __restrict__ x_z_lag,
                                            const double r_face,
                                            const double z_face,
                                            const int nr,
                                            const int nz,
                                            const std::uint8_t* __restrict__ cell_nverts) {
  double q = 0.0;
  double q_min = 0.0;
  double q_max = 0.0;
  const double dq = rz_reconstruction_delta_at(
      donor, Q, x_r_lag, x_z_lag, r_face, z_face, nr, nz, q, q_min, q_max,
      cell_nverts);
  return rz_barth_jespersen_limiter(1.0, q, q_min, q_max, dq);
}

__device__ inline double rz_reconstructed_value_at_with_psi(
    const int donor,
    const double* __restrict__ Q,
    const double* __restrict__ x_r_lag,
    const double* __restrict__ x_z_lag,
    const double r_face,
    const double z_face,
    const int nr,
    const int nz,
    const double Q_floor,
    const double psi_common,
    const std::uint8_t* __restrict__ cell_nverts) {
  double q = 0.0;
  double q_min = 0.0;
  double q_max = 0.0;
  const double dq = rz_reconstruction_delta_at(
      donor, Q, x_r_lag, x_z_lag, r_face, z_face, nr, nz, q, q_min, q_max,
      cell_nverts);
  double q_face = q + fmin(1.0, fmax(0.0, psi_common)) * dq;
  q_face = fmin(q_max, fmax(q_min, q_face));
  if (Q_floor >= 0.0) {
    q_face = fmax(q_face, Q_floor);
  }
  return isfinite(q_face) ? q_face : fmax(q, Q_floor);
}

__device__ double rz_reconstructed_face_value_r(const int K,
                                                const int Kp,
                                                const double swept_volume,
                                                const double* __restrict__ Q,
                                                const double* __restrict__ x_r_lag,
                                                const double* __restrict__ x_z_lag,
                                                const double* __restrict__ x_r_ref,
                                                const double* __restrict__ x_z_ref,
                                                const int i_face,
                                                const int j,
                                                const int nr,
                                                const int nz,
                                                const double Q_floor,
                                                const std::uint8_t* __restrict__ cell_nverts,
                                                const int button_outer_node_ring = 0) {
  const int donor = (swept_volume >= 0.0) ? K : Kp;
  const double inv_dV = 1.0 / swept_volume;
  double moment_r = detail::ms2_swept_moment_r_face_t<true>(
      x_r_lag, x_z_lag, x_r_ref, x_z_ref, i_face, j, nz);
  double moment_z = detail::ms2_swept_moment_z_r_face_t<true>(
      x_r_lag, x_z_lag, x_r_ref, x_z_ref, i_face, j, nz);
	  if (cell_nverts != nullptr || button_outer_node_ring > 0) {
	    moment_r = -moment_r;
	    moment_z = -moment_z;
	  }
  const double r_face = moment_r * inv_dV;
  const double z_face = moment_z * inv_dV;
  return rz_reconstructed_value_at(
      donor, Q, x_r_lag, x_z_lag, r_face, z_face, nr, nz, Q_floor,
      cell_nverts);
}

__device__ double rz_reconstructed_face_value_z(const int K,
                                                const int Kp,
                                                const double swept_volume,
                                                const double* __restrict__ Q,
                                                const double* __restrict__ x_r_lag,
                                                const double* __restrict__ x_z_lag,
                                                const double* __restrict__ x_r_ref,
                                                const double* __restrict__ x_z_ref,
                                                const int i,
                                                const int j_face,
	                                                const int nr,
	                                                const int nz,
	                                                const double Q_floor,
	                                                const std::uint8_t* __restrict__ cell_nverts,
	                                                const int button_outer_node_ring = 0) {
  const int donor = (swept_volume >= 0.0) ? K : Kp;
  const double inv_dV = 1.0 / swept_volume;
  double moment_r = detail::ms2_swept_moment_r_z_face_t<true>(
      x_r_lag, x_z_lag, x_r_ref, x_z_ref, i, j_face, nz);
  double moment_z = detail::ms2_swept_moment_z_face_t<true>(
      x_r_lag, x_z_lag, x_r_ref, x_z_ref, i, j_face, nz);
	  if (cell_nverts != nullptr || button_outer_node_ring > 0) {
	    moment_r = -moment_r;
	    moment_z = -moment_z;
	  }
  const double r_face = moment_r * inv_dV;
  const double z_face = moment_z * inv_dV;
  return rz_reconstructed_value_at(
      donor, Q, x_r_lag, x_z_lag, r_face, z_face, nr, nz, Q_floor,
      cell_nverts);
}

namespace {

CsrConsAuditContext g_csr_cons_audit_context{};

constexpr double kTinyVolume = 1.0e-300;
// Keep aligned with the standard-step donor-mass oracle in ale_driver.cu.
constexpr double kRemapMassClosureRejectFraction = 0.1;
constexpr int kCentralMacroRemapAuditMass = 0;
constexpr int kCentralMacroRemapAuditEnergy = 1;
constexpr int kCentralMacroRemapAuditTracer = 2;
constexpr int kCentralMacroRemapAuditSkippedFaces = 3;
constexpr int kCentralMacroRemapAuditCount = 4;

__global__ void burn_spectrum_scale_rows_kernel(double* dst,
                                                const double* src,
                                                const double* rho,
                                                int G,
                                                int J,
                                                bool divide) {
  const std::size_t total =
      static_cast<std::size_t>(G) * static_cast<std::size_t>(J);
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int j = static_cast<int>(idx % static_cast<std::size_t>(J));
  dst[idx] = divide ? (rho[j] > 0.0 ? src[idx] / rho[j] : 0.0)
                    : src[idx] * rho[j];
}

void burn_spectrum_scale_rows_by_cell(double* dst,
                                      const double* src,
                                      const double* rho,
                                      int G,
                                      int J) {
  const std::size_t total =
      static_cast<std::size_t>(G) * static_cast<std::size_t>(J);
  const int grid = static_cast<int>((total + 255U) / 256U);
  burn_spectrum_scale_rows_kernel<<<grid, 256>>>(
      dst, src, rho, G, J, false);
  CUDA_CHECK(cudaGetLastError());
}

void burn_spectrum_divide_rows_by_cell(double* dst,
                                       const double* src,
                                       const double* rho,
                                       int G,
                                       int J) {
  const std::size_t total =
      static_cast<std::size_t>(G) * static_cast<std::size_t>(J);
  const int grid = static_cast<int>((total + 255U) / 256U);
  burn_spectrum_scale_rows_kernel<<<grid, 256>>>(
      dst, src, rho, G, J, true);
  CUDA_CHECK(cudaGetLastError());
}

inline int velocity_bc_mode_local(const Boundary2DType bc) {
  if (bc == Boundary2DType::FIXED) {
    return 2;
  }
  if (bc == Boundary2DType::REFLECT) {
    return 1;
  }
  if (bc == Boundary2DType::STATE_SUPPLY) {
    return 3;
  }
  return 0;
}

__device__ inline bool finite_nonzero(const double x) {
  return isfinite(x) && x != 0.0;
}

__device__ inline bool csr_inactive_cell(
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int cell) {
  return inactive_cell_mask != nullptr && cell >= 0 &&
         inactive_cell_mask[cell] != 0U;
}

__device__ inline bool csr_face_touches_inactive(
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int cell_a,
    const int cell_b) {
  return csr_inactive_cell(inactive_cell_mask, cell_a) ||
         csr_inactive_cell(inactive_cell_mask, cell_b);
}

__device__ inline void csr_note_inactive_face_skip(
    double* __restrict__ macro_flux_audit) {
  if (macro_flux_audit != nullptr) {
    atomicAdd(macro_flux_audit + kCentralMacroRemapAuditSkippedFaces, 1.0);
  }
}

__device__ double rz_swept_volume_face(
    const double2 lag_p0, const double2 lag_p1,
    const double2 ref_p0, const double2 ref_p1) {
  return -detail::rz_signed_quad_volume(lag_p0.x,
                                        lag_p0.y,
                                        ref_p0.x,
                                        ref_p0.y,
                                        ref_p1.x,
                                        ref_p1.y,
                                        lag_p1.x,
                                        lag_p1.y);
}

__device__ int rz_donor_cell(const int K, const int Kp, const double swept_volume) {
  return (swept_volume >= 0.0) ? K : Kp;
}

__device__ inline bool rz_button_enabled(const int button_outer_node_ring) {
  return button_outer_node_ring > 0;
}

__device__ inline bool rz_is_button_cell(const int c,
                                         const int button_outer_node_ring) {
  return rz_button_enabled(button_outer_node_ring) && c == 0;
}

__device__ inline bool rz_is_dormant_button_cell(
    const int c,
    const int i,
    const int button_outer_node_ring) {
  return rz_button_enabled(button_outer_node_ring) && c != 0 &&
         i >= 0 && i < button_outer_node_ring;
}

__device__ inline bool rz_is_dormant_button_cell_index(
    const int c,
    const int nz,
    const int button_outer_node_ring) {
  if (c < 0 || nz <= 0) {
    return false;
  }
  return rz_is_dormant_button_cell(c, c / nz, button_outer_node_ring);
}

__device__ inline bool rz_button_face_touches_dormant(
    const int K,
    const int Kp,
    const int nz,
    const int button_outer_node_ring) {
  return rz_is_dormant_button_cell_index(K, nz, button_outer_node_ring) ||
         rz_is_dormant_button_cell_index(Kp, nz, button_outer_node_ring);
}

__device__ inline double rz_frozen_cell_mass(
    const double* __restrict__ mass_lag,
    const double* __restrict__ rho_lag,
    const double* __restrict__ vol_lag,
    const int c) {
  if (mass_lag != nullptr && isfinite(mass_lag[c])) {
    return mass_lag[c];
  }
  return rho_lag[c] * vol_lag[c];
}

__device__ inline bool rz_uses_polar_orientation(
    const std::uint8_t* __restrict__ cell_nverts,
    const int button_outer_node_ring) {
  return cell_nverts != nullptr || rz_button_enabled(button_outer_node_ring);
}

__device__ inline double2 node_point(const double* __restrict__ x_r,
                                     const double* __restrict__ x_z,
                                     const int n) {
  return make_double2(x_r[n], x_z[n]);
}

__device__ inline double rz_swept_volume_r_face_ref(
    const double* __restrict__ x_r_lag,
    const double* __restrict__ x_z_lag,
    const double* __restrict__ x_r_ref,
    const double* __restrict__ x_z_ref,
    const int i_face,
    const int j,
    const int nz) {
  const int n0 = detail::node_index(i_face, j, nz);
  const int n1 = detail::node_index(i_face, j + 1, nz);
  return rz_swept_volume_face(node_point(x_r_lag, x_z_lag, n0),
                              node_point(x_r_lag, x_z_lag, n1),
                              node_point(x_r_ref, x_z_ref, n0),
                              node_point(x_r_ref, x_z_ref, n1));
}

__device__ inline double rz_swept_volume_z_face_ref(
    const double* __restrict__ x_r_lag,
    const double* __restrict__ x_z_lag,
    const double* __restrict__ x_r_ref,
    const double* __restrict__ x_z_ref,
    const int i,
    const int j_face,
    const int nz) {
  const int n0 = detail::node_index(i, j_face, nz);
  const int n1 = detail::node_index(i + 1, j_face, nz);
  return -rz_swept_volume_face(node_point(x_r_lag, x_z_lag, n0),
                               node_point(x_r_lag, x_z_lag, n1),
                               node_point(x_r_ref, x_z_ref, n0),
                               node_point(x_r_ref, x_z_ref, n1));
}

__device__ inline bool rz_is_tri_cell(const std::uint8_t* __restrict__ cell_nverts,
                                      const int c) {
  return cell_nverts != nullptr && cell_nverts[c] == 3U;
}

__device__ inline bool rz_r_face_has_tri(const std::uint8_t* __restrict__ cell_nverts,
                                         const int i_face,
                                         const int j,
                                         const int nr,
                                         const int nz) {
  if (cell_nverts == nullptr || i_face <= 0 || i_face >= nr) {
    return false;
  }
  return rz_is_tri_cell(cell_nverts, detail::cell_index(i_face - 1, j, nz)) ||
         rz_is_tri_cell(cell_nverts, detail::cell_index(i_face, j, nz));
}

__device__ inline bool rz_z_face_has_tri(const std::uint8_t* __restrict__ cell_nverts,
                                         const int i,
                                         const int j_face,
                                         const int nr,
                                         const int nz) {
  if (cell_nverts == nullptr || j_face <= 0 || j_face >= nz || i < 0 || i >= nr) {
    return false;
  }
  return rz_is_tri_cell(cell_nverts, detail::cell_index(i, j_face - 1, nz)) ||
         rz_is_tri_cell(cell_nverts, detail::cell_index(i, j_face, nz));
}

__device__ inline double rz_swept_volume_r_face_ref_topology(
    const double* __restrict__ x_r_lag,
    const double* __restrict__ x_z_lag,
    const double* __restrict__ x_r_ref,
    const double* __restrict__ x_z_ref,
    const std::uint8_t* __restrict__ cell_nverts,
    const int i_face,
    const int j,
    const int nr,
    const int nz,
    const int button_outer_node_ring) {
  const double dV =
      rz_swept_volume_r_face_ref(x_r_lag, x_z_lag, x_r_ref, x_z_ref, i_face, j, nz);
  (void)nr;
  return rz_uses_polar_orientation(cell_nverts, button_outer_node_ring) ? -dV
                                                                        : dV;
}

__device__ inline double rz_swept_volume_z_face_ref_topology(
    const double* __restrict__ x_r_lag,
    const double* __restrict__ x_z_lag,
    const double* __restrict__ x_r_ref,
    const double* __restrict__ x_z_ref,
    const std::uint8_t* __restrict__ cell_nverts,
    const int i,
    const int j_face,
    const int nr,
    const int nz,
    const int button_outer_node_ring) {
  const double dV =
      rz_swept_volume_z_face_ref(x_r_lag, x_z_lag, x_r_ref, x_z_ref, i, j_face, nz);
  (void)nr;
  return rz_uses_polar_orientation(cell_nverts, button_outer_node_ring) ? -dV
                                                                        : dV;
}

__device__ inline double rz_button_seam_swept_volume_ref(
    const double* __restrict__ x_r_lag,
    const double* __restrict__ x_z_lag,
    const double* __restrict__ x_r_ref,
    const double* __restrict__ x_z_ref,
    const int button_outer_node_ring,
    const int nz,
    const int j) {
  return rz::button_seam_swept_volume_from_nodes(
      x_r_lag, x_z_lag, x_r_ref, x_z_ref, button_outer_node_ring, nz, j);
}

__device__ inline double z_face_area_rz(const double* __restrict__ x_r,
                                        const int i,
                                        const int j_face,
                                        const int nz) {
  const int n0 = detail::node_index(i, j_face, nz);
  const int n1 = detail::node_index(i + 1, j_face, nz);
  const double r0 = fmax(x_r[n0], 0.0);
  const double r1 = fmax(x_r[n1], 0.0);
  return detail::kPi * fabs(r1 * r1 - r0 * r0);
}

__global__ void project_cell_velocity_to_nodes_button_kernel(
    double* __restrict__ v_r_node,
    double* __restrict__ v_z_node,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const std::uint8_t* __restrict__ node_flags,
    const int nr,
    const int nz,
    const int button_outer_node_ring,
    const int r_outer_bc_mode,
    const int z_bottom_bc_mode,
    const int z_top_bc_mode) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_nodes = (nr + 1) * (nz + 1);
  if (n >= n_nodes) {
    return;
  }

  const int stride = nz + 1;
  const int i = n / stride;
  const int j = n - i * stride;

  double w_sum = 0.0;
  double vr_sum = 0.0;
  double vz_sum = 0.0;
  for (int di = -1; di <= 0; ++di) {
    for (int dj = -1; dj <= 0; ++dj) {
      const int ic = i + di;
      const int jc = j + dj;
      if (ic < 0 || ic >= nr || jc < 0 || jc >= nz) {
        continue;
      }
      const int c = ic * nz + jc;
      if (rz_is_dormant_button_cell(c, ic, button_outer_node_ring)) {
        continue;
      }
      const double w = fmax(rho[c] * vol[c], 0.0);
      w_sum += w;
      vr_sum += w * v_r_cell[c];
      vz_sum += w * v_z_cell[c];
    }
  }

  if (w_sum > 0.0) {
    v_r_node[n] = vr_sum / w_sum;
    v_z_node[n] = vz_sum / w_sum;
  } else {
    v_r_node[n] = 0.0;
    v_z_node[n] = 0.0;
  }

  pole_axis::apply_2d_boundary_vector_constraints(
      v_r_node[n],
      v_z_node[n],
      node_flags,
      n,
      i,
      j,
      nr,
      nz,
      r_outer_bc_mode,
      z_bottom_bc_mode,
      z_top_bc_mode,
      false);
}

// The remap never moves the pinned center point, so the degenerate center
// column's material velocity is remap-invariant: snapshot it before the
// cell-to-node projection and restore it afterwards. Deriving it from the
// neighborhood projection instead would overwrite the column with a
// legitimately nonzero field mean that is NOT the center's material
// velocity. Single-thread; both kernels walk the flags in the same
// ascending order, so slot k pairs with the same node in save and restore.
__global__ void save_center_column_velocity_kernel(
    double* __restrict__ saved_vr,
    double* __restrict__ saved_vz,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const std::uint8_t* __restrict__ node_flags,
    const int n_nodes) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  constexpr int kMaxCenterColumn = 1024;
  int count = 0;
  for (int n = 0; n < n_nodes && count < kMaxCenterColumn; ++n) {
    if ((node_flags[n] & pole_axis::kNodeCenterFlag) != 0U) {
      saved_vr[count] = v_r_node[n];
      saved_vz[count] = v_z_node[n];
      ++count;
    }
  }
}

__global__ void restore_center_column_velocity_kernel(
    double* __restrict__ v_r_node,
    double* __restrict__ v_z_node,
    const double* __restrict__ saved_vr,
    const double* __restrict__ saved_vz,
    const std::uint8_t* __restrict__ node_flags,
    const int n_nodes) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  constexpr int kMaxCenterColumn = 1024;
  int count = 0;
  for (int n = 0; n < n_nodes && count < kMaxCenterColumn; ++n) {
    if ((node_flags[n] & pole_axis::kNodeCenterFlag) != 0U) {
      v_r_node[n] = saved_vr[count];
      v_z_node[n] = saved_vz[count];
      ++count;
    }
  }
}

__global__ void state_supply_boundary_donor_radial_avg_kernel(
    const double* __restrict__ rho_lag,
    const double* __restrict__ u_r_lag,
    const double* __restrict__ u_z_lag,
    const double* __restrict__ e_e_lag,
    const double* __restrict__ e_i_lag,
    const double* __restrict__ e_rad_lag,
    double* __restrict__ donor_rho_avg_bottom,
    double* __restrict__ donor_u_r_avg_bottom,
    double* __restrict__ donor_u_z_avg_bottom,
    double* __restrict__ donor_e_gas_avg_bottom,
    double* __restrict__ donor_E_rad_avg_bottom,
    double* __restrict__ donor_rho_avg_top,
    double* __restrict__ donor_u_r_avg_top,
    double* __restrict__ donor_u_z_avg_top,
    double* __restrict__ donor_e_gas_avg_top,
    double* __restrict__ donor_E_rad_avg_top,
    const int nr,
    const int nz) {
  __shared__ double rho_bottom[256];
  __shared__ double u_r_bottom[256];
  __shared__ double u_z_bottom[256];
  __shared__ double e_gas_bottom[256];
  __shared__ double E_rad_bottom[256];
  __shared__ double rho_top[256];
  __shared__ double u_r_top[256];
  __shared__ double u_z_top[256];
  __shared__ double e_gas_top[256];
  __shared__ double E_rad_top[256];

  double sum_rho_bottom = 0.0;
  double sum_u_r_bottom = 0.0;
  double sum_u_z_bottom = 0.0;
  double sum_e_gas_bottom = 0.0;
  double sum_E_rad_bottom = 0.0;
  double sum_rho_top = 0.0;
  double sum_u_r_top = 0.0;
  double sum_u_z_top = 0.0;
  double sum_e_gas_top = 0.0;
  double sum_E_rad_top = 0.0;

  for (int i = threadIdx.x; i < nr; i += blockDim.x) {
    const int bottom = detail::cell_index(i, 0, nz);
    const int top = detail::cell_index(i, nz - 1, nz);
    if (rho_lag != nullptr) {
      sum_rho_bottom += fmax(rho_lag[bottom], 0.0);
      sum_rho_top += fmax(rho_lag[top], 0.0);
    }
    if (u_r_lag != nullptr) {
      sum_u_r_bottom += u_r_lag[bottom];
      sum_u_r_top += u_r_lag[top];
    }
    if (u_z_lag != nullptr) {
      sum_u_z_bottom += u_z_lag[bottom];
      sum_u_z_top += u_z_lag[top];
    }
    if (e_e_lag != nullptr && e_i_lag != nullptr) {
      sum_e_gas_bottom += fmax(e_e_lag[bottom], 0.0) + fmax(e_i_lag[bottom], 0.0);
      sum_e_gas_top += fmax(e_e_lag[top], 0.0) + fmax(e_i_lag[top], 0.0);
    }
    if (e_rad_lag != nullptr) {
      sum_E_rad_bottom += fmax(e_rad_lag[bottom], 0.0);
      sum_E_rad_top += fmax(e_rad_lag[top], 0.0);
    }
  }

  const int t = threadIdx.x;
  rho_bottom[t] = sum_rho_bottom;
  u_r_bottom[t] = sum_u_r_bottom;
  u_z_bottom[t] = sum_u_z_bottom;
  e_gas_bottom[t] = sum_e_gas_bottom;
  E_rad_bottom[t] = sum_E_rad_bottom;
  rho_top[t] = sum_rho_top;
  u_r_top[t] = sum_u_r_top;
  u_z_top[t] = sum_u_z_top;
  e_gas_top[t] = sum_e_gas_top;
  E_rad_top[t] = sum_E_rad_top;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (t < stride) {
      rho_bottom[t] += rho_bottom[t + stride];
      u_r_bottom[t] += u_r_bottom[t + stride];
      u_z_bottom[t] += u_z_bottom[t + stride];
      e_gas_bottom[t] += e_gas_bottom[t + stride];
      E_rad_bottom[t] += E_rad_bottom[t + stride];
      rho_top[t] += rho_top[t + stride];
      u_r_top[t] += u_r_top[t + stride];
      u_z_top[t] += u_z_top[t + stride];
      e_gas_top[t] += e_gas_top[t + stride];
      E_rad_top[t] += E_rad_top[t + stride];
    }
    __syncthreads();
  }

  if (t == 0) {
    const double inv_nr = nr > 0 ? 1.0 / static_cast<double>(nr) : 0.0;
    if (donor_rho_avg_bottom != nullptr) {
      donor_rho_avg_bottom[0] = rho_bottom[0] * inv_nr;
    }
    if (donor_u_r_avg_bottom != nullptr) {
      donor_u_r_avg_bottom[0] = u_r_bottom[0] * inv_nr;
    }
    if (donor_u_z_avg_bottom != nullptr) {
      donor_u_z_avg_bottom[0] = u_z_bottom[0] * inv_nr;
    }
    if (donor_e_gas_avg_bottom != nullptr) {
      donor_e_gas_avg_bottom[0] = e_gas_bottom[0] * inv_nr;
    }
    if (donor_E_rad_avg_bottom != nullptr) {
      donor_E_rad_avg_bottom[0] = E_rad_bottom[0] * inv_nr;
    }
    if (donor_rho_avg_top != nullptr) {
      donor_rho_avg_top[0] = rho_top[0] * inv_nr;
    }
    if (donor_u_r_avg_top != nullptr) {
      donor_u_r_avg_top[0] = u_r_top[0] * inv_nr;
    }
    if (donor_u_z_avg_top != nullptr) {
      donor_u_z_avg_top[0] = u_z_top[0] * inv_nr;
    }
    if (donor_e_gas_avg_top != nullptr) {
      donor_e_gas_avg_top[0] = e_gas_top[0] * inv_nr;
    }
    if (donor_E_rad_avg_top != nullptr) {
      donor_E_rad_avg_top[0] = E_rad_top[0] * inv_nr;
    }
  }
}

__global__ void state_supply_boundary_flux_2d_rz_kernel(
    const double* __restrict__ x_r_ref,
    const double* __restrict__ rho_lag,
    const double* __restrict__ u_z_lag,
    const double* __restrict__ e_e_lag,
    const double* __restrict__ e_i_lag,
    const double* __restrict__ e_rad_lag,
    double* __restrict__ boundary_mass_flux_z_bottom,
    double* __restrict__ boundary_mass_flux_z_top,
    double* __restrict__ boundary_momentum_flux_z_bottom,
    double* __restrict__ boundary_momentum_flux_z_top,
    double* __restrict__ boundary_energy_flux_z_bottom,
    double* __restrict__ boundary_energy_flux_z_top,
    double* __restrict__ boundary_rad_flux_z_bottom,
    double* __restrict__ boundary_rad_flux_z_top,
    const int nr,
    const int nz,
    const double dt,
    const int bottom_active,
    const int top_active,
    const double bottom_rho,
    const double bottom_u_z,
    const double bottom_e_gas,
    const double bottom_E_rad,
    const double top_rho,
    const double top_u_z,
    const double top_e_gas,
    const double top_E_rad,
    const int use_radial_average,
    const double* __restrict__ donor_rho_avg_bottom,
    const double* __restrict__ donor_u_z_avg_bottom,
    const double* __restrict__ donor_e_gas_avg_bottom,
    const double* __restrict__ donor_E_rad_avg_bottom,
    const double* __restrict__ donor_rho_avg_top,
    const double* __restrict__ donor_u_z_avg_top,
    const double* __restrict__ donor_e_gas_avg_top,
    const double* __restrict__ donor_E_rad_avg_top) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nr) {
    return;
  }

  if (bottom_active != 0) {
    const int c = detail::cell_index(i, 0, nz);
    const double v_n = bottom_u_z;
    const double u_z_interior =
        (u_z_lag != nullptr && isfinite(u_z_lag[c])) ? u_z_lag[c] : v_n;
    const bool supply_donor = v_n >= 0.0;
    double donor_rho = bottom_rho;
    double donor_u_z = bottom_u_z;
    double donor_e_gas = bottom_e_gas;
    double donor_E_rad = bottom_E_rad;
    if (!supply_donor) {
      if (use_radial_average != 0) {
        donor_rho =
            (donor_rho_avg_bottom != nullptr) ? donor_rho_avg_bottom[0] : bottom_rho;
        donor_u_z =
            (donor_u_z_avg_bottom != nullptr) ? donor_u_z_avg_bottom[0] : u_z_interior;
        donor_e_gas = (donor_e_gas_avg_bottom != nullptr)
                          ? donor_e_gas_avg_bottom[0]
                          : bottom_e_gas;
        donor_E_rad = (donor_E_rad_avg_bottom != nullptr)
                          ? donor_E_rad_avg_bottom[0]
                          : bottom_E_rad;
      } else {
        donor_rho = (rho_lag != nullptr) ? fmax(rho_lag[c], 0.0) : bottom_rho;
        donor_u_z = u_z_interior;
        donor_e_gas = ((e_e_lag != nullptr) && (e_i_lag != nullptr))
                          ? (fmax(e_e_lag[c], 0.0) + fmax(e_i_lag[c], 0.0))
                          : bottom_e_gas;
        donor_E_rad = (e_rad_lag != nullptr) ? fmax(e_rad_lag[c], 0.0) : bottom_E_rad;
      }
    }
    const double area = z_face_area_rz(x_r_ref, i, 0, nz);
    const double dV = v_n * area * dt;
    const double dm = donor_rho * dV;
    if (boundary_mass_flux_z_bottom != nullptr) {
      boundary_mass_flux_z_bottom[i] = dm;
    }
    if (boundary_momentum_flux_z_bottom != nullptr) {
      boundary_momentum_flux_z_bottom[i] = dm * donor_u_z;
    }
    if (boundary_energy_flux_z_bottom != nullptr) {
      boundary_energy_flux_z_bottom[i] = dm * donor_e_gas;
    }
    if (boundary_rad_flux_z_bottom != nullptr) {
      boundary_rad_flux_z_bottom[i] = donor_E_rad * dV;
    }
  }

  if (top_active != 0) {
    const int c = detail::cell_index(i, nz - 1, nz);
    const double v_n = top_u_z;
    const double u_z_interior =
        (u_z_lag != nullptr && isfinite(u_z_lag[c])) ? u_z_lag[c] : v_n;
    const bool interior_donor = v_n >= 0.0;
    double donor_rho = top_rho;
    double donor_u_z = top_u_z;
    double donor_e_gas = top_e_gas;
    double donor_E_rad = top_E_rad;
    if (interior_donor) {
      if (use_radial_average != 0) {
        donor_rho = (donor_rho_avg_top != nullptr) ? donor_rho_avg_top[0] : top_rho;
        donor_u_z = (donor_u_z_avg_top != nullptr) ? donor_u_z_avg_top[0] : u_z_interior;
        donor_e_gas =
            (donor_e_gas_avg_top != nullptr) ? donor_e_gas_avg_top[0] : top_e_gas;
        donor_E_rad =
            (donor_E_rad_avg_top != nullptr) ? donor_E_rad_avg_top[0] : top_E_rad;
      } else {
        donor_rho = (rho_lag != nullptr) ? fmax(rho_lag[c], 0.0) : top_rho;
        donor_u_z = u_z_interior;
        donor_e_gas = ((e_e_lag != nullptr) && (e_i_lag != nullptr))
                          ? (fmax(e_e_lag[c], 0.0) + fmax(e_i_lag[c], 0.0))
                          : top_e_gas;
        donor_E_rad = (e_rad_lag != nullptr) ? fmax(e_rad_lag[c], 0.0) : top_E_rad;
      }
    }
    const double area = z_face_area_rz(x_r_ref, i, nz, nz);
    const double dV = v_n * area * dt;
    const double dm_domain = -donor_rho * dV;
    if (boundary_mass_flux_z_top != nullptr) {
      boundary_mass_flux_z_top[i] = dm_domain;
    }
    if (boundary_momentum_flux_z_top != nullptr) {
      boundary_momentum_flux_z_top[i] = dm_domain * donor_u_z;
    }
    if (boundary_energy_flux_z_top != nullptr) {
      boundary_energy_flux_z_top[i] = dm_domain * donor_e_gas;
    }
    if (boundary_rad_flux_z_top != nullptr) {
      boundary_rad_flux_z_top[i] = -donor_E_rad * dV;
    }
  }
}

__device__ inline void apply_hydro_face_flux(double& mass,
                                             double& mom_r,
                                             double& mom_z,
                                             double& energy_e,
                                             double& energy_i,
                                             const double dV,
                                             const int donor,
                                             const double sign,
                                             const double* __restrict__ rho,
                                             const double* __restrict__ u_r,
                                             const double* __restrict__ u_z,
                                             const double* __restrict__ ee,
                                             const double* __restrict__ ei,
                                             const double* __restrict__ species_Y_lag = nullptr,
                                             double* species_ext = nullptr,
                                             const double* __restrict__ hot_e_eps_lag = nullptr,
                                             double* hot_e_eps_ext = nullptr,
                                             const double* __restrict__ burn_eps_lag = nullptr,
                                             double* burn_eps_ext = nullptr) {
  if (!finite_nonzero(dV)) {
    return;
  }
  const double rho_d = fmax(rho[donor], 0.0);
  const double dm = dV * rho_d;
  mass += sign * dm;
  if (species_Y_lag != nullptr && species_ext != nullptr) {
    for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
      species_ext[s] +=
          sign * dm * species_Y_lag[donor * tenryu::burn::kNumSpecies + s];
    }
  }
  if (hot_e_eps_lag != nullptr && hot_e_eps_ext != nullptr) {
    *hot_e_eps_ext += sign * dm * fmax(hot_e_eps_lag[donor], 0.0);
  }
  if (burn_eps_lag != nullptr && burn_eps_ext != nullptr) {
    *burn_eps_ext += sign * dm * fmax(burn_eps_lag[donor], 0.0);
  }
  mom_r += sign * dm * u_r[donor];
  mom_z += sign * dm * u_z[donor];
  energy_e += sign * dm * fmax(ee[donor], 0.0);
  energy_i += sign * dm * fmax(ei[donor], 0.0);
}

__device__ inline double clamp01_device(const double x) {
  return fmin(1.0, fmax(0.0, x));
}

__device__ inline double rz_corner_kinetic_for_cell(
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int c,
    const int i,
    const int j,
    const int nz) {
  const int n00 = detail::node_index(i, j, nz);
  const int n10 = detail::node_index(i + 1, j, nz);
  const int n11 = detail::node_index(i + 1, j + 1, nz);
  const int n01 = detail::node_index(i, j + 1, nz);
  const double m_cell = fmax(mass[c], 0.0);
  const double R_L = 0.5 * (x_r[n00] + x_r[n01]);
  const double R_R = 0.5 * (x_r[n10] + x_r[n11]);
  const double denom = (R_L + R_R) * 6.0;

  double m00 = 0.25 * m_cell;
  double m10 = 0.25 * m_cell;
  double m11 = 0.25 * m_cell;
  double m01 = 0.25 * m_cell;
  if (denom > 0.0 && isfinite(denom)) {
    const double w_L = (2.0 * R_L + R_R) / denom;
    const double w_R = (R_L + 2.0 * R_R) / denom;
    m00 = w_L * m_cell;
    m10 = w_R * m_cell;
    m11 = w_R * m_cell;
    m01 = w_L * m_cell;
  }

  const double vr00 = v_r_node[n00];
  const double vz00 = v_z_node[n00];
  const double vr10 = v_r_node[n10];
  const double vz10 = v_z_node[n10];
  const double vr11 = v_r_node[n11];
  const double vz11 = v_z_node[n11];
  const double vr01 = v_r_node[n01];
  const double vz01 = v_z_node[n01];
  return 0.5 * (m00 * (vr00 * vr00 + vz00 * vz00) +
                m10 * (vr10 * vr10 + vz10 * vz10) +
                m11 * (vr11 * vr11 + vz11 * vz11) +
	                m01 * (vr01 * vr01 + vz01 * vz01));
}

__device__ inline double rz_physical_corner_kinetic_for_cell(
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const std::uint8_t* __restrict__ cell_nverts,
    const int c,
    const int nz,
    const int corner_mass_convention,
    rz::CornerMassFallbackRecorder* fallback_recorder,
    const int fallback_stage,
    const int orientation) {
  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int nodes[4] = {
      i * stride + j,
      (i + 1) * stride + j,
      (i + 1) * stride + (j + 1),
      i * stride + (j + 1)};
  const int active_nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  const double m_cell = fmax(mass[c], 0.0);
  double m_corner[4] = {0.0, 0.0, 0.0, 0.0};
  rz::CornerMassFallbackProbe probe{};
  rz::compute_rz_corner_masses_from_nodes(
      c, nz, m_cell, x_r, x_z, cell_nverts, m_corner, &probe,
      corner_mass_convention);
  if (probe.fired == 1) {
    rz::record_corner_mass_fallback(fallback_recorder,
                                    probe,
                                    true,
                                    c,
                                    fallback_stage,
                                    orientation);
  }
  double kinetic = 0.0;
  for (int k = 0; k < active_nverts && k < 4; ++k) {
    const int n = nodes[k];
    const double cm =
        (m_corner[k] > 0.0 && isfinite(m_corner[k])) ? m_corner[k] : 0.0;
    const double vr = v_r_node[n];
    const double vz = v_z_node[n];
    if (cm > 0.0 && isfinite(vr) && isfinite(vz)) {
      kinetic += 0.5 * cm * (vr * vr + vz * vz);
    }
  }
  return isfinite(kinetic) ? kinetic : 0.0;
}

__device__ inline double rz_button_kinetic_for_cell(
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int button_outer_node_ring,
    const int nz) {
  const double m_cell = fmax(mass[0], 0.0);
  if (!(m_cell > 0.0) || !isfinite(m_cell)) {
    return 0.0;
  }
  double centroid_r = 0.0;
  double centroid_z = 0.0;
  rz::button_polygon_area_centroid_from_nodes(
      x_r, x_z, button_outer_node_ring, nz, &centroid_r, &centroid_z);
  const double volume = rz::button_polygon_volume_from_nodes(
      x_r, x_z, button_outer_node_ring, nz);
  if (!(volume > 0.0) || !isfinite(volume)) {
    return 0.0;
  }

  const double rho_button = m_cell / volume;
  double kinetic = 0.0;
  for (int k = 0; k <= nz; ++k) {
    const int n = rz::button_seam_node_index(button_outer_node_ring, k, nz);
    const double m_corner = rz::button_corner_mass_exact_subpolygon(
        rho_button, x_r, x_z, button_outer_node_ring, nz, k, centroid_r,
        centroid_z);
    kinetic += 0.5 * m_corner *
               (v_r_node[n] * v_r_node[n] + v_z_node[n] * v_z_node[n]);
  }
  return isfinite(kinetic) ? kinetic : 0.0;
}

__global__ void apply_button_center_cell_velocity_kernel(
    double* __restrict__ v_r_cell,
    double* __restrict__ v_z_cell,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int nr,
    const int nz,
    const int button_outer_node_ring) {
  if (button_outer_node_ring < 1) {
    return;
  }
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c < n_cells) {
    const int i = c / nz;
    if (c != 0 && i < button_outer_node_ring) {
      v_r_cell[c] = 0.0;
      v_z_cell[c] = 0.0;
    }
  }
  if (c == 0) {
    double vr = 0.0;
    double vz = 0.0;
    for (int k = 0; k <= nz; ++k) {
      const int n = rz::button_seam_node_index(button_outer_node_ring, k, nz);
      vr += v_r_node[n];
      vz += v_z_node[n];
    }
    const double inv = 1.0 / static_cast<double>(nz + 1);
    v_r_cell[0] = vr * inv;
    v_z_cell[0] = vz * inv;
  }
}

__global__ void build_total_energy_remap_state_kernel(
    double* __restrict__ e_tot_lag,
    double* __restrict__ ye_int_lag,
    const double* __restrict__ mass,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int nr,
    const int nz,
    const int button_outer_node_ring) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  if (button_outer_node_ring > 0 && c != 0 && i < button_outer_node_ring) {
    e_tot_lag[c] = 0.0;
    ye_int_lag[c] = 0.5;
    return;
  }
  const double e_e = fmax(ee[c], 0.0);
  const double e_i = fmax(ei[c], 0.0);
  const double e_int = e_e + e_i;
  const double m = fmax(mass[c], 0.0);
  const double K = (button_outer_node_ring > 0 && c == 0)
                       ? rz_button_kinetic_for_cell(
                             mass, x_r, x_z, v_r_node, v_z_node,
                             button_outer_node_ring, nz)
                       : rz_corner_kinetic_for_cell(
                             mass, x_r, v_r_node, v_z_node, c, i, j, nz);
  e_tot_lag[c] = (m > 0.0) ? (e_int + K / m) : e_int;
  ye_int_lag[c] = (e_int > 0.0 && isfinite(e_int)) ? clamp01_device(e_e / e_int)
                                                    : 0.5;
}

__global__ void build_total_energy_remap_state_physical_ke_kernel(
    double* __restrict__ e_tot_lag,
    double* __restrict__ ye_int_lag,
    const double* __restrict__ mass,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const std::uint8_t* __restrict__ cell_nverts,
    rz::CornerMassFallbackRecorder* fallback_recorder,
    const int corner_mass_convention,
    const int nr,
    const int nz,
    const int button_outer_node_ring) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = c / nz;
  if (button_outer_node_ring > 0 && c != 0 && i < button_outer_node_ring) {
    e_tot_lag[c] = 0.0;
    ye_int_lag[c] = 0.5;
    return;
  }
  const double e_e = fmax(ee[c], 0.0);
  const double e_i = fmax(ei[c], 0.0);
  const double e_int = e_e + e_i;
  const double m = fmax(mass[c], 0.0);
  const double K = (button_outer_node_ring > 0 && c == 0)
                       ? rz_button_kinetic_for_cell(
                             mass, x_r, x_z, v_r_node, v_z_node,
                             button_outer_node_ring, nz)
                       : rz_physical_corner_kinetic_for_cell(
                             mass, x_r, x_z, v_r_node, v_z_node, cell_nverts,
                             c, nz, corner_mass_convention, fallback_recorder,
                             rz::kCornerMassFallbackStageStructuredPhysicalKeBuild,
                             -2);
  e_tot_lag[c] = (m > 0.0) ? (e_int + K / m) : e_int;
  ye_int_lag[c] = (e_int > 0.0 && isfinite(e_int))
                      ? clamp01_device(e_e / e_int)
                      : 0.5;
}

__device__ inline double boundary_total_energy_flux(
    const double internal_energy_flux,
    const double mass_flux,
    const double momentum_z_flux) {
  if (!(mass_flux != 0.0) || !isfinite(mass_flux) ||
      !isfinite(momentum_z_flux)) {
    return internal_energy_flux;
  }
  return internal_energy_flux +
         0.5 * momentum_z_flux * momentum_z_flux / mass_flux;
}

__global__ void promote_boundary_energy_flux_to_total_kernel(
    double* __restrict__ boundary_energy_flux_z_bottom,
    double* __restrict__ boundary_energy_flux_z_top,
    const double* __restrict__ boundary_mass_flux_z_bottom,
    const double* __restrict__ boundary_mass_flux_z_top,
    const double* __restrict__ boundary_momentum_flux_z_bottom,
    const double* __restrict__ boundary_momentum_flux_z_top,
    const int nr) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nr) {
    return;
  }
  if (boundary_energy_flux_z_bottom != nullptr) {
    boundary_energy_flux_z_bottom[i] = boundary_total_energy_flux(
        boundary_energy_flux_z_bottom[i],
        boundary_mass_flux_z_bottom != nullptr ? boundary_mass_flux_z_bottom[i] : 0.0,
        boundary_momentum_flux_z_bottom != nullptr
            ? boundary_momentum_flux_z_bottom[i]
            : 0.0);
  }
  if (boundary_energy_flux_z_top != nullptr) {
    boundary_energy_flux_z_top[i] = boundary_total_energy_flux(
        boundary_energy_flux_z_top[i],
        boundary_mass_flux_z_top != nullptr ? boundary_mass_flux_z_top[i] : 0.0,
        boundary_momentum_flux_z_top != nullptr ? boundary_momentum_flux_z_top[i]
                                                : 0.0);
  }
}

__device__ inline void apply_hydro_face_flux_total_energy(
    double& mass,
    double& mom_r,
    double& mom_z,
    double& total_energy,
    double& ye_mass,
    const double dV,
    const int donor,
    const double sign,
    const double* __restrict__ rho,
    const double* __restrict__ u_r,
    const double* __restrict__ u_z,
    const double* __restrict__ e_tot,
    const double* __restrict__ ye_int) {
  if (!finite_nonzero(dV)) {
    return;
  }
  const double rho_d = fmax(rho[donor], 0.0);
  const double dm = dV * rho_d;
  mass += sign * dm;
  mom_r += sign * dm * u_r[donor];
  mom_z += sign * dm * u_z[donor];
  total_energy += sign * dm * fmax(e_tot[donor], 0.0);
  ye_mass += sign * dm * clamp01_device(ye_int[donor]);
}

__global__ void ale_remap_2d_rz_kernel(
    const double* __restrict__ rho_lag,
    const double* __restrict__ u_r_lag,
    const double* __restrict__ u_z_lag,
    const double* __restrict__ e_e_lag,
    const double* __restrict__ e_i_lag,
    const double* __restrict__ vol_lag,
    const double* __restrict__ mass_lag,
    const double* __restrict__ x_r_lag,
    const double* __restrict__ x_z_lag,
    const double* __restrict__ x_r_ref,
    const double* __restrict__ x_z_ref,
    const double* __restrict__ vol_ref,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ boundary_mass_flux_z_bottom,
    const double* __restrict__ boundary_mass_flux_z_top,
    const double* __restrict__ boundary_momentum_flux_z_bottom,
    const double* __restrict__ boundary_momentum_flux_z_top,
    const double* __restrict__ boundary_energy_flux_z_bottom,
    const double* __restrict__ boundary_energy_flux_z_top,
    double* __restrict__ rho_new,
    double* __restrict__ u_r_new,
    double* __restrict__ u_z_new,
    double* __restrict__ e_e_new,
    double* __restrict__ e_i_new,
    double* __restrict__ mass_new,
    const double* __restrict__ zbar,
    double* __restrict__ mass_floor_delta,
    double* __restrict__ energy_floor_delta,
    const int nr,
	    const int nz,
	    const int button_outer_node_ring,
	    const double rho_floor,
	    const double te_floor,
	    const double ti_floor,
	    const double gamma,
	    const double A,
    const double* __restrict__ burn_species_Y_lag,
    double* __restrict__ burn_species_Y_new,
    const double* __restrict__ hot_e_eps_lag,
    double* __restrict__ hot_e_eps_new,
    const double* __restrict__ burn_eps_lag,
    double* __restrict__ burn_eps_new) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const bool remap_species =
      (burn_species_Y_lag != nullptr && burn_species_Y_new != nullptr);
  const bool remap_hot_e_eps =
      (hot_e_eps_lag != nullptr && hot_e_eps_new != nullptr);
  const bool remap_burn_eps =
      (burn_eps_lag != nullptr && burn_eps_new != nullptr);

	  const int i = c / nz;
		  const int j = c - i * nz;
		  if (rz_is_dormant_button_cell(c, i, button_outer_node_ring)) {
		    rho_new[c] = 0.0;
		    mass_new[c] = 0.0;
		    u_r_new[c] = 0.0;
		    u_z_new[c] = 0.0;
		    e_e_new[c] = 0.0;
		    e_i_new[c] = 0.0;
        if (remap_species) {
          for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
            burn_species_Y_new[c * tenryu::burn::kNumSpecies + s] = 0.0;
          }
        }
        if (remap_hot_e_eps) {
          hot_e_eps_new[c] = 0.0;
        }
        if (remap_burn_eps) {
          burn_eps_new[c] = 0.0;
        }
	    return;
	  }
	  const double V_lag = fmax(vol_lag[c], kTinyVolume);
	  const double V_ref = fmax(vol_ref[c], kTinyVolume);
	  const double rho_c = fmax(rho_lag[c], 0.0);
	  double mass = rho_c * V_lag;
  double species_ext[tenryu::burn::kNumSpecies];
  if (remap_species) {
    for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
      species_ext[s] =
          mass * burn_species_Y_lag[c * tenryu::burn::kNumSpecies + s];
    }
  }
  double hot_e_eps_ext = 0.0;
  if (remap_hot_e_eps) {
    hot_e_eps_ext = mass * fmax(hot_e_eps_lag[c], 0.0);
  }
  double burn_eps_ext = 0.0;
  if (remap_burn_eps) {
    burn_eps_ext = mass * fmax(burn_eps_lag[c], 0.0);
  }
  double mom_r = mass * u_r_lag[c];
  double mom_z = mass * u_z_lag[c];
  double energy_e = mass * fmax(e_e_lag[c], 0.0);
  double energy_i = mass * fmax(e_i_lag[c], 0.0);
  const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : 1.0;
  const double A_safe = fmax(A, 1.0e-30);
  const double gm1 = fmax(gamma - 1.0, 1.0e-30);
  const double cv_i =
      tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  const double cv_e =
      z * tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  const double cv_sum = cv_e + cv_i;
	  const double e_e_fraction = (cv_sum > 0.0) ? (cv_e / cv_sum) : 0.5;
	  const double e_i_fraction = 1.0 - e_e_fraction;

		  if (rz_is_button_cell(c, button_outer_node_ring)) {
		    double seam_swept_sum = 0.0;
		    for (int js = 0; js < nz; ++js) {
		      const double dV = rz_button_seam_swept_volume_ref(
		          x_r_lag, x_z_lag, x_r_ref, x_z_ref, button_outer_node_ring, nz, js);
		      seam_swept_sum += dV;
		      const int shell = detail::cell_index(button_outer_node_ring, js, nz);
		      const int donor = rz_donor_cell(c, shell, dV);
		      apply_hydro_face_flux(mass, mom_r, mom_z, energy_e, energy_i, dV, donor,
		                            -1.0, rho_lag, u_r_lag, u_z_lag, e_e_lag, e_i_lag,
                                remap_species ? burn_species_Y_lag : nullptr,
                                remap_species ? species_ext : nullptr,
                                remap_hot_e_eps ? hot_e_eps_lag : nullptr,
                                remap_hot_e_eps ? &hot_e_eps_ext : nullptr,
                                remap_burn_eps ? burn_eps_lag : nullptr,
                                remap_burn_eps ? &burn_eps_ext : nullptr);
		    }
		    const double V_button_gcl = fmax(V_lag - seam_swept_sum, kTinyVolume);
		    const double mass_floor = rho_floor * V_button_gcl;
		    const double mass_raw = mass;
		    if (!(mass > mass_floor) || !isfinite(mass)) {
		      mass = mass_floor;
		      if (mass_floor_delta != nullptr && isfinite(mass_raw)) {
		        atomicAdd(mass_floor_delta, mass - mass_raw);
		      }
		    }
		
		    const double e_e_floor = cv_e * fmax(te_floor, 0.0);
		    const double e_i_floor = cv_i * fmax(ti_floor, 0.0);
		
		    double ee = (mass > 0.0) ? (energy_e / mass) : e_e_floor;
		    double ei = (mass > 0.0) ? (energy_i / mass) : e_i_floor;
		    if (!(ee >= e_e_floor) || !isfinite(ee)) {
		      if (energy_floor_delta != nullptr && isfinite(ee)) {
		        atomicAdd(energy_floor_delta, (e_e_floor - ee) * mass);
		      }
		      ee = e_e_floor;
		    }
		    if (!(ei >= e_i_floor) || !isfinite(ei)) {
		      if (energy_floor_delta != nullptr && isfinite(ei)) {
		        atomicAdd(energy_floor_delta, (e_i_floor - ei) * mass);
		      }
		      ei = e_i_floor;
		    }
		
		    mass_new[c] = mass;
		    rho_new[c] = mass / V_button_gcl;
		    u_r_new[c] = (mass > 0.0 && isfinite(mom_r)) ? (mom_r / mass) : 0.0;
		    u_z_new[c] = (mass > 0.0 && isfinite(mom_z)) ? (mom_z / mass) : 0.0;
		    e_e_new[c] = ee;
		    e_i_new[c] = ei;
        if (remap_species) {
          for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
            burn_species_Y_new[c * tenryu::burn::kNumSpecies + s] =
                (mass > 0.0 && isfinite(mass)) ? species_ext[s] / mass : 0.0;
          }
        }
        if (remap_hot_e_eps) {
          hot_e_eps_new[c] =
              (mass > 0.0 && isfinite(mass)) ? fmax(hot_e_eps_ext / mass, 0.0)
                                             : 0.0;
        }
        if (remap_burn_eps) {
          burn_eps_new[c] =
              (mass > 0.0 && isfinite(mass)) ? fmax(burn_eps_ext / mass, 0.0)
                                             : 0.0;
        }
		    return;
		  } else {
	    if (i + 1 < nr) {
	      const int Kp = detail::cell_index(i + 1, j, nz);
	      if (!rz_button_face_touches_dormant(c, Kp, nz, button_outer_node_ring)) {
	        const double dV = rz_swept_volume_r_face_ref_topology(
	            x_r_lag, x_z_lag, x_r_ref, x_z_ref, cell_nverts, i + 1, j, nr,
	            nz, button_outer_node_ring);
	        const int donor = rz_donor_cell(c, Kp, dV);
	        apply_hydro_face_flux(mass, mom_r, mom_z, energy_e, energy_i, dV, donor,
	                              -1.0, rho_lag, u_r_lag, u_z_lag, e_e_lag, e_i_lag,
                                  remap_species ? burn_species_Y_lag : nullptr,
                                  remap_species ? species_ext : nullptr,
                                  remap_hot_e_eps ? hot_e_eps_lag : nullptr,
                                  remap_hot_e_eps ? &hot_e_eps_ext : nullptr,
                                  remap_burn_eps ? burn_eps_lag : nullptr,
                                  remap_burn_eps ? &burn_eps_ext : nullptr);
	      }
	    }
	    if (i > 0) {
	      double dV = 0.0;
	      int donor = c;
	      const bool button_seam_face =
	          rz_button_enabled(button_outer_node_ring) &&
	          i == button_outer_node_ring;
	      const int K = button_seam_face ? 0 : detail::cell_index(i - 1, j, nz);
	      if (rz_button_enabled(button_outer_node_ring) &&
	          i == button_outer_node_ring) {
	        dV = rz_button_seam_swept_volume_ref(
	            x_r_lag, x_z_lag, x_r_ref, x_z_ref, button_outer_node_ring, nz, j);
	        donor = rz_donor_cell(0, c, dV);
	      } else {
	        dV = rz_swept_volume_r_face_ref_topology(
	            x_r_lag, x_z_lag, x_r_ref, x_z_ref, cell_nverts, i, j, nr, nz,
	            button_outer_node_ring);
	        donor = rz_donor_cell(K, c, dV);
	      }
	      if (!rz_button_face_touches_dormant(K, c, nz, button_outer_node_ring)) {
	        apply_hydro_face_flux(mass, mom_r, mom_z, energy_e, energy_i, dV, donor,
	                              1.0, rho_lag, u_r_lag, u_z_lag, e_e_lag, e_i_lag,
                                  remap_species ? burn_species_Y_lag : nullptr,
                                  remap_species ? species_ext : nullptr,
                                  remap_hot_e_eps ? hot_e_eps_lag : nullptr,
                                  remap_hot_e_eps ? &hot_e_eps_ext : nullptr,
                                  remap_burn_eps ? burn_eps_lag : nullptr,
                                  remap_burn_eps ? &burn_eps_ext : nullptr);
	      }
	    }
	    if (j + 1 < nz) {
	      const int Kp = detail::cell_index(i, j + 1, nz);
	      if (!rz_button_face_touches_dormant(c, Kp, nz, button_outer_node_ring)) {
	        const double dV = rz_swept_volume_z_face_ref_topology(
	            x_r_lag, x_z_lag, x_r_ref, x_z_ref, cell_nverts, i, j + 1, nr,
	            nz, button_outer_node_ring);
	        const int donor = rz_donor_cell(c, Kp, dV);
	        apply_hydro_face_flux(mass, mom_r, mom_z, energy_e, energy_i, dV, donor,
	                              -1.0, rho_lag, u_r_lag, u_z_lag, e_e_lag, e_i_lag,
                                  remap_species ? burn_species_Y_lag : nullptr,
                                  remap_species ? species_ext : nullptr,
                                  remap_hot_e_eps ? hot_e_eps_lag : nullptr,
                                  remap_hot_e_eps ? &hot_e_eps_ext : nullptr,
                                  remap_burn_eps ? burn_eps_lag : nullptr,
                                  remap_burn_eps ? &burn_eps_ext : nullptr);
	      }
	    } else if (boundary_mass_flux_z_top != nullptr) {
	      const double dm_bz = boundary_mass_flux_z_top[i];
	      mass += dm_bz;
        if (remap_species && dm_bz < 0.0) {
          for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
            species_ext[s] +=
                dm_bz * burn_species_Y_lag[c * tenryu::burn::kNumSpecies + s];
          }
        }
        if (remap_hot_e_eps && dm_bz < 0.0) {
          hot_e_eps_ext += dm_bz * fmax(hot_e_eps_lag[c], 0.0);
        }
        if (remap_burn_eps && dm_bz < 0.0) {
          burn_eps_ext += dm_bz * fmax(burn_eps_lag[c], 0.0);
        }
	      mom_z += boundary_momentum_flux_z_top[i];
	      energy_e += e_e_fraction * boundary_energy_flux_z_top[i];
	      energy_i += e_i_fraction * boundary_energy_flux_z_top[i];
	    }
	    if (j > 0) {
	      const int K = detail::cell_index(i, j - 1, nz);
	      if (!rz_button_face_touches_dormant(K, c, nz, button_outer_node_ring)) {
	        const double dV = rz_swept_volume_z_face_ref_topology(
	            x_r_lag, x_z_lag, x_r_ref, x_z_ref, cell_nverts, i, j, nr, nz,
	            button_outer_node_ring);
	        const int donor = rz_donor_cell(K, c, dV);
	        apply_hydro_face_flux(mass, mom_r, mom_z, energy_e, energy_i, dV, donor,
	                              1.0, rho_lag, u_r_lag, u_z_lag, e_e_lag, e_i_lag,
                                  remap_species ? burn_species_Y_lag : nullptr,
                                  remap_species ? species_ext : nullptr,
                                  remap_hot_e_eps ? hot_e_eps_lag : nullptr,
                                  remap_hot_e_eps ? &hot_e_eps_ext : nullptr,
                                  remap_burn_eps ? burn_eps_lag : nullptr,
                                  remap_burn_eps ? &burn_eps_ext : nullptr);
	      }
	    } else if (boundary_mass_flux_z_bottom != nullptr) {
	      const double dm_bz = boundary_mass_flux_z_bottom[i];
	      mass += dm_bz;
        if (remap_species && dm_bz < 0.0) {
          for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
            species_ext[s] +=
                dm_bz * burn_species_Y_lag[c * tenryu::burn::kNumSpecies + s];
          }
        }
        if (remap_hot_e_eps && dm_bz < 0.0) {
          hot_e_eps_ext += dm_bz * fmax(hot_e_eps_lag[c], 0.0);
        }
        if (remap_burn_eps && dm_bz < 0.0) {
          burn_eps_ext += dm_bz * fmax(burn_eps_lag[c], 0.0);
        }
	      mom_z += boundary_momentum_flux_z_bottom[i];
	      energy_e += e_e_fraction * boundary_energy_flux_z_bottom[i];
	      energy_i += e_i_fraction * boundary_energy_flux_z_bottom[i];
	    }
	  }

  const double mass_floor = rho_floor * V_ref;
  const double mass_raw = mass;
  if (!(mass > mass_floor) || !isfinite(mass)) {
    mass = mass_floor;
    if (mass_floor_delta != nullptr && isfinite(mass_raw)) {
      atomicAdd(mass_floor_delta, mass - mass_raw);
    }
  }

  const double e_e_floor = cv_e * fmax(te_floor, 0.0);
  const double e_i_floor = cv_i * fmax(ti_floor, 0.0);

  double ee = (mass > 0.0) ? (energy_e / mass) : e_e_floor;
  double ei = (mass > 0.0) ? (energy_i / mass) : e_i_floor;
  if (!(ee >= e_e_floor) || !isfinite(ee)) {
    if (energy_floor_delta != nullptr && isfinite(ee)) {
      atomicAdd(energy_floor_delta, (e_e_floor - ee) * mass);
    }
    ee = e_e_floor;
  }
  if (!(ei >= e_i_floor) || !isfinite(ei)) {
    if (energy_floor_delta != nullptr && isfinite(ei)) {
      atomicAdd(energy_floor_delta, (e_i_floor - ei) * mass);
    }
    ei = e_i_floor;
  }

  mass_new[c] = mass;
  rho_new[c] = mass / V_ref;
  u_r_new[c] = (mass > 0.0 && isfinite(mom_r)) ? (mom_r / mass) : 0.0;
  u_z_new[c] = (mass > 0.0 && isfinite(mom_z)) ? (mom_z / mass) : 0.0;
  e_e_new[c] = ee;
  e_i_new[c] = ei;
  if (remap_species) {
    for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
      burn_species_Y_new[c * tenryu::burn::kNumSpecies + s] =
          (mass > 0.0 && isfinite(mass)) ? species_ext[s] / mass : 0.0;
    }
  }
  if (remap_hot_e_eps) {
    hot_e_eps_new[c] =
        (mass > 0.0 && isfinite(mass)) ? fmax(hot_e_eps_ext / mass, 0.0)
                                       : 0.0;
  }
  if (remap_burn_eps) {
    burn_eps_new[c] =
        (mass > 0.0 && isfinite(mass)) ? fmax(burn_eps_ext / mass, 0.0)
                                       : 0.0;
  }
}

__global__ void ale_remap_2d_rz_total_energy_kernel(
    const double* __restrict__ rho_lag,
    const double* __restrict__ u_r_lag,
    const double* __restrict__ u_z_lag,
    const double* __restrict__ e_tot_lag,
    const double* __restrict__ ye_int_lag,
    const double* __restrict__ vol_lag,
    const double* __restrict__ mass_lag,
    const double* __restrict__ x_r_lag,
    const double* __restrict__ x_z_lag,
    const double* __restrict__ x_r_ref,
    const double* __restrict__ x_z_ref,
    const double* __restrict__ vol_ref,
    const double* __restrict__ boundary_mass_flux_z_bottom,
    const double* __restrict__ boundary_mass_flux_z_top,
    const double* __restrict__ boundary_momentum_flux_z_bottom,
    const double* __restrict__ boundary_momentum_flux_z_top,
    const double* __restrict__ boundary_total_energy_flux_z_bottom,
    const double* __restrict__ boundary_total_energy_flux_z_top,
    double* __restrict__ rho_new,
    double* __restrict__ u_r_new,
    double* __restrict__ u_z_new,
    double* __restrict__ total_energy_new,
    double* __restrict__ ye_int_new,
    double* __restrict__ mass_new,
	    double* __restrict__ mass_floor_delta,
	    const int nr,
	    const int nz,
	    const int button_outer_node_ring,
	    const double rho_floor,
	    const double e_e_fraction) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

	  const int i = c / nz;
		  const int j = c - i * nz;
		  if (rz_is_dormant_button_cell(c, i, button_outer_node_ring)) {
		    rho_new[c] = 0.0;
		    mass_new[c] = 0.0;
		    u_r_new[c] = 0.0;
		    u_z_new[c] = 0.0;
		    total_energy_new[c] = 0.0;
		    ye_int_new[c] = 0.0;
		    return;
		  }
	  const double V_lag = fmax(vol_lag[c], kTinyVolume);
	  const double V_ref = fmax(vol_ref[c], kTinyVolume);
  const double rho_c = fmax(rho_lag[c], 0.0);
  double mass = rho_c * V_lag;
  double mom_r = mass * u_r_lag[c];
  double mom_z = mass * u_z_lag[c];
	  double total_energy = mass * fmax(e_tot_lag[c], 0.0);
	  double ye_mass = mass * clamp01_device(ye_int_lag[c]);

		  if (rz_is_button_cell(c, button_outer_node_ring)) {
		    double seam_swept_sum = 0.0;
		    for (int js = 0; js < nz; ++js) {
		      const double dV = rz_button_seam_swept_volume_ref(
		          x_r_lag, x_z_lag, x_r_ref, x_z_ref, button_outer_node_ring, nz, js);
		      seam_swept_sum += dV;
		      const int shell = detail::cell_index(button_outer_node_ring, js, nz);
		      const int donor = rz_donor_cell(c, shell, dV);
		      apply_hydro_face_flux_total_energy(
		          mass, mom_r, mom_z, total_energy, ye_mass, dV, donor, -1.0,
		          rho_lag, u_r_lag, u_z_lag, e_tot_lag, ye_int_lag);
		    }
		    const double V_button_gcl = fmax(V_lag - seam_swept_sum, kTinyVolume);
		    const double mass_floor = rho_floor * V_button_gcl;
		    const double mass_raw = mass;
		    if (!(mass > mass_floor) || !isfinite(mass)) {
		      mass = mass_floor;
		      if (mass_floor_delta != nullptr && isfinite(mass_raw)) {
		        atomicAdd(mass_floor_delta, mass - mass_raw);
		      }
		    }
		
		    mass_new[c] = mass;
		    rho_new[c] = mass / V_button_gcl;
		    u_r_new[c] = (mass > 0.0 && isfinite(mom_r)) ? (mom_r / mass) : 0.0;
		    u_z_new[c] = (mass > 0.0 && isfinite(mom_z)) ? (mom_z / mass) : 0.0;
		    total_energy_new[c] = isfinite(total_energy) ? total_energy : 0.0;
		    ye_int_new[c] = (mass > 0.0 && isfinite(ye_mass))
		                        ? clamp01_device(ye_mass / mass)
		                        : 0.5;
		    return;
		  } else {
	    if (i + 1 < nr) {
	      const int Kp = detail::cell_index(i + 1, j, nz);
	      if (!rz_button_face_touches_dormant(c, Kp, nz, button_outer_node_ring)) {
	        const double dV = rz_swept_volume_r_face_ref_topology(
	            x_r_lag, x_z_lag, x_r_ref, x_z_ref, nullptr, i + 1, j, nr, nz,
	            button_outer_node_ring);
	        const int donor = rz_donor_cell(c, Kp, dV);
	        apply_hydro_face_flux_total_energy(
	            mass, mom_r, mom_z, total_energy, ye_mass, dV, donor, -1.0,
	            rho_lag, u_r_lag, u_z_lag, e_tot_lag, ye_int_lag);
	      }
	    }
	    if (i > 0) {
	      double dV = 0.0;
	      int donor = c;
	      const bool button_seam_face =
	          rz_button_enabled(button_outer_node_ring) &&
	          i == button_outer_node_ring;
	      const int K = button_seam_face ? 0 : detail::cell_index(i - 1, j, nz);
	      if (rz_button_enabled(button_outer_node_ring) &&
	          i == button_outer_node_ring) {
	        dV = rz_button_seam_swept_volume_ref(
	            x_r_lag, x_z_lag, x_r_ref, x_z_ref, button_outer_node_ring, nz, j);
	        donor = rz_donor_cell(0, c, dV);
	      } else {
	        dV = rz_swept_volume_r_face_ref_topology(
	            x_r_lag, x_z_lag, x_r_ref, x_z_ref, nullptr, i, j, nr, nz,
	            button_outer_node_ring);
	        donor = rz_donor_cell(K, c, dV);
	      }
	      if (!rz_button_face_touches_dormant(K, c, nz, button_outer_node_ring)) {
	        apply_hydro_face_flux_total_energy(
	            mass, mom_r, mom_z, total_energy, ye_mass, dV, donor, 1.0,
	            rho_lag, u_r_lag, u_z_lag, e_tot_lag, ye_int_lag);
	      }
	    }
	    if (j + 1 < nz) {
	      const int Kp = detail::cell_index(i, j + 1, nz);
	      if (!rz_button_face_touches_dormant(c, Kp, nz, button_outer_node_ring)) {
	        const double dV = rz_swept_volume_z_face_ref_topology(
	            x_r_lag, x_z_lag, x_r_ref, x_z_ref, nullptr, i, j + 1, nr, nz,
	            button_outer_node_ring);
	        const int donor = rz_donor_cell(c, Kp, dV);
	        apply_hydro_face_flux_total_energy(
	            mass, mom_r, mom_z, total_energy, ye_mass, dV, donor, -1.0,
	            rho_lag, u_r_lag, u_z_lag, e_tot_lag, ye_int_lag);
	      }
	    } else if (boundary_mass_flux_z_top != nullptr) {
	      const double dm = boundary_mass_flux_z_top[i];
	      mass += dm;
	      mom_z += boundary_momentum_flux_z_top[i];
	      total_energy += boundary_total_energy_flux_z_top[i];
	      const double ye_b = (dm > 0.0) ? clamp01_device(e_e_fraction)
	                                     : clamp01_device(ye_int_lag[c]);
	      ye_mass += dm * ye_b;
	    }
	    if (j > 0) {
	      const int K = detail::cell_index(i, j - 1, nz);
	      if (!rz_button_face_touches_dormant(K, c, nz, button_outer_node_ring)) {
	        const double dV = rz_swept_volume_z_face_ref_topology(
	            x_r_lag, x_z_lag, x_r_ref, x_z_ref, nullptr, i, j, nr, nz,
	            button_outer_node_ring);
	        const int donor = rz_donor_cell(K, c, dV);
	        apply_hydro_face_flux_total_energy(
	            mass, mom_r, mom_z, total_energy, ye_mass, dV, donor, 1.0,
	            rho_lag, u_r_lag, u_z_lag, e_tot_lag, ye_int_lag);
	      }
	    } else if (boundary_mass_flux_z_bottom != nullptr) {
	      const double dm = boundary_mass_flux_z_bottom[i];
	      mass += dm;
	      mom_z += boundary_momentum_flux_z_bottom[i];
	      total_energy += boundary_total_energy_flux_z_bottom[i];
	      const double ye_b = (dm > 0.0) ? clamp01_device(e_e_fraction)
	                                     : clamp01_device(ye_int_lag[c]);
	      ye_mass += dm * ye_b;
	    }
	  }

  const double mass_floor = rho_floor * V_ref;
  const double mass_raw = mass;
  if (!(mass > mass_floor) || !isfinite(mass)) {
    mass = mass_floor;
    if (mass_floor_delta != nullptr && isfinite(mass_raw)) {
      atomicAdd(mass_floor_delta, mass - mass_raw);
    }
  }

  mass_new[c] = mass;
  rho_new[c] = mass / V_ref;
  u_r_new[c] = (mass > 0.0 && isfinite(mom_r)) ? (mom_r / mass) : 0.0;
  u_z_new[c] = (mass > 0.0 && isfinite(mom_z)) ? (mom_z / mass) : 0.0;
  total_energy_new[c] = isfinite(total_energy) ? total_energy : 0.0;
  ye_int_new[c] = (mass > 0.0 && isfinite(ye_mass))
                      ? clamp01_device(ye_mass / mass)
                      : 0.5;
}

__device__ inline void apply_hydro_face_flux_second_order_r(
    double& mass,
    double& mom_r,
    double& mom_z,
    double& energy_e,
    double& energy_i,
    const double dV,
    const int K,
    const int Kp,
    const double sign,
    const double* __restrict__ rho,
    const double* __restrict__ u_r,
    const double* __restrict__ u_z,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ x_r_lag,
    const double* __restrict__ x_z_lag,
    const double* __restrict__ x_r_ref,
    const double* __restrict__ x_z_ref,
    const int i_face,
	    const int j,
	    const int nr,
	    const int nz,
	    const std::uint8_t* __restrict__ cell_nverts,
	    const int button_outer_node_ring = 0,
    const double* __restrict__ species_Y_lag = nullptr,
    double* species_ext = nullptr,
    const double* __restrict__ hot_e_eps_lag = nullptr,
    double* hot_e_eps_ext = nullptr,
    const double* __restrict__ burn_eps_lag = nullptr,
    double* burn_eps_ext = nullptr) {
  if (!finite_nonzero(dV)) {
    return;
  }
  const double rho_f = rz_reconstructed_face_value_r(K,
                                                     Kp,
                                                     dV,
                                                     rho,
                                                     x_r_lag,
                                                     x_z_lag,
                                                     x_r_ref,
                                                     x_z_ref,
                                                     i_face,
                                                     j,
                                                     nr,
                                                     nz,
                                                     0.0,
	                                                     cell_nverts,
	                                                     button_outer_node_ring);
  const double u_r_f = rz_reconstructed_face_value_r(K,
                                                     Kp,
                                                     dV,
                                                     u_r,
                                                     x_r_lag,
                                                     x_z_lag,
                                                     x_r_ref,
                                                     x_z_ref,
                                                     i_face,
                                                     j,
                                                     nr,
                                                     nz,
                                                     -1.0e300,
	                                                     cell_nverts,
	                                                     button_outer_node_ring);
  const double u_z_f = rz_reconstructed_face_value_r(K,
                                                     Kp,
                                                     dV,
                                                     u_z,
                                                     x_r_lag,
                                                     x_z_lag,
                                                     x_r_ref,
                                                     x_z_ref,
                                                     i_face,
                                                     j,
                                                     nr,
                                                     nz,
                                                     -1.0e300,
	                                                     cell_nverts,
	                                                     button_outer_node_ring);
  const double ee_f = rz_reconstructed_face_value_r(K,
                                                    Kp,
                                                    dV,
                                                    ee,
                                                    x_r_lag,
                                                    x_z_lag,
                                                    x_r_ref,
                                                    x_z_ref,
                                                    i_face,
                                                    j,
                                                    nr,
                                                    nz,
                                                    0.0,
	                                                    cell_nverts,
	                                                    button_outer_node_ring);
  const double ei_f = rz_reconstructed_face_value_r(K,
                                                    Kp,
                                                    dV,
                                                    ei,
                                                    x_r_lag,
                                                    x_z_lag,
                                                    x_r_ref,
                                                    x_z_ref,
                                                    i_face,
                                                    j,
                                                    nr,
                                                    nz,
                                                    0.0,
	                                                    cell_nverts,
	                                                    button_outer_node_ring);
  const double dm = dV * fmax(rho_f, 0.0);
  mass += sign * dm;
  if (species_Y_lag != nullptr && species_ext != nullptr) {
    const int donor = rz_donor_cell(K, Kp, dV);
    for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
      species_ext[s] +=
          sign * dm * species_Y_lag[donor * tenryu::burn::kNumSpecies + s];
    }
  }
  if (hot_e_eps_lag != nullptr && hot_e_eps_ext != nullptr) {
    const int donor = rz_donor_cell(K, Kp, dV);
    *hot_e_eps_ext += sign * dm * fmax(hot_e_eps_lag[donor], 0.0);
  }
  if (burn_eps_lag != nullptr && burn_eps_ext != nullptr) {
    const int donor = rz_donor_cell(K, Kp, dV);
    *burn_eps_ext += sign * dm * fmax(burn_eps_lag[donor], 0.0);
  }
  mom_r += sign * dm * u_r_f;
  mom_z += sign * dm * u_z_f;
  energy_e += sign * dm * fmax(ee_f, 0.0);
  energy_i += sign * dm * fmax(ei_f, 0.0);
}

__device__ inline void apply_hydro_face_flux_second_order_z(
    double& mass,
    double& mom_r,
    double& mom_z,
    double& energy_e,
    double& energy_i,
    const double dV,
    const int K,
    const int Kp,
    const double sign,
    const double* __restrict__ rho,
    const double* __restrict__ u_r,
    const double* __restrict__ u_z,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ x_r_lag,
    const double* __restrict__ x_z_lag,
    const double* __restrict__ x_r_ref,
    const double* __restrict__ x_z_ref,
    const int i,
	    const int j_face,
	    const int nr,
	    const int nz,
	    const std::uint8_t* __restrict__ cell_nverts,
	    const int button_outer_node_ring = 0,
    const double* __restrict__ species_Y_lag = nullptr,
    double* species_ext = nullptr,
    const double* __restrict__ hot_e_eps_lag = nullptr,
    double* hot_e_eps_ext = nullptr,
    const double* __restrict__ burn_eps_lag = nullptr,
    double* burn_eps_ext = nullptr) {
  if (!finite_nonzero(dV)) {
    return;
  }
  const double rho_f = rz_reconstructed_face_value_z(K,
                                                     Kp,
                                                     dV,
                                                     rho,
                                                     x_r_lag,
                                                     x_z_lag,
                                                     x_r_ref,
                                                     x_z_ref,
                                                     i,
                                                     j_face,
                                                     nr,
                                                     nz,
                                                     0.0,
	                                                     cell_nverts,
	                                                     button_outer_node_ring);
  const double u_r_f = rz_reconstructed_face_value_z(K,
                                                     Kp,
                                                     dV,
                                                     u_r,
                                                     x_r_lag,
                                                     x_z_lag,
                                                     x_r_ref,
                                                     x_z_ref,
                                                     i,
                                                     j_face,
                                                     nr,
                                                     nz,
                                                     -1.0e300,
	                                                     cell_nverts,
	                                                     button_outer_node_ring);
  const double u_z_f = rz_reconstructed_face_value_z(K,
                                                     Kp,
                                                     dV,
                                                     u_z,
                                                     x_r_lag,
                                                     x_z_lag,
                                                     x_r_ref,
                                                     x_z_ref,
                                                     i,
                                                     j_face,
                                                     nr,
                                                     nz,
                                                     -1.0e300,
	                                                     cell_nverts,
	                                                     button_outer_node_ring);
  const double ee_f = rz_reconstructed_face_value_z(K,
                                                    Kp,
                                                    dV,
                                                    ee,
                                                    x_r_lag,
                                                    x_z_lag,
                                                    x_r_ref,
                                                    x_z_ref,
                                                    i,
                                                    j_face,
                                                    nr,
                                                    nz,
                                                    0.0,
	                                                    cell_nverts,
	                                                    button_outer_node_ring);
  const double ei_f = rz_reconstructed_face_value_z(K,
                                                    Kp,
                                                    dV,
                                                    ei,
                                                    x_r_lag,
                                                    x_z_lag,
                                                    x_r_ref,
                                                    x_z_ref,
                                                    i,
                                                    j_face,
                                                    nr,
                                                    nz,
                                                    0.0,
	                                                    cell_nverts,
	                                                    button_outer_node_ring);
  const double dm = dV * fmax(rho_f, 0.0);
  mass += sign * dm;
  if (species_Y_lag != nullptr && species_ext != nullptr) {
    const int donor = rz_donor_cell(K, Kp, dV);
    for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
      species_ext[s] +=
          sign * dm * species_Y_lag[donor * tenryu::burn::kNumSpecies + s];
    }
  }
  if (hot_e_eps_lag != nullptr && hot_e_eps_ext != nullptr) {
    const int donor = rz_donor_cell(K, Kp, dV);
    *hot_e_eps_ext += sign * dm * fmax(hot_e_eps_lag[donor], 0.0);
  }
  if (burn_eps_lag != nullptr && burn_eps_ext != nullptr) {
    const int donor = rz_donor_cell(K, Kp, dV);
    *burn_eps_ext += sign * dm * fmax(burn_eps_lag[donor], 0.0);
  }
  mom_r += sign * dm * u_r_f;
  mom_z += sign * dm * u_z_f;
  energy_e += sign * dm * fmax(ee_f, 0.0);
  energy_i += sign * dm * fmax(ei_f, 0.0);
}

__device__ inline void apply_hydro_face_flux_total_energy_second_order_r(
    double& mass,
    double& mom_r,
    double& mom_z,
    double& total_energy,
    double& ye_mass,
    const double dV,
    const int K,
    const int Kp,
    const double sign,
    const double* __restrict__ rho,
    const double* __restrict__ u_r,
    const double* __restrict__ u_z,
    const double* __restrict__ e_tot,
    const double* __restrict__ ye_int,
    const double* __restrict__ x_r_lag,
    const double* __restrict__ x_z_lag,
    const double* __restrict__ x_r_ref,
    const double* __restrict__ x_z_ref,
    const int i_face,
    const int j,
    const int nr,
    const int nz,
    const int button_outer_node_ring = 0) {
  if (!finite_nonzero(dV)) {
    return;
  }
	  const int donor = (dV >= 0.0) ? K : Kp;
	  const double inv_dV = 1.0 / dV;
	  double moment_r = detail::ms2_swept_moment_r_face_t<true>(
	      x_r_lag, x_z_lag, x_r_ref, x_z_ref, i_face, j, nz);
	  double moment_z = detail::ms2_swept_moment_z_r_face_t<true>(
	      x_r_lag, x_z_lag, x_r_ref, x_z_ref, i_face, j, nz);
	  if (button_outer_node_ring > 0) {
	    moment_r = -moment_r;
	    moment_z = -moment_z;
	  }
	  const double r_face = moment_r * inv_dV;
	  const double z_face = moment_z * inv_dV;
  double psi = 1.0;
  psi = fmin(psi, rz_limiter_at_face(
                      donor, rho, x_r_lag, x_z_lag, r_face, z_face, nr, nz, nullptr));
  psi = fmin(psi, rz_limiter_at_face(
                      donor, u_r, x_r_lag, x_z_lag, r_face, z_face, nr, nz, nullptr));
  psi = fmin(psi, rz_limiter_at_face(
                      donor, u_z, x_r_lag, x_z_lag, r_face, z_face, nr, nz, nullptr));
  psi = fmin(psi, rz_limiter_at_face(
                      donor, e_tot, x_r_lag, x_z_lag, r_face, z_face, nr, nz, nullptr));
  psi = fmin(psi, rz_limiter_at_face(
                      donor, ye_int, x_r_lag, x_z_lag, r_face, z_face, nr, nz, nullptr));
  const double rho_f = rz_reconstructed_value_at_with_psi(
      donor, rho, x_r_lag, x_z_lag, r_face, z_face, nr, nz, 0.0, psi, nullptr);
  const double u_r_f = rz_reconstructed_value_at_with_psi(
      donor, u_r, x_r_lag, x_z_lag, r_face, z_face, nr, nz, -1.0e300, psi,
      nullptr);
  const double u_z_f = rz_reconstructed_value_at_with_psi(
      donor, u_z, x_r_lag, x_z_lag, r_face, z_face, nr, nz, -1.0e300, psi,
      nullptr);
  const double e_tot_f = rz_reconstructed_value_at_with_psi(
      donor, e_tot, x_r_lag, x_z_lag, r_face, z_face, nr, nz, 0.0, psi,
      nullptr);
  const double ye_f = clamp01_device(rz_reconstructed_value_at_with_psi(
      donor, ye_int, x_r_lag, x_z_lag, r_face, z_face, nr, nz, 0.0, psi,
      nullptr));
  const double dm = dV * fmax(rho_f, 0.0);
  mass += sign * dm;
  mom_r += sign * dm * u_r_f;
  mom_z += sign * dm * u_z_f;
  total_energy += sign * dm * fmax(e_tot_f, 0.0);
  ye_mass += sign * dm * ye_f;
}

__device__ inline void apply_hydro_face_flux_total_energy_second_order_z(
    double& mass,
    double& mom_r,
    double& mom_z,
    double& total_energy,
    double& ye_mass,
    const double dV,
    const int K,
    const int Kp,
    const double sign,
    const double* __restrict__ rho,
    const double* __restrict__ u_r,
    const double* __restrict__ u_z,
    const double* __restrict__ e_tot,
    const double* __restrict__ ye_int,
    const double* __restrict__ x_r_lag,
    const double* __restrict__ x_z_lag,
    const double* __restrict__ x_r_ref,
    const double* __restrict__ x_z_ref,
	    const int i,
	    const int j_face,
	    const int nr,
	    const int nz,
	    const int button_outer_node_ring = 0) {
  if (!finite_nonzero(dV)) {
    return;
  }
	  const int donor = (dV >= 0.0) ? K : Kp;
	  const double inv_dV = 1.0 / dV;
	  double moment_r = detail::ms2_swept_moment_r_z_face_t<true>(
	      x_r_lag, x_z_lag, x_r_ref, x_z_ref, i, j_face, nz);
	  double moment_z = detail::ms2_swept_moment_z_face_t<true>(
	      x_r_lag, x_z_lag, x_r_ref, x_z_ref, i, j_face, nz);
	  if (button_outer_node_ring > 0) {
	    moment_r = -moment_r;
	    moment_z = -moment_z;
	  }
	  const double r_face = moment_r * inv_dV;
	  const double z_face = moment_z * inv_dV;
  double psi = 1.0;
  psi = fmin(psi, rz_limiter_at_face(
                      donor, rho, x_r_lag, x_z_lag, r_face, z_face, nr, nz, nullptr));
  psi = fmin(psi, rz_limiter_at_face(
                      donor, u_r, x_r_lag, x_z_lag, r_face, z_face, nr, nz, nullptr));
  psi = fmin(psi, rz_limiter_at_face(
                      donor, u_z, x_r_lag, x_z_lag, r_face, z_face, nr, nz, nullptr));
  psi = fmin(psi, rz_limiter_at_face(
                      donor, e_tot, x_r_lag, x_z_lag, r_face, z_face, nr, nz, nullptr));
  psi = fmin(psi, rz_limiter_at_face(
                      donor, ye_int, x_r_lag, x_z_lag, r_face, z_face, nr, nz, nullptr));
  const double rho_f = rz_reconstructed_value_at_with_psi(
      donor, rho, x_r_lag, x_z_lag, r_face, z_face, nr, nz, 0.0, psi, nullptr);
  const double u_r_f = rz_reconstructed_value_at_with_psi(
      donor, u_r, x_r_lag, x_z_lag, r_face, z_face, nr, nz, -1.0e300, psi,
      nullptr);
  const double u_z_f = rz_reconstructed_value_at_with_psi(
      donor, u_z, x_r_lag, x_z_lag, r_face, z_face, nr, nz, -1.0e300, psi,
      nullptr);
  const double e_tot_f = rz_reconstructed_value_at_with_psi(
      donor, e_tot, x_r_lag, x_z_lag, r_face, z_face, nr, nz, 0.0, psi,
      nullptr);
  const double ye_f = clamp01_device(rz_reconstructed_value_at_with_psi(
      donor, ye_int, x_r_lag, x_z_lag, r_face, z_face, nr, nz, 0.0, psi,
      nullptr));
  const double dm = dV * fmax(rho_f, 0.0);
  mass += sign * dm;
  mom_r += sign * dm * u_r_f;
  mom_z += sign * dm * u_z_f;
  total_energy += sign * dm * fmax(e_tot_f, 0.0);
  ye_mass += sign * dm * ye_f;
}

__global__ void ale_remap_2d_rz_second_order_kernel(
    const double* __restrict__ rho_lag,
    const double* __restrict__ u_r_lag,
    const double* __restrict__ u_z_lag,
    const double* __restrict__ e_e_lag,
    const double* __restrict__ e_i_lag,
    const double* __restrict__ vol_lag,
    const double* __restrict__ mass_lag,
    const double* __restrict__ x_r_lag,
    const double* __restrict__ x_z_lag,
    const double* __restrict__ x_r_ref,
    const double* __restrict__ x_z_ref,
    const double* __restrict__ vol_ref,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ boundary_mass_flux_z_bottom,
    const double* __restrict__ boundary_mass_flux_z_top,
    const double* __restrict__ boundary_momentum_flux_z_bottom,
    const double* __restrict__ boundary_momentum_flux_z_top,
    const double* __restrict__ boundary_energy_flux_z_bottom,
    const double* __restrict__ boundary_energy_flux_z_top,
    double* __restrict__ rho_new,
    double* __restrict__ u_r_new,
    double* __restrict__ u_z_new,
    double* __restrict__ e_e_new,
    double* __restrict__ e_i_new,
    double* __restrict__ mass_new,
    const double* __restrict__ zbar,
    double* __restrict__ mass_floor_delta,
	    double* __restrict__ energy_floor_delta,
	    const int nr,
	    const int nz,
	    const int button_outer_node_ring,
	    const double rho_floor,
	    const double te_floor,
	    const double ti_floor,
	    const double gamma,
    const double A,
    const double* __restrict__ burn_species_Y_lag,
    double* __restrict__ burn_species_Y_new,
    const double* __restrict__ hot_e_eps_lag,
    double* __restrict__ hot_e_eps_new,
    const double* __restrict__ burn_eps_lag,
    double* __restrict__ burn_eps_new) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const bool remap_species =
      (burn_species_Y_lag != nullptr && burn_species_Y_new != nullptr);
  const bool remap_hot_e_eps =
      (hot_e_eps_lag != nullptr && hot_e_eps_new != nullptr);
  const bool remap_burn_eps =
      (burn_eps_lag != nullptr && burn_eps_new != nullptr);

	  const int i = c / nz;
		  const int j = c - i * nz;
		  if (rz_is_dormant_button_cell(c, i, button_outer_node_ring)) {
		    rho_new[c] = 0.0;
		    mass_new[c] = 0.0;
		    u_r_new[c] = 0.0;
		    u_z_new[c] = 0.0;
		    e_e_new[c] = 0.0;
		    e_i_new[c] = 0.0;
        if (remap_species) {
          for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
            burn_species_Y_new[c * tenryu::burn::kNumSpecies + s] = 0.0;
          }
        }
        if (remap_hot_e_eps) {
          hot_e_eps_new[c] = 0.0;
        }
        if (remap_burn_eps) {
          burn_eps_new[c] = 0.0;
        }
	    return;
	  }
	  const double V_lag = fmax(vol_lag[c], kTinyVolume);
	  const double V_ref = fmax(vol_ref[c], kTinyVolume);
  const double rho_c = fmax(rho_lag[c], 0.0);
  double mass = rho_c * V_lag;
  double species_ext[tenryu::burn::kNumSpecies];
  if (remap_species) {
    for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
      species_ext[s] =
          mass * burn_species_Y_lag[c * tenryu::burn::kNumSpecies + s];
    }
  }
  double hot_e_eps_ext = 0.0;
  if (remap_hot_e_eps) {
    hot_e_eps_ext = mass * fmax(hot_e_eps_lag[c], 0.0);
  }
  double burn_eps_ext = 0.0;
  if (remap_burn_eps) {
    burn_eps_ext = mass * fmax(burn_eps_lag[c], 0.0);
  }
  double mom_r = mass * u_r_lag[c];
  double mom_z = mass * u_z_lag[c];
  double energy_e = mass * fmax(e_e_lag[c], 0.0);
  double energy_i = mass * fmax(e_i_lag[c], 0.0);
  const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : 1.0;
  const double A_safe = fmax(A, 1.0e-30);
  const double gm1 = fmax(gamma - 1.0, 1.0e-30);
  const double cv_i =
      tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  const double cv_e =
      z * tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  const double cv_sum = cv_e + cv_i;
	  const double e_e_fraction = (cv_sum > 0.0) ? (cv_e / cv_sum) : 0.5;
	  const double e_i_fraction = 1.0 - e_e_fraction;

		  if (rz_is_button_cell(c, button_outer_node_ring)) {
		    double seam_swept_sum = 0.0;
		    for (int js = 0; js < nz; ++js) {
		      const double dV = rz_button_seam_swept_volume_ref(
		          x_r_lag, x_z_lag, x_r_ref, x_z_ref, button_outer_node_ring, nz, js);
		      seam_swept_sum += dV;
		      const int shell = detail::cell_index(button_outer_node_ring, js, nz);
		      const int donor = rz_donor_cell(c, shell, dV);
		      apply_hydro_face_flux(mass, mom_r, mom_z, energy_e, energy_i, dV, donor,
		                            -1.0, rho_lag, u_r_lag, u_z_lag, e_e_lag, e_i_lag,
                                remap_species ? burn_species_Y_lag : nullptr,
                                remap_species ? species_ext : nullptr,
                                remap_hot_e_eps ? hot_e_eps_lag : nullptr,
                                remap_hot_e_eps ? &hot_e_eps_ext : nullptr,
                                remap_burn_eps ? burn_eps_lag : nullptr,
                                remap_burn_eps ? &burn_eps_ext : nullptr);
		    }
		    const double V_button_gcl = fmax(V_lag - seam_swept_sum, kTinyVolume);
		    const double mass_floor = rho_floor * V_button_gcl;
		    const double mass_raw = mass;
		    if (!(mass > mass_floor) || !isfinite(mass)) {
		      mass = mass_floor;
		      if (mass_floor_delta != nullptr && isfinite(mass_raw)) {
		        atomicAdd(mass_floor_delta, mass - mass_raw);
		      }
		    }
		
		    const double e_e_floor = cv_e * fmax(te_floor, 0.0);
		    const double e_i_floor = cv_i * fmax(ti_floor, 0.0);
		
		    double ee = (mass > 0.0) ? (energy_e / mass) : e_e_floor;
		    double ei = (mass > 0.0) ? (energy_i / mass) : e_i_floor;
		    if (!(ee >= e_e_floor) || !isfinite(ee)) {
		      if (energy_floor_delta != nullptr && isfinite(ee)) {
		        atomicAdd(energy_floor_delta, (e_e_floor - ee) * mass);
		      }
		      ee = e_e_floor;
		    }
		    if (!(ei >= e_i_floor) || !isfinite(ei)) {
		      if (energy_floor_delta != nullptr && isfinite(ei)) {
		        atomicAdd(energy_floor_delta, (e_i_floor - ei) * mass);
		      }
		      ei = e_i_floor;
		    }
		
		    mass_new[c] = mass;
		    rho_new[c] = mass / V_button_gcl;
		    u_r_new[c] = (mass > 0.0 && isfinite(mom_r)) ? (mom_r / mass) : 0.0;
		    u_z_new[c] = (mass > 0.0 && isfinite(mom_z)) ? (mom_z / mass) : 0.0;
		    e_e_new[c] = ee;
		    e_i_new[c] = ei;
        if (remap_species) {
          for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
            burn_species_Y_new[c * tenryu::burn::kNumSpecies + s] =
                (mass > 0.0 && isfinite(mass)) ? species_ext[s] / mass : 0.0;
          }
        }
        if (remap_hot_e_eps) {
          hot_e_eps_new[c] =
              (mass > 0.0 && isfinite(mass)) ? fmax(hot_e_eps_ext / mass, 0.0)
                                             : 0.0;
        }
        if (remap_burn_eps) {
          burn_eps_new[c] =
              (mass > 0.0 && isfinite(mass)) ? fmax(burn_eps_ext / mass, 0.0)
                                             : 0.0;
        }
		    return;
		  } else {
	  if (i + 1 < nr) {
	    const int K = c;
	    const int Kp = detail::cell_index(i + 1, j, nz);
	    if (!rz_button_face_touches_dormant(K, Kp, nz, button_outer_node_ring)) {
		    const double dV = rz_swept_volume_r_face_ref_topology(
		        x_r_lag, x_z_lag, x_r_ref, x_z_ref, cell_nverts, i + 1, j, nr,
		        nz, button_outer_node_ring);
	    const bool button_first_shell_face =
	        rz_button_enabled(button_outer_node_ring) &&
	        i == button_outer_node_ring;
	    if (button_first_shell_face ||
	        rz_r_face_has_tri(cell_nverts, i + 1, j, nr, nz)) {
	      const int donor = rz_donor_cell(K, Kp, dV);
	      apply_hydro_face_flux(mass, mom_r, mom_z, energy_e, energy_i, dV, donor,
	                            -1.0, rho_lag, u_r_lag, u_z_lag, e_e_lag, e_i_lag,
                                remap_species ? burn_species_Y_lag : nullptr,
                                remap_species ? species_ext : nullptr,
                                remap_hot_e_eps ? hot_e_eps_lag : nullptr,
                                remap_hot_e_eps ? &hot_e_eps_ext : nullptr,
                                remap_burn_eps ? burn_eps_lag : nullptr,
                                remap_burn_eps ? &burn_eps_ext : nullptr);
    } else {
      apply_hydro_face_flux_second_order_r(mass,
                                           mom_r,
                                           mom_z,
                                           energy_e,
                                           energy_i,
                                           dV,
                                           K,
                                           Kp,
                                           -1.0,
                                           rho_lag,
                                           u_r_lag,
                                           u_z_lag,
                                           e_e_lag,
                                           e_i_lag,
                                           x_r_lag,
                                           x_z_lag,
                                           x_r_ref,
                                           x_z_ref,
                                           i + 1,
                                           j,
                                           nr,
	                                           nz,
	                                           cell_nverts,
	                                           button_outer_node_ring,
                                           remap_species ? burn_species_Y_lag : nullptr,
                                           remap_species ? species_ext : nullptr,
                                           remap_hot_e_eps ? hot_e_eps_lag : nullptr,
                                           remap_hot_e_eps ? &hot_e_eps_ext : nullptr,
                                           remap_burn_eps ? burn_eps_lag : nullptr,
                                           remap_burn_eps ? &burn_eps_ext : nullptr);
    }
	    }
	  }
	  if (i > 0) {
	    const bool button_seam_face =
	        rz_button_enabled(button_outer_node_ring) &&
	        i == button_outer_node_ring;
	    const int K = button_seam_face ? 0 : detail::cell_index(i - 1, j, nz);
	    const int Kp = c;
	    if (!rz_button_face_touches_dormant(K, Kp, nz, button_outer_node_ring)) {
		    const double dV =
		        button_seam_face
		            ? rz_button_seam_swept_volume_ref(
		                  x_r_lag, x_z_lag, x_r_ref, x_z_ref,
		                  button_outer_node_ring, nz, j)
		            : rz_swept_volume_r_face_ref_topology(
		                  x_r_lag, x_z_lag, x_r_ref, x_z_ref, cell_nverts, i, j,
		                  nr, nz, button_outer_node_ring);
	    const bool button_first_shell_face =
	        rz_button_enabled(button_outer_node_ring) &&
	        i == button_outer_node_ring + 1;
	    if (button_seam_face || button_first_shell_face ||
	        rz_r_face_has_tri(cell_nverts, i, j, nr, nz)) {
	      const int donor = rz_donor_cell(K, Kp, dV);
	      apply_hydro_face_flux(mass, mom_r, mom_z, energy_e, energy_i, dV, donor,
	                            1.0, rho_lag, u_r_lag, u_z_lag, e_e_lag, e_i_lag,
                                remap_species ? burn_species_Y_lag : nullptr,
                                remap_species ? species_ext : nullptr,
                                remap_hot_e_eps ? hot_e_eps_lag : nullptr,
                                remap_hot_e_eps ? &hot_e_eps_ext : nullptr,
                                remap_burn_eps ? burn_eps_lag : nullptr,
                                remap_burn_eps ? &burn_eps_ext : nullptr);
    } else {
      apply_hydro_face_flux_second_order_r(mass,
                                           mom_r,
                                           mom_z,
                                           energy_e,
                                           energy_i,
                                           dV,
                                           K,
                                           Kp,
                                           1.0,
                                           rho_lag,
                                           u_r_lag,
                                           u_z_lag,
                                           e_e_lag,
                                           e_i_lag,
                                           x_r_lag,
                                           x_z_lag,
                                           x_r_ref,
                                           x_z_ref,
                                           i,
                                           j,
                                           nr,
	                                           nz,
	                                           cell_nverts,
	                                           button_outer_node_ring,
                                           remap_species ? burn_species_Y_lag : nullptr,
                                           remap_species ? species_ext : nullptr,
                                           remap_hot_e_eps ? hot_e_eps_lag : nullptr,
                                           remap_hot_e_eps ? &hot_e_eps_ext : nullptr,
                                           remap_burn_eps ? burn_eps_lag : nullptr,
                                           remap_burn_eps ? &burn_eps_ext : nullptr);
    }
	    }
  }
	  if (j + 1 < nz) {
	    const int K = c;
	    const int Kp = detail::cell_index(i, j + 1, nz);
	    if (!rz_button_face_touches_dormant(K, Kp, nz, button_outer_node_ring)) {
		    const double dV = rz_swept_volume_z_face_ref_topology(
		        x_r_lag, x_z_lag, x_r_ref, x_z_ref, cell_nverts, i, j + 1, nr,
		        nz, button_outer_node_ring);
	    const bool button_first_shell_face =
	        rz_button_enabled(button_outer_node_ring) &&
	        i == button_outer_node_ring;
	    if (button_first_shell_face ||
	        rz_z_face_has_tri(cell_nverts, i, j + 1, nr, nz)) {
	      const int donor = rz_donor_cell(K, Kp, dV);
	      apply_hydro_face_flux(mass, mom_r, mom_z, energy_e, energy_i, dV, donor,
	                            -1.0, rho_lag, u_r_lag, u_z_lag, e_e_lag, e_i_lag,
                                remap_species ? burn_species_Y_lag : nullptr,
                                remap_species ? species_ext : nullptr,
                                remap_hot_e_eps ? hot_e_eps_lag : nullptr,
                                remap_hot_e_eps ? &hot_e_eps_ext : nullptr,
                                remap_burn_eps ? burn_eps_lag : nullptr,
                                remap_burn_eps ? &burn_eps_ext : nullptr);
    } else {
      apply_hydro_face_flux_second_order_z(mass,
                                           mom_r,
                                           mom_z,
                                           energy_e,
                                           energy_i,
                                           dV,
                                           K,
                                           Kp,
                                           -1.0,
                                           rho_lag,
                                           u_r_lag,
                                           u_z_lag,
                                           e_e_lag,
                                           e_i_lag,
                                           x_r_lag,
                                           x_z_lag,
                                           x_r_ref,
                                           x_z_ref,
                                           i,
                                           j + 1,
                                           nr,
	                                           nz,
	                                           cell_nverts,
	                                           button_outer_node_ring,
                                           remap_species ? burn_species_Y_lag : nullptr,
                                           remap_species ? species_ext : nullptr,
                                           remap_hot_e_eps ? hot_e_eps_lag : nullptr,
                                           remap_hot_e_eps ? &hot_e_eps_ext : nullptr,
                                           remap_burn_eps ? burn_eps_lag : nullptr,
                                           remap_burn_eps ? &burn_eps_ext : nullptr);
    }
	    }
  } else if (boundary_mass_flux_z_top != nullptr) {
    const double dm_bz = boundary_mass_flux_z_top[i];
    mass += dm_bz;
    if (remap_species && dm_bz < 0.0) {
      for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
        species_ext[s] +=
            dm_bz * burn_species_Y_lag[c * tenryu::burn::kNumSpecies + s];
      }
    }
    if (remap_hot_e_eps && dm_bz < 0.0) {
      hot_e_eps_ext += dm_bz * fmax(hot_e_eps_lag[c], 0.0);
    }
    if (remap_burn_eps && dm_bz < 0.0) {
      burn_eps_ext += dm_bz * fmax(burn_eps_lag[c], 0.0);
    }
    mom_z += boundary_momentum_flux_z_top[i];
    energy_e += e_e_fraction * boundary_energy_flux_z_top[i];
    energy_i += e_i_fraction * boundary_energy_flux_z_top[i];
  }
	  if (j > 0) {
	    const int K = detail::cell_index(i, j - 1, nz);
	    const int Kp = c;
	    if (!rz_button_face_touches_dormant(K, Kp, nz, button_outer_node_ring)) {
		    const double dV = rz_swept_volume_z_face_ref_topology(
		        x_r_lag, x_z_lag, x_r_ref, x_z_ref, cell_nverts, i, j, nr, nz,
		        button_outer_node_ring);
	    const bool button_first_shell_face =
	        rz_button_enabled(button_outer_node_ring) &&
	        i == button_outer_node_ring;
	    if (button_first_shell_face ||
	        rz_z_face_has_tri(cell_nverts, i, j, nr, nz)) {
	      const int donor = rz_donor_cell(K, Kp, dV);
	      apply_hydro_face_flux(mass, mom_r, mom_z, energy_e, energy_i, dV, donor,
	                            1.0, rho_lag, u_r_lag, u_z_lag, e_e_lag, e_i_lag,
                                remap_species ? burn_species_Y_lag : nullptr,
                                remap_species ? species_ext : nullptr,
                                remap_hot_e_eps ? hot_e_eps_lag : nullptr,
                                remap_hot_e_eps ? &hot_e_eps_ext : nullptr,
                                remap_burn_eps ? burn_eps_lag : nullptr,
                                remap_burn_eps ? &burn_eps_ext : nullptr);
    } else {
      apply_hydro_face_flux_second_order_z(mass,
                                           mom_r,
                                           mom_z,
                                           energy_e,
                                           energy_i,
                                           dV,
                                           K,
                                           Kp,
                                           1.0,
                                           rho_lag,
                                           u_r_lag,
                                           u_z_lag,
                                           e_e_lag,
                                           e_i_lag,
                                           x_r_lag,
                                           x_z_lag,
                                           x_r_ref,
                                           x_z_ref,
                                           i,
                                           j,
                                           nr,
	                                           nz,
	                                           cell_nverts,
	                                           button_outer_node_ring,
                                           remap_species ? burn_species_Y_lag : nullptr,
                                           remap_species ? species_ext : nullptr,
                                           remap_hot_e_eps ? hot_e_eps_lag : nullptr,
                                           remap_hot_e_eps ? &hot_e_eps_ext : nullptr,
                                           remap_burn_eps ? burn_eps_lag : nullptr,
                                           remap_burn_eps ? &burn_eps_ext : nullptr);
    }
	    }
	  } else if (boundary_mass_flux_z_bottom != nullptr) {
	    const double dm_bz = boundary_mass_flux_z_bottom[i];
	    mass += dm_bz;
      if (remap_species && dm_bz < 0.0) {
        for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
          species_ext[s] +=
              dm_bz * burn_species_Y_lag[c * tenryu::burn::kNumSpecies + s];
        }
      }
      if (remap_hot_e_eps && dm_bz < 0.0) {
        hot_e_eps_ext += dm_bz * fmax(hot_e_eps_lag[c], 0.0);
      }
      if (remap_burn_eps && dm_bz < 0.0) {
        burn_eps_ext += dm_bz * fmax(burn_eps_lag[c], 0.0);
      }
	    mom_z += boundary_momentum_flux_z_bottom[i];
	    energy_e += e_e_fraction * boundary_energy_flux_z_bottom[i];
	    energy_i += e_i_fraction * boundary_energy_flux_z_bottom[i];
	  }
	  }

	  const double mass_floor = rho_floor * V_ref;
  const double mass_raw = mass;
  if (!(mass > mass_floor) || !isfinite(mass)) {
    mass = mass_floor;
    if (mass_floor_delta != nullptr && isfinite(mass_raw)) {
      atomicAdd(mass_floor_delta, mass - mass_raw);
    }
  }

  const double e_e_floor = cv_e * fmax(te_floor, 0.0);
  const double e_i_floor = cv_i * fmax(ti_floor, 0.0);

  double ee = (mass > 0.0) ? (energy_e / mass) : e_e_floor;
  double ei = (mass > 0.0) ? (energy_i / mass) : e_i_floor;
  if (!(ee >= e_e_floor) || !isfinite(ee)) {
    if (energy_floor_delta != nullptr && isfinite(ee)) {
      atomicAdd(energy_floor_delta, (e_e_floor - ee) * mass);
    }
    ee = e_e_floor;
  }
  if (!(ei >= e_i_floor) || !isfinite(ei)) {
    if (energy_floor_delta != nullptr && isfinite(ei)) {
      atomicAdd(energy_floor_delta, (e_i_floor - ei) * mass);
    }
    ei = e_i_floor;
  }

  mass_new[c] = mass;
  rho_new[c] = mass / V_ref;
  u_r_new[c] = (mass > 0.0 && isfinite(mom_r)) ? (mom_r / mass) : 0.0;
  u_z_new[c] = (mass > 0.0 && isfinite(mom_z)) ? (mom_z / mass) : 0.0;
  e_e_new[c] = ee;
  e_i_new[c] = ei;
  if (remap_species) {
    for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
      burn_species_Y_new[c * tenryu::burn::kNumSpecies + s] =
          (mass > 0.0 && isfinite(mass)) ? species_ext[s] / mass : 0.0;
    }
  }
  if (remap_hot_e_eps) {
    hot_e_eps_new[c] =
        (mass > 0.0 && isfinite(mass)) ? fmax(hot_e_eps_ext / mass, 0.0)
                                       : 0.0;
  }
  if (remap_burn_eps) {
    burn_eps_new[c] =
        (mass > 0.0 && isfinite(mass)) ? fmax(burn_eps_ext / mass, 0.0)
                                       : 0.0;
  }
}

__global__ void ale_remap_2d_rz_total_energy_second_order_kernel(
    const double* __restrict__ rho_lag,
    const double* __restrict__ u_r_lag,
    const double* __restrict__ u_z_lag,
    const double* __restrict__ e_tot_lag,
    const double* __restrict__ ye_int_lag,
    const double* __restrict__ vol_lag,
    const double* __restrict__ mass_lag,
    const double* __restrict__ x_r_lag,
    const double* __restrict__ x_z_lag,
    const double* __restrict__ x_r_ref,
    const double* __restrict__ x_z_ref,
    const double* __restrict__ vol_ref,
    const double* __restrict__ boundary_mass_flux_z_bottom,
    const double* __restrict__ boundary_mass_flux_z_top,
    const double* __restrict__ boundary_momentum_flux_z_bottom,
    const double* __restrict__ boundary_momentum_flux_z_top,
    const double* __restrict__ boundary_total_energy_flux_z_bottom,
    const double* __restrict__ boundary_total_energy_flux_z_top,
    double* __restrict__ rho_new,
    double* __restrict__ u_r_new,
    double* __restrict__ u_z_new,
    double* __restrict__ total_energy_new,
    double* __restrict__ ye_int_new,
    double* __restrict__ mass_new,
	    double* __restrict__ mass_floor_delta,
	    const int nr,
	    const int nz,
	    const int button_outer_node_ring,
	    const double rho_floor,
	    const double e_e_fraction) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

	  const int i = c / nz;
		  const int j = c - i * nz;
		  if (rz_is_dormant_button_cell(c, i, button_outer_node_ring)) {
		    rho_new[c] = 0.0;
		    mass_new[c] = 0.0;
		    u_r_new[c] = 0.0;
		    u_z_new[c] = 0.0;
		    total_energy_new[c] = 0.0;
		    ye_int_new[c] = 0.0;
		    return;
		  }
	  const double V_lag = fmax(vol_lag[c], kTinyVolume);
	  const double V_ref = fmax(vol_ref[c], kTinyVolume);
  const double rho_c = fmax(rho_lag[c], 0.0);
  double mass = rho_c * V_lag;
  double mom_r = mass * u_r_lag[c];
  double mom_z = mass * u_z_lag[c];
	  double total_energy = mass * fmax(e_tot_lag[c], 0.0);
	  double ye_mass = mass * clamp01_device(ye_int_lag[c]);

		  if (rz_is_button_cell(c, button_outer_node_ring)) {
		    double seam_swept_sum = 0.0;
		    for (int js = 0; js < nz; ++js) {
		      const double dV = rz_button_seam_swept_volume_ref(
		          x_r_lag, x_z_lag, x_r_ref, x_z_ref, button_outer_node_ring, nz, js);
		      seam_swept_sum += dV;
		      const int shell = detail::cell_index(button_outer_node_ring, js, nz);
		      const int donor = rz_donor_cell(c, shell, dV);
		      apply_hydro_face_flux_total_energy(
		          mass, mom_r, mom_z, total_energy, ye_mass, dV, donor, -1.0,
		          rho_lag, u_r_lag, u_z_lag, e_tot_lag, ye_int_lag);
		    }
		    const double V_button_gcl = fmax(V_lag - seam_swept_sum, kTinyVolume);
		    const double mass_floor = rho_floor * V_button_gcl;
		    const double mass_raw = mass;
		    if (!(mass > mass_floor) || !isfinite(mass)) {
		      mass = mass_floor;
		      if (mass_floor_delta != nullptr && isfinite(mass_raw)) {
		        atomicAdd(mass_floor_delta, mass - mass_raw);
		      }
		    }
		
		    mass_new[c] = mass;
		    rho_new[c] = mass / V_button_gcl;
		    u_r_new[c] = (mass > 0.0 && isfinite(mom_r)) ? (mom_r / mass) : 0.0;
		    u_z_new[c] = (mass > 0.0 && isfinite(mom_z)) ? (mom_z / mass) : 0.0;
		    total_energy_new[c] = isfinite(total_energy) ? total_energy : 0.0;
		    ye_int_new[c] = (mass > 0.0 && isfinite(ye_mass))
		                        ? clamp01_device(ye_mass / mass)
		                        : 0.5;
		    return;
		  } else {
	  if (i + 1 < nr) {
		    const int Kp = detail::cell_index(i + 1, j, nz);
		    if (!rz_button_face_touches_dormant(c, Kp, nz, button_outer_node_ring)) {
		    const double dV = rz_swept_volume_r_face_ref_topology(
		        x_r_lag, x_z_lag, x_r_ref, x_z_ref, nullptr, i + 1, j, nr, nz,
		        button_outer_node_ring);
		    if (rz_button_enabled(button_outer_node_ring) &&
		        i == button_outer_node_ring) {
		      const int donor = rz_donor_cell(c, Kp, dV);
		      apply_hydro_face_flux_total_energy(
		          mass, mom_r, mom_z, total_energy, ye_mass, dV, donor, -1.0,
		          rho_lag, u_r_lag, u_z_lag, e_tot_lag, ye_int_lag);
		    } else {
		      apply_hydro_face_flux_total_energy_second_order_r(
		          mass, mom_r, mom_z, total_energy, ye_mass, dV, c, Kp, -1.0,
		          rho_lag, u_r_lag, u_z_lag, e_tot_lag, ye_int_lag, x_r_lag,
		          x_z_lag, x_r_ref, x_z_ref, i + 1, j, nr, nz,
		          button_outer_node_ring);
		    }
		    }
	  }
	  if (i > 0) {
	    const bool button_seam_face =
	        rz_button_enabled(button_outer_node_ring) &&
	        i == button_outer_node_ring;
	    const int K = button_seam_face ? 0 : detail::cell_index(i - 1, j, nz);
	    if (!rz_button_face_touches_dormant(K, c, nz, button_outer_node_ring)) {
	    const double dV =
	        button_seam_face
	            ? rz_button_seam_swept_volume_ref(
	                  x_r_lag, x_z_lag, x_r_ref, x_z_ref,
	                  button_outer_node_ring, nz, j)
	            : rz_swept_volume_r_face_ref_topology(
	                  x_r_lag, x_z_lag, x_r_ref, x_z_ref, nullptr, i, j, nr,
	                  nz, button_outer_node_ring);
		    const bool button_first_shell_face =
		        rz_button_enabled(button_outer_node_ring) &&
		        i == button_outer_node_ring + 1;
		    if (button_seam_face || button_first_shell_face) {
		      const int donor = rz_donor_cell(K, c, dV);
		      apply_hydro_face_flux_total_energy(
		          mass, mom_r, mom_z, total_energy, ye_mass, dV, donor, 1.0,
		          rho_lag, u_r_lag, u_z_lag, e_tot_lag, ye_int_lag);
		    } else {
		      apply_hydro_face_flux_total_energy_second_order_r(
		          mass, mom_r, mom_z, total_energy, ye_mass, dV, K, c, 1.0,
		          rho_lag, u_r_lag, u_z_lag, e_tot_lag, ye_int_lag, x_r_lag,
		          x_z_lag, x_r_ref, x_z_ref, i, j, nr, nz,
		          button_outer_node_ring);
		    }
	    }
	  }
	  if (j + 1 < nz) {
		    const int Kp = detail::cell_index(i, j + 1, nz);
		    if (!rz_button_face_touches_dormant(c, Kp, nz, button_outer_node_ring)) {
		    const double dV = rz_swept_volume_z_face_ref_topology(
		        x_r_lag, x_z_lag, x_r_ref, x_z_ref, nullptr, i, j + 1, nr, nz,
		        button_outer_node_ring);
		    if (rz_button_enabled(button_outer_node_ring) &&
		        i == button_outer_node_ring) {
		      const int donor = rz_donor_cell(c, Kp, dV);
		      apply_hydro_face_flux_total_energy(
		          mass, mom_r, mom_z, total_energy, ye_mass, dV, donor, -1.0,
		          rho_lag, u_r_lag, u_z_lag, e_tot_lag, ye_int_lag);
		    } else {
		      apply_hydro_face_flux_total_energy_second_order_z(
		          mass, mom_r, mom_z, total_energy, ye_mass, dV, c, Kp, -1.0,
		          rho_lag, u_r_lag, u_z_lag, e_tot_lag, ye_int_lag, x_r_lag,
		          x_z_lag, x_r_ref, x_z_ref, i, j + 1, nr, nz,
		          button_outer_node_ring);
		    }
		    }
	  } else if (boundary_mass_flux_z_top != nullptr) {
	    const double dm = boundary_mass_flux_z_top[i];
	    mass += dm;
    mom_z += boundary_momentum_flux_z_top[i];
    total_energy += boundary_total_energy_flux_z_top[i];
    const double ye_b = (dm > 0.0) ? clamp01_device(e_e_fraction)
                                   : clamp01_device(ye_int_lag[c]);
    ye_mass += dm * ye_b;
	  }
	  if (j > 0) {
	    const int K = detail::cell_index(i, j - 1, nz);
		    if (!rz_button_face_touches_dormant(K, c, nz, button_outer_node_ring)) {
		    const double dV = rz_swept_volume_z_face_ref_topology(
		        x_r_lag, x_z_lag, x_r_ref, x_z_ref, nullptr, i, j, nr, nz,
		        button_outer_node_ring);
		    if (rz_button_enabled(button_outer_node_ring) &&
		        i == button_outer_node_ring) {
		      const int donor = rz_donor_cell(K, c, dV);
		      apply_hydro_face_flux_total_energy(
		          mass, mom_r, mom_z, total_energy, ye_mass, dV, donor, 1.0,
		          rho_lag, u_r_lag, u_z_lag, e_tot_lag, ye_int_lag);
		    } else {
		      apply_hydro_face_flux_total_energy_second_order_z(
		          mass, mom_r, mom_z, total_energy, ye_mass, dV, K, c, 1.0,
		          rho_lag, u_r_lag, u_z_lag, e_tot_lag, ye_int_lag, x_r_lag,
		          x_z_lag, x_r_ref, x_z_ref, i, j, nr, nz,
		          button_outer_node_ring);
		    }
		    }
	  } else if (boundary_mass_flux_z_bottom != nullptr) {
	    const double dm = boundary_mass_flux_z_bottom[i];
	    mass += dm;
    mom_z += boundary_momentum_flux_z_bottom[i];
    total_energy += boundary_total_energy_flux_z_bottom[i];
	    const double ye_b = (dm > 0.0) ? clamp01_device(e_e_fraction)
	                                   : clamp01_device(ye_int_lag[c]);
	    ye_mass += dm * ye_b;
	  }
	  }

	  const double mass_floor = rho_floor * V_ref;
  const double mass_raw = mass;
  if (!(mass > mass_floor) || !isfinite(mass)) {
    mass = mass_floor;
    if (mass_floor_delta != nullptr && isfinite(mass_raw)) {
      atomicAdd(mass_floor_delta, mass - mass_raw);
    }
  }

  mass_new[c] = mass;
  rho_new[c] = mass / V_ref;
  u_r_new[c] = (mass > 0.0 && isfinite(mom_r)) ? (mom_r / mass) : 0.0;
  u_z_new[c] = (mass > 0.0 && isfinite(mom_z)) ? (mom_z / mass) : 0.0;
  total_energy_new[c] = isfinite(total_energy) ? total_energy : 0.0;
  ye_int_new[c] = (mass > 0.0 && isfinite(ye_mass))
                      ? clamp01_device(ye_mass / mass)
                      : 0.5;
}

__device__ inline void apply_radiation_face_flux(double& energy,
                                                 const double dV,
                                                 const int donor,
                                                 const double sign,
                                                 const double* __restrict__ rad_E) {
  if (!finite_nonzero(dV)) {
    return;
  }
  energy += sign * dV * fmax(rad_E[donor], 0.0);
}

__device__ inline void apply_radiation_face_flux_second_order_r(
    double& energy,
    const double dV,
    const int K,
    const int Kp,
    const double sign,
    const double* __restrict__ rad_E,
    const double* __restrict__ x_r_lag,
    const double* __restrict__ x_z_lag,
    const double* __restrict__ x_r_ref,
    const double* __restrict__ x_z_ref,
    const int i_face,
    const int j,
    const int nr,
    const int nz,
    const std::uint8_t* __restrict__ cell_nverts,
    const int button_outer_node_ring = 0) {
  if (!finite_nonzero(dV)) {
    return;
  }
  const double e_f = rz_reconstructed_face_value_r(K,
                                                   Kp,
                                                   dV,
                                                   rad_E,
                                                   x_r_lag,
                                                   x_z_lag,
                                                   x_r_ref,
                                                   x_z_ref,
                                                   i_face,
                                                   j,
                                                   nr,
                                                   nz,
                                                   0.0,
                                                   cell_nverts,
                                                   button_outer_node_ring);
  energy += sign * dV * fmax(e_f, 0.0);
}

__device__ inline void apply_radiation_face_flux_second_order_z(
    double& energy,
    const double dV,
    const int K,
    const int Kp,
    const double sign,
    const double* __restrict__ rad_E,
    const double* __restrict__ x_r_lag,
    const double* __restrict__ x_z_lag,
    const double* __restrict__ x_r_ref,
    const double* __restrict__ x_z_ref,
    const int i,
    const int j_face,
    const int nr,
    const int nz,
    const std::uint8_t* __restrict__ cell_nverts,
    const int button_outer_node_ring = 0) {
  if (!finite_nonzero(dV)) {
    return;
  }
  const double e_f = rz_reconstructed_face_value_z(K,
                                                   Kp,
                                                   dV,
                                                   rad_E,
                                                   x_r_lag,
                                                   x_z_lag,
                                                   x_r_ref,
                                                   x_z_ref,
                                                   i,
                                                   j_face,
                                                   nr,
                                                   nz,
                                                   0.0,
                                                   cell_nverts,
                                                   button_outer_node_ring);
  energy += sign * dV * fmax(e_f, 0.0);
}

template <bool ClampNonnegative>
__global__ void ale_remap_2d_rz_radiation_kernel(
    const double* __restrict__ e_rad_lag,
    const double* __restrict__ vol_lag,
    const double* __restrict__ x_r_lag,
    const double* __restrict__ x_z_lag,
    const double* __restrict__ x_r_ref,
    const double* __restrict__ x_z_ref,
    const double* __restrict__ vol_ref,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ boundary_rad_flux_z_bottom,
	    const double* __restrict__ boundary_rad_flux_z_top,
	    double* __restrict__ e_rad_new,
	    const int nr,
	    const int nz,
	    const int button_outer_node_ring) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

	  const int i = c / nz;
	  const int j = c - i * nz;
	  if (rz_is_dormant_button_cell(c, i, button_outer_node_ring)) {
	    e_rad_new[c] = 0.0;
	    return;
	  }
	  double energy = fmax(e_rad_lag[c], 0.0) * fmax(vol_lag[c], kTinyVolume);

	  if (rz_is_button_cell(c, button_outer_node_ring)) {
	    for (int js = 0; js < nz; ++js) {
	      const double dV = rz_button_seam_swept_volume_ref(
	          x_r_lag, x_z_lag, x_r_ref, x_z_ref, button_outer_node_ring, nz, js);
	      const int shell = detail::cell_index(button_outer_node_ring, js, nz);
	      const int donor = rz_donor_cell(c, shell, dV);
	      apply_radiation_face_flux(energy, dV, donor, -1.0, e_rad_lag);
	    }
	  } else {
	    if (i + 1 < nr) {
	      const int Kp = detail::cell_index(i + 1, j, nz);
	      if (!rz_button_face_touches_dormant(c, Kp, nz, button_outer_node_ring)) {
	        const double dV = rz_swept_volume_r_face_ref_topology(
	            x_r_lag, x_z_lag, x_r_ref, x_z_ref, cell_nverts, i + 1, j, nr,
	            nz, button_outer_node_ring);
	        const int donor = rz_donor_cell(c, Kp, dV);
	        apply_radiation_face_flux(energy, dV, donor, -1.0, e_rad_lag);
	      }
	    }
	    if (i > 0) {
	      double dV = 0.0;
	      int donor = c;
	      const bool button_seam_face =
	          rz_button_enabled(button_outer_node_ring) &&
	          i == button_outer_node_ring;
	      const int K = button_seam_face ? 0 : detail::cell_index(i - 1, j, nz);
	      if (rz_button_enabled(button_outer_node_ring) &&
	          i == button_outer_node_ring) {
	        dV = rz_button_seam_swept_volume_ref(
	            x_r_lag, x_z_lag, x_r_ref, x_z_ref, button_outer_node_ring, nz, j);
	        donor = rz_donor_cell(0, c, dV);
	      } else {
	        dV = rz_swept_volume_r_face_ref_topology(
	            x_r_lag, x_z_lag, x_r_ref, x_z_ref, cell_nverts, i, j, nr, nz,
	            button_outer_node_ring);
	        donor = rz_donor_cell(K, c, dV);
	      }
	      if (!rz_button_face_touches_dormant(K, c, nz, button_outer_node_ring)) {
	        apply_radiation_face_flux(energy, dV, donor, 1.0, e_rad_lag);
	      }
	    }
	    if (j + 1 < nz) {
	      const int Kp = detail::cell_index(i, j + 1, nz);
	      if (!rz_button_face_touches_dormant(c, Kp, nz, button_outer_node_ring)) {
	        const double dV = rz_swept_volume_z_face_ref_topology(
	            x_r_lag, x_z_lag, x_r_ref, x_z_ref, cell_nverts, i, j + 1, nr,
	            nz, button_outer_node_ring);
	        const int donor = rz_donor_cell(c, Kp, dV);
	        apply_radiation_face_flux(energy, dV, donor, -1.0, e_rad_lag);
	      }
	    } else if (boundary_rad_flux_z_top != nullptr) {
	      energy += boundary_rad_flux_z_top[i];
	    }
	    if (j > 0) {
	      const int K = detail::cell_index(i, j - 1, nz);
	      if (!rz_button_face_touches_dormant(K, c, nz, button_outer_node_ring)) {
	        const double dV = rz_swept_volume_z_face_ref_topology(
	            x_r_lag, x_z_lag, x_r_ref, x_z_ref, cell_nverts, i, j, nr, nz,
	            button_outer_node_ring);
	        const int donor = rz_donor_cell(K, c, dV);
	        apply_radiation_face_flux(energy, dV, donor, 1.0, e_rad_lag);
	      }
	    } else if (boundary_rad_flux_z_bottom != nullptr) {
	      energy += boundary_rad_flux_z_bottom[i];
	    }
	  }

  if constexpr (ClampNonnegative) {
    e_rad_new[c] = fmax(energy / fmax(vol_ref[c], kTinyVolume), 0.0);
  } else {
    e_rad_new[c] = energy / fmax(vol_ref[c], kTinyVolume);
  }
}

template <bool ClampNonnegative>
__global__ void ale_remap_2d_rz_radiation_second_order_kernel(
    const double* __restrict__ e_rad_lag,
    const double* __restrict__ vol_lag,
    const double* __restrict__ x_r_lag,
    const double* __restrict__ x_z_lag,
    const double* __restrict__ x_r_ref,
    const double* __restrict__ x_z_ref,
    const double* __restrict__ vol_ref,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ boundary_rad_flux_z_bottom,
	    const double* __restrict__ boundary_rad_flux_z_top,
	    double* __restrict__ e_rad_new,
	    const int nr,
	    const int nz,
	    const int button_outer_node_ring) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

	  const int i = c / nz;
	  const int j = c - i * nz;
	  if (rz_is_dormant_button_cell(c, i, button_outer_node_ring)) {
	    e_rad_new[c] = 0.0;
	    return;
	  }
	  double energy = fmax(e_rad_lag[c], 0.0) * fmax(vol_lag[c], kTinyVolume);

	  if (rz_is_button_cell(c, button_outer_node_ring)) {
	    for (int js = 0; js < nz; ++js) {
	      const double dV = rz_button_seam_swept_volume_ref(
	          x_r_lag, x_z_lag, x_r_ref, x_z_ref, button_outer_node_ring, nz, js);
	      const int shell = detail::cell_index(button_outer_node_ring, js, nz);
	      const int donor = rz_donor_cell(c, shell, dV);
	      apply_radiation_face_flux(energy, dV, donor, -1.0, e_rad_lag);
	    }
	  } else {
	  if (i + 1 < nr) {
	    const int K = c;
	    const int Kp = detail::cell_index(i + 1, j, nz);
	    if (!rz_button_face_touches_dormant(K, Kp, nz, button_outer_node_ring)) {
		    const double dV = rz_swept_volume_r_face_ref_topology(
		        x_r_lag, x_z_lag, x_r_ref, x_z_ref, cell_nverts, i + 1, j, nr,
		        nz, button_outer_node_ring);
	    const bool button_first_shell_face =
	        rz_button_enabled(button_outer_node_ring) &&
	        i == button_outer_node_ring;
	    if (button_first_shell_face ||
	        rz_r_face_has_tri(cell_nverts, i + 1, j, nr, nz)) {
	      const int donor = rz_donor_cell(K, Kp, dV);
	      apply_radiation_face_flux(energy, dV, donor, -1.0, e_rad_lag);
	    } else {
      apply_radiation_face_flux_second_order_r(energy,
                                               dV,
                                               K,
                                               Kp,
                                               -1.0,
                                               e_rad_lag,
                                               x_r_lag,
                                               x_z_lag,
                                               x_r_ref,
                                               x_z_ref,
                                               i + 1,
                                               j,
                                               nr,
	                                               nz,
	                                               cell_nverts,
	                                               button_outer_node_ring);
    }
	    }
	  }
	  if (i > 0) {
	    const bool button_seam_face =
	        rz_button_enabled(button_outer_node_ring) &&
	        i == button_outer_node_ring;
	    const int K = button_seam_face ? 0 : detail::cell_index(i - 1, j, nz);
	    const int Kp = c;
	    if (!rz_button_face_touches_dormant(K, Kp, nz, button_outer_node_ring)) {
		    const double dV =
		        button_seam_face
		            ? rz_button_seam_swept_volume_ref(
		                  x_r_lag, x_z_lag, x_r_ref, x_z_ref,
		                  button_outer_node_ring, nz, j)
		            : rz_swept_volume_r_face_ref_topology(
		                  x_r_lag, x_z_lag, x_r_ref, x_z_ref, cell_nverts, i, j,
		                  nr, nz, button_outer_node_ring);
	    const bool button_first_shell_face =
	        rz_button_enabled(button_outer_node_ring) &&
	        i == button_outer_node_ring + 1;
	    if (button_seam_face || button_first_shell_face ||
	        rz_r_face_has_tri(cell_nverts, i, j, nr, nz)) {
	      const int donor = rz_donor_cell(K, Kp, dV);
	      apply_radiation_face_flux(energy, dV, donor, 1.0, e_rad_lag);
	    } else {
      apply_radiation_face_flux_second_order_r(energy,
                                               dV,
                                               K,
                                               Kp,
                                               1.0,
                                               e_rad_lag,
                                               x_r_lag,
                                               x_z_lag,
                                               x_r_ref,
                                               x_z_ref,
                                               i,
                                               j,
                                               nr,
	                                               nz,
	                                               cell_nverts,
	                                               button_outer_node_ring);
    }
	    }
  }
	  if (j + 1 < nz) {
	    const int K = c;
	    const int Kp = detail::cell_index(i, j + 1, nz);
	    if (!rz_button_face_touches_dormant(K, Kp, nz, button_outer_node_ring)) {
		    const double dV = rz_swept_volume_z_face_ref_topology(
		        x_r_lag, x_z_lag, x_r_ref, x_z_ref, cell_nverts, i, j + 1, nr,
		        nz, button_outer_node_ring);
	    const bool button_first_shell_face =
	        rz_button_enabled(button_outer_node_ring) &&
	        i == button_outer_node_ring;
	    if (button_first_shell_face ||
	        rz_z_face_has_tri(cell_nverts, i, j + 1, nr, nz)) {
	      const int donor = rz_donor_cell(K, Kp, dV);
	      apply_radiation_face_flux(energy, dV, donor, -1.0, e_rad_lag);
	    } else {
      apply_radiation_face_flux_second_order_z(energy,
                                               dV,
                                               K,
                                               Kp,
                                               -1.0,
                                               e_rad_lag,
                                               x_r_lag,
                                               x_z_lag,
                                               x_r_ref,
                                               x_z_ref,
                                               i,
                                               j + 1,
                                               nr,
	                                               nz,
	                                               cell_nverts,
	                                               button_outer_node_ring);
    }
	    }
  } else if (boundary_rad_flux_z_top != nullptr) {
    energy += boundary_rad_flux_z_top[i];
  }
	  if (j > 0) {
	    const int K = detail::cell_index(i, j - 1, nz);
	    const int Kp = c;
	    if (!rz_button_face_touches_dormant(K, Kp, nz, button_outer_node_ring)) {
		    const double dV = rz_swept_volume_z_face_ref_topology(
		        x_r_lag, x_z_lag, x_r_ref, x_z_ref, cell_nverts, i, j, nr, nz,
		        button_outer_node_ring);
	    const bool button_first_shell_face =
	        rz_button_enabled(button_outer_node_ring) &&
	        i == button_outer_node_ring;
	    if (button_first_shell_face ||
	        rz_z_face_has_tri(cell_nverts, i, j, nr, nz)) {
	      const int donor = rz_donor_cell(K, Kp, dV);
	      apply_radiation_face_flux(energy, dV, donor, 1.0, e_rad_lag);
	    } else {
      apply_radiation_face_flux_second_order_z(energy,
                                               dV,
                                               K,
                                               Kp,
                                               1.0,
                                               e_rad_lag,
                                               x_r_lag,
                                               x_z_lag,
                                               x_r_ref,
                                               x_z_ref,
                                               i,
                                               j,
                                               nr,
	                                               nz,
	                                               cell_nverts,
	                                               button_outer_node_ring);
    }
	    }
	  } else if (boundary_rad_flux_z_bottom != nullptr) {
	    energy += boundary_rad_flux_z_bottom[i];
	  }
	  }

	  if constexpr (ClampNonnegative) {
	    e_rad_new[c] = fmax(energy / fmax(vol_ref[c], kTinyVolume), 0.0);
	  } else {
	    e_rad_new[c] = energy / fmax(vol_ref[c], kTinyVolume);
	  }
}

__global__ void ale_remap_2d_rz_mass_preflight_check_kernel(
    int* __restrict__ first_rejected_cell,
    const double* __restrict__ predicted_rho,
    const double* __restrict__ pre_rho,
    const double* __restrict__ vol_new,
    const double* __restrict__ vol_old,
    const double* __restrict__ veto_floor,
    const int n_cells,
    const double reject_fraction) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const double predicted_mass = predicted_rho[c] * vol_new[c];
  const double pre_mass = pre_rho[c] * vol_old[c];
  const double veto_mass = veto_floor != nullptr ? veto_floor[c] : 0.0;
  // A floor-band cell's clamp is the floor closure's normal job (healthy band
  // ~1e-16 fractions); only material cells may veto a remap — measured:
  // floor-band vetoes starved all relief for 1,313 consecutive events while
  // the seam neighbor inverted (A497).
  if (predicted_mass < -reject_fraction * fmax(pre_mass, veto_mass)) {
    atomicMin(first_rejected_cell, c);
  }
}

__global__ void gather_group_field_kernel(double* __restrict__ out,
                                          const double* __restrict__ in,
                                          const int n_cells,
                                          const int n_groups,
                                          const int g) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  out[c] = in[c * n_groups + g];
}

__global__ void scatter_group_field_kernel(double* __restrict__ out,
                                           const double* __restrict__ in,
                                           const int n_cells,
                                           const int n_groups,
                                           const int g) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  out[c * n_groups + g] = in[c];
}

__global__ void recover_internal_from_total_energy_remap_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ total_energy,
    const double* __restrict__ ye_int,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const double* __restrict__ zbar,
    double* __restrict__ energy_floor_delta,
    int* __restrict__ floor_hit_count,
    const int nr,
    const int nz,
    const int button_outer_node_ring,
    const double gamma,
    const double A,
    const double fallback_z,
    const double te_floor,
    const double ti_floor) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  if (button_outer_node_ring > 0 && c != 0 && i < button_outer_node_ring) {
    return;
  }
  const double m = fmax(mass[c], 0.0);
  const double A_safe = fmax(A, 1.0e-30);
  const double gm1 = fmax(gamma - 1.0, 1.0e-30);
  const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : fallback_z;
  const double cv_i =
      tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  const double cv_e =
      z * tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  const double e_i_floor = cv_i * fmax(ti_floor, 0.0);
  const double e_e_floor = cv_e * fmax(te_floor, 0.0);
  if (!(m > 0.0) || !isfinite(m)) {
    ee[c] = e_e_floor;
    ei[c] = e_i_floor;
    return;
  }

  const double K = (button_outer_node_ring > 0 && c == 0)
                       ? rz_button_kinetic_for_cell(
                             mass, x_r, x_z, v_r_node, v_z_node,
                             button_outer_node_ring, nz)
                       : rz_corner_kinetic_for_cell(
                             mass, x_r, v_r_node, v_z_node, c, i, j, nz);
  const double e_tot = total_energy[c] / m;
  const double e_int_raw = e_tot - K / m;
  const double ye = clamp01_device(ye_int[c]);
  double e_e_raw = ye * e_int_raw;
  double e_i_raw = (1.0 - ye) * e_int_raw;
  if (!isfinite(e_e_raw) || !isfinite(e_i_raw)) {
    e_e_raw = e_e_floor;
    e_i_raw = e_i_floor;
  }
  const double e_e_new = fmax(e_e_raw, e_e_floor);
  const double e_i_new = fmax(e_i_raw, e_i_floor);
  double delta = 0.0;
  if (isfinite(e_e_raw) && e_e_raw < e_e_floor) {
    delta += (e_e_floor - e_e_raw) * m;
  }
  if (isfinite(e_i_raw) && e_i_raw < e_i_floor) {
    delta += (e_i_floor - e_i_raw) * m;
  }
  if (delta > 0.0 && isfinite(delta)) {
    if (energy_floor_delta != nullptr) {
      atomicAdd(energy_floor_delta, delta);
    }
    if (floor_hit_count != nullptr) {
      atomicAdd(floor_hit_count, 1);
    }
  }
  ee[c] = e_e_new;
  ei[c] = e_i_new;
}

__global__ void recover_internal_from_total_energy_remap_physical_ke_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ total_energy,
    const double* __restrict__ ye_int,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const double* __restrict__ zbar,
    double* __restrict__ energy_floor_delta,
    int* __restrict__ floor_hit_count,
    const std::uint8_t* __restrict__ cell_nverts,
    rz::CornerMassFallbackRecorder* fallback_recorder,
    const int corner_mass_convention,
    const int nr,
    const int nz,
    const int button_outer_node_ring,
    const double gamma,
    const double A,
    const double fallback_z,
    const double te_floor,
    const double ti_floor) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  const int i = c / nz;
  if (button_outer_node_ring > 0 && c != 0 && i < button_outer_node_ring) {
    return;
  }
  const double m = fmax(mass[c], 0.0);
  const double A_safe = fmax(A, 1.0e-30);
  const double gm1 = fmax(gamma - 1.0, 1.0e-30);
  const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : fallback_z;
  const double cv_i =
      tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  const double cv_e =
      z * tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  const double e_i_floor = cv_i * fmax(ti_floor, 0.0);
  const double e_e_floor = cv_e * fmax(te_floor, 0.0);
  if (!(m > 0.0) || !isfinite(m)) {
    ee[c] = e_e_floor;
    ei[c] = e_i_floor;
    return;
  }

  const double K = (button_outer_node_ring > 0 && c == 0)
                       ? rz_button_kinetic_for_cell(
                             mass, x_r, x_z, v_r_node, v_z_node,
                             button_outer_node_ring, nz)
                       : rz_physical_corner_kinetic_for_cell(
                             mass, x_r, x_z, v_r_node, v_z_node, cell_nverts,
                             c, nz, corner_mass_convention, fallback_recorder,
                             rz::kCornerMassFallbackStageStructuredPhysicalKeRecover,
                             -2);
  const double e_tot = total_energy[c] / m;
  const double e_int_raw = e_tot - K / m;
  const double ye = clamp01_device(ye_int[c]);
  double e_e_raw = ye * e_int_raw;
  double e_i_raw = (1.0 - ye) * e_int_raw;
  if (!isfinite(e_e_raw) || !isfinite(e_i_raw)) {
    e_e_raw = e_e_floor;
    e_i_raw = e_i_floor;
  }
  const double e_e_new = fmax(e_e_raw, e_e_floor);
  const double e_i_new = fmax(e_i_raw, e_i_floor);
  double delta = 0.0;
  if (isfinite(e_e_raw) && e_e_raw < e_e_floor) {
    delta += (e_e_floor - e_e_raw) * m;
  }
  if (isfinite(e_i_raw) && e_i_raw < e_i_floor) {
    delta += (e_i_floor - e_i_raw) * m;
  }
  if (delta > 0.0 && isfinite(delta)) {
    if (energy_floor_delta != nullptr) {
      atomicAdd(energy_floor_delta, delta);
    }
    if (floor_hit_count != nullptr) {
      atomicAdd(floor_hit_count, 1);
    }
  }
  ee[c] = e_e_new;
  ei[c] = e_i_new;
}

__global__ void eos_reclosure_ideal_gas_kernel(
    double* __restrict__ Te,
    double* __restrict__ Ti,
    double* __restrict__ Pe,
    double* __restrict__ Pi,
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ rho,
    const double* __restrict__ zbar,
    const int n_cells,
    const int nz,
    const int button_outer_node_ring,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int two_temperature,
    const double gamma,
    const double A,
    const double rho_floor,
    const double te_floor,
    const double ti_floor) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  if (button_outer_node_ring > 0 && c != 0) {
    const int i = c / nz;
    if (i < button_outer_node_ring) {
      return;
    }
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    return;
  }

  const double A_safe = fmax(A, 1.0e-30);
  const double gm1 = fmax(gamma - 1.0, 1.0e-30);
  const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : 1.0;
  const double cv_i =
      tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  const double cv_e =
      z * tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  if (two_temperature == 1) {
    double e_i = fmax(ei[c], cv_i * fmax(ti_floor, 0.0));
    double e_e = (cv_e > 0.0) ? fmax(ee[c], cv_e * fmax(te_floor, 0.0)) : 0.0;
    const double rho_c = fmax(rho[c], rho_floor);

    Ti[c] = (cv_i > 0.0) ? fmax(e_i / cv_i, ti_floor) : ti_floor;
    Te[c] = (cv_e > 0.0) ? fmax(e_e / cv_e, te_floor) : te_floor;
    ei[c] = e_i;
    ee[c] = e_e;
    Pi[c] = gm1 * rho_c * e_i;
    Pe[c] = gm1 * rho_c * e_e;
  } else {
    // 1T convention: ee = total, Pi = 0, Te = Ti = total/(cv_e+cv_i).
    const double e_tot = ee[c] + ei[c];
    const double cv_tot = fmax(cv_e + cv_i, 1.0e-30);
    const double T = e_tot / cv_tot;
    Te[c] = Ti[c] = fmax(T, te_floor);
    Pe[c] = gm1 * fmax(rho[c], rho_floor) * e_tot;
    Pi[c] = 0.0;
  }
}

// §19 W1: table-EOS twin of eos_reclosure_ideal_gas_kernel. Energy-authoritative
// (design §19 item 1). NOT yet launched from any path — integration is §19-W2.
__global__ void eos_reclosure_table_lte_kernel(
    double* __restrict__ Te,
    double* __restrict__ Ti,
    double* __restrict__ Pe,
    double* __restrict__ Pi,
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ rho,
    const double* __restrict__ mass,
    const double* __restrict__ zbar,
    const int n_cells,
    const int nz,
    const int button_outer_node_ring,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int two_temperature,
    const double A,
    const double rho_floor,
    const double te_floor,
    const double ti_floor,
    const tenryu::materials::DeviceEOSTableView tab_ion,
    const tenryu::materials::DeviceEOSTableView tab_ele,
    const tenryu::materials::DeviceEOSTableView tab_total,
    const int low_density_extrap,
    double* __restrict__ d_de_floor,        // length-1 accumulator; may be nullptr
    int* __restrict__ d_closure_status) {   // per-cell; may be nullptr
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  if (button_outer_node_ring > 0 && c != 0) {
    const int i = c / nz;
    if (i < button_outer_node_ring) {
      return;
    }
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    return;
  }

  const double rho_c = fmax(rho[c], rho_floor);
  const double m_c = mass[c];
  const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : 1.0;
  double ledger = 0.0;
  int status = 0;

  if (two_temperature == 1) {
    const auto ion = remap_eos::close_species_energy_authoritative(
        tab_ion,
        rho_c,
        m_c,
        &ei[c],
        ti_floor,
        1.0,
        A,
        low_density_extrap != 0,
        &ledger);
    const auto ele = remap_eos::close_species_energy_authoritative(
        tab_ele,
        rho_c,
        m_c,
        &ee[c],
        te_floor,
        z,
        A,
        low_density_extrap != 0,
        &ledger);
    Ti[c] = ion.T;
    Te[c] = ele.T;
    Pi[c] = ion.P;
    Pe[c] = ele.P;
    status = ion.status | ele.status;
  } else {
    const auto total = remap_eos::close_total_1t_energy_authoritative(
        tab_total,
        rho_c,
        m_c,
        &ee[c],
        te_floor,
        z,
        A,
        low_density_extrap != 0,
        &ledger);
    Te[c] = Ti[c] = total.T;
    Pe[c] = total.P;
    Pi[c] = 0.0;
    status = total.status;
  }

  if (d_closure_status != nullptr) {
    d_closure_status[c] = status;
  }
  if (d_de_floor != nullptr && ledger != 0.0) {
    atomicAdd(&d_de_floor[0], ledger);
  }
}

__global__ void zero_button_dormant_hydro_state_kernel(
    double* __restrict__ rho,
    double* __restrict__ mass,
    double* __restrict__ ee,
    double* __restrict__ ei,
    double* __restrict__ Te,
    double* __restrict__ Ti,
    double* __restrict__ Pe,
    double* __restrict__ Pi,
    double* __restrict__ Qvisc,
    double* __restrict__ zbar,
    double* __restrict__ hllc_mom_z_cell,
    const int n_cells,
    const int nz,
    const int button_outer_node_ring) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells ||
      !rz_is_dormant_button_cell_index(c, nz, button_outer_node_ring)) {
    return;
  }
  if (rho != nullptr) {
    rho[c] = 0.0;
  }
  if (mass != nullptr) {
    mass[c] = 0.0;
  }
  if (ee != nullptr) {
    ee[c] = 0.0;
  }
  if (ei != nullptr) {
    ei[c] = 0.0;
  }
  if (Te != nullptr) {
    Te[c] = 0.0;
  }
  if (Ti != nullptr) {
    Ti[c] = 0.0;
  }
  if (Pe != nullptr) {
    Pe[c] = 0.0;
  }
  if (Pi != nullptr) {
    Pi[c] = 0.0;
  }
  if (Qvisc != nullptr) {
    Qvisc[c] = 0.0;
  }
  if (zbar != nullptr) {
    zbar[c] = 0.0;
  }
  if (hllc_mom_z_cell != nullptr) {
    hllc_mom_z_cell[c] = 0.0;
  }
}

__global__ void zero_button_dormant_group_state_kernel(
    double* __restrict__ field,
    const int n_cells,
    const int n_groups,
    const int nz,
    const int button_outer_node_ring) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (field == nullptr || c >= n_cells ||
      !rz_is_dormant_button_cell_index(c, nz, button_outer_node_ring)) {
    return;
  }
  for (int g = 0; g < n_groups; ++g) {
    field[c * n_groups + g] = 0.0;
  }
}

__global__ void accumulate_max_abs_kernel(
    const double* __restrict__ values,
    const int count,
    double* __restrict__ maximum) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || values == nullptr ||
      maximum == nullptr) {
    return;
  }
  double value_max = maximum[0];
  for (int i = 0; i < count; ++i) {
    value_max = fmax(value_max, fabs(values[i]));
  }
  maximum[0] = value_max;
}

void accumulate_device_array_max_abs(const double* const values,
                                     const int count,
                                     double* const maximum) {
  if (values == nullptr || count <= 0 || maximum == nullptr) {
    return;
  }
  accumulate_max_abs_kernel<<<1, 1>>>(values, count, maximum);
  CUDA_CHECK(cudaGetLastError());
}

double* snapshot_device_array(const char* const tag,
                              const double* const source,
                              const std::size_t count) {
  if (source == nullptr || count == 0U) {
    return nullptr;
  }
  double* const snapshot = static_cast<double*>(
      core::device_scratch_acquire(tag, count * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(snapshot,
                        source,
                        count * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  return snapshot;
}

void restore_device_array(double* const destination,
                          const double* const snapshot,
                          const std::size_t count) {
  if (destination == nullptr || snapshot == nullptr || count == 0U) {
    return;
  }
  CUDA_CHECK(cudaMemcpy(destination,
                        snapshot,
                        count * sizeof(double),
                        cudaMemcpyDeviceToDevice));
}

double sum_device_array(const double* d_values, const int n) {
  if (d_values == nullptr || n <= 0) {
    return 0.0;
  }
  std::vector<double> host(static_cast<std::size_t>(n), 0.0);
  CUDA_CHECK(cudaMemcpy(host.data(),
                        d_values,
                        static_cast<std::size_t>(n) * sizeof(double),
                        cudaMemcpyDeviceToHost));
  long double sum = 0.0L;
  for (const double v : host) {
    sum += static_cast<long double>(v);
  }
  return static_cast<double>(sum);
}

bool env_flag_value_from_raw(const char* const raw) {
  if (raw == nullptr) {
    return false;
  }
  const std::string value(raw);
  return value == "1" || value == "true" || value == "TRUE" ||
         value == "yes" || value == "YES" || value == "on" ||
         value == "ON";
}

bool env_flag_enabled(const char* const name) {
  return env_flag_value_from_raw(std::getenv(name));
}

bool ale_velcoherence_env_enabled() {
  static const bool enabled =
      env_flag_enabled("TENRYU_I1B_DISC_ALE_VELCOHERENCE");
  return enabled;
}

bool diff_ref_diag_env_enabled() {
  static const bool enabled =
      env_flag_enabled("TENRYU_I1B_DIFFREF_DIAG");
  return enabled;
}

bool ale_velcoherence_enabled_for_step(const core::State& state,
                                       const core::Config& cfg) {
  const auto& diag = cfg.numerics.diagnostics.ale_velcoherence;
  if (!diag.enabled && !ale_velcoherence_env_enabled()) {
    return false;
  }
  const int every = std::max(diag.every_n_steps, 1);
  return (state.step % every) == 0;
}

std::string format_ale_velcoherence_value(const double value) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(16) << value;
  return oss.str();
}

bool csr_near_vacuum_forensics_enabled() {
  return env_flag_enabled("TENRYU_CSR_NEAR_VACUUM_FORENSICS") ||
         env_flag_enabled("TENRYU_I1B_DISC_CSR_NEAR_VACUUM_FORENSICS");
}

bool ale_physical_ke_remap_env_enabled() {
  static const bool enabled =
      env_flag_enabled("TENRYU_ALE_PHYSICAL_KE_REMAP");
  return enabled;
}

bool central_macro_remap_audit_env_enabled() {
  static const bool enabled =
      env_flag_enabled("TENRYU_CENTRAL_MACRO_REMAP_AUDIT");
  return enabled;
}

int remap_energy_audit_every_n() {
  static const int every = []() {
    const char* const raw = std::getenv("TENRYU_I1B_REMAP_ENERGY_AUDIT_EVERY");
    if (raw == nullptr) {
      return 32;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || parsed <= 0 || parsed > 1000000L) {
      return 32;
    }
    return static_cast<int>(parsed);
  }();
  return every;
}

// Macro rim-KE audit (mixed-core terminal-phase energy forensics): the energy
// budget counts, at every macro BOUNDARY node, the kinetic energy carried
// by the MEMBER-side cached corner masses (real velocities, member corner
// weights), but the TER/OptionB-velocity-remap conservation ledger excludes
// member cells entirely — so any boundary-node velocity change across the
// velocity remap moves that rim KE with no compensation. This audit
// quantifies Sum_rim m_member-corner * |v|^2/2 before/after the velocity
// remap; if its delta matches the budget's per-remap energy drop, the
// mechanism is confirmed (env TENRYU_I1B_RIM_KE_AUDIT).
bool rim_ke_audit_env_enabled() {
  return env_flag_enabled("TENRYU_I1B_RIM_KE_AUDIT");
}

// Arm (D) of the mixed-core energy-drift adjudication: measure the TER
// kinetic energy in the FROZEN cached corner-mass basis (state.corner_mass
// — the basis the dynamics' nodal masses and the energy budget use) at
// build, recover, and ke-cell-scale, instead of the OptionB transient
// first-moment/transported basis. The velocity remap still sets the
// velocities by its own corner-momentum conservation; the basis difference
// then lands SIGNED in the recovered internal energy (the staggered-remap
// "kinetic-energy fixup" of Kucharik–Shashkov, expressed in the budget's
// ledger), so the budget-measured total energy is conserved across the
// remap. Default off.
bool ter_frozen_ke_basis_env_enabled() {
  return env_flag_enabled("TENRYU_I1B_TER_FROZEN_KE_BASIS");
}

// Experiment D of the corner-mass basis-contract verdict: F-basis momentum
// projection across the velocity remap (convection-compensated form).
bool fbasis_projection_env_enabled() {
  return env_flag_enabled("TENRYU_I1B_VREMAP_FBASIS_PROJECTION");
}

struct BasisDefectPre;
void apply_fbasis_momentum_projection(
    core::State& state,
    const core::DeviceArray<double>& optionb_corner_mass,
    const BasisDefectPre& pre);

// PR1 of the corner-mass basis-contract verdict: per-remap basis-defect
// audit. The budget leak is dK^F - dK^B and the REAL dynamics perturbation
// is the nodal impulse defect deltaP_n = M_n^F (v+ - v-) -
// (M_n^{B,+} v+ - M_n^{B,-} v-): the velocity remap conserves momentum in
// its first-moment basis B while the next Lagrangian update divides by the
// frozen-basis nodal mass M^F. Logs global aggregates plus the macro
// boundary band's radial impulse and pseudo-work (the Q2
// implicate/exonerate metric for the rebound under-compression).
// Audit-only; bit-identical when the env cadence is unset.
int basis_defect_audit_every() {
  static const int every = [] {
    const char* raw = std::getenv("TENRYU_I1B_BASIS_DEFECT_AUDIT_EVERY");
    const int v = raw != nullptr ? std::atoi(raw) : 0;
    return v > 0 ? v : 0;
  }();
  return every;
}

struct BasisDefectPre {
  std::vector<double> v_r;
  std::vector<double> v_z;
  std::vector<double> corner_mass_b;
};

void basis_defect_capture_pre(const core::State& state,
                              BasisDefectPre* pre) {
  state.v_r.copy_to_host(pre->v_r);
  state.v_z.copy_to_host(pre->v_z);
  // M^{B,-}: the Option-B first-moment corner masses on the PRE-remap mesh.
  // The remap component populates its corner-mass buffer only DURING the
  // call (the first audit attempt read an empty buffer and every emit
  // silently early-returned), so build them host-side with the same
  // __host__ __device__ helper the remap uses.
  pre->corner_mass_b.clear();
  const int n_cells = state.mesh.topo.n_cells;
  if (!state.mesh.topo.multiblock.has_value()) {
    return;
  }
  const auto& mb_topo = *state.mesh.topo.multiblock;
  if (mb_topo.cell_node_csr_indices.size() <
      static_cast<std::size_t>(n_cells) * 4U) {
    return;
  }
  std::vector<double> x_r;
  std::vector<double> x_z;
  std::vector<double> mass;
  state.x_r.copy_to_host(x_r);
  state.x_z.copy_to_host(x_z);
  state.mass.copy_to_host(mass);
  const int n_nodes = static_cast<int>(x_r.size());
  pre->corner_mass_b.assign(static_cast<std::size_t>(n_cells) * 4U, 0.0);
  // Match the remap's basis domain: the Option-B remap excludes INACTIVE
  // macro members (their transported corner masses are zero), so the
  // pre-remap first-moment basis must exclude them too — otherwise every
  // audited remap reports a constant ~rim-KE-sized pseudo-defect at the
  // macro boundary (apples-vs-oranges; observed +3.9e8 erg/remap).
  std::vector<std::uint8_t> inactive;
  if (state.central_pseudo_core.inactive_member_mask.size() ==
      static_cast<std::size_t>(n_cells)) {
    inactive = state.central_pseudo_core.inactive_member_mask;
  }
  if (state.pole_angular_derefine.inactive_member_mask.size() ==
      static_cast<std::size_t>(n_cells)) {
    if (inactive.empty()) {
      inactive.assign(static_cast<std::size_t>(n_cells), 0U);
    }
    for (int c = 0; c < n_cells; ++c) {
      if (state.pole_angular_derefine
              .inactive_member_mask[static_cast<std::size_t>(c)] != 0U) {
        inactive[static_cast<std::size_t>(c)] = 1U;
      }
    }
  }
  const bool have_inactive = static_cast<int>(inactive.size()) == n_cells;
  for (int c = 0; c < n_cells; ++c) {
    if (have_inactive && inactive[static_cast<std::size_t>(c)] != 0U) {
      continue;
    }
    const std::size_t off = static_cast<std::size_t>(c) * 4U;
    const int nverts = mesh::mesh_topo_cell_active_nverts(
        state.mesh.cell_nverts.empty() ? nullptr : state.mesh.cell_nverts.data(),
        c);
    double r[4] = {0.0, 0.0, 0.0, 0.0};
    double z[4] = {0.0, 0.0, 0.0, 0.0};
    bool ok = true;
    for (int k = 0; k < nverts; ++k) {
      const int n =
          mb_topo.cell_node_csr_indices[off + static_cast<std::size_t>(k)];
      if (n < 0 || n >= n_nodes) {
        ok = false;
        break;
      }
      r[k] = x_r[static_cast<std::size_t>(n)];
      z[k] = x_z[static_cast<std::size_t>(n)];
    }
    if (!ok || nverts < 3) {
      continue;
    }
    optionb::first_moment_corner_masses(
        mass[static_cast<std::size_t>(c)], r, z, nverts,
        &pre->corner_mass_b[off]);
  }
}

void basis_defect_emit(const core::State& state,
                       const core::DeviceArray<double>& optionb_corner_mass,
                       const BasisDefectPre& pre) {
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (!state.mesh.topo.multiblock.has_value() ||
      state.corner_mass.size() != static_cast<std::size_t>(n_cells) * 4U ||
      pre.corner_mass_b.size() != static_cast<std::size_t>(n_cells) * 4U ||
      static_cast<int>(pre.v_r.size()) != n_nodes) {
    return;
  }
  const auto& mb_topo = *state.mesh.topo.multiblock;
  if (mb_topo.cell_node_csr_indices.size() <
      static_cast<std::size_t>(n_cells) * 4U) {
    return;
  }
  std::vector<double> v_r_post;
  std::vector<double> v_z_post;
  std::vector<double> mB_post;
  std::vector<double> mF;
  std::vector<double> x_r;
  std::vector<double> x_z;
  state.v_r.copy_to_host(v_r_post);
  state.v_z.copy_to_host(v_z_post);
  optionb_corner_mass.copy_to_host(mB_post);
  state.corner_mass.copy_to_host(mF);
  state.x_r.copy_to_host(x_r);
  state.x_z.copy_to_host(x_z);
  if (mB_post.size() != static_cast<std::size_t>(n_cells) * 4U) {
    return;
  }
  std::vector<double> MF(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> MBm(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> MBp(static_cast<std::size_t>(n_nodes), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t off = static_cast<std::size_t>(c) * 4U;
    for (int k = 0; k < 4; ++k) {
      const int n =
          mb_topo.cell_node_csr_indices[off + static_cast<std::size_t>(k)];
      if (n < 0 || n >= n_nodes) {
        continue;
      }
      const std::size_t nn = static_cast<std::size_t>(n);
      const double f = mF[off + static_cast<std::size_t>(k)];
      const double bm = pre.corner_mass_b[off + static_cast<std::size_t>(k)];
      const double bp = mB_post[off + static_cast<std::size_t>(k)];
      if (std::isfinite(f) && f > 0.0) {
        MF[nn] += f;
      }
      if (std::isfinite(bm)) {
        MBm[nn] += bm;
      }
      if (std::isfinite(bp)) {
        MBp[nn] += bp;
      }
    }
  }
  const auto& boundary_mask = state.central_pseudo_core.boundary_node_mask;
  const bool have_band =
      static_cast<int>(boundary_mask.size()) == n_nodes;
  long double dK_F = 0.0L;
  long double dK_B = 0.0L;
  long double abs_dP = 0.0L;
  long double band_dI_r = 0.0L;
  long double band_dW = 0.0L;
  int band_nodes = 0;
  for (int n = 0; n < n_nodes; ++n) {
    const std::size_t nn = static_cast<std::size_t>(n);
    const double vrm = pre.v_r[nn];
    const double vzm = pre.v_z[nn];
    const double vrp = v_r_post[nn];
    const double vzp = v_z_post[nn];
    if (!std::isfinite(vrm) || !std::isfinite(vzm) || !std::isfinite(vrp) ||
        !std::isfinite(vzp)) {
      continue;
    }
    const double v2m = vrm * vrm + vzm * vzm;
    const double v2p = vrp * vrp + vzp * vzp;
    dK_F += 0.5L * static_cast<long double>(MF[nn]) *
            (static_cast<long double>(v2p) - v2m);
    dK_B += 0.5L * (static_cast<long double>(MBp[nn]) * v2p -
                    static_cast<long double>(MBm[nn]) * v2m);
    const double dPr =
        MF[nn] * (vrp - vrm) - (MBp[nn] * vrp - MBm[nn] * vrm);
    const double dPz =
        MF[nn] * (vzp - vzm) - (MBp[nn] * vzp - MBm[nn] * vzm);
    abs_dP += std::sqrt(dPr * dPr + dPz * dPz);
    if (have_band && boundary_mask[nn] != 0U) {
      const double r = x_r[nn];
      const double z = x_z[nn];
      const double s = std::sqrt(r * r + z * z);
      if (s > 0.0) {
        band_dI_r += (dPr * r + dPz * z) / s;
      }
      band_dW += 0.5 * (dPr * (vrm + vrp) + dPz * (vzm + vzp));
      ++band_nodes;
    }
  }
  std::ostringstream oss;
  oss << "[basis_defect_audit] step=" << state.step
      << " dK_F=" << format_ale_velcoherence_value(static_cast<double>(dK_F))
      << " dK_B=" << format_ale_velcoherence_value(static_cast<double>(dK_B))
      << " R=" << format_ale_velcoherence_value(
                      static_cast<double>(dK_F - dK_B))
      << " abs_dP=" << format_ale_velcoherence_value(
                           static_cast<double>(abs_dP))
      << " band_dI_r=" << format_ale_velcoherence_value(
                              static_cast<double>(band_dI_r))
      << " band_dW=" << format_ale_velcoherence_value(
                            static_cast<double>(band_dW))
      << " band_nodes=" << band_nodes;
  core::log_info(oss.str());
}

void apply_fbasis_momentum_projection(
    core::State& state,
    const core::DeviceArray<double>& optionb_corner_mass,
    const BasisDefectPre& pre) {
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (!state.mesh.topo.multiblock.has_value() ||
      state.corner_mass.size() != static_cast<std::size_t>(n_cells) * 4U ||
      pre.corner_mass_b.size() != static_cast<std::size_t>(n_cells) * 4U ||
      static_cast<int>(pre.v_r.size()) != n_nodes) {
    return;
  }
  const auto& mb_topo = *state.mesh.topo.multiblock;
  if (mb_topo.cell_node_csr_indices.size() <
      static_cast<std::size_t>(n_cells) * 4U) {
    return;
  }
  std::vector<double> v_r_post;
  std::vector<double> v_z_post;
  std::vector<double> mB_post;
  std::vector<double> mF;
  state.v_r.copy_to_host(v_r_post);
  state.v_z.copy_to_host(v_z_post);
  optionb_corner_mass.copy_to_host(mB_post);
  state.corner_mass.copy_to_host(mF);
  if (mB_post.size() != static_cast<std::size_t>(n_cells) * 4U) {
    return;
  }
  std::vector<double> MF(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> MBm(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> MBp(static_cast<std::size_t>(n_nodes), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t off = static_cast<std::size_t>(c) * 4U;
    for (int k = 0; k < 4; ++k) {
      const int n =
          mb_topo.cell_node_csr_indices[off + static_cast<std::size_t>(k)];
      if (n < 0 || n >= n_nodes) {
        continue;
      }
      const std::size_t nn = static_cast<std::size_t>(n);
      const double f = mF[off + static_cast<std::size_t>(k)];
      const double bm = pre.corner_mass_b[off + static_cast<std::size_t>(k)];
      const double bp = mB_post[off + static_cast<std::size_t>(k)];
      if (std::isfinite(f) && f > 0.0) {
        MF[nn] += f;
      }
      if (std::isfinite(bm) && bm > 0.0) {
        MBm[nn] += bm;
      }
      if (std::isfinite(bp) && bp > 0.0) {
        MBp[nn] += bp;
      }
    }
  }
  int corrected = 0;
  double max_dev = 0.0;
  for (int n = 0; n < n_nodes; ++n) {
    const std::size_t nn = static_cast<std::size_t>(n);
    const double dvr = v_r_post[nn] - pre.v_r[nn];
    const double dvz = v_z_post[nn] - pre.v_z[nn];
    if (dvr == 0.0 && dvz == 0.0) {
      continue;
    }
    const double mf = MF[nn];
    const double mb_mid = 0.5 * (MBm[nn] + MBp[nn]);
    if (!(mf > 0.0) || !(mb_mid > 0.0) || !std::isfinite(mf) ||
        !std::isfinite(mb_mid)) {
      continue;
    }
    const double scale = mb_mid / mf;
    if (!std::isfinite(scale) || scale <= 0.0) {
      continue;
    }
    v_r_post[nn] = pre.v_r[nn] + scale * dvr;
    v_z_post[nn] = pre.v_z[nn] + scale * dvz;
    ++corrected;
    max_dev = std::max(max_dev, std::fabs(1.0 - scale));
  }
  if (corrected > 0) {
    state.v_r.copy_from_host(v_r_post);
    state.v_z.copy_from_host(v_z_post);
  }
  static int proj_log_count = 0;
  if (proj_log_count < 5 || (proj_log_count % 500) == 0) {
    std::ostringstream oss;
    oss << "[fbasis_projection] step=" << state.step
        << " corrected_nodes=" << corrected
        << " max_abs_1_minus_scale="
        << format_ale_velcoherence_value(max_dev);
    core::log_info(oss.str());
  }
  ++proj_log_count;
}

double compute_macro_rim_member_ke(const core::State& state) {
  const auto& pc = state.central_pseudo_core;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (!pc.built || pc.boundary_node_mask.empty() ||
      pc.member_cells.empty() || !state.mesh.topo.multiblock.has_value() ||
      state.corner_mass.size() != static_cast<std::size_t>(n_cells) * 4U ||
      static_cast<int>(pc.boundary_node_mask.size()) != n_nodes) {
    return 0.0;
  }
  const auto& mb_topo = *state.mesh.topo.multiblock;
  if (mb_topo.cell_node_csr_indices.size() <
      static_cast<std::size_t>(n_cells) * 4U) {
    return 0.0;
  }
  std::vector<double> corner_mass;
  state.corner_mass.copy_to_host(corner_mass);
  std::vector<double> v_r;
  std::vector<double> v_z;
  state.v_r.copy_to_host(v_r);
  state.v_z.copy_to_host(v_z);
  std::vector<double> rim_mass(static_cast<std::size_t>(n_nodes), 0.0);
  for (const int c : pc.member_cells) {
    const std::size_t off = static_cast<std::size_t>(c) * 4U;
    for (int k = 0; k < 4; ++k) {
      const int n =
          mb_topo.cell_node_csr_indices[off + static_cast<std::size_t>(k)];
      if (n < 0 || n >= n_nodes ||
          pc.boundary_node_mask[static_cast<std::size_t>(n)] == 0U) {
        continue;
      }
      const double m = corner_mass[off + static_cast<std::size_t>(k)];
      if (m > 0.0 && std::isfinite(m)) {
        rim_mass[static_cast<std::size_t>(n)] += m;
      }
    }
  }
  long double ke = 0.0L;
  for (int n = 0; n < n_nodes; ++n) {
    const double m = rim_mass[static_cast<std::size_t>(n)];
    if (m <= 0.0) {
      continue;
    }
    const double vr = v_r[static_cast<std::size_t>(n)];
    const double vz = v_z[static_cast<std::size_t>(n)];
    if (std::isfinite(vr) && std::isfinite(vz)) {
      ke += 0.5L * static_cast<long double>(m) *
            (static_cast<long double>(vr) * vr +
             static_cast<long double>(vz) * vz);
    }
  }
  return static_cast<double>(ke);
}


bool remap_energy_audit_env_enabled() {
  static const bool enabled =
      env_flag_enabled("TENRYU_I1B_REMAP_ENERGY_AUDIT");
  return enabled;
}

int env_int_value(const char* const name, const int fallback) {
  const char* const raw = std::getenv(name);
  if (raw == nullptr) {
    return fallback;
  }
  char* end = nullptr;
  const long parsed = std::strtol(raw, &end, 10);
  if (end == raw || !std::isfinite(static_cast<double>(parsed))) {
    return fallback;
  }
  return static_cast<int>(parsed);
}

double env_double_value(const char* const name, const double fallback) {
  const char* const raw = std::getenv(name);
  if (raw == nullptr) {
    return fallback;
  }
  char* end = nullptr;
  const double parsed = std::strtod(raw, &end);
  if (end == raw || !std::isfinite(parsed)) {
    return fallback;
  }
  return parsed;
}

enum class P3OracleMode {
  Disabled,
  Vcell,
  Vnode,
  Energy,
  Combined,
};

struct P3OracleConfig {
  P3OracleMode mode = P3OracleMode::Disabled;
  double H = 5.0e4;
  double rho0 = 1.0;
  double pe0 = 1.0e10;
  double pi0 = 1.0e10;
  double gamma = 5.0 / 3.0;
};

const P3OracleConfig& p3_oracle_config() {
  static const P3OracleConfig config = [] {
    P3OracleConfig value;
    const char* const raw = std::getenv("TENRYU_I1B_P3_ORACLE");
    if (raw == nullptr || raw[0] == '\0') {
      return value;
    }
    const std::string mode(raw);
    if (mode == "vcell") {
      value.mode = P3OracleMode::Vcell;
    } else if (mode == "vnode") {
      value.mode = P3OracleMode::Vnode;
    } else if (mode == "e") {
      value.mode = P3OracleMode::Energy;
    } else if (mode == "combined") {
      value.mode = P3OracleMode::Combined;
    } else {
      TENRYU_ASSERT(false,
                    "TENRYU_I1B_P3_ORACLE must be one of "
                    "{vcell, vnode, e, combined}");
    }
    value.H = env_double_value("TENRYU_I1B_P3_ORACLE_H", 5.0e4);
    value.rho0 = env_double_value("TENRYU_I1B_P3_ORACLE_RHO0", 1.0);
    value.pe0 = env_double_value("TENRYU_I1B_P3_ORACLE_PE0", 1.0e10);
    value.pi0 = env_double_value("TENRYU_I1B_P3_ORACLE_PI0", 1.0e10);
    value.gamma =
        env_double_value("TENRYU_I1B_P3_ORACLE_GAMMA", 5.0 / 3.0);
    return value;
  }();
  return config;
}

const char* p3_oracle_mode_name(const P3OracleMode mode) {
  switch (mode) {
    case P3OracleMode::Vcell:
      return "vcell";
    case P3OracleMode::Vnode:
      return "vnode";
    case P3OracleMode::Energy:
      return "e";
    case P3OracleMode::Combined:
      return "combined";
    case P3OracleMode::Disabled:
      return "disabled";
  }
  return "disabled";
}

void p3_oracle_log_once(const P3OracleConfig& config,
                        const char* const station) {
  static bool emitted = false;
  if (emitted) {
    return;
  }
  emitted = true;
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(16)
      << "[p3-oracle] mode=" << p3_oracle_mode_name(config.mode)
      << " H=" << config.H << " applied_at=" << station;
  core::log_info(oss.str());
}

bool optionb_macroboundary_audit_env_enabled() {
  const char* const raw = std::getenv("TENRYU_I1B_OPTIONB_MACROBOUNDARY_AUDIT");
  if (raw == nullptr) {
    return false;
  }
  const std::string value(raw);
  return value != "0" && value != "false" && value != "FALSE" &&
         value != "no" && value != "NO" && value != "off" && value != "OFF";
}

int optionb_macroboundary_audit_mode() {
  const char* const mode_raw =
      std::getenv("TENRYU_I1B_OPTIONB_MACROBOUNDARY_AUDIT_MODE");
  if (mode_raw != nullptr) {
    const int mode =
        env_int_value("TENRYU_I1B_OPTIONB_MACROBOUNDARY_AUDIT_MODE", 3);
    return (mode >= 1 && mode <= 3) ? mode : 3;
  }
  const char* const audit_raw =
      std::getenv("TENRYU_I1B_OPTIONB_MACROBOUNDARY_AUDIT");
  if (audit_raw != nullptr && (audit_raw[0] == '2' || audit_raw[0] == '3') &&
      audit_raw[1] == '\0') {
    return audit_raw[0] - '0';
  }
  return 3;
}

double optionb_macroboundary_audit_H() {
  return env_double_value("TENRYU_I1B_OPTIONB_MACROBOUNDARY_H", 1.0e9);
}

double optionb_macroboundary_audit_x0_r() {
  return env_double_value("TENRYU_I1B_OPTIONB_MACROBOUNDARY_X0_R", 0.0);
}

double optionb_macroboundary_audit_x0_z() {
  return env_double_value("TENRYU_I1B_OPTIONB_MACROBOUNDARY_X0_Z", 0.0);
}

int optionb_macroboundary_ring_start() {
  return env_int_value("TENRYU_I1B_OPTIONB_MACROBOUNDARY_RING_START", 320);
}

int optionb_macroboundary_ring_end() {
  return env_int_value("TENRYU_I1B_OPTIONB_MACROBOUNDARY_RING_END", 383);
}

bool optionb_ring5_momentum_trace_env_enabled() {
  static const bool enabled =
      env_flag_enabled("TENRYU_I1B_OPTIONB_RING5_MOMENTUM_TRACE");
  return enabled;
}

int optionb_ring5_momentum_trace_steps() {
  static const int steps =
      std::max(0, env_int_value("TENRYU_I1B_OPTIONB_RING5_TRACE_STEPS", 5));
  return steps;
}

bool optionb_ring5_momentum_trace_enabled_for_state(
    const core::State& state) {
  return optionb_ring5_momentum_trace_env_enabled() &&
         state.step < optionb_ring5_momentum_trace_steps();
}

bool remap_energy_audit_enabled_for_invocation(const core::State& state) {
  if (!remap_energy_audit_env_enabled()) {
    return false;
  }
  const long long remap = state.ale_remaps_applied + 1;
  if (remap <= 8) {
    return true;
  }
  const int every = remap_energy_audit_every_n();
  return every > 0 && (remap % every) == 0;
}

template <typename T>
void copy_device_pointer_to_host(const T* const d_values,
                                 const int n,
                                 std::vector<T>& host) {
  host.assign(static_cast<std::size_t>(std::max(n, 0)), T{});
  if (d_values == nullptr || n <= 0) {
    return;
  }
  CUDA_CHECK(cudaMemcpy(host.data(),
                        d_values,
                        static_cast<std::size_t>(n) * sizeof(T),
                        cudaMemcpyDeviceToHost));
}

void record_gcl_audit_transaction(
    const int step,
    const bool configured_swept_volume_sign_fixed,
    const bool active_swept_volume_sign_fixed,
    const bool mass_flux_limited,
    const std::vector<double>& residual,
    const std::vector<double>& scale,
    const std::vector<std::uint8_t>& inactive_cell_mask) {
  GclAuditRuntime& runtime = gcl_audit_runtime();
  if (!runtime.initialized) {
    runtime.initialized = true;
  }
  double local_max_abs_R = 0.0;
  double local_scale = 0.0;
  double local_sum_R = 0.0;
  double local_sum_scale = 0.0;
  double local_active = 0.0;
  for (std::size_t c = 0; c < residual.size(); ++c) {
    if (!inactive_cell_mask.empty() && inactive_cell_mask[c] != 0U) {
      continue;
    }
    local_max_abs_R = std::max(local_max_abs_R, std::abs(residual[c]));
    local_scale = std::max(local_scale, scale[c]);
    local_sum_R += residual[c];
    local_sum_scale += scale[c];
    local_active += 1.0;
  }
  double maxima[2] = {local_max_abs_R, local_scale};
  double sums[3] = {local_sum_R, local_sum_scale, local_active};
  if (runtime.reduction != nullptr) {
    runtime.reduction->allreduce_max(maxima, 2);
    runtime.reduction->allreduce_sum(sums, 3);
  }

  GclAuditTransactionSummary summary;
  summary.step = step;
  summary.max_abs_R = maxima[0];
  summary.max_rel_R = maxima[1] > 0.0 ? maxima[0] / maxima[1] : 0.0;
  summary.sum_R = sums[0];
  summary.sum_rel = sums[1] > 0.0 ? std::abs(sums[0]) / sums[1] : 0.0;
  summary.n_cells_active = static_cast<int>(std::llround(sums[2]));

  if (!runtime.manifest_emitted && runtime.rank == 0) {
    runtime.manifest_emitted = true;
    std::ostringstream manifest;
    manifest << std::boolalpha
             << "[gcl-audit] manifest cfg.numerics.ale."
                "swept_volume_sign_fixed="
             << configured_swept_volume_sign_fixed
             << " active_swept_volume_sign_fixed="
             << active_swept_volume_sign_fixed
             << " audited_pass=primary_hydro"
             << " consumed_swept_set="
             << (mass_flux_limited ? "mass_flux_limited" : "raw_geometric");
    core::log_info(manifest.str());
  }
  if (runtime.rank == 0) {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(17)
        << "[gcl-audit] step=" << summary.step
        << " max_abs_R=" << summary.max_abs_R
        << " max_rel_R=" << summary.max_rel_R
        << " sum_R=" << summary.sum_R
        << " sum_rel=" << summary.sum_rel
        << " n_cells_active=" << summary.n_cells_active;
    core::log_info(oss.str());
  }
  ++runtime.transaction_count;
  if (!runtime.has_transaction ||
      summary.max_rel_R > runtime.worst.max_rel_R ||
      (summary.max_rel_R == runtime.worst.max_rel_R &&
       summary.max_abs_R > runtime.worst.max_abs_R)) {
    runtime.worst = summary;
    runtime.has_transaction = true;
  }
}

bool csr_swept_audit_valid_cell_face(
    const tenryu::mesh::MultiBlockTopology& mb,
    const int n_cells,
    const int cell,
    const int local_face) {
  if (cell < 0 || cell >= n_cells || local_face < 0) {
    return false;
  }
  if (mb.cell_node_csr_offsets.size() !=
          static_cast<std::size_t>(n_cells) + 1U ||
      mb.cell_node_csr_indices.size() <
          static_cast<std::size_t>(mb.cell_node_csr_offsets.back()) ||
      mb.cell_orientation_sign.size() != static_cast<std::size_t>(n_cells)) {
    return false;
  }
  const int slot_width =
      mb.cell_node_csr_offsets[static_cast<std::size_t>(cell + 1)] -
      mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  return local_face < slot_width;
}

bool evaluate_csr_swept_face_audit(
    const core::State& state,
    const tenryu::mesh::MultiBlockTopology& mb,
    const bool dgcl_commit_gate,
    const double dgcl_commit_rtol) {
  const bool audit_enabled = conservation_audit::enabled();
  if (!audit_enabled && !dgcl_commit_gate) {
    return true;
  }
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (n_cells <= 0 || n_nodes <= 0 ||
      state.x_r_reference.size() != state.x_r.size() ||
      state.x_z_reference.size() != state.x_z.size() ||
      state.cell_vol_initial.size() != state.vol.size()) {
    return true;
  }

  std::vector<double> x_r_old;
  std::vector<double> x_z_old;
  std::vector<double> x_r_new;
  std::vector<double> x_z_new;
  std::vector<double> vol_old;
  std::vector<double> vol_new;
  state.x_r.copy_to_host(x_r_old);
  state.x_z.copy_to_host(x_z_old);
  state.x_r_reference.copy_to_host(x_r_new);
  state.x_z_reference.copy_to_host(x_z_new);
  state.vol.copy_to_host(vol_old);
  state.cell_vol_initial.copy_to_host(vol_new);
  if (x_r_old.size() != static_cast<std::size_t>(n_nodes) ||
      x_z_old.size() != static_cast<std::size_t>(n_nodes) ||
      x_r_new.size() != static_cast<std::size_t>(n_nodes) ||
      x_z_new.size() != static_cast<std::size_t>(n_nodes) ||
      vol_old.size() != static_cast<std::size_t>(n_cells) ||
      vol_new.size() != static_cast<std::size_t>(n_cells)) {
    return true;
  }

  const std::uint8_t* const cell_nverts =
      state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)
          ? state.mesh.cell_nverts.data()
          : nullptr;
  const int* const offsets = mb.cell_node_csr_offsets.data();
  const int* const indices = mb.cell_node_csr_indices.data();
  const int* const orientation = mb.cell_orientation_sign.data();
  std::vector<double> swept_sum(static_cast<std::size_t>(n_cells), 0.0);
  std::unordered_map<long long, int> edge_owner_count;
  edge_owner_count.reserve(mb.unique_internal_faces.size() +
                           mb.boundary_faces.size());

  int invalid_faces = 0;
  int duplicate_internal_edges = 0;
  int nonreversed_internal_edges = 0;
  int swept_pair_bad = 0;
  int worst_pair_face = -1;
  int worst_pair_cell_a = -1;
  int worst_pair_cell_b = -1;
  double worst_pair_abs = 0.0;
  double worst_pair_rel = 0.0;
  double worst_pair_dv_a = 0.0;
  double worst_pair_dv_b = 0.0;

  const auto edge_key = [n_nodes](const int a, const int b) -> long long {
    const int lo = std::min(a, b);
    const int hi = std::max(a, b);
    return static_cast<long long>(lo) * static_cast<long long>(n_nodes) +
           static_cast<long long>(hi);
  };
  const auto note_edge = [&](const int a, const int b, const bool internal) {
    if (a < 0 || b < 0 || a >= n_nodes || b >= n_nodes || a == b) {
      return;
    }
    const long long key = edge_key(a, b);
    const int previous = edge_owner_count[key]++;
    if (internal && previous > 0) {
      ++duplicate_internal_edges;
    }
  };

  for (int f = 0; f < static_cast<int>(mb.unique_internal_faces.size()); ++f) {
    const auto& face = mb.unique_internal_faces[static_cast<std::size_t>(f)];
    if (!csr_swept_audit_valid_cell_face(mb, n_cells, face.cell_a,
                                         face.local_a) ||
        !csr_swept_audit_valid_cell_face(mb, n_cells, face.cell_b,
                                         face.local_b)) {
      ++invalid_faces;
      continue;
    }
    int a0 = -1;
    int a1 = -1;
    int b0 = -1;
    int b1 = -1;
    const bool ok_a = detail::csr_face_swept_node_indices(
        offsets, indices, cell_nverts, face.cell_a, face.local_a, &a0, &a1);
    const bool ok_b = detail::csr_face_swept_node_indices(
        offsets, indices, cell_nverts, face.cell_b, face.local_b, &b0, &b1);
    if (!ok_a || !ok_b) {
      ++invalid_faces;
      continue;
    }
    note_edge(a0, a1, true);
    if (!(a0 == b1 && a1 == b0)) {
      ++nonreversed_internal_edges;
    }
    const double dV_a = detail::csr_face_swept_volume_outward(
        x_r_old.data(), x_z_old.data(), x_r_new.data(), x_z_new.data(),
        offsets, indices, orientation, face.cell_a, face.local_a, cell_nverts);
    const double dV_b = detail::csr_face_swept_volume_outward(
        x_r_old.data(), x_z_old.data(), x_r_new.data(), x_z_new.data(),
        offsets, indices, orientation, face.cell_b, face.local_b, cell_nverts);
    swept_sum[static_cast<std::size_t>(face.cell_a)] += dV_a;
    swept_sum[static_cast<std::size_t>(face.cell_b)] += dV_b;
    const double pair_abs = std::abs(dV_a + dV_b);
    const double pair_rel =
        pair_abs / std::max(std::abs(dV_a) + std::abs(dV_b), 1.0e-300);
    const double pair_tol =
        256.0 * std::numeric_limits<double>::epsilon() *
        std::max(std::abs(dV_a) + std::abs(dV_b), 1.0e-300);
    if (pair_abs > pair_tol) {
      ++swept_pair_bad;
    }
    if (pair_abs > worst_pair_abs) {
      worst_pair_abs = pair_abs;
      worst_pair_rel = pair_rel;
      worst_pair_face = f;
      worst_pair_cell_a = face.cell_a;
      worst_pair_cell_b = face.cell_b;
      worst_pair_dv_a = dV_a;
      worst_pair_dv_b = dV_b;
    }
  }

  for (const auto& face : mb.boundary_faces) {
    if (!csr_swept_audit_valid_cell_face(mb, n_cells, face.cell_a,
                                         face.local_a)) {
      ++invalid_faces;
      continue;
    }
    int a0 = -1;
    int a1 = -1;
    if (!detail::csr_face_swept_node_indices(
            offsets, indices, cell_nverts, face.cell_a, face.local_a, &a0,
            &a1)) {
      ++invalid_faces;
      continue;
    }
    note_edge(a0, a1, false);
    swept_sum[static_cast<std::size_t>(face.cell_a)] +=
        detail::csr_face_swept_volume_outward(
            x_r_old.data(), x_z_old.data(), x_r_new.data(), x_z_new.data(),
            offsets, indices, orientation, face.cell_a, face.local_a,
            cell_nverts);
  }

  int volume_bad_cells = 0;
  int worst_volume_cell = -1;
  double worst_volume_abs = 0.0;
  double worst_volume_rel = 0.0;
  int first_dgcl_bad_cell = -1;
  double first_dgcl_bad_resid = 0.0;
  for (int c = 0; c < n_cells; ++c) {
    const double residual =
        vol_new[static_cast<std::size_t>(c)] -
        vol_old[static_cast<std::size_t>(c)] -
        swept_sum[static_cast<std::size_t>(c)];
    const double scale = std::max(
        std::max(std::abs(vol_new[static_cast<std::size_t>(c)]),
                 std::abs(vol_old[static_cast<std::size_t>(c)])),
        std::max(std::abs(swept_sum[static_cast<std::size_t>(c)]),
                 1.0e-300));
    const double rel = std::abs(residual) / scale;
    const double tol = 512.0 * std::numeric_limits<double>::epsilon() * scale;
    if (std::abs(residual) > tol) {
      ++volume_bad_cells;
    }
    if (std::abs(residual) > worst_volume_abs) {
      worst_volume_abs = std::abs(residual);
      worst_volume_rel = rel;
      worst_volume_cell = c;
    }
    if (dgcl_commit_gate && first_dgcl_bad_cell < 0) {
      const double dgcl_residual = std::abs(
          swept_sum[static_cast<std::size_t>(c)] -
          (vol_new[static_cast<std::size_t>(c)] -
           vol_old[static_cast<std::size_t>(c)]));
      const double dgcl_scale =
          std::max({vol_old[static_cast<std::size_t>(c)],
                    vol_new[static_cast<std::size_t>(c)],
                    1.0e-300});
      if (!(dgcl_residual <= dgcl_commit_rtol * dgcl_scale)) {
        first_dgcl_bad_cell = c;
        first_dgcl_bad_resid = dgcl_residual;
      }
    }
  }

  if (audit_enabled) {
    std::fprintf(stderr,
                 "[i1b_cons_swept_audit] step=%d t=%.17e "
                 "internal_faces=%zu boundary_faces=%zu invalid_faces=%d "
                 "duplicate_internal_edges=%d nonreversed_internal_edges=%d "
                 "swept_pair_bad=%d worst_pair_face=%d worst_pair_cells=%d:%d "
                 "worst_pair_abs=%.17e worst_pair_rel=%.17e "
                 "worst_pair_dV_a=%.17e worst_pair_dV_b=%.17e "
                 "volume_bad_cells=%d worst_volume_cell=%d "
                 "worst_volume_abs=%.17e worst_volume_rel=%.17e\n",
                 state.step,
                 state.t,
                 mb.unique_internal_faces.size(),
                 mb.boundary_faces.size(),
                 invalid_faces,
                 duplicate_internal_edges,
                 nonreversed_internal_edges,
                 swept_pair_bad,
                 worst_pair_face,
                 worst_pair_cell_a,
                 worst_pair_cell_b,
                 worst_pair_abs,
                 worst_pair_rel,
                 worst_pair_dv_a,
                 worst_pair_dv_b,
                 volume_bad_cells,
                 worst_volume_cell,
                 worst_volume_abs,
                 worst_volume_rel);
  }
  if (first_dgcl_bad_cell >= 0) {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(17)
        << "[dgcl-commit-gate] rejected: cell=" << first_dgcl_bad_cell
        << " resid=" << first_dgcl_bad_resid
        << " rtol=" << dgcl_commit_rtol;
    core::log_warning(oss.str());
    return false;
  }
  return true;
}

bool remap_energy_audit_inactive(
    const std::vector<std::uint8_t>& inactive_mask,
    const int c) {
  return c >= 0 && static_cast<std::size_t>(c) < inactive_mask.size() &&
         inactive_mask[static_cast<std::size_t>(c)] != 0U;
}

long double remap_energy_audit_reference_energy(const long double E0_pre) {
  static long double E_ref = 0.0L;
  if (!(E_ref > 0.0L) || !std::isfinite(E_ref)) {
    E_ref = (std::fabs(E0_pre) > 0.0L && std::isfinite(E0_pre))
                ? std::fabs(E0_pre)
                : 1.0L;
  }
  return E_ref;
}

long double remap_energy_audit_delta(const long double value,
                                     const long double reference) {
  if (!(std::fabs(reference) > 0.0L) || !std::isfinite(reference)) {
    return 0.0L;
  }
  return value / reference;
}

std::string remap_energy_audit_format(const long double value) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(18) << value;
  return oss.str();
}

std::string csr_cons_audit_format(const long double value) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(17) << value;
  return oss.str();
}

long double csr_cons_audit_sum_vector(const std::vector<double>& values) {
  long double sum = 0.0L;
  for (const double value : values) {
    if (std::isfinite(value)) {
      sum += static_cast<long double>(value);
    }
  }
  return sum;
}

double csr_cons_audit_reduce_sum(const double local,
                                 const parallel::Reduction* const reduction) {
  return reduction != nullptr ? reduction->allreduce_sum(local) : local;
}

struct CsrConsAuditTotals {
  double mass = 0.0;
  double pr = 0.0;
  double pz = 0.0;
  double K = 0.0;
  double Ui = 0.0;
  double Ue = 0.0;
  double Erad = 0.0;
  double E = 0.0;
};

struct CsrConsAuditTopCell {
  int cell = -1;
  long double value = 0.0L;
};

struct CsrConsAuditRepairSummary {
  int n_U_negative_before_repair = 0;
  long double U_deficit = 0.0L;
  long double U_donor_surplus = 0.0L;
  std::array<CsrConsAuditTopCell, 3> top_deficit{};
};

void csr_cons_audit_update_top(std::array<CsrConsAuditTopCell, 3>& top,
                               const int cell,
                               const long double value) {
  const long double magnitude = std::fabs(value);
  if (!(magnitude > 0.0L) || !std::isfinite(magnitude)) {
    return;
  }
  for (int k = 0; k < 3; ++k) {
    if (top[static_cast<std::size_t>(k)].cell < 0 ||
        magnitude > std::fabs(top[static_cast<std::size_t>(k)].value)) {
      for (int j = 2; j > k; --j) {
        top[static_cast<std::size_t>(j)] =
            top[static_cast<std::size_t>(j - 1)];
      }
      top[static_cast<std::size_t>(k)].cell = cell;
      top[static_cast<std::size_t>(k)].value = value;
      return;
    }
  }
}

std::array<CsrConsAuditTopCell, 3> csr_cons_audit_top_delta(
    const std::vector<double>& before,
    const std::vector<double>& after) {
  std::array<CsrConsAuditTopCell, 3> top{};
  const std::size_t n = std::min(before.size(), after.size());
  for (std::size_t c = 0; c < n; ++c) {
    const long double delta =
        static_cast<long double>(after[c]) - static_cast<long double>(before[c]);
    csr_cons_audit_update_top(top, static_cast<int>(c), delta);
  }
  return top;
}

std::string csr_cons_audit_format_top(
    const std::array<CsrConsAuditTopCell, 3>& top) {
  std::ostringstream oss;
  bool any = false;
  for (const CsrConsAuditTopCell& entry : top) {
    if (entry.cell < 0) {
      continue;
    }
    if (any) {
      oss << ";";
    }
    oss << entry.cell << ":" << csr_cons_audit_format(entry.value);
    any = true;
  }
  return any ? oss.str() : "none";
}

void csr_remap_energy_audit_corner_masses_host(
    double* m_corner,
    int c,
    const std::vector<double>& mass,
    const std::vector<double>& x_r,
    const std::vector<double>& x_z,
    const std::vector<int>& cell_node_offsets,
    const std::vector<int>& cell_node_indices,
    const std::vector<std::uint8_t>& cell_nverts,
    const bool partition_normalized,
    const bool polar_tier_equal_planar_area);

void csr_remap_energy_audit_physical_corner_masses_host(
    double* m_corner,
    int c,
    int nz,
    const std::vector<double>& mass,
    const std::vector<double>& x_r,
    const std::vector<double>& x_z,
    const std::vector<std::uint8_t>& cell_nverts,
    const int corner_mass_convention);

void csr_remap_energy_audit_physical_cell_nodes(int* nodes, int c, int nz);

long double csr_remap_energy_audit_floor_sum(
    int c,
    const std::vector<double>& zbar,
    double gamma,
    double A,
    double te_floor,
    double ti_floor);

double csr_cons_audit_rad_energy_local(const core::State& state,
                                       const core::Config& cfg) {
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_groups =
      static_cast<std::size_t>(std::max(cfg.radiation.groups, 1));
  if (n_cells == 0 || state.rad_E.size() != n_cells * n_groups ||
      state.vol.size() != n_cells) {
    return 0.0;
  }
  std::vector<double> rad_E;
  std::vector<double> vol;
  state.rad_E.copy_to_host(rad_E);
  state.vol.copy_to_host(vol);
  long double total = 0.0L;
  for (std::size_t c = 0; c < n_cells; ++c) {
    const long double V = static_cast<long double>(vol[c]);
    if (!(V > 0.0L) || !std::isfinite(V)) {
      continue;
    }
    for (std::size_t g = 0; g < n_groups; ++g) {
      const double E = rad_E[c * n_groups + g];
      if (std::isfinite(E)) {
        total += static_cast<long double>(E) * V;
      }
    }
  }
  return static_cast<double>(total);
}

bool csr_cons_audit_pair_momentum_local(const core::State& state,
                                        double* const pr,
                                        double* const pz) {
  if (pr == nullptr || pz == nullptr) {
    return false;
  }
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (n_cells <= 0 || n_nodes <= 0 ||
      !state.mesh.topo.multiblock.has_value() ||
      !state.corner_mass_initialized ||
      state.corner_mass.size() !=
          static_cast<std::size_t>(n_cells) *
              static_cast<std::size_t>(state.corner_stride) ||
      state.v_r.size() != static_cast<std::size_t>(n_nodes) ||
      state.v_z.size() != static_cast<std::size_t>(n_nodes) ||
      state.mesh.multiblock_cell_node_csr_offsets.size() !=
          static_cast<std::size_t>(n_cells + 1) ||
      state.mesh.multiblock_cell_node_csr_indices.empty()) {
    return false;
  }

  std::vector<double> corner_mass;
  std::vector<double> v_r;
  std::vector<double> v_z;
  std::vector<int> offsets;
  std::vector<int> indices;
  state.corner_mass.copy_to_host(corner_mass);
  state.v_r.copy_to_host(v_r);
  state.v_z.copy_to_host(v_z);
  state.mesh.multiblock_cell_node_csr_offsets.copy_to_host(offsets);
  state.mesh.multiblock_cell_node_csr_indices.copy_to_host(indices);
  if (offsets.size() < static_cast<std::size_t>(n_cells + 1)) {
    return false;
  }

  long double local_pr = 0.0L;
  long double local_pz = 0.0L;
  for (int c = 0; c < n_cells; ++c) {
    const int off = offsets[static_cast<std::size_t>(c)];
    const int end = offsets[static_cast<std::size_t>(c + 1)];
    if (off < 0 || end < off ||
        end > static_cast<int>(indices.size())) {
      continue;
    }
    int active_nverts =
        state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)
            ? mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c)
            : state.corner_stride;
    active_nverts = std::min({active_nverts, end - off,
                              state.corner_stride});
    for (int k = 0; k < active_nverts; ++k) {
      const int n = indices[static_cast<std::size_t>(off + k)];
      if (n < 0 || n >= n_nodes) {
        continue;
      }
      const double m =
          corner_mass[static_cast<std::size_t>(c) *
                          static_cast<std::size_t>(state.corner_stride) +
                      static_cast<std::size_t>(k)];
      if (!std::isfinite(m)) {
        continue;
      }
      local_pr += static_cast<long double>(m) *
                  static_cast<long double>(v_r[static_cast<std::size_t>(n)]);
      local_pz += static_cast<long double>(m) *
                  static_cast<long double>(v_z[static_cast<std::size_t>(n)]);
    }
  }
  *pr = static_cast<double>(local_pr);
  *pz = static_cast<double>(local_pz);
  return true;
}

std::vector<double> csr_cons_audit_capture_state_cell_energy(
    const core::State& state,
    const core::Config& cfg) {
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (n_cells <= 0 || n_nodes <= 0 ||
      !state.mesh.topo.multiblock.has_value() ||
      !state.corner_mass_initialized ||
      state.corner_mass.size() !=
          static_cast<std::size_t>(n_cells) *
              static_cast<std::size_t>(state.corner_stride) ||
      state.mass.size() != static_cast<std::size_t>(n_cells) ||
      state.ee.size() != static_cast<std::size_t>(n_cells) ||
      state.ei.size() != static_cast<std::size_t>(n_cells) ||
      state.v_r.size() != static_cast<std::size_t>(n_nodes) ||
      state.v_z.size() != static_cast<std::size_t>(n_nodes) ||
      state.mesh.multiblock_cell_node_csr_offsets.size() !=
          static_cast<std::size_t>(n_cells + 1) ||
      state.mesh.multiblock_cell_node_csr_indices.empty()) {
    return {};
  }

  std::vector<double> mass;
  std::vector<double> ee;
  std::vector<double> ei;
  std::vector<double> corner_mass;
  std::vector<double> v_r;
  std::vector<double> v_z;
  std::vector<double> rad_E;
  std::vector<double> vol;
  std::vector<int> offsets;
  std::vector<int> indices;
  state.mass.copy_to_host(mass);
  state.ee.copy_to_host(ee);
  state.ei.copy_to_host(ei);
  state.corner_mass.copy_to_host(corner_mass);
  state.v_r.copy_to_host(v_r);
  state.v_z.copy_to_host(v_z);
  state.mesh.multiblock_cell_node_csr_offsets.copy_to_host(offsets);
  state.mesh.multiblock_cell_node_csr_indices.copy_to_host(indices);

  const int n_groups = std::max(cfg.radiation.groups, 1);
  const std::size_t expected_rad =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  const bool include_rad =
      state.rad_E.size() == expected_rad &&
      state.vol.size() == static_cast<std::size_t>(n_cells);
  if (include_rad) {
    state.rad_E.copy_to_host(rad_E);
    state.vol.copy_to_host(vol);
  }

  std::vector<double> cells(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const auto ci = static_cast<std::size_t>(c);
    long double total = static_cast<long double>(mass[ci]) *
                        (static_cast<long double>(ee[ci]) +
                         static_cast<long double>(ei[ci]));
    const int off = offsets[ci];
    const int end = offsets[static_cast<std::size_t>(c + 1)];
    if (off < 0 || end < off ||
        end > static_cast<int>(indices.size())) {
      continue;
    }
    int active_nverts =
        state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)
            ? mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c)
            : state.corner_stride;
    active_nverts = std::min({active_nverts, end - off,
                              state.corner_stride});
    for (int k = 0; k < active_nverts; ++k) {
      const int n = indices[static_cast<std::size_t>(off + k)];
      if (n < 0 || n >= n_nodes) {
        continue;
      }
      const long double cm =
          static_cast<long double>(
              corner_mass[ci * static_cast<std::size_t>(state.corner_stride) +
                          static_cast<std::size_t>(k)]);
      const long double vr =
          static_cast<long double>(v_r[static_cast<std::size_t>(n)]);
      const long double vz =
          static_cast<long double>(v_z[static_cast<std::size_t>(n)]);
      if (std::isfinite(cm) && std::isfinite(vr) && std::isfinite(vz)) {
        total += 0.5L * cm * (vr * vr + vz * vz);
      }
    }
    if (include_rad) {
      const long double V = static_cast<long double>(vol[ci]);
      if (V > 0.0L && std::isfinite(V)) {
        const std::size_t base =
            ci * static_cast<std::size_t>(n_groups);
        for (int g = 0; g < n_groups; ++g) {
          const double E = rad_E[base + static_cast<std::size_t>(g)];
          if (std::isfinite(E)) {
            total += static_cast<long double>(E) * V;
          }
        }
      }
    }
    cells[ci] = static_cast<double>(total);
  }
  return cells;
}

CsrConsAuditTotals csr_cons_audit_capture_state_totals(
    const core::State& state,
    const core::Config& cfg,
    const parallel::Reduction* const reduction) {
  CsrConsAuditTotals out{};
  out.mass = csr_cons_audit_reduce_sum(sum_device_array(
                                           state.mass.data(),
                                           static_cast<int>(state.mass.size())),
                                       reduction);
  diagnostics::EnergyTotals energy = diagnostics::compute_energy_totals_2d(state);
  if (reduction != nullptr) {
    double values[3] = {energy.E_int_e, energy.E_int_i, energy.E_kin};
    reduction->allreduce_sum(values, 3);
    energy.E_int_e = values[0];
    energy.E_int_i = values[1];
    energy.E_kin = values[2];
  }
  out.Ue = energy.E_int_e;
  out.Ui = energy.E_int_i;
  out.K = energy.E_kin;
  out.Erad = csr_cons_audit_reduce_sum(
      csr_cons_audit_rad_energy_local(state, cfg), reduction);
  out.E = out.K + out.Ui + out.Ue + out.Erad;
  double pr = 0.0;
  double pz = 0.0;
  if (csr_cons_audit_pair_momentum_local(state, &pr, &pz)) {
    if (reduction != nullptr) {
      double values[2] = {pr, pz};
      reduction->allreduce_sum(values, 2);
      pr = values[0];
      pz = values[1];
    }
  }
  out.pr = pr;
  out.pz = pz;
  return out;
}

CsrConsAuditRepairSummary csr_cons_audit_repair_summary(
    const core::State& state,
    const core::Config& cfg,
    const std::vector<double>& total_energy,
    const double gamma,
    const double A,
    const double* const d_optionb_corner_mass,
    const bool optionb_velocity_authority,
    const bool physical_ke_remap,
    const parallel::Reduction* const reduction) {
  CsrConsAuditRepairSummary summary{};
  // 1T convention stores the total internal energy in ee with ei == 0; flooring
  // ei at cv_i*Ti_floor inside the remap fabricates unledgered energy (measured
  // +5e-5/pass via the next step's 1T fold-back). Zero the ion floor in 1T.
  const double ti_floor_remap =
      cfg.main.two_temperature ? cfg.numerics.floors.Ti : 0.0;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (n_cells <= 0 || n_nodes <= 0 ||
      total_energy.size() != static_cast<std::size_t>(n_cells)) {
    return summary;
  }
  std::vector<double> mass;
  std::vector<double> x_r;
  std::vector<double> x_z;
  std::vector<double> v_r;
  std::vector<double> v_z;
  std::vector<double> zbar;
  std::vector<int> cell_node_offsets;
  std::vector<int> cell_node_indices;
  std::vector<double> optionb_corner_mass;
  state.mass.copy_to_host(mass);
  state.x_r.copy_to_host(x_r);
  state.x_z.copy_to_host(x_z);
  state.v_r.copy_to_host(v_r);
  state.v_z.copy_to_host(v_z);
  if (!state.zbar.empty()) {
    state.zbar.copy_to_host(zbar);
  }
  state.mesh.multiblock_cell_node_csr_offsets.copy_to_host(cell_node_offsets);
  state.mesh.multiblock_cell_node_csr_indices.copy_to_host(cell_node_indices);
  if (optionb_velocity_authority && d_optionb_corner_mass != nullptr) {
    copy_device_pointer_to_host(d_optionb_corner_mass,
                                n_cells * 4,
                                optionb_corner_mass);
  }

  long double local_deficit = 0.0L;
  long double local_surplus = 0.0L;
  int local_negative = 0;
  std::array<CsrConsAuditTopCell, 3> local_top{};
  const int nz = state.mesh.topo.nz;
  for (int c = 0; c < n_cells; ++c) {
    double m_corner[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    if (optionb_velocity_authority &&
        optionb_corner_mass.size() == static_cast<std::size_t>(n_cells) * 4U) {
      const int base = 4 * c;
      for (int k = 0; k < 4; ++k) {
        m_corner[k] = optionb_corner_mass[static_cast<std::size_t>(base + k)];
      }
    } else if (physical_ke_remap) {
      csr_remap_energy_audit_physical_corner_masses_host(
          m_corner, c, nz, mass, x_r, x_z, state.mesh.cell_nverts,
          static_cast<int>(cfg.numerics.hydro.corner_mass_convention));
    } else {
      csr_remap_energy_audit_corner_masses_host(m_corner,
                                                c,
                                                mass,
                                                x_r,
                                                x_z,
                                                cell_node_offsets,
                                                cell_node_indices,
                                                state.mesh.cell_nverts,
                                                corner_mass_lagrangian_invariant_enabled(cfg),
                                                mesh::mesh_topo_polar_tier_family(cfg.mesh));
    }
    const int active_nverts =
        mesh::mesh_topo_cell_active_nverts(
            state.mesh.cell_nverts.empty() ? nullptr : state.mesh.cell_nverts.data(),
            c);
    const int off = physical_ke_remap
                        ? 0
                        : cell_node_offsets[static_cast<std::size_t>(c)];
    int physical_nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    if (physical_ke_remap) {
      csr_remap_energy_audit_physical_cell_nodes(physical_nodes, c, nz);
    }
    long double K_c = 0.0L;
    for (int k = 0; k < active_nverts; ++k) {
      const int n = physical_ke_remap
                        ? physical_nodes[k]
                        : cell_node_indices[static_cast<std::size_t>(off + k)];
      if (n < 0 || n >= n_nodes) {
        continue;
      }
      const long double cm =
          (m_corner[k] > 0.0 && std::isfinite(m_corner[k]))
              ? static_cast<long double>(m_corner[k])
              : 0.0L;
      const long double vr =
          static_cast<long double>(v_r[static_cast<std::size_t>(n)]);
      const long double vz =
          static_cast<long double>(v_z[static_cast<std::size_t>(n)]);
      if (cm > 0.0L && std::isfinite(vr) && std::isfinite(vz)) {
        K_c += 0.5L * cm * (vr * vr + vz * vz);
      }
    }
    const long double m = static_cast<long double>(mass[static_cast<std::size_t>(c)]);
    const long double U_raw =
        static_cast<long double>(total_energy[static_cast<std::size_t>(c)]) -
        K_c;
    const long double U_floor =
        (m > 0.0L && std::isfinite(m))
            ? m * csr_remap_energy_audit_floor_sum(c,
                                                   zbar,
                                                   gamma,
                                                   A,
                                                   cfg.numerics.floors.Te,
                                                   ti_floor_remap)
            : 0.0L;
    if (U_raw < 0.0L && std::isfinite(U_raw)) {
      local_negative += 1;
    }
    const long double deficit =
        std::isfinite(U_raw) ? std::max(0.0L, U_floor - U_raw) : U_floor;
    const long double surplus =
        std::isfinite(U_raw) ? std::max(0.0L, U_raw - U_floor) : 0.0L;
    local_deficit += deficit;
    local_surplus += surplus;
    csr_cons_audit_update_top(local_top, c, deficit);
  }

  double reduced[3] = {static_cast<double>(local_deficit),
                       static_cast<double>(local_surplus),
                       static_cast<double>(local_negative)};
  if (reduction != nullptr) {
    reduction->allreduce_sum(reduced, 3);
  }
  summary.U_deficit = reduced[0];
  summary.U_donor_surplus = reduced[1];
  summary.n_U_negative_before_repair =
      static_cast<int>(std::llround(reduced[2]));
  summary.top_deficit = local_top;
  return summary;
}

struct CsrConsAuditLedger {
  bool active = false;
  CsrConsAuditTotals B;
  CsrConsAuditTotals E;
  CsrConsAuditTotals F;
  CsrConsAuditTotals core_aggregate;
  CsrConsAuditTotals G;
  long double E_C = 0.0L;
  long double E_D = 0.0L;
  std::vector<double> cell_B;
  std::vector<double> cell_C;
  std::vector<double> cell_D;
  std::vector<double> cell_E;
  std::vector<double> cell_F;
  std::vector<double> cell_core_aggregate;
  std::vector<double> cell_G;
  CsrConsAuditRepairSummary repair;
};

void csr_cons_audit_capture_staged(const double* const d_total_energy,
                                   const int n_cells,
                                   std::vector<double>* const cells,
                                   long double* const total) {
  if (cells == nullptr || total == nullptr) {
    return;
  }
  copy_device_pointer_to_host(d_total_energy, n_cells, *cells);
  *total = csr_cons_audit_sum_vector(*cells);
}

void csr_cons_audit_reduce_staged(long double* const total,
                                  const parallel::Reduction* const reduction) {
  if (total == nullptr || reduction == nullptr) {
    return;
  }
  *total = static_cast<long double>(
      reduction->allreduce_sum(static_cast<double>(*total)));
}

void csr_cons_audit_emit(const CsrConsAuditLedger& ledger,
                         const CsrConsAuditContext& context) {
  if (!ledger.active) {
    return;
  }
  const long double E_B = ledger.B.E;
  const long double E_C = ledger.E_C + static_cast<long double>(ledger.B.Erad);
  const long double E_D = ledger.E_D + static_cast<long double>(ledger.B.Erad);
  const long double E_E = ledger.E.E;
  const long double E_F = ledger.F.E;
  const long double E_core_aggregate = ledger.core_aggregate.E;
  const long double E_G = ledger.G.E;
  const long double dE_CSR = E_C - E_B;
  const long double dE_core = E_D - E_C;
  const long double dE_recover = E_E - E_D;
  const long double dE_pos_repair = E_F - E_E;
  const long double dE_core_aggregate = E_core_aggregate - E_F;
  const long double dE_commit = E_G - E_F;
  const long double dE_post_core_commit = E_G - E_core_aggregate;
  const long double denom =
      std::max(std::fabs(E_B), static_cast<long double>(1.0e-300));
  const bool print_top =
      std::fabs(dE_CSR) / denom > 1.0e-9L ||
      std::fabs(dE_core) / denom > 1.0e-9L ||
      std::fabs(dE_recover) / denom > 1.0e-9L ||
      std::fabs(dE_pos_repair) / denom > 1.0e-9L ||
      std::fabs(dE_core_aggregate) / denom > 1.0e-9L ||
      std::fabs(dE_commit) / denom > 1.0e-9L;
  const auto top_csr = csr_cons_audit_top_delta(ledger.cell_B, ledger.cell_C);
  const auto top_core = csr_cons_audit_top_delta(ledger.cell_C, ledger.cell_D);
  const auto top_recover =
      csr_cons_audit_top_delta(ledger.cell_D, ledger.cell_E);
  const auto top_pos = csr_cons_audit_top_delta(ledger.cell_E, ledger.cell_F);
  const auto top_core_aggregate =
      csr_cons_audit_top_delta(ledger.cell_F, ledger.cell_core_aggregate);
  const auto top_commit =
      csr_cons_audit_top_delta(ledger.cell_core_aggregate, ledger.cell_G);

  std::ostringstream oss;
  oss << "[csr_cons_audit]"
      << " step=" << context.step
      << " t=" << csr_cons_audit_format(context.t)
      << " dt=" << csr_cons_audit_format(context.dt)
      << " remap_id=" << context.remap_id
      << " trigger_cell=" << context.trigger_cell
      << " min_path_margin=" << csr_cons_audit_format(context.min_path_margin)
      << " M_pre=" << csr_cons_audit_format(ledger.B.mass)
      << " M_post=" << csr_cons_audit_format(ledger.G.mass)
      << " dM=" << csr_cons_audit_format(ledger.G.mass - ledger.B.mass)
      << " dPr_pair=" << csr_cons_audit_format(ledger.G.pr - ledger.B.pr)
      << " dPz_pair=" << csr_cons_audit_format(ledger.G.pz - ledger.B.pz)
      << " E_B=" << csr_cons_audit_format(E_B)
      << " E_C=" << csr_cons_audit_format(E_C)
      << " dE_CSR=" << csr_cons_audit_format(dE_CSR)
      << " E_D=" << csr_cons_audit_format(E_D)
      << " dE_core=" << csr_cons_audit_format(dE_core)
      << " E_E=" << csr_cons_audit_format(E_E)
      << " dE_recover=" << csr_cons_audit_format(dE_recover)
      << " E_F=" << csr_cons_audit_format(E_F)
      << " dE_pos_repair=" << csr_cons_audit_format(dE_pos_repair)
      << " n_U_negative_before_repair="
      << ledger.repair.n_U_negative_before_repair
      << " U_deficit=" << csr_cons_audit_format(ledger.repair.U_deficit)
      << " U_donor_surplus="
      << csr_cons_audit_format(ledger.repair.U_donor_surplus)
      << " E_core_aggregate=" << csr_cons_audit_format(E_core_aggregate)
      << " dE_core_aggregate="
      << csr_cons_audit_format(dE_core_aggregate)
      << " E_G=" << csr_cons_audit_format(E_G)
      << " dE_commit=" << csr_cons_audit_format(dE_commit)
      << " dE_post_core_commit="
      << csr_cons_audit_format(dE_post_core_commit);
  if (print_top) {
    oss << " top_CSR=" << csr_cons_audit_format_top(top_csr)
        << " top_core=" << csr_cons_audit_format_top(top_core)
        << " top_recover=" << csr_cons_audit_format_top(top_recover)
        << " top_repair="
        << csr_cons_audit_format_top(ledger.repair.top_deficit)
        << " top_pos=" << csr_cons_audit_format_top(top_pos)
        << " top_core_aggregate="
        << csr_cons_audit_format_top(top_core_aggregate)
        << " top_post_core_commit="
        << csr_cons_audit_format_top(top_commit);
  }
  core::log_warning(oss.str());
}

struct CsrRemapEnergyAuditState {
  bool active = false;
  long long step = 0;
  long long remap = 0;
  long double E0_pre = 0.0L;
  long double E_swept = 0.0L;
  long double E_T_R = 0.0L;
  long double macro_E0_pre = 0.0L;
  long double macro_E_swept = 0.0L;
  long double macro_E_T_R = 0.0L;
  int macro_cells = 0;
  std::vector<std::uint8_t> inactive_mask;
  std::vector<double> total_energy_recover;
};

void csr_remap_energy_audit_compute_extensive_total(
    const std::vector<double>& extensive_total,
    const std::vector<std::uint8_t>& inactive_mask,
    long double* const total,
    long double* const macro_total) {
  *total = 0.0L;
  *macro_total = 0.0L;
  for (std::size_t c = 0; c < extensive_total.size(); ++c) {
    const long double E = static_cast<long double>(extensive_total[c]);
    *total += E;
    if (remap_energy_audit_inactive(inactive_mask, static_cast<int>(c))) {
      *macro_total += E;
    }
  }
}

void csr_remap_energy_audit_capture_pre(
    CsrRemapEnergyAuditState& audit,
    const core::State& state,
    const double* const d_total_energy_initial,
    const std::uint8_t* const d_inactive_cell_mask,
    const int n_cells) {
  (void)state;
  if (!audit.active) {
    return;
  }
  copy_device_pointer_to_host(d_inactive_cell_mask, n_cells, audit.inactive_mask);
  std::vector<double> total_energy_initial;
  copy_device_pointer_to_host(
      d_total_energy_initial, n_cells, total_energy_initial);
  csr_remap_energy_audit_compute_extensive_total(total_energy_initial,
                                                 audit.inactive_mask,
                                                 &audit.E0_pre,
                                                 &audit.macro_E0_pre);
  audit.macro_cells = 0;
  for (int c = 0; c < n_cells; ++c) {
    if (remap_energy_audit_inactive(audit.inactive_mask, c)) {
      audit.macro_cells += 1;
    }
  }
}

void csr_remap_energy_audit_capture_swept(
    CsrRemapEnergyAuditState& audit,
    const double* const d_total_energy_new,
    const int n_cells) {
  if (!audit.active) {
    return;
  }
  std::vector<double> total_energy_swept;
  copy_device_pointer_to_host(d_total_energy_new, n_cells, total_energy_swept);
  csr_remap_energy_audit_compute_extensive_total(total_energy_swept,
                                                 audit.inactive_mask,
                                                 &audit.E_swept,
                                                 &audit.macro_E_swept);
}

void csr_remap_energy_audit_capture_recover_total(
    CsrRemapEnergyAuditState& audit,
    const double* const d_total_energy_new,
    const int n_cells) {
  if (!audit.active) {
    return;
  }
  copy_device_pointer_to_host(
      d_total_energy_new, n_cells, audit.total_energy_recover);
  csr_remap_energy_audit_compute_extensive_total(audit.total_energy_recover,
                                                 audit.inactive_mask,
                                                 &audit.E_T_R,
                                                 &audit.macro_E_T_R);
}

void csr_remap_energy_audit_corner_masses_host(
    double* const m_corner,
    const int c,
    const std::vector<double>& mass,
    const std::vector<double>& x_r,
    const std::vector<double>& x_z,
    const std::vector<int>& cell_node_offsets,
    const std::vector<int>& cell_node_indices,
    const std::vector<std::uint8_t>& cell_nverts,
    const bool partition_normalized,
    const bool polar_tier_equal_planar_area) {
  for (int k = 0; k < mesh::kMeshTopoCellStorageSlotsMax; ++k) {
    m_corner[k] = 0.0;
  }
  const int active_nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  const int off = cell_node_offsets[static_cast<std::size_t>(c)];
  const double m_cell =
      (mass[static_cast<std::size_t>(c)] > 0.0 &&
       std::isfinite(mass[static_cast<std::size_t>(c)]))
          ? mass[static_cast<std::size_t>(c)]
          : 0.0;
  double r[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  double z[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  for (int k = 0; k < active_nverts; ++k) {
    const int n = cell_node_indices[static_cast<std::size_t>(off + k)];
    r[k] = x_r[static_cast<std::size_t>(n)];
    z[k] = x_z[static_cast<std::size_t>(n)];
  }
  // Audit basis == dynamical basis: the tri and quad legs mirror the
  // partition selection of compute_corner_mass_2d_multiblock_kernel
  // (hydro_2d.cu). Pentagon and star-P1 legs are deliberately unchanged
  // (A124(b) P-A ruling, 2026-08-15).
  if (active_nverts == 3) {
    if (polar_tier_equal_planar_area) {
      rz::compute_triangle_corner_masses_equal_planar_area(
          m_cell, r[0], z[0], r[1], z[1], r[2], z[2], m_corner);
    } else {
      rz::compute_triangle_corner_masses_exact(
          m_cell, r[0], z[0], r[1], z[1], r[2], z[2], m_corner);
    }
  } else if (active_nverts == 4) {
    if (partition_normalized) {
      rz::compute_quad_corner_masses_partitioned_subpolygon(
          m_cell, r[0], z[0], r[1], z[1], r[2], z[2], r[3], z[3], m_corner);
    } else {
      rz::compute_quad_corner_masses_exact_subpolygon(
          m_cell, r[0], z[0], r[1], z[1], r[2], z[2], r[3], z[3], m_corner);
    }
  } else if (active_nverts == 5) {
    csr_compute_pentagon_qk_corner_masses(m_cell, r, z, m_corner);
  } else if (active_nverts >= 6 &&
             active_nverts <= mesh::kMeshTopoCellStorageSlotsMax) {
    double w[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    mesh::moments::star_p1_vertex_r_moments(r, z, active_nverts, w);
    double W = 0.0;
    bool finite_weights = true;
    for (int k = 0; k < active_nverts; ++k) {
      W += w[k];
      finite_weights = finite_weights && std::isfinite(w[k]);
    }
    if (!(W > 0.0) || !finite_weights) {
      const double uniform = m_cell / static_cast<double>(active_nverts);
      for (int k = 0; k < active_nverts; ++k) {
        m_corner[k] = uniform;
      }
    } else {
      for (int k = 0; k < active_nverts; ++k) {
        m_corner[k] = m_cell * (w[k] / W);
      }
    }
  }
}

void csr_remap_energy_audit_physical_corner_masses_host(
    double* const m_corner,
    const int c,
    const int nz,
    const std::vector<double>& mass,
    const std::vector<double>& x_r,
    const std::vector<double>& x_z,
    const std::vector<std::uint8_t>& cell_nverts,
    const int corner_mass_convention) {
  for (int k = 0; k < mesh::kMeshTopoCellStorageSlotsMax; ++k) {
    m_corner[k] = 0.0;
  }
  if (nz <= 0) {
    return;
  }
  const double m_cell =
      (mass[static_cast<std::size_t>(c)] > 0.0 &&
       std::isfinite(mass[static_cast<std::size_t>(c)]))
          ? mass[static_cast<std::size_t>(c)]
          : 0.0;
  rz::compute_rz_corner_masses_from_nodes(
      c,
      nz,
      m_cell,
      x_r.data(),
      x_z.data(),
      cell_nverts.empty() ? nullptr : cell_nverts.data(),
      m_corner,
      nullptr,
      corner_mass_convention);
}

void csr_remap_energy_audit_physical_cell_nodes(int* const nodes,
                                                const int c,
                                                const int nz) {
  const int i = (nz > 0) ? c / nz : 0;
  const int j = (nz > 0) ? c - i * nz : 0;
  const int stride = nz + 1;
  nodes[0] = i * stride + j;
  nodes[1] = (i + 1) * stride + j;
  nodes[2] = (i + 1) * stride + (j + 1);
  nodes[3] = i * stride + (j + 1);
}

long double csr_remap_energy_audit_floor_sum(
    const int c,
    const std::vector<double>& zbar,
    const double gamma,
    const double A,
    const double te_floor,
    const double ti_floor) {
  const long double z =
      zbar.empty() ? 1.0L
                   : static_cast<long double>(
                         std::fmax(zbar[static_cast<std::size_t>(c)], 0.0));
  const long double A_safe = std::fmax(A, 1.0e-30);
  const long double gm1 = std::fmax(gamma - 1.0, 1.0e-30);
  const long double cv_i =
      static_cast<long double>(tenryu::core::constants::eV_to_erg) /
      (A_safe * static_cast<long double>(tenryu::core::constants::proton_mass) *
       gm1);
  const long double cv_e = z * cv_i;
  return cv_e * std::fmax(te_floor, 0.0) +
         cv_i * std::fmax(ti_floor, 0.0);
}

void csr_remap_energy_audit_emit(
    CsrRemapEnergyAuditState& audit,
    const core::State& state,
    const core::Config& cfg,
    const double gamma,
    const double A,
    const double* const d_optionb_corner_mass,
    const double* const d_optionb_node_mass,
    const bool optionb_velocity_authority,
    const bool physical_ke_remap) {
  if (!audit.active) {
    return;
  }
  // 1T convention stores the total internal energy in ee with ei == 0; flooring
  // ei at cv_i*Ti_floor inside the remap fabricates unledgered energy (measured
  // +5e-5/pass via the next step's 1T fold-back). Zero the ion floor in 1T.
  const double ti_floor_remap =
      cfg.main.two_temperature ? cfg.numerics.floors.Ti : 0.0;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (n_cells <= 0 || n_nodes <= 0 ||
      audit.total_energy_recover.size() != static_cast<std::size_t>(n_cells)) {
    return;
  }

  std::vector<double> mass;
  std::vector<double> ee;
  std::vector<double> ei;
  std::vector<double> x_r;
  std::vector<double> x_z;
  std::vector<double> v_r;
  std::vector<double> v_z;
  std::vector<double> zbar;
  std::vector<int> cell_node_offsets;
  std::vector<int> cell_node_indices;
  std::vector<double> optionb_corner_mass;
  state.mass.copy_to_host(mass);
  state.ee.copy_to_host(ee);
  state.ei.copy_to_host(ei);
  state.x_r.copy_to_host(x_r);
  state.x_z.copy_to_host(x_z);
  state.v_r.copy_to_host(v_r);
  state.v_z.copy_to_host(v_z);
  if (!state.zbar.empty()) {
    state.zbar.copy_to_host(zbar);
  }
  state.mesh.multiblock_cell_node_csr_offsets.copy_to_host(cell_node_offsets);
  state.mesh.multiblock_cell_node_csr_indices.copy_to_host(cell_node_indices);
  if (optionb_velocity_authority && d_optionb_corner_mass != nullptr) {
    copy_device_pointer_to_host(d_optionb_corner_mass,
                                n_cells * 4,
                                optionb_corner_mass);
  }

  std::vector<long double> corner_node_mass(
      static_cast<std::size_t>(n_nodes), 0.0L);
  const int nz = state.mesh.topo.nz;
  long double K_rec = 0.0L;
  long double U_raw = 0.0L;
  long double U_stored = 0.0L;
  long double dEfloor = 0.0L;
  long double macro_K_rec = 0.0L;
  long double macro_U_raw = 0.0L;
  long double macro_U_stored = 0.0L;
  long double macro_dEfloor = 0.0L;
  for (int c = 0; c < n_cells; ++c) {
    const bool inactive = remap_energy_audit_inactive(audit.inactive_mask, c);
    double m_corner[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    if (optionb_velocity_authority &&
        optionb_corner_mass.size() == static_cast<std::size_t>(n_cells) * 4U) {
      const int base = 4 * c;
      for (int k = 0; k < 4; ++k) {
        m_corner[k] = optionb_corner_mass[static_cast<std::size_t>(base + k)];
      }
    } else if (physical_ke_remap) {
      csr_remap_energy_audit_physical_corner_masses_host(
          m_corner, c, nz, mass, x_r, x_z, state.mesh.cell_nverts,
          static_cast<int>(cfg.numerics.hydro.corner_mass_convention));
    } else {
      csr_remap_energy_audit_corner_masses_host(m_corner,
                                                c,
                                                mass,
                                                x_r,
                                                x_z,
                                                cell_node_offsets,
                                                cell_node_indices,
                                                state.mesh.cell_nverts,
                                                corner_mass_lagrangian_invariant_enabled(cfg),
                                                mesh::mesh_topo_polar_tier_family(cfg.mesh));
    }
    const int active_nverts =
        tenryu::mesh::mesh_topo_cell_active_nverts(
            state.mesh.cell_nverts.empty() ? nullptr : state.mesh.cell_nverts.data(),
            c);
    const int off = cell_node_offsets[static_cast<std::size_t>(c)];
    int physical_nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    if (physical_ke_remap) {
      csr_remap_energy_audit_physical_cell_nodes(physical_nodes, c, nz);
    }
    long double K_c = 0.0L;
    for (int k = 0; k < active_nverts; ++k) {
      const int n = physical_ke_remap
                        ? physical_nodes[k]
                        : cell_node_indices[static_cast<std::size_t>(off + k)];
      if (n < 0 || n >= n_nodes) {
        continue;
      }
      const long double cm =
          (m_corner[k] > 0.0 && std::isfinite(m_corner[k]))
              ? static_cast<long double>(m_corner[k])
              : 0.0L;
      corner_node_mass[static_cast<std::size_t>(n)] += cm;
      const long double vr = static_cast<long double>(
          v_r[static_cast<std::size_t>(n)]);
      const long double vz = static_cast<long double>(
          v_z[static_cast<std::size_t>(n)]);
      if (cm > 0.0L && std::isfinite(vr) && std::isfinite(vz)) {
        K_c += 0.5L * cm * (vr * vr + vz * vz);
      }
    }
    const long double E_c =
        static_cast<long double>(audit.total_energy_recover[static_cast<std::size_t>(c)]);
    const long double U_raw_c = E_c - K_c;
    const long double m = static_cast<long double>(mass[static_cast<std::size_t>(c)]);
    const long double U_stored_c =
        m * (static_cast<long double>(ee[static_cast<std::size_t>(c)]) +
             static_cast<long double>(ei[static_cast<std::size_t>(c)]));
    const long double e_floor =
        csr_remap_energy_audit_floor_sum(c,
                                         zbar,
                                         gamma,
                                         A,
                                         cfg.numerics.floors.Te,
                                         ti_floor_remap);
    const long double e_raw_c =
        (m > 0.0L && std::isfinite(m)) ? (U_raw_c / m) : 0.0L;
    long double dEfloor_c = 0.0L;
    if (!std::isfinite(e_raw_c)) {
      dEfloor_c = (m > 0.0L && std::isfinite(m)) ? m * e_floor : 0.0L;
    } else if (m > 0.0L && e_floor > e_raw_c) {
      dEfloor_c = m * (e_floor - e_raw_c);
    }
    K_rec += K_c;
    U_raw += U_raw_c;
    U_stored += U_stored_c;
    dEfloor += dEfloor_c;
    if (inactive) {
      macro_K_rec += K_c;
      macro_U_raw += U_raw_c;
      macro_U_stored += U_stored_c;
      macro_dEfloor += dEfloor_c;
    }
  }

  std::vector<double> audit_node_mass;
  if (!physical_ke_remap && optionb_velocity_authority &&
      d_optionb_node_mass != nullptr) {
    copy_device_pointer_to_host(d_optionb_node_mass, n_nodes, audit_node_mass);
  } else {
    audit_node_mass.assign(static_cast<std::size_t>(n_nodes), 0.0);
    for (int n = 0; n < n_nodes; ++n) {
      audit_node_mass[static_cast<std::size_t>(n)] =
          static_cast<double>(corner_node_mass[static_cast<std::size_t>(n)]);
    }
  }

  long double K_audit = 0.0L;
  long double max_mass_mismatch = 0.0L;
  for (int n = 0; n < n_nodes; ++n) {
    const long double M_a =
        static_cast<long double>(audit_node_mass[static_cast<std::size_t>(n)]);
    const long double gathered =
        corner_node_mass[static_cast<std::size_t>(n)];
    max_mass_mismatch =
        std::max(max_mass_mismatch, std::fabs(M_a - gathered));
    const long double vr =
        static_cast<long double>(v_r[static_cast<std::size_t>(n)]);
    const long double vz =
        static_cast<long double>(v_z[static_cast<std::size_t>(n)]);
    if (M_a > 0.0L && std::isfinite(M_a) && std::isfinite(vr) &&
        std::isfinite(vz)) {
      K_audit += 0.5L * M_a * (vr * vr + vz * vz);
    }
  }

  const long double E_raw = U_raw + K_rec;
  const long double E_ref = remap_energy_audit_reference_energy(audit.E0_pre);
  std::ostringstream oss;
  oss << "[remap_energy_audit]"
      << " step=" << audit.step
      << " remap=" << audit.remap
      << " optionb=" << (optionb_velocity_authority ? 1 : 0)
      << " optionb_ke=" << (optionb_velocity_authority &&
                                    d_optionb_corner_mass != nullptr
                                ? 1
                                : 0)
      << " physical_ke=" << (physical_ke_remap ? 1 : 0)
      << " E_ref=" << remap_energy_audit_format(E_ref)
      << " E0_pre=" << remap_energy_audit_format(audit.E0_pre)
      << " E_swept=" << remap_energy_audit_format(audit.E_swept)
      << " E_T_R=" << remap_energy_audit_format(audit.E_T_R)
      << " K_rec=" << remap_energy_audit_format(K_rec)
      << " U_raw=" << remap_energy_audit_format(U_raw)
      << " E_raw=" << remap_energy_audit_format(E_raw)
      << " U_stored=" << remap_energy_audit_format(U_stored)
      << " K_audit=" << remap_energy_audit_format(K_audit)
      << " dEfloor=" << remap_energy_audit_format(dEfloor)
      << " maxMassMismatch=" << remap_energy_audit_format(max_mass_mismatch)
      << " dE_swept_rel="
      << remap_energy_audit_format(
             remap_energy_audit_delta(audit.E_swept - audit.E0_pre, E_ref))
      << " dE_transport_rel="
      << remap_energy_audit_format(
             remap_energy_audit_delta(audit.E_T_R - audit.E0_pre, E_ref))
      << " dE_finish_rel="
      << remap_energy_audit_format(
             remap_energy_audit_delta(audit.E_T_R - audit.E_swept, E_ref))
      << " dE_algebra_rel="
      << remap_energy_audit_format(
             remap_energy_audit_delta(E_raw - audit.E_T_R, E_ref))
      << " dK_cell_node_rel="
      << remap_energy_audit_format(
             remap_energy_audit_delta(K_rec - K_audit, E_ref))
      << " dU_store_rel="
      << remap_energy_audit_format(
             remap_energy_audit_delta(U_stored - U_raw, E_ref))
      << " macro_cells=" << audit.macro_cells
      << " macro_E0_pre=" << remap_energy_audit_format(audit.macro_E0_pre)
      << " macro_E_swept=" << remap_energy_audit_format(audit.macro_E_swept)
      << " macro_E_T_R=" << remap_energy_audit_format(audit.macro_E_T_R)
      << " macro_K_rec=" << remap_energy_audit_format(macro_K_rec)
      << " macro_U_raw=" << remap_energy_audit_format(macro_U_raw)
      << " macro_U_stored=" << remap_energy_audit_format(macro_U_stored)
      << " macro_dEfloor=" << remap_energy_audit_format(macro_dEfloor);
  core::log_info(oss.str());
}

void check_central_macro_remap_flux_audit(
    const core::State& state,
    const std::vector<double>& audit) {
  if (audit.size() < static_cast<std::size_t>(kCentralMacroRemapAuditCount)) {
    return;
  }
  const auto& pc = state.central_pseudo_core;
  const double mass_scale = std::max(std::fabs(pc.M_c), 1.0);
  const double energy_scale =
      std::max(std::fabs(pc.Ue_c) + std::fabs(pc.Ui_c), 1.0);
  const double tracer_scale = std::max(std::fabs(pc.M_Y_c), 1.0);
  const double tol = 1.0e-12;
  TENRYU_ASSERT(std::fabs(audit[kCentralMacroRemapAuditMass]) <=
                    tol * mass_scale,
                "central macro remap applied nonzero inactive-face mass flux");
  TENRYU_ASSERT(std::fabs(audit[kCentralMacroRemapAuditEnergy]) <=
                    tol * energy_scale,
                "central macro remap applied nonzero inactive-face energy flux");
  TENRYU_ASSERT(std::fabs(audit[kCentralMacroRemapAuditTracer]) <=
                    tol * tracer_scale,
                "central macro remap applied nonzero inactive-face tracer flux");
  if (central_macro_remap_audit_env_enabled()) {
    std::ostringstream oss;
    oss << "[central_macro_remap_audit] step=" << state.step
        << " remap=" << (state.ale_remaps_applied + 1)
        << " inactive_face_skips="
        << format_ale_velcoherence_value(
               audit[kCentralMacroRemapAuditSkippedFaces])
        << " applied_mass_flux="
        << format_ale_velcoherence_value(audit[kCentralMacroRemapAuditMass])
        << " applied_energy_flux="
        << format_ale_velcoherence_value(audit[kCentralMacroRemapAuditEnergy])
        << " applied_tracer_flux="
        << format_ale_velcoherence_value(audit[kCentralMacroRemapAuditTracer]);
    core::log_info(oss.str());
  }
}

constexpr int kCsrNearVacuumTopFaces = 4;

struct alignas(8) CsrNearVacuumFaceRecord {
  int face_id = -1;
  int boundary = 0;
  int local_face = -1;
  int neighbor_cell = -1;
  int donor_cell = -1;
  double signed_volume_to_cell = 0.0;
  double rho_face = 0.0;
  double dm_to_cell = 0.0;
  double abs_dm = -1.0;
};

struct alignas(8) CsrNearVacuumRecord {
  int found = 0;
  int cell = -1;
  int block_id = -1;
  int index0 = -1;
  int index1 = -1;
  int active_nverts = 0;
  int node_id[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  double centroid_r = 0.0;
  double centroid_z = 0.0;
  double centroid_rr = 0.0;
  double V_lag = 0.0;
  double V_ref = 0.0;
  double V_ref_over_lag = 0.0;
  double mass_floor = 0.0;
  double mass_lag = 0.0;
  double rho_lag = 0.0;
  double total_energy_lag = 0.0;
  double mom_r_lag = 0.0;
  double mom_z_lag = 0.0;
  double v_r_cell_lag = 0.0;
  double v_z_cell_lag = 0.0;
  double m_raw = 0.0;
  double rho_raw = 0.0;
  double total_energy_remapped = 0.0;
  double mom_r_raw = 0.0;
  double mom_z_raw = 0.0;
  double v_r_cell_raw = 0.0;
  double v_z_cell_raw = 0.0;
  double recovered_internal = 0.0;
  double internal_from_remapped_total = 0.0;
  double post_projection_ke = 0.0;
  double node_r[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  double node_z[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  double node_vr_pre[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  double node_vz_pre[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  double node_vr_post[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  double node_vz_post[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  double mass_flux_sum = 0.0;
  CsrNearVacuumFaceRecord top_faces[kCsrNearVacuumTopFaces];
};

__device__ inline void csr_near_vacuum_clear_face(
    CsrNearVacuumFaceRecord* const face) {
  face->face_id = -1;
  face->boundary = 0;
  face->local_face = -1;
  face->neighbor_cell = -1;
  face->donor_cell = -1;
  face->signed_volume_to_cell = 0.0;
  face->rho_face = 0.0;
  face->dm_to_cell = 0.0;
  face->abs_dm = -1.0;
}

__device__ inline void csr_near_vacuum_copy_face(
    CsrNearVacuumFaceRecord* const dst,
    const CsrNearVacuumFaceRecord& src) {
  dst->face_id = src.face_id;
  dst->boundary = src.boundary;
  dst->local_face = src.local_face;
  dst->neighbor_cell = src.neighbor_cell;
  dst->donor_cell = src.donor_cell;
  dst->signed_volume_to_cell = src.signed_volume_to_cell;
  dst->rho_face = src.rho_face;
  dst->dm_to_cell = src.dm_to_cell;
  dst->abs_dm = src.abs_dm;
}

__device__ inline void csr_near_vacuum_clear_record(
    CsrNearVacuumRecord* const record) {
  record->found = 0;
  record->cell = -1;
  record->block_id = -1;
  record->index0 = -1;
  record->index1 = -1;
  record->active_nverts = 0;
  record->centroid_r = 0.0;
  record->centroid_z = 0.0;
  record->centroid_rr = 0.0;
  record->V_lag = 0.0;
  record->V_ref = 0.0;
  record->V_ref_over_lag = 0.0;
  record->mass_floor = 0.0;
  record->mass_lag = 0.0;
  record->rho_lag = 0.0;
  record->total_energy_lag = 0.0;
  record->mom_r_lag = 0.0;
  record->mom_z_lag = 0.0;
  record->v_r_cell_lag = 0.0;
  record->v_z_cell_lag = 0.0;
  record->m_raw = 0.0;
  record->rho_raw = 0.0;
  record->total_energy_remapped = 0.0;
  record->mom_r_raw = 0.0;
  record->mom_z_raw = 0.0;
  record->v_r_cell_raw = 0.0;
  record->v_z_cell_raw = 0.0;
  record->recovered_internal = 0.0;
  record->internal_from_remapped_total = 0.0;
  record->post_projection_ke = 0.0;
  record->mass_flux_sum = 0.0;
  for (int k = 0; k < mesh::kMeshTopoCellStorageSlotsMax; ++k) {
    record->node_id[k] = -1;
    record->node_r[k] = 0.0;
    record->node_z[k] = 0.0;
    record->node_vr_pre[k] = 0.0;
    record->node_vz_pre[k] = 0.0;
    record->node_vr_post[k] = 0.0;
    record->node_vz_post[k] = 0.0;
  }
  for (int k = 0; k < kCsrNearVacuumTopFaces; ++k) {
    csr_near_vacuum_clear_face(&record->top_faces[k]);
  }
}

__device__ inline void csr_near_vacuum_classify_cell(
    const int cell,
    const int n_c,
    const int n_b,
    int* const block_id,
    int* const index0,
    int* const index1) {
  const int ntheta = 4 * n_c;
  const int core_cells = 4 * n_c * n_c;
  const int north_cells = n_b * n_c;
  const int east_cells = n_b * (2 * n_c);
  const int bridge_cells = 4 * n_c * n_b;
  if (cell < core_cells) {
    *block_id = 0;
    *index0 = (ntheta > 0) ? (cell / ntheta) : -1;
    *index1 = (ntheta > 0) ? (cell - (*index0) * ntheta) : -1;
    return;
  }
  if (cell < core_cells + north_cells) {
    const int rem = cell - core_cells;
    *block_id = 1;
    *index0 = (n_c > 0) ? (rem / n_c) : -1;
    *index1 = (n_c > 0) ? (rem - (*index0) * n_c) : -1;
    return;
  }
  if (cell < core_cells + north_cells + east_cells) {
    const int rem = cell - core_cells - north_cells;
    const int n_j = 2 * n_c;
    *block_id = 2;
    *index0 = (n_j > 0) ? (rem / n_j) : -1;
    *index1 = (n_j > 0) ? (rem - (*index0) * n_j) : -1;
    return;
  }
  if (cell < core_cells + bridge_cells) {
    const int rem = cell - core_cells - north_cells - east_cells;
    *block_id = 3;
    *index0 = (n_c > 0) ? (rem / n_c) : -1;
    *index1 = (n_c > 0) ? (rem - (*index0) * n_c) : -1;
    return;
  }
  const int rem = cell - core_cells - bridge_cells;
  *block_id = 4;
  *index0 = (ntheta > 0) ? (rem / ntheta) : -1;
  *index1 = (ntheta > 0) ? (rem - (*index0) * ntheta) : -1;
}

__device__ inline void csr_near_vacuum_polygon_centroid(
    const double* const r,
    const double* const z,
    const int nverts,
    double* const centroid_r,
    double* const centroid_z) {
  double cross_sum = 0.0;
  double cr_sum = 0.0;
  double cz_sum = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp1 = (k + 1) % nverts;
    const double cross = r[k] * z[kp1] - r[kp1] * z[k];
    cross_sum += cross;
    cr_sum += (r[k] + r[kp1]) * cross;
    cz_sum += (z[k] + z[kp1]) * cross;
  }
  if (fabs(cross_sum) > 0.0 && isfinite(cross_sum)) {
    const double inv = 1.0 / (3.0 * cross_sum);
    *centroid_r = cr_sum * inv;
    *centroid_z = cz_sum * inv;
    return;
  }
  double r_avg = 0.0;
  double z_avg = 0.0;
  for (int k = 0; k < nverts; ++k) {
    r_avg += r[k];
    z_avg += z[k];
  }
  const double inv = (nverts > 0) ? (1.0 / static_cast<double>(nverts)) : 0.0;
  *centroid_r = r_avg * inv;
  *centroid_z = z_avg * inv;
}

__device__ inline void csr_near_vacuum_insert_face(
    CsrNearVacuumRecord* const record,
    const CsrNearVacuumFaceRecord candidate) {
  if (candidate.face_id < 0 || !(candidate.abs_dm >= 0.0) ||
      !isfinite(candidate.abs_dm)) {
    return;
  }
  for (int slot = 0; slot < kCsrNearVacuumTopFaces; ++slot) {
    if (candidate.abs_dm > record->top_faces[slot].abs_dm) {
      for (int dst = kCsrNearVacuumTopFaces - 1; dst > slot; --dst) {
        csr_near_vacuum_copy_face(&record->top_faces[dst],
                                  record->top_faces[dst - 1]);
      }
      csr_near_vacuum_copy_face(&record->top_faces[slot], candidate);
      return;
    }
  }
}

__global__ void csr_find_near_vacuum_cell_kernel(
    const double* __restrict__ mass_raw,
    const double* __restrict__ vol_ref,
    int* __restrict__ first_cell,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells,
    const double rho_floor) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    return;
  }
  const double V = fmax(vol_ref[c], kTinyVolume);
  const double m_floor = rho_floor * V;
  const double m = mass_raw[c];
  if (!(m > m_floor) || !isfinite(m)) {
    atomicMin(first_cell, c);
  }
}

constexpr int kI1BSpuriousSensorBlockSize = 256;

struct I1BSpuriousSensorBlockStats {
  int sampled_cells = 0;
  double residual_ke_sum = 0.0;
  double residual_ke_max = 0.0;
};

__host__ __device__ inline I1BSpuriousSensorCell empty_sensor_cell() {
  I1BSpuriousSensorCell cell{};
  cell.rank = -1;
  cell.cell_id = -1;
  cell.active_nverts = 0;
  cell.residual_ke = 0.0;
  cell.affine_ke = 0.0;
  cell.eta2 = 0.0;
  cell.score = -1.0;
  return cell;
}

__host__ __device__ inline void insert_sensor_top(
    I1BSpuriousSensorCell* const top,
    const int top_k,
    const I1BSpuriousSensorCell candidate) {
  if (top == nullptr || top_k <= 0 || candidate.cell_id < 0 ||
      !(candidate.score > 0.0) || !isfinite(candidate.score)) {
    return;
  }
  for (int slot = 0; slot < top_k; ++slot) {
    if (candidate.score > top[slot].score) {
      for (int dst = top_k - 1; dst > slot; --dst) {
        top[dst] = top[dst - 1];
      }
      top[slot] = candidate;
      return;
    }
  }
}

__global__ void ale_cell_kinetic_from_velocity_kernel(
    double* __restrict__ kinetic,
    const double* __restrict__ mass,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const int n_cells,
    const int nz,
    const int button_outer_node_ring) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (button_outer_node_ring > 0 && c != 0) {
    const int i = (nz > 0) ? (c / nz) : 0;
    if (i >= 0 && i < button_outer_node_ring) {
      kinetic[c] = 0.0;
      return;
    }
  }
  const double m = fmax(mass[c], 0.0);
  const double vr = v_r_cell[c];
  const double vz = v_z_cell[c];
  kinetic[c] = (m > 0.0 && isfinite(vr) && isfinite(vz))
                   ? 0.5 * m * (vr * vr + vz * vz)
                   : 0.0;
}

__global__ void ale_structured_corner_kinetic_total_kernel(
    double* __restrict__ kinetic_total,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int nr,
    const int nz,
    const int button_outer_node_ring) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = (nz > 0) ? (c / nz) : 0;
  const int j = (nz > 0) ? (c - i * nz) : 0;
  if (button_outer_node_ring > 0 && c != 0 && i < button_outer_node_ring) {
    kinetic_total[c] = 0.0;
    return;
  }
  kinetic_total[c] =
      (button_outer_node_ring > 0 && c == 0)
          ? rz_button_kinetic_for_cell(
                mass, x_r, x_z, v_r_node, v_z_node, button_outer_node_ring, nz)
          : rz_corner_kinetic_for_cell(mass, x_r, v_r_node, v_z_node, c, i, j, nz);
}

__global__ void ale_ke_projection_sensor_reduce_kernel(
    I1BSpuriousSensorCell* __restrict__ block_top,
    I1BSpuriousSensorBlockStats* __restrict__ block_stats,
    const double* __restrict__ kinetic_before,
    const double* __restrict__ kinetic_after,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells,
    const int top_k) {
  __shared__ I1BSpuriousSensorCell entries[kI1BSpuriousSensorBlockSize];
  const int tid = threadIdx.x;
  const int c = blockIdx.x * blockDim.x + tid;
  I1BSpuriousSensorCell entry = empty_sensor_cell();
  if (c < n_cells) {
    const double before = kinetic_before[c];
    const double after = kinetic_after[c];
    if (isfinite(before) && isfinite(after)) {
      const double delta = after - before;
      entry.cell_id = c;
      entry.active_nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
      entry.residual_ke = delta;
      entry.affine_ke = before;
      entry.eta2 = delta / (fabs(before) + 1.0e-300);
      entry.score = fmax(delta, 0.0);
    }
  }
  entries[tid] = entry;
  __syncthreads();

  if (tid == 0) {
    I1BSpuriousSensorCell local_top[kI1BSpuriousSensorTopKMax];
    for (int k = 0; k < kI1BSpuriousSensorTopKMax; ++k) {
      local_top[k] = empty_sensor_cell();
    }
    I1BSpuriousSensorBlockStats stats{};
    for (int t = 0; t < blockDim.x; ++t) {
      const I1BSpuriousSensorCell candidate = entries[t];
      if (candidate.cell_id < 0) {
        continue;
      }
      ++stats.sampled_cells;
      stats.residual_ke_sum += candidate.residual_ke;
      stats.residual_ke_max =
          fmax(stats.residual_ke_max, candidate.residual_ke);
      insert_sensor_top(local_top, top_k, candidate);
    }
    const int base = blockIdx.x * kI1BSpuriousSensorTopKMax;
    for (int k = 0; k < kI1BSpuriousSensorTopKMax; ++k) {
      block_top[base + k] = local_top[k];
    }
    block_stats[blockIdx.x] = stats;
  }
}

I1BSpuriousSensorSummary reduce_ale_ke_projection_sensor(
    const double* const kinetic_before,
    const double* const kinetic_after,
    const std::uint8_t* const d_cell_nverts,
    const int n_cells,
    const int top_k) {
  I1BSpuriousSensorSummary out{};
  if (kinetic_before == nullptr || kinetic_after == nullptr || n_cells <= 0) {
    return out;
  }
  const int clamped_top_k =
      std::clamp(top_k, 1, kI1BSpuriousSensorTopKMax);
  const int blocks =
      (n_cells + kI1BSpuriousSensorBlockSize - 1) /
      kI1BSpuriousSensorBlockSize;
  core::DeviceArray<I1BSpuriousSensorCell> d_block_top("ale_remap:reduce_ale_ke_projection_sensor:d_block_top");
  core::DeviceArray<I1BSpuriousSensorBlockStats> d_block_stats("ale_remap:reduce_ale_ke_projection_sensor:d_block_stats");
  d_block_top.reset(static_cast<std::size_t>(blocks) *
                    static_cast<std::size_t>(kI1BSpuriousSensorTopKMax));
  d_block_stats.reset(static_cast<std::size_t>(blocks));

  ale_ke_projection_sensor_reduce_kernel<<<blocks, kI1BSpuriousSensorBlockSize>>>(
      d_block_top.data(),
      d_block_stats.data(),
      kinetic_before,
      kinetic_after,
      d_cell_nverts,
      n_cells,
      clamped_top_k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(core::debug_kernel_sync());

  std::vector<I1BSpuriousSensorBlockStats> stats;
  std::vector<I1BSpuriousSensorCell> block_top;
  d_block_stats.copy_to_host(stats);
  d_block_top.copy_to_host(block_top);

  out.valid = true;
  for (const auto& block : stats) {
    out.sampled_cells += block.sampled_cells;
    out.residual_ke_sum += block.residual_ke_sum;
    out.residual_ke_max = std::max(out.residual_ke_max, block.residual_ke_max);
  }
  for (const I1BSpuriousSensorCell& candidate : block_top) {
    insert_sensor_top(out.top.data(), clamped_top_k, candidate);
  }
  out.top_count = 0;
  for (int k = 0; k < clamped_top_k; ++k) {
    if (out.top[static_cast<std::size_t>(k)].cell_id >= 0) {
      ++out.top_count;
    }
  }
  return out;
}

void memset_zero(double* ptr, const std::size_t count) {
  if (ptr != nullptr && count > 0U) {
    CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(double)));
  }
}

void memset_zero_int(int* ptr, const std::size_t count) {
  if (ptr != nullptr && count > 0U) {
    CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(int)));
  }
}

bool has_tri_cells(const core::State& state) {
  const auto& cell_nverts = state.mesh.cell_nverts;
  return cell_nverts.size() == state.mass.size() &&
         std::any_of(cell_nverts.begin(), cell_nverts.end(),
                     [](const std::uint8_t nverts) { return nverts != 4U; });
}

bool has_node_flag_constraints(const core::State& state) {
  const auto& node_flags = state.mesh.topo.node_flags;
  return node_flags.size() == state.x_r.size() &&
         std::any_of(node_flags.begin(), node_flags.end(), [](const std::uint8_t flags) {
           return (flags & (pole_axis::kNodeCenterFlag |
                            pole_axis::kNodePoleAxisFlag)) != 0U;
         });
}

std::uint8_t* upload_cell_nverts_if_needed(const core::State& state,
                                           const bool upload) {
  if (!upload) {
    return nullptr;
  }
  const auto& cell_nverts = state.mesh.cell_nverts;
  std::uint8_t* d_cell_nverts = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cell_nverts),
                        cell_nverts.size() * sizeof(std::uint8_t)));
  CUDA_CHECK(cudaMemcpy(d_cell_nverts,
                        cell_nverts.data(),
                        cell_nverts.size() * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice));
  return d_cell_nverts;
}

std::uint8_t* upload_node_flags_if_needed(const core::State& state,
                                          const bool upload) {
  if (!upload) {
    return nullptr;
  }
  const auto& node_flags = state.mesh.topo.node_flags;
  TENRYU_ASSERT(node_flags.size() == state.x_r.size(),
                "2D RZ conservative remap requires node_flags size == n_nodes");
  std::uint8_t* d_node_flags = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_node_flags),
                        node_flags.size() * sizeof(std::uint8_t)));
  CUDA_CHECK(cudaMemcpy(d_node_flags,
                        node_flags.data(),
                        node_flags.size() * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice));
  return d_node_flags;
}

__host__ __device__ inline int csr_internal_flux_donor(
    const int cell_a,
    const int cell_b,
    const double dV_a) {
  // dV_a is gain-positive for cell_a.  Therefore positive flux comes from
  // cell_b, while negative flux comes from cell_a.
  return make_oriented_swept_volume(
             cell_a,
             cell_b,
             dV_a,
             SweptVolumeConvention::OrientedLowToHighV1)
      .donor;
}

__host__ __device__ inline int csr_internal_flux_losing_cell(
    const int cell_a,
    const int cell_b,
    const double dV_a) {
  return (dV_a > 0.0) ? cell_b : cell_a;
}

__device__ inline double csr_face_swept_volume_outward(
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ cell_orientation_sign,
    const int cell,
    const int local_face,
    const std::uint8_t* __restrict__ cell_nverts) {
  return detail::csr_face_swept_volume_outward(x_r_old,
                                               x_z_old,
                                               x_r_new,
                                               x_z_new,
                                               cell_node_csr_offsets,
                                               cell_node_csr_indices,
                                               cell_orientation_sign,
                                               cell,
                                               local_face,
                                               cell_nverts);
}

__device__ inline double csr_face_swept_raw_moments_outward(
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ cell_orientation_sign,
    const int cell,
    const int local_face,
    const std::uint8_t* __restrict__ cell_nverts,
    double* const dMr,
    double* const dMz) {
  *dMr = 0.0;
  *dMz = 0.0;
  int na = -1;
  int nb = -1;
  if (!detail::csr_face_swept_node_indices(cell_node_csr_offsets,
                                           cell_node_csr_indices,
                                           cell_nverts,
                                           cell,
                                           local_face,
                                           &na,
                                           &nb)) {
    return 0.0;
  }
  if (x_r_old[na] == x_r_new[na] && x_z_old[na] == x_z_new[na] &&
      x_r_old[nb] == x_r_new[nb] && x_z_old[nb] == x_z_new[nb]) {
    return 0.0;
  }

  const double r0 = x_r_old[na];
  const double z0 = x_z_old[na];
  const double r1 = x_r_old[nb];
  const double z1 = x_z_old[nb];
  const double r2 = x_r_new[nb];
  const double z2 = x_z_new[nb];
  const double r3 = x_r_new[na];
  const double z3 = x_z_new[na];

  // Q=(a_old,b_old,b_new,a_new) is clockwise when an outward-moving face
  // enlarges a positively oriented cell.  The common -orientation factor
  // therefore makes dV, dMr, and dMz positive for volume gained by cell.
  // Applying the same factor to all three preserves the swept centroid.
  const double gain_sign =
      -detail::csr_cell_orientation_sign(cell, cell_orientation_sign);
  *dMr = gain_sign *
         detail::rz_signed_quad_moment_r(r0, z0, r1, z1, r2, z2, r3, z3);
  *dMz = gain_sign *
         detail::rz_signed_quad_moment_z(r0, z0, r1, z1, r2, z2, r3, z3);
  return gain_sign *
         detail::rz_signed_quad_volume(r0, z0, r1, z1, r2, z2, r3, z3);
}

__device__ inline double csr_clamped_flux_scale(
    const double* __restrict__ mass_flux_scale,
    const int losing_cell) {
  if (mass_flux_scale == nullptr) {
    return 1.0;
  }
  double scale = mass_flux_scale[losing_cell];
  if (!isfinite(scale)) {
    scale = 0.0;
  }
  return fmin(1.0, fmax(0.0, scale));
}

__device__ inline double csr_barth_jespersen_face_alpha(
    const double* __restrict__ field,
    const double q_face,
    const int cell,
    const int n_cells,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const RemapDispatchAuditDeviceView remap_dispatch_audit) {
  const double q_c = field[cell];
  double q_min = q_c;
  double q_max = q_c;
  const int off = face_adj_csr_offsets[cell];
  const int end = face_adj_csr_offsets[cell + 1];
  const int active_nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, cell);
  for (int p = off; p < end; ++p) {
    const int local = p - off;
    if (!tenryu::mesh::mesh_topo_local_face_is_active(active_nverts, local)) {
      continue;
    }
    const int neighbor = face_adj_csr_indices[p];
    if (neighbor < 0 || neighbor >= n_cells ||
        csr_inactive_cell(inactive_cell_mask, neighbor)) {
      continue;
    }
    q_min = fmin(q_min, field[neighbor]);
    q_max = fmax(q_max, field[neighbor]);
  }

  if (!isfinite(q_face)) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::ReconstructionNonfiniteFallback,
        cell);
    return 0.0;
  }
  const double alpha = detail::csr_barth_jespersen_limiter_ratio(
      q_c, q_min, q_max, q_face);
  if (alpha < 1.0) {
    remap_dispatch_audit_count(remap_dispatch_audit,
                               RemapDispatchAuditCounter::LimiterActivation,
                               cell);
  }
  return alpha;
}

__global__ void csr_compute_old_rz_volume_centroids_kernel(
    double* __restrict__ centroid_r,
    double* __restrict__ centroid_z,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells,
    const RemapDispatchAuditDeviceView remap_dispatch_audit) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    centroid_r[c] = 0.0;
    centroid_z[c] = 0.0;
    return;
  }

  const int active_nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  const int off = cell_node_csr_offsets[c];
  double r[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  double z[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  for (int k = 0; k < active_nverts; ++k) {
    const int n = cell_node_csr_indices[off + k];
    r[k] = x_r_old[n];
    z[k] = x_z_old[n];
  }

  const mesh::moments::PolyRZMoments moments =
      mesh::moments::poly_rz_moments_fan(r, z, active_nverts);
  const double orientation =
      detail::csr_cell_orientation_sign(c, cell_orientation_sign);
  const double Mr = orientation * moments.mr;
  const double Mrr = orientation * moments.mrr;
  const double Mrz = orientation * moments.mrz;
  if (Mr > 0.0 && isfinite(Mr) && isfinite(Mrr) && isfinite(Mrz)) {
    const double c_r = Mrr / Mr;
    const double c_z = Mrz / Mr;
    if (isfinite(c_r) && isfinite(c_z)) {
      centroid_r[c] = c_r;
      centroid_z[c] = c_z;
      return;
    }
  }

  mesh::moments::poly_vertex_mean(
      r, z, active_nverts, centroid_r[c], centroid_z[c]);
  remap_dispatch_audit_count(
      remap_dispatch_audit,
      RemapDispatchAuditCounter::SweptCentroidAverageFallback,
      c);
}

// DIAGNOSTIC ORACLES (consult-17 §3.3), not physics.
__global__ void p3_oracle_overwrite_cell_velocity_kernel(
    double* __restrict__ v_r_cell,
    double* __restrict__ v_z_cell,
    const double* __restrict__ centroid_r,
    const double* __restrict__ centroid_z,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells,
    const double H) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells || csr_inactive_cell(inactive_cell_mask, c)) {
    return;
  }
  v_r_cell[c] = -H * centroid_r[c];
  v_z_cell[c] = -H * centroid_z[c];
}

// DIAGNOSTIC ORACLES (consult-17 §3.3), not physics.
__global__ void p3_oracle_overwrite_node_velocity_kernel(
    double* __restrict__ v_r,
    double* __restrict__ v_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const std::uint8_t* __restrict__ active_node_mask,
    const int n_nodes,
    const double H) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes ||
      (active_node_mask != nullptr && active_node_mask[n] == 0U)) {
    return;
  }
  v_r[n] = -H * x_r[n];
  v_z[n] = -H * x_z[n];
}

// DIAGNOSTIC ORACLES (consult-17 §3.3), not physics.
__global__ void p3_oracle_overwrite_energy_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ rho,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells,
    const double pressure_e,
    const double pressure_i,
    const double gamma) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells || csr_inactive_cell(inactive_cell_mask, c)) {
    return;
  }
  const double gm1_rho = (gamma - 1.0) * rho[c];
  ee[c] = pressure_e / gm1_rho;
  ei[c] = pressure_i / gm1_rho;
}

__device__ inline double csr_moments_direct_field_integral(
    const double* __restrict__ field,
    const double* __restrict__ grad_r,
    const double* __restrict__ grad_z,
    const double* __restrict__ old_centroid_r,
    const double* __restrict__ old_centroid_z,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const double donor_volume,
    const int donor,
    const int n_cells,
    const double dV,
    const double dMr,
    const double dMz,
    const bool no_face_clip,
    const RemapDispatchAuditDeviceView remap_dispatch_audit) {
  const double q_bar = field[donor];
  if (grad_r == nullptr || grad_z == nullptr) {
    return q_bar * dV;
  }
  const double r_bar = old_centroid_r[donor];
  const double z_bar = old_centroid_z[donor];
  const double dMr_centered = dMr - r_bar * dV;
  const double dMz_centered = dMz - z_bar * dV;
  const double gradient_integral = grad_r[donor] * dMr_centered +
                                   grad_z[donor] * dMz_centered;
  double integral = q_bar * dV + gradient_integral;

  const bool nondegenerate =
      isfinite(donor_volume) && donor_volume > 0.0 &&
      fabs(dV) >= 1.0e-12 * donor_volume;
  if (!no_face_clip && nondegenerate) {
    const double r_face = dMr / dV;
    const double z_face = dMz / dV;
    const double q_face = q_bar + grad_r[donor] * (r_face - r_bar) +
                          grad_z[donor] * (z_face - z_bar);
    const double alpha = csr_barth_jespersen_face_alpha(
        field,
        q_face,
        donor,
        n_cells,
        face_adj_csr_offsets,
        face_adj_csr_indices,
        cell_nverts,
        inactive_cell_mask,
        remap_dispatch_audit);
    integral = q_bar * dV + alpha * gradient_integral;
  }
  if (!isfinite(integral)) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::ReconstructionNonfiniteFallback,
        donor);
    integral = q_bar * dV;
  }
  return integral;
}

struct CsrGclAuditDeviceView {
  double* internal_face_dV_to_cell_a = nullptr;
  double* boundary_face_dV_to_cell = nullptr;
};

template <bool GclAudit>
__device__ inline void csr_gcl_audit_store_internal_face(
    const CsrGclAuditDeviceView audit,
    const int face,
    const double dV_to_cell_a) {
  if constexpr (GclAudit) {
    audit.internal_face_dV_to_cell_a[face] =
        finite_nonzero(dV_to_cell_a) ? dV_to_cell_a : 0.0;
  }
}

template <bool GclAudit>
__device__ inline void csr_gcl_audit_store_boundary_face(
    const CsrGclAuditDeviceView audit,
    const int face,
    const double dV_to_cell) {
  if constexpr (GclAudit) {
    audit.boundary_face_dV_to_cell[face] =
        finite_nonzero(dV_to_cell) ? dV_to_cell : 0.0;
  }
}

__global__ void csr_gcl_audit_accumulate_internal_faces_kernel(
    double* __restrict__ outward_swept_sum,
    double* __restrict__ swept_abs_sum,
    const double* __restrict__ internal_face_dV_to_cell_a,
    const int* __restrict__ unique_cell_a,
    const int* __restrict__ unique_cell_b,
    const int n_faces) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const double dV_to_cell_a = internal_face_dV_to_cell_a[f];
  if (!finite_nonzero(dV_to_cell_a)) {
    return;
  }
  const int cell_a = unique_cell_a[f];
  const int cell_b = unique_cell_b[f];
  // The remap update adds dV_to_cell_a to cell A and its opposite to cell B.
  // Consult-16 defines positive swept volume as outward from the cell, hence
  // the sign reversal used by the GCL residual.
  atomicAdd(outward_swept_sum + cell_a, -dV_to_cell_a);
  atomicAdd(outward_swept_sum + cell_b, dV_to_cell_a);
  const double abs_dV = fabs(dV_to_cell_a);
  atomicAdd(swept_abs_sum + cell_a, abs_dV);
  atomicAdd(swept_abs_sum + cell_b, abs_dV);
}

__global__ void csr_gcl_audit_accumulate_boundary_faces_kernel(
    double* __restrict__ outward_swept_sum,
    double* __restrict__ swept_abs_sum,
    const double* __restrict__ boundary_face_dV_to_cell,
    const int* __restrict__ boundary_cell,
    const int n_faces) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const double dV_to_cell = boundary_face_dV_to_cell[f];
  if (!finite_nonzero(dV_to_cell)) {
    return;
  }
  const int cell = boundary_cell[f];
  atomicAdd(outward_swept_sum + cell, -dV_to_cell);
  atomicAdd(swept_abs_sum + cell, fabs(dV_to_cell));
}

__global__ void csr_gcl_audit_finalize_cells_kernel(
    double* __restrict__ residual,
    double* __restrict__ scale,
    const double* __restrict__ outward_swept_sum,
    const double* __restrict__ swept_abs_sum,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    residual[c] = 0.0;
    scale[c] = 0.0;
    return;
  }
  const int offset = cell_node_csr_offsets[c];
  const int nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  double r_old[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  double z_old[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  double r_new[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  double z_new[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  for (int k = 0; k < nverts; ++k) {
    const int node = cell_node_csr_indices[offset + k];
    r_old[k] = x_r_old[node];
    z_old[k] = x_z_old[node];
    r_new[k] = x_r_new[node];
    z_new[k] = x_z_new[node];
  }
  const double orientation = static_cast<double>(cell_orientation_sign[c]);
  const double V_L =
      orientation * rz::rz_polygon_volume_exact(r_old, z_old, nverts);
  const double V_R =
      orientation * rz::rz_polygon_volume_exact(r_new, z_new, nverts);
  residual[c] = V_R - V_L + outward_swept_sum[c];
  scale[c] = fabs(V_L) + fabs(V_R) + swept_abs_sum[c];
}

__device__ inline double csr_face_swept_moments_outward(
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ cell_orientation_sign,
    const int cell,
    const int local_face,
    const std::uint8_t* __restrict__ cell_nverts,
    double* rq,
    double* zq,
    detail::CsrFaceSweptMomentsStatus* status,
    bool* exact_moments = nullptr) {
  return detail::csr_face_swept_moments_outward(x_r_old,
                                                x_z_old,
                                                x_r_new,
                                                x_z_new,
                                                cell_node_csr_offsets,
                                                cell_node_csr_indices,
                                                cell_orientation_sign,
                                                cell,
                                                local_face,
                                                cell_nverts,
                                                rq,
                                                zq,
                                                status,
                                                exact_moments);
}

constexpr int kCsrOptionBDiagFallback = 0;
constexpr int kCsrOptionBDiagExpanded = 1;
constexpr int kCsrOptionBDiagInvalid = 2;
constexpr int kCsrOptionBDiagSkipped = 3;
constexpr int kCsrOptionBDiagFilterInvalid = 4;
constexpr int kCsrOptionBDiagFilterDegenerate = 5;
constexpr int kCsrOptionBDiagExpandedRing1 = 6;
constexpr int kCsrOptionBDiagExpandedRing2 = 7;
constexpr int kCsrOptionBDiagExpandedFailed = 8;
constexpr int kCsrOptionBDiagCentroidOut = 9;
constexpr int kCsrOptionBDiagReceiverVertexOut = 10;
constexpr int kCsrOptionBDiagCentroidOnBoundary = 11;
constexpr int kCsrOptionBDiagCentroidFar = 12;
constexpr int kCsrOptionBDiagCentroidFarPrint = 13;
constexpr int kCsrOptionBDiagCount = 14;
constexpr int kCsrOptionBDiagRealAlphaMin = 0;
constexpr int kCsrOptionBDiagRealCount = 1;
constexpr int kReplayDiscardLedgerFaces = 0;
constexpr int kReplayDiscardLedgerMass = 1;
constexpr int kReplayDiscardLedgerPiR = 2;
constexpr int kReplayDiscardLedgerPiZ = 3;
constexpr int kReplayDiscardLedgerCount = 4;
constexpr int kCsrOptionBFaceCornerSlots = 8;
constexpr int kCsrOptionBMaxExpandedCells = 32;
constexpr int kCsrOptionBMaxExpandedNodes = 64;
constexpr int kCsrOptionBMaxExpandedRing = 2;
constexpr double kCsrOptionBCentroidBoundaryTolRel = 0.25;
constexpr int kCsrOptionBRing5FaceStatusNone = 0;
constexpr int kCsrOptionBRing5FaceStatusSkippedInactive = 1;
constexpr int kCsrOptionBRing5FaceStatusSkippedZero = 2;
constexpr int kCsrOptionBRing5FaceStatusPacketOk = 3;
constexpr int kCsrOptionBRing5FaceStatusExpandedOk = 4;
constexpr int kCsrOptionBRing5FaceStatusFallback = 5;
constexpr int kCsrOptionBRing5FaceStatusInvalid = 6;

__device__ inline void csr_optionb_atomic_min_double(double* address,
                                                     const double value) {
  if (address == nullptr || !isfinite(value)) {
    return;
  }
  unsigned long long* const ull =
      reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *ull;
  while (true) {
    const double old_value = __longlong_as_double(
        static_cast<long long>(old));
    if (old_value <= value) {
      return;
    }
    const unsigned long long assumed = old;
    old = atomicCAS(ull,
                    assumed,
                    static_cast<unsigned long long>(
                        __double_as_longlong(value)));
    if (old == assumed) {
      return;
    }
  }
}

__host__ __device__ inline
    tenryu::hydro::optionb::NodeVelocityProjector
    csr_optionb_projector_from_flags(const std::uint8_t flags) {
  if ((flags & tenryu::mesh::NODE_CENTER) != 0U) {
    return tenryu::hydro::optionb::NodeVelocityProjector::
        PINNED_OR_NODE_CENTER;
  }
  if ((flags & (tenryu::mesh::NODE_AXIS | tenryu::mesh::NODE_POLE_AXIS)) !=
      0U) {
    return tenryu::hydro::optionb::NodeVelocityProjector::RZ_AXIS;
  }
  return tenryu::hydro::optionb::NodeVelocityProjector::FREE;
}

__device__ inline void csr_optionb_cell_geometry(
    double* __restrict__ r,
    double* __restrict__ z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int cell,
    const int active_nverts) {
  const int off = cell_node_csr_offsets[cell];
  for (int k = 0; k < active_nverts; ++k) {
    const int n = cell_node_csr_indices[off + k];
    r[k] = x_r[n];
    z[k] = x_z[n];
  }
  for (int k = active_nverts; k < kCsrOptionBFaceCornerSlots; ++k) {
    r[k] = 0.0;
    z[k] = 0.0;
  }
  (void)cell_nverts;
}

__device__ inline void csr_optionb_unit_corner_weights(
    const double* __restrict__ r,
    const double* __restrict__ z,
    const int nverts,
    double* __restrict__ w) {
  tenryu::hydro::optionb::first_moment_corner_masses(1.0, r, z, nverts, w);
  double sum = 0.0;
  for (int k = 0; k < nverts; ++k) {
    if (!(w[k] > 0.0) || !isfinite(w[k])) {
      w[k] = 0.0;
    }
    sum += w[k];
  }
  if (!(sum > 0.0) || !isfinite(sum)) {
    const double uniform = 1.0 / static_cast<double>(nverts);
    for (int k = 0; k < nverts; ++k) {
      w[k] = uniform;
    }
    return;
  }
  const double inv_sum = 1.0 / sum;
  for (int k = 0; k < nverts; ++k) {
    w[k] *= inv_sum;
  }
}

__device__ inline bool csr_optionb_boundary_edge_weights(
    const double* __restrict__ r,
    const double* __restrict__ z,
    const int nverts,
    const double rq,
    const double zq,
    double* __restrict__ lambda) {
  if (!tenryu::hydro::optionb::detail::valid_cell_nverts(nverts) ||
      !isfinite(rq) || !isfinite(zq)) {
    return false;
  }
  double best_dist2 = 0.0;
  int best_i0 = -1;
  int best_i1 = -1;
  double best_t = 0.0;
  double scale = tenryu::hydro::optionb::detail::polygon_length_scale(
      r, z, nverts, rq, zq);
  for (int i = 0; i < nverts; ++i) {
    if (!isfinite(r[i]) || !isfinite(z[i])) {
      return false;
    }
    const int ip = (i + 1 == nverts) ? 0 : i + 1;
    const double er = r[ip] - r[i];
    const double ez = z[ip] - z[i];
    const double len2 = er * er + ez * ez;
    if (!(len2 > 0.0) || !isfinite(len2)) {
      continue;
    }
    const double edge_len = sqrt(len2);
    scale = fmax(scale, edge_len);
    const double qr = rq - r[i];
    const double qz = zq - z[i];
    const double dot = er * qr + ez * qz;
    const double t = fmax(0.0, fmin(1.0, dot / len2));
    const double pr = r[i] + t * er;
    const double pz = z[i] + t * ez;
    const double dr = rq - pr;
    const double dz = zq - pz;
    const double dist2 = dr * dr + dz * dz;
    const double tol =
        kCsrOptionBCentroidBoundaryTolRel * edge_len +
        1.0e-12 * fmax(1.0, scale);
    const double dot_tol = tol * edge_len;
    if (dot < -dot_tol || dot > len2 + dot_tol || dist2 > tol * tol) {
      continue;
    }
    if (best_i0 < 0 || dist2 < best_dist2) {
      best_dist2 = dist2;
      best_i0 = i;
      best_i1 = ip;
      best_t = t;
    }
  }
  if (best_i0 < 0) {
    return false;
  }
  tenryu::hydro::optionb::detail::fill_edge(
      nverts, best_i0, best_i1, best_t, lambda);
  return true;
}

__device__ inline bool csr_optionb_point_near_polygon_boundary(
    const double* __restrict__ r,
    const double* __restrict__ z,
    const int nverts,
    const double rq,
    const double zq) {
  double lambda[4] = {0.0, 0.0, 0.0, 0.0};
  return csr_optionb_boundary_edge_weights(r, z, nverts, rq, zq, lambda);
}

__device__ inline int csr_optionb_centroid_out_class(
    const double* __restrict__ donor_r,
    const double* __restrict__ donor_z,
    const int donor_nverts,
    const double* __restrict__ receiver_r,
    const double* __restrict__ receiver_z,
    const int receiver_nverts,
    const double rq,
    const double zq) {
  if (tenryu::hydro::optionb::detail::point_in_convex_hull(
          donor_r, donor_z, donor_nverts, rq, zq) &&
      tenryu::hydro::optionb::detail::point_in_convex_hull(
          receiver_r, receiver_z, receiver_nverts, rq, zq)) {
    return 0;
  }
  if (csr_optionb_point_near_polygon_boundary(
          donor_r, donor_z, donor_nverts, rq, zq) ||
      csr_optionb_point_near_polygon_boundary(
          receiver_r, receiver_z, receiver_nverts, rq, zq)) {
    return 1;
  }
  return 2;
}

__device__ inline double csr_optionb_signed_hull_distance(
    const double* __restrict__ r,
    const double* __restrict__ z,
    const int nverts,
    const double rq,
    const double zq) {
  const double area2 = tenryu::hydro::rz::rz_polygon_area2_exact(r, z, nverts);
  if (!isfinite(area2) || area2 == 0.0) {
    return 0.0;
  }
  const double orient = (area2 >= 0.0) ? 1.0 : -1.0;
  double min_signed = 1.0e300;
  for (int i = 0; i < nverts; ++i) {
    const int ip = (i + 1 == nverts) ? 0 : i + 1;
    const double er = r[ip] - r[i];
    const double ez = z[ip] - z[i];
    const double len = sqrt(er * er + ez * ez);
    if (!(len > 0.0) || !isfinite(len)) {
      continue;
    }
    const double qr = rq - r[i];
    const double qz = zq - z[i];
    const double signed_edge = orient * (er * qz - ez * qr) / len;
    min_signed = fmin(min_signed, signed_edge);
  }
  return isfinite(min_signed) ? min_signed : 0.0;
}

__device__ inline void csr_optionb_print_centroid_far_packet(
    int* __restrict__ diagnostics,
    const bool print_diag,
    const int f,
    const int cell_a,
    const int cell_b,
    const int donor,
    const int receiver,
    const int local_a,
    const double dV_a,
    const double dm_q,
    const double rq,
    const double zq,
    const double* __restrict__ donor_r,
    const double* __restrict__ donor_z,
    const int donor_nverts,
    const double* __restrict__ receiver_r,
    const double* __restrict__ receiver_z,
    const int receiver_nverts,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts) {
  if (!print_diag || diagnostics == nullptr) {
    return;
  }
  const int sample =
      atomicAdd(diagnostics + kCsrOptionBDiagCentroidFarPrint, 1);
  if (sample >= 5) {
    return;
  }

  int na = -1;
  int nb = -1;
  double old_ar = 0.0;
  double old_az = 0.0;
  double old_br = 0.0;
  double old_bz = 0.0;
  double new_ar = 0.0;
  double new_az = 0.0;
  double new_br = 0.0;
  double new_bz = 0.0;
  double face_thick_a = 0.0;
  double face_thick_b = 0.0;
  double face_thick_max = 0.0;
  double packet_quad_volume = 0.0;
  if (::tenryu::hydro::ale::detail::csr_face_swept_node_indices(
          cell_node_csr_offsets,
          cell_node_csr_indices,
          cell_nverts,
          cell_a,
          local_a,
          &na,
          &nb)) {
    old_ar = x_r_old[na];
    old_az = x_z_old[na];
    old_br = x_r_old[nb];
    old_bz = x_z_old[nb];
    new_ar = x_r_new[na];
    new_az = x_z_new[na];
    new_br = x_r_new[nb];
    new_bz = x_z_new[nb];
    const double dar = new_ar - old_ar;
    const double daz = new_az - old_az;
    const double dbr = new_br - old_br;
    const double dbz = new_bz - old_bz;
    face_thick_a = sqrt(dar * dar + daz * daz);
    face_thick_b = sqrt(dbr * dbr + dbz * dbz);
    face_thick_max = fmax(face_thick_a, face_thick_b);
    packet_quad_volume = ::tenryu::hydro::ale::detail::rz_signed_quad_volume(
        old_ar, old_az, old_br, old_bz, new_br, new_bz, new_ar, new_az);
  }

  const double donor_sd =
      csr_optionb_signed_hull_distance(donor_r, donor_z, donor_nverts, rq, zq);
  const double receiver_sd = csr_optionb_signed_hull_distance(
      receiver_r, receiver_z, receiver_nverts, rq, zq);
  printf("[optionb_centroid_far] sample=%d f=%d cell_a=%d cell_b=%d "
         "donor=%d receiver=%d local_a=%d na=%d nb=%d xq_frame=swept_quad "
         "donor_frame=x_L receiver_frame=x_R dV=%+.17e dm=%+.17e "
         "xq=(%+.17e,%+.17e) donor_sd=%+.17e receiver_sd=%+.17e "
         "face_thick_max=%+.17e face_thick_a=%+.17e face_thick_b=%+.17e "
         "packet_quad_volume=%+.17e\n",
         sample,
         f,
         cell_a,
         cell_b,
         donor,
         receiver,
         local_a,
         na,
         nb,
         dV_a,
         dm_q,
         rq,
         zq,
         donor_sd,
         receiver_sd,
         face_thick_max,
         face_thick_a,
         face_thick_b,
         packet_quad_volume);
  printf("[optionb_centroid_far] sample=%d face_old_a=(%+.17e,%+.17e) "
         "face_old_b=(%+.17e,%+.17e) face_new_a=(%+.17e,%+.17e) "
         "face_new_b=(%+.17e,%+.17e)\n",
         sample,
         old_ar,
         old_az,
         old_br,
         old_bz,
         new_ar,
         new_az,
         new_br,
         new_bz);
  printf("[optionb_centroid_far] sample=%d donor_nverts=%d "
         "donor_v0=(%+.17e,%+.17e) donor_v1=(%+.17e,%+.17e) "
         "donor_v2=(%+.17e,%+.17e) donor_v3=(%+.17e,%+.17e)\n",
         sample,
         donor_nverts,
         donor_r[0],
         donor_z[0],
         donor_r[1],
         donor_z[1],
         donor_r[2],
         donor_z[2],
         donor_r[3],
         donor_z[3]);
  printf("[optionb_centroid_far] sample=%d receiver_nverts=%d "
         "receiver_v0=(%+.17e,%+.17e) receiver_v1=(%+.17e,%+.17e) "
         "receiver_v2=(%+.17e,%+.17e) receiver_v3=(%+.17e,%+.17e)\n",
         sample,
         receiver_nverts,
         receiver_r[0],
         receiver_z[0],
         receiver_r[1],
         receiver_z[1],
         receiver_r[2],
         receiver_z[2],
         receiver_r[3],
         receiver_z[3]);
}

enum class CsrOptionBPacketCentroidWeightsStatus : std::uint8_t {
  OK = 0,
  NEAR_BOUNDARY = 1,
  FAR_OUTSIDE = 2,
  INVALID_INPUT = 3,
};

__device__ inline CsrOptionBPacketCentroidWeightsStatus
csr_optionb_packet_centroid_weights(
    const double* __restrict__ r,
    const double* __restrict__ z,
    const int nverts,
    const double rq,
    const double zq,
    double* __restrict__ lambda) {
  for (int k = 0; k < 4; ++k) {
    lambda[k] = 0.0;
  }
  if (!tenryu::hydro::optionb::detail::valid_cell_nverts(nverts) ||
      !isfinite(rq) || !isfinite(zq)) {
    return CsrOptionBPacketCentroidWeightsStatus::INVALID_INPUT;
  }
  if (tenryu::hydro::optionb::detail::point_in_convex_hull(
          r, z, nverts, rq, zq)) {
    tenryu::hydro::optionb::barycentric_weights(
        r, z, nverts, rq, zq, lambda);
    if (tenryu::hydro::optionb::detail::barycentric_query_ok(
            r, z, nverts, rq, zq, lambda)) {
      return CsrOptionBPacketCentroidWeightsStatus::OK;
    }
    return CsrOptionBPacketCentroidWeightsStatus::INVALID_INPUT;
  }
  if (csr_optionb_boundary_edge_weights(r, z, nverts, rq, zq, lambda)) {
    return CsrOptionBPacketCentroidWeightsStatus::NEAR_BOUNDARY;
  }
  return CsrOptionBPacketCentroidWeightsStatus::FAR_OUTSIDE;
}

__device__ inline void csr_optionb_live_cell_mean_velocity(
    const double* __restrict__ m_corner,
    const double* __restrict__ p_r,
    const double* __restrict__ p_z,
    const int nverts,
    double* __restrict__ ur,
    double* __restrict__ uz) {
  double m_sum = 0.0;
  double pr_sum = 0.0;
  double pz_sum = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const double m =
        (m_corner[k] > 0.0 && isfinite(m_corner[k])) ? m_corner[k] : 0.0;
    m_sum += m;
    pr_sum += isfinite(p_r[k]) ? p_r[k] : 0.0;
    pz_sum += isfinite(p_z[k]) ? p_z[k] : 0.0;
  }
  if (m_sum > 0.0 && isfinite(m_sum)) {
    *ur = pr_sum / m_sum;
    *uz = pz_sum / m_sum;
  } else {
    *ur = 0.0;
    *uz = 0.0;
  }
}

__device__ inline void csr_optionb_apply_first_order_packet(
    const double dm_q,
    const double* __restrict__ receiver_r,
    const double* __restrict__ receiver_z,
    const int donor_nverts,
    const int receiver_nverts,
    double* __restrict__ donor_m_corner,
    double* __restrict__ donor_p_r,
    double* __restrict__ donor_p_z,
    double* __restrict__ receiver_m_corner,
    double* __restrict__ receiver_p_r,
    double* __restrict__ receiver_p_z) {
  if (!(dm_q > 0.0) || !isfinite(dm_q)) {
    return;
  }
  double ur = 0.0;
  double uz = 0.0;
  csr_optionb_live_cell_mean_velocity(
      donor_m_corner, donor_p_r, donor_p_z, donor_nverts, &ur, &uz);

  double donor_m_sum = 0.0;
  for (int k = 0; k < donor_nverts; ++k) {
    donor_m_sum +=
        (donor_m_corner[k] > 0.0 && isfinite(donor_m_corner[k]))
            ? donor_m_corner[k]
            : 0.0;
  }
  if (donor_m_sum > 0.0 && isfinite(donor_m_sum)) {
    const double inv_m_sum = 1.0 / donor_m_sum;
    for (int k = 0; k < donor_nverts; ++k) {
      const double w =
          (donor_m_corner[k] > 0.0 && isfinite(donor_m_corner[k]))
              ? donor_m_corner[k] * inv_m_sum
              : 0.0;
      const double dm = dm_q * w;
      donor_m_corner[k] -= dm;
      donor_p_r[k] -= dm * ur;
      donor_p_z[k] -= dm * uz;
    }
  }

  double receiver_w[4] = {0.0, 0.0, 0.0, 0.0};
  csr_optionb_unit_corner_weights(
      receiver_r, receiver_z, receiver_nverts, receiver_w);
  for (int k = 0; k < receiver_nverts; ++k) {
    const double dm = dm_q * receiver_w[k];
    receiver_m_corner[k] += dm;
    receiver_p_r[k] += dm * ur;
    receiver_p_z[k] += dm * uz;
  }
}

__host__ __device__ inline bool csr_optionb_ring5_cell(
    const int c,
    const int cell_start,
    const int cell_end) {
  return c >= cell_start && c <= cell_end;
}

__host__ __device__ inline bool csr_optionb_ring5_face(
    const int cell_a,
    const int cell_b,
    const int cell_start,
    const int cell_end) {
  return csr_optionb_ring5_cell(cell_a, cell_start, cell_end) ||
         csr_optionb_ring5_cell(cell_b, cell_start, cell_end);
}

__device__ inline double csr_optionb_packet_momentum_magnitude(
    const double dm_q,
    const double ur,
    const double uz) {
  if (!(dm_q > 0.0) || !isfinite(dm_q) || !isfinite(ur) || !isfinite(uz)) {
    return 0.0;
  }
  return dm_q * sqrt(ur * ur + uz * uz);
}

__device__ inline void csr_optionb_store_ring5_face_trace(
    const bool enabled,
    const int f,
    const int status,
    const int centroid_class,
    const int cell_a,
    const int cell_b,
    const int donor,
    const int receiver,
    const double dm_q,
    const double ur,
    const double uz,
    double* __restrict__ face_p,
    double* __restrict__ face_dm,
    double* __restrict__ face_u,
    int* __restrict__ face_status,
    int* __restrict__ face_centroid_class,
    int* __restrict__ face_cell_a,
    int* __restrict__ face_cell_b,
    int* __restrict__ face_donor,
    int* __restrict__ face_receiver) {
  if (!enabled || f < 0 || face_status == nullptr) {
    return;
  }
  const double u = (isfinite(ur) && isfinite(uz)) ? sqrt(ur * ur + uz * uz)
                                                  : 0.0;
  face_status[f] = status;
  if (face_centroid_class != nullptr) {
    face_centroid_class[f] = centroid_class;
  }
  if (face_cell_a != nullptr) {
    face_cell_a[f] = cell_a;
  }
  if (face_cell_b != nullptr) {
    face_cell_b[f] = cell_b;
  }
  if (face_donor != nullptr) {
    face_donor[f] = donor;
  }
  if (face_receiver != nullptr) {
    face_receiver[f] = receiver;
  }
  if (face_dm != nullptr) {
    face_dm[f] = (dm_q > 0.0 && isfinite(dm_q)) ? dm_q : 0.0;
  }
  if (face_u != nullptr) {
    face_u[f] = isfinite(u) ? u : 0.0;
  }
  if (face_p != nullptr) {
    face_p[f] = csr_optionb_packet_momentum_magnitude(dm_q, ur, uz);
  }
}

__device__ inline void csr_optionb_note_discarded_dual_flux(
    const int cell_a,
    const int cell_b,
    const int donor,
    const int receiver,
    const double dm_q,
    const double ur,
    const double uz,
    const std::uint8_t* __restrict__ discard_reference_inactive_cell_mask,
    double* __restrict__ discard_ledger) {
  if (discard_reference_inactive_cell_mask == nullptr ||
      discard_ledger == nullptr || !(dm_q > 0.0) || !isfinite(dm_q) ||
      !isfinite(ur) || !isfinite(uz)) {
    return;
  }
  const bool a_was_active =
      !csr_inactive_cell(discard_reference_inactive_cell_mask, cell_a);
  const bool b_was_active =
      !csr_inactive_cell(discard_reference_inactive_cell_mask, cell_b);
  if (a_was_active == b_was_active) {
    return;
  }
  // Report the old one-sided G contribution; the new collar scatter adds the
  // opposite contribution and cancels this pre-fix residual.
  const int old_uncancelled_cell = a_was_active ? cell_a : cell_b;
  const double sign = (old_uncancelled_cell == receiver) ? 1.0
                       : (old_uncancelled_cell == donor) ? -1.0
                                                         : 0.0;
  if (sign == 0.0) {
    return;
  }
  atomicAdd(discard_ledger + kReplayDiscardLedgerFaces, 1.0);
  atomicAdd(discard_ledger + kReplayDiscardLedgerMass, sign * dm_q);
  atomicAdd(discard_ledger + kReplayDiscardLedgerPiR, sign * dm_q * ur);
  atomicAdd(discard_ledger + kReplayDiscardLedgerPiZ, sign * dm_q * uz);
}

__device__ inline bool csr_optionb_expanded_cell_present(
    const int* __restrict__ cells,
    const int count,
    const int cell) {
  for (int i = 0; i < count; ++i) {
    if (cells[i] == cell) {
      return true;
    }
  }
  return false;
}

__device__ inline bool csr_optionb_expanded_add_cell(
    int* __restrict__ cells,
    int* __restrict__ count,
    const int cell,
    const int n_cells,
    const std::uint8_t* __restrict__ inactive_cell_mask) {
  if (cell < 0 || cell >= n_cells || csr_inactive_cell(inactive_cell_mask, cell)) {
    return true;
  }
  if (csr_optionb_expanded_cell_present(cells, *count, cell)) {
    return true;
  }
  if (*count >= kCsrOptionBMaxExpandedCells) {
    return false;
  }
  cells[*count] = cell;
  ++(*count);
  return true;
}

__device__ inline bool csr_optionb_expanded_node_present(
    const int* __restrict__ nodes,
    const int count,
    const int node) {
  for (int i = 0; i < count; ++i) {
    if (nodes[i] == node) {
      return true;
    }
  }
  return false;
}

__device__ inline bool csr_optionb_expanded_add_node(
    int* __restrict__ nodes,
    double* __restrict__ r,
    double* __restrict__ z,
    double* __restrict__ ur,
    double* __restrict__ uz,
    int* __restrict__ count,
    const int node,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const std::uint8_t* __restrict__ node_flags) {
  if (node < 0) {
    return true;
  }
  if (csr_optionb_expanded_node_present(nodes, *count, node)) {
    return true;
  }
  if (*count >= kCsrOptionBMaxExpandedNodes) {
    return false;
  }
  double vr = v_r_node[node];
  double vz = v_z_node[node];
  const auto projector =
      node_flags == nullptr
          ? tenryu::hydro::optionb::NodeVelocityProjector::FREE
          : csr_optionb_projector_from_flags(node_flags[node]);
  tenryu::hydro::optionb::apply_node_velocity_projector(projector, &vr, &vz);
  nodes[*count] = node;
  r[*count] = x_r_old[node];
  z[*count] = x_z_old[node];
  ur[*count] = vr;
  uz[*count] = vz;
  ++(*count);
  return true;
}

__device__ inline bool csr_optionb_build_expanded_stencil(
    int* __restrict__ stencil_nodes,
    double* __restrict__ stencil_r,
    double* __restrict__ stencil_z,
    double* __restrict__ stencil_ur,
    double* __restrict__ stencil_uz,
    int* __restrict__ stencil_node_count,
    const int donor,
    const int ring_limit,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ node_flags,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells) {
  int stencil_cells[kCsrOptionBMaxExpandedCells];
  int cell_count = 0;
  *stencil_node_count = 0;
  if (!csr_optionb_expanded_add_cell(
          stencil_cells, &cell_count, donor, n_cells, inactive_cell_mask)) {
    return false;
  }

  for (int ring = 0; ring <= ring_limit; ++ring) {
    const int current_cell_count = cell_count;
    for (int ci = 0; ci < current_cell_count; ++ci) {
      const int c = stencil_cells[ci];
      const int nverts =
          tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
      const int off = cell_node_csr_offsets[c];
      for (int k = 0; k < nverts; ++k) {
        const int n = cell_node_csr_indices[off + k];
        if (!csr_optionb_expanded_add_node(stencil_nodes,
                                           stencil_r,
                                           stencil_z,
                                           stencil_ur,
                                           stencil_uz,
                                           stencil_node_count,
                                           n,
                                           x_r_old,
                                           x_z_old,
                                           v_r_node,
                                           v_z_node,
                                           node_flags)) {
          return false;
        }
      }
    }
    if (ring == ring_limit || face_adj_csr_offsets == nullptr ||
        face_adj_csr_indices == nullptr) {
      continue;
    }
    for (int ci = 0; ci < current_cell_count; ++ci) {
      const int c = stencil_cells[ci];
      const int off = face_adj_csr_offsets[c];
      const int end = face_adj_csr_offsets[c + 1];
      for (int p = off; p < end; ++p) {
        const int neighbor = face_adj_csr_indices[p];
        if (!csr_optionb_expanded_add_cell(stencil_cells,
                                           &cell_count,
                                           neighbor,
                                           n_cells,
                                           inactive_cell_mask)) {
          return false;
        }
      }
    }
  }
  return *stencil_node_count >= 3;
}

__device__ inline bool csr_optionb_triangle_interpolation_weights(
    const double r0,
    const double z0,
    const double r1,
    const double z1,
    const double r2,
    const double z2,
    const double rq,
    const double zq,
    double* __restrict__ w0,
    double* __restrict__ w1,
    double* __restrict__ w2) {
  const double er1 = r1 - r0;
  const double ez1 = z1 - z0;
  const double er2 = r2 - r0;
  const double ez2 = z2 - z0;
  const double denom = er1 * ez2 - ez1 * er2;
  const double scale =
      fmax(1.0, fmax(fabs(er1) + fabs(ez1), fabs(er2) + fabs(ez2)));
  const double tol = 1024.0 * tenryu::hydro::optionb::detail::kDoubleEps *
                     scale * scale;
  if (!(fabs(denom) > tol) || !isfinite(denom)) {
    return false;
  }
  const double qr = rq - r0;
  const double qz = zq - z0;
  *w1 = (qr * ez2 - qz * er2) / denom;
  *w2 = (er1 * qz - ez1 * qr) / denom;
  *w0 = 1.0 - *w1 - *w2;
  const double w_tol = 4096.0 * tenryu::hydro::optionb::detail::kDoubleEps;
  return isfinite(*w0) && isfinite(*w1) && isfinite(*w2) &&
         *w0 >= -w_tol && *w1 >= -w_tol && *w2 >= -w_tol &&
         *w0 <= 1.0 + w_tol && *w1 <= 1.0 + w_tol && *w2 <= 1.0 + w_tol;
}

__device__ inline bool csr_optionb_interpolate_expanded_velocity(
    const double* __restrict__ stencil_r,
    const double* __restrict__ stencil_z,
    const double* __restrict__ stencil_ur,
    const double* __restrict__ stencil_uz,
    const int stencil_node_count,
    const double rq,
    const double zq,
    double* __restrict__ ur,
    double* __restrict__ uz) {
  const double point_tol = 1.0e-24;
  for (int i = 0; i < stencil_node_count; ++i) {
    const double dr = rq - stencil_r[i];
    const double dz = zq - stencil_z[i];
    if (dr * dr + dz * dz <= point_tol) {
      *ur = stencil_ur[i];
      *uz = stencil_uz[i];
      return true;
    }
  }
  for (int i = 0; i < stencil_node_count; ++i) {
    for (int j = i + 1; j < stencil_node_count; ++j) {
      for (int k = j + 1; k < stencil_node_count; ++k) {
        double w0 = 0.0;
        double w1 = 0.0;
        double w2 = 0.0;
        if (!csr_optionb_triangle_interpolation_weights(stencil_r[i],
                                                        stencil_z[i],
                                                        stencil_r[j],
                                                        stencil_z[j],
                                                        stencil_r[k],
                                                        stencil_z[k],
                                                        rq,
                                                        zq,
                                                        &w0,
                                                        &w1,
                                                        &w2)) {
          continue;
        }
        *ur = w0 * stencil_ur[i] + w1 * stencil_ur[j] + w2 * stencil_ur[k];
        *uz = w0 * stencil_uz[i] + w1 * stencil_uz[j] + w2 * stencil_uz[k];
        return isfinite(*ur) && isfinite(*uz);
      }
    }
  }
  return false;
}

__device__ inline tenryu::hydro::optionb::VelocityMomentumPacketFctResult
csr_optionb_apply_expanded_packet_fct(
    const double dm_q,
    const double rq,
    const double zq,
    const double* __restrict__ donor_r,
    const double* __restrict__ donor_z,
    const int donor_nverts,
    const double* __restrict__ donor_ur,
    const double* __restrict__ donor_uz,
    const tenryu::hydro::optionb::NodeVelocityProjector* __restrict__
        donor_projector,
    double* __restrict__ donor_m_corner,
    double* __restrict__ donor_p_r,
    double* __restrict__ donor_p_z,
    const double* __restrict__ receiver_r,
    const double* __restrict__ receiver_z,
    const int receiver_nverts,
    double* __restrict__ receiver_m_corner,
    double* __restrict__ receiver_p_r,
    double* __restrict__ receiver_p_z,
    const int donor,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ node_flags,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells,
    int* __restrict__ ring_used,
    const bool disable_fct_limiter,
    // Basis-coherent donor shares (see remap_velocity_momentum_packet_fct):
    // fixed pre-remap basis corner shares replacing the lambda_d donor-side
    // mass partition; exchanged momentum stays dm_q*u_q.
    const double* __restrict__ donor_sigma = nullptr) {
  auto result = tenryu::hydro::optionb::detail::
      make_velocity_momentum_packet_fct_result(
          tenryu::hydro::optionb::VelocityRemapStatus::NEEDS_EXPANDED_STENCIL);
  if (ring_used != nullptr) {
    *ring_used = 0;
  }
  if (!(dm_q > 0.0) || !isfinite(dm_q) ||
      !tenryu::hydro::optionb::detail::valid_cell_nverts(donor_nverts) ||
      !tenryu::hydro::optionb::detail::valid_cell_nverts(receiver_nverts)) {
    return tenryu::hydro::optionb::detail::
        make_velocity_momentum_packet_fct_result(
            tenryu::hydro::optionb::VelocityRemapStatus::INVALID_INPUT);
  }
  double lambda_d_packet[4] = {0.0, 0.0, 0.0, 0.0};
  double lambda_r_packet[4] = {0.0, 0.0, 0.0, 0.0};
  const auto donor_weight_status =
      csr_optionb_packet_centroid_weights(donor_r,
                                          donor_z,
                                          donor_nverts,
                                          rq,
                                          zq,
                                          lambda_d_packet);
  const auto receiver_weight_status =
      csr_optionb_packet_centroid_weights(receiver_r,
                                          receiver_z,
                                          receiver_nverts,
                                          rq,
                                          zq,
                                          lambda_r_packet);
  if (donor_weight_status ==
          CsrOptionBPacketCentroidWeightsStatus::INVALID_INPUT ||
      receiver_weight_status ==
          CsrOptionBPacketCentroidWeightsStatus::INVALID_INPUT) {
    return tenryu::hydro::optionb::detail::
        make_velocity_momentum_packet_fct_result(
            tenryu::hydro::optionb::VelocityRemapStatus::INVALID_INPUT);
  }
  if (donor_weight_status ==
          CsrOptionBPacketCentroidWeightsStatus::FAR_OUTSIDE ||
      receiver_weight_status ==
          CsrOptionBPacketCentroidWeightsStatus::FAR_OUTSIDE) {
    return result;
  }

  double u_q_r = 0.0;
  double u_q_z = 0.0;
  double u_tilde_r = 0.0;
  double u_tilde_z = 0.0;
  for (int a = 0; a < donor_nverts; ++a) {
    double ui_r = 0.0;
    double ui_z = 0.0;
    tenryu::hydro::optionb::detail::projected_velocity(
        donor_ur[a],
        donor_uz[a],
        tenryu::hydro::optionb::detail::node_projector_at(donor_projector, a),
        &ui_r,
        &ui_z);
    u_q_r += lambda_d_packet[a] * ui_r;
    u_q_z += lambda_d_packet[a] * ui_z;
    if (donor_sigma != nullptr) {
      if (!isfinite(donor_sigma[a]) || donor_sigma[a] < 0.0) {
        return tenryu::hydro::optionb::detail::
            make_velocity_momentum_packet_fct_result(
                tenryu::hydro::optionb::VelocityRemapStatus::INVALID_INPUT);
      }
      u_tilde_r += donor_sigma[a] * ui_r;
      u_tilde_z += donor_sigma[a] * ui_z;
    }
  }

  for (int ring = 1; ring <= kCsrOptionBMaxExpandedRing; ++ring) {
    int stencil_nodes[kCsrOptionBMaxExpandedNodes];
    double stencil_r[kCsrOptionBMaxExpandedNodes];
    double stencil_z[kCsrOptionBMaxExpandedNodes];
    double stencil_ur[kCsrOptionBMaxExpandedNodes];
    double stencil_uz[kCsrOptionBMaxExpandedNodes];
    int stencil_node_count = 0;
    if (!csr_optionb_build_expanded_stencil(stencil_nodes,
                                            stencil_r,
                                            stencil_z,
                                            stencil_ur,
                                            stencil_uz,
                                            &stencil_node_count,
                                            donor,
                                            ring,
                                            x_r_old,
                                            x_z_old,
                                            v_r_node,
                                            v_z_node,
                                            cell_node_csr_offsets,
                                            cell_node_csr_indices,
                                            face_adj_csr_offsets,
                                            face_adj_csr_indices,
                                            cell_nverts,
                                            node_flags,
                                            inactive_cell_mask,
                                            n_cells)) {
      continue;
    }

    double stencil_min_ur = 0.0;
    double stencil_max_ur = 0.0;
    double stencil_min_uz = 0.0;
    double stencil_max_uz = 0.0;
    for (int i = 0; i < stencil_node_count; ++i) {
      if (i == 0) {
        stencil_min_ur = stencil_max_ur = stencil_ur[i];
        stencil_min_uz = stencil_max_uz = stencil_uz[i];
      } else {
        stencil_min_ur = fmin(stencil_min_ur, stencil_ur[i]);
        stencil_max_ur = fmax(stencil_max_ur, stencil_ur[i]);
        stencil_min_uz = fmin(stencil_min_uz, stencil_uz[i]);
        stencil_max_uz = fmax(stencil_max_uz, stencil_uz[i]);
      }
    }
    if (!tenryu::hydro::optionb::detail::packet_low_order_value_admissible(
            u_q_r, stencil_min_ur, stencil_max_ur) ||
        !tenryu::hydro::optionb::detail::packet_low_order_value_admissible(
            u_q_z, stencil_min_uz, stencil_max_uz)) {
      continue;
    }

    double u_d_r_at_receiver[4] = {0.0, 0.0, 0.0, 0.0};
    double u_d_z_at_receiver[4] = {0.0, 0.0, 0.0, 0.0};
    bool all_receiver_vertices_ok = true;
    for (int b = 0; b < receiver_nverts; ++b) {
      if (!csr_optionb_interpolate_expanded_velocity(stencil_r,
                                                     stencil_z,
                                                     stencil_ur,
                                                     stencil_uz,
                                                     stencil_node_count,
                                                     receiver_r[b],
                                                     receiver_z[b],
                                                     &u_d_r_at_receiver[b],
                                                     &u_d_z_at_receiver[b])) {
        all_receiver_vertices_ok = false;
        break;
      }
    }
    if (!all_receiver_vertices_ok) {
      continue;
    }

    double u_avg_r = 0.0;
    double u_avg_z = 0.0;
    for (int b = 0; b < receiver_nverts; ++b) {
      u_avg_r += lambda_r_packet[b] * u_d_r_at_receiver[b];
      u_avg_z += lambda_r_packet[b] * u_d_z_at_receiver[b];
    }
    const double correction_r = u_q_r - u_avg_r;
    const double correction_z = u_q_z - u_avg_z;

    result.status = tenryu::hydro::optionb::VelocityRemapStatus::OK;
    result.alpha = 1.0;
    result.alpha_mass = 1.0;
    result.alpha_momentum_r = 1.0;
    result.alpha_momentum_z = 1.0;

    for (int a = 0; a < donor_nverts; ++a) {
      const double dm_a =
          dm_q *
          (donor_sigma != nullptr ? donor_sigma[a] : lambda_d_packet[a]);
      const double m_final = donor_m_corner[a] - dm_a;
      const double tol =
          1024.0 * tenryu::hydro::optionb::detail::kDoubleEps *
          fmax(1.0, fabs(donor_m_corner[a]) + fabs(dm_a));
      if (!isfinite(dm_a) || !isfinite(m_final) || m_final < -tol) {
        return tenryu::hydro::optionb::detail::
            make_velocity_momentum_packet_fct_result(
                tenryu::hydro::optionb::VelocityRemapStatus::INVALID_INPUT);
      }
    }
    for (int b = 0; b < receiver_nverts; ++b) {
      const double dm_b = dm_q * lambda_r_packet[b];
      const double m_final = receiver_m_corner[b] + dm_b;
      const double tol =
          1024.0 * tenryu::hydro::optionb::detail::kDoubleEps *
          fmax(1.0, fabs(receiver_m_corner[b]) + fabs(dm_b));
      if (!isfinite(dm_b) || !isfinite(m_final) || m_final < -tol) {
        return tenryu::hydro::optionb::detail::
            make_velocity_momentum_packet_fct_result(
                tenryu::hydro::optionb::VelocityRemapStatus::INVALID_INPUT);
      }

      const double u_hi_r = u_d_r_at_receiver[b] + correction_r;
      const double u_hi_z = u_d_z_at_receiver[b] + correction_z;
      const double p_lo_r = receiver_p_r[b] + dm_b * u_q_r;
      const double p_lo_z = receiver_p_z[b] + dm_b * u_q_z;
      const double p_anti_r = dm_b * (u_hi_r - u_q_r);
      const double p_anti_z = dm_b * (u_hi_z - u_q_z);
      if (!disable_fct_limiter) {
        if (!tenryu::hydro::optionb::detail::fct_receiver_velocity_alpha_bound(
                m_final,
                p_lo_r,
                p_anti_r,
                stencil_min_ur,
                stencil_max_ur,
                &result.alpha_momentum_r) ||
            !tenryu::hydro::optionb::detail::fct_receiver_velocity_alpha_bound(
                m_final,
                p_lo_z,
                p_anti_z,
                stencil_min_uz,
                stencil_max_uz,
                &result.alpha_momentum_z)) {
          return tenryu::hydro::optionb::detail::
              make_velocity_momentum_packet_fct_result(
                  tenryu::hydro::optionb::VelocityRemapStatus::INVALID_INPUT);
        }
      }
    }

    result.alpha = fmin(result.alpha_mass,
                       fmin(result.alpha_momentum_r, result.alpha_momentum_z));
    result.alpha = fmax(0.0, fmin(1.0, result.alpha));
    if (!isfinite(result.alpha)) {
      return tenryu::hydro::optionb::detail::
          make_velocity_momentum_packet_fct_result(
              tenryu::hydro::optionb::VelocityRemapStatus::INVALID_INPUT);
    }

    for (int a = 0; a < donor_nverts; ++a) {
      double ui_r = 0.0;
      double ui_z = 0.0;
      tenryu::hydro::optionb::detail::projected_velocity(
          donor_ur[a],
          donor_uz[a],
          tenryu::hydro::optionb::detail::node_projector_at(donor_projector, a),
          &ui_r,
          &ui_z);
      if (donor_sigma != nullptr) {
        const double dm_a = dm_q * donor_sigma[a];
        donor_m_corner[a] -= dm_a;
        donor_p_r[a] -= dm_a * (ui_r + (u_q_r - u_tilde_r));
        donor_p_z[a] -= dm_a * (ui_z + (u_q_z - u_tilde_z));
      } else {
        const double dm_a = dm_q * lambda_d_packet[a];
        donor_m_corner[a] -= dm_a;
        donor_p_r[a] -= dm_a * ui_r;
        donor_p_z[a] -= dm_a * ui_z;
      }
    }
    for (int b = 0; b < receiver_nverts; ++b) {
      const double dm_b = dm_q * lambda_r_packet[b];
      const double u_hi_r = u_d_r_at_receiver[b] + correction_r;
      const double u_hi_z = u_d_z_at_receiver[b] + correction_z;
      receiver_m_corner[b] += dm_b;
      receiver_p_r[b] +=
          dm_b * (u_q_r + result.alpha * (u_hi_r - u_q_r));
      receiver_p_z[b] +=
          dm_b * (u_q_z + result.alpha * (u_hi_z - u_q_z));
    }
    if (ring_used != nullptr) {
      *ring_used = ring;
    }
    return result;
  }
  return result;
}

__device__ inline void csr_optionb_classify_expanded_need(
    const double rq,
    const double zq,
    const double* __restrict__ donor_r,
    const double* __restrict__ donor_z,
    const int donor_nverts,
    const double* __restrict__ receiver_r,
    const double* __restrict__ receiver_z,
    const int receiver_nverts,
    int* __restrict__ diagnostics) {
  if (diagnostics == nullptr) {
    return;
  }
  const int centroid_class = csr_optionb_centroid_out_class(donor_r,
                                                           donor_z,
                                                           donor_nverts,
                                                           receiver_r,
                                                           receiver_z,
                                                           receiver_nverts,
                                                           rq,
                                                           zq);
  if (centroid_class != 0) {
    atomicAdd(diagnostics + kCsrOptionBDiagCentroidOut, 1);
    if (centroid_class == 1) {
      atomicAdd(diagnostics + kCsrOptionBDiagCentroidOnBoundary, 1);
    } else {
      atomicAdd(diagnostics + kCsrOptionBDiagCentroidFar, 1);
    }
    return;
  }
  bool receiver_vertex_out = false;
  for (int b = 0; b < receiver_nverts; ++b) {
    if (!tenryu::hydro::optionb::detail::point_in_convex_hull(
            donor_r, donor_z, donor_nverts, receiver_r[b], receiver_z[b])) {
      receiver_vertex_out = true;
      break;
    }
  }
  if (receiver_vertex_out) {
    atomicAdd(diagnostics + kCsrOptionBDiagReceiverVertexOut, 1);
  }
}

__device__ inline void csr_optionb_apply_boundary_mass_change(
    const double dm_signed,
    const double* __restrict__ target_r,
    const double* __restrict__ target_z,
    const int nverts,
    double* __restrict__ m_corner,
    double* __restrict__ p_r,
    double* __restrict__ p_z) {
  if (!finite_nonzero(dm_signed)) {
    return;
  }
  double ur = 0.0;
  double uz = 0.0;
  csr_optionb_live_cell_mean_velocity(
      m_corner, p_r, p_z, nverts, &ur, &uz);
  if (dm_signed < 0.0) {
    const double dm_out = -dm_signed;
    double m_sum = 0.0;
    for (int k = 0; k < nverts; ++k) {
      m_sum += (m_corner[k] > 0.0 && isfinite(m_corner[k])) ? m_corner[k] : 0.0;
    }
    if (!(m_sum > 0.0) || !isfinite(m_sum)) {
      return;
    }
    const double inv_m_sum = 1.0 / m_sum;
    for (int k = 0; k < nverts; ++k) {
      const double w =
          (m_corner[k] > 0.0 && isfinite(m_corner[k])) ? m_corner[k] * inv_m_sum
                                                       : 0.0;
      const double dm = dm_out * w;
      m_corner[k] -= dm;
      p_r[k] -= dm * ur;
      p_z[k] -= dm * uz;
    }
    return;
  }

  double w[4] = {0.0, 0.0, 0.0, 0.0};
  csr_optionb_unit_corner_weights(target_r, target_z, nverts, w);
  for (int k = 0; k < nverts; ++k) {
    const double dm = dm_signed * w[k];
    m_corner[k] += dm;
    p_r[k] += dm * ur;
    p_z[k] += dm * uz;
  }
}

__global__ void csr_optionb_gather_corner_momentum_kernel(
    double* __restrict__ optionb_m_corner,
    double* __restrict__ optionb_p_r,
    double* __restrict__ optionb_p_z,
    const double* __restrict__ mass_lag,
    const double* __restrict__ rho_lag,
    const double* __restrict__ vol_lag,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ node_flags,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const double* __restrict__ basis_corner_mass,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    const int base = c * 4;
    for (int k = 0; k < 4; ++k) {
      optionb_m_corner[base + k] = 0.0;
      optionb_p_r[base + k] = 0.0;
      optionb_p_z[base + k] = 0.0;
    }
    return;
  }
  const int nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  const int off = cell_node_csr_offsets[c];
  double r[4] = {0.0, 0.0, 0.0, 0.0};
  double z[4] = {0.0, 0.0, 0.0, 0.0};
  double ur[4] = {0.0, 0.0, 0.0, 0.0};
  double uz[4] = {0.0, 0.0, 0.0, 0.0};
  tenryu::hydro::optionb::NodeVelocityProjector projector[4] = {
      tenryu::hydro::optionb::NodeVelocityProjector::FREE,
      tenryu::hydro::optionb::NodeVelocityProjector::FREE,
      tenryu::hydro::optionb::NodeVelocityProjector::FREE,
      tenryu::hydro::optionb::NodeVelocityProjector::FREE};
  for (int k = 0; k < nverts; ++k) {
    const int n = cell_node_csr_indices[off + k];
    r[k] = x_r_old[n];
    z[k] = x_z_old[n];
    ur[k] = v_r_node[n];
    uz[k] = v_z_node[n];
    projector[k] =
        node_flags == nullptr
            ? tenryu::hydro::optionb::NodeVelocityProjector::FREE
            : csr_optionb_projector_from_flags(node_flags[n]);
  }

  double m_cell = (mass_lag[c] > 0.0 && isfinite(mass_lag[c])) ? mass_lag[c] : 0.0;
  if (!(m_cell > 0.0) && rho_lag != nullptr && vol_lag != nullptr) {
    m_cell = fmax(rho_lag[c], 0.0) * fmax(vol_lag[c], 0.0);
  }
  double m_local[4] = {0.0, 0.0, 0.0, 0.0};
  double pr_local[4] = {0.0, 0.0, 0.0, 0.0};
  double pz_local[4] = {0.0, 0.0, 0.0, 0.0};
  if (basis_corner_mass != nullptr) {
    // PR5(b) of the corner-mass basis contract: build the corner momenta
    // with the CURRENT subzonal basis (state.corner_mass — the V-paired
    // remapped masses once PR6-LO installs run, the frozen CBSW masses
    // otherwise) instead of the transient geometric first moment, so the
    // velocity remap conserves momentum in the SAME basis the dynamics'
    // nodal masses and the energy budget use.
    for (int k = 0; k < nverts; ++k) {
      double ui_r = ur[k];
      double ui_z = uz[k];
      tenryu::hydro::optionb::apply_node_velocity_projector(projector[k],
                                                            &ui_r, &ui_z);
      const double m_k = fmax(basis_corner_mass[c * 4 + k], 0.0);
      m_local[k] = isfinite(m_k) ? m_k : 0.0;
      pr_local[k] = m_local[k] * ui_r;
      pz_local[k] = m_local[k] * ui_z;
    }
  } else {
    tenryu::hydro::optionb::gather_corner_momentum(m_cell,
                                                   r,
                                                   z,
                                                   nverts,
                                                   ur,
                                                   uz,
                                                   projector,
                                                   m_local,
                                                   pr_local,
                                                   pz_local);
  }
  const int base = c * 4;
  for (int k = 0; k < 4; ++k) {
    optionb_m_corner[base + k] = (k < nverts) ? m_local[k] : 0.0;
    optionb_p_r[base + k] = (k < nverts) ? pr_local[k] : 0.0;
    optionb_p_z[base + k] = (k < nverts) ? pz_local[k] : 0.0;
  }
}

__global__ void csr_optionb_apply_internal_packets_color_kernel(
    double* __restrict__ optionb_m_corner,
    double* __restrict__ optionb_p_r,
    double* __restrict__ optionb_p_z,
    const double* __restrict__ rho_lag,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ unique_cell_a,
    const int* __restrict__ unique_cell_b,
    const int* __restrict__ unique_local_a,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ node_flags,
    const double* __restrict__ mass_flux_scale,
    const int* __restrict__ face_color,
    const int active_color,
    const int n_faces,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const std::uint8_t* __restrict__ donor_fallback_cell_mask,
    const std::uint8_t* __restrict__ discard_reference_inactive_cell_mask,
    double* __restrict__ discard_ledger,
    // Basis-coherent transport: pre-remap basis corner masses
    // (state.corner_mass, unchanged for the whole remap); non-null only
    // under TENRYU_I1B_OPTIONB_COHERENT.
    const double* __restrict__ basis_corner_mass,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const int n_cells,
    const bool print_centroid_far_diag,
    const bool disable_fct_limiter,
    const bool ring5_trace_enabled,
    const int ring5_cell_start,
    const int ring5_cell_end,
    int* __restrict__ diagnostics,
    double* __restrict__ diagnostics_real,
    double* __restrict__ ring5_face_p,
    double* __restrict__ ring5_face_dm,
    double* __restrict__ ring5_face_u,
    int* __restrict__ ring5_face_status,
    int* __restrict__ ring5_face_centroid_class,
    int* __restrict__ ring5_face_cell_a,
    int* __restrict__ ring5_face_cell_b,
    int* __restrict__ ring5_face_donor,
    int* __restrict__ ring5_face_receiver,
    const RemapDispatchAuditDeviceView remap_dispatch_audit) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces || face_color[f] != active_color) {
    return;
  }
  const int cell_a = unique_cell_a[f];
  const int cell_b = unique_cell_b[f];
  const bool trace_face =
      ring5_trace_enabled &&
      csr_optionb_ring5_face(cell_a, cell_b, ring5_cell_start, ring5_cell_end);
  if (csr_face_touches_inactive(inactive_cell_mask, cell_a, cell_b)) {
    if (diagnostics != nullptr) {
      atomicAdd(diagnostics + kCsrOptionBDiagSkipped, 1);
    }
    csr_optionb_store_ring5_face_trace(trace_face,
                                       f,
                                       kCsrOptionBRing5FaceStatusSkippedInactive,
                                       0,
                                       cell_a,
                                       cell_b,
                                       -1,
                                       -1,
                                       0.0,
                                       0.0,
                                       0.0,
                                       ring5_face_p,
                                       ring5_face_dm,
                                       ring5_face_u,
                                       ring5_face_status,
                                       ring5_face_centroid_class,
                                       ring5_face_cell_a,
                                       ring5_face_cell_b,
                                       ring5_face_donor,
                                       ring5_face_receiver);
    return;
  }
  const int local_a = unique_local_a[f];
  double rq = 0.0;
  double zq = 0.0;
  detail::CsrFaceSweptMomentsStatus moment_status =
      detail::CsrFaceSweptMomentsStatus::SKIP;
  bool exact_moments = false;
  const double dV_a =
      ::tenryu::hydro::ale::csr_face_swept_moments_outward(
          x_r_old,
          x_z_old,
          x_r_new,
          x_z_new,
          cell_node_csr_offsets,
          cell_node_csr_indices,
          cell_orientation_sign,
          cell_a,
          local_a,
          cell_nverts,
          &rq,
          &zq,
          &moment_status,
          &exact_moments);
  if (moment_status != detail::CsrFaceSweptMomentsStatus::OK ||
      !finite_nonzero(dV_a)) {
    if (diagnostics != nullptr) {
      atomicAdd(diagnostics + kCsrOptionBDiagSkipped, 1);
    }
    csr_optionb_store_ring5_face_trace(trace_face,
                                       f,
                                       kCsrOptionBRing5FaceStatusSkippedZero,
                                       0,
                                       cell_a,
                                       cell_b,
                                       -1,
                                       -1,
                                       0.0,
                                       0.0,
                                       0.0,
                                       ring5_face_p,
                                       ring5_face_dm,
                                       ring5_face_u,
                                       ring5_face_status,
                                       ring5_face_centroid_class,
                                       ring5_face_cell_a,
                                       ring5_face_cell_b,
                                       ring5_face_donor,
                                       ring5_face_receiver);
    return;
  }

  const int donor = csr_internal_flux_donor(cell_a, cell_b, dV_a);
  const int receiver = (donor == cell_a) ? cell_b : cell_a;
  const int losing_cell = csr_internal_flux_losing_cell(cell_a, cell_b, dV_a);
  double dV_limited = dV_a;
  if (mass_flux_scale != nullptr) {
    double s = mass_flux_scale[losing_cell];
    if (!isfinite(s)) {
      s = 0.0;
    }
    dV_limited = dV_a * fmin(1.0, fmax(0.0, s));
  }
  const double dm_q = fmax(rho_lag[donor], 0.0) * fabs(dV_limited);
  if (!(dm_q > 0.0) || !isfinite(dm_q)) {
    if (diagnostics != nullptr) {
      atomicAdd(diagnostics + kCsrOptionBDiagSkipped, 1);
    }
    csr_optionb_store_ring5_face_trace(trace_face,
                                       f,
                                       kCsrOptionBRing5FaceStatusSkippedZero,
                                       0,
                                       cell_a,
                                       cell_b,
                                       donor,
                                       receiver,
                                       0.0,
                                       0.0,
                                       0.0,
                                       ring5_face_p,
                                       ring5_face_dm,
                                       ring5_face_u,
                                       ring5_face_status,
                                       ring5_face_centroid_class,
                                       ring5_face_cell_a,
                                       ring5_face_cell_b,
                                       ring5_face_donor,
                                       ring5_face_receiver);
    return;
  }
  remap_dispatch_audit_count(
      remap_dispatch_audit,
      exact_moments
          ? RemapDispatchAuditCounter::ExactSweptMoment
          : RemapDispatchAuditCounter::SweptCentroidAverageFallback,
      donor);

  const int donor_nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, donor);
  const int receiver_nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, receiver);
  double donor_r[kCsrOptionBFaceCornerSlots] = {};
  double donor_z[kCsrOptionBFaceCornerSlots] = {};
  double receiver_r[kCsrOptionBFaceCornerSlots] = {};
  double receiver_z[kCsrOptionBFaceCornerSlots] = {};
  double donor_ur[4] = {0.0, 0.0, 0.0, 0.0};
  double donor_uz[4] = {0.0, 0.0, 0.0, 0.0};
  tenryu::hydro::optionb::NodeVelocityProjector donor_projector[4] = {
      tenryu::hydro::optionb::NodeVelocityProjector::FREE,
      tenryu::hydro::optionb::NodeVelocityProjector::FREE,
      tenryu::hydro::optionb::NodeVelocityProjector::FREE,
      tenryu::hydro::optionb::NodeVelocityProjector::FREE};
  csr_optionb_cell_geometry(donor_r,
                            donor_z,
                            x_r_old,
                            x_z_old,
                            cell_node_csr_offsets,
                            cell_node_csr_indices,
                            cell_nverts,
                            donor,
                            donor_nverts);
  csr_optionb_cell_geometry(receiver_r,
                            receiver_z,
                            x_r_new,
                            x_z_new,
                            cell_node_csr_offsets,
                            cell_node_csr_indices,
                            cell_nverts,
                            receiver,
                            receiver_nverts);
  const int donor_off = cell_node_csr_offsets[donor];
  for (int k = 0; k < donor_nverts; ++k) {
    const int n = cell_node_csr_indices[donor_off + k];
    donor_ur[k] = v_r_node[n];
    donor_uz[k] = v_z_node[n];
    donor_projector[k] =
        node_flags == nullptr
            ? tenryu::hydro::optionb::NodeVelocityProjector::FREE
            : csr_optionb_projector_from_flags(node_flags[n]);
  }

  double lambda_d_packet[4] = {0.0, 0.0, 0.0, 0.0};
  double lambda_r_packet[4] = {0.0, 0.0, 0.0, 0.0};
  double lambda_d_vertex[4] = {0.0, 0.0, 0.0, 0.0};
  double u_d_r_at_receiver[4] = {0.0, 0.0, 0.0, 0.0};
  double u_d_z_at_receiver[4] = {0.0, 0.0, 0.0, 0.0};
  // Coherent donor shares: normalize the donor's FIXED pre-remap basis
  // corner masses. The total outgoing mass is clamped to the donor cell
  // mass by mass_flux_scale, so sigma-proportional removal keeps every
  // basis-seeded donor corner non-negative across the whole remap.
  double donor_sigma[4] = {0.0, 0.0, 0.0, 0.0};
  const double* donor_sigma_ptr = nullptr;
  if (basis_corner_mass != nullptr) {
    double sigma_sum = 0.0;
    bool sigma_ok = true;
    for (int k = 0; k < donor_nverts; ++k) {
      const double m_k = basis_corner_mass[donor * 4 + k];
      if (!isfinite(m_k) || m_k < 0.0) {
        sigma_ok = false;
        break;
      }
      donor_sigma[k] = m_k;
      sigma_sum += m_k;
    }
    if (sigma_ok && sigma_sum > 0.0 && isfinite(sigma_sum)) {
      for (int k = 0; k < donor_nverts; ++k) {
        donor_sigma[k] /= sigma_sum;
      }
      donor_sigma_ptr = donor_sigma;
    }
  }
  double* const donor_m = optionb_m_corner + donor * 4;
  double* const donor_pr = optionb_p_r + donor * 4;
  double* const donor_pz = optionb_p_z + donor * 4;
  double* const receiver_m = optionb_m_corner + receiver * 4;
  double* const receiver_pr = optionb_p_r + receiver * 4;
  double* const receiver_pz = optionb_p_z + receiver * 4;
  const int centroid_class_for_trace =
      trace_face ? csr_optionb_centroid_out_class(donor_r,
                                                  donor_z,
                                                  donor_nverts,
                                                  receiver_r,
                                                  receiver_z,
                                                  receiver_nverts,
                                                  rq,
                                                  zq)
                 : 0;
  if (donor_fallback_cell_mask != nullptr &&
      donor_fallback_cell_mask[donor] != 0U) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::FirstOrderDonorFallback,
        donor);
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::MomentumPacketFallback,
        donor);
    double donor_ur_mean = 0.0;
    double donor_uz_mean = 0.0;
    csr_optionb_live_cell_mean_velocity(donor_m,
                                        donor_pr,
                                        donor_pz,
                                        donor_nverts,
                                        &donor_ur_mean,
                                        &donor_uz_mean);
    csr_optionb_store_ring5_face_trace(
        trace_face,
        f,
        kCsrOptionBRing5FaceStatusFallback,
        centroid_class_for_trace,
        cell_a,
        cell_b,
        donor,
        receiver,
        dm_q,
        donor_ur_mean,
        donor_uz_mean,
        ring5_face_p,
        ring5_face_dm,
        ring5_face_u,
        ring5_face_status,
        ring5_face_centroid_class,
        ring5_face_cell_a,
        ring5_face_cell_b,
        ring5_face_donor,
        ring5_face_receiver);
    csr_optionb_note_discarded_dual_flux(
        cell_a,
        cell_b,
        donor,
        receiver,
        dm_q,
        donor_ur_mean,
        donor_uz_mean,
        discard_reference_inactive_cell_mask,
        discard_ledger);
    csr_optionb_atomic_min_double(
        diagnostics_real != nullptr
            ? diagnostics_real + kCsrOptionBDiagRealAlphaMin
            : nullptr,
        1.0);
    csr_optionb_apply_first_order_packet(dm_q,
                                         receiver_r,
                                         receiver_z,
                                         donor_nverts,
                                         receiver_nverts,
                                         donor_m,
                                         donor_pr,
                                         donor_pz,
                                         receiver_m,
                                         receiver_pr,
                                         receiver_pz);
    return;
  }
  const auto packet_result =
      tenryu::hydro::optionb::remap_velocity_momentum_packet_fct(
          dm_q,
          rq,
          zq,
          donor_r,
          donor_z,
          donor_nverts,
          donor_ur,
          donor_uz,
          donor_projector,
          donor_m,
          donor_pr,
          donor_pz,
          receiver_r,
          receiver_z,
          receiver_nverts,
          receiver_m,
          receiver_pr,
          receiver_pz,
          lambda_d_packet,
          lambda_r_packet,
          lambda_d_vertex,
          u_d_r_at_receiver,
          u_d_z_at_receiver,
          disable_fct_limiter,
          donor_sigma_ptr);
  if (packet_result.status ==
      tenryu::hydro::optionb::VelocityRemapStatus::OK) {
    if (packet_result.alpha < 1.0) {
      remap_dispatch_audit_count(
          remap_dispatch_audit,
          RemapDispatchAuditCounter::LimiterActivation,
          donor);
    }
    double u_q_r = 0.0;
    double u_q_z = 0.0;
    for (int a = 0; a < donor_nverts; ++a) {
      double ui_r = 0.0;
      double ui_z = 0.0;
      tenryu::hydro::optionb::detail::projected_velocity(
          donor_ur[a],
          donor_uz[a],
          tenryu::hydro::optionb::detail::node_projector_at(donor_projector, a),
          &ui_r,
          &ui_z);
      u_q_r += lambda_d_packet[a] * ui_r;
      u_q_z += lambda_d_packet[a] * ui_z;
    }
    csr_optionb_store_ring5_face_trace(trace_face,
                                       f,
                                       kCsrOptionBRing5FaceStatusPacketOk,
                                       centroid_class_for_trace,
                                       cell_a,
                                       cell_b,
                                       donor,
                                       receiver,
                                       dm_q,
                                       u_q_r,
                                       u_q_z,
                                       ring5_face_p,
                                       ring5_face_dm,
                                       ring5_face_u,
                                       ring5_face_status,
                                       ring5_face_centroid_class,
                                       ring5_face_cell_a,
                                       ring5_face_cell_b,
                                       ring5_face_donor,
                                       ring5_face_receiver);
    csr_optionb_note_discarded_dual_flux(cell_a,
                                         cell_b,
                                         donor,
                                         receiver,
                                         dm_q,
                                         u_q_r,
                                         u_q_z,
                                         discard_reference_inactive_cell_mask,
                                         discard_ledger);
    csr_optionb_atomic_min_double(
        diagnostics_real != nullptr
            ? diagnostics_real + kCsrOptionBDiagRealAlphaMin
            : nullptr,
        packet_result.alpha);
    return;
  }

  if (packet_result.status ==
      tenryu::hydro::optionb::VelocityRemapStatus::NEEDS_EXPANDED_STENCIL) {
    if (diagnostics != nullptr) {
      atomicAdd(diagnostics + kCsrOptionBDiagExpanded, 1);
      csr_optionb_classify_expanded_need(rq,
                                         zq,
                                         donor_r,
                                         donor_z,
                                         donor_nverts,
                                         receiver_r,
                                         receiver_z,
                                         receiver_nverts,
                                         diagnostics);
      if (print_centroid_far_diag &&
          csr_optionb_centroid_out_class(donor_r,
                                         donor_z,
                                         donor_nverts,
                                         receiver_r,
                                         receiver_z,
                                         receiver_nverts,
                                         rq,
                                         zq) == 2) {
        csr_optionb_print_centroid_far_packet(diagnostics,
                                              print_centroid_far_diag,
                                              f,
                                              cell_a,
                                              cell_b,
                                              donor,
                                              receiver,
                                              local_a,
                                              dV_a,
                                              dm_q,
                                              rq,
                                              zq,
                                              donor_r,
                                              donor_z,
                                              donor_nverts,
                                              receiver_r,
                                              receiver_z,
                                              receiver_nverts,
                                              x_r_old,
                                              x_z_old,
                                              x_r_new,
                                              x_z_new,
                                              cell_node_csr_offsets,
                                              cell_node_csr_indices,
                                              cell_nverts);
      }
    }
    int ring_used = 0;
    const auto expanded_result =
        csr_optionb_apply_expanded_packet_fct(dm_q,
                                              rq,
                                              zq,
                                              donor_r,
                                              donor_z,
                                              donor_nverts,
                                              donor_ur,
                                              donor_uz,
                                              donor_projector,
                                              donor_m,
                                              donor_pr,
                                              donor_pz,
                                              receiver_r,
                                              receiver_z,
                                              receiver_nverts,
                                              receiver_m,
                                              receiver_pr,
                                              receiver_pz,
                                              donor,
                                              x_r_old,
                                              x_z_old,
                                              v_r_node,
                                              v_z_node,
                                              cell_node_csr_offsets,
                                              cell_node_csr_indices,
                                              face_adj_csr_offsets,
                                              face_adj_csr_indices,
                                              cell_nverts,
                                              node_flags,
                                              inactive_cell_mask,
                                              n_cells,
                                              &ring_used,
                                              disable_fct_limiter,
                                              donor_sigma_ptr);
    if (expanded_result.status ==
        tenryu::hydro::optionb::VelocityRemapStatus::OK) {
      if (expanded_result.alpha < 1.0) {
        remap_dispatch_audit_count(
            remap_dispatch_audit,
            RemapDispatchAuditCounter::LimiterActivation,
            donor);
      }
      double u_q_r = 0.0;
      double u_q_z = 0.0;
      double lambda_sum = 0.0;
      for (int a = 0; a < donor_nverts; ++a) {
        double ui_r = 0.0;
        double ui_z = 0.0;
        tenryu::hydro::optionb::detail::projected_velocity(
            donor_ur[a],
            donor_uz[a],
            tenryu::hydro::optionb::detail::node_projector_at(
                donor_projector, a),
            &ui_r,
            &ui_z);
        u_q_r += lambda_d_packet[a] * ui_r;
        u_q_z += lambda_d_packet[a] * ui_z;
        lambda_sum += lambda_d_packet[a];
      }
      if (!(fabs(lambda_sum - 1.0) < 1.0e-8) || !isfinite(lambda_sum)) {
        csr_optionb_live_cell_mean_velocity(
            donor_m, donor_pr, donor_pz, donor_nverts, &u_q_r, &u_q_z);
      }
      csr_optionb_store_ring5_face_trace(trace_face,
                                         f,
                                         kCsrOptionBRing5FaceStatusExpandedOk,
                                         centroid_class_for_trace,
                                         cell_a,
                                         cell_b,
                                         donor,
                                         receiver,
                                         dm_q,
                                         u_q_r,
                                         u_q_z,
                                         ring5_face_p,
                                         ring5_face_dm,
                                         ring5_face_u,
                                         ring5_face_status,
                                         ring5_face_centroid_class,
                                         ring5_face_cell_a,
                                         ring5_face_cell_b,
                                         ring5_face_donor,
                                         ring5_face_receiver);
      csr_optionb_note_discarded_dual_flux(cell_a,
                                           cell_b,
                                           donor,
                                           receiver,
                                           dm_q,
                                           u_q_r,
                                           u_q_z,
                                           discard_reference_inactive_cell_mask,
                                           discard_ledger);
      csr_optionb_atomic_min_double(
          diagnostics_real != nullptr
              ? diagnostics_real + kCsrOptionBDiagRealAlphaMin
              : nullptr,
          expanded_result.alpha);
      if (diagnostics != nullptr) {
        if (ring_used == 1) {
          atomicAdd(diagnostics + kCsrOptionBDiagExpandedRing1, 1);
        } else if (ring_used == 2) {
          atomicAdd(diagnostics + kCsrOptionBDiagExpandedRing2, 1);
        }
      }
      return;
    }
    if (diagnostics != nullptr) {
      atomicAdd(diagnostics + kCsrOptionBDiagExpandedFailed, 1);
      if (expanded_result.status ==
          tenryu::hydro::optionb::VelocityRemapStatus::INVALID_INPUT) {
        atomicAdd(diagnostics + kCsrOptionBDiagInvalid, 1);
      }
    }
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::MomentumExpandedFailureFallback,
        donor);
    if (expanded_result.status ==
        tenryu::hydro::optionb::VelocityRemapStatus::INVALID_INPUT) {
      remap_dispatch_audit_count(
          remap_dispatch_audit,
          RemapDispatchAuditCounter::MomentumInvalidInputFallback,
          donor);
    }
  } else if (diagnostics != nullptr) {
    atomicAdd(diagnostics + kCsrOptionBDiagInvalid, 1);
  }

  if (packet_result.status ==
      tenryu::hydro::optionb::VelocityRemapStatus::INVALID_INPUT) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::MomentumInvalidInputFallback,
        donor);
  }

  if (diagnostics != nullptr) {
    atomicAdd(diagnostics + kCsrOptionBDiagFallback, 1);
  }
  remap_dispatch_audit_count(
      remap_dispatch_audit,
      RemapDispatchAuditCounter::FirstOrderDonorFallback,
      donor);
  remap_dispatch_audit_count(
      remap_dispatch_audit,
      RemapDispatchAuditCounter::MomentumPacketFallback,
      donor);
  double fallback_ur = 0.0;
  double fallback_uz = 0.0;
  csr_optionb_live_cell_mean_velocity(
      donor_m, donor_pr, donor_pz, donor_nverts, &fallback_ur, &fallback_uz);
  csr_optionb_store_ring5_face_trace(
      trace_face,
      f,
      (packet_result.status ==
           tenryu::hydro::optionb::VelocityRemapStatus::INVALID_INPUT)
          ? kCsrOptionBRing5FaceStatusInvalid
          : kCsrOptionBRing5FaceStatusFallback,
      centroid_class_for_trace,
      cell_a,
      cell_b,
      donor,
      receiver,
      dm_q,
      fallback_ur,
      fallback_uz,
      ring5_face_p,
      ring5_face_dm,
      ring5_face_u,
      ring5_face_status,
      ring5_face_centroid_class,
      ring5_face_cell_a,
      ring5_face_cell_b,
      ring5_face_donor,
      ring5_face_receiver);
  csr_optionb_note_discarded_dual_flux(cell_a,
                                       cell_b,
                                       donor,
                                       receiver,
                                       dm_q,
                                       fallback_ur,
                                       fallback_uz,
                                       discard_reference_inactive_cell_mask,
                                       discard_ledger);
  csr_optionb_apply_first_order_packet(dm_q,
                                       receiver_r,
                                       receiver_z,
                                       donor_nverts,
                                       receiver_nverts,
                                       donor_m,
                                       donor_pr,
                                       donor_pz,
                                       receiver_m,
                                       receiver_pr,
                                       receiver_pz);
}

__device__ inline bool csr_optionb_face_weights_usable(
    const CsrOptionBPacketCentroidWeightsStatus status) {
  return status == CsrOptionBPacketCentroidWeightsStatus::OK ||
         status == CsrOptionBPacketCentroidWeightsStatus::NEAR_BOUNDARY;
}

__device__ inline void csr_optionb_normalize_face_weights(
    double* __restrict__ w,
    const int nverts) {
  double sum = 0.0;
  for (int k = 0; k < nverts; ++k) {
    w[k] = (w[k] > 0.0 && isfinite(w[k])) ? w[k] : 0.0;
    sum += w[k];
  }
  if (!(sum > 0.0) || !isfinite(sum)) {
    const double uniform = 1.0 / static_cast<double>(nverts);
    for (int k = 0; k < nverts; ++k) {
      w[k] = uniform;
    }
    return;
  }
  const double inv_sum = 1.0 / sum;
  for (int k = 0; k < nverts; ++k) {
    w[k] *= inv_sum;
  }
}

__device__ inline bool csr_optionb_basis_face_weights(
    const double* __restrict__ basis_corner_mass,
    const int cell,
    const int nverts,
    double* __restrict__ w) {
  if (basis_corner_mass == nullptr) {
    return false;
  }
  double sum = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const double m = basis_corner_mass[cell * 4 + k];
    if (!isfinite(m) || m < 0.0) {
      return false;
    }
    w[k] = m;
    sum += m;
  }
  if (!(sum > 0.0) || !isfinite(sum)) {
    return false;
  }
  const double inv_sum = 1.0 / sum;
  for (int k = 0; k < nverts; ++k) {
    w[k] *= inv_sum;
  }
  return true;
}

__device__ inline void csr_optionb_velocity_from_face_weights(
    const double* __restrict__ ur,
    const double* __restrict__ uz,
    const tenryu::hydro::optionb::NodeVelocityProjector* __restrict__ projector,
    const double* __restrict__ w,
    const int nverts,
    double* __restrict__ uhat_r,
    double* __restrict__ uhat_z) {
  *uhat_r = 0.0;
  *uhat_z = 0.0;
  for (int k = 0; k < nverts; ++k) {
    double ui_r = 0.0;
    double ui_z = 0.0;
    tenryu::hydro::optionb::detail::projected_velocity(
        ur[k],
        uz[k],
        tenryu::hydro::optionb::detail::node_projector_at(projector, k),
        &ui_r,
        &ui_z);
    *uhat_r += w[k] * ui_r;
    *uhat_z += w[k] * ui_z;
  }
}

__device__ inline void csr_optionb_store_face_side_flux(
    double* __restrict__ face_delta_m,
    double* __restrict__ face_delta_pr,
    double* __restrict__ face_delta_pz,
    const int f,
    const int slot_base,
    const int nverts,
    const double sign,
    const double dm_flux,
    const double pr_flux,
    const double pz_flux,
    const double* __restrict__ w) {
  double dm_sum = 0.0;
  double pr_sum = 0.0;
  double pz_sum = 0.0;
  const int base = f * kCsrOptionBFaceCornerSlots + slot_base;
  for (int k = 0; k < nverts; ++k) {
    double dm = 0.0;
    double pr = 0.0;
    double pz = 0.0;
    if (k + 1 == nverts) {
      dm = dm_flux - dm_sum;
      pr = pr_flux - pr_sum;
      pz = pz_flux - pz_sum;
    } else {
      dm = dm_flux * w[k];
      pr = pr_flux * w[k];
      pz = pz_flux * w[k];
      dm_sum += dm;
      pr_sum += pr;
      pz_sum += pz;
    }
    face_delta_m[base + k] = sign * dm;
    face_delta_pr[base + k] = sign * pr;
    face_delta_pz[base + k] = sign * pz;
  }
}

__global__ void csr_optionb_compute_internal_face_flux_kernel(
    double* __restrict__ face_delta_m,
    double* __restrict__ face_delta_pr,
    double* __restrict__ face_delta_pz,
    const double* __restrict__ optionb_m_corner,
    const double* __restrict__ optionb_p_r,
    const double* __restrict__ optionb_p_z,
    const double* __restrict__ rho_lag,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ unique_cell_a,
    const int* __restrict__ unique_cell_b,
    const int* __restrict__ unique_local_a,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ node_flags,
    const double* __restrict__ mass_flux_scale,
    const int n_faces,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const std::uint8_t* __restrict__ donor_fallback_cell_mask,
    const std::uint8_t* __restrict__ discard_reference_inactive_cell_mask,
    double* __restrict__ discard_ledger,
    const double* __restrict__ basis_corner_mass,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const int n_cells,
    const bool print_centroid_far_diag,
    const bool ring5_trace_enabled,
    const int ring5_cell_start,
    const int ring5_cell_end,
    int* __restrict__ diagnostics,
    double* __restrict__ diagnostics_real,
    double* __restrict__ ring5_face_p,
    double* __restrict__ ring5_face_dm,
    double* __restrict__ ring5_face_u,
    int* __restrict__ ring5_face_status,
    int* __restrict__ ring5_face_centroid_class,
    int* __restrict__ ring5_face_cell_a,
    int* __restrict__ ring5_face_cell_b,
    int* __restrict__ ring5_face_donor,
    int* __restrict__ ring5_face_receiver,
    const RemapDispatchAuditDeviceView remap_dispatch_audit) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int cell_a = unique_cell_a[f];
  const int cell_b = unique_cell_b[f];
  const bool trace_face =
      ring5_trace_enabled &&
      csr_optionb_ring5_face(cell_a, cell_b, ring5_cell_start, ring5_cell_end);
  if (csr_face_touches_inactive(inactive_cell_mask, cell_a, cell_b)) {
    if (diagnostics != nullptr) {
      atomicAdd(diagnostics + kCsrOptionBDiagSkipped, 1);
    }
    csr_optionb_store_ring5_face_trace(trace_face,
                                       f,
                                       kCsrOptionBRing5FaceStatusSkippedInactive,
                                       0,
                                       cell_a,
                                       cell_b,
                                       -1,
                                       -1,
                                       0.0,
                                       0.0,
                                       0.0,
                                       ring5_face_p,
                                       ring5_face_dm,
                                       ring5_face_u,
                                       ring5_face_status,
                                       ring5_face_centroid_class,
                                       ring5_face_cell_a,
                                       ring5_face_cell_b,
                                       ring5_face_donor,
                                       ring5_face_receiver);
    return;
  }
  const int local_a = unique_local_a[f];
  double rq = 0.0;
  double zq = 0.0;
  detail::CsrFaceSweptMomentsStatus moment_status =
      detail::CsrFaceSweptMomentsStatus::SKIP;
  bool exact_moments = false;
  const double dV_a =
      ::tenryu::hydro::ale::csr_face_swept_moments_outward(
          x_r_old,
          x_z_old,
          x_r_new,
          x_z_new,
          cell_node_csr_offsets,
          cell_node_csr_indices,
          cell_orientation_sign,
          cell_a,
          local_a,
          cell_nverts,
          &rq,
          &zq,
          &moment_status,
          &exact_moments);
  if (moment_status != detail::CsrFaceSweptMomentsStatus::OK ||
      !finite_nonzero(dV_a)) {
    if (diagnostics != nullptr) {
      atomicAdd(diagnostics + kCsrOptionBDiagSkipped, 1);
    }
    csr_optionb_store_ring5_face_trace(trace_face,
                                       f,
                                       kCsrOptionBRing5FaceStatusSkippedZero,
                                       0,
                                       cell_a,
                                       cell_b,
                                       -1,
                                       -1,
                                       0.0,
                                       0.0,
                                       0.0,
                                       ring5_face_p,
                                       ring5_face_dm,
                                       ring5_face_u,
                                       ring5_face_status,
                                       ring5_face_centroid_class,
                                       ring5_face_cell_a,
                                       ring5_face_cell_b,
                                       ring5_face_donor,
                                       ring5_face_receiver);
    return;
  }

  const int donor = csr_internal_flux_donor(cell_a, cell_b, dV_a);
  const int receiver = (donor == cell_a) ? cell_b : cell_a;
  const int losing_cell = csr_internal_flux_losing_cell(cell_a, cell_b, dV_a);
  double dV_limited = dV_a;
  if (mass_flux_scale != nullptr) {
    double s = mass_flux_scale[losing_cell];
    if (!isfinite(s)) {
      s = 0.0;
    }
    dV_limited = dV_a * fmin(1.0, fmax(0.0, s));
  }
  const double dm_q = fmax(rho_lag[donor], 0.0) * fabs(dV_limited);
  if (!(dm_q > 0.0) || !isfinite(dm_q)) {
    if (diagnostics != nullptr) {
      atomicAdd(diagnostics + kCsrOptionBDiagSkipped, 1);
    }
    csr_optionb_store_ring5_face_trace(trace_face,
                                       f,
                                       kCsrOptionBRing5FaceStatusSkippedZero,
                                       0,
                                       cell_a,
                                       cell_b,
                                       donor,
                                       receiver,
                                       0.0,
                                       0.0,
                                       0.0,
                                       ring5_face_p,
                                       ring5_face_dm,
                                       ring5_face_u,
                                       ring5_face_status,
                                       ring5_face_centroid_class,
                                       ring5_face_cell_a,
                                       ring5_face_cell_b,
                                       ring5_face_donor,
                                       ring5_face_receiver);
    return;
  }
  remap_dispatch_audit_count(
      remap_dispatch_audit,
      exact_moments
          ? RemapDispatchAuditCounter::ExactSweptMoment
          : RemapDispatchAuditCounter::SweptCentroidAverageFallback,
      donor);

  const int donor_nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, donor);
  const int receiver_nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, receiver);
  double donor_r[kCsrOptionBFaceCornerSlots] = {};
  double donor_z[kCsrOptionBFaceCornerSlots] = {};
  double receiver_r[kCsrOptionBFaceCornerSlots] = {};
  double receiver_z[kCsrOptionBFaceCornerSlots] = {};
  double donor_ur[4] = {0.0, 0.0, 0.0, 0.0};
  double donor_uz[4] = {0.0, 0.0, 0.0, 0.0};
  tenryu::hydro::optionb::NodeVelocityProjector donor_projector[4] = {
      tenryu::hydro::optionb::NodeVelocityProjector::FREE,
      tenryu::hydro::optionb::NodeVelocityProjector::FREE,
      tenryu::hydro::optionb::NodeVelocityProjector::FREE,
      tenryu::hydro::optionb::NodeVelocityProjector::FREE};
  csr_optionb_cell_geometry(donor_r,
                            donor_z,
                            x_r_old,
                            x_z_old,
                            cell_node_csr_offsets,
                            cell_node_csr_indices,
                            cell_nverts,
                            donor,
                            donor_nverts);
  csr_optionb_cell_geometry(receiver_r,
                            receiver_z,
                            x_r_new,
                            x_z_new,
                            cell_node_csr_offsets,
                            cell_node_csr_indices,
                            cell_nverts,
                            receiver,
                            receiver_nverts);
  const int donor_off = cell_node_csr_offsets[donor];
  for (int k = 0; k < donor_nverts; ++k) {
    const int n = cell_node_csr_indices[donor_off + k];
    donor_ur[k] = v_r_node[n];
    donor_uz[k] = v_z_node[n];
    donor_projector[k] =
        node_flags == nullptr
            ? tenryu::hydro::optionb::NodeVelocityProjector::FREE
            : csr_optionb_projector_from_flags(node_flags[n]);
  }

  double packet_w[4] = {0.0, 0.0, 0.0, 0.0};
  const auto packet_weight_status =
      csr_optionb_packet_centroid_weights(
          donor_r, donor_z, donor_nverts, rq, zq, packet_w);
  double uhat_r = 0.0;
  double uhat_z = 0.0;
  int face_status = kCsrOptionBRing5FaceStatusPacketOk;
  const int centroid_class_for_trace =
      trace_face ? csr_optionb_centroid_out_class(donor_r,
                                                  donor_z,
                                                  donor_nverts,
                                                  receiver_r,
                                                  receiver_z,
                                                  receiver_nverts,
                                                  rq,
                                                  zq)
                 : 0;
  const bool donor_fallback =
      donor_fallback_cell_mask != nullptr &&
      donor_fallback_cell_mask[donor] != 0U;
  bool have_vhat = false;
  if (donor_fallback) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::FirstOrderDonorFallback,
        donor);
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::MomentumPacketFallback,
        donor);
    csr_optionb_live_cell_mean_velocity(optionb_m_corner + donor * 4,
                                        optionb_p_r + donor * 4,
                                        optionb_p_z + donor * 4,
                                        donor_nverts,
                                        &uhat_r,
                                        &uhat_z);
    face_status = kCsrOptionBRing5FaceStatusFallback;
    have_vhat = isfinite(uhat_r) && isfinite(uhat_z);
  } else if (csr_optionb_face_weights_usable(packet_weight_status)) {
    csr_optionb_normalize_face_weights(packet_w, donor_nverts);
    csr_optionb_velocity_from_face_weights(donor_ur,
                                           donor_uz,
                                           donor_projector,
                                           packet_w,
                                           donor_nverts,
                                           &uhat_r,
                                           &uhat_z);
    have_vhat = isfinite(uhat_r) && isfinite(uhat_z);
  }
  if (!have_vhat) {
    if (diagnostics != nullptr) {
      if (packet_weight_status ==
          CsrOptionBPacketCentroidWeightsStatus::INVALID_INPUT) {
        atomicAdd(diagnostics + kCsrOptionBDiagInvalid, 1);
      } else {
        atomicAdd(diagnostics + kCsrOptionBDiagExpanded, 1);
        csr_optionb_classify_expanded_need(rq,
                                           zq,
                                           donor_r,
                                           donor_z,
                                           donor_nverts,
                                           receiver_r,
                                           receiver_z,
                                           receiver_nverts,
                                           diagnostics);
        if (print_centroid_far_diag &&
            csr_optionb_centroid_out_class(donor_r,
                                           donor_z,
                                           donor_nverts,
                                           receiver_r,
                                           receiver_z,
                                           receiver_nverts,
                                           rq,
                                           zq) == 2) {
          csr_optionb_print_centroid_far_packet(diagnostics,
                                                print_centroid_far_diag,
                                                f,
                                                cell_a,
                                                cell_b,
                                                donor,
                                                receiver,
                                                local_a,
                                                dV_a,
                                                dm_q,
                                                rq,
                                                zq,
                                                donor_r,
                                                donor_z,
                                                donor_nverts,
                                                receiver_r,
                                                receiver_z,
                                                receiver_nverts,
                                                x_r_old,
                                                x_z_old,
                                                x_r_new,
                                                x_z_new,
                                                cell_node_csr_offsets,
                                                cell_node_csr_indices,
                                                cell_nverts);
        }
      }
    }
    int stencil_nodes[kCsrOptionBMaxExpandedNodes];
    double stencil_r[kCsrOptionBMaxExpandedNodes];
    double stencil_z[kCsrOptionBMaxExpandedNodes];
    double stencil_ur[kCsrOptionBMaxExpandedNodes];
    double stencil_uz[kCsrOptionBMaxExpandedNodes];
    int stencil_node_count = 0;
    int ring_used = 0;
    for (int ring = 1; ring <= kCsrOptionBMaxExpandedRing && !have_vhat;
         ++ring) {
      if (!csr_optionb_build_expanded_stencil(stencil_nodes,
                                              stencil_r,
                                              stencil_z,
                                              stencil_ur,
                                              stencil_uz,
                                              &stencil_node_count,
                                              donor,
                                              ring,
                                              x_r_old,
                                              x_z_old,
                                              v_r_node,
                                              v_z_node,
                                              cell_node_csr_offsets,
                                              cell_node_csr_indices,
                                              face_adj_csr_offsets,
                                              face_adj_csr_indices,
                                              cell_nverts,
                                              node_flags,
                                              inactive_cell_mask,
                                              n_cells)) {
        continue;
      }
      have_vhat = csr_optionb_interpolate_expanded_velocity(stencil_r,
                                                            stencil_z,
                                                            stencil_ur,
                                                            stencil_uz,
                                                            stencil_node_count,
                                                            rq,
                                                            zq,
                                                            &uhat_r,
                                                            &uhat_z) &&
                  isfinite(uhat_r) && isfinite(uhat_z);
      if (have_vhat) {
        ring_used = ring;
      }
    }
    if (have_vhat) {
      face_status = kCsrOptionBRing5FaceStatusExpandedOk;
      if (diagnostics != nullptr) {
        if (ring_used == 1) {
          atomicAdd(diagnostics + kCsrOptionBDiagExpandedRing1, 1);
        } else if (ring_used == 2) {
          atomicAdd(diagnostics + kCsrOptionBDiagExpandedRing2, 1);
        }
      }
    } else {
      remap_dispatch_audit_count(
          remap_dispatch_audit,
          RemapDispatchAuditCounter::FirstOrderDonorFallback,
          donor);
      remap_dispatch_audit_count(
          remap_dispatch_audit,
          RemapDispatchAuditCounter::MomentumPacketFallback,
          donor);
      remap_dispatch_audit_count(
          remap_dispatch_audit,
          RemapDispatchAuditCounter::MomentumExpandedFailureFallback,
          donor);
      if (packet_weight_status ==
          CsrOptionBPacketCentroidWeightsStatus::INVALID_INPUT) {
        remap_dispatch_audit_count(
            remap_dispatch_audit,
            RemapDispatchAuditCounter::MomentumInvalidInputFallback,
            donor);
      }
      if (diagnostics != nullptr) {
        atomicAdd(diagnostics + kCsrOptionBDiagExpandedFailed, 1);
        atomicAdd(diagnostics + kCsrOptionBDiagFallback, 1);
      }
      double fallback_w[4] = {0.0, 0.0, 0.0, 0.0};
      if (!csr_optionb_basis_face_weights(
              basis_corner_mass, donor, donor_nverts, fallback_w)) {
        csr_optionb_unit_corner_weights(
            donor_r, donor_z, donor_nverts, fallback_w);
      }
      csr_optionb_normalize_face_weights(fallback_w, donor_nverts);
      if (optionb_m_corner != nullptr && optionb_p_r != nullptr &&
          optionb_p_z != nullptr) {
        csr_optionb_live_cell_mean_velocity(optionb_m_corner + donor * 4,
                                            optionb_p_r + donor * 4,
                                            optionb_p_z + donor * 4,
                                            donor_nverts,
                                            &uhat_r,
                                            &uhat_z);
      }
      if (!isfinite(uhat_r) || !isfinite(uhat_z)) {
        csr_optionb_velocity_from_face_weights(donor_ur,
                                               donor_uz,
                                               donor_projector,
                                               fallback_w,
                                               donor_nverts,
                                               &uhat_r,
                                               &uhat_z);
      }
      face_status = kCsrOptionBRing5FaceStatusFallback;
    }
  }

  if (!isfinite(uhat_r) || !isfinite(uhat_z)) {
    if (diagnostics != nullptr) {
      atomicAdd(diagnostics + kCsrOptionBDiagInvalid, 1);
    }
    return;
  }
  csr_optionb_store_ring5_face_trace(trace_face,
                                     f,
                                     face_status,
                                     centroid_class_for_trace,
                                     cell_a,
                                     cell_b,
                                     donor,
                                     receiver,
                                     dm_q,
                                     uhat_r,
                                     uhat_z,
                                     ring5_face_p,
                                     ring5_face_dm,
                                     ring5_face_u,
                                     ring5_face_status,
                                     ring5_face_centroid_class,
                                     ring5_face_cell_a,
                                     ring5_face_cell_b,
                                     ring5_face_donor,
                                     ring5_face_receiver);
  csr_optionb_note_discarded_dual_flux(cell_a,
                                       cell_b,
                                       donor,
                                       receiver,
                                       dm_q,
                                       uhat_r,
                                       uhat_z,
                                       discard_reference_inactive_cell_mask,
                                       discard_ledger);
  csr_optionb_atomic_min_double(
      diagnostics_real != nullptr
          ? diagnostics_real + kCsrOptionBDiagRealAlphaMin
          : nullptr,
      1.0);

  double donor_w[4] = {0.0, 0.0, 0.0, 0.0};
  if (donor_fallback) {
    for (int k = 0; k < donor_nverts; ++k) {
      donor_w[k] = optionb_m_corner[donor * 4 + k];
    }
  } else if (!csr_optionb_basis_face_weights(
                 basis_corner_mass, donor, donor_nverts, donor_w)) {
    if (csr_optionb_face_weights_usable(packet_weight_status)) {
      for (int k = 0; k < donor_nverts; ++k) {
        donor_w[k] = packet_w[k];
      }
    } else {
      csr_optionb_unit_corner_weights(donor_r, donor_z, donor_nverts, donor_w);
    }
  }
  csr_optionb_normalize_face_weights(donor_w, donor_nverts);
  double receiver_w[4] = {0.0, 0.0, 0.0, 0.0};
  const auto receiver_weight_status = donor_fallback
      ? CsrOptionBPacketCentroidWeightsStatus::INVALID_INPUT
      : csr_optionb_packet_centroid_weights(receiver_r,
                                            receiver_z,
                                            receiver_nverts,
                                            rq,
                                            zq,
                                            receiver_w);
  if (donor_fallback ||
      !csr_optionb_face_weights_usable(receiver_weight_status)) {
    csr_optionb_unit_corner_weights(
        receiver_r, receiver_z, receiver_nverts, receiver_w);
  }
  csr_optionb_normalize_face_weights(receiver_w, receiver_nverts);

  const double pr_flux = dm_q * uhat_r;
  const double pz_flux = dm_q * uhat_z;
  const double sign_a = (cell_a == donor) ? -1.0 : 1.0;
  const double sign_b = -sign_a;
  csr_optionb_store_face_side_flux(face_delta_m,
                                   face_delta_pr,
                                   face_delta_pz,
                                   f,
                                   0,
                                   tenryu::mesh::mesh_topo_cell_active_nverts(
                                       cell_nverts, cell_a),
                                   sign_a,
                                   dm_q,
                                   pr_flux,
                                   pz_flux,
                                   (cell_a == donor) ? donor_w : receiver_w);
  csr_optionb_store_face_side_flux(face_delta_m,
                                   face_delta_pr,
                                   face_delta_pz,
                                   f,
                                   4,
                                   tenryu::mesh::mesh_topo_cell_active_nverts(
                                       cell_nverts, cell_b),
                                   sign_b,
                                   dm_q,
                                   pr_flux,
                                   pz_flux,
                                   (cell_b == donor) ? donor_w : receiver_w);
}

__global__ void csr_optionb_gather_internal_face_fluxes_kernel(
    double* __restrict__ optionb_m_corner,
    double* __restrict__ optionb_p_r,
    double* __restrict__ optionb_p_z,
    const double* __restrict__ face_delta_m,
    const double* __restrict__ face_delta_pr,
    const double* __restrict__ face_delta_pz,
    const int* __restrict__ unique_cell_a,
    const int* __restrict__ unique_cell_b,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_faces,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells || csr_inactive_cell(inactive_cell_mask, c)) {
    return;
  }
  const int nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  const int cell_base = c * 4;
  for (int f = 0; f < n_faces; ++f) {
    int slot_base = -1;
    if (unique_cell_a[f] == c) {
      slot_base = 0;
    } else if (unique_cell_b[f] == c) {
      slot_base = 4;
    }
    if (slot_base < 0) {
      continue;
    }
    const int face_base = f * kCsrOptionBFaceCornerSlots + slot_base;
    for (int k = 0; k < nverts; ++k) {
      optionb_m_corner[cell_base + k] += face_delta_m[face_base + k];
      optionb_p_r[cell_base + k] += face_delta_pr[face_base + k];
      optionb_p_z[cell_base + k] += face_delta_pz[face_base + k];
    }
  }
}

__global__ void csr_optionb_apply_boundary_packets_serial_kernel(
    double* __restrict__ optionb_m_corner,
    double* __restrict__ optionb_p_r,
    double* __restrict__ optionb_p_z,
    const double* __restrict__ rho_lag,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ boundary_cell,
    const int* __restrict__ boundary_local,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ mass_flux_scale,
    const int n_faces,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    int* __restrict__ diagnostics,
    const RemapDispatchAuditDeviceView remap_dispatch_audit) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int f = 0; f < n_faces; ++f) {
    const int cell = boundary_cell[f];
    if (csr_inactive_cell(inactive_cell_mask, cell)) {
      if (diagnostics != nullptr) {
        atomicAdd(diagnostics + kCsrOptionBDiagSkipped, 1);
      }
      continue;
    }
    const int local = boundary_local[f];
    double rq = 0.0;
    double zq = 0.0;
    detail::CsrFaceSweptMomentsStatus moment_status =
        detail::CsrFaceSweptMomentsStatus::SKIP;
    bool exact_moments = false;
    const double dV =
        ::tenryu::hydro::ale::csr_face_swept_moments_outward(
            x_r_old,
            x_z_old,
            x_r_new,
            x_z_new,
            cell_node_csr_offsets,
            cell_node_csr_indices,
            cell_orientation_sign,
            cell,
            local,
            cell_nverts,
            &rq,
            &zq,
            &moment_status,
            &exact_moments);
    (void)rq;
    (void)zq;
    if (moment_status != detail::CsrFaceSweptMomentsStatus::OK ||
        !finite_nonzero(dV)) {
      if (diagnostics != nullptr) {
        atomicAdd(diagnostics + kCsrOptionBDiagSkipped, 1);
      }
      continue;
    }
    double dV_limited = dV;
    if (mass_flux_scale != nullptr && dV < 0.0) {
      double s = mass_flux_scale[cell];
      if (!isfinite(s)) {
        s = 0.0;
      }
      dV_limited = dV * fmin(1.0, fmax(0.0, s));
    }
    const double dm_signed = fmax(rho_lag[cell], 0.0) * dV_limited;
    if (!finite_nonzero(dm_signed)) {
      continue;
    }
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        exact_moments
            ? RemapDispatchAuditCounter::ExactSweptMoment
            : RemapDispatchAuditCounter::SweptCentroidAverageFallback,
        cell);
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::BoundaryOneSided,
        cell);
    const int nverts =
        tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, cell);
    double target_r[kCsrOptionBFaceCornerSlots] = {};
    double target_z[kCsrOptionBFaceCornerSlots] = {};
    csr_optionb_cell_geometry(target_r,
                              target_z,
                              x_r_new,
                              x_z_new,
                              cell_node_csr_offsets,
                              cell_node_csr_indices,
                              cell_nverts,
                              cell,
                              nverts);
    csr_optionb_apply_boundary_mass_change(dm_signed,
                                           target_r,
                                           target_z,
                                           nverts,
                                           optionb_m_corner + cell * 4,
                                           optionb_p_r + cell * 4,
                                           optionb_p_z + cell * 4);
  }
}

__global__ void csr_optionb_add_floor_mass_kernel(
    double* __restrict__ optionb_m_corner,
    double* __restrict__ optionb_p_r,
    double* __restrict__ optionb_p_z,
    double* __restrict__ cell_mass_out,
    const double* __restrict__ target_cell_mass,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const double* __restrict__ vol_new,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ target_cell_mass_mask,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells,
    const double rho_floor,
    const bool preserve_momentum_on_target_scale) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    if (cell_mass_out != nullptr) {
      cell_mass_out[c] = 0.0;
    }
    return;
  }
  const int nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  double m_sum = 0.0;
  for (int k = 0; k < nverts; ++k) {
    m_sum += optionb_m_corner[c * 4 + k];
  }
  const double V = (vol_new != nullptr) ? fmax(vol_new[c], kTinyVolume) : 0.0;
  const double floor_mass = fmax(rho_floor, 0.0) * V;
  const bool use_target_mass =
      target_cell_mass != nullptr &&
      (target_cell_mass_mask == nullptr ||
       target_cell_mass_mask[c] != static_cast<std::uint8_t>(0));
  const double target_mass =
      use_target_mass && isfinite(target_cell_mass[c])
          ? fmax(target_cell_mass[c], 0.0)
          : floor_mass;
  if (use_target_mass && target_mass < m_sum && m_sum > 0.0 &&
      isfinite(target_mass) && isfinite(m_sum)) {
    const double scale = target_mass / m_sum;
    for (int k = 0; k < nverts; ++k) {
      optionb_m_corner[c * 4 + k] *= scale;
      if (!preserve_momentum_on_target_scale) {
        optionb_p_r[c * 4 + k] *= scale;
        optionb_p_z[c * 4 + k] *= scale;
      }
    }
    m_sum = target_mass;
  }
  if (target_mass > m_sum && isfinite(target_mass)) {
    double r[kCsrOptionBFaceCornerSlots] = {};
    double z[kCsrOptionBFaceCornerSlots] = {};
    double w[4] = {0.0, 0.0, 0.0, 0.0};
    csr_optionb_cell_geometry(r,
                              z,
                              x_r_new,
                              x_z_new,
                              cell_node_csr_offsets,
                              cell_node_csr_indices,
                              cell_nverts,
                              c,
                              nverts);
    csr_optionb_unit_corner_weights(r, z, nverts, w);
    const double dm_floor = target_mass - m_sum;
    for (int k = 0; k < nverts; ++k) {
      optionb_m_corner[c * 4 + k] += dm_floor * w[k];
    }
    m_sum = target_mass;
  }
  if (cell_mass_out != nullptr) {
    cell_mass_out[c] = m_sum;
  }
}

__global__ void csr_optionb_hourglass_filter_kernel(
    double* __restrict__ optionb_m_corner,
    double* __restrict__ optionb_p_r,
    double* __restrict__ optionb_p_z,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    int* __restrict__ diagnostics) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    return;
  }
  const int nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  double r[kCsrOptionBFaceCornerSlots] = {};
  double z[kCsrOptionBFaceCornerSlots] = {};
  double v_scratch[4] = {0.0, 0.0, 0.0, 0.0};
  double v_aff_scratch[4] = {0.0, 0.0, 0.0, 0.0};
  double h_scratch[4] = {0.0, 0.0, 0.0, 0.0};
  csr_optionb_cell_geometry(r,
                            z,
                            x_r_new,
                            x_z_new,
                            cell_node_csr_offsets,
                            cell_node_csr_indices,
                            cell_nverts,
                            c,
                            nverts);
  const auto result =
      tenryu::hydro::optionb::apply_affine_orthogonal_hourglass_filter(
          r,
          z,
          nverts,
          optionb_m_corner + c * 4,
          optionb_p_r + c * 4,
          optionb_p_z + c * 4,
          0.0,
          v_scratch,
          v_aff_scratch,
          h_scratch);
  if (diagnostics == nullptr) {
    return;
  }
  if (result.ur.status ==
          tenryu::hydro::optionb::AffineHourglassFilterStatus::INVALID_INPUT ||
      result.uz.status ==
          tenryu::hydro::optionb::AffineHourglassFilterStatus::INVALID_INPUT) {
    atomicAdd(diagnostics + kCsrOptionBDiagFilterInvalid, 1);
  }
  if (result.ur.status ==
          tenryu::hydro::optionb::AffineHourglassFilterStatus::DEGENERATE_BASIS ||
      result.uz.status ==
          tenryu::hydro::optionb::AffineHourglassFilterStatus::DEGENERATE_BASIS) {
    atomicAdd(diagnostics + kCsrOptionBDiagFilterDegenerate, 1);
  }
}

__device__ inline void csr_optionb_include_velocity_bound(
    const double ur,
    const double uz,
    double* __restrict__ ur_min,
    double* __restrict__ ur_max,
    double* __restrict__ uz_min,
    double* __restrict__ uz_max,
    bool* __restrict__ have_bounds) {
  if (!isfinite(ur) || !isfinite(uz)) {
    return;
  }
  if (!*have_bounds) {
    *ur_min = ur;
    *ur_max = ur;
    *uz_min = uz;
    *uz_max = uz;
    *have_bounds = true;
    return;
  }
  *ur_min = fmin(*ur_min, ur);
  *ur_max = fmax(*ur_max, ur);
  *uz_min = fmin(*uz_min, uz);
  *uz_max = fmax(*uz_max, uz);
}

__device__ inline double csr_optionb_velocity_bound_tol(const double lo,
                                                        const double hi) {
  return 1024.0 * tenryu::hydro::optionb::detail::kDoubleEps *
         fmax(1.0, fmax(fabs(lo), fabs(hi)));
}

__device__ inline bool csr_optionb_velocity_inside_bounds(
    const double ur,
    const double uz,
    const double ur_min,
    const double ur_max,
    const double uz_min,
    const double uz_max,
    const bool have_bounds) {
  if (!have_bounds || !isfinite(ur) || !isfinite(uz)) {
    return false;
  }
  const double tol_r = csr_optionb_velocity_bound_tol(ur_min, ur_max);
  const double tol_z = csr_optionb_velocity_bound_tol(uz_min, uz_max);
  return ur >= ur_min - tol_r && ur <= ur_max + tol_r &&
         uz >= uz_min - tol_z && uz <= uz_max + tol_z;
}

__device__ inline double csr_optionb_clamp_velocity_component(
    const double value,
    const double lo,
    const double hi) {
  if (!isfinite(value)) {
    return 0.0;
  }
  const double tol = csr_optionb_velocity_bound_tol(lo, hi);
  if (value < lo - tol) {
    return lo;
  }
  if (value > hi + tol) {
    return hi;
  }
  return value;
}

__device__ inline bool csr_optionb_assembly_cell_active(
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const std::uint8_t* __restrict__ assembly_cell_mask,
    const int c);

__global__ void csr_optionb_scatter_nodal_velocity_kernel(
    double* __restrict__ node_mass,
    double* __restrict__ node_p_r,
    double* __restrict__ node_p_z,
    double* __restrict__ v_r_out,
    double* __restrict__ v_z_out,
    const double* __restrict__ optionb_m_corner,
    const double* __restrict__ optionb_p_r,
    const double* __restrict__ optionb_p_z,
    const int* __restrict__ reverse_csr_node_offsets,
    const int* __restrict__ reverse_csr_node_cells,
    const int* __restrict__ reverse_csr_node_corners,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ node_flags,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const std::uint8_t* __restrict__ assembly_cell_mask,
    const double* __restrict__ v_r_source,
    const double* __restrict__ v_z_source,
    const int n_nodes,
    const int n_cells) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  const int off = reverse_csr_node_offsets[n];
  const int end = reverse_csr_node_offsets[n + 1];
  double m_sum = 0.0;
  double pr_sum = 0.0;
  double pz_sum = 0.0;
  for (int p = off; p < end; ++p) {
    const int c = reverse_csr_node_cells[p];
    if (!csr_optionb_assembly_cell_active(inactive_cell_mask,
                                          assembly_cell_mask,
                                          c)) {
      continue;
    }
    const int corner = reverse_csr_node_corners[p];
    const int nverts =
        tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
    if (corner < 0 || corner >= nverts) {
      continue;
    }
    const int idx = c * 4 + corner;
    m_sum += optionb_m_corner[idx];
    pr_sum += optionb_p_r[idx];
    pz_sum += optionb_p_z[idx];
  }
  const auto projector =
      node_flags == nullptr
          ? tenryu::hydro::optionb::NodeVelocityProjector::FREE
          : csr_optionb_projector_from_flags(node_flags[n]);
  double ur_min = 0.0;
  double ur_max = 0.0;
  double uz_min = 0.0;
  double uz_max = 0.0;
  bool have_bounds = false;
  const bool exact_recovery = assembly_cell_mask != nullptr;
  if (!exact_recovery && v_r_source != nullptr && v_z_source != nullptr) {
    double bound_ur = isfinite(v_r_source[n]) ? v_r_source[n] : 0.0;
    double bound_uz = isfinite(v_z_source[n]) ? v_z_source[n] : 0.0;
    tenryu::hydro::optionb::apply_node_velocity_projector(
        projector, &bound_ur, &bound_uz);
    csr_optionb_include_velocity_bound(bound_ur,
                                       bound_uz,
                                       &ur_min,
                                       &ur_max,
                                       &uz_min,
                                       &uz_max,
                                       &have_bounds);
    if (cell_node_csr_offsets != nullptr && cell_node_csr_indices != nullptr) {
      for (int p = off; p < end; ++p) {
        const int c = reverse_csr_node_cells[p];
        if (c < 0 || c >= n_cells) {
          continue;
        }
        const int cell_off = cell_node_csr_offsets[c];
        const int nverts =
            tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
        for (int k = 0; k < nverts; ++k) {
          const int node = cell_node_csr_indices[cell_off + k];
          if (node < 0 || node >= n_nodes) {
            continue;
          }
          bound_ur = isfinite(v_r_source[node]) ? v_r_source[node] : 0.0;
          bound_uz = isfinite(v_z_source[node]) ? v_z_source[node] : 0.0;
          const auto bound_projector =
              node_flags == nullptr
                  ? tenryu::hydro::optionb::NodeVelocityProjector::FREE
                  : csr_optionb_projector_from_flags(node_flags[node]);
          tenryu::hydro::optionb::apply_node_velocity_projector(
              bound_projector, &bound_ur, &bound_uz);
          csr_optionb_include_velocity_bound(bound_ur,
                                             bound_uz,
                                             &ur_min,
                                             &ur_max,
                                             &uz_min,
                                             &uz_max,
                                             &have_bounds);
        }
      }
    }
  }
  double ur = 0.0;
  double uz = 0.0;
  const bool have_scatter_mass = m_sum > 0.0 && isfinite(m_sum);
  if (have_scatter_mass) {
    ur = pr_sum / m_sum;
    uz = pz_sum / m_sum;
    tenryu::hydro::optionb::apply_node_velocity_projector(
        projector, &ur, &uz);
    if (!exact_recovery && have_bounds) {
      ur = csr_optionb_clamp_velocity_component(ur, ur_min, ur_max);
      uz = csr_optionb_clamp_velocity_component(uz, uz_min, uz_max);
    }
    pr_sum = m_sum * ur;
    pz_sum = m_sum * uz;
  } else {
    pr_sum = 0.0;
    pz_sum = 0.0;
  }
  node_mass[n] = m_sum;
  node_p_r[n] = pr_sum;
  node_p_z[n] = pz_sum;
  v_r_out[n] = ur;
  v_z_out[n] = uz;
}

__global__ void csr_optionb_axis_trace_velocity_kernel(
    double* __restrict__ node_p_r,
    double* __restrict__ node_p_z,
    double* __restrict__ v_r_out,
    double* __restrict__ v_z_out,
    const double* __restrict__ node_mass,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int* __restrict__ reverse_csr_node_offsets,
    const int* __restrict__ reverse_csr_node_cells,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ node_flags,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const std::uint8_t* __restrict__ active_node_velocity_mask,
    const int n_nodes,
    const int n_cells) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes || node_flags == nullptr) {
    return;
  }
  if (active_node_velocity_mask != nullptr &&
      active_node_velocity_mask[n] == static_cast<std::uint8_t>(0)) {
    return;
  }
  if (csr_optionb_projector_from_flags(node_flags[n]) !=
      tenryu::hydro::optionb::NodeVelocityProjector::RZ_AXIS) {
    return;
  }
  // r=0 regularity trace (Kenamond/Barlow-Shashkov RZ-compatible axis
  // treatment): the physical RZ nodal mass vanishes on the axis
  // (m ~ 2*pi*r), so the scatter's u = p/m is a singular ratio there and
  // amplifies any unpaired flux residual. The axis velocity is therefore
  // recovered as the r->0 limit of the adjacent off-axis flow: an affine
  // LSQ fit u_z(r,z) = a + b*r + c*(z - z_n) over the off-axis nodes of the
  // incident active cells (plus their face-adjacent active cells),
  // evaluated at (r=0, z=z_n). Exact on affine velocity fields.
  const double z_n = x_z_new[n];
  double cnt = 0.0;
  double s_r = 0.0, s_z = 0.0, s_rr = 0.0, s_zz = 0.0, s_rz = 0.0;
  double s_u = 0.0, s_ur = 0.0, s_uz = 0.0;
  double r_max = 0.0, z_max = 0.0;
  const int off = reverse_csr_node_offsets[n];
  const int end = reverse_csr_node_offsets[n + 1];
  for (int p = off; p < end; ++p) {
    const int c0 = reverse_csr_node_cells[p];
    if (c0 < 0 || c0 >= n_cells) {
      continue;
    }
    const int adj_off =
        face_adj_csr_offsets != nullptr ? face_adj_csr_offsets[c0] : 0;
    const int adj_end =
        face_adj_csr_offsets != nullptr ? face_adj_csr_offsets[c0 + 1] : 0;
    for (int q = adj_off - 1; q < adj_end; ++q) {
      const int c = (q < adj_off)
                        ? c0
                        : (face_adj_csr_indices != nullptr
                               ? face_adj_csr_indices[q]
                               : -1);
      if (c < 0 || c >= n_cells ||
          csr_inactive_cell(inactive_cell_mask, c)) {
        continue;
      }
      const int cell_off = cell_node_csr_offsets[c];
      const int nverts =
          tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
      for (int k = 0; k < nverts; ++k) {
        const int m = cell_node_csr_indices[cell_off + k];
        if (m < 0 || m >= n_nodes) {
          continue;
        }
        if (csr_optionb_projector_from_flags(node_flags[m]) !=
            tenryu::hydro::optionb::NodeVelocityProjector::FREE) {
          continue;
        }
        const double r = x_r_new[m];
        const double dz = x_z_new[m] - z_n;
        const double u = v_z_out[m];
        if (!(r > 0.0) || !isfinite(dz) || !isfinite(u) ||
            !(node_mass[m] > 0.0)) {
          continue;
        }
        cnt += 1.0;
        s_r += r;
        s_z += dz;
        s_rr += r * r;
        s_zz += dz * dz;
        s_rz += r * dz;
        s_u += u;
        s_ur += u * r;
        s_uz += u * dz;
        r_max = fmax(r_max, r);
        z_max = fmax(z_max, fabs(dz));
      }
    }
  }
  if (cnt <= 0.0) {
    return;
  }
  double uz = s_u / cnt;
  if (cnt >= 3.0 && r_max > 0.0 && z_max > 0.0) {
    const double a11 = cnt;
    const double a12 = s_r / r_max;
    const double a13 = s_z / z_max;
    const double a22 = s_rr / (r_max * r_max);
    const double a23 = s_rz / (r_max * z_max);
    const double a33 = s_zz / (z_max * z_max);
    const double b1 = s_u;
    const double b2 = s_ur / r_max;
    const double b3 = s_uz / z_max;
    const double det = a11 * (a22 * a33 - a23 * a23) -
                       a12 * (a12 * a33 - a13 * a23) +
                       a13 * (a12 * a23 - a13 * a22);
    if (fabs(det) > 1e-12 * cnt * cnt * cnt) {
      const double det_a = b1 * (a22 * a33 - a23 * a23) -
                           b2 * (a12 * a33 - a13 * a23) +
                           b3 * (a12 * a23 - a13 * a22);
      const double fitted = det_a / det;
      if (isfinite(fitted)) {
        uz = fitted;
      }
    }
  }
  const double m_node =
      (node_mass[n] > 0.0 && isfinite(node_mass[n])) ? node_mass[n] : 0.0;
  v_r_out[n] = 0.0;
  v_z_out[n] = uz;
  node_p_r[n] = 0.0;
  node_p_z[n] = m_node * uz;
}

__global__ void csr_optionb_macroboundary_reconstruct_velocity_kernel(
    double* __restrict__ node_p_r,
    double* __restrict__ node_p_z,
    double* __restrict__ v_r_out,
    double* __restrict__ v_z_out,
    const double* __restrict__ node_mass,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const double* __restrict__ v_r_source,
    const double* __restrict__ v_z_source,
    const int* __restrict__ reverse_csr_node_offsets,
    const int* __restrict__ reverse_csr_node_cells,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ node_flags,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const std::uint8_t* __restrict__ macro_boundary_node_mask,
    const std::uint8_t* __restrict__ active_node_velocity_mask,
    const int n_nodes,
    const int n_cells) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes || macro_boundary_node_mask == nullptr ||
      macro_boundary_node_mask[n] == 0U) {
    return;
  }
  if (active_node_velocity_mask != nullptr &&
      active_node_velocity_mask[n] == static_cast<std::uint8_t>(0)) {
    return;
  }
  const int off = reverse_csr_node_offsets[n];
  const int end = reverse_csr_node_offsets[n + 1];
  const auto projector =
      node_flags == nullptr
          ? tenryu::hydro::optionb::NodeVelocityProjector::FREE
          : csr_optionb_projector_from_flags(node_flags[n]);
  double macro_ur = isfinite(v_r_source[n]) ? v_r_source[n] : 0.0;
  double macro_uz = isfinite(v_z_source[n]) ? v_z_source[n] : 0.0;
  tenryu::hydro::optionb::apply_node_velocity_projector(
      projector, &macro_ur, &macro_uz);
  double ur_min = 0.0;
  double ur_max = 0.0;
  double uz_min = 0.0;
  double uz_max = 0.0;
  bool have_bounds = false;
  csr_optionb_include_velocity_bound(macro_ur,
                                     macro_uz,
                                     &ur_min,
                                     &ur_max,
                                     &uz_min,
                                     &uz_max,
                                     &have_bounds);
  for (int p = off; p < end; ++p) {
    const int c = reverse_csr_node_cells[p];
    if (c < 0 || c >= n_cells || csr_inactive_cell(inactive_cell_mask, c)) {
      continue;
    }
    const int cell_off = cell_node_csr_offsets[c];
    const int nverts =
        tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
    for (int k = 0; k < nverts; ++k) {
      const int node = cell_node_csr_indices[cell_off + k];
      if (node < 0) {
        continue;
      }
      double bound_ur = isfinite(v_r_source[node]) ? v_r_source[node] : 0.0;
      double bound_uz = isfinite(v_z_source[node]) ? v_z_source[node] : 0.0;
      const auto bound_projector =
          node_flags == nullptr
              ? tenryu::hydro::optionb::NodeVelocityProjector::FREE
              : csr_optionb_projector_from_flags(node_flags[node]);
      tenryu::hydro::optionb::apply_node_velocity_projector(
          bound_projector, &bound_ur, &bound_uz);
      csr_optionb_include_velocity_bound(bound_ur,
                                         bound_uz,
                                         &ur_min,
                                         &ur_max,
                                         &uz_min,
                                         &uz_max,
                                         &have_bounds);
    }
  }
  double ur_sum = 0.0;
  double uz_sum = 0.0;
  int count = 0;
  for (int p = off; p < end; ++p) {
    const int c = reverse_csr_node_cells[p];
    if (c < 0 || c >= n_cells || csr_inactive_cell(inactive_cell_mask, c)) {
      continue;
    }
    int stencil_nodes[kCsrOptionBMaxExpandedNodes];
    double stencil_r[kCsrOptionBMaxExpandedNodes];
    double stencil_z[kCsrOptionBMaxExpandedNodes];
    double stencil_ur[kCsrOptionBMaxExpandedNodes];
    double stencil_uz[kCsrOptionBMaxExpandedNodes];
    int stencil_node_count = 0;
    if (!csr_optionb_build_expanded_stencil(stencil_nodes,
                                            stencil_r,
                                            stencil_z,
                                            stencil_ur,
                                            stencil_uz,
                                            &stencil_node_count,
                                            c,
                                            kCsrOptionBMaxExpandedRing,
                                            x_r_old,
                                            x_z_old,
                                            v_r_source,
                                            v_z_source,
                                            cell_node_csr_offsets,
                                            cell_node_csr_indices,
                                            face_adj_csr_offsets,
                                            face_adj_csr_indices,
                                            cell_nverts,
                                            node_flags,
                                            inactive_cell_mask,
                                            n_cells)) {
      continue;
    }
    double ur = 0.0;
    double uz = 0.0;
    if (!csr_optionb_interpolate_expanded_velocity(stencil_r,
                                                   stencil_z,
                                                   stencil_ur,
                                                   stencil_uz,
                                                   stencil_node_count,
                                                   x_r_new[n],
                                                   x_z_new[n],
                                                   &ur,
                                                   &uz)) {
      continue;
    }
    if (!isfinite(ur) || !isfinite(uz)) {
      continue;
    }
    if (!csr_optionb_velocity_inside_bounds(
            ur, uz, ur_min, ur_max, uz_min, uz_max, have_bounds)) {
      continue;
    }
    ur_sum += ur;
    uz_sum += uz;
    ++count;
  }
  double ur = (count > 0) ? ur_sum / static_cast<double>(count) : macro_ur;
  double uz = (count > 0) ? uz_sum / static_cast<double>(count) : macro_uz;
  if (have_bounds) {
    ur = csr_optionb_clamp_velocity_component(ur, ur_min, ur_max);
    uz = csr_optionb_clamp_velocity_component(uz, uz_min, uz_max);
  }
  tenryu::hydro::optionb::apply_node_velocity_projector(projector, &ur, &uz);
  const double m =
      (node_mass[n] > 0.0 && isfinite(node_mass[n])) ? node_mass[n] : 0.0;
  v_r_out[n] = ur;
  v_z_out[n] = uz;
  node_p_r[n] = m * ur;
  node_p_z[n] = m * uz;
}

__global__ void csr_optionb_fill_homologous_velocity_kernel(
    double* __restrict__ v_r_out,
    double* __restrict__ v_z_out,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const std::uint8_t* __restrict__ node_flags,
    const double H,
    const double x0_r,
    const double x0_z,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  double ur = -H * (x_r[n] - x0_r);
  double uz = -H * (x_z[n] - x0_z);
  const auto projector =
      node_flags == nullptr
          ? tenryu::hydro::optionb::NodeVelocityProjector::FREE
          : csr_optionb_projector_from_flags(node_flags[n]);
  tenryu::hydro::optionb::apply_node_velocity_projector(projector, &ur, &uz);
  v_r_out[n] = ur;
  v_z_out[n] = uz;
}

__device__ inline bool csr_optionb_node_touches_cell_range(
    const int node,
    const int* __restrict__ reverse_csr_node_offsets,
    const int* __restrict__ reverse_csr_node_cells,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int cell_start,
    const int cell_end) {
  const int off = reverse_csr_node_offsets[node];
  const int end = reverse_csr_node_offsets[node + 1];
  for (int p = off; p < end; ++p) {
    const int c = reverse_csr_node_cells[p];
    if (c >= cell_start && c <= cell_end &&
        !csr_inactive_cell(inactive_cell_mask, c)) {
      return true;
    }
  }
  return false;
}

__global__ void csr_optionb_macroboundary_h_reduce_kernel(
    double* __restrict__ sums,
    const double* __restrict__ node_mass,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ reverse_csr_node_offsets,
    const int* __restrict__ reverse_csr_node_cells,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_nodes,
    const int cell_start,
    const int cell_end,
    const double x0_r,
    const double x0_z) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes ||
      !csr_optionb_node_touches_cell_range(n,
                                           reverse_csr_node_offsets,
                                           reverse_csr_node_cells,
                                           inactive_cell_mask,
                                           cell_start,
                                           cell_end)) {
    return;
  }
  const double mu =
      (node_mass[n] > 0.0 && isfinite(node_mass[n])) ? node_mass[n] : 0.0;
  if (!(mu > 0.0)) {
    return;
  }
  const double xr = x_r[n] - x0_r;
  const double xz = x_z[n] - x0_z;
  const double dot = v_r[n] * xr + v_z[n] * xz;
  const double xx = xr * xr + xz * xz;
  if (!isfinite(dot) || !isfinite(xx)) {
    return;
  }
  atomicAdd(sums + 0, mu * dot);
  atomicAdd(sums + 1, mu * xx);
  atomicAdd(sums + 2, mu);
  atomicAdd(sums + 3, 1.0);
}

__global__ void csr_optionb_macroboundary_residual_reduce_kernel(
    double* __restrict__ sums,
    const double* __restrict__ node_mass,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const std::uint8_t* __restrict__ node_flags,
    const int* __restrict__ reverse_csr_node_offsets,
    const int* __restrict__ reverse_csr_node_cells,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_nodes,
    const int cell_start,
    const int cell_end,
    const double H,
    const double x0_r,
    const double x0_z) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes ||
      !csr_optionb_node_touches_cell_range(n,
                                           reverse_csr_node_offsets,
                                           reverse_csr_node_cells,
                                           inactive_cell_mask,
                                           cell_start,
                                           cell_end)) {
    return;
  }
  const double mu =
      (node_mass[n] > 0.0 && isfinite(node_mass[n])) ? node_mass[n] : 0.0;
  if (!(mu > 0.0)) {
    return;
  }
  const double xr = x_r[n] - x0_r;
  const double xz = x_z[n] - x0_z;
  double expected_r = -H * xr;
  double expected_z = -H * xz;
  const auto projector =
      node_flags == nullptr
          ? tenryu::hydro::optionb::NodeVelocityProjector::FREE
          : csr_optionb_projector_from_flags(node_flags[n]);
  tenryu::hydro::optionb::apply_node_velocity_projector(
      projector, &expected_r, &expected_z);
  const double dr = v_r[n] - expected_r;
  const double dz = v_z[n] - expected_z;
  const double denom = expected_r * expected_r + expected_z * expected_z;
  const double resid = dr * dr + dz * dz;
  if (!isfinite(resid) || !isfinite(denom)) {
    return;
  }
  atomicAdd(sums + 0, mu * resid);
  atomicAdd(sums + 1, mu * denom);
  atomicAdd(sums + 2, mu);
  atomicAdd(sums + 3, 1.0);
}

struct CsrOptionBMacroBoundaryMetric {
  double numerator = 0.0;
  double denominator = 0.0;
  double mass = 0.0;
  double nodes = 0.0;
  double value = 0.0;
};

const std::uint8_t* csr_optionb_upload_node_flags_for_audit(
    const core::State& state,
    core::DeviceArray<std::uint8_t>& d_node_flags_owner) {
  const int n_nodes = state.mesh.topo.n_nodes;
  if (state.mesh.topo.node_flags.size() != static_cast<std::size_t>(n_nodes) ||
      !std::any_of(state.mesh.topo.node_flags.begin(),
                   state.mesh.topo.node_flags.end(),
                   [](const std::uint8_t flags) { return flags != 0U; })) {
    return nullptr;
  }
  d_node_flags_owner.reset(static_cast<std::size_t>(n_nodes));
  d_node_flags_owner.copy_from_host(state.mesh.topo.node_flags);
  return d_node_flags_owner.data();
}

const std::uint8_t* csr_optionb_macroboundary_inactive_mask(
    const core::State& state) {
  const int n_cells = state.mesh.topo.n_cells;
  if (tenryu::hydro::central_pseudo_core::active(state) &&
      state.central_pseudo_core.d_inactive_member_mask.size() ==
          static_cast<std::size_t>(n_cells)) {
    return state.central_pseudo_core.d_inactive_member_mask.data();
  }
  if (tenryu::hydro::pole_angular_derefine::active(state) &&
      state.pole_angular_derefine.d_inactive_member_mask.size() ==
          static_cast<std::size_t>(n_cells)) {
    return state.pole_angular_derefine.d_inactive_member_mask.data();
  }
  return nullptr;
}

CsrOptionBMacroBoundaryMetric csr_optionb_macroboundary_h_metric(
    const core::State& state,
    const double* const node_mass,
    const double* const v_r,
    const double* const v_z,
    const int cell_start,
    const int cell_end,
    const double x0_r,
    const double x0_z,
    double* const H_out) {
  CsrOptionBMacroBoundaryMetric metric;
  *H_out = 0.0;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (n_nodes <= 0 || node_mass == nullptr || v_r == nullptr || v_z == nullptr) {
    return metric;
  }
  core::DeviceArray<double> d_sums("ale_remap:csr_optionb_macroboundary_h_metric:d_sums");
  d_sums.reset(4U);
  CUDA_CHECK(cudaMemset(d_sums.data(), 0, 4U * sizeof(double)));
  const int blocks = (n_nodes + 255) / 256;
  csr_optionb_macroboundary_h_reduce_kernel<<<blocks, 256>>>(
      d_sums.data(),
      node_mass,
      v_r,
      v_z,
      state.x_r_reference.data(),
      state.x_z_reference.data(),
      state.mesh.multiblock_reverse_csr_node_offsets.data(),
      state.mesh.multiblock_reverse_csr_node_cells.data(),
      csr_optionb_macroboundary_inactive_mask(state),
      n_nodes,
      cell_start,
      cell_end,
      x0_r,
      x0_z);
  CUDA_CHECK(cudaGetLastError());
  std::vector<double> sums;
  d_sums.copy_to_host(sums);
  metric.numerator = sums[0];
  metric.denominator = sums[1];
  metric.mass = sums[2];
  metric.nodes = sums[3];
  if (metric.denominator > 0.0 && std::isfinite(metric.denominator)) {
    *H_out = -metric.numerator / metric.denominator;
  }
  metric.value = *H_out;
  return metric;
}

CsrOptionBMacroBoundaryMetric csr_optionb_macroboundary_residual_metric(
    const core::State& state,
    const double* const node_mass,
    const double* const v_r,
    const double* const v_z,
    const std::uint8_t* const node_flags,
    const int cell_start,
    const int cell_end,
    const double H,
    const double x0_r,
    const double x0_z) {
  CsrOptionBMacroBoundaryMetric metric;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (n_nodes <= 0 || node_mass == nullptr || v_r == nullptr || v_z == nullptr) {
    return metric;
  }
  core::DeviceArray<double> d_sums("ale_remap:csr_optionb_macroboundary_residual_metric:d_sums");
  d_sums.reset(4U);
  CUDA_CHECK(cudaMemset(d_sums.data(), 0, 4U * sizeof(double)));
  const int blocks = (n_nodes + 255) / 256;
  csr_optionb_macroboundary_residual_reduce_kernel<<<blocks, 256>>>(
      d_sums.data(),
      node_mass,
      v_r,
      v_z,
      state.x_r_reference.data(),
      state.x_z_reference.data(),
      node_flags,
      state.mesh.multiblock_reverse_csr_node_offsets.data(),
      state.mesh.multiblock_reverse_csr_node_cells.data(),
      csr_optionb_macroboundary_inactive_mask(state),
      n_nodes,
      cell_start,
      cell_end,
      H,
      x0_r,
      x0_z);
  CUDA_CHECK(cudaGetLastError());
  std::vector<double> sums;
  d_sums.copy_to_host(sums);
  metric.numerator = sums[0];
  metric.denominator = sums[1];
  metric.mass = sums[2];
  metric.nodes = sums[3];
  const double denom = metric.denominator + 1.0e-300;
  if (denom > 0.0 && std::isfinite(denom)) {
    metric.value = std::sqrt(std::max(0.0, metric.numerator) / denom);
  }
  return metric;
}

void csr_optionb_log_macroboundary_packet_stats(
    std::ostringstream& oss,
    const CsrOptionBCornerVelocityRemapResult& result) {
  const long long faces = result.total_internal_face_packets;
  const double fb_frac =
      static_cast<double>(result.fallback_packets) /
      static_cast<double>(faces > 0 ? faces : 1);
  oss << " faces=" << faces
      << " fallback=" << result.fallback_packets
      << " fallback_frac=" << fb_frac
      << " expanded=" << result.expanded_stencil_packets
      << " ring1=" << result.expanded_ring1_packets
      << " ring2=" << result.expanded_ring2_packets
      << " exp_fail=" << result.expanded_failed_packets
      << " centroid_far=" << result.centroid_far_packets
      << " receiver_vertex_out="
      << result.receiver_vertex_out_of_donor_packets
      << " alpha_min=" << result.alpha_min
      << " invalid=" << result.invalid_input_packets
      << " skipped=" << result.skipped_packets
      << " filt_inval=" << result.filter_invalid_cells
      << " filt_degen=" << result.filter_degenerate_cells;
}

const char* csr_optionb_ring5_face_status_name(const int status) {
  switch (status) {
    case kCsrOptionBRing5FaceStatusSkippedInactive:
      return "skipped_inactive";
    case kCsrOptionBRing5FaceStatusSkippedZero:
      return "skipped_zero";
    case kCsrOptionBRing5FaceStatusPacketOk:
      return "ok";
    case kCsrOptionBRing5FaceStatusExpandedOk:
      return "expanded_ok";
    case kCsrOptionBRing5FaceStatusFallback:
      return "fallback";
    case kCsrOptionBRing5FaceStatusInvalid:
      return "invalid";
    default:
      return "none";
  }
}

struct CsrOptionBRing5FaceSummary {
  int faces = 0;
  int skipped_inactive = 0;
  int skipped_zero = 0;
  int ok = 0;
  int expanded_ok = 0;
  int fallback = 0;
  int invalid = 0;
  double max_p = 0.0;
  double max_dm = 0.0;
  double max_u = 0.0;
  int max_face = -1;
  int max_status = kCsrOptionBRing5FaceStatusNone;
  int max_centroid_class = 0;
  int max_cell_a = -1;
  int max_cell_b = -1;
  int max_donor = -1;
  int max_receiver = -1;
  double fallback_max_p = 0.0;
  int fallback_max_face = -1;
  double centroid_far_max_p = 0.0;
  int centroid_far_max_face = -1;
};

struct CsrOptionBRing5FieldSummary {
  int corner_count = 0;
  long double corner_mass = 0.0L;
  long double corner_sum_pr = 0.0L;
  long double corner_sum_pz = 0.0L;
  double corner_max_p = 0.0;
  double corner_max_u = 0.0;
  int corner_max_cell = -1;
  int corner_max_corner = -1;
  int node_count = 0;
  long double node_mass = 0.0L;
  long double node_sum_pr = 0.0L;
  long double node_sum_pz = 0.0L;
  double node_max_p = 0.0;
  double node_max_u = 0.0;
  double node_max_mass = 0.0;
  double node_max_pr = 0.0;
  double node_max_pz = 0.0;
  int node_max_id = -1;
};

CsrOptionBRing5FaceSummary csr_optionb_ring5_face_summary(
    const core::DeviceArray<double>* const d_face_p,
    const core::DeviceArray<double>* const d_face_dm,
    const core::DeviceArray<double>* const d_face_u,
    const core::DeviceArray<int>* const d_face_status,
    const core::DeviceArray<int>* const d_face_centroid_class,
    const core::DeviceArray<int>* const d_face_cell_a,
    const core::DeviceArray<int>* const d_face_cell_b,
    const core::DeviceArray<int>* const d_face_donor,
    const core::DeviceArray<int>* const d_face_receiver) {
  CsrOptionBRing5FaceSummary summary;
  if (d_face_status == nullptr || d_face_status->empty()) {
    return summary;
  }
  std::vector<int> status;
  d_face_status->copy_to_host(status);
  std::vector<double> p;
  std::vector<double> dm;
  std::vector<double> u;
  std::vector<int> centroid_class;
  std::vector<int> cell_a;
  std::vector<int> cell_b;
  std::vector<int> donor;
  std::vector<int> receiver;
  if (d_face_p != nullptr) {
    d_face_p->copy_to_host(p);
  }
  if (d_face_dm != nullptr) {
    d_face_dm->copy_to_host(dm);
  }
  if (d_face_u != nullptr) {
    d_face_u->copy_to_host(u);
  }
  if (d_face_centroid_class != nullptr) {
    d_face_centroid_class->copy_to_host(centroid_class);
  }
  if (d_face_cell_a != nullptr) {
    d_face_cell_a->copy_to_host(cell_a);
  }
  if (d_face_cell_b != nullptr) {
    d_face_cell_b->copy_to_host(cell_b);
  }
  if (d_face_donor != nullptr) {
    d_face_donor->copy_to_host(donor);
  }
  if (d_face_receiver != nullptr) {
    d_face_receiver->copy_to_host(receiver);
  }

  for (std::size_t f = 0; f < status.size(); ++f) {
    const int s = status[f];
    if (s == kCsrOptionBRing5FaceStatusNone) {
      continue;
    }
    ++summary.faces;
    if (s == kCsrOptionBRing5FaceStatusSkippedInactive) {
      ++summary.skipped_inactive;
    } else if (s == kCsrOptionBRing5FaceStatusSkippedZero) {
      ++summary.skipped_zero;
    } else if (s == kCsrOptionBRing5FaceStatusPacketOk) {
      ++summary.ok;
    } else if (s == kCsrOptionBRing5FaceStatusExpandedOk) {
      ++summary.expanded_ok;
    } else if (s == kCsrOptionBRing5FaceStatusFallback) {
      ++summary.fallback;
    } else if (s == kCsrOptionBRing5FaceStatusInvalid) {
      ++summary.invalid;
    }
    const double pf =
        (f < p.size() && std::isfinite(p[f])) ? std::max(0.0, p[f]) : 0.0;
    const int cc =
        f < centroid_class.size() ? centroid_class[f] : 0;
    if (pf > summary.max_p) {
      summary.max_p = pf;
      summary.max_dm =
          (f < dm.size() && std::isfinite(dm[f])) ? dm[f] : 0.0;
      summary.max_u =
          (f < u.size() && std::isfinite(u[f])) ? u[f] : 0.0;
      summary.max_face = static_cast<int>(f);
      summary.max_status = s;
      summary.max_centroid_class = cc;
      summary.max_cell_a = f < cell_a.size() ? cell_a[f] : -1;
      summary.max_cell_b = f < cell_b.size() ? cell_b[f] : -1;
      summary.max_donor = f < donor.size() ? donor[f] : -1;
      summary.max_receiver = f < receiver.size() ? receiver[f] : -1;
    }
    if ((s == kCsrOptionBRing5FaceStatusFallback ||
         s == kCsrOptionBRing5FaceStatusInvalid) &&
        pf > summary.fallback_max_p) {
      summary.fallback_max_p = pf;
      summary.fallback_max_face = static_cast<int>(f);
    }
    if (cc == 2 && pf > summary.centroid_far_max_p) {
      summary.centroid_far_max_p = pf;
      summary.centroid_far_max_face = static_cast<int>(f);
    }
  }
  return summary;
}

CsrOptionBRing5FieldSummary csr_optionb_ring5_field_summary(
    const core::State& state,
    const int cell_start,
    const int cell_end,
    const core::DeviceArray<double>* const d_corner_m,
    const core::DeviceArray<double>* const d_corner_pr,
    const core::DeviceArray<double>* const d_corner_pz,
    const core::DeviceArray<double>* const d_node_m,
    const core::DeviceArray<double>* const d_node_pr,
    const core::DeviceArray<double>* const d_node_pz,
    const core::DeviceArray<double>* const d_node_vr,
    const core::DeviceArray<double>* const d_node_vz) {
  CsrOptionBRing5FieldSummary summary;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (n_cells <= 0 || n_nodes <= 0 || !state.mesh.topo.multiblock.has_value()) {
    return summary;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  std::vector<std::uint8_t> ring_node_mask(static_cast<std::size_t>(n_nodes),
                                           0U);

  std::vector<double> corner_m;
  std::vector<double> corner_pr;
  std::vector<double> corner_pz;
  if (d_corner_m != nullptr && d_corner_pr != nullptr &&
      d_corner_pz != nullptr && !d_corner_m->empty()) {
    d_corner_m->copy_to_host(corner_m);
    d_corner_pr->copy_to_host(corner_pr);
    d_corner_pz->copy_to_host(corner_pz);
  }
  const bool have_corners =
      corner_m.size() >= static_cast<std::size_t>(n_cells * 4) &&
      corner_pr.size() == corner_m.size() && corner_pz.size() == corner_m.size();
  for (int c = cell_start; c <= cell_end && c < n_cells; ++c) {
    if (c < 0 ||
        (state.central_pseudo_core.inactive_member_mask.size() ==
             static_cast<std::size_t>(n_cells) &&
         state.central_pseudo_core.inactive_member_mask[static_cast<std::size_t>(c)] !=
             0U) ||
        (state.pole_angular_derefine.inactive_member_mask.size() ==
             static_cast<std::size_t>(n_cells) &&
         state.pole_angular_derefine
                 .inactive_member_mask[static_cast<std::size_t>(c)] != 0U)) {
      continue;
    }
    const int nverts =
        tenryu::mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c);
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    for (int k = 0; k < nverts; ++k) {
      const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      if (n >= 0 && n < n_nodes) {
        ring_node_mask[static_cast<std::size_t>(n)] = 1U;
      }
      if (!have_corners) {
        continue;
      }
      const int idx = c * 4 + k;
      const double m = corner_m[static_cast<std::size_t>(idx)];
      const double pr = corner_pr[static_cast<std::size_t>(idx)];
      const double pz = corner_pz[static_cast<std::size_t>(idx)];
      if (!std::isfinite(m) || !std::isfinite(pr) || !std::isfinite(pz)) {
        continue;
      }
      const double p_abs = std::sqrt(pr * pr + pz * pz);
      const double u_abs = (m > 0.0) ? p_abs / m : 0.0;
      ++summary.corner_count;
      summary.corner_mass += static_cast<long double>(m);
      summary.corner_sum_pr += static_cast<long double>(pr);
      summary.corner_sum_pz += static_cast<long double>(pz);
      if (p_abs > summary.corner_max_p) {
        summary.corner_max_p = p_abs;
        summary.corner_max_u = u_abs;
        summary.corner_max_cell = c;
        summary.corner_max_corner = k;
      }
    }
  }

  if (d_node_m == nullptr || d_node_m->empty()) {
    return summary;
  }
  std::vector<double> node_m;
  std::vector<double> node_pr;
  std::vector<double> node_pz;
  std::vector<double> node_vr;
  std::vector<double> node_vz;
  d_node_m->copy_to_host(node_m);
  const bool have_node_p =
      d_node_pr != nullptr && d_node_pz != nullptr && !d_node_pr->empty();
  if (have_node_p) {
    d_node_pr->copy_to_host(node_pr);
    d_node_pz->copy_to_host(node_pz);
  }
  if (d_node_vr != nullptr && d_node_vz != nullptr && !d_node_vr->empty()) {
    d_node_vr->copy_to_host(node_vr);
    d_node_vz->copy_to_host(node_vz);
  }
  for (int n = 0; n < n_nodes; ++n) {
    if (ring_node_mask[static_cast<std::size_t>(n)] == 0U ||
        static_cast<std::size_t>(n) >= node_m.size()) {
      continue;
    }
    const double m = node_m[static_cast<std::size_t>(n)];
    double pr = 0.0;
    double pz = 0.0;
    if (have_node_p && static_cast<std::size_t>(n) < node_pr.size() &&
        static_cast<std::size_t>(n) < node_pz.size()) {
      pr = node_pr[static_cast<std::size_t>(n)];
      pz = node_pz[static_cast<std::size_t>(n)];
    } else if (static_cast<std::size_t>(n) < node_vr.size() &&
               static_cast<std::size_t>(n) < node_vz.size()) {
      pr = m * node_vr[static_cast<std::size_t>(n)];
      pz = m * node_vz[static_cast<std::size_t>(n)];
    }
    if (!std::isfinite(m) || !std::isfinite(pr) || !std::isfinite(pz)) {
      continue;
    }
    double u_abs = 0.0;
    if (static_cast<std::size_t>(n) < node_vr.size() &&
        static_cast<std::size_t>(n) < node_vz.size() &&
        std::isfinite(node_vr[static_cast<std::size_t>(n)]) &&
        std::isfinite(node_vz[static_cast<std::size_t>(n)])) {
      const double vr = node_vr[static_cast<std::size_t>(n)];
      const double vz = node_vz[static_cast<std::size_t>(n)];
      u_abs = std::sqrt(vr * vr + vz * vz);
    } else if (m > 0.0) {
      u_abs = std::sqrt(pr * pr + pz * pz) / m;
    }
    const double p_abs = std::sqrt(pr * pr + pz * pz);
    ++summary.node_count;
    summary.node_mass += static_cast<long double>(m);
    summary.node_sum_pr += static_cast<long double>(pr);
    summary.node_sum_pz += static_cast<long double>(pz);
    if (u_abs > summary.node_max_u) {
      summary.node_max_u = u_abs;
      summary.node_max_p = p_abs;
      summary.node_max_mass = m;
      summary.node_max_pr = pr;
      summary.node_max_pz = pz;
      summary.node_max_id = n;
    }
  }
  return summary;
}

void csr_optionb_emit_ring5_momentum_trace(
    const core::State& state,
    const char* const stage,
    const int cell_start,
    const int cell_end,
    const core::DeviceArray<double>* const d_corner_m,
    const core::DeviceArray<double>* const d_corner_pr,
    const core::DeviceArray<double>* const d_corner_pz,
    const core::DeviceArray<double>* const d_node_m,
    const core::DeviceArray<double>* const d_node_pr,
    const core::DeviceArray<double>* const d_node_pz,
    const core::DeviceArray<double>* const d_node_vr,
    const core::DeviceArray<double>* const d_node_vz,
    const CsrOptionBRing5FaceSummary* const face_summary) {
  const auto field =
      csr_optionb_ring5_field_summary(state,
                                      cell_start,
                                      cell_end,
                                      d_corner_m,
                                      d_corner_pr,
                                      d_corner_pz,
                                      d_node_m,
                                      d_node_pr,
                                      d_node_pz,
                                      d_node_vr,
                                      d_node_vz);
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(16)
      << "[optionb_ring5_momentum_trace]"
      << " step=" << state.step
      << " remap=" << (state.ale_remaps_applied + 1)
      << " stage=" << stage
      << " ring5_cell_range=" << cell_start << ".." << cell_end
      << " corner_count=" << field.corner_count
      << " corner_mass=" << static_cast<double>(field.corner_mass)
      << " corner_sum_pr=" << static_cast<double>(field.corner_sum_pr)
      << " corner_sum_pz=" << static_cast<double>(field.corner_sum_pz)
      << " corner_max_p=" << field.corner_max_p
      << " corner_max_u=" << field.corner_max_u
      << " corner_max_cell=" << field.corner_max_cell
      << " corner_max_corner=" << field.corner_max_corner
      << " node_count=" << field.node_count
      << " node_mass=" << static_cast<double>(field.node_mass)
      << " node_sum_pr=" << static_cast<double>(field.node_sum_pr)
      << " node_sum_pz=" << static_cast<double>(field.node_sum_pz)
      << " node_max_u=" << field.node_max_u
      << " node_max_p=" << field.node_max_p
      << " node_max_id=" << field.node_max_id
      << " node_max_mass=" << field.node_max_mass
      << " node_max_pr=" << field.node_max_pr
      << " node_max_pz=" << field.node_max_pz;
  if (face_summary != nullptr) {
    oss << " faces=" << face_summary->faces
        << " face_skipped_inactive=" << face_summary->skipped_inactive
        << " face_skipped_zero=" << face_summary->skipped_zero
        << " face_ok=" << face_summary->ok
        << " face_expanded_ok=" << face_summary->expanded_ok
        << " face_fallback=" << face_summary->fallback
        << " face_invalid=" << face_summary->invalid
        << " face_max_p=" << face_summary->max_p
        << " face_max_dm=" << face_summary->max_dm
        << " face_max_u=" << face_summary->max_u
        << " face_max_id=" << face_summary->max_face
        << " face_max_status="
        << csr_optionb_ring5_face_status_name(face_summary->max_status)
        << " face_max_centroid_class=" << face_summary->max_centroid_class
        << " face_max_cell_a=" << face_summary->max_cell_a
        << " face_max_cell_b=" << face_summary->max_cell_b
        << " face_max_donor=" << face_summary->max_donor
        << " face_max_receiver=" << face_summary->max_receiver
        << " face_fallback_max_p=" << face_summary->fallback_max_p
        << " face_fallback_max_id=" << face_summary->fallback_max_face
        << " face_centroid_far_max_p=" << face_summary->centroid_far_max_p
        << " face_centroid_far_max_id=" << face_summary->centroid_far_max_face;
  }
  core::log_info(oss.str());
}

void csr_optionb_emit_macroboundary_actual_audit(
    const core::State& state,
    const CsrOptionBCornerVelocityRemapBuffers& buffers,
    const CsrOptionBCornerVelocityRemapResult& result) {
  const int n_cells = state.mesh.topo.n_cells;
  if (n_cells <= 0) {
    return;
  }
  int cell_start = optionb_macroboundary_ring_start();
  int cell_end = optionb_macroboundary_ring_end();
  if (cell_end < cell_start) {
    std::swap(cell_start, cell_end);
  }
  cell_start = std::max(0, std::min(n_cells - 1, cell_start));
  cell_end = std::max(0, std::min(n_cells - 1, cell_end));
  core::DeviceArray<std::uint8_t> d_node_flags_owner("ale_remap:csr_optionb_emit_macroboundary_actual_audit:d_node_flags_owner");
  const std::uint8_t* const d_node_flags =
      csr_optionb_upload_node_flags_for_audit(state, d_node_flags_owner);
  const double x0_r = optionb_macroboundary_audit_x0_r();
  const double x0_z = optionb_macroboundary_audit_x0_z();
  double H5 = 0.0;
  const auto h_metric = csr_optionb_macroboundary_h_metric(state,
                                                           buffers.node_mass.data(),
                                                           buffers.v_r.data(),
                                                           buffers.v_z.data(),
                                                           cell_start,
                                                           cell_end,
                                                           x0_r,
                                                           x0_z,
                                                           &H5);
  const auto eta_metric = csr_optionb_macroboundary_residual_metric(
      state,
      buffers.node_mass.data(),
      buffers.v_r.data(),
      buffers.v_z.data(),
      d_node_flags,
      cell_start,
      cell_end,
      H5,
      x0_r,
      x0_z);
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(16)
      << "[optionb_macroboundary_audit] step=" << state.step
      << " remap=" << (state.ale_remaps_applied + 1)
      << " mode=actual limiters=production"
      << " ring5_cell_range=" << cell_start << ".." << cell_end
      << " H_5=" << H5
      << " eta_5=" << eta_metric.value
      << " ring_nodes=" << eta_metric.nodes
      << " ring_mass=" << eta_metric.mass
      << " numerator=" << eta_metric.numerator
      << " denominator=" << eta_metric.denominator;
  csr_optionb_log_macroboundary_packet_stats(oss, result);
  core::log_info(oss.str());
  (void)h_metric;
}

void csr_optionb_emit_macroboundary_manufactured_audit(
    const core::State& state,
    const core::Config& cfg,
    const double* const target_cell_mass,
    const bool disable_limiters_for_audit) {
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (n_cells <= 0 || n_nodes <= 0) {
    return;
  }
  int cell_start = optionb_macroboundary_ring_start();
  int cell_end = optionb_macroboundary_ring_end();
  if (cell_end < cell_start) {
    std::swap(cell_start, cell_end);
  }
  cell_start = std::max(0, std::min(n_cells - 1, cell_start));
  cell_end = std::max(0, std::min(n_cells - 1, cell_end));
  core::DeviceArray<std::uint8_t> d_node_flags_owner("ale_remap:csr_optionb_emit_macroboundary_manufactured_audit:d_node_flags_owner");
  const std::uint8_t* const d_node_flags =
      csr_optionb_upload_node_flags_for_audit(state, d_node_flags_owner);
  const double H = optionb_macroboundary_audit_H();
  const double x0_r = optionb_macroboundary_audit_x0_r();
  const double x0_z = optionb_macroboundary_audit_x0_z();
  core::DeviceArray<double> d_affine_v_r("ale_remap:csr_optionb_emit_macroboundary_manufactured_audit:d_affine_v_r");
  d_affine_v_r.reset(static_cast<std::size_t>(n_nodes));
  core::DeviceArray<double> d_affine_v_z("ale_remap:csr_optionb_emit_macroboundary_manufactured_audit:d_affine_v_z");
  d_affine_v_z.reset(static_cast<std::size_t>(n_nodes));
  const int blocks_nodes = (n_nodes + 255) / 256;
  csr_optionb_fill_homologous_velocity_kernel<<<blocks_nodes, 256>>>(
      d_affine_v_r.data(),
      d_affine_v_z.data(),
      state.x_r.data(),
      state.x_z.data(),
      d_node_flags,
      H,
      x0_r,
      x0_z,
      n_nodes);
  CUDA_CHECK(cudaGetLastError());

  CsrOptionBCornerVelocityRemapBuffers audit_buffers;
  const auto audit_result = csr_optionb_corner_velocity_remap_component(
      state,
      cfg,
      audit_buffers,
      true,
      target_cell_mass,
      d_affine_v_r.data(),
      d_affine_v_z.data(),
      disable_limiters_for_audit);
  if (!audit_result.applied) {
    core::log_info(
        "[optionb_macroboundary_audit] mode=manufactured status=not_applied");
    return;
  }
  const auto eps_metric = csr_optionb_macroboundary_residual_metric(
      state,
      audit_buffers.node_mass.data(),
      audit_buffers.v_r.data(),
      audit_buffers.v_z.data(),
      d_node_flags,
      cell_start,
      cell_end,
      H,
      x0_r,
      x0_z);
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(16)
      << "[optionb_macroboundary_audit] step=" << state.step
      << " remap=" << (state.ale_remaps_applied + 1)
      << " mode=manufactured"
      << " limiters=" << (disable_limiters_for_audit ? "off" : "on")
      << " ring5_cell_range=" << cell_start << ".." << cell_end
      << " H_input=" << H
      << " eps_aff5=" << eps_metric.value
      << " ring_nodes=" << eps_metric.nodes
      << " ring_mass=" << eps_metric.mass
      << " numerator=" << eps_metric.numerator
      << " denominator=" << eps_metric.denominator;
  csr_optionb_log_macroboundary_packet_stats(oss, audit_result);
  core::log_info(oss.str());
}

__global__ void csr_compute_cell_velocity_from_nodes_kernel(
    double* __restrict__ v_r_cell,
    double* __restrict__ v_z_cell,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const int off = cell_node_csr_offsets[c];
  if (cell_nverts == nullptr) {
    double vr = 0.0;
    double vz = 0.0;
    for (int k = 0; k < 4; ++k) {
      const int n = cell_node_csr_indices[off + k];
      vr += v_r_node[n];
      vz += v_z_node[n];
    }
    v_r_cell[c] = 0.25 * vr;
    v_z_cell[c] = 0.25 * vz;
    return;
  }
  const int active_nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  double vr = 0.0;
  double vz = 0.0;
  for (int k = 0; k < active_nverts; ++k) {
    const int n = cell_node_csr_indices[off + k];
    vr += v_r_node[n];
    vz += v_z_node[n];
  }
  const double inv_nverts = 1.0 / static_cast<double>(active_nverts);
  v_r_cell[c] = inv_nverts * vr;
  v_z_cell[c] = inv_nverts * vz;
}

__global__ void csr_initialize_hydro_extents_kernel(
    double* __restrict__ mass_new,
    double* __restrict__ mom_r_new,
    double* __restrict__ mom_z_new,
    double* __restrict__ energy_e_new,
    double* __restrict__ energy_i_new,
    const double* __restrict__ mass_lag,
    const double* __restrict__ rho_lag,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const double* __restrict__ e_e_lag,
    const double* __restrict__ e_i_lag,
    const double* __restrict__ vol_lag,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  double m = fmax(mass_lag[c], 0.0);
  if (!isfinite(m)) {
    m = fmax(rho_lag[c], 0.0) * fmax(vol_lag[c], 0.0);
  }
  mass_new[c] = m;
  mom_r_new[c] = m * v_r_cell[c];
  mom_z_new[c] = m * v_z_cell[c];
  energy_e_new[c] = m * fmax(e_e_lag[c], 0.0);
  energy_i_new[c] = m * fmax(e_i_lag[c], 0.0);
}

__global__ void csr_initialize_corner_fraction_remap_kernel(
    double* __restrict__ corner_fraction_lag,
    double* __restrict__ corner_fraction_mass_new,
    const double* __restrict__ corner_mass_lag,
    const double* __restrict__ mass_lag,
    const double* __restrict__ rho_lag,
    const double* __restrict__ vol_lag,
    const std::uint8_t* __restrict__ cell_nverts,
    const int corner_stride,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  double m = fmax(mass_lag[c], 0.0);
  if (!isfinite(m)) {
    m = fmax(rho_lag[c], 0.0) * fmax(vol_lag[c], 0.0);
  }
  const int base = c * corner_stride;
  if (cell_nverts == nullptr) {
    for (int k = 0; k < 4; ++k) {
      double f = 0.25;
      if (m > 0.0 && isfinite(m)) {
        const double raw = corner_mass_lag[base + k] / m;
        f = (raw > 0.0 && isfinite(raw)) ? raw : 0.0;
      }
      const int idx = k * n_cells + c;
      corner_fraction_lag[idx] = f;
      corner_fraction_mass_new[idx] = m * f;
    }
    for (int k = 4; k < corner_stride; ++k) {
      const int idx = k * n_cells + c;
      corner_fraction_lag[idx] = 0.0;
      corner_fraction_mass_new[idx] = 0.0;
    }
    return;
  }
  const int active_nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  const double uniform_fraction = 1.0 / static_cast<double>(active_nverts);
  for (int k = 0; k < corner_stride; ++k) {
    double f = (k < active_nverts) ? uniform_fraction : 0.0;
    if (k < active_nverts && m > 0.0 && isfinite(m)) {
      const double raw = corner_mass_lag[base + k] / m;
      f = (raw > 0.0 && isfinite(raw)) ? raw : 0.0;
    }
    const int idx = k * n_cells + c;
    corner_fraction_lag[idx] = f;
    corner_fraction_mass_new[idx] = m * f;
  }
}

__global__ void csr_initialize_gas_tracer_remap_kernel(
    double* __restrict__ gas_tracer_mass_new,
    const double* __restrict__ gas_tracer_lag,
    const double* __restrict__ mass_lag,
    const double* __restrict__ rho_lag,
    const double* __restrict__ vol_lag,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  double m = fmax(mass_lag[c], 0.0);
  if (!isfinite(m)) {
    m = fmax(rho_lag[c], 0.0) * fmax(vol_lag[c], 0.0);
  }
  gas_tracer_mass_new[c] = m * clamp01_device(gas_tracer_lag[c]);
}

__global__ void csr_initialize_hot_e_eps_remap_kernel(
    double* __restrict__ hot_e_eps_mass_new,
    const double* __restrict__ hot_e_eps_lag,
    const double* __restrict__ mass_lag,
    const double* __restrict__ rho_lag,
    const double* __restrict__ vol_lag,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  double m = fmax(mass_lag[c], 0.0);
  if (!isfinite(m)) {
    m = fmax(rho_lag[c], 0.0) * fmax(vol_lag[c], 0.0);
  }
  hot_e_eps_mass_new[c] = m * fmax(hot_e_eps_lag[c], 0.0);
}

__global__ void csr_initialize_burn_eps_remap_kernel(
    double* __restrict__ burn_eps_mass_new,
    const double* __restrict__ burn_eps_lag,
    const double* __restrict__ mass_lag,
    const double* __restrict__ rho_lag,
    const double* __restrict__ vol_lag,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  double m = fmax(mass_lag[c], 0.0);
  if (!isfinite(m)) {
    m = fmax(rho_lag[c], 0.0) * fmax(vol_lag[c], 0.0);
  }
  burn_eps_mass_new[c] = m * fmax(burn_eps_lag[c], 0.0);
}

__global__ void csr_initialize_burn_species_remap_kernel(
    double* __restrict__ species_mass_new,
    const double* __restrict__ species_Y_lag,
    const double* __restrict__ mass_lag,
    const double* __restrict__ rho_lag,
    const double* __restrict__ vol_lag,
    const int n_cells) {
  (void)mass_lag;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  double m = fmax(rho_lag[c], 0.0) * fmax(vol_lag[c], 0.0);
  if (!isfinite(m)) {
    m = 0.0;
  }
  for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
    species_mass_new[c * tenryu::burn::kNumSpecies + s] =
        m * species_Y_lag[c * tenryu::burn::kNumSpecies + s];
  }
}

struct CsrHydroFluxStageDeviceView {
  double* mass = nullptr;
  double* mom_r = nullptr;
  double* mom_z = nullptr;
  double* energy_e = nullptr;
  double* energy_i = nullptr;
  double* total_energy = nullptr;
  double* ye_mass = nullptr;
  double* corner_fraction_mass = nullptr;
  double* gas_tracer_mass = nullptr;
  double* burn_species_mass = nullptr;
  double* hot_e_eps_mass = nullptr;
  double* burn_eps_mass = nullptr;
  std::size_t n_face_sides = 0U;
  int corner_stride = 4;
};

struct CsrHydroFluxAccumulatorDeviceView {
  double* mass = nullptr;
  double* mom_r = nullptr;
  double* mom_z = nullptr;
  double* energy_e = nullptr;
  double* energy_i = nullptr;
  double* total_energy = nullptr;
  double* ye_mass = nullptr;
  double* corner_fraction_mass = nullptr;
  double* gas_tracer_mass = nullptr;
  double* burn_species_mass = nullptr;
  double* hot_e_eps_mass = nullptr;
  double* burn_eps_mass = nullptr;
};

__global__ void csr_gather_hydro_flux_stage_kernel(
    const CsrHydroFluxAccumulatorDeviceView accumulator,
    const CsrHydroFluxStageDeviceView stage,
    const int* __restrict__ cell_edge_csr_offsets,
    const int* __restrict__ cell_edge_csr_edges,
    const std::int8_t* __restrict__ cell_edge_csr_side,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  double mass = 0.0;
  double mom_r = 0.0;
  double mom_z = 0.0;
  double energy_e = 0.0;
  double energy_i = 0.0;
  double total_energy = 0.0;
  double ye_mass = 0.0;
  double corner_fraction_mass[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  double gas_tracer_mass = 0.0;
  double burn_species_mass[tenryu::burn::kNumSpecies] = {};
  double hot_e_eps_mass = 0.0;
  double burn_eps_mass = 0.0;
  for (int k = cell_edge_csr_offsets[c]; k < cell_edge_csr_offsets[c + 1];
       ++k) {
    const int edge = cell_edge_csr_edges[k];
    const int side = static_cast<int>(cell_edge_csr_side[k]);
    const int slot = 2 * edge + side;
    mass += stage.mass[slot];
    mom_r += stage.mom_r[slot];
    mom_z += stage.mom_z[slot];
    if (stage.total_energy != nullptr) {
      total_energy += stage.total_energy[slot];
      ye_mass += stage.ye_mass[slot];
    } else {
      energy_e += stage.energy_e[slot];
      energy_i += stage.energy_i[slot];
    }
    if (stage.corner_fraction_mass != nullptr) {
      for (int corner = 0; corner < stage.corner_stride; ++corner) {
        corner_fraction_mass[corner] +=
            stage.corner_fraction_mass[corner * stage.n_face_sides + slot];
      }
    }
    if (stage.gas_tracer_mass != nullptr) {
      gas_tracer_mass += stage.gas_tracer_mass[slot];
    }
    if (stage.burn_species_mass != nullptr) {
      for (int species = 0; species < tenryu::burn::kNumSpecies; ++species) {
        burn_species_mass[species] +=
            stage.burn_species_mass[species * stage.n_face_sides + slot];
      }
    }
    if (stage.hot_e_eps_mass != nullptr) {
      hot_e_eps_mass += stage.hot_e_eps_mass[slot];
    }
    if (stage.burn_eps_mass != nullptr) {
      burn_eps_mass += stage.burn_eps_mass[slot];
    }
  }
  accumulator.mass[c] += mass;
  accumulator.mom_r[c] += mom_r;
  accumulator.mom_z[c] += mom_z;
  if (stage.total_energy != nullptr) {
    accumulator.total_energy[c] += total_energy;
    accumulator.ye_mass[c] += ye_mass;
  } else {
    accumulator.energy_e[c] += energy_e;
    accumulator.energy_i[c] += energy_i;
  }
  if (stage.corner_fraction_mass != nullptr) {
    for (int corner = 0; corner < stage.corner_stride; ++corner) {
      accumulator.corner_fraction_mass[corner * n_cells + c] +=
          corner_fraction_mass[corner];
    }
  }
  if (stage.gas_tracer_mass != nullptr) {
    accumulator.gas_tracer_mass[c] += gas_tracer_mass;
  }
  if (stage.burn_species_mass != nullptr) {
    for (int species = 0; species < tenryu::burn::kNumSpecies; ++species) {
      accumulator.burn_species_mass[
          c * tenryu::burn::kNumSpecies + species] +=
          burn_species_mass[species];
    }
  }
  if (stage.hot_e_eps_mass != nullptr) {
    accumulator.hot_e_eps_mass[c] += hot_e_eps_mass;
  }
  if (stage.burn_eps_mass != nullptr) {
    accumulator.burn_eps_mass[c] += burn_eps_mass;
  }
}

__global__ void csr_gather_face_side_scalar_stage_kernel(
    double* __restrict__ accumulator,
    const double* __restrict__ stage,
    const int* __restrict__ cell_edge_csr_offsets,
    const int* __restrict__ cell_edge_csr_edges,
    const std::int8_t* __restrict__ cell_edge_csr_side,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  double total = 0.0;
  for (int k = cell_edge_csr_offsets[c]; k < cell_edge_csr_offsets[c + 1];
       ++k) {
    const int edge = cell_edge_csr_edges[k];
    const int side = static_cast<int>(cell_edge_csr_side[k]);
    total += stage[2 * edge + side];
  }
  accumulator[c] += total;
}

__device__ inline void csr_apply_hydro_flux(
    const CsrHydroFluxStageDeviceView stage,
    const double* __restrict__ rho_lag,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const double* __restrict__ e_e_lag,
    const double* __restrict__ e_i_lag,
    const double* __restrict__ e_tot_lag,
    const double* __restrict__ ye_int_lag,
    const double* __restrict__ corner_fraction_lag,
    const double* __restrict__ gas_tracer_lag,
    const double* __restrict__ burn_species_Y_lag,
    const double* __restrict__ hot_e_eps_lag,
    const double* __restrict__ burn_eps_lag,
    const std::uint8_t* __restrict__ cell_nverts,
    const int cell,
    const int donor,
    const int n_cells,
    const int stage_slot,
    const double signed_volume) {
  if (!finite_nonzero(signed_volume)) {
    return;
  }
  const double dm = fmax(rho_lag[donor], 0.0) * signed_volume;
  stage.mass[stage_slot] += dm;
  stage.mom_r[stage_slot] += dm * v_r_cell[donor];
  stage.mom_z[stage_slot] += dm * v_z_cell[donor];
  if (stage.total_energy != nullptr && stage.ye_mass != nullptr &&
      e_tot_lag != nullptr && ye_int_lag != nullptr) {
    stage.total_energy[stage_slot] += dm * fmax(e_tot_lag[donor], 0.0);
    stage.ye_mass[stage_slot] += dm * clamp01_device(ye_int_lag[donor]);
  } else {
    stage.energy_e[stage_slot] += dm * fmax(e_e_lag[donor], 0.0);
    stage.energy_i[stage_slot] += dm * fmax(e_i_lag[donor], 0.0);
  }
  if (stage.corner_fraction_mass != nullptr &&
      corner_fraction_lag != nullptr) {
    const int active_nverts =
        tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, cell);
    for (int k = 0; k < active_nverts; ++k) {
      const double raw = corner_fraction_lag[k * n_cells + donor];
      const double f = (raw > 0.0 && isfinite(raw)) ? raw : 0.0;
      stage.corner_fraction_mass[k * stage.n_face_sides + stage_slot] +=
          dm * f;
    }
  }
  if (stage.gas_tracer_mass != nullptr && gas_tracer_lag != nullptr) {
    stage.gas_tracer_mass[stage_slot] +=
        dm * clamp01_device(gas_tracer_lag[donor]);
  }
  if (stage.burn_species_mass != nullptr && burn_species_Y_lag != nullptr) {
    for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
      stage.burn_species_mass[s * stage.n_face_sides + stage_slot] +=
          dm * burn_species_Y_lag[donor * tenryu::burn::kNumSpecies + s];
    }
  }
  if (stage.hot_e_eps_mass != nullptr && hot_e_eps_lag != nullptr) {
    stage.hot_e_eps_mass[stage_slot] +=
        dm * fmax(hot_e_eps_lag[donor], 0.0);
  }
  if (stage.burn_eps_mass != nullptr && burn_eps_lag != nullptr) {
    stage.burn_eps_mass[stage_slot] +=
        dm * fmax(burn_eps_lag[donor], 0.0);
  }
}

template <bool GclAudit>
__global__ void csr_apply_internal_hydro_flux_kernel(
    const CsrHydroFluxStageDeviceView stage,
    const double* __restrict__ rho_lag,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const double* __restrict__ e_e_lag,
    const double* __restrict__ e_i_lag,
    const double* __restrict__ e_tot_lag,
    const double* __restrict__ ye_int_lag,
    const double* __restrict__ corner_fraction_lag,
    const double* __restrict__ gas_tracer_lag,
    const double* __restrict__ burn_species_Y_lag,
    const double* __restrict__ hot_e_eps_lag,
    const double* __restrict__ burn_eps_lag,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ unique_cell_a,
    const int* __restrict__ unique_cell_b,
    const int* __restrict__ unique_local_a,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ mass_flux_scale,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    double* __restrict__ macro_flux_audit,
    const int n_faces,
    const int n_cells,
    const RemapDispatchAuditDeviceView remap_dispatch_audit,
    const CsrGclAuditDeviceView gcl_audit) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int cell_a = unique_cell_a[f];
  const int cell_b = unique_cell_b[f];
  if (csr_face_touches_inactive(inactive_cell_mask, cell_a, cell_b)) {
    csr_note_inactive_face_skip(macro_flux_audit);
    return;
  }
  const int local_a = unique_local_a[f];
  const double dV_a = csr_face_swept_volume_outward(x_r_old,
                                                    x_z_old,
                                                    x_r_new,
                                                    x_z_new,
                                                    cell_node_csr_offsets,
                                                    cell_node_csr_indices,
                                                    cell_orientation_sign,
                                                    cell_a,
                                                    local_a,
                                                    cell_nverts);
  const int donor = csr_internal_flux_donor(cell_a, cell_b, dV_a);
  const int losing_cell = csr_internal_flux_losing_cell(cell_a, cell_b, dV_a);
  double dV_limited = dV_a;
  if (mass_flux_scale != nullptr && finite_nonzero(dV_a)) {
    double s = mass_flux_scale[losing_cell];
    if (!isfinite(s)) {
      s = 0.0;
    }
    s = fmin(1.0, fmax(0.0, s));
    dV_limited = dV_a * s;
  }
  if (finite_nonzero(dV_limited)) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::LegacySweptVolume,
        donor);
  }
  csr_apply_hydro_flux(stage,
                       rho_lag,
                       v_r_cell,
                       v_z_cell,
                       e_e_lag,
                       e_i_lag,
                       e_tot_lag,
                       ye_int_lag,
                       corner_fraction_lag,
                       gas_tracer_lag,
                       burn_species_Y_lag,
                       hot_e_eps_lag,
                       burn_eps_lag,
                       cell_nverts,
                       cell_a,
                       donor,
                       n_cells,
                       2 * f,
                       dV_limited);
  csr_apply_hydro_flux(stage,
                       rho_lag,
                       v_r_cell,
                       v_z_cell,
                       e_e_lag,
                       e_i_lag,
                       e_tot_lag,
                       ye_int_lag,
                       corner_fraction_lag,
                       gas_tracer_lag,
                       burn_species_Y_lag,
                       hot_e_eps_lag,
                       burn_eps_lag,
                       cell_nverts,
                       cell_b,
                       donor,
                       n_cells,
                       2 * f + 1,
                       -dV_limited);
  csr_gcl_audit_store_internal_face<GclAudit>(gcl_audit, f, dV_limited);
}

template <bool GclAudit>
__global__ void csr_apply_boundary_hydro_flux_kernel(
    const CsrHydroFluxStageDeviceView stage,
    const double* __restrict__ rho_lag,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const double* __restrict__ e_e_lag,
    const double* __restrict__ e_i_lag,
    const double* __restrict__ e_tot_lag,
    const double* __restrict__ ye_int_lag,
    const double* __restrict__ corner_fraction_lag,
    const double* __restrict__ gas_tracer_lag,
    const double* __restrict__ burn_species_Y_lag,
    const double* __restrict__ hot_e_eps_lag,
    const double* __restrict__ burn_eps_lag,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ boundary_cell,
    const int* __restrict__ boundary_local,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ mass_flux_scale,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    double* __restrict__ macro_flux_audit,
    const int edge_offset,
    const int n_faces,
    const int n_cells,
    const RemapDispatchAuditDeviceView remap_dispatch_audit,
    const CsrGclAuditDeviceView gcl_audit) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int cell = boundary_cell[f];
  if (csr_inactive_cell(inactive_cell_mask, cell)) {
    csr_note_inactive_face_skip(macro_flux_audit);
    return;
  }
  const int local = boundary_local[f];
  const double dV = csr_face_swept_volume_outward(x_r_old,
                                                  x_z_old,
                                                  x_r_new,
                                                  x_z_new,
                                                  cell_node_csr_offsets,
                                                  cell_node_csr_indices,
                                                  cell_orientation_sign,
                                                  cell,
                                                  local,
                                                  cell_nverts);
  double dV_limited = dV;
  if (mass_flux_scale != nullptr && dV < 0.0 && finite_nonzero(dV)) {
    double s = mass_flux_scale[cell];
    if (!isfinite(s)) {
      s = 0.0;
    }
    s = fmin(1.0, fmax(0.0, s));
    dV_limited = dV * s;
  }
  if (finite_nonzero(dV_limited)) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::LegacySweptVolume,
        cell);
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::BoundaryOneSided,
        cell);
  }
  csr_apply_hydro_flux(stage,
                       rho_lag,
                       v_r_cell,
                       v_z_cell,
                       e_e_lag,
                       e_i_lag,
                       e_tot_lag,
                       ye_int_lag,
                       corner_fraction_lag,
                       gas_tracer_lag,
                       burn_species_Y_lag,
                       hot_e_eps_lag,
                       burn_eps_lag,
                       cell_nverts,
                       cell,
                       cell,
                       n_cells,
                       2 * (edge_offset + f),
                       dV_limited);
  csr_gcl_audit_store_boundary_face<GclAudit>(gcl_audit, f, dV_limited);
}

__device__ inline double csr_apply_hydro_flux_second_order(
    const CsrHydroFluxStageDeviceView stage,
    const double* __restrict__ rho_lag,
    const double* __restrict__ rho_grad_r,
    const double* __restrict__ rho_grad_z,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const double* __restrict__ e_e_lag,
    const double* __restrict__ e_i_lag,
    const double* __restrict__ e_tot_lag,
    const double* __restrict__ ye_int_lag,
    const double* __restrict__ corner_fraction_lag,
    const double* __restrict__ corner_fraction_grad_r,
    const double* __restrict__ corner_fraction_grad_z,
    const double* __restrict__ gas_tracer_lag,
    const double* __restrict__ burn_species_Y_lag,
    const double* __restrict__ hot_e_eps_lag,
    const double* __restrict__ burn_eps_lag,
    const double* __restrict__ gas_tracer_grad_r,
    const double* __restrict__ gas_tracer_grad_z,
    const double* __restrict__ vol_lag,
    const double* __restrict__ old_centroid_r,
    const double* __restrict__ old_centroid_z,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int cell,
    const int donor,
    const int n_cells,
    const int stage_slot,
    const double dV,
    const double dMr,
    const double dMz,
    const double flux_scale,
    const bool no_face_clip,
    const RemapDispatchAuditDeviceView remap_dispatch_audit) {
  if (!finite_nonzero(dV)) {
    return 0.0;
  }
  const double rho_integral = csr_moments_direct_field_integral(
      rho_lag,
      rho_grad_r,
      rho_grad_z,
      old_centroid_r,
      old_centroid_z,
      x_r_old,
      x_z_old,
      face_adj_csr_offsets,
      face_adj_csr_indices,
      cell_node_csr_offsets,
      cell_node_csr_indices,
      cell_nverts,
      inactive_cell_mask,
      vol_lag[donor],
      donor,
      n_cells,
      dV,
      dMr,
      dMz,
      no_face_clip,
      remap_dispatch_audit);
  const double dm = flux_scale * rho_integral;
  stage.mass[stage_slot] += dm;
  stage.mom_r[stage_slot] += dm * v_r_cell[donor];
  stage.mom_z[stage_slot] += dm * v_z_cell[donor];
  if (stage.total_energy != nullptr && stage.ye_mass != nullptr &&
      e_tot_lag != nullptr && ye_int_lag != nullptr) {
    stage.total_energy[stage_slot] += dm * fmax(e_tot_lag[donor], 0.0);
    stage.ye_mass[stage_slot] += dm * clamp01_device(ye_int_lag[donor]);
  } else {
    stage.energy_e[stage_slot] += dm * fmax(e_e_lag[donor], 0.0);
    stage.energy_i[stage_slot] += dm * fmax(e_i_lag[donor], 0.0);
  }
  if (stage.corner_fraction_mass != nullptr && corner_fraction_lag != nullptr &&
      corner_fraction_grad_r != nullptr && corner_fraction_grad_z != nullptr) {
    const int active_nverts =
        tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, cell);
    for (int k = 0; k < active_nverts; ++k) {
      const double* f_lag = corner_fraction_lag + k * n_cells;
      const double* f_grad_r = corner_fraction_grad_r + k * n_cells;
      const double* f_grad_z = corner_fraction_grad_z + k * n_cells;
      const double f_integral = csr_moments_direct_field_integral(
          f_lag,
          f_grad_r,
          f_grad_z,
          old_centroid_r,
          old_centroid_z,
          x_r_old,
          x_z_old,
          face_adj_csr_offsets,
          face_adj_csr_indices,
          cell_node_csr_offsets,
          cell_node_csr_indices,
          cell_nverts,
          inactive_cell_mask,
          vol_lag[donor],
          donor,
          n_cells,
          dV,
          dMr,
          dMz,
          no_face_clip,
          remap_dispatch_audit);
      const double f_bar = f_lag[donor];
      const double product_integral =
          f_bar * rho_integral +
          rho_lag[donor] * (f_integral - f_bar * dV);
      stage.corner_fraction_mass[k * stage.n_face_sides + stage_slot] +=
          flux_scale * product_integral;
    }
  }
  if (stage.gas_tracer_mass != nullptr && gas_tracer_lag != nullptr &&
      gas_tracer_grad_r != nullptr && gas_tracer_grad_z != nullptr) {
    const double tracer_integral = csr_moments_direct_field_integral(
        gas_tracer_lag,
        gas_tracer_grad_r,
        gas_tracer_grad_z,
        old_centroid_r,
        old_centroid_z,
        x_r_old,
        x_z_old,
        face_adj_csr_offsets,
        face_adj_csr_indices,
        cell_node_csr_offsets,
        cell_node_csr_indices,
        cell_nverts,
        inactive_cell_mask,
        vol_lag[donor],
        donor,
        n_cells,
        dV,
        dMr,
        dMz,
        no_face_clip,
        remap_dispatch_audit);
    const double tracer_bar = gas_tracer_lag[donor];
    const double product_integral =
        tracer_bar * rho_integral +
        rho_lag[donor] * (tracer_integral - tracer_bar * dV);
    stage.gas_tracer_mass[stage_slot] += flux_scale * product_integral;
  }
  if (stage.burn_species_mass != nullptr && burn_species_Y_lag != nullptr) {
    for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
      stage.burn_species_mass[s * stage.n_face_sides + stage_slot] +=
          dm * burn_species_Y_lag[donor * tenryu::burn::kNumSpecies + s];
    }
  }
  if (stage.hot_e_eps_mass != nullptr && hot_e_eps_lag != nullptr) {
    stage.hot_e_eps_mass[stage_slot] +=
        dm * fmax(hot_e_eps_lag[donor], 0.0);
  }
  if (stage.burn_eps_mass != nullptr && burn_eps_lag != nullptr) {
    stage.burn_eps_mass[stage_slot] +=
        dm * fmax(burn_eps_lag[donor], 0.0);
  }
  return dm;
}

template <bool GclAudit>
__global__ void csr_apply_internal_hydro_flux_second_order_kernel(
    const CsrHydroFluxStageDeviceView stage,
    const double* __restrict__ rho_lag,
    const double* __restrict__ rho_grad_r,
    const double* __restrict__ rho_grad_z,
    const double* __restrict__ vol_lag,
    const double* __restrict__ old_centroid_r,
    const double* __restrict__ old_centroid_z,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const double* __restrict__ e_e_lag,
    const double* __restrict__ e_i_lag,
    const double* __restrict__ e_tot_lag,
    const double* __restrict__ ye_int_lag,
    const double* __restrict__ corner_fraction_lag,
    const double* __restrict__ corner_fraction_grad_r,
    const double* __restrict__ corner_fraction_grad_z,
    const double* __restrict__ gas_tracer_lag,
    const double* __restrict__ burn_species_Y_lag,
    const double* __restrict__ hot_e_eps_lag,
    const double* __restrict__ burn_eps_lag,
    const double* __restrict__ gas_tracer_grad_r,
    const double* __restrict__ gas_tracer_grad_z,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ unique_cell_a,
    const int* __restrict__ unique_cell_b,
    const int* __restrict__ unique_local_a,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ mass_flux_scale,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    double* __restrict__ macro_flux_audit,
    const int n_faces,
    const int n_cells,
    const bool no_face_clip,
    const int watch_cell,
    const RemapDispatchAuditDeviceView remap_dispatch_audit,
    const CsrGclAuditDeviceView gcl_audit) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int cell_a = unique_cell_a[f];
  const int cell_b = unique_cell_b[f];
  if (csr_face_touches_inactive(inactive_cell_mask, cell_a, cell_b)) {
    csr_note_inactive_face_skip(macro_flux_audit);
    return;
  }
  const int local_a = unique_local_a[f];
  double dMr = 0.0;
  double dMz = 0.0;
  const double dV_a = csr_face_swept_raw_moments_outward(
      x_r_old,
      x_z_old,
      x_r_new,
      x_z_new,
      cell_node_csr_offsets,
      cell_node_csr_indices,
      cell_orientation_sign,
      cell_a,
      local_a,
      cell_nverts,
      &dMr,
      &dMz);
  const int donor = csr_internal_flux_donor(cell_a, cell_b, dV_a);
  const int losing_cell = csr_internal_flux_losing_cell(cell_a, cell_b, dV_a);
  const double flux_scale =
      csr_clamped_flux_scale(mass_flux_scale, losing_cell);
  const double dV_limited = flux_scale * dV_a;
  if (finite_nonzero(dV_a)) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::ExactSweptMoment,
        donor);
  }
  const double Fmass =
      csr_apply_hydro_flux_second_order(stage,
                                    rho_lag,
                                    rho_grad_r,
                                    rho_grad_z,
                                    v_r_cell,
                                    v_z_cell,
                                    e_e_lag,
                                    e_i_lag,
                                    e_tot_lag,
                                    ye_int_lag,
                                    corner_fraction_lag,
                                    corner_fraction_grad_r,
                                    corner_fraction_grad_z,
                                    gas_tracer_lag,
                                    burn_species_Y_lag,
                                    hot_e_eps_lag,
                                    burn_eps_lag,
                                    gas_tracer_grad_r,
                                    gas_tracer_grad_z,
                                    vol_lag,
                                    old_centroid_r,
                                    old_centroid_z,
                                    cell_nverts,
                                    x_r_old,
                                    x_z_old,
                                    face_adj_csr_offsets,
                                    face_adj_csr_indices,
                                    cell_node_csr_offsets,
                                    cell_node_csr_indices,
                                    inactive_cell_mask,
                                    cell_a,
                                    donor,
                                    n_cells,
                                    2 * f,
                                    dV_a,
                                    dMr,
                                    dMz,
                                    flux_scale,
                                    no_face_clip,
                                    remap_dispatch_audit);
  csr_apply_hydro_flux_second_order(stage,
                                    rho_lag,
                                    rho_grad_r,
                                    rho_grad_z,
                                    v_r_cell,
                                    v_z_cell,
                                    e_e_lag,
                                    e_i_lag,
                                    e_tot_lag,
                                    ye_int_lag,
                                    corner_fraction_lag,
                                    corner_fraction_grad_r,
                                    corner_fraction_grad_z,
                                    gas_tracer_lag,
                                    burn_species_Y_lag,
                                    hot_e_eps_lag,
                                    burn_eps_lag,
                                    gas_tracer_grad_r,
                                    gas_tracer_grad_z,
                                    vol_lag,
                                    old_centroid_r,
                                    old_centroid_z,
                                    cell_nverts,
                                    x_r_old,
                                    x_z_old,
                                    face_adj_csr_offsets,
                                    face_adj_csr_indices,
                                    cell_node_csr_offsets,
                                    cell_node_csr_indices,
                                    inactive_cell_mask,
                                    cell_b,
                                    donor,
                                    n_cells,
                                    2 * f + 1,
                                    -dV_a,
                                    -dMr,
                                    -dMz,
                                    flux_scale,
                                    no_face_clip,
                                    remap_dispatch_audit);
  if (cell_a == watch_cell || cell_b == watch_cell) {
    int node_a = -1;
    int node_b = -1;
    if (detail::csr_face_swept_node_indices(cell_node_csr_offsets,
                                             cell_node_csr_indices,
                                             cell_nverts,
                                             cell_a,
                                             local_a,
                                             &node_a,
                                             &node_b)) {
      printf("[watchface] f=%d cellA=%d cellB=%d nA=%d nB=%d "
             "oldA=(%.17e,%.17e) oldB=(%.17e,%.17e) "
             "newA=(%.17e,%.17e) newB=(%.17e,%.17e) "
             "dVq=%.17e dMr=%.17e dMz=%.17e s=%.6e donor=%d "
             "Fmass=%.17e qbar=%.17e gradR=%.17e gradZ=%.17e "
             "cbarR=%.17e cbarZ=%.17e\n",
             f,
             cell_a,
             cell_b,
             node_a,
             node_b,
             x_r_old[node_a],
             x_z_old[node_a],
             x_r_old[node_b],
             x_z_old[node_b],
             x_r_new[node_a],
             x_z_new[node_a],
             x_r_new[node_b],
             x_z_new[node_b],
             dV_a,
             dMr,
             dMz,
             flux_scale,
             donor,
             Fmass,
             rho_lag[donor],
             rho_grad_r[donor],
             rho_grad_z[donor],
             old_centroid_r[donor],
             old_centroid_z[donor]);
    }
  }
  csr_gcl_audit_store_internal_face<GclAudit>(gcl_audit, f, dV_limited);
}

template <bool GclAudit>
__global__ void csr_apply_boundary_hydro_flux_second_order_kernel(
    const CsrHydroFluxStageDeviceView stage,
    const double* __restrict__ rho_lag,
    const double* __restrict__ rho_grad_r,
    const double* __restrict__ rho_grad_z,
    const double* __restrict__ vol_lag,
    const double* __restrict__ old_centroid_r,
    const double* __restrict__ old_centroid_z,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const double* __restrict__ e_e_lag,
    const double* __restrict__ e_i_lag,
    const double* __restrict__ e_tot_lag,
    const double* __restrict__ ye_int_lag,
    const double* __restrict__ corner_fraction_lag,
    const double* __restrict__ corner_fraction_grad_r,
    const double* __restrict__ corner_fraction_grad_z,
    const double* __restrict__ gas_tracer_lag,
    const double* __restrict__ burn_species_Y_lag,
    const double* __restrict__ hot_e_eps_lag,
    const double* __restrict__ burn_eps_lag,
    const double* __restrict__ gas_tracer_grad_r,
    const double* __restrict__ gas_tracer_grad_z,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ boundary_cell,
    const int* __restrict__ boundary_local,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ mass_flux_scale,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    double* __restrict__ macro_flux_audit,
    const int edge_offset,
    const int n_faces,
    const int n_cells,
    const bool no_face_clip,
    const int watch_cell,
    const RemapDispatchAuditDeviceView remap_dispatch_audit,
    const CsrGclAuditDeviceView gcl_audit) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int cell = boundary_cell[f];
  if (csr_inactive_cell(inactive_cell_mask, cell)) {
    csr_note_inactive_face_skip(macro_flux_audit);
    return;
  }
  const int local = boundary_local[f];
  double dMr = 0.0;
  double dMz = 0.0;
  const double dV = csr_face_swept_raw_moments_outward(
      x_r_old,
      x_z_old,
      x_r_new,
      x_z_new,
      cell_node_csr_offsets,
      cell_node_csr_indices,
      cell_orientation_sign,
      cell,
      local,
      cell_nverts,
      &dMr,
      &dMz);
  const double flux_scale =
      dV < 0.0 ? csr_clamped_flux_scale(mass_flux_scale, cell) : 1.0;
  const double dV_limited = flux_scale * dV;
  if (finite_nonzero(dV)) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::ExactSweptMoment,
        cell);
  }
  if (finite_nonzero(dV_limited)) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::BoundaryOneSided,
        cell);
  }
  const double Fmass =
      csr_apply_hydro_flux_second_order(stage,
                                    rho_lag,
                                    rho_grad_r,
                                    rho_grad_z,
                                    v_r_cell,
                                    v_z_cell,
                                    e_e_lag,
                                    e_i_lag,
                                    e_tot_lag,
                                    ye_int_lag,
                                    corner_fraction_lag,
                                    corner_fraction_grad_r,
                                    corner_fraction_grad_z,
                                    gas_tracer_lag,
                                    burn_species_Y_lag,
                                    hot_e_eps_lag,
                                    burn_eps_lag,
                                    gas_tracer_grad_r,
                                    gas_tracer_grad_z,
                                    vol_lag,
                                    old_centroid_r,
                                    old_centroid_z,
                                    cell_nverts,
                                    x_r_old,
                                    x_z_old,
                                    face_adj_csr_offsets,
                                    face_adj_csr_indices,
                                    cell_node_csr_offsets,
                                    cell_node_csr_indices,
                                    inactive_cell_mask,
                                    cell,
                                    cell,
                                    n_cells,
                                    2 * (edge_offset + f),
                                    dV,
                                    dMr,
                                    dMz,
                                    flux_scale,
                                    no_face_clip,
                                    remap_dispatch_audit);
  if (cell == watch_cell) {
    int node_a = -1;
    int node_b = -1;
    if (detail::csr_face_swept_node_indices(cell_node_csr_offsets,
                                             cell_node_csr_indices,
                                             cell_nverts,
                                             cell,
                                             local,
                                             &node_a,
                                             &node_b)) {
      printf("[watchface] f=%d cellA=%d cellB=%d nA=%d nB=%d "
             "oldA=(%.17e,%.17e) oldB=(%.17e,%.17e) "
             "newA=(%.17e,%.17e) newB=(%.17e,%.17e) "
             "dVq=%.17e dMr=%.17e dMz=%.17e s=%.6e donor=%d "
             "Fmass=%.17e qbar=%.17e gradR=%.17e gradZ=%.17e "
             "cbarR=%.17e cbarZ=%.17e\n",
             f,
             cell,
             -1,
             node_a,
             node_b,
             x_r_old[node_a],
             x_z_old[node_a],
             x_r_old[node_b],
             x_z_old[node_b],
             x_r_new[node_a],
             x_z_new[node_a],
             x_r_new[node_b],
             x_z_new[node_b],
             dV,
             dMr,
             dMz,
             flux_scale,
             cell,
             Fmass,
             rho_lag[cell],
             rho_grad_r[cell],
             rho_grad_z[cell],
             old_centroid_r[cell],
             old_centroid_z[cell]);
    }
  }
  csr_gcl_audit_store_boundary_face<GclAudit>(gcl_audit, f, dV_limited);
}

__global__ void csr_eta_contact_hotspot_volume_kernel(
    double* __restrict__ eta_contact_diag,
    const double* __restrict__ gas_tracer_Y,
    const double* __restrict__ vol,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    return;
  }
  const double Y = clamp01_device(gas_tracer_Y[c]);
  const double V = vol[c];
  if (Y > 0.0 && V > 0.0 && isfinite(Y) && isfinite(V)) {
    atomicAdd(eta_contact_diag + 1, Y * V);
  }
}

__global__ void csr_eta_contact_swept_volume_kernel(
    double* __restrict__ eta_contact_diag,
    const bool second_order,
    const double* __restrict__ gas_tracer_Y,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ unique_cell_a,
    const int* __restrict__ unique_cell_b,
    const int* __restrict__ unique_local_a,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ mass_flux_scale,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_faces) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int cell_a = unique_cell_a[f];
  const int cell_b = unique_cell_b[f];
  if (csr_face_touches_inactive(inactive_cell_mask, cell_a, cell_b)) {
    return;
  }
  const bool gas_a = gas_tracer_Y[cell_a] > 0.5;
  const bool gas_b = gas_tracer_Y[cell_b] > 0.5;
  if (gas_a == gas_b) {
    return;
  }
  const int local_a = unique_local_a[f];
  double dMr = 0.0;
  double dMz = 0.0;
  const double dV_a = second_order
                          ? csr_face_swept_raw_moments_outward(
                                x_r_old,
                                x_z_old,
                                x_r_new,
                                x_z_new,
                                cell_node_csr_offsets,
                                cell_node_csr_indices,
                                cell_orientation_sign,
                                cell_a,
                                local_a,
                                cell_nverts,
                                &dMr,
                                &dMz)
                          : csr_face_swept_volume_outward(
                                x_r_old,
                                x_z_old,
                                x_r_new,
                                x_z_new,
                                cell_node_csr_offsets,
                                cell_node_csr_indices,
                                cell_orientation_sign,
                                cell_a,
                                local_a,
                                cell_nverts);
  if (!finite_nonzero(dV_a)) {
    return;
  }
  const int losing_cell = csr_internal_flux_losing_cell(cell_a, cell_b, dV_a);
  const double dV_limited =
      dV_a * csr_clamped_flux_scale(mass_flux_scale, losing_cell);
  atomicAdd(eta_contact_diag, fabs(dV_limited));
}

__global__ void csr_accumulate_internal_hydro_outgoing_mass_kernel(
    double* __restrict__ outgoing_mass_stage,
    const bool second_order,
    const double* __restrict__ rho_lag,
    const double* __restrict__ rho_grad_r,
    const double* __restrict__ rho_grad_z,
    const double* __restrict__ vol_lag,
    const double* __restrict__ old_centroid_r,
    const double* __restrict__ old_centroid_z,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ unique_cell_a,
    const int* __restrict__ unique_cell_b,
    const int* __restrict__ unique_local_a,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_faces,
    const int n_cells) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int cell_a = unique_cell_a[f];
  const int cell_b = unique_cell_b[f];
  if (csr_face_touches_inactive(inactive_cell_mask, cell_a, cell_b)) {
    return;
  }
  const int local_a = unique_local_a[f];
  double dMr = 0.0;
  double dMz = 0.0;
  const double dV_a = second_order
                          ? csr_face_swept_raw_moments_outward(
                                x_r_old,
                                x_z_old,
                                x_r_new,
                                x_z_new,
                                cell_node_csr_offsets,
                                cell_node_csr_indices,
                                cell_orientation_sign,
                                cell_a,
                                local_a,
                                cell_nverts,
                                &dMr,
                                &dMz)
                          : csr_face_swept_volume_outward(
                                x_r_old,
                                x_z_old,
                                x_r_new,
                                x_z_new,
                                cell_node_csr_offsets,
                                cell_node_csr_indices,
                                cell_orientation_sign,
                                cell_a,
                                local_a,
                                cell_nverts);
  if (!finite_nonzero(dV_a)) {
    return;
  }
  const int donor = csr_internal_flux_donor(cell_a, cell_b, dV_a);
  const int losing_cell = csr_internal_flux_losing_cell(cell_a, cell_b, dV_a);
  double dm_out = 0.0;
  if (second_order) {
    const double rho_integral = csr_moments_direct_field_integral(
        rho_lag,
        rho_grad_r,
        rho_grad_z,
        old_centroid_r,
        old_centroid_z,
        x_r_old,
        x_z_old,
        face_adj_csr_offsets,
        face_adj_csr_indices,
        cell_node_csr_offsets,
        cell_node_csr_indices,
        cell_nverts,
        inactive_cell_mask,
        vol_lag[donor],
        donor,
        n_cells,
        dV_a,
        dMr,
        dMz,
        false,
        {});
    dm_out = fabs(rho_integral);
  } else {
    double rho_face = rho_lag[donor];
    rho_face = (rho_face > 0.0 && isfinite(rho_face)) ? rho_face : 0.0;
    dm_out = rho_face * fabs(dV_a);
  }
  if (dm_out > 0.0 && isfinite(dm_out)) {
    const int side = losing_cell == cell_a ? 0 : 1;
    outgoing_mass_stage[2 * f + side] += dm_out;
  }
}

__global__ void csr_accumulate_boundary_hydro_outgoing_mass_kernel(
    double* __restrict__ outgoing_mass_stage,
    const bool second_order,
    const double* __restrict__ rho_lag,
    const double* __restrict__ rho_grad_r,
    const double* __restrict__ rho_grad_z,
    const double* __restrict__ vol_lag,
    const double* __restrict__ old_centroid_r,
    const double* __restrict__ old_centroid_z,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ boundary_cell,
    const int* __restrict__ boundary_local,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int edge_offset,
    const int n_faces,
    const int n_cells) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int cell = boundary_cell[f];
  if (csr_inactive_cell(inactive_cell_mask, cell)) {
    return;
  }
  const int local = boundary_local[f];
  double dMr = 0.0;
  double dMz = 0.0;
  const double dV = second_order
                        ? csr_face_swept_raw_moments_outward(
                              x_r_old,
                              x_z_old,
                              x_r_new,
                              x_z_new,
                              cell_node_csr_offsets,
                              cell_node_csr_indices,
                              cell_orientation_sign,
                              cell,
                              local,
                              cell_nverts,
                              &dMr,
                              &dMz)
                        : csr_face_swept_volume_outward(
                              x_r_old,
                              x_z_old,
                              x_r_new,
                              x_z_new,
                              cell_node_csr_offsets,
                              cell_node_csr_indices,
                              cell_orientation_sign,
                              cell,
                              local,
                              cell_nverts);
  if (!(dV < 0.0) || !finite_nonzero(dV)) {
    return;
  }
  double dm_out = 0.0;
  if (second_order) {
    const double rho_integral = csr_moments_direct_field_integral(
        rho_lag,
        rho_grad_r,
        rho_grad_z,
        old_centroid_r,
        old_centroid_z,
        x_r_old,
        x_z_old,
        face_adj_csr_offsets,
        face_adj_csr_indices,
        cell_node_csr_offsets,
        cell_node_csr_indices,
        cell_nverts,
        inactive_cell_mask,
        vol_lag[cell],
        cell,
        n_cells,
        dV,
        dMr,
        dMz,
        false,
        {});
    dm_out = fabs(rho_integral);
  } else {
    double rho_face = rho_lag[cell];
    rho_face = (rho_face > 0.0 && isfinite(rho_face)) ? rho_face : 0.0;
    dm_out = rho_face * (-dV);
  }
  if (dm_out > 0.0 && isfinite(dm_out)) {
    outgoing_mass_stage[2 * (edge_offset + f)] += dm_out;
  }
}

__global__ void csr_compute_hydro_mass_flux_scale_kernel(
    double* __restrict__ mass_flux_scale,
    const double* __restrict__ outgoing_mass,
    const double* __restrict__ mass_lag,
    const double* __restrict__ rho_lag,
    const double* __restrict__ vol_lag,
    const double* __restrict__ vol_new,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells,
    const double rho_floor,
    const RemapDispatchAuditDeviceView remap_dispatch_audit) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    mass_flux_scale[c] = 0.0;
    return;
  }
  double m = fmax(mass_lag[c], 0.0);
  if (!isfinite(m)) {
    m = fmax(rho_lag[c], 0.0) * fmax(vol_lag[c], 0.0);
  }
  const double V = fmax(vol_new[c], kTinyVolume);
  const double mass_floor = fmax(rho_floor, 0.0) * V;
  const double available = fmax(m - mass_floor, 0.0);
  const double outgoing =
      (outgoing_mass[c] > 0.0 && isfinite(outgoing_mass[c])) ? outgoing_mass[c] : 0.0;
  double scale = 1.0;
  if (outgoing > available && outgoing > 0.0) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::LimiterActivation,
        c);
    const double guarded_available = available * (1.0 - 1.0e-12);
    scale = fmax(guarded_available, 0.0) / outgoing;
  }
  mass_flux_scale[c] = fmin(1.0, fmax(0.0, scale));
}

__global__ void csr_compute_lsq_gradients_inactive_masked_kernel(
    double* __restrict__ grad_r,
    double* __restrict__ grad_z,
    const double* __restrict__ field,
    const double* __restrict__ old_centroid_r,
    const double* __restrict__ old_centroid_z,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const std::uint8_t* __restrict__ donor_fallback_cell_mask,
    const int n_cells,
    const RemapDispatchAuditDeviceView remap_dispatch_audit,
    const double condition_floor = 0.0,
    const double* __restrict__ screen_mass = nullptr,
    const double mass_floor = 0.0) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    grad_r[c] = 0.0;
    grad_z[c] = 0.0;
    return;
  }
  if (donor_fallback_cell_mask != nullptr &&
      donor_fallback_cell_mask[c] != 0U) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::FirstOrderDonorFallback,
        c);
    grad_r[c] = 0.0;
    grad_z[c] = 0.0;
    return;
  }
  if (screen_mass != nullptr &&
      (screen_mass[c] <= mass_floor || !isfinite(screen_mass[c]) ||
       !isfinite(field[c]))) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::ProjectionGradientConditionFallback,
        c);
    grad_r[c] = 0.0;
    grad_z[c] = 0.0;
    return;
  }

  const double q_c = field[c];
  const double r_c = old_centroid_r[c];
  const double z_c = old_centroid_z[c];

  double a00 = 0.0;
  double a01 = 0.0;
  double a11 = 0.0;
  double b0 = 0.0;
  double b1 = 0.0;
  int valid_neighbors = 0;

  const int off = face_adj_csr_offsets[c];
  const int end = face_adj_csr_offsets[c + 1];
  const int active_nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  for (int p = off; p < end; ++p) {
    const int local = p - off;
    if (!tenryu::mesh::mesh_topo_local_face_is_active(active_nverts, local)) {
      continue;
    }
    const int nb = face_adj_csr_indices[p];
    if (nb < 0 || nb >= n_cells || csr_inactive_cell(inactive_cell_mask, nb)) {
      continue;
    }
    if (screen_mass != nullptr &&
        (screen_mass[nb] <= mass_floor || !isfinite(screen_mass[nb]) ||
         !isfinite(field[nb]))) {
      continue;
    }
    const double dr = old_centroid_r[nb] - r_c;
    const double dz = old_centroid_z[nb] - z_c;
    const double dist2 = dr * dr + dz * dz;
    if (!(dist2 > 1.0e-300) || !isfinite(dist2)) {
      continue;
    }
    const double w = 1.0 / dist2;
    const double dq = field[nb] - q_c;
    a00 += w * dr * dr;
    a01 += w * dr * dz;
    a11 += w * dz * dz;
    b0 += w * dr * dq;
    b1 += w * dz * dq;
    ++valid_neighbors;
  }

  if (valid_neighbors < 2) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::CsrGradientZeroFallback,
        c);
    grad_r[c] = 0.0;
    grad_z[c] = 0.0;
    return;
  }

  const double det = a00 * a11 - a01 * a01;
  const double scale = fabs(a00 * a11) + fabs(a01 * a01) + 1.0e-300;
  const bool condition_accepted =
      condition_floor > 0.0
          ? det > condition_floor * (a00 + a11) * (a00 + a11)
          : fabs(det) > 1.0e-24 * scale;
  if (condition_accepted && isfinite(det)) {
    grad_r[c] = (a11 * b0 - a01 * b1) / det;
    grad_z[c] = (-a01 * b0 + a00 * b1) / det;
  } else {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        condition_floor > 0.0
            ? RemapDispatchAuditCounter::ProjectionGradientConditionFallback
            : RemapDispatchAuditCounter::CsrGradientZeroFallback,
        c);
    grad_r[c] = 0.0;
    grad_z[c] = 0.0;
  }
}

__global__ void csr_apply_barth_jespersen_limiter_inactive_masked_kernel(
    double* __restrict__ grad_r,
    double* __restrict__ grad_z,
    const double* __restrict__ field,
    const double* __restrict__ old_centroid_r,
    const double* __restrict__ old_centroid_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const std::uint8_t* __restrict__ donor_fallback_cell_mask,
    const int n_cells,
    const RemapDispatchAuditDeviceView remap_dispatch_audit) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c) ||
      (donor_fallback_cell_mask != nullptr &&
       donor_fallback_cell_mask[c] != 0U)) {
    grad_r[c] = 0.0;
    grad_z[c] = 0.0;
    return;
  }

  const double q_c = field[c];
  double q_min = q_c;
  double q_max = q_c;
  const int adj_off = face_adj_csr_offsets[c];
  const int adj_end = face_adj_csr_offsets[c + 1];
  const int active_nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  for (int p = adj_off; p < adj_end; ++p) {
    const int local = p - adj_off;
    if (!tenryu::mesh::mesh_topo_local_face_is_active(active_nverts, local)) {
      continue;
    }
    const int nb = face_adj_csr_indices[p];
    if (nb < 0 || nb >= n_cells || csr_inactive_cell(inactive_cell_mask, nb)) {
      continue;
    }
    const double q = field[nb];
    q_min = fmin(q_min, q);
    q_max = fmax(q_max, q);
  }

  if (!(q_max > q_min) || !isfinite(grad_r[c]) || !isfinite(grad_z[c])) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::CsrGradientZeroFallback,
        c);
    grad_r[c] = 0.0;
    grad_z[c] = 0.0;
    return;
  }

  const double r_c = old_centroid_r[c];
  const double z_c = old_centroid_z[c];
  double alpha = 1.0;
  for (int local = 0; local < active_nverts; ++local) {
    double r_f = 0.0;
    double z_f = 0.0;
    if (!detail::csr_face_center(x_r,
                                 x_z,
                                 cell_node_csr_offsets,
                                 cell_node_csr_indices,
                                 c,
                                 local,
                                 r_f,
                                 z_f,
                                 cell_nverts)) {
      continue;
    }
    const double q_face =
        q_c + grad_r[c] * (r_f - r_c) + grad_z[c] * (z_f - z_c);
    alpha = fmin(alpha,
                 detail::csr_barth_jespersen_limiter_ratio(
                     q_c, q_min, q_max, q_face));
  }

  if (!isfinite(alpha)) {
    alpha = 0.0;
  }
  alpha = fmin(1.0, fmax(0.0, alpha));
  if (alpha < 1.0) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::LimiterActivation,
        c);
  }
  grad_r[c] *= alpha;
  grad_z[c] *= alpha;
}

__global__ void csr_finish_hydro_remap_kernel(
    double* __restrict__ rho_new,
    double* __restrict__ v_r_cell_new,
    double* __restrict__ v_z_cell_new,
    double* __restrict__ e_e_new,
    double* __restrict__ e_i_new,
    double* __restrict__ mass_new,
    const double* __restrict__ mom_r_new,
    const double* __restrict__ mom_z_new,
    const double* __restrict__ energy_e_new,
    const double* __restrict__ energy_i_new,
    const double* __restrict__ vol_new,
    const double* __restrict__ rho_lag,
    const double* __restrict__ e_e_lag,
    const double* __restrict__ e_i_lag,
    const double* __restrict__ zbar,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells,
    const double rho_floor,
    const double te_floor,
    const double ti_floor,
    const double gamma,
    const double A,
    double* __restrict__ mass_floor_delta,
    double* __restrict__ E_redistribution_unresolved) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    const double m =
        (mass_new[c] > 0.0 && isfinite(mass_new[c])) ? mass_new[c] : 0.0;
    mass_new[c] = m;
    rho_new[c] = rho_lag[c];
    e_e_new[c] = e_e_lag[c];
    e_i_new[c] = e_i_lag[c];
    v_r_cell_new[c] =
        (m > 0.0 && isfinite(mom_r_new[c])) ? (mom_r_new[c] / m) : 0.0;
    v_z_cell_new[c] =
        (m > 0.0 && isfinite(mom_z_new[c])) ? (mom_z_new[c] / m) : 0.0;
    return;
  }
  const double V = fmax(vol_new[c], kTinyVolume);
  const double m_raw = mass_new[c];
  double m = m_raw;
  const double mass_floor = rho_floor * V;

  const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : 1.0;
  const double A_safe = fmax(A, 1.0e-30);
  const double gm1 = fmax(gamma - 1.0, 1.0e-30);
  const double cv_i =
      tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  const double cv_e =
      z * tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  const double e_e_floor = cv_e * fmax(te_floor, 0.0);
  const double e_i_floor = cv_i * fmax(ti_floor, 0.0);

  double ee = e_e_floor;
  double ei = e_i_floor;
  if (m_raw > mass_floor && isfinite(m_raw)) {
    m = m_raw;
    ee = energy_e_new[c] / m;
    ei = energy_i_new[c] / m;
    if (!(ee >= e_e_floor) || !isfinite(ee)) {
      ee = e_e_floor;
    }
    if (!(ei >= e_i_floor) || !isfinite(ei)) {
      ei = e_i_floor;
    }

    rho_new[c] = m / V;
    v_r_cell_new[c] = (m > 0.0 && isfinite(mom_r_new[c])) ? (mom_r_new[c] / m) : 0.0;
    v_z_cell_new[c] = (m > 0.0 && isfinite(mom_z_new[c])) ? (mom_z_new[c] / m) : 0.0;
  } else {
    m = mass_floor;
    ee = e_e_floor;
    ei = e_i_floor;
    rho_new[c] = m / V;
    v_r_cell_new[c] = 0.0;
    v_z_cell_new[c] = 0.0;
    if (mass_floor_delta != nullptr) {
      atomicAdd(mass_floor_delta, fmax(mass_floor - m_raw, 0.0));
    }
    if (E_redistribution_unresolved != nullptr) {
      atomicAdd(E_redistribution_unresolved,
                (m * ee + m * ei) - (energy_e_new[c] + energy_i_new[c]));
    }
  }

  mass_new[c] = m;
  e_e_new[c] = ee;
  e_i_new[c] = ei;
}

__global__ void csr_finish_corner_fraction_remap_kernel(
    double* __restrict__ corner_mass,
    const double* __restrict__ corner_fraction_mass_new,
    const double* __restrict__ mass_new,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int corner_stride,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    return;
  }
  const double m = (mass_new[c] > 0.0 && isfinite(mass_new[c])) ? mass_new[c] : 0.0;
  if (cell_nverts == nullptr) {
    double f[4] = {0.0, 0.0, 0.0, 0.0};
    if (m > 0.0) {
      for (int k = 0; k < 4; ++k) {
        const double raw = corner_fraction_mass_new[k * n_cells + c] / m;
        f[k] = (raw > 0.0 && isfinite(raw)) ? raw : 0.0;
      }
    }
    const double sum = f[0] + f[1] + f[2] + f[3];
    double cm0 = 0.25 * m;
    double cm1 = 0.25 * m;
    double cm2 = 0.25 * m;
    double cm3 = m - (cm0 + cm1 + cm2);
    if (sum > 1.0e-300 && isfinite(sum)) {
      const double inv_sum = 1.0 / sum;
      cm0 = m * f[0] * inv_sum;
      cm1 = m * f[1] * inv_sum;
      cm2 = m * f[2] * inv_sum;
      cm3 = m - (cm0 + cm1 + cm2);
      if (!isfinite(cm3)) {
        cm3 = m * f[3] * inv_sum;
      }
    }
    const int base = c * corner_stride;
    corner_mass[base + 0] = cm0;
    corner_mass[base + 1] = cm1;
    corner_mass[base + 2] = cm2;
    corner_mass[base + 3] = cm3;
    for (int k = 4; k < corner_stride; ++k) {
      corner_mass[base + k] = 0.0;
    }
    return;
  }
  const int active_nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  double f[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  if (m > 0.0) {
    for (int k = 0; k < active_nverts; ++k) {
      const double raw = corner_fraction_mass_new[k * n_cells + c] / m;
      f[k] = (raw > 0.0 && isfinite(raw)) ? raw : 0.0;
    }
  }
  double sum = 0.0;
  for (int k = 0; k < active_nverts; ++k) {
    sum += f[k];
  }
  double cm[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  if (sum > 1.0e-300 && isfinite(sum)) {
    const double inv_sum = 1.0 / sum;
    double partial = 0.0;
    for (int k = 0; k + 1 < active_nverts; ++k) {
      cm[k] = m * f[k] * inv_sum;
      partial += cm[k];
    }
    cm[active_nverts - 1] = m - partial;
    if (!isfinite(cm[active_nverts - 1])) {
      cm[active_nverts - 1] = m * f[active_nverts - 1] * inv_sum;
    }
  } else {
    const double uniform = m / static_cast<double>(active_nverts);
    double partial = 0.0;
    for (int k = 0; k + 1 < active_nverts; ++k) {
      cm[k] = uniform;
      partial += cm[k];
    }
    cm[active_nverts - 1] = m - partial;
  }
  const int base = c * corner_stride;
  for (int k = 0; k < corner_stride; ++k) {
    corner_mass[base + k] = cm[k];
  }
}

__global__ void csr_finish_gas_tracer_remap_kernel(
    double* __restrict__ gas_tracer_Y,
    const double* __restrict__ gas_tracer_mass_new,
    const double* __restrict__ mass_new,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    return;
  }
  const double m = mass_new[c];
  if (!(m > 0.0) || !isfinite(m)) {
    gas_tracer_Y[c] = 0.0;
    return;
  }
  gas_tracer_Y[c] = clamp01_device(gas_tracer_mass_new[c] / m);
}

__global__ void csr_finish_hot_e_eps_remap_kernel(
    double* __restrict__ hot_e_eps_out,
    const double* __restrict__ hot_e_eps_mass_new,
    const double* __restrict__ mass_new,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    return;
  }
  const double m = mass_new[c];
  if (!(m > 0.0) || !isfinite(m)) {
    hot_e_eps_out[c] = 0.0;
    return;
  }
  hot_e_eps_out[c] = fmax(hot_e_eps_mass_new[c] / m, 0.0);
}

__global__ void csr_finish_burn_eps_remap_kernel(
    double* __restrict__ burn_eps_out,
    const double* __restrict__ burn_eps_mass_new,
    const double* __restrict__ mass_new,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    return;
  }
  const double m = mass_new[c];
  if (!(m > 0.0) || !isfinite(m)) {
    burn_eps_out[c] = 0.0;
    return;
  }
  burn_eps_out[c] = fmax(burn_eps_mass_new[c] / m, 0.0);
}

__global__ void csr_finish_burn_species_remap_kernel(
    double* __restrict__ species_Y,
    const double* __restrict__ species_mass_new,
    const double* __restrict__ mass_new,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    return;
  }
  const double m = mass_new[c];
  if (!(m > 0.0) || !isfinite(m)) {
    for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
      species_Y[c * tenryu::burn::kNumSpecies + s] = 0.0;
    }
    return;
  }
  for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
    species_Y[c * tenryu::burn::kNumSpecies + s] =
        species_mass_new[c * tenryu::burn::kNumSpecies + s] / m;
  }
}

__global__ void csr_initialize_volume_scalar_extents_kernel(
    double* __restrict__ scalar_ext_new,
    const double* __restrict__ scalar_lag,
    const double* __restrict__ vol_lag,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  scalar_ext_new[c] = fmax(scalar_lag[c], 0.0) * fmax(vol_lag[c], 0.0);
}

__device__ inline void csr_apply_volume_scalar_flux(
    double* __restrict__ scalar_ext_stage,
    const double* __restrict__ scalar_lag,
    const int donor,
    const int stage_slot,
    const double signed_volume) {
  if (!finite_nonzero(signed_volume)) {
    return;
  }
  scalar_ext_stage[stage_slot] +=
      fmax(scalar_lag[donor], 0.0) * signed_volume;
}

__global__ void csr_apply_internal_volume_scalar_flux_kernel(
    double* __restrict__ scalar_ext_stage,
    const double* __restrict__ scalar_lag,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ unique_cell_a,
    const int* __restrict__ unique_cell_b,
    const int* __restrict__ unique_local_a,
    const int n_faces,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    double* __restrict__ macro_flux_audit,
    const RemapDispatchAuditDeviceView remap_dispatch_audit) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int cell_a = unique_cell_a[f];
  const int cell_b = unique_cell_b[f];
  if (csr_face_touches_inactive(inactive_cell_mask, cell_a, cell_b)) {
    csr_note_inactive_face_skip(macro_flux_audit);
    return;
  }
  const double dV_a = csr_face_swept_volume_outward(x_r_old,
                                                    x_z_old,
                                                    x_r_new,
                                                    x_z_new,
                                                    cell_node_csr_offsets,
                                                    cell_node_csr_indices,
                                                    cell_orientation_sign,
                                                    cell_a,
                                                    unique_local_a[f],
                                                    cell_nverts);
  const int donor = csr_internal_flux_donor(cell_a, cell_b, dV_a);
  if (finite_nonzero(dV_a)) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::LegacySweptVolume,
        donor);
  }
  csr_apply_volume_scalar_flux(scalar_ext_stage, scalar_lag, donor, 2 * f, dV_a);
  csr_apply_volume_scalar_flux(
      scalar_ext_stage, scalar_lag, donor, 2 * f + 1, -dV_a);
}

__global__ void csr_apply_boundary_volume_scalar_flux_kernel(
    double* __restrict__ scalar_ext_stage,
    const double* __restrict__ scalar_lag,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ boundary_cell,
    const int* __restrict__ boundary_local,
    const int edge_offset,
    const int n_faces,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    double* __restrict__ macro_flux_audit,
    const RemapDispatchAuditDeviceView remap_dispatch_audit) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int cell = boundary_cell[f];
  if (csr_inactive_cell(inactive_cell_mask, cell)) {
    csr_note_inactive_face_skip(macro_flux_audit);
    return;
  }
  const double dV = csr_face_swept_volume_outward(x_r_old,
                                                  x_z_old,
                                                  x_r_new,
                                                  x_z_new,
                                                  cell_node_csr_offsets,
                                                  cell_node_csr_indices,
                                                  cell_orientation_sign,
                                                  cell,
                                                  boundary_local[f],
                                                  cell_nverts);
  if (finite_nonzero(dV)) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::LegacySweptVolume,
        cell);
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::BoundaryOneSided,
        cell);
  }
  csr_apply_volume_scalar_flux(
      scalar_ext_stage, scalar_lag, cell, 2 * (edge_offset + f), dV);
}

__device__ inline void csr_apply_volume_scalar_flux_second_order(
    double* __restrict__ scalar_ext_stage,
    const double* __restrict__ scalar_lag,
    const double* __restrict__ scalar_grad_r,
    const double* __restrict__ scalar_grad_z,
    const double* __restrict__ vol_lag,
    const double* __restrict__ old_centroid_r,
    const double* __restrict__ old_centroid_z,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int donor,
    const int n_cells,
    const int stage_slot,
    const double dV,
    const double dMr,
    const double dMz,
    const double flux_scale,
    const RemapDispatchAuditDeviceView remap_dispatch_audit) {
  if (!finite_nonzero(dV)) {
    return;
  }
  const double scalar_integral = csr_moments_direct_field_integral(
      scalar_lag,
      scalar_grad_r,
      scalar_grad_z,
      old_centroid_r,
      old_centroid_z,
      x_r_old,
      x_z_old,
      face_adj_csr_offsets,
      face_adj_csr_indices,
      cell_node_csr_offsets,
      cell_node_csr_indices,
      cell_nverts,
      inactive_cell_mask,
      vol_lag[donor],
      donor,
      n_cells,
      dV,
      dMr,
      dMz,
      false,
      remap_dispatch_audit);
  scalar_ext_stage[stage_slot] += flux_scale * scalar_integral;
}

__global__ void csr_apply_internal_volume_scalar_flux_second_order_kernel(
    double* __restrict__ scalar_ext_stage,
    const double* __restrict__ scalar_lag,
    const double* __restrict__ scalar_grad_r,
    const double* __restrict__ scalar_grad_z,
    const double* __restrict__ vol_lag,
    const double* __restrict__ old_centroid_r,
    const double* __restrict__ old_centroid_z,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ unique_cell_a,
    const int* __restrict__ unique_cell_b,
    const int* __restrict__ unique_local_a,
    const int n_faces,
    const int n_cells,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ mass_flux_scale,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    double* __restrict__ macro_flux_audit,
    const RemapDispatchAuditDeviceView remap_dispatch_audit) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int cell_a = unique_cell_a[f];
  const int cell_b = unique_cell_b[f];
  if (csr_face_touches_inactive(inactive_cell_mask, cell_a, cell_b)) {
    csr_note_inactive_face_skip(macro_flux_audit);
    return;
  }
  const int local_a = unique_local_a[f];
  double dMr = 0.0;
  double dMz = 0.0;
  const double dV_a = csr_face_swept_raw_moments_outward(
      x_r_old,
      x_z_old,
      x_r_new,
      x_z_new,
      cell_node_csr_offsets,
      cell_node_csr_indices,
      cell_orientation_sign,
      cell_a,
      local_a,
      cell_nverts,
      &dMr,
      &dMz);
  const int donor = csr_internal_flux_donor(cell_a, cell_b, dV_a);
  const int losing_cell = csr_internal_flux_losing_cell(cell_a, cell_b, dV_a);
  const double flux_scale =
      csr_clamped_flux_scale(mass_flux_scale, losing_cell);
  if (finite_nonzero(dV_a)) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::ExactSweptMoment,
        donor);
  }
  csr_apply_volume_scalar_flux_second_order(scalar_ext_stage,
                                            scalar_lag,
                                            scalar_grad_r,
                                            scalar_grad_z,
                                            vol_lag,
                                            old_centroid_r,
                                            old_centroid_z,
                                            x_r_old,
                                            x_z_old,
                                            face_adj_csr_offsets,
                                            face_adj_csr_indices,
                                            cell_node_csr_offsets,
                                            cell_node_csr_indices,
                                            cell_nverts,
                                            inactive_cell_mask,
                                            donor,
                                            n_cells,
                                            2 * f,
                                            dV_a,
                                            dMr,
                                            dMz,
                                            flux_scale,
                                            remap_dispatch_audit);
  csr_apply_volume_scalar_flux_second_order(scalar_ext_stage,
                                            scalar_lag,
                                            scalar_grad_r,
                                            scalar_grad_z,
                                            vol_lag,
                                            old_centroid_r,
                                            old_centroid_z,
                                            x_r_old,
                                            x_z_old,
                                            face_adj_csr_offsets,
                                            face_adj_csr_indices,
                                            cell_node_csr_offsets,
                                            cell_node_csr_indices,
                                            cell_nverts,
                                            inactive_cell_mask,
                                            donor,
                                            n_cells,
                                            2 * f + 1,
                                            -dV_a,
                                            -dMr,
                                            -dMz,
                                            flux_scale,
                                            remap_dispatch_audit);
}

__global__ void csr_apply_boundary_volume_scalar_flux_second_order_kernel(
    double* __restrict__ scalar_ext_stage,
    const double* __restrict__ scalar_lag,
    const double* __restrict__ scalar_grad_r,
    const double* __restrict__ scalar_grad_z,
    const double* __restrict__ vol_lag,
    const double* __restrict__ old_centroid_r,
    const double* __restrict__ old_centroid_z,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ boundary_cell,
    const int* __restrict__ boundary_local,
    const int edge_offset,
    const int n_faces,
    const int n_cells,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ mass_flux_scale,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    double* __restrict__ macro_flux_audit,
    const RemapDispatchAuditDeviceView remap_dispatch_audit) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int cell = boundary_cell[f];
  if (csr_inactive_cell(inactive_cell_mask, cell)) {
    csr_note_inactive_face_skip(macro_flux_audit);
    return;
  }
  const int local = boundary_local[f];
  double dMr = 0.0;
  double dMz = 0.0;
  const double dV = csr_face_swept_raw_moments_outward(
      x_r_old,
      x_z_old,
      x_r_new,
      x_z_new,
      cell_node_csr_offsets,
      cell_node_csr_indices,
      cell_orientation_sign,
      cell,
      local,
      cell_nverts,
      &dMr,
      &dMz);
  const double flux_scale =
      dV < 0.0 ? csr_clamped_flux_scale(mass_flux_scale, cell) : 1.0;
  if (finite_nonzero(dV)) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::ExactSweptMoment,
        cell);
  }
  if (finite_nonzero(flux_scale * dV)) {
    remap_dispatch_audit_count(
        remap_dispatch_audit,
        RemapDispatchAuditCounter::BoundaryOneSided,
        cell);
  }
  csr_apply_volume_scalar_flux_second_order(scalar_ext_stage,
                                            scalar_lag,
                                            scalar_grad_r,
                                            scalar_grad_z,
                                            vol_lag,
                                            old_centroid_r,
                                            old_centroid_z,
                                            x_r_old,
                                            x_z_old,
                                            face_adj_csr_offsets,
                                            face_adj_csr_indices,
                                            cell_node_csr_offsets,
                                            cell_node_csr_indices,
                                            cell_nverts,
                                            inactive_cell_mask,
                                            cell,
                                            n_cells,
                                            2 * (edge_offset + f),
                                            dV,
                                            dMr,
                                            dMz,
                                            flux_scale,
                                            remap_dispatch_audit);
}

__global__ void csr_finish_volume_scalar_remap_kernel(
    double* __restrict__ scalar_new,
    const double* __restrict__ scalar_ext_new,
    const double* __restrict__ vol_new,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    return;
  }
  scalar_new[c] = fmax(scalar_ext_new[c] / fmax(vol_new[c], kTinyVolume), 0.0);
}

__device__ inline int csr_active_nverts_for_cell(
    const int c,
    const std::uint8_t* __restrict__ cell_nverts) {
  return (cell_nverts != nullptr)
             ? tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c)
             : 4;
}

constexpr int kAleVelCoherenceBlockSize = 256;

__global__ void ale_velcoherence_reduce_kernel(
    double* __restrict__ block_sums,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ gas_tracer_Y,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells,
    const double R_g_cm) {
  __shared__ double s_M[kAleVelCoherenceBlockSize];
  __shared__ double s_Mur[kAleVelCoherenceBlockSize];
  __shared__ double s_rad_ke[kAleVelCoherenceBlockSize];
  __shared__ double s_tot_ke[kAleVelCoherenceBlockSize];

  const int tid = threadIdx.x;
  const int c = blockIdx.x * blockDim.x + tid;
  double M = 0.0;
  double Mur = 0.0;
  double rad_ke = 0.0;
  double tot_ke = 0.0;

  if (c < n_cells) {
    const int off = cell_node_csr_offsets[c];
    const int end = cell_node_csr_offsets[c + 1];
    const int available_nverts = (end > off) ? (end - off) : 0;
    int active_nverts = csr_active_nverts_for_cell(c, cell_nverts);
    if (active_nverts > available_nverts) {
      active_nverts = available_nverts;
    }
    const double m = mass[c];
    if (active_nverts > 0 && isfinite(m) && m > 0.0) {
      double r_sum = 0.0;
      double z_sum = 0.0;
      double ur_sum = 0.0;
      double vr_sum = 0.0;
      double vz_sum = 0.0;
      for (int k = 0; k < active_nverts; ++k) {
        const int n = cell_node_csr_indices[off + k];
        const double r = x_r[n];
        const double z = x_z[n];
        const double vr = v_r[n];
        const double vz = v_z[n];
        r_sum += r;
        z_sum += z;
        vr_sum += vr;
        vz_sum += vz;
        const double rr = hypot(r, z);
        if (isfinite(rr) && rr > 0.0) {
          ur_sum += (vr * r + vz * z) / rr;
        }
      }
      const double inv_n = 1.0 / static_cast<double>(active_nverts);
      const double rc = r_sum * inv_n;
      const double zc = z_sum * inv_n;
      const bool gas_cell =
          (gas_tracer_Y != nullptr)
              ? (gas_tracer_Y[c] > 0.5)
              : (R_g_cm > 0.0 && hypot(rc, zc) < R_g_cm);
      if (gas_cell) {
        const double ur_cell = ur_sum * inv_n;
        const double vr_cell = vr_sum * inv_n;
        const double vz_cell = vz_sum * inv_n;
        M = m;
        Mur = m * ur_cell;
        rad_ke = 0.5 * m * ur_cell * ur_cell;
        tot_ke = 0.5 * m * (vr_cell * vr_cell + vz_cell * vz_cell);
      }
    }
  }

  s_M[tid] = M;
  s_Mur[tid] = Mur;
  s_rad_ke[tid] = rad_ke;
  s_tot_ke[tid] = tot_ke;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_M[tid] += s_M[tid + stride];
      s_Mur[tid] += s_Mur[tid + stride];
      s_rad_ke[tid] += s_rad_ke[tid + stride];
      s_tot_ke[tid] += s_tot_ke[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const int base = 4 * blockIdx.x;
    block_sums[base] = s_M[0];
    block_sums[base + 1] = s_Mur[0];
    block_sums[base + 2] = s_rad_ke[0];
    block_sums[base + 3] = s_tot_ke[0];
  }
}

__device__ inline void csr_compute_cell_corner_masses(
    double* __restrict__ m_corner,
    const int c,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int corner_mass_convention,
    rz::CornerMassFallbackRecorder* fallback_recorder,
    const int fallback_stage,
    const int orientation) {
  for (int k = 0; k < mesh::kMeshTopoCellStorageSlotsMax; ++k) {
    m_corner[k] = 0.0;
  }
  const int active_nverts = csr_active_nverts_for_cell(c, cell_nverts);
  const int off = cell_node_csr_offsets[c];
  const double m_cell =
      (mass != nullptr && mass[c] > 0.0 && isfinite(mass[c])) ? mass[c] : 0.0;
  double r[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  double z[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  for (int k = 0; k < active_nverts; ++k) {
    const int n = cell_node_csr_indices[off + k];
    r[k] = x_r[n];
    z[k] = x_z[n];
  }
  rz::CornerMassFallbackProbe probe{};
  if (active_nverts == 3) {
    if (corner_mass_convention ==
        rz::kCornerMassConventionKinematicBasisRzV1) {
      rz::compute_tri_corner_masses_kinematic_rz_v1(
          m_cell, r[0], z[0], r[1], z[1], r[2], z[2], m_corner, &probe);
    } else {
      rz::compute_triangle_corner_masses_exact(
          m_cell, r[0], z[0], r[1], z[1], r[2], z[2], m_corner);
    }
  } else if (active_nverts == 4) {
    if (corner_mass_convention ==
        rz::kCornerMassConventionKinematicBasisRzV1) {
      rz::compute_quad_corner_masses_kinematic_rz_v1(
          m_cell,
          r[0],
          z[0],
          r[1],
          z[1],
          r[2],
          z[2],
          r[3],
          z[3],
          m_corner,
          &probe);
    } else {
      rz::compute_quad_corner_masses_bbsw(
          m_cell, r[0], r[1], r[2], r[3], m_corner, &probe);
    }
  } else if (active_nverts == 5) {
    csr_compute_pentagon_qk_corner_masses(m_cell, r, z, m_corner);
  } else if (active_nverts >= 6 &&
             active_nverts <= mesh::kMeshTopoCellStorageSlotsMax) {
    double w[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    mesh::moments::star_p1_vertex_r_moments(r, z, active_nverts, w);
    double W = 0.0;
    bool finite_weights = true;
    for (int k = 0; k < active_nverts; ++k) {
      W += w[k];
      finite_weights = finite_weights && rz::finite_double(w[k]);
    }
    if (!(W > 0.0) || !finite_weights) {
      probe.fired = 1;
      probe.vals[0] = w[0];
      probe.vals[1] = w[1];
      probe.vals[2] = w[2];
      probe.vals[3] = w[3];
      probe.vals[4] = W;
      const double uniform = m_cell / static_cast<double>(active_nverts);
      for (int k = 0; k < active_nverts; ++k) {
        m_corner[k] = uniform;
      }
    } else {
      for (int k = 0; k < active_nverts; ++k) {
        m_corner[k] = m_cell * (w[k] / W);
      }
    }
  }
  if (probe.fired == 1) {
    rz::record_corner_mass_fallback(fallback_recorder,
                                    probe,
                                    true,
                                    c,
                                    fallback_stage,
                                    orientation);
  }
}

__device__ inline void csr_optionb_compute_first_moment_corner_masses(
    double* __restrict__ m_corner,
    const int c,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts) {
  for (int k = 0; k < mesh::kMeshTopoCellStorageSlotsMax; ++k) {
    m_corner[k] = 0.0;
  }
  const int active_nverts = csr_active_nverts_for_cell(c, cell_nverts);
  const int off = cell_node_csr_offsets[c];
  const double m_cell =
      (mass != nullptr && mass[c] > 0.0 && isfinite(mass[c])) ? mass[c] : 0.0;
  double r[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  double z[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  for (int k = 0; k < active_nverts; ++k) {
    const int n = cell_node_csr_indices[off + k];
    r[k] = x_r[n];
    z[k] = x_z[n];
  }
  if (active_nverts == 5) {
    csr_compute_pentagon_qk_corner_masses(m_cell, r, z, m_corner);
  } else {
    tenryu::hydro::optionb::first_moment_corner_masses(
        m_cell, r, z, active_nverts, m_corner);
  }
}

__device__ inline double csr_optionb_corner_kinetic_for_cell_from_masses(
    const int c,
    const double* __restrict__ optionb_m_corner,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts) {
  const int active_nverts = csr_active_nverts_for_cell(c, cell_nverts);
  const int off = cell_node_csr_offsets[c];
  const int base = 4 * c;
  double total = 0.0;
  for (int k = 0; k < active_nverts; ++k) {
    const int n = cell_node_csr_indices[off + k];
    const double cm =
        (optionb_m_corner[base + k] > 0.0 && isfinite(optionb_m_corner[base + k]))
            ? optionb_m_corner[base + k]
            : 0.0;
    const double vr = v_r_node[n];
    const double vz = v_z_node[n];
    if (cm > 0.0 && isfinite(vr) && isfinite(vz)) {
      total += 0.5 * cm * (vr * vr + vz * vz);
    }
  }
  return isfinite(total) ? total : 0.0;
}

__device__ inline double csr_optionb_first_moment_corner_kinetic_for_cell(
    const int c,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts) {
  const int active_nverts = csr_active_nverts_for_cell(c, cell_nverts);
  const int off = cell_node_csr_offsets[c];
  double m_corner[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  csr_optionb_compute_first_moment_corner_masses(m_corner,
                                                 c,
                                                 mass,
                                                 x_r,
                                                 x_z,
                                                 cell_node_csr_offsets,
                                                 cell_node_csr_indices,
                                                 cell_nverts);
  double total = 0.0;
  for (int k = 0; k < active_nverts; ++k) {
    const int n = cell_node_csr_indices[off + k];
    const double cm =
        (m_corner[k] > 0.0 && isfinite(m_corner[k])) ? m_corner[k] : 0.0;
    const double vr = v_r_node[n];
    const double vz = v_z_node[n];
    if (cm > 0.0 && isfinite(vr) && isfinite(vz)) {
      total += 0.5 * cm * (vr * vr + vz * vz);
    }
  }
  return isfinite(total) ? total : 0.0;
}

__global__ void csr_optionb_build_first_moment_corner_mass_kernel(
    double* __restrict__ corner_mass,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int corner_stride,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const int base = corner_stride * c;
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    for (int k = 0; k < corner_stride; ++k) {
      corner_mass[base + k] = 0.0;
    }
    return;
  }
  double m_corner[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  csr_optionb_compute_first_moment_corner_masses(m_corner,
                                                 c,
                                                 mass,
                                                 x_r,
                                                 x_z,
                                                 cell_node_csr_offsets,
                                                 cell_node_csr_indices,
                                                 cell_nverts);
  for (int k = 0; k < corner_stride; ++k) {
    corner_mass[base + k] = m_corner[k];
  }
}

__global__ void csr_combine_inactive_with_active_mask_kernel(
    std::uint8_t* __restrict__ inactive_out,
    const std::uint8_t* __restrict__ inactive_in,
    const std::uint8_t* __restrict__ active_cell_mask,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const bool base_inactive =
      inactive_in != nullptr && inactive_in[c] != static_cast<std::uint8_t>(0);
  const bool outside_active_set =
      active_cell_mask != nullptr &&
      active_cell_mask[c] == static_cast<std::uint8_t>(0);
  inactive_out[c] = (base_inactive || outside_active_set) ? 1U : 0U;
}

__device__ inline bool csr_optionb_assembly_cell_active(
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const std::uint8_t* __restrict__ assembly_cell_mask,
    const int c) {
  return !csr_inactive_cell(inactive_cell_mask, c) ||
         (assembly_cell_mask != nullptr && c >= 0 &&
          assembly_cell_mask[c] != static_cast<std::uint8_t>(0));
}

__global__ void csr_optionb_seed_inactive_closure_corner_momentum_kernel(
    double* __restrict__ optionb_m_corner,
    double* __restrict__ optionb_p_r,
    double* __restrict__ optionb_p_z,
    const double* __restrict__ state_corner_mass,
    const double* __restrict__ v_r_source,
    const double* __restrict__ v_z_source,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ node_flags,
    const std::uint8_t* __restrict__ transport_inactive_cell_mask,
    const std::uint8_t* __restrict__ closure_cell_mask,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells || closure_cell_mask == nullptr ||
      closure_cell_mask[c] == static_cast<std::uint8_t>(0) ||
      !csr_inactive_cell(transport_inactive_cell_mask, c)) {
    return;
  }
  const int base = c * 4;
  const int nverts = csr_active_nverts_for_cell(c, cell_nverts);
  const int off = cell_node_csr_offsets[c];
  for (int k = 0; k < 4; ++k) {
    double m = 0.0;
    double pr = 0.0;
    double pz = 0.0;
    if (k < nverts && state_corner_mass != nullptr) {
      const int n = cell_node_csr_indices[off + k];
      if (n >= 0) {
        m = fmax(state_corner_mass[base + k], 0.0);
        m = isfinite(m) ? m : 0.0;
        double ur = v_r_source != nullptr && isfinite(v_r_source[n])
                        ? v_r_source[n]
                        : 0.0;
        double uz = v_z_source != nullptr && isfinite(v_z_source[n])
                        ? v_z_source[n]
                        : 0.0;
        const auto projector =
            node_flags == nullptr
                ? tenryu::hydro::optionb::NodeVelocityProjector::FREE
                : csr_optionb_projector_from_flags(node_flags[n]);
        tenryu::hydro::optionb::apply_node_velocity_projector(projector,
                                                              &ur,
                                                              &uz);
        pr = m * ur;
        pz = m * uz;
      }
    }
    optionb_m_corner[base + k] = m;
    optionb_p_r[base + k] = pr;
    optionb_p_z[base + k] = pz;
  }
}

// Basis-coherent transport (TENRYU_I1B_OPTIONB_COHERENT), post-transport
// V-pairing projection: repartition each cell's transported ledger corner
// masses onto the exact-subpolygon corner volumes of the POST-REMAP
// geometry, m'_a = m_c * V_a / sum(V) (the PR6-LO install product, computed
// on the component's own ledger so the installed basis and the re-recover
// divide by literally the same numbers). Cells whose exact corner volumes
// are degenerate (transiently tangled) keep their transported partition,
// clamped non-negative and renormalized to the ledger cell sum, so a single
// bad cell no longer abandons the whole install. Corner momenta are NOT
// rewritten (downstream consumers read corner masses + node fields only;
// corner momenta after this point are trace-only).
__global__ void csr_optionb_coherent_vpaired_project_kernel(
    double* __restrict__ corner_mass,
    const double* __restrict__ cell_mass,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    return;
  }
  const int base = 4 * c;
  const int nverts = csr_active_nverts_for_cell(c, cell_nverts);
  const double m_c =
      (cell_mass != nullptr && isfinite(cell_mass[c]) && cell_mass[c] > 0.0)
          ? cell_mass[c]
          : 0.0;
  if (!(m_c > 0.0)) {
    for (int k = 0; k < 4; ++k) {
      corner_mass[base + k] = 0.0;
    }
    return;
  }
  const int off = cell_node_csr_offsets[c];
  double r[4] = {0.0, 0.0, 0.0, 0.0};
  double z[4] = {0.0, 0.0, 0.0, 0.0};
  for (int k = 0; k < nverts; ++k) {
    const int n = cell_node_csr_indices[off + k];
    r[k] = x_r_new[n];
    z[k] = x_z_new[n];
  }
  double v_corner[4] = {0.0, 0.0, 0.0, 0.0};
  if (nverts == 3) {
    // Triangle quadrants — identical construction to the host V-paired
    // install (corner_mass_remap_audit.cu compute_vpaired_corner_mass) and
    // the subzonal-pressure consumer: vertex / edge midpoints / centroid,
    // |exact RZ polygon volume|.
    const double rc = (r[0] + r[1] + r[2]) / 3.0;
    const double zc = (z[0] + z[1] + z[2]) / 3.0;
    for (int k = 0; k < 3; ++k) {
      const int kp = (k + 1) % 3;
      const int km = (k + 2) % 3;
      const double r_sub[4] = {r[k], 0.5 * (r[k] + r[kp]), rc,
                               0.5 * (r[km] + r[k])};
      const double z_sub[4] = {z[k], 0.5 * (z[k] + z[kp]), zc,
                               0.5 * (z[km] + z[k])};
      v_corner[k] = fabs(rz::rz_polygon_volume_exact(r_sub, z_sub, 4));
    }
  } else {
    rz::compute_quad_corner_volumes_exact_subpolygon(
        r[0], z[0], r[1], z[1], r[2], z[2], r[3], z[3], v_corner);
  }
  double v_sum = 0.0;
  bool v_ok = true;
  for (int k = 0; k < nverts; ++k) {
    if (!(v_corner[k] > 0.0) || !isfinite(v_corner[k])) {
      v_ok = false;
      break;
    }
    v_sum += v_corner[k];
  }
  if (v_ok && v_sum > 0.0 && isfinite(v_sum)) {
    for (int k = 0; k < nverts; ++k) {
      corner_mass[base + k] = m_c * v_corner[k] / v_sum;
    }
    for (int k = nverts; k < 4; ++k) {
      corner_mass[base + k] = 0.0;
    }
    return;
  }
  // Degenerate geometry: keep the transported partition, clamped and
  // renormalized to the ledger cell sum (preserves the install contract
  // sum(m'_a) == m_c without abandoning the whole install).
  double s = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const double m_k = corner_mass[base + k];
    s += (isfinite(m_k) && m_k > 0.0) ? m_k : 0.0;
  }
  if (s > 0.0 && isfinite(s)) {
    const double scale = m_c / s;
    for (int k = 0; k < nverts; ++k) {
      const double m_k = corner_mass[base + k];
      corner_mass[base + k] =
          (isfinite(m_k) && m_k > 0.0) ? m_k * scale : 0.0;
    }
  } else {
    const double m_uniform = m_c / static_cast<double>(nverts);
    for (int k = 0; k < nverts; ++k) {
      corner_mass[base + k] = m_uniform;
    }
  }
  for (int k = nverts; k < 4; ++k) {
    corner_mass[base + k] = 0.0;
  }
}

// Basis-coherent momentum-conserving velocity re-recover: after the
// V-pairing projection changed the nodal masses M -> M', divide the
// UNCHANGED post-scatter nodal momentum sums by the new masses,
// v' = P_n / M'_n, so the velocity handed to the dynamics conserves
// momentum in exactly the basis the dynamics will use (the installed
// corner masses). Mirrors the scatter kernel's conventions: node
// projector applied, node_p rewritten as M'*v' afterwards.
__global__ void csr_optionb_coherent_rerecover_kernel(
    double* __restrict__ node_mass,
    double* __restrict__ node_p_r,
    double* __restrict__ node_p_z,
    double* __restrict__ v_r_out,
    double* __restrict__ v_z_out,
    const double* __restrict__ corner_mass_projected,
    const int* __restrict__ reverse_csr_node_offsets,
    const int* __restrict__ reverse_csr_node_cells,
    const int* __restrict__ reverse_csr_node_corners,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ node_flags,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const std::uint8_t* __restrict__ assembly_cell_mask,
    const std::uint8_t* __restrict__ active_node_velocity_mask,
    const double* __restrict__ v_r_source,
    const double* __restrict__ v_z_source,
    // true: momentum-conserving re-recover v' = P/M' (uniform flow perturbed
    // by M/M' every install — empirically pathological in the converging gas
    // core, full_r1 2026-06-12). false: velocity-preserving projection — v
    // kept, node bookkeeping (mass, p = M'v) made consistent with the
    // projected basis; the projection repartition then appears as a small
    // recorded momentum defect instead of a velocity ripple.
    const bool momentum_rerecover,
    // Projection impulse ledger (rebound-scope verdict Q4, the claim's
    // weakest point): accumulates deltaP_n = (M'_n − M^tr_n) v_n per node
    // into [net_r, net_z, L1, E_dP, sum_abs_P_tr] (5 doubles, atomicAdd).
    // node_mass[n] still holds the TRANSPORTED mass at kernel entry.
    double* __restrict__ impulse_ledger,
    const double near_massless_velocity_mass_floor,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  const int off = reverse_csr_node_offsets[n];
  const int end = reverse_csr_node_offsets[n + 1];
  double m_sum = 0.0;
  for (int p = off; p < end; ++p) {
    const int c = reverse_csr_node_cells[p];
    if (!csr_optionb_assembly_cell_active(inactive_cell_mask,
                                          assembly_cell_mask,
                                          c)) {
      continue;
    }
    const int corner = reverse_csr_node_corners[p];
    const int nverts = csr_active_nverts_for_cell(c, cell_nverts);
    if (corner < 0 || corner >= nverts) {
      continue;
    }
    m_sum += corner_mass_projected[c * 4 + corner];
  }
  if (active_node_velocity_mask != nullptr &&
      active_node_velocity_mask[n] == static_cast<std::uint8_t>(0)) {
    const double ur0 =
        v_r_source != nullptr && isfinite(v_r_source[n]) ? v_r_source[n]
                                                         : v_r_out[n];
    const double uz0 =
        v_z_source != nullptr && isfinite(v_z_source[n]) ? v_z_source[n]
                                                         : v_z_out[n];
    const double m = (m_sum > 0.0 && isfinite(m_sum)) ? m_sum : 0.0;
    node_mass[n] = m;
    node_p_r[n] = m * ur0;
    node_p_z[n] = m * uz0;
    v_r_out[n] = ur0;
    v_z_out[n] = uz0;
    return;
  }
  if (impulse_ledger != nullptr) {
    const double m_tr = node_mass[n];
    const double vr0 = v_r_out[n];
    const double vz0 = v_z_out[n];
    if (isfinite(m_tr) && isfinite(m_sum) && isfinite(vr0) &&
        isfinite(vz0)) {
      const double dm = m_sum - m_tr;
      const double dpr = dm * vr0;
      const double dpz = dm * vz0;
      const double dp_mag = sqrt(dpr * dpr + dpz * dpz);
      atomicAdd(impulse_ledger + 0, dpr);
      atomicAdd(impulse_ledger + 1, dpz);
      atomicAdd(impulse_ledger + 2, dp_mag);
      if (m_sum > 0.0) {
        atomicAdd(impulse_ledger + 3,
                  0.5 * (dpr * dpr + dpz * dpz) / m_sum);
      }
      atomicAdd(impulse_ledger + 4,
                fabs(m_tr) * sqrt(vr0 * vr0 + vz0 * vz0));
    }
  }
  double pr = node_p_r[n];
  double pz = node_p_z[n];
  double ur = 0.0;
  double uz = 0.0;
  if (!momentum_rerecover) {
    // Velocity-preserving: keep the scatter-recovered velocities, rebase
    // the nodal mass/momentum bookkeeping onto the projected corners.
    ur = v_r_out[n];
    uz = v_z_out[n];
    if (m_sum > 0.0 && isfinite(m_sum) && isfinite(ur) && isfinite(uz)) {
      node_mass[n] = m_sum;
      node_p_r[n] = m_sum * ur;
      node_p_z[n] = m_sum * uz;
    } else {
      node_mass[n] = isfinite(m_sum) ? m_sum : 0.0;
      node_p_r[n] = 0.0;
      node_p_z[n] = 0.0;
      v_r_out[n] = 0.0;
      v_z_out[n] = 0.0;
    }
    return;
  }
  if (near_massless_velocity_mass_floor > 0.0 &&
      (!isfinite(m_sum) || m_sum <= near_massless_velocity_mass_floor)) {
    ur = v_r_source != nullptr && isfinite(v_r_source[n]) ? v_r_source[n]
                                                          : v_r_out[n];
    uz = v_z_source != nullptr && isfinite(v_z_source[n]) ? v_z_source[n]
                                                          : v_z_out[n];
    const auto projector =
        node_flags == nullptr
            ? tenryu::hydro::optionb::NodeVelocityProjector::FREE
            : csr_optionb_projector_from_flags(node_flags[n]);
    tenryu::hydro::optionb::apply_node_velocity_projector(
        projector, &ur, &uz);
    m_sum = isfinite(m_sum) ? fmax(m_sum, 0.0) : 0.0;
    pr = m_sum * ur;
    pz = m_sum * uz;
  } else if (m_sum > 0.0 && isfinite(m_sum) && isfinite(pr) &&
             isfinite(pz)) {
    ur = pr / m_sum;
    uz = pz / m_sum;
    const auto projector =
        node_flags == nullptr
            ? tenryu::hydro::optionb::NodeVelocityProjector::FREE
            : csr_optionb_projector_from_flags(node_flags[n]);
    tenryu::hydro::optionb::apply_node_velocity_projector(
        projector, &ur, &uz);
    pr = m_sum * ur;
    pz = m_sum * uz;
  } else {
    m_sum = isfinite(m_sum) ? m_sum : 0.0;
    pr = 0.0;
    pz = 0.0;
  }
  node_mass[n] = m_sum;
  node_p_r[n] = pr;
  node_p_z[n] = pz;
  v_r_out[n] = ur;
  v_z_out[n] = uz;
}

__global__ void csr_copy_masked_node_velocity_kernel(
    double* __restrict__ v_r_dst,
    double* __restrict__ v_z_dst,
    const double* __restrict__ v_r_src,
    const double* __restrict__ v_z_src,
    const std::uint8_t* __restrict__ active_node_velocity_mask,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  if (active_node_velocity_mask != nullptr &&
      active_node_velocity_mask[n] == static_cast<std::uint8_t>(0)) {
    return;
  }
  v_r_dst[n] = v_r_src[n];
  v_z_dst[n] = v_z_src[n];
}

__global__ void csr_restore_unmasked_node_velocity_kernel(
    double* __restrict__ v_r_dst,
    double* __restrict__ v_z_dst,
    const double* __restrict__ v_r_src,
    const double* __restrict__ v_z_src,
    const std::uint8_t* __restrict__ active_node_velocity_mask,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes || active_node_velocity_mask == nullptr ||
      active_node_velocity_mask[n] != static_cast<std::uint8_t>(0)) {
    return;
  }
  v_r_dst[n] = v_r_src[n];
  v_z_dst[n] = v_z_src[n];
}

__global__ void csr_install_masked_corner_mass_kernel(
    double* __restrict__ dst_corner_mass,
    const double* __restrict__ src_corner_mass,
    const std::uint8_t* __restrict__ active_cell_mask,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (active_cell_mask != nullptr &&
      active_cell_mask[c] == static_cast<std::uint8_t>(0)) {
    return;
  }
  const int base = 4 * c;
  for (int k = 0; k < 4; ++k) {
    dst_corner_mass[base + k] = src_corner_mass[base + k];
  }
}

__global__ void csr_install_masked_cell_mass_from_optionb_kernel(
    double* __restrict__ mass_new,
    double* __restrict__ rho_new,
    const double* __restrict__ optionb_cell_mass,
    const double* __restrict__ vol_new,
    const std::uint8_t* __restrict__ active_cell_mask,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells || optionb_cell_mass == nullptr ||
      (active_cell_mask != nullptr &&
       active_cell_mask[c] == static_cast<std::uint8_t>(0))) {
    return;
  }
  const double m = optionb_cell_mass[c];
  if (!isfinite(m) || m < 0.0) {
    return;
  }
  mass_new[c] = m;
  if (rho_new != nullptr && vol_new != nullptr) {
    const double V = fmax(vol_new[c], kTinyVolume);
    rho_new[c] = m / V;
  }
}

__global__ void csr_project_cell_velocity_to_nodes_kernel(
    double* __restrict__ v_r_node,
    double* __restrict__ v_z_node,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const double* __restrict__ mass_cell,
    const int* __restrict__ reverse_csr_node_offsets,
    const int* __restrict__ reverse_csr_node_cells,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ frozen_node_mask,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  if (frozen_node_mask != nullptr && frozen_node_mask[n] != 0U) {
    return;
  }
  const int off = reverse_csr_node_offsets[n];
  const int end = reverse_csr_node_offsets[n + 1];
  double m_sum = 0.0;
  double pr = 0.0;
  double pz = 0.0;
  for (int p = off; p < end; ++p) {
    const int c = reverse_csr_node_cells[p];
    if (csr_inactive_cell(inactive_cell_mask, c)) {
      continue;
    }
    double m = 0.25 * fmax(mass_cell[c], 0.0);
    if (cell_nverts != nullptr) {
      const int active_nverts =
          tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
      m = fmax(mass_cell[c], 0.0) / static_cast<double>(active_nverts);
    }
    m_sum += m;
    pr += m * v_r_cell[c];
    pz += m * v_z_cell[c];
  }
  if (m_sum > 0.0 && isfinite(m_sum)) {
    v_r_node[n] = pr / m_sum;
    v_z_node[n] = pz / m_sum;
  } else {
    v_r_node[n] = 0.0;
    v_z_node[n] = 0.0;
  }
}

__global__ void csr_project_cell_velocity_to_nodes_gradient_kernel(
    double* __restrict__ v_r_node,
    double* __restrict__ v_z_node,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const double* __restrict__ v_r_grad_r,
    const double* __restrict__ v_r_grad_z,
    const double* __restrict__ v_z_grad_r,
    const double* __restrict__ v_z_grad_z,
    const double* __restrict__ centroid_r,
    const double* __restrict__ centroid_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ mass_cell,
    const int* __restrict__ reverse_csr_node_offsets,
    const int* __restrict__ reverse_csr_node_cells,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ frozen_node_mask,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  if (frozen_node_mask != nullptr && frozen_node_mask[n] != 0U) {
    return;
  }
  const int off = reverse_csr_node_offsets[n];
  const int end = reverse_csr_node_offsets[n + 1];
  bool apply_gradient = true;
  for (int p = off; p < end; ++p) {
    const int c = reverse_csr_node_cells[p];
    if (cell_nverts != nullptr &&
        tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c) != 4) {
      apply_gradient = false;
      break;
    }
  }
  double m_sum = 0.0;
  double pr = 0.0;
  double pz = 0.0;
  for (int p = off; p < end; ++p) {
    const int c = reverse_csr_node_cells[p];
    if (csr_inactive_cell(inactive_cell_mask, c)) {
      continue;
    }
    double m = 0.25 * fmax(mass_cell[c], 0.0);
    if (cell_nverts != nullptr) {
      const int active_nverts =
          tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
      m = fmax(mass_cell[c], 0.0) / static_cast<double>(active_nverts);
    }
    double vr = v_r_cell[c];
    double vz = v_z_cell[c];
    if (apply_gradient) {
      const double dr = x_r[n] - centroid_r[c];
      const double dz = x_z[n] - centroid_z[c];
      vr += v_r_grad_r[c] * dr + v_r_grad_z[c] * dz;
      vz += v_z_grad_r[c] * dr + v_z_grad_z[c] * dz;
    }
    m_sum += m;
    pr += m * vr;
    pz += m * vz;
  }
  if (m_sum > 0.0 && isfinite(m_sum)) {
    v_r_node[n] = pr / m_sum;
    v_z_node[n] = pz / m_sum;
  } else {
    v_r_node[n] = 0.0;
    v_z_node[n] = 0.0;
  }
}

__global__ void csr_project_cell_velocity_to_nodes_corner_mass_kernel(
    double* __restrict__ v_r_node,
    double* __restrict__ v_z_node,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const double* __restrict__ mass_cell,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ reverse_csr_node_offsets,
    const int* __restrict__ reverse_csr_node_cells,
    const int* __restrict__ reverse_csr_node_corners,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ frozen_node_mask,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int* __restrict__ cell_orientation_sign,
    rz::CornerMassFallbackRecorder* fallback_recorder,
    const int corner_mass_convention,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  if (frozen_node_mask != nullptr && frozen_node_mask[n] != 0U) {
    return;
  }
  const int off = reverse_csr_node_offsets[n];
  const int end = reverse_csr_node_offsets[n + 1];
  double m_sum = 0.0;
  double pr = 0.0;
  double pz = 0.0;
  for (int p = off; p < end; ++p) {
    const int c = reverse_csr_node_cells[p];
    if (csr_inactive_cell(inactive_cell_mask, c)) {
      continue;
    }
    const int corner = reverse_csr_node_corners[p];
    const int active_nverts = csr_active_nverts_for_cell(c, cell_nverts);
    if (corner < 0 || corner >= active_nverts) {
      continue;
    }
    double m_corner[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    csr_compute_cell_corner_masses(m_corner,
                                   c,
                                   mass_cell,
                                   x_r,
                                   x_z,
                                   cell_node_csr_offsets,
                                   cell_node_csr_indices,
                                   cell_nverts,
                                   corner_mass_convention,
                                   fallback_recorder,
                                   rz::kCornerMassFallbackStageCsrNodeProjection,
                                   cell_orientation_sign != nullptr
                                       ? cell_orientation_sign[c]
                                       : -2);
    const double m =
        (m_corner[corner] > 0.0 && isfinite(m_corner[corner]))
            ? m_corner[corner]
            : 0.0;
    m_sum += m;
    pr += m * v_r_cell[c];
    pz += m * v_z_cell[c];
  }
  if (m_sum > 0.0 && isfinite(m_sum)) {
    v_r_node[n] = pr / m_sum;
    v_z_node[n] = pz / m_sum;
  } else {
    v_r_node[n] = 0.0;
    v_z_node[n] = 0.0;
  }
}

__global__ void csr_project_cell_velocity_to_nodes_corner_mass_gradient_kernel(
    double* __restrict__ v_r_node,
    double* __restrict__ v_z_node,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const double* __restrict__ v_r_grad_r,
    const double* __restrict__ v_r_grad_z,
    const double* __restrict__ v_z_grad_r,
    const double* __restrict__ v_z_grad_z,
    const double* __restrict__ centroid_r,
    const double* __restrict__ centroid_z,
    const double* __restrict__ mass_cell,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ reverse_csr_node_offsets,
    const int* __restrict__ reverse_csr_node_cells,
    const int* __restrict__ reverse_csr_node_corners,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ frozen_node_mask,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int* __restrict__ cell_orientation_sign,
    rz::CornerMassFallbackRecorder* fallback_recorder,
    const int corner_mass_convention,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  if (frozen_node_mask != nullptr && frozen_node_mask[n] != 0U) {
    return;
  }
  const int off = reverse_csr_node_offsets[n];
  const int end = reverse_csr_node_offsets[n + 1];
  bool apply_gradient = true;
  for (int p = off; p < end; ++p) {
    const int c = reverse_csr_node_cells[p];
    if (csr_active_nverts_for_cell(c, cell_nverts) != 4) {
      apply_gradient = false;
      break;
    }
  }
  double m_sum = 0.0;
  double pr = 0.0;
  double pz = 0.0;
  for (int p = off; p < end; ++p) {
    const int c = reverse_csr_node_cells[p];
    if (csr_inactive_cell(inactive_cell_mask, c)) {
      continue;
    }
    const int corner = reverse_csr_node_corners[p];
    const int active_nverts = csr_active_nverts_for_cell(c, cell_nverts);
    if (corner < 0 || corner >= active_nverts) {
      continue;
    }
    double m_corner[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    csr_compute_cell_corner_masses(m_corner,
                                   c,
                                   mass_cell,
                                   x_r,
                                   x_z,
                                   cell_node_csr_offsets,
                                   cell_node_csr_indices,
                                   cell_nverts,
                                   corner_mass_convention,
                                   fallback_recorder,
                                   rz::kCornerMassFallbackStageCsrNodeProjection,
                                   cell_orientation_sign != nullptr
                                       ? cell_orientation_sign[c]
                                       : -2);
    const double m =
        (m_corner[corner] > 0.0 && isfinite(m_corner[corner]))
            ? m_corner[corner]
            : 0.0;
    double vr = v_r_cell[c];
    double vz = v_z_cell[c];
    if (apply_gradient) {
      const double dr = x_r[n] - centroid_r[c];
      const double dz = x_z[n] - centroid_z[c];
      vr += v_r_grad_r[c] * dr + v_r_grad_z[c] * dz;
      vz += v_z_grad_r[c] * dr + v_z_grad_z[c] * dz;
    }
    m_sum += m;
    pr += m * vr;
    pz += m * vz;
  }
  if (m_sum > 0.0 && isfinite(m_sum)) {
    v_r_node[n] = pr / m_sum;
    v_z_node[n] = pz / m_sum;
  } else {
    v_r_node[n] = 0.0;
    v_z_node[n] = 0.0;
  }
}

__device__ inline double csr_corner_kinetic_for_cell(
    const int c,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int* __restrict__ cell_orientation_sign,
    const int corner_mass_convention,
    rz::CornerMassFallbackRecorder* fallback_recorder,
    const int fallback_stage);

__global__ void csr_compute_total_energy_ke_cell_scale_kernel(
    double* __restrict__ cell_ke_scale,
    const double* __restrict__ total_energy,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const double* __restrict__ zbar,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int* __restrict__ cell_orientation_sign,
    rz::CornerMassFallbackRecorder* fallback_recorder,
    const int corner_mass_convention,
    const int n_cells,
    const double gamma,
    const double A,
    const double te_floor,
    const double ti_floor) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    cell_ke_scale[c] = 1.0;
    return;
  }
  double scale = 1.0;
  const double m = fmax(mass[c], 0.0);
  if (m > 0.0 && isfinite(m) && total_energy != nullptr) {
    const double A_safe = fmax(A, 1.0e-30);
    const double gm1 = fmax(gamma - 1.0, 1.0e-30);
    const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : 1.0;
    const double cv_i =
        tenryu::core::constants::eV_to_erg /
        (A_safe * tenryu::core::constants::proton_mass * gm1);
    const double cv_e =
        z * tenryu::core::constants::eV_to_erg /
        (A_safe * tenryu::core::constants::proton_mass * gm1);
    const double e_floor =
        cv_e * fmax(te_floor, 0.0) + cv_i * fmax(ti_floor, 0.0);
    const double K = csr_corner_kinetic_for_cell(c,
                                                 mass,
                                                 x_r,
                                                 x_z,
                                                 v_r_node,
                                                 v_z_node,
                                                 cell_node_csr_offsets,
                                                 cell_node_csr_indices,
                                                 cell_nverts,
                                                 cell_orientation_sign,
                                                 corner_mass_convention,
                                                 fallback_recorder,
                                                 rz::kCornerMassFallbackStageCsrTotalEnergyScale);
    const double K_max = total_energy[c] - m * e_floor;
    if (K > 0.0 && isfinite(K)) {
      if (!(K_max > 0.0) || !isfinite(K_max)) {
        scale = 0.0;
      } else if (K > K_max) {
        scale = sqrt(fmax(K_max / K, 0.0)) * (1.0 - 1.0e-12);
      }
    }
  }
  cell_ke_scale[c] = fmin(1.0, fmax(0.0, isfinite(scale) ? scale : 0.0));
}

__global__ void csr_compute_total_energy_ke_cell_scale_physical_ke_kernel(
    double* __restrict__ cell_ke_scale,
    const double* __restrict__ total_energy,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const double* __restrict__ zbar,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int* __restrict__ cell_orientation_sign,
    rz::CornerMassFallbackRecorder* fallback_recorder,
    const int corner_mass_convention,
    const int n_cells,
    const int nz,
    const double gamma,
    const double A,
    const double te_floor,
    const double ti_floor) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    cell_ke_scale[c] = 1.0;
    return;
  }
  double scale = 1.0;
  const double m = fmax(mass[c], 0.0);
  if (m > 0.0 && isfinite(m) && total_energy != nullptr) {
    const double A_safe = fmax(A, 1.0e-30);
    const double gm1 = fmax(gamma - 1.0, 1.0e-30);
    const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : 1.0;
    const double cv_i =
        tenryu::core::constants::eV_to_erg /
        (A_safe * tenryu::core::constants::proton_mass * gm1);
    const double cv_e =
        z * tenryu::core::constants::eV_to_erg /
        (A_safe * tenryu::core::constants::proton_mass * gm1);
    const double e_floor =
        cv_e * fmax(te_floor, 0.0) + cv_i * fmax(ti_floor, 0.0);
    const double K = rz_physical_corner_kinetic_for_cell(
        mass, x_r, x_z, v_r_node, v_z_node, cell_nverts, c, nz,
        corner_mass_convention, fallback_recorder,
        rz::kCornerMassFallbackStageCsrPhysicalKeScale,
        cell_orientation_sign != nullptr ? cell_orientation_sign[c] : -2);
    const double K_max = total_energy[c] - m * e_floor;
    if (K > 0.0 && isfinite(K)) {
      if (!(K_max > 0.0) || !isfinite(K_max)) {
        scale = 0.0;
      } else if (K > K_max) {
        scale = sqrt(fmax(K_max / K, 0.0)) * (1.0 - 1.0e-12);
      }
    }
  }
  cell_ke_scale[c] = fmin(1.0, fmax(0.0, isfinite(scale) ? scale : 0.0));
}

__global__ void csr_compute_total_energy_ke_cell_scale_optionb_kernel(
    double* __restrict__ cell_ke_scale,
    const double* __restrict__ total_energy,
    const double* __restrict__ mass,
    const double* __restrict__ optionb_m_corner,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const double* __restrict__ zbar,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells,
    const double gamma,
    const double A,
    const double te_floor,
    const double ti_floor) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    cell_ke_scale[c] = 1.0;
    return;
  }
  double scale = 1.0;
  const double m = fmax(mass[c], 0.0);
  if (m > 0.0 && isfinite(m) && total_energy != nullptr &&
      optionb_m_corner != nullptr) {
    const double A_safe = fmax(A, 1.0e-30);
    const double gm1 = fmax(gamma - 1.0, 1.0e-30);
    const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : 1.0;
    const double cv_i =
        tenryu::core::constants::eV_to_erg /
        (A_safe * tenryu::core::constants::proton_mass * gm1);
    const double cv_e =
        z * tenryu::core::constants::eV_to_erg /
        (A_safe * tenryu::core::constants::proton_mass * gm1);
    const double e_floor =
        cv_e * fmax(te_floor, 0.0) + cv_i * fmax(ti_floor, 0.0);
    const double K = csr_optionb_corner_kinetic_for_cell_from_masses(
        c,
        optionb_m_corner,
        v_r_node,
        v_z_node,
        cell_node_csr_offsets,
        cell_node_csr_indices,
        cell_nverts);
    const double K_max = total_energy[c] - m * e_floor;
    if (K > 0.0 && isfinite(K)) {
      if (!(K_max > 0.0) || !isfinite(K_max)) {
        scale = 0.0;
      } else if (K > K_max) {
        scale = sqrt(fmax(K_max / K, 0.0)) * (1.0 - 1.0e-12);
      }
    }
  }
  cell_ke_scale[c] = fmin(1.0, fmax(0.0, isfinite(scale) ? scale : 0.0));
}

__device__ inline void csr_update_physical_node_ke_scale(
    double& scale,
    const int c,
    const int corner,
    const double* __restrict__ cell_ke_scale,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells) {
  if (c < 0 || c >= n_cells || csr_inactive_cell(inactive_cell_mask, c)) {
    return;
  }
  const int active_nverts = csr_active_nverts_for_cell(c, cell_nverts);
  if (corner < 0 || corner >= active_nverts) {
    return;
  }
  double s = cell_ke_scale[c];
  if (!isfinite(s)) {
    s = 0.0;
  }
  scale = fmin(scale, fmin(1.0, fmax(0.0, s)));
}

__global__ void csr_compute_total_energy_ke_node_scale_physical_ke_kernel(
    double* __restrict__ node_ke_scale,
    const double* __restrict__ cell_ke_scale,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells,
    const int nz,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  double scale = 1.0;
  if (nz > 0) {
    const int stride = nz + 1;
    const int i = n / stride;
    const int j = n - i * stride;
    if (j >= 0 && j <= nz) {
      if (j < nz) {
        csr_update_physical_node_ke_scale(scale,
                                          i * nz + j,
                                          0,
                                          cell_ke_scale,
                                          cell_nverts,
                                          inactive_cell_mask,
                                          n_cells);
      }
      if (i > 0 && j < nz) {
        csr_update_physical_node_ke_scale(scale,
                                          (i - 1) * nz + j,
                                          1,
                                          cell_ke_scale,
                                          cell_nverts,
                                          inactive_cell_mask,
                                          n_cells);
      }
      if (i > 0 && j > 0) {
        csr_update_physical_node_ke_scale(scale,
                                          (i - 1) * nz + (j - 1),
                                          2,
                                          cell_ke_scale,
                                          cell_nverts,
                                          inactive_cell_mask,
                                          n_cells);
      }
      if (j > 0) {
        csr_update_physical_node_ke_scale(scale,
                                          i * nz + (j - 1),
                                          3,
                                          cell_ke_scale,
                                          cell_nverts,
                                          inactive_cell_mask,
                                          n_cells);
      }
    }
  }
  node_ke_scale[n] = scale;
}

__global__ void csr_compute_total_energy_ke_node_scale_kernel(
    double* __restrict__ node_ke_scale,
    const double* __restrict__ cell_ke_scale,
    const int* __restrict__ reverse_csr_node_offsets,
    const int* __restrict__ reverse_csr_node_cells,
    const int* __restrict__ reverse_csr_node_corners,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  double scale = 1.0;
  const int off = reverse_csr_node_offsets[n];
  const int end = reverse_csr_node_offsets[n + 1];
  for (int p = off; p < end; ++p) {
    const int c = reverse_csr_node_cells[p];
    if (csr_inactive_cell(inactive_cell_mask, c)) {
      continue;
    }
    const int corner = reverse_csr_node_corners[p];
    const int active_nverts = csr_active_nverts_for_cell(c, cell_nverts);
    if (corner < 0 || corner >= active_nverts) {
      continue;
    }
    double s = cell_ke_scale[c];
    if (!isfinite(s)) {
      s = 0.0;
    }
    scale = fmin(scale, fmin(1.0, fmax(0.0, s)));
  }
  node_ke_scale[n] = scale;
}

__global__ void csr_apply_total_energy_ke_node_scale_kernel(
    double* __restrict__ v_r_node,
    double* __restrict__ v_z_node,
    const double* __restrict__ node_ke_scale,
    const std::uint8_t* __restrict__ active_node_velocity_mask,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  if (active_node_velocity_mask != nullptr &&
      active_node_velocity_mask[n] == static_cast<std::uint8_t>(0)) {
    return;
  }
  double scale = node_ke_scale[n];
  if (!isfinite(scale)) {
    scale = 0.0;
  }
  scale = fmin(1.0, fmax(0.0, scale));
  const double vr = v_r_node[n];
  const double vz = v_z_node[n];
  v_r_node[n] = isfinite(vr) ? (scale * vr) : 0.0;
  v_z_node[n] = isfinite(vz) ? (scale * vz) : 0.0;
}

__device__ inline double csr_corner_kinetic_for_cell(
    const int c,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int* __restrict__ cell_orientation_sign,
    const int corner_mass_convention,
    rz::CornerMassFallbackRecorder* fallback_recorder,
    const int fallback_stage) {
  const int active_nverts = csr_active_nverts_for_cell(c, cell_nverts);
  const int off = cell_node_csr_offsets[c];
  double total = 0.0;
  double m_corner[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  csr_compute_cell_corner_masses(m_corner,
                                 c,
                                 mass,
                                 x_r,
                                 x_z,
                                 cell_node_csr_offsets,
                                 cell_node_csr_indices,
                                 cell_nverts,
                                 corner_mass_convention,
                                 fallback_recorder,
                                 fallback_stage,
                                 cell_orientation_sign != nullptr
                                     ? cell_orientation_sign[c]
                                     : -2);
  for (int k = 0; k < active_nverts; ++k) {
    const int n = cell_node_csr_indices[off + k];
    const double cm =
        (m_corner[k] > 0.0 && isfinite(m_corner[k])) ? m_corner[k] : 0.0;
    const double vr = v_r_node[n];
    const double vz = v_z_node[n];
    if (cm > 0.0 && isfinite(vr) && isfinite(vz)) {
      total += 0.5 * cm * (vr * vr + vz * vz);
    }
  }
  return total;
}

__global__ void csr_compute_corner_kinetic_density_kernel(
    double* __restrict__ ke_density_lag,
    const double* __restrict__ mass,
    const double* __restrict__ vol_lag,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int* __restrict__ cell_orientation_sign,
    rz::CornerMassFallbackRecorder* fallback_recorder,
    const int corner_mass_convention,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    ke_density_lag[c] = 0.0;
    return;
  }
  const double kinetic = csr_corner_kinetic_for_cell(c,
                                                     mass,
                                                     x_r,
                                                     x_z,
                                                     v_r_node,
                                                     v_z_node,
                                                     cell_node_csr_offsets,
                                                     cell_node_csr_indices,
                                                     cell_nverts,
                                                     cell_orientation_sign,
                                                     corner_mass_convention,
                                                     fallback_recorder,
                                                     rz::kCornerMassFallbackStageCsrKePre);
  const double vol = fmax(vol_lag[c], kTinyVolume);
  ke_density_lag[c] = fmax(kinetic / vol, 0.0);
}

__global__ void csr_compute_corner_kinetic_total_kernel(
    double* __restrict__ kinetic_total,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int* __restrict__ cell_orientation_sign,
    rz::CornerMassFallbackRecorder* fallback_recorder,
    const int corner_mass_convention,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    kinetic_total[c] = 0.0;
    return;
  }
  kinetic_total[c] = csr_corner_kinetic_for_cell(c,
                                                 mass,
                                                 x_r,
                                                 x_z,
                                                 v_r_node,
                                                 v_z_node,
                                                 cell_node_csr_offsets,
                                                 cell_node_csr_indices,
                                                 cell_nverts,
                                                 cell_orientation_sign,
                                                 corner_mass_convention,
                                                 fallback_recorder,
                                                 rz::kCornerMassFallbackStageCsrKePost);
}

__global__ void csr_build_total_energy_remap_state_kernel(
    double* __restrict__ e_tot_lag,
    double* __restrict__ ye_int_lag,
    const double* __restrict__ mass,
    const double* __restrict__ rho,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ vol_lag,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int* __restrict__ cell_orientation_sign,
    rz::CornerMassFallbackRecorder* fallback_recorder,
    const int corner_mass_convention,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  double m = fmax(mass[c], 0.0);
  if (!isfinite(m)) {
    m = fmax(rho[c], 0.0) * fmax(vol_lag[c], 0.0);
  }
  const double e_e = fmax(ee[c], 0.0);
  const double e_i = fmax(ei[c], 0.0);
  const double e_int = e_e + e_i;
  const double K = csr_corner_kinetic_for_cell(c,
                                               mass,
                                               x_r,
                                               x_z,
                                               v_r_node,
                                               v_z_node,
                                               cell_node_csr_offsets,
                                               cell_node_csr_indices,
                                               cell_nverts,
                                               cell_orientation_sign,
                                               corner_mass_convention,
                                               fallback_recorder,
                                               rz::kCornerMassFallbackStageCsrTotalEnergyBuild);
  e_tot_lag[c] = (m > 0.0 && isfinite(K)) ? (e_int + K / m) : e_int;
  ye_int_lag[c] = (e_int > 0.0 && isfinite(e_int))
                      ? clamp01_device(e_e / e_int)
                      : 0.5;
}

__global__ void csr_build_total_energy_remap_state_physical_ke_kernel(
    double* __restrict__ e_tot_lag,
    double* __restrict__ ye_int_lag,
    const double* __restrict__ mass,
    const double* __restrict__ rho,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ vol_lag,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const std::uint8_t* __restrict__ cell_nverts,
    const int* __restrict__ cell_orientation_sign,
    rz::CornerMassFallbackRecorder* fallback_recorder,
    const int corner_mass_convention,
    const int n_cells,
    const int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  double m = fmax(mass[c], 0.0);
  if (!isfinite(m)) {
    m = fmax(rho[c], 0.0) * fmax(vol_lag[c], 0.0);
  }
  const double e_e = fmax(ee[c], 0.0);
  const double e_i = fmax(ei[c], 0.0);
  const double e_int = e_e + e_i;
  const double K = rz_physical_corner_kinetic_for_cell(
      mass, x_r, x_z, v_r_node, v_z_node, cell_nverts, c, nz,
      corner_mass_convention, fallback_recorder,
      rz::kCornerMassFallbackStageCsrPhysicalKeBuild,
      cell_orientation_sign != nullptr ? cell_orientation_sign[c] : -2);
  e_tot_lag[c] = (m > 0.0 && isfinite(K)) ? (e_int + K / m) : e_int;
  ye_int_lag[c] = (e_int > 0.0 && isfinite(e_int))
	                      ? clamp01_device(e_e / e_int)
	                      : 0.5;
}

__global__ void csr_build_total_energy_remap_state_optionb_kernel(
    double* __restrict__ e_tot_lag,
    double* __restrict__ ye_int_lag,
    const double* __restrict__ mass,
    const double* __restrict__ rho,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ vol_lag,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const std::uint8_t* __restrict__ energy_closure_cell_mask,
    const double* __restrict__ frozen_corner_mass,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c) &&
      (energy_closure_cell_mask == nullptr ||
       energy_closure_cell_mask[c] == static_cast<std::uint8_t>(0))) {
    e_tot_lag[c] = 0.0;
    ye_int_lag[c] = 0.5;
    return;
  }
  double m = fmax(mass[c], 0.0);
  if (!isfinite(m)) {
    m = fmax(rho[c], 0.0) * fmax(vol_lag[c], 0.0);
  }
  const double e_e = fmax(ee[c], 0.0);
  const double e_i = fmax(ei[c], 0.0);
  const double e_int = e_e + e_i;
  const double K =
      frozen_corner_mass != nullptr
          ? csr_optionb_corner_kinetic_for_cell_from_masses(
                c,
                frozen_corner_mass,
                v_r_node,
                v_z_node,
                cell_node_csr_offsets,
                cell_node_csr_indices,
                cell_nverts)
          : csr_optionb_first_moment_corner_kinetic_for_cell(
                c,
                mass,
                x_r,
                x_z,
                v_r_node,
                v_z_node,
                cell_node_csr_offsets,
                cell_node_csr_indices,
                cell_nverts);
  e_tot_lag[c] = (m > 0.0 && isfinite(K)) ? (e_int + K / m) : e_int;
  ye_int_lag[c] = (e_int > 0.0 && isfinite(e_int))
                      ? clamp01_device(e_e / e_int)
                      : 0.5;
}

__global__ void csr_seed_pseudo_core_total_energy_remap_state_kernel(
    double* __restrict__ e_tot_lag,
    double* __restrict__ ye_int_lag,
    const std::uint8_t* __restrict__ member_mask,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells,
    const double ee_c,
    const double ei_c) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells || member_mask == nullptr || member_mask[c] == 0U) {
    return;
  }
  const double m = fmax(mass[c], 0.0);
  const double e_e = fmax(ee_c, 0.0);
  const double e_i = fmax(ei_c, 0.0);
  const double e_int = e_e + e_i;
  const double K = csr_corner_kinetic_for_cell(c,
                                               mass,
                                               x_r,
                                               x_z,
                                               v_r_node,
                                               v_z_node,
                                               cell_node_csr_offsets,
                                               cell_node_csr_indices,
                                               cell_nverts,
                                               nullptr,
                                               rz::kCornerMassConventionBbswRadialV0,
                                               nullptr,
                                               0);
  e_tot_lag[c] = (m > 0.0 && isfinite(K)) ? (e_int + K / m) : e_int;
  ye_int_lag[c] = (e_int > 0.0 && isfinite(e_int))
                      ? clamp01_device(e_e / e_int)
                      : 0.5;
}

__global__ void csr_initialize_hydro_total_extents_kernel(
    double* __restrict__ mass_new,
    double* __restrict__ mom_r_new,
    double* __restrict__ mom_z_new,
    double* __restrict__ total_energy_new,
    double* __restrict__ ye_mass_new,
    const double* __restrict__ mass_lag,
    const double* __restrict__ rho_lag,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const double* __restrict__ e_tot_lag,
    const double* __restrict__ ye_int_lag,
    const double* __restrict__ vol_lag,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  double m = fmax(mass_lag[c], 0.0);
  if (!isfinite(m)) {
    m = fmax(rho_lag[c], 0.0) * fmax(vol_lag[c], 0.0);
  }
  mass_new[c] = m;
  mom_r_new[c] = m * v_r_cell[c];
  mom_z_new[c] = m * v_z_cell[c];
  total_energy_new[c] = m * fmax(e_tot_lag[c], 0.0);
  ye_mass_new[c] = m * clamp01_device(ye_int_lag[c]);
}

__global__ void csr_finish_total_hydro_remap_kernel(
    double* __restrict__ rho_new,
    double* __restrict__ v_r_cell_new,
    double* __restrict__ v_z_cell_new,
    double* __restrict__ mass_new,
    const double* __restrict__ mom_r_new,
    const double* __restrict__ mom_z_new,
    double* __restrict__ total_energy_new,
    double* __restrict__ ye_int_new,
    const double* __restrict__ vol_new,
    const double* __restrict__ rho_lag,
    const double* __restrict__ zbar,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells,
    const double rho_floor,
    const double te_floor,
    const double ti_floor,
    const double gamma,
    const double A,
    double* __restrict__ mass_floor_delta,
    double* __restrict__ E_redistribution_unresolved,
    int* __restrict__ active_floor_hit_count) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    const double m =
        (mass_new[c] > 0.0 && isfinite(mass_new[c])) ? mass_new[c] : 0.0;
    mass_new[c] = m;
    rho_new[c] = rho_lag[c];
    v_r_cell_new[c] =
        (m > 0.0 && isfinite(mom_r_new[c])) ? (mom_r_new[c] / m) : 0.0;
    v_z_cell_new[c] =
        (m > 0.0 && isfinite(mom_z_new[c])) ? (mom_z_new[c] / m) : 0.0;
    ye_int_new[c] = (m > 0.0 && isfinite(ye_int_new[c]))
                        ? clamp01_device(ye_int_new[c] / m)
                        : 0.5;
    return;
  }
  const double V = fmax(vol_new[c], kTinyVolume);
  const double m_raw = mass_new[c];
  double m = m_raw;
  const double mass_floor = rho_floor * V;

  const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : 1.0;
  const double A_safe = fmax(A, 1.0e-30);
  const double gm1 = fmax(gamma - 1.0, 1.0e-30);
  const double cv_i =
      tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  const double cv_e =
      z * tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  const double e_e_floor = cv_e * fmax(te_floor, 0.0);
  const double e_i_floor = cv_i * fmax(ti_floor, 0.0);
  const double e_floor_sum = e_e_floor + e_i_floor;
  const double ye_floor =
      (e_floor_sum > 0.0 && isfinite(e_floor_sum))
          ? clamp01_device(e_e_floor / e_floor_sum)
          : 0.5;

  if (m_raw > mass_floor && isfinite(m_raw)) {
    m = m_raw;
    rho_new[c] = m / V;
    v_r_cell_new[c] = (m > 0.0 && isfinite(mom_r_new[c])) ? (mom_r_new[c] / m) : 0.0;
    v_z_cell_new[c] = (m > 0.0 && isfinite(mom_z_new[c])) ? (mom_z_new[c] / m) : 0.0;
    total_energy_new[c] =
        isfinite(total_energy_new[c]) ? total_energy_new[c] : 0.0;
    ye_int_new[c] = (m > 0.0 && isfinite(ye_int_new[c]))
                        ? clamp01_device(ye_int_new[c] / m)
                        : 0.5;
  } else {
    m = mass_floor;
    rho_new[c] = m / V;
    v_r_cell_new[c] = 0.0;
    v_z_cell_new[c] = 0.0;
    const double total_raw =
        isfinite(total_energy_new[c]) ? total_energy_new[c] : 0.0;
    const double total_floor = m * e_floor_sum;
    total_energy_new[c] = total_floor;
    ye_int_new[c] = ye_floor;
    if (mass_floor_delta != nullptr) {
      atomicAdd(mass_floor_delta, fmax(mass_floor - m_raw, 0.0));
    }
    if (E_redistribution_unresolved != nullptr) {
      atomicAdd(E_redistribution_unresolved, total_floor - total_raw);
    }
    if (active_floor_hit_count != nullptr) {
      atomicAdd(active_floor_hit_count, 1);
    }
  }
  mass_new[c] = m;
}

__global__ void csr_recover_internal_from_total_energy_remap_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ total_energy,
    const double* __restrict__ ye_int,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const double* __restrict__ zbar,
    double* __restrict__ energy_floor_delta,
    int* __restrict__ floor_hit_count,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int* __restrict__ cell_orientation_sign,
    rz::CornerMassFallbackRecorder* fallback_recorder,
    const int corner_mass_convention,
    const int n_cells,
    const double gamma,
    const double A,
    const double te_floor,
    const double ti_floor) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    return;
  }
  const double m = fmax(mass[c], 0.0);
  const double A_safe = fmax(A, 1.0e-30);
  const double gm1 = fmax(gamma - 1.0, 1.0e-30);
  const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : 1.0;
  const double cv_i =
      tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  const double cv_e =
      z * tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  const double e_i_floor = cv_i * fmax(ti_floor, 0.0);
  const double e_e_floor = cv_e * fmax(te_floor, 0.0);
  if (!(m > 0.0) || !isfinite(m)) {
    ee[c] = e_e_floor;
    ei[c] = e_i_floor;
    return;
  }

  const double K = csr_corner_kinetic_for_cell(c,
                                               mass,
                                               x_r,
                                               x_z,
                                               v_r_node,
                                               v_z_node,
                                               cell_node_csr_offsets,
                                               cell_node_csr_indices,
                                               cell_nverts,
                                               cell_orientation_sign,
                                               corner_mass_convention,
                                               fallback_recorder,
                                               rz::kCornerMassFallbackStageCsrTotalEnergyRecover);
  const double e_tot = total_energy[c] / m;
  const double e_int_raw = e_tot - K / m;
  const double ye = clamp01_device(ye_int[c]);
  double e_e_raw = ye * e_int_raw;
  double e_i_raw = (1.0 - ye) * e_int_raw;
  double delta = 0.0;
  if (!isfinite(e_e_raw)) {
    delta += e_e_floor * m;
    e_e_raw = e_e_floor;
  }
  if (!isfinite(e_i_raw)) {
    delta += e_i_floor * m;
    e_i_raw = e_i_floor;
  }
  const double e_e_new = fmax(e_e_raw, e_e_floor);
  const double e_i_new = fmax(e_i_raw, e_i_floor);
  if (e_e_raw < e_e_floor) {
    delta += (e_e_floor - e_e_raw) * m;
  }
  if (e_i_raw < e_i_floor) {
    delta += (e_i_floor - e_i_raw) * m;
  }
  if (energy_floor_delta != nullptr && delta > 0.0 && isfinite(delta)) {
    atomicAdd(energy_floor_delta, delta);
  }
  if (floor_hit_count != nullptr && delta > 0.0 && isfinite(delta)) {
    atomicAdd(floor_hit_count, 1);
  }
  ee[c] = e_e_new;
  ei[c] = e_i_new;
}

__global__ void csr_recover_internal_from_total_energy_remap_physical_ke_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ total_energy,
    const double* __restrict__ ye_int,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const double* __restrict__ zbar,
    double* __restrict__ energy_floor_delta,
    int* __restrict__ floor_hit_count,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int* __restrict__ cell_orientation_sign,
    rz::CornerMassFallbackRecorder* fallback_recorder,
    const int corner_mass_convention,
    const int n_cells,
    const int nz,
    const double gamma,
    const double A,
    const double te_floor,
    const double ti_floor) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    return;
  }
  const double m = fmax(mass[c], 0.0);
  const double A_safe = fmax(A, 1.0e-30);
  const double gm1 = fmax(gamma - 1.0, 1.0e-30);
  const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : 1.0;
  const double cv_i =
      tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  const double cv_e =
      z * tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  const double e_i_floor = cv_i * fmax(ti_floor, 0.0);
  const double e_e_floor = cv_e * fmax(te_floor, 0.0);
  if (!(m > 0.0) || !isfinite(m)) {
    ee[c] = e_e_floor;
    ei[c] = e_i_floor;
    return;
  }

  const double K = rz_physical_corner_kinetic_for_cell(
      mass, x_r, x_z, v_r_node, v_z_node, cell_nverts, c, nz,
      corner_mass_convention, fallback_recorder,
      rz::kCornerMassFallbackStageCsrPhysicalKeRecover,
      cell_orientation_sign != nullptr ? cell_orientation_sign[c] : -2);
  const double e_tot = total_energy[c] / m;
  const double e_int_raw = e_tot - K / m;
  const double ye = clamp01_device(ye_int[c]);
  double e_e_raw = ye * e_int_raw;
  double e_i_raw = (1.0 - ye) * e_int_raw;
  double delta = 0.0;
  if (!isfinite(e_e_raw)) {
    delta += e_e_floor * m;
    e_e_raw = e_e_floor;
  }
  if (!isfinite(e_i_raw)) {
    delta += e_i_floor * m;
    e_i_raw = e_i_floor;
  }
  const double e_e_new = fmax(e_e_raw, e_e_floor);
  const double e_i_new = fmax(e_i_raw, e_i_floor);
  if (e_e_raw < e_e_floor) {
    delta += (e_e_floor - e_e_raw) * m;
  }
  if (e_i_raw < e_i_floor) {
    delta += (e_i_floor - e_i_raw) * m;
  }
  if (energy_floor_delta != nullptr && delta > 0.0 && isfinite(delta)) {
    atomicAdd(energy_floor_delta, delta);
  }
  if (floor_hit_count != nullptr && delta > 0.0 && isfinite(delta)) {
    atomicAdd(floor_hit_count, 1);
  }
  ee[c] = e_e_new;
  ei[c] = e_i_new;
}

__global__ void csr_recover_internal_from_total_energy_remap_optionb_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ total_energy,
    const double* __restrict__ ye_int,
    const double* __restrict__ mass,
    const double* __restrict__ optionb_m_corner,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const double* __restrict__ zbar,
    double* __restrict__ energy_floor_delta,
    int* __restrict__ floor_hit_count,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells,
    const double gamma,
    const double A,
    const double te_floor,
    const double ti_floor) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (csr_inactive_cell(inactive_cell_mask, c)) {
    return;
  }
  const double m = fmax(mass[c], 0.0);
  const double A_safe = fmax(A, 1.0e-30);
  const double gm1 = fmax(gamma - 1.0, 1.0e-30);
  const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : 1.0;
  const double cv_i =
      tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  const double cv_e =
      z * tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  const double e_i_floor = cv_i * fmax(ti_floor, 0.0);
  const double e_e_floor = cv_e * fmax(te_floor, 0.0);
  if (!(m > 0.0) || !isfinite(m) || optionb_m_corner == nullptr) {
    ee[c] = e_e_floor;
    ei[c] = e_i_floor;
    return;
  }

  const double K = csr_optionb_corner_kinetic_for_cell_from_masses(
      c,
      optionb_m_corner,
      v_r_node,
      v_z_node,
      cell_node_csr_offsets,
      cell_node_csr_indices,
      cell_nverts);
  const double e_tot = total_energy[c] / m;
  const double e_int_raw = e_tot - K / m;
  const double ye = clamp01_device(ye_int[c]);
  double e_e_raw = ye * e_int_raw;
  double e_i_raw = (1.0 - ye) * e_int_raw;
  double delta = 0.0;
  if (!isfinite(e_e_raw)) {
    delta += e_e_floor * m;
    e_e_raw = e_e_floor;
  }
  if (!isfinite(e_i_raw)) {
    delta += e_i_floor * m;
    e_i_raw = e_i_floor;
  }
  const double e_e_new = fmax(e_e_raw, e_e_floor);
  const double e_i_new = fmax(e_i_raw, e_i_floor);
  if (e_e_raw < e_e_floor) {
    delta += (e_e_floor - e_e_raw) * m;
  }
  if (e_i_raw < e_i_floor) {
    delta += (e_i_floor - e_i_raw) * m;
  }
  if (energy_floor_delta != nullptr && delta > 0.0 && isfinite(delta)) {
    atomicAdd(energy_floor_delta, delta);
  }
  if (floor_hit_count != nullptr && delta > 0.0 && isfinite(delta)) {
    atomicAdd(floor_hit_count, 1);
  }
  ee[c] = e_e_new;
  ei[c] = e_i_new;
}

__device__ inline double csr_support_closed_floor_internal_energy(
    const double mass,
    const double* __restrict__ zbar,
    const int c,
    const double gamma,
    const double A,
    const double te_floor,
    const double ti_floor) {
  const double A_safe = fmax(A, 1.0e-30);
  const double gm1 = fmax(gamma - 1.0, 1.0e-30);
  const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : 1.0;
  const double cv_i =
      tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  const double cv_e =
      z * tenryu::core::constants::eV_to_erg /
      (A_safe * tenryu::core::constants::proton_mass * gm1);
  return fmax(mass, 0.0) *
         (cv_e * fmax(te_floor, 0.0) + cv_i * fmax(ti_floor, 0.0));
}

__global__ void csr_support_closed_redistribute_total_energy_optionb_kernel(
    double* __restrict__ total_energy,
    const double* __restrict__ mass,
    const double* __restrict__ optionb_m_corner,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const double* __restrict__ zbar,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ closure_inactive_cell_mask,
    const int n_cells,
    const double gamma,
    const double A,
    const double te_floor,
    const double ti_floor,
    double* __restrict__ unresolved_deficit,
    int* __restrict__ unresolved_count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int c = 0; c < n_cells; ++c) {
    if (csr_inactive_cell(closure_inactive_cell_mask, c)) {
      continue;
    }
    const double m_c = fmax(mass[c], 0.0);
    if (!(m_c > 0.0) || !isfinite(m_c)) {
      continue;
    }
    const double K_c = csr_optionb_corner_kinetic_for_cell_from_masses(
        c,
        optionb_m_corner,
        v_r_node,
        v_z_node,
        cell_node_csr_offsets,
        cell_node_csr_indices,
        cell_nverts);
    const double floor_c = csr_support_closed_floor_internal_energy(
        m_c, zbar, c, gamma, A, te_floor, ti_floor);
    double deficit = floor_c - (total_energy[c] - K_c);
    if (!(deficit > 0.0) || !isfinite(deficit)) {
      continue;
    }
    for (int donor = 0; donor < n_cells && deficit > 0.0; ++donor) {
      if (donor == c || csr_inactive_cell(closure_inactive_cell_mask, donor)) {
        continue;
      }
      const double m_d = fmax(mass[donor], 0.0);
      if (!(m_d > 0.0) || !isfinite(m_d)) {
        continue;
      }
      const double K_d = csr_optionb_corner_kinetic_for_cell_from_masses(
          donor,
          optionb_m_corner,
          v_r_node,
          v_z_node,
          cell_node_csr_offsets,
          cell_node_csr_indices,
          cell_nverts);
      const double floor_d = csr_support_closed_floor_internal_energy(
          m_d, zbar, donor, gamma, A, te_floor, ti_floor);
      double surplus = (total_energy[donor] - K_d) - floor_d;
      if (!(surplus > 0.0) || !isfinite(surplus)) {
        continue;
      }
      const double delta = fmin(deficit, surplus);
      total_energy[donor] -= delta;
      total_energy[c] += delta;
      deficit -= delta;
    }
    const double tol = 1024.0 * tenryu::hydro::optionb::detail::kDoubleEps *
                       fmax(fabs(total_energy[c]), 1.0);
    if (deficit > tol && isfinite(deficit)) {
      if (unresolved_deficit != nullptr) {
        *unresolved_deficit += deficit;
      }
      if (unresolved_count != nullptr) {
        *unresolved_count += 1;
      }
    }
  }
}

__global__ void csr_capture_near_vacuum_pre_kernel(
    CsrNearVacuumRecord* __restrict__ record,
    const int cell,
    const int n_c,
    const int n_b,
    const double rho_floor,
    const double* __restrict__ mass_lag,
    const double* __restrict__ rho_lag,
    const double* __restrict__ vol_lag,
    const double* __restrict__ vol_ref,
    const double* __restrict__ e_tot_lag,
    const double* __restrict__ v_r_cell_lag,
    const double* __restrict__ v_z_cell_lag,
    const double* __restrict__ mass_raw,
    const double* __restrict__ mom_r_raw,
    const double* __restrict__ mom_z_raw,
    const double* __restrict__ total_energy_raw,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_ref,
    const double* __restrict__ x_z_ref,
    const double* __restrict__ v_r_node_pre,
    const double* __restrict__ v_z_node_pre,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  csr_near_vacuum_clear_record(record);
  record->found = 1;
  record->cell = cell;
  csr_near_vacuum_classify_cell(
      cell, n_c, n_b, &record->block_id, &record->index0, &record->index1);
  const int active_nverts = csr_active_nverts_for_cell(cell, cell_nverts);
  record->active_nverts = active_nverts;
  const int off = cell_node_csr_offsets[cell];
  double r_ref[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  double z_ref[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  for (int k = 0; k < active_nverts; ++k) {
    const int n = cell_node_csr_indices[off + k];
    record->node_id[k] = n;
    record->node_r[k] = x_r_old[n];
    record->node_z[k] = x_z_old[n];
    record->node_vr_pre[k] = v_r_node_pre[n];
    record->node_vz_pre[k] = v_z_node_pre[n];
    r_ref[k] = x_r_ref[n];
    z_ref[k] = x_z_ref[n];
  }
  csr_near_vacuum_polygon_centroid(
      r_ref, z_ref, active_nverts, &record->centroid_r, &record->centroid_z);
  record->centroid_rr =
      sqrt(record->centroid_r * record->centroid_r +
           record->centroid_z * record->centroid_z);
  record->V_lag = vol_lag[cell];
  record->V_ref = vol_ref[cell];
  record->V_ref_over_lag =
      (record->V_lag > 0.0 && isfinite(record->V_lag))
          ? (record->V_ref / record->V_lag)
          : 0.0;
  record->mass_floor = rho_floor * fmax(record->V_ref, kTinyVolume);
  record->mass_lag = mass_lag[cell];
  record->rho_lag = rho_lag[cell];
  record->total_energy_lag = mass_lag[cell] * fmax(e_tot_lag[cell], 0.0);
  record->v_r_cell_lag = v_r_cell_lag[cell];
  record->v_z_cell_lag = v_z_cell_lag[cell];
  record->mom_r_lag = mass_lag[cell] * v_r_cell_lag[cell];
  record->mom_z_lag = mass_lag[cell] * v_z_cell_lag[cell];
  record->m_raw = mass_raw[cell];
  record->rho_raw =
      (record->V_ref > 0.0 && isfinite(record->V_ref))
          ? (record->m_raw / record->V_ref)
          : 0.0;
  record->total_energy_remapped = total_energy_raw[cell];
  record->mom_r_raw = mom_r_raw[cell];
  record->mom_z_raw = mom_z_raw[cell];
  record->v_r_cell_raw =
      (record->m_raw > 0.0 && isfinite(record->m_raw))
          ? (record->mom_r_raw / record->m_raw)
          : 0.0;
  record->v_z_cell_raw =
      (record->m_raw > 0.0 && isfinite(record->m_raw))
          ? (record->mom_z_raw / record->m_raw)
          : 0.0;
}

__device__ inline double csr_near_vacuum_face_dm_to_cell(
    const bool second_order,
    const int donor_cell,
    const double dV_to_cell,
    const double dMr_to_cell,
    const double dMz_to_cell,
    const double flux_scale,
    const double* __restrict__ rho_lag,
    const double* __restrict__ rho_grad_r,
    const double* __restrict__ rho_grad_z,
    const double* __restrict__ vol_lag,
    const double* __restrict__ old_centroid_r,
    const double* __restrict__ old_centroid_z,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells,
    double* const rho_face_out) {
  if (!finite_nonzero(dV_to_cell)) {
    *rho_face_out = 0.0;
    return 0.0;
  }
  if (second_order) {
    const double rho_integral = csr_moments_direct_field_integral(
        rho_lag,
        rho_grad_r,
        rho_grad_z,
        old_centroid_r,
        old_centroid_z,
        x_r_old,
        x_z_old,
        face_adj_csr_offsets,
        face_adj_csr_indices,
        cell_node_csr_offsets,
        cell_node_csr_indices,
        cell_nverts,
        inactive_cell_mask,
        vol_lag[donor_cell],
        donor_cell,
        n_cells,
        dV_to_cell,
        dMr_to_cell,
        dMz_to_cell,
        false,
        {});
    const bool nondegenerate =
        vol_lag[donor_cell] > 0.0 &&
        fabs(dV_to_cell) >= 1.0e-12 * vol_lag[donor_cell];
    *rho_face_out = nondegenerate ? rho_integral / dV_to_cell
                                  : rho_lag[donor_cell];
    return flux_scale * rho_integral;
  }
  double rho_face = rho_lag[donor_cell];
  rho_face = fmax(rho_face, 0.0);
  *rho_face_out = rho_face;
  return flux_scale * rho_face * dV_to_cell;
}

__global__ void csr_capture_near_vacuum_flux_kernel(
    CsrNearVacuumRecord* __restrict__ record,
    const bool second_order,
    const double* __restrict__ rho_lag,
    const double* __restrict__ rho_grad_r,
    const double* __restrict__ rho_grad_z,
    const double* __restrict__ vol_lag,
    const double* __restrict__ old_centroid_r,
    const double* __restrict__ old_centroid_z,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_ref,
    const double* __restrict__ x_z_ref,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ unique_cell_a,
    const int* __restrict__ unique_cell_b,
    const int* __restrict__ unique_local_a,
    const int* __restrict__ boundary_cell,
    const int* __restrict__ boundary_local,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ mass_flux_scale,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const int n_cells,
    const int n_internal_faces,
    const int n_boundary_faces) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || record->found == 0) {
    return;
  }
  const int target = record->cell;
  record->mass_flux_sum = 0.0;
  for (int k = 0; k < kCsrNearVacuumTopFaces; ++k) {
    csr_near_vacuum_clear_face(&record->top_faces[k]);
  }
  for (int f = 0; f < n_internal_faces; ++f) {
    const int cell_a = unique_cell_a[f];
    const int cell_b = unique_cell_b[f];
    if (target != cell_a && target != cell_b) {
      continue;
    }
    const int local_a = unique_local_a[f];
    double dMr = 0.0;
    double dMz = 0.0;
    const double dV_a = second_order
                            ? csr_face_swept_raw_moments_outward(
                                  x_r_old,
                                  x_z_old,
                                  x_r_ref,
                                  x_z_ref,
                                  cell_node_csr_offsets,
                                  cell_node_csr_indices,
                                  cell_orientation_sign,
                                  cell_a,
                                  local_a,
                                  cell_nverts,
                                  &dMr,
                                  &dMz)
                            : csr_face_swept_volume_outward(
                                  x_r_old,
                                  x_z_old,
                                  x_r_ref,
                                  x_z_ref,
                                  cell_node_csr_offsets,
                                  cell_node_csr_indices,
                                  cell_orientation_sign,
                                  cell_a,
                                  local_a,
                                  cell_nverts);
    const int donor = csr_internal_flux_donor(cell_a, cell_b, dV_a);
    const int losing_cell = csr_internal_flux_losing_cell(cell_a, cell_b, dV_a);
    const double flux_scale =
        csr_clamped_flux_scale(mass_flux_scale, losing_cell);
    const double sign_to_cell = target == cell_a ? 1.0 : -1.0;
    const double signed_volume_to_cell = sign_to_cell * flux_scale * dV_a;
    double rho_face = 0.0;
    const double dm = csr_near_vacuum_face_dm_to_cell(second_order,
                                                      donor,
                                                      sign_to_cell * dV_a,
                                                      sign_to_cell * dMr,
                                                      sign_to_cell * dMz,
                                                      flux_scale,
                                                      rho_lag,
                                                      rho_grad_r,
                                                      rho_grad_z,
                                                      vol_lag,
                                                      old_centroid_r,
                                                      old_centroid_z,
                                                      x_r_old,
                                                      x_z_old,
                                                      face_adj_csr_offsets,
                                                      face_adj_csr_indices,
                                                      cell_node_csr_offsets,
                                                      cell_node_csr_indices,
                                                      cell_nverts,
                                                      inactive_cell_mask,
                                                      n_cells,
                                                      &rho_face);
    record->mass_flux_sum += dm;
    CsrNearVacuumFaceRecord face;
    csr_near_vacuum_clear_face(&face);
    face.face_id = f;
    face.boundary = 0;
    face.local_face = local_a;
    face.neighbor_cell = (target == cell_a) ? cell_b : cell_a;
    face.donor_cell = donor;
    face.signed_volume_to_cell = signed_volume_to_cell;
    face.rho_face = rho_face;
    face.dm_to_cell = dm;
    face.abs_dm = fabs(dm);
    csr_near_vacuum_insert_face(record, face);
  }
  for (int f = 0; f < n_boundary_faces; ++f) {
    const int cell = boundary_cell[f];
    if (target != cell) {
      continue;
    }
    const int local = boundary_local[f];
    double dMr = 0.0;
    double dMz = 0.0;
    const double dV = second_order
                          ? csr_face_swept_raw_moments_outward(
                                x_r_old,
                                x_z_old,
                                x_r_ref,
                                x_z_ref,
                                cell_node_csr_offsets,
                                cell_node_csr_indices,
                                cell_orientation_sign,
                                cell,
                                local,
                                cell_nverts,
                                &dMr,
                                &dMz)
                          : csr_face_swept_volume_outward(
                                x_r_old,
                                x_z_old,
                                x_r_ref,
                                x_z_ref,
                                cell_node_csr_offsets,
                                cell_node_csr_indices,
                                cell_orientation_sign,
                                cell,
                                local,
                                cell_nverts);
    double rho_face = 0.0;
    const double flux_scale =
        dV < 0.0 ? csr_clamped_flux_scale(mass_flux_scale, cell) : 1.0;
    const double dV_limited = flux_scale * dV;
    const double dm = csr_near_vacuum_face_dm_to_cell(second_order,
                                                      cell,
                                                      dV,
                                                      dMr,
                                                      dMz,
                                                      flux_scale,
                                                      rho_lag,
                                                      rho_grad_r,
                                                      rho_grad_z,
                                                      vol_lag,
                                                      old_centroid_r,
                                                      old_centroid_z,
                                                      x_r_old,
                                                      x_z_old,
                                                      face_adj_csr_offsets,
                                                      face_adj_csr_indices,
                                                      cell_node_csr_offsets,
                                                      cell_node_csr_indices,
                                                      cell_nverts,
                                                      inactive_cell_mask,
                                                      n_cells,
                                                      &rho_face);
    record->mass_flux_sum += dm;
    CsrNearVacuumFaceRecord face;
    csr_near_vacuum_clear_face(&face);
    face.face_id = f;
    face.boundary = 1;
    face.local_face = local;
    face.neighbor_cell = -1;
    face.donor_cell = cell;
    face.signed_volume_to_cell = dV_limited;
    face.rho_face = rho_face;
    face.dm_to_cell = dm;
    face.abs_dm = fabs(dm);
    csr_near_vacuum_insert_face(record, face);
  }
}

__global__ void csr_capture_near_vacuum_post_kernel(
    CsrNearVacuumRecord* __restrict__ record,
    const double* __restrict__ mass,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || record->found == 0) {
    return;
  }
  const int c = record->cell;
  const int active_nverts = record->active_nverts;
  const int off = cell_node_csr_offsets[c];
  for (int k = 0; k < active_nverts; ++k) {
    const int n = cell_node_csr_indices[off + k];
    record->node_vr_post[k] = v_r_node[n];
    record->node_vz_post[k] = v_z_node[n];
  }
  const double m = fmax(mass[c], 0.0);
  record->recovered_internal = m * (fmax(ee[c], 0.0) + fmax(ei[c], 0.0));
  record->post_projection_ke = csr_corner_kinetic_for_cell(c,
                                                           mass,
                                                           x_r,
                                                           x_z,
                                                           v_r_node,
                                                           v_z_node,
                                                           cell_node_csr_offsets,
                                                           cell_node_csr_indices,
                                                           cell_nverts,
                                                           nullptr,
                                                           rz::kCornerMassConventionBbswRadialV0,
                                                           nullptr,
                                                           0);
  record->internal_from_remapped_total =
      record->total_energy_remapped - record->post_projection_ke;
}

const char* csr_near_vacuum_block_name(const int block_id) {
  switch (block_id) {
    case 0:
      return "cap_core";
    case 1:
      return "north_fan";
    case 2:
      return "east_fan";
    case 3:
      return "south_fan";
    case 4:
      return "shell";
    default:
      return "unknown";
  }
}

void log_csr_near_vacuum_record(const long long step,
                                const CsrNearVacuumRecord& record) {
  if (record.found == 0 || record.cell < 0) {
    return;
  }
  std::ostringstream oss;
  oss.precision(17);
  oss << "[csr_near_vacuum_forensics] step=" << step
      << " cell=" << record.cell
      << " block_id=" << record.block_id
      << " block=" << csr_near_vacuum_block_name(record.block_id)
      << " index_kind=" << (record.block_id == 4 ? "q,k" : "l,k")
      << " index0=" << record.index0
      << " index1=" << record.index1
      << " active_nverts=" << record.active_nverts
      << " centroid_r=" << record.centroid_r
      << " centroid_z=" << record.centroid_z
      << " centroid_rr=" << record.centroid_rr
      << " V_lag=" << record.V_lag
      << " V_ref=" << record.V_ref
      << " V_ref_over_lag=" << record.V_ref_over_lag
      << " mass_floor=" << record.mass_floor
      << " pre_mass=" << record.mass_lag
      << " pre_rho=" << record.rho_lag
      << " pre_total_energy=" << record.total_energy_lag
      << " pre_mom_r=" << record.mom_r_lag
      << " pre_mom_z=" << record.mom_z_lag
      << " pre_v_cell_r=" << record.v_r_cell_lag
      << " pre_v_cell_z=" << record.v_z_cell_lag
      << " post_m_raw=" << record.m_raw
      << " post_rho_raw=" << record.rho_raw
      << " post_total_energy_raw=" << record.total_energy_remapped
      << " post_mom_r_raw=" << record.mom_r_raw
      << " post_mom_z_raw=" << record.mom_z_raw
      << " post_v_cell_r_raw=" << record.v_r_cell_raw
      << " post_v_cell_z_raw=" << record.v_z_cell_raw
      << " recovered_internal=" << record.recovered_internal
      << " internal_from_raw_total_minus_post_ke="
      << record.internal_from_remapped_total
      << " post_projection_ke=" << record.post_projection_ke
      << " mass_flux_sum=" << record.mass_flux_sum
      << " mass_reconstructed_from_flux="
      << (record.mass_lag + record.mass_flux_sum)
      << " nodes=[";
  for (int k = 0; k < record.active_nverts; ++k) {
    if (k > 0) {
      oss << ",";
    }
    oss << "{slot=" << k
        << ",id=" << record.node_id[k]
        << ",r=" << record.node_r[k]
        << ",z=" << record.node_z[k]
        << ",vr_pre=" << record.node_vr_pre[k]
        << ",vz_pre=" << record.node_vz_pre[k]
        << ",vr_post=" << record.node_vr_post[k]
        << ",vz_post=" << record.node_vz_post[k] << "}";
  }
  oss << "] top_faces=[";
  for (int k = 0; k < kCsrNearVacuumTopFaces; ++k) {
    const CsrNearVacuumFaceRecord& face = record.top_faces[k];
    if (face.face_id < 0) {
      continue;
    }
    if (k > 0) {
      oss << ",";
    }
    oss << "{rank=" << k
        << ",face_id=" << face.face_id
        << ",boundary=" << face.boundary
        << ",local_face=" << face.local_face
        << ",neighbor=" << face.neighbor_cell
        << ",donor=" << face.donor_cell
        << ",dV_to_cell=" << face.signed_volume_to_cell
        << ",rho_face=" << face.rho_face
        << ",dm_to_cell=" << face.dm_to_cell
        << ",abs_dm=" << face.abs_dm << "}";
  }
  oss << "]";
  core::log_warning(oss.str());
}

// Step-scoped worst |mass_closure_rel| across all CSR remap invocations
// (reset by the ALE phase entry, harvested into AleStepResult and by the
// driver repair routes). The accessors are defined OUTSIDE this anonymous
// namespace (external linkage), next to ale_remap_2d_rz.
double g_remap_mass_closure_step_max = 0.0;
void note_remap_mass_closure(const double rel) {
  if (std::abs(rel) > std::abs(g_remap_mass_closure_step_max)) {
    g_remap_mass_closure_step_max = rel;
  }
}

AleRemap2DRZResult conservative_remap_csr(
    core::State& state,
    const core::Config& cfg,
    const HydroEOSContext* eos_ctx,
    const double dt,
    const std::uint8_t* core_freeze_frozen_node_mask,
    const AleRemap2DRZOverrides& overrides) {
  (void)core_freeze_frozen_node_mask;
  AleRemap2DRZResult result;
  if (!cfg.numerics.ale.conservative_remap_enabled) {
    return result;
  }
  // 1T convention stores the total internal energy in ee with ei == 0; flooring
  // ei at cv_i*Ti_floor inside the remap fabricates unledgered energy (measured
  // +5e-5/pass via the next step's 1T fold-back). Zero the ion floor in 1T.
  const double ti_floor_remap =
      cfg.main.two_temperature ? cfg.numerics.floors.Ti : 0.0;
  conservation_audit::emit_stage(state, "ale_csr_swept_remap_pre");
  {
    // Stale-reference guard at the exact consumption point (the mid-ALE
    // breakage window that a once-per-step check misses).
    static const bool ref_repair = [] {
      const char* raw = std::getenv("TENRYU_I1B_CORE_REF_REPAIR");
      return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
    }();
    if (ref_repair) {
      central_pseudo_core::detect_and_repair_stale_core_reference(state);
    }
  }
  // CSR closure ledger (env TENRYU_I1B_CSR_CLOSURE_LEDGER): total stored-mass
  // sums before/after the remap body; on mismatch, dump the top offending
  // cells with their reference/current exact RZ polygon volumes and
  // vol_initial — pins the broken geometry pair behind the intermittent
  // grown-core mass creation (measured ~3e-7/event).
  static const bool csr_closure_ledger_enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_CSR_CLOSURE_LEDGER");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  // The scalar closure (sum_post - sum_pre) is measured UNCONDITIONALLY —
  // it feeds result.mass_closure_rel and the step-scoped accumulator the
  // driver's rejection gate reads; the env only gates the verbose offender
  // forensics (which also always fire on a violation).
  std::vector<double> closure_mass_pre;
  state.mass.copy_to_host(closure_mass_pre);
  // Watch-cell forensics (env TENRYU_I1B_CSR_WATCH_CELL): dump one cell's
  // mass at the remap's phase boundaries to bisect which window rewrites
  // it (built for the seam-cell mass-doubling root cause).
  static const int csr_watch_cell = [] {
    const char* raw = std::getenv("TENRYU_I1B_CSR_WATCH_CELL");
    return raw != nullptr && raw[0] != '\0' ? std::atoi(raw) : -1;
  }();
  static const int remap_watch_cell =
      env_int_value("TENRYU_I1B_REMAP_WATCH_CELL", -1);
  const int n_cells_watch = state.mesh.topo.n_cells;
  const auto csr_watch_dump = [&](const char* tag, const double* d_field) {
    if (csr_watch_cell < 0 || csr_watch_cell >= n_cells_watch ||
        d_field == nullptr) {
      return;
    }
    double m = std::numeric_limits<double>::quiet_NaN();
    CUDA_CHECK(cudaMemcpy(&m,
                          d_field + csr_watch_cell,
                          sizeof(double),
                          cudaMemcpyDeviceToHost));
    std::fprintf(stderr,
                 "[csr_watch] step=%d cell=%d %s m=%.17e\n",
                 state.step,
                 csr_watch_cell,
                 tag,
                 m);
  };
  csr_watch_dump("state_pre", state.mass.data());
  TENRYU_ASSERT(cfg.main.dimension == "2D_RZ",
                "CSR conservative RZ remap requires 2D_RZ geometry");
  TENRYU_ASSERT(cfg.numerics.ale.conservative_remap_target == "reference",
                "CSR conservative RZ remap supports only reference target");
  TENRYU_ASSERT(cfg.numerics.ale.conservative_remap_order == "first_order_donor" ||
                    cfg.numerics.ale.conservative_remap_order == "second_order_van_leer",
                "CSR conservative RZ remap order must be first_order_donor or "
                "second_order_van_leer");
  const bool total_energy_remap =
      cfg.numerics.hydro.total_energy_remap_2d_rz ||
      overrides.force_total_energy_remap;
  const bool optionb_allowed = state.corner_stride == 4;
  const bool optionb_velocity_authority =
      optionb_allowed &&
      (csr_optionb_velocity_authority_enabled(cfg) ||
       overrides.force_optionb_velocity_authority);
  const bool optionb_energy_coupling =
      total_energy_remap && optionb_velocity_authority;
  const bool physical_ke_remap =
      total_energy_remap && !optionb_energy_coupling &&
      ale_physical_ke_remap_env_enabled();
  const int corner_mass_convention =
      static_cast<int>(cfg.numerics.hydro.corner_mass_convention);
  TENRYU_ASSERT(!total_energy_remap ||
                    cfg.mesh.topology_scheme ==
                        core::TopologyScheme::
                            MULTIBLOCK_HALF_BUTTERFLY_TRIFAN_CAP_5BLOCK,
                "CSR conservative RZ total-energy remap is supported only for "
                "multiblock_half_butterfly_trifan_cap_5block");
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "CSR conservative RZ remap requires multiblock topology");
  if (eos_ctx != nullptr && eos_ctx->any_table) {
    TENRYU_ASSERT(false,
                  "CSR conservative RZ remap currently supports ideal-gas EOS only");
  }
  if (cfg.numerics.materials.per_material_conservation_enabled) {
    TENRYU_ASSERT(false,
                  "CSR conservative RZ remap currently supports cell-mixture "
                  "conservation only");
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const int n_cells = state.mesh.topo.n_cells;
  const int nz = state.mesh.topo.nz;
  const int n_nodes = state.mesh.topo.n_nodes;
  const RemapDispatchAuditDeviceView remap_dispatch_audit =
      remap_dispatch_audit_device_view();
  if (n_cells <= 0 || n_nodes <= 0) {
    return result;
  }
  static const bool txn_vth_sub_diag = [] {
    const char* raw = std::getenv("TENRYU_ALE_TXN_VTH_DIAG");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  const auto txn_vth_sub_cell =
      [&](const char* stage,
          const std::vector<double>& node_r,
          const std::vector<double>& node_z,
          const std::vector<double>& cell_mass,
          const std::vector<double>& cell_momentum_r,
          const std::vector<double>& cell_momentum_z) {
        if (!txn_vth_sub_diag) {
          return;
        }
        double max_abs_u_theta = 0.0;
        double max_theta_deg = 0.0;
        int max_cell = -1;
        for (int cell = 0; cell < n_cells; ++cell) {
          const int begin =
              mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
          const int nverts =
              state.mesh.cell_nverts.empty()
                  ? 4
                  : mesh::mesh_topo_cell_active_nverts(
                        state.mesh.cell_nverts, cell);
          double cell_r[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
          double cell_z[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
          for (int k = 0; k < nverts; ++k) {
            const int node = mb.cell_node_csr_indices[
                static_cast<std::size_t>(begin + k)];
            cell_r[k] = node_r[static_cast<std::size_t>(node)];
            cell_z[k] = node_z[static_cast<std::size_t>(node)];
          }
          double centroid_r = 0.0;
          double centroid_z = 0.0;
          rz::rz_polygon_area_centroid_exact(
              cell_r, cell_z, nverts, &centroid_r, &centroid_z);
          const double radius = std::hypot(centroid_r, centroid_z);
          if (!(radius > 3.0e-5 && radius < 1.5e-3)) {
            continue;
          }
          const std::size_t index = static_cast<std::size_t>(cell);
          const double u_r = cell_momentum_r[index] / cell_mass[index];
          const double u_z = cell_momentum_z[index] / cell_mass[index];
          const double u_theta =
              (u_r * centroid_z - u_z * centroid_r) / radius;
          const double abs_u_theta = std::abs(u_theta);
          if (max_cell < 0 || abs_u_theta > max_abs_u_theta) {
            max_abs_u_theta = abs_u_theta;
            max_cell = cell;
            max_theta_deg =
                std::atan2(centroid_r, centroid_z) * 180.0 /
                3.14159265358979323846;
          }
        }
        std::fprintf(stderr,
                     "[txn_vth_sub] stage=%s max_abs_u_theta=%.17e "
                     "cell=%d theta_deg=%.17e\n",
                     stage,
                     max_abs_u_theta,
                     max_cell,
                     max_theta_deg);
      };
  const auto txn_vth_sub_node =
      [&](const char* stage,
          const std::vector<double>& node_r,
          const std::vector<double>& node_z,
          const std::vector<double>& node_velocity_r,
          const std::vector<double>& node_velocity_z) {
        if (!txn_vth_sub_diag) {
          return;
        }
        double max_abs_u_theta = 0.0;
        double max_theta_deg = 0.0;
        int max_node = -1;
        for (int node = 0; node < n_nodes; ++node) {
          const std::size_t index = static_cast<std::size_t>(node);
          const double radius = std::hypot(node_r[index], node_z[index]);
          if (!(radius > 3.0e-5 && radius < 1.5e-3)) {
            continue;
          }
          const double u_theta =
              (node_velocity_r[index] * node_z[index] -
               node_velocity_z[index] * node_r[index]) /
              radius;
          const double abs_u_theta = std::abs(u_theta);
          if (max_node < 0 || abs_u_theta > max_abs_u_theta) {
            max_abs_u_theta = abs_u_theta;
            max_node = node;
            max_theta_deg =
                std::atan2(node_r[index], node_z[index]) * 180.0 /
                3.14159265358979323846;
          }
        }
        std::fprintf(stderr,
                     "[txn_vth_sub] stage=%s max_abs_u_theta=%.17e "
                     "node=%d theta_deg=%.17e\n",
                     stage,
                     max_abs_u_theta,
                     max_node,
                     max_theta_deg);
      };
  const std::size_t node_bytes =
      static_cast<std::size_t>(n_nodes) * sizeof(double);
  // Unconditional velocity-authority bound: nodes outside the caller's
  // active_node_velocity_mask must exit this function with bitwise the
  // velocities they entered with. Snapshot at entry; restore before
  // every return past this point. The corner-distribution phase writes only
  // changed nodes, so this remains defense in depth.
  core::DeviceArray<double> d_velocity_bound_vr;
  core::DeviceArray<double> d_velocity_bound_vz;
  if (overrides.active_node_velocity_mask != nullptr) {
    d_velocity_bound_vr.reset(static_cast<std::size_t>(n_nodes));
    d_velocity_bound_vz.reset(static_cast<std::size_t>(n_nodes));
    CUDA_CHECK(cudaMemcpy(d_velocity_bound_vr.data(), state.v_r.data(),
                          node_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_velocity_bound_vz.data(), state.v_z.data(),
                          node_bytes, cudaMemcpyDeviceToDevice));
  }
  const auto restore_out_of_mask_velocities = [&]() {
    if (overrides.active_node_velocity_mask == nullptr ||
        d_velocity_bound_vr.size() == 0U) {
      return;
    }
    const int blocks_nodes = (n_nodes + 255) / 256;
    csr_restore_unmasked_node_velocity_kernel<<<blocks_nodes, 256>>>(
        state.v_r.data(),
        state.v_z.data(),
        d_velocity_bound_vr.data(),
        d_velocity_bound_vz.data(),
        overrides.active_node_velocity_mask,
        n_nodes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(core::debug_kernel_sync());
  };
  pole_angular_derefine::ensure_built(state, cfg);
  const bool polar_shell_derefine_active =
      pole_angular_derefine::active(state);
  TENRYU_ASSERT(!polar_shell_derefine_active || !optionb_velocity_authority ||
                    overrides.allow_polar_shell_derefine,
                "polar shell angular de-refine gates out Option-B velocity authority");
  TENRYU_ASSERT(!polar_shell_derefine_active || !total_energy_remap ||
                    overrides.allow_polar_shell_derefine,
                "polar shell angular de-refine gates out total-energy remap");
  TENRYU_ASSERT(state.x_r_reference.size() == state.x_r.size() &&
                    state.x_z_reference.size() == state.x_z.size(),
                "CSR conservative RZ remap requires reference node storage");
  TENRYU_ASSERT(state.cell_vol_initial.size() == state.vol.size(),
                "CSR conservative RZ remap requires reference cell volumes");
  if (!cfg.numerics.ale
           .multiblock_lagrangian_bulk_center_patch_reference_enabled) {
    if (!prepare_multiblock_differential_reference_if_enabled(state, cfg)) {
      prepare_scaled_gamma_mvp_reference_if_enabled(state, cfg);
    }
  }
  if (diff_ref_diag_env_enabled()) {
    ale_reference_diagnostics::sample_reference_pre_remap(
        state, dt, state.ale_remaps_applied + 1);
  }
  if (!evaluate_csr_swept_face_audit(
          state, mb, cfg.numerics.ale.dgcl_commit_gate,
          cfg.numerics.ale.dgcl_commit_rtol)) {
    restore_out_of_mask_velocities();
    return result;
  }
  mesh_trace::trace_cell0_geometry(state, cfg, "csr_remap_pre");
  TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U,
                "CSR conservative RZ remap requires device cell-node CSR offsets");
  TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                    static_cast<std::size_t>(n_cells) *
                        static_cast<std::size_t>(state.corner_stride),
                "CSR conservative RZ remap requires device cell-node CSR indices");
  TENRYU_ASSERT(state.mesh.multiblock_reverse_csr_node_offsets.size() ==
                    static_cast<std::size_t>(n_nodes) + 1U,
                "CSR conservative RZ remap requires reverse node CSR offsets");
  TENRYU_ASSERT(state.mesh.multiblock_reverse_csr_node_cells.size() ==
                    state.mesh.multiblock_reverse_csr_node_corners.size(),
                "CSR conservative RZ remap reverse CSR payload mismatch");
  TENRYU_ASSERT(mb.cell_orientation_sign.size() ==
                    static_cast<std::size_t>(n_cells),
                "CSR conservative RZ remap requires per-cell orientation signs");
  for (const int sign : mb.cell_orientation_sign) {
    TENRYU_ASSERT(sign == 1 || sign == -1,
                  "CSR conservative RZ remap orientation signs must be +/-1");
  }
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "CSR conservative RZ remap requires a material definition");

  const int blocks_cells = (n_cells + 255) / 256;
  const int blocks_nodes = (n_nodes + 255) / 256;
  const int n_internal_faces = static_cast<int>(mb.unique_internal_faces.size());
  const int n_boundary_faces = static_cast<int>(mb.boundary_faces.size());
  const int n_edges = n_internal_faces + n_boundary_faces;
  const std::size_t cell_edge_incidence_count =
      2U * static_cast<std::size_t>(n_internal_faces) +
      static_cast<std::size_t>(n_boundary_faces);
  TENRYU_ASSERT(state.mesh.multiblock_cell_edge_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U,
                "CSR remap requires cell-edge CSR offsets");
  TENRYU_ASSERT(state.mesh.multiblock_cell_edge_csr_edges.size() ==
                        cell_edge_incidence_count &&
                    state.mesh.multiblock_cell_edge_csr_side.size() ==
                        cell_edge_incidence_count,
                "CSR remap requires cell-edge CSR entries");
  const int blocks_internal = (n_internal_faces + 255) / 256;
  const int blocks_boundary = (n_boundary_faces + 255) / 256;
  const auto& mat = cfg.materials.materials.front();
  const double gamma = mat.ideal_gas_gamma;
  const double A = mat.A > 0.0 ? mat.A : 1.0;
  const P3OracleConfig& p3_oracle = p3_oracle_config();
  const bool second_order_remap =
      cfg.numerics.ale.conservative_remap_order == "second_order_van_leer";
  static const bool no_face_clip = [] {
    const char* raw = std::getenv("TENRYU_I1B_REMAP_NO_FACE_CLIP");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  const bool cap_energy_audit = tenryu::hydro::cap_energy_audit_enabled();
  const bool i1b_spurious_sensor =
      tenryu::hydro::i1b_spurious_sensor_enabled();
  const bool collect_replay_diagnostics = overrides.collect_replay_diagnostics;
  const bool ke_projection_audit = cap_energy_audit || i1b_spurious_sensor;
  core::DeviceArray<std::uint8_t> d_combined_inactive_cell_mask("ale_remap:conservative_remap_csr:d_combined_inactive_cell_mask");
  core::DeviceArray<std::uint8_t> d_replay_inactive_cell_mask("ale_remap:conservative_remap_csr:d_replay_inactive_cell_mask");
  core::DeviceArray<std::uint8_t> d_energy_closure_inactive_cell_mask("ale_remap:conservative_remap_csr:d_energy_closure_inactive_cell_mask");
  const std::uint8_t* d_base_inactive_cell_mask =
      pole_angular_derefine::combined_inactive_mask_device(
          state, cfg, d_combined_inactive_cell_mask);
  const bool central_macro_remap_mask_active =
      d_base_inactive_cell_mask != nullptr;
  const std::uint8_t* d_inactive_cell_mask = d_base_inactive_cell_mask;
  if (overrides.active_cell_mask != nullptr) {
    d_replay_inactive_cell_mask.reset(static_cast<std::size_t>(n_cells));
    csr_combine_inactive_with_active_mask_kernel<<<blocks_cells, 256>>>(
        d_replay_inactive_cell_mask.data(),
        d_inactive_cell_mask,
        overrides.active_cell_mask,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
    d_inactive_cell_mask = d_replay_inactive_cell_mask.data();
  }
  const bool support_closed_energy_closure =
      optionb_allowed && overrides.energy_closure_cell_mask != nullptr;
  const std::uint8_t* d_energy_inactive_cell_mask = d_inactive_cell_mask;
  if (support_closed_energy_closure) {
    d_energy_closure_inactive_cell_mask.reset(
        static_cast<std::size_t>(n_cells));
    csr_combine_inactive_with_active_mask_kernel<<<blocks_cells, 256>>>(
        d_energy_closure_inactive_cell_mask.data(),
        d_base_inactive_cell_mask,
        overrides.energy_closure_cell_mask,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
    d_energy_inactive_cell_mask =
        d_energy_closure_inactive_cell_mask.data();
  }
  static bool near_vacuum_forensics_emitted = false;
  const bool near_vacuum_forensics =
      total_energy_remap && !near_vacuum_forensics_emitted &&
      csr_near_vacuum_forensics_enabled();
  bool near_vacuum_forensics_triggered = false;
  CsrRemapEnergyAuditState remap_energy_audit;
  remap_energy_audit.active =
      total_energy_remap && remap_energy_audit_enabled_for_invocation(state);
  remap_energy_audit.step = state.step;
  remap_energy_audit.remap = state.ale_remaps_applied + 1;
  CsrConsAuditLedger csr_cons_ledger;
  const CsrConsAuditContext csr_cons_context = g_csr_cons_audit_context;
  csr_cons_ledger.active =
      csr_cons_context.enabled && total_energy_remap &&
      cfg.numerics.ale.multiblock_lagrangian_bulk_center_patch_reference_enabled;
  if (csr_cons_ledger.active) {
    csr_cons_ledger.B = csr_cons_audit_capture_state_totals(
        state, cfg, csr_cons_context.reduction);
  }

  double* d_vr_cell = nullptr;
  double* d_vz_cell = nullptr;
  double* d_vr_new = nullptr;
  double* d_vz_new = nullptr;
  double* d_mass_new = nullptr;
  double* d_mom_r_new = nullptr;
  double* d_mom_z_new = nullptr;
  double* d_outgoing_mass_flux = nullptr;
  double* d_mass_flux_scale = nullptr;
  double* d_energy_e_new = nullptr;
  double* d_energy_i_new = nullptr;
  double* d_total_energy_lag = nullptr;
  double* d_total_energy_new = nullptr;
  double* d_ye_int_lag = nullptr;
  double* d_ye_int_new = nullptr;
  double* d_ke_cell_scale = nullptr;
  double* d_ke_node_scale = nullptr;
  double* d_mass_floor_delta = nullptr;
  double* d_E_floor_injected = nullptr;
  double* d_E_redistribution_unresolved = nullptr;
  int* d_n_eint_floor_hits = nullptr;
  int* d_n_active_floor_hits = nullptr;
  double* d_rho_new = nullptr;
  double* d_ee_new = nullptr;
  double* d_ei_new = nullptr;
  double* d_corner_fraction_lag = nullptr;
  double* d_corner_fraction_mass_new = nullptr;
  double* d_corner_fraction_grad_r = nullptr;
  double* d_corner_fraction_grad_z = nullptr;
  double* d_gas_tracer_mass_new = nullptr;
  double* d_gas_tracer_grad_r = nullptr;
  double* d_gas_tracer_grad_z = nullptr;
  double* d_burn_species_Y_lag = nullptr;
  double* d_burn_species_mass_new = nullptr;
  double* d_hot_e_eps_lag = nullptr;
  double* d_hot_e_eps_mass_new = nullptr;
  double* d_burn_eps_lag = nullptr;
  double* d_burn_eps_mass_new = nullptr;
  double* d_rad_ext_new = nullptr;
  double* d_rad_old = nullptr;
  double* d_rho_grad_r = nullptr;
  double* d_rho_grad_z = nullptr;
  double* d_old_centroid_r = nullptr;
  double* d_old_centroid_z = nullptr;
  double* d_ke_density_lag = nullptr;
  double* d_ke_ext_new = nullptr;
  double* d_ke_grad_r = nullptr;
  double* d_ke_grad_z = nullptr;
  double* d_ke_actual = nullptr;
  int* d_near_vacuum_cell = nullptr;
  CsrNearVacuumRecord* d_near_vacuum_record = nullptr;
  std::uint8_t* d_cell_nverts = nullptr;
  int* d_face_adj_offsets = nullptr;
  int* d_face_adj_indices = nullptr;
  int* d_cell_orientation_sign = nullptr;

  auto cleanup = [&]() {
    cudaFree(d_cell_nverts);
  };

  const std::size_t cell_bytes =
      static_cast<std::size_t>(n_cells) * sizeof(double);
  const std::size_t corner_bytes =
      static_cast<std::size_t>(state.corner_stride) * cell_bytes;
  const std::size_t burn_species_bytes =
      static_cast<std::size_t>(n_cells) * tenryu::burn::kNumSpecies *
      sizeof(double);
  const bool remap_corner_mass = true;
  const std::size_t expected_corner_mass_size =
      static_cast<std::size_t>(n_cells) *
      static_cast<std::size_t>(state.corner_stride);
  if (!state.corner_mass_initialized ||
      state.corner_mass.size() != expected_corner_mass_size) {
    tenryu::hydro::ensure_hourglass_subzonal_masses_2d(state, cfg, true);
  }
  TENRYU_ASSERT(state.corner_mass_initialized &&
                    state.corner_mass.size() == expected_corner_mass_size,
                "CSR corner distribution requires initialized corner mass");
  const bool remap_hotspot_gas_tracer =
      cfg.numerics.diagnostics.hotspot_gas.enabled &&
      !state.gas_tracer_Y.empty() &&
      state.gas_tracer_Y.size() == static_cast<std::size_t>(n_cells);
  const bool remap_burn_species =
      cfg.burn.enabled &&
      state.burn_n_host.size() ==
          static_cast<std::size_t>(n_cells) * tenryu::burn::kNumSpecies;
  const bool remap_hot_e_eps =
      !state.hot_e_eps_cum_host.empty() &&
      state.hot_e_eps_cum_host.size() == static_cast<std::size_t>(n_cells);
  // Pseudo-core and pole-derefine non-transfer ruling: docs/design/hote_2d_completion_20260717.md.
  const bool remap_burn_eps =
      !state.burn_eps_cum_host.empty() &&
      state.burn_eps_cum_host.size() == static_cast<std::size_t>(n_cells);
  const bool eta_contact_diag_enabled =
      diff_ref_diag_env_enabled() &&
      state.gas_tracer_initialized &&
      !state.gas_tracer_Y.empty() &&
      state.gas_tracer_Y.size() == static_cast<std::size_t>(n_cells);
  const bool use_tri_topology = has_tri_cells(state);
  core::DeviceArray<double> d_eta_contact_diag("ale_remap:conservative_remap_csr:d_eta_contact_diag");
  if (eta_contact_diag_enabled) {
    d_eta_contact_diag.reset(2U);
  }
  const bool gcl_audit_active = gcl_audit_env_enabled();
  double* d_gcl_internal_face_dV_to_cell_a = nullptr;
  double* d_gcl_boundary_face_dV_to_cell = nullptr;
  double* d_gcl_outward_swept_sum = nullptr;
  double* d_gcl_swept_abs_sum = nullptr;
  double* d_gcl_residual = nullptr;
  double* d_gcl_scale = nullptr;
  CsrGclAuditDeviceView gcl_audit_device;
  if (gcl_audit_active) {
    if (n_internal_faces > 0) {
      const std::size_t bytes =
          static_cast<std::size_t>(n_internal_faces) * sizeof(double);
      d_gcl_internal_face_dV_to_cell_a = static_cast<double*>(
          core::device_scratch_acquire(
              "ale_remap:csr:d_gcl_internal_face_dV_to_cell_a", bytes));
      CUDA_CHECK(cudaMemset(d_gcl_internal_face_dV_to_cell_a, 0, bytes));
    }
    if (n_boundary_faces > 0) {
      const std::size_t bytes =
          static_cast<std::size_t>(n_boundary_faces) * sizeof(double);
      d_gcl_boundary_face_dV_to_cell = static_cast<double*>(
          core::device_scratch_acquire(
              "ale_remap:csr:d_gcl_boundary_face_dV_to_cell", bytes));
      CUDA_CHECK(cudaMemset(d_gcl_boundary_face_dV_to_cell, 0, bytes));
    }
    d_gcl_outward_swept_sum = static_cast<double*>(
        core::device_scratch_acquire(
            "ale_remap:csr:d_gcl_outward_swept_sum", cell_bytes));
    d_gcl_swept_abs_sum = static_cast<double*>(
        core::device_scratch_acquire(
            "ale_remap:csr:d_gcl_swept_abs_sum", cell_bytes));
    d_gcl_residual = static_cast<double*>(core::device_scratch_acquire(
        "ale_remap:csr:d_gcl_residual", cell_bytes));
    d_gcl_scale = static_cast<double*>(core::device_scratch_acquire(
        "ale_remap:csr:d_gcl_scale", cell_bytes));
    CUDA_CHECK(cudaMemset(d_gcl_outward_swept_sum, 0, cell_bytes));
    CUDA_CHECK(cudaMemset(d_gcl_swept_abs_sum, 0, cell_bytes));
    gcl_audit_device.internal_face_dV_to_cell_a =
        d_gcl_internal_face_dV_to_cell_a;
    gcl_audit_device.boundary_face_dV_to_cell =
        d_gcl_boundary_face_dV_to_cell;
  }
  core::DeviceArray<double> d_central_macro_remap_audit("ale_remap:conservative_remap_csr:d_central_macro_remap_audit");
  double* d_central_macro_remap_audit_ptr = nullptr;
  if (central_macro_remap_mask_active) {
    d_central_macro_remap_audit.reset(
        static_cast<std::size_t>(kCentralMacroRemapAuditCount));
    d_central_macro_remap_audit_ptr =
        d_central_macro_remap_audit.data();
  }
  d_cell_nverts = upload_cell_nverts_if_needed(state, use_tri_topology);
  d_vr_cell = static_cast<double*>(
      core::device_scratch_acquire("ale_remap:csr:d_vr_cell", cell_bytes));
  d_vz_cell = static_cast<double*>(
      core::device_scratch_acquire("ale_remap:csr:d_vz_cell", cell_bytes));
  d_vr_new = static_cast<double*>(
      core::device_scratch_acquire("ale_remap:csr:d_vr_new", cell_bytes));
  d_vz_new = static_cast<double*>(
      core::device_scratch_acquire("ale_remap:csr:d_vz_new", cell_bytes));
  d_mass_new = static_cast<double*>(
      core::device_scratch_acquire("ale_remap:csr:d_mass_new", cell_bytes));
  d_mom_r_new = static_cast<double*>(
      core::device_scratch_acquire("ale_remap:csr:d_mom_r_new", cell_bytes));
  d_mom_z_new = static_cast<double*>(
      core::device_scratch_acquire("ale_remap:csr:d_mom_z_new", cell_bytes));
  d_outgoing_mass_flux = static_cast<double*>(
      core::device_scratch_acquire("ale_remap:csr:d_outgoing_mass_flux",
                                   cell_bytes));
  d_mass_flux_scale = static_cast<double*>(
      core::device_scratch_acquire("ale_remap:csr:d_mass_flux_scale",
                                   cell_bytes));
  d_energy_e_new = static_cast<double*>(
      core::device_scratch_acquire("ale_remap:csr:d_energy_e_new", cell_bytes));
  d_energy_i_new = static_cast<double*>(
      core::device_scratch_acquire("ale_remap:csr:d_energy_i_new", cell_bytes));
  if (remap_hotspot_gas_tracer) {
    d_gas_tracer_mass_new = static_cast<double*>(
        core::device_scratch_acquire("ale_remap:csr:d_gas_tracer_mass_new",
                                     cell_bytes));
    if (second_order_remap) {
      d_gas_tracer_grad_r = static_cast<double*>(
          core::device_scratch_acquire("ale_remap:csr:d_gas_tracer_grad_r",
                                       cell_bytes));
      d_gas_tracer_grad_z = static_cast<double*>(
          core::device_scratch_acquire("ale_remap:csr:d_gas_tracer_grad_z",
                                       cell_bytes));
    }
  }
  if (remap_burn_species) {
    d_burn_species_Y_lag = static_cast<double*>(
        core::device_scratch_acquire("burn:remap:Y_lag", burn_species_bytes));
    d_burn_species_mass_new = static_cast<double*>(
        core::device_scratch_acquire("burn:remap:mass_new", burn_species_bytes));
    CUDA_CHECK(cudaMemcpy(d_burn_species_Y_lag,
                          state.burn_n_host.data(),
                          burn_species_bytes,
                          cudaMemcpyHostToDevice));
  }
  if (remap_hot_e_eps) {
    d_hot_e_eps_lag = static_cast<double*>(
        core::device_scratch_acquire("ale_remap:csr:d_hot_e_eps_lag",
                                     cell_bytes));
    d_hot_e_eps_mass_new = static_cast<double*>(
        core::device_scratch_acquire("ale_remap:csr:d_hot_e_eps_mass_new",
                                     cell_bytes));
    CUDA_CHECK(cudaMemcpy(d_hot_e_eps_lag,
                          state.hot_e_eps_cum_host.data(),
                          cell_bytes,
                          cudaMemcpyHostToDevice));
  }
  if (remap_burn_eps) {
    d_burn_eps_lag = static_cast<double*>(
        core::device_scratch_acquire("ale_remap:csr:d_burn_eps_lag",
                                     cell_bytes));
    d_burn_eps_mass_new = static_cast<double*>(
        core::device_scratch_acquire("ale_remap:csr:d_burn_eps_mass_new",
                                     cell_bytes));
    CUDA_CHECK(cudaMemcpy(d_burn_eps_lag,
                          state.burn_eps_cum_host.data(),
                          cell_bytes,
                          cudaMemcpyHostToDevice));
  }
  if (total_energy_remap) {
    d_total_energy_lag = static_cast<double*>(
        core::device_scratch_acquire("ale_remap:csr:d_total_energy_lag",
                                     cell_bytes));
    d_total_energy_new = static_cast<double*>(
        core::device_scratch_acquire("ale_remap:csr:d_total_energy_new",
                                     cell_bytes));
    d_ye_int_lag = static_cast<double*>(
        core::device_scratch_acquire("ale_remap:csr:d_ye_int_lag", cell_bytes));
    d_ye_int_new = static_cast<double*>(
        core::device_scratch_acquire("ale_remap:csr:d_ye_int_new", cell_bytes));
    d_ke_cell_scale = static_cast<double*>(
        core::device_scratch_acquire("ale_remap:csr:d_ke_cell_scale",
                                     cell_bytes));
    d_ke_node_scale = static_cast<double*>(
        core::device_scratch_acquire(
            "ale_remap:csr:d_ke_node_scale",
            static_cast<std::size_t>(n_nodes) * sizeof(double)));
  }
  d_mass_floor_delta = static_cast<double*>(
      core::device_scratch_acquire("ale_remap:csr:d_mass_floor_delta",
                                   sizeof(double)));
  d_E_floor_injected = static_cast<double*>(
      core::device_scratch_acquire("ale_remap:csr:d_E_floor_injected",
                                   sizeof(double)));
  d_E_redistribution_unresolved = static_cast<double*>(
      core::device_scratch_acquire(
          "ale_remap:csr:d_E_redistribution_unresolved",
          sizeof(double)));
  if (collect_replay_diagnostics) {
    d_n_eint_floor_hits = static_cast<int*>(
        core::device_scratch_acquire("ale_remap:csr:d_n_eint_floor_hits",
                                     sizeof(int)));
    d_n_active_floor_hits = static_cast<int*>(
        core::device_scratch_acquire("ale_remap:csr:d_n_active_floor_hits",
                                     sizeof(int)));
  }
  d_rho_new = static_cast<double*>(
      core::device_scratch_acquire("ale_remap:csr:d_rho_new", cell_bytes));
  d_ee_new = static_cast<double*>(
      core::device_scratch_acquire("ale_remap:csr:d_ee_new", cell_bytes));
  d_ei_new = static_cast<double*>(
      core::device_scratch_acquire("ale_remap:csr:d_ei_new", cell_bytes));
  if (ke_projection_audit) {
    d_ke_density_lag = static_cast<double*>(
        core::device_scratch_acquire("ale_remap:csr:d_ke_density_lag",
                                     cell_bytes));
    d_ke_ext_new = static_cast<double*>(
        core::device_scratch_acquire("ale_remap:csr:d_ke_ext_new", cell_bytes));
    d_ke_actual = static_cast<double*>(
        core::device_scratch_acquire("ale_remap:csr:d_ke_actual", cell_bytes));
    if (second_order_remap) {
      d_ke_grad_r = static_cast<double*>(
          core::device_scratch_acquire("ale_remap:csr:d_ke_grad_r", cell_bytes));
      d_ke_grad_z = static_cast<double*>(
          core::device_scratch_acquire("ale_remap:csr:d_ke_grad_z", cell_bytes));
    }
  }
  memset_zero(d_mass_floor_delta, 1);
  memset_zero(d_E_floor_injected, 1);
  memset_zero(d_E_redistribution_unresolved, 1);
  memset_zero_int(d_n_eint_floor_hits, 1);
  memset_zero_int(d_n_active_floor_hits, 1);
  d_cell_orientation_sign = static_cast<int*>(
      core::device_scratch_acquire(
          "ale_remap:csr:d_cell_orientation_sign",
          static_cast<std::size_t>(n_cells) * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_cell_orientation_sign,
                        mb.cell_orientation_sign.data(),
                        static_cast<std::size_t>(n_cells) * sizeof(int),
                        cudaMemcpyHostToDevice));
  if (remap_corner_mass) {
    d_corner_fraction_lag = static_cast<double*>(
        core::device_scratch_acquire("ale_remap:csr:d_corner_fraction_lag",
                                     corner_bytes));
    d_corner_fraction_mass_new = static_cast<double*>(
        core::device_scratch_acquire(
            "ale_remap:csr:d_corner_fraction_mass_new",
            corner_bytes));
    if (second_order_remap) {
      d_corner_fraction_grad_r = static_cast<double*>(
          core::device_scratch_acquire(
              "ale_remap:csr:d_corner_fraction_grad_r",
              corner_bytes));
      d_corner_fraction_grad_z = static_cast<double*>(
          core::device_scratch_acquire(
              "ale_remap:csr:d_corner_fraction_grad_z",
              corner_bytes));
    }
  }
  if (second_order_remap) {
    TENRYU_ASSERT(mb.face_adj_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "CSR second-order remap requires face-adjacency CSR offsets");
    TENRYU_ASSERT(mb.face_adj_csr_indices.size() ==
                      static_cast<std::size_t>(n_cells) *
                          static_cast<std::size_t>(state.corner_stride),
                  "CSR second-order remap requires face-adjacency CSR indices");
    d_rho_grad_r = static_cast<double*>(
        core::device_scratch_acquire("ale_remap:csr:d_rho_grad_r", cell_bytes));
    d_rho_grad_z = static_cast<double*>(
        core::device_scratch_acquire("ale_remap:csr:d_rho_grad_z", cell_bytes));
    d_old_centroid_r = static_cast<double*>(core::device_scratch_acquire(
        "ale_remap:csr:d_old_centroid_r", cell_bytes));
    d_old_centroid_z = static_cast<double*>(core::device_scratch_acquire(
        "ale_remap:csr:d_old_centroid_z", cell_bytes));
    d_face_adj_offsets = static_cast<int*>(
        core::device_scratch_acquire("ale_remap:csr:d_face_adj_offsets",
                                     mb.face_adj_csr_offsets.size() * sizeof(int)));
    d_face_adj_indices = static_cast<int*>(
        core::device_scratch_acquire("ale_remap:csr:d_face_adj_indices",
                                     mb.face_adj_csr_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_face_adj_offsets,
                          mb.face_adj_csr_offsets.data(),
                          mb.face_adj_csr_offsets.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_face_adj_indices,
                          mb.face_adj_csr_indices.data(),
                          mb.face_adj_csr_indices.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
  }

  const int* d_cell_node_offsets =
      state.mesh.multiblock_cell_node_csr_offsets.data();
  const int* d_cell_node_indices =
      state.mesh.multiblock_cell_node_csr_indices.data();
  const int* d_unique_cell_a =
      thrust::raw_pointer_cast(mb.d_unique_face_cell_a.data());
  const int* d_unique_cell_b =
      thrust::raw_pointer_cast(mb.d_unique_face_cell_b.data());
  const int* d_unique_local_a =
      thrust::raw_pointer_cast(mb.d_unique_face_local_a.data());
  const int* d_boundary_cell =
      thrust::raw_pointer_cast(mb.d_boundary_face_cell.data());
  const int* d_boundary_local =
      thrust::raw_pointer_cast(mb.d_boundary_face_local.data());
  const int* d_cell_edge_offsets =
      state.mesh.multiblock_cell_edge_csr_offsets.data();
  const int* d_cell_edge_edges =
      state.mesh.multiblock_cell_edge_csr_edges.data();
  const std::int8_t* d_cell_edge_side =
      state.mesh.multiblock_cell_edge_csr_side.data();

  const std::size_t n_face_sides =
      2U * static_cast<std::size_t>(n_edges);
  std::size_t hydro_stage_quantity_count = 5U;
  if (remap_corner_mass) {
    hydro_stage_quantity_count += static_cast<std::size_t>(state.corner_stride);
  }
  if (remap_hotspot_gas_tracer) {
    hydro_stage_quantity_count += 1U;
  }
  if (remap_burn_species) {
    hydro_stage_quantity_count +=
        static_cast<std::size_t>(tenryu::burn::kNumSpecies);
  }
  if (remap_hot_e_eps) {
    hydro_stage_quantity_count += 1U;
  }
  if (remap_burn_eps) {
    hydro_stage_quantity_count += 1U;
  }
  core::DeviceArray<double> d_hydro_flux_stage(
      "ale_remap:conservative_remap_csr:d_hydro_flux_stage");
  d_hydro_flux_stage.reset(hydro_stage_quantity_count * n_face_sides);
  core::DeviceArray<double> d_outgoing_mass_flux_stage(
      "ale_remap:conservative_remap_csr:d_outgoing_mass_flux_stage");
  d_outgoing_mass_flux_stage.reset(n_face_sides);
  core::DeviceArray<double> d_scalar_ext_flux_stage(
      "ale_remap:conservative_remap_csr:d_scalar_ext_flux_stage");
  d_scalar_ext_flux_stage.reset(n_face_sides);

  CsrHydroFluxStageDeviceView hydro_flux_stage;
  hydro_flux_stage.n_face_sides = n_face_sides;
  hydro_flux_stage.corner_stride = state.corner_stride;
  std::size_t hydro_stage_offset = 0U;
  const auto take_hydro_stage = [&](const std::size_t quantity_count) {
    double* const result = n_face_sides > 0U
                               ? d_hydro_flux_stage.data() +
                                     hydro_stage_offset * n_face_sides
                               : nullptr;
    hydro_stage_offset += quantity_count;
    return result;
  };
  hydro_flux_stage.mass = take_hydro_stage(1U);
  hydro_flux_stage.mom_r = take_hydro_stage(1U);
  hydro_flux_stage.mom_z = take_hydro_stage(1U);
  if (total_energy_remap) {
    hydro_flux_stage.total_energy = take_hydro_stage(1U);
    hydro_flux_stage.ye_mass = take_hydro_stage(1U);
  } else {
    hydro_flux_stage.energy_e = take_hydro_stage(1U);
    hydro_flux_stage.energy_i = take_hydro_stage(1U);
  }
  if (remap_corner_mass) {
    hydro_flux_stage.corner_fraction_mass =
        take_hydro_stage(static_cast<std::size_t>(state.corner_stride));
  }
  if (remap_hotspot_gas_tracer) {
    hydro_flux_stage.gas_tracer_mass = take_hydro_stage(1U);
  }
  if (remap_burn_species) {
    hydro_flux_stage.burn_species_mass = take_hydro_stage(
        static_cast<std::size_t>(tenryu::burn::kNumSpecies));
  }
  if (remap_hot_e_eps) {
    hydro_flux_stage.hot_e_eps_mass = take_hydro_stage(1U);
  }
  if (remap_burn_eps) {
    hydro_flux_stage.burn_eps_mass = take_hydro_stage(1U);
  }
  TENRYU_ASSERT(hydro_stage_offset == hydro_stage_quantity_count,
                "CSR hydro flux stage layout mismatch");

  CsrHydroFluxAccumulatorDeviceView hydro_flux_accumulator;
  hydro_flux_accumulator.mass = d_mass_new;
  hydro_flux_accumulator.mom_r = d_mom_r_new;
  hydro_flux_accumulator.mom_z = d_mom_z_new;
  if (total_energy_remap) {
    hydro_flux_accumulator.total_energy = d_total_energy_new;
    hydro_flux_accumulator.ye_mass = d_ye_int_new;
  } else {
    hydro_flux_accumulator.energy_e = d_energy_e_new;
    hydro_flux_accumulator.energy_i = d_energy_i_new;
  }
  hydro_flux_accumulator.corner_fraction_mass =
      d_corner_fraction_mass_new;
  hydro_flux_accumulator.gas_tracer_mass = d_gas_tracer_mass_new;
  hydro_flux_accumulator.burn_species_mass = d_burn_species_mass_new;
  hydro_flux_accumulator.hot_e_eps_mass = d_hot_e_eps_mass_new;
  hydro_flux_accumulator.burn_eps_mass = d_burn_eps_mass_new;

  if (second_order_remap) {
    csr_compute_old_rz_volume_centroids_kernel<<<blocks_cells, 256>>>(
        d_old_centroid_r,
        d_old_centroid_z,
        state.x_r.data(),
        state.x_z.data(),
        d_cell_node_offsets,
        d_cell_node_indices,
        d_cell_orientation_sign,
        d_cell_nverts,
        d_inactive_cell_mask,
        n_cells,
        remap_dispatch_audit);
    CUDA_CHECK(cudaGetLastError());
  }

  csr_compute_cell_velocity_from_nodes_kernel<<<blocks_cells, 256>>>(
      d_vr_cell,
      d_vz_cell,
      state.v_r.data(),
      state.v_z.data(),
      d_cell_node_offsets,
      d_cell_node_indices,
      d_cell_nverts,
      n_cells);
  CUDA_CHECK(cudaGetLastError());

  if (ke_projection_audit) {
    csr_compute_corner_kinetic_density_kernel<<<blocks_cells, 256>>>(
        d_ke_density_lag,
        state.mass.data(),
        state.vol.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.v_r.data(),
        state.v_z.data(),
        d_cell_node_offsets,
        d_cell_node_indices,
        d_cell_nverts,
        d_inactive_cell_mask,
        d_cell_orientation_sign,
        rz::corner_mass_fallback_device_recorder(),
        corner_mass_convention,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
    csr_initialize_volume_scalar_extents_kernel<<<blocks_cells, 256>>>(
        d_ke_ext_new, d_ke_density_lag, state.vol.data(), n_cells);
    CUDA_CHECK(cudaGetLastError());
  }

  // Basis-coherent transport: the gather seeds corner momenta from
  // state.corner_mass, so the TER's PRE-remap kinetic energy must be booked
  // in that same basis (the post side then uses the component's projected
  // ledger): the TER deposit chain spans exactly the basis trajectory the
  // budget sees. Coherent overrides the parked frozen-K reference arm.
  const bool optionb_coherent =
      optionb_allowed &&
      (overrides.force_optionb_coherent ||
       csr_optionb_coherent_enabled(cfg)) &&
      state.corner_mass_initialized &&
      state.corner_mass.size() == static_cast<std::size_t>(n_cells) * 4U;
  TENRYU_ASSERT(!optionb_coherent || optionb_energy_coupling ||
                    !optionb_velocity_authority,
                "TENRYU_I1B_OPTIONB_COHERENT requires the Option-B "
                "total-energy coupling (total_energy_remap_2d_rz) so the "
                "re-recover KE discrepancy lands in the TER deposit");
  const bool ter_frozen_ke_basis =
      optionb_allowed && !optionb_coherent &&
      ter_frozen_ke_basis_env_enabled() &&
      state.corner_mass_initialized &&
      state.corner_mass.size() == static_cast<std::size_t>(n_cells) * 4U;
  std::vector<double> old_node_r;
  std::vector<double> old_node_z;
  std::vector<double> old_node_vr;
  std::vector<double> old_node_vz;
  state.x_r.copy_to_host(old_node_r);
  state.x_z.copy_to_host(old_node_z);
  state.v_r.copy_to_host(old_node_vr);
  state.v_z.copy_to_host(old_node_vz);
  const auto active_nverts = [&](const int cell) {
    return state.mesh.cell_nverts.empty()
               ? 4
               : mesh::mesh_topo_cell_active_nverts(
                     state.mesh.cell_nverts, cell);
  };
  // Seed the swept remap with the same AW corner quadrature used by the
  // target-side distribution, evaluated on the source geometry.
  const auto synchronize_source_momentum_aw = [&]() {
    std::vector<double> source_cell_mass;
    std::vector<double> source_cell_momentum_r;
    std::vector<double> source_cell_momentum_z;
    copy_device_pointer_to_host(d_mass_new, n_cells, source_cell_mass);
    copy_device_pointer_to_host(
        d_mom_r_new, n_cells, source_cell_momentum_r);
    copy_device_pointer_to_host(
        d_mom_z_new, n_cells, source_cell_momentum_z);
    for (int c = 0; c < n_cells; ++c) {
      const double cell_mass =
          source_cell_mass[static_cast<std::size_t>(c)];
      const int begin =
          mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
      const int nverts = active_nverts(c);
      double source_r[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
      double source_z[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
      for (int k = 0; k < nverts; ++k) {
        const int node = mb.cell_node_csr_indices[
            static_cast<std::size_t>(begin + k)];
        source_r[k] = old_node_r[static_cast<std::size_t>(node)];
        source_z[k] = old_node_z[static_cast<std::size_t>(node)];
      }
      const double source_volume_per_radian =
          std::abs(rz::rz_polygon_volume_exact(
                       source_r, source_z, nverts)) /
          kAwTwoPi;
      if (!(cell_mass > 0.0) || !std::isfinite(cell_mass) ||
          !(source_volume_per_radian > 0.0) ||
          !std::isfinite(source_volume_per_radian)) {
        continue;
      }

      double corner_area[
          mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
      csr_aw_barlow_corner_area_partition(
          source_r, source_z, nverts, corner_area);
      double first_moment_area = 0.0;
      for (int k = 0; k < nverts; ++k) {
        first_moment_area += source_r[k] * corner_area[k];
      }
      TENRYU_ASSERT(
          std::abs(first_moment_area - source_volume_per_radian) <=
              1.0e-12 * source_volume_per_radian,
          "AW source Barlow first-moment identity failed");

      const double cell_density = cell_mass / source_volume_per_radian;
      double momentum_r = 0.0;
      double momentum_z = 0.0;
      for (int k = 0; k < nverts; ++k) {
        const int node = mb.cell_node_csr_indices[
            static_cast<std::size_t>(begin + k)];
        const double corner_inertia = cell_density * corner_area[k];
        const double paired_mass = source_r[k] * corner_inertia;
        momentum_r +=
            paired_mass * old_node_vr[static_cast<std::size_t>(node)];
        momentum_z +=
            paired_mass * old_node_vz[static_cast<std::size_t>(node)];
      }
      source_cell_momentum_r[static_cast<std::size_t>(c)] = momentum_r;
      source_cell_momentum_z[static_cast<std::size_t>(c)] = momentum_z;
    }
    CUDA_CHECK(cudaMemcpy(d_mom_r_new,
                          source_cell_momentum_r.data(),
                          cell_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mom_z_new,
                          source_cell_momentum_z.data(),
                          cell_bytes,
                          cudaMemcpyHostToDevice));
  };
  if (total_energy_remap) {
    if (optionb_energy_coupling) {
      csr_build_total_energy_remap_state_optionb_kernel<<<blocks_cells, 256>>>(
          d_total_energy_lag,
          d_ye_int_lag,
          state.mass.data(),
          state.rho.data(),
          state.ee.data(),
          state.ei.data(),
          state.vol.data(),
          state.x_r.data(),
          state.x_z.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_cell_node_offsets,
          d_cell_node_indices,
          d_cell_nverts,
          d_inactive_cell_mask,
          support_closed_energy_closure ? overrides.energy_closure_cell_mask
                                        : nullptr,
          (ter_frozen_ke_basis || optionb_coherent)
              ? state.corner_mass.data()
              : nullptr,
          n_cells);
    } else if (physical_ke_remap) {
      csr_build_total_energy_remap_state_physical_ke_kernel<<<blocks_cells, 256>>>(
          d_total_energy_lag,
          d_ye_int_lag,
          state.mass.data(),
          state.rho.data(),
          state.ee.data(),
          state.ei.data(),
          state.vol.data(),
          state.x_r.data(),
          state.x_z.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_cell_nverts,
          d_cell_orientation_sign,
          rz::corner_mass_fallback_device_recorder(),
          corner_mass_convention,
          n_cells,
          nz);
    } else {
      csr_build_total_energy_remap_state_kernel<<<blocks_cells, 256>>>(
          d_total_energy_lag,
          d_ye_int_lag,
          state.mass.data(),
          state.rho.data(),
          state.ee.data(),
          state.ei.data(),
          state.vol.data(),
          state.x_r.data(),
          state.x_z.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_cell_node_offsets,
          d_cell_node_indices,
          d_cell_nverts,
          d_cell_orientation_sign,
          rz::corner_mass_fallback_device_recorder(),
          corner_mass_convention,
          n_cells);
    }
    CUDA_CHECK(cudaGetLastError());
    csr_initialize_hydro_total_extents_kernel<<<blocks_cells, 256>>>(
        d_mass_new,
        d_mom_r_new,
        d_mom_z_new,
        d_total_energy_new,
        d_ye_int_new,
        state.mass.data(),
        state.rho.data(),
        d_vr_cell,
        d_vz_cell,
        d_total_energy_lag,
        d_ye_int_lag,
        state.vol.data(),
        n_cells);
    CUDA_CHECK(cudaGetLastError());
    synchronize_source_momentum_aw();
    csr_remap_energy_audit_capture_pre(remap_energy_audit,
                                       state,
                                       d_total_energy_new,
                                       d_inactive_cell_mask,
                                       n_cells);
    if (csr_cons_ledger.active) {
      long double E_B_staged = 0.0L;
      csr_cons_audit_capture_staged(d_total_energy_new,
                                    n_cells,
                                    &csr_cons_ledger.cell_B,
                                    &E_B_staged);
      csr_cons_audit_reduce_staged(&E_B_staged, csr_cons_context.reduction);
    }
  } else {
    csr_initialize_hydro_extents_kernel<<<blocks_cells, 256>>>(
        d_mass_new,
        d_mom_r_new,
        d_mom_z_new,
        d_energy_e_new,
        d_energy_i_new,
        state.mass.data(),
        state.rho.data(),
        d_vr_cell,
        d_vz_cell,
        state.ee.data(),
        state.ei.data(),
        state.vol.data(),
        n_cells);
    CUDA_CHECK(cudaGetLastError());
    synchronize_source_momentum_aw();
  }

  if (remap_corner_mass) {
    csr_initialize_corner_fraction_remap_kernel<<<blocks_cells, 256>>>(
        d_corner_fraction_lag,
        d_corner_fraction_mass_new,
        state.corner_mass.data(),
        state.mass.data(),
        state.rho.data(),
        state.vol.data(),
        d_cell_nverts,
        state.corner_stride,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
  }
  if (remap_hotspot_gas_tracer) {
    csr_initialize_gas_tracer_remap_kernel<<<blocks_cells, 256>>>(
        d_gas_tracer_mass_new,
        state.gas_tracer_Y.data(),
        state.mass.data(),
        state.rho.data(),
        state.vol.data(),
        n_cells);
    CUDA_CHECK(cudaGetLastError());
  }
  if (remap_burn_species) {
    csr_initialize_burn_species_remap_kernel<<<blocks_cells, 256>>>(
        d_burn_species_mass_new,
        d_burn_species_Y_lag,
        state.mass.data(),
        state.rho.data(),
        state.vol.data(),
        n_cells);
    CUDA_CHECK(cudaGetLastError());
  }
  if (remap_hot_e_eps) {
    csr_initialize_hot_e_eps_remap_kernel<<<blocks_cells, 256>>>(
        d_hot_e_eps_mass_new,
        d_hot_e_eps_lag,
        state.mass.data(),
        state.rho.data(),
        state.vol.data(),
        n_cells);
    CUDA_CHECK(cudaGetLastError());
  }
  if (remap_burn_eps) {
    csr_initialize_burn_eps_remap_kernel<<<blocks_cells, 256>>>(
        d_burn_eps_mass_new,
        d_burn_eps_lag,
        state.mass.data(),
        state.rho.data(),
        state.vol.data(),
        n_cells);
    CUDA_CHECK(cudaGetLastError());
  }

  if (second_order_remap) {
    csr_compute_lsq_gradients_inactive_masked_kernel<<<blocks_cells, 256>>>(
        d_rho_grad_r,
        d_rho_grad_z,
        state.rho.data(),
        d_old_centroid_r,
        d_old_centroid_z,
        d_face_adj_offsets,
        d_face_adj_indices,
        d_cell_nverts,
        d_inactive_cell_mask,
        overrides.donor_fallback_cell_mask,
        n_cells,
        remap_dispatch_audit);
    CUDA_CHECK(cudaGetLastError());
    // The cell-node-point BJ clip limits exact affine slopes on one-sided
    // axis-adjacent stencils (alpha = 0.684 on affine data). The per-face clip
    // at the swept-region centroid in csr_moments_direct_field_integral
    // enforces boundedness with Kucharik-style linearity preservation.
    if (ke_projection_audit) {
      csr_compute_lsq_gradients_inactive_masked_kernel<<<blocks_cells, 256>>>(
          d_ke_grad_r,
          d_ke_grad_z,
          d_ke_density_lag,
          d_old_centroid_r,
          d_old_centroid_z,
          d_face_adj_offsets,
          d_face_adj_indices,
          d_cell_nverts,
          d_inactive_cell_mask,
          overrides.donor_fallback_cell_mask,
          n_cells,
          remap_dispatch_audit);
      CUDA_CHECK(cudaGetLastError());
    }
    if (remap_hotspot_gas_tracer) {
      csr_compute_lsq_gradients_inactive_masked_kernel<<<blocks_cells, 256>>>(
          d_gas_tracer_grad_r,
          d_gas_tracer_grad_z,
          state.gas_tracer_Y.data(),
          d_old_centroid_r,
          d_old_centroid_z,
          d_face_adj_offsets,
          d_face_adj_indices,
          d_cell_nverts,
          d_inactive_cell_mask,
          overrides.donor_fallback_cell_mask,
          n_cells,
          remap_dispatch_audit);
      CUDA_CHECK(cudaGetLastError());
    }
  }
  if (remap_corner_mass && second_order_remap) {
    for (int k = 0; k < state.corner_stride; ++k) {
      double* const fraction = d_corner_fraction_lag + k * n_cells;
      double* const grad_r = d_corner_fraction_grad_r + k * n_cells;
      double* const grad_z = d_corner_fraction_grad_z + k * n_cells;
      csr_compute_lsq_gradients_inactive_masked_kernel<<<blocks_cells, 256>>>(
          grad_r,
          grad_z,
          fraction,
          d_old_centroid_r,
          d_old_centroid_z,
          d_face_adj_offsets,
          d_face_adj_indices,
          d_cell_nverts,
          d_inactive_cell_mask,
          overrides.donor_fallback_cell_mask,
          n_cells,
          remap_dispatch_audit);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  memset_zero(d_outgoing_mass_flux, n_cells);
  if (n_edges > 0) {
    CUDA_CHECK(cudaMemset(d_outgoing_mass_flux_stage.data(),
                          0,
                          n_face_sides * sizeof(double)));
  }
  if (n_internal_faces > 0) {
    csr_accumulate_internal_hydro_outgoing_mass_kernel<<<blocks_internal, 256>>>(
        d_outgoing_mass_flux_stage.data(),
        second_order_remap,
        state.rho.data(),
        second_order_remap ? d_rho_grad_r : nullptr,
        second_order_remap ? d_rho_grad_z : nullptr,
        state.vol.data(),
        d_old_centroid_r,
        d_old_centroid_z,
        state.x_r.data(),
        state.x_z.data(),
        state.x_r_reference.data(),
        state.x_z_reference.data(),
        d_face_adj_offsets,
        d_face_adj_indices,
        d_cell_node_offsets,
        d_cell_node_indices,
        d_unique_cell_a,
        d_unique_cell_b,
        d_unique_local_a,
        d_cell_orientation_sign,
        d_cell_nverts,
        d_inactive_cell_mask,
        n_internal_faces,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
  }
  if (n_boundary_faces > 0) {
    csr_accumulate_boundary_hydro_outgoing_mass_kernel<<<blocks_boundary, 256>>>(
        d_outgoing_mass_flux_stage.data(),
        second_order_remap,
        state.rho.data(),
        second_order_remap ? d_rho_grad_r : nullptr,
        second_order_remap ? d_rho_grad_z : nullptr,
        state.vol.data(),
        d_old_centroid_r,
        d_old_centroid_z,
        state.x_r.data(),
        state.x_z.data(),
        state.x_r_reference.data(),
        state.x_z_reference.data(),
        d_face_adj_offsets,
        d_face_adj_indices,
        d_cell_node_offsets,
        d_cell_node_indices,
        d_boundary_cell,
        d_boundary_local,
        d_cell_orientation_sign,
        d_cell_nverts,
        d_inactive_cell_mask,
        n_internal_faces,
        n_boundary_faces,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
  }
  if (n_edges > 0) {
    csr_gather_face_side_scalar_stage_kernel<<<blocks_cells, 256>>>(
        d_outgoing_mass_flux,
        d_outgoing_mass_flux_stage.data(),
        d_cell_edge_offsets,
        d_cell_edge_edges,
        d_cell_edge_side,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
  }
  csr_compute_hydro_mass_flux_scale_kernel<<<blocks_cells, 256>>>(
      d_mass_flux_scale,
      d_outgoing_mass_flux,
      state.mass.data(),
      state.rho.data(),
      state.vol.data(),
      state.cell_vol_initial.data(),
      d_inactive_cell_mask,
      n_cells,
      cfg.numerics.floors.rho,
      remap_dispatch_audit);
  CUDA_CHECK(cudaGetLastError());

  if (eta_contact_diag_enabled) {
    csr_eta_contact_hotspot_volume_kernel<<<blocks_cells, 256>>>(
        d_eta_contact_diag.data(),
        state.gas_tracer_Y.data(),
        state.vol.data(),
        d_inactive_cell_mask,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
    if (n_internal_faces > 0) {
      csr_eta_contact_swept_volume_kernel<<<blocks_internal, 256>>>(
          d_eta_contact_diag.data(),
          second_order_remap,
          state.gas_tracer_Y.data(),
          state.x_r.data(),
          state.x_z.data(),
          state.x_r_reference.data(),
          state.x_z_reference.data(),
          d_cell_node_offsets,
          d_cell_node_indices,
          d_unique_cell_a,
          d_unique_cell_b,
          d_unique_local_a,
          d_cell_orientation_sign,
          d_cell_nverts,
          d_mass_flux_scale,
          d_inactive_cell_mask,
          n_internal_faces);
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(core::debug_kernel_sync());
    std::vector<double> eta_contact_diag;
    d_eta_contact_diag.copy_to_host(eta_contact_diag);
    const double contact_swept_volume = eta_contact_diag[0];
    const double hotspot_volume = eta_contact_diag[1];
    if (hotspot_volume > 0.0 && std::isfinite(hotspot_volume) &&
        contact_swept_volume >= 0.0 && std::isfinite(contact_swept_volume)) {
      result.eta_contact_step = contact_swept_volume / hotspot_volume;
    }
    std::ostringstream oss;
    oss << "[eta_contact_diag] step=" << state.step
        << " remap=" << (state.ale_remaps_applied + 1)
        << " eta_contact_step="
        << format_ale_velcoherence_value(result.eta_contact_step)
        << " contact_swept_volume_cm3="
        << format_ale_velcoherence_value(contact_swept_volume)
        << " hotspot_volume_cm3="
        << format_ale_velcoherence_value(hotspot_volume);
    core::log_info(oss.str());
  }

  csr_watch_dump("staged_post_init", d_mass_new);
  if (n_edges > 0) {
    CUDA_CHECK(cudaMemset(d_hydro_flux_stage.data(),
                          0,
                          hydro_stage_quantity_count * n_face_sides *
                              sizeof(double)));
    if (ke_projection_audit) {
      CUDA_CHECK(cudaMemset(d_scalar_ext_flux_stage.data(),
                            0,
                            n_face_sides * sizeof(double)));
    }
  }
  const auto launch_internal_hydro_flux = [&](const auto gcl_audit_tag) {
    constexpr bool kGclAudit = decltype(gcl_audit_tag)::value;
    if (second_order_remap) {
      csr_apply_internal_hydro_flux_second_order_kernel<kGclAudit>
          <<<blocks_internal, 256>>>(
          hydro_flux_stage,
          state.rho.data(),
          d_rho_grad_r,
          d_rho_grad_z,
          state.vol.data(),
          d_old_centroid_r,
          d_old_centroid_z,
          d_vr_cell,
          d_vz_cell,
          state.ee.data(),
          state.ei.data(),
          total_energy_remap ? d_total_energy_lag : nullptr,
          total_energy_remap ? d_ye_int_lag : nullptr,
          remap_corner_mass ? d_corner_fraction_lag : nullptr,
          remap_corner_mass ? d_corner_fraction_grad_r : nullptr,
          remap_corner_mass ? d_corner_fraction_grad_z : nullptr,
          remap_hotspot_gas_tracer ? state.gas_tracer_Y.data() : nullptr,
          remap_burn_species ? d_burn_species_Y_lag : nullptr,
          remap_hot_e_eps ? d_hot_e_eps_lag : nullptr,
          remap_burn_eps ? d_burn_eps_lag : nullptr,
          remap_hotspot_gas_tracer ? d_gas_tracer_grad_r : nullptr,
          remap_hotspot_gas_tracer ? d_gas_tracer_grad_z : nullptr,
          state.x_r.data(),
          state.x_z.data(),
          state.x_r_reference.data(),
          state.x_z_reference.data(),
          d_face_adj_offsets,
          d_face_adj_indices,
          d_cell_node_offsets,
          d_cell_node_indices,
          d_unique_cell_a,
          d_unique_cell_b,
          d_unique_local_a,
          d_cell_orientation_sign,
          d_cell_nverts,
          d_mass_flux_scale,
          d_inactive_cell_mask,
          d_central_macro_remap_audit_ptr,
          n_internal_faces,
          n_cells,
          no_face_clip,
          remap_watch_cell,
          remap_dispatch_audit,
          gcl_audit_device);
    } else {
      csr_apply_internal_hydro_flux_kernel<kGclAudit>
          <<<blocks_internal, 256>>>(
          hydro_flux_stage,
          state.rho.data(),
          d_vr_cell,
          d_vz_cell,
          state.ee.data(),
          state.ei.data(),
          total_energy_remap ? d_total_energy_lag : nullptr,
          total_energy_remap ? d_ye_int_lag : nullptr,
          remap_corner_mass ? d_corner_fraction_lag : nullptr,
          remap_hotspot_gas_tracer ? state.gas_tracer_Y.data() : nullptr,
          remap_burn_species ? d_burn_species_Y_lag : nullptr,
          remap_hot_e_eps ? d_hot_e_eps_lag : nullptr,
          remap_burn_eps ? d_burn_eps_lag : nullptr,
          state.x_r.data(),
          state.x_z.data(),
          state.x_r_reference.data(),
          state.x_z_reference.data(),
          d_cell_node_offsets,
          d_cell_node_indices,
          d_unique_cell_a,
          d_unique_cell_b,
          d_unique_local_a,
          d_cell_orientation_sign,
          d_cell_nverts,
          d_mass_flux_scale,
          d_inactive_cell_mask,
          d_central_macro_remap_audit_ptr,
          n_internal_faces,
          n_cells,
          remap_dispatch_audit,
          gcl_audit_device);
    }
  };
  if (n_internal_faces > 0) {
    if (gcl_audit_active) {
      launch_internal_hydro_flux(std::true_type{});
    } else {
      launch_internal_hydro_flux(std::false_type{});
    }
    CUDA_CHECK(cudaGetLastError());
    if (ke_projection_audit) {
      if (second_order_remap) {
        csr_apply_internal_volume_scalar_flux_second_order_kernel<<<blocks_internal, 256>>>(
            d_scalar_ext_flux_stage.data(),
            d_ke_density_lag,
            d_ke_grad_r,
            d_ke_grad_z,
            state.vol.data(),
            d_old_centroid_r,
            d_old_centroid_z,
            state.x_r.data(),
            state.x_z.data(),
            state.x_r_reference.data(),
            state.x_z_reference.data(),
            d_cell_node_offsets,
            d_cell_node_indices,
            d_unique_cell_a,
            d_unique_cell_b,
            d_unique_local_a,
            n_internal_faces,
            n_cells,
            d_face_adj_offsets,
            d_face_adj_indices,
            d_cell_orientation_sign,
            d_cell_nverts,
            d_mass_flux_scale,
            d_inactive_cell_mask,
            d_central_macro_remap_audit_ptr,
            remap_dispatch_audit);
      } else {
        csr_apply_internal_volume_scalar_flux_kernel<<<blocks_internal, 256>>>(
            d_scalar_ext_flux_stage.data(),
            d_ke_density_lag,
            state.x_r.data(),
            state.x_z.data(),
            state.x_r_reference.data(),
            state.x_z_reference.data(),
            d_cell_node_offsets,
            d_cell_node_indices,
            d_unique_cell_a,
            d_unique_cell_b,
            d_unique_local_a,
            n_internal_faces,
            d_cell_orientation_sign,
            d_cell_nverts,
            d_inactive_cell_mask,
            d_central_macro_remap_audit_ptr,
            remap_dispatch_audit);
      }
      CUDA_CHECK(cudaGetLastError());
    }
  }
  const auto launch_boundary_hydro_flux = [&](const auto gcl_audit_tag) {
    constexpr bool kGclAudit = decltype(gcl_audit_tag)::value;
    if (second_order_remap) {
      csr_apply_boundary_hydro_flux_second_order_kernel<kGclAudit>
          <<<blocks_boundary, 256>>>(
          hydro_flux_stage,
          state.rho.data(),
          d_rho_grad_r,
          d_rho_grad_z,
          state.vol.data(),
          d_old_centroid_r,
          d_old_centroid_z,
          d_vr_cell,
          d_vz_cell,
          state.ee.data(),
          state.ei.data(),
          total_energy_remap ? d_total_energy_lag : nullptr,
          total_energy_remap ? d_ye_int_lag : nullptr,
          remap_corner_mass ? d_corner_fraction_lag : nullptr,
          remap_corner_mass ? d_corner_fraction_grad_r : nullptr,
          remap_corner_mass ? d_corner_fraction_grad_z : nullptr,
          remap_hotspot_gas_tracer ? state.gas_tracer_Y.data() : nullptr,
          remap_burn_species ? d_burn_species_Y_lag : nullptr,
          remap_hot_e_eps ? d_hot_e_eps_lag : nullptr,
          remap_burn_eps ? d_burn_eps_lag : nullptr,
          remap_hotspot_gas_tracer ? d_gas_tracer_grad_r : nullptr,
          remap_hotspot_gas_tracer ? d_gas_tracer_grad_z : nullptr,
          state.x_r.data(),
          state.x_z.data(),
          state.x_r_reference.data(),
          state.x_z_reference.data(),
          d_face_adj_offsets,
          d_face_adj_indices,
          d_cell_node_offsets,
          d_cell_node_indices,
          d_boundary_cell,
          d_boundary_local,
          d_cell_orientation_sign,
          d_cell_nverts,
          d_mass_flux_scale,
          d_inactive_cell_mask,
          d_central_macro_remap_audit_ptr,
          n_internal_faces,
          n_boundary_faces,
          n_cells,
          no_face_clip,
          remap_watch_cell,
          remap_dispatch_audit,
          gcl_audit_device);
    } else {
      csr_apply_boundary_hydro_flux_kernel<kGclAudit>
          <<<blocks_boundary, 256>>>(
          hydro_flux_stage,
          state.rho.data(),
          d_vr_cell,
          d_vz_cell,
          state.ee.data(),
          state.ei.data(),
          total_energy_remap ? d_total_energy_lag : nullptr,
          total_energy_remap ? d_ye_int_lag : nullptr,
          remap_corner_mass ? d_corner_fraction_lag : nullptr,
          remap_hotspot_gas_tracer ? state.gas_tracer_Y.data() : nullptr,
          remap_burn_species ? d_burn_species_Y_lag : nullptr,
          remap_hot_e_eps ? d_hot_e_eps_lag : nullptr,
          remap_burn_eps ? d_burn_eps_lag : nullptr,
          state.x_r.data(),
          state.x_z.data(),
          state.x_r_reference.data(),
          state.x_z_reference.data(),
          d_cell_node_offsets,
          d_cell_node_indices,
          d_boundary_cell,
          d_boundary_local,
          d_cell_orientation_sign,
          d_cell_nverts,
          d_mass_flux_scale,
          d_inactive_cell_mask,
          d_central_macro_remap_audit_ptr,
          n_internal_faces,
          n_boundary_faces,
          n_cells,
          remap_dispatch_audit,
          gcl_audit_device);
    }
  };
  if (n_boundary_faces > 0) {
    if (gcl_audit_active) {
      launch_boundary_hydro_flux(std::true_type{});
    } else {
      launch_boundary_hydro_flux(std::false_type{});
    }
    CUDA_CHECK(cudaGetLastError());
    if (ke_projection_audit) {
      if (second_order_remap) {
        csr_apply_boundary_volume_scalar_flux_second_order_kernel<<<blocks_boundary, 256>>>(
            d_scalar_ext_flux_stage.data(),
            d_ke_density_lag,
            d_ke_grad_r,
            d_ke_grad_z,
            state.vol.data(),
            d_old_centroid_r,
            d_old_centroid_z,
            state.x_r.data(),
            state.x_z.data(),
            state.x_r_reference.data(),
            state.x_z_reference.data(),
            d_cell_node_offsets,
            d_cell_node_indices,
            d_boundary_cell,
            d_boundary_local,
            n_internal_faces,
            n_boundary_faces,
            n_cells,
            d_face_adj_offsets,
            d_face_adj_indices,
            d_cell_orientation_sign,
            d_cell_nverts,
            d_mass_flux_scale,
            d_inactive_cell_mask,
            d_central_macro_remap_audit_ptr,
            remap_dispatch_audit);
      } else {
        csr_apply_boundary_volume_scalar_flux_kernel<<<blocks_boundary, 256>>>(
            d_scalar_ext_flux_stage.data(),
            d_ke_density_lag,
            state.x_r.data(),
            state.x_z.data(),
            state.x_r_reference.data(),
            state.x_z_reference.data(),
            d_cell_node_offsets,
            d_cell_node_indices,
            d_boundary_cell,
            d_boundary_local,
            n_internal_faces,
            n_boundary_faces,
            d_cell_orientation_sign,
            d_cell_nverts,
            d_inactive_cell_mask,
            d_central_macro_remap_audit_ptr,
            remap_dispatch_audit);
      }
      CUDA_CHECK(cudaGetLastError());
    }
  }
  if (n_edges > 0) {
    csr_gather_hydro_flux_stage_kernel<<<blocks_cells, 256>>>(
        hydro_flux_accumulator,
        hydro_flux_stage,
        d_cell_edge_offsets,
        d_cell_edge_edges,
        d_cell_edge_side,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
    if (ke_projection_audit) {
      csr_gather_face_side_scalar_stage_kernel<<<blocks_cells, 256>>>(
          d_ke_ext_new,
          d_scalar_ext_flux_stage.data(),
          d_cell_edge_offsets,
          d_cell_edge_edges,
          d_cell_edge_side,
          n_cells);
      CUDA_CHECK(cudaGetLastError());
    }
  }
  csr_watch_dump("staged_post_flux_gather", d_mass_new);
  if (txn_vth_sub_diag) {
    std::vector<double> current_node_r;
    std::vector<double> current_node_z;
    std::vector<double> post_sweep_cell_mass;
    std::vector<double> post_sweep_cell_momentum_r;
    std::vector<double> post_sweep_cell_momentum_z;
    state.x_r.copy_to_host(current_node_r);
    state.x_z.copy_to_host(current_node_z);
    copy_device_pointer_to_host(
        d_mass_new, n_cells, post_sweep_cell_mass);
    copy_device_pointer_to_host(
        d_mom_r_new, n_cells, post_sweep_cell_momentum_r);
    copy_device_pointer_to_host(
        d_mom_z_new, n_cells, post_sweep_cell_momentum_z);
    txn_vth_sub_cell("post_sweep_zonal",
                     current_node_r,
                     current_node_z,
                     post_sweep_cell_mass,
                     post_sweep_cell_momentum_r,
                     post_sweep_cell_momentum_z);
  }

  if (gcl_audit_active) {
    if (n_internal_faces > 0) {
      csr_gcl_audit_accumulate_internal_faces_kernel<<<blocks_internal, 256>>>(
          d_gcl_outward_swept_sum,
          d_gcl_swept_abs_sum,
          d_gcl_internal_face_dV_to_cell_a,
          d_unique_cell_a,
          d_unique_cell_b,
          n_internal_faces);
      CUDA_CHECK(cudaGetLastError());
    }
    if (n_boundary_faces > 0) {
      csr_gcl_audit_accumulate_boundary_faces_kernel<<<blocks_boundary, 256>>>(
          d_gcl_outward_swept_sum,
          d_gcl_swept_abs_sum,
          d_gcl_boundary_face_dV_to_cell,
          d_boundary_cell,
          n_boundary_faces);
      CUDA_CHECK(cudaGetLastError());
    }
    csr_gcl_audit_finalize_cells_kernel<<<blocks_cells, 256>>>(
        d_gcl_residual,
        d_gcl_scale,
        d_gcl_outward_swept_sum,
        d_gcl_swept_abs_sum,
        state.x_r.data(),
        state.x_z.data(),
        state.x_r_reference.data(),
        state.x_z_reference.data(),
        d_cell_node_offsets,
        d_cell_node_indices,
        d_cell_orientation_sign,
        d_cell_nverts,
        d_inactive_cell_mask,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> gcl_residual;
    std::vector<double> gcl_scale;
    std::vector<std::uint8_t> gcl_inactive_cell_mask;
    copy_device_pointer_to_host(d_gcl_residual, n_cells, gcl_residual);
    copy_device_pointer_to_host(d_gcl_scale, n_cells, gcl_scale);
    if (d_inactive_cell_mask != nullptr) {
      copy_device_pointer_to_host(
          d_inactive_cell_mask, n_cells, gcl_inactive_cell_mask);
    }
    record_gcl_audit_transaction(
        state.step,
        cfg.numerics.ale.swept_volume_sign_fixed,
        true,
        d_mass_flux_scale != nullptr,
        gcl_residual,
        gcl_scale,
        gcl_inactive_cell_mask);
  }

  csr_remap_energy_audit_capture_swept(
      remap_energy_audit, d_total_energy_new, n_cells);
  if (csr_cons_ledger.active) {
    csr_cons_audit_capture_staged(d_total_energy_new,
                                  n_cells,
                                  &csr_cons_ledger.cell_C,
                                  &csr_cons_ledger.E_C);
    csr_cons_audit_reduce_staged(&csr_cons_ledger.E_C,
                                 csr_cons_context.reduction);
  }

  csr_watch_dump("staged_post_fluxes", d_mass_new);
  if (near_vacuum_forensics) {
    int sentinel = n_cells;
    d_near_vacuum_cell = static_cast<int*>(
        core::device_scratch_acquire("ale_remap:csr:d_near_vacuum_cell",
                                     sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_near_vacuum_cell,
                          &sentinel,
                          sizeof(int),
                          cudaMemcpyHostToDevice));
    csr_find_near_vacuum_cell_kernel<<<blocks_cells, 256>>>(
        d_mass_new,
        state.cell_vol_initial.data(),
        d_near_vacuum_cell,
        d_inactive_cell_mask,
        n_cells,
        cfg.numerics.floors.rho);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&sentinel,
                          d_near_vacuum_cell,
                          sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (sentinel >= 0 && sentinel < n_cells) {
      near_vacuum_forensics_triggered = true;
      d_near_vacuum_record = static_cast<CsrNearVacuumRecord*>(
          core::device_scratch_acquire("ale_remap:csr:d_near_vacuum_record",
                                       sizeof(CsrNearVacuumRecord)));
      csr_capture_near_vacuum_pre_kernel<<<1, 1>>>(
          d_near_vacuum_record,
          sentinel,
          cfg.mesh.multiblock_cart_core_n_c,
          cfg.mesh.multiblock_cart_core_bridge_layers,
          cfg.numerics.floors.rho,
          state.mass.data(),
          state.rho.data(),
          state.vol.data(),
          state.cell_vol_initial.data(),
          d_total_energy_lag,
          d_vr_cell,
          d_vz_cell,
          d_mass_new,
          d_mom_r_new,
          d_mom_z_new,
          d_total_energy_new,
          state.x_r.data(),
          state.x_z.data(),
          state.x_r_reference.data(),
          state.x_z_reference.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_cell_node_offsets,
          d_cell_node_indices,
          d_cell_nverts);
      CUDA_CHECK(cudaGetLastError());
      csr_capture_near_vacuum_flux_kernel<<<1, 1>>>(
          d_near_vacuum_record,
          second_order_remap,
          state.rho.data(),
          second_order_remap ? d_rho_grad_r : nullptr,
          second_order_remap ? d_rho_grad_z : nullptr,
          state.vol.data(),
          d_old_centroid_r,
          d_old_centroid_z,
          state.x_r.data(),
          state.x_z.data(),
          state.x_r_reference.data(),
          state.x_z_reference.data(),
          d_face_adj_offsets,
          d_face_adj_indices,
          d_cell_node_offsets,
          d_cell_node_indices,
          d_unique_cell_a,
          d_unique_cell_b,
          d_unique_local_a,
          d_boundary_cell,
          d_boundary_local,
          d_cell_orientation_sign,
          d_cell_nverts,
          d_mass_flux_scale,
          d_inactive_cell_mask,
          n_cells,
          n_internal_faces,
          n_boundary_faces);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  if (total_energy_remap) {
    csr_finish_total_hydro_remap_kernel<<<blocks_cells, 256>>>(
        d_rho_new,
        d_vr_new,
        d_vz_new,
        d_mass_new,
        d_mom_r_new,
        d_mom_z_new,
        d_total_energy_new,
        d_ye_int_new,
        state.cell_vol_initial.data(),
        state.rho.data(),
        state.zbar.empty() ? nullptr : state.zbar.data(),
        d_inactive_cell_mask,
        n_cells,
        cfg.numerics.floors.rho,
        cfg.numerics.floors.Te,
        ti_floor_remap,
        gamma,
        A,
        d_mass_floor_delta,
        d_E_redistribution_unresolved,
        d_n_active_floor_hits);
    CUDA_CHECK(cudaGetLastError());
    csr_remap_energy_audit_capture_recover_total(
        remap_energy_audit, d_total_energy_new, n_cells);
    if (csr_cons_ledger.active) {
      csr_cons_audit_capture_staged(d_total_energy_new,
                                    n_cells,
                                    &csr_cons_ledger.cell_D,
                                    &csr_cons_ledger.E_D);
      csr_cons_audit_reduce_staged(&csr_cons_ledger.E_D,
                                   csr_cons_context.reduction);
    }
  } else {
    csr_finish_hydro_remap_kernel<<<blocks_cells, 256>>>(
        d_rho_new,
        d_vr_new,
        d_vz_new,
        d_ee_new,
        d_ei_new,
        d_mass_new,
        d_mom_r_new,
        d_mom_z_new,
        d_energy_e_new,
        d_energy_i_new,
        state.cell_vol_initial.data(),
        state.rho.data(),
        state.ee.data(),
        state.ei.data(),
        state.zbar.empty() ? nullptr : state.zbar.data(),
        d_inactive_cell_mask,
        n_cells,
        cfg.numerics.floors.rho,
        cfg.numerics.floors.Te,
        ti_floor_remap,
        gamma,
        A,
        d_mass_floor_delta,
        d_E_redistribution_unresolved);
    CUDA_CHECK(cudaGetLastError());
  }

  if (remap_hotspot_gas_tracer) {
    csr_finish_gas_tracer_remap_kernel<<<blocks_cells, 256>>>(
        state.gas_tracer_Y.data(),
        d_gas_tracer_mass_new,
        d_mass_new,
        d_inactive_cell_mask,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
  }
  if (remap_burn_species) {
    csr_finish_burn_species_remap_kernel<<<blocks_cells, 256>>>(
        d_burn_species_Y_lag,
        d_burn_species_mass_new,
        d_mass_new,
        d_inactive_cell_mask,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(state.burn_n_host.data(),
                          d_burn_species_Y_lag,
                          burn_species_bytes,
                          cudaMemcpyDeviceToHost));
  }
  if (remap_hot_e_eps) {
    csr_finish_hot_e_eps_remap_kernel<<<blocks_cells, 256>>>(
        d_hot_e_eps_lag,
        d_hot_e_eps_mass_new,
        d_mass_new,
        d_inactive_cell_mask,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(state.hot_e_eps_cum_host.data(),
                          d_hot_e_eps_lag,
                          cell_bytes,
                          cudaMemcpyDeviceToHost));
  }
  if (remap_burn_eps) {
    csr_finish_burn_eps_remap_kernel<<<blocks_cells, 256>>>(
        d_burn_eps_lag,
        d_burn_eps_mass_new,
        d_mass_new,
        d_inactive_cell_mask,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(state.burn_eps_cum_host.data(),
                          d_burn_eps_lag,
                          cell_bytes,
                          cudaMemcpyDeviceToHost));
  }

  const int n_groups = std::max(cfg.radiation.groups, 1);
  const std::size_t expected_rad_size =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  if (cfg.numerics.ale.conservative_remap_radiation_enabled &&
      state.rad_E.size() == expected_rad_size) {
    d_rad_ext_new = static_cast<double*>(
        core::device_scratch_acquire("ale_remap:csr:d_rad_ext_new",
                                     cell_bytes));
    d_rad_old = static_cast<double*>(
        core::device_scratch_acquire("ale_remap:csr:d_rad_old", cell_bytes));
    for (int g = 0; g < n_groups; ++g) {
      gather_group_field_kernel<<<blocks_cells, 256>>>(
          d_rad_old, state.rad_E.data(), n_cells, n_groups, g);
      CUDA_CHECK(cudaGetLastError());
      csr_initialize_volume_scalar_extents_kernel<<<blocks_cells, 256>>>(
          d_rad_ext_new, d_rad_old, state.vol.data(), n_cells);
      CUDA_CHECK(cudaGetLastError());
      if (second_order_remap) {
        csr_compute_lsq_gradients_inactive_masked_kernel<<<blocks_cells, 256>>>(
            d_rho_grad_r,
            d_rho_grad_z,
            d_rad_old,
            d_old_centroid_r,
            d_old_centroid_z,
            d_face_adj_offsets,
            d_face_adj_indices,
            d_cell_nverts,
            d_inactive_cell_mask,
            overrides.donor_fallback_cell_mask,
            n_cells,
            remap_dispatch_audit);
        CUDA_CHECK(cudaGetLastError());
      }
      if (n_edges > 0) {
        CUDA_CHECK(cudaMemset(d_scalar_ext_flux_stage.data(),
                              0,
                              n_face_sides * sizeof(double)));
      }
      if (n_internal_faces > 0) {
        if (second_order_remap) {
          csr_apply_internal_volume_scalar_flux_second_order_kernel<<<blocks_internal, 256>>>(
              d_scalar_ext_flux_stage.data(),
              d_rad_old,
              d_rho_grad_r,
              d_rho_grad_z,
              state.vol.data(),
              d_old_centroid_r,
              d_old_centroid_z,
              state.x_r.data(),
              state.x_z.data(),
              state.x_r_reference.data(),
              state.x_z_reference.data(),
              d_cell_node_offsets,
              d_cell_node_indices,
              d_unique_cell_a,
              d_unique_cell_b,
              d_unique_local_a,
              n_internal_faces,
              n_cells,
              d_face_adj_offsets,
              d_face_adj_indices,
              d_cell_orientation_sign,
              d_cell_nverts,
              d_mass_flux_scale,
              d_inactive_cell_mask,
              d_central_macro_remap_audit_ptr,
              remap_dispatch_audit);
        } else {
          csr_apply_internal_volume_scalar_flux_kernel<<<blocks_internal, 256>>>(
              d_scalar_ext_flux_stage.data(),
              d_rad_old,
              state.x_r.data(),
              state.x_z.data(),
              state.x_r_reference.data(),
              state.x_z_reference.data(),
              d_cell_node_offsets,
              d_cell_node_indices,
              d_unique_cell_a,
              d_unique_cell_b,
              d_unique_local_a,
              n_internal_faces,
              d_cell_orientation_sign,
              d_cell_nverts,
              d_inactive_cell_mask,
              d_central_macro_remap_audit_ptr,
              remap_dispatch_audit);
        }
        CUDA_CHECK(cudaGetLastError());
      }
      if (n_boundary_faces > 0) {
        if (second_order_remap) {
          csr_apply_boundary_volume_scalar_flux_second_order_kernel<<<blocks_boundary, 256>>>(
              d_scalar_ext_flux_stage.data(),
              d_rad_old,
              d_rho_grad_r,
              d_rho_grad_z,
              state.vol.data(),
              d_old_centroid_r,
              d_old_centroid_z,
              state.x_r.data(),
              state.x_z.data(),
              state.x_r_reference.data(),
              state.x_z_reference.data(),
              d_cell_node_offsets,
              d_cell_node_indices,
              d_boundary_cell,
              d_boundary_local,
              n_internal_faces,
              n_boundary_faces,
              n_cells,
              d_face_adj_offsets,
              d_face_adj_indices,
              d_cell_orientation_sign,
              d_cell_nverts,
              d_mass_flux_scale,
              d_inactive_cell_mask,
              d_central_macro_remap_audit_ptr,
              remap_dispatch_audit);
        } else {
          csr_apply_boundary_volume_scalar_flux_kernel<<<blocks_boundary, 256>>>(
              d_scalar_ext_flux_stage.data(),
              d_rad_old,
              state.x_r.data(),
              state.x_z.data(),
              state.x_r_reference.data(),
              state.x_z_reference.data(),
              d_cell_node_offsets,
              d_cell_node_indices,
              d_boundary_cell,
              d_boundary_local,
              n_internal_faces,
              n_boundary_faces,
              d_cell_orientation_sign,
              d_cell_nverts,
              d_inactive_cell_mask,
              d_central_macro_remap_audit_ptr,
              remap_dispatch_audit);
        }
        CUDA_CHECK(cudaGetLastError());
      }
      if (n_edges > 0) {
        csr_gather_face_side_scalar_stage_kernel<<<blocks_cells, 256>>>(
            d_rad_ext_new,
            d_scalar_ext_flux_stage.data(),
            d_cell_edge_offsets,
            d_cell_edge_edges,
            d_cell_edge_side,
            n_cells);
        CUDA_CHECK(cudaGetLastError());
      }
      csr_finish_volume_scalar_remap_kernel<<<blocks_cells, 256>>>(
          d_rad_old,
          d_rad_ext_new,
          state.cell_vol_initial.data(),
          d_inactive_cell_mask,
          n_cells);
      CUDA_CHECK(cudaGetLastError());
      scatter_group_field_kernel<<<blocks_cells, 256>>>(
          state.rad_E.data(), d_rad_old, n_cells, n_groups, g);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  // Qualification-scale host phase: download each required field once, then
  // traverse cells and nodes in ascending order. Same-connectivity remaps do
  // not need a corner-containment walk.
  TENRYU_ASSERT(mb.face_adj_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U,
                "CSR corner distribution requires face-adjacency offsets");
  TENRYU_ASSERT(mb.face_adj_csr_indices.size() ==
                    static_cast<std::size_t>(n_cells) *
                        static_cast<std::size_t>(state.corner_stride),
                "CSR corner distribution requires face-adjacency indices");
  std::vector<double> target_node_r;
  std::vector<double> target_node_z;
  std::vector<double> old_corner_mass;
  std::vector<double> transported_corner_mass;
  std::vector<double> remapped_cell_mass;
  std::vector<double> remapped_cell_momentum_r;
  std::vector<double> remapped_cell_momentum_z;
  std::vector<std::uint8_t> inactive_cell_mask;
  state.x_r_reference.copy_to_host(target_node_r);
  state.x_z_reference.copy_to_host(target_node_z);
  state.corner_mass.copy_to_host(old_corner_mass);
  copy_device_pointer_to_host(
      d_corner_fraction_mass_new,
      n_cells * state.corner_stride,
      transported_corner_mass);
  copy_device_pointer_to_host(d_mass_new, n_cells, remapped_cell_mass);
  copy_device_pointer_to_host(
      d_mom_r_new, n_cells, remapped_cell_momentum_r);
  copy_device_pointer_to_host(
      d_mom_z_new, n_cells, remapped_cell_momentum_z);
  if (d_inactive_cell_mask != nullptr) {
    copy_device_pointer_to_host(
        d_inactive_cell_mask, n_cells, inactive_cell_mask);
  }

  std::vector<std::vector<int>> node_cells(
      static_cast<std::size_t>(n_nodes));
  std::vector<std::vector<int>> node_corners(
      static_cast<std::size_t>(n_nodes));
  std::vector<std::uint8_t> changed_cell(
      static_cast<std::size_t>(n_cells), 0U);
  std::vector<std::uint8_t> changed_node(
      static_cast<std::size_t>(n_nodes), 0U);
  std::vector<int> changed_cells;
  std::vector<int> changed_nodes;
  for (int c = 0; c < n_cells; ++c) {
    const int begin = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int nverts = active_nverts(c);
    bool moved = false;
    for (int k = 0; k < nverts; ++k) {
      const int node =
          mb.cell_node_csr_indices[static_cast<std::size_t>(begin + k)];
      TENRYU_ASSERT(node >= 0 && node < n_nodes,
                    "CSR corner distribution cell node is out of range");
      node_cells[static_cast<std::size_t>(node)].push_back(c);
      node_corners[static_cast<std::size_t>(node)].push_back(
          c * state.corner_stride + k);
      moved =
          moved ||
          !csr_corner_distribution_same_bits(
              old_node_r[static_cast<std::size_t>(node)],
              target_node_r[static_cast<std::size_t>(node)]) ||
          !csr_corner_distribution_same_bits(
              old_node_z[static_cast<std::size_t>(node)],
              target_node_z[static_cast<std::size_t>(node)]);
    }
    if (moved) {
      changed_cell[static_cast<std::size_t>(c)] = 1U;
      changed_cells.push_back(c);
    }
  }
  for (int node = 0; node < n_nodes; ++node) {
    for (const int cell : node_cells[static_cast<std::size_t>(node)]) {
      if (changed_cell[static_cast<std::size_t>(cell)] != 0U) {
        changed_nodes.push_back(node);
        break;
      }
    }
  }
  for (const int node : changed_nodes) {
    changed_node[static_cast<std::size_t>(node)] = 1U;
  }

  const std::size_t corner_count =
      static_cast<std::size_t>(n_cells) *
      static_cast<std::size_t>(state.corner_stride);
  std::vector<double> omega_node = target_node_r;
  std::vector<std::uint8_t> node_axis_mask(
      static_cast<std::size_t>(n_nodes), 0U);
  for (int node = 0; node < n_nodes; ++node) {
    node_axis_mask[static_cast<std::size_t>(node)] =
        omega_node[static_cast<std::size_t>(node)] == 0.0 ? 1U : 0U;
  }
  std::vector<double> old_corner_volume(corner_count, 0.0);
  std::vector<double> target_corner_volume(corner_count, 0.0);
  std::vector<double> mu_corner(corner_count, 0.0);
  std::vector<double> aw_paired_corner_mass(corner_count, 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const int begin = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int nverts = active_nverts(c);
    double old_r[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    double old_z[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    double target_r[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    double target_z[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    for (int k = 0; k < nverts; ++k) {
      const int node =
          mb.cell_node_csr_indices[static_cast<std::size_t>(begin + k)];
      old_r[k] = old_node_r[static_cast<std::size_t>(node)];
      old_z[k] = old_node_z[static_cast<std::size_t>(node)];
      target_r[k] = target_node_r[static_cast<std::size_t>(node)];
      target_z[k] = target_node_z[static_cast<std::size_t>(node)];
    }
    const int base = c * state.corner_stride;
    double unused_centroid_r[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    double unused_centroid_z[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    csr_corner_distribution_cell_geometry(
        old_r,
        old_z,
        nverts,
        old_corner_volume.data() + base,
        unused_centroid_r,
        unused_centroid_z);
    csr_corner_distribution_cell_geometry(
        target_r,
        target_z,
        nverts,
        target_corner_volume.data() + base,
        unused_centroid_r,
        unused_centroid_z);

    double corner_area[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    csr_aw_barlow_corner_area_partition(
        target_r, target_z, nverts, corner_area);

    const double target_volume_per_radian =
        std::abs(rz::rz_polygon_volume_exact(
                     target_r, target_z, nverts)) /
        kAwTwoPi;
    TENRYU_ASSERT(
        target_volume_per_radian > 0.0 &&
            std::isfinite(target_volume_per_radian),
        "AW target RZ polygon volume per radian must be positive and finite");
    double first_moment_area = 0.0;
    for (int k = 0; k < nverts; ++k) {
      first_moment_area += target_r[k] * corner_area[k];
    }
    TENRYU_ASSERT(
        std::abs(first_moment_area - target_volume_per_radian) <=
            1.0e-12 * target_volume_per_radian,
        "AW Barlow first-moment identity failed");
    const double cell_mass =
        remapped_cell_mass[static_cast<std::size_t>(c)];
    const double cell_density = cell_mass / target_volume_per_radian;
    double paired_mass_sum = 0.0;
    for (int k = 0; k < nverts; ++k) {
      const int node =
          mb.cell_node_csr_indices[static_cast<std::size_t>(begin + k)];
      const std::size_t corner = static_cast<std::size_t>(base + k);
      mu_corner[corner] = cell_density * corner_area[k];
      aw_paired_corner_mass[corner] =
          omega_node[static_cast<std::size_t>(node)] * mu_corner[corner];
      paired_mass_sum += aw_paired_corner_mass[corner];
    }
    if (changed_cell[static_cast<std::size_t>(c)] != 0U) {
      TENRYU_ASSERT(
          std::abs(paired_mass_sum - cell_mass) <= 1.0e-10 * cell_mass,
          "AW paired corner masses must sum to the remapped cell mass");
    }
  }

  std::vector<double> distributed_corner_mass = old_corner_mass;
  for (int c = 0; c < n_cells; ++c) {
    if (!inactive_cell_mask.empty() &&
        inactive_cell_mask[static_cast<std::size_t>(c)] != 0U) {
      continue;
    }
    const int nverts = active_nverts(c);
    const int base = c * state.corner_stride;
    const double cell_mass =
        remapped_cell_mass[static_cast<std::size_t>(c)];
    double transported_sum = 0.0;
    for (int k = 0; k < nverts; ++k) {
      const double value =
          transported_corner_mass[static_cast<std::size_t>(k) *
                                      static_cast<std::size_t>(n_cells) +
                                  static_cast<std::size_t>(c)];
      if (value > 0.0 && std::isfinite(value)) {
        transported_sum += value;
      }
    }
    if (transported_sum > 1.0e-300 && std::isfinite(transported_sum)) {
      double partial = 0.0;
      for (int k = 0; k + 1 < nverts; ++k) {
        const double value =
            transported_corner_mass[static_cast<std::size_t>(k) *
                                        static_cast<std::size_t>(n_cells) +
                                    static_cast<std::size_t>(c)];
        distributed_corner_mass[static_cast<std::size_t>(base + k)] =
            (value > 0.0 && std::isfinite(value))
                ? cell_mass * value / transported_sum
                : 0.0;
        partial +=
            distributed_corner_mass[static_cast<std::size_t>(base + k)];
      }
      distributed_corner_mass[
          static_cast<std::size_t>(base + nverts - 1)] =
          cell_mass - partial;
    } else {
      const double uniform = cell_mass / static_cast<double>(nverts);
      double partial = 0.0;
      for (int k = 0; k + 1 < nverts; ++k) {
        distributed_corner_mass[static_cast<std::size_t>(base + k)] =
            uniform;
        partial += uniform;
      }
      distributed_corner_mass[
          static_cast<std::size_t>(base + nverts - 1)] =
          cell_mass - partial;
    }
    for (int k = nverts; k < state.corner_stride; ++k) {
      distributed_corner_mass[static_cast<std::size_t>(base + k)] = 0.0;
    }
  }

  const auto edge_stencil = [&](const int cell) {
    std::vector<int> stencil{cell};
    const int begin =
        mb.face_adj_csr_offsets[static_cast<std::size_t>(cell)];
    const int end =
        mb.face_adj_csr_offsets[static_cast<std::size_t>(cell) + 1U];
    for (int entry = begin; entry < end; ++entry) {
      const int adjacent =
          mb.face_adj_csr_indices[static_cast<std::size_t>(entry)];
      if (adjacent >= 0 && adjacent < n_cells) {
        stencil.push_back(adjacent);
      }
    }
    std::sort(stencil.begin(), stencil.end());
    stencil.erase(std::unique(stencil.begin(), stencil.end()), stencil.end());
    return stencil;
  };

  using tenryu::hydro::corner_distribution::AwAggregateProjectionInputs;
  using tenryu::hydro::corner_distribution::OldMeshView;
  const OldMeshView old_mesh{
      old_node_r.data(),
      old_node_z.data(),
      n_nodes,
      mb.cell_node_csr_offsets.data(),
      mb.cell_node_csr_indices.data(),
      n_cells,
      old_node_vr.data(),
      old_node_vz.data()};
  std::vector<double> u_ref_r = old_node_vr;
  std::vector<double> u_ref_z = old_node_vz;
  const std::size_t changed_node_count = changed_nodes.size();
  std::vector<int> changed_node_compact_index(
      static_cast<std::size_t>(n_nodes), -1);
  std::vector<double> u_ref_s(changed_node_count, 0.0);
  std::vector<double> u_ref_t(changed_node_count, 0.0);
  std::vector<double> s_min(changed_node_count, 0.0);
  std::vector<double> s_max(changed_node_count, 0.0);
  std::vector<double> t_min(changed_node_count, 0.0);
  std::vector<double> t_max(changed_node_count, 0.0);
  std::vector<double> basis_s_r(changed_node_count, 0.0);
  std::vector<double> basis_s_z(changed_node_count, 0.0);
  std::vector<double> basis_t_r(changed_node_count, 0.0);
  std::vector<double> basis_t_z(changed_node_count, 0.0);
  std::vector<std::uint8_t> compact_axis_mask(changed_node_count, 0U);
  for (std::size_t compact = 0; compact < changed_node_count; ++compact) {
    const int node = changed_nodes[compact];
    changed_node_compact_index[static_cast<std::size_t>(node)] =
        static_cast<int>(compact);
    const double target_r = target_node_r[static_cast<std::size_t>(node)];
    const double target_z = target_node_z[static_cast<std::size_t>(node)];
    const double target_s = std::hypot(target_r, target_z);
    TENRYU_ASSERT(std::isfinite(target_s),
                  "AW changed-node target radius must be finite");
    if (target_s > 0.0) {
      basis_s_r[compact] = target_r / target_s;
      basis_s_z[compact] = target_z / target_s;
      basis_t_r[compact] = target_z / target_s;
      basis_t_z[compact] = -target_r / target_s;
    } else {
      // Origin node: continuous +z-axis limit of the spherical basis.
      // r == 0 puts the node on the axis mask below, which zeroes the
      // tangential scalar and its bounds, so u_r = 0 binds exactly here.
      basis_s_r[compact] = 0.0;
      basis_s_z[compact] = 1.0;
      basis_t_r[compact] = 1.0;
      basis_t_z[compact] = 0.0;
    }
    compact_axis_mask[compact] =
        node_axis_mask[static_cast<std::size_t>(node)];
    tenryu::hydro::corner_distribution::reference_velocity_at(
        old_mesh,
        node_cells,
        node,
        target_r,
        target_z,
        &u_ref_r[static_cast<std::size_t>(node)],
        &u_ref_z[static_cast<std::size_t>(node)],
        &u_ref_s[compact],
        &u_ref_t[compact],
        &s_min[compact],
        &s_max[compact],
        &t_min[compact],
        &t_max[compact]);
    if (compact_axis_mask[compact] != 0U) {
      u_ref_t[compact] = 0.0;
      t_min[compact] = 0.0;
      t_max[compact] = 0.0;
    }
    u_ref_r[static_cast<std::size_t>(node)] =
        u_ref_s[compact] * basis_s_r[compact] +
        u_ref_t[compact] * basis_t_r[compact];
    u_ref_z[static_cast<std::size_t>(node)] =
        u_ref_s[compact] * basis_s_z[compact] +
        u_ref_t[compact] * basis_t_z[compact];
  }

  const std::vector<double> predictor_cell_momentum_r =
      remapped_cell_momentum_r;
  const std::vector<double> predictor_cell_momentum_z =
      remapped_cell_momentum_z;
  std::vector<double> a_node(changed_node_count, 0.0);
  std::vector<double> mu_node_region(changed_node_count, 0.0);
  double predictor_momentum_sum_r = 0.0;
  double predictor_momentum_sum_z = 0.0;
  for (const int c : changed_cells) {
    predictor_momentum_sum_r +=
        predictor_cell_momentum_r[static_cast<std::size_t>(c)];
    predictor_momentum_sum_z +=
        predictor_cell_momentum_z[static_cast<std::size_t>(c)];
    const int begin = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int nverts = active_nverts(c);
    const int base = c * state.corner_stride;
    for (int k = 0; k < nverts; ++k) {
      const int node =
          mb.cell_node_csr_indices[static_cast<std::size_t>(begin + k)];
      const int compact =
          changed_node_compact_index[static_cast<std::size_t>(node)];
      TENRYU_ASSERT(compact >= 0,
                    "AW changed-cell node is absent from compact node set");
      const double mu =
          mu_corner[static_cast<std::size_t>(base + k)];
      mu_node_region[static_cast<std::size_t>(compact)] += mu;
      a_node[static_cast<std::size_t>(compact)] +=
          omega_node[static_cast<std::size_t>(node)] * mu;
    }
  }

  double reference_momentum_sum_r = 0.0;
  double reference_momentum_sum_z = 0.0;
  double reference_momentum_scale_r = 0.0;
  double reference_momentum_scale_z = 0.0;
  for (std::size_t compact = 0; compact < changed_node_count; ++compact) {
    const int node = changed_nodes[compact];
    const double contribution_r =
        a_node[compact] * u_ref_r[static_cast<std::size_t>(node)];
    const double contribution_z =
        a_node[compact] * u_ref_z[static_cast<std::size_t>(node)];
    reference_momentum_sum_r += contribution_r;
    reference_momentum_sum_z += contribution_z;
    reference_momentum_scale_r += std::abs(contribution_r);
    reference_momentum_scale_z += std::abs(contribution_z);
  }
  const double aggregate_tolerance_r =
      64.0 * std::numeric_limits<double>::epsilon() *
      (std::abs(predictor_momentum_sum_r) + reference_momentum_scale_r);
  const double aggregate_tolerance_z =
      64.0 * std::numeric_limits<double>::epsilon() *
      (std::abs(predictor_momentum_sum_z) + reference_momentum_scale_z);
  const bool aggregate_exact =
      std::abs(reference_momentum_sum_r - predictor_momentum_sum_r) <=
          aggregate_tolerance_r &&
      std::abs(reference_momentum_sum_z - predictor_momentum_sum_z) <=
          aggregate_tolerance_z;
  std::vector<double> u_star_s = u_ref_s;
  std::vector<double> u_star_t = u_ref_t;
  if (!aggregate_exact) {
    AwAggregateProjectionInputs projection_inputs;
    projection_inputs.u_ref_s = u_ref_s.data();
    projection_inputs.u_ref_t = u_ref_t.data();
    projection_inputs.basis_s_r = basis_s_r.data();
    projection_inputs.basis_s_z = basis_s_z.data();
    projection_inputs.basis_t_r = basis_t_r.data();
    projection_inputs.basis_t_z = basis_t_z.data();
    projection_inputs.mu_node = mu_node_region.data();
    projection_inputs.a_node = a_node.data();
    projection_inputs.s_min = s_min.data();
    projection_inputs.s_max = s_max.data();
    projection_inputs.t_min = t_min.data();
    projection_inputs.t_max = t_max.data();
    projection_inputs.axis_mask = compact_axis_mask.data();
    projection_inputs.n_nodes = static_cast<int>(changed_node_count);
    projection_inputs.target_r = predictor_momentum_sum_r;
    projection_inputs.target_z = predictor_momentum_sum_z;
    const auto projection_result =
        tenryu::hydro::corner_distribution::project_aggregate_target_state(
            projection_inputs, u_star_s.data(), u_star_t.data());
    if (projection_result.infeasible) {
      cleanup();
      restore_out_of_mask_velocities();
      return result;
    }
    for (std::size_t compact = 0; compact < changed_node_count; ++compact) {
      const int node = changed_nodes[compact];
      u_ref_r[static_cast<std::size_t>(node)] =
          u_star_s[compact] * basis_s_r[compact] +
          u_star_t[compact] * basis_t_r[compact];
      u_ref_z[static_cast<std::size_t>(node)] =
          u_star_s[compact] * basis_s_z[compact] +
          u_star_t[compact] * basis_t_z[compact];
    }
  }

  double residual_norm_sum = 0.0;
  double residual_norm_max = 0.0;
  for (const int c : changed_cells) {
    const int begin = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int nverts = active_nverts(c);
    const int base = c * state.corner_stride;
    double target_momentum_r = 0.0;
    double target_momentum_z = 0.0;
    for (int k = 0; k < nverts; ++k) {
      const int node =
          mb.cell_node_csr_indices[static_cast<std::size_t>(begin + k)];
      const double a_cp =
          omega_node[static_cast<std::size_t>(node)] *
          mu_corner[static_cast<std::size_t>(base + k)];
      target_momentum_r +=
          a_cp * u_ref_r[static_cast<std::size_t>(node)];
      target_momentum_z +=
          a_cp * u_ref_z[static_cast<std::size_t>(node)];
    }
    const double residual_r = target_momentum_r -
        predictor_cell_momentum_r[static_cast<std::size_t>(c)];
    const double residual_z = target_momentum_z -
        predictor_cell_momentum_z[static_cast<std::size_t>(c)];
    const double residual_norm = std::hypot(residual_r, residual_z);
    residual_norm_sum += residual_norm;
    residual_norm_max = std::max(residual_norm_max, residual_norm);
    remapped_cell_momentum_r[static_cast<std::size_t>(c)] =
        target_momentum_r;
    remapped_cell_momentum_z[static_cast<std::size_t>(c)] =
        target_momentum_z;
  }
  if (txn_vth_sub_diag) {
    std::fprintf(stderr,
                 "[aw_commute] cells=%d nodes=%d aggregate_exact=%d "
                 "sum|R|=%.3e max|R|=%.3e\n",
                 static_cast<int>(changed_cells.size()),
                 static_cast<int>(changed_nodes.size()),
                 aggregate_exact ? 1 : 0,
                 residual_norm_sum,
                 residual_norm_max);
  }
  std::vector<double> corner_pi_r(corner_count, 0.0);
  std::vector<double> corner_pi_z(corner_count, 0.0);
  std::vector<double> corner_momentum_r(corner_count, 0.0);
  std::vector<double> corner_momentum_z(corner_count, 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const int begin = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int nverts = active_nverts(c);
    const int base = c * state.corner_stride;
    for (int k = 0; k < nverts; ++k) {
      const int node =
          mb.cell_node_csr_indices[static_cast<std::size_t>(begin + k)];
      const std::size_t corner = static_cast<std::size_t>(base + k);
      corner_pi_r[corner] =
          mu_corner[corner] * old_node_vr[static_cast<std::size_t>(node)];
      corner_pi_z[corner] =
          mu_corner[corner] * old_node_vz[static_cast<std::size_t>(node)];
      corner_momentum_r[corner] =
          omega_node[static_cast<std::size_t>(node)] * corner_pi_r[corner];
      corner_momentum_z[corner] =
          omega_node[static_cast<std::size_t>(node)] * corner_pi_z[corner];
    }
  }

  for (const int c : changed_cells) {
    const int begin = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int nverts = active_nverts(c);
    const int base = c * state.corner_stride;
    const std::vector<int> stencil = edge_stencil(c);

    double density_min = std::numeric_limits<double>::infinity();
    double density_max = -std::numeric_limits<double>::infinity();
    for (const int stencil_cell : stencil) {
      const int stencil_nverts = active_nverts(stencil_cell);
      const int stencil_base = stencil_cell * state.corner_stride;
      for (int k = 0; k < stencil_nverts; ++k) {
        const double volume =
            old_corner_volume[static_cast<std::size_t>(stencil_base + k)];
        const double mass =
            old_corner_mass[static_cast<std::size_t>(stencil_base + k)];
        if (volume > 0.0 && std::isfinite(volume) &&
            std::isfinite(mass)) {
          const double density = mass / volume;
          density_min = std::min(density_min, density);
          density_max = std::max(density_max, density);
        }
      }
    }
    TENRYU_ASSERT(
        std::isfinite(density_min) && std::isfinite(density_max),
        "CSR corner distribution density bounds must be finite");

    double reference_corner_mass[
        mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    double density_lower[
        mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    double density_upper[
        mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    for (int k = 0; k < nverts; ++k) {
      reference_corner_mass[k] =
          transported_corner_mass[static_cast<std::size_t>(k) *
                                      static_cast<std::size_t>(n_cells) +
                                  static_cast<std::size_t>(c)];
      density_lower[k] = density_min;
      density_upper[k] = density_max;
    }

    tenryu::hydro::corner_distribution::distribute_cell_mass_scaling(
        remapped_cell_mass[static_cast<std::size_t>(c)],
        target_corner_volume.data() + base,
        reference_corner_mass,
        density_lower,
        density_upper,
        nverts,
        distributed_corner_mass.data() + base);
    double corner_momentum_sum_r = 0.0;
    double corner_momentum_sum_z = 0.0;
    double corner_momentum_scale_r = 0.0;
    double corner_momentum_scale_z = 0.0;
    for (int k = 0; k < nverts; ++k) {
      const int node =
          mb.cell_node_csr_indices[static_cast<std::size_t>(begin + k)];
      const std::size_t corner = static_cast<std::size_t>(base + k);
      corner_pi_r[corner] =
          mu_corner[corner] * u_ref_r[static_cast<std::size_t>(node)];
      corner_pi_z[corner] =
          mu_corner[corner] * u_ref_z[static_cast<std::size_t>(node)];
      corner_momentum_r[corner] =
          omega_node[static_cast<std::size_t>(node)] * corner_pi_r[corner];
      corner_momentum_z[corner] =
          omega_node[static_cast<std::size_t>(node)] * corner_pi_z[corner];
      corner_momentum_sum_r += corner_momentum_r[corner];
      corner_momentum_sum_z += corner_momentum_z[corner];
      corner_momentum_scale_r += std::abs(corner_momentum_r[corner]);
      corner_momentum_scale_z += std::abs(corner_momentum_z[corner]);
    }
    const double target_momentum_r =
        remapped_cell_momentum_r[static_cast<std::size_t>(c)];
    const double target_momentum_z =
        remapped_cell_momentum_z[static_cast<std::size_t>(c)];
    const double tolerance_r =
        256.0 * std::numeric_limits<double>::epsilon() *
        (std::abs(target_momentum_r) + corner_momentum_scale_r);
    const double tolerance_z =
        256.0 * std::numeric_limits<double>::epsilon() *
        (std::abs(target_momentum_z) + corner_momentum_scale_z);
    TENRYU_ASSERT(
        std::abs(corner_momentum_sum_r - target_momentum_r) <= tolerance_r &&
            std::abs(corner_momentum_sum_z - target_momentum_z) <= tolerance_z,
        "AW commuting corner pass-through must reproduce target cell momentum");
  }
  if (txn_vth_sub_diag) {
    std::vector<double> distributed_cell_momentum_r(
        static_cast<std::size_t>(n_cells), 0.0);
    std::vector<double> distributed_cell_momentum_z(
        static_cast<std::size_t>(n_cells), 0.0);
    for (int c = 0; c < n_cells; ++c) {
      const int nverts = active_nverts(c);
      const int base = c * state.corner_stride;
      for (int k = 0; k < nverts; ++k) {
        distributed_cell_momentum_r[static_cast<std::size_t>(c)] +=
            corner_momentum_r[static_cast<std::size_t>(base + k)];
        distributed_cell_momentum_z[static_cast<std::size_t>(c)] +=
            corner_momentum_z[static_cast<std::size_t>(base + k)];
      }
    }
    txn_vth_sub_cell("post_distribution",
                     old_node_r,
                     old_node_z,
                     remapped_cell_mass,
                     distributed_cell_momentum_r,
                     distributed_cell_momentum_z);
  }
  std::vector<int> node_corner_offsets(
      static_cast<std::size_t>(n_nodes) + 1U, 0);
  std::vector<int> node_corner_indices;
  for (int node = 0; node < n_nodes; ++node) {
    node_corner_offsets[static_cast<std::size_t>(node)] =
        static_cast<int>(node_corner_indices.size());
    node_corner_indices.insert(
        node_corner_indices.end(),
        node_corners[static_cast<std::size_t>(node)].begin(),
        node_corners[static_cast<std::size_t>(node)].end());
  }
  node_corner_offsets[static_cast<std::size_t>(n_nodes)] =
      static_cast<int>(node_corner_indices.size());

  std::vector<double> recovered_node_mu(static_cast<std::size_t>(n_nodes));
  std::vector<double> recovered_node_mass_rz(
      static_cast<std::size_t>(n_nodes));
  std::vector<double> recovered_node_momentum_rz_r(
      static_cast<std::size_t>(n_nodes));
  std::vector<double> recovered_node_momentum_rz_z(
      static_cast<std::size_t>(n_nodes));
  std::vector<double> recovered_node_velocity_r(
      static_cast<std::size_t>(n_nodes));
  std::vector<double> recovered_node_velocity_z(
      static_cast<std::size_t>(n_nodes));
  tenryu::hydro::corner_distribution::recover_nodal_velocity_aw(
      node_corner_offsets.data(),
      node_corner_indices.data(),
      n_nodes,
      mu_corner.data(),
      corner_pi_r.data(),
      corner_pi_z.data(),
      u_ref_r.data(),
      u_ref_z.data(),
      omega_node.data(),
      node_axis_mask.data(),
      recovered_node_mu.data(),
      recovered_node_velocity_r.data(),
      recovered_node_velocity_z.data(),
      recovered_node_mass_rz.data(),
      recovered_node_momentum_rz_r.data(),
      recovered_node_momentum_rz_z.data());
  txn_vth_sub_node("post_recovery",
                   old_node_r,
                   old_node_z,
                   recovered_node_velocity_r,
                   recovered_node_velocity_z);

  core::DeviceArray<double> d_distribution_velocity_r(
      "ale_remap:conservative_remap_csr:d_distribution_velocity_r");
  core::DeviceArray<double> d_distribution_velocity_z(
      "ale_remap:conservative_remap_csr:d_distribution_velocity_z");
  core::DeviceArray<std::uint8_t> d_changed_node_mask(
      "ale_remap:conservative_remap_csr:d_changed_node_mask");
  d_distribution_velocity_r.reset(static_cast<std::size_t>(n_nodes));
  d_distribution_velocity_z.reset(static_cast<std::size_t>(n_nodes));
  d_changed_node_mask.reset(static_cast<std::size_t>(n_nodes));
  d_distribution_velocity_r.copy_from_host(recovered_node_velocity_r);
  d_distribution_velocity_z.copy_from_host(recovered_node_velocity_z);
  d_changed_node_mask.copy_from_host(changed_node);
  csr_copy_masked_node_velocity_kernel<<<blocks_nodes, 256>>>(
      state.v_r.data(),
      state.v_z.data(),
      d_distribution_velocity_r.data(),
      d_distribution_velocity_z.data(),
      d_changed_node_mask.data(),
      n_nodes);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(core::debug_kernel_sync());

  std::vector<double> aw_paired_corner_mass_packed(
      static_cast<std::size_t>(n_cells) * 4U, 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const int nverts = active_nverts(c);
    const int base = c * state.corner_stride;
    for (int k = 0; k < nverts; ++k) {
      aw_paired_corner_mass_packed[static_cast<std::size_t>(4 * c + k)] =
          aw_paired_corner_mass[static_cast<std::size_t>(base + k)];
    }
  }
  core::DeviceArray<double> d_aw_paired_corner_mass(
      "ale_remap:conservative_remap_csr:d_aw_paired_corner_mass");
  d_aw_paired_corner_mass.reset(aw_paired_corner_mass_packed.size());
  d_aw_paired_corner_mass.copy_from_host(aw_paired_corner_mass_packed);

  CsrOptionBCornerVelocityRemapBuffers optionb_velocity_buffers;
  if (optionb_velocity_authority) {
    TENRYU_ASSERT(state.corner_stride == 4,
                  "CSR Option-B energy coupling requires corner stride 4");
    optionb_velocity_buffers.reset(n_cells, n_nodes);
    optionb_velocity_buffers.corner_mass.copy_from_host(
        aw_paired_corner_mass);
    optionb_velocity_buffers.corner_p_r.copy_from_host(corner_momentum_r);
    optionb_velocity_buffers.corner_p_z.copy_from_host(corner_momentum_z);
    optionb_velocity_buffers.node_mass.copy_from_host(recovered_node_mass_rz);
    optionb_velocity_buffers.node_p_r.copy_from_host(
        recovered_node_momentum_rz_r);
    optionb_velocity_buffers.node_p_z.copy_from_host(
        recovered_node_momentum_rz_z);
    optionb_velocity_buffers.v_r.copy_from_host(recovered_node_velocity_r);
    optionb_velocity_buffers.v_z.copy_from_host(recovered_node_velocity_z);
    optionb_velocity_buffers.cell_mass.copy_from_host(remapped_cell_mass);
    if (optionb_coherent) {
      result.optionb_coherent_corner_mass = distributed_corner_mass;
    }
  }
  CUDA_CHECK(cudaMemcpy(state.rho.data(),
                        d_rho_new,
                        cell_bytes,
                        cudaMemcpyDeviceToDevice));
  if (!total_energy_remap) {
    CUDA_CHECK(cudaMemcpy(state.ee.data(),
                          d_ee_new,
                          cell_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.ei.data(),
                          d_ei_new,
                          cell_bytes,
                          cudaMemcpyDeviceToDevice));
  }
  csr_watch_dump("staged_pre_commit", d_mass_new);
  CUDA_CHECK(cudaMemcpy(state.mass.data(),
                        d_mass_new,
                        cell_bytes,
                        cudaMemcpyDeviceToDevice));
  csr_watch_dump("state_post_commit", state.mass.data());
  CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                        state.x_r_reference.data(),
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                        state.x_z_reference.data(),
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice));

  ale_velcoherence::sample(state, cfg, "s2_post_remap");
  if (optionb_velocity_authority &&
      optionb_ring5_momentum_trace_enabled_for_state(state)) {
    int ring5_cell_start = optionb_macroboundary_ring_start();
    int ring5_cell_end = optionb_macroboundary_ring_end();
    if (ring5_cell_end < ring5_cell_start) {
      std::swap(ring5_cell_start, ring5_cell_end);
    }
    ring5_cell_start = std::max(0, std::min(n_cells - 1, ring5_cell_start));
    ring5_cell_end = std::max(0, std::min(n_cells - 1, ring5_cell_end));
    core::DeviceArray<double> d_state_vr_trace(
        static_cast<std::size_t>(n_nodes));
    core::DeviceArray<double> d_state_vz_trace(
        static_cast<std::size_t>(n_nodes));
    CUDA_CHECK(cudaMemcpy(d_state_vr_trace.data(),
                          state.v_r.data(),
                          node_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_state_vz_trace.data(),
                          state.v_z.data(),
                          node_bytes,
                          cudaMemcpyDeviceToDevice));
    csr_optionb_emit_ring5_momentum_trace(
        state,
        "s6_post_aw_recovery",
        ring5_cell_start,
        ring5_cell_end,
        &optionb_velocity_buffers.corner_mass,
        &optionb_velocity_buffers.corner_p_r,
        &optionb_velocity_buffers.corner_p_z,
        &optionb_velocity_buffers.node_mass,
        nullptr,
        nullptr,
        &d_state_vr_trace,
        &d_state_vz_trace,
        nullptr);
  }
  if (total_energy_remap) {
    if (support_closed_energy_closure) {
      TENRYU_ASSERT(optionb_energy_coupling,
                    "support-closed shell replay requires Option-B total-energy coupling");
      csr_support_closed_redistribute_total_energy_optionb_kernel<<<1, 1>>>(
          d_total_energy_new,
          state.mass.data(),
          d_aw_paired_corner_mass.data(),
          state.v_r.data(),
          state.v_z.data(),
          state.zbar.empty() ? nullptr : state.zbar.data(),
          d_cell_node_offsets,
          d_cell_node_indices,
          d_cell_nverts,
          d_energy_inactive_cell_mask,
          n_cells,
          gamma,
          A,
          cfg.numerics.floors.Te,
          ti_floor_remap,
          d_E_floor_injected,
          d_n_eint_floor_hits);
      CUDA_CHECK(cudaGetLastError());
    } else {
      csr_compute_total_energy_ke_cell_scale_optionb_kernel<<<blocks_cells, 256>>>(
          d_ke_cell_scale,
          d_total_energy_new,
          state.mass.data(),
          d_aw_paired_corner_mass.data(),
          state.v_r.data(),
          state.v_z.data(),
          state.zbar.empty() ? nullptr : state.zbar.data(),
          d_cell_node_offsets,
          d_cell_node_indices,
          d_cell_nverts,
          d_energy_inactive_cell_mask,
          n_cells,
          gamma,
          A,
          cfg.numerics.floors.Te,
          ti_floor_remap);
      CUDA_CHECK(cudaGetLastError());
      if (physical_ke_remap) {
        csr_compute_total_energy_ke_node_scale_physical_ke_kernel<<<blocks_nodes, 256>>>(
            d_ke_node_scale,
            d_ke_cell_scale,
            d_cell_nverts,
            d_energy_inactive_cell_mask,
            n_cells,
            nz,
            n_nodes);
      } else {
        csr_compute_total_energy_ke_node_scale_kernel<<<blocks_nodes, 256>>>(
            d_ke_node_scale,
            d_ke_cell_scale,
            state.mesh.multiblock_reverse_csr_node_offsets.data(),
            state.mesh.multiblock_reverse_csr_node_cells.data(),
            state.mesh.multiblock_reverse_csr_node_corners.data(),
            d_cell_nverts,
            d_energy_inactive_cell_mask,
            n_nodes);
      }
      CUDA_CHECK(cudaGetLastError());
      csr_apply_total_energy_ke_node_scale_kernel<<<blocks_nodes, 256>>>(
          state.v_r.data(),
          state.v_z.data(),
          d_ke_node_scale,
          overrides.active_node_velocity_mask,
          n_nodes);
      CUDA_CHECK(cudaGetLastError());
    }
    csr_recover_internal_from_total_energy_remap_optionb_kernel<<<blocks_cells, 256>>>(
        state.ee.data(),
        state.ei.data(),
        d_total_energy_new,
        d_ye_int_new,
        state.mass.data(),
        d_aw_paired_corner_mass.data(),
        state.v_r.data(),
        state.v_z.data(),
        state.zbar.empty() ? nullptr : state.zbar.data(),
        d_E_floor_injected,
        d_n_eint_floor_hits,
        d_cell_node_offsets,
        d_cell_node_indices,
        d_cell_nverts,
        d_energy_inactive_cell_mask,
        n_cells,
        gamma,
        A,
        cfg.numerics.floors.Te,
        ti_floor_remap);
    CUDA_CHECK(cudaGetLastError());
    if (csr_cons_ledger.active) {
      CUDA_CHECK(cudaDeviceSynchronize());
      csr_cons_ledger.E = csr_cons_audit_capture_state_totals(
          state, cfg, csr_cons_context.reduction);
      csr_cons_ledger.cell_E =
          csr_cons_audit_capture_state_cell_energy(state, cfg);
      csr_cons_ledger.repair = csr_cons_audit_repair_summary(
          state,
          cfg,
          csr_cons_ledger.cell_D,
          gamma,
          A,
          d_aw_paired_corner_mass.data(),
          true,
          false,
          csr_cons_context.reduction);
    }
    csr_remap_energy_audit_emit(
        remap_energy_audit,
        state,
        cfg,
        gamma,
        A,
        d_aw_paired_corner_mass.data(),
        optionb_velocity_authority ? optionb_velocity_buffers.node_mass.data()
                                   : nullptr,
        true,
        false);
    if (near_vacuum_forensics_triggered) {
      csr_capture_near_vacuum_post_kernel<<<1, 1>>>(
          d_near_vacuum_record,
          state.mass.data(),
          state.ee.data(),
          state.ei.data(),
          state.x_r.data(),
          state.x_z.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_cell_node_offsets,
          d_cell_node_indices,
          d_cell_nverts);
      CUDA_CHECK(cudaGetLastError());
    }
  }
  if (optionb_velocity_authority &&
      optionb_ring5_momentum_trace_enabled_for_state(state)) {
    int ring5_cell_start = optionb_macroboundary_ring_start();
    int ring5_cell_end = optionb_macroboundary_ring_end();
    if (ring5_cell_end < ring5_cell_start) {
      std::swap(ring5_cell_start, ring5_cell_end);
    }
    ring5_cell_start = std::max(0, std::min(n_cells - 1, ring5_cell_start));
    ring5_cell_end = std::max(0, std::min(n_cells - 1, ring5_cell_end));
    core::DeviceArray<double> d_state_vr_trace(
        static_cast<std::size_t>(n_nodes));
    core::DeviceArray<double> d_state_vz_trace(
        static_cast<std::size_t>(n_nodes));
    CUDA_CHECK(cudaMemcpy(d_state_vr_trace.data(),
                          state.v_r.data(),
                          node_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_state_vz_trace.data(),
                          state.v_z.data(),
                          node_bytes,
                          cudaMemcpyDeviceToDevice));
    csr_optionb_emit_ring5_momentum_trace(state,
                                          "s7_post_energy_recovery",
                                          ring5_cell_start,
                                          ring5_cell_end,
                                          &optionb_velocity_buffers.corner_mass,
                                          &optionb_velocity_buffers.corner_p_r,
                                          &optionb_velocity_buffers.corner_p_z,
                                          &optionb_velocity_buffers.node_mass,
                                          nullptr,
                                          nullptr,
                                          &d_state_vr_trace,
                                          &d_state_vz_trace,
                                          nullptr);
  }

  tenryu::hydro::ale_diag::emit_mover_post_projection(state, cfg, dt, state.t + dt);

  // 1T runs keep the total internal energy in ee with ei == 0 by convention;
  // flooring ei at cv_i*Ti_floor would fabricate unledgered energy that the
  // next 1T projection folds into ee (a compounding per-remap pump).
  eos_reclosure_ideal_gas_kernel<<<blocks_cells, 256>>>(
      state.Te.data(),
      state.Ti.data(),
      state.Pe.data(),
      state.Pi.data(),
      state.ee.data(),
      state.ei.data(),
      state.rho.data(),
      state.zbar.empty() ? nullptr : state.zbar.data(),
      n_cells,
      nz,
      0,
      support_closed_energy_closure ? d_energy_inactive_cell_mask
                                    : d_inactive_cell_mask,
      cfg.main.two_temperature ? 1 : 0,
      gamma,
      A,
      cfg.numerics.floors.rho,
      cfg.numerics.floors.Te,
      ti_floor_remap);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  if (csr_cons_ledger.active) {
    csr_cons_ledger.F = csr_cons_audit_capture_state_totals(
        state, cfg, csr_cons_context.reduction);
    csr_cons_ledger.cell_F =
        csr_cons_audit_capture_state_cell_energy(state, cfg);
  }

  state.mesh.node_r = state.x_r.data();
  state.mesh.node_z = state.x_z.data();
  state.mesh.recompute_geometry();
  state.vol = state.mesh.cell_vol;
  if (central_macro_remap_mask_active) {
    tenryu::hydro::central_pseudo_core::aggregate_state(
        state, cfg, "post_ale_remap", false);
    pole_angular_derefine::aggregate_state(
        state, cfg, "post_ale_remap", true);
  }
  if (csr_cons_ledger.active) {
    csr_cons_ledger.core_aggregate = csr_cons_audit_capture_state_totals(
        state, cfg, csr_cons_context.reduction);
    csr_cons_ledger.cell_core_aggregate =
        csr_cons_audit_capture_state_cell_energy(state, cfg);
  }
  mesh_trace::trace_cell0_geometry(state, cfg, "csr_remap_post_geometry");
  const bool trace_mesh_motion = tenryu::hydro::mesh_trace::enabled(state, cfg);
  const std::uint64_t subzonal_hash_before =
      trace_mesh_motion
          ? tenryu::hydro::mesh_trace::coordinate_hash(state)
          : 0ULL;
  // The host distribution phase consumed the transported corner masses before
  // any write-back and replaced changed-cell entries in this authoritative
  // AoS ledger. Install that exact ledger once after geometry commit.
  state.corner_mass.copy_from_host(distributed_corner_mass);
  state.corner_mass_initialized = true;
  state.corner_mass_is_lagrangian_invariant =
      tenryu::hydro::corner_mass_lagrangian_invariant_enabled(cfg);
  if (ke_projection_audit || txn_vth_sub_diag) {
    std::vector<double> final_node_velocity_r;
    std::vector<double> final_node_velocity_z;
    state.v_r.copy_to_host(final_node_velocity_r);
    state.v_z.copy_to_host(final_node_velocity_z);
    std::vector<double> aw_kinetic_total(
        static_cast<std::size_t>(n_cells), 0.0);
    double commuting_ke_exchange = 0.0;
    for (int c = 0; c < n_cells; ++c) {
      if (!inactive_cell_mask.empty() &&
          inactive_cell_mask[static_cast<std::size_t>(c)] != 0U) {
        continue;
      }
      const int begin =
          mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
      const int nverts = active_nverts(c);
      const int base = c * state.corner_stride;
      double kinetic = 0.0;
      for (int k = 0; k < nverts; ++k) {
        const int node = mb.cell_node_csr_indices[
            static_cast<std::size_t>(begin + k)];
        const double vr =
            final_node_velocity_r[static_cast<std::size_t>(node)];
        const double vz =
            final_node_velocity_z[static_cast<std::size_t>(node)];
        kinetic += 0.5 *
                   aw_paired_corner_mass[
                       static_cast<std::size_t>(base + k)] *
                   (vr * vr + vz * vz);
      }
      aw_kinetic_total[static_cast<std::size_t>(c)] = kinetic;
      if (changed_cell[static_cast<std::size_t>(c)] != 0U) {
        const double predictor_momentum_r =
            predictor_cell_momentum_r[static_cast<std::size_t>(c)];
        const double predictor_momentum_z =
            predictor_cell_momentum_z[static_cast<std::size_t>(c)];
        const double predictor_kinetic =
            0.5 * (predictor_momentum_r * predictor_momentum_r +
                   predictor_momentum_z * predictor_momentum_z) /
            remapped_cell_mass[static_cast<std::size_t>(c)];
        commuting_ke_exchange += predictor_kinetic - kinetic;
      }
    }
    if (txn_vth_sub_diag) {
      std::fprintf(stderr,
                   "[aw_commute] ke_exchange=%.6e\n",
                   commuting_ke_exchange);
    }
    if (ke_projection_audit) {
      CUDA_CHECK(cudaMemcpy(d_ke_actual,
                            aw_kinetic_total.data(),
                            static_cast<std::size_t>(n_cells) * sizeof(double),
                            cudaMemcpyHostToDevice));
      if (cap_energy_audit) {
        const double K_cons = sum_device_array(d_ke_ext_new, n_cells);
        const double K_act = sum_device_array(d_ke_actual, n_cells);
        result.cap_energy_audit_D_K = K_cons - K_act;
      }
      if (i1b_spurious_sensor) {
        result.i1b_ale_ke_sensor = reduce_ale_ke_projection_sensor(
            d_ke_ext_new, d_ke_actual, d_cell_nverts, n_cells,
            tenryu::hydro::i1b_spurious_sensor_top_k());
      }
    }
  }
  mesh_trace::trace_cell0_geometry(state, cfg, "csr_remap_post_corner_mass");
  if (trace_mesh_motion) {
    const std::uint64_t subzonal_hash_after =
        tenryu::hydro::mesh_trace::coordinate_hash(state);
    tenryu::hydro::mesh_trace::trace_corner_mass_canary(
        state, cfg, subzonal_hash_before, subzonal_hash_after);
  }
  tenryu::hydro::reset_volume_rate_cfl_history_after_ale(state);
  state.holo_ale_invalidated = true;
  state.ale_remaps_applied += 1;
  tenryu::hydro::mesh_trace::trace_post_remap(state, cfg);
  ale_velcoherence::sample(state, cfg, "s3_post_velproj");
  CUDA_CHECK(cudaMemcpy(&result.mass_floor_delta,
                        d_mass_floor_delta,
                        sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&result.E_floor_injected,
                        d_E_floor_injected,
                        sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&result.E_redistribution_unresolved,
                        d_E_redistribution_unresolved,
                        sizeof(double),
                        cudaMemcpyDeviceToHost));
  result.E_eint_floor_deposit = result.E_floor_injected;
  result.E_active_floor = result.E_redistribution_unresolved;
  if (d_n_eint_floor_hits != nullptr) {
    CUDA_CHECK(cudaMemcpy(&result.n_eint_floor_hits,
                          d_n_eint_floor_hits,
                          sizeof(int),
                          cudaMemcpyDeviceToHost));
  }
  if (d_n_active_floor_hits != nullptr) {
    CUDA_CHECK(cudaMemcpy(&result.n_active_floor_hits,
                          d_n_active_floor_hits,
                          sizeof(int),
                          cudaMemcpyDeviceToHost));
  }
  if (central_macro_remap_mask_active) {
    std::vector<double> macro_remap_audit;
    d_central_macro_remap_audit.copy_to_host(macro_remap_audit);
    check_central_macro_remap_flux_audit(state, macro_remap_audit);
  }
  if (near_vacuum_forensics_triggered) {
    CsrNearVacuumRecord record{};
    CUDA_CHECK(cudaMemcpy(&record,
                          d_near_vacuum_record,
                          sizeof(CsrNearVacuumRecord),
                          cudaMemcpyDeviceToHost));
    log_csr_near_vacuum_record(static_cast<long long>(state.step), record);
    near_vacuum_forensics_emitted = true;
  }
  if (total_energy_remap && result.E_floor_injected > 0.0 &&
      std::isfinite(result.E_floor_injected)) {
    core::log_warning(
        std::string("[csr_total_energy_remap_2d_rz] floor energy injected=") +
        std::to_string(result.E_floor_injected) + " erg");
  }
  if (csr_cons_ledger.active) {
    csr_cons_ledger.G = csr_cons_audit_capture_state_totals(
        state, cfg, csr_cons_context.reduction);
    csr_cons_ledger.cell_G =
        csr_cons_audit_capture_state_cell_energy(state, cfg);
    csr_cons_audit_emit(csr_cons_ledger, csr_cons_context);
  }
  csr_watch_dump("state_final", state.mass.data());
  if (!closure_mass_pre.empty()) {
    std::vector<double> closure_mass_post;
    state.mass.copy_to_host(closure_mass_post);
    if (closure_mass_post.size() == closure_mass_pre.size()) {
      long double sum_pre = 0.0L;
      long double sum_post = 0.0L;
      for (const double m : closure_mass_pre) {
        sum_pre += m;
      }
      for (const double m : closure_mass_post) {
        sum_post += m;
      }
      const double d_mass = static_cast<double>(sum_post - sum_pre);
      const double rel =
          d_mass / std::max(1.0e-300,
                            std::abs(static_cast<double>(sum_pre)));
      result.mass_closure_rel = rel;
      note_remap_mass_closure(rel);
      const bool closure_violation = std::abs(rel) > 1.0e-10;
      if ((csr_closure_ledger_enabled || closure_violation) &&
          std::abs(rel) > 1.0e-12 &&
          state.mesh.topo.multiblock.has_value()) {
        const auto& mb_led = *state.mesh.topo.multiblock;
        std::vector<double> ref_r;
        std::vector<double> ref_z;
        std::vector<double> cur_r;
        std::vector<double> cur_z;
        std::vector<double> vol_init_h;
        std::vector<double> vol_h;
        state.x_r_reference.copy_to_host(ref_r);
        state.x_z_reference.copy_to_host(ref_z);
        state.x_r.copy_to_host(cur_r);
        state.x_z.copy_to_host(cur_z);
        state.cell_vol_initial.copy_to_host(vol_init_h);
        state.vol.copy_to_host(vol_h);
        const auto poly_vol = [&](const std::vector<double>& r,
                                  const std::vector<double>& z,
                                  const int cell) {
          const int off =
              mb_led.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
          const int end =
              mb_led.cell_node_csr_offsets[static_cast<std::size_t>(cell) +
                                           1U];
          long double acc = 0.0L;
          int prev = -1;
          int first = -1;
          for (int k = off; k < end; ++k) {
            const int node =
                mb_led.cell_node_csr_indices[static_cast<std::size_t>(k)];
            if (node < 0 || node >= static_cast<int>(r.size())) {
              continue;
            }
            if (first < 0) {
              first = node;
            }
            if (prev >= 0) {
              const long double ra = r[static_cast<std::size_t>(prev)];
              const long double rb = r[static_cast<std::size_t>(node)];
              acc += (ra * ra + ra * rb + rb * rb) *
                     (static_cast<long double>(z[static_cast<std::size_t>(
                          node)]) -
                      static_cast<long double>(z[static_cast<std::size_t>(
                          prev)]));
            }
            prev = node;
          }
          if (prev >= 0 && first >= 0 && prev != first) {
            const long double ra = r[static_cast<std::size_t>(prev)];
            const long double rb = r[static_cast<std::size_t>(first)];
            acc += (ra * ra + ra * rb + rb * rb) *
                   (static_cast<long double>(
                        z[static_cast<std::size_t>(first)]) -
                    static_cast<long double>(
                        z[static_cast<std::size_t>(prev)]));
          }
          constexpr long double kPiOver3 = 1.04719755119659774615L;
          return static_cast<double>(kPiOver3 * acc);
        };
        std::vector<std::pair<double, int>> dm;
        for (std::size_t c = 0; c < closure_mass_post.size(); ++c) {
          const double d = closure_mass_post[c] - closure_mass_pre[c];
          if (d != 0.0) {
            dm.emplace_back(std::abs(d), static_cast<int>(c));
          }
        }
        const std::size_t k_top = std::min<std::size_t>(6, dm.size());
        std::partial_sort(dm.begin(), dm.begin() + k_top, dm.end(),
                          [](const std::pair<double, int>& a,
                             const std::pair<double, int>& b) {
                            return a.first > b.first;
                          });
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(6);
        oss << "[csr_closure_ledger] step=" << state.step
            << " dM=" << d_mass << " dM_rel=" << rel << " offenders=";
        for (std::size_t i = 0; i < k_top; ++i) {
          const int c = dm[i].second;
          oss << (i ? ";" : "") << c << ":dm="
              << (closure_mass_post[static_cast<std::size_t>(c)] -
                  closure_mass_pre[static_cast<std::size_t>(c)])
              << ",v_ref=" << poly_vol(ref_r, ref_z, c)
              << ",v_cur=" << poly_vol(cur_r, cur_z, c)
              << ",vol_init=" << vol_init_h[static_cast<std::size_t>(c)]
              << ",vol=" << vol_h[static_cast<std::size_t>(c)];
        }
        core::log_warning(oss.str());
      }
    }
  }
  result.applied = true;
  conservation_audit::emit_stage(state, "ale_csr_swept_remap_post");
  if (p3_oracle.mode == P3OracleMode::Vnode ||
      p3_oracle.mode == P3OracleMode::Combined) {
    p3_oracle_overwrite_node_velocity_kernel<<<blocks_nodes, 256>>>(
        state.v_r.data(),
        state.v_z.data(),
        state.x_r.data(),
        state.x_z.data(),
        overrides.active_node_velocity_mask,
        n_nodes,
        p3_oracle.H);
    CUDA_CHECK(cudaGetLastError());
  }
  if (p3_oracle.mode == P3OracleMode::Energy ||
      p3_oracle.mode == P3OracleMode::Combined) {
    const double oracle_time = state.t + dt;
    const double one_minus_Ht = 1.0 - p3_oracle.H * oracle_time;
    const double rho_analytic =
        p3_oracle.rho0 /
        (one_minus_Ht * one_minus_Ht * one_minus_Ht);
    const double rho_ratio = rho_analytic / p3_oracle.rho0;
    const double pressure_e =
        p3_oracle.pe0 * std::pow(rho_ratio, p3_oracle.gamma);
    const double pressure_i =
        p3_oracle.pi0 * std::pow(rho_ratio, p3_oracle.gamma);
    const std::uint8_t* const oracle_inactive_cell_mask =
        support_closed_energy_closure ? d_energy_inactive_cell_mask
                                      : d_inactive_cell_mask;
    p3_oracle_overwrite_energy_kernel<<<blocks_cells, 256>>>(
        state.ee.data(),
        state.ei.data(),
        state.rho.data(),
        oracle_inactive_cell_mask,
        n_cells,
        pressure_e,
        pressure_i,
        p3_oracle.gamma);
    CUDA_CHECK(cudaGetLastError());
    eos_reclosure_ideal_gas_kernel<<<blocks_cells, 256>>>(
        state.Te.data(),
        state.Ti.data(),
        state.Pe.data(),
        state.Pi.data(),
        state.ee.data(),
        state.ei.data(),
        state.rho.data(),
        state.zbar.empty() ? nullptr : state.zbar.data(),
        n_cells,
        nz,
        0,
        oracle_inactive_cell_mask,
        cfg.main.two_temperature ? 1 : 0,
        p3_oracle.gamma,
        A,
        cfg.numerics.floors.rho,
        cfg.numerics.floors.Te,
        ti_floor_remap);
    CUDA_CHECK(cudaGetLastError());
  }
  if (p3_oracle.mode == P3OracleMode::Vnode) {
    p3_oracle_log_once(p3_oracle, "remap_exit_vnode");
  } else if (p3_oracle.mode == P3OracleMode::Energy) {
    p3_oracle_log_once(p3_oracle, "remap_exit_eos_reclosure");
  } else if (p3_oracle.mode == P3OracleMode::Combined) {
    p3_oracle_log_once(p3_oracle,
                       "remap_exit_vnode+eos_reclosure");
  }
  if (p3_oracle.mode == P3OracleMode::Vnode ||
      p3_oracle.mode == P3OracleMode::Energy ||
      p3_oracle.mode == P3OracleMode::Combined) {
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  cleanup();
  restore_out_of_mask_velocities();
  return result;
}

}  // namespace

bool csr_cons_audit_env_enabled() {
  static const bool enabled = env_flag_enabled("TENRYU_I1B_CSR_CONS_AUDIT");
  return enabled;
}

void set_csr_cons_audit_context(const CsrConsAuditContext& context) {
  g_csr_cons_audit_context = context;
}

void clear_csr_cons_audit_context() {
  g_csr_cons_audit_context = CsrConsAuditContext{};
}

void CsrOptionBCornerVelocityRemapBuffers::reset(const int n_cells,
                                                 const int n_nodes) {
  TENRYU_ASSERT(n_cells >= 0, "CSR Option B buffer cell count must be non-negative");
  TENRYU_ASSERT(n_nodes >= 0, "CSR Option B buffer node count must be non-negative");
  corner_mass.reset(static_cast<std::size_t>(n_cells) * 4U);
  corner_p_r.reset(static_cast<std::size_t>(n_cells) * 4U);
  corner_p_z.reset(static_cast<std::size_t>(n_cells) * 4U);
  node_mass.reset(static_cast<std::size_t>(n_nodes));
  node_p_r.reset(static_cast<std::size_t>(n_nodes));
  node_p_z.reset(static_cast<std::size_t>(n_nodes));
  v_r.reset(static_cast<std::size_t>(n_nodes));
  v_z.reset(static_cast<std::size_t>(n_nodes));
  cell_mass.reset(static_cast<std::size_t>(n_cells));
}

CsrOptionBFaceColoring csr_optionb_build_internal_face_coloring(
    const tenryu::mesh::MultiBlockTopology& mb,
    const int n_cells) {
  TENRYU_ASSERT(n_cells >= 0,
                "CSR Option B face coloring requires non-negative n_cells");
  CsrOptionBFaceColoring coloring;
  const int n_faces = static_cast<int>(mb.unique_internal_faces.size());
  coloring.face_color.assign(static_cast<std::size_t>(n_faces), -1);
  std::vector<std::vector<int>> cell_colors(static_cast<std::size_t>(n_cells));
  for (int f = 0; f < n_faces; ++f) {
    const auto& face = mb.unique_internal_faces[static_cast<std::size_t>(f)];
    TENRYU_ASSERT(face.cell_a >= 0 && face.cell_a < n_cells &&
                      face.cell_b >= 0 && face.cell_b < n_cells &&
                      face.cell_a != face.cell_b,
                  "CSR Option B face coloring received invalid internal face");
    int color = 0;
    for (;; ++color) {
      const auto& colors_a = cell_colors[static_cast<std::size_t>(face.cell_a)];
      const auto& colors_b = cell_colors[static_cast<std::size_t>(face.cell_b)];
      const bool used_a =
          std::find(colors_a.begin(), colors_a.end(), color) != colors_a.end();
      const bool used_b =
          std::find(colors_b.begin(), colors_b.end(), color) != colors_b.end();
      if (!used_a && !used_b) {
        break;
      }
    }
    coloring.face_color[static_cast<std::size_t>(f)] = color;
    cell_colors[static_cast<std::size_t>(face.cell_a)].push_back(color);
    cell_colors[static_cast<std::size_t>(face.cell_b)].push_back(color);
    coloring.n_colors = std::max(coloring.n_colors, color + 1);
  }
  coloring.pathological = coloring.n_colors > 8;
  return coloring;
}

bool csr_optionb_validate_internal_face_coloring(
    const tenryu::mesh::MultiBlockTopology& mb,
    const int n_cells,
    const CsrOptionBFaceColoring& coloring) {
  const int n_faces = static_cast<int>(mb.unique_internal_faces.size());
  if (n_cells < 0 ||
      coloring.face_color.size() != static_cast<std::size_t>(n_faces)) {
    return false;
  }
  if (n_faces == 0) {
    return coloring.n_colors == 0;
  }
  if (coloring.n_colors <= 0) {
    return false;
  }
  std::vector<int> seen(static_cast<std::size_t>(n_cells) *
                            static_cast<std::size_t>(coloring.n_colors),
                        -1);
  for (int f = 0; f < n_faces; ++f) {
    const auto& face = mb.unique_internal_faces[static_cast<std::size_t>(f)];
    const int color = coloring.face_color[static_cast<std::size_t>(f)];
    if (face.cell_a < 0 || face.cell_a >= n_cells || face.cell_b < 0 ||
        face.cell_b >= n_cells || face.cell_a == face.cell_b || color < 0 ||
        color >= coloring.n_colors) {
      return false;
    }
    const std::size_t idx_a =
        static_cast<std::size_t>(face.cell_a) *
            static_cast<std::size_t>(coloring.n_colors) +
        static_cast<std::size_t>(color);
    const std::size_t idx_b =
        static_cast<std::size_t>(face.cell_b) *
            static_cast<std::size_t>(coloring.n_colors) +
        static_cast<std::size_t>(color);
    if (seen[idx_a] >= 0 || seen[idx_b] >= 0) {
      return false;
    }
    seen[idx_a] = f;
    seen[idx_b] = f;
  }
  return true;
}

bool csr_optionb_corner_velocity_remap_env_enabled() {
  return env_flag_enabled("TENRYU_OPTIONB_CSR_CORNER_VELOCITY_REMAP");
}


bool csr_optionb_velocity_authority_enabled(const core::Config& cfg) {
  static const char* const raw =
      std::getenv("TENRYU_I1B_OPTIONB_VELREMAP");
  static const bool env_present = raw != nullptr && raw[0] != '\0';
  static const bool env_enabled = env_flag_value_from_raw(raw);
  return env_present ? env_enabled
                     : cfg.numerics.ale.csr_optionb_velocity_remap_enabled;
}

bool csr_optionb_coherent_enabled(const core::Config& cfg) {
  static const char* const raw =
      std::getenv("TENRYU_I1B_OPTIONB_COHERENT");
  static const bool env_present = raw != nullptr && raw[0] != '\0';
  static const bool env_enabled = env_flag_value_from_raw(raw);
  return env_present ? env_enabled
                     : cfg.numerics.ale.csr_optionb_coherent_enabled;
}

namespace {

// Projection impulse ledger (rebound-scope verdict Q4): cumulative
// deltaP = (M' − M^tr)·v of the velocity-preserving V-pair rebase — the
// coherent-lite claim's identified weakest point. Monotonic since run
// start; one log line every TENRYU_I1B_PROJ_IMPULSE_EVERY steps
// (default 512) under the coherent env.
struct CoherentImpulseLedger {
  long double net_r = 0.0L;
  long double net_z = 0.0L;
  long double l1 = 0.0L;
  long double e_dp = 0.0L;
  long double abs_p_tr = 0.0L;
  long long installs = 0;
  int last_emit_step = -1;
};

int coherent_impulse_emit_every() {
  static const int every = [] {
    const char* raw = std::getenv("TENRYU_I1B_PROJ_IMPULSE_EVERY");
    const int v = raw != nullptr ? std::atoi(raw) : 0;
    return v > 0 ? v : 512;
  }();
  return every;
}

void emit_coherent_projection_impulse_ledger(
    const core::State& state,
    const core::DeviceArray<double>& d_ledger) {
  static std::unordered_map<const core::State*, CoherentImpulseLedger>
      ledgers;
  std::vector<double> h;
  d_ledger.copy_to_host(h);
  if (h.size() < 5U) {
    return;
  }
  CoherentImpulseLedger& led = ledgers[&state];
  led.net_r += static_cast<long double>(h[0]);
  led.net_z += static_cast<long double>(h[1]);
  led.l1 += static_cast<long double>(h[2]);
  led.e_dp += static_cast<long double>(h[3]);
  led.abs_p_tr += static_cast<long double>(h[4]);
  ++led.installs;
  const int every = coherent_impulse_emit_every();
  if (led.last_emit_step >= 0 &&
      state.step < led.last_emit_step + every) {
    return;
  }
  led.last_emit_step = state.step;
  const double rel =
      led.abs_p_tr > 0.0L
          ? static_cast<double>(led.l1 / led.abs_p_tr)
          : 0.0;
  std::ostringstream oss;
  oss.setf(std::ios::scientific);
  oss.precision(6);
  oss << "[proj_impulse_ledger] step=" << state.step
      << " installs=" << led.installs
      << " net_r=" << static_cast<double>(led.net_r)
      << " net_z=" << static_cast<double>(led.net_z)
      << " L1=" << static_cast<double>(led.l1)
      << " E_dP=" << static_cast<double>(led.e_dp)
      << " rel_L1=" << rel;
  core::log_info(oss.str());
}

}  // namespace

void csr_optionb_canonicalize_corner_mass_basis(core::State& state,
                                                const core::Config& cfg) {
  if ((state.corner_stride != 4 && state.corner_stride != 8) ||
      !csr_optionb_velocity_authority_enabled(cfg)) {
    return;
  }
  // Default OFF: overwriting the hydro's CBSW subzonal corner masses with
  // geometry-recomputed first-moment masses violates Lagrangian invariance
  // and the subzonal-pressure (m,V) equilibrium, injecting spurious forces
  // near the axis (largest where the bases differ most, r->0). The Option-B
  // remap/energy pair carries its own transient first-moment basis (on-the-fly
  // build + transported recover) and does not need state.corner_mass.
  static const bool canonicalize_enabled =
      env_flag_enabled("TENRYU_I1B_OPTIONB_CANONICALIZE");
  if (!canonicalize_enabled) {
    return;
  }
  if (cfg.main.dimension != "2D_RZ" || !state.mesh.topo.multiblock.has_value()) {
    return;
  }
  const int n_cells = state.mesh.topo.n_cells;
  if (n_cells <= 0) {
    return;
  }
  const std::size_t expected_corner_mass_size =
      static_cast<std::size_t>(n_cells) *
      static_cast<std::size_t>(state.corner_stride);
  TENRYU_ASSERT(state.mass.size() == static_cast<std::size_t>(n_cells),
                "Option-B corner-mass canonicalization requires cell mass");
  TENRYU_ASSERT(state.x_r.size() ==
                    static_cast<std::size_t>(state.mesh.topo.n_nodes) &&
                    state.x_z.size() == state.x_r.size(),
                "Option-B corner-mass canonicalization requires node geometry");
  TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U,
                "Option-B corner-mass canonicalization requires CSR offsets");
  TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                    expected_corner_mass_size,
                "Option-B corner-mass canonicalization requires CSR indices");
  if (state.corner_mass.size() != expected_corner_mass_size) {
    state.corner_mass.reset(expected_corner_mass_size);
  }
  const bool use_tri_topology = has_tri_cells(state);
  std::uint8_t* d_cell_nverts =
      upload_cell_nverts_if_needed(state, use_tri_topology);
  core::DeviceArray<std::uint8_t> d_combined_inactive_cell_mask("ale_remap:csr_optionb_canonicalize_corner_mass_basis:d_combined_inactive_cell_mask");
  const std::uint8_t* d_inactive_cell_mask =
      pole_angular_derefine::combined_inactive_mask_device(
          state, cfg, d_combined_inactive_cell_mask);
  const int blocks_cells = (n_cells + 255) / 256;
  csr_optionb_build_first_moment_corner_mass_kernel<<<blocks_cells, 256>>>(
      state.corner_mass.data(),
      state.mass.data(),
      state.x_r.data(),
      state.x_z.data(),
      state.mesh.multiblock_cell_node_csr_offsets.data(),
      state.mesh.multiblock_cell_node_csr_indices.data(),
      d_cell_nverts,
      d_inactive_cell_mask,
      state.corner_stride,
      n_cells);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(core::debug_kernel_sync());
  if (d_cell_nverts != nullptr) {
    CUDA_CHECK(cudaFree(d_cell_nverts));
  }
  state.corner_mass_initialized = true;
  state.corner_mass_is_lagrangian_invariant = false;
}

struct ReplayMomentumCapture {
  bool valid = false;
  double pr = 0.0;
  double pz = 0.0;
  double raw_pi0_r = 0.0;
  double raw_pi0_z = 0.0;
  double raw_pi1_r = 0.0;
  double raw_pi1_z = 0.0;
  double assm_residual_r = 0.0;
  double assm_residual_z = 0.0;
  double rec_residual_r = 0.0;
  double rec_residual_z = 0.0;
};

ReplayMomentumCapture capture_optionb_replay_momentum(
    const core::State& state,
    const CsrOptionBCornerVelocityRemapBuffers& buffers,
    const std::uint8_t* const inactive_cell_mask,
    const std::uint8_t* const assembly_cell_mask,
    const std::uint8_t* const active_node_velocity_mask,
    const double* const source_v_r,
    const double* const source_v_z,
    const int n_cells,
    const int n_nodes) {
  ReplayMomentumCapture out;
  if (n_cells <= 0 || n_nodes <= 0 ||
      !state.mesh.topo.multiblock.has_value() ||
      !state.corner_mass_initialized ||
      state.corner_mass.size() != static_cast<std::size_t>(n_cells) * 4U ||
      buffers.corner_mass.size() != static_cast<std::size_t>(n_cells) * 4U ||
      buffers.corner_p_r.size() != static_cast<std::size_t>(n_cells) * 4U ||
      buffers.corner_p_z.size() != static_cast<std::size_t>(n_cells) * 4U ||
      buffers.node_mass.size() != static_cast<std::size_t>(n_nodes) ||
      buffers.node_p_r.size() != static_cast<std::size_t>(n_nodes) ||
      buffers.node_p_z.size() != static_cast<std::size_t>(n_nodes) ||
      buffers.v_r.size() != static_cast<std::size_t>(n_nodes) ||
      buffers.v_z.size() != static_cast<std::size_t>(n_nodes)) {
    return out;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  if (mb.cell_node_csr_offsets.size() != static_cast<std::size_t>(n_cells + 1) ||
      mb.cell_node_csr_indices.empty()) {
    return out;
  }
  std::vector<double> old_corner_mass;
  std::vector<double> new_corner_mass;
  std::vector<double> new_corner_p_r;
  std::vector<double> new_corner_p_z;
  std::vector<double> node_mass_buf;
  std::vector<double> node_p_r_buf;
  std::vector<double> node_p_z_buf;
  std::vector<double> candidate_v_r;
  std::vector<double> candidate_v_z;
  std::vector<double> source_vr(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> source_vz(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<std::uint8_t> inactive(static_cast<std::size_t>(n_cells), 0U);
  std::vector<std::uint8_t> assembly(static_cast<std::size_t>(n_cells), 0U);
  std::vector<std::uint8_t> active_node(static_cast<std::size_t>(n_nodes), 1U);
  state.corner_mass.copy_to_host(old_corner_mass);
  buffers.corner_mass.copy_to_host(new_corner_mass);
  buffers.corner_p_r.copy_to_host(new_corner_p_r);
  buffers.corner_p_z.copy_to_host(new_corner_p_z);
  buffers.node_mass.copy_to_host(node_mass_buf);
  buffers.node_p_r.copy_to_host(node_p_r_buf);
  buffers.node_p_z.copy_to_host(node_p_z_buf);
  buffers.v_r.copy_to_host(candidate_v_r);
  buffers.v_z.copy_to_host(candidate_v_z);
  CUDA_CHECK(cudaMemcpy(source_vr.data(),
                        source_v_r,
                        sizeof(double) * static_cast<std::size_t>(n_nodes),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(source_vz.data(),
                        source_v_z,
                        sizeof(double) * static_cast<std::size_t>(n_nodes),
                        cudaMemcpyDeviceToHost));
  if (inactive_cell_mask != nullptr) {
    CUDA_CHECK(cudaMemcpy(inactive.data(),
                          inactive_cell_mask,
                          sizeof(std::uint8_t) *
                              static_cast<std::size_t>(n_cells),
                          cudaMemcpyDeviceToHost));
  }
  if (assembly_cell_mask != nullptr) {
    CUDA_CHECK(cudaMemcpy(assembly.data(),
                          assembly_cell_mask,
                          sizeof(std::uint8_t) *
                              static_cast<std::size_t>(n_cells),
                          cudaMemcpyDeviceToHost));
  }
  if (active_node_velocity_mask != nullptr) {
    CUDA_CHECK(cudaMemcpy(active_node.data(),
                          active_node_velocity_mask,
                          sizeof(std::uint8_t) *
                              static_cast<std::size_t>(n_nodes),
                          cudaMemcpyDeviceToHost));
  }
  if (old_corner_mass.size() != static_cast<std::size_t>(n_cells) * 4U ||
      new_corner_mass.size() != static_cast<std::size_t>(n_cells) * 4U ||
      new_corner_p_r.size() != static_cast<std::size_t>(n_cells) * 4U ||
      new_corner_p_z.size() != static_cast<std::size_t>(n_cells) * 4U ||
      node_mass_buf.size() != static_cast<std::size_t>(n_nodes) ||
      node_p_r_buf.size() != static_cast<std::size_t>(n_nodes) ||
      node_p_z_buf.size() != static_cast<std::size_t>(n_nodes) ||
      candidate_v_r.size() != static_cast<std::size_t>(n_nodes) ||
      candidate_v_z.size() != static_cast<std::size_t>(n_nodes)) {
    return out;
  }
  std::vector<double> node_mass(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<std::uint8_t> node_touches_active(
      static_cast<std::size_t>(n_nodes), 0U);
  long double raw_pi0_r = 0.0L;
  long double raw_pi0_z = 0.0L;
  long double raw_pi1_r = 0.0L;
  long double raw_pi1_z = 0.0L;
  long double affected_corner_r = 0.0L;
  long double affected_corner_z = 0.0L;
  for (int c = 0; c < n_cells; ++c) {
    const bool active_cell = inactive[static_cast<std::size_t>(c)] == 0U;
    const bool closure_cell =
        assembly[static_cast<std::size_t>(c)] != 0U;
    const bool use_new_corner = active_cell || closure_cell;
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int end = mb.cell_node_csr_offsets[static_cast<std::size_t>(c + 1)];
    int active_nverts =
        state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)
            ? mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c)
            : mesh::kMeshTopoCellStorageSlots;
    active_nverts = std::min({active_nverts, end - off,
                              mesh::kMeshTopoCellStorageSlots});
    for (int k = 0; k < active_nverts; ++k) {
      const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      if (n < 0 || n >= n_nodes) {
        continue;
      }
      const std::size_t corner_idx =
          static_cast<std::size_t>(c) * 4U + static_cast<std::size_t>(k);
      const int node = n;
      const double old_m = old_corner_mass[corner_idx];
      const double old_pr =
          std::isfinite(old_m) && std::isfinite(source_vr[static_cast<std::size_t>(node)])
              ? old_m * source_vr[static_cast<std::size_t>(node)]
              : 0.0;
      const double old_pz =
          std::isfinite(old_m) && std::isfinite(source_vz[static_cast<std::size_t>(node)])
              ? old_m * source_vz[static_cast<std::size_t>(node)]
              : 0.0;
      const double new_pr = use_new_corner ? new_corner_p_r[corner_idx] : old_pr;
      const double new_pz = use_new_corner ? new_corner_p_z[corner_idx] : old_pz;
      if (std::isfinite(old_pr)) {
        raw_pi0_r += static_cast<long double>(old_pr);
      }
      if (std::isfinite(old_pz)) {
        raw_pi0_z += static_cast<long double>(old_pz);
      }
      if (std::isfinite(new_pr)) {
        raw_pi1_r += static_cast<long double>(new_pr);
      }
      if (std::isfinite(new_pz)) {
        raw_pi1_z += static_cast<long double>(new_pz);
      }
      if (active_node[static_cast<std::size_t>(node)] != 0U) {
        affected_corner_r += static_cast<long double>(
            std::isfinite(new_pr) ? new_pr : 0.0);
        affected_corner_z += static_cast<long double>(
            std::isfinite(new_pz) ? new_pz : 0.0);
      }
      const double m =
          use_new_corner ? new_corner_mass[corner_idx] : old_corner_mass[corner_idx];
      if (std::isfinite(m)) {
        node_mass[static_cast<std::size_t>(n)] += m;
      }
      if (use_new_corner) {
        node_touches_active[static_cast<std::size_t>(n)] = 1U;
      }
    }
  }
  long double affected_node_pr = 0.0L;
  long double affected_node_pz = 0.0L;
  long double affected_rec_r = 0.0L;
  long double affected_rec_z = 0.0L;
  for (int n = 0; n < n_nodes; ++n) {
    const auto idx = static_cast<std::size_t>(n);
    const double m = node_mass[idx];
    const double vr = node_touches_active[idx] != 0U ? candidate_v_r[idx]
                                                     : source_vr[idx];
    const double vz = node_touches_active[idx] != 0U ? candidate_v_z[idx]
                                                     : source_vz[idx];
    if (!std::isfinite(m) || !std::isfinite(vr) || !std::isfinite(vz)) {
      continue;
    }
    if (active_node[idx] != 0U) {
      const double node_pr = node_p_r_buf[idx];
      const double node_pz = node_p_z_buf[idx];
      affected_node_pr += static_cast<long double>(
          std::isfinite(node_pr) ? node_pr : 0.0);
      affected_node_pz += static_cast<long double>(
          std::isfinite(node_pz) ? node_pz : 0.0);
      const double bm = node_mass_buf[idx];
      const double bvr = candidate_v_r[idx];
      const double bvz = candidate_v_z[idx];
      if (std::isfinite(bm) && std::isfinite(bvr) &&
          std::isfinite(node_pr)) {
        affected_rec_r += static_cast<long double>(bm) *
                              static_cast<long double>(bvr) -
                          static_cast<long double>(node_pr);
      }
      if (std::isfinite(bm) && std::isfinite(bvz) &&
          std::isfinite(node_pz)) {
        affected_rec_z += static_cast<long double>(bm) *
                              static_cast<long double>(bvz) -
                          static_cast<long double>(node_pz);
      }
    }
  }
  out.valid = true;
  out.raw_pi0_r = static_cast<double>(raw_pi0_r);
  out.raw_pi0_z = static_cast<double>(raw_pi0_z);
  out.raw_pi1_r = static_cast<double>(raw_pi1_r);
  out.raw_pi1_z = static_cast<double>(raw_pi1_z);
  out.pr = out.raw_pi1_r;
  out.pz = out.raw_pi1_z;
  out.assm_residual_r =
      static_cast<double>(affected_node_pr - affected_corner_r);
  out.assm_residual_z =
      static_cast<double>(affected_node_pz - affected_corner_z);
  out.rec_residual_r = static_cast<double>(affected_rec_r);
  out.rec_residual_z = static_cast<double>(affected_rec_z);
  return out;
}

CsrOptionBCornerVelocityRemapResult
csr_optionb_corner_velocity_remap_component(
    const core::State& state,
    const core::Config& cfg,
    CsrOptionBCornerVelocityRemapBuffers& buffers,
    const bool force_apply,
    const double* const target_cell_mass,
    const double* const source_v_r_override,
    const double* const source_v_z_override,
    const bool disable_limiters_for_audit,
    const bool allow_polar_shell_derefine,
    const bool force_optionb_coherent,
    const bool force_optionb_coherent_transport,
    const bool force_optionb_coherent_rerecover,
    const bool collect_replay_diagnostics,
    const double near_massless_velocity_mass_floor,
    const std::uint8_t* const override_inactive_cell_mask,
    const std::uint8_t* const assembly_cell_mask,
    const std::uint8_t* const active_node_velocity_mask,
    const std::uint8_t* const target_cell_mass_mask,
    const std::uint8_t* const discard_reference_inactive_cell_mask,
    const std::uint8_t* const donor_fallback_cell_mask) {
  CsrOptionBCornerVelocityRemapResult result;
  if (state.corner_stride != 4) {
    return result;
  }
  if (!force_apply && !csr_optionb_corner_velocity_remap_env_enabled()) {
    return result;
  }
  TENRYU_ASSERT(allow_polar_shell_derefine ||
                    !pole_angular_derefine::active(state),
                "polar shell angular de-refine gates out Option-B velocity remap");

  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "CSR Option B velocity remap component requires multiblock topology");
  const auto& mb = *state.mesh.topo.multiblock;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  const RemapDispatchAuditDeviceView remap_dispatch_audit =
      remap_dispatch_audit_device_view();
  TENRYU_ASSERT(n_cells > 0 && n_nodes > 0,
                "CSR Option B velocity remap component requires non-empty mesh");
  TENRYU_ASSERT(state.mass.size() == static_cast<std::size_t>(n_cells) &&
                    state.rho.size() == static_cast<std::size_t>(n_cells) &&
                    state.vol.size() == static_cast<std::size_t>(n_cells) &&
                    state.x_r.size() == static_cast<std::size_t>(n_nodes) &&
                    state.x_z.size() == static_cast<std::size_t>(n_nodes) &&
                    state.x_r_reference.size() == static_cast<std::size_t>(n_nodes) &&
                    state.x_z_reference.size() == static_cast<std::size_t>(n_nodes) &&
                    state.v_r.size() == static_cast<std::size_t>(n_nodes) &&
                    state.v_z.size() == static_cast<std::size_t>(n_nodes),
                "CSR Option B velocity remap component state sizes mismatch");
  TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells + 1),
                "CSR Option B velocity remap requires device cell-node offsets");
  TENRYU_ASSERT(state.mesh.multiblock_reverse_csr_node_offsets.size() ==
                    static_cast<std::size_t>(n_nodes + 1),
                "CSR Option B velocity remap requires reverse CSR offsets");
  TENRYU_ASSERT(state.mesh.multiblock_reverse_csr_node_cells.size() ==
                    state.mesh.multiblock_reverse_csr_node_corners.size(),
                "CSR Option B velocity remap reverse CSR size mismatch");
  TENRYU_ASSERT(mb.cell_orientation_sign.size() ==
                    static_cast<std::size_t>(n_cells),
                "CSR Option B velocity remap requires cell orientation signs");
  const std::uint8_t* central_inactive_cell_mask =
      tenryu::hydro::central_pseudo_core::active(state) &&
              state.central_pseudo_core.d_inactive_member_mask.size() ==
                  static_cast<std::size_t>(n_cells)
          ? state.central_pseudo_core.d_inactive_member_mask.data()
          : nullptr;
  const std::uint8_t* inactive_cell_mask =
      override_inactive_cell_mask != nullptr ? override_inactive_cell_mask
                                             : central_inactive_cell_mask;
  const std::uint8_t* macro_boundary_node_mask =
      tenryu::hydro::central_pseudo_core::active(state) &&
              state.central_pseudo_core.d_boundary_node_mask.size() ==
                  static_cast<std::size_t>(n_nodes)
          ? state.central_pseudo_core.d_boundary_node_mask.data()
          : nullptr;
  // Basis-coherent lite (TENRYU_I1B_OPTIONB_COHERENT, DEFAULT): transport
  // internals stay FIRST-MOMENT; post-transport bookkeeping is coherent via
  // V-pair projection at the new geometry, velocity-preserving nodal rebase,
  // TER pre/post-K basis split, and export-install. Basis-momentum transport
  // (state.corner_mass gather seed + fixed sigma donor shares) is opt-in via
  // _COHERENT_TRANSPORT=1; v=P/M' re-recover via _COHERENT_RERECOVER=1.
  // Both opt-in arms are adjudicated pathological; supersedes _SUBZONAL_BASIS.
  const bool coherent_basis =
      (force_optionb_coherent || csr_optionb_coherent_enabled(cfg)) &&
      state.corner_mass_initialized &&
      state.corner_mass.size() == static_cast<std::size_t>(n_cells) * 4U;
  // TENRYU_I1B_OPTIONB_COHERENT_TRANSPORT=1 opts the TRANSPORT internals
  // into the basis convention (gather seeding + sigma donor shares).
  // DEFAULT OFF: ablation B (2026-06-13) adjudicated the basis transport
  // as the owner of early gas-shell-interface admissibility failures
  // (~1.58 ns; the sigma rigid-shift donor distribution delocalizes
  // momentum removal under strong intra-cell velocity gradients). The
  // production composition ("coherent-lite") keeps first-moment transport
  // and owns the energy gate via the projection/install/TER chain
  // (lite_r1: dE_rel=+2.7e-6 vs frozen +19.1%).
  const bool coherent_transport =
      coherent_basis &&
      (force_optionb_coherent_transport ||
       env_flag_enabled("TENRYU_I1B_OPTIONB_COHERENT_TRANSPORT"));

  const int n_internal_faces =
      static_cast<int>(mb.unique_internal_faces.size());
  const int n_boundary_faces = static_cast<int>(mb.boundary_faces.size());
  const int n_edges = n_internal_faces + n_boundary_faces;
  const std::size_t cell_edge_incidence_count =
      2U * static_cast<std::size_t>(n_internal_faces) +
      static_cast<std::size_t>(n_boundary_faces);
  TENRYU_ASSERT(state.mesh.multiblock_cell_edge_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U,
                "CSR Option B outgoing mass requires cell-edge CSR offsets");
  TENRYU_ASSERT(state.mesh.multiblock_cell_edge_csr_edges.size() ==
                        cell_edge_incidence_count &&
                    state.mesh.multiblock_cell_edge_csr_side.size() ==
                        cell_edge_incidence_count,
                "CSR Option B outgoing mass requires cell-edge CSR entries");
  result.n_internal_faces = n_internal_faces;
  result.total_internal_face_packets =
      static_cast<long long>(n_internal_faces);
  result.total_filter_cells = static_cast<long long>(n_cells);

  const CsrOptionBFaceColoring coloring =
      csr_optionb_build_internal_face_coloring(mb, n_cells);
  TENRYU_ASSERT(csr_optionb_validate_internal_face_coloring(
                    mb, n_cells, coloring),
                "CSR Option B internal face coloring validation failed");
  result.n_face_colors = coloring.n_colors;
  result.pathological_face_coloring = coloring.pathological;

  if (n_internal_faces > 0) {
    TENRYU_ASSERT(mb.d_unique_face_cell_a.size() ==
                      static_cast<std::size_t>(n_internal_faces) &&
                      mb.d_unique_face_cell_b.size() ==
                          static_cast<std::size_t>(n_internal_faces) &&
                      mb.d_unique_face_local_a.size() ==
                          static_cast<std::size_t>(n_internal_faces),
                  "CSR Option B velocity remap requires uploaded unique faces");
  }
  if (n_boundary_faces > 0) {
    TENRYU_ASSERT(mb.d_boundary_face_cell.size() ==
                      static_cast<std::size_t>(n_boundary_faces) &&
                      mb.d_boundary_face_local.size() ==
                          static_cast<std::size_t>(n_boundary_faces),
                  "CSR Option B velocity remap requires uploaded boundary faces");
  }

  buffers.reset(n_cells, n_nodes);
  core::DeviceArray<int> d_cell_orientation_sign("ale_remap:csr_optionb_corner_velocity_remap_component:d_cell_orientation_sign");
  d_cell_orientation_sign.reset(
      static_cast<std::size_t>(n_cells));
  d_cell_orientation_sign.copy_from_host(mb.cell_orientation_sign);

  core::DeviceArray<int> d_face_color("ale_remap:csr_optionb_corner_velocity_remap_component:d_face_color");
  if (n_internal_faces > 0) {
    d_face_color.reset(static_cast<std::size_t>(n_internal_faces));
    d_face_color.copy_from_host(coloring.face_color);
  }

  core::DeviceArray<std::uint8_t> d_cell_nverts_owner("ale_remap:csr_optionb_corner_velocity_remap_component:d_cell_nverts_owner");
  const std::uint8_t* d_cell_nverts = nullptr;
  const bool use_cell_nverts =
      state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells) &&
      std::any_of(state.mesh.cell_nverts.begin(),
                  state.mesh.cell_nverts.end(),
                  [](const std::uint8_t nverts) { return nverts == 3U; });
  if (use_cell_nverts) {
    d_cell_nverts_owner.reset(static_cast<std::size_t>(n_cells));
    d_cell_nverts_owner.copy_from_host(state.mesh.cell_nverts);
    d_cell_nverts = d_cell_nverts_owner.data();
  }

  core::DeviceArray<std::uint8_t> d_node_flags_owner("ale_remap:csr_optionb_corner_velocity_remap_component:d_node_flags_owner");
  const std::uint8_t* d_node_flags = nullptr;
  const bool use_node_flags =
      state.mesh.topo.node_flags.size() == static_cast<std::size_t>(n_nodes) &&
      std::any_of(state.mesh.topo.node_flags.begin(),
                  state.mesh.topo.node_flags.end(),
                  [](const std::uint8_t flags) { return flags != 0U; });
  if (use_node_flags) {
    d_node_flags_owner.reset(static_cast<std::size_t>(n_nodes));
    d_node_flags_owner.copy_from_host(state.mesh.topo.node_flags);
    d_node_flags = d_node_flags_owner.data();
  }

  core::DeviceArray<double> d_outgoing_mass("ale_remap:csr_optionb_corner_velocity_remap_component:d_outgoing_mass");
  d_outgoing_mass.reset(static_cast<std::size_t>(n_cells));
  core::DeviceArray<double> d_outgoing_mass_stage(
      "ale_remap:csr_optionb_corner_velocity_remap_component:d_outgoing_mass_stage");
  d_outgoing_mass_stage.reset(2U * static_cast<std::size_t>(n_edges));
  core::DeviceArray<double> d_mass_flux_scale("ale_remap:csr_optionb_corner_velocity_remap_component:d_mass_flux_scale");
  d_mass_flux_scale.reset(static_cast<std::size_t>(n_cells));
  core::DeviceArray<int> d_diagnostics("ale_remap:csr_optionb_corner_velocity_remap_component:d_diagnostics");
  d_diagnostics.reset(
      static_cast<std::size_t>(kCsrOptionBDiagCount));
  core::DeviceArray<double> d_diagnostics_real("ale_remap:csr_optionb_corner_velocity_remap_component:d_diagnostics_real");
  d_diagnostics_real.reset(
      static_cast<std::size_t>(kCsrOptionBDiagRealCount));
  std::vector<double> diagnostics_real_init(
      static_cast<std::size_t>(kCsrOptionBDiagRealCount), 1.0);
  d_diagnostics_real.copy_from_host(diagnostics_real_init);
  core::DeviceArray<double> d_discard_ledger("ale_remap:csr_optionb_corner_velocity_remap_component:d_discard_ledger");
  if (discard_reference_inactive_cell_mask != nullptr) {
    d_discard_ledger.reset(
        static_cast<std::size_t>(kReplayDiscardLedgerCount));
    CUDA_CHECK(cudaMemset(d_discard_ledger.data(),
                          0,
                          static_cast<std::size_t>(kReplayDiscardLedgerCount) *
                              sizeof(double)));
  }

  core::DeviceArray<int> d_face_adj_offsets_owner("ale_remap:csr_optionb_corner_velocity_remap_component:d_face_adj_offsets_owner");
  core::DeviceArray<int> d_face_adj_indices_owner("ale_remap:csr_optionb_corner_velocity_remap_component:d_face_adj_indices_owner");
  const int* d_face_adj_offsets = nullptr;
  const int* d_face_adj_indices = nullptr;
  const bool use_face_adj =
      mb.face_adj_csr_offsets.size() == static_cast<std::size_t>(n_cells + 1) &&
      !mb.face_adj_csr_indices.empty();
  if (use_face_adj) {
    d_face_adj_offsets_owner.reset(mb.face_adj_csr_offsets.size());
    d_face_adj_offsets_owner.copy_from_host(mb.face_adj_csr_offsets);
    d_face_adj_indices_owner.reset(mb.face_adj_csr_indices.size());
    d_face_adj_indices_owner.copy_from_host(mb.face_adj_csr_indices);
    d_face_adj_offsets = d_face_adj_offsets_owner.data();
    d_face_adj_indices = d_face_adj_indices_owner.data();
  }

  const int blocks_cells = (n_cells + 255) / 256;
  const int blocks_internal = (n_internal_faces + 255) / 256;
  const int* d_cell_node_offsets =
      state.mesh.multiblock_cell_node_csr_offsets.data();
  const int* d_cell_node_indices =
      state.mesh.multiblock_cell_node_csr_indices.data();
  const int* d_cell_edge_offsets =
      state.mesh.multiblock_cell_edge_csr_offsets.data();
  const int* d_cell_edge_edges =
      state.mesh.multiblock_cell_edge_csr_edges.data();
  const std::int8_t* d_cell_edge_side =
      state.mesh.multiblock_cell_edge_csr_side.data();
  const int* d_unique_cell_a =
      n_internal_faces > 0
          ? thrust::raw_pointer_cast(mb.d_unique_face_cell_a.data())
          : nullptr;
  const int* d_unique_cell_b =
      n_internal_faces > 0
          ? thrust::raw_pointer_cast(mb.d_unique_face_cell_b.data())
          : nullptr;
  const int* d_unique_local_a =
      n_internal_faces > 0
          ? thrust::raw_pointer_cast(mb.d_unique_face_local_a.data())
          : nullptr;
  const int* d_boundary_cell =
      n_boundary_faces > 0
          ? thrust::raw_pointer_cast(mb.d_boundary_face_cell.data())
          : nullptr;
  const int* d_boundary_local =
      n_boundary_faces > 0
          ? thrust::raw_pointer_cast(mb.d_boundary_face_local.data())
          : nullptr;
  const double* d_vol_new =
      state.cell_vol_initial.size() == static_cast<std::size_t>(n_cells)
          ? state.cell_vol_initial.data()
          : state.vol.data();
  const bool optionb_diag_enabled =
      env_flag_enabled("TENRYU_I1B_OPTIONB_VELREMAP_DIAG");
  const bool support_closed_replay =
      assembly_cell_mask != nullptr && active_node_velocity_mask != nullptr;
  const double* const source_v_r =
      source_v_r_override != nullptr ? source_v_r_override : state.v_r.data();
  const double* const source_v_z =
      source_v_z_override != nullptr ? source_v_z_override : state.v_z.data();
  const bool ring5_trace_enabled =
      source_v_r_override == nullptr && source_v_z_override == nullptr &&
      !disable_limiters_for_audit &&
      optionb_ring5_momentum_trace_enabled_for_state(state);
  int ring5_cell_start = optionb_macroboundary_ring_start();
  int ring5_cell_end = optionb_macroboundary_ring_end();
  if (ring5_cell_end < ring5_cell_start) {
    std::swap(ring5_cell_start, ring5_cell_end);
  }
  ring5_cell_start = std::max(0, std::min(n_cells - 1, ring5_cell_start));
  ring5_cell_end = std::max(0, std::min(n_cells - 1, ring5_cell_end));
  core::DeviceArray<double> d_ring5_face_p("ale_remap:csr_optionb_corner_velocity_remap_component:d_ring5_face_p");
  core::DeviceArray<double> d_ring5_face_dm("ale_remap:csr_optionb_corner_velocity_remap_component:d_ring5_face_dm");
  core::DeviceArray<double> d_ring5_face_u("ale_remap:csr_optionb_corner_velocity_remap_component:d_ring5_face_u");
  core::DeviceArray<int> d_ring5_face_status("ale_remap:csr_optionb_corner_velocity_remap_component:d_ring5_face_status");
  core::DeviceArray<int> d_ring5_face_centroid_class("ale_remap:csr_optionb_corner_velocity_remap_component:d_ring5_face_centroid_class");
  core::DeviceArray<int> d_ring5_face_cell_a("ale_remap:csr_optionb_corner_velocity_remap_component:d_ring5_face_cell_a");
  core::DeviceArray<int> d_ring5_face_cell_b("ale_remap:csr_optionb_corner_velocity_remap_component:d_ring5_face_cell_b");
  core::DeviceArray<int> d_ring5_face_donor("ale_remap:csr_optionb_corner_velocity_remap_component:d_ring5_face_donor");
  core::DeviceArray<int> d_ring5_face_receiver("ale_remap:csr_optionb_corner_velocity_remap_component:d_ring5_face_receiver");
  if (ring5_trace_enabled && n_internal_faces > 0) {
    d_ring5_face_p.reset(static_cast<std::size_t>(n_internal_faces));
    d_ring5_face_dm.reset(static_cast<std::size_t>(n_internal_faces));
    d_ring5_face_u.reset(static_cast<std::size_t>(n_internal_faces));
    d_ring5_face_status.reset(static_cast<std::size_t>(n_internal_faces));
    d_ring5_face_centroid_class.reset(
        static_cast<std::size_t>(n_internal_faces));
    d_ring5_face_cell_a.reset(static_cast<std::size_t>(n_internal_faces));
    d_ring5_face_cell_b.reset(static_cast<std::size_t>(n_internal_faces));
    d_ring5_face_donor.reset(static_cast<std::size_t>(n_internal_faces));
    d_ring5_face_receiver.reset(static_cast<std::size_t>(n_internal_faces));
  }

  csr_optionb_gather_corner_momentum_kernel<<<blocks_cells, 256>>>(
      buffers.corner_mass.data(),
      buffers.corner_p_r.data(),
      buffers.corner_p_z.data(),
      state.mass.data(),
      state.rho.data(),
      state.vol.data(),
      state.x_r.data(),
      state.x_z.data(),
      source_v_r,
      source_v_z,
      d_cell_node_offsets,
      d_cell_node_indices,
      d_cell_nverts,
      d_node_flags,
      inactive_cell_mask,
      (coherent_transport ||
       (env_flag_enabled("TENRYU_I1B_OPTIONB_SUBZONAL_BASIS") &&
        state.corner_mass_initialized &&
        state.corner_mass.size() == static_cast<std::size_t>(n_cells) * 4U))
          ? state.corner_mass.data()
          : nullptr,
      n_cells);
  CUDA_CHECK(cudaGetLastError());
  if (ring5_trace_enabled) {
    csr_optionb_emit_ring5_momentum_trace(state,
                                          "s0_post_gather",
                                          ring5_cell_start,
                                          ring5_cell_end,
                                          &buffers.corner_mass,
                                          &buffers.corner_p_r,
                                          &buffers.corner_p_z,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr);
  }

  if (n_edges > 0) {
    CUDA_CHECK(cudaMemset(d_outgoing_mass_stage.data(),
                          0,
                          2U * static_cast<std::size_t>(n_edges) *
                              sizeof(double)));
  }
  if (n_internal_faces > 0) {
    csr_accumulate_internal_hydro_outgoing_mass_kernel<<<blocks_internal, 256>>>(
        d_outgoing_mass_stage.data(),
        false,
        state.rho.data(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        state.x_r.data(),
        state.x_z.data(),
        state.x_r_reference.data(),
        state.x_z_reference.data(),
        nullptr,
        nullptr,
        d_cell_node_offsets,
        d_cell_node_indices,
        d_unique_cell_a,
        d_unique_cell_b,
        d_unique_local_a,
        d_cell_orientation_sign.data(),
        d_cell_nverts,
        inactive_cell_mask,
        n_internal_faces,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
  }
  if (n_boundary_faces > 0) {
    const int blocks_boundary = (n_boundary_faces + 255) / 256;
    csr_accumulate_boundary_hydro_outgoing_mass_kernel<<<blocks_boundary, 256>>>(
        d_outgoing_mass_stage.data(),
        false,
        state.rho.data(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        state.x_r.data(),
        state.x_z.data(),
        state.x_r_reference.data(),
        state.x_z_reference.data(),
        nullptr,
        nullptr,
        d_cell_node_offsets,
        d_cell_node_indices,
        d_boundary_cell,
        d_boundary_local,
        d_cell_orientation_sign.data(),
        d_cell_nverts,
        inactive_cell_mask,
        n_internal_faces,
        n_boundary_faces,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
  }
  if (n_edges > 0) {
    csr_gather_face_side_scalar_stage_kernel<<<blocks_cells, 256>>>(
        d_outgoing_mass.data(),
        d_outgoing_mass_stage.data(),
        d_cell_edge_offsets,
        d_cell_edge_edges,
        d_cell_edge_side,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
  }
  csr_compute_hydro_mass_flux_scale_kernel<<<blocks_cells, 256>>>(
      d_mass_flux_scale.data(),
      d_outgoing_mass.data(),
      state.mass.data(),
      state.rho.data(),
      state.vol.data(),
      d_vol_new,
      inactive_cell_mask,
      n_cells,
      cfg.numerics.floors.rho,
      remap_dispatch_audit);
  CUDA_CHECK(cudaGetLastError());

  if (support_closed_replay && n_internal_faces > 0) {
    core::DeviceArray<double> d_face_delta_m("ale_remap:csr_optionb_corner_velocity_remap_component:d_face_delta_m");
    d_face_delta_m.reset(
        static_cast<std::size_t>(n_internal_faces) *
        static_cast<std::size_t>(kCsrOptionBFaceCornerSlots));
    core::DeviceArray<double> d_face_delta_pr("ale_remap:csr_optionb_corner_velocity_remap_component:d_face_delta_pr");
    d_face_delta_pr.reset(
        static_cast<std::size_t>(n_internal_faces) *
        static_cast<std::size_t>(kCsrOptionBFaceCornerSlots));
    core::DeviceArray<double> d_face_delta_pz("ale_remap:csr_optionb_corner_velocity_remap_component:d_face_delta_pz");
    d_face_delta_pz.reset(
        static_cast<std::size_t>(n_internal_faces) *
        static_cast<std::size_t>(kCsrOptionBFaceCornerSlots));
    csr_optionb_compute_internal_face_flux_kernel<<<blocks_internal, 256>>>(
        d_face_delta_m.data(),
        d_face_delta_pr.data(),
        d_face_delta_pz.data(),
        buffers.corner_mass.data(),
        buffers.corner_p_r.data(),
        buffers.corner_p_z.data(),
        state.rho.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.x_r_reference.data(),
        state.x_z_reference.data(),
        source_v_r,
        source_v_z,
        d_cell_node_offsets,
        d_cell_node_indices,
        d_unique_cell_a,
        d_unique_cell_b,
        d_unique_local_a,
        d_cell_orientation_sign.data(),
        d_cell_nverts,
        d_node_flags,
        d_mass_flux_scale.data(),
        n_internal_faces,
        inactive_cell_mask,
        donor_fallback_cell_mask,
        discard_reference_inactive_cell_mask,
        d_discard_ledger.empty() ? nullptr : d_discard_ledger.data(),
        coherent_transport ? state.corner_mass.data() : nullptr,
        d_face_adj_offsets,
        d_face_adj_indices,
        n_cells,
        optionb_diag_enabled,
        ring5_trace_enabled,
        ring5_cell_start,
        ring5_cell_end,
        d_diagnostics.data(),
        d_diagnostics_real.data(),
        ring5_trace_enabled ? d_ring5_face_p.data() : nullptr,
        ring5_trace_enabled ? d_ring5_face_dm.data() : nullptr,
        ring5_trace_enabled ? d_ring5_face_u.data() : nullptr,
        ring5_trace_enabled ? d_ring5_face_status.data() : nullptr,
        ring5_trace_enabled ? d_ring5_face_centroid_class.data() : nullptr,
        ring5_trace_enabled ? d_ring5_face_cell_a.data() : nullptr,
        ring5_trace_enabled ? d_ring5_face_cell_b.data() : nullptr,
        ring5_trace_enabled ? d_ring5_face_donor.data() : nullptr,
        ring5_trace_enabled ? d_ring5_face_receiver.data() : nullptr,
        remap_dispatch_audit);
    CUDA_CHECK(cudaGetLastError());
    csr_optionb_gather_internal_face_fluxes_kernel<<<blocks_cells, 256>>>(
        buffers.corner_mass.data(),
        buffers.corner_p_r.data(),
        buffers.corner_p_z.data(),
        d_face_delta_m.data(),
        d_face_delta_pr.data(),
        d_face_delta_pz.data(),
        d_unique_cell_a,
        d_unique_cell_b,
        d_cell_nverts,
        inactive_cell_mask,
        n_internal_faces,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
  } else {
    for (int color = 0; color < coloring.n_colors; ++color) {
      csr_optionb_apply_internal_packets_color_kernel<<<blocks_internal, 256>>>(
          buffers.corner_mass.data(),
          buffers.corner_p_r.data(),
          buffers.corner_p_z.data(),
          state.rho.data(),
          state.x_r.data(),
          state.x_z.data(),
          state.x_r_reference.data(),
          state.x_z_reference.data(),
          source_v_r,
          source_v_z,
          d_cell_node_offsets,
          d_cell_node_indices,
          d_unique_cell_a,
          d_unique_cell_b,
          d_unique_local_a,
          d_cell_orientation_sign.data(),
          d_cell_nverts,
          d_node_flags,
          d_mass_flux_scale.data(),
          d_face_color.data(),
          color,
          n_internal_faces,
          inactive_cell_mask,
          donor_fallback_cell_mask,
          discard_reference_inactive_cell_mask,
          d_discard_ledger.empty() ? nullptr : d_discard_ledger.data(),
          coherent_transport ? state.corner_mass.data() : nullptr,
          d_face_adj_offsets,
          d_face_adj_indices,
          n_cells,
          optionb_diag_enabled,
          disable_limiters_for_audit,
          ring5_trace_enabled,
          ring5_cell_start,
          ring5_cell_end,
          d_diagnostics.data(),
          d_diagnostics_real.data(),
          ring5_trace_enabled ? d_ring5_face_p.data() : nullptr,
          ring5_trace_enabled ? d_ring5_face_dm.data() : nullptr,
          ring5_trace_enabled ? d_ring5_face_u.data() : nullptr,
          ring5_trace_enabled ? d_ring5_face_status.data() : nullptr,
          ring5_trace_enabled ? d_ring5_face_centroid_class.data() : nullptr,
          ring5_trace_enabled ? d_ring5_face_cell_a.data() : nullptr,
          ring5_trace_enabled ? d_ring5_face_cell_b.data() : nullptr,
          ring5_trace_enabled ? d_ring5_face_donor.data() : nullptr,
          ring5_trace_enabled ? d_ring5_face_receiver.data() : nullptr,
          remap_dispatch_audit);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  if (n_boundary_faces > 0) {
    csr_optionb_apply_boundary_packets_serial_kernel<<<1, 1>>>(
        buffers.corner_mass.data(),
        buffers.corner_p_r.data(),
        buffers.corner_p_z.data(),
        state.rho.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.x_r_reference.data(),
        state.x_z_reference.data(),
        d_cell_node_offsets,
        d_cell_node_indices,
        d_boundary_cell,
        d_boundary_local,
        d_cell_orientation_sign.data(),
        d_cell_nverts,
        d_mass_flux_scale.data(),
        n_boundary_faces,
        inactive_cell_mask,
        d_diagnostics.data(),
        remap_dispatch_audit);
    CUDA_CHECK(cudaGetLastError());
  }
  CsrOptionBRing5FaceSummary ring5_face_summary;
  if (ring5_trace_enabled) {
    ring5_face_summary =
        csr_optionb_ring5_face_summary(&d_ring5_face_p,
                                       &d_ring5_face_dm,
                                       &d_ring5_face_u,
                                       &d_ring5_face_status,
                                       &d_ring5_face_centroid_class,
                                       &d_ring5_face_cell_a,
                                       &d_ring5_face_cell_b,
                                       &d_ring5_face_donor,
                                       &d_ring5_face_receiver);
    csr_optionb_emit_ring5_momentum_trace(state,
                                          "s1_post_packets",
                                          ring5_cell_start,
                                          ring5_cell_end,
                                          &buffers.corner_mass,
                                          &buffers.corner_p_r,
                                          &buffers.corner_p_z,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          &ring5_face_summary);
  }

  csr_optionb_add_floor_mass_kernel<<<blocks_cells, 256>>>(
      buffers.corner_mass.data(),
      buffers.corner_p_r.data(),
      buffers.corner_p_z.data(),
      buffers.cell_mass.data(),
      target_cell_mass,
      state.x_r_reference.data(),
      state.x_z_reference.data(),
      d_vol_new,
      d_cell_node_offsets,
      d_cell_node_indices,
      d_cell_nverts,
      target_cell_mass_mask,
      inactive_cell_mask,
      n_cells,
      cfg.numerics.floors.rho,
      support_closed_replay);
  CUDA_CHECK(cudaGetLastError());
  if (ring5_trace_enabled) {
    csr_optionb_emit_ring5_momentum_trace(state,
                                          "s2_post_floor_mass",
                                          ring5_cell_start,
                                          ring5_cell_end,
                                          &buffers.corner_mass,
                                          &buffers.corner_p_r,
                                          &buffers.corner_p_z,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr);
  }

  if (!support_closed_replay) {
    csr_optionb_hourglass_filter_kernel<<<blocks_cells, 256>>>(
        buffers.corner_mass.data(),
        buffers.corner_p_r.data(),
        buffers.corner_p_z.data(),
        state.x_r_reference.data(),
        state.x_z_reference.data(),
        d_cell_node_offsets,
        d_cell_node_indices,
        d_cell_nverts,
        n_cells,
        inactive_cell_mask,
        d_diagnostics.data());
    CUDA_CHECK(cudaGetLastError());
  }
  if (ring5_trace_enabled) {
    csr_optionb_emit_ring5_momentum_trace(state,
                                          "s3_post_hourglass",
                                          ring5_cell_start,
                                          ring5_cell_end,
                                          &buffers.corner_mass,
                                          &buffers.corner_p_r,
                                          &buffers.corner_p_z,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr);
  }
  if (assembly_cell_mask != nullptr) {
    TENRYU_ASSERT(state.corner_mass_initialized &&
                      state.corner_mass.size() ==
                          static_cast<std::size_t>(n_cells) * 4U,
                  "support-closed Option-B replay requires corner_mass for collar assembly");
    csr_optionb_seed_inactive_closure_corner_momentum_kernel<<<blocks_cells, 256>>>(
        buffers.corner_mass.data(),
        buffers.corner_p_r.data(),
        buffers.corner_p_z.data(),
        state.corner_mass.data(),
        source_v_r,
        source_v_z,
        d_cell_node_offsets,
        d_cell_node_indices,
        d_cell_nverts,
        d_node_flags,
        inactive_cell_mask,
        assembly_cell_mask,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
  }

  const int blocks_nodes = (n_nodes + 255) / 256;
  csr_optionb_scatter_nodal_velocity_kernel<<<blocks_nodes, 256>>>(
      buffers.node_mass.data(),
      buffers.node_p_r.data(),
      buffers.node_p_z.data(),
      buffers.v_r.data(),
      buffers.v_z.data(),
      buffers.corner_mass.data(),
      buffers.corner_p_r.data(),
      buffers.corner_p_z.data(),
      state.mesh.multiblock_reverse_csr_node_offsets.data(),
      state.mesh.multiblock_reverse_csr_node_cells.data(),
      state.mesh.multiblock_reverse_csr_node_corners.data(),
      d_cell_node_offsets,
      d_cell_node_indices,
      d_cell_nverts,
      d_node_flags,
      inactive_cell_mask,
      assembly_cell_mask,
      source_v_r,
      source_v_z,
      n_nodes,
      n_cells);
  CUDA_CHECK(cudaGetLastError());
  if (ring5_trace_enabled) {
    csr_optionb_emit_ring5_momentum_trace(state,
                                          "s4_post_scatter",
                                          ring5_cell_start,
                                          ring5_cell_end,
                                          &buffers.corner_mass,
                                          &buffers.corner_p_r,
                                          &buffers.corner_p_z,
                                          &buffers.node_mass,
                                          &buffers.node_p_r,
                                          &buffers.node_p_z,
                                          &buffers.v_r,
                                          &buffers.v_z,
                                          nullptr);
  }
  if (coherent_basis) {
    // Keep the pre-projection transported ledger for the basis-defect /
    // gap-form instruments (the projection below overwrites corner_mass
    // in place with the V-paired product those audits compare AGAINST).
    buffers.corner_mass_transported.reset(
        static_cast<std::size_t>(n_cells) * 4U);
    CUDA_CHECK(cudaMemcpy(buffers.corner_mass_transported.data(),
                          buffers.corner_mass.data(),
                          static_cast<std::size_t>(n_cells) * 4U *
                              sizeof(double),
                          cudaMemcpyDeviceToDevice));
    csr_optionb_coherent_vpaired_project_kernel<<<blocks_cells, 256>>>(
        buffers.corner_mass.data(),
        buffers.cell_mass.data(),
        state.x_r_reference.data(),
        state.x_z_reference.data(),
        d_cell_node_offsets,
        d_cell_node_indices,
        d_cell_nverts,
        inactive_cell_mask,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
    // Default: velocity-preserving projection. The momentum-conserving
    // re-recover (v' = P/M' at every install) is opt-in via
    // TENRYU_I1B_OPTIONB_COHERENT_RERECOVER=1 — full_r1 (2026-06-12) showed
    // its per-install M/M' velocity ripple wrecks the converging gas-core
    // mesh ~1 ns before physical crushing (admissibility failures from
    // t=1.73 ns, absorption cascade ~1 ns early, "stagnation" at 2.0 ns vs
    // the 1D rail's 2.83 ns).
    const bool momentum_rerecover =
        force_optionb_coherent_rerecover ||
        env_flag_enabled("TENRYU_I1B_OPTIONB_COHERENT_RERECOVER");
    // Projection impulse ledger (verdict Q4): production-shaped calls only
    // (audit/manufactured invocations would pollute the cumulative sums).
    const bool impulse_ledger_active =
        source_v_r_override == nullptr && source_v_z_override == nullptr &&
        !disable_limiters_for_audit;
    core::DeviceArray<double> d_impulse_ledger("ale_remap:csr_optionb_corner_velocity_remap_component:d_impulse_ledger");
    if (impulse_ledger_active) {
      d_impulse_ledger.reset(5U);
    }
    csr_optionb_coherent_rerecover_kernel<<<blocks_nodes, 256>>>(
        buffers.node_mass.data(),
        buffers.node_p_r.data(),
        buffers.node_p_z.data(),
        buffers.v_r.data(),
        buffers.v_z.data(),
        buffers.corner_mass.data(),
        state.mesh.multiblock_reverse_csr_node_offsets.data(),
        state.mesh.multiblock_reverse_csr_node_cells.data(),
        state.mesh.multiblock_reverse_csr_node_corners.data(),
        d_cell_nverts,
        d_node_flags,
        inactive_cell_mask,
        assembly_cell_mask,
        active_node_velocity_mask,
        source_v_r,
        source_v_z,
        momentum_rerecover,
        impulse_ledger_active ? d_impulse_ledger.data() : nullptr,
        near_massless_velocity_mass_floor,
        n_nodes);
    CUDA_CHECK(cudaGetLastError());
    if (impulse_ledger_active) {
      emit_coherent_projection_impulse_ledger(state, d_impulse_ledger);
    }
    if (ring5_trace_enabled) {
      csr_optionb_emit_ring5_momentum_trace(state,
                                            "s4b_post_coherent_project",
                                            ring5_cell_start,
                                            ring5_cell_end,
                                            &buffers.corner_mass,
                                            &buffers.corner_p_r,
                                            &buffers.corner_p_z,
                                            &buffers.node_mass,
                                            &buffers.node_p_r,
                                            &buffers.node_p_z,
                                            &buffers.v_r,
                                            &buffers.v_z,
                                            nullptr);
    }
  }
  {
    static int probe_call_count = 0;
    const char* const probe_env =
        std::getenv("TENRYU_I1B_OPTIONB_PROBE_NODES");
    if (probe_env != nullptr && probe_call_count < 64) {
      ++probe_call_count;
      std::vector<double> h_vr;
      std::vector<double> h_vz;
      std::vector<double> h_mass;
      std::vector<double> h_pz;
      buffers.v_r.copy_to_host(h_vr);
      buffers.v_z.copy_to_host(h_vz);
      buffers.node_mass.copy_to_host(h_mass);
      buffers.node_p_z.copy_to_host(h_pz);
      std::vector<double> h_src_r(static_cast<std::size_t>(n_nodes), 0.0);
      std::vector<double> h_src_z(static_cast<std::size_t>(n_nodes), 0.0);
      CUDA_CHECK(cudaMemcpy(h_src_r.data(),
                            source_v_r,
                            sizeof(double) * static_cast<std::size_t>(n_nodes),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_src_z.data(),
                            source_v_z,
                            sizeof(double) * static_cast<std::size_t>(n_nodes),
                            cudaMemcpyDeviceToHost));
      const std::string spec(probe_env);
      std::size_t pos = 0;
      while (pos <= spec.size()) {
        std::size_t comma = spec.find(',', pos);
        if (comma == std::string::npos) {
          comma = spec.size();
        }
        const int node = std::atoi(spec.substr(pos, comma - pos).c_str());
        if (node >= 0 && node < n_nodes) {
          std::fprintf(stderr,
                       "[optionb_probe] call=%d node=%d src=(%.6e,%.6e) "
                       "post_scatter=(%.6e,%.6e) mass=%.6e p_z=%.6e\n",
                       probe_call_count,
                       node,
                       h_src_r[static_cast<std::size_t>(node)],
                       h_src_z[static_cast<std::size_t>(node)],
                       h_vr[static_cast<std::size_t>(node)],
                       h_vz[static_cast<std::size_t>(node)],
                       h_mass[static_cast<std::size_t>(node)],
                       h_pz[static_cast<std::size_t>(node)]);
        }
        pos = comma + 1;
      }
    }
  }
  if (macro_boundary_node_mask != nullptr) {
    csr_optionb_macroboundary_reconstruct_velocity_kernel<<<blocks_nodes, 256>>>(
        buffers.node_p_r.data(),
        buffers.node_p_z.data(),
        buffers.v_r.data(),
        buffers.v_z.data(),
        buffers.node_mass.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.x_r_reference.data(),
        state.x_z_reference.data(),
        source_v_r,
        source_v_z,
        state.mesh.multiblock_reverse_csr_node_offsets.data(),
        state.mesh.multiblock_reverse_csr_node_cells.data(),
        d_cell_node_offsets,
        d_cell_node_indices,
        d_face_adj_offsets,
        d_face_adj_indices,
        d_cell_nverts,
        d_node_flags,
        inactive_cell_mask,
        macro_boundary_node_mask,
        active_node_velocity_mask,
        n_nodes,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
    if (ring5_trace_enabled) {
      csr_optionb_emit_ring5_momentum_trace(state,
                                            "s5_post_macroboundary_repair",
                                            ring5_cell_start,
                                            ring5_cell_end,
                                            &buffers.corner_mass,
                                            &buffers.corner_p_r,
                                            &buffers.corner_p_z,
                                            &buffers.node_mass,
                                            &buffers.node_p_r,
                                            &buffers.node_p_z,
                                            &buffers.v_r,
                                            &buffers.v_z,
                                            nullptr);
    }
  }

  const bool axis_trace_enabled =
      d_node_flags != nullptr &&
      !env_flag_enabled("TENRYU_I1B_OPTIONB_AXIS_TRACE_DISABLE");
  if (axis_trace_enabled) {
    csr_optionb_axis_trace_velocity_kernel<<<blocks_nodes, 256>>>(
        buffers.node_p_r.data(),
        buffers.node_p_z.data(),
        buffers.v_r.data(),
        buffers.v_z.data(),
        buffers.node_mass.data(),
        state.x_r_reference.data(),
        state.x_z_reference.data(),
        state.mesh.multiblock_reverse_csr_node_offsets.data(),
        state.mesh.multiblock_reverse_csr_node_cells.data(),
        d_cell_node_offsets,
        d_cell_node_indices,
        d_face_adj_offsets,
        d_face_adj_indices,
        d_cell_nverts,
        d_node_flags,
        inactive_cell_mask,
        active_node_velocity_mask,
        n_nodes,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
    if (ring5_trace_enabled) {
      csr_optionb_emit_ring5_momentum_trace(state,
                                            "s5b_post_axis_trace",
                                            ring5_cell_start,
                                            ring5_cell_end,
                                            &buffers.corner_mass,
                                            &buffers.corner_p_r,
                                            &buffers.corner_p_z,
                                            &buffers.node_mass,
                                            &buffers.node_p_r,
                                            &buffers.node_p_z,
                                            &buffers.v_r,
                                            &buffers.v_z,
                                            nullptr);
    }
  }
  if (collect_replay_diagnostics) {
    const ReplayMomentumCapture replay_transport =
        capture_optionb_replay_momentum(state,
                                        buffers,
                                        inactive_cell_mask,
                                        assembly_cell_mask,
                                        active_node_velocity_mask,
                                        source_v_r,
                                        source_v_z,
                                        n_cells,
                                        n_nodes);
    result.replay_transport_momentum_valid = replay_transport.valid;
    result.replay_transport_pr = replay_transport.pr;
    result.replay_transport_pz = replay_transport.pz;
    result.replay_raw_pi0_r = replay_transport.raw_pi0_r;
    result.replay_raw_pi0_z = replay_transport.raw_pi0_z;
    result.replay_raw_pi1_r = replay_transport.raw_pi1_r;
    result.replay_raw_pi1_z = replay_transport.raw_pi1_z;
    result.replay_assm_residual_r = replay_transport.assm_residual_r;
    result.replay_assm_residual_z = replay_transport.assm_residual_z;
    result.replay_rec_residual_r = replay_transport.rec_residual_r;
    result.replay_rec_residual_z = replay_transport.rec_residual_z;
  }

  std::vector<int> diagnostics;
  d_diagnostics.copy_to_host(diagnostics);
  std::vector<double> diagnostics_real;
  d_diagnostics_real.copy_to_host(diagnostics_real);
  if (!d_discard_ledger.empty()) {
    std::vector<double> discard_ledger;
    d_discard_ledger.copy_to_host(discard_ledger);
    if (discard_ledger.size() ==
        static_cast<std::size_t>(kReplayDiscardLedgerCount)) {
      result.replay_discarded_dual_faces = static_cast<int>(
          std::llround(discard_ledger[kReplayDiscardLedgerFaces]));
      result.replay_discarded_dual_mass =
          discard_ledger[kReplayDiscardLedgerMass];
      result.replay_discarded_dual_pi_r =
          discard_ledger[kReplayDiscardLedgerPiR];
      result.replay_discarded_dual_pi_z =
          discard_ledger[kReplayDiscardLedgerPiZ];
    }
  }
  result.fallback_packets = diagnostics[kCsrOptionBDiagFallback];
  result.expanded_stencil_packets = diagnostics[kCsrOptionBDiagExpanded];
  result.expanded_ring1_packets = diagnostics[kCsrOptionBDiagExpandedRing1];
  result.expanded_ring2_packets = diagnostics[kCsrOptionBDiagExpandedRing2];
  result.expanded_failed_packets = diagnostics[kCsrOptionBDiagExpandedFailed];
  result.centroid_out_of_donor_packets = diagnostics[kCsrOptionBDiagCentroidOut];
  result.centroid_on_boundary_packets =
      diagnostics[kCsrOptionBDiagCentroidOnBoundary];
  result.centroid_far_packets = diagnostics[kCsrOptionBDiagCentroidFar];
  result.receiver_vertex_out_of_donor_packets =
      diagnostics[kCsrOptionBDiagReceiverVertexOut];
  result.invalid_input_packets = diagnostics[kCsrOptionBDiagInvalid];
  result.skipped_packets = diagnostics[kCsrOptionBDiagSkipped];
  result.filter_invalid_cells = diagnostics[kCsrOptionBDiagFilterInvalid];
  result.filter_degenerate_cells = diagnostics[kCsrOptionBDiagFilterDegenerate];
  result.alpha_min = diagnostics_real[kCsrOptionBDiagRealAlphaMin];
  result.applied = true;
  return result;
}

namespace ale_velcoherence {

void sample(const core::State& state,
            const core::Config& cfg,
            const char* checkpoint) {
  if (!ale_velcoherence_enabled_for_step(state, cfg)) {
    return;
  }
  if (!mesh::mesh_topo_is_multiblock(cfg.mesh) || state.mesh.dim != 2 ||
      !state.mesh.topo.multiblock.has_value()) {
    return;
  }
  const int n_cells = state.mesh.topo.n_cells;
  if (n_cells <= 0) {
    return;
  }
  TENRYU_ASSERT(state.mass.size() == static_cast<std::size_t>(n_cells),
                "ALE velocity coherence diagnostic requires cell mass");
  TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U,
                "ALE velocity coherence diagnostic requires cell-node CSR offsets");
  TENRYU_ASSERT(!state.mesh.multiblock_cell_node_csr_indices.empty(),
                "ALE velocity coherence diagnostic requires cell-node CSR indices");
  TENRYU_ASSERT(state.x_r.size() == state.v_r.size() &&
                    state.x_z.size() == state.v_z.size() &&
                    state.x_r.size() == state.x_z.size(),
                "ALE velocity coherence diagnostic requires nodal coordinates and velocities");

  const double* gas_tracer =
      (!state.gas_tracer_Y.empty() &&
       state.gas_tracer_initialized &&
       state.gas_tracer_Y.size() == static_cast<std::size_t>(n_cells))
          ? state.gas_tracer_Y.data()
          : nullptr;
  const double R_g_cm = cfg.numerics.diagnostics.hotspot_gas.R_g_cm;
  const int blocks =
      (n_cells + kAleVelCoherenceBlockSize - 1) / kAleVelCoherenceBlockSize;
  core::DeviceArray<double> d_block_sums("ale_remap:sample:d_block_sums");
  d_block_sums.reset(
      static_cast<std::size_t>(blocks) * 4U);
  std::uint8_t* d_cell_nverts =
      upload_cell_nverts_if_needed(state, has_tri_cells(state));
  ale_velcoherence_reduce_kernel<<<blocks, kAleVelCoherenceBlockSize>>>(
      d_block_sums.data(),
      state.mass.data(),
      state.x_r.data(),
      state.x_z.data(),
      state.v_r.data(),
      state.v_z.data(),
      gas_tracer,
      state.mesh.multiblock_cell_node_csr_offsets.data(),
      state.mesh.multiblock_cell_node_csr_indices.data(),
      d_cell_nverts,
      n_cells,
      R_g_cm);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(core::debug_kernel_sync());
  if (d_cell_nverts != nullptr) {
    CUDA_CHECK(cudaFree(d_cell_nverts));
  }

  std::vector<double> block_sums;
  d_block_sums.copy_to_host(block_sums);
  long double M = 0.0L;
  long double Mur = 0.0L;
  long double rad_ke = 0.0L;
  long double tot_ke = 0.0L;
  for (int b = 0; b < blocks; ++b) {
    const std::size_t base = static_cast<std::size_t>(4 * b);
    M += static_cast<long double>(block_sums[base]);
    Mur += static_cast<long double>(block_sums[base + 1U]);
    rad_ke += static_cast<long double>(block_sums[base + 2U]);
    tot_ke += static_cast<long double>(block_sums[base + 3U]);
  }
  const double M_gas = static_cast<double>(M);
  const double mw_ur = (M > 0.0L) ? static_cast<double>(Mur / M) : 0.0;

  std::ostringstream oss;
  oss << "[ale_velcoherence] step=" << state.step
      << " cp=" << (checkpoint != nullptr ? checkpoint : "unknown")
      << " M_gas=" << format_ale_velcoherence_value(M_gas)
      << " mw_ur=" << format_ale_velcoherence_value(mw_ur)
      << " rad_ke=" << format_ale_velcoherence_value(static_cast<double>(rad_ke))
      << " tot_ke=" << format_ale_velcoherence_value(static_cast<double>(tot_ke));
  core::log_info(oss.str());
}

}  // namespace ale_velcoherence

void reset_remap_mass_closure_step_max() {
  g_remap_mass_closure_step_max = 0.0;
}

double remap_mass_closure_step_max() {
  return g_remap_mass_closure_step_max;
}

AleRemap2DRZResult ale_remap_2d_rz(core::State& state,
                                   const core::Config& cfg,
                                   const HydroEOSContext* eos_ctx,
                                   const double dt,
                                   const std::uint8_t* core_freeze_frozen_node_mask,
                                   const AleRemap2DRZOverrides& overrides) {
  AleRemap2DRZResult result;
  std::ostringstream corner_stride_assert_message;
  corner_stride_assert_message
      << "corner_stride must be 4 or 8; stride-8 belt remap is enabled by ALE P4"
      << " [state.corner_stride=" << state.corner_stride
      << ", state.mesh.corner_stride=" << state.mesh.corner_stride
      << ", state.mesh.topo.n_cells=" << state.mesh.topo.n_cells
      << ", state.corner_mass.size()=" << state.corner_mass.size()
      << ", state.mesh.topo.multiblock.has_value()="
      << (state.mesh.topo.multiblock.has_value() ? "true" : "false") << "]";
  TENRYU_ASSERT(
      state.corner_stride == 4 || state.corner_stride == 8,
      corner_stride_assert_message.str());
  if (ale_identity_mode_enabled(cfg)) {
    return result;
  }
  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    // PR4 corner-mass-basis audit + PR5(a) install (both env-gated):
    // capture the pre-remap mesh/mass/corner basis, run the CSR remap,
    // then transport the corner basis over the remap's net effect. The
    // audit env logs the divergence (no state writes); the install env
    // writes the validated PR4 product into state.corner_mass (the
    // basis-contract first connection). Install fires only when the PR4
    // computation ran — the conditions are synchronized by construction.
    const bool corner_mass_audit_requested =
        (corner_mass_remap_audit_enabled() ||
         pr4_corner_mass_install_enabled() ||
         vpaired_corner_mass_install_enabled(cfg));
    const bool corner_mass_audit =
        state.corner_stride == 4 && corner_mass_audit_requested &&
        state.mesh.topo.multiblock.has_value();
    if (state.corner_stride == 8 && corner_mass_audit_requested) {
      core::log_warning(
          "[ale_remap] PR4 corner-mass audit/install skipped for stride-8 belt remap (ALE P4)");
    }
    CornerMassRemapAuditPre corner_mass_pre;
    if (corner_mass_audit) {
      corner_mass_remap_audit_capture_pre(state, &corner_mass_pre);
    }
    const AleRemap2DRZResult csr_result = conservative_remap_csr(
        state, cfg, eos_ctx, dt, core_freeze_frozen_node_mask, overrides);
    if (corner_mass_audit && csr_result.applied) {
      corner_mass_remap_audit_emit(
          state, cfg, corner_mass_pre,
          csr_result.optionb_coherent_corner_mass.empty()
              ? nullptr
              : &csr_result.optionb_coherent_corner_mass);
    }
    return csr_result;
  }
  if (!cfg.numerics.ale.conservative_remap_enabled) {
    return result;
  }
  // 1T convention stores the total internal energy in ee with ei == 0; flooring
  // ei at cv_i*Ti_floor inside the remap fabricates unledgered energy (measured
  // +5e-5/pass via the next step's 1T fold-back). Zero the ion floor in 1T.
  const double ti_floor_remap =
      cfg.main.two_temperature ? cfg.numerics.floors.Ti : 0.0;
  conservation_audit::emit_stage(state, "ale_structured_swept_remap_pre");
  TENRYU_ASSERT(cfg.main.dimension == "2D_RZ",
                "conservative RZ remap requires 2D_RZ geometry");
  TENRYU_ASSERT(cfg.numerics.ale.conservative_remap_target == "reference",
                "only reference conservative remap target is implemented");
  TENRYU_ASSERT(cfg.numerics.ale.conservative_remap_order == "first_order_donor" ||
                    cfg.numerics.ale.conservative_remap_order == "second_order_van_leer",
                "conservative RZ remap order must be first_order_donor or "
                "second_order_van_leer");
  TENRYU_ASSERT(dt >= 0.0, "conservative RZ remap requires non-negative dt");
  TENRYU_ASSERT(state.mesh.dim == 2, "conservative RZ remap requires 2D mesh");
  if (eos_ctx != nullptr && eos_ctx->any_table) {
    TENRYU_ASSERT(overrides.allow_table_eos_closure,
                  "conservative RZ remap with table EOS is transaction-only (§19 v1); "
                  "the per-step conservative remap path remains ideal-gas only");
    TENRYU_ASSERT(eos_ctx->n_materials == 1,
                  "§19 v1: table-EOS conservative remap supports one material");
  }
  const bool table_closure_mode =
      eos_ctx != nullptr && eos_ctx->any_table &&
      overrides.allow_table_eos_closure;
  if (cfg.numerics.materials.per_material_conservation_enabled) {
    TENRYU_ASSERT(false,
                  "conservative RZ remap currently supports cell-mixture "
                  "conservation only");
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = nr * nz;
  const int n_nodes = state.mesh.topo.n_nodes;
  const int corner_mass_convention =
      static_cast<int>(cfg.numerics.hydro.corner_mass_convention);
  if (n_cells <= 0 || n_nodes <= 0) {
    return result;
  }
  TENRYU_ASSERT(state.x_r_reference.size() == state.x_r.size() &&
                    state.x_z_reference.size() == state.x_z.size(),
                "conservative RZ remap requires reference node storage");
  TENRYU_ASSERT(state.cell_vol_initial.size() == state.vol.size(),
                "conservative RZ remap requires reference cell volumes");
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "conservative RZ remap requires a material definition");
  const bool tracking_reference_installed =
      install_tri_fan_tracking_reference_if_enabled(state, cfg).installed;
  const bool seamless_tracking_reference_enabled =
      cfg.numerics.ale.tri_fan_tracking_reference_enabled &&
      cfg.numerics.ale.tri_fan_tracking_reference_mode == "seamless_converging";
  if (!seamless_tracking_reference_enabled) {
    (void)finalize_tri_fan_lagrangian_bulk_reference_if_enabled(
        state, cfg, tracking_reference_installed);
  }

  const int blocks_cells = (n_cells + 255) / 256;
  const int blocks_nodes = (n_nodes + 255) / 256;
  const int blocks_r = (nr + 255) / 256;
  const auto& mat = cfg.materials.materials.front();
  const double gamma = mat.ideal_gas_gamma;
  const double A = mat.A > 0.0 ? mat.A : 1.0;
  const double z_bc = mat.Z > 0.0 ? mat.Z : 1.0;
  const double gm1 = std::max(gamma - 1.0, 1.0e-30);
  const double cv_i =
      core::constants::eV_to_erg /
      (A * core::constants::proton_mass * gm1);
  const double cv_e =
      z_bc * core::constants::eV_to_erg /
      (A * core::constants::proton_mass * gm1);
  const bool second_order_remap =
      cfg.numerics.ale.conservative_remap_order == "second_order_van_leer";
  const bool total_energy_remap = cfg.numerics.hydro.total_energy_remap_2d_rz;
  if (table_closure_mode) {
    TENRYU_ASSERT(!total_energy_remap,
                  "§19 v1: table-EOS conservative remap excludes total-energy modes");
  }
  const bool remap_burn_species =
      cfg.burn.enabled && !total_energy_remap &&
      state.burn_n_host.size() ==
          static_cast<std::size_t>(n_cells) * tenryu::burn::kNumSpecies;
  const bool remap_hot_e_eps =
      !state.hot_e_eps_cum_host.empty() &&
      state.hot_e_eps_cum_host.size() == static_cast<std::size_t>(n_cells);
  const bool remap_burn_eps =
      !state.burn_eps_cum_host.empty() &&
      state.burn_eps_cum_host.size() == static_cast<std::size_t>(n_cells);
  const std::size_t burn_G =
      static_cast<std::size_t>(cfg.burn.diffusion_groups);
  const bool remap_burn_spectrum =
      cfg.burn.enabled && cfg.burn.scheme == "diffusion" &&
      state.burn_Ng.size() ==
          6U * burn_G * static_cast<std::size_t>(n_cells);
  const bool physical_ke_remap =
      total_energy_remap && ale_physical_ke_remap_env_enabled();
  const bool i1b_spurious_sensor =
      tenryu::hydro::i1b_spurious_sensor_enabled();

  double* d_vr_cell = nullptr;
  double* d_vz_cell = nullptr;
  double* d_rho_new = nullptr;
  double* d_ee_new = nullptr;
  double* d_ei_new = nullptr;
  double* d_total_energy_lag = nullptr;
  double* d_total_energy_new = nullptr;
  double* d_ye_int_lag = nullptr;
  double* d_ye_int_new = nullptr;
  double* d_vr_new = nullptr;
  double* d_vz_new = nullptr;
  double* d_mass_new = nullptr;
  double* d_burn_species_Y_lag = nullptr;
  double* d_burn_species_Y_new = nullptr;
  double* d_hot_e_eps_lag = nullptr;
  double* d_hot_e_eps_new = nullptr;
  double* d_burn_eps_lag = nullptr;
  double* d_burn_eps_new = nullptr;
  double* d_burn_rho_lag = nullptr;
  double* d_dm_floor = nullptr;
  double* d_de_floor = nullptr;
  double* d_bm_bot = nullptr;
  double* d_bm_top = nullptr;
  double* d_bpz_bot = nullptr;
  double* d_bpz_top = nullptr;
  double* d_be_bot = nullptr;
  double* d_be_top = nullptr;
  double* d_brad_bot = nullptr;
  double* d_brad_top = nullptr;
  double* d_rad_old = nullptr;
  double* d_rad_new = nullptr;
  double* d_state_supply_donor_avg = nullptr;
  double* d_ke_before_projection = nullptr;
  double* d_ke_after_projection = nullptr;
  double* d_table_snapshot_rho = nullptr;
  double* d_table_snapshot_mass = nullptr;
  double* d_table_snapshot_ee = nullptr;
  double* d_table_snapshot_ei = nullptr;
  double* d_table_snapshot_Te = nullptr;
  double* d_table_snapshot_Ti = nullptr;
  double* d_table_snapshot_Pe = nullptr;
  double* d_table_snapshot_Pi = nullptr;
  double* d_table_snapshot_v_r = nullptr;
  double* d_table_snapshot_v_z = nullptr;
  double* d_table_snapshot_corner_mass = nullptr;
  double* d_table_snapshot_corner_volume = nullptr;
  double* d_table_snapshot_subzonal_mass_corner0 = nullptr;
  double* d_table_snapshot_subzonal_mass_corner1 = nullptr;
  double* d_table_snapshot_subzonal_mass_corner2 = nullptr;
  double* d_table_snapshot_subzonal_mass_corner3 = nullptr;
  double* d_table_snapshot_rad_E = nullptr;
  double* d_table_snapshot_burn_species = nullptr;
  double* d_table_snapshot_hot_e_eps = nullptr;
  double* d_table_snapshot_burn_eps = nullptr;
  double* d_table_snapshot_burn_Ng = nullptr;
  double* d_table_snapshot_burn_Ng_work = nullptr;
  double* d_table_snapshot_Qvisc = nullptr;
  double* d_table_snapshot_zbar = nullptr;
  double* d_table_snapshot_hllc_mom_z_cell = nullptr;
  int* d_table_closure_status = nullptr;
  double* d_table_boundary_max_abs = nullptr;
  std::size_t table_snapshot_burn_Ng_work_size = 0U;
  std::uint8_t* d_cell_nverts = nullptr;
  std::uint8_t* d_node_flags = nullptr;

  auto cleanup = [&]() {
    cudaFree(d_cell_nverts);
    cudaFree(d_node_flags);
  };

	  const bool use_tri_topology = has_tri_cells(state);
	  const bool use_node_flags = has_node_flag_constraints(state);
		  const int button_outer_node_ring =
		      (state.mesh.button_center && state.mesh.button_center->enabled)
		          ? state.mesh.button_center->outer_node_ring
		          : 0;
		  std::vector<double> button_dormant_vol_lag;
		  if (button_outer_node_ring > 0) {
		    button_dormant_vol_lag.resize(static_cast<std::size_t>(n_cells), 0.0);
		    state.vol.copy_to_host(button_dormant_vol_lag.data());
		  }
		  d_cell_nverts = upload_cell_nverts_if_needed(state, use_tri_topology);
		  d_node_flags = upload_node_flags_if_needed(state, use_node_flags);

  d_vr_cell = static_cast<double*>(
      core::device_scratch_acquire(
          "ale_remap:structured:d_vr_cell",
          static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_vz_cell = static_cast<double*>(
      core::device_scratch_acquire(
          "ale_remap:structured:d_vz_cell",
          static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_rho_new = static_cast<double*>(
      core::device_scratch_acquire(
          "ale_remap:structured:d_rho_new",
          static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_ee_new = static_cast<double*>(
      core::device_scratch_acquire(
          "ale_remap:structured:d_ee_new",
          static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_ei_new = static_cast<double*>(
      core::device_scratch_acquire(
          "ale_remap:structured:d_ei_new",
          static_cast<std::size_t>(n_cells) * sizeof(double)));
  if (total_energy_remap) {
    const std::size_t cell_bytes =
        static_cast<std::size_t>(n_cells) * sizeof(double);
    d_total_energy_lag = static_cast<double*>(
        core::device_scratch_acquire(
            "ale_remap:structured:d_total_energy_lag",
            cell_bytes));
    d_total_energy_new = static_cast<double*>(
        core::device_scratch_acquire(
            "ale_remap:structured:d_total_energy_new",
            cell_bytes));
    d_ye_int_lag = static_cast<double*>(
        core::device_scratch_acquire("ale_remap:structured:d_ye_int_lag",
                                     cell_bytes));
    d_ye_int_new = static_cast<double*>(
        core::device_scratch_acquire("ale_remap:structured:d_ye_int_new",
                                     cell_bytes));
  }
  d_vr_new = static_cast<double*>(
      core::device_scratch_acquire(
          "ale_remap:structured:d_vr_new",
          static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_vz_new = static_cast<double*>(
      core::device_scratch_acquire(
          "ale_remap:structured:d_vz_new",
          static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_mass_new = static_cast<double*>(
      core::device_scratch_acquire(
          "ale_remap:structured:d_mass_new",
          static_cast<std::size_t>(n_cells) * sizeof(double)));
  if (remap_burn_species) {
    const std::size_t burn_species_bytes =
        static_cast<std::size_t>(n_cells) * tenryu::burn::kNumSpecies *
        sizeof(double);
    d_burn_species_Y_lag = static_cast<double*>(
        core::device_scratch_acquire("burn:remap:st_Y_lag",
                                     burn_species_bytes));
    d_burn_species_Y_new = static_cast<double*>(
        core::device_scratch_acquire("burn:remap:st_Y_new",
                                     burn_species_bytes));
    CUDA_CHECK(cudaMemcpy(d_burn_species_Y_lag,
                          state.burn_n_host.data(),
                          burn_species_bytes,
                          cudaMemcpyHostToDevice));
  }
  if (remap_hot_e_eps) {
    d_hot_e_eps_lag = static_cast<double*>(core::device_scratch_acquire(
        "ale_remap:legacy:d_hot_e_eps_lag",
        static_cast<std::size_t>(n_cells) * sizeof(double)));
    d_hot_e_eps_new = static_cast<double*>(core::device_scratch_acquire(
        "ale_remap:legacy:d_hot_e_eps_new",
        static_cast<std::size_t>(n_cells) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_hot_e_eps_lag,
                          state.hot_e_eps_cum_host.data(),
                          static_cast<std::size_t>(n_cells) * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hot_e_eps_new,
                          d_hot_e_eps_lag,
                          static_cast<std::size_t>(n_cells) * sizeof(double),
                          cudaMemcpyDeviceToDevice));
  }
  if (remap_burn_eps) {
    d_burn_eps_lag = static_cast<double*>(core::device_scratch_acquire(
        "ale_remap:legacy:d_burn_eps_lag",
        static_cast<std::size_t>(n_cells) * sizeof(double)));
    d_burn_eps_new = static_cast<double*>(core::device_scratch_acquire(
        "ale_remap:legacy:d_burn_eps_new",
        static_cast<std::size_t>(n_cells) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_burn_eps_lag,
                          state.burn_eps_cum_host.data(),
                          static_cast<std::size_t>(n_cells) * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_burn_eps_new,
                          d_burn_eps_lag,
                          static_cast<std::size_t>(n_cells) * sizeof(double),
                          cudaMemcpyDeviceToDevice));
  }
  if (remap_burn_spectrum) {
    const std::size_t cell_bytes =
        static_cast<std::size_t>(n_cells) * sizeof(double);
    d_burn_rho_lag = static_cast<double*>(
        core::device_scratch_acquire("burn:remap:rho_lag", cell_bytes));
    CUDA_CHECK(cudaMemcpy(d_burn_rho_lag,
                          state.rho.data(),
                          cell_bytes,
                          cudaMemcpyDeviceToDevice));
  }
  if (i1b_spurious_sensor) {
    const std::size_t cell_bytes =
        static_cast<std::size_t>(n_cells) * sizeof(double);
    d_ke_before_projection = static_cast<double*>(
        core::device_scratch_acquire(
            "ale_remap:structured:d_ke_before_projection",
            cell_bytes));
    d_ke_after_projection = static_cast<double*>(
        core::device_scratch_acquire(
            "ale_remap:structured:d_ke_after_projection",
            cell_bytes));
  }
  d_dm_floor = static_cast<double*>(
      core::device_scratch_acquire("ale_remap:structured:d_dm_floor",
                                   sizeof(double)));
  d_de_floor = static_cast<double*>(
      core::device_scratch_acquire("ale_remap:structured:d_de_floor",
                                   sizeof(double)));
  d_bm_bot = static_cast<double*>(
      core::device_scratch_acquire(
          "ale_remap:structured:d_bm_bot",
          static_cast<std::size_t>(nr) * sizeof(double)));
  d_bm_top = static_cast<double*>(
      core::device_scratch_acquire(
          "ale_remap:structured:d_bm_top",
          static_cast<std::size_t>(nr) * sizeof(double)));
  d_bpz_bot = static_cast<double*>(
      core::device_scratch_acquire(
          "ale_remap:structured:d_bpz_bot",
          static_cast<std::size_t>(nr) * sizeof(double)));
  d_bpz_top = static_cast<double*>(
      core::device_scratch_acquire(
          "ale_remap:structured:d_bpz_top",
          static_cast<std::size_t>(nr) * sizeof(double)));
  d_be_bot = static_cast<double*>(
      core::device_scratch_acquire(
          "ale_remap:structured:d_be_bot",
          static_cast<std::size_t>(nr) * sizeof(double)));
  d_be_top = static_cast<double*>(
      core::device_scratch_acquire(
          "ale_remap:structured:d_be_top",
          static_cast<std::size_t>(nr) * sizeof(double)));
  d_brad_bot = static_cast<double*>(
      core::device_scratch_acquire(
          "ale_remap:structured:d_brad_bot",
          static_cast<std::size_t>(nr) * sizeof(double)));
  d_brad_top = static_cast<double*>(
      core::device_scratch_acquire(
          "ale_remap:structured:d_brad_top",
          static_cast<std::size_t>(nr) * sizeof(double)));
  memset_zero(d_dm_floor, 1);
  memset_zero(d_de_floor, 1);
  memset_zero(d_bm_bot, static_cast<std::size_t>(nr));
  memset_zero(d_bm_top, static_cast<std::size_t>(nr));
  memset_zero(d_bpz_bot, static_cast<std::size_t>(nr));
  memset_zero(d_bpz_top, static_cast<std::size_t>(nr));
  memset_zero(d_be_bot, static_cast<std::size_t>(nr));
  memset_zero(d_be_top, static_cast<std::size_t>(nr));
  memset_zero(d_brad_bot, static_cast<std::size_t>(nr));
  memset_zero(d_brad_top, static_cast<std::size_t>(nr));

  if (table_closure_mode) {
    d_table_snapshot_rho = snapshot_device_array(
        "ale_remap:structured:table_snapshot_rho",
        state.rho.data(),
        state.rho.size());
    d_table_snapshot_mass = snapshot_device_array(
        "ale_remap:structured:table_snapshot_mass",
        state.mass.data(),
        state.mass.size());
    d_table_snapshot_ee = snapshot_device_array(
        "ale_remap:structured:table_snapshot_ee",
        state.ee.data(),
        state.ee.size());
    d_table_snapshot_ei = snapshot_device_array(
        "ale_remap:structured:table_snapshot_ei",
        state.ei.data(),
        state.ei.size());
    d_table_snapshot_Te = snapshot_device_array(
        "ale_remap:structured:table_snapshot_Te",
        state.Te.data(),
        state.Te.size());
    d_table_snapshot_Ti = snapshot_device_array(
        "ale_remap:structured:table_snapshot_Ti",
        state.Ti.data(),
        state.Ti.size());
    d_table_snapshot_Pe = snapshot_device_array(
        "ale_remap:structured:table_snapshot_Pe",
        state.Pe.data(),
        state.Pe.size());
    d_table_snapshot_Pi = snapshot_device_array(
        "ale_remap:structured:table_snapshot_Pi",
        state.Pi.data(),
        state.Pi.size());
    d_table_snapshot_v_r = snapshot_device_array(
        "ale_remap:structured:table_snapshot_v_r",
        state.v_r.data(),
        state.v_r.size());
    d_table_snapshot_v_z = snapshot_device_array(
        "ale_remap:structured:table_snapshot_v_z",
        state.v_z.data(),
        state.v_z.size());
    d_table_snapshot_corner_mass = snapshot_device_array(
        "ale_remap:structured:table_snapshot_corner_mass",
        state.corner_mass.data(),
        state.corner_mass.size());
    d_table_snapshot_corner_volume = snapshot_device_array(
        "ale_remap:structured:table_snapshot_corner_volume",
        state.corner_volume.data(),
        state.corner_volume.size());
    d_table_snapshot_subzonal_mass_corner0 = snapshot_device_array(
        "ale_remap:structured:table_snapshot_subzonal_mass_corner0",
        state.subzonal_mass_corner0.data(),
        state.subzonal_mass_corner0.size());
    d_table_snapshot_subzonal_mass_corner1 = snapshot_device_array(
        "ale_remap:structured:table_snapshot_subzonal_mass_corner1",
        state.subzonal_mass_corner1.data(),
        state.subzonal_mass_corner1.size());
    d_table_snapshot_subzonal_mass_corner2 = snapshot_device_array(
        "ale_remap:structured:table_snapshot_subzonal_mass_corner2",
        state.subzonal_mass_corner2.data(),
        state.subzonal_mass_corner2.size());
    d_table_snapshot_subzonal_mass_corner3 = snapshot_device_array(
        "ale_remap:structured:table_snapshot_subzonal_mass_corner3",
        state.subzonal_mass_corner3.data(),
        state.subzonal_mass_corner3.size());
    d_table_snapshot_rad_E = snapshot_device_array(
        "ale_remap:structured:table_snapshot_rad_E",
        state.rad_E.data(),
        state.rad_E.size());
    if (remap_burn_species) {
      d_table_snapshot_burn_species = snapshot_device_array(
          "ale_remap:structured:table_snapshot_burn_species",
          d_burn_species_Y_lag,
          state.burn_n_host.size());
    }
    if (remap_hot_e_eps) {
      d_table_snapshot_hot_e_eps = snapshot_device_array(
          "ale_remap:structured:table_snapshot_hot_e_eps",
          d_hot_e_eps_lag,
          state.hot_e_eps_cum_host.size());
    }
    if (remap_burn_eps) {
      d_table_snapshot_burn_eps = snapshot_device_array(
          "ale_remap:structured:table_snapshot_burn_eps",
          d_burn_eps_lag,
          state.burn_eps_cum_host.size());
    }
    if (remap_burn_spectrum) {
      d_table_snapshot_burn_Ng = snapshot_device_array(
          "ale_remap:structured:table_snapshot_burn_Ng",
          state.burn_Ng.data(),
          state.burn_Ng.size());
      table_snapshot_burn_Ng_work_size = state.burn_Ng_work.size();
      d_table_snapshot_burn_Ng_work = snapshot_device_array(
          "ale_remap:structured:table_snapshot_burn_Ng_work",
          state.burn_Ng_work.data(),
          table_snapshot_burn_Ng_work_size);
    }
    if (button_outer_node_ring > 0) {
      d_table_snapshot_Qvisc = snapshot_device_array(
          "ale_remap:structured:table_snapshot_Qvisc",
          state.Qvisc.data(),
          state.Qvisc.size());
      d_table_snapshot_zbar = snapshot_device_array(
          "ale_remap:structured:table_snapshot_zbar",
          state.zbar.data(),
          state.zbar.size());
      d_table_snapshot_hllc_mom_z_cell = snapshot_device_array(
          "ale_remap:structured:table_snapshot_hllc_mom_z_cell",
          state.hllc_mom_z_cell.data(),
          state.hllc_mom_z_cell.size());
    }
    d_table_closure_status = static_cast<int*>(core::device_scratch_acquire(
        "ale_remap:structured:table_closure_status",
        static_cast<std::size_t>(n_cells) * sizeof(int)));
    d_table_boundary_max_abs = static_cast<double*>(
        core::device_scratch_acquire(
            "ale_remap:structured:table_boundary_max_abs",
            sizeof(double)));
    memset_zero_int(d_table_closure_status,
                    static_cast<std::size_t>(n_cells));
    memset_zero(d_table_boundary_max_abs, 1U);
  }

	  compute_cell_velocity_from_nodes_kernel<<<blocks_cells, 256>>>(
	      d_vr_cell,
	      d_vz_cell,
      state.v_r.data(),
      state.v_z.data(),
      d_cell_nverts,
	      nr,
	      nz);
	  CUDA_CHECK(cudaGetLastError());
	  if (button_outer_node_ring > 0) {
	    apply_button_center_cell_velocity_kernel<<<blocks_cells, 256>>>(
	        d_vr_cell,
	        d_vz_cell,
	        state.v_r.data(),
	        state.v_z.data(),
	        nr,
	        nz,
	        button_outer_node_ring);
	    CUDA_CHECK(cudaGetLastError());
		  }
		  if (total_energy_remap) {
		    if (physical_ke_remap) {
		      build_total_energy_remap_state_physical_ke_kernel<<<blocks_cells, 256>>>(
		          d_total_energy_lag,
		          d_ye_int_lag,
		          state.mass.data(),
		          state.ee.data(),
		          state.ei.data(),
		          state.x_r.data(),
		          state.x_z.data(),
		          state.v_r.data(),
		          state.v_z.data(),
		          d_cell_nverts,
		          rz::corner_mass_fallback_device_recorder(),
		          corner_mass_convention,
		          nr,
		          nz,
		          button_outer_node_ring);
		    } else {
		      build_total_energy_remap_state_kernel<<<blocks_cells, 256>>>(
		          d_total_energy_lag,
	          d_ye_int_lag,
		          state.mass.data(),
		          state.ee.data(),
		          state.ei.data(),
		          state.x_r.data(),
		          state.x_z.data(),
		          state.v_r.data(),
		          state.v_z.data(),
		          nr,
		          nz,
		          button_outer_node_ring);
		    }
	    CUDA_CHECK(cudaGetLastError());
	  }

  const auto& bc = cfg.numerics.hydro.boundary_2d;
  const bool bottom_active = bc.z_bottom_cfg.supply_active(state.t);
  const bool top_active = bc.z_top_cfg.supply_active(state.t);
  const bool use_state_supply_radial_avg =
      bc.state_supply_donor_mode == "interior_radial_average";
  const double e_bottom =
      (cv_i + cv_e) * std::max(bc.z_bottom_cfg.supply_T_eV, 0.0);
  const double e_top =
      (cv_i + cv_e) * std::max(bc.z_top_cfg.supply_T_eV, 0.0);
  if (use_state_supply_radial_avg && (bottom_active || top_active)) {
    d_state_supply_donor_avg = static_cast<double*>(
        core::device_scratch_acquire(
            "ale_remap:structured:d_state_supply_donor_avg",
            10U * sizeof(double)));
    memset_zero(d_state_supply_donor_avg, 10U);
    state_supply_boundary_donor_radial_avg_kernel<<<1, 256>>>(
        state.rho.data(),
        nullptr,
        d_vz_cell,
        state.ee.data(),
        state.ei.data(),
        nullptr,
        d_state_supply_donor_avg + 0,
        d_state_supply_donor_avg + 1,
        d_state_supply_donor_avg + 2,
        d_state_supply_donor_avg + 3,
        d_state_supply_donor_avg + 4,
        d_state_supply_donor_avg + 5,
        d_state_supply_donor_avg + 6,
        d_state_supply_donor_avg + 7,
        d_state_supply_donor_avg + 8,
        d_state_supply_donor_avg + 9,
        nr,
        nz);
    CUDA_CHECK(cudaGetLastError());
  }

  state_supply_boundary_flux_2d_rz_kernel<<<blocks_r, 256>>>(
      state.x_r_reference.data(),
      state.rho.data(),
      d_vz_cell,
      state.ee.data(),
      state.ei.data(),
      nullptr,
      d_bm_bot,
      d_bm_top,
      d_bpz_bot,
      d_bpz_top,
      d_be_bot,
      d_be_top,
      nullptr,
      nullptr,
      nr,
      nz,
      dt,
      bottom_active ? 1 : 0,
      top_active ? 1 : 0,
      bc.z_bottom_cfg.supply_rho_g_per_cc,
      bc.z_bottom_cfg.supply_u_z_cm_per_s,
      e_bottom,
      0.0,
      bc.z_top_cfg.supply_rho_g_per_cc,
      bc.z_top_cfg.supply_u_z_cm_per_s,
      e_top,
      0.0,
      use_state_supply_radial_avg ? 1 : 0,
      d_state_supply_donor_avg != nullptr ? d_state_supply_donor_avg + 0 : nullptr,
      d_state_supply_donor_avg != nullptr ? d_state_supply_donor_avg + 2 : nullptr,
      d_state_supply_donor_avg != nullptr ? d_state_supply_donor_avg + 3 : nullptr,
      d_state_supply_donor_avg != nullptr ? d_state_supply_donor_avg + 4 : nullptr,
      d_state_supply_donor_avg != nullptr ? d_state_supply_donor_avg + 5 : nullptr,
      d_state_supply_donor_avg != nullptr ? d_state_supply_donor_avg + 7 : nullptr,
      d_state_supply_donor_avg != nullptr ? d_state_supply_donor_avg + 8 : nullptr,
      d_state_supply_donor_avg != nullptr ? d_state_supply_donor_avg + 9 : nullptr);
  CUDA_CHECK(cudaGetLastError());
  if (table_closure_mode) {
    accumulate_device_array_max_abs(d_bm_bot, nr, d_table_boundary_max_abs);
    accumulate_device_array_max_abs(d_bm_top, nr, d_table_boundary_max_abs);
    accumulate_device_array_max_abs(d_bpz_bot, nr, d_table_boundary_max_abs);
    accumulate_device_array_max_abs(d_bpz_top, nr, d_table_boundary_max_abs);
    accumulate_device_array_max_abs(d_be_bot, nr, d_table_boundary_max_abs);
    accumulate_device_array_max_abs(d_be_top, nr, d_table_boundary_max_abs);
  }
  if (total_energy_remap) {
    promote_boundary_energy_flux_to_total_kernel<<<blocks_r, 256>>>(
        d_be_bot, d_be_top, d_bm_bot, d_bm_top, d_bpz_bot, d_bpz_top, nr);
    CUDA_CHECK(cudaGetLastError());
  }

  if (total_energy_remap && second_order_remap) {
    ale_remap_2d_rz_total_energy_second_order_kernel<<<blocks_cells, 256>>>(
        state.rho.data(),
        d_vr_cell,
        d_vz_cell,
	        d_total_energy_lag,
	        d_ye_int_lag,
	        state.vol.data(),
	        state.mass.data(),
	        state.x_r.data(),
	        state.x_z.data(),
	        state.x_r_reference.data(),
        state.x_z_reference.data(),
        state.cell_vol_initial.data(),
        d_bm_bot,
        d_bm_top,
        d_bpz_bot,
        d_bpz_top,
        d_be_bot,
        d_be_top,
        d_rho_new,
        d_vr_new,
        d_vz_new,
        d_total_energy_new,
        d_ye_int_new,
        d_mass_new,
	        d_dm_floor,
	        nr,
	        nz,
	        button_outer_node_ring,
	        cfg.numerics.floors.rho,
	        (cv_i + cv_e) > 0.0 ? (cv_e / (cv_i + cv_e)) : 0.5);
  } else if (total_energy_remap) {
    ale_remap_2d_rz_total_energy_kernel<<<blocks_cells, 256>>>(
        state.rho.data(),
        d_vr_cell,
        d_vz_cell,
	        d_total_energy_lag,
	        d_ye_int_lag,
	        state.vol.data(),
	        state.mass.data(),
	        state.x_r.data(),
	        state.x_z.data(),
	        state.x_r_reference.data(),
        state.x_z_reference.data(),
        state.cell_vol_initial.data(),
        d_bm_bot,
        d_bm_top,
        d_bpz_bot,
        d_bpz_top,
        d_be_bot,
        d_be_top,
        d_rho_new,
        d_vr_new,
        d_vz_new,
        d_total_energy_new,
        d_ye_int_new,
        d_mass_new,
	        d_dm_floor,
	        nr,
	        nz,
	        button_outer_node_ring,
	        cfg.numerics.floors.rho,
	        (cv_i + cv_e) > 0.0 ? (cv_e / (cv_i + cv_e)) : 0.5);
  } else if (second_order_remap) {
    ale_remap_2d_rz_second_order_kernel<<<blocks_cells, 256>>>(
        state.rho.data(),
        d_vr_cell,
        d_vz_cell,
	        state.ee.data(),
	        state.ei.data(),
	        state.vol.data(),
	        state.mass.data(),
	        state.x_r.data(),
	        state.x_z.data(),
	        state.x_r_reference.data(),
        state.x_z_reference.data(),
        state.cell_vol_initial.data(),
        d_cell_nverts,
        d_bm_bot,
        d_bm_top,
        d_bpz_bot,
        d_bpz_top,
        d_be_bot,
        d_be_top,
        d_rho_new,
        d_vr_new,
        d_vz_new,
        d_ee_new,
        d_ei_new,
        d_mass_new,
        state.zbar.empty() ? nullptr : state.zbar.data(),
        d_dm_floor,
	        d_de_floor,
	        nr,
	        nz,
	        button_outer_node_ring,
	        cfg.numerics.floors.rho,
	        cfg.numerics.floors.Te,
	        ti_floor_remap,
        gamma,
        A,
        remap_burn_species ? d_burn_species_Y_lag : nullptr,
        remap_burn_species ? d_burn_species_Y_new : nullptr,
        remap_hot_e_eps ? d_hot_e_eps_lag : nullptr,
        remap_hot_e_eps ? d_hot_e_eps_new : nullptr,
        remap_burn_eps ? d_burn_eps_lag : nullptr,
        remap_burn_eps ? d_burn_eps_new : nullptr);
  } else {
    ale_remap_2d_rz_kernel<<<blocks_cells, 256>>>(
        state.rho.data(),
        d_vr_cell,
        d_vz_cell,
	        state.ee.data(),
	        state.ei.data(),
	        state.vol.data(),
	        state.mass.data(),
	        state.x_r.data(),
	        state.x_z.data(),
	        state.x_r_reference.data(),
        state.x_z_reference.data(),
        state.cell_vol_initial.data(),
        d_cell_nverts,
        d_bm_bot,
        d_bm_top,
        d_bpz_bot,
        d_bpz_top,
        d_be_bot,
        d_be_top,
        d_rho_new,
        d_vr_new,
        d_vz_new,
        d_ee_new,
        d_ei_new,
        d_mass_new,
        state.zbar.empty() ? nullptr : state.zbar.data(),
        d_dm_floor,
	        d_de_floor,
	        nr,
	        nz,
	        button_outer_node_ring,
	        cfg.numerics.floors.rho,
	        cfg.numerics.floors.Te,
	        ti_floor_remap,
        gamma,
        A,
        remap_burn_species ? d_burn_species_Y_lag : nullptr,
        remap_burn_species ? d_burn_species_Y_new : nullptr,
        remap_hot_e_eps ? d_hot_e_eps_lag : nullptr,
        remap_hot_e_eps ? d_hot_e_eps_new : nullptr,
        remap_burn_eps ? d_burn_eps_lag : nullptr,
        remap_burn_eps ? d_burn_eps_new : nullptr);
  }
  CUDA_CHECK(cudaGetLastError());

  double* const d_rho_mass_preflight_lag = static_cast<double*>(
      core::device_scratch_acquire(
          "ale_remap:structured:d_rho_mass_preflight_lag",
          static_cast<std::size_t>(n_cells) * sizeof(double)));
  double* const d_rho_mass_preflight_new = static_cast<double*>(
      core::device_scratch_acquire(
          "ale_remap:structured:d_rho_mass_preflight_new",
          static_cast<std::size_t>(n_cells) * sizeof(double)));
  int* const d_mass_preflight_first_rejected_cell = static_cast<int*>(
      core::device_scratch_acquire(
          "ale_remap:structured:d_mass_preflight_first_rejected_cell",
          sizeof(int)));
  double* d_remap_mass_veto_floor = nullptr;
  if (!state.evacuated_cells.mass_ref.empty()) {
    TENRYU_ASSERT(
        state.evacuated_cells.mass_ref.size() ==
            static_cast<std::size_t>(n_cells),
        "ALE remap mass veto floor requires one reference mass per cell");
    std::vector<double> remap_mass_veto_floor(static_cast<std::size_t>(n_cells));
    for (int c = 0; c < n_cells; ++c) {
      remap_mass_veto_floor[static_cast<std::size_t>(c)] =
          cfg.numerics.ale.evacuated_cell.off_mass_fraction *
          state.evacuated_cells.mass_ref[static_cast<std::size_t>(c)];
    }
    d_remap_mass_veto_floor = static_cast<double*>(core::device_scratch_acquire(
        "ale_remap:structured:d_remap_mass_veto_floor",
        static_cast<std::size_t>(n_cells) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_remap_mass_veto_floor,
                          remap_mass_veto_floor.data(),
                          static_cast<std::size_t>(n_cells) * sizeof(double),
                          cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemcpy(d_rho_mass_preflight_lag,
                        state.rho.data(),
                        static_cast<std::size_t>(n_cells) * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  if (second_order_remap) {
    ale_remap_2d_rz_radiation_second_order_kernel<false>
        <<<blocks_cells, 256>>>(
        d_rho_mass_preflight_lag,
        state.vol.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.x_r_reference.data(),
        state.x_z_reference.data(),
        state.cell_vol_initial.data(),
        d_cell_nverts,
        d_bm_bot,
        d_bm_top,
        d_rho_mass_preflight_new,
        nr,
        nz,
        button_outer_node_ring);
  } else {
    ale_remap_2d_rz_radiation_kernel<false><<<blocks_cells, 256>>>(
        d_rho_mass_preflight_lag,
        state.vol.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.x_r_reference.data(),
        state.x_z_reference.data(),
        state.cell_vol_initial.data(),
        d_cell_nverts,
        d_bm_bot,
        d_bm_top,
        d_rho_mass_preflight_new,
        nr,
        nz,
        button_outer_node_ring);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(d_mass_preflight_first_rejected_cell,
                        &n_cells,
                        sizeof(int),
                        cudaMemcpyHostToDevice));
  ale_remap_2d_rz_mass_preflight_check_kernel<<<blocks_cells, 256>>>(
      d_mass_preflight_first_rejected_cell,
      d_rho_mass_preflight_new,
      state.rho.data(),
      state.cell_vol_initial.data(),
      state.vol.data(),
      d_remap_mass_veto_floor,
      n_cells,
      kRemapMassClosureRejectFraction);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(core::debug_kernel_sync());
  int mass_preflight_first_rejected_cell = n_cells;
  CUDA_CHECK(cudaMemcpy(&mass_preflight_first_rejected_cell,
                        d_mass_preflight_first_rejected_cell,
                        sizeof(int),
                        cudaMemcpyDeviceToHost));
  if (mass_preflight_first_rejected_cell >= 0 &&
      mass_preflight_first_rejected_cell < n_cells) {
    double predicted_rho = 0.0;
    double pre_rho = 0.0;
    double vol_new = 0.0;
    double vol_old = 0.0;
    CUDA_CHECK(cudaMemcpy(
        &predicted_rho,
        d_rho_mass_preflight_new + mass_preflight_first_rejected_cell,
        sizeof(double),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &pre_rho,
        state.rho.data() + mass_preflight_first_rejected_cell,
        sizeof(double),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &vol_new,
        state.cell_vol_initial.data() + mass_preflight_first_rejected_cell,
        sizeof(double),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &vol_old,
        state.vol.data() + mass_preflight_first_rejected_cell,
        sizeof(double),
        cudaMemcpyDeviceToHost));
    core::log_warning(
        std::string("[ale-remap-mass-preflight-2drz] REJECTED cell=") +
        std::to_string(mass_preflight_first_rejected_cell) +
        " predicted=" +
        format_ale_velcoherence_value(predicted_rho * vol_new) +
        " pre=" + format_ale_velcoherence_value(pre_rho * vol_old));
    cleanup();
    return result;
  }

  const int n_groups = std::max(cfg.radiation.groups, 1);
  const std::size_t expected_rad_size =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  if (cfg.numerics.ale.conservative_remap_radiation_enabled &&
      state.rad_E.size() == expected_rad_size) {
    d_rad_old = static_cast<double*>(
        core::device_scratch_acquire(
            "ale_remap:structured:d_rad_old",
            static_cast<std::size_t>(n_cells) * sizeof(double)));
    d_rad_new = static_cast<double*>(
        core::device_scratch_acquire(
            "ale_remap:structured:d_rad_new",
            static_cast<std::size_t>(n_cells) * sizeof(double)));
    const double bottom_E_lte =
        core::constants::a_eV * std::pow(std::max(bc.z_bottom_cfg.supply_T_eV, 0.0), 4);
    const double top_E_lte =
        core::constants::a_eV * std::pow(std::max(bc.z_top_cfg.supply_T_eV, 0.0), 4);
    for (int g = 0; g < n_groups; ++g) {
      memset_zero(d_brad_bot, static_cast<std::size_t>(nr));
      memset_zero(d_brad_top, static_cast<std::size_t>(nr));
      gather_group_field_kernel<<<blocks_cells, 256>>>(
          d_rad_old, state.rad_E.data(), n_cells, n_groups, g);
      CUDA_CHECK(cudaGetLastError());
      if (use_state_supply_radial_avg && d_state_supply_donor_avg != nullptr) {
        state_supply_boundary_donor_radial_avg_kernel<<<1, 256>>>(
            nullptr,
            nullptr,
            d_vz_cell,
            nullptr,
            nullptr,
            d_rad_old,
            d_state_supply_donor_avg + 0,
            d_state_supply_donor_avg + 1,
            d_state_supply_donor_avg + 2,
            d_state_supply_donor_avg + 3,
            d_state_supply_donor_avg + 4,
            d_state_supply_donor_avg + 5,
            d_state_supply_donor_avg + 6,
            d_state_supply_donor_avg + 7,
            d_state_supply_donor_avg + 8,
            d_state_supply_donor_avg + 9,
            nr,
            nz);
        CUDA_CHECK(cudaGetLastError());
      }
      state_supply_boundary_flux_2d_rz_kernel<<<blocks_r, 256>>>(
          state.x_r_reference.data(),
          nullptr,
          d_vz_cell,
          nullptr,
          nullptr,
          d_rad_old,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          d_brad_bot,
          d_brad_top,
          nr,
          nz,
          dt,
          bottom_active ? 1 : 0,
          top_active ? 1 : 0,
          0.0,
          bc.z_bottom_cfg.supply_u_z_cm_per_s,
          0.0,
          bottom_E_lte / static_cast<double>(n_groups),
          0.0,
          bc.z_top_cfg.supply_u_z_cm_per_s,
          0.0,
          top_E_lte / static_cast<double>(n_groups),
          use_state_supply_radial_avg ? 1 : 0,
          d_state_supply_donor_avg != nullptr ? d_state_supply_donor_avg + 0 : nullptr,
          d_state_supply_donor_avg != nullptr ? d_state_supply_donor_avg + 2 : nullptr,
          d_state_supply_donor_avg != nullptr ? d_state_supply_donor_avg + 3 : nullptr,
          d_state_supply_donor_avg != nullptr ? d_state_supply_donor_avg + 4 : nullptr,
          d_state_supply_donor_avg != nullptr ? d_state_supply_donor_avg + 5 : nullptr,
          d_state_supply_donor_avg != nullptr ? d_state_supply_donor_avg + 7 : nullptr,
          d_state_supply_donor_avg != nullptr ? d_state_supply_donor_avg + 8 : nullptr,
          d_state_supply_donor_avg != nullptr ? d_state_supply_donor_avg + 9 : nullptr);
      CUDA_CHECK(cudaGetLastError());
      if (table_closure_mode) {
        accumulate_device_array_max_abs(
            d_brad_bot, nr, d_table_boundary_max_abs);
        accumulate_device_array_max_abs(
            d_brad_top, nr, d_table_boundary_max_abs);
      }
      if (second_order_remap) {
        ale_remap_2d_rz_radiation_second_order_kernel<true>
            <<<blocks_cells, 256>>>(
            d_rad_old,
            state.vol.data(),
            state.x_r.data(),
            state.x_z.data(),
            state.x_r_reference.data(),
            state.x_z_reference.data(),
            state.cell_vol_initial.data(),
            d_cell_nverts,
            d_brad_bot,
	            d_brad_top,
	            d_rad_new,
	            nr,
	            nz,
	            button_outer_node_ring);
      } else {
        ale_remap_2d_rz_radiation_kernel<true><<<blocks_cells, 256>>>(
            d_rad_old,
            state.vol.data(),
            state.x_r.data(),
            state.x_z.data(),
            state.x_r_reference.data(),
            state.x_z_reference.data(),
            state.cell_vol_initial.data(),
            d_cell_nverts,
            d_brad_bot,
	            d_brad_top,
	            d_rad_new,
	            nr,
	            nz,
	            button_outer_node_ring);
      }
      CUDA_CHECK(cudaGetLastError());
      scatter_group_field_kernel<<<blocks_cells, 256>>>(
          state.rad_E.data(), d_rad_new, n_cells, n_groups, g);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  CUDA_CHECK(cudaMemcpy(state.rho.data(),
                        d_rho_new,
                        static_cast<std::size_t>(n_cells) * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  if (!total_energy_remap) {
    CUDA_CHECK(cudaMemcpy(state.ee.data(),
                          d_ee_new,
                          static_cast<std::size_t>(n_cells) * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.ei.data(),
                          d_ei_new,
                          static_cast<std::size_t>(n_cells) * sizeof(double),
                          cudaMemcpyDeviceToDevice));
  }
  CUDA_CHECK(cudaMemcpy(state.mass.data(),
                        d_mass_new,
                        static_cast<std::size_t>(n_cells) * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  if (remap_burn_species) {
    CUDA_CHECK(cudaMemcpy(
        state.burn_n_host.data(),
        d_burn_species_Y_new,
        static_cast<std::size_t>(n_cells) * tenryu::burn::kNumSpecies *
            sizeof(double),
        cudaMemcpyDeviceToHost));
  }
  if (remap_hot_e_eps) {
    CUDA_CHECK(cudaMemcpy(state.hot_e_eps_cum_host.data(),
                          d_hot_e_eps_new,
                          static_cast<std::size_t>(n_cells) * sizeof(double),
                          cudaMemcpyDeviceToHost));
  }
  if (remap_burn_eps) {
    CUDA_CHECK(cudaMemcpy(state.burn_eps_cum_host.data(),
                          d_burn_eps_new,
                          static_cast<std::size_t>(n_cells) * sizeof(double),
                          cudaMemcpyDeviceToHost));
  }
  if (remap_burn_spectrum) {
    const std::size_t burn_planes = 6U * burn_G;
    const std::size_t burn_Ng_size =
        burn_planes * static_cast<std::size_t>(n_cells);
    const std::size_t plane_bytes =
        static_cast<std::size_t>(n_cells) * sizeof(double);
    if (state.burn_Ng_work.size() != burn_Ng_size) {
      state.burn_Ng_work.reset(burn_Ng_size);
    }
    burn_spectrum_scale_rows_by_cell(
        state.burn_Ng_work.data(),
        state.burn_Ng.data(),
        d_burn_rho_lag,
        burn_planes,
        n_cells);
    double* const d_burn_Ng_plane_out = static_cast<double*>(
        core::device_scratch_acquire("burn:remap:Ng_plane_out", plane_bytes));
    for (std::size_t p = 0; p < burn_planes; ++p) {
      double* const plane =
          state.burn_Ng_work.data() + p * static_cast<std::size_t>(n_cells);
      if (second_order_remap) {
        ale_remap_2d_rz_radiation_second_order_kernel<true>
            <<<blocks_cells, 256>>>(
            plane,
            state.vol.data(),
            state.x_r.data(),
            state.x_z.data(),
            state.x_r_reference.data(),
            state.x_z_reference.data(),
            state.cell_vol_initial.data(),
            d_cell_nverts,
            nullptr,
            nullptr,
            d_burn_Ng_plane_out,
            nr,
            nz,
            button_outer_node_ring);
      } else {
        ale_remap_2d_rz_radiation_kernel<true><<<blocks_cells, 256>>>(
            plane,
            state.vol.data(),
            state.x_r.data(),
            state.x_z.data(),
            state.x_r_reference.data(),
            state.x_z_reference.data(),
            state.cell_vol_initial.data(),
            d_cell_nverts,
            nullptr,
            nullptr,
            d_burn_Ng_plane_out,
            nr,
            nz,
            button_outer_node_ring);
      }
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(plane,
                            d_burn_Ng_plane_out,
                            plane_bytes,
                            cudaMemcpyDeviceToDevice));
    }
    burn_spectrum_divide_rows_by_cell(
        state.burn_Ng.data(),
        state.burn_Ng_work.data(),
        state.rho.data(),
        burn_planes,
        n_cells);
  }
  CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                        state.x_r_reference.data(),
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                        state.x_z_reference.data(),
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice));

  core::DeviceArray<double> d_center_saved_v(
      "ale_remap:structured_remap:d_center_saved_v");
  if (d_node_flags != nullptr) {
    d_center_saved_v.reset(2048U);
    save_center_column_velocity_kernel<<<1, 1>>>(
        d_center_saved_v.data(), d_center_saved_v.data() + 1024,
        state.v_r.data(), state.v_z.data(), d_node_flags,
        (nr + 1) * (nz + 1));
    CUDA_CHECK(cudaGetLastError());
  }

  const auto r_outer_type = parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_outer);
  const auto z_bottom_type = parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_bottom);
  const auto z_top_type = parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_top);
  if (button_outer_node_ring > 0) {
    project_cell_velocity_to_nodes_button_kernel<<<blocks_nodes, 256>>>(
        state.v_r.data(),
        state.v_z.data(),
        d_vr_new,
        d_vz_new,
        state.rho.data(),
        state.cell_vol_initial.data(),
        d_node_flags,
        nr,
        nz,
        button_outer_node_ring,
        velocity_bc_mode_local(r_outer_type),
        velocity_bc_mode_local(z_bottom_type),
        velocity_bc_mode_local(z_top_type));
  } else {
    project_cell_velocity_to_nodes_kernel<<<blocks_nodes, 256>>>(
        state.v_r.data(),
        state.v_z.data(),
        d_vr_new,
        d_vz_new,
        state.rho.data(),
        state.cell_vol_initial.data(),
        d_node_flags,
        nr,
        nz,
        velocity_bc_mode_local(r_outer_type),
        velocity_bc_mode_local(z_bottom_type),
        velocity_bc_mode_local(z_top_type));
  }
  CUDA_CHECK(cudaGetLastError());

  if (d_node_flags != nullptr) {
    restore_center_column_velocity_kernel<<<1, 1>>>(
        state.v_r.data(), state.v_z.data(), d_center_saved_v.data(),
        d_center_saved_v.data() + 1024, d_node_flags,
        (nr + 1) * (nz + 1));
    CUDA_CHECK(cudaGetLastError());
  }

  if (i1b_spurious_sensor) {
    ale_cell_kinetic_from_velocity_kernel<<<blocks_cells, 256>>>(
        d_ke_before_projection,
        d_mass_new,
        d_vr_new,
        d_vz_new,
        n_cells,
        nz,
        button_outer_node_ring);
    CUDA_CHECK(cudaGetLastError());
    ale_structured_corner_kinetic_total_kernel<<<blocks_cells, 256>>>(
        d_ke_after_projection,
        state.mass.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.v_r.data(),
        state.v_z.data(),
        nr,
        nz,
        button_outer_node_ring);
    CUDA_CHECK(cudaGetLastError());
    result.i1b_ale_ke_sensor = reduce_ale_ke_projection_sensor(
        d_ke_before_projection,
        d_ke_after_projection,
        d_cell_nverts,
        n_cells,
        tenryu::hydro::i1b_spurious_sensor_top_k());
  }

  if (total_energy_remap) {
    if (physical_ke_remap) {
      recover_internal_from_total_energy_remap_physical_ke_kernel<<<blocks_cells, 256>>>(
          state.ee.data(),
          state.ei.data(),
          d_total_energy_new,
          d_ye_int_new,
          state.mass.data(),
          state.x_r.data(),
          state.x_z.data(),
          state.v_r.data(),
          state.v_z.data(),
          state.zbar.empty() ? nullptr : state.zbar.data(),
          d_de_floor,
          nullptr,
          d_cell_nverts,
          rz::corner_mass_fallback_device_recorder(),
          corner_mass_convention,
          nr,
          nz,
          button_outer_node_ring,
          gamma,
          A,
          z_bc,
          cfg.numerics.floors.Te,
          ti_floor_remap);
    } else {
      recover_internal_from_total_energy_remap_kernel<<<blocks_cells, 256>>>(
          state.ee.data(),
          state.ei.data(),
          d_total_energy_new,
          d_ye_int_new,
          state.mass.data(),
          state.x_r.data(),
          state.x_z.data(),
          state.v_r.data(),
          state.v_z.data(),
          state.zbar.empty() ? nullptr : state.zbar.data(),
          d_de_floor,
          nullptr,
          nr,
          nz,
          button_outer_node_ring,
          gamma,
          A,
          z_bc,
          cfg.numerics.floors.Te,
          ti_floor_remap);
    }
    CUDA_CHECK(cudaGetLastError());
  }

  // 1T runs keep the total internal energy in ee with ei == 0 by convention;
  // flooring ei at cv_i*Ti_floor would fabricate unledgered energy that the
  // next 1T projection folds into ee (a compounding per-remap pump).
  if (table_closure_mode) {
    eos_reclosure_table_lte_kernel<<<blocks_cells, 256>>>(
        state.Te.data(),
        state.Ti.data(),
        state.Pe.data(),
        state.Pi.data(),
        state.ee.data(),
        state.ei.data(),
        state.rho.data(),
        state.mass.data(),
        state.zbar.empty() ? nullptr : state.zbar.data(),
        n_cells,
        nz,
        button_outer_node_ring,
        nullptr,
        cfg.main.two_temperature ? 1 : 0,
        A,
        cfg.numerics.floors.rho,
        cfg.numerics.floors.Te,
        ti_floor_remap,
        eos_ctx->ion_view(0),
        eos_ctx->electron_view(0),
        eos_ctx->total_view(0),
        cfg.materials.low_density_extrapolation ? 1 : 0,
        d_de_floor,
        d_table_closure_status);
  }
  if (!table_closure_mode) {
  eos_reclosure_ideal_gas_kernel<<<blocks_cells, 256>>>(
      state.Te.data(),
      state.Ti.data(),
      state.Pe.data(),
      state.Pi.data(),
      state.ee.data(),
      state.ei.data(),
      state.rho.data(),
	      state.zbar.empty() ? nullptr : state.zbar.data(),
	      n_cells,
	      nz,
	      button_outer_node_ring,
	      nullptr,
	      cfg.main.two_temperature ? 1 : 0,
	      gamma,
	      A,
      cfg.numerics.floors.rho,
      cfg.numerics.floors.Te,
      ti_floor_remap);
  }
  CUDA_CHECK(cudaGetLastError());
  if (table_closure_mode) {
    std::vector<int> closure_status(static_cast<std::size_t>(n_cells), 0);
    CUDA_CHECK(cudaMemcpy(closure_status.data(),
                          d_table_closure_status,
                          static_cast<std::size_t>(n_cells) * sizeof(int),
                          cudaMemcpyDeviceToHost));
    int bad_cells = 0;
    int first_status = 0;
    for (const int status : closure_status) {
      if (status != 0) {
        if (bad_cells == 0) {
          first_status = status;
        }
        ++bad_cells;
      }
    }
    double boundary_max_abs = 0.0;
    CUDA_CHECK(cudaMemcpy(&boundary_max_abs,
                          d_table_boundary_max_abs,
                          sizeof(double),
                          cudaMemcpyDeviceToHost));
    const bool boundary_guard_fired = boundary_max_abs > 0.0;
    if (bad_cells > 0 || boundary_guard_fired) {
      restore_device_array(
          state.rho.data(), d_table_snapshot_rho, state.rho.size());
      restore_device_array(
          state.mass.data(), d_table_snapshot_mass, state.mass.size());
      restore_device_array(
          state.ee.data(), d_table_snapshot_ee, state.ee.size());
      restore_device_array(
          state.ei.data(), d_table_snapshot_ei, state.ei.size());
      restore_device_array(
          state.Te.data(), d_table_snapshot_Te, state.Te.size());
      restore_device_array(
          state.Ti.data(), d_table_snapshot_Ti, state.Ti.size());
      restore_device_array(
          state.Pe.data(), d_table_snapshot_Pe, state.Pe.size());
      restore_device_array(
          state.Pi.data(), d_table_snapshot_Pi, state.Pi.size());
      restore_device_array(
          state.v_r.data(), d_table_snapshot_v_r, state.v_r.size());
      restore_device_array(
          state.v_z.data(), d_table_snapshot_v_z, state.v_z.size());
      restore_device_array(state.corner_mass.data(),
                           d_table_snapshot_corner_mass,
                           state.corner_mass.size());
      restore_device_array(state.corner_volume.data(),
                           d_table_snapshot_corner_volume,
                           state.corner_volume.size());
      restore_device_array(state.subzonal_mass_corner0.data(),
                           d_table_snapshot_subzonal_mass_corner0,
                           state.subzonal_mass_corner0.size());
      restore_device_array(state.subzonal_mass_corner1.data(),
                           d_table_snapshot_subzonal_mass_corner1,
                           state.subzonal_mass_corner1.size());
      restore_device_array(state.subzonal_mass_corner2.data(),
                           d_table_snapshot_subzonal_mass_corner2,
                           state.subzonal_mass_corner2.size());
      restore_device_array(state.subzonal_mass_corner3.data(),
                           d_table_snapshot_subzonal_mass_corner3,
                           state.subzonal_mass_corner3.size());
      restore_device_array(
          state.rad_E.data(), d_table_snapshot_rad_E, state.rad_E.size());
      if (remap_burn_species) {
        CUDA_CHECK(cudaMemcpy(
            state.burn_n_host.data(),
            d_table_snapshot_burn_species,
            state.burn_n_host.size() * sizeof(double),
            cudaMemcpyDeviceToHost));
      }
      if (remap_hot_e_eps) {
        CUDA_CHECK(cudaMemcpy(
            state.hot_e_eps_cum_host.data(),
            d_table_snapshot_hot_e_eps,
            state.hot_e_eps_cum_host.size() * sizeof(double),
            cudaMemcpyDeviceToHost));
      }
      if (remap_burn_eps) {
        CUDA_CHECK(cudaMemcpy(
            state.burn_eps_cum_host.data(),
            d_table_snapshot_burn_eps,
            state.burn_eps_cum_host.size() * sizeof(double),
            cudaMemcpyDeviceToHost));
      }
      if (remap_burn_spectrum) {
        restore_device_array(state.burn_Ng.data(),
                             d_table_snapshot_burn_Ng,
                             state.burn_Ng.size());
        state.burn_Ng_work.reset(table_snapshot_burn_Ng_work_size);
        restore_device_array(state.burn_Ng_work.data(),
                             d_table_snapshot_burn_Ng_work,
                             table_snapshot_burn_Ng_work_size);
      }
      if (button_outer_node_ring > 0) {
        restore_device_array(state.Qvisc.data(),
                             d_table_snapshot_Qvisc,
                             state.Qvisc.size());
        restore_device_array(state.zbar.data(),
                             d_table_snapshot_zbar,
                             state.zbar.size());
        restore_device_array(state.hllc_mom_z_cell.data(),
                             d_table_snapshot_hllc_mom_z_cell,
                             state.hllc_mom_z_cell.size());
      }
      result = AleRemap2DRZResult{};
      result.applied = false;
      result.table_closure_rejected = true;
      result.table_closure_bad_cells = bad_cells;
      result.table_closure_first_status =
          bad_cells > 0 ? first_status : -1;
      core::log_info(
          std::string("[remap-table-eos] rejected bad_cells=") +
          std::to_string(bad_cells) + " first_status=" +
          std::to_string(result.table_closure_first_status) + " boundary=" +
          std::to_string(boundary_guard_fired ? 1 : 0));
      cleanup();
      return result;
    }
  }
  if (button_outer_node_ring > 0) {
    zero_button_dormant_hydro_state_kernel<<<blocks_cells, 256>>>(
        state.rho.data(),
        state.mass.data(),
        state.ee.data(),
        state.ei.data(),
        state.Te.data(),
        state.Ti.data(),
        state.Pe.data(),
        state.Pi.data(),
        state.Qvisc.data(),
        state.zbar.empty() ? nullptr : state.zbar.data(),
        state.hllc_mom_z_cell.empty() ? nullptr : state.hllc_mom_z_cell.data(),
        n_cells,
        nz,
        button_outer_node_ring);
    CUDA_CHECK(cudaGetLastError());
    if (state.rad_E.size() == expected_rad_size) {
      zero_button_dormant_group_state_kernel<<<blocks_cells, 256>>>(
          state.rad_E.data(),
          n_cells,
          n_groups,
          nz,
          button_outer_node_ring);
      CUDA_CHECK(cudaGetLastError());
    }
  }
	  CUDA_CHECK(core::debug_kernel_sync());
	
	  state.mesh.recompute_geometry();
	  if (button_outer_node_ring > 0 &&
	      button_dormant_vol_lag.size() == static_cast<std::size_t>(n_cells)) {
	    for (int c = 0; c < n_cells; ++c) {
	      const int i = c / nz;
	      if (c != 0 && i >= 0 && i < button_outer_node_ring) {
	        state.mesh.cell_vol[static_cast<std::size_t>(c)] =
	            button_dormant_vol_lag[static_cast<std::size_t>(c)];
	      }
	    }
	  }
	  state.vol = state.mesh.cell_vol;
  const bool trace_mesh_motion = mesh_trace::enabled(state, cfg);
  const std::uint64_t subzonal_hash_before =
      trace_mesh_motion ? mesh_trace::coordinate_hash(state) : 0ULL;
  ensure_hourglass_subzonal_masses_2d(state, cfg, true);
  if (trace_mesh_motion) {
    const std::uint64_t subzonal_hash_after =
        mesh_trace::coordinate_hash(state);
    mesh_trace::trace_corner_mass_canary(state,
                                         cfg,
                                         subzonal_hash_before,
                                         subzonal_hash_after);
    mesh_trace::trace_gate_metrics(state, cfg);
  }
  tenryu::hydro::reset_volume_rate_cfl_history_after_ale(state);
  state.holo_ale_invalidated = true;
  state.ale_remaps_applied += 1;
  mesh_trace::trace_post_remap(state, cfg);

  result.boundary_mass_flux_z_bottom = sum_device_array(d_bm_bot, nr);
  result.boundary_mass_flux_z_top = sum_device_array(d_bm_top, nr);
  result.boundary_momentum_flux_z_bottom = sum_device_array(d_bpz_bot, nr);
  result.boundary_momentum_flux_z_top = sum_device_array(d_bpz_top, nr);
  result.boundary_energy_flux_z_bottom = sum_device_array(d_be_bot, nr);
  result.boundary_energy_flux_z_top = sum_device_array(d_be_top, nr);

  const double dM = result.boundary_mass_flux_z_bottom +
                    result.boundary_mass_flux_z_top;
  const double dPz = result.boundary_momentum_flux_z_bottom +
                     result.boundary_momentum_flux_z_top;
  const double dE = result.boundary_energy_flux_z_bottom +
                    result.boundary_energy_flux_z_top;
  state.state_supply_dM_step += dM;
  state.state_supply_dPz_step += dPz;
  state.state_supply_dE_step += dE;
  state.state_supply_dM_cumulative += dM;
  state.state_supply_dPz_cumulative += dPz;
  state.state_supply_dE_cumulative += dE;
  if (total_energy_remap) {
    const double dE_floor = sum_device_array(d_de_floor, 1);
    result.E_floor_injected = (dE_floor > 0.0 && std::isfinite(dE_floor))
                                  ? dE_floor
                                  : 0.0;
    result.E_eint_floor_deposit = result.E_floor_injected;
    if (dE_floor > 0.0 && std::isfinite(dE_floor)) {
      core::log_warning(
          std::string("[total_energy_remap_2d_rz] floor energy injected=") +
          std::to_string(dE_floor) + " erg");
    }
  }
  if (table_closure_mode) {
    const double dE_floor = sum_device_array(d_de_floor, 1);
    result.E_floor_injected = (dE_floor > 0.0 && std::isfinite(dE_floor))
                                  ? dE_floor
                                  : 0.0;
    result.E_eint_floor_deposit = result.E_floor_injected;
  }

  result.applied = true;
  conservation_audit::emit_stage(state, "ale_structured_swept_remap_post");
  cleanup();
  return result;
}

}  // namespace tenryu::hydro::ale
