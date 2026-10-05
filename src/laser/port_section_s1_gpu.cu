#include "laser/port_section_s1_gpu.cuh"

#include <cuda_runtime.h>

#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/constants.hpp"
#include "core/error.hpp"

// Compiled with -fmad=false (src/laser/CMakeLists.txt): the arithmetic below repeats the host
// reference (sector_adapter.cpp, sector_phase_space.cpp and the staging in port_section_s1_host.cpp)
// operation by operation, so contraction into fused multiply-adds must not change it.

namespace tenryu::laser::port_section {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kThreads = 256;
constexpr double kBouguerAuditMuMin = 0.3;    // sector_phase_space.cpp bouguer_drift
constexpr double kBouguerDriftRefFrac = 0.1;  // sector_phase_space.cpp bouguer_drift

inline void s1_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

template <typename T>
void ensure_capacity(T** ptr, std::size_t* capacity, const std::size_t needed, const char* message) {
  if (needed <= *capacity && *ptr != nullptr) {
    return;
  }
  if (*ptr != nullptr) {
    static_cast<void>(cudaFree(*ptr));
    *ptr = nullptr;
  }
  *capacity = 0;
  if (needed == 0) {
    return;
  }
  s1_check(cudaMalloc(reinterpret_cast<void**>(ptr), needed * sizeof(T)), message);
  *capacity = needed;
}

int blocks_for(const long long n) {
  return static_cast<int>(std::max<long long>(1, (n + kThreads - 1) / kThreads));
}

// Device scalars of one build.
struct S1Scalars {
  int n_paths;
  int n_excluded_rays;
  long long total_nodes;
  long long total_crossings;
  double excluded_power_fraction;
  double bouguer_drift_max;
  int error_flags;
};

// ------------------------------------------------------------------------------------------------
// Electron-density fraction eps = max(0, 1 - n_e/n_c) at cell centres and at shells (hydro nodes)

__global__ void cell_eps_kernel(const int n_cells, const double* __restrict__ r_edges,
                                const double* __restrict__ rho, const double* __restrict__ zbar,
                                const double* __restrict__ A_eff, const double n_crit,
                                double* __restrict__ cell_r_center, double* __restrict__ cell_eps) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= n_cells) {
    return;
  }
  cell_r_center[cell] = 0.5 * (r_edges[cell] + r_edges[cell + 1]);
  const double z = ::fmax(zbar[cell], 0.0);
  const double a = ::fmax(A_eff[cell], 1.0e-30);
  const double ne = rho[cell] * z / (a * core::constants::proton_mass);
  cell_eps[cell] = ::fmax(0.0, 1.0 - ne / n_crit);
}

// First index in [0, n) with values[index] > x (std::upper_bound); n when none.
__device__ int upper_bound_index(const double* values, const int n, const double x) {
  int lo = 0;
  int hi = n;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (values[mid] <= x) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// First index in [0, n) with values[index] >= x (std::lower_bound); n when none.
__device__ int lower_bound_index(const double* values, const int n, const double x) {
  int lo = 0;
  int hi = n;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (values[mid] < x) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// Linear interpolation of eps at a radius from samples at increasing radii
// (sector_phase_space.cpp interpolate_eps; the shell eps of port_section_s1_host.cpp).
__device__ double interpolate_sampled(const double* radius_samples, const double* value_samples,
                                      const int n, const double radius) {
  if (radius <= radius_samples[0]) {
    return value_samples[0];
  }
  if (radius >= radius_samples[n - 1]) {
    return value_samples[n - 1];
  }
  const int hi = upper_bound_index(radius_samples, n, radius);
  const int lo = hi - 1;
  const double weight = (radius - radius_samples[lo]) / (radius_samples[hi] - radius_samples[lo]);
  return value_samples[lo] + weight * (value_samples[hi] - value_samples[lo]);
}

__global__ void shell_eps_kernel(const int n_cells, const double* __restrict__ r_edges,
                                 const double* __restrict__ cell_r_center,
                                 const double* __restrict__ cell_eps,
                                 double* __restrict__ shell_eps) {
  const int shell = blockIdx.x * blockDim.x + threadIdx.x;
  if (shell > n_cells) {
    return;
  }
  shell_eps[shell] = interpolate_sampled(cell_r_center, cell_eps, n_cells, r_edges[shell]);
}

// ------------------------------------------------------------------------------------------------
// Ray paths (sector_adapter.cpp build_ray_paths)

__global__ void ray_records_kernel(const int n_rays, const int cap_per_ray, const int n_cells,
                                   const std::int32_t* __restrict__ rec_count,
                                   const std::int32_t* __restrict__ rec_cell,
                                   const double* __restrict__ rec_ds,
                                   std::uint8_t* __restrict__ ray_valid,
                                   std::int32_t* __restrict__ ray_geometric) {
  const int ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray >= n_rays) {
    return;
  }
  const int count = ::min(::max(static_cast<int>(rec_count[ray]), 0), cap_per_ray);
  const long long base = static_cast<long long>(ray) * cap_per_ray;
  bool valid_cells = true;
  for (int k = 0; k < count; ++k) {
    const int cell = rec_cell[base + k];
    if (cell < 0 || cell >= n_cells) {
      valid_cells = false;
      break;
    }
  }
  int geometric = 0;
  if (count > 0 && valid_cells) {
    for (int k = 0; k < count; ++k) {
      if (rec_ds[base + k] > 0.0) {
        ++geometric;
      }
    }
  }
  ray_valid[ray] = (count > 0 && valid_cells && geometric > 0) ? 1U : 0U;
  ray_geometric[ray] = geometric;
}

// Paths in source-ray order; node storage is packed by path. One thread (n_rays is small and the
// order must be the host's).
__global__ void compact_paths_kernel(const int n_rays, const std::uint8_t* __restrict__ ray_valid,
                                     const std::int32_t* __restrict__ ray_geometric,
                                     std::int32_t* __restrict__ path_source,
                                     std::int32_t* __restrict__ path_nodes,
                                     long long* __restrict__ path_node_offset,
                                     S1Scalars* __restrict__ scalars) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int paths = 0;
  long long nodes = 0;
  for (int ray = 0; ray < n_rays; ++ray) {
    if (ray_valid[ray] == 0U) {
      continue;
    }
    path_source[paths] = ray;
    path_node_offset[paths] = nodes;
    path_nodes[paths] = ray_geometric[ray] + 1;
    nodes += ray_geometric[ray] + 1;
    ++paths;
  }
  path_node_offset[paths] = nodes;
  scalars->n_paths = paths;
  scalars->total_nodes = nodes;
}

__device__ double transverse_over_radius(const double mu_value, const double radius) {
  if (!(radius > 0.0)) {
    return 0.0;
  }
  const double mu = ::fmin(::fmax(mu_value, -1.0), 1.0);
  return ::sqrt(::fmax(0.0, 1.0 - mu * mu)) / radius;
}

__device__ double record_alpha(const float mu_value) {
  const double mu = ::fmin(::fmax(static_cast<double>(mu_value), -1.0), 1.0);
  return ::asin(::sqrt(::fmax(0.0, 1.0 - mu * mu)));
}

// Node radius at the end of a record from the next record's cell (or, with none or the same
// cell, the record's direction): sector_adapter.cpp build_ray_paths.
__device__ double record_end_radius(const double* r_edges, const int cell, const int next_cell,
                                    const bool has_next, const float mu) {
  if (has_next && next_cell < cell) {
    return r_edges[cell];
  }
  if (has_next && next_cell > cell) {
    return r_edges[cell + 1];
  }
  return (mu < 0.0f) ? r_edges[cell] : r_edges[cell + 1];
}

// One warp per path: node radii, node mu, theta, alpha, power and the turning node, with the former
// one-thread pass's values. The lanes take the geometric records (ds > 0) 32 at a time, in order;
// a record's node values depend only on it and the geometric record before it. Theta adds the
// records' Simpson increments in record order and the power attenuates record by record: lane 0
// runs those two chains over terms the lanes formed (each expm1 once; the pass called it twice
// with the same argument). The turning node is the first node of smallest radius.
__global__ void build_paths_kernel(const int n_rays_bound, const int cap_per_ray,
                                   const S1Scalars* __restrict__ scalars,
                                   const std::int32_t* __restrict__ path_source,
                                   const long long* __restrict__ path_node_offset,
                                   const std::int32_t* __restrict__ rec_count,
                                   const std::int32_t* __restrict__ rec_cell,
                                   const float* __restrict__ rec_mu,
                                   const double* __restrict__ rec_ds,
                                   const double* __restrict__ rec_S,
                                   const double* __restrict__ ray_P0,
                                   const double* __restrict__ r_edges,
                                   const double impact_spacing,
                                   double* __restrict__ node_r, double* __restrict__ node_mu,
                                   double* __restrict__ node_theta, double* __restrict__ node_alpha,
                                   double* __restrict__ node_power,
                                   std::int32_t* __restrict__ path_turning_node) {
  constexpr unsigned kAll = 0xffffffffU;
  __shared__ double sh_terms[kThreads];
  __shared__ int sh_record[kThreads];
  const int warp_in_block = static_cast<int>(threadIdx.x) / 32;
  double* const terms = sh_terms + 32 * warp_in_block;
  int* const records = sh_record + 32 * warp_in_block;
  const long long thread = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int path = static_cast<int>(thread / 32);
  const int lane = static_cast<int>(thread % 32);
  // The conditions are the same for the warp's lanes, which return together.
  if (path >= n_rays_bound || path >= scalars->n_paths) {
    return;
  }
  const int ray = path_source[path];
  const long long base = static_cast<long long>(ray) * cap_per_ray;
  const int count = ::min(::max(static_cast<int>(rec_count[ray]), 0), cap_per_ray);
  const long long off = path_node_offset[path];
  double* r = node_r + off;
  double* mu = node_mu + off;

  // Node radii and node mu over the geometric records: a chunk's records are listed in order;
  // record j of the path (j > 0) gives node j from record j - 1, and record 1 also the start node.
  int geometric = 0;     // geometric records before the chunk
  int previous = -1;     // the last geometric record before the chunk
  for (int chunk = 0; chunk < count; chunk += 32) {
    const int k_lane = chunk + lane;
    const bool is_geometric = k_lane < count && rec_ds[base + k_lane] > 0.0;
    const unsigned ballot = __ballot_sync(kAll, is_geometric);
    if (is_geometric) {
      records[__popc(ballot & ((1U << lane) - 1U))] = k_lane;
    }
    __syncwarp();
    const int in_chunk = __popc(ballot);
    if (lane < in_chunk) {
      const int j = geometric + lane;  // the record's index among the geometric records
      const int k = records[lane];
      if (j == 0) {
        mu[0] = static_cast<double>(rec_mu[base + k]);
      } else {
        const int prev = (lane > 0) ? records[lane - 1] : previous;
        const int cell_previous = rec_cell[base + prev];
        const int cell = rec_cell[base + k];
        const float mu_previous = rec_mu[base + prev];
        if (j == 1) {
          // Path start: the first record's entry edge, from the second record's cell.
          if (cell < cell_previous) {
            r[0] = r_edges[cell_previous + 1];
          } else if (cell > cell_previous) {
            r[0] = r_edges[cell_previous];
          } else {
            r[0] = (mu_previous < 0.0f) ? r_edges[cell_previous + 1] : r_edges[cell_previous];
          }
        }
        r[j] = record_end_radius(r_edges, cell_previous, cell, true, mu_previous);
        mu[j] = static_cast<double>(mu_previous);
      }
    }
    if (in_chunk > 0) {
      previous = records[in_chunk - 1];
    }
    geometric += in_chunk;
    __syncwarp();
  }
  const int ordinal = ::max(geometric - 1, 0);
  if (lane == 0) {
    const int cell_last = rec_cell[base + previous];
    const float mu_last = rec_mu[base + previous];
    if (ordinal == 0) {
      r[0] = (mu_last < 0.0f) ? r_edges[cell_last + 1] : r_edges[cell_last];
    }
    r[ordinal + 1] = record_end_radius(r_edges, cell_last, cell_last, false, mu_last);
    mu[ordinal + 1] = static_cast<double>(mu_last);
  }
  __syncwarp();
  const int n_records = ordinal + 1;  // geometric records

  // theta: entry angle from the impact parameter, then Simpson's rule per record (the increments
  // by the lanes, node g + 1's at theta[g + 1], then their sum in order by lane 0).
  double* theta = node_theta + off;
  for (int chunk = 0, g_base = 0; chunk < count && g_base < n_records; chunk += 32) {
    const int k_lane = chunk + lane;
    const bool is_geometric = k_lane < count && rec_ds[base + k_lane] > 0.0;
    const unsigned ballot = __ballot_sync(kAll, is_geometric);
    const int g = g_base + __popc(ballot & ((1U << lane) - 1U));
    if (is_geometric && g < n_records) {
      const double r_mid = 0.5 * (r[g] + r[g + 1]);
      const double mu_mid = 0.5 * (mu[g] + mu[g + 1]);
      theta[g + 1] =
          rec_ds[base + k_lane] / 6.0 *
          (transverse_over_radius(mu[g], r[g]) + 4.0 * transverse_over_radius(mu_mid, r_mid) +
           transverse_over_radius(mu[g + 1], r[g + 1]));
    }
    g_base += __popc(ballot);
  }
  __syncwarp();
  if (lane == 0) {
    const double impact = ::fmax(0.0, impact_spacing * (static_cast<double>(ray) + 0.5));
    const double r0 = r[0];
    const double entry_ratio = (r0 > 0.0) ? ::fmin(1.0, impact / r0) : 1.0;
    theta[0] = ::asin(entry_ratio);
    // One increment per geometric record (none on a path without one).
    for (int g = 0; g < geometric; ++g) {
      theta[g + 1] = theta[g] + theta[g + 1];
    }
  }

  double* alpha = node_alpha + off;
  for (int node = lane; node <= n_records; node += 32) {
    alpha[node] = record_alpha(static_cast<float>(mu[node]));
  }

  // Power along the path: every record (geometric or not) attenuates the incident power by its
  // optical depth in two half steps (apply_incident_attenuation in port_section_s1_host.cpp).
  double* power_out = node_power + off;
  double power = ray_P0[ray];
  if (!::isfinite(power) || power < 0.0) {
    power = 0.0;
  }
  if (lane == 0) {
    power_out[0] = power;
  }
  int geometric_node = 0;
  for (int chunk = 0; chunk < count; chunk += 32) {
    const int k_lane = chunk + lane;
    if (k_lane < count) {
      const double S_half = 0.5 * rec_S[base + k_lane];
      terms[lane] = ::expm1(-S_half);
    }
    __syncwarp();
    if (lane == 0) {
      const int end = ::min(32, count - chunk);
      for (int i = 0; i < end; ++i) {
        const double attenuation = terms[i];
        const double dP_first = -power * attenuation;
        power -= dP_first;
        const double dP_second = -power * attenuation;
        power -= dP_second;
        if (rec_ds[base + chunk + i] > 0.0) {
          ++geometric_node;
        }
        power_out[geometric_node] = power;
      }
    }
    __syncwarp();
  }

  // Turning node: the first node of smallest radius (std::min_element). A NaN radius is never
  // smaller; with a NaN first node the fold stays at node 0.
  int best = -1;
  double best_r = 0.0;
  for (int node = lane; node <= n_records; node += 32) {
    const double value = r[node];
    if (value != value) {
      continue;
    }
    if (best < 0 || value < best_r) {
      best = node;
      best_r = value;
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    const int other = __shfl_xor_sync(kAll, best, offset);
    const double other_r = __shfl_xor_sync(kAll, best_r, offset);
    if (other >= 0 && (best < 0 || other_r < best_r || (other_r == best_r && other < best))) {
      best = other;
      best_r = other_r;
    }
  }
  if (lane == 0) {
    const double r_first = r[0];
    path_turning_node[path] = (r_first != r_first || best < 0) ? 0 : best;
  }
}

// Theta of a neighbouring path at a radius on one leg (sector_adapter.cpp interpolate_theta_on_leg):
// an exact node match first, then the first segment that brackets the radius.
__device__ bool theta_on_leg(const double* r, const double* theta, const int n_nodes,
                             const int turning_node, const double radius, const int leg,
                             double* out) {
  const int first = (leg == 0) ? 0 : turning_node;
  const int last = (leg == 0) ? turning_node : n_nodes - 1;
  for (int node = first; node <= last; ++node) {
    if (r[node] == radius) {
      *out = theta[node];
      return true;
    }
  }
  for (int node = first; node < last; ++node) {
    const double r0 = r[node];
    const double r1 = r[node + 1];
    if (r0 == r1 || radius < ::fmin(r0, r1) || radius > ::fmax(r0, r1)) {
      continue;
    }
    const double weight = (radius - r0) / (r1 - r0);
    *out = theta[node] + weight * (theta[node + 1] - theta[node]);
    return true;
  }
  return false;
}

// One thread per (path, node): bundle area from the neighbouring paths' theta at the node radius.
__global__ void path_area_kernel(const int n_rays_bound, const S1Scalars* __restrict__ scalars,
                                 const std::int32_t* __restrict__ path_nodes,
                                 const long long* __restrict__ path_node_offset,
                                 const std::int32_t* __restrict__ path_turning_node,
                                 const double* __restrict__ node_r,
                                 const double* __restrict__ node_theta,
                                 const double* __restrict__ node_alpha,
                                 double* __restrict__ node_area) {
  const int path = blockIdx.y;
  const int node = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_paths = scalars->n_paths;
  if (path >= n_rays_bound || path >= n_paths || node >= path_nodes[path]) {
    return;
  }
  const long long off = path_node_offset[path];
  const double radius = node_r[off + node];
  const double theta_node = node_theta[off + node];
  const int leg = (node <= path_turning_node[path]) ? 0 : 1;
  double inner = 0.0;
  double outer = 0.0;
  bool has_inner = false;
  bool has_outer = false;
  if (path > 0) {
    const long long off_n = path_node_offset[path - 1];
    has_inner = theta_on_leg(node_r + off_n, node_theta + off_n, path_nodes[path - 1],
                             path_turning_node[path - 1], radius, leg, &inner);
  }
  if (path + 1 < n_paths) {
    const long long off_n = path_node_offset[path + 1];
    has_outer = theta_on_leg(node_r + off_n, node_theta + off_n, path_nodes[path + 1],
                             path_turning_node[path + 1], radius, leg, &outer);
  }
  double dtheta_eff = 0.0;
  if (has_inner && has_outer) {
    dtheta_eff = 0.5 * ::fabs(outer - inner);
  } else if (has_inner) {
    dtheta_eff = ::fabs(theta_node - inner);
  } else if (has_outer) {
    dtheta_eff = ::fabs(outer - theta_node);
  }
  dtheta_eff = ::fmax(dtheta_eff, 1.0e-12);
  const double sin_eff = ::fmax(::sin(theta_node), ::sin(0.5 * dtheta_eff));
  const double cos_alpha = ::fmax(::cos(node_alpha[off + node]), 1.0e-6);
  node_area[off + node] = 2.0 * kPi * radius * radius * dtheta_eff * sin_eff * cos_alpha;
}

// ------------------------------------------------------------------------------------------------
// Table (sector_phase_space.cpp build_table)

__device__ int sign_of(const double value) { return (value > 0.0) - (value < 0.0); }

// One warp per path: turning points, caustics, ambiguity and the Bouguer audit. Lane 0 scans the
// turning points in node order; the lanes share the per-node tests and terms of the rest, whose
// results do not depend on the order: the caustic count and first caustic (a count and a minimum
// index), the largest area (fmax: the threshold only enters comparisons, which a zero's sign does
// not change), the first node with |cos alpha| not below the audit's minimum (a minimum index) and
// the drift (the largest of non-negative terms, fmax passing over a NaN as the fold did).
__global__ void annotate_paths_kernel(const int n_rays_bound, const S1Scalars* __restrict__ scalars,
                                      const std::int32_t* __restrict__ path_nodes,
                                      const long long* __restrict__ path_node_offset,
                                      const double* __restrict__ node_r,
                                      const double* __restrict__ node_alpha,
                                      const double* __restrict__ node_area,
                                      const double* __restrict__ shell_r,
                                      const double* __restrict__ shell_eps, const int n_shells,
                                      const double caustic_area_rel_tol,
                                      std::int32_t* __restrict__ path_turning_first,
                                      std::int32_t* __restrict__ path_caustic,
                                      std::uint8_t* __restrict__ path_excluded,
                                      double* __restrict__ path_drift) {
  constexpr unsigned kAll = 0xffffffffU;
  const long long thread = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int path = static_cast<int>(thread / 32);
  const int lane = static_cast<int>(thread % 32);
  // The conditions are the same for the warp's lanes, which return together.
  if (path >= n_rays_bound || path >= scalars->n_paths) {
    return;
  }
  const int n = path_nodes[path];
  const long long off = path_node_offset[path];
  const double* r = node_r + off;
  const double* alpha = node_alpha + off;
  const double* area = node_area + off;

  // find_turning_points
  int turning_count = 0;
  int turning_first = -1;
  if (lane == 0) {
    int previous_sign = 0;
    for (int k = 0; k + 1 < n; ++k) {
      const int current_sign = sign_of(r[k + 1] - r[k]);
      if (current_sign == 0) {
        continue;
      }
      if (previous_sign != 0 && current_sign != previous_sign) {
        ++turning_count;
        if (turning_first < 0) {
          turning_first = k;
        }
      }
      previous_sign = current_sign;
    }
  }

  // find_caustics
  double max_area = area[0];
  for (int k = 1 + lane; k < n; k += 32) {
    max_area = ::fmax(max_area, area[k]);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_area = ::fmax(max_area, __shfl_xor_sync(kAll, max_area, offset));
  }
  const double threshold = caustic_area_rel_tol * max_area;
  int caustic_count = 0;
  int caustic_first = INT_MAX;
  for (int k = 1 + lane; k + 1 < n; k += 32) {
    if (area[k] < area[k - 1] && area[k] < area[k + 1] && area[k] < threshold) {
      caustic_first = ::min(caustic_first, k);
      ++caustic_count;
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    caustic_count += __shfl_xor_sync(kAll, caustic_count, offset);
    caustic_first = ::min(caustic_first, __shfl_xor_sync(kAll, caustic_first, offset));
  }

  // bouguer_drift (the path carries its node alpha: the supplied-alpha branch)
  const auto audited = [&](const int k) { return !(::fabs(::cos(alpha[k])) < kBouguerAuditMuMin); };
  int b0 = INT_MAX;
  for (int k = lane; k < n; k += 32) {
    if (audited(k)) {
      b0 = k;
      break;
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    b0 = ::min(b0, __shfl_xor_sync(kAll, b0, offset));
  }
  double drift = 0.0;
  if (b0 < n) {
    const double eps_b0 = interpolate_sampled(shell_r, shell_eps, n_shells, r[b0]);
    const double B0 = ::sqrt(eps_b0) * r[b0] * ::sin(alpha[b0]);
    const double denominator = ::fmax(::fabs(B0), kBouguerDriftRefFrac * shell_r[n_shells - 1]);
    for (int k = lane; k < n; k += 32) {
      if (!audited(k)) {
        continue;
      }
      const double eps_k = interpolate_sampled(shell_r, shell_eps, n_shells, r[k]);
      const double B = ::sqrt(eps_k) * r[k] * ::sin(alpha[k]);
      drift = ::fmax(drift, ::fabs(B - B0) / denominator);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      drift = ::fmax(drift, __shfl_xor_sync(kAll, drift, offset));
    }
  }

  if (lane == 0) {
    path_turning_first[path] = turning_first;
    path_caustic[path] = (caustic_count == 1) ? caustic_first : -1;
    path_excluded[path] = (turning_count > 1 || caustic_count > 1) ? 1U : 0U;
    path_drift[path] = drift;
  }
}

// Segment range of a sheet: sheet 0 up to the turning point, sheet 1 after it.
__device__ void segment_range(const int n_nodes, const int turning_index, const int sheet,
                              int* begin, int* end) {
  if (turning_index < 0) {
    *begin = 0;
    *end = (sheet == 0) ? n_nodes - 1 : 0;
    return;
  }
  if (sheet == 0) {
    *begin = 0;
    *end = turning_index;
    return;
  }
  *begin = turning_index;
  *end = n_nodes - 1;
}

// The crossings of one path's sheet with every shell, as find_crossing counts them: calls
// visit(shell, segment, weight) for each. The segments are shared by the lanes of a warp (all 32
// lanes call it), so the visits come in another order than find_crossing's.
template <typename Visit>
__device__ void for_each_crossing_warp(const double* r, const int n_nodes, const int turning_first,
                                       const int sheet, const double* shell_r, const int n_shells,
                                       const int lane, Visit visit) {
  int begin = 0;
  int end = 0;
  segment_range(n_nodes, turning_first, sheet, &begin, &end);
  if (begin == end) {
    return;
  }
  int last_nonzero_segment = -1;
  for (int k = begin + lane; k < end; k += 32) {
    if (r[k + 1] != r[k]) {
      last_nonzero_segment = k;
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    last_nonzero_segment =
        ::max(last_nonzero_segment, __shfl_xor_sync(0xffffffffU, last_nonzero_segment, offset));
  }
  if (last_nonzero_segment < 0) {
    return;
  }
  for (int k = begin + lane; k < end; k += 32) {
    const double r0 = r[k];
    const double r1 = r[k + 1];
    if (r0 == r1) {
      continue;
    }
    const double lo = ::fmin(r0, r1);
    const double hi = ::fmax(r0, r1);
    for (int shell = lower_bound_index(shell_r, n_shells, lo); shell < n_shells && shell_r[shell] <= hi;
         ++shell) {
      double weight = (shell_r[shell] - r0) / (r1 - r0);
      weight = ::fmin(::fmax(weight, 0.0), 1.0);
      if (weight == 1.0 && k != last_nonzero_segment) {
        continue;
      }
      visit(shell, k, weight);
    }
  }
}

// ++c up to 2 on one byte, by compare-and-swap on the aligned word that holds it (the counts buffer
// is a whole number of words).
__device__ void saturating_increment_byte(std::uint8_t* c) {
  auto* word = reinterpret_cast<unsigned int*>(reinterpret_cast<std::uintptr_t>(c) & ~std::uintptr_t{3});
  const unsigned int shift = static_cast<unsigned int>(reinterpret_cast<std::uintptr_t>(c) & 3U) * 8U;
  unsigned int old = *word;
  unsigned int assumed = 0U;
  do {
    assumed = old;
    if (((assumed >> shift) & 0xFFU) >= 2U) {
      return;
    }
    old = atomicCAS(word, assumed, assumed + (1U << shift));
  } while (old != assumed);
}

// One warp per (path, sheet): number of crossings per (path, shell, sheet) (saturating at 2; the
// count does not depend on the order of the visits).
__global__ void count_crossings_kernel(const int n_rays_bound, const S1Scalars* __restrict__ scalars,
                                       const std::int32_t* __restrict__ path_nodes,
                                       const long long* __restrict__ path_node_offset,
                                       const std::int32_t* __restrict__ path_turning_first,
                                       const std::uint8_t* __restrict__ path_excluded,
                                       const double* __restrict__ node_r,
                                       const double* __restrict__ shell_r, const int n_shells,
                                       std::uint8_t* __restrict__ counts) {
  const long long thread = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int item = static_cast<int>(thread / 32);
  const int lane = static_cast<int>(thread % 32);
  const int path = item / 2;
  const int sheet = item % 2;
  // The conditions are the same for the warp's lanes, which return together.
  if (path >= n_rays_bound || path >= scalars->n_paths || path_excluded[path] != 0U) {
    return;
  }
  const long long row = static_cast<long long>(path) * (2LL * n_shells);
  for_each_crossing_warp(node_r + path_node_offset[path], path_nodes[path], path_turning_first[path],
                         sheet, shell_r, n_shells, lane, [&](const int shell, const int, const double) {
                           saturating_increment_byte(counts + row + 2LL * shell + sheet);
                         });
}

// One warp per path: a shell crossed twice on one sheet excludes the ray (host
// invariant_violation); the lanes scan the path's bins together.
__global__ void violation_kernel(const int n_rays_bound, const S1Scalars* __restrict__ scalars,
                                 const int n_shells, const std::uint8_t* __restrict__ counts,
                                 std::uint8_t* __restrict__ path_excluded) {
  const long long thread = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int path = static_cast<int>(thread / 32);
  const int lane = static_cast<int>(thread % 32);
  // The conditions are the same for the warp's lanes, which return together.
  if (path >= n_rays_bound || path >= scalars->n_paths || path_excluded[path] != 0U) {
    return;
  }
  const long long row = static_cast<long long>(path) * (2LL * n_shells);
  bool crossed_twice = false;
  for (long long bin = lane; bin < 2LL * n_shells; bin += 32) {
    if (counts[row + bin] > 1U) {
      crossed_twice = true;
      break;
    }
  }
  if (__any_sync(0xffffffffU, crossed_twice) && lane == 0) {
    path_excluded[path] = 1U;
  }
}

// One thread per bin: rank of each kept path's crossing in the bin, in path order.
__global__ void bin_rank_kernel(const int n_rays_bound, const S1Scalars* __restrict__ scalars,
                                const int n_shells, const std::uint8_t* __restrict__ counts,
                                const std::uint8_t* __restrict__ path_excluded,
                                std::int32_t* __restrict__ ranks, int* __restrict__ bin_count) {
  const long long n_bins = 2LL * n_shells;
  const long long bin = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (bin >= n_bins) {
    return;
  }
  const int n_paths = ::min(scalars->n_paths, n_rays_bound);
  int running = 0;
  for (int path = 0; path < n_paths; ++path) {
    const long long index = static_cast<long long>(path) * n_bins + bin;
    if (path_excluded[path] == 0U && counts[index] == 1U) {
      ranks[index] = running;
      ++running;
    }
  }
  bin_count[bin] = running;
}

// One warp per (path, sheet): the crossing values at their slots (each crossing of a kept path has
// its own slot).
__global__ void emit_crossings_kernel(const int n_rays_bound, const S1Scalars* __restrict__ scalars,
                                      const std::int32_t* __restrict__ path_nodes,
                                      const long long* __restrict__ path_node_offset,
                                      const std::int32_t* __restrict__ path_turning_first,
                                      const std::int32_t* __restrict__ path_caustic,
                                      const std::uint8_t* __restrict__ path_excluded,
                                      const double* __restrict__ node_r,
                                      const double* __restrict__ node_theta,
                                      const double* __restrict__ node_alpha,
                                      const double* __restrict__ node_power,
                                      const double* __restrict__ node_area,
                                      const double* __restrict__ shell_r, const int n_shells,
                                      const std::int32_t* __restrict__ ranks,
                                      const int* __restrict__ offsets,
                                      double* __restrict__ out_theta, double* __restrict__ out_alpha,
                                      double* __restrict__ out_power, double* __restrict__ out_area,
                                      std::int32_t* __restrict__ out_ray,
                                      std::uint8_t* __restrict__ out_limiter) {
  const long long thread = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int item = static_cast<int>(thread / 32);
  const int lane = static_cast<int>(thread % 32);
  const int path = item / 2;
  const int sheet = item % 2;
  // The conditions are the same for the warp's lanes, which return together.
  if (path >= n_rays_bound || path >= scalars->n_paths || path_excluded[path] != 0U) {
    return;
  }
  const long long off = path_node_offset[path];
  const double* theta = node_theta + off;
  const double* alpha = node_alpha + off;
  const double* power = node_power + off;
  const double* area = node_area + off;
  const int turning_index = path_turning_first[path];
  const int caustic_index = path_caustic[path];
  const int limiter_lo = ::min(turning_index, caustic_index);
  const int limiter_hi = ::max(turning_index, caustic_index);
  const long long row = static_cast<long long>(path) * (2LL * n_shells);
  for_each_crossing_warp(node_r + off, path_nodes[path], turning_index, sheet, shell_r, n_shells, lane,
                    [&](const int shell, const int k, const double weight) {
                      const long long bin = 2LL * shell + sheet;
                      const long long slot = offsets[bin] + ranks[row + bin];
                      out_theta[slot] = theta[k] + weight * (theta[k + 1] - theta[k]);
                      out_alpha[slot] = alpha[k] + weight * (alpha[k + 1] - alpha[k]);
                      out_power[slot] = power[k] + weight * (power[k + 1] - power[k]);
                      out_area[slot] = area[k] + weight * (area[k + 1] - area[k]);
                      out_ray[slot] = path;
                      out_limiter[slot] =
                          (caustic_index >= 0 && k >= limiter_lo && k <= limiter_hi) ? 1U : 0U;
                    });
}

// One thread per bin: insertion sort by (theta, ray_index) (the host's std::sort order; the
// pairs are unique in a bin, so the order is unique).
__global__ void sort_bins_kernel(const int n_shells, const int* __restrict__ offsets,
                                 double* __restrict__ theta, double* __restrict__ alpha,
                                 double* __restrict__ power, double* __restrict__ area,
                                 std::int32_t* __restrict__ ray, std::uint8_t* __restrict__ limiter) {
  const long long bin = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (bin >= 2LL * n_shells) {
    return;
  }
  const int begin = offsets[bin];
  const int end = offsets[bin + 1];
  for (int i = begin + 1; i < end; ++i) {
    const double t = theta[i];
    const double a = alpha[i];
    const double p = power[i];
    const double s = area[i];
    const std::int32_t q = ray[i];
    const std::uint8_t l = limiter[i];
    int j = i - 1;
    while (j >= begin && (theta[j] > t || (theta[j] == t && ray[j] > q))) {
      theta[j + 1] = theta[j];
      alpha[j + 1] = alpha[j];
      power[j + 1] = power[j];
      area[j + 1] = area[j];
      ray[j + 1] = ray[j];
      limiter[j + 1] = limiter[j];
      --j;
    }
    theta[j + 1] = t;
    alpha[j + 1] = a;
    power[j + 1] = p;
    area[j + 1] = s;
    ray[j + 1] = q;
    limiter[j + 1] = l;
  }
}

// One thread: ledger and audit scalars in the host's order, and the ray impact bins.
__global__ void summarize_kernel(const int n_rays_bound, const int n_shells, const int n_bins,
                                 S1Scalars* __restrict__ scalars,
                                 const std::int32_t* __restrict__ path_source,
                                 const long long* __restrict__ path_node_offset,
                                 const std::uint8_t* __restrict__ path_excluded,
                                 const double* __restrict__ path_drift,
                                 const double* __restrict__ node_power,
                                 const std::int32_t* __restrict__ ray_group_base,
                                 const int* __restrict__ offsets,
                                 std::int32_t* __restrict__ ray_bin) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int n_paths = ::min(scalars->n_paths, n_rays_bound);
  double total_launch_power = 0.0;
  double excluded_launch_power = 0.0;
  int n_excluded = 0;
  double drift_max = 0.0;
  int error_flags = 0;
  for (int path = 0; path < n_paths; ++path) {
    const double launch = node_power[path_node_offset[path]];
    total_launch_power += launch;
    if (path_excluded[path] != 0U) {
      ++n_excluded;
      excluded_launch_power += launch;
    }
    drift_max = ::fmax(drift_max, path_drift[path]);
    const int bin = (n_bins > 0) ? ray_group_base[path_source[path]] % n_bins : -1;
    if (bin < 0 || bin >= n_bins) {
      error_flags |= 1;
    }
    ray_bin[path] = bin;
  }
  scalars->n_excluded_rays = n_excluded;
  scalars->excluded_power_fraction =
      total_launch_power != 0.0 ? excluded_launch_power / total_launch_power : 0.0;
  scalars->bouguer_drift_max = drift_max;
  scalars->total_crossings = offsets[2 * n_shells];
  scalars->error_flags = error_flags;
}

// One thread per (shell, sheet): the snapshot's intensity map (S1HostReference::ray_map).
__global__ void ray_map_kernel(const int n_shells, const int* __restrict__ offsets,
                               const double* __restrict__ theta, const double* __restrict__ power,
                               const double* __restrict__ area, double* __restrict__ map_out) {
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= 2 * n_shells) {
    return;
  }
  const int shell = item / 2;
  const int sheet = item % 2;
  double weighted_intensity[kRayMapThetaBins];
  double bin_power[kRayMapThetaBins];
  for (int b = 0; b < kRayMapThetaBins; ++b) {
    weighted_intensity[b] = 0.0;
    bin_power[b] = 0.0;
  }
  const int begin = offsets[item];
  const int end = offsets[item + 1];
  for (int i = begin; i < end; ++i) {
    const double scaled = ::fmin(::fmax(theta[i], 0.0), kPi) * static_cast<double>(kRayMapThetaBins) / kPi;
    const int b = ::min(::max(static_cast<int>(::floor(scaled)), 0), kRayMapThetaBins - 1);
    const double intensity = power[i] / area[i];
    weighted_intensity[b] += power[i] * intensity;
    bin_power[b] += power[i];
  }
  constexpr double kErgPerSToW = 1.0e-7;
  for (int b = 0; b < kRayMapThetaBins; ++b) {
    const long long index = (static_cast<long long>(shell) * kRayMapThetaBins + b) * 2 + sheet;
    map_out[index] = (bin_power[b] > 0.0) ? weighted_intensity[b] / bin_power[b] * kErgPerSToW : 0.0;
  }
}

}  // namespace

struct S1DeviceWorkspace::Impl {
  double* cell_r_center = nullptr;
  double* cell_eps = nullptr;
  double* shell_eps = nullptr;
  std::uint8_t* ray_valid = nullptr;
  std::int32_t* ray_geometric = nullptr;
  std::int32_t* path_source = nullptr;
  std::int32_t* path_nodes = nullptr;
  long long* path_node_offset = nullptr;
  std::int32_t* path_turning_node = nullptr;
  std::int32_t* path_turning_first = nullptr;
  std::int32_t* path_caustic = nullptr;
  std::uint8_t* path_excluded = nullptr;
  double* path_drift = nullptr;
  double* node_r = nullptr;
  double* node_mu = nullptr;
  double* node_theta = nullptr;
  double* node_alpha = nullptr;
  double* node_power = nullptr;
  double* node_area = nullptr;
  std::uint8_t* counts = nullptr;
  std::int32_t* ranks = nullptr;
  int* bin_count = nullptr;
  int* offsets = nullptr;
  double* out_theta = nullptr;
  double* out_alpha = nullptr;
  double* out_power = nullptr;
  double* out_area = nullptr;
  std::int32_t* out_ray = nullptr;
  std::uint8_t* out_limiter = nullptr;
  std::int32_t* ray_bin = nullptr;
  S1Scalars* scalars = nullptr;
  S1Scalars* host_scalars = nullptr;  // pinned
  void* scan_temp = nullptr;

  std::size_t cap_cells = 0, cap_cells2 = 0, cap_shells = 0;
  std::size_t cap_rays_u8 = 0, cap_rays_i32 = 0, cap_paths_src = 0, cap_paths_nodes = 0,
              cap_paths_offset = 0, cap_turn = 0, cap_turn_first = 0, cap_caustic = 0,
              cap_excluded = 0, cap_drift = 0, cap_ray_bin = 0;
  std::size_t cap_node_r = 0, cap_node_mu = 0, cap_node_theta = 0, cap_node_alpha = 0,
              cap_node_power = 0, cap_node_area = 0;
  std::size_t cap_counts = 0, cap_ranks = 0, cap_bin_count = 0, cap_offsets = 0;
  std::size_t cap_out_theta = 0, cap_out_alpha = 0, cap_out_power = 0, cap_out_area = 0,
              cap_out_ray = 0, cap_out_limiter = 0;
  std::size_t cap_scalars = 0;
  std::size_t scan_temp_bytes = 0;

  ~Impl() {
    void* device_ptrs[] = {cell_r_center, cell_eps, shell_eps, ray_valid, ray_geometric, path_source,
                           path_nodes, path_node_offset, path_turning_node, path_turning_first,
                           path_caustic, path_excluded, path_drift, node_r, node_mu, node_theta,
                           node_alpha, node_power, node_area, counts, ranks, bin_count, offsets,
                           out_theta, out_alpha, out_power, out_area, out_ray, out_limiter, ray_bin,
                           scalars, scan_temp};
    for (void* p : device_ptrs) {
      if (p != nullptr) {
        static_cast<void>(cudaFree(p));
      }
    }
    if (host_scalars != nullptr) {
      static_cast<void>(cudaFreeHost(host_scalars));
    }
  }
};

S1DeviceWorkspace::S1DeviceWorkspace() : impl_(new Impl) {}
S1DeviceWorkspace::~S1DeviceWorkspace() { delete impl_; }

S1DeviceSummary build_s1_table_device(const S1DeviceInput& input, S1DeviceWorkspace& workspace,
                                      cudaStream_t stream, S1DeviceTable* table_out) {
  TENRYU_ASSERT(input.n_rays >= 0 && input.n_cells > 0, "port_section s1 device: empty mesh");
  TENRYU_ASSERT(input.n_rays == 0 || input.cap_per_ray > 0,
                "port_section s1 device: no record capacity");
  TENRYU_ASSERT(input.n_bins > 0, "port_section s1 device: n_bins must be positive");
  // path_area_kernel puts the paths on the grid's y dimension.
  TENRYU_ASSERT(input.n_rays <= 65535, "port_section s1 device: more than 65535 rays");
  auto& w = *workspace.impl();
  const int R = input.n_rays;
  const int C = input.cap_per_ray;
  const int N = input.n_cells;
  const int n_shells = N + 1;
  const std::size_t n_bins_table = 2U * static_cast<std::size_t>(n_shells);
  if (R == 0) {
    // No ray this step (every beam with power gave an empty ray set): an empty table, whose
    // bins hold no crossing.
    ensure_capacity(&w.offsets, &w.cap_offsets, n_bins_table + 1U, "s1 offsets");
    s1_check(cudaMemsetAsync(w.offsets, 0, (n_bins_table + 1U) * sizeof(int), stream),
             "s1 empty offsets");
    if (table_out != nullptr) {
      *table_out = S1DeviceTable{};
      table_out->n_shells = n_shells;
      table_out->offsets = w.offsets;
      table_out->theta = w.out_theta;
      table_out->alpha = w.out_alpha;
      table_out->power = w.out_power;
      table_out->area = w.out_area;
      table_out->ray_index = w.out_ray;
      table_out->in_limiter = w.out_limiter;
      table_out->ray_bin = w.ray_bin;
      table_out->shell_r = input.r_edges;
    }
    return S1DeviceSummary{};
  }
  // Upper bounds (no readback before the table exists): every source ray may be a path, every
  // record a geometric node, and every (path, bin) a crossing.
  const std::size_t node_cap = static_cast<std::size_t>(R) * (static_cast<std::size_t>(C) + 1U);
  const std::size_t cross_cap = static_cast<std::size_t>(R) * n_bins_table;

  ensure_capacity(&w.cell_r_center, &w.cap_cells, static_cast<std::size_t>(N), "s1 cell_r_center");
  ensure_capacity(&w.cell_eps, &w.cap_cells2, static_cast<std::size_t>(N), "s1 cell_eps");
  ensure_capacity(&w.shell_eps, &w.cap_shells, static_cast<std::size_t>(n_shells), "s1 shell_eps");
  ensure_capacity(&w.ray_valid, &w.cap_rays_u8, static_cast<std::size_t>(R), "s1 ray_valid");
  ensure_capacity(&w.ray_geometric, &w.cap_rays_i32, static_cast<std::size_t>(R), "s1 ray_geometric");
  ensure_capacity(&w.path_source, &w.cap_paths_src, static_cast<std::size_t>(R), "s1 path_source");
  ensure_capacity(&w.path_nodes, &w.cap_paths_nodes, static_cast<std::size_t>(R), "s1 path_nodes");
  ensure_capacity(&w.path_node_offset, &w.cap_paths_offset, static_cast<std::size_t>(R) + 1U,
                  "s1 path_node_offset");
  ensure_capacity(&w.path_turning_node, &w.cap_turn, static_cast<std::size_t>(R), "s1 turning node");
  ensure_capacity(&w.path_turning_first, &w.cap_turn_first, static_cast<std::size_t>(R),
                  "s1 turning first");
  ensure_capacity(&w.path_caustic, &w.cap_caustic, static_cast<std::size_t>(R), "s1 caustic");
  ensure_capacity(&w.path_excluded, &w.cap_excluded, static_cast<std::size_t>(R), "s1 excluded");
  ensure_capacity(&w.path_drift, &w.cap_drift, static_cast<std::size_t>(R), "s1 drift");
  ensure_capacity(&w.ray_bin, &w.cap_ray_bin, static_cast<std::size_t>(R), "s1 ray_bin");
  ensure_capacity(&w.node_r, &w.cap_node_r, node_cap, "s1 node_r");
  ensure_capacity(&w.node_mu, &w.cap_node_mu, node_cap, "s1 node_mu");
  ensure_capacity(&w.node_theta, &w.cap_node_theta, node_cap, "s1 node_theta");
  ensure_capacity(&w.node_alpha, &w.cap_node_alpha, node_cap, "s1 node_alpha");
  ensure_capacity(&w.node_power, &w.cap_node_power, node_cap, "s1 node_power");
  ensure_capacity(&w.node_area, &w.cap_node_area, node_cap, "s1 node_area");
  // A whole number of 4-byte words (count_crossings_kernel updates the counts by word).
  ensure_capacity(&w.counts, &w.cap_counts, (cross_cap + 3U) & ~static_cast<std::size_t>(3U), "s1 counts");
  ensure_capacity(&w.ranks, &w.cap_ranks, cross_cap, "s1 ranks");
  // One extra zero entry so that the exclusive scan's last output is the total.
  ensure_capacity(&w.bin_count, &w.cap_bin_count, n_bins_table + 1U, "s1 bin_count");
  ensure_capacity(&w.offsets, &w.cap_offsets, n_bins_table + 1U, "s1 offsets");
  ensure_capacity(&w.out_theta, &w.cap_out_theta, cross_cap, "s1 out_theta");
  ensure_capacity(&w.out_alpha, &w.cap_out_alpha, cross_cap, "s1 out_alpha");
  ensure_capacity(&w.out_power, &w.cap_out_power, cross_cap, "s1 out_power");
  ensure_capacity(&w.out_area, &w.cap_out_area, cross_cap, "s1 out_area");
  ensure_capacity(&w.out_ray, &w.cap_out_ray, cross_cap, "s1 out_ray");
  ensure_capacity(&w.out_limiter, &w.cap_out_limiter, cross_cap, "s1 out_limiter");
  ensure_capacity(&w.scalars, &w.cap_scalars, 1U, "s1 scalars");
  if (w.host_scalars == nullptr) {
    s1_check(cudaMallocHost(reinterpret_cast<void**>(&w.host_scalars), sizeof(S1Scalars)),
             "s1 pinned scalars");
  }

  // eps at cell centres and shells
  cell_eps_kernel<<<blocks_for(N), kThreads, 0, stream>>>(N, input.r_edges, input.rho, input.zbar,
                                                          input.A_eff, input.n_crit, w.cell_r_center,
                                                          w.cell_eps);
  shell_eps_kernel<<<blocks_for(n_shells), kThreads, 0, stream>>>(N, input.r_edges, w.cell_r_center,
                                                                  w.cell_eps, w.shell_eps);
  // ray paths
  ray_records_kernel<<<blocks_for(R), kThreads, 0, stream>>>(R, C, N, input.rec_count, input.rec_cell,
                                                             input.rec_ds, w.ray_valid, w.ray_geometric);
  compact_paths_kernel<<<1, 1, 0, stream>>>(R, w.ray_valid, w.ray_geometric, w.path_source,
                                            w.path_nodes, w.path_node_offset, w.scalars);
  build_paths_kernel<<<blocks_for(32LL * R), kThreads, 0, stream>>>(
      R, C, w.scalars, w.path_source, w.path_node_offset, input.rec_count, input.rec_cell,
      input.rec_mu, input.rec_ds, input.rec_S, input.ray_P0, input.r_edges, input.impact_spacing,
      w.node_r, w.node_mu, w.node_theta, w.node_alpha, w.node_power, w.path_turning_node);
  {
    const dim3 grid(static_cast<unsigned>(blocks_for(static_cast<long long>(C) + 1)),
                    static_cast<unsigned>(R));
    path_area_kernel<<<grid, kThreads, 0, stream>>>(R, w.scalars, w.path_nodes, w.path_node_offset,
                                                     w.path_turning_node, w.node_r, w.node_theta,
                                                     w.node_alpha, w.node_area);
  }
  // table
  annotate_paths_kernel<<<blocks_for(32LL * R), kThreads, 0, stream>>>(
      R, w.scalars, w.path_nodes, w.path_node_offset, w.node_r, w.node_alpha, w.node_area,
      input.r_edges, w.shell_eps, n_shells, input.caustic_area_rel_tol, w.path_turning_first,
      w.path_caustic, w.path_excluded, w.path_drift);
  s1_check(cudaMemsetAsync(w.counts, 0, cross_cap * sizeof(std::uint8_t), stream), "s1 counts memset");
  count_crossings_kernel<<<blocks_for(64LL * R), kThreads, 0, stream>>>(
      R, w.scalars, w.path_nodes, w.path_node_offset, w.path_turning_first, w.path_excluded, w.node_r,
      input.r_edges, n_shells, w.counts);
  violation_kernel<<<blocks_for(32LL * R), kThreads, 0, stream>>>(R, w.scalars, n_shells, w.counts,
                                                                 w.path_excluded);
  bin_rank_kernel<<<blocks_for(static_cast<long long>(n_bins_table)), kThreads, 0, stream>>>(
      R, w.scalars, n_shells, w.counts, w.path_excluded, w.ranks, w.bin_count);
  {
    std::size_t temp_bytes = 0;
    s1_check(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, w.bin_count, w.offsets,
                                           static_cast<int>(n_bins_table) + 1, stream),
             "s1 scan size");
    if (temp_bytes > w.scan_temp_bytes) {
      if (w.scan_temp != nullptr) {
        static_cast<void>(cudaFree(w.scan_temp));
        w.scan_temp = nullptr;
      }
      s1_check(cudaMalloc(&w.scan_temp, temp_bytes), "s1 scan temp");
      w.scan_temp_bytes = temp_bytes;
    }
    // The scan covers n_bins + 1 entries: the extra input entry must be zero so that
    // offsets[n_bins] is the total.
    s1_check(cudaMemsetAsync(w.bin_count + n_bins_table, 0, sizeof(int), stream), "s1 bin_count tail");
    s1_check(cub::DeviceScan::ExclusiveSum(w.scan_temp, temp_bytes, w.bin_count, w.offsets,
                                           static_cast<int>(n_bins_table) + 1, stream),
             "s1 scan");
  }
  emit_crossings_kernel<<<blocks_for(64LL * R), kThreads, 0, stream>>>(
      R, w.scalars, w.path_nodes, w.path_node_offset, w.path_turning_first, w.path_caustic,
      w.path_excluded, w.node_r, w.node_theta, w.node_alpha, w.node_power, w.node_area,
      input.r_edges, n_shells, w.ranks, w.offsets, w.out_theta, w.out_alpha, w.out_power, w.out_area,
      w.out_ray, w.out_limiter);
  sort_bins_kernel<<<blocks_for(static_cast<long long>(n_bins_table)), kThreads, 0, stream>>>(
      n_shells, w.offsets, w.out_theta, w.out_alpha, w.out_power, w.out_area, w.out_ray, w.out_limiter);
  summarize_kernel<<<1, 1, 0, stream>>>(R, n_shells, input.n_bins, w.scalars, w.path_source,
                                        w.path_node_offset, w.path_excluded, w.path_drift,
                                        w.node_power, input.ray_group_base, w.offsets, w.ray_bin);
  s1_check(cudaGetLastError(), "s1 kernel launch");
  s1_check(cudaMemcpyAsync(w.host_scalars, w.scalars, sizeof(S1Scalars), cudaMemcpyDeviceToHost, stream),
           "s1 scalars readback");
  s1_check(cudaStreamSynchronize(stream), "s1 stream sync");

  const S1Scalars& s = *w.host_scalars;
  if (table_out != nullptr) {
    table_out->n_shells = n_shells;
    table_out->n_paths = s.n_paths;
    table_out->offsets = w.offsets;
    table_out->theta = w.out_theta;
    table_out->alpha = w.out_alpha;
    table_out->power = w.out_power;
    table_out->area = w.out_area;
    table_out->ray_index = w.out_ray;
    table_out->in_limiter = w.out_limiter;
    table_out->ray_bin = w.ray_bin;
    table_out->shell_r = input.r_edges;
  }
  S1DeviceSummary summary;
  summary.n_paths = s.n_paths;
  summary.n_excluded_rays = s.n_excluded_rays;
  summary.total_nodes = s.total_nodes;
  summary.total_crossings = s.total_crossings;
  summary.excluded_power_fraction = s.excluded_power_fraction;
  summary.bouguer_drift_max = s.bouguer_drift_max;
  summary.error_flags = s.error_flags;
  return summary;
}

void build_ray_map_device(const S1DeviceTable& table, double* map_out, cudaStream_t stream) {
  TENRYU_ASSERT(table.n_shells > 0 && map_out != nullptr, "port_section ray map: empty table");
  ray_map_kernel<<<blocks_for(2LL * table.n_shells), kThreads, 0, stream>>>(
      table.n_shells, table.offsets, table.theta, table.power, table.area, map_out);
  s1_check(cudaGetLastError(), "ray map kernel launch");
}

sector_ps::PhaseSpaceTable host_table_for_shell(const S1DeviceTable& table, const int shell,
                                                cudaStream_t stream) {
  TENRYU_ASSERT(shell >= 0 && shell < table.n_shells, "port_section host table: shell out of range");
  const std::size_t n_bins = 2U * static_cast<std::size_t>(table.n_shells);
  int range[3] = {0, 0, 0};
  s1_check(cudaMemcpyAsync(range, table.offsets + 2 * shell, 3 * sizeof(int), cudaMemcpyDeviceToHost,
                           stream),
           "port_section host table: offsets readback");
  sector_ps::FlatTable flat;
  flat.n_shells = table.n_shells;
  std::vector<double> shell_r(static_cast<std::size_t>(table.n_shells));
  s1_check(cudaMemcpyAsync(shell_r.data(), table.shell_r, shell_r.size() * sizeof(double),
                           cudaMemcpyDeviceToHost, stream),
           "port_section host table: shell radii readback");
  s1_check(cudaStreamSynchronize(stream), "port_section host table: sync");
  const int begin = range[0];
  const int end = range[2];
  const std::size_t count = static_cast<std::size_t>(end - begin);
  flat.theta.resize(count);
  flat.alpha.resize(count);
  flat.P.resize(count);
  flat.area.resize(count);
  flat.ray_index.resize(count);
  flat.in_limiter.resize(count);
  if (count > 0) {
    s1_check(cudaMemcpyAsync(flat.theta.data(), table.theta + begin, count * sizeof(double),
                             cudaMemcpyDeviceToHost, stream),
             "port_section host table: theta readback");
    s1_check(cudaMemcpyAsync(flat.alpha.data(), table.alpha + begin, count * sizeof(double),
                             cudaMemcpyDeviceToHost, stream),
             "port_section host table: alpha readback");
    s1_check(cudaMemcpyAsync(flat.P.data(), table.power + begin, count * sizeof(double),
                             cudaMemcpyDeviceToHost, stream),
             "port_section host table: power readback");
    s1_check(cudaMemcpyAsync(flat.area.data(), table.area + begin, count * sizeof(double),
                             cudaMemcpyDeviceToHost, stream),
             "port_section host table: area readback");
    s1_check(cudaMemcpyAsync(flat.ray_index.data(), table.ray_index + begin,
                             count * sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream),
             "port_section host table: ray index readback");
    s1_check(cudaMemcpyAsync(flat.in_limiter.data(), table.in_limiter + begin,
                             count * sizeof(std::uint8_t), cudaMemcpyDeviceToHost, stream),
             "port_section host table: limiter readback");
    s1_check(cudaStreamSynchronize(stream), "port_section host table: sync");
  }
  // Offsets of a table whose other bins are empty.
  flat.offsets.assign(n_bins + 1U, 0);
  const std::size_t bin0 = 2U * static_cast<std::size_t>(shell);
  const int sheet0 = range[1] - range[0];
  for (std::size_t bin = 0; bin <= n_bins; ++bin) {
    if (bin <= bin0) {
      flat.offsets[bin] = 0;
    } else if (bin == bin0 + 1U) {
      flat.offsets[bin] = sheet0;
    } else {
      flat.offsets[bin] = static_cast<int>(count);
    }
  }
  return sector_ps::table_from_flat(flat, shell_r);
}

}  // namespace tenryu::laser::port_section
