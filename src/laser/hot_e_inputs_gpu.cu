#include "laser/hot_e_inputs_gpu.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "core/constants.hpp"
#include "core/device_pack.hpp"
#include "core/error.hpp"
#include "laser/port_geometry.hpp"

// Compiled with -fmad=false (src/laser/CMakeLists.txt): the arithmetic repeats the host functions
// it replaces (laser.cu laser_step's hot-electron block, hot_e_eta::fit_kappa_abs_um,
// port_section::beam_profile_at_shell, illumination_at_shell, common_wave_drive,
// port_geom::illumination_metrics, build_candidate_axes, common_wave_cluster and
// sector_ps::lookup) operation by operation, so contraction into fused multiply-adds must not
// change it.

namespace tenryu::laser::hot_e_inputs {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kThreads = 128;
// The common-wave and illumination kernels keep a few arrays per port in thread-local memory.
constexpr int kMaxPorts = 192;
constexpr int kFitWindow = 7;  // cells eval_c - 4 .. eval_c + 2

inline void check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

template <typename T>
void ensure(T** ptr, std::size_t* capacity, const std::size_t needed, const char* message) {
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
  check(cudaMalloc(reinterpret_cast<void**>(ptr), needed * sizeof(T)), message);
  *capacity = needed;
}

int blocks_for(const long long n) {
  return static_cast<int>(std::max<long long>(1, (n + kThreads - 1) / kThreads));
}

// std::max(a, b) of the host code (a NaN first argument stays NaN) and std::clamp.
__device__ inline double host_max(const double a, const double b) { return (a < b) ? b : a; }
__device__ inline double host_clamp(const double v, const double lo, const double hi) {
  return (v < lo) ? lo : ((hi < v) ? hi : v);
}

struct Vec3 {
  double x, y, z;
};

__device__ inline double dot3(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

struct DeviceFrame {
  Vec3 e1, e2, b;
};

// ---------------------------------------------------------------------------------------------
// cells

__global__ void cells_kernel(const int n_cells, const double* __restrict__ rho,
                             const double* __restrict__ zbar, const double* __restrict__ A_eff,
                             const double* __restrict__ x_r,
                             const std::uint8_t* __restrict__ cell_is_void,
                             const double proton_mass, const double n_crit_safe,
                             double* __restrict__ n_e, double* __restrict__ n_hat,
                             double* __restrict__ cell_r) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  cell_r[c] = 0.5 * (x_r[c] + x_r[c + 1]);
  double ne = 0.0;
  if (!(cell_is_void != nullptr && cell_is_void[c] != 0U)) {
    ne = host_max(0.0, rho[c]) * host_max(0.0, zbar[c]) / (host_max(A_eff[c], 1.0e-30) * proton_mass);
  }
  n_e[c] = ne;
  n_hat[c] = ne / n_crit_safe;
}

// ---------------------------------------------------------------------------------------------
// evaluation surface, temperature, shell, fitted scale (one thread per channel)

struct EvalResult {
  int valid;
  int eval_c;
  int eval_shell;
  int table_shell;  // eval_shell when the phase-space table has that shell, else -1
  double alpha;
  double Te_s_eV;
  double eval_radius_cm;
  double kappa_um;
};

// hot_e_eta::fit_kappa_abs_um.
__device__ double fit_kappa_abs_um(const double* r_um, const double* ne, const int n,
                                   const double center_um) {
  if (n < 2) {
    return 0.0;
  }
  double spacing_sum = 0.0;
  for (int i = 1; i < n; ++i) {
    spacing_sum += ::fabs(r_um[i] - r_um[i - 1]);
  }
  const double w0 = 2.0 * spacing_sum / static_cast<double>(n - 1);
  int usable = 0;
  double weight_sum = 0.0;
  double weighted_r_sum = 0.0;
  double weighted_log_ne_sum = 0.0;
  for (int i = 0; i < n; ++i) {
    if (!::isfinite(ne[i]) || ne[i] <= 0.0) {
      continue;
    }
    const double d_over_w0 = ::fabs(r_um[i] - center_um) / w0;
    const double weight = ::exp(-(d_over_w0 * d_over_w0));
    const double log_ne = ::log(ne[i]);
    ++usable;
    weight_sum += weight;
    weighted_r_sum += weight * r_um[i];
    weighted_log_ne_sum += weight * log_ne;
  }
  if (usable < 2 || weight_sum == 0.0) {
    return 0.0;
  }
  const double mean_r = weighted_r_sum / weight_sum;
  const double mean_log_ne = weighted_log_ne_sum / weight_sum;
  double variance_r = 0.0;
  double covariance = 0.0;
  for (int i = 0; i < n; ++i) {
    if (!::isfinite(ne[i]) || ne[i] <= 0.0) {
      continue;
    }
    const double d_over_w0 = ::fabs(r_um[i] - center_um) / w0;
    const double weight = ::exp(-(d_over_w0 * d_over_w0));
    const double delta_r = r_um[i] - mean_r;
    variance_r += weight * delta_r * delta_r;
    covariance += weight * delta_r * (::log(ne[i]) - mean_log_ne);
  }
  if (variance_r == 0.0) {
    return 0.0;
  }
  return ::fabs(covariance / variance_r);
}

// laser.cu nearest_shell_index: the lower shell on exact midpoint ties.
__device__ int nearest_shell(const double* shell_r, const int n_shells, const double radius) {
  int lo = 0;
  int hi = n_shells;
  while (lo < hi) {  // std::lower_bound
    const int mid = lo + (hi - lo) / 2;
    if (shell_r[mid] < radius) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const int upper = lo;
  if (upper == 0) {
    return 0;
  }
  if (upper == n_shells) {
    return n_shells - 1;
  }
  const int lower = upper - 1;
  return (radius - shell_r[lower] <= shell_r[upper] - radius) ? lower : upper;
}

// One block per channel. The evaluation cell is the outermost cell c >= 1 whose density falls below
// the target from its inner neighbour (both cells non-void with positive density): the first match
// of the host's inward scan, found by a maximum over the block's threads.
__global__ void eval_kernel(const int n_channels, const int n_cells,
                            const double* __restrict__ eval_fraction,
                            const double n_crit_safe, const double* __restrict__ n_e,
                            const double* __restrict__ cell_r, const double* __restrict__ Te,
                            const std::uint8_t* __restrict__ cell_is_void,
                            const double* __restrict__ x_r, const int table_n_shells,
                            EvalResult* __restrict__ out) {
  __shared__ int sh_cell[kThreads];
  const int ch = static_cast<int>(blockIdx.x);
  if (ch >= n_channels) {
    return;
  }
  const double n_tgt = eval_fraction[ch] * n_crit_safe;
  int found = -1;
  for (int c = n_cells - 1 - static_cast<int>(threadIdx.x); c >= 1; c -= kThreads) {
    const bool void_outer = cell_is_void != nullptr && cell_is_void[c] != 0U;
    const bool void_inner = cell_is_void != nullptr && cell_is_void[c - 1] != 0U;
    if (void_outer || void_inner || !(n_e[c] > 0.0) || !(n_e[c - 1] > 0.0)) {
      continue;
    }
    if (n_e[c] < n_tgt && n_e[c - 1] >= n_tgt) {
      found = c;  // the thread's cells run outward to inward: its first match is its outermost
      break;
    }
  }
  sh_cell[threadIdx.x] = found;
  __syncthreads();
  if (threadIdx.x != 0) {
    return;
  }
  int eval_c = -1;
  for (int i = 0; i < kThreads; ++i) {
    eval_c = ::max(eval_c, sh_cell[i]);
  }
  double alpha = 0.0;
  if (eval_c >= 1) {
    alpha = (n_tgt - n_e[eval_c]) / (n_e[eval_c - 1] - n_e[eval_c]);
  }
  EvalResult result{};
  result.eval_c = eval_c;
  result.eval_shell = -1;
  result.table_shell = -1;
  result.alpha = alpha;
  int valid = (eval_c >= 1) ? 1 : 0;
  if (valid) {
    const int outer = eval_c;
    const int inner = eval_c - 1;
    result.Te_s_eV = (1.0 - alpha) * Te[outer] + alpha * Te[inner];
    result.eval_radius_cm = cell_r[outer] + alpha * (cell_r[inner] - cell_r[outer]);
    result.eval_shell = nearest_shell(x_r, n_cells + 1, result.eval_radius_cm);
    // The table was built on the previous step's faces: the same count for a fixed mesh. A shell
    // outside it is not read (the host checks this after the readback).
    result.table_shell = (result.eval_shell < table_n_shells) ? result.eval_shell : -1;
    const double r_s_um = result.eval_radius_cm * 1.0e4;
    const int fit_begin = ::max(0, eval_c - 4);
    const int fit_end = ::min(n_cells - 1, eval_c + 2);
    double fit_r_um[kFitWindow];
    double fit_ne[kFitWindow];
    int n_fit = 0;
    for (int c = fit_begin; c <= fit_end; ++c) {
      if ((cell_is_void != nullptr && cell_is_void[c] != 0U) || !(n_e[c] > 0.0)) {
        continue;
      }
      fit_r_um[n_fit] = cell_r[c] * 1.0e4;
      fit_ne[n_fit] = n_e[c];
      ++n_fit;
    }
    result.kappa_um = fit_kappa_abs_um(fit_r_um, fit_ne, n_fit, r_s_um);
    if (!(result.kappa_um > 0.0)) {
      valid = 0;
    }
  }
  result.valid = valid;
  out[ch] = result;
}

// ---------------------------------------------------------------------------------------------
// the device phase-space table at one shell

struct ShellBins {
  // sheet s holds crossings [begin[s], end[s]) of the table arrays
  int begin[2];
  int end[2];
};

__device__ ShellBins shell_bins(const int* offsets, const int shell) {
  ShellBins bins;
  for (int sheet = 0; sheet < 2; ++sheet) {
    bins.begin[sheet] = offsets[2 * shell + sheet];
    bins.end[sheet] = offsets[2 * shell + sheet + 1];
  }
  return bins;
}

struct PumpLookup {
  bool valid;
  double I;
  double alpha;
};

// sector_ps::lookup on one sheet of a shell of the device table.
__device__ PumpLookup table_lookup(const double* __restrict__ theta, const double* __restrict__ alpha,
                                   const double* __restrict__ power, const double* __restrict__ area,
                                   const std::uint8_t* __restrict__ in_limiter, const int begin,
                                   const int end, const double theta_p) {
  int first = begin;
  while (first < end && in_limiter[first] != 0U) {
    ++first;
  }
  if (first == end) {
    return PumpLookup{false, 0.0, 0.0};
  }
  int last = end - 1;
  while (in_limiter[last] != 0U) {
    --last;
  }
  if (theta_p > theta[last]) {
    return PumpLookup{false, 0.0, 0.0};
  }
  if (theta_p <= theta[first]) {
    return PumpLookup{true, power[first] / area[first], alpha[first]};
  }
  int hi = first + 1;
  while (hi < end && !(in_limiter[hi] == 0U && theta[hi] >= theta_p)) {
    ++hi;
  }
  // hi < end: the last field crossing has theta >= theta_p.
  if (theta[hi] == theta_p) {
    return PumpLookup{true, power[hi] / area[hi], alpha[hi]};
  }
  int lo = hi;
  do {
    --lo;
  } while (in_limiter[lo] != 0U);
  const double weight = (theta_p - theta[lo]) / (theta[hi] - theta[lo]);
  const double lower_intensity = power[lo] / area[lo];
  const double upper_intensity = power[hi] / area[hi];
  return PumpLookup{true, lower_intensity + weight * (upper_intensity - lower_intensity),
                    alpha[lo] + weight * (alpha[hi] - alpha[lo])};
}

// port_section::beam_profile_at_shell (one thread per channel).
__global__ void profile_kernel(const int n_channels, const int* __restrict__ wants_profile,
                               const EvalResult* __restrict__ eval, const int n_mu_profile,
                               const int* __restrict__ offsets, const double* __restrict__ theta,
                               const double* __restrict__ power, const double* __restrict__ area,
                               const std::uint8_t* __restrict__ in_limiter,
                               double* __restrict__ profile_mu, double* __restrict__ profile_I,
                               double* __restrict__ scratch) {
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= n_channels || wants_profile[ch] == 0 || eval[ch].table_shell < 0) {
    return;
  }
  double* mu = profile_mu + static_cast<long long>(ch) * n_mu_profile;
  double* I = profile_I + static_cast<long long>(ch) * n_mu_profile;
  double* weighted_intensity = scratch + static_cast<long long>(ch) * 2 * n_mu_profile;
  double* bin_power = weighted_intensity + n_mu_profile;
  for (int k = 0; k < n_mu_profile; ++k) {
    mu[k] = -1.0 + 2.0 * static_cast<double>(k) / static_cast<double>(n_mu_profile - 1);
    I[k] = 0.0;
    weighted_intensity[k] = 0.0;
    bin_power[k] = 0.0;
  }
  const ShellBins bins = shell_bins(offsets, eval[ch].table_shell);
  for (int sheet = 0; sheet < 2; ++sheet) {
    for (int i = bins.begin[sheet]; i < bins.end[sheet]; ++i) {
      if (in_limiter[i] != 0U) {
        continue;
      }
      const double mu_i = host_clamp(::cos(theta[i]), -1.0, 1.0);
      const double scaled = 0.5 * (mu_i + 1.0) * static_cast<double>(n_mu_profile - 1);
      const long rounded = ::lround(scaled);
      const int bin = static_cast<int>(
          rounded < 0 ? 0 : (rounded > n_mu_profile - 1 ? n_mu_profile - 1 : rounded));
      const double intensity = power[i] / area[i];
      weighted_intensity[bin] += power[i] * intensity;
      bin_power[bin] += power[i];
    }
  }
  int first = -1;
  int last = -1;
  for (int k = 0; k < n_mu_profile; ++k) {
    if (bin_power[k] != 0.0) {
      I[k] = weighted_intensity[k] / bin_power[k];
      if (first < 0) {
        first = k;
      }
      last = k;
    }
  }
  if (first < 0) {
    return;
  }
  for (int k = 0; k < first; ++k) {
    I[k] = I[first];
  }
  for (int k = last + 1; k < n_mu_profile; ++k) {
    I[k] = I[last];
  }
  int lo = first;
  for (int hi = first + 1; hi <= last; ++hi) {
    if (bin_power[hi] == 0.0) {
      continue;
    }
    for (int k = lo + 1; k < hi; ++k) {
      const double fraction = (mu[k] - mu[lo]) / (mu[hi] - mu[lo]);
      I[k] = I[lo] + fraction * (I[hi] - I[lo]);
    }
    lo = hi;
  }
}

// port_geom interpolate_profile / port_section profile_value (identical).
__device__ double profile_value(const double* mu, const double* I, const int n, const double x) {
  if (x <= mu[0]) {
    return I[0];
  }
  if (x >= mu[n - 1]) {
    return I[n - 1];
  }
  int lo_b = 0;
  int hi_b = n;
  while (lo_b < hi_b) {  // std::upper_bound
    const int mid = lo_b + (hi_b - lo_b) / 2;
    if (mu[mid] <= x) {
      lo_b = mid + 1;
    } else {
      hi_b = mid;
    }
  }
  const int hi = lo_b;
  const int lo = hi - 1;
  const double fraction = (x - mu[lo]) / (mu[hi] - mu[lo]);
  return I[lo] + fraction * (I[hi] - I[lo]);
}

// ---------------------------------------------------------------------------------------------
// illumination (port_geom::illumination_metrics; one thread per grid point, then one ordered sum)

__global__ void illumination_points_kernel(const int ch, const int n_ports,
                                           const DeviceFrame* __restrict__ frames,
                                           const double* __restrict__ port_weight,
                                           const double* __restrict__ grid_mu,
                                           const double* __restrict__ grid_phi, const int n_mu,
                                           const int n_phi, const double* __restrict__ profile_mu,
                                           const double* __restrict__ profile_I, const int n_profile,
                                           const EvalResult* __restrict__ eval,
                                           double* __restrict__ I_tot_out) {
  const int point = blockIdx.x * blockDim.x + threadIdx.x;
  if (point >= n_mu * n_phi || eval[ch].table_shell < 0) {
    return;
  }
  const int i_mu = point / n_phi;
  const int i_phi = point % n_phi;
  const double mu = grid_mu[i_mu];
  const double sin_theta = ::sqrt(host_max(0.0, 1.0 - mu * mu));
  const double phi = grid_phi[i_phi];
  const Vec3 omega{sin_theta * ::cos(phi), sin_theta * ::sin(phi), mu};
  const double* p_mu = profile_mu + static_cast<long long>(ch) * n_profile;
  const double* p_I = profile_I + static_cast<long long>(ch) * n_profile;
  double I_tot = 0.0;
  for (int i = 0; i < n_ports; ++i) {
    const double mu_axis = dot3(frames[i].b, omega);
    I_tot += port_weight[i] * profile_value(p_mu, p_I, n_profile, mu_axis);
  }
  I_tot_out[point] = I_tot;
}

// Two passes of the host: the first gives I_max, the second the union above cut_fraction * I_max.
__global__ void illumination_reduce_kernel(const int ch, const int n_mu, const int n_phi,
                                           const double* __restrict__ grid_wmu,
                                           const double cut_fraction,
                                           const double* __restrict__ I_tot,
                                           const EvalResult* __restrict__ eval,
                                           // [5]: f_illum2, f_union, I_mean, I_max, valid
                                           double* __restrict__ out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  out[4] = 0.0;
  if (eval[ch].table_shell < 0) {
    return;
  }
  out[4] = 1.0;
  const double phi_weight = 2.0 * kPi / static_cast<double>(n_phi);
  const double sphere_area = 4.0 * kPi;
  double I_max_first = -CUDART_INF;
  for (int pass = 0; pass < 2; ++pass) {
    const double I_cut = (pass == 0) ? 0.0 : cut_fraction * I_max_first;
    double integral_I = 0.0;
    double integral_I2 = 0.0;
    double union_solid_angle = 0.0;
    double I_max = -CUDART_INF;
    for (int i_mu = 0; i_mu < n_mu; ++i_mu) {
      const double weight = grid_wmu[i_mu] * phi_weight;
      for (int i_phi = 0; i_phi < n_phi; ++i_phi) {
        const double value = I_tot[i_mu * n_phi + i_phi];
        integral_I += weight * value;
        integral_I2 += weight * value * value;
        if (value > I_cut) {
          union_solid_angle += weight;
        }
        I_max = host_max(I_max, value);
      }
    }
    if (pass == 0) {
      I_max_first = I_max;
    } else {
      out[0] = integral_I * integral_I / (sphere_area * integral_I2);
      out[1] = union_solid_angle / sphere_area;
      out[2] = integral_I / sphere_area;
      out[3] = I_max;
    }
  }
}

// ---------------------------------------------------------------------------------------------
// common-wave drive (one thread per grid point, then one ordered sum)

// port_section lab_dir with sector_ps::meridional_direction.
__device__ Vec3 lab_dir(const DeviceFrame& frame, const double theta, const double phi,
                        const double alpha, const int sheet) {
  const double sin_theta = ::sin(theta);
  const double cos_theta = ::cos(theta);
  const double cos_phi = ::cos(phi);
  const double sin_phi = ::sin(phi);
  const Vec3 r_hat{sin_theta * cos_phi, sin_theta * sin_phi, cos_theta};
  const Vec3 theta_hat{cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta};
  const double radial = (sheet == 0) ? -::cos(alpha) : ::cos(alpha);
  const double tangential = ::sin(alpha);
  const Vec3 local{radial * r_hat.x + tangential * theta_hat.x,
                   radial * r_hat.y + tangential * theta_hat.y,
                   radial * r_hat.z + tangential * theta_hat.z};
  return Vec3{frame.e1.x * local.x + frame.e2.x * local.y + frame.b.x * local.z,
              frame.e1.y * local.x + frame.e2.y * local.y + frame.b.y * local.z,
              frame.e1.z * local.x + frame.e2.z * local.y + frame.b.z * local.z};
}

struct PointResult {
  double I_cw;
  double I_lower;
  double I_upper;
  int n_sigma;
};

__global__ void common_wave_points_kernel(
    const int ch, const int n_ports, const DeviceFrame* __restrict__ frames,
    const double* __restrict__ port_weight, const double* __restrict__ grid_mu,
    const double* __restrict__ grid_phi, const int n_mu, const int n_phi,
    const EvalResult* __restrict__ eval, const int point_begin, const int point_count,
    const int* __restrict__ offsets, const double* __restrict__ theta_t,
    const double* __restrict__ alpha_t,
    const double* __restrict__ power_t, const double* __restrict__ area_t,
    const std::uint8_t* __restrict__ limiter_t, const double* __restrict__ profile_mu,
    const double* __restrict__ profile_I, const int n_profile, const double delta_theta_deg,
    const int max_axes, Vec3* __restrict__ axes_scratch, PointResult* __restrict__ out) {
  const int local = blockIdx.x * blockDim.x + threadIdx.x;
  const int shell = eval[ch].table_shell;
  if (local >= point_count || shell < 0) {
    return;
  }
  const int point = point_begin + local;
  const int i_mu = point / n_phi;
  const int i_phi = point % n_phi;
  const double mu = grid_mu[i_mu];
  const double sin_theta = ::sqrt(host_max(0.0, 1.0 - mu * mu));
  const double phi = grid_phi[i_phi];
  const Vec3 omega{sin_theta * ::cos(phi), sin_theta * ::sin(phi), mu};
  const double* p_mu = profile_mu + static_cast<long long>(ch) * n_profile;
  const double* p_I = profile_I + static_cast<long long>(ch) * n_profile;
  const int begin = offsets[2 * shell];
  const int end = offsets[2 * shell + 1];

  Vec3 k_hat[kMaxPorts];
  double I[kMaxPorts];
  for (int i = 0; i < n_ports; ++i) {
    const DeviceFrame frame = frames[i];
    const double mu_axis = host_clamp(dot3(frame.b, omega), -1.0, 1.0);
    const double theta = ::acos(mu_axis);
    const double phi_port = ::atan2(dot3(omega, frame.e2), dot3(omega, frame.e1));
    const PumpLookup lookup =
        table_lookup(theta_t, alpha_t, power_t, area_t, limiter_t, begin, end, theta);
    k_hat[i] = lab_dir(frame, theta, phi_port, lookup.alpha, 0);
    I[i] = lookup.valid ? port_weight[i] * profile_value(p_mu, p_I, n_profile, mu_axis) : 0.0;
  }

  // port_geom::build_candidate_axes
  Vec3* axes = axes_scratch + static_cast<long long>(local) * max_axes;
  int n_axes = 0;
  axes[n_axes++] = omega;
  for (int i = 0; i < n_ports; ++i) {
    for (int j = i + 1; j < n_ports; ++j) {
      const Vec3 sum{k_hat[i].x + k_hat[j].x, k_hat[i].y + k_hat[j].y, k_hat[i].z + k_hat[j].z};
      const double magnitude = ::sqrt(dot3(sum, sum));
      if (magnitude < 1.0e-6) {
        continue;
      }
      const Vec3 candidate{sum.x / magnitude, sum.y / magnitude, sum.z / magnitude};
      bool duplicate = false;
      for (int a = 0; a < n_axes; ++a) {
        if (::fabs(dot3(candidate, axes[a])) > 1.0 - 1.0e-10) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        axes[n_axes++] = candidate;
      }
    }
  }

  // port_geom::common_wave_cluster
  PointResult result;
  result.I_cw = -CUDART_INF;
  result.I_lower = I[0];
  for (int i = 1; i < n_ports; ++i) {
    if (result.I_lower < I[i]) {  // std::max_element: the first maximum
      result.I_lower = I[i];
    }
  }
  result.I_upper = 0.0;
  for (int i = 0; i < n_ports; ++i) {
    result.I_upper += I[i];
  }
  result.n_sigma = 0;
  const double delta_theta = delta_theta_deg * kPi / 180.0;
  const double dedup_tolerance = delta_theta * 1.0e-3;
  double theta_i[kMaxPorts];
  double sorted[kMaxPorts];
  for (int a = 0; a < n_axes; ++a) {
    for (int i = 0; i < n_ports; ++i) {
      theta_i[i] = ::acos(host_clamp(dot3(k_hat[i], axes[a]), -1.0, 1.0));
      sorted[i] = theta_i[i];
    }
    for (int i = 1; i < n_ports; ++i) {  // ascending
      const double v = sorted[i];
      int j = i - 1;
      while (j >= 0 && sorted[j] > v) {
        sorted[j + 1] = sorted[j];
        --j;
      }
      sorted[j + 1] = v;
    }
    double previous_cone = 0.0;
    bool have_cone = false;
    for (int s = 0; s < n_ports; ++s) {
      const double angle = sorted[s];
      if (have_cone && !(::fabs(angle - previous_cone) > dedup_tolerance)) {
        continue;
      }
      have_cone = true;
      previous_cone = angle;
      double cluster_intensity = 0.0;
      int member_count = 0;
      for (int i = 0; i < n_ports; ++i) {
        if (::fabs(theta_i[i] - angle) <= delta_theta) {
          cluster_intensity += I[i];
          ++member_count;
        }
      }
      if (cluster_intensity > result.I_cw) {
        result.I_cw = cluster_intensity;
        result.n_sigma = member_count;
      }
    }
  }
  out[point] = result;
}

__global__ void common_wave_reduce_kernel(const int ch, const int n_mu, const int n_phi,
                                          const int n_ports, const double* __restrict__ grid_wmu,
                                          const PointResult* __restrict__ points,
                                          const EvalResult* __restrict__ eval,
                                          int* __restrict__ sigma_frequency /* [n_ports + 1] */,
                                          // [5]: I_drive, I_lower, I_upper, n_sigma_mode, valid
                                          double* __restrict__ out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  out[4] = 0.0;
  if (eval[ch].table_shell < 0) {
    return;
  }
  out[4] = 1.0;
  const double phi_weight = 2.0 * kPi / static_cast<double>(n_phi);
  double integral_I_cw = 0.0;
  double integral_I_cw2 = 0.0;
  double integral_lower = 0.0;
  double integral_upper = 0.0;
  for (int k = 0; k <= n_ports; ++k) {
    sigma_frequency[k] = 0;
  }
  for (int i_mu = 0; i_mu < n_mu; ++i_mu) {
    const double weight = grid_wmu[i_mu] * phi_weight;
    for (int i_phi = 0; i_phi < n_phi; ++i_phi) {
      const PointResult& p = points[i_mu * n_phi + i_phi];
      integral_I_cw += weight * p.I_cw;
      integral_I_cw2 += weight * p.I_cw * p.I_cw;
      integral_lower += weight * p.I_lower;
      integral_upper += weight * p.I_upper;
      if (p.I_cw > 0.0) {
        ++sigma_frequency[p.n_sigma];
      }
    }
  }
  int n_sigma_mode = 0;
  int mode_frequency = 0;
  for (int n_sigma = 0; n_sigma <= n_ports; ++n_sigma) {
    if (sigma_frequency[n_sigma] > mode_frequency) {
      mode_frequency = sigma_frequency[n_sigma];
      n_sigma_mode = n_sigma;
    }
  }
  const double sphere_area = 4.0 * kPi;
  out[0] = integral_I_cw == 0.0 ? 0.0 : integral_I_cw2 / integral_I_cw;
  out[1] = integral_lower / sphere_area;
  out[2] = integral_upper / sphere_area;
  out[3] = static_cast<double>(n_sigma_mode);
}

// The sky map of the snapshot: I_tot = I_upper, I_cw and n_sigma per point.
__global__ void sky_map_kernel(const int ch, const int n_points, const EvalResult* __restrict__ eval,
                               const PointResult* __restrict__ points, double* __restrict__ I_tot,
                               double* __restrict__ I_cw, double* __restrict__ n_sigma) {
  const int point = blockIdx.x * blockDim.x + threadIdx.x;
  if (point >= n_points || eval[ch].table_shell < 0) {
    return;
  }
  I_tot[point] = points[point].I_upper;
  I_cw[point] = points[point].I_cw;
  n_sigma[point] = static_cast<double>(points[point].n_sigma);
}

}  // namespace

struct Workspace::Impl {
  int n_cells = 0;
  double* n_e = nullptr;
  double* n_hat = nullptr;
  double* cell_r = nullptr;
  std::size_t cap_cells = 0, cap_cells2 = 0, cap_cells3 = 0;
  // ports
  std::vector<port_geom::PortFrame> frames_host;
  std::vector<double> weights_host;
  DeviceFrame* frames = nullptr;
  double* port_weight = nullptr;
  std::size_t cap_frames = 0, cap_weights = 0;
  int n_ports = 0;
  // channels
  double* eval_fraction = nullptr;
  int* wants_profile = nullptr;
  EvalResult* eval = nullptr;
  std::size_t cap_fraction = 0, cap_wants = 0, cap_eval = 0;
  EvalResult* eval_host = nullptr;  // pinned
  int cap_eval_host = 0;
  // profiles [channel * n_mu_profile]
  double* profile_mu = nullptr;
  double* profile_I = nullptr;
  double* profile_scratch = nullptr;
  std::size_t cap_profile_mu = 0, cap_profile_I = 0, cap_profile_scratch = 0;
  // grids (uploaded once per size)
  double* ill_mu = nullptr;
  double* ill_wmu = nullptr;
  double* ill_phi = nullptr;
  double* cw_mu = nullptr;
  double* cw_wmu = nullptr;
  double* cw_phi = nullptr;
  std::vector<double> cw_mu_host, cw_phi_host;
  // per-point scratch and outputs
  double* ill_I_tot = nullptr;
  std::size_t cap_ill_I_tot = 0;
  Vec3* axes = nullptr;
  std::size_t cap_axes = 0;
  PointResult* cw_points = nullptr;
  std::size_t cap_cw_points = 0;
  int* sigma_frequency = nullptr;
  std::size_t cap_sigma = 0;
  double* reduce_out = nullptr;   // [channel * 8]: illumination [4], common wave [4]
  std::size_t cap_reduce = 0;
  double* reduce_host = nullptr;  // pinned
  int cap_reduce_host = 0;
  // sky map
  double* sky_I_tot = nullptr;
  double* sky_I_cw = nullptr;
  double* sky_n_sigma = nullptr;
  bool sky_valid = false;

  ~Impl() {
    void* ptrs[] = {n_e, n_hat, cell_r, frames, port_weight, eval_fraction, wants_profile, eval,
                    profile_mu, profile_I, profile_scratch, ill_mu, ill_wmu, ill_phi, cw_mu, cw_wmu,
                    cw_phi, ill_I_tot, axes, cw_points, sigma_frequency, reduce_out, sky_I_tot,
                    sky_I_cw, sky_n_sigma};
    for (void* p : ptrs) {
      if (p != nullptr) {
        static_cast<void>(cudaFree(p));
      }
    }
    if (eval_host != nullptr) {
      static_cast<void>(cudaFreeHost(eval_host));
    }
    if (reduce_host != nullptr) {
      static_cast<void>(cudaFreeHost(reduce_host));
    }
  }
};

Workspace::Workspace() : impl_(new Impl) {}
Workspace::~Workspace() { delete impl_; }

namespace {

void upload_grid(const int n_mu, const int n_phi, double** mu, double** wmu, double** phi,
                 std::vector<double>* mu_host, std::vector<double>* phi_host) {
  if (*mu != nullptr) {
    return;
  }
  const port_geom::AngularGrid grid = port_geom::build_angular_grid(n_mu, n_phi);
  check(cudaMalloc(reinterpret_cast<void**>(mu), grid.mu.size() * sizeof(double)), "hot-e grid alloc");
  check(cudaMalloc(reinterpret_cast<void**>(wmu), grid.wmu.size() * sizeof(double)),
        "hot-e grid alloc");
  check(cudaMalloc(reinterpret_cast<void**>(phi), grid.phi.size() * sizeof(double)),
        "hot-e grid alloc");
  check(cudaMemcpy(*mu, grid.mu.data(), grid.mu.size() * sizeof(double), cudaMemcpyHostToDevice),
        "hot-e grid H2D");
  check(cudaMemcpy(*wmu, grid.wmu.data(), grid.wmu.size() * sizeof(double), cudaMemcpyHostToDevice),
        "hot-e grid H2D");
  check(cudaMemcpy(*phi, grid.phi.data(), grid.phi.size() * sizeof(double), cudaMemcpyHostToDevice),
        "hot-e grid H2D");
  if (mu_host != nullptr) {
    *mu_host = grid.mu;
  }
  if (phi_host != nullptr) {
    *phi_host = grid.phi;
  }
}

}  // namespace

void stage_cells(Workspace& ws, const core::State& state, const double* A_eff, const double n_crit,
                 cudaStream_t stream) {
  auto& w = *ws.impl();
  const int n_cells = static_cast<int>(state.rho.size());
  TENRYU_ASSERT(n_cells > 0 && static_cast<int>(state.x_r.size()) == n_cells + 1,
                "hot-e cells: state size mismatch");
  ensure(&w.n_e, &w.cap_cells, static_cast<std::size_t>(n_cells), "hot-e n_e alloc");
  ensure(&w.n_hat, &w.cap_cells2, static_cast<std::size_t>(n_cells), "hot-e n_hat alloc");
  ensure(&w.cell_r, &w.cap_cells3, static_cast<std::size_t>(n_cells), "hot-e cell_r alloc");
  w.n_cells = n_cells;
  const std::uint8_t* void_mask =
      state.cell_is_void.empty() ? nullptr : core::device_cell_is_void(state.cell_is_void);
  cells_kernel<<<blocks_for(n_cells), kThreads, 0, stream>>>(
      n_cells, state.rho.data(), state.zbar.data(), A_eff, state.x_r.data(), void_mask,
      core::constants::proton_mass, std::max(n_crit, 1.0e-30), w.n_e, w.n_hat, w.cell_r);
  check(cudaGetLastError(), "hot-e cells launch");
}

const double* cell_n_hat(const Workspace& ws) { return ws.impl()->n_hat; }
const double* cell_r(const Workspace& ws) { return ws.impl()->cell_r; }

void stage_ports(Workspace& ws, const port_geom::PortTable& ports) {
  auto& w = *ws.impl();
  const int n_ports = static_cast<int>(ports.ports.size());
  TENRYU_ASSERT(n_ports > 0 && n_ports <= kMaxPorts, "hot-e inputs: port count out of range");
  std::vector<double> weights(static_cast<std::size_t>(n_ports));
  for (int i = 0; i < n_ports; ++i) {
    weights[static_cast<std::size_t>(i)] = ports.ports[static_cast<std::size_t>(i)].power_weight;
  }
  const bool same = w.n_ports == n_ports && w.weights_host == weights &&
                    w.frames_host.size() == ports.frames.size() &&
                    std::memcmp(w.frames_host.data(), ports.frames.data(),
                                ports.frames.size() * sizeof(port_geom::PortFrame)) == 0;
  if (same) {
    return;
  }
  std::vector<DeviceFrame> frames(static_cast<std::size_t>(n_ports));
  for (int i = 0; i < n_ports; ++i) {
    const auto& f = ports.frames[static_cast<std::size_t>(i)];
    frames[static_cast<std::size_t>(i)] = DeviceFrame{Vec3{f.e1[0], f.e1[1], f.e1[2]},
                                                      Vec3{f.e2[0], f.e2[1], f.e2[2]},
                                                      Vec3{f.b[0], f.b[1], f.b[2]}};
  }
  ensure(&w.frames, &w.cap_frames, frames.size(), "hot-e frames alloc");
  ensure(&w.port_weight, &w.cap_weights, weights.size(), "hot-e weights alloc");
  check(cudaMemcpy(w.frames, frames.data(), frames.size() * sizeof(DeviceFrame),
                   cudaMemcpyHostToDevice),
        "hot-e frames H2D");
  check(cudaMemcpy(w.port_weight, weights.data(), weights.size() * sizeof(double), cudaMemcpyHostToDevice),
        "hot-e weights H2D");
  w.frames_host = ports.frames;
  w.weights_host = weights;
  w.n_ports = n_ports;
}

std::vector<ChannelResult> evaluate_channels(Workspace& ws, const core::State& state,
                                             const std::vector<ChannelSpec>& specs,
                                             const port_section::S1DeviceTable* table,
                                             const double n_crit, cudaStream_t stream) {
  auto& w = *ws.impl();
  w.sky_valid = false;  // set again below when this call computes the sky map
  const int n_channels = static_cast<int>(specs.size());
  std::vector<ChannelResult> results(specs.size());
  if (n_channels == 0) {
    return results;
  }
  TENRYU_ASSERT(w.n_e != nullptr && w.n_cells == static_cast<int>(state.rho.size()),
                "hot-e channels before stage_cells");
  const int n_cells = w.n_cells;
  std::vector<double> fractions(specs.size());
  std::vector<int> wants(specs.size(), 0);
  for (std::size_t c = 0; c < specs.size(); ++c) {
    fractions[c] = specs[c].eval_nc_fraction;
    const bool drive = specs[c].common_wave || specs[c].sky_map;
    wants[c] = (table != nullptr && (specs[c].illumination || drive)) ? 1 : 0;
  }
  ensure(&w.eval_fraction, &w.cap_fraction, fractions.size(), "hot-e fraction alloc");
  ensure(&w.wants_profile, &w.cap_wants, wants.size(), "hot-e wants alloc");
  ensure(&w.eval, &w.cap_eval, specs.size(), "hot-e eval alloc");
  check(cudaMemcpyAsync(w.eval_fraction, fractions.data(), fractions.size() * sizeof(double),
                        cudaMemcpyHostToDevice, stream),
        "hot-e fraction H2D");
  check(cudaMemcpyAsync(w.wants_profile, wants.data(), wants.size() * sizeof(int), cudaMemcpyHostToDevice,
                        stream),
        "hot-e wants H2D");
  const std::uint8_t* void_mask =
      state.cell_is_void.empty() ? nullptr : core::device_cell_is_void(state.cell_is_void);
  eval_kernel<<<std::max(n_channels, 1), kThreads, 0, stream>>>(
      n_channels, n_cells, w.eval_fraction, std::max(n_crit, 1.0e-30), w.n_e, w.cell_r,
      state.Te.data(), void_mask, state.x_r.data(), table != nullptr ? table->n_shells : 0,
      w.eval);
  check(cudaGetLastError(), "hot-e eval launch");

  constexpr int kOut = 10;  // [0..4] illumination, [5..9] common-wave drive
  ensure(&w.reduce_out, &w.cap_reduce, static_cast<std::size_t>(n_channels) * kOut, "hot-e reduce alloc");
  check(cudaMemsetAsync(w.reduce_out, 0, static_cast<std::size_t>(n_channels) * kOut * sizeof(double), stream),
        "hot-e reduce reset");
  const bool any_profile = std::any_of(wants.begin(), wants.end(), [](const int v) { return v != 0; });
  if (any_profile) {
    TENRYU_ASSERT(w.n_ports > 0 && w.frames != nullptr, "hot-e channels before stage_ports");
    TENRYU_ASSERT(kIlluminationMuProfile == kCommonWaveMuProfile,
                  "hot-e profile sizes differ between the illumination and the drive");
    const int n_profile = kIlluminationMuProfile;
    const std::size_t profile_size = static_cast<std::size_t>(n_channels) * n_profile;
    ensure(&w.profile_mu, &w.cap_profile_mu, profile_size, "hot-e profile alloc");
    ensure(&w.profile_I, &w.cap_profile_I, profile_size, "hot-e profile alloc");
    ensure(&w.profile_scratch, &w.cap_profile_scratch, 2U * profile_size, "hot-e profile alloc");
    profile_kernel<<<blocks_for(n_channels), kThreads, 0, stream>>>(
        n_channels, w.wants_profile, w.eval, n_profile, table->offsets, table->theta, table->power,
        table->area, table->in_limiter, w.profile_mu, w.profile_I, w.profile_scratch);
    check(cudaGetLastError(), "hot-e profile launch");
    for (int ch = 0; ch < n_channels; ++ch) {
      const ChannelSpec& spec = specs[static_cast<std::size_t>(ch)];
      if (spec.illumination) {
        upload_grid(kIlluminationMuGrid, kIlluminationPhiGrid, &w.ill_mu, &w.ill_wmu, &w.ill_phi, nullptr,
                    nullptr);
        const int n_points = kIlluminationMuGrid * kIlluminationPhiGrid;
        ensure(&w.ill_I_tot, &w.cap_ill_I_tot, static_cast<std::size_t>(n_points), "hot-e illumination alloc");
        illumination_points_kernel<<<blocks_for(n_points), kThreads, 0, stream>>>(
            ch, w.n_ports, w.frames, w.port_weight, w.ill_mu, w.ill_phi, kIlluminationMuGrid,
            kIlluminationPhiGrid, w.profile_mu, w.profile_I, n_profile, w.eval, w.ill_I_tot);
        check(cudaGetLastError(), "hot-e illumination launch");
        illumination_reduce_kernel<<<1, 1, 0, stream>>>(
            ch, kIlluminationMuGrid, kIlluminationPhiGrid, w.ill_wmu, kIlluminationCutFraction,
            w.ill_I_tot, w.eval, w.reduce_out + static_cast<std::size_t>(ch) * kOut);
        check(cudaGetLastError(), "hot-e illumination reduce launch");
      }
      if (spec.common_wave || spec.sky_map) {
        upload_grid(kCommonWaveMuGrid, kCommonWavePhiGrid, &w.cw_mu, &w.cw_wmu, &w.cw_phi, &w.cw_mu_host,
                    &w.cw_phi_host);
        const int n_points = kCommonWaveMuGrid * kCommonWavePhiGrid;
        const int max_axes = 1 + w.n_ports * (w.n_ports - 1) / 2;
        // Candidate axes of a batch of points in one scratch of at most ~64 MB.
        const std::size_t per_point = static_cast<std::size_t>(max_axes) * sizeof(Vec3);
        const int batch = static_cast<int>(std::max<std::size_t>(
            1U, std::min<std::size_t>(static_cast<std::size_t>(n_points), (64U << 20) / per_point)));
        ensure(&w.axes, &w.cap_axes, static_cast<std::size_t>(batch) * max_axes, "hot-e axes alloc");
        ensure(&w.cw_points, &w.cap_cw_points, static_cast<std::size_t>(n_points), "hot-e points alloc");
        for (int begin = 0; begin < n_points; begin += batch) {
          const int count = std::min(batch, n_points - begin);
          common_wave_points_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
              ch, w.n_ports, w.frames, w.port_weight, w.cw_mu, w.cw_phi, kCommonWaveMuGrid,
              kCommonWavePhiGrid, w.eval, begin, count, table->offsets, table->theta, table->alpha,
              table->power, table->area, table->in_limiter, w.profile_mu, w.profile_I, n_profile,
              spec.delta_theta_deg, max_axes, w.axes, w.cw_points);
          check(cudaGetLastError(), "hot-e common-wave launch");
        }
        ensure(&w.sigma_frequency, &w.cap_sigma, static_cast<std::size_t>(w.n_ports) + 1U,
               "hot-e sigma alloc");
        common_wave_reduce_kernel<<<1, 1, 0, stream>>>(
            ch, kCommonWaveMuGrid, kCommonWavePhiGrid, w.n_ports, w.cw_wmu, w.cw_points, w.eval,
            w.sigma_frequency, w.reduce_out + static_cast<std::size_t>(ch) * kOut + 5);
        check(cudaGetLastError(), "hot-e common-wave reduce launch");
        if (spec.sky_map) {
          std::size_t cap = 0;
          if (w.sky_I_tot == nullptr) {
            ensure(&w.sky_I_tot, &cap, static_cast<std::size_t>(n_points), "hot-e sky alloc");
            cap = 0;
            ensure(&w.sky_I_cw, &cap, static_cast<std::size_t>(n_points), "hot-e sky alloc");
            cap = 0;
            ensure(&w.sky_n_sigma, &cap, static_cast<std::size_t>(n_points), "hot-e sky alloc");
          }
          sky_map_kernel<<<blocks_for(n_points), kThreads, 0, stream>>>(
              ch, n_points, w.eval, w.cw_points, w.sky_I_tot, w.sky_I_cw, w.sky_n_sigma);
          check(cudaGetLastError(), "hot-e sky map launch");
        }
      }
    }
  }

  if (w.cap_eval_host < n_channels) {
    if (w.eval_host != nullptr) {
      static_cast<void>(cudaFreeHost(w.eval_host));
      w.eval_host = nullptr;
    }
    check(cudaMallocHost(reinterpret_cast<void**>(&w.eval_host),
                         static_cast<std::size_t>(n_channels) * sizeof(EvalResult)),
          "hot-e eval pinned alloc");
    w.cap_eval_host = n_channels;
  }
  if (w.cap_reduce_host < n_channels * kOut) {
    if (w.reduce_host != nullptr) {
      static_cast<void>(cudaFreeHost(w.reduce_host));
      w.reduce_host = nullptr;
    }
    check(cudaMallocHost(reinterpret_cast<void**>(&w.reduce_host),
                         static_cast<std::size_t>(n_channels) * kOut * sizeof(double)),
          "hot-e reduce pinned alloc");
    w.cap_reduce_host = n_channels * kOut;
  }
  check(cudaMemcpyAsync(w.eval_host, w.eval, static_cast<std::size_t>(n_channels) * sizeof(EvalResult),
                        cudaMemcpyDeviceToHost, stream),
        "hot-e eval D2H");
  check(cudaMemcpyAsync(w.reduce_host, w.reduce_out,
                        static_cast<std::size_t>(n_channels) * kOut * sizeof(double), cudaMemcpyDeviceToHost,
                        stream),
        "hot-e reduce D2H");
  check(cudaStreamSynchronize(stream), "hot-e readback sync");

  for (int ch = 0; ch < n_channels; ++ch) {
    const EvalResult& e = w.eval_host[ch];
    const double* out = w.reduce_host + static_cast<std::size_t>(ch) * kOut;
    ChannelResult& r = results[static_cast<std::size_t>(ch)];
    r.valid = e.valid != 0;
    r.Te_s_eV = e.Te_s_eV;
    r.kappa_um = e.kappa_um;
    r.eval_radius_cm = e.eval_radius_cm;
    r.eval_shell = e.eval_shell;
    const ChannelSpec& spec = specs[static_cast<std::size_t>(ch)];
    TENRYU_ASSERT(
        wants[static_cast<std::size_t>(ch)] == 0 || e.eval_shell < 0 || e.table_shell >= 0,
        "hot-e inputs: evaluation shell outside the phase-space table");
    if (table != nullptr && spec.illumination && out[4] != 0.0) {
      r.illumination_valid = true;
      r.f_illum2 = out[0];
      r.f_union = out[1];
    }
    if (table != nullptr && (spec.common_wave || spec.sky_map) && out[9] != 0.0) {
      r.drive_computed = true;
      r.I_drive = out[5];
      r.I_lower = out[6];
      r.I_upper = out[7];
      r.n_sigma_mode = static_cast<int>(out[8]);
      if (spec.sky_map) {
        w.sky_valid = true;
      }
    }
  }
  return results;
}

bool copy_sky_map(const Workspace& ws, double* I_tot, double* I_cw, double* n_sigma,
                  cudaStream_t stream) {
  const auto& w = *ws.impl();
  if (!w.sky_valid) {
    return false;
  }
  const std::size_t n_points = w.cw_mu_host.size() * w.cw_phi_host.size();
  check(cudaMemcpyAsync(I_tot, w.sky_I_tot, n_points * sizeof(double), cudaMemcpyDeviceToDevice, stream),
        "hot-e sky I_tot D2D");
  check(cudaMemcpyAsync(I_cw, w.sky_I_cw, n_points * sizeof(double), cudaMemcpyDeviceToDevice, stream),
        "hot-e sky I_cw D2D");
  check(cudaMemcpyAsync(n_sigma, w.sky_n_sigma, n_points * sizeof(double), cudaMemcpyDeviceToDevice,
                        stream),
        "hot-e sky n_sigma D2D");
  return true;
}

const std::vector<double>& sky_mu(const Workspace& ws) { return ws.impl()->cw_mu_host; }
const std::vector<double>& sky_phi(const Workspace& ws) { return ws.impl()->cw_phi_host; }

}  // namespace tenryu::laser::hot_e_inputs
