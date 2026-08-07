#include "hydro/corner_jacobian_quality.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "hydro/central_pseudo_core.cuh"
#include "hydro/mesh_regime.cuh"
#include "hydro/pole_angular_derefine.cuh"
#include "hydro/rz_corner_mass.cuh"
#include "parallel/reduction.hpp"

namespace tenryu::hydro {
namespace {

constexpr double kCornerJacobianRatioEps = 1.0e-300;
constexpr double kInHydroCornerJQuadraticDegeneracyEps = 1.0e-12;
constexpr double kInHydroCornerJDtSafety = 0.7;
constexpr double kStage24TelemetryEpsRel = 1.0e-6;
constexpr int kNoFailingCornerKey = std::numeric_limits<int>::max();
constexpr int kInHydroGaussSlotOffset = 4;
constexpr int kInHydroRzSlot = 8;
constexpr int kInHydroAxisSlot = 9;
constexpr int kMeshQualityGaussSlotOffset = 4;
constexpr int kMeshQualityRzSlot = 8;
constexpr int kMeshQualityAxisSlot = 9;
constexpr int kCompositeDetailSlotBits = 8;
constexpr unsigned int kCompositeDetailSlotMask = 0xffU;
constexpr unsigned long long kNoFailingCompositeKey =
    std::numeric_limits<unsigned long long>::max();

struct InHydroCornerJRoot {
  double sigma = 1.0;
  bool found = false;
};

__host__ __device__ inline int composite_detail_key(const int cell,
                                                    const int slot) {
  return ((cell & 0x00ffffff) << kCompositeDetailSlotBits) |
         (slot & static_cast<int>(kCompositeDetailSlotMask));
}

inline int composite_detail_cell(const std::uint32_t detail_key) {
  return static_cast<int>(detail_key >> kCompositeDetailSlotBits);
}

inline int composite_detail_slot(const std::uint32_t detail_key) {
  return static_cast<int>(detail_key & kCompositeDetailSlotMask);
}

inline double sigma_from_sortable_uint32(const std::uint32_t sigma_bits) {
  float sigma = 0.0F;
  std::memcpy(&sigma, &sigma_bits, sizeof(float));
  return static_cast<double>(sigma);
}

__host__ __device__ inline int in_hydro_corner_failure_key(const int cell,
                                                           const int corner) {
  return composite_detail_key(cell, corner);
}

__host__ __device__ inline int in_hydro_gauss_failure_key(const int cell,
                                                          const int gauss) {
  return composite_detail_key(cell, kInHydroGaussSlotOffset + gauss);
}

__host__ __device__ inline int in_hydro_rz_failure_key(const int cell) {
  return composite_detail_key(cell, kInHydroRzSlot);
}

__host__ __device__ inline int in_hydro_axis_failure_key(const int cell) {
  return composite_detail_key(cell, kInHydroAxisSlot);
}

__host__ __device__ inline int mesh_quality_corner_failure_key(const int cell,
                                                               const int corner) {
  return composite_detail_key(cell, corner);
}

__host__ __device__ inline int mesh_quality_gauss_failure_key(const int cell,
                                                              const int gauss) {
  return composite_detail_key(cell, kMeshQualityGaussSlotOffset + gauss);
}

__host__ __device__ inline int mesh_quality_rz_failure_key(const int cell) {
  return composite_detail_key(cell, kMeshQualityRzSlot);
}

__host__ __device__ inline int mesh_quality_axis_failure_key(const int cell) {
  return composite_detail_key(cell, kMeshQualityAxisSlot);
}

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline bool stage24_telemetry_enabled() {
  static const bool enabled = []() {
    const char* raw = std::getenv("TENRYU_STAGE24_TELEMETRY");
    if (raw == nullptr || raw[0] == '\0') {
      return false;
    }
    return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0 &&
           std::strcmp(raw, "FALSE") != 0;
  }();
  return enabled;
}

inline bool stage24_axis_classification_active(const core::Config& cfg) {
  return stage24_telemetry_enabled() ||
         cfg.numerics.hydro.dispatcher_state_sensitive_bypass_enabled;
}

__device__ double atomic_min_double(double* address, const double val) {
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  while (val < __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long>(__double_as_longlong(val)));
    if (assumed == old) {
      break;
    }
  }
  return __longlong_as_double(static_cast<long long>(old));
}

__device__ double atomic_max_double(double* address, const double val) {
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  while (val > __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long>(__double_as_longlong(val)));
    if (assumed == old) {
      break;
    }
  }
  return __longlong_as_double(static_cast<long long>(old));
}

__device__ __forceinline__ unsigned int sigma_to_sortable_uint32_device(
    const double sigma_in) {
  double sigma = sigma_in;
  if (!isfinite(sigma) || sigma < 0.0) {
    sigma = 0.0;
  }
  sigma = fmin(sigma, 1.0);
  return __float_as_uint(static_cast<float>(sigma));
}

__device__ __forceinline__ unsigned long long compose_min_key(
    const double sigma,
    const int detail_key) {
  return (static_cast<unsigned long long>(
              sigma_to_sortable_uint32_device(sigma))
          << 32) |
         static_cast<unsigned int>(detail_key < 0 ? 0 : detail_key);
}

__device__ __forceinline__ void atomic_min_u64(
    unsigned long long* address,
    const unsigned long long value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 350
  unsigned long long old = *address;
  while (value < old) {
    const unsigned long long assumed = old;
    old = atomicCAS(address, assumed, value);
    if (assumed == old) {
      break;
    }
  }
#else
  atomicMin(address, value);
#endif
}

__device__ __forceinline__ void record_composite_failure(
    double* __restrict__ min_sigma,
    unsigned long long* __restrict__ first_failing_composite_key,
    const double sigma_in,
    const int detail_key) {
  double sigma = sigma_in;
  if (!isfinite(sigma) || sigma < 0.0) {
    sigma = 0.0;
  }
  sigma = fmin(sigma, 1.0);
  atomic_min_double(min_sigma, sigma);
  atomic_min_u64(first_failing_composite_key,
                 compose_min_key(sigma, detail_key));
}

__device__ __forceinline__ double in_hydro_corner_j_at_sigma(const double j0,
                                                             const double j1,
                                                             const double j2,
                                                             const double sigma) {
  return j0 + sigma * j1 + sigma * sigma * j2;
}

__device__ __forceinline__ double in_hydro_corner_j_min_on_unit_interval(
    const double j0,
    const double j1,
    const double j2) {
  double min_j = fmin(j0, in_hydro_corner_j_at_sigma(j0, j1, j2, 1.0));
  if (j2 > 0.0) {
    const double sigma_vertex = -j1 / (2.0 * j2);
    if (sigma_vertex > 0.0 && sigma_vertex < 1.0) {
      min_j = fmin(min_j, in_hydro_corner_j_at_sigma(j0, j1, j2, sigma_vertex));
    }
  }
  return min_j;
}

__device__ __forceinline__ InHydroCornerJRoot
first_in_hydro_corner_j_floor_root(const double j0,
                                   const double j1,
                                   const double j2,
                                   const double floor) {
  const double scale =
      fmax(fmax(fabs(j0), fabs(j1)), fmax(fabs(floor), 1.0e-300));
  if (fabs(j2) <= kInHydroCornerJQuadraticDegeneracyEps * scale) {
    if (j1 >= 0.0 && j0 >= floor) {
      return InHydroCornerJRoot{};
    }
    if (j1 == 0.0) {
      return InHydroCornerJRoot{};
    }
    const double sigma = (floor - j0) / j1;
    if (isfinite(sigma) && sigma > 0.0 && sigma <= 1.0) {
      return InHydroCornerJRoot{sigma, true};
    }
    return InHydroCornerJRoot{};
  }

  const double discriminant = j1 * j1 - 4.0 * j2 * (j0 - floor);
  const double disc_scale = fmax(j1 * j1, fabs(4.0 * j2 * (j0 - floor)));
  if (discriminant < -kInHydroCornerJQuadraticDegeneracyEps * disc_scale) {
    return InHydroCornerJRoot{};
  }
  const double sqrt_disc = sqrt(fmax(0.0, discriminant));
  const double denom = 2.0 * j2;
  const double root_a = (-j1 - sqrt_disc) / denom;
  const double root_b = (-j1 + sqrt_disc) / denom;
  double sigma = 2.0;
  if (isfinite(root_a) && root_a > 0.0 && root_a <= 1.0) {
    sigma = root_a;
  }
  if (isfinite(root_b) && root_b > 0.0 && root_b <= 1.0) {
    sigma = fmin(sigma, root_b);
  }
  if (sigma <= 1.0) {
    return InHydroCornerJRoot{sigma, true};
  }
  return InHydroCornerJRoot{};
}

__device__ __forceinline__ InHydroCornerJRoot
first_in_hydro_rz_volume_floor_root(const double* rr0,
                                    const double* zz0,
                                    const double* rr1,
                                    const double* zz1,
                                    const double floor) {
  double a;
  double b;
  double c;
  double d;
  in_hydro_rz_volume_cubic_coefficients(rr0, zz0, rr1, zz1, a, b, c, d);
  if (!isfinite(a) || !isfinite(b) || !isfinite(c) || !isfinite(d)) {
    return InHydroCornerJRoot{0.0, true};
  }

  double roots[2] = {2.0, 2.0};
  int n_roots = 0;
  const double qa = 3.0 * d;
  const double qb = 2.0 * c;
  const double qc = b;
  const double scale = fmax(fmax(fabs(qa), fabs(qb)),
                            fmax(fabs(qc), 1.0e-300));
  if (fabs(qa) <= kInHydroCornerJQuadraticDegeneracyEps * scale) {
    if (fabs(qb) > kInHydroCornerJQuadraticDegeneracyEps * scale) {
      const double root = -qc / qb;
      if (isfinite(root) && root > 0.0 && root < 1.0) {
        roots[n_roots++] = root;
      }
    }
  } else {
    const double disc = qb * qb - 4.0 * qa * qc;
    const double disc_scale = fmax(qb * qb, fabs(4.0 * qa * qc));
    if (disc >= -kInHydroCornerJQuadraticDegeneracyEps * disc_scale) {
      const double sqrt_disc = sqrt(fmax(0.0, disc));
      const double root_a = (-qb - sqrt_disc) / (2.0 * qa);
      const double root_b = (-qb + sqrt_disc) / (2.0 * qa);
      if (isfinite(root_a) && root_a > 0.0 && root_a < 1.0) {
        roots[n_roots++] = root_a;
      }
      if (isfinite(root_b) && root_b > 0.0 && root_b < 1.0) {
        roots[n_roots++] = root_b;
      }
    }
  }
  if (n_roots == 2 && roots[1] < roots[0]) {
    const double tmp = roots[0];
    roots[0] = roots[1];
    roots[1] = tmp;
  }

  double lo = 0.0;
  double f_lo = a - floor;
  for (int idx = 0; idx <= n_roots; ++idx) {
    const double hi = (idx == n_roots) ? 1.0 : roots[idx];
    const double f_hi = in_hydro_cubic_at_sigma(a, b, c, d, hi) - floor;
    if (f_lo > 0.0 && f_hi <= 0.0) {
      double left = lo;
      double right = hi;
      for (int iter = 0; iter < 48; ++iter) {
        const double mid = 0.5 * (left + right);
        const double f_mid = in_hydro_cubic_at_sigma(a, b, c, d, mid) - floor;
        if (f_mid > 0.0) {
          left = mid;
        } else {
          right = mid;
        }
      }
      return InHydroCornerJRoot{right, true};
    }
    lo = hi;
    f_lo = f_hi;
  }
  return InHydroCornerJRoot{};
}

__device__ __forceinline__ double multiblock_oriented_rz_volume_at_sigma(
    const double* rr0,
    const double* zz0,
    const double* rr1,
    const double* zz1,
    const int nverts,
    const double orientation_sign,
    const double sigma) {
  double rr[4] = {0.0, 0.0, 0.0, 0.0};
  double zz[4] = {0.0, 0.0, 0.0, 0.0};
  for (int k = 0; k < nverts; ++k) {
    rr[k] = rr0[k] + sigma * (rr1[k] - rr0[k]);
    zz[k] = zz0[k] + sigma * (zz1[k] - zz0[k]);
  }
  return orientation_sign * rz::rz_polygon_volume_exact(rr, zz, nverts);
}

__device__ __forceinline__ void
multiblock_oriented_rz_volume_cubic_coefficients(const double* rr0,
                                                 const double* zz0,
                                                 const double* rr1,
                                                 const double* zz1,
                                                 const int nverts,
                                                 const double orientation_sign,
                                                 double& a,
                                                 double& b,
                                                 double& c,
                                                 double& d) {
  a = multiblock_oriented_rz_volume_at_sigma(
      rr0, zz0, rr1, zz1, nverts, orientation_sign, 0.0);
  const double v13 = multiblock_oriented_rz_volume_at_sigma(
      rr0, zz0, rr1, zz1, nverts, orientation_sign, 1.0 / 3.0);
  const double v23 = multiblock_oriented_rz_volume_at_sigma(
      rr0, zz0, rr1, zz1, nverts, orientation_sign, 2.0 / 3.0);
  const double v1 = multiblock_oriented_rz_volume_at_sigma(
      rr0, zz0, rr1, zz1, nverts, orientation_sign, 1.0);
  const double A = v13 - a;
  const double B = v23 - a;
  const double C = v1 - a;
  d = 0.5 * (9.0 * C + 27.0 * A - 27.0 * B);
  c = (9.0 * C - 27.0 * A - 8.0 * d) / 6.0;
  b = C - c - d;
}

__device__ __forceinline__ InHydroCornerJRoot
first_multiblock_oriented_rz_volume_floor_root(const double* rr0,
                                               const double* zz0,
                                               const double* rr1,
                                               const double* zz1,
                                               const int nverts,
                                               const double orientation_sign,
                                               const double floor) {
  double a;
  double b;
  double c;
  double d;
  multiblock_oriented_rz_volume_cubic_coefficients(
      rr0, zz0, rr1, zz1, nverts, orientation_sign, a, b, c, d);
  if (!in_hydro_isfinite(a) || !in_hydro_isfinite(b) ||
      !in_hydro_isfinite(c) || !in_hydro_isfinite(d)) {
    return InHydroCornerJRoot{0.0, true};
  }

  double roots[2] = {2.0, 2.0};
  int n_roots = 0;
  const double qa = 3.0 * d;
  const double qb = 2.0 * c;
  const double qc = b;
  const double scale = fmax(fmax(fabs(qa), fabs(qb)),
                            fmax(fabs(qc), 1.0e-300));
  if (fabs(qa) <= kInHydroCornerJQuadraticDegeneracyEps * scale) {
    if (fabs(qb) > kInHydroCornerJQuadraticDegeneracyEps * scale) {
      const double root = -qc / qb;
      if (in_hydro_isfinite(root) && root > 0.0 && root < 1.0) {
        roots[n_roots++] = root;
      }
    }
  } else {
    const double disc = qb * qb - 4.0 * qa * qc;
    const double disc_scale = fmax(qb * qb, fabs(4.0 * qa * qc));
    if (disc >= -kInHydroCornerJQuadraticDegeneracyEps * disc_scale) {
      const double sqrt_disc = sqrt(fmax(0.0, disc));
      const double root_a = (-qb - sqrt_disc) / (2.0 * qa);
      const double root_b = (-qb + sqrt_disc) / (2.0 * qa);
      if (in_hydro_isfinite(root_a) && root_a > 0.0 && root_a < 1.0) {
        roots[n_roots++] = root_a;
      }
      if (in_hydro_isfinite(root_b) && root_b > 0.0 && root_b < 1.0) {
        roots[n_roots++] = root_b;
      }
    }
  }
  if (n_roots == 2 && roots[1] < roots[0]) {
    const double tmp = roots[0];
    roots[0] = roots[1];
    roots[1] = tmp;
  }

  double lo = 0.0;
  double f_lo = a - floor;
  for (int idx = 0; idx <= n_roots; ++idx) {
    const double hi = (idx == n_roots) ? 1.0 : roots[idx];
    const double f_hi = in_hydro_cubic_at_sigma(a, b, c, d, hi) - floor;
    if (f_lo > 0.0 && f_hi <= 0.0) {
      double left = lo;
      double right = hi;
      for (int iter = 0; iter < 48; ++iter) {
        const double mid = 0.5 * (left + right);
        const double f_mid = in_hydro_cubic_at_sigma(a, b, c, d, mid) - floor;
        if (f_mid > 0.0) {
          left = mid;
        } else {
          right = mid;
        }
      }
      return InHydroCornerJRoot{right, true};
    }
    lo = hi;
    f_lo = f_hi;
  }
  return InHydroCornerJRoot{};
}

template <typename Field>
std::vector<double> copy_field_to_vector(const Field& field) {
  std::vector<double> host(field.size(), 0.0);
  field.copy_to_host(host.data());
  return host;
}

double corner_jacobian_at_tau_host(const double* r,
                                   const double* z,
                                   const double* vr,
                                   const double* vz,
                                   const int corner,
                                   const double tau) {
  double rr[4];
  double zz[4];
  for (int k = 0; k < 4; ++k) {
    rr[k] = r[k] + tau * vr[k];
    zz[k] = z[k] + tau * vz[k];
  }
  return corner_jacobian_from_quad(rr, zz, corner);
}

double corner_limited_tau_host(const double* r,
                               const double* z,
                               const double* vr,
                               const double* vz,
                               const int corner,
                               const double dt,
                               const double floor) {
  double lo = 0.0;
  double hi = dt;
  for (int iter = 0; iter < 48; ++iter) {
    const double mid = 0.5 * (lo + hi);
    const double j_mid = corner_jacobian_at_tau_host(r, z, vr, vz, corner, mid);
    if (j_mid >= floor) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

double axis_margin_limited_tau_host(const double* r,
                                    const double* z,
                                    const double* vr,
                                    const double* vz,
                                    const double dt,
                                    const double floor_eps) {
  double rr[4];
  double zz[4];
  const auto current = evaluate_axis_margin_predicate_quad(r, z, r, z, floor_eps);
  if (!current.admissible) {
    return 0.0;
  }
  double lo = 0.0;
  double hi = dt;
  for (int iter = 0; iter < 48; ++iter) {
    const double mid = 0.5 * (lo + hi);
    for (int k = 0; k < 4; ++k) {
      rr[k] = r[k] + mid * vr[k];
      zz[k] = z[k] + mid * vz[k];
    }
    const auto trial =
        evaluate_axis_margin_predicate_quad(r, z, rr, zz, floor_eps);
    if (trial.admissible) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

void gather_cell_nodes(const std::vector<double>& r_base,
                       const std::vector<double>& z_base,
                       const std::vector<double>& v_r,
                       const std::vector<double>& v_z,
                       const int nz,
                       const int c,
                       double* r,
                       double* z,
                       double* vr,
                       double* vz) {
  const int i = c / nz;
  const int j = c - i * nz;
  const int nodes[4] = {
      rz_node_index_2d(i, j, nz),
      rz_node_index_2d(i + 1, j, nz),
      rz_node_index_2d(i + 1, j + 1, nz),
      rz_node_index_2d(i, j + 1, nz),
  };
  for (int k = 0; k < 4; ++k) {
    r[k] = r_base[static_cast<std::size_t>(nodes[k])];
    z[k] = z_base[static_cast<std::size_t>(nodes[k])];
    vr[k] = v_r[static_cast<std::size_t>(nodes[k])];
    vz[k] = v_z[static_cast<std::size_t>(nodes[k])];
  }
}

__device__ __forceinline__ double corner_jacobian_at_tau_device(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int* __restrict__ nodes,
    const int corner,
    const double tau) {
  double rr[4];
  double zz[4];
  for (int k = 0; k < 4; ++k) {
    const int n = nodes[k];
    rr[k] = x_r[n] + tau * v_r[n];
    zz[k] = x_z[n] + tau * v_z[n];
  }
  return corner_jacobian_from_quad(rr, zz, corner);
}

__device__ __forceinline__ void cell_quad_at_tau_device(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int* __restrict__ nodes,
    const double tau,
    double* rr,
    double* zz) {
  for (int k = 0; k < 4; ++k) {
    const int n = nodes[k];
    rr[k] = x_r[n] + tau * v_r[n];
    zz[k] = x_z[n] + tau * v_z[n];
  }
}

__device__ __forceinline__ double axis_margin_trial_scale_device(
    const double* r_current,
    const double* z_current,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int* __restrict__ nodes,
    const double dt_proposed,
    const double floor_eps) {
  const AxisMarginPredicateResult current =
      evaluate_axis_margin_predicate_quad(r_current, z_current, r_current,
                                          z_current, floor_eps);
  if (!current.admissible) {
    return 0.0;
  }

  double r_trial[4];
  double z_trial[4];
  cell_quad_at_tau_device(x_r, x_z, v_r, v_z, nodes, dt_proposed, r_trial,
                          z_trial);
  const AxisMarginPredicateResult full =
      evaluate_axis_margin_predicate_quad(r_current, z_current, r_trial,
                                          z_trial, floor_eps);
  if (full.admissible) {
    return 1.0;
  }

  double lo = 0.0;
  double hi = 1.0;
  for (int iter = 0; iter < 48; ++iter) {
    const double scale = 0.5 * (lo + hi);
    cell_quad_at_tau_device(x_r, x_z, v_r, v_z, nodes, scale * dt_proposed,
                            r_trial, z_trial);
    const AxisMarginPredicateResult mid =
        evaluate_axis_margin_predicate_quad(r_current, z_current, r_trial,
                                            z_trial, floor_eps);
    if (mid.admissible) {
      lo = scale;
    } else {
      hi = scale;
    }
  }
  return lo;
}

__device__ __forceinline__ double axis_margin_candidate_sigma_device(
    const double* r_current,
    const double* z_current,
    const double* r_candidate,
    const double* z_candidate,
    const double floor_eps) {
  const AxisMarginPredicateResult current =
      evaluate_axis_margin_predicate_quad(r_current, z_current, r_current,
                                          z_current, floor_eps);
  if (!current.admissible) {
    return 0.0;
  }
  const AxisMarginPredicateResult full =
      evaluate_axis_margin_predicate_quad(r_current, z_current, r_candidate,
                                          z_candidate, floor_eps);
  if (full.admissible) {
    return 1.0;
  }

  double r_mid[4];
  double z_mid[4];
  double lo = 0.0;
  double hi = 1.0;
  for (int iter = 0; iter < 48; ++iter) {
    const double sigma = 0.5 * (lo + hi);
    for (int k = 0; k < 4; ++k) {
      r_mid[k] = r_current[k] + sigma * (r_candidate[k] - r_current[k]);
      z_mid[k] = z_current[k] + sigma * (z_candidate[k] - z_current[k]);
    }
    const AxisMarginPredicateResult mid =
        evaluate_axis_margin_predicate_quad(r_current, z_current, r_mid, z_mid,
                                            floor_eps);
    if (mid.admissible) {
      lo = sigma;
    } else {
      hi = sigma;
    }
  }
  return lo;
}

__global__ void corner_jacobian_trial_scale_kernel(
    double* __restrict__ min_scale,
    int* __restrict__ first_failing_cell,
    int* __restrict__ first_failing_corner,
    int* __restrict__ any_trigger,
    int* __restrict__ first_trigger_key,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const CellRegime* __restrict__ cell_regime,
    const int nr,
    const int nz,
    const double dt_proposed,
    const double floor_eps,
    const int regime_aware_corner_j_guard_enabled,
    const int axis_margin_guard_enabled,
    const int has_physical_rz_axis,
    const double legacy_trigger_scale) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int nodes[4] = {
      rz_node_index_2d(i, j, nz),
      rz_node_index_2d(i + 1, j, nz),
      rz_node_index_2d(i + 1, j + 1, nz),
      rz_node_index_2d(i, j + 1, nz),
  };
  const double trigger_threshold =
      (regime_aware_corner_j_guard_enabled != 0 && cell_regime != nullptr)
          ? static_cast<double>(cell_regime[c].trigger_scale_threshold)
          : legacy_trigger_scale;

  if (axis_margin_guard_enabled != 0 && has_physical_rz_axis != 0 &&
      i == 0) {
    double r_current[4];
    double z_current[4];
    cell_quad_at_tau_device(x_r, x_z, v_r, v_z, nodes, 0.0, r_current,
                            z_current);
    const double scale = axis_margin_trial_scale_device(
        r_current, z_current, x_r, x_z, v_r, v_z, nodes, dt_proposed,
        floor_eps);
    if (scale < 1.0) {
      atomic_min_double(min_scale, scale);
      atomicMin(first_failing_cell, c);
      atomicMin(first_failing_corner, 0);
    }
    if (scale < trigger_threshold) {
      atomicExch(any_trigger, 1);
      atomicMin(first_trigger_key, c * 4);
    }
    return;
  }

  for (int corner = 0; corner < 4; ++corner) {
    const double j_current =
        corner_jacobian_at_tau_device(x_r, x_z, v_r, v_z, nodes, corner, 0.0);
    if (!(j_current > 0.0) || !isfinite(j_current)) {
      atomic_min_double(min_scale, 0.0);
      atomicMin(first_failing_cell, c);
      atomicMin(first_failing_corner, corner);
      if (0.0 < trigger_threshold) {
        atomicExch(any_trigger, 1);
        atomicMin(first_trigger_key, c * 4 + corner);
      }
      continue;
    }

    const double floor = floor_eps * j_current;
    const double j_trial =
        corner_jacobian_at_tau_device(x_r, x_z, v_r, v_z, nodes, corner, dt_proposed);
    if (j_trial >= floor && isfinite(j_trial)) {
      continue;
    }

    double lo = 0.0;
    double hi = 1.0;
    for (int iter = 0; iter < 48; ++iter) {
      const double scale = 0.5 * (lo + hi);
      const double j_mid = corner_jacobian_at_tau_device(
          x_r, x_z, v_r, v_z, nodes, corner, scale * dt_proposed);
      if (j_mid >= floor) {
        lo = scale;
      } else {
        hi = scale;
      }
    }
    atomic_min_double(min_scale, lo);
    atomicMin(first_failing_cell, c);
    atomicMin(first_failing_corner, corner);
    if (lo < trigger_threshold) {
      atomicExch(any_trigger, 1);
      atomicMin(first_trigger_key, c * 4 + corner);
    }
  }
}

__global__ void in_hydro_candidate_corner_j_guard_kernel(
    double* __restrict__ min_corner_j,
    double* __restrict__ max_abs_corner_j,
    double* __restrict__ min_gauss_j,
    double* __restrict__ min_rz_volume,
    double* __restrict__ min_sigma,
    int* __restrict__ any_failure,
    int* __restrict__ any_gauss_j_failure,
    int* __restrict__ any_rz_volume_failure,
    unsigned long long* __restrict__ first_failing_composite_key,
    const double* __restrict__ x_r_start,
    const double* __restrict__ x_z_start,
    const double* __restrict__ x_r_candidate,
    const double* __restrict__ x_z_candidate,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const double floor_eps,
    const int corner_j_guard_enabled,
    const int axis_margin_guard_enabled,
    const int has_physical_rz_axis,
    const int gauss_j_guard_enabled,
    const double gauss_j_floor_rel,
    const int rz_volume_guard_enabled,
    const double rz_volume_floor_rel,
    const int axis_margin_additive) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int nodes[4] = {
      rz_node_index_2d(i, j, nz),
      rz_node_index_2d(i + 1, j, nz),
      rz_node_index_2d(i + 1, j + 1, nz),
      rz_node_index_2d(i, j + 1, nz),
  };

  double rr0[4];
  double zz0[4];
  double rr1[4];
  double zz1[4];
  for (int k = 0; k < 4; ++k) {
    const int n = nodes[k];
    rr0[k] = x_r_start[n];
    zz0[k] = x_z_start[n];
    rr1[k] = x_r_candidate[n];
    zz1[k] = x_z_candidate[n];
  }
  if (max_abs_corner_j != nullptr) {
    double local_max_abs = 0.0;
    for (int corner = 0; corner < 4; ++corner) {
      const double j_abs = fabs(corner_jacobian_from_quad(rr1, zz1, corner));
      if (isfinite(j_abs)) {
        local_max_abs = fmax(local_max_abs, j_abs);
      }
    }
    atomic_max_double(max_abs_corner_j, local_max_abs);
  }

  if (corner_j_guard_enabled != 0) {
    if (axis_margin_guard_enabled != 0 && has_physical_rz_axis != 0 &&
        i == 0 &&
        axis_margin_additive == 0) {
      const AxisMarginPredicateResult full =
          evaluate_axis_margin_predicate_quad(rr0, zz0, rr1, zz1, floor_eps);
      if (!full.admissible) {
        const double sigma =
            axis_margin_candidate_sigma_device(rr0, zz0, rr1, zz1, floor_eps);
        const double margin =
            isfinite(full.min_margin) ? full.min_margin : -INFINITY;
        atomic_min_double(min_corner_j, margin);
        atomicExch(any_failure, 1);
        record_composite_failure(min_sigma, first_failing_composite_key,
                                 sigma, in_hydro_corner_failure_key(c, 0));
        return;
      }
    } else {
      if (axis_margin_guard_enabled != 0 && has_physical_rz_axis != 0 &&
          i == 0 &&
          axis_margin_additive != 0) {
        const AxisMarginPredicateResult full =
            evaluate_axis_margin_predicate_quad(rr0, zz0, rr1, zz1, floor_eps);
        if (!full.admissible) {
          const double sigma =
              axis_margin_candidate_sigma_device(rr0, zz0, rr1, zz1, floor_eps);
          const double margin =
              isfinite(full.min_margin) ? full.min_margin : -INFINITY;
          atomic_min_double(min_corner_j, margin);
          atomicExch(any_failure, 1);
          record_composite_failure(min_sigma, first_failing_composite_key,
                                   sigma, in_hydro_axis_failure_key(c));
        }
      }
      for (int corner = 0; corner < 4; ++corner) {
        const int kp = (corner + 1) & 3;
        const int km = (corner + 3) & 3;
        const double a0_r = rr0[kp] - rr0[corner];
        const double a0_z = zz0[kp] - zz0[corner];
        const double b0_r = rr0[km] - rr0[corner];
        const double b0_z = zz0[km] - zz0[corner];
        const double a1_r = rr1[kp] - rr1[corner];
        const double a1_z = zz1[kp] - zz1[corner];
        const double b1_r = rr1[km] - rr1[corner];
        const double b1_z = zz1[km] - zz1[corner];
        const double da_r = a1_r - a0_r;
        const double da_z = a1_z - a0_z;
        const double db_r = b1_r - b0_r;
        const double db_z = b1_z - b0_z;

        const double j0 = corner_jacobian_cross2(a0_r, a0_z, b0_r, b0_z);
        const double j1 = corner_jacobian_cross2(da_r, da_z, b0_r, b0_z) +
                          corner_jacobian_cross2(a0_r, a0_z, db_r, db_z);
        const double j2 = corner_jacobian_cross2(da_r, da_z, db_r, db_z);
        const int key = in_hydro_corner_failure_key(c, corner);

        if (!isfinite(j0) || !isfinite(j1) || !isfinite(j2)) {
          atomic_min_double(min_corner_j, -INFINITY);
          atomicExch(any_failure, 1);
          record_composite_failure(min_sigma, first_failing_composite_key,
                                   0.0, key);
          continue;
        }

        const double floor = (j0 > 0.0) ? floor_eps * j0 : 0.0;
        if (!(j0 > floor)) {
          atomic_min_double(min_corner_j, j0);
          atomicExch(any_failure, 1);
          record_composite_failure(min_sigma, first_failing_composite_key,
                                   0.0, key);
          continue;
        }

        const double local_min_j =
            in_hydro_corner_j_min_on_unit_interval(j0, j1, j2);
        atomic_min_double(min_corner_j, local_min_j);
        if (local_min_j >= floor && isfinite(local_min_j)) {
          continue;
        }

        const InHydroCornerJRoot root =
            first_in_hydro_corner_j_floor_root(j0, j1, j2, floor);
        const double sigma = root.found ? root.sigma : 0.0;
        atomicExch(any_failure, 1);
        record_composite_failure(min_sigma, first_failing_composite_key,
                                 sigma, key);
      }
    }
  }

  if (gauss_j_guard_enabled != 0) {
    constexpr double g = 0.57735026918962576450914878050195745565;
    const double xi[4] = {-g, g, g, -g};
    const double eta[4] = {-g, -g, g, g};
    for (int q = 0; q < 4; ++q) {
      double j0;
      double j1;
      double j2;
      in_hydro_gauss_j_coefficients_at(rr0, zz0, rr1, zz1, xi[q], eta[q],
                                       j0, j1, j2);
      const int key = in_hydro_gauss_failure_key(c, q);
      if (!isfinite(j0) || !isfinite(j1) || !isfinite(j2)) {
        atomic_min_double(min_gauss_j, -INFINITY);
        atomicExch(any_failure, 1);
        atomicExch(any_gauss_j_failure, 1);
        record_composite_failure(min_sigma, first_failing_composite_key,
                                 0.0, key);
        continue;
      }
      const double floor = (j0 > 0.0) ? gauss_j_floor_rel * j0 : 0.0;
      if (!(j0 > floor)) {
        atomic_min_double(min_gauss_j, j0);
        atomicExch(any_failure, 1);
        atomicExch(any_gauss_j_failure, 1);
        record_composite_failure(min_sigma, first_failing_composite_key,
                                 0.0, key);
        continue;
      }
      const double local_min_j =
          in_hydro_trial_quadratic_min_on_unit_interval(j0, j1, j2);
      atomic_min_double(min_gauss_j, local_min_j);
      if (local_min_j >= floor && isfinite(local_min_j)) {
        continue;
      }
      const InHydroCornerJRoot root =
          first_in_hydro_corner_j_floor_root(j0, j1, j2, floor);
      const double sigma = root.found ? root.sigma : 0.0;
      atomicExch(any_failure, 1);
      atomicExch(any_gauss_j_failure, 1);
      record_composite_failure(min_sigma, first_failing_composite_key,
                               sigma, key);
    }
  }

  if (rz_volume_guard_enabled != 0) {
    const double v0 = in_hydro_rz_volume_at_sigma(rr0, zz0, rr1, zz1, 0.0);
    const double floor = (v0 > 0.0) ? rz_volume_floor_rel * v0 : 0.0;
    const int key = in_hydro_rz_failure_key(c);
    if (!isfinite(v0) || !(v0 > floor)) {
      atomic_min_double(min_rz_volume, isfinite(v0) ? v0 : -INFINITY);
      atomicExch(any_failure, 1);
      atomicExch(any_rz_volume_failure, 1);
      record_composite_failure(min_sigma, first_failing_composite_key,
                               0.0, key);
      return;
    }
    const double local_min_v =
        in_hydro_rz_volume_min_on_unit_interval(rr0, zz0, rr1, zz1);
    atomic_min_double(min_rz_volume, local_min_v);
    if (local_min_v >= floor && isfinite(local_min_v)) {
      return;
    }
    const InHydroCornerJRoot root =
        first_in_hydro_rz_volume_floor_root(rr0, zz0, rr1, zz1, floor);
    const double sigma = root.found ? root.sigma : 0.0;
    atomicExch(any_failure, 1);
    atomicExch(any_rz_volume_failure, 1);
    record_composite_failure(min_sigma, first_failing_composite_key,
                             sigma, key);
  }
}

__device__ __forceinline__ void mesh_quality_record_failure(
    double* __restrict__ min_sigma,
    unsigned long long* __restrict__ first_failing_composite_key,
    const double sigma,
    const int detail_key) {
  record_composite_failure(min_sigma, first_failing_composite_key,
                           sigma, detail_key);
}

__global__ void mesh_quality_dt_limit_kernel(
    double* __restrict__ min_sigma,
    unsigned long long* __restrict__ first_failing_composite_key,
    const double* __restrict__ x_r_start,
    const double* __restrict__ x_z_start,
    const double* __restrict__ v_r_half,
    const double* __restrict__ v_z_half,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const double dt,
    const int corner_j_enabled,
    const int gauss_j_enabled,
    const int rz_volume_enabled,
    const int axis_margin_additive,
    const int has_physical_rz_axis,
    const double corner_j_floor_rel,
    const double gauss_j_floor_rel,
    const double rz_volume_floor_rel) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int nodes[4] = {
      rz_node_index_2d(i, j, nz),
      rz_node_index_2d(i + 1, j, nz),
      rz_node_index_2d(i + 1, j + 1, nz),
      rz_node_index_2d(i, j + 1, nz),
  };

  double rr0[4];
  double zz0[4];
  double rr1[4];
  double zz1[4];
  for (int k = 0; k < 4; ++k) {
    const int n = nodes[k];
    rr0[k] = x_r_start[n];
    zz0[k] = x_z_start[n];
    rr1[k] = x_r_start[n] + dt * v_r_half[n];
    zz1[k] = x_z_start[n] + dt * v_z_half[n];
  }

  if (corner_j_enabled != 0) {
    for (int corner = 0; corner < 4; ++corner) {
      const int kp = (corner + 1) & 3;
      const int km = (corner + 3) & 3;
      const double a0_r = rr0[kp] - rr0[corner];
      const double a0_z = zz0[kp] - zz0[corner];
      const double b0_r = rr0[km] - rr0[corner];
      const double b0_z = zz0[km] - zz0[corner];
      const double a1_r = rr1[kp] - rr1[corner];
      const double a1_z = zz1[kp] - zz1[corner];
      const double b1_r = rr1[km] - rr1[corner];
      const double b1_z = zz1[km] - zz1[corner];
      const double da_r = a1_r - a0_r;
      const double da_z = a1_z - a0_z;
      const double db_r = b1_r - b0_r;
      const double db_z = b1_z - b0_z;

      const double j0 = corner_jacobian_cross2(a0_r, a0_z, b0_r, b0_z);
      const double j1 = corner_jacobian_cross2(da_r, da_z, b0_r, b0_z) +
                        corner_jacobian_cross2(a0_r, a0_z, db_r, db_z);
      const double j2 = corner_jacobian_cross2(da_r, da_z, db_r, db_z);
      const int key = mesh_quality_corner_failure_key(c, corner);

      if (!isfinite(j0) || !isfinite(j1) || !isfinite(j2)) {
        mesh_quality_record_failure(min_sigma, first_failing_composite_key, 0.0, key);
        continue;
      }
      const double floor = (j0 > 0.0) ? corner_j_floor_rel * j0 : 0.0;
      if (!(j0 > floor)) {
        mesh_quality_record_failure(min_sigma, first_failing_composite_key, 0.0, key);
        continue;
      }
      const double local_min_j =
          in_hydro_corner_j_min_on_unit_interval(j0, j1, j2);
      if (local_min_j >= floor && isfinite(local_min_j)) {
        continue;
      }
      const InHydroCornerJRoot root =
          first_in_hydro_corner_j_floor_root(j0, j1, j2, floor);
      const double sigma = root.found ? root.sigma : 0.0;
      mesh_quality_record_failure(min_sigma, first_failing_composite_key, sigma, key);
    }
  }

  if (gauss_j_enabled != 0) {
    constexpr double g = 0.57735026918962576450914878050195745565;
    const double xi[4] = {-g, g, g, -g};
    const double eta[4] = {-g, -g, g, g};
    for (int q = 0; q < 4; ++q) {
      double j0;
      double j1;
      double j2;
      in_hydro_gauss_j_coefficients_at(rr0, zz0, rr1, zz1, xi[q], eta[q],
                                       j0, j1, j2);
      const int key = mesh_quality_gauss_failure_key(c, q);
      if (!isfinite(j0) || !isfinite(j1) || !isfinite(j2)) {
        mesh_quality_record_failure(min_sigma, first_failing_composite_key, 0.0, key);
        continue;
      }
      const double floor = (j0 > 0.0) ? gauss_j_floor_rel * j0 : 0.0;
      if (!(j0 > floor)) {
        mesh_quality_record_failure(min_sigma, first_failing_composite_key, 0.0, key);
        continue;
      }
      const double local_min_j =
          in_hydro_trial_quadratic_min_on_unit_interval(j0, j1, j2);
      if (local_min_j >= floor && isfinite(local_min_j)) {
        continue;
      }
      const InHydroCornerJRoot root =
          first_in_hydro_corner_j_floor_root(j0, j1, j2, floor);
      const double sigma = root.found ? root.sigma : 0.0;
      mesh_quality_record_failure(min_sigma, first_failing_composite_key, sigma, key);
    }
  }

  if (rz_volume_enabled != 0) {
    const double v0 = in_hydro_rz_volume_at_sigma(rr0, zz0, rr1, zz1, 0.0);
    const double floor = (v0 > 0.0) ? rz_volume_floor_rel * v0 : 0.0;
    const int key = mesh_quality_rz_failure_key(c);
    if (!isfinite(v0) || !(v0 > floor)) {
      mesh_quality_record_failure(min_sigma, first_failing_composite_key, 0.0, key);
    } else {
      const double local_min_v =
          in_hydro_rz_volume_min_on_unit_interval(rr0, zz0, rr1, zz1);
      if (!(local_min_v >= floor && isfinite(local_min_v))) {
        const InHydroCornerJRoot root =
            first_in_hydro_rz_volume_floor_root(rr0, zz0, rr1, zz1, floor);
        const double sigma = root.found ? root.sigma : 0.0;
        mesh_quality_record_failure(min_sigma, first_failing_composite_key, sigma, key);
      }
    }
  }

  if (axis_margin_additive != 0 && has_physical_rz_axis != 0 && i == 0) {
    const AxisMarginPredicateResult current =
        evaluate_axis_margin_predicate_quad(rr0, zz0, rr0, zz0,
                                            corner_j_floor_rel);
    const int key = mesh_quality_axis_failure_key(c);
    if (!current.admissible) {
      mesh_quality_record_failure(min_sigma, first_failing_composite_key, 0.0, key);
      return;
    }
    const AxisMarginPredicateResult full =
        evaluate_axis_margin_predicate_quad(rr0, zz0, rr1, zz1,
                                            corner_j_floor_rel);
    if (!full.admissible) {
      const double sigma =
          axis_margin_candidate_sigma_device(rr0, zz0, rr1, zz1,
                                             corner_j_floor_rel);
      mesh_quality_record_failure(min_sigma, first_failing_composite_key, sigma, key);
    }
  }
}

__global__ void mesh_quality_dt_limit_multiblock_rz_volume_kernel(
    double* __restrict__ min_sigma,
    unsigned long long* __restrict__ first_failing_composite_key,
    int* __restrict__ debug_printed,
    const double* __restrict__ x_r_start,
    const double* __restrict__ x_z_start,
    const double* __restrict__ v_r_half,
    const double* __restrict__ v_z_half,
    const double* __restrict__ cell_vol,
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ pseudo_core_member,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells,
    const double dt,
    const double rz_volume_floor_rel,
    const int state_step) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }
  if (pseudo_core_member != nullptr && pseudo_core_member[c] != 0U) {
    return;
  }

  const double state_vol = cell_vol[c];
  if (!isfinite(state_vol) || !(state_vol > 0.0)) {
    return;
  }

  const int off = cell_node_csr_offsets[c];
  const int nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  const double orientation_sign =
      static_cast<double>(cell_orientation_sign[c]);
  int nodes[4] = {-1, -1, -1, -1};
  double rr0[4] = {0.0, 0.0, 0.0, 0.0};
  double zz0[4] = {0.0, 0.0, 0.0, 0.0};
  double rr1[4] = {0.0, 0.0, 0.0, 0.0};
  double zz1[4] = {0.0, 0.0, 0.0, 0.0};
  for (int k = 0; k < nverts; ++k) {
    const int n = cell_node_csr_indices[off + k];
    nodes[k] = n;
    rr0[k] = x_r_start[n];
    zz0[k] = x_z_start[n];
    rr1[k] = x_r_start[n] + dt * v_r_half[n];
    zz1[k] = x_z_start[n] + dt * v_z_half[n];
  }

  const double v0 = multiblock_oriented_rz_volume_at_sigma(
      rr0, zz0, rr1, zz1, nverts, orientation_sign, 0.0);
  const double floor = rz_volume_floor_rel * state_vol;
  const int key = mesh_quality_rz_failure_key(c);
  if (!isfinite(v0) || !(v0 > floor)) {
    if (state_step == 0 && debug_printed != nullptr &&
        atomicCAS(debug_printed, 0, 1) == 0) {
      printf("[mesh-quality-multiblock-rz-volume sigma0 step0] c=%d "
             "active_nverts=%d orientation_sign=%.0f V0=%.17e "
             "state_vol=%.17e floor=%.17e "
             "nodes=(%d,%.17e,%.17e),(%d,%.17e,%.17e),"
             "(%d,%.17e,%.17e),(%d,%.17e,%.17e)\n",
             c, nverts, orientation_sign, v0, state_vol, floor,
             nodes[0], rr0[0], zz0[0],
             nodes[1], rr0[1], zz0[1],
             nodes[2], rr0[2], zz0[2],
             nodes[3], rr0[3], zz0[3]);
    }
    mesh_quality_record_failure(min_sigma, first_failing_composite_key,
                                0.0, key);
    return;
  }

  const InHydroCornerJRoot root =
      first_multiblock_oriented_rz_volume_floor_root(
          rr0, zz0, rr1, zz1, nverts, orientation_sign, floor);
  const double sigma = root.found ? root.sigma : 0.0;
  if (!root.found) {
    return;
  }
  if (state_step == 0 && sigma == 0.0 && debug_printed != nullptr &&
      atomicCAS(debug_printed, 0, 1) == 0) {
    printf("[mesh-quality-multiblock-rz-volume sigma0 step0] c=%d "
           "active_nverts=%d orientation_sign=%.0f V0=%.17e "
           "state_vol=%.17e floor=%.17e "
           "nodes=(%d,%.17e,%.17e),(%d,%.17e,%.17e),"
           "(%d,%.17e,%.17e),(%d,%.17e,%.17e)\n",
           c, nverts, orientation_sign, v0, state_vol, floor,
           nodes[0], rr0[0], zz0[0],
           nodes[1], rr0[1], zz0[1],
           nodes[2], rr0[2], zz0[2],
           nodes[3], rr0[3], zz0[3]);
  }
  mesh_quality_record_failure(min_sigma, first_failing_composite_key,
                              sigma, key);
}

__global__ void mesh_quality_rz_volume_cell_margin_kernel(
    const int* __restrict__ requested_cells,
    int* __restrict__ out_cell,
    int* __restrict__ out_admissible,
    double* __restrict__ out_sigma,
    double* __restrict__ out_eta,
    double* __restrict__ out_v0,
    double* __restrict__ out_floor,
    const double* __restrict__ x_r_start,
    const double* __restrict__ x_z_start,
    const double* __restrict__ v_r_half,
    const double* __restrict__ v_z_half,
    const double* __restrict__ cell_vol,
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ pseudo_core_member,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells,
    const int n_requested,
    const double dt,
    const double rz_volume_floor_rel,
    const double safety_alpha) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_requested) {
    return;
  }

  const int c = requested_cells[idx];
  out_cell[idx] = c;
  out_admissible[idx] = 1;
  out_sigma[idx] = 1.0;
  out_eta[idx] = 1.0;
  out_v0[idx] = 0.0;
  out_floor[idx] = 0.0;

  if (c < 0 || c >= n_cells) {
    out_admissible[idx] = 0;
    out_sigma[idx] = 0.0;
    out_eta[idx] = 0.0;
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }
  if (pseudo_core_member != nullptr && pseudo_core_member[c] != 0U) {
    return;
  }

  const double state_vol = cell_vol[c];
  if (!isfinite(state_vol) || !(state_vol > 0.0)) {
    return;
  }

  const int off = cell_node_csr_offsets[c];
  const int nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  const double orientation_sign =
      static_cast<double>(cell_orientation_sign[c]);
  double rr0[4] = {0.0, 0.0, 0.0, 0.0};
  double zz0[4] = {0.0, 0.0, 0.0, 0.0};
  double rr1[4] = {0.0, 0.0, 0.0, 0.0};
  double zz1[4] = {0.0, 0.0, 0.0, 0.0};
  for (int k = 0; k < nverts; ++k) {
    const int n = cell_node_csr_indices[off + k];
    rr0[k] = x_r_start[n];
    zz0[k] = x_z_start[n];
    rr1[k] = x_r_start[n] + dt * v_r_half[n];
    zz1[k] = x_z_start[n] + dt * v_z_half[n];
  }

  const double v0 = multiblock_oriented_rz_volume_at_sigma(
      rr0, zz0, rr1, zz1, nverts, orientation_sign, 0.0);
  const double floor = rz_volume_floor_rel * state_vol;
  out_v0[idx] = v0;
  out_floor[idx] = floor;
  if (!isfinite(v0) || !(v0 > floor)) {
    out_admissible[idx] = 0;
    out_sigma[idx] = 0.0;
    out_eta[idx] = 0.0;
    return;
  }

  const InHydroCornerJRoot root =
      first_multiblock_oriented_rz_volume_floor_root(
          rr0, zz0, rr1, zz1, nverts, orientation_sign, floor);
  if (!root.found) {
    return;
  }
  const double sigma =
      (isfinite(root.sigma) && root.sigma >= 0.0)
          ? fmin(root.sigma, 1.0)
          : 0.0;
  out_admissible[idx] = 0;
  out_sigma[idx] = sigma;
  out_eta[idx] = safety_alpha * sigma;
}

MeshQualityDtLimit compute_multiblock_mesh_quality_dt_limit(
    const core::State& state,
    const core::Config& cfg,
    const double* d_x_r_start,
    const double* d_x_z_start,
    const double* d_v_r_half,
    const double* d_v_z_half,
    const double dt,
    const parallel::Reduction* reduction) {
  MeshQualityDtLimit result{};
  result.suggested_dt = dt;
  if (!cfg.numerics.hydro.mesh_quality_dt_rz_volume_enabled) {
    return result;
  }
  if (state.mesh.dim != 2) {
    return result;
  }

  TENRYU_ASSERT(dt > 0.0 && std::isfinite(dt),
                "mesh-quality multiblock dt CFL requires finite dt > 0");
  TENRYU_ASSERT(cfg.numerics.hydro.mesh_quality_dt_safety_alpha > 0.0 &&
                    cfg.numerics.hydro.mesh_quality_dt_safety_alpha <= 1.0,
                "mesh-quality multiblock dt CFL safety alpha must be in (0, 1]");
  TENRYU_ASSERT(cfg.numerics.hydro.mesh_quality_dt_rz_volume_floor_rel > 0.0 &&
                    std::isfinite(
                        cfg.numerics.hydro.mesh_quality_dt_rz_volume_floor_rel),
                "mesh-quality multiblock RZ-volume floor_rel must be finite > 0");
  TENRYU_ASSERT(d_x_r_start != nullptr && d_x_z_start != nullptr &&
                    d_v_r_half != nullptr && d_v_z_half != nullptr,
                "mesh-quality multiblock dt CFL requires non-null coordinate and velocity fields");
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "mesh-quality multiblock dt CFL requires multiblock topology");

  const auto& mb = state.mesh.topo.multiblock.value();
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (n_cells <= 0) {
    return result;
  }
  TENRYU_ASSERT(n_nodes > 0,
                "mesh-quality multiblock dt CFL requires positive node count");
  TENRYU_ASSERT(static_cast<int>(state.x_r.size()) == n_nodes &&
                    static_cast<int>(state.x_z.size()) == n_nodes,
                "mesh-quality multiblock dt CFL requires node field size == topo.n_nodes");
  TENRYU_ASSERT(static_cast<int>(state.vol.size()) == n_cells,
                "mesh-quality multiblock dt CFL requires vol size == n_cells");
  TENRYU_ASSERT(state.hydro_active.empty() ||
                    static_cast<int>(state.hydro_active.size()) == n_cells,
                "mesh-quality multiblock dt CFL requires hydro_active size == n_cells");
  TENRYU_ASSERT(static_cast<int>(mb.cell_orientation_sign.size()) == n_cells,
                "mesh-quality multiblock dt CFL requires orientation_sign size == n_cells");
  TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells + 1),
                "mesh-quality multiblock dt CFL requires CSR offsets size == n_cells+1");
  TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                    mb.cell_node_csr_indices.size(),
                "mesh-quality multiblock dt CFL requires CSR indices device mirror");

  core::DeviceArray<double> d_min_sigma("corner_j:compute_multiblock_mesh_quality_dt_limit:d_min_sigma");
  d_min_sigma.reset(1U);
  core::DeviceArray<unsigned long long> d_first_failing_composite_key("corner_j:compute_multiblock_mesh_quality_dt_limit:d_first_failing_composite_key");
  d_first_failing_composite_key.reset(1U);
  core::DeviceArray<int> d_cell_orientation_sign("corner_j:compute_multiblock_mesh_quality_dt_limit:d_cell_orientation_sign");
  d_cell_orientation_sign.reset(
      static_cast<std::size_t>(n_cells));
  d_cell_orientation_sign.copy_from_host(mb.cell_orientation_sign);

  core::DeviceArray<std::uint8_t> d_cell_nverts_owned("corner_j:compute_multiblock_mesh_quality_dt_limit:d_cell_nverts_owned");
  const std::uint8_t* d_cell_nverts = nullptr;
  if (state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)) {
    d_cell_nverts_owned.reset(static_cast<std::size_t>(n_cells));
    d_cell_nverts_owned.copy_from_host(state.mesh.cell_nverts);
    d_cell_nverts = d_cell_nverts_owned.data();
  }

  core::DeviceArray<std::int8_t> d_active_owned("corner_j:compute_multiblock_mesh_quality_dt_limit:d_active_owned");
  const std::int8_t* d_active = nullptr;
  if (!state.hydro_active.empty()) {
    d_active_owned.reset(state.hydro_active.size());
    d_active_owned.copy_from_host(state.hydro_active);
    d_active = d_active_owned.data();
  }

  core::DeviceArray<int> d_debug_printed("corner_j:compute_multiblock_mesh_quality_dt_limit:d_debug_printed");
  d_debug_printed.reset(1U);

  const double one = 1.0;
  const unsigned long long no_composite_key = kNoFailingCompositeKey;
  d_min_sigma.copy_from_host(&one);
  d_first_failing_composite_key.copy_from_host(&no_composite_key);

  core::DeviceArray<std::uint8_t> d_combined_inactive("corner_j:compute_multiblock_mesh_quality_dt_limit:d_combined_inactive");
  const std::uint8_t* d_inactive_mask =
      pole_angular_derefine::combined_inactive_mask_device(
          const_cast<core::State&>(state), d_combined_inactive);
  const int blocks = (n_cells + 255) / 256;
  mesh_quality_dt_limit_multiblock_rz_volume_kernel<<<blocks, 256>>>(
      d_min_sigma.data(),
      d_first_failing_composite_key.data(),
      d_debug_printed.data(),
      d_x_r_start,
      d_x_z_start,
      d_v_r_half,
      d_v_z_half,
      state.vol.data(),
      d_active,
      d_inactive_mask,
      state.mesh.multiblock_cell_node_csr_offsets.data(),
      state.mesh.multiblock_cell_node_csr_indices.data(),
      d_cell_orientation_sign.data(),
      d_cell_nverts,
      n_cells,
      dt,
      cfg.numerics.hydro.mesh_quality_dt_rz_volume_floor_rel,
      state.step);
  cuda_check(cudaGetLastError(),
             "mesh-quality multiblock dt CFL kernel launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "mesh-quality multiblock dt CFL kernel execution failed");

  unsigned long long local_composite_key = no_composite_key;
  cuda_check(cudaMemcpy(&local_composite_key,
                        d_first_failing_composite_key.data(),
                        sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost),
             "mesh-quality multiblock dt CFL: copy first_failing_key failed");

  std::uint64_t global_composite_key =
      static_cast<std::uint64_t>(local_composite_key);
  if (reduction != nullptr) {
    global_composite_key = reduction->allreduce_min(global_composite_key);
  }

  double global_sigma = 1.0;
  if (global_composite_key != no_composite_key) {
    const auto sigma_bits =
        static_cast<std::uint32_t>(global_composite_key >> 32);
    global_sigma = sigma_from_sortable_uint32(sigma_bits);
    if (!(global_sigma >= 0.0) || !std::isfinite(global_sigma)) {
      global_sigma = 0.0;
    }
    global_sigma = std::min(global_sigma, 1.0);
  }
  result.sigma_safe = global_sigma;
  if (global_composite_key == no_composite_key) {
    result.admissible = true;
    result.suggested_dt = dt;
    return result;
  }

  result.admissible = false;
  const double raw_suggested_dt =
      cfg.numerics.hydro.mesh_quality_dt_safety_alpha * global_sigma * dt;
  result.suggested_dt =
      (std::isfinite(raw_suggested_dt) && raw_suggested_dt > 0.0)
          ? std::min(raw_suggested_dt, dt)
          : std::numeric_limits<double>::denorm_min();
  result.suggested_dt = std::min(result.suggested_dt, dt);

  const auto detail_key =
      static_cast<std::uint32_t>(global_composite_key & 0xffffffffULL);
  result.first_cell = composite_detail_cell(detail_key);
  result.metric = MeshQualityFailingMetric::RzVolume;
  result.first_corner_or_gauss = -1;
  return result;
}

}  // namespace

std::vector<MeshQualityRzVolumeCellMargin>
compute_multiblock_mesh_quality_rz_volume_cell_margins(
    const core::State& state,
    const core::Config& cfg,
    const double* d_x_r_start,
    const double* d_x_z_start,
    const double* d_v_r_half,
    const double* d_v_z_half,
    const double dt,
    const std::vector<int>& cells) {
  return compute_multiblock_mesh_quality_rz_volume_cell_margins_with_volume(
      state, cfg, d_x_r_start, d_x_z_start, d_v_r_half, d_v_z_half,
      state.vol.data(), dt, cells);
}

std::vector<MeshQualityRzVolumeCellMargin>
compute_multiblock_mesh_quality_rz_volume_cell_margins_with_volume(
    const core::State& state,
    const core::Config& cfg,
    const double* d_x_r_start,
    const double* d_x_z_start,
    const double* d_v_r_half,
    const double* d_v_z_half,
    const double* d_cell_vol_for_floor,
    const double dt,
    const std::vector<int>& cells) {
  std::vector<MeshQualityRzVolumeCellMargin> out(cells.size());
  for (std::size_t i = 0; i < cells.size(); ++i) {
    out[i].cell = cells[i];
  }
  if (cells.empty()) {
    return out;
  }
  if (!cfg.numerics.hydro.mesh_quality_dt_rz_volume_enabled ||
      state.mesh.dim != 2 || !state.mesh.topo.multiblock.has_value()) {
    return out;
  }

  TENRYU_ASSERT(dt > 0.0 && std::isfinite(dt),
                "mesh-quality RZ-volume cell margins require finite dt > 0");
  TENRYU_ASSERT(cfg.numerics.hydro.mesh_quality_dt_safety_alpha > 0.0 &&
                    cfg.numerics.hydro.mesh_quality_dt_safety_alpha <= 1.0,
                "mesh-quality RZ-volume cell margins safety alpha must be in (0, 1]");
  TENRYU_ASSERT(cfg.numerics.hydro.mesh_quality_dt_rz_volume_floor_rel > 0.0 &&
                    std::isfinite(
                        cfg.numerics.hydro.mesh_quality_dt_rz_volume_floor_rel),
                "mesh-quality RZ-volume cell margins floor_rel must be finite > 0");
  TENRYU_ASSERT(d_x_r_start != nullptr && d_x_z_start != nullptr &&
                    d_v_r_half != nullptr && d_v_z_half != nullptr,
                "mesh-quality RZ-volume cell margins require non-null fields");
  TENRYU_ASSERT(d_cell_vol_for_floor != nullptr,
                "mesh-quality RZ-volume cell margins require non-null cell volumes");

  const auto& mb = state.mesh.topo.multiblock.value();
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  TENRYU_ASSERT(n_cells > 0 && n_nodes > 0,
                "mesh-quality RZ-volume cell margins require a non-empty mesh");
  TENRYU_ASSERT(static_cast<int>(state.vol.size()) == n_cells,
                "mesh-quality RZ-volume cell margins require vol size == n_cells");
  TENRYU_ASSERT(state.hydro_active.empty() ||
                    static_cast<int>(state.hydro_active.size()) == n_cells,
                "mesh-quality RZ-volume cell margins require hydro_active size == n_cells");
  TENRYU_ASSERT(static_cast<int>(mb.cell_orientation_sign.size()) == n_cells,
                "mesh-quality RZ-volume cell margins require orientation_sign size == n_cells");
  TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells + 1),
                "mesh-quality RZ-volume cell margins require CSR offsets size == n_cells+1");
  TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                    mb.cell_node_csr_indices.size(),
                "mesh-quality RZ-volume cell margins require CSR indices device mirror");

  core::DeviceArray<int> d_requested("corner_j:compute_multiblock_mesh_quality_rz_volume_cell_margins_with_volume:d_requested");
  d_requested.reset(cells.size());
  d_requested.copy_from_host(cells);
  core::DeviceArray<int> d_out_cell("corner_j:compute_multiblock_mesh_quality_rz_volume_cell_margins_with_volume:d_out_cell");
  d_out_cell.reset(cells.size());
  core::DeviceArray<int> d_out_admissible("corner_j:compute_multiblock_mesh_quality_rz_volume_cell_margins_with_volume:d_out_admissible");
  d_out_admissible.reset(cells.size());
  core::DeviceArray<double> d_out_sigma("corner_j:compute_multiblock_mesh_quality_rz_volume_cell_margins_with_volume:d_out_sigma");
  d_out_sigma.reset(cells.size());
  core::DeviceArray<double> d_out_eta("corner_j:compute_multiblock_mesh_quality_rz_volume_cell_margins_with_volume:d_out_eta");
  d_out_eta.reset(cells.size());
  core::DeviceArray<double> d_out_v0("corner_j:compute_multiblock_mesh_quality_rz_volume_cell_margins_with_volume:d_out_v0");
  d_out_v0.reset(cells.size());
  core::DeviceArray<double> d_out_floor("corner_j:compute_multiblock_mesh_quality_rz_volume_cell_margins_with_volume:d_out_floor");
  d_out_floor.reset(cells.size());

  core::DeviceArray<int> d_cell_orientation_sign("corner_j:compute_multiblock_mesh_quality_rz_volume_cell_margins_with_volume:d_cell_orientation_sign");
  d_cell_orientation_sign.reset(
      static_cast<std::size_t>(n_cells));
  d_cell_orientation_sign.copy_from_host(mb.cell_orientation_sign);

  core::DeviceArray<std::uint8_t> d_cell_nverts_owned("corner_j:compute_multiblock_mesh_quality_rz_volume_cell_margins_with_volume:d_cell_nverts_owned");
  const std::uint8_t* d_cell_nverts = nullptr;
  if (state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)) {
    d_cell_nverts_owned.reset(static_cast<std::size_t>(n_cells));
    d_cell_nverts_owned.copy_from_host(state.mesh.cell_nverts);
    d_cell_nverts = d_cell_nverts_owned.data();
  }

  core::DeviceArray<std::int8_t> d_active_owned("corner_j:compute_multiblock_mesh_quality_rz_volume_cell_margins_with_volume:d_active_owned");
  const std::int8_t* d_active = nullptr;
  if (!state.hydro_active.empty()) {
    d_active_owned.reset(state.hydro_active.size());
    d_active_owned.copy_from_host(state.hydro_active);
    d_active = d_active_owned.data();
  }

  core::DeviceArray<std::uint8_t> d_combined_inactive("corner_j:compute_multiblock_mesh_quality_rz_volume_cell_margins_with_volume:d_combined_inactive");
  const std::uint8_t* d_inactive_mask =
      pole_angular_derefine::combined_inactive_mask_device(
          const_cast<core::State&>(state), d_combined_inactive);

  const int n_requested = static_cast<int>(cells.size());
  const int blocks = (n_requested + 255) / 256;
  mesh_quality_rz_volume_cell_margin_kernel<<<blocks, 256>>>(
      d_requested.data(),
      d_out_cell.data(),
      d_out_admissible.data(),
      d_out_sigma.data(),
      d_out_eta.data(),
      d_out_v0.data(),
      d_out_floor.data(),
      d_x_r_start,
      d_x_z_start,
      d_v_r_half,
      d_v_z_half,
      d_cell_vol_for_floor,
      d_active,
      d_inactive_mask,
      state.mesh.multiblock_cell_node_csr_offsets.data(),
      state.mesh.multiblock_cell_node_csr_indices.data(),
      d_cell_orientation_sign.data(),
      d_cell_nverts,
      n_cells,
      n_requested,
      dt,
      cfg.numerics.hydro.mesh_quality_dt_rz_volume_floor_rel,
      cfg.numerics.hydro.mesh_quality_dt_safety_alpha);
  cuda_check(cudaGetLastError(),
             "mesh-quality RZ-volume cell margin kernel launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "mesh-quality RZ-volume cell margin kernel execution failed");

  std::vector<int> h_cell;
  std::vector<int> h_admissible;
  std::vector<double> h_sigma;
  std::vector<double> h_eta;
  std::vector<double> h_v0;
  std::vector<double> h_floor;
  d_out_cell.copy_to_host(h_cell);
  d_out_admissible.copy_to_host(h_admissible);
  d_out_sigma.copy_to_host(h_sigma);
  d_out_eta.copy_to_host(h_eta);
  d_out_v0.copy_to_host(h_v0);
  d_out_floor.copy_to_host(h_floor);

  for (std::size_t i = 0; i < cells.size(); ++i) {
    out[i].cell = h_cell[i];
    out[i].admissible = h_admissible[i] != 0;
    out[i].sigma_raw = h_sigma[i];
    out[i].eta_safe = h_eta[i];
    out[i].v0 = h_v0[i];
    out[i].floor = h_floor[i];
  }
  return out;
}

namespace {

__global__ void corner_balance_kernel(
    double* __restrict__ min_q_balance,
    int* __restrict__ first_failing_key,
    int* __restrict__ any_non_positive,
    int* __restrict__ any_non_finite,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const double balance_threshold) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int nodes[4] = {
      rz_node_index_2d(i, j, nz),
      rz_node_index_2d(i + 1, j, nz),
      rz_node_index_2d(i + 1, j + 1, nz),
      rz_node_index_2d(i, j + 1, nz),
  };

  double rr[4];
  double zz[4];
  for (int k = 0; k < 4; ++k) {
    const int n = nodes[k];
    rr[k] = x_r[n];
    zz[k] = x_z[n];
  }

  double min_pos_j = 1.0e300;
  double max_j = 0.0;
  int min_pos_corner = 0;
  int local_invalid_key = kNoFailingCornerKey;
  bool has_non_positive = false;
  bool has_non_finite = false;

  for (int corner = 0; corner < 4; ++corner) {
    const double j_current = corner_jacobian_from_quad(rr, zz, corner);
    if (!isfinite(j_current)) {
      has_non_finite = true;
      const int key = c * 4 + corner;
      local_invalid_key = key < local_invalid_key ? key : local_invalid_key;
      continue;
    }
    if (!(j_current > 0.0)) {
      has_non_positive = true;
      const int key = c * 4 + corner;
      local_invalid_key = key < local_invalid_key ? key : local_invalid_key;
      continue;
    }
    if (j_current < min_pos_j) {
      min_pos_j = j_current;
      min_pos_corner = corner;
    }
    max_j = fmax(max_j, j_current);
  }

  if (has_non_finite) {
    atomicExch(any_non_finite, 1);
  }
  if (has_non_positive) {
    atomicExch(any_non_positive, 1);
  }
  if (has_non_finite || has_non_positive) {
    atomic_min_double(min_q_balance, 0.0);
    atomicMin(first_failing_key, local_invalid_key);
    return;
  }

  const double q_balance = min_pos_j / fmax(max_j, kCornerJacobianRatioEps);
  atomic_min_double(min_q_balance, q_balance);
  if (q_balance < balance_threshold) {
    atomicMin(first_failing_key, c * 4 + min_pos_corner);
  }
}

}  // namespace

const char* mesh_quality_metric_string(const MeshQualityFailingMetric metric) {
  switch (metric) {
    case MeshQualityFailingMetric::CornerJ:
      return "mesh_quality_corner_j";
    case MeshQualityFailingMetric::GaussJ:
      return "mesh_quality_gauss_j";
    case MeshQualityFailingMetric::RzVolume:
      return "mesh_quality_rz_volume";
    case MeshQualityFailingMetric::AxisMargin:
      return "mesh_quality_axis_margin";
    case MeshQualityFailingMetric::None:
      return "";
  }
  return "";
}

MeshQualityDtLimit compute_mesh_quality_dt_limit(
    const core::State& state,
    const core::Config& cfg,
    const double* d_x_r_start,
    const double* d_x_z_start,
    const double* d_v_r_half,
    const double* d_v_z_half,
    const double dt,
    const parallel::Reduction* reduction) {
  MeshQualityDtLimit result{};
  result.suggested_dt = dt;
  if (!cfg.numerics.hydro.mesh_quality_dt_cfl_enabled) {
    return result;
  }
  if (state.mesh.topo.multiblock.has_value()) {
    return compute_multiblock_mesh_quality_dt_limit(
        state, cfg, d_x_r_start, d_x_z_start, d_v_r_half, d_v_z_half, dt,
        reduction);
  }

  const bool corner_j_enabled =
      cfg.numerics.hydro.mesh_quality_dt_corner_j_enabled;
  const bool gauss_j_enabled =
      cfg.numerics.hydro.mesh_quality_dt_gauss_j_enabled;
  const bool rz_volume_enabled =
      cfg.numerics.hydro.mesh_quality_dt_rz_volume_enabled;
  const bool axis_margin_enabled =
      cfg.numerics.hydro.mesh_quality_dt_axis_margin_additive &&
      cfg.numerics.has_physical_rz_axis;
  if (!corner_j_enabled && !gauss_j_enabled && !rz_volume_enabled &&
      !axis_margin_enabled) {
    return result;
  }
  if (state.mesh.dim != 2) {
    return result;
  }

  TENRYU_ASSERT(dt > 0.0 && std::isfinite(dt),
                "mesh-quality dt CFL requires finite dt > 0");
  TENRYU_ASSERT(cfg.numerics.hydro.mesh_quality_dt_safety_alpha > 0.0 &&
                    cfg.numerics.hydro.mesh_quality_dt_safety_alpha <= 1.0,
                "mesh-quality dt CFL safety alpha must be in (0, 1]");
  TENRYU_ASSERT(cfg.numerics.hydro.mesh_quality_dt_corner_j_floor_rel > 0.0,
                "mesh-quality corner-J floor_rel must be > 0");
  TENRYU_ASSERT(cfg.numerics.hydro.mesh_quality_dt_gauss_j_floor_rel > 0.0,
                "mesh-quality Gauss-J floor_rel must be > 0");
  TENRYU_ASSERT(cfg.numerics.hydro.mesh_quality_dt_rz_volume_floor_rel > 0.0,
                "mesh-quality RZ-volume floor_rel must be > 0");
  TENRYU_ASSERT(d_x_r_start != nullptr && d_x_z_start != nullptr &&
                    d_v_r_half != nullptr && d_v_z_half != nullptr,
                "mesh-quality dt CFL requires non-null coordinate and velocity fields");
  TENRYU_ASSERT(state.x_r.size() == state.x_z.size(),
                "mesh-quality dt CFL requires matching node fields");

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = nr * nz;
  if (n_cells <= 0) {
    return result;
  }
  const int n_nodes = (nr + 1) * (nz + 1);
  TENRYU_ASSERT(static_cast<int>(state.x_r.size()) == n_nodes,
                "mesh-quality dt CFL requires node field size == n_nodes");
  TENRYU_ASSERT(state.hydro_active.empty() ||
                    static_cast<int>(state.hydro_active.size()) == n_cells,
                "mesh-quality dt CFL requires hydro_active size == n_cells");

  double* d_min_sigma = nullptr;
  unsigned long long* d_first_failing_composite_key = nullptr;
  d_min_sigma = static_cast<double*>(core::device_scratch_acquire(
      "cjq:mesh_quality_dt_min_sigma", sizeof(double)));
  d_first_failing_composite_key =
      static_cast<unsigned long long*>(core::device_scratch_acquire(
          "cjq:mesh_quality_dt_first_failing_key",
          sizeof(unsigned long long)));

  const double one = 1.0;
  const unsigned long long no_composite_key = kNoFailingCompositeKey;
  cuda_check(cudaMemcpy(d_min_sigma, &one, sizeof(double), cudaMemcpyHostToDevice),
             "mesh-quality dt CFL: init min_sigma failed");
  cuda_check(cudaMemcpy(d_first_failing_composite_key, &no_composite_key,
                        sizeof(unsigned long long), cudaMemcpyHostToDevice),
             "mesh-quality dt CFL: init first_failing_key failed");

  std::int8_t* d_active_owned = nullptr;
  const std::int8_t* d_active = nullptr;
  if (!state.hydro_active.empty()) {
    d_active_owned = static_cast<std::int8_t*>(core::device_scratch_acquire(
        "cjq:mesh_quality_dt_hydro_active",
        state.hydro_active.size() * sizeof(std::int8_t)));
    cuda_check(cudaMemcpy(d_active_owned, state.hydro_active.data(),
                          state.hydro_active.size() * sizeof(std::int8_t),
                          cudaMemcpyHostToDevice),
               "mesh-quality dt CFL: cudaMemcpy hydro_active failed");
    d_active = d_active_owned;
  }

  const int blocks = (n_cells + 255) / 256;
  mesh_quality_dt_limit_kernel<<<blocks, 256>>>(
      d_min_sigma,
      d_first_failing_composite_key,
      d_x_r_start,
      d_x_z_start,
      d_v_r_half,
      d_v_z_half,
      d_active,
      nr,
      nz,
      dt,
      corner_j_enabled ? 1 : 0,
      gauss_j_enabled ? 1 : 0,
      rz_volume_enabled ? 1 : 0,
      axis_margin_enabled ? 1 : 0,
      cfg.numerics.has_physical_rz_axis ? 1 : 0,
      cfg.numerics.hydro.mesh_quality_dt_corner_j_floor_rel,
      cfg.numerics.hydro.mesh_quality_dt_gauss_j_floor_rel,
      cfg.numerics.hydro.mesh_quality_dt_rz_volume_floor_rel);
  cuda_check(cudaGetLastError(), "mesh-quality dt CFL kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "mesh-quality dt CFL kernel execution failed");

  unsigned long long local_composite_key = no_composite_key;
  cuda_check(cudaMemcpy(&local_composite_key, d_first_failing_composite_key,
                        sizeof(unsigned long long), cudaMemcpyDeviceToHost),
             "mesh-quality dt CFL: copy first_failing_key failed");

  std::uint64_t global_composite_key =
      static_cast<std::uint64_t>(local_composite_key);
  if (reduction != nullptr) {
    global_composite_key = reduction->allreduce_min(global_composite_key);
  }

  double global_sigma = 1.0;
  if (global_composite_key != no_composite_key) {
    const auto sigma_bits =
        static_cast<std::uint32_t>(global_composite_key >> 32);
    global_sigma = sigma_from_sortable_uint32(sigma_bits);
    if (!(global_sigma >= 0.0) || !std::isfinite(global_sigma)) {
      global_sigma = 0.0;
    }
    global_sigma = std::min(global_sigma, 1.0);
  }
  result.sigma_safe = global_sigma;
  if (global_composite_key == no_composite_key) {
    result.admissible = true;
    result.suggested_dt = dt;
    return result;
  }

  result.admissible = false;
  result.suggested_dt =
      cfg.numerics.hydro.mesh_quality_dt_safety_alpha * global_sigma * dt;

  const auto detail_key =
      static_cast<std::uint32_t>(global_composite_key & 0xffffffffULL);
  const int first_slot = composite_detail_slot(detail_key);
  result.first_cell = composite_detail_cell(detail_key);
  if (first_slot >= 0 && first_slot < kMeshQualityGaussSlotOffset) {
    result.metric = MeshQualityFailingMetric::CornerJ;
    result.first_corner_or_gauss = first_slot;
  } else if (first_slot >= kMeshQualityGaussSlotOffset &&
             first_slot < kMeshQualityRzSlot) {
    result.metric = MeshQualityFailingMetric::GaussJ;
    result.first_corner_or_gauss = first_slot - kMeshQualityGaussSlotOffset;
  } else if (first_slot == kMeshQualityRzSlot) {
    result.metric = MeshQualityFailingMetric::RzVolume;
    result.first_corner_or_gauss = -1;
  } else if (first_slot == kMeshQualityAxisSlot) {
    result.metric = MeshQualityFailingMetric::AxisMargin;
    result.first_corner_or_gauss = -1;
  }
  return result;
}

CornerJacobianQualityResult check_trial_lagrangian_corner_jacobian_host(
    const int nr,
    const int nz,
    const std::vector<double>& r_base,
    const std::vector<double>& z_base,
    const std::vector<double>& v_r,
    const std::vector<double>& v_z,
    const double dt,
    const double floor_eps,
    const parallel::Reduction* reduction,
    const std::vector<std::int8_t>& hydro_active,
    const bool axis_margin_guard_enabled,
    const bool has_physical_rz_axis) {
  TENRYU_ASSERT(dt > 0.0, "corner-J trial scale requires dt > 0");
  TENRYU_ASSERT(floor_eps >= 0.0 && floor_eps < 1.0,
                "corner-J floor_eps must be in [0, 1)");
  const int n_cells = nr * nz;
  const int n_nodes = (nr + 1) * (nz + 1);
  TENRYU_ASSERT(hydro_active.empty() ||
                    static_cast<int>(hydro_active.size()) == n_cells,
                "corner-J trial check requires hydro_active size == n_cells");
  TENRYU_ASSERT(static_cast<int>(r_base.size()) == n_nodes &&
                    static_cast<int>(z_base.size()) == n_nodes &&
                    static_cast<int>(v_r.size()) == n_nodes &&
                    static_cast<int>(v_z.size()) == n_nodes,
                "corner-J trial scale requires node field sizes == n_nodes");

  CornerJacobianQualityResult result{};
  result.suggested_dt = dt;
  double local_min_current = std::numeric_limits<double>::infinity();
  double local_min_trial = std::numeric_limits<double>::infinity();
  double local_min_ratio = std::numeric_limits<double>::infinity();
  double local_suggested_dt = dt;

  for (int c = 0; c < n_cells; ++c) {
    if (!hydro_active.empty() &&
        hydro_active[static_cast<std::size_t>(c)] == 0) {
      continue;
    }
    double r[4];
    double z[4];
    double vr[4];
    double vz[4];
    gather_cell_nodes(r_base, z_base, v_r, v_z, nz, c, r, z, vr, vz);
    const int i = c / nz;
    if (axis_margin_guard_enabled && has_physical_rz_axis && i == 0) {
      double rr[4];
      double zz[4];
      for (int k = 0; k < 4; ++k) {
        rr[k] = r[k] + dt * vr[k];
        zz[k] = z[k] + dt * vz[k];
      }
      const auto trial =
          evaluate_axis_margin_predicate_quad(r, z, rr, zz, floor_eps);
      local_min_trial = std::min(local_min_trial, trial.min_margin);
      local_min_current = std::min(local_min_current, trial.min_margin);
      local_min_ratio = std::min(local_min_ratio, trial.admissible ? 1.0 : 0.0);
      if (!trial.admissible) {
        if (result.first_failing_cell < 0) {
          result.first_failing_cell = c;
          result.first_failing_corner = 0;
        }
        result.admissible = false;
        local_suggested_dt = std::min(
            local_suggested_dt,
            axis_margin_limited_tau_host(r, z, vr, vz, dt, floor_eps));
      }
      continue;
    }
    for (int corner = 0; corner < 4; ++corner) {
      const double j_current = corner_jacobian_from_quad(r, z, corner);
      const double j_trial = corner_jacobian_at_tau_host(r, z, vr, vz, corner, dt);
      local_min_current = std::min(local_min_current, j_current);
      local_min_trial = std::min(local_min_trial, j_trial);
      if (j_current > 0.0 && std::isfinite(j_current)) {
        local_min_ratio =
            std::min(local_min_ratio, j_trial / std::max(j_current, kCornerJacobianRatioEps));
      }
      const double floor = (j_current > 0.0 && std::isfinite(j_current))
                               ? floor_eps * j_current
                               : 0.0;
      if (!(j_current > 0.0) || !std::isfinite(j_current) || !(j_trial >= floor) ||
          !std::isfinite(j_trial)) {
        if (result.first_failing_cell < 0) {
          result.first_failing_cell = c;
          result.first_failing_corner = corner;
        }
        result.admissible = false;
        const double corner_dt =
            (j_current > 0.0 && std::isfinite(j_current))
                ? corner_limited_tau_host(r, z, vr, vz, corner, dt, floor)
                : 0.0;
        local_suggested_dt = std::min(local_suggested_dt, corner_dt);
      }
    }
  }

  if (!std::isfinite(local_min_ratio)) {
    local_min_ratio = 1.0;
  }
  if (reduction != nullptr) {
    const double bad_local = result.admissible ? 0.0 : 1.0;
    result.admissible = reduction->allreduce_max(bad_local) < 0.5;
    local_min_current = reduction->allreduce_min(local_min_current);
    local_min_trial = reduction->allreduce_min(local_min_trial);
    local_min_ratio = reduction->allreduce_min(local_min_ratio);
    local_suggested_dt = reduction->allreduce_min(local_suggested_dt);
  }
  result.min_current_j = local_min_current;
  result.min_trial_j = local_min_trial;
  result.min_trial_ratio = local_min_ratio;
  result.suggested_dt = result.admissible ? dt : local_suggested_dt;
  if (result.admissible) {
    result.first_failing_cell = -1;
    result.first_failing_corner = -1;
  }
  return result;
}

CornerJacobianQualityResult check_trial_lagrangian_corner_jacobian_fields(
    const int nr,
    const int nz,
    const core::NodeField1D& r_base,
    const core::NodeField1D& z_base,
    const core::NodeField1D& v_r,
    const core::NodeField1D& v_z,
    const double dt,
    const double floor_eps,
    const parallel::Reduction* reduction,
    const std::vector<std::int8_t>& hydro_active,
    const bool axis_margin_guard_enabled,
    const bool has_physical_rz_axis) {
  return check_trial_lagrangian_corner_jacobian_host(
      nr, nz, copy_field_to_vector(r_base), copy_field_to_vector(z_base),
      copy_field_to_vector(v_r), copy_field_to_vector(v_z), dt, floor_eps,
      reduction, hydro_active, axis_margin_guard_enabled, has_physical_rz_axis);
}

CornerJacobianScaleResult compute_corner_jacobian_trial_scale(
    const core::State& state,
    const double dt_in,
    const double floor_eps,
    const parallel::Reduction* reduction,
    const CellRegime* d_cell_regime,
    const bool regime_aware_corner_j_guard_enabled,
    const bool axis_margin_guard_enabled,
    const double legacy_trigger_scale,
    const bool has_physical_rz_axis) {
  CornerJacobianScaleResult result{};
  result.trigger_threshold = legacy_trigger_scale;
  if (state.mesh.dim != 2 || !(dt_in > 0.0) || !std::isfinite(dt_in)) {
    return result;
  }
  TENRYU_ASSERT(floor_eps >= 0.0 && floor_eps < 1.0,
                "corner-J floor_eps must be in [0, 1)");
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = nr * nz;
  if (n_cells <= 0) {
    return result;
  }
  TENRYU_ASSERT(state.x_r.size() == state.v_r.size() &&
                    state.x_z.size() == state.v_z.size() &&
                    state.x_r.size() == state.x_z.size(),
                "corner-J trial scale requires matching node fields");

  double* d_min_scale = nullptr;
  int* d_first_failing_cell = nullptr;
  int* d_first_failing_corner = nullptr;
  int* d_any_trigger = nullptr;
  int* d_first_trigger_key = nullptr;
  d_min_scale = static_cast<double*>(core::device_scratch_acquire(
      "cjq:trial_scale_min_scale", sizeof(double)));
  d_first_failing_cell = static_cast<int*>(core::device_scratch_acquire(
      "cjq:trial_scale_first_failing_cell", sizeof(int)));
  d_first_failing_corner = static_cast<int*>(core::device_scratch_acquire(
      "cjq:trial_scale_first_failing_corner", sizeof(int)));
  d_any_trigger = static_cast<int*>(core::device_scratch_acquire(
      "cjq:trial_scale_any_trigger", sizeof(int)));
  d_first_trigger_key = static_cast<int*>(core::device_scratch_acquire(
      "cjq:trial_scale_first_trigger_key", sizeof(int)));

  const double one = 1.0;
  const int zero = 0;
  const int no_cell = std::numeric_limits<int>::max();
  const int no_key = kNoFailingCornerKey;
  cuda_check(cudaMemcpy(d_min_scale, &one, sizeof(double), cudaMemcpyHostToDevice),
             "corner-J trial scale: init min_scale failed");
  cuda_check(cudaMemcpy(d_first_failing_cell, &no_cell, sizeof(int),
                        cudaMemcpyHostToDevice),
             "corner-J trial scale: init first_failing_cell failed");
  cuda_check(cudaMemcpy(d_first_failing_corner, &no_cell, sizeof(int),
                        cudaMemcpyHostToDevice),
             "corner-J trial scale: init first_failing_corner failed");
  cuda_check(cudaMemcpy(d_any_trigger, &zero, sizeof(int), cudaMemcpyHostToDevice),
             "corner-J trial scale: init any_trigger failed");
  cuda_check(cudaMemcpy(d_first_trigger_key, &no_key, sizeof(int),
                        cudaMemcpyHostToDevice),
             "corner-J trial scale: init first_trigger_key failed");

  std::int8_t* d_active = nullptr;
  if (!state.hydro_active.empty()) {
    d_active = static_cast<std::int8_t*>(core::device_scratch_acquire(
        "cjq:trial_scale_hydro_active",
        state.hydro_active.size() * sizeof(std::int8_t)));
    cuda_check(cudaMemcpy(d_active, state.hydro_active.data(),
                          state.hydro_active.size() * sizeof(std::int8_t),
                          cudaMemcpyHostToDevice),
               "corner-J trial scale: cudaMemcpy hydro_active failed");
  }

  const int blocks = (n_cells + 255) / 256;
  corner_jacobian_trial_scale_kernel<<<blocks, 256>>>(
      d_min_scale, d_first_failing_cell, d_first_failing_corner,
      d_any_trigger, d_first_trigger_key,
      state.x_r.data(), state.x_z.data(), state.v_r.data(), state.v_z.data(),
      d_active, d_cell_regime, nr, nz, dt_in, floor_eps,
      regime_aware_corner_j_guard_enabled ? 1 : 0,
      axis_margin_guard_enabled ? 1 : 0,
      has_physical_rz_axis ? 1 : 0,
      legacy_trigger_scale);
  cuda_check(cudaGetLastError(), "corner-J trial scale kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "corner-J trial scale kernel execution failed");

  double local_scale = 1.0;
  int first_cell = no_cell;
  int first_corner = no_cell;
  int local_any_trigger = 0;
  int local_trigger_key = no_key;
  cuda_check(cudaMemcpy(&local_scale, d_min_scale, sizeof(double),
                        cudaMemcpyDeviceToHost),
             "corner-J trial scale: copy min_scale failed");
  cuda_check(cudaMemcpy(&first_cell, d_first_failing_cell, sizeof(int),
                        cudaMemcpyDeviceToHost),
             "corner-J trial scale: copy first_failing_cell failed");
  cuda_check(cudaMemcpy(&first_corner, d_first_failing_corner, sizeof(int),
                        cudaMemcpyDeviceToHost),
             "corner-J trial scale: copy first_failing_corner failed");
  cuda_check(cudaMemcpy(&local_any_trigger, d_any_trigger, sizeof(int),
                        cudaMemcpyDeviceToHost),
             "corner-J trial scale: copy any_trigger failed");
  cuda_check(cudaMemcpy(&local_trigger_key, d_first_trigger_key, sizeof(int),
                        cudaMemcpyDeviceToHost),
             "corner-J trial scale: copy first_trigger_key failed");

  double global_scale = local_scale;
  double global_any_trigger = static_cast<double>(local_any_trigger);
  double global_trigger_key = static_cast<double>(local_trigger_key);
  if (reduction != nullptr) {
    global_scale = reduction->allreduce_min(global_scale);
    global_any_trigger = reduction->allreduce_max(global_any_trigger);
    global_trigger_key = reduction->allreduce_min(global_trigger_key);
  }
  if (!(global_scale >= 0.0) || !std::isfinite(global_scale)) {
    global_scale = 0.0;
  }
  result.scale = std::min(global_scale, 1.0);
  result.trigger_recommended =
      regime_aware_corner_j_guard_enabled ||
              (axis_margin_guard_enabled && has_physical_rz_axis)
          ? (global_any_trigger > 0.5)
          : (result.scale < legacy_trigger_scale);
  result.trigger_scale = result.scale;
  result.first_failing_cell = first_cell == no_cell ? -1 : first_cell;
  result.first_failing_corner = first_corner == no_cell ? -1 : first_corner;
  const int first_trigger_key = static_cast<int>(global_trigger_key);
  result.first_trigger_cell =
      first_trigger_key == no_key ? -1 : first_trigger_key / 4;
  result.first_trigger_corner =
      first_trigger_key == no_key ? -1 : first_trigger_key % 4;
  return result;
}

tenryu::coupling::HydroStepResult compute_in_hydro_candidate_mesh_guard(
    const core::State& state,
    const double dt,
    const double* d_x_r_start,
    const double* d_x_z_start,
    const double* d_x_r_candidate,
    const double* d_x_z_candidate,
    const core::Config& cfg,
    const tenryu::coupling::HydroFailureStage stage,
    const parallel::Reduction* reduction,
    const std::int8_t* d_hydro_active,
    const CellRegime* d_cell_regime) {
  tenryu::coupling::HydroStepResult result{};
  const bool corner_j_guard_enabled =
      cfg.numerics.hydro.in_hydro_corner_j_guard_enabled;
  const bool gauss_j_guard_enabled =
      cfg.numerics.hydro.in_hydro_gauss_j_guard_enabled;
  const bool rz_volume_guard_enabled =
      cfg.numerics.hydro.in_hydro_rz_volume_guard_enabled;
  if (!corner_j_guard_enabled && !gauss_j_guard_enabled &&
      !rz_volume_guard_enabled) {
    return result;
  }
  result.stage = stage;
  result.suggested_dt = dt;
  result.trial_scale = 1.0;
  if (state.mesh.dim != 2) {
    return result;
  }
  TENRYU_ASSERT(dt > 0.0 && std::isfinite(dt),
                "in-hydro corner-J guard requires finite dt > 0");
  TENRYU_ASSERT(cfg.numerics.hydro.corner_jacobian_floor_eps >= 0.0 &&
                    cfg.numerics.hydro.corner_jacobian_floor_eps < 1.0,
                "corner-J floor_eps must be in [0, 1)");
  TENRYU_ASSERT(cfg.numerics.hydro.in_hydro_gauss_j_floor_rel > 0.0,
                "in-hydro Gauss-J floor_rel must be > 0");
  TENRYU_ASSERT(cfg.numerics.hydro.in_hydro_rz_volume_floor_rel > 0.0,
                "in-hydro RZ-volume floor_rel must be > 0");
  TENRYU_ASSERT(d_x_r_start != nullptr && d_x_z_start != nullptr &&
                    d_x_r_candidate != nullptr && d_x_z_candidate != nullptr,
                "in-hydro corner-J guard requires non-null coordinate fields");
  TENRYU_ASSERT(state.x_r.size() == state.x_z.size(),
                "in-hydro corner-J guard requires matching node fields");
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = nr * nz;
  if (n_cells <= 0) {
    return result;
  }
  const int n_nodes = (nr + 1) * (nz + 1);
  TENRYU_ASSERT(static_cast<int>(state.x_r.size()) == n_nodes,
                "in-hydro corner-J guard requires node field size == n_nodes");
  TENRYU_ASSERT(state.hydro_active.empty() ||
                    static_cast<int>(state.hydro_active.size()) == n_cells,
                "in-hydro corner-J guard requires hydro_active size == n_cells");

  double* d_min_corner_j = nullptr;
  double* d_min_gauss_j = nullptr;
  double* d_min_rz_volume = nullptr;
  double* d_min_sigma = nullptr;
  int* d_any_failure = nullptr;
  int* d_any_gauss_j_failure = nullptr;
  int* d_any_rz_volume_failure = nullptr;
  unsigned long long* d_first_failing_composite_key = nullptr;
  d_min_corner_j = static_cast<double*>(core::device_scratch_acquire(
      "cjq:in_hydro_min_corner_j", sizeof(double)));
  if (gauss_j_guard_enabled) {
    d_min_gauss_j = static_cast<double*>(core::device_scratch_acquire(
        "cjq:in_hydro_min_gauss_j", sizeof(double)));
    d_any_gauss_j_failure = static_cast<int*>(core::device_scratch_acquire(
        "cjq:in_hydro_any_gauss_j_failure", sizeof(int)));
  }
  if (rz_volume_guard_enabled) {
    d_min_rz_volume = static_cast<double*>(core::device_scratch_acquire(
        "cjq:in_hydro_min_rz_volume", sizeof(double)));
    d_any_rz_volume_failure = static_cast<int*>(core::device_scratch_acquire(
        "cjq:in_hydro_any_rz_volume_failure", sizeof(int)));
  }
  d_min_sigma = static_cast<double*>(core::device_scratch_acquire(
      "cjq:in_hydro_min_sigma", sizeof(double)));
  d_any_failure = static_cast<int*>(core::device_scratch_acquire(
      "cjq:in_hydro_any_failure", sizeof(int)));
  d_first_failing_composite_key =
      static_cast<unsigned long long*>(core::device_scratch_acquire(
          "cjq:in_hydro_first_failing_key", sizeof(unsigned long long)));

  const double inf = std::numeric_limits<double>::infinity();
  const double one = 1.0;
  const int zero = 0;
  const unsigned long long no_composite_key = kNoFailingCompositeKey;
  cuda_check(cudaMemcpy(d_min_corner_j, &inf, sizeof(double), cudaMemcpyHostToDevice),
             "in-hydro corner-J guard: init min_corner_j failed");
  if (gauss_j_guard_enabled) {
    cuda_check(cudaMemcpy(d_min_gauss_j, &inf, sizeof(double),
                          cudaMemcpyHostToDevice),
               "in-hydro corner-J guard: init min_gauss_j failed");
    cuda_check(cudaMemcpy(d_any_gauss_j_failure, &zero, sizeof(int),
                          cudaMemcpyHostToDevice),
               "in-hydro corner-J guard: init any_gauss_j_failure failed");
  }
  if (rz_volume_guard_enabled) {
    cuda_check(cudaMemcpy(d_min_rz_volume, &inf, sizeof(double),
                          cudaMemcpyHostToDevice),
               "in-hydro corner-J guard: init min_rz_volume failed");
    cuda_check(cudaMemcpy(d_any_rz_volume_failure, &zero, sizeof(int),
                          cudaMemcpyHostToDevice),
               "in-hydro corner-J guard: init any_rz_volume_failure failed");
  }
  cuda_check(cudaMemcpy(d_min_sigma, &one, sizeof(double), cudaMemcpyHostToDevice),
             "in-hydro corner-J guard: init min_sigma failed");
  cuda_check(cudaMemcpy(d_any_failure, &zero, sizeof(int), cudaMemcpyHostToDevice),
             "in-hydro corner-J guard: init any_failure failed");
  cuda_check(cudaMemcpy(d_first_failing_composite_key, &no_composite_key,
                        sizeof(unsigned long long),
                        cudaMemcpyHostToDevice),
             "in-hydro corner-J guard: init first_failing_key failed");

  std::int8_t* d_active_owned = nullptr;
  const std::int8_t* d_active = d_hydro_active;
  if (d_active == nullptr && !state.hydro_active.empty()) {
    d_active_owned = static_cast<std::int8_t*>(core::device_scratch_acquire(
        "cjq:in_hydro_hydro_active",
        state.hydro_active.size() * sizeof(std::int8_t)));
    cuda_check(cudaMemcpy(d_active_owned, state.hydro_active.data(),
                          state.hydro_active.size() * sizeof(std::int8_t),
                          cudaMemcpyHostToDevice),
               "in-hydro corner-J guard: cudaMemcpy hydro_active failed");
    d_active = d_active_owned;
  }

  const int blocks = (n_cells + 255) / 256;
  in_hydro_candidate_corner_j_guard_kernel<<<blocks, 256>>>(
      d_min_corner_j, nullptr, d_min_gauss_j, d_min_rz_volume, d_min_sigma,
      d_any_failure, d_any_gauss_j_failure, d_any_rz_volume_failure,
      d_first_failing_composite_key, d_x_r_start, d_x_z_start, d_x_r_candidate,
      d_x_z_candidate, d_active, nr, nz,
      cfg.numerics.hydro.corner_jacobian_floor_eps,
      corner_j_guard_enabled ? 1 : 0,
      cfg.numerics.hydro.axis_margin_guard_enabled ? 1 : 0,
      cfg.numerics.has_physical_rz_axis ? 1 : 0,
      gauss_j_guard_enabled ? 1 : 0,
      cfg.numerics.hydro.in_hydro_gauss_j_floor_rel,
      rz_volume_guard_enabled ? 1 : 0,
      cfg.numerics.hydro.in_hydro_rz_volume_floor_rel,
      cfg.numerics.hydro.axis_margin_additive_in_action8_enabled ? 1 : 0);
  cuda_check(cudaGetLastError(), "in-hydro corner-J guard kernel launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "in-hydro corner-J guard kernel execution failed");

  double local_min_corner_j = inf;
  double local_min_gauss_j = inf;
  double local_min_rz_volume = inf;
  double local_sigma = 1.0;
  int local_any_failure = 0;
  int local_any_gauss_j_failure = 0;
  int local_any_rz_volume_failure = 0;
  unsigned long long local_composite_key = no_composite_key;
  cuda_check(cudaMemcpy(&local_min_corner_j, d_min_corner_j, sizeof(double),
                        cudaMemcpyDeviceToHost),
             "in-hydro corner-J guard: copy min_corner_j failed");
  if (gauss_j_guard_enabled) {
    cuda_check(cudaMemcpy(&local_min_gauss_j, d_min_gauss_j, sizeof(double),
                          cudaMemcpyDeviceToHost),
               "in-hydro corner-J guard: copy min_gauss_j failed");
    cuda_check(cudaMemcpy(&local_any_gauss_j_failure,
                          d_any_gauss_j_failure,
                          sizeof(int),
                          cudaMemcpyDeviceToHost),
               "in-hydro corner-J guard: copy any_gauss_j_failure failed");
  }
  if (rz_volume_guard_enabled) {
    cuda_check(cudaMemcpy(&local_min_rz_volume, d_min_rz_volume, sizeof(double),
                          cudaMemcpyDeviceToHost),
               "in-hydro corner-J guard: copy min_rz_volume failed");
    cuda_check(cudaMemcpy(&local_any_rz_volume_failure,
                          d_any_rz_volume_failure,
                          sizeof(int),
                          cudaMemcpyDeviceToHost),
               "in-hydro corner-J guard: copy any_rz_volume_failure failed");
  }
  cuda_check(cudaMemcpy(&local_sigma, d_min_sigma, sizeof(double),
                        cudaMemcpyDeviceToHost),
             "in-hydro corner-J guard: copy min_sigma failed");
  cuda_check(cudaMemcpy(&local_any_failure, d_any_failure, sizeof(int),
                        cudaMemcpyDeviceToHost),
             "in-hydro corner-J guard: copy any_failure failed");
  cuda_check(cudaMemcpy(&local_composite_key, d_first_failing_composite_key,
                        sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost),
             "in-hydro corner-J guard: copy first_failing_key failed");

  double global_g0_rel = 0.0;
  double global_g1_rel = 0.0;
  double global_dt_star_est = 0.0;
  bool telemetry_dt_sensitive = false;
  bool telemetry_repair_sensitive = false;
  if (corner_j_guard_enabled && stage24_axis_classification_active(cfg)) {
    double* d_max_abs_corner_j = nullptr;
    d_max_abs_corner_j = static_cast<double*>(core::device_scratch_acquire(
        "cjq:in_hydro_max_abs_corner_j", sizeof(double)));
    const double zero_double = 0.0;
    cuda_check(cudaMemcpy(d_min_corner_j, &inf, sizeof(double),
                          cudaMemcpyHostToDevice),
               "in-hydro corner-J guard: reset telemetry min_corner_j failed");
    cuda_check(cudaMemcpy(d_min_sigma, &one, sizeof(double), cudaMemcpyHostToDevice),
               "in-hydro corner-J guard: reset telemetry min_sigma failed");
    cuda_check(cudaMemcpy(d_any_failure, &zero, sizeof(int), cudaMemcpyHostToDevice),
               "in-hydro corner-J guard: reset telemetry any_failure failed");
    cuda_check(cudaMemcpy(d_first_failing_composite_key, &no_composite_key,
                          sizeof(unsigned long long),
                          cudaMemcpyHostToDevice),
               "in-hydro corner-J guard: reset telemetry first_failing_key failed");
    cuda_check(cudaMemcpy(d_max_abs_corner_j, &zero_double, sizeof(double),
                          cudaMemcpyHostToDevice),
               "in-hydro corner-J guard: init telemetry max_abs_corner_j failed");
    in_hydro_candidate_corner_j_guard_kernel<<<blocks, 256>>>(
        d_min_corner_j, d_max_abs_corner_j, nullptr, nullptr, d_min_sigma,
        d_any_failure, nullptr, nullptr, d_first_failing_composite_key,
        d_x_r_start, d_x_z_start, d_x_r_start, d_x_z_start, d_active, nr, nz,
        cfg.numerics.hydro.corner_jacobian_floor_eps, 1,
        cfg.numerics.hydro.axis_margin_guard_enabled ? 1 : 0,
        cfg.numerics.has_physical_rz_axis ? 1 : 0,
        0, cfg.numerics.hydro.in_hydro_gauss_j_floor_rel,
        0, cfg.numerics.hydro.in_hydro_rz_volume_floor_rel,
        cfg.numerics.hydro.axis_margin_additive_in_action8_enabled ? 1 : 0);
    cuda_check(cudaGetLastError(),
               "in-hydro corner-J telemetry kernel launch failed");
    cuda_check(cudaDeviceSynchronize(),
               "in-hydro corner-J telemetry kernel execution failed");
    double local_min_corner_j_start = inf;
    double local_max_abs_corner_j = 0.0;
    cuda_check(cudaMemcpy(&local_min_corner_j_start,
                          d_min_corner_j,
                          sizeof(double),
                          cudaMemcpyDeviceToHost),
               "in-hydro corner-J guard: copy telemetry min_corner_j failed");
    cuda_check(cudaMemcpy(&local_max_abs_corner_j,
                          d_max_abs_corner_j,
                          sizeof(double),
                          cudaMemcpyDeviceToHost),
               "in-hydro corner-J guard: copy telemetry max_abs_corner_j failed");
    double global_min_corner_j_start = local_min_corner_j_start;
    double global_max_abs_corner_j = local_max_abs_corner_j;
    if (reduction != nullptr) {
      global_min_corner_j_start =
          reduction->allreduce_min(global_min_corner_j_start);
      global_max_abs_corner_j =
          reduction->allreduce_max(global_max_abs_corner_j);
    }
    double scale_j = global_max_abs_corner_j;
    if (!(scale_j > 0.0) || !std::isfinite(scale_j)) {
      scale_j = 1.0;
    }
    if (std::isfinite(global_min_corner_j_start) &&
        std::isfinite(local_min_corner_j)) {
      global_g0_rel = global_min_corner_j_start / scale_j;
      global_g1_rel = local_min_corner_j / scale_j;
      if (reduction != nullptr) {
        global_g1_rel = reduction->allreduce_min(global_g1_rel);
      }
      const double g_dot_est = (global_g1_rel - global_g0_rel) / dt;
      telemetry_dt_sensitive =
          global_g0_rel > kStage24TelemetryEpsRel &&
          global_g1_rel <= kStage24TelemetryEpsRel;
      telemetry_repair_sensitive =
          global_g0_rel <= kStage24TelemetryEpsRel;
      if (global_g0_rel > kStage24TelemetryEpsRel && g_dot_est < 0.0) {
        global_dt_star_est = global_g0_rel * dt / -g_dot_est;
      }
    }
  }

  double global_min_corner_j = local_min_corner_j;
  double global_min_gauss_j = local_min_gauss_j;
  double global_min_rz_volume = local_min_rz_volume;
  std::uint64_t global_composite_key =
      static_cast<std::uint64_t>(local_composite_key);
  double global_any_failure = static_cast<double>(local_any_failure);
  double global_any_gauss_j_failure =
      static_cast<double>(local_any_gauss_j_failure);
  double global_any_rz_volume_failure =
      static_cast<double>(local_any_rz_volume_failure);
  if (reduction != nullptr) {
    global_min_corner_j = reduction->allreduce_min(global_min_corner_j);
    global_min_gauss_j = reduction->allreduce_min(global_min_gauss_j);
    global_min_rz_volume = reduction->allreduce_min(global_min_rz_volume);
    global_composite_key = reduction->allreduce_min(global_composite_key);
    global_any_failure = reduction->allreduce_max(global_any_failure);
    global_any_gauss_j_failure =
        reduction->allreduce_max(global_any_gauss_j_failure);
    global_any_rz_volume_failure =
        reduction->allreduce_max(global_any_rz_volume_failure);
  }

  global_min_corner_j = sanitize_in_hydro_reduced_min_metric(
      global_min_corner_j, global_any_failure);
  double global_sigma = 1.0;
  if (global_composite_key != no_composite_key) {
    const auto sigma_bits =
        static_cast<std::uint32_t>(global_composite_key >> 32);
    global_sigma = sigma_from_sortable_uint32(sigma_bits);
    if (!(global_sigma >= 0.0) || !std::isfinite(global_sigma)) {
      global_sigma = 0.0;
    }
    global_sigma = std::min(global_sigma, 1.0);
  } else if (!(local_sigma >= 0.0) || !std::isfinite(local_sigma)) {
    global_sigma = 0.0;
  }
  result.min_corner_j = global_min_corner_j;
  result.min_gauss_j_trial = global_min_gauss_j;
  result.min_rz_volume_trial = global_min_rz_volume;
  result.gauss_j_failed = global_any_gauss_j_failure > 0.5;
  result.rz_volume_failed = global_any_rz_volume_failure > 0.5;
  result.min_metric = global_min_corner_j;
  result.trial_scale = global_sigma;
  result.suggested_dt = kInHydroCornerJDtSafety * global_sigma * dt;
  if (corner_j_guard_enabled && stage24_axis_classification_active(cfg)) {
    result.margin_rel = global_g1_rel;
    result.dt_sensitive = telemetry_dt_sensitive;
    result.state_sensitive = telemetry_repair_sensitive;
    result.dt_star_est = global_dt_star_est;
  }

  if (global_any_failure < 0.5) {
    result.trial_scale = 1.0;
    result.suggested_dt = dt;
    if (!std::isfinite(result.min_corner_j)) {
      result.min_corner_j = 0.0;
      result.min_metric = 0.0;
    }
    return result;
  }

  const auto first_detail_key =
      static_cast<std::uint32_t>(global_composite_key & 0xffffffffULL);
  const int first_cell =
      global_composite_key == no_composite_key
          ? -1
          : composite_detail_cell(first_detail_key);
  const int first_slot =
      global_composite_key == no_composite_key
          ? -1
          : composite_detail_slot(first_detail_key);
  const bool first_gauss =
      first_slot >= kInHydroGaussSlotOffset && first_slot < kInHydroRzSlot;
  const bool first_rz = first_slot == kInHydroRzSlot;
  const bool first_axis = first_slot == kInHydroAxisSlot;
  result.admissible = false;
  result.retry_required = true;
  result.reason = first_gauss ? "in_hydro_gauss_j"
                  : first_rz ? "in_hydro_rz_volume"
                  : first_axis ? "in_hydro_axis_margin"
                             : "in_hydro_corner_j";
  result.first_failing_cell = first_cell;
  result.first_failing_corner =
      global_composite_key == no_composite_key || first_rz || first_axis
          ? -1
          : (first_gauss ? first_slot - kInHydroGaussSlotOffset : first_slot);
  result.retry_action = tenryu::coupling::RetryActionHint::ReduceDtOnly;
  result.first_failing_i = first_cell < 0 ? -1 : first_cell / nz;
  result.first_failing_j = first_cell < 0 ? -1 : first_cell % nz;
  apply_in_hydro_location_retry_hint(result, nr, nz);
  const bool structural_location_classified =
      result.regime == MeshFailureRegime::AxisFace ||
      result.regime == MeshFailureRegime::DomainBoundary;
  if (!structural_location_classified &&
      cfg.numerics.hydro.regime_aware_corner_j_guard_enabled &&
      d_cell_regime != nullptr && first_cell >= 0) {
    CellRegime first_regime;
    cuda_check(cudaMemcpy(&first_regime,
                          d_cell_regime + first_cell,
                          sizeof(CellRegime),
                          cudaMemcpyDeviceToHost),
               "in-hydro corner-J guard: copy failing cell regime failed");
    result.regime =
        static_cast<MeshFailureRegime>(first_regime.primary_regime);
    switch (result.regime) {
      case MeshFailureRegime::AxisFace:
      case MeshFailureRegime::AxisBand:
        result.retry_action =
            tenryu::coupling::RetryActionHint::ForceAxisSpinePlusLocalAle;
        break;
      case MeshFailureRegime::DomainBoundary:
        result.retry_action =
            tenryu::coupling::RetryActionHint::ForceBoundaryPatchRepair;
        break;
      case MeshFailureRegime::InteriorCD:
        result.retry_action =
            tenryu::coupling::RetryActionHint::ForceCdLocalRezone;
        break;
      case MeshFailureRegime::InteriorSmooth:
      case MeshFailureRegime::Unknown:
        result.retry_action =
            tenryu::coupling::RetryActionHint::ForceFullWinslow;
        break;
      case MeshFailureRegime::VoidOrInactive:
        break;
    }
  }
  result.min_metric = first_gauss ? result.min_gauss_j_trial
                      : first_rz ? result.min_rz_volume_trial
                                 : result.min_corner_j;
  result.failing_value = std::min(result.min_metric, 0.0);
  return result;
}

CornerBalanceResult evaluate_corner_balance(
    const core::State& state,
    const double balance_threshold,
    const parallel::Reduction* reduction) {
  CornerBalanceResult result{};
  if (state.mesh.dim != 2) {
    return result;
  }
  TENRYU_ASSERT(balance_threshold > 0.0 && balance_threshold < 1.0,
                "corner balance threshold must be in (0, 1)");
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = nr * nz;
  if (n_cells <= 0) {
    return result;
  }
  TENRYU_ASSERT(state.x_r.size() == state.x_z.size(),
                "corner balance requires matching node fields");

  double* d_min_q_balance = nullptr;
  int* d_first_failing_key = nullptr;
  int* d_any_non_positive = nullptr;
  int* d_any_non_finite = nullptr;
  d_min_q_balance = static_cast<double*>(core::device_scratch_acquire(
      "cjq:corner_balance_min_q_balance", sizeof(double)));
  d_first_failing_key = static_cast<int*>(core::device_scratch_acquire(
      "cjq:corner_balance_first_failing_key", sizeof(int)));
  d_any_non_positive = static_cast<int*>(core::device_scratch_acquire(
      "cjq:corner_balance_any_non_positive", sizeof(int)));
  d_any_non_finite = static_cast<int*>(core::device_scratch_acquire(
      "cjq:corner_balance_any_non_finite", sizeof(int)));

  const double one = 1.0;
  const int no_key = kNoFailingCornerKey;
  const int zero = 0;
  cuda_check(cudaMemcpy(d_min_q_balance, &one, sizeof(double), cudaMemcpyHostToDevice),
             "corner balance: init min_q_balance failed");
  cuda_check(cudaMemcpy(d_first_failing_key, &no_key, sizeof(int),
                        cudaMemcpyHostToDevice),
             "corner balance: init first_failing_key failed");
  cuda_check(cudaMemcpy(d_any_non_positive, &zero, sizeof(int), cudaMemcpyHostToDevice),
             "corner balance: init any_non_positive failed");
  cuda_check(cudaMemcpy(d_any_non_finite, &zero, sizeof(int), cudaMemcpyHostToDevice),
             "corner balance: init any_non_finite failed");

  std::int8_t* d_active = nullptr;
  if (!state.hydro_active.empty()) {
    d_active = static_cast<std::int8_t*>(core::device_scratch_acquire(
        "cjq:corner_balance_hydro_active",
        state.hydro_active.size() * sizeof(std::int8_t)));
    cuda_check(cudaMemcpy(d_active, state.hydro_active.data(),
                          state.hydro_active.size() * sizeof(std::int8_t),
                          cudaMemcpyHostToDevice),
               "corner balance: cudaMemcpy hydro_active failed");
  }

  const int blocks = (n_cells + 255) / 256;
  corner_balance_kernel<<<blocks, 256>>>(
      d_min_q_balance, d_first_failing_key, d_any_non_positive, d_any_non_finite,
      state.x_r.data(), state.x_z.data(), d_active, nr, nz, balance_threshold);
  cuda_check(cudaGetLastError(), "corner balance kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "corner balance kernel execution failed");

  double local_q = 1.0;
  int first_key = no_key;
  int local_non_positive = 0;
  int local_non_finite = 0;
  cuda_check(cudaMemcpy(&local_q, d_min_q_balance, sizeof(double),
                        cudaMemcpyDeviceToHost),
             "corner balance: copy min_q_balance failed");
  cuda_check(cudaMemcpy(&first_key, d_first_failing_key, sizeof(int),
                        cudaMemcpyDeviceToHost),
             "corner balance: copy first_failing_key failed");
  cuda_check(cudaMemcpy(&local_non_positive, d_any_non_positive, sizeof(int),
                        cudaMemcpyDeviceToHost),
             "corner balance: copy any_non_positive failed");
  cuda_check(cudaMemcpy(&local_non_finite, d_any_non_finite, sizeof(int),
                        cudaMemcpyDeviceToHost),
             "corner balance: copy any_non_finite failed");

  double global_q = local_q;
  double global_key = static_cast<double>(first_key);
  double global_non_positive = static_cast<double>(local_non_positive);
  double global_non_finite = static_cast<double>(local_non_finite);
  if (reduction != nullptr) {
    global_q = reduction->allreduce_min(global_q);
    global_key = reduction->allreduce_min(global_key);
    global_non_positive = reduction->allreduce_max(global_non_positive);
    global_non_finite = reduction->allreduce_max(global_non_finite);
  }

  if (!(global_q >= 0.0) || !std::isfinite(global_q)) {
    global_q = 0.0;
  }
  const int failing_key = static_cast<int>(global_key);
  result.min_q_balance = std::min(global_q, 1.0);
  result.first_failing_cell = failing_key == no_key ? -1 : failing_key / 4;
  result.first_failing_corner = failing_key == no_key ? -1 : failing_key % 4;
  result.any_corner_non_positive = global_non_positive > 0.5;
  result.any_corner_non_finite = global_non_finite > 0.5;
  result.admissible = failing_key == no_key && result.min_q_balance >= balance_threshold &&
                      !result.any_corner_non_positive &&
                      !result.any_corner_non_finite;
  return result;
}

}  // namespace tenryu::hydro
