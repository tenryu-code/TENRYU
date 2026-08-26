#include "mesh/candidate_mesh_admissibility.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::mesh {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeviceInf = 1.0e300;
constexpr double kButtonOrientationSign = -1.0;
// Belt pentagons have a flat corner at the collinear 2:1 theta-subdivision
// midnode.
constexpr double kFlatCornerRelEps = 1.0e-9;
constexpr std::uint8_t kCandidateCellStructured = 0U;
constexpr std::uint8_t kCandidateCellButton = 1U;
constexpr std::uint8_t kCandidateCellDormant = 2U;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__host__ __device__ inline bool finite_double(const double x) {
  return x == x && x != INFINITY && x != -INFINITY;
}

__host__ __device__ inline double max_double(const double a, const double b) {
  return a > b ? a : b;
}

__host__ __device__ inline double min_double(const double a, const double b) {
  return a < b ? a : b;
}

__host__ __device__ inline double cross2(const double ar, const double az,
                                         const double br, const double bz) {
  return ar * bz - az * br;
}

__host__ __device__ inline void corner_jacobians(
    const double r0, const double z0,
    const double r1, const double z1,
    const double r2, const double z2,
    const double r3, const double z3,
    double* j) {
  j[0] = cross2(r1 - r0, z1 - z0, r3 - r0, z3 - z0);
  j[1] = cross2(r1 - r0, z1 - z0, r2 - r1, z2 - z1);
  j[2] = cross2(r2 - r3, z2 - z3, r2 - r1, z2 - z1);
  j[3] = cross2(r2 - r3, z2 - z3, r3 - r0, z3 - z0);
}

__host__ __device__ inline void triangle_corner_jacobians(
    const double* r,
    const double* z,
    double* j) {
  for (int k = 0; k < 3; ++k) {
    const int kp = (k + 1) % 3;
    const int km = (k + 2) % 3;
    j[k] = cross2(r[kp] - r[k], z[kp] - z[k],
                  r[km] - r[k], z[km] - z[k]);
  }
}

__host__ __device__ inline void polygon_corner_jacobians(
    const double* r,
    const double* z,
    const int nverts,
    double* j) {
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1 == nverts) ? 0 : (k + 1);
    const int km = (k == 0) ? (nverts - 1) : (k - 1);
    j[k] = cross2(r[kp] - r[k], z[kp] - z[k],
                  r[km] - r[k], z[km] - z[k]);
  }
}

__host__ __device__ inline double rz_polygon_volume_exact(
    const double* r,
    const double* z,
    const int nverts) {
  double sum = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1 == nverts) ? 0 : (k + 1);
    sum += (r[k] * z[kp] - r[kp] * z[k]) * (r[k] + r[kp]);
  }
  return kPi / 3.0 * sum;
}

__host__ __device__ inline double abs_double(const double x) {
  return x < 0.0 ? -x : x;
}

__host__ __device__ inline double signed_ratio(const double new_value,
                                               const double old_value) {
  return old_value != 0.0 ? new_value / old_value
                          : -kDeviceInf;
}

__host__ __device__ inline bool same_orientation(const double old_value,
                                                 const double new_value) {
  return (old_value > 0.0 && new_value > 0.0) ||
         (old_value < 0.0 && new_value < 0.0);
}

__host__ __device__ inline double relative_floor_baseline(
    const double previous_value,
    const double reference_value,
    const bool reference_available) {
  if (reference_available && finite_double(reference_value) &&
      same_orientation(previous_value, reference_value)) {
    return abs_double(reference_value);
  }
  return abs_double(previous_value);
}

// 2026-07-26 kernel-review continuous-path helpers (file-local clean-room;
// semantics mirror the path_admissibility.cuh unit-interval minimisation:
// endpoints plus interior stationary points, deterministic closed form).
__host__ __device__ inline double path_quadratic_value(const double q0,
                                                       const double q1,
                                                       const double q2,
                                                       const double lambda) {
  return q0 + lambda * (q1 + lambda * q2);
}

__host__ __device__ inline void path_quadratic_min_unit(const double q0,
                                                        const double q1,
                                                        const double q2,
                                                        double& min_value) {
  min_value = q0;
  const double end_value = path_quadratic_value(q0, q1, q2, 1.0);
  if (end_value < min_value) {
    min_value = end_value;
  }
  if (q2 > 0.0) {
    const double lambda_star = -0.5 * q1 / q2;
    if (lambda_star > 0.0 && lambda_star < 1.0) {
      const double v = path_quadratic_value(q0, q1, q2, lambda_star);
      if (v < min_value) {
        min_value = v;
      }
    }
  }
}

__host__ __device__ inline double path_cubic_value(const double c0,
                                                   const double c1,
                                                   const double c2,
                                                   const double c3,
                                                   const double lambda) {
  return c0 + lambda * (c1 + lambda * (c2 + lambda * c3));
}

__host__ __device__ inline void path_cubic_min_unit(const double c0,
                                                    const double c1,
                                                    const double c2,
                                                    const double c3,
                                                    double& min_value) {
  min_value = c0;
  const double end_value = path_cubic_value(c0, c1, c2, c3, 1.0);
  if (end_value < min_value) {
    min_value = end_value;
  }
  const double a = 3.0 * c3;
  const double b = 2.0 * c2;
  if (a == 0.0) {
    if (b != 0.0) {
      const double lambda_star = -c1 / b;
      if (lambda_star > 0.0 && lambda_star < 1.0) {
        const double v = path_cubic_value(c0, c1, c2, c3, lambda_star);
        if (v < min_value) {
          min_value = v;
        }
      }
    }
    return;
  }
  const double disc = b * b - 4.0 * a * c1;
  if (disc < 0.0) {
    return;
  }
  const double sqrt_disc = ::sqrt(disc);
  const double root_a = (-b - sqrt_disc) / (2.0 * a);
  const double root_b = (-b + sqrt_disc) / (2.0 * a);
  if (root_a > 0.0 && root_a < 1.0) {
    const double v = path_cubic_value(c0, c1, c2, c3, root_a);
    if (v < min_value) {
      min_value = v;
    }
  }
  if (root_b > 0.0 && root_b < 1.0) {
    const double v = path_cubic_value(c0, c1, c2, c3, root_b);
    if (v < min_value) {
      min_value = v;
    }
  }
}

// Quadratic coefficients of det(E1(lambda), E2(lambda)) when both edge
// vectors are linear in lambda: E(lambda) = E_old + lambda * (E_new - E_old).
__host__ __device__ inline void path_corner_j_coefficients(
    const double e1r_old, const double e1z_old,
    const double e1r_new, const double e1z_new,
    const double e2r_old, const double e2z_old,
    const double e2r_new, const double e2z_new,
    double& q0, double& q1, double& q2) {
  const double d1r = e1r_new - e1r_old;
  const double d1z = e1z_new - e1z_old;
  const double d2r = e2r_new - e2r_old;
  const double d2z = e2z_new - e2z_old;
  q0 = cross2(e1r_old, e1z_old, e2r_old, e2z_old);
  q1 = cross2(e1r_old, e1z_old, d2r, d2z) +
       cross2(d1r, d1z, e2r_old, e2z_old);
  q2 = cross2(d1r, d1z, d2r, d2z);
}

// Cubic coefficients of the exact revolved RZ polygon volume
// (pi/3) sum_k (r_k + r_{k+1})(r_k z_{k+1} - r_{k+1} z_k) along the linear
// node path, 3 <= nverts <= 8.
__host__ __device__ inline void path_rz_volume_coefficients(
    const double* r_old, const double* z_old,
    const double* r_new, const double* z_new,
    const int nverts,
    double& b0, double& b1, double& b2, double& b3) {
  b0 = 0.0;
  b1 = 0.0;
  b2 = 0.0;
  b3 = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1 == nverts) ? 0 : (k + 1);
    const double ra = r_old[k];
    const double za = z_old[k];
    const double rb = r_old[kp];
    const double zb = z_old[kp];
    const double dra = r_new[k] - ra;
    const double dza = z_new[k] - za;
    const double drb = r_new[kp] - rb;
    const double dzb = z_new[kp] - zb;
    const double s0 = ra + rb;
    const double s1 = dra + drb;
    const double c0 = ra * zb - rb * za;
    const double c1 = ra * dzb + dra * zb - rb * dza - drb * za;
    const double c2 = dra * dzb - drb * dza;
    b0 += s0 * c0;
    b1 += s0 * c1 + s1 * c0;
    b2 += s0 * c2 + s1 * c1;
    b3 += s1 * c2;
  }
  b0 *= kPi / 3.0;
  b1 *= kPi / 3.0;
  b2 *= kPi / 3.0;
  b3 *= kPi / 3.0;
}

// Shared corner/volume continuous-path check body for both kernels.
// flip: +1, -1 (polar canonical), or the per-cell orientation sign.
// old_j / v_old / volume_floor are the already-canonicalized values used by
// the endpoint gate. Returns kind == None if the whole path stays admissible.
__host__ __device__ inline void continuous_path_check(
    const double* r_old, const double* z_old,
    const double* r_new, const double* z_new,
    const int nverts,
    const double flip,
    const double* old_j,
    const double* reference_j,
    const bool reference_available,
    const double v_old,
    const double volume_floor,
    const CandidateMeshAdmissibilityFloors& floors,
    int& bad_corner,
    MeshGeometryFailureKind& kind) {
  kind = MeshGeometryFailureKind::None;
  bad_corner = -1;
  double j_scale = 0.0;
  if (nverts > 4) {
    for (int k = 0; k < nverts; ++k) {
      const double scale_j =
          reference_available ? reference_j[k] : old_j[k];
      j_scale = max_double(j_scale, abs_double(scale_j));
    }
  }
  for (int k = 0; k < nverts; ++k) {
    if (nverts > 4) {
      const double scale_j =
          reference_available ? reference_j[k] : old_j[k];
      if (abs_double(scale_j) <= kFlatCornerRelEps * j_scale) {
        continue;
      }
    }
    int ia = 0;
    int ib = 0;
    int ic = 0;
    int id = 0;
    if (nverts == 3) {
      ia = k;
      ib = (k + 1) % 3;
      ic = k;
      id = (k + 2) % 3;
    } else if (nverts == 4) {
      if (k == 0) {
        ia = 0; ib = 1; ic = 0; id = 3;
      } else if (k == 1) {
        ia = 0; ib = 1; ic = 1; id = 2;
      } else if (k == 2) {
        ia = 3; ib = 2; ic = 1; id = 2;
      } else {
        ia = 3; ib = 2; ic = 0; id = 3;
      }
    } else {
      ia = k;
      ib = (k + 1 == nverts) ? 0 : (k + 1);
      ic = k;
      id = (k == 0) ? (nverts - 1) : (k - 1);
    }
    double q0 = 0.0;
    double q1 = 0.0;
    double q2 = 0.0;
    path_corner_j_coefficients(
        r_old[ib] - r_old[ia], z_old[ib] - z_old[ia],
        r_new[ib] - r_new[ia], z_new[ib] - z_new[ia],
        r_old[id] - r_old[ic], z_old[id] - z_old[ic],
        r_new[id] - r_new[ic], z_new[id] - z_new[ic],
        q0, q1, q2);
    const double m = flip * ((old_j[k] > 0.0) ? 1.0 : -1.0);
    double min_j = 0.0;
    path_quadratic_min_unit(m * q0, m * q1, m * q2, min_j);
    const double floor_baseline =
        relative_floor_baseline(old_j[k],
                                reference_available ? reference_j[k] : 0.0,
                                reference_available);
    const double floor_k = max_double(
        floors.corner_j_abs, floors.corner_j_rel * floor_baseline);
    if (!(min_j > floor_k)) {
      bad_corner = k;
      kind = MeshGeometryFailureKind::NonPositiveCellArea;
      return;
    }
  }
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  path_rz_volume_coefficients(r_old, z_old, r_new, z_new, nverts,
                              b0, b1, b2, b3);
  const double mv = flip * ((v_old > 0.0) ? 1.0 : -1.0);
  double min_v = 0.0;
  path_cubic_min_unit(mv * b0, mv * b1, mv * b2, mv * b3, min_v);
  if (!(min_v > volume_floor)) {
    kind = MeshGeometryFailureKind::NonPositiveCellVolume;
  }
}

__device__ CandidateMeshQuality make_neutral_quality() {
  CandidateMeshQuality q{};
  q.min_rz_volume_rel = kDeviceInf;
  q.min_corner_j_rel = kDeviceInf;
  q.min_gauss_j_rel = kDeviceInf;
  q.first_bad_cell = -1;
  q.first_bad_stable_cell = -1;
  q.first_bad_corner = -1;
  q.kind = MeshGeometryFailureKind::None;
  return q;
}

__device__ CandidateMeshQuality make_default_quality() {
  CandidateMeshQuality q{};
  q.first_bad_cell = -1;
  q.first_bad_stable_cell = -1;
  q.first_bad_corner = -1;
  q.kind = MeshGeometryFailureKind::None;
  return q;
}

__global__ void fill_neutral_candidate_mesh_quality_kernel(
    CandidateMeshQuality* out,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  auto* bytes = reinterpret_cast<unsigned char*>(&out[c]);
  for (int i = 0; i < static_cast<int>(sizeof(CandidateMeshQuality)); ++i) {
    bytes[i] = 0U;
  }

  out[c].min_rz_volume_rel = kDeviceInf;
  out[c].min_corner_j_rel = kDeviceInf;
  out[c].min_gauss_j_rel = kDeviceInf;
  out[c].first_bad_cell = -1;
  out[c].first_bad_stable_cell = -1;
  out[c].first_bad_corner = -1;
  out[c].kind = MeshGeometryFailureKind::None;
}

void fill_neutral_candidate_mesh_quality(CandidateMeshQuality* d_quality,
                                         const int n_cells,
                                         const char* message) {
  constexpr int threads = 128;
  const int blocks = (n_cells + threads - 1) / threads;
  fill_neutral_candidate_mesh_quality_kernel<<<blocks, threads>>>(
      d_quality, n_cells);
  cuda_check(cudaGetLastError(), message);
}

__device__ inline void button_vertex(
    const double* node_r_old,
    const double* node_z_old,
    const double* delta_r,
    const double* delta_z,
    const double sigma,
    const int nz,
    const int outer_ring,
    const int j,
    double* r_old,
    double* z_old,
    double* r_new,
    double* z_new) {
  const int n = outer_ring * (nz + 1) + j;
  *r_old = node_r_old[n];
  *z_old = node_z_old[n];
  *r_new = *r_old + sigma * delta_r[n];
  *z_new = *z_old + sigma * delta_z[n];
}

__device__ inline void button_polygon_moments(
    const double* node_r_old,
    const double* node_z_old,
    const double* node_r_reference,
    const double* node_z_reference,
    const double* delta_r,
    const double* delta_z,
    const double sigma,
    const int nz,
    const int outer_ring,
    const bool use_new,
    const bool use_reference,
    double* volume,
    double* area2,
    double* centroid_r,
    double* centroid_z,
    bool* finite_nodes) {
  const int nverts = nz + 1;
  double cross_sum = 0.0;
  double volume_sum = 0.0;
  double cr_sum = 0.0;
  double cz_sum = 0.0;
  double r_avg = 0.0;
  double z_avg = 0.0;
  bool finite = true;
  for (int k = 0; k < nverts; ++k) {
    const int kp1 = (k + 1) % nverts;
    double r0_old = 0.0;
    double z0_old = 0.0;
    double r0_new = 0.0;
    double z0_new = 0.0;
    double r1_old = 0.0;
    double z1_old = 0.0;
    double r1_new = 0.0;
    double z1_new = 0.0;
    if (use_reference) {
      const int n0 = outer_ring * (nz + 1) + k;
      const int n1 = outer_ring * (nz + 1) + kp1;
      r0_old = node_r_reference[n0];
      z0_old = node_z_reference[n0];
      r1_old = node_r_reference[n1];
      z1_old = node_z_reference[n1];
    } else {
      button_vertex(node_r_old, node_z_old, delta_r, delta_z, sigma, nz,
                    outer_ring, k, &r0_old, &z0_old, &r0_new, &z0_new);
      button_vertex(node_r_old, node_z_old, delta_r, delta_z, sigma, nz,
                    outer_ring, kp1, &r1_old, &z1_old, &r1_new, &z1_new);
    }
    const double r0 = use_reference ? r0_old : (use_new ? r0_new : r0_old);
    const double z0 = use_reference ? z0_old : (use_new ? z0_new : z0_old);
    const double r1 = use_reference ? r1_old : (use_new ? r1_new : r1_old);
    const double z1 = use_reference ? z1_old : (use_new ? z1_new : z1_old);
    finite = finite && finite_double(r0) && finite_double(z0) &&
             finite_double(r1) && finite_double(z1);
    const double cross = r0 * z1 - r1 * z0;
    cross_sum += cross;
    volume_sum += (r0 + r1) * cross;
    cr_sum += (r0 + r1) * cross;
    cz_sum += (z0 + z1) * cross;
    r_avg += r0;
    z_avg += z0;
  }

  *volume = kButtonOrientationSign * (kPi / 3.0) * volume_sum;
  *area2 = kButtonOrientationSign * cross_sum;
  if (cross_sum != 0.0) {
    const double inv = 1.0 / (3.0 * cross_sum);
    *centroid_r = cr_sum * inv;
    *centroid_z = cz_sum * inv;
  } else {
    *centroid_r = r_avg / static_cast<double>(nverts);
    *centroid_z = z_avg / static_cast<double>(nverts);
  }
  *finite_nodes = finite && finite_double(*volume) && finite_double(*area2) &&
                  finite_double(*centroid_r) && finite_double(*centroid_z);
}

__device__ CandidateMeshQuality evaluate_button_candidate_quality(
    const double* node_r_old,
    const double* node_z_old,
    const double* node_r_reference,
    const double* node_z_reference,
    const double* delta_r,
    const double* delta_z,
    const double sigma,
    const int nz,
    const int outer_ring,
    const CandidateMeshAdmissibilityFloors floors) {
  CandidateMeshQuality q = make_default_quality();
  q.first_bad_cell = 0;
  const int nverts = nz + 1;
  const bool reference_available =
      (floors.volume_rel > 0.0 || floors.corner_j_rel > 0.0 ||
       floors.gauss_j_rel > 0.0) &&
      node_r_reference != nullptr && node_z_reference != nullptr;

  double v_old = 0.0;
  double area2_old = 0.0;
  double centroid_r_old = 0.0;
  double centroid_z_old = 0.0;
  bool finite_old = false;
  button_polygon_moments(node_r_old, node_z_old, node_r_reference,
                         node_z_reference, delta_r, delta_z, sigma, nz,
                         outer_ring, false, false, &v_old, &area2_old,
                         &centroid_r_old, &centroid_z_old, &finite_old);
  double v_new = 0.0;
  double area2_new = 0.0;
  double centroid_r_new = 0.0;
  double centroid_z_new = 0.0;
  bool finite_new = false;
  button_polygon_moments(node_r_old, node_z_old, node_r_reference,
                         node_z_reference, delta_r, delta_z, sigma, nz,
                         outer_ring, true, false, &v_new, &area2_new,
                         &centroid_r_new, &centroid_z_new, &finite_new);
  double v_reference = 0.0;
  double area2_reference = 0.0;
  double centroid_r_reference = 0.0;
  double centroid_z_reference = 0.0;
  bool finite_reference = false;
  if (reference_available) {
    button_polygon_moments(node_r_old, node_z_old, node_r_reference,
                           node_z_reference, delta_r, delta_z, sigma, nz,
                           outer_ring, false, true, &v_reference,
                           &area2_reference, &centroid_r_reference,
                           &centroid_z_reference, &finite_reference);
  }
  const bool usable_reference =
      reference_available && finite_reference && v_reference > 0.0 &&
      area2_reference > 0.0;
  if (!finite_old || !finite_new) {
    q.min_rz_volume_rel = -kDeviceInf;
    q.min_corner_j_rel = -kDeviceInf;
    q.min_gauss_j_rel = -kDeviceInf;
    q.kind = MeshGeometryFailureKind::NonFiniteNodeCoordinate;
    return q;
  }

  q.min_rz_volume_rel = signed_ratio(v_new, v_old);
  const double volume_floor = max_double(
      floors.volume_abs,
      floors.volume_rel *
          relative_floor_baseline(v_old, v_reference,
                                  usable_reference));
  if (!(v_old > 0.0) || !(v_new > volume_floor)) {
    q.kind = MeshGeometryFailureKind::NonPositiveCellVolume;
    return q;
  }
  if (!(area2_old > 0.0) || !(area2_new > 0.0)) {
    q.kind = MeshGeometryFailureKind::NonPositiveCellArea;
    return q;
  }

  q.min_corner_j_rel = kDeviceInf;
  double old_gauss = 0.0;
  double new_gauss = 0.0;
  double reference_gauss = 0.0;
  bool corner_ok = true;
  int bad_corner = -1;
  for (int k = 0; k < nverts; ++k) {
    const int kp1 = (k + 1) % nverts;
    double r0_old = 0.0;
    double z0_old = 0.0;
    double r0_new = 0.0;
    double z0_new = 0.0;
    double r1_old = 0.0;
    double z1_old = 0.0;
    double r1_new = 0.0;
    double z1_new = 0.0;
    button_vertex(node_r_old, node_z_old, delta_r, delta_z, sigma, nz,
                  outer_ring, k, &r0_old, &z0_old, &r0_new, &z0_new);
    button_vertex(node_r_old, node_z_old, delta_r, delta_z, sigma, nz,
                  outer_ring, kp1, &r1_old, &z1_old, &r1_new, &z1_new);
    const double old_area2 =
        kButtonOrientationSign *
        cross2(r0_old - centroid_r_old, z0_old - centroid_z_old,
               r1_old - centroid_r_old, z1_old - centroid_z_old);
    const double new_area2 =
        kButtonOrientationSign *
        cross2(r0_new - centroid_r_new, z0_new - centroid_z_new,
               r1_new - centroid_r_new, z1_new - centroid_z_new);
    double reference_area2 = 0.0;
    if (reference_available) {
      const int n0 = outer_ring * (nz + 1) + k;
      const int n1 = outer_ring * (nz + 1) + kp1;
      reference_area2 =
          kButtonOrientationSign *
          cross2(node_r_reference[n0] - centroid_r_reference,
                 node_z_reference[n0] - centroid_z_reference,
                 node_r_reference[n1] - centroid_r_reference,
                 node_z_reference[n1] - centroid_z_reference);
    }
    old_gauss += old_area2;
    new_gauss += new_area2;
    reference_gauss += reference_area2;
    const double ratio = signed_ratio(new_area2, old_area2);
    q.min_corner_j_rel = min_double(q.min_corner_j_rel, ratio);
    const double floor =
        max_double(floors.corner_j_abs,
                   floors.corner_j_rel *
                       relative_floor_baseline(
                           old_area2, reference_area2,
                           usable_reference));
    if (!finite_double(old_area2) || !finite_double(new_area2) ||
        !(old_area2 > 0.0) || !(new_area2 > floor)) {
      corner_ok = false;
      if (bad_corner < 0) {
        bad_corner = k;
      }
    }
  }
  old_gauss /= static_cast<double>(nverts);
  new_gauss /= static_cast<double>(nverts);
  reference_gauss /= static_cast<double>(nverts);
  q.min_gauss_j_rel = signed_ratio(new_gauss, old_gauss);

  const double gauss_floor = max_double(
      floors.gauss_j_abs,
      floors.gauss_j_rel *
          relative_floor_baseline(old_gauss, reference_gauss,
                                  usable_reference));
  if (!corner_ok) {
    q.first_bad_corner = bad_corner;
    q.kind = MeshGeometryFailureKind::NonPositiveCellArea;
  } else if (!finite_double(old_gauss) || !finite_double(new_gauss)) {
    q.kind = MeshGeometryFailureKind::NonFiniteCellArea;
  } else if (!(old_gauss > 0.0) || !(new_gauss > gauss_floor)) {
    q.kind = MeshGeometryFailureKind::NonPositiveCellArea;
  } else {
    q.first_bad_cell = -1;
  }
  return q;
}

__global__ void evaluate_candidate_mesh_quality_kernel(
    const double* node_r_old,
    const double* node_z_old,
    const double* node_r_reference,
    const double* node_z_reference,
    const double* delta_r,
    const double* delta_z,
    const double sigma,
    const int nr,
    const int nz,
    const CandidateMeshAdmissibilityFloors floors,
    const std::uint8_t* cell_nverts,
    const std::uint8_t* cell_kind,
    const int button_outer_node_ring,
    CandidateMeshQuality* out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  CandidateMeshQuality q = make_default_quality();
  const std::uint8_t kind =
      (cell_kind != nullptr) ? cell_kind[c] : kCandidateCellStructured;
  if (kind == kCandidateCellDormant) {
    out[c] = make_neutral_quality();
    return;
  }
  if (kind == kCandidateCellButton) {
    out[c] = evaluate_button_candidate_quality(
        node_r_old, node_z_old, node_r_reference, node_z_reference,
        delta_r, delta_z, sigma, nz,
        button_outer_node_ring, floors);
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n0 = i * stride + j;
  const int n1 = (i + 1) * stride + j;
  const int n2 = (i + 1) * stride + (j + 1);
  const int n3 = i * stride + (j + 1);

  const double r_old[4] = {
      node_r_old[n0],
      node_r_old[n1],
      node_r_old[n2],
      node_r_old[n3],
  };
  const double z_old[4] = {
      node_z_old[n0],
      node_z_old[n1],
      node_z_old[n2],
      node_z_old[n3],
  };
  const bool reference_available =
      (floors.volume_rel > 0.0 || floors.corner_j_rel > 0.0 ||
       floors.gauss_j_rel > 0.0) &&
      node_r_reference != nullptr && node_z_reference != nullptr;
  const double r_reference[4] = {
      reference_available ? node_r_reference[n0] : 0.0,
      reference_available ? node_r_reference[n1] : 0.0,
      reference_available ? node_r_reference[n2] : 0.0,
      reference_available ? node_r_reference[n3] : 0.0,
  };
  const double z_reference[4] = {
      reference_available ? node_z_reference[n0] : 0.0,
      reference_available ? node_z_reference[n1] : 0.0,
      reference_available ? node_z_reference[n2] : 0.0,
      reference_available ? node_z_reference[n3] : 0.0,
  };
  const double r_new[4] = {
      r_old[0] + sigma * delta_r[n0],
      r_old[1] + sigma * delta_r[n1],
      r_old[2] + sigma * delta_r[n2],
      r_old[3] + sigma * delta_r[n3],
  };
  const double z_new[4] = {
      z_old[0] + sigma * delta_z[n0],
      z_old[1] + sigma * delta_z[n1],
      z_old[2] + sigma * delta_z[n2],
      z_old[3] + sigma * delta_z[n3],
  };
  const bool polar_canonical_mode = (cell_nverts != nullptr);
  const int nverts = mesh_topo_cell_active_nverts(cell_nverts, c);

  bool finite_nodes = true;
  for (int k = 0; k < nverts; ++k) {
    finite_nodes = finite_nodes && finite_double(r_old[k]) &&
                   finite_double(z_old[k]) && finite_double(r_new[k]) &&
                   finite_double(z_new[k]);
  }
  if (!finite_nodes) {
    q.min_rz_volume_rel = -kDeviceInf;
    q.min_corner_j_rel = -kDeviceInf;
    q.min_gauss_j_rel = -kDeviceInf;
    q.first_bad_cell = c;
    q.kind = MeshGeometryFailureKind::NonFiniteNodeCoordinate;
    out[c] = q;
    return;
  }

  double v_old = 0.0;
  double v_new = 0.0;
  double v_reference = 0.0;
  if (nverts == 3) {
    v_old = rz_triangle_volume_exact(r_old[0], z_old[0],
                                     r_old[1], z_old[1],
                                     r_old[2], z_old[2]);
    v_new = rz_triangle_volume_exact(r_new[0], z_new[0],
                                     r_new[1], z_new[1],
                                     r_new[2], z_new[2]);
    if (reference_available) {
      v_reference = rz_triangle_volume_exact(
          r_reference[0], z_reference[0], r_reference[1], z_reference[1],
          r_reference[2], z_reference[2]);
    }
  } else {
    v_old = rz_quad_volume_exact(r_old[0], z_old[0],
                                 r_old[1], z_old[1],
                                 r_old[2], z_old[2],
                                 r_old[3], z_old[3]);
    v_new = rz_quad_volume_exact(r_new[0], z_new[0],
                                 r_new[1], z_new[1],
                                 r_new[2], z_new[2],
                                 r_new[3], z_new[3]);
    if (reference_available) {
      v_reference = rz_quad_volume_exact(
          r_reference[0], z_reference[0], r_reference[1], z_reference[1],
          r_reference[2], z_reference[2], r_reference[3], z_reference[3]);
    }
  }
  if (polar_canonical_mode) {
    v_old = -v_old;
    v_new = -v_new;
    v_reference = -v_reference;
  }
  q.min_rz_volume_rel = signed_ratio(v_new, v_old);

  double old_j[4]{};
  double new_j[4]{};
  double reference_j[4]{};
  if (nverts == 3) {
    triangle_corner_jacobians(r_old, z_old, old_j);
    triangle_corner_jacobians(r_new, z_new, new_j);
    if (reference_available) {
      triangle_corner_jacobians(r_reference, z_reference, reference_j);
    }
  } else {
    corner_jacobians(r_old[0], z_old[0], r_old[1], z_old[1],
                     r_old[2], z_old[2], r_old[3], z_old[3], old_j);
    corner_jacobians(r_new[0], z_new[0], r_new[1], z_new[1],
                     r_new[2], z_new[2], r_new[3], z_new[3], new_j);
    if (reference_available) {
      corner_jacobians(
          r_reference[0], z_reference[0], r_reference[1], z_reference[1],
          r_reference[2], z_reference[2], r_reference[3], z_reference[3],
          reference_j);
    }
  }
  if (polar_canonical_mode) {
    for (int k = 0; k < nverts; ++k) {
      old_j[k] = -old_j[k];
      new_j[k] = -new_j[k];
      reference_j[k] = -reference_j[k];
    }
  }

  q.min_corner_j_rel = kDeviceInf;
  bool corner_ok = true;
  int bad_corner = -1;
  for (int k = 0; k < nverts; ++k) {
    const double ratio = signed_ratio(new_j[k], old_j[k]);
    q.min_corner_j_rel = min_double(q.min_corner_j_rel, ratio);
    const double floor = max_double(floors.corner_j_abs,
                                    floors.corner_j_rel *
                                        relative_floor_baseline(
                                            old_j[k], reference_j[k],
                                            reference_available));
    const bool above_floor =
        polar_canonical_mode ? (new_j[k] > floor)
                             : (abs_double(new_j[k]) > floor);
    if (!finite_double(old_j[k]) || !finite_double(new_j[k]) ||
        old_j[k] == 0.0 || !same_orientation(old_j[k], new_j[k]) ||
        !above_floor) {
      corner_ok = false;
      if (bad_corner < 0) {
        bad_corner = k;
      }
    }
  }

  // "gauss" = arithmetic mean of the corner Jacobians == center Jacobian of
  // the bilinear map; NOT a Gauss-quadrature-point value (2026-07-26 kernel review).
  double old_gauss = 0.0;
  double new_gauss = 0.0;
  double reference_gauss = 0.0;
  for (int k = 0; k < nverts; ++k) {
    old_gauss += old_j[k];
    new_gauss += new_j[k];
    reference_gauss += reference_j[k];
  }
  old_gauss /= static_cast<double>(nverts);
  new_gauss /= static_cast<double>(nverts);
  reference_gauss /= static_cast<double>(nverts);
  q.min_gauss_j_rel = signed_ratio(new_gauss, old_gauss);

  const double volume_floor =
      max_double(floors.volume_abs,
                 floors.volume_rel *
                     relative_floor_baseline(v_old, v_reference,
                                             reference_available));
  const double gauss_floor =
      max_double(floors.gauss_j_abs,
                 floors.gauss_j_rel *
                     relative_floor_baseline(old_gauss, reference_gauss,
                                             reference_available));
  const bool gauss_above_floor =
      polar_canonical_mode ? (new_gauss > gauss_floor)
                           : (abs_double(new_gauss) > gauss_floor);

  if (!finite_double(v_old) || !finite_double(v_new)) {
    q.first_bad_cell = c;
    q.kind = MeshGeometryFailureKind::NonFiniteCellVolume;
  } else if (!(v_old > 0.0) || !(v_new > volume_floor)) {
    q.first_bad_cell = c;
    q.kind = MeshGeometryFailureKind::NonPositiveCellVolume;
  } else if (!corner_ok) {
    q.first_bad_cell = c;
    q.first_bad_corner = bad_corner;
    q.kind = MeshGeometryFailureKind::NonPositiveCellArea;
  } else if (!finite_double(old_gauss) || !finite_double(new_gauss)) {
    q.first_bad_cell = c;
    q.kind = MeshGeometryFailureKind::NonFiniteCellArea;
  } else if (old_gauss == 0.0 || !same_orientation(old_gauss, new_gauss) ||
             !gauss_above_floor) {
    q.first_bad_cell = c;
    q.kind = MeshGeometryFailureKind::NonPositiveCellArea;
  }

  if (q.kind == MeshGeometryFailureKind::None &&
      floors.require_nonnegative_r) {
    for (int k = 0; k < nverts; ++k) {
      if (!(r_new[k] >= 0.0)) {
        q.first_bad_cell = c;
        q.first_bad_corner = k;
        q.kind = MeshGeometryFailureKind::NegativeRadialCoordinate;
        break;
      }
    }
  }
  if (q.kind == MeshGeometryFailureKind::None && floors.continuous_path &&
      sigma != 0.0) {
    const double flip = polar_canonical_mode ? -1.0 : 1.0;
    int path_bad_corner = -1;
    MeshGeometryFailureKind path_kind = MeshGeometryFailureKind::None;
    continuous_path_check(r_old, z_old, r_new, z_new, nverts, flip, old_j,
                          reference_j, reference_available, v_old,
                          volume_floor, floors, path_bad_corner, path_kind);
    if (path_kind != MeshGeometryFailureKind::None) {
      q.first_bad_cell = c;
      q.first_bad_corner = path_bad_corner;
      q.kind = path_kind;
    }
  }

  out[c] = q;
}

__global__ void evaluate_candidate_mesh_quality_csr_kernel(
    const double* node_r_old,
    const double* node_z_old,
    const double* node_r_reference,
    const double* node_z_reference,
    const double* delta_r,
    const double* delta_z,
    const double sigma,
    const int n_cells_total,
    const int topology_stride,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const int* cell_id_stable,
    const int* cell_orientation_sign,
    const CandidateMeshAdmissibilityFloors floors,
    const std::uint8_t* cell_nverts,
    const std::uint8_t* inactive_cell_mask,
    CandidateMeshQuality* out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells_total) {
    return;
  }
  if (inactive_cell_mask != nullptr && inactive_cell_mask[c] != 0U) {
    out[c] = make_neutral_quality();
    return;
  }

  const int stable_cell = cell_id_stable[c];
  CandidateMeshQuality q{};
  q.first_bad_cell = -1;
  q.first_bad_stable_cell = -1;
  q.first_bad_corner = -1;
  q.kind = MeshGeometryFailureKind::None;

  const int off = cell_node_csr_offsets[c];
  const int next = cell_node_csr_offsets[c + 1];
  const int slot_width = next - off;
  if (off < 0 || next < off ||
      slot_width != topology_stride ||
      (slot_width != kMeshTopoCellStorageSlots &&
       slot_width != kMeshTopoCellStorageSlotsMax) ||
      (floors.n_csr_indices >= 0 && next > floors.n_csr_indices)) {
    q.min_rz_volume_rel = -kDeviceInf;
    q.min_corner_j_rel = -kDeviceInf;
    q.min_gauss_j_rel = -kDeviceInf;
    q.first_bad_cell = c;
    q.first_bad_stable_cell = stable_cell;
    q.kind = MeshGeometryFailureKind::TopologyInvariantViolation;
    out[c] = q;
    return;
  }

  int nodes[kMeshTopoCellStorageSlotsMax] =
      {-1, -1, -1, -1, -1, -1, -1, -1};
  double r_old[kMeshTopoCellStorageSlotsMax] = {0.0};
  double z_old[kMeshTopoCellStorageSlotsMax] = {0.0};
  double r_new[kMeshTopoCellStorageSlotsMax] = {0.0};
  double z_new[kMeshTopoCellStorageSlotsMax] = {0.0};
  double r_reference[kMeshTopoCellStorageSlotsMax] = {0.0};
  double z_reference[kMeshTopoCellStorageSlotsMax] = {0.0};
  const bool reference_available =
      (floors.volume_rel > 0.0 || floors.corner_j_rel > 0.0 ||
       floors.gauss_j_rel > 0.0) &&
      node_r_reference != nullptr && node_z_reference != nullptr;
  int nverts = 0;
  bool node_ids_ok = true;
  if (cell_nverts != nullptr) {
    nverts = static_cast<int>(cell_nverts[c]);
    node_ids_ok = nverts >= 3 && nverts <= kMeshTopoCellStorageSlotsMax &&
                  nverts <= slot_width;
    for (int k = 0; node_ids_ok && k < nverts; ++k) {
      nodes[k] = cell_node_csr_indices[off + k];
      if (nodes[k] < 0 ||
          (floors.n_nodes >= 0 && nodes[k] >= floors.n_nodes)) {
        node_ids_ok = false;
      }
    }
  } else {
    for (int k = 0; k < slot_width; ++k) {
      const int node = cell_node_csr_indices[off + k];
      if (node < 0) {
        continue;
      }
      nodes[nverts] = node;
      ++nverts;
      if (floors.n_nodes >= 0 && node >= floors.n_nodes) {
        node_ids_ok = false;
      }
    }
    if (nverts < 3 || nverts > kMeshTopoCellStorageSlotsMax ||
        nverts > slot_width) {
      node_ids_ok = false;
    }
  }
  if (!node_ids_ok) {
    q.min_rz_volume_rel = -kDeviceInf;
    q.min_corner_j_rel = -kDeviceInf;
    q.min_gauss_j_rel = -kDeviceInf;
    q.first_bad_cell = c;
    q.first_bad_stable_cell = stable_cell;
    q.kind = MeshGeometryFailureKind::TopologyInvariantViolation;
    out[c] = q;
    return;
  }
  for (int k = 0; k < nverts; ++k) {
    r_old[k] = node_r_old[nodes[k]];
    z_old[k] = node_z_old[nodes[k]];
    r_new[k] = r_old[k] + sigma * delta_r[nodes[k]];
    z_new[k] = z_old[k] + sigma * delta_z[nodes[k]];
    if (reference_available) {
      r_reference[k] = node_r_reference[nodes[k]];
      z_reference[k] = node_z_reference[nodes[k]];
    }
  }

  bool finite_nodes = true;
  for (int k = 0; k < nverts; ++k) {
    finite_nodes = finite_nodes && finite_double(r_old[k]) &&
                   finite_double(z_old[k]) && finite_double(r_new[k]) &&
                   finite_double(z_new[k]);
  }
  if (!finite_nodes) {
    q.min_rz_volume_rel = -kDeviceInf;
    q.min_corner_j_rel = -kDeviceInf;
    q.min_gauss_j_rel = -kDeviceInf;
    q.first_bad_cell = c;
    q.first_bad_stable_cell = stable_cell;
    q.kind = MeshGeometryFailureKind::NonFiniteNodeCoordinate;
    out[c] = q;
    return;
  }

  double v_old = 0.0;
  double v_new = 0.0;
  double v_reference = 0.0;
  if (nverts == 3) {
    v_old = rz_triangle_volume_exact(r_old[0], z_old[0],
                                     r_old[1], z_old[1],
                                     r_old[2], z_old[2]);
    v_new = rz_triangle_volume_exact(r_new[0], z_new[0],
                                     r_new[1], z_new[1],
                                     r_new[2], z_new[2]);
    if (reference_available) {
      v_reference = rz_triangle_volume_exact(
          r_reference[0], z_reference[0], r_reference[1], z_reference[1],
          r_reference[2], z_reference[2]);
    }
  } else if (nverts == 4) {
    v_old = rz_quad_volume_exact(r_old[0], z_old[0],
                                 r_old[1], z_old[1],
                                 r_old[2], z_old[2],
                                 r_old[3], z_old[3]);
    v_new = rz_quad_volume_exact(r_new[0], z_new[0],
                                 r_new[1], z_new[1],
                                 r_new[2], z_new[2],
                                 r_new[3], z_new[3]);
    if (reference_available) {
      v_reference = rz_quad_volume_exact(
          r_reference[0], z_reference[0], r_reference[1], z_reference[1],
          r_reference[2], z_reference[2], r_reference[3], z_reference[3]);
    }
  } else {
    v_old = rz_polygon_volume_exact(r_old, z_old, nverts);
    v_new = rz_polygon_volume_exact(r_new, z_new, nverts);
    if (reference_available) {
      v_reference =
          rz_polygon_volume_exact(r_reference, z_reference, nverts);
    }
  }
  const bool orientation_aware = (cell_orientation_sign != nullptr);
  const double orientation_sign =
      orientation_aware ? static_cast<double>(cell_orientation_sign[c]) : 1.0;
  if (orientation_aware) {
    v_old *= orientation_sign;
    v_new *= orientation_sign;
    v_reference *= orientation_sign;
  }
  q.min_rz_volume_rel = signed_ratio(v_new, v_old);

  double old_j[kMeshTopoCellStorageSlotsMax]{};
  double new_j[kMeshTopoCellStorageSlotsMax]{};
  double reference_j[kMeshTopoCellStorageSlotsMax]{};
  if (nverts == 3) {
    triangle_corner_jacobians(r_old, z_old, old_j);
    triangle_corner_jacobians(r_new, z_new, new_j);
    if (reference_available) {
      triangle_corner_jacobians(r_reference, z_reference, reference_j);
    }
  } else if (nverts == 4) {
    corner_jacobians(r_old[0], z_old[0], r_old[1], z_old[1],
                     r_old[2], z_old[2], r_old[3], z_old[3], old_j);
    corner_jacobians(r_new[0], z_new[0], r_new[1], z_new[1],
                     r_new[2], z_new[2], r_new[3], z_new[3], new_j);
    if (reference_available) {
      corner_jacobians(
          r_reference[0], z_reference[0], r_reference[1], z_reference[1],
          r_reference[2], z_reference[2], r_reference[3], z_reference[3],
          reference_j);
    }
  } else {
    polygon_corner_jacobians(r_old, z_old, nverts, old_j);
    polygon_corner_jacobians(r_new, z_new, nverts, new_j);
    if (reference_available) {
      polygon_corner_jacobians(
          r_reference, z_reference, nverts, reference_j);
    }
  }
  if (orientation_aware) {
    for (int k = 0; k < nverts; ++k) {
      old_j[k] *= orientation_sign;
      new_j[k] *= orientation_sign;
      reference_j[k] *= orientation_sign;
    }
  }

  q.min_corner_j_rel = kDeviceInf;
  bool corner_ok = true;
  int bad_corner = -1;
  double j_scale = 0.0;
  if (nverts > 4) {
    for (int k = 0; k < nverts; ++k) {
      const double scale_j =
          reference_available ? reference_j[k] : old_j[k];
      j_scale = max_double(j_scale, abs_double(scale_j));
    }
  }
  for (int k = 0; k < nverts; ++k) {
    if (nverts > 4) {
      const double scale_j =
          reference_available ? reference_j[k] : old_j[k];
      if (abs_double(scale_j) <= kFlatCornerRelEps * j_scale) {
        if (!finite_double(old_j[k]) || !finite_double(new_j[k])) {
          corner_ok = false;
          if (bad_corner < 0) {
            bad_corner = k;
          }
        }
        continue;
      }
    }
    const double ratio = signed_ratio(new_j[k], old_j[k]);
    q.min_corner_j_rel = min_double(q.min_corner_j_rel, ratio);
    const double floor = max_double(floors.corner_j_abs,
                                    floors.corner_j_rel *
                                        relative_floor_baseline(
                                            old_j[k], reference_j[k],
                                            reference_available));
    const bool above_floor =
        orientation_aware ? (new_j[k] > floor)
                          : (abs_double(new_j[k]) > floor);
    if (!finite_double(old_j[k]) || !finite_double(new_j[k]) ||
        old_j[k] == 0.0 || !same_orientation(old_j[k], new_j[k]) ||
        !above_floor) {
      corner_ok = false;
      if (bad_corner < 0) {
        bad_corner = k;
      }
    }
  }

  // "gauss" = arithmetic mean of the corner Jacobians; for quads this equals
  // the center Jacobian of the bilinear map. It is NOT a
  // Gauss-quadrature-point value (2026-07-26 kernel review).
  double old_gauss = 0.0;
  double new_gauss = 0.0;
  double reference_gauss = 0.0;
  for (int k = 0; k < nverts; ++k) {
    old_gauss += old_j[k];
    new_gauss += new_j[k];
    reference_gauss += reference_j[k];
  }
  if (nverts == 3) {
    old_gauss /= 3.0;
    new_gauss /= 3.0;
    reference_gauss /= 3.0;
  } else if (nverts == 4) {
    old_gauss *= 0.25;
    new_gauss *= 0.25;
    reference_gauss *= 0.25;
  } else {
    old_gauss /= static_cast<double>(nverts);
    new_gauss /= static_cast<double>(nverts);
    reference_gauss /= static_cast<double>(nverts);
  }
  q.min_gauss_j_rel = signed_ratio(new_gauss, old_gauss);

  const double volume_floor =
      max_double(floors.volume_abs,
                 floors.volume_rel *
                     relative_floor_baseline(v_old, v_reference,
                                             reference_available));
  const double gauss_floor =
      max_double(floors.gauss_j_abs,
                 floors.gauss_j_rel *
                     relative_floor_baseline(old_gauss, reference_gauss,
                                             reference_available));

  if (!finite_double(v_old) || !finite_double(v_new)) {
    q.first_bad_cell = c;
    q.first_bad_stable_cell = stable_cell;
    q.kind = MeshGeometryFailureKind::NonFiniteCellVolume;
  } else if (orientation_aware
                 ? (!(v_old > 0.0) || !(v_new > volume_floor))
                 : (v_old == 0.0 || !same_orientation(v_old, v_new) ||
                    !(abs_double(v_new) > volume_floor))) {
    q.first_bad_cell = c;
    q.first_bad_stable_cell = stable_cell;
    q.kind = MeshGeometryFailureKind::NonPositiveCellVolume;
  } else if (!corner_ok) {
    q.first_bad_cell = c;
    q.first_bad_stable_cell = stable_cell;
    q.first_bad_corner = bad_corner;
    q.kind = MeshGeometryFailureKind::NonPositiveCellArea;
  } else if (!finite_double(old_gauss) || !finite_double(new_gauss)) {
    q.first_bad_cell = c;
    q.first_bad_stable_cell = stable_cell;
    q.kind = MeshGeometryFailureKind::NonFiniteCellArea;
  } else if (orientation_aware
                 ? (!(old_gauss > 0.0) || !(new_gauss > gauss_floor))
                 : (old_gauss == 0.0 ||
                    !same_orientation(old_gauss, new_gauss) ||
                    !(abs_double(new_gauss) > gauss_floor))) {
    q.first_bad_cell = c;
    q.first_bad_stable_cell = stable_cell;
    q.kind = MeshGeometryFailureKind::NonPositiveCellArea;
  }

  if (q.kind == MeshGeometryFailureKind::None &&
      floors.require_nonnegative_r) {
    for (int k = 0; k < nverts; ++k) {
      if (!(r_new[k] >= 0.0)) {
        q.first_bad_cell = c;
        q.first_bad_stable_cell = stable_cell;
        q.first_bad_corner = k;
        q.kind = MeshGeometryFailureKind::NegativeRadialCoordinate;
        break;
      }
    }
  }
  if (q.kind == MeshGeometryFailureKind::None && floors.continuous_path &&
      sigma != 0.0) {
    const double flip = orientation_aware ? orientation_sign : 1.0;
    int path_bad_corner = -1;
    MeshGeometryFailureKind path_kind = MeshGeometryFailureKind::None;
    continuous_path_check(r_old, z_old, r_new, z_new, nverts, flip, old_j,
                          reference_j, reference_available, v_old,
                          volume_floor, floors, path_bad_corner, path_kind);
    if (path_kind != MeshGeometryFailureKind::None) {
      q.first_bad_cell = c;
      q.first_bad_stable_cell = stable_cell;
      q.first_bad_corner = path_bad_corner;
      q.kind = path_kind;
    }
  }

  out[c] = q;
}

inline double host_rz_polygon_volume(const std::vector<double>& r,
                                     const std::vector<double>& z) {
  const int n = static_cast<int>(r.size());
  if (n < 3 || z.size() != r.size()) {
    return -std::numeric_limits<double>::infinity();
  }
  double sum = 0.0;
  for (int k = 0; k < n; ++k) {
    const int kp = (k + 1) % n;
    const double ra = r[static_cast<std::size_t>(k)];
    const double za = z[static_cast<std::size_t>(k)];
    const double rb = r[static_cast<std::size_t>(kp)];
    const double zb = z[static_cast<std::size_t>(kp)];
    if (!std::isfinite(ra) || !std::isfinite(za) ||
        !std::isfinite(rb) || !std::isfinite(zb)) {
      return -std::numeric_limits<double>::infinity();
    }
    sum += (ra + rb) * (ra * zb - rb * za);
  }
  return kPi / 3.0 * sum;
}

inline double host_orient2d(const double ax,
                            const double az,
                            const double bx,
                            const double bz,
                            const double cx,
                            const double cz) {
  return (bx - ax) * (cz - az) - (bz - az) * (cx - ax);
}

inline bool host_on_segment(const double ax,
                            const double az,
                            const double bx,
                            const double bz,
                            const double px,
                            const double pz) {
  // Scale-aware predicate (2026-07-26 kernel review): the previous absolute
  // 1e-14 shared a length tolerance [cm] and an orient2d tolerance [cm^2],
  // which is dimensionally inconsistent and collapses for submicron cells.
  const double seg_scale = std::abs(bx - ax) + std::abs(bz - az);
  const double point_scale = std::abs(px - ax) + std::abs(pz - az);
  const double eps_len = 64.0 * std::numeric_limits<double>::epsilon() *
                         std::max(seg_scale, point_scale);
  const double eps_area = 64.0 * std::numeric_limits<double>::epsilon() *
                          seg_scale * point_scale;
  return px >= std::min(ax, bx) - eps_len &&
         px <= std::max(ax, bx) + eps_len &&
         pz >= std::min(az, bz) - eps_len &&
         pz <= std::max(az, bz) + eps_len &&
         std::abs(host_orient2d(ax, az, bx, bz, px, pz)) <= eps_area;
}

inline bool host_segments_intersect(const double ax,
                                    const double az,
                                    const double bx,
                                    const double bz,
                                    const double cx,
                                    const double cz,
                                    const double dx,
                                    const double dz) {
  const double ab_scale = std::abs(bx - ax) + std::abs(bz - az);
  const double cd_scale = std::abs(dx - cx) + std::abs(dz - cz);
  const double eps_ab =
      64.0 * std::numeric_limits<double>::epsilon() * ab_scale *
      std::max(std::abs(cx - ax) + std::abs(cz - az),
               std::abs(dx - ax) + std::abs(dz - az));
  const double eps_cd =
      64.0 * std::numeric_limits<double>::epsilon() * cd_scale *
      std::max(std::abs(ax - cx) + std::abs(az - cz),
               std::abs(bx - cx) + std::abs(bz - cz));
  const double o1 = host_orient2d(ax, az, bx, bz, cx, cz);
  const double o2 = host_orient2d(ax, az, bx, bz, dx, dz);
  const double o3 = host_orient2d(cx, cz, dx, dz, ax, az);
  const double o4 = host_orient2d(cx, cz, dx, dz, bx, bz);
  if (((o1 > eps_ab && o2 < -eps_ab) || (o1 < -eps_ab && o2 > eps_ab)) &&
      ((o3 > eps_cd && o4 < -eps_cd) || (o3 < -eps_cd && o4 > eps_cd))) {
    return true;
  }
  return host_on_segment(ax, az, bx, bz, cx, cz) ||
         host_on_segment(ax, az, bx, bz, dx, dz) ||
         host_on_segment(cx, cz, dx, dz, ax, az) ||
         host_on_segment(cx, cz, dx, dz, bx, bz);
}

inline bool host_simple_loop(const std::vector<double>& r,
                             const std::vector<double>& z) {
  const int n = static_cast<int>(r.size());
  if (n < 3 || z.size() != r.size()) {
    return false;
  }
  for (int a = 0; a < n; ++a) {
    for (int b = a + 1; b < n; ++b) {
      if (b == a || b == (a + 1) % n || (b + 1) % n == a) {
        continue;
      }
      if (host_segments_intersect(r[static_cast<std::size_t>(a)],
                                  z[static_cast<std::size_t>(a)],
                                  r[static_cast<std::size_t>((a + 1) % n)],
                                  z[static_cast<std::size_t>((a + 1) % n)],
                                  r[static_cast<std::size_t>(b)],
                                  z[static_cast<std::size_t>(b)],
                                  r[static_cast<std::size_t>((b + 1) % n)],
                                  z[static_cast<std::size_t>((b + 1) % n)])) {
        return false;
      }
    }
  }
  return true;
}

inline CandidateMeshQuality evaluate_macro_boundary_quality(
    const double* d_node_r_old,
    const double* d_node_z_old,
    const double* d_delta_r,
    const double* d_delta_z,
    const double sigma,
    const CandidateMeshAdmissibilityFloors& floors,
    const int* d_macro_boundary_nodes,
    const int n_macro_boundary_nodes,
    const double* d_node_r_reference,
    const double* d_node_z_reference) {
  CandidateMeshQuality q{};
  q.min_rz_volume_rel = std::numeric_limits<double>::infinity();
  q.min_corner_j_rel = std::numeric_limits<double>::infinity();
  q.min_gauss_j_rel = std::numeric_limits<double>::infinity();
  if (d_macro_boundary_nodes == nullptr || n_macro_boundary_nodes < 3) {
    return q;
  }
  std::vector<int> nodes(static_cast<std::size_t>(n_macro_boundary_nodes), -1);
  cuda_check(cudaMemcpy(nodes.data(),
                        d_macro_boundary_nodes,
                        nodes.size() * sizeof(int),
                        cudaMemcpyDeviceToHost),
             "CSR candidate macro boundary node copy failed");
  std::vector<double> r_old(nodes.size(), 0.0);
  std::vector<double> z_old(nodes.size(), 0.0);
  std::vector<double> r_new(nodes.size(), 0.0);
  std::vector<double> z_new(nodes.size(), 0.0);
  const bool reference_available =
      floors.volume_rel > 0.0 && d_node_r_reference != nullptr &&
      d_node_z_reference != nullptr;
  std::vector<double> r_reference;
  std::vector<double> z_reference;
  if (reference_available) {
    r_reference.resize(nodes.size(), 0.0);
    z_reference.resize(nodes.size(), 0.0);
  }
  for (int k = 0; k < n_macro_boundary_nodes; ++k) {
    const int node = nodes[static_cast<std::size_t>(k)];
    if (node < 0) {
      q.first_bad_cell = kCandidateMacroCoreSentinelCell;
      q.first_bad_stable_cell = kCandidateMacroCoreSentinelCell;
      q.kind = MeshGeometryFailureKind::TopologyInvariantViolation;
      return q;
    }
    double ro = 0.0;
    double zo = 0.0;
    double dr = 0.0;
    double dz = 0.0;
    cuda_check(cudaMemcpy(&ro, d_node_r_old + node, sizeof(double),
                          cudaMemcpyDeviceToHost),
               "CSR candidate macro r_old copy failed");
    cuda_check(cudaMemcpy(&zo, d_node_z_old + node, sizeof(double),
                          cudaMemcpyDeviceToHost),
               "CSR candidate macro z_old copy failed");
    cuda_check(cudaMemcpy(&dr, d_delta_r + node, sizeof(double),
                          cudaMemcpyDeviceToHost),
               "CSR candidate macro delta_r copy failed");
    cuda_check(cudaMemcpy(&dz, d_delta_z + node, sizeof(double),
                          cudaMemcpyDeviceToHost),
               "CSR candidate macro delta_z copy failed");
    r_old[static_cast<std::size_t>(k)] = ro;
    z_old[static_cast<std::size_t>(k)] = zo;
    r_new[static_cast<std::size_t>(k)] = ro + sigma * dr;
    z_new[static_cast<std::size_t>(k)] = zo + sigma * dz;
    if (reference_available) {
      cuda_check(cudaMemcpy(
                     &r_reference[static_cast<std::size_t>(k)],
                     d_node_r_reference + node, sizeof(double),
                     cudaMemcpyDeviceToHost),
                 "CSR candidate macro r_reference copy failed");
      cuda_check(cudaMemcpy(
                     &z_reference[static_cast<std::size_t>(k)],
                     d_node_z_reference + node, sizeof(double),
                     cudaMemcpyDeviceToHost),
                 "CSR candidate macro z_reference copy failed");
    }
  }
  const double v_old = host_rz_polygon_volume(r_old, z_old);
  const double v_new = host_rz_polygon_volume(r_new, z_new);
  const double v_reference =
      reference_available
          ? host_rz_polygon_volume(r_reference, z_reference)
          : 0.0;
  q.min_rz_volume_rel = signed_ratio(v_new, v_old);
  const double volume_floor =
      std::max(floors.volume_abs,
               floors.volume_rel *
                   relative_floor_baseline(v_old, v_reference,
                                           reference_available));
  if (!std::isfinite(v_old) || !std::isfinite(v_new) ||
      !(v_old > 0.0) || !(v_new > volume_floor) ||
      !host_simple_loop(r_new, z_new)) {
    q.first_bad_cell = kCandidateMacroCoreSentinelCell;
    q.first_bad_stable_cell = kCandidateMacroCoreSentinelCell;
    q.kind = MeshGeometryFailureKind::NonPositiveCellVolume;
  }
  return q;
}

}  // namespace

__device__ __host__ double rz_quad_volume_exact(
    const double r0, const double z0,
    const double r1, const double z1,
    const double r2, const double z2,
    const double r3, const double z3) noexcept {
  const double rs[4] = {r0, r1, r2, r3};
  const double zs[4] = {z0, z1, z2, z3};
  double sum = 0.0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int kp = (k + 1) & 3;
    sum += (rs[k] * zs[kp] - rs[kp] * zs[k]) * (rs[k] + rs[kp]);
  }
  return kPi / 3.0 * sum;
}

__device__ __host__ double rz_triangle_volume_exact(
    const double r0, const double z0,
    const double r1, const double z1,
    const double r2, const double z2) noexcept {
  const double rs[3] = {r0, r1, r2};
  const double zs[3] = {z0, z1, z2};
  double sum = 0.0;
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    const int kp = (k + 1) % 3;
    sum += (rs[k] * zs[kp] - rs[kp] * zs[k]) * (rs[k] + rs[kp]);
  }
  return kPi / 3.0 * sum;
}

CandidateMeshQuality evaluate_candidate_mesh_quality(
    const double* d_node_r_old,
    const double* d_node_z_old,
    const double* d_delta_r,
    const double* d_delta_z,
    const double sigma,
    const int nr,
    const int nz,
    const CandidateMeshAdmissibilityFloors& floors,
    const std::uint8_t* d_cell_nverts,
    const double* d_node_r_reference,
    const double* d_node_z_reference) {
  TENRYU_ASSERT(nr > 0, "candidate mesh quality requires nr > 0");
  TENRYU_ASSERT(nz > 0, "candidate mesh quality requires nz > 0");
  const int n_cells = nr * nz;

  CandidateMeshQuality* d_quality = nullptr;
  d_quality = static_cast<CandidateMeshQuality*>(
      core::device_scratch_acquire("cand_mesh:quality_structured:d_quality",
                                   static_cast<std::size_t>(n_cells) *
                                       sizeof(CandidateMeshQuality)));
  fill_neutral_candidate_mesh_quality(
      d_quality, n_cells,
      "candidate mesh quality neutral-fill kernel launch failed");

  constexpr int threads = 128;
  const int blocks = (n_cells + threads - 1) / threads;
  evaluate_candidate_mesh_quality_kernel<<<blocks, threads>>>(
      d_node_r_old, d_node_z_old, d_node_r_reference, d_node_z_reference,
      d_delta_r, d_delta_z, sigma, nr, nz, floors, d_cell_nverts, nullptr, 0,
      d_quality);
  cuda_check(cudaGetLastError(), "candidate mesh quality kernel launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "candidate mesh quality kernel synchronize failed");

  std::vector<CandidateMeshQuality> quality(static_cast<std::size_t>(n_cells));
  cuda_check(cudaMemcpy(quality.data(), d_quality,
                        static_cast<std::size_t>(n_cells) *
                            sizeof(CandidateMeshQuality),
                        cudaMemcpyDeviceToHost),
             "candidate mesh quality D2H copy failed");

  CandidateMeshQuality out{};
  out.min_rz_volume_rel = std::numeric_limits<double>::infinity();
  out.min_corner_j_rel = std::numeric_limits<double>::infinity();
  out.min_gauss_j_rel = std::numeric_limits<double>::infinity();

  for (int c = 0; c < n_cells; ++c) {
    const CandidateMeshQuality& q = quality[static_cast<std::size_t>(c)];
    out.min_rz_volume_rel =
        std::min(out.min_rz_volume_rel, q.min_rz_volume_rel);
    out.min_corner_j_rel =
        std::min(out.min_corner_j_rel, q.min_corner_j_rel);
    out.min_gauss_j_rel =
        std::min(out.min_gauss_j_rel, q.min_gauss_j_rel);
    if (out.kind == MeshGeometryFailureKind::None &&
        q.kind != MeshGeometryFailureKind::None) {
      out.kind = q.kind;
      out.first_bad_cell = q.first_bad_cell;
      out.first_bad_corner = q.first_bad_corner;
    }
  }

  return out;
}

CandidateMeshQuality evaluate_candidate_mesh_quality(
    const Mesh& mesh,
    const double* d_node_r_old,
    const double* d_node_z_old,
    const double* d_delta_r,
    const double* d_delta_z,
    const double sigma,
    const CandidateMeshAdmissibilityFloors& floors,
    const double* d_node_r_reference,
    const double* d_node_z_reference) {
  TENRYU_ASSERT(mesh.dim == 2,
                "mesh candidate quality overload requires a 2D mesh");
  TENRYU_ASSERT(!mesh.topo.multiblock.has_value(),
                "mesh candidate quality overload is single-block only");
  TENRYU_ASSERT(mesh.topo.nr > 0,
                "mesh candidate quality requires nr > 0");
  TENRYU_ASSERT(mesh.topo.nz > 0,
                "mesh candidate quality requires nz > 0");
  const int n_cells = mesh.topo.nr * mesh.topo.nz;
  TENRYU_ASSERT(n_cells == mesh.topo.n_cells,
                "mesh candidate quality topology cell count mismatch");

  std::vector<std::uint8_t> cell_kind(static_cast<std::size_t>(n_cells),
                                      kCandidateCellStructured);
  bool has_special_cells = false;
  for (int c = 0; c < n_cells; ++c) {
    if (mesh.is_button_cell(c)) {
      cell_kind[static_cast<std::size_t>(c)] = kCandidateCellButton;
      has_special_cells = true;
    } else if (mesh.is_dormant_cell(c)) {
      cell_kind[static_cast<std::size_t>(c)] = kCandidateCellDormant;
      has_special_cells = true;
    }
  }

  std::uint8_t* d_cell_kind = nullptr;
  if (has_special_cells) {
    d_cell_kind = static_cast<std::uint8_t*>(
        core::device_scratch_acquire("cand_mesh:quality_mesh:d_cell_kind",
                                     static_cast<std::size_t>(n_cells) *
                                         sizeof(std::uint8_t)));
    cuda_check(cudaMemcpy(d_cell_kind,
                          cell_kind.data(),
                          static_cast<std::size_t>(n_cells) *
                              sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice),
               "mesh candidate quality memcpy cell_kind failed");
  }

  std::uint8_t* d_cell_nverts = nullptr;
  const bool polar_family =
      mesh.logical == LogicalMesh2D::SphericalPolarHalfplane ||
      mesh.logical == LogicalMesh2D::PolarInBox;
  const bool use_cell_nverts =
      mesh.polar_center_treatment == PolarCenterTreatment::TriFan ||
      mesh.polar_center_treatment == PolarCenterTreatment::Button ||
      (polar_family &&
       mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells));
  if (use_cell_nverts) {
    TENRYU_ASSERT(mesh.cell_nverts.size() ==
                      static_cast<std::size_t>(n_cells),
                  "polar center mesh candidate quality requires cell_nverts");
    d_cell_nverts = static_cast<std::uint8_t*>(
        core::device_scratch_acquire("cand_mesh:quality_mesh:d_cell_nverts",
                                     static_cast<std::size_t>(n_cells) *
                                         sizeof(std::uint8_t)));
    cuda_check(cudaMemcpy(d_cell_nverts,
                          mesh.cell_nverts.data(),
                          static_cast<std::size_t>(n_cells) *
                              sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice),
               "mesh candidate quality memcpy cell_nverts failed");
  }

  CandidateMeshQuality* d_quality = nullptr;
  d_quality = static_cast<CandidateMeshQuality*>(
      core::device_scratch_acquire("cand_mesh:quality_mesh:d_quality",
                                   static_cast<std::size_t>(n_cells) *
                                       sizeof(CandidateMeshQuality)));
  fill_neutral_candidate_mesh_quality(
      d_quality, n_cells,
      "mesh candidate quality neutral-fill kernel launch failed");

  constexpr int threads = 128;
  const int blocks = (n_cells + threads - 1) / threads;
  const int button_outer_node_ring =
      (mesh.button_center && mesh.button_center->enabled)
          ? mesh.button_center->outer_node_ring
          : 0;
  evaluate_candidate_mesh_quality_kernel<<<blocks, threads>>>(
      d_node_r_old, d_node_z_old, d_node_r_reference, d_node_z_reference,
      d_delta_r, d_delta_z, sigma, mesh.topo.nr, mesh.topo.nz, floors,
      d_cell_nverts, d_cell_kind, button_outer_node_ring, d_quality);
  cuda_check(cudaGetLastError(),
             "mesh candidate quality kernel launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "mesh candidate quality kernel synchronize failed");

  std::vector<CandidateMeshQuality> quality(static_cast<std::size_t>(n_cells));
  cuda_check(cudaMemcpy(quality.data(), d_quality,
                        static_cast<std::size_t>(n_cells) *
                            sizeof(CandidateMeshQuality),
                        cudaMemcpyDeviceToHost),
             "mesh candidate quality D2H copy failed");

  CandidateMeshQuality out{};
  out.min_rz_volume_rel = std::numeric_limits<double>::infinity();
  out.min_corner_j_rel = std::numeric_limits<double>::infinity();
  out.min_gauss_j_rel = std::numeric_limits<double>::infinity();

  for (int c = 0; c < n_cells; ++c) {
    const CandidateMeshQuality& q = quality[static_cast<std::size_t>(c)];
    out.min_rz_volume_rel =
        std::min(out.min_rz_volume_rel, q.min_rz_volume_rel);
    out.min_corner_j_rel =
        std::min(out.min_corner_j_rel, q.min_corner_j_rel);
    out.min_gauss_j_rel =
        std::min(out.min_gauss_j_rel, q.min_gauss_j_rel);
    if (out.kind == MeshGeometryFailureKind::None &&
        q.kind != MeshGeometryFailureKind::None) {
      out.kind = q.kind;
      out.first_bad_cell = q.first_bad_cell;
      out.first_bad_corner = q.first_bad_corner;
    }
  }

  return out;
}

CandidateMeshQuality evaluate_candidate_mesh_quality_csr(
    const double* d_node_r_old,
    const double* d_node_z_old,
    const double* d_delta_r,
    const double* d_delta_z,
    const double sigma,
    const int n_cells_total,
    const int topology_stride,
    const int* d_cell_node_csr_offsets,
    const int* d_cell_node_csr_indices,
    const int* d_cell_id_stable,
    const CandidateMeshAdmissibilityFloors& floors,
    const std::uint8_t* d_cell_nverts,
    const std::uint8_t* d_inactive_cell_mask,
    const int* d_macro_boundary_nodes,
    const int n_macro_boundary_nodes,
    const double* d_node_r_reference,
    const double* d_node_z_reference) {
  return evaluate_candidate_mesh_quality_csr(
      d_node_r_old, d_node_z_old, d_delta_r, d_delta_z, sigma, n_cells_total,
      topology_stride, d_cell_node_csr_offsets, d_cell_node_csr_indices,
      d_cell_id_stable, nullptr, floors, d_cell_nverts, d_inactive_cell_mask,
      d_macro_boundary_nodes, n_macro_boundary_nodes, d_node_r_reference,
      d_node_z_reference);
}

CandidateMeshQuality evaluate_candidate_mesh_quality_csr(
    const double* d_node_r_old,
    const double* d_node_z_old,
    const double* d_delta_r,
    const double* d_delta_z,
    const double sigma,
    const int n_cells_total,
    const int topology_stride,
    const int* d_cell_node_csr_offsets,
    const int* d_cell_node_csr_indices,
    const int* d_cell_id_stable,
    const int* d_cell_orientation_sign,
    const CandidateMeshAdmissibilityFloors& floors,
    const std::uint8_t* d_cell_nverts,
    const std::uint8_t* d_inactive_cell_mask,
    const int* d_macro_boundary_nodes,
    const int n_macro_boundary_nodes,
    const double* d_node_r_reference,
    const double* d_node_z_reference) {
  TENRYU_ASSERT(n_cells_total > 0,
                "CSR candidate mesh quality requires n_cells_total > 0");
  TENRYU_ASSERT(d_cell_node_csr_offsets != nullptr,
                "CSR candidate mesh quality requires cell-node offsets");
  TENRYU_ASSERT(d_cell_node_csr_indices != nullptr,
                "CSR candidate mesh quality requires cell-node indices");
  TENRYU_ASSERT(d_cell_id_stable != nullptr,
                "CSR candidate mesh quality requires stable cell ids");

  CandidateMeshQuality* d_quality = nullptr;
  d_quality = static_cast<CandidateMeshQuality*>(
      core::device_scratch_acquire("cand_mesh:quality_csr:d_quality",
                                   static_cast<std::size_t>(n_cells_total) *
                                       sizeof(CandidateMeshQuality)));
  fill_neutral_candidate_mesh_quality(
      d_quality, n_cells_total,
      "CSR candidate mesh quality neutral-fill kernel launch failed");

  constexpr int threads = 128;
  const int blocks = (n_cells_total + threads - 1) / threads;
  evaluate_candidate_mesh_quality_csr_kernel<<<blocks, threads>>>(
      d_node_r_old, d_node_z_old, d_node_r_reference, d_node_z_reference,
      d_delta_r, d_delta_z, sigma, n_cells_total, topology_stride,
      d_cell_node_csr_offsets, d_cell_node_csr_indices, d_cell_id_stable,
      d_cell_orientation_sign, floors, d_cell_nverts, d_inactive_cell_mask,
      d_quality);
  cuda_check(cudaGetLastError(),
             "CSR candidate mesh quality kernel launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "CSR candidate mesh quality kernel synchronize failed");

  std::vector<CandidateMeshQuality> quality(
      static_cast<std::size_t>(n_cells_total));
  cuda_check(cudaMemcpy(quality.data(), d_quality,
                        static_cast<std::size_t>(n_cells_total) *
                            sizeof(CandidateMeshQuality),
                        cudaMemcpyDeviceToHost),
             "CSR candidate mesh quality D2H copy failed");

  CandidateMeshQuality out{};
  out.min_rz_volume_rel = std::numeric_limits<double>::infinity();
  out.min_corner_j_rel = std::numeric_limits<double>::infinity();
  out.min_gauss_j_rel = std::numeric_limits<double>::infinity();

  for (int c = 0; c < n_cells_total; ++c) {
    const CandidateMeshQuality& q = quality[static_cast<std::size_t>(c)];
    out.min_rz_volume_rel =
        std::min(out.min_rz_volume_rel, q.min_rz_volume_rel);
    out.min_corner_j_rel =
        std::min(out.min_corner_j_rel, q.min_corner_j_rel);
    out.min_gauss_j_rel =
        std::min(out.min_gauss_j_rel, q.min_gauss_j_rel);
    if (q.kind == MeshGeometryFailureKind::None) {
      continue;
    }
    const bool replace =
        out.kind == MeshGeometryFailureKind::None ||
        q.first_bad_stable_cell < out.first_bad_stable_cell;
    if (replace) {
      out.kind = q.kind;
      out.first_bad_cell = q.first_bad_cell;
      out.first_bad_stable_cell = q.first_bad_stable_cell;
      out.first_bad_corner = q.first_bad_corner;
    }
  }

  const CandidateMeshQuality macro_quality = evaluate_macro_boundary_quality(
      d_node_r_old, d_node_z_old, d_delta_r, d_delta_z, sigma, floors,
      d_macro_boundary_nodes, n_macro_boundary_nodes, d_node_r_reference,
      d_node_z_reference);
  out.min_rz_volume_rel =
      std::min(out.min_rz_volume_rel, macro_quality.min_rz_volume_rel);
  out.min_corner_j_rel =
      std::min(out.min_corner_j_rel, macro_quality.min_corner_j_rel);
  out.min_gauss_j_rel =
      std::min(out.min_gauss_j_rel, macro_quality.min_gauss_j_rel);
  if (macro_quality.kind != MeshGeometryFailureKind::None) {
    const bool replace =
        out.kind == MeshGeometryFailureKind::None ||
        macro_quality.first_bad_stable_cell < out.first_bad_stable_cell;
    if (replace) {
      out.kind = macro_quality.kind;
      out.first_bad_cell = macro_quality.first_bad_cell;
      out.first_bad_stable_cell = macro_quality.first_bad_stable_cell;
      out.first_bad_corner = macro_quality.first_bad_corner;
    }
  }

  return out;
}

// Contract note (2026-07-26 kernel review): this is a dyadic halving search — it
// returns the FIRST accepted sigma among sigma_init / 2^k, which is not the
// supremum of admissible sigma. If halving drops below sigma_min, the exact
// sigma_min value is never evaluated and 0.0 is returned.
LineSearchResult linesearch_largest_admissible_sigma(
    const double* d_node_r_old,
    const double* d_node_z_old,
    const double* d_delta_r,
    const double* d_delta_z,
    const double sigma_init,
    const double sigma_min,
    const int max_iters,
    const int nr,
    const int nz,
    const CandidateMeshAdmissibilityFloors& floors,
    const std::uint8_t* d_cell_nverts,
    const double* d_node_r_reference,
    const double* d_node_z_reference) {
  LineSearchResult result{};
  double sigma = sigma_init;
  const int iter_limit = std::max(0, max_iters);

  for (int iter = 0; iter <= iter_limit; ++iter) {
    result.quality = evaluate_candidate_mesh_quality(
        d_node_r_old, d_node_z_old, d_delta_r, d_delta_z, sigma, nr, nz,
        floors, d_cell_nverts, d_node_r_reference, d_node_z_reference);
    if (result.quality.admissible()) {
      result.sigma_accepted = sigma;
      return result;
    }
    if (iter == iter_limit) {
      break;
    }
    sigma *= 0.5;
    ++result.iters_used;
    if (sigma < sigma_min) {
      break;
    }
  }

  result.sigma_accepted = 0.0;
  return result;
}

LineSearchResult linesearch_largest_admissible_sigma(
    const Mesh& mesh,
    const double* d_node_r_old,
    const double* d_node_z_old,
    const double* d_delta_r,
    const double* d_delta_z,
    const double sigma_init,
    const double sigma_min,
    const int max_iters,
    const CandidateMeshAdmissibilityFloors& floors,
    const double* d_node_r_reference,
    const double* d_node_z_reference) {
  LineSearchResult result{};
  double sigma = sigma_init;
  const int iter_limit = std::max(0, max_iters);

  for (int iter = 0; iter <= iter_limit; ++iter) {
    result.quality = evaluate_candidate_mesh_quality(
        mesh, d_node_r_old, d_node_z_old, d_delta_r, d_delta_z, sigma,
        floors, d_node_r_reference, d_node_z_reference);
    if (result.quality.admissible()) {
      result.sigma_accepted = sigma;
      return result;
    }
    if (iter == iter_limit) {
      break;
    }
    sigma *= 0.5;
    ++result.iters_used;
    if (sigma < sigma_min) {
      break;
    }
  }

  result.sigma_accepted = 0.0;
  return result;
}

LineSearchResult linesearch_largest_admissible_sigma_csr(
    const double* d_node_r_old,
    const double* d_node_z_old,
    const double* d_delta_r,
    const double* d_delta_z,
    const double sigma_init,
    const double sigma_min,
    const int max_iters,
    const int n_cells_total,
    const int topology_stride,
    const int* d_cell_node_csr_offsets,
    const int* d_cell_node_csr_indices,
    const int* d_cell_id_stable,
    const CandidateMeshAdmissibilityFloors& floors,
    const std::uint8_t* d_cell_nverts,
    const std::uint8_t* d_inactive_cell_mask,
    const int* d_macro_boundary_nodes,
    const int n_macro_boundary_nodes,
    const double* d_node_r_reference,
    const double* d_node_z_reference) {
  return linesearch_largest_admissible_sigma_csr(
      d_node_r_old, d_node_z_old, d_delta_r, d_delta_z, sigma_init, sigma_min,
      max_iters, n_cells_total, topology_stride, d_cell_node_csr_offsets,
      d_cell_node_csr_indices, d_cell_id_stable, nullptr, floors,
      d_cell_nverts, d_inactive_cell_mask, d_macro_boundary_nodes,
      n_macro_boundary_nodes, d_node_r_reference, d_node_z_reference);
}

LineSearchResult linesearch_largest_admissible_sigma_csr(
    const double* d_node_r_old,
    const double* d_node_z_old,
    const double* d_delta_r,
    const double* d_delta_z,
    const double sigma_init,
    const double sigma_min,
    const int max_iters,
    const int n_cells_total,
    const int topology_stride,
    const int* d_cell_node_csr_offsets,
    const int* d_cell_node_csr_indices,
    const int* d_cell_id_stable,
    const int* d_cell_orientation_sign,
    const CandidateMeshAdmissibilityFloors& floors,
    const std::uint8_t* d_cell_nverts,
    const std::uint8_t* d_inactive_cell_mask,
    const int* d_macro_boundary_nodes,
    const int n_macro_boundary_nodes,
    const double* d_node_r_reference,
    const double* d_node_z_reference) {
  LineSearchResult result{};
  double sigma = sigma_init;
  const int iter_limit = std::max(0, max_iters);

  for (int iter = 0; iter <= iter_limit; ++iter) {
    result.quality = evaluate_candidate_mesh_quality_csr(
        d_node_r_old, d_node_z_old, d_delta_r, d_delta_z, sigma, n_cells_total,
        topology_stride, d_cell_node_csr_offsets, d_cell_node_csr_indices,
        d_cell_id_stable, d_cell_orientation_sign, floors, d_cell_nverts,
        d_inactive_cell_mask, d_macro_boundary_nodes, n_macro_boundary_nodes,
        d_node_r_reference, d_node_z_reference);
    if (result.quality.admissible()) {
      result.sigma_accepted = sigma;
      return result;
    }
    if (iter == iter_limit) {
      break;
    }
    sigma *= 0.5;
    ++result.iters_used;
    if (sigma < sigma_min) {
      break;
    }
  }

  result.sigma_accepted = 0.0;
  return result;
}

}  // namespace tenryu::mesh
