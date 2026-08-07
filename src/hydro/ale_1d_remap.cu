#include "hydro/ale_1d_remap.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <cub/cub.cuh>

#include "core/fancy_iterators.cuh"
#include "mesh/geometry_1d.cuh"

namespace tenryu::hydro::ale1d {
namespace {

constexpr int kBlockSize = 256;
constexpr double kFourPiOverThree =
    4.188790204786390984616857844372670512262892532500141094646;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTiny = 1.0e-300;
constexpr double kZeroSweep = 1.0e-30;
constexpr double kRoundoff = 2.220446049250313080847263336181640625e-16;
constexpr int kFallbackMass = 1 << 0;
constexpr int kFallbackEe = 1 << 1;
constexpr int kFallbackEi = 1 << 2;
constexpr int kFallbackRadiation = 1 << 3;
constexpr int kFallbackMaterial = 1 << 4;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

int blocks_for(const int n) {
  return (n + kBlockSize - 1) / kBlockSize;
}

int effective_cell_count(const core::State& state, const core::Config& cfg) {
  if (state.mesh.topo.n_cells > 0) {
    return state.mesh.topo.n_cells;
  }
  return cfg.mesh.nr;
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(const std::size_t count) {
    reset(count);
  }

  ~DeviceBuffer() {
    release();
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void reset(const std::size_t count) {
    release();
    size_ = count;
    if (size_ == 0) {
      return;
    }
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr_), size_ * sizeof(T)),
               "ALE1D remap cudaMalloc failed");
  }

  T* data() noexcept {
    return ptr_;
  }

  const T* data() const noexcept {
    return ptr_;
  }

 private:
  void release() {
    if (ptr_ != nullptr) {
      cuda_check(cudaFree(ptr_), "ALE1D remap cudaFree failed");
      ptr_ = nullptr;
    }
    size_ = 0;
  }

  T* ptr_ = nullptr;
  std::size_t size_ = 0;
};

__host__ __device__ double volume_coordinate(const double r, const int geom) {
  return (geom == 0) ? (kFourPiOverThree * r * r * r)
                     : tenryu::mesh::geometry_1d_shell_volume_cubes(geom, 0.0, r);
}

__host__ __device__ double cell_volume_from_nodes(
    const double* __restrict__ r,
    const int i,
    const int geom) {
  return volume_coordinate(r[i + 1], geom) - volume_coordinate(r[i], geom);
}

__device__ bool same_coordinate(const double a, const double b) {
  const double scale = fmax(1.0, fmax(fabs(a), fabs(b)));
  return fabs(a - b) <= 64.0 * kRoundoff * scale;
}

__device__ bool y_leq_with_roundoff(const double a, const double b) {
  const double scale = fmax(1.0, fmax(fabs(a), fabs(b)));
  return a <= b + 128.0 * kRoundoff * scale;
}

__device__ bool y_geq_with_roundoff(const double a, const double b) {
  const double scale = fmax(1.0, fmax(fabs(a), fabs(b)));
  return a + 128.0 * kRoundoff * scale >= b;
}

__global__ void build_faces_kernel(const double* __restrict__ r_old,
                                   const double* __restrict__ r_new,
                                   const std::uint8_t* __restrict__ pinned,
                                   double* __restrict__ delta_y,
                                   int* __restrict__ donor,
                                   int* __restrict__ invalid_count,
                                   const int n,
                                   const bool reject_multicell_sweeps,
                                   const int geom) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > n) {
    return;
  }

  donor[j] = -1;
  delta_y[j] = 0.0;

  const double ro = r_old[j];
  const double rn = r_new[j];
  if (!isfinite(ro) || !isfinite(rn)) {
    atomicAdd(invalid_count, 1);
    return;
  }

  if (pinned[j] != 0U) {
    if (!same_coordinate(ro, rn)) {
      atomicAdd(invalid_count, 1);
    }
    return;
  }

  const double yo = volume_coordinate(ro, geom);
  const double yn = volume_coordinate(rn, geom);
  if (!isfinite(yo) || !isfinite(yn)) {
    atomicAdd(invalid_count, 1);
    return;
  }

  const double dy = yn - yo;
  if (fabs(dy) <= kZeroSweep) {
    return;
  }

  if (dy > 0.0) {
    if (j >= n) {
      atomicAdd(invalid_count, 1);
      return;
    }
    if (reject_multicell_sweeps &&
        !y_leq_with_roundoff(yn, volume_coordinate(r_old[j + 1], geom))) {
      atomicAdd(invalid_count, 1);
      delta_y[j] = dy;
      return;
    }
    donor[j] = j;
    delta_y[j] = dy;
    return;
  }

  if (j <= 0) {
    atomicAdd(invalid_count, 1);
    return;
  }
  if (reject_multicell_sweeps &&
      !y_geq_with_roundoff(yn, volume_coordinate(r_old[j - 1], geom))) {
    atomicAdd(invalid_count, 1);
    delta_y[j] = dy;
    return;
  }
  donor[j] = j - 1;
  delta_y[j] = dy;
}

__global__ void build_new_volumes_kernel(const double* __restrict__ r_new,
                                         double* __restrict__ vol_new,
                                         int* __restrict__ invalid_count,
                                         const int n,
                                         const int geom) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const double vol = cell_volume_from_nodes(r_new, i, geom);
  vol_new[i] = vol;
  if (!isfinite(vol) || !(vol > 0.0)) {
    atomicAdd(invalid_count, 1);
  }
}

__global__ void build_phi_kernel(const std::uint8_t* __restrict__ protected_face,
                                 double* __restrict__ phi_face,
                                 const int n,
                                 const int ramp_cells) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > n) {
    return;
  }

  int distance = n + 1;
  for (int p = 0; p <= n; ++p) {
    if (protected_face[p] != 0U) {
      const int d = (j >= p) ? (j - p) : (p - j);
      distance = (distance < d) ? distance : d;
    }
  }

  if (distance == 0) {
    phi_face[j] = 0.0;
  } else if (ramp_cells <= 0 || distance >= ramp_cells) {
    phi_face[j] = 1.0;
  } else {
    phi_face[j] =
        0.5 * (1.0 - cos(kPi * static_cast<double>(distance) /
                         static_cast<double>(ramp_cells)));
  }
}

__device__ double cell_center_y(const double* __restrict__ r,
                                const int i,
                                const int geom) {
  return 0.5 * (volume_coordinate(r[i], geom) +
                volume_coordinate(r[i + 1], geom));
}

__device__ double minmod3(const double a, const double b, const double c) {
  if (a > 0.0 && b > 0.0 && c > 0.0) {
    return fmin(a, fmin(b, c));
  }
  if (a < 0.0 && b < 0.0 && c < 0.0) {
    return -fmin(fabs(a), fmin(fabs(b), fabs(c)));
  }
  return 0.0;
}

__device__ double slope_scale_for_face_value(const double q_face,
                                             const double q_cell,
                                             const double q_min,
                                             const double q_max) {
  if (q_face > q_max) {
    const double denom = q_face - q_cell;
    return denom > kTiny ? fmax(0.0, fmin(1.0, (q_max - q_cell) / denom))
                         : 0.0;
  }
  if (q_face < q_min) {
    const double denom = q_face - q_cell;
    return denom < -kTiny ? fmax(0.0, fmin(1.0, (q_min - q_cell) / denom))
                          : 0.0;
  }
  return 1.0;
}

__global__ void compute_limited_slopes_kernel(
    const double* __restrict__ q,
    const double* __restrict__ r_old,
    const std::uint8_t* __restrict__ protected_face,
    double* __restrict__ slope,
    const int n,
    const double theta,
    const int geom) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  slope[i] = 0.0;
  if (i <= 0 || i >= n - 1 || protected_face[i] != 0U ||
      protected_face[i + 1] != 0U) {
    return;
  }

  const double qm = q[i - 1];
  const double qi = q[i];
  const double qp = q[i + 1];
  if (!isfinite(qm) || !isfinite(qi) || !isfinite(qp)) {
    return;
  }

  const double ycm = cell_center_y(r_old, i - 1, geom);
  const double yc = cell_center_y(r_old, i, geom);
  const double ycp = cell_center_y(r_old, i + 1, geom);
  if (!(yc > ycm) || !(ycp > yc)) {
    return;
  }

  double s = minmod3(theta * (qi - qm) / (yc - ycm),
                     (qp - qm) / (ycp - ycm),
                     theta * (qp - qi) / (ycp - yc));
  if (s == 0.0 || !isfinite(s)) {
    return;
  }

  const double y_l = volume_coordinate(r_old[i], geom);
  const double y_r = volume_coordinate(r_old[i + 1], geom);
  const double q_l = qi + s * (y_l - yc);
  const double q_r = qi + s * (y_r - yc);
  const double q_min = fmin(qm, fmin(qi, qp));
  const double q_max = fmax(qm, fmax(qi, qp));
  const double alpha =
      fmin(slope_scale_for_face_value(q_l, qi, q_min, q_max),
           slope_scale_for_face_value(q_r, qi, q_min, q_max));
  slope[i] = alpha * s;
}

__global__ void fill_mass_density_kernel(const double* __restrict__ r_old,
                                         const double* __restrict__ mass,
                                         double* __restrict__ q,
                                         const int n,
                                         const int geom) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  q[i] = mass[i] / fmax(cell_volume_from_nodes(r_old, i, geom), kTiny);
}

__global__ void fill_material_energy_density_kernel(
    const double* __restrict__ r_old,
    const double* __restrict__ mass,
    const double* __restrict__ e,
    double* __restrict__ q,
    const int n,
    const int geom) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  q[i] = mass[i] * e[i] /
         fmax(cell_volume_from_nodes(r_old, i, geom), kTiny);
}

__global__ void fill_material_mass_density_kernel(
    const double* __restrict__ r_old,
    const double* __restrict__ mass,
    const double* __restrict__ volfrac,
    double* __restrict__ q,
    const int n,
    const int n_mat,
    const int mat,
    const int geom) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  q[i] = mass[i] * volfrac[i * n_mat + mat] /
         fmax(cell_volume_from_nodes(r_old, i, geom), kTiny);
}

__global__ void fill_radiation_density_kernel(
    const double* __restrict__ rad_e,
    double* __restrict__ q,
    const int n,
    const int n_groups,
    const int group) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  q[i] = rad_e[i * n_groups + group];
}

__device__ double limited_face_flux(const int face,
                                    const double* __restrict__ r_old,
                                    const double* __restrict__ delta_y,
                                    const int* __restrict__ donor,
                                    const double* __restrict__ phi_face,
                                    const double* __restrict__ q,
                                    const double* __restrict__ slope,
                                    const int geom) {
  const int d = donor[face];
  if (d < 0) {
    return 0.0;
  }
  const double dy = delta_y[face];
  const double f1 = dy * q[d];
  const double y_bar = volume_coordinate(r_old[face], geom) + 0.5 * dy;
  const double q_bar = q[d] + slope[d] * (y_bar - cell_center_y(r_old, d, geom));
  const double f2 = dy * q_bar;
  return f1 + phi_face[face] * (f2 - f1);
}

__global__ void remap_density_to_extensive_kernel(
    const double* __restrict__ r_old,
    const double* __restrict__ q,
    const double* __restrict__ delta_y,
    const int* __restrict__ donor,
    const double* __restrict__ phi_face,
    const double* __restrict__ slope,
    double* __restrict__ q_ext_new,
    const int n,
    const int geom) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const double q_old_ext = q[i] * cell_volume_from_nodes(r_old, i, geom);
  const double fp =
      limited_face_flux(i + 1, r_old, delta_y, donor, phi_face, q, slope, geom);
  const double fm =
      limited_face_flux(i, r_old, delta_y, donor, phi_face, q, slope, geom);
  q_ext_new[i] = q_old_ext + fp - fm;
}

__global__ void remap_density_to_density_kernel(
    const double* __restrict__ r_old,
    const double* __restrict__ vol_new,
    const double* __restrict__ q,
    const double* __restrict__ delta_y,
    const int* __restrict__ donor,
    const double* __restrict__ phi_face,
    const double* __restrict__ slope,
    double* __restrict__ q_new,
    const int n,
    const int geom) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const double q_old_ext = q[i] * cell_volume_from_nodes(r_old, i, geom);
  const double fp =
      limited_face_flux(i + 1, r_old, delta_y, donor, phi_face, q, slope, geom);
  const double fm =
      limited_face_flux(i, r_old, delta_y, donor, phi_face, q, slope, geom);
  q_new[i] = (q_old_ext + fp - fm) / fmax(vol_new[i], kTiny);
}

__global__ void extensive_to_specific_kernel(const double* __restrict__ q_ext,
                                             const double* __restrict__ mass,
                                             double* __restrict__ specific,
                                             const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  specific[i] = (mass[i] > kTiny) ? q_ext[i] / mass[i] : 0.0;
}

__global__ void scatter_material_extensive_kernel(
    const double* __restrict__ material_ext,
    double* __restrict__ material_ext_by_cell,
    const int n,
    const int n_mat,
    const int mat) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  material_ext_by_cell[i * n_mat + mat] = material_ext[i];
}

__global__ void scatter_radiation_density_kernel(
    const double* __restrict__ group_density,
    double* __restrict__ rad_e_new,
    const int n,
    const int n_groups,
    const int group) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  rad_e_new[i * n_groups + group] = group_density[i];
}

__device__ bool bounds_fail(const double q_new,
                            const double q_min,
                            const double q_max) {
  const double scale = fmax(1.0, fmax(fabs(q_min), fabs(q_max)));
  const double tol = 1.0e-11 * scale + 1.0e-14;
  return !isfinite(q_new) || q_new < -tol || q_new < q_min - tol ||
         q_new > q_max + tol;
}

__global__ void validate_extensive_density_bounds_kernel(
    const double* __restrict__ q_old,
    const double* __restrict__ vol_new,
    const double* __restrict__ q_ext_new,
    int* __restrict__ fallback_flags,
    int* __restrict__ fail_count,
    const int n,
    const int fallback_bit,
    const bool record_fallback) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int im = (i > 0) ? i - 1 : 0;
  const int ip = (i + 1 < n) ? i + 1 : n - 1;
  const double q_min = fmin(q_old[im], fmin(q_old[i], q_old[ip]));
  const double q_max = fmax(q_old[im], fmax(q_old[i], q_old[ip]));
  const double q_new = q_ext_new[i] / fmax(vol_new[i], kTiny);
  if (bounds_fail(q_new, q_min, q_max)) {
    atomicAdd(fail_count, 1);
    if (record_fallback && fallback_flags != nullptr) {
      atomicOr(&fallback_flags[i], fallback_bit);
    }
  }
}

__global__ void validate_density_bounds_kernel(
    const double* __restrict__ q_old,
    const double* __restrict__ q_new,
    int* __restrict__ fallback_flags,
    int* __restrict__ fail_count,
    const int n,
    const int fallback_bit,
    const bool record_fallback) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int im = (i > 0) ? i - 1 : 0;
  const int ip = (i + 1 < n) ? i + 1 : n - 1;
  const double q_min = fmin(q_old[im], fmin(q_old[i], q_old[ip]));
  const double q_max = fmax(q_old[im], fmax(q_old[i], q_old[ip]));
  if (bounds_fail(q_new[i], q_min, q_max)) {
    atomicAdd(fail_count, 1);
    if (record_fallback && fallback_flags != nullptr) {
      atomicOr(&fallback_flags[i], fallback_bit);
    }
  }
}

__global__ void validate_specific_bounds_kernel(
    const double* __restrict__ q_old,
    const double* __restrict__ mass_new,
    const double* __restrict__ q_ext_new,
    int* __restrict__ fallback_flags,
    int* __restrict__ fail_count,
    const int n,
    const int fallback_bit,
    const bool record_fallback) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int im = (i > 0) ? i - 1 : 0;
  const int ip = (i + 1 < n) ? i + 1 : n - 1;
  const double q_min = fmin(q_old[im], fmin(q_old[i], q_old[ip]));
  const double q_max = fmax(q_old[im], fmax(q_old[i], q_old[ip]));
  const double q_new = (mass_new[i] > kTiny) ? q_ext_new[i] / mass_new[i] : 0.0;
  if (bounds_fail(q_new, q_min, q_max)) {
    atomicAdd(fail_count, 1);
    if (record_fallback && fallback_flags != nullptr) {
      atomicOr(&fallback_flags[i], fallback_bit);
    }
  }
}

__device__ double mass_flux(const int face,
                            const double* __restrict__ delta_y,
                            const int* __restrict__ donor,
                            const double* __restrict__ mass,
                            const double* __restrict__ r_old,
                            const int geom) {
  const int d = donor[face];
  if (d < 0) {
    return 0.0;
  }
  const double vol = fmax(cell_volume_from_nodes(r_old, d, geom), kTiny);
  return delta_y[face] * mass[d] / vol;
}

__device__ double material_energy_flux(const int face,
                                       const double* __restrict__ delta_y,
                                       const int* __restrict__ donor,
                                       const double* __restrict__ mass,
                                       const double* __restrict__ e,
                                       const double* __restrict__ r_old,
                                       const int geom) {
  const int d = donor[face];
  if (d < 0) {
    return 0.0;
  }
  const double vol = fmax(cell_volume_from_nodes(r_old, d, geom), kTiny);
  return delta_y[face] * mass[d] * e[d] / vol;
}

__device__ double volfrac_mass_flux(const int face,
                                    const int mat,
                                    const int n_mat,
                                    const double* __restrict__ delta_y,
                                    const int* __restrict__ donor,
                                    const double* __restrict__ mass,
                                    const double* __restrict__ volfrac,
                                    const double* __restrict__ r_old,
                                    const int geom) {
  const int d = donor[face];
  if (d < 0) {
    return 0.0;
  }
  const double vol = fmax(cell_volume_from_nodes(r_old, d, geom), kTiny);
  return delta_y[face] * mass[d] * volfrac[d * n_mat + mat] / vol;
}

__device__ double radiation_flux(const int face,
                                 const int group,
                                 const int n_groups,
                                 const double* __restrict__ delta_y,
                                 const int* __restrict__ donor,
                                 const double* __restrict__ rad_e) {
  const int d = donor[face];
  if (d < 0) {
    return 0.0;
  }
  return delta_y[face] * rad_e[d * n_groups + group];
}

__global__ void remap_mass_energy_kernel(
    const double* __restrict__ r_old,
    const double* __restrict__ mass,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ delta_y,
    const int* __restrict__ donor,
    double* __restrict__ mass_new,
    double* __restrict__ ee_new,
    double* __restrict__ ei_new,
    const int n,
    const int geom) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  const double mp = mass_flux(i + 1, delta_y, donor, mass, r_old, geom);
  const double mm = mass_flux(i, delta_y, donor, mass, r_old, geom);
  const double m_new = mass[i] + mp - mm;
  mass_new[i] = m_new;

  const double eep =
      material_energy_flux(i + 1, delta_y, donor, mass, ee, r_old, geom);
  const double eem =
      material_energy_flux(i, delta_y, donor, mass, ee, r_old, geom);
  const double ee_ext_new = mass[i] * ee[i] + eep - eem;
  ee_new[i] = (m_new > kTiny) ? ee_ext_new / m_new : 0.0;

  const double eip =
      material_energy_flux(i + 1, delta_y, donor, mass, ei, r_old, geom);
  const double eim =
      material_energy_flux(i, delta_y, donor, mass, ei, r_old, geom);
  const double ei_ext_new = mass[i] * ei[i] + eip - eim;
  ei_new[i] = (m_new > kTiny) ? ei_ext_new / m_new : 0.0;
}

__global__ void remap_material_mass_kernel(
    const double* __restrict__ r_old,
    const double* __restrict__ mass,
    const double* __restrict__ volfrac,
    const double* __restrict__ delta_y,
    const int* __restrict__ donor,
    double* __restrict__ material_mass_new,
    const int n,
    const int n_mat,
    const int geom) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = n * n_mat;
  if (idx >= count) {
    return;
  }
  const int i = idx / n_mat;
  const int m = idx - i * n_mat;
  const double q_old = mass[i] * volfrac[idx];
  const double fp =
      volfrac_mass_flux(i + 1, m, n_mat, delta_y, donor, mass, volfrac, r_old,
                        geom);
  const double fm =
      volfrac_mass_flux(i, m, n_mat, delta_y, donor, mass, volfrac, r_old,
                        geom);
  material_mass_new[idx] = q_old + fp - fm;
}

__global__ void normalize_volfrac_kernel(double* __restrict__ volfrac_new,
                                         const int n,
                                         const int n_mat) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || n_mat <= 0) {
    return;
  }
  if (n_mat == 1) {
    volfrac_new[i] = 1.0;
    return;
  }

  double sum = 0.0;
  const int base = i * n_mat;
  for (int m = 0; m < n_mat; ++m) {
    const double material_mass = fmax(volfrac_new[base + m], 0.0);
    volfrac_new[base + m] = material_mass;
    sum += material_mass;
  }

  if (sum > kTiny) {
    for (int m = 0; m < n_mat; ++m) {
      volfrac_new[base + m] /= sum;
    }
    return;
  }

  volfrac_new[base] = 1.0;
  for (int m = 1; m < n_mat; ++m) {
    volfrac_new[base + m] = 0.0;
  }
}

__global__ void remap_radiation_kernel(const double* __restrict__ r_old,
                                       const double* __restrict__ rad_e,
                                       const double* __restrict__ vol_new,
                                       const double* __restrict__ delta_y,
                                       const int* __restrict__ donor,
                                       double* __restrict__ rad_e_new,
                                       const int n,
                                       const int n_groups,
                                       const int geom) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = n * n_groups;
  if (idx >= count) {
    return;
  }
  const int i = idx / n_groups;
  const int g = idx - i * n_groups;
  const double q_old = cell_volume_from_nodes(r_old, i, geom) * rad_e[idx];
  const double fp =
      radiation_flux(i + 1, g, n_groups, delta_y, donor, rad_e);
  const double fm = radiation_flux(i, g, n_groups, delta_y, donor, rad_e);
  const double q_new = q_old + fp - fm;
  rad_e_new[idx] = q_new / fmax(vol_new[i], kTiny);
}

template <typename InputIt>
double reduce_sum(InputIt input,
                  const int n,
                  cudaStream_t stream,
                  const char* label) {
  if (n <= 0) {
    return 0.0;
  }
  DeviceBuffer<double> out(1);
  std::size_t temp_bytes = 0;
  cuda_check(cub::DeviceReduce::Sum(nullptr, temp_bytes, input, out.data(), n,
                                    stream),
             label);
  DeviceBuffer<unsigned char> temp(temp_bytes);
  cuda_check(cub::DeviceReduce::Sum(temp.data(), temp_bytes, input, out.data(),
                                    n, stream),
             label);
  double host = 0.0;
  cuda_check(cudaMemcpyAsync(&host, out.data(), sizeof(double),
                             cudaMemcpyDeviceToHost, stream),
             label);
  cuda_check(cudaStreamSynchronize(stream), label);
  return host;
}

double relative_error(const double old_total, const double new_total) {
  const double diff = std::abs(new_total - old_total);
  const double denom = std::abs(old_total);
  return denom > kTiny ? diff / denom : diff;
}

struct MassEnergyOp {
  const double* mass = nullptr;
  const double* e = nullptr;

  __host__ __device__ double operator()(const int i) const {
    return mass[i] * e[i];
  }
};

struct MaterialMassOp {
  const double* mass = nullptr;
  const double* volfrac = nullptr;
  int n_mat = 0;
  int mat = 0;

  __host__ __device__ double operator()(const int i) const {
    return mass[i] * volfrac[i * n_mat + mat];
  }
};

struct OldRadiationEnergyOp {
  const double* r_old = nullptr;
  const double* rad_e = nullptr;
  int n_groups = 0;
  int group = 0;
  int geom = 0;

  __host__ __device__ double operator()(const int i) const {
    return cell_volume_from_nodes(r_old, i, geom) * rad_e[i * n_groups + group];
  }
};

struct NewRadiationEnergyOp {
  const double* vol_new = nullptr;
  const double* rad_e_new = nullptr;
  int n_groups = 0;
  int group = 0;

  __host__ __device__ double operator()(const int i) const {
    return vol_new[i] * rad_e_new[i * n_groups + group];
  }
};

template <typename Op>
double reduce_transformed_sum(const int n,
                              const Op& op,
                              cudaStream_t stream,
                              const char* label) {
  auto counting = core::CountingInputIterator<int>(0);
  auto transformed = core::TransformInputIterator<double, Op, decltype(counting)>(
      counting, op);
  return reduce_sum(transformed, n, stream, label);
}

Ale1dRemapResult compute_conservation(const core::State& state,
                                      const int n,
                                      const int n_groups,
                                      const int n_mat,
                                      const Ale1dRemapScratch& scratch,
                                      cudaStream_t stream,
                                      const int geom) {
  Ale1dRemapResult result;
  result.success = true;

  const double mass_old = reduce_sum(
      state.mass.data(), n, stream, "ALE1D remap old mass reduction failed");
  const double mass_new = reduce_sum(scratch.mass_new.data(), n, stream,
                                     "ALE1D remap new mass reduction failed");
  result.mass_conservation_rel_err = relative_error(mass_old, mass_new);

  const double ee_old = reduce_transformed_sum(
      n, MassEnergyOp{state.mass.data(), state.ee.data()}, stream,
      "ALE1D remap old electron energy reduction failed");
  const double ee_new = reduce_transformed_sum(
      n, MassEnergyOp{scratch.mass_new.data(), scratch.ee_new.data()}, stream,
      "ALE1D remap new electron energy reduction failed");
  result.ee_conservation_rel_err = relative_error(ee_old, ee_new);

  const double ei_old = reduce_transformed_sum(
      n, MassEnergyOp{state.mass.data(), state.ei.data()}, stream,
      "ALE1D remap old ion energy reduction failed");
  const double ei_new = reduce_transformed_sum(
      n, MassEnergyOp{scratch.mass_new.data(), scratch.ei_new.data()}, stream,
      "ALE1D remap new ion energy reduction failed");
  result.ei_conservation_rel_err = relative_error(ei_old, ei_new);

  double material_worst = 0.0;
  for (int m = 0; m < n_mat; ++m) {
    const double old_m = reduce_transformed_sum(
        n, MaterialMassOp{state.mass.data(), state.volFrac.data(), n_mat, m},
        stream, "ALE1D remap old material mass reduction failed");
    const double new_m = reduce_transformed_sum(
        n,
        MaterialMassOp{scratch.mass_new.data(), scratch.volFrac_new.data(),
                       n_mat, m},
        stream, "ALE1D remap new material mass reduction failed");
    material_worst = std::max(material_worst, relative_error(old_m, new_m));
  }
  result.material_mass_conservation_rel_err = material_worst;

  double radiation_worst = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    const double old_g = reduce_transformed_sum(
        n, OldRadiationEnergyOp{state.x_r.data(), state.rad_E.data(), n_groups,
                                g, geom},
        stream, "ALE1D remap old radiation energy reduction failed");
    const double new_g = reduce_transformed_sum(
        n,
        NewRadiationEnergyOp{scratch.vol_new.data(), scratch.rad_E_new.data(),
                             n_groups, g},
        stream, "ALE1D remap new radiation energy reduction failed");
    radiation_worst = std::max(radiation_worst, relative_error(old_g, new_g));
  }
  result.radiation_conservation_rel_err = radiation_worst;
  return result;
}

int download_count(DeviceBuffer<int>& d_count,
                   cudaStream_t stream,
                   const char* label) {
  int count = 0;
  cuda_check(cudaMemcpyAsync(&count,
                             d_count.data(),
                             sizeof(int),
                             cudaMemcpyDeviceToHost,
                             stream),
             label);
  cuda_check(cudaStreamSynchronize(stream), label);
  return count;
}

void compute_slopes_for_field(const double* q,
                              const double* r_old,
                              const std::uint8_t* protected_face,
                              double* slope,
                              const int n,
                              const double theta,
                              const bool high_order,
                              cudaStream_t stream,
                              const int geom) {
  if (!high_order) {
    cuda_check(cudaMemsetAsync(slope,
                               0,
                               static_cast<std::size_t>(n) * sizeof(double),
                               stream),
               "ALE1D remap slope memset failed");
    return;
  }
  compute_limited_slopes_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      q, r_old, protected_face, slope, n, theta, geom);
  cuda_check(cudaGetLastError(), "ALE1D remap slope kernel launch failed");
}

int validate_extensive_field(const double* q_old,
                             const double* vol_new,
                             const double* q_ext_new,
                             int* fallback_flags,
                             DeviceBuffer<int>& fail_count,
                             const int n,
                             const int fallback_bit,
                             const bool record_fallback,
                             cudaStream_t stream) {
  cuda_check(cudaMemsetAsync(fail_count.data(), 0, sizeof(int), stream),
             "ALE1D remap bounds count memset failed");
  validate_extensive_density_bounds_kernel<<<blocks_for(n), kBlockSize, 0,
                                             stream>>>(
      q_old,
      vol_new,
      q_ext_new,
      fallback_flags,
      fail_count.data(),
      n,
      fallback_bit,
      record_fallback);
  cuda_check(cudaGetLastError(), "ALE1D remap bounds kernel launch failed");
  return download_count(fail_count, stream,
                        "ALE1D remap bounds count download failed");
}

int validate_density_field(const double* q_old,
                           const double* q_new,
                           int* fallback_flags,
                           DeviceBuffer<int>& fail_count,
                           const int n,
                           const int fallback_bit,
                           const bool record_fallback,
                           cudaStream_t stream) {
  cuda_check(cudaMemsetAsync(fail_count.data(), 0, sizeof(int), stream),
             "ALE1D remap bounds count memset failed");
  validate_density_bounds_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      q_old,
      q_new,
      fallback_flags,
      fail_count.data(),
      n,
      fallback_bit,
      record_fallback);
  cuda_check(cudaGetLastError(), "ALE1D remap bounds kernel launch failed");
  return download_count(fail_count, stream,
                        "ALE1D remap bounds count download failed");
}

int validate_specific_field(const double* q_old,
                            const double* mass_new,
                            const double* q_ext_new,
                            int* fallback_flags,
                            DeviceBuffer<int>& fail_count,
                            const int n,
                            const int fallback_bit,
                            const bool record_fallback,
                            cudaStream_t stream) {
  cuda_check(cudaMemsetAsync(fail_count.data(), 0, sizeof(int), stream),
             "ALE1D remap bounds count memset failed");
  validate_specific_bounds_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      q_old,
      mass_new,
      q_ext_new,
      fallback_flags,
      fail_count.data(),
      n,
      fallback_bit,
      record_fallback);
  cuda_check(cudaGetLastError(),
             "ALE1D remap specific bounds kernel launch failed");
  return download_count(fail_count, stream,
                        "ALE1D remap bounds count download failed");
}

struct FieldRemapStatus {
  bool success = true;
  bool used_fallback = false;
  int fallback_cells = 0;
};

FieldRemapStatus remap_extensive_field_with_fallback(
    const double* r_old,
    const double* vol_new,
    const double* delta_y,
    const int* donor,
    const double* phi_face,
    const double* zero_phi,
    const std::uint8_t* protected_face,
    const double* q_old,
    double* q_ext_new,
    double* slope,
    DeviceBuffer<int>& fail_count,
    int* fallback_flags,
    const int n,
    const double theta,
    const bool high_order,
    const bool fallback_enabled,
    const int fallback_bit,
    cudaStream_t stream,
    const int geom) {
  FieldRemapStatus status;
  const double* active_phi = high_order ? phi_face : zero_phi;
  compute_slopes_for_field(
      q_old, r_old, protected_face, slope, n, theta, high_order, stream, geom);
  remap_density_to_extensive_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      r_old, q_old, delta_y, donor, active_phi, slope, q_ext_new, n, geom);
  cuda_check(cudaGetLastError(), "ALE1D remap extensive kernel launch failed");

  if (!high_order) {
    return status;
  }

  const int failed = validate_extensive_field(q_old,
                                              vol_new,
                                              q_ext_new,
                                              fallback_flags,
                                              fail_count,
                                              n,
                                              fallback_bit,
                                              true,
                                              stream);
  if (failed == 0) {
    return status;
  }
  status.used_fallback = true;
  status.fallback_cells = failed;
  if (!fallback_enabled) {
    status.success = false;
    return status;
  }

  compute_slopes_for_field(
      q_old, r_old, protected_face, slope, n, theta, false, stream, geom);
  remap_density_to_extensive_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      r_old, q_old, delta_y, donor, zero_phi, slope, q_ext_new, n, geom);
  cuda_check(cudaGetLastError(),
             "ALE1D remap fallback extensive kernel launch failed");
  const int fallback_failed = validate_extensive_field(q_old,
                                                       vol_new,
                                                       q_ext_new,
                                                       nullptr,
                                                       fail_count,
                                                       n,
                                                       fallback_bit,
                                                       false,
                                                       stream);
  status.success = fallback_failed == 0;
  return status;
}

FieldRemapStatus remap_density_field_with_fallback(
    const double* r_old,
    const double* vol_new,
    const double* delta_y,
    const int* donor,
    const double* phi_face,
    const double* zero_phi,
    const std::uint8_t* protected_face,
    const double* q_old,
    double* q_new,
    double* slope,
    DeviceBuffer<int>& fail_count,
    int* fallback_flags,
    const int n,
    const double theta,
    const bool high_order,
    const bool fallback_enabled,
    const int fallback_bit,
    cudaStream_t stream,
    const int geom) {
  FieldRemapStatus status;
  const double* active_phi = high_order ? phi_face : zero_phi;
  compute_slopes_for_field(
      q_old, r_old, protected_face, slope, n, theta, high_order, stream, geom);
  remap_density_to_density_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      r_old, vol_new, q_old, delta_y, donor, active_phi, slope, q_new, n, geom);
  cuda_check(cudaGetLastError(), "ALE1D remap density kernel launch failed");

  if (!high_order) {
    return status;
  }

  const int failed = validate_density_field(q_old,
                                            q_new,
                                            fallback_flags,
                                            fail_count,
                                            n,
                                            fallback_bit,
                                            true,
                                            stream);
  if (failed == 0) {
    return status;
  }
  status.used_fallback = true;
  status.fallback_cells = failed;
  if (!fallback_enabled) {
    status.success = false;
    return status;
  }

  compute_slopes_for_field(
      q_old, r_old, protected_face, slope, n, theta, false, stream, geom);
  remap_density_to_density_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      r_old, vol_new, q_old, delta_y, donor, zero_phi, slope, q_new, n, geom);
  cuda_check(cudaGetLastError(),
             "ALE1D remap fallback density kernel launch failed");
  const int fallback_failed = validate_density_field(q_old,
                                                     q_new,
                                                     nullptr,
                                                     fail_count,
                                                     n,
                                                     fallback_bit,
                                                     false,
                                                     stream);
  status.success = fallback_failed == 0;
  return status;
}

FieldRemapStatus remap_specific_field_with_fallback(
    const double* r_old,
    const double* delta_y,
    const int* donor,
    const double* phi_face,
    const double* zero_phi,
    const std::uint8_t* protected_face,
    const double* q_density_old,
    const double* q_specific_old,
    const double* mass_new,
    double* q_ext_new,
    double* slope,
    DeviceBuffer<int>& fail_count,
    int* fallback_flags,
    const int n,
    const double theta,
    const bool high_order,
    const bool fallback_enabled,
    const int fallback_bit,
    cudaStream_t stream,
    const int geom) {
  FieldRemapStatus status;
  const double* active_phi = high_order ? phi_face : zero_phi;
  compute_slopes_for_field(q_density_old,
                           r_old,
                           protected_face,
                           slope,
                           n,
                           theta,
                           high_order,
                           stream,
                           geom);
  remap_density_to_extensive_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      r_old, q_density_old, delta_y, donor, active_phi, slope, q_ext_new, n,
      geom);
  cuda_check(cudaGetLastError(),
             "ALE1D remap specific extensive kernel launch failed");

  if (!high_order) {
    return status;
  }

  const int failed = validate_specific_field(q_specific_old,
                                             mass_new,
                                             q_ext_new,
                                             fallback_flags,
                                             fail_count,
                                             n,
                                             fallback_bit,
                                             true,
                                             stream);
  if (failed == 0) {
    return status;
  }
  status.used_fallback = true;
  status.fallback_cells = failed;
  if (!fallback_enabled) {
    status.success = false;
    return status;
  }

  compute_slopes_for_field(q_density_old,
                           r_old,
                           protected_face,
                           slope,
                           n,
                           theta,
                           false,
                           stream,
                           geom);
  remap_density_to_extensive_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      r_old, q_density_old, delta_y, donor, zero_phi, slope, q_ext_new, n,
      geom);
  cuda_check(cudaGetLastError(),
             "ALE1D remap fallback specific kernel launch failed");
  const int fallback_failed = validate_specific_field(q_specific_old,
                                                      mass_new,
                                                      q_ext_new,
                                                      nullptr,
                                                      fail_count,
                                                      n,
                                                      fallback_bit,
                                                      false,
                                                      stream);
  status.success = fallback_failed == 0;
  return status;
}

void accumulate_fallback(Ale1dRemapResult& result,
                         const FieldRemapStatus& status) {
  if (!status.used_fallback) {
    return;
  }
  result.n_bound_fallback_fields += 1;
  result.n_bound_fallback_cells += status.fallback_cells;
}

}  // namespace

void Ale1dRemapScratch::resize(const int n_cells,
                               const int n_groups,
                               const int n_materials) {
  TENRYU_ASSERT(n_cells >= 0, "ALE1D remap scratch n_cells must be nonnegative");
  TENRYU_ASSERT(n_groups >= 0, "ALE1D remap scratch n_groups must be nonnegative");
  TENRYU_ASSERT(n_materials >= 0,
                "ALE1D remap scratch n_materials must be nonnegative");
  const auto n = static_cast<std::size_t>(n_cells);
  mass_new.resize(n);
  ee_new.resize(n);
  ei_new.resize(n);
  rad_E_new.resize(n * static_cast<std::size_t>(n_groups));
  volFrac_new.resize(n * static_cast<std::size_t>(n_materials));
  vol_new.resize(n);
  delta_Y.resize(n + 1U);
  phi_face.resize(n + 1U);
  donor.resize(n + 1U);
  fallback_flags.resize(n);
}

bool Ale1dRemapScratch::size_matches(const int n_cells,
                                     const int n_groups,
                                     const int n_materials) const {
  if (n_cells < 0 || n_groups < 0 || n_materials < 0) {
    return false;
  }
  const auto n = static_cast<std::size_t>(n_cells);
  return mass_new.size() == n && ee_new.size() == n && ei_new.size() == n &&
         rad_E_new.size() == n * static_cast<std::size_t>(n_groups) &&
         volFrac_new.size() == n * static_cast<std::size_t>(n_materials) &&
         vol_new.size() == n && delta_Y.size() == n + 1U &&
         phi_face.size() == n + 1U && donor.size() == n + 1U &&
         fallback_flags.size() == n;
}

Ale1dRemapResult remap_v3(const core::State& state,
                          const core::Config& cfg,
                          const std::vector<double>& r_candidate,
                          const NodeConstraintMask& node_mask,
                          const std::vector<int>& additional_protected_faces,
                          Ale1dRemapScratch& scratch) {
  Ale1dRemapResult result;

  if (cfg.main.dimension != "1D_SPH" || cfg.main.dim != 1 ||
      state.mesh.dim != 1) {
    result.skip_reason = Ale1dSkipReason::WrongGeometry;
    return result;
  }

  const int n = effective_cell_count(state, cfg);
  if (n <= 0) {
    result.skip_reason = Ale1dSkipReason::NTooSmall;
    return result;
  }
  const int geom = state.mesh.geometry_code;
  if (r_candidate.size() != static_cast<std::size_t>(n + 1)) {
    result.skip_reason = Ale1dSkipReason::CandidateInvalid;
    result.n_invalid_sweeps = 1;
    return result;
  }
  TENRYU_ASSERT(node_mask.pinned.size() == static_cast<std::size_t>(n + 1),
                "ALE1D remap node mask must have n_cells+1 entries");

  for (const int face : additional_protected_faces) {
    if (face < 0 || face > n) {
      result.skip_reason = Ale1dSkipReason::CandidateInvalid;
      result.n_invalid_sweeps = 1;
      return result;
    }
  }

  const int n_groups = std::max(0, cfg.radiation.groups);
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  TENRYU_ASSERT(state.x_r.size() >= static_cast<std::size_t>(n + 1),
                "ALE1D remap requires n_cells+1 old radial nodes");
  TENRYU_ASSERT(state.mass.size() >= static_cast<std::size_t>(n),
                "ALE1D remap requires mass field");
  TENRYU_ASSERT(state.ee.size() >= static_cast<std::size_t>(n),
                "ALE1D remap requires ee field");
  TENRYU_ASSERT(state.ei.size() >= static_cast<std::size_t>(n),
                "ALE1D remap requires ei field");
  TENRYU_ASSERT(state.rad_E.size() >=
                    static_cast<std::size_t>(n) * static_cast<std::size_t>(n_groups),
                "ALE1D remap requires rad_E field");
  TENRYU_ASSERT(state.volFrac.size() >=
                    static_cast<std::size_t>(n) * static_cast<std::size_t>(n_mat),
                "ALE1D remap requires volFrac field");
  TENRYU_ASSERT(scratch.size_matches(n, n_groups, n_mat),
                "ALE1D remap scratch size mismatch");

  cudaStream_t stream = nullptr;
  DeviceBuffer<double> d_r_candidate(static_cast<std::size_t>(n + 1));
  cuda_check(cudaMemcpyAsync(d_r_candidate.data(),
                             r_candidate.data(),
                             static_cast<std::size_t>(n + 1) * sizeof(double),
                             cudaMemcpyHostToDevice,
                             stream),
             "ALE1D remap r_candidate upload failed");

  std::vector<std::uint8_t> pinned(static_cast<std::size_t>(n + 1), 0U);
  std::vector<std::uint8_t> protected_face(static_cast<std::size_t>(n + 1),
                                           0U);
  protected_face.front() = 1U;
  protected_face.back() = 1U;
  for (std::size_t j = 0; j < pinned.size(); ++j) {
    pinned[j] = node_mask.pinned[j] ? 1U : 0U;
    if (node_mask.pinned[j]) {
      protected_face[j] = 1U;
    }
  }
  for (const int face : additional_protected_faces) {
    protected_face[static_cast<std::size_t>(face)] = 1U;
  }

  DeviceBuffer<std::uint8_t> d_pinned(static_cast<std::size_t>(n + 1));
  cuda_check(cudaMemcpyAsync(d_pinned.data(),
                             pinned.data(),
                             pinned.size() * sizeof(std::uint8_t),
                             cudaMemcpyHostToDevice,
                             stream),
             "ALE1D remap node mask upload failed");
  DeviceBuffer<std::uint8_t> d_protected_face(static_cast<std::size_t>(n + 1));
  cuda_check(cudaMemcpyAsync(d_protected_face.data(),
                             protected_face.data(),
                             protected_face.size() * sizeof(std::uint8_t),
                             cudaMemcpyHostToDevice,
                             stream),
             "ALE1D remap protected-face upload failed");

  DeviceBuffer<int> d_invalid_count(1);
  cuda_check(cudaMemsetAsync(d_invalid_count.data(), 0, sizeof(int), stream),
             "ALE1D remap invalid-count memset failed");

  build_faces_kernel<<<blocks_for(n + 1), kBlockSize, 0, stream>>>(
      state.x_r.data(),
      d_r_candidate.data(),
      d_pinned.data(),
      scratch.delta_Y.data(),
      scratch.donor.data(),
      d_invalid_count.data(),
      n,
      cfg.numerics.ale1d.remap.reject_multicell_sweeps,
      geom);
  cuda_check(cudaGetLastError(), "ALE1D remap face kernel launch failed");

  build_new_volumes_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      d_r_candidate.data(), scratch.vol_new.data(), d_invalid_count.data(), n,
      geom);
  cuda_check(cudaGetLastError(), "ALE1D remap volume kernel launch failed");

  const int invalid_count = download_count(
      d_invalid_count, stream, "ALE1D remap invalid-count download failed");
  result.n_invalid_sweeps = invalid_count;
  if (invalid_count > 0) {
    result.success = false;
    result.skip_reason = Ale1dSkipReason::CandidateInvalid;
    return result;
  }

  cuda_check(cudaMemsetAsync(scratch.fallback_flags.data(),
                             0,
                             static_cast<std::size_t>(n) * sizeof(int),
                             stream),
             "ALE1D remap fallback memset failed");

  const bool high_order = cfg.numerics.ale1d.remap.high_order_enabled;
  DeviceBuffer<double> d_zero_phi(static_cast<std::size_t>(n + 1));
  cuda_check(cudaMemsetAsync(d_zero_phi.data(),
                             0,
                             static_cast<std::size_t>(n + 1) * sizeof(double),
                             stream),
             "ALE1D remap zero-phi memset failed");
  DeviceBuffer<double> d_radiation_phi(static_cast<std::size_t>(n + 1));
  if (high_order) {
    build_phi_kernel<<<blocks_for(n + 1), kBlockSize, 0, stream>>>(
        d_protected_face.data(),
        scratch.phi_face.data(),
        n,
        cfg.numerics.ale1d.remap.high_order_ramp_cells);
    cuda_check(cudaGetLastError(), "ALE1D remap phi kernel launch failed");
    build_phi_kernel<<<blocks_for(n + 1), kBlockSize, 0, stream>>>(
        d_protected_face.data(),
        d_radiation_phi.data(),
        n,
        cfg.numerics.ale1d.remap.radiation_high_order_ramp_cells);
    cuda_check(cudaGetLastError(),
               "ALE1D remap radiation phi kernel launch failed");
  } else {
    cuda_check(cudaMemcpyAsync(scratch.phi_face.data(),
                               d_zero_phi.data(),
                               static_cast<std::size_t>(n + 1) * sizeof(double),
                               cudaMemcpyDeviceToDevice,
                               stream),
               "ALE1D remap zero phi copy failed");
    cuda_check(cudaMemcpyAsync(d_radiation_phi.data(),
                               d_zero_phi.data(),
                               static_cast<std::size_t>(n + 1) * sizeof(double),
                               cudaMemcpyDeviceToDevice,
                               stream),
               "ALE1D remap zero radiation phi copy failed");
  }

  DeviceBuffer<double> d_q(static_cast<std::size_t>(n));
  DeviceBuffer<double> d_slope(static_cast<std::size_t>(n));
  DeviceBuffer<int> d_fail_count(1);
  const double theta = cfg.numerics.ale1d.remap.limiter_theta;
  const bool fallback_enabled =
      cfg.numerics.ale1d.remap.fallback_to_first_order_on_bounds_fail;

  fill_mass_density_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      state.x_r.data(), state.mass.data(), d_q.data(), n, geom);
  cuda_check(cudaGetLastError(), "ALE1D remap mass density launch failed");
  const FieldRemapStatus mass_status = remap_extensive_field_with_fallback(
      state.x_r.data(),
      scratch.vol_new.data(),
      scratch.delta_Y.data(),
      scratch.donor.data(),
      scratch.phi_face.data(),
      d_zero_phi.data(),
      d_protected_face.data(),
      d_q.data(),
      scratch.mass_new.data(),
      d_slope.data(),
      d_fail_count,
      scratch.fallback_flags.data(),
      n,
      theta,
      high_order,
      fallback_enabled,
      kFallbackMass,
      stream,
      geom);
  accumulate_fallback(result, mass_status);
  if (!mass_status.success) {
    result.success = false;
    result.skip_reason = Ale1dSkipReason::ConservationRejected;
    return result;
  }

  bool mass_basis_first_order = !high_order || mass_status.used_fallback;
  const auto force_mass_first_order = [&]() {
    if (mass_basis_first_order) {
      return;
    }
    fill_mass_density_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
        state.x_r.data(), state.mass.data(), d_q.data(), n, geom);
    cuda_check(cudaGetLastError(),
               "ALE1D remap mass density fallback launch failed");
    remap_extensive_field_with_fallback(state.x_r.data(),
                                        scratch.vol_new.data(),
                                        scratch.delta_Y.data(),
                                        scratch.donor.data(),
                                        scratch.phi_face.data(),
                                        d_zero_phi.data(),
                                        d_protected_face.data(),
                                        d_q.data(),
                                        scratch.mass_new.data(),
                                        d_slope.data(),
                                        d_fail_count,
                                        scratch.fallback_flags.data(),
                                        n,
                                        theta,
                                        false,
                                        false,
                                        kFallbackMass,
                                        stream,
                                        geom);
    mass_basis_first_order = true;
  };

  FieldRemapStatus ee_status;
  FieldRemapStatus ei_status;
  const auto remap_specific_energies = [&]() {
    fill_material_energy_density_kernel<<<blocks_for(n), kBlockSize, 0,
                                           stream>>>(
        state.x_r.data(), state.mass.data(), state.ee.data(), d_q.data(), n,
        geom);
    cuda_check(cudaGetLastError(), "ALE1D remap ee density launch failed");
    ee_status = remap_specific_field_with_fallback(state.x_r.data(),
                                                   scratch.delta_Y.data(),
                                                   scratch.donor.data(),
                                                   scratch.phi_face.data(),
                                                   d_zero_phi.data(),
                                                   d_protected_face.data(),
                                                   d_q.data(),
                                                   state.ee.data(),
                                                   scratch.mass_new.data(),
                                                   scratch.ee_new.data(),
                                                   d_slope.data(),
                                                   d_fail_count,
                                                   scratch.fallback_flags.data(),
                                                   n,
                                                   theta,
                                                   high_order,
                                                   fallback_enabled,
                                                   kFallbackEe,
                                                   stream,
                                                   geom);
    if (!ee_status.success) {
      return false;
    }

    fill_material_energy_density_kernel<<<blocks_for(n), kBlockSize, 0,
                                           stream>>>(
        state.x_r.data(), state.mass.data(), state.ei.data(), d_q.data(), n,
        geom);
    cuda_check(cudaGetLastError(), "ALE1D remap ei density launch failed");
    ei_status = remap_specific_field_with_fallback(state.x_r.data(),
                                                   scratch.delta_Y.data(),
                                                   scratch.donor.data(),
                                                   scratch.phi_face.data(),
                                                   d_zero_phi.data(),
                                                   d_protected_face.data(),
                                                   d_q.data(),
                                                   state.ei.data(),
                                                   scratch.mass_new.data(),
                                                   scratch.ei_new.data(),
                                                   d_slope.data(),
                                                   d_fail_count,
                                                   scratch.fallback_flags.data(),
                                                   n,
                                                   theta,
                                                   high_order,
                                                   fallback_enabled,
                                                   kFallbackEi,
                                                   stream,
                                                   geom);
    return ei_status.success;
  };

  bool specific_energy_success = remap_specific_energies();
  if (!specific_energy_success && high_order && fallback_enabled &&
      !mass_basis_first_order) {
    force_mass_first_order();
    specific_energy_success = remap_specific_energies();
  }
  accumulate_fallback(result, ee_status);
  accumulate_fallback(result, ei_status);
  if (!specific_energy_success) {
    result.success = false;
    result.skip_reason = Ale1dSkipReason::ConservationRejected;
    return result;
  }

  extensive_to_specific_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      scratch.ee_new.data(), scratch.mass_new.data(), scratch.ee_new.data(), n);
  cuda_check(cudaGetLastError(),
             "ALE1D remap ee specific kernel launch failed");
  extensive_to_specific_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      scratch.ei_new.data(), scratch.mass_new.data(), scratch.ei_new.data(), n);
  cuda_check(cudaGetLastError(),
             "ALE1D remap ei specific kernel launch failed");

  if (n_mat > 0) {
    DeviceBuffer<double> d_material_ext(static_cast<std::size_t>(n));
    for (int m = 0; m < n_mat; ++m) {
      fill_material_mass_density_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
          state.x_r.data(),
          state.mass.data(),
          state.volFrac.data(),
          d_q.data(),
          n,
          n_mat,
          m,
          geom);
      cuda_check(cudaGetLastError(),
                 "ALE1D remap material density launch failed");
      const FieldRemapStatus mat_status = remap_extensive_field_with_fallback(
          state.x_r.data(),
          scratch.vol_new.data(),
          scratch.delta_Y.data(),
          scratch.donor.data(),
          scratch.phi_face.data(),
          d_zero_phi.data(),
          d_protected_face.data(),
          d_q.data(),
          d_material_ext.data(),
          d_slope.data(),
          d_fail_count,
          scratch.fallback_flags.data(),
          n,
          theta,
          high_order,
          fallback_enabled,
          kFallbackMaterial,
          stream,
          geom);
      accumulate_fallback(result, mat_status);
      if (!mat_status.success) {
        result.success = false;
        result.skip_reason = Ale1dSkipReason::ConservationRejected;
        return result;
      }
      scatter_material_extensive_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
          d_material_ext.data(), scratch.volFrac_new.data(), n, n_mat, m);
      cuda_check(cudaGetLastError(),
                 "ALE1D remap material scatter kernel launch failed");
    }
    normalize_volfrac_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
        scratch.volFrac_new.data(), n, n_mat);
    cuda_check(cudaGetLastError(),
               "ALE1D remap volFrac normalization kernel launch failed");
  }

  if (n_groups > 0) {
    DeviceBuffer<double> d_rad_new(static_cast<std::size_t>(n));
    for (int g = 0; g < n_groups; ++g) {
      fill_radiation_density_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
          state.rad_E.data(), d_q.data(), n, n_groups, g);
      cuda_check(cudaGetLastError(),
                 "ALE1D remap radiation density launch failed");
      const FieldRemapStatus rad_status = remap_density_field_with_fallback(
          state.x_r.data(),
          scratch.vol_new.data(),
          scratch.delta_Y.data(),
          scratch.donor.data(),
          d_radiation_phi.data(),
          d_zero_phi.data(),
          d_protected_face.data(),
          d_q.data(),
          d_rad_new.data(),
          d_slope.data(),
          d_fail_count,
          scratch.fallback_flags.data(),
          n,
          theta,
          high_order,
          fallback_enabled,
          kFallbackRadiation,
          stream,
          geom);
      accumulate_fallback(result, rad_status);
      if (!rad_status.success) {
        result.success = false;
        result.skip_reason = Ale1dSkipReason::ConservationRejected;
        return result;
      }
      scatter_radiation_density_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
          d_rad_new.data(), scratch.rad_E_new.data(), n, n_groups, g);
      cuda_check(cudaGetLastError(),
                 "ALE1D remap radiation scatter kernel launch failed");
    }
  }

  if (mass_basis_first_order && high_order) {
    cuda_check(cudaMemcpyAsync(scratch.phi_face.data(),
                               d_zero_phi.data(),
                               static_cast<std::size_t>(n + 1) * sizeof(double),
                               cudaMemcpyDeviceToDevice,
                               stream),
               "ALE1D remap mass fallback phi copy failed");
  }

  const int fallback_cells = result.n_bound_fallback_cells;
  const int fallback_fields = result.n_bound_fallback_fields;
  result = compute_conservation(state, n, n_groups, n_mat, scratch, stream, geom);
  result.skip_reason = Ale1dSkipReason::None;
  result.n_invalid_sweeps = 0;
  result.n_bound_fallback_cells = fallback_cells;
  result.n_bound_fallback_fields = fallback_fields;
  return result;
}

Ale1dRemapResult remap_first_order(const core::State& state,
                                   const core::Config& cfg,
                                   const std::vector<double>& r_candidate,
                                   const NodeConstraintMask& node_mask,
                                   Ale1dRemapScratch& scratch) {
  Ale1dRemapResult result;

  if (cfg.main.dimension != "1D_SPH" || cfg.main.dim != 1 ||
      state.mesh.dim != 1) {
    result.skip_reason = Ale1dSkipReason::WrongGeometry;
    return result;
  }

  const int n = effective_cell_count(state, cfg);
  if (n <= 0) {
    result.skip_reason = Ale1dSkipReason::NTooSmall;
    return result;
  }
  const int geom = state.mesh.geometry_code;
  if (r_candidate.size() != static_cast<std::size_t>(n + 1)) {
    result.skip_reason = Ale1dSkipReason::CandidateInvalid;
    result.n_invalid_sweeps = 1;
    return result;
  }
  TENRYU_ASSERT(node_mask.pinned.size() == static_cast<std::size_t>(n + 1),
                "ALE1D remap node mask must have n_cells+1 entries");

  const int n_groups = std::max(0, cfg.radiation.groups);
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  TENRYU_ASSERT(state.x_r.size() >= static_cast<std::size_t>(n + 1),
                "ALE1D remap requires n_cells+1 old radial nodes");
  TENRYU_ASSERT(state.mass.size() >= static_cast<std::size_t>(n),
                "ALE1D remap requires mass field");
  TENRYU_ASSERT(state.ee.size() >= static_cast<std::size_t>(n),
                "ALE1D remap requires ee field");
  TENRYU_ASSERT(state.ei.size() >= static_cast<std::size_t>(n),
                "ALE1D remap requires ei field");
  TENRYU_ASSERT(state.rad_E.size() >=
                    static_cast<std::size_t>(n) * static_cast<std::size_t>(n_groups),
                "ALE1D remap requires rad_E field");
  TENRYU_ASSERT(state.volFrac.size() >=
                    static_cast<std::size_t>(n) * static_cast<std::size_t>(n_mat),
                "ALE1D remap requires volFrac field");
  TENRYU_ASSERT(scratch.size_matches(n, n_groups, n_mat),
                "ALE1D remap scratch size mismatch");

  cudaStream_t stream = nullptr;
  DeviceBuffer<double> d_r_candidate(static_cast<std::size_t>(n + 1));
  cuda_check(cudaMemcpyAsync(d_r_candidate.data(),
                             r_candidate.data(),
                             static_cast<std::size_t>(n + 1) * sizeof(double),
                             cudaMemcpyHostToDevice,
                             stream),
             "ALE1D remap r_candidate upload failed");

  std::vector<std::uint8_t> pinned(static_cast<std::size_t>(n + 1), 0U);
  for (std::size_t j = 0; j < pinned.size(); ++j) {
    pinned[j] = node_mask.pinned[j] ? 1U : 0U;
  }
  DeviceBuffer<std::uint8_t> d_pinned(static_cast<std::size_t>(n + 1));
  cuda_check(cudaMemcpyAsync(d_pinned.data(),
                             pinned.data(),
                             pinned.size() * sizeof(std::uint8_t),
                             cudaMemcpyHostToDevice,
                             stream),
             "ALE1D remap node mask upload failed");

  DeviceBuffer<int> d_invalid_count(1);
  cuda_check(cudaMemsetAsync(d_invalid_count.data(), 0, sizeof(int), stream),
             "ALE1D remap invalid-count memset failed");

  build_faces_kernel<<<blocks_for(n + 1), kBlockSize, 0, stream>>>(
      state.x_r.data(),
      d_r_candidate.data(),
      d_pinned.data(),
      scratch.delta_Y.data(),
      scratch.donor.data(),
      d_invalid_count.data(),
      n,
      cfg.numerics.ale1d.remap.reject_multicell_sweeps,
      geom);
  cuda_check(cudaGetLastError(), "ALE1D remap face kernel launch failed");

  build_new_volumes_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      d_r_candidate.data(), scratch.vol_new.data(), d_invalid_count.data(), n,
      geom);
  cuda_check(cudaGetLastError(), "ALE1D remap volume kernel launch failed");

  int invalid_count = 0;
  cuda_check(cudaMemcpyAsync(&invalid_count,
                             d_invalid_count.data(),
                             sizeof(int),
                             cudaMemcpyDeviceToHost,
                             stream),
             "ALE1D remap invalid-count download failed");
  cuda_check(cudaStreamSynchronize(stream),
             "ALE1D remap candidate validation failed");
  result.n_invalid_sweeps = invalid_count;
  if (invalid_count > 0) {
    result.success = false;
    result.skip_reason = Ale1dSkipReason::CandidateInvalid;
    return result;
  }

  cuda_check(cudaMemsetAsync(scratch.phi_face.data(),
                             0,
                             static_cast<std::size_t>(n + 1) * sizeof(double),
                             stream),
             "ALE1D remap phi memset failed");
  cuda_check(cudaMemsetAsync(scratch.fallback_flags.data(),
                             0,
                             static_cast<std::size_t>(n) * sizeof(int),
                             stream),
             "ALE1D remap fallback memset failed");

  remap_mass_energy_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      state.x_r.data(),
      state.mass.data(),
      state.ee.data(),
      state.ei.data(),
      scratch.delta_Y.data(),
      scratch.donor.data(),
      scratch.mass_new.data(),
      scratch.ee_new.data(),
      scratch.ei_new.data(),
      n,
      geom);
  cuda_check(cudaGetLastError(), "ALE1D remap mass/energy kernel launch failed");

  if (n_mat > 0) {
    remap_material_mass_kernel<<<blocks_for(n * n_mat), kBlockSize, 0, stream>>>(
        state.x_r.data(),
        state.mass.data(),
        state.volFrac.data(),
        scratch.delta_Y.data(),
        scratch.donor.data(),
        scratch.volFrac_new.data(),
        n,
        n_mat,
        geom);
    cuda_check(cudaGetLastError(),
               "ALE1D remap material kernel launch failed");
    normalize_volfrac_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
        scratch.volFrac_new.data(), n, n_mat);
    cuda_check(cudaGetLastError(),
               "ALE1D remap volFrac normalization kernel launch failed");
  }

  if (n_groups > 0) {
    remap_radiation_kernel<<<blocks_for(n * n_groups), kBlockSize, 0, stream>>>(
        state.x_r.data(),
        state.rad_E.data(),
        scratch.vol_new.data(),
        scratch.delta_Y.data(),
        scratch.donor.data(),
        scratch.rad_E_new.data(),
        n,
        n_groups,
        geom);
    cuda_check(cudaGetLastError(),
               "ALE1D remap radiation kernel launch failed");
  }

  result = compute_conservation(state, n, n_groups, n_mat, scratch, stream, geom);
  result.skip_reason = Ale1dSkipReason::None;
  result.n_invalid_sweeps = 0;
  return result;
}

}  // namespace tenryu::hydro::ale1d
