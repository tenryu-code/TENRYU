#include "hydro/ale_1d_velocity_project.cuh"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <cub/cub.cuh>

#include "core/error.hpp"
#include "mesh/geometry_1d.cuh"

namespace tenryu::hydro::ale1d {
namespace {

constexpr int kBlockSize = 256;
constexpr double kFourPiOverThree =
    4.188790204786390984616857844372670512262892532500141094646;
constexpr double kTiny = 1.0e-300;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

int blocks_for(const int n) {
  return (n + kBlockSize - 1) / kBlockSize;
}

template <typename T>
class DeviceBuffer {
 public:
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
               "ALE1D velocity projection cudaMalloc failed");
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
      cuda_check(cudaFree(ptr_), "ALE1D velocity projection cudaFree failed");
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

__global__ void build_old_momentum_kernel(const double* __restrict__ r_old,
                                          const double* __restrict__ mass_old,
                                          const double* __restrict__ v_old,
                                          double* __restrict__ p_old,
                                          double* __restrict__ q_p,
                                          const int n,
                                          const int geom) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const double u = 0.5 * (v_old[i] + v_old[i + 1]);
  const double p = mass_old[i] * u;
  p_old[i] = p;
  q_p[i] = p / fmax(cell_volume_from_nodes(r_old, i, geom), kTiny);
}

__global__ void compute_limited_slopes_kernel(
    const double* __restrict__ q,
    const double* __restrict__ r_old,
    const double* __restrict__ phi_face,
    double* __restrict__ slope,
    const int n,
    const double theta,
    const int geom) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  slope[i] = 0.0;
  if (i <= 0 || i >= n - 1 || phi_face[i] == 0.0 ||
      phi_face[i + 1] == 0.0) {
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

__global__ void remap_momentum_kernel(const double* __restrict__ r_old,
                                      const double* __restrict__ q_p,
                                      const double* __restrict__ delta_y,
                                      const int* __restrict__ donor,
                                      const double* __restrict__ phi_face,
                                      const double* __restrict__ slope,
                                      double* __restrict__ p_new,
                                      const int n,
                                      const int geom) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const double p_old = q_p[i] * cell_volume_from_nodes(r_old, i, geom);
  const double fp =
      limited_face_flux(i + 1, r_old, delta_y, donor, phi_face, q_p, slope,
                        geom);
  const double fm =
      limited_face_flux(i, r_old, delta_y, donor, phi_face, q_p, slope, geom);
  p_new[i] = p_old + fp - fm;
}

__global__ void project_node_velocity_kernel(
    const double* __restrict__ mass_new,
    const double* __restrict__ p_new,
    double* __restrict__ u_new,
    double* __restrict__ v_new,
    const int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    u_new[idx] = (mass_new[idx] > kTiny) ? p_new[idx] / mass_new[idx] : 0.0;
  }
  __syncthreads();

  if (idx > n) {
    return;
  }
  if (idx == 0) {
    v_new[0] = 0.0;
    return;
  }
  if (idx == n) {
    v_new[n] = (mass_new[n - 1] > kTiny) ? p_new[n - 1] / mass_new[n - 1]
                                         : 0.0;
    return;
  }
  const double denom = mass_new[idx - 1] + mass_new[idx];
  v_new[idx] = (denom > kTiny) ? (p_new[idx - 1] + p_new[idx]) / denom : 0.0;
}

__global__ void kinetic_energy_kernel(const double* __restrict__ mass,
                                      const double* __restrict__ v,
                                      double* __restrict__ ke_node,
                                      const int n) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > n) {
    return;
  }
  const double left = (j > 0) ? mass[j - 1] : 0.0;
  const double right = (j < n) ? mass[j] : 0.0;
  const double m_node = 0.5 * (left + right);
  ke_node[j] = 0.5 * m_node * v[j] * v[j];
}

double reduce_sum(const double* input,
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

}  // namespace

void Ale1dVelocityProjectScratch::resize(const int n_cells) {
  TENRYU_ASSERT(n_cells >= 0,
                "ALE1D velocity projection n_cells must be nonnegative");
  const auto n = static_cast<std::size_t>(n_cells);
  p_old_cell.resize(n);
  p_new_cell.resize(n);
  u_new_cell.resize(n);
  v_new_node.resize(n + 1U);
}

bool Ale1dVelocityProjectScratch::size_matches(const int n_cells) const {
  if (n_cells < 0) {
    return false;
  }
  const auto n = static_cast<std::size_t>(n_cells);
  return p_old_cell.size() == n && p_new_cell.size() == n &&
         u_new_cell.size() == n && v_new_node.size() == n + 1U;
}

Ale1dVelocityProjectResult project_velocity(
    const core::State& state,
    const std::vector<double>& mass_new,
    const std::vector<double>& delta_Y,
    const std::vector<int>& donor,
    const std::vector<double>& phi_face,
    Ale1dVelocityProjectScratch& scratch) {
  Ale1dVelocityProjectResult result;
  const int n = static_cast<int>(mass_new.size());
  if (n <= 0 || delta_Y.size() != static_cast<std::size_t>(n + 1) ||
      donor.size() != static_cast<std::size_t>(n + 1) ||
      phi_face.size() != static_cast<std::size_t>(n + 1)) {
    return result;
  }
  const int geom = state.mesh.geometry_code;
  TENRYU_ASSERT(state.x_r.size() >= static_cast<std::size_t>(n + 1),
                "ALE1D velocity projection requires n_cells+1 old nodes");
  TENRYU_ASSERT(state.v_r.size() >= static_cast<std::size_t>(n + 1),
                "ALE1D velocity projection requires n_cells+1 velocities");
  TENRYU_ASSERT(state.mass.size() >= static_cast<std::size_t>(n),
                "ALE1D velocity projection requires old mass");
  TENRYU_ASSERT(scratch.size_matches(n),
                "ALE1D velocity projection scratch size mismatch");

  cudaStream_t stream = nullptr;
  DeviceBuffer<double> d_mass_new(static_cast<std::size_t>(n));
  DeviceBuffer<double> d_delta_y(static_cast<std::size_t>(n + 1));
  DeviceBuffer<int> d_donor(static_cast<std::size_t>(n + 1));
  DeviceBuffer<double> d_phi(static_cast<std::size_t>(n + 1));
  DeviceBuffer<double> d_q_p(static_cast<std::size_t>(n));
  DeviceBuffer<double> d_slope(static_cast<std::size_t>(n));
  DeviceBuffer<double> d_ke_old(static_cast<std::size_t>(n + 1));
  DeviceBuffer<double> d_ke_new(static_cast<std::size_t>(n + 1));

  cuda_check(cudaMemcpyAsync(d_mass_new.data(),
                             mass_new.data(),
                             static_cast<std::size_t>(n) * sizeof(double),
                             cudaMemcpyHostToDevice,
                             stream),
             "ALE1D velocity projection mass upload failed");
  cuda_check(cudaMemcpyAsync(d_delta_y.data(),
                             delta_Y.data(),
                             static_cast<std::size_t>(n + 1) * sizeof(double),
                             cudaMemcpyHostToDevice,
                             stream),
             "ALE1D velocity projection delta upload failed");
  cuda_check(cudaMemcpyAsync(d_donor.data(),
                             donor.data(),
                             static_cast<std::size_t>(n + 1) * sizeof(int),
                             cudaMemcpyHostToDevice,
                             stream),
             "ALE1D velocity projection donor upload failed");
  cuda_check(cudaMemcpyAsync(d_phi.data(),
                             phi_face.data(),
                             static_cast<std::size_t>(n + 1) * sizeof(double),
                             cudaMemcpyHostToDevice,
                             stream),
             "ALE1D velocity projection phi upload failed");

  build_old_momentum_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      state.x_r.data(),
      state.mass.data(),
      state.v_r.data(),
      scratch.p_old_cell.data(),
      d_q_p.data(),
      n,
      geom);
  cuda_check(cudaGetLastError(),
             "ALE1D velocity projection old momentum launch failed");

  compute_limited_slopes_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      d_q_p.data(), state.x_r.data(), d_phi.data(), d_slope.data(), n, 1.5,
      geom);
  cuda_check(cudaGetLastError(),
             "ALE1D velocity projection slope launch failed");

  remap_momentum_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      state.x_r.data(),
      d_q_p.data(),
      d_delta_y.data(),
      d_donor.data(),
      d_phi.data(),
      d_slope.data(),
      scratch.p_new_cell.data(),
      n,
      geom);
  cuda_check(cudaGetLastError(),
             "ALE1D velocity projection remap launch failed");

  project_node_velocity_kernel<<<blocks_for(n + 1), kBlockSize, 0, stream>>>(
      d_mass_new.data(),
      scratch.p_new_cell.data(),
      scratch.u_new_cell.data(),
      scratch.v_new_node.data(),
      n);
  cuda_check(cudaGetLastError(),
             "ALE1D velocity projection node launch failed");

  kinetic_energy_kernel<<<blocks_for(n + 1), kBlockSize, 0, stream>>>(
      state.mass.data(), state.v_r.data(), d_ke_old.data(), n);
  cuda_check(cudaGetLastError(),
             "ALE1D velocity projection old KE launch failed");
  kinetic_energy_kernel<<<blocks_for(n + 1), kBlockSize, 0, stream>>>(
      d_mass_new.data(), scratch.v_new_node.data(), d_ke_new.data(), n);
  cuda_check(cudaGetLastError(),
             "ALE1D velocity projection new KE launch failed");

  result.kinetic_energy_old = reduce_sum(
      d_ke_old.data(), n + 1, stream,
      "ALE1D velocity projection old KE reduction failed");
  result.kinetic_energy_new = reduce_sum(
      d_ke_new.data(), n + 1, stream,
      "ALE1D velocity projection new KE reduction failed");
  const double diff =
      std::abs(result.kinetic_energy_new - result.kinetic_energy_old);
  result.kinetic_energy_drift_rel =
      result.kinetic_energy_old > kTiny ? diff / result.kinetic_energy_old
                                        : diff;
  result.success = true;
  return result;
}

}  // namespace tenryu::hydro::ale1d
