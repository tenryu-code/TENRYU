#include "hydro/ale_1d_sensor.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <cub/cub.cuh>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

#include "core/error.hpp"
#include "core/fancy_iterators.cuh"

namespace tenryu::hydro::ale1d {
namespace {

constexpr int kBlockSize = 256;
constexpr double kEps = 1.0e-30;
constexpr double kTempAbsFloor = 5.0e-2;
constexpr double kLaserAbsFloor = 1.0e-99;
constexpr double kLaserRelFloor = 1.0e-30;
constexpr double kFieldRelFloor = 1.0e-12;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
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

  DeviceBuffer(DeviceBuffer&& other) noexcept {
    ptr_ = other.ptr_;
    count_ = other.count_;
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  void reset(const std::size_t count) {
    release();
    count_ = count;
    if (count_ == 0) {
      return;
    }
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)),
               "ALE1D sensor cudaMalloc failed");
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
      cuda_check(cudaFree(ptr_), "ALE1D sensor cudaFree failed");
      ptr_ = nullptr;
    }
    count_ = 0;
  }

  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

struct ComponentStats {
  double weight_sum;
  double r_sum;
  double r2_sum;
  double x_sum;
  double x2_sum;
};

struct SquareOp {
  __host__ __device__ double operator()(const double x) const {
    return x * x;
  }
};

__host__ __device__ inline double clamp01(const double x) {
  return fmin(1.0, fmax(0.0, x));
}

__host__ __device__ inline double smoothstep_value(const double x,
                                                   const double edge0,
                                                   const double edge1) {
  if (!(edge1 > edge0)) {
    return x >= edge1 ? 1.0 : 0.0;
  }
  const double t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3.0 - 2.0 * t);
}

__host__ __device__ inline double finite_or_zero(const double x) {
  return isfinite(x) ? x : 0.0;
}

__host__ __device__ inline double shell_centroid_from_nodes(const double r0,
                                                            const double r1) {
  const double r0_3 = r0 * r0 * r0;
  const double r1_3 = r1 * r1 * r1;
  if (r1_3 - r0_3 > 0.0) {
    const double r0_4 = r0_3 * r0;
    const double r1_4 = r1_3 * r1;
    return 0.75 * (r1_4 - r0_4) / (r1_3 - r0_3);
  }
  return 0.5 * (r0 + r1);
}

__device__ inline double cell_width_device(const double* __restrict__ x_r,
                                           const int n,
                                           const int i) {
  const double outer = fmax(fabs(x_r[n]), kEps);
  return fmax(x_r[i + 1] - x_r[i], 1.0e-14 * outer);
}

__device__ inline double shell_centroid_radius_device(
    const double* __restrict__ x_r,
    const int i) {
  return shell_centroid_from_nodes(x_r[i], x_r[i + 1]);
}

__device__ inline double log_safe_device(const double f,
                                         const double f_abs,
                                         const double f_rel,
                                         const double f_max) {
  return log(fmax(f, fmax(f_abs, f_rel * fmax(f_max, 0.0))));
}

__device__ inline double radial_log_gradient_device(
    const double* __restrict__ f,
    const double* __restrict__ x_r,
    const int n,
    const int i,
    const double f_abs,
    const double f_rel,
    const double f_max) {
  if (n <= 1) {
    return 0.0;
  }
  const int left = (i == 0) ? 0 : i - 1;
  const int right = (i == n - 1) ? n - 1 : i + 1;
  if (left == right) {
    return 0.0;
  }
  const double r_left = shell_centroid_radius_device(x_r, left);
  const double r_right = shell_centroid_radius_device(x_r, right);
  const double denom = fabs(r_right - r_left);
  if (!(denom > 0.0)) {
    return 0.0;
  }
  const double log_left = log_safe_device(f[left], f_abs, f_rel, f_max);
  const double log_right = log_safe_device(f[right], f_abs, f_rel, f_max);
  return fabs((log_right - log_left) / denom);
}

__global__ void laser_power_kernel(const double* __restrict__ laser_dep,
                                   const double* __restrict__ vol,
                                   double* __restrict__ power,
                                   const int n,
                                   const double dt_step) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const double dt = fmax(dt_step, kEps);
  const double v = vol[i];
  const double e = laser_dep[i];
  power[i] = (isfinite(e) && e > 0.0 && isfinite(v) && v > 0.0)
                 ? fmax(e / (v * dt), 0.0)
                 : 0.0;
}

__global__ void laser_signal_kernel(const double* __restrict__ power,
                                    const double* __restrict__ x_r,
                                    double* __restrict__ signal,
                                    const int n,
                                    const double power_max) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  if (!(power_max > 0.0)) {
    signal[i] = 0.0;
    return;
  }
  const double dr = cell_width_device(x_r, n, i);
  const double G = radial_log_gradient_device(
      power, x_r, n, i, kLaserAbsFloor, kLaserRelFloor, power_max);
  const double g = fmin(1.0, dr * G);
  const double norm = sqrt(fmax(power[i], 0.0) / (power_max + kEps));
  signal[i] = finite_or_zero(norm * (0.5 + 0.5 * g));
}

__global__ void ablation_signal_kernel(const double* __restrict__ rho,
                                       const double* __restrict__ Te,
                                       const double* __restrict__ x_r,
                                       double* __restrict__ signal,
                                       const int n,
                                       const double Te_max,
                                       const double rho_ref,
                                       const double rho_gate,
                                       const double rho_width,
                                       const double te_low,
                                       const double te_high) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  if (!(Te_max > 0.0)) {
    signal[i] = 0.0;
    return;
  }
  const double rho_ratio = rho[i] / fmax(rho_ref, kEps);
  const double H_rho = 0.5 * (1.0 + tanh((rho_ratio - rho_gate) /
                                        fmax(rho_width, kEps)));
  const double H_T = smoothstep_value(Te[i], te_low, te_high);
  const double dr = cell_width_device(x_r, n, i);
  const double G = radial_log_gradient_device(
      Te, x_r, n, i, kTempAbsFloor, kFieldRelFloor, Te_max);
  const double g = fmin(1.0, 4.0 * dr * G);
  const double norm = sqrt(fmax(Te[i], 0.0) / (Te_max + kEps));
  signal[i] = finite_or_zero(H_rho * H_T * norm * g);
}

__global__ void pressure_kernel(const double* __restrict__ Pe,
                                const double* __restrict__ Pi,
                                double* __restrict__ pressure,
                                const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  pressure[i] = fmax(finite_or_zero(Pe[i] + Pi[i]), 0.0);
}

__global__ void shock_signal_kernel(const double* __restrict__ v_r,
                                    const double* __restrict__ cs,
                                    const double* __restrict__ qvisc,
                                    const double* __restrict__ pressure,
                                    const double* __restrict__ x_r,
                                    double* __restrict__ signal,
                                    double* __restrict__ q_ratio,
                                    double* __restrict__ chi_out,
                                    const int n,
                                    const double pressure_floor) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const double dr = cell_width_device(x_r, n, i);
  const double du = v_r[i + 1] - v_r[i];
  const double chi = fmax(0.0, -du) / (fmax(fabs(cs[i]), 0.0) + kEps);
  const double q = finite_or_zero(qvisc[i]) / (pressure[i] + pressure_floor);
  q_ratio[i] = finite_or_zero(q);
  chi_out[i] = finite_or_zero(chi);
  (void)dr;
  const double q_term = q / (q + 0.05);
  const double chi_term = chi / (chi + 0.05);
  signal[i] = finite_or_zero(q_term * chi_term);
}

__global__ void interface_signal_kernel(const double* __restrict__ volfrac,
                                        double* __restrict__ face_signal,
                                        const int n,
                                        const int n_mat) {
  const int face = blockIdx.x * blockDim.x + threadIdx.x;
  if (face > n) {
    return;
  }
  if (face == 0 || face == n || n_mat <= 1) {
    face_signal[face] = 0.0;
    return;
  }
  double jump = 0.0;
  const int left_base = (face - 1) * n_mat;
  const int right_base = face * n_mat;
  for (int m = 0; m < n_mat; ++m) {
    jump += fabs(volfrac[right_base + m] - volfrac[left_base + m]);
  }
  face_signal[face] = finite_or_zero(0.5 * jump);
}

__global__ void interface_score_kernel(const double* __restrict__ face_signal,
                                       const int* __restrict__ selected_faces,
                                       double* __restrict__ score,
                                       const int n,
                                       const int n_selected,
                                       const int min_separation,
                                       const double jump_low) {
  const int face = blockIdx.x * blockDim.x + threadIdx.x;
  if (face > n) {
    return;
  }
  if (face == 0 || face == n) {
    score[face] = -1.0;
    return;
  }
  const double J = face_signal[face];
  const double left = (face > 1) ? face_signal[face - 1] : -1.0;
  const double right = (face < n - 1) ? face_signal[face + 1] : -1.0;
  bool eligible = (J >= jump_low && J >= left && J >= right);
  for (int k = 0; k < n_selected; ++k) {
    const int distance =
        (face >= selected_faces[k]) ? (face - selected_faces[k]) : (selected_faces[k] - face);
    if (distance < min_separation) {
      eligible = false;
    }
  }
  score[face] = eligible ? finite_or_zero(J) : -1.0;
}

__global__ void component_bounds_kernel(const double* __restrict__ signal,
                                        int* __restrict__ bounds,
                                        const int n,
                                        const int peak,
                                        const double threshold) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  if (signal[i] < threshold) {
    if (i < peak) {
      atomicMax(&bounds[0], i);
    } else if (i > peak) {
      atomicMin(&bounds[1], i);
    }
  }
}

// 2026-07-26 review: the component moments used to be
// accumulated with floating-point atomicAdd, whose arrival order is not
// deterministic — the rezone candidate could differ run-to-run, violating
// the 1D bitwise reproducibility contract whenever ALE v3 is enabled.
// Stage the per-cell contributions instead and reduce them with the same
// fixed-order cub::DeviceReduce path as the other sensor sums.
__global__ void component_stats_stage_kernel(
    const double* __restrict__ signal,
    const double* __restrict__ x_r,
    const double* __restrict__ mass,
    const double* __restrict__ mass_prefix,
    double* __restrict__ w_stage,
    double* __restrict__ wr_stage,
    double* __restrict__ wr2_stage,
    double* __restrict__ wx_stage,
    double* __restrict__ wx2_stage,
    const int n,
    const int left,
    const int right,
    const double threshold,
    const double total_mass) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  w_stage[i] = 0.0;
  wr_stage[i] = 0.0;
  wr2_stage[i] = 0.0;
  wx_stage[i] = 0.0;
  wx2_stage[i] = 0.0;
  if (i < left || i > right || signal[i] < threshold) {
    return;
  }
  const double w = signal[i];
  if (!(w > 0.0) || !isfinite(w)) {
    return;
  }
  const double r = shell_centroid_radius_device(x_r, i);
  double x = (static_cast<double>(i) + 0.5) / static_cast<double>(n);
  if (mass != nullptr && mass_prefix != nullptr && total_mass > 0.0) {
    x = clamp01((mass_prefix[i] + 0.5 * mass[i]) / total_mass);
  }
  w_stage[i] = w;
  wr_stage[i] = w * r;
  wr2_stage[i] = w * r * r;
  wx_stage[i] = w * x;
  wx2_stage[i] = w * x * x;
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
  cuda_check(cub::DeviceReduce::Sum(nullptr, temp_bytes, input, out.data(), n, stream),
             label);
  DeviceBuffer<unsigned char> temp(temp_bytes);
  cuda_check(cub::DeviceReduce::Sum(temp.data(), temp_bytes, input, out.data(), n, stream),
             label);
  double host = 0.0;
  cuda_check(cudaMemcpyAsync(&host, out.data(), sizeof(double),
                             cudaMemcpyDeviceToHost, stream),
             label);
  cuda_check(cudaStreamSynchronize(stream), label);
  return host;
}

template <typename InputIt>
double reduce_max(InputIt input,
                  const int n,
                  cudaStream_t stream,
                  const char* label) {
  if (n <= 0) {
    return 0.0;
  }
  DeviceBuffer<double> out(1);
  std::size_t temp_bytes = 0;
  cuda_check(cub::DeviceReduce::Max(nullptr, temp_bytes, input, out.data(), n, stream),
             label);
  DeviceBuffer<unsigned char> temp(temp_bytes);
  cuda_check(cub::DeviceReduce::Max(temp.data(), temp_bytes, input, out.data(), n, stream),
             label);
  double host = 0.0;
  cuda_check(cudaMemcpyAsync(&host, out.data(), sizeof(double),
                             cudaMemcpyDeviceToHost, stream),
             label);
  cuda_check(cudaStreamSynchronize(stream), label);
  return host;
}

cub::KeyValuePair<int, double> reduce_argmax(const double* values,
                                             const int n,
                                             cudaStream_t stream,
                                             const char* label) {
  cub::KeyValuePair<int, double> host{-1, 0.0};
  if (n <= 0) {
    return host;
  }
  DeviceBuffer<cub::KeyValuePair<int, double>> out(1);
  std::size_t temp_bytes = 0;
  cuda_check(cub::DeviceReduce::ArgMax(nullptr, temp_bytes, values, out.data(), n, stream),
             label);
  DeviceBuffer<unsigned char> temp(temp_bytes);
  cuda_check(cub::DeviceReduce::ArgMax(temp.data(), temp_bytes, values, out.data(), n, stream),
             label);
  cuda_check(cudaMemcpyAsync(&host, out.data(), sizeof(host),
                             cudaMemcpyDeviceToHost, stream),
             label);
  cuda_check(cudaStreamSynchronize(stream), label);
  return host;
}

double copy_device_double(const double* ptr, cudaStream_t stream, const char* label) {
  double value = 0.0;
  cuda_check(cudaMemcpyAsync(&value, ptr, sizeof(double),
                             cudaMemcpyDeviceToHost, stream),
             label);
  cuda_check(cudaStreamSynchronize(stream), label);
  return value;
}

int effective_cell_count(const core::State& state, const core::Config& cfg) {
  if (state.mesh.topo.n_cells > 0) {
    return state.mesh.topo.n_cells;
  }
  return cfg.mesh.nr;
}

int blocks_for(const int n) {
  return (n + kBlockSize - 1) / kBlockSize;
}

double host_cell_width(const double r0, const double r1, const double outer_radius) {
  return std::max(r1 - r0, 1.0e-14 * std::max(std::abs(outer_radius), kEps));
}

bool append_cell_feature(std::vector<Ale1dFeature>& features,
                         const FeatureKind kind,
                         const core::State& state,
                         const double* signal,
                         const double* mass_prefix,
                         const double total_mass,
                         const int n,
                         const double peak_fraction,
                         const int sigma_min_cells,
                         const int sigma_max_cells,
                         const double target_fraction,
                         const double confidence,
                         const bool pinned_face,
                         cudaStream_t stream) {
  if (!(confidence > 0.0)) {
    return false;
  }

  const auto peak = reduce_argmax(signal, n, stream, "ALE1D sensor argmax failed");
  if (peak.key < 0 || !(peak.value > 0.0)) {
    return false;
  }

  DeviceBuffer<int> bounds(2);
  const int init_bounds[2] = {-1, n};
  cuda_check(cudaMemcpyAsync(bounds.data(), init_bounds, sizeof(init_bounds),
                             cudaMemcpyHostToDevice, stream),
             "ALE1D sensor bounds init failed");
  const double threshold = peak_fraction * peak.value;
  component_bounds_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      signal, bounds.data(), n, peak.key, threshold);
  cuda_check(cudaGetLastError(), "ALE1D sensor bounds kernel launch failed");
  int host_bounds[2] = {-1, n};
  cuda_check(cudaMemcpyAsync(host_bounds, bounds.data(), sizeof(host_bounds),
                             cudaMemcpyDeviceToHost, stream),
             "ALE1D sensor bounds copy failed");
  cuda_check(cudaStreamSynchronize(stream), "ALE1D sensor bounds sync failed");

  const int left = std::max(0, host_bounds[0] + 1);
  const int right = std::min(n - 1, host_bounds[1] - 1);
  if (left > right) {
    return false;
  }

  DeviceBuffer<double> stats_stage(static_cast<std::size_t>(n) * 5U);
  double* w_stage = stats_stage.data();
  double* wr_stage = w_stage + n;
  double* wr2_stage = wr_stage + n;
  double* wx_stage = wr2_stage + n;
  double* wx2_stage = wx_stage + n;
  component_stats_stage_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      signal, state.x_r.data(), state.mass.data(), mass_prefix, w_stage,
      wr_stage, wr2_stage, wx_stage, wx2_stage, n, left, right, threshold,
      total_mass);
  cuda_check(cudaGetLastError(), "ALE1D sensor stats kernel launch failed");
  ComponentStats stats{};
  stats.weight_sum =
      reduce_sum(w_stage, n, stream, "ALE1D sensor stats weight reduce failed");
  stats.r_sum =
      reduce_sum(wr_stage, n, stream, "ALE1D sensor stats r reduce failed");
  stats.r2_sum =
      reduce_sum(wr2_stage, n, stream, "ALE1D sensor stats r2 reduce failed");
  stats.x_sum =
      reduce_sum(wx_stage, n, stream, "ALE1D sensor stats x reduce failed");
  stats.x2_sum =
      reduce_sum(wx2_stage, n, stream, "ALE1D sensor stats x2 reduce failed");

  const double x_outer = copy_device_double(
      state.x_r.data() + n, stream, "ALE1D sensor outer radius copy failed");
  const double left_node = copy_device_double(
      state.x_r.data() + left, stream, "ALE1D sensor left node copy failed");
  const double right_node = copy_device_double(
      state.x_r.data() + right + 1, stream, "ALE1D sensor right node copy failed");
  const double peak_left_node = copy_device_double(
      state.x_r.data() + peak.key, stream, "ALE1D sensor peak left node copy failed");
  const double peak_right_node = copy_device_double(
      state.x_r.data() + peak.key + 1, stream, "ALE1D sensor peak right node copy failed");
  const double dr_peak = host_cell_width(peak_left_node, peak_right_node, x_outer);
  const double r_peak = shell_centroid_from_nodes(peak_left_node, peak_right_node);

  double r_center = r_peak;
  double x_center = (static_cast<double>(peak.key) + 0.5) / static_cast<double>(n);
  double sigma_r_mom = 0.0;
  double sigma_x_mom = 0.0;
  if (stats.weight_sum > 0.0) {
    r_center = stats.r_sum / stats.weight_sum;
    x_center = clamp01(stats.x_sum / stats.weight_sum);
    sigma_r_mom = std::sqrt(std::max(0.0, stats.r2_sum / stats.weight_sum -
                                              r_center * r_center));
    sigma_x_mom = std::sqrt(std::max(0.0, stats.x2_sum / stats.weight_sum -
                                              x_center * x_center));
  }

  const double component_span = std::max(0.0, right_node - left_node);
  const double sigma_raw = std::max(sigma_r_mom, 0.5 * component_span);
  const double sigma_min = std::max(0, sigma_min_cells) * dr_peak;
  const double sigma_max = std::max(sigma_min_cells, sigma_max_cells) * dr_peak;

  Ale1dFeature feature;
  feature.kind = kind;
  feature.x_center = x_center;
  feature.r_center = r_center;
  feature.sigma_r = std::clamp(sigma_raw, sigma_min, sigma_max);
  const double sigma_x_min =
      std::max(1.0, static_cast<double>(sigma_min_cells)) / static_cast<double>(n);
  const double sigma_x_max =
      std::max(sigma_x_min,
               std::max(1.0, static_cast<double>(sigma_max_cells)) /
                   static_cast<double>(n));
  feature.sigma_x = std::clamp(sigma_x_mom, sigma_x_min, sigma_x_max);
  feature.confidence = clamp01(confidence);
  feature.target_cells = std::max(0.0, target_fraction) * static_cast<double>(n);
  feature.peak_cell_or_face = peak.key;
  feature.pinned_face = pinned_face;
  features.push_back(feature);
  return true;
}

void append_interface_features(std::vector<Ale1dFeature>& features,
                               const core::State& state,
                               const core::Config& cfg,
                               const double* mass_prefix,
                               const double total_mass,
                               const int n,
                               cudaStream_t stream) {
  const auto& sensor = cfg.numerics.ale1d.interface_sensor;
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  if (!sensor.enabled || n <= 1 || n_mat <= 1 ||
      state.volFrac.size() <
          static_cast<std::size_t>(n) * static_cast<std::size_t>(n_mat)) {
    return;
  }

  const int max_features = std::max(0, sensor.max_features);
  if (max_features == 0) {
    return;
  }

  DeviceBuffer<double> face_signal(static_cast<std::size_t>(n + 1));
  DeviceBuffer<double> score(static_cast<std::size_t>(n + 1));
  DeviceBuffer<int> selected(static_cast<std::size_t>(max_features));
  interface_signal_kernel<<<blocks_for(n + 1), kBlockSize, 0, stream>>>(
      state.volFrac.data(), face_signal.data(), n, n_mat);
  cuda_check(cudaGetLastError(), "ALE1D interface signal kernel launch failed");

  std::vector<int> selected_faces;
  std::vector<double> selected_jumps;
  selected_faces.reserve(static_cast<std::size_t>(max_features));
  selected_jumps.reserve(static_cast<std::size_t>(max_features));

  for (int k = 0; k < max_features; ++k) {
    if (!selected_faces.empty()) {
      cuda_check(cudaMemcpyAsync(selected.data(), selected_faces.data(),
                                 selected_faces.size() * sizeof(int),
                                 cudaMemcpyHostToDevice, stream),
                 "ALE1D interface selected-face copy failed");
    }
    interface_score_kernel<<<blocks_for(n + 1), kBlockSize, 0, stream>>>(
        face_signal.data(), selected.data(), score.data(), n,
        static_cast<int>(selected_faces.size()),
        std::max(1, sensor.min_separation_cells), sensor.jump_low);
    cuda_check(cudaGetLastError(), "ALE1D interface score kernel launch failed");

    const auto best = reduce_argmax(score.data(), n + 1, stream,
                                    "ALE1D interface argmax failed");
    if (best.key <= 0 || best.key >= n || !(best.value >= sensor.jump_low)) {
      break;
    }
    selected_faces.push_back(best.key);
    selected_jumps.push_back(best.value);
  }

  if (selected_faces.empty()) {
    return;
  }

  const double target_cap_total =
      std::max(0.0, sensor.target_cells_cap_fraction) * static_cast<double>(n);
  const double target_per_feature =
      std::min(std::max(0.0, sensor.target_cells_fraction) * static_cast<double>(n),
               target_cap_total / static_cast<double>(selected_faces.size()));
  const double x_outer = copy_device_double(
      state.x_r.data() + n, stream, "ALE1D interface outer radius copy failed");

  for (std::size_t k = 0; k < selected_faces.size(); ++k) {
    const int face = selected_faces[k];
    const double r_left = copy_device_double(
        state.x_r.data() + face - 1, stream, "ALE1D interface left radius copy failed");
    const double r_face = copy_device_double(
        state.x_r.data() + face, stream, "ALE1D interface radius copy failed");
    const double r_right = copy_device_double(
        state.x_r.data() + face + 1, stream, "ALE1D interface right radius copy failed");
    const double delta_r = std::max(0.5 * (r_right - r_left),
                                    1.0e-14 * std::max(std::abs(x_outer), kEps));

    double x_face = static_cast<double>(face) / static_cast<double>(n);
    if (mass_prefix != nullptr && total_mass > 0.0) {
      const double prefix = copy_device_double(
          mass_prefix + face, stream, "ALE1D interface mass-prefix copy failed");
      x_face = clamp01(prefix / total_mass);
    }

    Ale1dFeature feature;
    feature.kind = FeatureKind::MaterialInterface;
    feature.x_center = x_face;
    feature.r_center = r_face;
    feature.sigma_r = std::clamp(2.0 * delta_r,
                                 0.75 * std::max(0, sensor.sigma_min_cells) * delta_r,
                                 std::max(sensor.sigma_min_cells,
                                          sensor.sigma_max_cells) *
                                     delta_r);
    feature.sigma_x = std::max(1.0 / static_cast<double>(n),
                               target_per_feature / (3.0 * static_cast<double>(n)));
    feature.confidence =
        smoothstep_value(selected_jumps[k], sensor.jump_low, sensor.jump_high);
    feature.target_cells = target_per_feature;
    feature.peak_cell_or_face = face;
    feature.pinned_face = sensor.pin_interfaces;
    features.push_back(feature);
  }
}

void append_center_feature(std::vector<Ale1dFeature>& features,
                           const core::State& state,
                           const core::Config& cfg,
                           const int n,
                           cudaStream_t stream) {
  const auto& sensor = cfg.numerics.ale1d.center_sensor;
  if (!sensor.enabled || n <= 0) {
    return;
  }
  const double r0 = copy_device_double(
      state.x_r.data(), stream, "ALE1D center left radius copy failed");
  const double r1 = copy_device_double(
      state.x_r.data() + 1, stream, "ALE1D center right radius copy failed");
  const double x_outer = copy_device_double(
      state.x_r.data() + n, stream, "ALE1D center outer radius copy failed");
  const double dr0 = host_cell_width(r0, r1, x_outer);
  const double target = std::max(0.0, sensor.target_cells_fraction) *
                        static_cast<double>(n);

  Ale1dFeature feature;
  feature.kind = FeatureKind::CenterHotspot;
  feature.x_center = 0.0;
  feature.r_center = 0.0;
  feature.sigma_x = target / (3.0 * static_cast<double>(n));
  feature.sigma_r = std::clamp(8.0 * dr0,
                               std::max(0, sensor.sigma_min_cells) * dr0,
                               std::max(sensor.sigma_min_cells,
                                        sensor.sigma_max_cells) *
                                   dr0);
  feature.confidence = 1.0;
  feature.target_cells = target;
  feature.peak_cell_or_face = 0;
  feature.pinned_face = true;
  features.push_back(feature);
}

}  // namespace

std::vector<Ale1dFeature> compute_features(const core::State& state,
                                           const core::Config& cfg,
                                           const double dt_step,
                                           cudaStream_t stream) {
  std::vector<Ale1dFeature> features;
  const int n = effective_cell_count(state, cfg);
  if (n <= 0) {
    return features;
  }

  TENRYU_ASSERT(state.x_r.size() >= static_cast<std::size_t>(n + 1),
                "ALE1D sensors require n_cells+1 radial nodes");
  TENRYU_ASSERT(state.vol.size() >= static_cast<std::size_t>(n),
                "ALE1D sensors require cell volumes");

  core::CellField1D mass_prefix;
  double total_mass = 0.0;
  const double* mass_prefix_ptr = nullptr;
  if (state.mass.size() >= static_cast<std::size_t>(n)) {
    mass_prefix.reset(static_cast<std::size_t>(n));
    thrust::exclusive_scan(thrust::cuda::par.on(stream),
                           state.mass.data(), state.mass.data() + n,
                           mass_prefix.data());
    cuda_check(cudaGetLastError(), "ALE1D mass-prefix scan failed");
    total_mass = reduce_sum(state.mass.data(), n, stream,
                            "ALE1D mass sum reduction failed");
    mass_prefix_ptr = mass_prefix.data();
  }

  core::CellField1D signal;
  core::CellField1D aux1;
  core::CellField1D aux2;
  signal.reset(static_cast<std::size_t>(n));
  aux1.reset(static_cast<std::size_t>(n));
  aux2.reset(static_cast<std::size_t>(n));

  const int blocks = blocks_for(n);
  const auto& ale = cfg.numerics.ale1d;

  if (ale.laser_sensor.enabled &&
      state.laser_dep.size() >= static_cast<std::size_t>(n)) {
    laser_power_kernel<<<blocks, kBlockSize, 0, stream>>>(
        state.laser_dep.data(), state.vol.data(), aux1.data(), n, dt_step);
    cuda_check(cudaGetLastError(), "ALE1D laser power kernel launch failed");
    const double power_max =
        reduce_max(aux1.data(), n, stream, "ALE1D laser power max reduction failed");
    if (power_max > 0.0) {
      using PowerIt = double*;
      const double power_sum =
          reduce_sum(aux1.data(), n, stream, "ALE1D laser power sum reduction failed");
      auto square_it = core::TransformInputIterator<double, SquareOp, PowerIt>(
          aux1.data(), SquareOp{});
      const double power_sum_sq =
          reduce_sum(square_it, n, stream, "ALE1D laser power sumsq reduction failed");
      const double n_eff = (power_sum * power_sum) / (power_sum_sq + kEps);
      const double localization =
          1.0 - n_eff / std::max(1.0, static_cast<double>(n));
      const double confidence =
          smoothstep_value(localization,
                           ale.laser_sensor.conf_low,
                           ale.laser_sensor.conf_high);
      laser_signal_kernel<<<blocks, kBlockSize, 0, stream>>>(
          aux1.data(), state.x_r.data(), signal.data(), n, power_max);
      cuda_check(cudaGetLastError(), "ALE1D laser signal kernel launch failed");
      append_cell_feature(features, FeatureKind::LaserAbsorption, state,
                          signal.data(), mass_prefix_ptr, total_mass, n,
                          ale.laser_sensor.peak_fraction,
                          ale.laser_sensor.sigma_min_cells,
                          ale.laser_sensor.sigma_max_cells,
                          ale.laser_sensor.target_cells_fraction,
                          confidence, false, stream);
    }
  }

  if (ale.ablation_sensor.enabled &&
      state.rho.size() >= static_cast<std::size_t>(n) &&
      state.Te.size() >= static_cast<std::size_t>(n)) {
    const double Te_max =
        reduce_max(state.Te.data(), n, stream, "ALE1D Te max reduction failed");
    if (Te_max > 0.0) {
      ablation_signal_kernel<<<blocks, kBlockSize, 0, stream>>>(
          state.rho.data(), state.Te.data(), state.x_r.data(), signal.data(), n,
          Te_max, ale.ablation_sensor.reference_density_gcc,
          ale.ablation_sensor.rho_gate_frac,
          ale.ablation_sensor.rho_gate_width,
          ale.ablation_sensor.te_gate_low_eV,
          ale.ablation_sensor.te_gate_high_eV);
      cuda_check(cudaGetLastError(), "ALE1D ablation signal kernel launch failed");
      const double signal_max =
          reduce_max(signal.data(), n, stream, "ALE1D ablation signal max failed");
      const double confidence =
          smoothstep_value(signal_max,
                           ale.ablation_sensor.conf_low,
                           ale.ablation_sensor.conf_high);
      append_cell_feature(features, FeatureKind::AblationFront, state,
                          signal.data(), mass_prefix_ptr, total_mass, n,
                          ale.ablation_sensor.peak_fraction,
                          ale.ablation_sensor.sigma_min_cells,
                          ale.ablation_sensor.sigma_max_cells,
                          ale.ablation_sensor.target_cells_fraction,
                          confidence, false, stream);
    }
  }

  if (ale.shock_sensor.enabled &&
      state.v_r.size() >= static_cast<std::size_t>(n + 1) &&
      state.cs.size() >= static_cast<std::size_t>(n) &&
      state.Qvisc.size() >= static_cast<std::size_t>(n) &&
      state.Pe.size() >= static_cast<std::size_t>(n) &&
      state.Pi.size() >= static_cast<std::size_t>(n)) {
    pressure_kernel<<<blocks, kBlockSize, 0, stream>>>(
        state.Pe.data(), state.Pi.data(), aux1.data(), n);
    cuda_check(cudaGetLastError(), "ALE1D pressure kernel launch failed");
    const double pressure_max =
        reduce_max(aux1.data(), n, stream, "ALE1D pressure max reduction failed");
    const double pressure_floor = std::max(1.0e-30, 1.0e-12 * pressure_max);
    shock_signal_kernel<<<blocks, kBlockSize, 0, stream>>>(
        state.v_r.data(), state.cs.data(), state.Qvisc.data(), aux1.data(),
        state.x_r.data(), signal.data(), aux1.data(), aux2.data(), n,
        pressure_floor);
    cuda_check(cudaGetLastError(), "ALE1D shock signal kernel launch failed");
    const double q_max =
        reduce_max(aux1.data(), n, stream, "ALE1D qvisc ratio max reduction failed");
    const double chi_max =
        reduce_max(aux2.data(), n, stream, "ALE1D compression ratio max reduction failed");
    const double confidence =
        smoothstep_value(q_max,
                         ale.shock_sensor.qvisc_conf_low,
                         ale.shock_sensor.qvisc_conf_high) *
        smoothstep_value(chi_max,
                         ale.shock_sensor.du_cs_conf_low,
                         ale.shock_sensor.du_cs_conf_high);
    append_cell_feature(features, FeatureKind::Shock, state, signal.data(),
                        mass_prefix_ptr, total_mass, n,
                        ale.shock_sensor.peak_fraction,
                        ale.shock_sensor.sigma_min_cells,
                        ale.shock_sensor.sigma_max_cells,
                        ale.shock_sensor.target_cells_fraction,
                        confidence, false, stream);
  }

  append_interface_features(features, state, cfg, mass_prefix_ptr, total_mass, n, stream);
  append_center_feature(features, state, cfg, n, stream);
  return features;
}

}  // namespace tenryu::hydro::ale1d
