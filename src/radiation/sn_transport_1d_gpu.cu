#include "radiation/sn_transport_1d_gpu.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "core/error.hpp"
#include "materials/ionmix_reader.cuh"
#include "materials/ionmix_reader.hpp"
#include "materials/opacity.cuh"
#include "materials/tmat_reader.hpp"
#include "mesh/geometry_1d.cuh"
#include "radiation/group_structure.hpp"
#include "radiation/groups.cuh"
#include "radiation/lc_weights.cuh"
#include "radiation/nlte_coeffs.cuh"
#include "radiation/planck_table.cuh"
#include "radiation/sn_cyl_quadrature_1d.hpp"
#include "radiation/sn_dsa_1d_gpu.cuh"
#include "radiation/sn_material_newton_gpu.cuh"

namespace tenryu::radiation {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kFourPi = 12.566370614359172953850573533118;
constexpr double kFourPiOverThree = 4.1887902047863909846168578443727;
constexpr double kEnergyFloor = 1.0e-300;
constexpr int kBlock = 256;
constexpr int kSweepWarp = 32;

inline void cuda_check(const cudaError_t err, const char* message) {
  if (err != cudaSuccess) {
    const std::string error_message = std::string(message) + " [" +
                                      cudaGetErrorName(err) + ": " +
                                      cudaGetErrorString(err) + "]";
    TENRYU_ASSERT(false, error_message);
  }
}

enum class SnTimingBlock : int {
  OpacityEmission = 0,
  Sweep,
  Dsa,
  Residual,
  PhiToEChi,
  Newton,
  EscapedRadEOld,
  Count
};

constexpr int kSnTimingBlockCount = static_cast<int>(SnTimingBlock::Count);
constexpr std::array<const char*, kSnTimingBlockCount> kSnTimingBlockNames = {
    "opacity_emission",
    "sweep",
    "dsa",
    "residual",
    "phi_to_E_chi",
    "newton",
    "escaped_rad_E_old"};

int sn_timing_window() {
  constexpr int kDefaultWindow = 100;
  const char* value = std::getenv("TENRYU_SN_TIMING_WINDOW");
  if (value == nullptr || value[0] == '\0') {
    return kDefaultWindow;
  }
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  return (end != value && parsed > 0) ? static_cast<int>(parsed) : kDefaultWindow;
}

class SnTimingAccumulator {
 public:
  SnTimingAccumulator() {
    reset_window();
  }

  void begin_step(const int window) {
    const int positive_window = std::max(window, 1);
    if (positive_window != window_) {
      window_ = positive_window;
      reset_window();
    }
    current_gpu_ms_.fill(0.0);
    current_cpu_ms_ = 0.0;
    current_inner_iterations_ = 0;
    current_outer_iterations_ = 0;
    active_step_ = true;
  }

  void add(const SnTimingBlock block, const double gpu_ms, const double cpu_ms) {
    if (!active_step_) {
      return;
    }
    const int i = static_cast<int>(block);
    current_gpu_ms_[i] += std::max(gpu_ms, 0.0);
    current_cpu_ms_ += std::max(cpu_ms, 0.0);
  }

  void record_iterations(const int inner_iterations, const int outer_iterations) {
    if (!active_step_) {
      return;
    }
    current_inner_iterations_ = std::max(inner_iterations, 0);
    current_outer_iterations_ = std::max(outer_iterations, 0);
  }

  void end_step() {
    if (!active_step_) {
      return;
    }
    for (int i = 0; i < kSnTimingBlockCount; ++i) {
      sum_gpu_ms_[i] += current_gpu_ms_[i];
      min_gpu_ms_[i] = std::min(min_gpu_ms_[i], current_gpu_ms_[i]);
      max_gpu_ms_[i] = std::max(max_gpu_ms_[i], current_gpu_ms_[i]);
    }
    sum_cpu_ms_ += current_cpu_ms_;
    min_cpu_ms_ = std::min(min_cpu_ms_, current_cpu_ms_);
    max_cpu_ms_ = std::max(max_cpu_ms_, current_cpu_ms_);
    sum_inner_iterations_ += current_inner_iterations_;
    min_inner_iterations_ = std::min(min_inner_iterations_, current_inner_iterations_);
    max_inner_iterations_ = std::max(max_inner_iterations_, current_inner_iterations_);
    sum_outer_iterations_ += current_outer_iterations_;
    min_outer_iterations_ = std::min(min_outer_iterations_, current_outer_iterations_);
    max_outer_iterations_ = std::max(max_outer_iterations_, current_outer_iterations_);
    ++steps_;
    active_step_ = false;
    if (steps_ >= window_) {
      print_window();
      reset_window();
    }
  }

  void ensure_events() {
    if (events_ready_) {
      return;
    }
    for (int i = 0; i < kSnTimingBlockCount; ++i) {
      cuda_check(cudaEventCreate(&start_[i]), "SN timing cudaEventCreate start failed");
      cuda_check(cudaEventCreate(&stop_[i]), "SN timing cudaEventCreate stop failed");
    }
    events_ready_ = true;
  }

  cudaEvent_t start_event(const SnTimingBlock block) const {
    return start_[static_cast<int>(block)];
  }

  cudaEvent_t stop_event(const SnTimingBlock block) const {
    return stop_[static_cast<int>(block)];
  }

 private:
  void reset_window() {
    steps_ = 0;
    sum_gpu_ms_.fill(0.0);
    min_gpu_ms_.fill(std::numeric_limits<double>::infinity());
    max_gpu_ms_.fill(0.0);
    sum_cpu_ms_ = 0.0;
    min_cpu_ms_ = std::numeric_limits<double>::infinity();
    max_cpu_ms_ = 0.0;
    sum_inner_iterations_ = 0;
    min_inner_iterations_ = std::numeric_limits<int>::max();
    max_inner_iterations_ = 0;
    sum_outer_iterations_ = 0;
    min_outer_iterations_ = std::numeric_limits<int>::max();
    max_outer_iterations_ = 0;
  }

  void print_window() const {
    std::printf("[sn_timing] window=%d", steps_);
    for (int i = 0; i < kSnTimingBlockCount; ++i) {
      const double mean = (steps_ > 0) ? (sum_gpu_ms_[i] / static_cast<double>(steps_)) : 0.0;
      const double min_value = isfinite(min_gpu_ms_[i]) ? min_gpu_ms_[i] : 0.0;
      std::printf(" %s=%.6g/%.6g/%.6g ms",
                  kSnTimingBlockNames[static_cast<std::size_t>(i)],
                  min_value,
                  mean,
                  max_gpu_ms_[i]);
    }
    const double cpu_mean = (steps_ > 0) ? (sum_cpu_ms_ / static_cast<double>(steps_)) : 0.0;
    const double cpu_min = isfinite(min_cpu_ms_) ? min_cpu_ms_ : 0.0;
    const double inner_mean =
        (steps_ > 0) ? (static_cast<double>(sum_inner_iterations_) / steps_) : 0.0;
    const double outer_mean =
        (steps_ > 0) ? (static_cast<double>(sum_outer_iterations_) / steps_) : 0.0;
    const int inner_min =
        (min_inner_iterations_ == std::numeric_limits<int>::max()) ? 0 : min_inner_iterations_;
    const int outer_min =
        (min_outer_iterations_ == std::numeric_limits<int>::max()) ? 0 : min_outer_iterations_;
    std::printf(" cpu_wall=%.6g/%.6g/%.6g ms"
                " inner_iterations=%d/%.6g/%d"
                " outer_iterations=%d/%.6g/%d"
                " (min/mean/max per step)\n",
                cpu_min,
                cpu_mean,
                max_cpu_ms_,
                inner_min,
                inner_mean,
                max_inner_iterations_,
                outer_min,
                outer_mean,
                max_outer_iterations_);
    std::fflush(stdout);
  }

  int window_ = 100;
  int steps_ = 0;
  bool active_step_ = false;
  bool events_ready_ = false;
  std::array<cudaEvent_t, kSnTimingBlockCount> start_{};
  std::array<cudaEvent_t, kSnTimingBlockCount> stop_{};
  std::array<double, kSnTimingBlockCount> current_gpu_ms_{};
  std::array<double, kSnTimingBlockCount> sum_gpu_ms_{};
  std::array<double, kSnTimingBlockCount> min_gpu_ms_{};
  std::array<double, kSnTimingBlockCount> max_gpu_ms_{};
  double current_cpu_ms_ = 0.0;
  double sum_cpu_ms_ = 0.0;
  double min_cpu_ms_ = std::numeric_limits<double>::infinity();
  double max_cpu_ms_ = 0.0;
  int current_inner_iterations_ = 0;
  int current_outer_iterations_ = 0;
  long long sum_inner_iterations_ = 0;
  int min_inner_iterations_ = std::numeric_limits<int>::max();
  int max_inner_iterations_ = 0;
  long long sum_outer_iterations_ = 0;
  int min_outer_iterations_ = std::numeric_limits<int>::max();
  int max_outer_iterations_ = 0;
};

SnTimingAccumulator& sn_timing_accumulator() {
  static SnTimingAccumulator accumulator;
  return accumulator;
}

class SnTimingScope {
 public:
  SnTimingScope(SnTimingAccumulator* timing,
                const SnTimingBlock block,
                cudaStream_t stream = nullptr)
      : timing_(timing), block_(block), stream_(stream) {
    if (timing_ == nullptr) {
      return;
    }
    timing_->ensure_events();
    cpu_start_ = std::chrono::steady_clock::now();
    cuda_check(cudaEventRecord(timing_->start_event(block_), stream_),
               "SN timing cudaEventRecord start failed");
  }

  ~SnTimingScope() {
    if (timing_ == nullptr) {
      return;
    }
    cuda_check(cudaEventRecord(timing_->stop_event(block_), stream_),
               "SN timing cudaEventRecord stop failed");
    cuda_check(cudaEventSynchronize(timing_->stop_event(block_)),
               "SN timing cudaEventSynchronize failed");
    float gpu_ms = 0.0F;
    cuda_check(cudaEventElapsedTime(&gpu_ms,
                                    timing_->start_event(block_),
                                    timing_->stop_event(block_)),
               "SN timing cudaEventElapsedTime failed");
    const auto cpu_end = std::chrono::steady_clock::now();
    const double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start_).count();
    timing_->add(block_, static_cast<double>(gpu_ms), cpu_ms);
  }

 private:
  SnTimingAccumulator* timing_ = nullptr;
  SnTimingBlock block_ = SnTimingBlock::OpacityEmission;
  cudaStream_t stream_ = nullptr;
  std::chrono::steady_clock::time_point cpu_start_{};
};

__host__ __device__ inline double finite_or_zero(const double value) {
  return isfinite(value) ? value : 0.0;
}

__host__ __device__ inline double nonnegative_finite(const double value) {
  return isfinite(value) ? fmax(value, 0.0) : 0.0;
}

__device__ inline double smoothstep01(const double x) {
  const double t = fmax(fmin(finite_or_zero(x), 1.0), 0.0);
  return t * t * (3.0 - 2.0 * t);
}

__device__ inline double smooth_full_at_low(const double x,
                                            const double lo,
                                            const double hi) {
  const double width = fmax(hi - lo, kEnergyFloor);
  return 1.0 - smoothstep01((x - lo) / width);
}

__host__ __device__ inline std::size_t cg_index(const int c,
                                                const int g,
                                                const int n_groups) {
  return static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups) +
         static_cast<std::size_t>(g);
}

__host__ __device__ inline std::size_t psi_index(const int g,
                                                 const int n,
                                                 const int c,
                                                 const int n_groups,
                                                 const int n_angles) {
  return (static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups) +
          static_cast<std::size_t>(g)) *
             static_cast<std::size_t>(n_angles) +
         static_cast<std::size_t>(n);
}

__host__ __device__ inline double face_area(const double r, const int geom) {
  const double rr = fmax(finite_or_zero(r), 0.0);
  return (geom == 0) ? (kFourPi * rr * rr)
                     : tenryu::mesh::geometry_1d_face_area(geom, rr);
}

__host__ __device__ inline double shell_volume(const double r0,
                                               const double r1,
                                               const int geom) {
  const double a = fmax(finite_or_zero(r0), 0.0);
  const double b = fmax(finite_or_zero(r1), 0.0);
  return (geom == 0)
             ? fmax(kFourPiOverThree * (b * b * b - a * a * a), 0.0)
             : fmax(tenryu::mesh::geometry_1d_shell_volume_cubes(geom, a, b), 0.0);
}

void compute_gauss_legendre(const int n,
                            std::vector<double>& mu,
                            std::vector<double>& weight) {
  TENRYU_ASSERT(n > 0 && (n % 2) == 0,
                "S_N 1D requires a positive even number of angles");
  mu.assign(static_cast<std::size_t>(n), 0.0);
  weight.assign(static_cast<std::size_t>(n), 0.0);
  constexpr double tol = 1.0e-15;
  const int half = n / 2;
  for (int k = 0; k < half; ++k) {
    double x = std::cos(kPi * (static_cast<double>(k) + 0.75) /
                        (static_cast<double>(n) + 0.5));
    double p1 = 0.0;
    double p2 = 0.0;
    for (int iter = 0; iter < 64; ++iter) {
      p1 = 1.0;
      p2 = 0.0;
      for (int j = 1; j <= n; ++j) {
        const double p3 = p2;
        p2 = p1;
        p1 = ((2.0 * static_cast<double>(j) - 1.0) * x * p2 -
              (static_cast<double>(j) - 1.0) * p3) /
             static_cast<double>(j);
      }
      const double pp = static_cast<double>(n) * (x * p1 - p2) / (x * x - 1.0);
      const double dx = p1 / pp;
      x -= dx;
      if (std::abs(dx) <= tol) {
        break;
      }
    }
    p1 = 1.0;
    p2 = 0.0;
    for (int j = 1; j <= n; ++j) {
      const double p3 = p2;
      p2 = p1;
      p1 = ((2.0 * static_cast<double>(j) - 1.0) * x * p2 -
            (static_cast<double>(j) - 1.0) * p3) /
           static_cast<double>(j);
    }
    const double pp = static_cast<double>(n) * (x * p1 - p2) / (x * x - 1.0);
    const double w = 2.0 / ((1.0 - x * x) * pp * pp);
    mu[static_cast<std::size_t>(k)] = -x;
    mu[static_cast<std::size_t>(n - 1 - k)] = x;
    weight[static_cast<std::size_t>(k)] = w;
    weight[static_cast<std::size_t>(n - 1 - k)] = w;
  }
}

std::vector<double> angular_coefficients(const std::vector<double>& mu,
                                         const std::vector<double>& weight) {
  const int n_angles = static_cast<int>(mu.size());
  std::vector<double> alpha(static_cast<std::size_t>(n_angles + 1), 0.0);
  for (int n = 0; n < n_angles; ++n) {
    alpha[static_cast<std::size_t>(n + 1)] =
        alpha[static_cast<std::size_t>(n)] -
        mu[static_cast<std::size_t>(n)] * weight[static_cast<std::size_t>(n)];
  }
  alpha.front() = 0.0;
  alpha.back() = 0.0;
  for (double& value : alpha) {
    value = (std::abs(value) < 1.0e-14) ? 0.0 : std::max(value, 0.0);
  }
  return alpha;
}

struct DeviceOpacityTable {
  double* d_log_temps = nullptr;
  double* d_log_numdens = nullptr;
  double* d_kappa_PA = nullptr;
  double* d_kappa_PE = nullptr;
  double* d_kappa_R = nullptr;
  int ntemp = 0;
  int ndens = 0;
  int ngroups = 0;
  double log_T_min = 0.0;
  double log_T_max = 0.0;
  double log_ni_min = 0.0;
  double log_ni_max = 0.0;
  double T_min = 0.0;
  double T_max = 0.0;

  DeviceOpacityTable() = default;
  DeviceOpacityTable(const DeviceOpacityTable&) = delete;
  DeviceOpacityTable& operator=(const DeviceOpacityTable&) = delete;

  ~DeviceOpacityTable() {
    release();
  }

  void upload(const materials::IonmixOpacityData& host) {
    release();
    const std::size_t n_table =
        static_cast<std::size_t>(host.ngroups) *
        static_cast<std::size_t>(host.ndens) *
        static_cast<std::size_t>(host.ntemp);
    TENRYU_ASSERT(host.kappa_PA.size() == n_table &&
                      host.kappa_PE.size() == n_table &&
                      host.kappa_R.size() == n_table,
                  "SN opacity table size mismatch");
    const auto upload_vec = [](double** dst,
                               const std::vector<double>& src,
                               const char* label) {
      cuda_check(cudaMalloc(reinterpret_cast<void**>(dst),
                            sizeof(double) * src.size()),
                 label);
      cuda_check(cudaMemcpy(*dst,
                            src.data(),
                            sizeof(double) * src.size(),
                            cudaMemcpyHostToDevice),
                 label);
    };
    upload_vec(&d_log_temps, host.log_temps, "SN upload log_temps failed");
    upload_vec(&d_log_numdens, host.log_numdens, "SN upload log_numdens failed");
    upload_vec(&d_kappa_PA, host.kappa_PA, "SN upload kappa_PA failed");
    upload_vec(&d_kappa_PE, host.kappa_PE, "SN upload kappa_PE failed");
    upload_vec(&d_kappa_R, host.kappa_R, "SN upload kappa_R failed");
    ntemp = host.ntemp;
    ndens = host.ndens;
    ngroups = host.ngroups;
    log_T_min = host.log_temps.front();
    log_T_max = host.log_temps.back();
    log_ni_min = host.log_numdens.front();
    log_ni_max = host.log_numdens.back();
    T_min = host.temps_eV.front();
    T_max = host.temps_eV.back();
  }

  void release() {
    if (d_kappa_R != nullptr) {
      cuda_check(cudaFree(d_kappa_R), "SN free kappa_R failed");
      d_kappa_R = nullptr;
    }
    if (d_kappa_PE != nullptr) {
      cuda_check(cudaFree(d_kappa_PE), "SN free kappa_PE failed");
      d_kappa_PE = nullptr;
    }
    if (d_kappa_PA != nullptr) {
      cuda_check(cudaFree(d_kappa_PA), "SN free kappa_PA failed");
      d_kappa_PA = nullptr;
    }
    if (d_log_numdens != nullptr) {
      cuda_check(cudaFree(d_log_numdens), "SN free log_numdens failed");
      d_log_numdens = nullptr;
    }
    if (d_log_temps != nullptr) {
      cuda_check(cudaFree(d_log_temps), "SN free log_temps failed");
      d_log_temps = nullptr;
    }
  }

  materials::IonmixOpacityDeviceView view() const {
    materials::IonmixOpacityDeviceView out{};
    out.log_temps = d_log_temps;
    out.log_numdens = d_log_numdens;
    out.kappa_PA = d_kappa_PA;
    out.kappa_PE = d_kappa_PE;
    out.kappa_R = d_kappa_R;
    out.ntemp = ntemp;
    out.ndens = ndens;
    out.ngroups = ngroups;
    out.log_T_min = log_T_min;
    out.log_T_max = log_T_max;
    out.log_ni_min = log_ni_min;
    out.log_ni_max = log_ni_max;
    out.T_min = T_min;
    out.T_max = T_max;
    return out;
  }
};

struct NlteOpacityCache {
  std::unique_ptr<materials::IonmixOpacityData> host;
  DeviceOpacityTable device;
  std::string file;
  int groups = 0;
};

NlteOpacityCache& nlte_cache() {
  static NlteOpacityCache cache;
  return cache;
}

void ensure_nlte_table_uploaded(const core::Config& cfg,
                                const core::Config::MaterialsConfig::MatDef& mat,
                                const int n_groups) {
  auto& cache = nlte_cache();
  if (cache.host != nullptr && cache.file == mat.opacity_file &&
      cache.groups == n_groups) {
    return;
  }
  if (mat.opacity_model == "tmat") {
    const materials::TmatFile tmat = materials::load_tmat(mat.opacity_file);
    TENRYU_ASSERT(tmat.opacity.has_value(),
                  "SN tmat opacity.model requires /opacity payload");
    cache.host = std::make_unique<materials::IonmixOpacityData>(
        materials::tmat_to_ionmix_opacity(*tmat.opacity));
  } else {
    cache.host = std::make_unique<materials::IonmixOpacityData>(
        materials::load_ionmix_opacity(mat.opacity_file));
  }
  if (cfg.radiation.group_repack_hard_xray) {
    std::vector<double> target_bounds = cfg.radiation.group_bounds_eV;
    if (target_bounds.empty()) {
      const std::vector<double> range = resolve_compute_T_range_eV(cfg, false);
      target_bounds = repack_group_bounds_for_hard_xray(
          n_groups, range);
    }
    *cache.host = resample_opacity_groups_to_bounds(*cache.host, target_bounds);
  }
  TENRYU_ASSERT(cache.host->ngroups == n_groups,
                "SN NLTE opacity table group count must match Radiation.groups");
  cache.device.upload(*cache.host);
  cache.file = mat.opacity_file;
  cache.groups = n_groups;
}

__global__ void fill_kernel(double* __restrict__ values,
                            int n_values,
                            double value) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_values) {
    values[i] = value;
  }
}

__global__ void zero_sn_step_setup_kernel(
    double* __restrict__ diag_rad_E_pre,
    double* __restrict__ diag_rad_E_post,
    double* __restrict__ diag_rad_emission_at_Tn,
    double* __restrict__ diag_rad_emission_at_Tnp1,
    double* __restrict__ diag_rad_absorption,
    double* __restrict__ diag_clip_energy,
    double* __restrict__ diag_clip_full_deficit,
    double* __restrict__ diag_chi_opacity,
    double* __restrict__ diag_F_first_moment,
    double* __restrict__ E_star_flux,
    double* __restrict__ diag_E_star_flux,
    double* __restrict__ stream_theta,
    double* __restrict__ radial_fixup_count,
    double* __restrict__ radial_fixup_artificial_abs,
    double* __restrict__ angular_fixup_count,
    double* __restrict__ angular_fixup_artificial_abs,
    int n_values) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_values) {
    return;
  }
  diag_rad_E_pre[i] = 0.0;
  diag_rad_E_post[i] = 0.0;
  diag_rad_emission_at_Tn[i] = 0.0;
  diag_rad_emission_at_Tnp1[i] = 0.0;
  diag_rad_absorption[i] = 0.0;
  diag_clip_energy[i] = 0.0;
  diag_clip_full_deficit[i] = 0.0;
  diag_chi_opacity[i] = 0.0;
  diag_F_first_moment[i] = 0.0;
  E_star_flux[i] = 0.0;
  diag_E_star_flux[i] = 0.0;
  stream_theta[i] = 0.0;
  radial_fixup_count[i] = 0.0;
  radial_fixup_artificial_abs[i] = 0.0;
  angular_fixup_count[i] = 0.0;
  angular_fixup_artificial_abs[i] = 0.0;
}

__global__ void zero_sn_face_flux_buffers_kernel(
    double* __restrict__ face_flux_raw,
    double* __restrict__ face_flux_limited,
    double* __restrict__ face_flux_diff,
    double* __restrict__ face_flux_blended,
    double* __restrict__ face_alpha,
    int n_values) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_values) {
    return;
  }
  face_flux_raw[i] = 0.0;
  face_flux_limited[i] = 0.0;
  face_flux_diff[i] = 0.0;
  face_flux_blended[i] = 0.0;
  face_alpha[i] = 0.0;
}

__global__ void copy_kernel(const double* __restrict__ src,
                            double* __restrict__ dst,
                            int n_values) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_values) {
    dst[i] = finite_or_zero(src[i]);
  }
}

__global__ void build_eta_from_planck_kernel(
    const double* __restrict__ Te,
    const double* __restrict__ sigma_a,
    PlanckTableDeviceView planck,
    double* __restrict__ eta,
    int n_cells,
    int n_groups,
    double temperature_floor_eV) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int c = idx / n_groups;
  const int g = idx - c * n_groups;
  const double T = fmax(finite_or_zero(Te[c]), temperature_floor_eV);
  const double T2 = T * T;
  const double b = (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T), 0.0);
  eta[idx] = core::constants::c_light * nonnegative_finite(sigma_a[idx]) *
             core::constants::a_eV * T2 * T2 * b;
}

__global__ void initialize_phi_from_rad_E_kernel(
    const double* __restrict__ rad_E,
    double* __restrict__ phi,
    int n_total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_total) {
    phi[idx] = core::constants::c_light * nonnegative_finite(rad_E[idx]);
  }
}

__global__ void build_sweep_inputs_kernel(
    const double* __restrict__ sigma_a,
    const double* __restrict__ sigma_s,
    const double* __restrict__ eta,
    const double* __restrict__ phi_old,
    const double* __restrict__ rad_E_old,
    double* __restrict__ source_work,
    double* __restrict__ sigma_t_work,
    int n_total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_total) {
    return;
  }
  const double sig_a = nonnegative_finite(sigma_a[idx]);
  const double sig_s = nonnegative_finite(sigma_s[idx]);
  sigma_t_work[idx] = sig_a + sig_s;
  // Per-angle time-source fix: the backward-Euler time source is now per-angle
  // (inv_cdt * psi_prev[p], added inside the sweep kernels); the historic
  // isotropized 0.5*inv_cdt*c*rad_E_old term is gone from the scalar source.
  source_work[idx] =
      0.5 * nonnegative_finite(eta[idx]) +
      0.5 * sig_s * nonnegative_finite(phi_old[idx]);
}

// Persistent-angular-intensity fix: isotropic seed for the first step /
// mesh change / restart): psi_iso = phi/2 = 0.5 * c * rad_E (GL weights sum
// to 2). Step-1 sources are then bitwise-equivalent to the historic
// isotropized construction by design.
__global__ void seed_psi_prev_from_rad_E_kernel(
    double* __restrict__ psi_prev,
    const double* __restrict__ rad_E_old,
    int n_total,
    int n_angles) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_total * n_angles;
  if (idx >= total) {
    return;
  }
  const int cg = idx / n_angles;
  psi_prev[idx] = 0.5 * core::constants::c_light *
                  nonnegative_finite(rad_E_old[cg]);
}

#ifndef TENRYU_DEBUG_LANE_PARALLEL
__global__ void sn_sweep_spherical_serial_kernel(
    const double* __restrict__ source_work,
    const double* __restrict__ sigma_t_work,
    const double* __restrict__ node_r,
    const double* __restrict__ vol,
    const double* __restrict__ mu,
    const double* __restrict__ weights,
    const double* __restrict__ alpha_half,
    const double* __restrict__ psi_prev,
    double* __restrict__ psi,
    double* __restrict__ phi_sweep,
    double* __restrict__ F_sweep,
    double* __restrict__ Prr,
    double* __restrict__ face_flux_raw,
    double* __restrict__ origin,
    const double* __restrict__ outer_psi_in,  // W-B2: nullptr => vacuum
    double* __restrict__ radial_fixup_count,
    double* __restrict__ radial_fixup_artificial_abs,
    double* __restrict__ angular_fixup_count,
    double* __restrict__ angular_fixup_artificial_abs,
    int n_cells,
    int n_groups,
    int n_angles,
    double dt,
    const int geom) {
  const int g = blockIdx.x;
  if (g >= n_groups || threadIdx.x != 0) {
    return;
  }

  extern __shared__ double shared[];
  double* angular_edge = shared;
  double* inner_boundary = shared + n_cells;

  for (int c = 0; c < n_cells; ++c) {
    const std::size_t cg = cg_index(c, g, n_groups);
    phi_sweep[cg] = 0.0;
    F_sweep[cg] = 0.0;
    Prr[cg] = 0.0;
    angular_edge[c] = 0.0;
  }
  if (face_flux_raw != nullptr) {
    for (int f = 0; f <= n_cells; ++f) {
      face_flux_raw[static_cast<std::size_t>(f) *
                        static_cast<std::size_t>(n_groups) +
                    static_cast<std::size_t>(g)] = 0.0;
    }
  }
  for (int n = 0; n < n_angles; ++n) {
    inner_boundary[n] = 0.0;
    origin[static_cast<std::size_t>(g) * static_cast<std::size_t>(n_angles) +
           static_cast<std::size_t>(n)] = 0.0;
  }

  const double inv_cdt =
      (dt > 0.0) ? (1.0 / (core::constants::c_light * dt)) : 0.0;

  for (int n = 0; n < n_angles; ++n) {
    const double mu_n = finite_or_zero(mu[n]);
    if (mu_n == 0.0) {
      continue;
    }

    const double abs_mu = fabs(mu_n);
    const double alpha_prev_raw = finite_or_zero(alpha_half[n]);
    const double alpha_next_raw = finite_or_zero(alpha_half[n + 1]);
    const double w = finite_or_zero(weights[n]);
    const double mu2 = mu_n * mu_n;
    const bool lower_edge = (alpha_prev_raw == 0.0);
    const bool upper_edge = (alpha_next_raw == 0.0);

    if (mu_n < 0.0) {
      double incoming =
          (outer_psi_in != nullptr)
              ? fmax(finite_or_zero(outer_psi_in[g]), 0.0)
              : 0.0;
      for (int c = n_cells - 1; c >= 0; --c) {
        const std::size_t cg = cg_index(c, g, n_groups);
        const std::size_t p = psi_index(g, n, c, n_groups, n_angles);
        const double area_in = face_area(node_r[c], geom);
        const double area_out = face_area(node_r[c + 1], geom);
        const double V_in = nonnegative_finite(vol[c]);
        const double V =
            (V_in > 0.0) ? V_in : shell_volume(node_r[c], node_r[c + 1], geom);
        const double sigma_t = nonnegative_finite(sigma_t_work[cg]);
        const double source = nonnegative_finite(source_work[cg]) +
                              inv_cdt * nonnegative_finite(psi_prev[p]);
        const double dA = area_out - area_in;
        const double w_safe = fmax(w, 1.0e-300);
        const double ang_scale = dA / w_safe;
        const double alpha_prev = ang_scale * alpha_prev_raw;
        const double alpha_next = ang_scale * alpha_next_raw;
        double avg = 0.0;
        double edge_out = 0.0;
        if (lower_edge) {
          const double denom =
              2.0 * abs_mu * area_in + alpha_next + (sigma_t + inv_cdt) * V;
          if (denom > 0.0 && isfinite(denom)) {
            const double numer =
                V * source + abs_mu * (area_in + area_out) * incoming;
            avg = numer / denom;
          }
          avg = fmax(finite_or_zero(avg), 0.0);
          edge_out = avg;
        } else {
          const double edge_in = fmax(angular_edge[c], 0.0);
          const double denom =
              2.0 * abs_mu * area_in + 2.0 * alpha_next +
              (sigma_t + inv_cdt) * V;
          if (denom > 0.0 && isfinite(denom)) {
            const double numer =
                V * source + abs_mu * (area_in + area_out) * incoming +
                (alpha_prev + alpha_next) * edge_in;
            avg = numer / denom;
          }
          avg = fmax(finite_or_zero(avg), 0.0);
          const double d_ang = fabs(2.0 * avg - edge_in);
          edge_out = upper_edge ? avg : (2.0 * avg - edge_in);
          if (!isfinite(edge_out) || edge_out < 0.0) {
            angular_fixup_count[cg] += 1.0;
            angular_fixup_artificial_abs[cg] += d_ang;
            edge_out = 0.0;
          }
        }
        const double outgoing_predict = 2.0 * avg - incoming;
        const double d_rad = fabs(outgoing_predict);
        double outgoing = outgoing_predict;
        if (!isfinite(outgoing) || outgoing < 0.0) {
          radial_fixup_count[cg] += 1.0;
          radial_fixup_artificial_abs[cg] += d_rad;
          outgoing = 0.0;
          avg = 0.5 * incoming;
        }
        psi[p] = avg;
        phi_sweep[cg] += w * avg;
        F_sweep[cg] += w * mu_n * avg;
        if (face_flux_raw != nullptr) {
          atomicAdd(&face_flux_raw[static_cast<std::size_t>(c) *
                                       static_cast<std::size_t>(n_groups) +
                                   static_cast<std::size_t>(g)],
                    finite_or_zero(w * mu_n * outgoing));
        }
        Prr[cg] += w * mu2 * avg / core::constants::c_light;
        angular_edge[c] = edge_out;
        incoming = outgoing;
      }
      inner_boundary[n] = incoming;
      origin[static_cast<std::size_t>(g) * static_cast<std::size_t>(n_angles) +
             static_cast<std::size_t>(n)] = incoming;
    } else {
      const int reflected = n_angles - 1 - n;
      double incoming =
          (reflected >= 0 && reflected < n_angles)
              ? fmax(inner_boundary[reflected], 0.0)
              : 0.0;
      origin[static_cast<std::size_t>(g) * static_cast<std::size_t>(n_angles) +
             static_cast<std::size_t>(n)] = incoming;
      for (int c = 0; c < n_cells; ++c) {
        const std::size_t cg = cg_index(c, g, n_groups);
        const std::size_t p = psi_index(g, n, c, n_groups, n_angles);
        const double area_in = face_area(node_r[c], geom);
        const double area_out = face_area(node_r[c + 1], geom);
        const double V_in = nonnegative_finite(vol[c]);
        const double V =
            (V_in > 0.0) ? V_in : shell_volume(node_r[c], node_r[c + 1], geom);
        const double sigma_t = nonnegative_finite(sigma_t_work[cg]);
        const double source = nonnegative_finite(source_work[cg]) +
                              inv_cdt * nonnegative_finite(psi_prev[p]);
        const double dA = area_out - area_in;
        const double w_safe = fmax(w, 1.0e-300);
        const double ang_scale = dA / w_safe;
        const double alpha_prev = ang_scale * alpha_prev_raw;
        const double alpha_next = ang_scale * alpha_next_raw;
        double avg = 0.0;
        double edge_out = 0.0;
        if (lower_edge) {
          const double denom =
              2.0 * abs_mu * area_out + alpha_next + (sigma_t + inv_cdt) * V;
          if (denom > 0.0 && isfinite(denom)) {
            const double numer =
                V * source + abs_mu * (area_in + area_out) * incoming;
            avg = numer / denom;
          }
          avg = fmax(finite_or_zero(avg), 0.0);
          edge_out = avg;
        } else {
          const double edge_in = fmax(angular_edge[c], 0.0);
          const double denom =
              2.0 * abs_mu * area_out + 2.0 * alpha_next +
              (sigma_t + inv_cdt) * V;
          if (denom > 0.0 && isfinite(denom)) {
            const double numer =
                V * source + abs_mu * (area_in + area_out) * incoming +
                (alpha_prev + alpha_next) * edge_in;
            avg = numer / denom;
          }
          avg = fmax(finite_or_zero(avg), 0.0);
          const double d_ang = fabs(2.0 * avg - edge_in);
          edge_out = upper_edge ? avg : (2.0 * avg - edge_in);
          if (!isfinite(edge_out) || edge_out < 0.0) {
            angular_fixup_count[cg] += 1.0;
            angular_fixup_artificial_abs[cg] += d_ang;
            edge_out = 0.0;
          }
        }
        const double outgoing_predict = 2.0 * avg - incoming;
        const double d_rad = fabs(outgoing_predict);
        double outgoing = outgoing_predict;
        if (!isfinite(outgoing) || outgoing < 0.0) {
          radial_fixup_count[cg] += 1.0;
          radial_fixup_artificial_abs[cg] += d_rad;
          outgoing = 0.0;
          avg = 0.5 * incoming;
        }
        psi[p] = avg;
        phi_sweep[cg] += w * avg;
        F_sweep[cg] += w * mu_n * avg;
        if (face_flux_raw != nullptr) {
          atomicAdd(&face_flux_raw[static_cast<std::size_t>(c + 1) *
                                       static_cast<std::size_t>(n_groups) +
                                   static_cast<std::size_t>(g)],
                    finite_or_zero(w * mu_n * outgoing));
        }
        Prr[cg] += w * mu2 * avg / core::constants::c_light;
        angular_edge[c] = edge_out;
        incoming = outgoing;
      }
    }
  }
}

__global__ void sn_sweep_spherical_lc_kernel(
    const double* __restrict__ source_work,
    const double* __restrict__ sigma_t_work,
    const double* __restrict__ lc_E_work,
    const double* __restrict__ lc_A_work,
    const double* __restrict__ node_r,
    const double* __restrict__ vol,
    const double* __restrict__ mu,
    const double* __restrict__ weights,
    const double* __restrict__ alpha_half,
    const double* __restrict__ tau_wd,  // Weighted-diamond factors (conservative-streaming fix)
    const double* __restrict__ psi_prev,
    double* __restrict__ psi,
    double* __restrict__ psi_outgoing,
    double* __restrict__ origin,
    const double* __restrict__ outer_psi_in,  // W-B2: nullptr => vacuum
    int n_cells,
    int n_groups,
    int n_angles,
    double dt,
    const int geom) {
  const int g = blockIdx.x;
  const int lane = threadIdx.x;
  if (g >= n_groups || lane >= kSweepWarp) {
    return;
  }
  const int half_angles = n_angles / 2;
  const int diagonal_count = n_cells + half_angles - 1;

  extern __shared__ double shared[];
  double* angular_edge = shared;
  double* inner_boundary = shared + n_cells;

  for (int c = lane; c < n_cells; c += kSweepWarp) {
    angular_edge[c] = 0.0;
  }
  for (int n = lane; n < n_angles; n += kSweepWarp) {
    inner_boundary[n] = 0.0;
    origin[static_cast<std::size_t>(g) * static_cast<std::size_t>(n_angles) +
           static_cast<std::size_t>(n)] = 0.0;
  }
  __syncwarp();

  const double inv_cdt =
      (dt > 0.0) ? (1.0 / (core::constants::c_light * dt)) : 0.0;

  // Conservative-streaming fix (Miller-Alcouffe): the angular ladder is seeded with the
  // starting-direction (mu = -1) flux, obtained from the slab-geometry
  // equation along the diameter (no angular redistribution at zero angular
  // measure). Step-characteristic with the isotropic source.
  if (lane == 0) {
    double psi_sd = (outer_psi_in != nullptr)
                        ? fmax(finite_or_zero(outer_psi_in[g]), 0.0)
                        : 0.0;
    for (int c = n_cells - 1; c >= 0; --c) {
      const std::size_t cg_sd = cg_index(c, g, n_groups);
      const std::size_t p_sd = psi_index(g, 0, c, n_groups, n_angles);
      const double sigma_eff =
          nonnegative_finite(sigma_t_work[cg_sd]) + inv_cdt;
      const double q_sd = nonnegative_finite(source_work[cg_sd]) +
                          inv_cdt * nonnegative_finite(psi_prev[p_sd]);
      const double dr = fmax(node_r[c + 1] - node_r[c], 0.0);
      const double tau_opt = sigma_eff * dr;
      const double q_avg_sd = (sigma_eff > 0.0) ? (q_sd / sigma_eff) : 0.0;
      double att;
      double avg_w;
      if (tau_opt > 1.0e-12) {
        att = exp(-tau_opt);
        avg_w = (1.0 - att) / tau_opt;
      } else {
        att = 1.0 - tau_opt;
        avg_w = 1.0 - 0.5 * tau_opt;
      }
      angular_edge[c] = fmax(q_avg_sd + (psi_sd - q_avg_sd) * avg_w, 0.0);
      psi_sd = q_avg_sd + (psi_sd - q_avg_sd) * att;
    }
  }
  __syncwarp();

  const int n_neg = lane;
  double neg_incoming = 0.0;
  double neg_mu = 0.0;
  double neg_alpha_prev_raw = 0.0;
  double neg_alpha_next_raw = 0.0;
  double neg_w = 0.0;
  double neg_tau_wd = 0.5;
  bool neg_active = (n_neg < half_angles);
  if (neg_active) {
    neg_mu = finite_or_zero(mu[n_neg]);
    neg_active = (neg_mu < 0.0);
    if (neg_active) {
      neg_alpha_prev_raw = finite_or_zero(alpha_half[n_neg]);
      neg_alpha_next_raw = finite_or_zero(alpha_half[n_neg + 1]);
      neg_w = finite_or_zero(weights[n_neg]);
      neg_tau_wd = fmin(fmax(finite_or_zero(tau_wd[n_neg]), 1.0e-6), 1.0);
      // W-B2: Marshak outer boundary — half-range isotropic incoming
      // intensity 2*F_inc,g (equilibrium-exact; nullptr keeps vacuum).
      if (outer_psi_in != nullptr) {
        neg_incoming = fmax(finite_or_zero(outer_psi_in[g]), 0.0);
      }
    }
  }
  for (int d = 0; d < diagonal_count; ++d) {
    const int cp = d - lane;
    if (neg_active && cp >= 0 && cp < n_cells) {
      const int c = n_cells - 1 - cp;
      const std::size_t cg = cg_index(c, g, n_groups);
      const std::size_t p = psi_index(g, n_neg, c, n_groups, n_angles);
      const double area_in = face_area(node_r[c], geom);
      const double area_out = face_area(node_r[c + 1], geom);
      const double V_in = nonnegative_finite(vol[c]);
      const double V =
          (V_in > 0.0) ? V_in : shell_volume(node_r[c], node_r[c + 1], geom);
      const double sigma_t = nonnegative_finite(sigma_t_work[cg]);
      const double source = nonnegative_finite(source_work[cg]) +
                            inv_cdt * nonnegative_finite(psi_prev[p]);
      const double dA = area_out - area_in;
      const double w_safe = fmax(neg_w, 1.0e-300);
      const double ang_scale = dA / w_safe;
      const double alpha_prev = ang_scale * neg_alpha_prev_raw;
      const double alpha_next = ang_scale * neg_alpha_next_raw;
      const double inv_V = (V > 0.0) ? (1.0 / V) : 0.0;
      // Conservative-streaming fix: the lower angular edge carries the starting-direction seed
      // for m = 0 and the weighted-diamond ladder value afterwards.
      const double edge_in = fmax(angular_edge[c], 0.0);
      // Weighted-diamond balance: eliminating psi_{m+1/2} via the closure
      // psi_{m+1/2} = (psi_m - (1 - tau) psi_{m-1/2}) / tau puts
      // alpha_{m+1/2}/tau on the removal side and
      // (alpha_{m-1/2} + alpha_{m+1/2} (1 - tau)/tau) psi_{m-1/2} on the
      // source side. tau = 1 recovers the historic step-in-angle balance.
      const double kappa_total =
          sigma_t + inv_cdt + (alpha_next / neg_tau_wd) * inv_V;
      const double q_total =
          source +
          (alpha_prev + alpha_next * (1.0 - neg_tau_wd) / neg_tau_wd) *
              edge_in * inv_V;
      // Conservative-streaming rewrite: solve the conservative FV balance
      //   |mu| (A_dn psi_dn - A_up psi_up)/V + kappa psi_m = q_total
      // with the theta-weighted closure psi_m = theta psi_dn +
      // (1-theta) psi_up (theta precomputed into lc_A_work). For a
      // uniform isotropic field the streaming contributes
      // mu psi0 dA/V and cancels the angular divergence exactly, which
      // the previous slab-form characteristic could not do (the root of
      // the curvature-proportional center flux dip).
      const double theta_sp = finite_or_zero(lc_A_work[p]);
      const double abs_mu_n = fabs(neg_mu);
      // mu < 0: upstream face is the geometric outer face (c+1).
      const double stream_up = abs_mu_n * area_out * inv_V;
      const double stream_dn = abs_mu_n * area_in * inv_V;
      const double denom = stream_dn + kappa_total * theta_sp;
      const double outgoing =
          (denom > 0.0)
              ? fmax((q_total +
                      (stream_up - kappa_total * (1.0 - theta_sp)) *
                          neg_incoming) /
                         denom,
                     0.0)
              : 0.0;
      const double avg =
          fmax(theta_sp * outgoing + (1.0 - theta_sp) * neg_incoming, 0.0);
      // Weighted-diamond angular closure (Morel & Montry Eq. 16).
      const double edge_out =
          fmax((avg - (1.0 - neg_tau_wd) * edge_in) / neg_tau_wd, 0.0);
      psi[p] = avg;
      psi_outgoing[p] = outgoing;
      angular_edge[c] = edge_out;
      neg_incoming = outgoing;
    }
    __syncwarp();
  }
  if (neg_active) {
    inner_boundary[n_neg] = neg_incoming;
    origin[static_cast<std::size_t>(g) * static_cast<std::size_t>(n_angles) +
           static_cast<std::size_t>(n_neg)] = neg_incoming;
  }
  __syncwarp();

  const int n_pos = half_angles + lane;
  double pos_incoming = 0.0;
  double pos_mu = 0.0;
  double pos_alpha_prev_raw = 0.0;
  double pos_alpha_next_raw = 0.0;
  double pos_w = 0.0;
  double pos_tau_wd = 0.5;
  bool pos_lower_edge = false;
  bool pos_active = (n_pos < n_angles);
  if (pos_active) {
    pos_mu = finite_or_zero(mu[n_pos]);
    pos_active = (pos_mu > 0.0);
    if (pos_active) {
      pos_alpha_prev_raw = finite_or_zero(alpha_half[n_pos]);
      pos_alpha_next_raw = finite_or_zero(alpha_half[n_pos + 1]);
      pos_w = finite_or_zero(weights[n_pos]);
      pos_tau_wd = fmin(fmax(finite_or_zero(tau_wd[n_pos]), 1.0e-6), 1.0);
      pos_lower_edge = (pos_alpha_prev_raw == 0.0);
      const int reflected = n_angles - 1 - n_pos;
      pos_incoming =
          (reflected >= 0 && reflected < n_angles)
              ? fmax(inner_boundary[reflected], 0.0)
              : 0.0;
      origin[static_cast<std::size_t>(g) * static_cast<std::size_t>(n_angles) +
             static_cast<std::size_t>(n_pos)] = pos_incoming;
    }
  }
  for (int d = 0; d < diagonal_count; ++d) {
    const int c = d - lane;
    if (pos_active && c >= 0 && c < n_cells) {
      const std::size_t cg = cg_index(c, g, n_groups);
      const std::size_t p = psi_index(g, n_pos, c, n_groups, n_angles);
      const double area_in = face_area(node_r[c], geom);
      const double area_out = face_area(node_r[c + 1], geom);
      const double V_in = nonnegative_finite(vol[c]);
      const double V =
          (V_in > 0.0) ? V_in : shell_volume(node_r[c], node_r[c + 1], geom);
      const double sigma_t = nonnegative_finite(sigma_t_work[cg]);
      const double source = nonnegative_finite(source_work[cg]) +
                            inv_cdt * nonnegative_finite(psi_prev[p]);
      const double dA = area_out - area_in;
      const double w_safe = fmax(pos_w, 1.0e-300);
      const double ang_scale = dA / w_safe;
      const double alpha_prev = ang_scale * pos_alpha_prev_raw;
      const double alpha_next = ang_scale * pos_alpha_next_raw;
      const double inv_V = (V > 0.0) ? (1.0 / V) : 0.0;
      const double edge_in =
          pos_lower_edge ? 0.0 : fmax(angular_edge[c], 0.0);
      const double kappa_total =
          sigma_t + inv_cdt + (alpha_next / pos_tau_wd) * inv_V;
      const double q_total =
          source +
          (alpha_prev + alpha_next * (1.0 - pos_tau_wd) / pos_tau_wd) *
              edge_in * inv_V;
      // Conservative-streaming rewrite (see the negative branch). mu > 0:
      // upstream face is the geometric inner face (c).
      const double theta_sp = finite_or_zero(lc_A_work[p]);
      const double abs_mu_p = fabs(pos_mu);
      const double stream_up = abs_mu_p * area_in * inv_V;
      const double stream_dn = abs_mu_p * area_out * inv_V;
      const double denom = stream_dn + kappa_total * theta_sp;
      const double outgoing =
          (denom > 0.0)
              ? fmax((q_total +
                      (stream_up - kappa_total * (1.0 - theta_sp)) *
                          pos_incoming) /
                         denom,
                     0.0)
              : 0.0;
      const double avg =
          fmax(theta_sp * outgoing + (1.0 - theta_sp) * pos_incoming, 0.0);
      const double edge_out =
          fmax((avg - (1.0 - pos_tau_wd) * edge_in) / pos_tau_wd, 0.0);
      psi[p] = avg;
      psi_outgoing[p] = outgoing;
      angular_edge[c] = edge_out;
      pos_incoming = outgoing;
    }
    __syncwarp();
  }
}

// W-G3: 1D cylindrical product-quadrature serial sweep. Identical balance
// algebra to sn_sweep_spherical_serial_kernel (conservative FV streaming +
// standard alpha diamond in angle with step start at each level's first
// ordinate — the alpha == 0 level-boundary edges of the flattened product
// set fire the same lower_edge branch). The only structural difference is
// the axis reflection, taken within the ordinate's own xi-level: the mu > 0
// ordinate (l, m) takes its r = 0 incoming from (l, M-1-m) (Morel & Montry
// 1984 appendix).
__global__ void sn_sweep_cylindrical_serial_kernel(
    const double* __restrict__ source_work,
    const double* __restrict__ sigma_t_work,
    const double* __restrict__ node_r,
    const double* __restrict__ vol,
    const double* __restrict__ mu,
    const double* __restrict__ weights,
    const double* __restrict__ alpha_half,
    const double* __restrict__ psi_prev,
    double* __restrict__ psi,
    double* __restrict__ phi_sweep,
    double* __restrict__ F_sweep,
    double* __restrict__ Prr,
    double* __restrict__ face_flux_raw,
    double* __restrict__ origin,
    const double* __restrict__ outer_psi_in,  // W-B2: nullptr => vacuum
    double* __restrict__ radial_fixup_count,
    double* __restrict__ radial_fixup_artificial_abs,
    double* __restrict__ angular_fixup_count,
    double* __restrict__ angular_fixup_artificial_abs,
    int n_cells,
    int n_groups,
    int n_angles,
    int n_azim,
    double dt,
    const int geom) {
  const int g = blockIdx.x;
  if (g >= n_groups || threadIdx.x != 0) {
    return;
  }

  extern __shared__ double shared[];
  double* angular_edge = shared;
  double* inner_boundary = shared + n_cells;

  for (int c = 0; c < n_cells; ++c) {
    const std::size_t cg = cg_index(c, g, n_groups);
    phi_sweep[cg] = 0.0;
    F_sweep[cg] = 0.0;
    Prr[cg] = 0.0;
    angular_edge[c] = 0.0;
  }
  if (face_flux_raw != nullptr) {
    for (int f = 0; f <= n_cells; ++f) {
      face_flux_raw[static_cast<std::size_t>(f) *
                        static_cast<std::size_t>(n_groups) +
                    static_cast<std::size_t>(g)] = 0.0;
    }
  }
  for (int n = 0; n < n_angles; ++n) {
    inner_boundary[n] = 0.0;
    origin[static_cast<std::size_t>(g) * static_cast<std::size_t>(n_angles) +
           static_cast<std::size_t>(n)] = 0.0;
  }

  const double inv_cdt =
      (dt > 0.0) ? (1.0 / (core::constants::c_light * dt)) : 0.0;

  for (int n = 0; n < n_angles; ++n) {
    const double mu_n = finite_or_zero(mu[n]);
    if (mu_n == 0.0) {
      continue;
    }

    const double abs_mu = fabs(mu_n);
    const double alpha_prev_raw = finite_or_zero(alpha_half[n]);
    const double alpha_next_raw = finite_or_zero(alpha_half[n + 1]);
    const double w = finite_or_zero(weights[n]);
    const double mu2 = mu_n * mu_n;
    const bool lower_edge = (alpha_prev_raw == 0.0);
    const bool upper_edge = (alpha_next_raw == 0.0);

    if (mu_n < 0.0) {
      double incoming =
          (outer_psi_in != nullptr)
              ? fmax(finite_or_zero(outer_psi_in[g]), 0.0)
              : 0.0;
      for (int c = n_cells - 1; c >= 0; --c) {
        const std::size_t cg = cg_index(c, g, n_groups);
        const std::size_t p = psi_index(g, n, c, n_groups, n_angles);
        const double area_in = face_area(node_r[c], geom);
        const double area_out = face_area(node_r[c + 1], geom);
        const double V_in = nonnegative_finite(vol[c]);
        const double V =
            (V_in > 0.0) ? V_in : shell_volume(node_r[c], node_r[c + 1], geom);
        const double sigma_t = nonnegative_finite(sigma_t_work[cg]);
        const double source = nonnegative_finite(source_work[cg]) +
                              inv_cdt * nonnegative_finite(psi_prev[p]);
        const double dA = area_out - area_in;
        const double w_safe = fmax(w, 1.0e-300);
        const double ang_scale = dA / w_safe;
        const double alpha_prev = ang_scale * alpha_prev_raw;
        const double alpha_next = ang_scale * alpha_next_raw;
        double avg = 0.0;
        double edge_out = 0.0;
        if (lower_edge) {
          const double denom =
              2.0 * abs_mu * area_in + alpha_next + (sigma_t + inv_cdt) * V;
          if (denom > 0.0 && isfinite(denom)) {
            const double numer =
                V * source + abs_mu * (area_in + area_out) * incoming;
            avg = numer / denom;
          }
          avg = fmax(finite_or_zero(avg), 0.0);
          edge_out = avg;
        } else {
          const double edge_in = fmax(angular_edge[c], 0.0);
          const double denom =
              2.0 * abs_mu * area_in + 2.0 * alpha_next +
              (sigma_t + inv_cdt) * V;
          if (denom > 0.0 && isfinite(denom)) {
            const double numer =
                V * source + abs_mu * (area_in + area_out) * incoming +
                (alpha_prev + alpha_next) * edge_in;
            avg = numer / denom;
          }
          avg = fmax(finite_or_zero(avg), 0.0);
          const double d_ang = fabs(2.0 * avg - edge_in);
          edge_out = upper_edge ? avg : (2.0 * avg - edge_in);
          if (!isfinite(edge_out) || edge_out < 0.0) {
            angular_fixup_count[cg] += 1.0;
            angular_fixup_artificial_abs[cg] += d_ang;
            edge_out = 0.0;
          }
        }
        const double outgoing_predict = 2.0 * avg - incoming;
        const double d_rad = fabs(outgoing_predict);
        double outgoing = outgoing_predict;
        if (!isfinite(outgoing) || outgoing < 0.0) {
          radial_fixup_count[cg] += 1.0;
          radial_fixup_artificial_abs[cg] += d_rad;
          outgoing = 0.0;
          avg = 0.5 * incoming;
        }
        psi[p] = avg;
        phi_sweep[cg] += w * avg;
        F_sweep[cg] += w * mu_n * avg;
        if (face_flux_raw != nullptr) {
          atomicAdd(&face_flux_raw[static_cast<std::size_t>(c) *
                                       static_cast<std::size_t>(n_groups) +
                                   static_cast<std::size_t>(g)],
                    finite_or_zero(w * mu_n * outgoing));
        }
        Prr[cg] += w * mu2 * avg / core::constants::c_light;
        angular_edge[c] = edge_out;
        incoming = outgoing;
      }
      inner_boundary[n] = incoming;
      origin[static_cast<std::size_t>(g) * static_cast<std::size_t>(n_angles) +
             static_cast<std::size_t>(n)] = incoming;
    } else {
      const int level = n / n_azim;
      const int reflected =
          level * n_azim + (n_azim - 1 - (n - level * n_azim));
      double incoming =
          (reflected >= 0 && reflected < n_angles)
              ? fmax(inner_boundary[reflected], 0.0)
              : 0.0;
      origin[static_cast<std::size_t>(g) * static_cast<std::size_t>(n_angles) +
             static_cast<std::size_t>(n)] = incoming;
      for (int c = 0; c < n_cells; ++c) {
        const std::size_t cg = cg_index(c, g, n_groups);
        const std::size_t p = psi_index(g, n, c, n_groups, n_angles);
        const double area_in = face_area(node_r[c], geom);
        const double area_out = face_area(node_r[c + 1], geom);
        const double V_in = nonnegative_finite(vol[c]);
        const double V =
            (V_in > 0.0) ? V_in : shell_volume(node_r[c], node_r[c + 1], geom);
        const double sigma_t = nonnegative_finite(sigma_t_work[cg]);
        const double source = nonnegative_finite(source_work[cg]) +
                              inv_cdt * nonnegative_finite(psi_prev[p]);
        const double dA = area_out - area_in;
        const double w_safe = fmax(w, 1.0e-300);
        const double ang_scale = dA / w_safe;
        const double alpha_prev = ang_scale * alpha_prev_raw;
        const double alpha_next = ang_scale * alpha_next_raw;
        double avg = 0.0;
        double edge_out = 0.0;
        if (lower_edge) {
          const double denom =
              2.0 * abs_mu * area_out + alpha_next + (sigma_t + inv_cdt) * V;
          if (denom > 0.0 && isfinite(denom)) {
            const double numer =
                V * source + abs_mu * (area_in + area_out) * incoming;
            avg = numer / denom;
          }
          avg = fmax(finite_or_zero(avg), 0.0);
          edge_out = avg;
        } else {
          const double edge_in = fmax(angular_edge[c], 0.0);
          const double denom =
              2.0 * abs_mu * area_out + 2.0 * alpha_next +
              (sigma_t + inv_cdt) * V;
          if (denom > 0.0 && isfinite(denom)) {
            const double numer =
                V * source + abs_mu * (area_in + area_out) * incoming +
                (alpha_prev + alpha_next) * edge_in;
            avg = numer / denom;
          }
          avg = fmax(finite_or_zero(avg), 0.0);
          const double d_ang = fabs(2.0 * avg - edge_in);
          edge_out = upper_edge ? avg : (2.0 * avg - edge_in);
          if (!isfinite(edge_out) || edge_out < 0.0) {
            angular_fixup_count[cg] += 1.0;
            angular_fixup_artificial_abs[cg] += d_ang;
            edge_out = 0.0;
          }
        }
        const double outgoing_predict = 2.0 * avg - incoming;
        const double d_rad = fabs(outgoing_predict);
        double outgoing = outgoing_predict;
        if (!isfinite(outgoing) || outgoing < 0.0) {
          radial_fixup_count[cg] += 1.0;
          radial_fixup_artificial_abs[cg] += d_rad;
          outgoing = 0.0;
          avg = 0.5 * incoming;
        }
        psi[p] = avg;
        phi_sweep[cg] += w * avg;
        F_sweep[cg] += w * mu_n * avg;
        if (face_flux_raw != nullptr) {
          atomicAdd(&face_flux_raw[static_cast<std::size_t>(c + 1) *
                                       static_cast<std::size_t>(n_groups) +
                                   static_cast<std::size_t>(g)],
                    finite_or_zero(w * mu_n * outgoing));
        }
        Prr[cg] += w * mu2 * avg / core::constants::c_light;
        angular_edge[c] = edge_out;
        incoming = outgoing;
      }
    }
  }
}

// W-G3: 1D cylindrical product-quadrature LC sweep. Per xi-level (levels are
// independent angular chains, run sequentially reusing one angular_edge
// array): (a) the Miller-Alcouffe starting direction generalizes to one
// slab-form step-characteristic sweep per level with |mu| = sin(theta_l),
// seeding the level's weighted-diamond ladder (Morel & Montry 1984
// appendix); (b) the negative- then positive-mu wavefront passes reuse the
// spherical conservative-FV + theta(tau) + weighted-diamond update verbatim
// on the level's M ordinates; (c) the axis reflection pairs (l, m) with
// (l, M-1-m).
__global__ void sn_sweep_cylindrical_lc_kernel(
    const double* __restrict__ source_work,
    const double* __restrict__ sigma_t_work,
    const double* __restrict__ lc_A_work,
    const double* __restrict__ node_r,
    const double* __restrict__ vol,
    const double* __restrict__ mu,
    const double* __restrict__ weights,
    const double* __restrict__ alpha_half,
    const double* __restrict__ tau_wd,
    const double* __restrict__ sd_mu,
    const double* __restrict__ psi_prev,
    double* __restrict__ psi,
    double* __restrict__ psi_outgoing,
    double* __restrict__ origin,
    const double* __restrict__ outer_psi_in,  // W-B2: nullptr => vacuum
    int n_cells,
    int n_groups,
    int n_angles,
    int n_azim,
    double dt,
    const int geom) {
  const int g = blockIdx.x;
  const int lane = threadIdx.x;
  if (g >= n_groups || lane >= kSweepWarp) {
    return;
  }
  const int n_levels = n_angles / n_azim;
  const int half_azim = n_azim / 2;
  const int diagonal_count = n_cells + half_azim - 1;

  extern __shared__ double shared[];
  double* angular_edge = shared;
  double* inner_boundary = shared + n_cells;

  for (int c = lane; c < n_cells; c += kSweepWarp) {
    angular_edge[c] = 0.0;
  }
  for (int n = lane; n < n_angles; n += kSweepWarp) {
    inner_boundary[n] = 0.0;
    origin[static_cast<std::size_t>(g) * static_cast<std::size_t>(n_angles) +
           static_cast<std::size_t>(n)] = 0.0;
  }
  __syncwarp();

  const double inv_cdt =
      (dt > 0.0) ? (1.0 / (core::constants::c_light * dt)) : 0.0;

  for (int level = 0; level < n_levels; ++level) {
    const int base = level * n_azim;
    const double sd = fmin(fmax(finite_or_zero(sd_mu[level]), 1.0e-12), 1.0);

    // Per-level Miller-Alcouffe starting direction (omega = pi): the level's
    // diametral ray has radial cosine sin(theta_l), so the spherical slab
    // step-characteristic carries over with the optical depth scaled by
    // 1/sin(theta_l). Seeds the level's angular ladder edge psi_{l,1/2}.
    if (lane == 0) {
      double psi_sd = (outer_psi_in != nullptr)
                          ? fmax(finite_or_zero(outer_psi_in[g]), 0.0)
                          : 0.0;
      for (int c = n_cells - 1; c >= 0; --c) {
        const std::size_t cg_sd = cg_index(c, g, n_groups);
        const std::size_t p_sd =
            psi_index(g, base, c, n_groups, n_angles);
        const double sigma_eff =
            nonnegative_finite(sigma_t_work[cg_sd]) + inv_cdt;
        const double q_sd = nonnegative_finite(source_work[cg_sd]) +
                            inv_cdt * nonnegative_finite(psi_prev[p_sd]);
        const double dr = fmax(node_r[c + 1] - node_r[c], 0.0);
        const double tau_opt = sigma_eff * dr / sd;
        const double q_avg_sd = (sigma_eff > 0.0) ? (q_sd / sigma_eff) : 0.0;
        double att;
        double avg_w;
        if (tau_opt > 1.0e-12) {
          att = exp(-tau_opt);
          avg_w = (1.0 - att) / tau_opt;
        } else {
          att = 1.0 - tau_opt;
          avg_w = 1.0 - 0.5 * tau_opt;
        }
        angular_edge[c] = fmax(q_avg_sd + (psi_sd - q_avg_sd) * avg_w, 0.0);
        psi_sd = q_avg_sd + (psi_sd - q_avg_sd) * att;
      }
    }
    __syncwarp();

    const int n_neg = base + lane;
    double neg_incoming = 0.0;
    double neg_mu = 0.0;
    double neg_alpha_prev_raw = 0.0;
    double neg_alpha_next_raw = 0.0;
    double neg_w = 0.0;
    double neg_tau_wd = 0.5;
    bool neg_active = (lane < half_azim);
    if (neg_active) {
      neg_mu = finite_or_zero(mu[n_neg]);
      neg_active = (neg_mu < 0.0);
      if (neg_active) {
        neg_alpha_prev_raw = finite_or_zero(alpha_half[n_neg]);
        neg_alpha_next_raw = finite_or_zero(alpha_half[n_neg + 1]);
        neg_w = finite_or_zero(weights[n_neg]);
        neg_tau_wd = fmin(fmax(finite_or_zero(tau_wd[n_neg]), 1.0e-6), 1.0);
        // W-B2: Marshak outer boundary — half-range isotropic incoming
        // intensity 2*F_inc,g (equilibrium-exact; nullptr keeps vacuum).
        if (outer_psi_in != nullptr) {
          neg_incoming = fmax(finite_or_zero(outer_psi_in[g]), 0.0);
        }
      }
    }
    for (int d = 0; d < diagonal_count; ++d) {
      const int cp = d - lane;
      if (neg_active && cp >= 0 && cp < n_cells) {
        const int c = n_cells - 1 - cp;
        const std::size_t cg = cg_index(c, g, n_groups);
        const std::size_t p = psi_index(g, n_neg, c, n_groups, n_angles);
        const double area_in = face_area(node_r[c], geom);
        const double area_out = face_area(node_r[c + 1], geom);
        const double V_in = nonnegative_finite(vol[c]);
        const double V =
            (V_in > 0.0) ? V_in : shell_volume(node_r[c], node_r[c + 1], geom);
        const double sigma_t = nonnegative_finite(sigma_t_work[cg]);
        const double source = nonnegative_finite(source_work[cg]) +
                              inv_cdt * nonnegative_finite(psi_prev[p]);
        const double dA = area_out - area_in;
        const double w_safe = fmax(neg_w, 1.0e-300);
        const double ang_scale = dA / w_safe;
        const double alpha_prev = ang_scale * neg_alpha_prev_raw;
        const double alpha_next = ang_scale * neg_alpha_next_raw;
        const double inv_V = (V > 0.0) ? (1.0 / V) : 0.0;
        // The lower angular edge carries the per-level starting-direction
        // seed for m = 0 and the weighted-diamond ladder value afterwards.
        const double edge_in = fmax(angular_edge[c], 0.0);
        const double kappa_total =
            sigma_t + inv_cdt + (alpha_next / neg_tau_wd) * inv_V;
        const double q_total =
            source +
            (alpha_prev + alpha_next * (1.0 - neg_tau_wd) / neg_tau_wd) *
                edge_in * inv_V;
        const double theta_sp = finite_or_zero(lc_A_work[p]);
        const double abs_mu_n = fabs(neg_mu);
        // mu < 0: upstream face is the geometric outer face (c+1).
        const double stream_up = abs_mu_n * area_out * inv_V;
        const double stream_dn = abs_mu_n * area_in * inv_V;
        const double denom = stream_dn + kappa_total * theta_sp;
        const double outgoing =
            (denom > 0.0)
                ? fmax((q_total +
                        (stream_up - kappa_total * (1.0 - theta_sp)) *
                            neg_incoming) /
                           denom,
                       0.0)
                : 0.0;
        const double avg =
            fmax(theta_sp * outgoing + (1.0 - theta_sp) * neg_incoming, 0.0);
        // Weighted-diamond angular closure (Morel & Montry Eq. 16 / A1).
        const double edge_out =
            fmax((avg - (1.0 - neg_tau_wd) * edge_in) / neg_tau_wd, 0.0);
        psi[p] = avg;
        psi_outgoing[p] = outgoing;
        angular_edge[c] = edge_out;
        neg_incoming = outgoing;
      }
      __syncwarp();
    }
    if (neg_active) {
      inner_boundary[n_neg] = neg_incoming;
      origin[static_cast<std::size_t>(g) * static_cast<std::size_t>(n_angles) +
             static_cast<std::size_t>(n_neg)] = neg_incoming;
    }
    __syncwarp();

    const int n_pos = base + half_azim + lane;
    double pos_incoming = 0.0;
    double pos_mu = 0.0;
    double pos_alpha_prev_raw = 0.0;
    double pos_alpha_next_raw = 0.0;
    double pos_w = 0.0;
    double pos_tau_wd = 0.5;
    bool pos_lower_edge = false;
    bool pos_active = (lane < half_azim);
    if (pos_active) {
      pos_mu = finite_or_zero(mu[n_pos]);
      pos_active = (pos_mu > 0.0);
      if (pos_active) {
        pos_alpha_prev_raw = finite_or_zero(alpha_half[n_pos]);
        pos_alpha_next_raw = finite_or_zero(alpha_half[n_pos + 1]);
        pos_w = finite_or_zero(weights[n_pos]);
        pos_tau_wd = fmin(fmax(finite_or_zero(tau_wd[n_pos]), 1.0e-6), 1.0);
        pos_lower_edge = (pos_alpha_prev_raw == 0.0);
        // Axis reflection within the xi-level: (l, m) <- (l, M-1-m).
        const int reflected = base + (half_azim - 1 - lane);
        pos_incoming =
            (reflected >= base && reflected < base + half_azim)
                ? fmax(inner_boundary[reflected], 0.0)
                : 0.0;
        origin[static_cast<std::size_t>(g) *
                   static_cast<std::size_t>(n_angles) +
               static_cast<std::size_t>(n_pos)] = pos_incoming;
      }
    }
    for (int d = 0; d < diagonal_count; ++d) {
      const int c = d - lane;
      if (pos_active && c >= 0 && c < n_cells) {
        const std::size_t cg = cg_index(c, g, n_groups);
        const std::size_t p = psi_index(g, n_pos, c, n_groups, n_angles);
        const double area_in = face_area(node_r[c], geom);
        const double area_out = face_area(node_r[c + 1], geom);
        const double V_in = nonnegative_finite(vol[c]);
        const double V =
            (V_in > 0.0) ? V_in : shell_volume(node_r[c], node_r[c + 1], geom);
        const double sigma_t = nonnegative_finite(sigma_t_work[cg]);
        const double source = nonnegative_finite(source_work[cg]) +
                              inv_cdt * nonnegative_finite(psi_prev[p]);
        const double dA = area_out - area_in;
        const double w_safe = fmax(pos_w, 1.0e-300);
        const double ang_scale = dA / w_safe;
        const double alpha_prev = ang_scale * pos_alpha_prev_raw;
        const double alpha_next = ang_scale * pos_alpha_next_raw;
        const double inv_V = (V > 0.0) ? (1.0 / V) : 0.0;
        const double edge_in =
            pos_lower_edge ? 0.0 : fmax(angular_edge[c], 0.0);
        const double kappa_total =
            sigma_t + inv_cdt + (alpha_next / pos_tau_wd) * inv_V;
        const double q_total =
            source +
            (alpha_prev + alpha_next * (1.0 - pos_tau_wd) / pos_tau_wd) *
                edge_in * inv_V;
        const double theta_sp = finite_or_zero(lc_A_work[p]);
        const double abs_mu_p = fabs(pos_mu);
        // mu > 0: upstream face is the geometric inner face (c).
        const double stream_up = abs_mu_p * area_in * inv_V;
        const double stream_dn = abs_mu_p * area_out * inv_V;
        const double denom = stream_dn + kappa_total * theta_sp;
        const double outgoing =
            (denom > 0.0)
                ? fmax((q_total +
                        (stream_up - kappa_total * (1.0 - theta_sp)) *
                            pos_incoming) /
                           denom,
                       0.0)
                : 0.0;
        const double avg =
            fmax(theta_sp * outgoing + (1.0 - theta_sp) * pos_incoming, 0.0);
        const double edge_out =
            fmax((avg - (1.0 - pos_tau_wd) * edge_in) / pos_tau_wd, 0.0);
        psi[p] = avg;
        psi_outgoing[p] = outgoing;
        angular_edge[c] = edge_out;
        pos_incoming = outgoing;
      }
      __syncwarp();
    }
  }
}

__global__ void precompute_lc_weights_kernel(
    const double* __restrict__ sigma_t_work,
    const double* __restrict__ node_r,
    const double* __restrict__ vol,
    const double* __restrict__ mu,
    const double* __restrict__ weights,
    const double* __restrict__ alpha_half,
    double* __restrict__ lc_E_work,
    double* __restrict__ lc_A_work,
    int n_cells,
    int n_groups,
    int n_angles,
    double dt,
    const int geom) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups * n_angles;
  if (idx >= total) {
    return;
  }
  const int angles_per_cell = n_groups * n_angles;
  const int c = idx / angles_per_cell;
  const int rem = idx - c * angles_per_cell;
  const int g = rem / n_angles;
  const int n = rem - g * n_angles;
  const std::size_t cg = cg_index(c, g, n_groups);
  const double mu_n = finite_or_zero(mu[n]);
  const double abs_mu = fabs(mu_n);
  const double alpha_next_raw = finite_or_zero(alpha_half[n + 1]);
  const double w = finite_or_zero(weights[n]);
  const double area_in = face_area(node_r[c], geom);
  const double area_out = face_area(node_r[c + 1], geom);
  const double V_in = nonnegative_finite(vol[c]);
  const double V =
      (V_in > 0.0) ? V_in : shell_volume(node_r[c], node_r[c + 1], geom);
  const double sigma_t = nonnegative_finite(sigma_t_work[cg]);
  const double dA = area_out - area_in;
  const double w_safe = fmax(w, 1.0e-300);
  const double ang_scale = dA / w_safe;
  const double alpha_next = ang_scale * alpha_next_raw;
  const double inv_V = (V > 0.0) ? (1.0 / V) : 0.0;
  const double inv_cdt =
      (dt > 0.0) ? (1.0 / (core::constants::c_light * dt)) : 0.0;
  const double kappa_total = sigma_t + inv_cdt + alpha_next * inv_V;
  const double L = fmax(finite_or_zero(node_r[c + 1] - node_r[c]), 0.0);
  const double tau = (abs_mu > 0.0 && kappa_total > 0.0)
                         ? (kappa_total * L / abs_mu)
                         : 0.0;
  const LCWeights lc = compute_lc_weights(tau);
  lc_E_work[idx] = lc.E;
  // Conservative-streaming rewrite: the sweep now solves the CONSERVATIVE
  // finite-volume balance (streaming |mu| (A_dn psi_dn - A_up psi_up)/V)
  // with a theta-weighted spatial closure psi_m = theta psi_dn +
  // (1-theta) psi_up. theta(tau) = 1/(1-e^-tau) - 1/tau (classic
  // exponential weight: -> 1/2 diamond as tau -> 0, -> 1 step as
  // tau -> inf), computed stably from the series-protected A =
  // (1-e^-tau)/tau as theta = (1 - A)/(tau A).
  const double theta =
      (tau > 1.0e-8) ? fmin(fmax((1.0 - lc.A) / (tau * lc.A), 0.5), 1.0)
                     : 0.5;
  lc_A_work[idx] = theta;
}
#else
__global__ void sn_sweep_reflected_pair_kernel(
    const double* __restrict__ source_work,
    const double* __restrict__ sigma_t_work,
    const double* __restrict__ node_r,
    const double* __restrict__ vol,
    const double* __restrict__ mu,
    const double* __restrict__ alpha_half,
    const double* __restrict__ psi_prev,
    double* __restrict__ psi,
    double* __restrict__ origin,
    int n_cells,
    int n_groups,
    int n_angles,
    double dt,
    const int geom) {
  const int half_angles = n_angles / 2;
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_pairs = n_groups * half_angles;
  if (pair >= n_pairs) {
    return;
  }
  const int g = pair / half_angles;
  const int n_neg = pair - g * half_angles;
  const int n_pos = n_angles - 1 - n_neg;
  const double mu_neg = finite_or_zero(mu[n_neg]);
  const double mu_pos = finite_or_zero(mu[n_pos]);
  if (!(mu_neg < 0.0) || !(mu_pos > 0.0)) {
    return;
  }

  const double inv_cdt =
      (dt > 0.0) ? (1.0 / (core::constants::c_light * dt)) : 0.0;

  const double abs_mu_neg = fabs(mu_neg);
  const double alpha_next_neg = alpha_half[n_neg + 1];
  double incoming = 0.0;
  for (int c = n_cells - 1; c >= 0; --c) {
    const std::size_t cg = cg_index(c, g, n_groups);
    const std::size_t p = psi_index(g, n_neg, c, n_groups, n_angles);
    const double area_in = face_area(node_r[c], geom);
    const double area_out = face_area(node_r[c + 1], geom);
    const double V_in = nonnegative_finite(vol[c]);
    const double V = (V_in > 0.0) ? V_in : shell_volume(node_r[c], node_r[c + 1], geom);
    const double sigma_t = nonnegative_finite(sigma_t_work[cg]);
    const double source = nonnegative_finite(source_work[cg]) +
                          inv_cdt * nonnegative_finite(psi_prev[p]);
    const double denom =
        2.0 * abs_mu_neg * area_in + 2.0 * alpha_next_neg +
        (sigma_t + inv_cdt) * V;
    double avg = 0.0;
    if (denom > 0.0 && isfinite(denom)) {
      avg = (V * source + abs_mu_neg * (area_in + area_out) * incoming) / denom;
    }
    avg = fmax(finite_or_zero(avg), 0.0);
    double outgoing = 2.0 * avg - incoming;
    if (!isfinite(outgoing) || outgoing < 0.0) {
      outgoing = 0.0;
      avg = 0.5 * incoming;
    }
    psi[p] = avg;
    incoming = outgoing;
  }
  origin[g * n_angles + n_neg] = incoming;

  incoming = fmax(finite_or_zero(incoming), 0.0);
  origin[g * n_angles + n_pos] = incoming;
  const double alpha_next_pos = alpha_half[n_pos + 1];
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t cg = cg_index(c, g, n_groups);
    const std::size_t p = psi_index(g, n_pos, c, n_groups, n_angles);
    const double area_in = face_area(node_r[c], geom);
    const double area_out = face_area(node_r[c + 1], geom);
    const double V_in = nonnegative_finite(vol[c]);
    const double V = (V_in > 0.0) ? V_in : shell_volume(node_r[c], node_r[c + 1], geom);
    const double sigma_t = nonnegative_finite(sigma_t_work[cg]);
    const double source = nonnegative_finite(source_work[cg]) +
                          inv_cdt * nonnegative_finite(psi_prev[p]);
    const double denom =
        2.0 * mu_pos * area_out + 2.0 * alpha_next_pos +
        (sigma_t + inv_cdt) * V;
    double avg = 0.0;
    if (denom > 0.0 && isfinite(denom)) {
      avg = (V * source + mu_pos * (area_in + area_out) * incoming) / denom;
    }
    avg = fmax(finite_or_zero(avg), 0.0);
    double outgoing = 2.0 * avg - incoming;
    if (!isfinite(outgoing) || outgoing < 0.0) {
      outgoing = 0.0;
      avg = 0.5 * incoming;
    }
    psi[p] = avg;
    incoming = outgoing;
  }
}
#endif

#ifdef TENRYU_DEBUG_LANE_PARALLEL
__global__ void sn_reduce_moments_from_psi_kernel(
    const double* __restrict__ psi,
    const double* __restrict__ mu,
    const double* __restrict__ weights,
    double* __restrict__ phi_sweep,
    double* __restrict__ F_sweep,
    double* __restrict__ Prr,
    int n_cells,
    int n_groups,
    int n_angles) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int c = idx / n_groups;
  const int g = idx - c * n_groups;
  double phi = 0.0;
  double flux = 0.0;
  double pressure = 0.0;
  for (int n = 0; n < n_angles; ++n) {
    const double avg = fmax(
        finite_or_zero(psi[psi_index(g, n, c, n_groups, n_angles)]), 0.0);
    const double mu_n = finite_or_zero(mu[n]);
    const double w = finite_or_zero(weights[n]);
    phi += w * avg;
    flux += w * mu_n * avg;
    pressure += w * mu_n * mu_n * avg;
  }
  phi_sweep[idx] = phi;
  F_sweep[idx] = flux;
  Prr[idx] = pressure / core::constants::c_light;
}
#endif

__global__ void sn_moments_reduction_k2_kernel(
    const double* __restrict__ psi,
    const double* __restrict__ mu,
    const double* __restrict__ weights,
    double* __restrict__ phi_sweep,
    double* __restrict__ F_sweep,
    double* __restrict__ Prr,
    int n_cells,
    int n_groups,
    int n_angles) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int c = idx / n_groups;
  const int g = idx - c * n_groups;
  double phi_sum = 0.0;
  double F_sum = 0.0;
  double Prr_sum = 0.0;
  for (int n = 0; n < n_angles; ++n) {
    const double avg = psi[psi_index(g, n, c, n_groups, n_angles)];
    const double mu_n = finite_or_zero(mu[n]);
    const double w = finite_or_zero(weights[n]);
    const double mu2 = mu_n * mu_n;
    phi_sum += w * avg;
    F_sum += w * mu_n * avg;
    Prr_sum += w * mu2 * avg / core::constants::c_light;
  }
  phi_sweep[idx] = phi_sum;
  F_sweep[idx] = F_sum;
  Prr[idx] = Prr_sum;
}

__global__ void sn_face_flux_reduction_k2_kernel(
    const double* __restrict__ psi_outgoing,
    const double* __restrict__ mu,
    const double* __restrict__ weights,
    double* __restrict__ face_flux_raw,
    int n_cells,
    int n_groups,
    int n_angles) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = (n_cells + 1) * n_groups;
  if (idx >= total) {
    return;
  }
  const int face = idx / n_groups;
  const int g = idx - face * n_groups;
  double flux = 0.0;
  for (int n = 0; n < n_angles; ++n) {
    const double mu_n = finite_or_zero(mu[n]);
    if (mu_n == 0.0) {
      continue;
    }
    const double w = finite_or_zero(weights[n]);
    if (mu_n < 0.0) {
      const int donor_cell = face;
      if (donor_cell < n_cells) {
        const double outgoing =
            psi_outgoing[psi_index(g, n, donor_cell, n_groups, n_angles)];
        flux += finite_or_zero(w * mu_n * outgoing);
      }
    } else {
      const int donor_cell = face - 1;
      if (donor_cell >= 0) {
        const double outgoing =
            psi_outgoing[psi_index(g, n, donor_cell, n_groups, n_angles)];
        flux += finite_or_zero(w * mu_n * outgoing);
      }
    }
  }
  face_flux_raw[idx] = flux;
}

__global__ void phi_to_rad_E_kernel(const double* __restrict__ phi,
                                    double* __restrict__ rad_E,
                                    int n_total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_total) {
    rad_E[idx] = nonnegative_finite(phi[idx]) / core::constants::c_light;
  }
}

// Void-anchor fix: in effectively-void cells (per-step absorbed fraction
// lambda_pa = c dt sigma_pa below ~1e-4) the conservative flux-form E update
// has no relaxation mechanism — any transient bookkeeping imprint freezes
// (neutral equilibrium; measured +6..+87% vs the transport moments). There
// is also nothing for it to conserve against there (no matter coupling), so
// the transport moments are the truth: blend the finalized rad_E toward
// phi/c with a smoothstep in lambda_pa. w = 1 exactly for
// lambda_pa >= 1e-3 keeps every absorbing-regime gate bit-identical
// (su_olson lambda ~ 3e-2; the xc2b gap lambda ~ 1.5e-8 — 4+ orders of
// separation on both sides).
__global__ void anchor_void_rad_E_to_moments_kernel(
    double* __restrict__ rad_E,
    const double* __restrict__ phi_old,
    const double* __restrict__ sigma_a,
    const double* __restrict__ sigma_s,
    const double* __restrict__ vol,
    double* __restrict__ anchor_dE_tally,  // [0]=signed sum, [1]=|.| sum [erg]
    int n_total,
    int n_groups,
    double dt) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_total) {
    return;
  }
  // Total-extinction gate fix: a thick pure-scattering cell is in the
  // diffusion regime (AP-blend authority; E evolves via the flux form), not a
  // void. Only genuinely transparent cells (c dt (sigma_a + sigma_s) below the
  // band) anchor rad_E to the transport moments (void-anchor fix).
  const double lambda_ext = fmax(dt, 0.0) * core::constants::c_light *
                            (nonnegative_finite(sigma_a[idx]) +
                             nonnegative_finite(sigma_s[idx]));
  constexpr double kLambdaLo = 1.0e-4;
  constexpr double kLambdaHi = 1.0e-3;
  if (lambda_ext >= kLambdaHi) {
    return;  // w == 1 exactly: flux-form value untouched (bitwise).
  }
  const double E_moments =
      nonnegative_finite(phi_old[idx]) / core::constants::c_light;
  double w = 0.0;
  if (lambda_ext > kLambdaLo) {
    const double t = (lambda_ext - kLambdaLo) / (kLambdaHi - kLambdaLo);
    w = smoothstep01(t);
  }
  const double E_before = nonnegative_finite(rad_E[idx]);
  const double E_after = w * E_before + (1.0 - w) * E_moments;
  rad_E[idx] = E_after;
  // 2026-07-26 review: the anchor is deliberately nonconservative
  // (it replaces the flux-form state with the transport moment in transparent
  // cells); ledger the energy it moves so a validation run can assert the
  // anchor stayed inert (sum |dE| == 0) before trusting the run as an S_N
  // reference. Ledger-class scalar: atomicAdd order noise is within the
  // documented host-ledger replica band; the physics field update above is
  // untouched.
  if (anchor_dE_tally != nullptr && vol != nullptr) {
    const double V = fmax(finite_or_zero(vol[idx / n_groups]), 0.0);
    const double dE_erg = V * (E_after - E_before);
    if (dE_erg != 0.0) {
      atomicAdd(&anchor_dE_tally[0], dE_erg);
      atomicAdd(&anchor_dE_tally[1], fabs(dE_erg));
    }
  }
}

double* sn_anchor_tally_device() {
  static double* d_tally = nullptr;
  if (d_tally == nullptr) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_tally),
                          2U * sizeof(double)),
               "SN anchor tally alloc failed");
  }
  return d_tally;
}

__global__ void sn_chi_kernel(const double* __restrict__ rad_E,
                              const double* __restrict__ Prr,
                              double* __restrict__ chi,
                              int n_total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_total) {
    const double E = nonnegative_finite(rad_E[idx]);
    const double P = nonnegative_finite(Prr[idx]);
    const double value = P / fmax(E, kEnergyFloor);
    chi[idx] = isfinite(value) ? fmin(fmax(value, 0.0), 1.0) : (1.0 / 3.0);
  }
}

__global__ void sn_diag_emission_at_Tn_kernel(const double* __restrict__ eta,
                                              double* __restrict__ out,
                                              int n_total,
                                              double dt) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_total) {
    out[idx] = fmax(dt, 0.0) * nonnegative_finite(eta[idx]);
  }
}

__global__ void sn_diag_chi_opacity_kernel(
    const double* __restrict__ sigma_pa_old,
    const double* __restrict__ sigma_pe_old,
    const double* __restrict__ sigma_pa_new,
    const double* __restrict__ sigma_pe_new,
    double* __restrict__ chi_opacity,
    int n_total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_total) {
    return;
  }
  constexpr double eps = 1.0e-300;
  const double pa_old = nonnegative_finite(sigma_pa_old[idx]);
  const double pe_old = nonnegative_finite(sigma_pe_old[idx]);
  const double pa_new = nonnegative_finite(sigma_pa_new[idx]);
  const double pe_new = nonnegative_finite(sigma_pe_new[idx]);
  const double chi_pa = fabs(pa_new - pa_old) / fmax(pa_new, eps);
  const double chi_pe = fabs(pe_new - pe_old) / fmax(pe_new, eps);
  const double chi = fmax(chi_pa, chi_pe);
  chi_opacity[idx] = isfinite(chi) ? chi : 0.0;
}

#ifdef TENRYU_DEBUG_SN_METRICS
__global__ void sn_metric_identity_check_kernel(
    const double* __restrict__ node_r,
    const double* __restrict__ mu,
    const double* __restrict__ weights,
    const double* __restrict__ alpha_half,
    double* __restrict__ out,
    int n_cells,
    int n_angles,
    const int geom) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_angles;
  if (idx >= total) {
    return;
  }
  const int c = idx / n_angles;
  const int n = idx - c * n_angles;
  const double area_in = face_area(node_r[c], geom);
  const double area_out = face_area(node_r[c + 1], geom);
  const double dA = area_out - area_in;
  const double w = fmax(finite_or_zero(weights[n]), 1.0e-300);
  const double diff =
      finite_or_zero(alpha_half[n + 1]) - finite_or_zero(alpha_half[n]);
  const double lhs = (dA / w) * diff;
  const double rhs = -finite_or_zero(mu[n]) * dA;
  const double rel = fabs(lhs - rhs) / fmax(fabs(rhs), 1.0);
  auto* ull = reinterpret_cast<unsigned long long*>(out);
  unsigned long long old = *ull;
  unsigned long long assumed = old;
  const unsigned long long val = __double_as_longlong(rel);
  while (__longlong_as_double(static_cast<long long>(assumed)) < rel) {
    old = atomicCAS(ull, assumed, val);
    if (old == assumed) {
      break;
    }
    assumed = old;
  }
}
#endif

__global__ void max_relative_delta_kernel(const double* __restrict__ next,
                                          const double* __restrict__ prev,
                                          double* __restrict__ out,
                                          int n_values) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_values) {
    return;
  }
  const double a = finite_or_zero(next[idx]);
  const double b = finite_or_zero(prev[idx]);
  const double rel = fabs(a - b) / fmax(fabs(a), kEnergyFloor);
  auto* ull = reinterpret_cast<unsigned long long*>(out);
  unsigned long long old = *ull;
  unsigned long long assumed = old;
  const unsigned long long val = __double_as_longlong(rel);
  while (__longlong_as_double(static_cast<long long>(assumed)) < rel) {
    old = atomicCAS(ull, assumed, val);
    if (old == assumed) {
      break;
    }
    assumed = old;
  }
}

__global__ void escaped_energy_kernel(const double* __restrict__ x_r,
                                      const double* __restrict__ face_flux,
                                      double* __restrict__ out,
                                      int n_cells,
                                      int n_groups,
                                      double dt,
                                      const int geom) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n_groups || n_cells <= 0) {
    return;
  }
  const double r = x_r[n_cells];
  const double area = face_area(r, geom);
  const std::size_t outer =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups) +
      static_cast<std::size_t>(g);
  atomicAdd(out, dt * area * finite_or_zero(face_flux[outer]));
}

__global__ void apply_face_flux_boundary_kernel(
    double* __restrict__ face_flux,
    const double* __restrict__ outer_psi_in,  // W-B2: nullptr => vacuum
    const double S_neg,  // discrete half-range moment sum_{mu<0} w|mu|
    int n_cells,
    int n_groups) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n_groups || n_cells <= 0) {
    return;
  }
  face_flux[static_cast<std::size_t>(g)] = 0.0;
  const std::size_t outer =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups) +
      static_cast<std::size_t>(g);
  // Outer-boundary escape-flux fix: both branches use the sweep's DISCRETE outgoing half-range tally
  // already accumulated into face_flux[outer] by the face reduction; Marshak
  // subtracts the discrete incoming S_neg*psi_in, vacuum has zero inflow.
  // The historic vacuum-only "Milne escape estimate" 0.5*c*E was 2x the
  // discrete outgoing flux of a near-isotropic field (S_pos*psi ~= 0.25*c*E
  // with the sum-w=2 GL convention) and broke the volume-integrated
  // streaming-conservation identity once the void-anchor fix anchored rad_E to the
  // transport moments (tests 1600/1603; ledger diagnostic 2026-07-19).
  const double outgoing_tally = finite_or_zero(face_flux[outer]);
  double net = outgoing_tally;
  if (outer_psi_in != nullptr) {
    net -= S_neg * fmax(finite_or_zero(outer_psi_in[g]), 0.0);
  }
  face_flux[outer] = net;
}

__global__ void compute_E_star_flux_kernel(
    const double* __restrict__ rad_E_old,
    const double* __restrict__ node_r,
    const double* __restrict__ vol,
    const double* __restrict__ face_flux,
    double* __restrict__ E_star,
    double* __restrict__ diag_E_star_flux,
    int n_cells,
    int n_groups,
    double dt,
    const int geom) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int c = idx / n_groups;
  const int g = idx - c * n_groups;
  const double V = fmax(nonnegative_finite(vol[c]), kEnergyFloor);
  const double A_in = face_area(node_r[c], geom);
  const double A_out = face_area(node_r[c + 1], geom);
  const double F_in =
      finite_or_zero(face_flux[static_cast<std::size_t>(c) *
                                   static_cast<std::size_t>(n_groups) +
                               static_cast<std::size_t>(g)]);
  const double F_out =
      finite_or_zero(face_flux[static_cast<std::size_t>(c + 1) *
                                   static_cast<std::size_t>(n_groups) +
                               static_cast<std::size_t>(g)]);
  const double divF = (A_out * F_out - A_in * F_in) / V;
  const double Estar = nonnegative_finite(rad_E_old[idx]) - dt * divF;
  const double value = isfinite(Estar) ? Estar : 0.0;
  E_star[idx] = value;
  if (diag_E_star_flux != nullptr) {
    diag_E_star_flux[idx] = value;
  }
}

__global__ void compute_diffusion_face_flux_kernel(
    const double* __restrict__ rad_E_old,
    const double* __restrict__ sn_sigma_s,
    const double* __restrict__ node_r,
    double opacity_floor,
    double* __restrict__ face_flux_diff,
    int n_cells,
    int n_groups) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_faces = n_cells + 1;
  const int total = n_faces * n_groups;
  if (idx >= total) {
    return;
  }
  const int f = idx / n_groups;
  const int g = idx - f * n_groups;
  if (f == 0) {
    face_flux_diff[idx] = 0.0;
    return;
  }
  if (f == n_cells) {
    const int c_last = n_cells - 1;
    const double E_last =
        nonnegative_finite(rad_E_old[c_last * n_groups + g]);
    face_flux_diff[idx] = 0.5 * core::constants::c_light * E_last;
    return;
  }
  const int c_l = f - 1;
  const int c_r = f;
  const double sigma_l =
      fmax(nonnegative_finite(sn_sigma_s[c_l * n_groups + g]), opacity_floor);
  const double sigma_r =
      fmax(nonnegative_finite(sn_sigma_s[c_r * n_groups + g]), opacity_floor);
  const double D_l = core::constants::c_light / fmax(3.0 * sigma_l, 1.0e-100);
  const double D_r = core::constants::c_light / fmax(3.0 * sigma_r, 1.0e-100);
  const double D_face = 2.0 * D_l * D_r / fmax(D_l + D_r, kEnergyFloor);
  const double E_l = nonnegative_finite(rad_E_old[c_l * n_groups + g]);
  const double E_r = nonnegative_finite(rad_E_old[c_r * n_groups + g]);
  const double dr_centers =
      0.5 * finite_or_zero(node_r[f + 1] - node_r[f - 1]);
  face_flux_diff[idx] = -D_face * (E_r - E_l) / fmax(dr_centers, kEnergyFloor);
}

__global__ void compute_ap_blended_face_flux_kernel(
    const double* __restrict__ face_flux_sn,
    const double* __restrict__ face_flux_diff,
    const double* __restrict__ rad_E_old,
    const double* __restrict__ Te,
    const double* __restrict__ sn_sigma_s,
    const double* __restrict__ node_r,
    PlanckTableDeviceView planck,
    double opacity_floor,
    double tau_lo,
    double tau_hi,
    double eq_full,
    double eq_off,
    double f_full,
    double f_off,
    double* __restrict__ face_flux_blended,
    double* __restrict__ alpha_face,
    int n_cells,
    int n_groups) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_faces = n_cells + 1;
  const int total = n_faces * n_groups;
  if (idx >= total) {
    return;
  }
  const int f = idx / n_groups;
  const int g = idx - f * n_groups;
  if (f == 0) {
    face_flux_blended[idx] = finite_or_zero(face_flux_sn[idx]);
    alpha_face[idx] = 0.0;
    return;
  }

  // Outer-boundary escape-flux fix: blend the face one-sidedly (thick limit -> FLD-style
  // diffusion boundary flux, streaming limit -> discrete transport tally); the pre-fix Milne overwrite made them coincide.
  const bool outer_boundary = (f == n_cells);
  const int c_l = outer_boundary ? (n_cells - 1) : (f - 1);
  const int c_r = outer_boundary ? (n_cells - 1) : f;
  const double sigma_l =
      fmax(nonnegative_finite(sn_sigma_s[c_l * n_groups + g]), opacity_floor);
  const double sigma_r =
      fmax(nonnegative_finite(sn_sigma_s[c_r * n_groups + g]), opacity_floor);
  const double dr_l =
      fmax(finite_or_zero(node_r[c_l + 1] - node_r[c_l]), kEnergyFloor);
  const double dr_r =
      fmax(finite_or_zero(node_r[c_r + 1] - node_r[c_r]), kEnergyFloor);
  const double tau_l = sigma_l * dr_l;
  const double tau_r = sigma_r * dr_r;
  const double tau_min = fmin(tau_l, tau_r);
  const double alpha_tau =
      smoothstep01((tau_min - tau_lo) / fmax(tau_hi - tau_lo, kEnergyFloor));

  const double T_l = fmax(finite_or_zero(Te[c_l]), 1.0e-3);
  const double T_r = fmax(finite_or_zero(Te[c_r]), 1.0e-3);
  const double E_l = nonnegative_finite(rad_E_old[c_l * n_groups + g]);
  const double E_r = nonnegative_finite(rad_E_old[c_r * n_groups + g]);
  const double T2_l = T_l * T_l;
  const double T2_r = T_r * T_r;
  const double b_l =
      (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T_l), 0.0);
  const double b_r =
      (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T_r), 0.0);
  const double B_l = core::constants::a_eV * T2_l * T2_l * b_l;
  const double B_r = core::constants::a_eV * T2_r * T2_r * b_r;
  const double err_l = fabs(E_l - B_l) / fmax(B_l, 1.0e-30);
  const double err_r = fabs(E_r - B_r) / fmax(B_r, 1.0e-30);
  const double alpha_eq =
      smooth_full_at_low(fmax(err_l, err_r), eq_full, eq_off);

  const double F_abs = fabs(finite_or_zero(face_flux_sn[idx]));
  const double E_avg = 0.5 * (E_l + E_r);
  const double f_red =
      F_abs / fmax(core::constants::c_light * E_avg, kEnergyFloor);
  const double alpha_f = smooth_full_at_low(f_red, f_full, f_off);

  // Outer-boundary escape-flux fix: at the free surface the escape flux makes f_red ~ 0.25
  // intrinsically; the reduced-flux streaming detector must not veto the
  // thick-limit diffusion closure there. tau and equilibrium factors alone
  // decide the boundary regime.
  const double alpha = outer_boundary ? (alpha_tau * alpha_eq)
                                      : (alpha_tau * alpha_eq * alpha_f);
  if (alpha <= 0.0) {
    face_flux_blended[idx] = finite_or_zero(face_flux_sn[idx]);
  } else if (alpha >= 1.0) {
    face_flux_blended[idx] = finite_or_zero(face_flux_diff[idx]);
  } else {
    face_flux_blended[idx] =
        (1.0 - alpha) * finite_or_zero(face_flux_sn[idx]) +
        alpha * finite_or_zero(face_flux_diff[idx]);
  }
  alpha_face[idx] = alpha;
}

__global__ void compute_streaming_theta_kernel(
    const double* __restrict__ rad_E_old,
    const double* __restrict__ node_r,
    const double* __restrict__ vol,
    const double* __restrict__ face_flux,
    double* __restrict__ theta,
    int n_cells,
    int n_groups,
    double dt,
    const int geom) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int c = idx / n_groups;
  const int g = idx - c * n_groups;
  const double V = fmax(nonnegative_finite(vol[c]), kEnergyFloor);
  const double A_in = face_area(node_r[c], geom);
  const double A_out = face_area(node_r[c + 1], geom);
  const double F_in =
      finite_or_zero(face_flux[static_cast<std::size_t>(c) *
                                   static_cast<std::size_t>(n_groups) +
                               static_cast<std::size_t>(g)]);
  const double F_out =
      finite_or_zero(face_flux[static_cast<std::size_t>(c + 1) *
                                   static_cast<std::size_t>(n_groups) +
                               static_cast<std::size_t>(g)]);
  const double out =
      fmax(dt, 0.0) * (fmax(F_out, 0.0) * A_out + fmax(-F_in, 0.0) * A_in) / V;
  const double E_old = nonnegative_finite(rad_E_old[idx]);
  theta[idx] = (out > kEnergyFloor) ? fmin(1.0, E_old / out) : 1.0;
}

// Pass-2 inflow credit. The donor-only cap (pass 1) makes
// free-streaming pass-through impossible (a conduit cell is throttled by its
// stored E even though the inflow replenishes it within the same step) and
// freezes transient accumulation at a neutral equilibrium. Crediting the
// PASS-1-LIMITED inflow keeps positivity exact (E_new = E_old +
// dt (theta_up in - theta' out)/V >= 0 by construction) while a uniformly
// throttled stream (theta >= 1/2) relaxes to full pass-through.
__global__ void compute_streaming_theta_credit_kernel(
    const double* __restrict__ rad_E_old,
    const double* __restrict__ node_r,
    const double* __restrict__ vol,
    const double* __restrict__ face_flux,
    const double* __restrict__ theta_donor,
    double* __restrict__ theta_out,
    int n_cells,
    int n_groups,
    double dt,
    const int geom) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int c = idx / n_groups;
  const int g = idx - c * n_groups;
  const double V = fmax(nonnegative_finite(vol[c]), kEnergyFloor);
  const double A_in = face_area(node_r[c], geom);
  const double A_out = face_area(node_r[c + 1], geom);
  const double F_in =
      finite_or_zero(face_flux[static_cast<std::size_t>(c) *
                                   static_cast<std::size_t>(n_groups) +
                               static_cast<std::size_t>(g)]);
  const double F_out =
      finite_or_zero(face_flux[static_cast<std::size_t>(c + 1) *
                                   static_cast<std::size_t>(n_groups) +
                               static_cast<std::size_t>(g)]);
  const double out =
      fmax(dt, 0.0) * (fmax(F_out, 0.0) * A_out + fmax(-F_in, 0.0) * A_in) / V;
  // Pass-1-limited inflow credit: face c inflow (F_in > 0) is throttled by
  // the LEFT donor cell's theta; face c+1 inflow (F_out < 0) by the RIGHT
  // donor cell's. Domain-boundary inflow has no donor => theta = 1.
  const double th_l =
      (c > 0) ? finite_or_zero(theta_donor[static_cast<std::size_t>(c - 1) *
                                               static_cast<std::size_t>(n_groups) +
                                           static_cast<std::size_t>(g)])
              : 1.0;
  const double th_r =
      (c + 1 < n_cells)
          ? finite_or_zero(theta_donor[static_cast<std::size_t>(c + 1) *
                                           static_cast<std::size_t>(n_groups) +
                                       static_cast<std::size_t>(g)])
          : 1.0;
  const double in_credit =
      fmax(dt, 0.0) *
      (fmax(F_in, 0.0) * th_l * A_in + fmax(-F_out, 0.0) * th_r * A_out) / V;
  const double E_old = nonnegative_finite(rad_E_old[idx]);
  theta_out[idx] =
      (out > kEnergyFloor) ? fmin(1.0, (E_old + in_credit) / out) : 1.0;
}

__global__ void apply_donor_theta_face_flux_kernel(
    const double* __restrict__ face_flux_raw,
    const double* __restrict__ theta,
    double* __restrict__ face_flux_limited,
    int n_cells,
    int n_groups) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_faces = n_cells + 1;
  const int total = n_faces * n_groups;
  if (idx >= total) {
    return;
  }
  const int f = idx / n_groups;
  const int g = idx - f * n_groups;
  const double F = finite_or_zero(face_flux_raw[idx]);
  double th = 1.0;
  if (F > 0.0) {
    th = (f > 0) ? theta[static_cast<std::size_t>(f - 1) *
                             static_cast<std::size_t>(n_groups) +
                         static_cast<std::size_t>(g)]
                 : 1.0;
  } else if (F < 0.0) {
    th = (f < n_cells) ? theta[static_cast<std::size_t>(f) *
                                   static_cast<std::size_t>(n_groups) +
                               static_cast<std::size_t>(g)]
                       : 1.0;
  }
  face_flux_limited[idx] = th * F;
}

std::vector<double> radiation_group_bounds(const core::Config& cfg,
                                           const int n_groups) {
  if (static_cast<int>(cfg.radiation.group_bounds_eV.size()) == n_groups + 1) {
    return cfg.radiation.group_bounds_eV;
  }
  const std::vector<double> range = resolve_compute_T_range_eV(cfg, false);
  const double Tmin = std::max(range[0], 1.0e-3);
  const double Tmax = std::max(range[1], Tmin * 1.001);
  return Groups::make_log_uniform_bounds(n_groups, Tmin, Tmax);
}

void ensure_state_buffers(core::State& state,
                          const int n_cells,
                          const int n_groups,
                          const int n_angles) {
  const std::size_t n_total =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  const std::size_t n_faces =
      static_cast<std::size_t>(n_cells + 1) * static_cast<std::size_t>(n_groups);
  const std::size_t n_psi = n_total * static_cast<std::size_t>(n_angles);
  state.rad_E.reset(n_total);
  state.rad_E_old.reset(n_total);
  state.rad_dep.reset(n_total);
  state.rad_emit.reset(n_total);
  state.sn_sigma_a.reset(n_total);
  state.sn_sigma_pe.reset(n_total);
  state.sn_sigma_s.reset(n_total);
  state.sn_eta.reset(n_total);
  state.sn_phi_old.reset(n_total);
  state.sn_phi_new.reset(n_total);
  state.sn_phi_sweep.reset(n_total);
  state.sn_diag_rad_E_pre.reset(n_total);
  state.sn_diag_rad_E_post.reset(n_total);
  state.sn_diag_rad_emission_at_Tn.reset(n_total);
  state.sn_diag_rad_emission_at_Tnp1.reset(n_total);
  state.sn_diag_rad_absorption.reset(n_total);
  state.sn_diag_clip_energy.reset(n_total);
  state.sn_diag_clip_full_deficit.reset(n_total);
  state.sn_diag_chi_opacity.reset(n_total);
  state.sn_diag_F_first_moment.reset(n_total);
  state.sn_face_flux_raw.reset(n_faces);
  state.sn_face_flux_diff.reset(n_faces);
  state.sn_face_flux_blended.reset(n_faces);
  state.sn_face_flux_limited.reset(n_faces);
  state.sn_face_alpha.reset(n_faces);
  state.sn_stream_theta.reset(n_total);
  state.sn_stream_theta_donor.reset(n_total);
  state.sn_E_star_flux.reset(n_total);
  state.sn_diag_E_star_flux.reset(n_total);
  state.sn_psi_scratch.reset(n_psi);
  state.sn_psi_outgoing_scratch.reset(n_psi);
  state.sn_lc_E_scratch.reset(n_psi);
  state.sn_lc_A_scratch.reset(n_psi);
  state.sn_origin_boundary.reset(
      static_cast<std::size_t>(n_groups) * static_cast<std::size_t>(n_angles));
  state.sn_outer_marshak_psi_in.reset(static_cast<std::size_t>(n_groups));
  state.sn_radial_fixup_count.reset(n_total);
  state.sn_radial_fixup_artificial_abs.reset(n_total);
  state.sn_angular_fixup_count.reset(n_total);
  state.sn_angular_fixup_artificial_abs.reset(n_total);
  state.sn_Prr.reset(n_total);
  state.sn_chi.reset(n_total);
  state.sn_dsa_lower.reset(n_total);
  state.sn_dsa_diag.reset(n_total);
  state.sn_dsa_upper.reset(n_total);
  state.sn_dsa_rhs.reset(n_total);
  state.sn_Te_old.reset(static_cast<std::size_t>(n_cells));
  state.sn_delta_T.reset(static_cast<std::size_t>(n_cells));
  state.sn_reduction_work.reset(1);
}

void initialize_rad_E_old_if_needed(core::State& state,
                                    const int n_cells,
                                    const int n_groups) {
  const std::size_t bytes =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups) *
      sizeof(double);
  if (bytes == 0U) {
    return;
  }
  if (state.step == 0 || state.holo_ale_invalidated) {
    cuda_check(cudaMemcpy(state.rad_E_old.data(),
                          state.rad_E.data(),
                          bytes,
                          cudaMemcpyDeviceToDevice),
               "SN initialize rad_E_old failed");
  }
}

void evaluate_opacity_and_emission(core::State& state,
                                   const core::Config& cfg,
                                   const PlanckTable& planck,
                                   const core::Config::MaterialsConfig::MatDef& mat,
                                   const int n_cells,
                                   const int n_groups,
                                   const double dt) {
  const auto& sn = cfg.radiation.sn_transport;
  const int total = n_cells * n_groups;
  const bool use_nlte =
      mat.opacity_model == "table_nlte" || mat.opacity_model == "tmat";
  if (use_nlte) {
    ensure_nlte_table_uploaded(cfg, mat, n_groups);
    const std::size_t scratch_size = static_cast<std::size_t>(total);
    state.sn_nlte_f_work.reset(scratch_size);
    state.sn_nlte_sigma_eff_work.reset(scratch_size);
    state.sn_nlte_sigma_s_eff_work.reset(scratch_size);
    state.sn_nlte_eta_cdf_work.reset(scratch_size);
    state.sn_nlte_lambda_work.reset(scratch_size);
    auto& cache = nlte_cache();
    const double* cv_e_ptr =
        (state.cv_e.size() == static_cast<std::size_t>(n_cells)) ? state.cv_e.data()
                                                                 : nullptr;
    const auto result = compute_nlte_coefficients_cuda_pure_sn(
        state.rho.data(),
        state.Te.data(),
        state.zbar.data(),
        cv_e_ptr,
        state.cell_is_void.data(),
        state.cell_is_void.size(),
        cache.device.view(),
        planck.device_view(),
        n_cells,
        n_groups,
        dt,
        std::max(mat.A, 1.0e-12),
        1.0,
        0.0,
        1.0,
        mat.lambda_fd_delta_rel,
        mat.lambda_fd_abs_min,
        sn.opacity_cap,
        mat.cv_e_override,
        cfg.numerics.floors.Te,
        std::max(mat.ideal_gas_gamma - 1.0, 1.0e-12),
        false,
        false,
        false,
        state.sn_nlte_f_work.data(),
        state.sn_sigma_a.data(),
        state.sn_sigma_pe.data(),
        state.sn_dsa_rhs.data(),
        state.sn_nlte_sigma_eff_work.data(),
        state.sn_sigma_s.data(),
        state.sn_nlte_eta_cdf_work.data(),
        state.sn_eta.data(),
        state.sn_nlte_lambda_work.data(),
        0,
        cfg.materials.low_density_extrapolation);
    if (result.nan_inf_count != 0 || result.negative_eta_clamp_count != 0) {
      core::log_warning("SN NLTE coefficient clamp counts: nan_inf=" +
                        std::to_string(result.nan_inf_count) +
                        ", eta=" +
                        std::to_string(result.negative_eta_clamp_count));
    }
    return;
  }

  materials::OpacityEvalView opacity_view{};
  opacity_view.rho = state.rho.data();
  opacity_view.Te = state.Te.data();
  opacity_view.sigma_a = state.sn_sigma_a.data();
  opacity_view.sigma_R = state.sn_sigma_s.data();
  opacity_view.n_cells = n_cells;
  opacity_view.n_groups = n_groups;
  opacity_view.opacity_model =
      (mat.opacity_model == "freq_dep_marshak")
          ? materials::kOpacityModelFreqDepMarshak
          : ((mat.opacity_model == "power_law")
                 ? materials::kOpacityModelPowerLaw
                 : materials::kOpacityModelConstant);
  opacity_view.kappa_planck_const = std::max(0.0, mat.kappa_a_constant);
  opacity_view.kappa_rosseland_const = std::max(0.0, mat.kappa_s_constant);
  opacity_view.kappa_floor = sn.opacity_floor;
  opacity_view.kappa_cap = sn.opacity_cap;
  opacity_view.power_law_kappa0 = mat.opacity_power_law_kappa0_cm2_g;
  opacity_view.power_law_alpha_T = mat.opacity_power_law_alpha_T;
  opacity_view.power_law_lambda_rho = mat.opacity_power_law_lambda_rho;
  opacity_view.power_law_T_ref_eV = mat.opacity_power_law_T_ref_eV;
  opacity_view.power_law_rho_ref = mat.opacity_power_law_rho_ref_g_cc;
  std::vector<double> bounds;
  if (opacity_view.opacity_model == materials::kOpacityModelFreqDepMarshak) {
    bounds = radiation_group_bounds(cfg, n_groups);
    state.sn_group_bounds_work.reset(bounds.size());
    cuda_check(cudaMemcpy(state.sn_group_bounds_work.data(),
                          bounds.data(),
                          sizeof(double) * bounds.size(),
                          cudaMemcpyHostToDevice),
               "SN upload group bounds failed");
    opacity_view.group_bounds_eV = state.sn_group_bounds_work.data();
  }
  materials::evaluate_opacity_cuda(opacity_view, nullptr);
  const int grid = (total + kBlock - 1) / kBlock;
  if (grid > 0) {
    cuda_check(cudaMemcpy(state.sn_sigma_pe.data(),
                          state.sn_sigma_a.data(),
                          sizeof(double) * static_cast<std::size_t>(total),
                          cudaMemcpyDeviceToDevice),
               "SN copy sigma_a to sigma_pe failed");
    build_eta_from_planck_kernel<<<grid, kBlock>>>(state.Te.data(),
                                                   state.sn_sigma_pe.data(),
                                                   planck.device_view(),
                                                   state.sn_eta.data(),
                                                   n_cells,
                                                   n_groups,
                                                   cfg.numerics.floors.Te);
    cuda_check(cudaGetLastError(), "SN eta launch failed");
  }
}

double copy_reduction_scalar(core::State& state) {
  double value = 0.0;
  cuda_check(cudaMemcpy(&value,
                        state.sn_reduction_work.data(),
                        sizeof(double),
                        cudaMemcpyDeviceToHost),
             "SN copy reduction scalar failed");
  return value;
}

void zero_reduction(core::State& state) {
  cuda_check(cudaMemset(state.sn_reduction_work.data(), 0, sizeof(double)),
             "SN zero reduction scalar failed");
}

void zero_step_setup(core::State& state, const int total, cudaStream_t stream) {
  if (total <= 0) {
    return;
  }
  const int grid = (total + kBlock - 1) / kBlock;
  zero_sn_step_setup_kernel<<<grid, kBlock, 0, stream>>>(
      state.sn_diag_rad_E_pre.data(),
      state.sn_diag_rad_E_post.data(),
      state.sn_diag_rad_emission_at_Tn.data(),
      state.sn_diag_rad_emission_at_Tnp1.data(),
      state.sn_diag_rad_absorption.data(),
      state.sn_diag_clip_energy.data(),
      state.sn_diag_clip_full_deficit.data(),
      state.sn_diag_chi_opacity.data(),
      state.sn_diag_F_first_moment.data(),
      state.sn_E_star_flux.data(),
      state.sn_diag_E_star_flux.data(),
      state.sn_stream_theta.data(),
      state.sn_radial_fixup_count.data(),
      state.sn_radial_fixup_artificial_abs.data(),
      state.sn_angular_fixup_count.data(),
      state.sn_angular_fixup_artificial_abs.data(),
      total);
  cuda_check(cudaGetLastError(), "SN zero step setup launch failed");
}

void zero_face_flux_raw(core::State& state,
                        const int n_cells,
                        const int n_groups,
                        cudaStream_t stream) {
  const int total = (n_cells + 1) * n_groups;
  if (total <= 0) {
    return;
  }
  const int grid = (total + kBlock - 1) / kBlock;
  zero_sn_face_flux_buffers_kernel<<<grid, kBlock, 0, stream>>>(
      state.sn_face_flux_raw.data(),
      state.sn_face_flux_limited.data(),
      state.sn_face_flux_diff.data(),
      state.sn_face_flux_blended.data(),
      state.sn_face_alpha.data(),
      total);
  cuda_check(cudaGetLastError(), "SN zero face flux buffers launch failed");
}

#ifdef TENRYU_DEBUG_SN_METRICS
void check_sn_metric_identity(core::State& state,
                              const double* const mu,
                              const double* const weights,
                              const double* const alpha,
                              const int n_cells,
                              const int n_angles) {
  zero_reduction(state);
  const int total = n_cells * n_angles;
  const int grid = (total + kBlock - 1) / kBlock;
  const int geom = state.mesh.geometry_code;
  if (grid > 0) {
    sn_metric_identity_check_kernel<<<grid, kBlock>>>(state.x_r.data(),
                                                      mu,
                                                      weights,
                                                      alpha,
                                                      state.sn_reduction_work.data(),
                                                      n_cells,
                                                      n_angles,
                                                      geom);
    cuda_check(cudaGetLastError(), "SN metric identity check launch failed");
  }
  const double max_error = copy_reduction_scalar(state);
  TENRYU_ASSERT(max_error <= 1.0e-12, "SN metric identity check failed");
}
#endif

double reduce_relative_delta(core::State& state,
                             const double* next,
                             const double* prev,
                             const int n_values) {
  zero_reduction(state);
  const int grid = (n_values + kBlock - 1) / kBlock;
  if (grid > 0) {
    max_relative_delta_kernel<<<grid, kBlock>>>(
        next, prev, state.sn_reduction_work.data(), n_values);
    cuda_check(cudaGetLastError(), "SN residual reduction launch failed");
  }
  return copy_reduction_scalar(state);
}

double compute_escaped_energy(core::State& state,
                              const int n_cells,
                              const int n_groups,
                              const double dt) {
  zero_reduction(state);
  const int grid = (n_groups + kBlock - 1) / kBlock;
  if (grid > 0) {
    // The E*-flux path consumes sn_face_flux_limited, so escape booking now matches the array that actually updates E in absorbing regimes; in streaming tests theta == 1 and blend alpha == 0 make limited == raw, bit-unchanged there.
    escaped_energy_kernel<<<grid, kBlock>>>(
        state.x_r.data(),
        state.sn_face_flux_limited.data(),
        state.sn_reduction_work.data(),
        n_cells,
        n_groups,
        dt,
        state.mesh.geometry_code);
    cuda_check(cudaGetLastError(), "SN escaped energy launch failed");
  }
  return copy_reduction_scalar(state);
}

void copy_rad_E_to_old(core::State& state, const int n_cells, const int n_groups) {
  const std::size_t bytes =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups) *
      sizeof(double);
  if (bytes > 0U) {
    cuda_check(cudaMemcpy(state.rad_E_old.data(),
                          state.rad_E.data(),
                          bytes,
                          cudaMemcpyDeviceToDevice),
               "SN update rad_E_old failed");
  }
}

void compute_opacity_lag_diagnostic(core::State& state,
                                    const core::Config& cfg,
                                    const PlanckTable& planck,
                                    const core::Config::MaterialsConfig::MatDef& mat,
                                    const int n_cells,
                                    const int n_groups,
                                    const double dt) {
  const int total = n_cells * n_groups;
  if (total <= 0) {
    return;
  }
  const int total_grid = (total + kBlock - 1) / kBlock;
  const auto& sn = cfg.radiation.sn_transport;
  core::GroupField1D sigma_pa_new(static_cast<std::size_t>(total));
  core::GroupField1D sigma_pe_new(static_cast<std::size_t>(total));
  const bool use_nlte =
      mat.opacity_model == "table_nlte" || mat.opacity_model == "tmat";
  if (use_nlte) {
    ensure_nlte_table_uploaded(cfg, mat, n_groups);
    core::GroupField1D f_work(static_cast<std::size_t>(total));
    core::GroupField1D sigma_R_work(static_cast<std::size_t>(total));
    core::GroupField1D sigma_a_eff_work(static_cast<std::size_t>(total));
    core::GroupField1D sigma_s_eff_work(static_cast<std::size_t>(total));
    core::GroupField1D eta_cdf_work(static_cast<std::size_t>(total));
    core::GroupField1D eta_work(static_cast<std::size_t>(total));
    core::GroupField1D lambda_work(static_cast<std::size_t>(total));
    auto& cache = nlte_cache();
    const double* cv_e_ptr =
        (state.cv_e.size() == static_cast<std::size_t>(n_cells)) ? state.cv_e.data()
                                                                 : nullptr;
    static_cast<void>(compute_nlte_coefficients_cuda_pure_sn(
        state.rho.data(),
        state.Te.data(),
        state.zbar.data(),
        cv_e_ptr,
        state.cell_is_void.data(),
        state.cell_is_void.size(),
        cache.device.view(),
        planck.device_view(),
        n_cells,
        n_groups,
        dt,
        std::max(mat.A, 1.0e-12),
        1.0,
        0.0,
        1.0,
        mat.lambda_fd_delta_rel,
        mat.lambda_fd_abs_min,
        sn.opacity_cap,
        mat.cv_e_override,
        cfg.numerics.floors.Te,
        std::max(mat.ideal_gas_gamma - 1.0, 1.0e-12),
        false,
        false,
        false,
        f_work.data(),
        sigma_pa_new.data(),
        sigma_pe_new.data(),
        sigma_R_work.data(),
        sigma_a_eff_work.data(),
        sigma_s_eff_work.data(),
        eta_cdf_work.data(),
        eta_work.data(),
        lambda_work.data(),
        0,
        cfg.materials.low_density_extrapolation));
  } else {
    core::GroupField1D sigma_R_new(static_cast<std::size_t>(total));
    materials::OpacityEvalView opacity_view{};
    opacity_view.rho = state.rho.data();
    opacity_view.Te = state.Te.data();
    opacity_view.sigma_a = sigma_pa_new.data();
    opacity_view.sigma_R = sigma_R_new.data();
    opacity_view.n_cells = n_cells;
    opacity_view.n_groups = n_groups;
    opacity_view.opacity_model =
        (mat.opacity_model == "freq_dep_marshak")
            ? materials::kOpacityModelFreqDepMarshak
            : ((mat.opacity_model == "power_law")
                   ? materials::kOpacityModelPowerLaw
                   : materials::kOpacityModelConstant);
    opacity_view.kappa_planck_const = std::max(0.0, mat.kappa_a_constant);
    opacity_view.kappa_rosseland_const = std::max(0.0, mat.kappa_s_constant);
    opacity_view.kappa_floor = sn.opacity_floor;
    opacity_view.kappa_cap = sn.opacity_cap;
    opacity_view.power_law_kappa0 = mat.opacity_power_law_kappa0_cm2_g;
    opacity_view.power_law_alpha_T = mat.opacity_power_law_alpha_T;
    opacity_view.power_law_lambda_rho = mat.opacity_power_law_lambda_rho;
    opacity_view.power_law_T_ref_eV = mat.opacity_power_law_T_ref_eV;
    opacity_view.power_law_rho_ref = mat.opacity_power_law_rho_ref_g_cc;
    core::GroupField1D group_bounds_work;
    if (opacity_view.opacity_model == materials::kOpacityModelFreqDepMarshak) {
      const std::vector<double> bounds = radiation_group_bounds(cfg, n_groups);
      group_bounds_work.reset(bounds.size());
      group_bounds_work.copy_from_host(bounds);
      opacity_view.group_bounds_eV = group_bounds_work.data();
    }
    materials::evaluate_opacity_cuda(opacity_view, nullptr);
    cuda_check(cudaMemcpy(sigma_pe_new.data(),
                          sigma_pa_new.data(),
                          sizeof(double) * static_cast<std::size_t>(total),
                          cudaMemcpyDeviceToDevice),
               "SN diagnostic copy sigma_pa to sigma_pe failed");
  }
  sn_diag_chi_opacity_kernel<<<total_grid, kBlock>>>(
      state.sn_sigma_a.data(),
      state.sn_sigma_pe.data(),
      sigma_pa_new.data(),
      sigma_pe_new.data(),
      state.sn_diag_chi_opacity.data(),
      total);
  cuda_check(cudaGetLastError(), "SN diagnostic opacity chi launch failed");
  cuda_check(cudaDeviceSynchronize(), "SN diagnostic opacity chi sync failed");
}

struct SnQuadratureDeviceCache {
  void ensure(const int requested_angles, const bool cylindrical) {
    const int requested_mode = cylindrical ? 1 : 0;
    if (requested_angles == n_angles && requested_mode == mode) {
      return;
    }
    if (cylindrical) {
      // W-G3: 1D cylindrical product quadrature (Morel & Montry 1984
      // appendix); see docs/design/wg3_sn_cylindrical_quadrature_design.md.
      const SnCylQuadrature1D host_quad =
          build_sn_cyl_quadrature_1d(requested_angles);
      mu.reset(host_quad.mu.size());
      weight.reset(host_quad.weight.size());
      alpha.reset(host_quad.alpha_half.size());
      tau.reset(host_quad.tau_wd.size());
      sd_mu.reset(host_quad.sd_mu.size());
      mu.copy_from_host(host_quad.mu);
      weight.copy_from_host(host_quad.weight);
      alpha.copy_from_host(host_quad.alpha_half);
      tau.copy_from_host(host_quad.tau_wd);
      sd_mu.copy_from_host(host_quad.sd_mu);
      n_azim = host_quad.n_azim;
    } else {
      std::vector<double> host_mu;
      std::vector<double> host_weight;
      compute_gauss_legendre(requested_angles, host_mu, host_weight);
      const std::vector<double> host_alpha =
          angular_coefficients(host_mu, host_weight);
      // Conservative-streaming fix (Morel & Montry 1984, Eqs. 15-16): weighted-diamond
      // angular closure factors. Standard cell-edge cosines partition
      // [-1, 1] by the quadrature weights (sum W = 2 convention):
      // mu_{1/2} = -1, mu_{m+1/2} = mu_{m-1/2} + W_m; tau_m places the
      // quadrature node inside its angular cell so that the S_n
      // diffusion-limit coefficient beta vanishes identically (eliminates
      // the curvature-proportional flux dip).
      std::vector<double> host_tau(host_mu.size(), 0.5);
      {
        double mu_edge = -1.0;
        for (std::size_t m = 0; m < host_mu.size(); ++m) {
          const double mu_lo = mu_edge;
          const double mu_hi = mu_lo + host_weight[m];
          const double denom = mu_hi - mu_lo;
          host_tau[m] =
              (denom > 1.0e-300) ? ((host_mu[m] - mu_lo) / denom) : 0.5;
          mu_edge = mu_hi;
        }
      }
      mu.reset(host_mu.size());
      weight.reset(host_weight.size());
      alpha.reset(host_alpha.size());
      tau.reset(host_tau.size());
      mu.copy_from_host(host_mu);
      weight.copy_from_host(host_weight);
      alpha.copy_from_host(host_alpha);
      tau.copy_from_host(host_tau);
      n_azim = 0;
    }
    mode = requested_mode;
    n_angles = requested_angles;
  }

  int n_angles = 0;
  int mode = 0;    // 0 = mu-only GL (spherical/planar), 1 = cylindrical product
  int n_azim = 0;  // azimuthal ordinates per xi-level when mode == 1
  core::GroupField1D mu;
  core::GroupField1D weight;
  core::GroupField1D alpha;
  core::GroupField1D tau;
  core::GroupField1D sd_mu;
};

SnQuadratureDeviceCache& sn_quadrature_cache() {
  static SnQuadratureDeviceCache cache;
  return cache;
}

struct SnInnerGraphKey {
  int n_cells = 0;
  int n_groups = 0;
  int n_angles = 0;
  int unroll = 0;
  int geom = 0;
  int n_azim = 0;
  std::size_t sweep_shared_bytes = 0U;
  int sweep_block_threads = 0;
  bool dsa_enabled = false;
  bool linear_characteristic = false;
  double dt = 0.0;
  const void* sigma_a = nullptr;
  const void* sigma_s = nullptr;
  const void* eta = nullptr;
  const void* phi_old = nullptr;
  const void* rad_E_old = nullptr;
  const void* source_work = nullptr;
  const void* sigma_t_work = nullptr;
  const void* x_r = nullptr;
  const void* vol = nullptr;
  const void* mu = nullptr;
  const void* alpha = nullptr;
  const void* tau = nullptr;
  const void* sd_mu = nullptr;
  const void* weight = nullptr;
  const double* psi_prev = nullptr;
  const void* psi = nullptr;
  const void* psi_outgoing = nullptr;
  const void* lc_E = nullptr;
  const void* lc_A = nullptr;
  const void* origin = nullptr;
  const void* outer_marshak_psi_in = nullptr;  // nullptr => vacuum outer BC
  const void* radial_fixup_count = nullptr;
  const void* radial_fixup_artificial_abs = nullptr;
  const void* angular_fixup_count = nullptr;
  const void* angular_fixup_artificial_abs = nullptr;
  const void* phi_sweep = nullptr;
  const void* F_sweep = nullptr;
  const void* Prr = nullptr;
  const void* face_flux_raw = nullptr;
  const void* reduction = nullptr;
  const void* dsa_lower = nullptr;
  const void* dsa_diag = nullptr;
  const void* dsa_upper = nullptr;
  const void* dsa_buffer = nullptr;
};

bool operator==(const SnInnerGraphKey& a, const SnInnerGraphKey& b) {
  return a.n_cells == b.n_cells && a.n_groups == b.n_groups &&
         a.n_angles == b.n_angles && a.unroll == b.unroll &&
         a.geom == b.geom &&
         a.n_azim == b.n_azim &&
         a.sweep_shared_bytes == b.sweep_shared_bytes &&
         a.sweep_block_threads == b.sweep_block_threads &&
         a.dsa_enabled == b.dsa_enabled && a.dt == b.dt &&
         a.linear_characteristic == b.linear_characteristic &&
         a.sigma_a == b.sigma_a && a.sigma_s == b.sigma_s &&
         a.eta == b.eta && a.phi_old == b.phi_old &&
         a.rad_E_old == b.rad_E_old && a.source_work == b.source_work &&
         a.sigma_t_work == b.sigma_t_work && a.x_r == b.x_r &&
         a.vol == b.vol && a.mu == b.mu && a.alpha == b.alpha && a.tau == b.tau &&
         a.sd_mu == b.sd_mu &&
         a.weight == b.weight && a.psi_prev == b.psi_prev &&
         a.psi == b.psi &&
         a.psi_outgoing == b.psi_outgoing &&
         a.lc_E == b.lc_E && a.lc_A == b.lc_A &&
         a.origin == b.origin &&
         a.outer_marshak_psi_in == b.outer_marshak_psi_in &&
         a.radial_fixup_count == b.radial_fixup_count &&
         a.radial_fixup_artificial_abs == b.radial_fixup_artificial_abs &&
         a.angular_fixup_count == b.angular_fixup_count &&
         a.angular_fixup_artificial_abs == b.angular_fixup_artificial_abs &&
         a.phi_sweep == b.phi_sweep &&
         a.F_sweep == b.F_sweep &&
         a.Prr == b.Prr && a.face_flux_raw == b.face_flux_raw &&
         a.reduction == b.reduction &&
         a.dsa_lower == b.dsa_lower && a.dsa_diag == b.dsa_diag &&
         a.dsa_upper == b.dsa_upper && a.dsa_buffer == b.dsa_buffer;
}

struct SnInnerGraphCache {
  ~SnInnerGraphCache() {
    release_graph();
    if (stream != nullptr) {
      static_cast<void>(cudaStreamDestroy(stream));
      stream = nullptr;
    }
  }

  void ensure_stream() {
    if (stream == nullptr) {
      cuda_check(cudaStreamCreate(&stream), "SN inner graph stream create failed");
    }
  }

  void release_graph() {
    if (exec != nullptr) {
      static_cast<void>(cudaGraphExecDestroy(exec));
      exec = nullptr;
    }
    if (graph != nullptr) {
      static_cast<void>(cudaGraphDestroy(graph));
      graph = nullptr;
    }
    has_key = false;
  }

  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  SnInnerGraphKey key;
  bool has_key = false;
  bool capture_disabled = false;
};

SnInnerGraphCache& sn_inner_graph_cache() {
  static SnInnerGraphCache cache;
  return cache;
}

void warn_inner_graph_disabled(const char* reason) {
  static bool warned = false;
  if (!warned) {
    core::log_warning(std::string("SN inner CUDA graph disabled; falling back to "
                                  "streamed source iterations: ") +
                      reason);
    warned = true;
  }
}

SnInnerGraphKey make_inner_graph_key(const core::State& state,
                                     const SnQuadratureDeviceCache& quad,
                                     const int n_cells,
                                     const int n_groups,
                                     const int n_angles,
                                     const int unroll,
                                     const int geom,
                                     const std::size_t sweep_shared_bytes,
                                     const int sweep_block_threads,
                                     const bool dsa_enabled,
                                     const bool linear_characteristic,
                                     const double dt) {
  SnInnerGraphKey key{};
  key.n_cells = n_cells;
  key.n_groups = n_groups;
  key.n_angles = n_angles;
  key.unroll = unroll;
  key.geom = geom;
  key.n_azim = quad.n_azim;
  key.sweep_shared_bytes = sweep_shared_bytes;
  key.sweep_block_threads = sweep_block_threads;
  key.dsa_enabled = dsa_enabled;
  key.linear_characteristic = linear_characteristic;
  key.dt = dt;
  key.sigma_a = state.sn_sigma_a.data();
  key.sigma_s = state.sn_sigma_s.data();
  key.eta = state.sn_eta.data();
  key.phi_old = state.sn_phi_old.data();
  key.rad_E_old = state.rad_E_old.data();
  key.source_work = state.sn_phi_new.data();
  key.sigma_t_work = state.sn_dsa_rhs.data();
  key.x_r = state.x_r.data();
  key.vol = state.vol.data();
  key.mu = quad.mu.data();
  key.alpha = quad.alpha.data();
  key.tau = quad.tau.data();
  key.sd_mu = quad.sd_mu.data();
  key.weight = quad.weight.data();
  key.psi_prev = state.sn_psi_prev.data();
  key.psi = state.sn_psi_scratch.data();
  key.psi_outgoing = state.sn_psi_outgoing_scratch.data();
  key.lc_E = state.sn_lc_E_scratch.data();
  key.lc_A = state.sn_lc_A_scratch.data();
  key.origin = state.sn_origin_boundary.data();
  key.radial_fixup_count = state.sn_radial_fixup_count.data();
  key.radial_fixup_artificial_abs = state.sn_radial_fixup_artificial_abs.data();
  key.angular_fixup_count = state.sn_angular_fixup_count.data();
  key.angular_fixup_artificial_abs = state.sn_angular_fixup_artificial_abs.data();
  key.phi_sweep = state.sn_phi_sweep.data();
  key.F_sweep = state.sn_diag_F_first_moment.data();
  key.Prr = state.sn_Prr.data();
  key.face_flux_raw = state.sn_face_flux_raw.data();
  key.reduction = state.sn_reduction_work.data();
  key.dsa_lower = state.sn_dsa_lower.data();
  key.dsa_diag = state.sn_dsa_diag.data();
  key.dsa_upper = state.sn_dsa_upper.data();
  key.dsa_buffer = state.sn_dsa_cusparse_buffer.data();
  return key;
}

void launch_inner_iteration_body(core::State& state,
                                 const SnQuadratureDeviceCache& quad,
                                 const int n_cells,
                                 const int n_groups,
                                 const int n_angles,
                                 const int total,
                                 const int total_grid,
                                 const int sweep_grid,
                                 const std::size_t sweep_shared_bytes,
                                 const std::size_t total_bytes,
                                 const double dt,
                                 const bool dsa_enabled,
                                 const bool linear_characteristic,
                                 const bool reduce_residual,
                                 cudaStream_t stream,
                                 const bool check_errors,
                                 const double* outer_psi_in,
                                 const int geom) {
  if (total_grid > 0) {
    build_sweep_inputs_kernel<<<total_grid, kBlock, 0, stream>>>(
        state.sn_sigma_a.data(),
        state.sn_sigma_s.data(),
        state.sn_eta.data(),
        state.sn_phi_old.data(),
        state.rad_E_old.data(),
        state.sn_phi_new.data(),
        state.sn_dsa_rhs.data(),
        total);
    if (check_errors) {
      cuda_check(cudaGetLastError(), "SN sweep input build launch failed");
    }
  }
  if (sweep_grid > 0) {
#ifdef TENRYU_DEBUG_LANE_PARALLEL
    (void)linear_characteristic;
    sn_sweep_reflected_pair_kernel<<<sweep_grid, kSweepWarp, 0, stream>>>(
        state.sn_phi_new.data(),
        state.sn_dsa_rhs.data(),
        state.x_r.data(),
        state.vol.data(),
        quad.mu.data(),
        quad.alpha.data(),
        state.sn_psi_prev.data(),
        state.sn_psi_scratch.data(),
        state.sn_origin_boundary.data(),
        n_cells,
        n_groups,
        n_angles,
        dt,
        geom);
    if (check_errors) {
      cuda_check(cudaGetLastError(), "SN reflected-pair sweep launch failed");
    }
#else
    if (geom == 1) {
      // W-G3: 1D cylindrical product-quadrature sweeps (per-level chains).
      if (linear_characteristic) {
        sn_sweep_cylindrical_lc_kernel<<<sweep_grid, kSweepWarp, sweep_shared_bytes, stream>>>(
            state.sn_phi_new.data(),
            state.sn_dsa_rhs.data(),
            state.sn_lc_A_scratch.data(),
            state.x_r.data(),
            state.vol.data(),
            quad.mu.data(),
            quad.weight.data(),
            quad.alpha.data(),
            quad.tau.data(),
            quad.sd_mu.data(),
            state.sn_psi_prev.data(),
            state.sn_psi_scratch.data(),
            state.sn_psi_outgoing_scratch.data(),
            state.sn_origin_boundary.data(),
            outer_psi_in,
            n_cells,
            n_groups,
            n_angles,
            quad.n_azim,
            dt,
            geom);
        if (check_errors) {
          cuda_check(cudaGetLastError(),
                     "SN cylindrical LC sweep launch failed");
        }
        if (total_grid > 0) {
          sn_moments_reduction_k2_kernel<<<total_grid, kBlock, 0, stream>>>(
              state.sn_psi_scratch.data(),
              quad.mu.data(),
              quad.weight.data(),
              state.sn_phi_sweep.data(),
              state.sn_diag_F_first_moment.data(),
              state.sn_Prr.data(),
              n_cells,
              n_groups,
              n_angles);
          if (check_errors) {
            cuda_check(cudaGetLastError(),
                       "SN cylindrical K2 moment reduction launch failed");
          }
        }
        const int face_total = (n_cells + 1) * n_groups;
        const int face_grid = (face_total + kBlock - 1) / kBlock;
        if (face_grid > 0) {
          sn_face_flux_reduction_k2_kernel<<<face_grid, kBlock, 0, stream>>>(
              state.sn_psi_outgoing_scratch.data(),
              quad.mu.data(),
              quad.weight.data(),
              state.sn_face_flux_raw.data(),
              n_cells,
              n_groups,
              n_angles);
          if (check_errors) {
            cuda_check(cudaGetLastError(),
                       "SN cylindrical K2 face-flux reduction launch failed");
          }
        }
      } else {
        sn_sweep_cylindrical_serial_kernel<<<sweep_grid, 1, sweep_shared_bytes, stream>>>(
            state.sn_phi_new.data(),
            state.sn_dsa_rhs.data(),
            state.x_r.data(),
            state.vol.data(),
            quad.mu.data(),
            quad.weight.data(),
            quad.alpha.data(),
            state.sn_psi_prev.data(),
            state.sn_psi_scratch.data(),
            state.sn_phi_sweep.data(),
            state.sn_diag_F_first_moment.data(),
            state.sn_Prr.data(),
            state.sn_face_flux_raw.data(),
            state.sn_origin_boundary.data(),
            outer_psi_in,
            state.sn_radial_fixup_count.data(),
            state.sn_radial_fixup_artificial_abs.data(),
            state.sn_angular_fixup_count.data(),
            state.sn_angular_fixup_artificial_abs.data(),
            n_cells,
            n_groups,
            n_angles,
            quad.n_azim,
            dt,
            geom);
        if (check_errors) {
          cuda_check(cudaGetLastError(),
                     "SN cylindrical serial sweep launch failed");
        }
      }
    } else if (linear_characteristic) {
      sn_sweep_spherical_lc_kernel<<<sweep_grid, kSweepWarp, sweep_shared_bytes, stream>>>(
          state.sn_phi_new.data(),
          state.sn_dsa_rhs.data(),
          state.sn_lc_E_scratch.data(),
          state.sn_lc_A_scratch.data(),
          state.x_r.data(),
          state.vol.data(),
          quad.mu.data(),
          quad.weight.data(),
          quad.alpha.data(),
          quad.tau.data(),
          state.sn_psi_prev.data(),
          state.sn_psi_scratch.data(),
          state.sn_psi_outgoing_scratch.data(),
          state.sn_origin_boundary.data(),
          outer_psi_in,
          n_cells,
          n_groups,
          n_angles,
          dt,
          geom);
      if (check_errors) {
        cuda_check(cudaGetLastError(), "SN spherical LC sweep launch failed");
      }
      if (total_grid > 0) {
        sn_moments_reduction_k2_kernel<<<total_grid, kBlock, 0, stream>>>(
            state.sn_psi_scratch.data(),
            quad.mu.data(),
            quad.weight.data(),
            state.sn_phi_sweep.data(),
            state.sn_diag_F_first_moment.data(),
            state.sn_Prr.data(),
            n_cells,
            n_groups,
            n_angles);
        if (check_errors) {
          cuda_check(cudaGetLastError(), "SN K2 moment reduction launch failed");
        }
      }
      const int face_total = (n_cells + 1) * n_groups;
      const int face_grid = (face_total + kBlock - 1) / kBlock;
      if (face_grid > 0) {
        sn_face_flux_reduction_k2_kernel<<<face_grid, kBlock, 0, stream>>>(
            state.sn_psi_outgoing_scratch.data(),
            quad.mu.data(),
            quad.weight.data(),
            state.sn_face_flux_raw.data(),
            n_cells,
            n_groups,
            n_angles);
        if (check_errors) {
          cuda_check(cudaGetLastError(), "SN K2 face-flux reduction launch failed");
        }
      }
    } else {
      sn_sweep_spherical_serial_kernel<<<sweep_grid, 1, sweep_shared_bytes, stream>>>(
          state.sn_phi_new.data(),
          state.sn_dsa_rhs.data(),
          state.x_r.data(),
          state.vol.data(),
          quad.mu.data(),
          quad.weight.data(),
          quad.alpha.data(),
          state.sn_psi_prev.data(),
          state.sn_psi_scratch.data(),
          state.sn_phi_sweep.data(),
          state.sn_diag_F_first_moment.data(),
          state.sn_Prr.data(),
          state.sn_face_flux_raw.data(),
          state.sn_origin_boundary.data(),
          outer_psi_in,
          state.sn_radial_fixup_count.data(),
          state.sn_radial_fixup_artificial_abs.data(),
          state.sn_angular_fixup_count.data(),
          state.sn_angular_fixup_artificial_abs.data(),
          n_cells,
          n_groups,
          n_angles,
          dt,
          geom);
      if (check_errors) {
        cuda_check(cudaGetLastError(), "SN spherical serial sweep launch failed");
      }
    }
#endif
  }
#ifdef TENRYU_DEBUG_LANE_PARALLEL
  if (total_grid > 0) {
    sn_reduce_moments_from_psi_kernel<<<total_grid, kBlock, 0, stream>>>(
        state.sn_psi_scratch.data(),
        quad.mu.data(),
        quad.weight.data(),
        state.sn_phi_sweep.data(),
        state.sn_diag_F_first_moment.data(),
        state.sn_Prr.data(),
        n_cells,
        n_groups,
        n_angles);
    if (check_errors) {
      cuda_check(cudaGetLastError(), "SN moment reduction launch failed");
    }
  }
#endif
  if (dsa_enabled) {
    apply_sn_dsa_1d_gpu(state, n_cells, n_groups, dt, stream);
  }
  if (reduce_residual) {
    cuda_check(cudaMemsetAsync(state.sn_reduction_work.data(), 0, sizeof(double), stream),
               "SN zero source-iteration residual failed");
    if (total_grid > 0) {
      max_relative_delta_kernel<<<total_grid, kBlock, 0, stream>>>(
          state.sn_phi_sweep.data(),
          state.sn_phi_old.data(),
          state.sn_reduction_work.data(),
          total);
      if (check_errors) {
        cuda_check(cudaGetLastError(), "SN residual reduction launch failed");
      }
    }
  }
  cuda_check(cudaMemcpyAsync(state.sn_phi_old.data(),
                             state.sn_phi_sweep.data(),
                             total_bytes,
                             cudaMemcpyDeviceToDevice,
                             stream),
             "SN copy source-iteration phi failed");
}

void launch_lc_weight_precompute(core::State& state,
                                 const SnQuadratureDeviceCache& quad,
                                 const int n_cells,
                                 const int n_groups,
                                 const int n_angles,
                                 const int total,
                                 const int total_grid,
                                 const double dt,
                                 cudaStream_t stream) {
  if (total_grid > 0) {
    build_sweep_inputs_kernel<<<total_grid, kBlock, 0, stream>>>(
        state.sn_sigma_a.data(),
        state.sn_sigma_s.data(),
        state.sn_eta.data(),
        state.sn_phi_old.data(),
        state.rad_E_old.data(),
        state.sn_phi_new.data(),
        state.sn_dsa_rhs.data(),
        total);
    cuda_check(cudaGetLastError(), "SN LC precompute input build launch failed");
  }
  const int n_psi = total * n_angles;
  const int psi_grid = (n_psi + kBlock - 1) / kBlock;
#ifndef TENRYU_DEBUG_LANE_PARALLEL
  const int geom = state.mesh.geometry_code;
  if (psi_grid > 0) {
    precompute_lc_weights_kernel<<<psi_grid, kBlock, 0, stream>>>(
        state.sn_dsa_rhs.data(),
        state.x_r.data(),
        state.vol.data(),
        quad.mu.data(),
        quad.weight.data(),
        quad.alpha.data(),
        state.sn_lc_E_scratch.data(),
        state.sn_lc_A_scratch.data(),
        n_cells,
        n_groups,
        n_angles,
        dt,
        geom);
    cuda_check(cudaGetLastError(), "SN LC weight precompute launch failed");
  }
#else
  (void)psi_grid;
#endif
}

double copy_reduction_scalar_from_stream(core::State& state, cudaStream_t stream) {
  double value = 0.0;
  cuda_check(cudaMemcpyAsync(&value,
                             state.sn_reduction_work.data(),
                             sizeof(double),
                             cudaMemcpyDeviceToHost,
                             stream),
             "SN copy reduction scalar failed");
  cuda_check(cudaStreamSynchronize(stream), "SN source-iteration stream sync failed");
  return value;
}

bool ensure_inner_graph(SnInnerGraphCache& cache,
                        core::State& state,
                        const SnQuadratureDeviceCache& quad,
                        const int n_cells,
                        const int n_groups,
                        const int n_angles,
                        const int total,
                        const int total_grid,
                        const int sweep_grid,
                        const std::size_t sweep_shared_bytes,
                        const int sweep_block_threads,
                        const std::size_t total_bytes,
                        const double dt,
                        const bool dsa_enabled,
                        const bool linear_characteristic,
                        const int unroll,
                        const double* outer_psi_in,
                        const int geom) {
  if (unroll <= 1 || cache.capture_disabled) {
    return false;
  }
  if (dsa_enabled) {
    ensure_sn_dsa_1d_gpu_buffers(state, n_cells, n_groups);
  }
  cache.ensure_stream();
  const SnInnerGraphKey key = make_inner_graph_key(
      state,
      quad,
      n_cells,
      n_groups,
      n_angles,
      unroll,
      geom,
      sweep_shared_bytes,
      sweep_block_threads,
      dsa_enabled,
      linear_characteristic,
      dt);
  SnInnerGraphKey key_with_bc = key;
  key_with_bc.outer_marshak_psi_in = outer_psi_in;
  if (cache.exec != nullptr && cache.has_key && cache.key == key_with_bc) {
    return true;
  }
  cache.release_graph();
  cudaError_t err = cudaStreamBeginCapture(cache.stream, cudaStreamCaptureModeGlobal);
  if (err != cudaSuccess) {
    cache.capture_disabled = true;
    warn_inner_graph_disabled(cudaGetErrorString(err));
    return false;
  }
  for (int i = 0; i < unroll; ++i) {
    launch_inner_iteration_body(state,
                                quad,
                                n_cells,
                                n_groups,
                                n_angles,
                                total,
                                total_grid,
                                sweep_grid,
                                sweep_shared_bytes,
                                total_bytes,
                                dt,
                                dsa_enabled,
                                linear_characteristic,
                                i + 1 == unroll,
                                cache.stream,
                                false,
                                outer_psi_in,
                                geom);
  }
  err = cudaStreamEndCapture(cache.stream, &cache.graph);
  if (err != cudaSuccess) {
    cache.graph = nullptr;
    cache.capture_disabled = true;
    warn_inner_graph_disabled(cudaGetErrorString(err));
    return false;
  }
  cudaGraphNode_t error_node = nullptr;
  char log_buffer[1024] = {};
  err = cudaGraphInstantiate(
      &cache.exec, cache.graph, &error_node, log_buffer, sizeof(log_buffer));
  if (err != cudaSuccess) {
    cache.release_graph();
    cache.capture_disabled = true;
    warn_inner_graph_disabled(cudaGetErrorString(err));
    return false;
  }
  cache.key = key_with_bc;
  cache.has_key = true;
  return true;
}

void launch_inner_graph(SnInnerGraphCache& cache) {
  cuda_check(cudaGraphLaunch(cache.exec, cache.stream),
             "SN inner source-iteration graph launch failed");
}

}  // namespace

void compute_sn_E_star_flux_1d_gpu(
    const double* const rad_E_old,
    const double* const node_r,
    const double* const vol,
    const double* const face_flux,
    double* const E_star,
    double* const diag_E_star_flux,
    const int n_cells,
    const int n_groups,
    const double dt,
    cudaStream_t stream,
    const int geom) {
  TENRYU_ASSERT(rad_E_old != nullptr, "SN E_star_flux requires rad_E_old");
  TENRYU_ASSERT(node_r != nullptr, "SN E_star_flux requires node_r");
  TENRYU_ASSERT(vol != nullptr, "SN E_star_flux requires vol");
  TENRYU_ASSERT(face_flux != nullptr, "SN E_star_flux requires face_flux");
  TENRYU_ASSERT(E_star != nullptr, "SN E_star_flux requires output");
  TENRYU_ASSERT(n_cells >= 0, "SN E_star_flux requires n_cells >= 0");
  TENRYU_ASSERT(n_groups >= 0, "SN E_star_flux requires n_groups >= 0");
  const int total = n_cells * n_groups;
  if (total == 0) {
    return;
  }
  const int grid = (total + kBlock - 1) / kBlock;
  compute_E_star_flux_kernel<<<grid, kBlock, 0, stream>>>(
      rad_E_old,
      node_r,
      vol,
      face_flux,
      E_star,
      diag_E_star_flux,
      n_cells,
      n_groups,
      dt,
      geom);
  cuda_check(cudaGetLastError(), "SN E_star_flux launch failed");
}

void compute_sn_donor_theta_limited_face_flux_1d_gpu(
    const double* const rad_E_old,
    const double* const node_r,
    const double* const vol,
    const double* const face_flux_raw,
    double* const stream_theta_donor,
    double* const stream_theta,
    double* const face_flux_limited,
    const int n_cells,
    const int n_groups,
    const double dt,
    cudaStream_t stream,
    const int geom) {
  TENRYU_ASSERT(rad_E_old != nullptr, "SN streaming limiter requires rad_E_old");
  TENRYU_ASSERT(node_r != nullptr, "SN streaming limiter requires node_r");
  TENRYU_ASSERT(vol != nullptr, "SN streaming limiter requires vol");
  TENRYU_ASSERT(face_flux_raw != nullptr,
                "SN streaming limiter requires raw face flux");
  TENRYU_ASSERT(stream_theta_donor != nullptr,
                "SN streaming limiter requires donor theta output");
  TENRYU_ASSERT(stream_theta != nullptr,
                "SN streaming limiter requires theta output");
  TENRYU_ASSERT(face_flux_limited != nullptr,
                "SN streaming limiter requires limited face flux output");
  TENRYU_ASSERT(n_cells >= 0,
                "SN streaming limiter requires n_cells >= 0");
  TENRYU_ASSERT(n_groups >= 0,
                "SN streaming limiter requires n_groups >= 0");
  const int total = n_cells * n_groups;
  if (total == 0) {
    return;
  }
  const int cell_grid = (total + kBlock - 1) / kBlock;
  compute_streaming_theta_kernel<<<cell_grid, kBlock, 0, stream>>>(
      rad_E_old,
      node_r,
      vol,
      face_flux_raw,
      stream_theta_donor,
      n_cells,
      n_groups,
      dt,
      geom);
  cuda_check(cudaGetLastError(), "SN streaming theta pass-1 launch failed");
  compute_streaming_theta_credit_kernel<<<cell_grid, kBlock, 0, stream>>>(
      rad_E_old,
      node_r,
      vol,
      face_flux_raw,
      stream_theta_donor,
      stream_theta,
      n_cells,
      n_groups,
      dt,
      geom);
  cuda_check(cudaGetLastError(), "SN streaming theta pass-2 launch failed");
  const int face_total = (n_cells + 1) * n_groups;
  const int face_grid = (face_total + kBlock - 1) / kBlock;
  apply_donor_theta_face_flux_kernel<<<face_grid, kBlock, 0, stream>>>(
      face_flux_raw,
      stream_theta,
      face_flux_limited,
      n_cells,
      n_groups);
  cuda_check(cudaGetLastError(), "SN donor theta face flux launch failed");
}

void compute_sn_diffusion_face_flux_1d_gpu(
    const double* const rad_E_old,
    const double* const sn_sigma_s,
    const double* const node_r,
    double* const face_flux_diff,
    const int n_cells,
    const int n_groups,
    const double opacity_floor,
    cudaStream_t stream) {
  TENRYU_ASSERT(rad_E_old != nullptr, "SN AP diffusion flux requires rad_E_old");
  TENRYU_ASSERT(sn_sigma_s != nullptr, "SN AP diffusion flux requires sn_sigma_s");
  TENRYU_ASSERT(node_r != nullptr, "SN AP diffusion flux requires node_r");
  TENRYU_ASSERT(face_flux_diff != nullptr,
                "SN AP diffusion flux requires output");
  TENRYU_ASSERT(n_cells >= 0, "SN AP diffusion flux requires n_cells >= 0");
  TENRYU_ASSERT(n_groups >= 0, "SN AP diffusion flux requires n_groups >= 0");
  const int total = (n_cells + 1) * n_groups;
  if (total == 0) {
    return;
  }
  const int grid = (total + kBlock - 1) / kBlock;
  compute_diffusion_face_flux_kernel<<<grid, kBlock, 0, stream>>>(
      rad_E_old,
      sn_sigma_s,
      node_r,
      fmax(opacity_floor, 0.0),
      face_flux_diff,
      n_cells,
      n_groups);
  cuda_check(cudaGetLastError(), "SN AP diffusion face flux launch failed");
}

void compute_sn_ap_blended_face_flux_1d_gpu(
    const double* const face_flux_sn,
    const double* const face_flux_diff,
    const double* const rad_E_old,
    const double* const Te,
    const double* const sn_sigma_s,
    const double* const node_r,
    PlanckTableDeviceView planck,
    const double opacity_floor,
    const double tau_lo,
    const double tau_hi,
    const double eq_full,
    const double eq_off,
    const double f_full,
    const double f_off,
    double* const face_flux_blended,
    double* const alpha_face,
    const int n_cells,
    const int n_groups,
    cudaStream_t stream) {
  TENRYU_ASSERT(face_flux_sn != nullptr, "SN AP blend requires SN face flux");
  TENRYU_ASSERT(face_flux_diff != nullptr,
                "SN AP blend requires diffusion face flux");
  TENRYU_ASSERT(rad_E_old != nullptr, "SN AP blend requires rad_E_old");
  TENRYU_ASSERT(Te != nullptr, "SN AP blend requires Te");
  TENRYU_ASSERT(sn_sigma_s != nullptr, "SN AP blend requires sn_sigma_s");
  TENRYU_ASSERT(node_r != nullptr, "SN AP blend requires node_r");
  TENRYU_ASSERT(face_flux_blended != nullptr,
                "SN AP blend requires blended face flux output");
  TENRYU_ASSERT(alpha_face != nullptr, "SN AP blend requires alpha output");
  TENRYU_ASSERT(n_cells >= 0, "SN AP blend requires n_cells >= 0");
  TENRYU_ASSERT(n_groups >= 0, "SN AP blend requires n_groups >= 0");
  const int total = (n_cells + 1) * n_groups;
  if (total == 0) {
    return;
  }
  const int grid = (total + kBlock - 1) / kBlock;
  compute_ap_blended_face_flux_kernel<<<grid, kBlock, 0, stream>>>(
      face_flux_sn,
      face_flux_diff,
      rad_E_old,
      Te,
      sn_sigma_s,
      node_r,
      planck,
      fmax(opacity_floor, 0.0),
      tau_lo,
      tau_hi,
      eq_full,
      eq_off,
      f_full,
      f_off,
      face_flux_blended,
      alpha_face,
      n_cells,
      n_groups);
  cuda_check(cudaGetLastError(), "SN AP blended face flux launch failed");
}

void advance_radiation_step_sn_1d(
    core::State& state,
    const core::Config& cfg,
    const PlanckTable& planck,
    const core::Config::MaterialsConfig::MatDef& mat,
    const double dt) {
  TENRYU_ASSERT(cfg.radiation.mode == core::RadiationMode::SnTransport,
                "advance_radiation_step_sn_1d requires sn_transport mode");
  TENRYU_ASSERT(state.mesh.dim == 1 || cfg.main.dimension == "1D_SPH",
                "advance_radiation_step_sn_1d requires 1D_SPH state");
  TENRYU_ASSERT(dt > 0.0, "advance_radiation_step_sn_1d requires dt > 0");
  TENRYU_ASSERT(!state.rho.empty(), "advance_radiation_step_sn_1d requires cells");
  TENRYU_ASSERT(!state.x_r.empty(),
                "advance_radiation_step_sn_1d requires 1D node radii");
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_groups = std::max(cfg.radiation.groups, 1);
  const auto& sn = cfg.radiation.sn_transport;
  const int n_angles = std::max(sn.n_angles, 2);
  const int geom = state.mesh.geometry_code;
  TENRYU_ASSERT(state.x_r.size() == static_cast<std::size_t>(n_cells + 1),
                "SN requires x_r size n_cells+1");
  TENRYU_ASSERT(planck.n_groups() == n_groups,
                "SN requires Planck table group count to match Radiation.groups");
  // W-G3: 1D cylindrical S_N (product quadrature; design doc
  // docs/design/wg3_sn_cylindrical_quadrature_design.md).
#ifdef TENRYU_DEBUG_LANE_PARALLEL
  TENRYU_ASSERT(geom != 1,
                "cylindrical S_N is not supported under "
                "TENRYU_DEBUG_LANE_PARALLEL");
#endif
  if (geom == 1) {
    const int cyl_levels = sn_cyl_levels_for(n_angles);
    TENRYU_ASSERT(cyl_levels >= 2,
                  "cylindrical S_N requires Radiation.sn_transport.n_angles "
                  "= 2*L*L with integer L >= 2 (8, 18, 32, 50, 72, ...)");
    TENRYU_ASSERT(cyl_levels <= kSweepWarp,
                  "cylindrical S_N polar level count exceeds the sweep warp "
                  "width (n_angles too large)");
  }
  const bool dsa_effective = sn.dsa_enabled && (geom != 1);
  if (sn.dsa_enabled && !dsa_effective) {
    static bool warned_cyl_dsa = false;
    if (!warned_cyl_dsa) {
      core::log_warning(
          "SN DSA acceleration is spherical-only and is disabled for "
          "cylindrical geometry (plain source iteration; the converged "
          "answer is unchanged)");
      warned_cyl_dsa = true;
    }
  }

  const int total = n_cells * n_groups;
  const std::size_t n_psi =
      static_cast<std::size_t>(total) * static_cast<std::size_t>(n_angles);
  ensure_state_buffers(state, n_cells, n_groups, n_angles);
  initialize_rad_E_old_if_needed(state, n_cells, n_groups);

  // Persistent-angular-intensity fix: (re)seed when its size does not
  // match (first step, mesh change, restart) — isotropic from rad_E_old.
  if (state.sn_psi_prev.size() != n_psi) {
    state.sn_psi_prev.reset(n_psi);
    const int seed_grid =
        static_cast<int>((n_psi + kBlock - 1) / static_cast<std::size_t>(kBlock));
    seed_psi_prev_from_rad_E_kernel<<<seed_grid, kBlock>>>(
        state.sn_psi_prev.data(), state.rad_E_old.data(), total, n_angles);
    // 2026-07-26 review: sn_psi_prev is not checkpointed, so a
    // restart (or mesh change) silently replaces the converged angular
    // distribution with an isotropic seed — a one-step artificial-scattering
    // transient. Surface every reseed after step 0 so restarted S_N runs
    // carry the provenance in their log (checkpointing psi_prev is an HDF5
    // schema decision, escalated separately).
    if (state.step > 0) {
      core::log_warning(
          "SN psi_prev reseeded ISOTROPICALLY at step " +
          std::to_string(state.step) +
          " (mesh change or restart): the angular distribution restarts from "
          "phi/2 — expect a one-step isotropization transient");
    }
  }

  // W-B2 (NUMERICS §6.8 1D BC): outer Marshak boundary. Fill the persistent
  // per-group incoming intensity psi_in = 2*F_inc (equilibrium-exact for the
  // GL quadrature) outside any graph capture, and tally the DISCRETE injected
  // flux so the energy ledger balances exactly against the angular escape.
  const double* outer_psi_in = nullptr;
  double S_neg_marshak = 0.0;
  state.sn_marshak_in_step = 0.0;
  if (sn.boundary.outer_r == "marshak") {
    // Same marshak_Tr_eV/table precedence as the FLD 1D Marshak site.
    double T_r_eV = cfg.radiation.boundary.marshak_Tr_eV;
    if (!(T_r_eV > 0.0) && state.marshak_Tr_1d.has_value()) {
      T_r_eV = state.marshak_Tr_1d->eval(state.t);
    }
    double const_flux = sn.marshak.flux_erg_per_cm2_s;
    if (sn.marshak.flux_pulse_duration_s >= 0.0 &&
        state.t >= sn.marshak.flux_pulse_duration_s) {
      const_flux = 0.0;  // pulsed grey drive expired
    }
    std::vector<double> h_mu;
    std::vector<double> h_w;
    if (geom == 1) {
      // W-G3: the marshak ledger's discrete half-range moment must come
      // from the same product set the cylindrical sweep integrates with.
      const SnCylQuadrature1D cyl_quad = build_sn_cyl_quadrature_1d(n_angles);
      h_mu = cyl_quad.mu;
      h_w = cyl_quad.weight;
    } else {
      compute_gauss_legendre(n_angles, h_mu, h_w);
    }
    double S_neg = 0.0;
    for (int m = 0; m < n_angles; ++m) {
      if (h_mu[m] < 0.0) {
        S_neg += h_w[m] * (-h_mu[m]);
      }
    }
    S_neg_marshak = S_neg;
    std::vector<double> psi_in(static_cast<std::size_t>(n_groups), 0.0);
    double F_discrete_total = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      double F_inc = 0.0;
      if (T_r_eV > 0.0) {
        const double T4 = T_r_eV * T_r_eV * T_r_eV * T_r_eV;
        const double b =
            (n_groups == 1)
                ? 1.0
                : std::max(planck.interpolate_b_host(g, T_r_eV), 0.0);
        F_inc = 0.25 * core::constants::c_light * core::constants::a_eV * T4 * b;
      } else if (g == 0) {
        F_inc = std::max(const_flux, 0.0);
      }
      psi_in[static_cast<std::size_t>(g)] = 2.0 * F_inc;
      F_discrete_total += S_neg * psi_in[static_cast<std::size_t>(g)];
    }
    cuda_check(cudaMemcpy(state.sn_outer_marshak_psi_in.data(),
                          psi_in.data(),
                          sizeof(double) * psi_in.size(),
                          cudaMemcpyHostToDevice),
               "SN marshak psi_in upload failed");
    outer_psi_in = state.sn_outer_marshak_psi_in.data();
    double r_outer = 0.0;
    cuda_check(cudaMemcpy(&r_outer,
                          state.x_r.data() + n_cells,
                          sizeof(double),
                          cudaMemcpyDeviceToHost),
               "SN marshak outer radius copy failed");
    state.sn_marshak_in_step =
        dt * tenryu::mesh::geometry_1d_face_area(geom, r_outer) *
        F_discrete_total;
  }

  const int cell_grid = (n_cells + kBlock - 1) / kBlock;
  const int total_grid = (total + kBlock - 1) / kBlock;
  const std::size_t total_bytes = sizeof(double) * static_cast<std::size_t>(total);
  if (cell_grid > 0) {
    copy_kernel<<<cell_grid, kBlock>>>(state.Te.data(), state.sn_Te_old.data(), n_cells);
    cuda_check(cudaGetLastError(), "SN Te snapshot launch failed");
  }
  if (total_grid > 0) {
    initialize_phi_from_rad_E_kernel<<<total_grid, kBlock>>>(
        state.rad_E_old.data(), state.sn_phi_old.data(), total);
    cuda_check(cudaGetLastError(), "SN initialize phi launch failed");
  }

  auto& quad = sn_quadrature_cache();
  quad.ensure(n_angles, geom == 1);
#ifdef TENRYU_DEBUG_SN_METRICS
  check_sn_metric_identity(
      state, quad.mu.data(), quad.weight.data(), quad.alpha.data(), n_cells, n_angles);
#endif

  state.sn_converged = false;
  state.sn_outer_stagnated = false;
  state.sn_outer_residual = std::numeric_limits<double>::infinity();
  state.sn_inner_residual = std::numeric_limits<double>::infinity();
  state.sn_outer_iterations = 0;
  state.sn_inner_iterations = 0;
  state.sn_escaped_step = 0.0;
  state.sn_void_anchor_dE_step = 0.0;
  state.sn_void_anchor_dE_abs_step = 0.0;
  cuda_check(cudaMemset(sn_anchor_tally_device(), 0, 2U * sizeof(double)),
             "SN anchor tally reset failed");

  const int max_outer = std::max(sn.max_outer_iterations, 1);
  const int max_inner = std::max(sn.max_inner_iterations, 1);
  const int graph_unroll = std::max(sn.inner_graph_unroll, 1);
  const double effective_outer_tol =
      std::max(sn.outer_tol, 10.0 * sn.outer_tol_hydro_error_scale);
  const bool linear_characteristic =
      (sn.spatial_scheme == "linear_characteristic");
#ifdef TENRYU_DEBUG_LANE_PARALLEL
  const int sweep_pairs = n_groups * (n_angles / 2);
  const int sweep_grid = (sweep_pairs + kSweepWarp - 1) / kSweepWarp;
  const std::size_t sweep_shared_bytes = 0U;
  const int sweep_block_threads = kSweepWarp;
#else
  const int sweep_grid = n_groups;
  const std::size_t sweep_shared_bytes =
      static_cast<std::size_t>(n_cells + n_angles) * sizeof(double);
  const int sweep_block_threads = linear_characteristic ? kSweepWarp : 1;
#endif
  int step_max_inner_iterations = 0;
  SnTimingAccumulator* timing = nullptr;
  if (sn.timing_enabled) {
    timing = &sn_timing_accumulator();
    timing->begin_step(sn_timing_window());
  }
  auto& graph_cache = sn_inner_graph_cache();
  graph_cache.ensure_stream();
  const cudaStream_t inner_stream = graph_cache.stream;
  zero_step_setup(state, total, nullptr);
  if (total_bytes > 0U) {
    cuda_check(cudaMemcpy(state.sn_diag_rad_E_pre.data(),
                          state.rad_E.data(),
                          total_bytes,
                          cudaMemcpyDeviceToDevice),
               "SN diagnostic rad_E_pre snapshot failed");
  }
  std::vector<double> outer_residual_history;
  outer_residual_history.reserve(static_cast<std::size_t>(max_outer));

  for (int outer = 0; outer < max_outer; ++outer) {
    {
      SnTimingScope timing_scope(timing, SnTimingBlock::OpacityEmission);
      evaluate_opacity_and_emission(state, cfg, planck, mat, n_cells, n_groups, dt);
    }
    if (total_grid > 0) {
      sn_diag_emission_at_Tn_kernel<<<total_grid, kBlock>>>(
          state.sn_eta.data(),
          state.sn_diag_rad_emission_at_Tn.data(),
          total,
          dt);
      cuda_check(cudaGetLastError(), "SN diagnostic emission-at-Tn launch failed");
    }
    zero_face_flux_raw(state, n_cells, n_groups, inner_stream);

    bool inner_converged = false;
    cuda_check(cudaStreamSynchronize(nullptr),
               "SN source iteration wait for opacity/emission failed");
    if (dsa_effective) {
      ensure_sn_dsa_1d_gpu_buffers(state, n_cells, n_groups);
    }
    if (linear_characteristic) {
      launch_lc_weight_precompute(state,
                                  quad,
                                  n_cells,
                                  n_groups,
                                  n_angles,
                                  total,
                                  total_grid,
                                  dt,
                                  inner_stream);
    }
    int inner = 0;
    while (inner < max_inner) {
      const int remaining = max_inner - inner;
      if (remaining >= graph_unroll &&
          ensure_inner_graph(graph_cache,
                             state,
                             quad,
                             n_cells,
                             n_groups,
                             n_angles,
                             total,
                             total_grid,
                             sweep_grid,
                             sweep_shared_bytes,
                             sweep_block_threads,
                             total_bytes,
                             dt,
                             dsa_effective,
                             linear_characteristic,
                             graph_unroll,
                             outer_psi_in,
                             geom)) {
        {
          SnTimingScope timing_scope(timing, SnTimingBlock::Sweep, inner_stream);
          launch_inner_graph(graph_cache);
        }
        state.sn_inner_residual =
            copy_reduction_scalar_from_stream(state, inner_stream);
        inner += graph_unroll;
        state.sn_inner_iterations = inner;
        if (state.sn_inner_residual <= sn.inner_tol) {
          inner_converged = true;
          break;
        }
        continue;
      }
      {
        SnTimingScope timing_scope(timing, SnTimingBlock::Sweep, inner_stream);
        launch_inner_iteration_body(state,
                                    quad,
                                    n_cells,
                                    n_groups,
                                    n_angles,
                                    total,
                                    total_grid,
                                    sweep_grid,
                                    sweep_shared_bytes,
                                    total_bytes,
                                    dt,
                                    dsa_effective,
                                    linear_characteristic,
                                    true,
                                    inner_stream,
                                    true,
                                    outer_psi_in,
                                    geom);
      }
      {
        SnTimingScope timing_scope(timing, SnTimingBlock::Residual, inner_stream);
        state.sn_inner_residual =
            copy_reduction_scalar_from_stream(state, inner_stream);
      }
      ++inner;
      state.sn_inner_iterations = inner;
      if (state.sn_inner_residual <= sn.inner_tol) {
        inner_converged = true;
        break;
      }
    }
    step_max_inner_iterations =
        std::max(step_max_inner_iterations, state.sn_inner_iterations);
    if (!inner_converged && state.sn_inner_iterations >= max_inner) {
      static bool warned_inner_limit = false;
      if (!warned_inner_limit) {
        core::log_warning("SN source iteration reached max_inner_iterations=" +
                          std::to_string(max_inner) +
                          " without satisfying inner_tol=" +
                          std::to_string(sn.inner_tol) +
                          "; residual=" +
                          std::to_string(state.sn_inner_residual));
        warned_inner_limit = true;
      }
    }

    if (total_grid > 0) {
      SnTimingScope timing_scope(timing, SnTimingBlock::PhiToEChi);
      phi_to_rad_E_kernel<<<total_grid, kBlock>>>(
          state.sn_phi_old.data(), state.rad_E.data(), total);
      cuda_check(cudaGetLastError(), "SN phi-to-E launch failed");
      sn_chi_kernel<<<total_grid, kBlock>>>(
          state.rad_E.data(), state.sn_Prr.data(), state.sn_chi.data(), total);
      cuda_check(cudaGetLastError(), "SN chi launch failed");
      const int face_grid = (n_groups + kBlock - 1) / kBlock;
      apply_face_flux_boundary_kernel<<<face_grid, kBlock>>>(
          state.sn_face_flux_raw.data(),
          outer_psi_in,
          S_neg_marshak,
          n_cells,
          n_groups);
      cuda_check(cudaGetLastError(), "SN face-flux boundary launch failed");
      compute_sn_diffusion_face_flux_1d_gpu(state.rad_E_old.data(),
                                            state.sn_sigma_s.data(),
                                            state.x_r.data(),
                                            state.sn_face_flux_diff.data(),
                                            n_cells,
                                            n_groups,
                                            sn.opacity_floor);
      compute_sn_ap_blended_face_flux_1d_gpu(
          state.sn_face_flux_raw.data(),
          state.sn_face_flux_diff.data(),
          state.rad_E_old.data(),
          state.Te.data(),
          state.sn_sigma_s.data(),
          state.x_r.data(),
          planck.device_view(),
          sn.opacity_floor,
          10.0,
          20.0,
          0.05,
          0.10,
          0.15,
          0.25,
          state.sn_face_flux_blended.data(),
          state.sn_face_alpha.data(),
          n_cells,
          n_groups);
      compute_sn_donor_theta_limited_face_flux_1d_gpu(
          state.rad_E_old.data(),
          state.x_r.data(),
          state.vol.data(),
          state.sn_face_flux_blended.data(),
          state.sn_stream_theta_donor.data(),
          state.sn_stream_theta.data(),
          state.sn_face_flux_limited.data(),
          n_cells,
          n_groups,
          dt,
          nullptr,
          geom);
      compute_sn_E_star_flux_1d_gpu(state.rad_E_old.data(),
                                    state.x_r.data(),
                                    state.vol.data(),
                                    state.sn_face_flux_limited.data(),
                                    state.sn_E_star_flux.data(),
                                    state.sn_diag_E_star_flux.data(),
                                    n_cells,
                                    n_groups,
                                    dt,
                                    nullptr,
                                    geom);
    }

    SnMaterialNewton1DInputs newton{};
    newton.sigma_a = state.sn_sigma_a.data();
    newton.sigma_pe = state.sn_sigma_pe.data();
    newton.rad_E = state.rad_E.data();
    newton.rho = state.rho.data();
    newton.cv_e =
        (state.cv_e.size() == static_cast<std::size_t>(n_cells)) ? state.cv_e.data()
                                                                 : nullptr;
    newton.vol = state.vol.data();
    newton.zbar = state.zbar.data();
    newton.Te_old = state.sn_Te_old.data();
    newton.Te = state.Te.data();
    newton.ee = state.ee.data();
    newton.Pe = state.Pe.data();
    newton.rad_dep = state.rad_dep.data();
    newton.rad_emit = state.rad_emit.data();
    newton.diag_rad_emission_at_Tnp1 =
        state.sn_diag_rad_emission_at_Tnp1.data();
    newton.diag_rad_absorption = state.sn_diag_rad_absorption.data();
    newton.diag_clip_energy = state.sn_diag_clip_energy.data();
    newton.diag_clip_full_deficit = state.sn_diag_clip_full_deficit.data();
    newton.delta_T_rel = state.sn_delta_T.data();
    newton.planck = planck.device_view();
    if (mat.hydro_eos_backend != "exact_ideal_gas") {
      newton.electron_eos = sn_electron_eos_device_view(mat.eos_tables.get());
    }
    // Shared per-cell effective properties (multi-material fix; radiation can run
    // with hydro disabled, so ensure here).
    state.ensure_cell_material_props(cfg);
    newton.n_cells = n_cells;
    newton.n_groups = n_groups;
    newton.dt = dt;
    newton.Cv_e_const = mat.cv_e_override;
    newton.cv_e_const = 0.0;
    newton.A_eff = state.A_eff.data();
    newton.gamma_eff = state.gamma_eff.data();
    newton.temperature_floor_eV = cfg.numerics.floors.Te;
    newton.rad_E_out = state.rad_E.data();
    newton.E_star_override = state.sn_E_star_flux.data();
    newton.eos_low_density_extrap = cfg.materials.low_density_extrapolation;
    newton.energy_authoritative =
        (cfg.numerics.hydro.eos_closure_mode == "energy_authoritative");
    {
      SnTimingScope timing_scope(timing, SnTimingBlock::Newton);
      const std::size_t rad_E_bytes = sizeof(double) *
                                      static_cast<std::size_t>(n_cells) *
                                      static_cast<std::size_t>(n_groups);
      cuda_check(cudaMemcpy(state.sn_rad_E_pre_newton.data(),
                            state.rad_E.data(),
                            rad_E_bytes,
                            cudaMemcpyDeviceToDevice),
                 "snapshot rad_E to sn_rad_E_pre_newton failed");
      state.sn_material_retry_flag |=
          solve_sn_material_temperature_newton_gpu(newton);
    }

    if (total_grid > 0) {
      anchor_void_rad_E_to_moments_kernel<<<total_grid, kBlock>>>(
          state.rad_E.data(), state.sn_phi_old.data(),
          state.sn_sigma_a.data(), state.sn_sigma_s.data(), state.vol.data(),
          sn_anchor_tally_device(), total, n_groups, dt);
      cuda_check(cudaGetLastError(), "SN void rad_E anchor launch failed");
    }

    {
      SnTimingScope timing_scope(timing, SnTimingBlock::Residual);
      state.sn_outer_residual =
          reduce_relative_delta(state, state.Te.data(), state.sn_Te_old.data(), n_cells);
      if (cell_grid > 0) {
        copy_kernel<<<cell_grid, kBlock>>>(state.Te.data(), state.sn_Te_old.data(), n_cells);
        cuda_check(cudaGetLastError(), "SN update Te snapshot launch failed");
      }
    }
    state.sn_outer_iterations = outer + 1;
    outer_residual_history.push_back(state.sn_outer_residual);
    bool stagnated = false;
    if (state.sn_outer_iterations >= 5 && outer_residual_history.size() >= 3U) {
      const double previous =
          outer_residual_history[outer_residual_history.size() - 3U];
      if (std::isfinite(previous) && previous > 0.0 &&
          std::isfinite(state.sn_outer_residual)) {
        stagnated =
            (state.sn_outer_residual / previous) > sn.outer_tol_stagnation_factor;
      }
    }
    if (state.sn_outer_residual <= effective_outer_tol || stagnated) {
      if (stagnated && !(state.sn_outer_residual <= effective_outer_tol)) {
        state.sn_outer_stagnated = true;  // 2D sets this; 1D silently omitted it
        // 2026-07-26 review: a stagnation exit is a heuristic
        // acceptance above outer_tol, not convergence. The acceptance policy
        // itself is unchanged (a policy change is a user decision); this
        // makes every such exit visible so production runs can audit how
        // often the Picard loop stopped on the stagnation ratio.
        static int sn_stagnation_exit_count = 0;
        ++sn_stagnation_exit_count;
        if (sn_stagnation_exit_count <= 5 ||
            sn_stagnation_exit_count % 1000 == 0) {
          core::log_warning(
              "SN outer Picard accepted by stagnation heuristic: residual=" +
              std::to_string(state.sn_outer_residual) + " > outer_tol=" +
              std::to_string(effective_outer_tol) + " (occurrence #" +
              std::to_string(sn_stagnation_exit_count) + ")");
        }
      }
      state.sn_converged = true;
      break;
    }
  }

  {
    double h_anchor_tally[2] = {0.0, 0.0};
    cuda_check(cudaMemcpy(h_anchor_tally, sn_anchor_tally_device(),
                          sizeof(h_anchor_tally), cudaMemcpyDeviceToHost),
               "SN anchor tally D2H failed");
    state.sn_void_anchor_dE_step = h_anchor_tally[0];
    state.sn_void_anchor_dE_abs_step = h_anchor_tally[1];
    if (h_anchor_tally[1] > 0.0) {
      static int sn_anchor_note_count = 0;
      ++sn_anchor_note_count;
      if (sn_anchor_note_count <= 3 || sn_anchor_note_count % 1000 == 0) {
        core::log_info(
            "SN void anchor ledger: step signed dE=" +
            std::to_string(h_anchor_tally[0]) + " erg, |dE| sum=" +
            std::to_string(h_anchor_tally[1]) +
            " erg (nonconservative moment anchor engaged; occurrence #" +
            std::to_string(sn_anchor_note_count) + ")");
      }
    }
  }

  // Persistent-angular-intensity fix: converged psi^{n+1} becomes the next step's time-source memory.
  cuda_check(cudaMemcpyAsync(state.sn_psi_prev.data(),
                             state.sn_psi_scratch.data(),
                             n_psi * sizeof(double), cudaMemcpyDeviceToDevice,
                             inner_stream),
             "SN: psi_prev copy failed");
  if (!state.sn_converged && state.sn_outer_iterations >= max_outer) {
    static bool warned_outer_limit = false;
    if (!warned_outer_limit) {
      core::log_warning("SN Picard iteration reached max_outer_iterations=" +
                        std::to_string(max_outer) +
                        " without satisfying outer_tol=" +
                        std::to_string(effective_outer_tol) +
                        "; residual=" +
                        std::to_string(state.sn_outer_residual));
      warned_outer_limit = true;
    }
  }

  compute_opacity_lag_diagnostic(state, cfg, planck, mat, n_cells, n_groups, dt);
  if (total_bytes > 0U) {
    cuda_check(cudaMemcpy(state.sn_diag_rad_E_post.data(),
                          state.rad_E.data(),
                          total_bytes,
                          cudaMemcpyDeviceToDevice),
               "SN diagnostic rad_E_post snapshot failed");
  }

  {
    SnTimingScope timing_scope(timing, SnTimingBlock::EscapedRadEOld);
    state.sn_escaped_step = compute_escaped_energy(state, n_cells, n_groups, dt);
    // Outer-boundary escape-flux follow-up (2026-07-20): the driver energy budget pairs +marshak_in
    // (source) with +escape (sink), which requires GROSS conventions on both.
    // The face ledger is NET at a Marshak boundary (outgoing - discrete inflow),
    // so add the inflow back: escape reports the gross outgoing energy and the
    // budget identity closes exactly (the same scalar cancels on both sides).
    // Vacuum configs add +0.0 (bit-identical).
    state.sn_escaped_step += state.sn_marshak_in_step;
    copy_rad_E_to_old(state, n_cells, n_groups);
  }
  if (timing != nullptr) {
    timing->record_iterations(step_max_inner_iterations, state.sn_outer_iterations);
    timing->end_step();
  }
  state.holo_ale_invalidated = false;
}

}  // namespace tenryu::radiation
