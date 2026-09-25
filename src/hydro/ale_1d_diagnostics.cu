#include "hydro/ale_1d_diagnostics.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <cub/cub.cuh>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/fancy_iterators.cuh"

namespace tenryu::hydro::ale1d {
namespace {

constexpr double kTiny = 1.0e-300;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

// Work buffer from the persistent device scratch pool: one buffer per tag,
// kept across ALE attempts instead of a cudaMalloc/cudaFree pair per call
// (cudaFree synchronizes the device). Contents are not zeroed, as with
// cudaMalloc; two live buffers never share a tag.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  DeviceBuffer(const char* tag, const std::size_t count) {
    reset(tag, count);
  }

  void reset(const char* tag, const std::size_t count) {
    ptr_ = (count == 0)
               ? nullptr
               : static_cast<T*>(core::device_scratch_acquire(tag, count * sizeof(T)));
  }

  T* data() noexcept {
    return ptr_;
  }

  const T* data() const noexcept {
    return ptr_;
  }

 private:
  T* ptr_ = nullptr;
};

template <typename InputIt>
double reduce_sum(InputIt input,
                  const int n,
                  cudaStream_t stream,
                  const char* label) {
  if (n <= 0) {
    return 0.0;
  }
  DeviceBuffer<double> out("ale1d_diagnostics:reduce_sum:out", 1);
  std::size_t temp_bytes = 0;
  cuda_check(cub::DeviceReduce::Sum(nullptr, temp_bytes, input, out.data(), n,
                                    stream),
             label);
  DeviceBuffer<unsigned char> temp("ale1d_diagnostics:reduce_sum:temp", temp_bytes);
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

struct OldMaterialEnergyOp {
  const double* mass = nullptr;
  const double* ee = nullptr;
  const double* ei = nullptr;

  __host__ __device__ double operator()(const int i) const {
    return mass[i] * (ee[i] + ei[i]);
  }
};

struct NewMaterialEnergyOp {
  const double* mass = nullptr;
  const double* ee = nullptr;
  const double* ei = nullptr;

  __host__ __device__ double operator()(const int i) const {
    return mass[i] * (ee[i] + ei[i]);
  }
};

struct OldRadiationEnergyOp {
  const double* rad_E = nullptr;
  const double* vol = nullptr;
  int n_groups = 0;

  __host__ __device__ double operator()(const int idx) const {
    const int i = idx / n_groups;
    return rad_E[idx] * vol[i];
  }
};

struct NewRadiationEnergyOp {
  const double* rad_E = nullptr;
  const double* vol = nullptr;
  int n_groups = 0;

  __host__ __device__ double operator()(const int idx) const {
    const int i = idx / n_groups;
    return rad_E[idx] * vol[i];
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

bool exceeds(const double value, const double hard) {
  return !(std::isfinite(value)) || value > hard;
}

}  // namespace

Ale1dDiagnosticsResult compute_diagnostics(
    const core::State& state,
    const core::Config& cfg,
    const Ale1dRemapScratch& remap_scratch,
    const Ale1dRemapResult& remap_result,
    const Ale1dVelocityProjectResult& velocity_result) {
  Ale1dDiagnosticsResult out;
  const int n = static_cast<int>(remap_scratch.mass_new.size());
  cudaStream_t stream = nullptr;

  out.mass_conservation_rel_err = remap_result.mass_conservation_rel_err;
  out.material_internal_energy_component_rel_err =
      std::max(remap_result.ee_conservation_rel_err,
               remap_result.ei_conservation_rel_err);
  out.radiation_conservation_rel_err =
      remap_result.radiation_conservation_rel_err;
  out.kinetic_energy_drift_rel = velocity_result.kinetic_energy_drift_rel;

  double old_material_energy = 0.0;
  double new_material_energy = 0.0;
  if (n > 0) {
    old_material_energy = reduce_transformed_sum(
        n, OldMaterialEnergyOp{state.mass.data(), state.ee.data(), state.ei.data()},
        stream, "ALE1D diagnostics old material energy reduction failed");
    new_material_energy = reduce_transformed_sum(
        n,
        NewMaterialEnergyOp{remap_scratch.mass_new.data(),
                            remap_scratch.ee_new.data(),
                            remap_scratch.ei_new.data()},
        stream, "ALE1D diagnostics new material energy reduction failed");
    out.material_internal_energy_rel_err =
        relative_error(old_material_energy, new_material_energy);
  }

  double old_radiation_energy = 0.0;
  double new_radiation_energy = 0.0;
  const int n_groups = std::max(0, cfg.radiation.groups);
  if (n > 0 && n_groups > 0 && state.rad_E.size() >= static_cast<std::size_t>(n * n_groups) &&
      remap_scratch.rad_E_new.size() >= static_cast<std::size_t>(n * n_groups)) {
    old_radiation_energy = reduce_transformed_sum(
        n * n_groups,
        OldRadiationEnergyOp{state.rad_E.data(), state.vol.data(), n_groups},
        stream, "ALE1D diagnostics old radiation energy reduction failed");
    new_radiation_energy = reduce_transformed_sum(
        n * n_groups,
        NewRadiationEnergyOp{remap_scratch.rad_E_new.data(),
                             remap_scratch.vol_new.data(),
                             n_groups},
        stream, "ALE1D diagnostics new radiation energy reduction failed");
  }

  const double old_total_energy = old_material_energy + old_radiation_energy +
                                  velocity_result.kinetic_energy_old;
  const double new_total_energy = new_material_energy + new_radiation_energy +
                                  velocity_result.kinetic_energy_new;
  out.global_total_energy_rel_err =
      relative_error(old_total_energy, new_total_energy);

  const auto& ale = cfg.numerics.ale1d;
  out.hard_tolerance_passed =
      !exceeds(out.mass_conservation_rel_err, ale.total_mass_tol.hard) &&
      !exceeds(remap_result.material_mass_conservation_rel_err,
               ale.material_mass_tol.hard) &&
      !exceeds(out.material_internal_energy_component_rel_err,
               ale.material_internal_energy_tol.hard) &&
      !exceeds(out.material_internal_energy_rel_err,
               ale.total_material_energy_tol.hard) &&
      !exceeds(out.global_total_energy_rel_err,
               ale.global_total_energy_tol.hard) &&
      !exceeds(out.radiation_conservation_rel_err,
               ale.radiation_group_energy_tol.hard) &&
      !exceeds(out.kinetic_energy_drift_rel, ale.kinetic_energy_drift_tol.hard);

  return out;
}

}  // namespace tenryu::hydro::ale1d
