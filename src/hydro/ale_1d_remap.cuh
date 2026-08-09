#pragma once

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/ale_1d_rezone.cuh"
#include "hydro/ale_1d_types.cuh"

namespace tenryu::hydro::ale1d {

template <typename T>
class DeviceArray {
 public:
  DeviceArray() = default;

  explicit DeviceArray(const std::size_t count) {
    resize(count);
  }

  ~DeviceArray() {
    release();
  }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  DeviceArray(DeviceArray&& other) noexcept {
    ptr_ = other.ptr_;
    size_ = other.size_;
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  DeviceArray& operator=(DeviceArray&& other) noexcept {
    if (this != &other) {
      release();
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  void resize(const std::size_t count) {
    if (count == size_) {
      return;
    }
    release();
    size_ = count;
    if (size_ == 0) {
      return;
    }
    const cudaError_t alloc_err =
        cudaMalloc(reinterpret_cast<void**>(&ptr_), size_ * sizeof(T));
    TENRYU_ASSERT(alloc_err == cudaSuccess,
                  "ALE1D DeviceArray cudaMalloc failed");
    const cudaError_t memset_err = cudaMemset(ptr_, 0, size_ * sizeof(T));
    TENRYU_ASSERT(memset_err == cudaSuccess,
                  "ALE1D DeviceArray cudaMemset failed");
  }

  [[nodiscard]] std::size_t size() const noexcept {
    return size_;
  }

  [[nodiscard]] bool empty() const noexcept {
    return size_ == 0;
  }

  T* data() noexcept {
    return ptr_;
  }

  const T* data() const noexcept {
    return ptr_;
  }

  void copy_from_host(const std::vector<T>& src) {
    TENRYU_ASSERT(src.size() == size_,
                  "ALE1D DeviceArray::copy_from_host size mismatch");
    if (size_ == 0) {
      return;
    }
    const cudaError_t err =
        cudaMemcpy(ptr_, src.data(), size_ * sizeof(T), cudaMemcpyHostToDevice);
    TENRYU_ASSERT(err == cudaSuccess,
                  "ALE1D DeviceArray::copy_from_host cudaMemcpy failed");
  }

  void copy_to_host(std::vector<T>& dst) const {
    dst.resize(size_);
    if (size_ == 0) {
      return;
    }
    const cudaError_t err =
        cudaMemcpy(dst.data(), ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
    TENRYU_ASSERT(err == cudaSuccess,
                  "ALE1D DeviceArray::copy_to_host cudaMemcpy failed");
  }

 private:
  void release() {
    if (ptr_ != nullptr) {
      const cudaError_t err = cudaFree(ptr_);
      TENRYU_ASSERT(err == cudaSuccess, "ALE1D DeviceArray cudaFree failed");
      ptr_ = nullptr;
    }
    size_ = 0;
  }

  T* ptr_ = nullptr;
  std::size_t size_ = 0;
};

using Ale1dRemapConfig =
    core::Config::NumericsConfig::Ale1dConfig::RemapConfig;

struct Ale1dRemapScratch {
  DeviceArray<double> mass_new;
  DeviceArray<double> ee_new;
  DeviceArray<double> ei_new;
  DeviceArray<double> ke_remap;
  DeviceArray<double> rad_E_new;
  DeviceArray<double> volFrac_new;
  DeviceArray<double> vol_new;

  DeviceArray<double> delta_Y;
  DeviceArray<double> phi_face;
  DeviceArray<int> donor;
  DeviceArray<int> fallback_flags;

  void resize(int n_cells,
              int n_groups,
              int n_materials,
              bool ke_conservation_closure = false);
  [[nodiscard]] bool size_matches(int n_cells,
                                  int n_groups,
                                  int n_materials,
                                  bool ke_conservation_closure = false) const;
};

struct Ale1dRemapResult {
  bool success = false;
  Ale1dSkipReason skip_reason = Ale1dSkipReason::None;
  double mass_conservation_rel_err = 0.0;
  double material_mass_conservation_rel_err = 0.0;
  double ee_conservation_rel_err = 0.0;
  double ei_conservation_rel_err = 0.0;
  double radiation_conservation_rel_err = 0.0;
  int n_invalid_sweeps = 0;
  int n_bound_fallback_cells = 0;
  int n_bound_fallback_fields = 0;
};

Ale1dRemapResult remap_v3(
    const core::State& state,
    const core::Config& cfg,
    const std::vector<double>& r_candidate,
    const NodeConstraintMask& node_mask,
    const std::vector<int>& additional_protected_faces,
    Ale1dRemapScratch& scratch);

Ale1dRemapResult remap_first_order(const core::State& state,
                                   const core::Config& cfg,
                                   const std::vector<double>& r_candidate,
                                   const NodeConstraintMask& node_mask,
                                   Ale1dRemapScratch& scratch);

}  // namespace tenryu::hydro::ale1d
