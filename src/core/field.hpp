#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/device_scratch.hpp"

namespace tenryu::core {

struct CellTag {};
struct NodeTag {};
struct GroupTag {};

template <typename T>
class DeviceArray {
 public:
  DeviceArray() = default;

  explicit DeviceArray(const char* pool_tag) : pool_tag_(pool_tag) {}

  explicit DeviceArray(const std::size_t n) {
    reset(n);
  }

  ~DeviceArray() {
    release();
  }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  DeviceArray(DeviceArray&& other) noexcept {
    steal_from(other);
  }

  DeviceArray& operator=(DeviceArray&& other) noexcept {
    if (this != &other) {
      release();
      steal_from(other);
    }
    return *this;
  }

  [[nodiscard]] std::size_t size() const noexcept {
    return size_;
  }

  [[nodiscard]] bool empty() const noexcept {
    return size_ == 0;
  }

  T* data() noexcept {
    return data_;
  }

  const T* data() const noexcept {
    return data_;
  }

  void reset(const std::size_t n) {
    if (n == size_) {
      return;
    }

    if (pool_tag_ != nullptr) {
      if (n == 0) {
        data_ = nullptr;
        size_ = 0;
        return;
      }
      data_ = static_cast<T*>(device_scratch_acquire(pool_tag_, n * sizeof(T)));
      cudaError_t err = cudaMemset(data_, 0, n * sizeof(T));
      TENRYU_ASSERT(err == cudaSuccess, "DeviceArray::reset cudaMemset failed");
      size_ = n;
      return;
    }

    release();
    if (n == 0) {
      return;
    }

    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&data_), n * sizeof(T));
    TENRYU_ASSERT(err == cudaSuccess, "DeviceArray::reset cudaMalloc failed");
    err = cudaMemset(data_, 0, n * sizeof(T));
    TENRYU_ASSERT(err == cudaSuccess, "DeviceArray::reset cudaMemset failed");
    size_ = n;
  }

  void copy_from_host(const T* src) {
    TENRYU_ASSERT(src != nullptr || size_ == 0,
                  "DeviceArray::copy_from_host source pointer must not be null");
    if (size_ == 0) {
      return;
    }

    const cudaError_t err =
        cudaMemcpy(data_, src, size_ * sizeof(T), cudaMemcpyHostToDevice);
    TENRYU_ASSERT(err == cudaSuccess,
                  "DeviceArray::copy_from_host cudaMemcpy H2D failed");
  }

  void copy_from_host(const std::vector<T>& src) {
    TENRYU_ASSERT(src.size() == size_,
                  "DeviceArray::copy_from_host vector size mismatch");
    copy_from_host(src.data());
  }

  void copy_to_host(std::vector<T>& dst) const {
    dst.resize(size_);
    if (size_ == 0) {
      return;
    }

    const cudaError_t err =
        cudaMemcpy(dst.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
    TENRYU_ASSERT(err == cudaSuccess,
                  "DeviceArray::copy_to_host cudaMemcpy D2H failed");
  }

  void copy_from_device(const DeviceArray<T>& src) {
    reset(src.size_);
    if (size_ == 0) {
      return;
    }

    const cudaError_t err =
        cudaMemcpy(data_, src.data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
    TENRYU_ASSERT(err == cudaSuccess,
                  "DeviceArray::copy_from_device cudaMemcpy D2D failed");
  }

 private:
  void release() {
    if (pool_tag_ != nullptr) {
      data_ = nullptr;
      size_ = 0;
      return;
    }
    if (data_ != nullptr) {
      const cudaError_t err = cudaFree(data_);
      TENRYU_ASSERT(err == cudaSuccess, "DeviceArray::release cudaFree failed");
      data_ = nullptr;
    }
    size_ = 0;
  }

  void steal_from(DeviceArray& other) noexcept {
    data_ = other.data_;
    size_ = other.size_;
    pool_tag_ = other.pool_tag_;
    other.data_ = nullptr;
    other.size_ = 0;
    other.pool_tag_ = nullptr;
  }

  T* data_ = nullptr;
  std::size_t size_ = 0;
  const char* pool_tag_ = nullptr;
};

template <typename Tag>
class Field1D {
 public:
  using value_type = double;

  Field1D() = default;

  explicit Field1D(const char* pool_tag) : pool_tag_(pool_tag) {}

  explicit Field1D(const std::size_t n) {
    reset(n);
  }

  ~Field1D() {
    release();
  }

  Field1D(const Field1D&) = delete;
  Field1D& operator=(const Field1D&) = delete;

  Field1D(Field1D&& other) noexcept {
    steal_from(other);
  }

  Field1D& operator=(Field1D&& other) noexcept {
    if (this != &other) {
      release();
      steal_from(other);
    }
    return *this;
  }

  [[nodiscard]] std::size_t size() const noexcept {
    return size_;
  }

  [[nodiscard]] bool empty() const noexcept {
    return size_ == 0;
  }

  double* data() noexcept {
    return data_;
  }

  const double* data() const noexcept {
    return data_;
  }

  void reset(const std::size_t n) {
    if (n == size_) {
      return;
    }

    if (pool_tag_ != nullptr) {
      if (n == 0) {
        data_ = nullptr;
        size_ = 0;
        return;
      }
      data_ = static_cast<double*>(device_scratch_acquire(pool_tag_, n * sizeof(double)));
      cudaError_t err = cudaMemset(data_, 0, n * sizeof(double));
      TENRYU_ASSERT(err == cudaSuccess, "Field1D::reset cudaMemset failed");
      size_ = n;
      return;
    }

    release();
    if (n == 0) {
      return;
    }

    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&data_), n * sizeof(double));
    TENRYU_ASSERT(err == cudaSuccess, "Field1D::reset cudaMalloc failed");
    err = cudaMemset(data_, 0, n * sizeof(double));
    TENRYU_ASSERT(err == cudaSuccess, "Field1D::reset cudaMemset failed");
    size_ = n;
  }

  void resize(const std::size_t n) {
    TENRYU_ASSERT(pool_tag_ == nullptr,
                  "Field1D::resize is not supported on pooled fields (prefix-preserving grow "
                  "conflicts with the grow-only scratch pool)");
    if (n == size_) {
      return;
    }
    if (n == 0) {
      release();
      return;
    }

    double* new_data = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&new_data), n * sizeof(double));
    TENRYU_ASSERT(err == cudaSuccess, "Field1D::resize cudaMalloc failed");

    const std::size_t copy_count = std::min(size_, n);
    if (copy_count > 0) {
      err = cudaMemcpy(new_data, data_, copy_count * sizeof(double),
                       cudaMemcpyDeviceToDevice);
      TENRYU_ASSERT(err == cudaSuccess, "Field1D::resize cudaMemcpy D2D failed");
    }

    if (n > copy_count) {
      err = cudaMemset(new_data + copy_count, 0, (n - copy_count) * sizeof(double));
      TENRYU_ASSERT(err == cudaSuccess, "Field1D::resize cudaMemset failed");
    }

    if (data_ != nullptr) {
      err = cudaFree(data_);
      TENRYU_ASSERT(err == cudaSuccess, "Field1D::resize cudaFree failed");
    }
    data_ = new_data;
    size_ = n;
  }

  void fill(const double value) {
    if (size_ == 0) {
      return;
    }

    std::vector<double> host(size_, value);
    copy_from_host(host.data());
  }

  void copy_to_host(double* dst) const {
    TENRYU_ASSERT(dst != nullptr || size_ == 0,
                  "Field1D::copy_to_host destination pointer must not be null");
    if (size_ == 0) {
      return;
    }

    const cudaError_t err =
        cudaMemcpy(dst, data_, size_ * sizeof(double), cudaMemcpyDeviceToHost);
    TENRYU_ASSERT(err == cudaSuccess, "Field1D::copy_to_host cudaMemcpy D2H failed");
  }

  void copy_to_host(std::vector<double>& dst) const {
    dst.resize(size_);
    copy_to_host(dst.data());
  }

  void copy_from_host(const double* src) {
    TENRYU_ASSERT(src != nullptr || size_ == 0,
                  "Field1D::copy_from_host source pointer must not be null");
    if (size_ == 0) {
      return;
    }

    const cudaError_t err =
        cudaMemcpy(data_, src, size_ * sizeof(double), cudaMemcpyHostToDevice);
    TENRYU_ASSERT(err == cudaSuccess, "Field1D::copy_from_host cudaMemcpy H2D failed");
  }

  void copy_from_host(const std::vector<double>& src) {
    TENRYU_ASSERT(src.size() == size_, "Field1D::copy_from_host vector size mismatch");
    copy_from_host(src.data());
  }

  Field1D& operator=(const std::vector<double>& values) {
    reset(values.size());
    copy_from_host(values.data());
    return *this;
  }

 private:
  void release() {
    if (pool_tag_ != nullptr) {
      data_ = nullptr;
      size_ = 0;
      return;
    }
    if (data_ != nullptr) {
      const cudaError_t err = cudaFree(data_);
      TENRYU_ASSERT(err == cudaSuccess, "Field1D::release cudaFree failed");
      data_ = nullptr;
    }
    size_ = 0;
  }

  void steal_from(Field1D& other) noexcept {
    data_ = other.data_;
    size_ = other.size_;
    pool_tag_ = other.pool_tag_;
    other.data_ = nullptr;
    other.size_ = 0;
    other.pool_tag_ = nullptr;
  }

  double* data_ = nullptr;
  std::size_t size_ = 0;
  const char* pool_tag_ = nullptr;
};

// Pool-backed per-step scratch field. Same observable contract as constructing a
// fresh Field1D and calling reset(n): the buffer is zero-filled on every reset().
// Storage comes from the persistent device scratch pool (grow-only, tag-keyed), so
// steady-state steps perform no cudaMalloc/cudaFree. Non-owning: the pool owns the
// allocation; destruction releases nothing. The tag must be a stable string literal
// unique per call site (pool contract) — two live ScratchField1D with the same tag
// would alias the same storage. Not copyable or movable: these are strictly
// function-local step temporaries (State-owned fields stay Field1D).
template <typename Tag>
class ScratchField1D {
 public:
  using value_type = double;

  ScratchField1D() = default;
  ~ScratchField1D() = default;
  ScratchField1D(const ScratchField1D&) = delete;
  ScratchField1D& operator=(const ScratchField1D&) = delete;
  ScratchField1D(ScratchField1D&&) = delete;
  ScratchField1D& operator=(ScratchField1D&&) = delete;

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
  double* data() noexcept { return data_; }
  const double* data() const noexcept { return data_; }

  void reset(const char* tag, const std::size_t n) {
    if (n == 0) {
      data_ = nullptr;
      size_ = 0;
      return;
    }
    data_ = static_cast<double*>(device_scratch_acquire(tag, n * sizeof(double)));
    const cudaError_t err = cudaMemset(data_, 0, n * sizeof(double));
    TENRYU_ASSERT(err == cudaSuccess, "ScratchField1D::reset cudaMemset failed");
    size_ = n;
  }

  void copy_from_host(const std::vector<double>& src) {
    TENRYU_ASSERT(src.size() == size_,
                  "ScratchField1D::copy_from_host vector size mismatch");
    if (size_ == 0) {
      return;
    }

    const cudaError_t err =
        cudaMemcpy(data_, src.data(), size_ * sizeof(double), cudaMemcpyHostToDevice);
    TENRYU_ASSERT(err == cudaSuccess,
                  "ScratchField1D::copy_from_host cudaMemcpy H2D failed");
  }

 private:
  double* data_ = nullptr;
  std::size_t size_ = 0;
};

// Non-owning mutable view over a 1D field's device storage. Implicitly
// constructible from Field1D and ScratchField1D so helpers can accept either
// (persistent-kernel prep #57). An empty view doubles as "output not requested"
// for optional out-parameters.
template <typename Tag>
class Field1DView {
 public:
  Field1DView() = default;
  Field1DView(Field1D<Tag>& f) : data_(f.data()), size_(f.size()) {}       // NOLINT(implicit)
  Field1DView(ScratchField1D<Tag>& f) : data_(f.data()), size_(f.size()) {} // NOLINT(implicit)
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
  double* data() const noexcept { return data_; }

 private:
  double* data_ = nullptr;
  std::size_t size_ = 0;
};

// Read-only counterpart.
template <typename Tag>
class ConstField1DView {
 public:
  ConstField1DView() = default;
  ConstField1DView(const Field1D<Tag>& f) : data_(f.data()), size_(f.size()) {}
  ConstField1DView(const ScratchField1D<Tag>& f) : data_(f.data()), size_(f.size()) {}
  ConstField1DView(Field1DView<Tag> v) : data_(v.data()), size_(v.size()) {}
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
  const double* data() const noexcept { return data_; }

 private:
  const double* data_ = nullptr;
  std::size_t size_ = 0;
};

using CellField1D = Field1D<CellTag>;
using NodeField1D = Field1D<NodeTag>;
using GroupField1D = Field1D<GroupTag>;
using ScratchCellField1D = ScratchField1D<CellTag>;
using ScratchNodeField1D = ScratchField1D<NodeTag>;
using CellField1DView = Field1DView<CellTag>;
using NodeField1DView = Field1DView<NodeTag>;
using ConstCellField1DView = ConstField1DView<CellTag>;
using ConstNodeField1DView = ConstField1DView<NodeTag>;

}  // namespace tenryu::core
