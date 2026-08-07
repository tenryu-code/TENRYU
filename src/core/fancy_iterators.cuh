#ifndef TENRYU_CORE_FANCY_ITERATORS_CUH_
#define TENRYU_CORE_FANCY_ITERATORS_CUH_

#include <cstddef>
#include <iterator>

// Clean-room replacements for cub::CountingInputIterator and
// cub::TransformInputIterator (removed in CUDA 13 / CCCL 3.0), preserving
// the removed iterators' exact semantics: explicit ValueT, by-value
// reference, std::random_access_iterator_tag. Owned here so the call sites
// are independent of the cub/thrust version matrix (CUDA 12.0 local /
// 12.6 RunPod+SQUID / 13 BaseGPU-2026). The thrust::transform_iterator
// route is barred: under nvcc 12.0 -rdc whole-program device linking its
// cub::DeviceReduce instantiation yields a corrupted kernel image
// (deterministic SIGTRAP at an undecodable SASS pc; see
// tmp/repro_pool/ + docs/design/nohdet_cuda13_closure notes).

namespace tenryu::core {

template <typename ValueT, typename OffsetT = std::ptrdiff_t>
struct CountingInputIterator {
  using iterator_category = std::random_access_iterator_tag;
  using value_type = ValueT;
  using difference_type = OffsetT;
  using pointer = ValueT*;
  using reference = ValueT;

  ValueT val;

  __host__ __device__ __forceinline__ explicit CountingInputIterator(
      const ValueT v = ValueT{})
      : val(v) {}

  __host__ __device__ __forceinline__ reference operator*() const {
    return val;
  }
  __host__ __device__ __forceinline__ reference operator[](const OffsetT n) const {
    return static_cast<ValueT>(val + static_cast<ValueT>(n));
  }
  __host__ __device__ __forceinline__ CountingInputIterator& operator++() {
    ++val;
    return *this;
  }
  __host__ __device__ __forceinline__ CountingInputIterator operator++(int) {
    CountingInputIterator old = *this;
    ++val;
    return old;
  }
  __host__ __device__ __forceinline__ CountingInputIterator operator+(
      const OffsetT n) const {
    return CountingInputIterator(static_cast<ValueT>(val + static_cast<ValueT>(n)));
  }
  __host__ __device__ __forceinline__ CountingInputIterator& operator+=(
      const OffsetT n) {
    val = static_cast<ValueT>(val + static_cast<ValueT>(n));
    return *this;
  }
  __host__ __device__ __forceinline__ CountingInputIterator operator-(
      const OffsetT n) const {
    return CountingInputIterator(static_cast<ValueT>(val - static_cast<ValueT>(n)));
  }
  __host__ __device__ __forceinline__ difference_type operator-(
      const CountingInputIterator& other) const {
    return static_cast<difference_type>(val - other.val);
  }
  __host__ __device__ __forceinline__ bool operator==(
      const CountingInputIterator& other) const {
    return val == other.val;
  }
  __host__ __device__ __forceinline__ bool operator!=(
      const CountingInputIterator& other) const {
    return val != other.val;
  }
};

template <typename ValueT,
          typename ConversionOp,
          typename InputIteratorT,
          typename OffsetT = std::ptrdiff_t>
struct TransformInputIterator {
  using iterator_category = std::random_access_iterator_tag;
  using value_type = ValueT;
  using difference_type = OffsetT;
  using pointer = ValueT*;
  using reference = ValueT;

  InputIteratorT input;
  ConversionOp conversion_op;

  __host__ __device__ __forceinline__ TransformInputIterator(InputIteratorT it,
                                                             ConversionOp op)
      : input(it), conversion_op(op) {}

  __host__ __device__ __forceinline__ reference operator*() const {
    return conversion_op(*input);
  }
  __host__ __device__ __forceinline__ reference operator[](const OffsetT n) const {
    return conversion_op(input[n]);
  }
  __host__ __device__ __forceinline__ TransformInputIterator& operator++() {
    ++input;
    return *this;
  }
  __host__ __device__ __forceinline__ TransformInputIterator operator++(int) {
    TransformInputIterator old = *this;
    ++input;
    return old;
  }
  __host__ __device__ __forceinline__ TransformInputIterator operator+(
      const OffsetT n) const {
    return TransformInputIterator(input + n, conversion_op);
  }
  __host__ __device__ __forceinline__ TransformInputIterator& operator+=(
      const OffsetT n) {
    input += n;
    return *this;
  }
  __host__ __device__ __forceinline__ TransformInputIterator operator-(
      const OffsetT n) const {
    return TransformInputIterator(input - n, conversion_op);
  }
  __host__ __device__ __forceinline__ difference_type operator-(
      const TransformInputIterator& other) const {
    return static_cast<difference_type>(input - other.input);
  }
  __host__ __device__ __forceinline__ bool operator==(
      const TransformInputIterator& other) const {
    return input == other.input;
  }
  __host__ __device__ __forceinline__ bool operator!=(
      const TransformInputIterator& other) const {
    return input != other.input;
  }
};

}  // namespace tenryu::core

#endif  // TENRYU_CORE_FANCY_ITERATORS_CUH_
