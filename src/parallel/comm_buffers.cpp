#include "parallel/comm_buffers.hpp"

#include <cstdlib>
#include <string>
#include <string_view>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"

#if TENRYU_ENABLE_MPI
#include <mpi.h>
#endif

namespace tenryu::parallel {
namespace {

std::string cuda_error_message(const char* operation, const cudaError_t err) {
  return std::string(operation) + " failed: " + cudaGetErrorString(err);
}

void release_device_ptr(void*& ptr) {
  if (ptr == nullptr) {
    return;
  }
  const cudaError_t err = cudaFree(ptr);
  TENRYU_ASSERT(err == cudaSuccess, cuda_error_message("cudaFree", err));
  ptr = nullptr;
}

void release_pinned_ptr(void*& ptr) {
  if (ptr == nullptr) {
    return;
  }
  const cudaError_t err = cudaFreeHost(ptr);
  TENRYU_ASSERT(err == cudaSuccess, cuda_error_message("cudaFreeHost", err));
  ptr = nullptr;
}

bool env_enabled(const char* value) {
  if (value == nullptr) {
    return false;
  }
  const std::string_view text(value);
  return text == "1" || text == "true" || text == "TRUE" || text == "on" ||
         text == "ON";
}

}  // namespace

DeviceArray::DeviceArray(const char* pool_tag_in) : pool_tag(pool_tag_in) {}

DeviceArray::DeviceArray(DeviceArray&& other) noexcept
    : ptr(other.ptr),
      size(other.size),
      capacity(other.capacity),
      pool_tag(other.pool_tag) {
  other.ptr = nullptr;
  other.size = 0;
  other.capacity = 0;
  other.pool_tag = nullptr;
}

DeviceArray& DeviceArray::operator=(DeviceArray&& other) noexcept {
  if (this != &other) {
    if (pool_tag == nullptr) {
      release_device_ptr(ptr);
    }
    ptr = other.ptr;
    size = other.size;
    capacity = other.capacity;
    pool_tag = other.pool_tag;
    other.ptr = nullptr;
    other.size = 0;
    other.capacity = 0;
    other.pool_tag = nullptr;
  }
  return *this;
}

void DeviceArray::resize(const std::size_t new_capacity) {
  if (new_capacity <= capacity) {
    size = new_capacity;
    return;
  }

  if (pool_tag != nullptr) {
    ptr = tenryu::core::device_scratch_acquire(pool_tag, new_capacity);
    size = new_capacity;
    capacity = new_capacity;
    return;
  }

  release_device_ptr(ptr);
  void* new_ptr = nullptr;
  const cudaError_t alloc_err = cudaMalloc(&new_ptr, new_capacity);
  TENRYU_ASSERT(alloc_err == cudaSuccess,
                cuda_error_message("cudaMalloc", alloc_err));
  ptr = new_ptr;
  size = new_capacity;
  capacity = new_capacity;
}

DeviceArray::~DeviceArray() {
  if (pool_tag == nullptr) {
    release_device_ptr(ptr);
  } else {
    ptr = nullptr;
  }
  size = 0;
  capacity = 0;
}

PinnedArray::PinnedArray(PinnedArray&& other) noexcept
    : ptr(other.ptr), size(other.size), capacity(other.capacity) {
  other.ptr = nullptr;
  other.size = 0;
  other.capacity = 0;
}

PinnedArray& PinnedArray::operator=(PinnedArray&& other) noexcept {
  if (this != &other) {
    release_pinned_ptr(ptr);
    ptr = other.ptr;
    size = other.size;
    capacity = other.capacity;
    other.ptr = nullptr;
    other.size = 0;
    other.capacity = 0;
  }
  return *this;
}

void PinnedArray::resize(const std::size_t new_capacity) {
  if (new_capacity <= capacity) {
    size = new_capacity;
    return;
  }

  release_pinned_ptr(ptr);
  void* new_ptr = nullptr;
  const cudaError_t alloc_err = cudaMallocHost(&new_ptr, new_capacity);
  TENRYU_ASSERT(alloc_err == cudaSuccess,
                cuda_error_message("cudaMallocHost", alloc_err));
  ptr = new_ptr;
  size = new_capacity;
  capacity = new_capacity;
}

PinnedArray::~PinnedArray() {
  release_pinned_ptr(ptr);
  size = 0;
  capacity = 0;
}

void CommBuffers::resize_if_needed(const std::size_t required) {
  for (int dir = 0; dir < MAX_NEIGHBORS; ++dir) {
    send_halo[dir].resize(required);
    recv_halo[dir].resize(required);
    if (gpu_aware_mpi) {
      host_send[dir] = PinnedArray{};
      host_recv[dir] = PinnedArray{};
    } else {
      host_send[dir].resize(required);
      host_recv[dir].resize(required);
    }
  }

  emigrant_send.resize(required);
  emigrant_recv.resize(required);
}

bool detect_gpu_aware_mpi() {
#if TENRYU_GPU_AWARE_MPI_COMPILE == 0
  return false;
#else
#if TENRYU_ENABLE_MPI
  int mpi_initialized = 0;
  const int initialized_err = MPI_Initialized(&mpi_initialized);
  TENRYU_ASSERT(initialized_err == MPI_SUCCESS,
                "MPI_Initialized failed in detect_gpu_aware_mpi");
  if (!mpi_initialized) {
    return false;
  }

#if defined(MPIX_CUDA_AWARE_SUPPORT)
  if (MPIX_Query_cuda_support() == 1) {
    return true;
  }
#endif

  return env_enabled(std::getenv("MPICH_GPU_SUPPORT_ENABLED"));
#else
  return false;
#endif
#endif
}

}  // namespace tenryu::parallel
