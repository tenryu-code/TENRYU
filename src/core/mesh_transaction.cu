#include "core/mesh_transaction.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include "core/error.hpp"

namespace tenryu::core {
namespace {

constexpr std::size_t kSlotAlignment = 256;
constexpr std::size_t kHashStagingBytes = 1024 * 1024;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

std::size_t aligned_slot_bytes(const std::size_t bytes) {
  if (bytes == 0) {
    return 0;
  }
  return ((bytes + kSlotAlignment - 1) / kSlotAlignment) * kSlotAlignment;
}

}  // namespace

ShadowTransaction::~ShadowTransaction() {
  discard();
}

ShadowTransaction::ShadowTransaction(ShadowTransaction&& other) noexcept
    : arena_(std::exchange(other.arena_, nullptr)),
      arena_bytes_(std::exchange(other.arena_bytes_, 0)),
      slots_(std::move(other.slots_)),
      gates_(std::move(other.gates_)),
      counters_(std::move(other.counters_)),
      completion_event_(std::exchange(other.completion_event_, nullptr)),
      captured_(std::exchange(other.captured_, false)),
      failure_injection_point_(
          std::exchange(other.failure_injection_point_, 0)),
      last_error_(std::exchange(other.last_error_, TransactionError::kNone)) {}

ShadowTransaction& ShadowTransaction::operator=(
    ShadowTransaction&& other) noexcept {
  if (this != &other) {
    discard();
    arena_ = std::exchange(other.arena_, nullptr);
    arena_bytes_ = std::exchange(other.arena_bytes_, 0);
    slots_ = std::move(other.slots_);
    gates_ = std::move(other.gates_);
    counters_ = std::move(other.counters_);
    completion_event_ = std::exchange(other.completion_event_, nullptr);
    captured_ = std::exchange(other.captured_, false);
    failure_injection_point_ =
        std::exchange(other.failure_injection_point_, 0);
    last_error_ =
        std::exchange(other.last_error_, TransactionError::kNone);
  }
  return *this;
}

TransactionError ShadowTransaction::capture(
    const std::vector<TransactionBufferDesc>& buffers,
    const cudaStream_t stream) {
  if (captured_) {
    last_error_ = TransactionError::kAlreadyCaptured;
    return last_error_;
  }
  if (buffers.empty()) {
    last_error_ = TransactionError::kZeroBuffers;
    return last_error_;
  }
  for (const auto& buffer : buffers) {
    if (buffer.live_ptr == nullptr && buffer.bytes > 0) {
      last_error_ = TransactionError::kNullLivePtr;
      return last_error_;
    }
  }

  slots_.clear();
  gates_.clear();
  counters_.clear();
  slots_.reserve(buffers.size());
  arena_bytes_ = 0;
  for (const auto& buffer : buffers) {
    Slot slot;
    slot.name = (buffer.name != nullptr) ? buffer.name : "";
    slot.offset = arena_bytes_;
    slot.bytes = buffer.bytes;
    slot.live_ptr = buffer.live_ptr;
    slots_.push_back(std::move(slot));
    arena_bytes_ += aligned_slot_bytes(buffer.bytes);
  }

  if (arena_bytes_ > 0) {
    CUDA_CHECK(cudaMalloc(&arena_, arena_bytes_));
  }
  CUDA_CHECK(
      cudaEventCreateWithFlags(&completion_event_, cudaEventDisableTiming));
  auto* arena_bytes = static_cast<unsigned char*>(arena_);
  for (const auto& slot : slots_) {
    if (slot.bytes > 0) {
      CUDA_CHECK(cudaMemcpyAsync(arena_bytes + slot.offset, slot.live_ptr,
                                 slot.bytes, cudaMemcpyDeviceToDevice,
                                 stream));
    }
  }
  CUDA_CHECK(cudaEventRecord(completion_event_, stream));
  captured_ = true;
  last_error_ = TransactionError::kNone;
  return TransactionError::kNone;
}

const ShadowTransaction::Slot* ShadowTransaction::find_slot(
    const char* name) const {
  if (name == nullptr) {
    return nullptr;
  }
  for (const auto& slot : slots_) {
    if (slot.name == name) {
      return &slot;
    }
  }
  return nullptr;
}

void* ShadowTransaction::shadow_ptr(const char* name) const {
  if (!captured_) {
    last_error_ = TransactionError::kNotCaptured;
    return nullptr;
  }
  const Slot* slot = find_slot(name);
  if (slot == nullptr) {
    last_error_ = TransactionError::kUnknownName;
    return nullptr;
  }
  if (arena_ == nullptr) {
    return nullptr;
  }
  return static_cast<unsigned char*>(arena_) + slot->offset;
}

std::size_t ShadowTransaction::shadow_bytes(const char* name) const {
  if (!captured_) {
    last_error_ = TransactionError::kNotCaptured;
    return 0;
  }
  const Slot* slot = find_slot(name);
  if (slot == nullptr) {
    last_error_ = TransactionError::kUnknownName;
    return 0;
  }
  return slot->bytes;
}

void ShadowTransaction::record_gate(const char* gate_name, const bool passed) {
  gates_.emplace_back((gate_name != nullptr) ? gate_name : "", passed);
}

bool ShadowTransaction::all_gates_passed() const {
  if (gates_.empty()) {
    return false;
  }
  return std::all_of(gates_.begin(), gates_.end(),
                     [](const auto& gate) { return gate.second; });
}

bool ShadowTransaction::commit(const cudaStream_t stream) {
  if (!captured_) {
    last_error_ = TransactionError::kNotCaptured;
    return false;
  }

  CUDA_CHECK(cudaEventRecord(completion_event_, stream));
  CUDA_CHECK(cudaEventSynchronize(completion_event_));
  if (!all_gates_passed()) {
    return false;
  }

  auto* arena_bytes = static_cast<unsigned char*>(arena_);
  for (const auto& slot : slots_) {
    if (slot.bytes > 0) {
      CUDA_CHECK(cudaMemcpyAsync(slot.live_ptr, arena_bytes + slot.offset,
                                 slot.bytes, cudaMemcpyDeviceToDevice,
                                 stream));
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  release_capture();
  last_error_ = TransactionError::kNone;
  return true;
}

void ShadowTransaction::release_capture() {
  if (arena_ != nullptr) {
    CUDA_CHECK(cudaFree(arena_));
    arena_ = nullptr;
  }
  if (completion_event_ != nullptr) {
    CUDA_CHECK(cudaEventDestroy(completion_event_));
    completion_event_ = nullptr;
  }
  arena_bytes_ = 0;
  slots_.clear();
  gates_.clear();
  counters_.clear();
  captured_ = false;
}

void ShadowTransaction::discard() {
  release_capture();
  last_error_ = TransactionError::kNone;
}

bool ShadowTransaction::captured() const {
  return captured_;
}

void ShadowTransaction::telemetry_increment(const char* counter_name,
                                            const std::uint64_t delta) {
  counters_[(counter_name != nullptr) ? counter_name : ""] += delta;
}

std::uint64_t ShadowTransaction::telemetry_value(
    const char* counter_name) const {
  const auto it =
      counters_.find((counter_name != nullptr) ? counter_name : "");
  return (it != counters_.end()) ? it->second : 0;
}

void ShadowTransaction::set_failure_injection_point(const int point) {
  failure_injection_point_ = point;
}

bool ShadowTransaction::should_inject_failure(
    const int protocol_step) const {
  return failure_injection_point_ != 0 &&
         failure_injection_point_ == protocol_step;
}

TransactionError ShadowTransaction::last_error() const {
  return last_error_;
}

std::uint64_t hash_device_regions(
    const std::vector<TransactionBufferDesc>& regions) {
  std::vector<unsigned char> staging(kHashStagingBytes);
  std::uint64_t hash = kFnvOffset;
  for (const auto& region : regions) {
    auto* device_bytes =
        static_cast<const unsigned char*>(region.live_ptr);
    std::size_t offset = 0;
    while (offset < region.bytes) {
      const std::size_t chunk =
          std::min(kHashStagingBytes, region.bytes - offset);
      CUDA_CHECK(cudaMemcpy(staging.data(), device_bytes + offset, chunk,
                            cudaMemcpyDeviceToHost));
      for (std::size_t i = 0; i < chunk; ++i) {
        hash ^= static_cast<std::uint64_t>(staging[i]);
        hash *= kFnvPrime;
      }
      offset += chunk;
    }
  }
  return hash;
}

}  // namespace tenryu::core
