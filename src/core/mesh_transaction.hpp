#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tenryu::core {

enum class MeshEventKind : std::uint8_t {
  kRSameTopology,
  kRConnectivityChange,
  kHRefinement,
  kHCoarsening,
  kContactRepartition,
  kFrameSwitch,
  kAmrHierarchyChange,
};

enum class TransactionClientKind : std::uint8_t {
  kAxisBandRemap,
  kReferenceBarrierRezone,
  kBeltMorph,
  kRunningPatchRemap,
  kTopologyReconnect,
  kTopologyCoarsen,
  kTopologyRefine,
  kEulerWindowRemap,
};

constexpr MeshEventKind event_kind_for_client(
    const TransactionClientKind kind) {
  switch (kind) {
    case TransactionClientKind::kAxisBandRemap:
      return MeshEventKind::kRSameTopology;
    case TransactionClientKind::kReferenceBarrierRezone:
      return MeshEventKind::kRSameTopology;
    case TransactionClientKind::kBeltMorph:
      return MeshEventKind::kRConnectivityChange;
    case TransactionClientKind::kRunningPatchRemap:
      return MeshEventKind::kRSameTopology;
    case TransactionClientKind::kTopologyReconnect:
      return MeshEventKind::kRConnectivityChange;
    case TransactionClientKind::kTopologyCoarsen:
      return MeshEventKind::kHCoarsening;
    case TransactionClientKind::kTopologyRefine:
      return MeshEventKind::kHRefinement;
    case TransactionClientKind::kEulerWindowRemap:
      return MeshEventKind::kFrameSwitch;
  }
  return MeshEventKind::kRSameTopology;
}

enum class TriggerSeverity : std::uint8_t {
  kSoftQuality = 0,
  kHardAdmissibility = 1,
};

struct EventContract {
  bool requires_bit_reversibility;
  bool allows_information_loss;
};

constexpr EventContract event_contract(const MeshEventKind kind) {
  switch (kind) {
    case MeshEventKind::kRSameTopology:
      return {true, false};
    case MeshEventKind::kRConnectivityChange:
      return {false, false};
    case MeshEventKind::kHRefinement:
      return {false, false};
    case MeshEventKind::kHCoarsening:
      return {false, true};
    case MeshEventKind::kContactRepartition:
      return {false, false};
    case MeshEventKind::kFrameSwitch:
      return {false, true};
    case MeshEventKind::kAmrHierarchyChange:
      return {false, true};
  }
  return {false, false};
}

struct TransactionBufferDesc {
  const char* name;
  void* live_ptr;
  std::size_t bytes;
};

struct PendingTransaction {
  bool active = false;
  TransactionClientKind kind;
  TriggerSeverity severity;
  std::uint64_t state_epoch = 0;
  std::int32_t reason_code = 0;
  std::int32_t requested_k = 0;
};

enum class TransactionError : std::uint8_t {
  kNone,
  kAlreadyCaptured,
  kNotCaptured,
  kZeroBuffers,
  kNullLivePtr,
  kUnknownName,
};

class ShadowTransaction {
 public:
  ShadowTransaction() = default;
  ~ShadowTransaction();
  ShadowTransaction(const ShadowTransaction&) = delete;
  ShadowTransaction& operator=(const ShadowTransaction&) = delete;
  ShadowTransaction(ShadowTransaction&& other) noexcept;
  ShadowTransaction& operator=(ShadowTransaction&& other) noexcept;

  [[nodiscard]] TransactionError capture(
      const std::vector<TransactionBufferDesc>& buffers,
      cudaStream_t stream);

  void* shadow_ptr(const char* name) const;
  std::size_t shadow_bytes(const char* name) const;

  void record_gate(const char* gate_name, bool passed);
  bool all_gates_passed() const;

  [[nodiscard]] bool commit(cudaStream_t stream);

  void discard();

  bool captured() const;

  void telemetry_increment(const char* counter_name, std::uint64_t delta);
  std::uint64_t telemetry_value(const char* counter_name) const;

  void set_failure_injection_point(int point);
  bool should_inject_failure(int protocol_step) const;

  // The host control path is single-threaded. A failing accessor updates this
  // value; callers must not read or update it concurrently.
  TransactionError last_error() const;

 private:
  struct Slot {
    std::string name;
    std::size_t offset = 0;
    std::size_t bytes = 0;
    void* live_ptr = nullptr;
  };

  void release_capture();
  const Slot* find_slot(const char* name) const;

  void* arena_ = nullptr;
  std::size_t arena_bytes_ = 0;
  std::vector<Slot> slots_;
  std::vector<std::pair<std::string, bool>> gates_;
  std::unordered_map<std::string, std::uint64_t> counters_;
  cudaEvent_t completion_event_ = nullptr;
  bool captured_ = false;
  int failure_injection_point_ = 0;
  mutable TransactionError last_error_ = TransactionError::kNone;
};

std::uint64_t hash_device_regions(
    const std::vector<TransactionBufferDesc>& regions);

}  // namespace tenryu::core
