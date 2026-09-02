#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <string>
#include <string_view>
#include <vector>

namespace tenryu::hydro::phase_audit {

enum class EventKind {
  Capture,
  TargetBuilt,
  QualityEval,
  Accept,
  Reject,
  Rollback,
  RemapApply,
  Commit,
  Noop,
  TriggerEval,
  Output,
  Abort,
};

struct Event {
  std::int64_t step = 0;
  int attempt = 0;
  double time = 0.0;
  std::int64_t cell_or_unit = -1;
  std::string stage;
  std::string producer;
  std::string consumer;
  EventKind kind = EventKind::Noop;
  bool static_state = false;
};

struct ReplayResult {
  std::size_t events_read = 0;
  bool truncated_final_line = false;
  std::vector<std::string> violations;

  [[nodiscard]] bool passed() const { return violations.empty(); }
};

[[nodiscard]] std::string_view event_kind_name(EventKind kind);
[[nodiscard]] bool enabled();

// Default-off production emitter. When TENRYU_I1B_PHASE_LEDGER is set, each
// event is appended as one JSONL record and flushed immediately. OUTPUT
// production wiring and nonzero driver-attempt plumbing are deferred in v1.
void emit(const Event& event);

[[nodiscard]] ReplayResult replay_jsonl(std::string_view jsonl);
[[nodiscard]] ReplayResult replay_jsonl(std::istream& input);

}  // namespace tenryu::hydro::phase_audit
