#include "hydro/phase_audit.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace tenryu::hydro::phase_audit {
namespace {

std::string json_escape(const std::string_view value) {
  std::ostringstream out;
  for (const unsigned char ch : value) {
    switch (ch) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (ch < 0x20U) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<unsigned int>(ch) << std::dec;
        } else {
          out << static_cast<char>(ch);
        }
        break;
    }
  }
  return out.str();
}

class LedgerWriter {
 public:
  LedgerWriter() {
    const char* raw = std::getenv("TENRYU_I1B_PHASE_LEDGER");
    if (raw == nullptr || raw[0] == '\0') {
      return;
    }
    path_ = raw;
    output_.open(path_, std::ios::out | std::ios::trunc);
    if (!output_) {
      throw std::runtime_error("failed to open phase ledger: " + path_);
    }
  }

  void append(const Event& event) {
    if (!output_.is_open()) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    // There is deliberately no death/abort hook. Flushing every completed
    // line preserves the maximal valid prefix if the process terminates.
    output_ << "{\"n\":" << event.step
            << ",\"a\":" << event.attempt
            << ",\"tau\":" << std::setprecision(17) << event.time
            << ",\"c\":" << event.cell_or_unit
            << ",\"s\":\"" << json_escape(event.stage)
            << "\",\"p\":\"" << json_escape(event.producer)
            << "\",\"q\":\"" << json_escape(event.consumer)
            << "\",\"k\":\"" << event_kind_name(event.kind)
            << "\",\"static_state\":"
            << (event.static_state ? "true" : "false") << "}\n";
    output_.flush();
    if (!output_) {
      throw std::runtime_error("failed to write phase ledger: " + path_);
    }
  }

 private:
  std::string path_;
  std::ofstream output_;
  std::mutex mutex_;
};

enum class JsonValueKind { String, Number, Boolean };

struct JsonValue {
  JsonValueKind kind = JsonValueKind::String;
  std::string text;
  bool boolean = false;
};

class JsonCursor {
 public:
  explicit JsonCursor(const std::string_view input) : input_(input) {}

  bool parse_object(std::unordered_map<std::string, JsonValue>* fields,
                    std::string* error) {
    skip_space();
    if (!consume('{')) {
      *error = "expected '{'";
      return false;
    }
    skip_space();
    if (consume('}')) {
      skip_space();
      return finish(error);
    }
    while (position_ < input_.size()) {
      std::string key;
      if (!parse_string(&key, error)) {
        return false;
      }
      skip_space();
      if (!consume(':')) {
        *error = "expected ':'";
        return false;
      }
      skip_space();
      JsonValue value;
      if (!parse_value(&value, error)) {
        return false;
      }
      if (!fields->emplace(std::move(key), std::move(value)).second) {
        *error = "duplicate object field";
        return false;
      }
      skip_space();
      if (consume('}')) {
        skip_space();
        return finish(error);
      }
      if (!consume(',')) {
        *error = "expected ',' or '}'";
        return false;
      }
      skip_space();
    }
    *error = "truncated object";
    return false;
  }

 private:
  bool finish(std::string* error) const {
    if (position_ != input_.size()) {
      *error = "trailing characters";
      return false;
    }
    return true;
  }

  void skip_space() {
    while (position_ < input_.size()) {
      const char ch = input_[position_];
      if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') {
        break;
      }
      ++position_;
    }
  }

  bool consume(const char expected) {
    if (position_ >= input_.size() || input_[position_] != expected) {
      return false;
    }
    ++position_;
    return true;
  }

  static int hex_value(const char ch) {
    if (ch >= '0' && ch <= '9') {
      return ch - '0';
    }
    if (ch >= 'a' && ch <= 'f') {
      return ch - 'a' + 10;
    }
    if (ch >= 'A' && ch <= 'F') {
      return ch - 'A' + 10;
    }
    return -1;
  }

  bool parse_string(std::string* value, std::string* error) {
    if (!consume('"')) {
      *error = "expected string";
      return false;
    }
    while (position_ < input_.size()) {
      const unsigned char ch =
          static_cast<unsigned char>(input_[position_++]);
      if (ch == '"') {
        return true;
      }
      if (ch < 0x20U) {
        *error = "unescaped control character";
        return false;
      }
      if (ch != '\\') {
        value->push_back(static_cast<char>(ch));
        continue;
      }
      if (position_ >= input_.size()) {
        *error = "truncated string escape";
        return false;
      }
      const char escaped = input_[position_++];
      switch (escaped) {
        case '"':
        case '\\':
        case '/':
          value->push_back(escaped);
          break;
        case 'b':
          value->push_back('\b');
          break;
        case 'f':
          value->push_back('\f');
          break;
        case 'n':
          value->push_back('\n');
          break;
        case 'r':
          value->push_back('\r');
          break;
        case 't':
          value->push_back('\t');
          break;
        case 'u': {
          if (input_.size() - position_ < 4U) {
            *error = "truncated unicode escape";
            return false;
          }
          unsigned int code = 0U;
          for (int digit = 0; digit < 4; ++digit) {
            const int value_hex = hex_value(input_[position_++]);
            if (value_hex < 0) {
              *error = "invalid unicode escape";
              return false;
            }
            code = code * 16U + static_cast<unsigned int>(value_hex);
          }
          if (code <= 0x7fU) {
            value->push_back(static_cast<char>(code));
          } else if (code <= 0x7ffU) {
            value->push_back(static_cast<char>(0xc0U | (code >> 6U)));
            value->push_back(static_cast<char>(0x80U | (code & 0x3fU)));
          } else {
            value->push_back(static_cast<char>(0xe0U | (code >> 12U)));
            value->push_back(
                static_cast<char>(0x80U | ((code >> 6U) & 0x3fU)));
            value->push_back(static_cast<char>(0x80U | (code & 0x3fU)));
          }
          break;
        }
        default:
          *error = "invalid string escape";
          return false;
      }
    }
    *error = "truncated string";
    return false;
  }

  bool parse_value(JsonValue* value, std::string* error) {
    if (position_ >= input_.size()) {
      *error = "missing value";
      return false;
    }
    if (input_[position_] == '"') {
      value->kind = JsonValueKind::String;
      return parse_string(&value->text, error);
    }
    if (input_.substr(position_, 4U) == "true") {
      position_ += 4U;
      value->kind = JsonValueKind::Boolean;
      value->boolean = true;
      return true;
    }
    if (input_.substr(position_, 5U) == "false") {
      position_ += 5U;
      value->kind = JsonValueKind::Boolean;
      value->boolean = false;
      return true;
    }
    const std::size_t begin = position_;
    if (input_[position_] == '-') {
      ++position_;
    }
    if (position_ >= input_.size()) {
      *error = "truncated number";
      return false;
    }
    if (input_[position_] == '0') {
      ++position_;
    } else if (input_[position_] >= '1' && input_[position_] <= '9') {
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
    } else {
      *error = "invalid number";
      return false;
    }
    if (position_ < input_.size() && input_[position_] == '.') {
      ++position_;
      const std::size_t fractional_begin = position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
      if (position_ == fractional_begin) {
        *error = "invalid number fraction";
        return false;
      }
    }
    if (position_ < input_.size() &&
        (input_[position_] == 'e' || input_[position_] == 'E')) {
      ++position_;
      if (position_ < input_.size() &&
          (input_[position_] == '+' || input_[position_] == '-')) {
        ++position_;
      }
      const std::size_t exponent_begin = position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
      if (position_ == exponent_begin) {
        *error = "invalid number exponent";
        return false;
      }
    }
    value->kind = JsonValueKind::Number;
    value->text = std::string(input_.substr(begin, position_ - begin));
    return true;
  }

  std::string_view input_;
  std::size_t position_ = 0;
};

std::optional<EventKind> parse_event_kind(const std::string_view value) {
  constexpr EventKind kinds[] = {
      EventKind::Capture,     EventKind::TargetBuilt,
      EventKind::QualityEval, EventKind::Accept,
      EventKind::Reject,      EventKind::Rollback,
      EventKind::RemapApply,  EventKind::Commit,
      EventKind::Noop,        EventKind::TriggerEval,
      EventKind::Output,      EventKind::Abort,
  };
  for (const EventKind kind : kinds) {
    if (event_kind_name(kind) == value) {
      return kind;
    }
  }
  return std::nullopt;
}

bool required_field(const std::unordered_map<std::string, JsonValue>& fields,
                    const std::string& key,
                    const JsonValueKind kind,
                    const JsonValue** value,
                    std::string* error) {
  const auto found = fields.find(key);
  if (found == fields.end()) {
    *error = "missing field '" + key + "'";
    return false;
  }
  if (found->second.kind != kind) {
    *error = "wrong type for field '" + key + "'";
    return false;
  }
  *value = &found->second;
  return true;
}

bool parse_int64(const JsonValue& value,
                 std::int64_t* result,
                 std::string* error) {
  const char* begin = value.text.data();
  const char* end = begin + value.text.size();
  const auto parsed = std::from_chars(begin, end, *result);
  if (parsed.ec != std::errc{} || parsed.ptr != end) {
    *error = "integer is out of range or non-integral";
    return false;
  }
  return true;
}

bool parse_double(const JsonValue& value,
                  double* result,
                  std::string* error) {
  const char* begin = value.text.data();
  const char* end = begin + value.text.size();
  const auto parsed =
      std::from_chars(begin, end, *result, std::chars_format::general);
  if (parsed.ec != std::errc{} || parsed.ptr != end ||
      !std::isfinite(*result)) {
    *error = "floating-point value is invalid or out of range";
    return false;
  }
  return true;
}

bool parse_event_line(const std::string_view line,
                      Event* event,
                      std::string* error) {
  std::unordered_map<std::string, JsonValue> fields;
  JsonCursor cursor(line);
  if (!cursor.parse_object(&fields, error)) {
    return false;
  }

  const JsonValue* n = nullptr;
  const JsonValue* a = nullptr;
  const JsonValue* tau = nullptr;
  const JsonValue* c = nullptr;
  const JsonValue* s = nullptr;
  const JsonValue* p = nullptr;
  const JsonValue* q = nullptr;
  const JsonValue* k = nullptr;
  if (!required_field(fields, "n", JsonValueKind::Number, &n, error) ||
      !required_field(fields, "a", JsonValueKind::Number, &a, error) ||
      !required_field(fields, "tau", JsonValueKind::Number, &tau, error) ||
      !required_field(fields, "c", JsonValueKind::Number, &c, error) ||
      !required_field(fields, "s", JsonValueKind::String, &s, error) ||
      !required_field(fields, "p", JsonValueKind::String, &p, error) ||
      !required_field(fields, "q", JsonValueKind::String, &q, error) ||
      !required_field(fields, "k", JsonValueKind::String, &k, error)) {
    return false;
  }
  std::int64_t attempt = 0;
  if (!parse_int64(*n, &event->step, error) ||
      !parse_int64(*a, &attempt, error) ||
      !parse_double(*tau, &event->time, error) ||
      !parse_int64(*c, &event->cell_or_unit, error)) {
    return false;
  }
  if (event->step < 0 || attempt < 0 ||
      attempt > std::numeric_limits<int>::max()) {
    *error = "negative step/attempt or attempt out of range";
    return false;
  }
  event->attempt = static_cast<int>(attempt);
  event->stage = s->text;
  event->producer = p->text;
  event->consumer = q->text;
  const std::optional<EventKind> kind = parse_event_kind(k->text);
  if (!kind.has_value()) {
    *error = "unknown event kind '" + k->text + "'";
    return false;
  }
  event->kind = *kind;

  const auto static_field = fields.find("static_state");
  if (static_field != fields.end()) {
    if (static_field->second.kind != JsonValueKind::Boolean) {
      *error = "wrong type for field 'static_state'";
      return false;
    }
    event->static_state = static_field->second.boolean;
  } else if (event->kind == EventKind::TriggerEval) {
    *error = "missing field 'static_state' on TRIGGER_EVAL";
    return false;
  }
  return true;
}

std::string event_location(const Event& event) {
  return "n=" + std::to_string(event.step) +
         " a=" + std::to_string(event.attempt);
}

struct AttemptState {
  bool captured = false;
  bool quality_evaluated = false;
  bool accepted = false;
  bool rejected = false;
  bool committed = false;
};

void check_events(const std::vector<Event>& events, ReplayResult* result) {
  std::map<std::pair<std::int64_t, int>, AttemptState> attempts;
  std::map<std::int64_t, int> commits_per_step;
  std::map<std::int64_t, bool> static_trigger_seen;

  for (std::size_t index = 0; index < events.size(); ++index) {
    const Event& event = events[index];
    if (index > 0U) {
      const Event& previous = events[index - 1U];
      if (std::pair{event.step, event.attempt} <
          std::pair{previous.step, previous.attempt}) {
        result->violations.push_back(
            "non-monotone (n,a) at event " + std::to_string(index + 1U) +
            ": " + event_location(event));
      }
    }

    AttemptState& attempt = attempts[{event.step, event.attempt}];
    switch (event.kind) {
      case EventKind::Capture:
        attempt.captured = true;
        break;
      case EventKind::QualityEval:
        attempt.quality_evaluated = true;
        break;
      case EventKind::Accept:
        if (!attempt.captured) {
          result->violations.push_back(
              "ACCEPT without preceding CAPTURE: " + event_location(event));
        }
        if (!attempt.quality_evaluated) {
          result->violations.push_back(
              "ACCEPT without preceding QUALITY_EVAL: " +
              event_location(event));
        }
        attempt.accepted = true;
        break;
      case EventKind::Reject:
        attempt.rejected = true;
        break;
      case EventKind::RemapApply:
        if (!attempt.accepted) {
          result->violations.push_back(
              "REMAP_APPLY outside an accepted attempt: " +
              event_location(event));
        }
        break;
      case EventKind::Commit:
        if (!attempt.captured) {
          result->violations.push_back(
              "COMMIT without preceding CAPTURE: " + event_location(event));
        }
        if (!attempt.accepted) {
          result->violations.push_back(
              "COMMIT outside an accepted attempt: " + event_location(event));
        }
        attempt.committed = true;
        if (++commits_per_step[event.step] > 1) {
          result->violations.push_back(
              "duplicate COMMIT in accepted step n=" +
              std::to_string(event.step));
        }
        break;
      case EventKind::TriggerEval:
        if (event.static_state) {
          static_trigger_seen[event.step] = true;
        }
        break;
      case EventKind::TargetBuilt:
        if (static_trigger_seen[event.step]) {
          result->violations.push_back(
              "TARGET_BUILT after static-state TRIGGER_EVAL in n=" +
              std::to_string(event.step));
        }
        break;
      case EventKind::Rollback:
      case EventKind::Noop:
      case EventKind::Output:
      case EventKind::Abort:
        break;
    }
  }

  for (const auto& [key, attempt] : attempts) {
    if (attempt.rejected && attempt.committed) {
      result->violations.push_back(
          "REJECT and COMMIT in the same attempt: n=" +
          std::to_string(key.first) + " a=" + std::to_string(key.second));
    }
  }

  for (std::size_t index = 0; index < events.size(); ++index) {
    const Event& rejected = events[index];
    if (rejected.kind != EventKind::Reject) {
      continue;
    }
    if (index + 1U == events.size()) {
      continue;
    }
    bool resolved = false;
    for (std::size_t next = index + 1U; next < events.size(); ++next) {
      const Event& event = events[next];
      if (event.step != rejected.step) {
        break;
      }
      if (event.kind == EventKind::Output) {
        result->violations.push_back(
            "OUTPUT between REJECT and ROLLBACK/ABORT in n=" +
            std::to_string(rejected.step));
      }
      if (event.kind == EventKind::Rollback ||
          event.kind == EventKind::Abort) {
        resolved = true;
        break;
      }
    }
    if (!resolved) {
      result->violations.push_back(
          "REJECT not followed by ROLLBACK/ABORT in the same n: " +
          event_location(rejected));
    }
  }
}

}  // namespace

std::string_view event_kind_name(const EventKind kind) {
  switch (kind) {
    case EventKind::Capture:
      return "CAPTURE";
    case EventKind::TargetBuilt:
      return "TARGET_BUILT";
    case EventKind::QualityEval:
      return "QUALITY_EVAL";
    case EventKind::Accept:
      return "ACCEPT";
    case EventKind::Reject:
      return "REJECT";
    case EventKind::Rollback:
      return "ROLLBACK";
    case EventKind::RemapApply:
      return "REMAP_APPLY";
    case EventKind::Commit:
      return "COMMIT";
    case EventKind::Noop:
      return "NOOP";
    case EventKind::TriggerEval:
      return "TRIGGER_EVAL";
    case EventKind::Output:
      return "OUTPUT";
    case EventKind::Abort:
      return "ABORT";
  }
  return "UNKNOWN";
}

bool enabled() {
  static const bool requested = [] {
    const char* raw = std::getenv("TENRYU_I1B_PHASE_LEDGER");
    return raw != nullptr && raw[0] != '\0';
  }();
  return requested;
}

void emit(const Event& event) {
  static LedgerWriter writer;
  writer.append(event);
}

ReplayResult replay_jsonl(const std::string_view jsonl) {
  ReplayResult result;
  std::vector<Event> events;
  std::size_t line_number = 0U;
  std::size_t begin = 0U;
  while (begin < jsonl.size()) {
    const std::size_t newline = jsonl.find('\n', begin);
    const bool final_without_newline = newline == std::string_view::npos;
    const std::size_t end = final_without_newline ? jsonl.size() : newline;
    ++line_number;
    const std::string_view line = jsonl.substr(begin, end - begin);
    if (!line.empty() && line.find_first_not_of(" \t\r") !=
                             std::string_view::npos) {
      Event event;
      std::string error;
      if (parse_event_line(line, &event, &error)) {
        events.push_back(std::move(event));
      } else if (final_without_newline &&
                 line.find_last_not_of(" \t\r") != std::string_view::npos &&
                 line[line.find_last_not_of(" \t\r")] != '}') {
        result.truncated_final_line = true;
      } else {
        result.violations.push_back(
            "JSONL parse error on line " + std::to_string(line_number) +
            ": " + error);
      }
    }
    if (final_without_newline) {
      break;
    }
    begin = newline + 1U;
  }
  result.events_read = events.size();
  check_events(events, &result);
  return result;
}

ReplayResult replay_jsonl(std::istream& input) {
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return replay_jsonl(buffer.str());
}

}  // namespace tenryu::hydro::phase_audit
