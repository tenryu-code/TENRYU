#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace tenryu::io {

inline std::size_t skip_json_whitespace(const std::string& text,
                                        std::size_t pos,
                                        const std::size_t end) {
  while (pos < end &&
         (text[pos] == ' ' || text[pos] == '\n' || text[pos] == '\r' ||
          text[pos] == '\t')) {
    ++pos;
  }
  return pos;
}

inline std::optional<std::size_t> find_json_string_end(
    const std::string& text,
    const std::size_t begin,
    const std::size_t end) {
  if (begin >= end || text[begin] != '"') {
    return std::nullopt;
  }
  bool escaped = false;
  for (std::size_t pos = begin + 1; pos < end; ++pos) {
    if (escaped) {
      escaped = false;
    } else if (text[pos] == '\\') {
      escaped = true;
    } else if (text[pos] == '"') {
      return pos;
    }
  }
  return std::nullopt;
}

inline std::optional<std::size_t> find_json_container_end(
    const std::string& text,
    const std::size_t begin,
    const std::size_t end) {
  if (begin >= end || (text[begin] != '{' && text[begin] != '[')) {
    return std::nullopt;
  }
  int depth = 0;
  for (std::size_t pos = begin; pos < end; ++pos) {
    if (text[pos] == '"') {
      const auto string_end = find_json_string_end(text, pos, end);
      if (!string_end.has_value()) {
        return std::nullopt;
      }
      pos = *string_end;
      continue;
    }
    if (text[pos] == '{' || text[pos] == '[') {
      ++depth;
    } else if (text[pos] == '}' || text[pos] == ']') {
      --depth;
      if (depth == 0) {
        return pos;
      }
    }
  }
  return std::nullopt;
}

inline std::optional<std::size_t> find_direct_json_member(
    const std::string& text,
    const std::size_t object_begin,
    const std::size_t object_end,
    const std::string_view key) {
  std::size_t pos =
      skip_json_whitespace(text, object_begin + 1, object_end);
  while (pos < object_end) {
    const auto key_end = find_json_string_end(text, pos, object_end);
    if (!key_end.has_value()) {
      return std::nullopt;
    }
    const bool key_matches =
        text.compare(pos + 1, *key_end - pos - 1, key) == 0;
    pos = skip_json_whitespace(text, *key_end + 1, object_end);
    if (pos >= object_end || text[pos] != ':') {
      return std::nullopt;
    }
    pos = skip_json_whitespace(text, pos + 1, object_end);
    if (key_matches) {
      return pos;
    }

    if (pos < object_end && text[pos] == '"') {
      const auto value_end = find_json_string_end(text, pos, object_end);
      if (!value_end.has_value()) {
        return std::nullopt;
      }
      pos = *value_end + 1;
    } else if (pos < object_end &&
               (text[pos] == '{' || text[pos] == '[')) {
      const auto value_end = find_json_container_end(text, pos, object_end);
      if (!value_end.has_value()) {
        return std::nullopt;
      }
      pos = *value_end + 1;
    } else {
      while (pos < object_end && text[pos] != ',') {
        ++pos;
      }
    }
    pos = skip_json_whitespace(text, pos, object_end);
    if (pos < object_end && text[pos] == ',') {
      pos = skip_json_whitespace(text, pos + 1, object_end);
    }
  }
  return std::nullopt;
}

inline std::optional<bool> swept_volume_flag_from_frozen_config(
    const std::string& frozen_config) {
  const std::size_t root_begin =
      skip_json_whitespace(frozen_config, 0, frozen_config.size());
  const auto root_end =
      find_json_container_end(frozen_config, root_begin, frozen_config.size());
  if (!root_end.has_value()) {
    return std::nullopt;
  }
  const auto numerics_begin = find_direct_json_member(
      frozen_config, root_begin, *root_end, "numerics");
  if (!numerics_begin.has_value() || frozen_config[*numerics_begin] != '{') {
    return std::nullopt;
  }
  const auto numerics_end = find_json_container_end(
      frozen_config, *numerics_begin, *root_end + 1);
  if (!numerics_end.has_value()) {
    return std::nullopt;
  }
  const auto ale_begin = find_direct_json_member(
      frozen_config, *numerics_begin, *numerics_end, "ale");
  if (!ale_begin.has_value() || frozen_config[*ale_begin] != '{') {
    return std::nullopt;
  }
  const auto ale_end = find_json_container_end(
      frozen_config, *ale_begin, *numerics_end + 1);
  if (!ale_end.has_value()) {
    return std::nullopt;
  }

  auto flag_begin = find_direct_json_member(
      frozen_config, *ale_begin, *ale_end, "swept_volume_sign_fixed");
  if (!flag_begin.has_value()) {
    flag_begin = find_direct_json_member(
        frozen_config, *ale_begin, *ale_end, "donor_sign_fixed");
  }
  if (!flag_begin.has_value()) {
    return std::nullopt;
  }
  if (frozen_config.compare(*flag_begin, 4, "true") == 0) {
    return true;
  }
  if (frozen_config.compare(*flag_begin, 5, "false") == 0) {
    return false;
  }
  return std::nullopt;
}

inline std::optional<bool> plic_enabled_from_frozen_config(
    const std::string& frozen_config) {
  const std::size_t root_begin =
      skip_json_whitespace(frozen_config, 0, frozen_config.size());
  const auto root_end =
      find_json_container_end(frozen_config, root_begin, frozen_config.size());
  if (!root_end.has_value()) {
    return std::nullopt;
  }
  const auto numerics_begin = find_direct_json_member(
      frozen_config, root_begin, *root_end, "numerics");
  if (!numerics_begin.has_value() || frozen_config[*numerics_begin] != '{') {
    return std::nullopt;
  }
  const auto numerics_end = find_json_container_end(
      frozen_config, *numerics_begin, *root_end + 1);
  if (!numerics_end.has_value()) {
    return std::nullopt;
  }
  const auto plic_begin = find_direct_json_member(
      frozen_config, *numerics_begin, *numerics_end, "plic");
  if (!plic_begin.has_value() || frozen_config[*plic_begin] != '{') {
    return std::nullopt;
  }
  const auto plic_end = find_json_container_end(
      frozen_config, *plic_begin, *numerics_end + 1);
  if (!plic_end.has_value()) {
    return std::nullopt;
  }
  const auto enabled_begin = find_direct_json_member(
      frozen_config, *plic_begin, *plic_end, "enabled");
  if (!enabled_begin.has_value()) {
    return std::nullopt;
  }
  if (frozen_config.compare(*enabled_begin, 4, "true") == 0) {
    return true;
  }
  if (frozen_config.compare(*enabled_begin, 5, "false") == 0) {
    return false;
  }
  return std::nullopt;
}

}  // namespace tenryu::io
