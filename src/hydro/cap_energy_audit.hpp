#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>

namespace tenryu::hydro {

constexpr int kI1BSpuriousSensorTopKMax = 8;

struct I1BSpuriousSensorCell {
  int rank = -1;
  int cell_id = -1;
  int active_nverts = 0;
  double residual_ke = 0.0;
  double affine_ke = 0.0;
  double eta2 = 0.0;
  double score = 0.0;
};

struct I1BSpuriousSensorSummary {
  bool valid = false;
  int top_count = 0;
  int sampled_cells = 0;
  double residual_ke_sum = 0.0;
  double residual_ke_max = 0.0;
  std::array<I1BSpuriousSensorCell, kI1BSpuriousSensorTopKMax> top{};
};

inline bool env_flag_enabled(const char* const name) {
  const char* const raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return false;
  }
  const std::string value(raw);
  return value == "1" || value == "true" || value == "TRUE" ||
         value == "on" || value == "ON";
}

inline bool cap_energy_audit_enabled() {
  static const bool enabled = env_flag_enabled("TENRYU_CAP_ENERGY_AUDIT");
  return enabled;
}

inline bool i1b_spurious_sensor_enabled() {
  static const bool enabled = env_flag_enabled("TENRYU_I1B_SPURIOUS_SENSOR");
  return enabled;
}

inline int i1b_spurious_sensor_top_k() {
  static const int top_k = []() {
    const char* const raw = std::getenv("TENRYU_I1B_SPURIOUS_SENSOR_TOPK");
    if (raw == nullptr || raw[0] == '\0') {
      return kI1BSpuriousSensorTopKMax;
    }
    try {
      return std::clamp(std::stoi(std::string(raw)), 1,
                        kI1BSpuriousSensorTopKMax);
    } catch (...) {
      return kI1BSpuriousSensorTopKMax;
    }
  }();
  return top_k;
}

}  // namespace tenryu::hydro
