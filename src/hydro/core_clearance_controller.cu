#include "hydro/core_clearance_controller.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/namelist/errors.hpp"
#include "core/state.hpp"
#include "hydro/ale_remap_2d_rz.hpp"
#include "hydro/pentagon_geometry.cuh"
#include "mesh/candidate_mesh_admissibility.hpp"
#include "mesh/mesh.hpp"
#include "mesh/path_admissibility.cuh"
#include "mesh/rz_moments.cuh"
#include "parallel/partition.hpp"

namespace tenryu::hydro {
namespace {

constexpr int kClearanceBinCount = 40;
constexpr double kClearanceOuterRadius = 2.2e-3;
constexpr double kBcrOuterRadius = 3.0e-3;
constexpr int kBcrWatchCell = 65857;
constexpr double kBcrQualityOn = 0.25;
constexpr double kBcrQualityOff = 0.50;
constexpr double kBcrQualityHard = 0.10;  // q_hard: actuation floor
constexpr double kBcrEtaQ = 0.01;         // eta_q: minimum certified gain scale
constexpr double kBcrRelativeSpeedFloor = 1.0e6;
constexpr int kBcrDwellAcceptedSteps = 8;
constexpr int kBcrRampAcceptedSteps = 4;
constexpr int kBcrContinuousSweeps = 4;
constexpr int kBcrContinuousLogCadence = 10;

struct VectorRz {
  double r = 0.0;
  double z = 0.0;
};

struct ReplayPoint {
  double t = 0.0;
  double s_f = 0.0;
  double u_f = 0.0;
};

enum class ClearancePhase {
  STATIC,
  SPLICING,
  TRACKING,
};

enum class BcrPhase {
  ARMED,
  CAPTURE,
  RIDE,
  RECOVERY_HOLD,
  RELEASE_RAMP,
};

enum class G31Variant {
  LEGACY,
  A0,
  A1,
  A2,
};

const char* bcr_phase_name(BcrPhase phase);

struct BcrQualityHistorySample {
  int step = -1;
  double t = 0.0;
  double q_min = -std::numeric_limits<double>::infinity();
};

struct BcrTransition {
  BcrPhase from = BcrPhase::ARMED;
  BcrPhase to = BcrPhase::ARMED;
  int step = -1;
  double t = 0.0;
};

struct BcrReserveDiagnostic {
  int reserve_min = 9;
  int cell = -1;
  int corner = -1;
  double jnow = std::numeric_limits<double>::quiet_NaN();
  double jhat1 = std::numeric_limits<double>::quiet_NaN();
  double jhat4 = std::numeric_limits<double>::quiet_NaN();
};

struct BcrFrontSample {
  bool valid = false;
  double radius = std::numeric_limits<double>::quiet_NaN();
  double inward_speed = 0.0;
};

struct BcrQualityMetrics {
  double q_j_min = std::numeric_limits<double>::infinity();
  double q_v_min = std::numeric_limits<double>::infinity();
  double tau_geom = std::numeric_limits<double>::infinity();
  double tau_hard = std::numeric_limits<double>::infinity();
  double tau_cell = std::numeric_limits<double>::infinity();
  double h_r = std::numeric_limits<double>::quiet_NaN();
  double target_scale = 1.0;
  double shell_radius = std::numeric_limits<double>::quiet_NaN();
  double patch_inner_radius = std::numeric_limits<double>::quiet_NaN();
  bool trailing_clearance = false;
};

struct RankedValue {
  double value = 0.0;
  double weight = 0.0;
  int cell = -1;
};

struct ClearanceSample {
  int step = -1;
  double t = 0.0;
  ClearancePhase phase = ClearancePhase::STATIC;
  bool front_valid = false;
  double s_f = std::numeric_limits<double>::quiet_NaN();
  double u_f = 0.0;
  double tau_hit = std::numeric_limits<double>::infinity();
  double phi = 0.0;
  double r_cmd = 0.0;
  double r_acc = 0.0;
  double beta_eff = std::numeric_limits<double>::quiet_NaN();
  double sigma = -1.0;
  double g_guard = std::numeric_limits<double>::quiet_NaN();
  double h95 = std::numeric_limits<double>::quiet_NaN();
  double u95 = std::numeric_limits<double>::quiet_NaN();
  int n_eligible = 0;
};

struct CoreClearanceState {
  bool initialized = false;
  bool active = false;
  bool bcr_sets = false;
  bool bcr_continuous = false;
  bool bcr_predictor = false;
  bool bcr_predictor_triggered = false;
  bool bcr_rezone = false;
  bool bcr_target = false;
  bool bcr_capture_requested = false;
  bool bcr_feasible_seed_requested = false;
  bool bcr_guard_expanded = false;
  bool bcr_hold_heal = false;
  bool multirank_unsupported = false;
  bool multirank_logged = false;
  bool flushed = false;
  bool missed_trigger_logged = false;
  bool support_exhausted = false;
  bool support_exhausted_logged = false;
  bool previous_rplus_sample_valid = false;
  bool adot_observed = false;
  int rank = 0;
  int armed_step = -1;
  int tracking_step = -1;
  int low_beta_eff_steps = 0;
  int adot_cell = -1;
  int adot_from_step = 1700;
  int adot_min_jacobian_corner = -1;
  int adot_min_step = -1;
  int bcr_predictor_first_trigger_step = -1;
  int bcr_predictor_triggered_steps = 0;
  int bcr_rezone_invocations = 0;
  int bcr_feasible_seed_cell = -1;
  int bcr_clock_step = -1;
  int bcr_release_ramp_start_step = -1;
  int bcr_last_four_halvings_step = -1;
  int bcr_last_bf_first_reject_step = -1;
  int bcr_last_geometry_retry_step = -1;
  double armed_t = std::numeric_limits<double>::quiet_NaN();
  double tracking_t = std::numeric_limits<double>::quiet_NaN();
  double r0 = 0.0;
  double w0 = 0.0;
  double bin_width = 0.0;
  double tau_lead = 0.0;
  double tau_splice = 0.0;
  double beta = 0.0;
  double u_floor = 0.0;
  double r_cmd = 0.0;
  double r_acc = 0.0;
  double active_h95 = std::numeric_limits<double>::quiet_NaN();
  double active_g_guard = std::numeric_limits<double>::quiet_NaN();
  double active_u_half = 0.0;
  double active_s_end = std::numeric_limits<double>::quiet_NaN();
  double previous_evaluation_t = 0.0;
  double adot_dneg = 0.0;
  double adot_min_area = std::numeric_limits<double>::infinity();
  double adot_min_jacobian = std::numeric_limits<double>::infinity();
  double bcr_clock_t = 0.0;
  double bcr_tau_geom = std::numeric_limits<double>::infinity();
  double bcr_tau_cell = std::numeric_limits<double>::infinity();
  double bcr_q_j_min = std::numeric_limits<double>::infinity();
  double bcr_q_v_min = std::numeric_limits<double>::infinity();
  double bcr_target_scale = 1.0;
  double bcr_h_r = std::numeric_limits<double>::quiet_NaN();
  double bcr_shell_radius = std::numeric_limits<double>::quiet_NaN();
  double bcr_patch_inner_radius = std::numeric_limits<double>::quiet_NaN();
  double bcr_release_ramp_start_t = std::numeric_limits<double>::quiet_NaN();
  double bcr_release_ramp_tau = std::numeric_limits<double>::quiet_NaN();
  double bcr_last_four_halvings_t =
      -std::numeric_limits<double>::infinity();
  double bcr_last_bf_first_reject_t =
      -std::numeric_limits<double>::infinity();
  double bcr_last_geometry_retry_t =
      -std::numeric_limits<double>::infinity();
  // --- G3.1 episode / transaction bookkeeping ---
  int g31_capture_epoch_id = 0;  // increments when an episode opens
  bool g31_episode_open = false;
  bool g31_tau_hard_used = false;  // one-opportunity secondary consumed
  bool g31_trial_reject_pending = false;
  double g31_last_commit_t = std::numeric_limits<double>::quiet_NaN();
  int g31_last_commit_step = -1;
  // q_P^- at last record_step before commit
  double g31_q_pre_commit = std::numeric_limits<double>::quiet_NaN();
  // q_P^+ at first record_step after commit
  double g31_q_post_commit = std::numeric_limits<double>::quiet_NaN();
  // g_m = q_post - q_pre
  double g31_last_gain = std::numeric_limits<double>::quiet_NaN();
  bool g31_awaiting_post_sample = false;  // q_post not yet measured
  double g31_tau_cool = 0.0;  // tau_cell(t_last) captured at commit time
  // Set by note_event(GEOMETRY_RETRY), cleared each record_step.
  bool g31_geometry_retry_this_step = false;
  // Last-known watch metrics stored at record_step for pre_lagrange.
  double g31_tau_hard = std::numeric_limits<double>::infinity();
  int g31_reserve_min = 9;
  double g31_q_min = std::numeric_limits<double>::infinity();
  ClearancePhase phase = ClearancePhase::STATIC;
  BcrPhase bcr_phase = BcrPhase::ARMED;
  std::vector<double> initial_volume;
  std::vector<double> initial_radius;
  std::vector<double> initial_node_r;
  std::vector<double> initial_node_z;
  std::vector<double> initial_node_radius;
  std::vector<double> initial_director_r;
  std::vector<double> initial_director_z;
  std::vector<double> active_omega;
  std::vector<int> controlled_nodes;
  std::vector<ReplayPoint> replay;
  std::vector<double> replay_slope;
  std::uint64_t replay_hash = 0U;
  std::vector<int> cell_node_offsets;
  std::vector<int> cell_node_indices;
  std::vector<std::uint8_t> cell_nverts;
  std::array<std::vector<int>, kClearanceBinCount> bins;
  std::vector<int> wideband_cells;
  std::vector<int> feather_cells;
  std::vector<int> rplus_cells;
  std::vector<int> bcr_b0_cells;
  std::vector<int> bcr_bstar_cells;
  std::vector<int> bcr_bg_cells;
  std::vector<int> bcr_bf_cells;
  std::vector<int> bcr_bf_ring1_cells;
  std::vector<int> bcr_bf_ring2_cells;
  std::vector<int> bcr_patch_cells;
  std::vector<std::uint8_t> bcr_priority_cells;
  std::vector<double> bcr_omega;
  std::vector<BcrQualityHistorySample> bcr_dwell_history;
  std::vector<BcrTransition> bcr_transitions;
  std::vector<double> previous_rplus_volume;
  ClearanceSample last_sample;
};

bool core_clearance_shadow_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_CLEARANCE_SHADOW");
    return raw != nullptr && std::strcmp(raw, "1") == 0;
  }();
  return enabled;
}

bool bcr_sets_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_BCR_SETS");
    return raw != nullptr && std::strcmp(raw, "1") == 0;
  }();
  return enabled;
}

bool bcr_continuous_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_BCR_CONTINUOUS");
    return raw != nullptr && std::strcmp(raw, "1") == 0;
  }();
  return enabled;
}

bool bcr_predictor_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_BCR_PREDICTOR");
    return raw != nullptr && std::strcmp(raw, "1") == 0;
  }();
  return enabled;
}

G31Variant g31_variant() {
  static const G31Variant variant = [] {
    const char* raw = std::getenv("TENRYU_I1B_G31_VARIANT");
    if (raw == nullptr || raw[0] == '\0') {
      return G31Variant::LEGACY;
    }
    if (std::strcmp(raw, "A0") == 0) {
      return G31Variant::A0;
    }
    if (std::strcmp(raw, "A1") == 0) {
      return G31Variant::A1;
    }
    if (std::strcmp(raw, "A2") == 0) {
      return G31Variant::A2;
    }
    TENRYU_ASSERT(false,
                  "TENRYU_I1B_G31_VARIANT must be A0, A1, or A2");
    return G31Variant::LEGACY;
  }();
  return variant;
}

bool bcr_rezone_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_BCR_REZONE");
    return raw != nullptr && std::strcmp(raw, "1") == 0;
  }();
  return enabled;
}

bool bcr_target_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_BCR_TARGET");
    return raw != nullptr && std::strcmp(raw, "1") == 0;
  }();
  return enabled;
}

int bcr_iteration_count() {
  static const int iteration_count = [] {
    const char* raw = std::getenv("TENRYU_I1B_BCR_ITERS");
    if (raw == nullptr || raw[0] == '\0') {
      return 8;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed < 1 || parsed > 64) {
      return 8;
    }
    return static_cast<int>(parsed);
  }();
  return iteration_count;
}

double positive_env_value(const char* name, const double fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const double parsed = std::strtod(raw, &end);
  if (end == raw || *end != '\0' || !std::isfinite(parsed) ||
      !(parsed > 0.0)) {
    return fallback;
  }
  return parsed;
}

double nonnegative_env_value(const char* name, const double fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const double parsed = std::strtod(raw, &end);
  if (end == raw || *end != '\0' || !std::isfinite(parsed) || parsed < 0.0) {
    return fallback;
  }
  return parsed;
}

int core_clearance_cadence_steps() {
  static const int cadence_steps = [] {
    const char* raw = std::getenv("TENRYU_I1B_CLR_CADENCE");
    if (raw == nullptr || raw[0] == '\0') {
      return 5;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed < 1 ||
        parsed > static_cast<long>(std::numeric_limits<int>::max())) {
      return 5;
    }
    return static_cast<int>(parsed);
  }();
  return cadence_steps;
}

int adot_ledger_cell() {
  static const int cell = [] {
    const char* raw = std::getenv("TENRYU_I1B_ADOT_LEDGER");
    if (raw == nullptr || raw[0] == '\0') {
      return -1;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed < 0 ||
        parsed > static_cast<long>(std::numeric_limits<int>::max())) {
      return -1;
    }
    return static_cast<int>(parsed);
  }();
  return cell;
}

int adot_ledger_from_step() {
  static const int from_step = [] {
    const char* raw = std::getenv("TENRYU_I1B_ADOT_FROM");
    if (raw == nullptr || raw[0] == '\0') {
      return 1700;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed < 0 ||
        parsed > static_cast<long>(std::numeric_limits<int>::max())) {
      return 1700;
    }
    return static_cast<int>(parsed);
  }();
  return from_step;
}

CoreClearanceState& core_clearance_state() {
  static CoreClearanceState state;
  return state;
}

bool& core_clearance_active_requested() {
  static bool requested = false;
  return requested;
}

double quiet_nan() {
  return std::numeric_limits<double>::quiet_NaN();
}

std::uint64_t fnv1a64(const std::string& bytes) {
  std::uint64_t hash = UINT64_C(14695981039346656037);
  for (const unsigned char byte : bytes) {
    hash ^= static_cast<std::uint64_t>(byte);
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

long long parse_integer_field(const std::string& field,
                              const std::string& path,
                              const int line_number) {
  if (field.empty()) {
    throw core::namelist::ConfigError(
        path + ": empty step field at line " + std::to_string(line_number));
  }
  char* end = nullptr;
  const long long value = std::strtoll(field.c_str(), &end, 10);
  if (end == field.c_str() || *end != '\0') {
    throw core::namelist::ConfigError(
        path + ": invalid step field at line " + std::to_string(line_number));
  }
  return value;
}

double parse_double_field(const std::string& field,
                          const std::string& path,
                          const int line_number,
                          const char* name) {
  if (field.empty()) {
    throw core::namelist::ConfigError(
        path + ": empty " + name + " field at line " +
        std::to_string(line_number));
  }
  char* end = nullptr;
  const double value = std::strtod(field.c_str(), &end);
  if (end == field.c_str() || *end != '\0' || !std::isfinite(value)) {
    throw core::namelist::ConfigError(
        path + ": invalid " + name + " field at line " +
        std::to_string(line_number));
  }
  return value;
}

double pchip_endpoint_slope(const double h0,
                            const double h1,
                            const double delta0,
                            const double delta1) {
  double slope = ((2.0 * h0 + h1) * delta0 - h0 * delta1) /
                 (h0 + h1);
  if (slope * delta0 <= 0.0) {
    return 0.0;
  }
  if (delta0 * delta1 < 0.0 && std::abs(slope) > 3.0 * std::abs(delta0)) {
    return 3.0 * delta0;
  }
  return slope;
}

void build_replay_slopes(CoreClearanceState& controller) {
  const std::size_t count = controller.replay.size();
  controller.replay_slope.assign(count, 0.0);
  if (count == 2U) {
    const double slope =
        (controller.replay[1].s_f - controller.replay[0].s_f) /
        (controller.replay[1].t - controller.replay[0].t);
    controller.replay_slope[0] = slope;
    controller.replay_slope[1] = slope;
    return;
  }

  std::vector<double> h(count - 1U, 0.0);
  std::vector<double> delta(count - 1U, 0.0);
  for (std::size_t i = 0; i + 1U < count; ++i) {
    h[i] = controller.replay[i + 1U].t - controller.replay[i].t;
    delta[i] =
        (controller.replay[i + 1U].s_f - controller.replay[i].s_f) / h[i];
  }
  controller.replay_slope.front() =
      pchip_endpoint_slope(h[0], h[1], delta[0], delta[1]);
  for (std::size_t i = 1; i + 1U < count; ++i) {
    if (delta[i - 1U] == 0.0 || delta[i] == 0.0 ||
        delta[i - 1U] * delta[i] <= 0.0) {
      controller.replay_slope[i] = 0.0;
      continue;
    }
    const double w1 = 2.0 * h[i] + h[i - 1U];
    const double w2 = h[i] + 2.0 * h[i - 1U];
    controller.replay_slope[i] =
        (w1 + w2) / (w1 / delta[i - 1U] + w2 / delta[i]);
  }
  controller.replay_slope.back() = pchip_endpoint_slope(
      h[count - 2U],
      h[count - 3U],
      delta[count - 2U],
      delta[count - 3U]);
}

void load_replay_table(CoreClearanceState& controller,
                       const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw core::namelist::ConfigError(
        "Numerics.ale.euler_window.replay_table_path cannot be opened: " +
        path);
  }
  const std::string raw((std::istreambuf_iterator<char>(input)),
                        std::istreambuf_iterator<char>());
  if (!input.eof() && input.fail()) {
    throw core::namelist::ConfigError(
        "failed while reading replay table: " + path);
  }
  controller.replay_hash = fnv1a64(raw);

  std::istringstream rows(raw);
  std::string line;
  int line_number = 0;
  while (std::getline(rows, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    const std::size_t tab0 = line.find('\t');
    const std::size_t tab1 = tab0 == std::string::npos
                                 ? std::string::npos
                                 : line.find('\t', tab0 + 1U);
    const std::size_t tab2 = tab1 == std::string::npos
                                 ? std::string::npos
                                 : line.find('\t', tab1 + 1U);
    const std::size_t tab3 = tab2 == std::string::npos
                                 ? std::string::npos
                                 : line.find('\t', tab2 + 1U);
    if (tab0 == std::string::npos || tab1 == std::string::npos ||
        tab2 == std::string::npos || tab3 != std::string::npos) {
      throw core::namelist::ConfigError(
          path + ": expected four tab-separated fields at line " +
          std::to_string(line_number));
    }
    (void)parse_integer_field(line.substr(0, tab0), path, line_number);
    ReplayPoint point;
    point.t = parse_double_field(
        line.substr(tab0 + 1U, tab1 - tab0 - 1U),
        path,
        line_number,
        "t");
    point.s_f = parse_double_field(
        line.substr(tab1 + 1U, tab2 - tab1 - 1U),
        path,
        line_number,
        "s_f");
    point.u_f = parse_double_field(
        line.substr(tab2 + 1U), path, line_number, "U_f");
    if (!(point.s_f > 0.0)) {
      throw core::namelist::ConfigError(
          path + ": s_f must be positive at line " +
          std::to_string(line_number));
    }
    if (!controller.replay.empty()) {
      const ReplayPoint& previous = controller.replay.back();
      if (!(point.t > previous.t)) {
        throw core::namelist::ConfigError(
            path + ": t must be strictly increasing at line " +
            std::to_string(line_number));
      }
      if (point.s_f > previous.s_f) {
        throw core::namelist::ConfigError(
            path + ": s_f must be monotone non-increasing at line " +
            std::to_string(line_number));
      }
    }
    controller.replay.push_back(point);
  }
  if (controller.replay.size() < 2U) {
    throw core::namelist::ConfigError(
        path + ": replay table requires at least two rows");
  }
  build_replay_slopes(controller);
}

ReplayPoint interpolate_replay(const CoreClearanceState& controller,
                               const double t) {
  TENRYU_ASSERT(!controller.replay.empty() &&
                    controller.replay_slope.size() ==
                        controller.replay.size(),
                "clearance replay interpolant is not initialized");
  if (t <= controller.replay.front().t) {
    ReplayPoint result = controller.replay.front();
    result.u_f =
        std::max(result.u_f, std::numeric_limits<double>::min());
    return result;
  }
  if (t >= controller.replay.back().t) {
    ReplayPoint result = controller.replay.back();
    result.u_f =
        std::max(result.u_f, std::numeric_limits<double>::min());
    return result;
  }
  const auto upper = std::upper_bound(
      controller.replay.begin(),
      controller.replay.end(),
      t,
      [](const double value, const ReplayPoint& point) {
        return value < point.t;
      });
  const std::size_t i =
      static_cast<std::size_t>(upper - controller.replay.begin() - 1);
  const ReplayPoint& left = controller.replay[i];
  const ReplayPoint& right = controller.replay[i + 1U];
  const double h = right.t - left.t;
  const double x = (t - left.t) / h;
  const double x2 = x * x;
  const double x3 = x2 * x;
  const double h00 = 2.0 * x3 - 3.0 * x2 + 1.0;
  const double h10 = x3 - 2.0 * x2 + x;
  const double h01 = -2.0 * x3 + 3.0 * x2;
  const double h11 = x3 - x2;
  ReplayPoint result;
  result.t = t;
  result.s_f = h00 * left.s_f + h10 * h * controller.replay_slope[i] +
               h01 * right.s_f +
               h11 * h * controller.replay_slope[i + 1U];
  result.u_f = std::max(
      left.u_f + x * (right.u_f - left.u_f),
      std::numeric_limits<double>::min());
  return result;
}

int active_nverts(const CoreClearanceState& controller, const int cell) {
  return mesh::mesh_topo_cell_active_nverts(controller.cell_nverts, cell);
}

void validate_cell_csr(const CoreClearanceState& controller,
                       const int cell,
                       const int n_nodes) {
  TENRYU_ASSERT(cell >= 0 &&
                    static_cast<std::size_t>(cell) + 1U <
                        controller.cell_node_offsets.size(),
                "core clearance cell-node CSR offset is missing");
  const int offset =
      controller.cell_node_offsets[static_cast<std::size_t>(cell)];
  const int end =
      controller.cell_node_offsets[static_cast<std::size_t>(cell) + 1U];
  const int nverts = active_nverts(controller, cell);
  TENRYU_ASSERT(offset >= 0 && end - offset >= nverts &&
                    static_cast<std::size_t>(end) <=
                        controller.cell_node_indices.size(),
                "core clearance cell-node CSR is incomplete");
  for (int corner = 0; corner < nverts; ++corner) {
    const int node = controller.cell_node_indices[
        static_cast<std::size_t>(offset + corner)];
    TENRYU_ASSERT(node >= 0 && node < n_nodes,
                  "core clearance cell node is out of range");
  }
}

VectorRz cell_vertex_mean(const CoreClearanceState& controller,
                          const int cell,
                          const std::vector<double>& node_r,
                          const std::vector<double>& node_z) {
  const int offset =
      controller.cell_node_offsets[static_cast<std::size_t>(cell)];
  const int nverts = active_nverts(controller, cell);
  double sum_r = 0.0;
  double sum_z = 0.0;
  for (int corner = 0; corner < nverts; ++corner) {
    const int node = controller.cell_node_indices[
        static_cast<std::size_t>(offset + corner)];
    sum_r += node_r[static_cast<std::size_t>(node)];
    sum_z += node_z[static_cast<std::size_t>(node)];
  }
  const double inverse_count = 1.0 / static_cast<double>(nverts);
  return {sum_r * inverse_count, sum_z * inverse_count};
}

std::vector<int> cells_from_mask(const std::vector<std::uint8_t>& mask) {
  std::vector<int> cells;
  cells.reserve(mask.size());
  for (std::size_t cell = 0; cell < mask.size(); ++cell) {
    if (mask[cell] != 0U) {
      cells.push_back(static_cast<int>(cell));
    }
  }
  return cells;
}

std::vector<std::uint8_t> nodal_cell_expansion(
    const CoreClearanceState& controller,
    const std::vector<int>& seed_cells,
    const std::vector<std::vector<int>>& node_cells,
    const int n_cells) {
  std::vector<std::uint8_t> touched_nodes(node_cells.size(), 0U);
  for (const int cell : seed_cells) {
    const int offset = controller.cell_node_offsets[
        static_cast<std::size_t>(cell)];
    const int nverts = active_nverts(controller, cell);
    for (int corner = 0; corner < nverts; ++corner) {
      const int node = controller.cell_node_indices[
          static_cast<std::size_t>(offset + corner)];
      touched_nodes[static_cast<std::size_t>(node)] = 1U;
    }
  }

  std::vector<std::uint8_t> expanded(
      static_cast<std::size_t>(n_cells), 0U);
  for (std::size_t node = 0; node < node_cells.size(); ++node) {
    if (touched_nodes[node] == 0U) {
      continue;
    }
    for (const int cell : node_cells[node]) {
      expanded[static_cast<std::size_t>(cell)] = 1U;
    }
  }
  return expanded;
}

void remove_cells(std::vector<std::uint8_t>& mask,
                  const std::vector<std::uint8_t>& excluded) {
  TENRYU_ASSERT(mask.size() == excluded.size(),
                "BCR cell-mask sizes differ");
  for (std::size_t cell = 0; cell < mask.size(); ++cell) {
    if (excluded[cell] != 0U) {
      mask[cell] = 0U;
    }
  }
}

std::vector<std::uint8_t> union_masks(
    const std::vector<std::uint8_t>& lhs,
    const std::vector<std::uint8_t>& rhs) {
  TENRYU_ASSERT(lhs.size() == rhs.size(),
                "BCR cell-mask sizes differ");
  std::vector<std::uint8_t> result(lhs.size(), 0U);
  for (std::size_t cell = 0; cell < lhs.size(); ++cell) {
    result[cell] = (lhs[cell] != 0U || rhs[cell] != 0U) ? 1U : 0U;
  }
  return result;
}

void build_bcr_feather_rings(
    const CoreClearanceState& controller,
    const std::vector<std::vector<int>>& node_cells,
    const std::vector<std::uint8_t>& inner_mask,
    const std::vector<int>& guard_cells,
    std::vector<std::uint8_t>& ring1_mask,
    std::vector<std::uint8_t>& ring2_mask) {
  const int n_cells = static_cast<int>(inner_mask.size());
  ring1_mask = nodal_cell_expansion(
      controller, guard_cells, node_cells, n_cells);
  remove_cells(ring1_mask, inner_mask);

  const std::vector<int> ring1_cells = cells_from_mask(ring1_mask);
  ring2_mask = nodal_cell_expansion(
      controller, ring1_cells, node_cells, n_cells);
  const std::vector<std::uint8_t> through_ring1 =
      union_masks(inner_mask, ring1_mask);
  remove_cells(ring2_mask, through_ring1);
}

std::vector<int> bcr_feather_offenders(
    const CoreClearanceState& controller,
    const std::vector<int>& feather_cells,
    const std::vector<std::uint8_t>& pentagon_nodes) {
  std::vector<int> offenders;
  for (const int cell : feather_cells) {
    bool shares_pentagon_node = false;
    const int offset = controller.cell_node_offsets[
        static_cast<std::size_t>(cell)];
    const int nverts = active_nverts(controller, cell);
    for (int corner = 0; corner < nverts; ++corner) {
      const int node = controller.cell_node_indices[
          static_cast<std::size_t>(offset + corner)];
      shares_pentagon_node =
          shares_pentagon_node ||
          pentagon_nodes[static_cast<std::size_t>(node)] != 0U;
    }
    if (nverts != 4 || shares_pentagon_node) {
      offenders.push_back(cell);
    }
  }
  return offenders;
}

double bcr_quintic_weight(const int distance) {
  const double x = static_cast<double>(distance) / 3.0;
  const double s5 = x * x * x * (10.0 + x * (-15.0 + 6.0 * x));
  return 1.0 - s5;
}

void assign_bcr_node_distance(
    const CoreClearanceState& controller,
    const std::vector<int>& cells,
    const int distance,
    std::vector<int>& node_distance) {
  for (const int cell : cells) {
    const int offset = controller.cell_node_offsets[
        static_cast<std::size_t>(cell)];
    const int nverts = active_nverts(controller, cell);
    for (int corner = 0; corner < nverts; ++corner) {
      const int node = controller.cell_node_indices[
          static_cast<std::size_t>(offset + corner)];
      int& assigned = node_distance[static_cast<std::size_t>(node)];
      if (assigned < 0 || distance < assigned) {
        assigned = distance;
      }
    }
  }
}

void build_bcr_sets(CoreClearanceState& controller,
                    const int n_cells,
                    const int n_nodes) {
  const std::size_t cell_count = static_cast<std::size_t>(n_cells);
  std::vector<std::vector<int>> node_cells(
      static_cast<std::size_t>(n_nodes));
  std::vector<std::uint8_t> pentagon_nodes(
      static_cast<std::size_t>(n_nodes), 0U);
  std::vector<std::uint8_t> b0_mask(cell_count, 0U);
  std::array<int, 3> pentagons_per_band{};
  int nonfinite_pentagons = 0;

  for (int cell = 0; cell < n_cells; ++cell) {
    const int offset = controller.cell_node_offsets[
        static_cast<std::size_t>(cell)];
    const int nverts = active_nverts(controller, cell);
    for (int corner = 0; corner < nverts; ++corner) {
      const int node = controller.cell_node_indices[
          static_cast<std::size_t>(offset + corner)];
      node_cells[static_cast<std::size_t>(node)].push_back(cell);
      if (nverts == 5) {
        pentagon_nodes[static_cast<std::size_t>(node)] = 1U;
      }
    }
    if (nverts != 5) {
      continue;
    }
    const double radius = controller.initial_radius[
        static_cast<std::size_t>(cell)];
    if (!std::isfinite(radius)) {
      ++nonfinite_pentagons;
    } else if (radius <= controller.r0) {
      ++pentagons_per_band[0];
    } else if (radius < kBcrOuterRadius) {
      ++pentagons_per_band[1];
      b0_mask[static_cast<std::size_t>(cell)] = 1U;
    } else {
      ++pentagons_per_band[2];
    }
  }

  controller.bcr_b0_cells = cells_from_mask(b0_mask);
  std::vector<std::uint8_t> bstar_mask = nodal_cell_expansion(
      controller, controller.bcr_b0_cells, node_cells, n_cells);
  controller.bcr_bstar_cells = cells_from_mask(bstar_mask);

  std::vector<std::uint8_t> bg_mask = nodal_cell_expansion(
      controller, controller.bcr_bstar_cells, node_cells, n_cells);
  remove_cells(bg_mask, union_masks(bstar_mask, b0_mask));
  controller.bcr_bg_cells = cells_from_mask(bg_mask);

  std::vector<std::uint8_t> inner_mask =
      union_masks(union_masks(b0_mask, bstar_mask), bg_mask);
  std::vector<std::uint8_t> ring1_mask;
  std::vector<std::uint8_t> ring2_mask;
  build_bcr_feather_rings(controller, node_cells, inner_mask,
                          controller.bcr_bg_cells, ring1_mask, ring2_mask);
  std::vector<std::uint8_t> bf_mask = union_masks(ring1_mask, ring2_mask);
  std::vector<int> offenders = bcr_feather_offenders(
      controller, cells_from_mask(bf_mask), pentagon_nodes);

  if (!offenders.empty()) {
    controller.bcr_guard_expanded = true;
    bg_mask = union_masks(bg_mask, ring1_mask);
    controller.bcr_bg_cells = cells_from_mask(bg_mask);
    inner_mask = union_masks(union_masks(b0_mask, bstar_mask), bg_mask);
    build_bcr_feather_rings(controller, node_cells, inner_mask,
                            controller.bcr_bg_cells, ring1_mask, ring2_mask);
    bf_mask = union_masks(ring1_mask, ring2_mask);
    offenders = bcr_feather_offenders(
        controller, cells_from_mask(bf_mask), pentagon_nodes);
  }

  controller.bcr_bf_ring1_cells = cells_from_mask(ring1_mask);
  controller.bcr_bf_ring2_cells = cells_from_mask(ring2_mask);
  controller.bcr_bf_cells = cells_from_mask(bf_mask);
  controller.bcr_patch_cells = cells_from_mask(
      union_masks(bstar_mask, bg_mask));
  controller.bcr_priority_cells.assign(cell_count, 0U);

  std::vector<int> node_distance(static_cast<std::size_t>(n_nodes), -1);
  assign_bcr_node_distance(controller, controller.bcr_bstar_cells, 0,
                           node_distance);
  assign_bcr_node_distance(controller, controller.bcr_bg_cells, 0,
                           node_distance);
  assign_bcr_node_distance(controller, controller.bcr_bf_ring1_cells, 1,
                           node_distance);
  assign_bcr_node_distance(controller, controller.bcr_bf_ring2_cells, 2,
                           node_distance);
  controller.bcr_omega.assign(static_cast<std::size_t>(n_nodes), 0.0);
  int n_omega_one = 0;
  for (int node = 0; node < n_nodes; ++node) {
    const int distance = node_distance[static_cast<std::size_t>(node)];
    double omega = 0.0;
    if (distance == 0) {
      omega = 1.0;
      ++n_omega_one;
    } else if (distance == 1 || distance == 2) {
      omega = bcr_quintic_weight(distance);
    }
    controller.bcr_omega[static_cast<std::size_t>(node)] = omega;
  }

  const bool watch_in_b0 =
      kBcrWatchCell < n_cells &&
      b0_mask[static_cast<std::size_t>(kBcrWatchCell)] != 0U;
  if (controller.rank == 0) {
    std::ostringstream manifest;
    manifest << std::scientific << std::setprecision(3)
             << "[bcr-sets] nB0=" << controller.bcr_b0_cells.size()
             << " nBstar=" << controller.bcr_bstar_cells.size()
             << " nBg=" << controller.bcr_bg_cells.size()
             << " nBf=" << controller.bcr_bf_cells.size()
             << " nOmega1=" << n_omega_one
             << " radial_range=[" << controller.r0
             << "," << kBcrOuterRadius << "]"
             << " pentagons_per_band=[r<=Rout:"
             << pentagons_per_band[0]
             << ",Rout<r<3e-3:" << pentagons_per_band[1]
             << ",r>=3e-3:" << pentagons_per_band[2]
             << ",nonfinite:" << nonfinite_pentagons << "]"
             << " watch65857_B0=" << (watch_in_b0 ? 1 : 0)
             << " Bstar_includes_B0=1"
             << " guard_extra_ring="
             << (controller.bcr_guard_expanded ? 1 : 0)
             << " omega_policy=1-S5(d/3),d=1,2;inner=1;beyond=0"
             << " feather_constraint="
             << (offenders.empty() ? "PASS" : "VIOLATED");
    core::log_info(manifest.str());
    if (!offenders.empty()) {
      std::ostringstream warning;
      warning << "[bcr-sets] LOUD WARNING feather constraint violated"
              << " offending_cells=";
      for (std::size_t slot = 0; slot < offenders.size(); ++slot) {
        if (slot != 0U) {
          warning << ',';
        }
        const int cell = offenders[slot];
        warning << cell << "(nverts=" << active_nverts(controller, cell)
                << ')';
      }
      core::log_warning(warning.str());
    }
  }
  TENRYU_ASSERT(watch_in_b0,
                "BCR watch cell 65857 must belong to B0");
}

void emit_bcr_set_diagnostic(const CoreClearanceState& controller,
                             const core::State& state,
                             const int step) {
  if (controller.rank != 0) {
    return;
  }
  const std::size_t n_nodes =
      static_cast<std::size_t>(state.mesh.topo.n_nodes);
  TENRYU_ASSERT(state.x_r.size() == n_nodes && state.x_z.size() == n_nodes,
                "BCR diagnostic node field size changed");
  std::vector<double> node_r;
  std::vector<double> node_z;
  state.x_r.copy_to_host(node_r);
  state.x_z.copy_to_host(node_z);

  double minimum_jacobian = std::numeric_limits<double>::infinity();
  int minimum_cell = -1;
  int minimum_corner = -1;
  for (const int cell : controller.bcr_patch_cells) {
    const int offset = controller.cell_node_offsets[
        static_cast<std::size_t>(cell)];
    const int nverts = active_nverts(controller, cell);
    for (int corner = 0; corner < nverts; ++corner) {
      const int previous = (corner == 0) ? nverts - 1 : corner - 1;
      const int next = (corner + 1 == nverts) ? 0 : corner + 1;
      const int node = controller.cell_node_indices[
          static_cast<std::size_t>(offset + corner)];
      const int previous_node = controller.cell_node_indices[
          static_cast<std::size_t>(offset + previous)];
      const int next_node = controller.cell_node_indices[
          static_cast<std::size_t>(offset + next)];
      const PentagonPoint next_edge{
          node_r[static_cast<std::size_t>(next_node)] -
              node_r[static_cast<std::size_t>(node)],
          node_z[static_cast<std::size_t>(next_node)] -
              node_z[static_cast<std::size_t>(node)],
      };
      const PentagonPoint previous_edge{
          node_r[static_cast<std::size_t>(previous_node)] -
              node_r[static_cast<std::size_t>(node)],
          node_z[static_cast<std::size_t>(previous_node)] -
              node_z[static_cast<std::size_t>(node)],
      };
      const double jacobian =
          next_edge.r * previous_edge.z -
          next_edge.z * previous_edge.r;
      if (jacobian < minimum_jacobian) {
        minimum_jacobian = jacobian;
        minimum_cell = cell;
        minimum_corner = corner;
      }
    }
  }

  std::ostringstream line;
  line << std::scientific << std::setprecision(6)
       << "[bcr-sets] step=" << step
       << " Jmin_patch=" << minimum_jacobian
       << " cell=" << minimum_cell
       << " k=" << minimum_corner;
  core::log_info(line.str());
}

double planar_cross(const PentagonPoint& lhs, const PentagonPoint& rhs) {
  return lhs.r * rhs.z - lhs.z * rhs.r;
}

BcrReserveDiagnostic compute_bcr_reserve(
    const CoreClearanceState& controller,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const std::vector<double>& velocity_r,
    const std::vector<double>& velocity_z,
    const double dt) {
  BcrReserveDiagnostic diagnostic;
  for (const int cell : controller.bcr_patch_cells) {
    const int offset = controller.cell_node_offsets[
        static_cast<std::size_t>(cell)];
    const int nverts = active_nverts(controller, cell);
    for (int corner = 0; corner < nverts; ++corner) {
      const int previous = (corner == 0) ? nverts - 1 : corner - 1;
      const int next = (corner + 1 == nverts) ? 0 : corner + 1;
      const int node = controller.cell_node_indices[
          static_cast<std::size_t>(offset + corner)];
      const int previous_node = controller.cell_node_indices[
          static_cast<std::size_t>(offset + previous)];
      const int next_node = controller.cell_node_indices[
          static_cast<std::size_t>(offset + next)];
      const std::size_t node_index = static_cast<std::size_t>(node);
      const std::size_t previous_node_index =
          static_cast<std::size_t>(previous_node);
      const std::size_t next_node_index =
          static_cast<std::size_t>(next_node);
      const PentagonPoint next_edge{
          node_r[next_node_index] - node_r[node_index],
          node_z[next_node_index] - node_z[node_index],
      };
      const PentagonPoint previous_edge{
          node_r[previous_node_index] - node_r[node_index],
          node_z[previous_node_index] - node_z[node_index],
      };
      const PentagonPoint next_edge_rate{
          velocity_r[next_node_index] - velocity_r[node_index],
          velocity_z[next_node_index] - velocity_z[node_index],
      };
      const PentagonPoint previous_edge_rate{
          velocity_r[previous_node_index] - velocity_r[node_index],
          velocity_z[previous_node_index] - velocity_z[node_index],
      };
      const double jnow = planar_cross(next_edge, previous_edge);
      double jhat1 = quiet_nan();
      double jhat4 = quiet_nan();
      int reserve = 9;
      for (int m = 1; m <= 8; ++m) {
        const double prediction_dt = static_cast<double>(m) * dt;
        const PentagonPoint predicted_next_edge{
            next_edge.r + prediction_dt * next_edge_rate.r,
            next_edge.z + prediction_dt * next_edge_rate.z,
        };
        const PentagonPoint predicted_previous_edge{
            previous_edge.r + prediction_dt * previous_edge_rate.r,
            previous_edge.z + prediction_dt * previous_edge_rate.z,
        };
        const double predicted_jacobian =
            planar_cross(predicted_next_edge, predicted_previous_edge);
        if (m == 1) {
          jhat1 = predicted_jacobian;
        }
        if (m == 4) {
          jhat4 = predicted_jacobian;
        }
        if (reserve == 9 && predicted_jacobian <= 0.0) {
          reserve = m;
        }
      }
      if (diagnostic.cell < 0 || reserve < diagnostic.reserve_min) {
        diagnostic.reserve_min = reserve;
        diagnostic.cell = cell;
        diagnostic.corner = corner;
        diagnostic.jnow = jnow;
        diagnostic.jhat1 = jhat1;
        diagnostic.jhat4 = jhat4;
      }
    }
  }
  TENRYU_ASSERT(diagnostic.cell >= 0 && diagnostic.corner >= 0,
                "BCR predictor patch is empty");
  return diagnostic;
}

struct BcrShapeCorner {
  int cell = -1;
  int corner = -1;
  int node = -1;
  int previous_node = -1;
  int next_node = -1;
  double priority_weight = 1.0;
  double winv00 = 0.0;
  double winv01 = 0.0;
  double winv10 = 0.0;
  double winv11 = 0.0;
};

struct BcrCoMotionTarget {
  bool enabled = false;
  int fit_node_count = 0;
  double f00 = 1.0;
  double f01 = 0.0;
  double f10 = 0.0;
  double f11 = 1.0;
  double b_r = 0.0;
  double b_z = 0.0;
  double polar_r00 = 1.0;
  double polar_r01 = 0.0;
  double polar_r10 = 0.0;
  double polar_r11 = 1.0;
  double polar_u00 = 1.0;
  double polar_u01 = 0.0;
  double polar_u10 = 0.0;
  double polar_u11 = 1.0;
  double det_f = 1.0;
  double scale = 1.0;
  double hold_weight = 1.0;
  VectorRz initial_centroid;
  VectorRz predicted_centroid;
  std::vector<std::uint8_t> bstar_node_mask;
  std::vector<double> free_r;
  std::vector<double> free_z;
};

std::vector<std::uint8_t> bcr_node_mask_from_cells(
    const CoreClearanceState& controller,
    const std::vector<int>& cells,
    const int n_nodes) {
  std::vector<std::uint8_t> node_mask(
      static_cast<std::size_t>(n_nodes), 0U);
  for (const int cell : cells) {
    const int offset = controller.cell_node_offsets[
        static_cast<std::size_t>(cell)];
    const int nverts = active_nverts(controller, cell);
    for (int corner = 0; corner < nverts; ++corner) {
      const int node = controller.cell_node_indices[
          static_cast<std::size_t>(offset + corner)];
      node_mask[static_cast<std::size_t>(node)] = 1U;
    }
  }
  return node_mask;
}

BcrCoMotionTarget build_bcr_co_motion_target(
    const CoreClearanceState& controller,
    const std::vector<double>& source_r,
    const std::vector<double>& source_z,
    const std::vector<double>& velocity_r,
    const std::vector<double>& velocity_z,
    const double dt,
    const double hold_weight) {
  const int n_nodes = static_cast<int>(source_r.size());
  TENRYU_ASSERT(source_z.size() == source_r.size() &&
                    velocity_r.size() == source_r.size() &&
                    velocity_z.size() == source_r.size() &&
                    controller.initial_node_r.size() == source_r.size() &&
                    controller.initial_node_z.size() == source_r.size(),
                "BCR target affine fit node field size mismatch");
  TENRYU_ASSERT(std::isfinite(dt) && dt >= 0.0 &&
                    std::isfinite(hold_weight) && hold_weight >= 0.0 &&
                    hold_weight <= 1.0,
                "BCR target affine fit metadata is invalid");

  BcrCoMotionTarget target;
  target.enabled = true;
  target.hold_weight = hold_weight;
  target.free_r = source_r;
  target.free_z = source_z;
  target.bstar_node_mask = bcr_node_mask_from_cells(
      controller, controller.bcr_bstar_cells, n_nodes);
  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t index = static_cast<std::size_t>(node);
    if (target.bstar_node_mask[index] == 0U) {
      continue;
    }
    target.initial_centroid.r += controller.initial_node_r[index];
    target.initial_centroid.z += controller.initial_node_z[index];
    target.predicted_centroid.r += source_r[index] + dt * velocity_r[index];
    target.predicted_centroid.z += source_z[index] + dt * velocity_z[index];
    ++target.fit_node_count;
  }
  TENRYU_ASSERT(target.fit_node_count >= 3,
                "BCR target affine fit requires at least three Bstar nodes");
  const double inverse_weight_sum =
      1.0 / static_cast<double>(target.fit_node_count);
  target.initial_centroid.r *= inverse_weight_sum;
  target.initial_centroid.z *= inverse_weight_sum;
  target.predicted_centroid.r *= inverse_weight_sum;
  target.predicted_centroid.z *= inverse_weight_sum;

  double reference_moment00 = 0.0;
  double reference_moment01 = 0.0;
  double reference_moment11 = 0.0;
  double cross_moment00 = 0.0;
  double cross_moment01 = 0.0;
  double cross_moment10 = 0.0;
  double cross_moment11 = 0.0;
  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t index = static_cast<std::size_t>(node);
    if (target.bstar_node_mask[index] == 0U) {
      continue;
    }
    const double reference_r =
        controller.initial_node_r[index] - target.initial_centroid.r;
    const double reference_z =
        controller.initial_node_z[index] - target.initial_centroid.z;
    const double predicted_r =
        source_r[index] + dt * velocity_r[index] -
        target.predicted_centroid.r;
    const double predicted_z =
        source_z[index] + dt * velocity_z[index] -
        target.predicted_centroid.z;
    reference_moment00 += reference_r * reference_r;
    reference_moment01 += reference_r * reference_z;
    reference_moment11 += reference_z * reference_z;
    cross_moment00 += predicted_r * reference_r;
    cross_moment01 += predicted_r * reference_z;
    cross_moment10 += predicted_z * reference_r;
    cross_moment11 += predicted_z * reference_z;
  }
  const double reference_determinant =
      reference_moment00 * reference_moment11 -
      reference_moment01 * reference_moment01;
  TENRYU_ASSERT(std::isfinite(reference_determinant) &&
                    reference_determinant > 0.0,
                "BCR target affine fit reference moment is singular");
  const double inverse_reference00 =
      reference_moment11 / reference_determinant;
  const double inverse_reference01 =
      -reference_moment01 / reference_determinant;
  const double inverse_reference11 =
      reference_moment00 / reference_determinant;
  target.f00 = cross_moment00 * inverse_reference00 +
               cross_moment01 * inverse_reference01;
  target.f01 = cross_moment00 * inverse_reference01 +
               cross_moment01 * inverse_reference11;
  target.f10 = cross_moment10 * inverse_reference00 +
               cross_moment11 * inverse_reference01;
  target.f11 = cross_moment10 * inverse_reference01 +
               cross_moment11 * inverse_reference11;
  TENRYU_ASSERT(std::isfinite(target.f00) && std::isfinite(target.f01) &&
                    std::isfinite(target.f10) && std::isfinite(target.f11),
                "BCR target affine fit is nonfinite");
  target.b_r = target.predicted_centroid.r -
               target.f00 * target.initial_centroid.r -
               target.f01 * target.initial_centroid.z;
  target.b_z = target.predicted_centroid.z -
               target.f10 * target.initial_centroid.r -
               target.f11 * target.initial_centroid.z;
  target.det_f = target.f00 * target.f11 - target.f01 * target.f10;
  TENRYU_ASSERT(std::isfinite(target.det_f),
                "BCR target affine fit determinant is nonfinite");

  const double c00 = target.f00 * target.f00 + target.f10 * target.f10;
  const double c01 = target.f00 * target.f01 + target.f10 * target.f11;
  const double c11 = target.f01 * target.f01 + target.f11 * target.f11;
  // For a symmetric positive 2x2 C, sqrt(C) =
  // (C + sqrt(det(C)) I) / sqrt(trace(C) + 2 sqrt(det(C))).
  const double sqrt_det_c = std::abs(target.det_f);
  const double sqrt_trace_c =
      std::sqrt(c00 + c11 + 2.0 * sqrt_det_c);
  TENRYU_ASSERT(std::isfinite(sqrt_trace_c) && sqrt_trace_c > 0.0 &&
                    sqrt_det_c > 0.0,
                "BCR target polar decomposition is singular");
  target.polar_u00 = (c00 + sqrt_det_c) / sqrt_trace_c;
  target.polar_u01 = c01 / sqrt_trace_c;
  target.polar_u10 = target.polar_u01;
  target.polar_u11 = (c11 + sqrt_det_c) / sqrt_trace_c;
  const double inverse_sqrt_scale = 1.0 / (sqrt_det_c * sqrt_trace_c);
  const double inverse_sqrt00 =
      (c11 + sqrt_det_c) * inverse_sqrt_scale;
  const double inverse_sqrt01 = -c01 * inverse_sqrt_scale;
  const double inverse_sqrt11 =
      (c00 + sqrt_det_c) * inverse_sqrt_scale;
  target.polar_r00 =
      target.f00 * inverse_sqrt00 + target.f01 * inverse_sqrt01;
  target.polar_r01 =
      target.f00 * inverse_sqrt01 + target.f01 * inverse_sqrt11;
  target.polar_r10 =
      target.f10 * inverse_sqrt00 + target.f11 * inverse_sqrt01;
  target.polar_r11 =
      target.f10 * inverse_sqrt01 + target.f11 * inverse_sqrt11;
  TENRYU_ASSERT(std::isfinite(target.polar_r00) &&
                    std::isfinite(target.polar_r01) &&
                    std::isfinite(target.polar_r10) &&
                    std::isfinite(target.polar_r11),
                "BCR target polar factor is nonfinite");

  target.scale = std::sqrt(std::max(target.det_f, 1.0e-30));
  return target;
}

void emit_bcr_target_fit(const int step,
                         const BcrCoMotionTarget& target) {
  std::ostringstream line;
  line << std::scientific << std::setprecision(6)
       << "[bcr-target] step=" << step
       << " s_b=" << target.scale
       << " detF=" << target.det_f
       << " F=[" << target.f00 << ',' << target.f01 << ';'
       << target.f10 << ',' << target.f11 << ']'
       << " b=[" << target.b_r << ',' << target.b_z << ']'
       << " R_fit=[" << target.polar_r00 << ',' << target.polar_r01 << ';'
       << target.polar_r10 << ',' << target.polar_r11 << ']'
       << " R_used=identity"
       << " rotation_filter=v1"
       << " fit_weights=unit"
       << " nfit=" << target.fit_node_count;
  core::log_info(line.str());
}

std::vector<BcrShapeCorner> build_bcr_shape_corners(
    const CoreClearanceState& controller,
    const int n_nodes,
    const double target_scale,
    const std::vector<double>& source_r,
    const std::vector<double>& source_z,
    const double hold_weight) {
  TENRYU_ASSERT(
      controller.initial_node_r.size() == static_cast<std::size_t>(n_nodes) &&
          controller.initial_node_z.size() ==
              static_cast<std::size_t>(n_nodes),
      "BCR rezone requires initial node coordinates");
  std::vector<BcrShapeCorner> corners;
  for (const int cell : controller.bcr_patch_cells) {
    const int offset = controller.cell_node_offsets[
        static_cast<std::size_t>(cell)];
    const int nverts = active_nverts(controller, cell);
    for (int corner = 0; corner < nverts; ++corner) {
      const int previous = (corner == 0) ? nverts - 1 : corner - 1;
      const int next = (corner + 1 == nverts) ? 0 : corner + 1;
      BcrShapeCorner shape_corner;
      shape_corner.cell = cell;
      shape_corner.corner = corner;
      shape_corner.node = controller.cell_node_indices[
          static_cast<std::size_t>(offset + corner)];
      shape_corner.previous_node = controller.cell_node_indices[
          static_cast<std::size_t>(offset + previous)];
      shape_corner.next_node = controller.cell_node_indices[
          static_cast<std::size_t>(offset + next)];
      shape_corner.priority_weight =
          controller.bcr_priority_cells[
              static_cast<std::size_t>(cell)] != 0U
              ? 4.0
              : 1.0;
      const std::size_t node_index =
          static_cast<std::size_t>(shape_corner.node);
      const std::size_t previous_index =
          static_cast<std::size_t>(shape_corner.previous_node);
      const std::size_t next_index =
          static_cast<std::size_t>(shape_corner.next_node);
      TENRYU_ASSERT(source_r.size() == static_cast<std::size_t>(n_nodes) &&
                        source_z.size() == static_cast<std::size_t>(n_nodes) &&
                        std::isfinite(hold_weight) && hold_weight >= 0.0 &&
                        hold_weight <= 1.0,
                    "BCR rezone target blend metadata is invalid");
      const double free_weight = 1.0 - hold_weight;
      const double w00 =
          hold_weight * target_scale *
              (controller.initial_node_r[next_index] -
               controller.initial_node_r[node_index]) +
          free_weight * (source_r[next_index] - source_r[node_index]);
      const double w01 =
          hold_weight * target_scale *
              (controller.initial_node_r[previous_index] -
               controller.initial_node_r[node_index]) +
          free_weight * (source_r[previous_index] - source_r[node_index]);
      const double w10 =
          hold_weight * target_scale *
              (controller.initial_node_z[next_index] -
               controller.initial_node_z[node_index]) +
          free_weight * (source_z[next_index] - source_z[node_index]);
      const double w11 =
          hold_weight * target_scale *
              (controller.initial_node_z[previous_index] -
               controller.initial_node_z[node_index]) +
          free_weight * (source_z[previous_index] - source_z[node_index]);
      const double det_w = w00 * w11 - w01 * w10;
      TENRYU_ASSERT(std::isfinite(det_w) && det_w != 0.0 &&
                        std::isfinite(target_scale) && target_scale > 0.0,
                    "BCR rezone reference corner is singular");
      shape_corner.winv00 = w11 / det_w;
      shape_corner.winv01 = -w01 / det_w;
      shape_corner.winv10 = -w10 / det_w;
      shape_corner.winv11 = w00 / det_w;
      corners.push_back(shape_corner);
    }
  }
  TENRYU_ASSERT(!corners.empty(), "BCR rezone shape corner set is empty");
  return corners;
}

double bcr_shape_corner_metric(const BcrShapeCorner& corner,
                               const std::vector<double>& node_r,
                               const std::vector<double>& node_z,
                               const double det_floor) {
  const std::size_t node = static_cast<std::size_t>(corner.node);
  const std::size_t previous =
      static_cast<std::size_t>(corner.previous_node);
  const std::size_t next = static_cast<std::size_t>(corner.next_node);
  const double g00 = node_r[next] - node_r[node];
  const double g01 = node_r[previous] - node_r[node];
  const double g10 = node_z[next] - node_z[node];
  const double g11 = node_z[previous] - node_z[node];
  const double t00 = g00 * corner.winv00 + g01 * corner.winv10;
  const double t01 = g00 * corner.winv01 + g01 * corner.winv11;
  const double t10 = g10 * corner.winv00 + g11 * corner.winv10;
  const double t11 = g10 * corner.winv01 + g11 * corner.winv11;
  const double det_t = t00 * t11 - t01 * t10;
  if (!std::isfinite(det_t)) {
    return std::numeric_limits<double>::infinity();
  }
  if (det_t <= 0.0) {
    return 1.0e6 * (1.0 - det_t / det_floor);
  }
  const double frobenius2 =
      t00 * t00 + t01 * t01 + t10 * t10 + t11 * t11;
  return frobenius2 / (2.0 * det_t) - 1.0;
}

double bcr_shape_objective(const std::vector<BcrShapeCorner>& corners,
                           const std::vector<int>* corner_subset,
                           const std::vector<double>& node_r,
                           const std::vector<double>& node_z,
                           const double det_floor) {
  double phi = 0.0;
  if (corner_subset == nullptr) {
    for (const BcrShapeCorner& corner : corners) {
      phi += corner.priority_weight *
             bcr_shape_corner_metric(corner, node_r, node_z, det_floor);
    }
    return phi;
  }
  for (const int index : *corner_subset) {
    const BcrShapeCorner& corner =
        corners[static_cast<std::size_t>(index)];
    phi += corner.priority_weight *
           bcr_shape_corner_metric(corner, node_r, node_z, det_floor);
  }
  return phi;
}

std::vector<double> bcr_initial_node_scales(
    const CoreClearanceState& controller,
    const int n_nodes) {
  std::vector<double> scales(
      static_cast<std::size_t>(n_nodes),
      std::numeric_limits<double>::infinity());
  const int n_cells =
      static_cast<int>(controller.cell_node_offsets.size()) - 1;
  for (int cell = 0; cell < n_cells; ++cell) {
    const int offset = controller.cell_node_offsets[
        static_cast<std::size_t>(cell)];
    const int nverts = active_nverts(controller, cell);
    for (int corner = 0; corner < nverts; ++corner) {
      const int next = (corner + 1 == nverts) ? 0 : corner + 1;
      const int node = controller.cell_node_indices[
          static_cast<std::size_t>(offset + corner)];
      const int next_node = controller.cell_node_indices[
          static_cast<std::size_t>(offset + next)];
      const double dr =
          controller.initial_node_r[static_cast<std::size_t>(next_node)] -
          controller.initial_node_r[static_cast<std::size_t>(node)];
      const double dz =
          controller.initial_node_z[static_cast<std::size_t>(next_node)] -
          controller.initial_node_z[static_cast<std::size_t>(node)];
      const double edge_length = std::hypot(dr, dz);
      if (edge_length > 0.0 && std::isfinite(edge_length)) {
        scales[static_cast<std::size_t>(node)] = std::min(
            scales[static_cast<std::size_t>(node)], edge_length);
        scales[static_cast<std::size_t>(next_node)] = std::min(
            scales[static_cast<std::size_t>(next_node)], edge_length);
      }
    }
  }
  return scales;
}

std::vector<std::vector<int>> build_bcr_touching_corners(
    const std::vector<BcrShapeCorner>& corners,
    const int n_nodes) {
  std::vector<std::vector<int>> touching_corners(
      static_cast<std::size_t>(n_nodes));
  for (std::size_t index = 0; index < corners.size(); ++index) {
    const BcrShapeCorner& corner = corners[index];
    touching_corners[static_cast<std::size_t>(corner.node)].push_back(
        static_cast<int>(index));
    touching_corners[static_cast<std::size_t>(corner.previous_node)].push_back(
        static_cast<int>(index));
    touching_corners[static_cast<std::size_t>(corner.next_node)].push_back(
        static_cast<int>(index));
  }
  return touching_corners;
}

bool bcr_node_is_free(
    const CoreClearanceState& controller,
    const std::vector<double>& node_scale,
    const std::vector<std::vector<int>>& touching_corners,
    const int node) {
  const std::size_t node_index = static_cast<std::size_t>(node);
  const double h_v = node_scale[node_index];
  const bool axis = controller.initial_node_r[node_index] == 0.0;
  const bool origin = axis && controller.initial_node_z[node_index] == 0.0;
  return controller.bcr_omega[node_index] > 0.0 &&
         std::isfinite(h_v) && h_v > 0.0 &&
         !touching_corners[node_index].empty() && !origin;
}

double bcr_tether_node_metric(
    const CoreClearanceState& controller,
    const BcrCoMotionTarget& target,
    const std::vector<double>& node_scale,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const int node) {
  constexpr double kLambdaH = 0.1;
  const std::size_t index = static_cast<std::size_t>(node);
  if (!target.enabled || target.bstar_node_mask[index] == 0U ||
      !(controller.bcr_omega[index] > 0.0) ||
      !std::isfinite(node_scale[index]) || !(node_scale[index] > 0.0) ||
      (controller.initial_node_r[index] == 0.0 &&
       controller.initial_node_z[index] == 0.0)) {
    return 0.0;
  }
  TENRYU_ASSERT(target.free_r.size() == node_r.size() &&
                    target.free_z.size() == node_z.size(),
                "BCR tether target blend size mismatch");
  const double hold_target_r =
      target.predicted_centroid.r +
      target.scale * (controller.initial_node_r[index] -
                      target.initial_centroid.r);
  const double hold_target_z =
      target.predicted_centroid.z +
      target.scale * (controller.initial_node_z[index] -
                      target.initial_centroid.z);
  const double target_r = target.hold_weight * hold_target_r +
                          (1.0 - target.hold_weight) * target.free_r[index];
  const double target_z = target.hold_weight * hold_target_z +
                          (1.0 - target.hold_weight) * target.free_z[index];
  const double delta_r = node_r[index] - target_r;
  const double delta_z = node_z[index] - target_z;
  const double h_v = node_scale[index];
  return kLambdaH * (delta_r * delta_r + delta_z * delta_z) / (h_v * h_v);
}

double bcr_rezone_objective(
    const CoreClearanceState& controller,
    const std::vector<BcrShapeCorner>& corners,
    const std::vector<int>* corner_subset,
    const int tether_node,
    const std::vector<double>& node_scale,
    const BcrCoMotionTarget& target,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const double det_floor) {
  double phi = bcr_shape_objective(
      corners, corner_subset, node_r, node_z, det_floor);
  if (!target.enabled) {
    return phi;
  }
  if (tether_node >= 0) {
    return phi + bcr_tether_node_metric(
                     controller, target, node_scale, node_r, node_z,
                     tether_node);
  }
  for (std::size_t node = 0; node < node_r.size(); ++node) {
    phi += bcr_tether_node_metric(
        controller, target, node_scale, node_r, node_z,
        static_cast<int>(node));
  }
  return phi;
}

double bcr_corner_jacobian(const BcrShapeCorner& corner,
                           const std::vector<double>& node_r,
                           const std::vector<double>& node_z) {
  const std::size_t node = static_cast<std::size_t>(corner.node);
  const std::size_t previous =
      static_cast<std::size_t>(corner.previous_node);
  const std::size_t next = static_cast<std::size_t>(corner.next_node);
  return (node_r[next] - node_r[node]) *
             (node_z[previous] - node_z[node]) -
         (node_z[next] - node_z[node]) *
             (node_r[previous] - node_r[node]);
}

int bcr_apply_direct_feasible_seed(
    const CoreClearanceState& controller,
    const std::vector<BcrShapeCorner>& corners,
    const std::vector<double>& node_scale,
    const std::vector<std::vector<int>>& touching_corners,
    std::vector<double>& node_r,
    std::vector<double>& node_z) {
  if (!controller.bcr_feasible_seed_requested ||
      controller.bcr_feasible_seed_cell < 0) {
    return -1;
  }

  int worst_index = -1;
  double worst_jacobian = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index < corners.size(); ++index) {
    if (corners[index].cell != controller.bcr_feasible_seed_cell) {
      continue;
    }
    const double jacobian =
        bcr_corner_jacobian(corners[index], node_r, node_z);
    if (jacobian < worst_jacobian) {
      worst_jacobian = jacobian;
      worst_index = static_cast<int>(index);
    }
  }
  if (worst_index < 0) {
    core::log_warning(
        "[bcr-ladder] feasible seed skipped: promoted cell is not in P");
    return -1;
  }

  const BcrShapeCorner& worst =
      corners[static_cast<std::size_t>(worst_index)];
  if (!bcr_node_is_free(
          controller, node_scale, touching_corners, worst.node)) {
    core::log_warning(
        "[bcr-ladder] feasible seed skipped: worst-corner node is fixed");
    return -1;
  }

  double typical_sum = 0.0;
  int typical_count = 0;
  for (const int index :
       touching_corners[static_cast<std::size_t>(worst.node)]) {
    if (index == worst_index) {
      continue;
    }
    const double jacobian = std::abs(bcr_corner_jacobian(
        corners[static_cast<std::size_t>(index)], node_r, node_z));
    if (std::isfinite(jacobian) && jacobian > 0.0) {
      typical_sum += jacobian;
      ++typical_count;
    }
  }
  if (typical_count == 0) {
    core::log_warning(
        "[bcr-ladder] feasible seed skipped: no finite neighboring J");
    return -1;
  }

  const std::size_t node = static_cast<std::size_t>(worst.node);
  const std::size_t previous =
      static_cast<std::size_t>(worst.previous_node);
  const std::size_t next = static_cast<std::size_t>(worst.next_node);
  const bool axis = controller.initial_node_r[node] == 0.0;
  const double target_j =
      0.1 * typical_sum / static_cast<double>(typical_count);
  double gradient_r = node_z[next] - node_z[previous];
  const double gradient_z = node_r[previous] - node_r[next];
  if (axis) {
    gradient_r = 0.0;
  }
  const double gradient_norm2 =
      gradient_r * gradient_r + gradient_z * gradient_z;
  if (!(target_j > worst_jacobian) || !(gradient_norm2 > 0.0) ||
      !std::isfinite(gradient_norm2)) {
    return -1;
  }

  const double uncapped_scale =
      (target_j - worst_jacobian) / gradient_norm2;
  double delta_r = uncapped_scale * gradient_r;
  double delta_z = uncapped_scale * gradient_z;
  const double displacement = std::hypot(delta_r, delta_z);
  const double displacement_cap = 0.5 * node_scale[node];
  if (displacement > displacement_cap) {
    const double scale = displacement_cap / displacement;
    delta_r *= scale;
    delta_z *= scale;
  }
  node_r[node] += delta_r;
  node_z[node] += delta_z;
  if (axis) {
    node_r[node] = 0.0;
  }

  const double seeded_j = bcr_corner_jacobian(worst, node_r, node_z);
  std::ostringstream line;
  line << std::scientific << std::setprecision(6)
       << "[bcr-ladder] feasible-seed cell="
       << controller.bcr_feasible_seed_cell
       << " corner=" << worst.corner
       << " node=" << worst.node
       << " J_before=" << worst_jacobian
       << " J_target=" << target_j
       << " J_after=" << seeded_j
       << " displacement=" << std::hypot(delta_r, delta_z)
       << " cap=" << displacement_cap;
  core::log_info(line.str());
  return worst.node;
}

void bcr_relax_shape(const CoreClearanceState& controller,
                     const core::Config& cfg,
                     const std::vector<BcrShapeCorner>& corners,
                     const BcrCoMotionTarget& target,
                     const std::vector<double>& source_r,
                     const std::vector<double>& source_z,
                     std::vector<double>& candidate_r,
                     std::vector<double>& candidate_z,
                     double& phi0) {
  const int n_nodes = static_cast<int>(source_r.size());
  TENRYU_ASSERT(source_z.size() == source_r.size() &&
                    controller.bcr_omega.size() == source_r.size() &&
                    (!target.enabled ||
                     target.bstar_node_mask.size() == source_r.size()),
                "BCR rezone node field size mismatch");
  const double det_floor = std::max(
      cfg.numerics.ale.reference_corner_j_floor_rel,
      std::numeric_limits<double>::epsilon());
  const std::vector<double> node_scale =
      bcr_initial_node_scales(controller, n_nodes);
  const std::vector<std::vector<int>> touching_corners =
      build_bcr_touching_corners(corners, n_nodes);

  candidate_r = source_r;
  candidate_z = source_z;
  phi0 = bcr_rezone_objective(
      controller, corners, nullptr, -1, node_scale, target,
      candidate_r, candidate_z, det_floor);
  const int feasible_seed_node = bcr_apply_direct_feasible_seed(
      controller, corners, node_scale, touching_corners,
      candidate_r, candidate_z);
  const int iteration_count = controller.bcr_continuous
                                  ? kBcrContinuousSweeps
                                  : bcr_iteration_count();
  for (int iteration = 0; iteration < iteration_count; ++iteration) {
    for (int node = 0; node < n_nodes; ++node) {
      const std::size_t node_index = static_cast<std::size_t>(node);
      const double h_v = node_scale[node_index];
      if (!bcr_node_is_free(
              controller, node_scale, touching_corners, node)) {
        continue;
      }
      const bool axis = controller.initial_node_r[node_index] == 0.0;
      const double difference_h = 1.0e-9 * h_v;
      const double original_r = candidate_r[node_index];
      const double original_z = candidate_z[node_index];
      double gradient_r = 0.0;
      double gradient_z = 0.0;
      if (!axis) {
        candidate_r[node_index] = original_r + difference_h;
        const double phi_plus = bcr_rezone_objective(
            controller, corners, &touching_corners[node_index], node,
            node_scale, target, candidate_r, candidate_z, det_floor);
        candidate_r[node_index] = original_r - difference_h;
        const double phi_minus = bcr_rezone_objective(
            controller, corners, &touching_corners[node_index], node,
            node_scale, target, candidate_r, candidate_z, det_floor);
        candidate_r[node_index] = original_r;
        gradient_r = (phi_plus - phi_minus) / (2.0 * difference_h);
      }
      candidate_z[node_index] = original_z + difference_h;
      const double phi_plus = bcr_rezone_objective(
          controller, corners, &touching_corners[node_index], node,
          node_scale, target, candidate_r, candidate_z, det_floor);
      candidate_z[node_index] = original_z - difference_h;
      const double phi_minus = bcr_rezone_objective(
          controller, corners, &touching_corners[node_index], node,
          node_scale, target, candidate_r, candidate_z, det_floor);
      candidate_z[node_index] = original_z;
      gradient_z = (phi_plus - phi_minus) / (2.0 * difference_h);
      TENRYU_ASSERT(std::isfinite(gradient_r) && std::isfinite(gradient_z),
                    "BCR rezone numerical gradient is nonfinite");
      const double gradient_norm = std::hypot(gradient_r, gradient_z);
      TENRYU_ASSERT(std::isfinite(gradient_norm),
                    "BCR rezone numerical gradient norm is nonfinite");
      const double direction_r =
          gradient_norm > 0.0 ? gradient_r / gradient_norm : 0.0;
      const double direction_z =
          gradient_norm > 0.0 ? gradient_z / gradient_norm : 0.0;

      const double phi_before = bcr_rezone_objective(
          controller, corners, &touching_corners[node_index], node,
          node_scale, target, candidate_r, candidate_z, det_floor);
      // eta is a physical step length, so use the normalized gradient.
      double eta = 0.25 * h_v;
      for (int backtrack = 0; backtrack <= 4; ++backtrack) {
        double trial_r = axis ? original_r : original_r - eta * direction_r;
        double trial_z = original_z - eta * direction_z;
        double total_dr = trial_r - source_r[node_index];
        double total_dz = trial_z - source_z[node_index];
        const double displacement = std::hypot(total_dr, total_dz);
        const double displacement_cap =
            node == feasible_seed_node ? 0.5 * h_v : 0.2 * h_v;
        if (displacement > displacement_cap) {
          const double scale = displacement_cap / displacement;
          total_dr *= scale;
          total_dz *= scale;
          trial_r = source_r[node_index] + total_dr;
          trial_z = source_z[node_index] + total_dz;
        }
        if (axis) {
          trial_r = 0.0;
        }
        candidate_r[node_index] = trial_r;
        candidate_z[node_index] = trial_z;
        const double phi_trial = bcr_rezone_objective(
            controller, corners, &touching_corners[node_index], node,
            node_scale, target, candidate_r, candidate_z, det_floor);
        if (std::isfinite(phi_trial) && phi_trial <= phi_before) {
          break;
        }
        candidate_r[node_index] = original_r;
        candidate_z[node_index] = original_z;
        eta *= 0.5;
      }
    }
  }
}

struct BcrMinimumJacobian {
  double value = std::numeric_limits<double>::infinity();
  int cell = -1;
  int corner = -1;
  int node = -1;
  int previous_node = -1;
  int next_node = -1;
};

BcrMinimumJacobian bcr_patch_minimum_jacobian(
    const std::vector<BcrShapeCorner>& corners,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  BcrMinimumJacobian minimum;
  for (const BcrShapeCorner& corner : corners) {
    const std::size_t node = static_cast<std::size_t>(corner.node);
    const std::size_t previous =
        static_cast<std::size_t>(corner.previous_node);
    const std::size_t next = static_cast<std::size_t>(corner.next_node);
    const double next_r = node_r[next] - node_r[node];
    const double next_z = node_z[next] - node_z[node];
    const double previous_r = node_r[previous] - node_r[node];
    const double previous_z = node_z[previous] - node_z[node];
    const double jacobian =
        next_r * previous_z - next_z * previous_r;
    if (jacobian < minimum.value) {
      minimum.value = jacobian;
      minimum.cell = corner.cell;
      minimum.corner = corner.corner;
      minimum.node = corner.node;
      minimum.previous_node = corner.previous_node;
      minimum.next_node = corner.next_node;
    }
  }
  TENRYU_ASSERT(minimum.cell >= 0 && minimum.corner >= 0,
                "BCR rezone minimum-J corner set is empty");
  return minimum;
}

std::vector<double> bcr_target_volumes(
    const CoreClearanceState& controller,
    const core::State& state,
    const std::vector<double>& target_r,
    const std::vector<double>& target_z) {
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "BCR rezone target volumes require multiblock topology");
  const auto& topology = *state.mesh.topo.multiblock;
  const int n_cells = state.mesh.topo.n_cells;
  TENRYU_ASSERT(topology.cell_orientation_sign.size() ==
                    static_cast<std::size_t>(n_cells),
                "BCR rezone target volumes require orientation signs");
  std::vector<double> target_volume(static_cast<std::size_t>(n_cells), 0.0);
  for (int cell = 0; cell < n_cells; ++cell) {
    const int offset = controller.cell_node_offsets[
        static_cast<std::size_t>(cell)];
    const int nverts = active_nverts(controller, cell);
    double r[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    double z[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    for (int corner = 0; corner < nverts; ++corner) {
      const int node = controller.cell_node_indices[
          static_cast<std::size_t>(offset + corner)];
      r[corner] = target_r[static_cast<std::size_t>(node)];
      z[corner] = target_z[static_cast<std::size_t>(node)];
    }
    const auto moments = mesh::moments::poly_rz_moments_fan(r, z, nverts);
    const double oriented_mr =
        static_cast<double>(topology.cell_orientation_sign[
            static_cast<std::size_t>(cell)]) *
        moments.mr;
    const double volume = 2.0 * mesh::moments::kPi * oriented_mr;
    TENRYU_ASSERT(std::isfinite(volume) && volume > 0.0,
                  "BCR rezone target cell volume is invalid");
    target_volume[static_cast<std::size_t>(cell)] = volume;
  }
  return target_volume;
}

void emit_bcr_rezone_step(const int step,
                          const double phi0,
                          const double phi1,
                          const BcrMinimumJacobian& jmin_before,
                          const BcrMinimumJacobian& jmin_after,
                          const double max_displacement,
                          const std::array<double, 3>& argmin_node_displacements,
                          const std::array<int, 3>& argmin_free,
                          const int halvings,
                          const bool remap_applied) {
  const double argmin_nodes_displacement = *std::max_element(
      argmin_node_displacements.begin(), argmin_node_displacements.end());
  std::ostringstream line;
  line << std::scientific << std::setprecision(6)
       << "[bcr-rezone] step=" << step
       << " invoked Phi0=" << phi0
       << " Phi1=" << phi1
       << " Jmin_before=" << jmin_before.value
       << " Jmin_after=" << jmin_after.value
       << " Jmin_after_cell=" << jmin_after.cell
       << " Jmin_after_k=" << jmin_after.corner
       << " max_disp=" << std::setprecision(3) << max_displacement
       << " argmin_cell=" << jmin_before.cell
       << " argmin_k=" << jmin_before.corner
       << " argmin_nodes=[" << jmin_before.node << ','
       << jmin_before.previous_node << ',' << jmin_before.next_node << ']'
       << " argmin_node_disps=[" << argmin_node_displacements[0] << ','
       << argmin_node_displacements[1] << ','
       << argmin_node_displacements[2] << ']'
       << " argmin_free=[" << argmin_free[0] << ','
       << argmin_free[1] << ',' << argmin_free[2] << ']'
       << " argmin_nodes_disp=" << argmin_nodes_displacement
       << " halvings=" << halvings
       << " remap=" << (remap_applied ? 1 : 0);
  core::log_info(line.str());
}

void emit_bcr_predictor_final(const CoreClearanceState& controller) {
  if (!controller.bcr_predictor || controller.rank != 0) {
    return;
  }
  std::ostringstream line;
  line << "[bcr-pred] FINAL first_trigger_step="
       << controller.bcr_predictor_first_trigger_step
       << " total_triggered_steps="
       << controller.bcr_predictor_triggered_steps
       << " final_state=" << bcr_phase_name(controller.bcr_phase)
       << " transitions=" << controller.bcr_transitions.size()
       << " history=ARMED";
  for (const BcrTransition& transition : controller.bcr_transitions) {
    line << "->" << bcr_phase_name(transition.to)
         << '@' << transition.step;
  }
  core::log_info(line.str());
}

void emit_bcr_rezone_final(const CoreClearanceState& controller) {
  if (!controller.bcr_rezone || controller.rank != 0) {
    return;
  }
  core::log_info(
      "[bcr-rezone] FINAL invocations=" +
      std::to_string(controller.bcr_rezone_invocations));
}

double adot_corner_rate(const PentagonPoint& center,
                        const PentagonPoint& center_velocity,
                        const PentagonPoint& corner,
                        const PentagonPoint& next_corner,
                        const PentagonPoint& corner_velocity,
                        const PentagonPoint& next_corner_velocity) {
  const PentagonPoint edge{
      corner.r - center.r,
      corner.z - center.z,
  };
  const PentagonPoint next_edge{
      next_corner.r - center.r,
      next_corner.z - center.z,
  };
  const PentagonPoint relative_velocity{
      corner_velocity.r - center_velocity.r,
      corner_velocity.z - center_velocity.z,
  };
  const PentagonPoint next_relative_velocity{
      next_corner_velocity.r - center_velocity.r,
      next_corner_velocity.z - center_velocity.z,
  };
  return 0.5 * (planar_cross(relative_velocity, next_edge) +
                planar_cross(edge, next_relative_velocity));
}

void split_spherical_velocity(const PentagonPoint& position,
                              const PentagonPoint& velocity,
                              PentagonPoint& radial,
                              PentagonPoint& tangential) {
  const double radius = std::hypot(position.r, position.z);
  TENRYU_ASSERT(std::isfinite(radius) && radius > 0.0,
                "adot ledger requires nonzero finite node radius");
  const double radial_direction_r = position.r / radius;
  const double radial_direction_z = position.z / radius;
  const double tangential_direction_r = -radial_direction_z;
  const double tangential_direction_z = radial_direction_r;
  const double radial_speed =
      velocity.r * radial_direction_r + velocity.z * radial_direction_z;
  const double tangential_speed =
      velocity.r * tangential_direction_r +
      velocity.z * tangential_direction_z;
  radial = {
      radial_speed * radial_direction_r,
      radial_speed * radial_direction_z,
  };
  tangential = {
      tangential_speed * tangential_direction_r,
      tangential_speed * tangential_direction_z,
  };
}

void record_adot_ledger(CoreClearanceState& controller,
                        const core::State& state,
                        const int step,
                        const double t) {
  if (controller.adot_cell < 0 || step < controller.adot_from_step) {
    return;
  }
  TENRYU_ASSERT(state.dt > 0.0 && std::isfinite(state.dt),
                "adot ledger requires a positive finite timestep");
  const std::size_t n_nodes =
      static_cast<std::size_t>(state.mesh.topo.n_nodes);
  TENRYU_ASSERT(state.x_r.size() == n_nodes && state.x_z.size() == n_nodes &&
                    state.v_r.size() == n_nodes && state.v_z.size() == n_nodes,
                "adot ledger node field size changed");

  std::vector<double> node_r;
  std::vector<double> node_z;
  std::vector<double> velocity_r;
  std::vector<double> velocity_z;
  state.x_r.copy_to_host(node_r);
  state.x_z.copy_to_host(node_z);
  state.v_r.copy_to_host(velocity_r);
  state.v_z.copy_to_host(velocity_z);

  const int offset = controller.cell_node_offsets[
      static_cast<std::size_t>(controller.adot_cell)];
  PentagonPoint position[5];
  PentagonPoint velocity[5];
  PentagonPoint radial_velocity[5];
  PentagonPoint tangential_velocity[5];
  for (int corner = 0; corner < 5; ++corner) {
    const int node = controller.cell_node_indices[
        static_cast<std::size_t>(offset + corner)];
    const std::size_t node_index = static_cast<std::size_t>(node);
    position[corner] = {node_r[node_index], node_z[node_index]};
    velocity[corner] = {velocity_r[node_index], velocity_z[node_index]};
    split_spherical_velocity(position[corner], velocity[corner],
                             radial_velocity[corner],
                             tangential_velocity[corner]);
  }

  const PentagonPoint center = pentagon_center(position);
  const PentagonPoint center_velocity = pentagon_center(velocity);
  const PentagonPoint center_radial_velocity =
      pentagon_center(radial_velocity);
  const PentagonPoint center_tangential_velocity =
      pentagon_center(tangential_velocity);
  std::array<double, 5> area{};
  std::array<double, 5> adot{};
  std::array<double, 5> adot_radial{};
  std::array<double, 5> adot_tangential{};
  std::array<double, 5> jacobian{};
  std::array<double, 5> jacobian_rate{};
  int minimum_corner = 0;
  int minimum_jacobian_corner = 0;
  for (int corner = 0; corner < 5; ++corner) {
    const int next = (corner == 4) ? 0 : corner + 1;
    const int previous = (corner == 0) ? 4 : corner - 1;
    const PentagonPoint edge{
        position[corner].r - center.r,
        position[corner].z - center.z,
    };
    const PentagonPoint next_edge{
        position[next].r - center.r,
        position[next].z - center.z,
    };
    area[static_cast<std::size_t>(corner)] =
        0.5 * planar_cross(edge, next_edge);
    adot[static_cast<std::size_t>(corner)] = adot_corner_rate(
        center, center_velocity, position[corner], position[next],
        velocity[corner], velocity[next]);
    adot_radial[static_cast<std::size_t>(corner)] = adot_corner_rate(
        center, center_radial_velocity, position[corner], position[next],
        radial_velocity[corner], radial_velocity[next]);
    adot_tangential[static_cast<std::size_t>(corner)] = adot_corner_rate(
        center, center_tangential_velocity, position[corner], position[next],
        tangential_velocity[corner], tangential_velocity[next]);
    const PentagonPoint next_vertex_edge{
        position[next].r - position[corner].r,
        position[next].z - position[corner].z,
    };
    const PentagonPoint previous_vertex_edge{
        position[previous].r - position[corner].r,
        position[previous].z - position[corner].z,
    };
    const PentagonPoint next_vertex_edge_rate{
        velocity[next].r - velocity[corner].r,
        velocity[next].z - velocity[corner].z,
    };
    const PentagonPoint previous_vertex_edge_rate{
        velocity[previous].r - velocity[corner].r,
        velocity[previous].z - velocity[corner].z,
    };
    jacobian[static_cast<std::size_t>(corner)] =
        planar_cross(next_vertex_edge, previous_vertex_edge);
    jacobian_rate[static_cast<std::size_t>(corner)] =
        planar_cross(next_vertex_edge_rate, previous_vertex_edge) +
        planar_cross(next_vertex_edge, previous_vertex_edge_rate);

    const double split_sum =
        adot_radial[static_cast<std::size_t>(corner)] +
        adot_tangential[static_cast<std::size_t>(corner)];
    const double additivity_scale = std::max(
        {std::abs(adot[static_cast<std::size_t>(corner)]),
         std::abs(adot_radial[static_cast<std::size_t>(corner)]) +
             std::abs(adot_tangential[static_cast<std::size_t>(corner)]),
         std::numeric_limits<double>::min()});
    const double additivity_error =
        std::abs(adot[static_cast<std::size_t>(corner)] - split_sum);
    const double additivity_tolerance =
        512.0 * std::numeric_limits<double>::epsilon() * additivity_scale;
    if ((!std::isfinite(additivity_error) ||
         additivity_error > additivity_tolerance) &&
        controller.rank == 0) {
      std::ostringstream warning;
      warning << std::scientific << std::setprecision(6)
              << "[adot] SOFT_ASSERT additivity step=" << step
              << " k=" << corner
              << " error=" << additivity_error
              << " tolerance=" << additivity_tolerance;
      core::log_warning(warning.str());
    }
    if (area[static_cast<std::size_t>(corner)] <
        area[static_cast<std::size_t>(minimum_corner)]) {
      minimum_corner = corner;
    }
    if (jacobian[static_cast<std::size_t>(corner)] <
        jacobian[static_cast<std::size_t>(minimum_jacobian_corner)]) {
      minimum_jacobian_corner = corner;
    }
  }

  const double minimum_area = area[static_cast<std::size_t>(minimum_corner)];
  const double minimum_jacobian =
      jacobian[static_cast<std::size_t>(minimum_jacobian_corner)];
  const double minimum_adot = adot[static_cast<std::size_t>(minimum_corner)];
  controller.adot_dneg +=
      std::max(0.0, -minimum_adot / minimum_area) * state.dt;
  if (!controller.adot_observed || minimum_area < controller.adot_min_area) {
    controller.adot_observed = true;
    controller.adot_min_area = minimum_area;
    controller.adot_min_step = step;
  }
  if (minimum_jacobian < controller.adot_min_jacobian) {
    controller.adot_min_jacobian = minimum_jacobian;
    controller.adot_min_jacobian_corner = minimum_jacobian_corner;
  }

  if (controller.rank != 0) {
    return;
  }
  for (int corner = 0; corner < 5; ++corner) {
    const std::size_t index = static_cast<std::size_t>(corner);
    std::ostringstream line;
    line << std::scientific
         << "[adot] step=" << step
         << " t=" << std::setprecision(9) << t
         << " k=" << corner
         << " A=" << std::setprecision(9) << area[index]
         << " Adot=" << std::setprecision(6) << adot[index]
         << " Adot_rad=" << adot_radial[index]
         << " Adot_tan=" << adot_tangential[index]
         << " J=" << std::setprecision(9) << jacobian[index]
         << " Jdot=" << std::setprecision(6) << jacobian_rate[index];
    core::log_info(line.str());
  }
  std::ostringstream summary;
  summary << std::scientific
          << "[adot] step=" << step
          << " Amin=" << std::setprecision(9) << minimum_area
          << " k_argmin=" << minimum_corner
          << " Jmin=" << minimum_jacobian
          << " k_argminJ=" << minimum_jacobian_corner
          << " Dneg=" << std::setprecision(6) << controller.adot_dneg;
  core::log_info(summary.str());
}

void emit_adot_final(const CoreClearanceState& controller) {
  if (controller.adot_cell < 0 || controller.rank != 0) {
    return;
  }
  const double minimum_area =
      controller.adot_observed ? controller.adot_min_area : quiet_nan();
  const double minimum_jacobian =
      controller.adot_observed ? controller.adot_min_jacobian : quiet_nan();
  std::ostringstream line;
  line << std::scientific
       << "[adot] FINAL Dneg=" << std::setprecision(6)
       << controller.adot_dneg
       << " Amin_ever=" << std::setprecision(9) << minimum_area
       << " step_min=" << controller.adot_min_step
       << " Jmin=" << minimum_jacobian
       << " k_argminJ=" << controller.adot_min_jacobian_corner;
  core::log_info(line.str());
}

double weighted_median(const std::vector<int>& cells,
                       const std::vector<double>& values,
                       const std::vector<double>& weights) {
  std::vector<RankedValue> ranked;
  ranked.reserve(cells.size());
  double total_weight = 0.0;
  for (const int cell : cells) {
    const std::size_t index = static_cast<std::size_t>(cell);
    const double value = values[index];
    const double weight = weights[index];
    if (!std::isfinite(value) || !std::isfinite(weight) || !(weight > 0.0)) {
      continue;
    }
    ranked.push_back({value, weight, cell});
    total_weight += weight;
  }
  if (ranked.empty() || !(total_weight > 0.0)) {
    return quiet_nan();
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const RankedValue& lhs, const RankedValue& rhs) {
              if (lhs.value != rhs.value) {
                return lhs.value < rhs.value;
              }
              return lhs.cell < rhs.cell;
            });
  const double half_weight = 0.5 * total_weight;
  double cumulative_weight = 0.0;
  for (const RankedValue& entry : ranked) {
    cumulative_weight += entry.weight;
    if (cumulative_weight >= half_weight) {
      return entry.value;
    }
  }
  return ranked.back().value;
}

double deterministic_percentile(std::vector<RankedValue> ranked,
                                const double fraction) {
  if (ranked.empty()) {
    return quiet_nan();
  }
  const std::size_t index = std::min(
      static_cast<std::size_t>(fraction * static_cast<double>(ranked.size())),
      ranked.size() - 1U);
  std::nth_element(
      ranked.begin(), ranked.begin() + index, ranked.end(),
      [](const RankedValue& lhs, const RankedValue& rhs) {
        if (lhs.value != rhs.value) {
          return lhs.value < rhs.value;
        }
        return lhs.cell < rhs.cell;
      });
  return ranked[index].value;
}

double radial_cell_thickness(const CoreClearanceState& controller,
                             const int cell,
                             const std::vector<double>& node_r,
                             const std::vector<double>& node_z) {
  const int offset =
      controller.cell_node_offsets[static_cast<std::size_t>(cell)];
  const int nverts = active_nverts(controller, cell);
  double minimum_radius = std::numeric_limits<double>::infinity();
  double maximum_radius = -std::numeric_limits<double>::infinity();
  for (int corner = 0; corner < nverts; ++corner) {
    const int node = controller.cell_node_indices[
        static_cast<std::size_t>(offset + corner)];
    const double radius =
        std::hypot(node_r[static_cast<std::size_t>(node)],
                   node_z[static_cast<std::size_t>(node)]);
    minimum_radius = std::min(minimum_radius, radius);
    maximum_radius = std::max(maximum_radius, radius);
  }
  return maximum_radius - minimum_radius;
}

double bcr_patch_median_radial_thickness(
    const CoreClearanceState& controller,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  std::vector<RankedValue> ranked;
  ranked.reserve(controller.bcr_patch_cells.size());
  for (const int cell : controller.bcr_patch_cells) {
    const double thickness =
        radial_cell_thickness(controller, cell, node_r, node_z);
    if (std::isfinite(thickness) && thickness > 0.0) {
      ranked.push_back({thickness, 0.0, cell});
    }
  }
  return deterministic_percentile(std::move(ranked), 0.5);
}

const char* phase_name(const ClearancePhase phase) {
  switch (phase) {
    case ClearancePhase::STATIC:
      return "STATIC";
    case ClearancePhase::SPLICING:
      return "SPLICING";
    case ClearancePhase::TRACKING:
      return "TRACKING";
  }
  return "UNKNOWN";
}

const char* bcr_phase_name(const BcrPhase phase) {
  switch (phase) {
    case BcrPhase::ARMED:
      return "ARMED";
    case BcrPhase::CAPTURE:
      return "CAPTURE";
    case BcrPhase::RIDE:
      return "RIDE";
    case BcrPhase::RECOVERY_HOLD:
      return "RECOVERY_HOLD";
    case BcrPhase::RELEASE_RAMP:
      return "RELEASE_RAMP";
  }
  return "UNKNOWN";
}

double bcr_quintic_smoothstep(const double x) {
  const double clamped = std::clamp(x, 0.0, 1.0);
  return clamped * clamped * clamped *
         (10.0 + clamped * (-15.0 + 6.0 * clamped));
}

void transition_bcr_phase(CoreClearanceState& controller,
                          const BcrPhase next,
                          const int step,
                          const double t) {
  if (controller.bcr_phase == next) {
    return;
  }
  const BcrPhase previous = controller.bcr_phase;
  controller.bcr_phase = next;
  controller.bcr_transitions.push_back({previous, next, step, t});
  if (controller.rank == 0) {
    std::ostringstream line;
    line << std::scientific << std::setprecision(9)
         << "[bcr-state] step=" << step
         << " t=" << t
         << " transition=" << bcr_phase_name(previous)
         << "->" << bcr_phase_name(next);
    core::log_info(line.str());
  }
}

double bcr_hold_weight(const CoreClearanceState& controller,
                       const double t) {
  if (controller.bcr_phase != BcrPhase::RELEASE_RAMP) {
    return 1.0;
  }
  TENRYU_ASSERT(std::isfinite(controller.bcr_release_ramp_start_t) &&
                    std::isfinite(controller.bcr_release_ramp_tau) &&
                    controller.bcr_release_ramp_tau > 0.0,
                "BCR release ramp clock is invalid");
  const double xi =
      (t - controller.bcr_release_ramp_start_t) /
      controller.bcr_release_ramp_tau;
  return 1.0 - bcr_quintic_smoothstep(xi);
}

BcrFrontSample capture_bcr_front_sample(
    const CoreClearanceState& controller,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const std::vector<double>& velocity_r,
    const std::vector<double>& velocity_z,
    const std::vector<double>& current_volume) {
  const std::size_t n_cells = controller.initial_volume.size();
  std::vector<double> current_radius(n_cells, quiet_nan());
  std::vector<double> radial_velocity(n_cells, quiet_nan());
  std::vector<double> inward_speed(n_cells, quiet_nan());
  std::vector<double> volume_ratio(n_cells, quiet_nan());
  std::vector<RankedValue> inward_speed_ranked;
  inward_speed_ranked.reserve(controller.wideband_cells.size());
  for (const int cell : controller.wideband_cells) {
    const std::size_t index = static_cast<std::size_t>(cell);
    const VectorRz centroid =
        cell_vertex_mean(controller, cell, node_r, node_z);
    const VectorRz centroid_velocity =
        cell_vertex_mean(controller, cell, velocity_r, velocity_z);
    const double radius = std::hypot(centroid.r, centroid.z);
    current_radius[index] = radius;
    if (std::isfinite(radius) && radius > 0.0) {
      radial_velocity[index] =
          (centroid_velocity.r * centroid.r +
           centroid_velocity.z * centroid.z) /
          radius;
      inward_speed[index] = -radial_velocity[index];
    }
    volume_ratio[index] =
        current_volume[index] / controller.initial_volume[index];
    if (std::isfinite(inward_speed[index])) {
      inward_speed_ranked.push_back({inward_speed[index], 0.0, cell});
    }
  }

  const double u95 = deterministic_percentile(
      std::move(inward_speed_ranked), 0.95);
  const double eligibility_speed = std::max(
      controller.u_floor,
      std::isfinite(u95) ? 0.25 * u95 : controller.u_floor);
  std::array<double, kClearanceBinCount> bin_radius{};
  std::array<bool, kClearanceBinCount> eligible{};
  for (int bin = 0; bin < kClearanceBinCount; ++bin) {
    const std::vector<int>& members =
        controller.bins[static_cast<std::size_t>(bin)];
    bin_radius[static_cast<std::size_t>(bin)] = weighted_median(
        members, current_radius, controller.initial_volume);
    const double bin_speed = -weighted_median(
        members, radial_velocity, controller.initial_volume);
    const double bin_compression = weighted_median(
        members, volume_ratio, controller.initial_volume);
    eligible[static_cast<std::size_t>(bin)] =
        std::isfinite(bin_speed) && std::isfinite(bin_compression) &&
        bin_speed >= eligibility_speed && bin_compression <= 0.95;
  }

  int front_bin = -1;
  for (int bin = 0; bin + 1 < kClearanceBinCount; ++bin) {
    if (eligible[static_cast<std::size_t>(bin)] &&
        eligible[static_cast<std::size_t>(bin + 1)]) {
      front_bin = bin;
      break;
    }
  }
  BcrFrontSample sample;
  if (front_bin < 0) {
    return sample;
  }
  sample.valid = true;
  sample.radius = bin_radius[static_cast<std::size_t>(front_bin)];
  std::vector<int> speed_members =
      controller.bins[static_cast<std::size_t>(front_bin)];
  const std::vector<int>& outward_members =
      controller.bins[static_cast<std::size_t>(front_bin + 1)];
  speed_members.insert(speed_members.end(), outward_members.begin(),
                       outward_members.end());
  sample.inward_speed = weighted_median(
      speed_members, inward_speed, controller.initial_volume);
  return sample;
}

BcrQualityMetrics compute_bcr_quality_metrics(
    const CoreClearanceState& controller,
    const core::State& state,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const std::vector<double>& velocity_r,
    const std::vector<double>& velocity_z,
    const std::vector<double>& current_volume) {
  const int n_nodes = state.mesh.topo.n_nodes;
  TENRYU_ASSERT(node_r.size() == static_cast<std::size_t>(n_nodes) &&
                    node_z.size() == static_cast<std::size_t>(n_nodes) &&
                    velocity_r.size() == static_cast<std::size_t>(n_nodes) &&
                    velocity_z.size() == static_cast<std::size_t>(n_nodes) &&
                    current_volume.size() == controller.initial_volume.size(),
                "BCR quality field sizes changed");
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "BCR quality requires multiblock topology");
  const auto& topology = *state.mesh.topo.multiblock;
  TENRYU_ASSERT(topology.cell_orientation_sign.size() ==
                    controller.initial_volume.size(),
                "BCR quality orientation metadata changed");

  BcrCoMotionTarget target;
  if (controller.bcr_target) {
    target = build_bcr_co_motion_target(
        controller, node_r, node_z, velocity_r, velocity_z, 0.0, 1.0);
  }
  BcrQualityMetrics metrics;
  metrics.target_scale = target.scale;
  const std::vector<BcrShapeCorner> corners = build_bcr_shape_corners(
      controller, n_nodes, metrics.target_scale, node_r, node_z, 1.0);
  for (const BcrShapeCorner& corner : corners) {
    const std::size_t node = static_cast<std::size_t>(corner.node);
    const std::size_t previous =
        static_cast<std::size_t>(corner.previous_node);
    const std::size_t next = static_cast<std::size_t>(corner.next_node);
    const double next_r = node_r[next] - node_r[node];
    const double next_z = node_z[next] - node_z[node];
    const double previous_r = node_r[previous] - node_r[node];
    const double previous_z = node_z[previous] - node_z[node];
    const double jacobian =
        next_r * previous_z - next_z * previous_r;
    const double det_winv =
        corner.winv00 * corner.winv11 -
        corner.winv01 * corner.winv10;
    metrics.q_j_min = std::min(metrics.q_j_min, jacobian * det_winv);
  }

  const double target_volume_scale =
      metrics.target_scale * metrics.target_scale * metrics.target_scale;
  double patch_inner_radius = std::numeric_limits<double>::infinity();
  for (const int cell : controller.bcr_patch_cells) {
    const std::size_t index = static_cast<std::size_t>(cell);
    const double target_volume =
        target_volume_scale * controller.initial_volume[index];
    TENRYU_ASSERT(std::isfinite(target_volume) && target_volume > 0.0,
                  "BCR target-normalized volume is invalid");
    metrics.q_v_min =
        std::min(metrics.q_v_min, current_volume[index] / target_volume);
    const VectorRz centroid =
        cell_vertex_mean(controller, cell, node_r, node_z);
    const double radius = std::hypot(centroid.r, centroid.z);
    patch_inner_radius = std::min(patch_inner_radius, radius);
  }
  const double h_r = controller.bcr_h_r;
  BcrFrontSample shell;
  if (controller.active) {
    const ReplayPoint replay = interpolate_replay(controller, state.t);
    shell.valid = true;
    shell.radius = replay.s_f;
    shell.inward_speed = replay.u_f;
  } else {
    shell = capture_bcr_front_sample(
        controller, node_r, node_z, velocity_r, velocity_z, current_volume);
  }
  metrics.shell_radius = shell.radius;
  metrics.patch_inner_radius = patch_inner_radius;
  TENRYU_ASSERT(std::isfinite(h_r) && h_r > 0.0 &&
                    std::isfinite(patch_inner_radius),
                "BCR physical crossing scale is invalid");
  metrics.h_r = h_r;
  const std::vector<std::uint8_t> patch_node_mask =
      bcr_node_mask_from_cells(
          controller, controller.bcr_patch_cells, n_nodes);
  double patch_inward_speed_sum = 0.0;
  int patch_speed_count = 0;
  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t index = static_cast<std::size_t>(node);
    if (patch_node_mask[index] == 0U) {
      continue;
    }
    const double radius = std::hypot(node_r[index], node_z[index]);
    if (!(std::isfinite(radius) && radius > 0.0)) {
      continue;
    }
    const double inward_speed =
        -(velocity_r[index] * node_r[index] +
          velocity_z[index] * node_z[index]) /
        radius;
    if (std::isfinite(inward_speed)) {
      patch_inward_speed_sum += inward_speed;
      ++patch_speed_count;
    }
  }
  const double patch_inward_speed =
      patch_speed_count > 0
          ? patch_inward_speed_sum / static_cast<double>(patch_speed_count)
          : quiet_nan();
  const double relative_speed =
      shell.valid && std::isfinite(shell.inward_speed) &&
              std::isfinite(patch_inward_speed)
          ? std::abs(shell.inward_speed - patch_inward_speed)
          : 0.0;
  metrics.tau_cell =
      h_r / std::max(relative_speed, kBcrRelativeSpeedFloor);
  metrics.trailing_clearance =
      shell.valid &&
      shell.radius < patch_inner_radius - h_r;

  TENRYU_ASSERT(state.dt > 0.0 && std::isfinite(state.dt),
                "BCR predictor requires a positive finite timestep");
  const double horizon_steps_raw = std::ceil(metrics.tau_cell / state.dt);
  TENRYU_ASSERT(std::isfinite(horizon_steps_raw) &&
                    horizon_steps_raw >= 1.0 &&
                    horizon_steps_raw <=
                        static_cast<double>(std::numeric_limits<int>::max()),
                "BCR physical-time predictor horizon is invalid");
  const int horizon_steps = static_cast<int>(horizon_steps_raw);
  std::vector<double> interval_start_r(node_r.size(), 0.0);
  std::vector<double> interval_start_z(node_z.size(), 0.0);
  std::vector<double> interval_end_r(node_r.size(), 0.0);
  std::vector<double> interval_end_z(node_z.size(), 0.0);
  const bool g31_enabled = g31_variant() != G31Variant::LEGACY;
  for (int interval = 0; interval < horizon_steps; ++interval) {
    const double start_t = static_cast<double>(interval) * state.dt;
    const double end_t = static_cast<double>(interval + 1) * state.dt;
    for (int node = 0; node < n_nodes; ++node) {
      const std::size_t index = static_cast<std::size_t>(node);
      interval_start_r[index] = node_r[index] + start_t * velocity_r[index];
      interval_start_z[index] = node_z[index] + start_t * velocity_z[index];
      interval_end_r[index] = node_r[index] + end_t * velocity_r[index];
      interval_end_z[index] = node_z[index] + end_t * velocity_z[index];
    }
    for (const BcrShapeCorner& corner : corners) {
      const std::size_t node = static_cast<std::size_t>(corner.node);
      const std::size_t previous =
          static_cast<std::size_t>(corner.previous_node);
      const std::size_t next = static_cast<std::size_t>(corner.next_node);
      const double next_r =
          interval_start_r[next] - interval_start_r[node];
      const double next_z =
          interval_start_z[next] - interval_start_z[node];
      const double previous_r =
          interval_start_r[previous] - interval_start_r[node];
      const double previous_z =
          interval_start_z[previous] - interval_start_z[node];
      const double next_delta_r =
          interval_end_r[next] - interval_end_r[node] - next_r;
      const double next_delta_z =
          interval_end_z[next] - interval_end_z[node] - next_z;
      const double previous_delta_r =
          interval_end_r[previous] - interval_end_r[node] - previous_r;
      const double previous_delta_z =
          interval_end_z[previous] - interval_end_z[node] - previous_z;
      const double det_winv =
          corner.winv00 * corner.winv11 -
          corner.winv01 * corner.winv10;
      const double q0 =
          (next_r * previous_z - next_z * previous_r) * det_winv;
      const double q1 =
          (next_delta_r * previous_z - next_delta_z * previous_r +
           next_r * previous_delta_z - next_z * previous_delta_r) *
          det_winv;
      const double q2 =
          (next_delta_r * previous_delta_z -
           next_delta_z * previous_delta_r) *
          det_winv;
      const double root =
          mesh::path_admissibility_detail::quadratic_floor_root_unit(
              q0, q1, q2, kBcrQualityOn);
      if (root < 1.0) {
        metrics.tau_geom = std::min(
            metrics.tau_geom,
            (static_cast<double>(interval) + root) * state.dt);
      }
      if (g31_enabled) {
        const double hard_root =
            mesh::path_admissibility_detail::quadratic_floor_root_unit(
                q0, q1, q2, kBcrQualityHard);
        if (hard_root < 1.0) {
          metrics.tau_hard = std::min(
              metrics.tau_hard,
              (static_cast<double>(interval) + hard_root) * state.dt);
        }
      }
    }
    for (const int cell : controller.bcr_patch_cells) {
      const int offset =
          controller.cell_node_offsets[static_cast<std::size_t>(cell)];
      const int nverts = active_nverts(controller, cell);
      std::vector<int> nodes;
      nodes.reserve(static_cast<std::size_t>(nverts));
      for (int corner = 0; corner < nverts; ++corner) {
        nodes.push_back(controller.cell_node_indices[
            static_cast<std::size_t>(offset + corner)]);
      }
      double coefficient[4] = {};
      TENRYU_ASSERT(
          mesh::path_admissibility_detail::
              rz_polygon_volume_path_coefficients(
                  interval_start_r, interval_start_z, interval_end_r,
                  interval_end_z, nodes, coefficient),
          "BCR physical-time volume predictor is invalid");
      const double orientation =
          topology.cell_orientation_sign[static_cast<std::size_t>(cell)] < 0
              ? -1.0
              : 1.0;
      const double target_volume =
          target_volume_scale *
          controller.initial_volume[static_cast<std::size_t>(cell)];
      const double root =
          mesh::path_admissibility_detail::first_cubic_floor_root_sampled(
              orientation * coefficient[0] / target_volume,
              orientation * coefficient[1] / target_volume,
              orientation * coefficient[2] / target_volume,
              orientation * coefficient[3] / target_volume,
              kBcrQualityOn);
      if (root < 1.0) {
        metrics.tau_geom = std::min(
            metrics.tau_geom,
            (static_cast<double>(interval) + root) * state.dt);
      }
      if (g31_enabled) {
        const double hard_root =
            mesh::path_admissibility_detail::first_cubic_floor_root_sampled(
                orientation * coefficient[0] / target_volume,
                orientation * coefficient[1] / target_volume,
                orientation * coefficient[2] / target_volume,
                orientation * coefficient[3] / target_volume,
                kBcrQualityHard);
        if (hard_root < 1.0) {
          metrics.tau_hard = std::min(
              metrics.tau_hard,
              (static_cast<double>(interval) + hard_root) * state.dt);
        }
      }
    }
    if (std::isfinite(metrics.tau_geom) &&
        (!g31_enabled || std::isfinite(metrics.tau_hard))) {
      break;
    }
  }
  return metrics;
}

double bcr_quality_min(const BcrQualityMetrics& metrics) {
  return std::min(metrics.q_j_min, metrics.q_v_min);
}

void append_bcr_quality_history(CoreClearanceState& controller,
                                const int step,
                                const double t,
                                const double q_min) {
  if (!controller.bcr_dwell_history.empty() &&
      controller.bcr_dwell_history.back().step == step) {
    controller.bcr_dwell_history.back() = {step, t, q_min};
  } else {
    controller.bcr_dwell_history.push_back({step, t, q_min});
  }
  const double cutoff = t - controller.bcr_tau_cell;
  while (controller.bcr_dwell_history.size() > 1U &&
         controller.bcr_dwell_history[1].t <= cutoff) {
    controller.bcr_dwell_history.erase(
        controller.bcr_dwell_history.begin());
  }
}

double bcr_quality_history_slope(const CoreClearanceState& controller) {
  if (controller.bcr_dwell_history.size() < 2U) {
    return -std::numeric_limits<double>::infinity();
  }
  const double origin_t = controller.bcr_dwell_history.front().t;
  double sum_t = 0.0;
  double sum_q = 0.0;
  for (const BcrQualityHistorySample& sample :
       controller.bcr_dwell_history) {
    sum_t += sample.t - origin_t;
    sum_q += sample.q_min;
  }
  const double inverse_count =
      1.0 / static_cast<double>(controller.bcr_dwell_history.size());
  const double mean_t = sum_t * inverse_count;
  const double mean_q = sum_q * inverse_count;
  double numerator = 0.0;
  double denominator = 0.0;
  for (const BcrQualityHistorySample& sample :
       controller.bcr_dwell_history) {
    const double centered_t = sample.t - origin_t - mean_t;
    numerator += centered_t * (sample.q_min - mean_q);
    denominator += centered_t * centered_t;
  }
  return denominator > 0.0
             ? numerator / denominator
             : -std::numeric_limits<double>::infinity();
}

bool bcr_release_eligible(const CoreClearanceState& controller,
                          const BcrQualityMetrics& metrics,
                          const int step,
                          const double t,
                          const double dt) {
  const double reserve_required = std::max(metrics.tau_cell, 8.0 * dt);
  if (!metrics.trailing_clearance ||
      metrics.q_j_min < kBcrQualityOff ||
      metrics.q_v_min < kBcrQualityOff ||
      metrics.tau_geom < reserve_required ||
      controller.bcr_dwell_history.size() <
          static_cast<std::size_t>(kBcrDwellAcceptedSteps)) {
    return false;
  }
  const BcrQualityHistorySample& first =
      controller.bcr_dwell_history.front();
  const BcrQualityHistorySample& last =
      controller.bcr_dwell_history.back();
  if (last.step != step || t - first.t < metrics.tau_cell ||
      step - first.step < kBcrDwellAcceptedSteps - 1) {
    return false;
  }
  for (const BcrQualityHistorySample& sample :
       controller.bcr_dwell_history) {
    if (sample.q_min < kBcrQualityOff) {
      return false;
    }
  }
  const double dwell_start = first.t;
  if (controller.bcr_last_four_halvings_t >= dwell_start ||
      controller.bcr_last_bf_first_reject_t >= dwell_start ||
      controller.bcr_last_geometry_retry_t >= dwell_start) {
    return false;
  }
  const double slope = bcr_quality_history_slope(controller);
  if (slope < -0.01 / metrics.tau_cell) {
    return false;
  }
  double previous_min = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i + 1U < controller.bcr_dwell_history.size(); ++i) {
    previous_min =
        std::min(previous_min, controller.bcr_dwell_history[i].q_min);
  }
  return last.q_min >= previous_min;
}

void emit_bcr_state_sample(
    const CoreClearanceState& controller,
    const BcrReserveDiagnostic& reserve,
    const BcrQualityMetrics& metrics,
    const int step,
    const double t) {
  const bool transitioned =
      !controller.bcr_transitions.empty() &&
      controller.bcr_transitions.back().step == step;
  if (controller.rank != 0 ||
      (step % core_clearance_cadence_steps() != 0 && !transitioned)) {
    return;
  }
  std::ostringstream line;
  line << std::scientific
       << "[bcr-state] step=" << step
       << " t=" << std::setprecision(9) << t
       << " state=" << bcr_phase_name(controller.bcr_phase)
       << " tau_geom=" << std::setprecision(6) << metrics.tau_geom
       << " tau_cell=" << metrics.tau_cell
       << " q_Jmin=" << metrics.q_j_min
       << " q_Vmin=" << metrics.q_v_min
       << " shell_r=" << metrics.shell_radius
       << " patch_inner_r=" << metrics.patch_inner_radius
       << " reserve_min=" << reserve.reserve_min
       << " cell=" << reserve.cell
       << " k=" << reserve.corner
       << " Jnow=" << reserve.jnow
       << " Jhat1=" << reserve.jhat1
       << " Jhat4=" << reserve.jhat4
       << " hold_mode=" << (controller.bcr_hold_heal ? "HEAL" : "VERIFY");
  core::log_info(line.str());
}

void record_bcr_state_machine(CoreClearanceState& controller,
                              const core::State& state,
                              const int step,
                              const double t) {
  if (!controller.bcr_predictor || step < controller.adot_from_step) {
    return;
  }
  controller.bcr_clock_step = step;
  controller.bcr_clock_t = t;
  TENRYU_ASSERT(state.dt > 0.0 && std::isfinite(state.dt),
                "BCR predictor requires a positive finite timestep");
  const std::size_t n_nodes =
      static_cast<std::size_t>(state.mesh.topo.n_nodes);
  TENRYU_ASSERT(state.x_r.size() == n_nodes && state.x_z.size() == n_nodes &&
                    state.v_r.size() == n_nodes && state.v_z.size() == n_nodes,
                "BCR predictor node field size changed");

  std::vector<double> node_r;
  std::vector<double> node_z;
  std::vector<double> velocity_r;
  std::vector<double> velocity_z;
  std::vector<double> current_volume;
  state.x_r.copy_to_host(node_r);
  state.x_z.copy_to_host(node_z);
  state.v_r.copy_to_host(velocity_r);
  state.v_z.copy_to_host(velocity_z);
  state.vol.copy_to_host(current_volume);
  if (step % core_clearance_cadence_steps() == 0) {
    const double h_r = bcr_patch_median_radial_thickness(
        controller, node_r, node_z);
    TENRYU_ASSERT(std::isfinite(h_r) && h_r > 0.0,
                  "BCR radial thickness update is invalid");
    controller.bcr_h_r = h_r;
  }

  const BcrReserveDiagnostic reserve = compute_bcr_reserve(
      controller, node_r, node_z, velocity_r, velocity_z, state.dt);
  const BcrQualityMetrics metrics = compute_bcr_quality_metrics(
      controller, state, node_r, node_z, velocity_r, velocity_z,
      current_volume);
  const G31Variant variant = g31_variant();
  controller.bcr_tau_geom = metrics.tau_geom;
  controller.bcr_tau_cell = metrics.tau_cell;
  controller.bcr_q_j_min = metrics.q_j_min;
  controller.bcr_q_v_min = metrics.q_v_min;
  controller.bcr_target_scale = metrics.target_scale;
  controller.bcr_h_r = metrics.h_r;
  controller.bcr_shell_radius = metrics.shell_radius;
  controller.bcr_patch_inner_radius = metrics.patch_inner_radius;
  if (variant != G31Variant::LEGACY) {
    controller.g31_tau_hard = metrics.tau_hard;
    controller.g31_reserve_min = reserve.reserve_min;
    controller.g31_q_min = bcr_quality_min(metrics);
    if (controller.g31_awaiting_post_sample) {
      controller.g31_q_post_commit = controller.g31_q_min;
      controller.g31_last_gain =
          controller.g31_q_post_commit - controller.g31_q_pre_commit;
      controller.g31_awaiting_post_sample = false;
      if (controller.rank == 0) {
        std::ostringstream line;
        line << std::scientific << std::setprecision(6)
             << "[g31] step=" << step
             << " epoch=" << controller.g31_capture_epoch_id
             << " commit_gain g_m=" << controller.g31_last_gain
             << " q_pre=" << controller.g31_q_pre_commit
             << " q_post=" << controller.g31_q_post_commit;
        core::log_info(line.str());
      }
    }
  }

  if (!controller.bcr_rezone) {
    if (!controller.bcr_predictor_triggered && reserve.reserve_min <= 4) {
      controller.bcr_predictor_triggered = true;
      if (controller.bcr_predictor_first_trigger_step < 0) {
        controller.bcr_predictor_first_trigger_step = step;
      }
      if (controller.rank == 0) {
        core::log_info("[bcr-pred] TRIGGERED step=" +
                       std::to_string(step));
      }
    } else if (controller.bcr_predictor_triggered &&
               reserve.reserve_min >= 8) {
      controller.bcr_predictor_triggered = false;
      if (controller.rank == 0) {
        core::log_info("[bcr-pred] RELEASED step=" +
                       std::to_string(step));
      }
    }
    if (controller.bcr_predictor_triggered) {
      ++controller.bcr_predictor_triggered_steps;
    }
    emit_bcr_state_sample(controller, reserve, metrics, step, t);
    if (variant != G31Variant::LEGACY) {
      controller.g31_geometry_retry_this_step = false;
    }
    return;
  }

  const double q_min = bcr_quality_min(metrics);
  if (controller.bcr_phase == BcrPhase::ARMED) {
    const bool trigger =
        metrics.tau_geom <= metrics.tau_cell ||
        metrics.q_j_min <= kBcrQualityOn ||
        metrics.q_v_min <= kBcrQualityOn || reserve.reserve_min <= 4;
    if (trigger) {
      transition_bcr_phase(controller, BcrPhase::CAPTURE, step, t);
      controller.bcr_dwell_history.clear();
      controller.bcr_predictor_triggered = true;
      if (variant == G31Variant::LEGACY) {
        controller.bcr_capture_requested = true;
      } else {
        controller.g31_episode_open = true;
        ++controller.g31_capture_epoch_id;
        controller.g31_tau_hard_used = false;
      }
      if (controller.bcr_predictor_first_trigger_step < 0) {
        controller.bcr_predictor_first_trigger_step = step;
      }
    }
  } else if (controller.bcr_phase == BcrPhase::RIDE &&
             metrics.trailing_clearance) {
    transition_bcr_phase(controller, BcrPhase::RECOVERY_HOLD, step, t);
  }

  if (controller.bcr_phase == BcrPhase::RECOVERY_HOLD ||
      controller.bcr_phase == BcrPhase::RELEASE_RAMP) {
    append_bcr_quality_history(controller, step, t, q_min);
  }

  if (controller.bcr_phase == BcrPhase::RECOVERY_HOLD) {
    const double reserve_required =
        std::max(metrics.tau_cell, 8.0 * state.dt);
    controller.bcr_hold_heal =
        q_min < kBcrQualityOff ||
        metrics.tau_geom < reserve_required;
    controller.bcr_predictor_triggered = controller.bcr_hold_heal;
    if (bcr_release_eligible(controller, metrics, step, t, state.dt)) {
      transition_bcr_phase(controller, BcrPhase::RELEASE_RAMP, step, t);
      controller.bcr_release_ramp_start_step = step;
      controller.bcr_release_ramp_start_t = t;
      controller.bcr_release_ramp_tau =
          std::max(0.5 * metrics.tau_cell,
                   static_cast<double>(kBcrRampAcceptedSteps) * state.dt);
      controller.bcr_predictor_triggered = true;
      controller.bcr_hold_heal = false;
    }
  } else if (controller.bcr_phase == BcrPhase::RELEASE_RAMP) {
    if (!bcr_release_eligible(controller, metrics, step, t, state.dt)) {
      transition_bcr_phase(controller, BcrPhase::RECOVERY_HOLD, step, t);
      controller.bcr_hold_heal = true;
      controller.bcr_predictor_triggered = true;
    } else {
      controller.bcr_predictor_triggered = true;
      const bool physical_ramp_complete =
          t - controller.bcr_release_ramp_start_t >
          controller.bcr_release_ramp_tau;
      const bool step_ramp_complete =
          step - controller.bcr_release_ramp_start_step >
          kBcrRampAcceptedSteps;
      if (physical_ramp_complete && step_ramp_complete) {
        transition_bcr_phase(controller, BcrPhase::ARMED, step, t);
        controller.bcr_predictor_triggered = false;
        controller.bcr_capture_requested = false;
        controller.bcr_hold_heal = false;
        controller.bcr_dwell_history.clear();
        if (variant != G31Variant::LEGACY) {
          controller.g31_episode_open = false;
          controller.g31_tau_hard_used = false;
          controller.g31_last_gain =
              std::numeric_limits<double>::quiet_NaN();
          controller.g31_q_pre_commit =
              std::numeric_limits<double>::quiet_NaN();
          controller.g31_q_post_commit =
              std::numeric_limits<double>::quiet_NaN();
          controller.g31_awaiting_post_sample = false;
        }
      }
    }
  } else if (controller.bcr_phase == BcrPhase::CAPTURE ||
             controller.bcr_phase == BcrPhase::RIDE) {
    controller.bcr_predictor_triggered = true;
    controller.bcr_hold_heal = false;
  }

  if (controller.bcr_phase != BcrPhase::ARMED) {
    ++controller.bcr_predictor_triggered_steps;
  }
  emit_bcr_state_sample(controller, reserve, metrics, step, t);
  if (variant != G31Variant::LEGACY) {
    controller.g31_geometry_retry_this_step = false;
  }
}

double splice_fraction(const CoreClearanceState& controller,
                       const double t) {
  if (controller.phase == ClearancePhase::STATIC) {
    return 0.0;
  }
  if (controller.phase == ClearancePhase::TRACKING) {
    return 1.0;
  }
  const double x = std::clamp(
      (t - controller.armed_t) / controller.tau_splice, 0.0, 1.0);
  return x * x * x * (10.0 + x * (-15.0 + 6.0 * x));
}

void emit_sample(const CoreClearanceState& controller,
                 const ClearanceSample& sample,
                 const bool final) {
  if (controller.rank != 0) {
    return;
  }
  std::ostringstream line;
  line << (final ? "[clearance] FINAL" : "[clearance]")
       << " step=" << sample.step
       << " t=" << std::scientific << std::setprecision(9) << sample.t
       << " state=" << phase_name(sample.phase)
       << " front_valid=" << (sample.front_valid ? 1 : 0)
       << " s_f=" << std::setprecision(6) << sample.s_f
       << " U_f=" << sample.u_f
       << " tau_hit=" << std::setprecision(3) << sample.tau_hit
       << " phi=" << std::fixed << std::setprecision(4) << sample.phi
       << " R_cmd=" << std::scientific << std::setprecision(9)
       << sample.r_cmd
       << " g_guard=" << std::setprecision(6) << sample.g_guard
       << " h95=" << std::setprecision(3) << sample.h95
       << " U95=" << sample.u95
       << " nElig=" << sample.n_eligible;
  core::log_info(line.str());
}

void initialize_controller(CoreClearanceState& controller,
                           const core::State& state,
                           const core::Config& cfg,
                           const int rank) {
  TENRYU_ASSERT(state.mesh.dim == 2,
                "core clearance controller requires a two-dimensional mesh");
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "core clearance controller requires multiblock topology");
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  TENRYU_ASSERT(n_cells > 0 && n_nodes > 0,
                "core clearance controller requires nonempty mesh topology");
  TENRYU_ASSERT(state.x_r.size() == static_cast<std::size_t>(n_nodes) &&
                    state.x_z.size() == static_cast<std::size_t>(n_nodes) &&
                    state.vol.size() == static_cast<std::size_t>(n_cells),
                "core clearance initial field sizes are incomplete");

  std::vector<double> initial_node_r;
  std::vector<double> initial_node_z;
  state.x_r.copy_to_host(initial_node_r);
  state.x_z.copy_to_host(initial_node_z);
  if (controller.bcr_sets) {
    TENRYU_ASSERT(state.x_r_initial.size() ==
                      static_cast<std::size_t>(n_nodes) &&
                      state.x_z_initial.size() ==
                          static_cast<std::size_t>(n_nodes),
                  "BCR rezone requires t=0 node coordinates");
    state.x_r_initial.copy_to_host(controller.initial_node_r);
    state.x_z_initial.copy_to_host(controller.initial_node_z);
  }
  state.vol.copy_to_host(controller.initial_volume);
  state.mesh.multiblock_cell_node_csr_offsets.copy_to_host(
      controller.cell_node_offsets);
  state.mesh.multiblock_cell_node_csr_indices.copy_to_host(
      controller.cell_node_indices);
  controller.cell_nverts = state.mesh.cell_nverts;
  TENRYU_ASSERT(controller.cell_node_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U,
                "core clearance cell-node CSR offsets are incomplete");
  TENRYU_ASSERT(controller.cell_nverts.empty() ||
                    controller.cell_nverts.size() ==
                        static_cast<std::size_t>(n_cells),
                "core clearance active vertex counts are incomplete");
  controller.adot_cell = adot_ledger_cell();
  controller.adot_from_step = adot_ledger_from_step();
  if (controller.adot_cell >= 0) {
    TENRYU_ASSERT(controller.adot_cell < n_cells,
                  "adot ledger watch cell is out of range");
    validate_cell_csr(controller, controller.adot_cell, n_nodes);
    TENRYU_ASSERT(active_nverts(controller, controller.adot_cell) == 5,
                  "adot ledger watch cell must be a pentagon");
  }

  controller.r0 = cfg.numerics.ale.euler_window.rad_out;
  controller.w0 = cfg.numerics.ale.euler_window.transition_width;
  TENRYU_ASSERT(kClearanceOuterRadius > controller.r0,
                "core clearance outer bin radius must exceed R0");
  controller.bin_width =
      (kClearanceOuterRadius - controller.r0) /
      static_cast<double>(kClearanceBinCount);
  controller.tau_lead = controller.active
                            ? cfg.numerics.ale.euler_window.replay_tau_lead
                            : positive_env_value(
                                  "TENRYU_I1B_CLR_TAU_LEAD", 4.5e-12);
  controller.tau_splice = controller.active
                              ? cfg.numerics.ale.euler_window.replay_tau_splice
                              : positive_env_value(
                                    "TENRYU_I1B_CLR_TAU_SPLICE", 1.4e-12);
  controller.beta = controller.active
                        ? cfg.numerics.ale.euler_window.replay_beta
                        : nonnegative_env_value(
                              "TENRYU_I1B_CLR_BETA", 1.0);
  controller.u_floor = positive_env_value(
      "TENRYU_I1B_CLR_UFLOOR", 1.0e6);
  controller.r_cmd = controller.r0;
  controller.r_acc = controller.r0;
  controller.previous_evaluation_t = state.t;
  controller.bcr_clock_step = state.step;
  controller.bcr_clock_t = state.t;
  controller.rank = rank;

  if (controller.active) {
    load_replay_table(
        controller, cfg.numerics.ale.euler_window.replay_table_path);
    controller.initial_node_radius.resize(static_cast<std::size_t>(n_nodes));
    controller.initial_director_r.resize(static_cast<std::size_t>(n_nodes));
    controller.initial_director_z.resize(static_cast<std::size_t>(n_nodes));
    controller.active_omega.resize(static_cast<std::size_t>(n_nodes));
    for (int node = 0; node < n_nodes; ++node) {
      const std::size_t index = static_cast<std::size_t>(node);
      const double radius =
          std::hypot(initial_node_r[index], initial_node_z[index]);
      controller.initial_node_radius[index] = radius;
      if (radius > 0.0) {
        controller.initial_director_r[index] = initial_node_r[index] / radius;
        controller.initial_director_z[index] = initial_node_z[index] / radius;
      }
      double omega = 0.0;
      if (radius <= controller.r0) {
        omega = 1.0;
      } else if (radius < controller.r0 + controller.w0) {
        const double x = (radius - controller.r0) / controller.w0;
        const double s5 = x * x * x * (10.0 + x * (-15.0 + 6.0 * x));
        omega = 1.0 - s5;
      }
      controller.active_omega[index] = omega;
      if (radius <= controller.r0 + controller.w0) {
        controller.controlled_nodes.push_back(node);
      }
    }
  }

  controller.initial_radius.resize(static_cast<std::size_t>(n_cells));
  for (int cell = 0; cell < n_cells; ++cell) {
    validate_cell_csr(controller, cell, n_nodes);
    const std::size_t index = static_cast<std::size_t>(cell);
    const VectorRz centroid = cell_vertex_mean(
        controller, cell, initial_node_r, initial_node_z);
    const double radius = std::hypot(centroid.r, centroid.z);
    controller.initial_radius[index] = radius;
    const bool active = std::isfinite(centroid.r) &&
                        std::isfinite(centroid.z) &&
                        controller.initial_volume[index] > 0.0 &&
                        std::isfinite(controller.initial_volume[index]);
    if (active && radius > controller.r0 &&
        radius <= kClearanceOuterRadius) {
      int bin = static_cast<int>(
          (radius - controller.r0) / controller.bin_width);
      bin = std::clamp(bin, 0, kClearanceBinCount - 1);
      controller.bins[static_cast<std::size_t>(bin)].push_back(cell);
    }
    if (radius >= controller.r0 - controller.w0 &&
        radius <= controller.r0) {
      controller.feather_cells.push_back(cell);
    }
    if (radius > controller.r0 &&
        radius <= controller.r0 + 3.0 * controller.w0) {
      controller.rplus_cells.push_back(cell);
    }
  }
  if (controller.bcr_sets) {
    build_bcr_sets(controller, n_cells, n_nodes);
    controller.bcr_h_r = bcr_patch_median_radial_thickness(
        controller, initial_node_r, initial_node_z);
    TENRYU_ASSERT(std::isfinite(controller.bcr_h_r) &&
                      controller.bcr_h_r > 0.0,
                  "BCR initial radial thickness is invalid");
  }

  for (std::vector<int>& bin : controller.bins) {
    std::sort(bin.begin(), bin.end(),
              [&controller](const int lhs, const int rhs) {
                const double lhs_radius = controller.initial_radius[
                    static_cast<std::size_t>(lhs)];
                const double rhs_radius = controller.initial_radius[
                    static_cast<std::size_t>(rhs)];
                if (lhs_radius != rhs_radius) {
                  return lhs_radius < rhs_radius;
                }
                return lhs < rhs;
              });
    controller.wideband_cells.insert(controller.wideband_cells.end(),
                                     bin.begin(), bin.end());
  }

  controller.previous_rplus_volume.reserve(controller.rplus_cells.size());
  if (controller.active) {
    std::vector<RankedValue> thickness_ranked;
    thickness_ranked.reserve(controller.rplus_cells.size());
    for (const int cell : controller.rplus_cells) {
      const double thickness = radial_cell_thickness(
          controller, cell, initial_node_r, initial_node_z);
      if (std::isfinite(thickness)) {
        thickness_ranked.push_back({thickness, 0.0, cell});
      }
      controller.previous_rplus_volume.push_back(
          controller.initial_volume[static_cast<std::size_t>(cell)]);
    }
    controller.previous_rplus_sample_valid = true;
    controller.active_h95 =
        deterministic_percentile(std::move(thickness_ranked), 0.95);
    TENRYU_ASSERT(std::isfinite(controller.active_h95),
                  "clearance replay initial h95 is unavailable");
    controller.active_g_guard = controller.w0 + controller.active_h95;
  }
  controller.last_sample.step = state.step;
  controller.last_sample.t = state.t;
  controller.last_sample.r_cmd = controller.r_cmd;
  controller.last_sample.r_acc = controller.r_acc;
  controller.initialized = true;

  if (rank == 0) {
    if (controller.active) {
      std::ostringstream manifest;
      manifest << std::scientific << std::setprecision(9)
               << "[clearance-active] manifest R0=" << controller.r0
               << " W0=" << controller.w0
               << " h95=" << controller.active_h95
               << " g_guard=" << controller.active_g_guard
               << " tauLead=" << controller.tau_lead
               << " tauSplice=" << controller.tau_splice
               << " beta=" << controller.beta
               << " rows=" << controller.replay.size()
               << " controlled_nodes=" << controller.controlled_nodes.size()
               << " replay_hash=0x" << std::hex << std::setw(16)
               << std::setfill('0')
               << static_cast<unsigned long long>(controller.replay_hash);
      core::log_info(manifest.str());
      core::log_info(
          "[clearance-active] SHADOW-V1-SIMPLIFICATION W_now=W0 h95=initial");
      return;
    }
    if (!core_clearance_shadow_enabled()) {
      return;
    }
    std::ostringstream manifest;
    manifest << std::scientific << std::setprecision(9)
             << "[clearance] manifest R0=" << controller.r0
             << " W0=" << controller.w0
             << " nBins=" << kClearanceBinCount
             << " binWidth=" << controller.bin_width
             << " tauLead=" << controller.tau_lead
             << " tauSplice=" << controller.tau_splice
             << " beta=" << controller.beta
             << " cadence=" << core_clearance_cadence_steps();
    core::log_info(manifest.str());
    for (int bin = 0; bin < kClearanceBinCount; ++bin) {
      core::log_info(
          "[clearance] manifest bin=" + std::to_string(bin) +
          " count=" +
          std::to_string(controller.bins[static_cast<std::size_t>(bin)].size()));
    }
    core::log_info(
        "[clearance] SHADOW-V1-SIMPLIFICATION W_now=W0");
  }
}

ClearanceSample capture_sample(CoreClearanceState& controller,
                               const core::State& state,
                               const int step,
                               const double t) {
  const std::size_t n_cells = controller.initial_volume.size();
  const std::size_t n_nodes = state.mesh.topo.n_nodes;
  TENRYU_ASSERT(state.x_r.size() == n_nodes && state.x_z.size() == n_nodes &&
                    state.v_r.size() == n_nodes && state.v_z.size() == n_nodes,
                "core clearance node field size changed");
  TENRYU_ASSERT(state.vol.size() == n_cells,
                "core clearance cell field size changed");

  std::vector<double> node_r;
  std::vector<double> node_z;
  std::vector<double> velocity_r;
  std::vector<double> velocity_z;
  std::vector<double> current_volume;
  state.x_r.copy_to_host(node_r);
  state.x_z.copy_to_host(node_z);
  state.v_r.copy_to_host(velocity_r);
  state.v_z.copy_to_host(velocity_z);
  state.vol.copy_to_host(current_volume);

  std::vector<double> current_radius(n_cells, quiet_nan());
  std::vector<double> radial_velocity(n_cells, quiet_nan());
  std::vector<double> inward_speed(n_cells, quiet_nan());
  std::vector<double> volume_ratio(n_cells, quiet_nan());
  std::vector<RankedValue> inward_speed_ranked;
  inward_speed_ranked.reserve(controller.wideband_cells.size());
  for (const int cell : controller.wideband_cells) {
    const std::size_t index = static_cast<std::size_t>(cell);
    const VectorRz centroid = cell_vertex_mean(
        controller, cell, node_r, node_z);
    const VectorRz centroid_velocity = cell_vertex_mean(
        controller, cell, velocity_r, velocity_z);
    const double radius = std::hypot(centroid.r, centroid.z);
    current_radius[index] = radius;
    if (radius > 0.0 && std::isfinite(radius)) {
      radial_velocity[index] =
          (centroid_velocity.r * centroid.r +
           centroid_velocity.z * centroid.z) /
          radius;
      inward_speed[index] = -radial_velocity[index];
    }
    volume_ratio[index] =
        current_volume[index] / controller.initial_volume[index];
    if (std::isfinite(inward_speed[index])) {
      inward_speed_ranked.push_back(
          {inward_speed[index], 0.0, cell});
    }
  }

  ClearanceSample sample;
  sample.step = step;
  sample.t = t;
  sample.u95 = deterministic_percentile(
      std::move(inward_speed_ranked), 0.95);
  const double eligibility_speed = std::max(
      controller.u_floor,
      std::isfinite(sample.u95) ? 0.25 * sample.u95
                                : controller.u_floor);

  std::array<double, kClearanceBinCount> bin_radius{};
  std::array<double, kClearanceBinCount> bin_inward_speed{};
  std::array<double, kClearanceBinCount> bin_compression{};
  std::array<bool, kClearanceBinCount> eligible{};
  for (int bin = 0; bin < kClearanceBinCount; ++bin) {
    const std::vector<int>& members =
        controller.bins[static_cast<std::size_t>(bin)];
    bin_radius[static_cast<std::size_t>(bin)] = weighted_median(
        members, current_radius, controller.initial_volume);
    bin_inward_speed[static_cast<std::size_t>(bin)] = -weighted_median(
        members, radial_velocity, controller.initial_volume);
    bin_compression[static_cast<std::size_t>(bin)] = weighted_median(
        members, volume_ratio, controller.initial_volume);
    eligible[static_cast<std::size_t>(bin)] =
        std::isfinite(bin_inward_speed[static_cast<std::size_t>(bin)]) &&
        std::isfinite(bin_compression[static_cast<std::size_t>(bin)]) &&
        bin_inward_speed[static_cast<std::size_t>(bin)] >=
            eligibility_speed &&
        bin_compression[static_cast<std::size_t>(bin)] <= 0.95;
    sample.n_eligible +=
        eligible[static_cast<std::size_t>(bin)] ? 1 : 0;
  }

  int front_bin = -1;
  for (int bin = 0; bin + 1 < kClearanceBinCount; ++bin) {
    if (eligible[static_cast<std::size_t>(bin)] &&
        eligible[static_cast<std::size_t>(bin + 1)]) {
      front_bin = bin;
      break;
    }
  }
  if (front_bin >= 0) {
    sample.front_valid = true;
    sample.s_f = bin_radius[static_cast<std::size_t>(front_bin)];
    std::vector<int> front_speed_members =
        controller.bins[static_cast<std::size_t>(front_bin)];
    const std::vector<int>& outward_members =
        controller.bins[static_cast<std::size_t>(front_bin + 1)];
    front_speed_members.insert(front_speed_members.end(),
                               outward_members.begin(),
                               outward_members.end());
    sample.u_f = weighted_median(
        front_speed_members, inward_speed, controller.initial_volume);
  }

  std::vector<RankedValue> thickness_ranked;
  thickness_ranked.reserve(controller.rplus_cells.size());
  for (const int cell : controller.rplus_cells) {
    const double thickness =
        radial_cell_thickness(controller, cell, node_r, node_z);
    if (std::isfinite(thickness)) {
      thickness_ranked.push_back({thickness, 0.0, cell});
    }
  }
  sample.h95 = deterministic_percentile(std::move(thickness_ranked), 0.95);
  sample.g_guard = controller.w0 + sample.h95;
  if (sample.front_valid) {
    sample.tau_hit =
        (sample.s_f - controller.r0 - sample.g_guard) /
        std::max(sample.u_f, controller.u_floor);
  }

  if (controller.phase == ClearancePhase::STATIC && sample.front_valid &&
      sample.tau_hit <= controller.tau_lead) {
    controller.phase = ClearancePhase::SPLICING;
    controller.armed_step = step;
    controller.armed_t = t;
  }
  sample.phi = splice_fraction(controller, t);
  if (controller.phase == ClearancePhase::SPLICING && sample.phi == 1.0) {
    controller.phase = ClearancePhase::TRACKING;
    controller.tracking_step = step;
    controller.tracking_t = t;
  }

  const double evaluation_dt = t - controller.previous_evaluation_t;
  if (evaluation_dt > 0.0) {
    controller.r_cmd +=
        -controller.beta * sample.phi * sample.u_f * evaluation_dt;
  }

  std::vector<RankedValue> log_volume_rate_ranked;
  if (controller.previous_rplus_sample_valid && evaluation_dt > 0.0) {
    TENRYU_ASSERT(controller.previous_rplus_volume.size() ==
                      controller.rplus_cells.size(),
                  "core clearance RPLUS snapshot size changed");
    log_volume_rate_ranked.reserve(controller.rplus_cells.size());
    for (std::size_t slot = 0; slot < controller.rplus_cells.size(); ++slot) {
      const int cell = controller.rplus_cells[slot];
      const double volume = current_volume[static_cast<std::size_t>(cell)];
      const double previous_volume = controller.previous_rplus_volume[slot];
      if (volume > 0.0 && previous_volume > 0.0 &&
          std::isfinite(volume) && std::isfinite(previous_volume)) {
        log_volume_rate_ranked.push_back(
            {(std::log(volume) - std::log(previous_volume)) / evaluation_dt,
             0.0, cell});
      }
    }
  }
  const double rplus_log_volume_rate = deterministic_percentile(
      std::move(log_volume_rate_ranked), 0.5);
  if (controller.phase == ClearancePhase::STATIC &&
      std::isfinite(rplus_log_volume_rate) &&
      rplus_log_volume_rate < -1.0e8 &&
      !controller.missed_trigger_logged) {
    if (controller.rank == 0) {
      core::log_info(
          "[clearance] FAIL-CLOSED MISSED_PREDICTIVE_TRIGGER step=" +
          std::to_string(step));
    }
    controller.missed_trigger_logged = true;
  }

  controller.previous_rplus_volume.clear();
  controller.previous_rplus_volume.reserve(controller.rplus_cells.size());
  for (const int cell : controller.rplus_cells) {
    controller.previous_rplus_volume.push_back(
        current_volume[static_cast<std::size_t>(cell)]);
  }
  controller.previous_rplus_sample_valid = true;
  controller.previous_evaluation_t = t;

  sample.phase = controller.phase;
  sample.r_cmd = controller.r_cmd;
  return sample;
}

std::string active_state_message(const char* event,
                                 const CoreClearanceState& controller,
                                 const ClearanceSample& sample) {
  std::ostringstream message;
  message << std::scientific << std::setprecision(17)
          << event
          << " step=" << sample.step
          << " t=" << sample.t
          << " state=" << phase_name(sample.phase)
          << " s_f=" << sample.s_f
          << " U_f=" << sample.u_f
          << " tau_hit=" << sample.tau_hit
          << " phi=" << sample.phi
          << " R_cmd=" << sample.r_cmd
          << " R_acc=" << sample.r_acc
          << " beta_eff=" << sample.beta_eff
          << " sigma=" << sample.sigma
          << " g_guard=" << controller.active_g_guard
          << " clearance=" << (sample.s_f - sample.r_acc)
          << " low_beta_steps=" << controller.low_beta_eff_steps
          << " support_exhausted="
          << (controller.support_exhausted ? 1 : 0);
  return message.str();
}

void mark_replay_support_exhausted(CoreClearanceState& controller,
                                   const int step,
                                   const double t) {
  controller.support_exhausted = true;
  if (!controller.support_exhausted_logged && controller.rank == 0) {
    std::ostringstream message;
    message << std::scientific << std::setprecision(17)
            << "[clearance-active] REPLAY_SUPPORT_EXHAUSTED"
            << " step=" << step
            << " t=" << t
            << " table_t_last=" << controller.replay.back().t
            << " R_cmd_hold=" << controller.r_cmd
            << " R_acc=" << controller.r_acc;
    core::log_warning(message.str());
  }
  controller.support_exhausted_logged = true;
}

void refresh_active_rplus_snapshot(CoreClearanceState& controller,
                                   const std::vector<double>& current_volume) {
  TENRYU_ASSERT(current_volume.size() == controller.initial_volume.size(),
                "clearance replay cell field size changed");
  controller.previous_rplus_volume.clear();
  controller.previous_rplus_volume.reserve(controller.rplus_cells.size());
  for (const int cell : controller.rplus_cells) {
    controller.previous_rplus_volume.push_back(
        current_volume[static_cast<std::size_t>(cell)]);
  }
  controller.previous_rplus_sample_valid = true;
}

void update_active_late_alarm(CoreClearanceState& controller,
                              const core::State& state,
                              ClearanceSample& sample,
                              const double evaluation_t) {
  std::vector<double> current_volume;
  state.vol.copy_to_host(current_volume);
  TENRYU_ASSERT(current_volume.size() == controller.initial_volume.size(),
                "clearance replay cell field size changed");
  const double evaluation_dt = evaluation_t - controller.previous_evaluation_t;
  std::vector<RankedValue> rates;
  if (controller.previous_rplus_sample_valid && evaluation_dt > 0.0) {
    TENRYU_ASSERT(controller.previous_rplus_volume.size() ==
                      controller.rplus_cells.size(),
                  "clearance replay RPLUS snapshot size changed");
    rates.reserve(controller.rplus_cells.size());
    for (std::size_t slot = 0; slot < controller.rplus_cells.size(); ++slot) {
      const int cell = controller.rplus_cells[slot];
      const double volume = current_volume[static_cast<std::size_t>(cell)];
      const double previous_volume = controller.previous_rplus_volume[slot];
      if (volume > 0.0 && previous_volume > 0.0 &&
          std::isfinite(volume) && std::isfinite(previous_volume)) {
        rates.push_back(
            {(std::log(volume) - std::log(previous_volume)) / evaluation_dt,
             0.0,
             cell});
      }
    }
  }
  const double late_alarm =
      deterministic_percentile(std::move(rates), 0.5);
  refresh_active_rplus_snapshot(controller, current_volume);
  controller.previous_evaluation_t = evaluation_t;
  if (controller.phase == ClearancePhase::STATIC &&
      std::isfinite(late_alarm) && late_alarm < -1.0e8) {
    sample.phase = controller.phase;
    if (controller.beta == 0.0) {
      if (!controller.missed_trigger_logged && controller.rank == 0) {
        core::log_warning(active_state_message(
            "MISSED_PREDICTIVE_TRIGGER", controller, sample));
      }
      controller.missed_trigger_logged = true;
    } else {
      TENRYU_ASSERT(
          false,
          active_state_message(
              "MISSED_PREDICTIVE_TRIGGER", controller, sample));
    }
  }
}

void prepare_active_step(CoreClearanceState& controller,
                         const core::State& state,
                         const int step,
                         const double t,
                         const double dt) {
  TENRYU_ASSERT(dt > 0.0 && std::isfinite(dt),
                "clearance replay requires a positive finite timestep");
  ClearanceSample sample;
  sample.step = step;
  sample.t = t + 0.5 * dt;
  sample.r_cmd = controller.r_cmd;
  sample.r_acc = controller.r_acc;
  sample.g_guard = controller.active_g_guard;
  sample.h95 = controller.active_h95;
  sample.sigma = -1.0;

  if (controller.support_exhausted || t + dt > controller.replay.back().t) {
    mark_replay_support_exhausted(controller, step, t + dt);
    const ReplayPoint last =
        interpolate_replay(controller, controller.replay.back().t);
    sample.s_f = last.s_f;
    sample.u_f = last.u_f;
    sample.phi = splice_fraction(controller, sample.t);
    if (controller.phase == ClearancePhase::SPLICING && sample.phi == 1.0) {
      controller.phase = ClearancePhase::TRACKING;
      controller.tracking_step = step;
      controller.tracking_t = sample.t;
    }
    sample.phase = controller.phase;
    controller.active_u_half = last.u_f;
    controller.active_s_end = last.s_f;
    update_active_late_alarm(controller, state, sample, t + dt);
    controller.last_sample = sample;
    return;
  }

  const ReplayPoint front_now = interpolate_replay(controller, t);
  const ReplayPoint front_half = interpolate_replay(controller, t + 0.5 * dt);
  const ReplayPoint front_end = interpolate_replay(controller, t + dt);
  sample.s_f = front_half.s_f;
  sample.u_f = front_half.u_f;
  sample.tau_hit =
      (front_now.s_f - controller.r_acc - controller.active_g_guard) /
      front_now.u_f;
  if (controller.phase == ClearancePhase::STATIC &&
      sample.tau_hit <= controller.tau_lead) {
    controller.phase = ClearancePhase::SPLICING;
    controller.armed_step = step;
    controller.armed_t = t;
  }
  sample.phi = splice_fraction(controller, t + 0.5 * dt);
  if (controller.phase == ClearancePhase::SPLICING && sample.phi == 1.0) {
    controller.phase = ClearancePhase::TRACKING;
    controller.tracking_step = step;
    controller.tracking_t = t + 0.5 * dt;
  }
  controller.r_cmd =
      controller.r_acc - controller.beta * sample.phi * front_half.u_f * dt;
  controller.active_u_half = front_half.u_f;
  controller.active_s_end = front_end.s_f;
  sample.phase = controller.phase;
  sample.r_cmd = controller.r_cmd;
  update_active_late_alarm(controller, state, sample, t + dt);
  controller.last_sample = sample;
}

void emit_active_sample(const CoreClearanceState& controller,
                        const ClearanceSample& sample) {
  if (controller.rank != 0) {
    return;
  }
  std::ostringstream line;
  line << std::scientific << std::setprecision(9)
       << "[clearance-active] step=" << sample.step
       << " t=" << sample.t
       << " state=" << phase_name(sample.phase)
       << " s_f=" << sample.s_f
       << " U_f=" << sample.u_f
       << " tau_hit=" << sample.tau_hit
       << " phi=" << sample.phi
       << " R_cmd=" << sample.r_cmd
       << " R_acc=" << sample.r_acc
       << " beta_eff=" << sample.beta_eff
       << " sigma=" << sample.sigma;
  core::log_info(line.str());
}

}  // namespace

void core_clearance_controller_run_start(const core::State& state,
                                         const core::Config& cfg,
                                         const int rank) {
  const bool active =
      cfg.numerics.ale.euler_window.axis_core_transaction_mode ==
      "clearance_replay";
  const bool bcr_sets_requested = bcr_sets_enabled();
  const bool bcr_continuous_requested = bcr_continuous_enabled();
  const bool bcr_predictor_requested = bcr_predictor_enabled();
  const bool bcr_rezone_requested = bcr_rezone_enabled();
  const bool bcr_target_requested = bcr_target_enabled();
  (void)g31_variant();
  const bool bcr_continuous =
      bcr_continuous_requested && bcr_sets_requested &&
      bcr_rezone_requested && bcr_target_requested;
  if (bcr_continuous_requested && !bcr_continuous && rank == 0) {
    core::log_warning(
        "[bcr-continuous] TENRYU_I1B_BCR_CONTINUOUS=1 requires "
        "TENRYU_I1B_BCR_SETS=1, TENRYU_I1B_BCR_REZONE=1, and "
        "TENRYU_I1B_BCR_TARGET=1; continuous mode disabled");
  }
  if (bcr_predictor_requested && !bcr_sets_requested && rank == 0) {
    core::log_warning(
        "[bcr-pred] TENRYU_I1B_BCR_PREDICTOR=1 requires "
        "TENRYU_I1B_BCR_SETS=1; predictor disabled");
  }
  if (bcr_rezone_requested &&
      (!bcr_sets_requested ||
       (!bcr_predictor_requested && !bcr_continuous)) && rank == 0) {
    core::log_warning(
        "[bcr-rezone] TENRYU_I1B_BCR_REZONE=1 requires "
        "TENRYU_I1B_BCR_SETS=1 and TENRYU_I1B_BCR_PREDICTOR=1; "
        "rezone disabled");
  }
  if (bcr_target_requested && !bcr_rezone_requested && rank == 0) {
    core::log_warning(
        "[bcr-target] TENRYU_I1B_BCR_TARGET=1 requires "
        "TENRYU_I1B_BCR_REZONE=1; target disabled");
  }
  if (!active && !core_clearance_shadow_enabled() && !bcr_sets_requested) {
    return;
  }
  core_clearance_active_requested() = active;
  CoreClearanceState& controller = core_clearance_state();
  if (controller.initialized || controller.multirank_unsupported) {
    return;
  }
  int world_rank = 0;
  int n_ranks = 1;
  parallel::Partition::query_world(&world_rank, &n_ranks);
  (void)world_rank;
  if (n_ranks > 1 && !active) {
    controller.multirank_unsupported = true;
    if (rank == 0 && !controller.multirank_logged) {
      core::log_info("[clearance] multirank-unsupported");
      controller.multirank_logged = true;
    }
    return;
  }
  controller.active = active;
  controller.bcr_sets = bcr_sets_requested;
  controller.bcr_continuous = bcr_continuous;
  TENRYU_ASSERT(
      !(controller.bcr_continuous &&
        g31_variant() != G31Variant::LEGACY),
      "TENRYU_I1B_G31_VARIANT is incompatible with TENRYU_I1B_BCR_CONTINUOUS");
  controller.bcr_predictor =
      !controller.bcr_continuous &&
      bcr_predictor_requested && bcr_sets_requested;
  controller.bcr_rezone =
      bcr_rezone_requested &&
      (controller.bcr_continuous || controller.bcr_predictor);
  controller.bcr_target =
      bcr_target_requested && controller.bcr_rezone;
  initialize_controller(controller, state, cfg, rank);
}

void core_clearance_controller_record_step(const core::State& state,
                                           const int step,
                                           const double t) {
  CoreClearanceState& controller = core_clearance_state();
  if (!controller.initialized || controller.multirank_unsupported ||
      controller.flushed) {
    return;
  }
  record_adot_ledger(controller, state, step, t);
  if (!controller.bcr_continuous) {
    record_bcr_state_machine(controller, state, step, t);
  }
  if (controller.bcr_sets &&
      step % core_clearance_cadence_steps() == 0) {
    emit_bcr_set_diagnostic(controller, state, step);
  }
  if (!core_clearance_shadow_enabled() || controller.active ||
      step % core_clearance_cadence_steps() != 0) {
    return;
  }
  controller.last_sample = capture_sample(controller, state, step, t);
  emit_sample(controller, controller.last_sample, false);
}

bool core_clearance_controller_bcr_rezone_active() {
  const CoreClearanceState& controller = core_clearance_state();
  return bcr_rezone_enabled() && controller.initialized &&
         controller.bcr_rezone && !controller.bcr_continuous &&
         !controller.multirank_unsupported && !controller.flushed;
}

bool core_clearance_controller_promote_cell(const int cell_id) {
  if (!core_clearance_controller_bcr_rezone_active()) {
    return false;
  }
  CoreClearanceState& controller = core_clearance_state();
  const int n_cells =
      static_cast<int>(controller.cell_node_offsets.size()) - 1;
  const int n_nodes = static_cast<int>(controller.bcr_omega.size());
  if (cell_id < 0 || cell_id >= n_cells || n_nodes <= 0 ||
      controller.bcr_priority_cells.size() !=
          static_cast<std::size_t>(n_cells)) {
    return false;
  }

  controller.bcr_priority_cells[static_cast<std::size_t>(cell_id)] = 1U;
  const bool already_protected =
      std::binary_search(controller.bcr_patch_cells.begin(),
                         controller.bcr_patch_cells.end(), cell_id);
  if (!already_protected) {
    std::vector<std::uint8_t> incident_nodes(
        static_cast<std::size_t>(n_nodes), 0U);
    const int promoted_offset = controller.cell_node_offsets[
        static_cast<std::size_t>(cell_id)];
    const int promoted_nverts = active_nverts(controller, cell_id);
    for (int corner = 0; corner < promoted_nverts; ++corner) {
      const int node = controller.cell_node_indices[
          static_cast<std::size_t>(promoted_offset + corner)];
      incident_nodes[static_cast<std::size_t>(node)] = 1U;
    }

    for (int cell = 0; cell < n_cells; ++cell) {
      const int offset = controller.cell_node_offsets[
          static_cast<std::size_t>(cell)];
      const int nverts = active_nverts(controller, cell);
      bool in_closed_star = false;
      for (int corner = 0; corner < nverts; ++corner) {
        const int node = controller.cell_node_indices[
            static_cast<std::size_t>(offset + corner)];
        in_closed_star = in_closed_star ||
                         incident_nodes[static_cast<std::size_t>(node)] != 0U;
      }
      if (!in_closed_star) {
        continue;
      }
      controller.bcr_patch_cells.push_back(cell);
      for (int corner = 0; corner < nverts; ++corner) {
        const int node = controller.cell_node_indices[
            static_cast<std::size_t>(offset + corner)];
        controller.bcr_omega[static_cast<std::size_t>(node)] = 1.0;
      }
    }
    std::sort(controller.bcr_patch_cells.begin(),
              controller.bcr_patch_cells.end());
    controller.bcr_patch_cells.erase(
        std::unique(controller.bcr_patch_cells.begin(),
                    controller.bcr_patch_cells.end()),
        controller.bcr_patch_cells.end());
    core::log_info("[bcr-ladder] promoted cell=" +
                   std::to_string(cell_id));
  }

  transition_bcr_phase(controller, BcrPhase::CAPTURE,
                       controller.bcr_clock_step,
                       controller.bcr_clock_t);
  controller.bcr_predictor_triggered = true;
  controller.bcr_capture_requested = true;
  return true;
}

bool core_clearance_controller_request_feasible_seed(const int cell_id) {
  if (!core_clearance_controller_promote_cell(cell_id)) {
    return false;
  }
  CoreClearanceState& controller = core_clearance_state();
  controller.bcr_feasible_seed_requested = true;
  controller.bcr_feasible_seed_cell = cell_id;
  return true;
}

void core_clearance_controller_note_event(
    const CoreClearanceControllerEventKind kind) {
  CoreClearanceState& controller = core_clearance_state();
  if (!controller.initialized || !controller.bcr_predictor ||
      controller.multirank_unsupported || controller.flushed) {
    return;
  }
  switch (kind) {
    case CoreClearanceControllerEventKind::FOUR_HALVINGS:
      controller.bcr_last_four_halvings_step = controller.bcr_clock_step;
      controller.bcr_last_four_halvings_t = controller.bcr_clock_t;
      break;
    case CoreClearanceControllerEventKind::BF_FIRST_REJECT:
      controller.bcr_last_bf_first_reject_step = controller.bcr_clock_step;
      controller.bcr_last_bf_first_reject_t = controller.bcr_clock_t;
      break;
    case CoreClearanceControllerEventKind::GEOMETRY_RETRY:
      controller.bcr_last_geometry_retry_step = controller.bcr_clock_step;
      controller.bcr_last_geometry_retry_t = controller.bcr_clock_t;
      if (g31_variant() != G31Variant::LEGACY) {
        controller.g31_geometry_retry_this_step = true;
      }
      break;
  }
  if (controller.bcr_phase == BcrPhase::RELEASE_RAMP) {
    transition_bcr_phase(controller, BcrPhase::RECOVERY_HOLD,
                         controller.bcr_clock_step,
                         controller.bcr_clock_t);
  }
  if (controller.bcr_phase == BcrPhase::RECOVERY_HOLD) {
    controller.bcr_hold_heal = true;
    controller.bcr_predictor_triggered = true;
  }
}

bool core_clearance_controller_g31_active() {
  const CoreClearanceState& controller = core_clearance_state();
  return g31_variant() != G31Variant::LEGACY && bcr_rezone_enabled() &&
         controller.initialized && controller.bcr_rezone &&
         !controller.multirank_unsupported && !controller.flushed;
}

bool core_clearance_controller_g31_take_trial_reject() {
  CoreClearanceState& controller = core_clearance_state();
  const bool rejected = controller.g31_trial_reject_pending;
  controller.g31_trial_reject_pending = false;
  return rejected;
}

void core_clearance_controller_pre_lagrange_rezone(
    core::State& state,
    const core::Config& cfg) {
  if (!bcr_rezone_enabled()) {
    return;
  }
  CoreClearanceState& controller = core_clearance_state();
  if (!controller.initialized || !controller.bcr_rezone ||
      controller.multirank_unsupported || controller.flushed) {
    return;
  }
  const int invocation_step = state.step + 1;
  if (controller.bcr_continuous) {
    if (invocation_step < controller.adot_from_step) {
      return;
    }
  } else if (g31_variant() != G31Variant::LEGACY) {
    const G31Variant variant = g31_variant();
    const bool ladder_forced = controller.bcr_capture_requested;
    if (ladder_forced && !controller.g31_episode_open) {
      controller.g31_episode_open = true;
      ++controller.g31_capture_epoch_id;
      controller.g31_tau_hard_used = false;
    }
    if (!controller.g31_episode_open) {
      controller.bcr_capture_requested = false;
      return;
    }

    const bool acute =
        controller.g31_reserve_min <= 4 || ladder_forced ||
        controller.g31_q_min <= kBcrQualityHard ||
        controller.g31_geometry_retry_this_step;
    const double tau_cool =
        variant == G31Variant::A2 &&
                std::isfinite(controller.g31_last_commit_t)
            ? controller.g31_tau_cool
            : 0.0;
    const double t_next =
        std::max(state.dt,
                 std::isfinite(controller.g31_last_commit_t)
                     ? controller.g31_last_commit_t + tau_cool - state.t
                     : 0.0) +
        0.25 * state.dt;
    const bool tau_hard_fires =
        variant >= G31Variant::A1 && !controller.g31_tau_hard_used &&
        controller.g31_tau_hard <= t_next;
    const bool necessity = acute || tau_hard_fires;
    if (!necessity) {
      controller.bcr_capture_requested = false;
      return;
    }

    const auto emit_refusal = [&](const char* reason) {
      if (controller.rank != 0) {
        return;
      }
      std::ostringstream line;
      line << std::scientific << std::setprecision(3)
           << "[g31] step=" << invocation_step
           << " epoch=" << controller.g31_capture_epoch_id
           << " refuse reason=" << reason
           << " q=" << controller.g31_q_min
           << " reserve=" << controller.g31_reserve_min
           << " tau_hard=" << controller.g31_tau_hard;
      core::log_info(line.str());
    };
    if (!acute && controller.g31_last_commit_step >= 0 &&
        controller.g31_last_commit_t == state.t) {
      emit_refusal("same_t");
      return;
    }

    if (variant == G31Variant::A2 && !acute &&
        std::isfinite(controller.g31_last_commit_t)) {
      const bool cooled =
          state.t - controller.g31_last_commit_t >= controller.g31_tau_cool;
      const bool gain_ok =
          !std::isfinite(controller.g31_last_gain) ||
          !std::isfinite(controller.g31_q_post_commit) ||
          controller.g31_q_min <=
              controller.g31_q_post_commit -
                  0.5 * std::max(controller.g31_last_gain, kBcrEtaQ);
      if (!cooled || !gain_ok) {
        emit_refusal(!cooled ? "cooldown" : "gain");
        return;
      }
    }

    if (tau_hard_fires && !acute) {
      controller.g31_tau_hard_used = true;
    }
    if (controller.rank == 0) {
      std::ostringstream line;
      line << std::scientific << std::setprecision(3)
           << "[g31] step=" << invocation_step
           << " epoch=" << controller.g31_capture_epoch_id
           << " transact acute=" << (acute ? 1 : 0)
           << " tau_hard_fires=" << (tau_hard_fires ? 1 : 0)
           << " reserve=" << controller.g31_reserve_min
           << " q=" << controller.g31_q_min;
      core::log_info(line.str());
    }
  } else if (!controller.bcr_predictor_triggered &&
             !controller.bcr_capture_requested) {
    return;
  }
  controller.bcr_capture_requested = false;

  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  TENRYU_ASSERT(state.mesh.dim == 2 &&
                    state.mesh.topo.multiblock.has_value(),
                "BCR rezone requires a two-dimensional multiblock mesh");
  TENRYU_ASSERT(n_cells > 0 && n_nodes > 0 &&
                    state.x_r.size() == static_cast<std::size_t>(n_nodes) &&
                    state.x_z.size() == static_cast<std::size_t>(n_nodes) &&
                    state.vol.size() == static_cast<std::size_t>(n_cells),
                "BCR rezone state field sizes are incomplete");
  TENRYU_ASSERT(state.x_r_reference.size() == state.x_r.size() &&
                    state.x_z_reference.size() == state.x_z.size() &&
                    state.cell_vol_initial.size() == state.vol.size(),
                "BCR rezone requires reference geometry storage");

  ++controller.bcr_rezone_invocations;
  std::vector<double> source_r;
  std::vector<double> source_z;
  state.x_r.copy_to_host(source_r);
  state.x_z.copy_to_host(source_z);
  const double hold_weight = bcr_hold_weight(controller, state.t);

  BcrCoMotionTarget target;
  if (controller.bcr_target) {
    TENRYU_ASSERT(state.v_r.size() == static_cast<std::size_t>(n_nodes) &&
                      state.v_z.size() == static_cast<std::size_t>(n_nodes),
                  "BCR target velocity field sizes are incomplete");
    std::vector<double> velocity_r;
    std::vector<double> velocity_z;
    state.v_r.copy_to_host(velocity_r);
    state.v_z.copy_to_host(velocity_z);
    target = build_bcr_co_motion_target(
        controller, source_r, source_z, velocity_r, velocity_z, state.dt,
        hold_weight);
    emit_bcr_target_fit(invocation_step, target);
  }
  const std::vector<BcrShapeCorner> corners =
      build_bcr_shape_corners(controller, n_nodes, target.scale,
                              source_r, source_z, hold_weight);
  const BcrMinimumJacobian jmin_before =
      bcr_patch_minimum_jacobian(corners, source_r, source_z);

  std::vector<double> optimized_r;
  std::vector<double> optimized_z;
  double phi0 = 0.0;
  bcr_relax_shape(controller, cfg, corners, target, source_r, source_z,
                  optimized_r, optimized_z, phi0);
  controller.bcr_feasible_seed_requested = false;
  controller.bcr_feasible_seed_cell = -1;

  std::vector<double> displacement_r(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> displacement_z(static_cast<std::size_t>(n_nodes), 0.0);
  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t index = static_cast<std::size_t>(node);
    displacement_r[index] = optimized_r[index] - source_r[index];
    displacement_z[index] = optimized_z[index] - source_z[index];
  }

  const auto& topology = *state.mesh.topo.multiblock;
  TENRYU_ASSERT(
      topology.cell_id_stable.size() == static_cast<std::size_t>(n_cells) &&
          topology.cell_orientation_sign.size() ==
              static_cast<std::size_t>(n_cells),
      "BCR rezone requires stable cell ids and orientation signs");
  const std::vector<std::uint8_t> cell_nverts =
      state.mesh.cell_nverts.empty()
          ? std::vector<std::uint8_t>(
                static_cast<std::size_t>(n_cells), 4U)
          : state.mesh.cell_nverts;
  TENRYU_ASSERT(cell_nverts.size() == static_cast<std::size_t>(n_cells),
                "BCR rezone cell_nverts size mismatch");
  core::DeviceArray<double> d_delta_r(static_cast<std::size_t>(n_nodes));
  core::DeviceArray<double> d_delta_z(static_cast<std::size_t>(n_nodes));
  core::DeviceArray<int> d_cell_id_stable(topology.cell_id_stable.size());
  core::DeviceArray<int> d_cell_orientation_sign(
      topology.cell_orientation_sign.size());
  core::DeviceArray<std::uint8_t> d_cell_nverts(cell_nverts.size());
  d_cell_id_stable.copy_from_host(topology.cell_id_stable);
  d_cell_orientation_sign.copy_from_host(topology.cell_orientation_sign);
  d_cell_nverts.copy_from_host(cell_nverts);

  mesh::CandidateMeshAdmissibilityFloors floors;
  floors.volume_rel = cfg.numerics.ale.reference_volume_floor_rel;
  floors.corner_j_rel = cfg.numerics.ale.reference_corner_j_floor_rel;
  floors.gauss_j_rel = cfg.numerics.ale.reference_gauss_j_floor_rel;

  std::vector<double> candidate_r(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> candidate_z(static_cast<std::size_t>(n_nodes), 0.0);
  mesh::CandidateMeshQuality candidate_quality{};
  bool candidate_accepted = false;
  int halvings = 0;
  double displacement_scale = 1.0;
  for (halvings = 0; halvings <= 4; ++halvings) {
    std::vector<double> scaled_delta_r(
        static_cast<std::size_t>(n_nodes), 0.0);
    std::vector<double> scaled_delta_z(
        static_cast<std::size_t>(n_nodes), 0.0);
    for (int node = 0; node < n_nodes; ++node) {
      const std::size_t index = static_cast<std::size_t>(node);
      scaled_delta_r[index] = displacement_scale * displacement_r[index];
      scaled_delta_z[index] = displacement_scale * displacement_z[index];
      candidate_r[index] = source_r[index] + scaled_delta_r[index];
      candidate_z[index] = source_z[index] + scaled_delta_z[index];
    }
    d_delta_r.copy_from_host(scaled_delta_r);
    d_delta_z.copy_from_host(scaled_delta_z);
    candidate_quality = mesh::evaluate_candidate_mesh_quality_csr(
        state.x_r.data(),
        state.x_z.data(),
        d_delta_r.data(),
        d_delta_z.data(),
        1.0,
        n_cells,
        state.corner_stride,
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        d_cell_id_stable.data(),
        d_cell_orientation_sign.data(),
        floors,
        d_cell_nverts.data(),
        nullptr,
        nullptr,
        0,
        state.x_r_reference.data(),
        state.x_z_reference.data());
    if (candidate_quality.admissible()) {
      candidate_accepted = true;
      break;
    }
    if (std::binary_search(controller.bcr_bf_cells.begin(),
                           controller.bcr_bf_cells.end(),
                           candidate_quality.first_bad_cell)) {
      core_clearance_controller_note_event(
          CoreClearanceControllerEventKind::BF_FIRST_REJECT);
    }
    displacement_scale *= 0.5;
  }
  if (halvings >= 4) {
    core_clearance_controller_note_event(
        CoreClearanceControllerEventKind::FOUR_HALVINGS);
  }

  const double det_floor = std::max(
      cfg.numerics.ale.reference_corner_j_floor_rel,
      std::numeric_limits<double>::epsilon());
  const std::vector<double> node_scale =
      bcr_initial_node_scales(controller, n_nodes);
  const double phi1 = bcr_rezone_objective(
      controller, corners, nullptr, -1, node_scale, target,
      candidate_r, candidate_z, det_floor);
  if (controller.bcr_continuous && controller.rank == 0 &&
      invocation_step % kBcrContinuousLogCadence == 0) {
    std::ostringstream line;
    line << std::scientific << std::setprecision(3)
         << "[bcr-continuous] step=" << invocation_step
         << " Phi0=" << phi0
         << " Phi1=" << phi1;
    core::log_info(line.str());
  }
  const BcrMinimumJacobian jmin_after =
      bcr_patch_minimum_jacobian(corners, candidate_r, candidate_z);
  double max_displacement = 0.0;
  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t index = static_cast<std::size_t>(node);
    max_displacement = std::max(
        max_displacement,
        std::hypot(candidate_r[index] - source_r[index],
                   candidate_z[index] - source_z[index]));
  }
  const std::array<int, 3> argmin_nodes = {
      jmin_before.node,
      jmin_before.previous_node,
      jmin_before.next_node,
  };
  std::array<double, 3> argmin_node_displacements{};
  std::array<int, 3> argmin_free{};
  const std::vector<std::vector<int>> touching_corners =
      build_bcr_touching_corners(corners, n_nodes);
  for (std::size_t slot = 0; slot < argmin_nodes.size(); ++slot) {
    const std::size_t index =
        static_cast<std::size_t>(argmin_nodes[slot]);
    argmin_node_displacements[slot] = std::hypot(
        candidate_r[index] - source_r[index],
        candidate_z[index] - source_z[index]);
    argmin_free[slot] = bcr_node_is_free(
                            controller, node_scale, touching_corners,
                            argmin_nodes[slot])
                            ? 1
                            : 0;
  }

  if (!candidate_accepted) {
    core::log_warning(
        "[bcr-rezone] LOUD WARNING step=" +
        std::to_string(invocation_step) +
        " candidate remained inadmissible after 4 halvings; rezone skipped "
        "first_bad_cell=" +
        std::to_string(candidate_quality.first_bad_cell));
    emit_bcr_rezone_step(invocation_step, phi0, phi1, jmin_before,
                         jmin_after, max_displacement,
                         argmin_node_displacements, argmin_free, 4, false);
    return;
  }

  if (g31_variant() != G31Variant::LEGACY) {
    const auto normalized_corner_min = [&](const std::vector<double>& r,
                                           const std::vector<double>& z) {
      double qmin = std::numeric_limits<double>::infinity();
      for (const BcrShapeCorner& corner : corners) {
        const std::size_t node = static_cast<std::size_t>(corner.node);
        const std::size_t previous =
            static_cast<std::size_t>(corner.previous_node);
        const std::size_t next =
            static_cast<std::size_t>(corner.next_node);
        const double next_r = r[next] - r[node];
        const double next_z = z[next] - z[node];
        const double previous_r = r[previous] - r[node];
        const double previous_z = z[previous] - z[node];
        const double jacobian =
            next_r * previous_z - next_z * previous_r;
        const double det_winv = corner.winv00 * corner.winv11 -
                                corner.winv01 * corner.winv10;
        qmin = std::min(qmin, jacobian * det_winv);
      }
      return qmin;
    };
    const double q_p_source = normalized_corner_min(source_r, source_z);
    const double q_p_candidate =
        normalized_corner_min(candidate_r, candidate_z);
    constexpr double kG31EpsP = 1.0e-3;
    const double phi_gain_rel =
        (phi0 - phi1) / std::max(std::abs(phi0), 1.0e-30);
    const bool certified =
        (q_p_candidate - q_p_source >= kBcrEtaQ) ||
        (q_p_candidate >= q_p_source - kG31EpsP &&
         phi_gain_rel >= kBcrEtaQ);

    const auto cell_min_corner_jacobian =
        [&](const int cell, const std::vector<double>& r,
            const std::vector<double>& z) {
          const int offset = controller.cell_node_offsets[
              static_cast<std::size_t>(cell)];
          const int nverts = active_nverts(controller, cell);
          const double orientation = static_cast<double>(
              topology.cell_orientation_sign[static_cast<std::size_t>(cell)]);
          double minimum = std::numeric_limits<double>::infinity();
          for (int corner = 0; corner < nverts; ++corner) {
            const int node = controller.cell_node_indices[
                static_cast<std::size_t>(offset + corner)];
            const int previous = controller.cell_node_indices[
                static_cast<std::size_t>(
                    offset + (corner + nverts - 1) % nverts)];
            const int next = controller.cell_node_indices[
                static_cast<std::size_t>(offset + (corner + 1) % nverts)];
            const double next_r = r[static_cast<std::size_t>(next)] -
                                  r[static_cast<std::size_t>(node)];
            const double next_z = z[static_cast<std::size_t>(next)] -
                                  z[static_cast<std::size_t>(node)];
            const double previous_r = r[static_cast<std::size_t>(previous)] -
                                      r[static_cast<std::size_t>(node)];
            const double previous_z = z[static_cast<std::size_t>(previous)] -
                                      z[static_cast<std::size_t>(node)];
            const double jacobian =
                orientation *
                (next_r * previous_z - next_z * previous_r);
            minimum = std::min(minimum, jacobian);
          }
          return minimum;
        };
    constexpr double kG31BfRelTol = 1.0e-3;
    int bf_failing_cell = -1;
    for (const int cell : controller.bcr_bf_cells) {
      const double source_min =
          cell_min_corner_jacobian(cell, source_r, source_z);
      const double candidate_min =
          cell_min_corner_jacobian(cell, candidate_r, candidate_z);
      const double floor = source_min > 0.0
                               ? (1.0 - kG31BfRelTol) * source_min
                               : source_min;
      if (candidate_min < floor) {
        bf_failing_cell = cell;
        break;
      }
    }

    static const double g31_displacement_cap =
        positive_env_value("TENRYU_I1B_G31_DISP_CAP", 0.0);
    static const double g31_kappaf_cap =
        positive_env_value("TENRYU_I1B_G31_KAPPAF_MAX", 0.0);
    double kappa_f = 0.0;
    for (const int cell : controller.bcr_bf_cells) {
      const int offset = controller.cell_node_offsets[
          static_cast<std::size_t>(cell)];
      const int nverts = active_nverts(controller, cell);
      for (int corner = 0; corner < nverts; ++corner) {
        const int node_i = controller.cell_node_indices[
            static_cast<std::size_t>(offset + corner)];
        const int node_j = controller.cell_node_indices[
            static_cast<std::size_t>(offset + (corner + 1) % nverts)];
        const std::size_t i = static_cast<std::size_t>(node_i);
        const std::size_t j = static_cast<std::size_t>(node_j);
        const double edge_length =
            std::hypot(source_r[i] - source_r[j],
                       source_z[i] - source_z[j]);
        if (edge_length == 0.0) {
          continue;
        }
        const double displacement_difference_r =
            (candidate_r[i] - source_r[i]) -
            (candidate_r[j] - source_r[j]);
        const double displacement_difference_z =
            (candidate_z[i] - source_z[i]) -
            (candidate_z[j] - source_z[j]);
        kappa_f = std::max(
            kappa_f,
            std::hypot(displacement_difference_r,
                       displacement_difference_z) /
                edge_length);
      }
    }

    const char* reject_reason = nullptr;
    if (!certified) {
      reject_reason = "cert";
    } else if (bf_failing_cell >= 0) {
      reject_reason = "bf_harm";
    } else if (g31_displacement_cap > 0.0 &&
               max_displacement > g31_displacement_cap) {
      reject_reason = "disp_cap";
    } else if (g31_kappaf_cap > 0.0 && kappa_f > g31_kappaf_cap) {
      reject_reason = "kappa_f";
    }

    if (controller.rank == 0) {
      std::ostringstream line;
      line << "[g31] step=" << invocation_step
           << " epoch=" << controller.g31_capture_epoch_id
           << " phaseA " << (reject_reason != nullptr ? "reject" : "pass");
      if (reject_reason != nullptr) {
        line << " reason=" << reject_reason;
      }
      line << std::scientific << std::setprecision(6)
           << " qP0=" << q_p_source
           << " qP1=" << q_p_candidate
           << std::setprecision(3)
           << " phiGain=" << phi_gain_rel
           << " bfCell=" << bf_failing_cell
           << " kappaF=" << kappa_f;
      if (reject_reason != nullptr) {
        core::log_warning(line.str());
      } else {
        core::log_info(line.str());
      }
    }
    if (reject_reason != nullptr) {
      return;
    }
  }

  const std::vector<double> target_volume =
      bcr_target_volumes(controller, state, candidate_r, candidate_z);
  std::vector<double> persistent_reference_r;
  std::vector<double> persistent_reference_z;
  std::vector<double> persistent_reference_volume;
  state.x_r_reference.copy_to_host(persistent_reference_r);
  state.x_z_reference.copy_to_host(persistent_reference_z);
  state.cell_vol_initial.copy_to_host(persistent_reference_volume);
  state.x_r_reference.copy_from_host(candidate_r.data());
  state.x_z_reference.copy_from_host(candidate_z.data());
  state.cell_vol_initial.copy_from_host(target_volume.data());

  core::Config remap_cfg = cfg;
  remap_cfg.numerics.ale.conservative_remap_enabled = true;
  remap_cfg.numerics.ale.conservative_remap_target = "reference";
  const ale::AleRemap2DRZResult remap_result =
      ale::ale_remap_2d_rz(state, remap_cfg, nullptr, 0.0);

  state.x_r_reference.copy_from_host(persistent_reference_r.data());
  state.x_z_reference.copy_from_host(persistent_reference_z.data());
  state.cell_vol_initial.copy_from_host(persistent_reference_volume.data());
  if (g31_variant() == G31Variant::LEGACY) {
    if (!remap_result.applied) {
      core::log_warning(
          "[bcr-rezone] LOUD WARNING step=" +
          std::to_string(invocation_step) +
          " same-time conservative remap was not applied; rezone skipped");
    } else {
      if (controller.bcr_phase == BcrPhase::CAPTURE) {
        transition_bcr_phase(controller, BcrPhase::RIDE,
                             controller.bcr_clock_step,
                             controller.bcr_clock_t);
      }
    }
  } else {
    const double closure_tol = [] {
      static const char* raw =
          std::getenv("TENRYU_I1B_REMAP_CLOSURE_REJECT_TOL");
      if (raw != nullptr && raw[0] != '\0') {
        const double x = std::atof(raw);
        return (std::isfinite(x) && x > 0.0) ? x : 0.0;
      }
      return -1.0;  // sentinel: fall back to cfg below
    }();
    const double tol =
        closure_tol >= 0.0
            ? closure_tol
            : cfg.numerics.ale.remap_mass_closure_reject_tol;
    const bool closure_ok =
        tol <= 0.0 || std::abs(remap_result.mass_closure_rel) <= tol;
    const bool phase_b_pass = remap_result.applied && closure_ok;
    if (phase_b_pass) {
      controller.g31_last_commit_t = state.t;
      controller.g31_last_commit_step = controller.bcr_clock_step;
      controller.g31_q_pre_commit = controller.g31_q_min;
      controller.g31_awaiting_post_sample = true;
      controller.g31_tau_cool = controller.bcr_tau_cell;
      if (controller.bcr_phase == BcrPhase::CAPTURE) {
        transition_bcr_phase(controller, BcrPhase::RIDE,
                             controller.bcr_clock_step,
                             controller.bcr_clock_t);
      }
      if (controller.rank == 0) {
        std::ostringstream line;
        line << std::scientific << std::setprecision(3)
             << "[g31] step=" << invocation_step
             << " epoch=" << controller.g31_capture_epoch_id
             << " phaseB commit closure=" << remap_result.mass_closure_rel;
        core::log_info(line.str());
      }
    } else if (remap_result.applied) {
      controller.g31_trial_reject_pending = true;
      if (controller.rank == 0) {
        std::ostringstream line;
        line << std::scientific << std::setprecision(3)
             << "[g31] step=" << invocation_step
             << " epoch=" << controller.g31_capture_epoch_id
             << " phaseB reject closure=" << remap_result.mass_closure_rel
             << " tol=" << tol
             << " restore=driver";
        core::log_warning(line.str());
      }
    } else {
      core::log_warning(
          "[bcr-rezone] LOUD WARNING step=" +
          std::to_string(invocation_step) +
          " same-time conservative remap was not applied; rezone skipped");
    }
  }
  emit_bcr_rezone_step(invocation_step, phi0, phi1, jmin_before,
                       jmin_after, max_displacement,
                       argmin_node_displacements, argmin_free, halvings,
                       remap_result.applied);
}

void core_clearance_controller_build_active_target(
    const core::State& state,
    const int step,
    const double t,
    const double dt,
    std::vector<double>& lagrangian_r,
    std::vector<double>& lagrangian_z,
    std::vector<double>& target_r,
    std::vector<double>& target_z) {
  CoreClearanceState& controller = core_clearance_state();
  TENRYU_ASSERT(controller.initialized && controller.active &&
                    !controller.multirank_unsupported && !controller.flushed,
                "clearance replay target requested without active controller");
  const std::size_t n_nodes = static_cast<std::size_t>(state.mesh.topo.n_nodes);
  TENRYU_ASSERT(controller.initial_node_radius.size() == n_nodes &&
                    controller.initial_director_r.size() == n_nodes &&
                    controller.initial_director_z.size() == n_nodes &&
                    controller.active_omega.size() == n_nodes,
                "clearance replay node geometry size changed");
  prepare_active_step(controller, state, step, t, dt);
  state.x_r.copy_to_host(lagrangian_r);
  state.x_z.copy_to_host(lagrangian_z);
  TENRYU_ASSERT(lagrangian_r.size() == n_nodes &&
                    lagrangian_z.size() == n_nodes,
                "clearance replay Lagrangian coordinate size changed");
  target_r = lagrangian_r;
  target_z = lagrangian_z;
  const double a_cmd = controller.r_cmd / controller.r0;
  for (std::size_t node = 0; node < n_nodes; ++node) {
    const double radius = controller.initial_node_radius[node];
    if (radius == 0.0) {
      target_r[node] = 0.0;
      target_z[node] = 0.0;
      continue;
    }
    const double omega = controller.active_omega[node];
    if (omega == 0.0) {
      continue;
    }
    const double director_r = controller.initial_director_r[node];
    const double director_z = controller.initial_director_z[node];
    const double projection =
        lagrangian_r[node] * director_r +
        lagrangian_z[node] * director_z;
    const double displacement =
        omega * (a_cmd * radius - projection);
    target_r[node] += displacement * director_r;
    target_z[node] += displacement * director_z;
  }
}

void core_clearance_controller_record_active_accept(
    const core::State& state,
    const int step,
    const double t,
    const double dt,
    const double sigma) {
  CoreClearanceState& controller = core_clearance_state();
  TENRYU_ASSERT(controller.initialized && controller.active &&
                    !controller.multirank_unsupported && !controller.flushed,
                "clearance replay acceptance requested without active controller");
  TENRYU_ASSERT(dt > 0.0 && std::isfinite(dt) &&
                    sigma >= 0.0 && sigma <= 1.0,
                "clearance replay acceptance metadata is invalid");
  std::vector<double> node_r;
  std::vector<double> node_z;
  state.x_r.copy_to_host(node_r);
  state.x_z.copy_to_host(node_z);
  TENRYU_ASSERT(node_r.size() == controller.initial_node_radius.size() &&
                    node_z.size() == controller.initial_node_radius.size(),
                "clearance replay accepted coordinate size changed");

  double numerator = 0.0;
  double denominator = 0.0;
  for (const int node : controller.controlled_nodes) {
    const std::size_t index = static_cast<std::size_t>(node);
    const double radius = controller.initial_node_radius[index];
    if (controller.active_omega[index] != 1.0 || radius == 0.0) {
      continue;
    }
    const double projection =
        node_r[index] * controller.initial_director_r[index] +
        node_z[index] * controller.initial_director_z[index];
    const double weight = radius * radius;
    numerator += weight * radius * projection;
    denominator += weight * radius * radius;
  }
  TENRYU_ASSERT(std::isfinite(numerator) &&
                    std::isfinite(denominator) && denominator > 0.0,
                "clearance replay accepted-radius fit is invalid");
  const double previous_r_acc = controller.r_acc;
  controller.r_acc = controller.r0 * numerator / denominator;
  TENRYU_ASSERT(std::isfinite(controller.r_acc) && controller.r_acc > 0.0,
                "clearance replay accepted radius is invalid");
  std::vector<double> accepted_volume;
  state.vol.copy_to_host(accepted_volume);
  refresh_active_rplus_snapshot(controller, accepted_volume);

  ClearanceSample& sample = controller.last_sample;
  sample.step = step;
  sample.t = t;
  sample.phase = controller.phase;
  sample.s_f = controller.active_s_end;
  sample.u_f = controller.active_u_half;
  sample.r_cmd = controller.r_cmd;
  sample.r_acc = controller.r_acc;
  sample.beta_eff =
      (previous_r_acc - controller.r_acc) /
      (controller.active_u_half * dt);
  sample.sigma = sigma;
  sample.g_guard = controller.active_g_guard;
  sample.h95 = controller.active_h95;

  if (!controller.support_exhausted &&
      controller.phase == ClearancePhase::TRACKING) {
    TENRYU_ASSERT(
        controller.active_s_end - controller.r_acc >=
            controller.active_g_guard,
        active_state_message("CLEARANCE_LOST", controller, sample));
    if (controller.beta == 0.0) {
      controller.low_beta_eff_steps = 0;
    } else {
      if (sample.beta_eff < 0.8 * controller.beta) {
        ++controller.low_beta_eff_steps;
      } else {
        controller.low_beta_eff_steps = 0;
      }
      TENRYU_ASSERT(
          controller.low_beta_eff_steps <= 20,
          active_state_message(
              "ADMISSIBILITY_CANNOT_TRACK_FRONT", controller, sample));
    }
  } else {
    controller.low_beta_eff_steps = 0;
  }

  if ((step % 5) == 0 || controller.phase != ClearancePhase::STATIC) {
    emit_active_sample(controller, sample);
  }
}

void core_clearance_controller_flush() {
  if (!core_clearance_active_requested() &&
      !core_clearance_shadow_enabled() && !bcr_sets_enabled()) {
    return;
  }
  CoreClearanceState& controller = core_clearance_state();
  if (!controller.initialized) {
    return;
  }
  if (controller.multirank_unsupported || controller.flushed) {
    return;
  }
  emit_adot_final(controller);
  emit_bcr_predictor_final(controller);
  emit_bcr_rezone_final(controller);
  if (controller.active) {
    if (controller.rank == 0) {
      std::ostringstream history;
      history << std::scientific << std::setprecision(9)
              << "[clearance-active] transition-history STATIC; armed at step="
              << controller.armed_step << ", t=" << controller.armed_t
              << "; tracking at step=" << controller.tracking_step
              << ", t=" << controller.tracking_t
              << "; support_exhausted="
              << (controller.support_exhausted ? 1 : 0);
      core::log_info(history.str());
    }
    controller.flushed = true;
    return;
  }
  if (!core_clearance_shadow_enabled()) {
    controller.flushed = true;
    return;
  }
  emit_sample(controller, controller.last_sample, true);
  if (controller.rank == 0) {
    std::ostringstream history;
    history << std::scientific << std::setprecision(9)
            << "[clearance] transition-history STATIC; armed at step="
            << controller.armed_step << ", t=" << controller.armed_t
            << "; tracking at step=" << controller.tracking_step
            << ", t=" << controller.tracking_t;
    core::log_info(history.str());
  }
  controller.flushed = true;
}

}  // namespace tenryu::hydro
