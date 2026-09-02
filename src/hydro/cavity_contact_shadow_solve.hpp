#pragma once

#include <cstdint>
#include <vector>

namespace tenryu::hydro {

struct ShadowContactRow {
  std::int64_t constraint_id = 0;  // Canonical ordering key.
  int slave_node = -1;             // Mesh node ID for logs.
  std::int32_t segment_id = -1;    // Persistent SegmentID for logs.
  // Indices into the node-mass and velocity arrays.
  int node_s = -1;
  int node_a = -1;
  int node_b = -1;
  double xi = 0.0;
  double normal_r = 0.0;
  double normal_z = 0.0;
  double gap = 0.0;
  bool legacy_engaged = false;
};

struct ShadowContactSolveResult {
  std::vector<double> lambda;        // Per-row impulse magnitudes.
  std::vector<std::uint8_t> active;  // One for rows active at exit.
  int iterations = 0;
  int released_rows = 0;  // Rows removed by release or rank guards.
  bool singular = false;  // A deterministic rank guard tripped.
};

enum class GapTermMode { kFull, kNonNegative };

// Solves w = q + W lambda with unilateral complementarity. Rows must be in
// ascending constraint_id order on entry. GapTermMode::kNonNegative is the
// applied-impulse law: separated rows resist activation until contact, while
// penetrated rows receive pure velocity kill with no position correction.
ShadowContactSolveResult solve_shadow_contact_lcp(
    std::vector<ShadowContactRow>& rows,
    const std::vector<double>& node_mass,
    const std::vector<double>& v_r,
    const std::vector<double>& v_z,
    double dt,
    double gap_floor,
    GapTermMode gap_term_mode = GapTermMode::kFull);

}  // namespace tenryu::hydro
