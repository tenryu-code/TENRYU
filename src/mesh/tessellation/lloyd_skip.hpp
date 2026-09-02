#pragma once

#include <cstddef>
#include <vector>

#include "mesh/tessellation/delaunay_2d.hpp"

namespace tenryu::mesh::tess {

// d1 certified Lloyd-skip certifier (docs/design/
// t2_d1_certified_lloyd_skip_20260815.md). Decides, in pure double
// arithmetic, whether a per-site displacement field provably cannot flip
// the sign of any exact predicate certifying the triangulation: triangle
// orientations, interior-edge incircle tests, and hull convexity turns.
// Conservative by construction: any refusal is safe (the caller simply
// does not skip).
struct LloydSkipReport {
  bool certified = false;
  std::size_t certificates_checked = 0;
  std::size_t orient_checked = 0;
  std::size_t incircle_checked = 0;
  std::size_t hull_checked = 0;
  // First refusal class, nullptr when certified. One of:
  // "degenerate_structure", "slack_nonpositive_orient",
  // "slack_nonpositive_incircle", "slack_nonpositive_hull",
  // "margin_exceeded_orient", "margin_exceeded_incircle",
  // "margin_exceeded_hull".
  const char* refusal = nullptr;
};

// delta_r/delta_z are per-site nonnegative displacement magnitudes,
// indexed like dt.sites. Early-exits on the first failed certificate.
LloydSkipReport certify_lloyd_noop(const DelaunayTriangulation& dt,
                                   const std::vector<double>& delta_r,
                                   const std::vector<double>& delta_z);

}  // namespace tenryu::mesh::tess
