#pragma once

#include <optional>
#include <string>
#include <vector>

#include "mesh/tessellation/delaunay_2d.hpp"
#include "mesh/voronoi_rz.cuh"

namespace tenryu::mesh::voronoi {

struct GeneratorFlowSample {
  double v_r = 0.0;
  double v_z = 0.0;
};

struct TargetResult {
  std::vector<Generator> targets;
  Tessellation accepted_tessellation;
  // Advection blend followed by one accepted blend per Lloyd sweep.
  std::vector<double> sweep_blend;
  double target_generation_ms = 0.0;
  double tessellation_ms = 0.0;
  double legacy_assembly_ms = 0.0;
  bool valid = false;
  std::string reject_reason;
  int cvt_iterations_used = 0;
  // d1 certified skip telemetry (docs/design/
  // t2_d1_certified_lloyd_skip_20260815.md): set when the certified
  // predicate-margin exit ended the Lloyd loop, with the iteration index.
  bool lloyd_d1_fired = false;
  int lloyd_d1_iteration = -1;
};

// Lloyd iteration ceiling for build_generator_targets. Configuration
// plumbing mirrors mesh::tess::set_tess_gpu_dual_enabled: reale_mode
// installs Numerics.ale.reale_lloyd_max before each rezone. Default
// matches the historical kLloydMax = 4.
void set_reale_lloyd_max(int lloyd_max);

// Lexicographic target construction for one rezone cycle. Priority 1 is
// admissibility, the Stage-2 priority-3 proxy is a placeholder, and priority 4
// is compactness. Compactness uses up to reale_lloyd_max Lloyd sweeps (default 4) with the planar-area
// centroid of each Voronoi polygon; no weighted objective is used. Axis
// generators are identified by exact r == 0.0 in the input and remain exactly
// on the axis. The caller-sampled flow corresponds elementwise to generators.
TargetResult build_generator_targets(
    const std::vector<Generator>& generators,
    const std::vector<GeneratorFlowSample>& flow,
    const DomainBoundary& domain,
    const double dt,
    const bool exact_core_enabled = true,
    std::optional<mesh::tess::DelaunayTriangulation>* warm_dt = nullptr);

}  // namespace tenryu::mesh::voronoi
