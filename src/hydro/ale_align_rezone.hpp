#pragma once

#include <string>
#include <vector>

#include "hydro/ale_align_monitor.hpp"

namespace tenryu::hydro::ale_align {

enum class RezoneGeometry {
  Planar,
  Rz,
};

struct RezoneParams {
  RezoneGeometry geometry = RezoneGeometry::Planar;
  int iterations = 40;
  int max_backtracks = 8;
  double alpha = 0.3;
  double kappa_gain = 2.0;
  double omega_eta = 0.35;
  double epsilon_j = 0.05;
  double epsilon_L = 1.0e-30;
  double fd_relative_step = 1.0e-6;
};

struct RezoneInput {
  int nr = 0;
  int nz = 0;
  std::vector<double> node_r;
  std::vector<double> node_z;
  std::vector<CellDiagnostic> monitor;
};

struct RezoneResult {
  bool valid = false;
  std::string error;
  std::vector<double> node_r;
  std::vector<double> node_z;
  double initial_energy = 0.0;
  double final_energy = 0.0;
  // Final exact per-cell values in RZ mode; empty in planar mode.
  std::vector<double> cell_rz_volumes;
  int accepted_iterations = 0;
  int backtrack_attempts = 0;
  int frozen_retries = 0;
  int skipped_iterations = 0;
};

// Stage-2 host-only direction-control energy. The monitor is cellwise frozen,
// and reference/quality terms are absent. RZ mode includes the exact 2*pi*r
// quadrature weight.
double evaluate_planar_rezone_energy(const RezoneInput& input,
                                     const RezoneParams& params);

// Deterministic Stage-2 prototype. Boundary nodes slide on their initial
// straight boundary lines, RZ axis nodes remain at r=0, corners remain fixed,
// and no ALE state is mutated.
RezoneResult solve_planar_rezone(const RezoneInput& input,
                                 const RezoneParams& params = {});

}  // namespace tenryu::hydro::ale_align
