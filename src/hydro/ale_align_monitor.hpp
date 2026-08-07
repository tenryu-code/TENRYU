#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tenryu::core {
struct Config;
struct State;
}  // namespace tenryu::core

namespace tenryu::hydro::ale_align {

constexpr int kAngleHistogramBins = 90;

enum class LogicalMeshKind : std::uint8_t {
  RectangularRZ = 0,
  SphericalPolarHalfplane = 1,
  PolarInBox = 2,
};

enum class RegionClass : std::uint8_t {
  Bulk = 0,
  NearInterface = 1,
  VacuumMask = 2,
};

struct Params {
  double c_q_threshold = 0.2;
  double w_rho = 1.0;
  double w_p = 1.0;
  double floor_rel = 1.0e-12;
};

struct Input {
  int nr = 0;
  int nz = 0;
  LogicalMeshKind logical_mesh = LogicalMeshKind::RectangularRZ;
  int polar_prefix_nr = -1;
  int morph_rings = 0;
  int n_materials = 0;
  std::vector<double> rho;
  std::vector<double> pressure;
  std::vector<double> node_r;
  std::vector<double> node_z;
  std::vector<double> vol_frac;
  std::vector<std::uint8_t> vacuum_mask;
};

struct Tensor2 {
  double rr = 0.0;
  double rz = 0.0;
  double zz = 0.0;
};

struct WlsGradientResult {
  std::vector<double> grad_r;
  std::vector<double> grad_z;
  std::vector<std::uint8_t> available;
};

struct CellDiagnostic {
  double q_rho = 0.0;
  double q_p = 0.0;
  double grad_rho_r = 0.0;
  double grad_rho_z = 0.0;
  double grad_p_r = 0.0;
  double grad_p_z = 0.0;
  double S_rho = 0.0;
  double S_p = 0.0;
  Tensor2 Q;
  double lambda_1 = 0.0;
  double lambda_2 = 0.0;
  double coherence = 0.0;
  double director_r = 0.0;
  double director_z = 0.0;
  double e_A = 0.0;
  double angle_degrees = 0.0;
  double chi_topology = 1.0;
  RegionClass region = RegionClass::Bulk;
  bool interface_cell = false;
  bool near_interface = false;
  bool vacuum_masked = false;
  bool gradient_available = false;
  bool director_available = false;
  bool summary_included = false;
};

struct RegionSummary {
  std::size_t cell_count = 0;
  std::size_t masked_count = 0;
  std::size_t included_count = 0;
  std::array<std::uint64_t, kAngleHistogramBins> angle_histogram{};
  double angle_p50_degrees = 0.0;
  double angle_p90_degrees = 0.0;
  double angle_max_degrees = 0.0;
  double e_A_p50 = 0.0;
  double e_A_p90 = 0.0;
  double e_A_max = 0.0;
};

struct Summary {
  std::size_t total_cells = 0;
  std::size_t interface_masked_cells = 0;
  std::size_t vacuum_masked_cells = 0;
  std::size_t topology_masked_cells = 0;
  std::size_t wls_unavailable_cells = 0;
  std::size_t coherence_filtered_cells = 0;
  RegionSummary bulk;
  RegionSummary near_interface;
  RegionSummary vacuum_mask;
};

struct Result {
  bool valid = false;
  std::string error;
  double rho_floor = 0.0;
  double pressure_floor = 0.0;
  std::vector<CellDiagnostic> cells;
  Summary summary;
};

// Deterministic fixed-order WLS reconstruction for already transformed scalar
// values. Exposed so the host-only monitor algebra test can check exactness
// without coupling the assertion to the nonlinear log1p transform.
WlsGradientResult compute_one_ring_wls_gradients(
    int nr,
    int nz,
    const std::vector<double>& cell_r,
    const std::vector<double>& cell_z,
    const std::vector<double>& cell_length,
    const std::vector<double>& scalar,
    const std::vector<std::uint8_t>& interface_mask);

// Consult §5.3--§5.4 tensor construction for one cell. This helper is also
// used by the sign-independence algebra test.
Tensor2 build_structure_tensor(double grad_rho_r,
                               double grad_rho_z,
                               double grad_p_r,
                               double grad_p_z,
                               double cell_length,
                               const Params& params,
                               double* S_rho,
                               double* S_p);

Result compute_monitor(const Input& input, const Params& params);

// Stage 0 is deliberately log-only: run_info has no natural dynamic summary
// hook at the post-Lagrange ALE entry point. This adapter copies state to the
// host, never mutates simulation state, and emits at most one summary line.
void maybe_log_post_lagrange(const core::State& state,
                             const core::Config& cfg,
                             double dt_hydro_used,
                             int rank);

}  // namespace tenryu::hydro::ale_align
