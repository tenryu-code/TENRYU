#pragma once

#include <cstdint>

#include "mesh/mesh_geometry_result.hpp"

namespace tenryu::mesh {

inline constexpr int kCandidateMacroCoreSentinelCell = -2;

struct Mesh;

struct CandidateMeshQuality {
  double min_rz_volume_rel = 1.0;
  double min_corner_j_rel = 1.0;
  double min_gauss_j_rel = 1.0;
  // Signed endpoint value used by the selected failure predicate: RZ volume
  // for a volume failure, or corner/center Jacobian for an area failure.
  double failing_value = 0.0;
  int first_bad_cell = -1;
  int first_bad_stable_cell = -1;
  int first_bad_corner = -1;
  MeshGeometryFailureKind kind = MeshGeometryFailureKind::None;

  bool admissible() const noexcept {
    return kind == MeshGeometryFailureKind::None;
  }
};

struct CandidateMeshAdmissibilityFloors {
  double volume_rel = 1.0e-8;
  double corner_j_rel = 1.0e-8;
  double gauss_j_rel = 1.0e-8;
  double volume_abs = 0.0;
  double corner_j_abs = 0.0;
  double gauss_j_abs = 0.0;
  // 2026-07-26 kernel-review hardening. All fields below default to "off" so the
  // legacy endpoint-only contract is unchanged unless a caller opts in.
  // §8.7/§8.8: also check the interior of the linear node path
  // [0, sigma] — corner Jacobians are quadratic and the revolved RZ volume
  // is cubic in the path parameter, so endpoint checks alone can miss a
  // transient inversion that recovers by the endpoint.
  bool continuous_path = false;
  // §8.6: reject any active node with candidate r < 0 (axis crossing);
  // r is linear along the path, so endpoint checks suffice.
  bool require_nonnegative_r = false;
  // §8.5: when >= 0, CSR node ids are bounds-checked against n_nodes and
  // CSR offsets against n_csr_indices before any coordinate dereference.
  int n_nodes = -1;
  int n_csr_indices = -1;
};

__device__ __host__ double rz_quad_volume_exact(
    double r0, double z0,
    double r1, double z1,
    double r2, double z2,
    double r3, double z3) noexcept;

__device__ __host__ double rz_triangle_volume_exact(
    double r0, double z0,
    double r1, double z1,
    double r2, double z2) noexcept;

// Host-callable re-exports of the file-local polygon primitives, added so
// the committed-mesh quality-min diagnostic (coupling/profile_observability)
// evaluates EXACTLY the gate's formulas (gate-matched diagnostic). Thin
// wrappers only — the gate computations remain shared with the device path.
void host_polygon_corner_jacobians(const double* r,
                                   const double* z,
                                   const int nverts,
                                   double* j);

double host_rz_polygon_volume_exact(const double* r,
                                    const double* z,
                                    const int nverts);

CandidateMeshQuality host_evaluate_candidate_mesh_quality_csr_cell(
    const double* node_r_old,
    const double* node_z_old,
    const double* node_r_reference,
    const double* node_z_reference,
    const double* delta_r,
    const double* delta_z,
    double sigma,
    int cell,
    int topology_stride,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const int* cell_id_stable,
    const int* cell_orientation_sign,
    const CandidateMeshAdmissibilityFloors& floors,
    const std::uint8_t* cell_nverts = nullptr,
    const std::uint8_t* inactive_cell_mask = nullptr,
    const std::uint8_t* reference_flat_corner = nullptr);

CandidateMeshQuality evaluate_candidate_mesh_quality(
    const double* d_node_r_old,
    const double* d_node_z_old,
    const double* d_delta_r,
    const double* d_delta_z,
    double sigma,
    int nr,
    int nz,
    const CandidateMeshAdmissibilityFloors& floors,
    const std::uint8_t* d_cell_nverts = nullptr,
    const double* d_node_r_reference = nullptr,
    const double* d_node_z_reference = nullptr);

CandidateMeshQuality evaluate_candidate_mesh_quality(
    const Mesh& mesh,
    const double* d_node_r_old,
    const double* d_node_z_old,
    const double* d_delta_r,
    const double* d_delta_z,
    double sigma,
    const CandidateMeshAdmissibilityFloors& floors,
    const double* d_node_r_reference = nullptr,
    const double* d_node_z_reference = nullptr);

CandidateMeshQuality evaluate_candidate_mesh_quality_csr(
    const double* d_node_r_old,
    const double* d_node_z_old,
    const double* d_delta_r,
    const double* d_delta_z,
    double sigma,
    int n_cells_total,
    int topology_stride,
    const int* d_cell_node_csr_offsets,
    const int* d_cell_node_csr_indices,
    const int* d_cell_id_stable,
    const CandidateMeshAdmissibilityFloors& floors,
    const std::uint8_t* d_cell_nverts = nullptr,
    const std::uint8_t* d_inactive_cell_mask = nullptr,
    const int* d_macro_boundary_nodes = nullptr,
    int n_macro_boundary_nodes = 0,
    const double* d_node_r_reference = nullptr,
    const double* d_node_z_reference = nullptr,
    const std::uint8_t* d_reference_flat_corner = nullptr);

CandidateMeshQuality evaluate_candidate_mesh_quality_csr(
    const double* d_node_r_old,
    const double* d_node_z_old,
    const double* d_delta_r,
    const double* d_delta_z,
    double sigma,
    int n_cells_total,
    int topology_stride,
    const int* d_cell_node_csr_offsets,
    const int* d_cell_node_csr_indices,
    const int* d_cell_id_stable,
    const int* d_cell_orientation_sign,
    const CandidateMeshAdmissibilityFloors& floors,
    const std::uint8_t* d_cell_nverts = nullptr,
    const std::uint8_t* d_inactive_cell_mask = nullptr,
    const int* d_macro_boundary_nodes = nullptr,
    int n_macro_boundary_nodes = 0,
    const double* d_node_r_reference = nullptr,
    const double* d_node_z_reference = nullptr,
    const std::uint8_t* d_reference_flat_corner = nullptr);

struct LineSearchResult {
  double sigma_accepted = 0.0;
  CandidateMeshQuality quality{};
  int iters_used = 0;
  int refine_iters_used = 0;
};

// Dyadic halving search: returns the first accepted sigma among
// sigma_init / 2^k (not the supremum). If halving drops below sigma_min the
// exact sigma_min is never evaluated and sigma_accepted = 0.0.
LineSearchResult linesearch_largest_admissible_sigma(
    const double* d_node_r_old,
    const double* d_node_z_old,
    const double* d_delta_r,
    const double* d_delta_z,
    double sigma_init,
    double sigma_min,
    int max_iters,
    int nr,
    int nz,
    const CandidateMeshAdmissibilityFloors& floors,
    const std::uint8_t* d_cell_nverts = nullptr,
    const double* d_node_r_reference = nullptr,
    const double* d_node_z_reference = nullptr);

LineSearchResult linesearch_largest_admissible_sigma(
    const Mesh& mesh,
    const double* d_node_r_old,
    const double* d_node_z_old,
    const double* d_delta_r,
    const double* d_delta_z,
    double sigma_init,
    double sigma_min,
    int max_iters,
    const CandidateMeshAdmissibilityFloors& floors,
    const double* d_node_r_reference = nullptr,
    const double* d_node_z_reference = nullptr);

LineSearchResult linesearch_largest_admissible_sigma_csr(
    const double* d_node_r_old,
    const double* d_node_z_old,
    const double* d_delta_r,
    const double* d_delta_z,
    double sigma_init,
    double sigma_min,
    int max_iters,
    int n_cells_total,
    int topology_stride,
    const int* d_cell_node_csr_offsets,
    const int* d_cell_node_csr_indices,
    const int* d_cell_id_stable,
    const CandidateMeshAdmissibilityFloors& floors,
    const std::uint8_t* d_cell_nverts = nullptr,
    const std::uint8_t* d_inactive_cell_mask = nullptr,
    const int* d_macro_boundary_nodes = nullptr,
    int n_macro_boundary_nodes = 0,
    const double* d_node_r_reference = nullptr,
    const double* d_node_z_reference = nullptr,
    int refine_iters = 0);

LineSearchResult linesearch_largest_admissible_sigma_csr(
    const double* d_node_r_old,
    const double* d_node_z_old,
    const double* d_delta_r,
    const double* d_delta_z,
    double sigma_init,
    double sigma_min,
    int max_iters,
    int n_cells_total,
    int topology_stride,
    const int* d_cell_node_csr_offsets,
    const int* d_cell_node_csr_indices,
    const int* d_cell_id_stable,
    const int* d_cell_orientation_sign,
    const CandidateMeshAdmissibilityFloors& floors,
    const std::uint8_t* d_cell_nverts = nullptr,
    const std::uint8_t* d_inactive_cell_mask = nullptr,
    const int* d_macro_boundary_nodes = nullptr,
    int n_macro_boundary_nodes = 0,
    const double* d_node_r_reference = nullptr,
    const double* d_node_z_reference = nullptr,
    int refine_iters = 0);

}  // namespace tenryu::mesh
