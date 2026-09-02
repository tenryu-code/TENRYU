#pragma once

// Mass-orthogonal constrained projection of remapped boundary velocities onto
// the carrier constraint space, per consult-29 response §11
// (~/ai-consult/20260809-2254-ale-p2-consult29/response.md) and the C1 contract
// docs/design/boundary_carrier_c1_20260810.md. This is standalone C2 machinery
// with no dependence on State/StagedMesh; the parent wave wires it into the
// ReALE epoch.

#include <string>
#include <vector>

namespace tenryu::hydro {

// Boundary-node system after a conservative nodal remap. Node-indexed arrays have
// size n_nodes_total and are indexed by the mesh node ids stored in master_node /
// slave_node; entries of non-boundary nodes are ignored.
struct CarrierProjectionInput {
  int n_masters = 0;                            // N_B >= 3; carrier edge k = (k, (k+1) % N_B)
  const int* master_node = nullptr;             // size n_masters
  const unsigned char* master_fix_r = nullptr;  // size n_masters; 1 = axis master (u_r fixed to 0); may be null (= all free)
  int n_slaves = 0;                             // >= 0
  const int* slave_node = nullptr;              // size n_slaves (null allowed iff n_slaves == 0)
  const int* slave_edge = nullptr;              // size n_slaves; values in [0, n_masters)
  const double* slave_lambda = nullptr;         // size n_slaves; cached double lambda in OPEN (0, 1)
  int n_nodes_total = 0;                        // > 0
  const double* node_mass = nullptr;            // M* diagonal (g); > 0 and finite on every boundary node
  const double* node_pr = nullptr;              // p*_r (g cm/s)
  const double* node_pz = nullptr;              // p*_z
};

struct CarrierProjectionResult {
  bool ok = false;
  std::string reject_reason;                    // empty on success
  std::vector<double> master_qdot_r;            // size n_masters on success
  std::vector<double> master_qdot_z;
  double delta_k = 0.0;                         // (1/2)(u*-u)^T M* (u*-u) >= 0 (erg)
  double momentum_in_r = 0.0;                   // sum of p* over boundary nodes
  double momentum_in_z = 0.0;
  double momentum_out_r = 0.0;                  // sum of m u^{n+1} over boundary nodes
  double momentum_out_z = 0.0;
  int n_fixed_radial_dofs = 0;                  // number of masters with u_r Dirichlet-fixed
};

// Solves (C^T M* C) qdot = C^T p* per velocity component (r and z independently) on
// the cyclic master loop, writes u^{n+1} = C qdot into out_vr/out_vz (ONLY the
// entries of boundary nodes = masters + slaves are written; both arrays have size
// n_nodes_total), and reports conservation data in the result instead of asserting:
// momentum_out == momentum_in holds for every component without fixed dofs (z always
// in the current bc set; r only when n_fixed_radial_dofs == 0 — with axis masters the
// difference is the physical axis constraint impulse). delta_k is computed in the
// guaranteed-nonnegative difference form. On rejection (ok == false) out arrays are
// untouched and the qdot vectors are empty.
CarrierProjectionResult project_carrier_boundary_velocity(
    const CarrierProjectionInput& in,
    double* out_vr,
    double* out_vz);

struct CarrierDefectAllocation {
  bool ok = false;
  std::string reject_reason;
  double total = 0.0;                           // sum over cells of delta_e_cell
};

// Distributes the projection kinetic-energy defect to cells with corner-mass weights
// (consult-29 response §11 closing formula): DeltaE_c = sum_{a in c} (m_ac / m_a) *
// (1/2) m_a |u*_a - u^{n+1}_a|^2, implemented as the algebraically identical
// (1/2) m_ac |u*_a - u^{n+1}_a|^2 per corner. Corners whose node is not a boundary
// node of `in` contribute exactly zero. sum_c DeltaE_c == delta_k holds when the
// caller's corner masses sum to the node masses over boundary nodes (returned in
// `total` for the caller's cross-check; not enforced here). delta_e_cell (size
// n_cells) is OVERWRITTEN (zero-initialized, then accumulated in corner array order).
CarrierDefectAllocation allocate_projection_defect_to_cells(
    const CarrierProjectionInput& in,
    const double* node_vr_new,
    const double* node_vz_new,
    int n_corners,
    const int* corner_cell,                     // size n_corners; values in [0, n_cells)
    const int* corner_node,                     // size n_corners; values in [0, in.n_nodes_total)
    const double* corner_mass,                  // size n_corners; >= 0 and finite
    int n_cells,
    double* delta_e_cell);

}  // namespace tenryu::hydro
