#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "mesh/boundary_carrier.hpp"

// C1-w3b hydro constraint machinery over the persistent boundary carrier,
// implementing consult-29 §6 and docs/design/boundary_carrier_c1_20260810.md §4.
// This module is host-only, deterministic, and tunable-free; all loops are sequential
// in caller-provided order. v1 assumes one boundary component; multi-component decks
// will require one cyclic-tridiagonal block per component in a future wave.

namespace tenryu::hydro {

// View over one rezone epoch's boundary-slave records (parallel arrays owned by the
// caller; C1-w3a will move ownership into core::State). Slave i occupies mesh node
// node[i], lies on carrier edge edge[i] (edge k joins carrier masters k and
// (k+1) % n_masters), at cached interpolation parameter lambda[i], strictly in (0,1)
// (lambda == 0 / 1 are canonicalized to masters upstream, C1 contract §2).
struct CarrierSlaveView {
  const int* node = nullptr;
  const int* edge = nullptr;
  const double* lambda = nullptr;
  int count = 0;
};

// Condensed boundary-master system M_q qddot = f_q (consult-29 §6.2 boxed form,
// M_q = C^T M_b C, f_q = C^T f_b). v1 assumes the carrier is a SINGLE closed boundary
// component, so M_q is cyclic tridiagonal over the master loop: diag[k] is the (k,k)
// entry, off[k] couples masters k and (k+1) % n. r and z share the same M_q; masters
// with bc_class kAxis have their radial DOF removed at solve time (u_r == 0).
struct CondensedBoundarySystem {
  int n = 0;
  std::vector<double> diag;        // [n] condensed masses, g
  std::vector<double> off;         // [n] off-diagonal couplings, g
  std::vector<double> f_r;         // [n] condensed radial force, dyn
  std::vector<double> f_z;         // [n] condensed axial force, dyn
  std::vector<std::uint8_t> axis;  // [n] 1 iff masters[k].bc_class == kAxis
};

// Full O(node_count) structural validation of an epoch's slave view: every slave mesh
// node is in range, distinct from every master mesh node, and distinct from all other
// slave nodes. Call once per rezone epoch (after install); the per-call entry checks in
// the functions below are O(count) only.
void validate_carrier_slave_view(const mesh::BoundaryCarrier& carrier,
                                 const CarrierSlaveView& slaves,
                                 std::size_t node_count);

// (A) Fold slave nodal forces and masses onto their edge masters with weights
// (1-lambda, lambda) and assemble the condensed system. Pure read; deterministic
// (sequential accumulation in slave array order).
CondensedBoundarySystem condense_boundary_forces_and_masses(
    const mesh::BoundaryCarrier& carrier, const CarrierSlaveView& slaves,
    std::span<const double> node_mass, std::span<const double> force_r,
    std::span<const double> force_z);

// (B) Solve M_q qddot = f_q for master accelerations (cm/s²), r and z independently,
// via a deterministic bordered cyclic-tridiagonal LDL^T with certified positive pivots.
// Axis masters: the radial DOF is removed (row/column exactly eliminated with fixed
// value 0), so accel_r[k] == 0.0 for every axis master. accel spans have size n.
void solve_condensed_masters(const CondensedBoundarySystem& system,
                             std::span<double> accel_r, std::span<double> accel_z);

// (C) Overwrite every slave's position and velocity by the kinematic reconstruction
// X_s = (1-lambda)*X_A + lambda*X_B, u_s = (1-lambda)*u_A + lambda*u_B (double
// arithmetic on the cached lambda; A = edge start master, B = edge end master). Slaves
// depend only on masters, never on other slaves, so the pass is order-independent;
// it is executed in ascending slave index for determinism. On an axis edge (BOTH
// endpoint masters kAxis) the slave radial position is canonicalized to +0.0 (the
// masters are asserted to sit exactly at r == 0); radial velocity is plain
// interpolation (masters carry u_r == 0 whenever the axis BC has been applied).
void reconstruct_slaves(const mesh::BoundaryCarrier& carrier,
                        const CarrierSlaveView& slaves, std::span<double> pos_r,
                        std::span<double> pos_z, std::span<double> vel_r,
                        std::span<double> vel_z);

}  // namespace tenryu::hydro
