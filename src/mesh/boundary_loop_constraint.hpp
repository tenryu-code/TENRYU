#pragma once

#include <vector>

namespace tenryu::mesh {

// Low-mode constraint basis for a closed boundary loop (macro-boundary
// terminal-phase verdict, docs/design/external-ai-responses/
// 20260611-i1b-macro-boundary-endgame-verdict.md §2, basis option 2:
// boundary-loop graph Laplacian low modes).
//
// For a closed loop of N nodes the cyclic graph Laplacian eigenvectors are
// analytic: the constant mode plus the cos/sin(2*pi*k*l/N) pair per
// harmonic k = 1..L, i.e. the 2L+1 lowest scalar modes. The same scalar
// basis is applied independently to the r and z dof families, so the
// tangent matrix T = dX/da has 2N rows and 2*(2L+1) columns. The raw
// trigonometric basis is deliberately NOT mass-orthonormalized; the small
// Gram matrix G = T^T M_B T absorbs the normalization (verdict §2).
//
// Fixed layout contracts:
// - Dof rows are interleaved like the verdict's
//   x_B = (r_1, z_1, ..., r_N, z_N)^T: row 2*l is the r-dof and row 2*l+1
//   the z-dof of loop position l. Position l follows the caller's ORDERED
//   loop-node sequence (N distinct nodes; cyclic adjacency implied, no
//   duplicated closing node). Mapping positions to mesh node ids is the
//   caller's wiring concern, not this kernel's.
// - Columns: column m in [0, 2L+1) applies scalar mode m to the r-dofs
//   (z rows zero); column (2L+1)+m applies the same scalar mode to the
//   z-dofs. Scalar mode order: constant, then per harmonic k: cos, sin.
struct BoundaryLoopLowModeBasis {
  int n_nodes = 0;      // N (loop length)
  int n_harmonics = 0;  // L (highest retained harmonic)
  // Row-major (2N) x (2*(2L+1)) tangent matrix T.
  std::vector<double> tangent;

  int n_scalar_modes() const { return 2 * n_harmonics + 1; }
  int n_modes() const { return 2 * n_scalar_modes(); }
  int n_dofs() const { return 2 * n_nodes; }
};

// Builds the basis. Requirements: n_nodes >= 3 (a loop), n_harmonics >= 0,
// and 2*n_harmonics + 1 <= n_nodes (rank/aliasing guard; k <= L < N/2
// keeps every retained sin mode nonzero). Throws std::invalid_argument.
BoundaryLoopLowModeBasis build_boundary_loop_low_mode_basis(int n_nodes,
                                                            int n_harmonics);

struct BoundaryLoopConstrainedForce {
  // du_B/dt = T G^{-1} T^T F  (size 2N, interleaved like the basis rows)
  std::vector<double> accel;
  // R_B = M_B (du_B/dt) - F   (size 2N); T^T R_B = 0 in exact arithmetic,
  // so any constrained velocity u_B = T (da/dt) does zero work against it.
  std::vector<double> reaction;
};

// Zero-work constrained projection (verdict §2): with the diagonal nodal
// mass matrix M_B and the assembled boundary-node hydrodynamic force
// F_B^H, the generalized acceleration solves G d2a/dt2 = T^T F_B^H with
// G = T^T M_B T (small dense SPD; factorized by Cholesky), giving
// du_B/dt = T G^{-1} T^T F_B^H and the ideal constraint reaction R_B.
//
// node_mass: size N, strictly positive and finite; node_mass[l] weights
// BOTH dofs (rows 2l and 2l+1) of loop position l.
// force: size 2N, interleaved like the basis rows.
// Throws std::invalid_argument on size/positivity violations and
// std::runtime_error if G is not numerically SPD (cannot happen for a
// valid basis with positive masses; guards memory corruption).
BoundaryLoopConstrainedForce project_force_to_constrained_accel(
    const BoundaryLoopLowModeBasis& basis,
    const std::vector<double>& node_mass,
    const std::vector<double>& force);

}  // namespace tenryu::mesh
