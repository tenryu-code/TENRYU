#pragma once

// Linear-discontinuous S_N in 1D (Radiation.sn_transport.spatial_scheme =
// "linear_discontinuous"; NUMERICS 6.8.4). Every cell carries two nodal values
// (left and right face) of the angular intensity, of the material electron
// temperature and of the emission: basis b_L = (r_R - r)/h, b_R = (r - r_L)/h,
// lumped masses M_j = int b_j A dr with the geometry's face area A(r) (4 pi r^2,
// 2 pi r per unit length, 1 per unit area), the conservative streaming
// operator with upwind face traces, and for the sphere and the cylinder the
// existing weighted-diamond angular closure (alpha, tau) with a starting
// direction per angular chain solved as a planar linear-discontinuous
// transport along the chain's diameter.

#include <cstddef>

#include <cuda_runtime.h>

#include "core/config.hpp"

namespace tenryu::core {
struct State;
}  // namespace tenryu::core

namespace tenryu::radiation {
class PlanckTable;
}  // namespace tenryu::radiation

namespace tenryu::radiation::sn_ld {

// Per-cell geometry, 7 doubles per cell: h, M_L, M_R, N_L, N_R, A_L, A_R
// (N_j = int b_j A'(r) dr, the angular redistribution weight; A_L, A_R the
// face areas).
constexpr int kCellStride = 7;

// Angular quadrature of the sweep: ordinates in chains of chain_len ordinates
// with ascending mu (the first half inward, mu < 0); alpha has one entry per
// ordinate edge with 0 at both ends of every chain (the code's Carlson
// recursion alpha_{m+1/2} = alpha_{m-1/2} - mu_m w_m), tau the
// weighted-diamond factors, sd_mu the starting-direction cosine magnitude of
// every chain (nullptr: 1).
struct QuadratureView {
  const double* mu = nullptr;
  const double* weight = nullptr;
  const double* alpha = nullptr;
  const double* tau = nullptr;
  const double* sd_mu = nullptr;
  int n_angles = 0;
  int n_chains = 1;
  int chain_len = 0;
};

// cells[kCellStride * n] from the node radii x_r[n + 1].
void compute_cells(const double* x_r, int n_cells, int geom, double* cells,
                   cudaStream_t stream = nullptr);

// One transport sweep of every group (block per group). sigma_t[c * G + g]
// is the total cross section without the time term; q[g * 2n + 2c + j] the
// nodal isotropic per-angle source; psi_prev[(g * N + m) * 2n + 2c + j] and
// sd_prev[(g * n_chains + k) * 2n + 2c + j] the histories (nullptr: none);
// psi_in[g] the incoming intensity at the outer face (nullptr: vacuum).
// Writes psi (same layout as psi_prev) and, for curved geometries, sd.
void sweep(const double* cells, const double* sigma_t, const double* q,
           const double* psi_prev, const double* sd_prev, const double* psi_in,
           const QuadratureView& quad, double* psi, double* sd, int n_cells,
           int n_groups, double inv_cdt, int geom, cudaStream_t stream = nullptr);

// sweep in two parts for many sweeps with the same cross sections:
// sweep_prepare inverts every ordinate's and starting direction's cell
// matrix into inverse (sweep_inverse_doubles doubles); sweep_prepared
// sweeps with them (the values of sweep).
std::size_t sweep_inverse_doubles(int n_cells, int n_groups, const QuadratureView& quad,
                                  int geom);
void sweep_prepare(const double* cells, const double* sigma_t, const QuadratureView& quad,
                   double* inverse, int n_cells, int n_groups, double inv_cdt, int geom,
                   cudaStream_t stream = nullptr);
void sweep_prepared(const double* cells, const double* inverse, const double* q,
                    const double* psi_prev, const double* sd_prev, const double* psi_in,
                    const QuadratureView& quad, double* psi, double* sd, int n_cells,
                    int n_groups, double inv_cdt, int geom, cudaStream_t stream = nullptr);

// Nodal scalar flux phi[g * 2n + 2c + j] = sum_m w_m psi, face currents
// F[g * (n + 1) + f] = sum_m w_m mu_m psi_upwind(f) (F = 0 at the centre /
// axis / reflecting slab face), and the cell-average second moment
// Prr[c * G + g] = sum_m w_m mu_m^2 psi averaged with the lumped masses
// (nullptr: not computed).
void moments(const double* cells, const double* psi, const double* psi_in,
             const QuadratureView& quad, double* phi, double* face_current,
             double* prr, int n_cells, int n_groups, cudaStream_t stream = nullptr);

// Consistent P1 low-order system of the sweep (the diffusion-synthetic
// acceleration of the scattering source iteration and the grey preconditioner
// of the emission coupling): the P1 ansatz psi_m = a Phi + b mu_m J
// (a = 1 / sum w, b = 1 / sum w mu^2) at both nodes of every cell, inserted in
// the discrete equations of every ordinate (upwind face traces, the angular
// recursion started from the starting-direction ansatz a Phi - b s J), zeroth
// and first angular moments per node: four unknowns per cell
// (Phi_L, Phi_R, J_L, J_R), block tridiagonal. sigma_e[c * S + s] is the total
// cross section of system s including the time term, sigma_x[c * S + s] the
// part returned to the zeroth moment (the scattering), rhs[s * 2n + 2c + j]
// the nodal source of the zeroth-moment rows (divided by the lumped mass).
// Solves every system (one per s) and writes the nodal Phi into x[s * 2n +
// 2c + j]. scratch: at least p1_scratch_doubles(n_cells, n_systems) doubles.
std::size_t p1_scratch_doubles(int n_cells, int n_systems);
void p1_solve(const double* cells, const double* sigma_e, const double* sigma_x,
              const double* rhs, const QuadratureView& quad, double* x, double* scratch,
              int n_cells, int n_systems, int geom, cudaStream_t stream = nullptr);
// p1_solve in two parts for many right-hand sides of the same systems:
// p1_factor assembles every system and eliminates it block by block, each
// diagonal block stored as its inverse, into factors (at least
// p1_scratch_doubles(n_cells, n_systems) doubles); p1_apply solves for one
// right-hand side of every system with those factors (the values of
// p1_solve).
void p1_factor(const double* cells, const double* sigma_e, const double* sigma_x,
               const QuadratureView& quad, double* factors, int n_cells, int n_systems,
               int geom, cudaStream_t stream = nullptr);
void p1_apply(double* factors, const double* rhs, double* x, int n_cells, int n_systems,
              cudaStream_t stream = nullptr);

// One radiation step of the linear-discontinuous scheme (NUMERICS 6.8): the
// emission linearized about the nodal electron temperatures (Newton on the
// temperature; the Planck function's derivative includes db_g/dT), the
// emission coupling solved by GMRES on the nodal absorption-rate density
// sum_g sigma_a,g phi_g (each operator application a sweep of every group,
// with the scattering converged by source iteration and the consistent P1
// acceleration), the nodal electron energies updated conservatively from the
// transport's own absorption and emission, the in-cell energy offset and the
// angular histories carried to the next step. outer_psi_in[g] is the
// incoming intensity at the outer face (nullptr: vacuum), source_ext[c] the
// external volume source of group 0 (nullptr: none).
void advance_step(core::State& state, const core::Config& cfg, const PlanckTable& planck,
                  const core::Config::MaterialsConfig::MatDef& mat, double dt,
                  const double* outer_psi_in, const double* source_ext);

}  // namespace tenryu::radiation::sn_ld
