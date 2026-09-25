#include "radiation/sn_ld_1d_gpu.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "core/constants.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "materials/eos_device_table.hpp"
#include "mesh/geometry_1d.cuh"
#include "radiation/material_electron_eos_1d.hpp"
#include "radiation/planck_table.cuh"
#include "radiation/sn_electron_eos.cuh"
#include "radiation/sn_material_newton_gpu.cuh"
#include "radiation/sn_transport_1d_internal.hpp"

namespace tenryu::radiation::sn_ld {
namespace {

constexpr int kWarp = 32;
constexpr int kBlock = 256;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, std::string(message) + ": " + cudaGetErrorString(err));
}

__host__ __device__ inline double finite_or_zero(const double v) {
  return isfinite(v) ? v : 0.0;
}

// d A / d r of the geometry's face area.
__host__ __device__ inline double face_area_derivative(const int geom, const double r) {
  if (geom == static_cast<int>(mesh::Geometry1D::kCylindrical)) {
    return 2.0 * mesh::geometry_1d_detail::kPi;
  }
  if (geom == static_cast<int>(mesh::Geometry1D::kPlanar)) {
    return 0.0;
  }
  return 2.0 * mesh::geometry_1d_detail::kFourPi * r;
}

// Two-point Gauss-Legendre on [r_L, r_R] integrates b_j A (degree 3) and
// b_j A' (degree 2) exactly without the cancellation of the closed forms.
__global__ void compute_cells_kernel(const double* __restrict__ x_r, const int n,
                                     const int geom, double* __restrict__ cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) {
    return;
  }
  const double rl = x_r[c];
  const double rr = x_r[c + 1];
  const double h = rr - rl;
  const double mid = 0.5 * (rl + rr);
  const double half = 0.5 * h;
  constexpr double kNode = 0.57735026918962576450914878050196;  // 1/sqrt(3)
  double ML = 0.0, MR = 0.0, NL = 0.0, NR = 0.0;
  for (int q = 0; q < 2; ++q) {
    const double r = mid + ((q == 0) ? -kNode : kNode) * half;
    const double bL = (rr - r) / h;
    const double bR = (r - rl) / h;
    const double A = mesh::geometry_1d_face_area(geom, r);
    const double dA = face_area_derivative(geom, r);
    ML += half * bL * A;
    MR += half * bR * A;
    NL += half * bL * dA;
    NR += half * bR * dA;
  }
  double* out = cells + static_cast<std::size_t>(kCellStride) * c;
  out[0] = h;
  out[1] = ML;
  out[2] = MR;
  out[3] = NL;
  out[4] = NR;
  out[5] = mesh::geometry_1d_face_area(geom, rl);
  out[6] = mesh::geometry_1d_face_area(geom, rr);
}

// The inverse of the 2 x 2 cell matrix of every ordinate and group (the
// part of the cell equations that does not depend on the sweep's recursion),
// kInvStride doubles per (group, ordinate, cell): inv00, inv01, inv10, inv11.
constexpr int kInvStride = 4;

// The lumped linear-discontinuous equations of one ordinate in one cell
// (NUMERICS 6.8): -mu G psi + mu (A_R psi_R^ - A_L psi_L^) + sigma M psi
// + (alpha_+ e_+ - alpha_- e_-) N / w = M (q + psi_prev / (c dt)), with
// G = (1/h) [[-M_L, -M_R], [M_L, M_R]], the upwind face traces and the
// weighted-diamond edge psi = tau e_+ + (1 - tau) e_-. The matrix on the
// cell's own nodes (the outflow face's own node, the edge's own-ordinate
// part alpha_+ / (w tau)) is inverted here; the right-hand side (the
// source, the inflow trace and the previous ordinate's edge) is formed in
// the sweep.
__device__ inline void cell_matrix_inverse(const double* __restrict__ cell, const double mu,
                                           const double w, const double a_plus, const double tau,
                                           const double sigma_eff, const bool curved,
                                           double* __restrict__ inv) {
  const double h = cell[0];
  const double ML = cell[1];
  const double MR = cell[2];
  const double NL = cell[3];
  const double NR = cell[4];
  const double AL = cell[5];
  const double AR = cell[6];
  const double s = mu / h;
  double a00 = s * ML + sigma_eff * ML;
  const double a01 = s * MR;
  const double a10 = -s * ML;
  double a11 = -s * MR + sigma_eff * MR;
  if (mu > 0.0) {
    a11 += mu * AR;
  } else {
    a00 -= mu * AL;
  }
  if (curved) {
    const double cang = a_plus / (w * tau);
    a00 += cang * NL;
    a11 += cang * NR;
  }
  const double det = a00 * a11 - a01 * a10;
  inv[0] = a11 / det;
  inv[1] = -a01 / det;
  inv[2] = -a10 / det;
  inv[3] = a00 / det;
}

// The inverses of every ordinate's cell matrices, and of the starting
// directions' (the planar linear-discontinuous cell of -s dpsi/dr + sigma
// psi with weight dr: [[s/2 + sigma h/2, -s/2], [s/2, s/2 + sigma h/2]]),
// one thread per (group, ordinate or chain, cell). They depend on the cross
// sections and the mesh only, so every sweep with the same cross sections
// uses them.
__global__ void sweep_matrix_kernel(const double* __restrict__ cells,
                                    const double* __restrict__ sigma_t,
                                    const double* __restrict__ mu,
                                    const double* __restrict__ weight,
                                    const double* __restrict__ alpha,
                                    const double* __restrict__ tau,
                                    const double* __restrict__ sd_mu,
                                    double* __restrict__ inv,
                                    double* __restrict__ sd_inv,
                                    const int n,
                                    const int G,
                                    const int N,
                                    const int n_chains,
                                    const int curved,
                                    const double inv_cdt) {
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long n_ord = static_cast<long long>(G) * N * n;
  const long long n_sd = (curved != 0) ? static_cast<long long>(G) * n_chains * n : 0;
  if (idx >= n_ord + n_sd) {
    return;
  }
  if (idx < n_ord) {
    const int c = static_cast<int>(idx % n);
    const long long gm = idx / n;
    const int m = static_cast<int>(gm % N);
    const int g = static_cast<int>(gm / N);
    const double se =
        fmax(finite_or_zero(sigma_t[static_cast<std::size_t>(c) * G + g]), 0.0) + inv_cdt;
    const double tw = fmin(fmax(tau[m], 1.0e-6), 1.0);
    cell_matrix_inverse(cells + static_cast<std::size_t>(kCellStride) * c, mu[m], weight[m],
                        alpha[m + 1], tw, se, curved != 0,
                        inv + static_cast<std::size_t>(idx) * kInvStride);
    return;
  }
  const long long j = idx - n_ord;
  const int c = static_cast<int>(j % n);
  const long long gk = j / n;
  const int k = static_cast<int>(gk % n_chains);
  const int g = static_cast<int>(gk / n_chains);
  const double s = (sd_mu != nullptr) ? fmin(fmax(sd_mu[k], 1.0e-12), 1.0) : 1.0;
  const double h = cells[static_cast<std::size_t>(kCellStride) * c];
  const double se =
      fmax(finite_or_zero(sigma_t[static_cast<std::size_t>(c) * G + g]), 0.0) + inv_cdt;
  const double a00 = 0.5 * s + 0.5 * se * h;
  const double a01 = -0.5 * s;
  const double a10 = 0.5 * s;
  const double a11 = 0.5 * s + 0.5 * se * h;
  const double det = a00 * a11 - a01 * a10;
  double* out = sd_inv + static_cast<std::size_t>(j) * kInvStride;
  out[0] = a11 / det;
  out[1] = -a01 / det;
  out[2] = -a10 / det;
  out[3] = a00 / det;
}

// The operands of one cell of the sweep that do not depend on the sweep's
// recursion (loaded one diagonal ahead): the matrix inverse, the lumped
// masses and angular weights, the face area of the inflow face, the nodal
// source and the nodal history of the ordinate.
struct SweepOperands {
  double inv[kInvStride];
  double ML = 0.0;
  double MR = 0.0;
  double NL = 0.0;
  double NR = 0.0;
  double A_in = 0.0;
  double h = 0.0;
  double qL = 0.0;
  double qR = 0.0;
  double pL = 0.0;
  double pR = 0.0;
};

__device__ __forceinline__ SweepOperands load_sweep_operands(const double* __restrict__ cells,
                                                              const double* __restrict__ inv,
                                                              const double* __restrict__ qg,
                                                              const double* __restrict__ hist,
                                                              const int c, const bool inward) {
  SweepOperands o;
  const double* cell = cells + static_cast<std::size_t>(kCellStride) * c;
  const double* iv = inv + static_cast<std::size_t>(c) * kInvStride;
#pragma unroll
  for (int k = 0; k < kInvStride; ++k) {
    o.inv[k] = iv[k];
  }
  o.h = cell[0];
  o.ML = cell[1];
  o.MR = cell[2];
  o.NL = cell[3];
  o.NR = cell[4];
  o.A_in = inward ? cell[6] : cell[5];
  o.qL = qg[2 * c];
  o.qR = qg[2 * c + 1];
  if (hist != nullptr) {
    o.pL = hist[2 * c];
    o.pR = hist[2 * c + 1];
  }
  return o;
}

// One block (one warp) per group. Chains are processed one after the other;
// in a chain the starting direction runs first (lane 0), then the inward
// ordinates as a diagonal wavefront (lane k handles cell n - 1 - (d - k) at
// diagonal d, reading the angular edge lane k - 1 left in that cell at d - 1),
// then the outward ordinates from the centre the same way. The cell matrices
// are inverted beforehand (sweep_matrix_kernel), so a cell's solve is its
// right-hand side (the source, the inflow and the angular edge) times the
// inverse; every lane loads the operands of its next cell before solving the
// current one.
__global__ void sweep_kernel(const double* __restrict__ cells,
                             const double* __restrict__ inv,
                             const double* __restrict__ sd_inv,
                             const double* __restrict__ q,
                             const double* __restrict__ psi_prev,
                             const double* __restrict__ sd_prev,
                             const double* __restrict__ psi_in_g,
                             const double* __restrict__ mu,
                             const double* __restrict__ weight,
                             const double* __restrict__ alpha,
                             const double* __restrict__ tau,
                             const double* __restrict__ sd_mu,
                             double* __restrict__ psi,
                             double* __restrict__ sd,
                             const int n,
                             const int G,
                             const int N,
                             const int n_chains,
                             const int chain_len,
                             const double inv_cdt,
                             const int curved) {
  const int g = blockIdx.x;
  const int lane = threadIdx.x;
  if (g >= G) {
    return;
  }
  extern __shared__ double shared[];
  double* eL = shared;
  double* eR = shared + n;
  double* centre = shared + 2 * n;  // [chain_len / 2]
  const std::size_t two_n = 2 * static_cast<std::size_t>(n);
  const double* qg = q + static_cast<std::size_t>(g) * two_n;
  const double psi_in = (psi_in_g != nullptr) ? fmax(finite_or_zero(psi_in_g[g]), 0.0) : 0.0;
  const int half = chain_len / 2;
  for (int k = 0; k < n_chains; ++k) {
    const int base = k * chain_len;
    // Starting direction of the chain (curved geometries): -s dpsi/dr +
    // sigma psi = q + psi_sd_prev / (c dt), planar linear-discontinuous with
    // weight dr, inflow at the outer face.
    if (curved != 0) {
      if (lane == 0) {
        const double s = (sd_mu != nullptr) ? fmin(fmax(sd_mu[k], 1.0e-12), 1.0) : 1.0;
        double inflow = psi_in;
        const std::size_t sd_base = (static_cast<std::size_t>(g) * n_chains + k) * two_n;
        const double* sd_hist = (sd_prev != nullptr) ? sd_prev + sd_base : nullptr;
        const double* sinv =
            sd_inv + (static_cast<std::size_t>(g) * n_chains + k) * n * kInvStride;
        SweepOperands next = load_sweep_operands(cells, sinv, qg, sd_hist, n - 1, true);
        for (int c = n - 1; c >= 0; --c) {
          const SweepOperands cur = next;
          if (c > 0) {
            next = load_sweep_operands(cells, sinv, qg, sd_hist, c - 1, true);
          }
          const double h = cur.h;
          double qL = cur.qL;
          double qR = cur.qR;
          if (sd_hist != nullptr) {
            qL += inv_cdt * cur.pL;
            qR += inv_cdt * cur.pR;
          }
          const double b0 = 0.5 * h * qL;
          const double b1 = 0.5 * h * qR + s * inflow;
          const double xL = cur.inv[0] * b0 + cur.inv[1] * b1;
          const double xR = cur.inv[2] * b0 + cur.inv[3] * b1;
          eL[c] = xL;
          eR[c] = xR;
          if (sd != nullptr) {
            sd[sd_base + 2 * c] = xL;
            sd[sd_base + 2 * c + 1] = xR;
          }
          inflow = xL;
        }
      }
    }
    __syncwarp();
    // Inward half.
    {
      const int m = base + lane;
      const bool active = lane < half;
      double incoming = psi_in;
      double mu_m = 0.0, fac = 0.0, one_minus_tw = 0.0, inv_tw = 1.0;
      if (active) {
        mu_m = mu[m];
        const double w_m = weight[m];
        const double tw = fmin(fmax(tau[m], 1.0e-6), 1.0);
        fac = alpha[m + 1] * (1.0 - tw) / (w_m * tw) + alpha[m] / w_m;
        one_minus_tw = 1.0 - tw;
        inv_tw = 1.0 / tw;
      }
      const std::size_t p_base = (static_cast<std::size_t>(g) * N + (active ? m : 0)) * two_n;
      const double* hist = (psi_prev != nullptr) ? psi_prev + p_base : nullptr;
      const double* minv =
          inv + (static_cast<std::size_t>(g) * N + (active ? m : 0)) * n * kInvStride;
      const int diagonals = n + half - 1;
      // The lane's first cell (cp = 0) is reached at diagonal d = lane.
      SweepOperands next;
      if (active) {
        next = load_sweep_operands(cells, minv, qg, hist, n - 1, true);
      }
      for (int d = 0; d < diagonals; ++d) {
        const int cp = d - lane;
        if (active && cp >= 0 && cp < n) {
          const int c = n - 1 - cp;
          const SweepOperands cur = next;
          if (c > 0) {
            next = load_sweep_operands(cells, minv, qg, hist, c - 1, true);
          }
          double qL = cur.qL;
          double qR = cur.qR;
          if (hist != nullptr) {
            qL += inv_cdt * cur.pL;
            qR += inv_cdt * cur.pR;
          }
          // mu < 0: the inflow enters at the right face (node R's row).
          double b0 = cur.ML * qL;
          double b1 = cur.MR * qR - mu_m * cur.A_in * incoming;
          if (curved != 0) {
            b0 += fac * cur.NL * eL[c];
            b1 += fac * cur.NR * eR[c];
          }
          const double pL = cur.inv[0] * b0 + cur.inv[1] * b1;
          const double pR = cur.inv[2] * b0 + cur.inv[3] * b1;
          psi[p_base + 2 * c] = pL;
          psi[p_base + 2 * c + 1] = pR;
          if (curved != 0) {
            eL[c] = (pL - one_minus_tw * eL[c]) * inv_tw;
            eR[c] = (pR - one_minus_tw * eR[c]) * inv_tw;
          }
          incoming = pL;
        }
        __syncwarp();
      }
      if (active) {
        centre[lane] = incoming;  // outflow of cell 0 at the centre (node L)
      }
    }
    __syncwarp();
    // Outward half: ordinate base + half + lane, inflow at the centre from
    // its reflection partner (chain index half - 1 - lane).
    {
      const int m = base + half + lane;
      const bool active = lane < half;
      double incoming = 0.0;
      double mu_m = 0.0, fac = 0.0, one_minus_tw = 0.0, inv_tw = 1.0;
      if (active) {
        mu_m = mu[m];
        const double w_m = weight[m];
        const double tw = fmin(fmax(tau[m], 1.0e-6), 1.0);
        fac = alpha[m + 1] * (1.0 - tw) / (w_m * tw) + alpha[m] / w_m;
        one_minus_tw = 1.0 - tw;
        inv_tw = 1.0 / tw;
        incoming = centre[half - 1 - lane];
      }
      const std::size_t p_base = (static_cast<std::size_t>(g) * N + (active ? m : 0)) * two_n;
      const double* hist = (psi_prev != nullptr) ? psi_prev + p_base : nullptr;
      const double* minv =
          inv + (static_cast<std::size_t>(g) * N + (active ? m : 0)) * n * kInvStride;
      const int diagonals = n + half - 1;
      SweepOperands next;
      if (active) {
        next = load_sweep_operands(cells, minv, qg, hist, 0, false);
      }
      for (int d = 0; d < diagonals; ++d) {
        const int c = d - lane;
        if (active && c >= 0 && c < n) {
          const SweepOperands cur = next;
          if (c + 1 < n) {
            next = load_sweep_operands(cells, minv, qg, hist, c + 1, false);
          }
          double qL = cur.qL;
          double qR = cur.qR;
          if (hist != nullptr) {
            qL += inv_cdt * cur.pL;
            qR += inv_cdt * cur.pR;
          }
          // mu > 0: the inflow enters at the left face (node L's row).
          double b0 = cur.ML * qL + mu_m * cur.A_in * incoming;
          double b1 = cur.MR * qR;
          if (curved != 0) {
            b0 += fac * cur.NL * eL[c];
            b1 += fac * cur.NR * eR[c];
          }
          const double pL = cur.inv[0] * b0 + cur.inv[1] * b1;
          const double pR = cur.inv[2] * b0 + cur.inv[3] * b1;
          psi[p_base + 2 * c] = pL;
          psi[p_base + 2 * c + 1] = pR;
          if (curved != 0) {
            eL[c] = (pL - one_minus_tw * eL[c]) * inv_tw;
            eR[c] = (pR - one_minus_tw * eR[c]) * inv_tw;
          }
          incoming = pR;
        }
        __syncwarp();
      }
    }
    __syncwarp();
  }
}

// phi per (group, node) in ordinate order; face currents per (group, face).
__global__ void moments_kernel(const double* __restrict__ cells,
                               const double* __restrict__ psi,
                               const double* __restrict__ psi_in_g,
                               const double* __restrict__ mu,
                               const double* __restrict__ weight,
                               double* __restrict__ phi,
                               double* __restrict__ face_current,
                               double* __restrict__ prr,
                               const int n,
                               const int G,
                               const int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t two_n = 2 * static_cast<std::size_t>(n);
  if (idx < G * n) {
    const int g = idx / n;
    const int c = idx - g * n;
    double pL = 0.0, pR = 0.0, p2 = 0.0;
    for (int m = 0; m < N; ++m) {
      const std::size_t b = (static_cast<std::size_t>(g) * N + m) * two_n + 2 * c;
      const double wm = weight[m];
      const double vL = psi[b];
      const double vR = psi[b + 1];
      pL += wm * vL;
      pR += wm * vR;
      if (prr != nullptr) {
        const double* cell = cells + static_cast<std::size_t>(kCellStride) * c;
        p2 += wm * mu[m] * mu[m] * (cell[1] * vL + cell[2] * vR);
      }
    }
    phi[static_cast<std::size_t>(g) * two_n + 2 * c] = pL;
    phi[static_cast<std::size_t>(g) * two_n + 2 * c + 1] = pR;
    if (prr != nullptr) {
      const double* cell = cells + static_cast<std::size_t>(kCellStride) * c;
      const double V = cell[1] + cell[2];
      prr[static_cast<std::size_t>(c) * G + g] = (V > 0.0) ? p2 / V : 0.0;
    }
  }
  if (face_current != nullptr && idx < G * (n + 1)) {
    const int g = idx / (n + 1);
    const int f = idx - g * (n + 1);
    double F = 0.0;
    if (f > 0) {
      const double in = (psi_in_g != nullptr) ? fmax(finite_or_zero(psi_in_g[g]), 0.0) : 0.0;
      for (int m = 0; m < N; ++m) {
        const double mm = mu[m];
        double v;
        if (mm > 0.0) {
          v = psi[(static_cast<std::size_t>(g) * N + m) * two_n + 2 * (f - 1) + 1];
        } else {
          v = (f < n) ? psi[(static_cast<std::size_t>(g) * N + m) * two_n + 2 * f] : in;
        }
        F += weight[m] * mm * v;
      }
    }
    face_current[static_cast<std::size_t>(g) * (n + 1) + f] = F;
  }
}


// ---- Consistent P1 low-order system ----
//
// Per (system, cell) block, row-major: A[16] (own unknowns Phi_L, Phi_R, J_L,
// J_R; after the factorization the inverse of the block with the lower
// neighbour eliminated), Lnb[8] (rows x left neighbour's Phi_R, J_R), Unb[8]
// (rows x right neighbour's Phi_L, J_L), the lumped masses M_L, M_R that
// multiply the zeroth-moment rows' source, then the forward substitution's
// v[4] and W[8] = A^-1 Unb (rows x the right neighbour's Phi_L, J_L). Rows: zeroth moment at L and R, first moment at L and R.
constexpr int kP1Stride = 48;

struct Row8 {
  double own[4];
  double left[2];
  double right[2];
};

__device__ inline void row_zero(Row8& r) {
  for (int k = 0; k < 4; ++k) r.own[k] = 0.0;
  r.left[0] = r.left[1] = r.right[0] = r.right[1] = 0.0;
}

__device__ inline void row_axpy(Row8& y, const double a, const Row8& x) {
  for (int k = 0; k < 4; ++k) y.own[k] += a * x.own[k];
  y.left[0] += a * x.left[0];
  y.left[1] += a * x.left[1];
  y.right[0] += a * x.right[0];
  y.right[1] += a * x.right[1];
}

__global__ void p1_assemble_kernel(const double* __restrict__ cells,
                                   const double* __restrict__ sigma_e,
                                   const double* __restrict__ sigma_x,
                                   const double* __restrict__ mu,
                                   const double* __restrict__ weight,
                                   const double* __restrict__ alpha,
                                   const double* __restrict__ tau,
                                   const double* __restrict__ sd_mu,
                                   double* __restrict__ blocks,
                                   const int n,
                                   const int S,
                                   const int N,
                                   const int n_chains,
                                   const int chain_len,
                                   const int curved) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= S * n) {
    return;
  }
  const int sys = idx / n;
  const int c = idx - sys * n;
  double sum_w = 0.0, sum_wmu2 = 0.0;
  for (int m = 0; m < N; ++m) {
    sum_w += weight[m];
    sum_wmu2 += weight[m] * mu[m] * mu[m];
  }
  const double a = 1.0 / sum_w;
  const double b = 1.0 / sum_wmu2;
  const double* cell = cells + static_cast<std::size_t>(kCellStride) * c;
  const double h = cell[0], ML = cell[1], MR = cell[2], NL = cell[3], NR = cell[4],
               AL = cell[5], AR = cell[6];
  const double se = sigma_e[static_cast<std::size_t>(c) * S + sys];
  Row8 r0L, r0R, r1L, r1R;
  row_zero(r0L);
  row_zero(r0R);
  row_zero(r1L);
  row_zero(r1R);
  for (int k = 0; k < n_chains; ++k) {
    const double s = (sd_mu != nullptr) ? fmin(fmax(sd_mu[k], 1.0e-12), 1.0) : 1.0;
    double eL[4] = {a, 0.0, -b * s, 0.0};
    double eR[4] = {0.0, a, 0.0, -b * s};
    for (int jj = 0; jj < chain_len; ++jj) {
      const int m = k * chain_len + jj;
      const double mm = mu[m];
      const double wm = weight[m];
      const double tw = fmin(fmax(tau[m], 1.0e-6), 1.0);
      const double am = alpha[m];
      const double ap = alpha[m + 1];
      const double PL[4] = {a, 0.0, b * mm, 0.0};
      const double PR[4] = {0.0, a, 0.0, b * mm};
      Row8 RL, RR;
      row_zero(RL);
      row_zero(RR);
      const double sm = mm / h;
      for (int q = 0; q < 4; ++q) {
        const double stream = ML * PL[q] + MR * PR[q];
        RL.own[q] = sm * stream + se * ML * PL[q];
        RR.own[q] = -sm * stream + se * MR * PR[q];
      }
      if (mm > 0.0) {
        for (int q = 0; q < 4; ++q) RR.own[q] += mm * AR * PR[q];
        if (c > 0) {
          RL.left[0] -= mm * AL * a;
          RL.left[1] -= mm * AL * b * mm;
        } else {
          // reflection at the centre / axis / wall: the partner ordinate's
          // node-L value a Phi_L - b mu J_L of this cell
          RL.own[0] -= mm * AL * a;
          RL.own[2] -= mm * AL * (-b * mm);
        }
      } else {
        for (int q = 0; q < 4; ++q) RL.own[q] += -mm * AL * PL[q];
        if (c < n - 1) {
          RR.right[0] += mm * AR * a;
          RR.right[1] += mm * AR * b * mm;
        }
      }
      if (curved != 0) {
        const double cang = ap / (wm * tw);
        const double fac = ap * (1.0 - tw) / (wm * tw) + am / wm;
        for (int q = 0; q < 4; ++q) {
          RL.own[q] += cang * NL * PL[q] - fac * NL * eL[q];
          RR.own[q] += cang * NR * PR[q] - fac * NR * eR[q];
          eL[q] = (PL[q] - (1.0 - tw) * eL[q]) / tw;
          eR[q] = (PR[q] - (1.0 - tw) * eR[q]) / tw;
        }
      }
      row_axpy(r0L, wm, RL);
      row_axpy(r0R, wm, RR);
      row_axpy(r1L, wm * mm, RL);
      row_axpy(r1R, wm * mm, RR);
    }
  }
  const double sx = sigma_x[static_cast<std::size_t>(c) * S + sys];
  r0L.own[0] -= sx * ML;
  r0R.own[1] -= sx * MR;
  double* blk = blocks + (static_cast<std::size_t>(sys) * n + c) * kP1Stride;
  const Row8* rows[4] = {&r0L, &r0R, &r1L, &r1R};
  for (int r = 0; r < 4; ++r) {
    for (int q = 0; q < 4; ++q) blk[4 * r + q] = rows[r]->own[q];
    blk[16 + 2 * r] = rows[r]->left[0];
    blk[16 + 2 * r + 1] = rows[r]->left[1];
    blk[24 + 2 * r] = rows[r]->right[0];
    blk[24 + 2 * r + 1] = rows[r]->right[1];
  }
  blk[32] = ML;
  blk[33] = MR;
}

// Inverse of a 4 x 4 block (row-major, in registers) by Gauss-Jordan
// elimination with partial pivoting: one reciprocal per pivot.
__device__ __forceinline__ void inverse4(double (&A)[16], double (&X)[16]) {
#pragma unroll
  for (int q = 0; q < 16; ++q) X[q] = (q % 5 == 0) ? 1.0 : 0.0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    int p = k;
    double best = fabs(A[4 * k + k]);
#pragma unroll
    for (int r = k + 1; r < 4; ++r) {
      const double v = fabs(A[4 * r + k]);
      if (v > best) {
        best = v;
        p = r;
      }
    }
#pragma unroll
    for (int r = k + 1; r < 4; ++r) {
      if (p == r) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const double ta = A[4 * k + q];
          A[4 * k + q] = A[4 * r + q];
          A[4 * r + q] = ta;
          const double tx = X[4 * k + q];
          X[4 * k + q] = X[4 * r + q];
          X[4 * r + q] = tx;
        }
      }
    }
    const double inv_d = 1.0 / A[4 * k + k];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      A[4 * k + q] *= inv_d;
      X[4 * k + q] *= inv_d;
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      if (r != k) {
        const double f = A[4 * r + k];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          A[4 * r + q] -= f * A[4 * k + q];
          X[4 * r + q] -= f * X[4 * k + q];
        }
      }
    }
  }
}

// Block elimination of every system (one thread per system): the diagonal
// block with the lower neighbour eliminated, A_c - Lnb_c W_{c-1}, stored as
// its inverse (Gauss-Jordan with partial pivoting; the substitution is then a
// matrix-vector product), and W_c = A_c^-1 Unb_c for the backward
// substitution. The next cell's blocks are loaded before the current cell's
// elimination.
__global__ void p1_factor_kernel(double* __restrict__ blocks, const int n, const int S) {
  const int sys = blockIdx.x * blockDim.x + threadIdx.x;
  if (sys >= S) {
    return;
  }
  double* base = blocks + static_cast<std::size_t>(sys) * n * kP1Stride;
  double W_prev[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  // The operator part of a cell's block row: A[16], Lnb[8], Unb[8].
  double next[32];
#pragma unroll
  for (int q = 0; q < 32; ++q) next[q] = base[q];
  for (int c = 0; c < n; ++c) {
    double* blk = base + static_cast<std::size_t>(c) * kP1Stride;
    double cur[32];
#pragma unroll
    for (int q = 0; q < 32; ++q) cur[q] = next[q];
    if (c + 1 < n) {
      const double* nb = blk + kP1Stride;
#pragma unroll
      for (int q = 0; q < 32; ++q) next[q] = nb[q];
    }
    double A[16];
#pragma unroll
    for (int q = 0; q < 16; ++q) A[q] = cur[q];
    if (c > 0) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const double l0 = cur[16 + 2 * r];
        const double l1 = cur[16 + 2 * r + 1];
        // x_{c-1} components 1 (Phi_R) and 3 (J_R) = v - W (Phi_L, J_L)(c)
        A[4 * r + 0] -= l0 * W_prev[2 * 1 + 0] + l1 * W_prev[2 * 3 + 0];
        A[4 * r + 2] -= l0 * W_prev[2 * 1 + 1] + l1 * W_prev[2 * 3 + 1];
      }
    }
    double X[16];
    inverse4(A, X);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      double w0 = 0.0;
      double w1 = 0.0;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        w0 += X[4 * r + q] * cur[24 + 2 * q];
        w1 += X[4 * r + q] * cur[24 + 2 * q + 1];
      }
      W_prev[2 * r] = w0;
      W_prev[2 * r + 1] = w1;
    }
#pragma unroll
    for (int q = 0; q < 16; ++q) blk[q] = X[q];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      blk[40 + 2 * r] = W_prev[2 * r];
      blk[40 + 2 * r + 1] = W_prev[2 * r + 1];
    }
  }
}

// Forward and backward substitution of every system for the nodal source
// rhs[s * 2n + 2c + j] (one thread per system): v_c = A_c^-1 (b_c - Lnb_c
// v_{c-1}), x_c = v_c - W_c (Phi_L, J_L)(c + 1); writes the nodal Phi.
__global__ void p1_apply_kernel(double* __restrict__ blocks, const double* __restrict__ rhs,
                                double* __restrict__ x_out, const int n, const int S) {
  const int sys = blockIdx.x * blockDim.x + threadIdx.x;
  if (sys >= S) {
    return;
  }
  double* base = blocks + static_cast<std::size_t>(sys) * n * kP1Stride;
  const std::size_t two_n = 2 * static_cast<std::size_t>(n);
  const double* rs = rhs + static_cast<std::size_t>(sys) * two_n;
  double v_prev[4] = {0.0, 0.0, 0.0, 0.0};
  for (int c = 0; c < n; ++c) {
    double* blk = base + static_cast<std::size_t>(c) * kP1Stride;
    double bb[4];
    bb[0] = blk[32] * rs[2 * c];
    bb[1] = blk[33] * rs[2 * c + 1];
    bb[2] = 0.0;
    bb[3] = 0.0;
    if (c > 0) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const double l0 = blk[16 + 2 * r];
        const double l1 = blk[16 + 2 * r + 1];
        bb[r] -= l0 * v_prev[1] + l1 * v_prev[3];
      }
    }
    double v[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      v[r] = blk[4 * r] * bb[0] + blk[4 * r + 1] * bb[1] + blk[4 * r + 2] * bb[2] +
             blk[4 * r + 3] * bb[3];
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      blk[36 + r] = v[r];
      v_prev[r] = v[r];
    }
  }
  double x_next[4] = {0.0, 0.0, 0.0, 0.0};
  for (int c = n - 1; c >= 0; --c) {
    const double* blk = base + static_cast<std::size_t>(c) * kP1Stride;
    double x[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      x[r] = blk[36 + r];
      if (c < n - 1) {
        x[r] -= blk[40 + 2 * r] * x_next[0] + blk[40 + 2 * r + 1] * x_next[2];
      }
    }
    x_out[static_cast<std::size_t>(sys) * two_n + 2 * c] = x[0];
    x_out[static_cast<std::size_t>(sys) * two_n + 2 * c + 1] = x[1];
#pragma unroll
    for (int r = 0; r < 4; ++r) x_next[r] = x[r];
  }
}

}  // namespace

void compute_cells(const double* x_r, const int n_cells, const int geom, double* cells,
                   cudaStream_t stream) {
  if (n_cells <= 0) {
    return;
  }
  compute_cells_kernel<<<(n_cells + kBlock - 1) / kBlock, kBlock, 0, stream>>>(x_r, n_cells,
                                                                                geom, cells);
  cuda_check(cudaGetLastError(), "sn_ld compute_cells launch failed");
}

std::size_t sweep_inverse_doubles(const int n_cells, const int n_groups,
                                  const QuadratureView& quad, const int geom) {
  const bool curved = geom != static_cast<int>(mesh::Geometry1D::kPlanar);
  const std::size_t ord = static_cast<std::size_t>(std::max(n_groups, 0)) *
                          static_cast<std::size_t>(std::max(quad.n_angles, 0)) *
                          static_cast<std::size_t>(std::max(n_cells, 0));
  const std::size_t chains = curved ? static_cast<std::size_t>(std::max(n_groups, 0)) *
                                          static_cast<std::size_t>(std::max(quad.n_chains, 0)) *
                                          static_cast<std::size_t>(std::max(n_cells, 0))
                                    : 0U;
  return (ord + chains) * static_cast<std::size_t>(kInvStride);
}

void sweep_prepare(const double* cells, const double* sigma_t, const QuadratureView& quad,
                   double* inverse, const int n_cells, const int n_groups, const double inv_cdt,
                   const int geom, cudaStream_t stream) {
  if (n_cells <= 0 || n_groups <= 0) {
    return;
  }
  const int curved = (geom == static_cast<int>(mesh::Geometry1D::kPlanar)) ? 0 : 1;
  const long long n_ord =
      static_cast<long long>(n_groups) * quad.n_angles * static_cast<long long>(n_cells);
  const long long n_sd =
      (curved != 0) ? static_cast<long long>(n_groups) * quad.n_chains * n_cells : 0;
  const long long total = n_ord + n_sd;
  sweep_matrix_kernel<<<static_cast<unsigned>((total + kBlock - 1) / kBlock), kBlock, 0,
                        stream>>>(cells, sigma_t, quad.mu, quad.weight, quad.alpha, quad.tau,
                                  quad.sd_mu, inverse,
                                  inverse + static_cast<std::size_t>(n_ord) * kInvStride, n_cells,
                                  n_groups, quad.n_angles, quad.n_chains, curved, inv_cdt);
  cuda_check(cudaGetLastError(), "sn_ld sweep matrix launch failed");
}

void sweep_prepared(const double* cells, const double* inverse, const double* q,
                    const double* psi_prev, const double* sd_prev, const double* psi_in,
                    const QuadratureView& quad, double* psi, double* sd, const int n_cells,
                    const int n_groups, const double inv_cdt, const int geom,
                    cudaStream_t stream) {
  if (n_cells <= 0 || n_groups <= 0) {
    return;
  }
  TENRYU_ASSERT(quad.chain_len >= 2 && quad.chain_len % 2 == 0 &&
                    quad.chain_len / 2 <= kWarp &&
                    quad.n_chains * quad.chain_len == quad.n_angles,
                "sn_ld sweep: chains of at most 64 ordinates covering the quadrature");
  const int curved = (geom == static_cast<int>(mesh::Geometry1D::kPlanar)) ? 0 : 1;
  const std::size_t shared =
      (2 * static_cast<std::size_t>(n_cells) + static_cast<std::size_t>(quad.chain_len / 2)) *
      sizeof(double);
  static std::size_t configured_shared = 0;
  if (shared > 48U * 1024U && shared > configured_shared) {
    cuda_check(cudaFuncSetAttribute(sweep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(shared)),
               "sn_ld sweep shared memory attribute failed");
    configured_shared = shared;
  }
  const std::size_t n_ord = static_cast<std::size_t>(n_groups) *
                            static_cast<std::size_t>(quad.n_angles) *
                            static_cast<std::size_t>(n_cells);
  sweep_kernel<<<n_groups, kWarp, shared, stream>>>(
      cells, inverse, inverse + n_ord * kInvStride, q, psi_prev, sd_prev, psi_in, quad.mu,
      quad.weight, quad.alpha, quad.tau, quad.sd_mu, psi, sd, n_cells, n_groups, quad.n_angles,
      quad.n_chains, quad.chain_len, inv_cdt, curved);
  cuda_check(cudaGetLastError(), "sn_ld sweep launch failed");
}

void sweep(const double* cells, const double* sigma_t, const double* q, const double* psi_prev,
           const double* sd_prev, const double* psi_in, const QuadratureView& quad,
           double* psi, double* sd, const int n_cells, const int n_groups,
           const double inv_cdt, const int geom, cudaStream_t stream) {
  if (n_cells <= 0 || n_groups <= 0) {
    return;
  }
  double* inverse = static_cast<double*>(core::device_scratch_acquire(
      "sn_ld:sweep_inverse_standalone",
      std::max<std::size_t>(sweep_inverse_doubles(n_cells, n_groups, quad, geom), 1) *
          sizeof(double)));
  sweep_prepare(cells, sigma_t, quad, inverse, n_cells, n_groups, inv_cdt, geom, stream);
  sweep_prepared(cells, inverse, q, psi_prev, sd_prev, psi_in, quad, psi, sd, n_cells, n_groups,
                 inv_cdt, geom, stream);
}

void moments(const double* cells, const double* psi, const double* psi_in,
             const QuadratureView& quad, double* phi, double* face_current, double* prr,
             const int n_cells, const int n_groups, cudaStream_t stream) {
  if (n_cells <= 0 || n_groups <= 0) {
    return;
  }
  const int total = n_groups * (n_cells + 1);
  moments_kernel<<<(total + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      cells, psi, psi_in, quad.mu, quad.weight, phi, face_current, prr, n_cells, n_groups,
      quad.n_angles);
  cuda_check(cudaGetLastError(), "sn_ld moments launch failed");
}


std::size_t p1_scratch_doubles(const int n_cells, const int n_systems) {
  return static_cast<std::size_t>(std::max(n_cells, 0)) * static_cast<std::size_t>(std::max(n_systems, 0)) *
         static_cast<std::size_t>(kP1Stride);
}

void p1_factor(const double* cells, const double* sigma_e, const double* sigma_x,
               const QuadratureView& quad, double* factors, const int n_cells,
               const int n_systems, const int geom, cudaStream_t stream) {
  if (n_cells <= 0 || n_systems <= 0) {
    return;
  }
  const int curved = (geom == static_cast<int>(mesh::Geometry1D::kPlanar)) ? 0 : 1;
  const int total = n_cells * n_systems;
  p1_assemble_kernel<<<(total + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      cells, sigma_e, sigma_x, quad.mu, quad.weight, quad.alpha, quad.tau, quad.sd_mu, factors,
      n_cells, n_systems, quad.n_angles, quad.n_chains, quad.chain_len, curved);
  cuda_check(cudaGetLastError(), "sn_ld P1 assembly launch failed");
  p1_factor_kernel<<<(n_systems + 63) / 64, 64, 0, stream>>>(factors, n_cells, n_systems);
  cuda_check(cudaGetLastError(), "sn_ld P1 factorization launch failed");
}

void p1_apply(double* factors, const double* rhs, double* x, const int n_cells,
              const int n_systems, cudaStream_t stream) {
  if (n_cells <= 0 || n_systems <= 0) {
    return;
  }
  p1_apply_kernel<<<(n_systems + 63) / 64, 64, 0, stream>>>(factors, rhs, x, n_cells, n_systems);
  cuda_check(cudaGetLastError(), "sn_ld P1 substitution launch failed");
}

void p1_solve(const double* cells, const double* sigma_e, const double* sigma_x,
              const double* rhs, const QuadratureView& quad, double* x, double* scratch,
              const int n_cells, const int n_systems, const int geom, cudaStream_t stream) {
  p1_factor(cells, sigma_e, sigma_x, quad, scratch, n_cells, n_systems, geom, stream);
  p1_apply(scratch, rhs, x, n_cells, n_systems, stream);
}

// ---- One radiation step (advance_step) ----

namespace {

constexpr double kLdTiny = 1.0e-300;
constexpr int kReduceThreads = 256;
constexpr int kGmresRestart = 30;
// Largest factor the Newton linearization point moves by per iteration.
constexpr double kPointStep = 4.0;
// sn_material_retry_flag bits of this scheme (the cell-average Newton uses
// 1 and 2): 1 = a nodal electron energy fell below the temperature floor and
// was raised to it; 4 = the emission Newton iteration did not converge;
// 8 = the emission-coupling GMRES did not converge; 16 = the scattering
// source iteration did not converge.
constexpr int kRetryFloor = 1;
constexpr int kRetryNewton = 4;
constexpr int kRetryGmres = 8;
constexpr int kRetryScatter = 16;

// Per-cell electron closure inputs of the nodal matter update.
struct ClosureArgs {
  materials::DeviceEOSTableView run_eos{};
  materials::CellEOSTableSelector cell_tables{};
  const double* rho = nullptr;
  const double* zbar = nullptr;
  const double* cv_e = nullptr;
  const double* A_eff = nullptr;
  const double* gamma_eff = nullptr;
  // ee and Te at the step start: the reference state of the ideal-gas
  // closure e = e_ref + cv (T - T_ref).
  const double* e_start = nullptr;
  const double* T_start = nullptr;
  double cv_e_override = 0.0;
  double T_floor = 0.0;
  int low_density_extrap = 0;
};

// sn_electron_eos::invert_e_e_via_bisection<EOS_TAIL> computed by a full
// warp (every lane with the same arguments), with the same operations and
// result: the midpoints of the next five bisection levels — every node of the
// bisection tree below the current bracket, each from its own bracket by the
// same halvings — are evaluated in parallel, one per lane, and the bracket
// then descends the tree as the sequential loop does (the same decisions and
// the same stopping test), five levels per round instead of one.
template <bool EOS_TAIL>
__device__ double invert_e_warp(const materials::DeviceEOSTableView& tab,
                                const materials::RhoBracket& rb, const double e_target,
                                const double T_floor, const double rho, const double Zbar,
                                const double A, const bool lde, const int lane) {
  if constexpr (EOS_TAIL) {
    const double T_top = sn_electron_eos::sn_tail_T_top(tab);
    const double e_top = materials::device_eos_energy_extrap(tab, rb, rho, T_top, Zbar, A, lde);
    const double cv_top =
        fmax(materials::device_eos_cv_extrap(tab, rb, rho, T_top, Zbar, A, lde), 0.0);
    if (isfinite(e_top) && cv_top > 0.0 && e_target > e_top) {
      return T_top + (e_target - e_top) / cv_top;
    }
  }
  double xlo = log(fmax(T_floor, 1.0e-30));
  double xhi = fmax(tab.log_T_max, log(fmax(T_floor * 10.0, 1.0e-30)));
  constexpr int kMaxIterations = 60;
  int k = 0;
  bool done = false;
  while (!done) {
    // Node lane + 1 of the heap-ordered tree (root 1, children 2i and 2i + 1;
    // a right step is e(x_m) < e_target, x_lo = x_m).
    double e_node = 0.0;
    const int id = lane + 1;
    const int depth = 31 - __clz(id);
    if (id < 32 && k + depth < kMaxIterations) {
      double lo = xlo;
      double hi = xhi;
      for (int b = depth - 1; b >= 0; --b) {
        const double xm = 0.5 * (lo + hi);
        if (((id >> b) & 1) != 0) {
          lo = xm;
        } else {
          hi = xm;
        }
      }
      const double xm = 0.5 * (lo + hi);
      if constexpr (EOS_TAIL) {
        e_node = sn_electron_eos::sn_eval_e_tail(tab, rb, rho, exp(xm), Zbar, A, lde);
      } else {
        e_node = materials::device_eos_energy(tab, rb, xm);
      }
    }
    int node = 1;
    for (int level = 0; level < 5; ++level) {
      const double e_m = __shfl_sync(0xffffffffu, e_node, node - 1);
      const double xm = 0.5 * (xlo + xhi);
      if (e_m < e_target) {
        xlo = xm;
        node = 2 * node + 1;
      } else {
        xhi = xm;
        node = 2 * node;
      }
      ++k;
      if (xhi - xlo < 1.0e-14 || k >= kMaxIterations) {
        done = true;
        break;
      }
    }
  }
  return exp(0.5 * (xlo + xhi));
}

// The electron energy, heat capacity, temperature and pressure of one cell's
// material at the cell density (the table closures of the cell-average
// Newton, sn_material_newton_gpu, with the same tail convention).
template <bool EOS_TAIL>
struct CellClosure {
  materials::DeviceEOSTableView tab{};
  materials::RhoBracket rb{};
  bool table = false;
  bool lde = false;
  double rho = kLdTiny;
  double Zbar = 1.0;
  double A = 1.0;
  double gamma = 5.0 / 3.0;
  double cv_mass = 0.0;
  double e_ref = 0.0;
  double T_ref = 0.0;
  double T_floor = 0.0;

  __device__ CellClosure(const ClosureArgs& a, const int c) {
    tab = a.cell_tables.electron(c, a.run_eos);
    table = sn_electron_eos::has_electron_eos_table(tab);
    lde = a.low_density_extrap != 0;
    rho = fmax(sn_electron_eos::nonnegative_finite(a.rho[c]), kLdTiny);
    Zbar = (a.zbar != nullptr) ? a.zbar[c] : 1.0;
    A = (a.A_eff != nullptr) ? fmax(a.A_eff[c], 1.0e-12) : 1.0;
    gamma = (a.gamma_eff != nullptr) ? fmax(a.gamma_eff[c], 1.0 + 1.0e-12) : 5.0 / 3.0;
    const double Cv_override = a.cell_tables.cv_e_override(c, a.cv_e_override);
    cv_mass = sn_electron_eos::mass_heat_capacity(rho, Zbar, (a.cv_e != nullptr) ? a.cv_e[c] : 0.0,
                                                  0.0, Cv_override, A, gamma);
    T_floor = a.T_floor;
    e_ref = sn_electron_eos::finite_or_zero(a.e_start[c]);
    T_ref = fmax(sn_electron_eos::finite_or_zero(a.T_start[c]), T_floor);
    if (table) {
      rb = materials::find_rho_bracket(tab, rho);
    }
  }

  __device__ double energy(const double T) const {
    if (table) {
      if constexpr (EOS_TAIL) {
        return sn_electron_eos::sn_eval_e_tail(tab, rb, rho, T, Zbar, A, lde);
      } else {
        return materials::device_eos_energy_extrap(tab, rb, rho, T, Zbar, A, lde);
      }
    }
    return e_ref + cv_mass * (T - T_ref);
  }

  __device__ double heat_capacity(const double T) const {
    if (table) {
      if constexpr (EOS_TAIL) {
        return sn_electron_eos::nonnegative_finite(
            sn_electron_eos::sn_eval_cv_tail(tab, rb, rho, T, Zbar, A, lde));
      } else {
        return sn_electron_eos::nonnegative_finite(
            materials::device_eos_cv_extrap(tab, rb, rho, T, Zbar, A, lde));
      }
    }
    return cv_mass;
  }

  __device__ double temperature(const double e) const {
    if (table) {
      return sn_electron_eos::invert_e_e_via_bisection<EOS_TAIL>(tab, rb, e, T_floor, rho, Zbar,
                                                                  A, lde);
    }
    return (cv_mass > 0.0) ? fmax(T_ref + (e - e_ref) / cv_mass, T_floor) : T_ref;
  }

  // temperature(e) by a full warp (every lane with the same e; the same value).
  __device__ double temperature_warp(const double e, const int lane) const {
    if (table) {
      return invert_e_warp<EOS_TAIL>(tab, rb, e, T_floor, rho, Zbar, A, lde, lane);
    }
    return (cv_mass > 0.0) ? fmax(T_ref + (e - e_ref) / cv_mass, T_floor) : T_ref;
  }

  __device__ double pressure(const double T, const double e) const {
    if (table) {
      if constexpr (EOS_TAIL) {
        return sn_electron_eos::sn_eval_P_tail(tab, rb, rho, T, Zbar, A, lde);
      } else {
        return materials::device_eos_pressure_extrap(tab, rb, rho, T, Zbar, A, lde);
      }
    }
    return fmax(gamma - 1.0, 1.0e-12) * rho * e;
  }
};

__device__ inline bool cell_void(const std::uint8_t* is_void, const int c) {
  return is_void != nullptr && is_void[c] != 0U;
}

// Angular histories at the step start, per (group, cell): rescaled so that
// their lumped-mass zeroth moment is c E_old V (the radiation energy the
// other operators left: compression, remap, a restart), or seeded isotropic
// and flat in the cell (first step, size change, remap, no usable history).
// phi_hist: the nodal scalar flux of the prepared histories.
__global__ void history_kernel(const double* __restrict__ cells,
                               const double* __restrict__ rad_E_old,
                               const double* __restrict__ weight,
                               double* __restrict__ psi_prev,
                               double* __restrict__ sd_prev,
                               double* __restrict__ phi_hist,
                               const int n,
                               const int G,
                               const int N,
                               const int n_chains,
                               const int seed) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n * G) {
    return;
  }
  const int g = idx / n;
  const int c = idx - g * n;
  const double* cell = cells + static_cast<std::size_t>(kCellStride) * c;
  const double ML = cell[1];
  const double MR = cell[2];
  const double V = ML + MR;
  const std::size_t two_n = 2 * static_cast<std::size_t>(n);
  const double target =
      core::constants::c_light * finite_or_zero(rad_E_old[static_cast<std::size_t>(c) * G + g]);
  double sum_w = 0.0;
  double pL = 0.0;
  double pR = 0.0;
  for (int m = 0; m < N; ++m) {
    const double wm = weight[m];
    sum_w += wm;
    if (seed == 0) {
      const std::size_t b = (static_cast<std::size_t>(g) * N + m) * two_n + 2 * c;
      pL += wm * psi_prev[b];
      pR += wm * psi_prev[b + 1];
    }
  }
  const double S = ML * pL + MR * pR;
  const std::size_t hb = static_cast<std::size_t>(g) * two_n + 2 * c;
  if (seed == 0 && isfinite(S) && S > 0.0 && target >= 0.0 && V > 0.0) {
    const double f = target * V / S;
    for (int m = 0; m < N; ++m) {
      const std::size_t b = (static_cast<std::size_t>(g) * N + m) * two_n + 2 * c;
      psi_prev[b] *= f;
      psi_prev[b + 1] *= f;
    }
    if (sd_prev != nullptr) {
      for (int k = 0; k < n_chains; ++k) {
        const std::size_t b = (static_cast<std::size_t>(g) * n_chains + k) * two_n + 2 * c;
        sd_prev[b] *= f;
        sd_prev[b + 1] *= f;
      }
    }
    phi_hist[hb] = f * pL;
    phi_hist[hb + 1] = f * pR;
    return;
  }
  const double v = (sum_w > 0.0) ? target / sum_w : 0.0;
  for (int m = 0; m < N; ++m) {
    const std::size_t b = (static_cast<std::size_t>(g) * N + m) * two_n + 2 * c;
    psi_prev[b] = v;
    psi_prev[b + 1] = v;
  }
  if (sd_prev != nullptr) {
    for (int k = 0; k < n_chains; ++k) {
      const std::size_t b = (static_cast<std::size_t>(g) * n_chains + k) * two_n + 2 * c;
      sd_prev[b] = v;
      sd_prev[b + 1] = v;
    }
  }
  phi_hist[hb] = target;
  phi_hist[hb + 1] = target;
}

// Nodal electron energies at the step start from the cell's ee and the
// carried offset e_R - e_L (lumped-mass mean preserved; the offset is
// reduced where a node would fall below the temperature floor's energy).
template <bool EOS_TAIL>
__global__ void matter_setup_kernel(const ClosureArgs a,
                                    const double* __restrict__ cells,
                                    const std::uint8_t* __restrict__ is_void,
                                    const double* __restrict__ ee,
                                    double* __restrict__ offset,
                                    const int reset_offset,
                                    double* __restrict__ e_n,
                                    double* __restrict__ e_k,
                                    double* __restrict__ T_k,
                                    const int n) {
  // One warp per cell (the temperatures by the warp's inversion); lane 0
  // writes.
  const int lane = static_cast<int>(threadIdx.x) & (kWarp - 1);
  const int c = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) / kWarp) +
                static_cast<int>(threadIdx.x) / kWarp;
  if (c >= n) {
    return;
  }
  const double* cell = cells + static_cast<std::size_t>(kCellStride) * c;
  const double ML = cell[1];
  const double MR = cell[2];
  const double V = ML + MR;
  const CellClosure<EOS_TAIL> cl(a, c);
  const double ebar = finite_or_zero(ee[c]);
  double d = (reset_offset != 0) ? 0.0 : finite_or_zero(offset[c]);
  if (cell_void(is_void, c) || !(V > 0.0)) {
    d = 0.0;
  } else {
    const double e_min = cl.energy(a.T_floor);
    if (!(ebar > e_min)) {
      d = 0.0;
    } else if (d > 0.0 && ebar - (MR / V) * d < e_min) {
      d = (ebar - e_min) * V / MR;
    } else if (d < 0.0 && ebar + (ML / V) * d < e_min) {
      d = -(ebar - e_min) * V / ML;
    }
  }
  const double eL = (V > 0.0) ? ebar - (MR / V) * d : ebar;
  const double eR = (V > 0.0) ? ebar + (ML / V) * d : ebar;
  double TL = 0.0;
  double TR = 0.0;
  if (cell_void(is_void, c)) {
    TL = fmax(finite_or_zero(a.T_start[c]), a.T_floor);
    TR = TL;
  } else {
    TL = cl.temperature_warp(eL, lane);
    TR = cl.temperature_warp(eR, lane);
  }
  // Every lane has read offset[c] before lane 0 overwrites it.
  __syncwarp();
  if (lane != 0) {
    return;
  }
  offset[c] = d;
  e_n[2 * c] = eL;
  e_n[2 * c + 1] = eR;
  e_k[2 * c] = eL;
  e_k[2 * c + 1] = eR;
  T_k[2 * c] = TL;
  T_k[2 * c + 1] = TR;
}

// The cell temperature of the current nodal energies (the temperature the
// cell opacities are evaluated at).
template <bool EOS_TAIL>
__global__ void cell_temperature_kernel(const ClosureArgs a,
                                        const double* __restrict__ cells,
                                        const std::uint8_t* __restrict__ is_void,
                                        const double* __restrict__ e_k,
                                        double* __restrict__ Te,
                                        const int n) {
  // One warp per cell; lane 0 writes.
  const int lane = static_cast<int>(threadIdx.x) & (kWarp - 1);
  const int c = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) / kWarp) +
                static_cast<int>(threadIdx.x) / kWarp;
  if (c >= n || cell_void(is_void, c)) {
    return;
  }
  const double* cell = cells + static_cast<std::size_t>(kCellStride) * c;
  const double V = cell[1] + cell[2];
  if (!(V > 0.0)) {
    return;
  }
  const CellClosure<EOS_TAIL> cl(a, c);
  const double T =
      cl.temperature_warp((cell[1] * e_k[2 * c] + cell[2] * e_k[2 * c + 1]) / V, lane);
  if (lane == 0) {
    Te[c] = T;
  }
}

// Void cells carry no absorption, emission or scattering; the sweep's total
// cross section and the scattering DSA's (with the time term) per cell.
__global__ void opacity_post_kernel(const std::uint8_t* __restrict__ is_void,
                                    double* __restrict__ sigma_a,
                                    double* __restrict__ sigma_pe,
                                    double* __restrict__ sigma_s,
                                    double* __restrict__ sigma_t,
                                    double* __restrict__ sigma_e_dsa,
                                    int* __restrict__ scatter_flag,
                                    const int n,
                                    const int G,
                                    const double inv_cdt) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n * G) {
    return;
  }
  double sa = sn_electron_eos::nonnegative_finite(sigma_a[idx]);
  double spe = sn_electron_eos::nonnegative_finite(sigma_pe[idx]);
  double ss = sn_electron_eos::nonnegative_finite(sigma_s[idx]);
  if (cell_void(is_void, idx / G)) {
    sa = 0.0;
    spe = 0.0;
    ss = 0.0;
  }
  sigma_a[idx] = sa;
  sigma_pe[idx] = spe;
  sigma_s[idx] = ss;
  sigma_t[idx] = sa + ss;
  sigma_e_dsa[idx] = sa + ss + inv_cdt;
  if (ss > 0.0) {
    *scatter_flag = 1;
  }
}

// The emission linearized about the nodal temperatures T_k (NUMERICS 6.8):
// eps_g = c sigma_pe,g [B_g(T_k) + B'_g(T_k) (T - T_k)] with the nodal matter
// equation rho (e(T) - e^n) = dt (A - sum_g eps_g), e(T) ~ e_k + cv (T - T_k),
// A = sum_g sigma_a,g phi_g, gives eps_g = fixed_g + kappa_g A with
// D = rho cv + dt sum_h chi_h, chi_g = c sigma_pe,g B'_g,
// fixed_g = c sigma_pe,g B_g - chi_g [dt sum_h c sigma_pe,h B_h + rho (e_k - e^n)] / D,
// kappa_g = chi_g dt / D. Also the grey low-order coefficients of the
// emission-coupling preconditioner (spectral shape xi_g ~ kappa_g /
// (sigma_a,g + 1/(c dt)), sum xi = 1): kbar = sum kappa, the xi-mean
// absorption, the xi-harmonic transport cross section and the removal
// (1 - kbar) sigma_a,xi + 1/(c dt); the GMRES scale of every node
// sum_g sigma_a,g (|phi_ref| + c B) and the initial absorption-rate density
// sum_g sigma_a,g phi_ref.
template <bool EOS_TAIL>
__global__ void linearize_kernel(const ClosureArgs a,
                                 const std::uint8_t* __restrict__ is_void,
                                 PlanckTableDeviceView planck,
                                 const double* __restrict__ sigma_a,
                                 const double* __restrict__ sigma_pe,
                                 const double* __restrict__ sigma_s,
                                 const double* __restrict__ e_n,
                                 const double* __restrict__ e_k,
                                 const double* __restrict__ T_k,
                                 const double* __restrict__ phi_ref,
                                 double* __restrict__ fixed,
                                 double* __restrict__ kappa,
                                 double* __restrict__ kbar,
                                 double* __restrict__ sabar,
                                 double* __restrict__ se_node,
                                 double* __restrict__ rem_node,
                                 double* __restrict__ scale,
                                 double* __restrict__ x0,
                                 const int n,
                                 const int G,
                                 const double dt,
                                 const double inv_cdt) {
  // One warp per node: the Planck terms of the groups in parallel over the
  // lanes (shared memory, 3 G doubles per warp), the sums over the groups in
  // the group order by every lane.
  extern __shared__ double warp_shared[];
  const int lane = static_cast<int>(threadIdx.x) & (kWarp - 1);
  const int warp = static_cast<int>(threadIdx.x) / kWarp;
  const int node = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) / kWarp) + warp;
  const int two_n = 2 * n;
  if (node >= two_n) {
    return;
  }
  double* sh_cP = warp_shared + static_cast<std::size_t>(warp) * 3 * static_cast<std::size_t>(G);
  double* sh_cdP = sh_cP + G;
  double* sh_B = sh_cdP + G;
  const int c = node >> 1;
  const double cl_light = core::constants::c_light;
  const bool vd = cell_void(is_void, c);
  const double T = fmax(finite_or_zero(T_k[node]), a.T_floor);
  const PlanckTableDeviceView::Location loc = planck.locate_b(T);
  const double T3 = T * T * T;
  const double T4 = T3 * T;
  for (int g = lane; g < G; g += kWarp) {
    double b = 1.0;
    double db = 0.0;
    if (G != 1) {
      planck.interpolate_b_and_dT(g, loc, T, b, db);
      b = fmax(b, 0.0);
    }
    const double B = core::constants::a_eV * T4 * b;
    // d(T^4 b_g)/dT = int over the group of x f e^x / (e^x - 1) > 0 for the
    // Planck density f; a negative value is the table interpolation's.
    const double Bp = core::constants::a_eV * fmax(4.0 * T3 * b + T4 * db, 0.0);
    const double spe = sigma_pe[static_cast<std::size_t>(c) * G + g];
    sh_cP[g] = cl_light * spe * B;
    sh_cdP[g] = cl_light * spe * Bp;
    sh_B[g] = B;
  }
  __syncwarp();
  double sum_P = 0.0;
  double sum_dP = 0.0;
  double A0 = 0.0;
  double sc = 0.0;
  double sum_t = 0.0;
  for (int g = 0; g < G; ++g) {
    const std::size_t gi = static_cast<std::size_t>(g) * two_n + node;
    const std::size_t ci = static_cast<std::size_t>(c) * G + g;
    const double sa = sigma_a[ci];
    const double cP = sh_cP[g];
    const double cdP = sh_cdP[g];
    const double B = sh_B[g];
    sum_P += cP;
    sum_dP += cdP;
    const double ph = (phi_ref != nullptr) ? phi_ref[gi] : 0.0;
    A0 += sa * ph;
    sc += sa * (fabs(ph) + cl_light * B);
    sum_t += sa + sigma_s[ci];
  }
  double D = 0.0;
  double W = 0.0;
  if (!vd) {
    const CellClosure<EOS_TAIL> cl(a, c);
    D = cl.rho * cl.heat_capacity(T) + dt * sum_dP;
    W = dt * sum_P + cl.rho * (e_k[node] - e_n[node]);
  }
  const bool coupled = !vd && isfinite(D) && D > 0.0;
  double kb = 0.0;
  double sxi = 0.0;
  double sxa = 0.0;
  double sxinv = 0.0;
  for (int g = 0; g < G; ++g) {
    const std::size_t gi = static_cast<std::size_t>(g) * two_n + node;
    const std::size_t ci = static_cast<std::size_t>(c) * G + g;
    const double cP = sh_cP[g];
    const double chi = sh_cdP[g];
    double f = cP;
    double kap = 0.0;
    if (coupled) {
      f = cP - chi * W / D;
      kap = chi * dt / D;
    }
    if ((g & (kWarp - 1)) == lane) {
      fixed[gi] = f;
      kappa[gi] = kap;
    }
    kb += kap;
    const double sa = sigma_a[ci];
    const double xi = kap / (sa + inv_cdt);
    sxi += xi;
    sxa += xi * sa;
    sxinv += xi / (sa + sigma_s[ci] + inv_cdt);
  }
  if (lane != 0) {
    return;
  }
  kbar[node] = kb;
  double sa_xi = 0.0;
  double se = inv_cdt + sum_t / static_cast<double>(G);
  if (sxi > 0.0 && sxinv > 0.0) {
    sa_xi = sxa / sxi;
    se = sxi / sxinv;
  }
  sabar[node] = sa_xi;
  se_node[node] = se;
  rem_node[node] = fmax(1.0 - kb, 0.0) * sa_xi + inv_cdt;
  scale[node] = (isfinite(sc) && sc > 0.0) ? sc : 1.0;
  x0[node] = isfinite(A0) ? A0 : 0.0;
}

// The grey low-order system's cell coefficients: lumped-mass means of the
// nodal transport cross section and removal; sigma_x = sigma_e - removal.
__global__ void lmfg_cell_kernel(const double* __restrict__ cells,
                                 const double* __restrict__ se_node,
                                 const double* __restrict__ rem_node,
                                 double* __restrict__ p1_se,
                                 double* __restrict__ p1_sx,
                                 const int n) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) {
    return;
  }
  const double* cell = cells + static_cast<std::size_t>(kCellStride) * c;
  const double ML = cell[1];
  const double MR = cell[2];
  const double V = ML + MR;
  const double se = (ML * se_node[2 * c] + MR * se_node[2 * c + 1]) / V;
  const double rem = (ML * rem_node[2 * c] + MR * rem_node[2 * c + 1]) / V;
  p1_se[c] = se;
  p1_sx[c] = se - rem;
}

// Per-angle isotropic source of the sweep: (fixed_g + kappa_g x + S (group
// 0) + sigma_s,g phi_s,g) / 2 with x the nodal absorption-rate density
// (times xscale when given); fixed, S and phi_s optional.
__global__ void source_kernel(const double* __restrict__ fixed,
                              const double* __restrict__ kappa,
                              const double* __restrict__ x,
                              const double* __restrict__ xscale,
                              const double* __restrict__ source_ext,
                              const double* __restrict__ sigma_s,
                              const double* __restrict__ phi_s,
                              double* __restrict__ q,
                              const int n,
                              const int G) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int two_n = 2 * n;
  if (idx >= G * two_n) {
    return;
  }
  const int g = idx / two_n;
  const int node = idx - g * two_n;
  const int c = node >> 1;
  const double xv = x[node] * ((xscale != nullptr) ? xscale[node] : 1.0);
  double v = kappa[idx] * xv;
  if (fixed != nullptr) {
    v += fixed[idx];
  }
  if (source_ext != nullptr && g == 0) {
    v += source_ext[c];
  }
  if (phi_s != nullptr) {
    v += sigma_s[static_cast<std::size_t>(c) * G + g] * phi_s[idx];
  }
  q[idx] = 0.5 * v;
}

// A[node] = sum_g sigma_a,g phi_g at the node.
__global__ void absorption_kernel(const double* __restrict__ sigma_a,
                                  const double* __restrict__ phi,
                                  double* __restrict__ A,
                                  const int n,
                                  const int G) {
  const int node = blockIdx.x * blockDim.x + threadIdx.x;
  const int two_n = 2 * n;
  if (node >= two_n) {
    return;
  }
  const int c = node >> 1;
  double s = 0.0;
  for (int g = 0; g < G; ++g) {
    s += sigma_a[static_cast<std::size_t>(c) * G + g] * phi[static_cast<std::size_t>(g) * two_n + node];
  }
  A[node] = s;
}

// Deterministic block reductions: one block per output, a fixed strided
// order per thread and a fixed tree.
__device__ inline double block_sum(double v, double* sh) {
  sh[threadIdx.x] = v;
  __syncthreads();
  for (int s = kReduceThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sh[threadIdx.x] += sh[threadIdx.x + s];
    }
    __syncthreads();
  }
  return sh[0];
}

__device__ inline double block_max(double v, double* sh) {
  sh[threadIdx.x] = v;
  __syncthreads();
  for (int s = kReduceThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sh[threadIdx.x] = fmax(sh[threadIdx.x], sh[threadIdx.x + s]);
    }
    __syncthreads();
  }
  return sh[0];
}

// out[i] = sum_k V[i * ld + k] w[k], one block per i.
__global__ void dots_kernel(const double* __restrict__ V,
                            const std::size_t ld,
                            const double* __restrict__ w,
                            double* __restrict__ out,
                            const int m) {
  __shared__ double sh[kReduceThreads];
  const double* v = V + static_cast<std::size_t>(blockIdx.x) * ld;
  double s = 0.0;
  for (int k = threadIdx.x; k < m; k += kReduceThreads) {
    s += v[k] * w[k];
  }
  const double total = block_sum(s, sh);
  if (threadIdx.x == 0) {
    out[blockIdx.x] = total;
  }
}

// out[i] = sum_k V[i * ld + k] w[k] for i < nvec and out[nvec] = sum_k w[k]^2,
// one block per output.
__global__ void dots_norm_kernel(const double* __restrict__ V,
                                 const std::size_t ld,
                                 const double* __restrict__ w,
                                 double* __restrict__ out,
                                 const int nvec,
                                 const int m) {
  __shared__ double sh[kReduceThreads];
  const double* v =
      (static_cast<int>(blockIdx.x) < nvec) ? V + static_cast<std::size_t>(blockIdx.x) * ld : w;
  double s = 0.0;
  for (int k = threadIdx.x; k < m; k += kReduceThreads) {
    s += v[k] * w[k];
  }
  const double total = block_sum(s, sh);
  if (threadIdx.x == 0) {
    out[blockIdx.x] = total;
  }
}

// out[0] = max_k a[k], out[1] = max_k b[k] (b optional).
__global__ void max2_kernel(const double* __restrict__ a,
                            const double* __restrict__ b,
                            double* __restrict__ out,
                            const int m) {
  __shared__ double sh[kReduceThreads];
  double va = 0.0;
  double vb = 0.0;
  for (int k = threadIdx.x; k < m; k += kReduceThreads) {
    va = fmax(va, a[k]);
    if (b != nullptr) {
      vb = fmax(vb, b[k]);
    }
  }
  const double ra = block_max(va, sh);
  __syncthreads();
  const double rb = block_max(vb, sh);
  if (threadIdx.x == 0) {
    out[0] = ra;
    out[1] = rb;
  }
}

// out[0] = the largest local gain of the grey correction over the nodes, kbar
// sabar / rem: in an infinite medium a smooth absorption-rate perturbation v
// is corrected to (1 + gain) v, and the unpreconditioned coupling K multiplies
// it by gain / (1 + gain).
__global__ void precond_gain_max_kernel(const double* __restrict__ kbar,
                                        const double* __restrict__ sabar,
                                        const double* __restrict__ rem,
                                        double* __restrict__ out,
                                        const int m) {
  __shared__ double sh[kReduceThreads];
  double v = 0.0;
  for (int k = threadIdx.x; k < m; k += kReduceThreads) {
    const double g = (rem[k] > 0.0) ? kbar[k] * sabar[k] / rem[k] : 0.0;
    v = fmax(v, isfinite(g) ? g : 0.0);
  }
  const double r = block_max(v, sh);
  if (threadIdx.x == 0) {
    out[0] = r;
  }
}

// w -= sum_i h[i] V_i.
__global__ void orth_update_kernel(const double* __restrict__ V,
                                   const std::size_t ld,
                                   const double* __restrict__ h,
                                   const int nvec,
                                   double* __restrict__ w,
                                   const int m) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= m) {
    return;
  }
  double s = w[k];
  for (int i = 0; i < nvec; ++i) {
    s -= h[i] * V[static_cast<std::size_t>(i) * ld + k];
  }
  w[k] = s;
}

__global__ void fill_kernel(double* __restrict__ v, const double value, const int m) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < m) {
    v[k] = value;
  }
}

__global__ void scale_vector_kernel(const double* __restrict__ w, const double alpha,
                                    double* __restrict__ v, const int m) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < m) {
    v[k] = alpha * w[k];
  }
}

// x += s * sum_i y[i] Z_i.
__global__ void solution_update_kernel(const double* __restrict__ Z,
                                       const std::size_t ld,
                                       const double* __restrict__ y,
                                       const int nvec,
                                       const double* __restrict__ s,
                                       double* __restrict__ x,
                                       const int m) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= m) {
    return;
  }
  double acc = 0.0;
  for (int i = 0; i < nvec; ++i) {
    acc += y[i] * Z[static_cast<std::size_t>(i) * ld + k];
  }
  x[k] += s[k] * acc;
}

// r = (Tx - x) / s, and Tx / s, x / s for the reference norm.
__global__ void residual_kernel(const double* __restrict__ Tx,
                                const double* __restrict__ x,
                                const double* __restrict__ s,
                                double* __restrict__ r,
                                double* __restrict__ tx_s,
                                double* __restrict__ x_s,
                                const int m) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= m) {
    return;
  }
  const double inv = 1.0 / s[k];
  r[k] = (Tx[k] - x[k]) * inv;
  tx_s[k] = Tx[k] * inv;
  x_s[k] = x[k] * inv;
}

// Preconditioner: rhs = kbar (s v) for the grey low-order system, then
// z = v + sabar Phi / s.
__global__ void precond_rhs_kernel(const double* __restrict__ kbar,
                                   const double* __restrict__ s,
                                   const double* __restrict__ v,
                                   double* __restrict__ rhs,
                                   const int m) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < m) {
    rhs[k] = kbar[k] * s[k] * v[k];
  }
}

__global__ void precond_apply_kernel(const double* __restrict__ v,
                                     const double* __restrict__ sabar,
                                     const double* __restrict__ Phi,
                                     const double* __restrict__ s,
                                     double* __restrict__ z,
                                     const int m) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < m) {
    z[k] = v[k] + sabar[k] * Phi[k] / s[k];
  }
}

// w = z - (K s z) / s.
__global__ void matvec_finish_kernel(const double* __restrict__ z,
                                     const double* __restrict__ Kz,
                                     const double* __restrict__ s,
                                     double* __restrict__ w,
                                     const int m) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < m) {
    w[k] = z[k] - Kz[k] / s[k];
  }
}

// Scattering source iteration: the DSA source sigma_s (phi_half - phi_s).
__global__ void scatter_residual_kernel(const double* __restrict__ sigma_s,
                                        const double* __restrict__ phi_half,
                                        const double* __restrict__ phi_s,
                                        double* __restrict__ rhs,
                                        const int n,
                                        const int G) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int two_n = 2 * n;
  if (idx >= G * two_n) {
    return;
  }
  const int g = idx / two_n;
  const int c = (idx - g * two_n) >> 1;
  rhs[idx] = sigma_s[static_cast<std::size_t>(c) * G + g] * (phi_half[idx] - phi_s[idx]);
}

// phi_s <- phi_half + Phi; d_abs = |sigma_s (new - old)|, d_ref = |sigma_s new|.
__global__ void scatter_update_kernel(const double* __restrict__ sigma_s,
                                      const double* __restrict__ phi_half,
                                      const double* __restrict__ Phi,
                                      double* __restrict__ phi_s,
                                      double* __restrict__ d_abs,
                                      double* __restrict__ d_ref,
                                      const int n,
                                      const int G) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int two_n = 2 * n;
  if (idx >= G * two_n) {
    return;
  }
  const int g = idx / two_n;
  const int c = (idx - g * two_n) >> 1;
  const double ss = sigma_s[static_cast<std::size_t>(c) * G + g];
  const double nv = phi_half[idx] + Phi[idx];
  d_abs[idx] = fabs(ss * (nv - phi_s[idx]));
  d_ref[idx] = fabs(ss * nv);
  phi_s[idx] = nv;
}

// The node's own energy balance with its absorption-rate density held,
// f(T) = rho (e(T) - e^n) + dt sum_g c sigma_pe,g B_g(T) - dt A, increasing in
// T: its root is the next linearization point of the Newton iteration. It is
// bounded by the energy the absorption supplies (e(T) <= e^n + dt A / rho), so
// the iteration cannot run away through the convexity of B (a tangent taken
// far below the solution overshoots by orders of magnitude when the emission
// dominates the heat capacity). Safeguarded Newton inside the bracket
// [T_floor, T(e^n + dt max(A, 0) / rho)], geometric bisection.
template <bool EOS_TAIL>
__device__ double node_balance_temperature(const CellClosure<EOS_TAIL>& cl,
                                           const PlanckTableDeviceView& planck,
                                           const double* __restrict__ sigma_pe_c, const int G,
                                           const double e_n, const double A, const double dt,
                                           const double T_guess, double* __restrict__ sh_b,
                                           double* __restrict__ sh_db, const int lane) {
  const double k_emit = dt * core::constants::c_light * core::constants::a_eV;
  // Called by the whole warp with the same arguments: the group fractions
  // in parallel over the lanes, the sums in the group order by every lane.
  const auto eval = [&](const double T, double& fp) {
    const PlanckTableDeviceView::Location loc = planck.locate_b(T);
    const double T3 = T * T * T;
    const double T4 = T3 * T;
    for (int g = lane; g < G; g += kWarp) {
      double b = 1.0;
      double db = 0.0;
      if (G != 1) {
        planck.interpolate_b_and_dT(g, loc, T, b, db);
        b = fmax(b, 0.0);
      }
      sh_b[g] = b;
      sh_db[g] = db;
    }
    __syncwarp();
    double P = 0.0;
    double dP = 0.0;
    for (int g = 0; g < G; ++g) {
      const double b = sh_b[g];
      const double db = sh_db[g];
      const double sg = sigma_pe_c[g];
      P += sg * b;
      dP += sg * fmax(4.0 * T3 * b + T4 * db, 0.0);
    }
    __syncwarp();
    fp = cl.rho * cl.heat_capacity(T) + k_emit * dP;
    return cl.rho * (cl.energy(T) - e_n) + k_emit * T4 * P - dt * A;
  };
  double fp = 0.0;
  double lo = cl.T_floor;
  if (!(eval(lo, fp) < 0.0)) {
    return lo;
  }
  double hi = fmax(cl.temperature_warp(e_n + dt * fmax(A, 0.0) / cl.rho, lane), lo);
  double f_hi = eval(hi, fp);
  for (int i = 0; i < 64 && f_hi < 0.0; ++i) {
    hi *= 2.0;
    f_hi = eval(hi, fp);
  }
  if (f_hi < 0.0) {
    return hi;
  }
  double T = (T_guess > lo && T_guess < hi) ? T_guess : sqrt(fmax(lo, kLdTiny) * hi);
  for (int it = 0; it < 100; ++it) {
    const double f = eval(T, fp);
    if (f < 0.0) {
      lo = T;
    } else {
      hi = T;
    }
    if (f == 0.0 || hi - lo <= 1.0e-14 * hi) {
      break;
    }
    double Tn = (fp > 0.0) ? T - f / fp : 0.0;
    if (!(Tn > lo && Tn < hi)) {
      Tn = sqrt(fmax(lo, kLdTiny) * hi);
    }
    const bool done = fabs(Tn - T) <= 1.0e-14 * T;
    T = Tn;
    if (done) {
      break;
    }
  }
  return T;
}

// Nodal matter update from the transport's own absorption and the emission
// that entered the final sweep: e = e^n + dt (A - sum_g (fixed_g + kappa_g x))
// / rho (e_acc: the energy-conserving state of this iterate; a node below the
// floor energy is raised to it and flagged). The next linearization point is
// the node's own balance temperature with this absorption, approached by a
// fraction omega of the step: omega halves (down to 1/16) when the step
// reverses the previous one without shrinking below half of it, and doubles
// back towards 1 otherwise — the cell opacities are held at the current
// point, and where they vary steeply with the temperature the undamped point
// alternates between two values (a 643 eV corona cell under a table opacity,
// measured); a converging iteration whose steps alternate in sign while
// shrinking is left alone. dT_rel is the larger of the undamped step and the
// conserving state's change, relative.
template <bool EOS_TAIL>
__global__ void matter_update_kernel(const ClosureArgs a,
                                     const std::uint8_t* __restrict__ is_void,
                                     PlanckTableDeviceView planck,
                                     const double* __restrict__ sigma_a,
                                     const double* __restrict__ sigma_pe,
                                     const double* __restrict__ phi,
                                     const double* __restrict__ fixed,
                                     const double* __restrict__ kappa,
                                     const double* __restrict__ x,
                                     const double* __restrict__ e_n,
                                     double* __restrict__ e_acc,
                                     double* __restrict__ e_k,
                                     double* __restrict__ T_k,
                                     double* __restrict__ dT_rel,
                                     double* __restrict__ step_prev,
                                     double* __restrict__ omega,
                                     double* __restrict__ clip,
                                     int* __restrict__ flag,
                                     const int n,
                                     const int G,
                                     const double dt) {
  // One warp per node (the balance temperature's group sums in parallel
  // over the lanes, 2 G doubles of shared memory per warp); lane 0 writes.
  extern __shared__ double warp_shared[];
  const int lane = static_cast<int>(threadIdx.x) & (kWarp - 1);
  const int warp = static_cast<int>(threadIdx.x) / kWarp;
  const int node = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) / kWarp) + warp;
  const int two_n = 2 * n;
  if (node >= two_n) {
    return;
  }
  double* sh_b = warp_shared + static_cast<std::size_t>(warp) * 2 * static_cast<std::size_t>(G);
  double* sh_db = sh_b + G;
  const int c = node >> 1;
  if (cell_void(is_void, c)) {
    if (lane == 0) {
      clip[node] = 0.0;
      dT_rel[node] = 0.0;
      e_acc[node] = e_n[node];
    }
    return;
  }
  double A = 0.0;
  double eps = 0.0;
  const double xv = x[node];
  for (int g = 0; g < G; ++g) {
    const std::size_t gi = static_cast<std::size_t>(g) * two_n + node;
    A += sigma_a[static_cast<std::size_t>(c) * G + g] * phi[gi];
    eps += fixed[gi] + kappa[gi] * xv;
  }
  const CellClosure<EOS_TAIL> cl(a, c);
  const double e_start = e_n[node];
  double e_new = e_start + dt * (A - eps) / cl.rho;
  const double e_min = cl.energy(a.T_floor);
  double clipped = 0.0;
  bool floored = false;
  if (!(e_new >= e_min)) {
    clipped = isfinite(e_new) ? cl.rho * (e_min - e_new) : 0.0;
    e_new = e_min;
    floored = true;
  }
  const double T_cons = cl.temperature_warp(e_new, lane);
  const double T_old = fmax(T_k[node], a.T_floor);
  const double om_prev = omega[node];
  const double step_before = step_prev[node];
  // A negative absorption rate (a linearization far from the solution makes
  // the emission of a neighbour negative) has no physical balance: the
  // balance takes max(A, 0), which bounds the point from below by the
  // node's pure emission cooling. The point moves by at most a factor
  // kPointStep per iteration: the group opacities are held at the current
  // point, and far from it the balance with them overshoots.
  const double T_bal = node_balance_temperature(
      cl, planck, sigma_pe + static_cast<std::size_t>(c) * G, G, e_start, fmax(A, 0.0), dt,
      T_cons, sh_b, sh_db, lane);
  const double step = fmin(fmax(T_bal, T_old / kPointStep), T_old * kPointStep) - T_old;
  double om = om_prev;
  const bool alternating =
      step * step_before < 0.0 && fabs(step) > 0.5 * fabs(step_before);
  om = alternating ? fmax(0.5 * om, 1.0 / 16.0) : fmin(2.0 * om, 1.0);
  const double T_next = fmax(T_old + om * step, a.T_floor);
  __syncwarp();
  if (lane != 0) {
    return;
  }
  clip[node] = clipped;
  if (floored) {
    atomicOr(flag, kRetryFloor);
  }
  omega[node] = om;
  step_prev[node] = step;
  dT_rel[node] = fmax(fabs(step), fabs(T_cons - T_old)) / fmax(fabs(T_old + step), a.T_floor);
  e_acc[node] = e_new;
  e_k[node] = cl.energy(T_next);
  T_k[node] = T_next;
}

// Accepted cell state: ee the lumped-mass mean of the nodal energies, the
// carried offset e_R - e_L, Te = T(ee), Pe, and the last nodal temperature
// change of the cell.
template <bool EOS_TAIL>
__global__ void finalize_cell_kernel(const ClosureArgs a,
                                     const double* __restrict__ cells,
                                     const std::uint8_t* __restrict__ is_void,
                                     const double* __restrict__ e_acc,
                                     const double* __restrict__ dT_rel,
                                     double* __restrict__ ee,
                                     double* __restrict__ Te,
                                     double* __restrict__ Pe,
                                     double* __restrict__ offset,
                                     double* __restrict__ delta_T,
                                     const int n) {
  // One warp per cell; lane 0 writes.
  const int lane = static_cast<int>(threadIdx.x) & (kWarp - 1);
  const int c = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) / kWarp) +
                static_cast<int>(threadIdx.x) / kWarp;
  if (c >= n) {
    return;
  }
  const double* cell = cells + static_cast<std::size_t>(kCellStride) * c;
  const double V = cell[1] + cell[2];
  if (cell_void(is_void, c) || !(V > 0.0)) {
    if (lane == 0) {
      offset[c] = 0.0;
      if (delta_T != nullptr) {
        delta_T[c] = 0.0;
      }
    }
    return;
  }
  const double eL = e_acc[2 * c];
  const double eR = e_acc[2 * c + 1];
  const double ebar = (cell[1] * eL + cell[2] * eR) / V;
  const CellClosure<EOS_TAIL> cl(a, c);
  const double T = cl.temperature_warp(ebar, lane);
  if (lane != 0) {
    return;
  }
  ee[c] = ebar;
  offset[c] = eR - eL;
  Te[c] = T;
  if (Pe != nullptr) {
    Pe[c] = cl.pressure(T, ebar);
  }
  if (delta_T != nullptr) {
    delta_T[c] = fmax(dT_rel[2 * c], dT_rel[2 * c + 1]);
  }
}

// Per (cell, group): rad_E the lumped-mass mean of the nodal energy density
// (phi_cell the same mean of the scalar flux, c rad_E, the cell-average
// transport moment of the older schemes' sn_phi_old),
// rad_dep / rad_emit the absorbed / emitted energy of the step [erg], the
// radial pressure and Eddington factor, the absorbed and emitted energy
// densities, the energy density the temperature floor added (group 0), and
// a count of negative cell radiation energies.
__global__ void finalize_group_kernel(const double* __restrict__ cells,
                                      const double* __restrict__ sigma_a,
                                      const double* __restrict__ phi,
                                      const double* __restrict__ fixed,
                                      const double* __restrict__ kappa,
                                      const double* __restrict__ x,
                                      const double* __restrict__ prr_phi,
                                      const double* __restrict__ clip,
                                      double* __restrict__ rad_E,
                                      double* __restrict__ phi_cell,
                                      double* __restrict__ rad_dep,
                                      double* __restrict__ rad_emit,
                                      double* __restrict__ Prr,
                                      double* __restrict__ chi,
                                      double* __restrict__ diag_absorption,
                                      double* __restrict__ diag_emission,
                                      double* __restrict__ diag_clip,
                                      int* __restrict__ negative_count,
                                      const int n,
                                      const int G,
                                      const double dt) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n * G) {
    return;
  }
  const int c = idx / G;
  const int g = idx - c * G;
  const double* cell = cells + static_cast<std::size_t>(kCellStride) * c;
  const double ML = cell[1];
  const double MR = cell[2];
  const double V = ML + MR;
  const std::size_t two_n = 2 * static_cast<std::size_t>(n);
  const std::size_t gi = static_cast<std::size_t>(g) * two_n + 2 * c;
  const double cl_light = core::constants::c_light;
  const double mphi = ML * phi[gi] + MR * phi[gi + 1];
  const double E = (V > 0.0) ? mphi / (cl_light * V) : 0.0;
  const double eL = fixed[gi] + kappa[gi] * x[2 * c];
  const double eR = fixed[gi + 1] + kappa[gi + 1] * x[2 * c + 1];
  const double dep = dt * sigma_a[idx] * mphi;
  const double emit = dt * (ML * eL + MR * eR);
  rad_E[idx] = E;
  if (phi_cell != nullptr) {
    phi_cell[idx] = (V > 0.0) ? mphi / V : 0.0;
  }
  rad_dep[idx] = dep;
  rad_emit[idx] = emit;
  const double P = prr_phi[idx] / cl_light;
  Prr[idx] = P;
  const double ratio = P / fmax(E, 1.0e-300);
  chi[idx] = (isfinite(ratio) && E > 0.0) ? fmin(fmax(ratio, 0.0), 1.0) : (1.0 / 3.0);
  if (V > 0.0) {
    diag_absorption[idx] = dep / V;
    diag_emission[idx] = emit / V;
    diag_clip[idx] = (g == 0) ? (ML * clip[2 * c] + MR * clip[2 * c + 1]) / V : 0.0;
  }
  if (E < 0.0) {
    atomicAdd(negative_count, 1);
  }
}

// Face currents [g * (n + 1) + f] into the S_N face-flux arrays [f * G + g]
// (the linear-discontinuous face flux is the transport's own; no blending,
// limiting or diffusion flux).
__global__ void face_flux_kernel(const double* __restrict__ F,
                                 double* __restrict__ raw,
                                 double* __restrict__ limited,
                                 double* __restrict__ blended,
                                 double* __restrict__ diff,
                                 double* __restrict__ alpha,
                                 const int n,
                                 const int G) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (n + 1) * G) {
    return;
  }
  const int f = idx / G;
  const int g = idx - f * G;
  const double v = F[static_cast<std::size_t>(g) * (n + 1) + f];
  raw[idx] = v;
  limited[idx] = v;
  blended[idx] = v;
  diff[idx] = 0.0;
  alpha[idx] = 0.0;
}

// sum_g F_g at the outer face.
__global__ void outer_current_kernel(const double* __restrict__ F, double* __restrict__ out,
                                     const int n, const int G) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  double s = 0.0;
  for (int g = 0; g < G; ++g) {
    s += F[static_cast<std::size_t>(g) * (n + 1) + n];
  }
  out[0] = s;
}

double* scratch_doubles(const char* tag, const std::size_t count) {
  return static_cast<double*>(
      core::device_scratch_acquire(tag, std::max<std::size_t>(count, 1) * sizeof(double)));
}

inline int grid_for(const std::size_t count) {
  return static_cast<int>((count + kBlock - 1) / kBlock);
}

// One-warp-per-cell kernels (matter setup, cell temperature, accepted state):
// four warps per block.
constexpr int kCellWarpThreads = 4 * kWarp;
inline int cell_warp_blocks(const int n) {
  return (n + 3) / 4;
}

// Launch shape of the one-warp-per-node kernels (linearize, matter update):
// up to four warps per block, per_group doubles of shared memory per group
// and warp.
struct NodeWarpLaunch {
  int blocks = 0;
  int threads = 0;
  std::size_t shared = 0;
};

template <typename Kernel>
NodeWarpLaunch node_warp_launch(Kernel kernel, const std::size_t nodes, const int G,
                                const int per_group) {
  const std::size_t per_warp = static_cast<std::size_t>(per_group) *
                               static_cast<std::size_t>(std::max(G, 1)) * sizeof(double);
  constexpr std::size_t kDefaultShared = 48U * 1024U;
  int warps = 4;
  while (warps > 1 && per_warp * static_cast<std::size_t>(warps) > kDefaultShared) {
    --warps;
  }
  NodeWarpLaunch launch;
  launch.shared = per_warp * static_cast<std::size_t>(warps);
  if (launch.shared > kDefaultShared) {
    cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(launch.shared)),
               "sn_ld per-node shared memory attribute failed (too many groups)");
  }
  launch.threads = warps * kWarp;
  launch.blocks = static_cast<int>((nodes + static_cast<std::size_t>(warps) - 1) /
                                   static_cast<std::size_t>(warps));
  return launch;
}

template <bool EOS_TAIL>
void advance_step_impl(core::State& state, const core::Config& cfg, const PlanckTable& planck,
                       const core::Config::MaterialsConfig::MatDef& mat, const double dt,
                       const double* outer_psi_in, const double* source_ext) {
  const int n = static_cast<int>(state.rho.size());
  const int G = std::max(cfg.radiation.groups, 1);
  const auto& sn = cfg.radiation.sn_transport;
  const int N = std::max(sn.n_angles, 2);
  const int geom = state.mesh.geometry_code;
  const bool curved = geom != static_cast<int>(mesh::Geometry1D::kPlanar);
  TENRYU_ASSERT(n > 0 && state.x_r.size() == static_cast<std::size_t>(n + 1),
                "sn_ld advance_step requires n_cells > 0 and n_cells + 1 node radii");
  TENRYU_ASSERT(state.ee.size() == static_cast<std::size_t>(n) &&
                    state.Te.size() == static_cast<std::size_t>(n),
                "sn_ld advance_step requires the electron energy and temperature fields");
  const QuadratureView quad = sn1d_internal::quadrature(N, geom);
  const int n_chains = quad.n_chains;
  const std::size_t two_n = 2 * static_cast<std::size_t>(n);
  const std::size_t n_node_groups = static_cast<std::size_t>(G) * two_n;
  const std::size_t n_cell_groups = static_cast<std::size_t>(n) * G;
  const std::size_t n_psi = n_node_groups * static_cast<std::size_t>(N);
  const std::size_t n_sd = curved ? n_node_groups * static_cast<std::size_t>(n_chains) : 0U;
  const double inv_cdt = 1.0 / (core::constants::c_light * dt);
  const double T_floor = cfg.numerics.floors.Te;
  const int max_outer = std::max(sn.max_outer_iterations, 1);
  const int max_inner = std::max(sn.max_inner_iterations, 1);
  const double outer_tol = std::max(sn.outer_tol, 10.0 * sn.outer_tol_hydro_error_scale);
  const double inner_tol = sn.inner_tol;
  // The scattering converged below the GMRES tolerance, so that the
  // homogeneous transport the Krylov iteration applies is linear to within
  // its own tolerance.
  const double scatter_tol = 0.1 * inner_tol;

  // Grey preconditioner of the emission GMRES (Radiation.sn_transport.
  // grey_preconditioner): "auto" applies it in a Newton iteration only when the
  // largest local gain of the grey correction exceeds kGreyPreconditionerGain.
  // Below it the unpreconditioned coupling contracts a smooth mode by at most
  // gain / (1 + gain) < 0.091 per Krylov step (less where radiation leaks), so
  // the correction saves less than one transport sweep per Newton iteration,
  // while its P1 factorization and solves are serial block eliminations over
  // the cells (GXII S_N late: 3.8 ms of 20 ms per step for 0.3 Krylov steps per
  // Newton iteration). Read-only log per Newton iteration (gain, Krylov
  // iterations, residual ratios): TENRYU_SN_LD_PRECOND_AUDIT=1.
  constexpr double kGreyPreconditionerGain = 0.1;
  const std::string& precond_mode = sn.grey_preconditioner;
  static const bool precond_audit = [] {
    const char* v = std::getenv("TENRYU_SN_LD_PRECOND_AUDIT");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  static bool logged_scheme = false;
  if (!logged_scheme) {
    logged_scheme = true;
    core::log_info(
        "SN 1D: linear-discontinuous scheme (nodal intensities and electron temperatures, "
        "emission Newton with GMRES on the absorption-rate density)");
  }

  // Geometry of the current mesh.
  double* cells = scratch_doubles("sn_ld:cells", static_cast<std::size_t>(kCellStride) * n);
  compute_cells(state.x_r.data(), n, geom, cells);

  // Histories: sized for this scheme, seeded after a size change or a remap.
  bool seed = state.sn_psi_prev.size() != n_psi || state.holo_ale_invalidated;
  if (state.sn_psi_prev.size() != n_psi) {
    state.sn_psi_prev.reset(n_psi);
  }
  if (curved) {
    if (state.sn_psi_sd_prev.size() != n_sd) {
      state.sn_psi_sd_prev.reset(n_sd);
      seed = true;
    }
  } else if (state.sn_psi_sd_prev.size() != 0U) {
    state.sn_psi_sd_prev.reset(0);
  }
  const bool reset_offset = state.sn_ee_node_offset.size() != static_cast<std::size_t>(n) ||
                            state.holo_ale_invalidated;
  if (state.sn_ee_node_offset.size() != static_cast<std::size_t>(n)) {
    state.sn_ee_node_offset.reset(static_cast<std::size_t>(n));
  }
  if (seed && state.step > 0) {
    core::log_warning("SN linear-discontinuous histories reseeded isotropic at step " +
                      std::to_string(state.step) +
                      " (mesh size change, remap or a checkpoint without them)");
  }
  double* phi_hist = scratch_doubles("sn_ld:phi_hist", n_node_groups);
  history_kernel<<<grid_for(n_cell_groups), kBlock>>>(
      cells, state.rad_E_old.data(), quad.weight, state.sn_psi_prev.data(),
      curved ? state.sn_psi_sd_prev.data() : nullptr, phi_hist, n, G, N, n_chains, seed ? 1 : 0);
  cuda_check(cudaGetLastError(), "sn_ld history launch failed");

  // Void mask on the device.
  std::uint8_t* d_void = nullptr;
  if (state.cell_is_void.size() == static_cast<std::size_t>(n) &&
      std::any_of(state.cell_is_void.begin(), state.cell_is_void.end(),
                  [](const std::uint8_t v) { return v != 0U; })) {
    d_void = static_cast<std::uint8_t*>(
        core::device_scratch_acquire("sn_ld:void", static_cast<std::size_t>(n)));
    cuda_check(cudaMemcpy(d_void, state.cell_is_void.data(), static_cast<std::size_t>(n),
                          cudaMemcpyHostToDevice),
               "sn_ld void mask upload failed");
  }

  // Closure inputs and the nodal matter state.
  double* e_start = scratch_doubles("sn_ld:e_start", static_cast<std::size_t>(n));
  double* T_start = scratch_doubles("sn_ld:T_start", static_cast<std::size_t>(n));
  cuda_check(cudaMemcpy(e_start, state.ee.data(), sizeof(double) * n, cudaMemcpyDeviceToDevice),
             "sn_ld ee snapshot failed");
  cuda_check(cudaMemcpy(T_start, state.Te.data(), sizeof(double) * n, cudaMemcpyDeviceToDevice),
             "sn_ld Te snapshot failed");
  state.ensure_cell_material_props(cfg);
  ClosureArgs closure;
  if (mat.hydro_eos_backend != "exact_ideal_gas") {
    closure.run_eos = sn_electron_eos_device_view(mat.eos_tables.get());
  }
  closure.cell_tables = cell_electron_table_selector_1d(state, cfg, n);
  closure.rho = state.rho.data();
  closure.zbar = state.zbar.data();
  closure.cv_e =
      (state.cv_e.size() == static_cast<std::size_t>(n)) ? state.cv_e.data() : nullptr;
  closure.A_eff = state.A_eff.data();
  closure.gamma_eff = state.gamma_eff.data();
  closure.e_start = e_start;
  closure.T_start = T_start;
  closure.cv_e_override = mat.cv_e_override;
  closure.T_floor = T_floor;
  closure.low_density_extrap = cfg.materials.low_density_extrapolation ? 1 : 0;

  double* e_n = scratch_doubles("sn_ld:e_n", two_n);
  double* e_k = scratch_doubles("sn_ld:e_k", two_n);
  double* e_acc = scratch_doubles("sn_ld:e_acc", two_n);
  double* T_k = scratch_doubles("sn_ld:T_k", two_n);
  double* dT_rel = scratch_doubles("sn_ld:dT_rel", two_n);
  double* step_prev = scratch_doubles("sn_ld:step_prev", two_n);
  double* omega = scratch_doubles("sn_ld:omega", two_n);
  cuda_check(cudaMemset(step_prev, 0, two_n * sizeof(double)), "sn_ld step reset failed");
  fill_kernel<<<grid_for(two_n), kBlock>>>(omega, 1.0, static_cast<int>(two_n));
  cuda_check(cudaGetLastError(), "sn_ld relaxation reset launch failed");
  double* clip = scratch_doubles("sn_ld:clip", two_n);
  matter_setup_kernel<EOS_TAIL><<<cell_warp_blocks(n), kCellWarpThreads>>>(
      closure, cells, d_void, state.ee.data(), state.sn_ee_node_offset.data(),
      reset_offset ? 1 : 0, e_n, e_k, T_k, n);
  cuda_check(cudaGetLastError(), "sn_ld matter setup launch failed");

  // Work arrays.
  double* sigma_t = scratch_doubles("sn_ld:sigma_t", n_cell_groups);
  double* sigma_e_dsa = scratch_doubles("sn_ld:sigma_e_dsa", n_cell_groups);
  double* fixed = scratch_doubles("sn_ld:fixed", n_node_groups);
  double* kappa = scratch_doubles("sn_ld:kappa", n_node_groups);
  double* kbar = scratch_doubles("sn_ld:kbar", two_n);
  double* sabar = scratch_doubles("sn_ld:sabar", two_n);
  double* se_node = scratch_doubles("sn_ld:se_node", two_n);
  double* rem_node = scratch_doubles("sn_ld:rem_node", two_n);
  double* scale = scratch_doubles("sn_ld:scale", two_n);
  double* x = scratch_doubles("sn_ld:x", two_n);
  double* p1_se = scratch_doubles("sn_ld:p1_se", static_cast<std::size_t>(n));
  double* p1_sx = scratch_doubles("sn_ld:p1_sx", static_cast<std::size_t>(n));
  double* q = scratch_doubles("sn_ld:q", n_node_groups);
  double* psi_aff = scratch_doubles("sn_ld:psi_aff", n_psi);
  double* psi_hom = scratch_doubles("sn_ld:psi_hom", n_psi);
  double* sd_aff = curved ? scratch_doubles("sn_ld:sd_aff", n_sd) : nullptr;
  double* phi_aff = scratch_doubles("sn_ld:phi_aff", n_node_groups);
  double* phi_hom = scratch_doubles("sn_ld:phi_hom", n_node_groups);
  double* A_aff = scratch_doubles("sn_ld:A_aff", two_n);
  double* A_hom = scratch_doubles("sn_ld:A_hom", two_n);
  const int m_restart = std::max(1, std::min(kGmresRestart, max_inner));
  double* Vk = scratch_doubles("sn_ld:gmres_V", static_cast<std::size_t>(m_restart + 1) * two_n);
  double* Zk = scratch_doubles("sn_ld:gmres_Z", static_cast<std::size_t>(m_restart) * two_n);
  double* w = scratch_doubles("sn_ld:gmres_w", two_n);
  double* r = scratch_doubles("sn_ld:gmres_r", two_n);
  double* tmp1 = scratch_doubles("sn_ld:gmres_t1", two_n);
  double* tmp2 = scratch_doubles("sn_ld:gmres_t2", two_n);
  double* d_small = scratch_doubles("sn_ld:gmres_small", static_cast<std::size_t>(2 * m_restart + 8));
  double* d_small2 = scratch_doubles("sn_ld:gmres_small2", static_cast<std::size_t>(m_restart + 2));
  double* rhs1 = scratch_doubles("sn_ld:p1_rhs", two_n);
  double* Phi1 = scratch_doubles("sn_ld:p1_phi", two_n);
  // P1 factors: the grey preconditioner's (one system) and, with physical
  // scattering, the source iteration's acceleration (one system per group),
  // both factored once per Newton iteration.
  double* p1_pre = scratch_doubles("sn_ld:p1_pre", p1_scratch_doubles(n, 1));
  // The sweep's cell-matrix inverses, formed once per Newton iteration (the
  // cross sections are held during it).
  double* sweep_inverse =
      scratch_doubles("sn_ld:sweep_inverse", sweep_inverse_doubles(n, G, quad, geom));
  double* p1_dsa = nullptr;
  double* phi_s = scratch_doubles("sn_ld:phi_s", n_node_groups);
  double* rhs_s = scratch_doubles("sn_ld:rhs_s", n_node_groups);
  double* Phi_s = scratch_doubles("sn_ld:Phi_s", n_node_groups);
  double* d_abs = scratch_doubles("sn_ld:d_abs", n_node_groups);
  double* d_ref = scratch_doubles("sn_ld:d_ref", n_node_groups);
  int* d_flags = static_cast<int*>(core::device_scratch_acquire("sn_ld:flags", 4 * sizeof(int)));
  cuda_check(cudaMemset(d_flags, 0, 4 * sizeof(int)), "sn_ld flag reset failed");

  const int node_grid = grid_for(two_n);
  const int ng_grid = grid_for(n_node_groups);
  const int cg_grid = grid_for(n_cell_groups);

  auto reduce_max2 = [&](const double* a, const double* b, const std::size_t m, double out[2]) {
    max2_kernel<<<1, kReduceThreads>>>(a, b, d_small, static_cast<int>(m));
    cuda_check(cudaGetLastError(), "sn_ld max reduction launch failed");
    cuda_check(cudaMemcpy(out, d_small, 2 * sizeof(double), cudaMemcpyDeviceToHost),
               "sn_ld max reduction copy failed");
  };
  auto dots = [&](const double* V, const int nvec, const double* v, double* host_out) {
    dots_kernel<<<nvec, kReduceThreads>>>(V, two_n, v, d_small, static_cast<int>(two_n));
    cuda_check(cudaGetLastError(), "sn_ld dot launch failed");
    cuda_check(cudaMemcpy(host_out, d_small, static_cast<std::size_t>(nvec) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "sn_ld dot copy failed");
  };
  auto norm2 = [&](const double* v) {
    double h = 0.0;
    dots(v, 1, v, &h);
    return std::sqrt(std::max(h, 0.0));
  };

  bool has_scatter = false;
  int total_sweeps = 0;
  int retry_flags = 0;

  // Transport of every group for the per-angle source built from x (the
  // nodal absorption-rate density; times xscale when given): the affine
  // problem (the linearized emission's fixed part, the external source, the
  // histories, the outer inflow) or the homogeneous one (x only). Physical
  // scattering converged by source iteration with the consistent P1
  // correction. Writes psi_out (and sd_out), phi_out and A_out.
  auto transport = [&](const bool affine, const double* xv, const double* xscale, double* psi_out,
                       double* sd_out, double* phi_out, double* A_out, const double* phi_init) {
    const double* hist_psi = affine ? state.sn_psi_prev.data() : nullptr;
    const double* hist_sd = (affine && curved) ? state.sn_psi_sd_prev.data() : nullptr;
    const double* inflow = affine ? outer_psi_in : nullptr;
    const double* fx = affine ? fixed : nullptr;
    const double* sext = affine ? source_ext : nullptr;
    if (!has_scatter) {
      source_kernel<<<ng_grid, kBlock>>>(fx, kappa, xv, xscale, sext, state.sn_sigma_s.data(),
                                         nullptr, q, n, G);
      cuda_check(cudaGetLastError(), "sn_ld source launch failed");
      sweep_prepared(cells, sweep_inverse, q, hist_psi, hist_sd, inflow, quad, psi_out, sd_out, n,
                     G, inv_cdt, geom);
      moments(cells, psi_out, inflow, quad, phi_out, nullptr, nullptr, n, G);
      ++total_sweeps;
    } else {
      if (phi_init != nullptr) {
        cuda_check(cudaMemcpy(phi_s, phi_init, n_node_groups * sizeof(double),
                              cudaMemcpyDeviceToDevice),
                   "sn_ld scattering start copy failed");
      } else {
        cuda_check(cudaMemset(phi_s, 0, n_node_groups * sizeof(double)),
                   "sn_ld scattering start reset failed");
      }
      bool converged = false;
      for (int it = 0; it < max_inner; ++it) {
        source_kernel<<<ng_grid, kBlock>>>(fx, kappa, xv, xscale, sext, state.sn_sigma_s.data(),
                                           phi_s, q, n, G);
        cuda_check(cudaGetLastError(), "sn_ld source launch failed");
        sweep_prepared(cells, sweep_inverse, q, hist_psi, hist_sd, inflow, quad, psi_out, sd_out,
                       n, G, inv_cdt, geom);
        moments(cells, psi_out, inflow, quad, phi_out, nullptr, nullptr, n, G);
        ++total_sweeps;
        scatter_residual_kernel<<<ng_grid, kBlock>>>(state.sn_sigma_s.data(), phi_out, phi_s,
                                                     rhs_s, n, G);
        cuda_check(cudaGetLastError(), "sn_ld scattering residual launch failed");
        p1_apply(p1_dsa, rhs_s, Phi_s, n, G);
        scatter_update_kernel<<<ng_grid, kBlock>>>(state.sn_sigma_s.data(), phi_out, Phi_s, phi_s,
                                                   d_abs, d_ref, n, G);
        cuda_check(cudaGetLastError(), "sn_ld scattering update launch failed");
        double mx[2] = {0.0, 0.0};
        reduce_max2(d_abs, d_ref, n_node_groups, mx);
        if (!(mx[1] > 0.0) || mx[0] <= scatter_tol * mx[1]) {
          converged = true;
          break;
        }
      }
      if (!converged) {
        retry_flags |= kRetryScatter;
      }
    }
    absorption_kernel<<<node_grid, kBlock>>>(state.sn_sigma_a.data(), phi_out, A_out, n, G);
    cuda_check(cudaGetLastError(), "sn_ld absorption launch failed");
  };

  // Right-preconditioned GMRES on the scaled nodal absorption-rate density
  // y = A / s: (I - S^-1 K S) y = S^-1 b with K the homogeneous transport
  // response, b the affine one; the preconditioner is the grey low-order
  // correction z = v + sabar Phi / s, Phi from the consistent P1 system with
  // the source kbar s v; at most max_inner Krylov iterations per call.
  // Returns the residual relative to the scaled size of the solution; x
  // holds the solution and psi_aff / phi_aff / A_aff the transport of it.
  int gmres_iterations = 0;
  bool use_precond = true;
  std::vector<double> audit_ratios;
  double audit_beta0 = -1.0;
  auto solve_emission = [&](const double* phi_start, bool& converged) {
    converged = false;
    int its = 0;
    audit_ratios.clear();
    audit_beta0 = -1.0;
    std::vector<double> H(static_cast<std::size_t>(m_restart + 1) * m_restart, 0.0);
    std::vector<double> cs(static_cast<std::size_t>(m_restart), 0.0);
    std::vector<double> sn_rot(static_cast<std::size_t>(m_restart), 0.0);
    std::vector<double> gv(static_cast<std::size_t>(m_restart + 1), 0.0);
    std::vector<double> h1(static_cast<std::size_t>(m_restart + 2), 0.0);
    std::vector<double> h2(static_cast<std::size_t>(m_restart + 2), 0.0);
    std::vector<double> y(static_cast<std::size_t>(m_restart), 0.0);
    const auto Hij = [&](const int i, const int j) -> double& {
      return H[static_cast<std::size_t>(i) * m_restart + j];
    };
    transport(true, x, nullptr, psi_aff, sd_aff, phi_aff, A_aff, phi_start);
    double ref = -1.0;
    double target = 0.0;
    double res = std::numeric_limits<double>::infinity();
    while (true) {
      residual_kernel<<<node_grid, kBlock>>>(A_aff, x, scale, r, tmp1, tmp2,
                                             static_cast<int>(two_n));
      cuda_check(cudaGetLastError(), "sn_ld residual launch failed");
      const double beta = norm2(r);
      if (ref < 0.0) {
        ref = std::max(norm2(tmp1), norm2(tmp2));
        if (!(ref > 0.0) || !std::isfinite(ref)) {
          ref = 1.0;
        }
        // Reduce this Newton iteration's initial residual by inner_tol (so
        // that every iteration is a Newton step, also when the previous
        // solution already satisfies the new system loosely), down to the
        // rounding level of the solution.
        target = std::max(inner_tol * beta, 1.0e-14 * ref);
      }
      res = beta / ref;
      if (audit_beta0 < 0.0) {
        audit_beta0 = (beta > 0.0) ? beta : 1.0;
      }
      if (beta <= target) {
        converged = true;
        break;
      }
      if (!std::isfinite(res) || its >= max_inner) {
        break;
      }
      scale_vector_kernel<<<node_grid, kBlock>>>(r, 1.0 / beta, Vk, static_cast<int>(two_n));
      cuda_check(cudaGetLastError(), "sn_ld Krylov start launch failed");
      std::fill(gv.begin(), gv.end(), 0.0);
      gv[0] = beta;
      int jj = 0;
      for (int j = 0; j < m_restart && its < max_inner; ++j) {
        double* vj = Vk + static_cast<std::size_t>(j) * two_n;
        double* zj = Zk + static_cast<std::size_t>(j) * two_n;
        if (!use_precond) {
          cuda_check(cudaMemcpyAsync(zj, vj, two_n * sizeof(double), cudaMemcpyDeviceToDevice),
                     "sn_ld unpreconditioned copy failed");
        } else {
          precond_rhs_kernel<<<node_grid, kBlock>>>(kbar, scale, vj, rhs1,
                                                    static_cast<int>(two_n));
          cuda_check(cudaGetLastError(), "sn_ld preconditioner source launch failed");
          p1_apply(p1_pre, rhs1, Phi1, n, 1);
          precond_apply_kernel<<<node_grid, kBlock>>>(vj, sabar, Phi1, scale, zj,
                                                      static_cast<int>(two_n));
          cuda_check(cudaGetLastError(), "sn_ld preconditioner launch failed");
        }
        transport(false, zj, scale, psi_hom, nullptr, phi_hom, A_hom, nullptr);
        matvec_finish_kernel<<<node_grid, kBlock>>>(zj, A_hom, scale, w, static_cast<int>(two_n));
        cuda_check(cudaGetLastError(), "sn_ld operator launch failed");
        // Classical Gram-Schmidt with one reorthogonalization, both passes on
        // the device (each pass: the projections on V_0..V_j and |w|^2 in one
        // reduction, then w -= V h from the device coefficients); one copy of
        // both passes' coefficients to the host. The norm of the rest from
        // |w|^2 - |h|^2 of the second pass.
        const int nv = j + 1;
        dots_norm_kernel<<<nv + 1, kReduceThreads>>>(Vk, two_n, w, d_small, nv,
                                                     static_cast<int>(two_n));
        cuda_check(cudaGetLastError(), "sn_ld projection launch failed");
        orth_update_kernel<<<node_grid, kBlock>>>(Vk, two_n, d_small, nv, w,
                                                  static_cast<int>(two_n));
        cuda_check(cudaGetLastError(), "sn_ld orthogonalization launch failed");
        dots_norm_kernel<<<nv + 1, kReduceThreads>>>(Vk, two_n, w, d_small2, nv,
                                                     static_cast<int>(two_n));
        cuda_check(cudaGetLastError(), "sn_ld projection launch failed");
        orth_update_kernel<<<node_grid, kBlock>>>(Vk, two_n, d_small2, nv, w,
                                                  static_cast<int>(two_n));
        cuda_check(cudaGetLastError(), "sn_ld orthogonalization launch failed");
        cuda_check(cudaMemcpy(h1.data(), d_small, static_cast<std::size_t>(nv + 1) * sizeof(double),
                              cudaMemcpyDeviceToHost),
                   "sn_ld projection copy failed");
        cuda_check(cudaMemcpy(h2.data(), d_small2, static_cast<std::size_t>(nv + 1) * sizeof(double),
                              cudaMemcpyDeviceToHost),
                   "sn_ld projection copy failed");
        double proj2 = 0.0;
        for (int i = 0; i < nv; ++i) {
          Hij(i, j) = h1[static_cast<std::size_t>(i)] + h2[static_cast<std::size_t>(i)];
          proj2 += h2[static_cast<std::size_t>(i)] * h2[static_cast<std::size_t>(i)];
        }
        const double hn = std::sqrt(std::max(h2[static_cast<std::size_t>(nv)] - proj2, 0.0));
        Hij(j + 1, j) = hn;
        if (hn > 0.0 && std::isfinite(hn)) {
          scale_vector_kernel<<<node_grid, kBlock>>>(
              w, 1.0 / hn, Vk + static_cast<std::size_t>(j + 1) * two_n, static_cast<int>(two_n));
          cuda_check(cudaGetLastError(), "sn_ld Krylov vector launch failed");
        }
        for (int i = 0; i < j; ++i) {
          const double a0 = Hij(i, j);
          const double a1 = Hij(i + 1, j);
          Hij(i, j) = cs[static_cast<std::size_t>(i)] * a0 + sn_rot[static_cast<std::size_t>(i)] * a1;
          Hij(i + 1, j) =
              -sn_rot[static_cast<std::size_t>(i)] * a0 + cs[static_cast<std::size_t>(i)] * a1;
        }
        const double a0 = Hij(j, j);
        const double a1 = Hij(j + 1, j);
        const double rr = std::hypot(a0, a1);
        const double cj = (rr > 0.0) ? a0 / rr : 1.0;
        const double sj = (rr > 0.0) ? a1 / rr : 0.0;
        cs[static_cast<std::size_t>(j)] = cj;
        sn_rot[static_cast<std::size_t>(j)] = sj;
        Hij(j, j) = rr;
        Hij(j + 1, j) = 0.0;
        gv[static_cast<std::size_t>(j + 1)] = -sj * gv[static_cast<std::size_t>(j)];
        gv[static_cast<std::size_t>(j)] = cj * gv[static_cast<std::size_t>(j)];
        if (precond_audit && audit_ratios.size() < 8U) {
          audit_ratios.push_back(std::fabs(gv[static_cast<std::size_t>(j + 1)]) / audit_beta0);
        }
        ++gmres_iterations;
        ++its;
        jj = j + 1;
        if (std::fabs(gv[static_cast<std::size_t>(j + 1)]) <= target || !(hn > 0.0) ||
            !std::isfinite(hn)) {
          break;
        }
      }
      if (jj == 0) {
        break;
      }
      for (int i = jj - 1; i >= 0; --i) {
        double acc = gv[static_cast<std::size_t>(i)];
        for (int k = i + 1; k < jj; ++k) {
          acc -= Hij(i, k) * y[static_cast<std::size_t>(k)];
        }
        const double d = Hij(i, i);
        y[static_cast<std::size_t>(i)] = (d != 0.0) ? acc / d : 0.0;
      }
      cuda_check(cudaMemcpy(d_small + m_restart + 2, y.data(),
                            static_cast<std::size_t>(jj) * sizeof(double), cudaMemcpyHostToDevice),
                 "sn_ld solution coefficient upload failed");
      solution_update_kernel<<<node_grid, kBlock>>>(Zk, two_n, d_small + m_restart + 2, jj, scale,
                                                    x, static_cast<int>(two_n));
      cuda_check(cudaGetLastError(), "sn_ld solution update launch failed");
      transport(true, x, nullptr, psi_aff, sd_aff, phi_aff, A_aff, phi_aff);
    }
    return res;
  };

  // Newton iteration on the nodal temperatures.
  bool converged = false;
  int outer_done = 0;
  double outer_res = std::numeric_limits<double>::infinity();
  double gmres_res = std::numeric_limits<double>::infinity();
  for (int k = 0; k < max_outer; ++k) {
    if (k > 0) {
      cell_temperature_kernel<EOS_TAIL><<<cell_warp_blocks(n), kCellWarpThreads>>>(
          closure, cells, d_void, e_k, state.Te.data(), n);
      cuda_check(cudaGetLastError(), "sn_ld cell temperature launch failed");
    }
    sn1d_internal::evaluate_opacity(state, cfg, planck, mat, n, G, dt);
    cuda_check(cudaMemset(d_flags + 1, 0, sizeof(int)), "sn_ld scattering flag reset failed");
    opacity_post_kernel<<<cg_grid, kBlock>>>(d_void, state.sn_sigma_a.data(),
                                             state.sn_sigma_pe.data(), state.sn_sigma_s.data(),
                                             sigma_t, sigma_e_dsa, d_flags + 1, n, G, inv_cdt);
    cuda_check(cudaGetLastError(), "sn_ld opacity post-processing launch failed");
    int scatter_flag = 0;
    cuda_check(cudaMemcpy(&scatter_flag, d_flags + 1, sizeof(int), cudaMemcpyDeviceToHost),
               "sn_ld scattering flag copy failed");
    has_scatter = scatter_flag != 0;
    sweep_prepare(cells, sigma_t, quad, sweep_inverse, n, G, inv_cdt, geom);
    if (has_scatter) {
      if (p1_dsa == nullptr) {
        p1_dsa = scratch_doubles("sn_ld:p1_dsa", p1_scratch_doubles(n, G));
      }
      p1_factor(cells, sigma_e_dsa, state.sn_sigma_s.data(), quad, p1_dsa, n, G, geom);
    }
    const double* phi_ref = (k == 0) ? phi_hist : phi_aff;
    const NodeWarpLaunch lin_launch = node_warp_launch(linearize_kernel<EOS_TAIL>, two_n, G, 3);
    linearize_kernel<EOS_TAIL><<<lin_launch.blocks, lin_launch.threads, lin_launch.shared>>>(
        closure, d_void, planck.device_view(), state.sn_sigma_a.data(), state.sn_sigma_pe.data(),
        state.sn_sigma_s.data(), e_n, e_k, T_k, phi_ref, fixed, kappa, kbar, sabar, se_node,
        rem_node, scale, x, n, G, dt, inv_cdt);
    cuda_check(cudaGetLastError(), "sn_ld linearization launch failed");
    double gain_max = 0.0;
    if (precond_mode == "auto" || precond_audit) {
      precond_gain_max_kernel<<<1, kReduceThreads>>>(kbar, sabar, rem_node, d_small,
                                                     static_cast<int>(two_n));
      cuda_check(cudaGetLastError(), "sn_ld preconditioner gain launch failed");
      cuda_check(cudaMemcpy(&gain_max, d_small, sizeof(double), cudaMemcpyDeviceToHost),
                 "sn_ld preconditioner gain copy failed");
    }
    use_precond = (precond_mode == "on") ||
                  (precond_mode == "auto" && !(gain_max <= kGreyPreconditionerGain));
    if (use_precond) {
      lmfg_cell_kernel<<<grid_for(static_cast<std::size_t>(n)), kBlock>>>(cells, se_node,
                                                                         rem_node, p1_se,
                                                                         p1_sx, n);
      cuda_check(cudaGetLastError(), "sn_ld low-order coefficient launch failed");
      p1_factor(cells, p1_se, p1_sx, quad, p1_pre, n, 1, geom);
    }
    const int gmres_before = gmres_iterations;
    bool gmres_converged = false;
    gmres_res = solve_emission(phi_ref, gmres_converged);
    if (precond_audit) {
      std::ostringstream os;
      os << "[sn_ld_precond] step=" << state.step << " newton=" << k
         << " precond=" << (use_precond ? "on" : "off") << " gain_max=" << gain_max
         << " krylov=" << (gmres_iterations - gmres_before) << " ratios=";
      for (const double r : audit_ratios) {
        os << r << ",";
      }
      core::log_info(os.str());
    }
    const NodeWarpLaunch mu_launch = node_warp_launch(matter_update_kernel<EOS_TAIL>, two_n, G, 2);
    matter_update_kernel<EOS_TAIL><<<mu_launch.blocks, mu_launch.threads, mu_launch.shared>>>(
        closure, d_void, planck.device_view(), state.sn_sigma_a.data(), state.sn_sigma_pe.data(),
        phi_aff, fixed, kappa, x, e_n, e_acc, e_k, T_k, dT_rel, step_prev, omega, clip, d_flags,
        n, G, dt);
    cuda_check(cudaGetLastError(), "sn_ld matter update launch failed");
    double mx[2] = {0.0, 0.0};
    reduce_max2(dT_rel, nullptr, two_n, mx);
    outer_res = mx[0];
    outer_done = k + 1;
    if (outer_res <= outer_tol && gmres_converged) {
      converged = true;
      break;
    }
    if (k + 1 == max_outer && !gmres_converged) {
      retry_flags |= kRetryGmres;
    }
  }
  if (!converged) {
    retry_flags |= kRetryNewton;
  }

  // Accepted state.
  double* face = scratch_doubles("sn_ld:face_current", static_cast<std::size_t>(n + 1) * G);
  double* prr_phi = scratch_doubles("sn_ld:prr", n_cell_groups);
  moments(cells, psi_aff, outer_psi_in, quad, phi_hom, face, prr_phi, n, G);
  finalize_cell_kernel<EOS_TAIL><<<cell_warp_blocks(n), kCellWarpThreads>>>(
      closure, cells, d_void, e_acc, dT_rel, state.ee.data(), state.Te.data(),
      (state.Pe.size() == static_cast<std::size_t>(n)) ? state.Pe.data() : nullptr,
      state.sn_ee_node_offset.data(),
      (state.sn_delta_T.size() == static_cast<std::size_t>(n)) ? state.sn_delta_T.data() : nullptr,
      n);
  cuda_check(cudaGetLastError(), "sn_ld cell finalization launch failed");
  const std::size_t cg_bytes = n_cell_groups * sizeof(double);
  cuda_check(cudaMemcpy(state.sn_diag_rad_E_pre.data(), state.rad_E_old.data(), cg_bytes,
                        cudaMemcpyDeviceToDevice),
             "sn_ld rad_E_pre copy failed");
  cuda_check(cudaMemset(d_flags + 2, 0, sizeof(int)), "sn_ld negative count reset failed");
  finalize_group_kernel<<<cg_grid, kBlock>>>(
      cells, state.sn_sigma_a.data(), phi_aff, fixed, kappa, x, prr_phi, clip,
      state.rad_E.data(),
      (state.sn_phi_old.size() == n_cell_groups) ? state.sn_phi_old.data() : nullptr,
      state.rad_dep.data(), state.rad_emit.data(), state.sn_Prr.data(),
      state.sn_chi.data(), state.sn_diag_rad_absorption.data(),
      state.sn_diag_rad_emission_at_Tnp1.data(), state.sn_diag_clip_energy.data(), d_flags + 2,
      n, G, dt);
  cuda_check(cudaGetLastError(), "sn_ld group finalization launch failed");
  face_flux_kernel<<<grid_for(static_cast<std::size_t>(n + 1) * G), kBlock>>>(
      face, state.sn_face_flux_raw.data(), state.sn_face_flux_limited.data(),
      state.sn_face_flux_blended.data(), state.sn_face_flux_diff.data(),
      state.sn_face_alpha.data(), n, G);
  cuda_check(cudaGetLastError(), "sn_ld face flux launch failed");
  outer_current_kernel<<<1, 1>>>(face, d_small, n, G);
  cuda_check(cudaGetLastError(), "sn_ld outer current launch failed");
  double outer_current = 0.0;
  cuda_check(cudaMemcpy(&outer_current, d_small, sizeof(double), cudaMemcpyDeviceToHost),
             "sn_ld outer current copy failed");
  double r_outer = 0.0;
  cuda_check(cudaMemcpy(&r_outer, state.x_r.data() + n, sizeof(double), cudaMemcpyDeviceToHost),
             "sn_ld outer radius copy failed");
  // Net outflow plus the discrete inflow: the gross escape of the energy
  // budget (the inflow is booked as sn_marshak_in_step).
  state.sn_escaped_step =
      dt * mesh::geometry_1d_face_area(geom, r_outer) * outer_current + state.sn_marshak_in_step;
  cuda_check(cudaMemcpy(state.sn_psi_prev.data(), psi_aff, n_psi * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "sn_ld intensity history copy failed");
  if (curved && n_sd > 0U) {
    cuda_check(cudaMemcpy(state.sn_psi_sd_prev.data(), sd_aff, n_sd * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "sn_ld starting-direction history copy failed");
  }
  cuda_check(cudaMemcpy(state.sn_diag_rad_E_post.data(), state.rad_E.data(), cg_bytes,
                        cudaMemcpyDeviceToDevice),
             "sn_ld rad_E_post copy failed");
  cuda_check(cudaMemcpy(state.rad_E_old.data(), state.rad_E.data(), cg_bytes,
                        cudaMemcpyDeviceToDevice),
             "sn_ld rad_E_old copy failed");
  int host_flags[4] = {0, 0, 0, 0};
  cuda_check(cudaMemcpy(host_flags, d_flags, 4 * sizeof(int), cudaMemcpyDeviceToHost),
             "sn_ld flag copy failed");
  retry_flags |= host_flags[0];

  state.sn_converged = converged;
  state.sn_outer_stagnated = false;
  state.sn_outer_iterations = outer_done;
  state.sn_inner_iterations = gmres_iterations;
  state.sn_outer_residual = outer_res;
  state.sn_inner_residual = gmres_res;
  state.sn_void_anchor_dE_step = 0.0;
  state.sn_void_anchor_dE_abs_step = 0.0;
  state.sn_material_retry_flag |= retry_flags;
  state.holo_ale_invalidated = false;

  if (retry_flags != 0) {
    static int warn_count = 0;
    ++warn_count;
    if (warn_count <= 5 || warn_count % 1000 == 0) {
      // The node of the largest last temperature change: its cell, its next
      // linearization point, the accepted cell temperature and the node's
      // absorption-rate density.
      std::vector<double> h_dT(two_n), h_T(two_n), h_A(two_n), h_Te(static_cast<std::size_t>(n));
      cuda_check(cudaMemcpy(h_dT.data(), dT_rel, two_n * sizeof(double), cudaMemcpyDeviceToHost),
                 "sn_ld diagnostic copy failed");
      cuda_check(cudaMemcpy(h_T.data(), T_k, two_n * sizeof(double), cudaMemcpyDeviceToHost),
                 "sn_ld diagnostic copy failed");
      cuda_check(cudaMemcpy(h_A.data(), A_aff, two_n * sizeof(double), cudaMemcpyDeviceToHost),
                 "sn_ld diagnostic copy failed");
      state.Te.copy_to_host(h_Te.data());
      std::size_t worst = 0;
      for (std::size_t i = 1; i < two_n; ++i) {
        if (h_dT[i] > h_dT[worst]) {
          worst = i;
        }
      }
      core::log_warning("SN linear-discontinuous step flags=" + std::to_string(retry_flags) +
                        " (1 floor, 4 Newton, 8 GMRES, 16 scattering): Newton iterations " +
                        std::to_string(outer_done) + ", temperature change " +
                        std::to_string(outer_res) + " (cell " + std::to_string(worst / 2) +
                        ((worst % 2 == 0) ? " left" : " right") + " node: next point " +
                        std::to_string(h_T[worst]) + " eV, cell Te " +
                        std::to_string(h_Te[worst / 2]) + " eV, absorption " +
                        std::to_string(h_A[worst]) + " erg/cm3/s), GMRES residual " +
                        std::to_string(gmres_res) + ", sweeps " + std::to_string(total_sweeps) +
                        " (occurrence #" + std::to_string(warn_count) + ")");
    }
  }
  if (host_flags[2] > 0) {
    static int negative_note_count = 0;
    ++negative_note_count;
    if (negative_note_count <= 5 || negative_note_count % 1000 == 0) {
      core::log_warning("SN linear-discontinuous step: " + std::to_string(host_flags[2]) +
                        " (cell, group) radiation energies below zero (occurrence #" +
                        std::to_string(negative_note_count) + ")");
    }
  }
}

}  // namespace

void advance_step(core::State& state, const core::Config& cfg, const PlanckTable& planck,
                  const core::Config::MaterialsConfig::MatDef& mat, const double dt,
                  const double* outer_psi_in, const double* source_ext) {
  if (cfg.numerics.hydro.eos_closure_mode == "energy_authoritative") {
    advance_step_impl<true>(state, cfg, planck, mat, dt, outer_psi_in, source_ext);
  } else {
    advance_step_impl<false>(state, cfg, planck, mat, dt, outer_psi_in, source_ext);
  }
}

}  // namespace tenryu::radiation::sn_ld
