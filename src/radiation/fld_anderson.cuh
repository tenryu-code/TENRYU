#pragma once
// Anderson acceleration of the FLD outer fixed-point iteration — 1D_SPH port
// (2026-09-14) of the 2D_RZ scheme in fld_2d_rz_gpu.cu. Walker-Ni form:
//   u_{k+1} = u_k + beta f_k - sum_j gamma_j (dU_j + beta dF_j),
// with f_k = G(u_k) - u_k the outer residual (raw Newton output minus the
// linearization temperature) and gamma the solution of the Tikhonov-
// regularized normal equations of the last p <= m residual differences.
// The kernels are translation-unit local (static) so this header can be
// included by more than one .cu file.
#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>

namespace tenryu::radiation::fld_anderson {

constexpr int kMaxHistory = 4;
constexpr int kBlock = 256;

// f[i] = g[i] - u[i]
static __global__ void aa_diff_kernel(const double* __restrict__ g,
                                      const double* __restrict__ u,
                                      double* __restrict__ f,
                                      const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    f[i] = g[i] - u[i];
  }
}

// Single-block grid-stride dot product; n is the cell count of a 1D mesh.
static __global__ void aa_dot_kernel(const double* __restrict__ a,
                                     const double* __restrict__ b,
                                     const int n,
                                     double* __restrict__ out) {
  __shared__ double shared[kBlock];
  double local = 0.0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    local += a[i] * b[i];
  }
  shared[threadIdx.x] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[0] = shared[0];
  }
}

// u_next[i] = u[i] + beta*f[i] - sum_j gamma[j]*(dU_j[i] + beta*dF_j[i]),
// with dU_j/dF_j formed on the fly from the ring buffers; p <= 4.
// Non-finite results fall back to the raw Newton output g_raw[i]; the
// temperature floor te_floor is applied to the result.
static __global__ void aa_mix_kernel(const double* __restrict__ u,
                                     const double* __restrict__ f,
                                     const double* __restrict__ u_hist0,
                                     const double* __restrict__ u_hist1,
                                     const double* __restrict__ u_hist2,
                                     const double* __restrict__ u_hist3,
                                     const double* __restrict__ u_hist4,
                                     const double* __restrict__ f_hist0,
                                     const double* __restrict__ f_hist1,
                                     const double* __restrict__ f_hist2,
                                     const double* __restrict__ f_hist3,
                                     const double* __restrict__ f_hist4,
                                     const double g0,
                                     const double g1,
                                     const double g2,
                                     const double g3,
                                     const int p,
                                     const double beta,
                                     const double te_floor,
                                     const double* __restrict__ g_raw,
                                     double* __restrict__ te_out,
                                     const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const double* u_hist[5] = {u_hist0, u_hist1, u_hist2, u_hist3, u_hist4};
  const double* f_hist[5] = {f_hist0, f_hist1, f_hist2, f_hist3, f_hist4};
  const double gamma[4] = {g0, g1, g2, g3};
  double val = u[i] + beta * f[i];
  for (int j = 0; j < p; ++j) {
    const double du = u_hist[j + 1][i] - u_hist[j][i];
    const double df = f_hist[j + 1][i] - f_hist[j][i];
    val -= gamma[j] * (du + beta * df);
  }
  if (!isfinite(val)) {
    val = g_raw[i];
  }
  te_out[i] = fmax(val, te_floor);
}

// Ring of the last m+1 outer iterates u_j (linearization temperatures) and
// residuals f_j = G(u_j) - u_j. Slot of iteration k is k % (m+1).
struct Ring {
  double* u[kMaxHistory + 1] = {};
  double* f[kMaxHistory + 1] = {};
  int m = 1;      // history depth, 1..kMaxHistory
  int count = 0;  // (u, f) pairs recorded so far in this step
};

// Device dot product with a host read-back (synchronizing).
inline double dot(const double* a, const double* b, const int n, double* d_scalar) {
  aa_dot_kernel<<<1, kBlock>>>(a, b, n, d_scalar);
  double h = 0.0;
  if (cudaMemcpy(&h, d_scalar, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
    return std::nan("");
  }
  return h;
}

// Records f_k = te - u_k for the ring slot of iteration `iter` (te holds the
// raw Newton output of that iteration) and, once an earlier pair exists,
// overwrites te with the Anderson mix that becomes the next linearization
// temperature. Returns true when a mix was applied; a degenerate
// least-squares round leaves te (the raw Newton output) untouched. Kernel
// launch errors are left for the caller's cudaGetLastError().
inline bool record_and_mix(Ring& ring,
                           const int slot,
                           const int iter,
                           double* te,
                           const int n,
                           const double beta,
                           const double te_floor,
                           double* d_scalar) {
  const int grid = (n + kBlock - 1) / kBlock;
  aa_diff_kernel<<<grid, kBlock>>>(te, ring.u[slot], ring.f[slot], n);
  ++ring.count;
  const int p = (ring.count - 1 < ring.m) ? (ring.count - 1) : ring.m;
  if (p < 1) {
    return false;
  }
  // Ring order: oldest-to-newest pair indices for the last p+1 records.
  int order[kMaxHistory + 1];
  for (int j = 0; j <= p; ++j) {
    order[j] = (iter - p + j) % (ring.m + 1);
  }
  double gram[kMaxHistory][kMaxHistory];
  double rhs_v[kMaxHistory];
  bool finite_ok = true;
  for (int a = 0; a < p; ++a) {
    const double* fa1 = ring.f[order[a + 1]];
    const double* fa0 = ring.f[order[a]];
    // dot(dF_a, x) = dot(f_{a+1}, x) - dot(f_a, x)
    rhs_v[a] = dot(fa1, ring.f[slot], n, d_scalar) - dot(fa0, ring.f[slot], n, d_scalar);
    for (int b = 0; b <= a; ++b) {
      const double* fb1 = ring.f[order[b + 1]];
      const double* fb0 = ring.f[order[b]];
      const double dot_ab = dot(fa1, fb1, n, d_scalar) - dot(fa1, fb0, n, d_scalar) -
                            dot(fa0, fb1, n, d_scalar) + dot(fa0, fb0, n, d_scalar);
      gram[a][b] = dot_ab;
      gram[b][a] = dot_ab;
    }
  }
  double trace = 0.0;
  for (int a = 0; a < p; ++a) {
    trace += gram[a][a];
    if (!std::isfinite(gram[a][a]) || !std::isfinite(rhs_v[a])) {
      finite_ok = false;
    }
  }
  if (!(finite_ok && trace > 0.0)) {
    return false;
  }
  const double lambda = 1.0e-12 * trace / static_cast<double>(p);
  for (int a = 0; a < p; ++a) {
    gram[a][a] += lambda;
  }
  // Cholesky solve gram * gamma = rhs_v (p <= 4).
  double L[kMaxHistory][kMaxHistory] = {};
  for (int a = 0; a < p; ++a) {
    double diag = gram[a][a];
    for (int b = 0; b < a; ++b) {
      diag -= L[a][b] * L[a][b];
    }
    if (!(diag > 0.0)) {
      return false;
    }
    L[a][a] = std::sqrt(diag);
    for (int r2 = a + 1; r2 < p; ++r2) {
      double v2 = gram[r2][a];
      for (int b = 0; b < a; ++b) {
        v2 -= L[r2][b] * L[a][b];
      }
      L[r2][a] = v2 / L[a][a];
    }
  }
  double y[kMaxHistory] = {};
  for (int a = 0; a < p; ++a) {
    double v2 = rhs_v[a];
    for (int b = 0; b < a; ++b) {
      v2 -= L[a][b] * y[b];
    }
    y[a] = v2 / L[a][a];
  }
  double gamma_v[kMaxHistory] = {};
  for (int a = p - 1; a >= 0; --a) {
    double v2 = y[a];
    for (int b = a + 1; b < p; ++b) {
      v2 -= L[b][a] * gamma_v[b];
    }
    gamma_v[a] = v2 / L[a][a];
  }
  aa_mix_kernel<<<grid, kBlock>>>(
      ring.u[slot], ring.f[slot],
      ring.u[order[0]],
      ring.u[p >= 1 ? order[1] : order[0]],
      ring.u[p >= 2 ? order[2] : order[0]],
      ring.u[p >= 3 ? order[3] : order[0]],
      ring.u[p >= 4 ? order[4] : order[0]],
      ring.f[order[0]],
      ring.f[p >= 1 ? order[1] : order[0]],
      ring.f[p >= 2 ? order[2] : order[0]],
      ring.f[p >= 3 ? order[3] : order[0]],
      ring.f[p >= 4 ? order[4] : order[0]],
      gamma_v[0], gamma_v[1], gamma_v[2], gamma_v[3],
      p, beta, te_floor, te, te, n);
  return true;
}

}  // namespace tenryu::radiation::fld_anderson
