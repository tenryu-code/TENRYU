#pragma once
// Anderson acceleration of the FLD outer fixed-point iteration — 1D_SPH port
// (2026-09-14) of the 2D_RZ scheme in fld_2d_rz_gpu.cu; also the 1D S_N
// source iteration (sn_transport_1d_gpu.cu, 2026-09-24). Walker-Ni form:
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

// Device work area of one mix: the inner products of the window residuals
// (kDotStride x kDotStride, symmetric), then gamma[kMaxHistory] and the
// applied flag.
constexpr int kDotStride = kMaxHistory + 1;
constexpr int kWorkGamma = kDotStride * kDotStride;
constexpr int kWorkFlag = kWorkGamma + kMaxHistory;
constexpr int kWorkDoubles = kWorkFlag + 1;

struct Window {
  const double* f[kMaxHistory + 1];
};

// One block per pair (i <= j) of window residuals: the single-block
// grid-stride sum and tree reduction of the former per-dot kernel, so every
// inner product is bit-identical to it (a*b and b*a round alike).
static __global__ void aa_pair_dots_kernel(const Window w,
                                           const int p,
                                           const int n,
                                           double* __restrict__ work) {
  int pair = blockIdx.x;
  int i = 0;
  while (pair > p - i) {
    pair -= p - i + 1;
    ++i;
  }
  const int j = i + pair;
  const double* __restrict__ a = w.f[i];
  const double* __restrict__ b = w.f[j];
  __shared__ double shared[kBlock];
  double local = 0.0;
  for (int k = threadIdx.x; k < n; k += blockDim.x) {
    local += a[k] * b[k];
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
    work[i * kDotStride + j] = shared[0];
    work[j * kDotStride + i] = shared[0];
  }
}

// Gram/rhs assembly, Tikhonov regularization, Cholesky and triangular solves
// of the former host code, operation by operation in round-to-nearest
// intrinsics (no contraction), so gamma matches the host solve bitwise. D[x][y]
// is the inner product of window residuals x and y (window p is the newest).
// Writes gamma and the applied flag (0 for a degenerate round).
static __global__ void aa_solve_kernel(const int p, double* __restrict__ work) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const auto D = [&](const int x, const int y) { return work[x * kDotStride + y]; };
  double gram[kMaxHistory][kMaxHistory];
  double rhs_v[kMaxHistory];
  for (int a = 0; a < p; ++a) {
    rhs_v[a] = __dsub_rn(D(a + 1, p), D(a, p));
    for (int b = 0; b <= a; ++b) {
      const double dot_ab =
          __dadd_rn(__dsub_rn(__dsub_rn(D(a + 1, b + 1), D(a + 1, b)), D(a, b + 1)), D(a, b));
      gram[a][b] = dot_ab;
      gram[b][a] = dot_ab;
    }
  }
  work[kWorkFlag] = 0.0;
  double trace = 0.0;
  bool finite_ok = true;
  for (int a = 0; a < p; ++a) {
    trace = __dadd_rn(trace, gram[a][a]);
    if (!isfinite(gram[a][a]) || !isfinite(rhs_v[a])) {
      finite_ok = false;
    }
  }
  if (!(finite_ok && trace > 0.0)) {
    return;
  }
  const double lambda = __ddiv_rn(__dmul_rn(1.0e-12, trace), static_cast<double>(p));
  for (int a = 0; a < p; ++a) {
    gram[a][a] = __dadd_rn(gram[a][a], lambda);
  }
  double L[kMaxHistory][kMaxHistory] = {};
  for (int a = 0; a < p; ++a) {
    double diag = gram[a][a];
    for (int b = 0; b < a; ++b) {
      diag = __dsub_rn(diag, __dmul_rn(L[a][b], L[a][b]));
    }
    if (!(diag > 0.0)) {
      return;
    }
    L[a][a] = __dsqrt_rn(diag);
    for (int r2 = a + 1; r2 < p; ++r2) {
      double v2 = gram[r2][a];
      for (int b = 0; b < a; ++b) {
        v2 = __dsub_rn(v2, __dmul_rn(L[r2][b], L[a][b]));
      }
      L[r2][a] = __ddiv_rn(v2, L[a][a]);
    }
  }
  double y[kMaxHistory] = {};
  for (int a = 0; a < p; ++a) {
    double v2 = rhs_v[a];
    for (int b = 0; b < a; ++b) {
      v2 = __dsub_rn(v2, __dmul_rn(L[a][b], y[b]));
    }
    y[a] = __ddiv_rn(v2, L[a][a]);
  }
  double gamma_v[kMaxHistory] = {};
  for (int a = p - 1; a >= 0; --a) {
    double v2 = y[a];
    for (int b = a + 1; b < p; ++b) {
      v2 = __dsub_rn(v2, __dmul_rn(L[b][a], gamma_v[b]));
    }
    gamma_v[a] = __ddiv_rn(v2, L[a][a]);
  }
  for (int a = 0; a < kMaxHistory; ++a) {
    work[kWorkGamma + a] = gamma_v[a];
  }
  work[kWorkFlag] = 1.0;
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
                                     const double* __restrict__ work,
                                     const int p,
                                     const double beta,
                                     const double te_floor,
                                     const double* __restrict__ g_raw,
                                     double* __restrict__ te_out,
                                     const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || work[kWorkFlag] == 0.0) {
    return;  // degenerate round: te keeps the raw Newton output
  }
  const double* u_hist[5] = {u_hist0, u_hist1, u_hist2, u_hist3, u_hist4};
  const double* f_hist[5] = {f_hist0, f_hist1, f_hist2, f_hist3, f_hist4};
  const double gamma[4] = {work[kWorkGamma], work[kWorkGamma + 1], work[kWorkGamma + 2],
                           work[kWorkGamma + 3]};
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

// Records f_k = te - u_k for the ring slot of iteration `iter` (te holds the
// raw Newton output of that iteration) and, once an earlier pair exists,
// launches the Anderson mix that overwrites te with the next linearization
// temperature. Everything runs on the device without a host synchronization
// (2026-09-23; the former host solve read back each inner product): the
// inner products in one launch, the regularized least-squares solve in a
// single thread, then the mix, which a degenerate round skips (te keeps the
// raw Newton output). d_work holds kWorkDoubles doubles. Returns true when a
// mix was launched (p >= 1); read_mix_applied tells whether it applied.
// Kernel launch errors are left for the caller's cudaGetLastError().
inline bool record_and_mix(Ring& ring,
                           const int slot,
                           const int iter,
                           double* te,
                           const int n,
                           const double beta,
                           const double te_floor,
                           double* d_work,
                           cudaStream_t stream = nullptr) {
  const int grid = (n + kBlock - 1) / kBlock;
  aa_diff_kernel<<<grid, kBlock, 0, stream>>>(te, ring.u[slot], ring.f[slot], n);
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
  Window window{};
  for (int j = 0; j <= p; ++j) {
    window.f[j] = ring.f[order[j]];
  }
  const int n_pairs = (p + 1) * (p + 2) / 2;
  aa_pair_dots_kernel<<<n_pairs, kBlock, 0, stream>>>(window, p, n, d_work);
  aa_solve_kernel<<<1, 1, 0, stream>>>(p, d_work);
  aa_mix_kernel<<<grid, kBlock, 0, stream>>>(
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
      d_work,
      p, beta, te_floor, te, te, n);
  return true;
}

// Whether the last launched mix applied (synchronizing read; for tests and
// diagnostics only).
inline bool read_mix_applied(const double* d_work) {
  double flag = 0.0;
  if (cudaMemcpy(&flag, d_work + kWorkFlag, sizeof(double), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    return false;
  }
  return flag != 0.0;
}

}  // namespace tenryu::radiation::fld_anderson
