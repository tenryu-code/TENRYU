#include "radiation/sn_dsa_1d_gpu.cuh"

#include <cstddef>
#include <map>
#include <utility>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "core/constants.hpp"
#include "core/error.hpp"

namespace tenryu::radiation {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
// Diffusion-denominator regularization only (D = 1/(3 sigma_t)):
// mfp 10 km, physically invisible at lab scale; prevents ~1e98 matrix
// entries in void/pure-absorber cells (the DSA correction vanishes at
// convergence, so gate answers are unchanged).
constexpr double kSigmaFloor = 1.0e-6;
constexpr int kBlock = 256;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline void cusparse_check(const cusparseStatus_t status, const char* message) {
  TENRYU_ASSERT(status == CUSPARSE_STATUS_SUCCESS, message);
}

struct CusparseHandleCache {
  CusparseHandleCache() {
    cusparse_check(cusparseCreate(&handle), "SN DSA cusparseCreate failed");
  }

  ~CusparseHandleCache() {
    if (handle != nullptr) {
      static_cast<void>(cusparseDestroy(handle));
    }
  }

  cusparseHandle_t handle = nullptr;
};

CusparseHandleCache& cusparse_cache() {
  static CusparseHandleCache cache;
  return cache;
}

__host__ __device__ inline double finite_or_zero(const double value) {
  return isfinite(value) ? value : 0.0;
}

__host__ __device__ inline double nonnegative_finite(const double value) {
  return isfinite(value) ? fmax(value, 0.0) : 0.0;
}

__device__ double harmonic_positive(const double a, const double b) {
  if (!(a > 0.0) || !(b > 0.0)) {
    return 0.0;
  }
  return 2.0 / (1.0 / a + 1.0 / b);
}

__global__ void assemble_sn_dsa_tridiag_kernel(
    const double* __restrict__ x_r,
    const double* __restrict__ vol,
    const double* __restrict__ sigma_a,
    const double* __restrict__ sigma_s,
    const double* __restrict__ phi_old,
    const double* __restrict__ phi_sweep,
    double* __restrict__ lower,
    double* __restrict__ diag,
    double* __restrict__ upper,
    double* __restrict__ rhs,
    int n_cells,
    int n_groups,
    double dt) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int g = idx / n_cells;
  const int c = idx - g * n_cells;
  const int cg = c * n_groups + g;
  const double V = nonnegative_finite(vol[c]);
  const double sig_a = nonnegative_finite(sigma_a[cg]);
  const double sig_t =
      fmax(sig_a + nonnegative_finite(sigma_s[cg]), kSigmaFloor);
  const double D_c = 1.0 / (3.0 * sig_t);

  double d = V + dt * core::constants::c_light * sig_a * V;
  double l = 0.0;
  double u = 0.0;
  if (c > 0) {
    const int left_cg = (c - 1) * n_groups + g;
    const double sig_t_l =
        fmax(nonnegative_finite(sigma_a[left_cg]) +
                 nonnegative_finite(sigma_s[left_cg]),
             kSigmaFloor);
    const double D_l = 1.0 / (3.0 * sig_t_l);
    const double Df = harmonic_positive(D_l, D_c);
    const double area = 4.0 * kPi * x_r[c] * x_r[c];
    const double xl = 0.5 * (x_r[c - 1] + x_r[c]);
    const double xc = 0.5 * (x_r[c] + x_r[c + 1]);
    const double dist = fmax(xc - xl, 1.0e-300);
    const double coef = dt * core::constants::c_light * area * Df / dist;
    d += coef;
    l = -coef;
  }
  if (c + 1 < n_cells) {
    const int right_cg = (c + 1) * n_groups + g;
    const double sig_t_r =
        fmax(nonnegative_finite(sigma_a[right_cg]) +
                 nonnegative_finite(sigma_s[right_cg]),
             kSigmaFloor);
    const double D_r = 1.0 / (3.0 * sig_t_r);
    const double Df = harmonic_positive(D_c, D_r);
    const double area = 4.0 * kPi * x_r[c + 1] * x_r[c + 1];
    const double xc = 0.5 * (x_r[c] + x_r[c + 1]);
    const double xr = 0.5 * (x_r[c + 1] + x_r[c + 2]);
    const double dist = fmax(xr - xc, 1.0e-300);
    const double coef = dt * core::constants::c_light * area * Df / dist;
    d += coef;
    u = -coef;
  } else {
    const double A_out = 4.0 * kPi * x_r[n_cells] * x_r[n_cells];
    constexpr double beta_vac = 0.5;  // Matches escaped_energy_kernel F = 0.5 c E.
    d += dt * core::constants::c_light * beta_vac * A_out;
  }
  lower[idx] = isfinite(l) ? l : 0.0;
  diag[idx] = (isfinite(d) && d > 0.0) ? d : 1.0;
  upper[idx] = isfinite(u) ? u : 0.0;
  const double sig_s_cell = nonnegative_finite(sigma_s[cg]);
  const double delta_phi =
      finite_or_zero(phi_sweep[cg]) - finite_or_zero(phi_old[cg]);
  rhs[idx] = core::constants::c_light * dt * sig_s_cell * V * delta_phi;
  if (!isfinite(rhs[idx])) {
    rhs[idx] = 0.0;
  }
}

__global__ void apply_sn_dsa_correction_kernel(const double* __restrict__ rhs,
                                               double* __restrict__ phi,
                                               int n_cells,
                                               int n_groups) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int g = idx / n_cells;
  const int c = idx - g * n_cells;
  const int cg = c * n_groups + g;
  const double corrected = finite_or_zero(phi[cg]) + finite_or_zero(rhs[idx]);
  phi[cg] = fmax(corrected, 0.0);
}

}  // namespace

void ensure_sn_dsa_1d_gpu_buffers(core::State& state,
                                  const int n_cells,
                                  const int n_groups) {
  if (n_cells <= 0 || n_groups <= 0) {
    return;
  }
  const cusparseHandle_t handle = cusparse_cache().handle;
  const auto buffer_key = std::make_pair(n_cells, n_groups);
  static std::map<std::pair<int, int>, std::size_t> buffer_size_cache;
  auto buffer_it = buffer_size_cache.find(buffer_key);
  if (buffer_it == buffer_size_cache.end()) {
    std::size_t queried_size = 0U;
    cusparse_check(cusparseDgtsv2StridedBatch_bufferSizeExt(
                      handle,
                      n_cells,
                      state.sn_dsa_lower.data(),
                      state.sn_dsa_diag.data(),
                      state.sn_dsa_upper.data(),
                      state.sn_dsa_rhs.data(),
                      n_groups,
                      n_cells,
                      &queried_size),
                  "SN DSA cuSPARSE buffer size failed");
    buffer_it = buffer_size_cache.emplace(buffer_key, queried_size).first;
  }
  const std::size_t buffer_size = buffer_it->second;
  const std::size_t buffer_doubles =
      (buffer_size + sizeof(double) - 1U) / sizeof(double);
  if (state.sn_dsa_cusparse_buffer.size() < buffer_doubles) {
    state.sn_dsa_cusparse_buffer.reset(buffer_doubles);
  }
}

void apply_sn_dsa_1d_gpu(core::State& state,
                         const int n_cells,
                         const int n_groups,
                         const double dt,
                         cudaStream_t stream) {
  if (n_cells <= 0 || n_groups <= 0) {
    return;
  }
  ensure_sn_dsa_1d_gpu_buffers(state, n_cells, n_groups);
  const int total = n_cells * n_groups;
  const int grid = (total + kBlock - 1) / kBlock;
  assemble_sn_dsa_tridiag_kernel<<<grid, kBlock, 0, stream>>>(
      state.x_r.data(),
      state.vol.data(),
      state.sn_sigma_a.data(),
      state.sn_sigma_s.data(),
      state.sn_phi_old.data(),
      state.sn_phi_sweep.data(),
      state.sn_dsa_lower.data(),
      state.sn_dsa_diag.data(),
      state.sn_dsa_upper.data(),
      state.sn_dsa_rhs.data(),
      n_cells,
      n_groups,
      dt);
  cuda_check(cudaGetLastError(), "SN DSA tridiag assembly launch failed");

  const cusparseHandle_t handle = cusparse_cache().handle;
  cusparse_check(cusparseSetStream(handle, stream), "SN DSA cusparseSetStream failed");
  cusparse_check(cusparseDgtsv2StridedBatch(handle,
                                            n_cells,
                                            state.sn_dsa_lower.data(),
                                            state.sn_dsa_diag.data(),
                                            state.sn_dsa_upper.data(),
                                            state.sn_dsa_rhs.data(),
                                            n_groups,
                                            n_cells,
                                            state.sn_dsa_cusparse_buffer.data()),
                "SN DSA cuSPARSE tridiagonal solve failed");

  apply_sn_dsa_correction_kernel<<<grid, kBlock, 0, stream>>>(
      state.sn_dsa_rhs.data(),
      state.sn_phi_sweep.data(),
      n_cells,
      n_groups);
  cuda_check(cudaGetLastError(), "SN DSA correction launch failed");
}

}  // namespace tenryu::radiation
