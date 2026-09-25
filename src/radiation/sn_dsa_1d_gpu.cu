#include "radiation/sn_dsa_1d_gpu.cuh"

#include <algorithm>
#include <cstddef>
#include <map>
#include <utility>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "core/constants.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "mesh/geometry_1d.cuh"

namespace tenryu::radiation {
namespace {

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

// Diffusion-synthetic acceleration consistent with the sweep's discretization
// (NUMERICS 6.8, 2026-09-24). The correction f of the angular flux satisfies
// the swept equations with the scattering residual sigma_s (phi^{l+1/2} -
// phi^l) as source; with the P1 ansatz psi_m = a Phi + b mu_m J (a = 1 /
// sum w, b = 1 / sum w mu^2) at the cell faces, the zeroth and first angular
// moments of the discrete balance per cell, and the cell averages from the
// sweep's own spatial closure psi_m = theta psi_dn + (1 - theta) psi_up
// (theta per cell, group and angle), the face moments (Phi_f, J_f) solve
//   A_o J_o - A_i J_i + (sigma_a + 1/(c dt)) V Phi_c = sigma_s V R_c,
//   k (A_o Phi_o - A_i Phi_i) - k dA Phi_c + (sigma_t + 1/(c dt)) V J_c = 0,
// k = a sum w mu^2 (1/3), with Phi_c and J_c linear in the four face moments
// of the cell (the angular redistribution's first moment is -k dA Phi_c / V
// for any quadrature, by the discrete cancellation of the isotropic flux;
// its J part vanishes for the symmetric quadratures). Boundaries: J = 0 at
// the inner reflecting face; no incoming correction at the outer face,
// a Phi S1 = b J S2 with S1 = sum_{mu<0} w |mu|, S2 = sum_{mu<0} w mu^2.
// The unknowns ordered (Phi_0, J_0, ..., Phi_N, J_N) make the system
// pentadiagonal: row 0 the inner condition, rows 2c+1 the balance of cell c,
// rows 2c+2 its first moment, the last row the outer condition. The cell
// correction is Phi_c. The former cell-centred diffusion operator (harmonic
// D between cell centres) was not consistent with the swept closure: its
// iteration diverged on cells thicker than about 3 mean free paths
// (spectral radius 3.5 at sigma_t dr = 10.5, 5.5 at 104, against 0.21 and
// 0.21 for this operator; 0.15 and 0.18 at 0.3).
__global__ void assemble_sn_dsa_consistent_kernel(
    const double* __restrict__ x_r,
    const double* __restrict__ vol,
    const double* __restrict__ sigma_a,
    const double* __restrict__ sigma_s,
    const double* __restrict__ phi_old,
    const double* __restrict__ phi_sweep,
    const double* __restrict__ mu,
    const double* __restrict__ weight,
    const double* __restrict__ theta,  // [cell][group][angle]; nullptr: 1/2
    double* __restrict__ ds,
    double* __restrict__ dl,
    double* __restrict__ dd,
    double* __restrict__ du,
    double* __restrict__ dw,
    double* __restrict__ rhs,
    double* __restrict__ cell_map,  // [4][cell][group]: Phi_o, Phi_i, J_o, J_i coefficients of Phi_c
    const int n_cells,
    const int n_groups,
    const int n_angles,
    const double dt,
    const int geom) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int g = idx / n_cells;
  const int c = idx - g * n_cells;
  const int cg = c * n_groups + g;
  const double V_in = nonnegative_finite(vol[c]);
  const double V = (V_in > 0.0)
                       ? V_in
                       : mesh::geometry_1d_shell_volume(geom, fmax(x_r[c], 0.0),
                                                        fmax(x_r[c + 1], 0.0));
  const double A_i = mesh::geometry_1d_face_area(geom, fmax(x_r[c], 0.0));
  const double A_o = mesh::geometry_1d_face_area(geom, fmax(x_r[c + 1], 0.0));
  const double dA = A_o - A_i;
  const double inv_cdt = (dt > 0.0) ? 1.0 / (core::constants::c_light * dt) : 0.0;
  const double sig_a = nonnegative_finite(sigma_a[cg]);
  const double sig_s = nonnegative_finite(sigma_s[cg]);
  const double sig_ta = sig_a + inv_cdt;
  const double sig_tt = sig_a + sig_s + inv_cdt;

  // Angular sums with the sweep's theta: cell averages of the P1 flux.
  double sum_w = 0.0;
  double sum_w_mu2 = 0.0;
  double s1_in = 0.0;  // sum_{mu<0} w |mu|
  double s2_in = 0.0;  // sum_{mu<0} w mu^2
  double co = 0.0, ci = 0.0, jo = 0.0, ji = 0.0, qo = 0.0, qi = 0.0;
  const std::size_t theta_base =
      (static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups) +
       static_cast<std::size_t>(g)) *
      static_cast<std::size_t>(n_angles);
  for (int n = 0; n < n_angles; ++n) {
    const double w = finite_or_zero(weight[n]);
    const double m = finite_or_zero(mu[n]);
    const double th = (theta != nullptr)
                          ? fmin(fmax(finite_or_zero(theta[theta_base + n]), 0.0), 1.0)
                          : 0.5;
    sum_w += w;
    sum_w_mu2 += w * m * m;
    if (m > 0.0) {
      // Downstream face = outer.
      co += w * th;
      ci += w * (1.0 - th);
      jo += w * m * th;
      ji += w * m * (1.0 - th);
      qo += w * m * m * th;
      qi += w * m * m * (1.0 - th);
    } else if (m < 0.0) {
      co += w * (1.0 - th);
      ci += w * th;
      jo += w * m * (1.0 - th);
      ji += w * m * th;
      qo += w * m * m * (1.0 - th);
      qi += w * m * m * th;
      s1_in += w * (-m);
      s2_in += w * m * m;
    }
  }
  const double a = (sum_w > 0.0) ? 1.0 / sum_w : 0.0;
  const double b = (sum_w_mu2 > 0.0) ? 1.0 / sum_w_mu2 : 0.0;
  const double k = a * sum_w_mu2;
  // Phi_c = p_o Phi_o + p_i Phi_i + r_o J_o + r_i J_i;
  // J_c   = a jo Phi_o + a ji Phi_i + b qo J_o + b qi J_i.
  const double p_o = a * co, p_i = a * ci, r_o = b * jo, r_i = b * ji;
  const double j_po = a * jo, j_pi = a * ji, j_jo = b * qo, j_ji = b * qi;
  const std::size_t G = static_cast<std::size_t>(n_groups);
  const std::size_t N_cells = static_cast<std::size_t>(n_cells);
  cell_map[0 * N_cells * G + cg] = p_o;
  cell_map[1 * N_cells * G + cg] = p_i;
  cell_map[2 * N_cells * G + cg] = r_o;
  cell_map[3 * N_cells * G + cg] = r_i;

  // Interleaved layout: row r of group g at r * G + g.
  const auto at = [&](const int row) { return static_cast<std::size_t>(row) * G + g; };
  // Balance of cell c: row 2c+1, unknowns Phi_i (2c: -1), J_i (0), Phi_o (+1), J_o (+2).
  {
    const int row = 2 * c + 1;
    const double s = sig_ta * V;
    ds[at(row)] = 0.0;
    dl[at(row)] = s * p_i;
    dd[at(row)] = -A_i + s * r_i;
    du[at(row)] = s * p_o;
    dw[at(row)] = A_o + s * r_o;
    const double delta_phi = finite_or_zero(phi_sweep[cg]) - finite_or_zero(phi_old[cg]);
    const double r = sig_s * V * delta_phi;
    rhs[at(row)] = isfinite(r) ? r : 0.0;
  }
  // First moment of cell c: row 2c+2, unknowns Phi_i (-2), J_i (-1), Phi_o (0), J_o (+1).
  {
    const int row = 2 * c + 2;
    const double t = sig_tt * V;
    ds[at(row)] = -k * A_i - k * dA * p_i + t * j_pi;
    dl[at(row)] = -k * dA * r_i + t * j_ji;
    dd[at(row)] = k * A_o - k * dA * p_o + t * j_po;
    du[at(row)] = -k * dA * r_o + t * j_jo;
    dw[at(row)] = 0.0;
    rhs[at(row)] = 0.0;
  }
  if (c == 0) {
    // J_0 = 0 (reflecting centre, axis or symmetry plane).
    ds[at(0)] = 0.0;
    dl[at(0)] = 0.0;
    dd[at(0)] = 0.0;
    du[at(0)] = 1.0;
    dw[at(0)] = 0.0;
    rhs[at(0)] = 0.0;
  }
  if (c == n_cells - 1) {
    // No incoming correction through the outer face.
    const int row = 2 * n_cells + 1;
    ds[at(row)] = 0.0;
    dl[at(row)] = a * s1_in;
    dd[at(row)] = -b * s2_in;
    du[at(row)] = 0.0;
    dw[at(row)] = 0.0;
    rhs[at(row)] = 0.0;
  }
}

__global__ void apply_sn_dsa_consistent_correction_kernel(const double* __restrict__ x,
                                                          const double* __restrict__ cell_map,
                                                          double* __restrict__ phi,
                                                          const int n_cells,
                                                          const int n_groups) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int g = idx / n_cells;
  const int c = idx - g * n_cells;
  const int cg = c * n_groups + g;
  const std::size_t G = static_cast<std::size_t>(n_groups);
  const std::size_t N_cells = static_cast<std::size_t>(n_cells);
  const double phi_i = x[static_cast<std::size_t>(2 * c) * G + g];
  const double j_i = x[static_cast<std::size_t>(2 * c + 1) * G + g];
  const double phi_o = x[static_cast<std::size_t>(2 * c + 2) * G + g];
  const double j_o = x[static_cast<std::size_t>(2 * c + 3) * G + g];
  const double correction = cell_map[0 * N_cells * G + cg] * phi_o +
                            cell_map[1 * N_cells * G + cg] * phi_i +
                            cell_map[2 * N_cells * G + cg] * j_o +
                            cell_map[3 * N_cells * G + cg] * j_i;
  const double corrected = finite_or_zero(phi[cg]) + finite_or_zero(correction);
  phi[cg] = fmax(corrected, 0.0);
}

// Pentadiagonal slab: ds, dl, d, du, dw, x [n_rows * n_groups] each and the
// cell map [4 * n_cells * n_groups] (one scratch allocation).
struct SnDsaSlab {
  double* ds = nullptr;
  double* dl = nullptr;
  double* dd = nullptr;
  double* du = nullptr;
  double* dw = nullptr;
  double* x = nullptr;
  double* cell_map = nullptr;
};

SnDsaSlab sn_dsa_slab(const int n_cells, const int n_groups) {
  const std::size_t rows = 2U * (static_cast<std::size_t>(n_cells) + 1U);
  const std::size_t G = static_cast<std::size_t>(n_groups);
  const std::size_t n_row = rows * G;
  const std::size_t n_map = 4U * static_cast<std::size_t>(n_cells) * G;
  double* base = static_cast<double*>(core::device_scratch_acquire(
      "sn_dsa_1d:consistent_slab", (6U * n_row + n_map) * sizeof(double)));
  SnDsaSlab s;
  s.ds = base;
  s.dl = base + n_row;
  s.dd = base + 2U * n_row;
  s.du = base + 3U * n_row;
  s.dw = base + 4U * n_row;
  s.x = base + 5U * n_row;
  s.cell_map = base + 6U * n_row;
  return s;
}

}  // namespace

void ensure_sn_dsa_1d_gpu_buffers(core::State& state,
                                  const int n_cells,
                                  const int n_groups) {
  if (n_cells <= 0 || n_groups <= 0) {
    return;
  }
  const SnDsaSlab slab = sn_dsa_slab(n_cells, n_groups);
  const cusparseHandle_t handle = cusparse_cache().handle;
  const int rows = 2 * (n_cells + 1);
  const auto buffer_key = std::make_pair(n_cells, n_groups);
  static std::map<std::pair<int, int>, std::size_t> buffer_size_cache;
  auto buffer_it = buffer_size_cache.find(buffer_key);
  if (buffer_it == buffer_size_cache.end()) {
    std::size_t queried_size = 0U;
    cusparse_check(cusparseDgpsvInterleavedBatch_bufferSizeExt(
                       handle, 0, rows, slab.ds, slab.dl, slab.dd, slab.du, slab.dw, slab.x,
                       n_groups, &queried_size),
                   "SN DSA cuSPARSE pentadiagonal buffer size failed");
    buffer_it = buffer_size_cache.emplace(buffer_key, queried_size).first;
  }
  const std::size_t buffer_size = buffer_it->second;
  const std::size_t buffer_doubles =
      (buffer_size + sizeof(double) - 1U) / sizeof(double);
  if (state.sn_dsa_cusparse_buffer.size() < std::max<std::size_t>(buffer_doubles, 1U)) {
    state.sn_dsa_cusparse_buffer.reset(std::max<std::size_t>(buffer_doubles, 1U));
  }
}

const void* sn_dsa_1d_gpu_scratch(const int n_cells, const int n_groups) {
  if (n_cells <= 0 || n_groups <= 0) {
    return nullptr;
  }
  return sn_dsa_slab(n_cells, n_groups).ds;
}

void apply_sn_dsa_1d_gpu(core::State& state,
                         const int n_cells,
                         const int n_groups,
                         const double dt,
                         const int geom,
                         cudaStream_t stream,
                         const double* mu,
                         const double* weight,
                         const int n_angles,
                         const double* theta) {
  if (n_cells <= 0 || n_groups <= 0) {
    return;
  }
  TENRYU_ASSERT(mu != nullptr && weight != nullptr && n_angles > 0,
                "SN DSA requires the angular quadrature");
  ensure_sn_dsa_1d_gpu_buffers(state, n_cells, n_groups);
  const SnDsaSlab slab = sn_dsa_slab(n_cells, n_groups);
  const int total = n_cells * n_groups;
  const int grid = (total + kBlock - 1) / kBlock;
  assemble_sn_dsa_consistent_kernel<<<grid, kBlock, 0, stream>>>(
      state.x_r.data(), state.vol.data(), state.sn_sigma_a.data(), state.sn_sigma_s.data(),
      state.sn_phi_old.data(), state.sn_phi_sweep.data(), mu, weight, theta, slab.ds, slab.dl,
      slab.dd, slab.du, slab.dw, slab.x, slab.cell_map, n_cells, n_groups, n_angles, dt, geom);
  cuda_check(cudaGetLastError(), "SN DSA assembly launch failed");

  const cusparseHandle_t handle = cusparse_cache().handle;
  cusparse_check(cusparseSetStream(handle, stream), "SN DSA cusparseSetStream failed");
  cusparse_check(cusparseDgpsvInterleavedBatch(handle, 0, 2 * (n_cells + 1), slab.ds, slab.dl,
                                               slab.dd, slab.du, slab.dw, slab.x, n_groups,
                                               state.sn_dsa_cusparse_buffer.data()),
                 "SN DSA cuSPARSE pentadiagonal solve failed");

  apply_sn_dsa_consistent_correction_kernel<<<grid, kBlock, 0, stream>>>(
      slab.x, slab.cell_map, state.sn_phi_sweep.data(), n_cells, n_groups);
  cuda_check(cudaGetLastError(), "SN DSA correction launch failed");
}

}  // namespace tenryu::radiation
