#include "radiation/sn_transport_2d_gpu.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "core/error.hpp"
#include "parallel/comm_buffers.hpp"
#include "parallel/halo_exchange.hpp"
#include "parallel/reduction.hpp"
#include "materials/ionmix_reader.cuh"
#include "materials/ionmix_reader.hpp"
#include "materials/opacity.cuh"
#include "materials/tmat_reader.hpp"
#include "radiation/group_structure.hpp"
#include "radiation/groups.cuh"
#include "radiation/nlte_coeffs.cuh"
#include "radiation/planck_table.cuh"
#include "radiation/sn_material_newton_gpu.cuh"
#include "radiation/sn_transport_gpu.cuh"

namespace tenryu::radiation {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr int kBlock = 256;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

// Indirect-drive Tr(t) resolution for the deterministic 2D marshak z faces
// (docs/design/2d_tr_drive_port_spec.md §2). IMC-2D precedence verbatim
// (source.cu emit_marshak): the per-face table (canonical key, then alias)
// wins over the constant marshak_Tr_eV (>0), which wins over the scalar
// marshak_Tr table. Solve-entry time; tables are frozen at init (no runtime
// Python).
double resolve_marshak_tr_face_eV(const core::Config& cfg,
                                  const core::State& state,
                                  const char* key,
                                  const char* alias) {
  double t_r = cfg.radiation.boundary.marshak_Tr_eV;
  if (!(t_r > 0.0) && state.marshak_Tr_1d.has_value()) {
    t_r = state.marshak_Tr_1d->eval(state.t);
  }
  const auto it_key = state.marshak_Tr_face_tables.find(key);
  if (it_key != state.marshak_Tr_face_tables.end()) {
    t_r = it_key->second.eval(state.t);
  } else {
    const auto it_alias = state.marshak_Tr_face_tables.find(alias);
    if (it_alias != state.marshak_Tr_face_tables.end()) {
      t_r = it_alias->second.eval(state.t);
    }
  }
  return std::max(t_r, 0.0);
}

__host__ __device__ inline double finite_or_zero(const double value) {
  return isfinite(value) ? value : 0.0;
}

__host__ __device__ inline double nonnegative_finite(const double value) {
  return isfinite(value) ? fmax(value, 0.0) : 0.0;
}

__host__ __device__ inline int node_index(const int i, const int j, const int nz) {
  return i * (nz + 1) + j;
}

__host__ __device__ inline int cell_index(const int i, const int j, const int nz) {
  return i * nz + j;
}

__host__ __device__ inline int n_r_faces(const int nr, const int nz) {
  return (nr + 1) * nz;
}

__host__ __device__ inline int n_faces(const int nr, const int nz) {
  return n_r_faces(nr, nz) + nr * (nz + 1);
}

__host__ __device__ inline int r_face_index(const int i, const int j, const int nz) {
  return i * nz + j;
}

__host__ __device__ inline int z_face_index(const int i,
                                            const int j,
                                            const int nr,
                                            const int nz) {
  return n_r_faces(nr, nz) + i * (nz + 1) + j;
}

__host__ __device__ inline std::size_t face_group_index(const int face,
                                                        const int group,
                                                        const int n_groups) {
  return static_cast<std::size_t>(face) * static_cast<std::size_t>(n_groups) +
         static_cast<std::size_t>(group);
}

__host__ __device__ inline std::size_t cell_group_index(const int cell,
                                                        const int group,
                                                        const int n_groups) {
  return static_cast<std::size_t>(cell) * static_cast<std::size_t>(n_groups) +
         static_cast<std::size_t>(group);
}

int sn_boundary_code(const std::string& value) {
  if (value == "reflect") {
    return kSNBoundaryReflect;
  }
  if (value == "marshak") {
    return kSNBoundaryMarshak;
  }
  TENRYU_ASSERT(value == "vacuum", "unsupported SN boundary type");
  return kSNBoundaryVacuum;
}

__device__ inline void cell_geometry(const double* __restrict__ node_r,
                                     const double* __restrict__ node_z,
                                     const double* __restrict__ vol,
                                     const int i,
                                     const int j,
                                     const int nz,
                                     double* r_left,
                                     double* r_right,
                                     double* z_bottom,
                                     double* z_top,
                                     double* area_r_left,
                                     double* area_r_right,
                                     double* area_z,
                                     double* volume) {
  const int n00 = node_index(i, j, nz);
  const int n10 = node_index(i + 1, j, nz);
  const int n11 = node_index(i + 1, j + 1, nz);
  const int n01 = node_index(i, j + 1, nz);
  *r_left = fmax(0.5 * (finite_or_zero(node_r[n00]) +
                        finite_or_zero(node_r[n01])),
                 0.0);
  *r_right = fmax(0.5 * (finite_or_zero(node_r[n10]) +
                         finite_or_zero(node_r[n11])),
                  0.0);
  *z_bottom = 0.5 * (finite_or_zero(node_z[n00]) + finite_or_zero(node_z[n10]));
  *z_top = 0.5 * (finite_or_zero(node_z[n01]) + finite_or_zero(node_z[n11]));
  const double dz = fmax(fabs(*z_top - *z_bottom), 0.0);
  *area_r_left = kTwoPi * (*r_left) * dz;
  *area_r_right = kTwoPi * (*r_right) * dz;
  *area_z = kPi * fmax((*r_right) * (*r_right) - (*r_left) * (*r_left), 0.0);
  const int c = cell_index(i, j, nz);
  const double V_input = nonnegative_finite(vol[c]);
  *volume = (V_input > 0.0) ? V_input : (*area_z) * dz;
}

__host__ __device__ inline double positive_harmonic_mean(const double a,
                                                        const double b) {
  if (!(a > 0.0) || !(b > 0.0) || !isfinite(a) || !isfinite(b)) {
    return 0.0;
  }
  return 2.0 * a * b / fmax(a + b, 1.0e-300);
}

__host__ __device__ inline double smoothstep01(const double x) {
  const double t = fmin(fmax(x, 0.0), 1.0);
  return t * t * (3.0 - 2.0 * t);
}

__host__ __device__ inline double smooth_full_at_low(const double value,
                                                     const double full,
                                                     const double off) {
  if (!(off > full)) {
    return value <= full ? 1.0 : 0.0;
  }
  return 1.0 - smoothstep01((value - full) / (off - full));
}

__device__ inline void atomic_max_nonnegative_double(double* address,
                                                     const double value) {
  if (!(value >= 0.0) || !isfinite(value)) {
    return;
  }
  auto* ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *ull;
  unsigned long long assumed = old;
  const unsigned long long val = __double_as_longlong(value);
  while (__longlong_as_double(static_cast<long long>(assumed)) < value) {
    old = atomicCAS(ull, assumed, val);
    if (old == assumed) {
      break;
    }
    assumed = old;
  }
}

__global__ void fill_kernel(double* __restrict__ out, int n, double value) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = value;
  }
}

__global__ void copy_kernel(const double* __restrict__ src,
                            double* __restrict__ dst,
                            int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = finite_or_zero(src[i]);
  }
}

__global__ void initialize_phi_from_rad_E_kernel(const double* __restrict__ rad_E,
                                                 double* __restrict__ phi,
                                                 int n_total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_total) {
    phi[i] = fmax(finite_or_zero(rad_E[i]), 0.0) * core::constants::c_light;
  }
}

// Total-extinction gate fix (2D port): a thick pure-scattering cell
// is in the diffusion regime (AP-blend authority; E evolves via the flux
// form), not a void. Only genuinely transparent cells
// (c dt (sigma_a + sigma_s) below the band) anchor rad_E to the transport
// moments (void-anchor fix).
// 2D note: the converged sweep moments live in sn_phi_sweep (sn_phi_old is
// the step-start isotropic seed — do NOT anchor to it).
__global__ void anchor_void_rad_E_to_moments_2d_kernel(
    double* __restrict__ rad_E,
    const double* __restrict__ phi_sweep,
    const double* __restrict__ sigma_a,
    const double* __restrict__ sigma_s,
    int n_total,
    double dt) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_total) {
    return;
  }
  const double lambda_ext = fmax(dt, 0.0) * core::constants::c_light *
                            (nonnegative_finite(sigma_a[idx]) +
                             nonnegative_finite(sigma_s[idx]));
  constexpr double kLambdaLo = 1.0e-4;
  constexpr double kLambdaHi = 1.0e-3;
  if (lambda_ext >= kLambdaHi) {
    return;  // w == 1 exactly: flux-form value untouched (bitwise).
  }
  const double E_moments =
      nonnegative_finite(phi_sweep[idx]) / core::constants::c_light;
  double w = 0.0;
  if (lambda_ext > kLambdaLo) {
    const double t = (lambda_ext - kLambdaLo) / (kLambdaHi - kLambdaLo);
    w = smoothstep01(t);
  }
  rad_E[idx] = w * nonnegative_finite(rad_E[idx]) + (1.0 - w) * E_moments;
}

__global__ void max_relative_delta_kernel(const double* __restrict__ next,
                                          const double* __restrict__ prev,
                                          double* __restrict__ out,
                                          int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const double a = finite_or_zero(next[i]);
  const double b = finite_or_zero(prev[i]);
  const double rel = fabs(a - b) / fmax(fabs(a), 1.0e-300);
  atomic_max_nonnegative_double(out, rel);
}

// Outer-boundary escape-flux fix (2D port): the ledger books the sweep's discrete boundary
// face flux (the array the E*-flux update consumes) instead of the historic
// coefficient model 0.5cE (vacuum) / 0.25cE (marshak), which disagreed with
// the discrete outgoing tally by up to 2x for a near-isotropic field.
__global__ void escaped_energy_2d_kernel(const double* __restrict__ x_r,
                                         const double* __restrict__ x_z,
                                         const double* __restrict__ face_flux,
                                         double* __restrict__ out,
                                         int c_begin,
                                         int c_end,
                                         int nr,
                                         int nz,
                                         int n_groups,
                                         double dt,
                                         int r_outer_bc,
                                         int z_bottom_bc,
                                         int z_top_bc) {
  // Owned-window tally (Option C): per-rank partial over [c_begin, c_end)
  // completed by the driver budget Allreduce. Serial: (0, nr*nz).
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = (c_end - c_begin) * n_groups;
  if (idx >= total) {
    return;
  }
  const int c_local = idx / n_groups;
  const int g = idx - c_local * n_groups;
  const int c = c_begin + c_local;
  const int i = c / nz;
  const int j = c - i * nz;
  if (i + 1 < nr && j > 0 && j + 1 < nz) {
    return;
  }
  const int n00 = node_index(i, j, nz);
  const int n10 = node_index(i + 1, j, nz);
  const int n11 = node_index(i + 1, j + 1, nz);
  const int n01 = node_index(i, j + 1, nz);
  const double r_left = fmax(0.5 * (finite_or_zero(x_r[n00]) +
                                    finite_or_zero(x_r[n01])),
                             0.0);
  const double r_right = fmax(0.5 * (finite_or_zero(x_r[n10]) +
                                     finite_or_zero(x_r[n11])),
                              0.0);
  const double z_bottom =
      0.5 * (finite_or_zero(x_z[n00]) + finite_or_zero(x_z[n10]));
  const double z_top =
      0.5 * (finite_or_zero(x_z[n01]) + finite_or_zero(x_z[n11]));
  const double dz = fmax(fabs(z_top - z_bottom), 0.0);
  const double area_r_right = kTwoPi * r_right * dz;
  const double area_z = kPi * fmax(r_right * r_right - r_left * r_left, 0.0);
  double leak = 0.0;
  if (i + 1 == nr && r_outer_bc != kSNBoundaryReflect) {
    const int f = r_face_index(nr, j, nz);
    leak += area_r_right *
            finite_or_zero(face_flux[face_group_index(f, g, n_groups)]);
  }
  if (j == 0 && z_bottom_bc != kSNBoundaryReflect) {
    const int f = z_face_index(i, 0, nr, nz);
    leak -= area_z *
            finite_or_zero(face_flux[face_group_index(f, g, n_groups)]);
  }
  if (j + 1 == nz && z_top_bc != kSNBoundaryReflect) {
    const int f = z_face_index(i, nz, nr, nz);
    leak += area_z *
            finite_or_zero(face_flux[face_group_index(f, g, n_groups)]);
  }
  atomicAdd(out, dt * leak);
}

__global__ void marshak_in_energy_2d_kernel(const double* __restrict__ x_r,
                                            const double* __restrict__ x_z,
                                            double* __restrict__ out,
                                            int i_begin,
                                            int i_end,
                                            int nr,
                                            int nz,
                                            double dt,
                                            int z_bottom_bc,
                                            int z_top_bc,
                                            double marshak_flux_erg_per_cm2_s) {
  // Owned radial range (Option C): per-rank partial over [i_begin, i_end).
  const int i = i_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= i_end || !(marshak_flux_erg_per_cm2_s > 0.0)) {
    return;
  }
  const int n00 = node_index(i, 0, nz);
  const int n10 = node_index(i + 1, 0, nz);
  const int n0t = node_index(i, nz, nz);
  const int n1t = node_index(i + 1, nz, nz);
  double area = 0.0;
  if (z_bottom_bc == kSNBoundaryMarshak) {
    const double r_left = fmax(finite_or_zero(x_r[n00]), 0.0);
    const double r_right = fmax(finite_or_zero(x_r[n10]), 0.0);
    area += kPi * fmax(r_right * r_right - r_left * r_left, 0.0);
  }
  if (z_top_bc == kSNBoundaryMarshak) {
    const double r_left = fmax(finite_or_zero(x_r[n0t]), 0.0);
    const double r_right = fmax(finite_or_zero(x_r[n1t]), 0.0);
    area += kPi * fmax(r_right * r_right - r_left * r_left, 0.0);
  }
  atomicAdd(out, fmax(dt, 0.0) * area * marshak_flux_erg_per_cm2_s);
}

__global__ void publish_sn_fixup_tallies_2d_kernel(
    const unsigned long long* __restrict__ count_in,
    const double* __restrict__ abs_in,
    double* __restrict__ count_out,
    double* __restrict__ abs_out,
    int n_total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_total) {
    return;
  }
  count_out[idx] = static_cast<double>(count_in[idx]);
  abs_out[idx] = finite_or_zero(abs_in[idx]);
}

__global__ void reduce_ap_alpha_activity_2d_kernel(
    const double* __restrict__ alpha_face,
    double* __restrict__ block_max,
    unsigned long long* __restrict__ block_count,
    int n_total) {
  __shared__ double s_max[kBlock];
  __shared__ unsigned long long s_count[kBlock];
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  double local_max = 0.0;
  unsigned long long local_count = 0ULL;
  if (idx < n_total) {
    const double alpha = finite_or_zero(alpha_face[idx]);
    local_max = fmax(alpha, 0.0);
    local_count = (alpha > 0.0) ? 1ULL : 0ULL;
  }
  s_max[tid] = local_max;
  s_count[tid] = local_count;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_max[tid] = fmax(s_max[tid], s_max[tid + stride]);
      s_count[tid] += s_count[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    block_max[blockIdx.x] = s_max[0];
    block_count[blockIdx.x] = s_count[0];
  }
}

__global__ void zero_axis_faces_checked_kernel(double* __restrict__ face_field,
                                               double* __restrict__ bad_flag,
                                               int nz,
                                               int n_groups) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = nz * n_groups;
  if (idx >= total || face_field == nullptr) {
    return;
  }
  const double value = face_field[idx];
  if ((!isfinite(value) || value != 0.0) && bad_flag != nullptr) {
    bad_flag[0] = 1.0;
  }
  face_field[idx] = 0.0;
}

__global__ void compute_E_star_flux_2d_kernel(
    const double* __restrict__ rad_E_old,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const double* __restrict__ vol,
    const double* __restrict__ face_flux,
    double* __restrict__ E_star,
    double* __restrict__ diag_E_star_flux,
    int nr,
    int nz,
    int n_groups,
    double dt) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int c = idx / n_groups;
  const int g = idx - c * n_groups;
  const int i = c / nz;
  const int j = c - i * nz;
  double r_left = 0.0;
  double r_right = 0.0;
  double z_bottom = 0.0;
  double z_top = 0.0;
  double area_r_left = 0.0;
  double area_r_right = 0.0;
  double area_z = 0.0;
  double V = 0.0;
  cell_geometry(node_r, node_z, vol, i, j, nz,
                &r_left, &r_right, &z_bottom, &z_top,
                &area_r_left, &area_r_right, &area_z, &V);
  const double F_r_left =
      finite_or_zero(face_flux[face_group_index(r_face_index(i, j, nz), g, n_groups)]);
  const double F_r_right =
      finite_or_zero(face_flux[face_group_index(r_face_index(i + 1, j, nz), g, n_groups)]);
  const double F_z_bottom =
      finite_or_zero(face_flux[face_group_index(z_face_index(i, j, nr, nz), g, n_groups)]);
  const double F_z_top =
      finite_or_zero(face_flux[face_group_index(z_face_index(i, j + 1, nr, nz), g, n_groups)]);
  const double div = (area_r_right * F_r_right - area_r_left * F_r_left +
                      area_z * F_z_top - area_z * F_z_bottom) /
                     fmax(nonnegative_finite(V), 1.0e-300);
  const double Estar = finite_or_zero(rad_E_old[idx]) - fmax(dt, 0.0) * div;
  const double value = isfinite(Estar) ? Estar : 0.0;
  E_star[idx] = value;
  if (diag_E_star_flux != nullptr) {
    diag_E_star_flux[idx] = value;
  }
}

__global__ void compute_diffusion_face_flux_2d_kernel(
    const double* __restrict__ rad_E_old,
    const double* __restrict__ sn_sigma_s,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const double* __restrict__ vol,
    double opacity_floor,
    double* __restrict__ face_flux_diff,
    int nr,
    int nz,
    int n_groups) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_faces(nr, nz) * n_groups;
  if (idx >= total) {
    return;
  }
  const int face = idx / n_groups;
  const int g = idx - face * n_groups;
  const int nrf = n_r_faces(nr, nz);
  double flux = 0.0;
  if (face < nrf) {
    const int i_face = face / nz;
    const int j = face - i_face * nz;
    if (i_face > 0 && i_face < nr) {
      const int c_l = cell_index(i_face - 1, j, nz);
      const int c_r = cell_index(i_face, j, nz);
      double rl0 = 0.0;
      double rr0 = 0.0;
      double zl0 = 0.0;
      double zt0 = 0.0;
      double ar0 = 0.0;
      double ar1 = 0.0;
      double az0 = 0.0;
      double V0 = 0.0;
      double rl1 = 0.0;
      double rr1 = 0.0;
      double zl1 = 0.0;
      double zt1 = 0.0;
      double dummy = 0.0;
      cell_geometry(node_r, node_z, vol, i_face - 1, j, nz,
                    &rl0, &rr0, &zl0, &zt0, &ar0, &ar1, &az0, &V0);
      cell_geometry(node_r, node_z, vol, i_face, j, nz,
                    &rl1, &rr1, &zl1, &zt1, &dummy, &dummy, &dummy, &dummy);
      const double sigma_l =
          fmax(nonnegative_finite(sn_sigma_s[cell_group_index(c_l, g, n_groups)]),
               opacity_floor);
      const double sigma_r =
          fmax(nonnegative_finite(sn_sigma_s[cell_group_index(c_r, g, n_groups)]),
               opacity_floor);
      const double D_l = core::constants::c_light / fmax(3.0 * sigma_l, 1.0e-100);
      const double D_r = core::constants::c_light / fmax(3.0 * sigma_r, 1.0e-100);
      const double D_face = positive_harmonic_mean(D_l, D_r);
      const double E_l = nonnegative_finite(rad_E_old[cell_group_index(c_l, g, n_groups)]);
      const double E_r = nonnegative_finite(rad_E_old[cell_group_index(c_r, g, n_groups)]);
      const double rc_l = 0.5 * (rl0 + rr0);
      const double rc_r = 0.5 * (rl1 + rr1);
      flux = -D_face * (E_r - E_l) / fmax(rc_r - rc_l, 1.0e-300);
    }
  } else {
    const int local = face - nrf;
    const int i = local / (nz + 1);
    const int j_face = local - i * (nz + 1);
    if (j_face > 0 && j_face < nz) {
      const int c_b = cell_index(i, j_face - 1, nz);
      const int c_t = cell_index(i, j_face, nz);
      double rb0 = 0.0;
      double rr0 = 0.0;
      double zb0 = 0.0;
      double zt0 = 0.0;
      double dummy = 0.0;
      double rb1 = 0.0;
      double rr1 = 0.0;
      double zb1 = 0.0;
      double zt1 = 0.0;
      cell_geometry(node_r, node_z, vol, i, j_face - 1, nz,
                    &rb0, &rr0, &zb0, &zt0, &dummy, &dummy, &dummy, &dummy);
      cell_geometry(node_r, node_z, vol, i, j_face, nz,
                    &rb1, &rr1, &zb1, &zt1, &dummy, &dummy, &dummy, &dummy);
      const double sigma_b =
          fmax(nonnegative_finite(sn_sigma_s[cell_group_index(c_b, g, n_groups)]),
               opacity_floor);
      const double sigma_t =
          fmax(nonnegative_finite(sn_sigma_s[cell_group_index(c_t, g, n_groups)]),
               opacity_floor);
      const double D_b = core::constants::c_light / fmax(3.0 * sigma_b, 1.0e-100);
      const double D_t = core::constants::c_light / fmax(3.0 * sigma_t, 1.0e-100);
      const double D_face = positive_harmonic_mean(D_b, D_t);
      const double E_b = nonnegative_finite(rad_E_old[cell_group_index(c_b, g, n_groups)]);
      const double E_t = nonnegative_finite(rad_E_old[cell_group_index(c_t, g, n_groups)]);
      const double zc_b = 0.5 * (zb0 + zt0);
      const double zc_t = 0.5 * (zb1 + zt1);
      flux = -D_face * (E_t - E_b) / fmax(zc_t - zc_b, 1.0e-300);
    }
  }
  face_flux_diff[idx] = isfinite(flux) ? flux : 0.0;
}

__global__ void compute_ap_blended_face_flux_2d_kernel(
    const double* __restrict__ face_flux_sn,
    const double* __restrict__ face_flux_diff,
    const double* __restrict__ rad_E_old,
    const double* __restrict__ Te,
    const double* __restrict__ sn_sigma_s,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const double* __restrict__ vol,
    PlanckTableDeviceView planck,
    double opacity_floor,
    double tau_lo,
    double tau_hi,
    double eq_full,
    double eq_off,
    double f_full,
    double f_off,
    double* __restrict__ face_flux_blended,
    double* __restrict__ alpha_face,
    int nr,
    int nz,
    int n_groups) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_faces(nr, nz) * n_groups;
  if (idx >= total) {
    return;
  }
  const int face = idx / n_groups;
  const int g = idx - face * n_groups;
  const int nrf = n_r_faces(nr, nz);
  int c_l = -1;
  int c_r = -1;
  double width_l = 0.0;
  double width_r = 0.0;
  bool boundary = false;
  if (face < nrf) {
    const int i_face = face / nz;
    const int j = face - i_face * nz;
    boundary = (i_face == 0 || i_face == nr);
    if (!boundary) {
      c_l = cell_index(i_face - 1, j, nz);
      c_r = cell_index(i_face, j, nz);
      double rl = 0.0;
      double rr = 0.0;
      double zb = 0.0;
      double zt = 0.0;
      double dummy = 0.0;
      cell_geometry(node_r, node_z, vol, i_face - 1, j, nz,
                    &rl, &rr, &zb, &zt, &dummy, &dummy, &dummy, &dummy);
      width_l = fmax(rr - rl, 1.0e-300);
      cell_geometry(node_r, node_z, vol, i_face, j, nz,
                    &rl, &rr, &zb, &zt, &dummy, &dummy, &dummy, &dummy);
      width_r = fmax(rr - rl, 1.0e-300);
    }
  } else {
    const int local = face - nrf;
    const int i = local / (nz + 1);
    const int j_face = local - i * (nz + 1);
    boundary = (j_face == 0 || j_face == nz);
    if (!boundary) {
      c_l = cell_index(i, j_face - 1, nz);
      c_r = cell_index(i, j_face, nz);
      double rl = 0.0;
      double rr = 0.0;
      double zb = 0.0;
      double zt = 0.0;
      double dummy = 0.0;
      cell_geometry(node_r, node_z, vol, i, j_face - 1, nz,
                    &rl, &rr, &zb, &zt, &dummy, &dummy, &dummy, &dummy);
      width_l = fmax(fabs(zt - zb), 1.0e-300);
      cell_geometry(node_r, node_z, vol, i, j_face, nz,
                    &rl, &rr, &zb, &zt, &dummy, &dummy, &dummy, &dummy);
      width_r = fmax(fabs(zt - zb), 1.0e-300);
    }
  }
  if (boundary || c_l < 0 || c_r < 0) {
    face_flux_blended[idx] = finite_or_zero(face_flux_sn[idx]);
    alpha_face[idx] = 0.0;
    return;
  }
  const double sigma_l =
      fmax(nonnegative_finite(sn_sigma_s[cell_group_index(c_l, g, n_groups)]),
           opacity_floor);
  const double sigma_r =
      fmax(nonnegative_finite(sn_sigma_s[cell_group_index(c_r, g, n_groups)]),
           opacity_floor);
  const double tau_min = fmin(sigma_l * width_l, sigma_r * width_r);
  const double alpha_tau =
      smoothstep01((tau_min - tau_lo) / fmax(tau_hi - tau_lo, 1.0e-300));
  const double T_l = fmax(finite_or_zero(Te[c_l]), 1.0e-3);
  const double T_r = fmax(finite_or_zero(Te[c_r]), 1.0e-3);
  const double E_l = nonnegative_finite(rad_E_old[cell_group_index(c_l, g, n_groups)]);
  const double E_r = nonnegative_finite(rad_E_old[cell_group_index(c_r, g, n_groups)]);
  const double b_l = (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T_l), 0.0);
  const double b_r = (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T_r), 0.0);
  const double T2_l = T_l * T_l;
  const double T2_r = T_r * T_r;
  const double B_l = core::constants::a_eV * T2_l * T2_l * b_l;
  const double B_r = core::constants::a_eV * T2_r * T2_r * b_r;
  const double err_l = fabs(E_l - B_l) / fmax(B_l, 1.0e-30);
  const double err_r = fabs(E_r - B_r) / fmax(B_r, 1.0e-30);
  const double alpha_eq = smooth_full_at_low(fmax(err_l, err_r), eq_full, eq_off);
  const double F_abs = fabs(finite_or_zero(face_flux_sn[idx]));
  const double E_avg = 0.5 * (E_l + E_r);
  const double f_red = F_abs / fmax(core::constants::c_light * E_avg, 1.0e-300);
  const double alpha_f = smooth_full_at_low(f_red, f_full, f_off);
  const double alpha = alpha_tau * alpha_eq * alpha_f;
  const double F_sn = finite_or_zero(face_flux_sn[idx]);
  const double F_diff = finite_or_zero(face_flux_diff[idx]);
  face_flux_blended[idx] =
      (alpha <= 0.0) ? F_sn : ((alpha >= 1.0) ? F_diff : (1.0 - alpha) * F_sn + alpha * F_diff);
  alpha_face[idx] = alpha;
}

__global__ void compute_ap_cell_diagnostics_2d_kernel(
    const double* __restrict__ face_flux_sn,
    const double* __restrict__ rad_E_old,
    const double* __restrict__ sn_sigma_s,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const double* __restrict__ vol,
    const double* __restrict__ alpha_face,
    double opacity_floor,
    double* __restrict__ sn_tau_R,
    double* __restrict__ sn_reduced_flux,
    double* __restrict__ sn_ap_alpha,
    int nr,
    int nz,
    int n_groups) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  double r_left = 0.0;
  double r_right = 0.0;
  double z_bottom = 0.0;
  double z_top = 0.0;
  double area_r_left = 0.0;
  double area_r_right = 0.0;
  double area_z = 0.0;
  double V = 0.0;
  cell_geometry(node_r, node_z, vol, i, j, nz,
                &r_left, &r_right, &z_bottom, &z_top,
                &area_r_left, &area_r_right, &area_z, &V);
  const double width = fmax(fmin(fabs(r_right - r_left), fabs(z_top - z_bottom)),
                           1.0e-300);

  double tau_min = 1.0e300;
  double reduced_flux_max = 0.0;
  double alpha_max = 0.0;
  int faces[4] = {
      r_face_index(i, j, nz),
      r_face_index(i + 1, j, nz),
      z_face_index(i, j, nr, nz),
      z_face_index(i, j + 1, nr, nz)};
  int neighbors[4] = {-1, -1, -1, -1};
  if (i > 0) {
    neighbors[0] = cell_index(i - 1, j, nz);
  }
  if (i + 1 < nr) {
    neighbors[1] = cell_index(i + 1, j, nz);
  }
  if (j > 0) {
    neighbors[2] = cell_index(i, j - 1, nz);
  }
  if (j + 1 < nz) {
    neighbors[3] = cell_index(i, j + 1, nz);
  }

  for (int g = 0; g < n_groups; ++g) {
    const int cg = c * n_groups + g;
    const double sigma =
        fmax(nonnegative_finite(sn_sigma_s[cg]), opacity_floor);
    tau_min = fmin(tau_min, sigma * width);
    const double E_c = nonnegative_finite(rad_E_old[cg]);
    for (int face_slot = 0; face_slot < 4; ++face_slot) {
      const int fg = faces[face_slot] * n_groups + g;
      const double a = fmin(fmax(finite_or_zero(alpha_face[fg]), 0.0), 1.0);
      alpha_max = fmax(alpha_max, a);
      const int cn = neighbors[face_slot];
      if (cn < 0) {
        continue;
      }
      const double E_avg =
          0.5 * (E_c + nonnegative_finite(rad_E_old[cn * n_groups + g]));
      const double f_red = fabs(finite_or_zero(face_flux_sn[fg])) /
                           fmax(core::constants::c_light * E_avg, 1.0e-300);
      if (isfinite(f_red)) {
        reduced_flux_max = fmax(reduced_flux_max, f_red);
      }
    }
  }
  sn_tau_R[c] = (tau_min < 1.0e300 && isfinite(tau_min)) ? tau_min : 0.0;
  sn_reduced_flux[c] = isfinite(reduced_flux_max) ? reduced_flux_max : 0.0;
  sn_ap_alpha[c] = isfinite(alpha_max) ? alpha_max : 0.0;
}

// Pass-1 donor-only availability (2D port; no inflow credit).
// Used to throttle the inflow credit in pass 2 so positivity is exact:
// E_new = E_old + dt (theta_up·in − theta'·out)/V >= 0 by construction,
// while a uniformly throttled stream relaxes to full pass-through.
__global__ void compute_streaming_theta_donor_2d_kernel(
    const double* __restrict__ rad_E_old,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const double* __restrict__ vol,
    const double* __restrict__ face_flux,
    double* __restrict__ theta_donor,
    int nr,
    int nz,
    int n_groups,
    double dt) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int c = idx / n_groups;
  const int g = idx - c * n_groups;
  const int i = c / nz;
  const int j = c - i * nz;
  double r_left = 0.0;
  double r_right = 0.0;
  double z_bottom = 0.0;
  double z_top = 0.0;
  double area_r_left = 0.0;
  double area_r_right = 0.0;
  double area_z = 0.0;
  double V = 0.0;
  cell_geometry(node_r, node_z, vol, i, j, nz,
                &r_left, &r_right, &z_bottom, &z_top,
                &area_r_left, &area_r_right, &area_z, &V);
  const double F_left =
      (i == 0) ? 0.0 : finite_or_zero(
          face_flux[face_group_index(r_face_index(i, j, nz), g, n_groups)]);
  const double F_right =
      finite_or_zero(face_flux[face_group_index(r_face_index(i + 1, j, nz), g, n_groups)]);
  const double F_bottom =
      finite_or_zero(face_flux[face_group_index(z_face_index(i, j, nr, nz), g, n_groups)]);
  const double F_top =
      finite_or_zero(face_flux[face_group_index(z_face_index(i, j + 1, nr, nz), g, n_groups)]);
  const double clamped_dt = fmax(dt, 0.0);
  const double inv_V = 1.0 / fmax(nonnegative_finite(V), 1.0e-300);
  const double out_power =
      fmax(-F_left, 0.0) * area_r_left +
      fmax(F_right, 0.0) * area_r_right +
      fmax(-F_bottom, 0.0) * area_z +
      fmax(F_top, 0.0) * area_z;
  const double available = nonnegative_finite(rad_E_old[idx]);
  const double out = clamped_dt * out_power * inv_V;
  theta_donor[idx] = (out > 1.0e-300) ? fmin(1.0, available / out) : 1.0;
}

__global__ void compute_streaming_theta_2d_kernel(
    const double* __restrict__ rad_E_old,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const double* __restrict__ vol,
    const double* __restrict__ face_flux,
    const double* __restrict__ theta_donor,
    double* __restrict__ theta,
    int nr,
    int nz,
    int n_groups,
    double dt) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int c = idx / n_groups;
  const int g = idx - c * n_groups;
  const int i = c / nz;
  const int j = c - i * nz;
  double r_left = 0.0;
  double r_right = 0.0;
  double z_bottom = 0.0;
  double z_top = 0.0;
  double area_r_left = 0.0;
  double area_r_right = 0.0;
  double area_z = 0.0;
  double V = 0.0;
  cell_geometry(node_r, node_z, vol, i, j, nz,
                &r_left, &r_right, &z_bottom, &z_top,
                &area_r_left, &area_r_right, &area_z, &V);
  const double F_left =
      (i == 0) ? 0.0 : finite_or_zero(
          face_flux[face_group_index(r_face_index(i, j, nz), g, n_groups)]);
  const double F_right =
      finite_or_zero(face_flux[face_group_index(r_face_index(i + 1, j, nz), g, n_groups)]);
  const double F_bottom =
      finite_or_zero(face_flux[face_group_index(z_face_index(i, j, nr, nz), g, n_groups)]);
  const double F_top =
      finite_or_zero(face_flux[face_group_index(z_face_index(i, j + 1, nr, nz), g, n_groups)]);
  const double clamped_dt = fmax(dt, 0.0);
  const double inv_V = 1.0 / fmax(nonnegative_finite(V), 1.0e-300);
  const double th_left =
      (i > 0) ? finite_or_zero(
                    theta_donor[static_cast<std::size_t>((i - 1) * nz + j) *
                                    static_cast<std::size_t>(n_groups) +
                                static_cast<std::size_t>(g)])
              : 1.0;
  const double th_right =
      (i + 1 < nr) ? finite_or_zero(
                         theta_donor[static_cast<std::size_t>((i + 1) * nz + j) *
                                         static_cast<std::size_t>(n_groups) +
                                     static_cast<std::size_t>(g)])
                   : 1.0;
  const double th_bottom =
      (j > 0) ? finite_or_zero(
                    theta_donor[static_cast<std::size_t>(i * nz + (j - 1)) *
                                    static_cast<std::size_t>(n_groups) +
                                static_cast<std::size_t>(g)])
              : 1.0;
  const double th_top =
      (j + 1 < nz) ? finite_or_zero(
                         theta_donor[static_cast<std::size_t>(i * nz + (j + 1)) *
                                         static_cast<std::size_t>(n_groups) +
                                     static_cast<std::size_t>(g)])
                   : 1.0;
  const double in_power =
      th_left * fmax(F_left, 0.0) * area_r_left +
      th_right * fmax(-F_right, 0.0) * area_r_right +
      th_bottom * fmax(F_bottom, 0.0) * area_z +
      th_top * fmax(-F_top, 0.0) * area_z;
  const double out_power =
      fmax(-F_left, 0.0) * area_r_left +
      fmax(F_right, 0.0) * area_r_right +
      fmax(-F_bottom, 0.0) * area_z +
      fmax(F_top, 0.0) * area_z;
  const double available =
      nonnegative_finite(rad_E_old[idx]) + clamped_dt * in_power * inv_V;
  const double out = clamped_dt * out_power * inv_V;
  theta[idx] = (out > 1.0e-300) ? fmin(1.0, available / out) : 1.0;
}

__global__ void apply_donor_theta_face_flux_2d_kernel(
    const double* __restrict__ face_flux_raw,
    const double* __restrict__ theta,
    double* __restrict__ face_flux_limited,
    int nr,
    int nz,
    int n_groups) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_faces(nr, nz) * n_groups;
  if (idx >= total) {
    return;
  }
  const int face = idx / n_groups;
  const int g = idx - face * n_groups;
  const int nrf = n_r_faces(nr, nz);
  const double F = finite_or_zero(face_flux_raw[idx]);
  double th = 1.0;
  if (face < nrf) {
    const int i_face = face / nz;
    const int j = face - i_face * nz;
    if (i_face == 0) {
      face_flux_limited[idx] = 0.0;
      return;
    }
    if (F > 0.0) {
      th = (i_face > 0)
               ? theta[cell_group_index(cell_index(i_face - 1, j, nz), g, n_groups)]
               : 1.0;
    } else if (F < 0.0) {
      th = (i_face < nr)
               ? theta[cell_group_index(cell_index(i_face, j, nz), g, n_groups)]
               : 1.0;
    }
  } else {
    const int local = face - nrf;
    const int i = local / (nz + 1);
    const int j_face = local - i * (nz + 1);
    if (F > 0.0) {
      th = (j_face > 0)
               ? theta[cell_group_index(cell_index(i, j_face - 1, nz), g, n_groups)]
               : 1.0;
    } else if (F < 0.0) {
      th = (j_face < nz)
               ? theta[cell_group_index(cell_index(i, j_face, nz), g, n_groups)]
               : 1.0;
    }
  }
  face_flux_limited[idx] = th * F;
}

std::vector<double> radiation_group_bounds(const core::Config& cfg,
                                           const int n_groups) {
  if (static_cast<int>(cfg.radiation.group_bounds_eV.size()) == n_groups + 1) {
    return cfg.radiation.group_bounds_eV;
  }
  const std::vector<double> range = resolve_compute_T_range_eV(cfg, false);
  const double Tmin = std::max(range[0], 1.0e-3);
  const double Tmax = std::max(range[1], Tmin * 1.001);
  return Groups::make_log_uniform_bounds(n_groups, Tmin, Tmax);
}

struct DeviceOpacityTable {
  double* d_log_temps = nullptr;
  double* d_log_numdens = nullptr;
  double* d_kappa_PA = nullptr;
  double* d_kappa_PE = nullptr;
  double* d_kappa_R = nullptr;
  int ntemp = 0;
  int ndens = 0;
  int ngroups = 0;
  double log_T_min = 0.0;
  double log_T_max = 0.0;
  double log_ni_min = 0.0;
  double log_ni_max = 0.0;
  double T_min = 0.0;
  double T_max = 0.0;

  DeviceOpacityTable() = default;
  DeviceOpacityTable(const DeviceOpacityTable&) = delete;
  DeviceOpacityTable& operator=(const DeviceOpacityTable&) = delete;

  ~DeviceOpacityTable() {
    release();
  }

  void upload(const materials::IonmixOpacityData& host) {
    release();
    const std::size_t n_table =
        static_cast<std::size_t>(host.ngroups) *
        static_cast<std::size_t>(host.ndens) *
        static_cast<std::size_t>(host.ntemp);
    TENRYU_ASSERT(host.kappa_PA.size() == n_table &&
                      host.kappa_PE.size() == n_table &&
                      host.kappa_R.size() == n_table,
                  "SN2D opacity table size mismatch");
    const auto upload_vec = [](double** dst,
                               const std::vector<double>& src,
                               const char* label) {
      cuda_check(cudaMalloc(reinterpret_cast<void**>(dst),
                            sizeof(double) * src.size()),
                 label);
      cuda_check(cudaMemcpy(*dst,
                            src.data(),
                            sizeof(double) * src.size(),
                            cudaMemcpyHostToDevice),
                 label);
    };
    upload_vec(&d_log_temps, host.log_temps, "SN2D upload log_temps failed");
    upload_vec(&d_log_numdens, host.log_numdens, "SN2D upload log_numdens failed");
    upload_vec(&d_kappa_PA, host.kappa_PA, "SN2D upload kappa_PA failed");
    upload_vec(&d_kappa_PE, host.kappa_PE, "SN2D upload kappa_PE failed");
    upload_vec(&d_kappa_R, host.kappa_R, "SN2D upload kappa_R failed");
    ntemp = host.ntemp;
    ndens = host.ndens;
    ngroups = host.ngroups;
    log_T_min = host.log_temps.front();
    log_T_max = host.log_temps.back();
    log_ni_min = host.log_numdens.front();
    log_ni_max = host.log_numdens.back();
    T_min = host.temps_eV.front();
    T_max = host.temps_eV.back();
  }

  void release() {
    if (d_kappa_R != nullptr) {
      cuda_check(cudaFree(d_kappa_R), "SN2D free kappa_R failed");
      d_kappa_R = nullptr;
    }
    if (d_kappa_PE != nullptr) {
      cuda_check(cudaFree(d_kappa_PE), "SN2D free kappa_PE failed");
      d_kappa_PE = nullptr;
    }
    if (d_kappa_PA != nullptr) {
      cuda_check(cudaFree(d_kappa_PA), "SN2D free kappa_PA failed");
      d_kappa_PA = nullptr;
    }
    if (d_log_numdens != nullptr) {
      cuda_check(cudaFree(d_log_numdens), "SN2D free log_numdens failed");
      d_log_numdens = nullptr;
    }
    if (d_log_temps != nullptr) {
      cuda_check(cudaFree(d_log_temps), "SN2D free log_temps failed");
      d_log_temps = nullptr;
    }
  }

  materials::IonmixOpacityDeviceView view() const {
    materials::IonmixOpacityDeviceView out{};
    out.log_temps = d_log_temps;
    out.log_numdens = d_log_numdens;
    out.kappa_PA = d_kappa_PA;
    out.kappa_PE = d_kappa_PE;
    out.kappa_R = d_kappa_R;
    out.ntemp = ntemp;
    out.ndens = ndens;
    out.ngroups = ngroups;
    out.log_T_min = log_T_min;
    out.log_T_max = log_T_max;
    out.log_ni_min = log_ni_min;
    out.log_ni_max = log_ni_max;
    out.T_min = T_min;
    out.T_max = T_max;
    return out;
  }
};

struct NlteOpacityCache {
  std::unique_ptr<materials::IonmixOpacityData> host;
  DeviceOpacityTable device;
  std::string file;
  int groups = 0;
};

NlteOpacityCache& nlte_cache() {
  static NlteOpacityCache cache;
  return cache;
}

void ensure_nlte_table_uploaded(const core::Config& cfg,
                                const core::Config::MaterialsConfig::MatDef& mat,
                                const int n_groups) {
  auto& cache = nlte_cache();
  if (cache.host != nullptr && cache.file == mat.opacity_file &&
      cache.groups == n_groups) {
    return;
  }
  if (mat.opacity_model == "tmat") {
    const materials::TmatFile tmat = materials::load_tmat(mat.opacity_file);
    TENRYU_ASSERT(tmat.opacity.has_value(),
                  "SN2D tmat opacity.model requires /opacity payload");
    cache.host = std::make_unique<materials::IonmixOpacityData>(
        materials::tmat_to_ionmix_opacity(*tmat.opacity));
  } else {
    cache.host = std::make_unique<materials::IonmixOpacityData>(
        materials::load_ionmix_opacity(mat.opacity_file));
  }
  if (cfg.radiation.group_repack_hard_xray) {
    std::vector<double> target_bounds = cfg.radiation.group_bounds_eV;
    if (target_bounds.empty()) {
      const std::vector<double> range = resolve_compute_T_range_eV(cfg, false);
      target_bounds = repack_group_bounds_for_hard_xray(
          n_groups, range);
    }
    *cache.host = resample_opacity_groups_to_bounds(*cache.host, target_bounds);
  }
  TENRYU_ASSERT(cache.host->ngroups == n_groups,
                "SN2D NLTE opacity table group count must match Radiation.groups");
  cache.device.upload(*cache.host);
  cache.file = mat.opacity_file;
  cache.groups = n_groups;
}

void ensure_state_buffers(core::State& state,
                          const int n_cells,
                          const int n_faces,
                          const int n_groups) {
  const std::size_t n_total =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  const std::size_t n_face_total =
      static_cast<std::size_t>(n_faces) * static_cast<std::size_t>(n_groups);
  state.rad_E.reset(n_total);
  state.rad_dep.reset(n_total);
  state.rad_emit.reset(n_total);
  state.rad_E_old.reset(n_total);
  state.sn_sigma_a.reset(n_total);
  state.sn_rad_E_pre_newton.reset(n_total);
  state.sn_sigma_pe.reset(n_total);
  state.sn_sigma_s.reset(n_total);
  state.sn_eta.reset(n_total);
  state.sn_phi_old.reset(n_total);
  state.sn_phi_new.reset(n_total);
  state.sn_phi_sweep.reset(n_total);
  state.sn_theta_donor.reset(n_total);
  state.sn_diag_rad_E_pre.reset(n_total);
  state.sn_diag_rad_E_post.reset(n_total);
  state.sn_diag_rad_emission_at_Tn.reset(n_total);
  state.sn_diag_rad_emission_at_Tnp1.reset(n_total);
  state.sn_diag_rad_absorption.reset(n_total);
  state.sn_diag_clip_energy.reset(n_total);
  state.sn_diag_clip_full_deficit.reset(n_total);
  state.sn_diag_chi_opacity.reset(n_total);
  state.sn_diag_F_first_moment.reset(n_total);
  state.sn_face_flux_raw.reset(n_face_total);
  state.sn_face_flux_diff.reset(n_face_total);
  state.sn_face_flux_blended.reset(n_face_total);
  state.sn_face_flux_limited.reset(n_face_total);
  state.sn_face_alpha.reset(n_face_total);
  state.sn_stream_theta.reset(n_total);
  state.sn_E_star_flux.reset(n_total);
  state.sn_diag_E_star_flux.reset(n_total);
  state.sn_tau_R.reset(static_cast<std::size_t>(n_cells));
  state.sn_reduced_flux.reset(static_cast<std::size_t>(n_cells));
  state.sn_ap_alpha.reset(static_cast<std::size_t>(n_cells));
  state.sn_radial_fixup_count.reset(n_total);
  state.sn_radial_fixup_artificial_abs.reset(n_total);
  state.sn_angular_fixup_count.reset(n_total);
  state.sn_angular_fixup_artificial_abs.reset(n_total);
  state.sn_Prr.reset(n_total);
  state.sn_chi.reset(n_total);
  state.sn_F_z.reset(n_total);
  state.sn_P_zz.reset(n_total);
  state.sn_chi_z.reset(n_total);
  state.sn_dsa_rhs.reset(n_total);
  state.sn_Te_old.reset(static_cast<std::size_t>(n_cells));
  state.sn_delta_T.reset(static_cast<std::size_t>(n_cells));
  state.sn_reduction_work.reset(1);
}

void initialize_rad_E_old_if_needed(core::State& state,
                                    const int n_cells,
                                    const int n_groups) {
  const std::size_t bytes =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups) *
      sizeof(double);
  if (bytes == 0U) {
    return;
  }
  if (state.step == 0 || state.holo_ale_invalidated) {
    cuda_check(cudaMemcpy(state.rad_E_old.data(),
                          state.rad_E.data(),
                          bytes,
                          cudaMemcpyDeviceToDevice),
               "SN2D initialize rad_E_old failed");
  }
}

void evaluate_opacity(core::State& state,
                      const core::Config& cfg,
                      const PlanckTable& planck,
                      const core::Config::MaterialsConfig::MatDef& mat,
                      const int n_cells,
                      const int n_groups,
                      const double dt) {
  const auto& sn = cfg.radiation.sn_transport;
  const int total = n_cells * n_groups;
  const bool use_nlte =
      mat.opacity_model == "table_nlte" || mat.opacity_model == "tmat";
  if (use_nlte) {
    ensure_nlte_table_uploaded(cfg, mat, n_groups);
    const std::size_t scratch_size = static_cast<std::size_t>(total);
    state.sn_nlte_f_work.reset(scratch_size);
    state.sn_nlte_sigma_eff_work.reset(scratch_size);
    state.sn_nlte_sigma_s_eff_work.reset(scratch_size);
    state.sn_nlte_eta_cdf_work.reset(scratch_size);
    state.sn_nlte_lambda_work.reset(scratch_size);
    auto& cache = nlte_cache();
    const double* cv_e_ptr =
        (state.cv_e.size() == static_cast<std::size_t>(n_cells)) ? state.cv_e.data()
                                                                 : nullptr;
    const auto result = compute_nlte_coefficients_cuda_pure_sn(
        state.rho.data(),
        state.Te.data(),
        state.zbar.data(),
        cv_e_ptr,
        state.cell_is_void.data(),
        state.cell_is_void.size(),
        cache.device.view(),
        planck.device_view(),
        n_cells,
        n_groups,
        dt,
        std::max(mat.A, 1.0e-12),
        1.0,
        0.0,
        1.0,
        mat.lambda_fd_delta_rel,
        mat.lambda_fd_abs_min,
        sn.opacity_cap,
        mat.cv_e_override,
        cfg.numerics.floors.Te,
        std::max(mat.ideal_gas_gamma - 1.0, 1.0e-12),
        false,
        false,
        false,
        state.sn_nlte_f_work.data(),
        state.sn_sigma_a.data(),
        state.sn_sigma_pe.data(),
        state.sn_dsa_rhs.data(),
        state.sn_nlte_sigma_eff_work.data(),
        state.sn_sigma_s.data(),
        state.sn_nlte_eta_cdf_work.data(),
        state.sn_eta.data(),
        state.sn_nlte_lambda_work.data(),
        0,
        cfg.materials.low_density_extrapolation);
    if (result.nan_inf_count != 0 || result.negative_eta_clamp_count != 0) {
      core::log_warning("SN2D NLTE coefficient clamp counts: nan_inf=" +
                        std::to_string(result.nan_inf_count) +
                        ", eta=" +
                        std::to_string(result.negative_eta_clamp_count));
    }
    return;
  }

  materials::OpacityEvalView opacity_view{};
  opacity_view.rho = state.rho.data();
  opacity_view.Te = state.Te.data();
  opacity_view.sigma_a = state.sn_sigma_a.data();
  opacity_view.sigma_R = state.sn_sigma_s.data();
  opacity_view.n_cells = n_cells;
  opacity_view.n_groups = n_groups;
  opacity_view.opacity_model =
      (mat.opacity_model == "freq_dep_marshak")
          ? materials::kOpacityModelFreqDepMarshak
          : ((mat.opacity_model == "power_law")
                 ? materials::kOpacityModelPowerLaw
                 : materials::kOpacityModelConstant);
  opacity_view.kappa_planck_const = std::max(0.0, mat.kappa_a_constant);
  opacity_view.kappa_rosseland_const = std::max(0.0, mat.kappa_a_constant);
  opacity_view.kappa_floor = sn.opacity_floor;
  opacity_view.kappa_cap = sn.opacity_cap;
  opacity_view.power_law_kappa0 = mat.opacity_power_law_kappa0_cm2_g;
  opacity_view.power_law_alpha_T = mat.opacity_power_law_alpha_T;
  opacity_view.power_law_lambda_rho = mat.opacity_power_law_lambda_rho;
  opacity_view.power_law_T_ref_eV = mat.opacity_power_law_T_ref_eV;
  opacity_view.power_law_rho_ref = mat.opacity_power_law_rho_ref_g_cc;
  const auto n_nonvoid_materials = std::count_if(
      cfg.materials.materials.begin(), cfg.materials.materials.end(),
      [](const auto& m) { return !m.is_void; });
  if (n_nonvoid_materials > 1) {
    state.ensure_cell_material_props(cfg);
    opacity_view.kappa_planck_cell = state.kappa_planck_eff.data();
    opacity_view.kappa_rosseland_cell = state.kappa_rosseland_eff.data();
  }
  std::vector<double> bounds;
  if (opacity_view.opacity_model == materials::kOpacityModelFreqDepMarshak) {
    bounds = radiation_group_bounds(cfg, n_groups);
    state.sn_group_bounds_work.reset(bounds.size());
    cuda_check(cudaMemcpy(state.sn_group_bounds_work.data(),
                          bounds.data(),
                          sizeof(double) * bounds.size(),
                          cudaMemcpyHostToDevice),
               "SN2D upload group bounds failed");
    opacity_view.group_bounds_eV = state.sn_group_bounds_work.data();
  }
  materials::evaluate_opacity_cuda(opacity_view, nullptr);
  const int grid = (total + kBlock - 1) / kBlock;
  if (grid > 0) {
    cuda_check(cudaMemcpy(state.sn_sigma_pe.data(),
                          state.sn_sigma_a.data(),
                          sizeof(double) * static_cast<std::size_t>(total),
                          cudaMemcpyDeviceToDevice),
               "SN2D copy sigma_a to sigma_pe failed");
    fill_kernel<<<grid, kBlock>>>(state.sn_sigma_s.data(), total, 0.0);
    cuda_check(cudaGetLastError(), "SN2D zero sigma_s launch failed");
  }
}

double copy_reduction_scalar(core::State& state) {
  double value = 0.0;
  cuda_check(cudaMemcpy(&value,
                        state.sn_reduction_work.data(),
                        sizeof(double),
                        cudaMemcpyDeviceToHost),
             "SN2D copy reduction scalar failed");
  return value;
}

struct SnApAlphaActivity {
  double alpha_max = 0.0;
  double active_faces = 0.0;
};

SnApAlphaActivity reduce_ap_alpha_activity(const double* const alpha_face,
                                           const int n_total) {
  SnApAlphaActivity out{};
  if (n_total <= 0) {
    return out;
  }
  const int blocks = (n_total + kBlock - 1) / kBlock;
  parallel::DeviceArray d_block_max{"sn2d_ap_alpha:block_max"};
  parallel::DeviceArray d_block_count{"sn2d_ap_alpha:block_count"};
  d_block_max.resize(sizeof(double) * static_cast<std::size_t>(blocks));
  d_block_count.resize(sizeof(unsigned long long) *
                       static_cast<std::size_t>(blocks));
  reduce_ap_alpha_activity_2d_kernel<<<blocks, kBlock>>>(
      alpha_face,
      d_block_max.as<double>(),
      d_block_count.as<unsigned long long>(),
      n_total);
  cuda_check(cudaGetLastError(), "SN2D AP alpha activity reduction launch failed");
  std::vector<double> host_max(static_cast<std::size_t>(blocks), 0.0);
  std::vector<unsigned long long> host_count(static_cast<std::size_t>(blocks),
                                            0ULL);
  cuda_check(cudaMemcpy(host_max.data(),
                        d_block_max.as<double>(),
                        sizeof(double) * static_cast<std::size_t>(blocks),
                        cudaMemcpyDeviceToHost),
             "SN2D copy AP alpha max blocks failed");
  cuda_check(cudaMemcpy(host_count.data(),
                        d_block_count.as<unsigned long long>(),
                        sizeof(unsigned long long) *
                            static_cast<std::size_t>(blocks),
                        cudaMemcpyDeviceToHost),
             "SN2D copy AP alpha active blocks failed");
  unsigned long long active = 0ULL;
  for (int b = 0; b < blocks; ++b) {
    out.alpha_max = std::max(out.alpha_max, host_max[static_cast<std::size_t>(b)]);
    active += host_count[static_cast<std::size_t>(b)];
  }
  out.active_faces = static_cast<double>(active);
  return out;
}

void zero_reduction(core::State& state) {
  cuda_check(cudaMemset(state.sn_reduction_work.data(), 0, sizeof(double)),
             "SN2D zero reduction failed");
}

double reduce_relative_delta(core::State& state,
                             const double* next,
                             const double* prev,
                             const int n_values) {
  zero_reduction(state);
  const int grid = (n_values + kBlock - 1) / kBlock;
  if (grid > 0) {
    max_relative_delta_kernel<<<grid, kBlock>>>(
        next, prev, state.sn_reduction_work.data(), n_values);
    cuda_check(cudaGetLastError(), "SN2D relative delta launch failed");
  }
  return copy_reduction_scalar(state);
}

double compute_escaped_energy(core::State& state,
                              const int nr,
                              const int nz,
                              const int n_groups,
                              const double dt,
                              const int r_outer_bc,
                              const int z_bottom_bc,
                              const int z_top_bc) {
  zero_reduction(state);
  // Owned-window partial (Option C): serial window is [0, nr*nz).
  const auto cw = state.owned_cell_window(nr * nz);
  const int total = cw.count() * n_groups;
  const int grid = (total + kBlock - 1) / kBlock;
  if (grid > 0) {
    escaped_energy_2d_kernel<<<grid, kBlock>>>(
        state.x_r.data(),
        state.x_z.data(),
        state.sn_face_flux_limited.data(),
        state.sn_reduction_work.data(),
        cw.begin,
        cw.end,
        nr,
        nz,
        n_groups,
        dt,
        r_outer_bc,
        z_bottom_bc,
        z_top_bc);
    cuda_check(cudaGetLastError(), "SN2D escaped energy launch failed");
  }
  return copy_reduction_scalar(state);
}

double compute_marshak_in_energy(core::State& state,
                                 const int nr,
                                 const int nz,
                                 const double dt,
                                 const int z_bottom_bc,
                                 const int z_top_bc,
                                 const double marshak_flux_erg_per_cm2_s) {
  if (!(marshak_flux_erg_per_cm2_s > 0.0) ||
      (z_bottom_bc != kSNBoundaryMarshak && z_top_bc != kSNBoundaryMarshak)) {
    return 0.0;
  }
  zero_reduction(state);
  // Owned radial range (Option C): the r-slab cell window is i-plane
  // aligned, so [cw.begin/nz, cw.end/nz) is the owned i range.
  const auto cw = state.owned_cell_window(nr * nz);
  TENRYU_ASSERT(nz > 0 && cw.begin % nz == 0 && cw.end % nz == 0,
                "SN2D marshak tally requires an i-plane-aligned cell window");
  const int i_begin = cw.begin / nz;
  const int i_end = cw.end / nz;
  const int grid = (i_end - i_begin + kBlock - 1) / kBlock;
  if (grid > 0) {
    marshak_in_energy_2d_kernel<<<grid, kBlock>>>(
        state.x_r.data(),
        state.x_z.data(),
        state.sn_reduction_work.data(),
        i_begin,
        i_end,
        nr,
        nz,
        dt,
        z_bottom_bc,
        z_top_bc,
        marshak_flux_erg_per_cm2_s);
    cuda_check(cudaGetLastError(), "SN2D Marshak source energy launch failed");
  }
  return copy_reduction_scalar(state);
}

void copy_rad_E_to_old(core::State& state, const int n_cells, const int n_groups) {
  const std::size_t bytes =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups) *
      sizeof(double);
  if (bytes > 0U) {
    cuda_check(cudaMemcpy(state.rad_E_old.data(),
                          state.rad_E.data(),
                          bytes,
                          cudaMemcpyDeviceToDevice),
               "SN2D update rad_E_old failed");
  }
}

void zero_axis_faces_checked(core::State& state,
                             double* face_field,
                             const int nz,
                             const int n_groups,
                             const char* label) {
  const int total = nz * n_groups;
  const int grid = (total + kBlock - 1) / kBlock;
  if (grid == 0 || face_field == nullptr) {
    return;
  }
#ifdef TENRYU_DEBUG_SN_AXIS_FACE
  cuda_check(cudaMemset(state.sn_reduction_work.data(), 0, sizeof(double)),
             "SN2D zero axis diagnostic flag failed");
  zero_axis_faces_checked_kernel<<<grid, kBlock>>>(
      face_field, state.sn_reduction_work.data(), nz, n_groups);
#else
  zero_axis_faces_checked_kernel<<<grid, kBlock>>>(
      face_field, nullptr, nz, n_groups);
#endif
  cuda_check(cudaGetLastError(), label);
#ifdef TENRYU_DEBUG_SN_AXIS_FACE
  double bad = 0.0;
  cuda_check(cudaMemcpy(&bad,
                        state.sn_reduction_work.data(),
                        sizeof(double),
                        cudaMemcpyDeviceToHost),
             "SN2D copy axis diagnostic flag failed");
  TENRYU_ASSERT(bad == 0.0, "SN2D observed non-zero or non-finite axis face before zeroing");
#endif
}

}  // namespace

void compute_sn_E_star_flux_2d_rz_gpu(
    const double* const rad_E_old,
    const double* const node_r,
    const double* const node_z,
    const double* const vol,
    const double* const face_flux,
    double* const E_star,
    double* const diag_E_star_flux,
    const int nr,
    const int nz,
    const int n_groups,
    const double dt,
    cudaStream_t stream) {
  TENRYU_ASSERT(rad_E_old != nullptr, "SN2D E_star_flux requires rad_E_old");
  TENRYU_ASSERT(node_r != nullptr, "SN2D E_star_flux requires node_r");
  TENRYU_ASSERT(node_z != nullptr, "SN2D E_star_flux requires node_z");
  TENRYU_ASSERT(vol != nullptr, "SN2D E_star_flux requires vol");
  TENRYU_ASSERT(face_flux != nullptr, "SN2D E_star_flux requires face_flux");
  TENRYU_ASSERT(E_star != nullptr, "SN2D E_star_flux requires output");
  TENRYU_ASSERT(nr >= 0 && nz >= 0, "SN2D E_star_flux requires non-negative mesh");
  TENRYU_ASSERT(n_groups >= 0, "SN2D E_star_flux requires n_groups >= 0");
  const int total = nr * nz * n_groups;
  const int grid = (total + kBlock - 1) / kBlock;
  if (grid == 0) {
    return;
  }
  compute_E_star_flux_2d_kernel<<<grid, kBlock, 0, stream>>>(
      rad_E_old, node_r, node_z, vol, face_flux, E_star, diag_E_star_flux,
      nr, nz, n_groups, dt);
  cuda_check(cudaGetLastError(), "SN2D E_star_flux launch failed");
}

void compute_sn_donor_theta_limited_face_flux_2d_rz_gpu(
    const double* const rad_E_old,
    const double* const node_r,
    const double* const node_z,
    const double* const vol,
    const double* const face_flux_raw,
    double* const theta_donor,
    double* const stream_theta,
    double* const face_flux_limited,
    const int nr,
    const int nz,
    const int n_groups,
    const double dt,
    cudaStream_t stream) {
  TENRYU_ASSERT(rad_E_old != nullptr, "SN2D donor theta requires rad_E_old");
  TENRYU_ASSERT(node_r != nullptr, "SN2D donor theta requires node_r");
  TENRYU_ASSERT(node_z != nullptr, "SN2D donor theta requires node_z");
  TENRYU_ASSERT(vol != nullptr, "SN2D donor theta requires vol");
  TENRYU_ASSERT(face_flux_raw != nullptr, "SN2D donor theta requires face flux");
  TENRYU_ASSERT(theta_donor != nullptr,
                "SN2D donor theta requires donor theta output");
  TENRYU_ASSERT(stream_theta != nullptr, "SN2D donor theta requires theta output");
  TENRYU_ASSERT(face_flux_limited != nullptr,
                "SN2D donor theta requires limited flux output");
  TENRYU_ASSERT(nr >= 0 && nz >= 0, "SN2D donor theta requires non-negative mesh");
  TENRYU_ASSERT(n_groups >= 0, "SN2D donor theta requires n_groups >= 0");
  const int n_cells = nr * nz;
  const int cell_total = n_cells * n_groups;
  const int face_total = n_faces(nr, nz) * n_groups;
  const int cell_grid = (cell_total + kBlock - 1) / kBlock;
  if (cell_grid > 0) {
    compute_streaming_theta_donor_2d_kernel<<<cell_grid, kBlock, 0, stream>>>(
        rad_E_old, node_r, node_z, vol, face_flux_raw, theta_donor,
        nr, nz, n_groups, dt);
    cuda_check(cudaGetLastError(), "SN2D donor theta pass-1 launch failed");
    compute_streaming_theta_2d_kernel<<<cell_grid, kBlock, 0, stream>>>(
        rad_E_old, node_r, node_z, vol, face_flux_raw, theta_donor, stream_theta,
        nr, nz, n_groups, dt);
    cuda_check(cudaGetLastError(), "SN2D donor theta pass-2 launch failed");
  }
  const int face_grid = (face_total + kBlock - 1) / kBlock;
  if (face_grid > 0) {
    apply_donor_theta_face_flux_2d_kernel<<<face_grid, kBlock, 0, stream>>>(
        face_flux_raw, stream_theta, face_flux_limited, nr, nz, n_groups);
    cuda_check(cudaGetLastError(), "SN2D donor theta face flux launch failed");
  }
}

void compute_sn_diffusion_face_flux_2d_rz_gpu(
    const double* const rad_E_old,
    const double* const sn_sigma_s,
    const double* const node_r,
    const double* const node_z,
    const double* const vol,
    double* const face_flux_diff,
    const int nr,
    const int nz,
    const int n_groups,
    const double opacity_floor,
    cudaStream_t stream) {
  TENRYU_ASSERT(rad_E_old != nullptr, "SN2D diffusion face flux requires rad_E_old");
  TENRYU_ASSERT(sn_sigma_s != nullptr, "SN2D diffusion face flux requires sigma_s");
  TENRYU_ASSERT(node_r != nullptr, "SN2D diffusion face flux requires node_r");
  TENRYU_ASSERT(node_z != nullptr, "SN2D diffusion face flux requires node_z");
  TENRYU_ASSERT(vol != nullptr, "SN2D diffusion face flux requires vol");
  TENRYU_ASSERT(face_flux_diff != nullptr,
                "SN2D diffusion face flux requires output");
  TENRYU_ASSERT(nr >= 0 && nz >= 0,
                "SN2D diffusion face flux requires non-negative mesh");
  TENRYU_ASSERT(n_groups >= 0, "SN2D diffusion face flux requires n_groups >= 0");
  const int total = n_faces(nr, nz) * n_groups;
  const int grid = (total + kBlock - 1) / kBlock;
  if (grid == 0) {
    return;
  }
  compute_diffusion_face_flux_2d_kernel<<<grid, kBlock, 0, stream>>>(
      rad_E_old, sn_sigma_s, node_r, node_z, vol, opacity_floor,
      face_flux_diff, nr, nz, n_groups);
  cuda_check(cudaGetLastError(), "SN2D diffusion face flux launch failed");
}

void compute_sn_ap_blended_face_flux_2d_rz_gpu(
    const double* const face_flux_sn,
    const double* const face_flux_diff,
    const double* const rad_E_old,
    const double* const Te,
    const double* const sn_sigma_s,
    const double* const node_r,
    const double* const node_z,
    const double* const vol,
    PlanckTableDeviceView planck,
    const double opacity_floor,
    const double tau_lo,
    const double tau_hi,
    const double eq_full,
    const double eq_off,
    const double f_full,
    const double f_off,
    double* const face_flux_blended,
    double* const alpha_face,
    const int nr,
    const int nz,
    const int n_groups,
    cudaStream_t stream) {
  TENRYU_ASSERT(face_flux_sn != nullptr, "SN2D AP blend requires SN face flux");
  TENRYU_ASSERT(face_flux_diff != nullptr, "SN2D AP blend requires diffusion flux");
  TENRYU_ASSERT(rad_E_old != nullptr, "SN2D AP blend requires rad_E_old");
  TENRYU_ASSERT(Te != nullptr, "SN2D AP blend requires Te");
  TENRYU_ASSERT(sn_sigma_s != nullptr, "SN2D AP blend requires sigma_s");
  TENRYU_ASSERT(node_r != nullptr, "SN2D AP blend requires node_r");
  TENRYU_ASSERT(node_z != nullptr, "SN2D AP blend requires node_z");
  TENRYU_ASSERT(vol != nullptr, "SN2D AP blend requires vol");
  TENRYU_ASSERT(face_flux_blended != nullptr, "SN2D AP blend requires output");
  TENRYU_ASSERT(alpha_face != nullptr, "SN2D AP blend requires alpha output");
  TENRYU_ASSERT(nr >= 0 && nz >= 0, "SN2D AP blend requires non-negative mesh");
  TENRYU_ASSERT(n_groups >= 0, "SN2D AP blend requires n_groups >= 0");
  const int total = n_faces(nr, nz) * n_groups;
  const int grid = (total + kBlock - 1) / kBlock;
  if (grid == 0) {
    return;
  }
  compute_ap_blended_face_flux_2d_kernel<<<grid, kBlock, 0, stream>>>(
      face_flux_sn, face_flux_diff, rad_E_old, Te, sn_sigma_s, node_r, node_z,
      vol, planck, opacity_floor, tau_lo, tau_hi, eq_full, eq_off, f_full,
      f_off, face_flux_blended, alpha_face, nr, nz, n_groups);
  cuda_check(cudaGetLastError(), "SN2D AP blend launch failed");
}

void compute_sn_ap_cell_diagnostics_2d_rz_gpu(
    const double* const face_flux_sn,
    const double* const rad_E_old,
    const double* const sn_sigma_s,
    const double* const node_r,
    const double* const node_z,
    const double* const vol,
    const double* const alpha_face,
    const double opacity_floor,
    double* const sn_tau_R,
    double* const sn_reduced_flux,
    double* const sn_ap_alpha,
    const int nr,
    const int nz,
    const int n_groups,
    cudaStream_t stream) {
  TENRYU_ASSERT(face_flux_sn != nullptr, "SN2D AP diagnostics require SN face flux");
  TENRYU_ASSERT(rad_E_old != nullptr, "SN2D AP diagnostics require rad_E_old");
  TENRYU_ASSERT(sn_sigma_s != nullptr, "SN2D AP diagnostics require sigma_s");
  TENRYU_ASSERT(node_r != nullptr, "SN2D AP diagnostics require node_r");
  TENRYU_ASSERT(node_z != nullptr, "SN2D AP diagnostics require node_z");
  TENRYU_ASSERT(vol != nullptr, "SN2D AP diagnostics require vol");
  TENRYU_ASSERT(alpha_face != nullptr, "SN2D AP diagnostics require alpha");
  TENRYU_ASSERT(sn_tau_R != nullptr, "SN2D AP diagnostics require tau output");
  TENRYU_ASSERT(sn_reduced_flux != nullptr,
                "SN2D AP diagnostics require reduced flux output");
  TENRYU_ASSERT(sn_ap_alpha != nullptr,
                "SN2D AP diagnostics require alpha output");
  const int n_cells = nr * nz;
  const int grid = (n_cells + kBlock - 1) / kBlock;
  if (grid == 0) {
    return;
  }
  compute_ap_cell_diagnostics_2d_kernel<<<grid, kBlock, 0, stream>>>(
      face_flux_sn, rad_E_old, sn_sigma_s, node_r, node_z, vol, alpha_face,
      fmax(opacity_floor, 0.0), sn_tau_R, sn_reduced_flux, sn_ap_alpha,
      nr, nz, n_groups);
  cuda_check(cudaGetLastError(), "SN2D AP diagnostics launch failed");
}

void advance_radiation_step_sn_2d_rz(
    core::State& state,
    const core::Config& cfg,
    const PlanckTable& planck,
    const core::Config::MaterialsConfig::MatDef& mat,
    const double dt,
    const parallel::PartitionInfo& part,
    parallel::CommBuffers* bufs) {
  TENRYU_ASSERT(cfg.radiation.mode == core::RadiationMode::SnTransport,
                "advance_radiation_step_sn_2d_rz requires sn_transport mode");
  TENRYU_ASSERT(state.mesh.dim == 2 || cfg.main.dimension == "2D_RZ",
                "advance_radiation_step_sn_2d_rz requires 2D_RZ state");
  TENRYU_ASSERT(dt > 0.0, "advance_radiation_step_sn_2d_rz requires dt > 0");
  const int nr = cfg.mesh.nr;
  const int nz = cfg.mesh.nz;
  const int n_cells = nr * nz;
  const int n_unique_faces = n_faces(nr, nz);
  const int n_groups = std::max(cfg.radiation.groups, 1);
  const auto& sn = cfg.radiation.sn_transport;
  const bool use_nlte =
      mat.opacity_model == "table_nlte" || mat.opacity_model == "tmat";
  TENRYU_ASSERT(static_cast<int>(state.rho.size()) == n_cells,
                "SN2D requires state cell count to match nr*nz");
  TENRYU_ASSERT(state.x_r.size() == static_cast<std::size_t>((nr + 1) * (nz + 1)),
                "SN2D requires 2D node_r size");
  TENRYU_ASSERT(state.x_z.size() == static_cast<std::size_t>((nr + 1) * (nz + 1)),
                "SN2D requires 2D node_z size");
  TENRYU_ASSERT(planck.n_groups() == n_groups,
                "SN2D requires Planck table group count to match Radiation.groups");

  ensure_state_buffers(state, n_cells, n_unique_faces, n_groups);
  initialize_rad_E_old_if_needed(state, n_cells, n_groups);

  const int cell_grid = (n_cells + kBlock - 1) / kBlock;
  const int total = n_cells * n_groups;
  const int total_grid = (total + kBlock - 1) / kBlock;
  parallel::DeviceArray d_sn_Te_picard_prev{"sn2d_advance:Te_picard_prev"};
  d_sn_Te_picard_prev.resize(sizeof(double) *
                             static_cast<std::size_t>(n_cells));
  parallel::DeviceArray d_radial_fixup_count{"sn2d_advance:radial_fixup_count"};
  parallel::DeviceArray d_angular_fixup_count{"sn2d_advance:angular_fixup_count"};
  parallel::DeviceArray d_radial_fixup_artificial_abs{"sn2d_advance:radial_fixup_art_abs"};
  parallel::DeviceArray d_angular_fixup_artificial_abs{"sn2d_advance:angular_fixup_art_abs"};
  d_radial_fixup_count.resize(sizeof(unsigned long long) *
                              static_cast<std::size_t>(total));
  d_angular_fixup_count.resize(sizeof(unsigned long long) *
                               static_cast<std::size_t>(total));
  d_radial_fixup_artificial_abs.resize(sizeof(double) *
                                       static_cast<std::size_t>(total));
  d_angular_fixup_artificial_abs.resize(sizeof(double) *
                                        static_cast<std::size_t>(total));
  cuda_check(cudaMemset(d_radial_fixup_count.ptr,
                        0,
                        d_radial_fixup_count.size),
             "SN2D zero radial fixup count failed");
  cuda_check(cudaMemset(d_angular_fixup_count.ptr,
                        0,
                        d_angular_fixup_count.size),
             "SN2D zero angular fixup count failed");
  cuda_check(cudaMemset(d_radial_fixup_artificial_abs.ptr,
                        0,
                        d_radial_fixup_artificial_abs.size),
             "SN2D zero radial fixup artificial_abs failed");
  cuda_check(cudaMemset(d_angular_fixup_artificial_abs.ptr,
                        0,
                        d_angular_fixup_artificial_abs.size),
             "SN2D zero angular fixup artificial_abs failed");
  if (cell_grid > 0) {
    copy_kernel<<<cell_grid, kBlock>>>(state.Te.data(), state.sn_Te_old.data(), n_cells);
    cuda_check(cudaGetLastError(), "SN2D Te snapshot launch failed");
  }
  if (cell_grid > 0) {
    copy_kernel<<<cell_grid, kBlock>>>(state.Te.data(),
                                       d_sn_Te_picard_prev.as<double>(),
                                       n_cells);
    cuda_check(cudaGetLastError(), "SN2D Te picard snapshot launch failed");
  }
  if (total_grid > 0) {
    initialize_phi_from_rad_E_kernel<<<total_grid, kBlock>>>(
        state.rad_E_old.data(), state.sn_phi_old.data(), total);
    cuda_check(cudaGetLastError(), "SN2D initialize phi launch failed");
  }

  state.sn_converged = false;
  state.sn_outer_stagnated = false;
  state.sn_outer_residual = std::numeric_limits<double>::infinity();
  state.sn_inner_residual = std::numeric_limits<double>::infinity();
  state.sn_outer_iterations = 0;
  state.sn_inner_iterations = 0;
  state.sn_escaped_step = 0.0;
  state.sn_marshak_in_step = 0.0;
  state.sn_ap_alpha_max = 0.0;
  state.sn_ap_alpha_active_faces = 0.0;

  const int z_bottom_bc = sn_boundary_code(sn.boundary.z_bottom);
  const int z_top_bc = sn_boundary_code(sn.boundary.z_top);
  const int r_outer_bc = sn_boundary_code(sn.boundary.outer_r);
  double marshak_flux = std::max(sn.marshak.flux_erg_per_cm2_s, 0.0);
  // Indirect-drive Tr(t) marshak z boundary (grey v1; spec docs/design/
  // 2d_tr_drive_port_spec.md §4): resolved once at solve entry with the
  // IMC-2D precedence and fed through the existing scalar slot — the sweep
  // kernels are unchanged. Builder validation guarantees
  // flux_erg_per_cm2_s == 0 on this route, n_groups == 1, and (for dual-z
  // marshak) no per-face tables, so one scalar is exact.
  {
    double t_r_z = 0.0;
    if (z_bottom_bc == kSNBoundaryMarshak) {
      t_r_z = resolve_marshak_tr_face_eV(cfg, state, "bottom_z", "z_bottom");
    }
    if (!(t_r_z > 0.0) && z_top_bc == kSNBoundaryMarshak) {
      t_r_z = resolve_marshak_tr_face_eV(cfg, state, "top_z", "z_top");
    }
    if (t_r_z > 0.0) {
      const double t4 = t_r_z * t_r_z * t_r_z * t_r_z;
      marshak_flux =
          0.25 * core::constants::c_light * core::constants::a_eV * t4;
    }
  }

  const int max_outer = std::max(sn.max_outer_iterations, 1);
  constexpr double kPlateauTol = 1.0e-6;
  constexpr int kPlateauK = 3;
  double prev_inner_residual = std::numeric_limits<double>::infinity();
  int plateau_count = 0;
  for (int outer = 0; outer < max_outer; ++outer) {
    evaluate_opacity(state, cfg, planck, mat, n_cells, n_groups, dt);
    SNTransportGPUConfig gpu_cfg{};
    // I3 quadrature-ladder verification requires a true S_4 rung
    // (polar-order n_angles=4; docs/design/i3_sn_radshock_spec.md §4.4,
    // main ruling D1 2026-07-04). Clamp change is a bitwise no-op for every
    // existing deck: none requests n_angles < 8.
    gpu_cfg.n_angles = std::max(sn.n_angles, 4);
    gpu_cfg.max_iterations = std::max(sn.max_inner_iterations, 1);
    gpu_cfg.convergence_tol = sn.inner_tol;
    gpu_cfg.temperature_floor_eV = cfg.numerics.floors.Te;
    gpu_cfg.dsa_enabled = sn.dsa_enabled;
    gpu_cfg.z_boundary_reflect =
        z_bottom_bc == kSNBoundaryReflect && z_top_bc == kSNBoundaryReflect;
    gpu_cfg.z_bottom_boundary = z_bottom_bc;
    gpu_cfg.z_top_boundary = z_top_bc;
    gpu_cfg.r_outer_boundary = r_outer_bc;
    gpu_cfg.marshak_flux_erg_per_cm2_s = marshak_flux;
    gpu_cfg.block_threads_2d = 128;
    gpu_cfg.linear_characteristic = sn.spatial_scheme == "linear_characteristic";

    SNMaterialCouplingGPUInputs in{};
    in.sigma_a = state.sn_sigma_a.data();
    in.sigma_s = state.sn_sigma_s.data();
    in.source_emission = use_nlte ? state.sn_eta.data() : nullptr;
    in.Te = state.Te.data();
    in.ee = state.ee.data();
    in.node_r = state.x_r.data();
    in.node_z = state.x_z.data();
    in.vol = state.vol.data();
    in.rho = state.rho.data();
    in.cv_e =
        (state.cv_e.size() == static_cast<std::size_t>(n_cells)) ? state.cv_e.data()
                                                                 : nullptr;
    in.Te_old = state.sn_Te_old.data();
    in.planck = planck.device_view();
    in.E_out = state.rad_E.data();
    in.P_rr_out = state.sn_Prr.data();
    in.chi_out = state.sn_chi.data();
    in.F_z_out = state.sn_F_z.data();
    in.P_zz_out = state.sn_P_zz.data();
    in.chi_z_out = state.sn_chi_z.data();
    in.phi_sweep_out = state.sn_phi_sweep.data();
    in.face_flux_raw = state.sn_face_flux_raw.data();
    in.radial_fixup_count = d_radial_fixup_count.as<unsigned long long>();
    in.radial_fixup_artificial_abs = d_radial_fixup_artificial_abs.as<double>();
    in.angular_fixup_count = d_angular_fixup_count.as<unsigned long long>();
    in.angular_fixup_artificial_abs =
        d_angular_fixup_artificial_abs.as<double>();
    in.rad_dep = state.rad_dep.data();
    in.rad_emit = state.rad_emit.data();
    in.dim = 2;
    in.nr = nr;
    in.nz = nz;
    in.n_groups = n_groups;
    in.dt = dt;
    in.update_material = false;
    in.cv_e_const = mat.cv_e_override;
    in.mpi_part = &part;
    in.mpi_bufs = bufs;
    const auto sn_mpi_cw = state.owned_cell_window(n_cells);
    in.mpi_c_begin = sn_mpi_cw.begin;
    in.mpi_c_end = sn_mpi_cw.end;
    const SNTransportGPUResult result = solve_sn_material_coupling_gpu(in, gpu_cfg);
    state.sn_inner_iterations = result.iterations;
    state.sn_inner_residual = result.convergence_error;
    if (part.n_ranks > 1) {
      // Ghost strips for the moment-layer stencils: the solve publishes
      // phi_sweep FULL-range from D_avg, which is ZERO on non-swept
      // ghost cells; donor-theta/E_star/AP read it across the interface
      // every outer iteration. Root cause of the coupled-deck P4 1-step
      // ~4e-6 edge error — the hot-disk edge sits ON the P4 interface
      // (design doc §6i). Outer iteration count is rank-uniform, so this
      // collective is deadlock-safe.
      TENRYU_ASSERT(bufs != nullptr,
                    "SN2D advance requires CommBuffers under MPI");
      parallel::exchange_cell_strips_scaled(part, *bufs,
                                            state.sn_phi_sweep.data(),
                                            n_groups, 10);
      // Near-interface face planes of the RAW flux: the donor-theta
      // two-pass credit of a GHOST cell needs that cell's FULL face set.
      // The interface r-plane is sum-completed inside the solve, but the
      // neighbor-interior r-plane and the ghost row's z-faces exist only
      // on their owner (zero here), which starved the theta limiter at
      // the interface (coupled-deck P4 93% disk-corner deficit; design
      // doc §6i). Receiver slots are zero, so the adds act as
      // owner->ghost copies.
      {
        const auto fw_cw = state.owned_cell_window(n_cells);
        const int fi_begin = fw_cw.begin / nz;
        const int fi_end = fw_cw.end / nz;
        TENRYU_ASSERT(fi_end - fi_begin >= 2,
                      "SN2D KBA face transfers require >= 2 owned planes");
        const int fr_nrf = (nr + 1) * nz;
        const int fr_plane = nz * n_groups;
        const int fz_plane = (nz + 1) * n_groups;
        const auto rplane = [&](const int p) {
          return static_cast<std::size_t>(std::max(p, 0)) *
                 static_cast<std::size_t>(fr_plane);
        };
        const auto zrow = [&](const int i) {
          return (static_cast<std::size_t>(fr_nrf) +
                  static_cast<std::size_t>(std::max(i, 0)) *
                      static_cast<std::size_t>(nz + 1)) *
                 static_cast<std::size_t>(n_groups);
        };
        parallel::sendrecv_add_planes_asym(
            part, *bufs, state.sn_face_flux_raw.data(), fr_plane,
            rplane(fi_begin + 1), rplane(fi_begin - 1), rplane(fi_end - 1),
            rplane(fi_end + 1), 13);
        parallel::sendrecv_add_planes_asym(
            part, *bufs, state.sn_face_flux_raw.data(), fz_plane,
            zrow(fi_begin), zrow(fi_begin - 1), zrow(fi_end - 1),
            zrow(fi_end), 14);
      }
    }
    zero_axis_faces_checked(state,
                            state.sn_face_flux_raw.data(),
                            nz,
                            n_groups,
                            "SN2D zero raw axis face flux launch failed");
    compute_sn_diffusion_face_flux_2d_rz_gpu(state.rad_E_old.data(),
                                             state.sn_sigma_s.data(),
                                             state.x_r.data(),
                                             state.x_z.data(),
                                             state.vol.data(),
                                             state.sn_face_flux_diff.data(),
                                             nr,
                                             nz,
                                             n_groups,
                                             sn.opacity_floor);
    zero_axis_faces_checked(state,
                            state.sn_face_flux_diff.data(),
                            nz,
                            n_groups,
                            "SN2D zero diffusion axis face flux launch failed");
    compute_sn_ap_blended_face_flux_2d_rz_gpu(
        state.sn_face_flux_raw.data(),
        state.sn_face_flux_diff.data(),
        state.rad_E_old.data(),
        state.Te.data(),
        state.sn_sigma_s.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.vol.data(),
        planck.device_view(),
        sn.opacity_floor,
        10.0,
        20.0,
        0.05,
        0.10,
        0.15,
        0.25,
        state.sn_face_flux_blended.data(),
        state.sn_face_alpha.data(),
        nr,
        nz,
        n_groups);
    zero_axis_faces_checked(state,
                            state.sn_face_flux_blended.data(),
                            nz,
                            n_groups,
                            "SN2D zero blended axis face flux launch failed");
    zero_axis_faces_checked(state,
                            state.sn_face_alpha.data(),
                            nz,
                            n_groups,
                            "SN2D zero axis face alpha launch failed");
    {
      const SnApAlphaActivity alpha_activity =
          reduce_ap_alpha_activity(state.sn_face_alpha.data(),
                                   n_unique_faces * n_groups);
      state.sn_ap_alpha_max = alpha_activity.alpha_max;
      state.sn_ap_alpha_active_faces = alpha_activity.active_faces;
    }
    compute_sn_ap_cell_diagnostics_2d_rz_gpu(
        state.sn_face_flux_raw.data(),
        state.rad_E_old.data(),
        state.sn_sigma_s.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.vol.data(),
        state.sn_face_alpha.data(),
        sn.opacity_floor,
        state.sn_tau_R.data(),
        state.sn_reduced_flux.data(),
        state.sn_ap_alpha.data(),
        nr,
        nz,
        n_groups,
        nullptr);
    compute_sn_donor_theta_limited_face_flux_2d_rz_gpu(
        state.rad_E_old.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.vol.data(),
        state.sn_face_flux_blended.data(),
        state.sn_theta_donor.data(),
        state.sn_stream_theta.data(),
        state.sn_face_flux_limited.data(),
        nr,
        nz,
        n_groups,
        dt);
    zero_axis_faces_checked(state,
                            state.sn_face_flux_limited.data(),
                            nz,
                            n_groups,
                            "SN2D zero limited axis face flux launch failed");
    compute_sn_E_star_flux_2d_rz_gpu(state.rad_E_old.data(),
                                     state.x_r.data(),
                                     state.x_z.data(),
                                     state.vol.data(),
                                     state.sn_face_flux_limited.data(),
                                     state.sn_E_star_flux.data(),
                                     state.sn_diag_E_star_flux.data(),
                                     nr,
                                     nz,
                                     n_groups,
                                     dt);

    SnMaterialNewton1DInputs newton{};
    newton.sigma_a = state.sn_sigma_a.data();
    newton.sigma_pe = state.sn_sigma_pe.data();
    newton.rad_E = state.rad_E.data();
    newton.rho = state.rho.data();
    newton.cv_e =
        (state.cv_e.size() == static_cast<std::size_t>(n_cells)) ? state.cv_e.data()
                                                                 : nullptr;
    newton.vol = state.vol.data();
    newton.zbar = state.zbar.data();
    newton.Te_old = state.sn_Te_old.data();
    newton.Te = state.Te.data();
    newton.ee = state.ee.data();
    newton.Pe = state.Pe.data();
    newton.rad_dep = state.rad_dep.data();
    newton.rad_emit = state.rad_emit.data();
    newton.diag_rad_emission_at_Tnp1 =
        state.sn_diag_rad_emission_at_Tnp1.data();
    newton.diag_rad_absorption = state.sn_diag_rad_absorption.data();
    newton.diag_clip_energy = state.sn_diag_clip_energy.data();
    newton.diag_clip_full_deficit = state.sn_diag_clip_full_deficit.data();
    newton.delta_T_rel = state.sn_delta_T.data();
    newton.planck = planck.device_view();
    if (mat.hydro_eos_backend != "exact_ideal_gas") {
      newton.electron_eos = sn_electron_eos_device_view(mat.eos_tables.get());
    }
    // Shared per-cell effective properties (multi-material fix; radiation can run
    // with hydro disabled, so ensure here).
    state.ensure_cell_material_props(cfg);
    newton.n_cells = n_cells;
    newton.n_groups = n_groups;
    newton.dt = dt;
    newton.Cv_e_const = mat.cv_e_override;
    newton.cv_e_const = 0.0;
    newton.A_eff = state.A_eff.data();
    newton.gamma_eff = state.gamma_eff.data();
    newton.temperature_floor_eV = cfg.numerics.floors.Te;
    newton.rad_E_out = state.rad_E.data();
    newton.E_star_override = state.sn_E_star_flux.data();
    newton.eos_low_density_extrap = cfg.materials.low_density_extrapolation;
    // Owned-window Newton (Option C): stale-finite far cells must not
    // raise spurious global-retry flags. Serial: [0, n_cells) identity.
    {
      const auto newton_cw = state.owned_cell_window(n_cells);
      newton.c_begin = newton_cw.begin;
      newton.c_end = newton_cw.end;
    }
    const std::size_t rad_E_bytes = sizeof(double) *
                                    static_cast<std::size_t>(n_cells) *
                                    static_cast<std::size_t>(n_groups);
    cuda_check(cudaMemcpy(state.sn_rad_E_pre_newton.data(),
                          state.rad_E.data(),
                          rad_E_bytes,
                          cudaMemcpyDeviceToDevice),
               "snapshot rad_E to sn_rad_E_pre_newton failed");
    // rad_E aliases rad_E_out; per-cell launch geometry guarantees no cross-thread RAW hazard within a step.
    state.sn_material_retry_flag |=
        solve_sn_material_temperature_newton_gpu(newton);
    if (part.n_ranks > 1 && bufs != nullptr) {
      // The Newton just updated OWNED Te; the next outer iteration
      // rebuilds sigma/emission from Te at ALL cells and the moment
      // -layer diffusion-face closure stencils sigma across the
      // interface, so ghost Te must be re-freshed every iteration (the
      // stage-entry halo only covers the pre-Newton state). This is the
      // per-iteration seed of the coupled-deck interface error (P2 cold
      // interface ~1e-8, P4 hot-edge ~4e-6; design doc §6i). Iteration
      // count is rank-uniform — deadlock-safe collective.
      double* te_ptr[] = {state.Te.data()};
      parallel::exchange_cell_fields(part, *bufs, te_ptr, 1,
                                     state.mesh.topo.n_cells, nullptr, 12);
    }

    // Rank-uniform outer decision (Option C): reduce the Picard Te delta
    // over the OWNED contiguous span (far cells are stale-finite and must
    // not steer the loop) and Allreduce(MAX) so every rank runs the same
    // outer iteration count — a rank-local residual under evolving matter
    // splits the break/plateau decisions and deadlocks the KBA/DSA
    // collectives (root cause of the coupled-deck P>1 hang, spec §6i).
    if (part.n_ranks > 1) {
      const auto res_cw = state.owned_cell_window(n_cells);
      state.sn_outer_residual =
          parallel::Reduction(part.n_ranks)
              .allreduce_max(reduce_relative_delta(
                  state, state.Te.data() + res_cw.begin,
                  d_sn_Te_picard_prev.as<double>() + res_cw.begin,
                  res_cw.count()));
    } else {
      state.sn_outer_residual =
          reduce_relative_delta(state,
                                state.Te.data(),
                                d_sn_Te_picard_prev.as<double>(),
                                n_cells);
    }
    state.sn_outer_iterations = outer + 1;
    if (cell_grid > 0) {
      copy_kernel<<<cell_grid, kBlock>>>(state.Te.data(),
                                         d_sn_Te_picard_prev.as<double>(),
                                         n_cells);
      cuda_check(cudaGetLastError(), "SN2D update Te picard snapshot launch failed");
    }
    if (state.sn_outer_residual <= sn.outer_tol && result.converged) {
      state.sn_converged = true;
      break;
    }
    if (state.sn_outer_residual <= sn.outer_tol && !result.converged) {
      const double curr = state.sn_inner_residual;
      const double denom = std::fmax(std::fmax(std::fabs(curr),
                                               std::fabs(prev_inner_residual)),
                                     1.0e-300);
      const double rel = std::fabs(curr - prev_inner_residual) / denom;
      if (rel <= kPlateauTol) {
        ++plateau_count;
        if (plateau_count >= kPlateauK) {
          state.sn_outer_stagnated = true;
          break;
        }
      } else {
        plateau_count = 0;
      }
      prev_inner_residual = curr;
    }
  }
  if (total_grid > 0) {
    anchor_void_rad_E_to_moments_2d_kernel<<<total_grid, kBlock>>>(
        state.rad_E.data(), state.sn_phi_sweep.data(),
        state.sn_sigma_a.data(), state.sn_sigma_s.data(), total, dt);
    cuda_check(cudaGetLastError(), "SN2D void rad_E anchor launch failed");
  }
  if (total_grid > 0) {
    publish_sn_fixup_tallies_2d_kernel<<<total_grid, kBlock>>>(
        d_radial_fixup_count.as<unsigned long long>(),
        d_radial_fixup_artificial_abs.as<double>(),
        state.sn_radial_fixup_count.data(),
        state.sn_radial_fixup_artificial_abs.data(),
        total);
    cuda_check(cudaGetLastError(), "SN2D publish radial fixup tallies failed");
    publish_sn_fixup_tallies_2d_kernel<<<total_grid, kBlock>>>(
        d_angular_fixup_count.as<unsigned long long>(),
        d_angular_fixup_artificial_abs.as<double>(),
        state.sn_angular_fixup_count.data(),
        state.sn_angular_fixup_artificial_abs.data(),
        total);
    cuda_check(cudaGetLastError(), "SN2D publish angular fixup tallies failed");
  }
  if (sn.timing_enabled) {
    std::printf("[sn_2d_rz_timing] outer_iters=%d inner_iters=%d dsa_enabled=%d "
                "converged=%d outer_residual=%.6e inner_residual=%.6e\n",
                state.sn_outer_iterations,
                state.sn_inner_iterations,
                sn.dsa_enabled ? 1 : 0,
                state.sn_converged ? 1 : 0,
                state.sn_outer_residual,
                state.sn_inner_residual);
  }

  if (const char* dbg = std::getenv("TENRYU_SN2D_DEBUG_DUMP")) {
    // Diagnostic-only (M18c s3c localization): raw binary dumps of the
    // solve/moment intermediates for cross-rank comparison.
    const auto dump_arr = [&](const char* tag, const double* dptr,
                              const std::size_t count) {
      std::vector<double> h(count);
      cuda_check(cudaMemcpy(h.data(), dptr, count * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "SN2D debug dump D2H failed");
      const std::string path = std::string(dbg) + "_" + tag + "_r" +
                               std::to_string(part.rank) + ".bin";
      FILE* fp = std::fopen(path.c_str(), "wb");
      if (fp != nullptr) {
        std::fwrite(h.data(), sizeof(double), count, fp);
        std::fclose(fp);
      }
    };
    const std::size_t ct = static_cast<std::size_t>(n_cells) *
                           static_cast<std::size_t>(n_groups);
    const std::size_t ft = static_cast<std::size_t>(n_faces(nr, nz)) *
                           static_cast<std::size_t>(n_groups);
    dump_arr("phisweep", state.sn_phi_sweep.data(), ct);
    dump_arr("fraw", state.sn_face_flux_raw.data(), ft);
    dump_arr("fblend", state.sn_face_flux_blended.data(), ft);
    dump_arr("estar", state.sn_E_star_flux.data(), ct);
    dump_arr("radE", state.rad_E.data(), ct);
  }
  state.sn_escaped_step = compute_escaped_energy(
      state, nr, nz, n_groups, dt, r_outer_bc, z_bottom_bc, z_top_bc);
  state.sn_marshak_in_step = compute_marshak_in_energy(
      state, nr, nz, dt, z_bottom_bc, z_top_bc, marshak_flux);
  // Outer-boundary escape-flux follow-up (2026-07-20): the driver energy budget pairs +marshak_in
  // (source) with +escape (sink), which requires GROSS conventions on both.
  // The face ledger is NET at a Marshak boundary (outgoing - inflow),
  // so add the inflow back: escape reports the gross outgoing energy and the
  // budget identity closes exactly (the same scalar cancels on both sides).
  // Vacuum configs add +0.0 (bit-identical).
  state.sn_escaped_step += state.sn_marshak_in_step;
  if (part.n_ranks > 1 && bufs != nullptr) {
    // rad_E ghost strips: the moment layer integrates rad_E full-range
    // with garbage face inputs on ghost cells; refresh from the owners so
    // next step's rad_E_old stencils (diffusion-face/donor-theta/E_star)
    // and cross-module readers see owner-true ghosts. The copy below then
    // inherits fresh ghosts into rad_E_old.
    parallel::exchange_cell_strips_scaled(part, *bufs, state.rad_E.data(),
                                          n_groups, 11);
  }
  copy_rad_E_to_old(state, n_cells, n_groups);
  state.holo_ale_invalidated = false;
}

}  // namespace tenryu::radiation
