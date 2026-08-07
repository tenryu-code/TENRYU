#include "diagnostics/radial_fourier_audit.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/error.hpp"

namespace tenryu::diagnostics {
namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kNormalizationFloor = 1.0e-300;

void cuda_check(const cudaError_t err, const char* context) {
  TENRYU_ASSERT(err == cudaSuccess,
                std::string(context) + ": " + cudaGetErrorString(err));
}

int reduction_thread_count(const int nr) {
  int threads = 1;
  while (threads < nr && threads < 1024) {
    threads <<= 1;
  }
  return threads;
}

__device__ double finite_or_zero(const double x) {
  return isfinite(x) ? x : 0.0;
}

__global__ void radial_fourier_cell_velocity_kernel(
    const double* v_r_node,
    const double* v_z_node,
    const int nr,
    const int nz,
    double* u_r_cell,
    double* u_z_cell) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int n00 = i * (nz + 1) + j;
  const int n10 = (i + 1) * (nz + 1) + j;
  const int n11 = (i + 1) * (nz + 1) + j + 1;
  const int n01 = i * (nz + 1) + j + 1;
  u_r_cell[c] = 0.25 * (finite_or_zero(v_r_node[n00]) +
                        finite_or_zero(v_r_node[n10]) +
                        finite_or_zero(v_r_node[n11]) +
                        finite_or_zero(v_r_node[n01]));
  u_z_cell[c] = 0.25 * (finite_or_zero(v_z_node[n00]) +
                        finite_or_zero(v_z_node[n10]) +
                        finite_or_zero(v_z_node[n11]) +
                        finite_or_zero(v_z_node[n01]));
}

__device__ double radial_fourier_field_value(
    const int field,
    const int c,
    const double* rho,
    const double* Te,
    const double* Ti,
    const double* u_r,
    const double* u_z,
    const double* E_rad,
    const int n_groups) {
  switch (field) {
    case 0:
      return finite_or_zero(rho[c]);
    case 1:
      return Te != nullptr ? finite_or_zero(Te[c]) : 0.0;
    case 2:
      return Ti != nullptr ? finite_or_zero(Ti[c]) : 0.0;
    case 3:
      return u_r != nullptr ? finite_or_zero(u_r[c]) : 0.0;
    case 4:
      return u_z != nullptr ? finite_or_zero(u_z[c]) : 0.0;
    case 5: {
      if (E_rad == nullptr || n_groups <= 0) {
        return 0.0;
      }
      double sum = 0.0;
      for (int g = 0; g < n_groups; ++g) {
        sum += finite_or_zero(E_rad[c * n_groups + g]);
      }
      return sum;
    }
    default:
      return 0.0;
  }
}

#if TENRYU_RFA_V2_KERNEL_LAUNCHES
__device__ double radial_fourier_segment_area_rz(
    const double r0,
    const double z0,
    const double r1,
    const double z1) {
  const double r_mid = fmax(0.0, 0.5 * (r0 + r1));
  const double dr = r1 - r0;
  const double dz = z1 - z0;
  return kTwoPi * r_mid * sqrt(dr * dr + dz * dz);
}

__device__ double radial_fourier_complex_field_value(
    const int field,
    const int c,
    const int i,
    const int j,
    const int nz,
    const int n_groups,
    const double* rho,
    const double* mass,
    const double* vol,
    const double* Te,
    const double* Ti,
    const double* ee,
    const double* ei,
    const double* u_r,
    const double* u_z,
    const double* E_rad,
    const double* Qvisc,
    const double* fld_fleck,
    const double* x_r,
    const double* x_z) {
  const double rho_c = rho != nullptr ? finite_or_zero(rho[c]) : 0.0;
  const double vol_c = vol != nullptr ? finite_or_zero(vol[c]) : 0.0;
  switch (static_cast<RadialFourierComplexFieldId>(field)) {
    case RadialFourierComplexFieldId::Rho:
      return rho_c;
    case RadialFourierComplexFieldId::M:
      return mass != nullptr ? finite_or_zero(mass[c]) : rho_c * vol_c;
    case RadialFourierComplexFieldId::V:
      return vol_c;
    case RadialFourierComplexFieldId::MOverV: {
      const double m_c =
          mass != nullptr ? finite_or_zero(mass[c]) : rho_c * vol_c;
      return fabs(vol_c) > kNormalizationFloor ? m_c / vol_c : 0.0;
    }
    case RadialFourierComplexFieldId::Pr:
      return rho_c * (u_r != nullptr ? finite_or_zero(u_r[c]) : 0.0);
    case RadialFourierComplexFieldId::Pz:
      return rho_c * (u_z != nullptr ? finite_or_zero(u_z[c]) : 0.0);
    case RadialFourierComplexFieldId::Ur:
      return u_r != nullptr ? finite_or_zero(u_r[c]) : 0.0;
    case RadialFourierComplexFieldId::Uz:
      return u_z != nullptr ? finite_or_zero(u_z[c]) : 0.0;
    case RadialFourierComplexFieldId::Ee:
      return rho_c * (ee != nullptr ? finite_or_zero(ee[c]) : 0.0);
    case RadialFourierComplexFieldId::Ei:
      return rho_c * (ei != nullptr ? finite_or_zero(ei[c]) : 0.0);
    case RadialFourierComplexFieldId::Erad: {
      if (E_rad == nullptr || n_groups <= 0) {
        return 0.0;
      }
      double sum = 0.0;
      for (int g = 0; g < n_groups; ++g) {
        sum += finite_or_zero(E_rad[c * n_groups + g]);
      }
      return sum;
    }
    case RadialFourierComplexFieldId::Te:
      return Te != nullptr ? finite_or_zero(Te[c]) : 0.0;
    case RadialFourierComplexFieldId::Ti:
      return Ti != nullptr ? finite_or_zero(Ti[c]) : 0.0;
    case RadialFourierComplexFieldId::Xr:
    case RadialFourierComplexFieldId::Xz:
    case RadialFourierComplexFieldId::Ar:
    case RadialFourierComplexFieldId::Az: {
      if (x_r == nullptr || x_z == nullptr) {
        return 0.0;
      }
      const int stride = nz + 1;
      const int n00 = i * stride + j;
      const int n10 = (i + 1) * stride + j;
      const int n11 = (i + 1) * stride + (j + 1);
      const int n01 = i * stride + (j + 1);
      if (field == static_cast<int>(RadialFourierComplexFieldId::Xr)) {
        return 0.25 * (finite_or_zero(x_r[n00]) + finite_or_zero(x_r[n10]) +
                       finite_or_zero(x_r[n11]) + finite_or_zero(x_r[n01]));
      }
      if (field == static_cast<int>(RadialFourierComplexFieldId::Xz)) {
        return 0.25 * (finite_or_zero(x_z[n00]) + finite_or_zero(x_z[n10]) +
                       finite_or_zero(x_z[n11]) + finite_or_zero(x_z[n01]));
      }
      const double r00 = finite_or_zero(x_r[n00]);
      const double r10 = finite_or_zero(x_r[n10]);
      const double r11 = finite_or_zero(x_r[n11]);
      const double r01 = finite_or_zero(x_r[n01]);
      const double z00 = finite_or_zero(x_z[n00]);
      const double z10 = finite_or_zero(x_z[n10]);
      const double z11 = finite_or_zero(x_z[n11]);
      const double z01 = finite_or_zero(x_z[n01]);
      const double left = radial_fourier_segment_area_rz(r00, z00, r01, z01);
      const double right = radial_fourier_segment_area_rz(r10, z10, r11, z11);
      const double bottom = radial_fourier_segment_area_rz(r00, z00, r10, z10);
      const double top = radial_fourier_segment_area_rz(r01, z01, r11, z11);
      return field == static_cast<int>(RadialFourierComplexFieldId::Ar)
                 ? 0.5 * (left + right)
                 : 0.5 * (bottom + top);
    }
    case RadialFourierComplexFieldId::Qvisc:
      return Qvisc != nullptr ? finite_or_zero(Qvisc[c]) : 0.0;
    case RadialFourierComplexFieldId::FFleck:
      return fld_fleck != nullptr ? finite_or_zero(fld_fleck[c]) : 0.0;
    case RadialFourierComplexFieldId::DVSwpt:
    case RadialFourierComplexFieldId::LambdaFld:
    case RadialFourierComplexFieldId::RFld:
    case RadialFourierComplexFieldId::KappaEff:
    case RadialFourierComplexFieldId::NewtonIters:
    case RadialFourierComplexFieldId::NewtonResidual:
    case RadialFourierComplexFieldId::NotAvailable:
      return 0.0;
  }
  return 0.0;
}
#endif

__global__ void radial_fourier_audit_kernel(
    const double* rho,
    const double* Te,
    const double* Ti,
    const double* u_r,
    const double* u_z,
    const double* E_rad,
    const int nr,
    const int nz,
    const int n_groups,
    double* A_mode_amp,
    double* mean_j) {
  const int m = blockIdx.x;
  const int j = blockIdx.y;
  const int field = blockIdx.z;
  const int tid = threadIdx.x;
  extern __shared__ double shared[];
  double* sum_shared = shared;
  double* imag_shared = shared + blockDim.x;

  double local_sum = 0.0;
  for (int i = tid; i < nr; i += blockDim.x) {
    const int c = i * nz + j;
    local_sum += radial_fourier_field_value(
        field, c, rho, Te, Ti, u_r, u_z, E_rad, n_groups);
  }
  sum_shared[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sum_shared[tid] += sum_shared[tid + stride];
    }
    __syncthreads();
  }

  const double mean = sum_shared[0] / static_cast<double>(nr);
  if (tid == 0 && m == 0) {
    mean_j[field * nz + j] = mean;
  }
  __syncthreads();

  double local_real = 0.0;
  double local_imag = 0.0;
  for (int i = tid; i < nr; i += blockDim.x) {
    const int c = i * nz + j;
    const double centered = radial_fourier_field_value(
        field, c, rho, Te, Ti, u_r, u_z, E_rad, n_groups) - mean;
    const double theta =
        kTwoPi * static_cast<double>(m * i) / static_cast<double>(nr);
    local_real += centered * cos(theta);
    local_imag += centered * sin(theta);
  }

  sum_shared[tid] = local_real;
  imag_shared[tid] = local_imag;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sum_shared[tid] += sum_shared[tid + stride];
      imag_shared[tid] += imag_shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const bool nyquist = (nr % 2 == 0) && (m == nr / 2);
    const double real_amplitude_scale = (m == 0 || nyquist) ? 1.0 : 2.0;
    const double magnitude = real_amplitude_scale *
                             sqrt(sum_shared[0] * sum_shared[0] +
                                  imag_shared[0] * imag_shared[0]);
    const double denom =
        static_cast<double>(nr) * fabs(mean) + kNormalizationFloor;
    A_mode_amp[(field * gridDim.x + m) * nz + j] = magnitude / denom;
  }
}

#if TENRYU_RFA_V2_KERNEL_LAUNCHES
__global__ void radial_fourier_fixed_mode_kernel(
    const std::uint8_t* field_ids,
    const int* m_targets,
    const int* j_targets,
    const int n_fields,
    const int n_m,
    const int n_j,
    const double* rho,
    const double* mass,
    const double* vol,
    const double* Te,
    const double* Ti,
    const double* ee,
    const double* ei,
    const double* u_r,
    const double* u_z,
    const double* E_rad,
    const double* Qvisc,
    const double* fld_fleck,
    const double* x_r,
    const double* x_z,
    const int nr,
    const int nz,
    const int n_groups,
    RadialFourierComplexCoeff* records) {
  const int target = blockIdx.x;
  const int total = n_fields * n_m * n_j;
  if (target >= total) {
    return;
  }
  const int tid = threadIdx.x;
  const int j_idx = target % n_j;
  const int m_idx = (target / n_j) % n_m;
  const int field_idx = target / (n_j * n_m);
  const int field = static_cast<int>(field_ids[field_idx]);
  const int m = m_targets[m_idx];
  const int j = j_targets[j_idx];

  extern __shared__ double shared[];
  double* sum_unw = shared;
  double* sum_vol = shared + blockDim.x;
  double* wsum = shared + 2 * blockDim.x;
  double* qmin = shared + 3 * blockDim.x;
  double* qmax = shared + 4 * blockDim.x;

  double local_sum_unw = 0.0;
  double local_sum_vol = 0.0;
  double local_wsum = 0.0;
  double local_qmin = 1.0e300;
  double local_qmax = -1.0e300;
  for (int i = tid; i < nr; i += blockDim.x) {
    const int c = i * nz + j;
    const double q = radial_fourier_complex_field_value(
        field, c, i, j, nz, n_groups, rho, mass, vol, Te, Ti, ee, ei, u_r,
        u_z, E_rad, Qvisc, fld_fleck, x_r, x_z);
    const double w = vol != nullptr ? finite_or_zero(vol[c]) : 1.0;
    local_sum_unw += q;
    local_sum_vol += w * q;
    local_wsum += w;
    local_qmin = fmin(local_qmin, q);
    local_qmax = fmax(local_qmax, q);
  }
  sum_unw[tid] = local_sum_unw;
  sum_vol[tid] = local_sum_vol;
  wsum[tid] = local_wsum;
  qmin[tid] = local_qmin;
  qmax[tid] = local_qmax;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sum_unw[tid] += sum_unw[tid + stride];
      sum_vol[tid] += sum_vol[tid + stride];
      wsum[tid] += wsum[tid + stride];
      qmin[tid] = fmin(qmin[tid], qmin[tid + stride]);
      qmax[tid] = fmax(qmax[tid], qmax[tid + stride]);
    }
    __syncthreads();
  }

  const double mean_unw = sum_unw[0] / static_cast<double>(nr);
  const double wsum_vol = wsum[0];
  const double mean_vol =
      fabs(wsum_vol) > kNormalizationFloor ? sum_vol[0] / wsum_vol : mean_unw;
  const double q_min_j = qmin[0];
  const double q_max_j = qmax[0];
  __syncthreads();

  double local_cre_unw = 0.0;
  double local_cim_unw = 0.0;
  double local_cre_vol = 0.0;
  double local_cim_vol = 0.0;
  for (int i = tid; i < nr; i += blockDim.x) {
    const int c = i * nz + j;
    const double q = radial_fourier_complex_field_value(
        field, c, i, j, nz, n_groups, rho, mass, vol, Te, Ti, ee, ei, u_r,
        u_z, E_rad, Qvisc, fld_fleck, x_r, x_z);
    const double w = vol != nullptr ? finite_or_zero(vol[c]) : 1.0;
    const double residual = q - mean_vol;
    const double theta =
        kTwoPi * static_cast<double>(m * i) / static_cast<double>(nr);
    const double ct = cos(theta);
    const double st = sin(theta);
    local_cre_unw += residual * ct;
    local_cim_unw -= residual * st;
    local_cre_vol += w * residual * ct;
    local_cim_vol -= w * residual * st;
  }

  sum_unw[tid] = local_cre_unw;
  sum_vol[tid] = local_cim_unw;
  wsum[tid] = local_cre_vol;
  qmin[tid] = local_cim_vol;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sum_unw[tid] += sum_unw[tid + stride];
      sum_vol[tid] += sum_vol[tid + stride];
      wsum[tid] += wsum[tid + stride];
      qmin[tid] += qmin[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    auto& rec = records[target];
    rec.field_id = static_cast<std::uint8_t>(field);
    rec.m = m;
    rec.j = j;
    rec.mean_unw = mean_unw;
    rec.cre_unw = sum_unw[0];
    rec.cim_unw = sum_vol[0];
    rec.amp_unw = sqrt(rec.cre_unw * rec.cre_unw + rec.cim_unw * rec.cim_unw);
    rec.phase_unw = atan2(rec.cim_unw, rec.cre_unw);
    rec.mean_vol = mean_vol;
    rec.cre_vol = wsum[0];
    rec.cim_vol = qmin[0];
    rec.amp_vol = sqrt(rec.cre_vol * rec.cre_vol + rec.cim_vol * rec.cim_vol);
    rec.phase_vol = atan2(rec.cim_vol, rec.cre_vol);
    rec.q_min_j = q_min_j;
    rec.q_max_j = q_max_j;
    rec.wsum_vol = wsum_vol;
  }
}
#endif

__device__ double fld_substage_audit_field_value(
    const double* __restrict__ field,
    const int layout,
    const int c,
    const int g,
    const int n_cells,
    const int n_groups) {
  if (field == nullptr) {
    return 0.0;
  }
  switch (static_cast<FldSubstageAuditFieldLayout>(layout)) {
    case FldSubstageAuditFieldLayout::CellScalar:
      return finite_or_zero(field[c]);
    case FldSubstageAuditFieldLayout::CellMajorGroup:
      return finite_or_zero(field[c * n_groups + g]);
    case FldSubstageAuditFieldLayout::GroupMajor:
      return finite_or_zero(field[g * n_cells + c]);
  }
  return 0.0;
}

__global__ void fld_substage_fixed_mode_kernel(
    const double* __restrict__ field,
    const int layout,
    const int* __restrict__ m_targets,
    const int* __restrict__ j_targets,
    const int n_m,
    const int n_j,
    const int nr,
    const int nz,
    const int n_groups,
    const std::uint8_t substage_id,
    const std::uint8_t field_id,
    const int outer_iter,
    const double solver_residual_l2_rel,
    const double solver_residual_max,
    FldSubstageAuditRecord* __restrict__ records) {
  const int target = blockIdx.x;
  const int total = n_groups * n_m * n_j;
  if (target >= total) {
    return;
  }
  const int tid = threadIdx.x;
  const int j_idx = target % n_j;
  const int m_idx = (target / n_j) % n_m;
  const int g = target / (n_j * n_m);
  const int m = m_targets[m_idx];
  const int j = j_targets[j_idx];
  const int n_cells = nr * nz;

  extern __shared__ double shared[];
  double* sum = shared;
  double* qmin = shared + blockDim.x;
  double* qmax = shared + 2 * blockDim.x;
  double local_sum = 0.0;
  double local_min = 1.0e300;
  double local_max = -1.0e300;
  for (int i = tid; i < nr; i += blockDim.x) {
    const int c = i * nz + j;
    const double q =
        fld_substage_audit_field_value(field, layout, c, g, n_cells, n_groups);
    local_sum += q;
    local_min = fmin(local_min, q);
    local_max = fmax(local_max, q);
  }
  sum[tid] = local_sum;
  qmin[tid] = local_min;
  qmax[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sum[tid] += sum[tid + stride];
      qmin[tid] = fmin(qmin[tid], qmin[tid + stride]);
      qmax[tid] = fmax(qmax[tid], qmax[tid + stride]);
    }
    __syncthreads();
  }

  const double mean = sum[0] / static_cast<double>(nr);
  const double row_min = qmin[0];
  const double row_max = qmax[0];
  __syncthreads();

  double local_cre = 0.0;
  double local_cim = 0.0;
  for (int i = tid; i < nr; i += blockDim.x) {
    const int c = i * nz + j;
    const double q =
        fld_substage_audit_field_value(field, layout, c, g, n_cells, n_groups);
    const double residual = q - mean;
    const double theta =
        kTwoPi * static_cast<double>(m * i) / static_cast<double>(nr);
    local_cre += residual * cos(theta);
    local_cim -= residual * sin(theta);
  }
  sum[tid] = local_cre;
  qmin[tid] = local_cim;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sum[tid] += sum[tid + stride];
      qmin[tid] += qmin[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    auto& rec = records[target];
    rec.valid = true;
    rec.substage_id = substage_id;
    rec.field_id = field_id;
    rec.normalization_kind = static_cast<std::uint8_t>(
        FldSubstageAuditNormalization::MeanSubtractedUnweightedRawSum);
    rec.m = m;
    rec.j = j;
    rec.group = g;
    rec.outer_iter = outer_iter;
    rec.nr = nr;
    rec.nz = nz;
    rec.cre = sum[0];
    rec.cim = qmin[0];
    rec.amplitude = sqrt(rec.cre * rec.cre + rec.cim * rec.cim);
    rec.phase = atan2(rec.cim, rec.cre);
    rec.mean = mean;
    rec.q_min_j = row_min;
    rec.q_max_j = row_max;
    rec.normalization = static_cast<double>(nr);
    rec.solver_residual_l2_rel = solver_residual_l2_rel;
    rec.solver_residual_max = solver_residual_max;
  }
}

#if TENRYU_RFA_V2_KERNEL_LAUNCHES
std::string normalize_complex_field_name(std::string name) {
  for (char& ch : name) {
    if (ch == '-' || ch == '/') {
      ch = '_';
    } else {
      ch = static_cast<char>(
          std::tolower(static_cast<unsigned char>(ch)));
    }
  }
  return name;
}

RadialFourierComplexFieldId parse_complex_field_name(const std::string& raw) {
  const std::string name = normalize_complex_field_name(raw);
  if (name == "rho") {
    return RadialFourierComplexFieldId::Rho;
  }
  if (name == "m" || name == "mass") {
    return RadialFourierComplexFieldId::M;
  }
  if (name == "v" || name == "vol" || name == "volume") {
    return RadialFourierComplexFieldId::V;
  }
  if (name == "m_over_v" || name == "mass_over_volume") {
    return RadialFourierComplexFieldId::MOverV;
  }
  if (name == "p_r" || name == "pr") {
    return RadialFourierComplexFieldId::Pr;
  }
  if (name == "p_z" || name == "pz") {
    return RadialFourierComplexFieldId::Pz;
  }
  if (name == "u_r" || name == "ur") {
    return RadialFourierComplexFieldId::Ur;
  }
  if (name == "u_z" || name == "uz") {
    return RadialFourierComplexFieldId::Uz;
  }
  if (name == "e_e" || name == "ee") {
    return RadialFourierComplexFieldId::Ee;
  }
  if (name == "e_i" || name == "ei") {
    return RadialFourierComplexFieldId::Ei;
  }
  if (name == "e_rad" || name == "erad") {
    return RadialFourierComplexFieldId::Erad;
  }
  if (name == "t_e" || name == "te") {
    return RadialFourierComplexFieldId::Te;
  }
  if (name == "t_i" || name == "ti") {
    return RadialFourierComplexFieldId::Ti;
  }
  if (name == "x_r" || name == "xr") {
    return RadialFourierComplexFieldId::Xr;
  }
  if (name == "x_z" || name == "xz") {
    return RadialFourierComplexFieldId::Xz;
  }
  if (name == "a_r" || name == "ar") {
    return RadialFourierComplexFieldId::Ar;
  }
  if (name == "a_z" || name == "az") {
    return RadialFourierComplexFieldId::Az;
  }
  if (name == "dv_swept" || name == "delta_v_swept") {
    return RadialFourierComplexFieldId::DVSwpt;
  }
  if (name == "q_visc" || name == "qvisc") {
    return RadialFourierComplexFieldId::Qvisc;
  }
  if (name == "lambda_fld") {
    return RadialFourierComplexFieldId::LambdaFld;
  }
  if (name == "r_fld") {
    return RadialFourierComplexFieldId::RFld;
  }
  if (name == "kappa_eff") {
    return RadialFourierComplexFieldId::KappaEff;
  }
  if (name == "f_fleck" || name == "fleck") {
    return RadialFourierComplexFieldId::FFleck;
  }
  if (name == "newton_iters") {
    return RadialFourierComplexFieldId::NewtonIters;
  }
  if (name == "newton_residual") {
    return RadialFourierComplexFieldId::NewtonResidual;
  }
  return RadialFourierComplexFieldId::NotAvailable;
}

bool complex_field_needs_cell_velocity(const RadialFourierComplexFieldId field) {
  return field == RadialFourierComplexFieldId::Pr ||
         field == RadialFourierComplexFieldId::Pz ||
         field == RadialFourierComplexFieldId::Ur ||
         field == RadialFourierComplexFieldId::Uz;
}

bool complex_field_available(const RadialFourierComplexFieldId field,
                             const core::State& state,
                             const int n_cells,
                             const int n_nodes,
                             const int n_groups) {
  const auto cell_sized = [n_cells](const auto& f) {
    return f.size() == static_cast<std::size_t>(n_cells);
  };
  const auto node_sized = [n_nodes](const auto& f) {
    return f.size() == static_cast<std::size_t>(n_nodes);
  };
  switch (field) {
    case RadialFourierComplexFieldId::Rho:
      return cell_sized(state.rho);
    case RadialFourierComplexFieldId::M:
      return cell_sized(state.mass) ||
             (cell_sized(state.rho) && cell_sized(state.vol));
    case RadialFourierComplexFieldId::V:
      return cell_sized(state.vol);
    case RadialFourierComplexFieldId::MOverV:
      return cell_sized(state.vol) &&
             (cell_sized(state.mass) || cell_sized(state.rho));
    case RadialFourierComplexFieldId::Pr:
    case RadialFourierComplexFieldId::Pz:
      return cell_sized(state.rho) && node_sized(state.v_r) &&
             node_sized(state.v_z);
    case RadialFourierComplexFieldId::Ur:
    case RadialFourierComplexFieldId::Uz:
      return node_sized(state.v_r) && node_sized(state.v_z);
    case RadialFourierComplexFieldId::Ee:
      return cell_sized(state.rho) && cell_sized(state.ee);
    case RadialFourierComplexFieldId::Ei:
      return cell_sized(state.rho) && cell_sized(state.ei);
    case RadialFourierComplexFieldId::Erad:
      return n_groups > 0 &&
             state.rad_E.size() >=
                 static_cast<std::size_t>(n_cells) *
                     static_cast<std::size_t>(n_groups);
    case RadialFourierComplexFieldId::Te:
      return cell_sized(state.Te);
    case RadialFourierComplexFieldId::Ti:
      return cell_sized(state.Ti);
    case RadialFourierComplexFieldId::Xr:
    case RadialFourierComplexFieldId::Xz:
    case RadialFourierComplexFieldId::Ar:
    case RadialFourierComplexFieldId::Az:
      return node_sized(state.x_r) && node_sized(state.x_z);
    case RadialFourierComplexFieldId::Qvisc:
      return cell_sized(state.Qvisc);
    case RadialFourierComplexFieldId::FFleck:
      return cell_sized(state.fld_fleck);
    case RadialFourierComplexFieldId::DVSwpt:
    case RadialFourierComplexFieldId::LambdaFld:
    case RadialFourierComplexFieldId::RFld:
    case RadialFourierComplexFieldId::KappaEff:
    case RadialFourierComplexFieldId::NewtonIters:
    case RadialFourierComplexFieldId::NewtonResidual:
    case RadialFourierComplexFieldId::NotAvailable:
      return false;
  }
  return false;
}

int infer_radiation_group_count(const core::State& state,
                                const core::Config& cfg,
                                const int n_cells) {
  int n_groups = cfg.radiation.groups;
  if (n_groups <= 0 ||
      state.rad_E.size() <
          static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups)) {
    if (n_cells > 0 &&
        state.rad_E.size() % static_cast<std::size_t>(n_cells) == 0) {
      n_groups =
          static_cast<int>(state.rad_E.size() / static_cast<std::size_t>(n_cells));
    } else {
      n_groups = 0;
    }
  }
  return n_groups;
}
#endif

}  // namespace

const char* radial_fourier_stage_name(const RadialFourierStageId stage) {
  switch (stage) {
    case RadialFourierStageId::HydroLag:
      return "hydro_lag";
    case RadialFourierStageId::ArtificialViscosity:
      return "AV";
    case RadialFourierStageId::Winslow:
      return "winslow";
    case RadialFourierStageId::Remap:
      return "remap";
    case RadialFourierStageId::BoundaryFill:
      return "bc_fill";
    case RadialFourierStageId::FldSolve:
      return "FLD_solve";
    case RadialFourierStageId::NewtonSource:
      return "Newton_src";
    case RadialFourierStageId::Positivity:
      return "positivity";
    case RadialFourierStageId::DtController:
      return "dt_ctrl";
  }
  return "unknown";
}

const char* radial_fourier_field_name(const RadialFourierFieldId field) {
  switch (field) {
    case RadialFourierFieldId::Rho:
      return "rho";
    case RadialFourierFieldId::Te:
      return "Te";
    case RadialFourierFieldId::Ti:
      return "Ti";
    case RadialFourierFieldId::Ur:
      return "u_r";
    case RadialFourierFieldId::Uz:
      return "u_z";
    case RadialFourierFieldId::Erad:
      return "E_rad";
  }
  return "unknown";
}

const char* radial_fourier_complex_field_name(
    const RadialFourierComplexFieldId field) {
  switch (field) {
    case RadialFourierComplexFieldId::Rho:
      return "rho";
    case RadialFourierComplexFieldId::M:
      return "M";
    case RadialFourierComplexFieldId::V:
      return "V";
    case RadialFourierComplexFieldId::MOverV:
      return "M_over_V";
    case RadialFourierComplexFieldId::Pr:
      return "P_r";
    case RadialFourierComplexFieldId::Pz:
      return "P_z";
    case RadialFourierComplexFieldId::Ur:
      return "u_r";
    case RadialFourierComplexFieldId::Uz:
      return "u_z";
    case RadialFourierComplexFieldId::Ee:
      return "E_e";
    case RadialFourierComplexFieldId::Ei:
      return "E_i";
    case RadialFourierComplexFieldId::Erad:
      return "E_rad";
    case RadialFourierComplexFieldId::Te:
      return "T_e";
    case RadialFourierComplexFieldId::Ti:
      return "T_i";
    case RadialFourierComplexFieldId::Xr:
      return "x_r";
    case RadialFourierComplexFieldId::Xz:
      return "x_z";
    case RadialFourierComplexFieldId::Ar:
      return "A_r";
    case RadialFourierComplexFieldId::Az:
      return "A_z";
    case RadialFourierComplexFieldId::DVSwpt:
      return "dV_swept";
    case RadialFourierComplexFieldId::Qvisc:
      return "Q_visc";
    case RadialFourierComplexFieldId::LambdaFld:
      return "lambda_FLD";
    case RadialFourierComplexFieldId::RFld:
      return "R_FLD";
    case RadialFourierComplexFieldId::KappaEff:
      return "kappa_eff";
    case RadialFourierComplexFieldId::FFleck:
      return "f_Fleck";
    case RadialFourierComplexFieldId::NewtonIters:
      return "newton_iters";
    case RadialFourierComplexFieldId::NewtonResidual:
      return "newton_residual";
    case RadialFourierComplexFieldId::NotAvailable:
      return "NOT_AVAILABLE";
  }
  return "unknown";
}

std::vector<FldSubstageAuditRecord> compute_fld_substage_fixed_mode_audit(
    const double* const device_field,
    const FldSubstageAuditFieldLayout layout,
    const int nr,
    const int nz,
    const int n_groups,
    const FldSubstageAuditSubstageId substage,
    const FldSubstageAuditFieldId field,
    const int outer_iter,
    const std::vector<int>& requested_m_targets,
    const std::vector<int>& requested_j_targets,
    const double solver_residual_l2_rel,
    const double solver_residual_max) {
  std::vector<FldSubstageAuditRecord> result;
  if (device_field == nullptr || nr <= 0 || nz <= 0 || n_groups <= 0) {
    return result;
  }

  std::vector<int> m_targets;
  m_targets.reserve(requested_m_targets.size());
  const int max_mode = nr / 2;
  for (const int m : requested_m_targets) {
    if (m < 0 || m > max_mode) {
      continue;
    }
    if (std::find(m_targets.begin(), m_targets.end(), m) == m_targets.end()) {
      m_targets.push_back(m);
    }
  }

  std::vector<int> j_targets;
  j_targets.reserve(requested_j_targets.size());
  for (const int j : requested_j_targets) {
    if (j < 0 || j >= nz) {
      continue;
    }
    if (std::find(j_targets.begin(), j_targets.end(), j) == j_targets.end()) {
      j_targets.push_back(j);
    }
  }

  if (m_targets.empty() || j_targets.empty()) {
    return result;
  }

  int* d_m_targets = nullptr;
  int* d_j_targets = nullptr;
  FldSubstageAuditRecord* d_records = nullptr;
  const std::size_t n_m = m_targets.size();
  const std::size_t n_j = j_targets.size();
  const std::size_t n_records =
      static_cast<std::size_t>(n_groups) * n_m * n_j;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_m_targets),
                        n_m * sizeof(int)),
             "FLD substage audit cudaMalloc m_targets");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_j_targets),
                        n_j * sizeof(int)),
             "FLD substage audit cudaMalloc j_targets");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_records),
                        n_records * sizeof(FldSubstageAuditRecord)),
             "FLD substage audit cudaMalloc records");
  cuda_check(cudaMemcpy(d_m_targets,
                        m_targets.data(),
                        n_m * sizeof(int),
                        cudaMemcpyHostToDevice),
             "FLD substage audit cudaMemcpy m_targets H2D");
  cuda_check(cudaMemcpy(d_j_targets,
                        j_targets.data(),
                        n_j * sizeof(int),
                        cudaMemcpyHostToDevice),
             "FLD substage audit cudaMemcpy j_targets H2D");

  const int threads = reduction_thread_count(nr);
  const std::size_t shared_bytes =
      3U * static_cast<std::size_t>(threads) * sizeof(double);
  fld_substage_fixed_mode_kernel<<<static_cast<int>(n_records),
                                   threads,
                                   shared_bytes>>>(
      device_field,
      static_cast<int>(layout),
      d_m_targets,
      d_j_targets,
      static_cast<int>(n_m),
      static_cast<int>(n_j),
      nr,
      nz,
      n_groups,
      static_cast<std::uint8_t>(substage),
      static_cast<std::uint8_t>(field),
      outer_iter,
      solver_residual_l2_rel,
      solver_residual_max,
      d_records);
  cuda_check(cudaGetLastError(), "FLD substage audit fixed-mode launch");

  result.resize(n_records);
  cuda_check(cudaMemcpy(result.data(),
                        d_records,
                        n_records * sizeof(FldSubstageAuditRecord),
                        cudaMemcpyDeviceToHost),
             "FLD substage audit cudaMemcpy records D2H");
  cuda_check(cudaFree(d_records), "FLD substage audit cudaFree records");
  cuda_check(cudaFree(d_j_targets), "FLD substage audit cudaFree j_targets");
  cuda_check(cudaFree(d_m_targets), "FLD substage audit cudaFree m_targets");
  return result;
}

RadialFourierResult compute_radial_fourier_audit(
    const core::State& state,
    const core::Config& cfg) {
  RadialFourierResult result{};
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = nr * nz;
  const int n_nodes = (nr + 1) * (nz + 1);
  if (state.mesh.dim != 2 || nr <= 0 || nz <= 0 || n_cells <= 0 ||
      state.rho.size() != static_cast<std::size_t>(n_cells)) {
    return result;
  }
  TENRYU_ASSERT(state.Te.empty() || state.Te.size() == static_cast<std::size_t>(n_cells),
                "radial Fourier audit requires Te size == n_cells");
  TENRYU_ASSERT(state.Ti.empty() || state.Ti.size() == static_cast<std::size_t>(n_cells),
                "radial Fourier audit requires Ti size == n_cells");
  TENRYU_ASSERT(state.v_r.empty() ||
                    state.v_r.size() == static_cast<std::size_t>(n_nodes),
                "radial Fourier audit requires v_r size == n_nodes");
  TENRYU_ASSERT(state.v_z.empty() ||
                    state.v_z.size() == static_cast<std::size_t>(n_nodes),
                "radial Fourier audit requires v_z size == n_nodes");

  const int all_modes = nr / 2 + 1;
  const int max_mode = cfg.diagnostics.radial_fourier_max_mode < 0
                           ? all_modes - 1
                           : std::min(cfg.diagnostics.radial_fourier_max_mode,
                                      all_modes - 1);
  const int n_modes = max_mode + 1;
  if (n_modes <= 0) {
    return result;
  }

  int n_groups = cfg.radiation.groups;
  if (n_groups <= 0 ||
      state.rad_E.size() <
          static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups)) {
    if (state.rad_E.size() % static_cast<std::size_t>(n_cells) == 0) {
      n_groups =
          static_cast<int>(state.rad_E.size() / static_cast<std::size_t>(n_cells));
    } else {
      n_groups = 0;
    }
  }
  const double* E_rad_ptr = n_groups > 0 ? state.rad_E.data() : nullptr;

  double* u_r_cell = nullptr;
  double* u_z_cell = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&u_r_cell),
                        static_cast<std::size_t>(n_cells) * sizeof(double)),
             "radial Fourier audit cudaMalloc u_r_cell");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&u_z_cell),
                        static_cast<std::size_t>(n_cells) * sizeof(double)),
             "radial Fourier audit cudaMalloc u_z_cell");
  if (!state.v_r.empty() && !state.v_z.empty()) {
    const int threads = 256;
    const int blocks = (n_cells + threads - 1) / threads;
    radial_fourier_cell_velocity_kernel<<<blocks, threads>>>(
        state.v_r.data(), state.v_z.data(), nr, nz, u_r_cell, u_z_cell);
    cuda_check(cudaGetLastError(),
               "radial Fourier audit launch cell velocity kernel");
  } else {
    cuda_check(cudaMemset(u_r_cell, 0, static_cast<std::size_t>(n_cells) * sizeof(double)),
               "radial Fourier audit cudaMemset u_r_cell");
    cuda_check(cudaMemset(u_z_cell, 0, static_cast<std::size_t>(n_cells) * sizeof(double)),
               "radial Fourier audit cudaMemset u_z_cell");
  }

  const std::size_t amp_count =
      static_cast<std::size_t>(kRadialFourierFieldCount) *
      static_cast<std::size_t>(n_modes) * static_cast<std::size_t>(nz);
  const std::size_t mean_count =
      static_cast<std::size_t>(kRadialFourierFieldCount) *
      static_cast<std::size_t>(nz);
  double* d_amp = nullptr;
  double* d_mean = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_amp), amp_count * sizeof(double)),
             "radial Fourier audit cudaMalloc A_mode_amp");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mean), mean_count * sizeof(double)),
             "radial Fourier audit cudaMalloc mean_j");

  const int threads = reduction_thread_count(nr);
  const dim3 grid(n_modes, nz, kRadialFourierFieldCount);
  const std::size_t shared_bytes = 2U * static_cast<std::size_t>(threads) * sizeof(double);
  radial_fourier_audit_kernel<<<grid, threads, shared_bytes>>>(
      state.rho.data(),
      state.Te.empty() ? nullptr : state.Te.data(),
      state.Ti.empty() ? nullptr : state.Ti.data(),
      u_r_cell,
      u_z_cell,
      E_rad_ptr,
      nr,
      nz,
      n_groups,
      d_amp,
      d_mean);
  cuda_check(cudaGetLastError(), "radial Fourier audit launch audit kernel");

  std::vector<double> amp(amp_count, 0.0);
  cuda_check(cudaMemcpy(amp.data(), d_amp, amp_count * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "radial Fourier audit cudaMemcpy A_mode_amp D2H");

  cuda_check(cudaFree(d_mean), "radial Fourier audit cudaFree mean_j");
  cuda_check(cudaFree(d_amp), "radial Fourier audit cudaFree A_mode_amp");
  cuda_check(cudaFree(u_z_cell), "radial Fourier audit cudaFree u_z_cell");
  cuda_check(cudaFree(u_r_cell), "radial Fourier audit cudaFree u_r_cell");

  result.valid = true;
  result.nr = nr;
  result.nz = nz;
  result.n_modes = n_modes;
  for (int field = 0; field < kRadialFourierFieldCount; ++field) {
    auto& maximum = result.fields[static_cast<std::size_t>(field)];
    maximum.A_max = -std::numeric_limits<double>::infinity();
    maximum.m_max = 0;
    maximum.j_max = 0;
    for (int m = 0; m < n_modes; ++m) {
      for (int j = 0; j < nz; ++j) {
        const double value = amp[(field * n_modes + m) * nz + j];
        if (std::isfinite(value) && value > maximum.A_max) {
          maximum.A_max = value;
          maximum.m_max = m;
          maximum.j_max = j;
        }
      }
    }
    if (!std::isfinite(maximum.A_max)) {
      maximum.A_max = 0.0;
    }
  }
  return result;
}

RadialFourierComplexResult compute_radial_fourier_complex_audit(
    const core::State& state,
    const core::Config& cfg) {
  RadialFourierComplexResult result{};
#if TENRYU_RFA_V2_MODE_IS_OFF || TENRYU_RFA_V2_MODE_IS_STUB
  (void)state;
  (void)cfg;
  return result;
#else
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = nr * nz;
  const int n_nodes = (nr + 1) * (nz + 1);
  if (state.mesh.dim != 2 || nr <= 0 || nz <= 0 || n_cells <= 0 ||
      state.rho.size() != static_cast<std::size_t>(n_cells) ||
      state.vol.size() != static_cast<std::size_t>(n_cells)) {
    return result;
  }

  const int n_groups = infer_radiation_group_count(state, cfg, n_cells);
  std::vector<std::uint8_t> field_ids;
  field_ids.reserve(
      cfg.diagnostics.per_operator_radial_fourier_complex_fields.size());
  bool need_cell_velocity = false;
  for (const std::string& name :
       cfg.diagnostics.per_operator_radial_fourier_complex_fields) {
    const auto field = parse_complex_field_name(name);
    if (field == RadialFourierComplexFieldId::NotAvailable ||
        !complex_field_available(field, state, n_cells, n_nodes, n_groups)) {
      continue;
    }
    const auto id = static_cast<std::uint8_t>(field);
    if (std::find(field_ids.begin(), field_ids.end(), id) != field_ids.end()) {
      continue;
    }
    field_ids.push_back(id);
    need_cell_velocity = need_cell_velocity ||
                         complex_field_needs_cell_velocity(field);
  }

  const int max_mode = nr / 2;
  std::vector<int> m_targets;
  m_targets.reserve(
      cfg.diagnostics.per_operator_radial_fourier_complex_m_targets.size());
  for (const int m :
       cfg.diagnostics.per_operator_radial_fourier_complex_m_targets) {
    if (m < 0 || m > max_mode) {
      continue;
    }
    if (std::find(m_targets.begin(), m_targets.end(), m) == m_targets.end()) {
      m_targets.push_back(m);
    }
  }

  std::vector<int> j_targets;
  j_targets.reserve(
      cfg.diagnostics.per_operator_radial_fourier_complex_j_targets.size());
  for (const int j :
       cfg.diagnostics.per_operator_radial_fourier_complex_j_targets) {
    if (j < 0 || j >= nz) {
      continue;
    }
    if (std::find(j_targets.begin(), j_targets.end(), j) == j_targets.end()) {
      j_targets.push_back(j);
    }
  }

  if (field_ids.empty() || m_targets.empty() || j_targets.empty()) {
    return result;
  }
  result.valid = true;
  result.nr = nr;
  result.nz = nz;

  double* u_r_cell = nullptr;
  double* u_z_cell = nullptr;
  if (need_cell_velocity) {
    TENRYU_ASSERT(state.v_r.size() == static_cast<std::size_t>(n_nodes),
                  "radial Fourier v2 audit requires v_r size == n_nodes");
    TENRYU_ASSERT(state.v_z.size() == static_cast<std::size_t>(n_nodes),
                  "radial Fourier v2 audit requires v_z size == n_nodes");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&u_r_cell),
                          static_cast<std::size_t>(n_cells) * sizeof(double)),
               "radial Fourier v2 audit cudaMalloc u_r_cell");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&u_z_cell),
                          static_cast<std::size_t>(n_cells) * sizeof(double)),
               "radial Fourier v2 audit cudaMalloc u_z_cell");
    const int threads = 256;
    const int blocks = (n_cells + threads - 1) / threads;
    radial_fourier_cell_velocity_kernel<<<blocks, threads>>>(
        state.v_r.data(), state.v_z.data(), nr, nz, u_r_cell, u_z_cell);
    cuda_check(cudaGetLastError(),
               "radial Fourier v2 audit launch cell velocity kernel");
  }

  std::uint8_t* d_field_ids = nullptr;
  int* d_m_targets = nullptr;
  int* d_j_targets = nullptr;
  RadialFourierComplexCoeff* d_records = nullptr;
  const std::size_t n_fields = field_ids.size();
  const std::size_t n_m = m_targets.size();
  const std::size_t n_j = j_targets.size();
  const std::size_t n_records = n_fields * n_m * n_j;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_field_ids),
                        n_fields * sizeof(std::uint8_t)),
             "radial Fourier v2 audit cudaMalloc field_ids");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_m_targets),
                        n_m * sizeof(int)),
             "radial Fourier v2 audit cudaMalloc m_targets");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_j_targets),
                        n_j * sizeof(int)),
             "radial Fourier v2 audit cudaMalloc j_targets");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_records),
                        n_records * sizeof(RadialFourierComplexCoeff)),
             "radial Fourier v2 audit cudaMalloc records");
  cuda_check(cudaMemcpy(d_field_ids,
                        field_ids.data(),
                        n_fields * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             "radial Fourier v2 audit cudaMemcpy field_ids H2D");
  cuda_check(cudaMemcpy(d_m_targets,
                        m_targets.data(),
                        n_m * sizeof(int),
                        cudaMemcpyHostToDevice),
             "radial Fourier v2 audit cudaMemcpy m_targets H2D");
  cuda_check(cudaMemcpy(d_j_targets,
                        j_targets.data(),
                        n_j * sizeof(int),
                        cudaMemcpyHostToDevice),
             "radial Fourier v2 audit cudaMemcpy j_targets H2D");

  const int threads = reduction_thread_count(nr);
  const std::size_t shared_bytes =
      5U * static_cast<std::size_t>(threads) * sizeof(double);
  radial_fourier_fixed_mode_kernel<<<static_cast<int>(n_records),
                                     threads,
                                     shared_bytes>>>(
      d_field_ids,
      d_m_targets,
      d_j_targets,
      static_cast<int>(n_fields),
      static_cast<int>(n_m),
      static_cast<int>(n_j),
      state.rho.data(),
      state.mass.size() == static_cast<std::size_t>(n_cells) ? state.mass.data()
                                                             : nullptr,
      state.vol.data(),
      state.Te.size() == static_cast<std::size_t>(n_cells) ? state.Te.data()
                                                           : nullptr,
      state.Ti.size() == static_cast<std::size_t>(n_cells) ? state.Ti.data()
                                                           : nullptr,
      state.ee.size() == static_cast<std::size_t>(n_cells) ? state.ee.data()
                                                           : nullptr,
      state.ei.size() == static_cast<std::size_t>(n_cells) ? state.ei.data()
                                                           : nullptr,
      u_r_cell,
      u_z_cell,
      n_groups > 0 ? state.rad_E.data() : nullptr,
      state.Qvisc.size() == static_cast<std::size_t>(n_cells) ? state.Qvisc.data()
                                                              : nullptr,
      state.fld_fleck.size() == static_cast<std::size_t>(n_cells)
          ? state.fld_fleck.data()
          : nullptr,
      state.x_r.size() == static_cast<std::size_t>(n_nodes) ? state.x_r.data()
                                                            : nullptr,
      state.x_z.size() == static_cast<std::size_t>(n_nodes) ? state.x_z.data()
                                                            : nullptr,
      nr,
      nz,
      n_groups,
      d_records);
  cuda_check(cudaGetLastError(),
             "radial Fourier v2 audit launch fixed-mode kernel");

#if TENRYU_RFA_V2_CAPTURES_COEFFS
  result.coeffs.resize(n_records);
  cuda_check(cudaMemcpy(result.coeffs.data(),
                        d_records,
                        n_records * sizeof(RadialFourierComplexCoeff),
                        cudaMemcpyDeviceToHost),
             "radial Fourier v2 audit cudaMemcpy records D2H");
#else
  result.valid = false;
#endif

  cuda_check(cudaFree(d_records), "radial Fourier v2 audit cudaFree records");
  cuda_check(cudaFree(d_j_targets), "radial Fourier v2 audit cudaFree j_targets");
  cuda_check(cudaFree(d_m_targets), "radial Fourier v2 audit cudaFree m_targets");
  cuda_check(cudaFree(d_field_ids), "radial Fourier v2 audit cudaFree field_ids");
  if (u_z_cell != nullptr) {
    cuda_check(cudaFree(u_z_cell), "radial Fourier v2 audit cudaFree u_z_cell");
  }
  if (u_r_cell != nullptr) {
    cuda_check(cudaFree(u_r_cell), "radial Fourier v2 audit cudaFree u_r_cell");
  }
  return result;
#endif
}

RadialFourierAuditRecord make_radial_fourier_audit_record(
    const RadialFourierResult& result,
    const std::uint64_t cycle,
    const double t_s,
    const RadialFourierStageId stage,
    const RadialFourierStagePhase phase) {
  RadialFourierAuditRecord record{};
  record.valid = result.valid;
  record.cycle = cycle;
  record.t_s = t_s;
  record.stage_id = static_cast<std::uint8_t>(stage);
  record.stage_phase = static_cast<std::uint8_t>(phase);
  record.fields = result.fields;
  return record;
}

RadialFourierComplexAuditRecord make_radial_fourier_complex_audit_record(
    const RadialFourierComplexResult& result,
    const std::uint64_t cycle,
    const double t_s,
    const double dt_cycle,
    const RadialFourierStageId stage,
    const RadialFourierStagePhase phase) {
  RadialFourierComplexAuditRecord record{};
  record.valid = result.valid;
  record.cycle = cycle;
  record.t_s = t_s;
  record.dt_cycle = dt_cycle;
  record.stage_id = static_cast<std::uint8_t>(stage);
  record.stage_phase = static_cast<std::uint8_t>(phase);
  record.coeffs = result.coeffs;
  return record;
}

}  // namespace tenryu::diagnostics
