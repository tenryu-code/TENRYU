#include "radiation/deterministic_diffusion_1d.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "core/error.hpp"

namespace tenryu::radiation {
namespace {

constexpr int kBlock = 256;
constexpr double kFourPi = 12.56637061435917295385;
constexpr double kSigmaFloor = 1.0e-30;
constexpr double kEnergyBalanceRelTol = 1.0e-8;
constexpr double kEnergyBalanceFloor = 1.0e-30;
constexpr int kMaxRKL2Subcycles = 10;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__host__ __device__ inline double finite_or_zero(const double x) {
  return isfinite(x) ? x : 0.0;
}

__host__ __device__ inline double harmonic_face_d(const double a, const double b) {
  const double denom = a + b;
  return (denom > 0.0) ? (2.0 * a * b / denom) : 0.0;
}

__global__ void compute_d_cell_kernel(const double* __restrict__ sigma_R,
                                      const double* __restrict__ node_r,
                                      const std::uint8_t* __restrict__ diff_cell,
                                      double* __restrict__ D_cell,
                                      const int n_cells,
                                      const int n_groups) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_total = n_cells * n_groups;
  if (idx >= n_total) {
    return;
  }
  const int c = idx / n_groups;
  if (diff_cell[c] == 0U) {
    D_cell[idx] = 0.0;
    return;
  }
  const double sigma = fmax(finite_or_zero(sigma_R[idx]), kSigmaFloor);
  const double dx =
      fmax(finite_or_zero(node_r[c + 1]) - finite_or_zero(node_r[c]), 0.0);
  const double D_raw = tenryu::core::constants::c_light / (3.0 * sigma);
  const double D_max = tenryu::core::constants::c_light * dx / 6.0;
  D_cell[idx] = (D_max > 0.0) ? fmin(D_raw, D_max) : 0.0;
}

__global__ void diffusion_operator_1d_kernel(
    const double* __restrict__ E_in,
    double* __restrict__ L_out,
    const double* __restrict__ D_cell,
    const double* __restrict__ vol,
    const double* __restrict__ node_r,
    const std::uint8_t* __restrict__ diff_cell,
    const int n_cells,
    const int n_groups,
    const int bc_inner,
    const int bc_outer) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_total = n_cells * n_groups;
  if (idx >= n_total) {
    return;
  }

  const int c = idx / n_groups;
  const int g = idx % n_groups;
  if (diff_cell[c] == 0U) {
    L_out[idx] = 0.0;
    return;
  }

  const double r_left = finite_or_zero(node_r[c]);
  const double r_right = finite_or_zero(node_r[c + 1]);
  const double r_c = 0.5 * (r_left + r_right);
  const double V = finite_or_zero(vol[c]);
  if (!(V > 0.0)) {
    L_out[idx] = 0.0;
    return;
  }

  const double A_left = kFourPi * r_left * r_left;
  const double A_right = kFourPi * r_right * r_right;
  const double D_c = fmax(finite_or_zero(D_cell[idx]), 0.0);
  const double E_c = finite_or_zero(E_in[idx]);

  double F_left = 0.0;
  if (c > 0 && diff_cell[c - 1] != 0U) {
    const int idx_left = (c - 1) * n_groups + g;
    const double D_left = fmax(finite_or_zero(D_cell[idx_left]), 0.0);
    const double D_face = harmonic_face_d(D_c, D_left);
    const double r_c_left = 0.5 * (finite_or_zero(node_r[c - 1]) + r_left);
    const double dr = r_c - r_c_left;
    if (D_face > 0.0 && dr > 0.0) {
      F_left = -D_face * (E_c - finite_or_zero(E_in[idx_left])) / dr;
    }
  } else if (c == 0 && bc_inner == 1) {
    F_left = -0.25 * tenryu::core::constants::c_light * E_c;
  }

  double F_right = 0.0;
  if (c + 1 < n_cells && diff_cell[c + 1] != 0U) {
    const int idx_right = (c + 1) * n_groups + g;
    const double D_right = fmax(finite_or_zero(D_cell[idx_right]), 0.0);
    const double D_face = harmonic_face_d(D_c, D_right);
    const double r_c_right =
        0.5 * (r_right + finite_or_zero(node_r[c + 2]));
    const double dr = r_c_right - r_c;
    if (D_face > 0.0 && dr > 0.0) {
      F_right =
          -D_face * (finite_or_zero(E_in[idx_right]) - E_c) / dr;
    }
  } else if (c == n_cells - 1 && bc_outer == 1) {
    F_right = 0.25 * tenryu::core::constants::c_light * E_c;
  }

  double L = (A_left * F_left - A_right * F_right) / V;
  L_out[idx] = isfinite(L) ? L : 0.0;
}

__global__ void deposit_face_current_in_kernel(
    double* __restrict__ diff_E,
    double* __restrict__ face_current_in,
    const double* __restrict__ vol,
    const std::uint8_t* __restrict__ diff_cell,
    const int n_cells,
    const int n_groups) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_total = n_cells * n_groups;
  if (idx >= n_total) {
    return;
  }

  const int c = idx / n_groups;
  const int g = idx % n_groups;
  if (diff_cell[c] == 0U) {
    return;
  }

  const double V = finite_or_zero(vol[c]);
  if (!(V > 0.0)) {
    return;
  }

  double J_in = 0.0;
  if (c == 0 || diff_cell[c - 1] == 0U) {
    const int face_idx = c * n_groups + g;
    const double J = fmax(finite_or_zero(face_current_in[face_idx]), 0.0);
    if (J > 0.0) {
      J_in += J;
      face_current_in[face_idx] = 0.0;
    }
  }
  if (c + 1 == n_cells || diff_cell[c + 1] == 0U) {
    const int face_idx = (c + 1) * n_groups + g;
    const double J = fmax(finite_or_zero(face_current_in[face_idx]), 0.0);
    if (J > 0.0) {
      J_in += J;
      face_current_in[face_idx] = 0.0;
    }
  }

  if (J_in > 0.0) {
    diff_E[idx] += J_in / V;
  }
}

__global__ void copy_kernel(const double* __restrict__ src,
                            double* __restrict__ dst,
                            const int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

__global__ void stage1_kernel(const double* __restrict__ Y0,
                              const double* __restrict__ L0,
                              double* __restrict__ Y1,
                              const int n,
                              const double dt,
                              const double mu_tilde1) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    Y1[idx] = Y0[idx] + mu_tilde1 * dt * L0[idx];
  }
}

__global__ void stagej_kernel(const double* __restrict__ Y0,
                              const double* __restrict__ Yjm1,
                              const double* __restrict__ Yjm2,
                              const double* __restrict__ L0,
                              const double* __restrict__ Ljm1,
                              double* __restrict__ Yj,
                              const int n,
                              const double dt,
                              const double mu,
                              const double nu,
                              const double mu_tilde,
                              const double gamma_tilde) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    Yj[idx] = (1.0 - mu - nu) * Y0[idx] + mu * Yjm1[idx] +
              nu * Yjm2[idx] + mu_tilde * dt * Ljm1[idx] +
              gamma_tilde * dt * L0[idx];
  }
}

std::vector<double> copy_device_double_array(const double* ptr,
                                             const std::size_t n,
                                             const char* message) {
  std::vector<double> out(n, 0.0);
  if (n > 0U) {
    TENRYU_ASSERT(ptr != nullptr, message);
    cuda_check(cudaMemcpy(out.data(), ptr, sizeof(double) * n, cudaMemcpyDeviceToHost),
               message);
  }
  return out;
}

std::vector<std::uint8_t> copy_device_u8_array(const std::uint8_t* ptr,
                                               const std::size_t n,
                                               const char* message) {
  std::vector<std::uint8_t> out(n, 0U);
  if (n > 0U) {
    TENRYU_ASSERT(ptr != nullptr, message);
    cuda_check(cudaMemcpy(out.data(), ptr, sizeof(std::uint8_t) * n, cudaMemcpyDeviceToHost),
               message);
  }
  return out;
}

struct DiffusionEnergyMoments {
  double signed_total = 0.0;
  double positive_total = 0.0;
  double negative_total = 0.0;
};

struct PositivityLimiterResult {
  DiffusionEnergyMoments before;
  DiffusionEnergyMoments after;
  double removed_positive = 0.0;
  bool applied = false;
};

DiffusionEnergyMoments diffusion_energy_moments_host(
    const std::vector<double>& E,
    const std::vector<double>& vol,
    const std::vector<std::uint8_t>& diff_cell,
    const std::size_t n_cells,
    const std::size_t n_groups) {
  DiffusionEnergyMoments out{};
  long double signed_sum = 0.0L;
  long double positive_sum = 0.0L;
  long double negative_sum = 0.0L;
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (diff_cell[c] == 0U) {
      continue;
    }
    const double V = std::max(finite_or_zero(vol[c]), 0.0);
    if (!(V > 0.0)) {
      continue;
    }
    for (std::size_t g = 0; g < n_groups; ++g) {
      const std::size_t idx = c * n_groups + g;
      const double E_cg = finite_or_zero(E[idx]);
      const long double cell_E =
          static_cast<long double>(E_cg) * static_cast<long double>(V);
      signed_sum += cell_E;
      if (E_cg > 0.0) {
        positive_sum += cell_E;
      } else if (E_cg < 0.0) {
        negative_sum += -cell_E;
      }
    }
  }
  out.signed_total = static_cast<double>(signed_sum);
  out.positive_total = static_cast<double>(positive_sum);
  out.negative_total = static_cast<double>(negative_sum);
  return out;
}

PositivityLimiterResult apply_conservative_positivity_limiter(
    const DiffusionStepInputs& in) {
  PositivityLimiterResult out{};
  const std::size_t n_cells = static_cast<std::size_t>(std::max(in.n_cells, 0));
  const std::size_t n_groups = static_cast<std::size_t>(std::max(in.n_groups, 0));
  const std::size_t n_total = n_cells * n_groups;
  if (n_cells == 0U || n_groups == 0U || n_total == 0U) {
    return out;
  }
  std::vector<double> E =
      copy_device_double_array(in.diff_E, n_total, "diffusion limiter copy E failed");
  const std::vector<double> vol =
      copy_device_double_array(in.vol, n_cells, "diffusion limiter copy volume failed");
  const std::vector<std::uint8_t> diff_cell =
      copy_device_u8_array(in.diff_cell, n_cells, "diffusion limiter copy mask failed");

  out.before = diffusion_energy_moments_host(E, vol, diff_cell, n_cells, n_groups);
  if (!(out.before.negative_total > 0.0)) {
    out.after = out.before;
    return out;
  }

  for (std::size_t g = 0; g < n_groups; ++g) {
    long double signed_sum = 0.0L;
    long double positive_sum = 0.0L;
    for (std::size_t c = 0; c < n_cells; ++c) {
      if (diff_cell[c] == 0U) {
        continue;
      }
      const double V = std::max(finite_or_zero(vol[c]), 0.0);
      if (!(V > 0.0)) {
        continue;
      }
      const std::size_t idx = c * n_groups + g;
      const double E_cg = finite_or_zero(E[idx]);
      signed_sum += static_cast<long double>(E_cg) * static_cast<long double>(V);
      if (E_cg > 0.0) {
        positive_sum +=
            static_cast<long double>(E_cg) * static_cast<long double>(V);
      }
    }

    const double signed_group = static_cast<double>(signed_sum);
    const double positive_group = static_cast<double>(positive_sum);
    if (!(positive_group > 0.0)) {
      for (std::size_t c = 0; c < n_cells; ++c) {
        if (diff_cell[c] != 0U) {
          E[c * n_groups + g] = 0.0;
        }
      }
      out.applied = true;
      continue;
    }

    const double kept_group = std::max(signed_group, 0.0);
    const double scale = std::clamp(kept_group / positive_group, 0.0, 1.0);
    if (scale < 1.0) {
      out.removed_positive += positive_group * (1.0 - scale);
    }
    for (std::size_t c = 0; c < n_cells; ++c) {
      if (diff_cell[c] == 0U) {
        continue;
      }
      const std::size_t idx = c * n_groups + g;
      E[idx] = (E[idx] > 0.0) ? (E[idx] * scale) : 0.0;
    }
    out.applied = true;
  }

  out.after = diffusion_energy_moments_host(E, vol, diff_cell, n_cells, n_groups);
  cuda_check(cudaMemcpy(in.diff_E,
                        E.data(),
                        sizeof(double) * E.size(),
                        cudaMemcpyHostToDevice),
             "diffusion limiter copy E failed");
  return out;
}

double total_diffusion_energy_host(const DiffusionStepInputs& in) {
  const std::size_t n_cells = static_cast<std::size_t>(std::max(in.n_cells, 0));
  const std::size_t n_groups = static_cast<std::size_t>(std::max(in.n_groups, 0));
  const std::size_t n_total = n_cells * n_groups;
  const std::vector<double> E =
      copy_device_double_array(in.diff_E, n_total, "diffusion copy E failed");
  const std::vector<double> vol =
      copy_device_double_array(in.vol, n_cells, "diffusion copy volume failed");
  const std::vector<std::uint8_t> diff_cell =
      copy_device_u8_array(in.diff_cell, n_cells, "diffusion copy mask failed");

  long double sum = 0.0L;
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (diff_cell[c] == 0U) {
      continue;
    }
    for (std::size_t g = 0; g < n_groups; ++g) {
      const std::size_t idx = c * n_groups + g;
      sum += static_cast<long double>(E[idx]) *
             static_cast<long double>(std::max(vol[c], 0.0));
    }
  }
  return static_cast<double>(sum);
}

double face_current_source_energy_host(const DiffusionStepInputs& in) {
  if (in.face_current_in == nullptr || in.n_cells <= 0 || in.n_groups <= 0) {
    return 0.0;
  }
  const std::size_t n_cells = static_cast<std::size_t>(in.n_cells);
  const std::size_t n_groups = static_cast<std::size_t>(in.n_groups);
  const std::size_t n_faces = n_cells + 1U;
  const std::vector<double> J =
      copy_device_double_array(in.face_current_in,
                               n_faces * n_groups,
                               "diffusion copy face current failed");
  const std::vector<std::uint8_t> diff_cell =
      copy_device_u8_array(in.diff_cell, n_cells, "diffusion copy mask failed");
  long double sum = 0.0L;
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (diff_cell[c] == 0U) {
      continue;
    }
    for (std::size_t g = 0; g < n_groups; ++g) {
      if (c == 0U || diff_cell[c - 1U] == 0U) {
        sum += static_cast<long double>(
            std::max(finite_or_zero(J[c * n_groups + g]), 0.0));
      }
      if (c + 1U == n_cells || diff_cell[c + 1U] == 0U) {
        sum += static_cast<long double>(
            std::max(finite_or_zero(J[(c + 1U) * n_groups + g]), 0.0));
      }
    }
  }
  return static_cast<double>(sum);
}

double deposit_face_current_in(const DiffusionStepInputs& in) {
  const double source_energy = face_current_source_energy_host(in);
  if (in.face_current_in == nullptr || in.diff_E == nullptr ||
      in.vol == nullptr || in.diff_cell == nullptr ||
      in.n_cells <= 0 || in.n_groups <= 0) {
    return source_energy;
  }

  const int n_total = in.n_cells * in.n_groups;
  const int grid = (n_total + kBlock - 1) / kBlock;
  if (grid <= 0) {
    return source_energy;
  }
  deposit_face_current_in_kernel<<<grid, kBlock>>>(in.diff_E,
                                                   in.face_current_in,
                                                   in.vol,
                                                   in.diff_cell,
                                                   in.n_cells,
                                                   in.n_groups);
  cuda_check(cudaGetLastError(),
             "deterministic_diffusion_step_1d face-current deposit launch failed");
  return source_energy;
}

double diffusion_balance_scale(const double E_before,
                               const double E_face_in,
                               const double E_loss,
                               const double E_after) {
  return std::max({std::fabs(E_before),
                   std::fabs(E_face_in),
                   std::fabs(E_loss),
                   std::fabs(E_after),
                   kEnergyBalanceFloor});
}

void log_diffusion_balance_if_bad(const char* stage,
                                  const double E_before,
                                  const double E_face_in,
                                  const double E_loss,
                                  const double E_after) {
  const double residual = E_before + E_face_in - E_loss - E_after;
  const double scale =
      diffusion_balance_scale(E_before, E_face_in, E_loss, E_after);
  if (std::isfinite(residual) && std::isfinite(scale) &&
      std::fabs(residual) <= kEnergyBalanceRelTol * scale) {
    return;
  }
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(6);
  oss << "[diffusion_balance] critical stage=" << stage
      << " E_before=" << E_before
      << " E_face_in=" << E_face_in
      << " E_loss=" << E_loss
      << " E_after=" << E_after
      << " residual=" << residual
      << " rel=" << (std::fabs(residual) / scale);
  core::log_fatal(oss.str());
}

double estimate_explicit_dt_host(const DiffusionStepInputs& in) {
  const std::size_t n_cells = static_cast<std::size_t>(std::max(in.n_cells, 0));
  const std::size_t n_groups = static_cast<std::size_t>(std::max(in.n_groups, 0));
  if (n_cells == 0U || n_groups == 0U) {
    return std::numeric_limits<double>::infinity();
  }

  const std::size_t n_total = n_cells * n_groups;
  const std::vector<double> sigma =
      copy_device_double_array(in.sigma_R, n_total, "diffusion copy sigma_R failed");
  const std::vector<double> vol =
      copy_device_double_array(in.vol, n_cells, "diffusion copy volume failed");
  const std::vector<double> node_r =
      copy_device_double_array(in.node_r, n_cells + 1U, "diffusion copy node_r failed");
  const std::vector<std::uint8_t> diff_cell =
      copy_device_u8_array(in.diff_cell, n_cells, "diffusion copy mask failed");

  std::vector<double> D(n_total, 0.0);
  for (std::size_t c = 0; c < n_cells; ++c) {
    const double dx =
        std::max(finite_or_zero(node_r[c + 1U]) - finite_or_zero(node_r[c]), 0.0);
    const double D_max = tenryu::core::constants::c_light * dx / 6.0;
    for (std::size_t g = 0; g < n_groups; ++g) {
      const std::size_t idx = c * n_groups + g;
      const double sigma_g = std::max(finite_or_zero(sigma[idx]), kSigmaFloor);
      const double D_raw = tenryu::core::constants::c_light / (3.0 * sigma_g);
      D[idx] = (D_max > 0.0) ? std::min(D_raw, D_max) : 0.0;
    }
  }

  double dt_min = std::numeric_limits<double>::infinity();
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (diff_cell[c] == 0U || !(vol[c] > 0.0)) {
      continue;
    }
    const double r_left = node_r[c];
    const double r_right = node_r[c + 1U];
    const double r_c = 0.5 * (r_left + r_right);
    const double A_left = kFourPi * r_left * r_left;
    const double A_right = kFourPi * r_right * r_right;
    for (std::size_t g = 0; g < n_groups; ++g) {
      const std::size_t idx = c * n_groups + g;
      double denom = 0.0;
      if (c > 0U && diff_cell[c - 1U] != 0U) {
        const double r_c_left = 0.5 * (node_r[c - 1U] + r_left);
        const double dr = r_c - r_c_left;
        const double D_face =
            harmonic_face_d(D[idx], D[(c - 1U) * n_groups + g]);
        if (dr > 0.0 && D_face > 0.0) {
          denom += A_left * D_face / dr;
        }
      } else if (c == 0U && in.bc_inner == 1) {
        denom += A_left * 0.25 * tenryu::core::constants::c_light;
      }

      if (c + 1U < n_cells && diff_cell[c + 1U] != 0U) {
        const double r_c_right = 0.5 * (r_right + node_r[c + 2U]);
        const double dr = r_c_right - r_c;
        const double D_face =
            harmonic_face_d(D[idx], D[(c + 1U) * n_groups + g]);
        if (dr > 0.0 && D_face > 0.0) {
          denom += A_right * D_face / dr;
        }
      } else if (c + 1U == n_cells && in.bc_outer == 1) {
        denom += A_right * 0.25 * tenryu::core::constants::c_light;
      }

      if (denom > 0.0 && std::isfinite(denom)) {
        dt_min = std::min(dt_min, vol[c] / denom);
      }
    }
  }
  return dt_min;
}

DiffusionStepPlan make_diffusion_step_plan(const DiffusionStepInputs& in,
                                           const int sts_max_stages,
                                           const double safety) {
  DiffusionStepPlan plan{};
  if (in.n_cells <= 0 || in.n_groups <= 0 || !(in.dt > 0.0)) {
    return plan;
  }

  const double dt_explicit = estimate_explicit_dt_host(in);
  plan.dt_explicit = dt_explicit;
  if (!(dt_explicit > 0.0) || !std::isfinite(dt_explicit)) {
    return plan;
  }

  const double eta = (std::isfinite(safety) && safety > 0.0) ? safety : 0.8;
  const int cap = (sts_max_stages > 0) ? std::max(sts_max_stages, 2)
                                       : std::numeric_limits<int>::max();
  int n_sub =
      tenryu::numerics::estimate_rkl2_subcycles(in.dt, dt_explicit, sts_max_stages, eta);
  n_sub = std::max(n_sub, 1);
  plan.rkl2_subcycles = n_sub;
  if (sts_max_stages > 0 && n_sub > kMaxRKL2Subcycles) {
    plan.rkl2_stages = cap;
    plan.skip = true;
    return plan;
  }

  const double dt_sub = in.dt / static_cast<double>(n_sub);
  plan.rkl2_stages = std::min(
      tenryu::numerics::estimate_rkl2_stages(dt_sub, dt_explicit, eta), cap);
  return plan;
}

DiffusionStepResult run_single_rkl2_step(
    const DiffusionStepInputs& in,
    const tenryu::numerics::RKL2Coefficients& coeff) {
  DiffusionStepResult out{};
  out.rkl2_stages = coeff.s;
  out.rkl2_subcycles = 1;
  if (in.n_cells <= 0 || in.n_groups <= 0 || !(in.dt > 0.0)) {
    return out;
  }
  TENRYU_ASSERT(coeff.s >= 2, "deterministic_diffusion_step_1d requires s >= 2");
  TENRYU_ASSERT(static_cast<int>(coeff.mu.size()) > coeff.s &&
                    static_cast<int>(coeff.nu.size()) > coeff.s &&
                    static_cast<int>(coeff.mu_tilde.size()) > coeff.s &&
                    static_cast<int>(coeff.gamma_tilde.size()) > coeff.s,
                "deterministic_diffusion_step_1d coefficient size mismatch");
  TENRYU_ASSERT(in.diff_E != nullptr, "deterministic_diffusion_step_1d requires diff_E");
  TENRYU_ASSERT(in.sigma_R != nullptr, "deterministic_diffusion_step_1d requires sigma_R");
  TENRYU_ASSERT(in.vol != nullptr, "deterministic_diffusion_step_1d requires vol");
  TENRYU_ASSERT(in.node_r != nullptr, "deterministic_diffusion_step_1d requires node_r");
  TENRYU_ASSERT(in.diff_cell != nullptr, "deterministic_diffusion_step_1d requires diff_cell");

  out.E_before = total_diffusion_energy_host(in);
  out.E_positive_before = out.E_before;

  const int n_total = in.n_cells * in.n_groups;
  const int grid = (n_total + kBlock - 1) / kBlock;
  if (grid <= 0) {
    out.E_after = out.E_before;
    out.E_positive_after = out.E_positive_before;
    return out;
  }
  const double source_energy = deposit_face_current_in(in);

  double* D_cell = nullptr;
  double* Y0 = nullptr;
  double* Y1 = nullptr;
  double* Y2 = nullptr;
  double* L0 = nullptr;
  double* Lj = nullptr;
  const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(n_total);
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&D_cell), bytes),
             "deterministic_diffusion_step_1d cudaMalloc D_cell failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&Y0), bytes),
             "deterministic_diffusion_step_1d cudaMalloc Y0 failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&Y1), bytes),
             "deterministic_diffusion_step_1d cudaMalloc Y1 failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&Y2), bytes),
             "deterministic_diffusion_step_1d cudaMalloc Y2 failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&L0), bytes),
             "deterministic_diffusion_step_1d cudaMalloc L0 failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&Lj), bytes),
             "deterministic_diffusion_step_1d cudaMalloc Lj failed");

  compute_d_cell_kernel<<<grid, kBlock>>>(in.sigma_R,
                                          in.node_r,
                                          in.diff_cell,
                                          D_cell,
                                          in.n_cells,
                                          in.n_groups);
  cuda_check(cudaGetLastError(), "deterministic_diffusion_step_1d D kernel launch failed");
  copy_kernel<<<grid, kBlock>>>(in.diff_E, Y0, n_total);
  cuda_check(cudaGetLastError(), "deterministic_diffusion_step_1d copy Y0 launch failed");
  diffusion_operator_1d_kernel<<<grid, kBlock>>>(Y0,
                                                 L0,
                                                 D_cell,
                                                 in.vol,
                                                 in.node_r,
                                                 in.diff_cell,
                                                 in.n_cells,
                                                 in.n_groups,
                                                 in.bc_inner,
                                                 in.bc_outer);
  cuda_check(cudaGetLastError(), "deterministic_diffusion_step_1d L0 launch failed");
  stage1_kernel<<<grid, kBlock>>>(Y0,
                                  L0,
                                  Y1,
                                  n_total,
                                  in.dt,
                                  coeff.mu_tilde[1]);
  cuda_check(cudaGetLastError(), "deterministic_diffusion_step_1d stage1 launch failed");

  const double* Yjm2 = Y0;
  const double* Yjm1 = Y1;
  double* Yj = Y2;
  const double* final_stage = Y1;
  for (int j = 2; j <= coeff.s; ++j) {
    diffusion_operator_1d_kernel<<<grid, kBlock>>>(Yjm1,
                                                   Lj,
                                                   D_cell,
                                                   in.vol,
                                                   in.node_r,
                                                   in.diff_cell,
                                                   in.n_cells,
                                                   in.n_groups,
                                                   in.bc_inner,
                                                   in.bc_outer);
    cuda_check(cudaGetLastError(), "deterministic_diffusion_step_1d Lj launch failed");
    const std::size_t ju = static_cast<std::size_t>(j);
    stagej_kernel<<<grid, kBlock>>>(Y0,
                                    Yjm1,
                                    Yjm2,
                                    L0,
                                    Lj,
                                    Yj,
                                    n_total,
                                    in.dt,
                                    coeff.mu[ju],
                                    coeff.nu[ju],
                                    coeff.mu_tilde[ju],
                                    coeff.gamma_tilde[ju]);
    cuda_check(cudaGetLastError(), "deterministic_diffusion_step_1d stagej launch failed");
    final_stage = Yj;
    Yjm2 = Yjm1;
    Yjm1 = Yj;
    Yj = (Yj == Y1) ? Y2 : Y1;
  }

  copy_kernel<<<grid, kBlock>>>(final_stage, in.diff_E, n_total);
  cuda_check(cudaGetLastError(), "deterministic_diffusion_step_1d final copy launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "deterministic_diffusion_step_1d kernel execution failed");

  cuda_check(cudaFree(Lj), "deterministic_diffusion_step_1d cudaFree Lj failed");
  cuda_check(cudaFree(L0), "deterministic_diffusion_step_1d cudaFree L0 failed");
  cuda_check(cudaFree(Y2), "deterministic_diffusion_step_1d cudaFree Y2 failed");
  cuda_check(cudaFree(Y1), "deterministic_diffusion_step_1d cudaFree Y1 failed");
  cuda_check(cudaFree(Y0), "deterministic_diffusion_step_1d cudaFree Y0 failed");
  cuda_check(cudaFree(D_cell), "deterministic_diffusion_step_1d cudaFree D_cell failed");

  const PositivityLimiterResult limiter =
      apply_conservative_positivity_limiter(in);
  out.positivity_limited = limiter.applied;
  out.E_positive_before = limiter.before.positive_total;
  out.E_negative_before = limiter.before.negative_total;
  out.E_positive_after = limiter.after.positive_total;
  out.E_negative_after = limiter.after.negative_total;
  out.E_limiter_removed = limiter.removed_positive;
  out.E_after = limiter.after.signed_total;
  if (limiter.applied) {
    std::ostringstream limiter_log;
    limiter_log << std::scientific << std::setprecision(6);
    limiter_log << "[diffusion_limiter] stage=rkl2"
                << " E_signed_before=" << limiter.before.signed_total
                << " E_positive_before=" << limiter.before.positive_total
                << " E_negative_before=" << limiter.before.negative_total
                << " E_signed_after=" << limiter.after.signed_total
                << " E_positive_after=" << limiter.after.positive_total
                << " E_negative_after=" << limiter.after.negative_total
                << " E_removed_positive=" << limiter.removed_positive;
    core::log_warning(limiter_log.str());
  }
  out.E_leaked = std::max(0.0, out.E_before + source_energy - out.E_after);
  log_diffusion_balance_if_bad("rkl2",
                               out.E_before,
                               source_energy,
                               out.E_leaked,
                               out.E_after);
  return out;
}

}  // namespace

DiffusionStepResult deterministic_diffusion_step_1d(
    const DiffusionStepInputs& in,
    const tenryu::numerics::RKL2Coefficients& coeff) {
  return run_single_rkl2_step(in, coeff);
}

DiffusionStepPlan deterministic_diffusion_plan_1d(const DiffusionStepInputs& in,
                                                  const int sts_max_stages,
                                                  const double safety) {
  return make_diffusion_step_plan(in, sts_max_stages, safety);
}

DiffusionStepResult deterministic_diffusion_step_1d(const DiffusionStepInputs& in,
                                                    const int sts_max_stages,
                                                    const double damping,
                                                    const double safety) {
  DiffusionStepResult total{};
  if (in.n_cells <= 0 || in.n_groups <= 0 || !(in.dt > 0.0)) {
    return total;
  }

  const double E_start = total_diffusion_energy_host(in);
  const double source_energy = face_current_source_energy_host(in);
  const DiffusionStepPlan plan =
      make_diffusion_step_plan(in, sts_max_stages, safety);
  total.dt_explicit = plan.dt_explicit;
  total.rkl2_stages = plan.rkl2_stages;
  total.rkl2_subcycles = plan.rkl2_subcycles;
  total.rkl2_skipped = plan.skip;
  if (!(plan.dt_explicit > 0.0) || !std::isfinite(plan.dt_explicit) ||
      plan.rkl2_stages < 2 || plan.rkl2_subcycles < 1) {
    const double deposited_energy = deposit_face_current_in(in);
    total.E_before = E_start;
    total.E_after = total_diffusion_energy_host(in);
    total.E_positive_before = E_start;
    total.E_positive_after = total.E_after;
    total.E_leaked =
        std::max(0.0, total.E_before + deposited_energy - total.E_after);
    return total;
  }
  if (plan.skip) {
    const double deposited_energy = deposit_face_current_in(in);
    total.E_before = E_start;
    total.E_after = total_diffusion_energy_host(in);
    total.E_positive_before = E_start;
    total.E_positive_after = total.E_after;
    total.E_leaked =
        std::max(0.0, total.E_before + deposited_energy - total.E_after);
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(6);
    oss << "[rkl2] skipping diffusion step: stages=" << plan.rkl2_stages
        << " subcycles=" << plan.rkl2_subcycles
        << " max_subcycles=" << kMaxRKL2Subcycles
        << " dt_explicit=" << plan.dt_explicit
        << " dt=" << in.dt;
    core::log_warning(oss.str());
    return total;
  }

  std::ostringstream oss;
  oss << std::scientific << std::setprecision(6);
  oss << "[rkl2] stages=" << plan.rkl2_stages
      << " subcycles=" << plan.rkl2_subcycles
      << " dt_explicit=" << plan.dt_explicit
      << " dt=" << in.dt;
  core::log_info(oss.str());

  const double dt_sub = in.dt / static_cast<double>(plan.rkl2_subcycles);
  const tenryu::numerics::RKL2Coefficients coeff =
      tenryu::numerics::compute_rkl2_coefficients(plan.rkl2_stages, damping);

  total.E_before = E_start;
  total.E_positive_before = E_start;
  for (int sub = 0; sub < plan.rkl2_subcycles; ++sub) {
    DiffusionStepInputs sub_in = in;
    sub_in.dt = dt_sub;
    const DiffusionStepResult step = run_single_rkl2_step(sub_in, coeff);
    total.rkl2_stages = std::max(total.rkl2_stages, step.rkl2_stages);
    total.E_leaked += step.E_leaked;
    total.E_limiter_removed += step.E_limiter_removed;
    total.E_negative_before = std::max(total.E_negative_before,
                                       step.E_negative_before);
    total.E_negative_after = step.E_negative_after;
    total.E_positive_after = step.E_positive_after;
    total.positivity_limited =
        total.positivity_limited || step.positivity_limited;
  }
  total.E_after = total_diffusion_energy_host(in);
  if (!(total.E_positive_after > 0.0) && total.E_after > 0.0) {
    total.E_positive_after = total.E_after;
  }
  log_diffusion_balance_if_bad("rkl2_total",
                               total.E_before,
                               source_energy,
                               total.E_leaked,
                               total.E_after);
  return total;
}

}  // namespace tenryu::radiation
