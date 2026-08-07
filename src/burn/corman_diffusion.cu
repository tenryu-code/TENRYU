#include "burn/corman_diffusion.cuh"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include <cusparse.h>

#include "core/error.hpp"

namespace tenryu::burn {
namespace {

constexpr int kBlock = 128;
constexpr int kTotals = 5;
constexpr int kTotalEscaped = 0;
constexpr int kTotalInflight = 1;
constexpr int kTotalSourced = 2;
constexpr int kTotalDepE = 3;
constexpr int kTotalDepI = 4;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline void cusparse_check(const cusparseStatus_t status, const char* message) {
  TENRYU_ASSERT(status == CUSPARSE_STATUS_SUCCESS, message);
}

struct CormanCusparseCache {
  cusparseHandle_t handle = nullptr;

  ~CormanCusparseCache() {
    if (handle != nullptr) {
      static_cast<void>(cusparseDestroy(handle));
      handle = nullptr;
    }
  }
};

CormanCusparseCache& corman_cusparse_cache() {
  static CormanCusparseCache cache;
  return cache;
}

std::size_t align_up_size(const std::size_t value, const std::size_t align) {
  return (value + align - 1U) & ~(align - 1U);
}

std::uintptr_t align_up_ptr(const std::uintptr_t value, const std::size_t align) {
  return (value + align - 1U) & ~(static_cast<std::uintptr_t>(align) - 1U);
}

std::size_t conservative_cusparse_bytes(const int n_cells) {
  const std::size_t rows = static_cast<std::size_t>(std::max(n_cells, 1));
  const std::size_t row_bytes = 32U * rows * sizeof(double);
  return std::max<std::size_t>(1U << 20U, row_bytes + 131072U);
}

struct ScratchLayout {
  double* lower = nullptr;
  double* diag = nullptr;
  double* upper = nullptr;
  double* rhs = nullptr;
  double* N_old = nullptr;
  double* tE = nullptr;
  double* gamma = nullptr;
  double* lnL_I = nullptr;
  double* totals = nullptr;
  void* cusparse_buffer = nullptr;
  std::size_t cusparse_bytes = 0U;
};

ScratchLayout carve_scratch(void* scratch_pool, const std::size_t scratch_bytes,
                            const int n_groups, const int n_cells) {
  TENRYU_ASSERT(scratch_pool != nullptr, "Corman diffusion scratch_pool is null");
  const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(scratch_pool);
  std::uintptr_t cur = base;
  const std::uintptr_t end = base + scratch_bytes;

  auto take_doubles = [&](const std::size_t count) -> double* {
    cur = align_up_ptr(cur, alignof(double));
    double* ptr = reinterpret_cast<double*>(cur);
    cur += count * sizeof(double);
    TENRYU_ASSERT(cur <= end, "Corman diffusion scratch too small");
    return ptr;
  };

  ScratchLayout s;
  const std::size_t cells = static_cast<std::size_t>(n_cells);
  const std::size_t total =
      static_cast<std::size_t>(n_groups) * static_cast<std::size_t>(n_cells);
  s.lower = take_doubles(cells);
  s.diag = take_doubles(cells);
  s.upper = take_doubles(cells);
  s.rhs = take_doubles(cells);
  s.N_old = take_doubles(total);
  s.tE = take_doubles(cells);
  s.gamma = take_doubles(total);
  s.lnL_I = take_doubles(total);
  s.totals = take_doubles(kTotals);
  cur = align_up_ptr(cur, 128U);
  TENRYU_ASSERT(cur <= end, "Corman diffusion scratch too small");
  s.cusparse_buffer = reinterpret_cast<void*>(cur);
  s.cusparse_bytes = static_cast<std::size_t>(end - cur);
  return s;
}

__host__ __device__ inline double edge_keV(const CormanParams& p,
                                           const int k) {
  const double x = static_cast<double>(k) / static_cast<double>(p.n_groups);
  return p.E_min_keV * exp(log(p.E_max_keV / p.E_min_keV) * x);
}

__host__ __device__ inline double center_keV(const CormanParams& p,
                                             const int g) {
  return 0.5 * (edge_keV(p, g) + edge_keV(p, g + 1));
}

__host__ __device__ inline double cell_G(const double* __restrict__ r_node,
                                         const int j) {
  const double r0 = r_node[j];
  const double r1 = r_node[j + 1];
  return (r1 * r1 * r1 - r0 * r0 * r0) / 3.0;
}

__host__ __device__ inline double tau_group(const CormanParams& p, const int g,
                                            const double tE,
                                            const double gamma) {
  if (!(tE > 0.0) || !isfinite(tE)) {
    // Electron drag invalid: pure ion drag F = gamma/sqrt(E), so
    // tau = (2/3) (E1^{3/2} - E0^{3/2}) / gamma.
    if (gamma > 0.0 && isfinite(gamma)) {
      const double E0i = edge_keV(p, g) * corman_detail::kKeVToErg;
      const double E1i = edge_keV(p, g + 1) * corman_detail::kKeVToErg;
      if (E1i > E0i && E0i >= 0.0) {
        return (2.0 / 3.0) * (E1i * sqrt(E1i) - E0i * sqrt(E0i)) / gamma;
      }
    }
    return INFINITY;
  }
  const double E0 = edge_keV(p, g) * corman_detail::kKeVToErg;
  const double E1 = edge_keV(p, g + 1) * corman_detail::kKeVToErg;
  const double gamma_tE = gamma * tE;
  const double lo = gamma_tE + E0 * sqrt(E0);
  const double hi = gamma_tE + E1 * sqrt(E1);
  if (!(lo > 0.0) || !(hi > lo)) {
    return INFINITY;
  }
  return tE * (2.0 / 3.0) * log(hi / lo);
}

__host__ __device__ inline double ion_fraction_group(const CormanParams& p,
                                                     const int g,
                                                     const double tE,
                                                     const double gamma) {
  if (!(gamma > 0.0) || !isfinite(gamma)) {
    return 0.0;  // no ion drag: everything goes to electrons
  }
  if (!(tE > 0.0) || !isfinite(tE)) {
    return 1.0;  // no electron drag: everything goes to ions
  }
  const double E0 = edge_keV(p, g) * corman_detail::kKeVToErg;
  const double E1 = edge_keV(p, g + 1) * corman_detail::kKeVToErg;
  constexpr int n = 8;
  const double h = (E1 - E0) / static_cast<double>(n);
  double num = 0.0;
  for (int i = 0; i <= n; ++i) {
    const double E = E0 + h * static_cast<double>(i);
    const double sqrtE = sqrt(E);
    const double ion = gamma / sqrtE;
    const double F = E / tE + ion;
    const double w = ion / F;
    const double weight = (i == 0 || i == n) ? 1.0 : ((i % 2 == 0) ? 2.0 : 4.0);
    num += weight * w;
  }
  const double dE_i = num * (h / 3.0);
  const double f = dE_i / (E1 - E0);
  return (f < 0.0) ? 0.0 : ((f > 1.0) ? 1.0 : f);
}

__host__ __device__ inline double face_diffusion(
    const CormanParams& p, const int g, const int face,
    const double species_A, const double species_Z,
    const double* __restrict__ r_node, const double* __restrict__ rho,
    const double* __restrict__ N_old, const double* __restrict__ lnL_I,
    const int n_cells) {
  const double E = center_keV(p, g);
  int jl = face - 1;
  int jr = face;
  if (face <= 0) {
    jl = 0;
    jr = (n_cells > 1) ? 1 : 0;
  } else if (face >= n_cells) {
    jl = (n_cells > 1) ? n_cells - 2 : 0;
    jr = n_cells - 1;
  }
  const double Nl = N_old[g * n_cells + jl];
  const double Nr = N_old[g * n_cells + jr];
  double N_face = 0.5 * (Nl + Nr);
  if (jl == jr) {
    N_face = N_old[g * n_cells + jl];
  }
  const double rho_face = (jl == jr) ? rho[jl] : 0.5 * (rho[jl] + rho[jr]);
  const double lnL_face =
      (jl == jr) ? lnL_I[g * n_cells + jl]
                 : 0.5 * (lnL_I[g * n_cells + jl] +
                          lnL_I[g * n_cells + jr]);
  const double lambda =
      corman_lambda(species_A, species_Z, E, rho_face, lnL_face);
  const double m_s = species_A * corman_detail::kProtonMassG;
  const double v = sqrt(2.0 * E * corman_detail::kKeVToErg / m_s);
  if (!(lambda > 0.0) || !isfinite(lambda)) {
    return 0.0;
  }
  if (!(N_face > 1.0e-30)) {
    return v * lambda / 3.0;
  }

  const double xl = 0.5 * (r_node[jl] + r_node[jl + 1]);
  const double xr = 0.5 * (r_node[jr] + r_node[jr + 1]);
  const double dist = fmax(xr - xl, 1.0e-300);
  const double grad = (jr == jl) ? 0.0 : (Nr - Nl) / dist;
  const double r_face = fmax(r_node[face], 1.0e-300);
  const double inv_mubar =
      1.0 + 3.0 * exp(-0.5 * lambda *
                      fabs(grad / N_face - 3.6 / r_face));
  const double denom = 3.0 / lambda + fabs(grad) * inv_mubar / N_face;
  if (!(denom > 0.0)) {
    return v * lambda / 3.0;
  }
  return v / denom;
}

__host__ __device__ inline double outer_sink_coeff(
    const CormanParams& p, const int g, const double species_A,
    const double species_Z, const double* __restrict__ r_node,
    const double* __restrict__ rho, const double* __restrict__ N_old,
    const double* __restrict__ lnL_I, const int n_cells, const double dt_s) {
  const int j = n_cells - 1;
  const double D = face_diffusion(p, g, n_cells, species_A, species_Z, r_node,
                                  rho, N_old, lnL_I, n_cells);
  const double lambda = corman_lambda(species_A, species_Z, center_keV(p, g),
                                      rho[j], lnL_I[g * n_cells + j]);
  if (!(D > 0.0) || !(lambda > 0.0) || !isfinite(lambda)) {
    return 0.0;
  }
  const double r_outer = r_node[n_cells];
  const double inv_L = 1.0 / (0.71 * lambda) + 1.0 / fmax(r_outer, 1.0e-300);
  const double L = 1.0 / inv_L;
  const double dr = fmax(r_node[n_cells] - r_node[n_cells - 1], 1.0e-300);
  const double G = fmax(cell_G(r_node, j), 1.0e-300);
  return dt_s * r_outer * r_outer * D / ((0.5 * dr + L) * G);
}

struct BirthBinning {
  int lo = 0;
  int hi = -1;
  double w_lo = 1.0;
  double w_hi = 0.0;
  double top_excess_erg = 0.0;
};

BirthBinning birth_binning(const CormanParams& p, const double E_birth_keV) {
  BirthBinning b;
  const double c0 = center_keV(p, 0);
  const double ctop = center_keV(p, p.n_groups - 1);
  if (E_birth_keV <= c0) {
    b.lo = 0;
    b.hi = -1;
    b.w_lo = 1.0;
    return b;
  }
  if (E_birth_keV >= ctop) {
    b.lo = p.n_groups - 1;
    b.hi = -1;
    b.w_lo = 1.0;
    b.top_excess_erg =
        (E_birth_keV - ctop) * corman_detail::kKeVToErg;
    return b;
  }
  for (int g = 0; g + 1 < p.n_groups; ++g) {
    const double clo = center_keV(p, g);
    const double chi = center_keV(p, g + 1);
    if (E_birth_keV >= clo && E_birth_keV <= chi) {
      b.lo = g;
      b.hi = g + 1;
      b.w_hi = (E_birth_keV - clo) / (chi - clo);
      b.w_lo = 1.0 - b.w_hi;
      return b;
    }
  }
  return b;
}

__host__ __device__ inline double source_weight(const BirthBinning b,
                                                const int g) {
  double w = 0.0;
  if (g == b.lo) {
    w += b.w_lo;
  }
  if (g == b.hi) {
    w += b.w_hi;
  }
  return w;
}

__global__ void compute_coefficients_kernel(
    CormanParams p, int n_cells, double species_A, double species_Z,
    const double* __restrict__ rho, const double* __restrict__ Te_eV,
    const double* __restrict__ Ti_eV, const double* __restrict__ ne,
    double* __restrict__ tE, double* __restrict__ gamma,
    double* __restrict__ lnL_I) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n_cells) {
    return;
  }
  const double lnLe = (p.lnL_e > 0.0) ? p.lnL_e
                                      : corman_electron_log(Te_eV[j], ne[j]);
  tE[j] = corman_tE(species_A, species_Z, Te_eV[j], ne[j], lnLe);
  for (int g = 0; g < p.n_groups; ++g) {
    const double lnLI =
        (p.lnL_I > 0.0)
            ? p.lnL_I
            : corman_ion_log(species_A, species_Z, center_keV(p, g),
                             rho[j], Te_eV[j], Ti_eV[j], ne[j]);
    gamma[g * n_cells + j] =
        corman_gamma(species_A, species_Z, rho[j], lnLI);
    lnL_I[g * n_cells + j] = lnLI;
  }
}

__global__ void source_tally_kernel(
    CormanParams p, BirthBinning birth, int n_cells, double E_birth_keV,
    const double* __restrict__ vol, const double* __restrict__ S_birth,
    double dt_s, double* __restrict__ dep_e, double* __restrict__ totals) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  (void)p;
  const double E_birth_erg = E_birth_keV * corman_detail::kKeVToErg;
  double sourced = 0.0;
  double dep_e_step = 0.0;
  for (int j = 0; j < n_cells; ++j) {
    const double n_birth = S_birth[j] * dt_s * vol[j];
    sourced += n_birth * E_birth_erg;
    if (birth.top_excess_erg > 0.0) {
      const double e = n_birth * birth.top_excess_erg;
      dep_e[j] += e;
      dep_e_step += e;
    }
  }
  totals[kTotalSourced] += sourced;
  totals[kTotalDepE] += dep_e_step;
}

__global__ void assemble_group_kernel(
    CormanParams p, BirthBinning birth, int g, int n_cells, double species_A,
    double species_Z, const double* __restrict__ r_node,
    const double* __restrict__ rho, const double* __restrict__ S_birth,
    const double* __restrict__ N_old, const double* __restrict__ N,
    const double* __restrict__ tE, const double* __restrict__ gamma,
    const double* __restrict__ lnL_I, double dt_s, double* __restrict__ lower,
    double* __restrict__ diag, double* __restrict__ upper,
    double* __restrict__ rhs) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n_cells) {
    return;
  }
  const double G = fmax(cell_G(r_node, j), 1.0e-300);
  const double tau = tau_group(p, g, tE[j], gamma[g * n_cells + j]);
  const double sink = (tau > 0.0 && isfinite(tau)) ? dt_s / tau : 0.0;
  double d = 1.0 + sink;
  double l = 0.0;
  double u = 0.0;

  if (j > 0) {
    const double D = face_diffusion(p, g, j, species_A, species_Z, r_node, rho,
                                    N_old, lnL_I, n_cells);
    const double xl = 0.5 * (r_node[j - 1] + r_node[j]);
    const double xc = 0.5 * (r_node[j] + r_node[j + 1]);
    const double dist = fmax(xc - xl, 1.0e-300);
    const double w = r_node[j] * r_node[j] * D / dist;
    const double a = dt_s * w / G;
    d += a;
    l = -a;
  }
  if (j + 1 < n_cells) {
    const double D = face_diffusion(p, g, j + 1, species_A, species_Z, r_node,
                                    rho, N_old, lnL_I, n_cells);
    const double xc = 0.5 * (r_node[j] + r_node[j + 1]);
    const double xr = 0.5 * (r_node[j + 1] + r_node[j + 2]);
    const double dist = fmax(xr - xc, 1.0e-300);
    const double w = r_node[j + 1] * r_node[j + 1] * D / dist;
    const double c = dt_s * w / G;
    d += c;
    u = -c;
  } else {
    d += outer_sink_coeff(p, g, species_A, species_Z, r_node, rho, N_old,
                          lnL_I, n_cells, dt_s);
  }

  double b = N_old[g * n_cells + j] + dt_s * S_birth[j] * source_weight(birth, g);
  if (g + 1 < p.n_groups) {
    const double tau_up =
        tau_group(p, g + 1, tE[j], gamma[(g + 1) * n_cells + j]);
    if (tau_up > 0.0 && isfinite(tau_up)) {
      b += dt_s * N[(g + 1) * n_cells + j] / tau_up;
    }
  }
  lower[j] = l;
  diag[j] = d;
  upper[j] = u;
  rhs[j] = b;
}

__global__ void publish_group_kernel(int g, int n_cells,
                                     const double* __restrict__ rhs,
                                     double* __restrict__ N) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n_cells) {
    return;
  }
  N[g * n_cells + j] = rhs[j];
}

__global__ void tally_group_kernel(
    CormanParams p, int g, int n_cells, double species_A, double species_Z,
    const double* __restrict__ r_node, const double* __restrict__ vol,
    const double* __restrict__ rho, const double* __restrict__ N_old,
    const double* __restrict__ N_group, const double* __restrict__ tE,
    const double* __restrict__ gamma, const double* __restrict__ lnL_I,
    double dt_s, double* __restrict__ dep_e, double* __restrict__ dep_i,
    double* __restrict__ totals) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const double E_g = center_keV(p, g) * corman_detail::kKeVToErg;
  const double E_lower =
      (g > 0) ? center_keV(p, g - 1) * corman_detail::kKeVToErg : 0.0;
  double dep_e_step = 0.0;
  double dep_i_step = 0.0;
  double escaped = 0.0;
  for (int j = 0; j < n_cells; ++j) {
    const double Nj = N_group[j];
    const double tau = tau_group(p, g, tE[j], gamma[g * n_cells + j]);
    const double transfer_coeff =
        (tau > 0.0 && isfinite(tau)) ? dt_s / tau : 0.0;
    const double n_transfer = Nj * transfer_coeff * vol[j];
    if (g > 0) {
      const double dE = E_g - E_lower;
      const double fi =
          ion_fraction_group(p, g, tE[j], gamma[g * n_cells + j]);
      const double e_i = n_transfer * dE * fi;
      const double e_e = n_transfer * dE * (1.0 - fi);
      dep_e[j] += e_e;
      dep_i[j] += e_i;
      dep_e_step += e_e;
      dep_i_step += e_i;
    } else {
      const double e_i = n_transfer * E_g;
      dep_i[j] += e_i;
      dep_i_step += e_i;
    }
  }
  const int outer = n_cells - 1;
  const double sink = outer_sink_coeff(p, g, species_A, species_Z, r_node, rho,
                                       N_old, lnL_I, n_cells, dt_s);
  escaped += N_group[outer] * sink * vol[outer] * E_g;
  totals[kTotalEscaped] += escaped;
  totals[kTotalDepE] += dep_e_step;
  totals[kTotalDepI] += dep_i_step;
}

__global__ void inflight_kernel(CormanParams p, int n_cells,
                                const double* __restrict__ vol,
                                const double* __restrict__ N,
                                double* __restrict__ totals) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  double inflight = 0.0;
  for (int g = 0; g < p.n_groups; ++g) {
    const double E = center_keV(p, g) * corman_detail::kKeVToErg;
    for (int j = 0; j < n_cells; ++j) {
      inflight += N[g * n_cells + j] * E * vol[j];
    }
  }
  totals[kTotalInflight] = inflight;
}

__global__ void scale_rows_by_cell_kernel(double* __restrict__ dst,
                                          const double* __restrict__ src,
                                          const double* __restrict__ rho,
                                          int G, int J, bool divide) {
  const std::size_t total =
      static_cast<std::size_t>(G) * static_cast<std::size_t>(J);
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int j = static_cast<int>(idx % static_cast<std::size_t>(J));
  const double rho_j = rho[j];
  if (divide) {
    dst[idx] = (rho_j > 0.0) ? src[idx] / rho_j : 0.0;
  } else {
    dst[idx] = src[idx] * rho_j;
  }
}

}  // namespace

std::size_t corman_diffusion_scratch_bytes(const int n_groups,
                                           const int n_cells) {
  if (n_groups <= 0 || n_cells <= 0) {
    return 0U;
  }
  const std::size_t cells = static_cast<std::size_t>(n_cells);
  const std::size_t total =
      static_cast<std::size_t>(n_groups) * static_cast<std::size_t>(n_cells);
  std::size_t bytes = 0U;
  auto add_doubles = [&](const std::size_t count) {
    bytes = align_up_size(bytes, alignof(double));
    bytes += count * sizeof(double);
  };
  add_doubles(cells);
  add_doubles(cells);
  add_doubles(cells);
  add_doubles(cells);
  add_doubles(total);
  add_doubles(cells);
  add_doubles(total);
  add_doubles(total);
  add_doubles(kTotals);
  bytes = align_up_size(bytes, 128U);
  bytes += conservative_cusparse_bytes(n_cells);
  return bytes + 256U;
}

void scale_rows_by_cell(double* dst, const double* src, const double* rho,
                        const int G, const int J) {
  if (G <= 0 || J <= 0) {
    return;
  }
  TENRYU_ASSERT(dst != nullptr && src != nullptr && rho != nullptr,
                "Corman diffusion scale_rows_by_cell received a null pointer");
  const std::size_t total =
      static_cast<std::size_t>(G) * static_cast<std::size_t>(J);
  const int grid = static_cast<int>((total + kBlock - 1U) / kBlock);
  scale_rows_by_cell_kernel<<<grid, kBlock>>>(dst, src, rho, G, J, false);
  cuda_check(cudaGetLastError(),
             "Corman diffusion scale_rows_by_cell kernel launch failed");
}

void divide_rows_by_cell(double* dst, const double* src, const double* rho,
                         const int G, const int J) {
  if (G <= 0 || J <= 0) {
    return;
  }
  TENRYU_ASSERT(dst != nullptr && src != nullptr && rho != nullptr,
                "Corman diffusion divide_rows_by_cell received a null pointer");
  const std::size_t total =
      static_cast<std::size_t>(G) * static_cast<std::size_t>(J);
  const int grid = static_cast<int>((total + kBlock - 1U) / kBlock);
  scale_rows_by_cell_kernel<<<grid, kBlock>>>(dst, src, rho, G, J, true);
  cuda_check(cudaGetLastError(),
             "Corman diffusion divide_rows_by_cell kernel launch failed");
}

CormanStepResult corman_diffusion_step(
    const CormanParams& p_in, const int n_cells, const double species_A,
    const double species_Z, const double E_birth_keV,
    const double* r_node_dev, const double* vol_dev, const double* rho_dev,
    const double* Te_eV_dev, const double* Ti_eV_dev, const double* ne_dev,
    const double* S_birth_dev, const double dt_s, double* N_dev,
    double* dep_e_dev, double* dep_i_dev, void* scratch_pool,
    const std::size_t scratch_bytes, cudaStream_t stream) {
  CormanStepResult result;
  if (n_cells <= 0) {
    return result;
  }
  CormanParams p = p_in;
  TENRYU_ASSERT(n_cells >= 3,
                "Corman diffusion requires n_cells >= 3 (cusparse gtsv2 contract)");
  TENRYU_ASSERT(p.n_groups > 0, "Corman diffusion n_groups must be positive");
  TENRYU_ASSERT(p.E_min_keV > 0.0 && p.E_max_keV > p.E_min_keV,
                "Corman diffusion energy bounds are invalid");
  TENRYU_ASSERT(species_A > 0.0 && species_Z > 0.0,
                "Corman diffusion species must have positive A and Z");
  TENRYU_ASSERT(E_birth_keV > 0.0 && dt_s >= 0.0,
                "Corman diffusion birth energy and dt are invalid");
  TENRYU_ASSERT(E_birth_keV >= center_keV(p, 0),
                "Corman diffusion birth energy below bottom center is unsupported");
  TENRYU_ASSERT(r_node_dev != nullptr && vol_dev != nullptr &&
                    rho_dev != nullptr && Te_eV_dev != nullptr &&
                    Ti_eV_dev != nullptr && ne_dev != nullptr &&
                    S_birth_dev != nullptr && N_dev != nullptr &&
                    dep_e_dev != nullptr && dep_i_dev != nullptr,
                "Corman diffusion received a null device pointer");

  ScratchLayout scratch =
      carve_scratch(scratch_pool, scratch_bytes, p.n_groups, n_cells);
  cuda_check(cudaMemsetAsync(scratch.totals, 0, kTotals * sizeof(double), stream),
             "Corman diffusion zero totals failed");
  const std::size_t total =
      static_cast<std::size_t>(p.n_groups) * static_cast<std::size_t>(n_cells);
  cuda_check(cudaMemcpyAsync(scratch.N_old, N_dev, total * sizeof(double),
                             cudaMemcpyDeviceToDevice, stream),
             "Corman diffusion snapshot N failed");

  const int grid = (n_cells + kBlock - 1) / kBlock;
  compute_coefficients_kernel<<<grid, kBlock, 0, stream>>>(
      p, n_cells, species_A, species_Z, rho_dev, Te_eV_dev, Ti_eV_dev, ne_dev,
      scratch.tE, scratch.gamma, scratch.lnL_I);
  cuda_check(cudaGetLastError(),
             "Corman diffusion coefficient kernel launch failed");

  const BirthBinning birth = birth_binning(p, E_birth_keV);
  if (birth.top_excess_erg > 0.0) {
    static std::atomic<bool> warned_top_excess{false};
    bool expected = false;
    if (warned_top_excess.compare_exchange_strong(expected, true,
                                                  std::memory_order_relaxed)) {
      const double excess_keV =
          birth.top_excess_erg / corman_detail::kKeVToErg;
      core::log_warning(
          "Corman diffusion: product birth energy " +
          std::to_string(E_birth_keV) + " keV exceeds the top group center " +
          std::to_string(center_keV(p, p.n_groups - 1)) + " keV; " +
          std::to_string(excess_keV) + " keV (" +
          std::to_string(100.0 * excess_keV / E_birth_keV) +
          "% of birth energy) is deposited directly to electrons every step. "
          "Raise Burn diffusion E_max so the top group CENTER clears the "
          "highest product birth energy.");
    }
  }
  source_tally_kernel<<<1, 1, 0, stream>>>(
      p, birth, n_cells, E_birth_keV, vol_dev, S_birth_dev, dt_s, dep_e_dev,
      scratch.totals);
  cuda_check(cudaGetLastError(),
             "Corman diffusion source tally kernel launch failed");

  auto& cache = corman_cusparse_cache();
  if (cache.handle == nullptr) {
    cusparse_check(cusparseCreate(&cache.handle),
                   "Corman diffusion cusparseCreate failed");
  }
  cusparse_check(cusparseSetStream(cache.handle, stream),
                 "Corman diffusion cusparseSetStream failed");
  std::size_t actual_cusparse_bytes = 0U;
  cusparse_check(cusparseDgtsv2StridedBatch_bufferSizeExt(
                    cache.handle, n_cells, scratch.lower, scratch.diag,
                    scratch.upper, scratch.rhs, 1, n_cells,
                    &actual_cusparse_bytes),
                "Corman diffusion cuSPARSE buffer size failed");
  TENRYU_ASSERT(actual_cusparse_bytes <= scratch.cusparse_bytes,
                "Corman diffusion scratch lacks cuSPARSE workspace");

  for (int g = p.n_groups - 1; g >= 0; --g) {
    assemble_group_kernel<<<grid, kBlock, 0, stream>>>(
        p, birth, g, n_cells, species_A, species_Z, r_node_dev, rho_dev,
        S_birth_dev, scratch.N_old, N_dev, scratch.tE, scratch.gamma,
        scratch.lnL_I, dt_s, scratch.lower, scratch.diag, scratch.upper,
        scratch.rhs);
    cuda_check(cudaGetLastError(),
               "Corman diffusion assembly kernel launch failed");
    cusparse_check(cusparseDgtsv2StridedBatch(
                      cache.handle, n_cells, scratch.lower, scratch.diag,
                      scratch.upper, scratch.rhs, 1, n_cells,
                      scratch.cusparse_buffer),
                  "Corman diffusion cuSPARSE tridiagonal solve failed");
    publish_group_kernel<<<grid, kBlock, 0, stream>>>(g, n_cells, scratch.rhs,
                                                      N_dev);
    cuda_check(cudaGetLastError(),
               "Corman diffusion publish kernel launch failed");
    tally_group_kernel<<<1, 1, 0, stream>>>(
        p, g, n_cells, species_A, species_Z, r_node_dev, vol_dev, rho_dev,
        scratch.N_old, scratch.rhs, scratch.tE, scratch.gamma, scratch.lnL_I,
        dt_s, dep_e_dev, dep_i_dev, scratch.totals);
    cuda_check(cudaGetLastError(),
               "Corman diffusion tally kernel launch failed");
  }

  inflight_kernel<<<1, 1, 0, stream>>>(p, n_cells, vol_dev, N_dev,
                                       scratch.totals);
  cuda_check(cudaGetLastError(),
             "Corman diffusion inflight kernel launch failed");

  double totals[kTotals] = {0.0, 0.0, 0.0, 0.0, 0.0};
  cuda_check(cudaMemcpyAsync(totals, scratch.totals, sizeof(totals),
                             cudaMemcpyDeviceToHost, stream),
             "Corman diffusion copy totals failed");
  cuda_check(cudaStreamSynchronize(stream),
             "Corman diffusion stream synchronize failed");

  result.escaped_erg = totals[kTotalEscaped];
  result.inflight_erg = totals[kTotalInflight];
  result.sourced_erg = totals[kTotalSourced];
  return result;
}

}  // namespace tenryu::burn
