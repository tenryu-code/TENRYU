#include "radiation/sn_transport_gpu.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "core/error.hpp"
#include "parallel/comm_buffers.hpp"
#include "parallel/halo_exchange.hpp"
#include "parallel/reduction.hpp"
#include "radiation/lc_weights.cuh"
#include "radiation/sn_material_newton_gpu.cuh"

namespace tenryu::radiation {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kFourPi = 12.56637061435917295385;
constexpr double kFourPiOverThree = 4.18879020478639098462;
constexpr double kEnergyFloor = 1.0e-300;
constexpr double kSigmaFloor = 0.0;
constexpr double kDsaSigmaFloor = 1.0e-30;
constexpr double kDsaPivotFloor = 1.0e-280;
constexpr int kDsaMaxIterations = 50;

// TENRYU_SN_SWEEP_NO_GRAPH=1 forces the eager inner-iteration loop (the
// pre-W5 code path, byte-identical); default enables graph replay.
bool sn2d_sweep_graph_disabled() {
  static const bool disabled = [] {
    const char* v = std::getenv("TENRYU_SN_SWEEP_NO_GRAPH");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  return disabled;
}

constexpr double kDsaConvergenceTol = 1.0e-4;
constexpr int kSnTemperatureNewtonMaxIterations = 10;
constexpr double kSnTemperatureNewtonTol = 1.0e-6;
constexpr int kReduceBlock = 256;

inline void cuda_check(const cudaError_t err, const char* message) {
  if (err != cudaSuccess) {
    const std::string error_message = std::string(message) + " [" +
                                      cudaGetErrorName(err) + ": " +
                                      cudaGetErrorString(err) + "]";
    TENRYU_ASSERT(false, error_message);
  }
}

__host__ __device__ inline bool sn_finite(const double value) {
  return isfinite(value);
}

__host__ __device__ inline double finite_or_zero(const double value) {
  return sn_finite(value) ? value : 0.0;
}

__host__ __device__ inline double nonnegative_finite(const double value) {
  return sn_finite(value) ? fmax(value, 0.0) : 0.0;
}

__device__ inline void tally_sn_fixup(unsigned long long* __restrict__ count,
                                      double* __restrict__ artificial_abs,
                                      const std::size_t idx,
                                      const double clamped_value,
                                      const unsigned long long events = 1ULL) {
  if (count != nullptr) {
    atomicAdd(&count[idx], events);
  }
  if (artificial_abs != nullptr) {
    atomicAdd(&artificial_abs[idx],
              static_cast<double>(events) * fabs(clamped_value));
  }
}

__host__ __device__ inline double sn_mass_heat_capacity(
    const double rho,
    const double cv_e_value,
    const double cv_e_const,
    const double Cv_e_const) {
  const double cv_cell = finite_or_zero(cv_e_value);
  if (cv_cell > 0.0) {
    return cv_cell;
  }
  const double Cv_const = finite_or_zero(Cv_e_const);
  const double rho_c = nonnegative_finite(rho);
  if (Cv_const > 0.0 && rho_c > 0.0) {
    return Cv_const / rho_c;
  }
  return finite_or_zero(cv_e_const);
}

__host__ __device__ inline double sn_volume_heat_capacity(
    const double rho,
    const double cv_e_value,
    const double cv_e_const,
    const double Cv_e_const) {
  const double cv_cell = finite_or_zero(cv_e_value);
  const double rho_c = nonnegative_finite(rho);
  if (cv_cell > 0.0) {
    return rho_c * cv_cell;
  }
  const double Cv_const = finite_or_zero(Cv_e_const);
  if (Cv_const > 0.0) {
    return Cv_const;
  }
  return rho_c * finite_or_zero(cv_e_const);
}

__host__ __device__ inline std::size_t cell_group_index(const int cell,
                                                        const int group,
                                                        const int n_groups) {
  return static_cast<std::size_t>(cell) * static_cast<std::size_t>(n_groups) +
         static_cast<std::size_t>(group);
}

__host__ __device__ inline std::size_t psi_index(const int group,
                                                 const int angle,
                                                 const int cell,
                                                 const int n_angles,
                                                 const int n_cells) {
  return (static_cast<std::size_t>(group) * static_cast<std::size_t>(n_angles) +
          static_cast<std::size_t>(angle)) *
             static_cast<std::size_t>(n_cells) +
         static_cast<std::size_t>(cell);
}

__host__ __device__ inline double safe_radius(const double r) {
  return fmax(finite_or_zero(r), 0.0);
}

__host__ __device__ inline double face_area_1d(const double r) {
  const double rr = safe_radius(r);
  return kFourPi * rr * rr;
}

__host__ __device__ inline double shell_volume_1d(const double r_in,
                                                  const double r_out) {
  const double rin = safe_radius(r_in);
  const double rout = safe_radius(r_out);
  const double volume = kFourPiOverThree * (rout * rout * rout - rin * rin * rin);
  return fmax(volume, 0.0);
}

__host__ __device__ inline int cell_index_2d(const int i,
                                            const int j,
                                            const int nz) {
  return i * nz + j;
}

__host__ __device__ inline int node_index_2d(const int i,
                                            const int j,
                                            const int nz) {
  return i * (nz + 1) + j;
}

__host__ __device__ inline int n_r_faces_2d(const int nr, const int nz) {
  return (nr + 1) * nz;
}

__host__ __device__ inline int n_faces_2d(const int nr, const int nz) {
  return n_r_faces_2d(nr, nz) + nr * (nz + 1);
}

__host__ __device__ inline int r_face_index_2d(const int i,
                                              const int j,
                                              const int nz) {
  return i * nz + j;
}

__host__ __device__ inline int z_face_index_2d(const int i,
                                              const int j,
                                              const int nr,
                                              const int nz) {
  return n_r_faces_2d(nr, nz) + i * (nz + 1) + j;
}

__host__ __device__ inline std::size_t face_group_index_2d(const int face,
                                                           const int group,
                                                           const int n_groups) {
  return static_cast<std::size_t>(face) * static_cast<std::size_t>(n_groups) +
         static_cast<std::size_t>(group);
}

__host__ __device__ inline double sn_marshak_incoming_intensity(
    const int boundary_code,
    const double marshak_flux_erg_per_cm2_s) {
  return boundary_code == kSNBoundaryMarshak
             ? 2.0 * nonnegative_finite(marshak_flux_erg_per_cm2_s)
             : 0.0;
}

__host__ __device__ inline double safe_temperature_pow4(const double T) {
  const double Tc = fmin(fmax(finite_or_zero(T), 0.0), 1.0e6);
  const double T2 = Tc * Tc;
  return T2 * T2;
}

void compute_gauss_legendre(const int n,
                            std::vector<double>& mu,
                            std::vector<double>& weight) {
  TENRYU_ASSERT(n > 0 && (n % 2) == 0,
                "SN GPU transport requires a positive even number of angles");
  mu.assign(static_cast<std::size_t>(n), 0.0);
  weight.assign(static_cast<std::size_t>(n), 0.0);

  constexpr double tol = 1.0e-15;
  const int half = n / 2;
  for (int k = 0; k < half; ++k) {
    double x = std::cos(kPi * (static_cast<double>(k) + 0.75) /
                        (static_cast<double>(n) + 0.5));
    double p1 = 0.0;
    double p2 = 0.0;
    for (int iter = 0; iter < 64; ++iter) {
      p1 = 1.0;
      p2 = 0.0;
      for (int j = 1; j <= n; ++j) {
        const double p3 = p2;
        p2 = p1;
        p1 = ((2.0 * static_cast<double>(j) - 1.0) * x * p2 -
              (static_cast<double>(j) - 1.0) * p3) /
             static_cast<double>(j);
      }
      const double pp = static_cast<double>(n) * (x * p1 - p2) / (x * x - 1.0);
      const double dx = p1 / pp;
      x -= dx;
      if (std::abs(dx) <= tol) {
        break;
      }
    }

    p1 = 1.0;
    p2 = 0.0;
    for (int j = 1; j <= n; ++j) {
      const double p3 = p2;
      p2 = p1;
      p1 = ((2.0 * static_cast<double>(j) - 1.0) * x * p2 -
            (static_cast<double>(j) - 1.0) * p3) /
           static_cast<double>(j);
    }
    const double pp = static_cast<double>(n) * (x * p1 - p2) / (x * x - 1.0);
    const double w = 2.0 / ((1.0 - x * x) * pp * pp);
    mu[static_cast<std::size_t>(k)] = -x;
    mu[static_cast<std::size_t>(n - 1 - k)] = x;
    weight[static_cast<std::size_t>(k)] = w;
    weight[static_cast<std::size_t>(n - 1 - k)] = w;
  }
}

std::vector<double> angular_coefficients(const std::vector<double>& mu,
                                         const std::vector<double>& weight) {
  const int n_angles = static_cast<int>(mu.size());
  std::vector<double> alpha(static_cast<std::size_t>(n_angles + 1), 0.0);
  for (int n = 0; n < n_angles; ++n) {
    alpha[static_cast<std::size_t>(n + 1)] =
        alpha[static_cast<std::size_t>(n)] -
        mu[static_cast<std::size_t>(n)] * weight[static_cast<std::size_t>(n)];
  }
  alpha.front() = 0.0;
  alpha.back() = 0.0;
  for (double& value : alpha) {
    if (std::abs(value) < 1.0e-14) {
      value = 0.0;
    }
    value = std::max(value, 0.0);
  }
  return alpha;
}

template <typename T>
void upload_vector(parallel::DeviceArray& device,
                   const std::vector<T>& host,
                   const char* message) {
  const std::size_t bytes = sizeof(T) * host.size();
  device.resize(bytes);
  if (bytes > 0U) {
    cuda_check(cudaMemcpy(device.as<T>(), host.data(), bytes, cudaMemcpyHostToDevice),
               message);
  }
}

struct HostDirection2D {
  double mu_r = 0.0;
  double mu_z = 0.0;
  double weight = 0.0;
  int iz = 0;
  int iphi = 0;
};

struct OctantRange2D {
  int octant_id = 0;
  int dir_start = 0;
  int dir_count = 0;
};

int octant_id_2d(const double mu_r, const double mu_z) {
  return (mu_r < 0.0 ? 0 : 2) + (mu_z < 0.0 ? 0 : 1);
}

void compute_product_quadrature_2d(const int n_angles,
                                   std::vector<double>& mu_r,
                                   std::vector<double>& mu_z,
                                   std::vector<double>& weight,
                                   std::vector<int>& reflect_dir,
                                   std::vector<int>& reflect_z_dir,
                                   std::vector<double>& alpha_edge,
                                   std::vector<int>& sorted_dir_by_iz_m,
                                   std::vector<OctantRange2D>& octants) {
  std::vector<double> mu_z_1d;
  std::vector<double> w_z_1d;
  compute_gauss_legendre(n_angles, mu_z_1d, w_z_1d);

  const int n_phi_half = std::max(n_angles / 2, 1);
  std::vector<HostDirection2D> dirs;
  dirs.reserve(static_cast<std::size_t>(n_angles * n_phi_half));
  for (int iz = 0; iz < n_angles; ++iz) {
    const double mz = mu_z_1d[static_cast<std::size_t>(iz)];
    const double radial_mag = std::sqrt(std::max(0.0, 1.0 - mz * mz));
    for (int ip = 0; ip < n_phi_half; ++ip) {
      const double phi =
          (static_cast<double>(ip) + 0.5) * kPi / static_cast<double>(n_phi_half);
      HostDirection2D dir{};
      dir.mu_r = radial_mag * std::cos(phi);
      dir.mu_z = mz;
      dir.weight = w_z_1d[static_cast<std::size_t>(iz)] /
                   static_cast<double>(n_phi_half);
      dir.iz = iz;
      dir.iphi = ip;
      dirs.push_back(dir);
    }
  }

  std::stable_sort(dirs.begin(),
                   dirs.end(),
                   [](const HostDirection2D& a, const HostDirection2D& b) {
                     const bool a_neg = a.mu_r < 0.0;
                     const bool b_neg = b.mu_r < 0.0;
                     if (a_neg != b_neg) {
                       return a_neg;
                     }
                     if (a.iz != b.iz) {
                       return a.iz < b.iz;
                     }
                     return a.iphi < b.iphi;
                   });

  const int n_dirs = static_cast<int>(dirs.size());
  mu_r.assign(static_cast<std::size_t>(n_dirs), 0.0);
  mu_z.assign(static_cast<std::size_t>(n_dirs), 0.0);
  weight.assign(static_cast<std::size_t>(n_dirs), 0.0);
  reflect_dir.assign(static_cast<std::size_t>(n_dirs), -1);
  reflect_z_dir.assign(static_cast<std::size_t>(n_dirs), -1);
  alpha_edge.assign(
      static_cast<std::size_t>(n_angles) *
          static_cast<std::size_t>(n_phi_half + 1),
      0.0);
  sorted_dir_by_iz_m.assign(
      static_cast<std::size_t>(n_angles) * static_cast<std::size_t>(n_phi_half),
      -1);
  octants.assign(4U, OctantRange2D{});
  for (int octant = 0; octant < 4; ++octant) {
    octants[static_cast<std::size_t>(octant)].octant_id = octant;
  }
  std::vector<int> dir_by_iz_iphi(
      static_cast<std::size_t>(n_angles) * static_cast<std::size_t>(n_phi_half),
      -1);
  for (int d = 0; d < n_dirs; ++d) {
    const HostDirection2D& dir = dirs[static_cast<std::size_t>(d)];
    mu_r[static_cast<std::size_t>(d)] = dir.mu_r;
    mu_z[static_cast<std::size_t>(d)] = dir.mu_z;
    weight[static_cast<std::size_t>(d)] = dir.weight;
    dir_by_iz_iphi[static_cast<std::size_t>(dir.iz) *
                       static_cast<std::size_t>(n_phi_half) +
                   static_cast<std::size_t>(dir.iphi)] = d;
  }
  int cursor = 0;
  for (int octant = 0; octant < 4; ++octant) {
    OctantRange2D& range = octants[static_cast<std::size_t>(octant)];
    range.dir_start = cursor;
    while (cursor < n_dirs &&
           octant_id_2d(mu_r[static_cast<std::size_t>(cursor)],
                        mu_z[static_cast<std::size_t>(cursor)]) == octant) {
      ++cursor;
    }
    range.dir_count = cursor - range.dir_start;
  }
  TENRYU_ASSERT(cursor == n_dirs, "SN GPU 2D octant ranges require sorted directions");
  for (int d = 0; d < n_dirs; ++d) {
    const HostDirection2D& dir = dirs[static_cast<std::size_t>(d)];
    const int reflected_ip = n_phi_half - 1 - dir.iphi;
    const int reflected_iz = n_angles - 1 - dir.iz;
    for (int r = 0; r < n_dirs; ++r) {
      const HostDirection2D& candidate = dirs[static_cast<std::size_t>(r)];
      if (candidate.iz == dir.iz && candidate.iphi == reflected_ip) {
        reflect_dir[static_cast<std::size_t>(d)] = r;
      }
      if (candidate.iz == reflected_iz && candidate.iphi == dir.iphi) {
        reflect_z_dir[static_cast<std::size_t>(d)] = r;
      }
    }
  }
  for (int iz = 0; iz < n_angles; ++iz) {
    const std::size_t alpha_row =
        static_cast<std::size_t>(iz) * static_cast<std::size_t>(n_phi_half + 1);
    const std::size_t sorted_row =
        static_cast<std::size_t>(iz) * static_cast<std::size_t>(n_phi_half);
    alpha_edge[alpha_row] = 0.0;
    for (int m = 0; m < n_phi_half; ++m) {
      const int ip = n_phi_half - 1 - m;
      const int d =
          dir_by_iz_iphi[static_cast<std::size_t>(iz) *
                             static_cast<std::size_t>(n_phi_half) +
                         static_cast<std::size_t>(ip)];
      TENRYU_ASSERT(d >= 0, "SN GPU 2D sorted angular direction map incomplete");
      sorted_dir_by_iz_m[sorted_row + static_cast<std::size_t>(m)] = d;
      double next = alpha_edge[alpha_row + static_cast<std::size_t>(m)] -
                    mu_r[static_cast<std::size_t>(d)] *
                        weight[static_cast<std::size_t>(d)];
      if (next < 0.0 && std::abs(next) < 1.0e-14) {
        next = 0.0;
      }
      TENRYU_ASSERT(next >= -1.0e-12,
                    "SN GPU 2D alpha edge recurrence became negative");
      alpha_edge[alpha_row + static_cast<std::size_t>(m + 1)] =
          std::max(next, 0.0);
    }
    TENRYU_ASSERT(
        std::abs(alpha_edge[alpha_row + static_cast<std::size_t>(n_phi_half)]) <
            1.0e-12,
        "SN GPU 2D alpha edge recurrence did not close");
    alpha_edge[alpha_row + static_cast<std::size_t>(n_phi_half)] = 0.0;
  }
}

__global__ void sn_build_source_kernel(const double* __restrict__ sigma_a_eff,
                                       const double* __restrict__ Te,
                                       double* __restrict__ source_emission,
                                       const int n_cells,
                                       const int n_groups,
                                       const double temperature_floor_eV,
                                       const PlanckTableDeviceView planck) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_total = n_cells * n_groups;
  if (idx >= n_total) {
    return;
  }
  const int c = idx / n_groups;
  const int g = idx - c * n_groups;
  const double T = fmax(finite_or_zero(Te[c]), temperature_floor_eV);
  const double T4 = safe_temperature_pow4(T);
  const double b_g = (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T), 0.0);
  const double sigma = nonnegative_finite(sigma_a_eff[idx]);
  source_emission[idx] =
      tenryu::core::constants::c_light * sigma *
      tenryu::core::constants::a_eV * T4 * b_g;
}

__global__ void sn_publish_material_sources_kernel(
    const double* __restrict__ sigma_a_eff,
    const double* __restrict__ source_emission,
    const double* __restrict__ E_sn,
    const double* __restrict__ vol,
    double* __restrict__ rad_dep,
    double* __restrict__ rad_emit,
    double* __restrict__ coverage,
    const int n_cells,
    const int n_groups,
    const double dt) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_total = n_cells * n_groups;
  if (idx >= n_total) {
    return;
  }
  const int c = idx / n_groups;
  const double V = nonnegative_finite(vol[c]);
  const double step_volume = fmax(dt, 0.0) * V;
  const double sigma = nonnegative_finite(sigma_a_eff[idx]);
  const double E = nonnegative_finite(E_sn[idx]);
  const double dep =
      tenryu::core::constants::c_light * sigma * E * step_volume;
  const double emit = nonnegative_finite(source_emission[idx]) * step_volume;
  rad_dep[idx] = sn_finite(dep) ? fmax(dep, 0.0) : 0.0;
  rad_emit[idx] = sn_finite(emit) ? fmax(emit, 0.0) : 0.0;
  if (coverage != nullptr) {
    coverage[idx] = 1.0;
  }
}

__global__ void sn_update_material_energy_kernel(
    double* __restrict__ ee,
    const double* __restrict__ Te_new,
    const double* __restrict__ Te_old,
    const double* __restrict__ rho,
    const double* __restrict__ cv_e,
    const int n_cells,
    const double temperature_floor,
    const double cv_e_const,
    const double Cv_e_const) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const double T_new = fmax(finite_or_zero(Te_new[c]), temperature_floor);
  const double T_old = fmax(finite_or_zero(Te_old[c]), temperature_floor);
  const double cv_mass = sn_mass_heat_capacity(
      rho[c], (cv_e != nullptr) ? cv_e[c] : 0.0, cv_e_const, Cv_e_const);
  const double delta_ee = cv_mass * (T_new - T_old);
  if (cv_mass > 0.0 && sn_finite(delta_ee)) {
    ee[c] += delta_ee;
  }
}

__global__ void sn_fill_kernel(double* __restrict__ values,
                               const int n_values,
                               const double value) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_values) {
    values[idx] = value;
  }
}

double positive_harmonic_mean(const double a, const double b) {
  const double aa = std::max(finite_or_zero(a), kDsaSigmaFloor);
  const double bb = std::max(finite_or_zero(b), kDsaSigmaFloor);
  return 2.0 / (1.0 / aa + 1.0 / bb);
}

__device__ double positive_harmonic_mean_device(const double a, const double b) {
  const double aa = fmax(finite_or_zero(a), kDsaSigmaFloor);
  const double bb = fmax(finite_or_zero(b), kDsaSigmaFloor);
  return 2.0 / (1.0 / aa + 1.0 / bb);
}

void solve_dsa_correction_1d(const double* phi_half,
                             const double* phi_old,
                             const double* sigma_a,
                             const double* sigma_s,
                             const double* node_r,
                             const double* vol,
                             double* delta_phi,
                             const int n_cells,
                             const int n_groups) {
  const std::size_t n_total =
      static_cast<std::size_t>(std::max(n_cells, 0)) *
      static_cast<std::size_t>(std::max(n_groups, 0));
  if (delta_phi == nullptr) {
    return;
  }
  std::fill(delta_phi, delta_phi + n_total, 0.0);
  if (phi_half == nullptr || phi_old == nullptr || sigma_a == nullptr ||
      sigma_s == nullptr || node_r == nullptr || vol == nullptr ||
      n_cells <= 0 || n_groups <= 0) {
    return;
  }

  std::vector<double> lower(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> diag(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> upper(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> rhs(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> diffusion(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> correction(static_cast<std::size_t>(n_cells), 0.0);

  for (int g = 0; g < n_groups; ++g) {
    std::fill(lower.begin(), lower.end(), 0.0);
    std::fill(diag.begin(), diag.end(), 0.0);
    std::fill(upper.begin(), upper.end(), 0.0);
    std::fill(rhs.begin(), rhs.end(), 0.0);
    std::fill(correction.begin(), correction.end(), 0.0);

    for (int c = 0; c < n_cells; ++c) {
      const std::size_t c_us = static_cast<std::size_t>(c);
      const std::size_t cg = cell_group_index(c, g, n_groups);
      const double sigma_abs = std::max(finite_or_zero(sigma_a[cg]), 0.0);
      const double sigma_scat = std::max(finite_or_zero(sigma_s[cg]), 0.0);
      const double sigma_t = std::max(sigma_abs + sigma_scat, kDsaSigmaFloor);
      diffusion[c_us] = 1.0 / (3.0 * sigma_t);
      diag[c_us] = sigma_abs;
      rhs[c_us] =
          0.5 * sigma_scat *
          (finite_or_zero(phi_half[cg]) - finite_or_zero(phi_old[cg]));
      if (!sn_finite(rhs[c_us])) {
        rhs[c_us] = 0.0;
      }
    }

    for (int c = 0; c < n_cells; ++c) {
      const std::size_t c_us = static_cast<std::size_t>(c);
      const double V_input = std::max(finite_or_zero(vol[c_us]), 0.0);
      const double V = (V_input > 0.0)
                           ? V_input
                           : shell_volume_1d(node_r[c], node_r[c + 1]);
      if (!(V > 0.0)) {
        lower[c_us] = 0.0;
        diag[c_us] = 1.0;
        upper[c_us] = 0.0;
        rhs[c_us] = 0.0;
        continue;
      }

      const double center =
          0.5 * (safe_radius(node_r[c]) + safe_radius(node_r[c + 1]));
      if (c + 1 < n_cells) {
        const double center_right =
            0.5 * (safe_radius(node_r[c + 1]) + safe_radius(node_r[c + 2]));
        const double dr = center_right - center;
        const double A = face_area_1d(node_r[c + 1]);
        const double d_face =
            positive_harmonic_mean(diffusion[c_us], diffusion[c_us + 1U]);
        const double coeff = (dr > 0.0) ? (A * d_face / (dr * V)) : 0.0;
        if (sn_finite(coeff) && coeff > 0.0) {
          diag[c_us] += coeff;
          upper[c_us] -= coeff;
        }
      }
      if (c > 0) {
        const double center_left =
            0.5 * (safe_radius(node_r[c - 1]) + safe_radius(node_r[c]));
        const double dr = center - center_left;
        const double A = face_area_1d(node_r[c]);
        const double d_face =
            positive_harmonic_mean(diffusion[c_us - 1U], diffusion[c_us]);
        const double coeff = (dr > 0.0) ? (A * d_face / (dr * V)) : 0.0;
        if (sn_finite(coeff) && coeff > 0.0) {
          diag[c_us] += coeff;
          lower[c_us] -= coeff;
        }
      }
    }

    for (int c = 0; c < n_cells; ++c) {
      const std::size_t c_us = static_cast<std::size_t>(c);
      if (!sn_finite(lower[c_us])) {
        lower[c_us] = 0.0;
      }
      if (!sn_finite(upper[c_us])) {
        upper[c_us] = 0.0;
      }
      if (!sn_finite(diag[c_us]) || diag[c_us] <= 0.0) {
        lower[c_us] = 0.0;
        diag[c_us] = 1.0;
        upper[c_us] = 0.0;
        rhs[c_us] = 0.0;
      }
      if (!sn_finite(rhs[c_us])) {
        rhs[c_us] = 0.0;
      }
    }

    for (int c = 1; c < n_cells; ++c) {
      const std::size_t i = static_cast<std::size_t>(c);
      const std::size_t im1 = i - 1U;
      const double pivot =
          std::copysign(std::max(std::abs(diag[im1]), kDsaPivotFloor), diag[im1]);
      const double m = lower[i] / pivot;
      diag[i] -= m * upper[im1];
      rhs[i] -= m * rhs[im1];
    }

    for (int c = n_cells - 1; c >= 0; --c) {
      const std::size_t i = static_cast<std::size_t>(c);
      const double pivot =
          std::copysign(std::max(std::abs(diag[i]), kDsaPivotFloor), diag[i]);
      const double next_term =
          (c + 1 < n_cells) ? upper[i] * correction[i + 1U] : 0.0;
      correction[i] = (rhs[i] - next_term) / pivot;
      if (!sn_finite(correction[i])) {
        correction[i] = 0.0;
      }
    }

    for (int c = 0; c < n_cells; ++c) {
      delta_phi[cell_group_index(c, g, n_groups)] =
          correction[static_cast<std::size_t>(c)];
    }
  }
}

__global__ void sn_chi_kernel(const double* __restrict__ E,
                              const double* __restrict__ P_rr,
                              double* __restrict__ chi,
                              const int n_total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_total) {
    return;
  }
  const double E_i = nonnegative_finite(E[idx]);
  const double P_i = nonnegative_finite(P_rr[idx]);
  const double value = P_i / fmax(E_i, kEnergyFloor);
  chi[idx] = sn_finite(value) ? fmin(fmax(value, 0.0), 1.0) : (1.0 / 3.0);
}

__global__ void sn_chi_z_kernel(const double* __restrict__ E,
                                const double* __restrict__ P_zz,
                                double* __restrict__ chi_z,
                                const int n_total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_total) {
    return;
  }
  const double E_i = nonnegative_finite(E[idx]);
  const double P_i = nonnegative_finite(P_zz[idx]);
  const double value = P_i / fmax(E_i, kEnergyFloor);
  chi_z[idx] = sn_finite(value) ? fmin(fmax(value, 0.0), 1.0) : (1.0 / 3.0);
}

__global__ void sn_scalar_flux_to_E_kernel(const double* __restrict__ scalar_flux,
                                           double* __restrict__ E_out,
                                           const int n_total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_total) {
    return;
  }
  E_out[idx] =
      fmax(finite_or_zero(scalar_flux[idx]), 0.0) /
      tenryu::core::constants::c_light;
}

__global__ void sn_convergence_reduce_kernel(
    const double* __restrict__ phi_new,
    const double* __restrict__ phi_old,
    double* __restrict__ block_error,
    const int n_total) {
  extern __shared__ double shared[];
  double* s_error = shared;
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  double local_ratio = 0.0;
  if (idx < n_total) {
    const double next = finite_or_zero(phi_new[idx]);
    const double old = finite_or_zero(phi_old[idx]);
    const double denom = fmax(fabs(next), kEnergyFloor);
    local_ratio = fabs(next - old) / denom;
  }
  s_error[tid] = local_ratio;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_error[tid] = fmax(s_error[tid], s_error[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    block_error[blockIdx.x] = s_error[0];
  }
}

__global__ void sn_sweep_1d_kernel(const double* __restrict__ sigma_a,
                                   const double* __restrict__ sigma_s,
                                   const double* __restrict__ source_emission,
                                   const double* __restrict__ scalar_flux,
                                   const double* __restrict__ node_r,
                                   const double* __restrict__ vol,
                                   const double* __restrict__ mu,
                                   const double* __restrict__ weights,
                                   const double* __restrict__ alpha_half,
                                   double* __restrict__ psi_bar,
                                   double* __restrict__ new_scalar_flux,
                                   double* __restrict__ E_out,
                                   double* __restrict__ P_rr_out,
                                   const int n_cells,
                                   const int n_groups,
                                   const int n_angles,
                                   const double dt) {
  const int g = blockIdx.x;
  if (g >= n_groups || threadIdx.x != 0) {
    return;
  }
  const double inv_cdt =
      (dt > 0.0) ? (1.0 / (tenryu::core::constants::c_light * dt)) : 0.0;
  extern __shared__ double shared[];
  double* angular_edge = shared;
  double* inner_boundary = shared + n_cells;

  for (int c = 0; c < n_cells; ++c) {
    const std::size_t cg = cell_group_index(c, g, n_groups);
    new_scalar_flux[cg] = 0.0;
    E_out[cg] = 0.0;
    P_rr_out[cg] = 0.0;
    angular_edge[c] = 0.0;
  }
  for (int n = 0; n < n_angles; ++n) {
    inner_boundary[n] = 0.0;
  }

  for (int n = 0; n < n_angles; ++n) {
    const double mu_n = mu[n];
    if (mu_n == 0.0) {
      continue;
    }
    const double abs_mu = fabs(mu_n);
    const double alpha_prev = alpha_half[n];
    const double alpha_next = alpha_half[n + 1];
    const double w = weights[n];
    const double mu2 = mu_n * mu_n;

    if (mu_n < 0.0) {
      double incoming = 0.0;
      for (int c = n_cells - 1; c >= 0; --c) {
        const std::size_t cg = cell_group_index(c, g, n_groups);
        const double area_in = face_area_1d(node_r[c]);
        const double area_out = face_area_1d(node_r[c + 1]);
        const double V_input = nonnegative_finite(vol[c]);
        const double V =
            (V_input > 0.0) ? V_input : shell_volume_1d(node_r[c], node_r[c + 1]);
        const double sigma_t =
            fmax(nonnegative_finite(sigma_a[cg]) + nonnegative_finite(sigma_s[cg]),
                 kSigmaFloor);
        const double sigma_eff = sigma_t + inv_cdt;
        const double source =
            0.5 * nonnegative_finite(source_emission[cg]) +
            0.5 * nonnegative_finite(sigma_s[cg]) *
                fmax(finite_or_zero(scalar_flux[cg]), 0.0);
        const double edge_in = fmax(angular_edge[c], 0.0);
        const double denom =
            2.0 * abs_mu * area_in + 2.0 * alpha_next + sigma_eff * V;
        double average = 0.0;
        if (denom > 0.0 && sn_finite(denom)) {
          const double numer =
              V * source + abs_mu * (area_in + area_out) * incoming +
              (alpha_prev + alpha_next) * edge_in;
          average = numer / denom;
        }
        average = fmax(finite_or_zero(average), 0.0);
        double outgoing = 2.0 * average - incoming;
        double edge_out = 2.0 * average - edge_in;
        if (!sn_finite(outgoing) || outgoing < 0.0) {
          outgoing = 0.0;
          average = 0.5 * incoming;  // conservative half-range fixup
        }
        if (!sn_finite(edge_out) || edge_out < 0.0) {
          edge_out = 0.0;
        }
        if (psi_bar != nullptr) {
          psi_bar[psi_index(g, n, c, n_angles, n_cells)] = average;
        }
        new_scalar_flux[cg] += w * average;
        P_rr_out[cg] += w * mu2 * average / tenryu::core::constants::c_light;
        angular_edge[c] = edge_out;
        incoming = outgoing;
      }
      inner_boundary[n] = incoming;
    } else {
      const int reflected = n_angles - 1 - n;
      double incoming =
          (reflected >= 0 && reflected < n_angles) ? fmax(inner_boundary[reflected], 0.0)
                                                   : 0.0;
      for (int c = 0; c < n_cells; ++c) {
        const std::size_t cg = cell_group_index(c, g, n_groups);
        const double area_in = face_area_1d(node_r[c]);
        const double area_out = face_area_1d(node_r[c + 1]);
        const double V_input = nonnegative_finite(vol[c]);
        const double V =
            (V_input > 0.0) ? V_input : shell_volume_1d(node_r[c], node_r[c + 1]);
        const double sigma_t =
            fmax(nonnegative_finite(sigma_a[cg]) + nonnegative_finite(sigma_s[cg]),
                 kSigmaFloor);
        const double sigma_eff = sigma_t + inv_cdt;
        const double source =
            0.5 * nonnegative_finite(source_emission[cg]) +
            0.5 * nonnegative_finite(sigma_s[cg]) *
                fmax(finite_or_zero(scalar_flux[cg]), 0.0);
        const double edge_in = fmax(angular_edge[c], 0.0);
        const double denom =
            2.0 * mu_n * area_out + 2.0 * alpha_next + sigma_eff * V;
        double average = 0.0;
        if (denom > 0.0 && sn_finite(denom)) {
          const double numer =
              V * source + mu_n * (area_in + area_out) * incoming +
              (alpha_prev + alpha_next) * edge_in;
          average = numer / denom;
        }
        average = fmax(finite_or_zero(average), 0.0);
        double outgoing = 2.0 * average - incoming;
        double edge_out = 2.0 * average - edge_in;
        if (!sn_finite(outgoing) || outgoing < 0.0) {
          outgoing = 0.0;
          average = 0.5 * incoming;  // conservative half-range fixup
        }
        if (!sn_finite(edge_out) || edge_out < 0.0) {
          edge_out = 0.0;
        }
        if (psi_bar != nullptr) {
          psi_bar[psi_index(g, n, c, n_angles, n_cells)] = average;
        }
        new_scalar_flux[cg] += w * average;
        P_rr_out[cg] += w * mu2 * average / tenryu::core::constants::c_light;
        angular_edge[c] = edge_out;
        incoming = outgoing;
      }
    }
  }

  for (int c = 0; c < n_cells; ++c) {
    const std::size_t cg = cell_group_index(c, g, n_groups);
    E_out[cg] =
        fmax(finite_or_zero(new_scalar_flux[cg]), 0.0) /
        tenryu::core::constants::c_light;
  }
}

__device__ inline void cell_geometry_2d(const double* __restrict__ node_r,
                                        const double* __restrict__ node_z,
                                        const double* __restrict__ vol,
                                        const int i,
                                        const int j,
                                        const int nz,
                                        double* r_left,
                                        double* r_right,
                                        double* dz,
                                        double* area_r_left,
                                        double* area_r_right,
                                        double* area_z,
                                        double* volume) {
  const int n00 = node_index_2d(i, j, nz);
  const int n10 = node_index_2d(i + 1, j, nz);
  const int n11 = node_index_2d(i + 1, j + 1, nz);
  const int n01 = node_index_2d(i, j + 1, nz);
  *r_left = safe_radius(0.5 * (finite_or_zero(node_r[n00]) + finite_or_zero(node_r[n01])));
  *r_right = safe_radius(0.5 * (finite_or_zero(node_r[n10]) + finite_or_zero(node_r[n11])));
  const double z_bottom =
      0.5 * (finite_or_zero(node_z[n00]) + finite_or_zero(node_z[n10]));
  const double z_top =
      0.5 * (finite_or_zero(node_z[n01]) + finite_or_zero(node_z[n11]));
  *dz = fmax(fabs(z_top - z_bottom), 0.0);
  *area_r_left = kTwoPi * (*r_left) * (*dz);
  *area_r_right = kTwoPi * (*r_right) * (*dz);
  *area_z = kPi * fmax((*r_right) * (*r_right) - (*r_left) * (*r_left), 0.0);
  const int c = cell_index_2d(i, j, nz);
  const double V_input = nonnegative_finite(vol[c]);
  *volume = (V_input > 0.0) ? V_input : (*area_z) * (*dz);
}

__host__ __device__ inline std::size_t direction_cell_index(const int group,
                                                            const int dir,
                                                            const int cell,
                                                            const int n_dirs,
                                                            const int n_cells) {
  return (static_cast<std::size_t>(group) * static_cast<std::size_t>(n_dirs) +
          static_cast<std::size_t>(dir)) *
             static_cast<std::size_t>(n_cells) +
         static_cast<std::size_t>(cell);
}

__host__ __device__ inline std::size_t direction_face_index(const int group,
                                                            const int dir,
                                                            const int face,
                                                            const int n_dirs,
                                                            const int n_faces) {
  return (static_cast<std::size_t>(group) * static_cast<std::size_t>(n_dirs) +
          static_cast<std::size_t>(dir)) *
             static_cast<std::size_t>(n_faces) +
         static_cast<std::size_t>(face);
}

__host__ __device__ inline std::size_t phi_edge_index_2d(const int group,
                                                         const int cell,
                                                         const int iz,
                                                         const int n_cells,
                                                         const int n_polar) {
  return (static_cast<std::size_t>(group) * static_cast<std::size_t>(n_cells) +
          static_cast<std::size_t>(cell)) *
             static_cast<std::size_t>(n_polar) +
         static_cast<std::size_t>(iz);
}

__device__ inline void wavefront_ij(const int stage,
                                    const int local,
                                    const bool forward_r,
                                    const bool forward_z,
                                    const int nr,
                                    const int nz,
                                    int* i,
                                    int* j) {
  const int a_min = (stage > (nz - 1)) ? (stage - (nz - 1)) : 0;
  const int a = a_min + local;
  const int b = stage - a;
  *i = forward_r ? a : (nr - 1 - a);
  *j = forward_z ? b : (nz - 1 - b);
}

__device__ inline double read_incoming_r(
    const double* __restrict__ r_face_dir,
    const double* __restrict__ reflect_axis,
    const double* __restrict__ reflect_outer_r,
    const int g,
    const int d,
    const int reflected,
    const int i,
    const int j,
    const int nr,
    const int nz,
    const int n_dirs,
    const int n_cells,
    const bool forward_r,
    const double abs_mr,
    const int r_outer_boundary) {
  if (!(abs_mr > 0.0)) {
    return 0.0;
  }
  if (forward_r) {
    if (i == 0) {
      if (reflected >= 0) {
        const std::size_t ref_idx =
            (static_cast<std::size_t>(g) * static_cast<std::size_t>(n_dirs) +
             static_cast<std::size_t>(reflected)) *
                static_cast<std::size_t>(nz) +
            static_cast<std::size_t>(j);
        return fmax(finite_or_zero(reflect_axis[ref_idx]), 0.0);
      }
    } else {
      const int upwind = cell_index_2d(i - 1, j, nz);
      return fmax(finite_or_zero(
                      r_face_dir[direction_cell_index(g, d, upwind, n_dirs, n_cells)]),
                  0.0);
    }
  } else {
    if (i + 1 < nr) {
      const int upwind = cell_index_2d(i + 1, j, nz);
      return fmax(finite_or_zero(
                      r_face_dir[direction_cell_index(g, d, upwind, n_dirs, n_cells)]),
                  0.0);
    }
    if (r_outer_boundary == kSNBoundaryReflect && reflected >= 0) {
      const std::size_t ref_idx =
          (static_cast<std::size_t>(g) * static_cast<std::size_t>(n_dirs) +
           static_cast<std::size_t>(reflected)) *
              static_cast<std::size_t>(nz) +
          static_cast<std::size_t>(j);
      return fmax(finite_or_zero(reflect_outer_r[ref_idx]), 0.0);
    }
  }
  return 0.0;
}

__device__ inline double read_incoming_z(
    const double* __restrict__ z_face_dir,
    const double* __restrict__ reflect_z_bottom,
    const double* __restrict__ reflect_z_top,
    const int g,
    const int d,
    const int reflected_z,
    const int i,
    const int j,
    const int nr,
    const int nz,
    const int n_dirs,
    const int n_cells,
    const bool forward_z,
    const double abs_mz,
    const int z_bottom_boundary,
    const int z_top_boundary,
    const double marshak_flux_erg_per_cm2_s) {
  if (!(abs_mz > 0.0)) {
    return 0.0;
  }
  if (forward_z) {
    if (j > 0) {
      const int upwind = cell_index_2d(i, j - 1, nz);
      return fmax(finite_or_zero(
                      z_face_dir[direction_cell_index(g, d, upwind, n_dirs, n_cells)]),
                  0.0);
    }
    if (z_bottom_boundary == kSNBoundaryReflect && reflected_z >= 0) {
      const std::size_t ref_idx =
          (static_cast<std::size_t>(g) * static_cast<std::size_t>(n_dirs) +
           static_cast<std::size_t>(reflected_z)) *
              static_cast<std::size_t>(nr) +
          static_cast<std::size_t>(i);
      return fmax(finite_or_zero(reflect_z_bottom[ref_idx]), 0.0);
    }
    if (z_bottom_boundary == kSNBoundaryMarshak) {
      return sn_marshak_incoming_intensity(z_bottom_boundary,
                                           marshak_flux_erg_per_cm2_s);
    }
  } else {
    if (j + 1 < nz) {
      const int upwind = cell_index_2d(i, j + 1, nz);
      return fmax(finite_or_zero(
                      z_face_dir[direction_cell_index(g, d, upwind, n_dirs, n_cells)]),
                  0.0);
    }
    if (z_top_boundary == kSNBoundaryReflect && reflected_z >= 0) {
      const std::size_t ref_idx =
          (static_cast<std::size_t>(g) * static_cast<std::size_t>(n_dirs) +
           static_cast<std::size_t>(reflected_z)) *
              static_cast<std::size_t>(nr) +
          static_cast<std::size_t>(i);
      return fmax(finite_or_zero(reflect_z_top[ref_idx]), 0.0);
    }
    if (z_top_boundary == kSNBoundaryMarshak) {
      return sn_marshak_incoming_intensity(z_top_boundary,
                                           marshak_flux_erg_per_cm2_s);
    }
  }
  return 0.0;
}

__device__ inline void write_reflect_axis_if_needed(
    double* __restrict__ reflect_axis,
    const int g,
    const int d,
    const int i,
    const int j,
    const int nz,
    const int n_dirs,
    const bool forward_r,
    const double outgoing_r) {
  if (!forward_r && i == 0) {
    const std::size_t ref_idx =
        (static_cast<std::size_t>(g) * static_cast<std::size_t>(n_dirs) +
         static_cast<std::size_t>(d)) *
            static_cast<std::size_t>(nz) +
        static_cast<std::size_t>(j);
    reflect_axis[ref_idx] = outgoing_r;
  }
}

__device__ inline void write_reflect_outer_r_if_needed(
    double* __restrict__ reflect_outer_r,
    const int g,
    const int d,
    const int i,
    const int j,
    const int nr,
    const int nz,
    const int n_dirs,
    const bool forward_r,
    const int r_outer_boundary,
    const double outgoing_r) {
  if (r_outer_boundary == kSNBoundaryReflect && forward_r && i + 1 == nr) {
    const std::size_t ref_idx =
        (static_cast<std::size_t>(g) * static_cast<std::size_t>(n_dirs) +
         static_cast<std::size_t>(d)) *
            static_cast<std::size_t>(nz) +
        static_cast<std::size_t>(j);
    reflect_outer_r[ref_idx] = outgoing_r;
  }
}

__device__ inline void write_reflect_z_if_needed(
    double* __restrict__ reflect_z_bottom,
    double* __restrict__ reflect_z_top,
    const int g,
    const int d,
    const int i,
    const int j,
    const int nr,
    const int nz,
    const int n_dirs,
    const bool forward_z,
    const int z_bottom_boundary,
    const int z_top_boundary,
    const double outgoing_z) {
  if ((z_bottom_boundary == kSNBoundaryReflect ||
       z_bottom_boundary == kSNBoundaryMarshak) &&
      !forward_z && j == 0) {
    const std::size_t ref_idx =
        (static_cast<std::size_t>(g) * static_cast<std::size_t>(n_dirs) +
         static_cast<std::size_t>(d)) *
            static_cast<std::size_t>(nr) +
        static_cast<std::size_t>(i);
    reflect_z_bottom[ref_idx] = outgoing_z;
  }
  if ((z_top_boundary == kSNBoundaryReflect ||
       z_top_boundary == kSNBoundaryMarshak) &&
      forward_z && j + 1 == nz) {
    const std::size_t ref_idx =
        (static_cast<std::size_t>(g) * static_cast<std::size_t>(n_dirs) +
         static_cast<std::size_t>(d)) *
            static_cast<std::size_t>(nr) +
        static_cast<std::size_t>(i);
    reflect_z_top[ref_idx] = outgoing_z;
  }
}

__device__ inline void accumulate_unique_face_psi(
    double* __restrict__ D_face_psi,
    const int g,
    const int d,
    const int i,
    const int j,
    const int nr,
    const int nz,
    const int n_dirs,
    const int n_faces,
    const bool forward_r,
    const bool forward_z,
    const double abs_mr,
    const double abs_mz,
    const double outgoing_r,
    const double outgoing_z,
    const double incoming_r,
    const double incoming_z,
    const int r_outer_boundary,
    const int z_bottom_boundary,
    const int z_top_boundary) {
  if (abs_mr > 0.0) {
    if (forward_r) {
      const int face = r_face_index_2d(i + 1, j, nz);
      D_face_psi[direction_face_index(g, d, face, n_dirs, n_faces)] = outgoing_r;
    } else if (i > 0) {
      const int face = r_face_index_2d(i, j, nz);
      D_face_psi[direction_face_index(g, d, face, n_dirs, n_faces)] = outgoing_r;
    }
    if (r_outer_boundary == kSNBoundaryReflect && !forward_r && i + 1 == nr) {
      const int outer = r_face_index_2d(nr, j, nz);
      D_face_psi[direction_face_index(g, d, outer, n_dirs, n_faces)] =
          incoming_r;
    }
  }
  if (abs_mz > 0.0) {
    const int face = forward_z ? z_face_index_2d(i, j + 1, nr, nz)
                               : z_face_index_2d(i, j, nr, nz);
    D_face_psi[direction_face_index(g, d, face, n_dirs, n_faces)] = outgoing_z;
    if ((z_bottom_boundary == kSNBoundaryReflect ||
         z_bottom_boundary == kSNBoundaryMarshak) &&
        forward_z && j == 0) {
      const int bottom = z_face_index_2d(i, 0, nr, nz);
      D_face_psi[direction_face_index(g, d, bottom, n_dirs, n_faces)] =
          incoming_z;
    }
    if ((z_top_boundary == kSNBoundaryReflect ||
         z_top_boundary == kSNBoundaryMarshak) &&
        !forward_z && j + 1 == nz) {
      const int top = z_face_index_2d(i, nz, nr, nz);
      D_face_psi[direction_face_index(g, d, top, n_dirs, n_faces)] =
          incoming_z;
    }
  }
}

__global__ void sn_sweep_2d_kernel(const double* __restrict__ sigma_a,
                                   const double* __restrict__ sigma_s,
                                   const double* __restrict__ source_emission,
                                   const double* __restrict__ scalar_flux,
                                   const double* __restrict__ node_r,
                                   const double* __restrict__ node_z,
                                   const double* __restrict__ vol,
                                   const double* __restrict__ mu_r,
                                   const double* __restrict__ mu_z,
                                   const double* __restrict__ weights,
                                   const int* __restrict__ reflect_dir,
                                   const int* __restrict__ reflect_z_dir,
                                   const double* __restrict__ alpha_edge,
                                   const int* __restrict__ sorted_dir,
                                   double* __restrict__ phi_edge,
                                   double* __restrict__ r_face_dir,
                                   double* __restrict__ z_face_dir,
                                   double* __restrict__ reflect_axis,
                                   double* __restrict__ reflect_outer_r,
                                   double* __restrict__ reflect_z_bottom,
                                   double* __restrict__ reflect_z_top,
                                   double* __restrict__ D_avg,
                                   double* __restrict__ D_face_psi,
                                   unsigned long long* __restrict__ radial_fixup_count,
                                   double* __restrict__ radial_fixup_artificial_abs,
                                   unsigned long long* __restrict__ angular_fixup_count,
                                   double* __restrict__ angular_fixup_artificial_abs,
                                   const int nr,
                                   const int nz,
                                   const int n_groups,
                                   const int n_dirs,
                                   const int m,
                                   const int n_phi_half,
                                   const int n_polar,
                                   const int r_outer_boundary,
                                   const int z_bottom_boundary,
                                   const int z_top_boundary,
                                   const double marshak_flux_erg_per_cm2_s,
                                   const int i_begin,
                                   const int i_end,
                                   const int wanted_sign) {
  const int iz = blockIdx.x;
  const int g = blockIdx.y;
  if (g >= n_groups || iz >= n_polar || m >= n_phi_half) {
    return;
  }
  const int d = sorted_dir[iz * n_phi_half + m];
  if (d < 0 || d >= n_dirs) {
    return;
  }
  const int n_cells = nr * nz;
  const int n_faces = n_faces_2d(nr, nz);
  const double mr = mu_r[d];
  const double mz = mu_z[d];
  const double w = weights[d];
  const double abs_mr = fabs(mr);
  const double abs_mz = fabs(mz);
  const bool forward_r = mr >= 0.0;
  const bool forward_z = mz >= 0.0;
  // KBA sign-split (Option C, spec §2.2): under MPI each (m, r-sign)
  // class launches separately so slab sub-sweeps can run in that sign's
  // rank order; blocks of the other sign return (block-uniform, no
  // divergence). Serial passes wanted_sign=0 (both).
  if (wanted_sign != 0 && (forward_r ? 1 : -1) != wanted_sign) {
    return;
  }
  const int reflected = reflect_dir[d];
  const int reflected_z = reflect_z_dir[d];
  const double alpha_prev = alpha_edge[iz * (n_phi_half + 1) + m];
  const double alpha_next = alpha_edge[iz * (n_phi_half + 1) + m + 1];

  // Oriented owned-slab clamp (Option C KBA, spec §2.1): mesh range
  // [i_begin, i_end) maps to the diagonal coordinate range [a_lo, a_hi]
  // per r-sweep direction; the serial window (0, nr) reproduces the full
  // iteration space exactly (a_from == a_min, local_base == 0).
  const int a_lo = forward_r ? i_begin : (nr - i_end);
  const int a_hi = forward_r ? (i_end - 1) : (nr - 1 - i_begin);
  for (int stage = 0; stage <= nr + nz - 2; ++stage) {
    const int a_min = (stage > (nz - 1)) ? (stage - (nz - 1)) : 0;
    const int a_max = (stage < (nr - 1)) ? stage : (nr - 1);
    const int a_from = a_min > a_lo ? a_min : a_lo;
    const int a_to = a_max < a_hi ? a_max : a_hi;
    const int count = a_to - a_from + 1;
    const int local_base = a_from - a_min;
    for (int local = threadIdx.x; local < count; local += blockDim.x) {
      int i = 0;
      int j = 0;
      wavefront_ij(stage, local_base + local, forward_r, forward_z, nr, nz,
                   &i, &j);
      const int c = cell_index_2d(i, j, nz);
      const std::size_t cg = cell_group_index(c, g, n_groups);
      const std::size_t workspace_index =
          direction_cell_index(g, d, c, n_dirs, n_cells);

      double r_left = 0.0;
      double r_right = 0.0;
      double dz = 0.0;
      double area_r_left = 0.0;
      double area_r_right = 0.0;
      double area_z = 0.0;
      double V = 0.0;
      cell_geometry_2d(node_r,
                       node_z,
                       vol,
                       i,
                       j,
                       nz,
                       &r_left,
                       &r_right,
                       &dz,
                       &area_r_left,
                       &area_r_right,
                       &area_z,
                       &V);

      const double incoming_r = read_incoming_r(r_face_dir,
                                                reflect_axis,
                                                reflect_outer_r,
                                                g,
                                                d,
                                                reflected,
                                                i,
                                                j,
                                                nr,
                                                nz,
                                                n_dirs,
                                                n_cells,
                                                forward_r,
                                                abs_mr,
                                                r_outer_boundary);
      const double incoming_z = read_incoming_z(z_face_dir,
                                                reflect_z_bottom,
                                                reflect_z_top,
                                                g,
                                                d,
                                                reflected_z,
                                                i,
                                                j,
                                                nr,
                                                nz,
                                                n_dirs,
                                                n_cells,
                                                forward_z,
                                                abs_mz,
                                                z_bottom_boundary,
                                                z_top_boundary,
                                                marshak_flux_erg_per_cm2_s);

      const double dA = area_r_right - area_r_left;
      const double G = dA / fmax(w, 1.0e-300);
      const double angular_in_coeff = G * alpha_prev;
      const double angular_out_coeff = G * alpha_next;
      const std::size_t edge_index = phi_edge_index_2d(g, c, iz, n_cells, n_polar);
      const double edge_in = phi_edge[edge_index];
      const double area_r_down = forward_r ? area_r_right : area_r_left;
      const double area_z_down = area_z;
      const double sigma_t =
          fmax(nonnegative_finite(sigma_a[cg]) + nonnegative_finite(sigma_s[cg]),
               kSigmaFloor);
      const double source =
          0.5 * nonnegative_finite(source_emission[cg]) +
          0.5 * nonnegative_finite(sigma_s[cg]) *
              fmax(finite_or_zero(scalar_flux[cg]), 0.0);
      const double denom = sigma_t * V + 2.0 * abs_mr * area_r_down +
                           2.0 * abs_mz * area_z_down +
                           2.0 * angular_out_coeff;
      double average = 0.0;
      if (denom > 0.0 && sn_finite(denom)) {
        const double numer =
            V * source + abs_mr * (area_r_left + area_r_right) * incoming_r +
            abs_mz * (2.0 * area_z) * incoming_z +
            (angular_in_coeff + angular_out_coeff) * edge_in;
        average = numer / denom;
      }
      average = fmax(finite_or_zero(average), 0.0);

      double outgoing_r = (abs_mr > 0.0) ? (2.0 * average - incoming_r) : 0.0;
      double outgoing_z = (abs_mz > 0.0) ? (2.0 * average - incoming_z) : 0.0;
      double edge_out = 2.0 * average - edge_in;
      if (!sn_finite(outgoing_r) || outgoing_r < 0.0) {
        tally_sn_fixup(radial_fixup_count,
                       radial_fixup_artificial_abs,
                       cg,
                       outgoing_r);
        outgoing_r = 0.0;
      }
      if (!sn_finite(outgoing_z) || outgoing_z < 0.0) {
        tally_sn_fixup(radial_fixup_count,
                       radial_fixup_artificial_abs,
                       cg,
                       outgoing_z);
        outgoing_z = 0.0;
      }
      if (!sn_finite(edge_out) || edge_out < 0.0) {
        tally_sn_fixup(angular_fixup_count,
                       angular_fixup_artificial_abs,
                       cg,
                       edge_out);
        edge_out = 0.0;
      }

      r_face_dir[workspace_index] = outgoing_r;
      z_face_dir[workspace_index] = outgoing_z;
      phi_edge[edge_index] = edge_out;
      D_avg[workspace_index] = average;
      write_reflect_axis_if_needed(
          reflect_axis, g, d, i, j, nz, n_dirs, forward_r, outgoing_r);
      write_reflect_outer_r_if_needed(reflect_outer_r,
                                      g,
                                      d,
                                      i,
                                      j,
                                      nr,
                                      nz,
                                      n_dirs,
                                      forward_r,
                                      r_outer_boundary,
                                      outgoing_r);
      write_reflect_z_if_needed(reflect_z_bottom,
                                reflect_z_top,
                                g,
                                d,
                                i,
                                j,
                                nr,
                                nz,
                                n_dirs,
                                forward_z,
                                z_bottom_boundary,
                                z_top_boundary,
                                outgoing_z);
      accumulate_unique_face_psi(D_face_psi,
                                 g,
                                 d,
                                 i,
                                 j,
                                 nr,
                                 nz,
                                 n_dirs,
                                 n_faces,
                                 forward_r,
                                 forward_z,
                                 abs_mr,
                                 abs_mz,
                                 outgoing_r,
                                 outgoing_z,
                                 incoming_r,
                                 incoming_z,
                                 r_outer_boundary,
                                 z_bottom_boundary,
                                 z_top_boundary);
    }
    __syncthreads();
  }
}

__device__ inline double lc_unit_first_exp_moment(const double tau,
                                                 const double exp_tau) {
  if (!(tau > 0.0) || !sn_finite(tau)) {
    return 0.5;
  }
  if (tau < 1.0e-3) {
    const double t2 = tau * tau;
    const double t3 = t2 * tau;
    const double t4 = t2 * t2;
    const double t5 = t4 * tau;
    const double t6 = t3 * t3;
    return 0.5 - tau / 3.0 + t2 / 8.0 - t3 / 30.0 + t4 / 144.0 -
           t5 / 840.0 + t6 / 5760.0;
  }
  if (tau >= 745.0) {
    return 1.0 / (tau * tau);
  }
  return (1.0 - exp_tau * (1.0 + tau)) / (tau * tau);
}

__device__ inline double lc_r_weighted_exp_average(const LCWeights& lc,
                                                   const double tau,
                                                   const double r0,
                                                   const double dr_ds,
                                                   const double length) {
  const double denom = r0 + 0.5 * dr_ds * length;
  if (!(denom > 1.0e-300) || !(length > 0.0)) {
    return fmin(fmax(lc.A, 0.0), 1.0);
  }
  const double m1 = lc_unit_first_exp_moment(tau, lc.E);
  const double value = (r0 * lc.A + dr_ds * length * m1) / denom;
  return sn_finite(value) ? fmin(fmax(value, 0.0), 1.0) : fmin(fmax(lc.A, 0.0), 1.0);
}

__device__ inline double lc_r_weighted_mean_distance(const double r0,
                                                     const double dr_ds,
                                                     const double length) {
  const double denom = r0 + 0.5 * dr_ds * length;
  if (!(denom > 1.0e-300) || !(length > 0.0)) {
    return 0.5 * length;
  }
  const double value =
      length * (0.5 * r0 + (dr_ds * length) / 3.0) / denom;
  return sn_finite(value) ? fmax(value, 0.0) : (0.5 * length);
}

__global__ void sn_sweep_2d_rz_lc_kernel(
    const double* __restrict__ sigma_a,
    const double* __restrict__ sigma_s,
    const double* __restrict__ source_emission,
    const double* __restrict__ scalar_flux,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const double* __restrict__ vol,
    const double* __restrict__ mu_r,
    const double* __restrict__ mu_z,
    const double* __restrict__ weights,
    const int* __restrict__ reflect_dir,
    const int* __restrict__ reflect_z_dir,
    const double* __restrict__ alpha_edge,
    const int* __restrict__ sorted_dir,
    double* __restrict__ phi_edge,
    double* __restrict__ r_face_dir,
    double* __restrict__ z_face_dir,
    double* __restrict__ reflect_axis,
    double* __restrict__ reflect_outer_r,
    double* __restrict__ reflect_z_bottom,
    double* __restrict__ reflect_z_top,
    double* __restrict__ D_avg,
    double* __restrict__ D_face_psi,
    unsigned long long* __restrict__ radial_fixup_count,
    double* __restrict__ radial_fixup_artificial_abs,
    unsigned long long* __restrict__ angular_fixup_count,
    double* __restrict__ angular_fixup_artificial_abs,
    const int nr,
    const int nz,
    const int n_groups,
    const int n_dirs,
    const int m,
    const int n_phi_half,
    const int n_polar,
    const int r_outer_boundary,
    const int z_bottom_boundary,
    const int z_top_boundary,
    const double marshak_flux_erg_per_cm2_s,
    const int i_begin,
    const int i_end,
    const int wanted_sign) {
  const int iz = blockIdx.x;
  const int g = blockIdx.y;
  if (g >= n_groups || iz >= n_polar || m >= n_phi_half) {
    return;
  }
  const int d = sorted_dir[iz * n_phi_half + m];
  if (d < 0 || d >= n_dirs) {
    return;
  }
  const int n_cells = nr * nz;
  const int n_faces = n_faces_2d(nr, nz);
  const double mr = finite_or_zero(mu_r[d]);
  const double mz = finite_or_zero(mu_z[d]);
  const double w = weights[d];
  const double abs_mr = fabs(mr);
  const double abs_mz = fabs(mz);
  const bool forward_r = mr >= 0.0;
  const bool forward_z = mz >= 0.0;
  // KBA sign-split (Option C, spec §2.2): under MPI each (m, r-sign)
  // class launches separately so slab sub-sweeps can run in that sign's
  // rank order; blocks of the other sign return (block-uniform, no
  // divergence). Serial passes wanted_sign=0 (both).
  if (wanted_sign != 0 && (forward_r ? 1 : -1) != wanted_sign) {
    return;
  }
  const int reflected = reflect_dir[d];
  const int reflected_z = reflect_z_dir[d];
  const double alpha_prev = alpha_edge[iz * (n_phi_half + 1) + m];
  const double alpha_next = alpha_edge[iz * (n_phi_half + 1) + m + 1];

  // Oriented owned-slab clamp (Option C KBA, spec §2.1): mesh range
  // [i_begin, i_end) maps to the diagonal coordinate range [a_lo, a_hi]
  // per r-sweep direction; the serial window (0, nr) reproduces the full
  // iteration space exactly (a_from == a_min, local_base == 0).
  const int a_lo = forward_r ? i_begin : (nr - i_end);
  const int a_hi = forward_r ? (i_end - 1) : (nr - 1 - i_begin);
  for (int stage = 0; stage <= nr + nz - 2; ++stage) {
    const int a_min = (stage > (nz - 1)) ? (stage - (nz - 1)) : 0;
    const int a_max = (stage < (nr - 1)) ? stage : (nr - 1);
    const int a_from = a_min > a_lo ? a_min : a_lo;
    const int a_to = a_max < a_hi ? a_max : a_hi;
    const int count = a_to - a_from + 1;
    const int local_base = a_from - a_min;
    for (int local = threadIdx.x; local < count; local += blockDim.x) {
      int i = 0;
      int j = 0;
      wavefront_ij(stage, local_base + local, forward_r, forward_z, nr, nz,
                   &i, &j);
      const int c = cell_index_2d(i, j, nz);
      const std::size_t cg = cell_group_index(c, g, n_groups);
      const std::size_t workspace_index =
          direction_cell_index(g, d, c, n_dirs, n_cells);

      double r_left = 0.0;
      double r_right = 0.0;
      double dz = 0.0;
      double area_r_left = 0.0;
      double area_r_right = 0.0;
      double area_z = 0.0;
      double V = 0.0;
      cell_geometry_2d(node_r,
                       node_z,
                       vol,
                       i,
                       j,
                       nz,
                       &r_left,
                       &r_right,
                       &dz,
                       &area_r_left,
                       &area_r_right,
                       &area_z,
                       &V);

      const double incoming_r = read_incoming_r(r_face_dir,
                                                reflect_axis,
                                                reflect_outer_r,
                                                g,
                                                d,
                                                reflected,
                                                i,
                                                j,
                                                nr,
                                                nz,
                                                n_dirs,
                                                n_cells,
                                                forward_r,
                                                abs_mr,
                                                r_outer_boundary);
      const double incoming_z = read_incoming_z(z_face_dir,
                                                reflect_z_bottom,
                                                reflect_z_top,
                                                g,
                                                d,
                                                reflected_z,
                                                i,
                                                j,
                                                nr,
                                                nz,
                                                n_dirs,
                                                n_cells,
                                                forward_z,
                                                abs_mz,
                                                z_bottom_boundary,
                                                z_top_boundary,
                                                marshak_flux_erg_per_cm2_s);

      const double dA = area_r_right - area_r_left;
      const double G = dA / fmax(w, 1.0e-300);
      const double angular_in_coeff = G * alpha_prev;
      const double angular_out_coeff = G * alpha_next;
      const std::size_t edge_index = phi_edge_index_2d(g, c, iz, n_cells, n_polar);
      const double edge_in = phi_edge[edge_index];
      const double radial_in_coeff =
          abs_mr * (forward_r ? area_r_left : area_r_right);
      const double radial_out_coeff =
          abs_mr * (forward_r ? area_r_right : area_r_left);
      const double axial_coeff = abs_mz * area_z;
      const double projected = radial_out_coeff + axial_coeff + angular_out_coeff;
      const double length =
          (projected > 1.0e-300 && V > 0.0) ? (V / projected) : 0.0;
      double incoming = 0.0;
      if (projected > 1.0e-300) {
        incoming =
            (radial_in_coeff * incoming_r + axial_coeff * incoming_z +
             angular_in_coeff * edge_in) /
            projected;
      }
      incoming = fmax(finite_or_zero(incoming), 0.0);

      const double sigma_t =
          fmax(nonnegative_finite(sigma_a[cg]) + nonnegative_finite(sigma_s[cg]),
               kSigmaFloor);
      const double source =
          0.5 * nonnegative_finite(source_emission[cg]) +
          0.5 * nonnegative_finite(sigma_s[cg]) *
              fmax(finite_or_zero(scalar_flux[cg]), 0.0);
      const double r0 =
          (abs_mr > 0.0) ? (forward_r ? r_left : r_right)
                         : (0.5 * (r_left + r_right));
      const double dr_ds = forward_r ? abs_mr : -abs_mr;

      double average = incoming;
      double outgoing = incoming;
      if (length > 0.0 && sigma_t > 0.0) {
        const double tau = sigma_t * length;
        const LCWeights lc = compute_lc_weights(tau);
        const double q_avg = source / sigma_t;
        const double exp_avg =
            lc_r_weighted_exp_average(lc, tau, r0, dr_ds, length);
        average = exp_avg * incoming + (1.0 - exp_avg) * q_avg;
        outgoing = lc.E * incoming + (1.0 - lc.E) * q_avg;
      } else if (length > 0.0) {
        const double mean_s = lc_r_weighted_mean_distance(r0, dr_ds, length);
        average = incoming + source * mean_s;
        outgoing = incoming + source * length;
      }
      const double outgoing_raw = outgoing;
      if (!sn_finite(outgoing_raw) || outgoing_raw < 0.0) {
        tally_sn_fixup(radial_fixup_count,
                       radial_fixup_artificial_abs,
                       cg,
                       outgoing_raw,
                       2ULL);
        tally_sn_fixup(angular_fixup_count,
                       angular_fixup_artificial_abs,
                       cg,
                       outgoing_raw);
      }
      average = fmax(finite_or_zero(average), 0.0);
      outgoing = fmax(finite_or_zero(outgoing), 0.0);

      const double outgoing_r = outgoing;
      const double outgoing_z = outgoing;
      const double edge_out = outgoing;
      r_face_dir[workspace_index] = outgoing_r;
      z_face_dir[workspace_index] = outgoing_z;
      phi_edge[edge_index] = nonnegative_finite(edge_out);
      D_avg[workspace_index] = average;
      write_reflect_axis_if_needed(
          reflect_axis, g, d, i, j, nz, n_dirs, forward_r, outgoing_r);
      write_reflect_outer_r_if_needed(reflect_outer_r,
                                      g,
                                      d,
                                      i,
                                      j,
                                      nr,
                                      nz,
                                      n_dirs,
                                      forward_r,
                                      r_outer_boundary,
                                      outgoing_r);
      write_reflect_z_if_needed(reflect_z_bottom,
                                reflect_z_top,
                                g,
                                d,
                                i,
                                j,
                                nr,
                                nz,
                                n_dirs,
                                forward_z,
                                z_bottom_boundary,
                                z_top_boundary,
                                outgoing_z);
      accumulate_unique_face_psi(D_face_psi,
                                 g,
                                 d,
                                 i,
                                 j,
                                 nr,
                                 nz,
                                 n_dirs,
                                 n_faces,
                                 forward_r,
                                 forward_z,
                                 abs_mr,
                                 abs_mz,
                                 outgoing_r,
                                 outgoing_z,
                                 incoming_r,
                                 incoming_z,
                                 r_outer_boundary,
                                 z_bottom_boundary,
                                 z_top_boundary);
    }
    __syncthreads();
  }
}

__global__ void sn_reduce_cell_outputs_2d_kernel(
    const double* __restrict__ D_avg,
    const double* __restrict__ mu_r,
    const double* __restrict__ mu_z,
    const double* __restrict__ weights,
    double* __restrict__ new_scalar_flux,
    double* __restrict__ E_out,
    double* __restrict__ P_rr_out,
    double* __restrict__ F_z_out,
    double* __restrict__ P_zz_out,
    double* __restrict__ phi_sweep_out,
    const int n_cells,
    const int n_groups,
    const int n_dirs) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int c = idx / n_groups;
  const int g = idx - c * n_groups;
  double phi = 0.0;
  double P_rr = 0.0;
  double F_z = 0.0;
  double P_zz = 0.0;
  for (int d = 0; d < n_dirs; ++d) {
    const double average = D_avg[direction_cell_index(g, d, c, n_dirs, n_cells)];
    const double w = weights[d];
    const double mr = mu_r[d];
    const double mz = mu_z[d];
    phi += w * average;
    P_rr += w * mr * mr * average / tenryu::core::constants::c_light;
    F_z += w * mz * average;
    P_zz += w * mz * mz * average / tenryu::core::constants::c_light;
  }
  const std::size_t cg = cell_group_index(c, g, n_groups);
  new_scalar_flux[cg] = phi;
  E_out[cg] = phi / tenryu::core::constants::c_light;
  P_rr_out[cg] = P_rr;
  if (F_z_out != nullptr) {
    F_z_out[cg] = F_z;
  }
  if (P_zz_out != nullptr) {
    P_zz_out[cg] = P_zz;
  }
  if (phi_sweep_out != nullptr) {
    phi_sweep_out[cg] = phi;
  }
}

__global__ void sn_reduce_face_flux_2d_kernel(
    const double* __restrict__ D_face_psi,
    const double* __restrict__ mu_r,
    const double* __restrict__ mu_z,
    const double* __restrict__ weights,
    double* __restrict__ face_flux_raw,
    const int nr,
    const int nz,
    const int n_groups,
    const int n_dirs) {
  const int n_faces = n_faces_2d(nr, nz);
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_faces * n_groups;
  if (idx >= total) {
    return;
  }
  const int face = idx / n_groups;
  const int g = idx - face * n_groups;
  const bool radial_face = face < n_r_faces_2d(nr, nz);
  double flux = 0.0;
  for (int d = 0; d < n_dirs; ++d) {
    const double psi = D_face_psi[direction_face_index(g, d, face, n_dirs, n_faces)];
    const double mu = radial_face ? mu_r[d] : mu_z[d];
    flux += weights[d] * mu * psi;
  }
  face_flux_raw[face_group_index_2d(face, g, n_groups)] = flux;
}

__global__ void sn_dsa_setup_2d_kernel(const double* __restrict__ phi_sweep,
                                       const double* __restrict__ phi_old,
                                       const double* __restrict__ sigma_s,
                                       double* __restrict__ rhs,
                                       double* __restrict__ delta,
                                       int n_cells,
                                       int n_groups) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const double sigma_scat = nonnegative_finite(sigma_s[idx]);
  rhs[idx] = sigma_scat *
             (finite_or_zero(phi_sweep[idx]) - finite_or_zero(phi_old[idx]));
  delta[idx] = 0.0;
}

__global__ void sn_dsa_jacobi_2d_kernel(
    const double* __restrict__ sigma_a,
    const double* __restrict__ sigma_s,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const double* __restrict__ vol,
    const double* __restrict__ rhs,
    const double* __restrict__ delta_old,
    double* __restrict__ delta_new,
    int nr,
    int nz,
    int n_groups,
    double dt,
    int reflect_z_boundary) {
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
  double dz = 0.0;
  double area_r_left = 0.0;
  double area_r_right = 0.0;
  double area_z = 0.0;
  double V = 0.0;
  cell_geometry_2d(node_r,
                   node_z,
                   vol,
                   i,
                   j,
                   nz,
                   &r_left,
                   &r_right,
                   &dz,
                   &area_r_left,
                   &area_r_right,
                   &area_z,
                   &V);
  const int c_n00 = node_index_2d(i, j, nz);
  const int c_n10 = node_index_2d(i + 1, j, nz);
  const int c_n01 = node_index_2d(i, j + 1, nz);
  const int c_n11 = node_index_2d(i + 1, j + 1, nz);
  const double z_bottom =
      0.5 * (finite_or_zero(node_z[c_n00]) + finite_or_zero(node_z[c_n10]));
  const double z_top =
      0.5 * (finite_or_zero(node_z[c_n01]) + finite_or_zero(node_z[c_n11]));
  const double sig_a = nonnegative_finite(sigma_a[idx]);
  const double sig_t =
      fmax(sig_a + nonnegative_finite(sigma_s[idx]), kDsaSigmaFloor);
  const double D_c = 1.0 / (3.0 * sig_t);
  double diag = V + dt * tenryu::core::constants::c_light * sig_a * V;
  double accum = finite_or_zero(rhs[idx]) * V;
  if (i > 0) {
    const int cn = cell_index_2d(i - 1, j, nz);
    const int ng = cn * n_groups + g;
    const double sig_n =
        fmax(nonnegative_finite(sigma_a[ng]) + nonnegative_finite(sigma_s[ng]),
             kDsaSigmaFloor);
    const double Df = positive_harmonic_mean_device(D_c, 1.0 / (3.0 * sig_n));
    double dummy = 0.0;
    double rn_left = 0.0;
    double rn_right = 0.0;
    cell_geometry_2d(node_r, node_z, vol, i - 1, j, nz,
                     &rn_left, &rn_right, &dummy, &dummy, &dummy, &dummy, &dummy);
    const double rc = 0.5 * (r_left + r_right);
    const double rn = 0.5 * (rn_left + rn_right);
    const double coef = dt * area_r_left * Df / fmax(rc - rn, 1.0e-300);
    diag += coef;
    accum += coef * finite_or_zero(delta_old[ng]);
  }
  if (i + 1 < nr) {
    const int cn = cell_index_2d(i + 1, j, nz);
    const int ng = cn * n_groups + g;
    const double sig_n =
        fmax(nonnegative_finite(sigma_a[ng]) + nonnegative_finite(sigma_s[ng]),
             kDsaSigmaFloor);
    const double Df = positive_harmonic_mean_device(D_c, 1.0 / (3.0 * sig_n));
    double dummy = 0.0;
    double rn_left = 0.0;
    double rn_right = 0.0;
    cell_geometry_2d(node_r, node_z, vol, i + 1, j, nz,
                     &rn_left, &rn_right, &dummy, &dummy, &dummy, &dummy, &dummy);
    const double rc = 0.5 * (r_left + r_right);
    const double rn = 0.5 * (rn_left + rn_right);
    const double coef = dt * area_r_right * Df / fmax(rn - rc, 1.0e-300);
    diag += coef;
    accum += coef * finite_or_zero(delta_old[ng]);
  }
  if (j > 0) {
    const int cn = cell_index_2d(i, j - 1, nz);
    const int ng = cn * n_groups + g;
    const double sig_n =
        fmax(nonnegative_finite(sigma_a[ng]) + nonnegative_finite(sigma_s[ng]),
             kDsaSigmaFloor);
    const double Df = positive_harmonic_mean_device(D_c, 1.0 / (3.0 * sig_n));
    const int n00 = node_index_2d(i, j - 1, nz);
    const int n10 = node_index_2d(i + 1, j - 1, nz);
    const int n01 = node_index_2d(i, j, nz);
    const int n11 = node_index_2d(i + 1, j, nz);
    const double znb = 0.5 * (finite_or_zero(node_z[n00]) + finite_or_zero(node_z[n10]));
    const double znt = 0.5 * (finite_or_zero(node_z[n01]) + finite_or_zero(node_z[n11]));
    const double zc = 0.5 * (z_bottom + z_top);
    const double zn = 0.5 * (znb + znt);
    const double coef = dt * area_z * Df / fmax(zc - zn, 1.0e-300);
    diag += coef;
    accum += coef * finite_or_zero(delta_old[ng]);
  } else if (reflect_z_boundary == 0) {
    diag += dt * area_z * 0.5 * tenryu::core::constants::c_light;
  }
  if (j + 1 < nz) {
    const int cn = cell_index_2d(i, j + 1, nz);
    const int ng = cn * n_groups + g;
    const double sig_n =
        fmax(nonnegative_finite(sigma_a[ng]) + nonnegative_finite(sigma_s[ng]),
             kDsaSigmaFloor);
    const double Df = positive_harmonic_mean_device(D_c, 1.0 / (3.0 * sig_n));
    const int n00 = node_index_2d(i, j + 1, nz);
    const int n10 = node_index_2d(i + 1, j + 1, nz);
    const int n01 = node_index_2d(i, j + 2, nz);
    const int n11 = node_index_2d(i + 1, j + 2, nz);
    const double znb = 0.5 * (finite_or_zero(node_z[n00]) + finite_or_zero(node_z[n10]));
    const double znt = 0.5 * (finite_or_zero(node_z[n01]) + finite_or_zero(node_z[n11]));
    const double zc = 0.5 * (z_bottom + z_top);
    const double zn = 0.5 * (znb + znt);
    const double coef = dt * area_z * Df / fmax(zn - zc, 1.0e-300);
    diag += coef;
    accum += coef * finite_or_zero(delta_old[ng]);
  } else if (reflect_z_boundary == 0) {
    diag += dt * area_z * 0.5 * tenryu::core::constants::c_light;
  }
  delta_new[idx] =
      (isfinite(diag) && diag > 0.0) ? (accum / diag) : 0.0;
  if (!isfinite(delta_new[idx])) {
    delta_new[idx] = 0.0;
  }
}

__global__ void sn_dsa_apply_2d_kernel(const double* __restrict__ delta,
                                       double* __restrict__ phi,
                                       int n_total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_total) {
    phi[idx] = fmax(finite_or_zero(phi[idx]) + finite_or_zero(delta[idx]), 0.0);
  }
}

void apply_dsa_2d(const SNTransport2DRZGPUInputs& in,
                  double* phi_old,
                  double* phi_new,
                  parallel::DeviceArray& d_rhs,
                  parallel::DeviceArray& d_delta_a,
                  parallel::DeviceArray& d_delta_b,
                  const bool reflect_z_boundary) {
  const int n_cells = in.nr * in.nz;
  const int n_total = n_cells * in.n_groups;
  if (n_total <= 0) {
    return;
  }
  const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(n_total);
  d_rhs.resize(bytes);
  d_delta_a.resize(bytes);
  d_delta_b.resize(bytes);
  const int grid = (n_total + kReduceBlock - 1) / kReduceBlock;
  sn_dsa_setup_2d_kernel<<<grid, kReduceBlock>>>(
      phi_new, phi_old, in.sigma_s, d_rhs.as<double>(), d_delta_a.as<double>(),
      n_cells, in.n_groups);
  cuda_check(cudaGetLastError(), "SN GPU 2D DSA setup launch failed");
  const bool dsa_mpi_active =
      in.mpi_part != nullptr && in.mpi_part->n_ranks > 1;
  double* old_delta = d_delta_a.as<double>();
  double* new_delta = d_delta_b.as<double>();
  for (int iter = 0; iter < kDsaMaxIterations; ++iter) {
    if (dsa_mpi_active) {
      // Refresh the iterate's ghost i-planes: the Jacobi stencil reads
      // r-neighbors across the slab boundary. Iteration count is FIXED,
      // so ranks stay in lockstep with no collectives (KBA spec §3);
      // non-owned interior cells iterate on stale-finite inputs (Option
      // C) and are never consumed.
      parallel::exchange_cell_strips_scaled(*in.mpi_part, *in.mpi_bufs,
                                            old_delta, in.n_groups, 7);
    }
    sn_dsa_jacobi_2d_kernel<<<grid, kReduceBlock>>>(
        in.sigma_a,
        in.sigma_s,
        in.node_r,
        in.node_z,
        in.vol,
        d_rhs.as<double>(),
        old_delta,
        new_delta,
        in.nr,
        in.nz,
        in.n_groups,
        1.0,
        reflect_z_boundary ? 1 : 0);
    cuda_check(cudaGetLastError(), "SN GPU 2D DSA Jacobi launch failed");
    std::swap(old_delta, new_delta);
  }
  sn_dsa_apply_2d_kernel<<<grid, kReduceBlock>>>(old_delta, phi_new, n_total);
  cuda_check(cudaGetLastError(), "SN GPU 2D DSA apply launch failed");
}

void apply_dsa_2d_stream(cudaStream_t stream,
                         const SNTransport2DRZGPUInputs& in,
                         double* phi_old,
                         double* phi_new,
                         parallel::DeviceArray& d_rhs,
                         parallel::DeviceArray& d_delta_a,
                         parallel::DeviceArray& d_delta_b,
                         const bool reflect_z_boundary) {
  // Graph-captured variant: no MPI exchange can live inside a capture;
  // the solve gates graphs OFF under MPI, so this path must be serial.
  TENRYU_ASSERT(in.mpi_part == nullptr || in.mpi_part->n_ranks <= 1,
                "SN GPU 2D DSA stream/graph path must not run under MPI");
  const int n_cells = in.nr * in.nz;
  const int n_total = n_cells * in.n_groups;
  if (n_total <= 0) {
    return;
  }
  const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(n_total);
  // The caller pre-sizes DSA scratch before capture.
  const int grid = (n_total + kReduceBlock - 1) / kReduceBlock;
  sn_dsa_setup_2d_kernel<<<grid, kReduceBlock, 0, stream>>>(
      phi_new, phi_old, in.sigma_s, d_rhs.as<double>(), d_delta_a.as<double>(),
      n_cells, in.n_groups);
  cuda_check(cudaGetLastError(), "SN GPU 2D DSA setup launch failed");
  double* old_delta = d_delta_a.as<double>();
  double* new_delta = d_delta_b.as<double>();
  for (int iter = 0; iter < kDsaMaxIterations; ++iter) {
    sn_dsa_jacobi_2d_kernel<<<grid, kReduceBlock, 0, stream>>>(
        in.sigma_a,
        in.sigma_s,
        in.node_r,
        in.node_z,
        in.vol,
        d_rhs.as<double>(),
        old_delta,
        new_delta,
        in.nr,
        in.nz,
        in.n_groups,
        1.0,
        reflect_z_boundary ? 1 : 0);
    cuda_check(cudaGetLastError(), "SN GPU 2D DSA Jacobi launch failed");
    std::swap(old_delta, new_delta);
  }
  sn_dsa_apply_2d_kernel<<<grid, kReduceBlock, 0, stream>>>(
      old_delta, phi_new, n_total);
  cuda_check(cudaGetLastError(), "SN GPU 2D DSA apply launch failed");
}

double convergence_error(const double* phi_new,
                         const double* phi_old,
                         const int n_total,
                         parallel::DeviceArray& reduction) {
  if (n_total <= 0) {
    return 0.0;
  }
  const int blocks = (n_total + kReduceBlock - 1) / kReduceBlock;
  const std::size_t bytes = static_cast<std::size_t>(blocks) * sizeof(double);
  reduction.resize(bytes);
  auto* block_error = reduction.as<double>();
  sn_convergence_reduce_kernel<<<blocks,
                                 kReduceBlock,
                                 kReduceBlock * sizeof(double)>>>(
      phi_new, phi_old, block_error, n_total);
  cuda_check(cudaGetLastError(), "SN GPU convergence reduction launch failed");
  std::vector<double> host(static_cast<std::size_t>(blocks), 0.0);
  cuda_check(cudaMemcpy(host.data(), block_error, bytes, cudaMemcpyDeviceToHost),
             "SN GPU convergence reduction copy failed");
  double max_error = 0.0;
  for (int b = 0; b < blocks; ++b) {
    max_error = std::max(max_error, host[static_cast<std::size_t>(b)]);
  }
  return max_error;
}

void finalize_chi(const double* E,
                  const double* P_rr,
                  double* chi,
                  const int n_total) {
  if (n_total <= 0) {
    return;
  }
  const int blocks = (n_total + kReduceBlock - 1) / kReduceBlock;
  sn_chi_kernel<<<blocks, kReduceBlock>>>(E, P_rr, chi, n_total);
  cuda_check(cudaGetLastError(), "SN GPU chi kernel launch failed");
}

void finalize_chi_z(const double* E,
                    const double* P_zz,
                    double* chi_z,
                    const int n_total) {
  if (n_total <= 0) {
    return;
  }
  const int blocks = (n_total + kReduceBlock - 1) / kReduceBlock;
  sn_chi_z_kernel<<<blocks, kReduceBlock>>>(E, P_zz, chi_z, n_total);
  cuda_check(cudaGetLastError(), "SN GPU chi_z kernel launch failed");
}

void update_E_from_scalar_flux(const double* scalar_flux,
                               double* E_out,
                               const int n_total) {
  if (n_total <= 0) {
    return;
  }
  const int blocks = (n_total + kReduceBlock - 1) / kReduceBlock;
  sn_scalar_flux_to_E_kernel<<<blocks, kReduceBlock>>>(scalar_flux, E_out, n_total);
  cuda_check(cudaGetLastError(), "SN GPU scalar flux E update launch failed");
}

#ifdef TENRYU_DEBUG_CPU_FALLBACK
[[deprecated("Debug fallback only; production S_N material coupling uses GPU Newton")]]
void solve_material_temperature_newton_cpu(
    const SNMaterialCouplingGPUInputs& in,
    const int n_cells,
    const int n_groups,
    const double temperature_floor) {
  TENRYU_ASSERT(in.rho != nullptr, "SN IMEX Newton requires rho");
  TENRYU_ASSERT(in.ee != nullptr, "SN IMEX Newton requires ee");
  TENRYU_ASSERT(in.Te_old != nullptr, "SN IMEX Newton requires Te_old");
  TENRYU_ASSERT(in.planck_table_cpu != nullptr,
                "SN IMEX Newton requires CPU Planck table");
  TENRYU_ASSERT(in.planck_table_cpu->n_groups() == n_groups,
                "SN IMEX Newton Planck group count mismatch");

  const std::size_t n_cells_us = static_cast<std::size_t>(n_cells);
  const std::size_t n_total_us = n_cells_us * static_cast<std::size_t>(n_groups);
  const std::size_t cell_bytes = sizeof(double) * n_cells_us;
  const std::size_t total_bytes = sizeof(double) * n_total_us;

  std::vector<double> host_E(n_total_us, 0.0);
  std::vector<double> host_T(n_cells_us, 0.0);
  std::vector<double> host_T_old(n_cells_us, 0.0);
  std::vector<double> host_sigma_a(n_total_us, 0.0);
  std::vector<double> host_rho(n_cells_us, 0.0);
  std::vector<double> host_cv_e;
  if (in.cv_e != nullptr) {
    host_cv_e.resize(n_cells_us, 0.0);
  }

  cuda_check(cudaMemcpy(host_E.data(), in.E_out, total_bytes, cudaMemcpyDeviceToHost),
             "SN IMEX Newton copy E failed");
  cuda_check(cudaMemcpy(host_T.data(), in.Te, cell_bytes, cudaMemcpyDeviceToHost),
             "SN IMEX Newton copy Te failed");
  cuda_check(cudaMemcpy(host_T_old.data(),
                        in.Te_old,
                        cell_bytes,
                        cudaMemcpyDeviceToHost),
             "SN IMEX Newton copy Te_old failed");
  cuda_check(cudaMemcpy(host_sigma_a.data(),
                        in.sigma_a,
                        total_bytes,
                        cudaMemcpyDeviceToHost),
             "SN IMEX Newton copy sigma_a failed");
  cuda_check(cudaMemcpy(host_rho.data(), in.rho, cell_bytes, cudaMemcpyDeviceToHost),
             "SN IMEX Newton copy rho failed");
  if (!host_cv_e.empty()) {
    cuda_check(cudaMemcpy(host_cv_e.data(),
                          in.cv_e,
                          cell_bytes,
                          cudaMemcpyDeviceToHost),
               "SN IMEX Newton copy cv_e failed");
  }

  for (int c = 0; c < n_cells; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    host_T_old[c_us] =
        std::max(finite_or_zero(host_T_old[c_us]), temperature_floor);
    host_T[c_us] = host_T_old[c_us];
  }

  const double cv_e_const = finite_or_zero(in.cv_e_const);
  const double Cv_e_const = finite_or_zero(in.Cv_e_const);
  const double dt = in.dt;
  for (int iter = 0; iter < kSnTemperatureNewtonMaxIterations; ++iter) {
    double max_relative_update = 0.0;
    for (int c = 0; c < n_cells; ++c) {
      const std::size_t c_us = static_cast<std::size_t>(c);
      const double rho_c = nonnegative_finite(host_rho[c_us]);
      const double cv_mass =
          (!host_cv_e.empty()) ? finite_or_zero(host_cv_e[c_us]) : 0.0;
      const double Cv =
          sn_volume_heat_capacity(rho_c, cv_mass, cv_e_const, Cv_e_const);
      if (!(Cv > 0.0)) {
        continue;
      }

      const double T_old = host_T_old[c_us];
      const double T = std::max(finite_or_zero(host_T[c_us]), temperature_floor);
      double F = Cv * (T - T_old) / dt;
      double dF = Cv / dt;
      const int base = c * n_groups;
      for (int g = 0; g < n_groups; ++g) {
        const std::size_t cg = static_cast<std::size_t>(base + g);
        const double sigma = nonnegative_finite(host_sigma_a[cg]);
        if (!(sigma > 0.0)) {
          continue;
        }
        const double E = nonnegative_finite(host_E[cg]);
        const double c_sigma = tenryu::core::constants::c_light * sigma;
        const double b_g =
            std::max(in.planck_table_cpu->interpolate_b_host(g, T), 0.0);
        const double T4 = safe_temperature_pow4(T);
        const double B = tenryu::core::constants::a_eV * T4 * b_g;
        F -= c_sigma * (E - B);
        if (T > 0.0) {
          const double dBdT =
              4.0 * tenryu::core::constants::a_eV * (T4 / T) * b_g;
          dF += c_sigma * dBdT;
        }
      }

      if (!(sn_finite(F) && sn_finite(dF)) || !(dF > 0.0)) {
        continue;
      }
      const double dT = -F / dF;
      double T_next = T + dT;
      if (!sn_finite(T_next)) {
        T_next = temperature_floor;
      }
      T_next = std::max(T_next, temperature_floor);
      const double relative_update =
          std::abs(T_next - T) / std::max(std::abs(T_next), temperature_floor);
      max_relative_update = std::max(max_relative_update, relative_update);
      host_T[c_us] = T_next;
    }
    if (max_relative_update < kSnTemperatureNewtonTol) {
      break;
    }
  }

  cuda_check(cudaMemcpy(in.Te, host_T.data(), cell_bytes, cudaMemcpyHostToDevice),
             "SN IMEX Newton copy updated Te failed");
  const int blocks = (n_cells + kReduceBlock - 1) / kReduceBlock;
  sn_update_material_energy_kernel<<<blocks, kReduceBlock>>>(
      in.ee,
      in.Te,
      in.Te_old,
      in.rho,
      in.cv_e,
      n_cells,
      temperature_floor,
      cv_e_const,
      Cv_e_const);
  cuda_check(cudaGetLastError(), "SN IMEX Newton material energy update launch failed");
}
#endif

}  // namespace

SNTransportGPUResult solve_sn_transport_1d_gpu(
    const SNTransport1DGPUInputs& in,
    const SNTransportGPUConfig& config) {
  TENRYU_ASSERT(in.sigma_a != nullptr, "SN GPU 1D requires sigma_a");
  TENRYU_ASSERT(in.sigma_s != nullptr, "SN GPU 1D requires sigma_s");
  TENRYU_ASSERT(in.source_emission != nullptr, "SN GPU 1D requires source_emission");
  TENRYU_ASSERT(in.node_r != nullptr, "SN GPU 1D requires node_r");
  TENRYU_ASSERT(in.vol != nullptr, "SN GPU 1D requires vol");
  TENRYU_ASSERT(in.E_out != nullptr, "SN GPU 1D requires E_out");
  TENRYU_ASSERT(in.P_rr_out != nullptr, "SN GPU 1D requires P_rr_out");
  TENRYU_ASSERT(in.chi_out != nullptr, "SN GPU 1D requires chi_out");
  TENRYU_ASSERT(in.n_cells >= 0, "SN GPU 1D requires n_cells >= 0");
  TENRYU_ASSERT(in.n_groups >= 0, "SN GPU 1D requires n_groups >= 0");
  TENRYU_ASSERT(config.n_angles > 0 && (config.n_angles % 2) == 0,
                "SN GPU 1D n_angles must be positive and even");
  TENRYU_ASSERT(config.max_iterations >= 0,
                "SN GPU 1D max_iterations must be >= 0");
  TENRYU_ASSERT(config.convergence_tol >= 0.0,
                "SN GPU 1D convergence_tol must be >= 0");

  SNTransportGPUResult result{};
  result.n_directions = config.n_angles;
  const int n_total = in.n_cells * in.n_groups;
  if (in.n_cells == 0 || in.n_groups == 0) {
    result.converged = true;
    return result;
  }

  std::vector<double> mu;
  std::vector<double> weight;
  compute_gauss_legendre(config.n_angles, mu, weight);
  const std::vector<double> alpha = angular_coefficients(mu, weight);

  parallel::DeviceArray d_mu;
  parallel::DeviceArray d_weight;
  parallel::DeviceArray d_alpha;
  parallel::DeviceArray d_phi_a;
  parallel::DeviceArray d_phi_b;
  parallel::DeviceArray d_reduce;
  upload_vector(d_mu, mu, "SN GPU 1D copy mu failed");
  upload_vector(d_weight, weight, "SN GPU 1D copy weights failed");
  upload_vector(d_alpha, alpha, "SN GPU 1D copy alpha failed");

  const std::size_t phi_bytes = sizeof(double) * static_cast<std::size_t>(n_total);
  d_phi_a.resize(phi_bytes);
  d_phi_b.resize(phi_bytes);
  cuda_check(cudaMemset(d_phi_a.ptr, 0, phi_bytes), "SN GPU 1D zero phi failed");
  cuda_check(cudaMemset(d_phi_b.ptr, 0, phi_bytes), "SN GPU 1D zero phi_next failed");

  double* phi_old = d_phi_a.as<double>();
  double* phi_new = d_phi_b.as<double>();
  const int max_iterations = std::max(config.max_iterations, 1);
  const std::size_t shared_bytes =
      (static_cast<std::size_t>(in.n_cells) +
       static_cast<std::size_t>(config.n_angles)) *
      sizeof(double);
  for (int iter = 0; iter < max_iterations; ++iter) {
    sn_sweep_1d_kernel<<<in.n_groups, 1, shared_bytes>>>(
        in.sigma_a,
        in.sigma_s,
        in.source_emission,
        phi_old,
        in.node_r,
        in.vol,
        d_mu.as<double>(),
        d_weight.as<double>(),
        d_alpha.as<double>(),
        in.psi_bar,
        phi_new,
        in.E_out,
        in.P_rr_out,
        in.n_cells,
        in.n_groups,
        config.n_angles,
        0.0);
    cuda_check(cudaGetLastError(), "SN GPU 1D sweep launch failed");

    const double max_error =
        convergence_error(phi_new, phi_old, n_total, d_reduce);
    result.iterations = iter + 1;
    result.convergence_error = max_error;
    result.converged = result.convergence_error <= config.convergence_tol;
    if (result.converged) {
      break;
    }
    std::swap(phi_old, phi_new);
  }

  finalize_chi(in.E_out, in.P_rr_out, in.chi_out, n_total);
  return result;
}

struct Sn2dSweepGraphKey {
  int nr = 0;
  int nz = 0;
  int n_groups = 0;
  int n_dirs = 0;
  int n_phi_half = 0;
  bool linear_characteristic = false;
  bool dsa_enabled = false;
  bool has_face_flux = false;
  const double* phi_a = nullptr;
  const double* phi_b = nullptr;
  const double* sigma_a = nullptr;
  const double* sigma_s = nullptr;
  const double* source_emission = nullptr;
  const double* r_face = nullptr;
  const double* dsa_rhs = nullptr;
  bool operator==(const Sn2dSweepGraphKey&) const = default;
};

struct Sn2dSweepGraphState {
  cudaStream_t stream = nullptr;
  cudaGraphExec_t exec[2] = {nullptr, nullptr};
  bool capture_failed = false;
  Sn2dSweepGraphKey key;
};

Sn2dSweepGraphState& sn2d_sweep_graph_state() {
  static Sn2dSweepGraphState state;
  return state;
}

SNTransportGPUResult solve_sn_transport_2d_rz_gpu(
    const SNTransport2DRZGPUInputs& in,
    const SNTransportGPUConfig& config) {
  TENRYU_ASSERT(in.sigma_a != nullptr, "SN GPU 2D requires sigma_a");
  TENRYU_ASSERT(in.sigma_s != nullptr, "SN GPU 2D requires sigma_s");
  TENRYU_ASSERT(in.source_emission != nullptr, "SN GPU 2D requires source_emission");
  TENRYU_ASSERT(in.node_r != nullptr, "SN GPU 2D requires node_r");
  TENRYU_ASSERT(in.node_z != nullptr, "SN GPU 2D requires node_z");
  TENRYU_ASSERT(in.vol != nullptr, "SN GPU 2D requires vol");
  TENRYU_ASSERT(in.E_out != nullptr, "SN GPU 2D requires E_out");
  TENRYU_ASSERT(in.P_rr_out != nullptr, "SN GPU 2D requires P_rr_out");
  TENRYU_ASSERT(in.chi_out != nullptr, "SN GPU 2D requires chi_out");
  TENRYU_ASSERT(in.nr >= 0 && in.nz >= 0, "SN GPU 2D requires non-negative mesh size");
  TENRYU_ASSERT(in.n_groups >= 0, "SN GPU 2D requires n_groups >= 0");
  TENRYU_ASSERT(config.n_angles > 0 && (config.n_angles % 2) == 0,
                "SN GPU 2D n_angles must be positive and even");
  TENRYU_ASSERT(config.max_iterations >= 0,
                "SN GPU 2D max_iterations must be >= 0");
  TENRYU_ASSERT(config.convergence_tol >= 0.0,
                "SN GPU 2D convergence_tol must be >= 0");
  // MPI (Option C r-slab, M18c slice-3): rank-uniform outer decisions and
  // the graph policy hinge on this flag; missing context under MPI is
  // fail-loud (a silently serial rank would desynchronize the Allreduce
  // and deadlock the job).
  const bool mpi_active =
      in.mpi_part != nullptr && in.mpi_part->n_ranks > 1;
  if (mpi_active) {
    TENRYU_ASSERT(in.mpi_bufs != nullptr,
                  "SN GPU 2D MPI requires CommBuffers");
    TENRYU_ASSERT(in.mpi_c_begin >= 0 && in.mpi_c_end > in.mpi_c_begin &&
                      in.mpi_c_end <= in.nr * in.nz,
                  "SN GPU 2D MPI requires a valid owned cell window");
    TENRYU_ASSERT(in.nz > 0 && in.mpi_c_begin % in.nz == 0 &&
                      in.mpi_c_end % in.nz == 0,
                  "SN GPU 2D KBA requires an i-plane-aligned cell window");
  }

  SNTransportGPUResult result{};
  const int n_cells = in.nr * in.nz;
  const int n_total = n_cells * in.n_groups;
  if (n_cells == 0 || in.n_groups == 0) {
    result.converged = true;
    return result;
  }

  std::vector<double> mu_r;
  std::vector<double> mu_z;
  std::vector<double> weight;
  std::vector<int> reflect_dir;
  std::vector<int> reflect_z_dir;
  std::vector<double> alpha_edge;
  std::vector<int> sorted_dir_by_iz_m;
  std::vector<OctantRange2D> octants;
  compute_product_quadrature_2d(config.n_angles,
                                mu_r,
                                mu_z,
                                weight,
                                reflect_dir,
                                reflect_z_dir,
                                alpha_edge,
                                sorted_dir_by_iz_m,
                                octants);
  const int n_dirs = static_cast<int>(mu_r.size());
  const int n_polar = config.n_angles;
  const int n_phi_half = std::max(config.n_angles / 2, 1);
  result.n_directions = n_dirs;

  parallel::DeviceArray d_mu_r{"sn2dsweep:mu_r"};
  parallel::DeviceArray d_mu_z{"sn2dsweep:mu_z"};
  parallel::DeviceArray d_weight{"sn2dsweep:weight"};
  parallel::DeviceArray d_reflect_dir{"sn2dsweep:reflect_dir"};
  parallel::DeviceArray d_reflect_z_dir{"sn2dsweep:reflect_z_dir"};
  parallel::DeviceArray d_alpha_edge{"sn2dsweep:alpha_edge"};
  parallel::DeviceArray d_sorted_dir{"sn2dsweep:sorted_dir"};
  parallel::DeviceArray d_phi_edge{"sn2dsweep:phi_edge"};
  parallel::DeviceArray d_phi_a{"sn2dsweep:phi_a"};
  parallel::DeviceArray d_phi_b{"sn2dsweep:phi_b"};
  parallel::DeviceArray d_r_face{"sn2dsweep:r_face"};
  parallel::DeviceArray d_z_face{"sn2dsweep:z_face"};
  parallel::DeviceArray d_D_avg{"sn2dsweep:D_avg"};
  parallel::DeviceArray d_D_face_psi{"sn2dsweep:D_face_psi"};
  parallel::DeviceArray d_reflect_axis{"sn2dsweep:reflect_axis"};
  parallel::DeviceArray d_reflect_outer_r{"sn2dsweep:reflect_outer_r"};
  parallel::DeviceArray d_reflect_z_bottom{"sn2dsweep:reflect_z_bottom"};
  parallel::DeviceArray d_reflect_z_top{"sn2dsweep:reflect_z_top"};
  parallel::DeviceArray d_dsa_rhs{"sn2dsweep:dsa_rhs"};
  parallel::DeviceArray d_dsa_delta_a{"sn2dsweep:dsa_delta_a"};
  parallel::DeviceArray d_dsa_delta_b{"sn2dsweep:dsa_delta_b"};
  parallel::DeviceArray d_reduce{"sn2dsweep:reduce"};
  upload_vector(d_mu_r, mu_r, "SN GPU 2D copy mu_r failed");
  upload_vector(d_mu_z, mu_z, "SN GPU 2D copy mu_z failed");
  upload_vector(d_weight, weight, "SN GPU 2D copy weights failed");
  upload_vector(d_reflect_dir, reflect_dir, "SN GPU 2D copy reflect map failed");
  upload_vector(d_reflect_z_dir, reflect_z_dir, "SN GPU 2D copy z reflect map failed");
  upload_vector(d_alpha_edge, alpha_edge, "SN GPU 2D copy alpha edge failed");
  upload_vector(d_sorted_dir, sorted_dir_by_iz_m, "SN GPU 2D copy sorted directions failed");

  const std::size_t phi_bytes = sizeof(double) * static_cast<std::size_t>(n_total);
  const int n_unique_faces = n_faces_2d(in.nr, in.nz);
  const std::size_t direction_cell_bytes =
      sizeof(double) * static_cast<std::size_t>(in.n_groups) *
      static_cast<std::size_t>(n_dirs) * static_cast<std::size_t>(n_cells);
  const std::size_t phi_edge_bytes =
      sizeof(double) * static_cast<std::size_t>(in.n_groups) *
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_polar);
  const std::size_t direction_face_bytes =
      sizeof(double) * static_cast<std::size_t>(in.n_groups) *
      static_cast<std::size_t>(n_dirs) * static_cast<std::size_t>(n_unique_faces);
  const std::size_t face_flux_bytes =
      sizeof(double) * static_cast<std::size_t>(n_unique_faces) *
      static_cast<std::size_t>(in.n_groups);
  d_phi_a.resize(phi_bytes);
  d_phi_b.resize(phi_bytes);
  d_r_face.resize(direction_cell_bytes);
  d_z_face.resize(direction_cell_bytes);
  d_phi_edge.resize(phi_edge_bytes);
  d_D_avg.resize(direction_cell_bytes);
  d_D_face_psi.resize(direction_face_bytes);
  d_reflect_axis.resize(sizeof(double) * static_cast<std::size_t>(in.n_groups) *
                        static_cast<std::size_t>(n_dirs) *
                        static_cast<std::size_t>(in.nz));
  d_reflect_outer_r.resize(sizeof(double) *
                           static_cast<std::size_t>(in.n_groups) *
                           static_cast<std::size_t>(n_dirs) *
                           static_cast<std::size_t>(in.nz));
  d_reflect_z_bottom.resize(sizeof(double) * static_cast<std::size_t>(in.n_groups) *
                            static_cast<std::size_t>(n_dirs) *
                            static_cast<std::size_t>(in.nr));
  d_reflect_z_top.resize(sizeof(double) * static_cast<std::size_t>(in.n_groups) *
                         static_cast<std::size_t>(n_dirs) *
                         static_cast<std::size_t>(in.nr));
  cuda_check(cudaMemset(d_phi_a.ptr, 0, phi_bytes), "SN GPU 2D zero phi failed");
  cuda_check(cudaMemset(d_phi_b.ptr, 0, phi_bytes), "SN GPU 2D zero phi_next failed");
  cuda_check(cudaMemset(d_reflect_axis.ptr, 0, d_reflect_axis.size),
             "SN GPU 2D zero reflect axis failed");
  cuda_check(cudaMemset(d_reflect_outer_r.ptr, 0, d_reflect_outer_r.size),
             "SN GPU 2D zero outer r reflect failed");
  cuda_check(cudaMemset(d_reflect_z_bottom.ptr, 0, d_reflect_z_bottom.size),
             "SN GPU 2D zero z bottom reflect failed");
  cuda_check(cudaMemset(d_reflect_z_top.ptr, 0, d_reflect_z_top.size),
             "SN GPU 2D zero z top reflect failed");

  double* phi_old = d_phi_a.as<double>();
  double* phi_new = d_phi_b.as<double>();
  const int threads = std::max(32, std::min(config.block_threads_2d, 256));
  const int max_iterations = std::max(config.max_iterations, 1);
  const auto sanitize_z_boundary = [](const int code) {
    return (code == kSNBoundaryReflect || code == kSNBoundaryMarshak)
               ? code
               : kSNBoundaryVacuum;
  };
  const int z_bottom_boundary =
      config.z_boundary_reflect
          ? kSNBoundaryReflect
          : sanitize_z_boundary(config.z_bottom_boundary);
  const int z_top_boundary =
      config.z_boundary_reflect
          ? kSNBoundaryReflect
          : sanitize_z_boundary(config.z_top_boundary);
  const auto sanitize_r_outer = [](const int code) {
    return (code == kSNBoundaryReflect) ? kSNBoundaryReflect : kSNBoundaryVacuum;
  };
  const int r_outer_boundary = sanitize_r_outer(config.r_outer_boundary);
  const bool dsa_reflect_z =
      z_bottom_boundary == kSNBoundaryReflect &&
      z_top_boundary == kSNBoundaryReflect;
  // W5 (2026-07-09): the inner-iteration body between convergence checks
  // is a fixed kernel/memset sequence over solve-stable pooled pointers;
  // replay it as one instantiated graph per iteration (two parity
  // graphs, since phi_old/phi_new swap each iteration). The eager body
  // below is byte-identical to the pre-W5 loop and remains the fallback
  // (capture-failure latch + TENRYU_SN_SWEEP_NO_GRAPH=1).
  int sweep_graph_parity_base = 0;
  const auto record_sweep_block = [&](double* p_old,
                                      double* p_new,
                                      cudaStream_t s) {
    cuda_check(cudaMemsetAsync(d_r_face.ptr, 0, direction_cell_bytes, s),
               "SN GPU 2D zero directional r face failed");
    cuda_check(cudaMemsetAsync(d_z_face.ptr, 0, direction_cell_bytes, s),
               "SN GPU 2D zero directional z face failed");
    cuda_check(cudaMemsetAsync(d_D_avg.ptr, 0, direction_cell_bytes, s),
               "SN GPU 2D zero directional averages failed");
    cuda_check(cudaMemsetAsync(d_D_face_psi.ptr, 0, direction_face_bytes, s),
               "SN GPU 2D zero directional face psi failed");
    if (in.face_flux_raw != nullptr) {
      cuda_check(cudaMemsetAsync(in.face_flux_raw, 0, face_flux_bytes, s),
                 "SN GPU 2D zero unique face flux failed");
    }
    cuda_check(cudaMemsetAsync(d_phi_edge.ptr, 0, phi_edge_bytes, s),
               "SN GPU 2D zero angular phi edge failed");
    for (int m = 0; m < n_phi_half; ++m) {
      const dim3 sweep_grid(n_polar, in.n_groups);
      if (config.linear_characteristic) {
        sn_sweep_2d_rz_lc_kernel<<<sweep_grid, threads, 0, s>>>(
            in.sigma_a,
            in.sigma_s,
            in.source_emission,
            p_old,
            in.node_r,
            in.node_z,
            in.vol,
            d_mu_r.as<double>(),
            d_mu_z.as<double>(),
            d_weight.as<double>(),
            d_reflect_dir.as<int>(),
            d_reflect_z_dir.as<int>(),
            d_alpha_edge.as<double>(),
            d_sorted_dir.as<int>(),
            d_phi_edge.as<double>(),
            d_r_face.as<double>(),
            d_z_face.as<double>(),
            d_reflect_axis.as<double>(),
            d_reflect_outer_r.as<double>(),
            d_reflect_z_bottom.as<double>(),
            d_reflect_z_top.as<double>(),
            d_D_avg.as<double>(),
            d_D_face_psi.as<double>(),
            in.radial_fixup_count,
            in.radial_fixup_artificial_abs,
            in.angular_fixup_count,
            in.angular_fixup_artificial_abs,
            in.nr,
            in.nz,
            in.n_groups,
            n_dirs,
            m,
            n_phi_half,
            n_polar,
            r_outer_boundary,
            z_bottom_boundary,
            z_top_boundary,
            config.marshak_flux_erg_per_cm2_s,
            0,
            in.nr,
            0);
        cuda_check(cudaGetLastError(), "SN GPU 2D LC sweep launch failed");
      } else {
        sn_sweep_2d_kernel<<<sweep_grid, threads, 0, s>>>(
            in.sigma_a,
            in.sigma_s,
            in.source_emission,
            p_old,
            in.node_r,
            in.node_z,
            in.vol,
            d_mu_r.as<double>(),
            d_mu_z.as<double>(),
            d_weight.as<double>(),
            d_reflect_dir.as<int>(),
            d_reflect_z_dir.as<int>(),
            d_alpha_edge.as<double>(),
            d_sorted_dir.as<int>(),
            d_phi_edge.as<double>(),
            d_r_face.as<double>(),
            d_z_face.as<double>(),
            d_reflect_axis.as<double>(),
            d_reflect_outer_r.as<double>(),
            d_reflect_z_bottom.as<double>(),
            d_reflect_z_top.as<double>(),
            d_D_avg.as<double>(),
            d_D_face_psi.as<double>(),
            in.radial_fixup_count,
            in.radial_fixup_artificial_abs,
            in.angular_fixup_count,
            in.angular_fixup_artificial_abs,
            in.nr,
            in.nz,
            in.n_groups,
            n_dirs,
            m,
            n_phi_half,
            n_polar,
            r_outer_boundary,
            z_bottom_boundary,
            z_top_boundary,
            config.marshak_flux_erg_per_cm2_s,
            0,
            in.nr,
            0);
        cuda_check(cudaGetLastError(), "SN GPU 2D sweep launch failed");
      }
    }

    const int cell_reduce_grid = (n_total + kReduceBlock - 1) / kReduceBlock;
    sn_reduce_cell_outputs_2d_kernel<<<cell_reduce_grid, kReduceBlock, 0, s>>>(
        d_D_avg.as<double>(),
        d_mu_r.as<double>(),
        d_mu_z.as<double>(),
        d_weight.as<double>(),
        p_new,
        in.E_out,
        in.P_rr_out,
        in.F_z_out,
        in.P_zz_out,
        in.phi_sweep_out,
        n_cells,
        in.n_groups,
        n_dirs);
    cuda_check(cudaGetLastError(), "SN GPU 2D cell reduction launch failed");

    if (in.face_flux_raw != nullptr) {
      const int face_total = n_unique_faces * in.n_groups;
      const int face_reduce_grid = (face_total + kReduceBlock - 1) / kReduceBlock;
      sn_reduce_face_flux_2d_kernel<<<face_reduce_grid, kReduceBlock, 0, s>>>(
          d_D_face_psi.as<double>(),
          d_mu_r.as<double>(),
          d_mu_z.as<double>(),
          d_weight.as<double>(),
          in.face_flux_raw,
          in.nr,
          in.nz,
          in.n_groups,
          n_dirs);
      cuda_check(cudaGetLastError(), "SN GPU 2D face reduction launch failed");
    }

    if (config.dsa_enabled) {
      apply_dsa_2d_stream(s,
                          in,
                          p_old,
                          p_new,
                          d_dsa_rhs,
                          d_dsa_delta_a,
                          d_dsa_delta_b,
                          dsa_reflect_z);
    }
  };
  const auto ensure_sweep_graphs = [&](double* p0, double* p1) -> bool {
    auto& gs = sn2d_sweep_graph_state();
    if (gs.capture_failed) {
      return false;
    }
    // Pre-size the DSA scratch: no allocation is legal during capture.
    if (config.dsa_enabled) {
      const std::size_t dsa_bytes =
          sizeof(double) * static_cast<std::size_t>(n_total);
      d_dsa_rhs.resize(dsa_bytes);
      d_dsa_delta_a.resize(dsa_bytes);
      d_dsa_delta_b.resize(dsa_bytes);
    }
    Sn2dSweepGraphKey key;
    key.nr = in.nr;
    key.nz = in.nz;
    key.n_groups = in.n_groups;
    key.n_dirs = n_dirs;
    key.n_phi_half = n_phi_half;
    key.linear_characteristic = config.linear_characteristic;
    key.dsa_enabled = config.dsa_enabled;
    key.has_face_flux = in.face_flux_raw != nullptr;
    key.phi_a = p0;
    key.phi_b = p1;
    key.sigma_a = in.sigma_a;
    key.sigma_s = in.sigma_s;
    key.source_emission = in.source_emission;
    key.r_face = d_r_face.as<double>();
    key.dsa_rhs = d_dsa_rhs.as<double>();
    if (gs.exec[0] != nullptr && gs.exec[1] != nullptr) {
      if (key == gs.key) {
        sweep_graph_parity_base = 0;
        return true;
      }
      Sn2dSweepGraphKey swapped = key;
      std::swap(swapped.phi_a, swapped.phi_b);
      if (swapped == gs.key) {
        sweep_graph_parity_base = 1;
        return true;
      }
    }
    sweep_graph_parity_base = 0;
    if (gs.stream == nullptr && cudaStreamCreate(&gs.stream) != cudaSuccess) {
      gs.stream = nullptr;
      gs.capture_failed = true;
      static_cast<void>(cudaGetLastError());
      core::log_warning(
          "SN2D sweep graph: stream creation failed; using the eager loop");
      return false;
    }
    for (int p = 0; p < 2; ++p) {
      if (gs.exec[p] != nullptr) {
        static_cast<void>(cudaGraphExecDestroy(gs.exec[p]));
        gs.exec[p] = nullptr;
      }
    }
    bool ok = true;
    for (int p = 0; p < 2 && ok; ++p) {
      double* o = (p == 0) ? p0 : p1;
      double* n2 = (p == 0) ? p1 : p0;
      cudaGraph_t graph = nullptr;
      cudaError_t err =
          cudaStreamBeginCapture(gs.stream, cudaStreamCaptureModeThreadLocal);
      if (err == cudaSuccess) {
        record_sweep_block(o, n2, gs.stream);
        err = cudaStreamEndCapture(gs.stream, &graph);
      }
      ok = err == cudaSuccess && graph != nullptr;
      if (ok) {
        ok = cudaGraphInstantiate(&gs.exec[p], graph, 0) == cudaSuccess &&
             gs.exec[p] != nullptr;
      }
      if (graph != nullptr) {
        static_cast<void>(cudaGraphDestroy(graph));
      }
    }
    if (!ok) {
      for (int p = 0; p < 2; ++p) {
        if (gs.exec[p] != nullptr) {
          static_cast<void>(cudaGraphExecDestroy(gs.exec[p]));
          gs.exec[p] = nullptr;
        }
      }
      gs.key = Sn2dSweepGraphKey{};
      gs.capture_failed = true;
      static_cast<void>(cudaGetLastError());
      core::log_warning(
          "SN2D sweep graph capture failed; using the eager loop");
      return false;
    }
    gs.key = key;
    return true;
  };
  // CUDA graph replay is incompatible with mid-sweep MPI exchanges (KBA
  // face-plane pipeline, DSA ghost refresh) and with the rank-uniform
  // convergence Allreduce; force the eager loop under MPI (mirrors the
  // FLD-2D CG graph policy).
  const bool sweep_graph_enabled =
      !sn2d_sweep_graph_disabled() && !mpi_active;
  double* const phi_parity0 = phi_old;
  double* const phi_parity1 = phi_new;
  for (int iter = 0; iter < max_iterations; ++iter) {
    // At iter 0 (phi_old, phi_new) == (phi_parity0, phi_parity1), so
    // exec[0] bakes that ordering; each swap flips to exec[iter & 1].
    if (sweep_graph_enabled &&
        ensure_sweep_graphs(phi_parity0, phi_parity1)) {
      auto& gs = sn2d_sweep_graph_state();
      cuda_check(cudaGraphLaunch(gs.exec[(iter & 1) ^ sweep_graph_parity_base], gs.stream),
                 "SN2D sweep graph launch failed");
    } else {
    cuda_check(cudaMemset(d_r_face.ptr, 0, direction_cell_bytes),
               "SN GPU 2D zero directional r face failed");
    cuda_check(cudaMemset(d_z_face.ptr, 0, direction_cell_bytes),
               "SN GPU 2D zero directional z face failed");
    cuda_check(cudaMemset(d_D_avg.ptr, 0, direction_cell_bytes),
               "SN GPU 2D zero directional averages failed");
    cuda_check(cudaMemset(d_D_face_psi.ptr, 0, direction_face_bytes),
               "SN GPU 2D zero directional face psi failed");
    if (in.face_flux_raw != nullptr) {
      cuda_check(cudaMemset(in.face_flux_raw, 0, face_flux_bytes),
                 "SN GPU 2D zero unique face flux failed");
    }
    cuda_check(cudaMemset(d_phi_edge.ptr, 0, phi_edge_bytes),
               "SN GPU 2D zero angular phi edge failed");
    const auto launch_sweep_m = [&](const int m, const int i_begin,
                                    const int i_end,
                                    const int wanted_sign) {
      const dim3 sweep_grid(n_polar, in.n_groups);
      if (config.linear_characteristic) {
        sn_sweep_2d_rz_lc_kernel<<<sweep_grid, threads>>>(
            in.sigma_a,
            in.sigma_s,
            in.source_emission,
            phi_old,
            in.node_r,
            in.node_z,
            in.vol,
            d_mu_r.as<double>(),
            d_mu_z.as<double>(),
            d_weight.as<double>(),
            d_reflect_dir.as<int>(),
            d_reflect_z_dir.as<int>(),
            d_alpha_edge.as<double>(),
            d_sorted_dir.as<int>(),
            d_phi_edge.as<double>(),
            d_r_face.as<double>(),
            d_z_face.as<double>(),
            d_reflect_axis.as<double>(),
            d_reflect_outer_r.as<double>(),
            d_reflect_z_bottom.as<double>(),
            d_reflect_z_top.as<double>(),
            d_D_avg.as<double>(),
            d_D_face_psi.as<double>(),
            in.radial_fixup_count,
            in.radial_fixup_artificial_abs,
            in.angular_fixup_count,
            in.angular_fixup_artificial_abs,
            in.nr,
            in.nz,
            in.n_groups,
            n_dirs,
            m,
            n_phi_half,
            n_polar,
            r_outer_boundary,
            z_bottom_boundary,
            z_top_boundary,
            config.marshak_flux_erg_per_cm2_s,
            i_begin,
            i_end,
            wanted_sign);
        cuda_check(cudaGetLastError(), "SN GPU 2D LC sweep launch failed");
      } else {
        sn_sweep_2d_kernel<<<sweep_grid, threads>>>(
            in.sigma_a,
            in.sigma_s,
            in.source_emission,
            phi_old,
            in.node_r,
            in.node_z,
            in.vol,
            d_mu_r.as<double>(),
            d_mu_z.as<double>(),
            d_weight.as<double>(),
            d_reflect_dir.as<int>(),
            d_reflect_z_dir.as<int>(),
            d_alpha_edge.as<double>(),
            d_sorted_dir.as<int>(),
            d_phi_edge.as<double>(),
            d_r_face.as<double>(),
            d_z_face.as<double>(),
            d_reflect_axis.as<double>(),
            d_reflect_outer_r.as<double>(),
            d_reflect_z_bottom.as<double>(),
            d_reflect_z_top.as<double>(),
            d_D_avg.as<double>(),
            d_D_face_psi.as<double>(),
            in.radial_fixup_count,
            in.radial_fixup_artificial_abs,
            in.angular_fixup_count,
            in.angular_fixup_artificial_abs,
            in.nr,
            in.nz,
            in.n_groups,
            n_dirs,
            m,
            n_phi_half,
            n_polar,
            r_outer_boundary,
            z_bottom_boundary,
            z_top_boundary,
            config.marshak_flux_erg_per_cm2_s,
            i_begin,
            i_end,
            wanted_sign);
        cuda_check(cudaGetLastError(), "SN GPU 2D sweep launch failed");
      }
    };
    if (!mpi_active) {
      for (int m = 0; m < n_phi_half; ++m) {
        launch_sweep_m(m, 0, in.nr, 0);
      }
    } else {
      // KBA r-slab pipeline (Option C, spec §2.3): per (m, r-sign) the
      // slab sub-sweeps run in that sign's upwind->downwind rank order;
      // between hops the boundary i-planes of r_face_dir for the
      // participating (g, d) blocks are exchanged. The symmetric strip
      // exchange also moves a not-yet-swept plane the other way, which
      // the receiver never reads for this sign; later hops re-deliver
      // already-final planes (idempotent). v1 correctness schedule:
      // blocking hops, no cross-m overlap.
      const int kba_rank = in.mpi_part->rank;
      const int kba_ranks = in.mpi_part->n_ranks;
      const int kba_i_begin = in.mpi_c_begin / in.nz;
      const int kba_i_end = in.mpi_c_end / in.nz;
      const auto class_sign_of = [&](const int iz, const int m) {
        const int d =
            sorted_dir_by_iz_m[static_cast<std::size_t>(iz) *
                                   static_cast<std::size_t>(n_phi_half) +
                               static_cast<std::size_t>(m)];
        if (d < 0 || d >= n_dirs) {
          return 0;  // inactive row
        }
        return (mu_r[static_cast<std::size_t>(d)] >= 0.0) ? 1 : -1;
      };
      for (int m = 0; m < n_phi_half; ++m) {
        for (int sign = -1; sign <= 1; sign += 2) {
          bool class_nonempty = false;
          for (int iz = 0; iz < n_polar; ++iz) {
            if (class_sign_of(iz, m) == sign) {
              class_nonempty = true;
              break;
            }
          }
          if (!class_nonempty) {
            continue;  // rank-uniform: the host quadrature is identical
          }
          const int my_pos =
              (sign > 0) ? kba_rank : (kba_ranks - 1 - kba_rank);
          for (int hop = 0; hop < kba_ranks; ++hop) {
            if (hop == my_pos) {
              launch_sweep_m(m, kba_i_begin, kba_i_end, sign);
              cuda_check(cudaDeviceSynchronize(),
                         "SN GPU 2D KBA sub-sweep sync failed");
            }
            if (hop + 1 >= kba_ranks) {
              continue;
            }
            for (int iz = 0; iz < n_polar; ++iz) {
              if (class_sign_of(iz, m) != sign) {
                continue;
              }
              const int d =
                  sorted_dir_by_iz_m[static_cast<std::size_t>(iz) *
                                         static_cast<std::size_t>(
                                             n_phi_half) +
                                     static_cast<std::size_t>(m)];
              for (int g = 0; g < in.n_groups; ++g) {
                parallel::exchange_cell_strips_scaled(
                    *in.mpi_part, *in.mpi_bufs,
                    d_r_face.as<double>() +
                        (static_cast<std::size_t>(g) *
                             static_cast<std::size_t>(n_dirs) +
                         static_cast<std::size_t>(d)) *
                            static_cast<std::size_t>(n_cells),
                    1, 8);
              }
            }
          }
        }
      }
    }

    const int cell_reduce_grid = (n_total + kReduceBlock - 1) / kReduceBlock;
    sn_reduce_cell_outputs_2d_kernel<<<cell_reduce_grid, kReduceBlock>>>(
        d_D_avg.as<double>(),
        d_mu_r.as<double>(),
        d_mu_z.as<double>(),
        d_weight.as<double>(),
        phi_new,
        in.E_out,
        in.P_rr_out,
        in.F_z_out,
        in.P_zz_out,
        in.phi_sweep_out,
        n_cells,
        in.n_groups,
        n_dirs);
    cuda_check(cudaGetLastError(), "SN GPU 2D cell reduction launch failed");

    if (in.face_flux_raw != nullptr) {
      const int face_total = n_unique_faces * in.n_groups;
      const int face_reduce_grid = (face_total + kReduceBlock - 1) / kReduceBlock;
      sn_reduce_face_flux_2d_kernel<<<face_reduce_grid, kReduceBlock>>>(
          d_D_face_psi.as<double>(),
          d_mu_r.as<double>(),
          d_mu_z.as<double>(),
          d_weight.as<double>(),
          in.face_flux_raw,
          in.nr,
          in.nz,
          in.n_groups,
          n_dirs);
      cuda_check(cudaGetLastError(), "SN GPU 2D face reduction launch failed");
      if (mpi_active) {
        // Interface unique r-faces: forward-d contributions are written
        // by the lower side and backward-d by the upper side (single
        // writer per (d, face)), so each rank's reduced flux at a slab
        // interface is a partial over a DISJOINT direction set —
        // sum-complete with both neighbors. The r-face plane
        // [i*nz, (i+1)*nz) is contiguous in the face-major group-minor
        // layout.
        const int plane_elems = in.nz * in.n_groups;
        const std::size_t left_off =
            static_cast<std::size_t>(in.mpi_c_begin / in.nz) *
            static_cast<std::size_t>(plane_elems);
        const std::size_t right_off =
            static_cast<std::size_t>(in.mpi_c_end / in.nz) *
            static_cast<std::size_t>(plane_elems);
        parallel::sendrecv_add_planes(*in.mpi_part, *in.mpi_bufs,
                                      in.face_flux_raw, plane_elems,
                                      left_off, right_off, 9);
      }
    }

    if (config.dsa_enabled) {
      apply_dsa_2d(in,
                   phi_old,
                   phi_new,
                   d_dsa_rhs,
                   d_dsa_delta_a,
                   d_dsa_delta_b,
                   dsa_reflect_z);
    }
    }

    // Owned-window convergence norm + Allreduce(MAX): the owned span is
    // contiguous in the cell-major group-minor layout, so the serial
    // reduction applies unchanged via pointer offset. Rank-uniform outer
    // decisions by construction; P=1 keeps the full-span byte path.
    double max_error;
    if (mpi_active) {
      const int mpi_off = in.mpi_c_begin * in.n_groups;
      const int mpi_owned_total =
          (in.mpi_c_end - in.mpi_c_begin) * in.n_groups;
      max_error =
          parallel::Reduction(in.mpi_part->n_ranks)
              .allreduce_max(convergence_error(phi_new + mpi_off,
                                               phi_old + mpi_off,
                                               mpi_owned_total, d_reduce));
    } else {
      max_error = convergence_error(phi_new, phi_old, n_total, d_reduce);
    }
    result.iterations = iter + 1;
    result.convergence_error = max_error;
    result.converged = result.convergence_error <= config.convergence_tol;
    if (result.converged) {
      break;
    }
    std::swap(phi_old, phi_new);
  }

  finalize_chi(in.E_out, in.P_rr_out, in.chi_out, n_total);
  if (in.P_zz_out != nullptr && in.chi_z_out != nullptr) {
    finalize_chi_z(in.E_out, in.P_zz_out, in.chi_z_out, n_total);
  }
  return result;
}

SNTransportGPUResult solve_sn_material_coupling_gpu(
    const SNMaterialCouplingGPUInputs& in,
    const SNTransportGPUConfig& config) {
  TENRYU_ASSERT(in.sigma_a != nullptr,
                "SN material coupling requires sigma_a_eff");
  TENRYU_ASSERT(in.sigma_s != nullptr, "SN material coupling requires sigma_s");
  TENRYU_ASSERT(in.Te != nullptr, "SN material coupling requires Te");
  TENRYU_ASSERT(in.node_r != nullptr, "SN material coupling requires node_r");
  TENRYU_ASSERT(in.vol != nullptr, "SN material coupling requires vol");
  TENRYU_ASSERT(in.E_out != nullptr, "SN material coupling requires E_out");
  TENRYU_ASSERT(in.P_rr_out != nullptr, "SN material coupling requires P_rr_out");
  TENRYU_ASSERT(in.chi_out != nullptr, "SN material coupling requires chi_out");
  TENRYU_ASSERT(in.rad_dep != nullptr, "SN material coupling requires rad_dep");
  TENRYU_ASSERT(in.rad_emit != nullptr, "SN material coupling requires rad_emit");
  TENRYU_ASSERT(in.dim == 1 || in.dim == 2,
                "SN material coupling supports 1D_SPH and 2D_RZ");
  TENRYU_ASSERT(in.dim != 1 || in.ee != nullptr,
                "SN material coupling 1D requires ee");
  TENRYU_ASSERT(in.nr >= 0 && in.nz >= 0,
                "SN material coupling requires non-negative mesh size");
  TENRYU_ASSERT(in.n_groups >= 0,
                "SN material coupling requires n_groups >= 0");

  const int n_cells = (in.dim == 2) ? (in.nr * in.nz) : in.nr;
  const int n_total = n_cells * in.n_groups;
  SNTransportGPUResult result{};
  if (n_cells == 0 || in.n_groups == 0) {
    result.converged = true;
    return result;
  }
  if (!(in.dt > 0.0)) {
    result.converged = true;
    return result;
  }

  parallel::DeviceArray d_source_emission{"snmat:source_emission"};
  d_source_emission.resize(sizeof(double) * static_cast<std::size_t>(n_total));
  const double* source_emission_for_publish = d_source_emission.as<double>();
  const int blocks = (n_total + kReduceBlock - 1) / kReduceBlock;

  if (in.dim == 1) {
    TENRYU_ASSERT(config.n_angles > 0 && (config.n_angles % 2) == 0,
                  "SN IMEX DSA n_angles must be positive and even");

    result.n_directions = config.n_angles;
    std::vector<double> mu;
    std::vector<double> weight;
    compute_gauss_legendre(config.n_angles, mu, weight);
    const std::vector<double> alpha = angular_coefficients(mu, weight);

    parallel::DeviceArray d_mu;
    parallel::DeviceArray d_weight;
    parallel::DeviceArray d_alpha;
    parallel::DeviceArray d_phi_a;
    parallel::DeviceArray d_phi_b;
    parallel::DeviceArray d_reduce;
    upload_vector(d_mu, mu, "SN IMEX DSA copy mu failed");
    upload_vector(d_weight, weight, "SN IMEX DSA copy weights failed");
    upload_vector(d_alpha, alpha, "SN IMEX DSA copy alpha failed");

    const std::size_t phi_bytes =
        sizeof(double) * static_cast<std::size_t>(n_total);
    d_phi_a.resize(phi_bytes);
    d_phi_b.resize(phi_bytes);
    cuda_check(cudaMemset(d_phi_a.ptr, 0, phi_bytes), "SN IMEX DSA zero phi failed");
    cuda_check(cudaMemset(d_phi_b.ptr, 0, phi_bytes),
               "SN IMEX DSA zero phi_next failed");

    const std::size_t cell_bytes =
        sizeof(double) * static_cast<std::size_t>(n_cells);
    std::vector<double> host_sigma_a(static_cast<std::size_t>(n_total), 0.0);
    std::vector<double> host_sigma_s(static_cast<std::size_t>(n_total), 0.0);
    std::vector<double> host_phi_half(static_cast<std::size_t>(n_total), 0.0);
    std::vector<double> host_phi_old(static_cast<std::size_t>(n_total), 0.0);
    std::vector<double> host_delta_phi(static_cast<std::size_t>(n_total), 0.0);
    std::vector<double> host_phi_corrected(static_cast<std::size_t>(n_total), 0.0);
    std::vector<double> host_vol(static_cast<std::size_t>(n_cells), 0.0);
    std::vector<double> host_node_r(static_cast<std::size_t>(n_cells + 1), 0.0);

    cuda_check(cudaMemcpy(host_sigma_a.data(),
                          in.sigma_a,
                          sizeof(double) * host_sigma_a.size(),
                          cudaMemcpyDeviceToHost),
               "SN IMEX DSA copy sigma_a failed");
    cuda_check(cudaMemcpy(host_sigma_s.data(),
                          in.sigma_s,
                          sizeof(double) * host_sigma_s.size(),
                          cudaMemcpyDeviceToHost),
               "SN IMEX DSA copy sigma_s failed");
    cuda_check(cudaMemcpy(host_vol.data(),
                          in.vol,
                          cell_bytes,
                          cudaMemcpyDeviceToHost),
               "SN IMEX DSA copy vol failed");
    cuda_check(cudaMemcpy(host_node_r.data(),
                          in.node_r,
                          sizeof(double) * host_node_r.size(),
                          cudaMemcpyDeviceToHost),
               "SN IMEX DSA copy node_r failed");

    double* phi_old = d_phi_a.as<double>();
    double* phi_new = d_phi_b.as<double>();
    const double temperature_floor = std::max(config.temperature_floor_eV, 1.0e-12);
    const std::size_t shared_bytes =
        (static_cast<std::size_t>(in.nr) +
         static_cast<std::size_t>(config.n_angles)) *
        sizeof(double);

    sn_build_source_kernel<<<blocks, kReduceBlock>>>(
        in.sigma_a,
        in.Te,
        d_source_emission.as<double>(),
        n_cells,
        in.n_groups,
        temperature_floor,
        in.planck);
    cuda_check(cudaGetLastError(), "SN IMEX DSA source build launch failed");

    const int max_iterations = 500;
    for (int iter = 0; iter < max_iterations; ++iter) {
      sn_sweep_1d_kernel<<<in.n_groups, 1, shared_bytes>>>(
          in.sigma_a,
          in.sigma_s,
          d_source_emission.as<double>(),
          phi_old,
          in.node_r,
          in.vol,
          d_mu.as<double>(),
          d_weight.as<double>(),
          d_alpha.as<double>(),
          nullptr,
          phi_new,
          in.E_out,
          in.P_rr_out,
          in.nr,
          in.n_groups,
          config.n_angles,
          in.dt);
      cuda_check(cudaGetLastError(), "SN IMEX source iteration sweep launch failed");

      const double max_error =
          convergence_error(phi_new, phi_old, n_total, d_reduce);
      result.iterations = iter + 1;
      result.convergence_error = max_error;
      result.converged = result.convergence_error <= config.convergence_tol;
      if (result.converged) {
        break;
      }
      std::swap(phi_old, phi_new);
    }

    if (in.update_material) {
      solve_sn_material_temperature_newton_gpu(in,
                                               n_cells,
                                               in.n_groups,
                                               temperature_floor);
    }
    const std::size_t total_bytes =
        sizeof(double) * static_cast<std::size_t>(n_total);
    cuda_check(cudaMemset(in.rad_dep, 0, total_bytes),
               "SN IMEX zero rad_dep after Newton failed");
    cuda_check(cudaMemset(in.rad_emit, 0, total_bytes),
               "SN IMEX zero rad_emit after Newton failed");
    if (in.coverage != nullptr) {
      sn_fill_kernel<<<blocks, kReduceBlock>>>(in.coverage, n_total, 1.0);
      cuda_check(cudaGetLastError(), "SN IMEX coverage fill launch failed");
    }

    finalize_chi(in.E_out, in.P_rr_out, in.chi_out, n_total);
  } else {
    if (in.source_emission != nullptr) {
      source_emission_for_publish = in.source_emission;
    } else {
      sn_build_source_kernel<<<blocks, kReduceBlock>>>(
          in.sigma_a,
          in.Te,
          d_source_emission.as<double>(),
          n_cells,
          in.n_groups,
          std::max(config.temperature_floor_eV, 1.0e-12),
          in.planck);
      cuda_check(cudaGetLastError(), "SN material source build launch failed");
    }

    TENRYU_ASSERT(in.node_z != nullptr, "SN material coupling 2D requires node_z");
    SNTransport2DRZGPUInputs solve_in{};
    solve_in.sigma_a = in.sigma_a;
    solve_in.sigma_s = in.sigma_s;
    solve_in.source_emission = source_emission_for_publish;
    solve_in.node_r = in.node_r;
    solve_in.node_z = in.node_z;
    solve_in.vol = in.vol;
    solve_in.E_out = in.E_out;
    solve_in.P_rr_out = in.P_rr_out;
    solve_in.chi_out = in.chi_out;
    solve_in.F_z_out = in.F_z_out;
    solve_in.P_zz_out = in.P_zz_out;
    solve_in.chi_z_out = in.chi_z_out;
    solve_in.phi_sweep_out = in.phi_sweep_out;
    solve_in.face_flux_raw = in.face_flux_raw;
    solve_in.radial_fixup_count = in.radial_fixup_count;
    solve_in.radial_fixup_artificial_abs = in.radial_fixup_artificial_abs;
    solve_in.angular_fixup_count = in.angular_fixup_count;
    solve_in.angular_fixup_artificial_abs = in.angular_fixup_artificial_abs;
    solve_in.nr = in.nr;
    solve_in.nz = in.nz;
    solve_in.n_groups = in.n_groups;
    solve_in.mpi_part = in.mpi_part;
    solve_in.mpi_bufs = in.mpi_bufs;
    solve_in.mpi_c_begin = in.mpi_c_begin;
    solve_in.mpi_c_end = in.mpi_c_end;
    result = solve_sn_transport_2d_rz_gpu(solve_in, config);
  }

  if (in.dim == 2) {
    sn_publish_material_sources_kernel<<<blocks, kReduceBlock>>>(
        in.sigma_a,
        source_emission_for_publish,
        in.E_out,
        in.vol,
        in.rad_dep,
        in.rad_emit,
        in.coverage,
        n_cells,
        in.n_groups,
        in.dt);
    cuda_check(cudaGetLastError(), "SN material source publish launch failed");
  }
  return result;
}

void sn_reduce_cell_outputs_2d_for_test(
    const double* const D_avg,
    const double* const mu_r,
    const double* const mu_z,
    const double* const weights,
    double* const scalar_flux,
    double* const E_out,
    double* const P_rr_out,
    double* const F_z_out,
    double* const P_zz_out,
    double* const chi_out,
    double* const chi_z_out,
    const int n_cells,
    const int n_groups,
    const int n_dirs) {
  TENRYU_ASSERT(D_avg != nullptr, "SN 2D test reduction requires D_avg");
  TENRYU_ASSERT(mu_r != nullptr, "SN 2D test reduction requires mu_r");
  TENRYU_ASSERT(mu_z != nullptr, "SN 2D test reduction requires mu_z");
  TENRYU_ASSERT(weights != nullptr, "SN 2D test reduction requires weights");
  TENRYU_ASSERT(scalar_flux != nullptr, "SN 2D test reduction requires scalar_flux");
  TENRYU_ASSERT(E_out != nullptr, "SN 2D test reduction requires E_out");
  TENRYU_ASSERT(P_rr_out != nullptr, "SN 2D test reduction requires P_rr_out");
  const int n_total = n_cells * n_groups;
  if (n_total <= 0) {
    return;
  }
  const int blocks = (n_total + kReduceBlock - 1) / kReduceBlock;
  sn_reduce_cell_outputs_2d_kernel<<<blocks, kReduceBlock>>>(
      D_avg,
      mu_r,
      mu_z,
      weights,
      scalar_flux,
      E_out,
      P_rr_out,
      F_z_out,
      P_zz_out,
      nullptr,
      n_cells,
      n_groups,
      n_dirs);
  cuda_check(cudaGetLastError(), "SN GPU 2D test cell reduction launch failed");
  if (chi_out != nullptr) {
    finalize_chi(E_out, P_rr_out, chi_out, n_total);
  }
  if (P_zz_out != nullptr && chi_z_out != nullptr) {
    finalize_chi_z(E_out, P_zz_out, chi_z_out, n_total);
  }
}

}  // namespace tenryu::radiation
