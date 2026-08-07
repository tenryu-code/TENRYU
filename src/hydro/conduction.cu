#include "hydro/conduction.cuh"
#include "hydro/conduction_snb_2d.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/kernel_guard.hpp"
#include "hydro/conduction_bodies.cuh"
#include "hydro/conduction_snb_1d.cuh"
#include "hydro/eos_context.hpp"
#include "hydro/kershaw_boundary.cuh"
#include "hydro/kershaw_geometry.cuh"
#include "hydro/kershaw_stencil.cuh"
#include "hydro/hypre_solver.cuh"
#include "hydro/per_material_eos_accessors.cuh"
#include "hydro/per_material_eos_project.cuh"
#include "parallel/comm_buffers.hpp"
#include "parallel/halo_exchange.hpp"
#include "parallel/partition.hpp"
#include "parallel/reduction.hpp"

#include <type_traits>

#include "mesh/geometry_1d.cuh"

using tenryu::mesh::geometry_1d_face_area;

namespace tenryu::hydro {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kFourPi = 12.566370614359172953850573533118;
constexpr double kEvToErg = 1.6022e-12;
constexpr double kElectronMass = 9.1094e-28;
constexpr double kProtonMass = 1.6726219e-24;  // must match core::constants::proton_mass
constexpr double kDivEpsilon = 1.0e-30;
constexpr double kMaxTeForPow = 1.0e8;
constexpr double kMinEffectiveA = 1.0e-12;
constexpr double kMinEffectiveGamma = 1.0 + 1.0e-12;
constexpr int kBlockSize = 256;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline void cusparse_check(const cusparseStatus_t status, const char* message) {
  TENRYU_ASSERT(status == CUSPARSE_STATUS_SUCCESS, message);
}

std::string format_sci(const double value) {
  char buf[64] = {};
  std::snprintf(buf, sizeof(buf), "%.3e", value);
  return std::string(buf);
}

std::string format_val(const double value) {
  char buf[64] = {};
  std::snprintf(buf, sizeof(buf), "%.6g", value);
  return std::string(buf);
}

double diagonal_ratio_condition_estimate(const std::vector<double>& diag) {
  double min_diag = std::numeric_limits<double>::infinity();
  double max_diag = 0.0;
  for (const double value : diag) {
    const double mag = std::abs(value);
    if (std::isfinite(mag) && mag > 0.0) {
      min_diag = std::min(min_diag, mag);
      max_diag = std::max(max_diag, mag);
    }
  }
  if (!(std::isfinite(min_diag) && min_diag > 0.0 && max_diag > 0.0)) {
    return 0.0;
  }
  return max_diag / min_diag;
}

double tridiagonal_relative_residual(const std::vector<double>& lower,
                                     const std::vector<double>& diag,
                                     const std::vector<double>& upper,
                                     const std::vector<double>& rhs,
                                     const std::vector<double>& x) {
  const std::size_t n = diag.size();
  if (n == 0 || lower.size() != n || upper.size() != n || rhs.size() != n ||
      x.size() != n) {
    return 0.0;
  }
  long double r2 = 0.0L;
  long double b2 = 0.0L;
  for (std::size_t i = 0; i < n; ++i) {
    long double ax = static_cast<long double>(diag[i]) * x[i];
    if (i > 0) {
      ax += static_cast<long double>(lower[i]) * x[i - 1];
    }
    if (i + 1 < n) {
      ax += static_cast<long double>(upper[i]) * x[i + 1];
    }
    const long double b = rhs[i];
    const long double r = ax - b;
    r2 += r * r;
    b2 += b * b;
  }
  if (!(b2 > 0.0L)) {
    return std::sqrt(static_cast<double>(r2));
  }
  return std::sqrt(static_cast<double>(r2 / b2));
}

double kershaw_diagonal_ratio_condition_estimate(const core::State& state,
                                                 const double* d_stencil,
                                                 const double* d_rho_cv_e,
                                                 const double dt,
                                                 const int n_cells) {
  if (n_cells <= 0 || d_stencil == nullptr || d_rho_cv_e == nullptr) {
    return 0.0;
  }
  std::vector<double> stencil(static_cast<std::size_t>(n_cells) *
                              kershaw::kStencilSize);
  std::vector<double> rho_cv(static_cast<std::size_t>(n_cells));
  std::vector<double> vol(static_cast<std::size_t>(n_cells));
  cuda_check(cudaMemcpy(stencil.data(),
                        d_stencil,
                        stencil.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Conduction: copy Kershaw stencil for condition estimate failed");
  cuda_check(cudaMemcpy(rho_cv.data(),
                        d_rho_cv_e,
                        rho_cv.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Conduction: copy rho_cv for condition estimate failed");
  state.vol.copy_to_host(vol.data());

  constexpr double k4Pi = 4.0 * kPi;
  std::vector<double> diag(static_cast<std::size_t>(n_cells), 1.0);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t idx = static_cast<std::size_t>(c);
    const double denom = rho_cv[idx] * std::max(vol[idx], 1.0e-30);
    const double center =
        stencil[idx * kershaw::kStencilSize + kershaw::kC];
    if (std::isfinite(denom) && denom > 0.0 && std::isfinite(center)) {
      diag[idx] = 1.0 + dt * k4Pi * std::max(center, 0.0) / denom;
    }
  }
  return diagonal_ratio_condition_estimate(diag);
}

bool should_log_sts_subcycling() {
  static std::uint64_t call_count = 0;
  const bool emit = (call_count == 0 || (call_count % 100) == 0);
  ++call_count;
  return emit;
}

int ceil_to_int_clamped(const double value) {
  if (!std::isfinite(value)) {
    return std::numeric_limits<int>::max();
  }
  if (value <= 1.0) {
    return 1;
  }
  if (value >= static_cast<double>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  return std::max(1, static_cast<int>(std::ceil(value)));
}

bool conduction_1d_implicit_supported(const core::State& state,
                                      const parallel::PartitionInfo* part = nullptr) {
  if (state.mesh.dim != 1) {
    return false;
  }
  // OPEN-GXII-COND (design doc §6o): the implicit solve is MPI-capable via
  // full-line replication (redundant serial solve on owner-true gathered
  // lines), so support no longer depends on the rank count.
  (void)part;
  return true;
}

namespace {

// OPEN-GXII-COND replication gathers (design doc §6o): re-globalize a 1D
// line from the owned cell tiles so every rank can run the serial implicit
// solver redundantly (bitwise rank-count invariance by construction).
// Tiling mirrors parallel::split_axis / the driver line gathers.
void conduction_line_counts(const int n_cells, const int n_ranks,
                            const int per_cell, const bool node_line,
                            std::vector<int>& counts,
                            std::vector<int>& displs) {
  counts.assign(static_cast<std::size_t>(n_ranks), 0);
  displs.assign(static_cast<std::size_t>(n_ranks), 0);
  const int base = n_cells / n_ranks;
  const int rem = n_cells % n_ranks;
  for (int p = 0; p < n_ranks; ++p) {
    const int local = base + (p < rem ? 1 : 0) +
                      (node_line && p == n_ranks - 1 ? 1 : 0);
    counts[static_cast<std::size_t>(p)] = local * per_cell;
    displs[static_cast<std::size_t>(p)] =
        (p * base + std::min(p, rem)) * per_cell;
  }
}

void conduction_replicate_line(double* d_field, const int n_cells,
                               const int per_cell, const bool node_line,
                               const parallel::PartitionInfo& part,
                               parallel::Reduction& red) {
  if (d_field == nullptr || part.n_ranks <= 1 || n_cells <= 0 ||
      per_cell <= 0) {
    return;
  }
  const int total = (n_cells + (node_line ? 1 : 0)) * per_cell;
  std::vector<int> counts;
  std::vector<int> displs;
  conduction_line_counts(n_cells, part.n_ranks, per_cell, node_line, counts,
                         displs);
  std::vector<double> line(static_cast<std::size_t>(total));
  cuda_check(cudaMemcpy(line.data(), d_field,
                        static_cast<std::size_t>(total) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Conduction: replication line D2H failed");
  const int my_count = counts[static_cast<std::size_t>(part.rank)];
  const int my_displ = displs[static_cast<std::size_t>(part.rank)];
  std::vector<double> send(line.begin() + my_displ,
                           line.begin() + my_displ + my_count);
  red.allgatherv(send.data(), my_count, line.data(), counts.data(),
                 displs.data());
  cuda_check(cudaMemcpy(d_field, line.data(),
                        static_cast<std::size_t>(total) * sizeof(double),
                        cudaMemcpyHostToDevice),
             "Conduction: replication line H2D failed");
}

void conduction_replicate_void_host(std::vector<std::uint8_t>& host_field,
                                    const int n_cells,
                                    const parallel::PartitionInfo& part) {
#if TENRYU_ENABLE_MPI
  if (part.n_ranks <= 1 || n_cells <= 0 ||
      host_field.size() != static_cast<std::size_t>(n_cells)) {
    return;
  }
  std::vector<int> counts;
  std::vector<int> displs;
  conduction_line_counts(n_cells, part.n_ranks, 1, false, counts, displs);
  const int my_count = counts[static_cast<std::size_t>(part.rank)];
  const int my_displ = displs[static_cast<std::size_t>(part.rank)];
  std::vector<std::uint8_t> send(host_field.begin() + my_displ,
                                 host_field.begin() + my_displ + my_count);
  MPI_Allgatherv(send.data(), my_count, MPI_UNSIGNED_CHAR, host_field.data(),
                 counts.data(), displs.data(), MPI_UNSIGNED_CHAR,
                 MPI_COMM_WORLD);
#else
  (void)host_field;
  (void)n_cells;
  (void)part;
#endif
}

}  // namespace

void compute_effective_material_properties(const core::Config& cfg,
                                           const core::State& state,
                                           const int n_cells,
                                           std::vector<double>& A_eff,
                                           std::vector<double>& gamma_eff) {
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "Conduction requires at least one material");
  const auto& materials = cfg.materials.materials;
  const auto& mat0 = materials.front();
  const double A0 = std::max(mat0.A, kMinEffectiveA);
  const double gamma0 = std::max(mat0.ideal_gas_gamma, kMinEffectiveGamma);

  A_eff.assign(static_cast<std::size_t>(std::max(n_cells, 0)), A0);
  gamma_eff.assign(static_cast<std::size_t>(std::max(n_cells, 0)), gamma0);
  if (n_cells <= 0) {
    return;
  }

  // BUG-1 fix: when the state carries the shared per-cell effective
  // properties (State::ensure_cell_material_props — the single source of
  // truth also used by the hydro closure kernels), read them instead of
  // recomputing locally, so conduction and hydro can never disagree.
  if (state.A_eff.size() == static_cast<std::size_t>(n_cells) &&
      state.gamma_eff.size() == static_cast<std::size_t>(n_cells)) {
    state.A_eff.copy_to_host(A_eff.data());
    state.gamma_eff.copy_to_host(gamma_eff.data());
    return;
  }

  const int n_mat = static_cast<int>(materials.size());
  if (n_mat <= 1) {
    return;
  }

  const std::size_t expected =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_mat);
  if (state.volFrac.size() != expected) {
    static bool warned_volfrac_size_mismatch = false;
    if (!warned_volfrac_size_mismatch) {
      core::log_warning("Multi-material mixing: volFrac size mismatch (" +
                        std::to_string(state.volFrac.size()) + " vs expected " +
                        std::to_string(expected) +
                        "); falling back to materials[0].");
      warned_volfrac_size_mismatch = true;
    }
    return;
  }

  std::vector<double> volfrac(expected, 0.0);
  state.volFrac.copy_to_host(volfrac.data());

  std::vector<double> inv_A_m(static_cast<std::size_t>(n_mat), 0.0);
  std::vector<double> gamma_m(static_cast<std::size_t>(n_mat), gamma0);
  for (int m = 0; m < n_mat; ++m) {
    const auto& mat = materials[static_cast<std::size_t>(m)];
    const double A_m = std::max(mat.A, kMinEffectiveA);
    inv_A_m[static_cast<std::size_t>(m)] = 1.0 / A_m;
    gamma_m[static_cast<std::size_t>(m)] =
        std::max(mat.ideal_gas_gamma, kMinEffectiveGamma);
  }

  for (int c = 0; c < n_cells; ++c) {
    const std::size_t base = static_cast<std::size_t>(c) * static_cast<std::size_t>(n_mat);
    double frac_sum = 0.0;
    double inv_A_c = 0.0;
    double gamma_c = 0.0;
    for (int m = 0; m < n_mat; ++m) {
      const double frac_raw = volfrac[base + static_cast<std::size_t>(m)];
      const double frac =
          (std::isfinite(frac_raw) && frac_raw > 0.0) ? frac_raw : 0.0;
      frac_sum += frac;
      inv_A_c += frac * inv_A_m[static_cast<std::size_t>(m)];
      gamma_c += frac * gamma_m[static_cast<std::size_t>(m)];
    }
    if (frac_sum > 1.0e-30) {
      inv_A_c /= frac_sum;
      gamma_c /= frac_sum;
    }
    if (std::isfinite(inv_A_c) && inv_A_c > 1.0e-30) {
      A_eff[static_cast<std::size_t>(c)] =
          std::max(1.0 / inv_A_c, kMinEffectiveA);
    }
    if (std::isfinite(gamma_c) && gamma_c > 0.0) {
      gamma_eff[static_cast<std::size_t>(c)] =
          std::max(gamma_c, kMinEffectiveGamma);
    }
  }
}

struct ConductionMaterialParams {
  double A = 1.0;
  double Zbar = 1.0;
  double gamma = 5.0 / 3.0;
};

std::vector<ConductionMaterialParams> make_conduction_material_params(
    const core::Config& cfg) {
  std::vector<ConductionMaterialParams> params(cfg.materials.materials.size());
  for (std::size_t m = 0; m < cfg.materials.materials.size(); ++m) {
    const auto& mat = cfg.materials.materials[m];
    ConductionMaterialParams p{};
    p.A = std::max(mat.A, kMinEffectiveA);
    p.gamma = std::max(mat.ideal_gas_gamma, kMinEffectiveGamma);
    if (cfg.materials.zbar.model == "fixed" && cfg.materials.zbar.fixed_value >= 0.0) {
      p.Zbar = cfg.materials.zbar.fixed_value;
    } else {
      p.Zbar = (mat.Z > 0.0) ? mat.Z : 1.0;
    }
    if (mat.is_void) {
      p.Zbar = 0.0;
    }
    params[m] = p;
  }
  return params;
}

bool per_material_conduction_ready(const core::State& state,
                                   const core::Config& cfg,
                                   const HydroEOSContext* eos_ctx) {
  if (!cfg.numerics.materials.per_material_conservation_enabled || eos_ctx == nullptr) {
    return false;
  }
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_mat = cfg.materials.materials.size();
  if (n_cells == 0 || n_mat == 0) {
    return false;
  }
  const std::size_t n_cell_mat = n_cells * n_mat;
  return state.mass_per_material.size() == n_cell_mat &&
         state.Ee_per_material.size() == n_cell_mat &&
         state.volFrac.size() == n_cell_mat &&
         state.vol.size() == n_cells &&
         state.mass.size() == n_cells &&
         state.cv_e.size() == n_cells;
}

per_material::PerMaterialAccessorView make_conduction_per_material_view(
    const core::State& state,
    const core::Config& cfg,
    const int n_cells,
    const int n_mat) {
  per_material::PerMaterialAccessorView view{};
  view.mass_per_material = state.mass_per_material.data();
  view.Ee_per_material = state.Ee_per_material.data();
  view.Ei_per_material = state.Ei_per_material.data();
  view.volfrac = state.volFrac.data();
  view.vol = state.vol.data();
  view.Te_per_material = nullptr;
  view.Ti_per_material = nullptr;
  view.Te_per_material_valid = nullptr;
  view.Ti_per_material_valid = nullptr;
  view.lazy_cache_te_m_enabled = false;
  view.presence_threshold_volfrac =
      cfg.numerics.materials.presence_threshold_volfrac;
  view.presence_threshold_mass_density_g_per_cc =
      cfg.numerics.materials.presence_threshold_mass_density_g_per_cc;
  view.d_counts = nullptr;
  view.n_cells = n_cells;
  view.n_mat = n_mat;
  return view;
}

__host__ __device__ inline double sanitize_te_for_pow(const double Te_eV) {
  return (isfinite(Te_eV) && Te_eV > 0.0) ? fmin(Te_eV, kMaxTeForPow) : 0.0;
}

__host__ __device__ inline double electron_density_formula(const double rho,
                                                           const double zbar,
                                                           const double A) {
  const double rho_pos = fmax(rho, 0.0);
  const double z = fmax(zbar, 0.0);
  if (!(rho_pos > 0.0) || !(z > 0.0) || !(A > 0.0)) {
    return 0.0;
  }
  const double n_e = rho_pos * z / (A * kProtonMass);
  return isfinite(n_e) ? n_e : 0.0;
}

__host__ __device__ inline double coulomb_log_formula(const double n_e,
                                                      const double Te_eV,
                                                      const double Zbar) {
  const double te = sanitize_te_for_pow(Te_eV);
  if (!(n_e > 0.0) || !(te > 0.0) || !(Zbar > 0.0)) {
    return 2.0;
  }

  double ln_lambda_raw = 0.0;
  if (te >= 10.0 * Zbar * Zbar) {
    ln_lambda_raw = 24.0 - 0.5 * log(n_e) + log(te);
  } else {
    ln_lambda_raw = 23.0 - 0.5 * log(n_e) - log(Zbar) + 1.5 * log(te);
  }
  return fmax(2.0, ln_lambda_raw);
}

__host__ __device__ inline double spitzer_kappa_formula(const double Te_eV,
                                                        const double Zbar,
                                                        const double ln_lambda) {
  constexpr double kappa0 = 3.06e9;  // [erg/(cm*s*eV^{7/2})]
  const double te = sanitize_te_for_pow(Te_eV);
  if (!(te > 0.0) || !(Zbar > 0.0) || !(ln_lambda > 0.0)) {
    return 0.0;
  }
  const double kappa = kappa0 * pow(te, 2.5) / (Zbar * ln_lambda);
  return isfinite(kappa) ? kappa : 0.0;
}

__host__ __device__ inline double vth_e_formula(const double Te_eV) {
  const double te = sanitize_te_for_pow(Te_eV);
  if (!(te > 0.0)) {
    return 0.0;
  }
  const double vth = sqrt(kEvToErg * te / kElectronMass);
  return isfinite(vth) ? vth : 0.0;
}

__host__ __device__ inline double q_max_formula(const double f_lim,
                                                const double n_e,
                                                const double Te_eV) {
  const double te = sanitize_te_for_pow(Te_eV);
  if (!(f_lim > 0.0) || !(n_e > 0.0) || !(te > 0.0)) {
    return 0.0;
  }
  const double q_max = f_lim * n_e * kEvToErg * te * vth_e_formula(te);
  return isfinite(q_max) ? q_max : 0.0;
}

__host__ __device__ inline double mfp_limiter_factor_formula(double n_e,
                                                             double Te_eV,
                                                             double Zbar,
                                                             double ln_lambda,
                                                             double dl,
                                                             double mfp_limiter_C);

__device__ inline bool material_present_for_conduction(
    const per_material::PerMaterialAccessorView& view,
    const int c,
    const int m) {
  const int idx = c * view.n_mat + m;
  const double vf = view.volfrac[idx];
  if (!(vf > view.presence_threshold_volfrac) || !isfinite(vf)) {
    return false;
  }
  const double rho_m = per_material::get_rho_per_material(view, c, m);
  return rho_m > view.presence_threshold_mass_density_g_per_cc && isfinite(rho_m);
}

__device__ inline bool material_present_for_conduction_dt(
    const per_material::PerMaterialAccessorView& view,
    const int c,
    const int m) {
  const int idx = c * view.n_mat + m;
  const double vf = view.volfrac[idx];
  const double V = view.vol[c];
  const double mass_m = view.mass_per_material[idx];
  if (!(vf > view.presence_threshold_volfrac) || !(V > 0.0) ||
      !(mass_m > 0.0) || !isfinite(vf) || !isfinite(V) ||
      !isfinite(mass_m)) {
    return false;
  }
  const double rho_m_partial = mass_m / V;
  return rho_m_partial > view.presence_threshold_mass_density_g_per_cc &&
         isfinite(rho_m_partial);
}

__device__ inline double limited_spitzer_kappa_material(
    const double rho_m_face,
    const double Te_face,
    const double Zbar_m,
    const double A_m,
    const double grad_Te,
    const double dl,
    const double f_lim,
    const double test_kappa,
    const double mfp_limiter_C) {
  if (!(rho_m_face > 0.0) || !(Te_face > 0.0) || !(Zbar_m > 0.0) || !(A_m > 0.0)) {
    return 0.0;
  }
  double kappa_eff = 0.0;
  const double n_e_m = electron_density_formula(rho_m_face, Zbar_m, A_m);
  if (test_kappa > 0.0) {
    kappa_eff = test_kappa;
  } else if (n_e_m > 0.0) {
    const double ln_lambda_m = coulomb_log_formula(n_e_m, Te_face, Zbar_m);
    kappa_eff = spitzer_kappa_formula(Te_face, Zbar_m, ln_lambda_m);
    if (kappa_eff > 0.0 && mfp_limiter_C > 0.0 && dl > 0.0) {
      const double limiter =
          mfp_limiter_factor_formula(n_e_m, Te_face, Zbar_m, ln_lambda_m, dl, mfp_limiter_C);
      kappa_eff *= limiter;
    }
  }
  if (!(kappa_eff > 0.0) || !isfinite(kappa_eff)) {
    return 0.0;
  }
  const double q_max_m = q_max_formula(f_lim, n_e_m, Te_face);
  if (!(q_max_m > 0.0)) {
    return 0.0;
  }
  const double q_sh_m = -kappa_eff * grad_Te;
  const double limiter = 1.0 / (1.0 + fabs(q_sh_m) / q_max_m);
  return isfinite(limiter) ? (kappa_eff * fmin(1.0, fmax(limiter, 0.0))) : 0.0;
}

__host__ __device__ inline double electron_collision_frequency_formula(const double n_e,
                                                                       const double Te_eV,
                                                                       const double Zbar,
                                                                       const double ln_lambda) {
  const double te = sanitize_te_for_pow(Te_eV);
  if (!(n_e > 0.0) || !(te > 0.0) || !(Zbar > 0.0) || !(ln_lambda > 0.0)) {
    return 0.0;
  }
  const double te32 = te * sqrt(te);
  if (!(te32 > 0.0)) {
    return 0.0;
  }
  const double nu_ei = 2.91e-6 * n_e * Zbar * ln_lambda / te32;
  return isfinite(nu_ei) ? fmax(nu_ei, 0.0) : 0.0;
}

__host__ __device__ inline double mfp_limiter_factor_formula(const double n_e,
                                                              const double Te_eV,
                                                              const double Zbar,
                                                              const double ln_lambda,
                                                              const double dl,
                                                              const double mfp_limiter_C) {
  if (!(mfp_limiter_C > 0.0) || !(dl > 0.0)) {
    return 1.0;
  }
  const double nu_ei = electron_collision_frequency_formula(n_e, Te_eV, Zbar, ln_lambda);
  const double v_th = vth_e_formula(Te_eV);
  const double lambda_mfp = (nu_ei > 0.0 && v_th > 0.0) ? (v_th / nu_ei) : 1.0e30;
  if (!(lambda_mfp > 0.0) || !isfinite(lambda_mfp)) {
    return 0.0;
  }
  const double ratio = (mfp_limiter_C * dl) / lambda_mfp;
  if (!isfinite(ratio)) {
    return 0.0;
  }
  return fmin(1.0, fmax(ratio, 0.0));
}

__host__ __device__ inline double flux_limiter_formula(const double q_sh,
                                                       const double q_max) {
  if (!(q_max > 0.0)) {
    return 0.0;
  }
  return q_sh / (1.0 + fabs(q_sh) / q_max);
}

__host__ __device__ inline double deff_formula(const double q_limited,
                                               const double rho,
                                               const double cv_e,
                                               const double grad_Te,
                                               const double D_sh,
                                               const double eps_grad) {
  const double g = fabs(grad_Te);
  if (g < eps_grad) {
    return D_sh;
  }
  const double denom = rho * cv_e * g;
  if (!(denom > 0.0)) {
    return D_sh;
  }
  return fabs(q_limited) / denom;
}

__device__ double atomic_min_double(double* address, const double value) {
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  while (value < __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(value)));
    if (assumed == old) {
      break;
    }
  }
  return __longlong_as_double(static_cast<long long>(old));
}

__device__ double atomic_add_double(double* address, const double value) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, value);
#else
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(
                        value + __longlong_as_double(static_cast<long long>(assumed)))));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
#endif
}

__device__ inline double node_value(const double* __restrict__ x,
                                    const int ir,
                                    const int iz,
                                    const int nz) {
  return x[static_cast<std::size_t>(ir) * static_cast<std::size_t>(nz + 1) +
           static_cast<std::size_t>(iz)];
}

__device__ inline double cell_center_from_nodes(const double* __restrict__ x,
                                                const int i,
                                                const int j,
                                                const int nz) {
  return 0.25 * (node_value(x, i, j, nz) + node_value(x, i + 1, j, nz) +
                 node_value(x, i, j + 1, nz) + node_value(x, i + 1, j + 1, nz));
}

__device__ inline void apply_state_supply_conduction_face(
    double* __restrict__ Te,
    const double* __restrict__ kappa_eff,
    const double* __restrict__ rho_cv_e,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    double* __restrict__ boundary_energy,
    const int i,
    const int j,
    const int nz,
    const double dt,
    const int boundary_id,
    const double T_boundary) {
  const int c = i * nz + j;
  const double rho_cv = rho_cv_e[c];
  const double V = vol[c];
  const double kappa = kappa_eff[c];
  if (!(rho_cv > 0.0) || !(V > 0.0) || !(kappa > 0.0)) {
    return;
  }
  const double r0 = node_value(x_r, i, j, nz);
  const double r1 = node_value(x_r, i + 1, j, nz);
  const double area = kPi * fmax(r1 * r1 - r0 * r0, 0.0);
  const int face_j = (boundary_id == 2) ? j : (j + 1);
  const double z_face =
      0.5 * (node_value(x_z, i, face_j, nz) + node_value(x_z, i + 1, face_j, nz));
  const double z_cell = cell_center_from_nodes(x_z, i, j, nz);
  const double dx = fabs(z_cell - z_face);
  if (!(area > 0.0) || !(dx > 0.0)) {
    return;
  }
  const double flux_power = kappa * (T_boundary - Te[c]) * area / dx;
  const double dE = dt * flux_power;
  const double dT = dE / (rho_cv * V);
  if (isfinite(dT) && isfinite(dE)) {
    Te[c] += dT;
    atomic_add_double(boundary_energy + boundary_id, dE);
  }
}

__global__ void apply_state_supply_conduction_bc_kernel(
    double* __restrict__ Te,
    const double* __restrict__ kappa_eff,
    const double* __restrict__ rho_cv_e,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    double* __restrict__ boundary_energy,
    const int nr,
    const int nz,
    const double dt,
    const int z_bottom_source,
    const double T_z_bottom,
    const int z_top_source,
    const double T_z_top) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nr) {
    return;
  }

  if (z_bottom_source != 0) {
    apply_state_supply_conduction_face(Te,
                                       kappa_eff,
                                       rho_cv_e,
                                       vol,
                                       x_r,
                                       x_z,
                                       boundary_energy,
                                       i,
                                       0,
                                       nz,
                                       dt,
                                       2,
                                       T_z_bottom);
  }
  if (z_top_source != 0) {
    apply_state_supply_conduction_face(Te,
                                       kappa_eff,
                                       rho_cv_e,
                                       vol,
                                       x_r,
                                       x_z,
                                       boundary_energy,
                                       i,
                                       nz - 1,
                                       nz,
                                       dt,
                                       3,
                                       T_z_top);
  }
}

__device__ double atomic_max_double(double* address, const double value) {
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  while (value > __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(value)));
    if (assumed == old) {
      break;
    }
  }
  return __longlong_as_double(static_cast<long long>(old));
}

__device__ inline double harmonic_mean(const double a, const double b) {
  if (!(a > 0.0) || !(b > 0.0)) {
    return 0.0;
  }
  return (2.0 * a * b) / (a + b);
}

// W-G2 kirchhoff face_kappa_policy (external verdict 2026-07-04): same-material
// smooth-face Kirchhoff closure for the Spitzer exponent n = 5/2.
//   kappa0_c = kappa_c / T_c^{5/2} (cell kappa0 extraction; carries any
//   mfp-limiter factor folded into the cell kappa),
//   kappa_f = harmonic(kappa0_L, kappa0_R) * S_{5/2}(T_L, T_R),
//   S_{5/2} = (2/7)(T_R^{7/2} - T_L^{7/2})/(T_R - T_L); near-equal-T Taylor
//   form T_m^{5/2}(1 + (2.5*1.5/24) x^2), x = dT/T_m, |x| < 1e-4
//   (central-divided-difference cancellation guard; next term O(x^4)).
// Fallbacks that keep the historic resistance (harmonic) closure:
//   - nonpositive Te or nonpositive kappa on either side (voids stay
//     insulating: harmonic of a zero is zero),
//   - kappa0-jump detector |ln(kappa0_R/kappa0_L)| > ln 10, implemented as
//     ratio outside (0.1, 10): material interfaces / strong tabular jumps.
__device__ inline double kirchhoff_face_kappa_spitzer(const double kap_l,
                                                      const double kap_r,
                                                      const double te_l_raw,
                                                      const double te_r_raw) {
  const double te_l = sanitize_te_for_pow(te_l_raw);
  const double te_r = sanitize_te_for_pow(te_r_raw);
  if (!(te_l > 0.0) || !(te_r > 0.0) || !(kap_l > 0.0) || !(kap_r > 0.0)) {
    return harmonic_mean(kap_l, kap_r);
  }
  const double kap0_l = kap_l / pow(te_l, 2.5);
  const double kap0_r = kap_r / pow(te_r, 2.5);
  const double ratio = kap0_r / kap0_l;
  if (!(ratio > 0.1 && ratio < 10.0)) {
    return harmonic_mean(kap_l, kap_r);
  }
  const double kap0_f = harmonic_mean(kap0_l, kap0_r);
  const double t_m = 0.5 * (te_l + te_r);
  const double x = (te_r - te_l) / t_m;
  double s52;
  if (fabs(x) < 1.0e-4) {
    s52 = pow(t_m, 2.5) * (1.0 + (2.5 * 1.5 / 24.0) * x * x);
  } else {
    s52 = (2.0 / 7.0) * (pow(te_r, 3.5) - pow(te_l, 3.5)) / (te_r - te_l);
  }
  return kap0_f * s52;
}

__device__ inline double cell_center_radius(const double* x_r, const int i) {
  return 0.5 * (x_r[i] + x_r[i + 1]);
}

__device__ inline void compute_spitzer_deff_1d_kernel_per_material_body(
    const int tid,
    const int /*thread_tid*/,
    double* __restrict__ /*shared*/,
    double* __restrict__ kappa_face,
    double* __restrict__ kappa_eff_per_material,
    double* __restrict__ rho_cv_e,
    double* __restrict__ min_ratio,
    double* __restrict__ min_deff,
    double* __restrict__ max_deff,
    per_material::PerMaterialAccessorView view,
    const tenryu::materials::DeviceEOSTableView* __restrict__ electron_views,
    const ConductionMaterialParams* __restrict__ material_params,
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int n_cells,
    const int n_mat,
    const std::uint8_t* __restrict__ cell_is_void,
    const double f_lim,
    const double test_kappa,
    const double mfp_limiter_C,
    const double* __restrict__ state_cv_e,
    const double Te_floor,
    const bool low_density_extrap) {
  if (tid < n_cells) {
    const double rho_i = fmax(rho[tid], 0.0);
    const double cv_e = (state_cv_e != nullptr && state_cv_e[tid] > 0.0)
                            ? state_cv_e[tid]
                            : 0.0;
    rho_cv_e[tid] = rho_i * cv_e;
  }

  const int n_faces = n_cells - 1;
  if (tid >= n_faces || n_faces <= 0) {
    return;
  }
  const int iL = tid;
  const int iR = tid + 1;
  if ((cell_is_void != nullptr &&
       (cell_is_void[iL] != static_cast<std::uint8_t>(0) ||
        cell_is_void[iR] != static_cast<std::uint8_t>(0)))) {
    kappa_face[tid] = 0.0;
    for (int m = 0; m < n_mat; ++m) {
      kappa_eff_per_material[static_cast<std::size_t>(m) * n_faces + tid] = 0.0;
    }
    return;
  }

  const double rcl = cell_center_radius(x_r, iL);
  const double rcr = cell_center_radius(x_r, iR);
  const double dr = rcr - rcl;
  if (!(dr > 0.0)) {
    kappa_face[tid] = 0.0;
    for (int m = 0; m < n_mat; ++m) {
      kappa_eff_per_material[static_cast<std::size_t>(m) * n_faces + tid] = 0.0;
    }
    return;
  }

  double aggregate_kappa = 0.0;
  double aggregate_kappa_dt = 0.0;
  for (int m = 0; m < n_mat; ++m) {
    double kappa_m = 0.0;
    const int idxL = iL * n_mat + m;
    const int idxR = iR * n_mat + m;
    const double vfL = view.volfrac[idxL];
    const double vfR = view.volfrac[idxR];
    const double face_fraction_m = 0.5 * (fmax(vfL, 0.0) + fmax(vfR, 0.0));
    if (face_fraction_m > 0.0 &&
        (material_present_for_conduction(view, iL, m) ||
         material_present_for_conduction(view, iR, m))) {
      const ConductionMaterialParams p = material_params[m];
      const tenryu::materials::DeviceEOSTableView electron_view =
          (electron_views != nullptr) ? electron_views[m]
                                      : tenryu::materials::DeviceEOSTableView{};
      const per_material::PerMaterialThermo eL =
          per_material::get_electron_thermo_per_material(view,
                                                         electron_view,
                                                         iL,
                                                         m,
                                                         p.Zbar,
                                                         p.A,
                                                         Te_floor,
                                                         low_density_extrap,
                                                         p.gamma);
      const per_material::PerMaterialThermo eR =
          per_material::get_electron_thermo_per_material(view,
                                                         electron_view,
                                                         iR,
                                                         m,
                                                         p.Zbar,
                                                         p.A,
                                                         Te_floor,
                                                         low_density_extrap,
                                                         p.gamma);
      const double TeL = sanitize_te_for_pow(per_material::get_te(eL));
      const double TeR = sanitize_te_for_pow(per_material::get_te(eR));
      const double Te_face = 0.5 * (TeL + TeR);
      const double grad_Te = (TeR - TeL) / dr;
      const double rho_m_face =
          0.5 * (fmax(per_material::get_rho_per_material(view, iL, m), 0.0) +
                 fmax(per_material::get_rho_per_material(view, iR, m), 0.0));
      kappa_m = limited_spitzer_kappa_material(rho_m_face,
                                               Te_face,
                                               p.Zbar,
                                               p.A,
                                               grad_Te,
                                               dr,
                                               f_lim,
                                               test_kappa,
                                               mfp_limiter_C);
    }
    kappa_eff_per_material[static_cast<std::size_t>(m) * n_faces + tid] = kappa_m;
    const double kappa_weighted_m = face_fraction_m * kappa_m;
    aggregate_kappa += kappa_weighted_m;
    if (kappa_weighted_m > 0.0 && isfinite(kappa_weighted_m) &&
        material_present_for_conduction_dt(view, iL, m) &&
        material_present_for_conduction_dt(view, iR, m)) {
      aggregate_kappa_dt += kappa_weighted_m;
    }
  }
  if (!isfinite(aggregate_kappa)) {
    aggregate_kappa = 0.0;
  }
  aggregate_kappa_dt =
      (isfinite(aggregate_kappa_dt) && aggregate_kappa_dt > 0.0)
          ? aggregate_kappa_dt
          : 0.0;
  kappa_face[tid] = fmax(aggregate_kappa, 0.0);

  const double rho_cv_face = harmonic_mean(rho_cv_e[iL], rho_cv_e[iR]);
  if (aggregate_kappa_dt > 0.0 && rho_cv_face > 0.0) {
    const double D_eff = aggregate_kappa_dt / rho_cv_face;
    const double ratio = (dr * dr) / D_eff;
    if (min_ratio != nullptr && ratio > 0.0 && isfinite(ratio)) {
      atomic_min_double(min_ratio, ratio);
    }
    if (min_deff != nullptr && D_eff > 0.0 && isfinite(D_eff)) {
      atomic_min_double(min_deff, D_eff);
    }
    if (max_deff != nullptr && D_eff > 0.0 && isfinite(D_eff)) {
      atomic_max_double(max_deff, D_eff);
    }
  }
}

__global__ void compute_spitzer_deff_1d_kernel_per_material(
    double* __restrict__ kappa_face,
    double* __restrict__ kappa_eff_per_material,
    double* __restrict__ rho_cv_e,
    double* __restrict__ min_ratio,
    double* __restrict__ min_deff,
    double* __restrict__ max_deff,
    per_material::PerMaterialAccessorView view,
    const tenryu::materials::DeviceEOSTableView* __restrict__ electron_views,
    const ConductionMaterialParams* __restrict__ material_params,
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const int n_mat,
    const std::uint8_t* __restrict__ cell_is_void,
    const double f_lim,
    const double test_kappa,
    const double mfp_limiter_C,
    const double* __restrict__ state_cv_e,
    const double Te_floor,
    const bool low_density_extrap) {
  const int tid = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int thread_tid = threadIdx.x;
  extern __shared__ double shared[];
  if (tid >= c_end) {
    return;
  }

  compute_spitzer_deff_1d_kernel_per_material_body(
      tid, thread_tid, shared, kappa_face, kappa_eff_per_material, rho_cv_e,
      min_ratio, min_deff, max_deff, view, electron_views, material_params, Te,
      rho, vol, x_r, n_cells, n_mat, cell_is_void, f_lim, test_kappa,
      mfp_limiter_C, state_cv_e, Te_floor, low_density_extrap);
}

__global__ void compute_spitzer_deff_2d_kernel_per_material(
    double* __restrict__ kappa_eff_out,
    double* __restrict__ kappa_eff_per_material,
    double* __restrict__ rho_cv_e,
    double* __restrict__ min_ratio,
    double* __restrict__ min_deff,
    double* __restrict__ max_deff,
    per_material::PerMaterialAccessorView view,
    const tenryu::materials::DeviceEOSTableView* __restrict__ electron_views,
    const ConductionMaterialParams* __restrict__ material_params,
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int nr,
    const int nz,
    const int n_mat,
    const std::uint8_t* __restrict__ cell_is_void,
    const double f_lim,
    const double test_kappa,
    const double mfp_limiter_C,
    const double* __restrict__ state_cv_e,
    const double Te_floor,
    const bool low_density_extrap,
    const int c_begin,
    const int c_end) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = c_begin + tid;
  const int n_cells = nr * nz;
  if (c >= c_end) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;

  const double rho_c = fmax(rho[c], 0.0);
  const double cv_e = (state_cv_e != nullptr && state_cv_e[c] > 0.0) ? state_cv_e[c] : 0.0;
  const double rho_cv = rho_c * cv_e;
  if (rho_cv_e != nullptr) {
    rho_cv_e[c] = rho_cv;
  }
  if (cell_is_void != nullptr && cell_is_void[c] != static_cast<std::uint8_t>(0)) {
    if (kappa_eff_out != nullptr) {
      kappa_eff_out[c] = 0.0;
    }
    for (int m = 0; m < n_mat; ++m) {
      kappa_eff_per_material[static_cast<std::size_t>(m) * n_cells + c] = 0.0;
    }
    return;
  }

  double gr0 = 0.0;
  double gz0 = 0.0;
  double gr1 = 0.0;
  double gz1 = 0.0;
  double gr2 = 0.0;
  double gz2 = 0.0;
  double gr3 = 0.0;
  double gz3 = 0.0;
  kershaw::node_gradient_from_cells(Te, x_r, x_z, i, j, nr, nz, gr0, gz0);
  kershaw::node_gradient_from_cells(Te, x_r, x_z, i + 1, j, nr, nz, gr1, gz1);
  kershaw::node_gradient_from_cells(Te, x_r, x_z, i + 1, j + 1, nr, nz, gr2, gz2);
  kershaw::node_gradient_from_cells(Te, x_r, x_z, i, j + 1, nr, nz, gr3, gz3);
  const double grad_mag = 0.25 *
                          (sqrt(gr0 * gr0 + gz0 * gz0) + sqrt(gr1 * gr1 + gz1 * gz1) +
                           sqrt(gr2 * gr2 + gz2 * gz2) + sqrt(gr3 * gr3 + gz3 * gz3));
  const double dl = sqrt(fmax(kershaw::cell_area_from_nodes(x_r, x_z, i, j, nz), 1.0e-30));

  double aggregate_kappa = 0.0;
  double aggregate_kappa_dt = 0.0;
  for (int m = 0; m < n_mat; ++m) {
    double kappa_weighted_m = 0.0;
    const int idx = c * n_mat + m;
    const double vf = view.volfrac[idx];
    if (vf > 0.0 && material_present_for_conduction(view, c, m)) {
      const ConductionMaterialParams p = material_params[m];
      const tenryu::materials::DeviceEOSTableView electron_view =
          (electron_views != nullptr) ? electron_views[m]
                                      : tenryu::materials::DeviceEOSTableView{};
      const per_material::PerMaterialThermo e =
          per_material::get_electron_thermo_per_material(view,
                                                         electron_view,
                                                         c,
                                                         m,
                                                         p.Zbar,
                                                         p.A,
                                                         Te_floor,
                                                         low_density_extrap,
                                                         p.gamma);
      const double Te_m = sanitize_te_for_pow(per_material::get_te(e));
      const double rho_m = per_material::get_rho_per_material(view, c, m);
      const double kappa_m = limited_spitzer_kappa_material(rho_m,
                                                            Te_m,
                                                            p.Zbar,
                                                            p.A,
                                                            grad_mag,
                                                            dl,
                                                            f_lim,
                                                            test_kappa,
                                                            mfp_limiter_C);
      kappa_weighted_m = fmax(vf, 0.0) * kappa_m;
    }
    kappa_eff_per_material[static_cast<std::size_t>(m) * n_cells + c] =
        isfinite(kappa_weighted_m) ? kappa_weighted_m : 0.0;
    aggregate_kappa += kappa_weighted_m;
    if (kappa_weighted_m > 0.0 && isfinite(kappa_weighted_m) &&
        material_present_for_conduction_dt(view, c, m)) {
      aggregate_kappa_dt += kappa_weighted_m;
    }
  }
  aggregate_kappa = (isfinite(aggregate_kappa) && aggregate_kappa > 0.0) ? aggregate_kappa : 0.0;
  aggregate_kappa_dt =
      (isfinite(aggregate_kappa_dt) && aggregate_kappa_dt > 0.0)
          ? aggregate_kappa_dt
          : 0.0;
  if (kappa_eff_out != nullptr) {
    kappa_eff_out[c] = aggregate_kappa;
  }
  if (rho_cv > 0.0 && aggregate_kappa_dt > 0.0) {
    const double D_eff = aggregate_kappa_dt / rho_cv;
    const double ratio = (dl * dl) / D_eff;
    if (min_ratio != nullptr && ratio > 0.0 && isfinite(ratio)) {
      atomic_min_double(min_ratio, ratio);
    }
    if (min_deff != nullptr && D_eff > 0.0 && isfinite(D_eff)) {
      atomic_min_double(min_deff, D_eff);
    }
    if (max_deff != nullptr && D_eff > 0.0 && isfinite(D_eff)) {
      atomic_max_double(max_deff, D_eff);
    }
  }
}

template <bool kZMom>
__global__ void compute_spitzer_deff_1d_kernel(
    double* __restrict__ kappa_eff,
    double* __restrict__ rho_cv_e,
    double* __restrict__ min_ratio,
    double* __restrict__ min_deff,
    double* __restrict__ max_deff,
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const double* __restrict__ zbar,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const double* __restrict__ gamma_eff,
    const double* __restrict__ A_eff,
    const std::uint8_t* __restrict__ cell_is_void,
    const double f_lim,
    const double test_kappa,
    const double mfp_limiter_C,
    const double* __restrict__ state_cv_e,
    const double* __restrict__ zmom_r2) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  conduction_bodies::compute_spitzer_deff_1d_kernel_body<kZMom>(
      i, kappa_eff, rho_cv_e, min_ratio, min_deff, max_deff, Te, rho, zbar,
      vol, x_r, n_cells, gamma_eff, A_eff, cell_is_void, f_lim, test_kappa,
      mfp_limiter_C, state_cv_e, zmom_r2);
}

// W-G2 nlheat verify hook: power-law test conductivity
// kappa = test_kappa * Te^kappa_power (env TENRYU_CONDUCTION_TEST_KAPPA_POWER;
// default-inert). Mirrors the test_kappa branch of
// compute_spitzer_deff_1d_kernel — identical rho_cv fill, void handling, and
// dt reductions — with the constant kappa replaced by the power law. With
// kappa_power = 0 this reproduces the historic constant-kappa values exactly
// (IEEE pow(x, 0) == 1). 1D single-material diagnostics path only.
__global__ void compute_powerlaw_test_kappa_deff_1d_kernel(
    double* __restrict__ kappa_eff,
    double* __restrict__ rho_cv_e,
    double* __restrict__ min_ratio,
    double* __restrict__ min_deff,
    double* __restrict__ max_deff,
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const double* __restrict__ zbar,
    const double* __restrict__ x_r,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const double* __restrict__ gamma_eff,
    const double* __restrict__ A_eff,
    const std::uint8_t* __restrict__ cell_is_void,
    const double test_kappa,
    const double kappa_power,
    const double kappa_rho_power,
    const double* __restrict__ state_cv_e) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  conduction_bodies::compute_powerlaw_test_kappa_deff_1d_kernel_body(
      i, kappa_eff, rho_cv_e, min_ratio, min_deff, max_deff, Te, rho, zbar,
      x_r, n_cells, gamma_eff, A_eff, cell_is_void, test_kappa, kappa_power,
      kappa_rho_power, state_cv_e);
}

__global__ void compute_spitzer_deff_2d_kernel(
    double* __restrict__ kappa_eff_out,
    double* __restrict__ rho_cv_e,
    double* __restrict__ min_ratio,
    double* __restrict__ min_deff,
    double* __restrict__ max_deff,
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const double* __restrict__ zbar,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int nr,
    const int nz,
    const double* __restrict__ gamma_eff,
    const double* __restrict__ A_eff,
    const std::uint8_t* __restrict__ cell_is_void,
    const double f_lim,
    const double test_kappa,
    const double mfp_limiter_C,
    const double* __restrict__ state_cv_e,
    const int c_begin,
    const int c_end) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = c_begin + tid;
  const int n_cells = nr * nz;
  if (c >= c_end) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;

  const double z = fmax(zbar[c], 0.0);
  const double rho_c = fmax(rho[c], 0.0);
  const double te = (isfinite(Te[c]) && Te[c] > 0.0) ? fmin(Te[c], kMaxTeForPow) : 0.0;
  const double gamma_c = fmax(gamma_eff[c], kMinEffectiveGamma);
  const double A_c = fmax(A_eff[c], kMinEffectiveA);
  const double cv_e_ideal = (gamma_c > 1.0 && A_c > 0.0)
                                ? (z * kEvToErg / (A_c * kProtonMass * (gamma_c - 1.0)))
                                : 0.0;
  const double cv_e =
      (state_cv_e != nullptr && state_cv_e[c] > 0.0) ? state_cv_e[c] : cv_e_ideal;
  const double rho_cv = rho_c * cv_e;

  if (rho_cv_e != nullptr) {
    rho_cv_e[c] = rho_cv;
  }

  if (cell_is_void != nullptr && cell_is_void[c] != static_cast<std::uint8_t>(0)) {
    if (kappa_eff_out != nullptr) {
      kappa_eff_out[c] = 0.0;
    }
    return;
  }

  double kappa_local = 0.0;
  double D_eff = 0.0;
  if (test_kappa > 0.0) {
    if (rho_cv > 0.0) {
      kappa_local = test_kappa;
      D_eff = kappa_local / rho_cv;
    }
  } else if (rho_cv > 0.0 && z > 0.0 && te > 0.0) {
    const double n_e = electron_density_formula(rho_c, z, A_c);
    const double ln_lambda = coulomb_log_formula(n_e, te, z);
    const double kappa_sh = spitzer_kappa_formula(te, z, ln_lambda);
    double kappa_eff = fmax(kappa_sh, 0.0);
    if (kappa_eff > 0.0 && mfp_limiter_C > 0.0) {
      const double area = kershaw::cell_area_from_nodes(x_r, x_z, i, j, nz);
      const double dl = sqrt(fmax(area, 1.0e-30));
      const double limiter =
          mfp_limiter_factor_formula(n_e, te, z, ln_lambda, dl, mfp_limiter_C);
      kappa_eff *= limiter;
    }
    kappa_local = kappa_eff;
    const double D_sh = kappa_eff / rho_cv;

    double gr0 = 0.0;
    double gz0 = 0.0;
    double gr1 = 0.0;
    double gz1 = 0.0;
    double gr2 = 0.0;
    double gz2 = 0.0;
    double gr3 = 0.0;
    double gz3 = 0.0;
    kershaw::node_gradient_from_cells(Te, x_r, x_z, i, j, nr, nz, gr0, gz0);
    kershaw::node_gradient_from_cells(Te, x_r, x_z, i + 1, j, nr, nz, gr1, gz1);
    kershaw::node_gradient_from_cells(Te, x_r, x_z, i + 1, j + 1, nr, nz, gr2, gz2);
    kershaw::node_gradient_from_cells(Te, x_r, x_z, i, j + 1, nr, nz, gr3, gz3);

    const double grad = 0.25 *
                        (sqrt(gr0 * gr0 + gz0 * gz0) + sqrt(gr1 * gr1 + gz1 * gz1) +
                         sqrt(gr2 * gr2 + gz2 * gz2) + sqrt(gr3 * gr3 + gz3 * gz3));

    const double q_sh = isfinite(kappa_eff) ? (-kappa_eff * grad) : 0.0;
    const double q_max = q_max_formula(f_lim, n_e, te);
    const double q_limited = flux_limiter_formula(isfinite(q_sh) ? q_sh : 0.0, q_max);
    const double ell_c = cbrt(fmax(vol[c], 1.0e-30));
    const double eps_grad = fmax(1.0e-10 * te / ell_c, 1.0e-30);

    D_eff = deff_formula(q_limited, rho_c, cv_e, grad, D_sh, eps_grad);
    if (!isfinite(D_eff)) {
      D_eff = 0.0;
    }
    D_eff = fmax(D_eff, 0.0);
  }

  if (kappa_eff_out != nullptr) {
    kappa_eff_out[c] = kappa_local;
  }

  if (D_eff > 0.0) {
    const double area = kershaw::cell_area_from_nodes(x_r, x_z, i, j, nz);
    const double dl = sqrt(fmax(area, 1.0e-30));
    const double ratio = (dl * dl) / D_eff;
    if (min_ratio != nullptr && ratio > 0.0 && isfinite(ratio)) {
      atomic_min_double(min_ratio, ratio);
    }
    if (min_deff != nullptr) {
      atomic_min_double(min_deff, D_eff);
    }
    if (max_deff != nullptr) {
      atomic_max_double(max_deff, D_eff);
    }
  }
}

template <int GEOM>
__global__ void conduction_alpha_pass_kernel(
    double* __restrict__ alpha_cells,
    const double* __restrict__ Te_old,
    const double* __restrict__ kappa_sh,
    const double* __restrict__ flux_limiter_faces,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const double tau,
    const double Te_floor,
    const int floor_limiter_mode) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  alpha_cells[i] = 1.0;
  if (cell_is_void != nullptr && cell_is_void[i] != static_cast<std::uint8_t>(0)) {
    return;
  }

  double V;
  if constexpr (GEOM == 2) {
    V = fmax(x_r[i + 1] - x_r[i], 1.0e-30);
  } else {
    V = fmax(vol[i], 1.0e-30);
  }

  double left_term = 0.0;
  if (i > 0) {
    const double k_face = harmonic_mean(kappa_sh[i - 1], kappa_sh[i]);
    const double rcl = cell_center_radius(x_r, i - 1);
    const double rcr = cell_center_radius(x_r, i);
    const double dr = rcr - rcl;
    if (k_face > 0.0 && dr > 0.0) {
      const double grad = (Te_old[i] - Te_old[i - 1]) / dr;
      const double q_sh = k_face * grad;
      const double limiter =
          (flux_limiter_faces != nullptr) ? flux_limiter_faces[i - 1] : 1.0;
      const double q_face = q_sh * limiter;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i] * x_r[i];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i]);
      }
      left_term = area * q_face;
    }
  }

  double right_term = 0.0;
  if (i + 1 < n_cells) {
    const double k_face = harmonic_mean(kappa_sh[i], kappa_sh[i + 1]);
    const double rcl = cell_center_radius(x_r, i);
    const double rcr = cell_center_radius(x_r, i + 1);
    const double dr = rcr - rcl;
    if (k_face > 0.0 && dr > 0.0) {
      const double grad = (Te_old[i + 1] - Te_old[i]) / dr;
      const double q_sh = k_face * grad;
      const double limiter = (flux_limiter_faces != nullptr) ? flux_limiter_faces[i] : 1.0;
      const double q_face = q_sh * limiter;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i + 1] * x_r[i + 1];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i + 1]);
      }
      right_term = area * q_face;
    }
  }

  const double M = (right_term - left_term) / V;
  const double rho_cv = rho_cv_e[i];
  const double Te_old_i = Te_old[i];
  if (rho_cv > 0.0) {
    if (floor_limiter_mode == 1) {
      const double outflow_sum =
          ((isfinite(right_term) ? fmax(0.0, -right_term) : 0.0) +
           (isfinite(left_term) ? fmax(0.0, left_term) : 0.0)) /
          V;
      double alpha = 1.0;
      if (!(Te_old_i > Te_floor)) {
        alpha = 0.0;
      } else if (tau > 0.0 && outflow_sum > 0.0) {
        const double e_above = rho_cv * (Te_old_i - Te_floor);
        alpha = fmin(
            1.0, fmax((e_above / fmax(outflow_sum, kDivEpsilon)) / tau, 0.0));
      }
      alpha_cells[i] = alpha;
      return;
    }
    double alpha = 1.0;
    if (tau > 0.0 && Te_old_i > Te_floor && M < 0.0) {
      const double e_above_floor = rho_cv * (Te_old_i - Te_floor);
      const double dt_safe = e_above_floor / fmax(fabs(M), kDivEpsilon);
      alpha = fmin(1.0, fmax(dt_safe / tau, 0.0));
    }
    alpha_cells[i] = alpha;
  }
}

template <int GEOM>
__global__ void conduction_1d_sts_stage_kernel(
    const double* __restrict__ Te_old,
    double* __restrict__ Te_new,
    const double* __restrict__ kappa_sh,
    const double* __restrict__ flux_limiter_faces,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const double tau,
    const double Te_floor,
    const double* __restrict__ alpha_cells,
    int* __restrict__ clamp_count,
    double* __restrict__ E_floor,
    const int floor_limiter_mode) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  conduction_bodies::conduction_1d_sts_stage_kernel_body<GEOM>(
      i, Te_old, Te_new, kappa_sh, flux_limiter_faces, rho_cv_e, cell_is_void,
      vol, x_r, n_cells, tau, Te_floor, alpha_cells, clamp_count, E_floor,
      floor_limiter_mode);
}

// W-G2 kirchhoff face diagnostics (TENRYU_CONDUCTION_FACE_DIAG, read-only):
// per-face harmonic/Kirchhoff flux ratio R_HK (n = 5/2, theta = T_c/T_h),
// limiter saturation s = |F_K|/F_sat recovered from the cached limiter
// lambda = 1/(1+s), and the limiter-corrected ratio
// R_eff = R_HK (1+s)/(1+R_HK s) (external verdict 2026-07-04). A face is
// active when both kappas and temperatures are positive and T_L != T_R.
__global__ void face_kappa_diag_1d_kernel(
    double* __restrict__ r_hk,
    double* __restrict__ r_eff,
    double* __restrict__ s_sat,
    std::uint8_t* __restrict__ active,
    const double* __restrict__ Te,
    const double* __restrict__ kappa_sh,
    const double* __restrict__ flux_limiter_faces,
    const int n_faces) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  r_hk[f] = 1.0;
  r_eff[f] = 1.0;
  s_sat[f] = -1.0;
  active[f] = static_cast<std::uint8_t>(0);

  const double te_l = sanitize_te_for_pow(Te[f]);
  const double te_r = sanitize_te_for_pow(Te[f + 1]);
  const double kap_l = kappa_sh[f];
  const double kap_r = kappa_sh[f + 1];
  if (!(te_l > 0.0) || !(te_r > 0.0) || !(kap_l > 0.0) || !(kap_r > 0.0) ||
      te_l == te_r) {
    return;
  }
  const double t_h = fmax(te_l, te_r);
  const double t_c = fmin(te_l, te_r);
  const double theta = t_c / t_h;
  const double th_n = pow(theta, 2.5);
  const double th_np1 = pow(theta, 3.5);
  const double denom = (1.0 + th_n) * (1.0 - th_np1);
  const double rhk =
      (denom > 0.0) ? (2.0 * 3.5 * th_n * (1.0 - theta) / denom) : 1.0;

  double s = -1.0;
  if (flux_limiter_faces != nullptr) {
    const double lam = flux_limiter_faces[f];
    if (lam > 0.0 && lam <= 1.0) {
      s = (1.0 - lam) / lam;
    }
  }
  const double reff = (s >= 0.0) ? (rhk * (1.0 + s) / (1.0 + rhk * s)) : rhk;

  r_hk[f] = rhk;
  r_eff[f] = reff;
  s_sat[f] = s;
  active[f] = static_cast<std::uint8_t>(1);
}

template <int GEOM>
__device__ void conduction_alpha_pass_kirchhoff_cell(
    const int i,
    double* __restrict__ alpha_cells,
    const double* __restrict__ Te_old,
    const double* __restrict__ kappa_sh,
    const double* __restrict__ flux_limiter_faces,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int n_cells,
    const double tau,
    const double Te_floor,
    const int floor_limiter_mode) {
  alpha_cells[i] = 1.0;
  if (cell_is_void != nullptr && cell_is_void[i] != static_cast<std::uint8_t>(0)) {
    return;
  }

  double V;
  if constexpr (GEOM == 2) {
    V = fmax(x_r[i + 1] - x_r[i], 1.0e-30);
  } else {
    V = fmax(vol[i], 1.0e-30);
  }

  double left_term = 0.0;
  if (i > 0) {
    const double k_face =
        kirchhoff_face_kappa_spitzer(kappa_sh[i - 1], kappa_sh[i], Te_old[i - 1], Te_old[i]);
    const double rcl = cell_center_radius(x_r, i - 1);
    const double rcr = cell_center_radius(x_r, i);
    const double dr = rcr - rcl;
    if (k_face > 0.0 && dr > 0.0) {
      const double grad = (Te_old[i] - Te_old[i - 1]) / dr;
      const double q_sh = k_face * grad;
      const double limiter =
          (flux_limiter_faces != nullptr) ? flux_limiter_faces[i - 1] : 1.0;
      const double q_face = q_sh * limiter;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i] * x_r[i];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i]);
      }
      left_term = area * q_face;
    }
  }

  double right_term = 0.0;
  if (i + 1 < n_cells) {
    const double k_face =
        kirchhoff_face_kappa_spitzer(kappa_sh[i], kappa_sh[i + 1], Te_old[i], Te_old[i + 1]);
    const double rcl = cell_center_radius(x_r, i);
    const double rcr = cell_center_radius(x_r, i + 1);
    const double dr = rcr - rcl;
    if (k_face > 0.0 && dr > 0.0) {
      const double grad = (Te_old[i + 1] - Te_old[i]) / dr;
      const double q_sh = k_face * grad;
      const double limiter = (flux_limiter_faces != nullptr) ? flux_limiter_faces[i] : 1.0;
      const double q_face = q_sh * limiter;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i + 1] * x_r[i + 1];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i + 1]);
      }
      right_term = area * q_face;
    }
  }

  const double M = (right_term - left_term) / V;
  const double rho_cv = rho_cv_e[i];
  const double Te_old_i = Te_old[i];
  if (rho_cv > 0.0) {
    if (floor_limiter_mode == 1) {
      const double outflow_sum =
          ((isfinite(right_term) ? fmax(0.0, -right_term) : 0.0) +
           (isfinite(left_term) ? fmax(0.0, left_term) : 0.0)) /
          V;
      double alpha = 1.0;
      if (!(Te_old_i > Te_floor)) {
        alpha = 0.0;
      } else if (tau > 0.0 && outflow_sum > 0.0) {
        const double e_above = rho_cv * (Te_old_i - Te_floor);
        alpha = fmin(
            1.0, fmax((e_above / fmax(outflow_sum, kDivEpsilon)) / tau, 0.0));
      }
      alpha_cells[i] = alpha;
      return;
    }
    double alpha = 1.0;
    if (tau > 0.0 && Te_old_i > Te_floor && M < 0.0) {
      const double e_above_floor = rho_cv * (Te_old_i - Te_floor);
      const double dt_safe = e_above_floor / fmax(fabs(M), kDivEpsilon);
      alpha = fmin(1.0, fmax(dt_safe / tau, 0.0));
    }
    alpha_cells[i] = alpha;
  }
}

template <int GEOM>
__global__ void conduction_alpha_pass_kirchhoff_kernel(
    double* __restrict__ alpha_cells,
    const double* __restrict__ Te_old,
    const double* __restrict__ kappa_sh,
    const double* __restrict__ flux_limiter_faces,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const double tau,
    const double Te_floor,
    const int floor_limiter_mode) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  conduction_alpha_pass_kirchhoff_cell<GEOM>(
      i, alpha_cells, Te_old, kappa_sh, flux_limiter_faces, rho_cv_e,
      cell_is_void, vol, x_r, n_cells, tau, Te_floor, floor_limiter_mode);
}

template <int GEOM>
__global__ void conduction_1d_sts_stage_kirchhoff_kernel(
    const double* __restrict__ Te_old,
    double* __restrict__ Te_new,
    const double* __restrict__ kappa_sh,
    const double* __restrict__ flux_limiter_faces,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const double tau,
    const double Te_floor,
    const double* __restrict__ alpha_cells,
    int* __restrict__ clamp_count,
    double* __restrict__ E_floor,
    const int floor_limiter_mode) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  conduction_bodies::conduction_1d_sts_stage_kirchhoff_kernel_body<GEOM>(
      i, Te_old, Te_new, kappa_sh, flux_limiter_faces, rho_cv_e, cell_is_void,
      vol, x_r, n_cells, tau, Te_floor, alpha_cells, clamp_count, E_floor,
      floor_limiter_mode);
}

template <int GEOM>
__global__ void conduction_1d_sts_fused_kirchhoff_kernel(
    double* __restrict__ Te_a,
    double* __restrict__ Te_b,
    double* __restrict__ alpha_cells,
    const double* __restrict__ kappa_sh,
    const double* __restrict__ flux_limiter_faces,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int n_cells,
    const double* __restrict__ stage_tau,
    const int n_stages,
    const double Te_floor,
    int* __restrict__ clamp_count,
    double* __restrict__ E_floor,
    const int floor_limiter_mode) {
  double* te_curr = Te_a;
  double* te_next = Te_b;
  for (int stage = 0; stage < n_stages; ++stage) {
    const double tau = stage_tau[stage];
    for (int i = threadIdx.x; i < n_cells; i += blockDim.x) {
      conduction_alpha_pass_kirchhoff_cell<GEOM>(
          i, alpha_cells, te_curr, kappa_sh, flux_limiter_faces, rho_cv_e,
          cell_is_void, vol, x_r, n_cells, tau, Te_floor, floor_limiter_mode);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < n_cells; i += blockDim.x) {
      conduction_bodies::conduction_1d_sts_stage_kirchhoff_kernel_body<GEOM>(
          i, te_curr, te_next, kappa_sh, flux_limiter_faces, rho_cv_e,
          cell_is_void, vol, x_r, n_cells, tau, Te_floor, alpha_cells,
          clamp_count, E_floor, floor_limiter_mode);
    }
    __syncthreads();

    double* const te_swap = te_curr;
    te_curr = te_next;
    te_next = te_swap;
  }
}

// W-G2 kirchhoff face_kappa_policy implicit assembly kernel: identical to
// build_1d_implicit_system_kernel except the face conductivity comes from
// kirchhoff_face_kappa_spitzer.
template <int GEOM>
__global__ void build_1d_implicit_system_kirchhoff_kernel(
    double* __restrict__ lower,
    double* __restrict__ diag,
    double* __restrict__ upper,
    double* __restrict__ rhs,
    const double* __restrict__ Te_old,
    const double* __restrict__ kappa_sh,
    const double* __restrict__ flux_limiter_faces,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int n_cells,
    const double dt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }

  const bool is_void =
      (cell_is_void != nullptr && cell_is_void[i] != static_cast<std::uint8_t>(0));
  const double Te_old_i = Te_old[i];
  const double rho_cv = rho_cv_e[i];
  if (is_void || !(rho_cv > 0.0) || !(dt > 0.0) || !isfinite(Te_old_i)) {
    lower[i] = 0.0;
    diag[i] = 1.0;
    upper[i] = 0.0;
    rhs[i] = isfinite(Te_old_i) ? Te_old_i : 0.0;
    return;
  }

  double V;
  if constexpr (GEOM == 2) {
    V = fmax(x_r[i + 1] - x_r[i], 1.0e-30);
  } else {
    V = fmax(vol[i], 1.0e-30);
  }
  const double mass = rho_cv * V;
  const double mass_over_dt = mass / dt;

  double left_w = 0.0;
  if (i > 0) {
    const bool left_void =
        (cell_is_void != nullptr && cell_is_void[i - 1] != static_cast<std::uint8_t>(0));
    if (!left_void && rho_cv_e[i - 1] > 0.0) {
      const double k_face =
          kirchhoff_face_kappa_spitzer(kappa_sh[i - 1], kappa_sh[i], Te_old[i - 1], Te_old[i]);
      const double limiter =
          (flux_limiter_faces != nullptr) ? flux_limiter_faces[i - 1] : 1.0;
      const double rcl = cell_center_radius(x_r, i - 1);
      const double rcr = cell_center_radius(x_r, i);
      const double dr = rcr - rcl;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i] * x_r[i];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i]);
      }
      if (k_face > 0.0 && limiter > 0.0 && dr > 0.0 && area > 0.0) {
        left_w = area * k_face * fmin(1.0, fmax(limiter, 0.0)) / dr;
      }
    }
  }

  double right_w = 0.0;
  if (i + 1 < n_cells) {
    const bool right_void =
        (cell_is_void != nullptr && cell_is_void[i + 1] != static_cast<std::uint8_t>(0));
    if (!right_void && rho_cv_e[i + 1] > 0.0) {
      const double k_face =
          kirchhoff_face_kappa_spitzer(kappa_sh[i], kappa_sh[i + 1], Te_old[i], Te_old[i + 1]);
      const double limiter = (flux_limiter_faces != nullptr) ? flux_limiter_faces[i] : 1.0;
      const double rcl = cell_center_radius(x_r, i);
      const double rcr = cell_center_radius(x_r, i + 1);
      const double dr = rcr - rcl;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i + 1] * x_r[i + 1];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i + 1]);
      }
      if (k_face > 0.0 && limiter > 0.0 && dr > 0.0 && area > 0.0) {
        right_w = area * k_face * fmin(1.0, fmax(limiter, 0.0)) / dr;
      }
    }
  }

  double lower_i = -left_w;
  double diag_i = mass_over_dt + left_w + right_w;
  double upper_i = -right_w;
  double rhs_i = mass_over_dt * Te_old_i;

  if (!isfinite(lower_i) || !isfinite(diag_i) || !isfinite(upper_i) || !isfinite(rhs_i) ||
      !(diag_i > 0.0)) {
    lower_i = 0.0;
    diag_i = 1.0;
    upper_i = 0.0;
    rhs_i = isfinite(Te_old_i) ? Te_old_i : 0.0;
  }

  lower[i] = (i > 0) ? lower_i : 0.0;
  diag[i] = diag_i;
  upper[i] = (i + 1 < n_cells) ? upper_i : 0.0;
  rhs[i] = rhs_i;
}

template <int GEOM>
__global__ void kirchhoff_dt_ratio_1d_kernel(
    double* __restrict__ min_ratio,
    const double* __restrict__ kappa_eff,
    const double* __restrict__ rho_cv_e,
    const double* __restrict__ Te,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int c_begin,
    const int c_end,
    const int n_cells) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  conduction_bodies::kirchhoff_dt_ratio_1d_kernel_body<GEOM>(
      i, min_ratio, kappa_eff, rho_cv_e, Te, cell_is_void, vol, x_r, n_cells);
}

// W-G2 nlheat verify hook stage kernel: identical to
// conduction_1d_sts_stage_kernel except the face conductivity is the Kirchhoff
// secant of kappa(T) = kappa0 * T^p,
//   kappa_face = kappa0 * (T_R^{p+1} - T_L^{p+1}) / ((p+1) (T_R - T_L)),
// which is the flux-exact face coefficient for degenerate power-law diffusion:
// the harmonic mean of cell kappas pins a compact-support (Pattle/Zel'dovich)
// front to the grid because the cold-side kappa ~ kappa(T_floor) chokes the
// face flux (measured: front motion ~1e-4 cells/step). With p = 0 the secant
// reduces to kappa0 exactly (IEEE pow(x,1) == x), matching the historic
// constant-kappa flux bitwise. Launched only under the
// TENRYU_CONDUCTION_TEST_KAPPA_POWER env hook; no flux limiter (test path).
template <int GEOM>
__global__ void conduction_alpha_pass_secant_kernel(
    double* __restrict__ alpha_cells,
    const double* __restrict__ Te_old,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const double* __restrict__ rho,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const double tau,
    const double Te_floor,
    const double kappa0,
    const double kappa_power,
    const double kappa_rho_power,
    const int floor_limiter_mode) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  alpha_cells[i] = 1.0;
  if (cell_is_void != nullptr && cell_is_void[i] != static_cast<std::uint8_t>(0)) {
    return;
  }

  double V;
  if constexpr (GEOM == 2) {
    V = fmax(x_r[i + 1] - x_r[i], 1.0e-30);
  } else {
    V = fmax(vol[i], 1.0e-30);
  }

  const double np1 = kappa_power + 1.0;

  double left_term = 0.0;
  if (i > 0) {
    const double te_l = sanitize_te_for_pow(Te_old[i - 1]);
    const double te_r = sanitize_te_for_pow(Te_old[i]);
    double k_face;
    if (kappa_power == 0.0) {
      // Exact constant-kappa shortcut: keeps the p = 0 hook path bitwise equal
      // to the historic flux (device pow(x, 1.0) is not correctly rounded).
      k_face = kappa0;
    } else if (te_l == te_r) {
      k_face = kappa0 * pow(te_l, kappa_power);
    } else {
      k_face = kappa0 * (pow(te_r, np1) - pow(te_l, np1)) / (np1 * (te_r - te_l));
    }
    // RMtV rho-power: arithmetic face mean; a == 0 -> pow == 1 exactly.
    k_face *= pow(0.5 * (fmax(rho[i - 1], 0.0) + fmax(rho[i], 0.0)),
                  kappa_rho_power);
    const double rcl = cell_center_radius(x_r, i - 1);
    const double rcr = cell_center_radius(x_r, i);
    const double dr = rcr - rcl;
    if (k_face > 0.0 && dr > 0.0) {
      const double grad = (Te_old[i] - Te_old[i - 1]) / dr;
      const double q_face = k_face * grad;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i] * x_r[i];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i]);
      }
      left_term = area * q_face;
    }
  }

  double right_term = 0.0;
  if (i + 1 < n_cells) {
    const double te_l = sanitize_te_for_pow(Te_old[i]);
    const double te_r = sanitize_te_for_pow(Te_old[i + 1]);
    double k_face;
    if (kappa_power == 0.0) {
      // Exact constant-kappa shortcut: keeps the p = 0 hook path bitwise equal
      // to the historic flux (device pow(x, 1.0) is not correctly rounded).
      k_face = kappa0;
    } else if (te_l == te_r) {
      k_face = kappa0 * pow(te_l, kappa_power);
    } else {
      k_face = kappa0 * (pow(te_r, np1) - pow(te_l, np1)) / (np1 * (te_r - te_l));
    }
    k_face *= pow(0.5 * (fmax(rho[i], 0.0) + fmax(rho[i + 1], 0.0)),
                  kappa_rho_power);
    const double rcl = cell_center_radius(x_r, i);
    const double rcr = cell_center_radius(x_r, i + 1);
    const double dr = rcr - rcl;
    if (k_face > 0.0 && dr > 0.0) {
      const double grad = (Te_old[i + 1] - Te_old[i]) / dr;
      const double q_face = k_face * grad;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i + 1] * x_r[i + 1];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i + 1]);
      }
      right_term = area * q_face;
    }
  }

  const double M = (right_term - left_term) / V;
  const double rho_cv = rho_cv_e[i];
  const double Te_old_i = Te_old[i];
  if (rho_cv > 0.0) {
    if (floor_limiter_mode == 1) {
      const double outflow_sum =
          ((isfinite(right_term) ? fmax(0.0, -right_term) : 0.0) +
           (isfinite(left_term) ? fmax(0.0, left_term) : 0.0)) /
          V;
      double alpha = 1.0;
      if (!(Te_old_i > Te_floor)) {
        alpha = 0.0;
      } else if (tau > 0.0 && outflow_sum > 0.0) {
        const double e_above = rho_cv * (Te_old_i - Te_floor);
        alpha = fmin(
            1.0, fmax((e_above / fmax(outflow_sum, kDivEpsilon)) / tau, 0.0));
      }
      alpha_cells[i] = alpha;
      return;
    }
    double alpha = 1.0;
    if (tau > 0.0 && Te_old_i > Te_floor && M < 0.0) {
      const double e_above_floor = rho_cv * (Te_old_i - Te_floor);
      const double dt_safe = e_above_floor / fmax(fabs(M), kDivEpsilon);
      alpha = fmin(1.0, fmax(dt_safe / tau, 0.0));
    }
    alpha_cells[i] = alpha;
  }
}

template <int GEOM>
__global__ void conduction_1d_sts_stage_secant_kernel(
    const double* __restrict__ Te_old,
    double* __restrict__ Te_new,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const double* __restrict__ rho,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const double tau,
    const double Te_floor,
    const double kappa0,
    const double kappa_power,
    const double kappa_rho_power,
    const double* __restrict__ alpha_cells,
    int* __restrict__ clamp_count,
    double* __restrict__ E_floor,
    const int floor_limiter_mode) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  conduction_bodies::conduction_1d_sts_stage_secant_kernel_body<GEOM>(
      i, Te_old, Te_new, rho_cv_e, cell_is_void, vol, x_r, rho, n_cells, tau,
      Te_floor, kappa0, kappa_power, kappa_rho_power, alpha_cells, clamp_count,
      E_floor, floor_limiter_mode);
}

template <int GEOM>
__global__ void conduction_alpha_pass_kernel_per_material(
    double* __restrict__ alpha_cells,
    const double* __restrict__ Te_old,
    const double* __restrict__ kappa_eff_per_material,
    const double* __restrict__ volfrac,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const int n_mat,
    const double tau,
    const double Te_floor,
    const int floor_limiter_mode) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  alpha_cells[i] = 1.0;
  if (cell_is_void != nullptr && cell_is_void[i] != static_cast<std::uint8_t>(0)) {
    return;
  }

  const int n_faces = n_cells - 1;
  double V;
  if constexpr (GEOM == 2) {
    V = fmax(x_r[i + 1] - x_r[i], 1.0e-30);
  } else {
    V = fmax(vol[i], 1.0e-30);
  }
  const double rho_cv = rho_cv_e[i];
  if (!(rho_cv > 0.0)) {
    return;
  }

  double net_power = 0.0;
  double outflow_sum = 0.0;
  for (int m = 0; m < n_mat; ++m) {
    double left_term_m = 0.0;
    if (i > 0) {
      const int face = i - 1;
      const double coeff_m = kappa_eff_per_material[static_cast<std::size_t>(m) * n_faces + face];
      const double vf_face =
          0.5 * (fmax(volfrac[(i - 1) * n_mat + m], 0.0) +
                 fmax(volfrac[i * n_mat + m], 0.0));
      const double rcl = cell_center_radius(x_r, i - 1);
      const double rcr = cell_center_radius(x_r, i);
      const double dr = rcr - rcl;
      if (coeff_m > 0.0 && vf_face > 0.0 && dr > 0.0) {
        const double grad = (Te_old[i] - Te_old[i - 1]) / dr;
        double area;
        if constexpr (GEOM == 0) {
          area = kFourPi * x_r[i] * x_r[i];
        } else {
          area = geometry_1d_face_area(GEOM, x_r[i]);
        }
        left_term_m = area * vf_face * coeff_m * grad;
      }
    }
    double right_term_m = 0.0;
    if (i + 1 < n_cells) {
      const int face = i;
      const double coeff_m = kappa_eff_per_material[static_cast<std::size_t>(m) * n_faces + face];
      const double vf_face =
          0.5 * (fmax(volfrac[i * n_mat + m], 0.0) +
                 fmax(volfrac[(i + 1) * n_mat + m], 0.0));
      const double rcl = cell_center_radius(x_r, i);
      const double rcr = cell_center_radius(x_r, i + 1);
      const double dr = rcr - rcl;
      if (coeff_m > 0.0 && vf_face > 0.0 && dr > 0.0) {
        const double grad = (Te_old[i + 1] - Te_old[i]) / dr;
        double area;
        if constexpr (GEOM == 0) {
          area = kFourPi * x_r[i + 1] * x_r[i + 1];
        } else {
          area = geometry_1d_face_area(GEOM, x_r[i + 1]);
        }
        right_term_m = area * vf_face * coeff_m * grad;
      }
    }
    net_power += right_term_m - left_term_m;
    if (floor_limiter_mode == 1) {
      outflow_sum +=
          (isfinite(right_term_m) ? fmax(0.0, -right_term_m) : 0.0) +
          (isfinite(left_term_m) ? fmax(0.0, left_term_m) : 0.0);
    }
  }

  double alpha = 1.0;
  const double Te_old_i = Te_old[i];
  if (floor_limiter_mode == 1) {
    if (!(Te_old_i > Te_floor)) {
      alpha = 0.0;
    } else if (tau > 0.0 && outflow_sum > 0.0) {
      const double e_above = rho_cv * V * (Te_old_i - Te_floor);
      alpha = fmin(
          1.0, fmax((e_above / fmax(outflow_sum, kDivEpsilon)) / tau, 0.0));
    }
    alpha_cells[i] = alpha;
    return;
  }
  if (tau > 0.0 && Te_old_i > Te_floor && net_power < 0.0) {
    const double e_above_floor = rho_cv * V * (Te_old_i - Te_floor);
    const double dt_safe = e_above_floor / fmax(fabs(net_power), kDivEpsilon);
    alpha = fmin(1.0, fmax(dt_safe / tau, 0.0));
  }
  alpha_cells[i] = alpha;
}

template <int GEOM>
__global__ void conduction_1d_sts_stage_kernel_per_material(
    const double* __restrict__ Te_old,
    double* __restrict__ Te_new,
    const double* __restrict__ kappa_face,
    const double* __restrict__ kappa_eff_per_material,
    double* __restrict__ Ee_per_material,
    const double* __restrict__ volfrac,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const int n_mat,
    const double tau,
    const double Te_floor,
    const double* __restrict__ alpha_cells,
    int* __restrict__ clamp_count,
    double* __restrict__ E_floor,
    const int floor_limiter_mode) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }
  if (cell_is_void != nullptr && cell_is_void[i] != static_cast<std::uint8_t>(0)) {
    Te_new[i] = Te_old[i];
    return;
  }

  const int n_faces = n_cells - 1;
  double V;
  if constexpr (GEOM == 2) {
    V = fmax(x_r[i + 1] - x_r[i], 1.0e-30);
  } else {
    V = fmax(vol[i], 1.0e-30);
  }
  const double rho_cv = rho_cv_e[i];
  if (!(rho_cv > 0.0)) {
    Te_new[i] = Te_old[i];
    return;
  }

  double net_power = 0.0;
  for (int m = 0; m < n_mat; ++m) {
    double left_term_m = 0.0;
    if (i > 0) {
      const int face = i - 1;
      const double coeff_m = kappa_eff_per_material[static_cast<std::size_t>(m) * n_faces + face];
      const double vf_face =
          0.5 * (fmax(volfrac[(i - 1) * n_mat + m], 0.0) +
                 fmax(volfrac[i * n_mat + m], 0.0));
      const double rcl = cell_center_radius(x_r, i - 1);
      const double rcr = cell_center_radius(x_r, i);
      const double dr = rcr - rcl;
      if (coeff_m > 0.0 && vf_face > 0.0 && dr > 0.0) {
        const double grad = (Te_old[i] - Te_old[i - 1]) / dr;
        double area;
        if constexpr (GEOM == 0) {
          area = kFourPi * x_r[i] * x_r[i];
        } else {
          area = geometry_1d_face_area(GEOM, x_r[i]);
        }
        left_term_m = area * vf_face * coeff_m * grad;
      }
    }
    double right_term_m = 0.0;
    if (i + 1 < n_cells) {
      const int face = i;
      const double coeff_m = kappa_eff_per_material[static_cast<std::size_t>(m) * n_faces + face];
      const double vf_face =
          0.5 * (fmax(volfrac[i * n_mat + m], 0.0) +
                 fmax(volfrac[(i + 1) * n_mat + m], 0.0));
      const double rcl = cell_center_radius(x_r, i);
      const double rcr = cell_center_radius(x_r, i + 1);
      const double dr = rcr - rcl;
      if (coeff_m > 0.0 && vf_face > 0.0 && dr > 0.0) {
        const double grad = (Te_old[i + 1] - Te_old[i]) / dr;
        double area;
        if constexpr (GEOM == 0) {
          area = kFourPi * x_r[i + 1] * x_r[i + 1];
        } else {
          area = geometry_1d_face_area(GEOM, x_r[i + 1]);
        }
        right_term_m = area * vf_face * coeff_m * grad;
      }
    }
    const double scale_left = (floor_limiter_mode == 1)
        ? ((i > 0) ? ((left_term_m > 0.0) ? alpha_cells[i]
                                               : alpha_cells[i - 1])
                   : alpha_cells[i])
        : ((i > 0) ? fmin(alpha_cells[i - 1], alpha_cells[i]) : alpha_cells[i]);
    const double scale_right = (floor_limiter_mode == 1)
        ? ((i + 1 < n_cells) ? ((right_term_m > 0.0) ? alpha_cells[i + 1]
                                                        : alpha_cells[i])
                             : alpha_cells[i])
        : ((i + 1 < n_cells) ? fmin(alpha_cells[i], alpha_cells[i + 1])
                             : alpha_cells[i]);
    left_term_m *= scale_left;
    right_term_m *= scale_right;
    net_power += right_term_m - left_term_m;
  }

  const double Te_old_i = Te_old[i];

  for (int m = 0; m < n_mat; ++m) {
    double left_term_m = 0.0;
    if (i > 0) {
      const int face = i - 1;
      const double coeff_m = kappa_eff_per_material[static_cast<std::size_t>(m) * n_faces + face];
      const double vf_face =
          0.5 * (fmax(volfrac[(i - 1) * n_mat + m], 0.0) +
                 fmax(volfrac[i * n_mat + m], 0.0));
      const double dr = cell_center_radius(x_r, i) - cell_center_radius(x_r, i - 1);
      if (coeff_m > 0.0 && vf_face > 0.0 && dr > 0.0) {
        const double grad = (Te_old[i] - Te_old[i - 1]) / dr;
        double area;
        if constexpr (GEOM == 0) {
          area = kFourPi * x_r[i] * x_r[i];
        } else {
          area = geometry_1d_face_area(GEOM, x_r[i]);
        }
        left_term_m = area * vf_face * coeff_m * grad;
      }
    }
    double right_term_m = 0.0;
    if (i + 1 < n_cells) {
      const int face = i;
      const double coeff_m = kappa_eff_per_material[static_cast<std::size_t>(m) * n_faces + face];
      const double vf_face =
          0.5 * (fmax(volfrac[i * n_mat + m], 0.0) +
                 fmax(volfrac[(i + 1) * n_mat + m], 0.0));
      const double dr = cell_center_radius(x_r, i + 1) - cell_center_radius(x_r, i);
      if (coeff_m > 0.0 && vf_face > 0.0 && dr > 0.0) {
        const double grad = (Te_old[i + 1] - Te_old[i]) / dr;
        double area;
        if constexpr (GEOM == 0) {
          area = kFourPi * x_r[i + 1] * x_r[i + 1];
        } else {
          area = geometry_1d_face_area(GEOM, x_r[i + 1]);
        }
        right_term_m = area * vf_face * coeff_m * grad;
      }
    }
    const int idx = i * n_mat + m;
    const double scale_left = (floor_limiter_mode == 1)
        ? ((i > 0) ? ((left_term_m > 0.0) ? alpha_cells[i]
                                               : alpha_cells[i - 1])
                   : alpha_cells[i])
        : ((i > 0) ? fmin(alpha_cells[i - 1], alpha_cells[i]) : alpha_cells[i]);
    const double scale_right = (floor_limiter_mode == 1)
        ? ((i + 1 < n_cells) ? ((right_term_m > 0.0) ? alpha_cells[i + 1]
                                                        : alpha_cells[i])
                             : alpha_cells[i])
        : ((i + 1 < n_cells) ? fmin(alpha_cells[i], alpha_cells[i + 1])
                             : alpha_cells[i]);
    left_term_m *= scale_left;
    right_term_m *= scale_right;
    const double dE = tau * (right_term_m - left_term_m);
    if (isfinite(dE)) {
      Ee_per_material[idx] = fmax(Ee_per_material[idx] + dE, 0.0);
    }
  }

  double Te_computed = Te_old_i + tau * net_power / (rho_cv * V);
  if (!isfinite(Te_computed) || Te_computed < Te_floor) {
    if (isfinite(Te_computed) && Te_floor > Te_computed) {
      atomic_add_double(E_floor, rho_cv * (Te_floor - Te_computed) * V);
    } else if (!isfinite(Te_computed) && Te_floor > 0.0) {
      atomic_add_double(E_floor, rho_cv * Te_floor * V);
    }
    Te_new[i] = Te_floor;
    atomicAdd(clamp_count, 1);
    return;
  }
  (void)kappa_face;
  Te_new[i] = Te_computed;
}

__global__ void derive_cell_ee_from_per_material_kernel(double* __restrict__ ee,
                                                        const double* __restrict__ Ee_m,
                                                        const double* __restrict__ mass,
                                                        const int c_begin,
                                                        const int c_end,
                                                        const int n_cells,
                                                        const int n_mat) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  double sum_Ee = 0.0;
  for (int m = 0; m < n_mat; ++m) {
    sum_Ee += Ee_m[c * n_mat + m];
  }
  ee[c] = (mass[c] > 0.0 && isfinite(sum_Ee)) ? (sum_Ee / mass[c]) : 0.0;
}

__global__ void kershaw_alpha_pass_per_material_energy_kernel(
    double* __restrict__ alpha_cells,
    const double* __restrict__ Te,
    const double* __restrict__ stencil_per_material,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const int n_mat,
    const double dt_sub,
    const double T_floor,
    const int nr,
    const int nz,
    const int floor_limiter_mode,
    const int c_begin,
    const int c_end) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = c_begin + tid;
  const int n_cells = nr * nz;
  if (c >= c_end) {
    return;
  }
  alpha_cells[c] = 1.0;
  if (cell_is_void != nullptr && cell_is_void[c] != static_cast<std::uint8_t>(0)) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int iE = (i + 1 < nr) ? (i + 1) : i;
  const int iW = (i > 0) ? (i - 1) : i;
  const int jN = (j + 1 < nz) ? (j + 1) : j;
  const int jS = (j > 0) ? (j - 1) : j;
  const int cE = kershaw::cell_index(iE, j, nz);
  const int cW = kershaw::cell_index(iW, j, nz);
  const int cN = kershaw::cell_index(i, jN, nz);
  const int cS = kershaw::cell_index(i, jS, nz);
  const int cNE = kershaw::cell_index(iE, jN, nz);
  const int cNW = kershaw::cell_index(iW, jN, nz);
  const int cSE = kershaw::cell_index(iE, jS, nz);
  const int cSW = kershaw::cell_index(iW, jS, nz);

  const double V = fmax(vol[c], 1.0e-30);
  const double rho_cv = rho_cv_e[c];
  if (!(rho_cv > 0.0)) {
    return;
  }
  constexpr double k4Pi = 4.0 * 3.141592653589793238462643383279502884;

  double net_power = 0.0;
  double outflow_sum = 0.0;
  for (int m = 0; m < n_mat; ++m) {
    const double* stencil_m =
        stencil_per_material + static_cast<std::size_t>(m) * n_cells * kershaw::kStencilSize;
    double power_m = 0.0;
    const double pE = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cE, kershaw::kE, kershaw::kW);
    power_m += pE;
    const double pW = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cW, kershaw::kW, kershaw::kE);
    power_m += pW;
    const double pN = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cN, kershaw::kN, kershaw::kS);
    power_m += pN;
    const double pS = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cS, kershaw::kS, kershaw::kN);
    power_m += pS;
    const double pNE = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cNE, kershaw::kNE, kershaw::kSW);
    power_m += pNE;
    const double pNW = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cNW, kershaw::kNW, kershaw::kSE);
    power_m += pNW;
    const double pSE = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cSE, kershaw::kSE, kershaw::kNW);
    power_m += pSE;
    const double pSW = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cSW, kershaw::kSW, kershaw::kNE);
    power_m += pSW;
    if (floor_limiter_mode == 1) {
      outflow_sum += isfinite(pE) ? fmax(0.0, -pE) : 0.0;
      outflow_sum += isfinite(pW) ? fmax(0.0, -pW) : 0.0;
      outflow_sum += isfinite(pN) ? fmax(0.0, -pN) : 0.0;
      outflow_sum += isfinite(pS) ? fmax(0.0, -pS) : 0.0;
      outflow_sum += isfinite(pNE) ? fmax(0.0, -pNE) : 0.0;
      outflow_sum += isfinite(pNW) ? fmax(0.0, -pNW) : 0.0;
      outflow_sum += isfinite(pSE) ? fmax(0.0, -pSE) : 0.0;
      outflow_sum += isfinite(pSW) ? fmax(0.0, -pSW) : 0.0;
    }
    net_power += power_m * k4Pi;
  }

  double alpha = 1.0;
  const double Te_old_c = Te[c];
  if (floor_limiter_mode == 1) {
    outflow_sum *= k4Pi;
    if (!(Te_old_c > T_floor)) {
      alpha = 0.0;
    } else if (dt_sub > 0.0 && outflow_sum > 0.0) {
      const double e_above = rho_cv * V * (Te_old_c - T_floor);
      alpha = fmin(
          1.0, fmax((e_above / fmax(outflow_sum, kDivEpsilon)) / dt_sub, 0.0));
    }
    alpha_cells[c] = alpha;
    return;
  }
  if (dt_sub > 0.0 && Te_old_c > T_floor && net_power < 0.0) {
    const double energy_above_floor = rho_cv * V * (Te_old_c - T_floor);
    const double dt_safe = energy_above_floor / fmax(fabs(net_power), kDivEpsilon);
    alpha = fmin(1.0, fmax(dt_safe / dt_sub, 0.0));
  }
  alpha_cells[c] = alpha;
}

__global__ void kershaw_apply_per_material_energy_kernel(
    double* __restrict__ Te_new,
    const double* __restrict__ Te,
    const double* __restrict__ stencil_per_material,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    double* __restrict__ Ee_per_material,
    const int n_mat,
    const double dt_sub,
    const double T_floor,
    const int nr,
    const int nz,
    const double* __restrict__ alpha_cells,
    int* __restrict__ clamp_count,
    double* __restrict__ E_floor,
    const int floor_limiter_mode,
    const int c_begin,
    const int c_end) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = c_begin + tid;
  const int n_cells = nr * nz;
  if (c >= c_end) {
    return;
  }
  if (cell_is_void != nullptr && cell_is_void[c] != static_cast<std::uint8_t>(0)) {
    Te_new[c] = Te[c];
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int iE = (i + 1 < nr) ? (i + 1) : i;
  const int iW = (i > 0) ? (i - 1) : i;
  const int jN = (j + 1 < nz) ? (j + 1) : j;
  const int jS = (j > 0) ? (j - 1) : j;
  const int cE = kershaw::cell_index(iE, j, nz);
  const int cW = kershaw::cell_index(iW, j, nz);
  const int cN = kershaw::cell_index(i, jN, nz);
  const int cS = kershaw::cell_index(i, jS, nz);
  const int cNE = kershaw::cell_index(iE, jN, nz);
  const int cNW = kershaw::cell_index(iW, jN, nz);
  const int cSE = kershaw::cell_index(iE, jS, nz);
  const int cSW = kershaw::cell_index(iW, jS, nz);
  const double alpha_c = alpha_cells[c];

  const double V = fmax(vol[c], 1.0e-30);
  const double rho_cv = rho_cv_e[c];
  if (!(rho_cv > 0.0)) {
    Te_new[c] = Te[c];
    return;
  }
  constexpr double k4Pi = 4.0 * 3.141592653589793238462643383279502884;

  double net_power = 0.0;
  for (int m = 0; m < n_mat; ++m) {
    const double* stencil_m =
        stencil_per_material + static_cast<std::size_t>(m) * n_cells * kershaw::kStencilSize;
    double power_m = 0.0;
    const double pE = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cE, kershaw::kE, kershaw::kW);
    const double scaleE = (floor_limiter_mode == 1)
        ? ((pE > 0.0) ? alpha_cells[cE] : alpha_c)
        : fmin(alpha_c, alpha_cells[cE]);
    power_m += pE * scaleE;
    const double pW = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cW, kershaw::kW, kershaw::kE);
    const double scaleW = (floor_limiter_mode == 1)
        ? ((pW > 0.0) ? alpha_cells[cW] : alpha_c)
        : fmin(alpha_c, alpha_cells[cW]);
    power_m += pW * scaleW;
    const double pN = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cN, kershaw::kN, kershaw::kS);
    const double scaleN = (floor_limiter_mode == 1)
        ? ((pN > 0.0) ? alpha_cells[cN] : alpha_c)
        : fmin(alpha_c, alpha_cells[cN]);
    power_m += pN * scaleN;
    const double pS = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cS, kershaw::kS, kershaw::kN);
    const double scaleS = (floor_limiter_mode == 1)
        ? ((pS > 0.0) ? alpha_cells[cS] : alpha_c)
        : fmin(alpha_c, alpha_cells[cS]);
    power_m += pS * scaleS;
    const double pNE = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cNE, kershaw::kNE, kershaw::kSW);
    const double scaleNE = (floor_limiter_mode == 1)
        ? ((pNE > 0.0) ? alpha_cells[cNE] : alpha_c)
        : fmin(alpha_c, alpha_cells[cNE]);
    power_m += pNE * scaleNE;
    const double pNW = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cNW, kershaw::kNW, kershaw::kSE);
    const double scaleNW = (floor_limiter_mode == 1)
        ? ((pNW > 0.0) ? alpha_cells[cNW] : alpha_c)
        : fmin(alpha_c, alpha_cells[cNW]);
    power_m += pNW * scaleNW;
    const double pSE = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cSE, kershaw::kSE, kershaw::kNW);
    const double scaleSE = (floor_limiter_mode == 1)
        ? ((pSE > 0.0) ? alpha_cells[cSE] : alpha_c)
        : fmin(alpha_c, alpha_cells[cSE]);
    power_m += pSE * scaleSE;
    const double pSW = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cSW, kershaw::kSW, kershaw::kNE);
    const double scaleSW = (floor_limiter_mode == 1)
        ? ((pSW > 0.0) ? alpha_cells[cSW] : alpha_c)
        : fmin(alpha_c, alpha_cells[cSW]);
    power_m += pSW * scaleSW;
    net_power += power_m * k4Pi;
  }

  const double Te_old_c = Te[c];

  for (int m = 0; m < n_mat; ++m) {
    const double* stencil_m =
        stencil_per_material + static_cast<std::size_t>(m) * n_cells * kershaw::kStencilSize;
    double power_m = 0.0;
    const double pE = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cE, kershaw::kE, kershaw::kW);
    const double scaleE = (floor_limiter_mode == 1)
        ? ((pE > 0.0) ? alpha_cells[cE] : alpha_c)
        : fmin(alpha_c, alpha_cells[cE]);
    power_m += pE * scaleE;
    const double pW = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cW, kershaw::kW, kershaw::kE);
    const double scaleW = (floor_limiter_mode == 1)
        ? ((pW > 0.0) ? alpha_cells[cW] : alpha_c)
        : fmin(alpha_c, alpha_cells[cW]);
    power_m += pW * scaleW;
    const double pN = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cN, kershaw::kN, kershaw::kS);
    const double scaleN = (floor_limiter_mode == 1)
        ? ((pN > 0.0) ? alpha_cells[cN] : alpha_c)
        : fmin(alpha_c, alpha_cells[cN]);
    power_m += pN * scaleN;
    const double pS = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cS, kershaw::kS, kershaw::kN);
    const double scaleS = (floor_limiter_mode == 1)
        ? ((pS > 0.0) ? alpha_cells[cS] : alpha_c)
        : fmin(alpha_c, alpha_cells[cS]);
    power_m += pS * scaleS;
    const double pNE = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cNE, kershaw::kNE, kershaw::kSW);
    const double scaleNE = (floor_limiter_mode == 1)
        ? ((pNE > 0.0) ? alpha_cells[cNE] : alpha_c)
        : fmin(alpha_c, alpha_cells[cNE]);
    power_m += pNE * scaleNE;
    const double pNW = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cNW, kershaw::kNW, kershaw::kSE);
    const double scaleNW = (floor_limiter_mode == 1)
        ? ((pNW > 0.0) ? alpha_cells[cNW] : alpha_c)
        : fmin(alpha_c, alpha_cells[cNW]);
    power_m += pNW * scaleNW;
    const double pSE = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cSE, kershaw::kSE, kershaw::kNW);
    const double scaleSE = (floor_limiter_mode == 1)
        ? ((pSE > 0.0) ? alpha_cells[cSE] : alpha_c)
        : fmin(alpha_c, alpha_cells[cSE]);
    power_m += pSE * scaleSE;
    const double pSW = kershaw::symmetric_pair_power(
        stencil_m, Te, rho_cv_e, cell_is_void, c, cSW, kershaw::kSW, kershaw::kNE);
    const double scaleSW = (floor_limiter_mode == 1)
        ? ((pSW > 0.0) ? alpha_cells[cSW] : alpha_c)
        : fmin(alpha_c, alpha_cells[cSW]);
    power_m += pSW * scaleSW;
    const double dE = dt_sub * power_m * k4Pi;
    const int idx = c * n_mat + m;
    if (isfinite(dE)) {
      Ee_per_material[idx] = fmax(Ee_per_material[idx] + dE, 0.0);
    }
  }

  const double energy_trial = rho_cv * V * Te_old_c + dt_sub * net_power;
  const double Te_trial = energy_trial / (rho_cv * V);
  if (!isfinite(Te_trial) || Te_trial < T_floor) {
    if (isfinite(Te_trial) && Te_trial < T_floor) {
      atomic_add_double(E_floor, rho_cv * (T_floor - Te_trial) * V);
    } else if (!isfinite(Te_trial) && T_floor > 0.0) {
      atomic_add_double(E_floor, rho_cv * T_floor * V);
    }
    Te_new[c] = T_floor;
    atomicAdd(clamp_count, 1);
    return;
  }
  Te_new[c] = Te_trial;
}

__global__ void compute_1d_flux_limiter_faces_kernel(
    double* __restrict__ flux_limiter_faces,
    const double* __restrict__ Te,
    const double* __restrict__ kappa_sh,
    const double* __restrict__ rho,
    const double* __restrict__ zbar,
    const double* __restrict__ A_eff,
    const double* __restrict__ x_r,
    const int n_cells,
    const double f_lim,
    const bool kirchhoff_face) {
  const int face = blockIdx.x * blockDim.x + threadIdx.x;
  if (face + 1 >= n_cells) {
    return;
  }

  conduction_bodies::compute_1d_flux_limiter_faces_kernel_body(
      face, flux_limiter_faces, Te, kappa_sh, rho, zbar, A_eff, x_r, n_cells,
      f_lim, kirchhoff_face);
}

template <int GEOM>
__global__ void build_1d_implicit_system_kernel(
    double* __restrict__ lower,
    double* __restrict__ diag,
    double* __restrict__ upper,
    double* __restrict__ rhs,
    const double* __restrict__ Te_old,
    const double* __restrict__ kappa_sh,
    const double* __restrict__ flux_limiter_faces,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int n_cells,
    const double dt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }

  const bool is_void =
      (cell_is_void != nullptr && cell_is_void[i] != static_cast<std::uint8_t>(0));
  const double Te_old_i = Te_old[i];
  const double rho_cv = rho_cv_e[i];
  if (is_void || !(rho_cv > 0.0) || !(dt > 0.0) || !isfinite(Te_old_i)) {
    lower[i] = 0.0;
    diag[i] = 1.0;
    upper[i] = 0.0;
    rhs[i] = isfinite(Te_old_i) ? Te_old_i : 0.0;
    return;
  }

  double V;
  if constexpr (GEOM == 2) {
    V = fmax(x_r[i + 1] - x_r[i], 1.0e-30);
  } else {
    V = fmax(vol[i], 1.0e-30);
  }
  const double mass = rho_cv * V;
  const double mass_over_dt = mass / dt;

  double left_w = 0.0;
  if (i > 0) {
    const bool left_void =
        (cell_is_void != nullptr && cell_is_void[i - 1] != static_cast<std::uint8_t>(0));
    if (!left_void && rho_cv_e[i - 1] > 0.0) {
      const double k_face = harmonic_mean(kappa_sh[i - 1], kappa_sh[i]);
      const double limiter =
          (flux_limiter_faces != nullptr) ? flux_limiter_faces[i - 1] : 1.0;
      const double rcl = cell_center_radius(x_r, i - 1);
      const double rcr = cell_center_radius(x_r, i);
      const double dr = rcr - rcl;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i] * x_r[i];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i]);
      }
      if (k_face > 0.0 && limiter > 0.0 && dr > 0.0 && area > 0.0) {
        left_w = area * k_face * fmin(1.0, fmax(limiter, 0.0)) / dr;
      }
    }
  }

  double right_w = 0.0;
  if (i + 1 < n_cells) {
    const bool right_void =
        (cell_is_void != nullptr && cell_is_void[i + 1] != static_cast<std::uint8_t>(0));
    if (!right_void && rho_cv_e[i + 1] > 0.0) {
      const double k_face = harmonic_mean(kappa_sh[i], kappa_sh[i + 1]);
      const double limiter = (flux_limiter_faces != nullptr) ? flux_limiter_faces[i] : 1.0;
      const double rcl = cell_center_radius(x_r, i);
      const double rcr = cell_center_radius(x_r, i + 1);
      const double dr = rcr - rcl;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i + 1] * x_r[i + 1];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i + 1]);
      }
      if (k_face > 0.0 && limiter > 0.0 && dr > 0.0 && area > 0.0) {
        right_w = area * k_face * fmin(1.0, fmax(limiter, 0.0)) / dr;
      }
    }
  }

  double lower_i = -left_w;
  double diag_i = mass_over_dt + left_w + right_w;
  double upper_i = -right_w;
  double rhs_i = mass_over_dt * Te_old_i;

  if (!isfinite(lower_i) || !isfinite(diag_i) || !isfinite(upper_i) || !isfinite(rhs_i) ||
      !(diag_i > 0.0)) {
    lower_i = 0.0;
    diag_i = 1.0;
    upper_i = 0.0;
    rhs_i = isfinite(Te_old_i) ? Te_old_i : 0.0;
  }

  lower[i] = (i > 0) ? lower_i : 0.0;
  diag[i] = diag_i;
  upper[i] = (i + 1 < n_cells) ? upper_i : 0.0;
  rhs[i] = rhs_i;
}

template <int GEOM>
__global__ void clamp_1d_conduction_solution_kernel(
    double* __restrict__ Te,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int n_cells,
    const double Te_floor,
    int* __restrict__ clamp_count,
    double* __restrict__ E_floor) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }

  if (cell_is_void != nullptr && cell_is_void[i] != static_cast<std::uint8_t>(0)) {
    return;
  }

  const double Te_i = Te[i];
  if (isfinite(Te_i) && Te_i >= Te_floor) {
    return;
  }

  double V;
  if constexpr (GEOM == 2) {
    V = fmax(x_r[i + 1] - x_r[i], 1.0e-30);
  } else {
    V = fmax(vol[i], 1.0e-30);
  }
  const double rho_cv = rho_cv_e[i];
  if (rho_cv > 0.0) {
    if (isfinite(Te_i) && Te_floor > Te_i) {
      atomic_add_double(E_floor, rho_cv * (Te_floor - Te_i) * V);
    } else if (!isfinite(Te_i) && Te_floor > 0.0) {
      atomic_add_double(E_floor, rho_cv * Te_floor * V);
    }
  }

  Te[i] = Te_floor;
  atomicAdd(clamp_count, 1);
}

__global__ void eos_sync_electron_kernel(double* __restrict__ Te,
                                         double* __restrict__ ee,
                                         double* __restrict__ Pe,
                                         const double* __restrict__ rho,
                                         const double* __restrict__ zbar,
                                         const double* __restrict__ gamma_eff,
                                         const double* __restrict__ A_eff,
                                         const int c_begin,
                                         const int c_end,
                                         const int n_cells,
                                         const double Te_floor,
                                         const double* __restrict__ state_cv_e) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  conduction_bodies::eos_sync_electron_kernel_body(
      i, Te, ee, Pe, rho, zbar, gamma_eff, A_eff, n_cells, Te_floor,
      state_cv_e);
}

double compute_dt_cond_from_dt_exp(const double dt_exp, const int sts_max_stages);

void copy_material_params_to_device(const core::Config& cfg,
                                    ConductionMaterialParams** d_params) {
  const std::vector<ConductionMaterialParams> h_params =
      make_conduction_material_params(cfg);
  *d_params = static_cast<ConductionMaterialParams*>(core::device_scratch_acquire(
      "conduction:material_params",
      h_params.size() * sizeof(ConductionMaterialParams)));
  cuda_check(cudaMemcpy(*d_params,
                        h_params.data(),
                        h_params.size() * sizeof(ConductionMaterialParams),
                        cudaMemcpyHostToDevice),
             "Conduction: cudaMemcpy per-material params failed");
}

ConductionDiagnostics finish_conduction_diag(double* d_diag3,
                                             const core::Config& cfg) {
  ConductionDiagnostics diag;
  diag.dt_exp = std::numeric_limits<double>::infinity();
  diag.dt_cond = std::numeric_limits<double>::infinity();
  diag.deff_min = 0.0;
  diag.deff_max = 0.0;

  double h_diag3[3] = {0.0, 0.0, 0.0};
  cuda_check(cudaMemcpy(h_diag3, d_diag3, sizeof(h_diag3), cudaMemcpyDeviceToHost),
             "Conduction: cudaMemcpy diag3 pack failed");
  const double min_ratio = h_diag3[0];
  const double min_deff = h_diag3[1];
  const double max_deff = h_diag3[2];
  if (std::isfinite(min_deff) && min_deff > 0.0) {
    diag.deff_min = min_deff;
  }
  if (std::isfinite(max_deff) && max_deff > 0.0) {
    diag.deff_max = max_deff;
  }
  if (std::isfinite(min_ratio) && min_ratio > 0.0) {
    diag.dt_exp = cfg.numerics.dt.cfl_cond * min_ratio;
    diag.dt_cond =
        compute_dt_cond_from_dt_exp(diag.dt_exp, cfg.numerics.conduction.sts_max_stages);
  }
  return diag;
}

void log_plain_1d_dtexp_debug(const core::State& state,
                              const double* d_kappa_eff,
                              const double* d_rho_cv_e,
                              const double dt_exp) {
  if (std::getenv("TENRYU_COND_DTEXP_DEBUG") == nullptr || state.mesh.dim != 1 ||
      d_kappa_eff == nullptr || d_rho_cv_e == nullptr ||
      !std::isfinite(dt_exp) || !(dt_exp > 0.0)) {
    return;
  }

  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0) {
    return;
  }

  std::vector<double> kappa_eff(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> rho_cv(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> rho(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> x_r(static_cast<std::size_t>(n_cells + 1), 0.0);
  cuda_check(cudaMemcpy(kappa_eff.data(),
                        d_kappa_eff,
                        static_cast<std::size_t>(n_cells) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Conduction: copy kappa_eff for dt_exp debug failed");
  cuda_check(cudaMemcpy(rho_cv.data(),
                        d_rho_cv_e,
                        static_cast<std::size_t>(n_cells) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Conduction: copy rho_cv for dt_exp debug failed");
  state.rho.copy_to_host(rho);
  state.x_r.copy_to_host(x_r);

  int argmin_cell = -1;
  double argmin_ratio = std::numeric_limits<double>::infinity();
  double argmin_dl = 0.0;
  double argmin_deff = 0.0;
  for (int i = 0; i < n_cells; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    const double kappa = kappa_eff[idx];
    const double rho_cv_i = rho_cv[idx];
    const double dl = x_r[idx + 1] - x_r[idx];
    if (!(kappa > 0.0) || !(rho_cv_i > 0.0) || !(dl > 0.0) ||
        !std::isfinite(kappa) || !std::isfinite(rho_cv_i) || !std::isfinite(dl)) {
      continue;
    }
    const double D_eff = kappa / rho_cv_i;
    if (!(D_eff > 0.0) || !std::isfinite(D_eff)) {
      continue;
    }
    const double ratio = (dl * dl) / D_eff;
    if (!(ratio > 0.0) || !std::isfinite(ratio)) {
      continue;
    }
    if (ratio < argmin_ratio) {
      argmin_ratio = ratio;
      argmin_cell = i;
      argmin_dl = dl;
      argmin_deff = (dl * dl) / ratio;
    }
  }

  if (argmin_cell < 0) {
    return;
  }

  const std::size_t idx = static_cast<std::size_t>(argmin_cell);
  char line[256] = {};
  std::snprintf(line,
                sizeof(line),
                "[cond_dtexp_debug] cell=%d kappa=%e rho=%e rho_cv=%e dl=%e D_eff=%e dt_exp=%e",
                argmin_cell,
                kappa_eff[idx],
                rho[idx],
                rho_cv[idx],
                argmin_dl,
                argmin_deff,
                dt_exp);
  core::log_info(std::string(line));
}

ConductionDiagnostics compute_per_material_1d_operator(
    const core::State& state,
    const core::Config& cfg,
    const HydroEOSContext* eos_ctx,
    double* d_kappa_face,
    double* d_kappa_eff_per_material,
    double* d_rho_cv_e) {
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  TENRYU_ASSERT(n_cells > 0, "per-material 1D conduction requires cells");
  TENRYU_ASSERT(n_mat > 0, "per-material 1D conduction requires materials");

  const auto view = make_conduction_per_material_view(state, cfg, n_cells, n_mat);
  ConductionMaterialParams* d_params = nullptr;
  copy_material_params_to_device(cfg, &d_params);

  std::uint8_t* d_cell_is_void = nullptr;
  d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "conduction:compute_per_material_1d_operator:d_cell_is_void",
      static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t)));
  cuda_check(cudaMemcpy(d_cell_is_void,
                        state.cell_is_void.data(),
                        static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             "Conduction: cudaMemcpy per-material cell_is_void failed");

  double* d_diag3 = static_cast<double*>(core::device_scratch_acquire(
      "conduction:compute_per_material_1d_operator:diag3_pack", 3 * sizeof(double)));
  double* d_min_ratio = d_diag3 + 0;
  double* d_min_deff = d_diag3 + 1;
  double* d_max_deff = d_diag3 + 2;
  const double inf = std::numeric_limits<double>::infinity();
  const double h_diag3_init[3] = {inf, inf, 0.0};
  cuda_check(cudaMemcpy(d_diag3, h_diag3_init, sizeof(h_diag3_init),
                        cudaMemcpyHostToDevice),
             "Conduction: init per-material diag3 pack failed");

  const core::State::LaunchWindow dw = state.owned_cell_window_ghost(n_cells, 1);
  const auto* d_electron_views =
      (eos_ctx != nullptr && eos_ctx->n_materials >= n_mat) ? eos_ctx->d_electron_views
                                                            : nullptr;
  compute_spitzer_deff_1d_kernel_per_material<<<dw.blocks(), kBlockSize>>>(
      d_kappa_face,
      d_kappa_eff_per_material,
      d_rho_cv_e,
      d_min_ratio,
      d_min_deff,
      d_max_deff,
      view,
      d_electron_views,
      d_params,
      state.Te.data(),
      state.rho.data(),
      state.vol.data(),
      state.x_r.data(),
      dw.begin,
      dw.end,
      n_cells,
      n_mat,
      d_cell_is_void,
      cfg.numerics.conduction.f_lim,
      cfg.numerics.conduction.test_kappa,
      cfg.numerics.conduction.mfp_limiter_C,
      state.cv_e.empty() ? nullptr : state.cv_e.data(),
      cfg.numerics.floors.Te,
      cfg.materials.low_density_extrapolation);
  cuda_check(cudaGetLastError(), "Conduction: per-material 1D Spitzer launch failed");
  cuda_check(core::debug_kernel_sync(), "Conduction: per-material 1D Spitzer failed");
  const_cast<core::State&>(state).dispatch_counters.per_material_kernel_call_count.fetch_add(
      1, std::memory_order_relaxed);

  ConductionDiagnostics diag = finish_conduction_diag(d_diag3, cfg);
  return diag;
}

ConductionDiagnostics compute_per_material_2d_operator(
    const core::State& state,
    const core::Config& cfg,
    const HydroEOSContext* eos_ctx,
    double* d_kappa_eff,
    double* d_kappa_eff_per_material,
    double* d_rho_cv_e) {
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  TENRYU_ASSERT(nr * nz == n_cells, "per-material 2D conduction requires nr*nz cells");

  const auto view = make_conduction_per_material_view(state, cfg, n_cells, n_mat);
  ConductionMaterialParams* d_params = nullptr;
  copy_material_params_to_device(cfg, &d_params);

  std::uint8_t* d_cell_is_void = nullptr;
  d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "conduction:compute_per_material_2d_operator:d_cell_is_void",
      static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t)));
  cuda_check(cudaMemcpy(d_cell_is_void,
                        state.cell_is_void.data(),
                        static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             "Conduction: cudaMemcpy per-material 2D cell_is_void failed");

  double* d_diag3 = static_cast<double*>(core::device_scratch_acquire(
      "conduction:compute_per_material_2d_operator:diag3_pack", 3 * sizeof(double)));
  double* d_min_ratio = d_diag3 + 0;
  double* d_min_deff = d_diag3 + 1;
  double* d_max_deff = d_diag3 + 2;
  const double inf = std::numeric_limits<double>::infinity();
  const double h_diag3_init[3] = {inf, inf, 0.0};
  cuda_check(cudaMemcpy(d_diag3, h_diag3_init, sizeof(h_diag3_init),
                        cudaMemcpyHostToDevice),
             "Conduction: init per-material 2D diag3 pack failed");

  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  const auto* d_electron_views =
      (eos_ctx != nullptr && eos_ctx->n_materials >= n_mat) ? eos_ctx->d_electron_views
                                                            : nullptr;
  compute_spitzer_deff_2d_kernel_per_material<<<cw.blocks(), kBlockSize>>>(
      d_kappa_eff,
      d_kappa_eff_per_material,
      d_rho_cv_e,
      d_min_ratio,
      d_min_deff,
      d_max_deff,
      view,
      d_electron_views,
      d_params,
      state.Te.data(),
      state.rho.data(),
      state.vol.data(),
      state.x_r.data(),
      state.x_z.data(),
      nr,
      nz,
      n_mat,
      d_cell_is_void,
      cfg.numerics.conduction.f_lim,
      cfg.numerics.conduction.test_kappa,
      cfg.numerics.conduction.mfp_limiter_C,
      state.cv_e.empty() ? nullptr : state.cv_e.data(),
      cfg.numerics.floors.Te,
      cfg.materials.low_density_extrapolation,
      cw.begin,
      cw.end);
  cuda_check(cudaGetLastError(), "Conduction: per-material 2D Spitzer launch failed");
  cuda_check(core::debug_kernel_sync(), "Conduction: per-material 2D Spitzer failed");
  const_cast<core::State&>(state).dispatch_counters.per_material_kernel_call_count.fetch_add(
      1, std::memory_order_relaxed);

  ConductionDiagnostics diag = finish_conduction_diag(d_diag3, cfg);
  return diag;
}

ConductionDiagnostics compute_conduction_diagnostics_impl(const core::State& state,
                                                          const core::Config& cfg,
                                                          double* d_diffusion,
                                                          double* d_rho_cv_e,
                                                          const HydroEOSContext* eos_ctx) {
  ConductionDiagnostics diag;
  diag.dt_exp = std::numeric_limits<double>::infinity();
  diag.dt_cond = std::numeric_limits<double>::infinity();
  diag.deff_min = 0.0;
  diag.deff_max = 0.0;

  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0) {
    return diag;
  }

  TENRYU_ASSERT(state.Te.size() == state.rho.size(), "Conduction Te/rho size mismatch");
  TENRYU_ASSERT(state.zbar.size() == state.rho.size(),
                "Conduction zbar/rho size mismatch");
  TENRYU_ASSERT(state.vol.size() == state.rho.size(), "Conduction vol/rho size mismatch");
  TENRYU_ASSERT(state.cell_is_void.size() == state.rho.size(),
                "Conduction cell_is_void/rho size mismatch");
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "Conduction requires at least one material");

  if (state.mesh.dim == 1) {
    TENRYU_ASSERT(state.x_r.size() == state.rho.size() + 1,
                  "Conduction 1D requires node count = cell count + 1");
  } else if (state.mesh.dim == 2) {
    TENRYU_ASSERT(state.mesh.topo.n_cells == n_cells,
                  "Conduction 2D requires mesh/state cell consistency");
    TENRYU_ASSERT(state.x_r.size() == state.x_z.size(),
                  "Conduction 2D requires matching node coordinate arrays");
    TENRYU_ASSERT(state.x_r.size() == static_cast<std::size_t>(state.mesh.topo.n_nodes),
                  "Conduction 2D requires mesh/state node consistency");
  } else {
    return diag;
  }

  // dt-lineage argmin callers pass d_rho_cv_e == nullptr, but the Kirchhoff
  // dt-ratio kernel below dereferences it per cell (null deref caught by
  // compute-sanitizer under TENRYU_DT_LINEAGE=1). Bind a zeroed scratch that
  // the per-material/deff kernels then fill; callers passing a real buffer
  // never enter this branch.
  if (d_rho_cv_e == nullptr) {
    d_rho_cv_e = static_cast<double*>(core::device_scratch_acquire(
        "conduction:compute_conduction_diagnostics_impl:d_rho_cv_e_local",
        static_cast<std::size_t>(n_cells) * sizeof(double)));
    cuda_check(cudaMemset(d_rho_cv_e, 0,
                          static_cast<std::size_t>(n_cells) * sizeof(double)),
               "Conduction: cudaMemset local rho_cv_e failed");
  }

  if (per_material_conduction_ready(state, cfg, eos_ctx)) {
    const int n_mat = static_cast<int>(cfg.materials.materials.size());
    double* d_rho_cv_local = d_rho_cv_e;
    if (d_rho_cv_local == nullptr) {
      d_rho_cv_local = static_cast<double*>(core::device_scratch_acquire(
          "conduction:compute_conduction_diagnostics_impl:d_rho_cv_local",
          static_cast<std::size_t>(n_cells) * sizeof(double)));
    }
    if (state.mesh.dim == 1) {
      const int n_faces = std::max(0, n_cells - 1);
      double* d_kappa_face = nullptr;
      double* d_kappa_pm = nullptr;
      d_kappa_face = static_cast<double*>(core::device_scratch_acquire(
          "conduction:compute_conduction_diagnostics_impl:d_kappa_face",
          static_cast<std::size_t>(std::max(1, n_faces)) *
              sizeof(double)));
      d_kappa_pm = static_cast<double*>(core::device_scratch_acquire(
          "conduction:compute_conduction_diagnostics_impl:d_kappa_pm_faces",
          static_cast<std::size_t>(std::max(1, n_faces * n_mat)) *
              sizeof(double)));
      diag = compute_per_material_1d_operator(
          state, cfg, eos_ctx, d_kappa_face, d_kappa_pm, d_rho_cv_local);
      if (d_diffusion != nullptr) {
        cuda_check(cudaMemset(d_diffusion,
                              0,
                              static_cast<std::size_t>(n_cells) * sizeof(double)),
                   "Conduction: cudaMemset per-material diagnostics diffusion failed");
      }
    } else {
      double* d_kappa_local = d_diffusion;
      if (d_kappa_local == nullptr) {
        d_kappa_local = static_cast<double*>(core::device_scratch_acquire(
            "conduction:compute_conduction_diagnostics_impl:d_kappa_local",
            static_cast<std::size_t>(n_cells) * sizeof(double)));
      }
      double* d_kappa_pm = nullptr;
      d_kappa_pm = static_cast<double*>(core::device_scratch_acquire(
          "conduction:compute_conduction_diagnostics_impl:d_kappa_pm_kappa",
          static_cast<std::size_t>(n_cells) *
              static_cast<std::size_t>(n_mat) * sizeof(double)));
      diag = compute_per_material_2d_operator(
          state, cfg, eos_ctx, d_kappa_local, d_kappa_pm, d_rho_cv_local);
    }
    return diag;
  }

  double* d_A_eff = nullptr;
  double* d_gamma_eff = nullptr;
  std::uint8_t* d_cell_is_void = nullptr;
  if (state.A_eff.size() == static_cast<std::size_t>(n_cells) &&
      state.gamma_eff.size() == static_cast<std::size_t>(n_cells)) {
    d_A_eff = const_cast<double*>(state.A_eff.data());
    d_gamma_eff = const_cast<double*>(state.gamma_eff.data());
  } else {
    std::vector<double> A_eff;
    std::vector<double> gamma_eff;
    compute_effective_material_properties(cfg, state, n_cells, A_eff, gamma_eff);
    TENRYU_ASSERT(A_eff.size() == static_cast<std::size_t>(n_cells),
                  "Conduction A_eff size mismatch");
    TENRYU_ASSERT(gamma_eff.size() == static_cast<std::size_t>(n_cells),
                  "Conduction gamma_eff size mismatch");
    d_A_eff = static_cast<double*>(core::device_scratch_acquire(
        "conduction:compute_conduction_diagnostics_impl:d_A_eff",
        static_cast<std::size_t>(n_cells) * sizeof(double)));
    d_gamma_eff = static_cast<double*>(core::device_scratch_acquire(
        "conduction:compute_conduction_diagnostics_impl:d_gamma_eff",
        static_cast<std::size_t>(n_cells) * sizeof(double)));
    cuda_check(cudaMemcpy(d_A_eff,
                          A_eff.data(),
                          static_cast<std::size_t>(n_cells) * sizeof(double),
                          cudaMemcpyHostToDevice),
               "Conduction: cudaMemcpy d_A_eff failed");
    cuda_check(cudaMemcpy(d_gamma_eff,
                          gamma_eff.data(),
                          static_cast<std::size_t>(n_cells) * sizeof(double),
                          cudaMemcpyHostToDevice),
               "Conduction: cudaMemcpy d_gamma_eff failed");
  }
  d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "conduction:compute_conduction_diagnostics_impl:d_cell_is_void",
      static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t)));
  cuda_check(cudaMemcpy(d_cell_is_void,
                        state.cell_is_void.data(),
                        static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             "Conduction: cudaMemcpy d_cell_is_void failed");

  double* d_diag3 = static_cast<double*>(core::device_scratch_acquire(
      "conduction:compute_conduction_diagnostics_impl:diag3_pack", 3 * sizeof(double)));
  double* d_min_ratio = d_diag3 + 0;
  double* d_min_deff = d_diag3 + 1;
  double* d_max_deff = d_diag3 + 2;

  const double inf = std::numeric_limits<double>::infinity();
  const double h_diag3_init[3] = {inf, inf, 0.0};
  cuda_check(cudaMemcpy(d_diag3, h_diag3_init, sizeof(h_diag3_init),
                        cudaMemcpyHostToDevice),
             "Conduction: cudaMemcpy init diagnostics diag3 pack failed");

  const core::State::LaunchWindow cw = state.owned_cell_window_ghost(n_cells, 1);  // alpha/deff: ghost-extended (interface faces; min/max atomics idempotent)
  const double* state_cv_e = state.cv_e.empty() ? nullptr : state.cv_e.data();
  // W-G2 nlheat verify hook (default-inert): TENRYU_CONDUCTION_TEST_KAPPA_POWER=p
  // switches the 1D test_kappa constant to kappa = test_kappa * Te^p via a
  // dedicated kernel. The historic kernel and its launch below stay byte-
  // untouched when the env var is unset; env is read per call (FLD hook
  // precedent) so verify gates can toggle it in-process. 1D single-material
  // path only (2D and per-material paths ignore the hook).
  const char* kappa_power_env = std::getenv("TENRYU_CONDUCTION_TEST_KAPPA_POWER");
  if (state.mesh.dim == 1 && kappa_power_env != nullptr &&
      cfg.numerics.conduction.test_kappa > 0.0) {
    const double kappa_power = std::atof(kappa_power_env);
    const char* kappa_rho_power_env =
        std::getenv("TENRYU_CONDUCTION_TEST_KAPPA_RHO_POWER");
    const double kappa_rho_power =
        (kappa_rho_power_env != nullptr) ? std::atof(kappa_rho_power_env) : 0.0;
    compute_powerlaw_test_kappa_deff_1d_kernel<<<cw.blocks(), kBlockSize>>>(
        d_diffusion,
        d_rho_cv_e,
        d_min_ratio,
        d_min_deff,
        d_max_deff,
        state.Te.data(),
        state.rho.data(),
        state.zbar.data(),
        state.x_r.data(),
        cw.begin,
        cw.end,
        n_cells,
        d_gamma_eff,
        d_A_eff,
        d_cell_is_void,
        cfg.numerics.conduction.test_kappa,
        kappa_power,
        kappa_rho_power,
        state_cv_e);
  } else if (state.mesh.dim == 1) {
    const double* zmom_r2 =
        state.zmom_active ? state.zmom_r2.data() : nullptr;
    if (zmom_r2 != nullptr) {
      compute_spitzer_deff_1d_kernel<true><<<cw.blocks(), kBlockSize>>>(
          d_diffusion,
          d_rho_cv_e,
          d_min_ratio,
          d_min_deff,
          d_max_deff,
          state.Te.data(),
          state.rho.data(),
          state.zbar.data(),
          state.vol.data(),
          state.x_r.data(),
          cw.begin,
          cw.end,
          n_cells,
          d_gamma_eff,
          d_A_eff,
          d_cell_is_void,
          cfg.numerics.conduction.f_lim,
          cfg.numerics.conduction.test_kappa,
          cfg.numerics.conduction.mfp_limiter_C,
          state_cv_e,
          zmom_r2);
    } else {
      compute_spitzer_deff_1d_kernel<false><<<cw.blocks(), kBlockSize>>>(
          d_diffusion,
          d_rho_cv_e,
          d_min_ratio,
          d_min_deff,
          d_max_deff,
          state.Te.data(),
          state.rho.data(),
          state.zbar.data(),
          state.vol.data(),
          state.x_r.data(),
          cw.begin,
          cw.end,
          n_cells,
          d_gamma_eff,
          d_A_eff,
          d_cell_is_void,
          cfg.numerics.conduction.f_lim,
          cfg.numerics.conduction.test_kappa,
          cfg.numerics.conduction.mfp_limiter_C,
          state_cv_e,
          nullptr);
    }
  } else {
    compute_spitzer_deff_2d_kernel<<<cw.blocks(), kBlockSize>>>(
        d_diffusion,
        d_rho_cv_e,
        d_min_ratio,
        d_min_deff,
        d_max_deff,
        state.Te.data(),
        state.rho.data(),
        state.zbar.data(),
        state.vol.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.mesh.topo.nr,
        state.mesh.topo.nz,
        d_gamma_eff,
        d_A_eff,
        d_cell_is_void,
        cfg.numerics.conduction.f_lim,
        cfg.numerics.conduction.test_kappa,
        cfg.numerics.conduction.mfp_limiter_C,
        state_cv_e,
        cw.begin,
        cw.end);
  }
  cuda_check(cudaGetLastError(), "Conduction: compute_spitzer_deff launch failed");
  cuda_check(core::debug_kernel_sync(), "Conduction: compute_spitzer_deff failed");

  // W-G2 kirchhoff face_kappa_policy (env bridge until the namelist lands;
  // unset = historic harmonic closure, byte-identical). Production Spitzer
  // path only: the constant test_kappa closure is analytically identical
  // under both policies, and the nlheat power hook keeps its own exact
  // secant kernel.
  const char* face_policy_env = std::getenv("TENRYU_CONDUCTION_FACE_KAPPA_POLICY");
  const std::string face_policy =
      (face_policy_env != nullptr) ? std::string(face_policy_env)
                                   : cfg.numerics.conduction.face_kappa_policy;
  const bool face_policy_kirchhoff_base = (face_policy == "kirchhoff_same_material");
  const bool face_policy_kirchhoff =
      (face_policy_kirchhoff_base &&
       cfg.numerics.conduction.test_kappa <= 0.0 && kappa_power_env == nullptr);
  if (face_policy_kirchhoff && state.mesh.dim == 1 && d_min_ratio != nullptr &&
      d_diffusion != nullptr) {
    const int geom_code_policy = cfg.numerics.conduction.test_planar
                                     ? 2
                                     : state.mesh.geometry_code;
    auto launch_dt_ratio = [&](auto geom_tag) {
      constexpr int GEOM = decltype(geom_tag)::value;
      kirchhoff_dt_ratio_1d_kernel<GEOM><<<cw.blocks(), kBlockSize>>>(
          d_min_ratio,
          d_diffusion,
          d_rho_cv_e,
          state.Te.data(),
          d_cell_is_void,
          state.vol.data(),
          state.x_r.data(),
          cw.begin,
          cw.end,
          n_cells);
    };
    switch (geom_code_policy) {
      case 1:
        launch_dt_ratio(std::integral_constant<int, 1>{});
        break;
      case 2:
        launch_dt_ratio(std::integral_constant<int, 2>{});
        break;
      default:
        launch_dt_ratio(std::integral_constant<int, 0>{});
        break;
    }
    cuda_check(cudaGetLastError(), "Conduction: kirchhoff dt ratio launch failed");
    cuda_check(core::debug_kernel_sync(), "Conduction: kirchhoff dt ratio failed");
  }

  double h_diag3[3] = {0.0, 0.0, 0.0};
  cuda_check(cudaMemcpy(h_diag3, d_diag3, sizeof(h_diag3), cudaMemcpyDeviceToHost),
             "Conduction: cudaMemcpy diagnostics diag3 pack failed");
  double min_ratio = h_diag3[0];
  double min_deff = h_diag3[1];
  double max_deff = h_diag3[2];

  if (std::isfinite(min_deff) && min_deff > 0.0) {
    diag.deff_min = min_deff;
  }
  if (std::isfinite(max_deff) && max_deff > 0.0) {
    diag.deff_max = max_deff;
  }

  if (std::isfinite(min_ratio) && min_ratio > 0.0) {
    diag.dt_exp = cfg.numerics.dt.cfl_cond * min_ratio;
    diag.dt_cond =
        compute_dt_cond_from_dt_exp(diag.dt_exp, cfg.numerics.conduction.sts_max_stages);
  }
  log_plain_1d_dtexp_debug(state, d_diffusion, d_rho_cv_e, diag.dt_exp);
  return diag;
}

int map_hydro_bc_to_kershaw(const std::string& bc) {
  if (bc == "vacuum") {
    // Vacuum for heat conduction implies zero conductive heat flux (Neumann),
    // equivalent to reflective treatment for diffusion: no material beyond
    // the boundary means no conductive heat transfer across it.
    return kershaw::BC_REFLECT;
  }
  if (bc == "reflective" || bc == "reflect" || bc == "free" || bc == "fixed" ||
      bc == "pressure" || bc == "axis" || bc == "state_supply") {
    return kershaw::BC_REFLECT;
  }
  static bool warned_unknown_bc = false;
  if (!warned_unknown_bc) {
    core::log_warning("Conduction Kershaw BC mapping received unknown Hydro BC '" + bc +
                      "'; defaulting to reflective (Neumann).");
    warned_unknown_bc = true;
  }
  return kershaw::BC_REFLECT;
}

void initialize_sts_taus(std::vector<double>& tau,
                         const double dt,
                         const double dt_exp,
                         const int stages,
                         const double damping) {
  tau.assign(static_cast<std::size_t>(stages), 0.0);
  const double nu = std::clamp(damping, 1.0e-6, 0.999);

  double tau_sum = 0.0;
  for (int j = 1; j <= stages; ++j) {
    const double arg =
        kPi * (2.0 * static_cast<double>(j) - 1.0) /
        (4.0 * static_cast<double>(stages) + 2.0);
    const double c = std::cos(arg);
    const double denom = nu * nu + (1.0 - nu * nu) * c * c;
    const double tj = dt_exp / denom;
    tau[static_cast<std::size_t>(j - 1)] = tj;
    tau_sum += tj;
  }
  TENRYU_ASSERT(tau_sum > 0.0, "Conduction STS tau_sum must be positive");
  const double scale = dt / tau_sum;
  for (double& tj : tau) {
    tj *= scale;
  }
}

// Fail-closed ladder stability audit (AI kernel review 2026-07-26, k07 F-08;
// NUMERICS 4.2.1).  The uniform tau rescale above is only certified stable
// when the composite amplification |prod (1 - lambda tau_j)| stays <= 1 over
// the spectral certificate lambda <= 4 * cfl_cond / dt_exp that dt_exp itself
// encodes.  Shipped defaults shrink the ladder (bound ~ 0.175) and never trip
// this; inflating configurations (larger sts_damping at high stage counts, or
// cfl_cond raised past 0.25) previously ran silently unstable.
void audit_sts_ladder_stability(const std::vector<double>& tau,
                                const double dt_sub,
                                const double dt_exp,
                                const double cfl_cond,
                                const double damping,
                                const char* path_tag) {
  if (tau.empty() || !(dt_exp > 0.0) || !std::isfinite(dt_exp) ||
      !(cfl_cond > 0.0)) {
    return;
  }
  const double lambda_max = 4.0 * cfl_cond / dt_exp;
  const double bound = conduction_bodies::sts_ladder_amplification_bound(
      tau.data(), static_cast<int>(tau.size()), lambda_max);
  if (bound > 1.0 + 1.0e-9) {
    core::log_error(
        std::string("Conduction STS ladder amplification audit failed (") +
        path_tag + "): max|G| = " + format_sci(bound) +
        " over lambda*dt_exp <= " + format_val(4.0 * cfl_cond) +
        " with stages = " + std::to_string(tau.size()) +
        ", sts_damping = " + format_val(damping) +
        ", dt_sub/dt_exp = " + format_sci(dt_sub / dt_exp) +
        ". The uniformly rescaled Chebyshev ladder is unstable for this "
        "configuration; keep Numerics.dt.cfl_cond <= 0.25 and/or lower "
        "conduction.sts_damping (default 0.01), or reduce dt.");
    TENRYU_ASSERT(false, "Conduction STS ladder amplification audit failed");
  }
}

void conduction_sync_eos(core::State& state, const core::Config& cfg) {
  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0) {
    return;
  }
  // BUG-24: in energy-authoritative mode with a table backend the driver's
  // post-conduction sync (single, tail-aware table surface) performs the
  // Te -> ee conversion; the internal linear cv*T projection would fight
  // it (measured +-16 kJ/subcycle representation swing, net leak).
  if (cfg.numerics.hydro.eos_closure_mode == "energy_authoritative") {
    const int first_nonvoid = cfg.materials.first_nonvoid_material_index();
    if (first_nonvoid >= 0 &&
        cfg.materials.materials[static_cast<std::size_t>(first_nonvoid)]
            .eos_tables) {
      return;
    }
  }
  std::vector<double> A_eff;
  std::vector<double> gamma_eff;
  compute_effective_material_properties(cfg, state, n_cells, A_eff, gamma_eff);
  TENRYU_ASSERT(A_eff.size() == static_cast<std::size_t>(n_cells),
                "Conduction eos_sync A_eff size mismatch");
  TENRYU_ASSERT(gamma_eff.size() == static_cast<std::size_t>(n_cells),
                "Conduction eos_sync gamma_eff size mismatch");

  double* d_A_eff = nullptr;
  double* d_gamma_eff = nullptr;
  d_A_eff = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_sync_eos:d_A_eff",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_gamma_eff = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_sync_eos:d_gamma_eff",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  cuda_check(cudaMemcpy(d_A_eff,
                        A_eff.data(),
                        static_cast<std::size_t>(n_cells) * sizeof(double),
                        cudaMemcpyHostToDevice),
             "Conduction: cudaMemcpy eos d_A_eff failed");
  cuda_check(cudaMemcpy(d_gamma_eff,
                        gamma_eff.data(),
                        static_cast<std::size_t>(n_cells) * sizeof(double),
                        cudaMemcpyHostToDevice),
             "Conduction: cudaMemcpy eos d_gamma_eff failed");

  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  eos_sync_electron_kernel<<<cw.blocks(), kBlockSize>>>(
      state.Te.data(),
      state.ee.data(),
      state.Pe.data(),
      state.rho.data(),
      state.zbar.data(),
      d_gamma_eff,
      d_A_eff,
      cw.begin,
      cw.end,
      n_cells,
      cfg.numerics.floors.Te,
      state.cv_e.empty() ? nullptr : state.cv_e.data());
  cuda_check(cudaGetLastError(), "Conduction: eos_sync launch failed");
  cuda_check(core::debug_kernel_sync(), "Conduction: eos_sync execution failed");
}

void accumulate_state_supply_conduction_bc(core::State& state,
                                           const core::Config& cfg,
                                           const double dt,
                                           const double* d_kappa_eff,
                                           const double* d_rho_cv_e) {
  state.c1_bc_heat_flux_last.fill(0.0);
  if (state.mesh.dim != 2 || !(dt > 0.0) || d_kappa_eff == nullptr ||
      d_rho_cv_e == nullptr) {
    return;
  }
  const auto& b = cfg.numerics.hydro.boundary_2d;
  const int z_bottom_source = b.z_bottom_cfg.is_state_supply() ? 1 : 0;
  const int z_top_source = b.z_top_cfg.is_state_supply() ? 1 : 0;
  if (z_bottom_source == 0 && z_top_source == 0) {
    return;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  double* d_boundary_energy = nullptr;
  d_boundary_energy = static_cast<double*>(core::device_scratch_acquire(
      "conduction:accumulate_state_supply_conduction_bc:d_boundary_energy",
      state.c1_bc_heat_flux_last.size() * sizeof(double)));
  cuda_check(cudaMemset(d_boundary_energy,
                        0,
                        state.c1_bc_heat_flux_last.size() * sizeof(double)),
             "Conduction: zero boundary heat flux tally failed");
  const int blocks = (nr + kBlockSize - 1) / kBlockSize;
  apply_state_supply_conduction_bc_kernel<<<blocks, kBlockSize>>>(
      state.Te.data(),
      d_kappa_eff,
      d_rho_cv_e,
      state.vol.data(),
      state.x_r.data(),
      state.x_z.data(),
      d_boundary_energy,
      nr,
      nz,
      dt,
      z_bottom_source,
      b.z_bottom_cfg.supply_T_eV,
      z_top_source,
      b.z_top_cfg.supply_T_eV);
  cuda_check(cudaGetLastError(), "Conduction: boundary heat flux source launch failed");
  cuda_check(core::debug_kernel_sync(), "Conduction: boundary heat flux source failed");
  cuda_check(cudaMemcpy(state.c1_bc_heat_flux_last.data(),
                        d_boundary_energy,
                        state.c1_bc_heat_flux_last.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Conduction: copy boundary heat flux tally failed");
  for (std::size_t bidx = 0; bidx < state.c1_bc_heat_flux_integrated.size(); ++bidx) {
    state.c1_bc_heat_flux_integrated[bidx] += state.c1_bc_heat_flux_last[bidx];
  }
}

struct CondCusparseCache {
  cusparseHandle_t handle = nullptr;
  ~CondCusparseCache() {
    if (handle != nullptr) {
      cusparseDestroy(handle);
    }
  }
};

CondCusparseCache& cond_cusparse_cache() {
  static CondCusparseCache cache;
  return cache;
}

ConductionResult conduction_step_1d_implicit(core::State& state,
                                             const double dt,
                                             const core::Config& cfg,
                                             const parallel::PartitionInfo& part,
                                             parallel::CommBuffers* bufs,
                                             cudaStream_t stream) {
  (void)bufs;
  ConductionResult result;
  result.dt_cond = std::numeric_limits<double>::infinity();
  result.dt_exp = std::numeric_limits<double>::infinity();

  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0) {
    return result;
  }
  TENRYU_ASSERT(conduction_1d_implicit_supported(state, &part),
                "Conduction implicit 1D solver requires serial 1D_SPH execution");

  double* d_kappa_eff = nullptr;
  double* d_rho_cv_e = nullptr;
  std::uint8_t* d_cell_is_void = nullptr;
  double* d_A_eff = nullptr;
  double* d_flux_limiter_faces = nullptr;
  double* d_lower = nullptr;
  double* d_diag = nullptr;
  double* d_upper = nullptr;
  double* d_rhs = nullptr;
  int* d_clamp_count = nullptr;
  double* d_e_floor = nullptr;
  void* d_cusparse_buffer = nullptr;
  cusparseHandle_t cusparse_handle = nullptr;

  auto cleanup = [&]() {};

  d_kappa_eff = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_1d_implicit:d_kappa_eff",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_rho_cv_e = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_1d_implicit:d_rho_cv_e",
      static_cast<std::size_t>(n_cells) * sizeof(double)));

  const auto diag =
      compute_conduction_diagnostics_impl(state, cfg, d_kappa_eff, d_rho_cv_e, nullptr);
  result.dt_exp = diag.dt_exp;
  result.dt_cond = std::numeric_limits<double>::infinity();
  result.deff_min = diag.deff_min;
  result.deff_max = diag.deff_max;

  if (!std::isfinite(diag.dt_exp) || !(diag.dt_exp > 0.0)) {
    cleanup();
    return result;
  }

  d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "conduction:conduction_step_1d_implicit:d_cell_is_void",
      static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t)));
  cuda_check(cudaMemcpy(d_cell_is_void,
                        state.cell_is_void.data(),
                        static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             "Conduction: cudaMemcpy implicit d_cell_is_void failed");

  d_lower = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_1d_implicit:d_lower",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_diag = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_1d_implicit:d_diag",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_upper = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_1d_implicit:d_upper",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_rhs = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_1d_implicit:d_rhs",
      static_cast<std::size_t>(n_cells) * sizeof(double)));

  double* d_clampfloor = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_1d_implicit:clampfloor_pack", 2 * sizeof(double)));
  d_e_floor = d_clampfloor + 0;
  d_clamp_count = reinterpret_cast<int*>(d_clampfloor + 1);

  cuda_check(cudaMemsetAsync(d_clampfloor, 0, 2 * sizeof(double), stream),
             "Conduction: init clampfloor pack (implicit) failed");

  const int use_flux_limiter = (cfg.numerics.conduction.test_kappa > 0.0) ? 0 : 1;
  if (use_flux_limiter != 0 && n_cells > 1) {
    if (state.A_eff.size() == static_cast<std::size_t>(n_cells)) {
      d_A_eff = const_cast<double*>(state.A_eff.data());
    } else {
      std::vector<double> A_eff;
      std::vector<double> gamma_eff;
      compute_effective_material_properties(cfg, state, n_cells, A_eff, gamma_eff);
      (void)gamma_eff;
      TENRYU_ASSERT(A_eff.size() == static_cast<std::size_t>(n_cells),
                    "Conduction 1D implicit A_eff size mismatch");
      d_A_eff = static_cast<double*>(core::device_scratch_acquire(
          "conduction:conduction_step_1d_implicit:d_A_eff",
          static_cast<std::size_t>(n_cells) * sizeof(double)));
      cuda_check(cudaMemcpy(d_A_eff, A_eff.data(),
                            static_cast<std::size_t>(n_cells) * sizeof(double),
                            cudaMemcpyHostToDevice),
                 "Conduction: cudaMemcpy implicit d_A_eff failed");
    }
    d_flux_limiter_faces = static_cast<double*>(core::device_scratch_acquire(
        "conduction:conduction_step_1d_implicit:d_flux_limiter_faces",
        static_cast<std::size_t>(n_cells) * sizeof(double)));
    // BUG-19: the limiter estimate follows the face_kappa_policy of the
    // consuming build kernel (guards mirror the build dispatch below;
    // test-kappa paths never dispatch Kirchhoff).
    const char* limiter_policy_env =
        std::getenv("TENRYU_CONDUCTION_FACE_KAPPA_POLICY");
    const std::string limiter_policy =
        (limiter_policy_env != nullptr)
            ? std::string(limiter_policy_env)
            : cfg.numerics.conduction.face_kappa_policy;
    const bool limiter_face_kirchhoff =
        (limiter_policy == "kirchhoff_same_material" &&
         cfg.numerics.conduction.test_kappa <= 0.0);
    const int face_blocks = (n_cells - 1 + kBlockSize - 1) / kBlockSize;
    compute_1d_flux_limiter_faces_kernel<<<face_blocks, kBlockSize>>>(
        d_flux_limiter_faces,
        state.Te.data(),
        d_kappa_eff,
        state.rho.data(),
        state.zbar.data(),
        d_A_eff,
        state.x_r.data(),
        n_cells,
        cfg.numerics.conduction.f_lim,
        limiter_face_kirchhoff);
    cuda_check(cudaGetLastError(), "Conduction: implicit flux limiter launch failed");
    cuda_check(core::debug_kernel_sync(), "Conduction: implicit flux limiter failed");
  }

  if (n_cells == 1) {
    cuda_check(cudaMemcpy(d_rhs,
                          state.Te.data(),
                          sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Conduction: implicit single-cell Te copy failed");
    result.solver_residual = 0.0;
    result.solver_iterations = 1;
    result.solver_cond_number_est = 1.0;
  } else {
    const int blocks = (n_cells + kBlockSize - 1) / kBlockSize;
    // W-G2: test_planar is the historic planar alias (verify-only, spherical
    // meshes); otherwise Mesh.geometry_1d drives via geometry_code.
    const int geom_code = cfg.numerics.conduction.test_planar
                              ? 2
                              : state.mesh.geometry_code;
    // W-G2 kirchhoff face_kappa_policy (env bridge; unset = historic harmonic,
    // byte-identical). Production Spitzer path only.
    const char* face_policy_env = std::getenv("TENRYU_CONDUCTION_FACE_KAPPA_POLICY");
    const std::string face_policy =
        (face_policy_env != nullptr) ? std::string(face_policy_env)
                                     : cfg.numerics.conduction.face_kappa_policy;
    const bool face_policy_kirchhoff_base = (face_policy == "kirchhoff_same_material");
    const bool build_face_policy_kirchhoff =
        (face_policy_kirchhoff_base &&
         cfg.numerics.conduction.test_kappa <= 0.0);
    auto launch_build = [&](auto geom_tag) {
      constexpr int GEOM = decltype(geom_tag)::value;
      build_1d_implicit_system_kernel<GEOM><<<blocks, kBlockSize>>>(
          d_lower,
          d_diag,
          d_upper,
          d_rhs,
          state.Te.data(),
          d_kappa_eff,
          d_flux_limiter_faces,
          d_rho_cv_e,
          d_cell_is_void,
          state.vol.data(),
          state.x_r.data(),
          n_cells,
          dt);
    };
    auto launch_build_kirchhoff = [&](auto geom_tag) {
      constexpr int GEOM = decltype(geom_tag)::value;
      build_1d_implicit_system_kirchhoff_kernel<GEOM><<<blocks, kBlockSize>>>(
          d_lower,
          d_diag,
          d_upper,
          d_rhs,
          state.Te.data(),
          d_kappa_eff,
          d_flux_limiter_faces,
          d_rho_cv_e,
          d_cell_is_void,
          state.vol.data(),
          state.x_r.data(),
          n_cells,
          dt);
    };
    if (build_face_policy_kirchhoff) {
      switch (geom_code) {
        case 1:
          launch_build_kirchhoff(std::integral_constant<int, 1>{});
          break;
        case 2:
          launch_build_kirchhoff(std::integral_constant<int, 2>{});
          break;
        default:
          launch_build_kirchhoff(std::integral_constant<int, 0>{});
          break;
      }
    } else {
      switch (geom_code) {
        case 1:
          launch_build(std::integral_constant<int, 1>{});
          break;
        case 2:
          launch_build(std::integral_constant<int, 2>{});
          break;
        default:
          launch_build(std::integral_constant<int, 0>{});
          break;
      }
    }
    cuda_check(cudaGetLastError(), "Conduction: implicit system assembly launch failed");
    cuda_check(core::debug_kernel_sync(), "Conduction: implicit system assembly failed");

    if (n_cells == 2) {
      std::array<double, 2> h_lower = {};
      std::array<double, 2> h_diag = {};
      std::array<double, 2> h_upper = {};
      std::array<double, 2> h_rhs = {};
      cuda_check(cudaMemcpy(h_lower.data(),
                            d_lower,
                            2 * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "Conduction: implicit 2x2 lower copy failed");
      cuda_check(cudaMemcpy(h_diag.data(),
                            d_diag,
                            2 * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "Conduction: implicit 2x2 diag copy failed");
      cuda_check(cudaMemcpy(h_upper.data(),
                            d_upper,
                            2 * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "Conduction: implicit 2x2 upper copy failed");
      cuda_check(cudaMemcpy(h_rhs.data(), d_rhs, 2 * sizeof(double), cudaMemcpyDeviceToHost),
                 "Conduction: implicit 2x2 rhs copy failed");

      const double det = h_diag[0] * h_diag[1] - h_upper[0] * h_lower[1];
      TENRYU_ASSERT(std::isfinite(det) && std::fabs(det) > kDivEpsilon,
                    "Conduction: implicit 2x2 system is singular");

      std::array<double, 2> h_sol = {};
      h_sol[0] = (h_rhs[0] * h_diag[1] - h_upper[0] * h_rhs[1]) / det;
      h_sol[1] = (h_diag[0] * h_rhs[1] - h_lower[1] * h_rhs[0]) / det;
      TENRYU_ASSERT(std::isfinite(h_sol[0]) && std::isfinite(h_sol[1]),
                    "Conduction: implicit 2x2 solve produced non-finite solution");
      result.solver_residual = tridiagonal_relative_residual(
          std::vector<double>(h_lower.begin(), h_lower.end()),
          std::vector<double>(h_diag.begin(), h_diag.end()),
          std::vector<double>(h_upper.begin(), h_upper.end()),
          std::vector<double>(h_rhs.begin(), h_rhs.end()),
          std::vector<double>(h_sol.begin(), h_sol.end()));
      result.solver_iterations = 1;
      result.solver_cond_number_est =
          diagonal_ratio_condition_estimate(
              std::vector<double>(h_diag.begin(), h_diag.end()));
      cuda_check(cudaMemcpy(d_rhs, h_sol.data(), 2 * sizeof(double), cudaMemcpyHostToDevice),
                 "Conduction: implicit 2x2 solution copy failed");
    } else {
      std::vector<double> h_lower(static_cast<std::size_t>(n_cells));
      std::vector<double> h_diag(static_cast<std::size_t>(n_cells));
      std::vector<double> h_upper(static_cast<std::size_t>(n_cells));
      std::vector<double> h_rhs(static_cast<std::size_t>(n_cells));
      double* d_diag_stage = static_cast<double*>(core::device_scratch_acquire(
          "conduction:conduction_step_1d_implicit:diag_stage",
          static_cast<std::size_t>(n_cells) * 5 * sizeof(double)));
      const std::size_t nb = static_cast<std::size_t>(n_cells) * sizeof(double);
      cuda_check(cudaMemcpyAsync(d_diag_stage + 0 * n_cells, d_lower, nb,
                                 cudaMemcpyDeviceToDevice, stream),
                 "Conduction: implicit lower stage copy failed");
      cuda_check(cudaMemcpyAsync(d_diag_stage + 1 * n_cells, d_diag, nb,
                                 cudaMemcpyDeviceToDevice, stream),
                 "Conduction: implicit diag stage copy failed");
      cuda_check(cudaMemcpyAsync(d_diag_stage + 2 * n_cells, d_upper, nb,
                                 cudaMemcpyDeviceToDevice, stream),
                 "Conduction: implicit upper stage copy failed");
      cuda_check(cudaMemcpyAsync(d_diag_stage + 3 * n_cells, d_rhs, nb,
                                 cudaMemcpyDeviceToDevice, stream),
                 "Conduction: implicit rhs stage copy failed");
      auto& cus_cache = cond_cusparse_cache();
      if (cus_cache.handle == nullptr) {
        cusparse_check(cusparseCreate(&cus_cache.handle),
                       "Conduction: cusparseCreate failed");
      }
      cusparse_handle = cus_cache.handle;
      cusparse_check(cusparseSetStream(cusparse_handle, stream),
                     "Conduction: cusparseSetStream failed");

      size_t buffer_size = 0;
      cusparse_check(cusparseDgtsv2_bufferSizeExt(cusparse_handle,
                                                  n_cells,
                                                  1,
                                                  d_lower,
                                                  d_diag,
                                                  d_upper,
                                                  d_rhs,
                                                  n_cells,
                                                  &buffer_size),
                     "Conduction: cusparseDgtsv2_bufferSizeExt failed");
      const std::size_t buffer_size_alloc = std::max<std::size_t>(buffer_size, 1);
      d_cusparse_buffer = core::device_scratch_acquire(
          "conduction:conduction_step_1d_implicit:gtsv2_buffer", buffer_size_alloc);
      // cuSPARSE expects the full tridiagonal storage with dl[0] = 0 and du[m-1] = 0.
      cusparse_check(cusparseDgtsv2(cusparse_handle,
                                    n_cells,
                                    1,
                                    d_lower,
                                    d_diag,
                                    d_upper,
                                    d_rhs,
                                    n_cells,
                                    d_cusparse_buffer),
                     "Conduction: cusparseDgtsv2 failed");
      cuda_check(core::debug_kernel_sync(), "Conduction: implicit tridiagonal solve failed");
      cuda_check(cudaMemcpyAsync(d_diag_stage + 4 * n_cells, d_rhs, nb,
                                 cudaMemcpyDeviceToDevice, stream),
                 "Conduction: implicit solution stage copy failed");
      std::vector<double> h_stage(static_cast<std::size_t>(n_cells) * 5);
      cuda_check(cudaMemcpy(h_stage.data(), d_diag_stage,
                            static_cast<std::size_t>(n_cells) * 5 * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "Conduction: implicit diagnostic stage copy failed");
      std::memcpy(h_lower.data(), h_stage.data() + 0 * n_cells, nb);
      std::memcpy(h_diag.data(),  h_stage.data() + 1 * n_cells, nb);
      std::memcpy(h_upper.data(), h_stage.data() + 2 * n_cells, nb);
      std::memcpy(h_rhs.data(),   h_stage.data() + 3 * n_cells, nb);
      std::vector<double> h_sol(static_cast<std::size_t>(n_cells));
      std::memcpy(h_sol.data(),   h_stage.data() + 4 * n_cells, nb);
      result.solver_residual =
          tridiagonal_relative_residual(h_lower, h_diag, h_upper, h_rhs, h_sol);
      result.solver_iterations = 1;
      result.solver_cond_number_est = diagonal_ratio_condition_estimate(h_diag);
    }
  }

  const int blocks = (n_cells + kBlockSize - 1) / kBlockSize;
  // W-G2: test_planar is the historic planar alias (verify-only, spherical
  // meshes); otherwise Mesh.geometry_1d drives via geometry_code.
  const int geom_code = cfg.numerics.conduction.test_planar
                            ? 2
                            : state.mesh.geometry_code;
  auto launch_clamp = [&](auto geom_tag) {
    constexpr int GEOM = decltype(geom_tag)::value;
    clamp_1d_conduction_solution_kernel<GEOM><<<blocks, kBlockSize>>>(
        d_rhs,
        d_rho_cv_e,
        d_cell_is_void,
        state.vol.data(),
        state.x_r.data(),
        n_cells,
        cfg.numerics.floors.Te,
        d_clamp_count,
        d_e_floor);
  };
  switch (geom_code) {
    case 1:
      launch_clamp(std::integral_constant<int, 1>{});
      break;
    case 2:
      launch_clamp(std::integral_constant<int, 2>{});
      break;
    default:
      launch_clamp(std::integral_constant<int, 0>{});
      break;
  }
  cuda_check(cudaGetLastError(), "Conduction: implicit floor clamp launch failed");
  cuda_check(core::debug_kernel_sync(), "Conduction: implicit floor clamp failed");

  cuda_check(cudaMemcpy(state.Te.data(),
                        d_rhs,
                        static_cast<std::size_t>(n_cells) * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Conduction: implicit Te copy-back failed");

  conduction_sync_eos(state, cfg);

  double h_clampfloor[2] = {0.0, 0.0};
  cuda_check(cudaMemcpy(h_clampfloor, d_clampfloor, sizeof(h_clampfloor),
                        cudaMemcpyDeviceToHost),
             "Conduction: copy clampfloor pack (implicit) failed");
  result.E_floor_injected = h_clampfloor[0];
  std::memcpy(&result.clamp_count, &h_clampfloor[1], sizeof(int));
  if (part.n_ranks > 1 && part.rank != 0) {
    // Replicated solve: every rank computed the identical full-line tally;
    // only rank 0 reports it (same convention as replicated_tally_share).
    result.E_floor_injected = 0.0;
    result.clamp_count = 0;
  }

  if (result.clamp_count > cfg.numerics.safety.clamp_warn_threshold) {
    core::log_warning("Conduction floor clamp count exceeded warning threshold: " +
                      std::to_string(result.clamp_count));
  }

  cleanup();
  return result;
}

ConductionResult conduction_step_1d_sts_per_material(core::State& state,
                                                     const double dt,
                                                     const core::Config& cfg,
                                                     const HydroEOSContext* eos_ctx,
                                                     const parallel::PartitionInfo& part,
                                                     parallel::CommBuffers* bufs,
                                                     cudaStream_t stream) {
  ConductionResult result;
  result.dt_cond = std::numeric_limits<double>::infinity();
  result.dt_exp = std::numeric_limits<double>::infinity();
  const int floor_limiter_mode =
      (cfg.numerics.conduction.sts_floor_limiter == "donor") ? 1 : 0;

  const int n_cells = static_cast<int>(state.rho.size());
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  if (n_cells <= 0) {
    return result;
  }
  const bool do_exchange = (part.n_ranks > 1 && bufs != nullptr);
  auto exchange_te_halo = [&](double* te_ptr) {
    if (!do_exchange) {
      return;
    }
    double* Te_ptr = te_ptr;
    parallel::exchange_cell_fields(part, *bufs, &Te_ptr, 1, n_cells, stream, 3);
  };
  exchange_te_halo(state.Te.data());

  const int n_faces = std::max(0, n_cells - 1);
  double* d_kappa_face = nullptr;
  double* d_kappa_eff_per_material = nullptr;
  double* d_rho_cv_e = nullptr;
  d_kappa_face = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_1d_sts_per_material:d_kappa_face",
      static_cast<std::size_t>(std::max(1, n_faces)) * sizeof(double)));
  d_kappa_eff_per_material = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_1d_sts_per_material:d_kappa_eff_per_material",
      static_cast<std::size_t>(std::max(1, n_faces * n_mat)) *
          sizeof(double)));
  d_rho_cv_e = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_1d_sts_per_material:d_rho_cv_e",
      static_cast<std::size_t>(n_cells) * sizeof(double)));

  const auto diag = compute_per_material_1d_operator(
      state, cfg, eos_ctx, d_kappa_face, d_kappa_eff_per_material, d_rho_cv_e);
  double global_dt_exp = diag.dt_exp;
  if (part.n_ranks > 1) {
    parallel::Reduction red(part.n_ranks);
    global_dt_exp = (std::isfinite(global_dt_exp) && global_dt_exp > 0.0)
                        ? red.allreduce_min(global_dt_exp)
                        : red.allreduce_min(1.0e+30);
  }
  result.dt_exp = global_dt_exp;
  result.dt_cond =
      compute_dt_cond_from_dt_exp(global_dt_exp, cfg.numerics.conduction.sts_max_stages);
  result.deff_min = diag.deff_min;
  result.deff_max = diag.deff_max;

  if (!std::isfinite(global_dt_exp) || !(global_dt_exp > 0.0) ||
      (part.n_ranks > 1 && !(global_dt_exp < 1.0e+30))) {
    return result;
  }

  const int smax = (cfg.numerics.conduction.sts_max_stages > 0)
                       ? std::max(1, cfg.numerics.conduction.sts_max_stages)
                       : 100000;
  int n_sub = 1;
  double dt_sub = dt;
  if (cfg.numerics.conduction.sts_max_stages > 0) {
    const double eta = std::clamp(cfg.numerics.conduction.sts_subcycle_eta, 1.0e-6, 1.0);
    const double smax_factor =
        0.5 * static_cast<double>(smax) * static_cast<double>(smax + 1);
    const double dt_sts_max = smax_factor * global_dt_exp * eta;
    if (std::isfinite(dt_sts_max) && dt_sts_max > 0.0 && dt > dt_sts_max) {
      n_sub = ceil_to_int_clamped(dt / dt_sts_max);
      dt_sub = dt / static_cast<double>(n_sub);
    }
  }
  const int stages_per_sub = conduction::sts_stage_count(dt_sub, global_dt_exp, smax);
  const long long total_stages =
      static_cast<long long>(stages_per_sub) * static_cast<long long>(n_sub);
  const long long sts_total_cap =
      static_cast<long long>(cfg.numerics.conduction.sts_total_stages_max);
  if (sts_total_cap > 0 && total_stages > sts_total_cap) {
    core::log_warning(
        "[conduction] STS total stage bound tripped: total=" +
        std::to_string(total_stages) + " (n_sub=" + std::to_string(n_sub) +
        " x stages=" + std::to_string(stages_per_sub) +
        ") > sts_total_stages_max=" + std::to_string(sts_total_cap) +
        " dt=" + format_sci(dt) + " dt_exp=" + format_sci(global_dt_exp) +
        " — requesting driver full-step retry");
    result.retry_required = true;
    result.retry_reason = "sts_total_stages_overflow";
    result.retry_total_stages = total_stages;
    result.retry_n_sub = n_sub;
    result.retry_dt_exp = global_dt_exp;
    return result;
  }
  result.sts_subcycles = n_sub;
  result.sts_stages = (total_stages > static_cast<long long>(std::numeric_limits<int>::max()))
                          ? std::numeric_limits<int>::max()
                          : static_cast<int>(total_stages);
  result.solver_residual = 0.0;
  result.solver_iterations = result.sts_stages;
  result.solver_cond_number_est = 1.0;

  std::vector<double> tau;
  initialize_sts_taus(
      tau, dt_sub, global_dt_exp, stages_per_sub, cfg.numerics.conduction.sts_damping);
  audit_sts_ladder_stability(tau, dt_sub, global_dt_exp,
                             cfg.numerics.dt.cfl_cond,
                             cfg.numerics.conduction.sts_damping,
                             "1d_sts_per_material");

  double* d_te_tmp = nullptr;
  double* d_alpha_cells = nullptr;
  std::uint8_t* d_cell_is_void = nullptr;
  int* d_clamp_count = nullptr;
  double* d_e_floor = nullptr;
  d_te_tmp = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_1d_sts_per_material:d_te_tmp",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_alpha_cells = static_cast<double*>(core::device_scratch_acquire(
      "conduction:alpha_pass:1d_sts_per_material",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "conduction:conduction_step_1d_sts_per_material:d_cell_is_void",
      static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t)));
  cuda_check(cudaMemcpy(d_cell_is_void,
                        state.cell_is_void.data(),
                        static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             "Conduction: cudaMemcpy per-material d_cell_is_void failed");
  double* d_clampfloor = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_1d_sts_per_material:clampfloor_pack",
      2 * sizeof(double)));
  d_e_floor = d_clampfloor + 0;
  d_clamp_count = reinterpret_cast<int*>(d_clampfloor + 1);

  const double h_clampfloor_init[2] = {0.0, 0.0};
  cuda_check(cudaMemcpy(d_clampfloor, h_clampfloor_init, sizeof(h_clampfloor_init),
                        cudaMemcpyHostToDevice),
             "Conduction: init clampfloor pack (per-material) failed");

  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  // Ghost-extended window for alpha/diffusivity passes: interface-face
  // coefficients need the +-1 ghost cells; their min/max atomics are
  // idempotent under the duplicated ghost contributions.
  const core::State::LaunchWindow dwa = state.owned_cell_window_ghost(n_cells, 1);
  double* te_curr = state.Te.data();
  double* te_next = d_te_tmp;
  // W-G2: test_planar is the historic planar alias (verify-only, spherical
  // meshes); otherwise Mesh.geometry_1d drives via geometry_code.
  const int geom_code = cfg.numerics.conduction.test_planar
                            ? 2
                            : state.mesh.geometry_code;
  auto launch_stage = [&](auto geom_tag, const double tau_stage) {
    constexpr int GEOM = decltype(geom_tag)::value;
    conduction_alpha_pass_kernel_per_material<GEOM><<<dwa.blocks(), kBlockSize>>>(
        d_alpha_cells,
        te_curr,
        d_kappa_eff_per_material,
        state.volFrac.data(),
        d_rho_cv_e,
        d_cell_is_void,
        state.vol.data(),
        state.x_r.data(),
        dwa.begin,
        dwa.end,
        n_cells,
        n_mat,
        tau_stage,
        cfg.numerics.floors.Te,
        floor_limiter_mode);
    conduction_1d_sts_stage_kernel_per_material<GEOM><<<cw.blocks(), kBlockSize>>>(
        te_curr,
        te_next,
        d_kappa_face,
        d_kappa_eff_per_material,
        state.Ee_per_material.data(),
        state.volFrac.data(),
        d_rho_cv_e,
        d_cell_is_void,
        state.vol.data(),
        state.x_r.data(),
        cw.begin,
        cw.end,
        n_cells,
        n_mat,
        tau_stage,
        cfg.numerics.floors.Te,
        d_alpha_cells,
        d_clamp_count,
        d_e_floor,
        floor_limiter_mode);
  };
  for (int sub = 0; sub < n_sub; ++sub) {
    for (const double tau_j : tau) {
      exchange_te_halo(te_curr);
      switch (geom_code) {
        case 1:
          launch_stage(std::integral_constant<int, 1>{}, tau_j);
          break;
        case 2:
          launch_stage(std::integral_constant<int, 2>{}, tau_j);
          break;
        default:
          launch_stage(std::integral_constant<int, 0>{}, tau_j);
          break;
      }
      cuda_check(cudaGetLastError(), "Conduction: per-material STS stage launch failed");
      cuda_check(core::debug_kernel_sync(), "Conduction: per-material STS stage failed");
      std::swap(te_curr, te_next);
    }
  }
  if (te_curr != state.Te.data()) {
    cuda_check(cudaMemcpy(state.Te.data(),
                          te_curr,
                          static_cast<std::size_t>(n_cells) * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Conduction: per-material copy staged Te back failed");
  }

  derive_cell_ee_from_per_material_kernel<<<cw.blocks(), kBlockSize>>>(
      state.ee.data(), state.Ee_per_material.data(), state.mass.data(),
      cw.begin, cw.end, n_cells, n_mat);
  cuda_check(cudaGetLastError(), "Conduction: per-material ee reduction launch failed");
  cuda_check(core::debug_kernel_sync(), "Conduction: per-material ee reduction failed");
  per_material::refresh_per_material_derived_cell_fields(state, cfg, eos_ctx, true);

  if (part.n_ranks > 1 && bufs != nullptr) {
    cudaStream_t halo_stream = stream;
    double* te_ptr[] = {state.Te.data()};
    parallel::exchange_cell_fields(
        part, *bufs, te_ptr, 1, state.mesh.topo.n_cells, halo_stream, 4);
  }

  double h_clampfloor[2] = {0.0, 0.0};
  cuda_check(cudaMemcpy(h_clampfloor, d_clampfloor, sizeof(h_clampfloor),
                        cudaMemcpyDeviceToHost),
             "Conduction: copy clampfloor pack (per-material) failed");
  result.E_floor_injected = h_clampfloor[0];
  std::memcpy(&result.clamp_count, &h_clampfloor[1], sizeof(int));
  if (result.clamp_count > cfg.numerics.safety.clamp_warn_threshold) {
    core::log_warning("Conduction floor clamp count exceeded warning threshold: " +
                      std::to_string(result.clamp_count));
  }

  return result;
}

ConductionResult conduction_step_2d_sts_per_material(core::State& state,
                                                     const double dt,
                                                     const core::Config& cfg,
                                                     const HydroEOSContext* eos_ctx,
                                                     const parallel::PartitionInfo& part,
                                                     parallel::CommBuffers* bufs,
                                                     cudaStream_t stream) {
  ConductionResult result;
  result.dt_cond = std::numeric_limits<double>::infinity();
  result.dt_exp = std::numeric_limits<double>::infinity();
  const int floor_limiter_mode =
      (cfg.numerics.conduction.sts_floor_limiter == "donor") ? 1 : 0;

  const int n_cells = static_cast<int>(state.rho.size());
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  if (n_cells <= 0) {
    return result;
  }
  const bool do_exchange = (part.n_ranks > 1 && bufs != nullptr);
  auto exchange_te_halo = [&](double* te_ptr) {
    if (!do_exchange) {
      return;
    }
    double* Te_ptr = te_ptr;
    parallel::exchange_cell_fields(part, *bufs, &Te_ptr, 1, n_cells, stream, 3);
  };
  exchange_te_halo(state.Te.data());

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  TENRYU_ASSERT(nr * nz == n_cells, "Conduction 2D requires nr*nz == n_cells");

  double* d_kappa_eff = nullptr;
  double* d_kappa_eff_per_material = nullptr;
  double* d_rho_cv_e = nullptr;
  double* d_stencil = nullptr;
  double* d_stencil_per_material = nullptr;
  d_kappa_eff = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_2d_sts_per_material:d_kappa_eff",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_kappa_eff_per_material = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_2d_sts_per_material:d_kappa_eff_per_material",
      static_cast<std::size_t>(n_cells) *
          static_cast<std::size_t>(n_mat) * sizeof(double)));
  d_rho_cv_e = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_2d_sts_per_material:d_rho_cv_e",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_stencil = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_2d_sts_per_material:d_stencil",
      static_cast<std::size_t>(n_cells) * kershaw::kStencilSize *
          sizeof(double)));
  d_stencil_per_material = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_2d_sts_per_material:d_stencil_per_material",
      static_cast<std::size_t>(n_cells) *
          static_cast<std::size_t>(n_mat) * kershaw::kStencilSize *
          sizeof(double)));

  const auto diag = compute_per_material_2d_operator(
      state, cfg, eos_ctx, d_kappa_eff, d_kappa_eff_per_material, d_rho_cv_e);
  double global_dt_exp = diag.dt_exp;
  if (part.n_ranks > 1) {
    parallel::Reduction red(part.n_ranks);
    global_dt_exp = (std::isfinite(global_dt_exp) && global_dt_exp > 0.0)
                        ? red.allreduce_min(global_dt_exp)
                        : red.allreduce_min(1.0e+30);
  }
  result.dt_exp = global_dt_exp;
  result.dt_cond =
      compute_dt_cond_from_dt_exp(global_dt_exp, cfg.numerics.conduction.sts_max_stages);
  result.deff_min = diag.deff_min;
  result.deff_max = diag.deff_max;

  if (!std::isfinite(global_dt_exp) || !(global_dt_exp > 0.0) ||
      (part.n_ranks > 1 && !(global_dt_exp < 1.0e+30))) {
    return result;
  }

  const int smax = (cfg.numerics.conduction.sts_max_stages > 0)
                       ? std::max(1, cfg.numerics.conduction.sts_max_stages)
                       : 100000;
  int n_sub = 1;
  double dt_sub = dt;
  if (cfg.numerics.conduction.sts_max_stages > 0) {
    const double eta = std::clamp(cfg.numerics.conduction.sts_subcycle_eta, 1.0e-6, 1.0);
    const double smax_factor =
        0.5 * static_cast<double>(smax) * static_cast<double>(smax + 1);
    const double dt_sts_max = smax_factor * global_dt_exp * eta;
    if (std::isfinite(dt_sts_max) && dt_sts_max > 0.0 && dt > dt_sts_max) {
      n_sub = ceil_to_int_clamped(dt / dt_sts_max);
      dt_sub = dt / static_cast<double>(n_sub);
    }
  }
  const int stages_per_sub = conduction::sts_stage_count(dt_sub, global_dt_exp, smax);
  const long long total_stages =
      static_cast<long long>(stages_per_sub) * static_cast<long long>(n_sub);
  const long long sts_total_cap =
      static_cast<long long>(cfg.numerics.conduction.sts_total_stages_max);
  if (sts_total_cap > 0 && total_stages > sts_total_cap) {
    core::log_warning(
        "[conduction] STS total stage bound tripped: total=" +
        std::to_string(total_stages) + " (n_sub=" + std::to_string(n_sub) +
        " x stages=" + std::to_string(stages_per_sub) +
        ") > sts_total_stages_max=" + std::to_string(sts_total_cap) +
        " dt=" + format_sci(dt) + " dt_exp=" + format_sci(global_dt_exp) +
        " — requesting driver full-step retry");
    result.retry_required = true;
    result.retry_reason = "sts_total_stages_overflow";
    result.retry_total_stages = total_stages;
    result.retry_n_sub = n_sub;
    result.retry_dt_exp = global_dt_exp;
    return result;
  }
  result.sts_subcycles = n_sub;
  result.sts_stages = (total_stages > static_cast<long long>(std::numeric_limits<int>::max()))
                          ? std::numeric_limits<int>::max()
                          : static_cast<int>(total_stages);
  result.solver_residual = 0.0;
  result.solver_iterations = result.sts_stages;
  result.solver_cond_number_est = 1.0;

  std::vector<double> tau;
  initialize_sts_taus(
      tau, dt_sub, global_dt_exp, stages_per_sub, cfg.numerics.conduction.sts_damping);

  double* d_te_tmp = nullptr;
  double* d_alpha_cells = nullptr;
  std::uint8_t* d_cell_is_void = nullptr;
  int* d_clamp_count = nullptr;
  double* d_e_floor = nullptr;
  int* d_mmatrix_fix_count = nullptr;
  d_te_tmp = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_2d_sts_per_material:d_te_tmp",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_alpha_cells = static_cast<double*>(core::device_scratch_acquire(
      "conduction:alpha_pass:2d_sts_per_material",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "conduction:conduction_step_2d_sts_per_material:d_cell_is_void",
      static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t)));
  cuda_check(cudaMemcpy(d_cell_is_void,
                        state.cell_is_void.data(),
                        static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             "Conduction: cudaMemcpy per-material 2D d_cell_is_void failed");
  d_clamp_count = static_cast<int*>(core::device_scratch_acquire(
      "conduction:conduction_step_2d_sts_per_material:d_clamp_count", sizeof(int)));
  d_e_floor = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_2d_sts_per_material:d_e_floor", sizeof(double)));
  d_mmatrix_fix_count = static_cast<int*>(core::device_scratch_acquire(
      "conduction:conduction_step_2d_sts_per_material:d_mmatrix_fix_count", sizeof(int)));
  const int zero_i = 0;
  const double zero_d = 0.0;
  cuda_check(cudaMemcpy(d_clamp_count, &zero_i, sizeof(int), cudaMemcpyHostToDevice),
             "Conduction: init per-material 2D d_clamp_count failed");
  cuda_check(cudaMemcpy(d_e_floor, &zero_d, sizeof(double), cudaMemcpyHostToDevice),
             "Conduction: init per-material 2D d_e_floor failed");
  cuda_check(cudaMemcpy(d_mmatrix_fix_count, &zero_i, sizeof(int), cudaMemcpyHostToDevice),
             "Conduction: init per-material 2D d_mmatrix_fix_count failed");

  const int bc_r_lo = map_hydro_bc_to_kershaw("axis");
  const int bc_r_hi = map_hydro_bc_to_kershaw(cfg.numerics.hydro.boundary_2d.r_outer);
  const int bc_z_lo = map_hydro_bc_to_kershaw(cfg.numerics.hydro.boundary_2d.z_bottom);
  const int bc_z_hi = map_hydro_bc_to_kershaw(cfg.numerics.hydro.boundary_2d.z_top);
  kershaw::launch_kershaw_stencil_build(d_stencil,
                                        d_kappa_eff,
                                        state.x_r.data(),
                                        state.x_z.data(),
                                        nr,
                                        nz,
                                        bc_r_lo,
                                        bc_r_hi,
                                        bc_z_lo,
                                        bc_z_hi,
                                        true,
                                        d_mmatrix_fix_count);
  cuda_check(cudaGetLastError(), "Conduction: per-material 2D aggregate stencil launch failed");
  cuda_check(core::debug_kernel_sync(), "Conduction: per-material 2D aggregate stencil failed");
  for (int m = 0; m < n_mat; ++m) {
    kershaw::launch_kershaw_stencil_build(
        d_stencil_per_material +
            static_cast<std::size_t>(m) * n_cells * kershaw::kStencilSize,
        d_kappa_eff_per_material + static_cast<std::size_t>(m) * n_cells,
        state.x_r.data(),
        state.x_z.data(),
        nr,
        nz,
        bc_r_lo,
        bc_r_hi,
        bc_z_lo,
        bc_z_hi,
        true,
        nullptr);
    cuda_check(cudaGetLastError(), "Conduction: per-material 2D material stencil launch failed");
  }
  cuda_check(core::debug_kernel_sync(), "Conduction: per-material 2D material stencils failed");
  cuda_check(cudaMemcpy(&result.mmatrix_fix_count,
                        d_mmatrix_fix_count,
                        sizeof(int),
                        cudaMemcpyDeviceToHost),
             "Conduction: copy per-material 2D mmatrix_fix_count failed");
  result.solver_cond_number_est =
      kershaw_diagonal_ratio_condition_estimate(state, d_stencil, d_rho_cv_e, dt, n_cells);

  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  // Ghost-extended window for alpha/diffusivity passes: interface-face
  // coefficients need the +-1 ghost cells; their min/max atomics are
  // idempotent under the duplicated ghost contributions.
  const core::State::LaunchWindow dwa = state.owned_cell_window_ghost(n_cells, 1);
  double* te_curr = state.Te.data();
  double* te_next = d_te_tmp;
  for (int sub = 0; sub < n_sub; ++sub) {
    for (const double tau_j : tau) {
      exchange_te_halo(te_curr);
      kershaw_alpha_pass_per_material_energy_kernel<<<cw.blocks(), kBlockSize>>>(
          d_alpha_cells,
          te_curr,
          d_stencil_per_material,
          d_rho_cv_e,
          d_cell_is_void,
          state.vol.data(),
          n_mat,
          tau_j,
          cfg.numerics.floors.Te,
          nr,
          nz,
          floor_limiter_mode,
          cw.begin,
          cw.end);
      kershaw_apply_per_material_energy_kernel<<<cw.blocks(), kBlockSize>>>(
          te_next,
          te_curr,
          d_stencil_per_material,
          d_rho_cv_e,
          d_cell_is_void,
          state.vol.data(),
          state.Ee_per_material.data(),
          n_mat,
          tau_j,
          cfg.numerics.floors.Te,
          nr,
          nz,
          d_alpha_cells,
          d_clamp_count,
          d_e_floor,
          floor_limiter_mode,
          cw.begin,
          cw.end);
      cuda_check(cudaGetLastError(), "Conduction: per-material 2D apply launch failed");
      cuda_check(core::debug_kernel_sync(), "Conduction: per-material 2D apply failed");
      std::swap(te_curr, te_next);
    }
  }
  if (te_curr != state.Te.data()) {
    cuda_check(cudaMemcpy(state.Te.data(),
                          te_curr,
                          static_cast<std::size_t>(n_cells) * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Conduction: per-material 2D copy staged Te back failed");
  }
  derive_cell_ee_from_per_material_kernel<<<cw.blocks(), kBlockSize>>>(
      state.ee.data(), state.Ee_per_material.data(), state.mass.data(),
      cw.begin, cw.end, n_cells, n_mat);
  cuda_check(cudaGetLastError(), "Conduction: per-material 2D ee reduction launch failed");
  cuda_check(core::debug_kernel_sync(), "Conduction: per-material 2D ee reduction failed");
  per_material::refresh_per_material_derived_cell_fields(state, cfg, eos_ctx, true);

  if (part.n_ranks > 1 && bufs != nullptr) {
    cudaStream_t halo_stream = stream;
    double* te_ptr[] = {state.Te.data()};
    parallel::exchange_cell_fields(
        part, *bufs, te_ptr, 1, state.mesh.topo.n_cells, halo_stream, 4);
  }
  cuda_check(cudaMemcpy(&result.clamp_count, d_clamp_count, sizeof(int), cudaMemcpyDeviceToHost),
             "Conduction: copy per-material 2D clamp_count failed");
  cuda_check(cudaMemcpy(&result.E_floor_injected,
                        d_e_floor,
                        sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Conduction: copy per-material 2D E_floor failed");

  return result;
}

ConductionResult conduction_step_1d_sts(core::State& state,
                                        const double dt,
                                        const core::Config& cfg,
                                        const parallel::PartitionInfo& part,
                                        parallel::CommBuffers* bufs,
                                        cudaStream_t stream,
                                        const HydroEOSContext* eos_ctx) {
  ConductionResult result;
  result.dt_cond = std::numeric_limits<double>::infinity();
  result.dt_exp = std::numeric_limits<double>::infinity();
  const int floor_limiter_mode =
      (cfg.numerics.conduction.sts_floor_limiter == "donor") ? 1 : 0;

  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0) {
    return result;
  }
  if (per_material_conduction_ready(state, cfg, eos_ctx)) {
    return conduction_step_1d_sts_per_material(state, dt, cfg, eos_ctx, part, bufs, stream);
  }
  const bool do_exchange = (part.n_ranks > 1 && bufs != nullptr);
  const std::string& halo_strategy = cfg.numerics.conduction.halo_strategy;
  if (halo_strategy == "adaptive") {
    // TODO(v1.1): implement adaptive conduction halo strategy.
  }
  auto exchange_te_halo = [&](double* te_ptr) {
    if (!do_exchange) {
      return;
    }
    double* Te_ptr = te_ptr;
    parallel::exchange_cell_fields(part, *bufs, &Te_ptr, 1, n_cells, stream, 3);
  };
  exchange_te_halo(state.Te.data());

  double* d_kappa_eff = nullptr;
  double* d_rho_cv_e = nullptr;
  d_kappa_eff = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_1d_sts:d_kappa_eff",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_rho_cv_e = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_1d_sts:d_rho_cv_e",
      static_cast<std::size_t>(n_cells) * sizeof(double)));

  const auto diag =
      compute_conduction_diagnostics_impl(state, cfg, d_kappa_eff, d_rho_cv_e, eos_ctx);
  double global_dt_exp = diag.dt_exp;
  if (part.n_ranks > 1) {
    parallel::Reduction red(part.n_ranks);
    if (std::isfinite(global_dt_exp) && global_dt_exp > 0.0) {
      global_dt_exp = red.allreduce_min(global_dt_exp);
    } else {
      global_dt_exp = red.allreduce_min(1.0e+30);
    }
  }
  result.dt_exp = global_dt_exp;
  result.dt_cond =
      compute_dt_cond_from_dt_exp(global_dt_exp, cfg.numerics.conduction.sts_max_stages);
  result.deff_min = diag.deff_min;
  result.deff_max = diag.deff_max;

  if (!std::isfinite(global_dt_exp) || !(global_dt_exp > 0.0) ||
      (part.n_ranks > 1 && !(global_dt_exp < 1.0e+30))) {
    return result;
  }

  const int smax = (cfg.numerics.conduction.sts_max_stages > 0)
                       ? std::max(1, cfg.numerics.conduction.sts_max_stages)
                       : 100000;
  const double s_raw = std::ceil(std::sqrt(2.0 * dt / global_dt_exp));
  int n_sub = 1;
  double dt_sub = dt;
  if (cfg.numerics.conduction.sts_max_stages > 0) {
    const double eta = std::clamp(cfg.numerics.conduction.sts_subcycle_eta, 1.0e-6, 1.0);
    const double smax_factor =
        0.5 * static_cast<double>(smax) * static_cast<double>(smax + 1);
    const double dt_sts_max = smax_factor * global_dt_exp * eta;
    if (std::isfinite(dt_sts_max) && dt_sts_max > 0.0 && dt > dt_sts_max) {
      n_sub = ceil_to_int_clamped(dt / dt_sts_max);
      dt_sub = dt / static_cast<double>(n_sub);
      if (should_log_sts_subcycling()) {
        const double ratio_over_rmax =
            (smax_factor > 0.0) ? ((dt / global_dt_exp) / smax_factor)
                                : std::numeric_limits<double>::infinity();
        core::log_info("Conduction STS subcycling: n_sub=" + std::to_string(n_sub) +
                       " s_raw=" + format_val(s_raw) +
                       " smax=" + std::to_string(smax) + " dt=" + format_sci(dt) +
                       " dt_exp=" + format_sci(global_dt_exp) +
                       " dt_sub=" + format_sci(dt_sub) +
                       " R/Rmax=" + format_val(ratio_over_rmax));
      }
    }
  }
  const int stages_per_sub = conduction::sts_stage_count(dt_sub, global_dt_exp, smax);
  const long long total_stages =
      static_cast<long long>(stages_per_sub) * static_cast<long long>(n_sub);
  const long long sts_total_cap =
      static_cast<long long>(cfg.numerics.conduction.sts_total_stages_max);
  if (sts_total_cap > 0 && total_stages > sts_total_cap) {
    core::log_warning(
        "[conduction] STS total stage bound tripped: total=" +
        std::to_string(total_stages) + " (n_sub=" + std::to_string(n_sub) +
        " x stages=" + std::to_string(stages_per_sub) +
        ") > sts_total_stages_max=" + std::to_string(sts_total_cap) +
        " dt=" + format_sci(dt) + " dt_exp=" + format_sci(global_dt_exp) +
        " — requesting driver full-step retry");
    result.retry_required = true;
    result.retry_reason = "sts_total_stages_overflow";
    result.retry_total_stages = total_stages;
    result.retry_n_sub = n_sub;
    result.retry_dt_exp = global_dt_exp;
    return result;
  }
  result.sts_subcycles = n_sub;
  result.sts_stages = (total_stages > static_cast<long long>(std::numeric_limits<int>::max()))
                          ? std::numeric_limits<int>::max()
                          : static_cast<int>(total_stages);
  result.solver_residual = 0.0;
  result.solver_iterations = result.sts_stages;
  result.solver_cond_number_est = 1.0;
  if (cfg.numerics.conduction.sts_max_stages == 0 && stages_per_sub > 1000) {
    char ratio_buf[32] = {};
    std::snprintf(ratio_buf, sizeof(ratio_buf), "%.1e", dt_sub / global_dt_exp);
    core::log_info("Conduction STS auto stages: s=" + std::to_string(stages_per_sub) +
                   " (dt/dt_exp=" + std::string(ratio_buf) + ")");
  }

  std::vector<double> tau;
  initialize_sts_taus(
      tau, dt_sub, global_dt_exp, stages_per_sub, cfg.numerics.conduction.sts_damping);
  audit_sts_ladder_stability(tau, dt_sub, global_dt_exp,
                             cfg.numerics.dt.cfl_cond,
                             cfg.numerics.conduction.sts_damping, "1d_sts");

  double* d_te_tmp = nullptr;
  double* d_alpha_cells = nullptr;
  d_te_tmp = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_1d_sts:d_te_tmp",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_alpha_cells = static_cast<double*>(core::device_scratch_acquire(
      "conduction:alpha_pass:1d_sts",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  std::uint8_t* d_cell_is_void = nullptr;
  d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "conduction:conduction_step_1d_sts:d_cell_is_void",
      static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t)));
  cuda_check(cudaMemcpy(d_cell_is_void,
                        state.cell_is_void.data(),
                        static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             "Conduction: cudaMemcpy d_cell_is_void failed");

  int* d_clamp_count = nullptr;
  double* d_e_floor = nullptr;
  double* d_clampfloor = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_1d_sts:clampfloor_pack", 2 * sizeof(double)));
  d_e_floor = d_clampfloor + 0;
  d_clamp_count = reinterpret_cast<int*>(d_clampfloor + 1);

  const double h_clampfloor_init[2] = {0.0, 0.0};
  cuda_check(cudaMemcpy(d_clampfloor, h_clampfloor_init, sizeof(h_clampfloor_init),
                        cudaMemcpyHostToDevice),
             "Conduction: init clampfloor pack (sts) failed");

  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  // Ghost-extended window for alpha/diffusivity passes: interface-face
  // coefficients need the +-1 ghost cells; their min/max atomics are
  // idempotent under the duplicated ghost contributions.
  const core::State::LaunchWindow dwa = state.owned_cell_window_ghost(n_cells, 1);
  double* te_curr = state.Te.data();
  double* te_next = d_te_tmp;
  const double Te_floor = cfg.numerics.floors.Te;
  // W-G2: test_planar is the historic planar alias (verify-only, spherical
  // meshes); otherwise Mesh.geometry_1d drives via geometry_code.
  const int geom_code = cfg.numerics.conduction.test_planar
                            ? 2
                            : state.mesh.geometry_code;
  const int use_flux_limiter = (cfg.numerics.conduction.test_kappa > 0.0) ? 0 : 1;
  double* d_A_eff = nullptr;
  double* d_flux_limiter_faces = nullptr;
  if (use_flux_limiter != 0 && n_cells > 1) {
    std::vector<double> A_eff;
    std::vector<double> gamma_eff;
    compute_effective_material_properties(cfg, state, n_cells, A_eff, gamma_eff);
    (void)gamma_eff;
    TENRYU_ASSERT(A_eff.size() == static_cast<std::size_t>(n_cells),
                  "Conduction 1D STS A_eff size mismatch");
    d_A_eff = static_cast<double*>(core::device_scratch_acquire(
        "conduction:conduction_step_1d_sts:d_A_eff",
        static_cast<std::size_t>(n_cells) * sizeof(double)));
    cuda_check(cudaMemcpy(d_A_eff,
                          A_eff.data(),
                          static_cast<std::size_t>(n_cells) * sizeof(double),
                          cudaMemcpyHostToDevice),
               "Conduction: cudaMemcpy d_A_eff failed");
    d_flux_limiter_faces = static_cast<double*>(core::device_scratch_acquire(
        "conduction:conduction_step_1d_sts:d_flux_limiter_faces",
        static_cast<std::size_t>(n_cells) * sizeof(double)));
    // BUG-19: the limiter estimate follows the face_kappa_policy of the
    // consuming stage kernel (guards mirror the stage dispatch below; the
    // test-kappa/nlheat paths never dispatch Kirchhoff, so test_kappa <= 0
    // is the complete guard here).
    const char* limiter_policy_env =
        std::getenv("TENRYU_CONDUCTION_FACE_KAPPA_POLICY");
    const std::string limiter_policy =
        (limiter_policy_env != nullptr)
            ? std::string(limiter_policy_env)
            : cfg.numerics.conduction.face_kappa_policy;
    const bool limiter_face_kirchhoff =
        (limiter_policy == "kirchhoff_same_material" &&
         cfg.numerics.conduction.test_kappa <= 0.0);
    const int face_blocks = (n_cells - 1 + kBlockSize - 1) / kBlockSize;
    compute_1d_flux_limiter_faces_kernel<<<face_blocks, kBlockSize>>>(
        d_flux_limiter_faces,
        state.Te.data(),
        d_kappa_eff,
        state.rho.data(),
        state.zbar.data(),
        d_A_eff,
        state.x_r.data(),
        n_cells,
        cfg.numerics.conduction.f_lim,
        limiter_face_kirchhoff);
    cuda_check(cudaGetLastError(), "Conduction: flux limiter cache launch failed");
    cuda_check(core::debug_kernel_sync(), "Conduction: flux limiter cache failed");
  }

  // STS quasi-frozen operator:
  // Flux limiter and diffusion coefficients are computed once at super-step start
  // using initial Te. This is intentional: fully re-evaluating per stage would
  // violate Chebyshev STS stability theory, which assumes a fixed linear operator.
  // This approximation is valid when Te changes within one super-step are small
  // relative to the temperature itself.
  // W-G2 kirchhoff face diagnostics (default-inert): TENRYU_CONDUCTION_FACE_DIAG=1
  // logs a per-face closure summary every 100th conduction step (read-only;
  // required by the GXII A/B interpretation per the external verdict).
  if (n_cells > 1 && std::getenv("TENRYU_CONDUCTION_FACE_DIAG") != nullptr) {
    static long face_diag_counter = 0;
    const bool log_now = (face_diag_counter % 100 == 0);
    ++face_diag_counter;
    if (log_now) {
      const int n_faces = n_cells - 1;
      double* d_rhk = nullptr;
      double* d_reff = nullptr;
      double* d_ssat = nullptr;
      std::uint8_t* d_active = nullptr;
      d_rhk = static_cast<double*>(core::device_scratch_acquire(
          "conduction:conduction_step_1d_sts:d_rhk",
          static_cast<std::size_t>(n_faces) * sizeof(double)));
      d_reff = static_cast<double*>(core::device_scratch_acquire(
          "conduction:conduction_step_1d_sts:d_reff",
          static_cast<std::size_t>(n_faces) * sizeof(double)));
      d_ssat = static_cast<double*>(core::device_scratch_acquire(
          "conduction:conduction_step_1d_sts:d_ssat",
          static_cast<std::size_t>(n_faces) * sizeof(double)));
      d_active = static_cast<std::uint8_t*>(core::device_scratch_acquire(
          "conduction:conduction_step_1d_sts:d_active",
          static_cast<std::size_t>(n_faces) * sizeof(std::uint8_t)));
      const int face_blocks = (n_faces + kBlockSize - 1) / kBlockSize;
      face_kappa_diag_1d_kernel<<<face_blocks, kBlockSize>>>(
          d_rhk, d_reff, d_ssat, d_active, te_curr, d_kappa_eff,
          d_flux_limiter_faces, n_faces);
      cuda_check(cudaGetLastError(), "Conduction: face diag launch failed");
      cuda_check(core::debug_kernel_sync(), "Conduction: face diag failed");
      std::vector<double> h_rhk(static_cast<std::size_t>(n_faces), 1.0);
      std::vector<double> h_reff(static_cast<std::size_t>(n_faces), 1.0);
      std::vector<double> h_ssat(static_cast<std::size_t>(n_faces), -1.0);
      std::vector<std::uint8_t> h_active(static_cast<std::size_t>(n_faces), 0);
      cuda_check(cudaMemcpy(h_rhk.data(), d_rhk,
                            static_cast<std::size_t>(n_faces) * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "Conduction: face diag copy r_hk failed");
      cuda_check(cudaMemcpy(h_reff.data(), d_reff,
                            static_cast<std::size_t>(n_faces) * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "Conduction: face diag copy r_eff failed");
      cuda_check(cudaMemcpy(h_ssat.data(), d_ssat,
                            static_cast<std::size_t>(n_faces) * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "Conduction: face diag copy s failed");
      cuda_check(cudaMemcpy(h_active.data(), d_active,
                            static_cast<std::size_t>(n_faces) * sizeof(std::uint8_t),
                            cudaMemcpyDeviceToHost),
                 "Conduction: face diag copy active failed");
      std::vector<double> rhk_act;
      std::vector<double> reff_act;
      int n_sat = 0;
      int n_reff_low = 0;
      for (int f = 0; f < n_faces; ++f) {
        if (h_active[static_cast<std::size_t>(f)] == static_cast<std::uint8_t>(0)) {
          continue;
        }
        rhk_act.push_back(h_rhk[static_cast<std::size_t>(f)]);
        reff_act.push_back(h_reff[static_cast<std::size_t>(f)]);
        if (h_ssat[static_cast<std::size_t>(f)] > 1.0) {
          ++n_sat;
        }
        if (h_reff[static_cast<std::size_t>(f)] < 0.9) {
          ++n_reff_low;
        }
      }
      if (!rhk_act.empty()) {
        auto median_of = [](std::vector<double> v) {
          const std::size_t mid = v.size() / 2;
          std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid),
                           v.end());
          return v[mid];
        };
        const double rhk_min = *std::min_element(rhk_act.begin(), rhk_act.end());
        const double reff_min = *std::min_element(reff_act.begin(), reff_act.end());
        core::log_info(
            "[conduction:face_diag] step=" + std::to_string(face_diag_counter - 1) +
            " faces_active=" + std::to_string(rhk_act.size()) +
            " R_HK_min=" + format_sci(rhk_min) +
            " R_HK_med=" + format_sci(median_of(rhk_act)) +
            " R_eff_min=" + format_sci(reff_min) +
            " R_eff_med=" + format_sci(median_of(reff_act)) +
            " n_Reff_lt0.9=" + std::to_string(n_reff_low) +
            " n_sat_s_gt1=" + std::to_string(n_sat));
      }
    }
  }
  auto launch_stage = [&](auto geom_tag, const double tau_stage) {
    constexpr int GEOM = decltype(geom_tag)::value;
    conduction_alpha_pass_kernel<GEOM><<<dwa.blocks(), kBlockSize>>>(
        d_alpha_cells,
        te_curr,
        d_kappa_eff,
        d_flux_limiter_faces,
        d_rho_cv_e,
        d_cell_is_void,
        state.vol.data(),
        state.x_r.data(),
        dwa.begin,
        dwa.end,
        n_cells,
        tau_stage,
        Te_floor,
        floor_limiter_mode);
    conduction_1d_sts_stage_kernel<GEOM><<<cw.blocks(), kBlockSize>>>(
        te_curr,
        te_next,
        d_kappa_eff,
        d_flux_limiter_faces,
        d_rho_cv_e,
        d_cell_is_void,
        state.vol.data(),
        state.x_r.data(),
        cw.begin,
        cw.end,
        n_cells,
        tau_stage,
        Te_floor,
        d_alpha_cells,
        d_clamp_count,
        d_e_floor,
        floor_limiter_mode);
  };
  // W-G2 nlheat verify hook (default-inert): under
  // TENRYU_CONDUCTION_TEST_KAPPA_POWER on the test_kappa path, stage fluxes use
  // the Kirchhoff-secant face conductivity (degenerate-front-correct) instead
  // of the harmonic mean of cell kappas. Env unset -> historic path untouched.
  const char* stage_kappa_power_env =
      std::getenv("TENRYU_CONDUCTION_TEST_KAPPA_POWER");
  const bool stage_kappa_power_active =
      (stage_kappa_power_env != nullptr &&
       cfg.numerics.conduction.test_kappa > 0.0);
  const double stage_kappa_power =
      stage_kappa_power_active ? std::atof(stage_kappa_power_env) : 0.0;
  const char* stage_kappa_rho_power_env =
      std::getenv("TENRYU_CONDUCTION_TEST_KAPPA_RHO_POWER");
  const double stage_kappa_rho_power =
      (stage_kappa_power_active && stage_kappa_rho_power_env != nullptr)
          ? std::atof(stage_kappa_rho_power_env)
          : 0.0;
  auto launch_stage_secant = [&](auto geom_tag, const double tau_stage) {
    constexpr int GEOM = decltype(geom_tag)::value;
    conduction_alpha_pass_secant_kernel<GEOM><<<dwa.blocks(), kBlockSize>>>(
        d_alpha_cells,
        te_curr,
        d_rho_cv_e,
        d_cell_is_void,
        state.vol.data(),
        state.x_r.data(),
        state.rho.data(),
        dwa.begin,
        dwa.end,
        n_cells,
        tau_stage,
        Te_floor,
        cfg.numerics.conduction.test_kappa,
        stage_kappa_power,
        stage_kappa_rho_power,
        floor_limiter_mode);
    conduction_1d_sts_stage_secant_kernel<GEOM><<<cw.blocks(), kBlockSize>>>(
        te_curr,
        te_next,
        d_rho_cv_e,
        d_cell_is_void,
        state.vol.data(),
        state.x_r.data(),
        state.rho.data(),
        cw.begin,
        cw.end,
        n_cells,
        tau_stage,
        Te_floor,
        cfg.numerics.conduction.test_kappa,
        stage_kappa_power,
        stage_kappa_rho_power,
        d_alpha_cells,
        d_clamp_count,
        d_e_floor,
        floor_limiter_mode);
  };
  // W-G2 kirchhoff face_kappa_policy (env bridge; unset = historic harmonic,
  // byte-identical). Production Spitzer path only; the nlheat hook wins when
  // both are set.
  const char* face_policy_env = std::getenv("TENRYU_CONDUCTION_FACE_KAPPA_POLICY");
  const std::string face_policy =
      (face_policy_env != nullptr) ? std::string(face_policy_env)
                                   : cfg.numerics.conduction.face_kappa_policy;
  const bool face_policy_kirchhoff_base = (face_policy == "kirchhoff_same_material");
  const bool stage_face_policy_kirchhoff =
      (face_policy_kirchhoff_base &&
       cfg.numerics.conduction.test_kappa <= 0.0 && !stage_kappa_power_active);
  auto launch_stage_kirchhoff = [&](auto geom_tag, const double tau_stage) {
    constexpr int GEOM = decltype(geom_tag)::value;
    conduction_alpha_pass_kirchhoff_kernel<GEOM><<<dwa.blocks(), kBlockSize>>>(
        d_alpha_cells,
        te_curr,
        d_kappa_eff,
        d_flux_limiter_faces,
        d_rho_cv_e,
        d_cell_is_void,
        state.vol.data(),
        state.x_r.data(),
        dwa.begin,
        dwa.end,
        n_cells,
        tau_stage,
        Te_floor,
        floor_limiter_mode);
    conduction_1d_sts_stage_kirchhoff_kernel<GEOM><<<cw.blocks(), kBlockSize>>>(
        te_curr,
        te_next,
        d_kappa_eff,
        d_flux_limiter_faces,
        d_rho_cv_e,
        d_cell_is_void,
        state.vol.data(),
        state.x_r.data(),
        cw.begin,
        cw.end,
        n_cells,
        tau_stage,
        Te_floor,
        d_alpha_cells,
        d_clamp_count,
        d_e_floor,
        floor_limiter_mode);
  };
  const bool snb_on =
      (cfg.numerics.conduction.nonlocal_model == "snb") && (n_cells >= 3);
  const bool fused_kirchhoff_on =
      (part.n_ranks == 1 && stage_face_policy_kirchhoff && !snb_on);
  if (snb_on) {
    TENRYU_ASSERT(part.n_ranks == 1,
                  "SNB nonlocal conduction (v1) requires serial (single-rank) 1D");
    TENRYU_ASSERT(!stage_kappa_power_active,
                  "SNB nonlocal conduction does not support the "
                  "TENRYU_CONDUCTION_TEST_KAPPA_POWER hook");
    double* d_A_eff_snb = d_A_eff;
    if (d_A_eff_snb == nullptr) {
      std::vector<double> A_eff_snb;
      std::vector<double> gamma_eff_snb;
      compute_effective_material_properties(cfg, state, n_cells, A_eff_snb,
                                            gamma_eff_snb);
      (void)gamma_eff_snb;
      d_A_eff_snb = static_cast<double*>(core::device_scratch_acquire(
          "conduction:conduction_step_1d_sts:d_A_eff",
          static_cast<std::size_t>(n_cells) * sizeof(double)));
      cuda_check(cudaMemcpy(d_A_eff_snb, A_eff_snb.data(),
                            static_cast<std::size_t>(n_cells) * sizeof(double),
                            cudaMemcpyHostToDevice),
                 "Conduction: cudaMemcpy d_A_eff (snb) failed");
    }
    snb::conduction_step_1d_sts_snb(state, cfg, result, d_kappa_eff, d_rho_cv_e,
                                    d_cell_is_void, d_A_eff_snb, n_cells,
                                    geom_code, stage_face_policy_kirchhoff,
                                    use_flux_limiter != 0, tau, n_sub, Te_floor,
                                    d_clamp_count, d_e_floor,
                                    state.zmom_active ? state.zmom_r2.data()
                                                      : nullptr);
    cuda_check(core::debug_kernel_sync(), "Conduction: SNB step failed");
  } else if (fused_kirchhoff_on) {
    double* d_stage_tau = static_cast<double*>(core::device_scratch_acquire(
        "conduction:conduction_step_1d_sts:d_stage_tau",
        static_cast<std::size_t>(stages_per_sub) * sizeof(double)));
    auto launch_fused_kirchhoff = [&](auto geom_tag) {
      constexpr int GEOM = decltype(geom_tag)::value;
      conduction_1d_sts_fused_kirchhoff_kernel<GEOM><<<1, kBlockSize>>>(
          te_curr,
          te_next,
          d_alpha_cells,
          d_kappa_eff,
          d_flux_limiter_faces,
          d_rho_cv_e,
          d_cell_is_void,
          state.vol.data(),
          state.x_r.data(),
          n_cells,
          d_stage_tau,
          stages_per_sub,
          Te_floor,
          d_clamp_count,
          d_e_floor,
          floor_limiter_mode);
    };
    // Fused ping-pong parity:
    // stages/sub even, any n_sub: te_curr=state.Te, te_next=d_te_tmp
    // stages/sub odd, n_sub even: te_curr=state.Te, te_next=d_te_tmp
    // stages/sub odd, n_sub odd: te_curr=d_te_tmp, te_next=state.Te
    for (int sub = 0; sub < n_sub; ++sub) {
      cuda_check(cudaMemcpy(d_stage_tau,
                            tau.data(),
                            static_cast<std::size_t>(stages_per_sub) *
                                sizeof(double),
                            cudaMemcpyHostToDevice),
                 "Conduction: upload fused STS stage taus failed");
      switch (geom_code) {
        case 1:
          launch_fused_kirchhoff(std::integral_constant<int, 1>{});
          break;
        case 2:
          launch_fused_kirchhoff(std::integral_constant<int, 2>{});
          break;
        default:
          launch_fused_kirchhoff(std::integral_constant<int, 0>{});
          break;
      }
      cuda_check(cudaGetLastError(), "Conduction: fused STS stage launch failed");
      cuda_check(core::debug_kernel_sync(),
                 "Conduction: fused STS stage execution failed");
      if ((stages_per_sub & 1) != 0) {
        std::swap(te_curr, te_next);
      }
    }
  } else {
  // (historic stage loop below runs unchanged)
  for (int sub = 0; sub < n_sub; ++sub) {
    for (const double tau_j : tau) {
      exchange_te_halo(te_curr);
      if (stage_kappa_power_active) {
        switch (geom_code) {
          case 1:
            launch_stage_secant(std::integral_constant<int, 1>{}, tau_j);
            break;
          case 2:
            launch_stage_secant(std::integral_constant<int, 2>{}, tau_j);
            break;
          default:
            launch_stage_secant(std::integral_constant<int, 0>{}, tau_j);
            break;
        }
      } else if (stage_face_policy_kirchhoff) {
        switch (geom_code) {
          case 1:
            launch_stage_kirchhoff(std::integral_constant<int, 1>{}, tau_j);
            break;
          case 2:
            launch_stage_kirchhoff(std::integral_constant<int, 2>{}, tau_j);
            break;
          default:
            launch_stage_kirchhoff(std::integral_constant<int, 0>{}, tau_j);
            break;
        }
      } else {
        switch (geom_code) {
          case 1:
            launch_stage(std::integral_constant<int, 1>{}, tau_j);
            break;
          case 2:
            launch_stage(std::integral_constant<int, 2>{}, tau_j);
            break;
          default:
            launch_stage(std::integral_constant<int, 0>{}, tau_j);
            break;
        }
      }
      cuda_check(cudaGetLastError(), "Conduction: STS stage launch failed");
      cuda_check(core::debug_kernel_sync(), "Conduction: STS stage execution failed");
      std::swap(te_curr, te_next);
    }
  }
  }

  if (te_curr != state.Te.data()) {
    cuda_check(cudaMemcpy(state.Te.data(),
                          te_curr,
                          static_cast<std::size_t>(n_cells) * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Conduction: copy staged Te back failed");
  }

  conduction_sync_eos(state, cfg);
  if (part.n_ranks > 1 && bufs != nullptr) {
    cudaStream_t halo_stream = stream;
    double* te_ptr[] = {state.Te.data()};
    parallel::exchange_cell_fields(
        part, *bufs, te_ptr, 1, state.mesh.topo.n_cells, halo_stream, 4);
  }

  double h_clampfloor[2] = {0.0, 0.0};
  cuda_check(cudaMemcpy(h_clampfloor, d_clampfloor, sizeof(h_clampfloor),
                        cudaMemcpyDeviceToHost),
             "Conduction: copy clampfloor pack (sts) failed");
  result.E_floor_injected = h_clampfloor[0];
  std::memcpy(&result.clamp_count, &h_clampfloor[1], sizeof(int));

  if (result.clamp_count > cfg.numerics.safety.clamp_warn_threshold) {
    core::log_warning("Conduction floor clamp count exceeded warning threshold: " +
                      std::to_string(result.clamp_count));
  }

  return result;
}

ConductionResult conduction_step_2d_sts(core::State& state,
                                        const double dt,
                                        const core::Config& cfg,
                                        const parallel::PartitionInfo& part,
                                        parallel::CommBuffers* bufs,
                                        cudaStream_t stream,
                                        const HydroEOSContext* eos_ctx) {
  ConductionResult result;
  result.dt_cond = std::numeric_limits<double>::infinity();
  result.dt_exp = std::numeric_limits<double>::infinity();
  const int floor_limiter_mode =
      (cfg.numerics.conduction.sts_floor_limiter == "donor") ? 1 : 0;

  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0) {
    return result;
  }
  if (per_material_conduction_ready(state, cfg, eos_ctx)) {
    return conduction_step_2d_sts_per_material(state, dt, cfg, eos_ctx, part, bufs, stream);
  }
  const bool do_exchange = (part.n_ranks > 1 && bufs != nullptr);
  const std::string& halo_strategy = cfg.numerics.conduction.halo_strategy;
  if (halo_strategy == "adaptive") {
    // TODO(v1.1): implement adaptive conduction halo strategy.
  }
  auto exchange_te_halo = [&](double* te_ptr) {
    if (!do_exchange) {
      return;
    }
    double* Te_ptr = te_ptr;
    parallel::exchange_cell_fields(part, *bufs, &Te_ptr, 1, n_cells, stream, 3);
  };
  exchange_te_halo(state.Te.data());

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  TENRYU_ASSERT(nr * nz == n_cells, "Conduction 2D requires nr*nz == n_cells");

  double* d_kappa_eff = nullptr;
  double* d_rho_cv_e = nullptr;
  double* d_stencil = nullptr;
  d_kappa_eff = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_2d_sts:d_kappa_eff",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_rho_cv_e = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_2d_sts:d_rho_cv_e",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_stencil = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_2d_sts:d_stencil",
      static_cast<std::size_t>(n_cells) * kershaw::kStencilSize *
          sizeof(double)));

  const auto diag =
      compute_conduction_diagnostics_impl(state, cfg, d_kappa_eff, d_rho_cv_e, eos_ctx);
  double global_dt_exp = diag.dt_exp;
  if (part.n_ranks > 1) {
    parallel::Reduction red(part.n_ranks);
    if (std::isfinite(global_dt_exp) && global_dt_exp > 0.0) {
      global_dt_exp = red.allreduce_min(global_dt_exp);
    } else {
      global_dt_exp = red.allreduce_min(1.0e+30);
    }
  }
  result.dt_exp = global_dt_exp;
  result.dt_cond =
      compute_dt_cond_from_dt_exp(global_dt_exp, cfg.numerics.conduction.sts_max_stages);
  result.deff_min = diag.deff_min;
  result.deff_max = diag.deff_max;

  if (!std::isfinite(global_dt_exp) || !(global_dt_exp > 0.0) ||
      (part.n_ranks > 1 && !(global_dt_exp < 1.0e+30))) {
    return result;
  }

  const int smax = (cfg.numerics.conduction.sts_max_stages > 0)
                       ? std::max(1, cfg.numerics.conduction.sts_max_stages)
                       : 100000;
  const double s_raw = std::ceil(std::sqrt(2.0 * dt / global_dt_exp));
  int n_sub = 1;
  double dt_sub = dt;
  if (cfg.numerics.conduction.sts_max_stages > 0) {
    const double eta = std::clamp(cfg.numerics.conduction.sts_subcycle_eta, 1.0e-6, 1.0);
    const double smax_factor =
        0.5 * static_cast<double>(smax) * static_cast<double>(smax + 1);
    const double dt_sts_max = smax_factor * global_dt_exp * eta;
    if (std::isfinite(dt_sts_max) && dt_sts_max > 0.0 && dt > dt_sts_max) {
      n_sub = ceil_to_int_clamped(dt / dt_sts_max);
      dt_sub = dt / static_cast<double>(n_sub);
      if (should_log_sts_subcycling()) {
        const double ratio_over_rmax =
            (smax_factor > 0.0) ? ((dt / global_dt_exp) / smax_factor)
                                : std::numeric_limits<double>::infinity();
        core::log_info("Conduction STS subcycling: n_sub=" + std::to_string(n_sub) +
                       " s_raw=" + format_val(s_raw) +
                       " smax=" + std::to_string(smax) + " dt=" + format_sci(dt) +
                       " dt_exp=" + format_sci(global_dt_exp) +
                       " dt_sub=" + format_sci(dt_sub) +
                       " R/Rmax=" + format_val(ratio_over_rmax));
      }
    }
  }
  const int stages_per_sub = conduction::sts_stage_count(dt_sub, global_dt_exp, smax);
  const long long total_stages =
      static_cast<long long>(stages_per_sub) * static_cast<long long>(n_sub);
  const long long sts_total_cap =
      static_cast<long long>(cfg.numerics.conduction.sts_total_stages_max);
  if (sts_total_cap > 0 && total_stages > sts_total_cap) {
    core::log_warning(
        "[conduction] STS total stage bound tripped: total=" +
        std::to_string(total_stages) + " (n_sub=" + std::to_string(n_sub) +
        " x stages=" + std::to_string(stages_per_sub) +
        ") > sts_total_stages_max=" + std::to_string(sts_total_cap) +
        " dt=" + format_sci(dt) + " dt_exp=" + format_sci(global_dt_exp) +
        " — requesting driver full-step retry");
    result.retry_required = true;
    result.retry_reason = "sts_total_stages_overflow";
    result.retry_total_stages = total_stages;
    result.retry_n_sub = n_sub;
    result.retry_dt_exp = global_dt_exp;
    return result;
  }
  result.sts_subcycles = n_sub;
  result.sts_stages = (total_stages > static_cast<long long>(std::numeric_limits<int>::max()))
                          ? std::numeric_limits<int>::max()
                          : static_cast<int>(total_stages);
  result.solver_residual = 0.0;
  result.solver_iterations = result.sts_stages;
  result.solver_cond_number_est = 1.0;
  if (cfg.numerics.conduction.sts_max_stages == 0 && stages_per_sub > 1000) {
    char ratio_buf[32] = {};
    std::snprintf(ratio_buf, sizeof(ratio_buf), "%.1e", dt_sub / global_dt_exp);
    core::log_info("Conduction STS auto stages: s=" + std::to_string(stages_per_sub) +
                   " (dt/dt_exp=" + std::string(ratio_buf) + ")");
  }

  std::vector<double> tau;
  initialize_sts_taus(
      tau, dt_sub, global_dt_exp, stages_per_sub, cfg.numerics.conduction.sts_damping);

  double* d_te_tmp = nullptr;
  double* d_alpha_cells = nullptr;
  d_te_tmp = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_2d_sts:d_te_tmp",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_alpha_cells = static_cast<double*>(core::device_scratch_acquire(
      "conduction:alpha_pass:2d_sts",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  std::uint8_t* d_cell_is_void = nullptr;
  d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "conduction:conduction_step_2d_sts:d_cell_is_void",
      static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t)));
  cuda_check(cudaMemcpy(d_cell_is_void,
                        state.cell_is_void.data(),
                        static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             "Conduction: cudaMemcpy d_cell_is_void failed");

  int* d_clamp_count = nullptr;
  double* d_e_floor = nullptr;
  int* d_mmatrix_fix_count = nullptr;
  d_clamp_count = static_cast<int*>(core::device_scratch_acquire(
      "conduction:conduction_step_2d_sts:d_clamp_count", sizeof(int)));
  d_e_floor = static_cast<double*>(core::device_scratch_acquire(
      "conduction:conduction_step_2d_sts:d_e_floor", sizeof(double)));
  d_mmatrix_fix_count = static_cast<int*>(core::device_scratch_acquire(
      "conduction:conduction_step_2d_sts:d_mmatrix_fix_count", sizeof(int)));

  const int zero_i = 0;
  const double zero_d = 0.0;
  cuda_check(cudaMemcpy(d_clamp_count, &zero_i, sizeof(int), cudaMemcpyHostToDevice),
             "Conduction: init d_clamp_count failed");
  cuda_check(cudaMemcpy(d_e_floor, &zero_d, sizeof(double), cudaMemcpyHostToDevice),
             "Conduction: init d_e_floor failed");
  cuda_check(cudaMemcpy(d_mmatrix_fix_count, &zero_i, sizeof(int), cudaMemcpyHostToDevice),
             "Conduction: init d_mmatrix_fix_count failed");

  const int bc_r_lo = map_hydro_bc_to_kershaw("axis");
  const int bc_r_hi = map_hydro_bc_to_kershaw(cfg.numerics.hydro.boundary_2d.r_outer);
  const int bc_z_lo = map_hydro_bc_to_kershaw(cfg.numerics.hydro.boundary_2d.z_bottom);
  const int bc_z_hi = map_hydro_bc_to_kershaw(cfg.numerics.hydro.boundary_2d.z_top);

  kershaw::launch_kershaw_stencil_build(d_stencil,
                                        d_kappa_eff,
                                        state.x_r.data(),
                                        state.x_z.data(),
                                        nr,
                                        nz,
                                        bc_r_lo,
                                        bc_r_hi,
                                        bc_z_lo,
                                        bc_z_hi,
                                        true,
                                        d_mmatrix_fix_count);
  cuda_check(cudaGetLastError(), "Conduction: kershaw_stencil_build launch failed");
  cuda_check(core::debug_kernel_sync(), "Conduction: kershaw_stencil_build failed");

  cuda_check(cudaMemcpy(&result.mmatrix_fix_count,
                        d_mmatrix_fix_count,
                        sizeof(int),
                        cudaMemcpyDeviceToHost),
             "Conduction: copy mmatrix_fix_count failed");
  result.solver_cond_number_est =
      kershaw_diagonal_ratio_condition_estimate(state, d_stencil, d_rho_cv_e, dt, n_cells);
  if (result.mmatrix_fix_count > n_cells / 10) {
    core::log_warning("Conduction Kershaw M-matrix repair applied to >10% cells");
  }

  double* te_curr = state.Te.data();
  double* te_next = d_te_tmp;
  const double Te_floor = cfg.numerics.floors.Te;

  const bool snb_on =
      (cfg.numerics.conduction.nonlocal_model == "snb") && (n_cells >= 9);
  if (snb_on) {
    TENRYU_ASSERT(part.n_ranks == 1,
                  "SNB nonlocal conduction (2D v1) requires a single rank");
    std::vector<double> A_eff_snb;
    std::vector<double> gamma_eff_snb;
    compute_effective_material_properties(cfg, state, n_cells, A_eff_snb,
                                          gamma_eff_snb);
    (void)gamma_eff_snb;
    double* d_A_eff_snb = static_cast<double*>(core::device_scratch_acquire(
        "conduction:conduction_step_2d_sts:d_A_eff_snb",
        static_cast<std::size_t>(n_cells) * sizeof(double)));
    cuda_check(cudaMemcpy(d_A_eff_snb, A_eff_snb.data(),
                          static_cast<std::size_t>(n_cells) * sizeof(double),
                          cudaMemcpyHostToDevice),
               "Conduction: cudaMemcpy d_A_eff (snb2d) failed");
    snb2d::conduction_step_2d_sts_snb(state, cfg, result, d_stencil,
                                      d_kappa_eff, d_rho_cv_e, d_cell_is_void,
                                      d_A_eff_snb, nr, nz, tau, n_sub,
                                      Te_floor, d_clamp_count, d_e_floor);
    cuda_check(core::debug_kernel_sync(), "Conduction: SNB 2D step failed");
  } else {
  // (historic stage loop below runs unchanged)
  // Per-cell local safety alpha is computed before kershaw_apply_kernel; apply
  // scales each symmetric pair by min(alpha_c, alpha_n) before floor clamp.
  for (int sub = 0; sub < n_sub; ++sub) {
    for (const double tau_j : tau) {
      exchange_te_halo(te_curr);
      kershaw::launch_kershaw_alpha_pass(d_alpha_cells,
                                         te_curr,
                                         d_stencil,
                                         d_rho_cv_e,
                                         d_cell_is_void,
                                         state.vol.data(),
                                         tau_j,
                                         Te_floor,
                                         nr,
                                         nz,
                                         floor_limiter_mode);
      kershaw::launch_kershaw_apply(te_next,
                                    te_curr,
                                    d_stencil,
                                    d_rho_cv_e,
                                    d_cell_is_void,
                                    state.vol.data(),
                                    tau_j,
                                    Te_floor,
                                    nr,
                                    nz,
                                    d_clamp_count,
                                    d_e_floor,
                                    d_alpha_cells,
                                    floor_limiter_mode);
      cuda_check(cudaGetLastError(), "Conduction: kershaw_apply launch failed");
      cuda_check(core::debug_kernel_sync(), "Conduction: kershaw_apply execution failed");
      std::swap(te_curr, te_next);
    }
  }

  if (te_curr != state.Te.data()) {
    cuda_check(cudaMemcpy(state.Te.data(),
                          te_curr,
                          static_cast<std::size_t>(n_cells) * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Conduction: copy staged Te back failed");
  }
  }

  accumulate_state_supply_conduction_bc(state, cfg, dt, d_kappa_eff, d_rho_cv_e);
  conduction_sync_eos(state, cfg);
  if (part.n_ranks > 1 && bufs != nullptr) {
    cudaStream_t halo_stream = stream;
    double* te_ptr[] = {state.Te.data()};
    parallel::exchange_cell_fields(
        part, *bufs, te_ptr, 1, state.mesh.topo.n_cells, halo_stream, 4);
  }

  cuda_check(cudaMemcpy(&result.clamp_count, d_clamp_count, sizeof(int), cudaMemcpyDeviceToHost),
             "Conduction: copy clamp_count failed");
  cuda_check(cudaMemcpy(&result.E_floor_injected,
                        d_e_floor,
                        sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Conduction: copy E_floor failed");

  if (result.clamp_count > cfg.numerics.safety.clamp_warn_threshold) {
    core::log_warning("Conduction floor clamp count exceeded warning threshold: " +
                      std::to_string(result.clamp_count));
  }

  return result;
}

#if TENRYU_ENABLE_HYPRE
ConductionResult conduction_step_2d_hypre(core::State& state,
                                          const double dt,
                                          const core::Config& cfg,
                                          const parallel::PartitionInfo& part,
                                          parallel::CommBuffers* bufs,
                                          cudaStream_t stream) {
  ConductionResult result;
  result.dt_cond = std::numeric_limits<double>::infinity();
  result.dt_exp = std::numeric_limits<double>::infinity();

  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0) {
    return result;
  }
  // TODO(v1.1): complete full MPI integration for Hypre matrix assembly/solve.

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  TENRYU_ASSERT(nr * nz == n_cells, "Conduction 2D requires nr*nz == n_cells");

  double* d_kappa_eff = nullptr;
  double* d_rho_cv_e = nullptr;
  double* d_stencil = nullptr;
  double* d_te_out = nullptr;
  int* d_mmatrix_fix_count = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_kappa_eff),
                        static_cast<std::size_t>(n_cells) * sizeof(double)),
             "Conduction: cudaMalloc d_kappa_eff failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rho_cv_e),
                        static_cast<std::size_t>(n_cells) * sizeof(double)),
             "Conduction: cudaMalloc d_rho_cv_e failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_stencil),
                        static_cast<std::size_t>(n_cells) * kershaw::kStencilSize *
                            sizeof(double)),
             "Conduction: cudaMalloc d_stencil failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_te_out),
                        static_cast<std::size_t>(n_cells) * sizeof(double)),
             "Conduction: cudaMalloc d_te_out failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mmatrix_fix_count), sizeof(int)),
             "Conduction: cudaMalloc d_mmatrix_fix_count failed");

  const int zero_i = 0;
  cuda_check(cudaMemcpy(d_mmatrix_fix_count, &zero_i, sizeof(int), cudaMemcpyHostToDevice),
             "Conduction: init d_mmatrix_fix_count failed");

  const auto diag =
      compute_conduction_diagnostics_impl(state, cfg, d_kappa_eff, d_rho_cv_e, nullptr);
  double global_dt_exp = diag.dt_exp;
  if (part.n_ranks > 1) {
    parallel::Reduction red(part.n_ranks);
    if (std::isfinite(global_dt_exp) && global_dt_exp > 0.0) {
      global_dt_exp = red.allreduce_min(global_dt_exp);
    } else {
      global_dt_exp = red.allreduce_min(1.0e+30);
    }
  }
  result.dt_exp = global_dt_exp;
  result.dt_cond = std::numeric_limits<double>::infinity();
  result.deff_min = diag.deff_min;
  result.deff_max = diag.deff_max;

  if (!std::isfinite(global_dt_exp) || !(global_dt_exp > 0.0) ||
      (part.n_ranks > 1 && !(global_dt_exp < 1.0e+30))) {
    cuda_check(cudaFree(d_mmatrix_fix_count),
               "Conduction: cudaFree d_mmatrix_fix_count failed");
    cuda_check(cudaFree(d_te_out), "Conduction: cudaFree d_te_out failed");
    cuda_check(cudaFree(d_stencil), "Conduction: cudaFree d_stencil failed");
    cuda_check(cudaFree(d_rho_cv_e), "Conduction: cudaFree d_rho_cv_e failed");
    cuda_check(cudaFree(d_kappa_eff), "Conduction: cudaFree d_kappa_eff failed");
    return result;
  }

  const int bc_r_lo = map_hydro_bc_to_kershaw("axis");
  const int bc_r_hi = map_hydro_bc_to_kershaw(cfg.numerics.hydro.boundary_2d.r_outer);
  const int bc_z_lo = map_hydro_bc_to_kershaw(cfg.numerics.hydro.boundary_2d.z_bottom);
  const int bc_z_hi = map_hydro_bc_to_kershaw(cfg.numerics.hydro.boundary_2d.z_top);

  kershaw::launch_kershaw_stencil_build(d_stencil,
                                        d_kappa_eff,
                                        state.x_r.data(),
                                        state.x_z.data(),
                                        nr,
                                        nz,
                                        bc_r_lo,
                                        bc_r_hi,
                                        bc_z_lo,
                                        bc_z_hi,
                                        true,
                                        d_mmatrix_fix_count);
  cuda_check(cudaGetLastError(), "Conduction: kershaw_stencil_build launch failed");
  cuda_check(core::debug_kernel_sync(), "Conduction: kershaw_stencil_build failed");

  cuda_check(cudaMemcpy(&result.mmatrix_fix_count,
                        d_mmatrix_fix_count,
                        sizeof(int),
                        cudaMemcpyDeviceToHost),
             "Conduction: copy mmatrix_fix_count failed");
  result.solver_residual = 0.0;
  result.solver_iterations = 1;
  result.solver_cond_number_est =
      kershaw_diagonal_ratio_condition_estimate(state, d_stencil, d_rho_cv_e, dt, n_cells);
  if (result.mmatrix_fix_count > n_cells / 10) {
    core::log_warning("Conduction Kershaw M-matrix repair applied to >10% cells");
  }

  static HypreSolver hypre_solver;
  if (!hypre_solver.initialized || hypre_solver.n_local_cells != n_cells ||
      hypre_solver.stencil_width != kershaw::kStencilSize) {
    if (hypre_solver.initialized) {
      hypre_solver.destroy();
    }
    hypre_solver.init(n_cells, kershaw::kStencilSize);
  }

  hypre_solver.update_matrix(d_stencil, d_rho_cv_e, dt, n_cells, nullptr);
  TENRYU_ASSERT(hypre_solver.solve(d_te_out, state.Te.data(), nullptr) >= 0,
                "Conduction Hypre solve failed");

  cuda_check(cudaMemcpy(state.Te.data(),
                        d_te_out,
                        static_cast<std::size_t>(n_cells) * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Conduction: copy hypre Te back failed");

  accumulate_state_supply_conduction_bc(state, cfg, dt, d_kappa_eff, d_rho_cv_e);
  conduction_sync_eos(state, cfg);
  if (part.n_ranks > 1 && bufs != nullptr) {
    cudaStream_t halo_stream = stream;
    double* te_ptr[] = {state.Te.data()};
    parallel::exchange_cell_fields(
        part, *bufs, te_ptr, 1, state.mesh.topo.n_cells, halo_stream, 4);
  }

  cuda_check(cudaFree(d_mmatrix_fix_count),
             "Conduction: cudaFree d_mmatrix_fix_count failed");
  cuda_check(cudaFree(d_te_out), "Conduction: cudaFree d_te_out failed");
  cuda_check(cudaFree(d_stencil), "Conduction: cudaFree d_stencil failed");
  cuda_check(cudaFree(d_rho_cv_e), "Conduction: cudaFree d_rho_cv_e failed");
  cuda_check(cudaFree(d_kappa_eff), "Conduction: cudaFree d_kappa_eff failed");

  return result;
}
#endif

double compute_dt_cond_from_dt_exp(const double dt_exp, const int sts_max_stages) {
  if (!std::isfinite(dt_exp) || !(dt_exp > 0.0)) {
    return std::numeric_limits<double>::infinity();
  }
  if (sts_max_stages <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  const int smax = std::max(1, sts_max_stages);
  const double factor = 0.5 * static_cast<double>(smax) * static_cast<double>(smax + 1);
  return factor * dt_exp;
}

}  // namespace

namespace conduction {

double coulomb_log(const double n_e, const double Te_eV, const double Zbar) {
  return coulomb_log_formula(n_e, Te_eV, Zbar);
}

double spitzer_conductivity(const double Te_eV,
                            const double Zbar,
                            const double ln_lambda) {
  return spitzer_kappa_formula(Te_eV, Zbar, ln_lambda);
}

double electron_thermal_velocity(const double Te_eV) {
  return vth_e_formula(Te_eV);
}

double flux_limited_heat_flux(const double q_sh, const double q_max) {
  return flux_limiter_formula(q_sh, q_max);
}

double effective_diffusion(const double q_limited,
                           const double rho,
                           const double cv_e,
                           const double grad_Te,
                           const double D_sh,
                           const double eps_grad) {
  return deff_formula(q_limited, rho, cv_e, grad_Te, D_sh, eps_grad);
}

int sts_stage_count(const double dt,
                    const double dt_exp,
                    const int sts_max_stages) {
  if (!(dt > 0.0) || !(dt_exp > 0.0) || !std::isfinite(dt_exp)) {
    return 1;
  }
  const double s_raw = std::ceil(std::sqrt(2.0 * dt / dt_exp));
  if (sts_max_stages <= 0) {
    if (!std::isfinite(s_raw) ||
        s_raw > static_cast<double>(std::numeric_limits<int>::max())) {
      return 100000;
    }
    const int s = std::max(1, static_cast<int>(s_raw));
    return std::min(s, 100000);
  }
  const int smax = std::max(1, sts_max_stages);
  if (!std::isfinite(s_raw)) {
    return smax;
  }
  if (s_raw > static_cast<double>(std::numeric_limits<int>::max())) {
    return smax;
  }
  const int s = std::max(1, static_cast<int>(s_raw));
  return std::min(s, smax);
}

PerMaterialFaceCoefficients1D compute_per_material_face_coefficients_1d(
    core::State& state, const core::Config& cfg, const HydroEOSContext* eos_ctx) {
  TENRYU_ASSERT(per_material_conduction_ready(state, cfg, eos_ctx),
                "per-material face coefficients require enabled per-material state and EOS context");
  TENRYU_ASSERT(state.mesh.dim == 1, "per-material face coefficients require 1D mesh");
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  const int n_faces = std::max(0, n_cells - 1);

  double* d_kappa_face = nullptr;
  double* d_kappa_pm = nullptr;
  double* d_rho_cv_e = nullptr;
  d_kappa_face = static_cast<double*>(core::device_scratch_acquire(
      "conduction:compute_per_material_face_coefficients_1d:d_kappa_face",
      static_cast<std::size_t>(std::max(1, n_faces)) * sizeof(double)));
  d_kappa_pm = static_cast<double*>(core::device_scratch_acquire(
      "conduction:compute_per_material_face_coefficients_1d:d_kappa_pm",
      static_cast<std::size_t>(std::max(1, n_faces * n_mat)) *
          sizeof(double)));
  d_rho_cv_e = static_cast<double*>(core::device_scratch_acquire(
      "conduction:compute_per_material_face_coefficients_1d:d_rho_cv_e",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  (void)compute_per_material_1d_operator(
      state, cfg, eos_ctx, d_kappa_face, d_kappa_pm, d_rho_cv_e);

  PerMaterialFaceCoefficients1D out;
  out.n_faces = n_faces;
  out.n_mat = n_mat;
  out.aggregate_kappa.assign(static_cast<std::size_t>(n_faces), 0.0);
  out.kappa_eff_per_material.assign(
      static_cast<std::size_t>(n_faces) * static_cast<std::size_t>(n_mat), 0.0);
  if (n_faces > 0) {
    cuda_check(cudaMemcpy(out.aggregate_kappa.data(),
                          d_kappa_face,
                          static_cast<std::size_t>(n_faces) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "Conduction: cudaMemcpy test aggregate kappa failed");
    cuda_check(cudaMemcpy(out.kappa_eff_per_material.data(),
                          d_kappa_pm,
                          static_cast<std::size_t>(n_faces) *
                              static_cast<std::size_t>(n_mat) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "Conduction: cudaMemcpy test per-material kappa failed");
  }
  return out;
}

double per_material_dirichlet_boundary_flux_1d(const core::State& state,
                                               const core::Config& cfg,
                                               const int cell,
                                               const double T_boundary_eV,
                                               const double dx_face,
                                               const double f_lim) {
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  TENRYU_ASSERT(cell >= 0 && cell < n_cells, "Dirichlet boundary cell out of range");
  TENRYU_ASSERT(dx_face > 0.0, "Dirichlet boundary dx_face must be positive");
  TENRYU_ASSERT(state.volFrac.size() == static_cast<std::size_t>(n_cells) *
                                      static_cast<std::size_t>(n_mat),
                "Dirichlet boundary helper requires volFrac size n_cells*n_mat");

  std::vector<double> volfrac;
  std::vector<double> mass_m;
  std::vector<double> Ee_m;
  std::vector<double> vol;
  state.volFrac.copy_to_host(volfrac);
  state.mass_per_material.copy_to_host(mass_m);
  state.Ee_per_material.copy_to_host(Ee_m);
  state.vol.copy_to_host(vol);
  const auto params = make_conduction_material_params(cfg);

  double flux = 0.0;
  for (int m = 0; m < n_mat; ++m) {
    const std::size_t idx =
        static_cast<std::size_t>(cell) * static_cast<std::size_t>(n_mat) +
        static_cast<std::size_t>(m);
    const double vf = volfrac[idx];
    const double mass = mass_m[idx];
    if (!(vf > 0.0) || !(mass > 0.0) || !(vol[static_cast<std::size_t>(cell)] > 0.0)) {
      continue;
    }
    const ConductionMaterialParams p = params[static_cast<std::size_t>(m)];
    const double cv = per_material::ideal_species_cv(p.Zbar, p.A, p.gamma);
    if (!(cv > 0.0) || !(p.Zbar > 0.0)) {
      continue;
    }
    const double rho_m = mass / (vf * vol[static_cast<std::size_t>(cell)]);
    const double Te_m = fmax(Ee_m[idx] / mass / cv, cfg.numerics.floors.Te);
    const double Te_face = 0.5 * (sanitize_te_for_pow(Te_m) +
                                  sanitize_te_for_pow(T_boundary_eV));
    const double grad = (T_boundary_eV - Te_m) / dx_face;
    const double n_e = electron_density_formula(rho_m, p.Zbar, p.A);
    const double ln_lambda = coulomb_log_formula(n_e, Te_face, p.Zbar);
    double kappa = (cfg.numerics.conduction.test_kappa > 0.0)
                       ? cfg.numerics.conduction.test_kappa
                       : spitzer_kappa_formula(Te_face, p.Zbar, ln_lambda);
    if (kappa > 0.0 && cfg.numerics.conduction.mfp_limiter_C > 0.0) {
      kappa *= mfp_limiter_factor_formula(n_e,
                                          Te_face,
                                          p.Zbar,
                                          ln_lambda,
                                          dx_face,
                                          cfg.numerics.conduction.mfp_limiter_C);
    }
    const double q_max = q_max_formula(f_lim, n_e, Te_face);
    const double limiter =
        (q_max > 0.0) ? (1.0 / (1.0 + fabs(-kappa * grad) / q_max)) : 0.0;
    const double kappa_limited = (isfinite(limiter) && limiter > 0.0) ? kappa * limiter : 0.0;
    flux += vf * kappa_limited * grad;
  }
  return flux;
}

}  // namespace conduction

double compute_dt_conduction(const core::State& state,
                             const core::Config& cfg,
                             const HydroEOSContext* eos_ctx) {
  if (!cfg.numerics.conduction.enabled) {
    return std::numeric_limits<double>::infinity();
  }
  if (state.mesh.node_r == nullptr) {
    return std::numeric_limits<double>::infinity();
  }
  if (state.mesh.dim != 1 && state.mesh.dim != 2) {
    return std::numeric_limits<double>::infinity();
  }
  const bool per_material_ready = per_material_conduction_ready(state, cfg, eos_ctx);
  if (!per_material_ready && cfg.numerics.conduction.solver == "implicit" &&
      conduction_1d_implicit_supported(state)) {
    return std::numeric_limits<double>::infinity();
  }
  if (!per_material_ready && cfg.numerics.conduction.solver == "hypre" && state.mesh.dim == 2) {
    return std::numeric_limits<double>::infinity();
  }

  return compute_conduction_diagnostics_impl(state, cfg, nullptr, nullptr, eos_ctx).dt_cond;
}

ConductionDtArgmin compute_dt_conduction_argmin(const core::State& state,
                                                const core::Config& cfg,
                                                const HydroEOSContext* eos_ctx) {
  ConductionDtArgmin result;
  if (!cfg.numerics.conduction.enabled || state.mesh.node_r == nullptr ||
      (state.mesh.dim != 1 && state.mesh.dim != 2)) {
    return result;
  }
  const bool per_material_ready = per_material_conduction_ready(state, cfg, eos_ctx);
  if (!per_material_ready && cfg.numerics.conduction.solver == "implicit" &&
      conduction_1d_implicit_supported(state)) {
    return result;
  }
  if (!per_material_ready && cfg.numerics.conduction.solver == "hypre" && state.mesh.dim == 2) {
    return result;
  }

  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0) {
    return result;
  }
  double* d_diffusion = nullptr;
  d_diffusion = static_cast<double*>(core::device_scratch_acquire(
      "conduction:compute_dt_conduction_argmin:d_diffusion",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  cuda_check(cudaMemset(d_diffusion, 0, static_cast<std::size_t>(n_cells) * sizeof(double)),
             "Conduction: cudaMemset argmin diffusion failed");
  const ConductionDiagnostics diag =
      compute_conduction_diagnostics_impl(state, cfg, d_diffusion, nullptr, eos_ctx);

  std::vector<double> diffusion(static_cast<std::size_t>(n_cells), 0.0);
  cuda_check(cudaMemcpy(diffusion.data(),
                        d_diffusion,
                        static_cast<std::size_t>(n_cells) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Conduction: cudaMemcpy argmin diffusion failed");

  std::vector<double> x_r;
  std::vector<double> x_z;
  state.x_r.copy_to_host(x_r);
  if (state.mesh.dim == 2) {
    state.x_z.copy_to_host(x_z);
  }

  double min_ratio = std::numeric_limits<double>::infinity();
  for (int c = 0; c < n_cells; ++c) {
    const double deff = diffusion[static_cast<std::size_t>(c)];
    if (!(deff > 0.0) || !std::isfinite(deff)) {
      continue;
    }
    double dl = 0.0;
    if (state.mesh.dim == 1) {
      dl = x_r[static_cast<std::size_t>(c + 1)] - x_r[static_cast<std::size_t>(c)];
    } else {
      const int nz = state.mesh.topo.nz;
      const int i = c / nz;
      const int j = c - i * nz;
      const double area =
          kershaw::cell_area_from_nodes(x_r.data(), x_z.data(), i, j, nz);
      dl = std::sqrt(std::max(area, 1.0e-30));
    }
    if (!(dl > 0.0) || !std::isfinite(dl)) {
      continue;
    }
    const double ratio = (dl * dl) / deff;
    if (ratio < min_ratio) {
      min_ratio = ratio;
      result.argmin_cell = c;
      result.deff_at_argmin = deff;
      result.dl_at_argmin = dl;
    }
  }
  if (result.argmin_cell >= 0) {
    const double dt_exp = cfg.numerics.dt.cfl_cond * min_ratio;
    result.dt = compute_dt_cond_from_dt_exp(
        dt_exp, cfg.numerics.conduction.sts_max_stages);
  } else {
    result.dt = diag.dt_cond;
  }
  return result;
}

ConductionDiagnostics compute_conduction_diagnostics(const core::State& state,
                                                     const core::Config& cfg,
                                                     const HydroEOSContext* eos_ctx) {
  if (!cfg.numerics.conduction.enabled) {
    ConductionDiagnostics diag;
    diag.dt_exp = std::numeric_limits<double>::infinity();
    diag.dt_cond = std::numeric_limits<double>::infinity();
    return diag;
  }
  if (state.mesh.node_r == nullptr || (state.mesh.dim != 1 && state.mesh.dim != 2)) {
    ConductionDiagnostics diag;
    diag.dt_exp = std::numeric_limits<double>::infinity();
    diag.dt_cond = std::numeric_limits<double>::infinity();
    return diag;
  }
  return compute_conduction_diagnostics_impl(state, cfg, nullptr, nullptr, eos_ctx);
}

ConductionResult conduction_step(core::State& state,
                                 const double dt,
                                 const core::Config& cfg,
                                 const parallel::PartitionInfo& part,
                                 parallel::CommBuffers* bufs,
                                 cudaStream_t stream,
                                 const HydroEOSContext* eos_ctx) {
  ConductionResult result;
  result.dt_cond = std::numeric_limits<double>::infinity();
  result.dt_exp = std::numeric_limits<double>::infinity();
  result.deff_min = 0.0;
  result.deff_max = 0.0;
  result.sts_stages = 0;
  state.c1_bc_heat_flux_last.fill(0.0);

  if (!cfg.numerics.conduction.enabled || !(dt > 0.0)) {
    return result;
  }
  if (state.mesh.node_r == nullptr) {
    return result;
  }

  const std::string& solver = cfg.numerics.conduction.solver;
  const bool per_material_ready = per_material_conduction_ready(state, cfg, eos_ctx);
  if (per_material_ready && solver != "sts") {
    static bool warned_per_material_solver = false;
    if (!warned_per_material_solver) {
      core::log_warning("Per-material conduction uses STS; ignoring conduction solver='" +
                        solver + "' for this step");
      warned_per_material_solver = true;
    }
  }
  if (!per_material_ready && solver == "implicit") {
    if (conduction_1d_implicit_supported(state, &part)) {
      return conduction_step_1d_implicit(state, dt, cfg, part, bufs, stream);
    }
    static bool warned_implicit_unsupported = false;
    if (!warned_implicit_unsupported) {
      core::log_warning("Conduction solver='implicit' is only available for serial 1D_SPH; "
                        "falling back to STS");
      warned_implicit_unsupported = true;
    }
  } else if (!per_material_ready && solver == "hypre") {
#if TENRYU_ENABLE_HYPRE
    if (state.mesh.dim == 2) {
      return conduction_step_2d_hypre(state, dt, cfg, part, bufs, stream);
    }
    static bool warned_hypre_dim = false;
    if (!warned_hypre_dim) {
      core::log_warning("Conduction solver='hypre' is only available for 2D_RZ; "
                        "falling back to STS for this mesh");
      warned_hypre_dim = true;
    }
#else
    static bool warned_hypre_disabled = false;
    if (!warned_hypre_disabled) {
      core::log_warning("Conduction solver='hypre' requested but TENRYU_ENABLE_HYPRE=OFF; "
                        "falling back to STS");
      warned_hypre_disabled = true;
    }
#endif
  } else if (!per_material_ready && solver != "sts") {
    static bool warned_solver = false;
    if (!warned_solver) {
      core::log_warning("Unknown conduction solver '" + solver + "'; falling back to STS");
      warned_solver = true;
    }
  }

  if (state.mesh.dim == 1) {
    return conduction_step_1d_sts(state, dt, cfg, part, bufs, stream, eos_ctx);
  }
  if (state.mesh.dim == 2) {
    return conduction_step_2d_sts(state, dt, cfg, part, bufs, stream, eos_ctx);
  }

  return result;
}

namespace conduction {

SnbFluxProbe snb_probe_fluxes(core::State& state, const core::Config& cfg,
                              const HydroEOSContext* eos_ctx) {
  SnbFluxProbe probe;
  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells < 3 || state.mesh.dim != 1) {
    return probe;
  }
  double* d_kappa_eff = static_cast<double*>(core::device_scratch_acquire(
      "conduction:snb_probe:d_kappa_eff",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  double* d_rho_cv_e = static_cast<double*>(core::device_scratch_acquire(
      "conduction:snb_probe:d_rho_cv_e",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  const auto diag =
      compute_conduction_diagnostics_impl(state, cfg, d_kappa_eff, d_rho_cv_e, eos_ctx);
  (void)diag;

  std::vector<double> A_eff;
  std::vector<double> gamma_eff;
  compute_effective_material_properties(cfg, state, n_cells, A_eff, gamma_eff);
  (void)gamma_eff;
  double* d_A_eff = static_cast<double*>(core::device_scratch_acquire(
      "conduction:snb_probe:d_A_eff",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  cuda_check(cudaMemcpy(d_A_eff, A_eff.data(),
                        static_cast<std::size_t>(n_cells) * sizeof(double),
                        cudaMemcpyHostToDevice),
             "SNB probe: A_eff upload failed");
  std::uint8_t* d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "conduction:snb_probe:d_cell_is_void",
      static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t)));
  cuda_check(cudaMemcpy(d_cell_is_void, state.cell_is_void.data(),
                        static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             "SNB probe: cell_is_void upload failed");

  const int n_groups = cfg.numerics.conduction.snb_n_groups;
  const std::size_t nf = static_cast<std::size_t>(n_cells) + 1;
  const std::size_t ngc =
      static_cast<std::size_t>(n_groups) * static_cast<std::size_t>(n_cells);
  snb::SnbDeviceWork work;
  auto acquire = [&](const char* tag, const std::size_t bytes) {
    return core::device_scratch_acquire(tag, bytes);
  };
  work.edges = static_cast<double*>(acquire(
      "conduction:snb_probe:edges",
      (static_cast<std::size_t>(n_groups) + 1) * sizeof(double)));
  work.lam_tr = static_cast<double*>(acquire("conduction:snb_probe:lam_tr", ngc * sizeof(double)));
  work.lam_abs = static_cast<double*>(acquire("conduction:snb_probe:lam_abs", ngc * sizeof(double)));
  work.lower = static_cast<double*>(acquire("conduction:snb_probe:lower", ngc * sizeof(double)));
  work.diag = static_cast<double*>(acquire("conduction:snb_probe:diag", ngc * sizeof(double)));
  work.upper = static_cast<double*>(acquire("conduction:snb_probe:upper", ngc * sizeof(double)));
  work.rhs = static_cast<double*>(acquire("conduction:snb_probe:rhs", ngc * sizeof(double)));
  work.qsh_face = static_cast<double*>(acquire("conduction:snb_probe:qsh_face", nf * sizeof(double)));
  work.dq_face = static_cast<double*>(acquire("conduction:snb_probe:dq_face", nf * sizeof(double)));
  work.dq_face_prev =
      static_cast<double*>(acquire("conduction:snb_probe:dq_face_prev", nf * sizeof(double)));
  work.theta_face =
      static_cast<double*>(acquire("conduction:snb_probe:theta_face", nf * sizeof(double)));
  work.scalars = static_cast<double*>(acquire("conduction:snb_probe:scalars", 8 * sizeof(double)));
  work.counters = static_cast<int*>(acquire("conduction:snb_probe:counters", 3 * sizeof(int)));

  cuda_check(cudaMemset(work.scalars, 0, 8 * sizeof(double)),
             "SNB probe: scalars init failed");
  cuda_check(cudaMemset(work.dq_face, 0, nf * sizeof(double)),
             "SNB probe: dq init failed");
  cuda_check(cudaMemset(work.dq_face_prev, 0, nf * sizeof(double)),
             "SNB probe: dq prev init failed");
  snb::snb_fill_t_ref(state, d_cell_is_void, n_cells, work.scalars);
  double t_ref_ev = 0.0;
  cuda_check(cudaMemcpy(&t_ref_ev, work.scalars, sizeof(double),
                        cudaMemcpyDeviceToHost),
             "SNB probe: T_ref readback failed");
  if (!(t_ref_ev > 0.0)) {
    return probe;
  }
  const int geom_code = cfg.numerics.conduction.test_planar
                            ? 2
                            : state.mesh.geometry_code;
  const bool face_policy_kirchhoff =
      (cfg.numerics.conduction.face_kappa_policy == "kirchhoff_same_material" &&
       cfg.numerics.conduction.test_kappa <= 0.0);
  const bool use_outer_cap = (cfg.numerics.conduction.test_kappa <= 0.0);
  double resid = 0.0;
  double qsh_scale = 0.0;
  double dq_ratio = 0.0;
  snb::snb_pass(state, cfg, work, d_kappa_eff, d_A_eff, d_cell_is_void, n_cells,
                geom_code, face_policy_kirchhoff, use_outer_cap, t_ref_ev, &resid,
                &qsh_scale, &dq_ratio,
                state.zmom_active ? state.zmom_r2.data() : nullptr);
  probe.q_sh_face.assign(nf, 0.0);
  probe.dq_face.assign(nf, 0.0);
  probe.theta_face.assign(nf, 1.0);
  cuda_check(cudaMemcpy(probe.q_sh_face.data(), work.qsh_face, nf * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "SNB probe: qsh readback failed");
  cuda_check(cudaMemcpy(probe.dq_face.data(), work.dq_face, nf * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "SNB probe: dq readback failed");
  cuda_check(cudaMemcpy(probe.theta_face.data(), work.theta_face, nf * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "SNB probe: theta readback failed");
  probe.picard_iters = 1;
  probe.dq_over_qsh_max = dq_ratio;
  return probe;
}

}  // namespace conduction

namespace snb2d {

Snb2dProbe snb2d_probe(core::State& state, const core::Config& cfg) {
  Snb2dProbe empty;
  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0 || state.mesh.dim != 2) {
    return empty;
  }
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  TENRYU_ASSERT(nr * nz == n_cells, "snb2d_probe requires nr*nz == n_cells");

  double* d_kappa_eff = static_cast<double*>(core::device_scratch_acquire(
      "conduction:snb2d_probe:d_kappa_eff",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  double* d_rho_cv_e = static_cast<double*>(core::device_scratch_acquire(
      "conduction:snb2d_probe:d_rho_cv_e",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  double* d_stencil = static_cast<double*>(core::device_scratch_acquire(
      "conduction:snb2d_probe:d_stencil",
      static_cast<std::size_t>(n_cells) * kershaw::kStencilSize *
          sizeof(double)));
  static_cast<void>(
      compute_conduction_diagnostics_impl(state, cfg, d_kappa_eff, d_rho_cv_e,
                                          nullptr));

  std::uint8_t* d_cell_is_void = static_cast<std::uint8_t*>(
      core::device_scratch_acquire("conduction:snb2d_probe:d_cell_is_void",
                                   static_cast<std::size_t>(n_cells) *
                                       sizeof(std::uint8_t)));
  cuda_check(cudaMemcpy(d_cell_is_void, state.cell_is_void.data(),
                        static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             "snb2d_probe: cudaMemcpy d_cell_is_void failed");

  int* d_fix = static_cast<int*>(core::device_scratch_acquire(
      "conduction:snb2d_probe:d_mmatrix_fix_count", sizeof(int)));
  cuda_check(cudaMemset(d_fix, 0, sizeof(int)), "snb2d_probe: fix reset failed");
  const int bc_r_lo = map_hydro_bc_to_kershaw("axis");
  const int bc_r_hi = map_hydro_bc_to_kershaw(cfg.numerics.hydro.boundary_2d.r_outer);
  const int bc_z_lo = map_hydro_bc_to_kershaw(cfg.numerics.hydro.boundary_2d.z_bottom);
  const int bc_z_hi = map_hydro_bc_to_kershaw(cfg.numerics.hydro.boundary_2d.z_top);
  kershaw::launch_kershaw_stencil_build(d_stencil, d_kappa_eff,
                                        state.x_r.data(), state.x_z.data(),
                                        nr, nz, bc_r_lo, bc_r_hi, bc_z_lo,
                                        bc_z_hi, true, d_fix);
  cuda_check(cudaGetLastError(), "snb2d_probe: stencil build launch failed");

  std::vector<double> A_eff;
  std::vector<double> gamma_eff;
  compute_effective_material_properties(cfg, state, n_cells, A_eff, gamma_eff);
  (void)gamma_eff;
  double* d_A_eff = static_cast<double*>(core::device_scratch_acquire(
      "conduction:snb2d_probe:d_A_eff",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  cuda_check(cudaMemcpy(d_A_eff, A_eff.data(),
                        static_cast<std::size_t>(n_cells) * sizeof(double),
                        cudaMemcpyHostToDevice),
             "snb2d_probe: cudaMemcpy d_A_eff failed");

  return snb2d_probe_pass(state, cfg, d_stencil, d_kappa_eff, d_rho_cv_e,
                          d_cell_is_void, d_A_eff, nr, nz);
}

}  // namespace snb2d

}  // namespace tenryu::hydro
