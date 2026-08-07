#include "radiation/ddmc.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "core/error.hpp"
#include "core/rng/philox_cpu.hpp"
#include "materials/opacity.cuh"
#include "radiation/groups.cuh"
#include "radiation/particle_pool.cuh"
#include "radiation/planck_table.cuh"

namespace tenryu::radiation {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

constexpr double kTwoPi = 6.283185307179586476925286766559005768;
constexpr int kMaxRejectSamples = 64;

inline int node_index_2d_rz(const int i, const int j, const int nz) {
  return i * (nz + 1) + j;
}

template <typename T>
void copy_device_to_host(std::vector<T>& host,
                         const T* device,
                         const int n,
                         const char* message) {
  host.resize(static_cast<std::size_t>(n));
  if (n <= 0) {
    return;
  }
  cuda_check(cudaMemcpy(host.data(),
                        device,
                        sizeof(T) * static_cast<std::size_t>(n),
                        cudaMemcpyDeviceToHost),
             message);
}

template <typename T>
void copy_host_to_device(T* device,
                         const std::vector<T>& host,
                         const char* message) {
  if (host.empty()) {
    return;
  }
  cuda_check(cudaMemcpy(device,
                        host.data(),
                        sizeof(T) * host.size(),
                        cudaMemcpyHostToDevice),
             message);
}

inline void bilinear_map(const double eta,
                         const double zeta,
                         const double r00,
                         const double z00,
                         const double r10,
                         const double z10,
                         const double r11,
                         const double z11,
                         const double r01,
                         const double z01,
                         double* r_out,
                         double* z_out) {
  const double n00 = (1.0 - eta) * (1.0 - zeta);
  const double n10 = eta * (1.0 - zeta);
  const double n11 = eta * zeta;
  const double n01 = (1.0 - eta) * zeta;
  *r_out = n00 * r00 + n10 * r10 + n11 * r11 + n01 * r01;
  *z_out = n00 * z00 + n10 * z10 + n11 * z11 + n01 * z01;
}

void sample_isotropic_direction_1d(core::rng::PhiloxCpu& rng,
                                   double* dir_r,
                                   double* dir_z,
                                   double* dir_phi) {
  const double mu = 2.0 * rng.uniform() - 1.0;
  const double phi = kTwoPi * rng.uniform();
  const double sin_theta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
  *dir_r = mu;
  *dir_z = sin_theta * std::cos(phi);
  *dir_phi = sin_theta * std::sin(phi);
}

void sample_isotropic_direction_2d(core::rng::PhiloxCpu& rng,
                                   double* dir_r,
                                   double* dir_z,
                                   double* dir_phi) {
  const double mu_z = 2.0 * rng.uniform() - 1.0;
  const double phi = kTwoPi * rng.uniform();
  const double sin_theta = std::sqrt(std::max(0.0, 1.0 - mu_z * mu_z));
  *dir_r = sin_theta * std::cos(phi);
  *dir_z = mu_z;
  *dir_phi = sin_theta * std::sin(phi);
}

void resample_demoted_ddmc_to_imc_1d(core::rng::PhiloxCpu& rng,
                                     const std::vector<double>& node_r,
                                     const std::int64_t cell,
                                     double* pos_r,
                                     double* pos_z,
                                     double* dir_r,
                                     double* dir_z,
                                     double* dir_phi) {
  const std::size_t c_us = static_cast<std::size_t>(cell);
  const double r_lo = node_r[c_us];
  const double r_hi = node_r[c_us + 1];
  const double r_lo3 = r_lo * r_lo * r_lo;
  const double r_hi3 = r_hi * r_hi * r_hi;
  const double xi_r = rng.uniform();

  if (r_hi > r_lo) {
    *pos_r = std::max(std::cbrt(r_lo3 + xi_r * (r_hi3 - r_lo3)), 0.0);
  } else {
    *pos_r = std::max(0.5 * (r_lo + r_hi), 0.0);
  }
  *pos_z = 0.0;
  sample_isotropic_direction_1d(rng, dir_r, dir_z, dir_phi);
}

void resample_demoted_ddmc_to_imc_2d(core::rng::PhiloxCpu& rng,
                                     const std::vector<double>& node_r,
                                     const std::vector<double>& node_z,
                                     const int nr,
                                     const int nz,
                                     const std::int64_t cell,
                                     double* pos_r,
                                     double* pos_z,
                                     double* dir_r,
                                     double* dir_z,
                                     double* dir_phi) {
  if (nr <= 0 || nz <= 0 || node_r.size() != node_z.size()) {
    *pos_r = 0.0;
    *pos_z = 0.0;
    sample_isotropic_direction_2d(rng, dir_r, dir_z, dir_phi);
    return;
  }

  const int c = static_cast<int>(cell);
  const int i = c / nz;
  const int j = c - i * nz;

  if (i < 0 || i >= nr || j < 0 || j >= nz) {
    *pos_r = 0.0;
    *pos_z = 0.0;
    sample_isotropic_direction_2d(rng, dir_r, dir_z, dir_phi);
    return;
  }

  const int n00 = node_index_2d_rz(i, j, nz);
  const int n10 = node_index_2d_rz(i + 1, j, nz);
  const int n11 = node_index_2d_rz(i + 1, j + 1, nz);
  const int n01 = node_index_2d_rz(i, j + 1, nz);
  const int n_nodes = static_cast<int>(node_r.size());
  if (n00 < 0 || n10 < 0 || n11 < 0 || n01 < 0 || n00 >= n_nodes || n10 >= n_nodes ||
      n11 >= n_nodes || n01 >= n_nodes) {
    *pos_r = 0.0;
    *pos_z = 0.0;
    sample_isotropic_direction_2d(rng, dir_r, dir_z, dir_phi);
    return;
  }

  const double r00 = node_r[static_cast<std::size_t>(n00)];
  const double r10 = node_r[static_cast<std::size_t>(n10)];
  const double r11 = node_r[static_cast<std::size_t>(n11)];
  const double r01 = node_r[static_cast<std::size_t>(n01)];
  const double z00 = node_z[static_cast<std::size_t>(n00)];
  const double z10 = node_z[static_cast<std::size_t>(n10)];
  const double z11 = node_z[static_cast<std::size_t>(n11)];
  const double z01 = node_z[static_cast<std::size_t>(n01)];
  const double r_max_cell = std::max(std::max(r00, r10), std::max(r11, r01));

  double r_p = 0.25 * (r00 + r10 + r11 + r01);
  double z_p = 0.25 * (z00 + z10 + z11 + z01);
  bool accepted = false;
  for (int n_try = 0; n_try < kMaxRejectSamples; ++n_try) {
    const double xi_eta = rng.uniform();
    const double xi_zeta = rng.uniform();
    const double xi_reject = rng.uniform();

    bilinear_map(xi_eta,
                 xi_zeta,
                 r00,
                 z00,
                 r10,
                 z10,
                 r11,
                 z11,
                 r01,
                 z01,
                 &r_p,
                 &z_p);
    const double w =
        (r_max_cell > 0.0) ? std::clamp(r_p / r_max_cell, 0.0, 1.0) : 1.0;
    if (xi_reject <= w) {
      accepted = true;
      break;
    }
  }
  if (!accepted) {
    r_p = std::max(r_p, 0.0);
  }
  *pos_r = std::max(r_p, 0.0);
  *pos_z = z_p;
  sample_isotropic_direction_2d(rng, dir_r, dir_z, dir_phi);
}

class FrequencyDependentOpacityProvider final : public OpacityProvider {
 public:
  explicit FrequencyDependentOpacityProvider(std::vector<double> group_bounds_eV)
      : opacity_(std::move(group_bounds_eV)) {}

  [[nodiscard]] double sigma_rosseland(int,
                                       const int group,
                                       const double rho,
                                       const double Te_eV) const override {
    return opacity_.sigma_rosseland(group, rho, Te_eV);
  }

 private:
  materials::FrequencyDependentOpacity opacity_;
};

std::vector<double> resolve_group_bounds(const core::Config& cfg, const int n_groups) {
  std::vector<double> bounds = cfg.radiation.group_bounds_eV;
  if (static_cast<int>(bounds.size()) == n_groups + 1) {
    return bounds;
  }

  const std::vector<double> range = resolve_compute_T_range_eV(cfg, false);
  const double t_min = std::max(range[0], 1.0e-3);
  const double t_max = std::max(range[1], t_min * 1.001);
  return Groups::make_log_uniform_bounds(n_groups, t_min, t_max);
}

std::unique_ptr<OpacityProvider> make_ddmc_face_opacity_provider(const core::Config& cfg,
                                                                 const int n_groups) {
  if (n_groups <= 0 || cfg.materials.materials.empty()) {
    return nullptr;
  }

  if (cfg.radiation.ddmc.face_opacity_temperature != "radiative_mean") {
    static bool warned_once = false;
    if (!warned_once) {
      core::log_warning("Radiation.ddmc.face_opacity_temperature='" +
                        cfg.radiation.ddmc.face_opacity_temperature +
                        "' is not supported in M01; using center opacities for DDMC leaks");
      warned_once = true;
    }
    return nullptr;
  }

  const auto& mat = cfg.materials.materials.front();
  if (mat.opacity_model == "freq_dep_marshak") {
    return std::make_unique<FrequencyDependentOpacityProvider>(
        resolve_group_bounds(cfg, n_groups));
  }
  if (mat.opacity_model == "table_nlte" || mat.opacity_model == "tmat") {
    // NLTE path already provides cell-centered sigma_R; keep DDMC face terms on
    // that path instead of incorrectly using kappa_a_constant.
    return nullptr;
  }

  const double kappa_r = std::max(mat.kappa_a_constant, 0.0);
  return std::make_unique<ConstantOpacityProvider>(
      std::vector<double>(static_cast<std::size_t>(n_groups), kappa_r));
}

}  // namespace

NLTEDDMCCellClosureStats apply_true_nlte_ddmc_cell_closure(
    ModeSelector* mode_selector,
    const CellRadiationCoeffs& coeffs,
    const double support_tol,
    const double sigma_s_tol) {
  TENRYU_ASSERT(mode_selector != nullptr,
                "apply_true_nlte_ddmc_cell_closure requires mode_selector");
  TENRYU_ASSERT(coeffs.n_cells == mode_selector->n_cells(),
                "apply_true_nlte_ddmc_cell_closure n_cells mismatch");
  TENRYU_ASSERT(coeffs.n_groups == mode_selector->n_groups(),
                "apply_true_nlte_ddmc_cell_closure n_groups mismatch");

  const std::size_t n_cell_groups =
      static_cast<std::size_t>(coeffs.n_cells) * static_cast<std::size_t>(coeffs.n_groups);
  TENRYU_ASSERT(coeffs.s.size() == n_cell_groups,
                "apply_true_nlte_ddmc_cell_closure s size mismatch");
  TENRYU_ASSERT(coeffs.sigma_s_eff.size() == n_cell_groups,
                "apply_true_nlte_ddmc_cell_closure sigma_s_eff size mismatch");

  NLTEDDMCCellClosureStats out{};
  for (std::int64_t c = 0; c < mode_selector->n_cells(); ++c) {
    const std::size_t base =
        static_cast<std::size_t>(c) * static_cast<std::size_t>(mode_selector->n_groups());
    bool has_ddmc = false;
    bool has_ddmc_scatter = false;
    bool needs_imc_target = false;
    for (int g = 0; g < mode_selector->n_groups(); ++g) {
      const std::size_t idx = base + static_cast<std::size_t>(g);
      if (mode_selector->get_mode(c, g) == TransportMode::DDMC) {
        has_ddmc = true;
        has_ddmc_scatter = has_ddmc_scatter || (coeffs.sigma_s_eff[idx] > sigma_s_tol);
      } else if (coeffs.s[idx] > support_tol) {
        needs_imc_target = true;
      }
    }
    if (!(has_ddmc && has_ddmc_scatter && needs_imc_target)) {
      continue;
    }
    ++out.cells_forced_imc;
    for (int g = 0; g < mode_selector->n_groups(); ++g) {
      if (mode_selector->get_mode(c, g) == TransportMode::DDMC) {
        mode_selector->force_imc(c, g);
        ++out.ddmc_groups_forced_imc;
      }
    }
  }
  return out;
}

DDMCPreparation prepare_ddmc_step(PhotonPool& pool,
                                  const core::Config& cfg,
                                  const std::uint64_t step_number,
                                  const double dt,
                                  const int mesh_dim,
                                  const int n_cells,
                                  const int n_groups,
                                  const std::vector<double>& node_r,
                                  const std::vector<double>& node_z,
                                  const int nr,
                                  const int nz,
                                  const std::vector<double>& cell_vol,
                                  const std::vector<double>& rho,
                                  const std::vector<double>& Te,
                                  const std::vector<double>& sigma_R,
                                  const std::vector<double>& sigma_a,
                                  const std::vector<double>& sigma_a_eff,
                                  const std::vector<double>& fleck_f,
                                  const CellRadiationCoeffs* coeffs,
                                  const DDMCBoundaryType bc_inner,
                                  const DDMCBoundaryType bc_outer,
                                  const DDMCBoundaryType bc_bottom_z,
                                  const DDMCBoundaryType bc_top_z,
                                  HysteresisState* hysteresis,
                                  const std::vector<std::uint8_t>* force_imc_cells) {
  DDMCPreparation out{};
  if (hysteresis != nullptr) {
    hysteresis->switches_imc_to_ddmc = 0;
    hysteresis->switches_ddmc_to_imc = 0;
  }

  if (!cfg.radiation.ddmc.enabled) {
    return out;
  }
  if (mesh_dim != 1 && mesh_dim != 2) {
    static bool warned_non_1d = false;
    if (!warned_non_1d) {
      core::log_warning("DDMC is supported only for mesh_dim=1 or 2 (mesh_dim=" +
                        std::to_string(mesh_dim) +
                        "); falling back to pure IMC");
      warned_non_1d = true;
    }
    return out;
  }

  // Stencil support in this CPU DDMC preparation path:
  // - 1D supports ddmc.leak_stencil in {"4", "9_kershaw"}.
  // - 2D_RZ supports ddmc.leak_stencil="4" only.
  // Non-"4" 2D_RZ DDMC is GPU-only; CPU-only runs should use leak_stencil="4"
  // or run pure IMC for 2D geometry.
  const std::string& leak_stencil = cfg.radiation.ddmc.leak_stencil;
  const bool leak_stencil_ok_1d = (leak_stencil == "4" || leak_stencil == "9_kershaw");
  const bool leak_stencil_ok_2d = (leak_stencil == "4");
  if ((mesh_dim == 1 && !leak_stencil_ok_1d) || (mesh_dim == 2 && !leak_stencil_ok_2d)) {
    static bool warned_once = false;
    if (!warned_once) {
      if (mesh_dim == 2) {
        core::log_warning(
            "DDMC 2D_RZ CPU implementation supports ddmc.leak_stencil='4' only; "
            "falling back to pure IMC");
      } else {
        core::log_warning(
            "DDMC 1D CPU implementation supports ddmc.leak_stencil='4' or "
            "'9_kershaw' only; falling back to pure IMC");
      }
      warned_once = true;
    }
    return out;
  }

  ModeSelectorConfig selector_cfg;
  selector_cfg.tau_ddmc = cfg.radiation.ddmc.tau_ddmc;
  selector_cfg.tau_rw = cfg.radiation.ddmc.tau_rw;
  selector_cfg.omega_ddmc = cfg.radiation.ddmc.omega_ddmc;
  if (cfg.radiation.ddmc.implicit_diffusion) {
    selector_cfg.tau_ddmc = std::min(selector_cfg.tau_ddmc, 1.0);
    selector_cfg.omega_ddmc = 0.0;
  }
  selector_cfg.tau_ddmc_off = cfg.radiation.ddmc.tau_ddmc_off;
  selector_cfg.omega_ddmc_off = cfg.radiation.ddmc.omega_ddmc_off;
  selector_cfg.mode_hold = cfg.radiation.ddmc.mode_hold;
  selector_cfg.rate_max = cfg.radiation.ddmc.rate_max;
  selector_cfg.emissivity_preserving = cfg.radiation.ddmc.emissivity_preserving;
  selector_cfg.sigma_floor = cfg.numerics.safety.opacity_floor;
  const double alpha_imc = (cfg.radiation.imc.alpha > 0.0) ? cfg.radiation.imc.alpha : 1.0;

  ModeSelector mode_selector(n_cells, n_groups, selector_cfg);
  if (mesh_dim == 2) {
    mode_selector.compute_modes_2d_rz(node_r,
                                      node_z,
                                      nr,
                                      nz,
                                      sigma_R,
                                      fleck_f,
                                      sigma_a,
                                      {},
                                      coeffs,
                                      dt,
                                      alpha_imc);
  } else {
    mode_selector.compute_modes(node_r,
                                sigma_R,
                                fleck_f,
                                sigma_a,
                                {},
                                coeffs,
                                dt,
                                alpha_imc);
  }
  if (hysteresis != nullptr && hysteresis->prev_mode != nullptr &&
      hysteresis->prev_tau != nullptr && hysteresis->hold_count != nullptr) {
    const auto result = mode_selector.apply_hysteresis(*hysteresis->prev_mode,
                                                       *hysteresis->prev_tau,
                                                       *hysteresis->hold_count);
    hysteresis->switches_imc_to_ddmc = result.switches_imc_to_ddmc;
    hysteresis->switches_ddmc_to_imc = result.switches_ddmc_to_imc;
  }
  if (force_imc_cells != nullptr) {
    TENRYU_ASSERT(static_cast<int>(force_imc_cells->size()) == n_cells,
                  "prepare_ddmc_step force_imc_cells size mismatch");
    for (std::int64_t c = 0; c < n_cells; ++c) {
      if ((*force_imc_cells)[static_cast<std::size_t>(c)] == 0U) {
        continue;
      }
      for (int g = 0; g < n_groups; ++g) {
        mode_selector.force_imc(c, g);
      }
    }
  }

  const std::int64_t selected_ddmc_count = mode_selector.count_ddmc();
  const std::int64_t selected_rw_count = mode_selector.count_rw();
  const bool selected_has_ddmc_cells = (selected_ddmc_count > 0);
  const bool selected_has_rw_cells = (selected_rw_count > 0);
  const int n_particles = pool.n_alive;
  std::vector<std::uint8_t> mode;
  std::vector<std::uint8_t> alive;
  bool has_alive_accel_particles = false;
  bool probed_alive_accel_particles = false;
  if (!(selected_has_ddmc_cells || selected_has_rw_cells) && n_particles > 0) {
    copy_device_to_host(mode, pool.mode, n_particles, "ddmc prep D2H mode probe failed");
    copy_device_to_host(alive, pool.alive, n_particles, "ddmc prep D2H alive probe failed");
    probed_alive_accel_particles = true;
    for (int p = 0; p < n_particles; ++p) {
      const std::size_t p_us = static_cast<std::size_t>(p);
      if (alive[p_us] == kAlive &&
          (mode[p_us] == kModeDDMC || mode[p_us] == kModeRW)) {
        has_alive_accel_particles = true;
        break;
      }
    }
  }

  DDMCCoefficients coefficients;
  MMatrixDiagnostics mmatrix{};
  if (selected_has_ddmc_cells || selected_has_rw_cells) {
    const auto opacity_provider = make_ddmc_face_opacity_provider(cfg, n_groups);
    coefficients = DDMCCoefficients(n_cells,
                                    n_groups,
                                    cfg.numerics.safety.opacity_floor,
                                    cfg.numerics.safety.opacity_cap);
    if (mesh_dim == 2) {
      coefficients.compute_2d(node_r,
                              node_z,
                              nr,
                              nz,
                              cell_vol,
                              rho,
                              Te,
                              sigma_R,
                              mode_selector,
                              bc_inner,
                              bc_outer,
                              bc_bottom_z,
                              bc_top_z,
                              cfg.radiation.ddmc.m_matrix_check,
                              opacity_provider.get());
    } else {
      coefficients.compute_1d(node_r,
                              rho,
                              Te,
                              sigma_R,
                              mode_selector,
                              bc_inner,
                              bc_outer,
                              cfg.radiation.ddmc.m_matrix_check,
                              opacity_provider.get());
    }

    if (cfg.radiation.ddmc.m_matrix_check) {
      mmatrix = check_mmatrix_condition(coefficients, mode_selector, sigma_a_eff);
      if (mmatrix.total_violations > 0) {
        if (mesh_dim == 2) {
          coefficients.compute_2d(node_r,
                                  node_z,
                                  nr,
                                  nz,
                                  cell_vol,
                                  rho,
                                  Te,
                                  sigma_R,
                                  mode_selector,
                                  bc_inner,
                                  bc_outer,
                                  bc_bottom_z,
                                  bc_top_z,
                                  cfg.radiation.ddmc.m_matrix_check,
                                  opacity_provider.get());
        } else {
          coefficients.compute_1d(node_r,
                                  rho,
                                  Te,
                                  sigma_R,
                                  mode_selector,
                                  bc_inner,
                                  bc_outer,
                                  cfg.radiation.ddmc.m_matrix_check,
                                  opacity_provider.get());
        }
      }
    }
  }

  const bool has_ddmc_cells = (mode_selector.count_ddmc() > 0);
  const bool has_rw_cells = (mode_selector.count_rw() > 0);
  if (!(has_ddmc_cells || has_rw_cells) && !probed_alive_accel_particles &&
      n_particles > 0) {
    copy_device_to_host(mode, pool.mode, n_particles, "ddmc prep D2H mode probe failed");
    copy_device_to_host(alive, pool.alive, n_particles, "ddmc prep D2H alive probe failed");
    probed_alive_accel_particles = true;
    for (int p = 0; p < n_particles; ++p) {
      const std::size_t p_us = static_cast<std::size_t>(p);
      if (alive[p_us] == kAlive &&
          (mode[p_us] == kModeDDMC || mode[p_us] == kModeRW)) {
        has_alive_accel_particles = true;
        break;
      }
    }
  }

  if (!(has_ddmc_cells || has_rw_cells) && !has_alive_accel_particles) {
    out.mode_selector = std::move(mode_selector);
    out.coefficients = std::move(coefficients);
    out.mmatrix = mmatrix;
    out.ddmc_mode_count = 0;
    out.rw_mode_count = 0;
    out.imc_mode_count = out.mode_selector.count_imc();
    out.omega_below_threshold = out.mode_selector.count_omega_below_threshold();
    out.active = true;
    return out;
  }

  std::vector<double> pos_r;
  std::vector<double> pos_z;
  std::vector<double> dir_r;
  std::vector<double> dir_z;
  std::vector<double> dir_phi;
  std::vector<double> time_remain;
  std::vector<std::uint64_t> global_id;
  std::vector<std::uint32_t> rng_counter;
  std::vector<std::int32_t> cell_id;
  std::vector<std::uint16_t> group_id;

  copy_device_to_host(pos_r, pool.pos_r, n_particles, "ddmc prep D2H pos_r failed");
  copy_device_to_host(pos_z, pool.pos_z, n_particles, "ddmc prep D2H pos_z failed");
  copy_device_to_host(dir_r, pool.dir_r, n_particles, "ddmc prep D2H dir_r failed");
  copy_device_to_host(dir_z, pool.dir_z, n_particles, "ddmc prep D2H dir_z failed");
  copy_device_to_host(dir_phi,
                      pool.dir_phi,
                      n_particles,
                      "ddmc prep D2H dir_phi failed");
  copy_device_to_host(time_remain,
                      pool.time_remain,
                      n_particles,
                      "ddmc prep D2H time_remain failed");
  copy_device_to_host(global_id,
                      pool.global_id,
                      n_particles,
                      "ddmc prep D2H global_id failed");
  copy_device_to_host(rng_counter,
                      pool.rng_counter,
                      n_particles,
                      "ddmc prep D2H rng_counter failed");
  copy_device_to_host(cell_id,
                      pool.cell_id,
                      n_particles,
                      "ddmc prep D2H cell_id failed");
  copy_device_to_host(group_id,
                      pool.group_id,
                      n_particles,
                      "ddmc prep D2H group_id failed");
  if (mode.empty() && n_particles > 0) {
    copy_device_to_host(mode, pool.mode, n_particles, "ddmc prep D2H mode failed");
  }
  if (alive.empty() && n_particles > 0) {
    copy_device_to_host(alive, pool.alive, n_particles, "ddmc prep D2H alive failed");
  }

  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (int p = 0; p < n_particles; ++p) {
    const std::size_t p_us = static_cast<std::size_t>(p);
    if (alive[p_us] != kAlive) {
      continue;
    }

    const std::int64_t c = static_cast<std::int64_t>(cell_id[p_us]);
    const int g = static_cast<int>(group_id[p_us]);
    const std::uint8_t old_mode = mode[p_us];
    if (c < 0 || c >= n_cells || g < 0 || g >= n_groups) {
      mode[p_us] = kModeIMC;
      time_remain[p_us] = dt;
      continue;
    }

    const TransportMode target_mode = mode_selector.get_mode(c, g);
    mode[p_us] = (target_mode == TransportMode::DDMC)
                     ? kModeDDMC
                     : ((target_mode == TransportMode::RW) ? kModeRW : kModeIMC);
    if (target_mode == TransportMode::DDMC) {
      pos_r[p_us] = nan;
      pos_z[p_us] = nan;
      dir_r[p_us] = nan;
      dir_z[p_us] = nan;
      dir_phi[p_us] = nan;
      time_remain[p_us] = 0.0;
      continue;
    }

    if (old_mode == kModeDDMC) {
      core::rng::PhiloxCpu rng(global_id[p_us], cfg.main.seed, step_number, rng_counter[p_us]);
      if (mesh_dim == 2) {
        resample_demoted_ddmc_to_imc_2d(rng,
                                        node_r,
                                        node_z,
                                        nr,
                                        nz,
                                        c,
                                        &pos_r[p_us],
                                        &pos_z[p_us],
                                        &dir_r[p_us],
                                        &dir_z[p_us],
                                        &dir_phi[p_us]);
      } else {
        resample_demoted_ddmc_to_imc_1d(rng,
                                        node_r,
                                        c,
                                        &pos_r[p_us],
                                        &pos_z[p_us],
                                        &dir_r[p_us],
                                        &dir_z[p_us],
                                        &dir_phi[p_us]);
      }
      rng_counter[p_us] = rng.counter();
      time_remain[p_us] = dt;
    } else if (target_mode == TransportMode::RW) {
      time_remain[p_us] = dt;
    }
  }

  copy_host_to_device(pool.pos_r, pos_r, "ddmc prep H2D pos_r failed");
  copy_host_to_device(pool.pos_z, pos_z, "ddmc prep H2D pos_z failed");
  copy_host_to_device(pool.dir_r, dir_r, "ddmc prep H2D dir_r failed");
  copy_host_to_device(pool.dir_z, dir_z, "ddmc prep H2D dir_z failed");
  copy_host_to_device(pool.dir_phi, dir_phi, "ddmc prep H2D dir_phi failed");
  copy_host_to_device(pool.time_remain, time_remain, "ddmc prep H2D time_remain failed");
  copy_host_to_device(pool.rng_counter, rng_counter, "ddmc prep H2D rng_counter failed");
  copy_host_to_device(pool.mode, mode, "ddmc prep H2D mode failed");

  out.mode_selector = std::move(mode_selector);
  out.coefficients = std::move(coefficients);
  out.mmatrix = mmatrix;
  out.ddmc_mode_count = out.mode_selector.count_ddmc();
  out.rw_mode_count = out.mode_selector.count_rw();
  out.imc_mode_count = out.mode_selector.count_imc();
  out.omega_below_threshold = out.mode_selector.count_omega_below_threshold();
  out.active = true;
  return out;
}

DDMCDiagnostics run_ddmc_transport_step(PhotonPool& pool,
                                        const int start_index,
                                        const int end_index,
                                        const core::Config& cfg,
                                        const double dt,
                                        const std::uint64_t step_number,
                                        const int mesh_dim,
                                        const std::vector<double>& node_r,
                                        const std::vector<double>& node_z,
                                        const int nr,
                                        const int nz,
                                        const ModeSelector& mode_selector,
                                        const DDMCCoefficients& coefficients,
                                        const std::vector<double>& sigma_a_eff,
                                        const std::vector<double>* sigma_s_eff,
                                        const std::vector<double>* eta_cdf,
                                        std::vector<double>& rad_dep,
                                        std::vector<double>& rad_E_tally,
                                        std::vector<double>& E_escape,
                                        double* E_numerical_loss,
                                        DDMCMomentumEstimator* momentum) {
  const bool interface_exit_half_isotropic =
      (cfg.radiation.ddmc.interface_exit_distribution == "half_isotropic");
  if (mesh_dim == 2) {
    DDMCTransport2D transport(mode_selector.n_cells(),
                              mode_selector.n_groups(),
                              nr,
                              nz,
                              interface_exit_half_isotropic);
    transport.process_range(pool,
                            start_index,
                            end_index,
                            dt,
                            step_number,
                            cfg.main.seed,
                            node_r,
                            node_z,
                            mode_selector,
                            coefficients,
                            sigma_a_eff,
                            sigma_s_eff,
                            eta_cdf,
                            rad_dep,
                            rad_E_tally,
                            E_escape,
                            E_numerical_loss,
                            momentum);
    return transport.diagnostics();
  }

  DDMCTransport1D transport(mode_selector.n_cells(),
                            mode_selector.n_groups(),
                            interface_exit_half_isotropic);
  transport.process_range(pool,
                          start_index,
                          end_index,
                          dt,
                          step_number,
                          cfg.main.seed,
                          node_r,
                          mode_selector,
                          coefficients,
                          sigma_a_eff,
                          sigma_s_eff,
                          eta_cdf,
                          rad_dep,
                          rad_E_tally,
                          E_escape,
                          E_numerical_loss,
                          momentum);
  return transport.diagnostics();
}

}  // namespace tenryu::radiation
