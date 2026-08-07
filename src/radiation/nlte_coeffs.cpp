#include "radiation/nlte_coeffs.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>
#include <vector>

#include "core/constants.hpp"
#include "core/error.hpp"

namespace tenryu::radiation {
namespace {

constexpr double kMaxTemperatureForT4 = 1.0e6;
constexpr double kEtaFloor = 1.0e-300;
constexpr double kKappaFloor = 1.0e-100;
constexpr double kDistributionTol = 1.0e-10;
constexpr double kSigmaDiagEps = 1.0e-30;
constexpr double kFleckTol = 1.0e-12;
constexpr double kSpectrumBlendThresholdRatio = 1.0e-6;
constexpr double kSpectrumBlendFloorFraction = 0.1;
constexpr double kCorrectedFleckDeltaRel = 1.0e-3;
constexpr double kCorrectedFleckSigmaFloor = 1.0e-30;
constexpr double kCorrectedFleckOnePlusXiMin = 0.1;
constexpr double kCorrectedFleckOnePlusXiMax = 10.0;

inline double clamped_temperature_for_t4(const double temperature_eV) {
  if (!std::isfinite(temperature_eV) || temperature_eV <= 0.0) {
    return 0.0;
  }
  return std::min(temperature_eV, kMaxTemperatureForT4);
}

inline double safe_temperature_pow4(const double temperature_eV) {
  const double t = clamped_temperature_for_t4(temperature_eV);
  return t * t * t * t;
}

double planck_fraction_safe(const PlanckTable& planck,
                            const int g,
                            const double T_eV,
                            const int n_groups) {
  if (n_groups <= 1) {
    return 1.0;
  }
  return std::max(planck.interpolate_b_host(g, T_eV), 0.0);
}

void write_uniform_cdf(double* cdf, const int n_groups) {
  TENRYU_ASSERT(cdf != nullptr, "write_uniform_cdf requires cdf");
  for (int g = 0; g < n_groups; ++g) {
    cdf[g] = static_cast<double>(g + 1) / static_cast<double>(std::max(n_groups, 1));
  }
}

void normalize_nonnegative_distribution(double* values, const int n_groups) {
  TENRYU_ASSERT(values != nullptr, "normalize_nonnegative_distribution requires values");
  if (n_groups <= 0) {
    return;
  }
  if (n_groups == 1) {
    values[0] = 1.0;
    return;
  }

  double sum = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    values[g] = std::max(values[g], 0.0);
    sum += values[g];
  }
  if (!(sum > 0.0) || !std::isfinite(sum)) {
    const double uniform = 1.0 / static_cast<double>(n_groups);
    for (int g = 0; g < n_groups; ++g) {
      values[g] = uniform;
    }
    return;
  }
  const double inv_sum = 1.0 / sum;
  for (int g = 0; g < n_groups; ++g) {
    values[g] *= inv_sum;
  }
}

double safe_log_kappa(const double kappa) {
  return std::log(std::max(kappa, kKappaFloor));
}

double edge_log_slope_temperature_upper(
    const materials::IonmixOpacityData& table,
    const std::vector<double>& kappa_table,
    const int group,
    const double ni_cm3) {
  if (table.temps_eV.size() < 2) {
    return 0.0;
  }
  const double T_hi = table.temps_eV.back();
  const double T_lo = table.temps_eV[table.temps_eV.size() - 2];
  if (!(T_hi > T_lo)) {
    return 0.0;
  }
  const double k_hi = table.interpolate_kappa(kappa_table, group, ni_cm3, T_hi);
  const double k_lo = table.interpolate_kappa(kappa_table, group, ni_cm3, T_lo);
  return (safe_log_kappa(k_hi) - safe_log_kappa(k_lo)) / (std::log(T_hi) - std::log(T_lo));
}

double edge_log_slope_density_lower(const materials::IonmixOpacityData& table,
                                    const std::vector<double>& kappa_table,
                                    const int group,
                                    const double T_eV) {
  if (table.numdens_cm3.size() < 2) {
    return 0.0;
  }
  const double ni_lo = table.numdens_cm3.front();
  const double ni_hi = table.numdens_cm3[1];
  if (!(ni_hi > ni_lo)) {
    return 0.0;
  }
  const double k_lo = table.interpolate_kappa(kappa_table, group, ni_lo, T_eV);
  const double k_hi = table.interpolate_kappa(kappa_table, group, ni_hi, T_eV);
  return (safe_log_kappa(k_hi) - safe_log_kappa(k_lo)) /
         (std::log(ni_hi) - std::log(ni_lo));
}

double interpolate_kappa_with_smooth_edges(
    const materials::IonmixOpacityData& table,
    const std::vector<double>& kappa_table,
    const int group,
    const double ni_cm3,
    const double T_eV) {
  const double ni_min = table.numdens_cm3.front();
  const double ni_max = table.numdens_cm3.back();
  const double T_min = table.temps_eV.front();
  const double T_max = table.temps_eV.back();

  double ni_eval = ni_cm3;
  if (!std::isfinite(ni_eval) || !(ni_eval > 0.0)) {
    ni_eval = ni_min;
  }
  double T_eval = T_eV;
  if (!std::isfinite(T_eval) || !(T_eval > 0.0)) {
    T_eval = T_min;
  }

  const bool rarefied_extrapolation = ni_eval < ni_min;
  const bool hot_extrapolation = T_eval > T_max;
  const double ni_edge = std::clamp(ni_eval, ni_min, ni_max);
  const double T_edge = std::clamp(T_eval, T_min, T_max);

  double log_kappa =
      safe_log_kappa(table.interpolate_kappa(kappa_table, group, ni_edge, T_edge));
  if (rarefied_extrapolation) {
    const double slope_ni =
        edge_log_slope_density_lower(table, kappa_table, group, T_edge);
    log_kappa += slope_ni * (std::log(ni_eval) - std::log(ni_min));
  }
  if (hot_extrapolation) {
    const double slope_T =
        edge_log_slope_temperature_upper(table, kappa_table, group, ni_edge);
    log_kappa += slope_T * (std::log(T_eval) - std::log(T_max));
  }

  const double log_kappa_min = std::log(kKappaFloor);
  const double log_kappa_max = std::log(std::numeric_limits<double>::max());
  log_kappa = std::clamp(log_kappa, log_kappa_min, log_kappa_max);
  return std::exp(log_kappa);
}

// Compute Planck-emission-weighted mean opacity at a given temperature.
// Used for finite-difference derivative in corrected Fleck factor.
double compute_sigma_p_em_at_T(const materials::IonmixOpacityData& table,
                               const PlanckTable& planck,
                               const double rho,
                               const double ni,
                               const double T_eV,
                               const int n_groups,
                               const double sigma_cap) {
  if (!(rho > 0.0) || n_groups <= 0) {
    return 0.0;
  }

  double sigma_p_em = 0.0;
  double sigma_pe_sum = 0.0;
  double b_sum = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    const double b_g = planck_fraction_safe(planck, g, T_eV, n_groups);
    double sigma_pe =
        rho * interpolate_kappa_with_smooth_edges(table, table.kappa_PE, g, ni, T_eV);
    if (!std::isfinite(sigma_pe) || sigma_pe < 0.0) {
      sigma_pe = 0.0;
    }
    if (sigma_cap > 0.0) {
      sigma_pe = std::min(sigma_pe, sigma_cap);
    }
    sigma_p_em += sigma_pe * b_g;
    sigma_pe_sum += sigma_pe;
    b_sum += b_g;
  }
  if (!(b_sum > 0.0) || !std::isfinite(b_sum)) {
    return sigma_pe_sum / static_cast<double>(n_groups);
  }
  return sigma_p_em / b_sum;
}

double compute_cv_e_host(const core::Config&,
                         const core::Config::MaterialsConfig::MatDef& mat,
                         const double rho_c,
                         const double zbar_c,
                         const bool has_state_cv_e,
                         const double state_cv_e) {
  if (mat.cv_e_override > 0.0) {
    return std::max(mat.cv_e_override, 1.0e-30);
  }
  if (has_state_cv_e && state_cv_e > 0.0) {
    return std::max(rho_c, 0.0) * state_cv_e;
  }

  const double A = std::max(mat.A, 1.0e-12);
  const double gm1 = std::max(mat.ideal_gas_gamma - 1.0, 1.0e-12);
  const double cv_mass_e = zbar_c * core::constants::eV_to_erg /
                           (A * core::constants::proton_mass * gm1);
  return std::max(rho_c, 0.0) * std::max(cv_mass_e, 0.0);
}

double compute_beta_host(const core::Config& cfg,
                         const core::Config::MaterialsConfig::MatDef& mat,
                         const double rho_c,
                         const double Te_c,
                         const double zbar_c,
                         const bool has_state_cv_e,
                         const double state_cv_e) {
  const double Cv_e = std::max(
      compute_cv_e_host(cfg, mat, rho_c, zbar_c, has_state_cv_e, state_cv_e), 1.0e-30);
  double beta = 0.0;
  if (cfg.radiation.imc.linearized_planck && mat.cv_e_override > 0.0) {
    beta = 1.0;
  } else {
    const double T = std::max(Te_c, 0.0);
    beta = 4.0 * core::constants::a_eV * T * T * T / Cv_e;
  }
  if (mat.cv_e_override <= 0.0 && !has_state_cv_e) {
    beta = std::min(beta, 1.0);
  }
  return std::max(beta, 0.0);
}

struct SeparateEmissivityScalarsView {
  double sigma_p_abs = 0.0;
  double sigma_p_em = 0.0;
  double gamma_diag = 0.0;
  double f = 1.0;
};

SeparateEmissivityScalarsView compute_separate_emissivity_scalars_impl(
    const double* alpha_g,
    const double* eta_g,
    const double* b_g,
    const int n_groups,
    const double temperature_eV,
    const double beta,
    const double dt,
    const double alpha_fleck,
    const double temperature_floor_eV,
    double* s_out,
    double* cdf_out) {
  TENRYU_ASSERT(alpha_g != nullptr, "compute_separate_emissivity_scalars_impl alpha_g null");
  TENRYU_ASSERT(eta_g != nullptr, "compute_separate_emissivity_scalars_impl eta_g null");
  TENRYU_ASSERT(b_g != nullptr, "compute_separate_emissivity_scalars_impl b_g null");
  TENRYU_ASSERT(s_out != nullptr, "compute_separate_emissivity_scalars_impl s_out null");
  TENRYU_ASSERT(cdf_out != nullptr, "compute_separate_emissivity_scalars_impl cdf_out null");

  SeparateEmissivityScalarsView out{};
  const double alpha_safe = (alpha_fleck > 0.0) ? alpha_fleck : 1.0;
  const double dt_safe = std::max(dt, 0.0);

  double eta_sum = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    out.sigma_p_abs += std::max(b_g[g], 0.0) * std::max(alpha_g[g], 0.0);
    eta_sum += std::max(eta_g[g], 0.0);
  }

  const double T4 = safe_temperature_pow4(temperature_eV);
  if (!(temperature_eV > temperature_floor_eV) || !(eta_sum > kEtaFloor) || !(T4 > 0.0)) {
    for (int g = 0; g < n_groups; ++g) {
      s_out[g] = 0.0;
    }
    write_uniform_cdf(cdf_out, n_groups);
    return out;
  }

  out.sigma_p_em =
      eta_sum / (core::constants::a_eV * core::constants::c_light * T4);
  out.gamma_diag = out.sigma_p_em / std::max(out.sigma_p_abs, kSigmaDiagEps);

  double s_sum = 0.0;
  double running = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    s_out[g] = std::max(eta_g[g], 0.0) / eta_sum;
    s_sum += s_out[g];
    running += s_out[g];
    cdf_out[g] = std::clamp(running, 0.0, 1.0);
  }
  cdf_out[n_groups - 1] = 1.0;
  TENRYU_ASSERT(std::abs(s_sum - 1.0) <= kDistributionTol,
                "compute_separate_emissivity_scalars_impl s_g normalization failure");

  out.f = 1.0 /
          (1.0 + alpha_safe * beta * core::constants::c_light * dt_safe * out.sigma_p_em);
  TENRYU_ASSERT(std::isfinite(out.f),
                "compute_separate_emissivity_scalars_impl produced non-finite Fleck factor");
  TENRYU_ASSERT(out.f >= -kFleckTol && out.f <= 1.0 + kFleckTol,
                "compute_separate_emissivity_scalars_impl Fleck factor left [0,1]");
  out.f = std::clamp(out.f, 0.0, 1.0);
  return out;
}

}  // namespace

SeparateEmissivityScalars compute_separate_emissivity_scalars(
    const std::vector<double>& alpha_g,
    const std::vector<double>& eta_g,
    const std::vector<double>& b_g,
    const double temperature_eV,
    const double beta,
    const double dt,
    const double alpha_fleck,
    const double temperature_floor_eV) {
  TENRYU_ASSERT(alpha_g.size() == eta_g.size(),
                "compute_separate_emissivity_scalars alpha/eta size mismatch");
  TENRYU_ASSERT(alpha_g.size() == b_g.size(),
                "compute_separate_emissivity_scalars alpha/b size mismatch");

  SeparateEmissivityScalars out{};
  out.b = b_g;
  out.s.assign(alpha_g.size(), 0.0);
  out.cdf.assign(alpha_g.size(), 0.0);
  if (alpha_g.empty()) {
    return out;
  }

  std::vector<double> alpha(alpha_g.size(), 0.0);
  std::vector<double> eta(eta_g.size(), 0.0);
  for (std::size_t i = 0; i < alpha_g.size(); ++i) {
    alpha[i] = std::max(alpha_g[i], 0.0);
    eta[i] = std::max(eta_g[i], 0.0);
  }
  normalize_nonnegative_distribution(out.b.data(), static_cast<int>(out.b.size()));
  const auto scalars = compute_separate_emissivity_scalars_impl(
      alpha.data(),
      eta.data(),
      out.b.data(),
      static_cast<int>(alpha.size()),
      temperature_eV,
      beta,
      dt,
      alpha_fleck,
      temperature_floor_eV,
      out.s.data(),
      out.cdf.data());
  out.sigma_p_abs = scalars.sigma_p_abs;
  out.sigma_p_em = scalars.sigma_p_em;
  out.gamma_diag = scalars.gamma_diag;
  out.f = scalars.f;
  return out;
}

void apply_nlte_transport_linearized_correction(
    const int n_groups,
    const std::vector<double>& eta0,
    const std::vector<double>& alpha0,
    const std::vector<double>& J0,
    const std::vector<double>& deta_dJ,
    const std::vector<double>& dalpha_dJ,
    const std::vector<double>& J,
    std::vector<double>* eta,
    std::vector<double>* alpha) {
  TENRYU_ASSERT(n_groups >= 0, "apply_nlte_transport_linearized_correction invalid n_groups");
  TENRYU_ASSERT(eta != nullptr, "apply_nlte_transport_linearized_correction eta null");
  TENRYU_ASSERT(alpha != nullptr, "apply_nlte_transport_linearized_correction alpha null");
  TENRYU_ASSERT(eta0.size() == static_cast<std::size_t>(n_groups),
                "apply_nlte_transport_linearized_correction eta0 size mismatch");
  TENRYU_ASSERT(alpha0.size() == static_cast<std::size_t>(n_groups),
                "apply_nlte_transport_linearized_correction alpha0 size mismatch");
  TENRYU_ASSERT(J0.size() == static_cast<std::size_t>(n_groups),
                "apply_nlte_transport_linearized_correction J0 size mismatch");
  TENRYU_ASSERT(J.size() == static_cast<std::size_t>(n_groups),
                "apply_nlte_transport_linearized_correction J size mismatch");
  TENRYU_ASSERT(deta_dJ.size() == static_cast<std::size_t>(n_groups * n_groups),
                "apply_nlte_transport_linearized_correction deta_dJ size mismatch");
  TENRYU_ASSERT(dalpha_dJ.size() == static_cast<std::size_t>(n_groups * n_groups),
                "apply_nlte_transport_linearized_correction dalpha_dJ size mismatch");

  eta->assign(static_cast<std::size_t>(n_groups), 0.0);
  alpha->assign(static_cast<std::size_t>(n_groups), 0.0);
  for (int g = 0; g < n_groups; ++g) {
    double eta_g = eta0[static_cast<std::size_t>(g)];
    double alpha_g = alpha0[static_cast<std::size_t>(g)];
    for (int i = 0; i < n_groups; ++i) {
      const std::size_t idx =
          static_cast<std::size_t>(g) * static_cast<std::size_t>(n_groups) +
          static_cast<std::size_t>(i);
      const double delta_J = J[static_cast<std::size_t>(i)] - J0[static_cast<std::size_t>(i)];
      eta_g += deta_dJ[idx] * delta_J;
      alpha_g += dalpha_dJ[idx] * delta_J;
    }
    (*eta)[static_cast<std::size_t>(g)] = eta_g;
    (*alpha)[static_cast<std::size_t>(g)] = alpha_g;
  }
}

SpectrumReconstructionDiagnostics reconstruct_nlte_group_spectrum(
    const core::State& state,
    const PlanckTable& planck,
    const int n_cells,
    const int n_groups,
    std::vector<double>* J,
    std::vector<std::uint8_t>* used_planck_fallback) {
  TENRYU_ASSERT(J != nullptr, "reconstruct_nlte_group_spectrum J null");
  TENRYU_ASSERT(used_planck_fallback != nullptr,
                "reconstruct_nlte_group_spectrum used_planck_fallback null");
  TENRYU_ASSERT(n_cells >= 0, "reconstruct_nlte_group_spectrum invalid n_cells");
  TENRYU_ASSERT(n_groups >= 1, "reconstruct_nlte_group_spectrum invalid n_groups");
  TENRYU_ASSERT(planck.n_groups() == n_groups,
                "reconstruct_nlte_group_spectrum planck/group mismatch");

  SpectrumReconstructionDiagnostics diag{};
  const std::size_t n_cell_groups =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  J->assign(n_cell_groups, 0.0);
  used_planck_fallback->assign(n_cell_groups, static_cast<std::uint8_t>(0));
  if (n_cells == 0) {
    return diag;
  }

  std::vector<double> host_rad_E(n_cell_groups, 0.0);
  if (state.rad_E.size() == n_cell_groups) {
    state.rad_E.copy_to_host(host_rad_E.data());
  } else if (!state.rad_E.empty()) {
    static int warned_rad_E_size = 0;
    ++warned_rad_E_size;
    if (warned_rad_E_size == 1 || warned_rad_E_size % 100 == 0) {
      core::log_warning("NLTE spectrum reconstruction: state.rad_E size mismatch; "
                        "falling back to zero spectrum");
    }
  }

  std::vector<double> planck_shape(static_cast<std::size_t>(n_groups), 0.0);
  std::vector<double> measured_shape(static_cast<std::size_t>(n_groups), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t base =
        static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups);
    double total_rad_E = 0.0;
    bool used_fallback = false;
    int fallback_groups = 0;
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx = base + static_cast<std::size_t>(g);
      const double J_raw = host_rad_E[idx];
      if (std::isfinite(J_raw) && J_raw > 0.0) {
        (*J)[idx] = J_raw;
        measured_shape[static_cast<std::size_t>(g)] = J_raw;
        total_rad_E += J_raw;
      } else {
        (*J)[idx] = 0.0;
        measured_shape[static_cast<std::size_t>(g)] = 0.0;
        used_fallback = true;
        ++fallback_groups;
        (*used_planck_fallback)[idx] = static_cast<std::uint8_t>(1);
      }
    }

    if (!used_fallback) {
      continue;
    }

    ++diag.fallback_cell_count;
    diag.fallback_group_count += fallback_groups;
    if (!(total_rad_E > 0.0)) {
      continue;
    }

    normalize_nonnegative_distribution(measured_shape.data(), n_groups);
    const double Tr = std::pow(total_rad_E / core::constants::a_eV, 0.25);
    for (int g = 0; g < n_groups; ++g) {
      planck_shape[static_cast<std::size_t>(g)] =
          planck_fraction_safe(planck, g, Tr, n_groups);
    }
    normalize_nonnegative_distribution(planck_shape.data(), n_groups);

    const double mean_group_E =
        total_rad_E / static_cast<double>(std::max(n_groups, 1));
    const double blend_threshold =
        std::max(kSpectrumBlendThresholdRatio * mean_group_E, kEtaFloor);
    const double blend_floor = kSpectrumBlendFloorFraction * blend_threshold;
    double min_group_E = std::numeric_limits<double>::infinity();
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx = base + static_cast<std::size_t>(g);
      min_group_E = std::min(min_group_E, std::max((*J)[idx], 0.0));
    }
    const double blend_weight = std::clamp(
        (min_group_E + blend_floor) / std::max(blend_threshold + blend_floor, kEtaFloor),
        0.0,
        1.0);

    double provisional_sum = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx = base + static_cast<std::size_t>(g);
      const double blended_shape =
          blend_weight * measured_shape[static_cast<std::size_t>(g)] +
          (1.0 - blend_weight) * planck_shape[static_cast<std::size_t>(g)];
      (*J)[idx] = total_rad_E * blended_shape;
      provisional_sum += (*J)[idx];
    }
    if (!(provisional_sum > 0.0) || !std::isfinite(provisional_sum)) {
      continue;
    }
    const double scale = total_rad_E / provisional_sum;
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx = base + static_cast<std::size_t>(g);
      (*J)[idx] *= scale;
    }
  }

  return diag;
}

std::vector<double> compute_nlte_sigma_p_em(const core::State& state,
                                            const core::Config& cfg,
                                            const materials::IonmixOpacityData& table,
                                            const PlanckTable& planck,
                                            const int n_cells,
                                            const int n_groups) {
  TENRYU_ASSERT(n_cells >= 0, "compute_nlte_sigma_p_em requires n_cells >= 0");
  TENRYU_ASSERT(n_groups >= 1, "compute_nlte_sigma_p_em requires n_groups >= 1");
  TENRYU_ASSERT(static_cast<int>(state.rho.size()) == n_cells,
                "compute_nlte_sigma_p_em rho size mismatch");
  TENRYU_ASSERT(static_cast<int>(state.Te.size()) == n_cells,
                "compute_nlte_sigma_p_em Te size mismatch");
  TENRYU_ASSERT(static_cast<int>(state.cell_is_void.size()) == n_cells,
                "compute_nlte_sigma_p_em cell_is_void size mismatch");
  TENRYU_ASSERT(planck.n_groups() == n_groups,
                "compute_nlte_sigma_p_em planck/group size mismatch");
  TENRYU_ASSERT(table.ngroups == n_groups,
                "compute_nlte_sigma_p_em IONMIX/group size mismatch");
  TENRYU_ASSERT(!table.temps_eV.empty(),
                "compute_nlte_sigma_p_em requires non-empty IONMIX temperatures");
  TENRYU_ASSERT(!table.numdens_cm3.empty(),
                "compute_nlte_sigma_p_em requires non-empty IONMIX number densities");

  std::vector<double> sigma_p_em(static_cast<std::size_t>(n_cells), 0.0);
  if (n_cells == 0) {
    return sigma_p_em;
  }

  const int mat_idx = cfg.materials.first_nonvoid_material_index();
  TENRYU_ASSERT(mat_idx >= 0,
                "compute_nlte_sigma_p_em requires at least one non-void material");
  const auto& mat = cfg.materials.materials[static_cast<std::size_t>(mat_idx)];
  TENRYU_ASSERT(mat.opacity_model == "table_nlte" || mat.opacity_model == "tmat",
                "compute_nlte_sigma_p_em requires opacity.model=table_nlte or tmat");

  const double sigma_cap = std::max(cfg.numerics.safety.opacity_cap, 0.0);
  const double T_floor = std::max(cfg.numerics.floors.Te, 0.0);
  const double table_T_min = table.temps_eV.front();
  const double table_ni_min = table.numdens_cm3.front();
  const double A = std::max(mat.A, 1.0e-12);

  std::vector<double> rho(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> Te(static_cast<std::size_t>(n_cells), 0.0);
  state.rho.copy_to_host(rho.data());
  state.Te.copy_to_host(Te.data());

  std::vector<double> b_row(static_cast<std::size_t>(n_groups), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    if (state.cell_is_void[c_us] != 0U) {
      continue;
    }

    const double rho_c = std::max(rho[c_us], 0.0);
    if (!(rho_c > 0.0)) {
      continue;
    }

    double T_eval = Te[c_us];
    if (!std::isfinite(T_eval) || !(T_eval > 0.0)) {
      T_eval = table_T_min;
    }
    if (!(T_eval > T_floor)) {
      continue;
    }

    double ni_c = rho_c / (A * core::constants::proton_mass);
    if (!std::isfinite(ni_c) || !(ni_c > 0.0)) {
      ni_c = table_ni_min;
    }

    for (int g = 0; g < n_groups; ++g) {
      b_row[static_cast<std::size_t>(g)] =
          planck_fraction_safe(planck, g, T_eval, n_groups);
    }
    normalize_nonnegative_distribution(b_row.data(), n_groups);

    double sigma_p_em_c = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      double sigma_pe =
          rho_c * interpolate_kappa_with_smooth_edges(table, table.kappa_PE, g, ni_c, T_eval);
      if (!std::isfinite(sigma_pe) || sigma_pe < 0.0) {
        sigma_pe = 0.0;
      }
      if (sigma_cap > 0.0) {
        sigma_pe = std::min(sigma_pe, sigma_cap);
      }
      sigma_p_em_c += sigma_pe * b_row[static_cast<std::size_t>(g)];
    }
    sigma_p_em[c_us] = sigma_p_em_c;
  }

  return sigma_p_em;
}

CellRadiationCoeffs compute_nlte_coefficients(const core::State& state,
                                              const core::Config& cfg,
                                              const materials::IonmixOpacityData& table,
                                              const PlanckTable& planck,
                                              const int n_cells,
                                              const int n_groups,
                                              const double dt) {
  TENRYU_ASSERT(n_cells >= 0, "compute_nlte_coefficients requires n_cells >= 0");
  TENRYU_ASSERT(n_groups >= 1, "compute_nlte_coefficients requires n_groups >= 1");
  TENRYU_ASSERT(static_cast<int>(state.rho.size()) == n_cells,
                "compute_nlte_coefficients rho size mismatch");
  TENRYU_ASSERT(static_cast<int>(state.Te.size()) == n_cells,
                "compute_nlte_coefficients Te size mismatch");
  TENRYU_ASSERT(static_cast<int>(state.zbar.size()) == n_cells,
                "compute_nlte_coefficients zbar size mismatch");
  TENRYU_ASSERT(static_cast<int>(state.cell_is_void.size()) == n_cells,
                "compute_nlte_coefficients cell_is_void size mismatch");
  TENRYU_ASSERT(planck.n_groups() == n_groups,
                "compute_nlte_coefficients planck/group size mismatch");
  TENRYU_ASSERT(table.ngroups == n_groups,
                "compute_nlte_coefficients IONMIX/group size mismatch");
  TENRYU_ASSERT(!table.temps_eV.empty(),
                "compute_nlte_coefficients requires non-empty IONMIX temperatures");
  TENRYU_ASSERT(!table.numdens_cm3.empty(),
                "compute_nlte_coefficients requires non-empty IONMIX number densities");

  CellRadiationCoeffs out{};
  out.n_cells = n_cells;
  out.n_groups = n_groups;
  out.is_nlte = true;

  const std::size_t n_cells_us = static_cast<std::size_t>(n_cells);
  const std::size_t n_groups_us = static_cast<std::size_t>(n_groups);
  const std::size_t n_cell_groups = n_cells_us * n_groups_us;
  out.rho_eval.assign(n_cells_us, 0.0);
  out.Te_eval.assign(n_cells_us, 0.0);
  out.cv_e.assign(n_cells_us, 0.0);
  out.beta.assign(n_cells_us, 0.0);
  out.f.assign(n_cells_us, 1.0);
  out.sigma_p_abs.assign(n_cells_us, 0.0);
  out.sigma_p_em.assign(n_cells_us, 0.0);
  out.gamma_diag.assign(n_cells_us, 0.0);
  out.sigma_pa.assign(n_cell_groups, 0.0);
  out.sigma_pe.assign(n_cell_groups, 0.0);
  out.sigma_R.assign(n_cell_groups, 0.0);
  out.sigma_a_eff.assign(n_cell_groups, 0.0);
  out.sigma_s_eff.assign(n_cell_groups, 0.0);
  out.b.assign(n_cell_groups, 0.0);
  out.s.assign(n_cell_groups, 0.0);
  out.J.assign(n_cell_groups, 0.0);
  out.eta.assign(n_cell_groups, 0.0);
  out.eta_tot.assign(n_cells_us, 0.0);
  out.eta_cdf.assign(n_cell_groups, 0.0);
  out.used_planck_fallback.assign(n_cell_groups, static_cast<std::uint8_t>(0));
  out.clamped_negative_alpha.assign(n_cell_groups, static_cast<std::uint8_t>(0));
  out.clamped_negative_eta.assign(n_cell_groups, static_cast<std::uint8_t>(0));

  if (n_cells == 0) {
    return out;
  }

  const int mat_idx = cfg.materials.first_nonvoid_material_index();
  TENRYU_ASSERT(mat_idx >= 0,
                "compute_nlte_coefficients requires at least one non-void material");
  const auto& mat = cfg.materials.materials[static_cast<std::size_t>(mat_idx)];
  TENRYU_ASSERT(mat.opacity_model == "table_nlte" || mat.opacity_model == "tmat",
                "compute_nlte_coefficients requires opacity.model=table_nlte or tmat");

  static int warned_legacy_fleck_controls = 0;
  const bool legacy_controls_requested =
      (mat.lambda_method != "finite_difference") ||
      (std::abs(mat.lambda_fd_delta_rel - 1.0e-4) > 0.0) ||
      (std::abs(mat.lambda_fd_abs_min - 1.0e-6) > 0.0) ||
      (std::abs(mat.nlte_f_min - 1.0e-4) > 0.0);
  if (legacy_controls_requested) {
    ++warned_legacy_fleck_controls;
    if (warned_legacy_fleck_controls == 1 || warned_legacy_fleck_controls % 100 == 0) {
      core::log_warning("NLTE separate emissivity: lambda_method, lambda_fd_* and "
                        "Materials[*].nlte_f_min are ignored; Fleck uses sigma_p_em");
    }
  }
  static int warned_fmax = 0;
  if (cfg.radiation.imc.f_max < 1.0) {
    ++warned_fmax;
    if (warned_fmax == 1 || warned_fmax % 100 == 0) {
      core::log_warning("NLTE separate emissivity: Radiation.imc.f_max is ignored "
                        "in the sigma_p_em-based Fleck path");
    }
  }

  const double alpha_fleck =
      (cfg.radiation.imc.alpha > 0.0) ? cfg.radiation.imc.alpha : 1.0;
  const double dt_safe = std::max(dt, 0.0);
  const double sigma_cap = std::max(cfg.numerics.safety.opacity_cap, 0.0);
  const double T_floor = std::max(cfg.numerics.floors.Te, 0.0);
  const double table_T_min = table.temps_eV.front();
  const double table_T_max = table.temps_eV.back();
  const double table_ni_min = table.numdens_cm3.front();
  const double table_ni_max = table.numdens_cm3.back();
  const double A = std::max(mat.A, 1.0e-12);
  const bool corrected_fleck = cfg.radiation.imc.corrected_fleck;
  const double corrected_fleck_log_delta = std::log(1.0 + kCorrectedFleckDeltaRel);

  std::vector<double> rho(n_cells_us, 0.0);
  std::vector<double> Te(n_cells_us, 0.0);
  std::vector<double> zbar(n_cells_us, 0.0);
  std::vector<double> host_cv_e;
  state.rho.copy_to_host(rho.data());
  state.Te.copy_to_host(Te.data());
  state.zbar.copy_to_host(zbar.data());
  const bool has_table_cv_e = !state.cv_e.empty();
  if (has_table_cv_e) {
    host_cv_e.resize(state.cv_e.size());
    state.cv_e.copy_to_host(host_cv_e.data());
  }

  const auto spectrum_diag = reconstruct_nlte_group_spectrum(
      state, planck, n_cells, n_groups, &out.J, &out.used_planck_fallback);
  out.planck_fallback_cell_count = spectrum_diag.fallback_cell_count;
  out.planck_fallback_group_count = spectrum_diag.fallback_group_count;

  int te_bound_fix_count = 0;
  int ni_bound_fix_count = 0;
  int active_cells = 0;
  double sum_f = 0.0;
  out.min_f = std::numeric_limits<double>::infinity();
  out.max_f = 0.0;
  out.min_sigma_p_abs = std::numeric_limits<double>::infinity();
  out.max_sigma_p_abs = 0.0;
  out.min_sigma_p_em = std::numeric_limits<double>::infinity();
  out.max_sigma_p_em = 0.0;
  out.min_gamma_diag = std::numeric_limits<double>::infinity();
  out.max_gamma_diag = 0.0;

  for (int c = 0; c < n_cells; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    const std::size_t base =
        static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups);

    if (state.cell_is_void[c_us] != 0U) {
      out.f[c_us] = 1.0;
      write_uniform_cdf(out.eta_cdf.data() + base, n_groups);
      std::fill(out.J.begin() + static_cast<std::ptrdiff_t>(base),
                out.J.begin() + static_cast<std::ptrdiff_t>(base + n_groups_us),
                0.0);
      std::fill(out.used_planck_fallback.begin() + static_cast<std::ptrdiff_t>(base),
                out.used_planck_fallback.begin() + static_cast<std::ptrdiff_t>(base + n_groups_us),
                static_cast<std::uint8_t>(0));
      continue;
    }

    ++active_cells;
    const double rho_c = std::max(rho[c_us], 0.0);
    const double zbar_c = std::max(zbar[c_us], 0.0);

    double T_eval = Te[c_us];
    bool te_bound_fixed = false;
    if (!std::isfinite(T_eval) || !(T_eval > 0.0)) {
      T_eval = table_T_min;
      te_bound_fixed = true;
    }
    if (T_eval < table_T_min || T_eval > table_T_max) {
      te_bound_fixed = true;
    }
    if (te_bound_fixed) {
      ++te_bound_fix_count;
    }

    double ni_c = table_ni_min;
    if (rho_c > 0.0) {
      ni_c = rho_c / (A * core::constants::proton_mass);
      bool ni_bound_fixed = false;
      if (!std::isfinite(ni_c) || !(ni_c > 0.0)) {
        ni_c = table_ni_min;
        ni_bound_fixed = true;
      }
      if (ni_c < table_ni_min || ni_c > table_ni_max) {
        ni_bound_fixed = true;
      }
      if (ni_bound_fixed) {
        ++ni_bound_fix_count;
      }
    }

    for (int g = 0; g < n_groups; ++g) {
      out.b[base + static_cast<std::size_t>(g)] =
          planck_fraction_safe(planck, g, T_eval, n_groups);
    }
    normalize_nonnegative_distribution(out.b.data() + base, n_groups);

    const bool has_state_cv_e_cell =
        has_table_cv_e && c_us < host_cv_e.size() && host_cv_e[c_us] > 0.0;
    const double state_cv_e = has_state_cv_e_cell ? host_cv_e[c_us] : 0.0;
    const double cv_e_c = compute_cv_e_host(
        cfg, mat, rho_c, zbar_c, has_state_cv_e_cell, state_cv_e);
    out.rho_eval[c_us] = rho_c;
    out.Te_eval[c_us] = T_eval;
    out.cv_e[c_us] = cv_e_c;

    double eta_tot_c = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx = base + static_cast<std::size_t>(g);
      double sigma_pa_raw = 0.0;
      double sigma_pe_raw = 0.0;
      double sigma_R_raw = 0.0;
      if (rho_c > 0.0) {
        sigma_pa_raw =
            rho_c * interpolate_kappa_with_smooth_edges(table, table.kappa_PA, g, ni_c, T_eval);
        sigma_pe_raw =
            rho_c * interpolate_kappa_with_smooth_edges(table, table.kappa_PE, g, ni_c, T_eval);
        sigma_R_raw =
            rho_c * interpolate_kappa_with_smooth_edges(table, table.kappa_R, g, ni_c, T_eval);
      }

      if (!std::isfinite(sigma_pa_raw) || sigma_pa_raw < 0.0) {
        out.clamped_negative_alpha[idx] = static_cast<std::uint8_t>(1);
        ++out.negative_alpha_clamp_count;
        if (!std::isfinite(sigma_pa_raw)) {
          ++out.nan_inf_count;
        }
      }
      if (!std::isfinite(sigma_pe_raw) || sigma_pe_raw < 0.0) {
        out.clamped_negative_eta[idx] = static_cast<std::uint8_t>(1);
        ++out.negative_eta_clamp_count;
        if (!std::isfinite(sigma_pe_raw)) {
          ++out.nan_inf_count;
        }
      }

      double sigma_pa = std::max(sigma_pa_raw, 0.0);
      double sigma_pe = std::max(sigma_pe_raw, 0.0);
      double sigma_R = std::max(sigma_R_raw, 0.0);
      if (!std::isfinite(sigma_pa)) {
        sigma_pa = 0.0;
      }
      if (!std::isfinite(sigma_pe)) {
        sigma_pe = 0.0;
      }
      if (!std::isfinite(sigma_R)) {
        sigma_R = 0.0;
        ++out.nan_inf_count;
      }
      if (sigma_cap > 0.0) {
        sigma_pa = std::min(sigma_pa, sigma_cap);
        sigma_pe = std::min(sigma_pe, sigma_cap);
        sigma_R = std::min(sigma_R, sigma_cap);
      }

      out.sigma_pa[idx] = sigma_pa;
      out.sigma_pe[idx] = sigma_pe;
      out.sigma_R[idx] = sigma_R;

      double eta_g = 0.0;
      if (T_eval > T_floor) {
        eta_g = sigma_pe * core::constants::c_light * core::constants::a_eV *
                safe_temperature_pow4(T_eval) * out.b[idx];
      }
      if (!std::isfinite(eta_g) || eta_g < 0.0) {
        out.clamped_negative_eta[idx] = static_cast<std::uint8_t>(1);
        ++out.negative_eta_clamp_count;
        ++out.nan_inf_count;
        eta_g = 0.0;
      }
      out.eta[idx] = eta_g;
      eta_tot_c += eta_g;
    }

    if (!(T_eval > T_floor) || !(eta_tot_c > kEtaFloor)) {
      eta_tot_c = 0.0;
      for (int g = 0; g < n_groups; ++g) {
        out.eta[base + static_cast<std::size_t>(g)] = 0.0;
      }
    }
    out.eta_tot[c_us] = eta_tot_c;

    const double beta = compute_beta_host(
        cfg, mat, rho_c, T_eval, zbar_c, has_state_cv_e_cell, state_cv_e);
    out.beta[c_us] = beta;
    const auto scalars = compute_separate_emissivity_scalars_impl(
        out.sigma_pa.data() + base,
        out.eta.data() + base,
        out.b.data() + base,
        n_groups,
        T_eval,
        beta,
        dt,
        alpha_fleck,
        T_floor,
        out.s.data() + base,
        out.eta_cdf.data() + base);

    TENRYU_ASSERT(std::isfinite(scalars.sigma_p_abs) && std::isfinite(scalars.sigma_p_em) &&
                      std::isfinite(scalars.gamma_diag) && std::isfinite(scalars.f),
                  "compute_nlte_coefficients produced non-finite separate-emissivity scalars");

    out.sigma_p_abs[c_us] = scalars.sigma_p_abs;
    out.sigma_p_em[c_us] = scalars.sigma_p_em;
    out.gamma_diag[c_us] = scalars.gamma_diag;
    out.f[c_us] = scalars.f;
    if (corrected_fleck && scalars.sigma_p_em > kCorrectedFleckSigmaFloor) {
      const double T_plus = T_eval * (1.0 + kCorrectedFleckDeltaRel);
      const double spe_plus =
          compute_sigma_p_em_at_T(table, planck, rho_c, ni_c, T_plus, n_groups, sigma_cap);
      double xi = 0.0;
      if (spe_plus > kCorrectedFleckSigmaFloor) {
        xi = (std::log(spe_plus) - std::log(scalars.sigma_p_em)) /
             (4.0 * corrected_fleck_log_delta);
      }
      if (!std::isfinite(xi)) {
        xi = 0.0;
      }
      const double one_plus_xi =
          std::clamp(1.0 + xi, kCorrectedFleckOnePlusXiMin, kCorrectedFleckOnePlusXiMax);
      double f_corr = 1.0 / (1.0 + alpha_fleck * beta * core::constants::c_light * dt_safe *
                                      scalars.sigma_p_em * one_plus_xi);
      f_corr = std::clamp(f_corr, 0.0, 1.0);
      out.f[c_us] = f_corr;
    }
    const double f_cell = out.f[c_us];
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx = base + static_cast<std::size_t>(g);
      out.sigma_a_eff[idx] = f_cell * out.sigma_pa[idx];
      out.sigma_s_eff[idx] = (1.0 - f_cell) * out.sigma_pa[idx];
    }

    sum_f += f_cell;
    out.min_f = std::min(out.min_f, f_cell);
    out.max_f = std::max(out.max_f, f_cell);
    out.min_sigma_p_abs = std::min(out.min_sigma_p_abs, scalars.sigma_p_abs);
    out.max_sigma_p_abs = std::max(out.max_sigma_p_abs, scalars.sigma_p_abs);
    out.min_sigma_p_em = std::min(out.min_sigma_p_em, scalars.sigma_p_em);
    out.max_sigma_p_em = std::max(out.max_sigma_p_em, scalars.sigma_p_em);
    out.min_gamma_diag = std::min(out.min_gamma_diag, scalars.gamma_diag);
    out.max_gamma_diag = std::max(out.max_gamma_diag, scalars.gamma_diag);
  }

  if (active_cells > 0) {
    out.mean_f = sum_f / static_cast<double>(active_cells);
  } else {
    out.min_f = 1.0;
    out.max_f = 1.0;
    out.mean_f = 1.0;
    out.min_sigma_p_abs = 0.0;
    out.max_sigma_p_abs = 0.0;
    out.min_sigma_p_em = 0.0;
    out.max_sigma_p_em = 0.0;
    out.min_gamma_diag = 0.0;
    out.max_gamma_diag = 0.0;
  }

  if (te_bound_fix_count > 0) {
    static int warn_count = 0;
    ++warn_count;
    if (warn_count == 1 || warn_count % 100 == 0) {
      core::log_warning("NLTE: resolved " + std::to_string(te_bound_fix_count) +
                        " cells' Te outside opacity-table bounds "
                        "(hot-side edge extrapolation, otherwise bound clamp)");
    }
  }
  if (ni_bound_fix_count > 0) {
    static int warn_count = 0;
    ++warn_count;
    if (warn_count == 1 || warn_count % 100 == 0) {
      core::log_warning("NLTE: resolved " + std::to_string(ni_bound_fix_count) +
                        " cells' ni outside opacity-table bounds "
                        "(low-density edge extrapolation, otherwise bound clamp)");
    }
  }
  if (out.planck_fallback_group_count > 0) {
    static int warn_count = 0;
    ++warn_count;
    if (warn_count == 1 || warn_count % 100 == 0) {
      core::log_warning("NLTE spectrum reconstruction: used Planck fallback in " +
                        std::to_string(out.planck_fallback_group_count) +
                        " cell-groups across " +
                        std::to_string(out.planck_fallback_cell_count) + " cells");
    }
  }
  if (out.negative_alpha_clamp_count > 0) {
    static int warn_count = 0;
    ++warn_count;
    if (warn_count == 1 || warn_count % 100 == 0) {
      core::log_warning("NLTE: clamped " +
                        std::to_string(out.negative_alpha_clamp_count) +
                        " negative/non-finite alpha_g values to zero");
    }
  }
  if (out.negative_eta_clamp_count > 0) {
    static int warn_count = 0;
    ++warn_count;
    if (warn_count == 1 || warn_count % 100 == 0) {
      core::log_warning("NLTE: clamped " +
                        std::to_string(out.negative_eta_clamp_count) +
                        " negative/non-finite eta_g values to zero");
    }
  }

  if (cfg.main.verbosity == "verbose") {
    core::log_info("NLTE separate emissivity: f[min,max,mean]=[" +
                   std::to_string(out.min_f) + ", " + std::to_string(out.max_f) + ", " +
                   std::to_string(out.mean_f) + "] sigma_p_abs[min,max]=[" +
                   std::to_string(out.min_sigma_p_abs) + ", " +
                   std::to_string(out.max_sigma_p_abs) + "] sigma_p_em[min,max]=[" +
                   std::to_string(out.min_sigma_p_em) + ", " +
                   std::to_string(out.max_sigma_p_em) + "] gamma[min,max]=[" +
                   std::to_string(out.min_gamma_diag) + ", " +
                   std::to_string(out.max_gamma_diag) + "] nan_inf_count=" +
                   std::to_string(out.nan_inf_count));
  }

  return out;
}

}  // namespace tenryu::radiation
