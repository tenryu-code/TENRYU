#pragma once

#include <cstddef>
#include <cmath>

#include <cuda_runtime.h>

#include "burn/burn_constants.hpp"

namespace tenryu::burn {

struct CormanParams {
  int n_groups = 30;
  double E_min_keV = 20.0;
  double E_max_keV = 15500.0;
  double lnL_e = 0.0;
  double lnL_I = 0.0;
};

struct CormanStepResult {
  double escaped_erg = 0.0;
  double inflight_erg = 0.0;
  double sourced_erg = 0.0;
};

namespace corman_detail {

inline constexpr double kPi = 3.141592653589793238462643383279502884;
inline constexpr double kElectronChargeEsu = 4.80320425e-10;
inline constexpr double kElectronMassG = 9.1093837015e-28;
inline constexpr double kProtonMassG = 1.6726219e-24;
inline constexpr double kEVToErg = 1.602176634e-12;
inline constexpr double kKeVToErg = kMeVToErg / 1000.0;
inline constexpr double kFieldA = 2.51505;

__host__ __device__ inline double corman_finite_or_zero(const double x) {
#if defined(__CUDA_ARCH__)
  return isfinite(x) ? x : 0.0;
#else
  return std::isfinite(x) ? x : 0.0;
#endif
}

}  // namespace corman_detail

__host__ __device__ inline double corman_electron_log(double Te_eV,
                                                      double ne_cm3) {
  if (!(Te_eV > 0.0) || !(ne_cm3 > 0.0)) {
    return 2.0;
  }
  const double lnTe = log(Te_eV);
  const double ln_arg = 0.5 * log(ne_cm3) - 1.25 * lnTe;
  const double correction =
      sqrt(1.0e-5 + ((lnTe - 2.0) * (lnTe - 2.0)) / 16.0);
  const double lnL = 23.5 - ln_arg - correction;
  return (lnL > 2.0) ? lnL : 2.0;
}

__host__ __device__ inline double corman_ion_log(
    double species_A, double species_Z, double E_birth_keV, double rho_gcc,
    double Te_eV, double Ti_eV, double ne_cm3) {
  if (!(species_A > 0.0) || !(species_Z > 0.0) || !(E_birth_keV > 0.0) ||
      !(rho_gcc > 0.0) || !(Te_eV > 0.0) || !(Ti_eV > 0.0) ||
      !(ne_cm3 > 0.0)) {
    return 2.0;
  }
  const double e2 = corman_detail::kElectronChargeEsu *
                    corman_detail::kElectronChargeEsu;
  const double kTe = Te_eV * corman_detail::kEVToErg;
  const double kTi = Ti_eV * corman_detail::kEVToErg;
  const double n_i = rho_gcc / (2.5 * corman_detail::kProtonMassG);
  const double inv_lambda2 =
      4.0 * corman_detail::kPi * e2 * (n_i / kTi + ne_cm3 / kTe);
  if (!(inv_lambda2 > 0.0)) {
    return 2.0;
  }
  const double lambda_D = 1.0 / sqrt(inv_lambda2);
  const double m_s = species_A * corman_detail::kProtonMassG;
  const double m_field = 2.5 * corman_detail::kProtonMassG;
  const double mu_rel = m_s * m_field / (m_s + m_field);
  const double E_birth_erg = E_birth_keV * corman_detail::kKeVToErg;
  const double u2 = 2.0 * E_birth_erg / m_s;
  const double p_perp = species_Z * e2 / (mu_rel * u2);
  if (!(lambda_D > 0.0) || !(p_perp > 0.0)) {
    return 2.0;
  }
  const double lnL = log(lambda_D / p_perp);
  return (lnL > 2.0) ? lnL : 2.0;
}

__host__ __device__ inline double corman_tE(double species_A,
                                            double species_Z,
                                            double Te_eV,
                                            double ne_cm3,
                                            double lnL_e) {
  if (!(species_A > 0.0) || !(species_Z > 0.0) || !(Te_eV > 0.0) ||
      !(ne_cm3 > 0.0) || !(lnL_e > 0.0)) {
    return INFINITY;
  }
  const double m_s = species_A * corman_detail::kProtonMassG;
  const double kTe = Te_eV * corman_detail::kEVToErg;
  const double e2 = corman_detail::kElectronChargeEsu *
                    corman_detail::kElectronChargeEsu;
  const double e4 = e2 * e2;
  const double numerator = 3.0 * m_s * kTe * sqrt(kTe);
  const double denominator =
      4.0 * sqrt(2.0 * corman_detail::kPi * corman_detail::kElectronMassG) *
      ne_cm3 * species_Z * species_Z * e4 * lnL_e;
  return numerator / denominator;
}

__host__ __device__ inline double corman_gamma(double species_A,
                                               double species_Z,
                                               double rho_gcc,
                                               double lnL_I) {
  if (!(species_A > 0.0) || !(species_Z > 0.0) || !(rho_gcc > 0.0) ||
      !(lnL_I > 0.0)) {
    return 0.0;
  }
  const double m_s = species_A * corman_detail::kProtonMassG;
  const double m_D = 2.0141 * corman_detail::kProtonMassG;
  const double m_T = 3.0160 * corman_detail::kProtonMassG;
  const double n_i = rho_gcc /
                     (corman_detail::kFieldA * corman_detail::kProtonMassG);
  const double sum_nz2_over_m = 0.5 * n_i * (1.0 / m_D + 1.0 / m_T);
  const double e2 = corman_detail::kElectronChargeEsu *
                    corman_detail::kElectronChargeEsu;
  const double e4 = e2 * e2;
  return 4.0 * corman_detail::kPi * e4 * species_Z * species_Z * lnL_I *
         sum_nz2_over_m * sqrt(0.5 * m_s);
}

__host__ __device__ inline double corman_lambda(double species_A,
                                                double species_Z,
                                                double E_keV,
                                                double rho_gcc,
                                                double lnL_I) {
  if (!(species_A > 0.0) || !(species_Z > 0.0) || !(E_keV > 0.0) ||
      !(rho_gcc > 0.0) || !(lnL_I > 0.0)) {
    return INFINITY;
  }
  const double m_s = species_A * corman_detail::kProtonMassG;
  const double E_erg = E_keV * corman_detail::kKeVToErg;
  const double v = sqrt(2.0 * E_erg / m_s);
  const double n_i = rho_gcc /
                     (corman_detail::kFieldA * corman_detail::kProtonMassG);
  const double sum_nz2 = n_i;
  const double e2 = corman_detail::kElectronChargeEsu *
                    corman_detail::kElectronChargeEsu;
  const double e4 = e2 * e2;
  const double nu_D =
      8.0 * corman_detail::kPi * e4 * species_Z * species_Z * lnL_I *
      sum_nz2 / (m_s * m_s * v * v * v);
  if (!(nu_D > 0.0)) {
    return INFINITY;
  }
  return 2.0 * v / nu_D;
}

CormanStepResult corman_diffusion_step(
    const CormanParams& p, int n_cells, double species_A, double species_Z,
    double E_birth_keV, const double* r_node_dev, const double* vol_dev,
    const double* rho_dev, const double* Te_eV_dev, const double* Ti_eV_dev,
    const double* ne_dev, const double* S_birth_dev, double dt_s,
    double* N_dev, double* dep_e_dev, double* dep_i_dev, void* scratch_pool,
    std::size_t scratch_bytes, cudaStream_t stream);

std::size_t corman_diffusion_scratch_bytes(int n_groups, int n_cells);

void scale_rows_by_cell(double* dst, const double* src, const double* rho,
                        int G, int J);
void divide_rows_by_cell(double* dst, const double* src, const double* rho,
                         int G, int J);

}  // namespace tenryu::burn
