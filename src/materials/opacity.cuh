#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "core/device_error_flags.cuh"

namespace tenryu::materials {

inline constexpr int kOpacityModelConstant = 0;
inline constexpr int kOpacityModelFreqDepMarshak = 1;
inline constexpr int kOpacityModelTableNLTE = 2;
inline constexpr int kOpacityModelTableNlte = kOpacityModelTableNLTE;
inline constexpr int kOpacityModelPowerLaw = 3;

[[nodiscard]] inline double power_law_opacity_kappa_cm2_g(
    const double rho,
    const double T_eV,
    const double kappa0_cm2_g,
    const double alpha_T,
    const double lambda_rho,
    const double T_ref_eV,
    const double rho_ref_g_cc) {
  return kappa0_cm2_g * std::pow(T_eV / T_ref_eV, -alpha_T) *
         std::pow(rho / rho_ref_g_cc, lambda_rho);
}

class Opacity {
 public:
  virtual ~Opacity() = default;

  [[nodiscard]] virtual double kappa_planck(int group,
                                             double rho,
                                             double T_eV) const = 0;
  [[nodiscard]] virtual double kappa_rosseland(int group,
                                                double rho,
                                                double T_eV) const = 0;

  [[nodiscard]] virtual double sigma_absorption(int group,
                                                 double rho,
                                                 double T_eV) const {
    return std::max(0.0, rho) * kappa_planck(group, rho, T_eV);
  }

  [[nodiscard]] virtual double sigma_rosseland(int group,
                                                double rho,
                                                double T_eV) const {
    return std::max(0.0, rho) * kappa_rosseland(group, rho, T_eV);
  }
};

class ConstantOpacity final : public Opacity {
 public:
  ConstantOpacity(double kappa_planck_cm2_g, double kappa_rosseland_cm2_g)
      : kappa_planck_(kappa_planck_cm2_g), kappa_rosseland_(kappa_rosseland_cm2_g) {}

  [[nodiscard]] double kappa_planck(int,
                                     double,
                                     double) const override {
    return kappa_planck_;
  }

  [[nodiscard]] double kappa_rosseland(int,
                                        double,
                                        double) const override {
    return kappa_rosseland_;
  }

 private:
  double kappa_planck_ = 0.0;
  double kappa_rosseland_ = 0.0;
};

class FrequencyDependentOpacity final : public Opacity {
 public:
  explicit FrequencyDependentOpacity(std::vector<double> group_bounds_eV)
      : bounds_eV_(std::move(group_bounds_eV)) {}

  [[nodiscard]] static double sigma_nu(double nu_eV, double T_eV);

  [[nodiscard]] double kappa_planck(int group,
                                     double rho,
                                     double T_eV) const override;

  [[nodiscard]] double kappa_rosseland(int group,
                                        double rho,
                                        double T_eV) const override;

 private:
  [[nodiscard]] double energy_lo(int group) const;
  [[nodiscard]] double energy_hi(int group) const;

  std::vector<double> bounds_eV_;
};

struct OpacityEvalView {
  const double* rho = nullptr;
  const double* Te = nullptr;
  double* sigma_a = nullptr;
  double* sigma_R = nullptr;
  const double* group_bounds_eV = nullptr;
  int n_cells = 0;
  int n_groups = 1;
  int opacity_model = kOpacityModelConstant;
  double kappa_planck_const = 0.0;
  double kappa_rosseland_const = 0.0;
  const double* kappa_planck_cell = nullptr;
  const double* kappa_rosseland_cell = nullptr;
  double kappa_floor = 0.0;
  double kappa_cap = 1.0e20;
  double power_law_kappa0 = 0.0;
  double power_law_alpha_T = 0.0;
  double power_law_lambda_rho = 0.0;
  double power_law_T_ref_eV = 1.0;
  double power_law_rho_ref = 1.0;
};

void evaluate_opacity_cuda(const OpacityEvalView& view,
                           tenryu::core::DeviceErrorFlags* host_flags_out = nullptr);
void evaluate_constant_opacity_cuda(const OpacityEvalView& view,
                                    tenryu::core::DeviceErrorFlags* host_flags_out = nullptr);
void evaluate_power_law_opacity_cuda(const OpacityEvalView& view,
                                     tenryu::core::DeviceErrorFlags* host_flags_out = nullptr);
void evaluate_frequency_dependent_opacity_cuda(
    const OpacityEvalView& view,
    tenryu::core::DeviceErrorFlags* host_flags_out = nullptr);

}  // namespace tenryu::materials
