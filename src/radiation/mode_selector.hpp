#pragma once

#include <cstdint>
#include <vector>

#include "radiation/cell_radiation_coeffs.hpp"

namespace tenryu::radiation {

enum class TransportMode : std::uint8_t {
  IMC = 0,
  DDMC = 1,
  RW = 2,
  Diffusion = 3,
};

struct ModeSelectorConfig {
  double tau_ddmc = 4.0;
  double tau_rw = 1.0;
  double omega_ddmc = 0.9;
  double tau_ddmc_off = -1.0;
  double omega_ddmc_off = -1.0;
  int mode_hold = 0;
  double rate_max = 1.0e30;
  bool emissivity_preserving = true;
  double sigma_floor = 1.0e-20;
};

class ModeSelector {
 public:
  struct HysteresisResult {
    std::int64_t switches_imc_to_ddmc = 0;
    std::int64_t switches_ddmc_to_imc = 0;
  };

  ModeSelector() = default;
  ModeSelector(std::int64_t n_cells,
               int n_groups,
               const ModeSelectorConfig& config = ModeSelectorConfig{});

  [[nodiscard]] static double compute_optical_depth(double sigma_tr,
                                                    double cell_length_cm);

  [[nodiscard]] static double compute_scattering_ratio(double fleck_f,
                                                       double sigma_a = 0.0,
                                                       double sigma_s_phys = 0.0);

  [[nodiscard]] static double compute_cell_length(double r_inner,
                                                  double r_outer);

  [[nodiscard]] bool check_conversion_probability_constraint(double sigma_R,
                                                             double dx,
                                                             double omega = 1.0) const;

  void compute_modes(const std::vector<double>& node_r,
                     const std::vector<double>& sigma_R,
                     const std::vector<double>& fleck_f,
                     const std::vector<double>& sigma_a,
                     const std::vector<double>& sigma_s_phys = {},
                     const CellRadiationCoeffs* coeffs = nullptr,
                     double dt = 0.0,
                     double alpha = 1.0);

  void compute_modes_2d_rz(const std::vector<double>& node_r,
                           const std::vector<double>& node_z,
                           int nr,
                           int nz,
                           const std::vector<double>& sigma_R,
                           const std::vector<double>& fleck_f,
                           const std::vector<double>& sigma_a,
                           const std::vector<double>& sigma_s_phys = {},
                           const CellRadiationCoeffs* coeffs = nullptr,
                           double dt = 0.0,
                           double alpha = 1.0);

  [[nodiscard]] TransportMode get_mode(std::int64_t cell, int group) const;
  [[nodiscard]] double get_tau(std::int64_t cell, int group) const;
  [[nodiscard]] double get_omega(std::int64_t cell, int group) const;

  void force_imc(std::int64_t cell, int group);

  HysteresisResult apply_hysteresis(
      const std::vector<TransportMode>& prev_mode,
      const std::vector<double>& prev_tau,
      std::vector<std::uint8_t>& hold_count);

  [[nodiscard]] std::int64_t count_ddmc() const;
  [[nodiscard]] std::int64_t count_rw() const;
  [[nodiscard]] std::int64_t count_imc() const;
  [[nodiscard]] std::int64_t count_omega_below_threshold() const;

  [[nodiscard]] std::int64_t n_cells() const { return n_cells_; }
  [[nodiscard]] int n_groups() const { return n_groups_; }
  [[nodiscard]] const ModeSelectorConfig& config() const { return config_; }
  [[nodiscard]] const std::vector<double>& cell_lengths() const {
    return cell_length_;
  }
  [[nodiscard]] const std::vector<TransportMode>& modes() const { return mode_; }

 private:
  [[nodiscard]] std::size_t index(std::int64_t cell, int group) const;
  void classify_modes_from_lengths(const std::vector<double>& sigma_R,
                                   const std::vector<double>& fleck_f,
                                   const std::vector<double>& sigma_a,
                                   const std::vector<double>& sigma_s_phys,
                                   const CellRadiationCoeffs* coeffs,
                                   double dt,
                                   double alpha,
                                   bool allow_rw);

  std::int64_t n_cells_ = 0;
  int n_groups_ = 0;
  ModeSelectorConfig config_{};

  std::vector<TransportMode> mode_;
  std::vector<double> tau_;
  std::vector<double> omega_;
  std::vector<double> cell_length_;
  std::int64_t omega_below_threshold_ = 0;
};

}  // namespace tenryu::radiation
