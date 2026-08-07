#pragma once

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "radiation/mode_selector.hpp"

namespace tenryu::radiation {

enum class DDMCBoundaryType : std::uint8_t {
  Internal = 0,
  Vacuum = 1,
  Reflective = 2,
  Interface = 3,
};

struct CellDDMCData {
  static constexpr int kFaceCount = 4;
  // Face convention: 0=R_left, 1=R_right, 2=Z_bottom, 3=Z_top.
  double sigma_leak_face[kFaceCount] = {0.0, 0.0, 0.0, 0.0};
  DDMCBoundaryType bc_face[kFaceCount] = {DDMCBoundaryType::Internal,
                                          DDMCBoundaryType::Internal,
                                          DDMCBoundaryType::Internal,
                                          DDMCBoundaryType::Internal};
  int neighbor_face[kFaceCount] = {-1, -1, -1, -1};
  double A_face[kFaceCount] = {0.0, 0.0, 0.0, 0.0};
  double delta_x_face[kFaceCount] = {0.0, 0.0, 0.0, 0.0};

  // Legacy aliases kept for 1D compatibility.
  double sigma_leak_left = 0.0;
  double sigma_leak_right = 0.0;
  double sigma_leak_out = 0.0;
  double sigma_leak_bnd = 0.0;
  DDMCBoundaryType bc_left = DDMCBoundaryType::Internal;
  DDMCBoundaryType bc_right = DDMCBoundaryType::Internal;
};

struct OpacityProvider {
  virtual ~OpacityProvider() = default;
  [[nodiscard]] virtual double sigma_rosseland(int cell,
                                               int group,
                                               double rho,
                                               double Te_eV) const = 0;
};

class ConstantOpacityProvider final : public OpacityProvider {
 public:
  explicit ConstantOpacityProvider(std::vector<double> kappa_rosseland_cm2_per_g)
      : kappa_R_(std::move(kappa_rosseland_cm2_per_g)) {}

  [[nodiscard]] double sigma_rosseland(int,
                                       int group,
                                       double rho,
                                       double) const override;

 private:
  std::vector<double> kappa_R_;
};

class DDMCCoefficients {
 public:
  DDMCCoefficients() = default;
  DDMCCoefficients(std::int64_t n_cells,
                   int n_groups,
                   double sigma_floor = 1.0e-20,
                   double sigma_cap = std::numeric_limits<double>::infinity());

  [[nodiscard]] static double compute_diffusion_coeff(double sigma_R);
  [[nodiscard]] static double compute_face_temperature(double T_left_eV,
                                                       double T_right_eV);

  void compute_1d(const std::vector<double>& node_r,
                  const std::vector<double>& rho,
                  const std::vector<double>& Te,
                  const std::vector<double>& sigma_R_center,
                  const ModeSelector& mode_selector,
                  DDMCBoundaryType bc_inner,
                  DDMCBoundaryType bc_outer,
                  bool m_matrix_check,
                  const OpacityProvider* opacity_provider = nullptr);

  void compute_2d(const std::vector<double>& node_r,
                  const std::vector<double>& node_z,
                  int nr,
                  int nz,
                  const std::vector<double>& cell_vol,
                  const std::vector<double>& rho,
                  const std::vector<double>& Te,
                  const std::vector<double>& sigma_R_center,
                  const ModeSelector& mode_selector,
                  DDMCBoundaryType bc_inner_r,
                  DDMCBoundaryType bc_outer_r,
                  DDMCBoundaryType bc_bottom_z,
                  DDMCBoundaryType bc_top_z,
                  bool m_matrix_check,
                  const OpacityProvider* opacity_provider = nullptr);

  [[nodiscard]] const CellDDMCData& get_cell_data(std::int64_t cell,
                                                  int group) const;

  [[nodiscard]] double get_face_sigma_minus(std::int64_t face,
                                            int group) const;
  [[nodiscard]] double get_face_sigma_plus(std::int64_t face,
                                           int group) const;

  [[nodiscard]] std::int64_t n_cells() const { return n_cells_; }
  [[nodiscard]] int n_groups() const { return n_groups_; }
  [[nodiscard]] double sigma_floor() const { return sigma_floor_; }
  [[nodiscard]] double sigma_cap() const { return sigma_cap_; }

 private:
  [[nodiscard]] std::size_t cell_index(std::int64_t cell, int group) const;
  [[nodiscard]] std::size_t face_index(std::int64_t face, int group) const;
  [[nodiscard]] double clamp_sigma(double sigma) const;
  [[nodiscard]] double compute_vacuum_leak(double D,
                                           double dx,
                                           double sigma_tr) const;
  [[nodiscard]] double compute_interface_leak(double sigma_R,
                                              double dx) const;
  void sync_legacy_aliases(CellDDMCData& data) const;

  std::int64_t n_cells_ = 0;
  int n_groups_ = 0;
  double sigma_floor_ = 1.0e-20;
  double sigma_cap_ = std::numeric_limits<double>::infinity();

  std::vector<CellDDMCData> cell_data_;
  std::vector<double> face_sigma_minus_;
  std::vector<double> face_sigma_plus_;
};

}  // namespace tenryu::radiation
