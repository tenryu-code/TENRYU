#pragma once

#include "materials/helmholtz_jet.hpp"
#include "materials/helmholtz_jet_device.cuh"

namespace tenryu::materials {

class DeviceHelmholtzJet {
 public:
  DeviceHelmholtzJet() = default;
  ~DeviceHelmholtzJet();
  DeviceHelmholtzJet(DeviceHelmholtzJet&&) noexcept;
  DeviceHelmholtzJet& operator=(DeviceHelmholtzJet&&) noexcept;
  DeviceHelmholtzJet(const DeviceHelmholtzJet&) = delete;
  DeviceHelmholtzJet& operator=(const DeviceHelmholtzJet&) = delete;

  void upload(const HelmholtzJetEOS& cpu_jet);
  [[nodiscard]] HelmholtzJetDeviceView view() const;
  [[nodiscard]] bool empty() const { return n_rho_ == 0; }

 private:
  double* d_log_rho_grid_ = nullptr;
  double* d_log_T_grid_ = nullptr;
  double* d_cell_coeffs_ = nullptr;
  int n_rho_ = 0;
  int n_T_ = 0;
  double log_rho_min_ = 0.0;
  double log_rho_max_ = 0.0;
  double log_T_min_ = 0.0;
  double log_T_max_ = 0.0;
  double d_log_rho_inv_ = 0.0;
  double d_log_T_inv_ = 0.0;

  void free_all();
};

}  // namespace tenryu::materials
