#pragma once

#include "materials/helmholtz_spline.hpp"
#include "materials/helmholtz_spline_device.cuh"

namespace tenryu::materials {

class DeviceHelmholtzSpline {
 public:
  DeviceHelmholtzSpline() = default;
  ~DeviceHelmholtzSpline();
  DeviceHelmholtzSpline(DeviceHelmholtzSpline&&) noexcept;
  DeviceHelmholtzSpline& operator=(DeviceHelmholtzSpline&&) noexcept;
  DeviceHelmholtzSpline(const DeviceHelmholtzSpline&) = delete;
  DeviceHelmholtzSpline& operator=(const DeviceHelmholtzSpline&) = delete;

  void upload(const HelmholtzSpline1T& cpu_spline);
  [[nodiscard]] HelmholtzSplineDeviceView view() const;
  [[nodiscard]] bool empty() const {
    return n_rho_ == 0;
  }

 private:
  double* d_log_rho_grid_ = nullptr;
  double* d_log_T_grid_ = nullptr;

  double* d_P_nodes_ = nullptr;
  double* d_P_dx_nodes_ = nullptr;
  double* d_P_dy_nodes_ = nullptr;
  double* d_P_dxdy_nodes_ = nullptr;

  double* d_e_nodes_ = nullptr;
  double* d_e_dx_nodes_ = nullptr;
  double* d_e_dy_nodes_ = nullptr;
  double* d_e_dxdy_nodes_ = nullptr;

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
