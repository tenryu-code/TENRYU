#pragma once

#include "materials/eos_rho_e_device.cuh"

namespace tenryu::materials {

class EOSRhoETable;

class DeviceEOSRhoETable {
 public:
  DeviceEOSRhoETable() = default;
  ~DeviceEOSRhoETable();
  DeviceEOSRhoETable(DeviceEOSRhoETable&&) noexcept;
  DeviceEOSRhoETable& operator=(DeviceEOSRhoETable&&) noexcept;
  DeviceEOSRhoETable(const DeviceEOSRhoETable&) = delete;
  DeviceEOSRhoETable& operator=(const DeviceEOSRhoETable&) = delete;

  void upload(const EOSRhoETable& cpu_table);

  [[nodiscard]] EOSRhoEDeviceView view() const;

  [[nodiscard]] bool empty() const {
    return n_rho_ == 0;
  }

 private:
  double* d_log_rho_grid_ = nullptr;
  double* d_log_e_grid_ = nullptr;
  double* d_P_table_ = nullptr;
  double* d_T_table_ = nullptr;
  double* d_P_xx_ = nullptr;
  double* d_P_yy_ = nullptr;
  double* d_P_xxyy_ = nullptr;
  double* d_T_xx_ = nullptr;
  double* d_T_yy_ = nullptr;
  double* d_T_xxyy_ = nullptr;
  bool use_log_grid_ = true;
  int n_rho_ = 0;
  int n_e_ = 0;
  double log_rho_min_ = 0.0;
  double log_rho_max_ = 0.0;
  double log_e_min_ = 0.0;
  double log_e_max_ = 0.0;
  double d_log_rho_inv_ = 0.0;
  double d_log_e_inv_ = 0.0;

  void free_all();
};

}  // namespace tenryu::materials
