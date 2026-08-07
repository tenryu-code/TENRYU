#pragma once

#include <cstdint>

#include "materials/eos_device_table.cuh"

namespace tenryu::materials {

struct EOSTable;  // forward declaration

class DeviceEOSTable {
 public:
  DeviceEOSTable() = default;
  ~DeviceEOSTable();
  DeviceEOSTable(DeviceEOSTable&&) noexcept;
  DeviceEOSTable& operator=(DeviceEOSTable&&) noexcept;
  DeviceEOSTable(const DeviceEOSTable&) = delete;
  DeviceEOSTable& operator=(const DeviceEOSTable&) = delete;

  void upload(const EOSTable& cpu_table);

  [[nodiscard]] DeviceEOSTableView view() const;

  [[nodiscard]] bool empty() const {
    return n_rho_ == 0;
  }

  [[nodiscard]] bool supports_rho_e_reclosure() const {
    return supports_rho_e_reclosure_ != 0u;
  }

 private:
  double* d_log_rho_grid_ = nullptr;
  double* d_log_T_grid_ = nullptr;
  double* d_P_table_ = nullptr;
  double* d_e_table_ = nullptr;
  double* d_cv_table_ = nullptr;
  int n_rho_ = 0;
  int n_T_ = 0;
  double log_rho_min_ = 0.0;
  double log_rho_max_ = 0.0;
  double log_T_min_ = 0.0;
  double log_T_max_ = 0.0;
  double d_log_rho_inv_ = 0.0;
  double d_log_T_inv_ = 0.0;
  std::uint8_t supports_rho_e_reclosure_ = 0u;

  void free_all();
};

}  // namespace tenryu::materials
