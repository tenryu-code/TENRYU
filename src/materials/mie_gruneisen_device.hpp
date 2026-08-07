#pragma once

#include "materials/mie_gruneisen_device.cuh"

namespace tenryu::materials {

class MieGruneisenEOS;

class DeviceMieGruneisen {
 public:
  DeviceMieGruneisen() = default;
  ~DeviceMieGruneisen();
  DeviceMieGruneisen(DeviceMieGruneisen&& other) noexcept;
  DeviceMieGruneisen& operator=(DeviceMieGruneisen&& other) noexcept;
  DeviceMieGruneisen(const DeviceMieGruneisen&) = delete;
  DeviceMieGruneisen& operator=(const DeviceMieGruneisen&) = delete;

  void upload(const MieGruneisenEOS& cpu_eos);

  [[nodiscard]] MieGruneisenDeviceView view() const;

  [[nodiscard]] bool empty() const {
    return n_rho_ == 0;
  }

 private:
  double* d_log_rho_grid_ = nullptr;
  double* d_gamma_ion_ = nullptr;
  double* d_gamma_ele_ = nullptr;
  double* d_p_ref_ion_ = nullptr;
  double* d_p_ref_ele_ = nullptr;
  double* d_e_ref_ion_ = nullptr;
  double* d_e_ref_ele_ = nullptr;
  double* d_dgamma_ion_dlogrho_ = nullptr;
  double* d_dgamma_ele_dlogrho_ = nullptr;
  double* d_dp_ref_ion_dlogrho_ = nullptr;
  double* d_dp_ref_ele_dlogrho_ = nullptr;
  double* d_de_ref_ion_dlogrho_ = nullptr;
  double* d_de_ref_ele_dlogrho_ = nullptr;
  int n_rho_ = 0;
  double log_rho_min_ = 0.0;
  double log_rho_max_ = 0.0;
  double d_log_rho_inv_ = 0.0;

  void free_all();
};

}  // namespace tenryu::materials
