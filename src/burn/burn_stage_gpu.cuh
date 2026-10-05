#pragma once

#include <vector>

#include "burn/burn_stage.hpp"

namespace tenryu::burn {

BurnStageResult compute_burn_step_1d_device_stage(
    const BurnStageInputs& in, const BurnStageParams& p,
    const PartitionTable& table,
    std::vector<double>& burn_y, std::vector<double>& dE_e,
    std::vector<double>& dE_i, std::vector<double>& rate_diag,
    std::vector<double>& Qe_diag, std::vector<double>& Qi_diag,
    std::vector<double>* S_birth, std::vector<double>& nh_emit,
    double& dt_limit_subcycle, unsigned int& screening_warning_flags,
    std::vector<double>* neutron_births = nullptr);

// The 1D stage on the device-resident arrays (BurnDeviceInputs, BurnDeviceArrays): the same
// kernels as compute_burn_step_1d_device_stage, the cells' velocities and range fit factors
// formed on the device, the result packet the only copy to the host; then eps_cum and the
// neutron count updated on the device.
BurnStageResult compute_burn_step_1d_resident(const BurnDeviceInputs& in,
                                              const BurnStageParams& p,
                                              const PartitionTable& table,
                                              const BurnDeviceArrays& out,
                                              double& dt_limit_subcycle,
                                              unsigned int& screening_warning_flags);

}  // namespace tenryu::burn
