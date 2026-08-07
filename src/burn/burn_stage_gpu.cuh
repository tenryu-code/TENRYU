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
    double& dt_limit_subcycle, unsigned int& screening_warning_flags);

}  // namespace tenryu::burn
