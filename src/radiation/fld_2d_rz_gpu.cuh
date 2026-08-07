#pragma once

#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"
#include "diagnostics/radial_fourier_audit.hpp"
#include "parallel/comm_buffers.hpp"
#include "parallel/partition.hpp"
#include "radiation/planck_table.cuh"

namespace tenryu::radiation {

void advance_radiation_step_fld_2d_rz(
    core::State& state,
    const core::Config& cfg,
    const PlanckTable& planck,
    const core::Config::MaterialsConfig::MatDef& mat,
    double dt,
    const parallel::PartitionInfo& part = parallel::PartitionInfo{},
    parallel::CommBuffers* bufs = nullptr);

double fld_compute_max_reduced_flux_2d_rz(
    core::State& state,
    const core::Config& cfg,
    const PlanckTable& planck,
    const core::Config::MaterialsConfig::MatDef& mat);

std::vector<diagnostics::FldSubstageAuditRecord>
drain_fld_substage_audit_records();

}  // namespace tenryu::radiation
