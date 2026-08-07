#pragma once

#include "core/config.hpp"
#include "core/state.hpp"
#include "hydro/ale_mode.hpp"
#include "hydro/ale_rezone.cuh"
#include "hydro/mesh_regime.hpp"
#include "parallel/partition.hpp"

namespace tenryu::parallel {
struct CommBuffers;
class Reduction;
}  // namespace tenryu::parallel

namespace tenryu::hydro::ale {

RezoneResult run_cd_local_winslow_rezone(
    core::State& state,
    const core::Config& cfg,
    const parallel::PartitionInfo& part,
    parallel::CommBuffers* bufs,
    const parallel::Reduction* reduction,
    const AleRequest& request,
    const CellRegime* d_cell_regime);

}  // namespace tenryu::hydro::ale
