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

RezoneResult run_axis_spine_plus_local_rezone(
    core::State& state,
    const core::Config& cfg,
    const parallel::PartitionInfo& part,
    parallel::CommBuffers* bufs,
    const parallel::Reduction* reduction,
    const AleRequest& request,
    const CellRegime* d_cell_regime);

RezoneResult run_axis_variational_projection(
    core::State& state,
    const core::Config& cfg,
    const parallel::PartitionInfo& part,
    parallel::CommBuffers* bufs,
    const parallel::Reduction* reduction,
    const AleRequest& request,
    const CellRegime* d_cell_regime);

RezoneResult run_boundary_patch_projection(
    core::State& state,
    const core::Config& cfg,
    const parallel::PartitionInfo& part,
    parallel::CommBuffers* bufs,
    const parallel::Reduction* reduction,
    const AleRequest& request);

RezoneResult run_interior_multi_node_projection(
    core::State& state,
    const core::Config& cfg,
    const parallel::PartitionInfo& part,
    parallel::CommBuffers* bufs,
    const parallel::Reduction* reduction,
    const AleRequest& request);

}  // namespace tenryu::hydro::ale
