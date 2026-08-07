#pragma once

#include "core/config.hpp"
#include "core/field.hpp"
#include "core/state.hpp"

namespace tenryu::hydro::ale_diag {

void emit_identity_field_diag(const core::State& state,
                              const core::Config& cfg,
                              const core::NodeField1D& node_mass,
                              const char* stage,
                              double t_s,
                              double dt_s,
                              int rank,
                              const core::CellField1D* cs_override = nullptr);

void capture_and_emit_mover_post_hydro(const core::State& state,
                                       const core::Config& cfg,
                                       const core::NodeField1D& r_old,
                                       const core::NodeField1D& z_old,
                                       const core::NodeField1D& node_mass,
                                       double dt_s,
                                       double t_s,
                                       int rank);

void emit_mover_post_projection(const core::State& state,
                                const core::Config& cfg,
                                double dt_s,
                                double t_s);

}  // namespace tenryu::hydro::ale_diag
