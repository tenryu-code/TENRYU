#pragma once

#include <limits>

namespace tenryu::core {
struct Config;
}

namespace tenryu::coupling {

struct StagedDtResult {
  double edge_av_dt = std::numeric_limits<double>::infinity();
  double crossing_dt = std::numeric_limits<double>::infinity();
  double min_dt = std::numeric_limits<double>::infinity();
  int binding_cell = -1;
};

StagedDtResult evaluate_staged_hydro_dt(
    const int* csr_offsets,
    const int* csr_indices,
    const unsigned char* nverts,
    int n_cells,
    int n_nodes,
    const double* node_r,
    const double* node_z,
    const double* velocity_r,
    const double* velocity_z,
    const core::Config& cfg);

}  // namespace tenryu::coupling
