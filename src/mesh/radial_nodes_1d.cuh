#pragma once

#include <vector>

#include "core/config.hpp"

namespace tenryu::mesh {

// Radial nodes of a 1D mesh exactly as create_mesh builds them (same
// expressions, same translation unit).
std::vector<double> build_1d_radial_nodes(const core::Config& cfg, int nr);

}  // namespace tenryu::mesh
