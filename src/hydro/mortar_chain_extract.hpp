#pragma once

#include <cstdint>
#include <vector>

#include "hydro/mortar_slide_line.hpp"

namespace tenryu::hydro {

struct BoundaryChain {
  std::vector<int> node_ids;
  std::vector<double> r;
  std::vector<double> z;
};

BoundaryChain mortar_extract_boundary_chain(
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const std::uint8_t* cell_nverts,
    int n_cells,
    int corner_stride,
    int n_nodes,
    const std::uint8_t* node_flags,
    std::uint8_t node_flag_mask,
    const double* node_r,
    const double* node_z);

MortarSlideLine mortar_build_from_chains(const BoundaryChain& master,
                                         const BoundaryChain& slave);

}  // namespace tenryu::hydro
