#pragma once

#include <cmath>
#include <limits>

#include "core/config.hpp"
#include "core/error.hpp"

namespace tenryu::hydro::button_morph {

// Clean-room duplicate of the private multiblock indexing arithmetic in
// mesh/mesh.cu. That translation unit is bitwise-load-bearing, so this header
// deliberately does not include or modify it; the topology-consistency twin
// test locks these formulas to the constructed mesh instead.
struct ButtonIndexing {
  int n_c = 0;
  int n_b = 0;
  int ntheta = 0;
  int nr_shell = 0;
  int n_cells_core = 0;
  int n_cells_bridge = 0;
  int n_cells_shell = 0;
  int n_cells_total = 0;
  int n_nodes_core = 0;
  int n_nodes_bridge_interior = 0;
  int n_nodes_shell = 0;
  int n_nodes_total = 0;
  int bridge_cell_offset = 0;
  int shell_cell_offset = 0;
  int bridge_interior_node_offset = 0;
  int shell_node_offset = 0;
  double r_c = 0.0;
  double r_match = 0.0;
  double s_max = 0.0;
};

inline ButtonIndexing make_button_indexing(
    const tenryu::core::Config::MeshConfig& cfg) {
  TENRYU_ASSERT(
      cfg.topology_scheme ==
          tenryu::core::TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL,
      "button morph indexing requires multiblock_cart_core_polar_shell topology");

  ButtonIndexing idx;
  idx.n_c = cfg.multiblock_cart_core_n_c;
  idx.n_b = cfg.multiblock_cart_core_bridge_layers;
  idx.ntheta = 4 * idx.n_c;
  idx.nr_shell = cfg.nr;
  idx.r_c = cfg.multiblock_cart_core_r_c;
  idx.r_match = cfg.multiblock_cart_core_r_match;
  idx.s_max = cfg.spherical_polar_s_max;

  TENRYU_ASSERT(idx.n_c > 0, "multiblock N_c must be positive");
  TENRYU_ASSERT(idx.n_b > 0,
                "multiblock bridge layer count must be positive");
  TENRYU_ASSERT(idx.nr_shell > 0,
                "multiblock shell radial count must be positive");
  TENRYU_ASSERT(cfg.nz == idx.ntheta,
                "multiblock Mesh.nz must equal 4*Mesh.multiblock_cart_core_n_c");
  TENRYU_ASSERT(std::isfinite(idx.r_c) && idx.r_c > 0.0,
                "multiblock r_c must be finite and positive");
  TENRYU_ASSERT(std::isfinite(idx.r_match) &&
                    std::sqrt(2.0) * idx.r_c < idx.r_match,
                "multiblock r_match must exceed sqrt(2)*r_c");
  TENRYU_ASSERT(std::isfinite(idx.s_max) && idx.r_match < idx.s_max,
                "multiblock s_max must exceed r_match");

  const long long n_cells_core = 2LL * idx.n_c * idx.n_c;
  const long long n_cells_bridge = 4LL * idx.n_c * idx.n_b;
  const long long n_cells_shell =
      static_cast<long long>(idx.nr_shell) * (4LL * idx.n_c);
  const long long n_cells_total =
      n_cells_core + n_cells_bridge + n_cells_shell;
  const long long n_nodes_core =
      (static_cast<long long>(idx.n_c) + 1LL) * (2LL * idx.n_c + 1LL);
  const long long n_nodes_bridge_interior =
      (4LL * idx.n_c + 1LL) * (static_cast<long long>(idx.n_b) - 1LL);
  const long long n_nodes_shell =
      (static_cast<long long>(idx.nr_shell) + 1LL) * (4LL * idx.n_c + 1LL);
  const long long n_nodes_total =
      n_nodes_core + n_nodes_bridge_interior + n_nodes_shell;
  constexpr long long int_max =
      static_cast<long long>(std::numeric_limits<int>::max());

  TENRYU_ASSERT(n_cells_core > 0 && n_cells_core <= int_max,
                "multiblock core cell count exceeds int range");
  TENRYU_ASSERT(n_cells_bridge > 0 && n_cells_bridge <= int_max,
                "multiblock bridge cell count exceeds int range");
  TENRYU_ASSERT(n_cells_shell > 0 && n_cells_shell <= int_max,
                "multiblock shell cell count exceeds int range");
  TENRYU_ASSERT(n_cells_total > 0 && n_cells_total <= int_max,
                "multiblock total cell count exceeds int range");
  TENRYU_ASSERT(n_nodes_core > 0 && n_nodes_core <= int_max,
                "multiblock core node count exceeds int range");
  TENRYU_ASSERT(n_nodes_bridge_interior >= 0 &&
                    n_nodes_bridge_interior <= int_max,
                "multiblock bridge interior node count exceeds int range");
  TENRYU_ASSERT(n_nodes_shell > 0 && n_nodes_shell <= int_max,
                "multiblock shell node count exceeds int range");
  TENRYU_ASSERT(n_nodes_total > 0 && n_nodes_total <= int_max,
                "multiblock total node count exceeds int range");

  idx.n_cells_core = static_cast<int>(n_cells_core);
  idx.n_cells_bridge = static_cast<int>(n_cells_bridge);
  idx.n_cells_shell = static_cast<int>(n_cells_shell);
  idx.n_cells_total = static_cast<int>(n_cells_total);
  idx.n_nodes_core = static_cast<int>(n_nodes_core);
  idx.n_nodes_bridge_interior =
      static_cast<int>(n_nodes_bridge_interior);
  idx.n_nodes_shell = static_cast<int>(n_nodes_shell);
  idx.n_nodes_total = static_cast<int>(n_nodes_total);

  TENRYU_ASSERT(idx.n_cells_total <= std::numeric_limits<int>::max() / 4,
                "multiblock CSR face/corner slot count exceeds int range");
  idx.bridge_cell_offset = idx.n_cells_core;
  idx.shell_cell_offset = idx.n_cells_core + idx.n_cells_bridge;
  idx.bridge_interior_node_offset = idx.n_nodes_core;
  idx.shell_node_offset = idx.n_nodes_core + idx.n_nodes_bridge_interior;
  return idx;
}

// Core node ranges: 0 <= i <= n_c and 0 <= j <= 2*n_c.
inline int core_node_id(const ButtonIndexing& idx, const int i, const int j) {
  TENRYU_ASSERT(i >= 0 && i <= idx.n_c, "core node i out of range");
  TENRYU_ASSERT(j >= 0 && j <= 2 * idx.n_c,
                "core node j out of range");
  return i * (2 * idx.n_c + 1) + j;
}

// Bridge-interior node ranges: 0 < layer < n_b and 0 <= k <= ntheta.
inline int bridge_interior_node_id(const ButtonIndexing& idx,
                                   const int layer,
                                   const int k) {
  TENRYU_ASSERT(layer > 0 && layer < idx.n_b,
                "bridge interior layer out of range");
  TENRYU_ASSERT(k >= 0 && k <= idx.ntheta,
                "bridge node angular index out of range");
  return idx.bridge_interior_node_offset +
         (layer - 1) * (idx.ntheta + 1) + k;
}

// Shell node ranges: 0 <= q <= nr_shell and 0 <= k <= ntheta.
inline int shell_node_id(const ButtonIndexing& idx, const int q, const int k) {
  TENRYU_ASSERT(q >= 0 && q <= idx.nr_shell,
                "shell node radial index out of range");
  TENRYU_ASSERT(k >= 0 && k <= idx.ntheta,
                "shell node angular index out of range");
  return idx.shell_node_offset + q * (idx.ntheta + 1) + k;
}

}  // namespace tenryu::hydro::button_morph
