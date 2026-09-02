#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <thrust/device_vector.h>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/field.hpp"
#include "mesh/boundary_kind.hpp"
#include "mesh/mesh_geometry_result.hpp"
#include "mesh/ring_pages.hpp"

namespace tenryu::core {
struct State;
}

namespace tenryu::mesh {

enum class LogicalMesh2D : std::uint8_t {
  RectangularRZ = 0,
  SphericalPolarHalfplane = 1,
  PolarInBox = 2,
  ConeShell = 3,
};

enum class PolarCenterTreatment : std::uint8_t {
  Annular = 0,
  TriFan = 1,
  Button = 2,
};

inline const char* logical_mesh_2d_name(const LogicalMesh2D m) {
  switch (m) {
    case LogicalMesh2D::RectangularRZ:
      return "rectangular_rz";
    case LogicalMesh2D::SphericalPolarHalfplane:
      return "spherical_polar_halfplane";
    case LogicalMesh2D::PolarInBox:
      return "polar_in_box";
    case LogicalMesh2D::ConeShell:
      return "cone_shell";
  }
  return "unknown";
}

inline LogicalMesh2D parse_logical_mesh_2d(const std::string& s) {
  if (s == "rectangular_rz") {
    return LogicalMesh2D::RectangularRZ;
  }
  if (s == "spherical_polar_halfplane") {
    return LogicalMesh2D::SphericalPolarHalfplane;
  }
  if (s == "polar_in_box") {
    return LogicalMesh2D::PolarInBox;
  }
  if (s == "cone_shell") {
    return LogicalMesh2D::ConeShell;
  }
  throw std::invalid_argument("Unknown logical_mesh_2d: " + s);
}

inline const char* polar_center_treatment_name(const PolarCenterTreatment p) {
  switch (p) {
    case PolarCenterTreatment::Annular:
      return "annular";
    case PolarCenterTreatment::TriFan:
      return "tri_fan";
    case PolarCenterTreatment::Button:
      return "button";
  }
  return "unknown";
}

inline PolarCenterTreatment parse_polar_center_treatment(const std::string& s) {
  if (s == "annular") {
    return PolarCenterTreatment::Annular;
  }
  if (s == "tri_fan") {
    return PolarCenterTreatment::TriFan;
  }
  if (s == "button") {
    return PolarCenterTreatment::Button;
  }
  throw std::invalid_argument("Unknown polar_center_treatment: " + s);
}

enum NodeFlag : std::uint8_t {
  NODE_NONE = 0,
  NODE_BOUNDARY = 1u << 0,
  NODE_AXIS = 1u << 1,
  NODE_CENTER = 1u << 2,
  NODE_POLE_AXIS = 1u << 3,
  NODE_OUTER_PHYSICAL_BOUNDARY = 1u << 4,
  NODE_INNER_PHYSICAL_BOUNDARY = 1u << 5,
};

struct UniqueOrientedFace {
  int cell_a = -1;
  int cell_b = -1;
  int local_a = -1;
  int local_b = -1;
  std::int8_t bc_tag = 0;
};

enum class BlockRole : std::uint8_t {
  CENTRAL_CORE = 0,
  BRIDGE = 1,
  NORTH_FAN = 2,
  EAST_FAN = 3,
  SOUTH_FAN = 4,
  POLAR_SHELL = 5,
  POLAR_TIER = 6,
  TRANSITION_BELT = 7,
  CENTER_FAN = 8,
  PENTAGON_BELT = 9,
};

enum class BlockSide : std::uint8_t {
  I_MINUS = 0,
  I_PLUS = 1,
  J_MINUS = 2,
  J_PLUS = 3,
  COMPOSITE_OUTER = 4,
};

struct BlockInfo {
  BlockRole role = BlockRole::CENTRAL_CORE;
  int n_i_cells = 0;
  int n_j_cells = 0;
  int cell_begin = 0;
  int cell_count = 0;
  int owned_node_begin = 0;
  int owned_node_count = 0;
};

struct SeamInfo {
  int block_a = -1;
  BlockSide side_a = BlockSide::I_MINUS;
  int block_b = -1;
  BlockSide side_b = BlockSide::I_MINUS;
  int orientation = 1;
  int index_begin = 0;
  int index_count = 0;
};

struct ConeShellTipPort {
  // C4b consumes this inner-to-outer seam. Segment labels are
  // 0=TIP_CORNER_I, 1=END_NF, and 2=TIP_CORNER_O.
  std::vector<int> node_ids;
  std::vector<std::uint8_t> segment;
};

struct DendriteBlockRings {
  std::vector<int> outer_ring;
  std::vector<int> outer_adjacent_ring;
  std::vector<int> center_chain;
  std::vector<int> inner_adjacent_ring;
  std::vector<int> inner_ring;
};

struct MultiBlockTopology {
  std::vector<int> cell_block_id;
  std::vector<int> cell_id_stable;
  std::vector<int> cell_orientation_sign;
  std::vector<int> cell_node_csr_offsets;
  std::vector<int> cell_node_csr_indices;
  std::vector<int> face_adj_csr_offsets;
  std::vector<int> face_adj_csr_indices;
  std::vector<int> face_bc_tags;
  std::vector<BlockInfo> blocks;
  std::vector<SeamInfo> seams;
  ConeShellTipPort cone_shell_tip_port;
  std::vector<UniqueOrientedFace> unique_internal_faces;
  std::vector<UniqueOrientedFace> boundary_faces;
  // csw98 phase-line continuation topology (static; independent of
  // hydro_active). Index space: unique_internal_faces then
  // boundary_faces. Sentinels: -1 none (true corner), -2 axis mirror,
  // -3 multi-candidate (see the candidate CSR).
  std::vector<int> csw_line_edge_n0;
  std::vector<int> csw_line_edge_n1;
  std::vector<int> csw_line_prev_edge;
  std::vector<int> csw_line_next_edge;
  std::vector<std::int8_t> csw_line_prev_sign;
  std::vector<std::int8_t> csw_line_next_sign;
  std::vector<int> csw_line_cand_offsets;
  std::vector<int> csw_line_cand_edges;
  std::vector<std::int8_t> csw_line_cand_signs;
  thrust::device_vector<int> d_csw_line_edge_n0;
  thrust::device_vector<int> d_csw_line_edge_n1;
  thrust::device_vector<int> d_csw_line_prev_edge;
  thrust::device_vector<int> d_csw_line_next_edge;
  thrust::device_vector<std::int8_t> d_csw_line_prev_sign;
  thrust::device_vector<std::int8_t> d_csw_line_next_sign;
  thrust::device_vector<int> d_csw_line_cand_offsets;
  thrust::device_vector<int> d_csw_line_cand_edges;
  thrust::device_vector<std::int8_t> d_csw_line_cand_signs;
  thrust::device_vector<int> d_unique_face_cell_a;
  thrust::device_vector<int> d_unique_face_cell_b;
  thrust::device_vector<int> d_unique_face_local_a;
  thrust::device_vector<int> d_unique_face_local_b;
  thrust::device_vector<std::int8_t> d_unique_face_bc_tag;
  thrust::device_vector<int> d_face_adj_csr_offsets;
  thrust::device_vector<int> d_face_adj_csr_indices;
  thrust::device_vector<int> d_boundary_face_cell;
  thrust::device_vector<int> d_boundary_face_local;
  thrust::device_vector<std::int8_t> d_boundary_face_bc_tag;
  int block_count = 3;
  int n_cells_core = 0;
  int n_cells_bridge = 0;
  int n_cells_shell = 0;
  int n_nodes_core = 0;
  int n_nodes_bridge_interior = 0;
  int n_nodes_shell = 0;
  bool has_trifan_cap = false;
  int n_cap = 0;
  int n_cells_cap = 0;
  int n_nodes_cap = 0;
  bool has_polar_tier = false;
  double polar_tier_h_r = 0.0;
  double polar_tier_fan_radius = 0.0;
  std::vector<int> polar_tier_columns;
  std::vector<double> polar_tier_transition_radii;
  std::vector<DendriteBlockRings> dendrite_block_rings;
  std::vector<int> south_node_of;
};

struct MeshTopology {
  int nr = 0;
  int nz = 1;
  int n_cells = 0;
  int n_nodes = 0;
  int polar_in_box_j_tr = -1;
  int polar_in_box_j_br = -1;

  // linear indexing: cell(i,j)=i*nz+j, node(i,j)=i*(nz+1)+j.
  std::vector<std::uint8_t> node_flags;
  bool general_polygonal = false;  // set when a ReALE tessellation is installed: cell/node layouts are CSR-general and structured block indexing is not meaningful
  std::optional<MultiBlockTopology> multiblock;

  [[nodiscard]] int cell_index(const int i, const int j) const {
    return i * nz + j;
  }

  [[nodiscard]] int node_index(const int i, const int j) const {
    return i * (nz + 1) + j;
  }

  [[nodiscard]] int radial_edge_index(const int i, const int j) const {
    return i * (nz + 1) + j;
  }

  [[nodiscard]] int angular_edge_index(const int i, const int j) const {
    return nr * (nz + 1) + i * nz + j;
  }

  [[nodiscard]] int n_edges() const {
    return nr * (nz + 1) + (nr + 1) * nz;
  }
};

struct ButtonCenterTopology {
  bool enabled = false;
  int outer_node_ring = 2;
};

inline int mesh_topo_checked_int_count(const long long n, const char* label) {
  TENRYU_ASSERT(n > 0, std::string(label) + " must be positive");
  TENRYU_ASSERT(n <= static_cast<long long>(std::numeric_limits<int>::max()),
                std::string(label) + " exceeds int range");
  return static_cast<int>(n);
}

inline int mesh_topo_checked_nonnegative_int_count(const long long n,
                                                   const char* label) {
  TENRYU_ASSERT(n >= 0, std::string(label) + " must be non-negative");
  TENRYU_ASSERT(n <= static_cast<long long>(std::numeric_limits<int>::max()),
                std::string(label) + " exceeds int range");
  return static_cast<int>(n);
}

#ifdef __CUDACC__
#define TENRYU_MESH_HOST_DEVICE __host__ __device__
#else
#define TENRYU_MESH_HOST_DEVICE
#endif

TENRYU_MESH_HOST_DEVICE inline bool mesh_topo_is_single_block(
    const tenryu::core::TopologyScheme scheme) noexcept {
  return scheme == tenryu::core::TopologyScheme::SINGLE_BLOCK;
}

TENRYU_MESH_HOST_DEVICE inline bool mesh_topo_is_multiblock(
    const tenryu::core::TopologyScheme scheme) noexcept {
  return scheme ==
             tenryu::core::TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL ||
         scheme ==
             tenryu::core::TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_5BLOCK ||
         scheme == tenryu::core::TopologyScheme::
                       MULTIBLOCK_HALF_BUTTERFLY_TRIFAN_CAP_5BLOCK ||
         scheme ==
             tenryu::core::TopologyScheme::MULTIBLOCK_POLAR_TIER ||
         scheme == tenryu::core::TopologyScheme::
                       MULTIBLOCK_POLAR_TIER_CART_CENTER ||
         scheme == tenryu::core::TopologyScheme::CONE_SHELL_SPINE ||
         scheme == tenryu::core::TopologyScheme::PENTAGON_BELT_SHELL;
}

TENRYU_MESH_HOST_DEVICE inline bool mesh_topo_has_trifan_cap(
    const tenryu::core::TopologyScheme scheme) noexcept {
  return scheme == tenryu::core::TopologyScheme::
                       MULTIBLOCK_HALF_BUTTERFLY_TRIFAN_CAP_5BLOCK;
}

TENRYU_MESH_HOST_DEVICE inline bool mesh_topo_has_polar_tier(
    const tenryu::core::TopologyScheme scheme) noexcept {
  return scheme ==
         tenryu::core::TopologyScheme::MULTIBLOCK_POLAR_TIER;
}

TENRYU_MESH_HOST_DEVICE inline bool mesh_topo_has_polar_tier_cart_center(
    const tenryu::core::TopologyScheme scheme) noexcept {
  return scheme == tenryu::core::TopologyScheme::
                       MULTIBLOCK_POLAR_TIER_CART_CENTER;
}

TENRYU_MESH_HOST_DEVICE inline bool mesh_topo_polar_tier_family(
    const tenryu::core::TopologyScheme scheme) noexcept {
  return mesh_topo_has_polar_tier(scheme) ||
         mesh_topo_has_polar_tier_cart_center(scheme);
}

constexpr int kMeshTopoCellStorageSlots = 4;
constexpr int kMeshTopoCellStorageSlotsMax = 8;
// General-polygonal (reale_v2) storage cap; fixed meshes stay at kMeshTopoCellStorageSlotsMax.
constexpr int kMeshTopoCellStorageSlotsMaxGeneral = 16;
inline constexpr int kConeShellSpineBlockCount = 11;

/// Active-slot topology contract:
///   - Cell storage uses the mesh's fixed corner_stride (4 or 8).
///   - If cell_nverts is absent, all four slots are active.
///   - If cell_nverts[c] == 3, vertex slots 0,1,2 are active and slot 3 is
///     inactive storage.
///   - For triangle faces, local face 0 is inactive; active local faces
///     1,2,3 map to canonical polygon edges (1,2), (0,1), (2,0).
TENRYU_MESH_HOST_DEVICE inline int mesh_topo_cell_active_nverts(
    const std::uint8_t* cell_nverts,
    const int cell) noexcept {
  if (cell_nverts != nullptr && cell_nverts[cell] >= 3U &&
      cell_nverts[cell] <= kMeshTopoCellStorageSlotsMaxGeneral) {
    return static_cast<int>(cell_nverts[cell]);
  }
  return kMeshTopoCellStorageSlots;
}

TENRYU_MESH_HOST_DEVICE inline int mesh_topo_local_face_canonical_edge0(
    const int active_nverts,
    const int local_face) noexcept {
  if (active_nverts == 3) {
    if (local_face == 1) {
      return 1;
    }
    if (local_face == 2) {
      return 0;
    }
    if (local_face == 3) {
      return 2;
    }
    return -1;
  }
  if (active_nverts == 4) {
    if (local_face > 3) {
      return -1;
    }
    return (local_face == 0) ? 3
                             : ((local_face == 1) ? 1
                                                  : ((local_face == 2) ? 0 : 2));
  }
  if (local_face >= active_nverts) {
    return -1;
  }
  return local_face;
}

TENRYU_MESH_HOST_DEVICE inline bool mesh_topo_active_local_face_corners(
    const int active_nverts,
    const int local_face,
    int* corner0,
    int* corner1) noexcept {
  const int edge0 =
      mesh_topo_local_face_canonical_edge0(active_nverts, local_face);
  if (edge0 < 0) {
    return false;
  }
  *corner0 = edge0;
  *corner1 = (edge0 + 1) % active_nverts;
  return true;
}

TENRYU_MESH_HOST_DEVICE inline bool mesh_topo_local_face_is_active(
    const int active_nverts,
    const int local_face) noexcept {
  int corner0 = 0;
  int corner1 = 0;
  return mesh_topo_active_local_face_corners(active_nverts, local_face,
                                             &corner0, &corner1);
}

#undef TENRYU_MESH_HOST_DEVICE

inline int mesh_topo_cell_active_nverts(
    const std::vector<std::uint8_t>& cell_nverts,
    const int cell) {
  if (cell_nverts.empty()) {
    return kMeshTopoCellStorageSlots;
  }
  TENRYU_ASSERT(cell >= 0 &&
                    static_cast<std::size_t>(cell) < cell_nverts.size(),
                "cell_nverts lookup cell out of range");
  const std::uint8_t nverts = cell_nverts[static_cast<std::size_t>(cell)];
  TENRYU_ASSERT(
      nverts >= 3U && nverts <= kMeshTopoCellStorageSlotsMaxGeneral,
      "cell_nverts entries must be in [3, kMeshTopoCellStorageSlotsMaxGeneral]");
  return static_cast<int>(nverts);
}

inline bool mesh_topo_is_single_block(
    const tenryu::core::Config::MeshConfig& cfg) noexcept {
  return mesh_topo_is_single_block(cfg.topology_scheme);
}

inline bool mesh_topo_is_multiblock(
    const tenryu::core::Config::MeshConfig& cfg) noexcept {
  return mesh_topo_is_multiblock(cfg.topology_scheme);
}

inline bool mesh_topo_has_trifan_cap(
    const tenryu::core::Config::MeshConfig& cfg) noexcept {
  return mesh_topo_has_trifan_cap(cfg.topology_scheme);
}

inline bool mesh_topo_has_polar_tier(
    const tenryu::core::Config::MeshConfig& cfg) noexcept {
  return mesh_topo_has_polar_tier(cfg.topology_scheme);
}

inline bool mesh_topo_has_polar_tier_cart_center(
    const tenryu::core::Config::MeshConfig& cfg) noexcept {
  return mesh_topo_has_polar_tier_cart_center(cfg.topology_scheme);
}

inline bool mesh_topo_polar_tier_family(
    const tenryu::core::Config::MeshConfig& cfg) noexcept {
  return mesh_topo_polar_tier_family(cfg.topology_scheme);
}

inline bool mesh_topo_polar_tier_cart_center_has_trifan_cap(
    const tenryu::core::Config::MeshConfig& cfg) noexcept {
  return mesh_topo_has_polar_tier_cart_center(cfg) &&
         cfg.polar_tier_center_kind == "trifan_cap";
}

inline int mesh_topo_n_kept_polar_nodes(const MultiBlockTopology& mb) {
  int bridge_count = 0;
  int kept_polar_nodes = -1;
  for (const BlockInfo& block : mb.blocks) {
    if (block.role != BlockRole::BRIDGE) {
      continue;
    }
    if (bridge_count == 0) {
      kept_polar_nodes = block.owned_node_begin;
    }
    ++bridge_count;
  }
  TENRYU_ASSERT(
      bridge_count == 1,
      "polar-tier cart-center topology requires exactly one BRIDGE block");
  TENRYU_ASSERT(kept_polar_nodes >= 0,
                "polar-tier cart-center kept-node prefix must be non-negative");
  return kept_polar_nodes;
}

// AW axis discipline covers kept + bridge-owned nodes; the center-block
// interior is BBSW-or-aggregate territory.
inline int mesh_topo_n_aw_axis_prefix_nodes(const MultiBlockTopology& mb) {
  int central_core_count = 0;
  int aw_axis_prefix_nodes = -1;
  for (const BlockInfo& block : mb.blocks) {
    if (block.role != BlockRole::CENTRAL_CORE) {
      continue;
    }
    if (central_core_count == 0) {
      aw_axis_prefix_nodes = block.owned_node_begin;
    }
    ++central_core_count;
  }
  TENRYU_ASSERT(
      central_core_count == 1,
      "polar-tier cart-center topology requires exactly one CENTRAL_CORE block");
  TENRYU_ASSERT(aw_axis_prefix_nodes >= 0,
                "polar-tier cart-center AW-axis prefix must be non-negative");
  return aw_axis_prefix_nodes;
}

inline int mesh_topo_n_cap_with_n_c(const int n_c) {
  TENRYU_ASSERT(n_c > 0, "Mesh.multiblock_cart_core_n_c must be positive");
  return n_c;
}

inline int mesh_topo_n_cap(const tenryu::core::Config::MeshConfig& cfg) {
  return mesh_topo_n_cap_with_n_c(cfg.multiblock_cart_core_n_c);
}

inline int mesh_topo_n_cells_cap_with_n_c(const int n_c) {
  const int n_cap = mesh_topo_n_cap_with_n_c(n_c);
  TENRYU_ASSERT(n_c > 0, "Mesh.multiblock_cart_core_n_c must be positive");
  return mesh_topo_checked_int_count(4LL * n_c * n_cap,
                                     "multiblock cap cell count");
}

inline int mesh_topo_n_cells_cap(
    const tenryu::core::Config::MeshConfig& cfg) {
  return mesh_topo_n_cells_cap_with_n_c(cfg.multiblock_cart_core_n_c);
}

inline int mesh_topo_n_nodes_cap_with_n_c(const int n_c) {
  const int n_cap = mesh_topo_n_cap_with_n_c(n_c);
  TENRYU_ASSERT(n_c > 0, "Mesh.multiblock_cart_core_n_c must be positive");
  return mesh_topo_checked_int_count(
      1LL + static_cast<long long>(n_cap) * (4LL * n_c + 1LL),
      "multiblock cap node count");
}

inline int mesh_topo_n_nodes_cap(
    const tenryu::core::Config::MeshConfig& cfg) {
  return mesh_topo_n_nodes_cap_with_n_c(cfg.multiblock_cart_core_n_c);
}

inline int mesh_topo_n_nodes_cap_not_shared_with_bridge_with_n_c(
    const int n_c) {
  const int n_cap = mesh_topo_n_cap_with_n_c(n_c);
  return mesh_topo_checked_int_count(
      1LL + static_cast<long long>(n_cap - 1) * (4LL * n_c + 1LL),
      "multiblock cap non-shared node count");
}

inline int mesh_topo_n_cells_core_with_n_c(
    const tenryu::core::Config::MeshConfig& cfg,
    const int n_c) {
  if (mesh_topo_has_polar_tier(cfg)) {
    return cfg.polar_tier_fan_sectors;
  }
  if (mesh_topo_has_trifan_cap(cfg)) {
    return mesh_topo_n_cells_cap_with_n_c(n_c);
  }
  if (mesh_topo_polar_tier_cart_center_has_trifan_cap(cfg)) {
    return mesh_topo_n_cells_cap_with_n_c(n_c);
  }
  TENRYU_ASSERT(n_c > 0, "Mesh.multiblock_cart_core_n_c must be positive");
  return mesh_topo_checked_int_count(2LL * n_c * n_c,
                                     "multiblock core cell count");
}

inline int mesh_topo_n_cells_core(
    const tenryu::core::Config::MeshConfig& cfg) {
  return mesh_topo_n_cells_core_with_n_c(
      cfg, cfg.multiblock_cart_core_n_c);
}

inline int mesh_topo_n_cells_bridge_with_n_c(
    const tenryu::core::Config::MeshConfig& cfg,
    const int n_c) {
  if (mesh_topo_has_polar_tier(cfg)) {
    const auto layout = tenryu::core::make_polar_tier_layout(cfg);
    const long long shell_cells =
        cfg.shell_polar_cap_dendrite
            ? layout.shell_n_cells
            : static_cast<long long>(cfg.nr) * cfg.nz;
    return mesh_topo_checked_int_count(
        layout.n_cells - shell_cells - cfg.polar_tier_fan_sectors,
        "polar-tier tier and belt cell count");
  }
  const int n_b = cfg.multiblock_cart_core_bridge_layers;
  TENRYU_ASSERT(n_c > 0, "Mesh.multiblock_cart_core_n_c must be positive");
  TENRYU_ASSERT(n_b > 0,
                "Mesh.multiblock_cart_core_bridge_layers must be positive");
  return mesh_topo_checked_int_count(4LL * n_c * n_b,
                                     "multiblock bridge cell count");
}

inline int mesh_topo_n_cells_bridge(
    const tenryu::core::Config::MeshConfig& cfg) {
  return mesh_topo_n_cells_bridge_with_n_c(
      cfg, cfg.multiblock_cart_core_n_c);
}

inline int mesh_topo_n_cells_shell_with_n_c(
    const tenryu::core::Config::MeshConfig& cfg,
    const int n_c) {
  if (mesh_topo_has_polar_tier(cfg)) {
    if (cfg.shell_polar_cap_dendrite) {
      const auto layout = tenryu::core::make_polar_tier_layout(cfg);
      return mesh_topo_checked_int_count(
          layout.shell_n_cells,
          "polar-tier shell cell count");
    }
    return mesh_topo_checked_int_count(
        static_cast<long long>(cfg.nr) * cfg.nz,
        "polar-tier shell cell count");
  }
  TENRYU_ASSERT(cfg.nr > 0, "Mesh.nr must be positive");
  TENRYU_ASSERT(n_c > 0, "Mesh.multiblock_cart_core_n_c must be positive");
  return mesh_topo_checked_int_count(
      static_cast<long long>(cfg.nr) * (4LL * n_c),
      "multiblock shell cell count");
}

inline int mesh_topo_n_cells_shell(
    const tenryu::core::Config::MeshConfig& cfg) {
  return mesh_topo_n_cells_shell_with_n_c(
      cfg, cfg.multiblock_cart_core_n_c);
}

inline int mesh_topo_cone_shell_corner_layers(const int face_layers,
                                              const int end_layers) {
  TENRYU_ASSERT(
      face_layers == end_layers,
      "cone_shell corner closure requires equal face and end layer counts");
  return face_layers;
}

inline int mesh_topo_cone_shell_corner_layer_sum(
    const tenryu::core::Config::MeshConfig& cfg) {
  return mesh_topo_cone_shell_corner_layers(
             cfg.cone_shell_outer_vac_layers,
             cfg.cone_shell_end_vac_layers) +
         mesh_topo_cone_shell_corner_layers(
             cfg.cone_shell_inner_vac_layers,
             cfg.cone_shell_end_vac_layers);
}

inline int mesh_topo_cone_shell_tip_port_unique_nodes(
    const tenryu::core::Config::MeshConfig& cfg) {
  const int outer_corner_layers = mesh_topo_cone_shell_corner_layers(
      cfg.cone_shell_outer_vac_layers, cfg.cone_shell_end_vac_layers);
  const int inner_corner_layers = mesh_topo_cone_shell_corner_layers(
      cfg.cone_shell_inner_vac_layers, cfg.cone_shell_end_vac_layers);
  return cfg.cone_shell_inner_vac_layers +
         cfg.cone_shell_outer_vac_layers +
         2 * cfg.cone_shell_end_vac_layers - 2 * inner_corner_layers -
         2 * outer_corner_layers + cfg.nz + 5;
}

inline int mesh_topo_cone_shell_tip_fill_west_segments(
    const tenryu::core::Config::MeshConfig& cfg) {
  return cfg.cone_shell_cavity_cells;
}

inline int mesh_topo_cone_shell_tip_fill_mid_segments(
    const tenryu::core::Config::MeshConfig& cfg) {
  return cfg.nz + 2;
}

inline int mesh_topo_cone_shell_tip_fill_east_segments(
    const tenryu::core::Config::MeshConfig& cfg) {
  return cfg.cone_shell_exterior_cells;
}

inline long long mesh_topo_cone_shell_tip_fill_cell_count(
    const tenryu::core::Config::MeshConfig& cfg) {
  const long long layers = cfg.cone_shell_tip_fill_layers;
  return layers *
             (mesh_topo_cone_shell_tip_fill_west_segments(cfg) +
              mesh_topo_cone_shell_tip_fill_east_segments(cfg)) +
         (layers - 1LL) *
             mesh_topo_cone_shell_tip_fill_mid_segments(cfg);
}

inline long long mesh_topo_cone_shell_tip_fill_owned_node_count(
    const tenryu::core::Config::MeshConfig& cfg) {
  const long long layers = cfg.cone_shell_tip_fill_layers;
  const long long west_nodes =
      (mesh_topo_cone_shell_tip_fill_west_segments(cfg) + 1LL) * layers -
      1LL;
  const long long east_nodes =
      (mesh_topo_cone_shell_tip_fill_east_segments(cfg) + 1LL) * layers -
      1LL;
  const long long mid_nodes =
      (mesh_topo_cone_shell_tip_fill_mid_segments(cfg) - 1LL) *
      (layers - 1LL);
  return west_nodes + mid_nodes + east_nodes;
}

inline int mesh_topo_pentagon_belt_ring_zones(
    const tenryu::core::Config::MeshConfig& cfg,
    const int node_ring) {
  TENRYU_ASSERT(node_ring >= 0 && node_ring <= cfg.nr,
                "pentagon-belt node ring out of range");
  TENRYU_ASSERT(cfg.nz > 0, "Mesh.nz must be positive");
  int belts_at_or_outside = 0;
  for (const int belt_layer : cfg.pentagon_belt_layers) {
    if (belt_layer >= node_ring) {
      ++belts_at_or_outside;
    }
  }
  return cfg.nz >> belts_at_or_outside;
}

inline int mesh_topo_n_cells_total(
    const tenryu::core::Config::MeshConfig& cfg) {
  if (cfg.topology_scheme == tenryu::core::TopologyScheme::SINGLE_BLOCK) {
    return mesh_topo_checked_int_count(
        static_cast<long long>(cfg.nr) * static_cast<long long>(cfg.nz),
        "single-block cell count");
  }
  if (cfg.topology_scheme ==
      tenryu::core::TopologyScheme::CONE_SHELL_SPINE) {
    return mesh_topo_checked_int_count(
        static_cast<long long>(cfg.nr) * cfg.nz +
            static_cast<long long>(cfg.nr) *
                (cfg.cone_shell_outer_vac_layers +
                 cfg.cone_shell_inner_vac_layers) +
            static_cast<long long>(cfg.nz) *
                cfg.cone_shell_end_vac_layers +
            2LL * mesh_topo_cone_shell_corner_layer_sum(cfg) - 2LL +
            static_cast<long long>(cfg.nr) *
                (cfg.cone_shell_cavity_cells +
                 cfg.cone_shell_exterior_cells) +
            mesh_topo_cone_shell_tip_fill_cell_count(cfg),
        "cone-shell-spine cell count");
  }
  if (mesh_topo_has_polar_tier(cfg)) {
    return mesh_topo_checked_int_count(
        tenryu::core::make_polar_tier_layout(cfg).n_cells,
        "polar-tier total cell count");
  }
  if (mesh_topo_has_polar_tier_cart_center(cfg)) {
    tenryu::core::PolarTierTruncation truncation;
    const auto layout = tenryu::core::make_polar_tier_layout_truncated(
        cfg, cfg.polar_tier_cart_cut_ring, &truncation);
    const int n_c = truncation.n_theta_cut / 4;
    return mesh_topo_checked_int_count(
        layout.n_cells +
            static_cast<long long>(
                mesh_topo_n_cells_core_with_n_c(cfg, n_c)) +
            static_cast<long long>(
                mesh_topo_n_cells_bridge_with_n_c(cfg, n_c)),
        "polar-tier cart-center total cell count");
  }
  if (cfg.topology_scheme ==
      tenryu::core::TopologyScheme::PENTAGON_BELT_SHELL) {
    long long n_cells = 0;
    for (int cell_layer = 0; cell_layer < cfg.nr; ++cell_layer) {
      n_cells += mesh_topo_pentagon_belt_ring_zones(cfg, cell_layer);
    }
    return mesh_topo_checked_int_count(n_cells,
                                       "pentagon-belt total cell count");
  }
  TENRYU_ASSERT(mesh_topo_is_multiblock(cfg.topology_scheme),
                "Mesh.topology_scheme must be single_block or multiblock");

  const long long n_cells =
      static_cast<long long>(mesh_topo_n_cells_core(cfg)) +
      static_cast<long long>(mesh_topo_n_cells_bridge(cfg)) +
      static_cast<long long>(mesh_topo_n_cells_shell(cfg));
  return mesh_topo_checked_int_count(n_cells, "multiblock total cell count");
}

inline int mesh_topo_n_nodes_core_with_n_c(
    const tenryu::core::Config::MeshConfig& cfg,
    const int n_c) {
  if (mesh_topo_has_polar_tier(cfg)) {
    return 1;
  }
  if (mesh_topo_has_trifan_cap(cfg)) {
    return mesh_topo_n_nodes_cap_with_n_c(n_c);
  }
  if (mesh_topo_polar_tier_cart_center_has_trifan_cap(cfg)) {
    return mesh_topo_n_nodes_cap_not_shared_with_bridge_with_n_c(n_c);
  }
  TENRYU_ASSERT(n_c > 0, "Mesh.multiblock_cart_core_n_c must be positive");
  return mesh_topo_checked_int_count(
      (static_cast<long long>(n_c) + 1LL) * (2LL * n_c + 1LL),
      "multiblock core node count");
}

inline int mesh_topo_n_nodes_core(
    const tenryu::core::Config::MeshConfig& cfg) {
  return mesh_topo_n_nodes_core_with_n_c(
      cfg, cfg.multiblock_cart_core_n_c);
}

inline int mesh_topo_n_nodes_bridge_interior_with_n_c(
    const tenryu::core::Config::MeshConfig& cfg,
    const int n_c) {
  if (mesh_topo_has_polar_tier(cfg)) {
    const auto layout = tenryu::core::make_polar_tier_layout(cfg);
    const long long shell_nodes =
        cfg.shell_polar_cap_dendrite
            ? layout.shell_n_nodes
            : (static_cast<long long>(cfg.nr) + 1LL) * (cfg.nz + 1LL);
    return mesh_topo_checked_nonnegative_int_count(
        layout.n_nodes - shell_nodes - 1LL,
        "polar-tier tier and belt node count");
  }
  const int n_b = cfg.multiblock_cart_core_bridge_layers;
  TENRYU_ASSERT(n_c > 0, "Mesh.multiblock_cart_core_n_c must be positive");
  TENRYU_ASSERT(n_b > 0,
                "Mesh.multiblock_cart_core_bridge_layers must be positive");
  return mesh_topo_checked_nonnegative_int_count(
      (4LL * n_c + 1LL) * (static_cast<long long>(n_b) - 1LL),
      "multiblock bridge interior node count");
}

inline int mesh_topo_n_nodes_bridge_interior(
    const tenryu::core::Config::MeshConfig& cfg) {
  return mesh_topo_n_nodes_bridge_interior_with_n_c(
      cfg, cfg.multiblock_cart_core_n_c);
}

inline int mesh_topo_n_nodes_shell_with_n_c(
    const tenryu::core::Config::MeshConfig& cfg,
    const int n_c) {
  if (mesh_topo_has_polar_tier(cfg)) {
    if (cfg.shell_polar_cap_dendrite) {
      const auto layout = tenryu::core::make_polar_tier_layout(cfg);
      return mesh_topo_checked_int_count(
          layout.shell_n_nodes,
          "polar-tier shell node count");
    }
    return mesh_topo_checked_int_count(
        (static_cast<long long>(cfg.nr) + 1LL) * (cfg.nz + 1LL),
        "polar-tier shell node count");
  }
  TENRYU_ASSERT(cfg.nr > 0, "Mesh.nr must be positive");
  TENRYU_ASSERT(n_c > 0, "Mesh.multiblock_cart_core_n_c must be positive");
  return mesh_topo_checked_int_count(
      (static_cast<long long>(cfg.nr) + 1LL) * (4LL * n_c + 1LL),
      "multiblock shell node count");
}

inline int mesh_topo_n_nodes_shell(
    const tenryu::core::Config::MeshConfig& cfg) {
  return mesh_topo_n_nodes_shell_with_n_c(
      cfg, cfg.multiblock_cart_core_n_c);
}

inline int mesh_topo_n_nodes_total(
    const tenryu::core::Config::MeshConfig& cfg) {
  if (cfg.topology_scheme == tenryu::core::TopologyScheme::SINGLE_BLOCK) {
    return mesh_topo_checked_int_count(
        (static_cast<long long>(cfg.nr) + 1LL) *
            (static_cast<long long>(cfg.nz) + 1LL),
        "single-block node count");
  }
  if (cfg.topology_scheme ==
      tenryu::core::TopologyScheme::CONE_SHELL_SPINE) {
    return mesh_topo_checked_int_count(
        (static_cast<long long>(cfg.nr) + 1LL) *
                (static_cast<long long>(cfg.nz) + 1LL) +
            static_cast<long long>(cfg.cone_shell_outer_vac_layers) *
                (static_cast<long long>(cfg.nr) + 1LL) +
            static_cast<long long>(cfg.cone_shell_inner_vac_layers) *
                (static_cast<long long>(cfg.nr) + 1LL) +
            static_cast<long long>(cfg.cone_shell_end_vac_layers) *
                (static_cast<long long>(cfg.nz) + 1LL) +
            mesh_topo_cone_shell_corner_layer_sum(cfg) +
            static_cast<long long>(cfg.nr + 1) *
                (cfg.cone_shell_cavity_cells +
                 cfg.cone_shell_exterior_cells) +
            mesh_topo_cone_shell_tip_fill_owned_node_count(cfg),
        "cone-shell-spine node count");
  }
  if (mesh_topo_has_polar_tier(cfg)) {
    return mesh_topo_checked_int_count(
        tenryu::core::make_polar_tier_layout(cfg).n_nodes,
        "polar-tier total node count");
  }
  if (mesh_topo_has_polar_tier_cart_center(cfg)) {
    tenryu::core::PolarTierTruncation truncation;
    const auto layout = tenryu::core::make_polar_tier_layout_truncated(
        cfg, cfg.polar_tier_cart_cut_ring, &truncation);
    const int n_c = truncation.n_theta_cut / 4;
    const long long cap_bridge_inner_ring_nodes =
        mesh_topo_polar_tier_cart_center_has_trifan_cap(cfg)
            ? static_cast<long long>(truncation.n_theta_cut) + 1LL
            : 0LL;
    return mesh_topo_checked_int_count(
        layout.n_nodes +
            static_cast<long long>(
                mesh_topo_n_nodes_core_with_n_c(cfg, n_c)) +
            static_cast<long long>(
                mesh_topo_n_nodes_bridge_interior_with_n_c(cfg, n_c)) +
            cap_bridge_inner_ring_nodes,
        "polar-tier cart-center total node count");
  }
  if (cfg.topology_scheme ==
      tenryu::core::TopologyScheme::PENTAGON_BELT_SHELL) {
    long long n_nodes = 0;
    for (int node_ring = 0; node_ring <= cfg.nr; ++node_ring) {
      n_nodes +=
          static_cast<long long>(
              mesh_topo_pentagon_belt_ring_zones(cfg, node_ring)) +
          1LL;
    }
    return mesh_topo_checked_int_count(n_nodes,
                                       "pentagon-belt total node count");
  }
  TENRYU_ASSERT(mesh_topo_is_multiblock(cfg.topology_scheme),
                "Mesh.topology_scheme must be single_block or multiblock");

  const long long n_nodes =
      static_cast<long long>(mesh_topo_n_nodes_core(cfg)) +
      static_cast<long long>(mesh_topo_n_nodes_bridge_interior(cfg)) +
      static_cast<long long>(mesh_topo_n_nodes_shell(cfg));
  return mesh_topo_checked_int_count(n_nodes, "multiblock total node count");
}

inline MultiBlockTopology mesh_topo_make_empty_multiblock_topology(
    const tenryu::core::Config::MeshConfig& cfg) {
  TENRYU_ASSERT(mesh_topo_is_multiblock(cfg.topology_scheme),
                "Mesh.topology_scheme must be a multiblock topology");
  MultiBlockTopology mb;
  if (cfg.topology_scheme ==
      tenryu::core::TopologyScheme::CONE_SHELL_SPINE) {
    mb.block_count = kConeShellSpineBlockCount;
    return mb;
  }
  if (mesh_topo_has_polar_tier(cfg)) {
    const auto layout = tenryu::core::make_polar_tier_layout(cfg);
    mb.block_count = layout.block_count;
    mb.n_cells_core = mesh_topo_n_cells_core(cfg);
    mb.n_cells_bridge = mesh_topo_n_cells_bridge(cfg);
    mb.n_cells_shell = mesh_topo_n_cells_shell(cfg);
    mb.n_nodes_core = mesh_topo_n_nodes_core(cfg);
    mb.n_nodes_bridge_interior =
        mesh_topo_n_nodes_bridge_interior(cfg);
    mb.n_nodes_shell = mesh_topo_n_nodes_shell(cfg);
    mb.has_polar_tier = true;
    mb.polar_tier_h_r = layout.h_r;
    mb.polar_tier_fan_radius = layout.fan_radius;
    mb.polar_tier_columns = layout.tier_columns;
    mb.polar_tier_transition_radii = layout.transition_radii;
    return mb;
  }
  if (mesh_topo_has_polar_tier_cart_center(cfg)) {
    tenryu::core::PolarTierTruncation truncation;
    const auto layout = tenryu::core::make_polar_tier_layout_truncated(
        cfg, cfg.polar_tier_cart_cut_ring, &truncation);
    const int n_c = truncation.n_theta_cut / 4;
    const int shell_cells =
        cfg.shell_polar_cap_dendrite
            ? mesh_topo_checked_int_count(layout.shell_n_cells,
                                          "hybrid polar-tier shell cells")
            : mesh_topo_checked_int_count(
                  static_cast<long long>(cfg.nr) * cfg.nz,
                  "hybrid polar-tier shell cells");
    const int shell_nodes =
        cfg.shell_polar_cap_dendrite
            ? mesh_topo_checked_int_count(layout.shell_n_nodes,
                                          "hybrid polar-tier shell nodes")
            : mesh_topo_checked_int_count(
                  (static_cast<long long>(cfg.nr) + 1LL) * (cfg.nz + 1LL),
                  "hybrid polar-tier shell nodes");
    mb.block_count = layout.block_count + 2;
    mb.n_cells_core = mesh_topo_n_cells_core_with_n_c(cfg, n_c);
    mb.n_cells_bridge = mesh_topo_checked_int_count(
        layout.n_cells - shell_cells +
            mesh_topo_n_cells_bridge_with_n_c(cfg, n_c),
        "hybrid polar-tier bridge cells");
    mb.n_cells_shell = shell_cells;
    mb.n_nodes_core = mesh_topo_n_nodes_core_with_n_c(cfg, n_c);
    mb.n_nodes_bridge_interior =
        mesh_topo_checked_nonnegative_int_count(
            layout.n_nodes - shell_nodes +
                mesh_topo_n_nodes_bridge_interior_with_n_c(cfg, n_c) +
                (mesh_topo_polar_tier_cart_center_has_trifan_cap(cfg)
                     ? static_cast<long long>(truncation.n_theta_cut) + 1LL
                     : 0LL),
            "hybrid polar-tier bridge interior nodes");
    mb.n_nodes_shell = shell_nodes;
    mb.has_polar_tier = false;
    mb.polar_tier_h_r = layout.h_r;
    mb.polar_tier_fan_radius = layout.fan_radius;
    mb.polar_tier_columns =
        cfg.polar_tier_dendrite_enabled
            ? layout.dendrite_actual_tier_columns
            : layout.tier_columns;
    mb.polar_tier_transition_radii = layout.transition_radii;
    return mb;
  }
  if (cfg.topology_scheme ==
      tenryu::core::TopologyScheme::PENTAGON_BELT_SHELL) {
    mb.block_count = mesh_topo_checked_int_count(
        2LL * static_cast<long long>(cfg.pentagon_belt_layers.size()) + 1LL,
        "pentagon-belt block count");
    return mb;
  }
  mb.block_count =
      (cfg.topology_scheme ==
           tenryu::core::TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_5BLOCK ||
       mesh_topo_has_trifan_cap(cfg))
          ? 5
          : 3;
  mb.n_cells_core = mesh_topo_n_cells_core(cfg);
  mb.n_cells_bridge = mesh_topo_n_cells_bridge(cfg);
  mb.n_cells_shell = mesh_topo_n_cells_shell(cfg);
  mb.n_nodes_core = mesh_topo_n_nodes_core(cfg);
  mb.n_nodes_bridge_interior = mesh_topo_n_nodes_bridge_interior(cfg);
  mb.n_nodes_shell = mesh_topo_n_nodes_shell(cfg);
  mb.has_trifan_cap = mesh_topo_has_trifan_cap(cfg);
  if (mb.has_trifan_cap) {
    mb.n_cap = mesh_topo_n_cap(cfg);
    mb.n_cells_cap = mesh_topo_n_cells_cap(cfg);
    mb.n_nodes_cap = mesh_topo_n_nodes_cap(cfg);
  }
  return mb;
}

inline int mesh_topo_cap_apex_node_id(const MultiBlockTopology& mb) {
  TENRYU_ASSERT(mb.has_trifan_cap, "cap apex id requires trifan cap metadata");
  TENRYU_ASSERT(mb.n_nodes_cap > 0, "cap node count must be positive");
  return 0;
}

inline int mesh_topo_cap_ring_node_id(const MultiBlockTopology& mb,
                                      const int l,
                                      const int k) {
  TENRYU_ASSERT(mb.has_trifan_cap, "cap ring id requires trifan cap metadata");
  TENRYU_ASSERT(mb.n_cap > 0, "cap radial count must be positive");
  const int ntheta = 4 * mb.n_cap;
  TENRYU_ASSERT(l >= 1 && l <= mb.n_cap, "cap ring layer out of range");
  TENRYU_ASSERT(k >= 0 && k <= ntheta, "cap ring angular index out of range");
  const int node = 1 + (l - 1) * (ntheta + 1) + k;
  TENRYU_ASSERT(node >= 0 && node < mb.n_nodes_cap,
                "cap ring node id out of range");
  return node;
}

inline const BlockInfo& mesh_topo_trifan_fan_block(const MultiBlockTopology& mb,
                                                   const BlockRole fan) {
  TENRYU_ASSERT(fan == BlockRole::NORTH_FAN || fan == BlockRole::EAST_FAN ||
                    fan == BlockRole::SOUTH_FAN,
                "trifan fan block lookup requires a fan role");
  for (const BlockInfo& block : mb.blocks) {
    if (block.role == fan) {
      return block;
    }
  }
  TENRYU_ASSERT(false, "trifan fan block not found in multiblock metadata");
  return mb.blocks.front();
}

inline int mesh_topo_multiblock_polar_shell_block_count(
    const MultiBlockTopology& mb) {
  int shell_count = 0;
  for (const BlockInfo& block : mb.blocks) {
    if (block.role == BlockRole::POLAR_SHELL) {
      ++shell_count;
    }
  }
  return shell_count;
}

inline std::vector<int> mesh_topo_multiblock_ordered_polar_shell_chain(
    const MultiBlockTopology& mb) {
  std::vector<int> shell_blocks;
  for (int block = 0; block < static_cast<int>(mb.blocks.size()); ++block) {
    if (mb.blocks[static_cast<std::size_t>(block)].role ==
        BlockRole::POLAR_SHELL) {
      shell_blocks.push_back(block);
    }
  }
  TENRYU_ASSERT(!shell_blocks.empty(),
                "multiblock topology requires a POLAR_SHELL block");
  if (shell_blocks.size() == 1U) {
    return shell_blocks;
  }

  TENRYU_ASSERT(mb.has_polar_tier && shell_blocks.front() == 0,
                "split polar-shell chain must be the polar-tier block prefix");
  const int first = shell_blocks.front();
  const int last = shell_blocks.back();
  std::vector<int> chain;
  chain.reserve(static_cast<std::size_t>(last - first + 1));
  TENRYU_ASSERT(mb.dendrite_block_rings.size() == mb.blocks.size(),
                "split polar-shell chain requires block ring metadata");
  for (int block = first; block <= last; ++block) {
    const BlockRole role = mb.blocks[static_cast<std::size_t>(block)].role;
    TENRYU_ASSERT(role == BlockRole::POLAR_SHELL ||
                      role == BlockRole::TRANSITION_BELT,
                  "split polar-shell chain must contain only shell and transition blocks");
    chain.push_back(block);
  }
  for (std::size_t radial = 1; radial < chain.size(); ++radial) {
    const int inner = chain[radial - 1U];
    const int outer = chain[radial];
    const DendriteBlockRings& inner_rings =
        mb.dendrite_block_rings[static_cast<std::size_t>(inner)];
    const DendriteBlockRings& outer_rings =
        mb.dendrite_block_rings[static_cast<std::size_t>(outer)];
    TENRYU_ASSERT(!inner_rings.outer_ring.empty() &&
                      inner_rings.outer_ring == outer_rings.inner_ring,
                  "split polar-shell chain must share exact adjacent ring node IDs");
    bool joined = false;
    for (const SeamInfo& seam : mb.seams) {
      if ((seam.block_a == inner && seam.block_b == outer) ||
          (seam.block_a == outer && seam.block_b == inner)) {
        joined = true;
        break;
      }
    }
    TENRYU_ASSERT(joined,
                  "split polar-shell chain blocks must be joined by a seam");
  }

  const BlockInfo& outermost = mb.blocks[static_cast<std::size_t>(last)];
  int outer_face_count = 0;
  for (const UniqueOrientedFace& face : mb.boundary_faces) {
    if (face.bc_tag != static_cast<std::int8_t>(
                           BoundaryKind::SphericalOuterFree)) {
      continue;
    }
    TENRYU_ASSERT(
        face.cell_a >= outermost.cell_begin &&
            face.cell_a < outermost.cell_begin + outermost.cell_count,
        "outermost POLAR_SHELL block must own every spherical outer boundary face");
    ++outer_face_count;
  }
  TENRYU_ASSERT(
      outer_face_count == outermost.n_j_cells,
      "outermost POLAR_SHELL block must own one outer boundary face per angular cell");
  return chain;
}

inline const BlockInfo& mesh_topo_multiblock_outermost_polar_shell_block(
    const MultiBlockTopology& mb) {
  const BlockInfo* shell_block = nullptr;
  int shell_count = 0;
  for (const BlockInfo& block : mb.blocks) {
    if (block.role == BlockRole::POLAR_SHELL) {
      shell_block = &block;
      ++shell_count;
    }
  }
  TENRYU_ASSERT(shell_block != nullptr,
                "multiblock topology requires a POLAR_SHELL block");
  TENRYU_ASSERT(shell_count >= 1,
                "multiblock topology requires at least one POLAR_SHELL block");
  TENRYU_ASSERT(shell_block->n_i_cells > 0 && shell_block->n_j_cells > 0,
                "POLAR_SHELL block dimensions must be positive");
  TENRYU_ASSERT(shell_block->cell_count ==
                    shell_block->n_i_cells * shell_block->n_j_cells,
                "POLAR_SHELL block cell count must match block dimensions");
  if (shell_count == 1) {
    TENRYU_ASSERT(shell_block->owned_node_count ==
                      (shell_block->n_i_cells + 1) *
                          (shell_block->n_j_cells + 1),
                  "POLAR_SHELL owned nodes must be contiguous block-grid nodes");
  } else {
    const std::vector<int> chain =
        mesh_topo_multiblock_ordered_polar_shell_chain(mb);
    TENRYU_ASSERT(
        &mb.blocks[static_cast<std::size_t>(chain.back())] == shell_block,
        "last split POLAR_SHELL block must own the outer boundary");
    TENRYU_ASSERT(
        shell_block->owned_node_count ==
            shell_block->n_i_cells * (shell_block->n_j_cells + 1),
        "outermost split POLAR_SHELL owned nodes must exclude only its shared inner ring");
  }
  return *shell_block;
}

inline const BlockInfo& mesh_topo_multiblock_polar_shell_block(
    const MultiBlockTopology& mb) {
  return mesh_topo_multiblock_outermost_polar_shell_block(mb);
}

inline int mesh_topo_multiblock_outermost_polar_shell_node_offset(
    const MultiBlockTopology& mb) {
  const BlockInfo& shell_block =
      mesh_topo_multiblock_outermost_polar_shell_block(mb);
  if (mesh_topo_multiblock_polar_shell_block_count(mb) == 1) {
    return shell_block.owned_node_begin;
  }

  const std::vector<int> chain =
      mesh_topo_multiblock_ordered_polar_shell_chain(mb);
  const int outermost_block = chain.back();
  TENRYU_ASSERT(mb.dendrite_block_rings.size() == mb.blocks.size(),
                "split polar-shell node offset requires block ring metadata");
  const DendriteBlockRings& rings =
      mb.dendrite_block_rings[static_cast<std::size_t>(outermost_block)];
  const int stride = shell_block.n_j_cells + 1;
  TENRYU_ASSERT(rings.inner_ring.size() == static_cast<std::size_t>(stride) &&
                    rings.outer_ring.size() == static_cast<std::size_t>(stride),
                "outermost split POLAR_SHELL requires full inner and outer rings");
  for (int k = 0; k < stride; ++k) {
    TENRYU_ASSERT(
        rings.inner_ring[static_cast<std::size_t>(k)] ==
                rings.inner_ring.front() + k &&
            rings.outer_ring[static_cast<std::size_t>(k)] ==
                rings.outer_ring.front() + k,
        "outermost split POLAR_SHELL boundary rings must be contiguous");
  }
  TENRYU_ASSERT(
      rings.outer_ring.front() ==
              rings.inner_ring.front() + shell_block.n_i_cells * stride &&
          shell_block.owned_node_begin == rings.inner_ring.front() + stride,
      "outermost split POLAR_SHELL structured rows must run from its shared "
      "inner ring to the outer boundary");
  return rings.inner_ring.front();
}

inline int mesh_topo_multiblock_outermost_polar_shell_node_offset(
    const MeshTopology& topo) {
  TENRYU_ASSERT(topo.multiblock.has_value(),
                "multiblock polar-shell node offset requires topology metadata");
  return mesh_topo_multiblock_outermost_polar_shell_node_offset(
      *topo.multiblock);
}

inline int mesh_topo_multiblock_outermost_polar_shell_nr(
    const MeshTopology& topo) {
  TENRYU_ASSERT(topo.multiblock.has_value(),
                "multiblock polar-shell nr requires topology metadata");
  const BlockInfo& shell_block =
      mesh_topo_multiblock_outermost_polar_shell_block(*topo.multiblock);
  if (mesh_topo_multiblock_polar_shell_block_count(*topo.multiblock) == 1) {
    TENRYU_ASSERT(topo.nr == shell_block.n_i_cells,
                  "MeshTopology nr must match POLAR_SHELL radial cells");
  } else {
    const std::vector<int> chain =
        mesh_topo_multiblock_ordered_polar_shell_chain(*topo.multiblock);
    int chain_rows = 0;
    for (const int block : chain) {
      chain_rows += topo.multiblock->blocks[static_cast<std::size_t>(block)]
                        .n_i_cells;
    }
    TENRYU_ASSERT(topo.nr == chain_rows,
                  "MeshTopology nr must match the split polar-shell radial chain");
  }
  return shell_block.n_i_cells;
}

inline int mesh_topo_multiblock_polar_shell_node_offset(
    const MultiBlockTopology& mb) {
  return mesh_topo_multiblock_outermost_polar_shell_node_offset(mb);
}

inline int mesh_topo_multiblock_polar_shell_node_offset(
    const MeshTopology& topo) {
  return mesh_topo_multiblock_outermost_polar_shell_node_offset(topo);
}

inline int mesh_topo_multiblock_polar_shell_nr(const MeshTopology& topo) {
  return mesh_topo_multiblock_outermost_polar_shell_nr(topo);
}

// Trifan-cap fan node id, mirroring the canonical mesh-construction indexing
// (fan_node_id in mesh.cu, cross-checked there at topology build): layer l in
// [0, n_b] with l=0 on the cap outer ring, l=n_b on the polar-shell row 0,
// and seam-shared end columns NORTH(k=n_c)=EAST(k=0), EAST(k=2n_c)=SOUTH(k=0).
inline int mesh_topo_trifan_fan_node_id(const MultiBlockTopology& mb,
                                        const BlockRole fan,
                                        const int l,
                                        const int k) {
  TENRYU_ASSERT(mb.has_trifan_cap,
                "trifan fan node id requires trifan cap metadata");
  const BlockInfo& north = mesh_topo_trifan_fan_block(mb, BlockRole::NORTH_FAN);
  const BlockInfo& east = mesh_topo_trifan_fan_block(mb, BlockRole::EAST_FAN);
  const BlockInfo& south = mesh_topo_trifan_fan_block(mb, BlockRole::SOUTH_FAN);
  const int n_c = north.n_j_cells;
  const int n_b = north.n_i_cells;
  TENRYU_ASSERT(n_c == mb.n_cap && east.n_j_cells == 2 * n_c &&
                    south.n_j_cells == n_c && east.n_i_cells == n_b &&
                    south.n_i_cells == n_b,
                "trifan fan blocks have inconsistent dimensions");
  TENRYU_ASSERT(l >= 0 && l <= n_b, "trifan fan node layer out of range");
  const int n_j = (fan == BlockRole::EAST_FAN) ? 2 * n_c : n_c;
  TENRYU_ASSERT(k >= 0 && k <= n_j,
                "trifan fan node angular index out of range");
  const int g = (fan == BlockRole::NORTH_FAN)
                    ? k
                    : (fan == BlockRole::EAST_FAN ? n_c + k : 3 * n_c + k);
  if (l == 0) {
    return mesh_topo_cap_ring_node_id(mb, mb.n_cap, g);
  }
  if (l == n_b) {
    const BlockInfo& shell = mesh_topo_multiblock_polar_shell_block(mb);
    TENRYU_ASSERT(shell.n_j_cells == 4 * n_c,
                  "polar shell angular count must match trifan ntheta");
    return shell.owned_node_begin + g;
  }
  switch (fan) {
    case BlockRole::NORTH_FAN:
      return north.owned_node_begin + (l - 1) * (n_c + 1) + k;
    case BlockRole::EAST_FAN:
      if (k == 0) {
        return north.owned_node_begin + (l - 1) * (n_c + 1) + n_c;
      }
      return east.owned_node_begin + (l - 1) * (2 * n_c) + (k - 1);
    case BlockRole::SOUTH_FAN:
      if (k == 0) {
        return east.owned_node_begin + (l - 1) * (2 * n_c) + (2 * n_c - 1);
      }
      return south.owned_node_begin + (l - 1) * n_c + (k - 1);
    default:
      TENRYU_ASSERT(false, "trifan fan node id requires a fan role");
      return -1;
  }
}

inline int mesh_topo_multiblock_outermost_polar_shell_nz(
    const MeshTopology& topo) {
  TENRYU_ASSERT(topo.multiblock.has_value(),
                "multiblock polar-shell nz requires topology metadata");
  const BlockInfo& shell_block =
      mesh_topo_multiblock_outermost_polar_shell_block(*topo.multiblock);
  TENRYU_ASSERT(topo.nz == shell_block.n_j_cells,
                "MeshTopology nz must match POLAR_SHELL angular cells");
  return shell_block.n_j_cells;
}

inline int mesh_topo_multiblock_polar_shell_nz(const MeshTopology& topo) {
  return mesh_topo_multiblock_outermost_polar_shell_nz(topo);
}

inline void mesh_topo_assert_multiblock_outermost_polar_shell_outer_boundary(
    const MeshTopology& topo) {
  if (topo.general_polygonal) {
    // General CSR form: every boundary-face node must carry a boundary flag,
    // and any non-axis boundary node must carry NODE_OUTER_PHYSICAL_BOUNDARY.
    TENRYU_ASSERT(topo.multiblock.has_value(),
                  "general-polygonal boundary check requires topology metadata");
    TENRYU_ASSERT(topo.node_flags.size() ==
                      static_cast<std::size_t>(topo.n_nodes),
                  "general-polygonal boundary check requires node_flags");
    for (std::size_t node = 0; node < topo.node_flags.size(); ++node) {
      const std::uint8_t flags = topo.node_flags[node];
      if ((flags & NODE_BOUNDARY) == 0U) {
        continue;
      }
      TENRYU_ASSERT((flags & (NODE_AXIS | NODE_OUTER_PHYSICAL_BOUNDARY)) != 0U,
                    "general-polygonal boundary node must be axis or outer");
    }
    return;
  }
  TENRYU_ASSERT(topo.multiblock.has_value(),
                "multiblock polar-shell boundary check requires topology metadata");
  const BlockInfo& shell_block =
      mesh_topo_multiblock_outermost_polar_shell_block(*topo.multiblock);
  TENRYU_ASSERT(topo.node_flags.size() == static_cast<std::size_t>(topo.n_nodes),
                "multiblock polar-shell boundary check requires node_flags");
  const int stride = shell_block.n_j_cells + 1;
  const int outer_ring_begin =
      mesh_topo_multiblock_outermost_polar_shell_node_offset(topo) +
      shell_block.n_i_cells * stride;
  for (int k = 0; k <= shell_block.n_j_cells; ++k) {
    const int n = outer_ring_begin + k;
    TENRYU_ASSERT(n >= 0 && n < topo.n_nodes,
                  "POLAR_SHELL outer boundary node out of range");
    TENRYU_ASSERT((topo.node_flags[static_cast<std::size_t>(n)] &
                   NODE_OUTER_PHYSICAL_BOUNDARY) != 0U,
                  "POLAR_SHELL outer ring must carry NODE_OUTER_PHYSICAL_BOUNDARY");
  }
}

inline void mesh_topo_assert_multiblock_polar_shell_outer_boundary(
    const MeshTopology& topo) {
  mesh_topo_assert_multiblock_outermost_polar_shell_outer_boundary(topo);
}

struct MeshTopoCellCornerNodesN {
  std::array<int, kMeshTopoCellStorageSlotsMax> values;
  int count;
};

struct MeshTopoCellFaceAdjacentCellsN {
  std::array<int, kMeshTopoCellStorageSlotsMax> values;
  int count;
};

struct MeshTopoCellFaceBcTagsN {
  std::array<int, kMeshTopoCellStorageSlotsMax> values;
  int count;
};

inline std::array<int, 4> mesh_topo_cell_corner_nodes(
    const MeshTopology& topo,
    const int c,
    const tenryu::core::Config::MeshConfig& cfg) {
  TENRYU_ASSERT(c >= 0, "cell index must be non-negative");
  if (cfg.topology_scheme == tenryu::core::TopologyScheme::SINGLE_BLOCK) {
    TENRYU_ASSERT(cfg.nr > 0 && cfg.nz > 0,
                  "single-block mesh dimensions must be positive");
    TENRYU_ASSERT(static_cast<long long>(c) <
                      static_cast<long long>(cfg.nr) *
                          static_cast<long long>(cfg.nz),
                  "single-block cell index out of range");
    const int i = c / cfg.nz;
    const int j = c % cfg.nz;
    const int nz_plus_1 = cfg.nz + 1;
    return {{
        i * nz_plus_1 + j,
        (i + 1) * nz_plus_1 + j,
        (i + 1) * nz_plus_1 + (j + 1),
        i * nz_plus_1 + (j + 1),
    }};
  }

  TENRYU_ASSERT(topo.multiblock.has_value(),
                "multiblock topology requested but not initialized");
  const auto& mb = *topo.multiblock;
  TENRYU_ASSERT(static_cast<std::size_t>(c) + 1U <
                    mb.cell_node_csr_offsets.size(),
                "multiblock cell-node CSR offset missing");
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
  const int next =
      mb.cell_node_csr_offsets[static_cast<std::size_t>(c) + 1U];
  TENRYU_ASSERT(next - off == kMeshTopoCellStorageSlots,
                "quad (stride-4) storage accessor; pentagon-belt meshes are not "
                "supported here yet");
  TENRYU_ASSERT(off >= 0, "multiblock cell-node CSR offset must be non-negative");
  TENRYU_ASSERT(static_cast<std::size_t>(off) +
                    static_cast<std::size_t>(kMeshTopoCellStorageSlots) <=
                    mb.cell_node_csr_indices.size(),
                "multiblock cell-node CSR indices missing");
  return {{
      mb.cell_node_csr_indices[static_cast<std::size_t>(off + 0)],
      mb.cell_node_csr_indices[static_cast<std::size_t>(off + 1)],
      mb.cell_node_csr_indices[static_cast<std::size_t>(off + 2)],
      mb.cell_node_csr_indices[static_cast<std::size_t>(off + 3)],
  }};
}

inline MeshTopoCellCornerNodesN mesh_topo_cell_corner_nodes_n(
    const MeshTopology& topo,
    const std::vector<std::uint8_t>& cell_nverts,
    const int c,
    const tenryu::core::Config::MeshConfig& cfg) {
  TENRYU_ASSERT(c >= 0, "cell index must be non-negative");
  MeshTopoCellCornerNodesN result{};
  result.count = mesh_topo_cell_active_nverts(cell_nverts, c);
  if (cfg.topology_scheme == tenryu::core::TopologyScheme::SINGLE_BLOCK) {
    TENRYU_ASSERT(cfg.nr > 0 && cfg.nz > 0,
                  "single-block mesh dimensions must be positive");
    TENRYU_ASSERT(static_cast<long long>(c) <
                      static_cast<long long>(cfg.nr) *
                          static_cast<long long>(cfg.nz),
                  "single-block cell index out of range");
    TENRYU_ASSERT(result.count <= kMeshTopoCellStorageSlots,
                  "single-block cell_nverts exceeds stride-4 storage");
    const int i = c / cfg.nz;
    const int j = c % cfg.nz;
    const int nz_plus_1 = cfg.nz + 1;
    const std::array<int, kMeshTopoCellStorageSlots> nodes{{
        i * nz_plus_1 + j,
        (i + 1) * nz_plus_1 + j,
        (i + 1) * nz_plus_1 + (j + 1),
        i * nz_plus_1 + (j + 1),
    }};
    for (int k = 0; k < result.count; ++k) {
      result.values[static_cast<std::size_t>(k)] =
          nodes[static_cast<std::size_t>(k)];
    }
    return result;
  }

  TENRYU_ASSERT(topo.multiblock.has_value(),
                "multiblock topology requested but not initialized");
  const auto& mb = *topo.multiblock;
  TENRYU_ASSERT(static_cast<std::size_t>(c) + 1U <
                    mb.cell_node_csr_offsets.size(),
                "multiblock cell-node CSR offset missing");
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
  const int next =
      mb.cell_node_csr_offsets[static_cast<std::size_t>(c) + 1U];
  const int width = next - off;
  TENRYU_ASSERT(width >= result.count &&
                    width <= kMeshTopoCellStorageSlotsMax,
                "cell-node CSR width must contain all active vertices and fit "
                "the topology storage capacity");
  TENRYU_ASSERT(off >= 0, "multiblock cell-node CSR offset must be non-negative");
  TENRYU_ASSERT(static_cast<std::size_t>(off) +
                    static_cast<std::size_t>(width) <=
                    mb.cell_node_csr_indices.size(),
                "multiblock cell-node CSR indices missing");
  for (int k = 0; k < result.count; ++k) {
    result.values[static_cast<std::size_t>(k)] =
        mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
  }
  return result;
}

inline std::array<int, 4> mesh_topo_cell_face_adjacent_cells(
    const MeshTopology& topo,
    const int c,
    const tenryu::core::Config::MeshConfig& cfg) {
  TENRYU_ASSERT(c >= 0, "cell index must be non-negative");
  if (cfg.topology_scheme == tenryu::core::TopologyScheme::SINGLE_BLOCK) {
    TENRYU_ASSERT(cfg.nr > 0 && cfg.nz > 0,
                  "single-block mesh dimensions must be positive");
    TENRYU_ASSERT(static_cast<long long>(c) <
                      static_cast<long long>(cfg.nr) *
                          static_cast<long long>(cfg.nz),
                  "single-block cell index out of range");
    const int i = c / cfg.nz;
    const int j = c % cfg.nz;
    return {{
        (i > 0) ? ((i - 1) * cfg.nz + j) : -1,
        (i + 1 < cfg.nr) ? ((i + 1) * cfg.nz + j) : -1,
        (j > 0) ? (i * cfg.nz + (j - 1)) : -1,
        (j + 1 < cfg.nz) ? (i * cfg.nz + (j + 1)) : -1,
    }};
  }

  TENRYU_ASSERT(topo.multiblock.has_value(),
                "multiblock topology requested but not initialized");
  const auto& mb = *topo.multiblock;
  TENRYU_ASSERT(static_cast<std::size_t>(c) + 1U <
                    mb.face_adj_csr_offsets.size(),
                "multiblock face-adjacency CSR offset missing");
  const int off = mb.face_adj_csr_offsets[static_cast<std::size_t>(c)];
  const int next =
      mb.face_adj_csr_offsets[static_cast<std::size_t>(c) + 1U];
  TENRYU_ASSERT(next - off == kMeshTopoCellStorageSlots,
                "quad (stride-4) storage accessor; pentagon-belt meshes are not "
                "supported here yet");
  TENRYU_ASSERT(off >= 0,
                "multiblock face-adjacency CSR offset must be non-negative");
  TENRYU_ASSERT(static_cast<std::size_t>(off) +
                    static_cast<std::size_t>(kMeshTopoCellStorageSlots) <=
                    mb.face_adj_csr_indices.size(),
                "multiblock face-adjacency CSR indices missing");
  return {{
      mb.face_adj_csr_indices[static_cast<std::size_t>(off + 0)],
      mb.face_adj_csr_indices[static_cast<std::size_t>(off + 1)],
      mb.face_adj_csr_indices[static_cast<std::size_t>(off + 2)],
      mb.face_adj_csr_indices[static_cast<std::size_t>(off + 3)],
  }};
}

inline MeshTopoCellFaceAdjacentCellsN mesh_topo_cell_face_adjacent_cells_n(
    const MeshTopology& topo,
    const std::vector<std::uint8_t>& cell_nverts,
    const int c,
    const tenryu::core::Config::MeshConfig& cfg) {
  TENRYU_ASSERT(c >= 0, "cell index must be non-negative");
  MeshTopoCellFaceAdjacentCellsN result{};
  result.count = mesh_topo_cell_active_nverts(cell_nverts, c);
  if (cfg.topology_scheme == tenryu::core::TopologyScheme::SINGLE_BLOCK) {
    TENRYU_ASSERT(cfg.nr > 0 && cfg.nz > 0,
                  "single-block mesh dimensions must be positive");
    TENRYU_ASSERT(static_cast<long long>(c) <
                      static_cast<long long>(cfg.nr) *
                          static_cast<long long>(cfg.nz),
                  "single-block cell index out of range");
    TENRYU_ASSERT(result.count <= kMeshTopoCellStorageSlots,
                  "single-block cell_nverts exceeds stride-4 storage");
    const int i = c / cfg.nz;
    const int j = c % cfg.nz;
    const std::array<int, kMeshTopoCellStorageSlots> adjacent{{
        (i > 0) ? ((i - 1) * cfg.nz + j) : -1,
        (i + 1 < cfg.nr) ? ((i + 1) * cfg.nz + j) : -1,
        (j > 0) ? (i * cfg.nz + (j - 1)) : -1,
        (j + 1 < cfg.nz) ? (i * cfg.nz + (j + 1)) : -1,
    }};
    for (int k = 0; k < result.count; ++k) {
      result.values[static_cast<std::size_t>(k)] =
          adjacent[static_cast<std::size_t>(k)];
    }
    return result;
  }

  TENRYU_ASSERT(topo.multiblock.has_value(),
                "multiblock topology requested but not initialized");
  const auto& mb = *topo.multiblock;
  TENRYU_ASSERT(static_cast<std::size_t>(c) + 1U <
                    mb.face_adj_csr_offsets.size(),
                "multiblock face-adjacency CSR offset missing");
  const int off = mb.face_adj_csr_offsets[static_cast<std::size_t>(c)];
  const int next =
      mb.face_adj_csr_offsets[static_cast<std::size_t>(c) + 1U];
  const int width = next - off;
  TENRYU_ASSERT(width >= result.count &&
                    width <= kMeshTopoCellStorageSlotsMax,
                "face-adjacency CSR width must contain all active faces and fit "
                "the topology storage capacity");
  TENRYU_ASSERT(off >= 0,
                "multiblock face-adjacency CSR offset must be non-negative");
  TENRYU_ASSERT(static_cast<std::size_t>(off) +
                    static_cast<std::size_t>(width) <=
                    mb.face_adj_csr_indices.size(),
                "multiblock face-adjacency CSR indices missing");
  for (int k = 0; k < result.count; ++k) {
    result.values[static_cast<std::size_t>(k)] =
        mb.face_adj_csr_indices[static_cast<std::size_t>(off + k)];
  }
  return result;
}

inline std::array<int, 4> mesh_topo_cell_face_bc_tags(
    const MeshTopology& topo,
    const int c,
    const tenryu::core::Config::MeshConfig& cfg) {
  TENRYU_ASSERT(c >= 0, "cell index must be non-negative");
  if (cfg.topology_scheme == tenryu::core::TopologyScheme::SINGLE_BLOCK) {
    return {{0, 0, 0, 0}};
  }

  TENRYU_ASSERT(topo.multiblock.has_value(),
                "multiblock topology requested but not initialized");
  const auto& mb = *topo.multiblock;
  TENRYU_ASSERT(static_cast<std::size_t>(c) + 1U <
                    mb.face_adj_csr_offsets.size(),
                "multiblock face-tag CSR offset missing");
  const int off = mb.face_adj_csr_offsets[static_cast<std::size_t>(c)];
  const int next =
      mb.face_adj_csr_offsets[static_cast<std::size_t>(c) + 1U];
  TENRYU_ASSERT(next - off == kMeshTopoCellStorageSlots,
                "quad (stride-4) storage accessor; pentagon-belt meshes are not "
                "supported here yet");
  TENRYU_ASSERT(off >= 0, "multiblock face-tag offset must be non-negative");
  TENRYU_ASSERT(static_cast<std::size_t>(off) +
                    static_cast<std::size_t>(kMeshTopoCellStorageSlots) <=
                    mb.face_bc_tags.size(),
                "multiblock face tags missing");
  return {{
      mb.face_bc_tags[static_cast<std::size_t>(off + 0)],
      mb.face_bc_tags[static_cast<std::size_t>(off + 1)],
      mb.face_bc_tags[static_cast<std::size_t>(off + 2)],
      mb.face_bc_tags[static_cast<std::size_t>(off + 3)],
  }};
}

inline MeshTopoCellFaceBcTagsN mesh_topo_cell_face_bc_tags_n(
    const MeshTopology& topo,
    const std::vector<std::uint8_t>& cell_nverts,
    const int c,
    const tenryu::core::Config::MeshConfig& cfg) {
  TENRYU_ASSERT(c >= 0, "cell index must be non-negative");
  MeshTopoCellFaceBcTagsN result{};
  result.count = mesh_topo_cell_active_nverts(cell_nverts, c);
  if (cfg.topology_scheme == tenryu::core::TopologyScheme::SINGLE_BLOCK) {
    TENRYU_ASSERT(result.count <= kMeshTopoCellStorageSlots,
                  "single-block cell_nverts exceeds stride-4 storage");
    return result;
  }

  TENRYU_ASSERT(topo.multiblock.has_value(),
                "multiblock topology requested but not initialized");
  const auto& mb = *topo.multiblock;
  TENRYU_ASSERT(static_cast<std::size_t>(c) + 1U <
                    mb.face_adj_csr_offsets.size(),
                "multiblock face-tag CSR offset missing");
  const int off = mb.face_adj_csr_offsets[static_cast<std::size_t>(c)];
  const int next =
      mb.face_adj_csr_offsets[static_cast<std::size_t>(c) + 1U];
  const int width = next - off;
  TENRYU_ASSERT(width >= result.count &&
                    width <= kMeshTopoCellStorageSlotsMax,
                "face-tag CSR width must contain all active faces and fit the "
                "topology storage capacity");
  TENRYU_ASSERT(off >= 0, "multiblock face-tag offset must be non-negative");
  TENRYU_ASSERT(static_cast<std::size_t>(off) +
                    static_cast<std::size_t>(width) <=
                    mb.face_bc_tags.size(),
                "multiblock face tags missing");
  for (int k = 0; k < result.count; ++k) {
    result.values[static_cast<std::size_t>(k)] =
        mb.face_bc_tags[static_cast<std::size_t>(off + k)];
  }
  return result;
}

struct Mesh {
  Mesh() = default;
  ~Mesh();

  Mesh(const Mesh& other);
  Mesh& operator=(const Mesh& other);
  Mesh(Mesh&& other) noexcept;
  Mesh& operator=(Mesh&& other) noexcept;

  MeshTopology topo;

  // Borrowed pointers into State.
  double* node_r = nullptr;
  double* node_z = nullptr;
  const mesh::RingPageSet* ring_pages = nullptr;

  // Owned geometry caches.
  std::vector<double> cell_vol;
  std::vector<double> cell_area;
  std::vector<double> cell_centroid_r;
  tenryu::core::DeviceArray<double> cell_centroid_r_device;
  tenryu::core::DeviceArray<double> cell_vol_device;
  tenryu::core::DeviceArray<double> cell_area_device;
  tenryu::core::DeviceArray<double> cell_centroid_z_device;
  tenryu::core::DeviceArray<double> cell_Svec_r_device;
  tenryu::core::DeviceArray<double> cell_Svec_z_device;
  std::vector<double> cell_centroid_z;
  // Area vectors: corner_stride slots per cell.
  int corner_stride = 4;
  std::vector<double> cell_Svec_r;
  std::vector<double> cell_Svec_z;
  std::vector<std::uint8_t> cell_nverts;
  std::vector<BoundaryKind> edge_tags;
  tenryu::core::DeviceArray<int> multiblock_cell_node_csr_offsets;
  tenryu::core::DeviceArray<int> multiblock_cell_node_csr_indices;
  // Reverse cell-node CSR device mirrors. Populated only for multiblock meshes.
  tenryu::core::DeviceArray<int> multiblock_reverse_csr_node_offsets;
  tenryu::core::DeviceArray<int> multiblock_reverse_csr_node_cells;
  tenryu::core::DeviceArray<int> multiblock_reverse_csr_node_corners;
  tenryu::core::DeviceArray<int> multiblock_node_edge_csr_offsets;
  tenryu::core::DeviceArray<int> multiblock_node_edge_csr_edges;
  tenryu::core::DeviceArray<std::int8_t> multiblock_node_edge_csr_end;
  tenryu::core::DeviceArray<int> multiblock_cell_edge_csr_offsets;
  tenryu::core::DeviceArray<int> multiblock_cell_edge_csr_edges;
  tenryu::core::DeviceArray<std::int8_t> multiblock_cell_edge_csr_side;
  // Device mirrors of static multiblock topology consumed by
  // recompute_geometry_checked; refreshed (with topology_serial) at every
  // canonical topology-(re)build site alongside the CSR mirrors above.
  tenryu::core::DeviceArray<int> multiblock_cell_orientation_sign_device;
  tenryu::core::DeviceArray<std::uint8_t> multiblock_cell_nverts_device;
  std::uint64_t topology_serial = 1;

  int dim = 1;
  // W-G: 1D coordinate geometry code (mesh/geometry_1d.cuh Geometry1D);
  // 0 = spherical (historic default), 1 = cylindrical, 2 = planar.
  int geometry_code = 0;
  LogicalMesh2D logical = LogicalMesh2D::RectangularRZ;
  PolarCenterTreatment polar_center_treatment = PolarCenterTreatment::Annular;
  bool multiblock_outer_svec_tangent_balance = true;
  std::optional<ButtonCenterTopology> button_center;

  [[nodiscard]] bool is_button_cell(const int c) const {
    return button_center && button_center->enabled && c == topo.cell_index(0, 0);
  }

  [[nodiscard]] bool is_dormant_cell(const int c) const {
    if (!button_center || !button_center->enabled || c < 0 || c >= topo.n_cells ||
        topo.nz <= 0 || is_button_cell(c)) {
      return false;
    }
    const int i = c / topo.nz;
    return i >= 0 && i < button_center->outer_node_ring;
  }

  // Cells whose geometry is virtual (e.g. central pseudo-core members whose
  // guaranteed-positive star-map reconstruction can legitimately fold once
  // the macro boundary is no longer star-shaped); geometry validation must
  // not hard-assert on them. Empty when no agglomeration is active.
  std::vector<std::uint8_t> geometry_exempt_cells;
  [[nodiscard]] bool is_geometry_exempt_cell(const int c) const {
    return c >= 0 &&
           static_cast<std::size_t>(c) < geometry_exempt_cells.size() &&
           geometry_exempt_cells[static_cast<std::size_t>(c)] != 0U;
  }

  MeshGeometryResult recompute_geometry_checked(
      const MeshGeometryCheckOptions& opts);
  void recompute_geometry();
  // w1-B2a: on the multiblock path the host Svec mirrors are LAZY — the
  // device members are authoritative; call this before any host read.
  void materialize_host_svec();
  // w1-B2a: laziness is scoped — only the hydro lagrangian step enables it;
  // out-of-scope recomputes keep the eager host pulls.
  void begin_lazy_svec() { svec_lazy_mode_ = true; }
  void end_lazy_svec() {
    materialize_host_svec();
    svec_lazy_mode_ = false;
  }
  void recompute_geometry_device_only(double* state_vol);
  MeshGeometryResult sync_device_geometry_to_host_checked(
      const double* state_vol,
      const MeshGeometryCheckOptions& opts);
  void sync_device_geometry_to_host(const double* state_vol);

 private:
  void ensure_persistent_geometry_buffers_(std::size_t n_cells);
  void release_persistent_geometry_buffers_();

  std::uint64_t geometry_topo_checked_serial_ = 0;
  bool geometry_use_cell_nverts_cached_ = false;
  bool host_svec_stale_ = false;
  bool svec_lazy_mode_ = false;
  double* d_cell_vol_persistent_ = nullptr;
  double* d_cell_centroid_r_persistent_ = nullptr;
  double* d_cell_centroid_z_persistent_ = nullptr;
  std::size_t d_persistent_capacity_ = 0;
};

void rebuild_multiblock_node_edge_csr(Mesh& mesh);
void rebuild_multiblock_csw_line_topology(Mesh& mesh);

// RAII scope for Mesh lazy host-Svec mode (return/exception safe).
class MeshSvecLazyScope {
 public:
  explicit MeshSvecLazyScope(Mesh& mesh) : mesh_(mesh) {
    mesh_.begin_lazy_svec();
  }
  ~MeshSvecLazyScope() { mesh_.end_lazy_svec(); }
  MeshSvecLazyScope(const MeshSvecLazyScope&) = delete;
  MeshSvecLazyScope& operator=(const MeshSvecLazyScope&) = delete;

 private:
  Mesh& mesh_;
};

// Option-C MPI decomposition: restrict host-side geometry validity checks to
// the owned(+ghost) cell window (stale non-owned regions are expected).
// -1 end = full mesh (default). Set by the driver when n_ranks > 1.
void set_geometry_check_window(int begin, int end);

struct FullAxisNodeChain {
  std::vector<int> node_ids;
  std::vector<double> z;
  std::vector<double> lumped_mass;

  [[nodiscard]] bool active() const noexcept {
    return !node_ids.empty();
  }
};

FullAxisNodeChain build_full_axis_node_chain(
    const Mesh& mesh,
    const std::vector<double>& corner_mass);

inline bool has_tri_fan_center_topology(const Mesh& mesh) {
  if (mesh.dim != 2 || mesh.topo.nr <= 0 || mesh.topo.nz <= 0) {
    return false;
  }
  if (mesh.cell_nverts.size() == static_cast<std::size_t>(mesh.topo.n_cells)) {
    for (const std::uint8_t nverts : mesh.cell_nverts) {
      if (nverts == 3U) {
        return true;
      }
    }
  }
  if (mesh.topo.node_flags.size() != static_cast<std::size_t>(mesh.topo.n_nodes)) {
    return false;
  }
  for (int j = 0; j <= mesh.topo.nz; ++j) {
    const int n = mesh.topo.node_index(0, j);
    if ((mesh.topo.node_flags[static_cast<std::size_t>(n)] & NODE_CENTER) == 0U) {
      return false;
    }
  }
  return true;
}

Mesh create_mesh(const tenryu::core::Config& cfg, tenryu::core::State& state);
Mesh create_cone_shell_mesh(const tenryu::core::Config& cfg,
                            tenryu::core::State& state);
Mesh create_pentagon_belt_shell_mesh(const tenryu::core::Config& cfg,
                                     tenryu::core::State& state);

// Rebuild every derived topology cache after a runtime cell-connectivity
// edit (cell-node CSR rows + cell_nverts and, for ReALE, node coordinates/count
// changed in place; no cells added): reverse node->cell CSR, face adjacency,
// unique internal
// faces, boundary faces, BC tags, and all their device payloads -- in the
// same order and with the same builders the initial mesh construction uses.
void rebuild_runtime_topology_caches(core::State& state,
                                     const core::Config& cfg,
                                     bool rebuild_node_flags = false);

}  // namespace tenryu::mesh
