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
  PENTAGON_BELT = 6,
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
  thrust::device_vector<int> d_unique_face_cell_a;
  thrust::device_vector<int> d_unique_face_cell_b;
  thrust::device_vector<int> d_unique_face_local_a;
  thrust::device_vector<int> d_unique_face_local_b;
  thrust::device_vector<std::int8_t> d_unique_face_bc_tag;
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
         scheme == tenryu::core::TopologyScheme::CONE_SHELL_SPINE ||
         scheme == tenryu::core::TopologyScheme::PENTAGON_BELT_SHELL;
}

TENRYU_MESH_HOST_DEVICE inline bool mesh_topo_has_trifan_cap(
    const tenryu::core::TopologyScheme scheme) noexcept {
  return scheme == tenryu::core::TopologyScheme::
                       MULTIBLOCK_HALF_BUTTERFLY_TRIFAN_CAP_5BLOCK;
}

constexpr int kMeshTopoCellStorageSlots = 4;
constexpr int kMeshTopoCellStorageSlotsMax = 8;
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
      cell_nverts[cell] <= 8U) {
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
  TENRYU_ASSERT(nverts >= 3U && nverts <= 8U,
                "cell_nverts entries must be in [3, 8]");
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

inline int mesh_topo_n_cap(const tenryu::core::Config::MeshConfig& cfg) {
  const int n_c = cfg.multiblock_cart_core_n_c;
  TENRYU_ASSERT(n_c > 0, "Mesh.multiblock_cart_core_n_c must be positive");
  return n_c;
}

inline int mesh_topo_n_cells_cap(
    const tenryu::core::Config::MeshConfig& cfg) {
  const int n_c = cfg.multiblock_cart_core_n_c;
  const int n_cap = mesh_topo_n_cap(cfg);
  TENRYU_ASSERT(n_c > 0, "Mesh.multiblock_cart_core_n_c must be positive");
  return mesh_topo_checked_int_count(4LL * n_c * n_cap,
                                     "multiblock cap cell count");
}

inline int mesh_topo_n_nodes_cap(
    const tenryu::core::Config::MeshConfig& cfg) {
  const int n_c = cfg.multiblock_cart_core_n_c;
  const int n_cap = mesh_topo_n_cap(cfg);
  TENRYU_ASSERT(n_c > 0, "Mesh.multiblock_cart_core_n_c must be positive");
  return mesh_topo_checked_int_count(
      1LL + static_cast<long long>(n_cap) * (4LL * n_c + 1LL),
      "multiblock cap node count");
}

inline int mesh_topo_n_cells_core(
    const tenryu::core::Config::MeshConfig& cfg) {
  if (mesh_topo_has_trifan_cap(cfg)) {
    return mesh_topo_n_cells_cap(cfg);
  }
  const int n_c = cfg.multiblock_cart_core_n_c;
  TENRYU_ASSERT(n_c > 0, "Mesh.multiblock_cart_core_n_c must be positive");
  return mesh_topo_checked_int_count(2LL * n_c * n_c,
                                     "multiblock core cell count");
}

inline int mesh_topo_n_cells_bridge(
    const tenryu::core::Config::MeshConfig& cfg) {
  const int n_c = cfg.multiblock_cart_core_n_c;
  const int n_b = cfg.multiblock_cart_core_bridge_layers;
  TENRYU_ASSERT(n_c > 0, "Mesh.multiblock_cart_core_n_c must be positive");
  TENRYU_ASSERT(n_b > 0,
                "Mesh.multiblock_cart_core_bridge_layers must be positive");
  return mesh_topo_checked_int_count(4LL * n_c * n_b,
                                     "multiblock bridge cell count");
}

inline int mesh_topo_n_cells_shell(
    const tenryu::core::Config::MeshConfig& cfg) {
  const int n_c = cfg.multiblock_cart_core_n_c;
  TENRYU_ASSERT(cfg.nr > 0, "Mesh.nr must be positive");
  TENRYU_ASSERT(n_c > 0, "Mesh.multiblock_cart_core_n_c must be positive");
  return mesh_topo_checked_int_count(
      static_cast<long long>(cfg.nr) * (4LL * n_c),
      "multiblock shell cell count");
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

inline int mesh_topo_n_nodes_core(
    const tenryu::core::Config::MeshConfig& cfg) {
  if (mesh_topo_has_trifan_cap(cfg)) {
    return mesh_topo_n_nodes_cap(cfg);
  }
  const int n_c = cfg.multiblock_cart_core_n_c;
  TENRYU_ASSERT(n_c > 0, "Mesh.multiblock_cart_core_n_c must be positive");
  return mesh_topo_checked_int_count(
      (static_cast<long long>(n_c) + 1LL) * (2LL * n_c + 1LL),
      "multiblock core node count");
}

inline int mesh_topo_n_nodes_bridge_interior(
    const tenryu::core::Config::MeshConfig& cfg) {
  const int n_c = cfg.multiblock_cart_core_n_c;
  const int n_b = cfg.multiblock_cart_core_bridge_layers;
  TENRYU_ASSERT(n_c > 0, "Mesh.multiblock_cart_core_n_c must be positive");
  TENRYU_ASSERT(n_b > 0,
                "Mesh.multiblock_cart_core_bridge_layers must be positive");
  return mesh_topo_checked_nonnegative_int_count(
      (4LL * n_c + 1LL) * (static_cast<long long>(n_b) - 1LL),
      "multiblock bridge interior node count");
}

inline int mesh_topo_n_nodes_shell(
    const tenryu::core::Config::MeshConfig& cfg) {
  const int n_c = cfg.multiblock_cart_core_n_c;
  TENRYU_ASSERT(cfg.nr > 0, "Mesh.nr must be positive");
  TENRYU_ASSERT(n_c > 0, "Mesh.multiblock_cart_core_n_c must be positive");
  return mesh_topo_checked_int_count(
      (static_cast<long long>(cfg.nr) + 1LL) * (4LL * n_c + 1LL),
      "multiblock shell node count");
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

inline const BlockInfo& mesh_topo_multiblock_polar_shell_block(
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
  TENRYU_ASSERT(shell_count == 1,
                "multiblock topology requires exactly one POLAR_SHELL block");
  TENRYU_ASSERT(shell_block->n_i_cells > 0 && shell_block->n_j_cells > 0,
                "POLAR_SHELL block dimensions must be positive");
  TENRYU_ASSERT(shell_block->cell_count ==
                    shell_block->n_i_cells * shell_block->n_j_cells,
                "POLAR_SHELL block cell count must match block dimensions");
  TENRYU_ASSERT(shell_block->owned_node_count ==
                    (shell_block->n_i_cells + 1) *
                        (shell_block->n_j_cells + 1),
                "POLAR_SHELL owned nodes must be contiguous block-grid nodes");
  return *shell_block;
}

inline int mesh_topo_multiblock_polar_shell_node_offset(
    const MultiBlockTopology& mb) {
  return mesh_topo_multiblock_polar_shell_block(mb).owned_node_begin;
}

inline int mesh_topo_multiblock_polar_shell_node_offset(
    const MeshTopology& topo) {
  TENRYU_ASSERT(topo.multiblock.has_value(),
                "multiblock polar-shell node offset requires topology metadata");
  return mesh_topo_multiblock_polar_shell_node_offset(*topo.multiblock);
}

inline int mesh_topo_multiblock_polar_shell_nr(const MeshTopology& topo) {
  TENRYU_ASSERT(topo.multiblock.has_value(),
                "multiblock polar-shell nr requires topology metadata");
  const BlockInfo& shell_block =
      mesh_topo_multiblock_polar_shell_block(*topo.multiblock);
  TENRYU_ASSERT(topo.nr == shell_block.n_i_cells,
                "MeshTopology nr must match POLAR_SHELL radial cells");
  return shell_block.n_i_cells;
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

inline int mesh_topo_multiblock_polar_shell_nz(const MeshTopology& topo) {
  TENRYU_ASSERT(topo.multiblock.has_value(),
                "multiblock polar-shell nz requires topology metadata");
  const BlockInfo& shell_block =
      mesh_topo_multiblock_polar_shell_block(*topo.multiblock);
  TENRYU_ASSERT(topo.nz == shell_block.n_j_cells,
                "MeshTopology nz must match POLAR_SHELL angular cells");
  return shell_block.n_j_cells;
}

inline void mesh_topo_assert_multiblock_polar_shell_outer_boundary(
    const MeshTopology& topo) {
  TENRYU_ASSERT(topo.multiblock.has_value(),
                "multiblock polar-shell boundary check requires topology metadata");
  const BlockInfo& shell_block =
      mesh_topo_multiblock_polar_shell_block(*topo.multiblock);
  TENRYU_ASSERT(topo.node_flags.size() == static_cast<std::size_t>(topo.n_nodes),
                "multiblock polar-shell boundary check requires node_flags");
  const int stride = shell_block.n_j_cells + 1;
  const int outer_ring_begin =
      shell_block.owned_node_begin + shell_block.n_i_cells * stride;
  for (int k = 0; k <= shell_block.n_j_cells; ++k) {
    const int n = outer_ring_begin + k;
    TENRYU_ASSERT(n >= 0 && n < topo.n_nodes,
                  "POLAR_SHELL outer boundary node out of range");
    TENRYU_ASSERT((topo.node_flags[static_cast<std::size_t>(n)] &
                   NODE_OUTER_PHYSICAL_BOUNDARY) != 0U,
                  "POLAR_SHELL outer ring must carry NODE_OUTER_PHYSICAL_BOUNDARY");
  }
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

  // Owned geometry caches.
  std::vector<double> cell_vol;
  std::vector<double> cell_area;
  std::vector<double> cell_centroid_r;
  tenryu::core::DeviceArray<double> cell_centroid_r_device;
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
  void recompute_geometry_device_only(double* state_vol);
  MeshGeometryResult sync_device_geometry_to_host_checked(
      const double* state_vol,
      const MeshGeometryCheckOptions& opts);
  void sync_device_geometry_to_host(const double* state_vol);

 private:
  void ensure_persistent_geometry_buffers_(std::size_t n_cells);
  void release_persistent_geometry_buffers_();

  double* d_cell_vol_persistent_ = nullptr;
  double* d_cell_centroid_r_persistent_ = nullptr;
  double* d_cell_centroid_z_persistent_ = nullptr;
  std::size_t d_persistent_capacity_ = 0;
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

}  // namespace tenryu::mesh
