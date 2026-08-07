#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tenryu::mesh {

inline constexpr int kMaxPolygonDegree = 8;  // Warp-regular CSR degree bound.
inline constexpr std::uint64_t kInvalidStableId =
    static_cast<std::uint64_t>(-1);

struct PolygonMeshTopology {
  // CSR cell -> vertex (node) lists, CCW order per cell.
  std::vector<std::int32_t> cell_vertex_offsets;
  std::vector<std::int32_t> cell_vertex_ids;
  std::int32_t num_cells = 0;
  std::int32_t num_nodes = 0;
};

enum class PolygonMeshDefect : std::uint8_t {
  kNone,
  kOffsetsMalformed,
  kDegreeOutOfRange,
  kVertexIdOutOfRange,
  kRepeatedVertexInCell,
  kDanglingNode,
  kNegativeRadius,
  kNonPositiveArea,
  kArraySizeMismatch,
};

struct PolygonMeshValidation {
  bool ok = true;
  PolygonMeshDefect defect = PolygonMeshDefect::kNone;
  std::int64_t failing_index = -1;
  std::string message;
};

class PolygonMesh {
 public:
  // Construct from explicit arrays, validate them, and compute the initial hash.
  PolygonMesh(PolygonMeshTopology topology,
              std::vector<double> node_r,
              std::vector<double> node_z,
              std::vector<std::int32_t> cell_material);

  // Import a structured quad mesh. Nodes use (nr+1)x(nz+1) row-major indexing:
  // node(i,j) = i*(nz+1)+j. Cells use nr x nz row-major indexing:
  // cell(i,j) = i*nz+j, with vertices (i,j), (i+1,j), (i+1,j+1),
  // (i,j+1) in CCW order. Stable IDs equal the row-major indices.
  static PolygonMesh from_structured_quads(
      std::int32_t nr,
      std::int32_t nz,
      const std::vector<double>& node_r,
      const std::vector<double>& node_z,
      const std::vector<std::int32_t>& cell_material);

  static PolygonMeshValidation validate(
      const PolygonMeshTopology& topology,
      const std::vector<double>& node_r,
      const std::vector<double>& node_z,
      const std::vector<std::int32_t>& cell_material);

  [[nodiscard]] const PolygonMeshTopology& topology() const noexcept {
    return topology_;
  }

  [[nodiscard]] const std::vector<double>& node_r() const noexcept {
    return node_r_;
  }

  [[nodiscard]] const std::vector<double>& node_z() const noexcept {
    return node_z_;
  }

  [[nodiscard]] const std::vector<std::int32_t>& cell_material() const noexcept {
    return cell_material_;
  }

  [[nodiscard]] const std::vector<std::uint64_t>& cell_stable_id() const noexcept {
    return cell_stable_id_;
  }

  [[nodiscard]] const std::vector<std::uint64_t>& node_stable_id() const noexcept {
    return node_stable_id_;
  }

  [[nodiscard]] const std::vector<std::uint64_t>& cell_parent_id() const noexcept {
    return cell_parent_id_;
  }

  [[nodiscard]] const std::vector<std::uint32_t>&
  cell_lineage_epoch() const noexcept {
    return cell_lineage_epoch_;
  }

  [[nodiscard]] const std::uint64_t& topology_epoch() const noexcept {
    return topology_epoch_;
  }

  [[nodiscard]] const std::uint64_t& contract_hash() const noexcept {
    return contract_hash_;
  }

 private:
  PolygonMeshTopology topology_;
  std::vector<double> node_r_;
  std::vector<double> node_z_;
  std::vector<std::int32_t> cell_material_;

  // Stable IDs never change on future topology events. New IDs are allocated
  // monotonically from these counters.
  std::vector<std::uint64_t> cell_stable_id_;
  std::vector<std::uint64_t> node_stable_id_;
  std::uint64_t next_cell_stable_id_ = 0;
  std::uint64_t next_node_stable_id_ = 0;

  std::vector<std::uint64_t> cell_parent_id_;
  std::vector<std::uint32_t> cell_lineage_epoch_;

  // Incremented on every future topology transaction commit. This epoch is
  // distinct from reference epochs.
  std::uint64_t topology_epoch_ = 0;
  std::uint64_t contract_hash_ = 0;
};

}  // namespace tenryu::mesh
