#include "mesh/polygon_mesh.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/error.hpp"

namespace tenryu::mesh {
namespace {

constexpr std::uint64_t kFnv1aOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnv1aPrime = 1099511628211ULL;

static void fnv1a_append_bytes(std::uint64_t& hash,
                               const void* data,
                               const std::size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= static_cast<std::uint64_t>(bytes[i]);
    hash *= kFnv1aPrime;
  }
}

template <typename T>
static void fnv1a_append_value(std::uint64_t& hash, const T& value) {
  fnv1a_append_bytes(hash, &value, sizeof(T));
}

template <typename T>
static void fnv1a_append_vector(std::uint64_t& hash,
                                const std::vector<T>& values) {
  fnv1a_append_bytes(hash, values.data(), values.size() * sizeof(T));
}

static std::uint64_t compute_contract_hash(
    const PolygonMeshTopology& topology,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const std::vector<std::int32_t>& cell_material,
    const std::vector<std::uint64_t>& cell_stable_id,
    const std::vector<std::uint64_t>& node_stable_id,
    const std::vector<std::uint64_t>& cell_parent_id,
    const std::vector<std::uint32_t>& cell_lineage_epoch,
    const std::uint64_t topology_epoch) {
  std::uint64_t hash = kFnv1aOffset;
  fnv1a_append_value(hash, topology.num_cells);
  fnv1a_append_value(hash, topology.num_nodes);
  fnv1a_append_vector(hash, topology.cell_vertex_offsets);
  fnv1a_append_vector(hash, topology.cell_vertex_ids);
  fnv1a_append_vector(hash, node_r);
  fnv1a_append_vector(hash, node_z);
  fnv1a_append_vector(hash, cell_material);
  fnv1a_append_vector(hash, cell_stable_id);
  fnv1a_append_vector(hash, node_stable_id);
  fnv1a_append_vector(hash, cell_parent_id);
  fnv1a_append_vector(hash, cell_lineage_epoch);
  fnv1a_append_value(hash, topology_epoch);
  return hash;
}

// This shoelace area is a validation predicate only. The F4 moments library
// owns production geometry.
static double polygon_signed_area(const PolygonMeshTopology& topology,
                                  const std::vector<double>& node_r,
                                  const std::vector<double>& node_z,
                                  const std::int32_t cell) {
  const std::int32_t begin =
      topology.cell_vertex_offsets[static_cast<std::size_t>(cell)];
  const std::int32_t end =
      topology.cell_vertex_offsets[static_cast<std::size_t>(cell + 1)];
  double twice_area = 0.0;
  for (std::int32_t offset = begin; offset < end; ++offset) {
    const std::int32_t next = offset + 1 == end ? begin : offset + 1;
    const std::int32_t node =
        topology.cell_vertex_ids[static_cast<std::size_t>(offset)];
    const std::int32_t next_node =
        topology.cell_vertex_ids[static_cast<std::size_t>(next)];
    twice_area +=
        node_r[static_cast<std::size_t>(node)] *
            node_z[static_cast<std::size_t>(next_node)] -
        node_r[static_cast<std::size_t>(next_node)] *
            node_z[static_cast<std::size_t>(node)];
  }
  return 0.5 * twice_area;
}

static PolygonMeshValidation validation_failure(
    const PolygonMeshDefect defect,
    const std::int64_t failing_index,
    std::string message) {
  return {false, defect, failing_index, std::move(message)};
}

}  // namespace

PolygonMesh::PolygonMesh(PolygonMeshTopology topology,
                         std::vector<double> node_r,
                         std::vector<double> node_z,
                         std::vector<std::int32_t> cell_material)
    : topology_(std::move(topology)),
      node_r_(std::move(node_r)),
      node_z_(std::move(node_z)),
      cell_material_(std::move(cell_material)) {
  const PolygonMeshValidation validation =
      validate(topology_, node_r_, node_z_, cell_material_);
  TENRYU_ASSERT(validation.ok, validation.message);

  const std::size_t num_cells =
      static_cast<std::size_t>(topology_.num_cells);
  const std::size_t num_nodes =
      static_cast<std::size_t>(topology_.num_nodes);
  cell_stable_id_.resize(num_cells);
  node_stable_id_.resize(num_nodes);
  for (std::size_t cell = 0; cell < num_cells; ++cell) {
    cell_stable_id_[cell] = static_cast<std::uint64_t>(cell);
  }
  for (std::size_t node = 0; node < num_nodes; ++node) {
    node_stable_id_[node] = static_cast<std::uint64_t>(node);
  }
  next_cell_stable_id_ = static_cast<std::uint64_t>(num_cells);
  next_node_stable_id_ = static_cast<std::uint64_t>(num_nodes);
  cell_parent_id_.assign(num_cells, kInvalidStableId);
  cell_lineage_epoch_.assign(num_cells, 0);

  contract_hash_ =
      compute_contract_hash(topology_, node_r_, node_z_, cell_material_,
                            cell_stable_id_, node_stable_id_, cell_parent_id_,
                            cell_lineage_epoch_, topology_epoch_);
}

PolygonMesh PolygonMesh::from_structured_quads(
    const std::int32_t nr,
    const std::int32_t nz,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const std::vector<std::int32_t>& cell_material) {
  PolygonMeshTopology topology;
  topology.num_cells = nr * nz;
  topology.num_nodes = (nr + 1) * (nz + 1);
  topology.cell_vertex_offsets.resize(
      static_cast<std::size_t>(topology.num_cells) + 1U);
  topology.cell_vertex_ids.resize(
      static_cast<std::size_t>(topology.num_cells) * 4U);

  for (std::int32_t cell = 0; cell <= topology.num_cells; ++cell) {
    topology.cell_vertex_offsets[static_cast<std::size_t>(cell)] = 4 * cell;
  }
  for (std::int32_t i = 0; i < nr; ++i) {
    for (std::int32_t j = 0; j < nz; ++j) {
      const std::int32_t cell = i * nz + j;
      const std::int32_t begin = 4 * cell;
      topology.cell_vertex_ids[static_cast<std::size_t>(begin)] =
          i * (nz + 1) + j;
      topology.cell_vertex_ids[static_cast<std::size_t>(begin + 1)] =
          (i + 1) * (nz + 1) + j;
      topology.cell_vertex_ids[static_cast<std::size_t>(begin + 2)] =
          (i + 1) * (nz + 1) + j + 1;
      topology.cell_vertex_ids[static_cast<std::size_t>(begin + 3)] =
          i * (nz + 1) + j + 1;
    }
  }

  const PolygonMeshValidation validation =
      validate(topology, node_r, node_z, cell_material);
  TENRYU_ASSERT(validation.ok, validation.message);
  return PolygonMesh(std::move(topology), node_r, node_z, cell_material);
}

PolygonMeshValidation PolygonMesh::validate(
    const PolygonMeshTopology& topology,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const std::vector<std::int32_t>& cell_material) {
  const std::int64_t expected_offset_count =
      static_cast<std::int64_t>(topology.num_cells) + 1;
  if (expected_offset_count < 1 ||
      topology.cell_vertex_offsets.size() !=
          static_cast<std::size_t>(expected_offset_count)) {
    const std::int64_t failing_index =
        std::min<std::int64_t>(
            static_cast<std::int64_t>(topology.cell_vertex_offsets.size()),
            std::max<std::int64_t>(expected_offset_count, 0));
    return validation_failure(
        PolygonMeshDefect::kOffsetsMalformed, failing_index,
        "PolygonMesh cell_vertex_offsets size is malformed at index=" +
            std::to_string(failing_index) + ": expected " +
            std::to_string(expected_offset_count) + " entries, got " +
            std::to_string(topology.cell_vertex_offsets.size()));
  }
  if (topology.cell_vertex_offsets[0] != 0) {
    return validation_failure(
        PolygonMeshDefect::kOffsetsMalformed, 0,
        "PolygonMesh cell_vertex_offsets is malformed at index=0: first "
        "offset must be zero");
  }
  for (std::int32_t cell = 0; cell < topology.num_cells; ++cell) {
    const std::int32_t begin =
        topology.cell_vertex_offsets[static_cast<std::size_t>(cell)];
    const std::int32_t end =
        topology.cell_vertex_offsets[static_cast<std::size_t>(cell + 1)];
    if (end < begin) {
      return validation_failure(
          PolygonMeshDefect::kOffsetsMalformed,
          static_cast<std::int64_t>(cell) + 1,
          "PolygonMesh cell_vertex_offsets decreases at index=" +
              std::to_string(static_cast<std::int64_t>(cell) + 1));
    }
    const std::int32_t degree = end - begin;
    if (degree < 3 || degree > kMaxPolygonDegree) {
      return validation_failure(
          PolygonMeshDefect::kDegreeOutOfRange, cell,
          "PolygonMesh cell degree is out of range at cell=" +
              std::to_string(cell) + ": degree=" + std::to_string(degree));
    }
  }

  const std::int64_t expected_id_count =
      topology.cell_vertex_offsets.back();
  if (expected_id_count < 0 ||
      topology.cell_vertex_ids.size() !=
          static_cast<std::size_t>(expected_id_count)) {
    const std::int64_t failing_index =
        std::min<std::int64_t>(
            static_cast<std::int64_t>(topology.cell_vertex_ids.size()),
            std::max<std::int64_t>(expected_id_count, 0));
    return validation_failure(
        PolygonMeshDefect::kArraySizeMismatch, failing_index,
        "PolygonMesh cell_vertex_ids size mismatch at index=" +
            std::to_string(failing_index) + ": expected " +
            std::to_string(expected_id_count) + " entries, got " +
            std::to_string(topology.cell_vertex_ids.size()));
  }

  const std::int64_t expected_node_count = topology.num_nodes;
  if (expected_node_count < 0 ||
      node_r.size() != static_cast<std::size_t>(expected_node_count)) {
    const std::int64_t failing_index =
        std::min<std::int64_t>(
            static_cast<std::int64_t>(node_r.size()),
            std::max<std::int64_t>(expected_node_count, 0));
    return validation_failure(
        PolygonMeshDefect::kArraySizeMismatch, failing_index,
        "PolygonMesh node_r size mismatch at index=" +
            std::to_string(failing_index) + ": expected " +
            std::to_string(expected_node_count) + " entries, got " +
            std::to_string(node_r.size()));
  }
  if (expected_node_count < 0 ||
      node_z.size() != static_cast<std::size_t>(expected_node_count)) {
    const std::int64_t failing_index =
        std::min<std::int64_t>(
            static_cast<std::int64_t>(node_z.size()),
            std::max<std::int64_t>(expected_node_count, 0));
    return validation_failure(
        PolygonMeshDefect::kArraySizeMismatch, failing_index,
        "PolygonMesh node_z size mismatch at index=" +
            std::to_string(failing_index) + ": expected " +
            std::to_string(expected_node_count) + " entries, got " +
            std::to_string(node_z.size()));
  }

  const std::int64_t expected_material_count = topology.num_cells;
  if (expected_material_count < 0 ||
      cell_material.size() !=
          static_cast<std::size_t>(expected_material_count)) {
    const std::int64_t failing_index =
        std::min<std::int64_t>(
            static_cast<std::int64_t>(cell_material.size()),
            std::max<std::int64_t>(expected_material_count, 0));
    return validation_failure(
        PolygonMeshDefect::kArraySizeMismatch, failing_index,
        "PolygonMesh cell_material size mismatch at index=" +
            std::to_string(failing_index) + ": expected " +
            std::to_string(expected_material_count) + " entries, got " +
            std::to_string(cell_material.size()));
  }

  for (std::size_t offset = 0; offset < topology.cell_vertex_ids.size();
       ++offset) {
    const std::int32_t node = topology.cell_vertex_ids[offset];
    if (node < 0 || node >= topology.num_nodes) {
      return validation_failure(
          PolygonMeshDefect::kVertexIdOutOfRange,
          static_cast<std::int64_t>(offset),
          "PolygonMesh vertex id is out of range at index=" +
              std::to_string(offset) + ": node=" + std::to_string(node));
    }
  }

  std::vector<bool> referenced(static_cast<std::size_t>(topology.num_nodes),
                               false);
  for (std::int32_t cell = 0; cell < topology.num_cells; ++cell) {
    const std::int32_t begin =
        topology.cell_vertex_offsets[static_cast<std::size_t>(cell)];
    const std::int32_t end =
        topology.cell_vertex_offsets[static_cast<std::size_t>(cell + 1)];
    for (std::int32_t offset = begin; offset < end; ++offset) {
      const std::int32_t node =
          topology.cell_vertex_ids[static_cast<std::size_t>(offset)];
      referenced[static_cast<std::size_t>(node)] = true;
      for (std::int32_t previous = begin; previous < offset; ++previous) {
        if (topology.cell_vertex_ids[static_cast<std::size_t>(previous)] ==
            node) {
          return validation_failure(
              PolygonMeshDefect::kRepeatedVertexInCell, cell,
              "PolygonMesh cell has a repeated vertex at cell=" +
                  std::to_string(cell) + ": node=" + std::to_string(node));
        }
      }
    }
  }

  for (std::int32_t node = 0; node < topology.num_nodes; ++node) {
    if (!referenced[static_cast<std::size_t>(node)]) {
      return validation_failure(
          PolygonMeshDefect::kDanglingNode, node,
          "PolygonMesh has a dangling node at node=" + std::to_string(node));
    }
  }

  for (std::int32_t node = 0; node < topology.num_nodes; ++node) {
    if (!(node_r[static_cast<std::size_t>(node)] >= 0.0)) {
      return validation_failure(
          PolygonMeshDefect::kNegativeRadius, node,
          "PolygonMesh radius is negative at node=" + std::to_string(node) +
              ": r=" +
              std::to_string(node_r[static_cast<std::size_t>(node)]));
    }
  }

  for (std::int32_t cell = 0; cell < topology.num_cells; ++cell) {
    const double area =
        polygon_signed_area(topology, node_r, node_z, cell);
    if (!(area > 0.0)) {
      return validation_failure(
          PolygonMeshDefect::kNonPositiveArea, cell,
          "PolygonMesh signed area is non-positive at cell=" +
              std::to_string(cell) + ": area=" + std::to_string(area));
    }
  }

  return {};
}

}  // namespace tenryu::mesh
