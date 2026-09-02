#include "mesh/z_reflection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "mesh/mesh.hpp"

namespace tenryu::mesh {
namespace {

ZReflectionMaps fail_maps(ZReflectionMaps maps, std::string failure) {
  maps.valid = false;
  maps.failure = std::move(failure);
  return maps;
}

double audit_channel(const std::vector<int>& mirror,
                     const std::vector<double>& values,
                     const double transform_sign) {
  if (values.empty() || values.size() != mirror.size()) {
    return 0.0;
  }

  long double value_square_sum = 0.0L;
  for (const double value : values) {
    value_square_sum +=
        static_cast<long double>(value) * static_cast<long double>(value);
  }
  const long double value_rms =
      std::sqrt(value_square_sum / static_cast<long double>(values.size()));
  if (!(value_rms > 0.0L)) {
    return 0.0;
  }

  long double defect_square_sum = 0.0L;
  std::size_t pair_count = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const int mirrored = mirror[i];
    if (mirrored < 0 ||
        static_cast<std::size_t>(mirrored) >= values.size() ||
        static_cast<int>(i) > mirrored) {
      continue;
    }
    const long double defect =
        0.5L * (static_cast<long double>(values[i]) -
                static_cast<long double>(transform_sign) *
                    static_cast<long double>(values[static_cast<std::size_t>(mirrored)]));
    defect_square_sum += defect * defect;
    ++pair_count;
  }
  if (pair_count == 0U) {
    return 0.0;
  }
  const long double defect_rms =
      std::sqrt(defect_square_sum / static_cast<long double>(pair_count));
  return static_cast<double>(defect_rms / value_rms);
}

const std::vector<int>* momentum_mirror(const ZReflectionMaps& maps,
                                        const std::vector<double>& momentum) {
  if (momentum.size() == maps.cell_mirror.size()) {
    return &maps.cell_mirror;
  }
  if (momentum.size() == maps.node_mirror.size()) {
    return &maps.node_mirror;
  }
  return nullptr;
}

int active_cell_node_end(const MultiBlockTopology& topology,
                         const std::size_t cell,
                         const int node_begin,
                         const int node_end) {
  const auto& face_offsets = topology.face_adj_csr_offsets;
  const auto& face_indices = topology.face_adj_csr_indices;
  const auto& face_tags = topology.face_bc_tags;
  if (face_offsets.size() != topology.cell_node_csr_offsets.size() ||
      face_indices.size() != topology.cell_node_csr_indices.size() ||
      face_tags.size() != topology.cell_node_csr_indices.size()) {
    return node_end;
  }
  const int face_begin = face_offsets[cell];
  const int face_end = face_offsets[cell + 1U];
  if (face_begin < 0 || face_end < face_begin ||
      face_end - face_begin != node_end - node_begin ||
      static_cast<std::size_t>(face_end) > face_indices.size()) {
    return node_end;
  }

  int last_active = -1;
  const int interior_tag = static_cast<int>(BoundaryKind::Interior);
  for (int local = 0; local < face_end - face_begin; ++local) {
    const std::size_t slot =
        static_cast<std::size_t>(face_begin + local);
    if (face_indices[slot] >= 0 || face_tags[slot] != interior_tag) {
      last_active = local;
    }
  }
  return last_active >= 0 ? node_begin + last_active + 1 : node_end;
}

}  // namespace

ZReflectionMaps build_z_reflection_maps(
    const MultiBlockTopology& topology,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  ZReflectionMaps maps;
  if (node_r.size() != node_z.size()) {
    return fail_maps(std::move(maps), "node coordinate extent mismatch");
  }
  for (std::size_t p = 0; p < node_z.size(); ++p) {
    if (!std::isfinite(node_r[p]) || !std::isfinite(node_z[p])) {
      return fail_maps(std::move(maps),
                       "node " + std::to_string(p) +
                           " has non-finite coordinates");
    }
  }

  double domain_scale = 0.0;
  for (const double z : node_z) {
    domain_scale = std::max(domain_scale, std::abs(z));
  }
  const double tolerance = 1.0e-9 * domain_scale;

  std::vector<int> node_order(node_z.size());
  std::iota(node_order.begin(), node_order.end(), 0);
  std::sort(node_order.begin(), node_order.end(), [&](const int lhs, const int rhs) {
    if (node_z[static_cast<std::size_t>(lhs)] !=
        node_z[static_cast<std::size_t>(rhs)]) {
      return node_z[static_cast<std::size_t>(lhs)] <
             node_z[static_cast<std::size_t>(rhs)];
    }
    if (node_r[static_cast<std::size_t>(lhs)] !=
        node_r[static_cast<std::size_t>(rhs)]) {
      return node_r[static_cast<std::size_t>(lhs)] <
             node_r[static_cast<std::size_t>(rhs)];
    }
    return lhs < rhs;
  });

  maps.node_mirror.assign(node_z.size(), -1);
  for (std::size_t p = 0; p < node_z.size(); ++p) {
    if (std::abs(node_z[p]) <= tolerance) {
      maps.node_mirror[p] = static_cast<int>(p);
      continue;
    }

    const double target_z = -node_z[p];
    const auto begin = std::lower_bound(
        node_order.begin(), node_order.end(), target_z - tolerance,
        [&](const int node, const double z) {
          return node_z[static_cast<std::size_t>(node)] < z;
        });
    int best = -1;
    double best_distance_square = std::numeric_limits<double>::infinity();
    for (auto it = begin; it != node_order.end(); ++it) {
      const int candidate = *it;
      const double candidate_z = node_z[static_cast<std::size_t>(candidate)];
      if (candidate_z > target_z + tolerance) {
        break;
      }
      const double dr = node_r[static_cast<std::size_t>(candidate)] - node_r[p];
      const double dz = candidate_z - target_z;
      const double distance_square = dr * dr + dz * dz;
      if (distance_square <= tolerance * tolerance &&
          (distance_square < best_distance_square ||
           (distance_square == best_distance_square && candidate < best))) {
        best = candidate;
        best_distance_square = distance_square;
      }
    }
    if (best < 0) {
      return fail_maps(std::move(maps),
                       "node " + std::to_string(p) +
                           " has no reflected match");
    }
    maps.node_mirror[p] = best;
  }

  for (std::size_t p = 0; p < maps.node_mirror.size(); ++p) {
    const int mirrored = maps.node_mirror[p];
    if (mirrored < 0 ||
        static_cast<std::size_t>(mirrored) >= maps.node_mirror.size() ||
        maps.node_mirror[static_cast<std::size_t>(mirrored)] !=
            static_cast<int>(p)) {
      return fail_maps(std::move(maps),
                       "node " + std::to_string(p) +
                           " fails the reflection involution");
    }
    if (std::abs(node_z[p]) <= tolerance && mirrored != static_cast<int>(p)) {
      return fail_maps(std::move(maps),
                       "equatorial node " + std::to_string(p) +
                           " is not fixed");
    }
  }

  const auto& offsets = topology.cell_node_csr_offsets;
  const auto& indices = topology.cell_node_csr_indices;
  if (offsets.empty() || offsets.front() != 0) {
    return fail_maps(std::move(maps), "cell CSR offsets are missing or invalid");
  }
  const std::size_t n_cells = offsets.size() - 1U;
  if (offsets.back() < 0 ||
      static_cast<std::size_t>(offsets.back()) != indices.size()) {
    return fail_maps(std::move(maps), "cell CSR payload extent mismatch");
  }

  std::vector<std::vector<int>> cell_node_sets(n_cells);
  std::map<std::vector<int>, int> cell_by_nodes;
  for (std::size_t c = 0; c < n_cells; ++c) {
    const int begin = offsets[c];
    const int storage_end = offsets[c + 1U];
    const int end =
        active_cell_node_end(topology, c, begin, storage_end);
    if (begin < 0 || end < begin ||
        end > storage_end ||
        static_cast<std::size_t>(storage_end) > indices.size()) {
      return fail_maps(std::move(maps),
                       "cell " + std::to_string(c) +
                           " has invalid CSR offsets");
    }
    auto& nodes = cell_node_sets[c];
    nodes.reserve(static_cast<std::size_t>(end - begin));
    for (int k = begin; k < end; ++k) {
      const int node = indices[static_cast<std::size_t>(k)];
      if (node < 0 || static_cast<std::size_t>(node) >= node_r.size()) {
        return fail_maps(std::move(maps),
                         "cell " + std::to_string(c) +
                             " references invalid node " +
                             std::to_string(node));
      }
      nodes.push_back(node);
    }
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    const auto inserted = cell_by_nodes.emplace(nodes, static_cast<int>(c));
    if (!inserted.second) {
      return fail_maps(std::move(maps),
                       "cell " + std::to_string(c) +
                           " duplicates the node set of cell " +
                           std::to_string(inserted.first->second));
    }
  }

  maps.cell_mirror.assign(n_cells, -1);
  for (std::size_t c = 0; c < n_cells; ++c) {
    std::vector<int> mirrored_nodes;
    mirrored_nodes.reserve(cell_node_sets[c].size());
    for (const int node : cell_node_sets[c]) {
      mirrored_nodes.push_back(
          maps.node_mirror[static_cast<std::size_t>(node)]);
    }
    std::sort(mirrored_nodes.begin(), mirrored_nodes.end());
    const auto match = cell_by_nodes.find(mirrored_nodes);
    if (match == cell_by_nodes.end()) {
      return fail_maps(std::move(maps),
                       "cell " + std::to_string(c) +
                           " has no reflected node-set match");
    }
    maps.cell_mirror[c] = match->second;
  }
  for (std::size_t c = 0; c < maps.cell_mirror.size(); ++c) {
    const int mirrored = maps.cell_mirror[c];
    if (mirrored < 0 ||
        static_cast<std::size_t>(mirrored) >= maps.cell_mirror.size() ||
        maps.cell_mirror[static_cast<std::size_t>(mirrored)] !=
            static_cast<int>(c)) {
      return fail_maps(std::move(maps),
                       "cell " + std::to_string(c) +
                           " fails the reflection involution");
    }
  }

  maps.valid = true;
  maps.failure.clear();
  return maps;
}

ZReflectionAudit audit_z_reflection(
    const ZReflectionMaps& maps,
    const std::vector<double>& mass,
    const std::vector<double>& mom_r,
    const std::vector<double>& mom_z,
    const std::vector<double>& energy,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  ZReflectionAudit audit;
  if (!maps.valid) {
    return audit;
  }

  audit.odd_mass = audit_channel(maps.cell_mirror, mass, 1.0);
  if (const auto* mirror = momentum_mirror(maps, mom_r)) {
    audit.odd_mom_r = audit_channel(*mirror, mom_r, 1.0);
  }
  if (const auto* mirror = momentum_mirror(maps, mom_z)) {
    audit.odd_mom_z = audit_channel(*mirror, mom_z, -1.0);
  }
  audit.odd_energy = audit_channel(maps.cell_mirror, energy, 1.0);
  audit.odd_node_r = audit_channel(maps.node_mirror, node_r, 1.0);
  audit.odd_node_z = audit_channel(maps.node_mirror, node_z, -1.0);
  return audit;
}

}  // namespace tenryu::mesh
