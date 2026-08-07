#include "hydro/euler_window_blend.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tenryu::hydro {
namespace {

void validate_spec(const EulerWindowSpec& spec) {
  if (spec.transition_width <= 0.0) {
    throw std::invalid_argument(
        "Eulerian window transition width must be positive");
  }

  switch (spec.shape) {
    case EulerWindowSpec::Shape::Rectangle:
      if (spec.r1 <= spec.r0 || spec.z1 <= spec.z0) {
        throw std::invalid_argument(
            "Eulerian rectangle window bounds must have positive extent");
      }
      break;
    case EulerWindowSpec::Shape::Annulus:
      if (spec.rad_out <= spec.rad_in) {
        throw std::invalid_argument(
            "Eulerian annulus window radii must have positive extent");
      }
      break;
    default:
      throw std::invalid_argument("Eulerian window shape is invalid");
  }
}

double signed_distance_outside(const EulerWindowSpec& spec,
                               const double r,
                               const double z) {
  if (spec.shape == EulerWindowSpec::Shape::Rectangle) {
    return std::max(
        std::max(spec.r0 - r, r - spec.r1),
        std::max(spec.z0 - z, z - spec.z1));
  }

  const double radius = std::hypot(r - spec.cr, z - spec.cz);
  return std::max(spec.rad_in - radius, radius - spec.rad_out);
}

double transition_weight(const double signed_distance,
                         const double transition_width) {
  if (signed_distance <= 0.0) {
    return 1.0;
  }
  if (signed_distance >= transition_width) {
    return 0.0;
  }

  const double t = signed_distance / transition_width;
  const double smoothstep = t * t * (3.0 - 2.0 * t);
  return 1.0 - smoothstep;
}

}  // namespace

void euler_window_cell_weights(const EulerWindowSpec& spec,
                               const double* cell_cr,
                               const double* cell_cz,
                               const int n_cells,
                               double* w_cell) {
  validate_spec(spec);
  if (cell_cr == nullptr || cell_cz == nullptr || w_cell == nullptr) {
    throw std::invalid_argument(
        "Eulerian window cell-weight arrays must be non-null");
  }

  for (int cell = 0; cell < n_cells; ++cell) {
    const std::size_t index = static_cast<std::size_t>(cell);
    w_cell[index] = transition_weight(
        signed_distance_outside(spec, cell_cr[index], cell_cz[index]),
        spec.transition_width);
  }
}

void euler_window_node_weights(const int* cell_node_csr_offsets,
                               const int* cell_node_csr_indices,
                               const std::uint8_t* cell_nverts,
                               const int n_cells,
                               const int corner_stride,
                               const int n_nodes,
                               const double* w_cell,
                               double* w_node) {
  if (cell_node_csr_offsets == nullptr ||
      cell_node_csr_indices == nullptr || w_cell == nullptr ||
      w_node == nullptr) {
    throw std::invalid_argument(
        "Eulerian window node-weight arrays must be non-null");
  }

  std::fill(w_node, w_node + n_nodes, 0.0);
  std::vector<int> incident_count(static_cast<std::size_t>(n_nodes), 0);
  for (int cell = 0; cell < n_cells; ++cell) {
    const int nverts =
        cell_nverts == nullptr ? 4 : static_cast<int>(cell_nverts[cell]);
    if (nverts > corner_stride) {
      throw std::invalid_argument(
          "Eulerian window cell vertex count exceeds corner stride");
    }
    const int offset = cell_node_csr_offsets[cell];
    for (int local = 0; local < nverts; ++local) {
      const int node = cell_node_csr_indices[offset + local];
      w_node[node] += w_cell[cell];
      ++incident_count[static_cast<std::size_t>(node)];
    }
  }

  for (int node = 0; node < n_nodes; ++node) {
    const int count = incident_count[static_cast<std::size_t>(node)];
    if (count > 0) {
      w_node[node] /= static_cast<double>(count);
    }
  }
}

void euler_window_blend_targets(const double* w_node,
                                const int n_nodes,
                                const double* x_lagr_r,
                                const double* x_lagr_z,
                                const double* x_euler_r,
                                const double* x_euler_z,
                                double* x_target_r,
                                double* x_target_z) {
  if (w_node == nullptr || x_lagr_r == nullptr || x_lagr_z == nullptr ||
      x_euler_r == nullptr || x_euler_z == nullptr ||
      x_target_r == nullptr || x_target_z == nullptr) {
    throw std::invalid_argument(
        "Eulerian window blend arrays must be non-null");
  }

  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t index = static_cast<std::size_t>(node);
    const double weight = w_node[index];
    if (weight == 1.0) {
      x_target_r[index] = x_euler_r[index];
      x_target_z[index] = x_euler_z[index];
    } else if (weight == 0.0) {
      x_target_r[index] = x_lagr_r[index];
      x_target_z[index] = x_lagr_z[index];
    } else {
      x_target_r[index] =
          weight * x_euler_r[index] +
          (1.0 - weight) * x_lagr_r[index];
      x_target_z[index] =
          weight * x_euler_z[index] +
          (1.0 - weight) * x_lagr_z[index];
    }
  }
}

}  // namespace tenryu::hydro
