#pragma once

#include <cstdint>

namespace tenryu::hydro {

struct EulerWindowSpec {
  enum class Shape : std::uint8_t {
    Rectangle = 0,
    Annulus = 1,
  };

  Shape shape;
  double r0;
  double r1;
  double z0;
  double z1;
  double cr;
  double cz;
  double rad_in;
  double rad_out;
  double transition_width;
};

void euler_window_cell_weights(const EulerWindowSpec& spec,
                               const double* cell_cr,
                               const double* cell_cz,
                               int n_cells,
                               double* w_cell);

void euler_window_node_weights(const int* cell_node_csr_offsets,
                               const int* cell_node_csr_indices,
                               const std::uint8_t* cell_nverts,
                               int n_cells,
                               int corner_stride,
                               int n_nodes,
                               const double* w_cell,
                               double* w_node);

void euler_window_blend_targets(const double* w_node,
                                int n_nodes,
                                const double* x_lagr_r,
                                const double* x_lagr_z,
                                const double* x_euler_r,
                                const double* x_euler_z,
                                double* x_target_r,
                                double* x_target_z);

}  // namespace tenryu::hydro
