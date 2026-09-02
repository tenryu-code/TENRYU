#pragma once

namespace tenryu::mesh {

enum class ReferenceCornerClass {
  RegularReference,
  FlatReference,
  InvalidOrReflexReference,
};

struct ReferenceFlatCellPathResult {
  bool has_flat_corner = false;
  bool admissible = true;
  bool embedding_failure = false;
  double first_failure_tau = 1.0;
  double min_oriented_turn = 0.0;
  int first_flat_corner = -1;
};

ReferenceCornerClass classify_reference_corner(
    double previous_r,
    double previous_z,
    double node_r,
    double node_z,
    double next_r,
    double next_z,
    int orientation_sign);

ReferenceFlatCellPathResult evaluate_reference_flat_cell_path(
    const double* reference_r,
    const double* reference_z,
    const double* start_r,
    const double* start_z,
    const double* end_r,
    const double* end_z,
    int nverts,
    int orientation_sign);

}  // namespace tenryu::mesh
