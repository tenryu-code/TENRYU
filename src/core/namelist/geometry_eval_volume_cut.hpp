#pragma once

#include <cstddef>
#include <vector>

namespace tenryu::core::namelist {

struct GeometryCallables;
using NamelistCallables = GeometryCallables;

struct VolumeCutResult {
  double total_volume = 0.0;
  double rho_volume_avg = 0.0;
  double Te_volume_avg = 0.0;
  double Ti_volume_avg = 0.0;
  std::vector<double> volfrac;
  bool converged = false;
  int max_depth_reached = 0;
  int leaf_count = 0;
};

VolumeCutResult adaptive_volume_cut_sample_cell(double r0,
                                                double z0,
                                                double r1,
                                                double z1,
                                                double r2,
                                                double z2,
                                                double r3,
                                                double z3,
                                                std::size_t n_mat,
                                                const NamelistCallables& callables,
                                                int max_depth,
                                                double volfrac_tol,
                                                bool use_3x3_quadrature);

}  // namespace tenryu::core::namelist
