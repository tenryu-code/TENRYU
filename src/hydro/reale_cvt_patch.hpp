#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "hydro/reale_cvt_relocation.hpp"

namespace tenryu::hydro {

struct CvtPatchResult {
  int n_moved;
  int n_empty;
  double cv_initial;
  double cv_final;
  double max_displacement;
  double boundary_drift;
};

CvtPatchResult reale_cvt_patch_relax(
    std::vector<double>& gen_r,
    std::vector<double>& gen_z,
    const std::vector<std::uint8_t>& movable,
    const std::function<double(double, double)>& monitor,
    const CvtDomain& domain,
    int n_iters,
    int grid_nr,
    int grid_nz,
    double damping);

}  // namespace tenryu::hydro
