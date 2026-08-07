#pragma once

#include <functional>
#include <vector>

namespace tenryu::hydro {

struct CvtDomain {
  double r0;
  double r1;
  double z0;
  double z1;
};

struct CvtRelaxResult {
  int n_empty;
  double cv_initial;
  double cv_final;
  std::vector<double> cv_per_iteration;
};

CvtRelaxResult reale_cvt_relax(
    std::vector<double>& gen_r,
    std::vector<double>& gen_z,
    const std::function<double(double, double)>& monitor,
    const CvtDomain& domain,
    int n_iters,
    int grid_nr,
    int grid_nz,
    double damping);

}  // namespace tenryu::hydro
