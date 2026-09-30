#pragma once

#include <vector>

namespace tenryu::numerics {

struct RKL2Coefficients {
  int s = 0;
  std::vector<double> mu_tilde;
  std::vector<double> mu;
  std::vector<double> nu;
  std::vector<double> gamma_tilde;
};

RKL2Coefficients compute_rkl2_coefficients(int s, double damping = 0.05);
int estimate_rkl2_stages(double dt, double dt_explicit, double safety = 0.8);
int estimate_rkl2_subcycles(double dt,
                            double dt_explicit,
                            int max_stages,
                            double safety = 0.8);

}  // namespace tenryu::numerics
