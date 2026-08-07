#pragma once

#include <vector>

namespace tenryu::hydro {

struct MortarSlideLine {
  std::vector<double> m_r;
  std::vector<double> m_z;
  std::vector<double> s_r;
  std::vector<double> s_z;
  std::vector<double> m_arc;
};

MortarSlideLine mortar_build(const double* m_r,
                             const double* m_z,
                             int n_m,
                             const double* s_r,
                             const double* s_z,
                             int n_s);

struct MortarFoot {
  int segment;
  double t;
  double normal_r;
  double normal_z;
  double tangent_r;
  double tangent_z;
};

MortarFoot mortar_project_point(const MortarSlideLine& line,
                                double p_r,
                                double p_z);

struct MortarContactResult {
  int n_active;
  double impulse_sum_r;
  double impulse_sum_z;
  double contact_work;
};

MortarContactResult mortar_enforce_no_penetration(
    const MortarSlideLine& line,
    const double* m_mass,
    double* m_vr,
    double* m_vz,
    const double* s_mass,
    double* s_vr,
    double* s_vz);

}  // namespace tenryu::hydro
