#include "hydro/mortar_slide_line.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tenryu::hydro {

MortarSlideLine mortar_build(const double* m_r,
                             const double* m_z,
                             const int n_m,
                             const double* s_r,
                             const double* s_z,
                             const int n_s) {
  MortarSlideLine line;
  if (n_m > 0) {
    line.m_r.assign(m_r, m_r + n_m);
    line.m_z.assign(m_z, m_z + n_m);
    line.m_arc.assign(static_cast<std::size_t>(n_m), 0.0);
    for (int k = 1; k < n_m; ++k) {
      const double dr = line.m_r[static_cast<std::size_t>(k)] -
                        line.m_r[static_cast<std::size_t>(k - 1)];
      const double dz = line.m_z[static_cast<std::size_t>(k)] -
                        line.m_z[static_cast<std::size_t>(k - 1)];
      line.m_arc[static_cast<std::size_t>(k)] =
          line.m_arc[static_cast<std::size_t>(k - 1)] + std::hypot(dr, dz);
    }
  }
  if (n_s > 0) {
    line.s_r.assign(s_r, s_r + n_s);
    line.s_z.assign(s_z, s_z + n_s);
  }
  return line;
}

MortarFoot mortar_project_point(const MortarSlideLine& line,
                                const double p_r,
                                const double p_z) {
  MortarFoot best{-1, 0.0, 0.0, 0.0, 0.0, 0.0};
  double best_distance2 = std::numeric_limits<double>::infinity();

  for (std::size_t k = 0; k + 1U < line.m_r.size(); ++k) {
    const double tangent_r = line.m_r[k + 1U] - line.m_r[k];
    const double tangent_z = line.m_z[k + 1U] - line.m_z[k];
    const double length2 =
        tangent_r * tangent_r + tangent_z * tangent_z;
    if (!(length2 > 0.0)) {
      continue;
    }

    const double raw_t =
        ((p_r - line.m_r[k]) * tangent_r +
         (p_z - line.m_z[k]) * tangent_z) /
        length2;
    const double t = std::clamp(raw_t, 0.0, 1.0);
    const double foot_r = line.m_r[k] + t * tangent_r;
    const double foot_z = line.m_z[k] + t * tangent_z;
    const double distance_r = p_r - foot_r;
    const double distance_z = p_z - foot_z;
    const double distance2 =
        distance_r * distance_r + distance_z * distance_z;
    if (!(distance2 < best_distance2)) {
      continue;
    }

    const double inverse_length = 1.0 / std::sqrt(length2);
    best.segment = static_cast<int>(k);
    best.t = t;
    best.tangent_r = tangent_r * inverse_length;
    best.tangent_z = tangent_z * inverse_length;
    best.normal_r = -best.tangent_z;
    best.normal_z = best.tangent_r;
    best_distance2 = distance2;
  }

  return best;
}

MortarContactResult mortar_enforce_no_penetration(
    const MortarSlideLine& line,
    const double* m_mass,
    double* m_vr,
    double* m_vz,
    const double* s_mass,
    double* s_vr,
    double* s_vz) {
  MortarContactResult result{};

  for (std::size_t i = 0; i < line.s_r.size(); ++i) {
    const MortarFoot foot =
        mortar_project_point(line, line.s_r[i], line.s_z[i]);
    if (foot.segment < 0) {
      continue;
    }

    const std::size_t k = static_cast<std::size_t>(foot.segment);
    const double weight_a = 1.0 - foot.t;
    const double weight_b = foot.t;
    const double foot_vr = weight_a * m_vr[k] + weight_b * m_vr[k + 1U];
    const double foot_vz = weight_a * m_vz[k] + weight_b * m_vz[k + 1U];
    const double relative_normal_velocity =
        (s_vr[i] - foot_vr) * foot.normal_r +
        (s_vz[i] - foot_vz) * foot.normal_z;
    if (!(relative_normal_velocity < 0.0)) {
      continue;
    }

    const double inverse_effective_mass =
        1.0 / s_mass[i] +
        weight_a * weight_a / m_mass[k] +
        weight_b * weight_b / m_mass[k + 1U];
    const double impulse_scale =
        -relative_normal_velocity / inverse_effective_mass;
    const double impulse_r = impulse_scale * foot.normal_r;
    const double impulse_z = impulse_scale * foot.normal_z;

    s_vr[i] += impulse_r / s_mass[i];
    s_vz[i] += impulse_z / s_mass[i];
    m_vr[k] -= weight_a * impulse_r / m_mass[k];
    m_vz[k] -= weight_a * impulse_z / m_mass[k];
    m_vr[k + 1U] -= weight_b * impulse_r / m_mass[k + 1U];
    m_vz[k + 1U] -= weight_b * impulse_z / m_mass[k + 1U];

    const double foot_vr_after =
        weight_a * m_vr[k] + weight_b * m_vr[k + 1U];
    const double foot_vz_after =
        weight_a * m_vz[k] + weight_b * m_vz[k + 1U];
    result.contact_work +=
        impulse_r * (s_vr[i] - foot_vr_after) +
        impulse_z * (s_vz[i] - foot_vz_after);
    result.impulse_sum_r += impulse_r;
    result.impulse_sum_z += impulse_z;
    ++result.n_active;
  }

  return result;
}

}  // namespace tenryu::hydro
