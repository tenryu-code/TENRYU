#pragma once

#include <vector>

#include "core/radiation_group_structure.hpp"
#include "materials/ionmix_reader.hpp"

namespace tenryu::radiation {

inline std::vector<double> repack_group_bounds_for_hard_xray(
    const int n_groups,
    const std::vector<double>& compute_T_range_eV) {
  return tenryu::core::repack_radiation_group_bounds_for_hard_xray(
      n_groups, compute_T_range_eV);
}

inline int count_groups_inside_energy_band(const std::vector<double>& bounds_eV,
                                           const double lo_eV,
                                           const double hi_eV) {
  return tenryu::core::count_radiation_groups_inside_energy_band(
      bounds_eV, lo_eV, hi_eV);
}

materials::IonmixOpacityData resample_opacity_groups_to_bounds(
    const materials::IonmixOpacityData& input,
    const std::vector<double>& target_bounds_eV);

}  // namespace tenryu::radiation
