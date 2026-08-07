#pragma once

#include <vector>

namespace tenryu::verification {

struct MarshakReferenceSample {
  double x_cm = 0.0;
  double T_over_Tsrc = 0.0;
};

[[nodiscard]] const std::vector<MarshakReferenceSample>&
marshak_reference_profile_ct15();

[[nodiscard]] double marshak_reference_Tsrc_eV();
[[nodiscard]] double marshak_reference_ct_end_cm();

}  // namespace tenryu::verification

