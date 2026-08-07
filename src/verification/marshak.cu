#include "verification/marshak.cuh"

namespace tenryu::verification {

const std::vector<MarshakReferenceSample>& marshak_reference_profile_ct15() {
  static const std::vector<MarshakReferenceSample> kProfile = {
      {0.0, 0.98},
      {0.5, 0.89},
      {1.0, 0.72},
      {1.5, 0.50},
      {2.0, 0.30},
      {2.5, 0.12},
      {3.0, 0.03},
  };
  return kProfile;
}

double marshak_reference_Tsrc_eV() {
  return 1000.0;
}

double marshak_reference_ct_end_cm() {
  return 15.0;
}

}  // namespace tenryu::verification

