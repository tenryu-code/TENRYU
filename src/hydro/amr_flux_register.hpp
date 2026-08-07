#pragma once

#include <cstdint>
#include <vector>

namespace tenryu::hydro {

struct AmrFluxRegister {
  // One coarse face is adjoined by refine_ratio fine faces.
  int n_coarse_faces;
  int refine_ratio;
  int n_quantities;
  // [face][quantity]: coarse-side flux integral F_c * A_c * dt.
  std::vector<double> coarse_flux;
  // [face][quantity]: sum of fine-side flux integrals F_f * A_f * dt.
  std::vector<double> fine_flux_sum;
  // [face][sub]: nonzero after that fine slot contributes in this cycle.
  std::vector<std::uint8_t> fine_slot_contributed;
};

AmrFluxRegister amr_flux_register_create(int n_coarse_faces,
                                         int refine_ratio,
                                         int n_quantities);

void amr_flux_register_reset(AmrFluxRegister& flux_register);

void amr_flux_register_add_coarse(AmrFluxRegister& flux_register,
                                  int face,
                                  const double* q_flux);

void amr_flux_register_add_fine(AmrFluxRegister& flux_register,
                                int face,
                                int sub,
                                const double* q_flux);

void amr_flux_register_reflux(const AmrFluxRegister& flux_register,
                              double* corrections);

}  // namespace tenryu::hydro
