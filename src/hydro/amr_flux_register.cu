#include "hydro/amr_flux_register.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace tenryu::hydro {
namespace {

void validate_face(const AmrFluxRegister& flux_register, const int face) {
  if (face < 0 || face >= flux_register.n_coarse_faces) {
    throw std::out_of_range("AMR flux-register face is out of range");
  }
}

std::size_t flux_offset(const AmrFluxRegister& flux_register,
                        const int face) {
  return static_cast<std::size_t>(face) *
         static_cast<std::size_t>(flux_register.n_quantities);
}

std::size_t fine_slot_index(const AmrFluxRegister& flux_register,
                            const int face,
                            const int sub) {
  return static_cast<std::size_t>(face) *
             static_cast<std::size_t>(flux_register.refine_ratio) +
         static_cast<std::size_t>(sub);
}

}  // namespace

AmrFluxRegister amr_flux_register_create(const int n_coarse_faces,
                                         const int refine_ratio,
                                         const int n_quantities) {
  if (n_coarse_faces <= 0 || refine_ratio <= 0 || n_quantities <= 0) {
    throw std::invalid_argument(
        "AMR flux-register dimensions and refinement ratio must be positive");
  }

  const std::size_t n_flux_values =
      static_cast<std::size_t>(n_coarse_faces) *
      static_cast<std::size_t>(n_quantities);
  const std::size_t n_fine_slots =
      static_cast<std::size_t>(n_coarse_faces) *
      static_cast<std::size_t>(refine_ratio);
  return AmrFluxRegister{
      n_coarse_faces,
      refine_ratio,
      n_quantities,
      std::vector<double>(n_flux_values, 0.0),
      std::vector<double>(n_flux_values, 0.0),
      std::vector<std::uint8_t>(n_fine_slots, 0U)};
}

void amr_flux_register_reset(AmrFluxRegister& flux_register) {
  std::fill(
      flux_register.coarse_flux.begin(), flux_register.coarse_flux.end(), 0.0);
  std::fill(flux_register.fine_flux_sum.begin(),
            flux_register.fine_flux_sum.end(),
            0.0);
  std::fill(flux_register.fine_slot_contributed.begin(),
            flux_register.fine_slot_contributed.end(),
            0U);
}

void amr_flux_register_add_coarse(AmrFluxRegister& flux_register,
                                  const int face,
                                  const double* q_flux) {
  validate_face(flux_register, face);
  const std::size_t offset = flux_offset(flux_register, face);
  for (int quantity = 0; quantity < flux_register.n_quantities; ++quantity) {
    flux_register.coarse_flux[
        offset + static_cast<std::size_t>(quantity)] += q_flux[quantity];
  }
}

void amr_flux_register_add_fine(AmrFluxRegister& flux_register,
                                const int face,
                                const int sub,
                                const double* q_flux) {
  validate_face(flux_register, face);
  if (sub < 0 || sub >= flux_register.refine_ratio) {
    throw std::out_of_range("AMR flux-register fine slot is out of range");
  }

  const std::size_t slot = fine_slot_index(flux_register, face, sub);
  if (flux_register.fine_slot_contributed[slot] != 0U) {
    throw std::logic_error(
        "AMR flux-register fine slot was contributed more than once");
  }

  const std::size_t offset = flux_offset(flux_register, face);
  for (int quantity = 0; quantity < flux_register.n_quantities; ++quantity) {
    flux_register.fine_flux_sum[
        offset + static_cast<std::size_t>(quantity)] += q_flux[quantity];
  }
  flux_register.fine_slot_contributed[slot] = 1U;
}

void amr_flux_register_reflux(const AmrFluxRegister& flux_register,
                              double* corrections) {
  const std::size_t n_flux_values = flux_register.coarse_flux.size();
  for (std::size_t index = 0; index < n_flux_values; ++index) {
    corrections[index] =
        flux_register.fine_flux_sum[index] -
        flux_register.coarse_flux[index];
  }
}

}  // namespace tenryu::hydro
