#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "core/error.hpp"

namespace tenryu::materials {

struct IonmixOpacityData {
  int ntemp = 0;
  int ndens = 0;
  int ngroups = 0;
  std::vector<double> temps_eV;     // [ntemp]
  std::vector<double> numdens_cm3;  // [ndens], ion number density
  std::vector<double> bounds_eV;    // [ngroups+1]
  std::vector<double> log_temps;
  std::vector<double> log_numdens;

  // 3D opacity tables: [ngroups * ndens * ntemp], T fastest.
  std::vector<double> kappa_R;   // Rosseland [cm^2/g]
  std::vector<double> kappa_PA;  // Planck absorption [cm^2/g]
  std::vector<double> kappa_PE;  // Planck emission [cm^2/g]
  bool has_PE = false;
  bool is_lte = true;

  [[nodiscard]] std::size_t flat_index(const int g,
                                       const int d,
                                       const int t) const {
    TENRYU_ASSERT(g >= 0 && g < ngroups, "flat_index group out of range");
    TENRYU_ASSERT(d >= 0 && d < ndens, "flat_index density out of range");
    TENRYU_ASSERT(t >= 0 && t < ntemp, "flat_index temperature out of range");
    return static_cast<std::size_t>(g) * static_cast<std::size_t>(ndens) *
               static_cast<std::size_t>(ntemp) +
           static_cast<std::size_t>(d) * static_cast<std::size_t>(ntemp) +
           static_cast<std::size_t>(t);
  }

  // Bilinear interpolation in log(n_i)-log(T) space for opacity.
  [[nodiscard]] double interpolate_kappa(const std::vector<double>& kappa_table,
                                         int group,
                                         double ni_cm3,
                                         double T_eV) const;

  // Get sigma = rho * kappa.
  [[nodiscard]] double sigma_PA(int group, double rho, double T_eV, double A) const;
  [[nodiscard]] double sigma_PE(int group, double rho, double T_eV, double A) const;
  [[nodiscard]] double sigma_R(int group, double rho, double T_eV, double A) const;
};

struct IonmixEOSData {
  int ntemp = 0;
  int ndens = 0;
  std::vector<double> temps_eV;     // [ntemp]
  std::vector<double> numdens_cm3;  // [ndens], ion number density
  // 2D tables: [ndens * ntemp], T fastest (IONMIX native layout).
  std::vector<double> zbar;     // block 1, dimensionless
  std::vector<double> P_i_cgs;  // block 3, dyne/cm^2
  std::vector<double> P_e_cgs;  // block 4, dyne/cm^2
  std::vector<double> e_i_cgs;  // block 7, erg/g
  std::vector<double> e_e_cgs;  // block 8, erg/g

  [[nodiscard]] std::size_t flat_index(const int d, const int t) const {
    return static_cast<std::size_t>(d) * static_cast<std::size_t>(ntemp) +
           static_cast<std::size_t>(t);
  }
};

struct IonmixZbarTable {
  std::vector<double> rho_grid;      // [g/cm^3]
  std::vector<double> T_grid_eV;     // [eV]
  std::vector<double> log_rho_grid;  // [log(g/cm^3)]
  std::vector<double> log_T_grid;    // [log(eV)]
  // EOSTable layout: idx = j_T * n_rho + i_rho.
  std::vector<double> zbar_table;  // [n_rho * n_T]

  [[nodiscard]] std::size_t n_rho() const noexcept {
    return rho_grid.size();
  }

  [[nodiscard]] std::size_t n_T() const noexcept {
    return T_grid_eV.size();
  }

  [[nodiscard]] std::size_t flat_index(const std::size_t i_rho,
                                       const std::size_t j_T) const noexcept {
    return j_T * n_rho() + i_rho;
  }

  [[nodiscard]] double interpolate(double rho, double T_eV) const;
};

IonmixEOSData load_ionmix_binary_eos(const std::string& filename);
// Caller must verify returned data.ngroups matches config radiation.groups.
IonmixOpacityData load_ionmix_opacity(const std::string& filename);
IonmixZbarTable ionmix_eos_to_zbar_table(const IonmixEOSData& eos, double A);

}  // namespace tenryu::materials
