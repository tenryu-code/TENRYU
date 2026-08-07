#pragma once

#include <optional>
#include <string>
#include <vector>

#include "materials/eos_table.hpp"
#include "materials/ionmix_reader.hpp"

namespace tenryu::materials {

struct TmatEOSData {
  int ndens = 0;
  int ntemp = 0;
  std::vector<double> rho_grid;   // [nD] cm^-3 (ion number density)
  std::vector<double> T_grid_eV;  // [nT] eV
  std::vector<double> zbar;       // [nD * nT] idx = d*nT+t
  std::vector<double> P_i;        // [nD * nT] dyne/cm^2
  std::vector<double> P_e;        // [nD * nT] dyne/cm^2
  std::vector<double> e_i;        // [nD * nT] erg/g
  std::vector<double> e_e;        // [nD * nT] erg/g
  std::optional<std::vector<double>> cv_i;  // [nD * nT] erg/(g*eV)
  std::optional<std::vector<double>> cv_e;  // [nD * nT] erg/(g*eV)
};

struct TmatOpacityData {
  int ngroups = 0;
  int ndens = 0;
  int ntemp = 0;
  std::vector<double> rho_grid;   // [nD] cm^-3 (ion number density)
  std::vector<double> T_grid_eV;  // [nT] eV
  std::vector<double> bounds_eV;  // [nG+1] eV
  std::vector<double> kappa_R;    // [nG * nD * nT] cm^2/g
  std::vector<double> kappa_PA;   // [nG * nD * nT] cm^2/g
  std::vector<double> kappa_PE;   // [nG * nD * nT] cm^2/g
  bool is_lte = false;
};

struct TmatIonizationData {
  int ndens = 0;
  int ntemp = 0;
  int n_stages = 0;
  std::vector<double> rho_grid;    // [nD] ni in cm^-3 (naming mirrors TmatEOSData)
  std::vector<double> T_grid_eV;   // [nT]
  std::vector<int> stage_element;  // [nS]
  std::vector<int> stage_charge;   // [nS]
  std::vector<double> fractions;   // [nS * nD * nT], idx = (s*nD + d)*nT + t
};

struct ZeffRatioTable {
  int ndens = 0;
  int ntemp = 0;
  std::vector<double> ni_grid;     // [nD] cm^-3
  std::vector<double> T_grid_eV;   // [nT]
  std::vector<double> ratio;       // [nD * nT], idx = d*nT + t; Zeff/<Z>, clamped [1,10]
  std::vector<double> ratio4;      // [nD * nT], <Z^4>/zbar^4, clamped [1,100]
};

struct TmatMaterialInfo {
  std::string name;
  std::vector<int> Z;
  std::vector<double> A_amu;
  std::vector<double> mass_fraction;
  std::vector<double> number_fraction;
  double Abar_ion_amu = 0.0;
};

struct TmatFile {
  std::string schema_version;
  std::string units_system;
  std::vector<std::string> required_features;
  TmatMaterialInfo material;
  std::optional<TmatEOSData> eos;
  std::optional<TmatOpacityData> opacity;
  std::optional<TmatIonizationData> ionization;
};

enum class TmatLoadMode {
  Strict,
  Compatible,
};

TmatFile load_tmat(const std::string& filepath,
                   TmatLoadMode mode = TmatLoadMode::Compatible);
EOSTablePair tmat_eos_to_table_pair(const TmatEOSData& eos);
EOSTableTriplet tmat_eos_to_table_triplet(const TmatEOSData& eos, double A_amu);
IonmixZbarTable tmat_eos_to_zbar_table(const TmatEOSData& eos, double A_amu);
IonmixOpacityData tmat_to_ionmix_opacity(const TmatOpacityData& opacity,
                                         bool skip_lte_repair = false);
// Number-fraction weights come from material.number_fraction if present; otherwise
// they are derived as x_e = (w_e/A_e) / sum(w/A).
ZeffRatioTable tmat_ionization_to_zeff_ratio(const TmatIonizationData& ion,
                                             const TmatMaterialInfo& material);
ZeffRatioTable resample_zeff_ratio_log_uniform(const ZeffRatioTable& src,
                                               int nd_out,
                                               int nt_out);

}  // namespace tenryu::materials
