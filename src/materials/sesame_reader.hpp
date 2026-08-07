#pragma once

#include <optional>
#include <string>
#include <vector>

namespace tenryu::materials {

struct SesameEOSTableRaw {
  std::vector<double> rho_grid;
  std::vector<double> T_grid_eV;
  std::vector<double> pressure_dyne_per_cm2;
  std::vector<double> energy_erg_per_g;
};

struct SesameOpacityTableRaw {
  std::vector<double> rho_grid;
  std::vector<double> T_grid_eV;
  std::vector<double> values;
};

struct SesameData {
  std::optional<SesameEOSTableRaw> table_301_total;
  std::optional<SesameEOSTableRaw> table_304_electron;
  std::optional<SesameOpacityTableRaw> table_502_rosseland;
  std::optional<SesameOpacityTableRaw> table_505_planck;
};

// Parse SESAME ASCII (auto-detect standard 3-int or xSESAME 4-int+flag header,
// 80-char fixed-width payload) and extract selected tables for one material id.
SesameData read_xsesame(const std::string& filename, int material_id);

}  // namespace tenryu::materials
