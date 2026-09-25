#pragma once

#include <string>
#include <utility>
#include <vector>

#include "core/config.hpp"

namespace tenryu::core::namelist {

// Species composition of one material: (nuclear charge, number fraction),
// charges strictly ascending, fractions summing to 1. Empty = unknown.
using SpeciesComposition = std::vector<std::pair<double, double>>;

struct LaserZeffResolution {
  std::string model;               // resolved Laser.ib.zeff_model
  std::vector<double> species_z;   // filled when the species list is derived
  std::vector<double> species_x;
  std::string info;                // log_info text (empty = none)
  std::string warning;             // log_warning text (empty = none)
};

// Resolution of Laser.ib.zeff_model="auto" (NUMERICS §5.4.5(a)): TMAT
// ionization-stage table > explicit Laser.ib.species > one multi-species
// composition shared by every non-void material > "off". Materials whose
// composition is unknown and whose mean charge is not an integer count as
// mixtures; a mixture that ends at "off" is reported in `warning` (inverse
// bremsstrahlung then uses the mean Zbar and underestimates the collision
// charge by Zeff/Zbar). The IB extensions exist only in the 1D_SPH ray trace:
// without them (`ib_extensions_supported` false) only the table rule applies
// and everything else resolves to "off" without a warning.
LaserZeffResolution resolve_laser_zeff_auto(
    bool have_ionization_table,
    const std::vector<double>& species_z_given,
    const std::vector<Config::MaterialsConfig::MatDef>& materials,
    const std::vector<SpeciesComposition>& compositions,
    bool laser_enabled,
    bool ib_extensions_supported);

}  // namespace tenryu::core::namelist
