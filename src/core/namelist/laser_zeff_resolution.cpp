#include "core/namelist/laser_zeff_resolution.hpp"

#include <cmath>
#include <cstddef>
#include <sstream>

namespace tenryu::core::namelist {

LaserZeffResolution resolve_laser_zeff_auto(
    const bool have_ionization_table,
    const std::vector<double>& species_z_given,
    const std::vector<Config::MaterialsConfig::MatDef>& materials,
    const std::vector<SpeciesComposition>& compositions,
    const bool laser_enabled,
    const bool ib_extensions_supported) {
  LaserZeffResolution out;
  bool all_share_composition = true;
  bool any_mixture = false;
  const SpeciesComposition* shared = nullptr;
  for (std::size_t m = 0; m < materials.size(); ++m) {
    const auto& mat = materials[m];
    if (mat.is_void) {
      continue;
    }
    static const SpeciesComposition kUnknown;
    const SpeciesComposition& composition =
        (m < compositions.size()) ? compositions[m] : kUnknown;
    if (composition.size() > 1 ||
        (composition.empty() && std::abs(mat.Z - std::round(mat.Z)) > 1.0e-12)) {
      any_mixture = true;
    }
    if (composition.empty()) {
      all_share_composition = false;
      continue;
    }
    if (shared == nullptr) {
      shared = &composition;
      continue;
    }
    bool same = shared->size() == composition.size();
    for (std::size_t e = 0; same && e < composition.size(); ++e) {
      same = (*shared)[e].first == composition[e].first &&
             std::abs((*shared)[e].second - composition[e].second) <= 1.0e-9;
    }
    if (!same) {
      all_share_composition = false;
    }
  }

  if (have_ionization_table) {
    out.model = "table";
    out.info = "Laser.ib.zeff_model=auto -> table (TMAT ionization fractions detected)";
    return out;
  }
  // Outside the 1D_SPH ray trace the laser step refuses active IB extensions,
  // so "auto" stays "off" there, as it did before the species rules existed.
  if (!ib_extensions_supported) {
    out.model = "off";
    return out;
  }
  if (!species_z_given.empty()) {
    out.model = "sequential_strip";
    out.info = "Laser.ib.zeff_model=auto -> sequential_strip (Laser.ib.species given)";
    return out;
  }
  // A single-species composition has Zeff = Zbar exactly: "off" keeps the
  // historic arithmetic. Laser.ib.species accepts at most four entries.
  if (all_share_composition && shared != nullptr && shared->size() > 1 &&
      shared->size() <= 4) {
    out.model = "sequential_strip";
    std::ostringstream oss;
    oss << "Laser.ib.zeff_model=auto -> sequential_strip with the TMAT composition";
    for (const auto& entry : *shared) {
      out.species_z.push_back(entry.first);
      out.species_x.push_back(entry.second);
      oss << " [Z=" << entry.first << ", x=" << entry.second << "]";
    }
    out.info = oss.str();
    return out;
  }
  out.model = "off";
  if (laser_enabled && any_mixture) {
    out.warning =
        "Laser.ib.zeff_model=auto -> off: the non-void materials do not share one known "
        "multi-species composition, so inverse bremsstrahlung uses the mean Zbar and "
        "underestimates the collision charge of mixtures (Zeff/Zbar up to 1.5 for fully "
        "ionized CD). Set Laser.ib.species=[[Z, x], ...] to include the mixed-ion "
        "correction.";
  }
  return out;
}

}  // namespace tenryu::core::namelist
