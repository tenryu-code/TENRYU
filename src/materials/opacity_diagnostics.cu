#include "materials/opacity_diagnostics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "core/constants.hpp"
#include "core/error.hpp"
#include "materials/ionmix_reader.hpp"
#include "materials/tmat_reader.hpp"

namespace tenryu::materials {
namespace {

[[nodiscard]] int find_energy_group(const std::vector<double>& bounds_eV,
                                    const double energy_eV) {
  if (bounds_eV.size() < 2U) {
    return 0;
  }
  if (energy_eV <= bounds_eV.front()) {
    return 0;
  }
  for (std::size_t g = 0; g + 1U < bounds_eV.size(); ++g) {
    if (energy_eV >= bounds_eV[g] && energy_eV < bounds_eV[g + 1U]) {
      return static_cast<int>(g);
    }
  }
  return static_cast<int>(bounds_eV.size() - 2U);
}

[[nodiscard]] IonmixOpacityData load_opacity_for_material(
    const core::Config::MaterialsConfig::MatDef& mat) {
  if (mat.opacity_model == "tmat") {
    const TmatFile tmat = load_tmat(mat.opacity_file);
    TENRYU_ASSERT(tmat.opacity.has_value(),
                  "hard-X-ray opacity diagnostic requires TMAT /opacity payload");
    return tmat_to_ionmix_opacity(*tmat.opacity);
  }
  if (mat.opacity_model == "table_nlte" || mat.opacity_model == "ionmix") {
    return load_ionmix_opacity(mat.opacity_file);
  }
  IonmixOpacityData out;
  out.ngroups = 1;
  out.ndens = 1;
  out.ntemp = 1;
  out.temps_eV = {1.0};
  out.numdens_cm3 = {1.0};
  out.bounds_eV = {1.0, 1.0e7};
  out.log_temps = {0.0};
  out.log_numdens = {0.0};
  out.kappa_R = {std::max(mat.kappa_a_constant, 0.0)};
  out.kappa_PA = {std::max(mat.kappa_a_constant, 0.0)};
  out.kappa_PE = out.kappa_PA;
  out.has_PE = true;
  out.is_lte = true;
  return out;
}

}  // namespace

void log_hard_xray_opacity_diagnostic(const tenryu::core::Config& cfg) {
  if (!cfg.radiation.diagnose_hard_xray_opacity) {
    return;
  }
  static bool emitted = false;
  if (emitted) {
    return;
  }
  emitted = true;

  const auto& mats = cfg.materials.materials;
  auto mat_it = std::find_if(mats.begin(), mats.end(), [](const auto& mat) {
    return !mat.is_void && mat.name == "CD";
  });
  if (mat_it == mats.end()) {
    mat_it = std::find_if(mats.begin(), mats.end(), [](const auto& mat) {
      return !mat.is_void;
    });
  }
  if (mat_it == mats.end()) {
    core::log_warning("[hard_xray_opacity_diag] skipped: no non-void material");
    return;
  }

  const auto& mat = *mat_it;
  if (mat.opacity_file.empty() && mat.opacity_model != "constant") {
    core::log_warning("[hard_xray_opacity_diag] skipped: material=" + mat.name +
                      " has no opacity file");
    return;
  }

  const IonmixOpacityData table = load_opacity_for_material(mat);
  const double A = std::max(mat.A, 1.0e-12);
  constexpr std::array<double, 4> rhos = {0.2, 1.0, 3.0, 6.0};
  constexpr std::array<double, 4> temps = {10.0, 30.0, 80.0, 150.0};
  constexpr std::array<double, 6> energies_keV = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  for (const double rho : rhos) {
    const double ni_cm3 = rho / (A * core::constants::proton_mass);
    for (const double energy_keV : energies_keV) {
      const int g = find_energy_group(table.bounds_eV, energy_keV * 1000.0);
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(6);
      oss << "[hard_xray_opacity_diag]"
          << " material=" << mat.name
          << " rho=" << rho
          << " E=" << energy_keV << "keV"
          << " kappa_PA_cm2_g(T_eV=10,30,80,150)=";
      for (std::size_t i = 0; i < temps.size(); ++i) {
        const double kappa =
            table.interpolate_kappa(table.kappa_PA, g, ni_cm3, temps[i]);
        if (i == 0U) {
          oss << "[";
        } else {
          oss << ",";
        }
        oss << kappa;
      }
      oss << "] nist_polystyrene_cm2_g(E_keV=2,3,4,5)=[278,83,34,17]";
      core::log_info(oss.str());
    }
  }
}

}  // namespace tenryu::materials
