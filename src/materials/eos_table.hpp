#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace tenryu::materials {

struct IonmixEOSData;
struct ColdEquilibriumTable;

struct EOSTable {
  std::vector<double> rho_grid;
  std::vector<double> T_grid_eV;
  std::vector<double> P_table;
  std::vector<double> e_table;
  std::vector<double> cv_table;

  std::vector<double> log_rho_grid;
  std::vector<double> log_T_grid;

  // Cold-equilibrium branch attached to an electron table (non-owning; the
  // owner is Config::MaterialsConfig::MatDef::cold_table). When set, pressure /
  // energy / cv / temperature_from_energy return the corrected quantities of
  // materials/cold_equilibrium.hpp; nullptr keeps the historic arithmetic.
  const ColdEquilibriumTable* cold = nullptr;

  [[nodiscard]] std::size_t n_rho() const noexcept {
    return rho_grid.size();
  }

  [[nodiscard]] std::size_t n_T() const noexcept {
    return T_grid_eV.size();
  }

  [[nodiscard]] std::size_t flat_index(std::size_t i_rho,
                                       std::size_t j_T) const noexcept {
    return j_T * n_rho() + i_rho;
  }

  [[nodiscard]] bool empty() const noexcept {
    return rho_grid.empty() || T_grid_eV.empty();
  }

  void finalize();

  [[nodiscard]] double pressure(double rho, double T_eV) const;
  [[nodiscard]] double energy(double rho, double T_eV) const;
  [[nodiscard]] double cv(double rho, double T_eV) const;
  [[nodiscard]] double temperature_from_energy(double rho, double e) const;
  [[nodiscard]] double sound_speed(double rho, double P) const;
};

using EOSTablePair = std::pair<EOSTable, EOSTable>;

struct EOSTableTriplet {
  EOSTable ion;
  EOSTable electron;
  EOSTable total;
};

// SESAME (rho, T) table from a raw 301/304 payload. A T = 0 row (cold curve)
// is kept as `cold_curve_rows` synthetic rows T_k = T_1 / 2^k that hold the
// linear-in-T interpolation between the cold curve and the first positive
// isotherm; 0 drops the T <= 0 rows (historic behaviour). NUMERICS §1 (b).
struct SesameEOSTableRaw;
EOSTable sesame_table_from_raw(const SesameEOSTableRaw& raw, int cold_curve_rows);
EOSTablePair load_sesame(const std::string& filename,
                         int mat_id,
                         double zbar_hint = 0.0,
                         int cold_curve_rows = 12);
// Ion table on the total (301) grid: ion = total - electron with the electron
// (304) table resampled onto the 301 nodes by the same bilinear-log
// interpolant used at runtime. 301/304 grids may differ (e.g. Polystyrene
// 73x41 vs 63x33) — node-wise subtraction across arrays of different shapes is
// undefined behavior and is never done. Identical grids reproduce the plain
// node-wise difference bitwise. Negative ion-pressure/energy nodes are
// permitted (energy references differ between 301 and 304) and are counted to
// a one-shot warning.
EOSTable build_sesame_ion_table(const EOSTable& total, const EOSTable& electron);
EOSTablePair ionmix_eos_to_table_pair(const IonmixEOSData& eos, double A);
EOSTableTriplet ionmix_eos_to_table_triplet(const IonmixEOSData& eos, double A);
EOSTableTriplet build_power_law_te_tables(double A,
                                          double ideal_gas_gamma,
                                          double f_erg_g,
                                          double beta,
                                          double mu_rho,
                                          double gamma_p,
                                          double step_D_erg_g_eV = 0.0,
                                          double step_Tc_eV = 0.0,
                                          double step_w_eV = 0.0);
EOSTablePair load_ionmix(const std::string& filename);

}  // namespace tenryu::materials
