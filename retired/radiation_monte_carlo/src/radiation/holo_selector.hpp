#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "radiation/holo_geometry.hpp"

namespace tenryu::radiation {

class PlanckTable;

struct HoloSelectorConfig {
  bool enabled = false;
  double coupling_tau = 5.0;
  int guard_cells = 3;
  int blend_cells = 3;
  int min_lo_cells = 20;
  double tau_on = 0.0;
  double tau_off = 0.0;
  int min_dwell_steps = 0;
  double temperature_floor = 1.0e-12;
};

struct HoloSelectorInputs {
  HoloGeometryDimension dimension = HoloGeometryDimension::Spherical1D;
  int n_cells = 0;
  int n_groups = 1;
  int step = 0;
  const std::vector<double>* node_r = nullptr;       // [n_cells + 1]
  const std::vector<double>* mass = nullptr;         // [n_cells]
  const std::vector<double>* Te = nullptr;           // [n_cells], eV
  const std::vector<double>* sigma_R = nullptr;      // [n_cells * n_groups]
  const PlanckTable* planck = nullptr;               // Planck group weights
  const std::vector<std::uint8_t>* cell_is_void = nullptr; // [n_cells]
};

struct HoloSelectorStateView {
  std::vector<std::uint8_t>& core_mask;
  std::vector<std::uint8_t>& patch_mask;
  std::vector<std::uint8_t>& prev_core_mask;
  std::vector<std::int32_t>& hold_count;
  std::vector<std::int32_t>& dwell_count;
  std::vector<double>& tau_R;
  std::vector<double>& reduced_flux;
  std::vector<double>& mass_q;
  std::vector<double>& lo_weight;
  bool& valid;
};

struct HoloSelectorDiagnostics {
  bool active = false;
  std::int64_t n_core_cells = 0;
  std::int64_t n_patch_cells = 0;
  std::int64_t n_blend_cells = 0;
  std::int64_t n_entered = 0;
  std::int64_t n_exited = 0;
  std::int64_t n_hard_exited = 0;
  std::int64_t n_island_rejected = 0;
  double tau_R_min = 0.0;
  double tau_R_max = 0.0;
  double reduced_flux_max = 0.0;
};

[[nodiscard]] double temperature_weighted_rosseland_sigma_R(
    const PlanckTable& planck,
    const std::vector<double>& sigma_R,
    std::size_t base,
    int n_groups,
    double temperature_eV,
    double temperature_floor_eV);

HoloSelectorDiagnostics update_holo_core_mask(const HoloSelectorConfig& cfg,
                                              const HoloSelectorInputs& in,
                                              HoloSelectorStateView state);

}  // namespace tenryu::radiation
