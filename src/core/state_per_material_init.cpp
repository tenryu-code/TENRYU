#include "core/state_per_material_init.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/error.hpp"

namespace tenryu::core {
namespace {

template <typename Tag>
std::vector<double> field_to_host(const Field1D<Tag>& field) {
  std::vector<double> host(field.size(), 0.0);
  if (!host.empty()) {
    field.copy_to_host(host.data());
  }
  return host;
}

}  // namespace

void populate_per_material_conserved_from_cell_mean(State& state,
                                                    const Config& cfg) {
  if (!cfg.numerics.materials.per_material_conservation_enabled) {
    return;
  }

  const std::size_t n_cells = state.mass.size();
  const std::size_t n_mat = cfg.materials.materials.size();
  const std::size_t n_cell_mat = n_cells * n_mat;

  TENRYU_ASSERT(state.ee.size() == n_cells,
                "per-material init requires ee size n_cells");
  TENRYU_ASSERT(state.ei.size() == n_cells,
                "per-material init requires ei size n_cells");
  TENRYU_ASSERT(state.volFrac.size() == n_cell_mat,
                "per-material init requires volFrac size n_cells*n_materials");

  if (state.mass_per_material.size() != n_cell_mat) {
    state.mass_per_material.reset(n_cell_mat);
  }
  if (state.Ee_per_material.size() != n_cell_mat) {
    state.Ee_per_material.reset(n_cell_mat);
  }
  if (state.Ei_per_material.size() != n_cell_mat) {
    state.Ei_per_material.reset(n_cell_mat);
  }

  if (n_cell_mat == 0) {
    if (cfg.numerics.materials.lazy_cache_te_m_enabled) {
      state.Te_per_material.reset(0);
      state.Ti_per_material.reset(0);
      state.Te_per_material_valid.clear();
      state.Ti_per_material_valid.clear();
    }
    return;
  }

  const auto mass = field_to_host(state.mass);
  const auto ee = field_to_host(state.ee);
  const auto ei = field_to_host(state.ei);
  const auto volfrac = field_to_host(state.volFrac);

  std::vector<double> mass_m(n_cell_mat, 0.0);
  std::vector<double> Ee_m(n_cell_mat, 0.0);
  std::vector<double> Ei_m(n_cell_mat, 0.0);

  for (std::size_t c = 0; c < n_cells; ++c) {
    const double cell_mass = mass[c];
    if (!(cell_mass > 0.0)) {
      continue;
    }
    for (std::size_t m = 0; m < n_mat; ++m) {
      const std::size_t idx = c * n_mat + m;
      const double m_cm = volfrac[idx] * cell_mass;
      mass_m[idx] = m_cm;
      Ee_m[idx] = m_cm * ee[c];
      Ei_m[idx] = m_cm * ei[c];
    }
  }

  state.mass_per_material.copy_from_host(mass_m.data());
  state.Ee_per_material.copy_from_host(Ee_m.data());
  state.Ei_per_material.copy_from_host(Ei_m.data());
  state.invalidate_cell_material_props();

  if (cfg.numerics.materials.lazy_cache_te_m_enabled) {
    if (state.Te_per_material.size() != n_cell_mat) {
      state.Te_per_material.reset(n_cell_mat);
    }
    if (state.Ti_per_material.size() != n_cell_mat) {
      state.Ti_per_material.reset(n_cell_mat);
    }
    state.Te_per_material_valid.assign(n_cell_mat, static_cast<std::uint8_t>(0));
    state.Ti_per_material_valid.assign(n_cell_mat, static_cast<std::uint8_t>(0));
  } else {
    state.Te_per_material.reset(0);
    state.Ti_per_material.reset(0);
    state.Te_per_material_valid.clear();
    state.Ti_per_material_valid.clear();
  }
}

}  // namespace tenryu::core
