#include "hydro/conservation_audit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "diagnostics/energy_budget.hpp"

namespace tenryu::hydro::conservation_audit {
namespace {

bool env_flag_enabled(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr) {
    return false;
  }
  const std::string value(raw);
  return value == "1" || value == "true" || value == "TRUE" ||
         value == "yes" || value == "YES" || value == "on" ||
         value == "ON";
}

template <typename Field>
std::vector<double> copy_double_field(const Field& field) {
  std::vector<double> out;
  if (!field.empty()) {
    field.copy_to_host(out);
  }
  return out;
}

bool cell_masked(const std::vector<std::uint8_t>& mask, const int c) {
  return c >= 0 && static_cast<std::size_t>(c) < mask.size() &&
         mask[static_cast<std::size_t>(c)] != 0U;
}

bool cell_active(const core::State& state, const int c) {
  if (state.hydro_active.empty()) {
    return true;
  }
  return c >= 0 && static_cast<std::size_t>(c) < state.hydro_active.size() &&
         state.hydro_active[static_cast<std::size_t>(c)] != 0;
}

struct Totals {
  double raw_mass = 0.0;
  double active_mass = 0.0;
  double owner_mass = 0.0;
  double raw_internal = 0.0;
  double raw_energy = 0.0;
  double central_mass = 0.0;
  double pole_mass = 0.0;
  int active_cells = 0;
  int inactive_cells = 0;
  int central_members = 0;
  int pole_members = 0;
  int zero_owner_cells = 0;
  int double_owner_cells = 0;
  int first_bad_owner_cell = -1;
  int first_bad_owner_count = 0;
};

double sum_cells(const std::vector<int>& cells,
                 const std::vector<double>& values) {
  long double sum = 0.0L;
  for (const int c : cells) {
    if (c >= 0 && static_cast<std::size_t>(c) < values.size()) {
      sum += static_cast<long double>(values[static_cast<std::size_t>(c)]);
    }
  }
  return static_cast<double>(sum);
}

double cell_internal(const std::vector<double>& mass,
                     const std::vector<double>& ee,
                     const std::vector<double>& ei,
                     const int c) {
  if (c < 0 || static_cast<std::size_t>(c) >= mass.size() ||
      static_cast<std::size_t>(c) >= ee.size()) {
    return 0.0;
  }
  const std::size_t idx = static_cast<std::size_t>(c);
  const double ion = idx < ei.size() ? ei[idx] : 0.0;
  return mass[idx] * (ee[idx] + ion);
}

Totals compute_totals(const core::State& state) {
  Totals out{};
  const int n_cells = static_cast<int>(state.mass.size());
  if (n_cells <= 0) {
    return out;
  }

  const std::vector<double> mass = copy_double_field(state.mass);
  const std::vector<double> ee = copy_double_field(state.ee);
  const std::vector<double> ei = copy_double_field(state.ei);
  if (mass.size() != static_cast<std::size_t>(n_cells) ||
      ee.size() != static_cast<std::size_t>(n_cells)) {
    return out;
  }

  for (int c = 0; c < n_cells; ++c) {
    const std::size_t idx = static_cast<std::size_t>(c);
    out.raw_mass += mass[idx];
    out.raw_internal += cell_internal(mass, ee, ei, c);
  }

  if (state.mesh.dim == 2) {
    const diagnostics::EnergyTotals energy =
        diagnostics::compute_energy_totals_2d(state);
    out.raw_energy = energy.E_int_e + energy.E_int_i + energy.E_kin;
  } else if (state.mesh.dim == 1) {
    const diagnostics::EnergyTotals energy =
        diagnostics::compute_energy_totals_1d(state);
    out.raw_energy = energy.E_int_e + energy.E_int_i + energy.E_kin;
  } else {
    out.raw_energy = out.raw_internal;
  }

  std::vector<int> owner_count(static_cast<std::size_t>(n_cells), 0);
  std::vector<std::uint8_t> inactive(static_cast<std::size_t>(n_cells), 0U);
  const auto mark_inactive = [&](const std::vector<std::uint8_t>& mask) {
    if (mask.size() != static_cast<std::size_t>(n_cells)) {
      return;
    }
    for (int c = 0; c < n_cells; ++c) {
      if (mask[static_cast<std::size_t>(c)] != 0U) {
        inactive[static_cast<std::size_t>(c)] = 1U;
      }
    }
  };
  mark_inactive(state.central_pseudo_core.inactive_member_mask);
  mark_inactive(state.pole_angular_derefine.inactive_member_mask);

  for (int c = 0; c < n_cells; ++c) {
    if (cell_active(state, c)) {
      out.active_mass += mass[static_cast<std::size_t>(c)];
      ++out.active_cells;
      if (inactive[static_cast<std::size_t>(c)] == 0U) {
        ++owner_count[static_cast<std::size_t>(c)];
        out.owner_mass += mass[static_cast<std::size_t>(c)];
      }
    }
    if (inactive[static_cast<std::size_t>(c)] != 0U) {
      ++out.inactive_cells;
    }
  }

  const auto& central = state.central_pseudo_core;
  if (central.valid && !central.member_cells.empty()) {
    const double M = central.M_c > 0.0
                         ? central.M_c
                         : sum_cells(central.member_cells, mass);
    out.central_mass += M;
    out.owner_mass += M;
    out.central_members += static_cast<int>(central.member_cells.size());
    for (const int c : central.member_cells) {
      if (c >= 0 && c < n_cells) {
        ++owner_count[static_cast<std::size_t>(c)];
      }
    }
  }

  const auto& pole = state.pole_angular_derefine;
  if (pole.valid) {
    for (const auto& macro : pole.macros) {
      if (macro.member_cells.empty()) {
        continue;
      }
      const double M = macro.M_c > 0.0
                           ? macro.M_c
                           : sum_cells(macro.member_cells, mass);
      out.pole_mass += M;
      out.owner_mass += M;
      out.pole_members += static_cast<int>(macro.member_cells.size());
      for (const int c : macro.member_cells) {
        if (c >= 0 && c < n_cells) {
          ++owner_count[static_cast<std::size_t>(c)];
        }
      }
    }
  }

  for (int c = 0; c < n_cells; ++c) {
    const bool participates = cell_active(state, c) ||
                              cell_masked(inactive, c) ||
                              owner_count[static_cast<std::size_t>(c)] > 0;
    if (!participates) {
      continue;
    }
    const int count = owner_count[static_cast<std::size_t>(c)];
    if (count == 0) {
      ++out.zero_owner_cells;
    } else if (count > 1) {
      ++out.double_owner_cells;
    }
    if (count != 1 && out.first_bad_owner_cell < 0) {
      out.first_bad_owner_cell = c;
      out.first_bad_owner_count = count;
    }
  }

  return out;
}

struct Ledger {
  bool initialized = false;
  bool has_previous = false;
  double initial_raw_mass = 0.0;
  double initial_owner_mass = 0.0;
  double initial_raw_energy = 0.0;
  double previous_raw_mass = 0.0;
  double previous_owner_mass = 0.0;
  double previous_raw_energy = 0.0;
  double boundary_work_cumulative = 0.0;
  std::string previous_stage;
};

Ledger& ledger() {
  static Ledger state;
  return state;
}

double rel_delta(const double delta, const double ref) {
  return delta / std::max(std::abs(ref), 1.0e-300);
}

}  // namespace

bool enabled() {
  return env_flag_enabled("TENRYU_I1B_CONS_AUDIT");
}

void emit_stage(const core::State& state,
                const char* stage,
                const double boundary_work_delta) {
  if (!enabled()) {
    return;
  }
  Totals totals = compute_totals(state);
  Ledger& l = ledger();
  if (!l.initialized) {
    l.initialized = true;
    l.initial_raw_mass = totals.raw_mass;
    l.initial_owner_mass = totals.owner_mass;
    l.initial_raw_energy = totals.raw_energy;
  }
  if (std::isfinite(boundary_work_delta)) {
    l.boundary_work_cumulative += boundary_work_delta;
  }

  const double dM_raw_initial = totals.raw_mass - l.initial_raw_mass;
  const double dM_owner_initial = totals.owner_mass - l.initial_owner_mass;
  const double dE_raw_initial =
      totals.raw_energy - l.initial_raw_energy - l.boundary_work_cumulative;
  const double dM_raw_prev =
      l.has_previous ? totals.raw_mass - l.previous_raw_mass : 0.0;
  const double dM_owner_prev =
      l.has_previous ? totals.owner_mass - l.previous_owner_mass : 0.0;
  const double dE_raw_prev =
      l.has_previous ? totals.raw_energy - l.previous_raw_energy : 0.0;

  std::fprintf(stderr,
               "[i1b_cons_audit] step=%d t=%.17e stage=%s "
               "raw_mass=%.17e owner_mass=%.17e active_mass=%.17e "
               "dM_raw_initial=%.17e dM_raw_initial_rel=%.17e "
               "dM_owner_initial=%.17e dM_owner_initial_rel=%.17e "
               "dM_raw_prev=%.17e dM_owner_prev=%.17e prev_stage=%s "
               "raw_E=%.17e raw_internal=%.17e dE_raw_prev=%.17e "
               "dE_raw_minus_init_W=%.17e dE_raw_minus_init_W_rel=%.17e "
               "W_boundary_cumulative=%.17e W_boundary_delta=%.17e "
               "central_M=%.17e pole_M=%.17e active_cells=%d inactive_cells=%d "
               "central_members=%d pole_members=%d zero_owner_cells=%d "
               "double_owner_cells=%d first_bad_owner_cell=%d "
               "first_bad_owner_count=%d\n",
               state.step,
               state.t,
               stage != nullptr ? stage : "unspecified",
               totals.raw_mass,
               totals.owner_mass,
               totals.active_mass,
               dM_raw_initial,
               rel_delta(dM_raw_initial, l.initial_raw_mass),
               dM_owner_initial,
               rel_delta(dM_owner_initial, l.initial_owner_mass),
               dM_raw_prev,
               dM_owner_prev,
               l.has_previous ? l.previous_stage.c_str() : "none",
               totals.raw_energy,
               totals.raw_internal,
               dE_raw_prev,
               dE_raw_initial,
               rel_delta(dE_raw_initial, l.initial_raw_energy),
               l.boundary_work_cumulative,
               boundary_work_delta,
               totals.central_mass,
               totals.pole_mass,
               totals.active_cells,
               totals.inactive_cells,
               totals.central_members,
               totals.pole_members,
               totals.zero_owner_cells,
               totals.double_owner_cells,
               totals.first_bad_owner_cell,
               totals.first_bad_owner_count);

  l.has_previous = true;
  l.previous_raw_mass = totals.raw_mass;
  l.previous_owner_mass = totals.owner_mass;
  l.previous_raw_energy = totals.raw_energy;
  l.previous_stage = stage != nullptr ? stage : "unspecified";
}

}  // namespace tenryu::hydro::conservation_audit
