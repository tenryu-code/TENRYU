#pragma once

#include <cstdint>
#include <vector>

#include "core/state.hpp"

namespace tenryu::coupling {

struct DeviceTemperatureAudit {
  double max_te = 0.0;
  bool te_has_non_finite = false;
  bool ti_has_non_finite = false;
  bool rho_has_non_finite = false;
  bool ee_has_non_finite = false;
  bool ei_has_non_finite = false;
};

struct OvershootDeviceMetrics {
  int count;
  double max_ratio;
};

DeviceTemperatureAudit compute_temperature_audit_device(const core::State& state);

// Launch the same temperature-audit reduction as the eager API, but leave the
// result in caller-owned device slots. d_flags has six entries in the same
// order as DeviceTemperatureAudit plus the finite-Te presence flag at index 5.
void compute_temperature_audit_device_to_slots(const core::State& state,
                                               double* d_max_te,
                                               int* d_flags);

// Set d_violation from a staged before/after audit pair. If d_max_te_before is
// null, max_te_before_host is used instead.
void compute_phase_safety_violation_device(
    const double* d_max_te_before,
    const int* d_before_flags,
    double max_te_before_host,
    const double* d_max_te_after,
    const int* d_after_flags,
    bool nan_fatal,
    bool overshoot_fatal_enabled,
    double te_floor,
    double overshoot_fatal,
    int* d_violation);

void compute_overshoot_metrics_device_to_slots(
    const double* d_te,
    int n_cells,
    const double* d_max_te_before,
    const int* d_before_flags,
    double boundary_temperature,
    int* d_count,
    double* d_max_ratio);

OvershootDeviceMetrics compute_overshoot_metrics_device(
    const double* d_te, int n_cells, double t_max_n, double denom);

DeviceTemperatureAudit audit_temperatures_host(const core::State& state);

// Device twin of the 2T-collapse scan: returns true when every
// finite-(Te,Ti) active cell has |Te-Ti|/max(max(Te,te_floor),1e-30)
// < 1e-10 and at least one active cell exists. Per-cell arithmetic and
// filters mirror the retired host loop exactly (sub + div only, no
// fusable sequence), and the two counters are exact integer sums, so
// the boolean is bit-equivalent for any scheduling.
bool all_active_cells_collapsed_device(const core::State& state,
                                       double te_floor);

// Device twin of update_hydro_active's temperature rule: the cells inactive in the
// State's device mirror of hydro_active whose Te >= t_start_eV (the host
// comparison: a NaN Te stays inactive). Returns their number (one small copy to
// the host); when it is nonzero, reached (n_cells) holds 1 for those cells and 0
// elsewhere, else it is left empty.
int hydro_active_t_start_reached_device(const core::State& state,
                                        double t_start_eV,
                                        std::vector<std::uint8_t>& reached);

}  // namespace tenryu::coupling
