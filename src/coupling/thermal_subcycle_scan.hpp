#pragma once

namespace tenryu::coupling {

// Device scans for the thermal subcycle (NUMERICS §2.1). Each returns one
// value to the host with a single small copy, instead of copying the Te and
// rho fields to the host.

// True when a cell in [0, n) has rho > rho_min and Te <= te_threshold (the
// compressed floor hit that triggers a floor retry).
bool thermal_subcycle_floor_hit(const double* d_Te,
                                const double* d_rho,
                                int n,
                                double rho_min,
                                double te_threshold);

// Minimum over cells in [begin, end) with rho >= rho_min and
// te_guard < Te < te_guard + margin_max of (Te - te_guard) / max(Te, 1);
// returns 1.0e30 when no cell qualifies (the host loop's initial value).
double thermal_subcycle_min_margin_ratio(const double* d_Te,
                                         const double* d_rho,
                                         int begin,
                                         int end,
                                         double rho_min,
                                         double te_guard,
                                         double margin_max);

}  // namespace tenryu::coupling
