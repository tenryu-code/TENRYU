#pragma once

#include "core/namelist/frozen_table.hpp"

namespace tenryu::laser {

double waveform_power_at_time(const core::namelist::FrozenTable1D& table, double t);

// Exact integral of the frozen waveform over [t0, t1] (the table is piecewise
// linear between its samples; outside [x_min, x_max] it is 0 when
// zero_outside, otherwise the end values — the same function eval() returns).
double waveform_integral(const core::namelist::FrozenTable1D& table, double t0, double t1);

// Time average of the waveform over [t0, t1]; the instantaneous value at t0
// when the interval is empty. A step of length dt that starts at t deposits
// waveform_average_power(table, t, t + dt) * dt, which sums exactly to the
// pulse integral for any step sequence.
double waveform_average_power(const core::namelist::FrozenTable1D& table, double t0, double t1);

}  // namespace tenryu::laser
