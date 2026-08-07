#pragma once

#include "core/namelist/frozen_table.hpp"

namespace tenryu::laser {

double waveform_power_at_time(const core::namelist::FrozenTable1D& table, double t);

}  // namespace tenryu::laser

