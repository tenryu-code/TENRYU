#include "laser/waveform_table.cuh"

namespace tenryu::laser {

double waveform_power_at_time(const core::namelist::FrozenTable1D& table,
                              const double t) {
  return table.eval(t);
}

}  // namespace tenryu::laser

