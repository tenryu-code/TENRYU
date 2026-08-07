#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "materials/eos_device_table.cuh"

namespace tenryu::coupling {

struct DriverRecloseContext {
  ~DriverRecloseContext();
  DriverRecloseContext() = default;
  DriverRecloseContext(const DriverRecloseContext&) = delete;
  DriverRecloseContext& operator=(const DriverRecloseContext&) = delete;

  std::uint8_t* d_cell_is_void = nullptr;
  std::size_t cell_is_void_capacity = 0;
  std::vector<std::uint8_t> cached_cell_is_void;
};

void launch_tabular_eos_reclose(
    DriverRecloseContext& context,
    materials::DeviceEOSTableView electron_table,
    materials::DeviceEOSTableView ion_table,
    const std::vector<std::uint8_t>& cell_is_void,
    std::size_t n_cells,
    const double* rho,
    double* ee,
    double* ei,
    double* Te,
    double* Ti,
    double* Pe,
    double* Pi,
    double* cv_e,
    double* cv_i,
    double te_floor,
    double ti_floor);

}  // namespace tenryu::coupling
