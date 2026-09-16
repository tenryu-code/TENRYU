#pragma once

#include "materials/cold_equilibrium_table.hpp"

namespace tenryu::materials {

class DeviceColdEquilibriumTable {
 public:
  DeviceColdEquilibriumTable() = default;
  ~DeviceColdEquilibriumTable();
  DeviceColdEquilibriumTable(DeviceColdEquilibriumTable&&) noexcept;
  DeviceColdEquilibriumTable& operator=(DeviceColdEquilibriumTable&&) noexcept;
  DeviceColdEquilibriumTable(const DeviceColdEquilibriumTable&) = delete;
  DeviceColdEquilibriumTable& operator=(const DeviceColdEquilibriumTable&) = delete;

  void upload(const ColdEquilibriumTable& cpu_table);
  [[nodiscard]] ColdEquilibriumView view() const;
  [[nodiscard]] bool empty() const { return view_.n_knots == 0; }

 private:
  ColdEquilibriumView view_{};
  void free_all();
};

}  // namespace tenryu::materials
