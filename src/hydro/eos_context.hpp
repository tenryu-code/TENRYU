#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "materials/eos_device_table.cuh"
#include "materials/eos_device_table.hpp"
#include "materials/eos_rho_e_device.cuh"
#include "materials/eos_rho_e_device.hpp"
#include "materials/helmholtz_jet_device.cuh"
#include "materials/helmholtz_jet_device.hpp"
#include "materials/helmholtz_spline_device.cuh"
#include "materials/helmholtz_spline_device.hpp"
#include "materials/mie_gruneisen_device.cuh"
#include "materials/mie_gruneisen_device.hpp"

namespace tenryu::core {
struct Config;
}

namespace tenryu::hydro {

/// Manages per-material device EOS backends for GPU hydro kernels.
/// Owns raw DeviceEOSTable and optional DeviceHelmholtzSpline objects via RAII.
struct HydroEOSContext {
  /// Per-material device tables (indexed by material slot in Config).
  std::vector<materials::DeviceEOSTable> ion;
  std::vector<materials::DeviceEOSTable> electron;
  std::vector<materials::DeviceEOSTable> total;
  std::vector<materials::DeviceEOSRhoETable> total_rho_e;
  std::vector<materials::DeviceHelmholtzSpline> ion_helmholtz;
  std::vector<materials::DeviceHelmholtzSpline> electron_helmholtz;
  std::vector<materials::DeviceHelmholtzSpline> total_helmholtz;
  std::vector<materials::DeviceHelmholtzJet> total_helmholtz_jet;
  std::vector<materials::DeviceMieGruneisen> mie_gruneisen;
  std::vector<std::uint8_t> hydro_backend_kind;
  std::vector<std::uint8_t> rho_e_reclosure_supported;

  /// Device arrays of DeviceEOSTableView, one per material.
  materials::DeviceEOSTableView* d_ion_views = nullptr;
  materials::DeviceEOSTableView* d_electron_views = nullptr;
  materials::DeviceEOSTableView* d_total_views = nullptr;

  /// Number of materials.
  int n_materials = 0;

  /// True if at least one material has a table EOS (n_rho > 0).
  bool any_table = false;
  bool any_helmholtz_spline = false;
  bool any_helmholtz_jet = false;

  /// Allocate and upload tables from Config. Must be called once after Config is finalized.
  void initialize(const core::Config& cfg);

  [[nodiscard]] bool material_uses_helmholtz(const int material_index) const;
  [[nodiscard]] std::uint8_t material_hydro_backend_kind(const int material_index) const;
  [[nodiscard]] bool material_supports_rho_e_reclosure(const int material_index) const;
  [[nodiscard]] materials::DeviceEOSTableView ion_view(const int material_index) const;
  [[nodiscard]] materials::DeviceEOSTableView electron_view(const int material_index) const;
  [[nodiscard]] materials::DeviceEOSTableView total_view(const int material_index) const;
  [[nodiscard]] materials::EOSRhoEDeviceView total_rho_e_view(const int material_index) const;
  [[nodiscard]] materials::HelmholtzSplineDeviceView ion_helmholtz_view(
      const int material_index) const;
  [[nodiscard]] materials::HelmholtzSplineDeviceView electron_helmholtz_view(
      const int material_index) const;
  [[nodiscard]] materials::HelmholtzSplineDeviceView total_helmholtz_view(
      const int material_index) const;
  [[nodiscard]] materials::HelmholtzJetDeviceView total_helmholtz_jet_view(
      const int material_index) const;
  [[nodiscard]] materials::MieGruneisenDeviceView mie_gruneisen_view(
      const int material_index) const;

  /// Free device view arrays. DeviceEOSTable objects free themselves via RAII.
  void destroy();

  ~HydroEOSContext();
  HydroEOSContext() = default;
  HydroEOSContext(HydroEOSContext&&) noexcept;
  HydroEOSContext& operator=(HydroEOSContext&&) noexcept;
  HydroEOSContext(const HydroEOSContext&) = delete;
  HydroEOSContext& operator=(const HydroEOSContext&) = delete;
};

}  // namespace tenryu::hydro
