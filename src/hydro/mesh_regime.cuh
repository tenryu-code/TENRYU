#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "core/state.hpp"
#include "hydro/mesh_regime.hpp"

namespace tenryu::hydro {

struct AxisMarginPredicateResult {
  bool admissible = true;
  double min_margin = 0.0;
  int failed_condition = 0;
};

class MeshRegimeDeviceCache {
 public:
  MeshRegimeDeviceCache() = default;
  ~MeshRegimeDeviceCache();

  MeshRegimeDeviceCache(const MeshRegimeDeviceCache&) = delete;
  MeshRegimeDeviceCache& operator=(const MeshRegimeDeviceCache&) = delete;

  MeshRegimeDeviceCache(MeshRegimeDeviceCache&& other) noexcept;
  MeshRegimeDeviceCache& operator=(MeshRegimeDeviceCache&& other) noexcept;

  void ensure_size(std::size_t count);
  void release();
  void invalidate() noexcept { valid_ = false; }
  void mark_valid() noexcept { valid_ = true; }

  [[nodiscard]] bool valid() const noexcept { return valid_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] CellRegime* current() noexcept { return current_; }
  [[nodiscard]] const CellRegime* current() const noexcept { return current_; }
  [[nodiscard]] std::uint8_t* previous_primary() noexcept {
    return previous_primary_;
  }
  [[nodiscard]] const std::uint8_t* previous_primary() const noexcept {
    return previous_primary_;
  }

 private:
  CellRegime* current_ = nullptr;
  std::uint8_t* previous_primary_ = nullptr;
  std::size_t size_ = 0;
  bool valid_ = false;
};

__host__ __device__ AxisMarginPredicateResult evaluate_axis_margin_predicate_quad(
    const double* r_current,
    const double* z_current,
    const double* r_trial,
    const double* z_trial,
    double floor_eps);

__global__ void classify_mesh_regimes_kernel(
    CellRegime* __restrict__ out,
    std::uint8_t* __restrict__ previous_primary,
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const double* __restrict__ cell_area,
    const double* __restrict__ cell_Svec_r,
    const double* __restrict__ cell_Svec_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::uint8_t* __restrict__ cell_is_void,
    const std::int8_t* __restrict__ hydro_active,
    int nr,
    int nz,
    double dt,
    int axis_guard_band_cells,
    bool has_physical_rz_axis);

void classify_mesh_regimes(const core::State& state,
                           double dt,
                           int axis_guard_band_cells,
                           MeshRegimeDeviceCache& cache,
                           const std::int8_t* d_hydro_active = nullptr,
                           bool has_physical_rz_axis = true);

std::vector<CellRegime> copy_mesh_regimes_to_host(
    const MeshRegimeDeviceCache& cache);

}  // namespace tenryu::hydro
