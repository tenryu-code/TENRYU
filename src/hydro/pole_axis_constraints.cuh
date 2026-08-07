#pragma once

#include <cstdint>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

namespace tenryu::hydro::pole_axis {

constexpr std::uint8_t kNodeCenterFlag = 1U << 2;
constexpr std::uint8_t kNodePoleAxisFlag = 1U << 3;
constexpr int kVelocityBcFree = 0;
constexpr int kVelocityBcReflect = 1;
constexpr int kVelocityBcFixed = 2;
constexpr int kVelocityBcStateSupply = 3;

__host__ __device__ inline bool has_flag(const std::uint8_t* __restrict__ node_flags,
                                         const int n,
                                         const std::uint8_t flag) {
  return node_flags != nullptr && (node_flags[n] & flag) != 0U;
}

__host__ __device__ inline bool is_reflect_like_z_mode(const int mode) {
  return mode == kVelocityBcReflect || mode == kVelocityBcStateSupply;
}

__host__ __device__ inline bool is_rectangular_z_reflect_mode(
    const int mode,
    const bool state_supply_zeros_rectangular_z) {
  return mode == kVelocityBcReflect ||
         (state_supply_zeros_rectangular_z && mode == kVelocityBcStateSupply);
}

__host__ __device__ inline void apply_2d_boundary_vector_constraints(
    double& value_r,
    double& value_z,
    const std::uint8_t* __restrict__ node_flags,
    const int n,
    const int i,
    const int j,
    const int nr,
    const int nz,
    const int r_outer_bc_mode,
    const int z_bottom_bc_mode,
    const int z_top_bc_mode,
    const bool state_supply_zeros_rectangular_z) {
  const bool pole_axis = has_flag(node_flags, n, kNodePoleAxisFlag);

  if (i == 0) {
    value_r = 0.0;
  }

  if (i == nr) {
    if (r_outer_bc_mode == kVelocityBcFixed) {
      value_r = 0.0;
      value_z = 0.0;
    } else if (r_outer_bc_mode == kVelocityBcReflect) {
      value_r = 0.0;
    }
  }

  if (j == 0) {
    if (z_bottom_bc_mode == kVelocityBcFixed) {
      value_r = 0.0;
      value_z = 0.0;
    } else if (pole_axis && is_reflect_like_z_mode(z_bottom_bc_mode)) {
      value_r = 0.0;
    } else if (is_rectangular_z_reflect_mode(z_bottom_bc_mode,
                                             state_supply_zeros_rectangular_z)) {
      value_z = 0.0;
    }
  }

  if (j == nz) {
    if (z_top_bc_mode == kVelocityBcFixed) {
      value_r = 0.0;
      value_z = 0.0;
    } else if (pole_axis && is_reflect_like_z_mode(z_top_bc_mode)) {
      value_r = 0.0;
    } else if (is_rectangular_z_reflect_mode(z_top_bc_mode,
                                             state_supply_zeros_rectangular_z)) {
      value_z = 0.0;
    }
  }

  if (i == 0) {
    value_r = 0.0;
  }

  if (has_flag(node_flags, n, kNodeCenterFlag)) {
    value_r = 0.0;
    value_z = 0.0;
  }
}

}  // namespace tenryu::hydro::pole_axis
