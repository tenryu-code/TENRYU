#include "hydro/boundary.hpp"

#include <string>

#include <cuda_runtime.h>

#include "core/error.hpp"

namespace tenryu::hydro {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__device__ inline void apply_boundary_1d_kernel_body(
    const int idx,
    double* __restrict__ x_r,
    double* __restrict__ v_r,
    const int n_nodes,
    const int apply_outer,
    const double r_min,
    const double r_max) {
  if (idx == 0 && n_nodes > 0) {
    v_r[0] = 0.0;
    x_r[0] = r_min;
  }

  if (idx == 1 && n_nodes > 1 && apply_outer != 0) {
    const int j_outer = n_nodes - 1;
    v_r[j_outer] = 0.0;
    x_r[j_outer] = r_max;
  }
}

__global__ void apply_boundary_1d_kernel(double* __restrict__ x_r,
                                         double* __restrict__ v_r,
                                         const int n_nodes,
                                         const int apply_outer,
                                         const double r_min,
                                         const double r_max) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 2) {
    return;
  }

  apply_boundary_1d_kernel_body(idx, x_r, v_r, n_nodes, apply_outer, r_min,
                                r_max);
}

}  // namespace

HydroBoundaryType parse_boundary_type_1d(const core::Config& cfg) {
  const std::string& bc = cfg.numerics.hydro.boundary_1d;
  if (bc == "free") {
    return HydroBoundaryType::FREE;
  }
  if (bc == "fixed") {
    return HydroBoundaryType::FIXED;
  }
  if (bc == "reflect") {
    return HydroBoundaryType::REFLECT;
  }
  if (bc == "pressure") {
    return HydroBoundaryType::PRESSURE;
  }
  TENRYU_ASSERT(false, "Unsupported 1D hydro boundary type");
  return HydroBoundaryType::FREE;
}

double pressure_ghost_1d(const core::State& state,
                         const core::Config&,
                         const HydroBoundaryType bc_type,
                         const double eval_time) {
  switch (bc_type) {
    case HydroBoundaryType::FREE:
      // Free boundary: external thermal pressure is zero. Ghost Q is added
      // from the boundary cell by the acceleration kernel.
      return 0.0;
    case HydroBoundaryType::FIXED:
    case HydroBoundaryType::REFLECT:
      // Boundary-cell Pe, Pi, and Q are added by the acceleration kernel.
      return 0.0;
    case HydroBoundaryType::PRESSURE:
      TENRYU_ASSERT(state.pressure_drive_1d.has_value(),
                    "pressure boundary requires initialized pressure_drive_1d table");
      // Pressure boundary overrides only external thermal pressure. Ghost Q
      // is added from the boundary cell by the acceleration kernel.
      return state.pressure_drive_1d->eval(eval_time);
  }

  TENRYU_ASSERT(false, "Unhandled HydroBoundaryType in pressure_ghost_1d");
  return 0.0;
}

void apply_boundary_1d(core::State& state, const core::Config& cfg) {
  TENRYU_ASSERT(state.x_r.size() > 1, "1D hydro requires at least two nodes");
  TENRYU_ASSERT(state.v_r.size() == state.x_r.size(),
                "Node velocity/position size mismatch");

  const std::size_t j_outer = state.x_r.size() - 1;
  (void)j_outer;
  const HydroBoundaryType bc_type = parse_boundary_type_1d(cfg);
  const int apply_outer =
      (bc_type == HydroBoundaryType::FIXED || bc_type == HydroBoundaryType::REFLECT)
          ? 1
          : 0;

  apply_boundary_1d_kernel<<<1, 2>>>(state.x_r.data(), state.v_r.data(),
                                      static_cast<int>(state.x_r.size()),
                                      apply_outer, cfg.mesh.r_min, cfg.mesh.r_max);
  cuda_check(cudaGetLastError(), "apply_boundary_1d kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "apply_boundary_1d kernel execution failed");
}

}  // namespace tenryu::hydro
