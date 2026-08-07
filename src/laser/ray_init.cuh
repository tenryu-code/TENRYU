#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "core/error.hpp"
#include "laser/beams.cuh"
#include "laser/laser_mesh.cuh"

namespace tenryu::laser {

struct Ray2D {
  double R = 0.0;
  double Z = 0.0;
  double vR = 0.0;
  double vZ = 1.0;
  double I = 0.0;
  double I0 = 0.0;
  int beam_id = 0;
  int alive = 1;
};

struct Ray3D {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double vx = 0.0;
  double vy = 0.0;
  double vz = 1.0;
  double I = 0.0;
  double I0 = 0.0;
  int beam_id = 0;
  int alive = 1;
};

struct RayArray1D {
  double* R0 = nullptr;
  double* Z0 = nullptr;
  double* vR0 = nullptr;
  double* vZ0 = nullptr;
  double* power = nullptr;
  double* power0 = nullptr;
  int n_rays = 0;
  int n_rays_capacity = 0;
  bool pooled = false;  // slab from core::device_scratch -- release() must not cudaFree

  RayArray1D() = default;
  ~RayArray1D();
  RayArray1D(const RayArray1D&) = delete;
  RayArray1D& operator=(const RayArray1D&) = delete;
  RayArray1D(RayArray1D&& other) noexcept;
  RayArray1D& operator=(RayArray1D&& other) noexcept;

  void release();
  [[nodiscard]] bool empty() const;
  void allocate(int n);
  void allocate_pooled(int n);
  void copy_from_host(const std::vector<Ray2D>& rays, cudaStream_t stream = nullptr);
};

RayArray1D initialize_rays_1d(const Beam& beam,
                              const LaserMesh& lmesh,
                              int rays_per_beam,
                              double beam_power,
                              cudaStream_t stream = nullptr);

struct RayArray2D {
  double* x0 = nullptr;
  double* y0 = nullptr;
  double* z0 = nullptr;
  double* vx0 = nullptr;
  double* vy0 = nullptr;
  double* vz0 = nullptr;
  double* power = nullptr;
  double* power0 = nullptr;
  int n_rays = 0;
  int n_rays_capacity = 0;

  RayArray2D() = default;
  ~RayArray2D();
  RayArray2D(const RayArray2D&) = delete;
  RayArray2D& operator=(const RayArray2D&) = delete;
  RayArray2D(RayArray2D&& other) noexcept;
  RayArray2D& operator=(RayArray2D&& other) noexcept;

  void release();
  [[nodiscard]] bool empty() const;
  void allocate(int n);
  void copy_from_host(const std::vector<Ray3D>& rays, cudaStream_t stream = nullptr);
};

struct BeamGroup {
  double theta = 0.0;
  std::vector<int> beam_indices;
  double total_power = 0.0;
};

RayArray2D initialize_rays_2d(const Beam& beam,
                              const LaserMesh& lmesh,
                              int rays_per_beam,
                              double beam_power,
                              cudaStream_t stream = nullptr);

std::vector<BeamGroup> group_beams_by_theta(const std::vector<Beam>& beams,
                                            bool split_delta_lambda = false);

}  // namespace tenryu::laser
