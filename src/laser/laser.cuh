#pragma once

#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/field.hpp"
#include "core/state.hpp"
#include "laser/laser_mesh.cuh"
#include "parallel/partition.hpp"
#include "parallel/reduction.hpp"

namespace tenryu::laser {

struct RayOutputData {
  struct Ray2DRecord {
    double R0 = 0.0;
    double Z0 = 0.0;
    double vR0 = 0.0;
    double vZ0 = 0.0;
    double power0 = 0.0;
    double power = 0.0;
  };

  struct Ray3DRecord {
    double x0 = 0.0;
    double y0 = 0.0;
    double z0 = 0.0;
    double vx0 = 0.0;
    double vy0 = 0.0;
    double vz0 = 0.0;
    double power0 = 0.0;
  };

  std::vector<Ray2DRecord> rays_2d;
  std::vector<Ray3DRecord> rays_3d;
  int beam_id = -1;
};

void laser_step(core::State& state,
                LaserMesh& lmesh,
                const core::Config::LaserConfig& laser,
                double dt,
                double t,
                const parallel::PartitionInfo& part,
                cudaStream_t stream = nullptr,
                double rho_floor = 1.0e-10,
                double Te_floor = 1.0e-3,
                bool* used_skip = nullptr,
                std::vector<RayOutputData>* ray_output = nullptr,
                const parallel::Reduction* reduction = nullptr,
                bool collect_trajectory = false,
                bool verbose = false,
                bool collect_density_diag = true,
                const std::string& output_dir = {});

void invalidate_global_skip_cache();

}  // namespace tenryu::laser
