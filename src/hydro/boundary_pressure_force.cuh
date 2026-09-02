#pragma once

#include <cuda_runtime.h>

#include "hydro/pressure_drive_perturbation.cuh"

namespace tenryu::hydro::detail {

struct RzBoundaryPressureFaceAreaVector {
  double r = 0.0;
  double z = 0.0;
};

struct RzBoundaryPressureEndpointAreaVectors {
  RzBoundaryPressureFaceAreaVector node0;
  RzBoundaryPressureFaceAreaVector node1;
};

__host__ __device__ inline RzBoundaryPressureEndpointAreaVectors
r_outer_boundary_planar_pressure_endpoint_area_vectors(const double* x_r,
                                                       const double* x_z,
                                                       const int nr,
                                                       const int nz,
                                                       const int j) {
  const int stride = nz + 1;
  const int n0 = nr * stride + j;
  const int n1 = nr * stride + (j + 1);
  const int n_inner0 = (nr - 1) * stride + j;
  const int n_inner1 = (nr - 1) * stride + (j + 1);

  const double r0 = x_r[n0];
  const double z0 = x_z[n0];
  const double r1 = x_r[n1];
  const double z1 = x_z[n1];
  const double dr = r1 - r0;
  const double dz = z1 - z0;

  RzBoundaryPressureEndpointAreaVectors areas{};
  areas.node0.r = 0.5 * dz;
  areas.node0.z = -0.5 * dr;
  areas.node1 = areas.node0;

  const double face_r = 0.5 * (r0 + r1);
  const double face_z = 0.5 * (z0 + z1);
  const double cell_r = 0.25 * (x_r[n_inner0] + r0 + r1 + x_r[n_inner1]);
  const double cell_z = 0.25 * (x_z[n_inner0] + z0 + z1 + x_z[n_inner1]);
  const double outward_dot =
      areas.node0.r * (face_r - cell_r) +
      areas.node0.z * (face_z - cell_z);
  if (outward_dot < 0.0) {
    areas.node0.r = -areas.node0.r;
    areas.node0.z = -areas.node0.z;
    areas.node1 = areas.node0;
  }
  return areas;
}

__host__ __device__ inline RzBoundaryPressureFaceAreaVector
r_outer_boundary_pressure_area_vector(const double* x_r,
                                      const double* x_z,
                                      const int nr,
                                      const int nz,
                                      const int j) {
  constexpr double kPi = 3.141592653589793238462643383279502884;
  const int stride = nz + 1;
  const int n0 = nr * stride + j;
  const int n1 = nr * stride + (j + 1);
  const int n_inner0 = (nr - 1) * stride + j;
  const int n_inner1 = (nr - 1) * stride + (j + 1);

  const double r0 = x_r[n0];
  const double z0 = x_z[n0];
  const double r1 = x_r[n1];
  const double z1 = x_z[n1];

  const double rbar_face = 0.5 * (r0 + r1);
  RzBoundaryPressureFaceAreaVector area{};
  area.r = kPi * rbar_face * (z1 - z0);
  area.z = -kPi * rbar_face * (r1 - r0);

  const double face_r = 0.5 * (r0 + r1);
  const double face_z = 0.5 * (z0 + z1);
  const double cell_r = 0.25 * (x_r[n_inner0] + r0 + r1 + x_r[n_inner1]);
  const double cell_z = 0.25 * (x_z[n_inner0] + z0 + z1 + x_z[n_inner1]);
  const double outward_dot =
      area.r * (face_r - cell_r) + area.z * (face_z - cell_z);
  if (outward_dot < 0.0) {
    area.r = -area.r;
    area.z = -area.z;
  }
  return area;
}

__host__ __device__ inline RzBoundaryPressureEndpointAreaVectors
r_outer_boundary_pressure_endpoint_area_vectors(const double* x_r,
                                                const double* x_z,
                                                const int nr,
                                                const int nz,
                                                const int j) {
  constexpr double kPi = 3.141592653589793238462643383279502884;
  const int stride = nz + 1;
  const int n0 = nr * stride + j;
  const int n1 = nr * stride + (j + 1);
  const int n_inner0 = (nr - 1) * stride + j;
  const int n_inner1 = (nr - 1) * stride + (j + 1);

  const double r0 = x_r[n0];
  const double z0 = x_z[n0];
  const double r1 = x_r[n1];
  const double z1 = x_z[n1];
  const double dr = r1 - r0;
  const double dz = z1 - z0;

  RzBoundaryPressureEndpointAreaVectors areas{};
  const double coeff0 = (kPi / 3.0) * (2.0 * r0 + r1);
  const double coeff1 = (kPi / 3.0) * (r0 + 2.0 * r1);
  areas.node0.r = coeff0 * dz;
  areas.node0.z = -coeff0 * dr;
  areas.node1.r = coeff1 * dz;
  areas.node1.z = -coeff1 * dr;

  const double rbar_face = 0.5 * (r0 + r1);
  RzBoundaryPressureFaceAreaVector midpoint{};
  midpoint.r = kPi * rbar_face * dz;
  midpoint.z = -kPi * rbar_face * dr;

  const double face_r = 0.5 * (r0 + r1);
  const double face_z = 0.5 * (z0 + z1);
  const double cell_r = 0.25 * (x_r[n_inner0] + r0 + r1 + x_r[n_inner1]);
  const double cell_z = 0.25 * (x_z[n_inner0] + z0 + z1 + x_z[n_inner1]);
  const double outward_dot =
      midpoint.r * (face_r - cell_r) + midpoint.z * (face_z - cell_z);
  if (outward_dot < 0.0) {
    areas.node0.r = -areas.node0.r;
    areas.node0.z = -areas.node0.z;
    areas.node1.r = -areas.node1.r;
    areas.node1.z = -areas.node1.z;
  }
  return areas;
}

void launch_r_outer_boundary_pressure_forces(double* force_r,
                                             double* force_z,
                                             const double* x_r,
                                             const double* x_z,
                                             int nr,
                                             int nz,
                                             double p_ext,
                                             bool rz_exact_endpoint,
                                             int rz_scheme,
                                             const PressureDrivePerturbationParams& drive_pert,
                                             const bool drive_pert_enabled);

void launch_r_outer_boundary_mirror_forces(double* force_r,
                                           double* force_z,
                                           const double* x_r,
                                           const double* x_z,
                                           const double* cell_pq,
                                           int nr,
                                           int nz,
                                           bool rz_exact_endpoint,
                                           int rz_scheme);

void launch_multiblock_polar_shell_pressure_forces(double* force_r,
                                                   double* force_z,
                                                   const double* x_r,
                                                   const double* x_z,
                                                   int polar_shell_node_begin,
                                                   int nr_shell,
                                                   int nz_polar,
                                                   double p_ext,
                                                   bool rz_exact_endpoint,
                                                   const double* node_mass,
                                                   int rz_scheme,
                                                   const PressureDrivePerturbationParams& drive_pert,
                                                   const bool drive_pert_enabled);

}  // namespace tenryu::hydro::detail
