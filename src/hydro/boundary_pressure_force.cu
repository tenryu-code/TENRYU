#include <cstdlib>
#include "hydro/boundary_pressure_force.cuh"

namespace tenryu::hydro::detail {
namespace {

__device__ inline double pressure_atomic_add_double(double* address,
                                                    const double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(
                        val + __longlong_as_double(static_cast<long long>(assumed)))));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
#endif
}

template <int RZ_SCHEME>
__global__ void add_r_outer_boundary_pressure_forces_kernel(
    double* __restrict__ force_r,
    double* __restrict__ force_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int nr,
    const int nz,
    const double p_ext,
    const bool rz_exact_endpoint) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nz) {
    return;
  }

  const int stride = nz + 1;
  const int n0 = nr * stride + j;
  const int n1 = nr * stride + (j + 1);
  if constexpr (RZ_SCHEME == 1) {
    const RzBoundaryPressureEndpointAreaVectors areas =
        r_outer_boundary_planar_pressure_endpoint_area_vectors(
            x_r, x_z, nr, nz, j);
    pressure_atomic_add_double(force_r + n0, -p_ext * areas.node0.r);
    pressure_atomic_add_double(force_r + n1, -p_ext * areas.node1.r);
    pressure_atomic_add_double(force_z + n0, -p_ext * areas.node0.z);
    pressure_atomic_add_double(force_z + n1, -p_ext * areas.node1.z);
    return;
  }
  if (rz_exact_endpoint) {
    const RzBoundaryPressureEndpointAreaVectors areas =
        r_outer_boundary_pressure_endpoint_area_vectors(x_r, x_z, nr, nz, j);
    pressure_atomic_add_double(force_r + n0, -p_ext * areas.node0.r);
    pressure_atomic_add_double(force_r + n1, -p_ext * areas.node1.r);
    pressure_atomic_add_double(force_z + n0, -p_ext * areas.node0.z);
    pressure_atomic_add_double(force_z + n1, -p_ext * areas.node1.z);
    return;
  }

  const RzBoundaryPressureFaceAreaVector area =
      r_outer_boundary_pressure_area_vector(x_r, x_z, nr, nz, j);

  pressure_atomic_add_double(force_r + n0, -p_ext * area.r);
  pressure_atomic_add_double(force_r + n1, -p_ext * area.r);
  pressure_atomic_add_double(force_z + n0, -p_ext * area.z);
  pressure_atomic_add_double(force_z + n1, -p_ext * area.z);
}

// Mirror-ghost closure — the ghost cell carries the interior cell's full
// corner-force pressure (Pe+Pi+Qvisc), closing the arc node contour exactly;
// without it the open contour's steady ambient-pressure kick drives the
// boundary-layer secular instability (measured 2026-07-17; NO_PIN
// discrimination confirmed the missing-reaction mechanism).
template <int RZ_SCHEME>
__global__ void add_r_outer_boundary_mirror_forces_kernel(
    double* __restrict__ force_r,
    double* __restrict__ force_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ cell_pq,
    const int nr,
    const int nz,
    const bool rz_exact_endpoint) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nz) {
    return;
  }

  const double p_ghost = cell_pq[(nr - 1) * nz + j];
  const int stride = nz + 1;
  const int n0 = nr * stride + j;
  const int n1 = nr * stride + (j + 1);
  if constexpr (RZ_SCHEME == 1) {
    const RzBoundaryPressureEndpointAreaVectors areas =
        r_outer_boundary_planar_pressure_endpoint_area_vectors(
            x_r, x_z, nr, nz, j);
    pressure_atomic_add_double(force_r + n0, -p_ghost * areas.node0.r);
    pressure_atomic_add_double(force_r + n1, -p_ghost * areas.node1.r);
    pressure_atomic_add_double(force_z + n0, -p_ghost * areas.node0.z);
    pressure_atomic_add_double(force_z + n1, -p_ghost * areas.node1.z);
    return;
  }
  if (rz_exact_endpoint) {
    const RzBoundaryPressureEndpointAreaVectors areas =
        r_outer_boundary_pressure_endpoint_area_vectors(x_r, x_z, nr, nz, j);
    pressure_atomic_add_double(force_r + n0, -p_ghost * areas.node0.r);
    pressure_atomic_add_double(force_r + n1, -p_ghost * areas.node1.r);
    pressure_atomic_add_double(force_z + n0, -p_ghost * areas.node0.z);
    pressure_atomic_add_double(force_z + n1, -p_ghost * areas.node1.z);
    return;
  }

  const RzBoundaryPressureFaceAreaVector area =
      r_outer_boundary_pressure_area_vector(x_r, x_z, nr, nz, j);

  pressure_atomic_add_double(force_r + n0, -p_ghost * area.r);
  pressure_atomic_add_double(force_r + n1, -p_ghost * area.r);
  pressure_atomic_add_double(force_z + n0, -p_ghost * area.z);
  pressure_atomic_add_double(force_z + n1, -p_ghost * area.z);
}

template <int RZ_SCHEME>
__global__ void add_multiblock_polar_shell_pressure_forces_kernel(
    double* __restrict__ force_r,
    double* __restrict__ force_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ node_mass,
    const int polar_shell_node_begin,
    const int nr_shell,
    const int nz_polar,
    const double p_ext,
    const bool rz_exact_endpoint,
    const bool mass_proportional_split,
    const int split_mode) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nz_polar) {
    return;
  }

  const int stride = nz_polar + 1;
  const int n0_local = nr_shell * stride + j;
  const int n1_local = nr_shell * stride + (j + 1);
  const int n0 = polar_shell_node_begin + n0_local;
  const int n1 = polar_shell_node_begin + n1_local;
  const double* const x_shell_r = x_r + polar_shell_node_begin;
  const double* const x_shell_z = x_z + polar_shell_node_begin;
  if constexpr (RZ_SCHEME == 1) {
    // The I1-B environment split modes are scheme-0-only; the planar
    // convention gives each endpoint P * 0.5 * L * n_hat directly.
    const RzBoundaryPressureEndpointAreaVectors areas =
        r_outer_boundary_planar_pressure_endpoint_area_vectors(
            x_shell_r, x_shell_z, nr_shell, nz_polar, j);
    pressure_atomic_add_double(force_r + n0, -p_ext * areas.node0.r);
    pressure_atomic_add_double(force_r + n1, -p_ext * areas.node1.r);
    pressure_atomic_add_double(force_z + n0, -p_ext * areas.node0.z);
    pressure_atomic_add_double(force_z + n1, -p_ext * areas.node1.z);
    return;
  }
  if (rz_exact_endpoint) {
    const RzBoundaryPressureEndpointAreaVectors areas =
        r_outer_boundary_pressure_endpoint_area_vectors(
            x_shell_r, x_shell_z, nr_shell, nz_polar, j);
    if (mass_proportional_split && node_mass != nullptr &&
        split_mode == 2) {
      // MODE 2 (mass): per-edge mass-proportional split. Under-drives the
      // pole (measured a/mean=0.46 at t=0) yet gives the SMOOTHEST drive
      // dynamics measured (mean eps_rad 0.25 vs 0.43 variational) — the
      // slightly lagging pole is dynamically stabilizing.
      const double tot_r = areas.node0.r + areas.node1.r;
      const double tot_z = areas.node0.z + areas.node1.z;
      const double m0 = node_mass[n0];
      const double m1 = node_mass[n1];
      const double msum = m0 + m1;
      const double w0 = msum > 0.0 ? m0 / msum : 0.5;
      const double w1 = 1.0 - w0;
      pressure_atomic_add_double(force_r + n0, -p_ext * tot_r * w0);
      pressure_atomic_add_double(force_r + n1, -p_ext * tot_r * w1);
      pressure_atomic_add_double(force_z + n0, -p_ext * tot_z * w0);
      pressure_atomic_add_double(force_z + n1, -p_ext * tot_z * w1);
      return;
    }
    if (mass_proportional_split && split_mode == 1) {
      // Acceleration-consistent POLE lumping (I1-B-W, Benson 2.4.3 remedy
      // class). The variational FE-hat split gives the axis-endpoint node
      // 1/3 of the edge total, which against the corner-mass inertia makes
      // the pole accelerate 4/3x the interior under uniform pressure
      // (measured; the deterministic seed of the pole impedance mode). The
      // r-weighted HALF-SEGMENT attribution assigns the axis endpoint 1/4
      // of the edge total — restoring angle-uniform acceleration
      // ((1/4)/(1/3)*4/3 = 1) while every edge total (and hence every
      // integral identity) is unchanged. A mass-proportional split was
      // measured to overshoot the other way (pole 0.46x). Only edges with
      // an on-axis endpoint are re-split; interior edges keep the
      // variational weights (their acceleration is already uniform).
      const double r_node0 = x_shell_r[n0_local];
      const double r_node1 = x_shell_r[n1_local];
      const double s_edge =
          fmax(fabs(r_node0), fabs(r_node1));
      const bool pole0 = fabs(r_node0) <= 1.0e-12 * fmax(s_edge, 1.0e-300);
      const bool pole1 = fabs(r_node1) <= 1.0e-12 * fmax(s_edge, 1.0e-300);
      if (pole0 != pole1) {
        const double tot_r = areas.node0.r + areas.node1.r;
        const double tot_z = areas.node0.z + areas.node1.z;
        const double w0 = pole0 ? 0.25 : 0.75;
        const double w1 = 1.0 - w0;
        pressure_atomic_add_double(force_r + n0, -p_ext * tot_r * w0);
        pressure_atomic_add_double(force_r + n1, -p_ext * tot_r * w1);
        pressure_atomic_add_double(force_z + n0, -p_ext * tot_z * w0);
        pressure_atomic_add_double(force_z + n1, -p_ext * tot_z * w1);
        return;
      }
    }
    pressure_atomic_add_double(force_r + n0, -p_ext * areas.node0.r);
    pressure_atomic_add_double(force_r + n1, -p_ext * areas.node1.r);
    pressure_atomic_add_double(force_z + n0, -p_ext * areas.node0.z);
    pressure_atomic_add_double(force_z + n1, -p_ext * areas.node1.z);
    return;
  }

  const RzBoundaryPressureFaceAreaVector area =
      r_outer_boundary_pressure_area_vector(
          x_shell_r, x_shell_z, nr_shell, nz_polar, j);

  pressure_atomic_add_double(force_r + n0, -p_ext * area.r);
  pressure_atomic_add_double(force_r + n1, -p_ext * area.r);
  pressure_atomic_add_double(force_z + n0, -p_ext * area.z);
  pressure_atomic_add_double(force_z + n1, -p_ext * area.z);
}

}  // namespace

void launch_r_outer_boundary_pressure_forces(double* force_r,
                                             double* force_z,
                                             const double* x_r,
                                             const double* x_z,
                                             const int nr,
                                             const int nz,
                                             const double p_ext,
                                             const bool rz_exact_endpoint,
                                             const int rz_scheme) {
  if (nr <= 0 || nz <= 0) {
    return;
  }
  const int blocks_j = (nz + 255) / 256;
  switch (rz_scheme) {
    case 0:
      add_r_outer_boundary_pressure_forces_kernel<0><<<blocks_j, 256>>>(
          force_r, force_z, x_r, x_z, nr, nz, p_ext, rz_exact_endpoint);
      break;
    case 1:
      add_r_outer_boundary_pressure_forces_kernel<1><<<blocks_j, 256>>>(
          force_r, force_z, x_r, x_z, nr, nz, p_ext, rz_exact_endpoint);
      break;
    default:
      std::abort();
  }
}

void launch_r_outer_boundary_mirror_forces(double* force_r,
                                           double* force_z,
                                           const double* x_r,
                                           const double* x_z,
                                           const double* cell_pq,
                                           const int nr,
                                           const int nz,
                                           const bool rz_exact_endpoint,
                                           const int rz_scheme) {
  if (nr <= 0 || nz <= 0) {
    return;
  }
  const int blocks_j = (nz + 255) / 256;
  switch (rz_scheme) {
    case 0:
      add_r_outer_boundary_mirror_forces_kernel<0><<<blocks_j, 256>>>(
          force_r, force_z, x_r, x_z, cell_pq, nr, nz, rz_exact_endpoint);
      break;
    case 1:
      add_r_outer_boundary_mirror_forces_kernel<1><<<blocks_j, 256>>>(
          force_r, force_z, x_r, x_z, cell_pq, nr, nz, rz_exact_endpoint);
      break;
    default:
      std::abort();
  }
}

void launch_multiblock_polar_shell_pressure_forces(
    double* force_r,
    double* force_z,
    const double* x_r,
    const double* x_z,
    const int polar_shell_node_begin,
    const int nr_shell,
    const int nz_polar,
    const double p_ext,
    const bool rz_exact_endpoint,
    const double* node_mass,
    const int rz_scheme) {
  if (polar_shell_node_begin < 0 || nr_shell <= 0 || nz_polar <= 0) {
    return;
  }
  // I1-B-W acceleration-consistent endpoint lumping (env, default off):
  // active only when the caller supplies nodal masses.
  // Split mode: 0/unset = legacy variational; 1 = pole quarter-split
  // (fixes the t=0 4/3 seed, dynamically over-corrects); 2 = per-edge
  // mass-proportional (best measured drive dynamics).
  static const int split_mode = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_MASS_SPLIT_DRIVE");
    if (raw == nullptr || raw[0] == '\0' || raw[0] == '0') {
      return 0;
    }
    const int v = std::atoi(raw);
    return v >= 1 && v <= 2 ? v : 1;
  }();
  const int blocks_j = (nz_polar + 255) / 256;
  switch (rz_scheme) {
    case 0:
      add_multiblock_polar_shell_pressure_forces_kernel<0><<<blocks_j, 256>>>(
          force_r, force_z, x_r, x_z, node_mass, polar_shell_node_begin,
          nr_shell, nz_polar, p_ext, rz_exact_endpoint, split_mode != 0,
          split_mode);
      break;
    case 1:
      add_multiblock_polar_shell_pressure_forces_kernel<1><<<blocks_j, 256>>>(
          force_r, force_z, x_r, x_z, node_mass, polar_shell_node_begin,
          nr_shell, nz_polar, p_ext, rz_exact_endpoint, split_mode != 0,
          split_mode);
      break;
    default:
      std::abort();
  }
}

}  // namespace tenryu::hydro::detail
