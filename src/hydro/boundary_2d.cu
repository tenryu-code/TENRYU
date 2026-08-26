#include "hydro/boundary_2d.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <vector>

#include <cuda_runtime.h>

#include "core/axis_tolerance.hpp"
#include "core/error.hpp"
#include "core/kernel_guard.hpp"
#include "core/namelist/errors.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::hydro {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

// NOTE: Direct boundary-node BC (no ghost cells). See NUMERICS H4/H16.
//       Ghost-cell model will be implemented with MPI (M18).

__global__ void apply_axis_boundary_kernel(double* __restrict__ v_r,
                                           double* __restrict__ x_r,
                                           const int nr,
                                           const int nz,
                                           const bool has_physical_rz_axis) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > nz) {
    return;
  }
  const int n = j;
  const int stride = nz + 1;
  const int idx = 0 * stride + n;
  v_r[idx] = 0.0;
  if (has_physical_rz_axis) {
    x_r[idx] = 0.0;
  }
}

__global__ void apply_r_outer_boundary_kernel(double* __restrict__ x_r,
                                              double* __restrict__ x_z,
                                              double* __restrict__ v_r,
                                              double* __restrict__ v_z,
                                              const int nr,
                                              const int nz,
                                              const int bc_type,
                                              const double r_max,
                                              const double z_min,
                                              const double dz) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > nz) {
    return;
  }

  const int stride = nz + 1;
  const int n = nr * stride + j;
  if (bc_type == static_cast<int>(Boundary2DType::FREE)) {
    // FREE applies no constraint; the boundary node keeps its Lagrangian motion.
    return;
  }
  if (bc_type == static_cast<int>(Boundary2DType::FIXED)) {
    v_r[n] = 0.0;
    v_z[n] = 0.0;
    x_r[n] = r_max;
    x_z[n] = z_min + dz * static_cast<double>(j);
  } else if (bc_type == static_cast<int>(Boundary2DType::REFLECT)) {
    v_r[n] = 0.0;
    x_r[n] = r_max;
  }
}

__global__ void apply_z_bottom_boundary_kernel(double* __restrict__ x_r,
                                               double* __restrict__ x_z,
                                               double* __restrict__ v_r,
                                               double* __restrict__ v_z,
                                               const int nr,
                                               const int nz,
                                               const int bc_type,
                                               const double r_min,
                                               const double z_min,
                                               const double dr) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > nr) {
    return;
  }

  const int stride = nz + 1;
  const int n = i * stride;
  if (bc_type == static_cast<int>(Boundary2DType::FREE)) {
    // FREE applies no constraint; the boundary node keeps its Lagrangian motion.
    return;
  }
  if (bc_type == static_cast<int>(Boundary2DType::FIXED)) {
    v_r[n] = 0.0;
    v_z[n] = 0.0;
    x_r[n] = r_min + dr * static_cast<double>(i);
    x_z[n] = z_min;
  } else if (bc_type == static_cast<int>(Boundary2DType::REFLECT)) {
    v_z[n] = 0.0;
    x_z[n] = z_min;
  } else if (bc_type == static_cast<int>(Boundary2DType::STATE_SUPPLY)) {
    v_z[n] = 0.0;
    x_z[n] = z_min;
  }
}

__global__ void apply_z_top_boundary_kernel(double* __restrict__ x_r,
                                            double* __restrict__ x_z,
                                            double* __restrict__ v_r,
                                            double* __restrict__ v_z,
                                            const int nr,
                                            const int nz,
                                            const int bc_type,
                                            const double r_min,
                                            const double z_max,
                                            const double dr) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > nr) {
    return;
  }

  const int stride = nz + 1;
  const int n = i * stride + nz;
  if (bc_type == static_cast<int>(Boundary2DType::FREE)) {
    // FREE applies no constraint; the boundary node keeps its Lagrangian motion.
    return;
  }
  if (bc_type == static_cast<int>(Boundary2DType::FIXED)) {
    v_r[n] = 0.0;
    v_z[n] = 0.0;
    x_r[n] = r_min + dr * static_cast<double>(i);
    x_z[n] = z_max;
  } else if (bc_type == static_cast<int>(Boundary2DType::REFLECT)) {
    v_z[n] = 0.0;
    x_z[n] = z_max;
  } else if (bc_type == static_cast<int>(Boundary2DType::STATE_SUPPLY)) {
    v_z[n] = 0.0;
    x_z[n] = z_max;
  }
}

__global__ void apply_state_supply_z_bottom_node_kernel(double* __restrict__ x_z,
                                                        double* __restrict__ v_z,
                                                        const int nr,
                                                        const int nz,
                                                        const double z_min) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > nr) {
    return;
  }

  const int stride = nz + 1;
  const int n = i * stride;
  // state-supply boundary mesh node:
  //   x_z[n] is held at z_min/z_max (boundary doesn't move)
  //   v_z[n] = 0 (mesh velocity is zero; node is mesh-stationary)
  //   Material velocity for upstream advective flux is supplied via the
  //   ghost-cell state-supply value (see flux closure kernels), NOT via v_z.
  //
  // This decoupling is required for ALE-safe state-supply: an ALE rezone
  // may move *interior* nodes, but the boundary node remains anchored.
  x_z[n] = z_min;
  v_z[n] = 0.0;
}

__global__ void apply_state_supply_z_top_node_kernel(double* __restrict__ x_z,
                                                     double* __restrict__ v_z,
                                                     const int nr,
                                                     const int nz,
                                                     const double z_max) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > nr) {
    return;
  }

  const int stride = nz + 1;
  const int n = i * stride + nz;
  // state-supply boundary mesh node:
  //   x_z[n] is held at z_min/z_max (boundary doesn't move)
  //   v_z[n] = 0 (mesh velocity is zero; node is mesh-stationary)
  //   Material velocity for upstream advective flux is supplied via the
  //   ghost-cell state-supply value (see flux closure kernels), NOT via v_z.
  //
  // This decoupling is required for ALE-safe state-supply: an ALE rezone
  // may move *interior* nodes, but the boundary node remains anchored.
  x_z[n] = z_max;
  v_z[n] = 0.0;
}

__global__ void apply_polar_axis_theta_boundary_kernel(double* __restrict__ v_r,
                                                       double* __restrict__ x_r,
                                                       const int nr,
                                                       const int nz,
                                                       const int j_boundary) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > nr) {
    return;
  }
  const int stride = nz + 1;
  const int n = i * stride + j_boundary;
  v_r[n] = 0.0;
  x_r[n] = 0.0;
}

__global__ void apply_polar_cut_ray_boundary_kernel(
    double* __restrict__ v_r,
    double* __restrict__ v_z,
    double* __restrict__ x_r,
    double* __restrict__ x_z,
    const int nr,
    const int nz,
    const double theta_min) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > nr) {
    return;
  }
  const int stride = nz + 1;
  const int n = i * stride;
  const double normal_r = cos(theta_min);
  const double normal_z = -sin(theta_min);
  const double x_normal = x_r[n] * normal_r + x_z[n] * normal_z;
  const double v_normal = v_r[n] * normal_r + v_z[n] * normal_z;
  x_r[n] -= x_normal * normal_r;
  x_z[n] -= x_normal * normal_z;
  v_r[n] -= v_normal * normal_r;
  v_z[n] -= v_normal * normal_z;
}

__global__ void apply_polar_inner_core_boundary_kernel(double* __restrict__ v_r,
                                                       double* __restrict__ v_z,
                                                       const double* __restrict__ x_r,
                                                       const double* __restrict__ x_z,
                                                       const int nz) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > nz) {
    return;
  }
  const int n = j;
  const double r = x_r[n];
  const double z = x_z[n];
  const double s = hypot(r, z);
  if (!(s > 0.0)) {
    v_r[n] = 0.0;
    v_z[n] = 0.0;
    return;
  }
  const double e_r = r / s;
  const double e_z = z / s;
  const double u_s = v_r[n] * e_r + v_z[n] * e_z;
  v_r[n] -= u_s * e_r;
  v_z[n] -= u_s * e_z;
}

__global__ void apply_polar_outer_arc_reflect_kernel(
    double* __restrict__ x_r,
    double* __restrict__ x_z,
    double* __restrict__ v_r,
    double* __restrict__ v_z,
    const int nr,
    const int nz,
    const double s_max,
    const int skip_radius_pin) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > nz) {
    return;
  }
  const int stride = nz + 1;
  const int n = nr * stride + j;
  const double xr = x_r[n];
  const double xz = x_z[n];
  const double s = sqrt(xr * xr + xz * xz);
  if (!(s > 0.0)) {
    return;  // degenerate; leave to the axis/core kernels
  }
  const double inv_s = 1.0 / s;
  const double sr = xr * inv_s;
  const double sz = xz * inv_s;
  const double v_s = v_r[n] * sr + v_z[n] * sz;
  v_r[n] -= v_s * sr;
  v_z[n] -= v_s * sz;
  if (skip_radius_pin != 0) {
    return;
  }
  const double scale = s_max * inv_s;
  x_r[n] = xr * scale;
  x_z[n] = xz * scale;
}

__device__ inline void remove_spherical_normal_component(double& vector_r,
                                                        double& vector_z,
                                                        const double R,
                                                        const double Z) {
  const double s = sqrt(R * R + Z * Z);
  if (s > 0.0) {
    const double inv_s = 1.0 / s;
    const double nr = R * inv_s;
    const double nz = Z * inv_s;
    const double vr = vector_r;
    const double vz = vector_z;
    const double vn = vr * nr + vz * nz;
    vector_r = vr - vn * nr;
    vector_z = vz - vn * nz;
  }
}

__device__ inline void apply_r_outer_physical_vector_constraint(
    double& vector_r,
    double& vector_z,
    const double R,
    const double Z,
    const int r_outer_type) {
  switch (static_cast<Boundary2DType>(r_outer_type)) {
    case Boundary2DType::FIXED:
      vector_r = 0.0;
      vector_z = 0.0;
      break;
    case Boundary2DType::REFLECT:
      remove_spherical_normal_component(vector_r, vector_z, R, Z);
      break;
    case Boundary2DType::FREE:
    case Boundary2DType::PRESSURE:
      break;
    case Boundary2DType::STATE_SUPPLY:
    default:
      remove_spherical_normal_component(vector_r, vector_z, R, Z);
      break;
  }
}

__global__ void apply_multiblock_axis_reflect_velocity_kernel(
    double* __restrict__ v_r,
    double* __restrict__ v_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const std::uint8_t* __restrict__ node_flags,
    const int n_nodes,
    const int r_inner_type,
    const int r_outer_type) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes || node_flags == nullptr) {
    return;
  }

  const std::uint8_t flags = node_flags[n];
  if ((flags & mesh::NODE_CENTER) != 0U) {
    v_r[n] = 0.0;
    v_z[n] = 0.0;
    return;
  }
  if ((flags & (mesh::NODE_AXIS | mesh::NODE_POLE_AXIS)) != 0U) {
    v_r[n] = 0.0;
  }
  if ((flags & mesh::NODE_INNER_PHYSICAL_BOUNDARY) != 0U) {
    const auto inner_type = static_cast<Boundary2DType>(r_inner_type);
    if (inner_type == Boundary2DType::PINNED) {
      v_r[n] = 0.0;
      v_z[n] = 0.0;
      return;
    }
    if (inner_type == Boundary2DType::AXIS) {
      remove_spherical_normal_component(
          v_r[n], v_z[n], x_r[n], x_z[n]);
      return;
    }
  }
  if ((flags & mesh::NODE_OUTER_PHYSICAL_BOUNDARY) != 0U) {
    apply_r_outer_physical_vector_constraint(
        v_r[n], v_z[n], x_r[n], x_z[n], r_outer_type);
    return;
  }
  // NODE_BOUNDARY without axis/center/inner/outer flags does not occur in any
  // multiblock builder; internal block-seam nodes receive no velocity
  // constraint (2026-07-26 review — unreachable seam
  // tangent-projection fallthrough removed).
}

void maybe_write_multiblock_axis_boundary_diagnostic(
    const core::State& state,
    const core::Config& cfg,
    const double t,
    const char* const phase) {
  const char* const path = std::getenv("TENRYU_MULTIBLOCK_AXIS_BC_DIAG_PATH");
  if (path == nullptr || path[0] == '\0' || t != 0.0 ||
      cfg.main.name != "multiblock_5block_homothetic_symmetry_g3" ||
      cfg.mesh.topology_scheme !=
          core::TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_5BLOCK) {
    return;
  }

  const auto& topo = state.mesh.topo;
  TENRYU_ASSERT(topo.node_flags.size() == static_cast<std::size_t>(topo.n_nodes),
                "axis BC diagnostic requires node_flags");

  std::vector<double> x_r;
  std::vector<double> x_z;
  std::vector<double> v_r;
  std::vector<double> v_z;
  state.x_r.copy_to_host(x_r);
  state.x_z.copy_to_host(x_z);
  state.v_r.copy_to_host(v_r);
  state.v_z.copy_to_host(v_z);

  const double axis_tol = core::axis_tolerance(cfg.mesh.spherical_polar_s_max);
  std::ofstream out(path, std::ios::app);
  TENRYU_ASSERT(out.good(), "failed to open multiblock axis BC diagnostic");
  out << std::scientific << std::setprecision(17);
  out << "{\"phase\":\"" << phase << "\",\"t_s\":" << t
      << ",\"case\":\"" << cfg.main.name << "\",\"axis_tolerance_cm\":"
      << axis_tol << ",\"nodes\":[";
  bool first = true;
  for (int n = 0; n < topo.n_nodes; ++n) {
    const std::size_t idx = static_cast<std::size_t>(n);
    const std::uint8_t flags = topo.node_flags[idx];
    if ((flags & (mesh::NODE_AXIS | mesh::NODE_POLE_AXIS)) !=
            (mesh::NODE_AXIS | mesh::NODE_POLE_AXIS) ||
        std::fabs(x_r[idx]) > axis_tol) {
      continue;
    }
    if (!first) {
      out << ",";
    }
    first = false;
    out << "{\"node\":" << n << ",\"flags\":"
        << static_cast<unsigned int>(flags)
        << ",\"NODE_AXIS\":"
        << (((flags & mesh::NODE_AXIS) != 0U) ? "true" : "false")
        << ",\"NODE_CENTER\":"
        << (((flags & mesh::NODE_CENTER) != 0U) ? "true" : "false")
        << ",\"NODE_POLE_AXIS\":"
        << (((flags & mesh::NODE_POLE_AXIS) != 0U) ? "true" : "false")
        << ",\"NODE_BOUNDARY\":"
        << (((flags & mesh::NODE_BOUNDARY) != 0U) ? "true" : "false")
        << ",\"NODE_OUTER_PHYSICAL_BOUNDARY\":"
        << (((flags & mesh::NODE_OUTER_PHYSICAL_BOUNDARY) != 0U) ? "true"
                                                                 : "false")
        << ",\"interior_axis_v_z_free\":"
        << (((flags & (mesh::NODE_CENTER | mesh::NODE_OUTER_PHYSICAL_BOUNDARY)) == 0U)
                ? "true"
                : "false")
        << ",\"R_cm\":" << x_r[idx] << ",\"Z_cm\":" << x_z[idx]
        << ",\"v_r_cm_per_s\":" << v_r[idx]
        << ",\"v_z_cm_per_s\":" << v_z[idx] << "}";
  }
  out << "]}\n";
}

void validate_polar_edge_tags(const core::State& state,
                              const double polar_theta_min) {
  const auto& topo = state.mesh.topo;
  TENRYU_ASSERT(state.mesh.edge_tags.size() == static_cast<std::size_t>(topo.n_edges()),
                "spherical-polar boundary requires edge_tags");
  for (int i = 0; i < topo.nr; ++i) {
    if (polar_theta_min > 0.0) {
      TENRYU_ASSERT(state.mesh.edge_tags[static_cast<std::size_t>(
                        topo.radial_edge_index(i, 0))] ==
                        mesh::BoundaryKind::PolarCutFace,
                    "truncated spherical-polar boundary expected PolarCutFace edge tag");
    } else {
      TENRYU_ASSERT(state.mesh.edge_tags[static_cast<std::size_t>(
                        topo.radial_edge_index(i, 0))] ==
                        mesh::BoundaryKind::RZAxisTheta0,
                    "spherical-polar boundary expected RZAxisTheta0 edge tag");
    }
    TENRYU_ASSERT(state.mesh.edge_tags[static_cast<std::size_t>(
                      topo.radial_edge_index(i, topo.nz))] ==
                      mesh::BoundaryKind::RZAxisThetaPi,
                  "spherical-polar boundary expected RZAxisThetaPi edge tag");
  }
  for (int j = 0; j < topo.nz; ++j) {
    TENRYU_ASSERT(state.mesh.edge_tags[static_cast<std::size_t>(
                      topo.angular_edge_index(0, j))] ==
                      mesh::BoundaryKind::SphericalInnerCore,
                  "spherical-polar boundary expected SphericalInnerCore edge tag");
    TENRYU_ASSERT(state.mesh.edge_tags[static_cast<std::size_t>(
                      topo.angular_edge_index(topo.nr, j))] ==
                      mesh::BoundaryKind::SphericalOuterFree,
                  "spherical-polar boundary expected SphericalOuterFree edge tag");
  }
}

void apply_boundary_2d_spherical_polar(core::State& state, const core::Config& cfg) {
  TENRYU_ASSERT(state.mesh.dim == 2, "apply_boundary_2d requires 2D mesh");
  TENRYU_ASSERT(state.v_r.size() == state.v_z.size(),
                "2D boundary requires matching node velocities");
  TENRYU_ASSERT(state.x_r.size() == state.v_r.size(),
                "2D boundary requires matching node arrays");
  TENRYU_ASSERT(state.x_z.size() == state.v_r.size(),
                "2D boundary requires matching node arrays");

  const auto& b2d = cfg.numerics.hydro.boundary_2d;
  const auto r_outer_type = parse_boundary_2d_type(b2d.r_outer);
  // PRESSURE is constraint-free here; drive forces are applied by the pressure-drive path.
  if (r_outer_type != Boundary2DType::FREE &&
      r_outer_type != Boundary2DType::REFLECT &&
      r_outer_type != Boundary2DType::PRESSURE) {
    throw core::namelist::ConfigError(
        "spherical_polar_halfplane supports r_outer 'free', 'reflect', or 'pressure' "
        "only (got '" + b2d.r_outer + "')");
  }
  const auto z_bottom_type = parse_boundary_2d_type(b2d.z_bottom);
  if (z_bottom_type != Boundary2DType::FREE &&
      z_bottom_type != Boundary2DType::REFLECT) {
    throw core::namelist::ConfigError(
        "spherical_polar_halfplane theta boundaries are structural pole rays; "
        "use 'free' (or legacy 'reflect'); got '" + b2d.z_bottom + "'");
  }
  const auto z_top_type = parse_boundary_2d_type(b2d.z_top);
  if (z_top_type != Boundary2DType::FREE &&
      z_top_type != Boundary2DType::REFLECT) {
    throw core::namelist::ConfigError(
        "spherical_polar_halfplane theta boundaries are structural pole rays; "
        "use 'free' (or legacy 'reflect'); got '" + b2d.z_top + "'");
  }
  if (parse_boundary_2d_type(b2d.r_inner) != Boundary2DType::AXIS) {
    throw core::namelist::ConfigError(
        "spherical_polar_halfplane requires r_inner 'axis' (the center/"
        "inner treatment is structural)");
  }

  const auto& topo = state.mesh.topo;
  TENRYU_ASSERT(topo.n_nodes == static_cast<int>(state.x_r.size()),
                "2D boundary requires topo/state node-size consistency");
  validate_polar_edge_tags(state, cfg.mesh.polar_theta_min);

  const int blocks_j = (topo.nz + 1 + 255) / 256;
  const int blocks_i = (topo.nr + 1 + 255) / 256;

  if (cfg.mesh.polar_theta_min > 0.0) {
    apply_polar_cut_ray_boundary_kernel<<<blocks_i, 256>>>(
        state.v_r.data(), state.v_z.data(), state.x_r.data(),
        state.x_z.data(), topo.nr, topo.nz, cfg.mesh.polar_theta_min);
  } else {
    apply_polar_axis_theta_boundary_kernel<<<blocks_i, 256>>>(
        state.v_r.data(), state.x_r.data(), topo.nr, topo.nz, 0);
  }
  apply_polar_axis_theta_boundary_kernel<<<blocks_i, 256>>>(
      state.v_r.data(), state.x_r.data(), topo.nr, topo.nz, topo.nz);
  apply_polar_inner_core_boundary_kernel<<<blocks_j, 256>>>(
      state.v_r.data(), state.v_z.data(), state.x_r.data(), state.x_z.data(), topo.nz);

  if (r_outer_type == Boundary2DType::REFLECT) {
    // diagnostic-only knob — discriminates 'position snap-back drives the boundary-layer
    // secular instability' vs 'missing contour reaction drives it' (design ledger mesh2d
    // findings section); default (unset) is bitwise-identical.
    static const bool no_pin = []() {
      const char* v = std::getenv("TENRYU_POLAR_ARC_REFLECT_NO_PIN");
      return v != nullptr && v[0] == '1';
    }();
    apply_polar_outer_arc_reflect_kernel<<<blocks_j, 256>>>(
        state.x_r.data(), state.x_z.data(), state.v_r.data(),
        state.v_z.data(), topo.nr, topo.nz,
        cfg.mesh.spherical_polar_s_max, no_pin);
  }

  // Theta-boundary priority at inner-core corners.
  if (cfg.mesh.polar_theta_min > 0.0) {
    apply_polar_cut_ray_boundary_kernel<<<blocks_i, 256>>>(
        state.v_r.data(), state.v_z.data(), state.x_r.data(),
        state.x_z.data(), topo.nr, topo.nz, cfg.mesh.polar_theta_min);
  } else {
    apply_polar_axis_theta_boundary_kernel<<<blocks_i, 256>>>(
        state.v_r.data(), state.x_r.data(), topo.nr, topo.nz, 0);
  }
  apply_polar_axis_theta_boundary_kernel<<<blocks_i, 256>>>(
      state.v_r.data(), state.x_r.data(), topo.nr, topo.nz, topo.nz);

  cuda_check(cudaGetLastError(), "apply_boundary_2d spherical-polar kernel launch failed");
  cuda_check(core::debug_kernel_sync(),
             "apply_boundary_2d spherical-polar kernel execution failed");
}

void apply_multiblock_boundary_2d_via_node_flags(core::State& state,
                                                 const int r_inner_type,
                                                 const int r_outer_type) {
  TENRYU_ASSERT(state.mesh.dim == 2, "apply_boundary_2d requires 2D mesh");
  TENRYU_ASSERT(state.v_r.size() == state.v_z.size(),
                "2D boundary requires matching node velocities");
  TENRYU_ASSERT(state.x_r.size() == state.v_r.size(),
                "2D boundary requires matching node arrays");
  TENRYU_ASSERT(state.x_z.size() == state.v_r.size(),
                "2D boundary requires matching node arrays");

  const auto& topo = state.mesh.topo;
  TENRYU_ASSERT(topo.multiblock.has_value(),
                "multiblock boundary requires multiblock topology");
  TENRYU_ASSERT(topo.n_nodes == static_cast<int>(state.x_r.size()),
                "2D boundary requires topo/state node-size consistency");
  TENRYU_ASSERT(topo.node_flags.size() == static_cast<std::size_t>(topo.n_nodes),
                "multiblock boundary requires node_flags");

  std::uint8_t* d_node_flags = nullptr;
  const std::size_t flag_bytes =
      topo.node_flags.size() * sizeof(std::uint8_t);
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_node_flags), flag_bytes),
             "apply_boundary_2d multiblock cudaMalloc node_flags failed");
  cuda_check(cudaMemcpy(d_node_flags,
                        topo.node_flags.data(),
                        flag_bytes,
                        cudaMemcpyHostToDevice),
             "apply_boundary_2d multiblock cudaMemcpy node_flags failed");

  const int blocks = (topo.n_nodes + 255) / 256;
  apply_multiblock_axis_reflect_velocity_kernel<<<blocks, 256>>>(
      state.v_r.data(),
      state.v_z.data(),
      state.x_r.data(),
      state.x_z.data(),
      d_node_flags,
      topo.n_nodes,
      r_inner_type,
      r_outer_type);
  cuda_check(cudaGetLastError(),
             "apply_boundary_2d multiblock kernel launch failed");
  cuda_check(core::debug_kernel_sync(),
             "apply_boundary_2d multiblock kernel execution failed");
  cuda_check(cudaFree(d_node_flags),
             "apply_boundary_2d multiblock cudaFree node_flags failed");
}

}  // namespace

Boundary2DType parse_boundary_2d_type(const std::string& value) {
  if (value == "free") {
    return Boundary2DType::FREE;
  }
  if (value == "fixed") {
    return Boundary2DType::FIXED;
  }
  if (value == "reflect") {
    return Boundary2DType::REFLECT;
  }
  if (value == "pressure") {
    return Boundary2DType::PRESSURE;
  }
  if (value == "axis") {
    return Boundary2DType::AXIS;
  }
  if (value == "pinned") {
    return Boundary2DType::PINNED;
  }
  if (value == "state_supply") {
    return Boundary2DType::STATE_SUPPLY;
  }
  throw core::namelist::ConfigError("Unsupported 2D hydro boundary type: " + value);
}

void validate_boundary_2d(const core::Config& cfg) {
  const auto& b = cfg.numerics.hydro.boundary_2d;
  const auto r_inner = parse_boundary_2d_type(b.r_inner);
  const auto r_outer = parse_boundary_2d_type(b.r_outer);
  const auto z_bottom = parse_boundary_2d_type(b.z_bottom);
  const auto z_top = parse_boundary_2d_type(b.z_top);

  const bool r_min_is_axis = std::abs(cfg.mesh.r_min) <= cfg.numerics.axis_eps_cm;
  if (r_min_is_axis) {
    if (r_inner != Boundary2DType::AXIS &&
        !(r_inner == Boundary2DType::PINNED &&
          cfg.mesh.topology_scheme == core::TopologyScheme::PENTAGON_BELT_SHELL)) {
      throw core::namelist::ConfigError(
          "Numerics.hydro.boundary_2d.r_inner must be \"axis\" in 2D_RZ when Mesh.r_min == 0");
    }
  } else if (r_inner != Boundary2DType::AXIS &&
             r_inner != Boundary2DType::REFLECT &&
             r_inner != Boundary2DType::PINNED) {
    throw core::namelist::ConfigError(
        "Numerics.hydro.boundary_2d.r_inner must be \"reflect\" (preferred) or "
        "\"axis\" (legacy) or \"pinned\" in 2D_RZ when Mesh.r_min > 0");
  }
  if (r_outer == Boundary2DType::AXIS) {
    throw core::namelist::ConfigError(
        "Numerics.hydro.boundary_2d.r_outer cannot be \"axis\"");
  }
  if (z_bottom == Boundary2DType::AXIS || z_top == Boundary2DType::AXIS) {
    throw core::namelist::ConfigError(
        "Numerics.hydro.boundary_2d.z_bottom/z_top cannot be \"axis\"");
  }
  if (z_bottom == Boundary2DType::PRESSURE || z_top == Boundary2DType::PRESSURE) {
    throw core::namelist::ConfigError(
        "Numerics.hydro.boundary_2d.z_bottom/z_top='pressure' is not supported");
  }
  if (r_inner == Boundary2DType::STATE_SUPPLY || r_outer == Boundary2DType::STATE_SUPPLY) {
    throw core::namelist::ConfigError(
        "Numerics.hydro.boundary_2d state_supply is supported only on z_bottom/z_top");
  }
  if (b.has_any_state_supply() && cfg.numerics.ale.enabled) {
    static bool logged_state_supply_ale = false;
    if (!logged_state_supply_ale) {
      logged_state_supply_ale = true;
      core::log_info(
          "Numerics.hydro.boundary_2d state_supply combined with "
          "Numerics.ale.enabled=true - using ALE-safe mesh-velocity "
          "decoupling.");
    }
  }
}

void apply_boundary_2d(core::State& state, const core::Config& cfg, const double t) {
  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    const auto r_inner_type =
        parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_inner);
    const auto r_outer_type =
        parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_outer);
    maybe_write_multiblock_axis_boundary_diagnostic(state, cfg, t, "before");
    apply_multiblock_boundary_2d_via_node_flags(
        state, static_cast<int>(r_inner_type), static_cast<int>(r_outer_type));
    maybe_write_multiblock_axis_boundary_diagnostic(state, cfg, t, "after");
    return;
  }
  if (state.mesh.logical == mesh::LogicalMesh2D::SphericalPolarHalfplane) {
    apply_boundary_2d_spherical_polar(state, cfg);
    return;
  }

  validate_boundary_2d(cfg);

  TENRYU_ASSERT(state.mesh.dim == 2, "apply_boundary_2d requires 2D mesh");
  TENRYU_ASSERT(state.v_r.size() == state.v_z.size(),
                "2D boundary requires matching node velocities");
  TENRYU_ASSERT(state.x_r.size() == state.v_r.size(),
                "2D boundary requires matching node arrays");
  TENRYU_ASSERT(state.x_z.size() == state.v_r.size(),
                "2D boundary requires matching node arrays");

  const auto& topo = state.mesh.topo;
  TENRYU_ASSERT(topo.n_nodes == static_cast<int>(state.x_r.size()),
                "2D boundary requires topo/state node-size consistency");

  const auto r_outer_type = parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_outer);
  const auto z_bottom_type = parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_bottom);
  const auto z_top_type = parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_top);
  if (r_outer_type == Boundary2DType::FREE) {
    core::log_debug("Hydro2D r_outer FREE boundary: no constraint applied");
  }
  if (z_bottom_type == Boundary2DType::FREE) {
    core::log_debug("Hydro2D z_bottom FREE boundary: no constraint applied");
  }
  if (z_top_type == Boundary2DType::FREE) {
    core::log_debug("Hydro2D z_top FREE boundary: no constraint applied");
  }
  const double dr =
      (cfg.mesh.r_max - cfg.mesh.r_min) / static_cast<double>(topo.nr);
  const double dz =
      (cfg.mesh.z_max - cfg.mesh.z_min) / static_cast<double>(topo.nz);

  const int blocks_j = (topo.nz + 1 + 255) / 256;
  const int blocks_i = (topo.nr + 1 + 255) / 256;

  apply_axis_boundary_kernel<<<blocks_j, 256>>>(
      state.v_r.data(), state.x_r.data(), topo.nr, topo.nz,
      cfg.numerics.has_physical_rz_axis);
  apply_r_outer_boundary_kernel<<<blocks_j, 256>>>(
      state.x_r.data(), state.x_z.data(), state.v_r.data(), state.v_z.data(), topo.nr,
      topo.nz, static_cast<int>(r_outer_type), cfg.mesh.r_max, cfg.mesh.z_min, dz);
  apply_z_bottom_boundary_kernel<<<blocks_i, 256>>>(
      state.x_r.data(), state.x_z.data(), state.v_r.data(), state.v_z.data(), topo.nr,
      topo.nz, static_cast<int>(z_bottom_type), cfg.mesh.r_min, cfg.mesh.z_min, dr);
  apply_z_top_boundary_kernel<<<blocks_i, 256>>>(
      state.x_r.data(), state.x_z.data(), state.v_r.data(), state.v_z.data(), topo.nr,
      topo.nz, static_cast<int>(z_top_type), cfg.mesh.r_min, cfg.mesh.z_max, dr);

  if (z_bottom_type == Boundary2DType::STATE_SUPPLY) {
    apply_state_supply_z_bottom_node_kernel<<<blocks_i, 256>>>(
        state.x_z.data(), state.v_z.data(), topo.nr, topo.nz, cfg.mesh.z_min);
  }
  if (z_top_type == Boundary2DType::STATE_SUPPLY) {
    apply_state_supply_z_top_node_kernel<<<blocks_i, 256>>>(
        state.x_z.data(), state.v_z.data(), topo.nr, topo.nz, cfg.mesh.z_max);
  }

  // Axis priority: always enforce v_r=0, x_r=0 on r=0 after corner updates.
  apply_axis_boundary_kernel<<<blocks_j, 256>>>(
      state.v_r.data(), state.x_r.data(), topo.nr, topo.nz,
      cfg.numerics.has_physical_rz_axis);

  cuda_check(cudaGetLastError(), "apply_boundary_2d kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "apply_boundary_2d kernel execution failed");
}

}  // namespace tenryu::hydro
