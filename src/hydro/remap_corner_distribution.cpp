#include "hydro/remap_corner_distribution.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace tenryu::hydro::corner_distribution {

CellDistributionResult distribute_cell_mass_scaling(
    const double cell_mass,
    const double* corner_volume,
    const double* ref_corner_mass,
    const double* rho_min,
    const double* rho_max,
    const int n_corners,
    double* out_corner_mass) {
  CellDistributionResult result;

  double cell_volume = 0.0;
  double ref_mass_sum = 0.0;
  for (int p = 0; p < n_corners; ++p) {
    if (corner_volume[p] == 0.0) {
      continue;
    }
    cell_volume += corner_volume[p];
    ref_mass_sum += ref_corner_mass[p];
  }

  if (cell_volume == 0.0) {
    for (int p = 0; p < n_corners; ++p) {
      out_corner_mass[p] = 0.0;
    }
    if (cell_mass != 0.0) {
      result.bounds_infeasible = true;
      result.beta = 0.0;
    }
    return result;
  }

  const double reference_correction = cell_mass - ref_mass_sum;
  const double mean_density = cell_mass / cell_volume;

  for (int p = 0; p < n_corners; ++p) {
    const double volume = corner_volume[p];
    if (volume == 0.0) {
      continue;
    }
    const double mean_mass = mean_density * volume;
    const double min_mass = rho_min[p] * volume;
    const double max_mass = rho_max[p] * volume;
    if (mean_mass < min_mass || mean_mass > max_mass) {
      result.bounds_infeasible = true;
      result.beta = 0.0;
    }
  }

  if (!result.bounds_infeasible) {
    for (int p = 0; p < n_corners; ++p) {
      const double volume = corner_volume[p];
      if (volume == 0.0) {
        continue;
      }
      const double normalized_ref_mass =
          ref_corner_mass[p] +
          (volume / cell_volume) * reference_correction;
      const double mean_mass = mean_density * volume;
      const double min_mass = rho_min[p] * volume;
      const double max_mass = rho_max[p] * volume;
      if (normalized_ref_mass > max_mass) {
        const double candidate =
            (max_mass - mean_mass) /
            (normalized_ref_mass - mean_mass);
        result.beta = std::min(result.beta, candidate);
      } else if (normalized_ref_mass < min_mass) {
        const double candidate =
            (min_mass - mean_mass) /
            (normalized_ref_mass - mean_mass);
        result.beta = std::min(result.beta, candidate);
      }
    }
  }

  for (int p = 0; p < n_corners; ++p) {
    const double volume = corner_volume[p];
    if (volume == 0.0) {
      out_corner_mass[p] = 0.0;
      continue;
    }
    const double normalized_ref_mass =
        ref_corner_mass[p] +
        (volume / cell_volume) * reference_correction;
    const double mean_mass = mean_density * volume;
    if (result.beta == 1.0) {
      out_corner_mass[p] = normalized_ref_mass;
    } else if (result.beta == 0.0) {
      out_corner_mass[p] = mean_mass;
    } else {
      out_corner_mass[p] =
          mean_mass + result.beta * (normalized_ref_mass - mean_mass);
    }
  }

  return result;
}

namespace {

struct ComponentGradient {
  double r = 0.0;
  double z = 0.0;
};

double limited_reconstruction(
    const double center_value,
    const ComponentGradient& gradient,
    const double delta_r,
    const double delta_z,
    const double stencil_min,
    const double stencil_max) {
  const double increment =
      gradient.r * delta_r + gradient.z * delta_z;
  double alpha = 1.0;
  if (increment > 0.0) {
    alpha = std::min(1.0, (stencil_max - center_value) / increment);
  } else if (increment < 0.0) {
    alpha = std::min(1.0, (stencil_min - center_value) / increment);
  }
  alpha = std::max(0.0, alpha);
  return center_value + alpha * increment;
}

void store_spherical_components(
    const double r,
    const double z,
    const double velocity_r,
    const double velocity_z,
    double* spherical_r_out,
    double* spherical_theta_out,
    double* spherical_r_min_out,
    double* spherical_r_max_out,
    double* spherical_theta_min_out,
    double* spherical_theta_max_out) {
  if (spherical_r_out == nullptr && spherical_theta_out == nullptr &&
      spherical_r_min_out == nullptr && spherical_r_max_out == nullptr &&
      spherical_theta_min_out == nullptr &&
      spherical_theta_max_out == nullptr) {
    return;
  }
  const double radius = std::sqrt(r * r + z * z);
  // Origin query point: continuous +z-axis limit of the spherical basis
  // (matches the origin convention of the AW engine's basis build).
  const double basis_r = radius > 0.0 ? r / radius : 0.0;
  const double basis_z = radius > 0.0 ? z / radius : 1.0;
  const double radial_r = std::fma(velocity_r, basis_r, 0.0);
  const double radial_z = std::fma(velocity_z, basis_z, 0.0);
  const double theta_r = std::fma(velocity_r, basis_z, 0.0);
  const double theta_z = std::fma(velocity_z, basis_r, 0.0);
  const double spherical_r = radial_r + radial_z;
  const double spherical_theta = theta_r - theta_z;
  if (spherical_r_out != nullptr) {
    *spherical_r_out = spherical_r;
  }
  if (spherical_theta_out != nullptr) {
    *spherical_theta_out = spherical_theta;
  }
  if (spherical_r_min_out != nullptr) {
    *spherical_r_min_out = spherical_r;
  }
  if (spherical_r_max_out != nullptr) {
    *spherical_r_max_out = spherical_r;
  }
  if (spherical_theta_min_out != nullptr) {
    *spherical_theta_min_out = spherical_theta;
  }
  if (spherical_theta_max_out != nullptr) {
    *spherical_theta_max_out = spherical_theta;
  }
}

}  // namespace

void reference_velocity_at(
    const OldMeshView& mesh,
    const std::vector<std::vector<int>>& node_cells,
    const int old_node,
    const double r,
    const double z,
    double* u_r_out,
    double* u_z_out,
    double* u_spherical_r_out,
    double* u_spherical_theta_out,
    double* u_spherical_r_min_out,
    double* u_spherical_r_max_out,
    double* u_spherical_theta_min_out,
    double* u_spherical_theta_max_out) {
  std::vector<int> incident_cells =
      node_cells[static_cast<std::size_t>(old_node)];
  std::sort(incident_cells.begin(), incident_cells.end());
  incident_cells.erase(
      std::unique(incident_cells.begin(), incident_cells.end()),
      incident_cells.end());

  std::vector<int> stencil_nodes;
  for (const int cell : incident_cells) {
    if (cell < 0 || cell >= mesh.n_cells) {
      continue;
    }
    const int begin = mesh.cell_node_csr_offsets[cell];
    const int end = mesh.cell_node_csr_offsets[cell + 1];
    for (int entry = begin; entry < end; ++entry) {
      const int node = mesh.cell_node_csr_indices[entry];
      if (node >= 0 && node < mesh.n_nodes) {
        stencil_nodes.push_back(node);
      }
    }
  }
  std::sort(stencil_nodes.begin(), stencil_nodes.end());
  stencil_nodes.erase(
      std::unique(stencil_nodes.begin(), stencil_nodes.end()),
      stencil_nodes.end());

  const double center_r = mesh.node_r[old_node];
  const double center_z = mesh.node_z[old_node];
  const double center_velocity_r = mesh.vel_r[old_node];
  const double center_velocity_z = mesh.vel_z[old_node];
  bool uniform_velocity = true;
  for (const int node : stencil_nodes) {
    if (std::memcmp(
            mesh.vel_r + node, mesh.vel_r + old_node,
            sizeof(double)) != 0 ||
        std::memcmp(
            mesh.vel_z + node, mesh.vel_z + old_node,
            sizeof(double)) != 0) {
      uniform_velocity = false;
      break;
    }
  }
  if (uniform_velocity) {
    *u_r_out = center_velocity_r;
    *u_z_out = center_velocity_z;
    store_spherical_components(
        r, z, center_velocity_r, center_velocity_z,
        u_spherical_r_out, u_spherical_theta_out,
        u_spherical_r_min_out, u_spherical_r_max_out,
        u_spherical_theta_min_out, u_spherical_theta_max_out);
    return;
  }

  const double center_radius =
      std::sqrt(center_r * center_r + center_z * center_z);
  if (center_radius == 0.0) {
    // Per the consult ruling, the r=0 degree of freedom belongs to the
    // center/pool topology. Qualification vehicles keep the origin inside the
    // pool; this constant path supports general use and synthetic test meshes.
    *u_r_out = center_velocity_r;
    *u_z_out = center_velocity_z;
    store_spherical_components(
        r, z, center_velocity_r, center_velocity_z,
        u_spherical_r_out, u_spherical_theta_out,
        u_spherical_r_min_out, u_spherical_r_max_out,
        u_spherical_theta_min_out, u_spherical_theta_max_out);
    return;
  }
  if (r == center_r && z == center_z) {
    *u_r_out = center_velocity_r;
    *u_z_out = center_velocity_z;
    store_spherical_components(
        r, z, center_velocity_r, center_velocity_z,
        u_spherical_r_out, u_spherical_theta_out,
        u_spherical_r_min_out, u_spherical_r_max_out,
        u_spherical_theta_min_out, u_spherical_theta_max_out);
    return;
  }

  const double center_basis_r = center_r / center_radius;
  const double center_basis_z = center_z / center_radius;
  // Match the rz_moments FMA discipline: round each product explicitly before
  // the dot-product addition/subtraction so contraction cannot spoil a null.
  const double center_velocity_radial_r =
      std::fma(center_velocity_r, center_basis_r, 0.0);
  const double center_velocity_radial_z =
      std::fma(center_velocity_z, center_basis_z, 0.0);
  const double center_velocity_theta_r =
      std::fma(center_velocity_r, center_basis_z, 0.0);
  const double center_velocity_theta_z =
      std::fma(center_velocity_z, center_basis_r, 0.0);
  const double center_velocity_spherical_r =
      center_velocity_radial_r + center_velocity_radial_z;
  const double center_velocity_spherical_theta =
      center_velocity_theta_r - center_velocity_theta_z;
  double velocity_spherical_r_min = center_velocity_spherical_r;
  double velocity_spherical_r_max = center_velocity_spherical_r;
  double velocity_spherical_theta_min = center_velocity_spherical_theta;
  double velocity_spherical_theta_max = center_velocity_spherical_theta;
  bool velocity_spherical_r_is_null = center_velocity_spherical_r == 0.0;
  bool velocity_spherical_theta_is_null =
      center_velocity_spherical_theta == 0.0;
  double matrix_rr = 0.0;
  double matrix_rz = 0.0;
  double matrix_zz = 0.0;
  double rhs_rr = 0.0;
  double rhs_rz = 0.0;
  double rhs_zr = 0.0;
  double rhs_zz = 0.0;
  for (const int node : stencil_nodes) {
    const double node_r = mesh.node_r[node];
    const double node_z = mesh.node_z[node];
    const double node_radius =
        std::sqrt(node_r * node_r + node_z * node_z);
    if (node_radius == 0.0) {
      continue;
    }
    const double node_basis_r = node_r / node_radius;
    const double node_basis_z = node_z / node_radius;
    const double node_velocity_radial_r =
        std::fma(mesh.vel_r[node], node_basis_r, 0.0);
    const double node_velocity_radial_z =
        std::fma(mesh.vel_z[node], node_basis_z, 0.0);
    const double node_velocity_theta_r =
        std::fma(mesh.vel_r[node], node_basis_z, 0.0);
    const double node_velocity_theta_z =
        std::fma(mesh.vel_z[node], node_basis_r, 0.0);
    const double velocity_spherical_r =
        node_velocity_radial_r + node_velocity_radial_z;
    const double velocity_spherical_theta =
        node_velocity_theta_r - node_velocity_theta_z;
    velocity_spherical_r_is_null =
        velocity_spherical_r_is_null && velocity_spherical_r == 0.0;
    velocity_spherical_theta_is_null =
        velocity_spherical_theta_is_null &&
        velocity_spherical_theta == 0.0;
    const double delta_node_r = node_r - center_r;
    const double delta_node_z = node_z - center_z;
    const double delta_velocity_r =
        velocity_spherical_r - center_velocity_spherical_r;
    const double delta_velocity_z =
        velocity_spherical_theta - center_velocity_spherical_theta;
    matrix_rr += delta_node_r * delta_node_r;
    matrix_rz += delta_node_r * delta_node_z;
    matrix_zz += delta_node_z * delta_node_z;
    rhs_rr += delta_node_r * delta_velocity_r;
    rhs_rz += delta_node_z * delta_velocity_r;
    rhs_zr += delta_node_r * delta_velocity_z;
    rhs_zz += delta_node_z * delta_velocity_z;
    velocity_spherical_r_min =
        std::min(velocity_spherical_r_min, velocity_spherical_r);
    velocity_spherical_r_max =
        std::max(velocity_spherical_r_max, velocity_spherical_r);
    velocity_spherical_theta_min =
        std::min(velocity_spherical_theta_min, velocity_spherical_theta);
    velocity_spherical_theta_max =
        std::max(velocity_spherical_theta_max, velocity_spherical_theta);
  }

  ComponentGradient gradient_r;
  ComponentGradient gradient_z;
  const double determinant =
      matrix_rr * matrix_zz - matrix_rz * matrix_rz;
  if (determinant > 0.0) {
    if (!velocity_spherical_r_is_null) {
      gradient_r.r =
          (rhs_rr * matrix_zz - rhs_rz * matrix_rz) / determinant;
      gradient_r.z =
          (rhs_rz * matrix_rr - rhs_rr * matrix_rz) / determinant;
    }
    if (!velocity_spherical_theta_is_null) {
      gradient_z.r =
          (rhs_zr * matrix_zz - rhs_zz * matrix_rz) / determinant;
      gradient_z.z =
          (rhs_zz * matrix_rr - rhs_zr * matrix_rz) / determinant;
    }
  }

  const double delta_r = r - center_r;
  const double delta_z = z - center_z;
  const double velocity_spherical_r =
      velocity_spherical_r_is_null
          ? 0.0
          : limited_reconstruction(
                center_velocity_spherical_r, gradient_r, delta_r, delta_z,
                velocity_spherical_r_min, velocity_spherical_r_max);
  const double velocity_spherical_theta =
      velocity_spherical_theta_is_null
          ? 0.0
          : limited_reconstruction(
                center_velocity_spherical_theta, gradient_z, delta_r, delta_z,
                velocity_spherical_theta_min, velocity_spherical_theta_max);
  if (u_spherical_r_out != nullptr) {
    *u_spherical_r_out = velocity_spherical_r;
  }
  if (u_spherical_theta_out != nullptr) {
    *u_spherical_theta_out = velocity_spherical_theta;
  }
  if (u_spherical_r_min_out != nullptr) {
    *u_spherical_r_min_out = velocity_spherical_r_min;
  }
  if (u_spherical_r_max_out != nullptr) {
    *u_spherical_r_max_out = velocity_spherical_r_max;
  }
  if (u_spherical_theta_min_out != nullptr) {
    *u_spherical_theta_min_out = velocity_spherical_theta_min;
  }
  if (u_spherical_theta_max_out != nullptr) {
    *u_spherical_theta_max_out = velocity_spherical_theta_max;
  }
  const double query_radius = std::sqrt(r * r + z * z);
  assert(query_radius > 0.0);
  const double query_basis_r = r / query_radius;
  const double query_basis_z = z / query_radius;
  const double rebuilt_radial_r =
      std::fma(velocity_spherical_r, query_basis_r, 0.0);
  const double rebuilt_theta_r =
      std::fma(velocity_spherical_theta, query_basis_z, 0.0);
  const double rebuilt_radial_z =
      std::fma(velocity_spherical_r, query_basis_z, 0.0);
  const double rebuilt_theta_z =
      std::fma(velocity_spherical_theta, query_basis_r, 0.0);
  *u_r_out = rebuilt_radial_r + rebuilt_theta_r;
  *u_z_out = rebuilt_radial_z - rebuilt_theta_z;
}

namespace {

double aggregate_component_tolerance(
    const double target,
    const double contribution_scale,
    const double epsilon_factor) {
  return epsilon_factor * std::numeric_limits<double>::epsilon() *
         (std::abs(target) + contribution_scale);
}

bool solve_aggregate_multiplier(
    const double matrix_rr,
    const double matrix_rz,
    const double matrix_zz,
    const double residual_r,
    const double residual_z,
    const double tolerance_r,
    const double tolerance_z,
    double* lambda_r,
    double* lambda_z) {
  const double determinant =
      std::fma(matrix_rr, matrix_zz, -matrix_rz * matrix_rz);
  if (determinant > 0.0) {
    *lambda_r =
        (residual_r * matrix_zz - residual_z * matrix_rz) / determinant;
    *lambda_z =
        (residual_z * matrix_rr - residual_r * matrix_rz) / determinant;
    return true;
  }

  *lambda_r = 0.0;
  *lambda_z = 0.0;
  if (matrix_rr == 0.0 && matrix_zz == 0.0) {
    return std::abs(residual_r) <= tolerance_r &&
           std::abs(residual_z) <= tolerance_z;
  }
  if (matrix_rr >= matrix_zz) {
    assert(matrix_rr > 0.0);
    *lambda_r = residual_r / matrix_rr;
  } else {
    assert(matrix_zz > 0.0);
    *lambda_z = residual_z / matrix_zz;
  }
  const double reproduced_r =
      matrix_rr * *lambda_r + matrix_rz * *lambda_z;
  const double reproduced_z =
      matrix_rz * *lambda_r + matrix_zz * *lambda_z;
  return std::abs(reproduced_r - residual_r) <= tolerance_r &&
         std::abs(reproduced_z - residual_z) <= tolerance_z;
}

int count_aggregate_active_nodes(
    const AwAggregateProjectionInputs& in,
    const std::vector<std::uint8_t>& active_s,
    const std::vector<std::uint8_t>& active_t) {
  int active_nodes = 0;
  for (int node = 0; node < in.n_nodes; ++node) {
    if (active_s[static_cast<std::size_t>(node)] != 0 ||
        (in.axis_mask[node] == 0 &&
         active_t[static_cast<std::size_t>(node)] != 0)) {
      ++active_nodes;
    }
  }
  return active_nodes;
}

void audit_aggregate_conservation(
    const AwAggregateProjectionInputs& in,
    const double* u_s,
    const double* u_t) {
  double momentum_r = 0.0;
  double momentum_z = 0.0;
  double scale_r = 0.0;
  double scale_z = 0.0;
  for (int node = 0; node < in.n_nodes; ++node) {
    const double velocity_r =
        u_s[node] * in.basis_s_r[node] +
        u_t[node] * in.basis_t_r[node];
    const double velocity_z =
        u_s[node] * in.basis_s_z[node] +
        u_t[node] * in.basis_t_z[node];
    const double contribution_r = in.a_node[node] * velocity_r;
    const double contribution_z = in.a_node[node] * velocity_z;
    momentum_r += contribution_r;
    momentum_z += contribution_z;
    scale_r += std::abs(contribution_r);
    scale_z += std::abs(contribution_z);
  }
  const double tolerance_r =
      aggregate_component_tolerance(in.target_r, scale_r, 256.0);
  const double tolerance_z =
      aggregate_component_tolerance(in.target_z, scale_z, 256.0);
  assert(std::abs(momentum_r - in.target_r) <= tolerance_r);
  assert(std::abs(momentum_z - in.target_z) <= tolerance_z);
}

}  // namespace

AwAggregateProjectionResult project_aggregate_target_state(
    const AwAggregateProjectionInputs& in,
    double* out_u_s,
    double* out_u_t) {
  assert(in.n_nodes >= 0);
  assert(in.u_ref_s != nullptr);
  assert(in.u_ref_t != nullptr);
  assert(in.basis_s_r != nullptr);
  assert(in.basis_s_z != nullptr);
  assert(in.basis_t_r != nullptr);
  assert(in.basis_t_z != nullptr);
  assert(in.mu_node != nullptr);
  assert(in.a_node != nullptr);
  assert(in.s_min != nullptr);
  assert(in.s_max != nullptr);
  assert(in.t_min != nullptr);
  assert(in.t_max != nullptr);
  assert(in.axis_mask != nullptr);
  assert(out_u_s != nullptr);
  assert(out_u_t != nullptr);

  double reference_momentum_r = 0.0;
  double reference_momentum_z = 0.0;
  double reference_scale_r = 0.0;
  double reference_scale_z = 0.0;
  for (int node = 0; node < in.n_nodes; ++node) {
    assert(in.mu_node[node] > 0.0);
    assert(in.s_min[node] <= in.s_max[node]);
    assert(in.t_min[node] <= in.t_max[node]);
    if (in.axis_mask[node] != 0) {
      assert(in.a_node[node] == 0.0);
      assert(in.t_min[node] == 0.0);
      assert(in.t_max[node] == 0.0);
      assert(in.u_ref_t[node] == 0.0);
    }
    out_u_s[node] = in.u_ref_s[node];
    out_u_t[node] = in.u_ref_t[node];
    const double velocity_r =
        in.u_ref_s[node] * in.basis_s_r[node] +
        in.u_ref_t[node] * in.basis_t_r[node];
    const double velocity_z =
        in.u_ref_s[node] * in.basis_s_z[node] +
        in.u_ref_t[node] * in.basis_t_z[node];
    const double contribution_r = in.a_node[node] * velocity_r;
    const double contribution_z = in.a_node[node] * velocity_z;
    reference_momentum_r += contribution_r;
    reference_momentum_z += contribution_z;
    reference_scale_r += std::abs(contribution_r);
    reference_scale_z += std::abs(contribution_z);
  }

  AwAggregateProjectionResult result;
  const double reference_tolerance_r = aggregate_component_tolerance(
      in.target_r, reference_scale_r, 64.0);
  const double reference_tolerance_z = aggregate_component_tolerance(
      in.target_z, reference_scale_z, 64.0);
  if (std::abs(in.target_r - reference_momentum_r) <=
          reference_tolerance_r &&
      std::abs(in.target_z - reference_momentum_z) <=
          reference_tolerance_z) {
    result.exact_reference = true;
    audit_aggregate_conservation(in, out_u_s, out_u_t);
    return result;
  }

  std::vector<std::uint8_t> active_s(
      static_cast<std::size_t>(in.n_nodes), 0);
  std::vector<std::uint8_t> active_t(
      static_cast<std::size_t>(in.n_nodes), 0);
  for (int node = 0; node < in.n_nodes; ++node) {
    if (in.s_min[node] == in.s_max[node]) {
      active_s[static_cast<std::size_t>(node)] = 1;
      out_u_s[node] = in.s_min[node];
    }
    if (in.axis_mask[node] != 0 || in.t_min[node] == in.t_max[node]) {
      active_t[static_cast<std::size_t>(node)] = 1;
      out_u_t[node] = in.axis_mask[node] != 0 ? 0.0 : in.t_min[node];
    }
  }

  const int max_iterations = 2 * in.n_nodes + 1;
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    double base_r = 0.0;
    double base_z = 0.0;
    double base_scale_r = 0.0;
    double base_scale_z = 0.0;
    double matrix_rr = 0.0;
    double matrix_rz = 0.0;
    double matrix_zz = 0.0;
    for (int node = 0; node < in.n_nodes; ++node) {
      const bool s_is_active =
          active_s[static_cast<std::size_t>(node)] != 0;
      const bool t_is_active =
          active_t[static_cast<std::size_t>(node)] != 0;
      const double base_s = s_is_active ? out_u_s[node] : in.u_ref_s[node];
      const double base_t = t_is_active ? out_u_t[node] : in.u_ref_t[node];
      const double contribution_r = in.a_node[node] *
          (base_s * in.basis_s_r[node] +
           base_t * in.basis_t_r[node]);
      const double contribution_z = in.a_node[node] *
          (base_s * in.basis_s_z[node] +
           base_t * in.basis_t_z[node]);
      base_r += contribution_r;
      base_z += contribution_z;
      base_scale_r += std::abs(contribution_r);
      base_scale_z += std::abs(contribution_z);

      const double rank_weight =
          in.a_node[node] * in.a_node[node] / in.mu_node[node];
      if (!s_is_active) {
        matrix_rr += rank_weight * in.basis_s_r[node] * in.basis_s_r[node];
        matrix_rz += rank_weight * in.basis_s_r[node] * in.basis_s_z[node];
        matrix_zz += rank_weight * in.basis_s_z[node] * in.basis_s_z[node];
      }
      if (!t_is_active) {
        matrix_rr += rank_weight * in.basis_t_r[node] * in.basis_t_r[node];
        matrix_rz += rank_weight * in.basis_t_r[node] * in.basis_t_z[node];
        matrix_zz += rank_weight * in.basis_t_z[node] * in.basis_t_z[node];
      }
    }

    const double residual_r = in.target_r - base_r;
    const double residual_z = in.target_z - base_z;
    const double tolerance_r =
        aggregate_component_tolerance(in.target_r, base_scale_r, 256.0);
    const double tolerance_z =
        aggregate_component_tolerance(in.target_z, base_scale_z, 256.0);
    double lambda_r = 0.0;
    double lambda_z = 0.0;
    if (!solve_aggregate_multiplier(
            matrix_rr, matrix_rz, matrix_zz, residual_r, residual_z,
            tolerance_r, tolerance_z, &lambda_r, &lambda_z)) {
      result.infeasible = true;
      result.active_bound_nodes =
          count_aggregate_active_nodes(in, active_s, active_t);
      return result;
    }

    bool added_bound = false;
    for (int node = 0; node < in.n_nodes; ++node) {
      const double correction_scale = in.a_node[node] / in.mu_node[node];
      if (active_s[static_cast<std::size_t>(node)] == 0) {
        const double candidate = in.u_ref_s[node] + correction_scale *
            (lambda_r * in.basis_s_r[node] +
             lambda_z * in.basis_s_z[node]);
        if (candidate < in.s_min[node]) {
          out_u_s[node] = in.s_min[node];
          active_s[static_cast<std::size_t>(node)] = 1;
          added_bound = true;
        } else if (candidate > in.s_max[node]) {
          out_u_s[node] = in.s_max[node];
          active_s[static_cast<std::size_t>(node)] = 1;
          added_bound = true;
        } else {
          out_u_s[node] = candidate;
        }
      }
      if (active_t[static_cast<std::size_t>(node)] == 0) {
        const double candidate = in.u_ref_t[node] + correction_scale *
            (lambda_r * in.basis_t_r[node] +
             lambda_z * in.basis_t_z[node]);
        if (candidate < in.t_min[node]) {
          out_u_t[node] = in.t_min[node];
          active_t[static_cast<std::size_t>(node)] = 1;
          added_bound = true;
        } else if (candidate > in.t_max[node]) {
          out_u_t[node] = in.t_max[node];
          active_t[static_cast<std::size_t>(node)] = 1;
          added_bound = true;
        } else {
          out_u_t[node] = candidate;
        }
      }
    }
    if (!added_bound) {
      result.active_bound_nodes =
          count_aggregate_active_nodes(in, active_s, active_t);
      audit_aggregate_conservation(in, out_u_s, out_u_t);
      return result;
    }
  }

  result.infeasible = true;
  result.active_bound_nodes =
      count_aggregate_active_nodes(in, active_s, active_t);
  return result;
}

CornerMomentumDistributionResult distribute_cell_momentum_scaling(
    const OldMeshView& old_mesh,
    const std::vector<std::vector<int>>& old_node_cells,
    const CornerRemapInputs& inputs,
    double* out_corner_momentum_r,
    double* out_corner_momentum_z) {
  std::vector<double> reference_momentum_r(
      static_cast<std::size_t>(inputs.n_corners), 0.0);
  std::vector<double> reference_momentum_z(
      static_cast<std::size_t>(inputs.n_corners), 0.0);
  double reference_momentum_r_sum = 0.0;
  double reference_momentum_z_sum = 0.0;
  for (int corner = 0; corner < inputs.n_corners; ++corner) {
    double reference_velocity_r = 0.0;
    double reference_velocity_z = 0.0;
    reference_velocity_at(
        old_mesh, old_node_cells, inputs.containing_old_node[corner],
        inputs.corner_centroid_r[corner], inputs.corner_centroid_z[corner],
        &reference_velocity_r, &reference_velocity_z);
    reference_momentum_r[static_cast<std::size_t>(corner)] =
        reference_velocity_r * inputs.corner_mass[corner];
    reference_momentum_z[static_cast<std::size_t>(corner)] =
        reference_velocity_z * inputs.corner_mass[corner];
    reference_momentum_r_sum +=
        reference_momentum_r[static_cast<std::size_t>(corner)];
    reference_momentum_z_sum +=
        reference_momentum_z[static_cast<std::size_t>(corner)];
  }

  CornerMomentumDistributionResult result;
  if (inputs.cell_momentum_r == reference_momentum_r_sum) {
    std::copy(
        reference_momentum_r.begin(), reference_momentum_r.end(),
        out_corner_momentum_r);
  } else {
    result.momentum_r = distribute_cell_mass_scaling(
        inputs.cell_momentum_r, inputs.corner_mass,
        reference_momentum_r.data(), inputs.velocity_r_min,
        inputs.velocity_r_max, inputs.n_corners, out_corner_momentum_r);
  }
  if (inputs.cell_momentum_z == reference_momentum_z_sum) {
    std::copy(
        reference_momentum_z.begin(), reference_momentum_z.end(),
        out_corner_momentum_z);
  } else {
    result.momentum_z = distribute_cell_mass_scaling(
        inputs.cell_momentum_z, inputs.corner_mass,
        reference_momentum_z.data(), inputs.velocity_z_min,
        inputs.velocity_z_max, inputs.n_corners, out_corner_momentum_z);
  }
  return result;
}

void recover_nodal_velocity(
    const int* node_corner_csr_offsets,
    const int* node_corner_csr_indices,
    const int n_nodes,
    const double* corner_mass,
    const double* corner_momentum_r,
    const double* corner_momentum_z,
    double* out_nodal_mass,
    double* out_nodal_momentum_r,
    double* out_nodal_momentum_z,
    double* out_nodal_velocity_r,
    double* out_nodal_velocity_z) {
  for (int node = 0; node < n_nodes; ++node) {
    double mass = 0.0;
    double momentum_r = 0.0;
    double momentum_z = 0.0;
    double uniform_corner_velocity_r = 0.0;
    double uniform_corner_velocity_z = 0.0;
    bool have_corner_velocity = false;
    bool uniform_corner_velocity = true;
    const int begin = node_corner_csr_offsets[node];
    const int end = node_corner_csr_offsets[node + 1];
    for (int entry = begin; entry < end; ++entry) {
      const int corner = node_corner_csr_indices[entry];
      if (corner < 0) {
        continue;
      }
      mass += corner_mass[corner];
      momentum_r += corner_momentum_r[corner];
      momentum_z += corner_momentum_z[corner];
      if (corner_mass[corner] == 0.0) {
        if (corner_momentum_r[corner] != 0.0 ||
            corner_momentum_z[corner] != 0.0) {
          uniform_corner_velocity = false;
        }
        continue;
      }
      const double corner_velocity_r =
          corner_momentum_r[corner] / corner_mass[corner];
      const double corner_velocity_z =
          corner_momentum_z[corner] / corner_mass[corner];
      if (!have_corner_velocity) {
        uniform_corner_velocity_r = corner_velocity_r;
        uniform_corner_velocity_z = corner_velocity_z;
        have_corner_velocity = true;
      } else if (
          std::memcmp(
              &corner_velocity_r, &uniform_corner_velocity_r,
              sizeof(double)) != 0 ||
          std::memcmp(
              &corner_velocity_z, &uniform_corner_velocity_z,
              sizeof(double)) != 0) {
        uniform_corner_velocity = false;
      }
    }
    out_nodal_mass[node] = mass;
    out_nodal_momentum_r[node] = momentum_r;
    out_nodal_momentum_z[node] = momentum_z;
    if (mass == 0.0) {
      out_nodal_velocity_r[node] = 0.0;
      out_nodal_velocity_z[node] = 0.0;
    } else if (have_corner_velocity && uniform_corner_velocity) {
      out_nodal_velocity_r[node] = uniform_corner_velocity_r;
      out_nodal_velocity_z[node] = uniform_corner_velocity_z;
    } else {
      out_nodal_velocity_r[node] = momentum_r / mass;
      out_nodal_velocity_z[node] = momentum_z / mass;
    }
  }
}

namespace {

struct AwComponentSolveResult {
  double lambda = 0.0;
  bool exact_reference = false;
  bool infeasible = false;
};

double aw_component_tolerance(
    const double cell_momentum,
    const AwCornerDistributionInputs& inputs,
    const double* values) {
  double scale = std::abs(cell_momentum);
  for (int corner = 0; corner < inputs.n_corners; ++corner) {
    const int node = inputs.corner_node[corner];
    const double lifted_momentum =
        inputs.omega_node[node] * inputs.mu_corner[corner] * values[corner];
    scale += std::abs(lifted_momentum);
  }
  return 64.0 * std::numeric_limits<double>::epsilon() * scale;
}

AwComponentSolveResult solve_aw_component(
    const AwCornerDistributionInputs& inputs,
    const double cell_momentum,
    const double* reference_velocity,
    const double* velocity_min,
    const double* velocity_max,
    const bool fix_axis_component,
    std::vector<double>* velocity,
    std::vector<std::uint8_t>* active) {
  AwComponentSolveResult result;
  velocity->assign(
      reference_velocity, reference_velocity + inputs.n_corners);
  active->assign(static_cast<std::size_t>(inputs.n_corners), 0);

  double reference_momentum = 0.0;
  for (int corner = 0; corner < inputs.n_corners; ++corner) {
    const int node = inputs.corner_node[corner];
    assert(node >= 0);
    assert(inputs.mu_corner[corner] > 0.0);
    assert(inputs.omega_node[node] >= 0.0);
    assert(velocity_min[corner] <= velocity_max[corner]);
    if (inputs.node_axis_mask[node] != 0) {
      assert(inputs.omega_node[node] == 0.0);
      assert(inputs.u_ref_r[node] == 0.0);
    }
    reference_momentum +=
        inputs.omega_node[node] * inputs.mu_corner[corner] *
        reference_velocity[corner];
  }
  const double reference_residual = cell_momentum - reference_momentum;
  if (std::abs(reference_residual) <=
      aw_component_tolerance(
          cell_momentum, inputs, reference_velocity)) {
    result.exact_reference = true;
    return result;
  }

  for (int iteration = 0; iteration <= inputs.n_corners; ++iteration) {
    double base_momentum = 0.0;
    double denominator = 0.0;
    for (int corner = 0; corner < inputs.n_corners; ++corner) {
      const int node = inputs.corner_node[corner];
      const double omega = inputs.omega_node[node];
      const double mu = inputs.mu_corner[corner];
      const bool fixed_axis =
          fix_axis_component && inputs.node_axis_mask[node] != 0;
      if (fixed_axis) {
        (*velocity)[static_cast<std::size_t>(corner)] = 0.0;
        continue;
      }
      if ((*active)[static_cast<std::size_t>(corner)] != 0) {
        base_momentum +=
            omega * mu * (*velocity)[static_cast<std::size_t>(corner)];
      } else {
        base_momentum += omega * mu * reference_velocity[corner];
        denominator += mu * omega * omega;
      }
    }

    const double residual = cell_momentum - base_momentum;
    std::vector<double> base_velocity(
        static_cast<std::size_t>(inputs.n_corners));
    for (int corner = 0; corner < inputs.n_corners; ++corner) {
      base_velocity[static_cast<std::size_t>(corner)] =
          ((*active)[static_cast<std::size_t>(corner)] != 0)
              ? (*velocity)[static_cast<std::size_t>(corner)]
              : reference_velocity[corner];
    }
    const double tolerance =
        aw_component_tolerance(cell_momentum, inputs, base_velocity.data());
    if (denominator == 0.0) {
      if (std::abs(residual) > tolerance) {
        result.infeasible = true;
        return result;
      }
      result.lambda = 0.0;
    } else {
      result.lambda = residual / denominator;
    }

    bool added_bound = false;
    for (int corner = 0; corner < inputs.n_corners; ++corner) {
      const int node = inputs.corner_node[corner];
      const bool fixed_axis =
          fix_axis_component && inputs.node_axis_mask[node] != 0;
      if (fixed_axis ||
          (*active)[static_cast<std::size_t>(corner)] != 0) {
        continue;
      }
      const double candidate =
          reference_velocity[corner] +
          inputs.omega_node[node] * result.lambda;
      if (candidate < velocity_min[corner]) {
        (*velocity)[static_cast<std::size_t>(corner)] = velocity_min[corner];
        (*active)[static_cast<std::size_t>(corner)] = 1;
        added_bound = true;
      } else if (candidate > velocity_max[corner]) {
        (*velocity)[static_cast<std::size_t>(corner)] = velocity_max[corner];
        (*active)[static_cast<std::size_t>(corner)] = 1;
        added_bound = true;
      } else {
        (*velocity)[static_cast<std::size_t>(corner)] = candidate;
      }
    }
    if (!added_bound) {
      return result;
    }
  }

  result.infeasible = true;
  return result;
}

void audit_aw_component_conservation(
    const AwCornerDistributionInputs& inputs,
    const double cell_momentum,
    const double* pi) {
  double distributed_momentum = 0.0;
  double scale = std::abs(cell_momentum);
  for (int corner = 0; corner < inputs.n_corners; ++corner) {
    const int node = inputs.corner_node[corner];
    const double contribution = inputs.omega_node[node] * pi[corner];
    distributed_momentum += contribution;
    scale += std::abs(contribution);
  }
  const double tolerance =
      256.0 * std::numeric_limits<double>::epsilon() * scale;
  assert(std::abs(distributed_momentum - cell_momentum) <= tolerance);
}

}  // namespace

AwMomentumDistributionResult distribute_cell_momentum_aw(
    const AwCornerDistributionInputs& inputs,
    double* out_pi_r,
    double* out_pi_z) {
  assert(inputs.n_corners >= 0);
  assert(inputs.mu_corner != nullptr);
  assert(inputs.omega_node != nullptr);
  assert(inputs.corner_node != nullptr);
  assert(inputs.u_ref_r != nullptr);
  assert(inputs.u_ref_z != nullptr);
  assert(inputs.velocity_r_min != nullptr);
  assert(inputs.velocity_r_max != nullptr);
  assert(inputs.velocity_z_min != nullptr);
  assert(inputs.velocity_z_max != nullptr);
  assert(inputs.node_axis_mask != nullptr);
  assert(out_pi_r != nullptr);
  assert(out_pi_z != nullptr);

  std::vector<double> reference_r(
      static_cast<std::size_t>(inputs.n_corners));
  std::vector<double> reference_z(
      static_cast<std::size_t>(inputs.n_corners));
  for (int corner = 0; corner < inputs.n_corners; ++corner) {
    const int node = inputs.corner_node[corner];
    reference_r[static_cast<std::size_t>(corner)] = inputs.u_ref_r[node];
    reference_z[static_cast<std::size_t>(corner)] = inputs.u_ref_z[node];
    out_pi_r[corner] = inputs.mu_corner[corner] * inputs.u_ref_r[node];
    out_pi_z[corner] = inputs.mu_corner[corner] * inputs.u_ref_z[node];
  }

  std::vector<double> velocity_r;
  std::vector<double> velocity_z;
  std::vector<std::uint8_t> active_r;
  std::vector<std::uint8_t> active_z;
  const AwComponentSolveResult solved_r = solve_aw_component(
      inputs, inputs.cell_momentum_r, reference_r.data(),
      inputs.velocity_r_min, inputs.velocity_r_max, true,
      &velocity_r, &active_r);
  const AwComponentSolveResult solved_z = solve_aw_component(
      inputs, inputs.cell_momentum_z, reference_z.data(),
      inputs.velocity_z_min, inputs.velocity_z_max, false,
      &velocity_z, &active_z);

  AwMomentumDistributionResult result;
  result.lambda_r = solved_r.lambda;
  result.lambda_z = solved_z.lambda;
  result.exact_reference =
      solved_r.exact_reference && solved_z.exact_reference;
  if (solved_r.infeasible || solved_z.infeasible) {
    result.active_bound_corners = -1;
    return result;
  }

  for (int corner = 0; corner < inputs.n_corners; ++corner) {
    out_pi_r[corner] =
        inputs.mu_corner[corner] * velocity_r[static_cast<std::size_t>(corner)];
    out_pi_z[corner] =
        inputs.mu_corner[corner] * velocity_z[static_cast<std::size_t>(corner)];
    if (active_r[static_cast<std::size_t>(corner)] != 0 ||
        active_z[static_cast<std::size_t>(corner)] != 0) {
      ++result.active_bound_corners;
    }
  }
  audit_aw_component_conservation(
      inputs, inputs.cell_momentum_r, out_pi_r);
  audit_aw_component_conservation(
      inputs, inputs.cell_momentum_z, out_pi_z);
  return result;
}

void recover_nodal_velocity_aw(
    const int* node_corner_csr_offsets,
    const int* node_corner_csr_indices,
    const int n_nodes,
    const double* mu_corner,
    const double* pi_r,
    const double* pi_z,
    const double* u_ref_r,
    const double* u_ref_z,
    const double* omega_node,
    const std::uint8_t* node_axis_mask,
    double* out_mu_node,
    double* out_velocity_r,
    double* out_velocity_z,
    double* out_mass_rz,
    double* out_momentum_rz_r,
    double* out_momentum_rz_z) {
  for (int node = 0; node < n_nodes; ++node) {
    const bool axis_node = node_axis_mask[node] != 0;
    if (axis_node) {
      assert(omega_node[node] == 0.0);
      assert(u_ref_r[node] == 0.0);
    }
    double mu_sum = 0.0;
    double delta_pi_r_sum = 0.0;
    double delta_pi_z_sum = 0.0;
    const int begin = node_corner_csr_offsets[node];
    const int end = node_corner_csr_offsets[node + 1];
    for (int entry = begin; entry < end; ++entry) {
      const int corner = node_corner_csr_indices[entry];
      if (corner < 0) {
        continue;
      }
      assert(mu_corner[corner] >= 0.0);
      mu_sum += mu_corner[corner];
      delta_pi_r_sum +=
          pi_r[corner] - mu_corner[corner] * u_ref_r[node];
      delta_pi_z_sum +=
          pi_z[corner] - mu_corner[corner] * u_ref_z[node];
    }

    double velocity_r = u_ref_r[node];
    double velocity_z = u_ref_z[node];
    if (mu_sum != 0.0) {
      if (delta_pi_r_sum != 0.0) {
        velocity_r += delta_pi_r_sum / mu_sum;
      }
      if (delta_pi_z_sum != 0.0) {
        velocity_z += delta_pi_z_sum / mu_sum;
      }
    }
    if (axis_node) {
      velocity_r = 0.0;
    }

    out_mu_node[node] = mu_sum;
    out_velocity_r[node] = velocity_r;
    out_velocity_z[node] = velocity_z;
    const double mass_rz = axis_node ? 0.0 : omega_node[node] * mu_sum;
    if (out_mass_rz != nullptr) {
      out_mass_rz[node] = mass_rz;
    }
    if (out_momentum_rz_r != nullptr) {
      out_momentum_rz_r[node] = mass_rz * velocity_r;
    }
    if (out_momentum_rz_z != nullptr) {
      out_momentum_rz_z[node] = mass_rz * velocity_z;
    }
  }
}

}  // namespace tenryu::hydro::corner_distribution
