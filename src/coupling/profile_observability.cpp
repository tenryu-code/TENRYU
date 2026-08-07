#include "coupling/profile_observability.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "core/config_validate.hpp"
#include "core/state.hpp"
#include "parallel/reduction.hpp"

namespace tenryu::core {

void validate_icf_standard_ale_profile_config(
    const Config& config,
    tenryu::coupling::ProfileObservability* observability) {
  const auto counters =
      validate_icf_standard_ale_profile_config_counted(config, false);
  if (observability != nullptr) {
    observability->note_validator_violations(
        counters.forbidden_config_violations,
        counters.escape_valve_activations);
  }
}

}  // namespace tenryu::core

namespace tenryu::coupling {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kGauss = 0.57735026918962576450914878050195745565;
constexpr double kSignedRatioZeroDenominator = -1.0e300;

double cross2(const double ar, const double az, const double br, const double bz) {
  return ar * bz - az * br;
}

double signed_ratio(const double current, const double initial) {
  return initial != 0.0 ? current / initial : kSignedRatioZeroDenominator;
}

void accumulate_finite_min(const double value, double& min_value, bool& observed) {
  if (!std::isfinite(value)) {
    return;
  }
  min_value = observed ? std::min(min_value, value) : value;
  observed = true;
}

void accumulate_non_nan_max(const double value, double& max_value, bool& observed) {
  if (std::isnan(value)) {
    return;
  }
  max_value = observed ? std::max(max_value, value) : value;
  observed = true;
}

double corner_jacobian_from_quad(const std::array<double, 4>& r,
                                 const std::array<double, 4>& z,
                                 const int corner) {
  const int kp = (corner + 1) & 3;
  const int km = (corner + 3) & 3;
  return cross2(r[static_cast<std::size_t>(kp)] - r[static_cast<std::size_t>(corner)],
                z[static_cast<std::size_t>(kp)] - z[static_cast<std::size_t>(corner)],
                r[static_cast<std::size_t>(km)] - r[static_cast<std::size_t>(corner)],
                z[static_cast<std::size_t>(km)] - z[static_cast<std::size_t>(corner)]);
}

double gauss_jacobian_from_quad(const std::array<double, 4>& r,
                                const std::array<double, 4>& z,
                                const double xi,
                                const double eta) {
  const double dN_dxi[4] = {
      -0.25 * (1.0 - eta),
       0.25 * (1.0 - eta),
       0.25 * (1.0 + eta),
      -0.25 * (1.0 + eta),
  };
  const double dN_deta[4] = {
      -0.25 * (1.0 - xi),
      -0.25 * (1.0 + xi),
       0.25 * (1.0 + xi),
       0.25 * (1.0 - xi),
  };
  double dr_dxi = 0.0;
  double dz_dxi = 0.0;
  double dr_deta = 0.0;
  double dz_deta = 0.0;
  for (int k = 0; k < 4; ++k) {
    const auto idx = static_cast<std::size_t>(k);
    dr_dxi += dN_dxi[k] * r[idx];
    dz_dxi += dN_dxi[k] * z[idx];
    dr_deta += dN_deta[k] * r[idx];
    dz_deta += dN_deta[k] * z[idx];
  }
  return cross2(dr_dxi, dz_dxi, dr_deta, dz_deta);
}

void gauss_jacobian_matrix_from_quad(const std::array<double, 4>& r,
                                     const std::array<double, 4>& z,
                                     const double xi,
                                     const double eta,
                                     double& dr_dxi,
                                     double& dz_dxi,
                                     double& dr_deta,
                                     double& dz_deta) {
  const double dN_dxi[4] = {
      -0.25 * (1.0 - eta),
       0.25 * (1.0 - eta),
       0.25 * (1.0 + eta),
      -0.25 * (1.0 + eta),
  };
  const double dN_deta[4] = {
      -0.25 * (1.0 - xi),
      -0.25 * (1.0 + xi),
       0.25 * (1.0 + xi),
       0.25 * (1.0 - xi),
  };
  dr_dxi = 0.0;
  dz_dxi = 0.0;
  dr_deta = 0.0;
  dz_deta = 0.0;
  for (int k = 0; k < 4; ++k) {
    const auto idx = static_cast<std::size_t>(k);
    dr_dxi += dN_dxi[k] * r[idx];
    dz_dxi += dN_dxi[k] * z[idx];
    dr_deta += dN_deta[k] * r[idx];
    dz_deta += dN_deta[k] * z[idx];
  }
}

double condition_number_2x2(const double a,
                            const double b,
                            const double c,
                            const double d) {
  const double frob2 = a * a + b * b + c * c + d * d;
  const double det = a * d - b * c;
  if (!std::isfinite(frob2) || !std::isfinite(det)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!(frob2 > 0.0)) {
    return std::numeric_limits<double>::infinity();
  }
  const double det2 = det * det;
  if (!(det2 > 0.0) || !std::isfinite(det2)) {
    return std::numeric_limits<double>::infinity();
  }
  const double disc = std::max(0.0, frob2 * frob2 - 4.0 * det2);
  const double sigma_max2 = 0.5 * (frob2 + std::sqrt(disc));
  if (!(sigma_max2 > 0.0) || !std::isfinite(sigma_max2)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double sigma_min2 = det2 / sigma_max2;
  if (!(sigma_min2 > 0.0)) {
    return std::numeric_limits<double>::infinity();
  }
  return std::sqrt(sigma_max2 / sigma_min2);
}

double edge_length_from_quad(const std::array<double, 4>& r,
                             const std::array<double, 4>& z,
                             const int edge) {
  const int ep = (edge + 1) & 3;
  const auto idx = static_cast<std::size_t>(edge);
  const auto idxp = static_cast<std::size_t>(ep);
  return std::hypot(r[idxp] - r[idx], z[idxp] - z[idx]);
}

double min_edge_length_from_quad(const std::array<double, 4>& r,
                                 const std::array<double, 4>& z) {
  double value = std::numeric_limits<double>::infinity();
  for (int edge = 0; edge < 4; ++edge) {
    const double length = edge_length_from_quad(r, z, edge);
    if (!std::isfinite(length)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    value = std::min(value, length);
  }
  return value;
}

double point_to_line_distance(const double pr,
                              const double pz,
                              const double ar,
                              const double az,
                              const double br,
                              const double bz) {
  const double er = br - ar;
  const double ez = bz - az;
  const double edge_length = std::hypot(er, ez);
  if (!std::isfinite(edge_length)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (edge_length == 0.0) {
    return 0.0;
  }
  const double distance = std::abs(cross2(er, ez, pr - ar, pz - az)) / edge_length;
  return std::isfinite(distance) ? distance
                                 : std::numeric_limits<double>::quiet_NaN();
}

double min_altitude_from_quad(const std::array<double, 4>& r,
                              const std::array<double, 4>& z) {
  double value = std::numeric_limits<double>::infinity();
  for (int node = 0; node < 4; ++node) {
    const int e0 = (node + 2) & 3;
    const int e1 = (node + 3) & 3;
    const auto idx = static_cast<std::size_t>(node);
    const auto idx0 = static_cast<std::size_t>(e0);
    const auto idx1 = static_cast<std::size_t>(e1);
    const double altitude =
        point_to_line_distance(r[idx], z[idx], r[idx0], z[idx0], r[idx1], z[idx1]);
    if (!std::isfinite(altitude)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    value = std::min(value, altitude);
  }
  return value;
}

double rz_quad_volume_exact(const std::array<double, 4>& r,
                            const std::array<double, 4>& z) {
  double sum = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int kp = (k + 1) & 3;
    const auto idx = static_cast<std::size_t>(k);
    const auto idxp = static_cast<std::size_t>(kp);
    sum += (r[idx] * z[idxp] - r[idxp] * z[idx]) * (r[idx] + r[idxp]);
  }
  return kPi / 3.0 * sum;
}

std::uint64_t saturating_add_u64(const std::uint64_t lhs,
                                 const std::uint64_t rhs) {
  if (std::numeric_limits<std::uint64_t>::max() - lhs < rhs) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return lhs + rhs;
}

}  // namespace

void ProfileObservability::note_committed_mesh_quality_min(
    const tenryu::core::State& state,
    const tenryu::parallel::Reduction* reduction) {
  if (state.corner_stride != 4) {
    tenryu::core::log_warning(
        "pentagon-belt: mesh-quality-min diagnostic is staged (quad accessor); "
        "skipped");
    return;
  }
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const bool multiblock_topology = state.mesh.topo.multiblock.has_value();
  const std::size_t expected_nodes =
      nr >= 0 && nz >= 0
          ? static_cast<std::size_t>(nr + 1) * static_cast<std::size_t>(nz + 1)
          : 0U;
  const bool fields_available =
      !multiblock_topology && state.mesh.dim == 2 && nr > 0 && nz > 0 &&
      state.x_r.size() == expected_nodes &&
      state.x_z.size() == expected_nodes &&
      state.x_r_initial.size() == expected_nodes &&
      state.x_z_initial.size() == expected_nodes;

  double local_min_corner = std::numeric_limits<double>::infinity();
  double local_min_gauss = std::numeric_limits<double>::infinity();
  double local_min_volume = std::numeric_limits<double>::infinity();
  double local_min_edge_length = std::numeric_limits<double>::infinity();
  double local_min_altitude = std::numeric_limits<double>::infinity();
  double local_max_condition = -std::numeric_limits<double>::infinity();
  bool corner_observed = false;
  bool gauss_observed = false;
  bool volume_observed = false;
  bool edge_length_observed = false;
  bool altitude_observed = false;
  bool condition_observed = false;
  bool local_sampled = false;
  std::uint64_t local_negative_volume_count = 0;

  if (fields_available) {
    std::vector<double> r_current;
    std::vector<double> z_current;
    std::vector<double> r_initial;
    std::vector<double> z_initial;
    state.x_r.copy_to_host(r_current);
    state.x_z.copy_to_host(z_current);
    state.x_r_initial.copy_to_host(r_initial);
    state.x_z_initial.copy_to_host(z_initial);

    local_sampled = true;
    const double gauss_xi[4] = {-kGauss, kGauss, kGauss, -kGauss};
    const double gauss_eta[4] = {-kGauss, -kGauss, kGauss, kGauss};
    const int stride = nz + 1;
    for (int i = 0; i < nr; ++i) {
      for (int j = 0; j < nz; ++j) {
        const int nodes[4] = {
            i * stride + j,
            (i + 1) * stride + j,
            (i + 1) * stride + (j + 1),
            i * stride + (j + 1),
        };
        std::array<double, 4> r0{};
        std::array<double, 4> z0{};
        std::array<double, 4> r1{};
        std::array<double, 4> z1{};
        for (int k = 0; k < 4; ++k) {
          const auto n = static_cast<std::size_t>(nodes[k]);
          r0[static_cast<std::size_t>(k)] = r_initial[n];
          z0[static_cast<std::size_t>(k)] = z_initial[n];
          r1[static_cast<std::size_t>(k)] = r_current[n];
          z1[static_cast<std::size_t>(k)] = z_current[n];
        }

        for (int corner = 0; corner < 4; ++corner) {
          const double j0 = corner_jacobian_from_quad(r0, z0, corner);
          const double j1 = corner_jacobian_from_quad(r1, z1, corner);
          accumulate_finite_min(signed_ratio(j1, j0),
                                local_min_corner,
                                corner_observed);
        }

        for (int q = 0; q < 4; ++q) {
          const double j0 =
              gauss_jacobian_from_quad(r0, z0, gauss_xi[q], gauss_eta[q]);
          const double j1 =
              gauss_jacobian_from_quad(r1, z1, gauss_xi[q], gauss_eta[q]);
          accumulate_finite_min(signed_ratio(j1, j0),
                                local_min_gauss,
                                gauss_observed);

          double dr_dxi = 0.0;
          double dz_dxi = 0.0;
          double dr_deta = 0.0;
          double dz_deta = 0.0;
          gauss_jacobian_matrix_from_quad(r1,
                                          z1,
                                          gauss_xi[q],
                                          gauss_eta[q],
                                          dr_dxi,
                                          dz_dxi,
                                          dr_deta,
                                          dz_deta);
          accumulate_non_nan_max(
              condition_number_2x2(dr_dxi, dr_deta, dz_dxi, dz_deta),
              local_max_condition,
              condition_observed);
        }

        const double v0 = rz_quad_volume_exact(r0, z0);
        const double v1 = rz_quad_volume_exact(r1, z1);
        accumulate_finite_min(signed_ratio(v1, v0),
                              local_min_volume,
                              volume_observed);
        accumulate_finite_min(
            signed_ratio(min_edge_length_from_quad(r1, z1),
                         min_edge_length_from_quad(r0, z0)),
            local_min_edge_length,
            edge_length_observed);
        accumulate_finite_min(
            signed_ratio(min_altitude_from_quad(r1, z1),
                         min_altitude_from_quad(r0, z0)),
            local_min_altitude,
            altitude_observed);
        if (!std::isfinite(v1) || v1 <= 0.0) {
          ++local_negative_volume_count;
        }
      }
    }
  } else if (multiblock_topology) {
    const int n_cells_total = state.mesh.topo.n_cells;
    const std::size_t expected_multiblock_nodes =
        state.mesh.topo.n_nodes > 0
            ? static_cast<std::size_t>(state.mesh.topo.n_nodes)
            : 0U;
    const auto& multiblock = *state.mesh.topo.multiblock;
    const bool multiblock_fields_available =
        state.mesh.dim == 2 && n_cells_total > 0 &&
        state.x_r.size() == expected_multiblock_nodes &&
        state.x_z.size() == expected_multiblock_nodes &&
        state.x_r_initial.size() == expected_multiblock_nodes &&
        state.x_z_initial.size() == expected_multiblock_nodes &&
        multiblock.cell_orientation_sign.size() ==
            static_cast<std::size_t>(n_cells_total);
    if (multiblock_fields_available) {
      std::vector<double> r_current;
      std::vector<double> z_current;
      std::vector<double> r_initial;
      std::vector<double> z_initial;
      state.x_r.copy_to_host(r_current);
      state.x_z.copy_to_host(z_current);
      state.x_r_initial.copy_to_host(r_initial);
      state.x_z_initial.copy_to_host(z_initial);

      local_sampled = true;
      const double gauss_xi[4] = {-kGauss, kGauss, kGauss, -kGauss};
      const double gauss_eta[4] = {-kGauss, -kGauss, kGauss, kGauss};
      tenryu::core::Config::MeshConfig multiblock_mesh_cfg;
      multiblock_mesh_cfg.topology_scheme =
          tenryu::core::TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL;
      for (int c = 0; c < n_cells_total; ++c) {
        const auto nodes = tenryu::mesh::mesh_topo_cell_corner_nodes(
            state.mesh.topo, c, multiblock_mesh_cfg);
        const double orientation_sign = static_cast<double>(
            multiblock.cell_orientation_sign[static_cast<std::size_t>(c)]);
        std::array<double, 4> r0{};
        std::array<double, 4> z0{};
        std::array<double, 4> r1{};
        std::array<double, 4> z1{};
        for (int k = 0; k < 4; ++k) {
          const auto n =
              static_cast<std::size_t>(nodes[static_cast<std::size_t>(k)]);
          r0[static_cast<std::size_t>(k)] = r_initial[n];
          z0[static_cast<std::size_t>(k)] = z_initial[n];
          r1[static_cast<std::size_t>(k)] = r_current[n];
          z1[static_cast<std::size_t>(k)] = z_current[n];
        }

        for (int corner = 0; corner < 4; ++corner) {
          const double j0 =
              orientation_sign * corner_jacobian_from_quad(r0, z0, corner);
          const double j1 =
              orientation_sign * corner_jacobian_from_quad(r1, z1, corner);
          accumulate_finite_min(signed_ratio(j1, j0),
                                local_min_corner,
                                corner_observed);
        }

        for (int q = 0; q < 4; ++q) {
          const double j0 =
              orientation_sign *
              gauss_jacobian_from_quad(r0, z0, gauss_xi[q], gauss_eta[q]);
          const double j1 =
              orientation_sign *
              gauss_jacobian_from_quad(r1, z1, gauss_xi[q], gauss_eta[q]);
          accumulate_finite_min(signed_ratio(j1, j0),
                                local_min_gauss,
                                gauss_observed);

          double dr_dxi = 0.0;
          double dz_dxi = 0.0;
          double dr_deta = 0.0;
          double dz_deta = 0.0;
          gauss_jacobian_matrix_from_quad(r1,
                                          z1,
                                          gauss_xi[q],
                                          gauss_eta[q],
                                          dr_dxi,
                                          dz_dxi,
                                          dr_deta,
                                          dz_deta);
          accumulate_non_nan_max(
              condition_number_2x2(dr_dxi, dr_deta, dz_dxi, dz_deta),
              local_max_condition,
              condition_observed);
        }

        const double v0 = orientation_sign * rz_quad_volume_exact(r0, z0);
        const double v1 = orientation_sign * rz_quad_volume_exact(r1, z1);
        accumulate_finite_min(signed_ratio(v1, v0),
                              local_min_volume,
                              volume_observed);
        accumulate_finite_min(
            signed_ratio(min_edge_length_from_quad(r1, z1),
                         min_edge_length_from_quad(r0, z0)),
            local_min_edge_length,
            edge_length_observed);
        accumulate_finite_min(
            signed_ratio(min_altitude_from_quad(r1, z1),
                         min_altitude_from_quad(r0, z0)),
            local_min_altitude,
            altitude_observed);
        if (!std::isfinite(v1) || v1 <= 0.0) {
          ++local_negative_volume_count;
        }
      }
    }
  }

  if (local_sampled && !corner_observed) {
    local_min_corner = -std::numeric_limits<double>::infinity();
  }
  if (local_sampled && !gauss_observed) {
    local_min_gauss = -std::numeric_limits<double>::infinity();
  }
  if (local_sampled && !volume_observed) {
    local_min_volume = -std::numeric_limits<double>::infinity();
  }
  if (local_sampled && !edge_length_observed) {
    local_min_edge_length = -std::numeric_limits<double>::infinity();
  }
  if (local_sampled && !altitude_observed) {
    local_min_altitude = -std::numeric_limits<double>::infinity();
  }
  if (local_sampled && !condition_observed) {
    local_max_condition = std::numeric_limits<double>::infinity();
  }

  double global_min_corner = local_min_corner;
  double global_min_gauss = local_min_gauss;
  double global_min_volume = local_min_volume;
  double global_min_edge_length = local_min_edge_length;
  double global_min_altitude = local_min_altitude;
  double global_max_condition = local_max_condition;
  double global_negative_volume_count =
      static_cast<double>(local_negative_volume_count);
  double global_sampled = local_sampled ? 1.0 : 0.0;
  if (reduction != nullptr) {
    global_min_corner = reduction->allreduce_min(global_min_corner);
    global_min_gauss = reduction->allreduce_min(global_min_gauss);
    global_min_volume = reduction->allreduce_min(global_min_volume);
    global_min_edge_length = reduction->allreduce_min(global_min_edge_length);
    global_min_altitude = reduction->allreduce_min(global_min_altitude);
    global_max_condition = reduction->allreduce_max(global_max_condition);
    global_negative_volume_count =
        reduction->allreduce_sum(global_negative_volume_count);
    global_sampled = reduction->allreduce_sum(global_sampled);
  }
  if (!(global_sampled > 0.0)) {
    return;
  }

  mesh_quality_min_observed = true;
  achieved_min_corner_j_rel =
      std::min(achieved_min_corner_j_rel, global_min_corner);
  achieved_min_gauss_j_rel =
      std::min(achieved_min_gauss_j_rel, global_min_gauss);
  achieved_min_rz_volume_rel =
      std::min(achieved_min_rz_volume_rel, global_min_volume);
  achieved_min_edge_length_rel =
      std::min(achieved_min_edge_length_rel, global_min_edge_length);
  achieved_min_altitude_rel =
      std::min(achieved_min_altitude_rel, global_min_altitude);
  achieved_max_condition_number =
      std::max(achieved_max_condition_number, global_max_condition);
  if (global_negative_volume_count > 0.0 &&
      std::isfinite(global_negative_volume_count)) {
    const auto increment =
        static_cast<std::uint64_t>(std::llround(global_negative_volume_count));
    negative_rz_volume_count_total =
        saturating_add_u64(negative_rz_volume_count_total, increment);
  }
}

void ProfileObservability::finalize_at_run_end(
    const tenryu::core::Config& cfg,
    const bool reached_t_end_in) {
  reached_t_end = reached_t_end_in;
  if (!reached_t_end && profile_enabled && class_c_runtime_fires == 0 &&
      escape_valve_activations == 0) {
    ++public_baseline_terminal_failures;
  }
  const AleProvenance final_label = compute_label();
  tenryu::core::log_info(
      std::string("[ale_provenance] final: ") +
      ale_provenance_name(final_label));

  plic_gate_status_recorded = false;
  plic_gate_status.clear();
  if (claim_level == ClaimLevel::ProductionComparable) {
    const std::string status = compute_plic_gate_status(cfg);
    if (cfg.numerics.plic.enabled) {
      plic_gate_status_recorded = true;
      plic_gate_status = status;
    }
    if (status != "disabled" && status != "passed") {
      tenryu::core::log_warning(
          "[plic_gate] downgrading claim_level from production_comparable to "
          "pre_plic_smoke: " +
          status);
      claim_level = ClaimLevel::PrePlicSmoke;
    }
  }
}

AleProvenance ProfileObservability::compute_label() const {
  if (!profile_enabled) {
    return AleProvenance::LegacyDefaultUnclassified;
  }
  if (emergency_cell_deactivation_fired) {
    return AleProvenance::TenryuExtendedAleWithCellDeactivation;
  }
  if (class_c_runtime_fires > 0 || escape_valve_activations > 0 ||
      forbidden_config_violations > 0) {
    return AleProvenance::TenryuExtendedAle;
  }
  if (!reached_t_end &&
      (mesh_geometry_failures_observed > 0 ||
       public_baseline_terminal_failures > 0)) {
    return AleProvenance::PublicBaselineFailedNoEscape;
  }
  return AleProvenance::PublicBaseline;
}

std::string ProfileObservability::compute_plic_gate_status(
    const tenryu::core::Config& cfg) const {
  if (!cfg.numerics.plic.enabled) {
    return std::string("disabled");
  }

  if (!reached_t_end) {
    return std::string("failed_did_not_reach_t_end");
  }

  const auto provenance = compute_label();
  if (provenance != AleProvenance::PublicBaseline) {
    return std::string("failed_provenance_not_public_baseline");
  }

  const std::string final_class_d_aggregate =
      compute_final_class_d_aggregate();
  if (final_class_d_aggregate != "none" &&
      final_class_d_aggregate != "soft_only") {
    return std::string("failed_class_d_aggregate_") +
           final_class_d_aggregate;
  }

  const int denom =
      std::max(interface_reconstruction_attempt_count -
                   axis_exempt_cells_count,
               1);
  const double success_rate =
      static_cast<double>(interface_reconstruction_success_count) /
      static_cast<double>(denom);
  if (success_rate < 0.999) {
    return std::string("failed_success_rate_below_0.999");
  }

  return std::string("passed");
}

ProductionComparableWaveFCriteria
ProfileObservability::compute_production_comparable_wave_f_criteria(
    const tenryu::core::Config& cfg,
    const double per_material_conservation_max_rel_residual) const {
  ProductionComparableWaveFCriteria criteria{};
  criteria.plic_enabled = cfg.numerics.plic.enabled;
  criteria.wave_f_enabled =
      cfg.numerics.materials.per_material_conservation_enabled;
  criteria.reached_t_end = reached_t_end;
  criteria.public_baseline_provenance =
      compute_label() == AleProvenance::PublicBaseline;
  const std::string final_class_d_aggregate =
      compute_final_class_d_aggregate();
  criteria.class_d_aggregate_ok =
      final_class_d_aggregate == "none" ||
      final_class_d_aggregate == "soft_only";
  const int denom =
      std::max(interface_reconstruction_attempt_count -
                   axis_exempt_cells_count,
               1);
  criteria.plic_success_rate =
      static_cast<double>(interface_reconstruction_success_count) /
      static_cast<double>(denom);
  criteria.plic_success_rate_ok = criteria.plic_success_rate >= 0.999;
  criteria.per_material_conservation_max_rel_residual =
      per_material_conservation_max_rel_residual;
  criteria.per_material_conservation_residual_ok =
      per_material_conservation_max_rel_residual <=
      cfg.numerics.materials.conservation_residual_hard_warning_threshold_rel;
  return criteria;
}

ProductionComparableWaveFStatus
ProfileObservability::compute_production_comparable_wave_f_status(
    const tenryu::core::Config& cfg,
    const double per_material_conservation_max_rel_residual,
    const double convergence_ratio,
    const double epsilon_E,
    const bool epsilon_E_sustained) const {
  const auto criteria = compute_production_comparable_wave_f_criteria(
      cfg, per_material_conservation_max_rel_residual);
  if (!criteria.wave_f_enabled) {
    return ProductionComparableWaveFStatus::DISABLED;
  }
  if (!criteria.reached_t_end) {
    return ProductionComparableWaveFStatus::FAIL;
  }
  const bool all_criteria_met =
      criteria.plic_enabled && criteria.wave_f_enabled &&
      criteria.reached_t_end && criteria.public_baseline_provenance &&
      criteria.class_d_aggregate_ok && criteria.plic_success_rate_ok &&
      criteria.per_material_conservation_residual_ok;
  if (all_criteria_met) {
    return ProductionComparableWaveFStatus::PASS;
  }
  if (convergence_ratio > 30.0) {
    return ProductionComparableWaveFStatus::PARTIAL_A_CR_PROGRESSION;
  }
  if (convergence_ratio < 30.0 && epsilon_E < 0.10 && epsilon_E_sustained) {
    return ProductionComparableWaveFStatus::PARTIAL_B_PRODUCTION_RESIDUAL;
  }
  return ProductionComparableWaveFStatus::INCONCLUSIVE;
}

std::string ProfileObservability::compute_final_class_d_aggregate() const {
  bool soft_present = false;
  bool hard_present = false;
  bool dense_present = false;
  for (const auto& row : class_d_runtime_fires_matrix) {
    soft_present = soft_present || row[0] > 0;
    hard_present = hard_present || row[1] > 0;
    dense_present = dense_present || row[2] > 0;
  }
  if (dense_present) {
    return std::string("dense_present");
  }
  if (hard_present) {
    return std::string("hard_present");
  }
  if (soft_present) {
    return std::string("soft_only");
  }
  return std::string("none");
}

void ProfileObservability::note_plic_attempt_success(const int cell_idx,
                                                     const double eta_E) {
  (void)cell_idx;
  ++interface_reconstruction_attempt_count;
  ++interface_reconstruction_success_count;
  if (eta_E > plic_max_eta_E_observed) {
    plic_max_eta_E_observed = eta_E;
  }
}

void ProfileObservability::note_plic_attempt_failure(
    const int cell_idx,
    const std::string& fallback_used) {
  (void)cell_idx;
  (void)fallback_used;
  ++interface_reconstruction_attempt_count;
}

void ProfileObservability::note_axis_exempt_cell(const int cell_idx) {
  (void)cell_idx;
  ++axis_exempt_cells_count;
}

void ProfileObservability::note_plic_degradation(
    const PlicDegradationEvent& evt) {
  if (evt.case_id < 1 || evt.case_id > 3) {
    return;
  }
  if (evt.severity > 2) {
    return;
  }

  ++class_d_runtime_fires_matrix[evt.case_id - 1][evt.severity];
  plic_events.push_back(evt);
  tenryu::core::log_warning(
      "[plic_degradation] case=" +
      std::to_string(static_cast<int>(evt.case_id)) + " severity=" +
      std::to_string(static_cast<int>(evt.severity)) + " cell=" +
      std::to_string(evt.cell_idx) + " fallback=" + evt.fallback_used);
}

void ProfileObservability::note_plic_remap_fallback(const std::string& reason,
                                                    const int step,
                                                    const double time) {
  plic_remap_fallback_engaged = true;
  PlicDegradationEvent evt;
  evt.case_id = 3;
  evt.severity = 1;
  evt.step = step;
  evt.time = time;
  evt.fallback_used = reason;
  note_plic_degradation(evt);
  tenryu::core::log_warning("[plic_degradation] PLIC remap scalar fallback: " +
                            reason);
}

const char* ale_provenance_name(const AleProvenance label) {
  switch (label) {
    case AleProvenance::LegacyDefaultUnclassified:
      return "LEGACY_DEFAULT_UNCLASSIFIED";
    case AleProvenance::PublicBaseline:
      return "PUBLIC_BASELINE";
    case AleProvenance::PublicBaselineFailedNoEscape:
      return "PUBLIC_BASELINE_FAILED_NO_ESCAPE";
    case AleProvenance::TenryuExtendedAle:
      return "TENRYU_EXTENDED_ALE";
    case AleProvenance::TenryuExtendedAleWithCellDeactivation:
      return "TENRYU_EXTENDED_ALE_WITH_CELL_DEACTIVATION";
  }
  return "UNKNOWN";
}

const char* claim_level_name(const ClaimLevel level) {
  switch (level) {
    case ClaimLevel::Characterization:
      return "characterization";
    case ClaimLevel::PrePlicSmoke:
      return "pre_plic_smoke";
    case ClaimLevel::ProductionComparable:
      return "production_comparable";
  }
  return "unknown";
}

const char* production_comparable_wave_f_status_name(
    const ProductionComparableWaveFStatus status) {
  switch (status) {
    case ProductionComparableWaveFStatus::PASS:
      return "PASS";
    case ProductionComparableWaveFStatus::PARTIAL_A_CR_PROGRESSION:
      return "PARTIAL_A_CR_PROGRESSION";
    case ProductionComparableWaveFStatus::PARTIAL_B_PRODUCTION_RESIDUAL:
      return "PARTIAL_B_PRODUCTION_RESIDUAL";
    case ProductionComparableWaveFStatus::INCONCLUSIVE:
      return "INCONCLUSIVE";
    case ProductionComparableWaveFStatus::FAIL:
      return "FAIL";
    case ProductionComparableWaveFStatus::DISABLED:
      return "DISABLED";
  }
  return "UNKNOWN";
}

void log_profile_observability(const ProfileObservability& obs,
                               const char* phase_tag) {
  std::ostringstream oss;
  oss << "[ale_provenance] phase="
      << (phase_tag != nullptr ? phase_tag : "unknown")
      << " provenance=" << ale_provenance_name(obs.compute_label())
      << " claim_level=" << claim_level_name(obs.claim_level)
      << " profile_enabled=" << (obs.profile_enabled ? "true" : "false")
      << " forbidden_config_violations="
      << obs.forbidden_config_violations
      << " escape_valve_activations=" << obs.escape_valve_activations
      << " class_c_runtime_fires=" << obs.class_c_runtime_fires
      << " mesh_geometry_failures_observed="
      << obs.mesh_geometry_failures_observed
      << " public_baseline_terminal_failures="
      << obs.public_baseline_terminal_failures
      << " emergency_cell_deactivation_fired="
      << (obs.emergency_cell_deactivation_fired ? "true" : "false")
      << " plic_remap_fallback_engaged="
      << (obs.plic_remap_fallback_engaged ? "true" : "false")
      << " last_failing_cell=" << obs.last_failing_cell
      << " last_failing_i=" << obs.last_failing_i
      << " last_failing_j=" << obs.last_failing_j
      << " last_min_cell_vol=" << obs.last_min_cell_vol
      << " last_min_corner_j=" << obs.last_min_corner_j
      << " last_failure_kind="
      << tenryu::mesh::mesh_geometry_failure_kind_name(obs.last_failure_kind)
      << " reached_t_end=" << (obs.reached_t_end ? "true" : "false");
  tenryu::core::log_info(oss.str());
}

}  // namespace tenryu::coupling
