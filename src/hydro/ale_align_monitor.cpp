#include "hydro/ale_align_monitor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::ale_align {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kWlsDetFloor = 1.0e-30;
constexpr double kSmoothingBlend = 0.25;
// PROVISIONAL(Stage 0): §5.9 does not provide s_{phi,0}, epsilon_g,
// epsilon_Q, or epsilon_h. Keep these diagnostics-local until a later stage
// promotes them to configured monitor parameters.
constexpr double kStrengthScaleRho = 1.0;
constexpr double kStrengthScalePressure = 1.0;
constexpr double kGradientEpsilon = 1.0e-30;
constexpr double kCoherenceEpsilon = 1.0e-30;
constexpr double kWlsDistanceEpsilonRelative = 1.0e-30;

std::size_t cell_index(const int i, const int j, const int nz) {
  return static_cast<std::size_t>(i * nz + j);
}

std::size_t node_index(const int i, const int j, const int nz) {
  return static_cast<std::size_t>(i * (nz + 1) + j);
}

double clamp01(const double value) {
  return std::max(0.0, std::min(1.0, value));
}

double quad_area(const std::array<double, 4>& r,
                 const std::array<double, 4>& z) {
  double twice_area = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int kp = (k + 1) % 4;
    twice_area += r[static_cast<std::size_t>(k)] *
                      z[static_cast<std::size_t>(kp)] -
                  r[static_cast<std::size_t>(kp)] *
                      z[static_cast<std::size_t>(k)];
  }
  return 0.5 * std::abs(twice_area);
}

double nearest_rank(std::vector<double> values, const double probability) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double rank_real =
      std::ceil(probability * static_cast<double>(values.size()));
  const std::size_t rank = static_cast<std::size_t>(
      std::max(1.0, std::min(rank_real, static_cast<double>(values.size()))));
  return values[rank - 1U];
}

RegionSummary* select_region(Summary& summary, const RegionClass region) {
  if (region == RegionClass::NearInterface) {
    return &summary.near_interface;
  }
  if (region == RegionClass::VacuumMask) {
    return &summary.vacuum_mask;
  }
  return &summary.bulk;
}

void finalize_region(RegionSummary& summary,
                     const std::vector<double>& angles,
                     const std::vector<double>& errors) {
  summary.angle_p50_degrees = nearest_rank(angles, 0.50);
  summary.angle_p90_degrees = nearest_rank(angles, 0.90);
  summary.angle_max_degrees = nearest_rank(angles, 1.00);
  summary.e_A_p50 = nearest_rank(errors, 0.50);
  summary.e_A_p90 = nearest_rank(errors, 0.90);
  summary.e_A_max = nearest_rank(errors, 1.00);
}

bool finite_nonnegative_field(const std::vector<double>& field) {
  for (const double value : field) {
    if (!std::isfinite(value) || value < 0.0) {
      return false;
    }
  }
  return true;
}

Tensor2 add_scaled(const Tensor2& a, const Tensor2& b, const double scale) {
  return Tensor2{a.rr + scale * b.rr,
                 a.rz + scale * b.rz,
                 a.zz + scale * b.zz};
}

Tensor2 scaled(const Tensor2& a, const double scale) {
  return Tensor2{scale * a.rr, scale * a.rz, scale * a.zz};
}

bool unit_i_face_normal(const Input& input,
                        const int i,
                        const int j,
                        double& normal_r,
                        double& normal_z) {
  const std::size_t n00 = node_index(i, j, input.nz);
  const std::size_t n01 = node_index(i, j + 1, input.nz);
  const std::size_t n10 = node_index(i + 1, j, input.nz);
  const std::size_t n11 = node_index(i + 1, j + 1, input.nz);
  const double left_t_r = input.node_r[n01] - input.node_r[n00];
  const double left_t_z = input.node_z[n01] - input.node_z[n00];
  const double right_t_r = input.node_r[n11] - input.node_r[n10];
  const double right_t_z = input.node_z[n11] - input.node_z[n10];
  const double left_norm = std::hypot(left_t_z, left_t_r);
  const double right_norm = std::hypot(right_t_z, right_t_r);
  if (!(left_norm > 0.0) || !(right_norm > 0.0)) {
    return false;
  }
  const double left_n_r = left_t_z / left_norm;
  const double left_n_z = -left_t_r / left_norm;
  const double right_n_r = right_t_z / right_norm;
  const double right_n_z = -right_t_r / right_norm;
  const double sum_r = left_n_r + right_n_r;
  const double sum_z = left_n_z + right_n_z;
  const double sum_norm = std::hypot(sum_r, sum_z);
  if (!(sum_norm > 0.0)) {
    return false;
  }
  normal_r = sum_r / sum_norm;
  normal_z = sum_z / sum_norm;
  return true;
}

double topology_weight(const Input& input, const int i) {
  if (input.logical_mesh != LogicalMeshKind::PolarInBox) {
    return 1.0;
  }
  if (i < input.polar_prefix_nr) {
    return 1.0;
  }
  if (input.morph_rings <= 0 ||
      i >= input.polar_prefix_nr + input.morph_rings) {
    return 0.0;
  }
  // PROVISIONAL(Stage 0): Stage 4 will replace this index-only topology
  // approximation with mesh-generation metadata.
  return clamp01(1.0 -
                 (static_cast<double>(i - input.polar_prefix_nr) + 0.5) /
                     static_cast<double>(input.morph_rings));
}

void eigen_decompose(const Tensor2& Q,
                     double& lambda_1,
                     double& lambda_2,
                     double& coherence,
                     double& director_r,
                     double& director_z,
                     bool& director_available) {
  const double split =
      std::hypot(Q.rr - Q.zz, 2.0 * Q.rz);
  const double trace = Q.rr + Q.zz;
  lambda_1 = std::max(0.0, 0.5 * (trace + split));
  lambda_2 = std::max(0.0, 0.5 * (trace - split));
  coherence = (lambda_1 - lambda_2) /
              (lambda_1 + lambda_2 + kCoherenceEpsilon);
  director_available = split > kCoherenceEpsilon;
  if (!director_available) {
    director_r = 0.0;
    director_z = 0.0;
    return;
  }
  double v_r = Q.rz;
  double v_z = lambda_1 - Q.rr;
  if (std::abs(v_r) + std::abs(v_z) < split) {
    v_r = lambda_1 - Q.zz;
    v_z = Q.rz;
  }
  const double norm = std::hypot(v_r, v_z);
  if (!(norm > 0.0)) {
    director_available = false;
    director_r = 0.0;
    director_z = 0.0;
    return;
  }
  director_r = v_r / norm;
  director_z = v_z / norm;
}

bool should_emit(const core::State& state,
                 const core::Config& cfg,
                 const double dt_hydro_used) {
  const auto& diag = cfg.numerics.ale.align_diagnostics;
  if (!diag.enabled) {
    return false;
  }
  if (diag.every_n_steps > 0) {
    return (state.step % diag.every_n_steps) == 0;
  }
  const double t_after = state.t + dt_hydro_used;
  const double t_eps =
      1.0e-12 * std::max(1.0, std::abs(cfg.main.t_end));
  const bool last_by_time =
      cfg.main.t_end > 0.0 && t_after + t_eps >= cfg.main.t_end;
  const bool last_by_step = state.step + 1 >= cfg.main.max_steps;
  return state.step == 0 || last_by_time || last_by_step;
}

}  // namespace

WlsGradientResult compute_one_ring_wls_gradients(
    const int nr,
    const int nz,
    const std::vector<double>& cell_r,
    const std::vector<double>& cell_z,
    const std::vector<double>& cell_length,
    const std::vector<double>& scalar,
    const std::vector<std::uint8_t>& interface_mask) {
  WlsGradientResult result;
  if (nr <= 0 || nz <= 0) {
    return result;
  }
  const std::size_t n_cells = static_cast<std::size_t>(nr * nz);
  result.grad_r.assign(n_cells, 0.0);
  result.grad_z.assign(n_cells, 0.0);
  result.available.assign(n_cells, 0U);
  if (cell_r.size() != n_cells || cell_z.size() != n_cells ||
      cell_length.size() != n_cells || scalar.size() != n_cells ||
      (!interface_mask.empty() && interface_mask.size() != n_cells)) {
    return result;
  }

  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const std::size_t c = cell_index(i, j, nz);
      double B_rr = 0.0;
      double B_rz = 0.0;
      double B_zz = 0.0;
      double b_r = 0.0;
      double b_z = 0.0;
      int neighbor_count = 0;
      for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
          if (di == 0 && dj == 0) {
            continue;
          }
          const int ni = i + di;
          const int nj = j + dj;
          if (ni < 0 || ni >= nr || nj < 0 || nj >= nz) {
            continue;
          }
          const std::size_t d = cell_index(ni, nj, nz);
          if (!interface_mask.empty() &&
              (interface_mask[c] != 0U || interface_mask[d] != 0U)) {
            continue;
          }
          const double dr = cell_r[d] - cell_r[c];
          const double dz = cell_z[d] - cell_z[c];
          const double distance2 = dr * dr + dz * dz;
          const double h2 = cell_length[c] * cell_length[c];
          const double epsilon_h =
              kWlsDistanceEpsilonRelative *
              std::max(h2, std::numeric_limits<double>::min());
          const double weight = 1.0 / (distance2 + epsilon_h);
          const double delta_q = scalar[d] - scalar[c];
          B_rr += weight * dr * dr;
          B_rz += weight * dr * dz;
          B_zz += weight * dz * dz;
          b_r += weight * dr * delta_q;
          b_z += weight * dz * delta_q;
          ++neighbor_count;
        }
      }
      if (neighbor_count < 3) {
        continue;
      }
      const double det = B_rr * B_zz - B_rz * B_rz;
      if (!std::isfinite(det) || std::abs(det) < kWlsDetFloor) {
        continue;
      }
      result.grad_r[c] = (B_zz * b_r - B_rz * b_z) / det;
      result.grad_z[c] = (B_rr * b_z - B_rz * b_r) / det;
      if (std::isfinite(result.grad_r[c]) &&
          std::isfinite(result.grad_z[c])) {
        result.available[c] = 1U;
      } else {
        result.grad_r[c] = 0.0;
        result.grad_z[c] = 0.0;
      }
    }
  }
  return result;
}

Tensor2 build_structure_tensor(const double grad_rho_r,
                               const double grad_rho_z,
                               const double grad_p_r,
                               const double grad_p_z,
                               const double cell_length,
                               const Params& params,
                               double* S_rho,
                               double* S_p) {
  const double rho_norm = std::hypot(grad_rho_r, grad_rho_z);
  const double pressure_norm = std::hypot(grad_p_r, grad_p_z);
  // PROVISIONAL(Stage 0): h_cell=sqrt(general-quad area) represents the
  // Jacobian-derived reference length requested by consult §5.3.
  const double s_rho = cell_length * rho_norm;
  const double s_pressure = cell_length * pressure_norm;
  const double strength_rho =
      (s_rho * s_rho) /
      (s_rho * s_rho + kStrengthScaleRho * kStrengthScaleRho);
  const double strength_pressure =
      (s_pressure * s_pressure) /
      (s_pressure * s_pressure +
       kStrengthScalePressure * kStrengthScalePressure);
  if (S_rho != nullptr) {
    *S_rho = strength_rho;
  }
  if (S_p != nullptr) {
    *S_p = strength_pressure;
  }
  const double rho_denom = rho_norm + kGradientEpsilon;
  const double pressure_denom = pressure_norm + kGradientEpsilon;
  const double rho_r = grad_rho_r / rho_denom;
  const double rho_z = grad_rho_z / rho_denom;
  const double pressure_r = grad_p_r / pressure_denom;
  const double pressure_z = grad_p_z / pressure_denom;
  return Tensor2{
      params.w_rho * strength_rho * rho_r * rho_r +
          params.w_p * strength_pressure * pressure_r * pressure_r,
      params.w_rho * strength_rho * rho_r * rho_z +
          params.w_p * strength_pressure * pressure_r * pressure_z,
      params.w_rho * strength_rho * rho_z * rho_z +
          params.w_p * strength_pressure * pressure_z * pressure_z};
}

Result compute_monitor(const Input& input, const Params& params) {
  Result result;
  if (input.nr <= 0 || input.nz <= 0) {
    result.error = "non-positive structured topology";
    return result;
  }
  const std::size_t n_cells =
      static_cast<std::size_t>(input.nr * input.nz);
  const std::size_t n_nodes =
      static_cast<std::size_t>((input.nr + 1) * (input.nz + 1));
  if (input.rho.size() != n_cells || input.pressure.size() != n_cells) {
    result.error = "cell field size mismatch";
    return result;
  }
  if (input.node_r.size() != n_nodes || input.node_z.size() != n_nodes) {
    result.error = "node coordinate size mismatch";
    return result;
  }
  if (!input.vacuum_mask.empty() && input.vacuum_mask.size() != n_cells) {
    result.error = "vacuum mask size mismatch";
    return result;
  }
  if (input.n_materials < 0 ||
      (input.n_materials == 0 && !input.vol_frac.empty()) ||
      (input.n_materials > 0 &&
       input.vol_frac.size() !=
           n_cells * static_cast<std::size_t>(input.n_materials))) {
    result.error = "material volume-fraction size mismatch";
    return result;
  }
  if (!finite_nonnegative_field(input.rho) ||
      !finite_nonnegative_field(input.pressure)) {
    result.error = "rho and alignment pressure must be finite non-negative";
    return result;
  }
  if (!(params.floor_rel > 0.0) || !std::isfinite(params.floor_rel) ||
      !std::isfinite(params.c_q_threshold) ||
      !std::isfinite(params.w_rho) || !std::isfinite(params.w_p)) {
    result.error = "non-finite or non-positive monitor parameter";
    return result;
  }

  double rho_max = 0.0;
  double pressure_max = 0.0;
  for (std::size_t c = 0; c < n_cells; ++c) {
    rho_max = std::max(rho_max, input.rho[c]);
    pressure_max = std::max(pressure_max, input.pressure[c]);
  }
  result.rho_floor = params.floor_rel * rho_max;
  result.pressure_floor = params.floor_rel * pressure_max;

  std::vector<double> cell_r(n_cells, 0.0);
  std::vector<double> cell_z(n_cells, 0.0);
  std::vector<double> cell_length(n_cells, 0.0);
  std::vector<double> q_rho(n_cells, 0.0);
  std::vector<double> q_pressure(n_cells, 0.0);
  std::vector<std::uint8_t> interface_mask(n_cells, 0U);
  std::vector<std::uint8_t> near_interface(n_cells, 0U);

  for (int i = 0; i < input.nr; ++i) {
    for (int j = 0; j < input.nz; ++j) {
      const std::size_t c = cell_index(i, j, input.nz);
      const std::size_t n00 = node_index(i, j, input.nz);
      const std::size_t n10 = node_index(i + 1, j, input.nz);
      const std::size_t n11 = node_index(i + 1, j + 1, input.nz);
      const std::size_t n01 = node_index(i, j + 1, input.nz);
      const std::array<double, 4> r = {input.node_r[n00], input.node_r[n10],
                                       input.node_r[n11], input.node_r[n01]};
      const std::array<double, 4> z = {input.node_z[n00], input.node_z[n10],
                                       input.node_z[n11], input.node_z[n01]};
      cell_r[c] = 0.25 * (r[0] + r[1] + r[2] + r[3]);
      cell_z[c] = 0.25 * (z[0] + z[1] + z[2] + z[3]);
      cell_length[c] = std::sqrt(quad_area(r, z));
      if (result.rho_floor > 0.0) {
        q_rho[c] = std::log1p(input.rho[c] / result.rho_floor);
      }
      if (result.pressure_floor > 0.0) {
        q_pressure[c] =
            std::log1p(input.pressure[c] / result.pressure_floor);
      }
      for (int m = 0; m < input.n_materials; ++m) {
        const double fraction =
            input.vol_frac[c * static_cast<std::size_t>(input.n_materials) +
                           static_cast<std::size_t>(m)];
        // Stage 0 uses volFrac only; PLIC-derived interface masks are ignored.
        if (fraction > 0.01 && fraction < 0.99) {
          interface_mask[c] = 1U;
          break;
        }
      }
    }
  }

  for (int i = 0; i < input.nr; ++i) {
    for (int j = 0; j < input.nz; ++j) {
      const std::size_t c = cell_index(i, j, input.nz);
      if (interface_mask[c] == 0U) {
        continue;
      }
      for (int di = -2; di <= 2; ++di) {
        for (int dj = -2; dj <= 2; ++dj) {
          const int ni = i + di;
          const int nj = j + dj;
          if (ni >= 0 && ni < input.nr && nj >= 0 && nj < input.nz) {
            near_interface[cell_index(ni, nj, input.nz)] = 1U;
          }
        }
      }
    }
  }

  const WlsGradientResult grad_rho = compute_one_ring_wls_gradients(
      input.nr, input.nz, cell_r, cell_z, cell_length, q_rho, interface_mask);
  const WlsGradientResult grad_pressure = compute_one_ring_wls_gradients(
      input.nr, input.nz, cell_r, cell_z, cell_length, q_pressure,
      interface_mask);

  result.cells.resize(n_cells);
  std::vector<Tensor2> raw_Q(n_cells);
  std::vector<std::uint8_t> tensor_available(n_cells, 0U);
  for (int i = 0; i < input.nr; ++i) {
    for (int j = 0; j < input.nz; ++j) {
      const std::size_t c = cell_index(i, j, input.nz);
      CellDiagnostic& cell = result.cells[c];
      cell.q_rho = q_rho[c];
      cell.q_p = q_pressure[c];
      cell.grad_rho_r = grad_rho.grad_r[c];
      cell.grad_rho_z = grad_rho.grad_z[c];
      cell.grad_p_r = grad_pressure.grad_r[c];
      cell.grad_p_z = grad_pressure.grad_z[c];
      cell.interface_cell = interface_mask[c] != 0U;
      cell.near_interface = near_interface[c] != 0U;
      cell.vacuum_masked =
          !input.vacuum_mask.empty() && input.vacuum_mask[c] != 0U;
      cell.chi_topology = topology_weight(input, i);
      cell.region = cell.vacuum_masked
                        ? RegionClass::VacuumMask
                        : (cell.near_interface ? RegionClass::NearInterface
                                               : RegionClass::Bulk);
      cell.gradient_available = grad_rho.available[c] != 0U &&
                                grad_pressure.available[c] != 0U;
      if (!cell.gradient_available) {
        continue;
      }
      raw_Q[c] = build_structure_tensor(
          cell.grad_rho_r, cell.grad_rho_z, cell.grad_p_r, cell.grad_p_z,
          cell_length[c], params, &cell.S_rho, &cell.S_p);
      tensor_available[c] = 1U;
    }
  }

  // One fixed smoothing pass. PROVISIONAL(Stage 0): §5.5 leaves omega_cd
  // unspecified, so face neighbors use unit weights. Polar and polar_in_box
  // tensors are smoothed in global (r,z) components until Stage 4 supplies an
  // unambiguous local reference basis.
  std::vector<Tensor2> smooth_Q = raw_Q;
  constexpr std::array<std::pair<int, int>, 4> face_neighbors = {
      std::pair<int, int>{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  for (int i = 0; i < input.nr; ++i) {
    for (int j = 0; j < input.nz; ++j) {
      const std::size_t c = cell_index(i, j, input.nz);
      if (tensor_available[c] == 0U) {
        continue;
      }
      Tensor2 neighbor_sum;
      int neighbor_count = 0;
      for (const auto& offset : face_neighbors) {
        const int ni = i + offset.first;
        const int nj = j + offset.second;
        if (ni < 0 || ni >= input.nr || nj < 0 || nj >= input.nz) {
          continue;
        }
        const std::size_t d = cell_index(ni, nj, input.nz);
        if (tensor_available[d] == 0U || interface_mask[c] != 0U ||
            interface_mask[d] != 0U) {
          continue;
        }
        neighbor_sum = add_scaled(neighbor_sum, raw_Q[d], 1.0);
        ++neighbor_count;
      }
      if (neighbor_count > 0) {
        const Tensor2 average =
            scaled(neighbor_sum, 1.0 / static_cast<double>(neighbor_count));
        smooth_Q[c] = add_scaled(scaled(raw_Q[c], 1.0 - kSmoothingBlend),
                                 average, kSmoothingBlend);
      }
    }
  }

  std::array<std::vector<double>, 3> region_angles;
  std::array<std::vector<double>, 3> region_errors;
  result.summary.total_cells = n_cells;
  for (int i = 0; i < input.nr; ++i) {
    for (int j = 0; j < input.nz; ++j) {
      const std::size_t c = cell_index(i, j, input.nz);
      CellDiagnostic& cell = result.cells[c];
      RegionSummary* region = select_region(result.summary, cell.region);
      ++region->cell_count;
      if (cell.interface_cell) {
        ++result.summary.interface_masked_cells;
      }
      if (cell.vacuum_masked) {
        ++result.summary.vacuum_masked_cells;
      }
      if (!(cell.chi_topology > 0.0)) {
        ++result.summary.topology_masked_cells;
      }
      if (!cell.gradient_available) {
        ++result.summary.wls_unavailable_cells;
      }
      if (tensor_available[c] != 0U) {
        cell.Q = smooth_Q[c];
        eigen_decompose(cell.Q, cell.lambda_1, cell.lambda_2, cell.coherence,
                        cell.director_r, cell.director_z,
                        cell.director_available);
      }
      double normal_r = 0.0;
      double normal_z = 0.0;
      const bool normal_available =
          unit_i_face_normal(input, i, j, normal_r, normal_z);
      if (cell.director_available && normal_available) {
        const double abs_dot = clamp01(std::abs(
            normal_r * cell.director_r + normal_z * cell.director_z));
        cell.e_A = 1.0 - abs_dot * abs_dot;
        cell.angle_degrees = std::acos(abs_dot) * 180.0 / kPi;
      }
      const bool hard_masked =
          cell.interface_cell || cell.vacuum_masked ||
          !(cell.chi_topology > 0.0) || !cell.gradient_available ||
          !cell.director_available || !normal_available;
      if (hard_masked) {
        ++region->masked_count;
        continue;
      }
      if (cell.coherence < params.c_q_threshold) {
        ++result.summary.coherence_filtered_cells;
        continue;
      }
      cell.summary_included = true;
      ++region->included_count;
      const int bin = std::max(
          0, std::min(kAngleHistogramBins - 1,
                      static_cast<int>(std::floor(cell.angle_degrees))));
      ++region->angle_histogram[static_cast<std::size_t>(bin)];
      const std::size_t region_index =
          static_cast<std::size_t>(cell.region);
      region_angles[region_index].push_back(cell.angle_degrees);
      region_errors[region_index].push_back(cell.e_A);
    }
  }
  finalize_region(result.summary.bulk,
                  region_angles[static_cast<std::size_t>(RegionClass::Bulk)],
                  region_errors[static_cast<std::size_t>(RegionClass::Bulk)]);
  finalize_region(
      result.summary.near_interface,
      region_angles[static_cast<std::size_t>(RegionClass::NearInterface)],
      region_errors[static_cast<std::size_t>(RegionClass::NearInterface)]);
  finalize_region(
      result.summary.vacuum_mask,
      region_angles[static_cast<std::size_t>(RegionClass::VacuumMask)],
      region_errors[static_cast<std::size_t>(RegionClass::VacuumMask)]);
  result.valid = true;
  return result;
}

void maybe_log_post_lagrange(const core::State& state,
                             const core::Config& cfg,
                             const double dt_hydro_used,
                             const int rank) {
  if (rank != 0 || !should_emit(state, cfg, dt_hydro_used)) {
    return;
  }
  if (state.mesh.topo.multiblock.has_value()) {
    static bool multiblock_notice_emitted = false;
    if (!multiblock_notice_emitted) {
      multiblock_notice_emitted = true;
      core::log_info("[ale-align-diag] skipped=multiblock stage0=single-block-only");
    }
    return;
  }

  Input input;
  input.nr = state.mesh.topo.nr;
  input.nz = state.mesh.topo.nz;
  input.polar_prefix_nr = cfg.mesh.polar_prefix_nr;
  input.morph_rings = cfg.mesh.morph_rings;
  switch (state.mesh.logical) {
    case mesh::LogicalMesh2D::RectangularRZ:
      input.logical_mesh = LogicalMeshKind::RectangularRZ;
      break;
    case mesh::LogicalMesh2D::SphericalPolarHalfplane:
      input.logical_mesh = LogicalMeshKind::SphericalPolarHalfplane;
      break;
    case mesh::LogicalMesh2D::PolarInBox:
      input.logical_mesh = LogicalMeshKind::PolarInBox;
      break;
  }
  state.rho.copy_to_host(input.rho);
  std::vector<double> pressure_e;
  std::vector<double> pressure_i;
  state.Pe.copy_to_host(pressure_e);
  state.Pi.copy_to_host(pressure_i);
  if (pressure_e.size() != pressure_i.size()) {
    core::log_info(
        "[ale-align-diag] skipped=invalid-input reason=pressure-size-mismatch");
    return;
  }
  input.pressure.resize(pressure_e.size(), 0.0);
  for (std::size_t c = 0; c < pressure_e.size(); ++c) {
    input.pressure[c] = pressure_e[c] + pressure_i[c];
  }
  state.x_r.copy_to_host(input.node_r);
  state.x_z.copy_to_host(input.node_z);
  if (!state.volFrac.empty()) {
    state.volFrac.copy_to_host(input.vol_frac);
    if (!input.rho.empty() && input.vol_frac.size() % input.rho.size() == 0U) {
      input.n_materials = static_cast<int>(input.vol_frac.size() /
                                           input.rho.size());
    }
  }
  if (state.cell_is_void.size() == input.rho.size()) {
    input.vacuum_mask = state.cell_is_void;
  }

  const auto& diag = cfg.numerics.ale.align_diagnostics;
  Params params;
  params.c_q_threshold = diag.c_q_threshold;
  params.w_rho = diag.w_rho;
  params.w_p = diag.w_p;
  params.floor_rel = diag.floor_rel;
  const Result result = compute_monitor(input, params);
  if (!result.valid) {
    core::log_info("[ale-align-diag] skipped=invalid-input reason=" +
                   result.error);
    return;
  }

  const auto append_region = [](std::ostringstream& oss,
                                const char* name,
                                const RegionSummary& region) {
    oss << " " << name << "_n=" << region.included_count
        << " " << name << "_p50_deg=" << region.angle_p50_degrees
        << " " << name << "_p90_deg=" << region.angle_p90_degrees
        << " " << name << "_max_deg=" << region.angle_max_degrees
        << " " << name << "_eA_p50=" << region.e_A_p50
        << " " << name << "_eA_p90=" << region.e_A_p90
        << " " << name << "_eA_max=" << region.e_A_max
        << " " << name << "_hist=";
    for (std::size_t bin = 0; bin < region.angle_histogram.size(); ++bin) {
      if (bin != 0U) {
        oss << ',';
      }
      oss << region.angle_histogram[bin];
    }
  };
  std::ostringstream oss;
  oss << std::setprecision(6)
      << "[ale-align-diag] step=" << state.step
      << " t=" << (state.t + dt_hydro_used)
      << " c_q_threshold=" << params.c_q_threshold
      << " w_rho=" << params.w_rho
      << " w_p=" << params.w_p
      << " rho_floor=" << result.rho_floor
      << " p_floor=" << result.pressure_floor
      << " cells=" << result.summary.total_cells
      << " interface_masked=" << result.summary.interface_masked_cells
      << " vacuum_masked=" << result.summary.vacuum_masked_cells
      << " topology_masked=" << result.summary.topology_masked_cells
      << " wls_unavailable=" << result.summary.wls_unavailable_cells
      << " coherence_filtered=" << result.summary.coherence_filtered_cells;
  append_region(oss, "bulk", result.summary.bulk);
  append_region(oss, "near_interface", result.summary.near_interface);
  append_region(oss, "vacuum", result.summary.vacuum_mask);
  core::log_info(oss.str());
}

}  // namespace tenryu::hydro::ale_align
