#include "hydro/ale_align_rezone.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace tenryu::hydro::ale_align {
namespace {

constexpr double kGaussOffset = 0.28867513459481288225457439025097873;
constexpr double kTwoPi = 6.28318530717958647692528676655900577;

struct Matrix2 {
  double xx = 1.0;
  double xy = 0.0;
  double yy = 1.0;
};

struct CellMetric {
  Matrix2 A_xi;
  Matrix2 A_eta;
  double preconditioner_weight = 1.0;
};

struct BoundaryLine {
  double origin_r = 0.0;
  double origin_z = 0.0;
  double tangent_r = 0.0;
  double tangent_z = 0.0;
};

std::size_t cell_index(const int i, const int j, const int nz) {
  return static_cast<std::size_t>(i * nz + j);
}

std::size_t node_index(const int i, const int j, const int nz) {
  return static_cast<std::size_t>(i * (nz + 1) + j);
}

double signed_cell_area(const std::vector<double>& r,
                        const std::vector<double>& z,
                        const int i,
                        const int j,
                        const int nz) {
  const std::size_t n00 = node_index(i, j, nz);
  const std::size_t n10 = node_index(i + 1, j, nz);
  const std::size_t n11 = node_index(i + 1, j + 1, nz);
  const std::size_t n01 = node_index(i, j + 1, nz);
  return 0.5 *
         (r[n00] * z[n10] - r[n10] * z[n00] +
          r[n10] * z[n11] - r[n11] * z[n10] +
          r[n11] * z[n01] - r[n01] * z[n11] +
          r[n01] * z[n00] - r[n00] * z[n01]);
}

std::array<double, 4> cell_corner_jacobians(
    const std::vector<double>& r,
    const std::vector<double>& z,
    const int i,
    const int j,
    const int nz) {
  const std::size_t n00 = node_index(i, j, nz);
  const std::size_t n10 = node_index(i + 1, j, nz);
  const std::size_t n11 = node_index(i + 1, j + 1, nz);
  const std::size_t n01 = node_index(i, j + 1, nz);
  const double r00 = r[n00];
  const double r10 = r[n10];
  const double r11 = r[n11];
  const double r01 = r[n01];
  const double z00 = z[n00];
  const double z10 = z[n10];
  const double z11 = z[n11];
  const double z01 = z[n01];
  const auto determinant = [](const double ar,
                              const double az,
                              const double br,
                              const double bz) {
    return ar * bz - az * br;
  };
  return {
      determinant(r10 - r00, z10 - z00, r01 - r00, z01 - z00),
      determinant(r10 - r00, z10 - z00, r11 - r10, z11 - z10),
      determinant(r11 - r01, z11 - z01, r11 - r10, z11 - z10),
      determinant(r11 - r01, z11 - z01, r01 - r00, z01 - z00)};
}

bool finite_coordinates(const std::vector<double>& r,
                        const std::vector<double>& z) {
  for (std::size_t n = 0; n < r.size(); ++n) {
    if (!std::isfinite(r[n]) || !std::isfinite(z[n])) {
      return false;
    }
  }
  return true;
}

bool build_cell_metrics(const RezoneInput& input,
                        const RezoneParams& params,
                        std::vector<CellMetric>& metrics,
                        bool& has_active_cells) {
  const std::size_t n_cells =
      static_cast<std::size_t>(input.nr * input.nz);
  metrics.resize(n_cells);
  has_active_cells = false;
  for (std::size_t c = 0; c < n_cells; ++c) {
    const CellDiagnostic& monitor = input.monitor[c];
    if (!std::isfinite(monitor.coherence) ||
        !std::isfinite(monitor.director_r) ||
        !std::isfinite(monitor.director_z)) {
      return false;
    }
    const double director_norm =
        std::hypot(monitor.director_r, monitor.director_z);
    const bool active = monitor.director_available &&
                        monitor.coherence > 0.0 && director_norm > 0.0;
    const double coherence =
        active ? std::clamp(monitor.coherence, 0.0, 1.0) : 0.0;
    const double n_r = active ? monitor.director_r / director_norm : 1.0;
    const double n_z = active ? monitor.director_z / director_norm : 0.0;
    const double p_rr = n_r * n_r;
    const double p_rz = n_r * n_z;
    const double p_zz = n_z * n_z;
    const double kappa_xi = 1.0 + params.kappa_gain * coherence;
    const double kappa_eta =
        1.0 + 0.25 * params.kappa_gain * coherence;
    const double sqrt_xi = std::sqrt(kappa_xi);
    const double inv_sqrt_xi = 1.0 / sqrt_xi;
    const double sqrt_eta = std::sqrt(kappa_eta);
    const double inv_sqrt_eta = 1.0 / sqrt_eta;
    CellMetric& metric = metrics[c];
    metric.A_xi = {
        inv_sqrt_xi * p_rr + sqrt_xi * (1.0 - p_rr),
        (inv_sqrt_xi - sqrt_xi) * p_rz,
        inv_sqrt_xi * p_zz + sqrt_xi * (1.0 - p_zz)};
    metric.A_eta = {
        inv_sqrt_eta * (1.0 - p_rr) + sqrt_eta * p_rr,
        (sqrt_eta - inv_sqrt_eta) * p_rz,
        inv_sqrt_eta * (1.0 - p_zz) + sqrt_eta * p_zz};
    metric.preconditioner_weight = 1.0 + params.kappa_gain * coherence;
    has_active_cells = has_active_cells || active;
  }
  return true;
}

double quadratic_form(const Matrix2& matrix,
                      const double x,
                      const double y) {
  return matrix.xx * x * x + 2.0 * matrix.xy * x * y +
         matrix.yy * y * y;
}

double cell_energy(const std::vector<double>& r,
                   const std::vector<double>& z,
                   const int i,
                   const int j,
                   const int nz,
                   const CellMetric& metric,
                   const double omega_eta,
                   const RezoneGeometry geometry) {
  const std::size_t n00 = node_index(i, j, nz);
  const std::size_t n10 = node_index(i + 1, j, nz);
  const std::size_t n11 = node_index(i + 1, j + 1, nz);
  const std::size_t n01 = node_index(i, j + 1, nz);
  const std::array<double, 4> cell_r = {r[n00], r[n10], r[n11], r[n01]};
  const std::array<double, 4> cell_z = {z[n00], z[n10], z[n11], z[n01]};
  constexpr std::array<double, 2> points = {0.5 - kGaussOffset,
                                             0.5 + kGaussOffset};
  double energy = 0.0;
  for (const double xi : points) {
    for (const double eta : points) {
      const std::array<double, 4> dN_dxi = {
          -(1.0 - eta), 1.0 - eta, eta, -eta};
      const std::array<double, 4> dN_deta = {
          -(1.0 - xi), -xi, xi, 1.0 - xi};
      double r_xi = 0.0;
      double z_xi = 0.0;
      double r_eta = 0.0;
      double z_eta = 0.0;
      for (std::size_t a = 0; a < 4; ++a) {
        r_xi += cell_r[a] * dN_dxi[a];
        z_xi += cell_z[a] * dN_dxi[a];
        r_eta += cell_r[a] * dN_deta[a];
        z_eta += cell_z[a] * dN_deta[a];
      }
      const double J = r_xi * z_eta - r_eta * z_xi;
      if (!(J > 0.0) || !std::isfinite(J)) {
        return std::numeric_limits<double>::infinity();
      }
      const double grad_xi_r = z_eta / J;
      const double grad_xi_z = -r_eta / J;
      const double grad_eta_r = -z_xi / J;
      const double grad_eta_z = r_xi / J;
      const double density =
          quadratic_form(metric.A_xi, grad_xi_r, grad_xi_z) +
          omega_eta *
              quadratic_form(metric.A_eta, grad_eta_r, grad_eta_z);
      if (geometry == RezoneGeometry::Planar) {
        energy += 0.125 * density * J;
      } else {
        const std::array<double, 4> N = {
            (1.0 - xi) * (1.0 - eta), xi * (1.0 - eta),
            xi * eta, (1.0 - xi) * eta};
        double quadrature_r = 0.0;
        for (std::size_t a = 0; a < 4; ++a) {
          quadrature_r += cell_r[a] * N[a];
        }
        energy += 0.125 * kTwoPi * quadrature_r * density * J;
      }
    }
  }
  return energy;
}

double total_energy(const std::vector<double>& r,
                    const std::vector<double>& z,
                    const int nr,
                    const int nz,
                    const std::vector<CellMetric>& metrics,
                    const double omega_eta,
                    const RezoneGeometry geometry) {
  double energy = 0.0;
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      energy += cell_energy(r, z, i, j, nz,
                            metrics[cell_index(i, j, nz)], omega_eta, geometry);
    }
  }
  return energy;
}

double adjacent_patch_energy(const std::vector<double>& r,
                             const std::vector<double>& z,
                             const int node_i,
                             const int node_j,
                             const int nr,
                             const int nz,
                             const std::vector<CellMetric>& metrics,
                             const double omega_eta,
                             const RezoneGeometry geometry) {
  double energy = 0.0;
  for (int di = -1; di <= 0; ++di) {
    const int i = node_i + di;
    if (i < 0 || i >= nr) {
      continue;
    }
    for (int dj = -1; dj <= 0; ++dj) {
      const int j = node_j + dj;
      if (j < 0 || j >= nz) {
        continue;
      }
      energy += cell_energy(r, z, i, j, nz,
                            metrics[cell_index(i, j, nz)], omega_eta, geometry);
    }
  }
  return energy;
}

double signed_cell_rz_volume(const std::vector<double>& r,
                             const std::vector<double>& z,
                             const int i,
                             const int j,
                             const int nz) {
  // Symmetric Q1 quadrature exactly integrates 2*pi*integral(r dA). This is
  // the diagonal-independent cell sum of the signed triangle volumes in 4.6.
  const std::size_t n00 = node_index(i, j, nz);
  const std::size_t n10 = node_index(i + 1, j, nz);
  const std::size_t n11 = node_index(i + 1, j + 1, nz);
  const std::size_t n01 = node_index(i, j + 1, nz);
  const std::array<double, 4> cell_r = {r[n00], r[n10], r[n11], r[n01]};
  const std::array<double, 4> cell_z = {z[n00], z[n10], z[n11], z[n01]};
  constexpr std::array<double, 2> points = {0.5 - kGaussOffset,
                                             0.5 + kGaussOffset};
  double volume = 0.0;
  for (const double xi : points) {
    for (const double eta : points) {
      const std::array<double, 4> N = {
          (1.0 - xi) * (1.0 - eta), xi * (1.0 - eta),
          xi * eta, (1.0 - xi) * eta};
      const std::array<double, 4> dN_dxi = {
          -(1.0 - eta), 1.0 - eta, eta, -eta};
      const std::array<double, 4> dN_deta = {
          -(1.0 - xi), -xi, xi, 1.0 - xi};
      double quadrature_r = 0.0;
      double r_xi = 0.0;
      double z_xi = 0.0;
      double r_eta = 0.0;
      double z_eta = 0.0;
      for (std::size_t a = 0; a < 4; ++a) {
        quadrature_r += cell_r[a] * N[a];
        r_xi += cell_r[a] * dN_dxi[a];
        z_xi += cell_z[a] * dN_dxi[a];
        r_eta += cell_r[a] * dN_deta[a];
        z_eta += cell_z[a] * dN_deta[a];
      }
      const double J = r_xi * z_eta - r_eta * z_xi;
      volume += 0.25 * kTwoPi * quadrature_r * J;
    }
  }
  return volume;
}

void compute_cell_rz_volumes(const std::vector<double>& r,
                             const std::vector<double>& z,
                             const int nr,
                             const int nz,
                             std::vector<double>& volumes) {
  volumes.resize(static_cast<std::size_t>(nr * nz));
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      volumes[cell_index(i, j, nz)] =
          signed_cell_rz_volume(r, z, i, j, nz);
    }
  }
}

double minimum_adjacent_area(const std::vector<double>& r,
                             const std::vector<double>& z,
                             const int node_i,
                             const int node_j,
                             const int nr,
                             const int nz) {
  double minimum = std::numeric_limits<double>::infinity();
  for (int di = -1; di <= 0; ++di) {
    const int i = node_i + di;
    if (i < 0 || i >= nr) {
      continue;
    }
    for (int dj = -1; dj <= 0; ++dj) {
      const int j = node_j + dj;
      if (j < 0 || j >= nz) {
        continue;
      }
      minimum = std::min(minimum, signed_cell_area(r, z, i, j, nz));
    }
  }
  return minimum;
}

double local_preconditioner(const std::vector<double>& r,
                            const std::vector<double>& z,
                            const int node_i,
                            const int node_j,
                            const int nr,
                            const int nz,
                            const std::vector<CellMetric>& metrics,
                            const RezoneGeometry geometry) {
  double value = 0.0;
  for (int di = -1; di <= 0; ++di) {
    const int i = node_i + di;
    if (i < 0 || i >= nr) {
      continue;
    }
    for (int dj = -1; dj <= 0; ++dj) {
      const int j = node_j + dj;
      if (j < 0 || j >= nz) {
        continue;
      }
      const std::size_t c = cell_index(i, j, nz);
      const double area = signed_cell_area(r, z, i, j, nz);
      if (geometry == RezoneGeometry::Planar) {
        value += metrics[c].preconditioner_weight / area;
      } else {
        const double center_r =
            0.25 * (r[node_index(i, j, nz)] +
                    r[node_index(i + 1, j, nz)] +
                    r[node_index(i + 1, j + 1, nz)] +
                    r[node_index(i, j + 1, nz)]);
        value += kTwoPi * center_r * metrics[c].preconditioner_weight / area;
      }
    }
  }
  return value;
}

BoundaryLine make_boundary_line(const std::vector<double>& r,
                                const std::vector<double>& z,
                                const std::size_t first,
                                const std::size_t last) {
  const double delta_r = r[last] - r[first];
  const double delta_z = z[last] - z[first];
  const double length = std::hypot(delta_r, delta_z);
  return BoundaryLine{r[first], z[first], delta_r / length, delta_z / length};
}

const BoundaryLine* boundary_line_for_node(
    const int i,
    const int j,
    const int nr,
    const int nz,
    const std::array<BoundaryLine, 4>& lines) {
  if (i == 0) {
    return &lines[0];
  }
  if (i == nr) {
    return &lines[1];
  }
  if (j == 0) {
    return &lines[2];
  }
  if (j == nz) {
    return &lines[3];
  }
  return nullptr;
}

void project_vector(const BoundaryLine* line, double& r, double& z) {
  if (line == nullptr) {
    return;
  }
  const double tangent_component =
      r * line->tangent_r + z * line->tangent_z;
  r = tangent_component * line->tangent_r;
  z = tangent_component * line->tangent_z;
}

void project_position(const BoundaryLine& line, double& r, double& z) {
  const double relative_r = r - line.origin_r;
  const double relative_z = z - line.origin_z;
  const double tangent_component =
      relative_r * line.tangent_r + relative_z * line.tangent_z;
  r = line.origin_r + tangent_component * line.tangent_r;
  z = line.origin_z + tangent_component * line.tangent_z;
}

bool all_cells_admissible(const std::vector<double>& candidate_r,
                          const std::vector<double>& candidate_z,
                          const std::vector<double>& current_r,
                          const std::vector<double>& current_z,
                          const int nr,
                          const int nz,
                          const std::vector<double>& initial_areas,
                          const double epsilon_j,
                          const RezoneGeometry geometry,
                          std::vector<std::size_t>* violating_cells = nullptr) {
  bool all_admissible = true;
  if (violating_cells != nullptr) {
    violating_cells->clear();
  }
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const std::size_t c = cell_index(i, j, nz);
      const double floor = epsilon_j * initial_areas[c];
      const auto candidate_jacobians =
          cell_corner_jacobians(candidate_r, candidate_z, i, j, nz);
      const auto current_jacobians =
          cell_corner_jacobians(current_r, current_z, i, j, nz);
      bool cell_admissible = true;
      for (std::size_t k = 0; k < candidate_jacobians.size(); ++k) {
        const double minimum_jacobian = std::min(floor, current_jacobians[k]);
        if (!(candidate_jacobians[k] >= minimum_jacobian) ||
            !std::isfinite(candidate_jacobians[k])) {
          cell_admissible = false;
        }
      }
      if (geometry == RezoneGeometry::Rz) {
        const double volume = signed_cell_rz_volume(
            candidate_r, candidate_z, i, j, nz);
        if (!(volume > 0.0) || !std::isfinite(volume)) {
          cell_admissible = false;
        }
      }
      if (!cell_admissible) {
        all_admissible = false;
        if (violating_cells != nullptr) {
          violating_cells->push_back(c);
        }
      }
    }
  }
  return all_admissible;
}

bool valid_params(const RezoneParams& params) {
  return (params.geometry == RezoneGeometry::Planar ||
          params.geometry == RezoneGeometry::Rz) &&
         params.iterations >= 0 && params.max_backtracks >= 0 &&
         std::isfinite(params.alpha) && params.alpha > 0.0 &&
         std::isfinite(params.kappa_gain) && params.kappa_gain >= 0.0 &&
         std::isfinite(params.omega_eta) && params.omega_eta >= 0.0 &&
         std::isfinite(params.epsilon_j) && params.epsilon_j > 0.0 &&
         std::isfinite(params.epsilon_L) && params.epsilon_L >= 0.0 &&
         std::isfinite(params.fd_relative_step) &&
         params.fd_relative_step > 0.0;
}

bool validate_input(const RezoneInput& input,
                    const RezoneParams& params,
                    std::string& error) {
  if (input.nr <= 0 || input.nz <= 0) {
    error = "non-positive structured topology";
    return false;
  }
  const std::size_t n_nodes =
      static_cast<std::size_t>((input.nr + 1) * (input.nz + 1));
  const std::size_t n_cells =
      static_cast<std::size_t>(input.nr * input.nz);
  if (input.node_r.size() != n_nodes || input.node_z.size() != n_nodes) {
    error = "node coordinate size mismatch";
    return false;
  }
  if (input.monitor.size() != n_cells) {
    error = "cell monitor size mismatch";
    return false;
  }
  if (!finite_coordinates(input.node_r, input.node_z)) {
    error = "non-finite node coordinate";
    return false;
  }
  if (!valid_params(params)) {
    error = "invalid rezone parameter";
    return false;
  }
  return true;
}

}  // namespace

double evaluate_planar_rezone_energy(const RezoneInput& input,
                                     const RezoneParams& params) {
  std::string error;
  if (!validate_input(input, params, error)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::vector<CellMetric> metrics;
  bool has_active_cells = false;
  if (!build_cell_metrics(input, params, metrics, has_active_cells)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return total_energy(input.node_r, input.node_z, input.nr, input.nz, metrics,
                      params.omega_eta, params.geometry);
}

RezoneResult solve_planar_rezone(const RezoneInput& input,
                                 const RezoneParams& params) {
  RezoneResult result;
  result.node_r = input.node_r;
  result.node_z = input.node_z;
  if (!validate_input(input, params, result.error)) {
    return result;
  }
  if (params.geometry == RezoneGeometry::Rz) {
    for (std::size_t n = 0; n < result.node_r.size(); ++n) {
      if (input.node_r[n] == 0.0) {
        result.node_r[n] = 0.0;
      }
    }
  }

  std::vector<CellMetric> metrics;
  bool has_active_cells = false;
  if (!build_cell_metrics(input, params, metrics, has_active_cells)) {
    result.error = "non-finite cell monitor";
    return result;
  }

  std::vector<double> initial_areas(metrics.size(), 0.0);
  for (int i = 0; i < input.nr; ++i) {
    for (int j = 0; j < input.nz; ++j) {
      const std::size_t c = cell_index(i, j, input.nz);
      initial_areas[c] =
          signed_cell_area(input.node_r, input.node_z, i, j, input.nz);
      const auto jacobians = cell_corner_jacobians(
          input.node_r, input.node_z, i, j, input.nz);
      if (!(initial_areas[c] > 0.0) ||
          std::any_of(jacobians.begin(), jacobians.end(),
                      [](const double J) { return !(J > 0.0); })) {
        result.error = "initial mesh is folded or degenerate";
        return result;
      }
      if (params.geometry == RezoneGeometry::Rz) {
        const double volume = signed_cell_rz_volume(
            result.node_r, result.node_z, i, j, input.nz);
        if (!(volume > 0.0) || !std::isfinite(volume)) {
          result.error = "initial mesh has non-positive RZ cell volume";
          return result;
        }
      }
    }
  }

  const std::array<BoundaryLine, 4> boundary_lines = {
      make_boundary_line(input.node_r, input.node_z,
                         node_index(0, 0, input.nz),
                         node_index(0, input.nz, input.nz)),
      make_boundary_line(input.node_r, input.node_z,
                         node_index(input.nr, 0, input.nz),
                         node_index(input.nr, input.nz, input.nz)),
      make_boundary_line(input.node_r, input.node_z,
                         node_index(0, 0, input.nz),
                         node_index(input.nr, 0, input.nz)),
      make_boundary_line(input.node_r, input.node_z,
                         node_index(0, input.nz, input.nz),
                         node_index(input.nr, input.nz, input.nz))};
  for (const BoundaryLine& line : boundary_lines) {
    if (!std::isfinite(line.tangent_r) || !std::isfinite(line.tangent_z)) {
      result.error = "degenerate initial boundary line";
      return result;
    }
  }

  result.initial_energy = total_energy(result.node_r, result.node_z, input.nr,
                                       input.nz, metrics, params.omega_eta,
                                       params.geometry);
  result.final_energy = result.initial_energy;
  if (!std::isfinite(result.initial_energy)) {
    result.error = "non-finite initial energy";
    return result;
  }
  if (params.geometry == RezoneGeometry::Rz) {
    compute_cell_rz_volumes(result.node_r, result.node_z, input.nr, input.nz,
                            result.cell_rz_volumes);
  }
  if (!has_active_cells || params.iterations == 0) {
    result.valid = true;
    return result;
  }

  const std::size_t n_nodes = result.node_r.size();
  std::vector<double> displacement_r(n_nodes, 0.0);
  std::vector<double> displacement_z(n_nodes, 0.0);
  std::vector<double> trial_r(n_nodes, 0.0);
  std::vector<double> trial_z(n_nodes, 0.0);

  for (int iteration = 0; iteration < params.iterations; ++iteration) {
    std::fill(displacement_r.begin(), displacement_r.end(), 0.0);
    std::fill(displacement_z.begin(), displacement_z.end(), 0.0);

    for (int i = 0; i <= input.nr; ++i) {
      for (int j = 0; j <= input.nz; ++j) {
        const bool corner = (i == 0 || i == input.nr) &&
                            (j == 0 || j == input.nz);
        if (corner) {
          continue;
        }
        const std::size_t n = node_index(i, j, input.nz);
        const bool axis_node = params.geometry == RezoneGeometry::Rz &&
                               input.node_r[n] == 0.0;
        const double minimum_area = minimum_adjacent_area(
            result.node_r, result.node_z, i, j, input.nr, input.nz);
        if (!(minimum_area > 0.0)) {
          result.error = "non-positive current adjacent-cell area";
          return result;
        }
        const double h_fd =
            params.fd_relative_step * std::sqrt(minimum_area);

        // PROTOTYPE-ONLY(Stage 1-2): deterministic central finite differences
        // replace the analytic nodal gradient until the Stage-6 GPU preparation.
        const double original_r = result.node_r[n];
        result.node_r[n] = original_r + h_fd;
        const double energy_r_plus = adjacent_patch_energy(
            result.node_r, result.node_z, i, j, input.nr, input.nz, metrics,
            params.omega_eta, params.geometry);
        result.node_r[n] = original_r - h_fd;
        const double energy_r_minus = adjacent_patch_energy(
            result.node_r, result.node_z, i, j, input.nr, input.nz, metrics,
            params.omega_eta, params.geometry);
        result.node_r[n] = original_r;

        const double original_z = result.node_z[n];
        result.node_z[n] = original_z + h_fd;
        const double energy_z_plus = adjacent_patch_energy(
            result.node_r, result.node_z, i, j, input.nr, input.nz, metrics,
            params.omega_eta, params.geometry);
        result.node_z[n] = original_z - h_fd;
        const double energy_z_minus = adjacent_patch_energy(
            result.node_r, result.node_z, i, j, input.nr, input.nz, metrics,
            params.omega_eta, params.geometry);
        result.node_z[n] = original_z;

        double gradient_r =
            (energy_r_plus - energy_r_minus) / (2.0 * h_fd);
        double gradient_z =
            (energy_z_plus - energy_z_minus) / (2.0 * h_fd);
        const BoundaryLine* line = boundary_line_for_node(
            i, j, input.nr, input.nz, boundary_lines);
        project_vector(line, gradient_r, gradient_z);
        if (axis_node) {
          gradient_r = 0.0;
        }
        const double L = local_preconditioner(
            result.node_r, result.node_z, i, j, input.nr, input.nz, metrics,
            params.geometry);
        displacement_r[n] =
            -params.alpha * gradient_r / (L + params.epsilon_L);
        displacement_z[n] =
            -params.alpha * gradient_z / (L + params.epsilon_L);
        project_vector(line, displacement_r[n], displacement_z[n]);
        if (axis_node) {
          displacement_r[n] = 0.0;
        }
      }
    }

    std::vector<std::size_t> violating_cells;
    const auto run_line_search = [&](const bool collect_smallest_violations) {
      double lambda = 1.0;
      for (int backtrack = 0; backtrack <= params.max_backtracks;
           ++backtrack) {
        for (int i = 0; i <= input.nr; ++i) {
          for (int j = 0; j <= input.nz; ++j) {
            const std::size_t n = node_index(i, j, input.nz);
            const bool axis_node = params.geometry == RezoneGeometry::Rz &&
                                   input.node_r[n] == 0.0;
            const bool corner = (i == 0 || i == input.nr) &&
                                (j == 0 || j == input.nz);
            if (corner) {
              trial_r[n] = axis_node ? 0.0 : input.node_r[n];
              trial_z[n] = input.node_z[n];
              continue;
            }
            trial_r[n] = result.node_r[n] + lambda * displacement_r[n];
            trial_z[n] = result.node_z[n] + lambda * displacement_z[n];
            const BoundaryLine* line = boundary_line_for_node(
                i, j, input.nr, input.nz, boundary_lines);
            if (line != nullptr) {
              project_position(*line, trial_r[n], trial_z[n]);
            }
            if (axis_node) {
              trial_r[n] = 0.0;
            }
          }
        }
        std::vector<std::size_t>* violations =
            collect_smallest_violations && backtrack == params.max_backtracks
                ? &violating_cells
                : nullptr;
        if (all_cells_admissible(
                trial_r, trial_z, result.node_r, result.node_z, input.nr,
                input.nz, initial_areas, params.epsilon_j, params.geometry,
                violations)) {
          const double trial_energy = total_energy(
              trial_r, trial_z, input.nr, input.nz, metrics, params.omega_eta,
              params.geometry);
          if (trial_energy < result.final_energy) {
            result.node_r.swap(trial_r);
            result.node_z.swap(trial_z);
            result.final_energy = trial_energy;
            ++result.accepted_iterations;
            return true;
          }
        }
        if (backtrack < params.max_backtracks) {
          ++result.backtrack_attempts;
          lambda *= 0.5;
        }
      }
      return false;
    };

    bool accepted = run_line_search(true);
    if (!accepted) {
      ++result.frozen_retries;
      for (const std::size_t c : violating_cells) {
        const int i = static_cast<int>(c) / input.nz;
        const int j = static_cast<int>(c) % input.nz;
        const std::array<std::size_t, 4> nodes = {
            node_index(i, j, input.nz),
            node_index(i + 1, j, input.nz),
            node_index(i + 1, j + 1, input.nz),
            node_index(i, j + 1, input.nz)};
        for (const std::size_t n : nodes) {
          displacement_r[n] = 0.0;
          displacement_z[n] = 0.0;
        }
      }
      accepted = run_line_search(false);
    }
    if (!accepted) {
      ++result.skipped_iterations;
    }
  }

  if (params.geometry == RezoneGeometry::Rz) {
    compute_cell_rz_volumes(result.node_r, result.node_z, input.nr, input.nz,
                            result.cell_rz_volumes);
  }
  result.valid = true;
  return result;
}

}  // namespace tenryu::hydro::ale_align
