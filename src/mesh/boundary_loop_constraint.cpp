#include "mesh/boundary_loop_constraint.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace tenryu::mesh {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Dense in-place lower Cholesky of a row-major symmetric n x n matrix.
// Returns false if a pivot is not strictly positive and finite (matrix not
// numerically SPD). The strict upper triangle is zeroed.
bool cholesky_lower_in_place(std::vector<double>& a, const int n) {
  for (int j = 0; j < n; ++j) {
    double d = a[static_cast<std::size_t>(j) * n + j];
    for (int k = 0; k < j; ++k) {
      const double ljk = a[static_cast<std::size_t>(j) * n + k];
      d -= ljk * ljk;
    }
    if (!std::isfinite(d) || !(d > 0.0)) {
      return false;
    }
    const double ljj = std::sqrt(d);
    a[static_cast<std::size_t>(j) * n + j] = ljj;
    for (int i = j + 1; i < n; ++i) {
      double s = a[static_cast<std::size_t>(i) * n + j];
      for (int k = 0; k < j; ++k) {
        s -= a[static_cast<std::size_t>(i) * n + k] *
             a[static_cast<std::size_t>(j) * n + k];
      }
      a[static_cast<std::size_t>(i) * n + j] = s / ljj;
    }
    for (int k = j + 1; k < n; ++k) {
      a[static_cast<std::size_t>(j) * n + k] = 0.0;
    }
  }
  return true;
}

// Solves (L L^T) x = b in place given the lower factor from
// cholesky_lower_in_place.
void cholesky_solve_in_place(const std::vector<double>& l,
                             const int n,
                             std::vector<double>& x) {
  for (int i = 0; i < n; ++i) {
    double s = x[static_cast<std::size_t>(i)];
    for (int k = 0; k < i; ++k) {
      s -= l[static_cast<std::size_t>(i) * n + k] * x[static_cast<std::size_t>(k)];
    }
    x[static_cast<std::size_t>(i)] = s / l[static_cast<std::size_t>(i) * n + i];
  }
  for (int i = n - 1; i >= 0; --i) {
    double s = x[static_cast<std::size_t>(i)];
    for (int k = i + 1; k < n; ++k) {
      s -= l[static_cast<std::size_t>(k) * n + i] * x[static_cast<std::size_t>(k)];
    }
    x[static_cast<std::size_t>(i)] = s / l[static_cast<std::size_t>(i) * n + i];
  }
}

}  // namespace

BoundaryLoopLowModeBasis build_boundary_loop_low_mode_basis(
    const int n_nodes, const int n_harmonics) {
  if (n_nodes < 3) {
    throw std::invalid_argument(
        "boundary_loop_constraint: n_nodes must be >= 3 (closed loop), got " +
        std::to_string(n_nodes));
  }
  if (n_harmonics < 0) {
    throw std::invalid_argument(
        "boundary_loop_constraint: n_harmonics must be >= 0, got " +
        std::to_string(n_harmonics));
  }
  if (2 * n_harmonics + 1 > n_nodes) {
    throw std::invalid_argument(
        "boundary_loop_constraint: 2*n_harmonics+1 (= " +
        std::to_string(2 * n_harmonics + 1) + ") must be <= n_nodes (= " +
        std::to_string(n_nodes) + ") for a full-rank, alias-free basis");
  }

  BoundaryLoopLowModeBasis basis;
  basis.n_nodes = n_nodes;
  basis.n_harmonics = n_harmonics;
  const int n_scalar = basis.n_scalar_modes();
  const int n_modes = basis.n_modes();
  basis.tangent.assign(
      static_cast<std::size_t>(basis.n_dofs()) * static_cast<std::size_t>(n_modes),
      0.0);

  for (int l = 0; l < n_nodes; ++l) {
    for (int m = 0; m < n_scalar; ++m) {
      double phi;
      if (m == 0) {
        phi = 1.0;
      } else {
        const int k = (m + 1) / 2;
        const double angle = 2.0 * kPi * static_cast<double>(k) *
                             static_cast<double>(l) /
                             static_cast<double>(n_nodes);
        phi = (m % 2 == 1) ? std::cos(angle) : std::sin(angle);
      }
      // r-dof row gets the r-block column; z-dof row the z-block column.
      basis.tangent[static_cast<std::size_t>(2 * l) * n_modes + m] = phi;
      basis.tangent[static_cast<std::size_t>(2 * l + 1) * n_modes +
                    (n_scalar + m)] = phi;
    }
  }
  return basis;
}

BoundaryLoopConstrainedForce project_force_to_constrained_accel(
    const BoundaryLoopLowModeBasis& basis,
    const std::vector<double>& node_mass,
    const std::vector<double>& force) {
  const int n_dofs = basis.n_dofs();
  const int n_modes = basis.n_modes();
  if (basis.n_nodes < 3 ||
      basis.tangent.size() != static_cast<std::size_t>(n_dofs) *
                                  static_cast<std::size_t>(n_modes)) {
    throw std::invalid_argument(
        "boundary_loop_constraint: basis tangent size inconsistent with "
        "n_nodes/n_harmonics");
  }
  if (node_mass.size() != static_cast<std::size_t>(basis.n_nodes)) {
    throw std::invalid_argument(
        "boundary_loop_constraint: node_mass must have n_nodes entries");
  }
  if (force.size() != static_cast<std::size_t>(n_dofs)) {
    throw std::invalid_argument(
        "boundary_loop_constraint: force must have 2*n_nodes entries");
  }
  for (const double m : node_mass) {
    if (!std::isfinite(m) || !(m > 0.0)) {
      throw std::invalid_argument(
          "boundary_loop_constraint: node masses must be finite and > 0");
    }
  }

  // G = T^T M_B T (upper triangle, then mirrored) and rhs = T^T F.
  std::vector<double> g(static_cast<std::size_t>(n_modes) * n_modes, 0.0);
  std::vector<double> coeff(static_cast<std::size_t>(n_modes), 0.0);
  for (int row = 0; row < n_dofs; ++row) {
    const double m = node_mass[static_cast<std::size_t>(row / 2)];
    const double f = force[static_cast<std::size_t>(row)];
    const double* t_row = &basis.tangent[static_cast<std::size_t>(row) * n_modes];
    for (int a = 0; a < n_modes; ++a) {
      const double ta = t_row[a];
      if (ta == 0.0) {
        continue;  // the off-family half of each row is structurally zero
      }
      coeff[static_cast<std::size_t>(a)] += ta * f;
      const double mta = m * ta;
      for (int b = a; b < n_modes; ++b) {
        g[static_cast<std::size_t>(a) * n_modes + b] += mta * t_row[b];
      }
    }
  }
  for (int a = 0; a < n_modes; ++a) {
    for (int b = a + 1; b < n_modes; ++b) {
      g[static_cast<std::size_t>(b) * n_modes + a] =
          g[static_cast<std::size_t>(a) * n_modes + b];
    }
  }

  if (!cholesky_lower_in_place(g, n_modes)) {
    throw std::runtime_error(
        "boundary_loop_constraint: G = T^T M_B T is not numerically SPD");
  }
  cholesky_solve_in_place(g, n_modes, coeff);  // coeff = G^{-1} T^T F

  BoundaryLoopConstrainedForce out;
  out.accel.assign(static_cast<std::size_t>(n_dofs), 0.0);
  out.reaction.assign(static_cast<std::size_t>(n_dofs), 0.0);
  for (int row = 0; row < n_dofs; ++row) {
    const double* t_row = &basis.tangent[static_cast<std::size_t>(row) * n_modes];
    double a_row = 0.0;
    for (int a = 0; a < n_modes; ++a) {
      a_row += t_row[a] * coeff[static_cast<std::size_t>(a)];
    }
    out.accel[static_cast<std::size_t>(row)] = a_row;
    out.reaction[static_cast<std::size_t>(row)] =
        node_mass[static_cast<std::size_t>(row / 2)] * a_row -
        force[static_cast<std::size_t>(row)];
  }
  return out;
}

}  // namespace tenryu::mesh
