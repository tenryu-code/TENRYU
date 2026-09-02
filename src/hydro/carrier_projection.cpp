#include "hydro/carrier_projection.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace tenryu::hydro {

namespace {

bool solve_cyclic_tridiagonal_ldlt(int n,
                                   std::vector<double>& d,
                                   const std::vector<double>& off,
                                   const std::vector<double>& rhs,
                                   std::vector<double>& q) {
  const int m = n - 1;
  std::vector<double> border(m, 0.0);
  std::vector<double> lsub(m, 0.0);
  std::vector<double> lbord(m, 0.0);
  border[0] = off[m];
  border[m - 1] += off[m - 1];

  for (int i = 0; i < m; ++i) {
    const double pivot = d[i];
    if (!(std::isfinite(pivot) && pivot > 0.0)) {
      return false;
    }
    const double inv = 1.0 / pivot;
    const double fraw = border[i];
    const double li = (i < m - 1) ? off[i] * inv : 0.0;
    const double fi = fraw * inv;
    if (i < m - 1) {
      d[i + 1] -= li * off[i];
      border[i + 1] -= li * fraw;
    }
    d[m] -= fi * fraw;
    lsub[i] = li;
    lbord[i] = fi;
  }
  if (!(std::isfinite(d[m]) && d[m] > 0.0)) {
    return false;
  }

  q.assign(static_cast<std::size_t>(n), 0.0);
  q[0] = rhs[0];
  for (int i = 1; i < m; ++i) {
    q[i] = rhs[i] - lsub[i - 1] * q[i - 1];
  }
  double acc = rhs[m];
  for (int i = 0; i < m; ++i) {
    acc -= lbord[i] * q[i];
  }
  q[m] = acc;

  for (int i = 0; i <= m; ++i) {
    q[i] /= d[i];
  }
  for (int i = m - 1; i >= 0; --i) {
    q[i] -= lbord[i] * q[m];
    if (i < m - 1) {
      q[i] -= lsub[i] * q[i + 1];
    }
  }
  return true;
}

}  // namespace

CarrierProjectionResult project_carrier_boundary_velocity(
    const CarrierProjectionInput& in,
    double* out_vr,
    double* out_vz) {
  CarrierProjectionResult result;
  if (in.n_masters < 3) {
    result.reject_reason = "carrier_projection_too_few_masters";
    return result;
  }
  if (in.n_nodes_total <= 0 || in.n_slaves < 0 ||
      in.master_node == nullptr || in.node_mass == nullptr ||
      in.node_pr == nullptr || in.node_pz == nullptr || out_vr == nullptr ||
      out_vz == nullptr ||
      (in.n_slaves > 0 &&
       (in.slave_node == nullptr || in.slave_edge == nullptr ||
        in.slave_lambda == nullptr))) {
    result.reject_reason = "carrier_projection_bad_input";
    return result;
  }
  for (int k = 0; k < in.n_masters; ++k) {
    if (in.master_node[k] < 0 || in.master_node[k] >= in.n_nodes_total) {
      result.reject_reason = "carrier_projection_node_out_of_range";
      return result;
    }
  }
  for (int s = 0; s < in.n_slaves; ++s) {
    if (in.slave_node[s] < 0 || in.slave_node[s] >= in.n_nodes_total) {
      result.reject_reason = "carrier_projection_node_out_of_range";
      return result;
    }
  }
  for (int s = 0; s < in.n_slaves; ++s) {
    if (in.slave_edge[s] < 0 || in.slave_edge[s] >= in.n_masters) {
      result.reject_reason = "carrier_projection_edge_out_of_range";
      return result;
    }
  }
  for (int s = 0; s < in.n_slaves; ++s) {
    const double lambda = in.slave_lambda[s];
    if (!std::isfinite(lambda) || lambda <= 0.0 || lambda >= 1.0) {
      result.reject_reason = "carrier_projection_lambda_out_of_range";
      return result;
    }
  }

  std::vector<char> seen(static_cast<std::size_t>(in.n_nodes_total), 0);
  for (int k = 0; k < in.n_masters; ++k) {
    const int node = in.master_node[k];
    if (seen[node] != 0) {
      result.reject_reason = "carrier_projection_duplicate_node";
      return result;
    }
    seen[node] = 1;
  }
  for (int s = 0; s < in.n_slaves; ++s) {
    const int node = in.slave_node[s];
    if (seen[node] != 0) {
      result.reject_reason = "carrier_projection_duplicate_node";
      return result;
    }
    seen[node] = 1;
  }

  for (int k = 0; k < in.n_masters; ++k) {
    const double mass = in.node_mass[in.master_node[k]];
    if (!std::isfinite(mass) || mass <= 0.0) {
      result.reject_reason = "carrier_projection_nonpositive_mass";
      return result;
    }
  }
  for (int s = 0; s < in.n_slaves; ++s) {
    const double mass = in.node_mass[in.slave_node[s]];
    if (!std::isfinite(mass) || mass <= 0.0) {
      result.reject_reason = "carrier_projection_nonpositive_mass";
      return result;
    }
  }
  for (int k = 0; k < in.n_masters; ++k) {
    const int node = in.master_node[k];
    if (!std::isfinite(in.node_pr[node]) ||
        !std::isfinite(in.node_pz[node])) {
      result.reject_reason = "carrier_projection_nonfinite_momentum";
      return result;
    }
  }
  for (int s = 0; s < in.n_slaves; ++s) {
    const int node = in.slave_node[s];
    if (!std::isfinite(in.node_pr[node]) ||
        !std::isfinite(in.node_pz[node])) {
      result.reject_reason = "carrier_projection_nonfinite_momentum";
      return result;
    }
  }

  const int n = in.n_masters;
  std::vector<double> diag(static_cast<std::size_t>(n));
  std::vector<double> off(static_cast<std::size_t>(n));
  std::vector<double> rhs_r(static_cast<std::size_t>(n));
  std::vector<double> rhs_z(static_cast<std::size_t>(n));
  for (int k = 0; k < n; ++k) {
    const int node = in.master_node[k];
    diag[k] = in.node_mass[node];
    off[k] = 0.0;
    rhs_r[k] = in.node_pr[node];
    rhs_z[k] = in.node_pz[node];
  }
  for (int s = 0; s < in.n_slaves; ++s) {
    const int e = in.slave_edge[s];
    const int ep = (e + 1) % n;
    const double lam = in.slave_lambda[s];
    const double oml = 1.0 - lam;
    const int node = in.slave_node[s];
    const double mass = in.node_mass[node];
    diag[e] += oml * oml * mass;
    diag[ep] += lam * lam * mass;
    off[e] += oml * lam * mass;
    rhs_r[e] += oml * in.node_pr[node];
    rhs_r[ep] += lam * in.node_pr[node];
    rhs_z[e] += oml * in.node_pz[node];
    rhs_z[ep] += lam * in.node_pz[node];
  }

  std::vector<double> diag_r = diag;
  std::vector<double> off_r = off;
  std::vector<double> diag_z = diag;
  const std::vector<double> off_z = off;
  for (int k = 0; k < n; ++k) {
    if (in.master_fix_r != nullptr && in.master_fix_r[k] != 0) {
      rhs_r[k] = 0.0;
      diag_r[k] = 1.0;
      off_r[k] = 0.0;
      off_r[(k - 1 + n) % n] = 0.0;
      ++result.n_fixed_radial_dofs;
    }
  }
  // Fixed r rows enter as identity rows with zero rhs, so their solution is
  // exactly 0.0 and the factorization stays SPD.
  std::vector<double> qdot_r;
  std::vector<double> qdot_z;
  if (!solve_cyclic_tridiagonal_ldlt(n, diag_r, off_r, rhs_r, qdot_r) ||
      !solve_cyclic_tridiagonal_ldlt(n, diag_z, off_z, rhs_z, qdot_z)) {
    result.reject_reason = "carrier_projection_nonpositive_pivot";
    return result;
  }

  result.master_qdot_r = qdot_r;
  result.master_qdot_z = qdot_z;
  for (int k = 0; k < n; ++k) {
    const int node = in.master_node[k];
    out_vr[node] = qdot_r[k];
    out_vz[node] = qdot_z[k];
  }
  for (int s = 0; s < in.n_slaves; ++s) {
    const int e = in.slave_edge[s];
    const int ep = (e + 1) % n;
    const double lam = in.slave_lambda[s];
    const double oml = 1.0 - lam;
    const int node = in.slave_node[s];
    out_vr[node] = oml * qdot_r[e] + lam * qdot_r[ep];
    out_vz[node] = oml * qdot_z[e] + lam * qdot_z[ep];
  }

  for (int k = 0; k < n; ++k) {
    const int node = in.master_node[k];
    result.momentum_in_r += in.node_pr[node];
    result.momentum_out_r += in.node_mass[node] * out_vr[node];
    result.momentum_in_z += in.node_pz[node];
    result.momentum_out_z += in.node_mass[node] * out_vz[node];
  }
  for (int s = 0; s < in.n_slaves; ++s) {
    const int node = in.slave_node[s];
    result.momentum_in_r += in.node_pr[node];
    result.momentum_out_r += in.node_mass[node] * out_vr[node];
    result.momentum_in_z += in.node_pz[node];
    result.momentum_out_z += in.node_mass[node] * out_vz[node];
  }
  for (int k = 0; k < n; ++k) {
    const int node = in.master_node[k];
    const double ur_star = in.node_pr[node] / in.node_mass[node];
    const double uz_star = in.node_pz[node] / in.node_mass[node];
    const double dr = ur_star - out_vr[node];
    const double dz = uz_star - out_vz[node];
    result.delta_k +=
        0.5 * in.node_mass[node] * (dr * dr + dz * dz);
  }
  for (int s = 0; s < in.n_slaves; ++s) {
    const int node = in.slave_node[s];
    const double ur_star = in.node_pr[node] / in.node_mass[node];
    const double uz_star = in.node_pz[node] / in.node_mass[node];
    const double dr = ur_star - out_vr[node];
    const double dz = uz_star - out_vz[node];
    result.delta_k +=
        0.5 * in.node_mass[node] * (dr * dr + dz * dz);
  }

  result.ok = true;
  return result;
}

CarrierDefectAllocation allocate_projection_defect_to_cells(
    const CarrierProjectionInput& in,
    const double* node_vr_new,
    const double* node_vz_new,
    int n_corners,
    const int* corner_cell,
    const int* corner_node,
    const double* corner_mass,
    int n_cells,
    double* delta_e_cell) {
  CarrierDefectAllocation result;
  if (n_cells <= 0 || n_corners < 0 || delta_e_cell == nullptr ||
      node_vr_new == nullptr || node_vz_new == nullptr ||
      (n_corners > 0 &&
       (corner_cell == nullptr || corner_node == nullptr ||
        corner_mass == nullptr)) ||
      in.master_node == nullptr || in.node_mass == nullptr ||
      in.node_pr == nullptr || in.node_pz == nullptr || in.n_masters <= 0 ||
      in.n_slaves < 0 ||
      (in.n_slaves > 0 && in.slave_node == nullptr) ||
      in.n_nodes_total <= 0) {
    result.reject_reason = "carrier_defect_bad_input";
    return result;
  }
  for (int i = 0; i < n_corners; ++i) {
    if (corner_cell[i] < 0 || corner_cell[i] >= n_cells ||
        corner_node[i] < 0 || corner_node[i] >= in.n_nodes_total) {
      result.reject_reason = "carrier_defect_index_out_of_range";
      return result;
    }
  }
  for (int i = 0; i < n_corners; ++i) {
    if (!std::isfinite(corner_mass[i]) || corner_mass[i] < 0.0) {
      result.reject_reason = "carrier_defect_bad_corner_mass";
      return result;
    }
  }

  std::vector<char> is_boundary(
      static_cast<std::size_t>(in.n_nodes_total), 0);
  for (int k = 0; k < in.n_masters; ++k) {
    const int node = in.master_node[k];
    if (node < 0 || node >= in.n_nodes_total) {
      result.reject_reason = "carrier_defect_index_out_of_range";
      return result;
    }
    if (!std::isfinite(in.node_mass[node]) || in.node_mass[node] <= 0.0) {
      result.reject_reason = "carrier_defect_nonpositive_mass";
      return result;
    }
    is_boundary[node] = 1;
  }
  for (int s = 0; s < in.n_slaves; ++s) {
    const int node = in.slave_node[s];
    if (node < 0 || node >= in.n_nodes_total) {
      result.reject_reason = "carrier_defect_index_out_of_range";
      return result;
    }
    if (!std::isfinite(in.node_mass[node]) || in.node_mass[node] <= 0.0) {
      result.reject_reason = "carrier_defect_nonpositive_mass";
      return result;
    }
    is_boundary[node] = 1;
  }

  for (int cell = 0; cell < n_cells; ++cell) {
    delta_e_cell[cell] = 0.0;
  }
  for (int i = 0; i < n_corners; ++i) {
    const int node = corner_node[i];
    if (is_boundary[node] != 0) {
      const double ur_star = in.node_pr[node] / in.node_mass[node];
      const double uz_star = in.node_pz[node] / in.node_mass[node];
      const double dr = ur_star - node_vr_new[node];
      const double dz = uz_star - node_vz_new[node];
      const double contrib =
          0.5 * corner_mass[i] * (dr * dr + dz * dz);
      delta_e_cell[corner_cell[i]] += contrib;
      result.total += contrib;
    }
  }
  result.ok = true;
  return result;
}

}  // namespace tenryu::hydro
