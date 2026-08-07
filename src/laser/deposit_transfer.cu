#include "laser/deposit_transfer.cuh"

#include "parallel/reduction.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#include "core/constants.hpp"
#include "core/error.hpp"
#include "laser/bilinear_interpolation.cuh"

namespace tenryu::laser {
namespace {

constexpr double kGhostTransitionMinWeightFactor = 0.25;
constexpr double kGhostTransitionMaxWeightFactor = 4.0;
constexpr int kDepositSmoothBoundaryGuardCells = 1;
constexpr int kTransfer2dDiagTopN = 5;

struct Transfer2dDiagNode {
  int idx = -1;
  double value = 0.0;
};

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

int locate_cell_1d(const std::vector<double>& edges, const double r) {
  const int n_cells = static_cast<int>(edges.size()) - 1;
  if (n_cells <= 0) {
    return 0;
  }
  if (r <= edges.front()) {
    return 0;
  }
  if (r >= edges.back()) {
    return n_cells - 1;
  }
  auto it = std::upper_bound(edges.begin(), edges.end(), r);
  const int idx = static_cast<int>(std::distance(edges.begin(), it)) - 1;
  return std::max(0, std::min(n_cells - 1, idx));
}

std::vector<Transfer2dDiagNode> top_deposit_nodes(const std::vector<double>& dep_lm,
                                                  const int top_n) {
  std::vector<Transfer2dDiagNode> top;
  top.reserve(static_cast<std::size_t>(top_n));
  for (int idx = 0; idx < static_cast<int>(dep_lm.size()); ++idx) {
    const double value = dep_lm[static_cast<std::size_t>(idx)];
    Transfer2dDiagNode node{idx, value};
    auto it = std::lower_bound(
        top.begin(), top.end(), node,
        [](const Transfer2dDiagNode& a, const Transfer2dDiagNode& b) {
          return a.value > b.value;
        });
    top.insert(it, node);
    if (static_cast<int>(top.size()) > top_n) {
      top.pop_back();
    }
  }
  return top;
}

void log_transfer_2d_detail_diag(const std::vector<double>& dep_lm,
                                 const std::vector<double>& dep_hm_kernel,
                                 const std::vector<double>& node_R,
                                 const std::vector<double>& node_Z,
                                 const std::vector<double>& r_cell,
                                 const std::vector<double>& z_cell,
                                 const double dt,
                                 const int n_nodes_r,
                                 const int n_nodes_z) {
  if (dep_lm.empty() || dep_hm_kernel.empty() || node_R.empty() || node_Z.empty() ||
      r_cell.size() != dep_hm_kernel.size() || z_cell.size() != dep_hm_kernel.size() ||
      n_nodes_r <= 1 || n_nodes_z <= 1) {
    core::log_warning("[laser_transfer_2d_detail] skipped: size mismatch");
    return;
  }

  int nonzero = 0;
  long double sum = 0.0L;
  double min_value = dep_lm.front();
  double max_value = dep_lm.front();
  for (const double value : dep_lm) {
    nonzero += (value != 0.0) ? 1 : 0;
    sum += static_cast<long double>(value);
    min_value = std::min(min_value, value);
    max_value = std::max(max_value, value);
  }

  const auto top = top_deposit_nodes(dep_lm, kTransfer2dDiagTopN);
  std::ostringstream summary;
  summary << std::setprecision(17)
          << "[laser_transfer_2d_detail] lm_stats nonzero=" << nonzero
          << " sum=" << static_cast<double>(sum)
          << " min=" << min_value
          << " max=" << max_value;
  core::log_warning(summary.str());

  for (int rank = 0; rank < static_cast<int>(top.size()); ++rank) {
    const int i_lm = top[static_cast<std::size_t>(rank)].idx / n_nodes_z;
    const int j_lm = top[static_cast<std::size_t>(rank)].idx % n_nodes_z;
    std::ostringstream msg;
    msg << std::setprecision(17)
        << "[laser_transfer_2d_detail] top_lm rank=" << rank
        << " i=" << i_lm
        << " j=" << j_lm
        << " R=" << node_R[static_cast<std::size_t>(i_lm)]
        << " Z=" << node_Z[static_cast<std::size_t>(j_lm)]
        << " value=" << top[static_cast<std::size_t>(rank)].value;
    core::log_warning(msg.str());
  }

  for (int rank = 0; rank < static_cast<int>(top.size()); ++rank) {
    const int i_lm = top[static_cast<std::size_t>(rank)].idx / n_nodes_z;
    const int j_lm = top[static_cast<std::size_t>(rank)].idx % n_nodes_z;
    const double r_node = node_R[static_cast<std::size_t>(i_lm)];
    const double z_node = node_Z[static_cast<std::size_t>(j_lm)];

    int closest_c = 0;
    double closest_d2 = std::numeric_limits<double>::infinity();
    for (int c = 0; c < static_cast<int>(dep_hm_kernel.size()); ++c) {
      const double dr = r_cell[static_cast<std::size_t>(c)] - r_node;
      const double dz = z_cell[static_cast<std::size_t>(c)] - z_node;
      const double d2 = dr * dr + dz * dz;
      if (d2 < closest_d2) {
        closest_d2 = d2;
        closest_c = c;
      }
    }

    const double rc = r_cell[static_cast<std::size_t>(closest_c)];
    const double zc = z_cell[static_cast<std::size_t>(closest_c)];
    const double r_min = node_R.front();
    const double r_max = node_R[static_cast<std::size_t>(n_nodes_r - 1)];
    const double z_min = node_Z.front();
    const double z_max = node_Z[static_cast<std::size_t>(n_nodes_z - 1)];
    const double scale =
        std::max(1.0, std::max(std::abs(r_max), std::max(std::abs(z_min), std::abs(z_max))));
    const double tol = 1.0e-12 * scale;
    double interp = 0.0;
    BilinearCell cell;
    BilinearWeights w;
    if (rc >= r_min - tol && rc <= r_max + tol && zc >= z_min - tol && zc <= z_max + tol) {
      cell = BilinearInterp::locate_cell(node_R.data(), node_Z.data(), n_nodes_r, n_nodes_z,
                                         rc, zc);
      w = BilinearInterp::compute_weights(cell.xi, cell.eta);
      interp = BilinearInterp::interpolate(dep_lm.data(), n_nodes_z, cell, w);
    }

    const int n00 = cell.i * n_nodes_z + cell.j;
    const int n10 = (cell.i + 1) * n_nodes_z + cell.j;
    const int n01 = cell.i * n_nodes_z + (cell.j + 1);
    const int n11 = (cell.i + 1) * n_nodes_z + (cell.j + 1);
    std::ostringstream msg;
    msg << std::setprecision(17)
        << "[laser_transfer_2d_detail] nearest_hm rank=" << rank
        << " lm_i=" << i_lm
        << " lm_j=" << j_lm
        << " c=" << closest_c
        << " rc=" << rc
        << " zc=" << zc
        << " nodes=((" << cell.i << "," << cell.j << "),("
        << cell.i + 1 << "," << cell.j << "),("
        << cell.i << "," << cell.j + 1 << "),("
        << cell.i + 1 << "," << cell.j + 1 << "))"
        << " vals=(" << dep_lm[static_cast<std::size_t>(n00)]
        << "," << dep_lm[static_cast<std::size_t>(n10)]
        << "," << dep_lm[static_cast<std::size_t>(n01)]
        << "," << dep_lm[static_cast<std::size_t>(n11)] << ")"
        << " xi=" << cell.xi
        << " eta=" << cell.eta
        << " interp_power=" << interp
        << " actual_energy=" << dep_hm_kernel[static_cast<std::size_t>(closest_c)]
        << " actual_power=" << dep_hm_kernel[static_cast<std::size_t>(closest_c)] / dt;
    core::log_warning(msg.str());
  }
}

int find_active_anchor_1d(const std::vector<std::uint8_t>& cell_is_blocked,
                          const int c,
                          int& direction,
                          const bool prefer_outward,
                          const bool allow_opposite_fallback) {
  const int n_cells = static_cast<int>(cell_is_blocked.size());
  direction = 0;
  auto search = [&](const int begin, const int end, const int step, const int dir) {
    for (int d = begin; d != end; d += step) {
      if (cell_is_blocked[static_cast<std::size_t>(d)] == 0U) {
        direction = dir;
        return d;
      }
    }
    return -1;
  };

  if (prefer_outward) {
    const int outward = search(c + 1, n_cells, 1, 1);
    if (outward >= 0 || !allow_opposite_fallback) {
      return outward;
    }
    return search(c - 1, -1, -1, -1);
  }

  const int inward = search(c - 1, -1, -1, -1);
  if (inward >= 0 || !allow_opposite_fallback) {
    return inward;
  }
  return search(c + 1, n_cells, 1, 1);
}

std::vector<double> compute_cell_effective_A(const LaserMesh& mesh,
                                             const core::State& state,
                                             const std::size_t n_cells) {
  std::vector<double> A_eff(n_cells, std::max(mesh.material_A, 1.0e-12));
  const int n_mat = static_cast<int>(mesh.material_A_list.size());
  if (n_cells == 0 || n_mat <= 0) {
    return A_eff;
  }

  const std::size_t expected = n_cells * static_cast<std::size_t>(n_mat);
  if (state.volFrac.size() != expected) {
    return A_eff;
  }

  std::vector<double> volfrac(state.volFrac.size(), 0.0);
  state.volFrac.copy_to_host(volfrac.data());
  for (std::size_t c = 0; c < n_cells; ++c) {
    const std::size_t base = c * static_cast<std::size_t>(n_mat);
    double frac_sum = 0.0;
    double inv_A_c = 0.0;
    for (int m = 0; m < n_mat; ++m) {
      const double frac = std::max(volfrac[base + static_cast<std::size_t>(m)], 0.0);
      frac_sum += frac;
      const double A_m =
          std::max(mesh.material_A_list[static_cast<std::size_t>(m)], 1.0e-12);
      inv_A_c += frac / A_m;
    }
    if (frac_sum > 1.0e-30) {
      inv_A_c /= frac_sum;
    }
    if (std::isfinite(inv_A_c) && inv_A_c > 1.0e-30) {
      A_eff[c] = 1.0 / inv_A_c;
    }
  }
  return A_eff;
}

std::vector<double> load_cell_mass_1d(const core::State& state,
                                      const std::vector<double>& rho,
                                      const std::size_t n_cells) {
  std::vector<double> cell_mass(n_cells, 0.0);
  if (state.mass.size() == n_cells) {
    state.mass.copy_to_host(cell_mass.data());
  } else if (state.vol.size() == n_cells && rho.size() == n_cells) {
    std::vector<double> vol(n_cells, 0.0);
    state.vol.copy_to_host(vol.data());
    for (std::size_t c = 0; c < n_cells; ++c) {
      cell_mass[c] = rho[c] * vol[c];
    }
  }

  for (double& mass : cell_mass) {
    if (!(std::isfinite(mass) && mass > 0.0)) {
      mass = 0.0;
    }
  }
  return cell_mass;
}

double compute_cell_n_hat_approx(const double rho,
                                 const double zbar,
                                 const double A_eff,
                                 const double n_crit) {
  if (!(rho > 0.0) || !(zbar > 0.0) || !(A_eff > 0.0) || !(n_crit > 0.0)) {
    return 0.0;
  }
  return rho * zbar /
         (A_eff * core::constants::proton_mass * n_crit);
}

struct CriticalSurfaceEstimate1D {
  int fcrit = -1;
  double r_interp = -1.0;
};

CriticalSurfaceEstimate1D estimate_critical_surface_1d(
    const std::vector<double>& n_hat_cell,
    const std::vector<double>& r_edges) {
  const int n_cells = static_cast<int>(n_hat_cell.size());
  if (n_cells <= 0 || static_cast<int>(r_edges.size()) != n_cells + 1) {
    return {};
  }

  int outermost_critical = -1;
  for (int c = 0; c < n_cells; ++c) {
    if (n_hat_cell[static_cast<std::size_t>(c)] >= 1.0) {
      outermost_critical = c;
    }
  }
  if (outermost_critical < 0) {
    return {};
  }

  int fcrit = -1;
  for (int c = outermost_critical; c < n_cells; ++c) {
    const bool this_supercritical = (n_hat_cell[static_cast<std::size_t>(c)] >= 1.0);
    const bool next_subcritical =
        (c + 1 >= n_cells) || (n_hat_cell[static_cast<std::size_t>(c + 1)] < 1.0);
    if (this_supercritical && next_subcritical) {
      fcrit = c + 1;
      break;
    }
  }
  if (fcrit <= 0 || fcrit >= n_cells) {
    return {};
  }

  double r_interp = r_edges[static_cast<std::size_t>(fcrit)];
  const int c_hi = fcrit - 1;
  const int c_lo = fcrit;
  const double n_hi = n_hat_cell[static_cast<std::size_t>(c_hi)];
  const double n_lo = n_hat_cell[static_cast<std::size_t>(c_lo)];
  const double r_hi = 0.5 * (r_edges[static_cast<std::size_t>(c_hi)] +
                             r_edges[static_cast<std::size_t>(c_hi + 1)]);
  const double r_lo = 0.5 * (r_edges[static_cast<std::size_t>(c_lo)] +
                             r_edges[static_cast<std::size_t>(c_lo + 1)]);
  if (n_hi > 1.0 && n_lo > 0.0 && n_lo < 1.0 && r_lo > r_hi) {
    const double denom = std::log(n_lo / n_hi);
    if (std::isfinite(denom) && std::abs(denom) > 1.0e-12) {
      const double theta = std::clamp(std::log(1.0 / n_hi) / denom, 0.0, 1.0);
      r_interp = std::clamp(r_hi + theta * (r_lo - r_hi), r_hi, r_lo);
    }
  }

  return CriticalSurfaceEstimate1D{fcrit, r_interp};
}

int find_outer_surface_cell_1d(const std::vector<std::uint8_t>& cell_is_void) {
  for (int c = static_cast<int>(cell_is_void.size()) - 1; c >= 0; --c) {
    if (cell_is_void[static_cast<std::size_t>(c)] == 0U) {
      return c;
    }
  }
  return -1;
}

int count_resolved_subcritical_cells_1d(const std::vector<std::uint8_t>& cell_is_void,
                                        const std::vector<double>& rho,
                                        const std::vector<double>& zbar,
                                        const std::vector<double>& A_eff,
                                        const int outer_surface_cell,
                                        const double resolved_nhat,
                                        const double n_crit) {
  if (outer_surface_cell < 0 || !(resolved_nhat > 0.0) || !(n_crit > 0.0)) {
    return 0;
  }
  int count = 0;
  for (int c = outer_surface_cell; c >= 0; --c) {
    const std::size_t idx = static_cast<std::size_t>(c);
    if (cell_is_void[idx] != 0U) {
      continue;
    }
    const double n_hat = compute_cell_n_hat_approx(rho[idx], zbar[idx], A_eff[idx], n_crit);
    if (n_hat < resolved_nhat) {
      ++count;
    } else {
      break;
    }
  }
  return count;
}

AllowedSupercriticalCell1D find_allowed_supercritical_cell_1d_impl(
    const std::vector<std::uint8_t>& cell_is_void,
    const std::vector<double>& rho,
    const std::vector<double>& zbar,
    const std::vector<double>& A_eff,
    const std::vector<double>& r_edges,
    const double n_crit) {
  const std::size_t n_cells = rho.size();
  if (n_cells == 0) {
    return {};
  }

  std::vector<double> n_hat_cell(n_cells, 0.0);
  int outermost_real = -1;
  bool any_subcritical_real = false;
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (!cell_is_void.empty() && cell_is_void[c] != 0U) {
      continue;
    }
    outermost_real = static_cast<int>(c);
    n_hat_cell[c] = compute_cell_n_hat_approx(rho[c], zbar[c], A_eff[c], n_crit);
    any_subcritical_real = any_subcritical_real || (n_hat_cell[c] < 1.0);
  }
  if (outermost_real < 0) {
    return {};
  }

  const CriticalSurfaceEstimate1D crit_est = estimate_critical_surface_1d(n_hat_cell, r_edges);
  if (crit_est.fcrit > 0 && crit_est.fcrit < static_cast<int>(n_cells) &&
      (cell_is_void.empty() || cell_is_void[static_cast<std::size_t>(crit_est.fcrit)] == 0U) &&
      (cell_is_void.empty() || cell_is_void[static_cast<std::size_t>(crit_est.fcrit - 1)] == 0U) &&
      n_hat_cell[static_cast<std::size_t>(crit_est.fcrit - 1)] >= 1.0 &&
      n_hat_cell[static_cast<std::size_t>(crit_est.fcrit)] < 1.0) {
    return AllowedSupercriticalCell1D{
        crit_est.fcrit - 1, crit_est.fcrit, crit_est.r_interp, false};
  }

  if (any_subcritical_real) {
    return {};
  }

  int fallback = -1;
  for (int c = outermost_real; c >= 0; --c) {
    if (!cell_is_void.empty() && cell_is_void[static_cast<std::size_t>(c)] != 0U) {
      continue;
    }
    fallback = c;
    break;
  }
  if (fallback < 0) {
    return {};
  }
  return AllowedSupercriticalCell1D{fallback, -1, crit_est.r_interp, true};
}

std::vector<std::uint8_t> build_blocked_cell_mask(
    const std::vector<std::uint8_t>& cell_is_void,
    const std::vector<double>& rho,
    const std::vector<double>& zbar,
    const std::vector<double>& A_eff,
    const double n_crit,
    const int allowed_supercritical_cell = -1) {
  const std::size_t n_cells = rho.size();
  std::vector<std::uint8_t> blocked(n_cells, 0U);
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (c < cell_is_void.size() && cell_is_void[c] != 0U) {
      blocked[c] = 1U;
      continue;
    }
    const double n_hat =
        compute_cell_n_hat_approx(rho[c], zbar[c], A_eff[c], n_crit);
    if (n_hat >= 1.0 && static_cast<int>(c) != allowed_supercritical_cell) {
      blocked[c] = 1U;
    }
  }
  return blocked;
}

void distribute_to_stencil_1d(std::vector<double>& dep_power_cell,
                              const std::vector<std::uint8_t>& cell_is_blocked,
                              const std::vector<double>* rho,
                              const std::vector<double>* zbar,
                              const std::vector<double>* A_eff,
                              const double n_crit,
                              const int anchor,
                              const int direction,
                              const double power,
                              const int handoff_cells,
                              const double handoff_decay,
                              const double transition_blend,
                              const double transition_resolved_nhat,
                              const double transition_density_exponent) {
  if (!(power > 0.0) || anchor < 0 || anchor >= static_cast<int>(dep_power_cell.size())) {
    return;
  }

  const int span = std::max(handoff_cells, 1);
  const double decay = std::max(handoff_decay, 1.0e-12);
  const double resolved_nhat_safe = std::max(transition_resolved_nhat, 1.0e-12);
  const double density_exponent = std::max(transition_density_exponent, 0.0);
  double weight_sum = 0.0;
  std::vector<std::pair<int, double>> stencil;
  stencil.reserve(static_cast<std::size_t>(span));

  for (int k = 0; k < span; ++k) {
    const int idx = anchor + direction * k;
    if (idx < 0 || idx >= static_cast<int>(dep_power_cell.size())) {
      break;
    }
    if (cell_is_blocked[static_cast<std::size_t>(idx)] != 0U) {
      continue;
    }
    const double w_base = std::exp(-static_cast<double>(k) / decay);
    double w = w_base;
    if (transition_blend > 0.0) {
      double density_factor = 1.0;
      if (density_exponent > 0.0 && rho != nullptr && zbar != nullptr && A_eff != nullptr &&
          static_cast<std::size_t>(idx) < rho->size() &&
          static_cast<std::size_t>(idx) < zbar->size() &&
          static_cast<std::size_t>(idx) < A_eff->size()) {
        const double n_hat = compute_cell_n_hat_approx((*rho)[static_cast<std::size_t>(idx)],
                                                       (*zbar)[static_cast<std::size_t>(idx)],
                                                       (*A_eff)[static_cast<std::size_t>(idx)],
                                                       n_crit);
        density_factor = std::pow(
            std::clamp(n_hat / resolved_nhat_safe, kGhostTransitionMinWeightFactor,
                       kGhostTransitionMaxWeightFactor),
            density_exponent);
      }
      const double w_transition = w_base * density_factor;
      w = (1.0 - transition_blend) * w_base + transition_blend * w_transition;
    }
    stencil.emplace_back(idx, w);
    weight_sum += w;
  }

  if (!(weight_sum > 0.0)) {
    dep_power_cell[static_cast<std::size_t>(anchor)] += power;
    return;
  }

  for (const auto& [idx, w] : stencil) {
    dep_power_cell[static_cast<std::size_t>(idx)] += power * (w / weight_sum);
  }
}

// Option C layout: arrays are global-size on every rank, so ownership is a
// pure global-index range check (see PartitionInfo in parallel/partition.hpp).
bool is_owned_cell_1d(const int c,
                      const int n_cells,
                      const core::State& state,
                      const parallel::PartitionInfo& part) {
  (void)n_cells;
  (void)state;
  if (part.n_ranks <= 1) {
    return true;
  }
  const int i_global_begin = part.local_cell_range[0][0];
  const int i_global_end = part.local_cell_range[0][1];
  return c >= i_global_begin && c < i_global_end;
}

bool is_owned_cell_2d(const int c,
                      const int n_cells,
                      const core::State& state,
                      const parallel::PartitionInfo& part) {
  (void)n_cells;
  if (part.n_ranks <= 1) {
    return true;
  }
  const int nz_global = std::max(state.mesh.topo.nz, 1);
  return part.owns_cell(c / nz_global, c % nz_global);
}

__global__ void transfer_2d_kernel(double* __restrict__ laser_dep,
                                   const double* __restrict__ deposit,
                                   const double* __restrict__ node_R,
                                   const double* __restrict__ node_Z,
                                   const double* __restrict__ r_cell,
                                   const double* __restrict__ z_cell,
                                   const double dt,
                                   const int n_nodes_r,
                                   const int n_nodes_z,
                                   const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const double rc = r_cell[c];
  const double zc = z_cell[c];
  const double r_min = node_R[0];
  const double r_max = node_R[n_nodes_r - 1];
  const double z_min = node_Z[0];
  const double z_max = node_Z[n_nodes_z - 1];
  const double scale = fmax(1.0, fmax(fabs(r_max), fmax(fabs(z_min), fabs(z_max))));
  const double tol = 1.0e-12 * scale;
  if (rc < r_min - tol || rc > r_max + tol || zc < z_min - tol || zc > z_max + tol) {
    laser_dep[c] = 0.0;
    return;
  }

  const BilinearCell cell =
      BilinearInterp::locate_cell(node_R, node_Z, n_nodes_r, n_nodes_z, rc, zc);
  const BilinearWeights w = BilinearInterp::compute_weights(cell.xi, cell.eta);
  const double p = BilinearInterp::interpolate(deposit, n_nodes_z, cell, w);
  laser_dep[c] = p * dt;
}

}  // namespace

AllowedSupercriticalCell1D find_allowed_supercritical_cell_1d(
    const std::vector<std::uint8_t>& cell_is_void,
    const std::vector<double>& rho,
    const std::vector<double>& zbar,
    const std::vector<double>& A_eff,
    const std::vector<double>& r_edges,
    const double n_crit) {
  return find_allowed_supercritical_cell_1d_impl(cell_is_void, rho, zbar, A_eff, r_edges,
                                                 n_crit);
}

static void apply_deposit_redistribution_2d_host(
    std::vector<double>& dep_power_cell,
    const std::vector<std::uint8_t>& cell_is_void,
    const int nr,
    const int nz,
    const int smooth_passes,
    const double smooth_alpha) {
  TENRYU_ASSERT(smooth_passes >= 0,
                "apply_deposit_redistribution_2d_host smooth_passes must be >= 0");
  TENRYU_ASSERT(smooth_alpha >= 0.0 && smooth_alpha <= 0.5,
                "apply_deposit_redistribution_2d_host smooth_alpha must be in [0, 0.5]");
  TENRYU_ASSERT(nr >= 0 && nz >= 0,
                "apply_deposit_redistribution_2d_host mesh shape must be non-negative");
  const std::int64_t expected =
      static_cast<std::int64_t>(nr) * static_cast<std::int64_t>(nz);
  TENRYU_ASSERT(expected >= 0 &&
                    static_cast<std::size_t>(expected) == dep_power_cell.size(),
                "apply_deposit_redistribution_2d_host deposit size mismatch");
  if (!cell_is_void.empty()) {
    TENRYU_ASSERT(cell_is_void.size() == dep_power_cell.size(),
                  "apply_deposit_redistribution_2d_host cell_is_void size mismatch");
  }
  if (smooth_passes <= 0 || !(smooth_alpha > 0.0) || nr <= 2 || nz <= 2) {
    return;
  }

  const auto is_void = [&](const int c) {
    return !cell_is_void.empty() && cell_is_void[static_cast<std::size_t>(c)] != 0U;
  };

  std::vector<std::uint8_t> cell_is_smoothable(dep_power_cell.size(), 0U);
  for (int i = 1; i < nr - 1; ++i) {
    for (int j = 1; j < nz - 1; ++j) {
      const int c = i * nz + j;
      if (is_void(c) || is_void(c - nz) || is_void(c + nz) ||
          is_void(c - 1) || is_void(c + 1)) {
        continue;
      }
      cell_is_smoothable[static_cast<std::size_t>(c)] = 1U;
    }
  }

  std::vector<double> dep_power_next = dep_power_cell;
  for (int pass = 0; pass < smooth_passes; ++pass) {
    dep_power_next = dep_power_cell;
    for (int i = 0; i + 1 < nr; ++i) {
      for (int j = 0; j < nz; ++j) {
        const int c0 = i * nz + j;
        const int c1 = c0 + nz;
        const std::size_t idx0 = static_cast<std::size_t>(c0);
        const std::size_t idx1 = static_cast<std::size_t>(c1);
        if (cell_is_smoothable[idx0] == 0U || cell_is_smoothable[idx1] == 0U) {
          continue;
        }
        const double flux = smooth_alpha * (dep_power_cell[idx1] - dep_power_cell[idx0]);
        dep_power_next[idx0] += flux;
        dep_power_next[idx1] -= flux;
      }
    }
    for (int i = 0; i < nr; ++i) {
      for (int j = 0; j + 1 < nz; ++j) {
        const int c0 = i * nz + j;
        const int c1 = c0 + 1;
        const std::size_t idx0 = static_cast<std::size_t>(c0);
        const std::size_t idx1 = static_cast<std::size_t>(c1);
        if (cell_is_smoothable[idx0] == 0U || cell_is_smoothable[idx1] == 0U) {
          continue;
        }
        const double flux = smooth_alpha * (dep_power_cell[idx1] - dep_power_cell[idx0]);
        dep_power_next[idx0] += flux;
        dep_power_next[idx1] -= flux;
      }
    }
    dep_power_cell.swap(dep_power_next);
  }
}

void apply_deposit_redistribution_1d(
    core::State& state,
    LaserMesh& mesh,
    const HydroMirror1D& hydro,
    const std::vector<double>& deposit_power_cell,
    const double dt,
    const double conservation_tol,
    const parallel::PartitionInfo& part,
    const int smooth_passes,
    const double smooth_alpha,
    const std::vector<double>* hot_e_extra_power) {
  TENRYU_ASSERT(state.mesh.dim == 1, "apply_deposit_redistribution_1d expects 1D_SPH state");
  TENRYU_ASSERT(smooth_passes >= 0,
                "apply_deposit_redistribution_1d smooth_passes must be >= 0");
  TENRYU_ASSERT(smooth_alpha >= 0.0 && smooth_alpha <= 0.5,
                "apply_deposit_redistribution_1d smooth_alpha must be in [0, 0.5]");
  if (!(dt > 0.0)) {
    mesh.last_ghost_transition_blend = 0.0;
    mesh.last_ghost_transition_resolved_cells = 0;
    mesh.last_transfer_blocked_power = 0.0;
    state.laser_dep.fill(0.0);
    return;
  }

  const std::vector<double>& r_edges = hydro.r_edges;
  const std::vector<double>& rho = hydro.rho;
  const std::vector<double>& zbar = hydro.zbar;
  const std::vector<double>& A_eff = hydro.A_eff;
  const std::vector<std::uint8_t>& cell_is_void =
      hydro.cell_is_void.empty() ? state.cell_is_void : hydro.cell_is_void;
  TENRYU_ASSERT(deposit_power_cell.size() == state.laser_dep.size(),
                "apply_deposit_redistribution_1d deposit size mismatch");
  TENRYU_ASSERT(r_edges.size() == state.laser_dep.size() + 1,
                "apply_deposit_redistribution_1d r_edges size mismatch");
  if (!cell_is_void.empty()) {
    TENRYU_ASSERT(cell_is_void.size() == state.laser_dep.size(),
                  "apply_deposit_redistribution_1d cell_is_void size mismatch");
  }
  if (!rho.empty() || !zbar.empty() || !A_eff.empty()) {
    TENRYU_ASSERT(rho.size() == state.laser_dep.size(),
                  "apply_deposit_redistribution_1d rho size mismatch");
    TENRYU_ASSERT(zbar.size() == state.laser_dep.size(),
                  "apply_deposit_redistribution_1d zbar size mismatch");
    TENRYU_ASSERT(A_eff.size() == state.laser_dep.size(),
                  "apply_deposit_redistribution_1d A_eff size mismatch");
  }

  mesh.last_ghost_transition_blend = 0.0;
  mesh.last_ghost_transition_resolved_cells = 0;
  mesh.last_transfer_blocked_power = 0.0;
  double transition_blend = 0.0;
  std::vector<double> dep_power_cell = deposit_power_cell;
  std::vector<std::uint8_t> cell_is_blocked(dep_power_cell.size(), 0U);
  std::vector<std::uint8_t> cell_is_subcritical_receiver(dep_power_cell.size(), 0U);
    AllowedSupercriticalCell1D allowed_supercritical;
  if (rho.size() == dep_power_cell.size() && zbar.size() == dep_power_cell.size() &&
      A_eff.size() == dep_power_cell.size()) {
    allowed_supercritical =
        find_allowed_supercritical_cell_1d_impl(cell_is_void, rho, zbar, A_eff, r_edges,
                                                mesh.n_crit);
    cell_is_blocked =
        build_blocked_cell_mask(cell_is_void, rho, zbar, A_eff, mesh.n_crit,
                                allowed_supercritical.allowed_cell);
    cell_is_subcritical_receiver =
        build_blocked_cell_mask(cell_is_void, rho, zbar, A_eff, mesh.n_crit);
  } else if (!cell_is_void.empty()) {
    cell_is_blocked = cell_is_void;
    cell_is_subcritical_receiver = cell_is_void;
  }
  if (mesh.ghost_corona_enabled && mesh.ghost_transition_enabled && !cell_is_void.empty()) {
    const int outer_surface_cell = find_outer_surface_cell_1d(cell_is_void);
    const int resolved_cells = count_resolved_subcritical_cells_1d(
        cell_is_void, rho, zbar, A_eff, outer_surface_cell,
        mesh.ghost_transition_resolved_nhat, mesh.n_crit);
    mesh.last_ghost_transition_resolved_cells = resolved_cells;
    const double required_cells =
        static_cast<double>(std::max(mesh.ghost_transition_resolved_cells, 1));
    transition_blend =
        std::clamp(1.0 - static_cast<double>(resolved_cells) / required_cells, 0.0, 1.0);
    mesh.last_ghost_transition_blend = transition_blend;
  }

  long double sum_input = 0.0L;
  for (const double p : deposit_power_cell) {
    sum_input += static_cast<long double>(p);
  }

  const int n_cells = static_cast<int>(dep_power_cell.size());
  if (!allowed_supercritical.fallback_only &&
      allowed_supercritical.critical_adjacent_subcritical_cell >= 0 &&
      allowed_supercritical.critical_adjacent_subcritical_cell < n_cells &&
      transition_blend > 0.0) {
    const int anchor = allowed_supercritical.critical_adjacent_subcritical_cell;
    const double power = dep_power_cell[static_cast<std::size_t>(anchor)];
    if (power > 0.0) {
      dep_power_cell[static_cast<std::size_t>(anchor)] = 0.0;
      distribute_to_stencil_1d(dep_power_cell, cell_is_blocked,
                               nullptr, nullptr, nullptr, mesh.n_crit, anchor, -1, power,
                               mesh.ghost_handoff_cells, mesh.ghost_handoff_decay,
                               /*transition_blend=*/0.0,
                               mesh.ghost_transition_resolved_nhat,
                               mesh.ghost_transition_density_exponent);
    }
  }

  // Blocked receivers (void or supercritical) cannot couple directly to Hydro.
  // Void-side power uses the existing ghost-corona handoff, while power mapped
  // to supercritical real cells is pushed to the nearest outward subcritical
  // receiver when one exists.
  long double blocked_power = 0.0L;
  if (!cell_is_blocked.empty()) {
    for (int c = 0; c < n_cells; ++c) {
      if (cell_is_blocked[static_cast<std::size_t>(c)] == 0U) {
        continue;
      }
      if (!(dep_power_cell[static_cast<std::size_t>(c)] > 0.0)) {
        continue;
      }
      const bool is_void_source =
          !cell_is_void.empty() &&
          cell_is_void[static_cast<std::size_t>(c)] != 0U;
      const auto& anchor_mask =
          allowed_supercritical.fallback_only ? cell_is_blocked : cell_is_subcritical_receiver;
      int direction = 0;
      const int target = find_active_anchor_1d(
          anchor_mask, c, direction,
          /*prefer_outward=*/!is_void_source,
          /*allow_opposite_fallback=*/is_void_source);
      if (target >= 0) {
        const double power = dep_power_cell[static_cast<std::size_t>(c)];
        if (is_void_source && mesh.ghost_corona_enabled) {
          const std::vector<double>* rho_ptr = transition_blend > 0.0 ? &rho : nullptr;
          const std::vector<double>* zbar_ptr = transition_blend > 0.0 ? &zbar : nullptr;
          const std::vector<double>* A_ptr = transition_blend > 0.0 ? &A_eff : nullptr;
          distribute_to_stencil_1d(dep_power_cell, cell_is_blocked, rho_ptr, zbar_ptr,
                                   A_ptr, mesh.n_crit, target, direction, power,
                                   mesh.ghost_handoff_cells, mesh.ghost_handoff_decay,
                                   transition_blend, mesh.ghost_transition_resolved_nhat,
                                   mesh.ghost_transition_density_exponent);
        } else {
          dep_power_cell[static_cast<std::size_t>(target)] += power;
        }
      } else {
        blocked_power += static_cast<long double>(dep_power_cell[static_cast<std::size_t>(c)]);
      }
      dep_power_cell[static_cast<std::size_t>(c)] = 0.0;
    }
  }

  if (smooth_passes > 0 && smooth_alpha > 0.0 && n_cells > 2) {
    const auto is_smoothable = [&](const int c) {
      if (c <= 0 || c >= n_cells - 1) {
        return false;
      }
      const std::size_t idx = static_cast<std::size_t>(c);
      if (!cell_is_void.empty() && cell_is_void[idx] != 0U) {
        return false;
      }
      if (!cell_is_blocked.empty() && cell_is_blocked[idx] != 0U) {
        return false;
      }
      return true;
    };
    const auto is_blocked_or_void = [&](const int c) {
      if (c < 0 || c >= n_cells) {
        return false;
      }
      const std::size_t idx = static_cast<std::size_t>(c);
      if (!cell_is_void.empty() && cell_is_void[idx] != 0U) {
        return true;
      }
      if (!cell_is_blocked.empty() && cell_is_blocked[idx] != 0U) {
        return true;
      }
      return false;
    };

    long double sum_before_smoothing = 0.0L;
    for (const double p : dep_power_cell) {
      sum_before_smoothing += static_cast<long double>(p);
    }

    const std::vector<double> cell_mass =
        load_cell_mass_1d(state, rho, static_cast<std::size_t>(n_cells));
    std::vector<std::uint8_t> cell_is_smoothable(static_cast<std::size_t>(n_cells), 0U);
    std::vector<std::uint8_t> cell_is_guarded(static_cast<std::size_t>(n_cells), 0U);
    for (int c = 0; c < n_cells; ++c) {
      if (!is_smoothable(c) || !(cell_mass[static_cast<std::size_t>(c)] > 0.0)) {
        continue;
      }
      cell_is_smoothable[static_cast<std::size_t>(c)] = 1U;
      bool guarded = false;
      for (int d = 1; d <= kDepositSmoothBoundaryGuardCells; ++d) {
        if (is_blocked_or_void(c - d) || is_blocked_or_void(c + d)) {
          guarded = true;
          break;
        }
      }
      if (guarded) {
        cell_is_guarded[static_cast<std::size_t>(c)] = 1U;
      }
    }

    std::vector<double> dep_power_next = dep_power_cell;
    for (int pass = 0; pass < smooth_passes; ++pass) {
      dep_power_next = dep_power_cell;
      for (int c = 1; c < n_cells - 1; ++c) {
        const int left = c;
        const int right = c + 1;
        if (right >= n_cells) {
          continue;
        }
        const std::size_t left_idx = static_cast<std::size_t>(left);
        const std::size_t right_idx = static_cast<std::size_t>(right);
        if (cell_is_smoothable[left_idx] == 0U || cell_is_smoothable[right_idx] == 0U) {
          continue;
        }
        if (cell_is_guarded[left_idx] != 0U || cell_is_guarded[right_idx] != 0U) {
          continue;
        }

        const double mass_left = cell_mass[left_idx];
        const double mass_right = cell_mass[right_idx];
        if (!(mass_left > 0.0) || !(mass_right > 0.0)) {
          continue;
        }
        const double specific_left = dep_power_cell[left_idx] / mass_left;
        const double specific_right = dep_power_cell[right_idx] / mass_right;
        const double face_mass = std::min(mass_left, mass_right);
        if (!(face_mass > 0.0)) {
          continue;
        }
        const double flux = smooth_alpha * face_mass * (specific_right - specific_left);
        dep_power_next[left_idx] += flux;
        dep_power_next[right_idx] -= flux;
      }
      dep_power_cell.swap(dep_power_next);
    }

    long double sum_after_smoothing = 0.0L;
    for (const double p : dep_power_cell) {
      sum_after_smoothing += static_cast<long double>(p);
    }
    const double smoothing_denom =
        std::max(std::abs(static_cast<double>(sum_before_smoothing)), 1.0e-30);
    const double smoothing_rel =
        std::abs(static_cast<double>(sum_before_smoothing - sum_after_smoothing)) /
        smoothing_denom;
    if (std::abs(static_cast<double>(sum_before_smoothing)) > 1.0e-20 &&
        smoothing_rel > conservation_tol) {
      core::log_warning("Laser deposit smoothing conservation check failed: rel=" +
                        std::to_string(smoothing_rel));
    }
  }

  long double hot_e_extra_sum = 0.0L;
  if (hot_e_extra_power != nullptr) {
    TENRYU_ASSERT(hot_e_extra_power->size() == dep_power_cell.size(),
                  "apply_deposit_redistribution_1d hot_e_extra_power size mismatch");
    for (std::size_t c = 0; c < dep_power_cell.size(); ++c) {
      const double extra = (*hot_e_extra_power)[c];
      if (extra != 0.0) {
        dep_power_cell[c] += extra;
        hot_e_extra_sum += static_cast<long double>(extra);
      }
    }
  }

  long double sum_cells = 0.0L;
  std::vector<double> dep_energy(state.laser_dep.size(), 0.0);
  for (std::size_t c = 0; c < dep_power_cell.size(); ++c) {
    sum_cells += static_cast<long double>(dep_power_cell[c]);
    dep_energy[c] = dep_power_cell[c] * dt;
    if (part.n_ranks > 1 &&
        !is_owned_cell_1d(static_cast<int>(c), n_cells, state, part)) {
      dep_energy[c] = 0.0;
    }
  }

  const long double sum_expected = sum_input + hot_e_extra_sum;
  const double denom = std::max(std::abs(static_cast<double>(sum_expected)), 1.0e-30);
  const double rel =
      std::abs(static_cast<double>(sum_expected - (sum_cells + blocked_power))) / denom;
  mesh.last_transfer_blocked_power = std::max(0.0, static_cast<double>(blocked_power));
  if (std::abs(static_cast<double>(sum_input)) > 1.0e-20 && rel > conservation_tol) {
    core::log_warning("Laser transfer conservation check failed: rel=" +
                      std::to_string(rel));
  }

  state.laser_dep.copy_from_host(dep_energy.data());
}

void transfer_to_1d(core::State& state,
                    LaserMesh& mesh,
                    const HydroMirror1D& hydro,
                    const std::vector<double>& deposit_lm,
                    const std::vector<double>& node_R,
                    const std::vector<double>& node_Z,
                    const double dt,
                    const double conservation_tol,
                    const parallel::PartitionInfo& part,
                    const int smooth_passes,
                    const double smooth_alpha) {
  TENRYU_ASSERT(state.mesh.dim == 1, "transfer_to_1d expects 1D_SPH state");
  TENRYU_ASSERT(node_R.size() == static_cast<std::size_t>(mesh.n_nodes_r),
                "transfer_to_1d node_R size mismatch");
  TENRYU_ASSERT(node_Z.size() == static_cast<std::size_t>(mesh.n_nodes_z),
                "transfer_to_1d node_Z size mismatch");
  TENRYU_ASSERT(deposit_lm.size() == static_cast<std::size_t>(mesh.n_nodes()),
                "transfer_to_1d deposit size mismatch");

  std::vector<double> dep_power_cell(state.laser_dep.size(), 0.0);
  const std::vector<double>& r_edges = hydro.r_edges;
  const std::vector<double>& rho = hydro.rho;
  const std::vector<double>& zbar = hydro.zbar;
  const std::vector<double>& A_eff = hydro.A_eff;
  const std::vector<std::uint8_t>& cell_is_void =
      hydro.cell_is_void.empty() ? state.cell_is_void : hydro.cell_is_void;
  const AllowedSupercriticalCell1D allowed_supercritical =
      (rho.size() == dep_power_cell.size() && zbar.size() == dep_power_cell.size() &&
       A_eff.size() == dep_power_cell.size())
          ? find_allowed_supercritical_cell_1d_impl(cell_is_void, rho, zbar, A_eff, r_edges,
                                                    mesh.n_crit)
          : AllowedSupercriticalCell1D{};
  for (int i = 0; i < mesh.n_nodes_r; ++i) {
    const double R = node_R[static_cast<std::size_t>(i)];
    for (int j = 0; j < mesh.n_nodes_z; ++j) {
      const double Z = node_Z[static_cast<std::size_t>(j)];
      const int n = mesh.node_index(i, j);
      const double p = deposit_lm[static_cast<std::size_t>(n)];
      const double r = std::sqrt(R * R + Z * Z);
      int c = locate_cell_1d(r_edges, r);
      if (allowed_supercritical.critical_adjacent_subcritical_cell == c &&
          allowed_supercritical.allowed_cell == c - 1 &&
          allowed_supercritical.r_crit > r_edges[static_cast<std::size_t>(c)] &&
          allowed_supercritical.r_crit < r_edges[static_cast<std::size_t>(c + 1)] &&
          r < allowed_supercritical.r_crit) {
        c = allowed_supercritical.allowed_cell;
      }
      dep_power_cell[static_cast<std::size_t>(c)] += p;
    }
  }

  apply_deposit_redistribution_1d(state, mesh, hydro, dep_power_cell, dt, conservation_tol,
                                  part, smooth_passes, smooth_alpha);
}

void transfer_to_1d(core::State& state,
                    LaserMesh& mesh,
                    const double dt,
                    const double conservation_tol,
                    cudaStream_t stream,
                    const parallel::PartitionInfo& part,
                    const int smooth_passes,
                    const double smooth_alpha) {
  TENRYU_ASSERT(state.mesh.dim == 1, "transfer_to_1d expects 1D_SPH state");
  if (!(dt > 0.0)) {
    mesh.last_ghost_transition_blend = 0.0;
    mesh.last_ghost_transition_resolved_cells = 0;
    mesh.last_transfer_blocked_power = 0.0;
    state.laser_dep.fill(0.0);
    return;
  }

  std::vector<double> node_R(static_cast<std::size_t>(mesh.n_nodes_r), 0.0);
  std::vector<double> node_Z(static_cast<std::size_t>(mesh.n_nodes_z), 0.0);
  std::vector<double> deposit_lm(static_cast<std::size_t>(mesh.n_nodes()), 0.0);
  cuda_check(cudaMemcpyAsync(node_R.data(), mesh.node_R, node_R.size() * sizeof(double),
                             cudaMemcpyDeviceToHost, stream),
             "transfer_to_1d memcpyAsync node_R failed");
  cuda_check(cudaMemcpyAsync(node_Z.data(), mesh.node_Z, node_Z.size() * sizeof(double),
                             cudaMemcpyDeviceToHost, stream),
             "transfer_to_1d memcpyAsync node_Z failed");
  cuda_check(cudaMemcpyAsync(deposit_lm.data(), mesh.deposit,
                             deposit_lm.size() * sizeof(double), cudaMemcpyDeviceToHost,
                             stream),
             "transfer_to_1d memcpyAsync deposit failed");
  cuda_check(cudaStreamSynchronize(stream), "transfer_to_1d stream synchronize failed");

  HydroMirror1D hydro;
  build_hydro_mirror_1d(mesh, state, hydro);
  transfer_to_1d(state, mesh, hydro, deposit_lm, node_R, node_Z, dt, conservation_tol, part,
                 smooth_passes, smooth_alpha);
}

void transfer_to_2d(core::State& state,
                    LaserMesh& mesh,
                    const double dt,
                    const double conservation_tol,
                    cudaStream_t stream,
                    double* applied_scale,
                    const parallel::PartitionInfo& part,
                    const int smooth_passes,
                    const double smooth_alpha,
                    const std::vector<double>* hot_e_power_cell,
                    const parallel::Reduction* reduction) {
  TENRYU_ASSERT(state.mesh.dim == 2, "transfer_to_2d expects 2D_RZ state");
  TENRYU_ASSERT(smooth_passes >= 0, "transfer_to_2d smooth_passes must be >= 0");
  TENRYU_ASSERT(smooth_alpha >= 0.0 && smooth_alpha <= 0.5,
                "transfer_to_2d smooth_alpha must be in [0, 0.5]");
  const int n_cells = state.mesh.topo.n_cells;
  double scale_applied = 1.0;
  mesh.last_transfer_blocked_power = 0.0;
  if (n_cells <= 0 || !(dt > 0.0)) {
    state.laser_dep.fill(0.0);
    if (applied_scale != nullptr) {
      *applied_scale = scale_applied;
    }
    return;
  }

  double* d_r_cell = nullptr;
  double* d_z_cell = nullptr;
  const std::size_t bytes = static_cast<std::size_t>(n_cells) * sizeof(double);
  std::vector<double> dep_lm_pre(static_cast<std::size_t>(mesh.n_nodes()), 0.0);
  cuda_check(cudaMemcpyAsync(dep_lm_pre.data(), mesh.deposit,
                             dep_lm_pre.size() * sizeof(double), cudaMemcpyDeviceToHost,
                             stream),
             "transfer_to_2d memcpyAsync deposit LM pre failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_r_cell), bytes),
             "transfer_to_2d cudaMalloc d_r_cell failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_z_cell), bytes),
             "transfer_to_2d cudaMalloc d_z_cell failed");
  cuda_check(cudaMemcpyAsync(d_r_cell, state.mesh.cell_centroid_r.data(), bytes,
                             cudaMemcpyHostToDevice, stream),
             "transfer_to_2d memcpyAsync r_cell failed");
  cuda_check(cudaMemcpyAsync(d_z_cell, state.mesh.cell_centroid_z.data(), bytes,
                             cudaMemcpyHostToDevice, stream),
             "transfer_to_2d memcpyAsync z_cell failed");

  const int block = 256;
  const int grid = (n_cells + block - 1) / block;
  transfer_2d_kernel<<<grid, block, 0, stream>>>(state.laser_dep.data(), mesh.deposit, mesh.node_R,
                                                  mesh.node_Z, d_r_cell, d_z_cell, dt,
                                                  mesh.n_nodes_r, mesh.n_nodes_z, n_cells);
  cuda_check(cudaGetLastError(), "transfer_to_2d kernel launch failed");

  std::vector<double> dep_hm(static_cast<std::size_t>(n_cells), 0.0);
  cuda_check(cudaMemcpyAsync(dep_hm.data(), state.laser_dep.data(), dep_hm.size() * sizeof(double),
                             cudaMemcpyDeviceToHost, stream),
             "transfer_to_2d memcpyAsync laser_dep HM failed");
  cuda_check(cudaStreamSynchronize(stream), "transfer_to_2d stream synchronize failed");

  cuda_check(cudaFree(d_z_cell), "transfer_to_2d cudaFree d_z_cell failed");
  cuda_check(cudaFree(d_r_cell), "transfer_to_2d cudaFree d_r_cell failed");

  // MPI (Option C, OPEN-LASER-FAR): the LM->hydro binning uses the
  // per-rank HOST centroids, which are stale outside owned+ghost, so the
  // far-region binned deposit differs per rank. An owned-partial sum +
  // Allreduce made the conservation rescale rank-UNIFORM but not
  // rank-count-INVARIANT: the MPI summation order differs from the
  // serial loop by ulps, and the ulp-scaled deposit seeds the ablation
  // front's exponential absorption feedback (measured 1.2e-4 ee drift at
  // 10 steps from a bitwise-clean step 1). Instead, Allgatherv the OWNED
  // deposit windows so every rank holds the owner-true full array, then
  // compute the rescale with the SAME canonical full-array loop as the
  // serial path -- the factor is bitwise rank-count-invariant, and the
  // smoothing/blocking passes below see owner-true data everywhere (the
  // former ghost-ring zeroing and smoothing-reach cap are obsolete).
  const bool xfer_mpi = part.n_ranks > 1;
  if (xfer_mpi) {
    TENRYU_ASSERT(reduction != nullptr,
                  "transfer_to_2d requires a reducer under MPI (a rank-"
                  "local conservation rescale drifts the deposit)");
    TENRYU_ASSERT(part.cart_dims[1] == 1,
                  "transfer_to_2d deposit allgatherv supports r-slab "
                  "decompositions only in v1");
    const int nz_h = std::max(state.mesh.topo.nz, 1);
    TENRYU_ASSERT(n_cells == part.global_nr * nz_h,
                  "transfer_to_2d deposit allgatherv requires the "
                  "structured c = i*nz + j layout");
    const int n_ranks = part.n_ranks;
    std::vector<int> counts(static_cast<std::size_t>(n_ranks));
    std::vector<int> displs(static_cast<std::size_t>(n_ranks));
    const int base = part.global_nr / n_ranks;
    const int rem = part.global_nr % n_ranks;
    for (int p = 0; p < n_ranks; ++p) {
      counts[static_cast<std::size_t>(p)] =
          (base + (p < rem ? 1 : 0)) * nz_h;
      displs[static_cast<std::size_t>(p)] =
          (p * base + std::min(p, rem)) * nz_h;
    }
    const int my_count = counts[static_cast<std::size_t>(part.rank)];
    const int my_displ = displs[static_cast<std::size_t>(part.rank)];
    std::vector<double> send(dep_hm.begin() + my_displ,
                             dep_hm.begin() + my_displ + my_count);
    reduction->allgatherv(send.data(), my_count, dep_hm.data(),
                          counts.data(), displs.data());
  }
  long double sum_lm = 0.0L;
  for (const double p : dep_lm_pre) {
    sum_lm += static_cast<long double>(p);
  }
  long double sum_hm_power = 0.0L;
  for (int c = 0; c < n_cells; ++c) {
    sum_hm_power +=
        static_cast<long double>(dep_hm[static_cast<std::size_t>(c)] / dt);
  }

  double sum_lm_d = static_cast<double>(sum_lm);
  double sum_hm_power_pre_rescale_d = static_cast<double>(sum_hm_power);
  if (xfer_mpi) {
    // Broadcast rank 0's sums (allreduce over {value, 0, ...} adds exact
    // zeros, so every rank gets rank 0's bits): dep_hm is gathered
    // owner-true above, but the LM-side numerator comes from per-rank GPU
    // atomics and is not guaranteed bitwise across ranks. Rank 0's
    // canonical factor is both rank-uniform and equal to the serial one.
    sum_lm_d = reduction->allreduce_sum(part.rank == 0 ? sum_lm_d : 0.0);
    sum_hm_power_pre_rescale_d = reduction->allreduce_sum(
        part.rank == 0 ? sum_hm_power_pre_rescale_d : 0.0);
  }
  bool scaled = false;
  if (std::abs(sum_lm_d) > 1.0e-20 &&
      std::abs(sum_hm_power_pre_rescale_d) > 1.0e-30) {
    const double scale = sum_lm_d / sum_hm_power_pre_rescale_d;
    if (std::isfinite(scale) && scale > 0.0) {
      scale_applied = scale;
      for (double& e : dep_hm) {
        e *= scale;
      }
      scaled = true;
      sum_hm_power = 0.0L;
      for (int c = 0; c < n_cells; ++c) {
        sum_hm_power += static_cast<long double>(
            dep_hm[static_cast<std::size_t>(c)] / dt);
      }
    }
  }
  const double sum_hm_power_post_rescale_d = static_cast<double>(sum_hm_power);

  long double blocked_power = 0.0L;
  bool filtered = false;
  const bool smoothing_requested = smooth_passes > 0 && smooth_alpha > 0.0 && n_cells > 2;
  std::vector<std::uint8_t> smooth_guard;
  if (smoothing_requested) {
    smooth_guard = state.cell_is_void;
    if (smooth_guard.empty()) {
      smooth_guard.assign(dep_hm.size(), 0U);
    } else {
      TENRYU_ASSERT(smooth_guard.size() == dep_hm.size(),
                    "transfer_to_2d cell_is_void size mismatch");
    }
  }
  if (state.rho.size() == dep_hm.size() &&
      state.zbar.size() == dep_hm.size()) {
    std::vector<double> rho(state.rho.size(), 0.0);
    std::vector<double> zbar(state.zbar.size(), 0.0);
    state.rho.copy_to_host(rho.data());
    state.zbar.copy_to_host(zbar.data());
    const auto A_eff = compute_cell_effective_A(mesh, state, dep_hm.size());
    const auto cell_is_blocked =
        build_blocked_cell_mask(state.cell_is_void, rho, zbar, A_eff, mesh.n_crit);
    for (std::size_t c = 0; c < dep_hm.size(); ++c) {
      if (smoothing_requested && cell_is_blocked[c] != 0U) {
        smooth_guard[c] = 1U;
      }
      if (cell_is_blocked[c] == 0U || !(dep_hm[c] > 0.0)) {
        continue;
      }
      blocked_power += static_cast<long double>(dep_hm[c] / dt);
      dep_hm[c] = 0.0;
      filtered = true;
    }
  }

  bool smoothed = false;
  if (smoothing_requested) {
    long double sum_before_smoothing = 0.0L;
    for (const double e : dep_hm) {
      sum_before_smoothing += static_cast<long double>(e);
    }

    apply_deposit_redistribution_2d_host(dep_hm, smooth_guard, state.mesh.topo.nr,
                                         state.mesh.topo.nz, smooth_passes, smooth_alpha);
    smoothed = true;

    long double sum_after_smoothing = 0.0L;
    for (const double e : dep_hm) {
      sum_after_smoothing += static_cast<long double>(e);
    }
    const double smoothing_denom =
        std::max(std::abs(static_cast<double>(sum_before_smoothing)), 1.0e-30);
    const double smoothing_rel =
        std::abs(static_cast<double>(sum_before_smoothing - sum_after_smoothing)) /
        smoothing_denom;
    if (std::abs(static_cast<double>(sum_before_smoothing)) > 1.0e-20 &&
        smoothing_rel > conservation_tol) {
      core::log_warning("Laser deposit smoothing(2D) conservation check failed: rel=" +
                        std::to_string(smoothing_rel));
    }
  }

  sum_hm_power = 0.0L;
  for (const double e : dep_hm) {
    sum_hm_power += static_cast<long double>(e / dt);
  }

  const double denom = std::max(std::abs(sum_lm_d), 1.0e-30);
  const double rel =
      std::abs(static_cast<double>(sum_lm - (sum_hm_power + blocked_power))) / denom;
  mesh.last_transfer_blocked_power = std::max(0.0, static_cast<double>(blocked_power));
  if (std::abs(sum_lm_d) > 1.0e-20 && rel > conservation_tol) {
    std::ostringstream msg;
    msg << std::setprecision(17)
        << "Laser transfer(2D) conservation check failed: rel=" << rel
        << " sum_lm=" << sum_lm_d
        << " sum_hm_power_pre_rescale=" << sum_hm_power_pre_rescale_d
        << " sum_hm_power_post_rescale=" << sum_hm_power_post_rescale_d
        << " sum_hm_power_final=" << static_cast<double>(sum_hm_power)
        << " blocked_power=" << static_cast<double>(blocked_power);
    core::log_warning(msg.str());

    std::vector<double> dep_hm_kernel(static_cast<std::size_t>(n_cells), 0.0);
    std::vector<double> node_R(static_cast<std::size_t>(mesh.n_nodes_r), 0.0);
    std::vector<double> node_Z(static_cast<std::size_t>(mesh.n_nodes_z), 0.0);
    cuda_check(cudaMemcpyAsync(dep_hm_kernel.data(), state.laser_dep.data(),
                               dep_hm_kernel.size() * sizeof(double), cudaMemcpyDeviceToHost,
                               stream),
               "transfer_to_2d detail diag laser_dep D2H failed");
    cuda_check(cudaMemcpyAsync(node_R.data(), mesh.node_R, node_R.size() * sizeof(double),
                               cudaMemcpyDeviceToHost, stream),
               "transfer_to_2d detail diag node_R D2H failed");
    cuda_check(cudaMemcpyAsync(node_Z.data(), mesh.node_Z, node_Z.size() * sizeof(double),
                               cudaMemcpyDeviceToHost, stream),
               "transfer_to_2d detail diag node_Z D2H failed");
    cuda_check(cudaStreamSynchronize(stream),
               "transfer_to_2d detail diag stream synchronize failed");
    log_transfer_2d_detail_diag(dep_lm_pre, dep_hm_kernel, node_R, node_Z,
                                state.mesh.cell_centroid_r, state.mesh.cell_centroid_z, dt,
                                mesh.n_nodes_r, mesh.n_nodes_z);
  }
  if (hot_e_power_cell != nullptr) {
    TENRYU_ASSERT(hot_e_power_cell->size() == dep_hm.size(),
                  "transfer_to_2d hot_e_power_cell size mismatch");
    for (std::size_t c = 0; c < dep_hm.size(); ++c) {
      dep_hm[c] += (*hot_e_power_cell)[c] * dt;
    }
  }
  if (part.n_ranks > 1) {
    for (int c = 0; c < n_cells; ++c) {
      if (!is_owned_cell_2d(c, n_cells, state, part)) {
        dep_hm[static_cast<std::size_t>(c)] = 0.0;
      }
    }
  }
  if (scaled || filtered || smoothed || part.n_ranks > 1 || hot_e_power_cell != nullptr) {
    cuda_check(cudaMemcpyAsync(state.laser_dep.data(), dep_hm.data(),
                               dep_hm.size() * sizeof(double), cudaMemcpyHostToDevice, stream),
               "transfer_to_2d memcpyAsync laser_dep H2D failed");
    cuda_check(cudaStreamSynchronize(stream),
               "transfer_to_2d stream synchronize after H2D writeback failed");
  }
  if (applied_scale != nullptr) {
    *applied_scale = scale_applied;
  }
}

}  // namespace tenryu::laser
