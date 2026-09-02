#include "hydro/evacuated_cell_shadow.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/config.hpp"
#include "core/constants.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/axis_edge_collapse.hpp"
#include "hydro/cavity_contact_shadow_solve.hpp"
#include "hydro/corner_jacobian_quality.cuh"
#include "hydro/flank_tangential_strip.hpp"

namespace tenryu::hydro {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

std::array<int, 4> structured_cell_nodes(const int cell, const int nz) {
  const int i = cell / nz;
  const int j = cell - i * nz;
  return {i * (nz + 1) + j,
          (i + 1) * (nz + 1) + j,
          (i + 1) * (nz + 1) + j + 1,
          i * (nz + 1) + j + 1};
}

struct NodeCellCorner {
  int cell = -1;
  int corner = -1;
};

std::vector<NodeCellCorner> structured_node_cell_corners(const int node,
                                                         const int nr,
                                                         const int nz) {
  const int ni = node / (nz + 1);
  const int nj = node - ni * (nz + 1);
  TENRYU_ASSERT(node >= 0 && ni <= nr && nj <= nz,
                "evacuated-cell contact node index mismatch");
  std::vector<NodeCellCorner> adjacent;
  adjacent.reserve(4);
  if (ni < nr && nj < nz) {
    adjacent.push_back({ni * nz + nj, 0});
  }
  if (ni > 0 && nj < nz) {
    adjacent.push_back({(ni - 1) * nz + nj, 1});
  }
  if (ni > 0 && nj > 0) {
    adjacent.push_back({(ni - 1) * nz + nj - 1, 2});
  }
  if (ni < nr && nj > 0) {
    adjacent.push_back({ni * nz + nj - 1, 3});
  }
  return adjacent;
}

double contact_axis_thickness(const int cell,
                              const int axis,
                              const int nz,
                              const std::vector<double>& node_r,
                              const std::vector<double>& node_z) {
  const std::array<int, 4> nodes = structured_cell_nodes(cell, nz);
  const int node_a[2] = {nodes[0], axis == 1 ? nodes[1] : nodes[3]};
  const int node_b[2] = {axis == 1 ? nodes[3] : nodes[1], nodes[2]};
  double thickness = 0.0;
  for (int pair = 0; pair < 2; ++pair) {
    const std::size_t a = static_cast<std::size_t>(node_a[pair]);
    const std::size_t b = static_cast<std::size_t>(node_b[pair]);
    thickness += std::hypot(node_r[b] - node_r[a], node_z[b] - node_z[a]);
  }
  return 0.5 * thickness;
}

double revolved_face_area(const int node_a,
                          const int node_b,
                          const std::vector<double>& node_r,
                          const std::vector<double>& node_z) {
  const double dr = node_r[static_cast<std::size_t>(node_b)] -
                    node_r[static_cast<std::size_t>(node_a)];
  const double dz = node_z[static_cast<std::size_t>(node_b)] -
                    node_z[static_cast<std::size_t>(node_a)];
  return kPi * (node_r[static_cast<std::size_t>(node_a)] +
                node_r[static_cast<std::size_t>(node_b)]) *
         std::hypot(dr, dz);
}

bool evacuated_cell_off_eligible(
    const std::size_t c,
    const std::vector<double>& mass,
    const std::vector<double>& mass_ref,
    const std::vector<double>& vol_ref,
    const EvacCellShadowParams& params) {
  const double m_off =
      std::max(params.off_mass_fraction * mass_ref[c],
               params.rho_vacuum_policy * vol_ref[c]);
  return mass[c] <= m_off;
}

std::pair<double, double> corner_momentum(
    const int nr,
    const int nz,
    const std::vector<double>& corner_mass,
    const std::vector<double>& v_r,
    const std::vector<double>& v_z) {
  double momentum_r = 0.0;
  double momentum_z = 0.0;
  for (int c = 0; c < nr * nz; ++c) {
    const std::array<int, 4> nodes = structured_cell_nodes(c, nz);
    for (int k = 0; k < 4; ++k) {
      const double mass = corner_mass[static_cast<std::size_t>(4 * c + k)];
      const std::size_t node = static_cast<std::size_t>(nodes[k]);
      momentum_r += mass * v_r[node];
      momentum_z += mass * v_z[node];
    }
  }
  return {momentum_r, momentum_z};
}

void transfer_donor_corner_masses(
    const int donor_cell,
    const std::vector<int>& target_corner_indices,
    std::vector<double>& corner_mass) {
  for (int donor_corner = 0; donor_corner < 4; ++donor_corner) {
    const std::size_t donor_index =
        static_cast<std::size_t>(4 * donor_cell + donor_corner);
    const std::size_t target_index =
        static_cast<std::size_t>(target_corner_indices[donor_corner]);
    corner_mass[target_index] += corner_mass[donor_index];
    corner_mass[donor_index] = 0.0;
  }
}

void transfer_rematerialized_corner_masses(
    const int target_cell,
    const std::vector<int>& donors,
    const std::vector<double>& donor_mass_withdrawals,
    std::vector<double>& corner_mass) {
  TENRYU_ASSERT(donors.size() == donor_mass_withdrawals.size() &&
                    target_cell >= 0 &&
                    4U * (static_cast<std::size_t>(target_cell) + 1U) <=
                        corner_mass.size(),
                "evacuated-cell rematerialization corner size mismatch");
  for (std::size_t r = 0; r < donors.size(); ++r) {
    const double dm = donor_mass_withdrawals[r];
    if (!(dm > 0.0)) {
      continue;
    }
    const int donor_cell = donors[r];
    TENRYU_ASSERT(donor_cell >= 0 &&
                      4U * (static_cast<std::size_t>(donor_cell) + 1U) <=
                          corner_mass.size(),
                  "evacuated-cell rematerialization corner donor mismatch");
    const std::size_t donor_base = 4U * static_cast<std::size_t>(donor_cell);
    double donor_corner_sum = 0.0;
    for (int k = 0; k < 4; ++k) {
      donor_corner_sum +=
          corner_mass[donor_base + static_cast<std::size_t>(k)];
    }
    TENRYU_ASSERT(donor_corner_sum > dm,
                  "evacuated-cell rematerialization corner mass exhausted");
    double remaining = dm;
    for (int k = 0; k < 3; ++k) {
      const std::size_t corner = donor_base + static_cast<std::size_t>(k);
      const double withdrawal =
          dm * corner_mass[corner] / donor_corner_sum;
      corner_mass[corner] -= withdrawal;
      remaining -= withdrawal;
    }
    corner_mass[donor_base + 3U] -= remaining;
  }

  // Simple conservative stride-4 closure: the revived cell receives an equal
  // quarter of the withdrawn mass at each corner.
  const double target_mass = std::accumulate(donor_mass_withdrawals.begin(),
                                             donor_mass_withdrawals.end(),
                                             0.0);
  const std::size_t target_base = 4U * static_cast<std::size_t>(target_cell);
  const double quarter_mass = 0.25 * target_mass;
  for (int k = 0; k < 4; ++k) {
    corner_mass[target_base + static_cast<std::size_t>(k)] += quarter_mass;
  }
}

double total_cell_mass(const std::vector<double>& mass) {
  return std::accumulate(mass.begin(), mass.end(), 0.0);
}

double total_cell_internal_energy(const std::vector<double>& mass,
                                  const std::vector<double>& ee,
                                  const std::vector<double>& ei) {
  double total = 0.0;
  for (std::size_t c = 0; c < mass.size(); ++c) {
    total += mass[c] * (ee[c] + ei[c]);
  }
  return total;
}

double total_cell_internal_energy_abs(const std::vector<double>& mass,
                                      const std::vector<double>& ee,
                                      const std::vector<double>& ei) {
  double total = 0.0;
  for (std::size_t c = 0; c < mass.size(); ++c) {
    total += std::abs(mass[c] * (ee[c] + ei[c]));
  }
  return total;
}

double median(std::vector<double> values) {
  TENRYU_ASSERT(!values.empty(),
                "evacuated-cell contact median requires samples");
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2U;
  if ((values.size() & 1U) != 0U) {
    return values[middle];
  }
  return 0.5 * (values[middle - 1U] + values[middle]);
}

double contact_gap_with_frozen_normal(
    const core::EvacContactSlot& slot,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    double gap_pair[2]) {
  for (int pair = 0; pair < 2; ++pair) {
    const std::size_t a = static_cast<std::size_t>(slot.node_a[pair]);
    const std::size_t b = static_cast<std::size_t>(slot.node_b[pair]);
    gap_pair[pair] =
        (node_r[b] - node_r[a]) * slot.normal_pair_r[pair] +
        (node_z[b] - node_z[a]) * slot.normal_pair_z[pair];
  }
  return std::min(gap_pair[0], gap_pair[1]);
}

static double evac_cell_min_corner_jacobian(
    const int cell,
    const int nz,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  const std::array<int, 4> nodes = structured_cell_nodes(cell, nz);
  double r[4];
  double z[4];
  for (int k = 0; k < 4; ++k) {
    const std::size_t node = static_cast<std::size_t>(nodes[k]);
    r[k] = node_r[node];
    z[k] = node_z[node];
  }
  double corner_j_min = std::numeric_limits<double>::infinity();
  for (int k = 0; k < 4; ++k) {
    corner_j_min = std::min(
        corner_j_min, corner_jacobian_from_quad(r, z, k));
  }
  return corner_j_min;
}

void log_seam_forensic(const core::EvacContactSlot& slot,
                       const int nr,
                       const int nz,
                       const int step,
                       const std::vector<double>& node_r,
                       const std::vector<double>& node_z,
                       const std::vector<double>& v_r,
                       const std::vector<double>& v_z,
                       const double gap_pair[2]) {
  const int neighbor_cell = slot.cell + 1;
  TENRYU_ASSERT(slot.cell >= 0 && slot.cell % nz + 1 < nz &&
                    neighbor_cell < nr * nz,
                "seam forensic +z neighbor mismatch");
  int contact_nodes[4];
  int neighbor_nodes[4];
  evacuated_cell_corner_nodes(slot.cell, nr, nz, contact_nodes);
  evacuated_cell_corner_nodes(neighbor_cell, nr, nz, neighbor_nodes);

  // The quality helper uses perimeter order n00,n10,n11,n01. At corner k,
  // J_k = cross(x_{k+1} - x_k, x_{k-1} - x_k), with indices modulo four.
  const int neighbor_perimeter_nodes[4] = {
      neighbor_nodes[0], neighbor_nodes[1], neighbor_nodes[3],
      neighbor_nodes[2]};
  double neighbor_r[4];
  double neighbor_z[4];
  double neighbor_j[4];
  for (int k = 0; k < 4; ++k) {
    const std::size_t node =
        static_cast<std::size_t>(neighbor_perimeter_nodes[k]);
    neighbor_r[k] = node_r[node];
    neighbor_z[k] = node_z[node];
  }
  for (int k = 0; k < 4; ++k) {
    neighbor_j[k] =
        corner_jacobian_from_quad(neighbor_r, neighbor_z, k);
  }

  const std::size_t a0 = static_cast<std::size_t>(slot.node_a[0]);
  const std::size_t a1 = static_cast<std::size_t>(slot.node_a[1]);
  const std::size_t b0 = static_cast<std::size_t>(slot.node_b[0]);
  const std::size_t b1 = static_cast<std::size_t>(slot.node_b[1]);
  const double tangent_mid_r =
      0.5 * ((node_r[a1] - node_r[a0]) + (node_r[b1] - node_r[b0]));
  const double tangent_mid_z =
      0.5 * ((node_z[a1] - node_z[a0]) + (node_z[b1] - node_z[b0]));
  const double tangent_mid_norm = std::hypot(tangent_mid_r, tangent_mid_z);
  TENRYU_ASSERT(tangent_mid_norm > 0.0,
                "seam forensic mortar midline tangent mismatch");
  const double tangent_r = tangent_mid_r / tangent_mid_norm;
  const double tangent_z = tangent_mid_z / tangent_mid_norm;
  double normal_r = tangent_z;
  double normal_z = -tangent_r;
  // The separation-vector sign fix is ill-defined under interpenetration; the
  // topology-anchored pair0 normal is the stable orientation.
  if (normal_r * slot.normal_pair_r[0] +
          normal_z * slot.normal_pair_z[0] <
      0.0) {
    normal_r = -normal_r;
    normal_z = -normal_z;
  }

  constexpr double xi_samples[5] = {0.0, 0.25, 0.5, 0.75, 1.0};
  const double gap_xi0 = (node_r[b0] - node_r[a0]) * normal_r +
                         (node_z[b0] - node_z[a0]) * normal_z;
  const double gap_xi1 = (node_r[b1] - node_r[a1]) * normal_r +
                         (node_z[b1] - node_z[a1]) * normal_z;
  double chi_est;
  if (gap_xi0 <= slot.g0 && gap_xi1 <= slot.g0) {
    chi_est = 1.0;
  } else if (gap_xi0 > slot.g0 && gap_xi1 > slot.g0) {
    chi_est = 0.0;
  } else {
    chi_est = std::clamp(
        (slot.g0 - gap_xi0) / (gap_xi1 - gap_xi0), 0.0, 1.0);
  }
  std::ostringstream block;
  block << "[seam-forensic] step=" << step << " cell=" << slot.cell
        << "\n  corner_nodes cell=[" << contact_nodes[0] << ","
        << contact_nodes[1] << "," << contact_nodes[2] << ","
        << contact_nodes[3] << "] neighbor_cell=" << neighbor_cell << " ["
        << neighbor_nodes[0] << "," << neighbor_nodes[1] << ","
        << neighbor_nodes[2] << "," << neighbor_nodes[3] << "]"
        << std::scientific << std::setprecision(17)
        << "\n  neighbor_corner_J perimeter=[" << neighbor_j[0] << ","
        << neighbor_j[1] << "," << neighbor_j[2] << "," << neighbor_j[3]
        << "]";
  for (const double xi : xi_samples) {
    const double one_minus_xi = 1.0 - xi;
    const double minus_r = one_minus_xi * node_r[a0] + xi * node_r[a1];
    const double minus_z = one_minus_xi * node_z[a0] + xi * node_z[a1];
    const double plus_r = one_minus_xi * node_r[b0] + xi * node_r[b1];
    const double plus_z = one_minus_xi * node_z[b0] + xi * node_z[b1];
    const double minus_v_r = one_minus_xi * v_r[a0] + xi * v_r[a1];
    const double minus_v_z = one_minus_xi * v_z[a0] + xi * v_z[a1];
    const double plus_v_r = one_minus_xi * v_r[b0] + xi * v_r[b1];
    const double plus_v_z = one_minus_xi * v_z[b0] + xi * v_z[b1];
    const double common_gap =
        (plus_r - minus_r) * normal_r + (plus_z - minus_z) * normal_z;
    const double common_gap_rate = (plus_v_r - minus_v_r) * normal_r +
                                   (plus_v_z - minus_v_z) * normal_z;
    block << "\n  common_normal xi=" << xi << " t=(" << tangent_r << ","
          << tangent_z << ") n=(" << normal_r << "," << normal_z
          << ") gap=" << common_gap << " gdot=" << common_gap_rate;
  }
  for (int pair = 0; pair < 2; ++pair) {
    const std::size_t a = static_cast<std::size_t>(slot.node_a[pair]);
    const std::size_t b = static_cast<std::size_t>(slot.node_b[pair]);
    const double vA_n = v_r[a] * slot.normal_pair_r[pair] +
                        v_z[a] * slot.normal_pair_z[pair];
    const double vB_n = v_r[b] * slot.normal_pair_r[pair] +
                        v_z[b] * slot.normal_pair_z[pair];
    block << "\n  pair=" << pair << " gap_pair=" << gap_pair[pair]
          << " u_c=" << vA_n - vB_n;
  }
  block << "\n  lambda_last=" << slot.lambda_last << " g0=" << slot.g0
        << " chi_est=" << chi_est;
  core::log_info(block.str());
}

bool project_contact_pair_velocity(
    const core::EvacContactSlot& slot,
    const int pair,
    const double mA,
    const double mB,
    std::vector<double>& v_r,
    std::vector<double>& v_z,
    EvacContactImpact& impact) {
  TENRYU_ASSERT(pair >= 0 && pair < 2,
                "evacuated-cell contact velocity projection index mismatch");
  const int node_a = slot.node_a[pair];
  const int node_b = slot.node_b[pair];
  TENRYU_ASSERT(node_a >= 0 && node_b >= 0 && v_r.size() == v_z.size() &&
                    static_cast<std::size_t>(node_a) < v_r.size() &&
                    static_cast<std::size_t>(node_b) < v_r.size(),
                "evacuated-cell contact velocity projection input mismatch");
  const std::size_t a = static_cast<std::size_t>(node_a);
  const std::size_t b = static_cast<std::size_t>(node_b);
  const double normal_r = slot.normal_pair_r[pair];
  const double normal_z = slot.normal_pair_z[pair];
  const double vA_n = v_r[a] * normal_r + v_z[a] * normal_z;
  const double vB_n = v_r[b] * normal_r + v_z[b] * normal_z;
  impact = evacuated_cell_contact_impact(mA, mB, vA_n, vB_n);
  if (!(vA_n > vB_n)) {
    return false;
  }
  v_r[a] += (impact.vA_n - vA_n) * normal_r;
  v_z[a] += (impact.vA_n - vA_n) * normal_z;
  v_r[b] += (impact.vB_n - vB_n) * normal_r;
  v_z[b] += (impact.vB_n - vB_n) * normal_z;
  return true;
}

void deposit_contact_side_heat(
    const int node,
    const double side_heat,
    const int nr,
    const int nz,
    const std::vector<double>& corner_mass,
    const std::vector<std::uint8_t>& inactive_member_mask,
    const std::vector<std::int8_t>* const hydro_active_or_null,
    const std::vector<double>& mass,
    std::vector<double>& ei,
    double& deposited_heat,
    std::vector<int>* const heat_cells) {
  if (!(side_heat > 0.0)) {
    return;
  }
  const std::vector<NodeCellCorner> adjacent =
      structured_node_cell_corners(node, nr, nz);
  double active_corner_mass = 0.0;
  for (const NodeCellCorner entry : adjacent) {
    const std::size_t cell = static_cast<std::size_t>(entry.cell);
    if (inactive_member_mask[cell] != 0U ||
        (hydro_active_or_null != nullptr &&
         (*hydro_active_or_null)[cell] == 0)) {
      continue;
    }
    active_corner_mass +=
        corner_mass[4U * cell + static_cast<std::size_t>(entry.corner)];
  }
  TENRYU_ASSERT(active_corner_mass > 0.0,
                "evacuated-cell contact heat has no active adjacency");
  for (const NodeCellCorner entry : adjacent) {
    const std::size_t cell = static_cast<std::size_t>(entry.cell);
    if (inactive_member_mask[cell] != 0U ||
        (hydro_active_or_null != nullptr &&
         (*hydro_active_or_null)[cell] == 0)) {
      continue;
    }
    const double weight =
        corner_mass[4U * cell + static_cast<std::size_t>(entry.corner)] /
        active_corner_mass;
    const double dU = weight * side_heat;
    TENRYU_ASSERT(mass[cell] > 0.0,
                  "evacuated-cell contact heat target has nonpositive mass");
    ei[cell] += dU / mass[cell];
    deposited_heat += dU;
    if (heat_cells != nullptr &&
        std::find(heat_cells->begin(), heat_cells->end(), entry.cell) ==
            heat_cells->end()) {
      heat_cells->push_back(entry.cell);
    }
  }
}

core::EvacContactSlot* find_contact_slot(
    std::vector<core::EvacContactSlot>& slots,
    const int cell) {
  const auto found = std::find_if(
      slots.begin(), slots.end(), [cell](const core::EvacContactSlot& slot) {
        return slot.cell == cell;
      });
  return found == slots.end() ? nullptr : &*found;
}

const core::EvacContactSlot* find_contact_slot(
    const std::vector<core::EvacContactSlot>& slots,
    const int cell) {
  const auto found = std::find_if(
      slots.begin(), slots.end(), [cell](const core::EvacContactSlot& slot) {
        return slot.cell == cell;
      });
  return found == slots.end() ? nullptr : &*found;
}

}  // namespace

EvacContactActiveCells evacuated_cell_rebuild_contact_active(
    const std::vector<core::EvacContactSlot>& slots,
    const std::size_t n_cells) {
  EvacContactActiveCells active;
  active.mask.assign(n_cells, 0U);
  active.cells.reserve(slots.size());
  active.axis.reserve(slots.size());
  active.face_nodes.reserve(4U * slots.size());
  active.face_normal_ref.reserve(2U * slots.size());
  active.face_g0.reserve(slots.size());
  active.mortar_g_hold.reserve(
      core::kEvacContactMortarRowCapacity * slots.size());
  active.mortar_g_hold_valid.reserve(
      core::kEvacContactMortarRowCapacity * slots.size());
  active.vol_at_engagement.reserve(slots.size());
  active.devolumized.reserve(slots.size());
  active.pair_dk_owner.reserve(slots.size());
  int active_pair_count = 0;
  for (const core::EvacContactSlot& slot : slots) {
    if (slot.state != core::EvacContactState::kActive) {
      continue;
    }
    TENRYU_ASSERT(slot.pair_engaged[0] != 0U ||
                      slot.pair_engaged[1] != 0U,
                  "evacuated-cell active contact has no engaged pair");
    active.mask[static_cast<std::size_t>(slot.cell)] = 1U;
    active.cells.push_back(slot.cell);
    active.axis.push_back(slot.axis);
    active.face_nodes.push_back(slot.node_a[0]);
    active.face_nodes.push_back(slot.node_a[1]);
    active.face_nodes.push_back(slot.node_b[0]);
    active.face_nodes.push_back(slot.node_b[1]);
    // The arming scan freezes pair0's topology-anchored normal, so it is
    // valid even when pair0 itself is not the engaged pair.
    active.face_normal_ref.push_back(slot.normal_pair_r[0]);
    active.face_normal_ref.push_back(slot.normal_pair_z[0]);
    active.face_g0.push_back(slot.g0);
    for (int q = 0; q < core::kEvacContactMortarRowCapacity; ++q) {
      active.mortar_g_hold.push_back(slot.mortar_g_hold[q]);
      active.mortar_g_hold_valid.push_back(slot.mortar_g_hold_valid[q]);
    }
    active.vol_at_engagement.push_back(slot.vol_at_engagement);
    active.devolumized.push_back(slot.devolumized);
    active.pair_dk_owner.push_back(active_pair_count);
    active_pair_count += (slot.pair_engaged[0] != 0U ? 1 : 0) +
                         (slot.pair_engaged[1] != 0U ? 1 : 0);
  }
  return active;
}

EvacContactPairBuffers evacuated_cell_rebuild_contact_pairs(
    const std::vector<core::EvacContactSlot>& slots) {
  EvacContactPairBuffers buffers;
  for (const core::EvacContactSlot& slot : slots) {
    const bool any_pair_engaged =
        slot.pair_engaged[0] != 0U || slot.pair_engaged[1] != 0U;
    TENRYU_ASSERT((slot.state == core::EvacContactState::kActive) ==
                      any_pair_engaged,
                  "evacuated-cell contact pair buffer state mismatch");
    for (int pair = 0; pair < 2; ++pair) {
      if (slot.pair_engaged[pair] == 0U) {
        continue;
      }
      TENRYU_ASSERT(slot.state == core::EvacContactState::kActive,
                    "evacuated-cell engaged pair requires active slot");
      buffers.nodes.push_back(slot.node_a[pair]);
      buffers.nodes.push_back(slot.node_b[pair]);
      buffers.normals.push_back(slot.normal_pair_r[pair]);
      buffers.normals.push_back(slot.normal_pair_z[pair]);
    }
  }
  return buffers;
}

static void evacuated_cell_upload_contact_pair_row_coverage(
    core::EvacuatedCellsState& evacuated,
    const EvacContactPairBuffers& pair_buffers) {
  if (evacuated.n_contact_rows <= 0) {
    evacuated.contact_pair_row_covered.clear();
    evacuated.d_contact_pair_row_covered.reset(0U);
    return;
  }
  const std::size_t row_count =
      static_cast<std::size_t>(evacuated.n_contact_rows);
  TENRYU_ASSERT(evacuated.contact_row_nodes.size() == 3U * row_count,
                "evacuated-cell contact pair row coverage staging mismatch");
  std::vector<int> row_slave_nodes;
  row_slave_nodes.reserve(row_count);
  for (std::size_t row = 0; row < row_count; ++row) {
    row_slave_nodes.push_back(evacuated.contact_row_nodes[3U * row]);
  }
  std::sort(row_slave_nodes.begin(), row_slave_nodes.end());
  row_slave_nodes.erase(
      std::unique(row_slave_nodes.begin(), row_slave_nodes.end()),
      row_slave_nodes.end());

  const std::size_t pair_count = pair_buffers.nodes.size() / 2U;
  evacuated.contact_pair_row_covered.assign(pair_count, 0U);
  for (std::size_t pair = 0; pair < pair_count; ++pair) {
    const int node_b = pair_buffers.nodes[2U * pair + 1U];
    if (std::binary_search(row_slave_nodes.begin(), row_slave_nodes.end(),
                           node_b)) {
      evacuated.contact_pair_row_covered[pair] = 1U;
    }
  }
  evacuated.d_contact_pair_row_covered.reset(
      evacuated.contact_pair_row_covered.size());
  evacuated.d_contact_pair_row_covered.copy_from_host(
      evacuated.contact_pair_row_covered);
}

void evacuated_cell_upload_contact_device_state(core::State& state) {
  auto& evacuated = state.evacuated_cells;
  EvacContactActiveCells contact_active =
      evacuated_cell_rebuild_contact_active(
          state.contact_graph.records,
          static_cast<std::size_t>(state.mesh.topo.n_cells));
  evacuated.contact_active_mask = std::move(contact_active.mask);
  evacuated.n_contact_active_cells =
      static_cast<int>(contact_active.cells.size());
  evacuated.d_contact_active_mask.reset(evacuated.contact_active_mask.size());
  evacuated.d_contact_active_mask.copy_from_host(
      evacuated.contact_active_mask);
  evacuated.d_contact_active_cells.reset(contact_active.cells.size());
  evacuated.d_contact_active_cells.copy_from_host(contact_active.cells);
  evacuated.d_contact_active_axis.reset(contact_active.axis.size());
  evacuated.d_contact_active_axis.copy_from_host(contact_active.axis);
  evacuated.d_contact_face_nodes.reset(contact_active.face_nodes.size());
  evacuated.d_contact_face_nodes.copy_from_host(contact_active.face_nodes);
  evacuated.d_contact_face_normal_ref.reset(
      contact_active.face_normal_ref.size());
  evacuated.d_contact_face_normal_ref.copy_from_host(
      contact_active.face_normal_ref);
  evacuated.d_contact_face_g0.reset(contact_active.face_g0.size());
  evacuated.d_contact_face_g0.copy_from_host(contact_active.face_g0);
  evacuated.d_contact_mortar_g_hold.reset(
      contact_active.mortar_g_hold.size());
  evacuated.d_contact_mortar_g_hold.copy_from_host(
      contact_active.mortar_g_hold);
  evacuated.d_contact_mortar_g_hold_valid.reset(
      contact_active.mortar_g_hold_valid.size());
  evacuated.d_contact_mortar_g_hold_valid.copy_from_host(
      contact_active.mortar_g_hold_valid);
  evacuated.d_contact_active_vol_at_engagement.reset(
      contact_active.vol_at_engagement.size());
  evacuated.d_contact_active_vol_at_engagement.copy_from_host(
      contact_active.vol_at_engagement);
  evacuated.d_contact_active_devolumized.reset(
      contact_active.devolumized.size());
  evacuated.d_contact_active_devolumized.copy_from_host(
      contact_active.devolumized);
  evacuated.d_contact_active_pair_dk_owner.reset(
      contact_active.pair_dk_owner.size());
  evacuated.d_contact_active_pair_dk_owner.copy_from_host(
      contact_active.pair_dk_owner);
  if (evacuated.d_contact_volume_projection_count.size() != 2U) {
    evacuated.d_contact_volume_projection_count.reset(2U);
  }
  if (evacuated.d_contact_mortar_drift_count.size() != 3U) {
    const std::vector<int> zero_count(3U, 0);
    evacuated.d_contact_mortar_drift_count.reset(zero_count.size());
    evacuated.d_contact_mortar_drift_count.copy_from_host(zero_count);
  }
  if (evacuated.d_contact_mortar_drift_max_ucorr.size() != 1U) {
    const std::vector<double> zero_max(1U, 0.0);
    evacuated.d_contact_mortar_drift_max_ucorr.reset(zero_max.size());
    evacuated.d_contact_mortar_drift_max_ucorr.copy_from_host(zero_max);
  }

  const EvacContactPairBuffers pair_buffers =
      evacuated_cell_rebuild_contact_pairs(state.contact_graph.records);
  evacuated.n_active_pairs =
      static_cast<int>(pair_buffers.nodes.size() / 2U);
  evacuated.d_contact_pair_nodes.reset(pair_buffers.nodes.size());
  evacuated.d_contact_pair_nodes.copy_from_host(pair_buffers.nodes);
  evacuated.d_contact_pair_normal.reset(pair_buffers.normals.size());
  evacuated.d_contact_pair_normal.copy_from_host(pair_buffers.normals);
  evacuated_cell_upload_contact_pair_row_coverage(evacuated, pair_buffers);
  const std::vector<double> zero_lambda(
      static_cast<std::size_t>(evacuated.n_active_pairs), 0.0);
  evacuated.d_contact_pair_lambda.reset(zero_lambda.size());
  evacuated.d_contact_pair_lambda.copy_from_host(zero_lambda);
  const std::vector<double> zero_dk(
      static_cast<std::size_t>(evacuated.n_active_pairs), 0.0);
  evacuated.d_contact_pair_dk.reset(zero_dk.size());
  evacuated.d_contact_pair_dk.copy_from_host(zero_dk);
}

bool evacuated_cell_seam_devolumize(core::State& state,
                                    const core::Config& cfg,
                                    core::EvacContactSlot& slot,
                                    const std::vector<double>& mass,
                                    const std::vector<double>& vol) {
  const auto reject = [&](const char* const check) {
    core::log_warning(
        "[evac-cell-contact] seam devolumization rejected cell=" +
        std::to_string(slot.cell) + " failed_check=" + check);
    return false;
  };
  if (slot.state != core::EvacContactState::kActive) {
    return reject("slot_active");
  }
  if (slot.devolumized != 0U) {
    return reject("slot_not_already_devolumized");
  }
  const std::size_t cell = static_cast<std::size_t>(slot.cell);
  const auto& evacuated = state.evacuated_cells;
  if (evacuated.inactive_member_mask[cell] == 0U) {
    return reject("inactive_member");
  }
  const auto& evacuated_config = cfg.numerics.ale.evacuated_cell;
  const double policy_ballast =
      std::max(evacuated_config.off_mass_fraction *
                   evacuated.mass_ref[cell],
               evacuated_config.rho_vacuum_policy_g_per_cc *
                   evacuated.vol_ref[cell]) *
      4.0;
  if (!(mass[cell] <= policy_ballast)) {
    return reject("policy_ballast_mass");
  }

  slot.devolumized = 1U;
  if (state.mesh.geometry_exempt_cells.empty()) {
    state.mesh.geometry_exempt_cells.assign(
        static_cast<std::size_t>(state.mesh.topo.n_cells), 0U);
  }
  state.mesh.geometry_exempt_cells[cell] = 1U;
  evacuated_cell_upload_contact_device_state(state);
  rebuild_geometry_policy_exempt(state);

  std::ostringstream line;
  line << "[evac-cell-contact] seam devolumized cell=" << slot.cell
       << " vol=" << std::scientific << std::setprecision(3) << vol[cell]
       << " vol_eng=" << slot.vol_at_engagement
       << " chi=" << evacuated_cell_contact_patch_fraction(
                          slot.gap_pair[0], slot.gap_pair[1], slot.g0);
  core::log_info(line.str());
  return true;
}

bool evacuated_cell_contact_should_arm(const double gap_pair[2],
                                       const double gap_prev_pair[2],
                                       const double g_arm) {
  if (std::min(gap_pair[0], gap_pair[1]) < g_arm) {
    return true;
  }
  for (int pair = 0; pair < 2; ++pair) {
    if (std::isfinite(gap_prev_pair[pair]) &&
        gap_pair[pair] +
                2.0 * (gap_pair[pair] - gap_prev_pair[pair]) <
            g_arm) {
      return true;
    }
  }
  return false;
}

EvacContactGeometry evacuated_cell_contact_geometry(
    const int cell,
    const int axis,
    const int nz,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  TENRYU_ASSERT((axis == 0 || axis == 1) && cell >= 0 && nz > 0 &&
                    node_r.size() == node_z.size(),
                "evacuated-cell contact geometry input mismatch");
  const std::array<int, 4> nodes = structured_cell_nodes(cell, nz);
  EvacContactGeometry geometry;
  geometry.node_a[0] = nodes[0];
  geometry.node_a[1] = axis == 1 ? nodes[1] : nodes[3];
  geometry.node_b[0] = axis == 1 ? nodes[3] : nodes[1];
  geometry.node_b[1] = nodes[2];
  TENRYU_ASSERT(static_cast<std::size_t>(geometry.node_b[1]) < node_r.size(),
                "evacuated-cell contact geometry node range mismatch");

  for (int pair = 0; pair < 2; ++pair) {
    const std::size_t a = static_cast<std::size_t>(geometry.node_a[pair]);
    const std::size_t b = static_cast<std::size_t>(geometry.node_b[pair]);
    const double delta_r = node_r[b] - node_r[a];
    const double delta_z = node_z[b] - node_z[a];
    const double magnitude = std::hypot(delta_r, delta_z);
    TENRYU_ASSERT(magnitude > 0.0,
                  "evacuated-cell contact geometry has zero thickness");
    geometry.normal_pair_r[pair] = delta_r / magnitude;
    geometry.normal_pair_z[pair] = delta_z / magnitude;
    const int ni_a = geometry.node_a[pair] / (nz + 1);
    const int ni_b = geometry.node_b[pair] / (nz + 1);
    if (ni_a == 0 && ni_b == 0) {
      // An axis pair must use the direction that commutes with v_r = 0.
      geometry.normal_pair_r[pair] = 0.0;
      geometry.normal_pair_z[pair] =
          std::copysign(1.0, delta_z == 0.0 ? 1.0 : delta_z);
    } else if (std::abs(geometry.normal_pair_z[pair]) > 0.999) {
      geometry.normal_pair_r[pair] = 0.0;
      geometry.normal_pair_z[pair] =
          std::copysign(1.0, geometry.normal_pair_z[pair]);
    } else if (std::abs(geometry.normal_pair_r[pair]) > 0.999) {
      geometry.normal_pair_r[pair] =
          std::copysign(1.0, geometry.normal_pair_r[pair]);
      geometry.normal_pair_z[pair] = 0.0;
    }
    geometry.gap_pair[pair] =
        delta_r * geometry.normal_pair_r[pair] +
        delta_z * geometry.normal_pair_z[pair];
  }
  geometry.gap = std::min(geometry.gap_pair[0], geometry.gap_pair[1]);
  return geometry;
}

EvacContactAxisSelection evacuated_cell_closing_axis(
    const int cell,
    const int nz,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const std::vector<double>& x_r_ref,
    const std::vector<double>& x_z_ref) {
  TENRYU_ASSERT(node_r.size() == node_z.size() &&
                    x_r_ref.size() == node_r.size() &&
                    x_z_ref.size() == node_z.size(),
                "evacuated-cell closing-axis field size mismatch");
  const double h_i_ref =
      contact_axis_thickness(cell, 0, nz, x_r_ref, x_z_ref);
  const double h_j_ref =
      contact_axis_thickness(cell, 1, nz, x_r_ref, x_z_ref);
  TENRYU_ASSERT(h_i_ref > 0.0 && h_j_ref > 0.0,
                "evacuated-cell closing-axis reference thickness mismatch");
  const double ratio_i =
      contact_axis_thickness(cell, 0, nz, node_r, node_z) / h_i_ref;
  const double ratio_j =
      contact_axis_thickness(cell, 1, nz, node_r, node_z) / h_j_ref;
  if (ratio_j <= ratio_i) {
    return {1, h_j_ref};
  }
  return {0, h_i_ref};
}

EvacContactImpact evacuated_cell_contact_impact(const double mA,
                                                const double mB,
                                                const double vA_n,
                                                const double vB_n) {
  TENRYU_ASSERT(mA > 0.0 && mB > 0.0,
                "evacuated-cell contact impact requires positive masses");
  EvacContactImpact impact;
  impact.vA_n = vA_n;
  impact.vB_n = vB_n;
  const double u_c = vA_n - vB_n;
  if (!(u_c > 0.0)) {
    return impact;
  }
  const double reduced_mass = mA * mB / (mA + mB);
  const double common = (mA * vA_n + mB * vB_n) / (mA + mB);
  impact.vA_n = common;
  impact.vB_n = common;
  impact.J = reduced_mass * u_c;
  impact.dK = 0.5 * reduced_mass * u_c * u_c;
  impact.dK_A = impact.dK * mB / (mA + mB);
  impact.dK_B = impact.dK * mA / (mA + mB);
  return impact;
}

bool evacuated_cell_contact_engage_pair(
    core::EvacContactSlot& slot,
    const int pair,
    const double gap,
    const double mA,
    const double mB,
    const double cell_mass,
    const double cell_volume,
    std::vector<double>& v_r,
    std::vector<double>& v_z,
    EvacContactImpact& impact,
    const bool force_crossed,
    const bool skip_velocity_projection) {
  TENRYU_ASSERT(pair >= 0 && pair < 2,
                "evacuated-cell contact pair engagement index mismatch");
  TENRYU_ASSERT(slot.state == core::EvacContactState::kArmed ||
                    slot.state == core::EvacContactState::kActive,
                "evacuated-cell contact pair engagement state mismatch");
  if (slot.pair_engaged[pair] != 0U ||
      (!force_crossed && gap > slot.g0)) {
    impact = {};
    return false;
  }
  impact = {};
  if (!skip_velocity_projection) {
    (void)project_contact_pair_velocity(
        slot, pair, mA, mB, v_r, v_z, impact);
  }
  const bool first_pair_engagement =
      slot.pair_engaged[0] == 0U && slot.pair_engaged[1] == 0U;
  slot.pair_engaged[pair] = 1U;
  std::fill(std::begin(slot.mortar_g_hold_valid),
            std::end(slot.mortar_g_hold_valid), 0U);
  slot.gap_at_engagement_pair[pair] = gap;
  slot.state = core::EvacContactState::kActive;
  if (first_pair_engagement) {
    slot.mass_at_engagement = cell_mass;
    slot.vol_at_engagement = cell_volume;
  }
  return true;
}

bool evacuated_cell_contact_update_pair_release(
    core::EvacContactSlot& slot,
    const int pair,
    const double lambda,
    const double threshold,
    const int persistence,
    const bool geometric_reopen) {
  TENRYU_ASSERT(pair >= 0 && pair < 2 && persistence > 0,
                "evacuated-cell contact release input mismatch");
  TENRYU_ASSERT(slot.state == core::EvacContactState::kActive,
                "evacuated-cell contact release state mismatch");
  if (slot.pair_engaged[pair] == 0U) {
    slot.tensile_streak_pair[pair] = 0;
    return false;
  }
  // Tension and geometric separation intentionally share persistence
  // semantics and the same per-pair streak counter.
  if (lambda < -threshold || geometric_reopen) {
    ++slot.tensile_streak_pair[pair];
  } else {
    slot.tensile_streak_pair[pair] = 0;
  }
  if (slot.tensile_streak_pair[pair] < persistence) {
    return false;
  }
  slot.pair_engaged[pair] = 0U;
  std::fill(std::begin(slot.mortar_g_hold_valid),
            std::end(slot.mortar_g_hold_valid), 0U);
  slot.gap_at_engagement_pair[pair] = 0.0;
  slot.tensile_streak_pair[pair] = 0;
  ++slot.release_count;
  if (slot.pair_engaged[0] == 0U && slot.pair_engaged[1] == 0U) {
    slot.state = core::EvacContactState::kOpenSpacer;
    slot.mass_at_engagement = std::numeric_limits<double>::quiet_NaN();
    slot.vol_at_engagement = std::numeric_limits<double>::quiet_NaN();
  }
  return true;
}

double evacuated_cell_active_node_mass(
    const int node,
    const int nr,
    const int nz,
    const std::vector<double>& corner_mass,
    const std::vector<std::uint8_t>& inactive_member_mask,
    const std::vector<std::int8_t>* const hydro_active_or_null) {
  const std::size_t n_cells = static_cast<std::size_t>(nr * nz);
  TENRYU_ASSERT(corner_mass.size() == 4U * n_cells &&
                    inactive_member_mask.size() == n_cells &&
                    (hydro_active_or_null == nullptr ||
                     hydro_active_or_null->size() == n_cells),
                "evacuated-cell active-node-mass field size mismatch");
  double mass = 0.0;
  for (const NodeCellCorner adjacent :
       structured_node_cell_corners(node, nr, nz)) {
    const std::size_t cell = static_cast<std::size_t>(adjacent.cell);
    if (inactive_member_mask[cell] != 0U ||
        (hydro_active_or_null != nullptr &&
         (*hydro_active_or_null)[cell] == 0)) {
      continue;
    }
    mass += corner_mass[4U * cell + static_cast<std::size_t>(adjacent.corner)];
  }
  return mass;
}

bool evacuated_cell_contact_reproject_pair(
    core::EvacContactSlot& slot,
    const int pair,
    const int nr,
    const int nz,
    const std::vector<double>& corner_mass,
    const std::vector<std::uint8_t>& inactive_member_mask,
    const std::vector<std::int8_t>* const hydro_active_or_null,
    const std::vector<double>& mass,
    std::vector<double>& v_r,
    std::vector<double>& v_z,
    std::vector<double>& ei) {
  TENRYU_ASSERT(pair >= 0 && pair < 2 &&
                    slot.state == core::EvacContactState::kActive &&
                    slot.pair_engaged[pair] != 0U,
                "evacuated-cell contact backstop state mismatch");
  TENRYU_ASSERT(mass.size() == ei.size() &&
                    mass.size() == inactive_member_mask.size(),
                "evacuated-cell contact backstop field size mismatch");
  const int node_a = slot.node_a[pair];
  const int node_b = slot.node_b[pair];
  TENRYU_ASSERT(node_a >= 0 && node_b >= 0 && v_r.size() == v_z.size() &&
                    static_cast<std::size_t>(node_a) < v_r.size() &&
                    static_cast<std::size_t>(node_b) < v_r.size(),
                "evacuated-cell contact backstop velocity size mismatch");
  const double mA = evacuated_cell_active_node_mass(
      node_a,
      nr,
      nz,
      corner_mass,
      inactive_member_mask,
      hydro_active_or_null);
  const double mB = evacuated_cell_active_node_mass(
      node_b,
      nr,
      nz,
      corner_mass,
      inactive_member_mask,
      hydro_active_or_null);
  TENRYU_ASSERT(mA > 0.0 && mB > 0.0,
                "evacuated-cell contact backstop has nonpositive node mass");
  const std::size_t a = static_cast<std::size_t>(node_a);
  const std::size_t b = static_cast<std::size_t>(node_b);
  const double vA_n = v_r[a] * slot.normal_pair_r[pair] +
                      v_z[a] * slot.normal_pair_z[pair];
  const double vB_n = v_r[b] * slot.normal_pair_r[pair] +
                      v_z[b] * slot.normal_pair_z[pair];
  const double kinetic_before =
      0.5 * mA * vA_n * vA_n + 0.5 * mB * vB_n * vB_n;

  EvacContactImpact impact;
  if (!project_contact_pair_velocity(
          slot, pair, mA, mB, v_r, v_z, impact)) {
    return false;
  }
  const double kinetic_after =
      0.5 * mA * impact.vA_n * impact.vA_n +
      0.5 * mB * impact.vB_n * impact.vB_n;
  double deposited_heat = 0.0;
  deposit_contact_side_heat(node_a,
                            impact.dK_A,
                            nr,
                            nz,
                            corner_mass,
                            inactive_member_mask,
                            hydro_active_or_null,
                            mass,
                            ei,
                            deposited_heat,
                            nullptr);
  deposit_contact_side_heat(node_b,
                            impact.dK_B,
                            nr,
                            nz,
                            corner_mass,
                            inactive_member_mask,
                            hydro_active_or_null,
                            mass,
                            ei,
                            deposited_heat,
                            nullptr);

  const double lost_kinetic = kinetic_before - kinetic_after;
  const double scale = std::max(
      {std::abs(kinetic_before),
       std::abs(kinetic_after),
       std::abs(deposited_heat),
       std::numeric_limits<double>::min()});
  TENRYU_ASSERT(
      std::abs(lost_kinetic - deposited_heat) <=
          64.0 * std::numeric_limits<double>::epsilon() * scale,
      "evacuated-cell contact backstop energy audit failed");
  ++slot.reproject_count;
  slot.reproject_heat_total += deposited_heat;
  return true;
}

bool evacuated_cell_refill_viable(
    const double m_fill,
    const double m_ref,
    const double rho_fill,
    const double rho_donor_median,
    const core::Config::NumericsConfig::AleConfig::EvacuatedCellConfig::
        EvacuatedCellClosureContactConfig& cfg) {
  return m_fill >= cfg.refill_min_mass_fraction * m_ref &&
         rho_fill >= cfg.refill_min_density_ratio * rho_donor_median;
}

bool evacuated_cell_live_closure_eligible(
    const double mass,
    const double m_ref,
    const double vol,
    const double V_ref,
    const std::uint8_t lineage,
    const core::Config::NumericsConfig::AleConfig::EvacuatedCellConfig::
        EvacuatedCellClosureContactConfig& cfg) {
  return lineage != 0U && mass < cfg.live_mass_gate * m_ref &&
         vol < cfg.live_volume_gate * V_ref;
}

EvacCellShadowSummary classify_evacuated_cells(
    const std::vector<double>& mass,
    const std::vector<double>& mass_ref,
    const std::vector<double>& vol_ref,
    const std::vector<double>& rho,
    const std::vector<double>& ne_over_ncrit,
    const std::vector<double>& Te,
    const EvacCellShadowParams& params,
    const std::vector<std::uint8_t>* inactive_member_mask) {
  TENRYU_ASSERT(mass_ref.size() == mass.size() &&
                    vol_ref.size() == mass.size() && rho.size() == mass.size() &&
                    ne_over_ncrit.size() == mass.size() &&
                    Te.size() == mass.size(),
                "evacuated-cell classifier field size mismatch");
  TENRYU_ASSERT(inactive_member_mask == nullptr ||
                    inactive_member_mask->size() == mass.size(),
                "evacuated-cell classifier mask size mismatch");
  EvacCellShadowSummary summary;
  std::vector<EvacCellShadowCell> cells;
  cells.reserve(mass.size());
  for (std::size_t c = 0; c < mass.size(); ++c) {
    if (inactive_member_mask != nullptr &&
        (*inactive_member_mask)[c] != static_cast<std::uint8_t>(0)) {
      continue;
    }
    const double m_off =
        std::max(params.off_mass_fraction * mass_ref[c],
                 params.rho_vacuum_policy * vol_ref[c]);
    const double m_arm =
        std::max(params.arm_mass_fraction * mass_ref[c], 100.0 * m_off);
    if (mass[c] <= m_off) {
      ++summary.n_off_eligible;
    } else if (mass[c] <= m_arm) {
      ++summary.n_armed;
    } else {
      ++summary.n_active;
    }
    cells.push_back({static_cast<int>(c),
                     mass[c] / mass_ref[c],
                     rho[c],
                     ne_over_ncrit[c],
                     Te[c]});
  }
  std::stable_sort(cells.begin(),
                   cells.end(),
                   [](const EvacCellShadowCell& lhs,
                      const EvacCellShadowCell& rhs) {
                     return lhs.mass_fraction < rhs.mass_fraction;
                   });
  if (cells.size() > 4U) {
    cells.resize(4U);
  }
  summary.worst = std::move(cells);
  return summary;
}

bool update_evacuated_cell_off_streak(const bool off_eligible,
                                      const int off_hold_evaluations,
                                      int& off_streak) {
  TENRYU_ASSERT(off_hold_evaluations >= 1,
                "evacuated-cell off-hold evaluations must be >= 1");
  if (!off_eligible) {
    off_streak = 0;
    return false;
  }
  if (off_streak < std::numeric_limits<int>::max()) {
    ++off_streak;
  }
  return off_streak >= off_hold_evaluations;
}

EvacCellFaceWeights evacuated_cell_face_weights(
    const int nr,
    const int nz,
    const int donor_cell,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const std::vector<std::uint8_t>& inactive_member_mask) {
  TENRYU_ASSERT(nr > 0 && nz > 0 && donor_cell >= 0 &&
                    donor_cell < nr * nz,
                "evacuated-cell face-weight lattice mismatch");
  const std::size_t n_nodes =
      static_cast<std::size_t>(nr + 1) * static_cast<std::size_t>(nz + 1);
  TENRYU_ASSERT(node_r.size() == n_nodes && node_z.size() == n_nodes,
                "evacuated-cell face-weight node size mismatch");
  TENRYU_ASSERT(inactive_member_mask.empty() ||
                    inactive_member_mask.size() ==
                        static_cast<std::size_t>(nr * nz),
                "evacuated-cell face-weight mask size mismatch");

  EvacCellFaceWeights result;
  std::vector<double> areas;
  const int i = donor_cell / nz;
  const int j = donor_cell - i * nz;
  const auto append_face = [&](const int recipient,
                               const int node_a,
                               const int node_b) {
    if (!inactive_member_mask.empty() &&
        inactive_member_mask[static_cast<std::size_t>(recipient)] != 0U) {
      return;
    }
    const double area = revolved_face_area(node_a, node_b, node_r, node_z);
    if (area > 0.0 && std::isfinite(area)) {
      result.recipients.push_back(recipient);
      areas.push_back(area);
    }
  };
  if (i > 0) {
    // For i == 0 the inner face is the RZ axis and has no neighbor.
    append_face((i - 1) * nz + j,
                i * (nz + 1) + j,
                i * (nz + 1) + j + 1);
  }
  if (i + 1 < nr) {
    append_face((i + 1) * nz + j,
                (i + 1) * (nz + 1) + j,
                (i + 1) * (nz + 1) + j + 1);
  }
  if (j > 0) {
    append_face(i * nz + j - 1,
                i * (nz + 1) + j,
                (i + 1) * (nz + 1) + j);
  }
  if (j + 1 < nz) {
    append_face(i * nz + j + 1,
                i * (nz + 1) + j + 1,
                (i + 1) * (nz + 1) + j + 1);
  }
  const double area_sum = std::accumulate(areas.begin(), areas.end(), 0.0);
  if (!(area_sum > 0.0) || !std::isfinite(area_sum)) {
    result.recipients.clear();
    return result;
  }
  result.weights.reserve(areas.size());
  for (const double area : areas) {
    result.weights.push_back(area / area_sum);
  }
  return result;
}

bool evacuated_cell_is_domain_corner(const int nr,
                                      const int nz,
                                      const int cell) {
  TENRYU_ASSERT(nr > 0 && nz > 0 && cell >= 0 && cell < nr * nz,
                "evacuated-cell domain-corner lattice mismatch");
  const int i = cell / nz;
  const int j = cell - i * nz;
  return (i == 0 || i == nr - 1) && (j == 0 || j == nz - 1);
}

std::vector<int> evacuated_cell_corner_mass_targets(
    const int nr,
    const int nz,
    const int donor_cell,
    const std::vector<int>& recipients,
    const std::vector<double>& weights) {
  TENRYU_ASSERT(nr > 0 && nz > 0 && donor_cell >= 0 &&
                    donor_cell < nr * nz &&
                    recipients.size() == weights.size(),
                "evacuated-cell corner-target lattice mismatch");
  if (evacuated_cell_is_domain_corner(nr, nz, donor_cell)) {
    return {};
  }

  const std::array<int, 4> donor_nodes =
      structured_cell_nodes(donor_cell, nz);
  std::vector<int> targets;
  targets.reserve(4U);
  for (int donor_corner = 0; donor_corner < 4; ++donor_corner) {
    int target = -1;
    double target_weight = -1.0;
    for (std::size_t r = 0; r < recipients.size(); ++r) {
      TENRYU_ASSERT(recipients[r] >= 0 && recipients[r] < nr * nz,
                    "evacuated-cell corner-target recipient mismatch");
      const std::array<int, 4> recipient_nodes =
          structured_cell_nodes(recipients[r], nz);
      for (int recipient_corner = 0; recipient_corner < 4;
           ++recipient_corner) {
        if (recipient_nodes[recipient_corner] == donor_nodes[donor_corner] &&
            weights[r] > target_weight) {
          target = 4 * recipients[r] + recipient_corner;
          target_weight = weights[r];
        }
      }
    }
    if (target < 0) {
      return {};
    }
    targets.push_back(target);
  }
  return targets;
}

void transfer_evacuated_cell_extensives(
    const int donor_cell,
    const std::vector<int>& recipients,
    const std::vector<double>& weights,
    std::vector<double>& mass,
    std::vector<double>& ee,
    std::vector<double>& ei) {
  TENRYU_ASSERT(mass.size() == ee.size() && mass.size() == ei.size() &&
                    donor_cell >= 0 &&
                    static_cast<std::size_t>(donor_cell) < mass.size() &&
                    recipients.size() == weights.size(),
                "evacuated-cell extensive transfer size mismatch");
  const double donor_mass = mass[static_cast<std::size_t>(donor_cell)];
  const double donor_Ue = donor_mass * ee[static_cast<std::size_t>(donor_cell)];
  const double donor_Ui = donor_mass * ei[static_cast<std::size_t>(donor_cell)];
  for (std::size_t r = 0; r < recipients.size(); ++r) {
    const std::size_t recipient = static_cast<std::size_t>(recipients[r]);
    TENRYU_ASSERT(recipient < mass.size() &&
                      recipients[r] != donor_cell,
                  "evacuated-cell recipient index mismatch");
    const double recipient_Ue = mass[recipient] * ee[recipient];
    const double recipient_Ui = mass[recipient] * ei[recipient];
    mass[recipient] += weights[r] * donor_mass;
    const double Ue = recipient_Ue + weights[r] * donor_Ue;
    const double Ui = recipient_Ui + weights[r] * donor_Ui;
    ee[recipient] = Ue / mass[recipient];
    ei[recipient] = Ui / mass[recipient];
  }
  mass[static_cast<std::size_t>(donor_cell)] = 0.0;
  ee[static_cast<std::size_t>(donor_cell)] = 0.0;
  ei[static_cast<std::size_t>(donor_cell)] = 0.0;
}

EvacCellRematerializeTransfer rematerialize_evacuated_cell_extensives(
    const int cell,
    const double cell_volume,
    const std::vector<int>& donors,
    const std::vector<double>& weights,
    const double neighbor_change_max,
    const std::vector<double>& rho,
    std::vector<double>& mass,
    std::vector<double>& ee,
    std::vector<double>& ei) {
  TENRYU_ASSERT(mass.size() == rho.size() && mass.size() == ee.size() &&
                    mass.size() == ei.size() && cell >= 0 &&
                    static_cast<std::size_t>(cell) < mass.size() &&
                    !donors.empty() && donors.size() == weights.size() &&
                    neighbor_change_max > 0.0,
                "evacuated-cell rematerialization transfer size mismatch");

  double donor_mass_sum = 0.0;
  double donor_density_mass_sum = 0.0;
  for (const int donor_cell : donors) {
    TENRYU_ASSERT(donor_cell >= 0 &&
                      static_cast<std::size_t>(donor_cell) < mass.size() &&
                      donor_cell != cell,
                  "evacuated-cell rematerialization donor index mismatch");
    const std::size_t donor = static_cast<std::size_t>(donor_cell);
    donor_mass_sum += mass[donor];
    donor_density_mass_sum += mass[donor] * rho[donor];
  }

  EvacCellRematerializeTransfer result;
  if (!(donor_mass_sum > 0.0) || !std::isfinite(donor_mass_sum) ||
      !std::isfinite(donor_density_mass_sum)) {
    return result;
  }
  const double rho_target = donor_density_mass_sum / donor_mass_sum;
  const double target_mass =
      rho_target * std::max(cell_volume, 1.0e-300);
  if (!(target_mass > 0.0) || !std::isfinite(target_mass)) {
    return result;
  }

  result.donor_mass_withdrawals.resize(donors.size(), 0.0);
  for (std::size_t r = 0; r < donors.size(); ++r) {
    const std::size_t donor = static_cast<std::size_t>(donors[r]);
    const double dm =
        std::min(weights[r] * target_mass,
                 neighbor_change_max * mass[donor]);
    if (!(dm >= 0.0) || !std::isfinite(dm)) {
      return result;
    }
    result.donor_mass_withdrawals[r] = dm;
    result.mass += dm;
  }
  result.fill_fraction = result.mass / target_mass;
  if (!(result.fill_fraction >= 1.0e-3)) {
    return result;
  }

  const std::size_t target = static_cast<std::size_t>(cell);
  const double target_Ue = mass[target] * ee[target];
  const double target_Ui = mass[target] * ei[target];
  for (std::size_t r = 0; r < donors.size(); ++r) {
    const std::size_t donor = static_cast<std::size_t>(donors[r]);
    const double dm = result.donor_mass_withdrawals[r];
    if (!(dm > 0.0)) {
      continue;
    }
    const double dUe = dm * ee[donor];
    const double dUi = dm * ei[donor];
    const double donor_Ue = mass[donor] * ee[donor] - dUe;
    const double donor_Ui = mass[donor] * ei[donor] - dUi;
    mass[donor] -= dm;
    ee[donor] = donor_Ue / mass[donor];
    ei[donor] = donor_Ui / mass[donor];
    result.electron_energy += dUe;
    result.ion_energy += dUi;
  }
  mass[target] += result.mass;
  ee[target] = (target_Ue + result.electron_energy) / mass[target];
  ei[target] = (target_Ui + result.ion_energy) / mass[target];
  result.accepted = true;
  return result;
}

EvacCellRematerializeTrigger evacuated_cell_rematerialize_trigger(
    const double volume,
    const double previous_volume,
    const double reference_volume,
    const double volume_fraction,
    const int rematerialize_after_evaluations,
    int& evaluations_since_conversion) {
  TENRYU_ASSERT(rematerialize_after_evaluations >= 1 &&
                    evaluations_since_conversion >= 0,
                "evacuated-cell rematerialization evaluation count mismatch");
  if (evaluations_since_conversion < rematerialize_after_evaluations) {
    ++evaluations_since_conversion;
  }
  if (evaluations_since_conversion >= rematerialize_after_evaluations) {
    return EvacCellRematerializeTrigger::TIMED;
  }
  const double volume_threshold = volume_fraction * reference_volume;
  if (volume < volume_threshold) {
    return EvacCellRematerializeTrigger::VOLUME;
  }
  if (volume + (volume - previous_volume) < volume_threshold) {
    return EvacCellRematerializeTrigger::PREDICTED;
  }
  return EvacCellRematerializeTrigger::NONE;
}

bool consume_evacuated_cell_rematerialize_dwell(int& dwell_remaining) {
  TENRYU_ASSERT(dwell_remaining >= 0,
                "evacuated-cell rematerialization dwell must be non-negative");
  if (dwell_remaining == 0) {
    return false;
  }
  --dwell_remaining;
  return true;
}

void evacuated_cell_shadow_step(core::State& state, const core::Config& cfg) {
  const auto& shadow = cfg.numerics.diagnostics.evacuated_cell_shadow;
  const auto& topo = state.mesh.topo;
  if (!shadow.enabled || cfg.main.dim != 2 ||
      topo.n_cells != topo.nr * topo.nz ||
      state.step % shadow.every_n_steps != 0) {
    return;
  }

  struct ReferenceCache {
    int n_cells = -1;
    std::vector<double> mass_ref;
    std::vector<double> vol_ref;
  };
  static ReferenceCache cache;
  if (cache.n_cells != topo.n_cells) {
    // V1 restart caveat: the reference is the first shadow evaluation of the
    // process, not a persisted pre-restart reference.
    state.mass.copy_to_host(cache.mass_ref);
    state.vol.copy_to_host(cache.vol_ref);
    cache.n_cells = topo.n_cells;
    core::log_info("[evac-cell] reference captured step=" +
                   std::to_string(state.step) + " n_cells=" +
                   std::to_string(topo.n_cells));
  }

  std::vector<double> mass;
  std::vector<double> rho;
  std::vector<double> zbar;
  std::vector<double> Te;
  state.mass.copy_to_host(mass);
  state.rho.copy_to_host(rho);
  state.zbar.copy_to_host(zbar);
  state.Te.copy_to_host(Te);

  const double Abar = cfg.materials.materials.front().A;
  const double lambda_um = shadow.laser_wavelength_nm * 1.0e-3;
  const double n_crit = 1.115e21 / (lambda_um * lambda_um);
  std::vector<double> ne_over_ncrit(mass.size(), 0.0);
  for (std::size_t c = 0; c < mass.size(); ++c) {
    const double n_e =
        rho[c] * std::max(zbar[c], 0.0) /
        (Abar * core::constants::proton_mass);
    ne_over_ncrit[c] = n_e / n_crit;
  }

  const EvacCellShadowSummary summary = classify_evacuated_cells(
      mass,
      cache.mass_ref,
      cache.vol_ref,
      rho,
      ne_over_ncrit,
      Te,
      {shadow.arm_mass_fraction,
       shadow.off_mass_fraction,
       shadow.rho_vacuum_policy_g_per_cc});

  std::ostringstream line;
  line << "[evac-cell] step=" << state.step << " t=" << std::scientific
       << std::setprecision(17) << state.t << " active=" << summary.n_active
       << " armed=" << summary.n_armed
       << " off_eligible=" << summary.n_off_eligible;
  core::log_info(line.str());

  if (!summary.worst.empty() &&
      summary.n_armed + summary.n_off_eligible > 0) {
    for (const EvacCellShadowCell& cell : summary.worst) {
      std::ostringstream worst_line;
      worst_line << "[evac-cell] worst cell=" << cell.cell
                 << " i=" << cell.cell / topo.nz
                 << " j=" << cell.cell % topo.nz << " m_frac="
                 << std::scientific << std::setprecision(17)
                 << cell.mass_fraction << " rho=" << cell.rho
                 << " ne_over_ncrit=" << cell.ne_over_ncrit
                 << " Te=" << cell.Te;
      core::log_info(worst_line.str());
    }
  }
}

void evacuated_cell_controller_step(core::State& state,
                                    const core::Config& cfg) {
  const auto& config = cfg.numerics.ale.evacuated_cell;
  const auto& topo = state.mesh.topo;
  if (!config.enabled || cfg.main.dim != 2 ||
      cfg.mesh.topology_scheme != core::TopologyScheme::SINGLE_BLOCK ||
      topo.multiblock.has_value() || topo.n_cells != topo.nr * topo.nz ||
      state.step % config.every_n_steps != 0) {
    return;
  }

  const std::size_t n_cells = static_cast<std::size_t>(topo.n_cells);
  auto& evacuated = state.evacuated_cells;
  std::vector<double> node_r;
  std::vector<double> node_z;
  state.x_r.copy_to_host(node_r);
  state.x_z.copy_to_host(node_z);
  if (evacuated.mass_ref.empty()) {
    state.mass.copy_to_host(evacuated.mass_ref);
    state.vol.copy_to_host(evacuated.vol_ref);
    evacuated.x_r_ref = node_r;
    evacuated.x_z_ref = node_z;
    evacuated.inactive_member_mask.assign(n_cells, 0U);
    evacuated.contact_active_mask.assign(n_cells, 0U);
    evacuated.off_streak.assign(n_cells, 0);
    evacuated.closure_lineage.assign(n_cells, 0U);
    evacuated.controller_dwell_remaining.assign(n_cells, 0);
    evacuated.controller_evaluations_since_conversion.assign(n_cells, 0);
    evacuated.controller_previous_volume.assign(
        n_cells, std::numeric_limits<double>::quiet_NaN());
    evacuated.controller_previous_contact_gap.assign(
        n_cells,
        {std::numeric_limits<double>::quiet_NaN(),
         std::numeric_limits<double>::quiet_NaN()});
    evacuated.d_inactive_member_mask.reset(n_cells);
    evacuated.d_inactive_member_mask.copy_from_host(
        evacuated.inactive_member_mask);
    core::log_info("[evac-cell-txn] reference captured step=" +
                   std::to_string(state.step) + " n_cells=" +
                   std::to_string(topo.n_cells));
  }
  TENRYU_ASSERT(evacuated.mass_ref.size() == n_cells &&
                    evacuated.vol_ref.size() == n_cells &&
                    evacuated.x_r_ref.size() == state.x_r.size() &&
                    evacuated.x_z_ref.size() == state.x_z.size() &&
                    evacuated.off_streak.size() == n_cells &&
                    evacuated.closure_lineage.size() == n_cells &&
                    evacuated.inactive_member_mask.size() == n_cells &&
                    evacuated.contact_active_mask.size() == n_cells &&
                    evacuated.controller_dwell_remaining.size() == n_cells &&
                    evacuated.controller_evaluations_since_conversion.size() ==
                        n_cells &&
                    evacuated.controller_previous_volume.size() == n_cells &&
                    evacuated.controller_previous_contact_gap.size() == n_cells,
                "evacuated-cell controller state size mismatch");
  TENRYU_ASSERT(state.conduction_e_rate.size() == n_cells,
                "evacuated-cell controller conduction_e_rate size mismatch");

  std::vector<double> mass;
  std::vector<double> vol;
  std::vector<double> rho;
  std::vector<double> zbar;
  std::vector<double> Te;
  std::vector<double> Ti;
  std::vector<double> ee;
  std::vector<double> ei;
  std::vector<double> Pe;
  std::vector<double> Pi;
  std::vector<double> conduction_e_rate;
  std::vector<double> laser_dep(n_cells, 0.0);
  std::vector<double> v_r;
  std::vector<double> v_z;
  std::vector<double> corner_mass;
  state.mass.copy_to_host(mass);
  state.vol.copy_to_host(vol);
  state.rho.copy_to_host(rho);
  state.zbar.copy_to_host(zbar);
  state.Te.copy_to_host(Te);
  state.Ti.copy_to_host(Ti);
  state.ee.copy_to_host(ee);
  state.ei.copy_to_host(ei);
  state.Pe.copy_to_host(Pe);
  state.Pi.copy_to_host(Pi);
  state.conduction_e_rate.copy_to_host(conduction_e_rate);
  if (state.laser_dep.size() == n_cells) {
    state.laser_dep.copy_to_host(laser_dep);
  }
  state.v_r.copy_to_host(v_r);
  state.v_z.copy_to_host(v_z);
  state.corner_mass.copy_to_host(corner_mass);
  TENRYU_ASSERT(state.corner_stride == 4 &&
                    corner_mass.size() == 4U * n_cells,
                "evacuated-cell controller requires stride-4 corner masses");

  int rematerialized = 0;
  if (config.rematerialize_enabled) {
    const double mass_before_remat = total_cell_mass(mass);
    const double energy_before_remat =
        total_cell_internal_energy(mass, ee, ei);
    double mass_abs_scale_remat = 0.0;
    for (const double value : mass) {
      mass_abs_scale_remat += std::abs(value);
    }
    const double energy_abs_scale_remat =
        total_cell_internal_energy_abs(mass, ee, ei);

    // Conversion purges the sub-continuum state; timed re-materialization
    // returns a neighbor-consistent parcel. Repeated quarantine/reset cycles
    // are the management mechanism, bounded by dwell and hysteresis.
    for (std::size_t c = 0; c < n_cells; ++c) {
      if (evacuated.inactive_member_mask[c] == 0U) {
        continue;
      }
      const core::EvacContactSlot* const contact_slot = find_contact_slot(
          state.contact_graph.records, static_cast<int>(c));
      if (contact_slot != nullptr &&
          contact_slot->state == core::EvacContactState::kActive) {
        core::log_info("[evac-cell-contact] refill_blocked_by_contact cell=" +
                       std::to_string(c));
        continue;
      }
      if (!(evac_cell_min_corner_jacobian(
                static_cast<int>(c), topo.nz, node_r, node_z) > 0.0)) {
        core::log_info("[evac-cell-remat] rejected cell=" +
                       std::to_string(c) + " reason=folded_geometry");
        continue;
      }
      if (!std::isfinite(evacuated.controller_previous_volume[c])) {
        evacuated.controller_previous_volume[c] = vol[c];
      }
      const EvacCellRematerializeTrigger trigger =
          evacuated_cell_rematerialize_trigger(
              vol[c],
              evacuated.controller_previous_volume[c],
              evacuated.vol_ref[c],
              config.rematerialize_volume_fraction,
              config.rematerialize_after_evaluations,
              evacuated.controller_evaluations_since_conversion[c]);
      evacuated.controller_previous_volume[c] = vol[c];
      if (trigger == EvacCellRematerializeTrigger::NONE) {
        continue;
      }
      const char* trigger_name =
          trigger == EvacCellRematerializeTrigger::TIMED
              ? "timed"
              : (trigger == EvacCellRematerializeTrigger::VOLUME ? "volume"
                                                                  : "predicted");
      const int cell = static_cast<int>(c);
      const EvacCellFaceWeights face_weights = evacuated_cell_face_weights(
          topo.nr,
          topo.nz,
          cell,
          node_r,
          node_z,
          evacuated.inactive_member_mask);
      if (face_weights.recipients.empty()) {
        core::log_info("[evac-cell-remat] rejected cell=" +
                       std::to_string(cell) + " reason=insufficient_fill");
        continue;
      }

      std::vector<double> trial_mass = mass;
      std::vector<double> trial_ee = ee;
      std::vector<double> trial_ei = ei;
      const EvacCellRematerializeTransfer transfer =
          rematerialize_evacuated_cell_extensives(
              cell,
              vol[c],
              face_weights.recipients,
              face_weights.weights,
              config.rematerialize_neighbor_change_max,
              rho,
              trial_mass,
              trial_ee,
              trial_ei);
      if (!transfer.accepted) {
        core::log_info("[evac-cell-remat] rejected cell=" +
                       std::to_string(cell) + " reason=insufficient_fill");
        continue;
      }
      std::vector<double> donor_rho;
      donor_rho.reserve(face_weights.recipients.size());
      for (const int donor : face_weights.recipients) {
        donor_rho.push_back(rho[static_cast<std::size_t>(donor)]);
      }
      const double rho_donor_median = median(std::move(donor_rho));
      const double rho_fill =
          transfer.mass / std::max(vol[c], std::numeric_limits<double>::min());
      if (!evacuated_cell_refill_viable(transfer.mass,
                                        evacuated.mass_ref[c],
                                        rho_fill,
                                        rho_donor_median,
                                        config.closure_contact)) {
        std::ostringstream line;
        line << "[evac-cell-contact] refill_refused cell=" << cell
             << " m_fill=" << std::scientific << std::setprecision(17)
             << transfer.mass << " rho_fill=" << rho_fill
             << " median=" << rho_donor_median;
        core::log_info(line.str());
        continue;
      }
      mass.swap(trial_mass);
      ee.swap(trial_ee);
      ei.swap(trial_ei);

      transfer_rematerialized_corner_masses(cell,
                                             face_weights.recipients,
                                             transfer.donor_mass_withdrawals,
                                             corner_mass);
      for (const int donor_cell : face_weights.recipients) {
        const std::size_t donor = static_cast<std::size_t>(donor_cell);
        rho[donor] = mass[donor] / vol[donor];
      }
      rho[c] = mass[c] / vol[c];
      // Te/Ti/Pe/Pi intentionally remain unchanged; per-step EOS closure
      // re-derives them on the next step. The remaining convergence continues
      // naturally; later cycles top up a partially filled live cell.
      evacuated.inactive_member_mask[c] = 0U;
      state.contact_graph.records.erase(
          std::remove_if(
              state.contact_graph.records.begin(),
              state.contact_graph.records.end(),
              [cell](const core::EvacContactSlot& slot) {
                return slot.cell == cell;
              }),
          state.contact_graph.records.end());
      evacuated.controller_previous_contact_gap[c] =
          {std::numeric_limits<double>::quiet_NaN(),
           std::numeric_limits<double>::quiet_NaN()};
      evacuated.d_inactive_member_mask.copy_from_host(
          evacuated.inactive_member_mask);
      evacuated.off_streak[c] = 0;
      evacuated.controller_dwell_remaining[c] =
          config.rematerialize_dwell_evaluations;
      evacuated.controller_evaluations_since_conversion[c] = 0;
      ++rematerialized;

      std::ostringstream donors_stream;
      std::ostringstream weights_stream;
      donors_stream << "[";
      weights_stream << "[" << std::scientific << std::setprecision(17);
      for (std::size_t r = 0; r < face_weights.recipients.size(); ++r) {
        if (r != 0U) {
          donors_stream << ",";
          weights_stream << ",";
        }
        donors_stream << face_weights.recipients[r];
        weights_stream << face_weights.weights[r];
      }
      donors_stream << "]";
      weights_stream << "]";
      const int i = cell / topo.nz;
      const int j = cell - i * topo.nz;
      std::ostringstream line;
      line << "[evac-cell-remat] step=" << state.step << " t="
           << std::scientific << std::setprecision(17) << state.t
           << " cell=" << cell << " i=" << i << " j=" << j
           << " m_new=" << transfer.mass
           << " fill=" << transfer.fill_fraction
           << " Ue=" << transfer.electron_energy
           << " Ui=" << transfer.ion_energy
           << " donors=" << donors_stream.str()
           << " weights=" << weights_stream.str() << " vol=" << vol[c]
           << " vol_ref_frac=" << vol[c] / evacuated.vol_ref[c]
           << " trigger=" << trigger_name;
      core::log_info(line.str());
    }

    if (rematerialized > 0) {
      const double mass_after_remat = total_cell_mass(mass);
      const double energy_after_remat =
          total_cell_internal_energy(mass, ee, ei);
      const double tolerance_factor =
          64.0 * std::numeric_limits<double>::epsilon();
      TENRYU_ASSERT(std::abs(mass_after_remat - mass_before_remat) <=
                        tolerance_factor * mass_abs_scale_remat,
                    "evacuated-cell rematerialization mass conservation audit failed");
      TENRYU_ASSERT(
          std::abs(energy_after_remat - energy_before_remat) <=
              tolerance_factor * energy_abs_scale_remat,
          "evacuated-cell rematerialization energy conservation audit failed");
    }
  }

  const auto arm_contact_cell = [&](const int cell, const bool force_arm) {
    if (!config.closure_contact.enabled ||
        find_contact_slot(state.contact_graph.records, cell) != nullptr) {
      return;
    }
    const EvacContactAxisSelection selection = evacuated_cell_closing_axis(
        cell,
        topo.nz,
        node_r,
        node_z,
        evacuated.x_r_ref,
        evacuated.x_z_ref);
    const EvacContactGeometry geometry = evacuated_cell_contact_geometry(
        cell, selection.axis, topo.nz, node_r, node_z);
    const double g0 = config.closure_contact.gap_floor_fraction *
                      selection.h_perp_ref;
    const double g_arm = config.closure_contact.gap_arm_fraction *
                         selection.h_perp_ref;
    const std::array<double, 2> previous_gap_pair =
        evacuated.controller_previous_contact_gap[static_cast<std::size_t>(cell)];
    evacuated.controller_previous_contact_gap[static_cast<std::size_t>(cell)] =
        {geometry.gap_pair[0], geometry.gap_pair[1]};
    if (!force_arm &&
        !evacuated_cell_contact_should_arm(
            geometry.gap_pair, previous_gap_pair.data(), g_arm)) {
      return;
    }
    // Contiguous contact-front growth needs additional slot headroom.
    TENRYU_ASSERT(state.contact_graph.records.size() < 16U,
                  "evacuated-cell contact supports at most 16 slots in v1");
    core::EvacContactSlot slot;
    slot.cell = cell;
    slot.axis = selection.axis;
    for (int pair = 0; pair < 2; ++pair) {
      slot.node_a[pair] = geometry.node_a[pair];
      slot.node_b[pair] = geometry.node_b[pair];
      slot.normal_pair_r[pair] = geometry.normal_pair_r[pair];
      slot.normal_pair_z[pair] = geometry.normal_pair_z[pair];
      slot.gap_pair[pair] = geometry.gap_pair[pair];
      slot.gap_prev_pair[pair] = previous_gap_pair[pair];
    }
    slot.h_perp_ref = selection.h_perp_ref;
    slot.g0 = g0;
    slot.g_arm = g_arm;
    slot.gap = geometry.gap;
    slot.gap_prev = std::fmin(previous_gap_pair[0], previous_gap_pair[1]);
    slot.state = core::EvacContactState::kArmed;
    state.contact_graph.records.push_back(slot);

    std::ostringstream line;
    line << "[evac-cell-contact] armed cell=" << cell
         << " axis=" << selection.axis << " gap=" << std::scientific
         << std::setprecision(17) << geometry.gap
         << " gap0=" << geometry.gap_pair[0]
         << " gap1=" << geometry.gap_pair[1]
         << " n0=(" << geometry.normal_pair_r[0] << ","
         << geometry.normal_pair_z[0] << ")"
         << " n1=(" << geometry.normal_pair_r[1] << ","
         << geometry.normal_pair_z[1] << ")" << " g0=" << g0
         << " g_arm=" << g_arm << " h_ref=" << selection.h_perp_ref;
    core::log_info(line.str());
  };

  if (config.closure_contact.enabled) {
    for (std::size_t c = 0; c < n_cells; ++c) {
      if (evacuated.inactive_member_mask[c] != 0U &&
          find_contact_slot(state.contact_graph.records,
                            static_cast<int>(c)) == nullptr) {
        arm_contact_cell(static_cast<int>(c), false);
      }
    }
    for (int j = 0; j < topo.nz; ++j) {
      const int cell = j;
      const std::size_t c = static_cast<std::size_t>(cell);
      if (find_contact_slot(state.contact_graph.records, cell) != nullptr ||
          (c < evacuated.geometry_policy_exempt_cells.size() &&
           evacuated.geometry_policy_exempt_cells[c] != 0U)) {
        continue;
      }
      bool adjacent_to_contact_front = std::any_of(
          state.contact_graph.records.begin(),
          state.contact_graph.records.end(),
          [&](const core::EvacContactSlot& slot) {
            return std::abs(slot.cell % topo.nz - j) <= 1;
          });
      for (std::size_t collapsed_cell = 0;
           !adjacent_to_contact_front &&
           collapsed_cell < evacuated.cell_axis_edge_collapsed.size();
           ++collapsed_cell) {
        adjacent_to_contact_front =
            evacuated.cell_axis_edge_collapsed[collapsed_cell] != 0U &&
            std::abs(static_cast<int>(collapsed_cell) % topo.nz - j) <= 1;
      }
      if (adjacent_to_contact_front) {
        arm_contact_cell(cell, false);
      }
    }
  }

  const auto upload_material_state = [&]() {
    state.mass.copy_from_host(mass);
    state.rho.copy_from_host(rho);
    state.ee.copy_from_host(ee);
    state.ei.copy_from_host(ei);
    state.Te.copy_from_host(Te);
    state.Ti.copy_from_host(Ti);
    state.Pe.copy_from_host(Pe);
    state.Pi.copy_from_host(Pi);
    state.corner_mass.copy_from_host(corner_mass);
  };

  const double Abar = cfg.materials.materials.front().A;
  const double lambda_um = config.laser_wavelength_nm * 1.0e-3;
  const double n_crit = 1.115e21 / (lambda_um * lambda_um);
  std::vector<double> ne_over_ncrit(n_cells, 0.0);
  for (std::size_t c = 0; c < n_cells; ++c) {
    const double n_e =
        rho[c] * std::max(zbar[c], 0.0) /
        (Abar * core::constants::proton_mass);
    ne_over_ncrit[c] = n_e / n_crit;
  }

  const EvacCellShadowParams classify_params =
      {config.arm_mass_fraction,
       config.off_mass_fraction,
       config.rho_vacuum_policy_g_per_cc};
  (void)classify_evacuated_cells(mass,
                                 evacuated.mass_ref,
                                 evacuated.vol_ref,
                                 rho,
                                 ne_over_ncrit,
                                 Te,
                                 classify_params,
                                 &evacuated.inactive_member_mask);

  struct GateRejectionCacheEntry {
    std::string gate;
    double value = 0.0;
    bool valid = false;
  };
  struct GateRejectionCache {
    const core::State* state = nullptr;
    std::vector<GateRejectionCacheEntry> cells;
  };
  static GateRejectionCache gate_rejection_cache;
  if (gate_rejection_cache.state != &state ||
      gate_rejection_cache.cells.size() != n_cells) {
    gate_rejection_cache.state = &state;
    gate_rejection_cache.cells.assign(n_cells, {});
  }
  const auto log_gate_rejection = [&](const std::size_t cell,
                                      const char* gate,
                                      const double value,
                                      const double threshold) {
    GateRejectionCacheEntry& cached = gate_rejection_cache.cells[cell];
    bool materially_changed = !cached.valid || cached.gate != gate;
    if (!materially_changed) {
      const double value_abs = std::abs(value);
      const double cached_abs = std::abs(cached.value);
      if (std::isnan(value_abs) || std::isnan(cached_abs)) {
        materially_changed = std::isnan(value_abs) != std::isnan(cached_abs);
      } else if (std::isinf(value_abs) || std::isinf(cached_abs)) {
        materially_changed = value_abs != cached_abs;
      } else {
        materially_changed = value_abs > 2.0 * cached_abs ||
                             cached_abs > 2.0 * value_abs;
      }
    }
    if (!materially_changed) {
      return;
    }
    cached.gate = gate;
    cached.value = value;
    cached.valid = true;
    std::ostringstream line;
    line << "[evac-cell-gate] step=" << state.step << " cell=" << cell
         << " rejected=" << gate << " value=" << std::scientific
         << std::setprecision(17) << value << " threshold=" << threshold;
    core::log_info(line.str());
  };

  struct ConversionCandidate {
    int cell = -1;
    bool live_closure = false;
  };
  std::vector<ConversionCandidate> candidates;
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (consume_evacuated_cell_rematerialize_dwell(
            evacuated.controller_dwell_remaining[c])) {
      evacuated.off_streak[c] = 0;
      gate_rejection_cache.cells[c].valid = false;
      continue;
    }
    const bool active = evacuated.inactive_member_mask[c] == 0U;
    const bool eligible =
        active && evacuated_cell_off_eligible(c,
                                              mass,
                                              evacuated.mass_ref,
                                              evacuated.vol_ref,
                                              classify_params);
    const bool live_closure =
        active && config.closure_contact.enabled &&
        evacuated_cell_live_closure_eligible(mass[c],
                                              evacuated.mass_ref[c],
                                              vol[c],
                                              evacuated.vol_ref[c],
                                              evacuated.closure_lineage[c],
                                              config.closure_contact);
    const bool off_streak_satisfied = update_evacuated_cell_off_streak(
        eligible,
        config.off_hold_evaluations,
        evacuated.off_streak[c]);
    if ((!off_streak_satisfied && !live_closure) || !active) {
      gate_rejection_cache.cells[c].valid = false;
      continue;
    }
    const int candidate = static_cast<int>(c);
    if (evacuated_cell_is_domain_corner(topo.nr, topo.nz, candidate)) {
      log_gate_rejection(c, "domain_corner", 1.0, 0.0);
      continue;
    }
    candidates.push_back({candidate, live_closure});
  }
  if (candidates.size() >
      static_cast<std::size_t>(config.max_cells_per_event)) {
    core::log_info("[evac-cell-txn] rejected count=" +
                   std::to_string(candidates.size()));
    if (rematerialized > 0) {
      upload_material_state();
    }
    return;
  }

  const double mass_before = total_cell_mass(mass);
  const double energy_before = total_cell_internal_energy(mass, ee, ei);
  double mass_abs_scale = 0.0;
  for (const double value : mass) {
    mass_abs_scale += std::abs(value);
  }
  const double energy_abs_scale =
      total_cell_internal_energy_abs(mass, ee, ei);
  int converted = 0;
  for (const ConversionCandidate candidate : candidates) {
    const int donor = candidate.cell;
    const std::size_t c = static_cast<std::size_t>(donor);
    if (!(ne_over_ncrit[c] < config.laser_ne_over_ncrit_max)) {
      log_gate_rejection(c,
                         "laser",
                         ne_over_ncrit[c],
                         config.laser_ne_over_ncrit_max);
      continue;
    }

    double patch_energy = mass[c] * (ee[c] + ei[c]);
    const int i = donor / topo.nz;
    const int j = donor - i * topo.nz;
    if (i > 0) {
      const std::size_t neighbor = static_cast<std::size_t>((i - 1) * topo.nz + j);
      patch_energy += mass[neighbor] * (ee[neighbor] + ei[neighbor]);
    }
    if (i + 1 < topo.nr) {
      const std::size_t neighbor = static_cast<std::size_t>((i + 1) * topo.nz + j);
      patch_energy += mass[neighbor] * (ee[neighbor] + ei[neighbor]);
    }
    if (j > 0) {
      const std::size_t neighbor = static_cast<std::size_t>(i * topo.nz + j - 1);
      patch_energy += mass[neighbor] * (ee[neighbor] + ei[neighbor]);
    }
    if (j + 1 < topo.nz) {
      const std::size_t neighbor = static_cast<std::size_t>(i * topo.nz + j + 1);
      patch_energy += mass[neighbor] * (ee[neighbor] + ei[neighbor]);
    }
    double coupling_energy =
        std::abs(conduction_e_rate[c]) * vol[c] * state.dt;
    if (rho[c] > config.rho_vacuum_policy_g_per_cc) {
      // Consultation #21 §4.1: "a vacuum cell has no physical dose"; below
      // rho_vacuum_policy the cell is outside the continuum regime, so the
      // laser deposition value is a model artifact.
      coupling_energy += std::abs(laser_dep[c]);
    }
    const double coupling_fraction =
        coupling_energy / std::max(patch_energy, 1.0e-300);
    if (!(coupling_fraction < config.coupling_fraction_max)) {
      log_gate_rejection(c,
                         "coupling",
                         coupling_fraction,
                         config.coupling_fraction_max);
      continue;
    }

    const EvacCellFaceWeights face_weights = evacuated_cell_face_weights(
        topo.nr,
        topo.nz,
        donor,
        node_r,
        node_z,
        evacuated.inactive_member_mask);
    if (face_weights.recipients.empty()) {
      log_gate_rejection(c, "recipient", 0.0, 1.0);
      continue;
    }
    const std::vector<int> corner_mass_targets =
        evacuated_cell_corner_mass_targets(topo.nr,
                                           topo.nz,
                                           donor,
                                           face_weights.recipients,
                                           face_weights.weights);
    if (corner_mass_targets.size() != 4U) {
      log_gate_rejection(c,
                         "recipient",
                         static_cast<double>(corner_mass_targets.size()),
                         4.0);
      continue;
    }
    bool realizable = true;
    double recipient_value = 0.0;
    for (std::size_t r = 0; r < face_weights.recipients.size(); ++r) {
      const std::size_t recipient =
          static_cast<std::size_t>(face_weights.recipients[r]);
      const double new_mass = mass[recipient] + face_weights.weights[r] * mass[c];
      const double old_Ue = mass[recipient] * ee[recipient];
      const double new_Ue =
          old_Ue + face_weights.weights[r] * mass[c] * ee[c];
      if (!(new_mass > 0.0) || !(vol[recipient] > 0.0)) {
        realizable = false;
        recipient_value = std::numeric_limits<double>::infinity();
        break;
      }
      const double new_rho = new_mass / vol[recipient];
      const double new_ee = new_Ue / new_mass;
      const double rho_change =
          std::abs(new_rho - rho[recipient]) /
          std::max(std::abs(rho[recipient]), 1.0e-300);
      const double ee_change =
          std::abs(new_ee - ee[recipient]) /
          std::max(std::abs(ee[recipient]), 1.0e-300);
      recipient_value = std::max(rho_change, ee_change);
      if (!(rho_change < 1.0e-3) || !(ee_change < 1.0e-3)) {
        realizable = false;
        break;
      }
    }
    if (!realizable) {
      log_gate_rejection(c, "recipient", recipient_value, 1.0e-3);
      continue;
    }

    const double donor_mass = mass[c];
    const double donor_Ue = donor_mass * ee[c];
    const double donor_Ui = donor_mass * ei[c];
    const auto momentum_before =
        corner_momentum(topo.nr, topo.nz, corner_mass, v_r, v_z);
    transfer_evacuated_cell_extensives(donor,
                                       face_weights.recipients,
                                       face_weights.weights,
                                       mass,
                                       ee,
                                       ei);
    for (const int recipient_cell : face_weights.recipients) {
      const std::size_t recipient = static_cast<std::size_t>(recipient_cell);
      rho[recipient] = mass[recipient] / vol[recipient];
    }
    rho[c] = 0.0;
    Te[c] = 0.0;
    Ti[c] = 0.0;
    Pe[c] = 0.0;
    Pi[c] = 0.0;
    transfer_donor_corner_masses(donor,
                                 corner_mass_targets,
                                 corner_mass);
    const auto momentum_after =
        corner_momentum(topo.nr, topo.nz, corner_mass, v_r, v_z);

    evacuated.inactive_member_mask[c] = 1U;
    evacuated.closure_lineage[c] = 1U;
    evacuated.controller_evaluations_since_conversion[c] = 0;
    evacuated.controller_previous_volume[c] = vol[c];
    // §21 v2a adjudication: FLD retains the evacuated cell. Hydro, CFL, and
    // local conduction consume this mask; radiation intentionally does not.
    evacuated.d_inactive_member_mask.copy_from_host(
        evacuated.inactive_member_mask);
    ++evacuated.conversions_total;
    ++converted;
    if (candidate.live_closure) {
      arm_contact_cell(donor, true);
    }

    std::ostringstream recipients_stream;
    std::ostringstream weights_stream;
    recipients_stream << "[";
    weights_stream << "[" << std::scientific << std::setprecision(17);
    for (std::size_t r = 0; r < face_weights.recipients.size(); ++r) {
      if (r != 0U) {
        recipients_stream << ",";
        weights_stream << ",";
      }
      recipients_stream << face_weights.recipients[r];
      weights_stream << face_weights.weights[r];
    }
    recipients_stream << "]";
    weights_stream << "]";
    std::ostringstream line;
    line << "[evac-cell-txn] step=" << state.step << " t=" << std::scientific
         << std::setprecision(17) << state.t << " cell=" << donor
         << " i=" << i << " j=" << j << " m_transferred=" << donor_mass
         << " Ue_transferred=" << donor_Ue
         << " Ui_transferred=" << donor_Ui
         << " recipients=" << recipients_stream.str()
         << " weights=" << weights_stream.str()
         << " mom_delta_r=" << (momentum_after.first - momentum_before.first)
         << " mom_delta_z=" << (momentum_after.second - momentum_before.second);
    if (candidate.live_closure) {
      line << " trigger=live_closure";
    }
    core::log_info(line.str());
  }

  if (converted == 0) {
    if (rematerialized > 0) {
      upload_material_state();
    }
    return;
  }
  const double mass_after = total_cell_mass(mass);
  const double energy_after = total_cell_internal_energy(mass, ee, ei);
  const double tolerance_factor =
      64.0 * std::numeric_limits<double>::epsilon();
  TENRYU_ASSERT(std::abs(mass_after - mass_before) <=
                    tolerance_factor * mass_abs_scale,
                "evacuated-cell transaction mass conservation audit failed");
  TENRYU_ASSERT(std::abs(energy_after - energy_before) <=
                    tolerance_factor * energy_abs_scale,
                "evacuated-cell transaction energy conservation audit failed");

  upload_material_state();
}

void evacuated_cell_contact_probe(core::State& state,
                                  const core::Config& cfg,
                                  const char* const label) {
  static const bool enabled = []() {
    const char* const value = std::getenv("TENRYU_EVAC_CONTACT_PROBE");
    return value != nullptr && std::string(value) != "0";
  }();
  if (!enabled) {
    return;
  }
  (void)cfg;

  const auto& evacuated = state.evacuated_cells;
  if (evacuated.n_active_pairs <= 0) {
    return;
  }

  std::vector<double> v_r;
  std::vector<double> v_z;
  std::vector<double> x_r;
  std::vector<double> x_z;
  state.v_r.copy_to_host(v_r);
  state.v_z.copy_to_host(v_z);
  state.x_r.copy_to_host(x_r);
  state.x_z.copy_to_host(x_z);

  for (const core::EvacContactSlot& slot : state.contact_graph.records) {
    if (slot.state != core::EvacContactState::kActive) {
      continue;
    }
    for (int pair = 0; pair < 2; ++pair) {
      if (slot.pair_engaged[pair] == 0U) {
        continue;
      }
      const std::size_t a = static_cast<std::size_t>(slot.node_a[pair]);
      const std::size_t b = static_cast<std::size_t>(slot.node_b[pair]);
      const double u_c =
          (v_r[a] - v_r[b]) * slot.normal_pair_r[pair] +
          (v_z[a] - v_z[b]) * slot.normal_pair_z[pair];
      const double gap =
          (x_r[b] - x_r[a]) * slot.normal_pair_r[pair] +
          (x_z[b] - x_z[a]) * slot.normal_pair_z[pair];
      std::ostringstream line;
      line << "[evac-cell-probe] step=" << state.step << " label=" << label
           << " cell=" << slot.cell << " pair=" << pair << " u_c="
           << std::scientific << std::setprecision(17) << u_c
           << " gap=" << gap << " vAr=" << v_r[a] << " vAz=" << v_z[a]
           << " vBr=" << v_r[b] << " vBz=" << v_z[b]
           << " nr=" << slot.normal_pair_r[pair]
           << " nz=" << slot.normal_pair_z[pair]
           << " nA=" << slot.node_a[pair] << " nB=" << slot.node_b[pair];
      core::log_info(line.str());
    }
  }
}

void evacuated_cell_contact_step(core::State& state,
                                 const core::Config& cfg) {
  const auto& evacuated_config = cfg.numerics.ale.evacuated_cell;
  auto& evacuated = state.evacuated_cells;
  if (!evacuated_config.enabled) {
    return;
  }
  const auto& config = evacuated_config.closure_contact;
  if (!config.enabled || state.contact_graph.records.empty()) {
    evacuated.contact_dt_cap_s = std::numeric_limits<double>::infinity();
    return;
  }

  const auto& topo = state.mesh.topo;
  TENRYU_ASSERT(cfg.main.dim == 2 &&
                    cfg.mesh.topology_scheme ==
                        core::TopologyScheme::SINGLE_BLOCK &&
                    !topo.multiblock.has_value() &&
                    topo.n_cells == topo.nr * topo.nz &&
                    state.contact_graph.records.size() <= 16U,
                "evacuated-cell contact requires structured 2D single block");

  std::vector<double> node_r;
  std::vector<double> node_z;
  state.x_r.copy_to_host(node_r);
  state.x_z.copy_to_host(node_z);

  bool engagement_possible = false;
  bool active_slot_present = false;
  for (const core::EvacContactSlot& slot : state.contact_graph.records) {
    const bool any_pair_engaged =
        slot.pair_engaged[0] != 0U || slot.pair_engaged[1] != 0U;
    TENRYU_ASSERT((slot.state == core::EvacContactState::kActive) ==
                      any_pair_engaged,
                  "evacuated-cell contact active state mismatch");
    for (int pair = 0; pair < 2; ++pair) {
      engagement_possible =
          engagement_possible ||
          ((slot.state == core::EvacContactState::kArmed ||
            slot.state == core::EvacContactState::kActive) &&
           slot.pair_engaged[pair] == 0U);
    }
    active_slot_present =
        active_slot_present || slot.state == core::EvacContactState::kActive;
  }

  if (active_slot_present) {
    const std::size_t hold_count =
        static_cast<std::size_t>(evacuated.n_contact_active_cells) *
        core::kEvacContactMortarRowCapacity;
    TENRYU_ASSERT(
        evacuated.d_contact_mortar_g_hold.size() == hold_count &&
            evacuated.d_contact_mortar_g_hold_valid.size() == hold_count,
        "evacuated-cell mortar hold device state mismatch");
    std::vector<double> mortar_g_hold;
    std::vector<std::uint8_t> mortar_g_hold_valid;
    evacuated.d_contact_mortar_g_hold.copy_to_host(mortar_g_hold);
    evacuated.d_contact_mortar_g_hold_valid.copy_to_host(
        mortar_g_hold_valid);
    std::size_t active = 0U;
    for (core::EvacContactSlot& slot : state.contact_graph.records) {
      if (slot.state != core::EvacContactState::kActive) {
        continue;
      }
      for (int q = 0; q < core::kEvacContactMortarRowCapacity; ++q) {
        const std::size_t index =
            active * core::kEvacContactMortarRowCapacity +
            static_cast<std::size_t>(q);
        slot.mortar_g_hold[q] = mortar_g_hold[index];
        slot.mortar_g_hold_valid[q] = mortar_g_hold_valid[index];
      }
      ++active;
    }
    TENRYU_ASSERT(active ==
                      static_cast<std::size_t>(
                          evacuated.n_contact_active_cells),
                  "evacuated-cell mortar hold active count mismatch");
  }

  std::vector<double> v_r;
  std::vector<double> v_z;
  std::vector<double> vol;
  if (engagement_possible || active_slot_present) {
    state.v_r.copy_to_host(v_r);
    state.v_z.copy_to_host(v_z);
    state.vol.copy_to_host(vol);
  }

  std::vector<int> active_pair_base(state.contact_graph.records.size(), -1);
  std::vector<std::array<int, 2>> active_pair_offset(
      state.contact_graph.records.size(), {-1, -1});
  int active_pair_count = 0;
  for (std::size_t s = 0; s < state.contact_graph.records.size(); ++s) {
    int slot_pair_count = 0;
    for (int pair = 0; pair < 2; ++pair) {
      if (state.contact_graph.records[s].pair_engaged[pair] == 0U) {
        continue;
      }
      if (active_pair_base[s] < 0) {
        active_pair_base[s] = active_pair_count;
      }
      active_pair_offset[s][pair] = slot_pair_count;
      ++slot_pair_count;
      ++active_pair_count;
    }
  }
  TENRYU_ASSERT(active_pair_count == evacuated.n_active_pairs,
                "evacuated-cell contact active pair count mismatch");
  std::vector<double> pair_lambda;
  std::vector<double> pair_dk;
  if (active_pair_count > 0) {
    TENRYU_ASSERT(evacuated.d_contact_pair_lambda.size() ==
                      static_cast<std::size_t>(active_pair_count),
                  "evacuated-cell contact lambda size mismatch");
    evacuated.d_contact_pair_lambda.copy_to_host(pair_lambda);
    TENRYU_ASSERT(evacuated.d_contact_pair_dk.size() ==
                      static_cast<std::size_t>(active_pair_count),
                  "evacuated-cell contact projection dK size mismatch");
    evacuated.d_contact_pair_dk.copy_to_host(pair_dk);
  }
  int volume_projection_count = 0;
  int patch_projection_count = 0;
  const bool volume_projection_log_cadence =
      state.step % evacuated_config.every_n_steps == 0;
  const char* const seam_forensic_value =
      std::getenv("TENRYU_EVAC_SEAM_FORENSIC");
  const bool seam_forensic_log_cadence =
      seam_forensic_value != nullptr &&
      std::string(seam_forensic_value) == "1" &&
      volume_projection_log_cadence;
  if (volume_projection_log_cadence &&
      evacuated.d_contact_volume_projection_count.size() == 2U) {
    std::vector<int> count;
    evacuated.d_contact_volume_projection_count.copy_to_host(count);
    volume_projection_count = count[0];
    patch_projection_count = count[1];
  }
  if (volume_projection_log_cadence &&
      evacuated.d_contact_mortar_drift_count.size() == 3U &&
      evacuated.d_contact_mortar_drift_max_ucorr.size() == 1U) {
    std::vector<int> count;
    std::vector<double> max_ucorr;
    evacuated.d_contact_mortar_drift_count.copy_to_host(count);
    evacuated.d_contact_mortar_drift_max_ucorr.copy_to_host(max_ucorr);
    if (count[0] > 0 || count[1] > 0 || count[2] > 0) {
      std::ostringstream line;
      line << "[mortar] position drift correction engaged rows="
           << count[0] << " max_ucorr=" << std::scientific
           << std::setprecision(3) << max_ucorr[0]
           << " structural=" << count[1]
           << " ke_reject=" << count[2];
      core::log_info(line.str());
    }
    const std::vector<int> zero_count(3U, 0);
    const std::vector<double> zero_max(1U, 0.0);
    evacuated.d_contact_mortar_drift_count.copy_from_host(zero_count);
    evacuated.d_contact_mortar_drift_max_ucorr.copy_from_host(zero_max);
  }

  std::vector<double> Pe;
  std::vector<double> Pi;
  std::vector<double> mass;
  if (active_slot_present) {
    state.Pe.copy_to_host(Pe);
    state.Pi.copy_to_host(Pi);
    state.mass.copy_to_host(mass);
  }

  std::vector<double> corner_mass;
  std::vector<double> ei;
  bool contact_fields_loaded = false;
  const auto ensure_contact_fields_loaded = [&]() {
    if (contact_fields_loaded) {
      return;
    }
    state.corner_mass.copy_to_host(corner_mass);
    if (mass.empty()) {
      state.mass.copy_to_host(mass);
    }
    state.ei.copy_to_host(ei);
    contact_fields_loaded = true;
  };
  bool contact_fields_dirty = false;
  bool active_pair_set_changed = false;
  double commit_project_dk_sum = 0.0;
  double commit_project_deposited_sum = 0.0;
  evacuated.contact_dt_cap_s = std::numeric_limits<double>::infinity();
  int engaged_gap_invariant_warning_count = 0;

  for (std::size_t s = 0; s < state.contact_graph.records.size(); ++s) {
    core::EvacContactSlot& slot = state.contact_graph.records[s];
    const core::EvacContactState state_at_entry = slot.state;
    const bool engaged_at_entry[2] = {slot.pair_engaged[0] != 0U,
                                      slot.pair_engaged[1] != 0U};
    double gap_pair[2] = {0.0, 0.0};
    const double gap =
        contact_gap_with_frozen_normal(slot, node_r, node_z, gap_pair);

    if (seam_forensic_log_cadence &&
        state_at_entry == core::EvacContactState::kActive) {
      log_seam_forensic(slot,
                        topo.nr,
                        topo.nz,
                        state.step,
                        node_r,
                        node_z,
                        v_r,
                        v_z,
                        gap_pair);
    }

    if (state_at_entry == core::EvacContactState::kActive) {
      const int pair_base = active_pair_base[s];
      TENRYU_ASSERT(pair_base >= 0,
                    "evacuated-cell contact projection dK index mismatch");
      for (int pair = 0; pair < 2; ++pair) {
        if (!engaged_at_entry[pair]) {
          continue;
        }
        const int pair_offset = active_pair_offset[s][pair];
        const int dk_index = pair_base + pair_offset;
        TENRYU_ASSERT(pair_offset >= 0 && dk_index < active_pair_count,
                      "evacuated-cell contact projection dK index mismatch");
        const double projection_dk =
            pair_dk[static_cast<std::size_t>(dk_index)];
        if (!(projection_dk > 0.0)) {
          continue;
        }
        ensure_contact_fields_loaded();
        const std::vector<std::int8_t>* const hydro_active =
            state.hydro_active.empty() ? nullptr : &state.hydro_active;
        const int node_a = slot.node_a[pair];
        const int node_b = slot.node_b[pair];
        const double mA = evacuated_cell_active_node_mass(
            node_a,
            topo.nr,
            topo.nz,
            corner_mass,
            evacuated.inactive_member_mask,
            hydro_active);
        const double mB = evacuated_cell_active_node_mass(
            node_b,
            topo.nr,
            topo.nz,
            corner_mass,
            evacuated.inactive_member_mask,
            hydro_active);
        TENRYU_ASSERT(
            mA > 0.0 && mB > 0.0,
            "evacuated-cell contact projection heat has nonpositive node mass");
        double deposited_heat = 0.0;
        deposit_contact_side_heat(node_a,
                                  projection_dk * mB / (mA + mB),
                                  topo.nr,
                                  topo.nz,
                                  corner_mass,
                                  evacuated.inactive_member_mask,
                                  hydro_active,
                                  mass,
                                  ei,
                                  deposited_heat,
                                  nullptr);
        deposit_contact_side_heat(node_b,
                                  projection_dk * mA / (mA + mB),
                                  topo.nr,
                                  topo.nz,
                                  corner_mass,
                                  evacuated.inactive_member_mask,
                                  hydro_active,
                                  mass,
                                  ei,
                                  deposited_heat,
                                  nullptr);
        commit_project_dk_sum += projection_dk;
        commit_project_deposited_sum += deposited_heat;
        ++slot.reproject_count;
        slot.reproject_heat_total += deposited_heat;
        slot.commit_project_heat_total += deposited_heat;
        contact_fields_dirty = true;
      }
    }

    if (state_at_entry == core::EvacContactState::kArmed ||
        state_at_entry == core::EvacContactState::kActive) {
      double min_t_hit = std::numeric_limits<double>::infinity();
      for (int pair = 0; pair < 2; ++pair) {
        if (engaged_at_entry[pair] ||
            gap_pair[pair] <= slot.g0 ||
            !std::isfinite(slot.gap_prev_pair[pair]) ||
            gap_pair[pair] >= slot.gap_prev_pair[pair]) {
          continue;
        }
        const double t_hit =
            (gap_pair[pair] - slot.g0) * state.dt /
            (slot.gap_prev_pair[pair] - gap_pair[pair]);
        min_t_hit = std::min(min_t_hit, t_hit);
      }
      if (std::isfinite(min_t_hit)) {
        const double cap = std::max(
            {0.5 * min_t_hit,
             kEvacContactDtCapThrottle * state.dt,
             cfg.numerics.dt.min_s});
        evacuated.contact_dt_cap_s =
            std::min(evacuated.contact_dt_cap_s, cap);
      }
    }

    if (state_at_entry == core::EvacContactState::kArmed ||
        state_at_entry == core::EvacContactState::kActive) {
      const bool no_engaged_pair =
          !engaged_at_entry[0] && !engaged_at_entry[1];
      const double corner_j_min =
          no_engaged_pair
              ? evac_cell_min_corner_jacobian(
                    slot.cell, topo.nz, node_r, node_z)
              : std::numeric_limits<double>::infinity();
      const bool crossed = no_engaged_pair && !(corner_j_min > 0.0);
      const int crossed_pair = gap_pair[0] <= gap_pair[1] ? 0 : 1;
      if (crossed) {
        std::ostringstream line;
        line << "[evac-cell-contact] crossed_engage cell=" << slot.cell
             << " pair=" << crossed_pair << " corner_j_min="
             << std::scientific << std::setprecision(17) << corner_j_min
             << " gap0=" << gap_pair[0] << " gap1=" << gap_pair[1];
        core::log_info(line.str());
      }
      for (int pair = 0; pair < 2; ++pair) {
        // Surface-measured engagement (migration phase 4): the moving-side node's
        // point-to-segment gap from the cavity-boundary shadow detector (previous
        // step's geometry) replaces the frozen-normal pair projection.
        double engage_gap = gap_pair[pair];
        double engage_normal_r = slot.normal_pair_r[pair];
        double engage_normal_z = slot.normal_pair_z[pair];
        bool surface_measured = false;
        // After the prior-step controller, the detector records this one-step-old gap;
        // it shares legacy gap_prev's staleness class; the contact dt cap bounds tunneling.
        if (config.surface_engage_enabled) {
          const auto& boundary = state.cavity_boundary;
          const auto it = std::lower_bound(boundary.boundary_nodes.begin(),
                                           boundary.boundary_nodes.end(),
                                           slot.node_b[pair]);
          if (it != boundary.boundary_nodes.end() &&
              *it == slot.node_b[pair]) {
            const std::size_t bn = static_cast<std::size_t>(
                it - boundary.boundary_nodes.begin());
            const std::int32_t best =
                bn < boundary.shadow_best_seg.size()
                    ? boundary.shadow_best_seg[bn]
                    : -1;
            if (best >= 0) {
              engage_gap = boundary.shadow_best_gap[bn];
              engage_normal_r =
                  boundary.seg_normal_r[static_cast<std::size_t>(best)];
              engage_normal_z =
                  boundary.seg_normal_z[static_cast<std::size_t>(best)];
              surface_measured = true;
            }
          }
        }
        if (engaged_at_entry[pair] ||
            (engage_gap > slot.g0 &&
             !(crossed && pair == crossed_pair))) {
          continue;
        }
        TENRYU_ASSERT(engagement_possible && !v_r.empty() && !v_z.empty(),
                      "evacuated-cell contact engagement velocity not staged");
        ensure_contact_fields_loaded();

        const int node_a = slot.node_a[pair];
        const int node_b = slot.node_b[pair];
        const std::vector<std::int8_t>* const hydro_active =
            state.hydro_active.empty() ? nullptr : &state.hydro_active;
        const double mA = evacuated_cell_active_node_mass(
            node_a,
            topo.nr,
            topo.nz,
            corner_mass,
            evacuated.inactive_member_mask,
            hydro_active);
        const double mB = evacuated_cell_active_node_mass(
            node_b,
            topo.nr,
            topo.nz,
            corner_mass,
            evacuated.inactive_member_mask,
            hydro_active);
        TENRYU_ASSERT(mA > 0.0 && mB > 0.0,
                      "evacuated-cell contact engagement has nonpositive node mass");
        const std::size_t a = static_cast<std::size_t>(node_a);
        const std::size_t b = static_cast<std::size_t>(node_b);
        const double vA_n =
            v_r[a] * engage_normal_r + v_z[a] * engage_normal_z;
        const double vB_n =
            v_r[b] * engage_normal_r + v_z[b] * engage_normal_z;
        const double u_c = vA_n - vB_n;
        EvacContactImpact impact;
        double gap_anchor = engage_gap;
        if (surface_measured) {
          slot.normal_pair_r[pair] = engage_normal_r;
          slot.normal_pair_z[pair] = engage_normal_z;
          // Keep the engagement anchor consistent with the slot-pair gap that
          // the engaged-gap invariant monitors under the refreshed normal.
          gap_anchor = (node_r[b] - node_r[a]) * engage_normal_r +
                       (node_z[b] - node_z[a]) * engage_normal_z;
        }
        TENRYU_ASSERT(evacuated_cell_contact_engage_pair(
                          slot,
                          pair,
                          gap_anchor,
                          mA,
                          mB,
                          mass[static_cast<std::size_t>(slot.cell)],
                          vol[static_cast<std::size_t>(slot.cell)],
                          v_r,
                          v_z,
                          impact,
                          (crossed && pair == crossed_pair) || surface_measured,
                          // Keep the first-impact impulse on the legacy path under
                          // lcp_apply. The velocity-level LCP is complementarity-
                          // consistent, so non-closing rows receive no double impulse.
                          false),
                      "evacuated-cell contact pair engagement mismatch");

        const double kinetic_before =
            0.5 * mA * vA_n * vA_n + 0.5 * mB * vB_n * vB_n;
        const double kinetic_after =
            0.5 * mA * impact.vA_n * impact.vA_n +
            0.5 * mB * impact.vB_n * impact.vB_n;

        double deposited_heat = 0.0;
        std::vector<int> heat_cells;
        deposit_contact_side_heat(node_a,
                                  impact.dK_A,
                                  topo.nr,
                                  topo.nz,
                                  corner_mass,
                                  evacuated.inactive_member_mask,
                                  hydro_active,
                                  mass,
                                  ei,
                                  deposited_heat,
                                  &heat_cells);
        deposit_contact_side_heat(node_b,
                                  impact.dK_B,
                                  topo.nr,
                                  topo.nz,
                                  corner_mass,
                                  evacuated.inactive_member_mask,
                                  hydro_active,
                                  mass,
                                  ei,
                                  deposited_heat,
                                  &heat_cells);

        const double lost_kinetic = kinetic_before - kinetic_after;
        const double scale = std::max(
            {std::abs(kinetic_before),
             std::abs(kinetic_after),
             std::abs(deposited_heat),
             std::numeric_limits<double>::min()});
        TENRYU_ASSERT(
            std::abs(lost_kinetic - deposited_heat) <=
                64.0 * std::numeric_limits<double>::epsilon() * scale,
            "evacuated-cell contact impact energy audit failed");
        slot.tensile_streak_pair[pair] = 0;
        slot.impact_heat_total += deposited_heat;
        ++slot.engage_count;
        active_pair_set_changed = true;
        contact_fields_dirty = true;

        std::ostringstream heat_cells_stream;
        heat_cells_stream << "[";
        for (std::size_t index = 0; index < heat_cells.size(); ++index) {
          if (index != 0U) {
            heat_cells_stream << ",";
          }
          heat_cells_stream << heat_cells[index];
        }
        heat_cells_stream << "]";
        std::ostringstream line;
        line << "[evac-cell-contact] engaged cell=" << slot.cell
             << " pair=" << pair << " step=" << state.step
             << " t=" << std::scientific << std::setprecision(17)
             << state.t << " gap=" << gap_anchor << " u_c=" << u_c
             << " J=" << impact.J << " dK=" << deposited_heat
             << " heat_cells=" << heat_cells_stream.str()
             << " state=active"
             << " surface=" << (surface_measured ? 1 : 0);
        core::log_info(line.str());
      }
    }

    if (slot.state == core::EvacContactState::kActive) {
      TENRYU_ASSERT(!v_r.empty() && !v_z.empty(),
                    "evacuated-cell contact backstop velocity not staged");
      for (int pair = 0; pair < 2; ++pair) {
        if (slot.pair_engaged[pair] == 0U) {
          continue;
        }
        const std::size_t a =
            static_cast<std::size_t>(slot.node_a[pair]);
        const std::size_t b =
            static_cast<std::size_t>(slot.node_b[pair]);
        const double vA_n = v_r[a] * slot.normal_pair_r[pair] +
                            v_z[a] * slot.normal_pair_z[pair];
        const double vB_n = v_r[b] * slot.normal_pair_r[pair] +
                            v_z[b] * slot.normal_pair_z[pair];
        if (!(vA_n - vB_n > 0.0)) {
          continue;
        }
        ensure_contact_fields_loaded();
        const std::vector<std::int8_t>* const hydro_active =
            state.hydro_active.empty() ? nullptr : &state.hydro_active;
        TENRYU_ASSERT(
            evacuated_cell_contact_reproject_pair(
                slot,
                pair,
                topo.nr,
                topo.nz,
                corner_mass,
                evacuated.inactive_member_mask,
                hydro_active,
                mass,
                v_r,
                v_z,
                ei),
            "evacuated-cell contact backstop projection mismatch");
        contact_fields_dirty = true;
      }
    }

    if (slot.state == core::EvacContactState::kActive) {
      for (int pair = 0; pair < 2; ++pair) {
        if (slot.pair_engaged[pair] != 0U) {
          if (config.lcp_apply_enabled) {
            const double floor =
                evacuated_cell_contact_gap_invariant_floor(slot, pair);
            if (!(gap_pair[pair] >= floor) &&
                engaged_gap_invariant_warning_count < 4) {
              std::ostringstream line;
              line << "[evac-cell-contact] engaged-gap invariant drift cell="
                   << slot.cell << " pair=" << pair << " gap="
                   << std::scientific << std::setprecision(17)
                   << gap_pair[pair] << " floor=" << floor;
              core::log_warning(line.str());
              ++engaged_gap_invariant_warning_count;
            }
          } else {
            TENRYU_ASSERT(
                gap_pair[pair] >=
                    evacuated_cell_contact_gap_invariant_floor(slot, pair),
                "evacuated-cell contact gap invariant failed");
          }
        }
      }
      const std::size_t cell = static_cast<std::size_t>(slot.cell);
      if (slot.vol_at_engagement > 0.0 && slot.devolumized == 0U &&
          !(vol[cell] >= 0.5 * kEvacContactVolumeFloorFraction *
                             slot.vol_at_engagement)) {
        if (config.seam_interface_owner_enabled &&
            evacuated_cell_seam_devolumize(
                state, cfg, slot, mass, vol)) {
          // Converted; the invariant is retired for this slot.
        } else {
          TENRYU_ASSERT(false,
                        "evacuated-cell contact volume invariant failed");
        }
      }
      const double m_off =
          std::max(evacuated_config.off_mass_fraction *
                       evacuated.mass_ref[cell],
                   evacuated_config.rho_vacuum_policy_g_per_cc *
                       evacuated.vol_ref[cell]);
      // Kernel-level cross-contact flux zeroing is the §22 Stage-3 item;
      // this invariant keeps v1 honest.
      TENRYU_ASSERT(std::abs(mass[cell] - slot.mass_at_engagement) < m_off,
                    "evacuated-cell contact mass invariant failed");
    }

    if (state_at_entry == core::EvacContactState::kActive) {
      const int pair_base = active_pair_base[s];
      TENRYU_ASSERT(pair_base >= 0,
                    "evacuated-cell contact release pair index mismatch");
      const int i = slot.cell / topo.nz;
      const int j = slot.cell - i * topo.nz;
      std::vector<int> face_neighbors;
      if (slot.axis == 1) {
        if (j > 0) {
          face_neighbors.push_back(i * topo.nz + j - 1);
        }
        if (j + 1 < topo.nz) {
          face_neighbors.push_back(i * topo.nz + j + 1);
        }
      } else {
        if (i > 0) {
          face_neighbors.push_back((i - 1) * topo.nz + j);
        }
        if (i + 1 < topo.nr) {
          face_neighbors.push_back((i + 1) * topo.nz + j);
        }
      }
      double pressure_scale = 1.0e-30;
      for (const int neighbor : face_neighbors) {
        const std::size_t cell = static_cast<std::size_t>(neighbor);
        pressure_scale =
            std::max(pressure_scale, std::abs(Pe[cell]) + std::abs(Pi[cell]));
      }
      const double area = revolved_face_area(
          slot.node_a[0], slot.node_a[1], node_r, node_z);
      const double threshold = config.release_force_c * area * pressure_scale;
      double lambda_min = std::numeric_limits<double>::infinity();
      const double corner_j_min = evac_cell_min_corner_jacobian(
          slot.cell, topo.nz, node_r, node_z);
      for (int pair = 0; pair < 2; ++pair) {
        if (!engaged_at_entry[pair]) {
          continue;
        }
        const int pair_offset = active_pair_offset[s][pair];
        const int lambda_index = pair_base + pair_offset;
        TENRYU_ASSERT(pair_offset >= 0 && lambda_index < active_pair_count,
                      "evacuated-cell contact release pair index mismatch");
        const double lambda =
            pair_lambda[static_cast<std::size_t>(lambda_index)];
        lambda_min = std::min(lambda_min, lambda);
        const bool geometric_reopen =
            gap_pair[pair] > slot.g_arm && corner_j_min > 0.0;
        if (evacuated_cell_contact_update_pair_release(
                slot,
                pair,
                lambda,
                threshold,
                config.release_persistence_stages,
                geometric_reopen)) {
          active_pair_set_changed = true;
          std::ostringstream line;
          line << "[evac-cell-contact] released cell=" << slot.cell
               << " pair=" << pair << " lambda=" << std::scientific
               << std::setprecision(17) << lambda
               << " threshold=" << threshold;
          if (geometric_reopen) {
            line << " reason=geometric_reopen";
          }
          core::log_info(line.str());
        }
      }
      TENRYU_ASSERT(std::isfinite(lambda_min),
                    "evacuated-cell contact release lambda missing");
      slot.lambda_last = lambda_min;
    }

    if (state_at_entry == core::EvacContactState::kOpenSpacer) {
      bool minimum_pair_closing = false;
      for (int pair = 0; pair < 2; ++pair) {
        minimum_pair_closing =
            minimum_pair_closing ||
            (gap_pair[pair] == gap &&
             std::isfinite(slot.gap_prev_pair[pair]) &&
             gap_pair[pair] < slot.gap_prev_pair[pair]);
      }
      if (gap < slot.g0 * (1.0 + config.reengage_gap_margin) &&
          minimum_pair_closing) {
        slot.state = core::EvacContactState::kArmed;
        slot.tensile_streak_pair[0] = 0;
        slot.tensile_streak_pair[1] = 0;
      }
    }

    for (int pair = 0; pair < 2; ++pair) {
      slot.gap_prev_pair[pair] = gap_pair[pair];
      slot.gap_pair[pair] = gap_pair[pair];
    }
    slot.gap_prev = gap;
    slot.gap = gap;
  }

  const double commit_project_scale =
      std::max({std::abs(commit_project_dk_sum),
                std::abs(commit_project_deposited_sum),
                std::numeric_limits<double>::min()});
  TENRYU_ASSERT(
      std::abs(commit_project_dk_sum - commit_project_deposited_sum) <=
          64.0 * std::numeric_limits<double>::epsilon() *
              commit_project_scale,
      "evacuated-cell contact projection heat audit failed");
  if (active_pair_count > 0) {
    std::fill(pair_dk.begin(), pair_dk.end(), 0.0);
    evacuated.d_contact_pair_dk.copy_from_host(pair_dk);
  }

  if (volume_projection_log_cadence) {
    if (config.flank_tangential_strip.enabled && active_slot_present) {
      (void)flank_strip_update_and_evaluate(
          state, cfg, state.dt_prev_hydro);
    }
    if (volume_projection_count > 0 || patch_projection_count > 0) {
      std::ostringstream line;
      line << "[evac-cell-contact] volume-hold step=" << state.step
           << " projections=" << volume_projection_count
           << " patch=" << patch_projection_count;
      core::log_info(line.str());
    }
    if (evacuated.d_contact_volume_projection_count.size() == 2U) {
      const std::vector<int> zero_count(2U, 0);
      evacuated.d_contact_volume_projection_count.copy_from_host(zero_count);
    }
    for (core::EvacContactSlot& slot : state.contact_graph.records) {
      if (slot.devolumized != 0U) {
        const std::size_t a0 = static_cast<std::size_t>(slot.node_a[0]);
        const std::size_t a1 = static_cast<std::size_t>(slot.node_a[1]);
        const std::size_t b0 = static_cast<std::size_t>(slot.node_b[0]);
        const std::size_t b1 = static_cast<std::size_t>(slot.node_b[1]);
        const double coordinate_magnitude =
            std::max({std::abs(node_r[a0]), std::abs(node_z[a0]),
                      std::abs(node_r[a1]), std::abs(node_z[a1]),
                      std::abs(node_r[b0]), std::abs(node_z[b0]),
                      std::abs(node_r[b1]), std::abs(node_z[b1])});
        const double length_floor =
            4096.0 * DBL_EPSILON * coordinate_magnitude;
        const double face_a_length =
            std::hypot(node_r[a0] - node_r[a1],
                       node_z[a0] - node_z[a1]);
        const double face_b_length =
            std::hypot(node_r[b0] - node_r[b1],
                       node_z[b0] - node_z[b1]);
        TENRYU_ASSERT(face_a_length > length_floor,
                      "evacuated-cell seam guard face-a length failed");
        TENRYU_ASSERT(face_b_length > length_floor,
                      "evacuated-cell seam guard face-b length failed");
        TENRYU_ASSERT(slot.gap_pair[1] >= -1.0e-30,
                      "evacuated-cell seam guard open-end gap failed");
      }
      const int count_delta =
          slot.reproject_count - slot.reproject_count_last_logged;
      if (slot.state != core::EvacContactState::kActive ||
          count_delta <= 0) {
        continue;
      }
      const double heat_delta =
          slot.reproject_heat_total - slot.reproject_heat_last_logged;
      const double A_a = revolved_face_area(
          slot.node_a[0], slot.node_a[1], node_r, node_z);
      const double A_b = revolved_face_area(
          slot.node_b[0], slot.node_b[1], node_r, node_z);
      const double h_rz =
          vol[slot.cell] /
          std::max(0.5 * (A_a + A_b),
                   std::numeric_limits<double>::min());
      std::ostringstream line;
      line << "[evac-cell-contact] hold cell=" << slot.cell
           << " reprojects=" << count_delta << " heat=" << std::scientific
           << std::setprecision(17) << heat_delta
           << " gap0=" << slot.gap_pair[0]
           << " gap1=" << slot.gap_pair[1]
           << " lambda=" << slot.lambda_last
           << " vol=" << vol[slot.cell]
           << " vol_floor="
           << kEvacContactVolumeFloorFraction * slot.vol_at_engagement
           << " chi=" << evacuated_cell_contact_patch_fraction(
                              slot.gap_pair[0], slot.gap_pair[1], slot.g0)
           << " h_rz=" << h_rz;
      core::log_info(line.str());
      slot.reproject_count_last_logged = slot.reproject_count;
      slot.reproject_heat_last_logged = slot.reproject_heat_total;
    }
  }

  if (!state.cavity_boundary.boundary_nodes.empty() &&
      (engagement_possible || active_slot_present)) {
    constexpr std::size_t kShadowContactRowCapacity = 64U;
    constexpr double gap_floor = 0.0;
    const auto& boundary = state.cavity_boundary;
    const std::vector<std::int8_t>* const hydro_active =
        state.hydro_active.empty() ? nullptr : &state.hydro_active;
    double row_dk_feedback_total = 0.0;
    if (config.lcp_apply_enabled && evacuated.n_contact_rows > 0) {
      const std::size_t previous_row_count =
          static_cast<std::size_t>(evacuated.n_contact_rows);
      TENRYU_ASSERT(
          evacuated.d_contact_row_dk.size() == previous_row_count &&
              evacuated.contact_row_nodes.size() == 3U * previous_row_count,
          "shadow contact row dK feedback staging mismatch");
      std::vector<double> row_dk_feedback;
      evacuated.d_contact_row_dk.copy_to_host(row_dk_feedback);
      ensure_contact_fields_loaded();
      for (std::size_t alpha = 0; alpha < previous_row_count; ++alpha) {
        const double row_dk = row_dk_feedback[alpha];
        if (!(row_dk > 0.0)) {
          continue;
        }
        deposit_contact_side_heat(
            evacuated.contact_row_nodes[3U * alpha],
            row_dk,
            topo.nr,
            topo.nz,
            corner_mass,
            evacuated.inactive_member_mask,
            hydro_active,
            mass,
            ei,
            row_dk_feedback_total,
            nullptr);
      }
      if (row_dk_feedback_total > 0.0) {
        contact_fields_dirty = true;
      }
    }
    double max_g_arm = gap_floor;
    for (const core::EvacContactSlot& record :
         state.contact_graph.records) {
      max_g_arm = std::max(max_g_arm, record.g_arm);
    }

    std::vector<ShadowContactRow> rows;
    rows.reserve(std::min(boundary.boundary_nodes.size(),
                          kShadowContactRowCapacity));
    bool row_cap_logged = false;
    for (std::size_t bn = 0; bn < boundary.boundary_nodes.size(); ++bn) {
      if (bn >= boundary.shadow_best_seg.size() ||
          bn >= boundary.shadow_best_gap.size() ||
          bn >= boundary.shadow_best_xi.size()) {
        continue;
      }
      const std::int32_t segment = boundary.shadow_best_seg[bn];
      if (segment < 0) {
        continue;
      }
      const std::size_t segment_index =
          static_cast<std::size_t>(segment);
      if (segment_index >= boundary.segment_ids.size() ||
          segment_index >= boundary.seg_node_a.size() ||
          segment_index >= boundary.seg_node_b.size() ||
          segment_index >= boundary.seg_normal_r.size() ||
          segment_index >= boundary.seg_normal_z.size()) {
        continue;
      }
      const double gap = boundary.shadow_best_gap[bn];
      if (gap > 4.0 * max_g_arm) {
        continue;
      }
      if (rows.size() == kShadowContactRowCapacity) {
        if (!row_cap_logged) {
          core::log_warning(
              "[cavity-lcp] row cap step=" + std::to_string(state.step) +
              " cap=" + std::to_string(kShadowContactRowCapacity));
          row_cap_logged = true;
        }
        continue;
      }

      const int slave_node = boundary.boundary_nodes[bn];
      bool legacy_engaged = false;
      for (const core::EvacContactSlot& record :
           state.contact_graph.records) {
        for (int pair = 0; pair < 2; ++pair) {
          legacy_engaged = legacy_engaged ||
                           (record.node_b[pair] == slave_node &&
                            record.pair_engaged[pair] != 0U);
        }
      }

      ShadowContactRow row;
      row.slave_node = slave_node;
      row.segment_id = boundary.segment_ids[segment_index];
      row.constraint_id =
          (static_cast<std::int64_t>(slave_node) << 32) |
          static_cast<std::uint32_t>(row.segment_id);
      row.node_s = slave_node;
      row.node_a = boundary.seg_node_a[segment_index];
      row.node_b = boundary.seg_node_b[segment_index];
      row.xi = boundary.shadow_best_xi[bn];
      row.normal_r = boundary.seg_normal_r[segment_index];
      row.normal_z = boundary.seg_normal_z[segment_index];
      row.gap = gap;
      row.legacy_engaged = legacy_engaged;
      rows.push_back(row);
    }
    TENRYU_ASSERT(
        std::is_sorted(rows.begin(), rows.end(),
                       [](const ShadowContactRow& lhs,
                          const ShadowContactRow& rhs) {
                         return lhs.constraint_id < rhs.constraint_id;
                       }),
        "shadow contact row assembly is not in canonical order");

    std::vector<std::array<int, 3>> row_mesh_nodes;
    row_mesh_nodes.reserve(rows.size());
    std::vector<int> mesh_nodes;
    mesh_nodes.reserve(3U * rows.size());
    for (const ShadowContactRow& row : rows) {
      row_mesh_nodes.push_back({row.node_s, row.node_a, row.node_b});
      mesh_nodes.push_back(row.node_s);
      mesh_nodes.push_back(row.node_a);
      mesh_nodes.push_back(row.node_b);
    }
    std::sort(mesh_nodes.begin(), mesh_nodes.end());
    mesh_nodes.erase(std::unique(mesh_nodes.begin(), mesh_nodes.end()),
                     mesh_nodes.end());

    if (!rows.empty() && corner_mass.empty()) {
      state.corner_mass.copy_to_host(corner_mass);
    }
    std::vector<double> node_mass;
    std::vector<double> shadow_v_r;
    std::vector<double> shadow_v_z;
    node_mass.reserve(mesh_nodes.size());
    shadow_v_r.reserve(mesh_nodes.size());
    shadow_v_z.reserve(mesh_nodes.size());
    for (const int mesh_node : mesh_nodes) {
      node_mass.push_back(evacuated_cell_active_node_mass(
          mesh_node,
          topo.nr,
          topo.nz,
          corner_mass,
          evacuated.inactive_member_mask,
          hydro_active));
      shadow_v_r.push_back(v_r[static_cast<std::size_t>(mesh_node)]);
      shadow_v_z.push_back(v_z[static_cast<std::size_t>(mesh_node)]);
    }
    for (ShadowContactRow& row : rows) {
      const auto compact_node = [&](const int mesh_node) {
        const auto found =
            std::lower_bound(mesh_nodes.begin(), mesh_nodes.end(), mesh_node);
        TENRYU_ASSERT(found != mesh_nodes.end() && *found == mesh_node,
                      "shadow contact compact node lookup failed");
        return static_cast<int>(found - mesh_nodes.begin());
      };
      row.node_s = compact_node(row.node_s);
      row.node_a = compact_node(row.node_a);
      row.node_b = compact_node(row.node_b);
    }

    ShadowContactSolveResult result = solve_shadow_contact_lcp(
        rows, node_mass, shadow_v_r, shadow_v_z, state.dt, gap_floor);
    ShadowContactSolveResult dynamic_result;
    double lcp_dk_applied = 0.0;
    bool any_lcp_impulse_applied = false;
    const bool apply_lcp_result =
        config.lcp_apply_enabled && !result.singular;
    const int previous_contact_row_count = evacuated.n_contact_rows;
    if (apply_lcp_result) {
      dynamic_result = solve_shadow_contact_lcp(
          rows,
          node_mass,
          shadow_v_r,
          shadow_v_z,
          state.dt,
          gap_floor,
          GapTermMode::kNonNegative);
      ensure_contact_fields_loaded();
      for (std::size_t alpha = 0; alpha < rows.size(); ++alpha) {
        const double lambda = dynamic_result.lambda[alpha];
        if (!(lambda > 0.0)) {
          continue;
        }
        any_lcp_impulse_applied = true;
        const ShadowContactRow& row = rows[alpha];
        const std::array<int, 3>& mesh_row = row_mesh_nodes[alpha];
        const std::size_t compact_s = static_cast<std::size_t>(row.node_s);
        const std::size_t compact_a = static_cast<std::size_t>(row.node_a);
        const std::size_t compact_b = static_cast<std::size_t>(row.node_b);
        const double mass_s = node_mass[compact_s];
        const double mass_a = node_mass[compact_a];
        const double mass_b = node_mass[compact_b];
        TENRYU_ASSERT(mass_s > 0.0 && mass_a > 0.0 && mass_b > 0.0,
                      "shadow contact impulse has nonpositive node mass");
        const std::size_t mesh_s = static_cast<std::size_t>(mesh_row[0]);
        const std::size_t mesh_a = static_cast<std::size_t>(mesh_row[1]);
        const std::size_t mesh_b = static_cast<std::size_t>(mesh_row[2]);
        const double kinetic_before =
            0.5 * mass_s *
                (v_r[mesh_s] * v_r[mesh_s] +
                 v_z[mesh_s] * v_z[mesh_s]) +
            0.5 * mass_a *
                (v_r[mesh_a] * v_r[mesh_a] +
                 v_z[mesh_a] * v_z[mesh_a]) +
            0.5 * mass_b *
                (v_r[mesh_b] * v_r[mesh_b] +
                 v_z[mesh_b] * v_z[mesh_b]);

        v_r[mesh_s] += lambda / mass_s * row.normal_r;
        v_z[mesh_s] += lambda / mass_s * row.normal_z;
        v_r[mesh_a] -=
            (1.0 - row.xi) * lambda / mass_a * row.normal_r;
        v_z[mesh_a] -=
            (1.0 - row.xi) * lambda / mass_a * row.normal_z;
        v_r[mesh_b] -= row.xi * lambda / mass_b * row.normal_r;
        v_z[mesh_b] -= row.xi * lambda / mass_b * row.normal_z;

        const double kinetic_after =
            0.5 * mass_s *
                (v_r[mesh_s] * v_r[mesh_s] +
                 v_z[mesh_s] * v_z[mesh_s]) +
            0.5 * mass_a *
                (v_r[mesh_a] * v_r[mesh_a] +
                 v_z[mesh_a] * v_z[mesh_a]) +
            0.5 * mass_b *
                (v_r[mesh_b] * v_r[mesh_b] +
                 v_z[mesh_b] * v_z[mesh_b]);
        const double row_dk = kinetic_before - kinetic_after;
        TENRYU_ASSERT(
            row_dk >= -64.0 * std::numeric_limits<double>::epsilon() *
                          std::max(kinetic_before, 1.0),
            "contact impulse must not inject kinetic energy");
        lcp_dk_applied += row_dk;
        if (row_dk > 0.0) {
          double deposited_heat = 0.0;
          deposit_contact_side_heat(mesh_row[0],
                                    row_dk,
                                    topo.nr,
                                    topo.nz,
                                    corner_mass,
                                    evacuated.inactive_member_mask,
                                    hydro_active,
                                    mass,
                                    ei,
                                    deposited_heat,
                                    nullptr);
        }
      }
      if (any_lcp_impulse_applied) {
        contact_fields_dirty = true;
      }

      int active_row_count = 0;
      for (const std::uint8_t active : dynamic_result.active) {
        active_row_count += active != 0U ? 1 : 0;
      }
      if (active_row_count > 0) {
        evacuated.contact_row_nodes.clear();
        evacuated.contact_row_xi.clear();
        evacuated.contact_row_normal.clear();
        evacuated.contact_row_nodes.reserve(
            3U * static_cast<std::size_t>(active_row_count));
        evacuated.contact_row_xi.reserve(
            static_cast<std::size_t>(active_row_count));
        evacuated.contact_row_normal.reserve(
            2U * static_cast<std::size_t>(active_row_count));
        for (std::size_t alpha = 0; alpha < rows.size(); ++alpha) {
          if (dynamic_result.active[alpha] == 0U) {
            continue;
          }
          const std::array<int, 3>& mesh_row = row_mesh_nodes[alpha];
          evacuated.contact_row_nodes.push_back(mesh_row[0]);
          evacuated.contact_row_nodes.push_back(mesh_row[1]);
          evacuated.contact_row_nodes.push_back(mesh_row[2]);
          evacuated.contact_row_xi.push_back(rows[alpha].xi);
          evacuated.contact_row_normal.push_back(rows[alpha].normal_r);
          evacuated.contact_row_normal.push_back(rows[alpha].normal_z);
        }
        evacuated.n_contact_rows = active_row_count;
        evacuated.d_contact_row_nodes.reset(
            evacuated.contact_row_nodes.size());
        evacuated.d_contact_row_nodes.copy_from_host(
            evacuated.contact_row_nodes);
        evacuated.d_contact_row_xi.reset(evacuated.contact_row_xi.size());
        evacuated.d_contact_row_xi.copy_from_host(
            evacuated.contact_row_xi);
        evacuated.d_contact_row_normal.reset(
            evacuated.contact_row_normal.size());
        evacuated.d_contact_row_normal.copy_from_host(
            evacuated.contact_row_normal);
        const std::vector<double> zero_row_dk(
            static_cast<std::size_t>(active_row_count), 0.0);
        evacuated.d_contact_row_dk.reset(zero_row_dk.size());
        evacuated.d_contact_row_dk.copy_from_host(zero_row_dk);
      } else {
        evacuated.n_contact_rows = 0;
      }
    } else {
      evacuated.n_contact_rows = 0;
    }
    if (evacuated.n_contact_rows != previous_contact_row_count) {
      const double lambda_max_dyn =
          dynamic_result.lambda.empty()
              ? 0.0
              : *std::max_element(dynamic_result.lambda.begin(),
                                  dynamic_result.lambda.end());
      std::ostringstream line;
      line << "[cavity-lcp] rows_staged step=" << state.step
           << " n=" << evacuated.n_contact_rows
           << " prev=" << previous_contact_row_count
           << " lambda_max_dyn=" << std::scientific
           << std::setprecision(17) << lambda_max_dyn;
      core::log_info(line.str());
    }

    static thread_local const core::State* shadow_lcp_log_state = nullptr;
    static thread_local std::uint64_t shadow_lcp_log_epoch =
        std::numeric_limits<std::uint64_t>::max();
    const bool topology_epoch_changed =
        shadow_lcp_log_state != &state ||
        shadow_lcp_log_epoch != boundary.topology_epoch;
    shadow_lcp_log_state = &state;
    shadow_lcp_log_epoch = boundary.topology_epoch;
    if (topology_epoch_changed || state.step % 5000 == 0) {
      const int active_count = static_cast<int>(
          std::count(result.active.begin(), result.active.end(), 1U));
      const double lambda_max = result.lambda.empty()
                                    ? 0.0
                                    : *std::max_element(result.lambda.begin(),
                                                        result.lambda.end());
      const double lambda_max_dyn =
          dynamic_result.lambda.empty()
              ? 0.0
              : *std::max_element(dynamic_result.lambda.begin(),
                                  dynamic_result.lambda.end());
      std::ostringstream line;
      line << "[cavity-lcp] step=" << state.step
           << " rows=" << rows.size() << " active=" << active_count
           << " iter=" << result.iterations
           << " released=" << result.released_rows
           << " lambda_max=" << std::scientific << std::setprecision(17)
           << lambda_max << " lambda_max_dyn=" << lambda_max_dyn
           << " singular=" << (result.singular ? 1 : 0)
           << " applied=" << (apply_lcp_result ? 1 : 0)
           << " dK=" << (lcp_dk_applied + row_dk_feedback_total)
           << " dk_rows=" << row_dk_feedback_total;
      core::log_info(line.str());
    }

    int disagreement_count = 0;
    for (std::size_t alpha = 0;
         alpha < rows.size() && disagreement_count < 4; ++alpha) {
      const ShadowContactRow& row = rows[alpha];
      if (row.legacy_engaged && result.active[alpha] == 0U) {
        std::ostringstream line;
        line << "[cavity-lcp] disagree step=" << state.step
             << " node=" << row.slave_node << " seg=" << row.segment_id
             << " legacy=held lcp=released lambda_trial="
             << std::scientific << std::setprecision(17)
             << result.lambda[alpha];
        core::log_info(line.str());
        ++disagreement_count;
      } else if (!row.legacy_engaged && result.active[alpha] != 0U &&
                 result.lambda[alpha] > 0.0) {
        std::ostringstream line;
        line << "[cavity-lcp] disagree step=" << state.step
             << " node=" << row.slave_node << " seg=" << row.segment_id
             << " legacy=free lcp=compressive lambda=" << std::scientific
             << std::setprecision(17) << result.lambda[alpha];
        core::log_info(line.str());
        ++disagreement_count;
      }
    }
  } else {
    evacuated.n_contact_rows = 0;
  }

  if (contact_fields_dirty) {
    state.v_r.copy_from_host(v_r);
    state.v_z.copy_from_host(v_z);
    state.ei.copy_from_host(ei);
  }

  if (active_pair_set_changed) {
    evacuated_cell_upload_contact_device_state(state);
  } else {
    const EvacContactPairBuffers pair_buffers =
        evacuated_cell_rebuild_contact_pairs(state.contact_graph.records);
    TENRYU_ASSERT(pair_buffers.nodes.size() ==
                      2U * static_cast<std::size_t>(evacuated.n_active_pairs),
                  "evacuated-cell contact pair row coverage count mismatch");
    evacuated_cell_upload_contact_pair_row_coverage(evacuated, pair_buffers);
  }
}

}  // namespace tenryu::hydro
