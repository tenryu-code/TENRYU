#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::hydro {

#if defined(__CUDACC__)
#define TENRYU_EVAC_CONTACT_HOST_DEVICE __host__ __device__
#else
#define TENRYU_EVAC_CONTACT_HOST_DEVICE
#endif

// Contact dt-cap lower bound as a fraction of the current dt: the cap may
// throttle dt by at most this factor per step. Approach resolution stays
// geometric while a collapse to dt.min_s (which a growth_factor<=1 config
// can never recover from) is structurally impossible.
inline constexpr double kEvacContactDtCapThrottle = 0.1;

// Engaged-seam volume floor as a fraction of the engagement-time volume:
// below it the corner velocities are projected to dV/dt >= 0 (the
// tangential/polygon counterpart of the pair-normal gap floor --
// consultation #24: the contact must preserve V > V_geom_min > 0).
inline constexpr double kEvacContactVolumeFloorFraction = 0.25;
inline constexpr double kEvacContactVolumeNullspaceEps = 1.0e-12;

struct EvacCellShadowParams {
  double arm_mass_fraction;
  double off_mass_fraction;
  double rho_vacuum_policy;  // g/cc
};

struct EvacCellShadowCell {
  int cell = -1;
  double mass_fraction = 0.0;
  double rho = 0.0;
  double ne_over_ncrit = 0.0;
  double Te = 0.0;
};

struct EvacCellShadowSummary {
  int n_active = 0;
  int n_armed = 0;
  int n_off_eligible = 0;
  // Up to 4 smallest-mass-fraction cells, ascending.
  std::vector<EvacCellShadowCell> worst;
};

// Pure host classification: m_off_c = max(off_mass_fraction * m_ref[c],
// rho_vacuum_policy * v_ref[c]); m_arm_c = max(arm_mass_fraction * m_ref[c],
// 100.0 * m_off_c). OFF-eligible: m <= m_off_c; ARMED: m <= m_arm_c; else
// ACTIVE.
EvacCellShadowSummary classify_evacuated_cells(
    const std::vector<double>& mass,
    const std::vector<double>& mass_ref,
    const std::vector<double>& vol_ref,
    const std::vector<double>& rho,
    const std::vector<double>& ne_over_ncrit,
    const std::vector<double>& Te,
    const EvacCellShadowParams& params,
    const std::vector<std::uint8_t>* inactive_member_mask = nullptr);

bool update_evacuated_cell_off_streak(bool off_eligible,
                                      int off_hold_evaluations,
                                      int& off_streak);

struct EvacCellFaceWeights {
  std::vector<int> recipients;
  std::vector<double> weights;
};

EvacCellFaceWeights evacuated_cell_face_weights(
    int nr,
    int nz,
    int donor_cell,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const std::vector<std::uint8_t>& inactive_member_mask);

bool evacuated_cell_is_domain_corner(int nr, int nz, int cell);

// One flattened corner-mass target index per donor corner, in donor-corner
// order. Domain-corner donors and unresolved inputs return an empty vector.
std::vector<int> evacuated_cell_corner_mass_targets(
    int nr,
    int nz,
    int donor_cell,
    const std::vector<int>& recipients,
    const std::vector<double>& weights);

void transfer_evacuated_cell_extensives(
    int donor_cell,
    const std::vector<int>& recipients,
    const std::vector<double>& weights,
    std::vector<double>& mass,
    std::vector<double>& ee,
    std::vector<double>& ei);

struct EvacCellRematerializeTransfer {
  bool accepted = false;
  double mass = 0.0;
  double fill_fraction = 0.0;
  double electron_energy = 0.0;
  double ion_energy = 0.0;
  std::vector<double> donor_mass_withdrawals;
};

struct EvacContactGeometry {
  int node_a[2] = {-1, -1};
  int node_b[2] = {-1, -1};
  double normal_pair_r[2] = {0.0, 0.0};
  double normal_pair_z[2] = {1.0, 1.0};
  double gap_pair[2] = {0.0, 0.0};
  double gap = 0.0;
};

EvacContactGeometry evacuated_cell_contact_geometry(
    int cell,
    int axis,
    int nz,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z);

struct EvacContactAxisSelection {
  int axis = 1;
  double h_perp_ref = 0.0;
};

EvacContactAxisSelection evacuated_cell_closing_axis(
    int cell,
    int nz,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const std::vector<double>& x_r_ref,
    const std::vector<double>& x_z_ref);

struct EvacContactImpact {
  double vA_n = 0.0;
  double vB_n = 0.0;
  double J = 0.0;
  double dK = 0.0;
  double dK_A = 0.0;
  double dK_B = 0.0;
};

struct EvacContactActiveCells {
  std::vector<std::uint8_t> mask;
  std::vector<int> cells;
  std::vector<int> axis;
  std::vector<int> face_nodes;
  std::vector<double> face_normal_ref;
  std::vector<double> face_g0;
  std::vector<double> mortar_g_hold;
  std::vector<std::uint8_t> mortar_g_hold_valid;
  std::vector<double> vol_at_engagement;
  std::vector<std::uint8_t> devolumized;
  std::vector<int> pair_dk_owner;
};

struct EvacContactPairBuffers {
  std::vector<int> nodes;
  std::vector<double> normals;
};

EvacContactActiveCells evacuated_cell_rebuild_contact_active(
    const std::vector<core::EvacContactSlot>& slots,
    std::size_t n_cells);

EvacContactPairBuffers evacuated_cell_rebuild_contact_pairs(
    const std::vector<core::EvacContactSlot>& slots);

void evacuated_cell_upload_contact_device_state(core::State& state);

bool evacuated_cell_seam_devolumize(core::State& state,
                                    const core::Config& cfg,
                                    core::EvacContactSlot& slot,
                                    const std::vector<double>& mass,
                                    const std::vector<double>& vol);

bool evacuated_cell_contact_should_arm(const double gap_pair[2],
                                       const double gap_prev_pair[2],
                                       double g_arm);

EvacContactImpact evacuated_cell_contact_impact(double mA,
                                                double mB,
                                                double vA_n,
                                                double vB_n);

bool evacuated_cell_contact_engage_pair(
    core::EvacContactSlot& slot,
    int pair,
    double gap,
    double mA,
    double mB,
    double cell_mass,
    double cell_volume,
    std::vector<double>& v_r,
    std::vector<double>& v_z,
    EvacContactImpact& impact,
    const bool force_crossed = false,
    const bool skip_velocity_projection = false);

bool evacuated_cell_contact_update_pair_release(core::EvacContactSlot& slot,
                                                int pair,
                                                double lambda,
                                                double threshold,
                                                int persistence,
                                                bool geometric_reopen = false);

inline double evacuated_cell_contact_gap_invariant_floor(
    const core::EvacContactSlot& slot, const int pair) {
  const double g_eng = slot.gap_at_engagement_pair[pair];
  const double anchor =
      (g_eng > 0.0) ? std::min(slot.g0, g_eng) : slot.g0;
  return 0.5 * anchor;
}

TENRYU_EVAC_CONTACT_HOST_DEVICE inline void
evacuated_cell_contact_common_frame(
    const double a0r,
    const double a0z,
    const double a1r,
    const double a1z,
    const double b0r,
    const double b0z,
    const double b1r,
    const double b1z,
    const double ref_nr,
    const double ref_nz,
    double t[2],
    double n[2]) {
  const double tangent_r = 0.5 * ((a1r + b1r) - (a0r + b0r));
  const double tangent_z = 0.5 * ((a1z + b1z) - (a0z + b0z));
  const double tangent_norm =
      std::sqrt(tangent_r * tangent_r + tangent_z * tangent_z);
  if (!(tangent_norm > 0.0)) {
    const double ref_norm = std::sqrt(ref_nr * ref_nr + ref_nz * ref_nz);
    n[0] = ref_nr / ref_norm;
    n[1] = ref_nz / ref_norm;
    t[0] = -n[1];
    t[1] = n[0];
    return;
  }

  t[0] = tangent_r / tangent_norm;
  t[1] = tangent_z / tangent_norm;
  n[0] = t[1];
  n[1] = -t[0];
  // A separation-vector sign fix is ill-defined after interpenetration;
  // the slot's topology-anchored frozen pair0 normal fixes the orientation.
  if (n[0] * ref_nr + n[1] * ref_nz < 0.0) {
    t[0] = -t[0];
    t[1] = -t[1];
    n[0] = -n[0];
    n[1] = -n[1];
  }
}

TENRYU_EVAC_CONTACT_HOST_DEVICE inline double
evacuated_cell_contact_patch_fraction(
    const double gap0, const double gap1, const double g0) {
  if (gap0 == gap1) {
    return gap0 <= g0 ? 1.0 : 0.0;
  }
  if (gap0 <= g0 && gap1 <= g0) {
    return 1.0;
  }
  if (gap0 >= g0 && gap1 >= g0) {
    return 0.0;
  }
  const double crossing = (g0 - gap0) / (gap1 - gap0);
  return crossing < 0.0 ? 0.0 : (crossing > 1.0 ? 1.0 : crossing);
}

inline constexpr double kEvacContactPi =
    3.1415926535897932384626433832795028841971693993751058209749;
inline constexpr double kEvacContactGaussAbscissa =
    0.7745966692414833770358530799564799221665843410583181653175;

TENRYU_EVAC_CONTACT_HOST_DEVICE inline void
evacuated_cell_contact_patch_constraint(
    const double r[4],
    const double z[4],
    const double n[2],
    const double chi,
    double* const w_a0,
    double* const w_a1,
    double* const w_b0,
    double* const w_b1) {
  constexpr double gauss_x[3] = {
      0.5 * (1.0 - kEvacContactGaussAbscissa),
      0.5,
      0.5 * (1.0 + kEvacContactGaussAbscissa)};
  constexpr double gauss_w[3] = {
      5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0};
  *w_a0 = 0.0;
  *w_a1 = 0.0;
  *w_b0 = 0.0;
  *w_b1 = 0.0;
  const double patch_fraction =
      chi < 0.0 ? 0.0 : (chi > 1.0 ? 1.0 : chi);
  if (!(patch_fraction > 0.0)) {
    return;
  }
  (void)n;
  const double mid0_r = 0.5 * (r[0] + r[2]);
  const double mid0_z = 0.5 * (z[0] + z[2]);
  const double mid1_r = 0.5 * (r[1] + r[3]);
  const double mid1_z = 0.5 * (z[1] + z[3]);
  const double delta_r = mid1_r - mid0_r;
  const double delta_z = mid1_z - mid0_z;
  const double face_length =
      std::sqrt(delta_r * delta_r + delta_z * delta_z);
  for (int q = 0; q < 3; ++q) {
    const double xi = patch_fraction * gauss_x[q];
    const double radius = mid0_r + xi * delta_r;
    const double area =
        2.0 * kEvacContactPi * radius * face_length *
        gauss_w[q] * patch_fraction;
    const double N0 = 1.0 - xi;
    const double N1 = xi;
    *w_b0 += area * N0;
    *w_b1 += area * N1;
    *w_a0 -= area * N0;
    *w_a1 -= area * N1;
  }
}

TENRYU_EVAC_CONTACT_HOST_DEVICE inline bool
evacuated_cell_contact_patch_project(
    const double mass[4],
    const int face_nodes[4],
    const double n[2],
    const double weights[4],
    const int* engaged_pair_nodes,
    const double* engaged_pair_normal,
    const int engaged_pair_count,
    double v_r[4],
    double v_z[4],
    double* const kinetic_energy_loss) {
  if (kinetic_energy_loss != nullptr) {
    *kinetic_energy_loss = 0.0;
  }
  if (engaged_pair_count < 0 || engaged_pair_count > 2 ||
      (engaged_pair_count > 0 &&
       (face_nodes == nullptr || engaged_pair_nodes == nullptr ||
        engaged_pair_normal == nullptr))) {
    return false;
  }
  double constraint_value = 0.0;
  double row_r[4] = {};
  double row_z[4] = {};
  for (int k = 0; k < 4; ++k) {
    if (!(mass[k] > 0.0)) {
      return false;
    }
    row_r[k] = weights[k] * n[0];
    row_z[k] = weights[k] * n[1];
    constraint_value += row_r[k] * v_r[k] + row_z[k] * v_z[k];
  }
  if (!(constraint_value < 0.0)) {
    return false;
  }

  double basis_r[2][4] = {};
  double basis_z[2][4] = {};
  for (int pair = 0; pair < engaged_pair_count; ++pair) {
    int local_a = -1;
    int local_b = -1;
    for (int k = 0; k < 4; ++k) {
      if (face_nodes[k] == engaged_pair_nodes[2 * pair]) {
        local_a = k;
      }
      if (face_nodes[k] == engaged_pair_nodes[2 * pair + 1]) {
        local_b = k;
      }
    }
    if (local_a < 0 || local_b < 0) {
      return false;
    }
    const double normal_r = engaged_pair_normal[2 * pair];
    const double normal_z = engaged_pair_normal[2 * pair + 1];
    basis_r[pair][local_a] -= normal_r;
    basis_z[pair][local_a] -= normal_z;
    basis_r[pair][local_b] += normal_r;
    basis_z[pair][local_b] += normal_z;
    // Match the C20 modified Gram-Schmidt ordering before removing the
    // engaged-pair row space from the patch row.
    for (int previous = 0; previous < pair; ++previous) {
      double basis_denominator = 0.0;
      double basis_inner = 0.0;
      for (int k = 0; k < 4; ++k) {
        basis_denominator +=
            (basis_r[previous][k] * basis_r[previous][k] +
             basis_z[previous][k] * basis_z[previous][k]) /
            mass[k];
        basis_inner +=
            (basis_r[previous][k] * basis_r[pair][k] +
             basis_z[previous][k] * basis_z[pair][k]) /
            mass[k];
      }
      if (basis_denominator > 0.0) {
        const double coefficient = basis_inner / basis_denominator;
        for (int k = 0; k < 4; ++k) {
          basis_r[pair][k] -= coefficient * basis_r[previous][k];
          basis_z[pair][k] -= coefficient * basis_z[previous][k];
        }
      }
    }
  }

  double projected_row_r[4] = {row_r[0], row_r[1], row_r[2], row_r[3]};
  double projected_row_z[4] = {row_z[0], row_z[1], row_z[2], row_z[3]};
  for (int pair = 0; pair < engaged_pair_count; ++pair) {
    double basis_denominator = 0.0;
    double basis_row_inner = 0.0;
    for (int k = 0; k < 4; ++k) {
      basis_denominator +=
          (basis_r[pair][k] * basis_r[pair][k] +
           basis_z[pair][k] * basis_z[pair][k]) /
          mass[k];
      basis_row_inner +=
          (basis_r[pair][k] * projected_row_r[k] +
           basis_z[pair][k] * projected_row_z[k]) /
          mass[k];
    }
    if (basis_denominator > 0.0) {
      const double coefficient = basis_row_inner / basis_denominator;
      for (int k = 0; k < 4; ++k) {
        projected_row_r[k] -= coefficient * basis_r[pair][k];
        projected_row_z[k] -= coefficient * basis_z[pair][k];
      }
    }
  }

  double denominator = 0.0;
  for (int k = 0; k < 4; ++k) {
    denominator +=
        (projected_row_r[k] * projected_row_r[k] +
         projected_row_z[k] * projected_row_z[k]) /
        mass[k];
  }
  if (!(denominator > 0.0)) {
    return false;
  }
  const double dlambda = -constraint_value / denominator;
  for (int k = 0; k < 4; ++k) {
    v_r[k] += dlambda * projected_row_r[k] / mass[k];
    v_z[k] += dlambda * projected_row_z[k] / mass[k];
  }
  if (kinetic_energy_loss != nullptr) {
    *kinetic_energy_loss =
        0.5 * constraint_value * constraint_value / denominator;
  }
  return true;
}

inline constexpr int kEvacContactMortarQuadCount = 3;
inline constexpr int kEvacContactMortarSweeps = 8;
inline constexpr double kEvacContactMortarImpulseTol = 0.0;
static_assert(kEvacContactMortarQuadCount ==
              core::kEvacContactMortarRowCapacity,
              "evacuated-cell mortar hold capacity mismatch");

TENRYU_EVAC_CONTACT_HOST_DEVICE inline int
evacuated_cell_contact_mortar_rows(
    const double r[4],
    const double z[4],
    const double n[2],
    const double chi,
    const double gap0,
    const double gap1,
    const double g0,
    double row_r[kEvacContactMortarQuadCount][4],
    double row_z[kEvacContactMortarQuadCount][4],
    double area_q[kEvacContactMortarQuadCount],
    double phi_q[kEvacContactMortarQuadCount]) {
  constexpr double gauss_x[3] = {
      0.5 * (1.0 - kEvacContactGaussAbscissa),
      0.5,
      0.5 * (1.0 + kEvacContactGaussAbscissa)};
  constexpr double gauss_w[3] = {
      5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0};
  for (int q = 0; q < kEvacContactMortarQuadCount; ++q) {
    area_q[q] = 0.0;
    phi_q[q] = 0.0;
    for (int k = 0; k < 4; ++k) {
      row_r[q][k] = 0.0;
      row_z[q][k] = 0.0;
    }
  }

  const double patch_fraction =
      chi < 0.0 ? 0.0 : (chi > 1.0 ? 1.0 : chi);
  if (!(patch_fraction > 0.0)) {
    return 0;
  }

  const double mid0_r = 0.5 * (r[0] + r[2]);
  const double mid0_z = 0.5 * (z[0] + z[2]);
  const double mid1_r = 0.5 * (r[1] + r[3]);
  const double mid1_z = 0.5 * (z[1] + z[3]);
  const double delta_r = mid1_r - mid0_r;
  const double delta_z = mid1_z - mid0_z;
  const double face_length =
      std::sqrt(delta_r * delta_r + delta_z * delta_z);
  for (int q = 0; q < kEvacContactMortarQuadCount; ++q) {
    const double xi = patch_fraction * gauss_x[q];
    const double radius = mid0_r + xi * delta_r;
    const double area =
        2.0 * kEvacContactPi * radius * face_length *
        gauss_w[q] * patch_fraction;
    const double N0 = 1.0 - xi;
    const double N1 = xi;
    const double b0_r = N0 * n[0] * area;
    const double b0_z = N0 * n[1] * area;
    const double b1_r = N1 * n[0] * area;
    const double b1_z = N1 * n[1] * area;
    row_r[q][0] = -b0_r;
    row_z[q][0] = -b0_z;
    row_r[q][1] = -b1_r;
    row_z[q][1] = -b1_z;
    row_r[q][2] = b0_r;
    row_z[q][2] = b0_z;
    row_r[q][3] = b1_r;
    row_z[q][3] = b1_z;
    area_q[q] = area;
    phi_q[q] = gap0 + xi * (gap1 - gap0) - g0;
  }
  return kEvacContactMortarQuadCount;
}

TENRYU_EVAC_CONTACT_HOST_DEVICE inline bool
evacuated_cell_contact_mortar_project(
    const double mass[4],
    const int face_nodes[4],
    const int* engaged_pair_nodes,
    const double* engaged_pair_normal,
    const int engaged_pair_count,
    const double row_r[kEvacContactMortarQuadCount][4],
    const double row_z[kEvacContactMortarQuadCount][4],
    const double area_q[kEvacContactMortarQuadCount],
    const double phi_q[kEvacContactMortarQuadCount],
    const int n_rows,
    const double dt,
    double v_r[4],
    double v_z[4],
    double* const kinetic_energy_loss,
    double j_out[kEvacContactMortarQuadCount],
    const double* const correction_velocity = nullptr) {
  if (kinetic_energy_loss != nullptr) {
    *kinetic_energy_loss = 0.0;
  }
  for (int q = 0; q < kEvacContactMortarQuadCount; ++q) {
    j_out[q] = 0.0;
  }
  if (n_rows < 1 || n_rows > kEvacContactMortarQuadCount || !(dt > 0.0) ||
      engaged_pair_count < 0 || engaged_pair_count > 2 ||
      (engaged_pair_count > 0 &&
       (face_nodes == nullptr || engaged_pair_nodes == nullptr ||
        engaged_pair_normal == nullptr))) {
    return false;
  }
  for (int k = 0; k < 4; ++k) {
    if (!(mass[k] > 0.0)) {
      return false;
    }
  }

  double basis_r[2][4] = {};
  double basis_z[2][4] = {};
  for (int pair = 0; pair < engaged_pair_count; ++pair) {
    int local_a = -1;
    int local_b = -1;
    for (int k = 0; k < 4; ++k) {
      if (face_nodes[k] == engaged_pair_nodes[2 * pair]) {
        local_a = k;
      }
      if (face_nodes[k] == engaged_pair_nodes[2 * pair + 1]) {
        local_b = k;
      }
    }
    if (local_a < 0 || local_b < 0) {
      return false;
    }
    const double normal_r = engaged_pair_normal[2 * pair];
    const double normal_z = engaged_pair_normal[2 * pair + 1];
    basis_r[pair][local_a] -= normal_r;
    basis_z[pair][local_a] -= normal_z;
    basis_r[pair][local_b] += normal_r;
    basis_z[pair][local_b] += normal_z;
    for (int previous = 0; previous < pair; ++previous) {
      double basis_denominator = 0.0;
      double basis_inner = 0.0;
      for (int k = 0; k < 4; ++k) {
        basis_denominator +=
            (basis_r[previous][k] * basis_r[previous][k] +
             basis_z[previous][k] * basis_z[previous][k]) /
            mass[k];
        basis_inner +=
            (basis_r[previous][k] * basis_r[pair][k] +
             basis_z[previous][k] * basis_z[pair][k]) /
            mass[k];
      }
      if (basis_denominator > 0.0) {
        const double coefficient = basis_inner / basis_denominator;
        for (int k = 0; k < 4; ++k) {
          basis_r[pair][k] -= coefficient * basis_r[previous][k];
          basis_z[pair][k] -= coefficient * basis_z[previous][k];
        }
      }
    }
  }

  double ghat_row_r[kEvacContactMortarQuadCount][4] = {};
  double ghat_row_z[kEvacContactMortarQuadCount][4] = {};
  double phistar[kEvacContactMortarQuadCount] = {};
  bool has_candidate = false;
  for (int q = 0; q < n_rows; ++q) {
    for (int k = 0; k < 4; ++k) {
      ghat_row_r[q][k] = row_r[q][k];
      ghat_row_z[q][k] = row_z[q][k];
    }
    for (int pair = 0; pair < engaged_pair_count; ++pair) {
      double basis_denominator = 0.0;
      double basis_row_inner = 0.0;
      for (int k = 0; k < 4; ++k) {
        basis_denominator +=
            (basis_r[pair][k] * basis_r[pair][k] +
             basis_z[pair][k] * basis_z[pair][k]) /
            mass[k];
        basis_row_inner +=
            (basis_r[pair][k] * ghat_row_r[q][k] +
             basis_z[pair][k] * ghat_row_z[q][k]) /
            mass[k];
      }
      if (basis_denominator > 0.0) {
        const double coefficient = basis_row_inner / basis_denominator;
        for (int k = 0; k < 4; ++k) {
          ghat_row_r[q][k] -= coefficient * basis_r[pair][k];
          ghat_row_z[q][k] -= coefficient * basis_z[pair][k];
        }
      }
    }

    const double tiny = std::numeric_limits<double>::min();
    const double area = area_q[q] > tiny ? area_q[q] : tiny;
    double velocity_action = 0.0;
    for (int k = 0; k < 4; ++k) {
      ghat_row_r[q][k] /= area;
      ghat_row_z[q][k] /= area;
      velocity_action +=
          ghat_row_r[q][k] * v_r[k] + ghat_row_z[q][k] * v_z[k];
    }
    if (correction_velocity == nullptr) {
      phistar[q] = phi_q[q] + dt * velocity_action;
    } else {
      phistar[q] =
          phi_q[q] + dt * (velocity_action - correction_velocity[q]);
    }
    has_candidate = has_candidate || phistar[q] <= 0.0;
  }
  if (!has_candidate) {
    return false;
  }

  double delassus[kEvacContactMortarQuadCount]
                  [kEvacContactMortarQuadCount] = {};
  for (int q = 0; q < n_rows; ++q) {
    for (int p = 0; p < n_rows; ++p) {
      double inner = 0.0;
      for (int k = 0; k < 4; ++k) {
        inner +=
            (ghat_row_r[q][k] * ghat_row_r[p][k] +
             ghat_row_z[q][k] * ghat_row_z[p][k]) /
            mass[k];
      }
      delassus[q][p] = dt * inner;
    }
  }

  double impulse[kEvacContactMortarQuadCount] = {};
  for (int sweep = 0; sweep < kEvacContactMortarSweeps; ++sweep) {
    for (int q = 0; q < n_rows; ++q) {
      if (phistar[q] > 0.0 || !(delassus[q][q] > 0.0)) {
        continue;
      }
      double residual_without_diagonal = phistar[q];
      for (int p = 0; p < n_rows; ++p) {
        if (p != q) {
          residual_without_diagonal += delassus[q][p] * impulse[p];
        }
      }
      const double candidate =
          -residual_without_diagonal / delassus[q][q];
      impulse[q] = candidate > kEvacContactMortarImpulseTol
                       ? candidate
                       : kEvacContactMortarImpulseTol;
    }
  }

  bool has_impulse = false;
  for (int q = 0; q < n_rows; ++q) {
    has_impulse = has_impulse || impulse[q] > 0.0;
  }
  if (!has_impulse) {
    return false;
  }

  double projected_v_r[4] = {v_r[0], v_r[1], v_r[2], v_r[3]};
  double projected_v_z[4] = {v_z[0], v_z[1], v_z[2], v_z[3]};
  double kinetic_energy_before = 0.0;
  for (int k = 0; k < 4; ++k) {
    kinetic_energy_before +=
        0.5 * mass[k] * (v_r[k] * v_r[k] + v_z[k] * v_z[k]);
  }
  for (int q = 0; q < n_rows; ++q) {
    for (int k = 0; k < 4; ++k) {
      projected_v_r[k] += impulse[q] * ghat_row_r[q][k] / mass[k];
      projected_v_z[k] += impulse[q] * ghat_row_z[q][k] / mass[k];
    }
  }
  double kinetic_energy_after = 0.0;
  for (int k = 0; k < 4; ++k) {
    kinetic_energy_after +=
        0.5 * mass[k] *
        (projected_v_r[k] * projected_v_r[k] +
         projected_v_z[k] * projected_v_z[k]);
  }
  double energy_loss = kinetic_energy_before - kinetic_energy_after;
  const double kinetic_energy_scale =
      kinetic_energy_before > kinetic_energy_after
          ? kinetic_energy_before
          : kinetic_energy_after;
  const double energy_tolerance =
      64.0 * std::numeric_limits<double>::epsilon() * kinetic_energy_scale;
  if (energy_loss < -energy_tolerance) {
    return false;
  }
  if (energy_loss < 0.0) {
    energy_loss = 0.0;
  }

  for (int k = 0; k < 4; ++k) {
    v_r[k] = projected_v_r[k];
    v_z[k] = projected_v_z[k];
  }
  for (int q = 0; q < n_rows; ++q) {
    j_out[q] = impulse[q];
  }
  if (kinetic_energy_loss != nullptr) {
    *kinetic_energy_loss = energy_loss;
  }
  return true;
}

// Corner nodes of structured cell c (= i*nz + j) with the node convention
// n = i*(nz+1) + j. Order: (i,j), (i+1,j), (i,j+1), (i+1,j+1).
TENRYU_EVAC_CONTACT_HOST_DEVICE inline void
evacuated_cell_corner_nodes(const int cell, const int nr, const int nz,
                            int out[4]) {
  const int i = cell / nz;
  const int j = cell - i * nz;
  (void)nr;
  out[0] = i * (nz + 1) + j;
  out[1] = (i + 1) * (nz + 1) + j;
  out[2] = out[0] + 1;
  out[3] = out[1] + 1;
}

// Analytic gradient of the 2D_RZ revolved quad volume with respect to the
// 4 corner node positions, plus the volume itself. Inputs use the perimeter
// order (i,j),(i+1,j),(i+1,j+1),(i,j+1), obtained from
// evacuated_cell_corner_nodes as indices 0,1,3,2. The general structured mesh
// volume kernel traverses n00,n10,n11,n01 and uses ORIENT=+1 for ordinary
// 2D_RZ meshes. This helper maps to that same traversal, so CCW is positive.
TENRYU_EVAC_CONTACT_HOST_DEVICE inline void evacuated_cell_volume_gradient(
    const double r[4],
    const double z[4],
    double* const volume,
    double grad_r[4],
    double grad_z[4]) {
  constexpr double pi_over_three =
      1.0471975511965977461542144610931676280657231331250352736615;
  constexpr double orientation_sign = 1.0;
  double volume_sum = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int next = (k + 1) % 4;
    volume_sum +=
        (r[k] + r[next]) * (r[k] * z[next] - r[next] * z[k]);
  }
  *volume = orientation_sign * pi_over_three * volume_sum;

  for (int k = 0; k < 4; ++k) {
    const int prev = (k + 3) % 4;
    const int next = (k + 1) % 4;
    grad_r[k] = orientation_sign * pi_over_three *
                ((2.0 * r[k] + r[next]) * (z[next] - z[k]) +
                 (r[prev] + 2.0 * r[k]) * (z[k] - z[prev]));
    const double incoming = r[prev] * r[prev] + r[prev] * r[k] +
                            r[k] * r[k];
    const double outgoing = r[k] * r[k] + r[k] * r[next] +
                            r[next] * r[next];
    grad_z[k] = orientation_sign * pi_over_three *
                (incoming - outgoing);
  }
  // Preserve z-translation invariance exactly in floating-point summation.
  grad_z[3] = -((grad_z[0] + grad_z[1]) + grad_z[2]);
}

TENRYU_EVAC_CONTACT_HOST_DEVICE inline bool
evacuated_cell_contact_volume_floor_project(
    const double r[4],
    const double z[4],
    const double mass[4],
    const int cell_nodes[4],
    const int* engaged_pair_nodes,
    const double* engaged_pair_normal,
    const int engaged_pair_count,
    const double vol_at_engagement,
    double v_r[4],
    double v_z[4],
    double* const kinetic_energy_loss) {
  if (kinetic_energy_loss != nullptr) {
    *kinetic_energy_loss = 0.0;
  }
  if (!(vol_at_engagement > 0.0)) {
    return false;
  }
  double volume = 0.0;
  double grad_r[4] = {};
  double grad_z[4] = {};
  evacuated_cell_volume_gradient(r, z, &volume, grad_r, grad_z);
  double volume_rate = 0.0;
  for (int k = 0; k < 4; ++k) {
    volume_rate += grad_r[k] * v_r[k] + grad_z[k] * v_z[k];
  }
  if (!(volume <=
            kEvacContactVolumeFloorFraction * vol_at_engagement &&
        volume_rate < 0.0)) {
    return false;
  }
  if (engaged_pair_count < 0 || engaged_pair_count > 2 ||
      (engaged_pair_count > 0 &&
       (cell_nodes == nullptr || engaged_pair_nodes == nullptr ||
        engaged_pair_normal == nullptr))) {
    return false;
  }
  double denominator_full = 0.0;
  for (int k = 0; k < 4; ++k) {
    if (!(mass[k] > 0.0)) {
      return false;
    }
    denominator_full +=
        (grad_r[k] * grad_r[k] + grad_z[k] * grad_z[k]) / mass[k];
  }
  if (!(denominator_full > 0.0)) {
    return false;
  }

  double basis_r[2][4] = {};
  double basis_z[2][4] = {};
  for (int pair = 0; pair < engaged_pair_count; ++pair) {
    int local_a = -1;
    int local_b = -1;
    for (int k = 0; k < 4; ++k) {
      if (cell_nodes[k] == engaged_pair_nodes[2 * pair]) {
        local_a = k;
      }
      if (cell_nodes[k] == engaged_pair_nodes[2 * pair + 1]) {
        local_b = k;
      }
    }
    if (local_a < 0 || local_b < 0) {
      return false;
    }
    const double normal_r = engaged_pair_normal[2 * pair];
    const double normal_z = engaged_pair_normal[2 * pair + 1];
    basis_r[pair][local_a] -= normal_r;
    basis_z[pair][local_a] -= normal_z;
    basis_r[pair][local_b] += normal_r;
    basis_z[pair][local_b] += normal_z;

    // Modified Gram-Schmidt keeps the one or two contact Jacobians mutually
    // M-orthogonal before the volume gradient is projected against them.
    for (int previous = 0; previous < pair; ++previous) {
      double basis_denominator = 0.0;
      double basis_inner = 0.0;
      for (int k = 0; k < 4; ++k) {
        basis_denominator +=
            (basis_r[previous][k] * basis_r[previous][k] +
             basis_z[previous][k] * basis_z[previous][k]) /
            mass[k];
        basis_inner +=
            (basis_r[previous][k] * basis_r[pair][k] +
             basis_z[previous][k] * basis_z[pair][k]) /
            mass[k];
      }
      if (basis_denominator > 0.0) {
        const double coefficient = basis_inner / basis_denominator;
        for (int k = 0; k < 4; ++k) {
          basis_r[pair][k] -= coefficient * basis_r[previous][k];
          basis_z[pair][k] -= coefficient * basis_z[previous][k];
        }
      }
    }
  }

  double projected_grad_r[4] = {
      grad_r[0], grad_r[1], grad_r[2], grad_r[3]};
  double projected_grad_z[4] = {
      grad_z[0], grad_z[1], grad_z[2], grad_z[3]};
  for (int pair = 0; pair < engaged_pair_count; ++pair) {
    double basis_denominator = 0.0;
    double basis_gradient_inner = 0.0;
    for (int k = 0; k < 4; ++k) {
      basis_denominator +=
          (basis_r[pair][k] * basis_r[pair][k] +
           basis_z[pair][k] * basis_z[pair][k]) /
          mass[k];
      basis_gradient_inner +=
          (basis_r[pair][k] * projected_grad_r[k] +
           basis_z[pair][k] * projected_grad_z[k]) /
          mass[k];
    }
    if (basis_denominator > 0.0) {
      const double coefficient = basis_gradient_inner / basis_denominator;
      for (int k = 0; k < 4; ++k) {
        projected_grad_r[k] -= coefficient * basis_r[pair][k];
        projected_grad_z[k] -= coefficient * basis_z[pair][k];
      }
    }
  }

  double denominator = 0.0;
  for (int k = 0; k < 4; ++k) {
    denominator +=
        (projected_grad_r[k] * projected_grad_r[k] +
         projected_grad_z[k] * projected_grad_z[k]) /
        mass[k];
  }
  if (!(denominator >
        kEvacContactVolumeNullspaceEps * denominator_full)) {
    return false;
  }
  const double dlambda = -volume_rate / denominator;
  for (int k = 0; k < 4; ++k) {
    v_r[k] += dlambda * projected_grad_r[k] / mass[k];
    v_z[k] += dlambda * projected_grad_z[k] / mass[k];
  }
  // Invariants by construction: g.v_new == 0 and
  // J_pair.v_new == J_pair.v_old for every engaged pair.
  if (kinetic_energy_loss != nullptr) {
    *kinetic_energy_loss =
        0.5 * volume_rate * volume_rate / denominator;
  }
  return true;
}

double evacuated_cell_active_node_mass(
    int node,
    int nr,
    int nz,
    const std::vector<double>& corner_mass,
    const std::vector<std::uint8_t>& inactive_member_mask,
    const std::vector<std::int8_t>* hydro_active_or_null);

bool evacuated_cell_contact_reproject_pair(
    core::EvacContactSlot& slot,
    int pair,
    int nr,
    int nz,
    const std::vector<double>& corner_mass,
    const std::vector<std::uint8_t>& inactive_member_mask,
    const std::vector<std::int8_t>* hydro_active_or_null,
    const std::vector<double>& mass,
    std::vector<double>& v_r,
    std::vector<double>& v_z,
    std::vector<double>& ei);

bool evacuated_cell_refill_viable(
    double m_fill,
    double m_ref,
    double rho_fill,
    double rho_donor_median,
    const core::Config::NumericsConfig::AleConfig::EvacuatedCellConfig::
        EvacuatedCellClosureContactConfig& cfg);

bool evacuated_cell_live_closure_eligible(
    double mass,
    double m_ref,
    double vol,
    double V_ref,
    std::uint8_t lineage,
    const core::Config::NumericsConfig::AleConfig::EvacuatedCellConfig::
        EvacuatedCellClosureContactConfig& cfg);

inline double evacuated_cell_contact_lambda(const double mA,
                                            const double mB,
                                            const double aA_n,
                                            const double aB_n) {
  if (!(mA > 0.0) || !(mB > 0.0)) {
    return 0.0;
  }
  return (aA_n - aB_n) * (mA * mB / (mA + mB));
}

inline double evacuated_cell_contact_common_accel(const double mA,
                                                  const double mB,
                                                  const double aA_n,
                                                  const double aB_n) {
  if (!(mA > 0.0) || !(mB > 0.0)) {
    return 0.0;
  }
  return (mA * aA_n + mB * aB_n) / (mA + mB);
}

struct EvacContactProjectionCorrection {
  double a = 0.0;
  double b = 0.0;
};

TENRYU_EVAC_CONTACT_HOST_DEVICE inline EvacContactProjectionCorrection
evacuated_cell_contact_projection_correction(const double x_r_a_cand,
                                             const double x_z_a_cand,
                                             const double x_r_b_cand,
                                             const double x_z_b_cand,
                                             const double x_r_a_old,
                                             const double x_z_a_old,
                                             const double x_r_b_old,
                                             const double x_z_b_old,
                                             const double normal_r,
                                             const double normal_z) {
  const double da_n = (x_r_a_cand - x_r_a_old) * normal_r +
                      (x_z_a_cand - x_z_a_old) * normal_z;
  const double db_n = (x_r_b_cand - x_r_b_old) * normal_r +
                      (x_z_b_cand - x_z_b_old) * normal_z;
  const double dc_n = 0.5 * (da_n + db_n);
  return {dc_n - da_n, dc_n - db_n};
}

TENRYU_EVAC_CONTACT_HOST_DEVICE inline double
evacuated_cell_contact_junction_deff(const double* const deff,
                                     const int cell,
                                     const int nr,
                                     const int nz) {
  const int i = cell / nz;
  const int j = cell - i * nz;
  double neighbor_max = 0.0;
  if (i > 0 && deff[cell - nz] > neighbor_max) {
    neighbor_max = deff[cell - nz];
  }
  if (i + 1 < nr && deff[cell + nz] > neighbor_max) {
    neighbor_max = deff[cell + nz];
  }
  if (j > 0 && deff[cell - 1] > neighbor_max) {
    neighbor_max = deff[cell - 1];
  }
  if (j + 1 < nz && deff[cell + 1] > neighbor_max) {
    neighbor_max = deff[cell + 1];
  }
  return neighbor_max > 0.0 ? 1.0e3 * neighbor_max : 0.0;
}

EvacCellRematerializeTransfer rematerialize_evacuated_cell_extensives(
    int cell,
    double cell_volume,
    const std::vector<int>& donors,
    const std::vector<double>& weights,
    double neighbor_change_max,
    const std::vector<double>& rho,
    std::vector<double>& mass,
    std::vector<double>& ee,
    std::vector<double>& ei);

enum class EvacCellRematerializeTrigger {
  NONE,
  TIMED,
  VOLUME,
  PREDICTED,
};

EvacCellRematerializeTrigger evacuated_cell_rematerialize_trigger(
    double volume,
    double previous_volume,
    double reference_volume,
    double volume_fraction,
    int rematerialize_after_evaluations,
    int& evaluations_since_conversion);

// Returns whether conversion is blocked for this evaluation and consumes one
// evaluation from a positive dwell counter.
bool consume_evacuated_cell_rematerialize_dwell(int& dwell_remaining);

// No-op unless enabled on an evaluation step for a structured 2D single block.
void evacuated_cell_shadow_step(core::State& state, const core::Config& cfg);

// Transactional action-mode conversion after the committed ALE phase.
void evacuated_cell_controller_step(core::State& state, const core::Config& cfg);

// Per-committed-step finite-gap contact monitor and engagement transaction.
void evacuated_cell_contact_step(core::State& state, const core::Config& cfg);

void evacuated_cell_contact_probe(core::State& state,
                                  const core::Config& cfg,
                                  const char* label);

#undef TENRYU_EVAC_CONTACT_HOST_DEVICE

}  // namespace tenryu::hydro
