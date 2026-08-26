#include "hydro/corner_mass_remap_audit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/ale_remap.cuh"
#include "hydro/ke_fixup_deposit.hpp"
#include "hydro/oriented_swept_volume.cuh"
#include "hydro/rz_corner_mass.cuh"

namespace tenryu::hydro::ale {

namespace {

// Production donor conventions, mirrored verbatim from the TU-local
// helpers in ale_remap_2d_rz.cu (csr_internal_flux_donor /
// csr_internal_flux_losing_cell). swept_volume_sign_fixed=false keeps the
// documented legacy sign-reversed donor; the audit must transport with
// whatever convention the production remap actually used.
inline int flux_donor(const int cell_a,
                      const int cell_b,
                      const double dV_a,
                      const bool sign_fixed) {
  return make_oriented_swept_volume(
             cell_a,
             cell_b,
             dV_a,
             swept_volume_convention_from_flag(sign_fixed))
      .donor;
}

inline int flux_losing_cell(const int cell_a,
                            const int cell_b,
                            const double dV_a) {
  return (dV_a > 0.0) ? cell_b : cell_a;
}

inline int active_nverts(const std::uint8_t* cell_nverts, const int cell) {
  if (cell_nverts == nullptr) {
    return 4;
  }
  const int n = static_cast<int>(cell_nverts[cell]);
  return (n == 3) ? 3 : 4;
}

inline bool cell_inactive(const std::uint8_t* mask, const int cell) {
  return mask != nullptr && mask[cell] != 0U;
}

inline double clamp01_host(const double v) {
  return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

}  // namespace

CornerMassRemapLoResult lo_corner_mass_remap(const CornerMassRemapLoInput& in) {
  CornerMassRemapLoResult out;
  if (in.n_cells <= 0 || in.n_nodes <= 0 || in.n_faces < 0 ||
      in.cell_node_csr_offsets == nullptr ||
      in.cell_node_csr_indices == nullptr || in.x_r_old == nullptr ||
      in.x_z_old == nullptr || in.x_r_new == nullptr ||
      in.x_z_new == nullptr || in.corner_mass_old == nullptr ||
      in.cell_mass_old == nullptr || in.cell_vol_old == nullptr ||
      in.cell_mass_new == nullptr ||
      (in.n_faces > 0 &&
       (in.face_cell_a == nullptr || in.face_cell_b == nullptr ||
        in.face_local_a == nullptr || in.face_local_b == nullptr))) {
    return out;
  }

  const std::size_t n4 = static_cast<std::size_t>(in.n_cells) * 4U;
  out.corner_mass.assign(n4, 0.0);
  for (int c = 0; c < in.n_cells; ++c) {
    if (cell_inactive(in.inactive_mask, c)) {
      continue;
    }
    const int nv = active_nverts(in.cell_nverts, c);
    for (int k = 0; k < nv; ++k) {
      out.corner_mass[static_cast<std::size_t>(c) * 4U + k] =
          in.corner_mass_old[static_cast<std::size_t>(c) * 4U + k];
    }
  }

  // ---- Stage 1: LO face transport (half split to the face's 2 corners) --
  for (int f = 0; f < in.n_faces; ++f) {
    const int cell_a = in.face_cell_a[f];
    const int cell_b = in.face_cell_b[f];
    if (cell_a < 0 || cell_b < 0 || cell_a >= in.n_cells ||
        cell_b >= in.n_cells) {
      continue;
    }
    if (cell_inactive(in.inactive_mask, cell_a) ||
        cell_inactive(in.inactive_mask, cell_b)) {
      continue;  // the remap's faces-touching-inactive are zero-flux
    }
    const double dV_a = detail::csr_face_swept_volume_outward(
        in.x_r_old, in.x_z_old, in.x_r_new, in.x_z_new,
        in.cell_node_csr_offsets, in.cell_node_csr_indices,
        in.cell_orientation_sign, cell_a, in.face_local_a[f],
        in.cell_nverts);
    if (!std::isfinite(dV_a) || dV_a == 0.0) {
      continue;
    }
    const int donor =
        flux_donor(cell_a, cell_b, dV_a, in.swept_volume_sign_fixed);
    const int loser = flux_losing_cell(cell_a, cell_b, dV_a);
    const int gainer = (loser == cell_a) ? cell_b : cell_a;
    const double vol_donor = in.cell_vol_old[donor];
    if (!(vol_donor > 0.0) || !std::isfinite(vol_donor)) {
      continue;
    }
    const double rho_donor = in.cell_mass_old[donor] / vol_donor;
    const double flux = std::abs(dV_a) * std::max(rho_donor, 0.0);
    if (!(flux > 0.0) || !std::isfinite(flux)) {
      continue;
    }
    out.flux_abs_sum += flux;

    int corner_l0 = 0;
    int corner_l1 = 0;
    int corner_g0 = 0;
    int corner_g1 = 0;
    const int local_l = (loser == cell_a) ? in.face_local_a[f]
                                          : in.face_local_b[f];
    const int local_g = (gainer == cell_a) ? in.face_local_a[f]
                                           : in.face_local_b[f];
    if (!detail::csr_active_local_face_corners(in.cell_nverts, loser,
                                               local_l, &corner_l0,
                                               &corner_l1) ||
        !detail::csr_active_local_face_corners(in.cell_nverts, gainer,
                                               local_g, &corner_g0,
                                               &corner_g1)) {
      continue;
    }
    const double half = 0.5 * flux;
    out.corner_mass[static_cast<std::size_t>(loser) * 4U + corner_l0] -= half;
    out.corner_mass[static_cast<std::size_t>(loser) * 4U + corner_l1] -= half;
    out.corner_mass[static_cast<std::size_t>(gainer) * 4U + corner_g0] += half;
    out.corner_mass[static_cast<std::size_t>(gainer) * 4U + corner_g1] += half;
  }

  // ---- Stage 2: positivity clip + per-cell cell-sum projection ----------
  // Constraint (non-negotiable, verdict (a)): sum_{a in c} m_a = m_c^new
  // with m_a >= 0. Small active-set least-squares: distribute the cell
  // residual uniformly over free corners, clip new negatives to the bound,
  // repeat (<= 4 rounds for 4 unknowns).
  for (int c = 0; c < in.n_cells; ++c) {
    if (cell_inactive(in.inactive_mask, c)) {
      continue;
    }
    ++out.n_active_cells;
    const int nv = active_nverts(in.cell_nverts, c);
    double* m = &out.corner_mass[static_cast<std::size_t>(c) * 4U];
    const double target = in.cell_mass_new[c];
    double pre_projection[4] = {m[0], m[1], m[2], m[3]};

    for (int k = 0; k < nv; ++k) {
      if (m[k] < 0.0) {
        ++out.negative_clip_count;
        m[k] = 0.0;
      }
    }
    if (!(target > 0.0) || !std::isfinite(target)) {
      ++out.nonpositive_target_cells;
      for (int k = 0; k < nv; ++k) {
        m[k] = 0.0;
      }
    } else {
      bool at_bound[4] = {false, false, false, false};
      for (int round = 0; round < 4; ++round) {
        double sum = 0.0;
        int n_free = 0;
        for (int k = 0; k < nv; ++k) {
          sum += m[k];
          if (!at_bound[k]) {
            ++n_free;
          }
        }
        const double residual = target - sum;
        if (n_free == 0 || std::abs(residual) <=
                               1.0e-15 * std::max(target, 1.0e-300)) {
          break;
        }
        const double delta = residual / static_cast<double>(n_free);
        bool clipped = false;
        for (int k = 0; k < nv; ++k) {
          if (at_bound[k]) {
            continue;
          }
          m[k] += delta;
          if (m[k] < 0.0) {
            m[k] = 0.0;
            at_bound[k] = true;
            clipped = true;
          }
        }
        if (!clipped) {
          break;
        }
      }
      // Exact final balance on the free set (kills the residual the loop
      // tolerance left behind; free corners stay positive because the
      // last unclipped pass ended above the bound).
      double sum = 0.0;
      int n_free = 0;
      for (int k = 0; k < nv; ++k) {
        sum += m[k];
        if (m[k] > 0.0) {
          ++n_free;
        }
      }
      if (n_free > 0) {
        const double delta = (target - sum) / static_cast<double>(n_free);
        for (int k = 0; k < nv; ++k) {
          if (m[k] > 0.0) {
            m[k] += delta;
          }
        }
      }
    }
    for (int k = 0; k < nv; ++k) {
      out.projection_repair_abs_sum += std::abs(m[k] - pre_projection[k]);
    }
  }

  // ---- Stage 3: nodal sums (M_n > 0 probe) ------------------------------
  std::vector<double> node_mass(static_cast<std::size_t>(in.n_nodes), 0.0);
  std::vector<std::uint8_t> node_has(static_cast<std::size_t>(in.n_nodes),
                                     0U);
  for (int c = 0; c < in.n_cells; ++c) {
    if (cell_inactive(in.inactive_mask, c)) {
      continue;
    }
    const int nv = active_nverts(in.cell_nverts, c);
    const int off = in.cell_node_csr_offsets[c];
    for (int k = 0; k < nv; ++k) {
      const int n = in.cell_node_csr_indices[off + k];
      if (n < 0 || n >= in.n_nodes) {
        continue;
      }
      node_mass[static_cast<std::size_t>(n)] +=
          out.corner_mass[static_cast<std::size_t>(c) * 4U + k];
      node_has[static_cast<std::size_t>(n)] = 1U;
    }
  }
  double min_node = std::numeric_limits<double>::infinity();
  for (int n = 0; n < in.n_nodes; ++n) {
    if (node_has[static_cast<std::size_t>(n)] != 0U) {
      min_node = std::min(min_node, node_mass[static_cast<std::size_t>(n)]);
    }
  }
  out.min_node_mass_sum = std::isfinite(min_node) ? min_node : 0.0;
  out.valid = true;
  return out;
}

bool corner_mass_remap_audit_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_CORNER_MASS_REMAP_AUDIT");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  return enabled;
}

bool pr4_corner_mass_install_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_INSTALL_PR4_CORNER_MASS");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  return enabled;
}

bool vpaired_corner_mass_install_enabled(const core::Config& cfg) {
  static const char* const raw =
      std::getenv("TENRYU_I1B_INSTALL_VPAIRED_CORNER_MASS");
  static const bool enabled =
      raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  if (enabled) {
    return true;
  }
  // The basis-coherent transport REQUIRES the install (its velocity
  // re-recover divides by the V-paired masses the next Lagrangian step
  // must use), so the coherent env implies it — one switch.
  static const char* const coherent =
      std::getenv("TENRYU_I1B_OPTIONB_COHERENT");
  static const bool coherent_env_present =
      coherent != nullptr && coherent[0] != '\0';
  static const bool coherent_env_enabled =
      coherent_env_present && coherent[0] != '0';
  return coherent_env_present ? coherent_env_enabled
                              : cfg.numerics.ale.csr_optionb_coherent_enabled;
}

bool install_ke_compensation_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_INSTALL_KE_COMPENSATION");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  return enabled;
}

InstallKeJump compute_install_ke_jump(const int n_cells,
                                      const int n_nodes,
                                      const int* cell_node_csr_offsets,
                                      const int* cell_node_csr_indices,
                                      const std::uint8_t* cell_nverts,
                                      const double* corner_mass_old,
                                      const double* corner_mass_new,
                                      const double* v_r,
                                      const double* v_z,
                                      const std::uint8_t* inactive_mask) {
  InstallKeJump out;
  if (n_cells <= 0 || n_nodes <= 0 || cell_node_csr_offsets == nullptr ||
      cell_node_csr_indices == nullptr || corner_mass_old == nullptr ||
      corner_mass_new == nullptr || v_r == nullptr || v_z == nullptr) {
    return out;
  }
  out.dKE.assign(static_cast<std::size_t>(n_cells), 0.0);
  long double total = 0.0L;
  for (int c = 0; c < n_cells; ++c) {
    if (cell_inactive(inactive_mask, c)) {
      continue;
    }
    const int nv = active_nverts(cell_nverts, c);
    const int off = cell_node_csr_offsets[c];
    long double dke = 0.0L;
    for (int k = 0; k < nv; ++k) {
      const int n = cell_node_csr_indices[off + k];
      if (n < 0 || n >= n_nodes) {
        continue;
      }
      const std::size_t idx = static_cast<std::size_t>(c) * 4U +
                              static_cast<std::size_t>(k);
      const double dm = corner_mass_new[idx] - corner_mass_old[idx];
      if (dm == 0.0) {
        continue;
      }
      const double vr = v_r[static_cast<std::size_t>(n)];
      const double vz = v_z[static_cast<std::size_t>(n)];
      if (!std::isfinite(vr) || !std::isfinite(vz)) {
        continue;
      }
      dke += 0.5L * static_cast<long double>(dm) *
             (static_cast<long double>(vr) * vr +
              static_cast<long double>(vz) * vz);
    }
    out.dKE[static_cast<std::size_t>(c)] = static_cast<double>(dke);
    total += dke;
  }
  out.total = static_cast<double>(total);
  out.valid = true;
  return out;
}

VPairedCornerMassResult compute_vpaired_corner_mass(
    const int n_cells,
    const int n_nodes,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const std::uint8_t* cell_nverts,
    const double* x_r,
    const double* x_z,
    const double* cell_mass,
    const double* cell_vol,
    const std::uint8_t* inactive_mask) {
  VPairedCornerMassResult out;
  if (n_cells <= 0 || n_nodes <= 0 || cell_node_csr_offsets == nullptr ||
      cell_node_csr_indices == nullptr || x_r == nullptr || x_z == nullptr ||
      cell_mass == nullptr) {
    return out;
  }
  out.corner_mass.assign(static_cast<std::size_t>(n_cells) * 4U, 0.0);
  for (int c = 0; c < n_cells; ++c) {
    if (cell_inactive(inactive_mask, c)) {
      continue;
    }
    const double m_c = cell_mass[static_cast<std::size_t>(c)];
    if (!(m_c > 0.0) || !std::isfinite(m_c)) {
      ++out.degenerate_cells;
      continue;
    }
    const int nv = active_nverts(cell_nverts, c);
    const int off = cell_node_csr_offsets[c];
    double r[4] = {0.0, 0.0, 0.0, 0.0};
    double z[4] = {0.0, 0.0, 0.0, 0.0};
    bool ok = true;
    for (int k = 0; k < nv; ++k) {
      const int n = cell_node_csr_indices[off + k];
      if (n < 0 || n >= n_nodes) {
        ok = false;
        break;
      }
      r[k] = x_r[n];
      z[k] = x_z[n];
    }
    if (!ok) {
      ++out.degenerate_cells;
      continue;
    }
    double v_corner[4] = {0.0, 0.0, 0.0, 0.0};
    if (nv == 3) {
      // Triangle quadrants — same construction as the subzonal-pressure
      // consumer (compatible_subzonal_pressure.cu triangle_corner_volumes):
      // vertex / edge midpoints / centroid, |exact RZ polygon volume|.
      const double rc = (r[0] + r[1] + r[2]) / 3.0;
      const double zc = (z[0] + z[1] + z[2]) / 3.0;
      for (int k = 0; k < 3; ++k) {
        const int kp = (k + 1) % 3;
        const int km = (k + 2) % 3;
        const double r_sub[4] = {r[k], 0.5 * (r[k] + r[kp]), rc,
                                 0.5 * (r[km] + r[k])};
        const double z_sub[4] = {z[k], 0.5 * (z[k] + z[kp]), zc,
                                 0.5 * (z[km] + z[k])};
        v_corner[k] = std::abs(rz::rz_polygon_volume_exact(r_sub, z_sub, 4));
      }
    } else {
      rz::compute_quad_corner_volumes_exact_subpolygon(
          r[0], z[0], r[1], z[1], r[2], z[2], r[3], z[3], v_corner);
    }
    double v_sum = 0.0;
    bool v_ok = true;
    for (int k = 0; k < nv; ++k) {
      if (!(v_corner[k] > 0.0) || !std::isfinite(v_corner[k])) {
        v_ok = false;
        break;
      }
      v_sum += v_corner[k];
    }
    if (!v_ok || !(v_sum > 0.0)) {
      ++out.degenerate_cells;
      continue;
    }
    // m_a = rho_c * V_a with the multiplicative cell-sum scale folded in:
    // m_a = m_c * V_a / sum(V) (exact sum by construction). The scale
    // deviation vs the state's own cell volume is the audit quantity —
    // and identically the post-install corner-density deviation
    // |rho_a/rho_c - 1| (see header).
    for (int k = 0; k < nv; ++k) {
      out.corner_mass[static_cast<std::size_t>(c) * 4U +
                      static_cast<std::size_t>(k)] = m_c * v_corner[k] / v_sum;
    }
    if (cell_vol != nullptr) {
      const double v_state = cell_vol[static_cast<std::size_t>(c)];
      if (std::isfinite(v_state) && v_state > 0.0) {
        out.max_scale_dev =
            std::max(out.max_scale_dev, std::abs(v_state / v_sum - 1.0));
      }
    }
  }
  out.valid = true;
  return out;
}

CornerMassInstallCheck check_corner_mass_for_install(
    const int n_cells,
    const int n_nodes,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const std::uint8_t* cell_nverts,
    const double* corner_mass_candidate,
    const double* cell_mass,
    const std::uint8_t* inactive_mask) {
  CornerMassInstallCheck check;
  if (n_cells <= 0 || n_nodes <= 0 || cell_node_csr_offsets == nullptr ||
      cell_node_csr_indices == nullptr || corner_mass_candidate == nullptr ||
      cell_mass == nullptr) {
    return check;
  }
  std::vector<double> node_mass(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<std::uint8_t> node_has(static_cast<std::size_t>(n_nodes), 0U);
  for (int c = 0; c < n_cells; ++c) {
    const bool inactive = cell_inactive(inactive_mask, c);
    const int nv = active_nverts(cell_nverts, c);
    const int off = cell_node_csr_offsets[c];
    double sum = 0.0;
    for (int k = 0; k < nv; ++k) {
      const double m =
          corner_mass_candidate[static_cast<std::size_t>(c) * 4U +
                                static_cast<std::size_t>(k)];
      if (!inactive) {
        if (!(m >= 0.0) || !std::isfinite(m)) {
          ++check.negative_corners;
        }
        sum += m;
      }
      const int n = cell_node_csr_indices[off + k];
      if (n < 0 || n >= n_nodes) {
        continue;
      }
      if (!inactive && std::isfinite(m) && m > 0.0) {
        node_mass[static_cast<std::size_t>(n)] += m;
        node_has[static_cast<std::size_t>(n)] = 1U;
      }
    }
    if (!inactive) {
      const double target = cell_mass[static_cast<std::size_t>(c)];
      if (std::isfinite(target) && target > 0.0) {
        const double rel = std::abs(sum - target) / target;
        check.max_cell_sum_rel_err = std::max(check.max_cell_sum_rel_err, rel);
        if (rel > 1.0e-12) {
          ++check.bad_sum_cells;
        }
      }
    }
  }
  double min_node = std::numeric_limits<double>::infinity();
  for (int n = 0; n < n_nodes; ++n) {
    if (node_has[static_cast<std::size_t>(n)] != 0U) {
      min_node = std::min(min_node, node_mass[static_cast<std::size_t>(n)]);
    }
  }
  check.min_node_mass = std::isfinite(min_node) ? min_node : 0.0;
  check.ok = check.negative_corners == 0 && check.bad_sum_cells == 0 &&
             check.min_node_mass > 0.0;
  return check;
}

void corner_mass_remap_audit_capture_pre(const core::State& state,
                                         CornerMassRemapAuditPre* pre) {
  TENRYU_ASSERT(
      state.corner_stride == 4,
      "corner_stride != 4: remap-audit corner path is staged for a later revision");
  state.x_r.copy_to_host(pre->x_r);
  state.x_z.copy_to_host(pre->x_z);
  state.mass.copy_to_host(pre->mass);
  state.vol.copy_to_host(pre->vol);
  state.corner_mass.copy_to_host(pre->corner_mass);
}

void corner_mass_remap_audit_emit(
    core::State& state,
    const core::Config& cfg,
    const CornerMassRemapAuditPre& pre,
    const std::vector<double>* coherent_corner_mass) {
  if (!state.mesh.topo.multiblock.has_value()) {
    return;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (n_cells <= 0 || n_nodes <= 0 ||
      pre.corner_mass.size() != static_cast<std::size_t>(n_cells) * 4U ||
      static_cast<int>(pre.mass.size()) != n_cells ||
      static_cast<int>(pre.x_r.size()) != n_nodes ||
      mb.cell_node_csr_indices.size() <
          static_cast<std::size_t>(n_cells) * 4U) {
    return;
  }

  std::vector<double> x_r_new;
  std::vector<double> x_z_new;
  std::vector<double> mass_new;
  state.x_r.copy_to_host(x_r_new);
  state.x_z.copy_to_host(x_z_new);
  state.mass.copy_to_host(mass_new);

  const int n_faces = static_cast<int>(mb.unique_internal_faces.size());
  std::vector<int> face_cell_a(static_cast<std::size_t>(n_faces));
  std::vector<int> face_cell_b(static_cast<std::size_t>(n_faces));
  std::vector<int> face_local_a(static_cast<std::size_t>(n_faces));
  std::vector<int> face_local_b(static_cast<std::size_t>(n_faces));
  for (int f = 0; f < n_faces; ++f) {
    const auto& face = mb.unique_internal_faces[static_cast<std::size_t>(f)];
    face_cell_a[static_cast<std::size_t>(f)] = face.cell_a;
    face_cell_b[static_cast<std::size_t>(f)] = face.cell_b;
    face_local_a[static_cast<std::size_t>(f)] = face.local_a;
    face_local_b[static_cast<std::size_t>(f)] = face.local_b;
  }

  std::vector<std::uint8_t> inactive;
  if (state.central_pseudo_core.inactive_member_mask.size() ==
      static_cast<std::size_t>(n_cells)) {
    inactive = state.central_pseudo_core.inactive_member_mask;
  }
  if (state.pole_angular_derefine.inactive_member_mask.size() ==
      static_cast<std::size_t>(n_cells)) {
    if (inactive.empty()) {
      inactive.assign(static_cast<std::size_t>(n_cells), 0U);
    }
    for (int c = 0; c < n_cells; ++c) {
      if (state.pole_angular_derefine
              .inactive_member_mask[static_cast<std::size_t>(c)] != 0U) {
        inactive[static_cast<std::size_t>(c)] = 1U;
      }
    }
  }
  const bool have_inactive =
      static_cast<int>(inactive.size()) == n_cells;

  CornerMassRemapLoInput in;
  in.n_cells = n_cells;
  in.n_nodes = n_nodes;
  in.cell_node_csr_offsets = mb.cell_node_csr_offsets.data();
  in.cell_node_csr_indices = mb.cell_node_csr_indices.data();
  in.cell_nverts = state.mesh.cell_nverts.empty()
                       ? nullptr
                       : state.mesh.cell_nverts.data();
  in.cell_orientation_sign = mb.cell_orientation_sign.empty()
                                 ? nullptr
                                 : mb.cell_orientation_sign.data();
  in.n_faces = n_faces;
  in.face_cell_a = face_cell_a.data();
  in.face_cell_b = face_cell_b.data();
  in.face_local_a = face_local_a.data();
  in.face_local_b = face_local_b.data();
  in.x_r_old = pre.x_r.data();
  in.x_z_old = pre.x_z.data();
  in.x_r_new = x_r_new.data();
  in.x_z_new = x_z_new.data();
  in.corner_mass_old = pre.corner_mass.data();
  in.cell_mass_old = pre.mass.data();
  in.cell_vol_old = pre.vol.data();
  in.cell_mass_new = mass_new.data();
  in.inactive_mask = have_inactive ? inactive.data() : nullptr;
  in.swept_volume_sign_fixed = cfg.numerics.ale.swept_volume_sign_fixed;

  // PR6-LO: V-paired install takes precedence — the (m,V)-paired product
  // is the basis-contract construction; the PR4 LO transport stays as the
  // audit/refuted-install reference. Under the basis-coherent transport
  // the candidate is the component's exported post-projection ledger
  // (identical formula, evaluated on the component's own cell sums) so the
  // installed basis and the velocity re-recover share literally the same
  // numbers; degenerate-geometry cells carry the transported partition
  // renormalized to the cell sum instead of abandoning the whole install.
  if (vpaired_corner_mass_install_enabled(cfg)) {
    const bool use_coherent =
        coherent_corner_mass != nullptr &&
        coherent_corner_mass->size() == static_cast<std::size_t>(n_cells) * 4U;
    VPairedCornerMassResult vp;
    if (!use_coherent) {
      std::vector<double> vol_new;
      state.vol.copy_to_host(vol_new);
      vp = compute_vpaired_corner_mass(
          n_cells, n_nodes, in.cell_node_csr_offsets, in.cell_node_csr_indices,
          in.cell_nverts, x_r_new.data(), x_z_new.data(), mass_new.data(),
          vol_new.data(), in.inactive_mask);
    }
    if (use_coherent || vp.valid) {
      std::vector<double> candidate =
          use_coherent ? *coherent_corner_mass : vp.corner_mass;
      if (have_inactive) {
        for (int c = 0; c < n_cells; ++c) {
          if (inactive[static_cast<std::size_t>(c)] != 0U) {
            for (int k = 0; k < 4; ++k) {
              const std::size_t idx = static_cast<std::size_t>(c) * 4U +
                                      static_cast<std::size_t>(k);
              candidate[idx] = pre.corner_mass[idx];
            }
          }
        }
      }
      const CornerMassInstallCheck check = check_corner_mass_for_install(
          n_cells, n_nodes, in.cell_node_csr_offsets,
          in.cell_node_csr_indices, in.cell_nverts, candidate.data(),
          mass_new.data(), in.inactive_mask);
      bool comp_ok = true;
      double comp_dke_total = 0.0;
      double comp_deposited = 0.0;
      long long comp_shortfall_cells = 0;
      std::vector<double> comp_ee;
      std::vector<double> comp_ei;
      bool comp_write = false;
      // Under the coherent transport the TER deposit chain already books
      // the basis-swap KE (pre-K in the pre basis, post-K in the projected
      // basis), so the per-install jump compensation would double-count.
      if (check.ok && !use_coherent && install_ke_compensation_enabled()) {
        // Per-install KE compensation: replacing the basis at fixed node
        // velocities jumps the budget KE by
        // dKE = sum 1/2 (m_new - m_old) |v|^2 — pure bookkeeping, the
        // adjudicated ~5.6e-6/install positive leak. Deposit -dKE_c into
        // each cell's internal energy (PR3 ye split) BEFORE installing,
        // staged on host arrays so deposit+install is atomic. Capacity: a
        // cell whose deposit would drain U_c below 10% marks a shortfall;
        // the cell-sum contract forbids reverting single cells (their old
        // corner sums no longer match the post-remap masses), so ANY
        // shortfall abandons the WHOLE install with a warning — never
        // floor-and-lose.
        std::vector<double> v_r_now;
        std::vector<double> v_z_now;
        state.v_r.copy_to_host(v_r_now);
        state.v_z.copy_to_host(v_z_now);
        const InstallKeJump jump = compute_install_ke_jump(
            n_cells, n_nodes, in.cell_node_csr_offsets,
            in.cell_node_csr_indices, in.cell_nverts,
            pre.corner_mass.data(), candidate.data(), v_r_now.data(),
            v_z_now.data(), in.inactive_mask);
        if (!jump.valid) {
          comp_ok = false;
        } else {
          comp_dke_total = jump.total;
          state.ee.copy_to_host(comp_ee);
          state.ei.copy_to_host(comp_ei);
          for (int c = 0; c < n_cells; ++c) {
            const std::size_t uc = static_cast<std::size_t>(c);
            const double dke = jump.dKE[uc];
            if (dke == 0.0) {
              continue;
            }
            const double m = mass_new[uc];
            if (!(m > 0.0) || !std::isfinite(m)) {
              ++comp_shortfall_cells;
              comp_ok = false;
              break;
            }
            const double e_int = comp_ee[uc] + comp_ei[uc];
            const double u_c = m * e_int;
            const double delta = -dke;
            if (!(u_c > 0.0) || u_c + delta < 0.1 * u_c) {
              ++comp_shortfall_cells;
              comp_ok = false;
              break;
            }
            const double ye = (e_int > 0.0 && std::isfinite(e_int))
                                  ? clamp01_host(comp_ee[uc] / e_int)
                                  : 0.5;
            const double de = delta / m;
            comp_ee[uc] += ye * de;
            comp_ei[uc] += (1.0 - ye) * de;
            comp_deposited += delta;
            comp_write = true;
          }
        }
      }
      if (check.ok && comp_ok) {
        if (comp_write) {
          state.ee.copy_from_host(comp_ee);
          state.ei.copy_from_host(comp_ei);
        }
        state.corner_mass.copy_from_host(candidate);
        state.corner_mass_is_lagrangian_invariant = false;
        std::ostringstream voss;
        voss.setf(std::ios::scientific);
        voss.precision(6);
        // max_scale_dev is identically the post-install corner-density
        // deviation max|rho_a/rho_c - 1| (one number, both task metrics).
        voss << "[vpaired_corner_mass_install] step=" << state.step
             << " installed=1 product="
             << (use_coherent ? "coherent" : "recompute")
             << " max_scale_dev=" << vp.max_scale_dev
             << " degenerate_cells=" << vp.degenerate_cells
             << " max_cell_sum_rel_err=" << check.max_cell_sum_rel_err
             << " min_node_mass=" << check.min_node_mass;
        if (!use_coherent && install_ke_compensation_enabled()) {
          voss << " ke_comp_dKE=" << comp_dke_total
               << " ke_comp_deposited=" << comp_deposited;
        }
        core::log_info(voss.str());
      } else if (check.ok) {
        std::ostringstream coss;
        coss.setf(std::ios::scientific);
        coss.precision(6);
        coss << "[vpaired_corner_mass_install] step=" << state.step
             << " installed=0 KE_COMP_ABANDONED shortfall_cells="
             << comp_shortfall_cells << " dKE_total=" << comp_dke_total;
        core::log_warning(coss.str());
      } else {
        std::ostringstream woss;
        woss.setf(std::ios::scientific);
        woss.precision(6);
        woss << "[vpaired_corner_mass_install] step=" << state.step
             << " installed=0 ABANDONED product="
             << (use_coherent ? "coherent" : "recompute")
             << " negative_corners=" << check.negative_corners
             << " bad_sum_cells=" << check.bad_sum_cells
             << " max_cell_sum_rel_err=" << check.max_cell_sum_rel_err
             << " min_node_mass=" << check.min_node_mass
             << " degenerate_cells=" << vp.degenerate_cells;
        core::log_warning(woss.str());
      }
    }
  }

  const bool want_lo = corner_mass_remap_audit_enabled() ||
                       (pr4_corner_mass_install_enabled() &&
                        !vpaired_corner_mass_install_enabled(cfg));
  if (!want_lo) {
    return;
  }

  const CornerMassRemapLoResult lo = lo_corner_mass_remap(in);
  if (!lo.valid) {
    return;
  }

  // PR5(a): install the PR4 product into state.corner_mass. The composite
  // candidate keeps the FROZEN mirror values on inactive macro members
  // (the rim machinery reads member-side mirrors; PR4's zeros there are a
  // transport-domain convention, not a basis statement). Violations
  // abandon the install with a warning — never install a bad field.
  // Superseded by the V-paired install when both envs are set.
  if (pr4_corner_mass_install_enabled() &&
      !vpaired_corner_mass_install_enabled(cfg)) {
    std::vector<double> candidate = lo.corner_mass;
    if (have_inactive) {
      for (int c = 0; c < n_cells; ++c) {
        if (inactive[static_cast<std::size_t>(c)] != 0U) {
          for (int k = 0; k < 4; ++k) {
            const std::size_t idx = static_cast<std::size_t>(c) * 4U +
                                    static_cast<std::size_t>(k);
            candidate[idx] = pre.corner_mass[idx];
          }
        }
      }
    }
    const CornerMassInstallCheck check = check_corner_mass_for_install(
        n_cells, n_nodes, in.cell_node_csr_offsets, in.cell_node_csr_indices,
        in.cell_nverts, candidate.data(), mass_new.data(), in.inactive_mask);
    if (check.ok) {
      state.corner_mass.copy_from_host(candidate);
      // The installed basis is the CURRENT conservatively-remapped subzonal
      // mass state: invariant during the Lagrangian substep, UPDATED by
      // remaps — by definition no longer the t=0 Lagrangian invariant.
      state.corner_mass_is_lagrangian_invariant = false;
      std::ostringstream ioss;
      ioss.setf(std::ios::scientific);
      ioss.precision(6);
      ioss << "[pr4_corner_mass_install] step=" << state.step
           << " installed=1 max_cell_sum_rel_err="
           << check.max_cell_sum_rel_err
           << " min_node_mass=" << check.min_node_mass
           << " neg_clips=" << lo.negative_clip_count
           << " repair_abs=" << lo.projection_repair_abs_sum;
      core::log_info(ioss.str());
    } else {
      std::ostringstream woss;
      woss.setf(std::ios::scientific);
      woss.precision(6);
      woss << "[pr4_corner_mass_install] step=" << state.step
           << " installed=0 ABANDONED negative_corners="
           << check.negative_corners
           << " bad_sum_cells=" << check.bad_sum_cells
           << " max_cell_sum_rel_err=" << check.max_cell_sum_rel_err
           << " min_node_mass=" << check.min_node_mass;
      core::log_warning(woss.str());
    }
  }

  if (!corner_mass_remap_audit_enabled()) {
    return;
  }

  // Divergence of the transported basis from the FROZEN basis at the same
  // post-remap instant (state.corner_mass is Lagrangian-invariant, so the
  // pre-capture array is still the frozen basis now). Per-remap transport
  // starts FROM the frozen basis, so this is the per-remap staleness
  // increment, not a cumulative drift.
  double max_rel = 0.0;
  double sum_rel = 0.0;
  long long n_rel = 0;
  double frozen_mass_total = 0.0;
  double transported_mass_total = 0.0;
  for (int c = 0; c < n_cells; ++c) {
    if (have_inactive && inactive[static_cast<std::size_t>(c)] != 0U) {
      continue;
    }
    const int nv = (in.cell_nverts != nullptr &&
                    in.cell_nverts[static_cast<std::size_t>(c)] == 3U)
                       ? 3
                       : 4;
    for (int k = 0; k < nv; ++k) {
      const std::size_t idx = static_cast<std::size_t>(c) * 4U +
                              static_cast<std::size_t>(k);
      const double frozen = pre.corner_mass[idx];
      const double transported = lo.corner_mass[idx];
      frozen_mass_total += frozen;
      transported_mass_total += transported;
      if (frozen > 0.0 && std::isfinite(frozen)) {
        const double rel = std::abs(transported - frozen) / frozen;
        max_rel = std::max(max_rel, rel);
        sum_rel += rel;
        ++n_rel;
      }
    }
  }
  const double mean_rel =
      (n_rel > 0) ? sum_rel / static_cast<double>(n_rel) : 0.0;

  std::ostringstream oss;
  oss.setf(std::ios::scientific);
  oss.precision(6);
  oss << "[corner_mass_remap_audit] step=" << state.step
      << " faces=" << n_faces << " active_cells=" << lo.n_active_cells
      << " max_rel_div=" << max_rel << " mean_rel_div=" << mean_rel
      << " flux_abs=" << lo.flux_abs_sum
      << " repair_abs=" << lo.projection_repair_abs_sum << " repair_rel="
      << (frozen_mass_total > 0.0
              ? lo.projection_repair_abs_sum / frozen_mass_total
              : 0.0)
      << " neg_clips=" << lo.negative_clip_count
      << " nonpos_targets=" << lo.nonpositive_target_cells
      << " min_node_mass=" << lo.min_node_mass_sum
      << " mass_total_frozen=" << frozen_mass_total
      << " mass_total_transported=" << transported_mass_total;
  core::log_info(oss.str());
}

void gap_form_compensation_bracket(
    core::State& state,
    const std::vector<double>& corner_mass_vp_pre,
    const std::vector<double>& corner_mass_fm_pre,
    const std::vector<double>& v_r_pre,
    const std::vector<double>& v_z_pre,
    const std::vector<double>& corner_mass_fm_post,
    const std::vector<double>& cell_mass_new,
    const bool audit_only) {
  TENRYU_ASSERT(
      state.corner_stride == 4,
      "corner_stride != 4: remap-audit corner path is staged for a later revision");
  if (!state.mesh.topo.multiblock.has_value()) {
    return;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  const std::size_t n4 = static_cast<std::size_t>(n_cells) * 4U;
  if (n_cells <= 0 || n_nodes <= 0 || corner_mass_vp_pre.size() != n4 ||
      corner_mass_fm_pre.size() != n4 || corner_mass_fm_post.size() != n4 ||
      static_cast<int>(cell_mass_new.size()) != n_cells ||
      static_cast<int>(v_r_pre.size()) != n_nodes ||
      mb.cell_node_csr_indices.size() < n4) {
    return;
  }

  std::vector<double> x_r_now;
  std::vector<double> x_z_now;
  std::vector<double> v_r_now;
  std::vector<double> v_z_now;
  std::vector<double> ee;
  std::vector<double> ei;
  state.x_r.copy_to_host(x_r_now);
  state.x_z.copy_to_host(x_z_now);
  state.v_r.copy_to_host(v_r_now);
  state.v_z.copy_to_host(v_z_now);
  state.ee.copy_to_host(ee);
  state.ei.copy_to_host(ei);

  std::vector<std::uint8_t> inactive;
  if (state.central_pseudo_core.inactive_member_mask.size() ==
      static_cast<std::size_t>(n_cells)) {
    inactive = state.central_pseudo_core.inactive_member_mask;
  }
  if (state.pole_angular_derefine.inactive_member_mask.size() ==
      static_cast<std::size_t>(n_cells)) {
    if (inactive.empty()) {
      inactive.assign(static_cast<std::size_t>(n_cells), 0U);
    }
    for (int c = 0; c < n_cells; ++c) {
      if (state.pole_angular_derefine
              .inactive_member_mask[static_cast<std::size_t>(c)] != 0U) {
        inactive[static_cast<std::size_t>(c)] = 1U;
      }
    }
  }
  const bool have_inactive = static_cast<int>(inactive.size()) == n_cells;
  const std::uint8_t* nverts = state.mesh.cell_nverts.empty()
                                   ? nullptr
                                   : state.mesh.cell_nverts.data();

  // The install candidate this remap will install (same pure kernel +
  // mirror preservation as the install path).
  const VPairedCornerMassResult vp = compute_vpaired_corner_mass(
      n_cells, n_nodes, mb.cell_node_csr_offsets.data(),
      mb.cell_node_csr_indices.data(), nverts, x_r_now.data(),
      x_z_now.data(), cell_mass_new.data(), nullptr,
      have_inactive ? inactive.data() : nullptr);
  if (!vp.valid) {
    return;
  }
  std::vector<double> candidate = vp.corner_mass;
  if (have_inactive) {
    for (int c = 0; c < n_cells; ++c) {
      if (inactive[static_cast<std::size_t>(c)] != 0U) {
        for (int k = 0; k < 4; ++k) {
          const std::size_t idx = static_cast<std::size_t>(c) * 4U +
                                  static_cast<std::size_t>(k);
          candidate[idx] = corner_mass_vp_pre[idx];
        }
      }
    }
  }

  std::vector<double> u_cell(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t uc = static_cast<std::size_t>(c);
    const double m = cell_mass_new[uc];
    const double e_int = ee[uc] + ei[uc];
    u_cell[uc] = (std::isfinite(m) && m > 0.0 && std::isfinite(e_int) &&
                  e_int > 0.0)
                     ? m * e_int
                     : 0.0;
  }

  GapFormInput gin;
  gin.n_cells = n_cells;
  gin.n_nodes = n_nodes;
  gin.cell_node_csr_offsets = mb.cell_node_csr_offsets.data();
  gin.cell_node_csr_indices = mb.cell_node_csr_indices.data();
  gin.corner_mass_vp_pre = corner_mass_vp_pre.data();
  gin.corner_mass_vp_post = candidate.data();
  gin.corner_mass_fm_pre = corner_mass_fm_pre.data();
  gin.corner_mass_fm_post = corner_mass_fm_post.data();
  gin.v_r_pre = v_r_pre.data();
  gin.v_z_pre = v_z_pre.data();
  gin.v_r_post = v_r_now.data();
  gin.v_z_post = v_z_now.data();
  gin.cell_internal_energy = u_cell.data();
  gin.inactive_mask = have_inactive ? inactive.data() : nullptr;
  gin.chi = ke_fixup_deposit_chi();

  const KeFixupResult fix = compute_gap_form_deposit(gin);
  if (!fix.valid) {
    return;
  }

  bool deposited = false;
  if (!audit_only && !fix.abandoned && fix.deposited_total != 0.0) {
    for (int c = 0; c < n_cells; ++c) {
      const std::size_t uc = static_cast<std::size_t>(c);
      const double du = fix.dU[uc];
      if (du == 0.0) {
        continue;
      }
      const double m = cell_mass_new[uc];
      if (!(m > 0.0) || !std::isfinite(m)) {
        continue;
      }
      const double e_int = ee[uc] + ei[uc];
      const double ye = (e_int > 0.0 && std::isfinite(e_int))
                            ? clamp01_host(ee[uc] / e_int)
                            : 0.5;
      const double de = du / m;
      ee[uc] += ye * de;
      ei[uc] += (1.0 - ye) * de;
      deposited = true;
    }
    if (deposited) {
      state.ee.copy_from_host(ee);
      state.ei.copy_from_host(ei);
    }
  }

  std::ostringstream oss;
  oss.setf(std::ios::scientific);
  oss.precision(6);
  oss << "[gap_form_comp] step=" << state.step
      << " mode=" << (audit_only ? "audit" : "deposit")
      << " R_gap=" << fix.R_global
      << " deposited=" << (deposited ? fix.deposited_total : 0.0)
      << " clipped_cells=" << fix.clipped_cells
      << " undepositable=" << fix.undepositable_abs
      << " abandoned=" << (fix.abandoned ? 1 : 0);
  if (fix.abandoned) {
    core::log_warning(oss.str());
  } else {
    core::log_info(oss.str());
  }
}

}  // namespace tenryu::hydro::ale
