#include "hydro/ke_fixup_deposit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include "core/error.hpp"
#include "core/state.hpp"

namespace tenryu::hydro::ale {

namespace {

inline double clamp01(const double v) {
  return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

inline bool cell_inactive(const std::uint8_t* mask, const int cell) {
  return mask != nullptr && mask[cell] != 0U;
}

void finish_node_residual_deposit(int n_cells,
                                  int n_nodes,
                                  int corner_stride,
                                  const int* cell_node_csr_offsets,
                                  const int* cell_node_csr_indices,
                                  const double* weight_corner_mass,
                                  const std::uint8_t* inactive_mask,
                                  const double* cell_internal_energy,
                                  double chi,
                                  const double* R,
                                  const double* W_active,
                                  KeFixupResult* out_ptr);

}  // namespace

KeFixupResult compute_ke_fixup_deposit(const KeFixupInput& in) {
  KeFixupResult out;
  if (in.n_cells <= 0 || in.n_nodes <= 0 || in.corner_stride <= 0 ||
      in.cell_node_csr_offsets == nullptr ||
      in.cell_node_csr_indices == nullptr ||
      in.corner_mass_frozen == nullptr || in.corner_mass_b_pre == nullptr ||
      in.corner_mass_b_post == nullptr || in.v_r_pre == nullptr ||
      in.v_z_pre == nullptr || in.v_r_post == nullptr ||
      in.v_z_post == nullptr || in.cell_internal_energy == nullptr ||
      !(in.chi > 0.0) || !std::isfinite(in.chi)) {
    return out;
  }
  const std::size_t nn = static_cast<std::size_t>(in.n_nodes);

  // Nodal sums, mirroring the basis-defect audit accumulation exactly:
  // M^F from positive finite frozen corners (member mirrors INCLUDED, as
  // the dynamics divides by them); M^{B,+/-} from finite first-moment
  // corners (inactive members are zero/excluded upstream by convention).
  std::vector<double> MF(nn, 0.0);
  std::vector<double> MBm(nn, 0.0);
  std::vector<double> MBp(nn, 0.0);
  // Active-renormalized deposit weight denominator: frozen corner masses
  // of ACTIVE cells only (macro members never receive deposits).
  std::vector<double> W_active(nn, 0.0);
  for (int c = 0; c < in.n_cells; ++c) {
    const std::size_t corner_offset =
        static_cast<std::size_t>(c) *
        static_cast<std::size_t>(in.corner_stride);
    const int off = in.cell_node_csr_offsets[c];
    const int n_corners = std::min(
        in.corner_stride, in.cell_node_csr_offsets[c + 1] - off);
    const bool inactive = cell_inactive(in.inactive_mask, c);
    for (int k = 0; k < n_corners; ++k) {
      const int n = in.cell_node_csr_indices[off + k];
      if (n < 0 || n >= in.n_nodes) {
        continue;
      }
      const double f =
          in.corner_mass_frozen[corner_offset + static_cast<std::size_t>(k)];
      const double bm =
          in.corner_mass_b_pre[corner_offset + static_cast<std::size_t>(k)];
      const double bp =
          in.corner_mass_b_post[corner_offset + static_cast<std::size_t>(k)];
      const std::size_t un = static_cast<std::size_t>(n);
      if (std::isfinite(f) && f > 0.0) {
        MF[un] += f;
        if (!inactive) {
          W_active[un] += f;
        }
      }
      if (std::isfinite(bm)) {
        MBm[un] += bm;
      }
      if (std::isfinite(bp)) {
        MBp[un] += bp;
      }
    }
  }

  std::vector<double> Mpre(nn, 0.0), Mpost(nn, 0.0);
  if (in.full_ke_residual) {
    for (int c = 0; c < in.n_cells; ++c) {
      const int off = in.cell_node_csr_offsets[c];
      int active_nverts = 4;
      if (in.cell_nverts != nullptr) {
        const int nv = static_cast<int>(in.cell_nverts[c]);
        active_nverts = nv >= 5 ? nv : 4;
      }
      const std::size_t base =
          static_cast<std::size_t>(c) *
          static_cast<std::size_t>(in.corner_stride);
      for (int q = 0; q < active_nverts && q < in.corner_stride; ++q) {
        const int n = in.cell_node_csr_indices[off + q];
        if (n < 0 || n >= in.n_nodes) continue;
        const double mp =
            in.corner_mass_b_pre[base + static_cast<std::size_t>(q)];
        const double mq =
            in.corner_mass_b_post[base + static_cast<std::size_t>(q)];
        if (std::isfinite(mp)) Mpre[static_cast<std::size_t>(n)] += mp;
        if (std::isfinite(mq)) Mpost[static_cast<std::size_t>(n)] += mq;
      }
    }
  }

  // Node residuals (reduced midpoint-convection form; see header).
  std::vector<double> R(nn, 0.0);
  long double r_global = 0.0L;
  long double undepositable = 0.0L;
  for (int n = 0; n < in.n_nodes; ++n) {
    const std::size_t un = static_cast<std::size_t>(n);
    const double vrm = in.v_r_pre[un];
    const double vzm = in.v_z_pre[un];
    const double vrp = in.v_r_post[un];
    const double vzp = in.v_z_post[un];
    if (!std::isfinite(vrm) || !std::isfinite(vzm) || !std::isfinite(vrp) ||
        !std::isfinite(vzp)) {
      continue;
    }
    double r_n;
    if (in.full_ke_residual) {
      r_n = 0.5 * (Mpost[un] * (vrp * vrp + vzp * vzp) -
                   Mpre[un] * (vrm * vrm + vzm * vzm));
    } else {
      const double dv2 =
          (vrp * vrp + vzp * vzp) - (vrm * vrm + vzm * vzm);
      const double m_mismatch = MF[un] - 0.5 * (MBp[un] + MBm[un]);
      r_n = 0.5 * m_mismatch * dv2;
    }
    if (r_n == 0.0) {
      continue;
    }
    R[un] = r_n;
    r_global += static_cast<long double>(r_n);
    if (!(W_active[un] > 0.0)) {
      undepositable += std::abs(static_cast<long double>(r_n));
    }
  }
  out.R_global = static_cast<double>(r_global);
  out.undepositable_abs = static_cast<double>(undepositable);

  if (in.full_ke_residual && in.global_mass_weighted_deposit) {
    std::vector<double> cell_mass(static_cast<std::size_t>(in.n_cells), 0.0);
    long double mass_total = 0.0L;
    for (int c = 0; c < in.n_cells; ++c) {
      if (cell_inactive(in.inactive_mask, c)) {
        continue;
      }
      const int off = in.cell_node_csr_offsets[c];
      int active_nverts = 4;
      if (in.cell_nverts != nullptr) {
        const int nv = static_cast<int>(in.cell_nverts[c]);
        active_nverts = nv >= 5 ? nv : 4;
      }
      const std::size_t base =
          static_cast<std::size_t>(c) *
          static_cast<std::size_t>(in.corner_stride);
      double mass_c = 0.0;
      for (int q = 0; q < active_nverts && q < in.corner_stride; ++q) {
        const int n = in.cell_node_csr_indices[off + q];
        if (n < 0 || n >= in.n_nodes) continue;
        const double m =
            in.corner_mass_frozen[base + static_cast<std::size_t>(q)];
        if (std::isfinite(m)) mass_c += m;
      }
      cell_mass[static_cast<std::size_t>(c)] = mass_c;
      mass_total += static_cast<long double>(mass_c);
    }

    if (!(mass_total > 0.0L)) {
      out.valid = true;
      out.abandoned = true;
      return out;
    }

    out.dU.assign(static_cast<std::size_t>(in.n_cells), 0.0);
    for (int c = 0; c < in.n_cells; ++c) {
      if (cell_inactive(in.inactive_mask, c)) {
        continue;
      }
      out.dU[static_cast<std::size_t>(c)] =
          static_cast<double>(
              -r_global * static_cast<long double>(
                              cell_mass[static_cast<std::size_t>(c)]) /
              mass_total);
    }
    out.deposited_total = -static_cast<double>(r_global);
    out.valid = true;
    out.clipped_cells = 0;
    out.redistribution_rounds = 0;
    out.undepositable_abs = 0.0;
    out.abandoned = false;
    return out;
  }

  finish_node_residual_deposit(in.n_cells, in.n_nodes, in.corner_stride,
                               in.cell_node_csr_offsets,
                               in.cell_node_csr_indices,
                               in.corner_mass_frozen, in.inactive_mask,
                               in.cell_internal_energy, in.chi, R.data(),
                               W_active.data(), &out);
  return out;
}

namespace {

// Shared deposit tail (PR3 stages 2-3): distribute -R_n over active-cell
// corner shares of the weight basis, chi-capacity clip with conservative
// same-sign redistribution, whole-deposit abandon on global shortfall or
// undepositable residual. Used by the PR3 midpoint residual and the
// gap-form residual.
void finish_node_residual_deposit(
    const int n_cells,
    const int n_nodes,
    const int corner_stride,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const double* weight_corner_mass,
    const std::uint8_t* inactive_mask,
    const double* cell_internal_energy,
    const double chi,
    const double* R,
    const double* W_active,
    KeFixupResult* out_ptr) {
  KeFixupResult& out = *out_ptr;
  // Raw deposits: dU_c = -sum_n w_cn R_n with weight-basis corner shares
  // renormalized over active cells.
  out.dU.assign(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    if (cell_inactive(inactive_mask, c)) {
      continue;
    }
    const std::size_t corner_offset =
        static_cast<std::size_t>(c) *
        static_cast<std::size_t>(corner_stride);
    const int off = cell_node_csr_offsets[c];
    const int n_corners =
        std::min(corner_stride, cell_node_csr_offsets[c + 1] - off);
    double du = 0.0;
    for (int k = 0; k < n_corners; ++k) {
      const int n = cell_node_csr_indices[off + k];
      if (n < 0 || n >= n_nodes) {
        continue;
      }
      const std::size_t un = static_cast<std::size_t>(n);
      if (R[un] == 0.0 || !(W_active[un] > 0.0)) {
        continue;
      }
      const double f =
          weight_corner_mass[corner_offset + static_cast<std::size_t>(k)];
      if (!(f > 0.0) || !std::isfinite(f)) {
        continue;
      }
      du -= (f / W_active[un]) * R[un];
    }
    out.dU[static_cast<std::size_t>(c)] = du;
  }

  // Capacity guard |dU_c| <= chi * U_c with conservative redistribution:
  // clipped excess moves to cells with remaining same-sign headroom; if
  // the pool cannot be absorbed, abandon the WHOLE deposit (never floor
  // and lose conservation).
  std::vector<double> cap(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    if (cell_inactive(inactive_mask, c)) {
      continue;
    }
    const double u = cell_internal_energy[static_cast<std::size_t>(c)];
    cap[static_cast<std::size_t>(c)] =
        (std::isfinite(u) && u > 0.0) ? chi * u : 0.0;
  }

  long double target_total = 0.0L;
  for (int c = 0; c < n_cells; ++c) {
    target_total += static_cast<long double>(out.dU[static_cast<std::size_t>(c)]);
  }

  double pool = 0.0;
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t uc = static_cast<std::size_t>(c);
    if (std::abs(out.dU[uc]) > cap[uc]) {
      const double clipped = (out.dU[uc] > 0.0) ? cap[uc] : -cap[uc];
      pool += out.dU[uc] - clipped;
      out.dU[uc] = clipped;
      ++out.clipped_cells;
    }
  }

  const double pool_tol =
      1.0e-14 * std::max(std::abs(static_cast<double>(target_total)), 1.0e-300);
  int rounds = 0;
  while (std::abs(pool) > pool_tol && rounds < 8) {
    ++rounds;
    // Headroom for absorbing pool of this sign.
    long double headroom_total = 0.0L;
    for (int c = 0; c < n_cells; ++c) {
      const std::size_t uc = static_cast<std::size_t>(c);
      const double head =
          (pool > 0.0) ? (cap[uc] - out.dU[uc]) : (cap[uc] + out.dU[uc]);
      if (head > 0.0) {
        headroom_total += static_cast<long double>(head);
      }
    }
    if (!(headroom_total > 0.0L)) {
      break;
    }
    const double scale =
        std::min(1.0, static_cast<double>(
                          std::abs(static_cast<long double>(pool)) /
                          headroom_total));
    double absorbed = 0.0;
    for (int c = 0; c < n_cells; ++c) {
      const std::size_t uc = static_cast<std::size_t>(c);
      const double head =
          (pool > 0.0) ? (cap[uc] - out.dU[uc]) : (cap[uc] + out.dU[uc]);
      if (!(head > 0.0)) {
        continue;
      }
      const double take = (pool > 0.0 ? 1.0 : -1.0) * head * scale;
      out.dU[uc] += take;
      absorbed += take;
    }
    pool -= absorbed;
  }
  out.redistribution_rounds = rounds;

  if (std::abs(pool) > pool_tol || out.undepositable_abs >
                                       1.0e-12 * std::max(std::abs(out.R_global),
                                                          1.0e-300) +
                                           1.0e-300) {
    // Capacity exhausted or residual without a receiver: abandon, deposit
    // nothing, keep conservation honest (the budget keeps the open leak
    // for THIS remap and the caller logs a warning).
    std::fill(out.dU.begin(), out.dU.end(), 0.0);
    out.deposited_total = 0.0;
    out.abandoned = true;
    out.valid = true;
    return;
  }

  long double total = 0.0L;
  for (int c = 0; c < n_cells; ++c) {
    total += static_cast<long double>(out.dU[static_cast<std::size_t>(c)]);
  }
  out.deposited_total = static_cast<double>(total);
  out.valid = true;
}

}  // namespace

int gap_form_compensation_mode() {
  static const int mode = [] {
    const char* raw = std::getenv("TENRYU_I1B_GAP_FORM_COMPENSATION");
    if (raw == nullptr || raw[0] == '\0' || raw[0] == '0') {
      return 0;
    }
    if (std::strcmp(raw, "audit") == 0 || std::strcmp(raw, "2") == 0) {
      return 2;
    }
    return 1;
  }();
  return mode;
}

KeFixupResult compute_gap_form_deposit(const GapFormInput& in) {
  KeFixupResult out;
  if (in.n_cells <= 0 || in.n_nodes <= 0 ||
      in.cell_node_csr_offsets == nullptr ||
      in.cell_node_csr_indices == nullptr ||
      in.corner_mass_vp_pre == nullptr || in.corner_mass_vp_post == nullptr ||
      in.corner_mass_fm_pre == nullptr || in.corner_mass_fm_post == nullptr ||
      in.v_r_pre == nullptr || in.v_z_pre == nullptr ||
      in.v_r_post == nullptr || in.v_z_post == nullptr ||
      in.cell_internal_energy == nullptr || !(in.chi > 0.0) ||
      !std::isfinite(in.chi)) {
    return out;
  }
  const std::size_t nn = static_cast<std::size_t>(in.n_nodes);
  std::vector<double> Mvp_pre(nn, 0.0);
  std::vector<double> Mvp_post(nn, 0.0);
  std::vector<double> Mfm_pre(nn, 0.0);
  std::vector<double> Mfm_post(nn, 0.0);
  std::vector<double> W_active(nn, 0.0);
  for (int c = 0; c < in.n_cells; ++c) {
    const std::size_t off4 = static_cast<std::size_t>(c) * 4U;
    const int off = in.cell_node_csr_offsets[c];
    const bool inactive = cell_inactive(in.inactive_mask, c);
    for (int k = 0; k < 4; ++k) {
      const int n = in.cell_node_csr_indices[off + k];
      if (n < 0 || n >= in.n_nodes) {
        continue;
      }
      const std::size_t un = static_cast<std::size_t>(n);
      const std::size_t idx = off4 + static_cast<std::size_t>(k);
      const double vpm = in.corner_mass_vp_pre[idx];
      const double vpp = in.corner_mass_vp_post[idx];
      const double fmm = in.corner_mass_fm_pre[idx];
      const double fmp = in.corner_mass_fm_post[idx];
      if (std::isfinite(vpm) && vpm > 0.0) {
        Mvp_pre[un] += vpm;
      }
      if (std::isfinite(vpp) && vpp > 0.0) {
        Mvp_post[un] += vpp;
        if (!inactive) {
          W_active[un] += vpp;
        }
      }
      if (std::isfinite(fmm)) {
        Mfm_pre[un] += fmm;
      }
      if (std::isfinite(fmp)) {
        Mfm_post[un] += fmp;
      }
    }
  }

  std::vector<double> R(nn, 0.0);
  long double r_global = 0.0L;
  long double undepositable = 0.0L;
  for (int n = 0; n < in.n_nodes; ++n) {
    const std::size_t un = static_cast<std::size_t>(n);
    const double vrm = in.v_r_pre[un];
    const double vzm = in.v_z_pre[un];
    const double vrp = in.v_r_post[un];
    const double vzp = in.v_z_post[un];
    if (!std::isfinite(vrm) || !std::isfinite(vzm) || !std::isfinite(vrp) ||
        !std::isfinite(vzp)) {
      continue;
    }
    const double g_post = Mvp_post[un] - Mfm_post[un];
    const double g_pre = Mvp_pre[un] - Mfm_pre[un];
    const double r_n = 0.5 * (g_post * (vrp * vrp + vzp * vzp) -
                              g_pre * (vrm * vrm + vzm * vzm));
    if (r_n == 0.0) {
      continue;
    }
    R[un] = r_n;
    r_global += static_cast<long double>(r_n);
    if (!(W_active[un] > 0.0)) {
      undepositable += std::abs(static_cast<long double>(r_n));
    }
  }
  out.R_global = static_cast<double>(r_global);
  out.undepositable_abs = static_cast<double>(undepositable);

  finish_node_residual_deposit(in.n_cells, in.n_nodes, 4,
                               in.cell_node_csr_offsets,
                               in.cell_node_csr_indices,
                               in.corner_mass_vp_post, in.inactive_mask,
                               in.cell_internal_energy, in.chi, R.data(),
                               W_active.data(), &out);
  return out;
}

bool ke_fixup_deposit_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_KE_FIXUP_DEPOSIT");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  return enabled;
}

double ke_fixup_deposit_chi() {
  static const double chi = [] {
    const char* raw = std::getenv("TENRYU_I1B_KE_FIXUP_DEPOSIT_CHI");
    const double v = raw != nullptr ? std::atof(raw) : 0.0;
    return (std::isfinite(v) && v > 0.0 && v <= 1.0) ? v : 0.1;
  }();
  return chi;
}

void ke_fixup_apply_deposit(core::State& state,
                            const std::vector<double>& corner_mass_b_pre,
                            const std::vector<double>& corner_mass_b_post,
                            const std::vector<double>& v_r_pre,
                            const std::vector<double>& v_z_pre,
                            const int corner_stride,
                            const double chi_override,
                            const bool full_ke_residual,
                            const bool global_deposit) {
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (!state.mesh.topo.multiblock.has_value() || n_cells <= 0 ||
      n_nodes <= 0 || corner_stride <= 0 ||
      state.corner_mass.size() !=
          static_cast<std::size_t>(n_cells) *
              static_cast<std::size_t>(corner_stride) ||
      corner_mass_b_pre.size() !=
          static_cast<std::size_t>(n_cells) *
              static_cast<std::size_t>(corner_stride) ||
      corner_mass_b_post.size() !=
          static_cast<std::size_t>(n_cells) *
              static_cast<std::size_t>(corner_stride) ||
      static_cast<int>(v_r_pre.size()) != n_nodes ||
      static_cast<int>(v_z_pre.size()) != n_nodes) {
    return;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  if (mb.cell_node_csr_indices.size() <
      static_cast<std::size_t>(n_cells) *
          static_cast<std::size_t>(corner_stride)) {
    return;
  }

  std::vector<double> v_r_post;
  std::vector<double> v_z_post;
  std::vector<double> corner_mass_frozen;
  std::vector<double> mass;
  std::vector<double> ee;
  std::vector<double> ei;
  state.v_r.copy_to_host(v_r_post);
  state.v_z.copy_to_host(v_z_post);
  state.corner_mass.copy_to_host(corner_mass_frozen);
  state.mass.copy_to_host(mass);
  state.ee.copy_to_host(ee);
  state.ei.copy_to_host(ei);

  std::vector<double> u_cell(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t uc = static_cast<std::size_t>(c);
    const double m = mass[uc];
    const double e_int = ee[uc] + ei[uc];
    u_cell[uc] = (std::isfinite(m) && m > 0.0 && std::isfinite(e_int) &&
                  e_int > 0.0)
                     ? m * e_int
                     : 0.0;
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
  const bool have_inactive = static_cast<int>(inactive.size()) == n_cells;

  KeFixupInput in;
  in.n_cells = n_cells;
  in.n_nodes = n_nodes;
  in.corner_stride = corner_stride;
  in.full_ke_residual = full_ke_residual;
  in.global_mass_weighted_deposit = global_deposit;
  in.cell_nverts = state.mesh.cell_nverts.empty()
                       ? nullptr
                       : state.mesh.cell_nverts.data();
  in.cell_node_csr_offsets = mb.cell_node_csr_offsets.data();
  in.cell_node_csr_indices = mb.cell_node_csr_indices.data();
  in.corner_mass_frozen = corner_mass_frozen.data();
  in.corner_mass_b_pre = corner_mass_b_pre.data();
  in.corner_mass_b_post = corner_mass_b_post.data();
  in.v_r_pre = v_r_pre.data();
  in.v_z_pre = v_z_pre.data();
  in.v_r_post = v_r_post.data();
  in.v_z_post = v_z_post.data();
  in.cell_internal_energy = u_cell.data();
  in.inactive_mask = have_inactive ? inactive.data() : nullptr;
  in.chi = chi_override > 0.0 ? chi_override : ke_fixup_deposit_chi();

  const KeFixupResult fix = compute_ke_fixup_deposit(in);
  if (!fix.valid) {
    return;
  }

  if (!fix.abandoned && fix.deposited_total != 0.0) {
    bool wrote = false;
    for (int c = 0; c < n_cells; ++c) {
      const std::size_t uc = static_cast<std::size_t>(c);
      const double du = fix.dU[uc];
      if (du == 0.0) {
        continue;
      }
      const double m = mass[uc];
      if (!(m > 0.0) || !std::isfinite(m)) {
        continue;
      }
      const double e_int = ee[uc] + ei[uc];
      const double ye =
          (e_int > 0.0 && std::isfinite(e_int)) ? clamp01(ee[uc] / e_int)
                                                : 0.5;
      const double de = du / m;
      ee[uc] += ye * de;
      ei[uc] += (1.0 - ye) * de;
      wrote = true;
    }
    if (wrote) {
      state.ee.copy_from_host(ee);
      state.ei.copy_from_host(ei);
    }
  }

  std::ostringstream oss;
  oss.setf(std::ios::scientific);
  oss.precision(6);
  oss << "[ke_fixup_deposit] step=" << state.step
      << " R_global=" << fix.R_global
      << " deposited=" << fix.deposited_total
      << " clipped_cells=" << fix.clipped_cells
      << " rounds=" << fix.redistribution_rounds
      << " undepositable=" << fix.undepositable_abs
      << " abandoned=" << (fix.abandoned ? 1 : 0)
      << " chi=" << in.chi;
  if (fix.abandoned) {
    core::log_warning(oss.str());
  } else {
    core::log_info(oss.str());
  }
}

}  // namespace tenryu::hydro::ale
