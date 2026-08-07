#include "drivers/verify_snb_2d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/conduction.cuh"
#include "hydro/conduction_snb_2d.cuh"
#include "mesh/mesh.hpp"
#include "parallel/partition.hpp"

namespace tenryu::drivers {
namespace {

// Constant spellings match src/hydro/conduction_bodies.cuh / conduction_snb_1d.cu:
// the analytic references must use bit-identical physical constants.
constexpr double kEvToErg = 1.6022e-12;
constexpr double kProtonMass = 1.6726219e-24;
constexpr double kEsuCharge4 = 4.8032e-10 * 4.8032e-10 * 4.8032e-10 * 4.8032e-10;
constexpr double kPi = 3.141592653589793238462643383279502884;

std::string fmt(const double v) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(6) << v;
  return oss.str();
}

// Normalized SNB source-spectrum cumulative (design doc §1.1) — independent
// host transcription used for the discrete-analytic dispersion reference.
double spectrum_cdf(const double b) {
  if (!(b > 0.0)) {
    return 0.0;
  }
  const double poly = ((((b + 4.0) * b + 12.0) * b + 24.0) * b + 24.0);
  const double value = 1.0 - std::exp(-b) * poly / 24.0;
  return std::min(std::max(value, 0.0), 1.0);
}

// Group mfp at energy eps [erg] (design doc §1.4); variant 0 = geometric_r2,
// 1 = original.
double lambda_g_host(const double eps_erg, const double n_e, const double zbar,
                     const double ln_lambda, const int variant) {
  const double denom = 4.0 * kPi * n_e * kEsuCharge4 * ln_lambda;
  if (variant == 1) {
    return 2.0 * eps_erg * eps_erg / (denom * std::sqrt(zbar + 1.0));
  }
  const double phi = (zbar + 4.2) / (zbar + 0.24);
  return 2.0 * std::sqrt(2.0) * eps_erg * eps_erg /
         (denom * std::sqrt(zbar * phi));
}

// Marocchino 2013 axis convention: lam_ei = 3 T^2 / (4 sqrt(2 pi) e^4 Z n lnL).
double lambda_ei_maro(const double n_e, const double te_ev, const double zbar,
                      const double ln_lambda) {
  const double T = te_ev * kEvToErg;
  return 3.0 * T * T /
         (4.0 * std::sqrt(2.0 * kPi) * kEsuCharge4 * zbar * n_e * ln_lambda);
}

// Sherlock 2017 Eq (1) thermal-scale mfp lambda_0 (geometric_r2 base).
double lambda0_host(const double n_e, const double te_ev, const double zbar,
                    const double ln_lambda) {
  const double T = te_ev * kEvToErg;
  const double phi = (zbar + 4.2) / (zbar + 0.24);
  return T * T /
         (4.0 * kPi * n_e * kEsuCharge4 * ln_lambda * std::sqrt(zbar * phi));
}

// Discrete-analytic SNB dispersion for a uniform plasma (design doc §1.7):
// R(k) = sum_g xi_g / (1 + lam_g^2 k^2 / 3) + (1 - sum_g xi_g).
double dispersion_R_disc(const double k, const double n_e, const double t0_ev,
                         const double t_ref_ev, const double zbar,
                         const double ln_lambda, const int n_groups,
                         const double e_max_over_te, const int variant) {
  const std::vector<double> beta_edges =
      tenryu::hydro::snb2d::group_edges_beta(n_groups, e_max_over_te);
  const double kT0 = kEvToErg * t0_ev;
  double sum = 0.0;
  double xi_total = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    const double e_lo = beta_edges[static_cast<std::size_t>(g)] * kEvToErg * t_ref_ev;
    const double e_hi =
        beta_edges[static_cast<std::size_t>(g) + 1] * kEvToErg * t_ref_ev;
    const double xi = spectrum_cdf(e_hi / kT0) - spectrum_cdf(e_lo / kT0);
    const double eps = 0.5 * (e_lo + e_hi);
    const double lam = lambda_g_host(eps, n_e, zbar, ln_lambda, variant);
    sum += xi / (1.0 + lam * lam * k * k / 3.0);
    xi_total += xi;
  }
  return sum + (1.0 - xi_total);
}

struct ModeSetup {
  core::Config cfg;
  double k = 0.0;
  double t0_ev = 0.0;
  double n_e = 0.0;
  double zbar = 1.0;
};

core::Config base_config_2d(const int nr, const int nz, const double L_z,
                            const double zbar, const double A) {
  core::Config cfg;
  cfg.main.name = "verify_snb_2d";
  cfg.main.dimension = "2D_RZ";
  cfg.main.dim = 2;
  cfg.main.two_temperature = true;
  cfg.main.t_end = 1.0;
  cfg.mesh.nr = nr;
  cfg.mesh.nz = nz;
  cfg.mesh.r_min = 0.0;
  cfg.mesh.r_max = L_z * static_cast<double>(nr) / static_cast<double>(nz);
  cfg.mesh.z_min = 0.0;
  cfg.mesh.z_max = L_z;
  cfg.mesh.grid_type_r = "uniform";
  cfg.mesh.grid_type_z = "uniform";
  cfg.numerics.hydro.boundary_2d.r_outer = "reflect";
  cfg.numerics.hydro.boundary_2d.z_bottom = "reflect";
  cfg.numerics.hydro.boundary_2d.z_top = "reflect";
  cfg.radiation.groups = 0;
  cfg.numerics.conduction.enabled = true;
  cfg.numerics.conduction.solver = "sts";
  cfg.numerics.conduction.nonlocal_model = "snb";
  core::Config::MaterialsConfig::MatDef mat;
  mat.name = "fuel";
  mat.A = A;
  mat.Z = zbar;
  cfg.materials.materials = {mat};
  return cfg;
}

// Builds the 2D state: uniform rho from (n_e, Z, A), Te from a z-profile.
template <typename ProfileFn>
core::State make_state_2d(const core::Config& cfg, const double n_e,
                          const double zbar, const double A,
                          ProfileFn&& te_of_z) {
  auto state = core::State::allocate(cfg);
  state.mesh = mesh::create_mesh(cfg, state);
  state.mesh.recompute_geometry();
  state.vol = state.mesh.cell_vol;

  const int nr = cfg.mesh.nr;
  const int nz = cfg.mesh.nz;
  const int n = nr * nz;
  const int nnode = (nr + 1) * (nz + 1);
  std::vector<double> xz(static_cast<std::size_t>(nnode), 0.0);
  state.x_z.copy_to_host(xz.data());
  std::vector<double> vol(static_cast<std::size_t>(n), 0.0);
  state.vol.copy_to_host(vol.data());

  const double rho_val = n_e * A * kProtonMass / zbar;
  std::vector<double> rho(static_cast<std::size_t>(n), rho_val);
  std::vector<double> zb(static_cast<std::size_t>(n), zbar);
  std::vector<double> Te(static_cast<std::size_t>(n), 0.0);
  std::vector<double> mass(static_cast<std::size_t>(n), 0.0);
  const auto node = [nz](const int i, const int j) { return i * (nz + 1) + j; };
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const std::size_t c = static_cast<std::size_t>(i * nz + j);
      const double zc = 0.25 * (xz[node(i, j)] + xz[node(i + 1, j)] +
                                xz[node(i, j + 1)] + xz[node(i + 1, j + 1)]);
      Te[c] = te_of_z(zc);
      mass[c] = rho[c] * vol[c];
    }
  }
  state.rho.copy_from_host(rho);
  state.zbar.copy_from_host(zb);
  state.Te.copy_from_host(Te);
  state.mass.copy_from_host(mass);
  return state;
}

struct ProbeRatio2d {
  double ratio = 0.0;
  double theta_min = 1.0;
  int pairs_theta_99 = 0;
  double dq_over_qsh = 0.0;
  int cg_iters = 0;
  double cg_resid = 0.0;
  bool ok = false;
};

// Flux-weighted projection over pair powers (the 1D face-flux projection
// generalized: R = sum(P_tot*P_sh)/sum(P_sh^2); pairs are double-counted
// symmetrically, which cancels in the ratio).
ProbeRatio2d probe_ratio_2d(core::State& state, const core::Config& cfg) {
  ProbeRatio2d out;
  const auto probe = hydro::snb2d::snb2d_probe(state, cfg);
  if (probe.p_sh.empty()) {
    return out;
  }
  double num = 0.0;
  double den = 0.0;
  double dq_max = 0.0;
  double sh_max = 0.0;
  for (std::size_t k = 0; k < probe.p_sh.size(); ++k) {
    const double ps = probe.p_sh[k];
    const double pt = ps + probe.p_dq[k];
    num += pt * ps;
    den += ps * ps;
    dq_max = std::max(dq_max, std::abs(probe.p_dq[k]));
    sh_max = std::max(sh_max, std::abs(ps));
  }
  for (std::size_t c = 0; c < probe.theta.size(); ++c) {
    out.theta_min = std::min(out.theta_min, probe.theta[c]);
    if (probe.theta[c] < 0.99) {
      ++out.pairs_theta_99;
    }
  }
  if (!(den > 0.0)) {
    return out;
  }
  out.ratio = num / den;
  out.dq_over_qsh = (sh_max > 0.0) ? (dq_max / sh_max) : 0.0;
  out.cg_iters = probe.cg_iters;
  out.cg_resid = probe.cg_resid;
  out.ok = true;
  return out;
}

struct ModeResult {
  double r_meas = 0.0;
  double r_disc = 0.0;
  bool ok = false;
};

// One dispersion rung: cosine mode with k*L = pi (zero-flux compatible),
// measured through the production probe; discrete-analytic reference from the
// same group edges / mfp variant.
ModeResult run_mode_case_2d(const std::string& label, const double n_e,
                            const double t0_ev, const double zbar, const double A,
                            const double k, const int nz, const int n_groups,
                            const std::string& mfp_variant) {
  ModeResult out;
  const double L = kPi / k;
  core::Config cfg = base_config_2d(4, nz, L, zbar, A);
  cfg.numerics.conduction.snb_n_groups = n_groups;
  cfg.numerics.conduction.snb_mfp = mfp_variant;
  const double dT = 1.0e-4 * t0_ev;
  auto state = make_state_2d(cfg, n_e, zbar, A,
                             [&](const double z) { return t0_ev + dT * std::cos(k * z); });

  const auto pr = probe_ratio_2d(state, cfg);
  if (!pr.ok) {
    core::log_error("[verify:" + label + "] probe produced no flux");
    return out;
  }
  const double ln_lambda = hydro::conduction::coulomb_log(n_e, t0_ev, zbar);
  const double t_ref = t0_ev + dT;  // crest = device max-Te reduction value
  const int variant = (mfp_variant == "original") ? 1 : 0;
  out.r_disc = dispersion_R_disc(k, n_e, t0_ev, t_ref, zbar, ln_lambda, n_groups,
                                 cfg.numerics.conduction.snb_E_max_over_Te, variant);
  out.r_meas = pr.ratio;
  out.ok = true;
  core::log_info("[verify:" + label + "] k=" + fmt(k) + " nz=" + std::to_string(nz) +
                 " ng=" + std::to_string(n_groups) + " variant=" + mfp_variant +
                 " R_meas=" + fmt(out.r_meas) + " R_disc=" + fmt(out.r_disc) +
                 " diff=" + fmt(out.r_meas - out.r_disc) +
                 " theta_min=" + fmt(pr.theta_min) +
                 " cg_iters=" + std::to_string(pr.cg_iters));
  return out;
}

}  // namespace

bool run_snb_local_limit_2d(const std::string& label) {
  bool pass = true;
  const double n_e = 1.0e21;
  const double t0 = 307.0;
  const double zbar = 1.0;
  const double A = 1.0;
  const double ln_lambda = hydro::conduction::coulomb_log(n_e, t0, zbar);
  const double lam0 = lambda0_host(n_e, t0, zbar, ln_lambda);

  // --- Mode ladder: k*lam0 in {1e-3, 3e-4, 1e-4} (design doc §6 G2). ---
  const std::vector<double> k_lam0 = {1.0e-3, 3.0e-4, 1.0e-4};
  std::vector<double> devs;
  for (const double t : k_lam0) {
    const double k = t / lam0;
    const auto m =
        run_mode_case_2d(label, n_e, t0, zbar, A, k, 256, 24, "geometric_r2");
    if (!m.ok) {
      pass = false;
      continue;
    }
    const double diff = std::abs(m.r_meas - m.r_disc);
    if (!(diff <= 1.0e-3)) {
      core::log_error("[verify:" + label + "] |R_meas-R_disc|=" + fmt(diff) +
                      " exceeds 1e-3 at k*lam0=" + fmt(t));
      pass = false;
    }
    devs.push_back(1.0 - m.r_meas);
  }
  // Quadratic local-limit law between the outer rungs (design doc §1.7/§7B).
  if (devs.size() == k_lam0.size() && devs.front() > 0.0 && devs.back() > 0.0) {
    const double slope = std::log(devs.front() / devs.back()) /
                         std::log(k_lam0.front() / k_lam0.back());
    core::log_info("[verify:" + label + "] local-limit slope=" + fmt(slope) +
                   " (expect ~2)");
    if (!(slope >= 1.75 && slope <= 2.25)) {
      core::log_error("[verify:" + label + "] quadratic-law slope out of [1.75,2.25]");
      pass = false;
    }
  } else {
    pass = false;
  }

  // --- Tanh ramps (mission G2 wording): SNB->SH in the collisional limit.
  // The flux-weighted projection is the judged object in 2D. Two
  // collisionality rungs verify inhibition and the quadratic law in 1/w.
  {
    const double t_top = 400.0;
    const double t_bot = 100.0;
    const double t_mid = 0.5 * (t_top + t_bot);
    const double lnl_mid = hydro::conduction::coulomb_log(n_e, t_mid, zbar);
    const double lam_ei_mid = lambda_ei_maro(n_e, t_mid, zbar, lnl_mid);
    std::vector<double> dev_rung;
    for (const double w_over_lam : {1.0e3, 1.0e4}) {
      const double w = w_over_lam * lam_ei_mid;
      const double L = 8.0 * w;
      const int nz = 512;
      core::Config cfg = base_config_2d(4, nz, L, zbar, A);
      auto state = make_state_2d(cfg, n_e, zbar, A, [&](const double z) {
        return t_mid - 0.5 * (t_top - t_bot) * std::tanh((z - 0.5 * L) / w);
      });
      const auto pr = probe_ratio_2d(state, cfg);
      if (!pr.ok) {
        core::log_error("[verify:" + label + "] ramp probe empty");
        pass = false;
        continue;
      }
      const double dev = 1.0 - pr.ratio;
      core::log_info("[verify:" + label + "] ramp w/lam_ei=" + fmt(w_over_lam) +
                     ": R_meas=" + fmt(pr.ratio) + " dev=" + fmt(dev) +
                     " theta_min=" + fmt(pr.theta_min) +
                     " cap_pairs_99=" + std::to_string(pr.pairs_theta_99) +
                     " cg_iters=" + std::to_string(pr.cg_iters));
      if (!(pr.ratio < 1.0)) {
        core::log_error("[verify:" + label +
                        "] sign sentinel: no aggregate inhibition");
        pass = false;
      }
      dev_rung.push_back(dev);
    }
    if (dev_rung.size() == 2 && dev_rung[0] > 0.0 && dev_rung[1] > 0.0) {
      const double gain = dev_rung[0] / dev_rung[1];
      core::log_info("[verify:" + label + "] ramp quadratic-law gain=" +
                     fmt(gain) + " (expect ~100)");
      if (!(gain >= 40.0 && gain <= 250.0)) {
        core::log_error("[verify:" + label + "] ramp quadratic law violated");
        pass = false;
      }
    } else {
      pass = false;
    }
  }

  if (pass) {
    core::log_info("[verify:" + label + "] PASSED");
  } else {
    core::log_error("[verify:" + label + "] FAILED");
  }
  return pass;
}

bool run_snb_dispersion_2d(const std::string& label) {
  bool pass = true;
  const double n_e = 1.0e21;
  const double t0 = 307.0;
  const double A = 1.0;

  // banded=false rungs are recorded-only: at k*lam_ei=0.5 the known SNB-family
  // 1/k^2 tail deficiency vs the kinetic 1/k (E-S 1991) dominates, so a band
  // there would certify the model's documented limitation, not this
  // implementation. Band envelope for gated rungs: paper-internal SNB-vs-VFP
  // scatter (~x1.4-1.5 in Marocchino Fig 1a) (+) digitization ~15% => x1.6.
  struct Anchor {
    double k_lam_ei;
    double r_vfp;
    bool banded;
  };
  const std::vector<Anchor> anchors_z1 = {
      {0.05, 0.72, true}, {0.1, 0.53, true}, {0.2, 0.35, true}, {0.5, 0.20, false}};
  const std::vector<Anchor> anchors_z4 = {
      {0.05, 0.55, true}, {0.1, 0.40, true}, {0.2, 0.26, true}, {0.5, 0.15, false}};

  for (const double zbar : {1.0, 4.0}) {
    const auto& anchors = (zbar == 1.0) ? anchors_z1 : anchors_z4;
    const double ln_lambda = hydro::conduction::coulomb_log(n_e, t0, zbar);
    const double lam_ei = lambda_ei_maro(n_e, t0, zbar, ln_lambda);
    double prev_R = 2.0;
    for (const auto& a : anchors) {
      const double k = a.k_lam_ei / lam_ei;
      // Tier A: production resolution + mesh ladder.
      const auto m256 =
          run_mode_case_2d(label, n_e, t0, zbar, A, k, 256, 24, "geometric_r2");
      const auto m128 =
          run_mode_case_2d(label, n_e, t0, zbar, A, k, 128, 24, "geometric_r2");
      const auto m512 =
          run_mode_case_2d(label, n_e, t0, zbar, A, k, 512, 24, "geometric_r2");
      if (!m256.ok || !m128.ok || !m512.ok) {
        pass = false;
        continue;
      }
      const double d128 = std::abs(m128.r_meas - m128.r_disc);
      const double d256 = std::abs(m256.r_meas - m256.r_disc);
      const double d512 = std::abs(m512.r_meas - m512.r_disc);
      if (!(d256 <= 3.0e-3)) {
        core::log_error("[verify:" + label + "] Tier A fail: |diff|=" + fmt(d256) +
                        " at Z=" + fmt(zbar) + " k*lam_ei=" + fmt(a.k_lam_ei));
        pass = false;
      }
      const bool ladder_ok = (d512 <= d256 * 1.05 && d256 <= d128 * 1.05) ||
                             (d512 <= 5.0e-4 && d256 <= 5.0e-4);
      if (!ladder_ok) {
        core::log_error("[verify:" + label + "] Tier A mesh ladder not monotone: " +
                        fmt(d128) + " -> " + fmt(d256) + " -> " + fmt(d512));
        pass = false;
      }
      // Group ladder (informational, logged): 48/96 groups at nz=256.
      static_cast<void>(run_mode_case_2d(label, n_e, t0, zbar, A, k, 256, 48,
                                         "geometric_r2"));
      static_cast<void>(run_mode_case_2d(label, n_e, t0, zbar, A, k, 256, 96,
                                         "geometric_r2"));
      // Variant record (informational): original mfp at production resolution.
      static_cast<void>(
          run_mode_case_2d(label, n_e, t0, zbar, A, k, 256, 24, "original"));
      // Tier B: band vs digitized VFP anchor + monotonic decrease.
      const double rel = m256.r_meas / a.r_vfp;
      core::log_info("[verify:" + label + "] TierB Z=" + fmt(zbar) + " k*lam_ei=" +
                     fmt(a.k_lam_ei) + " R_snb=" + fmt(m256.r_meas) + " R_vfp~" +
                     fmt(a.r_vfp) + " ratio=" + fmt(rel));
      if (a.banded) {
        if (!(rel >= 1.0 / 1.6 && rel <= 1.6)) {
          core::log_error("[verify:" + label + "] Tier B band exceeded");
          pass = false;
        }
      } else {
        core::log_info("[verify:" + label +
                       "] TierB recorded-only rung (tail-deficiency tier)");
      }
      if (!(m256.r_meas < prev_R)) {
        core::log_error("[verify:" + label + "] dispersion not monotone in k");
        pass = false;
      }
      prev_R = m256.r_meas;
    }
  }

  if (pass) {
    core::log_info("[verify:" + label + "] PASSED");
  } else {
    core::log_error("[verify:" + label + "] FAILED");
  }
  return pass;
}

bool run_snb_conservation_2d(const std::string& label) {
  const double n_e = 1.0e21;
  const double zbar = 1.0;
  const double A = 1.0;
  const double t_mid = 250.0;
  const double lnl = hydro::conduction::coulomb_log(n_e, t_mid, zbar);
  const double w = 1.0e2 * lambda_ei_maro(n_e, t_mid, zbar, lnl);
  const double L = 8.0 * w;

  bool pass = true;
  for (const std::string mesh_case : {"orthogonal", "distorted", "polar"}) {
    // Polar leg (2026-07-17 completion audit): curvilinear body-fit meshes
    // need more radial cells than the 4-column slab legs; existing legs keep
    // their exact dims. Logical i = radius (96 cells), j = theta over
    // [0, pi] (32 cells, pole wedges included).
    const int nr = (mesh_case == "polar") ? 96 : 4;
    const int nz = (mesh_case == "polar") ? 32 : 256;
    const int n = nr * nz;
    double e_rel_off = 0.0;
    for (const std::string model : {"none", "snb"}) {
      core::Config cfg = base_config_2d(nr, nz, L, zbar, A);
      cfg.numerics.conduction.nonlocal_model = model;
      if (mesh_case == "polar") {
        // Production polar mesh path (mesh.cu SphericalPolarHalfplane):
        // volume orientation sign -1 (theta increases with j), annular
        // center treatment (config default), explicit geometric-graded
        // radial nodes with dr(last)/dr(first) = 3 over [0.75L, 1.75L]
        // (graded-shell class; curvature ratio r_out/r_in ~ 2.33). A
        // hand-overwritten annulus on the rectangular logical mesh is NOT
        // equivalent: the +1 orientation convention rejects it (measured:
        // negative volumes, 2026-07-17).
        cfg.mesh.logical_mesh_2d = "spherical_polar_halfplane";
        cfg.mesh.spherical_polar_s_max = 1.75 * L;
        cfg.numerics.hydro.boundary_2d.r_inner = "reflect";
        const double r_in = 0.75 * L;
        const double grade = std::pow(3.0, 1.0 / static_cast<double>(nr - 1));
        const double dr0 = L * (grade - 1.0) /
                           (std::pow(grade, static_cast<double>(nr)) - 1.0);
        std::vector<double> r_node(static_cast<std::size_t>(nr) + 1, r_in);
        for (int i = 1; i <= nr; ++i) {
          r_node[static_cast<std::size_t>(i)] =
              r_node[static_cast<std::size_t>(i - 1)] +
              dr0 * std::pow(grade, static_cast<double>(i - 1));
        }
        // create_mesh checks the last node against s_max to 1e-12: pin it.
        r_node.back() = cfg.mesh.spherical_polar_s_max;
        cfg.mesh.explicit_nodes = r_node;
      }
      auto state = make_state_2d(cfg, n_e, zbar, A, [&](const double z) {
        // Polar re-seeds Te(r_sph) below; the z-profile only applies to the
        // slab legs.
        return (mesh_case == "polar")
                   ? t_mid
                   : t_mid - 150.0 * std::tanh((z - 0.5 * L) / w);
      });

      if (mesh_case == "distorted") {
        const int nnode = (nr + 1) * (nz + 1);
        std::vector<double> x_r(static_cast<std::size_t>(nnode), 0.0);
        std::vector<double> x_z(static_cast<std::size_t>(nnode), 0.0);
        state.x_r.copy_to_host(x_r.data());
        state.x_z.copy_to_host(x_z.data());
        const double hr = (cfg.mesh.r_max - cfg.mesh.r_min) / static_cast<double>(nr);
        const double hz = (cfg.mesh.z_max - cfg.mesh.z_min) / static_cast<double>(nz);
        const auto node = [nz](const int i, const int j) { return i * (nz + 1) + j; };
        std::vector<double> vol_before(static_cast<std::size_t>(n), 0.0);
        state.vol.copy_to_host(vol_before.data());
        long double vol_sum_before = 0.0L;
        for (int c = 0; c < n; ++c) {
          vol_sum_before += static_cast<long double>(vol_before[static_cast<std::size_t>(c)]);
        }
        for (int i = 1; i < nr; ++i) {
          for (int j = 1; j < nz; ++j) {
            const std::size_t v = static_cast<std::size_t>(node(i, j));
            x_r[v] += 0.15 * hr * std::sin(2.0 * kPi * static_cast<double>(j) /
                                          static_cast<double>(nz)) *
                      std::sin(kPi * static_cast<double>(i) / static_cast<double>(nr));
            x_z[v] += 0.15 * hz * std::sin(2.0 * kPi * static_cast<double>(i) /
                                          static_cast<double>(nr)) *
                      std::sin(kPi * static_cast<double>(j) / static_cast<double>(nz));
          }
        }
        state.x_r.copy_from_host(x_r);
        state.x_z.copy_from_host(x_z);
        state.mesh.recompute_geometry();
        state.vol = state.mesh.cell_vol;
        std::vector<double> rho(static_cast<std::size_t>(n), 0.0);
        std::vector<double> vol(static_cast<std::size_t>(n), 0.0);
        state.rho.copy_to_host(rho.data());
        state.vol.copy_to_host(vol.data());
        long double vol_sum_after = 0.0L;
        for (int c = 0; c < n; ++c) {
          vol_sum_after += static_cast<long double>(vol[static_cast<std::size_t>(c)]);
        }
        const double vol_sum_rel = std::abs(static_cast<double>(vol_sum_after - vol_sum_before)) /
                                   std::max(std::abs(static_cast<double>(vol_sum_before)),
                                            1.0e-300);
        core::log_info("[verify:" + label + "] mesh=" + mesh_case + " model=" + model +
                       " vol_sum_before=" + fmt(static_cast<double>(vol_sum_before)) +
                       " vol_sum_after=" + fmt(static_cast<double>(vol_sum_after)) +
                       " vol_sum_rel=" + fmt(vol_sum_rel));
        std::vector<double> mass(static_cast<std::size_t>(n), 0.0);
        for (int c = 0; c < n; ++c) {
          mass[static_cast<std::size_t>(c)] =
              rho[static_cast<std::size_t>(c)] * vol[static_cast<std::size_t>(c)];
        }
        state.mass.copy_from_host(mass);
      } else if (mesh_case == "polar") {
        // create_mesh already built the production polar annulus (validated
        // positive volumes under the polar orientation sign). Re-seed Te as
        // a radial tanh ramp in spherical radius; refresh mass from the
        // polar volumes.
        const int nnode = (nr + 1) * (nz + 1);
        std::vector<double> x_r(static_cast<std::size_t>(nnode), 0.0);
        std::vector<double> x_z(static_cast<std::size_t>(nnode), 0.0);
        state.x_r.copy_to_host(x_r.data());
        state.x_z.copy_to_host(x_z.data());
        std::vector<double> vol(static_cast<std::size_t>(n), 0.0);
        std::vector<double> rho(static_cast<std::size_t>(n), 0.0);
        state.vol.copy_to_host(vol.data());
        state.rho.copy_to_host(rho.data());
        std::vector<double> Te(static_cast<std::size_t>(n), 0.0);
        std::vector<double> mass(static_cast<std::size_t>(n), 0.0);
        const double r_in = 0.75 * L;
        const double r_mid = r_in + 0.5 * L;
        const auto node = [nz](const int i, const int j) {
          return i * (nz + 1) + j;
        };
        double vol_min = 1.0e300;
        long double vol_sum = 0.0L;
        for (int i = 0; i < nr; ++i) {
          for (int j = 0; j < nz; ++j) {
            const std::size_t c = static_cast<std::size_t>(i * nz + j);
            const double Rc =
                0.25 * (x_r[node(i, j)] + x_r[node(i + 1, j)] +
                        x_r[node(i, j + 1)] + x_r[node(i + 1, j + 1)]);
            const double Zc =
                0.25 * (x_z[node(i, j)] + x_z[node(i + 1, j)] +
                        x_z[node(i, j + 1)] + x_z[node(i + 1, j + 1)]);
            const double r_sph = std::sqrt(Rc * Rc + Zc * Zc);
            Te[c] = t_mid - 150.0 * std::tanh((r_sph - r_mid) / w);
            mass[c] = rho[c] * vol[c];
            vol_min = std::min(vol_min, vol[c]);
            vol_sum += static_cast<long double>(vol[c]);
          }
        }
        state.Te.copy_from_host(Te);
        state.mass.copy_from_host(mass);
        core::log_info("[verify:" + label + "] mesh=" + mesh_case +
                       " model=" + model + " r_in=" + fmt(r_in) +
                       " s_max=" + fmt(cfg.mesh.spherical_polar_s_max) +
                       " vol_min=" + fmt(vol_min) +
                       " vol_sum=" + fmt(static_cast<double>(vol_sum)));
        if (!(vol_min > 0.0)) {
          core::log_error("[verify:" + label +
                          "] polar mesh degenerate (vol_min<=0)");
          pass = false;
          continue;
        }
      }

      std::vector<double> vol(static_cast<std::size_t>(n), 0.0);
      state.vol.copy_to_host(vol.data());
      std::vector<double> rho(static_cast<std::size_t>(n), 0.0);
      state.rho.copy_to_host(rho.data());

      const double gamma = cfg.materials.materials.front().ideal_gas_gamma;
      const double cv_e = zbar * kEvToErg / (A * kProtonMass * (gamma - 1.0));

      std::vector<double> te0(static_cast<std::size_t>(n), 0.0);
      state.Te.copy_to_host(te0.data());
      long double e0 = 0.0L;
      for (int c = 0; c < n; ++c) {
        const std::size_t cell = static_cast<std::size_t>(c);
        e0 += static_cast<long double>(rho[cell]) * static_cast<long double>(cv_e) *
              static_cast<long double>(te0[cell]) * static_cast<long double>(vol[cell]);
      }
      core::log_info("[verify:" + label + "] mesh=" + mesh_case + " model=" + model +
                     " e0=" + fmt(static_cast<double>(e0)));

      const auto diag = hydro::compute_conduction_diagnostics(state, cfg);
      if (!(diag.dt_exp > 0.0) || !std::isfinite(diag.dt_exp)) {
        core::log_error("[verify:" + label + "] mesh=" + mesh_case + " model=" + model +
                        " no explicit dt limit");
        if (model == "snb") {
          pass = false;
        }
        continue;
      }
      // 2D dt margin: 50x dt_exp (the 1D gate's choice) leaves zero STS margin in
      // 2D (the OFF control leg already floor-clamps at 50x); the gate certifies
      // structural conservation at a CONVERGING dt (1D design section 8.8), so 2D
      // uses 10x. Measured adjudication in the session log 2026-07-11.
      // Polar (2026-07-17): the wedge cells at the theta=0/pi poles are stiffer
      // than the dt_exp diagnostic models — at 10x the run floor-clamps and the
      // Picard loop diverges; even at 1x the OFF control clamps ~20 cells/step
      // (dt_exp is marginally optimistic there; conduction-estate note). At
      // 0.1x both models are clamp-free and machine-exact (E_rel ~ 1.4e-17
      // measured), so the polar leg certifies at 0.1x per the same
      // converging-dt principle.
      double dtx = (mesh_case == "polar") ? 0.1 : 10.0;
      if (const char* env = std::getenv("TENRYU_SNB2D_G4_DTX")) { dtx = std::atof(env); }
      const double dt = dtx * diag.dt_exp;
      int clamp_total = 0;
      bool converged_all = true;
      int iters_max = 0;
      double dq_ratio_max = 0.0;
      double theta_min = 1.0;
      for (int s = 0; s < 20; ++s) {
        const auto step = hydro::conduction_step(state, dt, cfg);
        core::log_info("[verify:" + label + "] mesh=" + mesh_case +
                       " model=" + model +
                       " step=" + std::to_string(s) +
                       " iters=" + std::to_string(step.snb_picard_iters) +
                       " conv=" + std::to_string(step.snb_converged ? 1 : 0) +
                       " resid=" + fmt(step.snb_picard_resid) +
                       " dqr=" + fmt(step.snb_dq_over_qsh_max) +
                       " thmin=" + fmt(step.snb_cap_theta_min) +
                       " cg=" + std::to_string(step.snb_solver_iters) +
                       " cgres=" + fmt(step.snb_solver_resid) +
                       " clamps=" + std::to_string(step.clamp_count));
        clamp_total += step.clamp_count;
        converged_all = converged_all && step.snb_converged;
        iters_max = std::max(iters_max, step.snb_picard_iters);
        dq_ratio_max = std::max(dq_ratio_max, step.snb_dq_over_qsh_max);
        theta_min = std::min(theta_min, step.snb_cap_theta_min);
        if (s % 5 == 4) {
          std::vector<double> te_now(static_cast<std::size_t>(n), 0.0);
          state.Te.copy_to_host(te_now.data());
          long double e_now = 0.0L;
          for (int c = 0; c < n; ++c) {
            const std::size_t cell = static_cast<std::size_t>(c);
            e_now += static_cast<long double>(rho[cell]) * static_cast<long double>(cv_e) *
                     static_cast<long double>(te_now[cell]) *
                     static_cast<long double>(vol[cell]);
          }
          const double e_now_rel = std::abs(static_cast<double>(e_now - e0)) /
                                   std::max(std::abs(static_cast<double>(e0)), 1.0e-300);
          core::log_info("[verify:" + label + "] mesh=" + mesh_case + " model=" + model +
                         " step=" + std::to_string(s) +
                         " E_rel_running=" + fmt(e_now_rel));
        }
      }

      std::vector<double> te1(static_cast<std::size_t>(n), 0.0);
      state.Te.copy_to_host(te1.data());
      long double e1 = 0.0L;
      for (int c = 0; c < n; ++c) {
        const std::size_t cell = static_cast<std::size_t>(c);
        e1 += static_cast<long double>(rho[cell]) * static_cast<long double>(cv_e) *
              static_cast<long double>(te1[cell]) * static_cast<long double>(vol[cell]);
      }
      const double e_rel = std::abs(static_cast<double>(e1 - e0)) /
                           std::max(std::abs(static_cast<double>(e0)), 1.0e-300);
      if (model == "snb") {
        core::log_info("[verify:" + label + "] mesh=" + mesh_case + " model=" + model +
                       " E_rel=" + fmt(e_rel) +
                       " clamps=" + std::to_string(clamp_total) +
                       " converged=" + std::string(converged_all ? "true" : "false") +
                       " iters_max=" + std::to_string(iters_max) +
                       " dq_ratio_max=" + fmt(dq_ratio_max) +
                       " theta_min=" + fmt(theta_min));
        double e_rel_limit = 1.0e-14;
        if (mesh_case == "distorted" || mesh_case == "polar") {
          // Found-issue (2026-07-11, this gate's discovery): the pre-existing OFF
          // Kershaw 2D path leaks energy on distorted meshes (~1.3e-6/step at 15%
          // sine node displacement; step-linear; orthogonal is machine-exact).
          // SNB adds NO leak on top (measured equal-rate), so the distorted leg
          // certifies no-added-leak vs the OFF control; the OFF behavior itself is
          // recorded in the port closure notes as a conduction-estate issue.
          // The polar leg (production spherical-polar graded annulus) uses
          // the same no-added-leak criterion for the same reason.
          e_rel_limit = std::max(1.0e-14, 2.0 * e_rel_off);
        }
        if (!(e_rel <= e_rel_limit)) {
          core::log_error("[verify:" + label + "] mesh=" + mesh_case +
                          " model=" + model + " energy ledger violated");
          pass = false;
        }
        if (clamp_total != 0) {
          core::log_error("[verify:" + label + "] mesh=" + mesh_case +
                          " model=" + model + " unexpected floor clamps");
          pass = false;
        }
        if (!converged_all || iters_max < 2) {
          core::log_error("[verify:" + label + "] mesh=" + mesh_case +
                          " model=" + model +
                          " Picard machinery not engaged/converged");
          pass = false;
        }
        if (!(dq_ratio_max > 1.0e-6)) {
          core::log_error("[verify:" + label + "] mesh=" + mesh_case +
                          " model=" + model + " SNB correction inert (dq~0) — "
                          "gate would not exercise the nonlocal path");
          pass = false;
        }
      } else {
        e_rel_off = e_rel;
        core::log_info("[verify:" + label + "] mesh=" + mesh_case + " model=" + model +
                       " E_rel=" + fmt(e_rel) +
                       " clamps=" + std::to_string(clamp_total));
        if (clamp_total != 0) {
          // A clamped OFF leg books floor energy and would silently inflate
          // the no-added-leak reference e_rel_off.
          core::log_error("[verify:" + label + "] mesh=" + mesh_case +
                          " OFF control clamped — reference invalid");
          pass = false;
        }
      }
    }
  }

  if (pass) {
    core::log_info("[verify:" + label + "] PASSED");
  } else {
    core::log_error("[verify:" + label + "] FAILED");
  }
  return pass;
}

}  // namespace tenryu::drivers
