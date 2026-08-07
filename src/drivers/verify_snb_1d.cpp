#include "drivers/verify_snb_1d.hpp"

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
#include "hydro/conduction_snb_1d.cuh"
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
      tenryu::hydro::snb::group_edges_beta(n_groups, e_max_over_te);
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

struct ProbeRatio {
  double ratio = 0.0;         // <q_t, q_sh> / <q_sh, q_sh> (uncapped)
  double theta_min = 1.0;
  double dq_over_qsh = 0.0;
  int faces_theta_99 = 0;
  bool ok = false;
};

core::Config base_config(const std::string& geometry, const int nr, const double L,
                         const double zbar, const double A) {
  core::Config cfg;
  cfg.main.name = "verify_snb";
  cfg.main.dimension = "1D_SPH";
  cfg.main.dim = 1;
  cfg.main.two_temperature = true;
  cfg.main.t_end = 1.0;
  cfg.mesh.nr = nr;
  cfg.mesh.nz = 1;
  cfg.mesh.r_min = 0.0;
  cfg.mesh.r_max = L;
  cfg.mesh.grid_type_r = "uniform";
  cfg.mesh.geometry_1d = geometry;
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

// Builds the state with rho from (n_e, Z, A) and Te from a callable profile.
template <typename ProfileFn>
core::State make_state(const core::Config& cfg, const double n_e, const double zbar,
                       const double A, ProfileFn&& te_of_r) {
  auto state = core::State::allocate(cfg);
  state.mesh = mesh::create_mesh(cfg, state);
  state.mesh.recompute_geometry();
  state.vol = state.mesh.cell_vol;

  const int n = cfg.mesh.nr;
  std::vector<double> xr(static_cast<std::size_t>(n) + 1, 0.0);
  state.x_r.copy_to_host(xr.data());
  std::vector<double> vol(static_cast<std::size_t>(n), 0.0);
  state.vol.copy_to_host(vol.data());

  const double rho_val = n_e * A * kProtonMass / zbar;
  std::vector<double> rho(static_cast<std::size_t>(n), rho_val);
  std::vector<double> zb(static_cast<std::size_t>(n), zbar);
  std::vector<double> Te(static_cast<std::size_t>(n), 0.0);
  std::vector<double> mass(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    const std::size_t c = static_cast<std::size_t>(i);
    const double rc = 0.5 * (xr[c] + xr[c + 1]);
    Te[c] = te_of_r(rc);
    mass[c] = rho[c] * vol[c];
  }
  state.rho.copy_from_host(rho);
  state.zbar.copy_from_host(zb);
  state.Te.copy_from_host(Te);
  state.mass.copy_from_host(mass);
  return state;
}

ProbeRatio probe_ratio(core::State& state, const core::Config& cfg) {
  ProbeRatio out;
  const auto probe = hydro::conduction::snb_probe_fluxes(state, cfg);
  if (probe.q_sh_face.empty()) {
    return out;
  }
  const int nfaces = static_cast<int>(probe.q_sh_face.size());
  double num = 0.0;
  double den = 0.0;
  for (int f = 1; f + 1 < nfaces; ++f) {
    const double qs = probe.q_sh_face[static_cast<std::size_t>(f)];
    const double qt = qs + probe.dq_face[static_cast<std::size_t>(f)];
    num += qt * qs;
    den += qs * qs;
    out.theta_min =
        std::min(out.theta_min, probe.theta_face[static_cast<std::size_t>(f)]);
    if (probe.theta_face[static_cast<std::size_t>(f)] < 0.99) {
      ++out.faces_theta_99;
    }
  }
  if (!(den > 0.0)) {
    return out;
  }
  out.ratio = num / den;
  out.dq_over_qsh = probe.dq_over_qsh_max;
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
ModeResult run_mode_case(const std::string& label, const double n_e,
                         const double t0_ev, const double zbar, const double A,
                         const double k, const int nr, const int n_groups,
                         const std::string& mfp_variant) {
  ModeResult out;
  const double L = kPi / k;
  core::Config cfg = base_config("planar", nr, L, zbar, A);
  cfg.numerics.conduction.snb_n_groups = n_groups;
  cfg.numerics.conduction.snb_mfp = mfp_variant;
  const double dT = 1.0e-4 * t0_ev;
  auto state = make_state(cfg, n_e, zbar, A,
                          [&](const double r) { return t0_ev + dT * std::cos(k * r); });

  const auto pr = probe_ratio(state, cfg);
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
  core::log_info("[verify:" + label + "] k=" + fmt(k) + " nr=" + std::to_string(nr) +
                 " ng=" + std::to_string(n_groups) + " variant=" + mfp_variant +
                 " R_meas=" + fmt(out.r_meas) + " R_disc=" + fmt(out.r_disc) +
                 " diff=" + fmt(out.r_meas - out.r_disc) +
                 " theta_min=" + fmt(pr.theta_min));
  return out;
}

}  // namespace

bool run_snb_local_limit_1d(const std::string& label) {
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
    const auto m = run_mode_case(label, n_e, t0, zbar, A, k, 256, 24, "geometric_r2");
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
  // Local-limit relation (design doc §1.7 + Addendum): in the collisional
  // limit dq ~= 4480 * lam0(T)^2 * d2(q_sh)/dx2, so the derived face-local
  // deviation bound is 10 x 4480 x lam0(T_f)^2 x |d2 q_sh / q_sh|_f. Two
  // collisionality rungs verify the bound and the quadratic law in 1/w.
  {
    const double t_top = 400.0;
    const double t_bot = 100.0;
    const double t_mid = 0.5 * (t_top + t_bot);
    const double lnl_mid = hydro::conduction::coulomb_log(n_e, t_mid, zbar);
    const double lam_ei_mid = lambda_ei_maro(n_e, t_mid, zbar, lnl_mid);
    std::vector<double> worst_rel_rung;
    for (const double w_over_lam : {1.0e3, 1.0e4}) {
      const double w = w_over_lam * lam_ei_mid;
      const double L = 8.0 * w;
      const int nr = 512;
      core::Config cfg = base_config("planar", nr, L, zbar, A);
      auto state = make_state(cfg, n_e, zbar, A, [&](const double r) {
        return t_mid - 0.5 * (t_top - t_bot) * std::tanh((r - 0.5 * L) / w);
      });
      const auto probe = hydro::conduction::snb_probe_fluxes(state, cfg);
      if (probe.q_sh_face.empty()) {
        core::log_error("[verify:" + label + "] ramp probe empty");
        pass = false;
        continue;
      }
      const int nfaces = static_cast<int>(probe.q_sh_face.size());
      const double dx = L / static_cast<double>(nr);
      double qsh_max = 0.0;
      int f_peak = 1;
      for (int f = 1; f + 1 < nfaces; ++f) {
        const double aq = std::abs(probe.q_sh_face[static_cast<std::size_t>(f)]);
        if (aq > qsh_max) {
          qsh_max = aq;
          f_peak = f;
        }
      }
      std::vector<double> te_h(static_cast<std::size_t>(nr), 0.0);
      state.Te.copy_to_host(te_h.data());
      double worst = 0.0;
      double bound = 0.0;
      double theta_min = 1.0;
      int cap99 = 0;
      for (int f = 2; f + 2 < nfaces; ++f) {
        const double qs = probe.q_sh_face[static_cast<std::size_t>(f)];
        if (std::abs(qs) < 0.01 * qsh_max) {
          continue;
        }
        const double qt = qs + probe.dq_face[static_cast<std::size_t>(f)];
        worst = std::max(worst, std::abs(qt / qs - 1.0));
        const double d2q = (probe.q_sh_face[static_cast<std::size_t>(f) + 1] -
                            2.0 * qs +
                            probe.q_sh_face[static_cast<std::size_t>(f) - 1]) /
                           (dx * dx);
        const double tf = 0.5 * (te_h[static_cast<std::size_t>(f - 1)] +
                                 te_h[static_cast<std::size_t>(f)]);
        const double lnl_f = hydro::conduction::coulomb_log(n_e, tf, zbar);
        const double lam0_f = lambda0_host(n_e, tf, zbar, lnl_f);
        bound = std::max(bound, 10.0 * 4480.0 * lam0_f * lam0_f *
                                    std::abs(d2q / qs));
        theta_min =
            std::min(theta_min, probe.theta_face[static_cast<std::size_t>(f)]);
        if (probe.theta_face[static_cast<std::size_t>(f)] < 0.99) {
          ++cap99;
        }
      }
      const double q_peak_sh = probe.q_sh_face[static_cast<std::size_t>(f_peak)];
      const double q_peak_t =
          q_peak_sh + probe.dq_face[static_cast<std::size_t>(f_peak)];
      core::log_info("[verify:" + label + "] ramp w/lam_ei=" + fmt(w_over_lam) +
                     ": worst_rel=" + fmt(worst) + " bound=" + fmt(bound) +
                     " theta_min=" + fmt(theta_min) +
                     " cap_faces_99=" + std::to_string(cap99) +
                     " inhibition(peak)=" + fmt(std::abs(q_peak_t / q_peak_sh)));
      if (!(worst <= bound)) {
        core::log_error("[verify:" + label +
                        "] ramp deviation exceeds derived curvature bound");
        pass = false;
      }
      if (!(std::abs(q_peak_t) < std::abs(q_peak_sh))) {
        core::log_error("[verify:" + label +
                        "] sign sentinel: no inhibition at peak-flux face");
        pass = false;
      }
      worst_rel_rung.push_back(worst);
    }
    // Quadratic law in 1/w between the two rungs (x100 nominal; wide factor
    // covers worst-face migration across the T-dependent lam0^2 profile).
    if (worst_rel_rung.size() == 2 && worst_rel_rung[0] > 0.0 &&
        worst_rel_rung[1] > 0.0) {
      const double gain = worst_rel_rung[0] / worst_rel_rung[1];
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

bool run_snb_dispersion_1d(const std::string& label) {
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
          run_mode_case(label, n_e, t0, zbar, A, k, 256, 24, "geometric_r2");
      const auto m128 =
          run_mode_case(label, n_e, t0, zbar, A, k, 128, 24, "geometric_r2");
      const auto m512 =
          run_mode_case(label, n_e, t0, zbar, A, k, 512, 24, "geometric_r2");
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
      // Group ladder (informational, logged): 48/96 groups at nr=256.
      static_cast<void>(run_mode_case(label, n_e, t0, zbar, A, k, 256, 48,
                                      "geometric_r2"));
      static_cast<void>(run_mode_case(label, n_e, t0, zbar, A, k, 256, 96,
                                      "geometric_r2"));
      // Variant record (informational): original mfp at production resolution.
      static_cast<void>(run_mode_case(label, n_e, t0, zbar, A, k, 256, 24,
                                      "original"));
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

bool run_snb_conservation_1d(const std::string& label, const std::string& geometry) {
  const double n_e = 1.0e21;
  const double zbar = 1.0;
  const double A = 1.0;
  const double t_mid = 250.0;
  const double lnl = hydro::conduction::coulomb_log(n_e, t_mid, zbar);
  const double w = 1.0e2 * lambda_ei_maro(n_e, t_mid, zbar, lnl);
  const double L = 8.0 * w;
  const int nr = 256;

  core::Config cfg = base_config(geometry, nr, L, zbar, A);
  auto state = make_state(cfg, n_e, zbar, A, [&](const double r) {
    if (geometry == "planar") {
      return t_mid - 150.0 * std::tanh((r - 0.5 * L) / w);
    }
    const double s = r / (0.35 * L);
    return 100.0 + 300.0 * std::exp(-s * s);
  });

  const int n = nr;
  std::vector<double> xr(static_cast<std::size_t>(n) + 1, 0.0);
  state.x_r.copy_to_host(xr.data());
  std::vector<double> vol(static_cast<std::size_t>(n), 0.0);
  state.vol.copy_to_host(vol.data());
  std::vector<double> rho(static_cast<std::size_t>(n), 0.0);
  state.rho.copy_to_host(rho.data());

  const double gamma = cfg.materials.materials.front().ideal_gas_gamma;
  const double cv_e = zbar * kEvToErg / (A * kProtonMass * (gamma - 1.0));

  std::vector<double> te0(static_cast<std::size_t>(n), 0.0);
  state.Te.copy_to_host(te0.data());
  long double e0 = 0.0L;
  for (int i = 0; i < n; ++i) {
    const std::size_t c = static_cast<std::size_t>(i);
    e0 += static_cast<long double>(rho[c]) * static_cast<long double>(cv_e) *
          static_cast<long double>(te0[c]) * static_cast<long double>(vol[c]);
  }

  const auto diag = hydro::compute_conduction_diagnostics(state, cfg);
  if (!(diag.dt_exp > 0.0) || !std::isfinite(diag.dt_exp)) {
    core::log_error("[verify:" + label + "] no explicit dt limit");
    core::log_error("[verify:" + label + "] FAILED");
    return false;
  }
  const double dt = 50.0 * diag.dt_exp;
  bool pass = true;
  int clamp_total = 0;
  bool converged_all = true;
  int iters_max = 0;
  double dq_ratio_max = 0.0;
  double theta_min = 1.0;
  for (int s = 0; s < 20; ++s) {
    const auto step = hydro::conduction_step(state, dt, cfg);
    clamp_total += step.clamp_count;
    converged_all = converged_all && step.snb_converged;
    iters_max = std::max(iters_max, step.snb_picard_iters);
    dq_ratio_max = std::max(dq_ratio_max, step.snb_dq_over_qsh_max);
    theta_min = std::min(theta_min, step.snb_cap_theta_min);
  }

  std::vector<double> te1(static_cast<std::size_t>(n), 0.0);
  state.Te.copy_to_host(te1.data());
  long double e1 = 0.0L;
  for (int i = 0; i < n; ++i) {
    const std::size_t c = static_cast<std::size_t>(i);
    e1 += static_cast<long double>(rho[c]) * static_cast<long double>(cv_e) *
          static_cast<long double>(te1[c]) * static_cast<long double>(vol[c]);
  }
  const double e_rel = std::abs(static_cast<double>(e1 - e0)) /
                       std::max(std::abs(static_cast<double>(e0)), 1.0e-300);
  core::log_info("[verify:" + label + "] geometry=" + geometry + " E_rel=" +
                 fmt(e_rel) + " clamps=" + std::to_string(clamp_total) +
                 " converged=" + std::string(converged_all ? "true" : "false") +
                 " iters_max=" + std::to_string(iters_max) +
                 " dq_ratio_max=" + fmt(dq_ratio_max) +
                 " theta_min=" + fmt(theta_min));
  if (!(e_rel <= 1.0e-14)) {
    core::log_error("[verify:" + label + "] energy ledger violated");
    pass = false;
  }
  if (clamp_total != 0) {
    core::log_error("[verify:" + label + "] unexpected floor clamps");
    pass = false;
  }
  if (!converged_all || iters_max < 2) {
    core::log_error("[verify:" + label + "] Picard machinery not engaged/converged");
    pass = false;
  }
  if (!(dq_ratio_max > 1.0e-6)) {
    core::log_error("[verify:" + label + "] SNB correction inert (dq~0) — "
                    "gate would not exercise the nonlocal path");
    pass = false;
  }
  if (pass) {
    core::log_info("[verify:" + label + "] PASSED");
  } else {
    core::log_error("[verify:" + label + "] FAILED");
  }
  return pass;
}

bool run_snb_max_principle_1d(const std::string& label) {
  const double n_e = 1.0e21;
  const double zbar = 1.0;
  const double A = 1.0;
  const double t_mid = 250.0;
  const double lnl = hydro::conduction::coulomb_log(n_e, t_mid, zbar);
  const double w = 1.0e2 * lambda_ei_maro(n_e, t_mid, zbar, lnl);
  const double L = 8.0 * w;
  const int nr = 256;
  const auto te_profile = [&](const double r) {
    const double s = r / (0.35 * L);
    return 100.0 + 300.0 * std::exp(-s * s);
  };

  bool pass = true;
  {
    core::Config cfg_control = base_config("spherical", nr, L, zbar, A);
    auto state_control = make_state(cfg_control, n_e, zbar, A, te_profile);
    const auto diag_control =
        hydro::compute_conduction_diagnostics(state_control, cfg_control);
    if (!(diag_control.dt_exp > 0.0) || !std::isfinite(diag_control.dt_exp)) {
      core::log_error("[verify:" + label + "] control has no explicit dt limit");
      core::log_error("[verify:" + label + "] FAILED");
      return false;
    }
    const double dt_control = 500.0 * diag_control.dt_exp;
    int ceiling_total_control = 0;
    for (int s = 0; s < 3; ++s) {
      const auto step = hydro::conduction_step(state_control, dt_control, cfg_control);
      ceiling_total_control += step.snb_ceiling_clamp_count;
    }
    if (ceiling_total_control != 0) {
      core::log_error("[verify:" + label +
                      "] control unexpectedly engaged the ceiling guard");
      pass = false;
    }
  }

  core::Config cfg = base_config("spherical", nr, L, zbar, A);
  auto state = make_state(cfg, n_e, zbar, A, te_profile);
  const auto diag = hydro::compute_conduction_diagnostics(state, cfg);
  if (!(diag.dt_exp > 0.0) || !std::isfinite(diag.dt_exp)) {
    core::log_error("[verify:" + label + "] no explicit dt limit");
    core::log_error("[verify:" + label + "] FAILED");
    return false;
  }
  const double dt = 50.0 * diag.dt_exp;
  const double kCeilingHookEv = 200.0;  // below the 400 eV profile peak

  const int n = nr;
  std::vector<double> te0(static_cast<std::size_t>(n), 0.0);
  state.Te.copy_to_host(te0.data());
  std::vector<double> rho(static_cast<std::size_t>(n), 0.0);
  state.rho.copy_to_host(rho.data());

  std::vector<double> vol(static_cast<std::size_t>(n), 0.0);
  state.vol.copy_to_host(vol.data());
  const double gamma = cfg.materials.materials.front().ideal_gas_gamma;
  const double cv_e = zbar * kEvToErg / (A * kProtonMass * (gamma - 1.0));
  long double e0 = 0.0L;
  for (int i = 0; i < n; ++i) {
    const std::size_t c = static_cast<std::size_t>(i);
    e0 += static_cast<long double>(rho[c]) * static_cast<long double>(cv_e) *
          static_cast<long double>(te0[c]) * static_cast<long double>(vol[c]);
  }

  int ceiling_total = 0;
  double e_ceiling_sum = 0.0;
  double e_floor_sum = 0.0;
  double te_new_max_final = 0.0;
  std::vector<double> te_entry(static_cast<std::size_t>(n), 0.0);
  std::vector<double> te_new(static_cast<std::size_t>(n), 0.0);
  setenv("TENRYU_SNB_TEST_CEILING_EV", "200", 1);
  for (int s = 0; s < 6; ++s) {
    state.Te.copy_to_host(te_entry.data());
    const auto step = hydro::conduction_step(state, dt, cfg);
    ceiling_total += step.snb_ceiling_clamp_count;
    e_ceiling_sum += step.snb_E_ceiling_removed;
    e_floor_sum += step.E_floor_injected;
    state.Te.copy_to_host(te_new.data());
    const double te_new_max = *std::max_element(te_new.begin(), te_new.end());
    te_new_max_final = te_new_max;
    if (!(te_new_max <= kCeilingHookEv * (1.0 + 1.0e-12))) {
      core::log_error("[verify:" + label + "] max principle violated at step " +
                      std::to_string(s) + ": Te_new_max=" + fmt(te_new_max));
      pass = false;
    }
  }
  unsetenv("TENRYU_SNB_TEST_CEILING_EV");

  std::vector<double> rho1(static_cast<std::size_t>(n), 0.0);
  state.rho.copy_to_host(rho1.data());
  long double e1 = 0.0L;
  for (int i = 0; i < n; ++i) {
    const std::size_t c = static_cast<std::size_t>(i);
    e1 += static_cast<long double>(rho1[c]) * static_cast<long double>(cv_e) *
          static_cast<long double>(te_new[c]) * static_cast<long double>(vol[c]);
  }
  const long double ledger =
      e1 - e0 + static_cast<long double>(e_ceiling_sum) -
      static_cast<long double>(e_floor_sum);
  const long double e_rel_ld =
      std::abs(ledger) / std::max(std::abs(e0), 1.0e-300L);
  const double e_rel = static_cast<double>(e_rel_ld);

  core::log_info("[verify:" + label + "] ceiling_total=" +
                 std::to_string(ceiling_total) +
                 " e_ceiling_sum=" + fmt(e_ceiling_sum) +
                 " e_floor_sum=" + fmt(e_floor_sum) + " E_rel=" + fmt(e_rel) +
                 " ceiling_hook_eV=" + fmt(kCeilingHookEv) +
                 " Te_final_to_hook_ratio=" +
                 fmt(te_new_max_final / kCeilingHookEv));
  if (!(e_rel <= 1.0e-13)) {
    core::log_error("[verify:" + label + "] energy ledger violated");
    pass = false;
  }
  if (ceiling_total <= 0) {
    core::log_error("[verify:" + label +
                    "] ceiling guard never engaged — gate inert; retune the "
                    "receiver band");
    pass = false;
  }
  if (pass) {
    core::log_info("[verify:" + label + "] PASSED");
  } else {
    core::log_error("[verify:" + label + "] FAILED");
  }
  return pass;
}

}  // namespace tenryu::drivers
