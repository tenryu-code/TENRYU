#include "drivers/verify_sn_lathrop.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/constants.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "radiation/groups.cuh"
#include "radiation/planck_table.cuh"
#include "radiation/sn_transport_1d_gpu.cuh"

namespace tenryu::drivers {
namespace {

constexpr char kLabel[] = "sn_1d_spherical_lathrop_two_region";

bool lathrop_cuda_available() {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err == cudaSuccess && device_count > 0) {
    return true;
  }
  core::log_info(std::string("[verify:") + kLabel +
                 "] SKIPPED (no CUDA device)");
  return false;
}

std::string fmt(const double value) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.17g", value);
  return std::string(buffer);
}

std::string fmte(const double value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.6e", value);
  return std::string(buffer);
}

// Literature Gauss-Legendre abscissas/weights (Abramowitz & Stegun 25.4.30),
// deliberately hardcoded — NOT recomputed with the solver's own builder — so
// a quadrature bug in the code cannot cancel out of this gate (same
// anti-cancellation convention as sn_1d_planar_slab_attenuation Part A).
// Full-range ascending sets are built from the positive half nodes.
const double kGl8Pos[4] = {0.1834346424956498,
                           0.5255324099163290,
                           0.7966664774136267,
                           0.9602898564975363};
const double kGl8W[4] = {0.3626837833783620,
                         0.3137066458778873,
                         0.2223810344533745,
                         0.1012285362903763};
const double kGl16Pos[8] = {0.0950125098376374,
                            0.2816035507792589,
                            0.4580167776572274,
                            0.6178762444026438,
                            0.7554044083550030,
                            0.8656312023878318,
                            0.9445750230732326,
                            0.9894009349916499};
const double kGl16W[8] = {0.1894506104550685,
                          0.1826034150449236,
                          0.1691565193950025,
                          0.1495959888165767,
                          0.1246289712555339,
                          0.0951585116824928,
                          0.0622535239386479,
                          0.0271524594117541};

void literature_gl_full_range(const int n_angles,
                              std::vector<double>& mu,
                              std::vector<double>& w) {
  const int half = n_angles / 2;
  const double* pos = (n_angles == 8) ? kGl8Pos : kGl16Pos;
  const double* wgt = (n_angles == 8) ? kGl8W : kGl16W;
  mu.assign(static_cast<std::size_t>(n_angles), 0.0);
  w.assign(static_cast<std::size_t>(n_angles), 0.0);
  for (int k = 0; k < half; ++k) {
    // ascending: most negative first (mirrors the solver's storage order)
    mu[static_cast<std::size_t>(k)] = -pos[half - 1 - k];
    w[static_cast<std::size_t>(k)] = wgt[half - 1 - k];
    mu[static_cast<std::size_t>(n_angles - 1 - k)] = pos[half - 1 - k];
    w[static_cast<std::size_t>(n_angles - 1 - k)] = wgt[half - 1 - k];
  }
}

struct LathropTwoRegion {
  double S1 = 0.0;
  double S2 = 0.0;
  double sigma1 = 0.0;
  double sigma2 = 0.0;
  double a = 0.0;
  double b = 0.0;
};

double clamped_sqrt(const double value) {
  return std::sqrt(std::max(value, 0.0));
}

// Eq. (3): single region radius a, constant S and sigma, vacuum incoming.
double lathrop_eq3(const double S,
                   const double sigma,
                   const double a,
                   const double r,
                   const double mu) {
  const double root = clamped_sqrt(a * a - r * r * (1.0 - mu * mu));
  return (S / sigma) * (1.0 - std::exp(-sigma * (r * mu + root)));
}

// Eq. (5a-d) with the NOTE-1 RESOLUTION: (5a) term-2 exponent is
// (sigma2-sigma1)*s - sigma1*r*mu (inner-region optical depth
// sigma1*(r*mu+s)); the printed sigma2*r*mu is discontinuous with (5c) at
// r=a and contradicts the backward-characteristic derivation. (5b), (5c),
// (5d) are as printed (5c independently re-derived term-by-term).
double lathrop_eq5b(const LathropTwoRegion& p, const double r,
                    const double mu) {
  const double t = clamped_sqrt(p.b * p.b - r * r * (1.0 - mu * mu));
  return (p.S2 / p.sigma2) * (1.0 - std::exp(-p.sigma2 * (r * mu + t)));
}

double lathrop_eq5c(const LathropTwoRegion& p, const double r,
                    const double mu) {
  const double one_m_mu2 = 1.0 - mu * mu;
  const double s = clamped_sqrt(p.a * p.a - r * r * one_m_mu2);
  const double t = clamped_sqrt(p.b * p.b - r * r * one_m_mu2);
  return (p.S1 / p.sigma1) * std::exp(p.sigma2 * s - p.sigma2 * r * mu) *
             (1.0 - std::exp(-2.0 * p.sigma1 * s)) +
         (p.S2 / p.sigma2) *
             std::exp(2.0 * (p.sigma2 - p.sigma1) * s - p.sigma2 * r * mu) *
             (std::exp(-p.sigma2 * s) - std::exp(-p.sigma2 * t)) +
         (p.S2 / p.sigma2) * (1.0 - std::exp(-p.sigma2 * (r * mu - s)));
}

double lathrop_eq5(const LathropTwoRegion& p, const double r, const double mu) {
  const double one_m_mu2 = 1.0 - mu * mu;
  const double s = clamped_sqrt(p.a * p.a - r * r * one_m_mu2);
  const double t = clamped_sqrt(p.b * p.b - r * r * one_m_mu2);
  if (r <= p.a) {
    // (5a, corrected)
    return (p.S1 / p.sigma1) *
               (1.0 - std::exp(-p.sigma1 * (r * mu + s))) +
           (p.S2 / p.sigma2) *
               std::exp((p.sigma2 - p.sigma1) * s - p.sigma1 * r * mu) *
               (std::exp(-p.sigma2 * s) - std::exp(-p.sigma2 * t));
  }
  const double mu_c = clamped_sqrt(1.0 - (p.a * p.a) / (r * r));
  if (mu <= mu_c) {
    return lathrop_eq5b(p, r, mu);
  }
  return lathrop_eq5c(p, r, mu);
}

// Part 0: the design doc's transcription-verification protocol, run
// numerically at gate startup so the NOTE-1 resolution stays enforced.
bool verify_analytic_transcription() {
  const LathropTwoRegion ref{4.0, 1.0, 1.0, 0.5, 1.0, 4.0};
  double worst = 0.0;
  // (i) continuity at r=a: outgoing mu>=0 joins (5c); incoming mu<0 joins
  // (5b). Evaluate (5a) at r=a-eps and the outer branch at r=a+eps.
  const double eps = 1.0e-9;
  for (int k = 1; k <= 40; ++k) {
    const double mu = -1.0 + 2.0 * static_cast<double>(k) / 41.0;
    const double inner = lathrop_eq5(ref, ref.a - eps, mu);
    const double outer = lathrop_eq5(ref, ref.a + eps, mu);
    worst = std::max(worst, std::abs(inner - outer));
  }
  const double scale = ref.S1 / ref.sigma1;
  if (worst > 1.0e-6 * scale) {
    core::log_info(std::string("[verify:") + kLabel +
                   "] FAILED transcription check (i): |5a-5(bc)| at r=a = " +
                   fmt(worst) + " (allowed 1e-6*scale; eps-limited)");
    return false;
  }
  // (i') continuity at mu_c for r>a: evaluated AT the kink (one-sided
  // sqrt(mu-mu_c) slope makes an eps-straddle O(sqrt(eps)); at mu_c itself
  // s is a floating-point residual and (5c) must reduce to (5b) exactly).
  double worst_muc = 0.0;
  for (const double r : {1.1, 1.5, 2.0, 3.0, 3.9}) {
    const double mu_c = clamped_sqrt(1.0 - (ref.a * ref.a) / (r * r));
    const double below = lathrop_eq5b(ref, r, mu_c);
    const double above = lathrop_eq5c(ref, r, mu_c);
    worst_muc = std::max(worst_muc, std::abs(below - above));
  }
  if (worst_muc > 1.0e-6 * scale) {
    core::log_info(std::string("[verify:") + kLabel +
                   "] FAILED transcription check (i'): |5c-5b| at mu_c = " +
                   fmt(worst_muc));
    return false;
  }
  // (ii) reduction: S1=S2=S, sigma1=sigma2=sigma -> Eq. (3) with radius b,
  // exact (not eps-limited), all three branches.
  const LathropTwoRegion red{1.0, 1.0, 1.0, 1.0, 1.0, 4.0};
  double worst_red = 0.0;
  for (int ir = 0; ir <= 20; ++ir) {
    const double r = 0.05 + 3.9 * static_cast<double>(ir) / 20.0;
    for (int k = 0; k <= 20; ++k) {
      const double mu = -1.0 + 2.0 * static_cast<double>(k) / 20.0;
      const double e5 = lathrop_eq5(red, r, mu);
      const double e3 = lathrop_eq3(1.0, 1.0, red.b, r, mu);
      worst_red = std::max(worst_red, std::abs(e5 - e3));
    }
  }
  if (worst_red > 1.0e-12) {
    core::log_info(std::string("[verify:") + kLabel +
                   "] FAILED transcription check (ii): |5-3| reduction = " +
                   fmt(worst_red));
    return false;
  }
  core::log_info(std::string("[verify:") + kLabel +
                 "] transcription protocol ok: r=a " + fmte(worst) +
                 " mu_c " + fmte(worst_muc) + " reduction " +
                 fmte(worst_red));
  return true;
}

core::Config make_lathrop_config(const int n_cells, const int n_angles) {
  core::Config cfg;
  cfg.main.dimension = "1D_SPH";
  cfg.main.dim = 1;
  cfg.mesh.nr = n_cells;
  cfg.mesh.nz = 1;
  cfg.mesh.geometry_1d = "spherical";
  cfg.radiation.mode = core::RadiationMode::SnTransport;
  cfg.radiation.groups = 1;
  cfg.radiation.group_bounds_eV = {0.0, 1.0e4};
  cfg.radiation.imc.enabled = false;
  cfg.radiation.ddmc.enabled = false;
  cfg.radiation.holo.enabled = false;
  cfg.radiation.imc.difference.enabled = false;
  cfg.radiation.sn_transport.n_angles = n_angles;
  cfg.radiation.sn_transport.spatial_scheme = "linear_characteristic";
  cfg.radiation.sn_transport.max_outer_iterations = 1;
  cfg.radiation.sn_transport.max_inner_iterations = 2;
  cfg.radiation.sn_transport.outer_tol = 1.0;
  cfg.radiation.sn_transport.inner_tol = 1.0e-12;
  cfg.radiation.sn_transport.inner_graph_unroll = 1;
  cfg.radiation.sn_transport.dsa_enabled = false;
  cfg.numerics.floors.Te = 1.0e-3;
  core::Config::MaterialsConfig::MatDef mat;
  mat.name = "constant";
  mat.A = 1.0;
  mat.Z = 1.0;
  mat.opacity_model = "constant";
  mat.kappa_a_constant = 1.0;  // sigma_a = kappa * rho => rho carries sigma(r)
  mat.kappa_s_constant = 0.0;
  mat.cv_e_override = 1.0e60;  // freeze Te across the single step
  mat.ideal_gas_gamma = 5.0 / 3.0;
  cfg.materials.materials.push_back(mat);
  return cfg;
}

struct SweptCase {
  std::vector<double> node;
  std::vector<double> r_center;
  std::vector<double> psi;      // (c*n_angles + m), G = 1
  std::vector<double> phi;      // per cell
  std::vector<double> sigma_a;  // per cell, read back from the device
  std::vector<double> sigma_s;
  int n_cells = 0;
  int n_angles = 0;
  double sigma_eff_shift = 0.0;  // 1/(c*dt) additive removal
};

// Run one frozen pure-absorber solve: two-region sigma(r) via rho(r) with
// kappa=1, per-cell emission temperature setting the per-ordinate source
// S(r) = 0.5 * c * sigma_a(r) * a_eV * T(r)^4 (G=1 => Planck fraction b=1).
bool run_swept_case(const int n_cells,
                    const int n_angles,
                    const double r_outer,
                    const double region_a,
                    const double sigma1,
                    const double sigma2,
                    const double T1,
                    const double T2,
                    SweptCase& out) {
  core::Config cfg = make_lathrop_config(n_cells, n_angles);
  const char* scheme = std::getenv("TENRYU_SN_LATHROP_DIAG_SCHEME");
  if (scheme != nullptr && scheme[0] != 0) {
    cfg.radiation.sn_transport.spatial_scheme = scheme;
  }
  core::State state = core::State::allocate(cfg);
  if (state.mesh.geometry_code != 0) {
    core::log_info(std::string("[verify:") + kLabel +
                   "] FAILED (geometry_code=" +
                   std::to_string(state.mesh.geometry_code) +
                   " want=0 — State::allocate did not bind Mesh.geometry_1d)");
    return false;
  }
  const std::size_t n = static_cast<std::size_t>(n_cells);
  out.node.assign(n + 1U, 0.0);
  const double dr = r_outer / static_cast<double>(n_cells);
  for (int i = 0; i <= n_cells; ++i) {
    out.node[static_cast<std::size_t>(i)] = dr * static_cast<double>(i);
  }
  out.r_center.assign(n, 0.0);
  std::vector<double> vol(n, 0.0);
  std::vector<double> rho(n, 0.0);
  std::vector<double> Te(n, 0.0);
  constexpr double four_pi_over_three = 4.18879020478639098462;
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t cs = static_cast<std::size_t>(c);
    const double r0 = out.node[cs];
    const double r1 = out.node[cs + 1U];
    out.r_center[cs] = 0.5 * (r0 + r1);
    vol[cs] = four_pi_over_three * (r1 * r1 * r1 - r0 * r0 * r0);
    const bool inner = out.r_center[cs] < region_a;
    rho[cs] = inner ? sigma1 : sigma2;  // kappa = 1 => sigma_a = rho
    Te[cs] = inner ? T1 : T2;
  }
  state.mesh.dim = 1;
  state.x_r.copy_from_host(out.node);
  state.vol.copy_from_host(vol);
  state.rho.copy_from_host(rho);
  state.zbar.copy_from_host(std::vector<double>(n, 1.0));
  state.Te.copy_from_host(Te);
  std::vector<double> ee(n, 0.0);
  for (std::size_t c = 0; c < n; ++c) {
    ee[c] = 1.0e60 * Te[c] / rho[c];
  }
  state.ee.copy_from_host(ee);
  state.Pe.copy_from_host(std::vector<double>(n, 1.0e8));
  state.cv_e.reset(n);
  state.cv_e.copy_from_host(std::vector<double>(n, 1.0e60));
  state.rad_E.copy_from_host(std::vector<double>(n, 0.0));

  radiation::Groups groups(cfg.radiation.group_bounds_eV);
  radiation::PlanckTable planck;
  planck.build(groups, 32, 1.0e-2, 1.0e4);
  const double dt = 1.0;
  radiation::advance_radiation_step_sn_1d(
      state, cfg, planck, cfg.materials.materials.front(), dt);

  out.n_cells = n_cells;
  out.n_angles = n_angles;
  out.sigma_eff_shift = 1.0 / (core::constants::c_light * dt);
  state.sn_psi_scratch.copy_to_host(out.psi);
  state.sn_phi_sweep.copy_to_host(out.phi);
  state.sn_sigma_a.copy_to_host(out.sigma_a);
  state.sn_sigma_s.copy_to_host(out.sigma_s);
  // Defensive: the rho-carries-sigma trick and the pure-absorber premise
  // must actually hold on the device before any comparison is trusted.
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t cs = static_cast<std::size_t>(c);
    const double want = rho[cs];
    if (std::abs(out.sigma_a[cs] - want) > 1.0e-12 * want ||
        out.sigma_s[cs] > 1.0e-30) {
      core::log_info(std::string("[verify:") + kLabel +
                     "] FAILED (device opacity mismatch at cell " +
                     std::to_string(c) + ": sigma_a=" + fmt(out.sigma_a[cs]) +
                     " want " + fmt(want) + " sigma_s=" +
                     fmt(out.sigma_s[cs]) + ")");
      return false;
    }
  }
  return true;
}

struct ZoneErrors {
  double max_rel[3] = {0.0, 0.0, 0.0};  // inner / outer below mu_c / collimated
  double sum_rel = 0.0;
  double max_phi_rel = 0.0;
  long count = 0;
  double overall_max() const {
    return std::max(max_rel[0], std::max(max_rel[1], max_rel[2]));
  }
};

// Compare the swept case against an analytic psi(r, mu) callback evaluated
// at the literature GL ordinates; phi against the same-quadrature sum.
template <typename PsiExact>
ZoneErrors compare_case(const SweptCase& sc,
                        const double region_a,
                        PsiExact psi_exact) {
  std::vector<double> mu;
  std::vector<double> w;
  literature_gl_full_range(sc.n_angles, mu, w);
  ZoneErrors ze;
  double psi_ref = 0.0;
  for (int c = 0; c < sc.n_cells; ++c) {
    for (int m = 0; m < sc.n_angles; ++m) {
      psi_ref = std::max(
          psi_ref, std::abs(psi_exact(sc.r_center[static_cast<std::size_t>(c)],
                                      mu[static_cast<std::size_t>(m)])));
    }
  }
  const double floor = 1.0e-3 * psi_ref;
  for (int c = 0; c < sc.n_cells; ++c) {
    const double r = sc.r_center[static_cast<std::size_t>(c)];
    const double mu_c =
        (r > region_a) ? clamped_sqrt(1.0 - (region_a * region_a) / (r * r))
                       : 2.0;  // inner cells: no collimation zone
    double phi_exact = 0.0;
    for (int m = 0; m < sc.n_angles; ++m) {
      const double mum = mu[static_cast<std::size_t>(m)];
      const double exact = psi_exact(r, mum);
      phi_exact += w[static_cast<std::size_t>(m)] * exact;
      const double code =
          sc.psi[static_cast<std::size_t>(c) *
                     static_cast<std::size_t>(sc.n_angles) +
                 static_cast<std::size_t>(m)];
      const double rel =
          std::abs(code - exact) / std::max(std::abs(exact), floor);
      const int zone = (r < region_a) ? 0 : ((mum < mu_c) ? 1 : 2);
      ze.max_rel[zone] = std::max(ze.max_rel[zone], rel);
      ze.sum_rel += rel;
      ++ze.count;
    }
    const double phi_code = sc.phi[static_cast<std::size_t>(c)];
    const double phi_rel = std::abs(phi_code - phi_exact) /
                           std::max(std::abs(phi_exact), floor);
    ze.max_phi_rel = std::max(ze.max_phi_rel, phi_rel);
  }
  return ze;
}

}  // namespace

bool run_sn_1d_spherical_lathrop_two_region_verify() {
  if (!lathrop_cuda_available()) {
    return true;
  }
  if (!verify_analytic_transcription()) {
    return false;
  }

  // Reference case (paper Figs. 3-4): S1=4X, S2=X via emission temperatures
  // T2 = 1 eV, T1 = 2^(1/4) eV: S_k = 0.5*c*sigma_k*a_eV*T_k^4 gives
  // S1/S2 = (sigma1/sigma2)*(T1/T2)^4 = 2*2 = 4 exactly. The analytic
  // reference uses sigma_eff = sigma_k + 1/(c*dt) as the removal (the
  // sweep's quasi-steady shift, ~3.3e-11) and the same S_k — exact
  // consistency, no approximation.
  const double sigma1 = 1.0;
  const double sigma2 = 0.5;
  const double region_a = 1.0;
  const double r_outer = 4.0;
  const double T2 = 1.0;
  const double T1 = std::pow(2.0, 0.25);
  const char* env_nr = std::getenv("TENRYU_SN_LATHROP_DIAG_NR");
  const int nr = (env_nr != nullptr && env_nr[0] != 0) ? std::atoi(env_nr)
                                                       : 400;
  const char* env_na = std::getenv("TENRYU_SN_LATHROP_DIAG_NANGLES");
  const int na = (env_na != nullptr && env_na[0] != 0) ? std::atoi(env_na)
                                                       : 16;
  if (na != 8 && na != 16) {
    core::log_info(std::string("[verify:") + kLabel +
                   "] FAILED (NANGLES must be 8 or 16 — literature GL "
                   "tables)");
    return false;
  }

  const double c_light = core::constants::c_light;
  const double a_eV = core::constants::a_eV;
  const double T14 = T1 * T1 * T1 * T1;
  const double T24 = T2 * T2 * T2 * T2;
  const double S1 = 0.5 * c_light * sigma1 * a_eV * T14;
  const double S2 = 0.5 * c_light * sigma2 * a_eV * T24;

  SweptCase s16;
  if (!run_swept_case(nr, na, r_outer, region_a, sigma1, sigma2, T1, T2,
                      s16)) {
    return false;
  }
  const LathropTwoRegion ref{S1, S2, sigma1 + s16.sigma_eff_shift,
                             sigma2 + s16.sigma_eff_shift, region_a,
                             r_outer};
  const ZoneErrors z16 = compare_case(
      s16, region_a,
      [&](const double r, const double mu) { return lathrop_eq5(ref, r, mu); });
  const double avg16 =
      (z16.count > 0) ? (z16.sum_rel / static_cast<double>(z16.count)) : 0.0;
  core::log_info(std::string("[verify:") + kLabel + "] partA S" +
                 std::to_string(na) + " nr=" + std::to_string(nr) +
                 " max_rel inner=" + fmte(z16.max_rel[0]) +
                 " outer=" + fmte(z16.max_rel[1]) +
                 " collimated=" + fmte(z16.max_rel[2]) +
                 " avg=" + fmte(avg16) +
                 " phi_max=" + fmte(z16.max_phi_rel));

  // S8 refinement companion at the same mesh (only when the main run is the
  // S16 default — a diagnostic NANGLES=8 run skips the ratio).
  double ratio = std::numeric_limits<double>::infinity();
  double ratio_avg = std::numeric_limits<double>::infinity();
  if (na == 16) {
    SweptCase s8;
    if (!run_swept_case(nr, 8, r_outer, region_a, sigma1, sigma2, T1, T2,
                        s8)) {
      return false;
    }
    const ZoneErrors z8 = compare_case(
        s8, region_a,
        [&](const double r, const double mu) { return lathrop_eq5(ref, r, mu); });
    ratio = (z16.overall_max() > 0.0) ? (z8.overall_max() / z16.overall_max())
                                      : std::numeric_limits<double>::infinity();
    const double avg8 =
        (z8.count > 0) ? (z8.sum_rel / static_cast<double>(z8.count)) : 0.0;
    ratio_avg = (avg16 > 0.0) ? (avg8 / avg16)
                              : std::numeric_limits<double>::infinity();
    core::log_info(std::string("[verify:") + kLabel +
                   "] partA S8 max_rel inner=" + fmte(z8.max_rel[0]) +
                   " outer=" + fmte(z8.max_rel[1]) +
                   " collimated=" + fmte(z8.max_rel[2]) +
                   " avg=" + fmte(avg8) +
                   " phi_max=" + fmte(z8.max_phi_rel) +
                   " ratio_max=" + fmte(ratio) +
                   " ratio_avg=" + fmte(ratio_avg));
  }

  // Part B: single-region Eq. (3), S = sigma = a = 1 scaled (T = 1 eV,
  // sigma = 1), mesh [0, 1], nr = 100.
  SweptCase sb;
  if (!run_swept_case(100, na, 1.0, 2.0 /* all cells inner */, 1.0, 1.0, T2,
                      T2, sb)) {
    return false;
  }
  const double Sb = 0.5 * c_light * 1.0 * a_eV * T24;
  const double sigma_b = 1.0 + sb.sigma_eff_shift;
  const ZoneErrors zb = compare_case(
      sb, 2.0, [&](const double r, const double mu) {
        return lathrop_eq3(Sb, sigma_b, 1.0, r, mu);
      });
  const double avgb =
      (zb.count > 0) ? (zb.sum_rel / static_cast<double>(zb.count)) : 0.0;
  core::log_info(std::string("[verify:") + kLabel + "] partB S" +
                 std::to_string(na) + " nr=100 max_rel=" +
                 fmte(zb.overall_max()) + " avg=" + fmte(avgb) +
                 " phi_max=" + fmte(zb.max_phi_rel));

  // MEASURE-THEN-FREEZE (frozen 2026-07-04 at first landing; bounds =
  // measured * 1.5 on the S16/nr=400 default). Measured: inner 8.159e-3,
  // outer 1.681e-1, collimated 1.099e-1 (the outer/collimated maxima are
  // kink-band angular errors — nr-INDEPENDENT across 400/800/1600, i.e.
  // not a spatial artifact but the documented continuous-scheme error
  // localized at the mu_c visibility kink), avg 6.498e-3, phi 1.157e-2,
  // partB 5.854e-2. Refinement: the S8->S16 AVG ratio is 2.00 and
  // nr-stable (1.999/2.005/2.006) — the clean angular-convergence witness;
  // the MAX ratio (1.24-1.32) is kink-floor-dominated and is logged as
  // informational only (criterion revision recorded in the design doc).
  const double kMaxRelBound[3] = {1.25e-2, 2.6e-1, 1.7e-1};
  const double kAvgBound = 1.0e-2;
  const double kPhiBound = 1.8e-2;
  const double kPartBMaxBound = 9.0e-2;
  const double kMinRefinementRatioAvg = 1.5;

  bool pass = true;
  for (int zone = 0; zone < 3; ++zone) {
    if (!(z16.max_rel[zone] <= kMaxRelBound[zone])) {
      pass = false;
    }
  }
  if (!(avg16 <= kAvgBound) || !(z16.max_phi_rel <= kPhiBound) ||
      !(zb.overall_max() <= kPartBMaxBound)) {
    pass = false;
  }
  if (na == 16 && !(ratio_avg >= kMinRefinementRatioAvg)) {
    pass = false;
  }
  core::log_info(std::string("[verify:") + kLabel + "] " +
                 (pass ? "PASSED" : "FAILED"));
  return pass;
}

}  // namespace tenryu::drivers
