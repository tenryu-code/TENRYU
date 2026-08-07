#include "drivers/verify_conduction_pattle_1d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <vector>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/conduction.cuh"
#include "mesh/mesh.hpp"
#include "parallel/partition.hpp"

namespace tenryu::drivers {
namespace {

// Constant spellings match src/hydro/conduction.cu (the kernel ideal-gas cv
// fallback); the analytic t0/chi must reproduce that cv bit-for-bit.
constexpr double kEvToErg = 1.6022e-12;
constexpr double kProtonMass = 1.6726219e-24;

constexpr double kPowerN = 2.5;      // kappa ~ Te^n (Spitzer-like exponent)
constexpr double kC0 = 30.0;         // [eV] central temperature at t0
constexpr double kR0 = 0.3;          // [cm] front radius at t0
constexpr double kKappa0 = 1.0e10;   // [erg/(cm s eV^{n+1})] test_kappa
constexpr double kDomain = 1.0;      // [cm]
constexpr double kTimeMultEnd = 6.0; // evolve t0 -> 6 t0
constexpr int kNumCheckpoints = 5;   // log-spaced in t
constexpr double kDtOverDtExp = 50.0;
constexpr double kFrontThresholdEv = 0.05;  // >> Te floor 1e-3, << C0

const char* kPowerEnv = "TENRYU_CONDUCTION_TEST_KAPPA_POWER";

std::string fmt(const double v) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(16) << v;
  return oss.str();
}

int geometry_dim_s(const std::string& geometry) {
  return (geometry == "cylindrical") ? 2 : ((geometry == "planar") ? 1 : 3);
}

int geometry_want_code(const std::string& geometry) {
  return (geometry == "cylindrical") ? 1 : ((geometry == "planar") ? 2 : 0);
}

struct PattleParams {
  int s = 3;
  double p = 0.0;      // s n + 2
  double cv_e = 0.0;   // [erg/(g eV)] ideal, rho = 1
  double t0 = 0.0;     // [s]
};

PattleParams make_params(const std::string& geometry) {
  PattleParams pp;
  pp.s = geometry_dim_s(geometry);
  pp.p = static_cast<double>(pp.s) * kPowerN + 2.0;
  const double gamma = 5.0 / 3.0;
  pp.cv_e = 1.0 * kEvToErg / (1.0 * kProtonMass * (gamma - 1.0));
  const double d0 = kKappa0 * std::pow(kC0, kPowerN) / (1.0 * pp.cv_e);
  pp.t0 = kR0 * kR0 * kPowerN / (2.0 * d0 * pp.p);
  return pp;
}

// Raw Pattle profile (no floor), r >= 0, t > 0.
double pattle_raw(const PattleParams& pp, const double r, const double t) {
  const double r1 = kR0 * std::pow(t / pp.t0, 1.0 / pp.p);
  const double x = 1.0 - (r * r) / (r1 * r1);
  if (x <= 0.0) {
    return 0.0;
  }
  return kC0 * std::pow(t / pp.t0, -static_cast<double>(pp.s) / pp.p) *
         std::pow(x, 1.0 / kPowerN);
}

// In-gate transcription self-check: finite-difference PDE residual of the
// closed form against dT/dt = r^{1-s} d/dr ( r^{s-1} (kappa0/(rho cv)) T^n dT/dr ),
// sampled away from the singular front edge. Catches any exponent or t0
// transcription error at O(1); tolerance is FD-limited.
bool transcription_self_check(const std::string& label, const PattleParams& pp) {
  const double d0_over_c0n = kKappa0 / (1.0 * pp.cv_e);
  double worst = 0.0;
  const double r_fracs[] = {0.15, 0.35, 0.55, 0.7};
  const double t_mults[] = {1.0, 2.5, 5.0};
  for (const double tf : t_mults) {
    const double t = pp.t0 * tf;
    const double r1 = kR0 * std::pow(t / pp.t0, 1.0 / pp.p);
    for (const double rf : r_fracs) {
      const double r = r1 * rf;
      const double hr = 1.0e-6 * r1;
      const double ht = 1.0e-6 * t;
      const auto Tof = [&](const double rr, const double tt) {
        return pattle_raw(pp, rr, tt);
      };
      const double lhs = (Tof(r, t + ht) - Tof(r, t - ht)) / (2.0 * ht);
      const auto flux = [&](const double rr) {
        const double Tm = Tof(rr - hr, t);
        const double Tp = Tof(rr + hr, t);
        const double Tc = Tof(rr, t);
        const double dTdr = (Tp - Tm) / (2.0 * hr);
        return std::pow(rr, pp.s - 1) * d0_over_c0n * std::pow(Tc, kPowerN) * dTdr;
      };
      const double rhs =
          std::pow(r, 1 - pp.s) * (flux(r + hr) - flux(r - hr)) / (2.0 * hr);
      const double scale = std::abs(Tof(r, t) / t) + 1.0e-300;
      worst = std::max(worst, std::abs(lhs - rhs) / scale);
    }
  }
  core::log_info("[verify:" + label + "] transcription FD residual worst=" + fmt(worst));
  if (!(worst <= 1.0e-4)) {
    core::log_error("[verify:" + label + "] transcription self-check FAILED");
    return false;
  }
  return true;
}

struct GateState {
  core::Config cfg;
  core::State state;
  int n = 0;
  std::vector<double> xr;
  std::vector<double> vol;
  std::vector<double> rc;

  GateState(const std::string& geometry, const int nr, const double t_end)
      : state(make_state(cfg, geometry, nr, t_end)), n(nr) {
    xr.assign(static_cast<std::size_t>(nr) + 1, 0.0);
    state.x_r.copy_to_host(xr.data());
    vol.assign(static_cast<std::size_t>(nr), 0.0);
    state.vol.copy_to_host(vol.data());
    rc.assign(static_cast<std::size_t>(nr), 0.0);
    for (int i = 0; i < nr; ++i) {
      rc[static_cast<std::size_t>(i)] =
          0.5 * (xr[static_cast<std::size_t>(i)] + xr[static_cast<std::size_t>(i) + 1]);
    }
  }

  static core::State make_state(core::Config& cfg,
                                const std::string& geometry,
                                const int nr,
                                const double t_end) {
    cfg.main.dim = 1;
    cfg.main.dimension = "1D_SPH";
    cfg.main.two_temperature = true;
    cfg.main.t_end = t_end;
    cfg.mesh.nr = nr;
    cfg.mesh.nz = 1;
    cfg.mesh.r_min = 0.0;
    cfg.mesh.r_max = kDomain;
    cfg.mesh.grid_type_r = "uniform";
    cfg.mesh.geometry_1d = geometry;
    cfg.radiation.groups = 0;
    cfg.numerics.conduction.enabled = true;
    cfg.numerics.conduction.solver = "sts";
    cfg.numerics.conduction.test_kappa = kKappa0;
    cfg.numerics.conduction.sts_damping = 0.01;
    cfg.numerics.conduction.sts_max_stages = 40;
    cfg.numerics.dt.cfl_cond = 0.25;
    core::Config::MaterialsConfig::MatDef mat;
    mat.name = "fuel";
    mat.A = 1.0;
    mat.Z = 1.0;
    cfg.materials.materials = {mat};
    auto st = core::State::allocate(cfg);
    st.mesh = mesh::create_mesh(cfg, st);
    st.mesh.recompute_geometry();
    st.vol = st.mesh.cell_vol;
    return st;
  }

  bool bind_ok(const std::string& label, const std::string& geometry) const {
    const int want = geometry_want_code(geometry);
    if (state.mesh.geometry_code != want) {
      core::log_error("[verify:" + label + "] geometry_code=" +
                      std::to_string(state.mesh.geometry_code) + " want=" +
                      std::to_string(want) +
                      " (State::allocate/create_mesh did not bind Mesh.geometry_1d)");
      return false;
    }
    return true;
  }

  void fill_fields(const std::vector<double>& Te) {
    std::vector<double> rho(static_cast<std::size_t>(n), 1.0);
    std::vector<double> zbar(static_cast<std::size_t>(n), 1.0);
    std::vector<double> mass(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
      mass[static_cast<std::size_t>(i)] = rho[static_cast<std::size_t>(i)] *
                                          vol[static_cast<std::size_t>(i)];
    }
    state.rho.copy_from_host(rho);
    state.zbar.copy_from_host(zbar);
    state.Te.copy_from_host(Te);
    state.mass.copy_from_host(mass);
  }

  long double te_vol_sum() const {
    std::vector<double> Te(static_cast<std::size_t>(n), 0.0);
    state.Te.copy_to_host(Te.data());
    long double acc = 0.0L;
    for (int i = 0; i < n; ++i) {
      acc += static_cast<long double>(Te[static_cast<std::size_t>(i)]) *
             static_cast<long double>(vol[static_cast<std::size_t>(i)]);
    }
    return acc;
  }
};

double front_from_profile(const std::vector<double>& Te,
                          const std::vector<double>& rc,
                          const int n) {
  for (int i = n - 1; i >= 0; --i) {
    if (Te[static_cast<std::size_t>(i)] > kFrontThresholdEv) {
      if (i == n - 1) {
        return -1.0;  // front reached the wall — invalid
      }
      const double te_in = Te[static_cast<std::size_t>(i)];
      const double te_out = Te[static_cast<std::size_t>(i) + 1];
      const double w = (te_in - kFrontThresholdEv) /
                       std::max(te_in - te_out, 1.0e-300);
      return rc[static_cast<std::size_t>(i)] +
             w * (rc[static_cast<std::size_t>(i) + 1] - rc[static_cast<std::size_t>(i)]);
    }
  }
  return -1.0;
}

double fit_slope(const std::vector<double>& x, const std::vector<double>& y) {
  double sx = 0.0;
  double sy = 0.0;
  double sxx = 0.0;
  double sxy = 0.0;
  const double m = static_cast<double>(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    sx += x[i];
    sy += y[i];
    sxx += x[i] * x[i];
    sxy += x[i] * y[i];
  }
  return (m * sxy - sx * sy) / (m * sxx - sx * sx);
}

struct CaseMetrics {
  bool ok = false;
  double l2_rel = 0.0;
  double cons_rel = 0.0;
  double front_slope = 0.0;
  double front_slope_relerr = 0.0;
  int clamp_total = 0;
};

CaseMetrics run_case(const std::string& label,
                     const std::string& geometry,
                     const PattleParams& pp,
                     const int nr) {
  CaseMetrics out;
  const double t1 = pp.t0 * kTimeMultEnd;
  GateState g(geometry, nr, t1);
  if (!g.bind_ok(label, geometry)) {
    return out;
  }
  const double te_floor = g.cfg.numerics.floors.Te;

  std::vector<double> Te0(static_cast<std::size_t>(nr), 0.0);
  for (int i = 0; i < nr; ++i) {
    Te0[static_cast<std::size_t>(i)] =
        std::max(pattle_raw(pp, g.rc[static_cast<std::size_t>(i)], pp.t0), te_floor);
  }
  g.fill_fields(Te0);

  const long double e_initial = g.te_vol_sum();

  const auto diag0 = hydro::compute_conduction_diagnostics(g.state, g.cfg);
  double dt_exp_cur = diag0.dt_exp;
  if (!(dt_exp_cur > 0.0) || !std::isfinite(dt_exp_cur)) {
    core::log_error("[verify:" + label + "] non-finite initial dt_exp");
    return out;
  }

  std::vector<double> ln_t;
  std::vector<double> ln_rf;
  std::vector<double> Te_host(static_cast<std::size_t>(nr), 0.0);
  double t = pp.t0;
  for (int j = 1; j <= kNumCheckpoints; ++j) {
    const double t_target =
        pp.t0 * std::pow(kTimeMultEnd,
                         static_cast<double>(j) / static_cast<double>(kNumCheckpoints));
    while (t < t_target) {
      const double dt = std::min(kDtOverDtExp * dt_exp_cur, t_target - t);
      const auto step = hydro::conduction_step(g.state, dt, g.cfg);
      out.clamp_total += step.clamp_count;
      if (step.dt_exp > 0.0 && std::isfinite(step.dt_exp)) {
        dt_exp_cur = step.dt_exp;
      }
      t += dt;
    }
    g.state.Te.copy_to_host(Te_host.data());
    const double rf = front_from_profile(Te_host, g.rc, nr);
    if (!(rf > 0.0)) {
      core::log_error("[verify:" + label + "] front detection failed at checkpoint " +
                      std::to_string(j));
      return out;
    }
    const double rf_theory =
        kR0 * std::pow(t_target / pp.t0, 1.0 / pp.p);
    core::log_info("[verify:" + label + "] cp" + std::to_string(j) + " nr=" +
                   std::to_string(nr) + " t=" + fmt(t) + " dt_exp=" +
                   fmt(dt_exp_cur) + " Te_center=" + fmt(Te_host[0]) +
                   " r_f=" + fmt(rf) + " r_f_theory=" + fmt(rf_theory));
    ln_t.push_back(std::log(t_target));
    ln_rf.push_back(std::log(rf));
  }

  g.state.Te.copy_to_host(Te_host.data());
  long double accum = 0.0L;
  for (int i = 0; i < nr; ++i) {
    const double ref =
        std::max(pattle_raw(pp, g.rc[static_cast<std::size_t>(i)], t1), te_floor);
    const double diff = Te_host[static_cast<std::size_t>(i)] - ref;
    accum += static_cast<long double>(diff) * static_cast<long double>(diff);
  }
  const double l2 =
      std::sqrt(static_cast<double>(accum / static_cast<long double>(nr)));
  const double center_amp =
      kC0 * std::pow(kTimeMultEnd, -static_cast<double>(pp.s) / pp.p);
  out.l2_rel = l2 / center_amp;
  core::log_info("[verify:" + label + "] final nr=" + std::to_string(nr) +
                 " Te_center_sim=" + fmt(Te_host[0]) + " Te_center_ref=" +
                 fmt(std::max(pattle_raw(pp, g.rc[0], t1),
                              g.cfg.numerics.floors.Te)) +
                 " Te_mid_sim=" + fmt(Te_host[static_cast<std::size_t>(nr / 8)]) +
                 " Te_mid_ref=" +
                 fmt(std::max(pattle_raw(pp, g.rc[static_cast<std::size_t>(nr / 8)], t1),
                              g.cfg.numerics.floors.Te)));

  const long double e_final = g.te_vol_sum();
  out.cons_rel = std::abs(static_cast<double>(e_final - e_initial)) /
                 std::max(std::abs(static_cast<double>(e_initial)), 1.0e-300);

  out.front_slope = fit_slope(ln_t, ln_rf);
  const double beta = 1.0 / pp.p;
  out.front_slope_relerr = std::abs(out.front_slope - beta) / beta;
  out.ok = true;
  core::log_info("[verify:" + label + "] nr=" + std::to_string(nr) + " geometry=" +
                 geometry + " L2_rel=" + fmt(out.l2_rel) + " cons_rel=" +
                 fmt(out.cons_rel) + " front_slope=" + fmt(out.front_slope) +
                 " beta_theory=" + fmt(beta) + " slope_relerr=" +
                 fmt(out.front_slope_relerr) + " clamps=" +
                 std::to_string(out.clamp_total));
  return out;
}

}  // namespace

bool run_conduction_pattle_1d(const std::string& label, const std::string& geometry) {
  const PattleParams pp = make_params(geometry);
  core::log_info("[verify:" + label + "] s=" + std::to_string(pp.s) + " n=" +
                 fmt(kPowerN) + " t0=" + fmt(pp.t0) + " beta=" + fmt(1.0 / pp.p));
  if (!transcription_self_check(label, pp)) {
    core::log_error("[verify:" + label + "] FAILED");
    return false;
  }
  setenv(kPowerEnv, "2.5", 1);
  const auto m200 = run_case(label, geometry, pp, 200);
  const auto m400 = run_case(label, geometry, pp, 400);
  unsetenv(kPowerEnv);
  if (!m200.ok || !m400.ok) {
    core::log_error("[verify:" + label + "] FAILED");
    return false;
  }
  const double order = std::log2(m200.l2_rel / m400.l2_rel);
  // Frozen gate bounds (measured 2026-07-04 with the Kirchhoff-secant stage
  // kernel: L2_rel(400) = 8.0e-4 sph / 6.3e-4 cyl / 8.8e-4 planar;
  // slope_relerr(400) = 1.5% / 0.96% / 0.91%; order = 2.28 / 2.16 / 1.45;
  // cons_rel <= 7.7e-16; clamps 0. Order bound spans the planar 1.45 —
  // degenerate-front local error reduces the formal order; front and L2
  // bounds carry the discriminating power).
  const bool pass_cons = (m200.cons_rel <= 1.0e-13) && (m400.cons_rel <= 1.0e-13);
  const bool pass_clamp = (m200.clamp_total == 0) && (m400.clamp_total == 0);
  const bool pass_front = (m400.front_slope_relerr <= 0.05);
  const bool pass_l2 = (m400.l2_rel <= 2.0e-3);
  const bool pass_order = (order >= 1.2 && order <= 2.6);
  const bool pass = pass_cons && pass_clamp && pass_front && pass_l2 && pass_order;
  core::log_info("[verify:" + label + "] order=" + fmt(order) + " cons_pass=" +
                 std::string(pass_cons ? "true" : "false") + " clamp_pass=" +
                 std::string(pass_clamp ? "true" : "false") + " front_pass=" +
                 std::string(pass_front ? "true" : "false") + " l2_pass=" +
                 std::string(pass_l2 ? "true" : "false") + " order_pass=" +
                 std::string(pass_order ? "true" : "false"));
  if (!pass) {
    core::log_error("[verify:" + label + "] FAILED");
  } else {
    core::log_info("[verify:" + label + "] PASSED");
  }
  return pass;
}

bool run_conduction_pattle_kappa_power_zero_connection(const std::string& label) {
  const PattleParams pp = make_params("spherical");
  const int nr = 100;
  const int n_steps = 20;
  // Smooth constant-kappa problem: the raw Pattle profile at t0 is just a
  // convenient smooth IC here; with the hook at power 0 the conductivity is
  // the constant test_kappa either way, so the two runs must agree bitwise.
  std::vector<double> te_a;
  std::vector<double> te_b;
  for (int variant = 0; variant < 2; ++variant) {
    GateState g("spherical", nr, 1.0);
    if (!g.bind_ok(label, "spherical")) {
      return false;
    }
    const double te_floor = g.cfg.numerics.floors.Te;
    std::vector<double> Te0(static_cast<std::size_t>(nr), 0.0);
    for (int i = 0; i < nr; ++i) {
      Te0[static_cast<std::size_t>(i)] =
          std::max(pattle_raw(pp, g.rc[static_cast<std::size_t>(i)], pp.t0), te_floor);
    }
    g.fill_fields(Te0);
    if (variant == 1) {
      setenv(kPowerEnv, "0", 1);
    }
    const auto diag0 = hydro::compute_conduction_diagnostics(g.state, g.cfg);
    const double dt = 10.0 * diag0.dt_exp;
    for (int k = 0; k < n_steps; ++k) {
      (void)hydro::conduction_step(g.state, dt, g.cfg);
    }
    if (variant == 1) {
      unsetenv(kPowerEnv);
    }
    std::vector<double> Te1(static_cast<std::size_t>(nr), 0.0);
    g.state.Te.copy_to_host(Te1.data());
    if (variant == 0) {
      te_a = Te1;
    } else {
      te_b = Te1;
    }
  }
  double max_abs = 0.0;
  for (int i = 0; i < nr; ++i) {
    max_abs = std::max(max_abs, std::abs(te_a[static_cast<std::size_t>(i)] -
                                         te_b[static_cast<std::size_t>(i)]));
  }
  const bool pass = (max_abs == 0.0);
  core::log_info("[verify:" + label + "] hook-off vs power=0 max|dTe|=" + fmt(max_abs) +
                 (pass ? " (bitwise identical)" : ""));
  if (!pass) {
    core::log_error("[verify:" + label + "] FAILED");
  } else {
    core::log_info("[verify:" + label + "] PASSED");
  }
  return pass;
}

}  // namespace tenryu::drivers
