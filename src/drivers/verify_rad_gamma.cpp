// W-K gates.
// A (adiabat): cold homologous coasting, zero opacity, reflective rad
//   boundaries, gate-seeded uniform E_r0. Coupling ON: per-cell
//   E_g V^{4/3} invariant (law). Coupling OFF: frozen density means
//   E_g V^{4/3} grows by (V1/V0)^{1/3} — asserted too, so a silently
//   inert coupling cannot fake a PASS.
// B (c_eff): standing-wave modal frequency ratio ON/OFF equals
//   sqrt(1 + (4/9) E_r0 / (gamma_g p_g)).

#include "drivers/verify_rad_gamma.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "coupling/driver.hpp"
#include "diagnostics/energy_budget.hpp"

#if defined(TENRYU_ENABLE_PYTHON) && TENRYU_ENABLE_PYTHON
#include <pybind11/embed.h>

#include "core/namelist/geometry_eval.hpp"
#include "core/namelist/runtime.hpp"
#include "mesh/mesh.hpp"
#endif

namespace tenryu::drivers {

namespace {

std::string rg_fmt(const double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.6e", v);
  return std::string(buf);
}

bool rg_cuda_available(const char* label) {
  int n = 0;
  const cudaError_t err = cudaGetDeviceCount(&n);
  if (err != cudaSuccess || n <= 0) {
    core::log_info(std::string("[SKIP] ") + label + ": CUDA not available");
    return false;
  }
  return true;
}

void rg_set_coupling_env(const bool on) {
  // Both branches run with the radiation SOLVE skipped (diag hook): the
  // field must evolve under the coupling alone (ON) or stay frozen (OFF).
  setenv("TENRYU_RAD_DIAG_SKIP_SOLVE", "1", 1);
  if (on) {
    setenv("TENRYU_RAD_HYDRO_COUPLING", "gamma_r_43", 1);
  } else {
    unsetenv("TENRYU_RAD_HYDRO_COUPLING");
  }
}

#if defined(TENRYU_ENABLE_PYTHON) && TENRYU_ENABLE_PYTHON

void rg_ensure_python() {
  if (!Py_IsInitialized()) {
    pybind11::initialize_interpreter();
  }
}

core::State rg_load_state(const std::string& deck, core::Config& cfg) {
  core::State state;
  core::namelist::Runtime runtime;
  runtime.execute(deck);
  cfg = runtime.config();
  state = core::State::allocate(cfg, runtime.builder().hydro_t_start_eV);
  state.mesh = mesh::create_mesh(cfg, state);
  state.vol = state.mesh.cell_vol;
  core::namelist::evaluate_geometry(cfg, runtime.builder(), state);
  state.t_next_plot = -1.0;
  state.t_next_history = -1.0;
  state.t_next_checkpoint = -1.0;
  return state;
}

std::vector<double> rg_to_host(const double* d, const std::size_t n) {
  std::vector<double> h(n, 0.0);
  if (d != nullptr && n > 0) {
    cudaMemcpy(h.data(), d, n * sizeof(double), cudaMemcpyDeviceToHost);
  }
  return h;
}

void rg_seed_uniform_rad(core::State& state, const double e_r0) {
  const std::size_t n = state.rad_E.size();
  if (n == 0 || !(e_r0 > 0.0)) {
    return;
  }
  std::vector<double> h(n, e_r0);
  cudaMemcpy(state.rad_E.data(), h.data(), n * sizeof(double),
             cudaMemcpyHostToDevice);
}

struct RgRunOut {
  std::vector<double> rad_E;
  std::vector<double> vol;
  std::vector<double> u;
  std::vector<double> x;
  double t = 0.0;
  double E_tot_init = 0.0;   // matter(thermal+kinetic) + field, post-seed
  double E_tot_end = 0.0;    // same, after the run
  bool ok = false;
};

RgRunOut rg_run(const std::string& deck, const bool coupling_on,
                const double e_r0, const double t_end, const int want_geom,
                const char* label) {
  RgRunOut out;
  rg_set_coupling_env(coupling_on);
  core::Config cfg;
  core::State state = rg_load_state(deck, cfg);
  if (state.mesh.geometry_code != want_geom) {
    core::log_error(std::string(label) + " geometry_code=" +
                    std::to_string(state.mesh.geometry_code) +
                    " want=" + std::to_string(want_geom));
    rg_set_coupling_env(false);
    unsetenv("TENRYU_RAD_DIAG_SKIP_SOLVE");
    return out;
  }
  rg_seed_uniform_rad(state, e_r0);
  {
    const auto tot0 = diagnostics::compute_energy_totals_1d(state);
    const auto e_r = rg_to_host(state.rad_E.data(), state.rad_E.size());
    const auto v0 = rg_to_host(state.vol.data(), state.rho.size());
    double u_rad = 0.0;
    const std::size_t ng = e_r.size() / std::max<std::size_t>(v0.size(), 1);
    for (std::size_t c = 0; c < v0.size(); ++c) {
      for (std::size_t g = 0; g < ng; ++g) {
        u_rad += e_r[c * ng + g] * v0[c];
      }
    }
    // Gate A: the discretely conserved kinetic content is the NODAL form
    // (delta K = dt F ubar holds per node); the budget's averaged E_kin is
    // intentionally NOT used here.
    out.E_tot_init = tot0.E_int_e + tot0.E_int_i + tot0.E_kin_nodal + u_rad;
  }
  cfg.main.t_end = t_end;
  coupling::Driver driver;
  driver.run(state, cfg);
  rg_set_coupling_env(false);
  unsetenv("TENRYU_RAD_DIAG_SKIP_SOLVE");
  const std::size_t n_cells = state.rho.size();
  out.rad_E = rg_to_host(state.rad_E.data(), state.rad_E.size());
  out.vol = rg_to_host(state.vol.data(), n_cells);
  out.u = rg_to_host(state.v_r.data(), n_cells + 1);
  out.x = rg_to_host(state.x_r.data(), n_cells + 1);
  out.t = state.t;
  {
    const auto tot0 = diagnostics::compute_energy_totals_1d(state);
    const auto e_r = rg_to_host(state.rad_E.data(), state.rad_E.size());
    const auto v0 = rg_to_host(state.vol.data(), state.rho.size());
    double u_rad = 0.0;
    const std::size_t ng = e_r.size() / std::max<std::size_t>(v0.size(), 1);
    for (std::size_t c = 0; c < v0.size(); ++c) {
      for (std::size_t g = 0; g < ng; ++g) {
        u_rad += e_r[c * ng + g] * v0[c];
      }
    }
    // Gate A: the discretely conserved kinetic content is the NODAL form
    // (delta K = dt F ubar holds per node); the budget's averaged E_kin is
    // intentionally NOT used here.
    out.E_tot_end = tot0.E_int_e + tot0.E_int_i + tot0.E_kin_nodal + u_rad;
  }
  out.ok = true;
  return out;
}

bool rg_adiabat_one_geom(const std::string& deck, const int want_geom,
                         const char* label) {
  // Four runs (gate-side seed; the radiation solve is skipped):
  // ON t~0 / ON t_end / OFF t~0 / OFF t_end. Only the interior cells
  // (index < 0.7 n) are asserted — the outer-edge rarefaction from the
  // free hydro boundary reaches ~cs*t ~ 0.2 of the domain by t_end.
  const double kT1 = 1.0e-9;
  const double kT2 = 2.9e-8;
  const double kEr0 = 1.0e8;  // gate-side seed persists: solve is skipped
  const RgRunOut s0 = rg_run(deck, true, kEr0, kT1, want_geom, label);
  const RgRunOut s1 = rg_run(deck, true, kEr0, kT2, want_geom, label);
  const RgRunOut f0 = rg_run(deck, false, kEr0, kT1, want_geom, label);
  const RgRunOut f1 = rg_run(deck, false, kEr0, kT2, want_geom, label);
  if (!s0.ok || !s1.ok || !f0.ok || !f1.ok) {
    return false;
  }
  const std::size_t n = s0.vol.size();
  if (s1.rad_E.size() != n || s0.rad_E.size() != n || f0.rad_E.size() != n ||
      f1.rad_E.size() != n) {
    core::log_error(std::string(label) +
                    " group/cell layout mismatch (need groups=1)");
    return false;
  }
  const std::size_t n_interior = (n * 7U) / 10U;
  double max_law_dev = 0.0;
  double e_total = 0.0;
  double max_off_dev = 0.0;
  double max_expansion = 0.0;
  for (std::size_t c = 0; c < n_interior; ++c) {
    const double lhs = s1.rad_E[c] * std::pow(s1.vol[c], 4.0 / 3.0);
    const double rhs = s0.rad_E[c] * std::pow(s0.vol[c], 4.0 / 3.0);
    if (rhs > 0.0) {
      max_law_dev = std::fmax(max_law_dev, std::fabs(lhs / rhs - 1.0));
    }
    e_total += s1.rad_E[c] * s1.vol[c];
    max_expansion = std::fmax(max_expansion, s1.vol[c] / s0.vol[c]);
    // Coupling OFF: the DENSITY is frozen (W-J attribution), so
    // E V^{4/3} grows by (V1/V0)^{4/3}, referenced to the OFF run's own
    // t~0 sample. (Measured: dev == (V1/V0)-1 when this exponent was
    // mistakenly 1/3 — the code path was right, the formula was not.)
    const double off_lhs = f1.rad_E[c] * std::pow(f1.vol[c], 4.0 / 3.0);
    const double off_expect = f0.rad_E[c] * std::pow(f0.vol[c], 4.0 / 3.0) *
                              std::pow(f1.vol[c] / f0.vol[c], 4.0 / 3.0);
    if (off_expect > 0.0) {
      max_off_dev =
          std::fmax(max_off_dev, std::fabs(off_lhs / off_expect - 1.0));
    }
  }
  core::log_info(std::string(label) + " max_law_dev=" + rg_fmt(max_law_dev) +
                 " E_rad_total_end=" + rg_fmt(e_total) +
                 " max_off_frozen_dev=" + rg_fmt(max_off_dev) +
                 " max_expansion=" + rg_fmt(max_expansion) +
                 " dE_on_t1=" +
                 rg_fmt(std::fabs(s0.E_tot_end - s0.E_tot_init)) +
                 " dE_on_t2=" +
                 rg_fmt(std::fabs(s1.E_tot_end - s1.E_tot_init)));
  bool pass = true;
  if (!(e_total > 1.0e-6)) {
    core::log_error(std::string(label) +
                    " equilibrium field not persistent (solver ate it)");
    pass = false;
  }
  if (!(max_expansion > 1.02)) {
    core::log_error(std::string(label) +
                    " mesh barely moved — law check not exercised");
    pass = false;
  }
  // v3 work-conjugate payment: the continuum adiabat holds to integrator
  // order (O(delta^2) per half-step), NOT as an identity — track it loosely
  // as a physics-sanity check.
  if (!(max_law_dev <= 1.0e-3)) {
    core::log_error(std::string(label) + " E V^{4/3} tracking out of band");
    pass = false;
  }
  // Gate A (verdict 2026-07-06): TOTAL energy conservation of the coupled
  // piston (free outer boundary, P_ext=0 => no boundary work; solve skipped;
  // the p_r force work must be exactly paid by the field).
  for (const RgRunOut* leg : {&s0, &s1}) {
    const double scale = std::max(std::fabs(leg->E_tot_init), 1.0);
    const double drift = std::fabs(leg->E_tot_end - leg->E_tot_init) / scale;
    if (!(drift <= 1.0e-11)) {
      core::log_error(std::string(label) + " coupled piston total-energy " +
                      "drift " + rg_fmt(drift) + " exceeds 1e-11");
      pass = false;
    }
  }
  if (!(max_off_dev <= 1.0e-2)) {
    core::log_error(std::string(label) +
                    " coupling-OFF frozen-density expectation violated");
    pass = false;
  }
  return pass;
}

double rg_modal_amplitude(const RgRunOut& s) {
  // A = sum_j u_j sin(pi x_j / L), L = 1 (deck domain).
  double a = 0.0;
  for (std::size_t j = 0; j < s.u.size(); ++j) {
    a += s.u[j] * std::sin(M_PI * s.x[j]);
  }
  return a;
}

double rg_fit_omega(const std::vector<double>& t, const std::vector<double>& a,
                    const double w_lo, const double w_hi) {
  // Grid search omega; per omega solve linear LS for A cos(wt) + B sin(wt).
  double best_w = w_lo;
  double best_sse = 1.0e300;
  const int kn = 4000;
  for (int i = 0; i <= kn; ++i) {
    const double w = w_lo + (w_hi - w_lo) * static_cast<double>(i) / kn;
    double scc = 0.0, sss = 0.0, scs = 0.0, syc = 0.0, sys = 0.0;
    for (std::size_t m = 0; m < t.size(); ++m) {
      const double cw = std::cos(w * t[m]);
      const double sw = std::sin(w * t[m]);
      scc += cw * cw;
      sss += sw * sw;
      scs += cw * sw;
      syc += a[m] * cw;
      sys += a[m] * sw;
    }
    const double det = scc * sss - scs * scs;
    if (!(std::fabs(det) > 1.0e-30)) {
      continue;
    }
    const double A = (syc * sss - sys * scs) / det;
    const double B = (sys * scc - syc * scs) / det;
    double sse = 0.0;
    for (std::size_t m = 0; m < t.size(); ++m) {
      const double r = a[m] - A * std::cos(w * t[m]) - B * std::sin(w * t[m]);
      sse += r * r;
    }
    if (sse < best_sse) {
      best_sse = sse;
      best_w = w;
    }
  }
  return best_w;
}

double rg_measure_omega(const std::string& deck, const bool coupling_on,
                        const double e_r0, const char* label, bool* ok) {
  const double kTGas = 1.41421e-5;  // 2L/c_s, c_s=sqrt(2)*1e5 (Z=1, 2T)
  std::vector<double> ts;
  std::vector<double> as;
  for (int m = 1; m <= 8; ++m) {
    const double tm = kTGas * static_cast<double>(m) / 8.0;
    const RgRunOut s = rg_run(deck, coupling_on, e_r0, tm, 2, label);
    if (!s.ok) {
      *ok = false;
      return 0.0;
    }
    ts.push_back(s.t);
    as.push_back(rg_modal_amplitude(s));
  }
  *ok = true;
  const double w_gas = M_PI * 1.41421356e5;  // k c_s = (pi/L) c_s
  return rg_fit_omega(ts, as, 0.7 * w_gas, 1.6 * w_gas);
}

#endif  // TENRYU_ENABLE_PYTHON

}  // namespace

bool run_rad_gamma_adiabat_verify() {
  const char* label = "[verify:rad_gamma_adiabat]";
#if defined(TENRYU_ENABLE_PYTHON) && TENRYU_ENABLE_PYTHON
  if (!rg_cuda_available("rad_gamma_adiabat")) {
    return true;
  }
  rg_ensure_python();
  bool pass = true;
  if (!rg_adiabat_one_geom("examples/verification/rad_gamma_adiabat_spherical.py",
                           0, "[verify:rad_gamma_adiabat_sph]")) {
    pass = false;
  }
  if (!rg_adiabat_one_geom("examples/verification/rad_gamma_adiabat_planar.py",
                           2, "[verify:rad_gamma_adiabat_pla]")) {
    pass = false;
  }
  core::log_info(std::string(label) + (pass ? " PASSED" : " FAILED"));
  return pass;
#else
  core::log_info(std::string("[SKIP] ") + label + ": python disabled");
  return true;
#endif
}

bool run_rad_gamma_ceff_verify() {
  const char* label = "[verify:rad_gamma_ceff]";
#if defined(TENRYU_ENABLE_PYTHON) && TENRYU_ENABLE_PYTHON
  if (!rg_cuda_available("rad_gamma_ceff")) {
    return true;
  }
  rg_ensure_python();
  const std::string deck = "examples/verification/rad_gamma_ceff.py";
  // Seeded field (solve skipped): Z=1 2T doubles the gas pressure
  // (gamma_g p_g = rho c_s^2 = 2e10, c_s = sqrt(2)e5 — measured
  // omega_off/omega(1T-assumption) = 1.414 exactly). (4/9) E_r0 =
  // 0.3 gamma_g p_g => E_r0 = 1.35e10, ratio sqrt(1.3). (With the old
  // 6.75e9 seed the measured ratio was 1.072393 vs sqrt(1.15) =
  // 1.072381 — 1.1e-5 agreement, physics confirmed.)
  const double kEr0 = 1.35e10;
  const double kExpectRatio = std::sqrt(1.3);
  bool ok_on = false;
  bool ok_off = false;
  const double w_off = rg_measure_omega(deck, false, kEr0, label, &ok_off);
  const double w_on = rg_measure_omega(deck, true, kEr0, label, &ok_on);
  if (!ok_on || !ok_off || !(w_off > 0.0)) {
    core::log_error(std::string(label) + " FAILED (run error)");
    return false;
  }
  const double ratio = w_on / w_off;
  const double w_gas = M_PI * 1.41421356e5;
  core::log_info(std::string(label) + " omega_on=" + rg_fmt(w_on) +
                 " omega_off=" + rg_fmt(w_off) +
                 " ratio=" + rg_fmt(ratio) +
                 " expect=" + rg_fmt(kExpectRatio) +
                 " omega_gas_theory=" + rg_fmt(w_gas));
  bool pass = true;
  if (!(std::fabs(w_off / w_gas - 1.0) <= 2.0e-2)) {
    core::log_error(std::string(label) + " OFF-run gas frequency off theory");
    pass = false;
  }
  if (!(std::fabs(ratio / kExpectRatio - 1.0) <= 1.0e-2)) {
    core::log_error(std::string(label) + " c_eff ratio out of tolerance");
    pass = false;
  }
  core::log_info(std::string(label) + (pass ? " PASSED" : " FAILED"));
  return pass;
#else
  core::log_info(std::string("[SKIP] ") + label + ": python disabled");
  return true;
#endif
}

}  // namespace tenryu::drivers
