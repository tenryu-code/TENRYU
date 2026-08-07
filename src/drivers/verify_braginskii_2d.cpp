// 2D RZ Braginskii viscosity verify gates.
//
// V2a/V2b fit standing-wave energy decay from fresh deck loads, subtract
// the inviscid numerical damping, and compare the differential rate with
// gamma = (2/3)(eta/rho) k^2. Each gate covers both 2D AV paths.
// V3 checks that viscous heating is routed to ion energy in a 2T run.

#include "drivers/verify_braginskii_2d.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "coupling/driver.hpp"
#include "hydro/braginskii_viscosity.cuh"

#if defined(TENRYU_ENABLE_PYTHON) && TENRYU_ENABLE_PYTHON
#include <pybind11/embed.h>

#include "core/namelist/geometry_eval.hpp"
#include "core/namelist/runtime.hpp"
#include "mesh/mesh.hpp"
#endif

namespace tenryu::drivers {

namespace {

std::string braginskii2d_fmt(const double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.6e", v);
  return std::string(buf);
}

bool braginskii2d_cuda_available(const char* label) {
  int n = 0;
  const cudaError_t err = cudaGetDeviceCount(&n);
  if (err != cudaSuccess || n <= 0) {
    core::log_info(std::string("[SKIP] ") + label + ": CUDA not available");
    return false;
  }
  return true;
}

void braginskii2d_set_env(const bool enabled, const char* model,
                          const double eta_const, const double dt_safety,
                          const char* species = nullptr) {
  if (enabled) {
    setenv("TENRYU_BRAG_ENABLE", "1", 1);
    setenv("TENRYU_BRAG_MODEL", model, 1);
    setenv("TENRYU_BRAG_ETA_CONST", braginskii2d_fmt(eta_const).c_str(), 1);
    setenv("TENRYU_BRAG_DT_SAFETY", braginskii2d_fmt(dt_safety).c_str(), 1);
    if (species != nullptr) {
      setenv("TENRYU_BRAG_SPECIES", species, 1);
    } else {
      unsetenv("TENRYU_BRAG_SPECIES");
    }
  } else {
    unsetenv("TENRYU_BRAG_ENABLE");
    unsetenv("TENRYU_BRAG_MODEL");
    unsetenv("TENRYU_BRAG_ETA_CONST");
    unsetenv("TENRYU_BRAG_DT_SAFETY");
    unsetenv("TENRYU_BRAG_SPECIES");
  }
}

#if defined(TENRYU_ENABLE_PYTHON) && TENRYU_ENABLE_PYTHON

void braginskii2d_ensure_python() {
  if (!Py_IsInitialized()) {
    pybind11::initialize_interpreter();
  }
}

core::State braginskii2d_load_state(
    const std::string& deck, core::Config& cfg,
    const std::optional<core::AvModel> av_override = std::nullopt) {
  core::State state;
  core::namelist::Runtime runtime;
  runtime.execute(deck);
  cfg = runtime.config();
  if (av_override.has_value()) {
    cfg.numerics.hydro.av_model = *av_override;
  }
  state = core::State::allocate(cfg, runtime.builder().hydro_t_start_eV);
  state.mesh = mesh::create_mesh(cfg, state);
  state.vol = state.mesh.cell_vol;
  core::namelist::evaluate_geometry(cfg, runtime.builder(), state);
  state.t_next_plot =
      (cfg.output.plot_every_s > 0.0) ? (state.t + cfg.output.plot_every_s)
                                      : -1.0;
  state.t_next_history = (cfg.output.history_every_s > 0.0)
                             ? (state.t + cfg.output.history_every_s)
                             : -1.0;
  state.t_next_checkpoint = (cfg.output.checkpoint_every_s > 0.0)
                                ? (state.t + cfg.output.checkpoint_every_s)
                                : -1.0;
  return state;
}

std::vector<double> braginskii2d_to_host(const double* dev,
                                         const std::size_t n) {
  std::vector<double> h(n, 0.0);
  if (dev != nullptr && n > 0) {
    cudaMemcpy(h.data(), dev, n * sizeof(double), cudaMemcpyDeviceToHost);
  }
  return h;
}

struct WaveSample {
  double t = 0.0;
  double e_wave = 0.0;
  double e_total = 0.0;
};

WaveSample braginskii2d_sample_wave(const core::State& state,
                                    const double gamma_gas) {
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_nodes = state.v_r.size();
  const std::vector<double> v_r =
      braginskii2d_to_host(state.v_r.data(), n_nodes);
  const std::vector<double> v_z =
      braginskii2d_to_host(state.v_z.data(), n_nodes);
  const std::vector<double> m =
      braginskii2d_to_host(state.mass.data(), n_cells);
  const std::vector<double> vol =
      braginskii2d_to_host(state.vol.data(), n_cells);
  const std::vector<double> pe =
      braginskii2d_to_host(state.Pe.data(), n_cells);
  const std::vector<double> pi =
      braginskii2d_to_host(state.Pi.data(), n_cells);
  const std::vector<double> ee =
      braginskii2d_to_host(state.ee.data(), n_cells);
  std::vector<double> ei;
  if (!state.ei.empty()) {
    ei = braginskii2d_to_host(state.ei.data(), n_cells);
  }

  double ke = 0.0;
  for (int i = 0; i <= nr; ++i) {
    for (int j = 0; j <= nz; ++j) {
      double m_node = 0.0;
      for (int ci = i - 1; ci <= i; ++ci) {
        for (int cj = j - 1; cj <= j; ++cj) {
          if (ci >= 0 && ci < nr && cj >= 0 && cj < nz) {
            const std::size_t c =
                static_cast<std::size_t>(ci * nz + cj);
            m_node += 0.25 * m[c];
          }
        }
      }
      const std::size_t n =
          static_cast<std::size_t>(i * (nz + 1) + j);
      ke += 0.5 * m_node * (v_r[n] * v_r[n] + v_z[n] * v_z[n]);
    }
  }

  double vsum = 0.0;
  double psum = 0.0;
  double msum = 0.0;
  double esum = 0.0;
  for (std::size_t c = 0; c < n_cells; ++c) {
    const double p = pe[c] + pi[c];
    vsum += vol[c];
    psum += p * vol[c];
    msum += m[c];
    double e_spec = ee[c];
    if (!ei.empty()) {
      e_spec += ei[c];
    }
    esum += m[c] * e_spec;
  }
  const double pbar = psum / vsum;
  const double rhobar = msum / vsum;
  const double cs2 = gamma_gas * pbar / rhobar;
  double pe_wave = 0.0;
  if (cs2 > 0.0) {
    for (std::size_t c = 0; c < n_cells; ++c) {
      const double dp = (pe[c] + pi[c]) - pbar;
      pe_wave += dp * dp * vol[c] / (2.0 * rhobar * cs2);
    }
  }
  WaveSample s;
  s.t = state.t;
  s.e_wave = ke + pe_wave;
  s.e_total = ke + esum;
  return s;
}

double braginskii2d_fit_decay_rate(const std::vector<WaveSample>& samples) {
  // ln E_w = a - 2 gamma t, least squares over positive samples.
  double st = 0.0;
  double sy = 0.0;
  double stt = 0.0;
  double sty = 0.0;
  std::size_t n = 0;
  for (const WaveSample& s : samples) {
    if (!(s.e_wave > 0.0)) {
      continue;
    }
    const double y = std::log(s.e_wave);
    st += s.t;
    sy += y;
    stt += s.t * s.t;
    sty += s.t * y;
    ++n;
  }
  if (n < 3) {
    return 0.0;
  }
  const double denom = static_cast<double>(n) * stt - st * st;
  if (!(std::fabs(denom) > 0.0)) {
    return 0.0;
  }
  const double slope = (static_cast<double>(n) * sty - st * sy) / denom;
  return -0.5 * slope;
}

struct WaveRunResult {
  double gamma_fit = 0.0;
  double e_total_drift_rel = 0.0;
  long steps = 0;
  bool ok = false;
};

WaveRunResult braginskii2d_run_wave(
    const std::string& deck, const bool viscous, const double eta_const,
    const double t_run, const double gamma_gas,
    const std::optional<core::AvModel> av_override, const char* label,
    const char* species = nullptr) {
  WaveRunResult out;
  braginskii2d_set_env(viscous, "constant", eta_const, 0.3, species);
  // Sample the decay with FRESH deck loads run to increasing t_end. All
  // samples follow driver.run because EOS energy fields are initialized there.
  static const double kFractions[] = {1.25e-5, 0.5, 1.0};
  std::vector<WaveSample> samples;
  double e_total_0 = 0.0;
  double e_total_end = 0.0;
  for (const double frac : kFractions) {
    core::Config cfg;
    core::State state = braginskii2d_load_state(deck, cfg, av_override);
    TENRYU_ASSERT(state.mesh.dim == 2,
                  std::string(label) + " requires a 2D mesh");
    cfg.main.t_end = frac * t_run;
    coupling::Driver driver;
    driver.run(state, cfg);
    const WaveSample s = braginskii2d_sample_wave(state, gamma_gas);
    samples.push_back(s);
    if (samples.size() == 1) {
      e_total_0 = s.e_total;
    }
    e_total_end = s.e_total;
    out.steps = static_cast<long>(state.step);
  }
  braginskii2d_set_env(false, "", 0.0, 0.0);
  out.gamma_fit = braginskii2d_fit_decay_rate(samples);
  out.e_total_drift_rel =
      std::fabs(e_total_end - e_total_0) / std::fabs(e_total_0);
  out.ok = true;
  return out;
}

bool braginskii2d_check_decay_leg(
    const std::string& deck, const double eta_const, const double t_run,
    const double gamma_gas, const double gamma_th,
    const std::optional<core::AvModel> av_override, const char* av_label,
    const char* label) {
  const WaveRunResult off = braginskii2d_run_wave(
      deck, false, 0.0, t_run, gamma_gas, av_override, label);
  const WaveRunResult on = braginskii2d_run_wave(
      deck, true, eta_const, t_run, gamma_gas, av_override, label);
  if (!off.ok || !on.ok) {
    core::log_error(std::string(label) + " " + av_label +
                    " FAILED (run error)");
    return false;
  }
  const double gamma_diff = on.gamma_fit - off.gamma_fit;
  const double ratio = gamma_diff / gamma_th;
  const double drift_diff =
      std::fabs(on.e_total_drift_rel - off.e_total_drift_rel);
  core::log_info(std::string(label) + " " + av_label +
                 " gamma_on=" + braginskii2d_fmt(on.gamma_fit) +
                 " gamma_off=" + braginskii2d_fmt(off.gamma_fit) +
                 " gamma_th=" + braginskii2d_fmt(gamma_th) +
                 " ratio=" + braginskii2d_fmt(ratio));
  core::log_info(std::string(label) + " " + av_label +
                 " E_drift_on=" + braginskii2d_fmt(on.e_total_drift_rel) +
                 " E_drift_off=" + braginskii2d_fmt(off.e_total_drift_rel) +
                 " E_drift_diff=" + braginskii2d_fmt(drift_diff) +
                 " steps_on=" + std::to_string(on.steps) +
                 " steps_off=" + std::to_string(off.steps));
  bool pass = true;
  if (!(std::fabs(ratio - 1.0) <= 0.05)) {
    core::log_error(std::string(label) + " " + av_label +
                    " decay-rate ratio out of tolerance");
    pass = false;
  }
  if (!(drift_diff <= 1.0e-10)) {
    core::log_error(std::string(label) + " " + av_label +
                    " differential total-energy drift too large");
    pass = false;
  }
  return pass;
}

struct RoutingRunResult {
  double ei_total = 0.0;
  double ee_total = 0.0;
  double e_total = 0.0;
  long steps = 0;
  bool ok = false;
};

RoutingRunResult braginskii2d_run_routing(
    const std::string& deck, const bool viscous, const double eta_const,
    const char* species = nullptr,
    const std::optional<core::AvModel> av_override = std::nullopt) {
  RoutingRunResult out;
  braginskii2d_set_env(viscous, "constant", eta_const, 0.3, species);
  core::Config cfg;
  core::State state = braginskii2d_load_state(deck, cfg, av_override);
  TENRYU_ASSERT(state.mesh.dim == 2,
                "braginskii2d 2T routing gate requires a 2D mesh");
  coupling::Driver driver;
  driver.run(state, cfg);
  braginskii2d_set_env(false, "", 0.0, 0.0);

  const std::size_t n_cells = state.rho.size();
  const std::vector<double> mass =
      braginskii2d_to_host(state.mass.data(), n_cells);
  const std::vector<double> ee =
      braginskii2d_to_host(state.ee.data(), n_cells);
  const std::vector<double> ei =
      braginskii2d_to_host(state.ei.data(), n_cells);
  for (std::size_t c = 0; c < n_cells; ++c) {
    out.ee_total += mass[c] * ee[c];
    out.ei_total += mass[c] * ei[c];
  }
  out.e_total = braginskii2d_sample_wave(state, 5.0 / 3.0).e_total;
  out.steps = static_cast<long>(state.step);
  out.ok = true;
  return out;
}

#endif  // TENRYU_ENABLE_PYTHON

}  // namespace

bool run_braginskii_2d_planar_wave_decay_verify() {
  const char* label = "[verify:braginskii2d_planar_wave]";
#if defined(TENRYU_ENABLE_PYTHON) && TENRYU_ENABLE_PYTHON
  if (!braginskii2d_cuda_available("braginskii_2d_planar_wave_decay")) {
    return true;
  }
  braginskii2d_ensure_python();
  const std::string deck =
      "examples/verification/braginskii_wave_2d_planar.py";
  const double kGammaGas = 5.0 / 3.0;
  const double kDomainL = 1.0;
  const double kRho0 = 1.0;
  const double kEtaConst = 2280.0;
  const double kPi = 3.14159265358979323846;
  const double t_run = 8.0e-5;
  const double k_wave = kPi / kDomainL;
  const double gamma_th =
      (2.0 / 3.0) * (kEtaConst / kRho0) * k_wave * k_wave;

  const bool legacy_pass = braginskii2d_check_decay_leg(
      deck, kEtaConst, t_run, kGammaGas, gamma_th, std::nullopt,
      "av=scalar_vnr_legacy", label);
  const bool csw98_pass = braginskii2d_check_decay_leg(
      deck, kEtaConst, t_run, kGammaGas, gamma_th,
      core::AvModel::CswEdgeCsw98, "av=csw_edge_csw98", label);
  const bool pass = legacy_pass && csw98_pass;
  core::log_info(std::string(label) + (pass ? " PASSED" : " FAILED"));
  return pass;
#else
  core::log_info(std::string("[SKIP] ") + label + ": python disabled");
  return true;
#endif
}

bool run_braginskii_2d_cyl_wave_decay_verify() {
  const char* label = "[verify:braginskii2d_cyl_wave]";
#if defined(TENRYU_ENABLE_PYTHON) && TENRYU_ENABLE_PYTHON
  if (!braginskii2d_cuda_available("braginskii_2d_cyl_wave_decay")) {
    return true;
  }
  braginskii2d_ensure_python();
  const std::string deck =
      "examples/verification/braginskii_wave_2d_cyl.py";
  const double kGammaGas = 5.0 / 3.0;
  const double kDomainL = 1.0;
  const double kSoundSpeed = 1.0e5;
  const double kRho0 = 1.0;
  const double kPi = 3.14159265358979323846;
  const double kJ11 = 3.8317059702075123;
  const double j1_at_j11 = std::cyl_bessel_j(1.0, kJ11);
  if (std::fabs(j1_at_j11) > 1.0e-14) {
    core::log_error(std::string(label) + " J1(kJ11) residual too large");
    return false;
  }
  const double k_wave = kJ11 / kDomainL;
  const double kEtaConst = 2280.0 * kPi / kJ11;
  const double t_run = 8.0 * kPi / (k_wave * kSoundSpeed);
  const double gamma_th =
      (2.0 / 3.0) * (kEtaConst / kRho0) * k_wave * k_wave;

  const bool legacy_pass = braginskii2d_check_decay_leg(
      deck, kEtaConst, t_run, kGammaGas, gamma_th, std::nullopt,
      "av=scalar_vnr_legacy", label);
  const bool csw98_pass = braginskii2d_check_decay_leg(
      deck, kEtaConst, t_run, kGammaGas, gamma_th,
      core::AvModel::CswEdgeCsw98, "av=csw_edge_csw98", label);
  const bool pass = legacy_pass && csw98_pass;
  core::log_info(std::string(label) + (pass ? " PASSED" : " FAILED"));
  return pass;
#else
  core::log_info(std::string("[SKIP] ") + label + ": python disabled");
  return true;
#endif
}

bool run_braginskii_2d_two_temp_routing_verify() {
  const char* label = "[verify:braginskii2d_two_temp_routing]";
#if defined(TENRYU_ENABLE_PYTHON) && TENRYU_ENABLE_PYTHON
  if (!braginskii2d_cuda_available("braginskii_2d_two_temp_routing")) {
    return true;
  }
  braginskii2d_ensure_python();
  const std::string deck =
      "examples/verification/braginskii_wave_2d_planar_2t.py";
  const RoutingRunResult off = braginskii2d_run_routing(deck, false, 0.0);
  const RoutingRunResult on = braginskii2d_run_routing(deck, true, 2280.0);
  if (!off.ok || !on.ok) {
    core::log_error(std::string(label) + " FAILED (run error)");
    return false;
  }
  const double d_ei = on.ei_total - off.ei_total;
  const double d_ee = on.ee_total - off.ee_total;
  const double e_total_diff_rel =
      std::fabs(on.e_total - off.e_total) / std::fabs(off.e_total);
  core::log_info(std::string(label) + " d_ei=" + braginskii2d_fmt(d_ei) +
                 " d_ee=" + braginskii2d_fmt(d_ee) +
                 " E_total_diff_rel=" +
                 braginskii2d_fmt(e_total_diff_rel) +
                 " steps_on=" + std::to_string(on.steps) +
                 " steps_off=" + std::to_string(off.steps));
  bool pass = true;
  if (!(d_ei > 0.0)) {
    core::log_error(std::string(label) + " ion energy did not increase");
    pass = false;
  }
  if (!(std::fabs(d_ee) <= 0.05 * d_ei)) {
    core::log_error(std::string(label) +
                    " electron contamination out of tolerance");
    pass = false;
  }
  if (!(e_total_diff_rel <= 1.0e-9)) {
    core::log_error(std::string(label) +
                    " differential total-energy drift too large");
    pass = false;
  }
  core::log_info(std::string(label) + (pass ? " PASSED" : " FAILED"));
  return pass;
#else
  core::log_info(std::string("[SKIP] ") + label + ": python disabled");
  return true;
#endif
}

bool run_braginskii_2d_electron_wave_decay_verify() {
  const char* label = "[verify:braginskii2d_electron_wave]";
#if defined(TENRYU_ENABLE_PYTHON) && TENRYU_ENABLE_PYTHON
  if (!braginskii2d_cuda_available("braginskii_2d_electron_wave_decay")) {
    return true;
  }
  braginskii2d_ensure_python();
  const std::string deck =
      "examples/verification/braginskii_wave_2d_planar.py";
  const double kGammaGas = 5.0 / 3.0;
  const double kDomainL = 1.0;
  const double kRho0 = 1.0;
  const double kEtaConst = 2280.0;
  const double kPi = 3.14159265358979323846;
  const double t_run = 8.0e-5;
  const double k_wave = kPi / kDomainL;
  const double gamma_th =
      (2.0 / 3.0) * (kEtaConst / kRho0) * k_wave * k_wave;

  // species="electron"/"both" with model="constant" leave eta_eff (and so
  // the momentum physics and gamma) identical to the ion gate; only the
  // heat routing differs. One AV path suffices here (both AV paths are
  // covered by the ion V2a gate and by the electron routing gate below).
  const WaveRunResult off = braginskii2d_run_wave(
      deck, false, 0.0, t_run, kGammaGas, std::nullopt, label);
  const WaveRunResult on_e = braginskii2d_run_wave(
      deck, true, kEtaConst, t_run, kGammaGas, std::nullopt, label,
      "electron");
  const WaveRunResult on_b = braginskii2d_run_wave(
      deck, true, kEtaConst, t_run, kGammaGas, std::nullopt, label, "both");
  if (!off.ok || !on_e.ok || !on_b.ok) {
    core::log_error(std::string(label) + " FAILED (run error)");
    return false;
  }
  const double ratio_e = (on_e.gamma_fit - off.gamma_fit) / gamma_th;
  const double ratio_b = (on_b.gamma_fit - off.gamma_fit) / gamma_th;
  const double drift_e =
      std::fabs(on_e.e_total_drift_rel - off.e_total_drift_rel);
  const double drift_b =
      std::fabs(on_b.e_total_drift_rel - off.e_total_drift_rel);
  core::log_info(std::string(label) +
                 " gamma_e=" + braginskii2d_fmt(on_e.gamma_fit) +
                 " gamma_b=" + braginskii2d_fmt(on_b.gamma_fit) +
                 " gamma_off=" + braginskii2d_fmt(off.gamma_fit) +
                 " gamma_th=" + braginskii2d_fmt(gamma_th) +
                 " ratio_e=" + braginskii2d_fmt(ratio_e) +
                 " ratio_b=" + braginskii2d_fmt(ratio_b));
  core::log_info(std::string(label) + " E_drift_e=" +
                 braginskii2d_fmt(drift_e) + " E_drift_b=" +
                 braginskii2d_fmt(drift_b));
  bool pass = true;
  if (!(std::fabs(ratio_e - 1.0) <= 0.05)) {
    core::log_error(std::string(label) +
                    " electron decay-rate ratio out of tolerance");
    pass = false;
  }
  if (!(std::fabs(ratio_b - 1.0) <= 0.05)) {
    core::log_error(std::string(label) +
                    " both decay-rate ratio out of tolerance");
    pass = false;
  }
  if (!(drift_e <= 1.0e-10)) {
    core::log_error(std::string(label) +
                    " electron differential total-energy drift too large");
    pass = false;
  }
  if (!(drift_b <= 1.0e-10)) {
    core::log_error(std::string(label) +
                    " both differential total-energy drift too large");
    pass = false;
  }
  // eta_eff is species-invariant by construction under model="constant":
  // the two fits must agree far inside the gate tolerance.
  if (!(std::fabs(on_e.gamma_fit - on_b.gamma_fit) <= 1.0e-6 * gamma_th)) {
    core::log_error(std::string(label) +
                    " electron/both gamma mismatch (eta_eff not invariant)");
    pass = false;
  }
  core::log_info(std::string(label) + (pass ? " PASSED" : " FAILED"));
  return pass;
#else
  core::log_info(std::string("[SKIP] ") + label + ": python disabled");
  return true;
#endif
}

bool run_braginskii_2d_electron_two_temp_routing_verify() {
  const char* label = "[verify:braginskii2d_electron_two_temp_routing]";
#if defined(TENRYU_ENABLE_PYTHON) && TENRYU_ENABLE_PYTHON
  if (!braginskii2d_cuda_available(
          "braginskii_2d_electron_two_temp_routing")) {
    return true;
  }
  braginskii2d_ensure_python();
  const std::string deck =
      "examples/verification/braginskii_wave_2d_planar_2t.py";
  // Two AV paths: legacy exercises the second apply_visc_heat launch
  // (use_two_temp=0 routing of heat_rate_e -> ee), csw98 exercises the
  // compatible-path VISC_SPLIT=1 work split (f_e = 1 for pure electron).
  struct Leg {
    std::optional<core::AvModel> av;
    const char* name;
  };
  const Leg legs[] = {
      {std::nullopt, "av=scalar_vnr_legacy"},
      {core::AvModel::CswEdgeCsw98, "av=csw_edge_csw98"},
  };
  bool pass = true;
  for (const Leg& leg : legs) {
    const RoutingRunResult off =
        braginskii2d_run_routing(deck, false, 0.0, nullptr, leg.av);
    const RoutingRunResult on =
        braginskii2d_run_routing(deck, true, 2280.0, "electron", leg.av);
    if (!off.ok || !on.ok) {
      core::log_error(std::string(label) + " " + leg.name +
                      " FAILED (run error)");
      return false;
    }
    const double d_ei = on.ei_total - off.ei_total;
    const double d_ee = on.ee_total - off.ee_total;
    const double e_total_diff_rel =
        std::fabs(on.e_total - off.e_total) / std::fabs(off.e_total);
    core::log_info(std::string(label) + " " + leg.name +
                   " d_ee=" + braginskii2d_fmt(d_ee) +
                   " d_ei=" + braginskii2d_fmt(d_ei) +
                   " E_total_diff_rel=" +
                   braginskii2d_fmt(e_total_diff_rel) +
                   " steps_on=" + std::to_string(on.steps) +
                   " steps_off=" + std::to_string(off.steps));
    if (!(d_ee > 0.0)) {
      core::log_error(std::string(label) + " " + leg.name +
                      " electron energy did not increase");
      pass = false;
    }
    if (!(std::fabs(d_ei) <= 0.05 * d_ee)) {
      core::log_error(std::string(label) + " " + leg.name +
                      " ion contamination out of tolerance");
      pass = false;
    }
    if (!(e_total_diff_rel <= 1.0e-9)) {
      core::log_error(std::string(label) + " " + leg.name +
                      " differential total-energy drift too large");
      pass = false;
    }
  }
  core::log_info(std::string(label) + (pass ? " PASSED" : " FAILED"));
  return pass;
#else
  core::log_info(std::string("[SKIP] ") + label + ": python disabled");
  return true;
#endif
}

bool run_braginskii_2d_regime_diag_smoke_verify() {
  const char* label = "[verify:braginskii2d_regime_diag]";
#if defined(TENRYU_ENABLE_PYTHON) && TENRYU_ENABLE_PYTHON
  if (!braginskii2d_cuda_available("braginskii_2d_regime_diag_smoke")) {
    return true;
  }
  braginskii2d_ensure_python();
  struct SmokeLeg {
    const char* deck;
    const char* name;
    bool multiblock;
  };
  const SmokeLeg legs[] = {
      {"examples/verification/braginskii_wave_2d_planar_2t.py",
       "structured", false},
      // 4-step 2T multiblock butterfly smoke deck: exercises the
      // history_diag_2d_multiblock_kernel + CSR staging end-to-end.
      {"examples/verification/2d_rz_multiblock_5block_hydro_smoke.py",
       "multiblock", true},
  };
  bool pass = true;
  for (const SmokeLeg& leg : legs) {
    // Run with the safe constant model, then evaluate the instantaneous
    // diagnostics with model=braginskii (params are env-aware and the diag
    // is a pure function of the state — no stepping under real-eta dt).
    braginskii2d_set_env(true, "constant", 2280.0, 0.3, "both");
    core::Config cfg;
    core::State state = braginskii2d_load_state(leg.deck, cfg);
    TENRYU_ASSERT(state.mesh.dim == 2,
                  "braginskii2d regime diag smoke requires a 2D mesh");
    if (leg.multiblock) {
      TENRYU_ASSERT(mesh::mesh_topo_is_multiblock(cfg.mesh),
                    "braginskii2d regime diag smoke multiblock leg requires "
                    "a multiblock deck");
    }
    coupling::Driver driver;
    driver.run(state, cfg);
    setenv("TENRYU_BRAG_MODEL", "braginskii", 1);
    const auto diag =
        tenryu::hydro::braginskii::compute_history_diagnostics(state, cfg);
    braginskii2d_set_env(false, "", 0.0, 0.0);
    const int n_cells = static_cast<int>(state.rho.size());
    core::log_info(
        std::string(label) + " " + leg.name + " valid=" +
        std::to_string(diag.valid ? 1 : 0) + " active=" +
        std::to_string(diag.n_cells_active) + "/" + std::to_string(n_cells) +
        " eta_i_max=" + braginskii2d_fmt(diag.eta_i_max) + " eta_e_max=" +
        braginskii2d_fmt(diag.eta_e_max) + " R[min,geo,max]=[" +
        braginskii2d_fmt(diag.ratio_min) + "," +
        braginskii2d_fmt(diag.ratio_geomean_masswt) + "," +
        braginskii2d_fmt(diag.ratio_max) + "] counts e/i/mix=" +
        std::to_string(diag.n_cells_e_dom) + "/" +
        std::to_string(diag.n_cells_i_dom) + "/" +
        std::to_string(diag.n_cells_mixed) + " heat_i_tot=" +
        braginskii2d_fmt(diag.heat_rate_i_tot) + " heat_e_tot=" +
        braginskii2d_fmt(diag.heat_rate_e_tot));
    if (!diag.valid || diag.n_cells_active != n_cells) {
      core::log_error(std::string(label) + " " + leg.name +
                      " diagnostics invalid/incomplete");
      pass = false;
    }
    if (!(diag.eta_i_max > 0.0) || !(diag.eta_e_max > 0.0)) {
      core::log_error(std::string(label) + " " + leg.name +
                      " physical channel etas not > 0");
      pass = false;
    }
    const int classified =
        diag.n_cells_e_dom + diag.n_cells_i_dom + diag.n_cells_mixed;
    if (classified != diag.n_cells_active) {
      core::log_error(std::string(label) + " " + leg.name +
                      " regime classification does not cover active cells");
      pass = false;
    }
    // The mass-weighted geometric mean is exp(sum m ln R / sum m): the
    // exp/log round trip carries a few-ulp error, so in the degenerate
    // all-cells-equal-R state it can land marginally outside [min, max]
    // (first observed on the multiblock leg, 2026-07-17). Allow an
    // ulp-scale relative slack on the ordering only.
    const double r_tol = 1.0e-12;
    if (!(diag.ratio_min > 0.0) ||
        !(diag.ratio_min * (1.0 - r_tol) <= diag.ratio_geomean_masswt) ||
        !(diag.ratio_geomean_masswt <= diag.ratio_max * (1.0 + r_tol))) {
      core::log_error(std::string(label) + " " + leg.name +
                      " ratio statistics inconsistent");
      pass = false;
    }
    if (!(diag.heat_rate_i_tot >= 0.0) || !(diag.heat_rate_e_tot >= 0.0)) {
      core::log_error(std::string(label) + " " + leg.name +
                      " negative heat totals");
      pass = false;
    }
  }
  core::log_info(std::string(label) + (pass ? " PASSED" : " FAILED"));
  return pass;
#else
  core::log_info(std::string("[SKIP] ") + label + ": python disabled");
  return true;
#endif
}

}  // namespace tenryu::drivers
