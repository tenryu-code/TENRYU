// Braginskii viscosity verify gates.
//
// B2 (braginskii_wave): planar standing sound wave with the constant-eta
//   model. The wave energy E_w = KE + acoustic PE decays as exp(-2 gamma t)
//   with gamma = (2/3)(eta/rho) k^2 (nu_L = (4/3) eta/rho, amplitude rate
//   nu_L k^2 / 2). Samples come from FRESH deck loads run to increasing
//   t_end (in-memory driver.run re-entry is unsupported and pumps energy).
//   The gate fits gamma on a viscous and an inviscid run and
//   checks the DIFFERENCE against theory (subtracting the scheme's own
//   numerical damping), plus total-energy conservation of the viscous run
//   (force-work vs heating compatibility).
//
// B3 (braginskii_hubble): cold spherical homologous expansion u = H r is
//   exactly stress-free for the trace-free Braginskii tensor. With the dt
//   path pinned by Numerics.dt.max_s, the viscous run must match the
//   inviscid trajectory to roundoff; with dt released, the viscous dt
//   limiter must engage (step count increases).

#include "drivers/verify_braginskii.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#if TENRYU_ENABLE_HDF5
#include <hdf5.h>
#endif

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

std::string brag_fmt(const double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.6e", v);
  return std::string(buf);
}

bool braginskii_cuda_available(const char* label) {
  int n = 0;
  const cudaError_t err = cudaGetDeviceCount(&n);
  if (err != cudaSuccess || n <= 0) {
    core::log_info(std::string("[SKIP] ") + label + ": CUDA not available");
    return false;
  }
  return true;
}

void braginskii_set_env(const bool enabled, const char* model,
                        const double eta_const, const double dt_safety) {
  if (enabled) {
    setenv("TENRYU_BRAG_ENABLE", "1", 1);
    setenv("TENRYU_BRAG_MODEL", model, 1);
    setenv("TENRYU_BRAG_ETA_CONST", brag_fmt(eta_const).c_str(), 1);
    setenv("TENRYU_BRAG_DT_SAFETY", brag_fmt(dt_safety).c_str(), 1);
  } else {
    unsetenv("TENRYU_BRAG_ENABLE");
    unsetenv("TENRYU_BRAG_MODEL");
    unsetenv("TENRYU_BRAG_ETA_CONST");
    unsetenv("TENRYU_BRAG_DT_SAFETY");
    unsetenv("TENRYU_BRAG_SPECIES");
  }
}

#if defined(TENRYU_ENABLE_PYTHON) && TENRYU_ENABLE_PYTHON

void braginskii_ensure_python() {
  if (!Py_IsInitialized()) {
    pybind11::initialize_interpreter();
  }
}

core::State braginskii_load_state(const std::string& deck,
                                  core::Config& cfg) {
  core::State state;
  core::namelist::Runtime runtime;
  runtime.execute(deck);
  cfg = runtime.config();
  const char* diag_nr = std::getenv("TENRYU_BRAG_DIAG_NR");
  if (diag_nr != nullptr && diag_nr[0] != 0) {
    const int nr_override = std::atoi(diag_nr);
    if (nr_override > 0) {
      cfg.mesh.nr = nr_override;
    }
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

std::vector<double> braginskii_to_host(const double* dev,
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

WaveSample braginskii_sample_wave(const core::State& state,
                                  const double gamma_gas) {
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_nodes = n_cells + 1;
  const std::vector<double> u = braginskii_to_host(state.v_r.data(), n_nodes);
  const std::vector<double> m = braginskii_to_host(state.mass.data(), n_cells);
  const std::vector<double> vol = braginskii_to_host(state.vol.data(), n_cells);
  const std::vector<double> pe = braginskii_to_host(state.Pe.data(), n_cells);
  const std::vector<double> pi = braginskii_to_host(state.Pi.data(), n_cells);
  const std::vector<double> ee = braginskii_to_host(state.ee.data(), n_cells);
  std::vector<double> ei;
  if (!state.ei.empty()) {
    ei = braginskii_to_host(state.ei.data(), n_cells);
  }

  double ke = 0.0;
  for (std::size_t j = 0; j < n_nodes; ++j) {
    double mn = 0.0;
    if (j == 0) {
      mn = 0.5 * m[0];
    } else if (j == n_cells) {
      mn = 0.5 * m[n_cells - 1];
    } else {
      mn = 0.5 * (m[j - 1] + m[j]);
    }
    ke += 0.5 * mn * u[j] * u[j];
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

double braginskii_fit_decay_rate(const std::vector<WaveSample>& samples) {
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

WaveRunResult braginskii_run_wave(const std::string& deck, const bool viscous,
                                  const double eta_const, const double t_run,
                                  const double gamma_gas, const int want_geom,
                                  const char* label) {
  WaveRunResult out;
  braginskii_set_env(viscous, "constant", eta_const, 0.3);
  // driver.run in-memory re-entry is unsupported (re-running the start-up
  // sequence on an evolved state pumps total energy ~x1.9 per call —
  // measured 2026-07-04; a single run conserves E_total to 4e-16). Sample
  // the decay with FRESH deck loads run to increasing t_end instead.
  // 3-point fit (t~0, t/2, t): E_w is a clean exponential (single-run E
  // conserved to 4e-16), so slope accuracy is systematic-limited, not
  // noise-limited; fewer samples halve the dominant viscous-run cost.
  // ALL samples are taken after a driver.run: EOS energy fields are only
  // initialized inside the driver, so a pre-run sample would miss the
  // internal energy entirely (measured: drift == E_int/KE_0). The first
  // sample runs to 1e-9 s (~1 step; wave phase 6e-4 rad — negligible).
  static const double kFractions[] = {1.25e-5, 0.5, 1.0};
  std::vector<WaveSample> samples;
  double e_total_0 = 0.0;
  double e_total_end = 0.0;
  for (const double frac : kFractions) {
    core::Config cfg;
    core::State state = braginskii_load_state(deck, cfg);
    if (state.mesh.geometry_code != want_geom) {
      core::log_error(std::string(label) + " geometry_code=" +
                      std::to_string(state.mesh.geometry_code) +
                      " want=" + std::to_string(want_geom));
      braginskii_set_env(false, "", 0.0, 0.0);
      return out;
    }
    cfg.main.t_end = frac * t_run;
    coupling::Driver driver;
    driver.run(state, cfg);
    const WaveSample s = braginskii_sample_wave(state, gamma_gas);
    samples.push_back(s);
    if (samples.size() == 1) {
      e_total_0 = s.e_total;
    }
    e_total_end = s.e_total;
    out.steps = static_cast<long>(state.step);
  }
  braginskii_set_env(false, "", 0.0, 0.0);
  out.gamma_fit = braginskii_fit_decay_rate(samples);
  out.e_total_drift_rel =
      std::fabs(e_total_end - e_total_0) / std::fabs(e_total_0);
  out.ok = true;
  return out;
}

struct HubbleRunResult {
  std::vector<double> r;
  std::vector<double> u;
  std::vector<double> ee;
  long steps = 0;
  bool ok = false;
};

HubbleRunResult braginskii_run_hubble(const std::string& deck,
                                      const bool viscous,
                                      const double eta_const,
                                      const double max_s_override,
                                      const char* label) {
  HubbleRunResult out;
  braginskii_set_env(viscous, "constant", eta_const, 0.3);
  core::Config cfg;
  core::State state = braginskii_load_state(deck, cfg);
  if (state.mesh.geometry_code != 0) {
    core::log_error(std::string(label) + " geometry_code=" +
                    std::to_string(state.mesh.geometry_code) + " want=0");
    braginskii_set_env(false, "", 0.0, 0.0);
    return out;
  }
  cfg.numerics.dt.max_s = max_s_override;
  coupling::Driver driver;
  driver.run(state, cfg);
  braginskii_set_env(false, "", 0.0, 0.0);
  const std::size_t n_cells = state.rho.size();
  out.r = braginskii_to_host(state.x_r.data(), n_cells + 1);
  out.u = braginskii_to_host(state.v_r.data(), n_cells + 1);
  out.ee = braginskii_to_host(state.ee.data(), n_cells);
  out.steps = static_cast<long>(state.step);
  out.ok = true;
  return out;
}

#if TENRYU_ENABLE_HDF5
// Reads the full 1-D double dataset at `path`; empty vector on any failure.
std::vector<double> braginskii_read_h5_series(const std::string& file_path,
                                              const std::string& path) {
  std::vector<double> out;
  const hid_t file = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    return out;
  }
  const hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  if (dset >= 0) {
    const hid_t space = H5Dget_space(dset);
    if (space >= 0) {
      hsize_t dims[1] = {0};
      if (H5Sget_simple_extent_ndims(space) == 1 &&
          H5Sget_simple_extent_dims(space, dims, nullptr) == 1 &&
          dims[0] > 0) {
        out.resize(static_cast<std::size_t>(dims[0]), 0.0);
        if (H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                    out.data()) < 0) {
          out.clear();
        }
      }
      H5Sclose(space);
    }
    H5Dclose(dset);
  }
  H5Fclose(file);
  return out;
}
#endif

#endif  // TENRYU_ENABLE_PYTHON

}  // namespace

bool run_braginskii_planar_wave_decay_verify() {
  const char* label = "[verify:braginskii_wave]";
#if defined(TENRYU_ENABLE_PYTHON) && TENRYU_ENABLE_PYTHON
  if (!braginskii_cuda_available("braginskii_wave")) {
    return true;
  }
  braginskii_ensure_python();
  const std::string deck = "examples/verification/braginskii_wave.py";
  const double kGammaGas = 5.0 / 3.0;
  const double kDomainL = 1.0;
  const double kSoundSpeed = 1.0e5;
  const double kRho0 = 1.0;
  const double kEtaConst = 2.28e3;
  const double kPi = 3.14159265358979323846;
  const double t_run = 4.0 * 2.0 * kDomainL / kSoundSpeed;
  const double k_wave = kPi / kDomainL;
  const double gamma_th = (2.0 / 3.0) * (kEtaConst / kRho0) * k_wave * k_wave;

  const WaveRunResult off =
      braginskii_run_wave(deck, false, 0.0, t_run, kGammaGas, 2, label);
  const WaveRunResult on =
      braginskii_run_wave(deck, true, kEtaConst, t_run, kGammaGas, 2, label);
  if (!off.ok || !on.ok) {
    core::log_error(std::string(label) + " FAILED (run error)");
    return false;
  }
  const double gamma_diff = on.gamma_fit - off.gamma_fit;
  const double ratio = gamma_diff / gamma_th;
  core::log_info(std::string(label) + " gamma_on=" + brag_fmt(on.gamma_fit) +
                 " gamma_off=" + brag_fmt(off.gamma_fit) +
                 " gamma_th=" + brag_fmt(gamma_th) +
                 " ratio=" + brag_fmt(ratio));
  core::log_info(std::string(label) +
                 " E_drift_on=" + brag_fmt(on.e_total_drift_rel) +
                 " E_drift_off=" + brag_fmt(off.e_total_drift_rel) +
                 " steps_on=" + std::to_string(on.steps) +
                 " steps_off=" + std::to_string(off.steps));
  bool pass = true;
  if (!(std::fabs(ratio - 1.0) <= 0.05)) {
    core::log_error(std::string(label) + " decay-rate ratio out of tolerance");
    pass = false;
  }
  if (!(on.e_total_drift_rel <= 5.0e-9)) {
    core::log_error(std::string(label) + " viscous total-energy drift too large");
    pass = false;
  }
  core::log_info(std::string(label) + (pass ? " PASSED" : " FAILED"));
  return pass;
#else
  core::log_info(std::string("[SKIP] ") + label + ": python disabled");
  return true;
#endif
}

bool run_braginskii_cylindrical_wave_decay_verify() {
  const char* label = "[verify:braginskii_wave_cyl]";
#if defined(TENRYU_ENABLE_PYTHON) && TENRYU_ENABLE_PYTHON
  if (!braginskii_cuda_available("braginskii_wave_cyl")) {
    return true;
  }
  braginskii_ensure_python();
  const std::string deck = "examples/verification/braginskii_wave_cyl.py";
  const double kGammaGas = 5.0 / 3.0;
  const double kDomainL = 1.0;
  const double kSoundSpeed = 1.0e5;
  const double kRho0 = 1.0;
  const double kPi = 3.14159265358979323846;
  const double kX1 = 3.8317059702075123;   // j_{1,1}, first zero of J1
  const double j1_at_x1 = std::cyl_bessel_j(1.0, kX1);
  if (std::fabs(j1_at_x1) > 1e-14) {
    core::log_error(std::string(label) + " J1(kX1) residual too large");
    return false;
  }
  const double k_wave = kX1 / kDomainL;
  const double kEtaConst = 2.28e3 * kPi / kX1;
  const double t_run = 8.0 * kPi / (k_wave * kSoundSpeed);
  const double gamma_th = (2.0 / 3.0) * (kEtaConst / kRho0) * k_wave * k_wave;

  const WaveRunResult off =
      braginskii_run_wave(deck, false, 0.0, t_run, kGammaGas, 1, label);
  const WaveRunResult on =
      braginskii_run_wave(deck, true, kEtaConst, t_run, kGammaGas, 1, label);
  if (!off.ok || !on.ok) {
    core::log_error(std::string(label) + " FAILED (run error)");
    return false;
  }
  const double gamma_diff = on.gamma_fit - off.gamma_fit;
  const double ratio = gamma_diff / gamma_th;
  core::log_info(std::string(label) + " gamma_on=" + brag_fmt(on.gamma_fit) +
                 " gamma_off=" + brag_fmt(off.gamma_fit) +
                 " gamma_th=" + brag_fmt(gamma_th) +
                 " ratio=" + brag_fmt(ratio));
  core::log_info(std::string(label) +
                 " E_drift_on=" + brag_fmt(on.e_total_drift_rel) +
                 " E_drift_off=" + brag_fmt(off.e_total_drift_rel) +
                 " steps_on=" + std::to_string(on.steps) +
                 " steps_off=" + std::to_string(off.steps));
  bool pass = true;
  if (!(std::fabs(ratio - 1.0) <= 0.05)) {
    core::log_error(std::string(label) + " decay-rate ratio out of tolerance");
    pass = false;
  }
  if (!(on.e_total_drift_rel <= 5.0e-9)) {
    core::log_error(std::string(label) + " viscous total-energy drift too large");
    pass = false;
  }
  core::log_info(std::string(label) + (pass ? " PASSED" : " FAILED"));
  return pass;
#else
  core::log_info(std::string("[SKIP] ") + label + ": python disabled");
  return true;
#endif
}

bool run_braginskii_spherical_wave_decay_verify() {
  const char* label = "[verify:braginskii_wave_sph]";
#if defined(TENRYU_ENABLE_PYTHON) && TENRYU_ENABLE_PYTHON
  if (!braginskii_cuda_available("braginskii_wave_sph")) {
    return true;
  }
  braginskii_ensure_python();
  const std::string deck = "examples/verification/braginskii_wave_sph.py";
  const double kGammaGas = 5.0 / 3.0;
  const double kDomainL = 1.0;
  const double kSoundSpeed = 1.0e5;
  const double kRho0 = 1.0;
  const double kPi = 3.14159265358979323846;
  const double kX1 = 4.4934094579090642;   // first positive root of tan x = x (zero of j1)
  const double j1s = std::sin(kX1) / (kX1 * kX1) - std::cos(kX1) / kX1;
  if (std::fabs(j1s) > 1e-15) {
    core::log_error(std::string(label) + " j1(kX1) residual too large");
    return false;
  }
  const double k_wave = kX1 / kDomainL;
  const double kEtaConst = 2.28e3 * kPi / kX1;
  const double t_run = 8.0 * kPi / (k_wave * kSoundSpeed);
  const double gamma_th = (2.0 / 3.0) * (kEtaConst / kRho0) * k_wave * k_wave;

  const WaveRunResult off =
      braginskii_run_wave(deck, false, 0.0, t_run, kGammaGas, 0, label);
  const WaveRunResult on =
      braginskii_run_wave(deck, true, kEtaConst, t_run, kGammaGas, 0, label);
  if (!off.ok || !on.ok) {
    core::log_error(std::string(label) + " FAILED (run error)");
    return false;
  }
  const double gamma_diff = on.gamma_fit - off.gamma_fit;
  const double ratio = gamma_diff / gamma_th;
  core::log_info(std::string(label) + " gamma_on=" + brag_fmt(on.gamma_fit) +
                 " gamma_off=" + brag_fmt(off.gamma_fit) +
                 " gamma_th=" + brag_fmt(gamma_th) +
                 " ratio=" + brag_fmt(ratio));
  core::log_info(std::string(label) +
                 " E_drift_on=" + brag_fmt(on.e_total_drift_rel) +
                 " E_drift_off=" + brag_fmt(off.e_total_drift_rel) +
                 " steps_on=" + std::to_string(on.steps) +
                 " steps_off=" + std::to_string(off.steps));
  bool pass = true;
  if (!(std::fabs(ratio - 1.0) <= 0.05)) {
    core::log_error(std::string(label) + " decay-rate ratio out of tolerance");
    pass = false;
  }
  if (!(on.e_total_drift_rel <= 5.0e-9)) {
    core::log_error(std::string(label) + " viscous total-energy drift too large");
    pass = false;
  }
  core::log_info(std::string(label) + (pass ? " PASSED" : " FAILED"));
  return pass;
#else
  core::log_info(std::string("[SKIP] ") + label + ": python disabled");
  return true;
#endif
}

bool run_braginskii_hubble_null_verify() {
  const char* label = "[verify:braginskii_hubble]";
#if defined(TENRYU_ENABLE_PYTHON) && TENRYU_ENABLE_PYTHON
  if (!braginskii_cuda_available("braginskii_hubble")) {
    return true;
  }
  braginskii_ensure_python();
  const std::string deck = "examples/verification/braginskii_hubble.py";
  const double kEtaConst = 9.0e2;
  const double kHubble = 2.0e5;
  const double kPinnedMaxS = 2.0e-8;
  const double kFreeMaxS = 1.0e-7;

  const HubbleRunResult run_a =
      braginskii_run_hubble(deck, false, 0.0, kPinnedMaxS, label);
  const HubbleRunResult run_b =
      braginskii_run_hubble(deck, true, kEtaConst, kPinnedMaxS, label);
  const HubbleRunResult run_c =
      braginskii_run_hubble(deck, false, 0.0, kFreeMaxS, label);
  const HubbleRunResult run_d =
      braginskii_run_hubble(deck, true, kEtaConst, kFreeMaxS, label);
  if (!run_a.ok || !run_b.ok || !run_c.ok || !run_d.ok) {
    core::log_error(std::string(label) + " FAILED (run error)");
    return false;
  }
  const std::size_t n_nodes = run_a.r.size();
  const std::size_t n_cells = run_a.ee.size();
  double r_dev = 0.0;
  double u_dev = 0.0;
  for (std::size_t j = 1; j + 2 < n_nodes; ++j) {
    r_dev = std::fmax(r_dev, std::fabs(run_b.r[j] - run_a.r[j]) /
                                 std::fabs(run_a.r[j]));
    u_dev = std::fmax(u_dev,
                      std::fabs(run_b.u[j] - run_a.u[j]) / kHubble);
  }
  double ee_dev = 0.0;
  for (std::size_t c = 0; c + 2 < n_cells; ++c) {
    ee_dev = std::fmax(ee_dev, std::fabs(run_b.ee[c] - run_a.ee[c]) /
                                   std::fmax(std::fabs(run_a.ee[c]), 1.0));
  }
  core::log_info(std::string(label) + " r_dev=" + brag_fmt(r_dev) +
                 " u_dev=" + brag_fmt(u_dev) + " ee_dev=" + brag_fmt(ee_dev));
  core::log_info(std::string(label) + " steps A/B/C/D = " +
                 std::to_string(run_a.steps) + "/" +
                 std::to_string(run_b.steps) + "/" +
                 std::to_string(run_c.steps) + "/" +
                 std::to_string(run_d.steps));
  bool pass = true;
  if (!(r_dev <= 1.0e-12)) {
    core::log_error(std::string(label) + " position null violated");
    pass = false;
  }
  if (!(u_dev <= 1.0e-12)) {
    core::log_error(std::string(label) + " velocity null violated");
    pass = false;
  }
  if (!(ee_dev <= 1.0e-11)) {
    core::log_error(std::string(label) + " energy null violated");
    pass = false;
  }
  if (run_b.steps != run_a.steps) {
    core::log_error(std::string(label) +
                    " pinned-dt step counts differ (dt hook fired below cap)");
    pass = false;
  }
  if (!(2 * run_d.steps >= 3 * run_c.steps)) {
    core::log_error(std::string(label) +
                    " viscous dt limiter did not engage (steps_D < 1.5x steps_C)");
    pass = false;
  }
  core::log_info(std::string(label) + (pass ? " PASSED" : " FAILED"));
  return pass;
#else
  core::log_info(std::string("[SKIP] ") + label + ": python disabled");
  return true;
#endif
}

bool run_braginskii_electron_wave_decay_verify() {
  const char* label = "[verify:braginskii_electron_wave]";
#if defined(TENRYU_ENABLE_PYTHON) && TENRYU_ENABLE_PYTHON
  if (!braginskii_cuda_available("braginskii_electron_wave")) {
    return true;
  }
  braginskii_ensure_python();
  const std::string deck = "examples/verification/braginskii_wave.py";
  const double kGammaGas = 5.0 / 3.0;
  const double kDomainL = 1.0;
  const double kSoundSpeed = 1.0e5;
  const double kRho0 = 1.0;
  const double kEtaConst = 2.28e3;
  const double kPi = 3.14159265358979323846;
  const double t_run = 4.0 * 2.0 * kDomainL / kSoundSpeed;
  const double k_wave = kPi / kDomainL;
  const double gamma_th = (2.0 / 3.0) * (kEtaConst / kRho0) * k_wave * k_wave;

  const WaveRunResult off =
      braginskii_run_wave(deck, false, 0.0, t_run, kGammaGas, 2, label);
  // species="electron": same eta_eff = eta_const (momentum physics
  // identical to the ion constant run), heating routed to the electron
  // channel; species="both": eta_i = eta_e = eta_const/2.
  setenv("TENRYU_BRAG_SPECIES", "electron", 1);
  const WaveRunResult on_e =
      braginskii_run_wave(deck, true, kEtaConst, t_run, kGammaGas, 2, label);
  setenv("TENRYU_BRAG_SPECIES", "both", 1);
  const WaveRunResult on_b =
      braginskii_run_wave(deck, true, kEtaConst, t_run, kGammaGas, 2, label);
  if (!off.ok || !on_e.ok || !on_b.ok) {
    core::log_error(std::string(label) + " FAILED (run error)");
    return false;
  }
  const double ratio_e = (on_e.gamma_fit - off.gamma_fit) / gamma_th;
  const double ratio_b = (on_b.gamma_fit - off.gamma_fit) / gamma_th;
  core::log_info(std::string(label) + " gamma_e=" + brag_fmt(on_e.gamma_fit) +
                 " gamma_b=" + brag_fmt(on_b.gamma_fit) +
                 " gamma_off=" + brag_fmt(off.gamma_fit) +
                 " gamma_th=" + brag_fmt(gamma_th) +
                 " ratio_e=" + brag_fmt(ratio_e) +
                 " ratio_b=" + brag_fmt(ratio_b));
  core::log_info(std::string(label) +
                 " E_drift_e=" + brag_fmt(on_e.e_total_drift_rel) +
                 " E_drift_b=" + brag_fmt(on_b.e_total_drift_rel));
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
  if (!(on_e.e_total_drift_rel <= 5.0e-9)) {
    core::log_error(std::string(label) +
                    " electron viscous total-energy drift too large");
    pass = false;
  }
  if (!(on_b.e_total_drift_rel <= 5.0e-9)) {
    core::log_error(std::string(label) +
                    " both viscous total-energy drift too large");
    pass = false;
  }
  // eta_eff is species-invariant by construction: the two fits must agree
  // far inside the gate tolerance.
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

bool run_braginskii_regime_map_verify() {
  const char* label = "[verify:braginskii_regime_map]";
#if defined(TENRYU_ENABLE_PYTHON) && TENRYU_ENABLE_PYTHON
  if (!braginskii_cuda_available("braginskii_regime_map")) {
    return true;
  }
  braginskii_ensure_python();
  struct RegimeCase {
    const char* deck;
    const char* name;
    // in-memory assertions on braginskii::compute_history_diagnostics
    double ratio_min_gt;   // require ratio_min >  this (0 = skip)
    double ratio_max_lt;   // require ratio_max <  this (0 = skip)
    int want_all;          // 0: e_dom == active, 1: i_dom == active,
                           // 2: mixed == active
  };
  const RegimeCase cases[] = {
      {"examples/verification/braginskii_regime_edom.py",
       "braginskii_regime_edom", 50.0, 0.0, 0},
      {"examples/verification/braginskii_regime_idom.py",
       "braginskii_regime_idom", 0.0, 0.05, 1},
      {"examples/verification/braginskii_regime_mixed.py",
       "braginskii_regime_mixed", 0.2, 5.0, 2},
  };
  bool pass = true;
  for (const RegimeCase& rc : cases) {
    core::Config cfg;
    core::State state = braginskii_load_state(rc.deck, cfg);
    coupling::Driver driver;
    driver.run(state, cfg);
    const auto diag =
        tenryu::hydro::braginskii::compute_history_diagnostics(state, cfg);
    core::log_info(std::string(label) + " " + rc.name +
                   " R[min,geo,max]=[" + brag_fmt(diag.ratio_min) + "," +
                   brag_fmt(diag.ratio_geomean_masswt) + "," +
                   brag_fmt(diag.ratio_max) + "] counts e/i/mix/act=" +
                   std::to_string(diag.n_cells_e_dom) + "/" +
                   std::to_string(diag.n_cells_i_dom) + "/" +
                   std::to_string(diag.n_cells_mixed) + "/" +
                   std::to_string(diag.n_cells_active));
    if (!diag.valid || diag.n_cells_active <= 0) {
      core::log_error(std::string(label) + " " + rc.name +
                      " diagnostics invalid/empty");
      pass = false;
      continue;
    }
    if (rc.ratio_min_gt > 0.0 && !(diag.ratio_min > rc.ratio_min_gt)) {
      core::log_error(std::string(label) + " " + rc.name + " ratio_min");
      pass = false;
    }
    if (rc.ratio_max_lt > 0.0 && !(diag.ratio_max < rc.ratio_max_lt)) {
      core::log_error(std::string(label) + " " + rc.name + " ratio_max");
      pass = false;
    }
    const int want =
        (rc.want_all == 0)
            ? diag.n_cells_e_dom
            : ((rc.want_all == 1) ? diag.n_cells_i_dom : diag.n_cells_mixed);
    if (want != diag.n_cells_active) {
      core::log_error(std::string(label) + " " + rc.name +
                      " regime count != active count");
      pass = false;
    }
#if TENRYU_ENABLE_HDF5
    // End-to-end plumbing: the history file must carry the ratio series and
    // its last sample must match the in-memory value (static deck). The
    // history file lands in <Output.directory>/results/<case>_history.h5
    // (OutputManager::init appends "results"; HistoryWriter::init appends
    // "<case_name>_history.h5").
    const std::string h5_path = cfg.output.directory + "/results/" +
                                std::string(rc.name) + "_history.h5";
    const std::vector<double> series = braginskii_read_h5_series(
        h5_path, "/diagnostics/plasma_viscosity_history/ratio_max");
    if (series.empty()) {
      core::log_error(std::string(label) + " " + rc.name +
                      " history dataset missing: " + h5_path);
      pass = false;
    } else if (!(std::fabs(series.back() - diag.ratio_max) <=
                 0.1 * diag.ratio_max)) {
      core::log_error(std::string(label) + " " + rc.name +
                      " history ratio_max mismatch: file=" +
                      brag_fmt(series.back()) + " mem=" +
                      brag_fmt(diag.ratio_max));
      pass = false;
    }
#endif
  }
  core::log_info(std::string(label) + (pass ? " PASSED" : " FAILED"));
  return pass;
#else
  core::log_info(std::string("[SKIP] ") + label + ": python disabled");
  return true;
#endif
}

}  // namespace tenryu::drivers
