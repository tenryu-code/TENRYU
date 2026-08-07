#include "drivers/verify_conduction_eigenmode_1d.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <vector>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/conduction.cuh"
#include "hydro/eos_context.hpp"
#include "hydro/per_material_eos_project.cuh"
#include "mesh/mesh.hpp"
#include "parallel/partition.hpp"

namespace tenryu::drivers {
namespace {

// Constant spellings match src/hydro/conduction.cu (NOT core::constants): the
// conduction kernels' ideal-gas cv fallback uses these literals and the
// analytic chi must reproduce that cv bit-for-bit.
constexpr double kEvToErg = 1.6022e-12;
constexpr double kProtonMass = 1.6726219e-24;

constexpr double kT0 = 10.0;           // [eV] mode pedestal
constexpr double kAmp = 1.0;           // [eV] mode amplitude
constexpr double kRadius = 1.0;        // [cm] domain size
constexpr double kKappaTest = 1.0e10;  // [erg/(cm s eV)] fixed conductivity
constexpr double kTEnd = 2.0e-2;       // [s]

// First zero-flux eigenvalues k*R (runtime self-checked):
constexpr double kPlanarKR = 3.141592653589793238462643383279502884;
// first positive zero of J1; |J1(kCylKR)| = 7.4e-17 in double
constexpr double kCylKR = 3.831705970207512315614435886308160766;
// first positive root of tan(x) = x; |sin - x cos| = 7.7e-16 in double
constexpr double kSphKR = 4.493409457909064175307880927276469;

std::string fmt(const double v) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(16) << v;
  return oss.str();
}

int geometry_want_code(const std::string& geometry) {
  return (geometry == "cylindrical") ? 1 : ((geometry == "planar") ? 2 : 0);
}

double mode_kr(const int geom_code) {
  if (geom_code == 1) {
    return kCylKR;
  }
  if (geom_code == 2) {
    return kPlanarKR;
  }
  return kSphKR;
}

double mode_phi(const int geom_code, const double x) {
  if (geom_code == 1) {
    return std::cyl_bessel_j(0.0, x);
  }
  if (geom_code == 2) {
    return std::cos(x);
  }
  return (x == 0.0) ? 1.0 : (std::sin(x) / x);
}

bool eigenvalue_self_check(const std::string& label, const int geom_code) {
  if (geom_code == 1) {
    const double resid = std::cyl_bessel_j(1.0, kCylKR);
    if (!(std::abs(resid) <= 1.0e-14)) {
      core::log_error("[verify:" + label +
                      "] J1(j11) eigenvalue self-check failed: " + fmt(resid));
      return false;
    }
  } else if (geom_code == 0) {
    const double resid = std::sin(kSphKR) - kSphKR * std::cos(kSphKR);
    if (!(std::abs(resid) <= 1.0e-13)) {
      core::log_error("[verify:" + label +
                      "] tan(x)=x eigenvalue self-check failed: " + fmt(resid));
      return false;
    }
  }
  return true;
}

double fit_loglog_slope(const std::vector<double>& h, const std::vector<double>& e) {
  double sx = 0.0;
  double sy = 0.0;
  double sxx = 0.0;
  double sxy = 0.0;
  const double n = static_cast<double>(h.size());
  for (std::size_t i = 0; i < h.size(); ++i) {
    const double x = std::log(h[i]);
    const double y = std::log(e[i]);
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

struct CaseMetrics {
  bool ok = false;
  double l2 = 0.0;
  double e_rel = 0.0;
  bool pass_energy = false;
};

CaseMetrics run_case(const std::string& label,
                     const std::string& geometry,
                     const std::string& solver,
                     const bool per_material,
                     const int nr,
                     const double dt_scale = 1.0) {
  CaseMetrics out;
  const int want_geom = geometry_want_code(geometry);

  core::Config cfg;
  cfg.main.dim = 1;
  cfg.main.dimension = "1D_SPH";
  cfg.main.two_temperature = true;
  cfg.main.t_end = kTEnd;
  cfg.mesh.nr = nr;
  cfg.mesh.nz = 1;
  cfg.mesh.r_min = 0.0;
  cfg.mesh.r_max = kRadius;
  cfg.mesh.grid_type_r = "uniform";
  cfg.mesh.geometry_1d = geometry;
  cfg.radiation.groups = 0;
  cfg.numerics.conduction.enabled = true;
  cfg.numerics.conduction.solver = solver;
  cfg.numerics.conduction.test_kappa = kKappaTest;
  cfg.numerics.conduction.sts_damping = 0.01;
  cfg.numerics.conduction.sts_max_stages = 40;
  cfg.numerics.dt.cfl_cond = 0.25;
  if (per_material) {
    cfg.numerics.materials.per_material_conservation_enabled = true;
  }
  core::Config::MaterialsConfig::MatDef mat;
  mat.name = "fuel";
  mat.A = 1.0;
  mat.Z = 1.0;
  cfg.materials.materials = {mat};
  if (per_material) {
    core::Config::MaterialsConfig::MatDef mat_b = mat;
    mat_b.name = "fuel_b";
    cfg.materials.materials.push_back(mat_b);
  }

  auto state = core::State::allocate(cfg);
  state.mesh = mesh::create_mesh(cfg, state);
  state.mesh.recompute_geometry();
  state.vol = state.mesh.cell_vol;

  // BUG-10 binding assert: fail fast if the geometry did not reach the mesh.
  if (state.mesh.geometry_code != want_geom) {
    core::log_error("[verify:" + label + "] geometry_code=" +
                    std::to_string(state.mesh.geometry_code) + " want=" +
                    std::to_string(want_geom) +
                    " (State::allocate/create_mesh did not bind Mesh.geometry_1d)");
    return out;
  }

  const int n = nr;
  std::vector<double> xr(static_cast<std::size_t>(n) + 1, 0.0);
  state.x_r.copy_to_host(xr.data());
  std::vector<double> vol(static_cast<std::size_t>(n), 0.0);
  state.vol.copy_to_host(vol.data());

  const double gamma = cfg.materials.materials.front().ideal_gas_gamma;
  const double cv_e = 1.0 * kEvToErg / (1.0 * kProtonMass * (gamma - 1.0));
  const double chi = kKappaTest / (1.0 * cv_e);
  const double kk = mode_kr(want_geom) / kRadius;

  std::vector<double> rho(static_cast<std::size_t>(n), 1.0);
  std::vector<double> zbar(static_cast<std::size_t>(n), 1.0);
  std::vector<double> Te(static_cast<std::size_t>(n), 0.0);
  std::vector<double> mass(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    const std::size_t c = static_cast<std::size_t>(i);
    const double rc = 0.5 * (xr[c] + xr[c + 1]);
    Te[c] = kT0 + kAmp * mode_phi(want_geom, kk * rc);
    mass[c] = rho[c] * vol[c];
  }
  state.rho.copy_from_host(rho);
  state.zbar.copy_from_host(zbar);
  state.Te.copy_from_host(Te);
  state.mass.copy_from_host(mass);

  hydro::HydroEOSContext eos_ctx;
  if (per_material) {
    const int n_mat = 2;
    std::vector<double> vf(static_cast<std::size_t>(n * n_mat), 0.5);
    std::vector<double> mass_m(static_cast<std::size_t>(n * n_mat), 0.0);
    std::vector<double> Ee_m(static_cast<std::size_t>(n * n_mat), 0.0);
    std::vector<double> Ei_m(static_cast<std::size_t>(n * n_mat), 0.0);
    std::vector<double> ee(static_cast<std::size_t>(n), 0.0);
    std::vector<double> ei(static_cast<std::size_t>(n), 0.0);
    for (int c = 0; c < n; ++c) {
      for (int m = 0; m < n_mat; ++m) {
        const std::size_t idx = static_cast<std::size_t>(c * n_mat + m);
        const std::size_t cc = static_cast<std::size_t>(c);
        mass_m[idx] = 0.5 * rho[cc] * vol[cc];
        Ee_m[idx] = mass_m[idx] * cv_e * Te[cc];
        Ei_m[idx] = mass_m[idx] * (kEvToErg / (1.0 * kProtonMass * (gamma - 1.0))) * kT0;
        ee[cc] += Ee_m[idx];
        ei[cc] += Ei_m[idx];
      }
      const std::size_t cc = static_cast<std::size_t>(c);
      ee[cc] /= mass[cc];
      ei[cc] /= mass[cc];
    }
    state.volFrac.copy_from_host(vf);
    state.mass_per_material.copy_from_host(mass_m);
    state.Ee_per_material.copy_from_host(Ee_m);
    state.Ei_per_material.copy_from_host(Ei_m);
    state.ee.copy_from_host(ee);
    state.ei.copy_from_host(ei);
    eos_ctx.initialize(cfg);
    hydro::per_material::refresh_per_material_derived_cell_fields(state, cfg,
                                                                  &eos_ctx, true);
  }

  // Snapshot AFTER the per-material refresh so the analytic reference and the
  // energy ledger see exactly the state the operator starts from.
  std::vector<double> Te0(static_cast<std::size_t>(n), 0.0);
  state.Te.copy_to_host(Te0.data());
  long double e_initial = 0.0L;
  for (int i = 0; i < n; ++i) {
    const std::size_t c = static_cast<std::size_t>(i);
    e_initial += static_cast<long double>(rho[c]) * static_cast<long double>(cv_e) *
                 static_cast<long double>(Te0[c]) * static_cast<long double>(vol[c]);
  }

  const double dx = (cfg.mesh.r_max - cfg.mesh.r_min) / static_cast<double>(nr);
  const double dt_fixed = dt_scale * 0.20 * dx * dx / chi;
  double t = 0.0;
  int clamp_total = 0;
  while (t < cfg.main.t_end) {
    const double dt = std::min(dt_fixed, cfg.main.t_end - t);
    const auto step =
        hydro::conduction_step(state, dt, cfg, parallel::PartitionInfo{}, nullptr,
                               nullptr, per_material ? &eos_ctx : nullptr);
    clamp_total += step.clamp_count;
    t += dt;
  }

  std::vector<double> Te1(static_cast<std::size_t>(n), 0.0);
  state.Te.copy_to_host(Te1.data());
  const double amp_t = kAmp * std::exp(-chi * kk * kk * cfg.main.t_end);
  long double accum = 0.0L;
  long double e_final = 0.0L;
  for (int i = 0; i < n; ++i) {
    const std::size_t c = static_cast<std::size_t>(i);
    const double rc = 0.5 * (xr[c] + xr[c + 1]);
    const double exact = kT0 + amp_t * mode_phi(want_geom, kk * rc);
    const double diff = Te1[c] - exact;
    accum += static_cast<long double>(diff) * static_cast<long double>(diff);
    e_final += static_cast<long double>(rho[c]) * static_cast<long double>(cv_e) *
               static_cast<long double>(Te1[c]) * static_cast<long double>(vol[c]);
  }
  out.l2 = std::sqrt(static_cast<double>(accum / static_cast<long double>(n)));
  out.e_rel = std::abs(static_cast<double>(e_final - e_initial)) /
              std::max(std::abs(static_cast<double>(e_initial)), 1.0e-300);
  out.pass_energy = (out.e_rel <= 1.0e-14);
  out.ok = true;
  core::log_info("[verify:" + label + "] nr=" + std::to_string(nr) + " geometry=" +
                 geometry + " solver=" + solver +
                 std::string(per_material ? " per_material=1" : "") +
                 " L2=" + fmt(out.l2) + " E_rel=" + fmt(out.e_rel) +
                 " clamps=" + std::to_string(clamp_total) +
                 " energy_pass=" + std::string(out.pass_energy ? "true" : "false"));
  return out;
}

}  // namespace

bool run_conduction_eigenmode_1d_order_study(const std::string& label,
                                             const std::string& geometry) {
  const int want_geom = geometry_want_code(geometry);
  if (!eigenvalue_self_check(label, want_geom)) {
    core::log_error("[verify:" + label + "] FAILED");
    return false;
  }
  const std::vector<int> resolutions = {50, 100, 200, 400};
  std::vector<double> h;
  std::vector<double> l2;
  bool all_energy = true;
  for (const int nr : resolutions) {
    const auto m = run_case(label, geometry, "sts", false, nr);
    if (!m.ok) {
      core::log_error("[verify:" + label + "] FAILED");
      return false;
    }
    h.push_back(kRadius / static_cast<double>(nr));
    l2.push_back(m.l2);
    all_energy = all_energy && m.pass_energy;
  }
  const double order = fit_loglog_slope(h, l2);
  const bool pass_order = (order >= 1.8 && order <= 2.2);
  const bool pass = pass_order && all_energy;
  core::log_info("[verify:" + label + "] final order=" + fmt(order) + ", order_pass=" +
                 std::string(pass_order ? "true" : "false") + ", energy_pass=" +
                 std::string(all_energy ? "true" : "false"));
  if (!pass) {
    core::log_error("[verify:" + label + "] FAILED");
  } else {
    core::log_info("[verify:" + label + "] PASSED");
  }
  return pass;
}

bool run_conduction_eigenmode_1d_solver_pair(const std::string& label,
                                             const std::string& geometry) {
  const int want_geom = geometry_want_code(geometry);
  if (!eigenvalue_self_check(label, want_geom)) {
    core::log_error("[verify:" + label + "] FAILED");
    return false;
  }
  const auto sts = run_case(label, geometry, "sts", false, 100);
  // Backward Euler at dt = 0.2 dx^2/chi carries an O(dt) amplitude-error
  // constant of the same order as the O(dx^2) spatial error (that scaling is
  // what makes the order study work), and its sign composes differently from
  // the 2-stage Chebyshev STS ladder (measured ratio ~5 at equal dt). Run the
  // implicit solve at dt/8 so its time error is subdominant; the ratio bound 4
  // then pins the SHARED spatial operator (a broken geometry factor in either
  // solver path shifts L2 by orders of magnitude, far past this bound).
  const auto imp = run_case(label, geometry, "implicit", false, 100, 0.125);
  if (!sts.ok || !imp.ok) {
    core::log_error("[verify:" + label + "] FAILED");
    return false;
  }
  const bool pass_ratio = (sts.l2 > 0.0) && (imp.l2 <= 4.0 * sts.l2);
  const bool pass = pass_ratio && sts.pass_energy && imp.pass_energy;
  core::log_info("[verify:" + label + "] L2_sts=" + fmt(sts.l2) +
                 " L2_implicit=" + fmt(imp.l2) + " ratio_pass=" +
                 std::string(pass_ratio ? "true" : "false"));
  if (!pass) {
    core::log_error("[verify:" + label + "] FAILED");
  } else {
    core::log_info("[verify:" + label + "] PASSED");
  }
  return pass;
}

bool run_conduction_eigenmode_1d_per_material_pair(const std::string& label,
                                                   const std::string& geometry) {
  const int want_geom = geometry_want_code(geometry);
  if (!eigenvalue_self_check(label, want_geom)) {
    core::log_error("[verify:" + label + "] FAILED");
    return false;
  }
  const auto single = run_case(label, geometry, "sts", false, 100);
  const auto pm = run_case(label, geometry, "sts", true, 100);
  if (!single.ok || !pm.ok) {
    core::log_error("[verify:" + label + "] FAILED");
    return false;
  }
  const double ratio = pm.l2 / std::max(single.l2, 1.0e-300);
  const bool pass_ratio = (ratio >= 0.5 && ratio <= 1.5);
  const bool pass = pass_ratio && single.pass_energy && pm.pass_energy;
  core::log_info("[verify:" + label + "] L2_single=" + fmt(single.l2) +
                 " L2_per_material=" + fmt(pm.l2) + " ratio=" + fmt(ratio) +
                 " ratio_pass=" + std::string(pass_ratio ? "true" : "false"));
  if (!pass) {
    core::log_error("[verify:" + label + "] FAILED");
  } else {
    core::log_info("[verify:" + label + "] PASSED");
  }
  return pass;
}

}  // namespace tenryu::drivers
