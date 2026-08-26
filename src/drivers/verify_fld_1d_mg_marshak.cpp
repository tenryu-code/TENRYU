#include "drivers/verify_fld_1d_mg_marshak.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/constants.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "materials/ionmix_reader.hpp"
#include "radiation/fld_1d_gpu.cuh"
#include "radiation/groups.cuh"
#include "radiation/planck_table.cuh"

// W-MG (multigroup planar marshak gates, child session mg-marshak).
//
// Anti-tautology design (CLAUDE.md section 4, "when verification works,
// suspect the verification"): the expected Planck group fractions b_g are NOT
// taken from Groups::planck_fraction_raw (fixed n=256 composite Simpson,
// src/radiation/groups.cu:29-45) nor from PlanckTable (linear-in-T table of
// the same integrals). The gate carries TWO independent evaluations of
// integral_x^inf t^3/(e^t - 1) dt — an exponential series (integration by
// parts of the Bose kernel) and an adaptive Simpson with its own refinement
// — cross-checks them against each other, and only then uses them as the
// reference. The renormalize-to-1-over-configured-groups semantics of
// PlanckTable::build (src/radiation/planck_table.cu:140-156) is REPLICATED
// (semantics, not numerics) so the reference matches the code's b_g
// convention.
//
// Group bounds are capped at 1500 eV (x <= 30 at T_eq = 50 eV) deliberately:
// the code-side fixed-256-point Simpson has composite error ~(h^4/180)|f'''|
// which reaches ~5e-4 RELATIVE on a wide catch-all group like [700, 1e4] eV
// (x-width 186, h = 0.727). With the capped bounds every group keeps the
// table error below ~1e-7 so a 1e-6 gate tolerance genuinely verifies the
// b_g implementation instead of its wide-group truncation artifacts.

namespace tenryu::drivers {
namespace {

constexpr double kPi4Over15 = 6.4939394022668291;  // pi^4 / 15
constexpr double kPiMgPf = 3.141592653589793238462643383279502884;

std::string mg_format_double(const double value) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(16) << value;
  return oss.str();
}

bool mg_verify_cuda_available(const char* verify_name) {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count <= 0) {
    std::cout << "[SKIP] " << verify_name << ": CUDA not available\n";
    return false;
  }
  return true;
}

template <typename Tag>
std::vector<double> mg_copy_field_to_host(const core::Field1D<Tag>& field) {
  std::vector<double> host(field.size(), 0.0);
  field.copy_to_host(host.data());
  return host;
}

template <typename Tag>
void mg_copy_field_from_host(core::Field1D<Tag>& field,
                             const std::vector<double>& host) {
  TENRYU_ASSERT(field.size() == host.size(),
                "Field size mismatch in mg host copy");
  field.copy_from_host(host.data());
}

// integral_x^inf t^3/(e^t - 1) dt.
// x >= 0.7: exponential series sum_{k>=1} e^{-kx} (x^3/k + 3x^2/k^2 +
//           6x/k^3 + 6/k^4) — geometric convergence.
// x <  0.7: pi^4/15 minus the Bernoulli/Taylor expansion of
//           integral_0^x t^3/(e^t - 1) dt = x^3/3 - x^4/8 + x^5/60
//           - x^7/5040 + x^9/272160 - x^11/13305600 + x^13/622702080
//           (truncation < 2e-13 absolute at x = 0.7).
double mg_planck_integral_above_series(const double x) {
  if (!(x > 0.0)) {
    return kPi4Over15;
  }
  if (x < 0.7) {
    const double x2 = x * x;
    const double x3 = x2 * x;
    const double x4 = x3 * x;
    const double x5 = x4 * x;
    const double x7 = x5 * x2;
    const double x9 = x7 * x2;
    const double x11 = x9 * x2;
    const double x13 = x11 * x2;
    const double below = x3 / 3.0 - x4 / 8.0 + x5 / 60.0 - x7 / 5040.0 +
                         x9 / 272160.0 - x11 / 13305600.0 +
                         x13 / 622702080.0;
    return kPi4Over15 - below;
  }
  double sum = 0.0;
  for (int k = 1; k <= 2000; ++k) {
    const double kk = static_cast<double>(k);
    const double kx = kk * x;
    if (kx > 745.0) {
      break;
    }
    const double term =
        std::exp(-kx) * (x * x * x / kk + 3.0 * x * x / (kk * kk) +
                         6.0 * x / (kk * kk * kk) +
                         6.0 / (kk * kk * kk * kk));
    sum += term;
    if (term < 1.0e-17 * sum) {
      break;
    }
  }
  return sum;
}

double mg_planck_kernel(const double x) {
  if (!(x > 0.0) || x > 700.0) {
    return 0.0;
  }
  return x * x * x / std::expm1(x);
}

double mg_adaptive_simpson_rec(const double a,
                               const double b,
                               const double fa,
                               const double fm,
                               const double fb,
                               const double whole,
                               const double tol,
                               const int depth) {
  const double m = 0.5 * (a + b);
  const double lm = 0.5 * (a + m);
  const double rm = 0.5 * (m + b);
  const double flm = mg_planck_kernel(lm);
  const double frm = mg_planck_kernel(rm);
  const double left = (m - a) / 6.0 * (fa + 4.0 * flm + fm);
  const double right = (b - m) / 6.0 * (fm + 4.0 * frm + fb);
  const double delta = left + right - whole;
  if (depth <= 0 || std::abs(delta) <= 15.0 * tol) {
    return left + right + delta / 15.0;
  }
  return mg_adaptive_simpson_rec(a, m, fa, flm, fm, left, 0.5 * tol,
                                 depth - 1) +
         mg_adaptive_simpson_rec(m, b, fm, frm, fb, right, 0.5 * tol,
                                 depth - 1);
}

// Independent cross-check integrator (adaptive refinement, unlike the code's
// fixed-n Simpson). Upper cut at x = 80: the tail beyond is < 1e-29 absolute.
double mg_planck_integral_adaptive(const double x_lo, const double x_hi) {
  const double a = std::max(x_lo, 0.0);
  const double b = std::min(x_hi, 80.0);
  if (!(b > a)) {
    return 0.0;
  }
  const double fa = mg_planck_kernel(a);
  const double fb = mg_planck_kernel(b);
  const double m = 0.5 * (a + b);
  const double fm = mg_planck_kernel(m);
  const double whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
  return mg_adaptive_simpson_rec(a, b, fa, fm, fb, whole, 1.0e-13, 40);
}

struct MgReference {
  std::vector<double> b_ref;      // renormalized fractions (sum == 1)
  std::vector<double> I_g;        // raw integral over the group [x units]
  double raw_sum = 0.0;           // sum of raw fractions before renorm
  double max_internal_diff = 0.0; // series-vs-adaptive relative disagreement
};

MgReference mg_build_reference(const std::vector<double>& bounds_eV,
                               const double T_eV) {
  const int n_groups = static_cast<int>(bounds_eV.size()) - 1;
  MgReference ref;
  ref.b_ref.assign(static_cast<std::size_t>(n_groups), 0.0);
  ref.I_g.assign(static_cast<std::size_t>(n_groups), 0.0);
  for (int g = 0; g < n_groups; ++g) {
    const double x_lo = bounds_eV[static_cast<std::size_t>(g)] / T_eV;
    const double x_hi = bounds_eV[static_cast<std::size_t>(g + 1)] / T_eV;
    const double I_series = mg_planck_integral_above_series(x_lo) -
                            mg_planck_integral_above_series(x_hi);
    const double I_adaptive = mg_planck_integral_adaptive(x_lo, x_hi);
    if (I_series > 1.0e-20) {
      ref.max_internal_diff =
          std::max(ref.max_internal_diff,
                   std::abs(I_series - I_adaptive) / I_series);
    }
    ref.I_g[static_cast<std::size_t>(g)] = I_series;
    ref.raw_sum += I_series / kPi4Over15;
  }
  TENRYU_ASSERT(ref.raw_sum > 0.0,
                "mg reference: raw Planck fraction sum must be positive");
  // Replicates the renormalize-to-1 semantics of PlanckTable::build
  // (src/radiation/planck_table.cu:140-156): tails outside the configured
  // bounds are redistributed proportionally over the represented groups.
  for (int g = 0; g < n_groups; ++g) {
    ref.b_ref[static_cast<std::size_t>(g)] =
        (ref.I_g[static_cast<std::size_t>(g)] / kPi4Over15) / ref.raw_sum;
  }
  return ref;
}

// Closed-form group Planck mean of the freq_dep_marshak opacity
// sigma(nu, T) = (27e9 / nu^3) * (1 - e^{-nu/T})  [cm^-1]
// (src/materials/opacity_eval.cuh:9-19). The Planck-weighted numerator
// collapses analytically: sigma * w_P = 27e9 * e^{-x}, so
//   <sigma>_P,g = 27e9 * (e^{-x_lo} - e^{-x_hi}) / (T^3 * I_g)
// with I_g the same independent Planck integral used for b_g. Nothing of the
// code's 128-point log-Simpson group averaging
// (src/materials/opacity_eval.cuh:139-192) is reused.
double mg_freqdep_sigma_planck_ref(const double lo_eV,
                                   const double hi_eV,
                                   const double T_eV,
                                   const double I_g) {
  const double x_lo = lo_eV / T_eV;
  const double x_hi = hi_eV / T_eV;
  const double numerator = 27.0e9 * (std::exp(-x_lo) - std::exp(-x_hi));
  return numerator / (T_eV * T_eV * T_eV * std::max(I_g, 1.0e-300));
}

core::Config mg_make_config(const int n_cells,
                            const std::vector<double>& bounds_eV,
                            const std::string& opacity_model,
                            const double kappa,
                            const double cv_e) {
  core::Config cfg;
  cfg.main.dimension = "1D_SPH";
  cfg.main.dim = 1;
  cfg.mesh.nr = n_cells;
  cfg.mesh.nz = 1;
  cfg.mesh.geometry_1d = "planar";
  cfg.radiation.mode = core::RadiationMode::MultigroupDiffusion;
  cfg.radiation.groups = static_cast<int>(bounds_eV.size()) - 1;
  cfg.radiation.group_bounds_eV = bounds_eV;
  cfg.radiation.imc.enabled = false;
  cfg.radiation.ddmc.enabled = false;
  cfg.radiation.multigroup_diffusion.flux_limiter = "levermore_pomraning";
  cfg.radiation.multigroup_diffusion.outer_tol = 1.0e-10;
  cfg.radiation.multigroup_diffusion.max_outer_iterations = 8;
  core::Config::MaterialsConfig::MatDef mat;
  mat.A = 1.0;
  mat.Z = 1.0;
  mat.opacity_model = opacity_model;
  mat.kappa_a_constant = kappa;
  mat.cv_e_override = cv_e;
  cfg.materials.materials.push_back(mat);
  return cfg;
}

void mg_fill_state(core::State& state,
                   const int n_cells,
                   const double dr,
                   const std::vector<double>& rad_E,
                   const double Te,
                   const double cv_e) {
  const std::size_t n = static_cast<std::size_t>(n_cells);
  std::vector<double> node(n + 1U, 0.0);
  for (int i = 0; i <= n_cells; ++i) {
    node[static_cast<std::size_t>(i)] = dr * static_cast<double>(i);
  }
  state.mesh.dim = 1;
  mg_copy_field_from_host(state.x_r, node);
  mg_copy_field_from_host(state.vol, std::vector<double>(n, dr));
  mg_copy_field_from_host(state.rho, std::vector<double>(n, 1.0));
  mg_copy_field_from_host(state.zbar, std::vector<double>(n, 1.0));
  mg_copy_field_from_host(state.Te, std::vector<double>(n, Te));
  mg_copy_field_from_host(state.ee, std::vector<double>(n, cv_e * Te));
  mg_copy_field_from_host(state.Pe, std::vector<double>(n, 1.0e8));
  state.cv_e.reset(n);
  mg_copy_field_from_host(state.cv_e, std::vector<double>(n, cv_e));
  mg_copy_field_from_host(state.rad_E, rad_E);
}

bool mg_check_planar_geometry(const core::State& state,
                              const std::string& label) {
  if (state.mesh.geometry_code != 2) {
    core::log_info("[verify:" + label + "] FAILED (geometry_code=" +
                   std::to_string(state.mesh.geometry_code) +
                   " want=2 — State::allocate did not bind planar geometry)");
    return false;
  }
  return true;
}

// Shared G=8 bounds for both parts. bounds[0] = 1 eV (NOT 0): the
// freq_dep_marshak device group-mean integrates in log(nu) from
// max(lo, 1e-300) — a zero lower bound stretches its 128-point Simpson over
// ~690 nats and destroys the group mean. Upper cap 1500 eV: see file header.
// At T = 50 eV the renormalization tail (below 1 eV ~4.1e-7 raw, above
// 1500 eV ~4.3e-10 raw) is replicated by the reference renormalization.
const std::vector<double>& mg_gate_bounds_eV() {
  static const std::vector<double> bounds = {1.0,   30.0,  60.0,
                                             100.0, 150.0, 220.0,
                                             350.0, 700.0, 1500.0};
  return bounds;
}

// ---- Multigroup-coverage extension: table_nlte fixture plumbing ----

constexpr const char* kMgNlteFixturePath = "tests/data/ionmix_mg_fld_g4.cn4";

// G=4 bounds for the NLTE gates; must match the committed fixture
// (tools/generate_ionmix_mg_fld.py). Same design rules as mg_gate_bounds_eV:
// bounds[0] > 0 and a 1500 eV cap keep the code-side fixed-256 Simpson b_g
// error well below the gate tolerances; every group carries b_g >= 3e-2 at
// T = 50 eV (b_ref ~ {0.0346, 0.3584, 0.5678, 0.0392}).
const std::vector<double>& mg_nlte_bounds_eV() {
  static const std::vector<double> bounds = {1.0, 50.0, 150.0, 400.0, 1500.0};
  return bounds;
}

struct MgNlteFixture {
  bool ok = false;
  std::string error;
  std::vector<double> kappa_g;  // per-group constant [cm^2/g]
};

// Host-side load + integrity check of the committed fixture: bounds must
// equal mg_nlte_bounds_eV() and kappa_PA == kappa_PE == kappa_R must be
// constant over the whole (rho, T) grid per group — that is what makes the
// bilinear log-log interpolation exact at the gate's state points and the
// table Kirchhoff/LTE-consistent. The kappa values are INPUTS to the
// reference algebra (not under test), so reading them from the same file the
// solver loads removes fixture-drift risk without weakening the independent
// b_g / f references.
MgNlteFixture mg_load_nlte_fixture(const int n_groups) {
  MgNlteFixture fx;
  materials::IonmixOpacityData table;
  try {
    table = materials::load_ionmix_opacity(kMgNlteFixturePath);
  } catch (const std::exception& e) {
    fx.error = std::string("fixture load failed: ") + e.what() +
               " (run tools/generate_ionmix_mg_fld.py and run the gate from "
               "the repo root)";
    return fx;
  }
  if (table.ngroups != n_groups) {
    fx.error = "fixture ngroups=" + std::to_string(table.ngroups) +
               " want=" + std::to_string(n_groups);
    return fx;
  }
  if (!table.has_PE) {
    fx.error = "fixture has no kappa_PE block";
    return fx;
  }
  const std::vector<double>& bounds = mg_nlte_bounds_eV();
  if (table.bounds_eV.size() != bounds.size()) {
    fx.error = "fixture bounds size=" + std::to_string(table.bounds_eV.size()) +
               " want=" + std::to_string(bounds.size());
    return fx;
  }
  for (std::size_t i = 0; i < bounds.size(); ++i) {
    const double rel = std::abs(table.bounds_eV[i] - bounds[i]) /
                       std::max(std::abs(bounds[i]), 1.0e-300);
    if (!(rel < 1.0e-12)) {
      fx.error = "fixture bound[" + std::to_string(i) + "]=" +
                 mg_format_double(table.bounds_eV[i]) +
                 " want=" + mg_format_double(bounds[i]);
      return fx;
    }
  }
  fx.kappa_g.assign(static_cast<std::size_t>(n_groups), 0.0);
  for (int g = 0; g < n_groups; ++g) {
    const double k0 = table.kappa_PA[table.flat_index(g, 0, 0)];
    if (!(k0 > 0.0)) {
      fx.error = "fixture kappa[g=" + std::to_string(g) + "] not positive";
      return fx;
    }
    for (int d = 0; d < table.ndens; ++d) {
      for (int t = 0; t < table.ntemp; ++t) {
        const std::size_t i = table.flat_index(g, d, t);
        if (table.kappa_PA[i] != k0 || table.kappa_PE[i] != k0 ||
            table.kappa_R[i] != k0) {
          fx.error = "fixture kappa not per-group-constant/Kirchhoff at g=" +
                     std::to_string(g);
          return fx;
        }
      }
    }
    fx.kappa_g[static_cast<std::size_t>(g)] = k0;
  }
  fx.ok = true;
  return fx;
}

// ---- Mission 3: Su & Olson (1999) picket-fence diffusion reference ----
//
// Semi-analytic evaluator of the paper's diffusion solution, Eq. (41):
//   W(x,tau) = (1/pi) Int_0^inf dk cos(kx) Q1(k)
//              Sum_j (A_j/s_j)(e^{-s_j tau*} - e^{-s_j tau}),
// Q1(k) = sin(k x0)/(k x0), tau* = max(0, tau - tau0), with the three real
// s_j > 0 the negated roots of the dispersion cubic
//   eps^2 s^3 + c2 s^2 + c1 s + c0   (paper Eqs. 26-28)
// and A_j = N(-s_j)/D'(-s_j) with the field-specific numerators of
// Eqs. (21)-(22). The c2/c1/c0 and numerator forms below were RE-DERIVED
// from Eqs. (16)-(17) under the constraint p1 w1 + p2 w2 = 1 and match the
// printed equations (transcription protocol, formula level); the root
// solver is an independent trigonometric implementation, not a copy of the
// paper's Eqs. (30)-(36). Numerical guards: k = 0 is evaluated at 1e-9
// (removable) and the (e^{-s tau*} - e^{-s tau})/s factor uses expm1 to
// survive s -> 0.

struct MgPfParams {
  double w1 = 1.0;
  double w2 = 1.0;
  double p1 = 0.5;
  double p2 = 0.5;
  double eps = 1.0;
  double x0 = 0.5;
  double tau0 = 10.0;
};

struct MgPfTriple {
  double U1 = 0.0;
  double U2 = 0.0;
  double V = 0.0;
};

// Ascending real roots of e2 s^3 + c2 s^2 + c1 s + c0 (all real for the
// picket-fence cubic; trigonometric method).
void mg_pf_cubic_roots(const double e2,
                       const double c2,
                       const double c1,
                       const double c0,
                       double roots[3]) {
  const double a = c2 / e2;
  const double b = c1 / e2;
  const double c = c0 / e2;
  const double p = b - a * a / 3.0;
  const double q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c;
  const double m = 2.0 * std::sqrt(std::max(-p / 3.0, 0.0));
  double cos_arg = 0.0;
  if (m > 0.0) {
    cos_arg = 3.0 * q / (p * m);
    cos_arg = std::min(1.0, std::max(-1.0, cos_arg));
  }
  const double th = std::acos(cos_arg) / 3.0;
  double y[3];
  for (int k = 0; k < 3; ++k) {
    y[k] = m * std::cos(th - 2.0 * kPiMgPf * static_cast<double>(k) / 3.0);
  }
  std::sort(y, y + 3);
  for (int k = 0; k < 3; ++k) {
    roots[k] = y[k] - a / 3.0;
  }
}

// Sum_j (A_j/s_j)(e^{-s_j tau*} - e^{-s_j tau}) for all three fields at one k.
MgPfTriple mg_pf_terms(const double k_in,
                       const double tau,
                       const MgPfParams& pf) {
  const double k = std::max(k_in, 1.0e-9);
  const double k2 = k * k;
  const double w1 = pf.w1;
  const double w2 = pf.w2;
  const double eps = pf.eps;
  const double a1 = k2 / (3.0 * w1);
  const double a2 = k2 / (3.0 * w2);
  const double e2 = eps * eps;
  const double c2 = eps * (w1 + w2 + eps) + eps * (w1 + w2) * k2 / (3.0 * w1 * w2);
  const double c1 = w1 * w2 * (1.0 + eps) +
                    (w1 * w1 + w2 * w2 + eps * w1 + eps * w2) * k2 /
                        (3.0 * w1 * w2) +
                    k2 * k2 / (9.0 * w1 * w2);
  const double c0 = (pf.p1 * w2 + pf.p2 * w1) * k2 / 3.0 +
                    k2 * k2 / (9.0 * w1 * w2);
  double rr[3];
  mg_pf_cubic_roots(e2, c2, c1, c0, rr);
  double sj[3];
  for (int j = 0; j < 3; ++j) {
    sj[j] = std::max(-rr[2 - j], 1.0e-300);  // positive, ascending
  }
  const double tau_star = std::max(0.0, tau - pf.tau0);
  MgPfTriple out;
  for (int j = 0; j < 3; ++j) {
    const double s = -sj[j];
    const double nu1 = pf.p1 * (eps * s * s + (eps + w2 + a2) * s +
                                w1 * w2 + a2);
    const double nu2 = pf.p2 * (eps * s * s + (eps + w1 + a1) * s +
                                w1 * w2 + a1);
    const double nv = eps * s + w1 * w2 +
                      (pf.p1 * w1 * w1 + pf.p2 * w2 * w2) * k2 /
                          (3.0 * w1 * w2);
    // D'(-s_j) = e2 * prod_{i != j} (s_i - s_j): of the three product terms
    // of D'(s), only the one skipping factor j survives at the root.
    double dprime = e2;
    for (int ii = 0; ii < 3; ++ii) {
      if (ii != j) {
        dprime *= (s + sj[ii]);
      }
    }
    // time factor (e^{-s_j tau*} - e^{-s_j tau})/s_j, expm1-safe:
    const double tf = std::exp(-sj[j] * tau_star) *
                      (-std::expm1(-sj[j] * (tau - tau_star))) / sj[j];
    out.U1 += nu1 / dprime * tf;
    out.U2 += nu2 / dprime * tf;
    out.V += nv / dprime * tf;
  }
  return out;
}

// Full evaluator: subrange decomposition k_l = 2 pi l^2/(x + x0) with
// Simpson doubling per subrange (rel 1e-9), truncated when a subrange's
// contribution falls below 1e-9 in all three fields.
MgPfTriple mg_pf_eval(const double x, const double tau, const MgPfParams& pf) {
  MgPfTriple total;
  const double span = x + pf.x0;
  for (int l = 0; l < 64; ++l) {
    const double ka = 2.0 * kPiMgPf * static_cast<double>(l) *
                      static_cast<double>(l) / span;
    const double kb = 2.0 * kPiMgPf * static_cast<double>(l + 1) *
                      static_cast<double>(l + 1) / span;
    auto f = [&](const double k) {
      const double kq = std::max(k, 1.0e-9);
      const double q1 = std::sin(kq * pf.x0) / (kq * pf.x0);
      const MgPfTriple t = mg_pf_terms(k, tau, pf);
      const double ck = std::cos(kq * x);
      MgPfTriple r;
      r.U1 = ck * q1 * t.U1;
      r.U2 = ck * q1 * t.U2;
      r.V = ck * q1 * t.V;
      return r;
    };
    MgPfTriple prev;
    MgPfTriple cur;
    bool have_prev = false;
    for (int n = 8; n <= (1 << 16); n *= 2) {
      const double h = (kb - ka) / static_cast<double>(n);
      MgPfTriple acc = f(ka);
      const MgPfTriple fe = f(kb);
      acc.U1 += fe.U1;
      acc.U2 += fe.U2;
      acc.V += fe.V;
      for (int i = 1; i < n; ++i) {
        const double wgt = (i % 2 == 1) ? 4.0 : 2.0;
        const MgPfTriple fi = f(ka + h * static_cast<double>(i));
        acc.U1 += wgt * fi.U1;
        acc.U2 += wgt * fi.U2;
        acc.V += wgt * fi.V;
      }
      cur.U1 = acc.U1 * h / 3.0;
      cur.U2 = acc.U2 * h / 3.0;
      cur.V = acc.V * h / 3.0;
      if (have_prev) {
        const double d = std::max({std::abs(cur.U1 - prev.U1),
                                   std::abs(cur.U2 - prev.U2),
                                   std::abs(cur.V - prev.V)});
        const double scale = std::max(
            {1.0, std::abs(cur.U1), std::abs(cur.U2), std::abs(cur.V)});
        if (d <= 1.0e-9 * scale) {
          break;
        }
      }
      prev = cur;
      have_prev = true;
    }
    total.U1 += cur.U1;
    total.U2 += cur.U2;
    total.V += cur.V;
    if (l > 2 && std::abs(cur.U1) < 1.0e-9 && std::abs(cur.U2) < 1.0e-9 &&
        std::abs(cur.V) < 1.0e-9) {
      break;
    }
  }
  total.U1 /= kPiMgPf;
  total.U2 /= kPiMgPf;
  total.V /= kPiMgPf;
  return total;
}

// Transcribed anchor values (Su & Olson 1999, Tables 2 and 3; x, tau, U1,
// U2, V). The tables are accurate to at least the fourth decimal digit
// (max abs error O(1e-5) per the paper). The gate REQUIRES the internal
// evaluator to reproduce every anchor to 2e-5 absolute before any code
// comparison happens (transcription protocol, numeric level; prototype
// measured 5.0e-6 max).
struct MgPfAnchor {
  double x;
  double tau;
  double U1;
  double U2;
  double V;
};

const MgPfAnchor kMgPfAnchorsCaseB[] = {
    {0.0, 0.1, 0.03873, 0.04578, 0.00452},
    {0.0, 1.0, 0.16857, 0.23845, 0.20363},
    {0.0, 3.0, 0.30028, 0.44678, 0.66656},
    {0.5, 1.0, 0.14164, 0.14007, 0.11740},
    {1.0, 1.0, 0.09478, 0.02364, 0.01903},
    {1.0, 3.0, 0.21218, 0.09963, 0.14403},
    {0.0, 10.0, 0.54038, 0.91470, 1.65679},
    {0.5, 10.0, 0.50483, 0.71131, 1.28823},
    {5.0, 10.0, 0.10537, 0.02290, 0.05035},
    {0.0, 30.0, 0.18634, 0.27740, 0.55480},
    {1.0, 30.0, 0.18117, 0.25663, 0.51207},
    {10.0, 30.0, 0.03742, 0.02297, 0.04666},
};

const MgPfAnchor kMgPfAnchorsCaseC[] = {
    {0.0, 0.1, 0.01828, 0.04546, 0.00451},
    {0.0, 1.0, 0.06484, 0.23441, 0.20434},
    {1.0, 3.0, 0.10367, 0.08076, 0.10346},
    {0.0, 10.0, 0.20855, 0.93964, 1.74573},
    {0.5, 10.0, 0.20482, 0.70579, 1.29579},
    {0.0, 30.0, 0.06128, 0.35004, 0.71204},
    {1.0, 30.0, 0.06108, 0.29535, 0.59613},
};

// Parametrized picket fixture loader (mirrors mg_load_nlte_fixture but for a
// caller-given path and expected per-group kappas).
MgNlteFixture mg_load_pf_fixture(const char* path,
                                 const double expected_kappa[2]) {
  MgNlteFixture fx;
  materials::IonmixOpacityData table;
  try {
    table = materials::load_ionmix_opacity(path);
  } catch (const std::exception& e) {
    fx.error = std::string("fixture load failed: ") + e.what() +
               " (run tools/generate_ionmix_mg_picket.py from the repo root)";
    return fx;
  }
  if (table.ngroups != 2 || !table.has_PE) {
    fx.error = "picket fixture must be G=2 with PE";
    return fx;
  }
  fx.kappa_g.assign(2U, 0.0);
  for (int g = 0; g < 2; ++g) {
    const double k0 = table.kappa_PA[table.flat_index(g, 0, 0)];
    const double rel = std::abs(k0 - expected_kappa[g]) /
                       std::max(expected_kappa[g], 1.0e-300);
    if (!(rel < 1.0e-12)) {
      fx.error = "picket fixture kappa[g=" + std::to_string(g) + "]=" +
                 mg_format_double(k0) + " want=" +
                 mg_format_double(expected_kappa[g]);
      return fx;
    }
    for (int d = 0; d < table.ndens; ++d) {
      for (int t = 0; t < table.ntemp; ++t) {
        const std::size_t i = table.flat_index(g, d, t);
        if (table.kappa_PA[i] != k0 || table.kappa_PE[i] != k0 ||
            table.kappa_R[i] != k0) {
          fx.error = "picket fixture kappa not constant at g=" +
                     std::to_string(g);
          return fx;
        }
      }
    }
    fx.kappa_g[static_cast<std::size_t>(g)] = k0;
  }
  fx.ok = true;
  return fx;
}

}  // namespace

// Part A. Same construction as the grey
// run_fld_1d_marshak_equilibration_impl (src/drivers/cmd_verify.cpp:7846):
// cold slab (Te0 = 1 eV), constant grey opacity kappa = 50 cm^2/g, rho = 1,
// marshak blackbody drive at T_r = 50 eV on the outer face, fixed
// dt = 1e-10 s. Milne balance forces the plateau at T_r; per-group detailed
// balance (eta_g = c sigma a T^4 b_g vs removal c sigma E_g,
// src/radiation/fld_1d_gpu.cu:337-357,568-569) forces E_g = b_g a T_r^4 for
// ANY positive sigma — so this part verifies the emission/boundary SPECTRUM
// (b_g wiring), not sigma_g (Part B covers that; the fixed point is provably
// sigma-independent).
bool run_fld_1d_mg_planar_marshak_spectrum_verify() {
  const std::string label = "fld_1d_mg_planar_marshak_spectrum";
  if (!mg_verify_cuda_available(label.c_str())) {
    return true;
  }
  constexpr int n = 8;
  const std::vector<double>& bounds = mg_gate_bounds_eV();
  const int n_groups = static_cast<int>(bounds.size()) - 1;
  const double Tr = 50.0;      // eV — drive AND expected plateau
  const double Te0 = 1.0;      // eV — cold initial matter
  const double cv_e = 1.0e10;  // erg/(g*eV)
  const double dr = 0.05;      // cm; tau = rho*kappa*n*dr = 20 (thick)
  const double dt = 1.0e-10;   // s

  core::Config cfg = mg_make_config(n, bounds, "constant", 50.0, cv_e);
  cfg.radiation.multigroup_diffusion.boundary.outer_r = "marshak";
  cfg.radiation.boundary.marshak_Tr_eV = Tr;

  core::State state = core::State::allocate(cfg);
  if (!mg_check_planar_geometry(state, label)) {
    return false;
  }

  const MgReference ref = mg_build_reference(bounds, Tr);
  if (!(ref.max_internal_diff < 1.0e-9)) {
    core::log_info("[verify:" + label +
                   "] FAILED (independent reference integrators disagree: " +
                   mg_format_double(ref.max_internal_diff) + ")");
    return false;
  }

  radiation::Groups groups(cfg.radiation.group_bounds_eV);
  radiation::PlanckTable planck;
  // n_T = 257 log nodes over [0.5, 5000] eV: 50 eV = sqrt(0.5 * 5000) sits
  // exactly on the middle node (odd n_T), so at the plateau the solver
  // consumes node values and the comparison against the analytic reference
  // is not polluted by the table's linear-in-T interpolation. The transient
  // range Te in [1, 50] eV is interior to the table.
  planck.build(groups, 257, 0.5, 5000.0);

  // Pure b_g implementation check (code table vs independent analytic),
  // BEFORE any dynamics: this is the sharp anti-tautology comparison.
  double bg_code_vs_ref = 0.0;
  {
    std::ostringstream oss;
    oss << "[verify:" << label << "] b_g table-vs-ref rel diff @Tr:";
    for (int g = 0; g < n_groups; ++g) {
      const double b_code = planck.interpolate_b_host(g, Tr);
      const double rel =
          std::abs(b_code - ref.b_ref[static_cast<std::size_t>(g)]) /
          std::max(ref.b_ref[static_cast<std::size_t>(g)], 1.0e-300);
      bg_code_vs_ref = std::max(bg_code_vs_ref, rel);
      oss << " g" << g << "=" << mg_format_double(rel);
    }
    core::log_info(oss.str());
  }

  // Planckian initial radiation at Te0 from the INDEPENDENT reference (the
  // code's table is deliberately not used to seed the state).
  const MgReference ref0 = mg_build_reference(bounds, Te0);
  const double E0_total = core::constants::a_eV * Te0 * Te0 * Te0 * Te0;
  std::vector<double> rad_E0(static_cast<std::size_t>(n * n_groups), 0.0);
  for (int c = 0; c < n; ++c) {
    for (int g = 0; g < n_groups; ++g) {
      rad_E0[static_cast<std::size_t>(c * n_groups + g)] =
          E0_total * ref0.b_ref[static_cast<std::size_t>(g)];
    }
  }
  mg_fill_state(state, n, dr, rad_E0, Te0, cv_e);

  const double E_total_target = core::constants::a_eV * Tr * Tr * Tr * Tr;
  int max_steps = 30000;
  const char* max_env = std::getenv("TENRYU_FLD_MG_MARSHAK_DIAG_MAX_STEPS");
  if (max_env != nullptr && max_env[0] != 0) {
    max_steps = std::atoi(max_env);
  }

  // Settling detector: distance to the CODE-TABLE fixed point (what the
  // dynamics converge to; reaches ~1e-10). The analytic reference is NOT
  // used here on purpose — its distance to the plateau is floored by the
  // table's own integration error (~3e-8 on the hardest group), which would
  // make a 1e-8 exit criterion unreachable. All PASS criteria below still
  // compare against the analytic reference only.
  std::vector<double> settle_target(static_cast<std::size_t>(n_groups), 0.0);
  for (int g = 0; g < n_groups; ++g) {
    settle_target[static_cast<std::size_t>(g)] =
        E_total_target * std::max(planck.interpolate_b_host(g, Tr), 1.0e-300);
  }
  double settle_rel = std::numeric_limits<double>::infinity();
  int steps = 0;
  for (; steps < max_steps && settle_rel > 1.0e-8; ++steps) {
    radiation::advance_radiation_step_fld_1d(
        state, cfg, planck, cfg.materials.materials.front(), dt);
    if (steps % 50 != 49) {
      continue;
    }
    const auto out_E = mg_copy_field_to_host(state.rad_E);
    const auto out_Te = mg_copy_field_to_host(state.Te);
    settle_rel = 0.0;
    for (int c = 0; c < n; ++c) {
      for (int g = 0; g < n_groups; ++g) {
        settle_rel = std::max(
            settle_rel,
            std::abs(out_E[static_cast<std::size_t>(c * n_groups + g)] -
                     settle_target[static_cast<std::size_t>(g)]) /
                settle_target[static_cast<std::size_t>(g)]);
      }
    }
    for (const double T : out_Te) {
      settle_rel = std::max(settle_rel, std::abs(T - Tr) / Tr);
    }
  }

  const auto out_E = mg_copy_field_to_host(state.rad_E);
  const auto out_Te_final = mg_copy_field_to_host(state.Te);
  double max_rel = 0.0;
  {
    std::ostringstream oss;
    oss << "[verify:" << label << "] plateau per-group max|E_g/(b_g aT^4)-1|:";
    for (int g = 0; g < n_groups; ++g) {
      double worst = 0.0;
      const double target =
          E_total_target * ref.b_ref[static_cast<std::size_t>(g)];
      for (int c = 0; c < n; ++c) {
        worst = std::max(
            worst,
            std::abs(out_E[static_cast<std::size_t>(c * n_groups + g)] -
                     target) /
                target);
      }
      max_rel = std::max(max_rel, worst);
      oss << " g" << g << "=" << mg_format_double(worst);
    }
    core::log_info(oss.str());
  }
  for (const double T : out_Te_final) {
    max_rel = std::max(max_rel, std::abs(T - Tr) / Tr);
  }
  double sum_rel = 0.0;
  for (int c = 0; c < n; ++c) {
    double sum = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      sum += out_E[static_cast<std::size_t>(c * n_groups + g)];
    }
    sum_rel = std::max(sum_rel,
                       std::abs(sum - E_total_target) / E_total_target);
  }
  const double flux_rel =
      std::abs(state.fld_escaped_step - state.fld_marshak_in_step) /
      std::max(state.fld_marshak_in_step, 1.0e-300);

  const bool pass = max_rel < 1.0e-6 && sum_rel < 1.0e-6 &&
                    flux_rel < 1.0e-6 && bg_code_vs_ref < 1.0e-6;
  core::log_info("[verify:" + label + "] steps=" + std::to_string(steps) +
                 " max_rel=" + mg_format_double(max_rel) +
                 " settle_rel=" + mg_format_double(settle_rel) +
                 " sum_rel=" + mg_format_double(sum_rel) +
                 " flux_rel=" + mg_format_double(flux_rel) +
                 " bg_code_vs_ref=" + mg_format_double(bg_code_vs_ref) +
                 " ref_internal=" + mg_format_double(ref.max_internal_diff));
  core::log_info("[verify:" + label + "] " +
                 std::string(pass ? "PASSED" : "FAILED"));
  return pass;
}

// Part B. freq_dep_marshak opacity (NOT in use_fld_fleck,
// src/radiation/fld_1d_gpu.cu:401-404) so the Fleck blend is not consumed
// and each step is plain backward Euler per group:
//   E^{n+1} = (E^n + dt c sigma_g B_g) / (1 + dt c sigma_g),
// B_g = a Te^4 b_g(Te). Matter is quasi-frozen (cv_e = 1e22: relative Te
// drift ~1e-22 per step), the slab is uniform with reflecting outer face so
// the diffusion term cancels row-wise and sigma_R never matters. One step
// from E = 0 inverts to the measured group opacity
//   sigma_meas,g = E1 / (B_g - E1) / (c dt),
// compared against the closed-form group Planck mean — a DIRECT per-group
// opacity-wiring measurement (a solver that consumed another group's sigma
// moves each E_g at the wrong rate). A second checkpoint at N = 200 steps
// verifies the composed recurrence E_N = B (1 - (1 + w)^{-N}).
bool run_fld_1d_mg_planar_freqdep_relaxation_verify() {
  const std::string label = "fld_1d_mg_planar_freqdep_relaxation";
  if (!mg_verify_cuda_available(label.c_str())) {
    return true;
  }
  constexpr int n = 4;
  const std::vector<double>& bounds = mg_gate_bounds_eV();
  const int n_groups = static_cast<int>(bounds.size()) - 1;
  const double Te0 = 50.0;     // eV, quasi-frozen
  const double cv_e = 1.0e22;  // erg/(g*eV)
  const double dr = 0.05;      // cm
  const double dt = 1.0e-15;   // s; c*sigma*dt spans ~1.9e-3 .. ~49 over g
  constexpr int kStepsTotal = 200;

  core::Config cfg = mg_make_config(n, bounds, "freq_dep_marshak", 0.0, cv_e);
  cfg.radiation.multigroup_diffusion.boundary.outer_r = "reflect";

  core::State state = core::State::allocate(cfg);
  if (!mg_check_planar_geometry(state, label)) {
    return false;
  }

  const MgReference ref = mg_build_reference(bounds, Te0);
  if (!(ref.max_internal_diff < 1.0e-9)) {
    core::log_info("[verify:" + label +
                   "] FAILED (independent reference integrators disagree: " +
                   mg_format_double(ref.max_internal_diff) + ")");
    return false;
  }

  radiation::Groups groups(cfg.radiation.group_bounds_eV);
  radiation::PlanckTable planck;
  planck.build(groups, 257, 0.5, 5000.0);  // Te0 = 50 eV on the middle node

  const double T4 = Te0 * Te0 * Te0 * Te0;
  std::vector<double> sigma_ref(static_cast<std::size_t>(n_groups), 0.0);
  std::vector<double> w_ref(static_cast<std::size_t>(n_groups), 0.0);
  std::vector<double> B_ref(static_cast<std::size_t>(n_groups), 0.0);
  for (int g = 0; g < n_groups; ++g) {
    const std::size_t gs = static_cast<std::size_t>(g);
    sigma_ref[gs] = mg_freqdep_sigma_planck_ref(
        bounds[gs], bounds[gs + 1U], Te0, ref.I_g[gs]);
    w_ref[gs] = core::constants::c_light * sigma_ref[gs] * dt;
    B_ref[gs] = core::constants::a_eV * T4 * ref.b_ref[gs];
  }

  mg_fill_state(state, n, dr,
                std::vector<double>(static_cast<std::size_t>(n * n_groups),
                                    0.0),
                Te0, cv_e);

  // Phase 1: one step, invert for the consumed sigma_g.
  radiation::advance_radiation_step_fld_1d(
      state, cfg, planck, cfg.materials.materials.front(), dt);
  const auto E1 = mg_copy_field_to_host(state.rad_E);
  double sigma_rel_max = 0.0;
  double uniform_rel_max = 0.0;
  {
    std::ostringstream oss;
    oss << "[verify:" << label << "] sigma_meas/sigma_ref-1 per group:";
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t gs = static_cast<std::size_t>(g);
      double e_min = std::numeric_limits<double>::infinity();
      double e_max = 0.0;
      for (int c = 0; c < n; ++c) {
        const double e = E1[static_cast<std::size_t>(c * n_groups + g)];
        e_min = std::min(e_min, e);
        e_max = std::max(e_max, e);
      }
      uniform_rel_max = std::max(
          uniform_rel_max, (e_max - e_min) / std::max(e_max, 1.0e-300));
      const double e0 = E1[gs];  // cell 0
      const double w_meas = e0 / std::max(B_ref[gs] - e0, 1.0e-300);
      const double sigma_meas =
          w_meas / (core::constants::c_light * dt);
      const double rel = std::abs(sigma_meas / sigma_ref[gs] - 1.0);
      sigma_rel_max = std::max(sigma_rel_max, rel);
      oss << " g" << g << "=" << mg_format_double(rel);
    }
    core::log_info(oss.str());
  }

  // Phase 2: continue to kStepsTotal steps, verify the composed recurrence.
  for (int s = 1; s < kStepsTotal; ++s) {
    radiation::advance_radiation_step_fld_1d(
        state, cfg, planck, cfg.materials.materials.front(), dt);
  }
  const auto EN = mg_copy_field_to_host(state.rad_E);
  const auto TeN = mg_copy_field_to_host(state.Te);
  double recur_rel_max = 0.0;
  {
    std::ostringstream oss;
    oss << "[verify:" << label << "] E(N)/E_ref(N)-1 per group:";
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t gs = static_cast<std::size_t>(g);
      const double decay =
          std::pow(1.0 + w_ref[gs], -static_cast<double>(kStepsTotal));
      const double e_ref = B_ref[gs] * (1.0 - decay);
      double worst = 0.0;
      for (int c = 0; c < n; ++c) {
        worst = std::max(
            worst,
            std::abs(EN[static_cast<std::size_t>(c * n_groups + g)] /
                         std::max(e_ref, 1.0e-300) -
                     1.0));
      }
      recur_rel_max = std::max(recur_rel_max, worst);
      oss << " g" << g << "=" << mg_format_double(worst);
    }
    core::log_info(oss.str());
  }
  double te_drift = 0.0;
  for (const double T : TeN) {
    te_drift = std::max(te_drift, std::abs(T - Te0) / Te0);
  }

  const bool pass = sigma_rel_max < 1.0e-5 && recur_rel_max < 1.0e-5 &&
                    uniform_rel_max < 1.0e-12 && te_drift < 1.0e-9;
  core::log_info("[verify:" + label +
                 "] sigma_rel_max=" + mg_format_double(sigma_rel_max) +
                 " recur_rel_max=" + mg_format_double(recur_rel_max) +
                 " uniform_rel_max=" + mg_format_double(uniform_rel_max) +
                 " te_drift=" + mg_format_double(te_drift) +
                 " ref_internal=" + mg_format_double(ref.max_internal_diff));
  core::log_info("[verify:" + label + "] " +
                 std::string(pass ? "PASSED" : "FAILED"));
  return pass;
}

// Multigroup-coverage gate C. Same physics target and check structure as
// run_fld_1d_mg_planar_marshak_spectrum_verify, but every coefficient flows
// through the table_nlte path (evaluate_fld_opacity_and_emission ->
// compute_nlte_coefficients_cuda_with_pe, src/radiation/nlte_coeffs.cu):
// eta_g = sigma_PE,g c a T^4 b_g with table sigma; gray Fleck
// f = 1/(1 + beta c dt sigma_p_em) from the group-summed Planck-mean
// emission opacity (linearized_planck / corrected_fleck are passed false on
// this path, so the formula is exact); per-group removal sigma_PA,g. With
// the Kirchhoff fixture (PA == PE) the fixed point is E_g = b_g a T_r^4 for
// ANY f > 0 and ANY sigma_g > 0 — the plateau is provably insensitive to
// the fleck LAYOUT, which is why the fleck probe below exists. dt = 4e-11
// puts the plateau Fleck parameter at z ~ 2 (f ~ 1/3): the blend machinery
// is genuinely exercised during the approach.
bool run_fld_1d_mg_nlte_marshak_spectrum_verify() {
  const std::string label = "fld_1d_mg_nlte_marshak_spectrum";
  if (!mg_verify_cuda_available(label.c_str())) {
    return true;
  }
  constexpr int n = 8;
  const std::vector<double>& bounds = mg_nlte_bounds_eV();
  const int n_groups = static_cast<int>(bounds.size()) - 1;
  const double Tr = 50.0;      // eV — drive AND expected plateau
  const double Te0 = 1.0;      // eV — cold initial matter
  const double cv_e = 1.0e10;  // erg/(g*eV)
  const double dr = 0.05;      // cm; tau_g = kappa_g*rho*n*dr >= 12 (thick)
  const double dt = 4.0e-11;   // s

  const MgNlteFixture fx = mg_load_nlte_fixture(n_groups);
  if (!fx.ok) {
    core::log_info("[verify:" + label + "] FAILED (" + fx.error + ")");
    return false;
  }

  core::Config cfg = mg_make_config(n, bounds, "table_nlte", 0.0, cv_e);
  cfg.materials.materials.front().opacity_file = kMgNlteFixturePath;
  cfg.radiation.multigroup_diffusion.boundary.outer_r = "marshak";
  cfg.radiation.boundary.marshak_Tr_eV = Tr;

  core::State state = core::State::allocate(cfg);
  if (!mg_check_planar_geometry(state, label)) {
    return false;
  }

  const MgReference ref = mg_build_reference(bounds, Tr);
  if (!(ref.max_internal_diff < 1.0e-9)) {
    core::log_info("[verify:" + label +
                   "] FAILED (independent reference integrators disagree: " +
                   mg_format_double(ref.max_internal_diff) + ")");
    return false;
  }

  radiation::Groups groups(cfg.radiation.group_bounds_eV);
  radiation::PlanckTable planck;
  planck.build(groups, 257, 0.5, 5000.0);  // 50 eV on the middle node

  double bg_code_vs_ref = 0.0;
  {
    std::ostringstream oss;
    oss << "[verify:" << label << "] b_g table-vs-ref rel diff @Tr:";
    for (int g = 0; g < n_groups; ++g) {
      const double b_code = planck.interpolate_b_host(g, Tr);
      const double rel =
          std::abs(b_code - ref.b_ref[static_cast<std::size_t>(g)]) /
          std::max(ref.b_ref[static_cast<std::size_t>(g)], 1.0e-300);
      bg_code_vs_ref = std::max(bg_code_vs_ref, rel);
      oss << " g" << g << "=" << mg_format_double(rel);
    }
    core::log_info(oss.str());
  }

  const MgReference ref0 = mg_build_reference(bounds, Te0);
  const double E0_total = core::constants::a_eV * Te0 * Te0 * Te0 * Te0;
  std::vector<double> rad_E0(static_cast<std::size_t>(n * n_groups), 0.0);
  for (int c = 0; c < n; ++c) {
    for (int g = 0; g < n_groups; ++g) {
      rad_E0[static_cast<std::size_t>(c * n_groups + g)] =
          E0_total * ref0.b_ref[static_cast<std::size_t>(g)];
    }
  }
  mg_fill_state(state, n, dr, rad_E0, Te0, cv_e);

  const double E_total_target = core::constants::a_eV * Tr * Tr * Tr * Tr;
  // z ~ 2 (f ~ 1/3) Fleck-suppressed equilibration self-exits at ~122k steps
  // (measured 2026-07-04); 160k gives ~30% headroom. The first-run 60k budget
  // was the sole cause of the initial FAIL (exponential approach confirmed:
  // 7.2e-3 @40k -> 8.7e-5 @60k -> settle 9.97e-9 @121.9k).
  int max_steps = 160000;
  const char* max_env =
      std::getenv("TENRYU_FLD_MG_NLTE_MARSHAK_DIAG_MAX_STEPS");
  if (max_env != nullptr && max_env[0] != 0) {
    max_steps = std::atoi(max_env);
  }

  std::vector<double> settle_target(static_cast<std::size_t>(n_groups), 0.0);
  for (int g = 0; g < n_groups; ++g) {
    settle_target[static_cast<std::size_t>(g)] =
        E_total_target * std::max(planck.interpolate_b_host(g, Tr), 1.0e-300);
  }
  double settle_rel = std::numeric_limits<double>::infinity();
  int steps = 0;
  for (; steps < max_steps && settle_rel > 1.0e-8; ++steps) {
    radiation::advance_radiation_step_fld_1d(
        state, cfg, planck, cfg.materials.materials.front(), dt);
    if (steps % 50 != 49) {
      continue;
    }
    const auto out_E = mg_copy_field_to_host(state.rad_E);
    const auto out_Te = mg_copy_field_to_host(state.Te);
    settle_rel = 0.0;
    for (int c = 0; c < n; ++c) {
      for (int g = 0; g < n_groups; ++g) {
        settle_rel = std::max(
            settle_rel,
            std::abs(out_E[static_cast<std::size_t>(c * n_groups + g)] -
                     settle_target[static_cast<std::size_t>(g)]) /
                settle_target[static_cast<std::size_t>(g)]);
      }
    }
    for (const double T : out_Te) {
      settle_rel = std::max(settle_rel, std::abs(T - Tr) / Tr);
    }
  }

  const auto out_E = mg_copy_field_to_host(state.rad_E);
  const auto out_Te_final = mg_copy_field_to_host(state.Te);
  double max_rel = 0.0;
  {
    std::ostringstream oss;
    oss << "[verify:" << label << "] plateau per-group max|E_g/(b_g aT^4)-1|:";
    for (int g = 0; g < n_groups; ++g) {
      double worst = 0.0;
      const double target =
          E_total_target * ref.b_ref[static_cast<std::size_t>(g)];
      for (int c = 0; c < n; ++c) {
        worst = std::max(
            worst,
            std::abs(out_E[static_cast<std::size_t>(c * n_groups + g)] -
                     target) /
                target);
      }
      max_rel = std::max(max_rel, worst);
      oss << " g" << g << "=" << mg_format_double(worst);
    }
    core::log_info(oss.str());
  }
  for (const double T : out_Te_final) {
    max_rel = std::max(max_rel, std::abs(T - Tr) / Tr);
  }
  double sum_rel = 0.0;
  for (int c = 0; c < n; ++c) {
    double sum = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      sum += out_E[static_cast<std::size_t>(c * n_groups + g)];
    }
    sum_rel = std::max(sum_rel,
                       std::abs(sum - E_total_target) / E_total_target);
  }
  const double flux_rel =
      std::abs(state.fld_escaped_step - state.fld_marshak_in_step) /
      std::max(state.fld_marshak_in_step, 1.0e-300);

  const bool pass = max_rel < 1.0e-6 && sum_rel < 1.0e-6 &&
                    flux_rel < 1.0e-6 && bg_code_vs_ref < 1.0e-6;
  core::log_info("[verify:" + label + "] steps=" + std::to_string(steps) +
                 " max_rel=" + mg_format_double(max_rel) +
                 " settle_rel=" + mg_format_double(settle_rel) +
                 " sum_rel=" + mg_format_double(sum_rel) +
                 " flux_rel=" + mg_format_double(flux_rel) +
                 " bg_code_vs_ref=" + mg_format_double(bg_code_vs_ref) +
                 " ref_internal=" + mg_format_double(ref.max_internal_diff));
  core::log_info("[verify:" + label + "] " +
                 std::string(pass ? "PASSED" : "FAILED"));
  return pass;
}

// Multigroup-coverage gate D — the fleck-array layout detector. table_nlte, uniform slab,
// reflect outer, E(0) = 0, Te0 = 50 eV, and max_outer_iterations = 1 so the
// published E1 is EXACTLY the single backward-Euler solve with eta(Te0) and
// f(Te0):
//   E1_g = dt f eta_g / (1 + dt c sigma_PA,g),
//   eta_g = sigma_PE,g c a T^4 b_g,  sigma_PE == sigma_PA == rho kappa_g
//   =>  f_meas(c,g) = E1 (1 + w_g) / (w_g B_g),  w_g = c rho kappa_g dt.
// (A converged outer at z ~ 1 would move Te by ~ f z T / 4 ~ 12% within the
// step — a Cv-independent identity — so a frozen-matter f ~ 0.5 probe is
// impossible; the single-iteration form sidesteps that exactly.)
// dt = 2e-11 puts z = beta c dt sigma_p_em ~ 1.01 => f_ref ~ 0.4967 with
// sigma_p_em = rho sum_g b_g kappa_g ~ 246.4 /cm. Checks: (i) f_meas is
// group-INDEPENDENT (the gray f is broadcast to every group slot — the
// fleck-array layout contract), (ii) cell-uniform, (iii) matches the replicated
// formula. Under the pre-fix code the g > 0 slots of cells c >= 1 read
// stale memory (cell 0's slots alias the old per-cell values), so (i)/(ii)
// fail decisively — this gate is the regression lock for the layout fix.
bool run_fld_1d_mg_nlte_fleck_probe_verify() {
  const std::string label = "fld_1d_mg_nlte_fleck_probe";
  if (!mg_verify_cuda_available(label.c_str())) {
    return true;
  }
  constexpr int n = 4;
  const std::vector<double>& bounds = mg_nlte_bounds_eV();
  const int n_groups = static_cast<int>(bounds.size()) - 1;
  const double Te0 = 50.0;     // eV, on the Planck-table middle node
  const double cv_e = 1.0e10;  // erg/(g*eV)
  const double dr = 0.05;      // cm
  const double dt = 2.0e-11;   // s; z ~ 1.01 by design
  const double rho = 1.0;      // g/cc (mg_fill_state fills rho = 1)

  const MgNlteFixture fx = mg_load_nlte_fixture(n_groups);
  if (!fx.ok) {
    core::log_info("[verify:" + label + "] FAILED (" + fx.error + ")");
    return false;
  }

  core::Config cfg = mg_make_config(n, bounds, "table_nlte", 0.0, cv_e);
  cfg.materials.materials.front().opacity_file = kMgNlteFixturePath;
  cfg.radiation.multigroup_diffusion.boundary.outer_r = "reflect";
  cfg.radiation.multigroup_diffusion.max_outer_iterations = 1;

  core::State state = core::State::allocate(cfg);
  if (!mg_check_planar_geometry(state, label)) {
    return false;
  }

  const MgReference ref = mg_build_reference(bounds, Te0);
  if (!(ref.max_internal_diff < 1.0e-9)) {
    core::log_info("[verify:" + label +
                   "] FAILED (independent reference integrators disagree: " +
                   mg_format_double(ref.max_internal_diff) + ")");
    return false;
  }

  radiation::Groups groups(cfg.radiation.group_bounds_eV);
  radiation::PlanckTable planck;
  planck.build(groups, 257, 0.5, 5000.0);  // Te0 = 50 eV on the middle node

  const double T4 = Te0 * Te0 * Te0 * Te0;
  std::vector<double> B_ref(static_cast<std::size_t>(n_groups), 0.0);
  std::vector<double> w_ref(static_cast<std::size_t>(n_groups), 0.0);
  double sigma_p_em_ref = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    const std::size_t gs = static_cast<std::size_t>(g);
    B_ref[gs] = core::constants::a_eV * T4 * ref.b_ref[gs];
    w_ref[gs] = core::constants::c_light * rho * fx.kappa_g[gs] * dt;
    sigma_p_em_ref += rho * fx.kappa_g[gs] * ref.b_ref[gs];
  }
  const double beta_ref = 4.0 * core::constants::a_eV * Te0 * Te0 * Te0 / cv_e;
  const double f_ref =
      1.0 / (1.0 + beta_ref * core::constants::c_light * dt * sigma_p_em_ref);
  if (!(f_ref > 0.2 && f_ref < 0.8)) {
    core::log_info("[verify:" + label + "] FAILED (probe design broken: " +
                   "f_ref=" + mg_format_double(f_ref) +
                   " outside (0.2, 0.8) — dt/kappa/cv_e drifted)");
    return false;
  }

  mg_fill_state(state, n, dr,
                std::vector<double>(static_cast<std::size_t>(n * n_groups),
                                    0.0),
                Te0, cv_e);

  radiation::advance_radiation_step_fld_1d(
      state, cfg, planck, cfg.materials.materials.front(), dt);
  const auto E1 = mg_copy_field_to_host(state.rad_E);

  std::vector<double> f_meas(static_cast<std::size_t>(n * n_groups), 0.0);
  for (int c = 0; c < n; ++c) {
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t gs = static_cast<std::size_t>(g);
      const std::size_t cg = static_cast<std::size_t>(c * n_groups + g);
      f_meas[cg] = E1[cg] * (1.0 + w_ref[gs]) /
                   (w_ref[gs] * std::max(B_ref[gs], 1.0e-300));
    }
  }

  double group_dev_max = 0.0;
  double cell_dev_max = 0.0;
  double f_acc_max = 0.0;
  for (int c = 0; c < n; ++c) {
    const double f0 = f_meas[static_cast<std::size_t>(c * n_groups)];
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t cg = static_cast<std::size_t>(c * n_groups + g);
      group_dev_max =
          std::max(group_dev_max,
                   std::abs(f_meas[cg] - f0) / std::max(f0, 1.0e-300));
      f_acc_max = std::max(f_acc_max, std::abs(f_meas[cg] / f_ref - 1.0));
    }
  }
  for (int g = 0; g < n_groups; ++g) {
    double f_min = std::numeric_limits<double>::infinity();
    double f_max = 0.0;
    for (int c = 0; c < n; ++c) {
      const double f = f_meas[static_cast<std::size_t>(c * n_groups + g)];
      f_min = std::min(f_min, f);
      f_max = std::max(f_max, f);
    }
    cell_dev_max =
        std::max(cell_dev_max, (f_max - f_min) / std::max(f_max, 1.0e-300));
  }

  {
    std::ostringstream oss;
    oss << "[verify:" << label << "] f_meas(cell0)/f_ref-1 per group:";
    for (int g = 0; g < n_groups; ++g) {
      oss << " g" << g << "="
          << mg_format_double(
                 f_meas[static_cast<std::size_t>(g)] / f_ref - 1.0);
    }
    core::log_info(oss.str());
  }

  const bool pass = f_acc_max < 1.0e-5 && group_dev_max < 1.0e-6 &&
                    cell_dev_max < 1.0e-6;
  core::log_info("[verify:" + label +
                 "] f_ref=" + mg_format_double(f_ref) +
                 " f_acc_max=" + mg_format_double(f_acc_max) +
                 " group_dev_max=" + mg_format_double(group_dev_max) +
                 " cell_dev_max=" + mg_format_double(cell_dev_max) +
                 " ref_internal=" + mg_format_double(ref.max_internal_diff));
  core::log_info("[verify:" + label + "] " +
                 std::string(pass ? "PASSED" : "FAILED"));
  return pass;
}

// Mission 3. Su & Olson (1999) picket-fence benchmark, diffusion column
// (design memo: tmp/diff_proposals/mg-picket/00_design_decisions.md).
// Dimensional mapping: kbar = 1/cm (rho = 1, kappa_g = w_g cm^2/g), T0 =
// 100 eV, eps = 1 (=> Cv = 4 a T^3, maintained by per-step host updates of
// state.cv_e), S0 = a c T0^4/(2 x0) over x in [0, x0], t = tau/(c kbar).
// flux_limiter="none" makes the FLD operator the paper's Eq. (16) exactly;
// fleck_mode="afi" removes the Fleck blend from the transient. The g0-only
// volume source is lifted to the p_n split by band-swap superposition
// (linearity of the Su-Olson-linearized system): with run A on fixture
// (k1,k2) and run B on the swapped fixture,
//   U1 = p1 A[g0] + p2 B[g1],  U2 = p1 A[g1] + p2 B[g0],
//   V = p1 V_A + p2 V_B   (V = (Te/T0)^4 is the linear matter variable).
bool run_fld_1d_mg_picket_fence_suolson_verify() {
  const std::string label = "fld_1d_mg_picket_fence_suolson";
  if (!mg_verify_cuda_available(label.c_str())) {
    return true;
  }
  const char* case_env = std::getenv("TENRYU_MG_PICKET_CASE");
  const bool use_case_c =
      (case_env != nullptr && (case_env[0] == 'C' || case_env[0] == 'c'));

  MgPfParams pf;
  pf.p1 = 0.5;
  pf.p2 = 0.5;
  pf.eps = 1.0;
  pf.x0 = 0.5;
  pf.tau0 = 10.0;
  pf.w1 = use_case_c ? (2.0 / 101.0) : (2.0 / 11.0);
  pf.w2 = use_case_c ? (200.0 / 101.0) : (20.0 / 11.0);
  const char* fixture_a = use_case_c ? "tests/data/ionmix_mg_picket_c.cn4"
                                     : "tests/data/ionmix_mg_picket_b.cn4";
  const char* fixture_b = use_case_c
                              ? "tests/data/ionmix_mg_picket_c_swap.cn4"
                              : "tests/data/ionmix_mg_picket_b_swap.cn4";
  const MgPfAnchor* anchors =
      use_case_c ? kMgPfAnchorsCaseC : kMgPfAnchorsCaseB;
  const int n_anchors =
      use_case_c
          ? static_cast<int>(sizeof(kMgPfAnchorsCaseC) / sizeof(MgPfAnchor))
          : static_cast<int>(sizeof(kMgPfAnchorsCaseB) / sizeof(MgPfAnchor));
  core::log_info("[verify:" + label + "] case " +
                 std::string(use_case_c ? "C (w2/w1=100, diagnostic)"
                                        : "B (w2/w1=10)"));

  // Phase 0: transcription protocol — the internal evaluator must reproduce
  // the transcribed Table 2/3 anchors before any code comparison.
  double anchor_err = 0.0;
  for (int i = 0; i < n_anchors; ++i) {
    const MgPfTriple r = mg_pf_eval(anchors[i].x, anchors[i].tau, pf);
    anchor_err = std::max({anchor_err, std::abs(r.U1 - anchors[i].U1),
                           std::abs(r.U2 - anchors[i].U2),
                           std::abs(r.V - anchors[i].V)});
  }
  core::log_info("[verify:" + label + "] evaluator-vs-table anchors max abs=" +
                 mg_format_double(anchor_err) + " over " +
                 std::to_string(n_anchors) + " points x 3 fields");
  if (!(anchor_err < 2.0e-5)) {
    core::log_info("[verify:" + label +
                   "] FAILED (evaluator does not reproduce the paper tables)");
    return false;
  }

  const double kappa_a[2] = {pf.w1, pf.w2};
  const double kappa_b[2] = {pf.w2, pf.w1};
  const MgNlteFixture fxa = mg_load_pf_fixture(fixture_a, kappa_a);
  const MgNlteFixture fxb = mg_load_pf_fixture(fixture_b, kappa_b);
  if (!fxa.ok || !fxb.ok) {
    core::log_info("[verify:" + label + "] FAILED (" +
                   (fxa.ok ? fxb.error : fxa.error) + ")");
    return false;
  }

  // Dimensional constants.
  constexpr int n = 800;
  const double dr = 0.05;       // cm; x_max = 40
  const double T0 = 100.0;      // eV
  const double Te_init = 0.1;   // eV: V(0) = 1e-12 ~ 0
  const double Te_cv_floor = 1.0e-3;  // eV floor inside the Cv(T) update
  const double aT04 =
      core::constants::a_eV * T0 * T0 * T0 * T0;
  const double S0 = core::constants::a_eV * core::constants::c_light * T0 *
                    T0 * T0 * T0 / (2.0 * pf.x0);
  const std::vector<double> bounds = {1.0, 100.0, 1.0e4};  // placeholders

  // tau schedule segments {dtau, tau_end}; checkpoints at tau = 1, 3, 30.
  struct Segment {
    double dtau;
    double tau_end;
  };
  Segment segments[] = {
      {1.0e-5, 0.01}, {1.0e-4, 0.1}, {5.0e-4, 1.0}, {1.0e-3, 3.0},
      {2.0e-3, 30.0},
  };
  // Diagnostic dtau ladder (integer divisor keeps checkpoint alignment).
  const char* dtau_env = std::getenv("TENRYU_MG_PICKET_DTAU_DIV");
  if (dtau_env != nullptr && dtau_env[0] != 0) {
    const int div = std::atoi(dtau_env);
    if (div > 1) {
      for (Segment& seg : segments) {
        seg.dtau /= static_cast<double>(div);
      }
    }
  }
  const double tau_checks[] = {1.0, 3.0, 30.0};
  constexpr int n_checks = 3;

  // Two superposition runs.
  std::vector<double> E_run[2];    // [run][checkpoint*n*2 + c*2 + g]
  std::vector<double> V_run[2];    // [run][checkpoint*n + c]
  for (int run = 0; run < 2; ++run) {
    E_run[run].assign(static_cast<std::size_t>(n_checks * n * 2), 0.0);
    V_run[run].assign(static_cast<std::size_t>(n_checks * n), 0.0);

    core::Config cfg = mg_make_config(n, bounds, "table_nlte", 0.0,
                                      /*cv_e placeholder*/ 0.0);
    cfg.materials.materials.front().opacity_file =
        (run == 0) ? fixture_a : fixture_b;
    cfg.materials.materials.front().cv_e_override = 0.0;  // per-cell cv_e
    cfg.radiation.multigroup_diffusion.flux_limiter = "none";
    cfg.radiation.multigroup_diffusion.fleck_mode = "afi";
    cfg.radiation.multigroup_diffusion.max_outer_iterations = 60;
    cfg.radiation.multigroup_diffusion.boundary.outer_r = "vacuum";
    cfg.radiation.volume_source_rate = S0;
    cfg.radiation.volume_source_x_max = pf.x0;

    core::State state = core::State::allocate(cfg);
    if (!mg_check_planar_geometry(state, label)) {
      return false;
    }
    radiation::Groups groups(cfg.radiation.group_bounds_eV);
    radiation::PlanckTable planck;
    planck.build_constant_fractions(groups, {pf.p1, pf.p2}, 1.0, 1.0e4);

    mg_fill_state(state, n, dr,
                  std::vector<double>(static_cast<std::size_t>(n * 2), 0.0),
                  Te_init, 0.0);
    // Cv = 4 a T^3 (eps = 1): per-step host maintenance of state.cv_e.
    std::vector<double> te_host(static_cast<std::size_t>(n), Te_init);
    std::vector<double> cv_host(static_cast<std::size_t>(n), 0.0);
    auto update_cv = [&]() {
      for (int c = 0; c < n; ++c) {
        const double T =
            std::max(te_host[static_cast<std::size_t>(c)], Te_cv_floor);
        cv_host[static_cast<std::size_t>(c)] =
            4.0 * core::constants::a_eV * T * T * T;
      }
      mg_copy_field_from_host(state.cv_e, cv_host);
    };
    update_cv();

    double tau = 0.0;
    bool source_on = true;
    int check_idx = 0;
    for (const Segment& seg : segments) {
      const double dt_dim =
          seg.dtau / (core::constants::c_light * 1.0 /*kbar*/);
      while (tau < seg.tau_end - 0.5 * seg.dtau) {
        if (source_on && tau >= pf.tau0 - 0.25 * seg.dtau) {
          cfg.radiation.volume_source_rate = 0.0;
          source_on = false;
        }
        radiation::advance_radiation_step_fld_1d(
            state, cfg, planck, cfg.materials.materials.front(), dt_dim);
        tau += seg.dtau;
        state.Te.copy_to_host(te_host.data());
        update_cv();
        if (check_idx < n_checks &&
            std::abs(tau - tau_checks[check_idx]) < 0.25 * seg.dtau) {
          const auto e_now = mg_copy_field_to_host(state.rad_E);
          for (int c = 0; c < n; ++c) {
            for (int g = 0; g < 2; ++g) {
              E_run[run][static_cast<std::size_t>(
                  (check_idx * n + c) * 2 + g)] =
                  e_now[static_cast<std::size_t>(c * 2 + g)];
            }
            const double vt = te_host[static_cast<std::size_t>(c)] / T0;
            V_run[run][static_cast<std::size_t>(check_idx * n + c)] =
                vt * vt * vt * vt;
          }
          ++check_idx;
        }
      }
    }
    if (check_idx != n_checks) {
      core::log_info("[verify:" + label + "] FAILED (checkpoint scheduling: " +
                     std::to_string(check_idx) + "/" +
                     std::to_string(n_checks) + ")");
      return false;
    }
  }

  // Phase 2+3: superpose and compare against the evaluator per checkpoint.
  bool pass = true;
  double worst_l2 = 0.0;
  double worst_max = 0.0;
  double far_bc = 0.0;
  for (int ck = 0; ck < n_checks; ++ck) {
    const double tau = tau_checks[ck];
    double l2_num[3] = {0.0, 0.0, 0.0};
    double l2_den[3] = {0.0, 0.0, 0.0};
    double max_abs[3] = {0.0, 0.0, 0.0};
    for (int c = 0; c < n; ++c) {
      const double x = (static_cast<double>(c) + 0.5) * dr;
      const MgPfTriple ref = mg_pf_eval(x, tau, pf);
      if (ref.U1 + ref.U2 < 1.0e-4) {
        break;  // outside the comparison window (monotone decay in x)
      }
      const std::size_t iA0 = static_cast<std::size_t>((ck * n + c) * 2);
      const double u1 =
          (pf.p1 * E_run[0][iA0] + pf.p2 * E_run[1][iA0 + 1U]) / aT04;
      const double u2 =
          (pf.p1 * E_run[0][iA0 + 1U] + pf.p2 * E_run[1][iA0]) / aT04;
      const double v =
          pf.p1 * V_run[0][static_cast<std::size_t>(ck * n + c)] +
          pf.p2 * V_run[1][static_cast<std::size_t>(ck * n + c)];
      const double dev[3] = {u1 - ref.U1, u2 - ref.U2, v - ref.V};
      const double rv[3] = {ref.U1, ref.U2, ref.V};
      for (int fchan = 0; fchan < 3; ++fchan) {
        l2_num[fchan] += dev[fchan] * dev[fchan];
        l2_den[fchan] += rv[fchan] * rv[fchan];
        max_abs[fchan] = std::max(max_abs[fchan], std::abs(dev[fchan]));
      }
    }
    std::ostringstream oss;
    oss << "[verify:" << label << "] tau=" << tau;
    const char* names[3] = {"U1", "U2", "V"};
    for (int fchan = 0; fchan < 3; ++fchan) {
      const double l2 =
          std::sqrt(l2_num[fchan] / std::max(l2_den[fchan], 1.0e-300));
      worst_l2 = std::max(worst_l2, l2);
      worst_max = std::max(worst_max, max_abs[fchan]);
      oss << " " << names[fchan] << ": L2rel=" << mg_format_double(l2)
          << " maxabs=" << mg_format_double(max_abs[fchan]);
    }
    core::log_info(oss.str());
  }
  {
    const auto eA = E_run[0];
    for (int g = 0; g < 2; ++g) {
      far_bc = std::max(
          far_bc,
          eA[static_cast<std::size_t>(((n_checks - 1) * n + (n - 1)) * 2 + g)] /
              aT04);
    }
  }

  // Frozen at measured x 1.5 (first run 2026-07-04: worst_L2rel=1.838e-3
  // (V at tau=1), worst_maxabs=7.92e-4, decreasing with tau). The dtau/2
  // ladder (TENRYU_MG_PICKET_DTAU_DIV=2) moved worst_L2rel only 1.838e-3 ->
  // 1.742e-3, so the dominant error is SPATIAL (dx^2 / cell-average-vs-point
  // at the source-edge kink x0), not time discretization; the decay with tau
  // tracks the kink smoothing out. far_bc threshold 1e-5: the measured 2.25e-6
  // at x_max=40 is the REAL infinite-medium fast-group tail
  // (D1 = 1/(3 w1) = 1.83, exp(-x^2/(4 D1 tau)) scale), not truncation
  // error; the requirement is only that it stays well below the 1e-4
  // comparison-window floor.
  const double tol_l2 = 2.8e-3;
  const double tol_max = 1.2e-3;
  pass = worst_l2 < tol_l2 && worst_max < tol_max && far_bc < 1.0e-5;
  core::log_info("[verify:" + label + "] worst_L2rel=" +
                 mg_format_double(worst_l2) + " worst_maxabs=" +
                 mg_format_double(worst_max) + " far_bc=" +
                 mg_format_double(far_bc) + " anchor_err=" +
                 mg_format_double(anchor_err) +
                 (use_case_c ? " [case C diagnostic — tolerances frozen for "
                               "case B]"
                             : ""));
  core::log_info("[verify:" + label + "] " +
                 std::string(pass ? "PASSED" : "FAILED"));
  return pass;
}

}  // namespace tenryu::drivers
