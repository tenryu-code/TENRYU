#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "core/config.hpp"

#ifndef TENRYU_HOTE_HOST_DEVICE
#if defined(__CUDACC__)
#define TENRYU_HOTE_HOST_DEVICE __host__ __device__
#else
#define TENRYU_HOTE_HOST_DEVICE
#endif
#endif

namespace tenryu::laser::hot_electron {

// Physical constants (cgs-Gaussian). kElectronMass/kElementaryCharge mirror
// the file-local laser-module values (cbet.cu / refraction.cu); proton mass
// and eV->erg come from core::constants at the .cpp side.
inline constexpr double kElectronMass = 9.1094e-28;      // g
inline constexpr double kElementaryCharge = 4.8032e-10;  // statC
inline constexpr double kHbar = 1.0546e-27;              // erg s
inline constexpr double kCLight = 2.99792458e10;         // cm/s (== core::constants::c_light)
inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kSubstepEnergyFraction = 0.02;   // f_sub: <=2% energy change per RK4 substep
inline constexpr int kMaxSubstepsPerCell = 256;
inline constexpr int kAxisBins = 16;                     // mu_axis reduction bins (design §11.6)

struct GroupSpec {
  double E_rep = 0.0;   // representative energy [erg] (energy-flux average in the group)
  double weight = 0.0;  // fraction of P_hot carried by the group; sum over groups == 1
};

// Exponential-spectrum multigroup layout (host-only; design §3.3).
std::vector<GroupSpec> build_groups(double T_hot_erg, int n_groups,
                                    double e_min_over_th, double e_max_over_th);

// Free-electron Coulomb stopping power S_m [erg cm^2 / g] (design §3.4).
// Returns 0 for non-positive inputs (void/vacuum: free traversal).
// Fast path: x > 5 => G(x)=1 exactly (error < 1e-11) — skips erf/exp.
TENRYU_HOTE_HOST_DEVICE inline double stopping_power_erg_cm2_per_g(
    const double E_erg, const double ne_cm3, const double Te_erg, const double rho) {
  if (!(E_erg > 0.0) || !(ne_cm3 > 0.0) || !(Te_erg > 0.0) || !(rho > 0.0)) {
    return 0.0;
  }
  const double me_c2 = kElectronMass * kCLight * kCLight;
  const double gamma = 1.0 + E_erg / me_c2;
  const double beta2 = 1.0 - 1.0 / (gamma * gamma);
  const double v2 = kCLight * kCLight * (beta2 > 0.0 ? beta2 : 0.0);
  if (!(v2 > 0.0)) {
    return 0.0;
  }
  const double v = sqrt(v2);
  const double x2 = 0.5 * kElectronMass * v2 / Te_erg;  // x^2 = v^2 / (2 vte^2)
  double G;
  if (x2 > 25.0) {
    G = 1.0;
  } else {
    const double x = sqrt(x2);
    G = erf(x) - (2.0 / sqrt(kPi)) * x * exp(-x2);
    if (!(G > 0.0)) {
      return 0.0;
    }
  }
  const double e2 = kElementaryCharge * kElementaryCharge;
  const double lambda_D = sqrt(Te_erg / (4.0 * kPi * ne_cm3 * e2));
  const double b_min_c = e2 / (kElectronMass * v2);
  const double b_min_q = kHbar / (2.0 * kElectronMass * v);
  const double b_min = (b_min_c > b_min_q) ? b_min_c : b_min_q;
  double lnL = log(lambda_D / b_min);
  if (!(lnL > 2.0)) {
    lnL = 2.0;
  }
  return 4.0 * kPi * e2 * e2 * ne_cm3 * lnL * G / (rho * kElectronMass * v2);
}

// March one electron through areal density dSigma [g/cm^2] with local
// stopping functor S(E)->S_m, integrating dE/dSigma = -S(E) by classical RK4
// with adaptive substeps (energy-change target f_sub). Returns the energy
// LEAVING the region (>= 0): 0.0 when thermalized (E <= E_floor) or when the
// substep cap trips (conservative fallback; increments *cap_counter);
// E_in unchanged when stopping is zero.
template <class StoppingFn>
TENRYU_HOTE_HOST_DEVICE inline double march_cell(
    const double E_in, const double dSigma, StoppingFn&& S_of_E,
    const double E_floor, const double f_sub, const int max_substeps,
    int* cap_counter = nullptr) {
  double E = E_in;
  if (!(E > E_floor)) {
    return 0.0;
  }
  double remaining = dSigma;
  int steps = 0;
  while (remaining > 0.0 && steps < max_substeps) {
    const double S0 = S_of_E(E);
    if (!(S0 > 0.0)) {
      return E;
    }
    double dSig = f_sub * E / S0;
    if (dSig > remaining) {
      dSig = remaining;
    }
    const double k1 = S0;
    const double e2s = E - 0.5 * dSig * k1;
    const double k2 = S_of_E(e2s > 0.0 ? e2s : 0.0);
    const double e3s = E - 0.5 * dSig * k2;
    const double k3 = S_of_E(e3s > 0.0 ? e3s : 0.0);
    const double e4s = E - dSig * k3;
    const double k4 = S_of_E(e4s > 0.0 ? e4s : 0.0);
    E = E - (dSig / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    if (E < 0.0) {
      E = 0.0;
    }
    remaining -= dSig;
    ++steps;
    if (!(E > E_floor)) {
      return 0.0;
    }
  }
  if (remaining > 0.0 && steps >= max_substeps) {
    if (cap_counter != nullptr) {
      ++(*cap_counter);
    }
    return 0.0;
  }
  return E;
}

// Allocation-free walker over spherical-shell segments of the ray
// x(s) = x0 + s*omega (|omega| = 1, s >= 0) against node radii
// r_nodes[0..n_nodes-1] (monotone increasing; cell k spans
// [r_nodes[k], r_nodes[k+1]]). Emits segments in increasing s; a segment is
// (cell, ds). Terminates when the ray leaves the outermost node (escape).
// The central hole [0, r_nodes[0]) (annular meshes) is traversed without
// emission; r_nodes[0] == 0 is handled by the pericenter branch.
struct ChordWalkerSph {
  const double* r_nodes;
  int n_nodes;
  double b;      // x0 . omega
  double p2;     // impact parameter squared, max(r0^2 - b^2, 0)
  double s_cur;  // current path coordinate
  int j;         // current cell
  bool inbound;  // heading toward pericenter
  bool done;

  TENRYU_HOTE_HOST_DEVICE ChordWalkerSph(const double r0, const double b_in,
                                         const double* nodes, const int n)
      : r_nodes(nodes), n_nodes(n), b(b_in), s_cur(0.0), j(-1),
        inbound(b_in < 0.0), done(n < 2) {
    const double r0c = (r0 > 0.0) ? r0 : 0.0;
    p2 = r0c * r0c - b * b;
    if (p2 < 0.0) {
      p2 = 0.0;
    }
    if (done) {
      return;
    }
    const double r_outer = r_nodes[n_nodes - 1];
    if (r0c >= r_outer) {
      const double disc = r_outer * r_outer - p2;
      if (!(b < 0.0) || !(disc > 0.0)) {
        done = true;
        return;
      }
      s_cur = -b - sqrt(disc);
      j = n_nodes - 2;
      inbound = true;
    } else {
      j = n_nodes - 2;
      while (j > 0 && r0c < r_nodes[j]) {
        --j;
      }
    }
  }

  TENRYU_HOTE_HOST_DEVICE bool next(double& ds, int& cell) {
    if (done) {
      return false;
    }
    for (;;) {
      if (inbound) {
        const double R_in = r_nodes[j];
        const double disc = R_in * R_in - p2;
        if (disc > 0.0) {
          const double root = sqrt(disc);
          const double s_hit = -b - root;
          ds = s_hit - s_cur;
          cell = j;
          s_cur = s_hit;
          if (j == 0) {
            // crossed into the central hole: skip to its symmetric exit
            inbound = false;
            s_cur = -b + root;
          } else {
            --j;
          }
          if (ds > 0.0) {
            return true;
          }
          continue;
        }
        // pericenter inside cell j
        const double s_peri = -b;
        ds = s_peri - s_cur;
        cell = j;
        s_cur = s_peri;
        inbound = false;
        if (ds > 0.0) {
          return true;
        }
        continue;
      }
      const double R_out = r_nodes[j + 1];
      double disc = R_out * R_out - p2;
      if (disc < 0.0) {
        disc = 0.0;
      }
      const double s_hit = -b + sqrt(disc);
      ds = s_hit - s_cur;
      cell = j;
      s_cur = s_hit;
      if (j + 1 >= n_nodes - 1) {
        done = true;
      } else {
        ++j;
      }
      if (ds > 0.0) {
        return true;
      }
      if (done) {
        return false;
      }
    }
  }
};

// Planar-slab walker: nodes x_nodes[0..n_nodes-1] along the slab axis,
// direction cosine mu_x = omega . x_hat. |mu_x| below kMuxEpsilon emits no
// segments (the cone pipelines deposit such in-plane chords locally in the
// source cell — an infinite lateral slab path thermalizes in place). Path
// per cell = dx / |mu_x|. Escapes at either slab face.
//
// Interior starts emit a partial first segment from x0 to the exit face in
// the travel direction (2026-07-26 review: the previous full-cell first
// segment overstated the first chord and, for a source exactly on an
// internal face with mu_x < 0, walked the wrong (outward) cell first).
inline constexpr double kMuxEpsilon = 1.0e-12;

struct ChordWalkerPln {
  const double* x_nodes;
  int n_nodes;
  double inv_abs_mu;
  double first_seg;  // >=0: axial length of the partial first segment
  int j;
  int step;  // -1 toward x_nodes[0], +1 toward x_nodes[n-1]
  bool done;

  TENRYU_HOTE_HOST_DEVICE ChordWalkerPln(const double x0, const double mu_x,
                                         const double* nodes, const int n)
      : x_nodes(nodes), n_nodes(n), inv_abs_mu(0.0), first_seg(-1.0), j(-1),
        step(0), done(true) {
    if (n < 2) {
      return;
    }
    const double amu = (mu_x < 0.0) ? -mu_x : mu_x;
    if (amu < kMuxEpsilon) {
      return;  // in-plane: caller deposits locally
    }
    if (x0 <= x_nodes[0] || x0 >= x_nodes[n - 1]) {
      // start on/outside a face: only entering directions traverse
      if (x0 >= x_nodes[n - 1] && mu_x < 0.0) {
        j = n - 2;
      } else if (x0 <= x_nodes[0] && mu_x > 0.0) {
        j = 0;
      } else {
        return;
      }
    } else {
      j = n - 2;
      while (j > 0 && x0 < x_nodes[j]) {
        --j;
      }
      first_seg = (mu_x > 0.0) ? (x_nodes[j + 1] - x0) : (x0 - x_nodes[j]);
      if (first_seg < 0.0) {
        first_seg = 0.0;
      }
    }
    inv_abs_mu = 1.0 / amu;
    step = (mu_x < 0.0) ? -1 : +1;
    done = false;
  }

  TENRYU_HOTE_HOST_DEVICE bool next(double& ds, int& cell) {
    while (!done) {
      double seg = x_nodes[j + 1] - x_nodes[j];
      if (first_seg >= 0.0) {
        seg = first_seg;
        first_seg = -1.0;
      }
      ds = seg * inv_abs_mu;
      cell = j;
      j += step;
      if (j < 0 || j > n_nodes - 2) {
        done = true;
      }
      // A zero-length first segment (source exactly on the entry-side face)
      // is skipped so the walk starts in the correct neighbour cell.
      if (ds > 0.0) {
        return true;
      }
    }
    return false;
  }
};

// --- Cone quadrature (host-only) ---
struct AngleNode {
  double mu_dir;   // direction-to-radial cosine at the source (the only geometric dof in 1D)
  double weight;   // sum over nodes == 1
};

// Gauss-Legendre nodes/weights on [-1, 1] (Newton on Legendre P_n; host-only,
// deterministic). n >= 1.
void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w);

// Resolved per-channel transport spec (design §3-§4.3): all angles collapsed
// to a mu band [mu_lo, mu_hi] around the capture axis, temperatures in erg.
// Built host-side once per step per channel via make_channel_spec*().
struct HotEChannelSpec {
  double T_hot_erg = 0.0;
  int n_energy_groups = 30;
  double E_min_over_Th = 0.2;
  double E_max_over_Th = 8.0;
  double mu_lo = 0.5;   // cos of the band's outer polar angle
  double mu_hi = 1.0;   // cos of the band's inner polar angle
  int n_mu = 6;
  int n_phi = 8;
  bool inner_escape = false;  // radial-mode inner BC: true == "escape"
};

// Channel spec from the multi-channel config form. Angular band:
//   cone/srs: [cos(theta_div_deg), 1]
//   tpd:      [cos(min(tpd_theta+tpd_delta,180) deg), cos(max(tpd_theta-tpd_delta,0) deg)]
// (ring = exact azimuth average of the +-theta bilobe, design §3.1).
HotEChannelSpec make_channel_spec(
    const core::Config::LaserConfig::HotElectronConfig& he,
    const core::Config::LaserConfig::HotEChannelConfig& channel);

// Channel spec from the v1 scalar shorthand (single cone channel).
HotEChannelSpec make_channel_spec_from_shorthand(
    const core::Config::LaserConfig::HotElectronConfig& he);

// Deterministic quadrature of the solid-angle band mu in [mu_lo, mu_hi]
// around an axis with direction-to-radial cosine mu_axis: Gauss-Legendre in
// mu (uniform solid angle) x midpoint-uniform phi, weights summing to 1.
// Degenerate cases: mu_lo == mu_hi == 1 -> single node {mu_axis, 1};
// mu_lo == mu_hi < 1 -> single-mu ring (n_phi azimuthal nodes).
// Requires mu_lo <= mu_hi (callers construct bands in order).
std::vector<AngleNode> build_band_nodes(double mu_axis, double mu_lo,
                                        double mu_hi, int n_mu, int n_phi);

// Cone of half-angle theta_div (radians) around an axis whose angle to the
// local radial direction has cosine mu_axis. Returns nodes in mu_dir =
// Omega.r_hat with weights summing to 1:
//   mu_dir(mu, phi) = mu*mu_axis + sqrt(1-mu^2)*sqrt(1-mu_axis^2)*cos(phi)
// with mu Gauss-Legendre on [cos(theta_div), 1] (n_mu nodes; uniform
// solid-angle measure => plain GL weight normalized by the interval) and phi
// midpoint-uniform on [0, 2pi) (n_phi nodes, weight 1/n_phi each).
// theta_div == 0 degenerates to a single node {mu_axis, 1.0}.
std::vector<AngleNode> build_cone_nodes(double mu_axis, double theta_div_rad,
                                        int n_mu, int n_phi);

// --- Sources and pipelines (host-only) ---
struct HotESource {
  int cell = -1;         // mesh cell containing r_s
  double r_s = 0.0;      // launch radius [cm] (planar: launch x)
  double mu_axis = -1.0; // cone-axis-to-radial cosine (radial-inward = -1)
  double P_hot = 0.0;    // erg/s
};

struct RayCapture {
  double r_s = 0.0;      // |x| at the crossing [cm]
  double mu_axis = 0.0;  // k_hat . r_hat at the crossing
  double P_hot = 0.0;    // eta * P_before [erg/s]
};

// Power-weighted reduction of per-ray captures onto (source cell, mu_axis
// bin) aggregates (design §11.6): kAxisBins uniform bins over [-1, 1],
// representative mu_axis and r_s are power-weighted bin means, fixed
// ascending (cell, bin) emission order (deterministic).
std::vector<HotESource> reduce_captures(const std::vector<RayCapture>& captures,
                                        const std::vector<double>& r_nodes);

enum class Geometry1D : int { spherical = 0, planar = 1 };

struct DepositResult {
  bool active = false;
  int n_sources = 0;
  double r_source_mean = 0.0;      // power-weighted mean launch radius [cm]
  double P_hot = 0.0;              // total hot-electron power [erg/s]
  double P_deposited = 0.0;        // into cells (incl. radial-mode inner residual) [erg/s]
  double P_residual_inner = 0.0;   // radial mode inner_bc="deposit_residual" share [erg/s]
  double P_escaped = 0.0;          // left the mesh [erg/s]
  double conservation_resid = 0.0; // |P_deposited + P_escaped - P_hot| / max(P_hot, tiny)
  int substep_cap_hits = 0;
};

// Cone pipeline (angular_model="cone"; spherical or planar geometry).
// Fixed loop order: sources asc -> angle nodes asc -> (chord) -> groups asc;
// each chord accumulates into a local row which is folded into
// dep_power_cell after the chord completes (order-stable for a future
// device port; design §11.6).
DepositResult deposit_hot_electrons_cone_1d(
    const HotEChannelSpec& spec,
    Geometry1D geom,
    const std::vector<HotESource>& sources,
    const std::vector<double>& rho,
    const std::vector<double>& zbar,
    const std::vector<double>& A_eff,
    const std::vector<double>& Te_eV,
    const std::vector<std::uint8_t>& cell_is_void,
    const std::vector<double>& r_nodes,
    std::vector<double>& dep_power_cell);

// Shorthand-resolution overload (kept: the unit-tested single-channel equivalence surface).
DepositResult deposit_hot_electrons_cone_1d(
    const core::Config::LaserConfig::HotElectronConfig& cfg,
    Geometry1D geom,
    const std::vector<HotESource>& sources,
    const std::vector<double>& rho,
    const std::vector<double>& zbar,
    const std::vector<double>& A_eff,
    const std::vector<double>& Te_eV,
    const std::vector<std::uint8_t>& cell_is_void,
    const std::vector<double>& r_nodes,
    std::vector<double>& dep_power_cell);

// Radial pipeline (angular_model="radial"; mu = 1 verification/fallback
// mode, design §3.4): marches inward from the outermost source only.
DepositResult deposit_hot_electrons_radial_1d(
    const HotEChannelSpec& spec,
    const std::vector<HotESource>& sources,
    const std::vector<double>& rho,
    const std::vector<double>& zbar,
    const std::vector<double>& A_eff,
    const std::vector<double>& Te_eV,
    const std::vector<std::uint8_t>& cell_is_void,
    const std::vector<double>& r_nodes,
    std::vector<double>& dep_power_cell);

// Shorthand-resolution overload (kept: the unit-tested single-channel equivalence surface).
DepositResult deposit_hot_electrons_radial_1d(
    const core::Config::LaserConfig::HotElectronConfig& cfg,
    const std::vector<HotESource>& sources,
    const std::vector<double>& rho,
    const std::vector<double>& zbar,
    const std::vector<double>& A_eff,
    const std::vector<double>& Te_eV,
    const std::vector<std::uint8_t>& cell_is_void,
    const std::vector<double>& r_nodes,
    std::vector<double>& dep_power_cell);

}  // namespace tenryu::laser::hot_electron
