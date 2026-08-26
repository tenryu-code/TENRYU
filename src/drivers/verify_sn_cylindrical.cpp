#include "drivers/verify_sn_cylindrical.hpp"

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
#include "radiation/sn_cyl_quadrature_1d.hpp"
#include "radiation/sn_transport_1d_gpu.cuh"

namespace tenryu::drivers {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

bool cyl_gate_cuda_available(const std::string& label) {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err == cudaSuccess && device_count > 0) {
    return true;
  }
  core::log_info("[verify:" + label + "] SKIPPED (no CUDA device)");
  return false;
}

std::string cyl_format_double(const double value) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.17g", value);
  return std::string(buffer);
}

std::vector<double> cyl_nodes(const int n, const double r0, const double dr) {
  std::vector<double> node(static_cast<std::size_t>(n + 1), 0.0);
  for (int i = 0; i <= n; ++i) {
    node[static_cast<std::size_t>(i)] = r0 + dr * static_cast<double>(i);
  }
  return node;
}

// Cylindrical shell volumes per unit axial length: pi * (r1^2 - r0^2).
std::vector<double> cyl_volumes(const std::vector<double>& node) {
  std::vector<double> vol(node.size() - 1U, 0.0);
  for (std::size_t c = 0; c < vol.size(); ++c) {
    const double r0 = node[c];
    const double r1 = node[c + 1U];
    vol[c] = kPi * (r1 - r0) * (r1 + r0);
  }
  return vol;
}

core::Config make_cyl_sn_gate_config(const int n_cells,
                                     const double kappa,
                                     const double cv_e) {
  core::Config cfg;
  cfg.main.dimension = "1D_SPH";
  cfg.main.dim = 1;
  cfg.mesh.nr = n_cells;
  cfg.mesh.nz = 1;
  cfg.mesh.geometry_1d = "cylindrical";
  cfg.radiation.mode = core::RadiationMode::SnTransport;
  cfg.radiation.groups = 1;
  cfg.radiation.group_bounds_eV = {0.0, 1.0e4};
  cfg.radiation.imc.enabled = false;
  cfg.radiation.ddmc.enabled = false;
  cfg.radiation.holo.enabled = false;
  cfg.radiation.imc.difference.enabled = false;
  cfg.radiation.sn_transport.n_angles = 8;
  cfg.radiation.sn_transport.max_outer_iterations = 8;
  cfg.radiation.sn_transport.max_inner_iterations = 40;
  cfg.radiation.sn_transport.outer_tol = 1.0e-8;
  cfg.radiation.sn_transport.inner_tol = 1.0e-8;
  cfg.radiation.sn_transport.boundary.outer_r = "marshak";
  cfg.radiation.boundary.marshak_Tr_eV = 50.0;
  core::Config::MaterialsConfig::MatDef mat;
  mat.A = 1.0;
  mat.Z = 1.0;
  mat.kappa_a_constant = kappa;
  mat.cv_e_override = cv_e;
  cfg.materials.materials.push_back(mat);
  return cfg;
}

void fill_cyl_state(core::State& state,
                    const std::vector<double>& node,
                    const std::vector<double>& vol,
                    const double rad_E,
                    const double Te,
                    const double cv_e) {
  const std::size_t n = vol.size();
  state.mesh.dim = 1;
  state.x_r.copy_from_host(node);
  state.vol.copy_from_host(vol);
  state.rho.copy_from_host(std::vector<double>(n, 1.0));
  state.zbar.copy_from_host(std::vector<double>(n, 1.0));
  state.Te.copy_from_host(std::vector<double>(n, Te));
  state.ee.copy_from_host(std::vector<double>(n, cv_e * Te));
  state.Pe.copy_from_host(std::vector<double>(n, 1.0e8));
  state.cv_e.reset(n);
  state.cv_e.copy_from_host(std::vector<double>(n, cv_e));
  state.rad_E.copy_from_host(std::vector<double>(n, rad_E));
}

}  // namespace

bool run_sn_1d_cylindrical_marshak_equilibration_verify() {
  const std::string label = "sn_1d_cylindrical_marshak_equilibration";
  if (!cyl_gate_cuda_available(label)) {
    return true;
  }
  constexpr int n = 8;
  const double Tr = 50.0;
  const double cv_e = 1.0e10;
  core::Config cfg = make_cyl_sn_gate_config(n, 50.0, cv_e);
  const char* diag_na = std::getenv("TENRYU_SN_MARSHAK_DIAG_NANGLES");
  if (diag_na != nullptr && diag_na[0] != 0) {
    cfg.radiation.sn_transport.n_angles = std::atoi(diag_na);
  }
  const char* diag_scheme = std::getenv("TENRYU_SN_MARSHAK_DIAG_SCHEME");
  if (diag_scheme != nullptr && diag_scheme[0] != 0) {
    cfg.radiation.sn_transport.spatial_scheme = diag_scheme;
  }
  const int n_angles = cfg.radiation.sn_transport.n_angles;
  if (radiation::sn_cyl_levels_for(n_angles) < 2) {
    core::log_info("[verify:" + label + "] FAILED (n_angles=" +
                   std::to_string(n_angles) +
                   " is not 2*L*L with L >= 2; valid: 8, 18, 32, 50, ...)");
    return false;
  }
  // Full cylinder by default: the axis cell exercises the per-level
  // starting-direction + reflection machinery (the cylindrical analog of
  // the conservative-streaming spherical center-dip fix).
  const char* diag_r0 = std::getenv("TENRYU_SN_MARSHAK_DIAG_R0");
  const double gate_r0 = (diag_r0 != nullptr && diag_r0[0] != 0)
                             ? std::atof(diag_r0)
                             : 0.0;
  const auto node = cyl_nodes(n, gate_r0, 0.05);
  const auto vol = cyl_volumes(node);
  radiation::Groups groups(cfg.radiation.group_bounds_eV);
  radiation::PlanckTable planck;
  planck.build(groups, 32, 1.0e-2, 1.0e4);
  const double dt = 1.0e-10;
  const double E_target = core::constants::a_eV * Tr * Tr * Tr * Tr;

  // ---- Phase A: uniform blackbody fixed point (Te0 = Tr) ----
  bool pass_a = false;
  {
    core::State state = core::State::allocate(cfg);
    if (state.mesh.geometry_code != 1) {
      core::log_info("[verify:" + label + "] FAILED (geometry_code=" +
                     std::to_string(state.mesh.geometry_code) +
                     " want=1 — State::allocate did not bind Mesh.geometry_1d)");
      return false;
    }
    fill_cyl_state(state, node, vol, E_target, Tr, cv_e);
    double drift = 0.0;
    for (int steps = 0; steps < 200; ++steps) {
      radiation::advance_radiation_step_sn_1d(
          state, cfg, planck, cfg.materials.materials.front(), dt);
      if (steps % 50 != 49) {
        continue;
      }
      std::vector<double> out_E;
      std::vector<double> out_Te;
      state.rad_E.copy_to_host(out_E);
      state.Te.copy_to_host(out_Te);
      for (const double E : out_E) {
        drift = std::max(drift, std::abs(E - E_target) / E_target);
      }
      for (const double T : out_Te) {
        drift = std::max(drift, std::abs(T - Tr) / Tr);
      }
      if (drift > 1.0e-12) {
        break;
      }
    }
    pass_a = drift <= 1.0e-12;
    core::log_info("[verify:" + label + "] phase A (uniform fixed point, "
                   "r0=" + cyl_format_double(gate_r0) + ") drift=" +
                   cyl_format_double(drift) + " -> " +
                   (pass_a ? "ok" : "FAIL"));
  }

  // ---- Phase B: cold-start equilibration to the Tr plateau ----
  const char* diag_te0 = std::getenv("TENRYU_SN_MARSHAK_DIAG_TE0");
  const double Te0 = (diag_te0 != nullptr && diag_te0[0] != 0)
                         ? std::atof(diag_te0)
                         : 1.0;
  core::State state = core::State::allocate(cfg);
  if (state.mesh.geometry_code != 1) {
    core::log_info("[verify:" + label + "] FAILED (geometry_code=" +
                   std::to_string(state.mesh.geometry_code) +
                   " want=1 — State::allocate did not bind Mesh.geometry_1d)");
    return false;
  }
  const double E0 = core::constants::a_eV * Te0 * Te0 * Te0 * Te0;
  fill_cyl_state(state, node, vol, E0, Te0, cv_e);
  double max_rel = std::numeric_limits<double>::infinity();
  int steps = 0;
  const char* diag_max_steps = std::getenv("TENRYU_SN_MARSHAK_DIAG_MAX_STEPS");
  const int max_steps = (diag_max_steps != nullptr && diag_max_steps[0] != 0)
                            ? std::atoi(diag_max_steps)
                            : 4000;
  for (; steps < max_steps && max_rel > 1.0e-6; ++steps) {
    radiation::advance_radiation_step_sn_1d(
        state, cfg, planck, cfg.materials.materials.front(), dt);
    if (steps % 50 != 49) {
      continue;
    }
    std::vector<double> out_E;
    std::vector<double> out_Te;
    state.rad_E.copy_to_host(out_E);
    state.Te.copy_to_host(out_Te);
    max_rel = 0.0;
    for (const double E : out_E) {
      max_rel = std::max(max_rel, std::abs(E - E_target) / E_target);
    }
    for (const double T : out_Te) {
      max_rel = std::max(max_rel, std::abs(T - Tr) / Tr);
    }
    if (steps % 1000 == 999) {
      core::log_info(
          "[verify:" + label + "] DIAG step=" + std::to_string(steps) +
          " E0=" + cyl_format_double(out_E.front()) +
          " En=" + cyl_format_double(out_E.back()) +
          " Te0=" + cyl_format_double(out_Te.front()) +
          " Ten=" + cyl_format_double(out_Te.back()) +
          " target=" + cyl_format_double(E_target));
    }
  }
  std::vector<double> out_E_final;
  state.rad_E.copy_to_host(out_E_final);
  const double outer_rel =
      std::abs(out_E_final.back() - E_target) / E_target;
  const bool pass_b = outer_rel < 1.0e-5 && max_rel < 1.0e-5;
  core::log_info("[verify:" + label + "] phase B steps=" +
                 std::to_string(steps) + " outer_rel=" +
                 cyl_format_double(outer_rel) + " max_rel=" +
                 cyl_format_double(max_rel) +
                 " marshak_in_step=" +
                 cyl_format_double(state.sn_marshak_in_step));
  const bool pass = pass_a && pass_b;
  core::log_info("[verify:" + label + "] " + (pass ? "PASSED" : "FAILED"));
  return pass;
}

}  // namespace tenryu::drivers
