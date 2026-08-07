#include "laser/laser_mesh.cuh"
#include "laser/laser_mesh_bodies.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

#include "core/constants.hpp"
#include "core/device_pack.hpp"
#include "core/error.hpp"
#include "laser/cbet_lm_fields.cuh"
#include "laser/ib_absorption.cuh"
#include "laser/laser_phys_ext.cuh"
#include "laser/refraction.cuh"

namespace tenryu::laser {
namespace {

constexpr double kProtonMass = 1.6726219e-24;  // must match core::constants::proton_mass
constexpr double kGradedCoreRatio = 1.08;
constexpr double kGradedCoronaRatio = 1.05;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

double average_outer_cells(const std::vector<double>& values,
                           const std::vector<std::uint8_t>& cell_is_void,
                           const int outer_surface_cell,
                           const int span,
                           const double fallback) {
  if (outer_surface_cell < 0 || values.empty() || cell_is_void.empty()) {
    return fallback;
  }
  double sum = 0.0;
  double w_sum = 0.0;
  const int n_cells = static_cast<int>(values.size());
  for (int k = 0; k < span; ++k) {
    const int c = outer_surface_cell - k;
    if (c < 0 || c >= n_cells) {
      break;
    }
    if (cell_is_void[static_cast<std::size_t>(c)] != 0U) {
      continue;
    }
    const double w = 1.0 / static_cast<double>(k + 1);
    sum += w * values[static_cast<std::size_t>(c)];
    w_sum += w;
  }
  if (!(w_sum > 0.0)) {
    return fallback;
  }
  return sum / w_sum;
}

double compute_ghost_sound_speed_cm_s(const double Te_eV,
                                      const double zbar_eff,
                                      const double A_eff) {
  const double Te_safe = std::max(Te_eV, 0.0);
  const double z_safe = std::max(zbar_eff, 1.0);
  const double A_safe = std::max(A_eff, 1.0e-30);
  return std::sqrt(z_safe * core::constants::eV_to_erg * Te_safe /
                   (A_safe * kProtonMass));
}

int locate_interval_centers(const std::vector<double>& centers, const double x) {
  const int n = static_cast<int>(centers.size());
  if (n <= 2) {
    return 0;
  }
  if (x <= centers.front()) {
    return 0;
  }
  if (x >= centers.back()) {
    return n - 2;
  }
  auto it = std::upper_bound(centers.begin(), centers.end(), x);
  const int idx = static_cast<int>(std::distance(centers.begin(), it)) - 1;
  return std::max(0, std::min(n - 2, idx));
}

struct CriticalSurfaceEstimate1D {
  int fcrit = -1;
  double r_face = -1.0;
  double r_interp = -1.0;
};

CriticalSurfaceEstimate1D estimate_critical_surface_1d(
    const std::vector<double>& n_hat_cell,
    const std::vector<double>& r_edges) {
  const int n_cells = static_cast<int>(n_hat_cell.size());
  if (n_cells <= 0 || static_cast<int>(r_edges.size()) != n_cells + 1) {
    return {};
  }

  int outermost_critical = -1;
  for (int c = 0; c < n_cells; ++c) {
    if (n_hat_cell[static_cast<std::size_t>(c)] >= 1.0) {
      outermost_critical = c;
    }
  }
  if (outermost_critical < 0) {
    return {};
  }

  int fcrit = -1;
  for (int c = outermost_critical; c < n_cells; ++c) {
    const bool this_supercritical = (n_hat_cell[static_cast<std::size_t>(c)] >= 1.0);
    const bool next_subcritical =
        (c + 1 >= n_cells) || (n_hat_cell[static_cast<std::size_t>(c + 1)] < 1.0);
    if (this_supercritical && next_subcritical) {
      fcrit = c + 1;
      break;
    }
  }
  if (fcrit < 0 || fcrit >= static_cast<int>(r_edges.size())) {
    return {};
  }

  const double r_face = r_edges[static_cast<std::size_t>(fcrit)];
  double r_interp = r_face;
  if (fcrit > 0 && fcrit < n_cells) {
    const int c_hi = fcrit - 1;
    const int c_lo = fcrit;
    const double n_hi = n_hat_cell[static_cast<std::size_t>(c_hi)];
    const double n_lo = n_hat_cell[static_cast<std::size_t>(c_lo)];
    const double r_hi = 0.5 * (r_edges[static_cast<std::size_t>(c_hi)] +
                               r_edges[static_cast<std::size_t>(c_hi + 1)]);
    const double r_lo = 0.5 * (r_edges[static_cast<std::size_t>(c_lo)] +
                               r_edges[static_cast<std::size_t>(c_lo + 1)]);
    if (n_hi > 1.0 && n_lo > 0.0 && n_lo < 1.0 && r_lo > r_hi) {
      const double denom = std::log(n_lo / n_hi);
      if (std::isfinite(denom) && std::abs(denom) > 1.0e-12) {
        const double theta = std::clamp(std::log(1.0 / n_hi) / denom, 0.0, 1.0);
        r_interp = std::clamp(r_hi + theta * (r_lo - r_hi), r_hi, r_lo);
      }
    }
  }

  return CriticalSurfaceEstimate1D{fcrit, r_face, r_interp};
}

std::vector<double> compute_cell_effective_A(const LaserMesh& mesh,
                                             const core::State& state,
                                             const std::size_t n_cells) {
  std::vector<double> A_eff(
      n_cells, std::max(mesh.material_A, 1.0e-12));
  const int n_mat = static_cast<int>(mesh.material_A_list.size());
  if (n_cells == 0 || n_mat <= 0) {
    return A_eff;
  }

  const std::size_t expected =
      n_cells * static_cast<std::size_t>(n_mat);
  if (state.volFrac.size() != expected) {
    return A_eff;
  }

  std::vector<double> volfrac(state.volFrac.size(), 0.0);
  state.volFrac.copy_to_host(volfrac.data());
  for (std::size_t c = 0; c < n_cells; ++c) {
    const std::size_t base = c * static_cast<std::size_t>(n_mat);
    double frac_sum = 0.0;
    double inv_A_c = 0.0;
    for (int m = 0; m < n_mat; ++m) {
      const double frac = std::max(volfrac[base + static_cast<std::size_t>(m)], 0.0);
      frac_sum += frac;
      const double A_m =
          std::max(mesh.material_A_list[static_cast<std::size_t>(m)], 1.0e-12);
      inv_A_c += frac / A_m;
    }
    if (frac_sum > 1.0e-30) {
      inv_A_c /= frac_sum;
    }
    if (std::isfinite(inv_A_c) && inv_A_c > 1.0e-30) {
      A_eff[c] = 1.0 / inv_A_c;
    }
  }
  return A_eff;
}

std::vector<double> build_uniform_nodes(const double x0,
                                        const double x1,
                                        const int n_cells) {
  TENRYU_ASSERT(n_cells > 0, "LaserMesh requires n_cells > 0 for node construction");
  std::vector<double> nodes(static_cast<std::size_t>(n_cells + 1), 0.0);
  const double dx = (x1 - x0) / static_cast<double>(n_cells);
  for (int i = 0; i <= n_cells; ++i) {
    nodes[static_cast<std::size_t>(i)] = x0 + dx * static_cast<double>(i);
  }
  return nodes;
}

static std::vector<double> geometric_widths_from_interface(const double L, double d0, double g) {
  std::vector<double> out;
  if (!(L > 0.0)) {
    return out;
  }
  d0 = std::max(d0, 1.0e-12);
  g = std::max(g, 1.0001);
  double used = 0.0;
  double d = d0;
  while (used + d < L) {
    out.push_back(d);
    used += d;
    d *= g;
  }
  const double tail = L - used;
  if (tail > 0.0) {
    if (!out.empty() && tail < 0.35 * out.back()) {
      out.back() += tail;
    } else {
      out.push_back(tail);
    }
  }
  return out;
}

static void append_uniform_widths(std::vector<double>& w, const double L, const double d_target) {
  if (!(L > 0.0)) {
    return;
  }
  const int n = std::max(1, static_cast<int>(std::ceil(L / std::max(d_target, 1.0e-12))));
  const double d = L / static_cast<double>(n);
  for (int k = 0; k < n; ++k) {
    w.push_back(d);
  }
}

std::vector<double> build_graded_nodes_1d(const double R_crit,
                                          const double dR_fine,
                                          const double R_max,
                                          const int nr_max,
                                          const double g_core,
                                          const double g_corona) {
  const double R_max_s = std::max(R_max, 1.0e-12);
  const double R_crit_s = std::clamp(R_crit, 0.0, R_max_s);
  const double dF_base = std::max(dR_fine, 1.0e-12);
  const int nr_max_s = std::max(nr_max, 4);

  constexpr int kFineSideTarget = 64;
  auto enforce_min_cells = [&](std::vector<double> nodes) {
    if (static_cast<int>(nodes.size()) - 1 >= 4) {
      return nodes;
    }
    return build_uniform_nodes(0.0, R_max_s, 4);
  };

  auto build_for_scale = [&](const double s) -> std::vector<double> {
    const double dF = dF_base * std::max(s, 1.0);
    const double delta_target = kFineSideTarget * dF;
    const double delta_in = std::min(delta_target, R_crit_s);
    const double delta_out = std::min(delta_target, R_max_s - R_crit_s);
    const double R_left = R_crit_s - delta_in;
    const double R_right = R_crit_s + delta_out;

    std::vector<double> widths;
    widths.reserve(4096);

    {
      auto w_if = geometric_widths_from_interface(R_left, dF, g_core);
      std::reverse(w_if.begin(), w_if.end());
      widths.insert(widths.end(), w_if.begin(), w_if.end());
    }

    append_uniform_widths(widths, R_crit_s - R_left, dF);
    append_uniform_widths(widths, R_right - R_crit_s, dF);

    {
      auto w_if = geometric_widths_from_interface(R_max_s - R_right, dF, g_corona);
      widths.insert(widths.end(), w_if.begin(), w_if.end());
    }

    std::vector<double> nodes;
    nodes.reserve(widths.size() + 1);
    nodes.push_back(0.0);
    for (const double w : widths) {
      nodes.push_back(nodes.back() + std::max(w, 1.0e-12));
    }
    nodes.back() = R_max_s;
    return nodes;
  };

  auto first = enforce_min_cells(build_for_scale(1.0));
  if (static_cast<int>(first.size()) - 1 <= nr_max_s) {
    return first;
  }

  double lo = 1.0;
  double hi = 2.0;
  while (static_cast<int>(build_for_scale(hi).size()) - 1 > nr_max_s) {
    hi *= 2.0;
  }
  for (int it = 0; it < 40; ++it) {
    const double mid = 0.5 * (lo + hi);
    if (static_cast<int>(build_for_scale(mid).size()) - 1 > nr_max_s) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return enforce_min_cells(build_for_scale(hi));
}

std::vector<double> build_mirrored_Z_from_R(const std::vector<double>& node_R) {
  const int nr = static_cast<int>(node_R.size()) - 1;
  std::vector<double> node_Z(static_cast<std::size_t>(2 * nr + 1), 0.0);
  for (int k = 0; k <= nr; ++k) {
    node_Z[static_cast<std::size_t>(k)] = -node_R[static_cast<std::size_t>(nr - k)];
  }
  for (int k = 1; k <= nr; ++k) {
    node_Z[static_cast<std::size_t>(nr + k)] = node_R[static_cast<std::size_t>(k)];
  }
  return node_Z;
}

struct DynamicMeshParams1D {
  double R_max;
  double dR_fine;
  int nr;
  int nz;
  double R_crit;
  std::vector<double> node_R;
};

DynamicMeshParams1D compute_dynamic_mesh_params_1d(
    const std::vector<double>& rho,
    const std::vector<double>& zbar,
    const std::vector<double>& r_edges,
    const std::vector<double>& A_eff_cell,
    const std::vector<std::uint8_t>& cell_is_void,
    const double n_crit,
    const double mesh_factor,
    const double rmax_n_hat_threshold,
    const double r_max_factor,
    const double target_radius,
    const int nr_max) {
  const int n_cells = static_cast<int>(rho.size());
  TENRYU_ASSERT(static_cast<int>(zbar.size()) == n_cells,
                "compute_dynamic_mesh_params_1d zbar size mismatch");
  TENRYU_ASSERT(static_cast<int>(A_eff_cell.size()) == n_cells,
                "compute_dynamic_mesh_params_1d A_eff_cell size mismatch");
  TENRYU_ASSERT(static_cast<int>(cell_is_void.size()) == n_cells,
                "compute_dynamic_mesh_params_1d cell_is_void size mismatch");
  TENRYU_ASSERT(static_cast<int>(r_edges.size()) == n_cells + 1,
                "compute_dynamic_mesh_params_1d r_edges size mismatch");

  const double n_crit_safe = std::max(n_crit, 1.0e-30);
  std::vector<double> n_hat_cell(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    if (cell_is_void[static_cast<std::size_t>(c)] != 0U) {
      n_hat_cell[static_cast<std::size_t>(c)] = 0.0;
      continue;
    }
    const double A_eff = std::max(A_eff_cell[static_cast<std::size_t>(c)], 1.0e-30);
    const double n_e = std::max(0.0, rho[static_cast<std::size_t>(c)]) *
                       std::max(0.0, zbar[static_cast<std::size_t>(c)]) /
                       (A_eff * kProtonMass);
    n_hat_cell[static_cast<std::size_t>(c)] = std::max(0.0, n_e / n_crit_safe);
  }

  const CriticalSurfaceEstimate1D crit_est =
      estimate_critical_surface_1d(n_hat_cell, r_edges);
  const int fcrit = crit_est.fcrit;

  const double fallback_r_max = r_max_factor * std::max(target_radius, 1.0e-12);
  int outermost_threshold = -1;
  for (int c = 0; c < n_cells; ++c) {
    if (n_hat_cell[static_cast<std::size_t>(c)] >= rmax_n_hat_threshold) {
      outermost_threshold = c;
    }
  }
  double R_max = (outermost_threshold >= 0)
                     ? r_max_factor * r_edges[static_cast<std::size_t>(outermost_threshold + 1)]
                     : fallback_r_max;

  const double R_crit_raw =
      (fcrit >= 0 && fcrit < static_cast<int>(r_edges.size()))
          ? r_edges[static_cast<std::size_t>(fcrit)]
          : (0.5 * R_max);
  double min_dr_crit = std::numeric_limits<double>::max();
  double min_dr_global = std::numeric_limits<double>::max();
  const double r_window = 0.05 * std::max(R_crit_raw, 1.0e-12);
  for (int c = 0; c < n_cells; ++c) {
    const double r_l = r_edges[static_cast<std::size_t>(c)];
    const double r_r = r_edges[static_cast<std::size_t>(c + 1)];
    const double dr = r_r - r_l;
    if (!(dr > 0.0) || !std::isfinite(dr)) {
      continue;
    }
    min_dr_global = std::min(min_dr_global, dr);
    const double r_c = 0.5 * (r_l + r_r);
    const bool near_fcrit = (fcrit >= 0) ? (std::abs(c - fcrit) <= 10) : false;
    const bool near_rcrit = std::abs(r_c - R_crit_raw) <= r_window;
    if (near_fcrit || near_rcrit) {
      min_dr_crit = std::min(min_dr_crit, dr);
    }
  }
  if (!(min_dr_crit > 0.0) || !std::isfinite(min_dr_crit)) {
    min_dr_crit = min_dr_global;
  }
  if (!(min_dr_crit > 0.0) || !std::isfinite(min_dr_crit)) {
    min_dr_crit = std::max(R_max / 4.0, 1.0e-12);
  }

  R_max = std::max(R_max, 4.0 * min_dr_crit);
  const double R_crit = std::clamp(R_crit_raw, 0.0, R_max);
  const double dR_fine = std::max(mesh_factor * min_dr_crit, 1.0e-12);
  std::vector<double> node_R =
      build_graded_nodes_1d(R_crit, dR_fine, R_max, nr_max, kGradedCoreRatio,
                            kGradedCoronaRatio);
  const int nr = static_cast<int>(node_R.size()) - 1;
  const int nz = 2 * nr;
  const double R_max_exact = node_R.back();
  return DynamicMeshParams1D{R_max_exact, dR_fine, nr, nz, R_crit, std::move(node_R)};
}

using laser_mesh_bodies::compute_gradient_kernel_body;
using laser_mesh_bodies::compute_radial_gradient_kernel_body;
using laser_mesh_bodies::compute_smooth_kappa_ext_kernel_body;
using laser_mesh_bodies::compute_smooth_kappa_kernel_body;
using laser_mesh_bodies::ema_smooth_n_hat_kernel_body;
using laser_mesh_bodies::extract_radial_profile_kernel_body;
using laser_mesh_bodies::extract_radial_te_kernel_body;
using laser_mesh_bodies::map_hydro_to_laser_1d_kernel_body;

__global__ void map_hydro_to_laser_1d_kernel(
    const double* __restrict__ node_R,
    const double* __restrict__ node_Z,
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    const double* __restrict__ zbar,
    const double* __restrict__ A_eff_cell,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ r_edges,
    double* __restrict__ n_hat_out,
    double* __restrict__ n_hat_raw_out,
    double* __restrict__ Te_out,
    double* __restrict__ Z_out,
    const int n_nodes_total,
    const int n_nodes_z,
    const int n_cells,
    const double n_crit_safe,
    const int use_ghost_corona,
    const int outer_surface_cell,
    const double ghost_ne_inner,
    const double ghost_scale_length,
    const double ghost_ne_min,
    const double r_surface_outer,
    const double r_ghost_outer,
    const double Te_anchor,
    const double zbar_anchor,
    const double ghost_zbar_min,
    const double ghost_zbar_max,
    const double ghost_Te_min_eV,
    const int critical_clip,
    const double n_hat_margin,
    const int fcrit_cell,
    const double r_crit_interp,
    const double ne_fcrit_center,
    const double r_fcrit_center) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_nodes_total) {
    return;
  }

  map_hydro_to_laser_1d_kernel_body(
      idx, node_R, node_Z, rho, Te, zbar, A_eff_cell, cell_is_void, r_edges,
      n_hat_out, n_hat_raw_out, Te_out, Z_out, n_nodes_total, n_nodes_z,
      n_cells, n_crit_safe, use_ghost_corona, outer_surface_cell,
      ghost_ne_inner, ghost_scale_length, ghost_ne_min, r_surface_outer,
      r_ghost_outer, Te_anchor, zbar_anchor, ghost_zbar_min, ghost_zbar_max,
      ghost_Te_min_eV, critical_clip, n_hat_margin, fcrit_cell,
      r_crit_interp, ne_fcrit_center, r_fcrit_center);
}

__global__ void ema_smooth_n_hat_kernel(double* __restrict__ n_hat,
                                        const double* __restrict__ prev_n_hat,
                                        const int n_nodes_total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_nodes_total) {
    return;
  }

  ema_smooth_n_hat_kernel_body(idx, n_hat, prev_n_hat, n_nodes_total);
}

void ensure_hydro_mapping_capacity(LaserMesh& mesh, const int n_cells) {
  if (n_cells <= mesh.hydro_cell_capacity) {
    return;
  }

  double* new_A_eff = nullptr;
  std::uint8_t* new_cell_is_void = nullptr;
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&new_A_eff),
                               static_cast<std::size_t>(n_cells) * sizeof(double));
  TENRYU_ASSERT(err == cudaSuccess,
                "ensure_hydro_mapping_capacity hydro_A_eff_device cudaMalloc failed");
  err = cudaMalloc(reinterpret_cast<void**>(&new_cell_is_void),
                   static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t));
  if (err != cudaSuccess) {
    static_cast<void>(cudaFree(new_A_eff));
    TENRYU_ASSERT(false,
                  "ensure_hydro_mapping_capacity hydro_cell_is_void_device cudaMalloc failed");
  }

  if (mesh.hydro_A_eff_device != nullptr) {
    cuda_check(cudaFree(mesh.hydro_A_eff_device),
               "ensure_hydro_mapping_capacity hydro_A_eff_device cudaFree failed");
  }
  if (mesh.hydro_cell_is_void_device != nullptr) {
    cuda_check(cudaFree(mesh.hydro_cell_is_void_device),
               "ensure_hydro_mapping_capacity hydro_cell_is_void_device cudaFree failed");
  }
  mesh.hydro_A_eff_device = new_A_eff;
  mesh.hydro_cell_is_void_device = new_cell_is_void;
  mesh.hydro_cell_capacity = n_cells;
}

__global__ void compute_gradient_kernel(double* __restrict__ grad_R,
                                        double* __restrict__ grad_Z,
                                        const double* __restrict__ n_hat,
                                        const double* __restrict__ node_R,
                                        const double* __restrict__ node_Z,
                                        const int n_nodes_r,
                                        const int n_nodes_z) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_nodes = n_nodes_r * n_nodes_z;
  if (idx >= n_nodes) {
    return;
  }

  compute_gradient_kernel_body(idx, grad_R, grad_Z, n_hat, node_R, node_Z,
                               n_nodes_r, n_nodes_z);
}

__global__ void compute_smooth_kappa_kernel(double* __restrict__ smooth_kappa_factor,
                                            const double* __restrict__ n_hat,
                                            const double* __restrict__ T_e,
                                            const double* __restrict__ Zbar,
                                            const double lambda_cm,
                                            const double eps_n,
                                            const double coulomb_log_floor,
                                            const int n_nodes_total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_nodes_total) {
    return;
  }
  compute_smooth_kappa_kernel_body(idx, smooth_kappa_factor, n_hat, T_e, Zbar,
                                   lambda_cm, eps_n, coulomb_log_floor,
                                   n_nodes_total);
}

__global__ void compute_smooth_kappa_ext_kernel(
    double* __restrict__ smooth_kappa_factor,
    const double* __restrict__ n_hat,
    const double* __restrict__ T_e,
    const double* __restrict__ Zbar,
    const double lambda_cm,
    const double eps_n,
    const double coulomb_log_floor,
    const int n_nodes_total,
    const LaserPhysExtOptions opt) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  compute_smooth_kappa_ext_kernel_body(
      idx, smooth_kappa_factor, n_hat, T_e, Zbar, lambda_cm, eps_n,
      coulomb_log_floor, n_nodes_total, opt);
}

__global__ void extract_radial_profile_kernel(
    double* __restrict__ radial_node_r,
    double* __restrict__ radial_n_hat,
    double* __restrict__ radial_n_hat_raw,
    double* __restrict__ radial_smooth_kappa,
    const double* __restrict__ node_R,
    const double* __restrict__ n_hat,
    const double* __restrict__ n_hat_raw,
    const double* __restrict__ smooth_kappa_factor,
    const int n_nodes_r,
    const int n_nodes_z,
    const int j_center,
    const int copy_smooth) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_nodes_r) {
    return;
  }

  extract_radial_profile_kernel_body(i, radial_node_r, radial_n_hat,
                                     radial_n_hat_raw, radial_smooth_kappa,
                                     node_R, n_hat, n_hat_raw,
                                     smooth_kappa_factor, n_nodes_r,
                                     n_nodes_z, j_center, copy_smooth);
}

__global__ void extract_radial_te_kernel(
    double* __restrict__ radial_T_e,
    const double* __restrict__ T_e,
    const int n_nodes_z,
    const int j_center,
    const int n_nodes_r) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_nodes_r) {
    return;
  }

  extract_radial_te_kernel_body(
      i, radial_T_e, T_e, n_nodes_z, j_center, n_nodes_r);
}

__global__ void compute_radial_gradient_kernel(double* __restrict__ radial_dn_dr,
                                               const double* __restrict__ radial_node_r,
                                               const double* __restrict__ radial_n_hat,
                                               const int n_nodes_r) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_nodes_r) {
    return;
  }

  compute_radial_gradient_kernel_body(i, radial_dn_dr, radial_node_r,
                                      radial_n_hat, n_nodes_r);
}

void refresh_radial_profile_1d(LaserMesh& mesh,
                               const bool copy_smooth,
                               cudaStream_t stream,
                               double* radial_T_e_or_null = nullptr) {
  if (mesh.n_nodes_r <= 0 || mesh.n_nodes_z <= 0 || mesh.radial_node_r == nullptr ||
      mesh.radial_n_hat == nullptr || mesh.radial_n_hat_raw == nullptr ||
      mesh.radial_smooth_kappa == nullptr || mesh.radial_dn_dr == nullptr) {
    return;
  }

  mesh.radial_n_nodes = mesh.n_nodes_r;
  const int block = 256;
  const int grid = (mesh.n_nodes_r + block - 1) / block;
  const int j_center = mesh.n_nodes_z / 2;
  extract_radial_profile_kernel<<<grid, block, 0, stream>>>(
      mesh.radial_node_r, mesh.radial_n_hat, mesh.radial_n_hat_raw,
      mesh.radial_smooth_kappa, mesh.node_R, mesh.n_e_hat, mesh.n_e_hat_raw,
      copy_smooth ? mesh.smooth_kappa_factor : nullptr, mesh.n_nodes_r, mesh.n_nodes_z,
      j_center, copy_smooth ? 1 : 0);
  cuda_check(cudaGetLastError(), "refresh_radial_profile_1d extract kernel launch failed");
  if (radial_T_e_or_null != nullptr) {
    extract_radial_te_kernel<<<grid, block, 0, stream>>>(
        radial_T_e_or_null, mesh.T_e, mesh.n_nodes_z, j_center,
        mesh.n_nodes_r);
    cuda_check(cudaGetLastError(),
               "refresh_radial_profile_1d Te extract kernel launch failed");
  }

  compute_radial_gradient_kernel<<<grid, block, 0, stream>>>(
      mesh.radial_dn_dr, mesh.radial_node_r, mesh.radial_n_hat, mesh.n_nodes_r);
  cuda_check(cudaGetLastError(), "refresh_radial_profile_1d gradient kernel launch failed");
}

}  // namespace

LaserMesh::~LaserMesh() {
  release();
}

LaserMesh::LaserMesh(LaserMesh&& other) noexcept {
  *this = std::move(other);
}

LaserMesh& LaserMesh::operator=(LaserMesh&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();

  nr = other.nr;
  nz = other.nz;
  n_nodes_r = other.n_nodes_r;
  n_nodes_z = other.n_nodes_z;
  nr_capacity = other.nr_capacity;
  nz_capacity = other.nz_capacity;
  n_nodes_r_capacity = other.n_nodes_r_capacity;
  n_nodes_z_capacity = other.n_nodes_z_capacity;
  underuse_steps = other.underuse_steps;
  steps_since_realloc = other.steps_since_realloc;
  radial_n_nodes = other.radial_n_nodes;

  node_R = other.node_R;
  node_Z = other.node_Z;
  n_e_hat = other.n_e_hat;
  n_e_hat_raw = other.n_e_hat_raw;
  T_e = other.T_e;
  Zbar = other.Zbar;
  smooth_kappa_factor = other.smooth_kappa_factor;
  grad_n_hat_R = other.grad_n_hat_R;
  grad_n_hat_Z = other.grad_n_hat_Z;
  radial_node_r = other.radial_node_r;
  radial_n_hat = other.radial_n_hat;
  radial_n_hat_raw = other.radial_n_hat_raw;
  radial_smooth_kappa = other.radial_smooth_kappa;
  radial_T_e = other.radial_T_e;
  radial_dn_dr = other.radial_dn_dr;
  deposit = other.deposit;
  prev_n_hat_device = other.prev_n_hat_device;
  hydro_A_eff_device = other.hydro_A_eff_device;
  hydro_cell_is_void_device = other.hydro_cell_is_void_device;
  hydro_cell_capacity = other.hydro_cell_capacity;
  zeff_table_dev = other.zeff_table_dev;
  zeff_table_capacity = other.zeff_table_capacity;
  scratch_step_histogram = other.scratch_step_histogram;
  scratch_step_tally_slab = other.scratch_step_tally_slab;
  scratch_unabsorbed = other.scratch_unabsorbed;
  scratch_tail_closure_count = other.scratch_tail_closure_count;
  scratch_tail_closure_absorbed_power = other.scratch_tail_closure_absorbed_power;
  scratch_critical_surface_hit_count = other.scratch_critical_surface_hit_count;
  scratch_ra_power_total = other.scratch_ra_power_total;
  scratch_error_flags = other.scratch_error_flags;
  scratch_step_pack_device = other.scratch_step_pack_device;
  scratch_step_pack_host = other.scratch_step_pack_host;
  scratch_step_pack_capacity = other.scratch_step_pack_capacity;
  scratch_per_ray_step_count = other.scratch_per_ray_step_count;
  scratch_sorted_step_count = other.scratch_sorted_step_count;
  scratch_per_warp_step_max = other.scratch_per_warp_step_max;
  scratch_per_warp_step_sum = other.scratch_per_warp_step_sum;
  hot_e_capture = other.hot_e_capture;
  scratch_per_ray_step_capacity = other.scratch_per_ray_step_capacity;
  scratch_per_warp_step_capacity = other.scratch_per_warp_step_capacity;
  hot_e_capture_capacity = other.hot_e_capture_capacity;

  R_max = other.R_max;
  Z_min = other.Z_min;
  Z_max = other.Z_max;
  n_crit = other.n_crit;
  n_hat_margin = other.n_hat_margin;
  dx_min = other.dx_min;
  target_radius = other.target_radius;
  material_A = other.material_A;
  material_A_list = std::move(other.material_A_list);
  ghost_corona_enabled = other.ghost_corona_enabled;
  ghost_n_out = other.ghost_n_out;
  ghost_ne_min_frac = other.ghost_ne_min_frac;
  ghost_ne_max_frac = other.ghost_ne_max_frac;
  ghost_Te_min_eV = other.ghost_Te_min_eV;
  ghost_zbar_min = other.ghost_zbar_min;
  ghost_zbar_max = other.ghost_zbar_max;
  ghost_handoff_cells = other.ghost_handoff_cells;
  ghost_handoff_decay = other.ghost_handoff_decay;
  ghost_transition_enabled = other.ghost_transition_enabled;
  ghost_transition_resolved_nhat = other.ghost_transition_resolved_nhat;
  ghost_transition_resolved_cells = other.ghost_transition_resolved_cells;
  ghost_transition_density_exponent = other.ghost_transition_density_exponent;
  last_ghost_transition_blend = other.last_ghost_transition_blend;
  last_ghost_transition_resolved_cells = other.last_ghost_transition_resolved_cells;
  last_trace_unabsorbed_power = other.last_trace_unabsorbed_power;
  last_transfer_blocked_power = other.last_transfer_blocked_power;
  last_unabsorbed_power = other.last_unabsorbed_power;
  last_ra_power = other.last_ra_power;
  last_commanded_energy = other.last_commanded_energy;
  last_tail_closure_count = other.last_tail_closure_count;
  last_tail_closure_absorbed_power = other.last_tail_closure_absorbed_power;
  last_critical_surface_hit_count = other.last_critical_surface_hit_count;
  last_cbet_exchanged_power = other.last_cbet_exchanged_power;
  last_cbet_ledger_residual = other.last_cbet_ledger_residual;
  last_cbet_conv_final = other.last_cbet_conv_final;
  last_cbet_clamp_count = other.last_cbet_clamp_count;
  last_cbet_overflow_rays = other.last_cbet_overflow_rays;
  last_cbet_iterations = other.last_cbet_iterations;
  last_cbet_converged = other.last_cbet_converged;
  prev_n_hat_host = std::move(other.prev_n_hat_host);
  prev_n_hat_valid = other.prev_n_hat_valid;
  ray_steps_previous = std::move(other.ray_steps_previous);
  ray_steps_output = std::move(other.ray_steps_output);
  ray_order = std::move(other.ray_order);

  other.nr = 0;
  other.nz = 0;
  other.n_nodes_r = 0;
  other.n_nodes_z = 0;
  other.nr_capacity = 0;
  other.nz_capacity = 0;
  other.n_nodes_r_capacity = 0;
  other.n_nodes_z_capacity = 0;
  other.underuse_steps = 0;
  other.steps_since_realloc = 0;
  other.radial_n_nodes = 0;
  other.node_R = nullptr;
  other.node_Z = nullptr;
  other.n_e_hat = nullptr;
  other.n_e_hat_raw = nullptr;
  other.T_e = nullptr;
  other.Zbar = nullptr;
  other.smooth_kappa_factor = nullptr;
  other.grad_n_hat_R = nullptr;
  other.grad_n_hat_Z = nullptr;
  other.radial_node_r = nullptr;
  other.radial_n_hat = nullptr;
  other.radial_n_hat_raw = nullptr;
  other.radial_smooth_kappa = nullptr;
  other.radial_T_e = nullptr;
  other.radial_dn_dr = nullptr;
  other.deposit = nullptr;
  other.prev_n_hat_device = nullptr;
  other.hydro_A_eff_device = nullptr;
  other.hydro_cell_is_void_device = nullptr;
  other.hydro_cell_capacity = 0;
  other.zeff_table_dev = nullptr;
  other.zeff_table_capacity = 0;
  other.scratch_step_histogram = nullptr;
  other.scratch_step_tally_slab = nullptr;
  other.scratch_unabsorbed = nullptr;
  other.scratch_tail_closure_count = nullptr;
  other.scratch_tail_closure_absorbed_power = nullptr;
  other.scratch_critical_surface_hit_count = nullptr;
  other.scratch_ra_power_total = nullptr;
  other.scratch_error_flags = nullptr;
  other.scratch_step_pack_device = nullptr;
  other.scratch_step_pack_host = nullptr;
  other.scratch_step_pack_capacity = 0;
  other.scratch_per_ray_step_count = nullptr;
  other.scratch_sorted_step_count = nullptr;
  other.scratch_per_warp_step_max = nullptr;
  other.scratch_per_warp_step_sum = nullptr;
  other.hot_e_capture = nullptr;
  other.scratch_per_ray_step_capacity = 0;
  other.scratch_per_warp_step_capacity = 0;
  other.hot_e_capture_capacity = 0;
  other.ghost_corona_enabled = false;
  other.ghost_n_out = 0;
  other.ghost_transition_enabled = false;
  other.last_ghost_transition_blend = 0.0;
  other.last_ghost_transition_resolved_cells = 0;
  other.last_trace_unabsorbed_power = 0.0;
  other.last_transfer_blocked_power = 0.0;
  other.last_unabsorbed_power = 0.0;
  other.last_ra_power = 0.0;
  other.last_commanded_energy = 0.0;
  other.last_tail_closure_count = 0;
  other.last_tail_closure_absorbed_power = 0.0;
  other.last_critical_surface_hit_count = 0;
  other.last_cbet_exchanged_power = 0.0;
  other.last_cbet_ledger_residual = 0.0;
  other.last_cbet_conv_final = 0.0;
  other.last_cbet_clamp_count = 0;
  other.last_cbet_overflow_rays = 0;
  other.last_cbet_iterations = 0;
  other.last_cbet_converged = true;
  other.prev_n_hat_host.clear();
  other.prev_n_hat_valid = false;
  other.ray_steps_previous.clear();
  other.ray_steps_output.clear();
  other.ray_order.clear();
  return *this;
}

void LaserMesh::release() {
  if (node_R != nullptr) {
    cuda_check(cudaFree(node_R), "LaserMesh::release node_R cudaFree failed");
    node_R = nullptr;
  }
  if (node_Z != nullptr) {
    cuda_check(cudaFree(node_Z), "LaserMesh::release node_Z cudaFree failed");
    node_Z = nullptr;
  }
  if (n_e_hat != nullptr) {
    cuda_check(cudaFree(n_e_hat), "LaserMesh::release n_e_hat cudaFree failed");
    n_e_hat = nullptr;
  }
  if (n_e_hat_raw != nullptr) {
    cuda_check(cudaFree(n_e_hat_raw), "LaserMesh::release n_e_hat_raw cudaFree failed");
    n_e_hat_raw = nullptr;
  }
  if (T_e != nullptr) {
    cuda_check(cudaFree(T_e), "LaserMesh::release T_e cudaFree failed");
    T_e = nullptr;
  }
  if (Zbar != nullptr) {
    cuda_check(cudaFree(Zbar), "LaserMesh::release Zbar cudaFree failed");
    Zbar = nullptr;
  }
  if (smooth_kappa_factor != nullptr) {
    cuda_check(cudaFree(smooth_kappa_factor),
               "LaserMesh::release smooth_kappa_factor cudaFree failed");
    smooth_kappa_factor = nullptr;
  }
  if (grad_n_hat_R != nullptr) {
    cuda_check(cudaFree(grad_n_hat_R), "LaserMesh::release grad_n_hat_R cudaFree failed");
    grad_n_hat_R = nullptr;
  }
  if (grad_n_hat_Z != nullptr) {
    cuda_check(cudaFree(grad_n_hat_Z), "LaserMesh::release grad_n_hat_Z cudaFree failed");
    grad_n_hat_Z = nullptr;
  }
  if (radial_node_r != nullptr) {
    cuda_check(cudaFree(radial_node_r), "LaserMesh::release radial_node_r cudaFree failed");
    radial_node_r = nullptr;
  }
  if (radial_n_hat != nullptr) {
    cuda_check(cudaFree(radial_n_hat), "LaserMesh::release radial_n_hat cudaFree failed");
    radial_n_hat = nullptr;
  }
  if (radial_n_hat_raw != nullptr) {
    cuda_check(cudaFree(radial_n_hat_raw), "LaserMesh::release radial_n_hat_raw cudaFree failed");
    radial_n_hat_raw = nullptr;
  }
  if (radial_smooth_kappa != nullptr) {
    cuda_check(cudaFree(radial_smooth_kappa),
               "LaserMesh::release radial_smooth_kappa cudaFree failed");
    radial_smooth_kappa = nullptr;
  }
  if (radial_T_e != nullptr) {
    cuda_check(cudaFree(radial_T_e),
               "LaserMesh::release radial_T_e cudaFree failed");
    radial_T_e = nullptr;
  }
  if (radial_dn_dr != nullptr) {
    cuda_check(cudaFree(radial_dn_dr), "LaserMesh::release radial_dn_dr cudaFree failed");
    radial_dn_dr = nullptr;
  }
  if (deposit != nullptr) {
    cuda_check(cudaFree(deposit), "LaserMesh::release deposit cudaFree failed");
    deposit = nullptr;
  }
  if (prev_n_hat_device != nullptr) {
    cuda_check(cudaFree(prev_n_hat_device),
               "LaserMesh::release prev_n_hat_device cudaFree failed");
    prev_n_hat_device = nullptr;
  }
  if (hydro_A_eff_device != nullptr) {
    cuda_check(cudaFree(hydro_A_eff_device),
               "LaserMesh::release hydro_A_eff_device cudaFree failed");
    hydro_A_eff_device = nullptr;
  }
  if (hydro_cell_is_void_device != nullptr) {
    cuda_check(cudaFree(hydro_cell_is_void_device),
               "LaserMesh::release hydro_cell_is_void_device cudaFree failed");
    hydro_cell_is_void_device = nullptr;
  }
  if (zeff_table_dev != nullptr) {
    cuda_check(cudaFree(zeff_table_dev),
               "LaserMesh::release zeff_table_dev cudaFree failed");
    zeff_table_dev = nullptr;
  }
  if (scratch_step_histogram != nullptr) {
    cuda_check(cudaFree(scratch_step_histogram),
               "LaserMesh::release scratch_step_histogram cudaFree failed");
    scratch_step_histogram = nullptr;
  }
  if (scratch_step_tally_slab != nullptr) {
    cuda_check(cudaFree(scratch_step_tally_slab),
               "LaserMesh::release scratch_step_tally_slab cudaFree failed");
    scratch_step_tally_slab = nullptr;
  }
  scratch_unabsorbed = nullptr;
  scratch_tail_closure_count = nullptr;
  scratch_tail_closure_absorbed_power = nullptr;
  scratch_critical_surface_hit_count = nullptr;
  scratch_ra_power_total = nullptr;
  if (scratch_error_flags != nullptr) {
    cuda_check(cudaFree(scratch_error_flags),
               "LaserMesh::release scratch_error_flags cudaFree failed");
    scratch_error_flags = nullptr;
  }
  if (scratch_step_pack_device != nullptr) {
    cuda_check(cudaFree(scratch_step_pack_device),
               "LaserMesh::release scratch_step_pack_device cudaFree failed");
    scratch_step_pack_device = nullptr;
  }
  if (scratch_step_pack_host != nullptr) {
    cuda_check(cudaFreeHost(scratch_step_pack_host),
               "LaserMesh::release scratch_step_pack_host cudaFreeHost failed");
    scratch_step_pack_host = nullptr;
  }
  if (scratch_per_ray_step_count != nullptr) {
    cuda_check(cudaFree(scratch_per_ray_step_count),
               "LaserMesh::release scratch_per_ray_step_count cudaFree failed");
    scratch_per_ray_step_count = nullptr;
  }
  if (scratch_sorted_step_count != nullptr) {
    cuda_check(cudaFree(scratch_sorted_step_count),
               "LaserMesh::release scratch_sorted_step_count cudaFree failed");
    scratch_sorted_step_count = nullptr;
  }
  if (scratch_per_warp_step_max != nullptr) {
    cuda_check(cudaFree(scratch_per_warp_step_max),
               "LaserMesh::release scratch_per_warp_step_max cudaFree failed");
    scratch_per_warp_step_max = nullptr;
  }
  if (scratch_per_warp_step_sum != nullptr) {
    cuda_check(cudaFree(scratch_per_warp_step_sum),
               "LaserMesh::release scratch_per_warp_step_sum cudaFree failed");
    scratch_per_warp_step_sum = nullptr;
  }
  if (hot_e_capture != nullptr) {
    cuda_check(cudaFree(hot_e_capture), "LaserMesh::release hot_e_capture cudaFree failed");
    hot_e_capture = nullptr;
  }

  nr = 0;
  nz = 0;
  n_nodes_r = 0;
  n_nodes_z = 0;
  nr_capacity = 0;
  nz_capacity = 0;
  n_nodes_r_capacity = 0;
  n_nodes_z_capacity = 0;
  underuse_steps = 0;
  steps_since_realloc = 0;
  radial_n_nodes = 0;
  hydro_cell_capacity = 0;
  zeff_table_capacity = 0;
  scratch_per_ray_step_capacity = 0;
  scratch_per_warp_step_capacity = 0;
  hot_e_capture_capacity = 0;
  scratch_step_pack_capacity = 0;
  dx_min = 0.0;
  material_A_list.clear();
  last_trace_unabsorbed_power = 0.0;
  last_transfer_blocked_power = 0.0;
  last_unabsorbed_power = 0.0;
  last_ra_power = 0.0;
  last_commanded_energy = 0.0;
  last_tail_closure_count = 0;
  last_tail_closure_absorbed_power = 0.0;
  last_critical_surface_hit_count = 0;
  last_cbet_exchanged_power = 0.0;
  last_cbet_ledger_residual = 0.0;
  last_cbet_conv_final = 0.0;
  last_cbet_clamp_count = 0;
  last_cbet_overflow_rays = 0;
  last_cbet_iterations = 0;
  last_cbet_converged = true;
  prev_n_hat_host.clear();
  prev_n_hat_valid = false;
  ray_steps_previous.clear();
  ray_steps_output.clear();
  ray_order.clear();
}

bool LaserMesh::is_allocated() const {
  return node_R != nullptr && node_Z != nullptr && n_e_hat != nullptr &&
         n_e_hat_raw != nullptr && T_e != nullptr && Zbar != nullptr &&
         smooth_kappa_factor != nullptr && grad_n_hat_R != nullptr &&
         grad_n_hat_Z != nullptr && radial_node_r != nullptr &&
         radial_n_hat != nullptr && radial_n_hat_raw != nullptr &&
         radial_smooth_kappa != nullptr && radial_T_e != nullptr &&
         radial_dn_dr != nullptr && deposit != nullptr && prev_n_hat_device != nullptr;
}

int LaserMesh::n_nodes() const {
  return n_nodes_r * n_nodes_z;
}

void LaserMesh::allocate(const int nr_cells, const int nz_cells) {
  if (nr_cells <= nr_capacity && nz_cells <= nz_capacity && is_allocated()) {
    const bool size_changed = (nr != nr_cells) || (nz != nz_cells);
    nr = nr_cells;
    nz = nz_cells;
    n_nodes_r = nr + 1;
    n_nodes_z = nz + 1;
    radial_n_nodes = n_nodes_r;
    if (size_changed) {
      prev_n_hat_valid = false;
    }
    clear_deposit();
    return;
  }

  const int new_nr_capacity = std::max(nr_cells, nr_capacity);
  const int new_nz_capacity = std::max(nz_cells, nz_capacity);
  release();
  nr = nr_cells;
  nz = nz_cells;
  n_nodes_r = nr + 1;
  n_nodes_z = nz + 1;
  nr_capacity = new_nr_capacity;
  nz_capacity = new_nz_capacity;
  n_nodes_r_capacity = nr_capacity + 1;
  n_nodes_z_capacity = nz_capacity + 1;
  underuse_steps = 0;
  steps_since_realloc = 0;
  radial_n_nodes = n_nodes_r;
  const std::size_t n_nodes_capacity_total =
      static_cast<std::size_t>(n_nodes_r_capacity) *
      static_cast<std::size_t>(n_nodes_z_capacity);

  auto alloc_or_cleanup = [&](double*& ptr, const std::size_t bytes, const char* message) {
    const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), bytes);
    if (err == cudaSuccess) {
      return;
    }
    release();
    TENRYU_ASSERT(false, message);
  };

  alloc_or_cleanup(node_R, n_nodes_r_capacity * sizeof(double),
                   "LaserMesh::allocate node_R cudaMalloc failed");
  alloc_or_cleanup(node_Z, n_nodes_z_capacity * sizeof(double),
                   "LaserMesh::allocate node_Z cudaMalloc failed");
  alloc_or_cleanup(n_e_hat, n_nodes_capacity_total * sizeof(double),
                   "LaserMesh::allocate n_e_hat cudaMalloc failed");
  alloc_or_cleanup(n_e_hat_raw, n_nodes_capacity_total * sizeof(double),
                   "LaserMesh::allocate n_e_hat_raw cudaMalloc failed");
  alloc_or_cleanup(T_e, n_nodes_capacity_total * sizeof(double),
                   "LaserMesh::allocate T_e cudaMalloc failed");
  alloc_or_cleanup(Zbar, n_nodes_capacity_total * sizeof(double),
                   "LaserMesh::allocate Zbar cudaMalloc failed");
  alloc_or_cleanup(smooth_kappa_factor, n_nodes_capacity_total * sizeof(double),
                   "LaserMesh::allocate smooth_kappa_factor cudaMalloc failed");
  alloc_or_cleanup(grad_n_hat_R, n_nodes_capacity_total * sizeof(double),
                   "LaserMesh::allocate grad_n_hat_R cudaMalloc failed");
  alloc_or_cleanup(grad_n_hat_Z, n_nodes_capacity_total * sizeof(double),
                   "LaserMesh::allocate grad_n_hat_Z cudaMalloc failed");
  alloc_or_cleanup(radial_node_r,
                   static_cast<std::size_t>(n_nodes_r_capacity) * sizeof(double),
                   "LaserMesh::allocate radial_node_r cudaMalloc failed");
  alloc_or_cleanup(radial_n_hat,
                   static_cast<std::size_t>(n_nodes_r_capacity) * sizeof(double),
                   "LaserMesh::allocate radial_n_hat cudaMalloc failed");
  alloc_or_cleanup(radial_n_hat_raw,
                   static_cast<std::size_t>(n_nodes_r_capacity) * sizeof(double),
                   "LaserMesh::allocate radial_n_hat_raw cudaMalloc failed");
  alloc_or_cleanup(radial_smooth_kappa,
                   static_cast<std::size_t>(n_nodes_r_capacity) * sizeof(double),
                   "LaserMesh::allocate radial_smooth_kappa cudaMalloc failed");
  alloc_or_cleanup(radial_T_e,
                   static_cast<std::size_t>(n_nodes_r_capacity) * sizeof(double),
                   "LaserMesh::allocate radial_T_e cudaMalloc failed");
  alloc_or_cleanup(radial_dn_dr,
                   static_cast<std::size_t>(n_nodes_r_capacity) * sizeof(double),
                   "LaserMesh::allocate radial_dn_dr cudaMalloc failed");
  alloc_or_cleanup(deposit, n_nodes_capacity_total * sizeof(double),
                   "LaserMesh::allocate deposit cudaMalloc failed");
  alloc_or_cleanup(prev_n_hat_device, n_nodes_capacity_total * sizeof(double),
                   "LaserMesh::allocate prev_n_hat_device cudaMalloc failed");

  clear_deposit();
}

void LaserMesh::ensure_capacity(const int nr_new, const int nz_new) {
  if (nr_new <= nr_capacity && nz_new <= nz_capacity) {
    const bool size_changed = (nr != nr_new) || (nz != nz_new);
    nr = nr_new;
    nz = nz_new;
    n_nodes_r = nr + 1;
    n_nodes_z = nz + 1;
    radial_n_nodes = n_nodes_r;
    if (size_changed) {
      prev_n_hat_valid = false;
    }
    ++steps_since_realloc;
    if (nr_new * 2 <= nr_capacity && nz_new * 2 <= nz_capacity) {
      ++underuse_steps;
    } else {
      underuse_steps = 0;
    }
    return;
  }

  auto saved_A_list = std::move(material_A_list);
  auto saved_ray_steps_previous = std::move(ray_steps_previous);
  auto saved_ray_steps_output = std::move(ray_steps_output);
  auto saved_ray_order = std::move(ray_order);
  const double saved_unabsorbed = last_unabsorbed_power;
  const double saved_ra_power = last_ra_power;
  const double saved_commanded = last_commanded_energy;
  const std::int64_t saved_tail_closure_count = last_tail_closure_count;
  const double saved_tail_closure_absorbed = last_tail_closure_absorbed_power;
  const std::int64_t saved_critical_surface_hit_count = last_critical_surface_hit_count;

  const int new_nr_cap =
      (nr_capacity > 0) ? std::max(nr_new, nr_capacity * 2) : std::max(nr_new * 2, 4);
  const int new_nz_cap =
      (nz_capacity > 0) ? std::max(nz_new, nz_capacity * 2) : std::max(nz_new * 2, 8);
  allocate(new_nr_cap, new_nz_cap);
  material_A_list = std::move(saved_A_list);
  ray_steps_previous = std::move(saved_ray_steps_previous);
  ray_steps_output = std::move(saved_ray_steps_output);
  ray_order = std::move(saved_ray_order);
  last_unabsorbed_power = saved_unabsorbed;
  last_ra_power = saved_ra_power;
  last_commanded_energy = saved_commanded;
  last_tail_closure_count = saved_tail_closure_count;
  last_tail_closure_absorbed_power = saved_tail_closure_absorbed;
  last_critical_surface_hit_count = saved_critical_surface_hit_count;
  nr_capacity = new_nr_cap;
  nz_capacity = new_nz_cap;
  n_nodes_r_capacity = new_nr_cap + 1;
  n_nodes_z_capacity = new_nz_cap + 1;
  underuse_steps = 0;
  steps_since_realloc = 0;

  nr = nr_new;
  nz = nz_new;
  n_nodes_r = nr + 1;
  n_nodes_z = nz + 1;
  radial_n_nodes = n_nodes_r;
}

void LaserMesh::clear_deposit(cudaStream_t stream) const {
  if (deposit == nullptr) {
    return;
  }
  const std::size_t bytes = static_cast<std::size_t>(n_nodes()) * sizeof(double);
  cuda_check(cudaMemsetAsync(deposit, 0, bytes, stream),
             "LaserMesh::clear_deposit cudaMemsetAsync failed");
}

void LaserMesh::ensure_per_ray_step_scratch(const int n_rays) {
  if (n_rays <= 0 || n_rays <= scratch_per_ray_step_capacity) {
    return;
  }

  auto release_per_ray_step_scratch_noassert = [&]() {
    if (scratch_per_ray_step_count != nullptr) {
      static_cast<void>(cudaFree(scratch_per_ray_step_count));
      scratch_per_ray_step_count = nullptr;
    }
    if (scratch_sorted_step_count != nullptr) {
      static_cast<void>(cudaFree(scratch_sorted_step_count));
      scratch_sorted_step_count = nullptr;
    }
    if (scratch_per_warp_step_max != nullptr) {
      static_cast<void>(cudaFree(scratch_per_warp_step_max));
      scratch_per_warp_step_max = nullptr;
    }
    if (scratch_per_warp_step_sum != nullptr) {
      static_cast<void>(cudaFree(scratch_per_warp_step_sum));
      scratch_per_warp_step_sum = nullptr;
    }
    scratch_per_ray_step_capacity = 0;
    scratch_per_warp_step_capacity = 0;
  };
  auto free_checked = [](auto*& ptr, const char* message) {
    if (ptr == nullptr) {
      return;
    }
    cuda_check(cudaFree(ptr), message);
    ptr = nullptr;
  };
  auto release_per_ray_step_scratch = [&]() {
    free_checked(scratch_per_ray_step_count,
                 "LaserMesh::ensure_per_ray_step_scratch scratch_per_ray_step_count "
                 "cudaFree failed");
    free_checked(scratch_sorted_step_count,
                 "LaserMesh::ensure_per_ray_step_scratch scratch_sorted_step_count "
                 "cudaFree failed");
    free_checked(scratch_per_warp_step_max,
                 "LaserMesh::ensure_per_ray_step_scratch scratch_per_warp_step_max "
                 "cudaFree failed");
    free_checked(scratch_per_warp_step_sum,
                 "LaserMesh::ensure_per_ray_step_scratch scratch_per_warp_step_sum "
                 "cudaFree failed");
    scratch_per_ray_step_capacity = 0;
    scratch_per_warp_step_capacity = 0;
  };
  auto alloc_or_cleanup = [&](auto*& ptr, const std::size_t bytes, const char* message) {
    const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), bytes);
    if (err == cudaSuccess) {
      return;
    }
    release_per_ray_step_scratch_noassert();
    TENRYU_ASSERT(false, message);
  };

  release_per_ray_step_scratch();
  const int per_warp_capacity = (n_rays + 31) / 32;
  const std::size_t per_ray_bytes = static_cast<std::size_t>(n_rays) * sizeof(int);
  const std::size_t per_warp_int_bytes =
      static_cast<std::size_t>(per_warp_capacity) * sizeof(int);
  const std::size_t per_warp_sum_bytes =
      static_cast<std::size_t>(per_warp_capacity) * sizeof(unsigned long long);

  alloc_or_cleanup(scratch_per_ray_step_count, per_ray_bytes,
                   "LaserMesh::ensure_per_ray_step_scratch scratch_per_ray_step_count "
                   "cudaMalloc failed");
  alloc_or_cleanup(scratch_sorted_step_count, per_ray_bytes,
                   "LaserMesh::ensure_per_ray_step_scratch scratch_sorted_step_count "
                   "cudaMalloc failed");
  alloc_or_cleanup(scratch_per_warp_step_max, per_warp_int_bytes,
                   "LaserMesh::ensure_per_ray_step_scratch scratch_per_warp_step_max "
                   "cudaMalloc failed");
  alloc_or_cleanup(scratch_per_warp_step_sum, per_warp_sum_bytes,
                   "LaserMesh::ensure_per_ray_step_scratch scratch_per_warp_step_sum "
                   "cudaMalloc failed");
  scratch_per_ray_step_capacity = n_rays;
  scratch_per_warp_step_capacity = per_warp_capacity;
}

void LaserMesh::ensure_hot_e_capture(const int n_rays, const int n_channels) {
  const int needed = n_rays * n_channels * 8;
  if (needed <= 0 || needed <= hot_e_capture_capacity) {
    return;
  }
  if (hot_e_capture != nullptr) {
    cuda_check(cudaFree(hot_e_capture),
               "LaserMesh::ensure_hot_e_capture hot_e_capture cudaFree failed");
    hot_e_capture = nullptr;
    hot_e_capture_capacity = 0;
  }
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&hot_e_capture),
                        static_cast<std::size_t>(needed) * sizeof(double)),
             "LaserMesh::ensure_hot_e_capture hot_e_capture cudaMalloc failed");
  hot_e_capture_capacity = needed;
}

void LaserMesh::ensure_step_scratch() {
  auto release_scratch_noassert = [&]() {
    if (scratch_step_histogram != nullptr) {
      static_cast<void>(cudaFree(scratch_step_histogram));
      scratch_step_histogram = nullptr;
    }
    if (scratch_step_tally_slab != nullptr) {
      static_cast<void>(cudaFree(scratch_step_tally_slab));
      scratch_step_tally_slab = nullptr;
    }
    scratch_unabsorbed = nullptr;
    scratch_tail_closure_count = nullptr;
    scratch_tail_closure_absorbed_power = nullptr;
    scratch_critical_surface_hit_count = nullptr;
    scratch_ra_power_total = nullptr;
    if (scratch_error_flags != nullptr) {
      static_cast<void>(cudaFree(scratch_error_flags));
      scratch_error_flags = nullptr;
    }
  };
  auto alloc_or_cleanup = [&](auto*& ptr, const std::size_t bytes, const char* message) {
    if (ptr != nullptr) {
      return;
    }
    const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), bytes);
    if (err == cudaSuccess) {
      return;
    }
    release_scratch_noassert();
    TENRYU_ASSERT(false, message);
  };

  alloc_or_cleanup(scratch_step_histogram,
                   static_cast<std::size_t>(kTraceStepHistSize) * sizeof(int),
                   "LaserMesh::ensure_step_scratch scratch_step_histogram cudaMalloc failed");
  if (scratch_step_tally_slab == nullptr) {
    alloc_or_cleanup(scratch_step_tally_slab, 40,
                     "LaserMesh::ensure_step_scratch scratch_step_tally_slab "
                     "cudaMalloc failed");
  }
  unsigned char* base = static_cast<unsigned char*>(scratch_step_tally_slab);
  scratch_unabsorbed = reinterpret_cast<double*>(base + 0);
  scratch_tail_closure_count = reinterpret_cast<unsigned long long*>(base + 8);
  scratch_tail_closure_absorbed_power = reinterpret_cast<double*>(base + 16);
  scratch_critical_surface_hit_count = reinterpret_cast<unsigned long long*>(base + 24);
  scratch_ra_power_total = reinterpret_cast<double*>(base + 32);
  alloc_or_cleanup(scratch_error_flags, sizeof(core::DeviceErrorFlags),
                   "LaserMesh::ensure_step_scratch scratch_error_flags cudaMalloc failed");
}

void LaserMesh::ensure_step_pack(const std::size_t bytes) {
  if (bytes <= scratch_step_pack_capacity) {
    return;
  }
  if (scratch_step_pack_device != nullptr) {
    cuda_check(cudaFree(scratch_step_pack_device),
               "LaserMesh::ensure_step_pack scratch_step_pack_device cudaFree failed");
    scratch_step_pack_device = nullptr;
  }
  if (scratch_step_pack_host != nullptr) {
    cuda_check(cudaFreeHost(scratch_step_pack_host),
               "LaserMesh::ensure_step_pack scratch_step_pack_host cudaFreeHost failed");
    scratch_step_pack_host = nullptr;
  }
  scratch_step_pack_capacity = 0;

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&scratch_step_pack_device), bytes),
             "LaserMesh::ensure_step_pack scratch_step_pack_device cudaMalloc failed");
  const cudaError_t host_err =
      cudaMallocHost(reinterpret_cast<void**>(&scratch_step_pack_host), bytes);
  if (host_err != cudaSuccess) {
    static_cast<void>(cudaFree(scratch_step_pack_device));
    scratch_step_pack_device = nullptr;
    TENRYU_ASSERT(false,
                  "LaserMesh::ensure_step_pack scratch_step_pack_host cudaMallocHost failed");
  }
  scratch_step_pack_capacity = bytes;
}

void LaserMesh::clear_step_scratch(cudaStream_t stream) const {
  TENRYU_ASSERT(scratch_step_histogram != nullptr,
                "LaserMesh::clear_step_scratch scratch_step_histogram is null");
  TENRYU_ASSERT(scratch_step_tally_slab != nullptr,
                "LaserMesh::clear_step_scratch scratch_step_tally_slab is null");
  TENRYU_ASSERT(scratch_unabsorbed != nullptr,
                "LaserMesh::clear_step_scratch scratch_unabsorbed is null");
  TENRYU_ASSERT(scratch_tail_closure_count != nullptr,
                "LaserMesh::clear_step_scratch scratch_tail_closure_count is null");
  TENRYU_ASSERT(scratch_tail_closure_absorbed_power != nullptr,
                "LaserMesh::clear_step_scratch scratch_tail_closure_absorbed_power is null");
  TENRYU_ASSERT(scratch_critical_surface_hit_count != nullptr,
                "LaserMesh::clear_step_scratch scratch_critical_surface_hit_count is null");
  TENRYU_ASSERT(scratch_ra_power_total != nullptr,
                "LaserMesh::clear_step_scratch scratch_ra_power_total is null");
  TENRYU_ASSERT(scratch_error_flags != nullptr,
                "LaserMesh::clear_step_scratch scratch_error_flags is null");

  cuda_check(cudaMemsetAsync(scratch_step_histogram, 0,
                             static_cast<std::size_t>(kTraceStepHistSize) * sizeof(int), stream),
             "LaserMesh::clear_step_scratch memset scratch_step_histogram failed");
  cuda_check(cudaMemsetAsync(scratch_step_tally_slab, 0, 40, stream),
             "LaserMesh::clear_step_scratch memset scratch_step_tally_slab failed");
  cuda_check(cudaMemsetAsync(scratch_error_flags, 0, sizeof(core::DeviceErrorFlags), stream),
             "LaserMesh::clear_step_scratch memset scratch_error_flags failed");
}

LaserMesh create_from_config(const core::Config& cfg) {
  LaserMesh mesh;

  const int nr = cfg.laser.lasermesh.nr;
  const int nz = cfg.laser.lasermesh.nz;
  TENRYU_ASSERT(nr > 0, "Laser.lasermesh.nr must be > 0");
  TENRYU_ASSERT(nz > 0, "Laser.lasermesh.nz must be > 0");
  if (cfg.main.dimension == "1D_SPH") {
    mesh.allocate(nr, 2 * nr);
  } else {
    mesh.allocate(nr, nz);
  }

  const double r_abs = std::max(std::abs(cfg.mesh.r_min), std::abs(cfg.mesh.r_max));
  mesh.target_radius = (r_abs > 0.0) ? r_abs : std::max(cfg.mesh.r_max - cfg.mesh.r_min, 1.0e-12);
  mesh.R_max = cfg.laser.lasermesh.r_max_factor * mesh.target_radius;

  if (cfg.main.dimension == "1D_SPH") {
    mesh.Z_min = -mesh.R_max;
    mesh.Z_max = mesh.R_max;
  } else {
    const double z_center = (cfg.main.dimension == "2D_RZ")
                                ? 0.5 * (cfg.mesh.z_min + cfg.mesh.z_max)
                                : 0.0;
    mesh.Z_min = z_center - cfg.laser.lasermesh.z_span_factor * mesh.target_radius;
    mesh.Z_max = z_center + cfg.laser.lasermesh.z_span_factor * mesh.target_radius;
  }
  mesh.n_hat_margin = cfg.laser.lasermesh.critical_margin;
  mesh.ghost_corona_enabled = cfg.laser.lasermesh.ghost_corona.enabled;
  mesh.ghost_n_out = cfg.laser.lasermesh.ghost_corona.n_out;
  mesh.ghost_ne_min_frac = cfg.laser.lasermesh.ghost_corona.ne_min_frac;
  mesh.ghost_ne_max_frac = cfg.laser.lasermesh.ghost_corona.ne_max_frac;
  mesh.ghost_Te_min_eV = cfg.laser.lasermesh.ghost_corona.Te_min_eV;
  mesh.ghost_zbar_min = cfg.laser.lasermesh.ghost_corona.zbar_min;
  mesh.ghost_zbar_max = cfg.laser.lasermesh.ghost_corona.zbar_max;
  mesh.ghost_handoff_cells = cfg.laser.lasermesh.ghost_corona.handoff_cells;
  mesh.ghost_handoff_decay = cfg.laser.lasermesh.ghost_corona.handoff_decay;
  mesh.ghost_transition_enabled = cfg.laser.lasermesh.ghost_corona.transition_enabled;
  mesh.ghost_transition_resolved_nhat =
      cfg.laser.lasermesh.ghost_corona.transition_resolved_nhat;
  mesh.ghost_transition_resolved_cells =
      cfg.laser.lasermesh.ghost_corona.transition_resolved_cells;
  mesh.ghost_transition_density_exponent =
      cfg.laser.lasermesh.ghost_corona.transition_density_exponent;

  const double lambda_cm = cfg.laser.wavelength_nm * 1.0e-7;
  mesh.n_crit = compute_critical_density_from_wavelength_cm(lambda_cm);

  if (!cfg.materials.materials.empty()) {
    const int first_nonvoid = cfg.materials.first_nonvoid_material_index();
    TENRYU_ASSERT(first_nonvoid >= 0,
                  "create_from_config requires at least one non-void material");
    mesh.material_A = std::max(
        cfg.materials.materials[static_cast<std::size_t>(first_nonvoid)].A, 1.0e-12);
    mesh.material_A_list.reserve(cfg.materials.materials.size());
    for (const auto& mat : cfg.materials.materials) {
      mesh.material_A_list.push_back(std::max(mat.A, 1.0e-12));
    }
  }

  const auto node_R_host = build_uniform_nodes(0.0, mesh.R_max, mesh.nr);
  const auto node_Z_host = build_uniform_nodes(mesh.Z_min, mesh.Z_max, mesh.nz);
  cuda_check(cudaMemcpy(mesh.node_R, node_R_host.data(), node_R_host.size() * sizeof(double),
                        cudaMemcpyHostToDevice),
             "create_from_config memcpy node_R failed");
  cuda_check(cudaMemcpy(mesh.node_Z, node_Z_host.data(), node_Z_host.size() * sizeof(double),
                        cudaMemcpyHostToDevice),
             "create_from_config memcpy node_Z failed");

  mesh.dx_min = compute_min_cell_spacing(mesh);
  return mesh;
}

void build_hydro_mirror_1d(const LaserMesh& mesh,
                           const core::State& state,
                           HydroMirror1D& mirror) {
  TENRYU_ASSERT(state.mesh.dim == 1, "build_hydro_mirror_1d expects 1D_SPH state");

  mirror.rho.assign(state.rho.size(), 0.0);
  mirror.Te.assign(state.Te.size(), 0.0);
  mirror.zbar.assign(state.zbar.size(), 0.0);
  mirror.r_edges.assign(state.x_r.size(), 0.0);
  {
    const int n = static_cast<int>(state.rho.size());
    std::vector<double> mir_pack(3 * static_cast<std::size_t>(n));
    const double* srcs[3] = {state.rho.data(), state.zbar.data(), state.Te.data()};
    core::pack_pull_fields(srcs, 3, n, mir_pack.data(),
                           "laser:build_hydro_mirror_1d:pull");
    std::memcpy(mirror.rho.data(), mir_pack.data(), n * sizeof(double));
    std::memcpy(mirror.zbar.data(), mir_pack.data() + n, n * sizeof(double));
    std::memcpy(mirror.Te.data(), mir_pack.data() + 2 * n, n * sizeof(double));
  }
  state.x_r.copy_to_host(mirror.r_edges.data());
  mirror.cell_is_void = state.cell_is_void;
  TENRYU_ASSERT(mirror.cell_is_void.size() == mirror.rho.size(),
                "build_hydro_mirror_1d cell_is_void size mismatch");
  mirror.A_eff = compute_cell_effective_A(mesh, state, mirror.rho.size());
}

void map_from_hydro_1d(LaserMesh& mesh,
                       const core::State& state,
                       const core::Config::LaserConfig& laser_cfg,
                       cudaStream_t stream) {
  HydroMirror1D hydro;
  build_hydro_mirror_1d(mesh, state, hydro);
  map_from_hydro_1d(mesh, state, laser_cfg, hydro, stream);
}

void map_from_hydro_1d(LaserMesh& mesh,
                       const core::State& state,
                       const core::Config::LaserConfig& laser_cfg,
                       const HydroMirror1D& hydro,
                       cudaStream_t stream) {
  TENRYU_ASSERT(mesh.is_allocated(), "map_from_hydro_1d requires allocated LaserMesh");
  TENRYU_ASSERT(state.mesh.dim == 1, "map_from_hydro_1d expects 1D_SPH state");

  const std::vector<double>& rho = hydro.rho;
  const std::vector<double>& Te_host = hydro.Te;
  const std::vector<double>& zbar = hydro.zbar;
  const std::vector<double>& r_edges = hydro.r_edges;
  const std::vector<double>& A_eff_cell = hydro.A_eff;
  const std::vector<std::uint8_t>& cell_is_void =
      hydro.cell_is_void.empty() ? state.cell_is_void : hydro.cell_is_void;
  TENRYU_ASSERT(cell_is_void.size() == rho.size(),
                "map_from_hydro_1d cell_is_void size mismatch");
  TENRYU_ASSERT(Te_host.empty() || Te_host.size() == rho.size(),
                "map_from_hydro_1d Te host size mismatch");
  TENRYU_ASSERT(zbar.size() == rho.size(), "map_from_hydro_1d zbar size mismatch");
  TENRYU_ASSERT(A_eff_cell.size() == rho.size(), "map_from_hydro_1d A_eff size mismatch");
  TENRYU_ASSERT(r_edges.size() == rho.size() + 1, "map_from_hydro_1d r_edges size mismatch");
  TENRYU_ASSERT(state.rho.size() == rho.size(), "map_from_hydro_1d device rho size mismatch");
  TENRYU_ASSERT(state.Te.size() == rho.size(), "map_from_hydro_1d device Te size mismatch");
  TENRYU_ASSERT(state.zbar.size() == rho.size(),
                "map_from_hydro_1d device zbar size mismatch");
  TENRYU_ASSERT(state.x_r.size() == r_edges.size(),
                "map_from_hydro_1d device r_edges size mismatch");

  for (std::size_t c = 0; c < rho.size(); ++c) {
    TENRYU_ASSERT(std::isfinite(rho[c]), "map_from_hydro_1d encountered non-finite rho");
    TENRYU_ASSERT(std::isfinite(zbar[c]), "map_from_hydro_1d encountered non-finite zbar");
    if (!Te_host.empty()) {
      TENRYU_ASSERT(std::isfinite(Te_host[c]),
                    "map_from_hydro_1d encountered non-finite Te");
    }
  }

  const DynamicMeshParams1D params = compute_dynamic_mesh_params_1d(
      rho, zbar, r_edges, A_eff_cell, cell_is_void, mesh.n_crit,
      laser_cfg.lasermesh.mesh_factor, laser_cfg.lasermesh.rmax_n_hat_threshold,
      laser_cfg.lasermesh.r_max_factor, mesh.target_radius, laser_cfg.lasermesh.nr_max);
  mesh.ensure_capacity(params.nr, params.nz);
  static int prev_nr = 0;
  static int prev_nz = 0;
  const bool first_call = (prev_nr == 0 && prev_nz == 0);
  const bool significant_change =
      (prev_nr > 0) && (std::abs(params.nr - prev_nr) > std::max(1, prev_nr / 20));
  if (first_call || significant_change) {
    const double active_mem_mb = 10.0 * static_cast<double>(params.nr + 1) *
                                 static_cast<double>(params.nz + 1) * 8.0 /
                                 (1024.0 * 1024.0);
    std::ostringstream oss;
    oss << "LaserMesh1D: nr=" << params.nr << ", nz=" << params.nz << ", R_crit="
        << std::scientific << std::setprecision(4) << params.R_crit << ", dR_fine="
        << params.dR_fine << ", nr_max=" << std::defaultfloat << laser_cfg.lasermesh.nr_max
        << ", active_mem=" << std::fixed << std::setprecision(1) << active_mem_mb << "MB";
    core::log_info(oss.str());
    prev_nr = params.nr;
    prev_nz = params.nz;
  }
  mesh.R_max = params.R_max;
  mesh.Z_min = -params.R_max;
  mesh.Z_max = params.R_max;

  const std::vector<double>& node_R = params.node_R;
  const std::vector<double> node_Z = build_mirrored_Z_from_R(node_R);
  const int n_nodes_total = mesh.n_nodes();
  const int n_cells = static_cast<int>(rho.size());
  const double n_crit_safe = std::max(mesh.n_crit, 1.0e-30);

  std::vector<double> ne_raw_cell(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t c_idx = static_cast<std::size_t>(c);
    if (cell_is_void[c_idx] != 0U) {
      continue;
    }
    const double A_eff = std::max(A_eff_cell[c_idx], 1.0e-30);
    ne_raw_cell[c_idx] = std::max(0.0, rho[c_idx]) * std::max(0.0, zbar[c_idx]) /
                         (A_eff * kProtonMass * n_crit_safe);
  }

  const CriticalSurfaceEstimate1D crit_est =
      estimate_critical_surface_1d(ne_raw_cell, r_edges);
  const int fcrit = crit_est.fcrit;
  const double r_crit_interp = crit_est.r_interp;
  const int fcrit_cell = fcrit;
  const double ne_fcrit_center =
      (fcrit_cell >= 0 && fcrit_cell < n_cells) ? ne_raw_cell[static_cast<std::size_t>(fcrit_cell)]
                                                 : 0.0;
  const double r_fcrit_center = (fcrit_cell >= 0 && fcrit_cell < n_cells)
                                    ? 0.5 * (r_edges[static_cast<std::size_t>(fcrit_cell)] +
                                             r_edges[static_cast<std::size_t>(fcrit_cell + 1)])
                                    : 0.0;
  int outer_surface_cell = -1;
  for (int c = n_cells - 1; c >= 0; --c) {
    if (cell_is_void[static_cast<std::size_t>(c)] != 0U) {
      continue;
    }
    if (c + 1 >= n_cells || cell_is_void[static_cast<std::size_t>(c + 1)] != 0U) {
      outer_surface_cell = c;
      break;
    }
  }
  const bool use_ghost_corona =
      mesh.ghost_corona_enabled && outer_surface_cell >= 0 && mesh.ghost_n_out > 0;
  const double r_surface_outer =
      (outer_surface_cell >= 0) ? r_edges[static_cast<std::size_t>(outer_surface_cell + 1)] : 0.0;
  const double base_ghost_width =
      use_ghost_corona
          ? std::max(static_cast<double>(mesh.ghost_n_out) * params.dR_fine, 1.0e-30)
          : 0.0;
  constexpr int kGhostAnchorSpan = 3;
  double Te_anchor_raw = mesh.ghost_Te_min_eV;
  if (!Te_host.empty()) {
    Te_anchor_raw = average_outer_cells(Te_host, cell_is_void, outer_surface_cell,
                                        kGhostAnchorSpan, mesh.ghost_Te_min_eV);
  } else if (use_ghost_corona && outer_surface_cell >= 0) {
    double Te_samples[kGhostAnchorSpan] = {};
    double Te_weights[kGhostAnchorSpan] = {};
    int n_Te_samples = 0;
    for (int k = 0; k < kGhostAnchorSpan; ++k) {
      const int c = outer_surface_cell - k;
      if (c < 0 || c >= n_cells) {
        break;
      }
      if (cell_is_void[static_cast<std::size_t>(c)] != 0U) {
        continue;
      }
      cuda_check(cudaMemcpyAsync(&Te_samples[n_Te_samples], state.Te.data() + c,
                                 sizeof(double), cudaMemcpyDeviceToHost, stream),
                 "map_from_hydro_1d memcpyAsync Te anchor failed");
      Te_weights[n_Te_samples] = 1.0 / static_cast<double>(k + 1);
      ++n_Te_samples;
    }
    if (n_Te_samples > 0) {
      cuda_check(cudaStreamSynchronize(stream),
                 "map_from_hydro_1d Te anchor stream synchronize failed");
      double sum = 0.0;
      double w_sum = 0.0;
      for (int s = 0; s < n_Te_samples; ++s) {
        sum += Te_weights[s] * Te_samples[s];
        w_sum += Te_weights[s];
      }
      if (w_sum > 0.0) {
        Te_anchor_raw = sum / w_sum;
      }
    }
  }
  const double Te_anchor = std::max(Te_anchor_raw, mesh.ghost_Te_min_eV);
  const double zbar_anchor =
      std::clamp(std::max(average_outer_cells(zbar, cell_is_void, outer_surface_cell,
                                              kGhostAnchorSpan, mesh.ghost_zbar_max),
                          mesh.ghost_zbar_min),
                 mesh.ghost_zbar_min,
                 std::max(mesh.ghost_zbar_max, mesh.ghost_zbar_min));
  const double A_anchor =
      std::max(average_outer_cells(A_eff_cell, cell_is_void, outer_surface_cell,
                                   kGhostAnchorSpan, mesh.material_A),
               1.0e-30);
  const double ghost_cs = compute_ghost_sound_speed_cm_s(Te_anchor, zbar_anchor, A_anchor);
  const double transient_ghost_width =
      std::max(base_ghost_width, ghost_cs * std::max(state.t, 0.0));
  const double ghost_width = use_ghost_corona ? transient_ghost_width : 0.0;
  const double r_ghost_outer = r_surface_outer + ghost_width;
  const double ghost_ne_min =
      std::max(mesh.ghost_ne_min_frac, 1.0e-12);
  const double ghost_ne_max =
      std::max(mesh.ghost_ne_max_frac, ghost_ne_min * 1.0001);
  const double ghost_ne_inner =
      (outer_surface_cell >= 0 &&
       ne_raw_cell[static_cast<std::size_t>(outer_surface_cell)] < 1.0)
          ? std::clamp(std::max(ne_raw_cell[static_cast<std::size_t>(outer_surface_cell)],
                                ghost_ne_min * 1.0001),
                       ghost_ne_min * 1.0001,
                       ghost_ne_max)
          : ghost_ne_max;
  const double ghost_log_span =
      std::log(std::max(ghost_ne_inner, ghost_ne_min * 1.0001) / ghost_ne_min);
  const double ghost_scale_length =
      (ghost_log_span > 0.0) ? std::max(ghost_width / ghost_log_span, 1.0e-30) : ghost_width;

  if (mesh.prev_n_hat_valid && !mesh.prev_n_hat_host.empty() &&
      static_cast<int>(mesh.prev_n_hat_host.size()) != n_nodes_total) {
    mesh.prev_n_hat_valid = false;
    mesh.prev_n_hat_host.clear();
  }
  if (mesh.prev_n_hat_valid && !mesh.prev_n_hat_host.empty()) {
    cuda_check(cudaMemcpyAsync(mesh.prev_n_hat_device, mesh.prev_n_hat_host.data(),
                               static_cast<std::size_t>(n_nodes_total) * sizeof(double),
                               cudaMemcpyHostToDevice, stream),
               "map_from_hydro_1d memcpyAsync prev_n_hat_host failed");
    cuda_check(cudaStreamSynchronize(stream),
               "map_from_hydro_1d prev_n_hat_host stream synchronize failed");
    mesh.prev_n_hat_host.clear();
  }
  const bool apply_ema = mesh.prev_n_hat_valid;

  cuda_check(cudaMemcpyAsync(mesh.node_R, node_R.data(), node_R.size() * sizeof(double),
                             cudaMemcpyHostToDevice, stream),
             "map_from_hydro_1d memcpyAsync node_R failed");
  cuda_check(cudaMemcpyAsync(mesh.node_Z, node_Z.data(), node_Z.size() * sizeof(double),
                             cudaMemcpyHostToDevice, stream),
             "map_from_hydro_1d memcpyAsync node_Z failed");

  ensure_hydro_mapping_capacity(mesh, n_cells);
  cuda_check(cudaMemcpyAsync(mesh.hydro_A_eff_device, A_eff_cell.data(),
                             static_cast<std::size_t>(n_cells) * sizeof(double),
                             cudaMemcpyHostToDevice, stream),
             "map_from_hydro_1d memcpyAsync A_eff failed");
  cuda_check(cudaMemcpyAsync(mesh.hydro_cell_is_void_device, cell_is_void.data(),
                             static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t),
                             cudaMemcpyHostToDevice, stream),
             "map_from_hydro_1d memcpyAsync cell_is_void failed");

  const int block = 256;
  const int grid = (n_nodes_total + block - 1) / block;
  map_hydro_to_laser_1d_kernel<<<grid, block, 0, stream>>>(
      mesh.node_R, mesh.node_Z, state.rho.data(), state.Te.data(), state.zbar.data(),
      mesh.hydro_A_eff_device, mesh.hydro_cell_is_void_device, state.x_r.data(),
      mesh.n_e_hat, mesh.n_e_hat_raw, mesh.T_e, mesh.Zbar, n_nodes_total,
      mesh.n_nodes_z, n_cells, n_crit_safe, use_ghost_corona ? 1 : 0,
      outer_surface_cell, ghost_ne_inner, ghost_scale_length, ghost_ne_min, r_surface_outer,
      r_ghost_outer, Te_anchor, zbar_anchor, mesh.ghost_zbar_min, mesh.ghost_zbar_max,
      mesh.ghost_Te_min_eV, laser_cfg.lasermesh.critical_clip ? 1 : 0,
      mesh.n_hat_margin, fcrit_cell, r_crit_interp, ne_fcrit_center, r_fcrit_center);
  cuda_check(cudaGetLastError(), "map_from_hydro_1d map kernel launch failed");

  if (apply_ema) {
    ema_smooth_n_hat_kernel<<<grid, block, 0, stream>>>(
        mesh.n_e_hat, mesh.prev_n_hat_device, n_nodes_total);
    cuda_check(cudaGetLastError(), "map_from_hydro_1d EMA kernel launch failed");
  }
  cuda_check(cudaMemcpyAsync(mesh.prev_n_hat_device, mesh.n_e_hat,
                             static_cast<std::size_t>(n_nodes_total) * sizeof(double),
                             cudaMemcpyDeviceToDevice, stream),
             "map_from_hydro_1d memcpyAsync prev_n_hat_device failed");
  mesh.prev_n_hat_valid = true;

  refresh_radial_profile_1d(mesh, false, stream);
  mesh.dx_min = params.dR_fine;
}

static void map_from_hydro_2d_impl(LaserMesh& mesh,
                                   const core::State& state,
                                   const core::Config::LaserConfig& laser_cfg,
                                   cudaStream_t stream,
                                   CbetLmFields* cbet_out,
                                   const parallel::PartitionInfo* part,
                                   const parallel::Reduction* reduction) {
  TENRYU_ASSERT(mesh.is_allocated(), "map_from_hydro_2d requires allocated LaserMesh");
  TENRYU_ASSERT(state.mesh.dim == 2, "map_from_hydro_2d expects 2D_RZ state");

  const int nr_h = state.mesh.topo.nr;
  const int nz_h = state.mesh.topo.nz;
  TENRYU_ASSERT(nr_h >= 2 && nz_h >= 2, "map_from_hydro_2d requires at least 2x2 hydro cells");

  std::vector<double> rho(state.rho.size(), 0.0);
  std::vector<double> Te(state.Te.size(), 0.0);
  std::vector<double> zbar(state.zbar.size(), 0.0);
  const std::vector<double> A_eff_cell = compute_cell_effective_A(mesh, state, rho.size());
  state.rho.copy_to_host(rho.data());
  state.Te.copy_to_host(Te.data());
  state.zbar.copy_to_host(zbar.data());
  TENRYU_ASSERT(state.cell_is_void.size() == rho.size(),
                "map_from_hydro_2d cell_is_void size mismatch");

  std::vector<double> node_R(static_cast<std::size_t>(mesh.n_nodes_r), 0.0);
  std::vector<double> node_Z(static_cast<std::size_t>(mesh.n_nodes_z), 0.0);
  cuda_check(cudaMemcpy(node_R.data(), mesh.node_R, node_R.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "map_from_hydro_2d memcpy node_R D2H failed");
  cuda_check(cudaMemcpy(node_Z.data(), mesh.node_Z, node_Z.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "map_from_hydro_2d memcpy node_Z D2H failed");

  const std::vector<double>& hydro_r = state.mesh.cell_centroid_r;
  const std::vector<double>& hydro_z = state.mesh.cell_centroid_z;
  TENRYU_ASSERT(hydro_r.size() == static_cast<std::size_t>(nr_h * nz_h),
                "map_from_hydro_2d hydro_r size mismatch");
  TENRYU_ASSERT(hydro_z.size() == static_cast<std::size_t>(nr_h * nz_h),
                "map_from_hydro_2d hydro_z size mismatch");

  std::vector<double> r_centers(static_cast<std::size_t>(nr_h), 0.0);
  std::vector<double> z_centers(static_cast<std::size_t>(nz_h), 0.0);
  for (int i = 0; i < nr_h; ++i) {
    long double r_sum = 0.0L;
    for (int j = 0; j < nz_h; ++j) {
      const int c = i * nz_h + j;
      r_sum += static_cast<long double>(hydro_r[static_cast<std::size_t>(c)]);
    }
    r_centers[static_cast<std::size_t>(i)] = static_cast<double>(r_sum / std::max(1, nz_h));
  }
  for (int j = 0; j < nz_h; ++j) {
    long double z_sum = 0.0L;
    for (int i = 0; i < nr_h; ++i) {
      const int c = i * nz_h + j;
      z_sum += static_cast<long double>(hydro_z[static_cast<std::size_t>(c)]);
    }
    z_centers[static_cast<std::size_t>(j)] = static_cast<double>(z_sum / std::max(1, nr_h));
  }

  const double r_scale = std::max(1.0, std::abs(r_centers.back() - r_centers.front()));
  const double z_scale = std::max(1.0, std::abs(z_centers.back() - z_centers.front()));
  const double tol_r = 1.0e-10 * r_scale;
  const double tol_z = 1.0e-10 * z_scale;
  bool tensor_like_centroids = true;
  for (int i = 0; i < nr_h && tensor_like_centroids; ++i) {
    for (int j = 0; j < nz_h; ++j) {
      const int c = i * nz_h + j;
      if (std::abs(hydro_r[static_cast<std::size_t>(c)] - r_centers[static_cast<std::size_t>(i)]) >
              tol_r ||
          std::abs(hydro_z[static_cast<std::size_t>(c)] - z_centers[static_cast<std::size_t>(j)]) >
              tol_z) {
        tensor_like_centroids = false;
        break;
      }
    }
  }

  const double r_min_host =
      r_centers.front() - 0.5 * (r_centers[1] - r_centers.front());
  const double r_max_host = r_centers.back() +
                            0.5 * (r_centers.back() -
                                   r_centers[static_cast<std::size_t>(nr_h - 2)]);
  const double z_min_host =
      z_centers.front() - 0.5 * (z_centers[1] - z_centers.front());
  const double z_max_host = z_centers.back() +
                            0.5 * (z_centers.back() -
                                   z_centers[static_cast<std::size_t>(nz_h - 2)]);
  const double host_extent_scale =
      std::max({1.0, std::abs(r_min_host), std::abs(r_max_host), std::abs(z_min_host),
                std::abs(z_max_host)});
  const double host_extent_tol = 1.0e-12 * host_extent_scale;

  auto outside_host_extent = [&](const double R, const double Z) {
    return R < r_min_host - host_extent_tol || R > r_max_host + host_extent_tol ||
           Z < z_min_host - host_extent_tol || Z > z_max_host + host_extent_tol;
  };

  const int n_nodes_total = mesh.n_nodes();
  std::vector<double> n_hat_lm(static_cast<std::size_t>(n_nodes_total), 0.0);
  std::vector<double> n_hat_raw_lm(static_cast<std::size_t>(n_nodes_total), 0.0);
  std::vector<double> Te_lm(static_cast<std::size_t>(n_nodes_total), 0.0);
  std::vector<double> Z_lm(static_cast<std::size_t>(n_nodes_total), 0.0);

  auto interp_hydro = [&](const std::vector<double>& f, const double R, const double Z) {
    const int i = locate_interval_centers(r_centers, R);
    const int j = locate_interval_centers(z_centers, Z);
    const int c00 = i * nz_h + j;
    const int c10 = (i + 1) * nz_h + j;
    const int c01 = i * nz_h + (j + 1);
    const int c11 = (i + 1) * nz_h + (j + 1);
    if (tensor_like_centroids) {
      const double r0 = r_centers[static_cast<std::size_t>(i)];
      const double r1 = r_centers[static_cast<std::size_t>(i + 1)];
      const double z0 = z_centers[static_cast<std::size_t>(j)];
      const double z1 = z_centers[static_cast<std::size_t>(j + 1)];
      const double xi = (r1 > r0) ? std::clamp((R - r0) / (r1 - r0), 0.0, 1.0) : 0.0;
      const double eta = (z1 > z0) ? std::clamp((Z - z0) / (z1 - z0), 0.0, 1.0) : 0.0;
      const double w00 = (1.0 - xi) * (1.0 - eta);
      const double w10 = xi * (1.0 - eta);
      const double w01 = (1.0 - xi) * eta;
      const double w11 = xi * eta;
      return w00 * f[static_cast<std::size_t>(c00)] + w10 * f[static_cast<std::size_t>(c10)] +
             w01 * f[static_cast<std::size_t>(c01)] + w11 * f[static_cast<std::size_t>(c11)];
    }

    // ALE歪みなどで重心が非テンソル積になった場合は、局所4セルの重心座標に基づく
    // 逆距離重みで補間して縮約誤差を抑える。
    auto inv_dist_weight = [&](const int c) {
      const double dr = R - hydro_r[static_cast<std::size_t>(c)];
      const double dz = Z - hydro_z[static_cast<std::size_t>(c)];
      const double d2 = dr * dr + dz * dz;
      if (d2 <= 1.0e-30) {
        return std::numeric_limits<double>::infinity();
      }
      return 1.0 / d2;
    };
    const double w00 = inv_dist_weight(c00);
    if (std::isinf(w00)) {
      return f[static_cast<std::size_t>(c00)];
    }
    const double w10 = inv_dist_weight(c10);
    if (std::isinf(w10)) {
      return f[static_cast<std::size_t>(c10)];
    }
    const double w01 = inv_dist_weight(c01);
    if (std::isinf(w01)) {
      return f[static_cast<std::size_t>(c01)];
    }
    const double w11 = inv_dist_weight(c11);
    if (std::isinf(w11)) {
      return f[static_cast<std::size_t>(c11)];
    }
    const double wsum = w00 + w10 + w01 + w11;
    if (!(wsum > 0.0) || !std::isfinite(wsum)) {
      return f[static_cast<std::size_t>(c00)];
    }
    const double num = w00 * f[static_cast<std::size_t>(c00)] +
                       w10 * f[static_cast<std::size_t>(c10)] +
                       w01 * f[static_cast<std::size_t>(c01)] +
                       w11 * f[static_cast<std::size_t>(c11)];
    return num / wsum;
  };

  auto any_void_source_cell = [&](const double R, const double Z) {
    const int i = locate_interval_centers(r_centers, R);
    const int j = locate_interval_centers(z_centers, Z);
    const int c00 = i * nz_h + j;
    const int c10 = (i + 1) * nz_h + j;
    const int c01 = i * nz_h + (j + 1);
    const int c11 = (i + 1) * nz_h + (j + 1);
    return state.cell_is_void[static_cast<std::size_t>(c00)] != 0U ||
           state.cell_is_void[static_cast<std::size_t>(c10)] != 0U ||
           state.cell_is_void[static_cast<std::size_t>(c01)] != 0U ||
           state.cell_is_void[static_cast<std::size_t>(c11)] != 0U;
  };

  // MPI partition-of-unity (Option C, NUMERICS §12.4.2): each LM node is
  // computed by the single rank owning its located base hydro cell (the
  // 4-cell interpolation stencil reaches at most one ghost ring, which
  // is halo-fresh); the other ranks leave the pre-zeroed entries and the
  // Allreduce(SUM) below assembles the identical owner-true LaserMesh on
  // every rank. CBET fields are serial-only (validate-FATAL under MPI).
  const bool lm_pou_active = part != nullptr && part->n_ranks > 1;
  TENRYU_ASSERT(!lm_pou_active || reduction != nullptr,
                "map_from_hydro_2d requires a reducer under MPI (a silent "
                "serial projection would trace a stale medium)");
  TENRYU_ASSERT(!lm_pou_active || cbet_out == nullptr,
                "map_from_hydro_2d: CBET LM fields are serial-only under "
                "the v1 MPI limitation");
  const auto lm_node_owned = [&](const double R, const double Z) {
    if (!lm_pou_active) {
      return true;
    }
    const int oi = locate_interval_centers(r_centers, R);
    const int oj = locate_interval_centers(z_centers, Z);
    return part->owns_cell(oi, oj);
  };
  const double n_crit = std::max(mesh.n_crit, 1.0e-30);
  for (int i = 0; i < mesh.n_nodes_r; ++i) {
    for (int j = 0; j < mesh.n_nodes_z; ++j) {
      const double R = node_R[static_cast<std::size_t>(i)];
      const double Z = node_Z[static_cast<std::size_t>(j)];
      if (!lm_node_owned(R, Z)) {
        continue;
      }

      const double Te_n = std::max(0.0, interp_hydro(Te, R, Z));
      double Z_n = 0.0;
      double nh_raw = 0.0;
      if (!outside_host_extent(R, Z) && !any_void_source_cell(R, Z)) {
        const double rho_n = std::max(0.0, interp_hydro(rho, R, Z));
        Z_n = std::max(0.0, interp_hydro(zbar, R, Z));
        const double A_eff_n = std::max(1.0e-30, interp_hydro(A_eff_cell, R, Z));
        const double n_e = rho_n * Z_n / (A_eff_n * kProtonMass);
        nh_raw = std::max(0.0, n_e / n_crit);
      }

      double n_hat = std::clamp(nh_raw, 0.0, 1.0);
      if (laser_cfg.lasermesh.critical_clip) {
        n_hat = std::min(n_hat, mesh.n_hat_margin);
      }

      const int n = mesh.node_index(i, j);
      n_hat_lm[static_cast<std::size_t>(n)] = n_hat;
      n_hat_raw_lm[static_cast<std::size_t>(n)] = nh_raw;
      Te_lm[static_cast<std::size_t>(n)] = Te_n;
      Z_lm[static_cast<std::size_t>(n)] = Z_n;
    }
  }
  if (lm_pou_active) {
    // Assemble the partition-of-unity: non-owned entries are exactly
    // zero, so the sum reproduces the owner's value bit-for-bit on every
    // rank.
    reduction->allreduce_sum(n_hat_lm.data(),
                             static_cast<int>(n_hat_lm.size()));
    reduction->allreduce_sum(n_hat_raw_lm.data(),
                             static_cast<int>(n_hat_raw_lm.size()));
    reduction->allreduce_sum(Te_lm.data(), static_cast<int>(Te_lm.size()));
    reduction->allreduce_sum(Z_lm.data(), static_cast<int>(Z_lm.size()));
  }
  if (const char* lmdbg = std::getenv("TENRYU_LM_DEBUG_DUMP")) {
    // Diagnostic-only (§16.6 gate): raw binary dump of the assembled LM
    // fields, one file per rank, for cross-rank machine-exactness checks.
    const int dump_rank = (part != nullptr) ? part->rank : 0;
    const auto dump_vec = [&](const char* tag,
                              const std::vector<double>& v) {
      const std::string path = std::string(lmdbg) + "_" + tag + "_r" +
                               std::to_string(dump_rank) + ".bin";
      FILE* fp = std::fopen(path.c_str(), "wb");
      if (fp != nullptr) {
        std::fwrite(v.data(), sizeof(double), v.size(), fp);
        std::fclose(fp);
      }
    };
    dump_vec("nhat", n_hat_lm);
    dump_vec("nhatraw", n_hat_raw_lm);
    dump_vec("te", Te_lm);
    dump_vec("zbar", Z_lm);
  }

  if (cbet_out != nullptr) {
    cbet_out->node_R = node_R;
    cbet_out->node_Z = node_Z;
    cbet_out->n_hat_raw = n_hat_raw_lm;
    cbet_out->Te = Te_lm;
    cbet_out->Zbar = Z_lm;

    std::vector<double> Ti(state.Ti.size(), 0.0);
    state.Ti.copy_to_host(Ti.data());
    TENRYU_ASSERT(Ti.size() == rho.size(), "map_from_hydro_2d Ti size mismatch");

    std::vector<double> v_r_node;
    std::vector<double> v_z_node;
    state.v_r.copy_to_host(v_r_node);
    state.v_z.copy_to_host(v_z_node);
    const std::size_t n_hydro_nodes =
        static_cast<std::size_t>(nr_h + 1) * static_cast<std::size_t>(nz_h + 1);
    TENRYU_ASSERT(v_r_node.size() >= n_hydro_nodes && v_z_node.size() >= n_hydro_nodes,
                  "map_from_hydro_2d node velocity size mismatch");

    std::vector<double> u_R_cell(rho.size(), 0.0);
    std::vector<double> u_Z_cell(rho.size(), 0.0);
    const int hydro_node_stride = nz_h + 1;
    for (int i = 0; i < nr_h; ++i) {
      for (int j = 0; j < nz_h; ++j) {
        const int c = i * nz_h + j;
        const int n00 = i * hydro_node_stride + j;
        const int n10 = (i + 1) * hydro_node_stride + j;
        const int n11 = (i + 1) * hydro_node_stride + (j + 1);
        const int n01 = i * hydro_node_stride + (j + 1);
        u_R_cell[static_cast<std::size_t>(c)] =
            0.25 * (v_r_node[static_cast<std::size_t>(n00)] +
                    v_r_node[static_cast<std::size_t>(n10)] +
                    v_r_node[static_cast<std::size_t>(n11)] +
                    v_r_node[static_cast<std::size_t>(n01)]);
        u_Z_cell[static_cast<std::size_t>(c)] =
            0.25 * (v_z_node[static_cast<std::size_t>(n00)] +
                    v_z_node[static_cast<std::size_t>(n10)] +
                    v_z_node[static_cast<std::size_t>(n11)] +
                    v_z_node[static_cast<std::size_t>(n01)]);
      }
    }

    cbet_out->Ti.assign(static_cast<std::size_t>(n_nodes_total), 0.0);
    cbet_out->u_R.assign(static_cast<std::size_t>(n_nodes_total), 0.0);
    cbet_out->u_Z.assign(static_cast<std::size_t>(n_nodes_total), 0.0);
    cbet_out->A_eff.assign(static_cast<std::size_t>(n_nodes_total), 0.0);
    cbet_out->covered.assign(static_cast<std::size_t>(n_nodes_total), 0U);
    for (int i = 0; i < mesh.n_nodes_r; ++i) {
      for (int j = 0; j < mesh.n_nodes_z; ++j) {
        const double R = node_R[static_cast<std::size_t>(i)];
        const double Z = node_Z[static_cast<std::size_t>(j)];
        const int n = mesh.node_index(i, j);
        const std::size_t sn = static_cast<std::size_t>(n);
        cbet_out->Ti[sn] = std::max(0.0, interp_hydro(Ti, R, Z));
        cbet_out->u_R[sn] = interp_hydro(u_R_cell, R, Z);
        cbet_out->u_Z[sn] = interp_hydro(u_Z_cell, R, Z);
        cbet_out->A_eff[sn] = interp_hydro(A_eff_cell, R, Z);
        cbet_out->covered[sn] =
            (!outside_host_extent(R, Z) && !any_void_source_cell(R, Z)) ? 1U : 0U;
      }
    }
  }

  cuda_check(cudaMemcpyAsync(mesh.n_e_hat, n_hat_lm.data(), n_hat_lm.size() * sizeof(double),
                             cudaMemcpyHostToDevice, stream),
             "map_from_hydro_2d memcpyAsync n_e_hat failed");
  cuda_check(cudaMemcpyAsync(mesh.n_e_hat_raw, n_hat_raw_lm.data(),
                             n_hat_raw_lm.size() * sizeof(double), cudaMemcpyHostToDevice,
                             stream),
             "map_from_hydro_2d memcpyAsync n_e_hat_raw failed");
  cuda_check(cudaMemcpyAsync(mesh.T_e, Te_lm.data(), Te_lm.size() * sizeof(double),
                             cudaMemcpyHostToDevice, stream),
             "map_from_hydro_2d memcpyAsync T_e failed");
  cuda_check(cudaMemcpyAsync(mesh.Zbar, Z_lm.data(), Z_lm.size() * sizeof(double),
                             cudaMemcpyHostToDevice, stream),
             "map_from_hydro_2d memcpyAsync Zbar failed");

  mesh.dx_min = compute_min_cell_spacing(mesh);
}

void map_from_hydro_2d(LaserMesh& mesh,
                       const core::State& state,
                       const core::Config::LaserConfig& laser_cfg,
                       cudaStream_t stream,
                       const parallel::PartitionInfo* part,
                       const parallel::Reduction* reduction) {
  map_from_hydro_2d_impl(mesh, state, laser_cfg, stream, nullptr, part,
                         reduction);
}

void map_from_hydro_2d_cbet(LaserMesh& mesh,
                            const core::State& state,
                            const core::Config::LaserConfig& laser_cfg,
                            CbetLmFields& cbet_out,
                            cudaStream_t stream) {
  // CBET is validate-FATAL under n_ranks > 1 (v1 limitation, design doc
  // mpi_m18d_laser_burn_spec.md §2) — this path stays serial-only.
  map_from_hydro_2d_impl(mesh, state, laser_cfg, stream, &cbet_out,
                         nullptr, nullptr);
}

void compute_gradients(LaserMesh& mesh, cudaStream_t stream) {
  TENRYU_ASSERT(mesh.is_allocated(), "compute_gradients requires allocated LaserMesh");
  const int n_nodes_total = mesh.n_nodes();
  const int blocks = (n_nodes_total + 255) / 256;
  compute_gradient_kernel<<<blocks, 256, 0, stream>>>(mesh.grad_n_hat_R, mesh.grad_n_hat_Z,
                                                       mesh.n_e_hat, mesh.node_R, mesh.node_Z,
                                                       mesh.n_nodes_r, mesh.n_nodes_z);
  cuda_check(cudaGetLastError(), "compute_gradients kernel launch failed");
}

void compute_smooth_kappa(LaserMesh& mesh,
                          const double lambda_cm,
                          const double eps_n,
                          const double coulomb_log_floor,
                          cudaStream_t stream,
                          const LaserPhysExtOptions* phys_ext) {
  TENRYU_ASSERT(mesh.is_allocated(), "compute_smooth_kappa requires allocated LaserMesh");
  const int n_nodes_total = mesh.n_nodes();
  const int blocks = (n_nodes_total + 255) / 256;
  if (phys_ext != nullptr) {
    compute_smooth_kappa_ext_kernel<<<blocks, 256, 0, stream>>>(
        mesh.smooth_kappa_factor, mesh.n_e_hat, mesh.T_e, mesh.Zbar,
        lambda_cm, eps_n, coulomb_log_floor, n_nodes_total, *phys_ext);
  } else {
    compute_smooth_kappa_kernel<<<blocks, 256, 0, stream>>>(
        mesh.smooth_kappa_factor, mesh.n_e_hat, mesh.T_e, mesh.Zbar, lambda_cm, eps_n,
        coulomb_log_floor, n_nodes_total);
  }
  cuda_check(cudaGetLastError(), "compute_smooth_kappa kernel launch failed");
  refresh_radial_profile_1d(
      mesh, true, stream, phys_ext != nullptr ? mesh.radial_T_e : nullptr);
}

void upload_zeff_table(LaserMesh& mesh,
                       const double* host_ratio,
                       const int n) {
  TENRYU_ASSERT(host_ratio != nullptr,
                "upload_zeff_table requires a non-null host table");
  TENRYU_ASSERT(n > 0, "upload_zeff_table requires n > 0");
  TENRYU_ASSERT(mesh.zeff_table_dev == nullptr ||
                    mesh.zeff_table_capacity == n,
                "upload_zeff_table size mismatch");
  if (mesh.zeff_table_dev == nullptr) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&mesh.zeff_table_dev),
                          static_cast<std::size_t>(n) * sizeof(double)),
               "upload_zeff_table cudaMalloc failed");
    mesh.zeff_table_capacity = n;
  }
  cuda_check(cudaMemcpy(mesh.zeff_table_dev, host_ratio,
                        static_cast<std::size_t>(n) * sizeof(double),
                        cudaMemcpyHostToDevice),
             "upload_zeff_table cudaMemcpy H2D failed");
}

double compute_min_cell_spacing(const LaserMesh& mesh) {
  if (!mesh.is_allocated()) {
    return 0.0;
  }

  std::vector<double> node_R(static_cast<std::size_t>(mesh.n_nodes_r), 0.0);
  std::vector<double> node_Z(static_cast<std::size_t>(mesh.n_nodes_z), 0.0);
  cuda_check(cudaMemcpy(node_R.data(), mesh.node_R, node_R.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "compute_min_cell_spacing memcpy node_R failed");
  cuda_check(cudaMemcpy(node_Z.data(), mesh.node_Z, node_Z.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "compute_min_cell_spacing memcpy node_Z failed");

  double min_dr = std::numeric_limits<double>::max();
  for (int i = 0; i < mesh.nr; ++i) {
    min_dr = std::min(min_dr, node_R[static_cast<std::size_t>(i + 1)] -
                                  node_R[static_cast<std::size_t>(i)]);
  }
  double min_dz = std::numeric_limits<double>::max();
  for (int j = 0; j < mesh.nz; ++j) {
    min_dz = std::min(min_dz, node_Z[static_cast<std::size_t>(j + 1)] -
                                  node_Z[static_cast<std::size_t>(j)]);
  }
  if (!(min_dr > 0.0)) {
    min_dr = min_dz;
  }
  if (!(min_dz > 0.0)) {
    min_dz = min_dr;
  }
  return std::min(min_dr, min_dz);
}

}  // namespace tenryu::laser
