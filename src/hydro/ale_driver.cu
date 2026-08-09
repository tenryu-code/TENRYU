#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/config_validate.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/kernel_guard.hpp"
#include "core/namelist/errors.hpp"
#include "core/state.hpp"
#include "coupling/profile_observability.hpp"
#include "diagnostics/energy_budget.hpp"
#include "diagnostics/mesh_deform_attribution.hpp"
#include "diagnostics/mesh_diag_dump.hpp"
#include "hydro/anti_hourglass.cuh"
#include "hydro/ale_align_monitor.hpp"
#include "hydro/ale_driver.cuh"
#include "hydro/ale_gcl.hpp"
#include "hydro/ale_escalation_predicate.hpp"
#include "hydro/ale_motion_trigger.cuh"
#include "hydro/cd_local_rezone.cuh"
#include "hydro/ale_remap.cuh"
#include "hydro/ale_remap_2d_rz.hpp"
#include "hydro/ale_rezone.cuh"
#include "hydro/ale_velocity_project.cuh"
#include "hydro/axis_ale_rezone.cuh"
#include "hydro/boundary_2d.hpp"
#include "hydro/center_patch_barrier_optimizer.cuh"
#include "hydro/cfl.hpp"
#include "hydro/central_pseudo_core.cuh"
#include "hydro/conservation_audit.hpp"
#include "hydro/core_freeze_ale.cuh"
#include "hydro/eos_context.hpp"
#include "hydro/euler_window_blend.hpp"
#include "hydro/hydro_2d.hpp"
#include "hydro/local_rezone.cuh"
#include "hydro/multiblock_center_patch_reference.cuh"
#include "hydro/per_material_eos_project.cuh"
#include "hydro/plic_remap.cuh"
#include "hydro/pole_angular_derefine.cuh"
#include "hydro/pole_axis_diag.hpp"
#include "hydro/button_morph_ale.hpp"
#include "hydro/reference_barrier_ale.hpp"
#include "hydro/rezone_objective.cuh"
#include "hydro/rollback_guard.hpp"
#include "hydro/rz_corner_mass.cuh"
#include "mesh/candidate_mesh_admissibility.hpp"
#include "mesh/mesh.hpp"
#include "mesh/path_admissibility.cuh"
#include "parallel/comm_buffers.hpp"
#include "parallel/halo_exchange.hpp"
#include "parallel/partition.hpp"
#include "parallel/reduction.hpp"

namespace tenryu::hydro::rz {
namespace {

CornerMassFallbackRecorder* g_corner_mass_fallback_recorder = nullptr;

}  // namespace

void corner_mass_fallback_run_start() {
  g_corner_mass_fallback_recorder =
      static_cast<CornerMassFallbackRecorder*>(core::device_scratch_acquire(
          "f09:corner_mass_fallback_recorder",
          sizeof(CornerMassFallbackRecorder)));
  const cudaError_t err = cudaMemset(g_corner_mass_fallback_recorder,
                                    0,
                                    sizeof(CornerMassFallbackRecorder));
  TENRYU_ASSERT(err == cudaSuccess,
                "F-09 corner-mass fallback recorder reset failed");
}

CornerMassFallbackRecorder* corner_mass_fallback_device_recorder() {
  return g_corner_mass_fallback_recorder;
}

CornerMassFallbackRecorder corner_mass_fallback_copy_to_host() {
  CornerMassFallbackRecorder host{};
  if (g_corner_mass_fallback_recorder == nullptr) {
    return host;
  }
  const cudaError_t err = cudaMemcpy(&host,
                                     g_corner_mass_fallback_recorder,
                                     sizeof(CornerMassFallbackRecorder),
                                     cudaMemcpyDeviceToHost);
  TENRYU_ASSERT(err == cudaSuccess,
                "F-09 corner-mass fallback recorder D2H failed");
  return host;
}

}  // namespace tenryu::hydro::rz

namespace tenryu::hydro::ale {

namespace detail {

constexpr double kEvToErg = 1.6022e-12;
constexpr double kProtonMass = 1.6726219e-24;  // must match core::constants::proton_mass
constexpr int kSafeBacktrackDistributionBins = 64;

std::array<std::atomic<std::uint64_t>, kSafeBacktrackDistributionBins>
    g_safe_backtrack_lambda_distribution{};
std::atomic<int> g_safe_backtrack_distribution_max_exp{0};

__global__ void restore_velocity_outside_mask_kernel(
    double* __restrict__ v_r,
    double* __restrict__ v_z,
    const double* __restrict__ v_r_old,
    const double* __restrict__ v_z_old,
    const std::uint8_t* __restrict__ allowed_node_mask,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes || allowed_node_mask[n] != 0U) {
    return;
  }
  v_r[n] = v_r_old[n];
  v_z[n] = v_z_old[n];
}

void warn_invariant_corner_mass_ke_reinit_once() {
  static bool warned = false;
  if (warned) {
    return;
  }
  warned = true;
  core::log_warning(
      "ALE KE closure is recomputing Lagrangian-invariant corner_mass; "
      "conservative subzonal mass remap is deferred to Stage F");
}

int clamp_safe_backtrack_max_exp(const int max_exp) {
  return std::clamp(max_exp, 0, kSafeBacktrackDistributionBins - 1);
}

void update_safe_backtrack_distribution_max_exp(const int max_exp) {
  int observed = g_safe_backtrack_distribution_max_exp.load(std::memory_order_relaxed);
  while (observed < max_exp &&
         !g_safe_backtrack_distribution_max_exp.compare_exchange_weak(
             observed, max_exp, std::memory_order_relaxed, std::memory_order_relaxed)) {
  }
}

int safe_backtrack_lambda_bin(const double lambda, const int max_exp) {
  const int max_exp_clamped = clamp_safe_backtrack_max_exp(max_exp);
  if (!(lambda > 0.0) || !std::isfinite(lambda)) {
    return max_exp_clamped;
  }
  const double raw_bin = std::floor(-std::log2(lambda));
  if (!std::isfinite(raw_bin)) {
    return max_exp_clamped;
  }
  return std::clamp(static_cast<int>(raw_bin), 0, max_exp_clamped);
}

void record_safe_backtrack_lambda(const double lambda, const int max_exp) {
  if (!(lambda > 0.0) || !std::isfinite(lambda)) {
    return;
  }
  const int max_exp_clamped = clamp_safe_backtrack_max_exp(max_exp);
  update_safe_backtrack_distribution_max_exp(max_exp_clamped);
  const int bin = safe_backtrack_lambda_bin(lambda, max_exp_clamped);
  g_safe_backtrack_lambda_distribution[static_cast<std::size_t>(bin)].fetch_add(
      1, std::memory_order_relaxed);
}

SafeBacktrackSearchResult select_safe_backtrack_lambda(
    const int min_exp,
    const int binary_iters,
    const std::function<bool(double)>& admissible_at_lambda) {
  SafeBacktrackSearchResult result;

  ++result.trials;
  if (!admissible_at_lambda(0.0)) {
    result.status = AleStatus::PreRezoneInvalid;
    result.accepted_lambda = 0.0;
    return result;
  }

  bool found = false;
  double lo = 0.0;
  for (int e = 0; e <= min_exp; ++e) {
    const double lambda = std::ldexp(1.0, -e);
    ++result.trials;
    if (admissible_at_lambda(lambda)) {
      lo = lambda;
      found = true;
      break;
    }
  }

  if (!found) {
    result.status = AleStatus::NoLambdaAdmissible;
    result.accepted_lambda = 0.0;
    return result;
  }

  double hi = std::min(1.0, 2.0 * lo);
  for (int k = 0; k < binary_iters && hi > lo; ++k) {
    const double mid = 0.5 * (lo + hi);
    ++result.trials;
    if (admissible_at_lambda(mid)) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  result.status = AleStatus::Ok;
  result.accepted_lambda = lo;
  return result;
}

__device__ inline double atomic_add_double(double* address, const double value) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, value);
#else
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(
                        value + __longlong_as_double(static_cast<long long>(assumed)))));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
#endif
}

int velocity_bc_mode(const Boundary2DType bc) {
  if (bc == Boundary2DType::FIXED) {
    return 2;
  }
  if (bc == Boundary2DType::REFLECT) {
    return 1;
  }
  if (bc == Boundary2DType::STATE_SUPPLY) {
    return 3;
  }
  return 0;
}

__global__ void pack_conserved_kernel(
    double* __restrict__ mom_r,
    double* __restrict__ mom_z,
    double* __restrict__ e_e_cons,
    double* __restrict__ e_i_cons,
    double* __restrict__ volfrac_mat,
    const double* __restrict__ rho,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const double* __restrict__ volfrac,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const int n_mat) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }

  const double rho_c = fmax(rho[c], 0.0);
  mom_r[c] = rho_c * v_r_cell[c];
  mom_z[c] = rho_c * v_z_cell[c];
  e_e_cons[c] = rho_c * fmax(ee[c], 0.0);
  e_i_cons[c] = rho_c * fmax(ei[c], 0.0);

  for (int mat = 0; mat < n_mat; ++mat) {
    const int idx_cm = c * n_mat + mat;
    const int idx_mc = mat * n_cells + c;
    volfrac_mat[idx_mc] = fmax(volfrac[idx_cm], 0.0);
  }
}

__global__ void recover_primitive_kernel(
    double* __restrict__ rho,
    double* __restrict__ mass,
    double* __restrict__ ee,
    double* __restrict__ ei,
    double* __restrict__ v_r_cell,
    double* __restrict__ v_z_cell,
    double* __restrict__ volfrac,
    const double* __restrict__ mom_r,
    const double* __restrict__ mom_z,
    const double* __restrict__ e_e_cons,
    const double* __restrict__ e_i_cons,
    const double* __restrict__ volfrac_mat,
    const double* __restrict__ vol,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const int n_mat,
    const double rho_floor,
    double* __restrict__ dm_floor,
    double* __restrict__ E_floor) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }

  const double V = fmax(vol[c], 1.0e-30);
  const double rho_old = rho[c];
  const double rho_c = fmax(rho_old, rho_floor);
  const double rho_div = rho_c;
  if (dm_floor != nullptr) {
    atomic_add_double(dm_floor, (rho_c - rho_old) * V);
  }
  rho[c] = rho_c;
  mass[c] = rho_c * V;

  const double e_e_raw = e_e_cons[c];
  const double e_i_raw = e_i_cons[c];
  const double e_e_clamped = fmax(e_e_raw, 0.0);
  const double e_i_clamped = fmax(e_i_raw, 0.0);
  if (E_floor != nullptr) {
    const double de_density = (e_e_clamped - e_e_raw) + (e_i_clamped - e_i_raw);
    if (de_density > 0.0) {
      atomic_add_double(E_floor, de_density * V);
    }
  }

  if (rho_c > 0.0) {
    v_r_cell[c] = mom_r[c] / rho_div;
    v_z_cell[c] = mom_z[c] / rho_div;
    ee[c] = e_e_clamped / rho_div;
    ei[c] = e_i_clamped / rho_div;
  } else {
    v_r_cell[c] = 0.0;
    v_z_cell[c] = 0.0;
    ee[c] = 0.0;
    ei[c] = 0.0;
  }

  for (int mat = 0; mat < n_mat; ++mat) {
    const int idx_cm = c * n_mat + mat;
    const int idx_mc = mat * n_cells + c;
    volfrac[idx_cm] = fmax(volfrac_mat[idx_mc], 0.0);
  }
}

__global__ void pack_conserved_kernel_per_material(
    double* __restrict__ eta_rho_m,
    double* __restrict__ eta_rho_ee_m,
    double* __restrict__ eta_rho_ei_m,
    double* __restrict__ eta_rho_vr_m,
    double* __restrict__ eta_rho_vz_m,
    double* __restrict__ volfrac_mat,
    const double* __restrict__ mass_per_material,
    const double* __restrict__ Ee_per_material,
    const double* __restrict__ Ei_per_material,
    const double* __restrict__ volfrac,
    const double* __restrict__ vol,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const int n_mat,
    double* __restrict__ dm_floor,
    double* __restrict__ E_floor,
    int* __restrict__ repair_count) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end || n_mat <= 0) {
    return;
  }

  const double V = vol[c];
  if (!(V > 0.0) || !isfinite(V)) {
    if (repair_count != nullptr) {
      atomicAdd(repair_count, 1);
    }
    return;
  }
  const double vr = v_r_cell[c];
  const double vz = v_z_cell[c];

  for (int mat = 0; mat < n_mat; ++mat) {
    const int idx_cm = c * n_mat + mat;
    const int idx_mc = mat * n_cells + c;

    const double mass_raw = mass_per_material[idx_cm];
    const bool repair_mass = !isfinite(mass_raw) || mass_raw < 0.0;
    const double mass_m = repair_mass ? 0.0 : mass_raw;
    if (repair_mass && repair_count != nullptr) {
      atomicAdd(repair_count, 1);
    }
    if (repair_mass && dm_floor != nullptr && isfinite(mass_raw) && mass_raw < 0.0) {
      atomic_add_double(dm_floor, -mass_raw);
    }

    const double Ee_raw = Ee_per_material[idx_cm];
    const double Ei_raw = Ei_per_material[idx_cm];
    const bool repair_Ee = !isfinite(Ee_raw) || Ee_raw < 0.0;
    const bool repair_Ei = !isfinite(Ei_raw) || Ei_raw < 0.0;
    const double Ee_m = repair_Ee ? 0.0 : Ee_raw;
    const double Ei_m = repair_Ei ? 0.0 : Ei_raw;
    if ((repair_Ee || repair_Ei) && repair_count != nullptr) {
      atomicAdd(repair_count, 1);
    }
    if (E_floor != nullptr) {
      double dE = 0.0;
      if (repair_Ee && isfinite(Ee_raw) && Ee_raw < 0.0) {
        dE += -Ee_raw;
      }
      if (repair_Ei && isfinite(Ei_raw) && Ei_raw < 0.0) {
        dE += -Ei_raw;
      }
      if (dE > 0.0) {
        atomic_add_double(E_floor, dE);
      }
    }

    const double rho_m = mass_m / V;
    eta_rho_m[idx_mc] = rho_m;
    eta_rho_ee_m[idx_mc] = Ee_m / V;
    eta_rho_ei_m[idx_mc] = Ei_m / V;
    eta_rho_vr_m[idx_mc] = rho_m * vr;
    eta_rho_vz_m[idx_mc] = rho_m * vz;

    const double vf_raw = volfrac[idx_cm];
    volfrac_mat[idx_mc] = (isfinite(vf_raw) && vf_raw > 0.0) ? vf_raw : 0.0;
  }
}

__global__ void reduce_per_material_conserved_to_cell_kernel(
    double* __restrict__ rho,
    double* __restrict__ mom_r,
    double* __restrict__ mom_z,
    double* __restrict__ e_e_cons,
    double* __restrict__ e_i_cons,
    const double* __restrict__ eta_rho_m,
    const double* __restrict__ eta_rho_ee_m,
    const double* __restrict__ eta_rho_ei_m,
    const double* __restrict__ eta_rho_vr_m,
    const double* __restrict__ eta_rho_vz_m,
    const int n_cells,
    const int n_mat) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells || n_mat <= 0) {
    return;
  }

  double rho_sum = 0.0;
  double ee_sum = 0.0;
  double ei_sum = 0.0;
  double mom_r_sum = 0.0;
  double mom_z_sum = 0.0;
  for (int mat = 0; mat < n_mat; ++mat) {
    const int idx_mc = mat * n_cells + c;
    const double rho_m = eta_rho_m[idx_mc];
    const double ee_m = eta_rho_ee_m[idx_mc];
    const double ei_m = eta_rho_ei_m[idx_mc];
    const double mr_m = eta_rho_vr_m[idx_mc];
    const double mz_m = eta_rho_vz_m[idx_mc];
    rho_sum += (isfinite(rho_m) && rho_m > 0.0) ? rho_m : 0.0;
    ee_sum += (isfinite(ee_m) && ee_m > 0.0) ? ee_m : 0.0;
    ei_sum += (isfinite(ei_m) && ei_m > 0.0) ? ei_m : 0.0;
    mom_r_sum += isfinite(mr_m) ? mr_m : 0.0;
    mom_z_sum += isfinite(mz_m) ? mz_m : 0.0;
  }
  rho[c] = rho_sum;
  mom_r[c] = mom_r_sum;
  mom_z[c] = mom_z_sum;
  e_e_cons[c] = ee_sum;
  e_i_cons[c] = ei_sum;
}

__device__ inline void plic_unified_material_face_flux(
    double& mass_flux,
    double& Ee_flux,
    double& Ei_flux,
    double& mom_r_flux,
    double& mom_z_flux,
    const double* __restrict__ face_flux,
    const double* __restrict__ eta_rho_m_old,
    const double* __restrict__ eta_rho_ee_m_old,
    const double* __restrict__ eta_rho_ei_m_old,
    const double* __restrict__ volfrac_mat_old,
    const double* __restrict__ v_r_cell_old,
    const double* __restrict__ v_z_cell_old,
    const int face,
    const int donor_minus,
    const int donor_plus,
    const int n_cells,
    const int n_mat,
    const int mat,
    const double presence_threshold_volfrac,
    const double presence_threshold_mass_density) {
  double signed_swept_volume = 0.0;
  for (int mm = 0; mm < n_mat; ++mm) {
    signed_swept_volume += face_flux[face * n_mat + mm];
  }
  if (!(signed_swept_volume != 0.0) || !isfinite(signed_swept_volume)) {
    return;
  }

  const double dV_fm = face_flux[face * n_mat + mat];
  if (!(dV_fm != 0.0) || !isfinite(dV_fm)) {
    return;
  }
  // Donor selection here is sign-of-stored-face-flux. The stored face flux
  // convention is determined by PLIC face-flux production (see plic_remap.cu).
  // Legacy mode: stored face flux closes to +geometric swept volume; this
  //   selection then yields the legacy (sign-reversed) donor.
  // Fixed mode: stored face flux closes to -geometric swept volume; this
  //   same selection yields the physically correct (upwind) donor.
  // Do not gate this expression — Task 4 fixes the upstream face flux sign.
  const int donor = (signed_swept_volume > 0.0) ? donor_minus : donor_plus;
  if (donor < 0 || donor >= n_cells) {
    return;
  }
  const int donor_idx = mat * n_cells + donor;
  const double vf_old = volfrac_mat_old[donor_idx];
  const double rho_m_eta_old = eta_rho_m_old[donor_idx];
  if (!(vf_old > presence_threshold_volfrac) ||
      !(rho_m_eta_old > presence_threshold_mass_density) ||
      !isfinite(vf_old) || !isfinite(rho_m_eta_old)) {
    return;
  }

  const double rho_m = rho_m_eta_old / vf_old;
  const double ee_m = eta_rho_ee_m_old[donor_idx] / rho_m_eta_old;
  const double ei_m = eta_rho_ei_m_old[donor_idx] / rho_m_eta_old;
  const double F_mass = rho_m * dV_fm;
  if (!isfinite(F_mass) || !isfinite(ee_m) || !isfinite(ei_m)) {
    return;
  }

  mass_flux = F_mass;
  Ee_flux = ee_m * F_mass;
  Ei_flux = ei_m * F_mass;
  mom_r_flux = v_r_cell_old[donor] * F_mass;
  mom_z_flux = v_z_cell_old[donor] * F_mass;
}

__global__ void plic_unified_per_material_remap_kernel(
    double* __restrict__ eta_rho_m,
    double* __restrict__ eta_rho_ee_m,
    double* __restrict__ eta_rho_ei_m,
    double* __restrict__ eta_rho_vr_m,
    double* __restrict__ eta_rho_vz_m,
    const double* __restrict__ eta_rho_m_old,
    const double* __restrict__ eta_rho_ee_m_old,
    const double* __restrict__ eta_rho_ei_m_old,
    const double* __restrict__ eta_rho_vr_m_old,
    const double* __restrict__ eta_rho_vz_m_old,
    const double* __restrict__ volfrac_mat_old,
    const double* __restrict__ v_r_cell_old,
    const double* __restrict__ v_z_cell_old,
    const double* __restrict__ plic_face_flux_r,
    const double* __restrict__ plic_face_flux_z,
    const double* __restrict__ vol_old,
    const double* __restrict__ vol_new,
    const int c_begin,
    const int c_end,
    const int nr,
    const int nz,
    const int n_mat,
    const double presence_threshold_volfrac,
    const double presence_threshold_mass_density) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  const int window_n_cells = c_end - c_begin;
  const int total = window_n_cells * n_mat;
  if (tid >= total) {
    return;
  }

  const int mat = tid / window_n_cells;
  const int c = c_begin + tid - mat * window_n_cells;
  const int idx = mat * n_cells + c;
  const int i = c / nz;
  const int j = c - i * nz;
  const double V_old = vol_old[c];
  const double V_new = vol_new[c];
  if (!(V_old > 0.0) || !(V_new > 0.0) ||
      !isfinite(V_old) || !isfinite(V_new)) {
    return;
  }

  double mass_r_plus = 0.0;
  double Ee_r_plus = 0.0;
  double Ei_r_plus = 0.0;
  double mom_r_r_plus = 0.0;
  double mom_z_r_plus = 0.0;
  if (i + 1 < nr) {
    const int face = (i + 1) * nz + j;
    plic_unified_material_face_flux(mass_r_plus,
                                    Ee_r_plus,
                                    Ei_r_plus,
                                    mom_r_r_plus,
                                    mom_z_r_plus,
                                    plic_face_flux_r,
                                    eta_rho_m_old,
                                    eta_rho_ee_m_old,
                                    eta_rho_ei_m_old,
                                    volfrac_mat_old,
                                    v_r_cell_old,
                                    v_z_cell_old,
                                    face,
                                    cell_index(i, j, nz),
                                    cell_index(i + 1, j, nz),
                                    n_cells,
                                    n_mat,
                                    mat,
                                    presence_threshold_volfrac,
                                    presence_threshold_mass_density);
  }

  double mass_r_minus = 0.0;
  double Ee_r_minus = 0.0;
  double Ei_r_minus = 0.0;
  double mom_r_r_minus = 0.0;
  double mom_z_r_minus = 0.0;
  if (i > 0) {
    const int face = i * nz + j;
    plic_unified_material_face_flux(mass_r_minus,
                                    Ee_r_minus,
                                    Ei_r_minus,
                                    mom_r_r_minus,
                                    mom_z_r_minus,
                                    plic_face_flux_r,
                                    eta_rho_m_old,
                                    eta_rho_ee_m_old,
                                    eta_rho_ei_m_old,
                                    volfrac_mat_old,
                                    v_r_cell_old,
                                    v_z_cell_old,
                                    face,
                                    cell_index(i - 1, j, nz),
                                    cell_index(i, j, nz),
                                    n_cells,
                                    n_mat,
                                    mat,
                                    presence_threshold_volfrac,
                                    presence_threshold_mass_density);
  }

  double mass_z_plus = 0.0;
  double Ee_z_plus = 0.0;
  double Ei_z_plus = 0.0;
  double mom_r_z_plus = 0.0;
  double mom_z_z_plus = 0.0;
  if (j + 1 < nz) {
    const int face = i * (nz + 1) + (j + 1);
    plic_unified_material_face_flux(mass_z_plus,
                                    Ee_z_plus,
                                    Ei_z_plus,
                                    mom_r_z_plus,
                                    mom_z_z_plus,
                                    plic_face_flux_z,
                                    eta_rho_m_old,
                                    eta_rho_ee_m_old,
                                    eta_rho_ei_m_old,
                                    volfrac_mat_old,
                                    v_r_cell_old,
                                    v_z_cell_old,
                                    face,
                                    cell_index(i, j, nz),
                                    cell_index(i, j + 1, nz),
                                    n_cells,
                                    n_mat,
                                    mat,
                                    presence_threshold_volfrac,
                                    presence_threshold_mass_density);
  }

  double mass_z_minus = 0.0;
  double Ee_z_minus = 0.0;
  double Ei_z_minus = 0.0;
  double mom_r_z_minus = 0.0;
  double mom_z_z_minus = 0.0;
  if (j > 0) {
    const int face = i * (nz + 1) + j;
    plic_unified_material_face_flux(mass_z_minus,
                                    Ee_z_minus,
                                    Ei_z_minus,
                                    mom_r_z_minus,
                                    mom_z_z_minus,
                                    plic_face_flux_z,
                                    eta_rho_m_old,
                                    eta_rho_ee_m_old,
                                    eta_rho_ei_m_old,
                                    volfrac_mat_old,
                                    v_r_cell_old,
                                    v_z_cell_old,
                                    face,
                                    cell_index(i, j - 1, nz),
                                    cell_index(i, j, nz),
                                    n_cells,
                                    n_mat,
                                    mat,
                                    presence_threshold_volfrac,
                                    presence_threshold_mass_density);
  }

  eta_rho_m[idx] =
      (eta_rho_m_old[idx] * V_old - mass_r_plus + mass_r_minus -
       mass_z_plus + mass_z_minus) /
      V_new;
  eta_rho_ee_m[idx] =
      (eta_rho_ee_m_old[idx] * V_old - Ee_r_plus + Ee_r_minus -
       Ee_z_plus + Ee_z_minus) /
      V_new;
  eta_rho_ei_m[idx] =
      (eta_rho_ei_m_old[idx] * V_old - Ei_r_plus + Ei_r_minus -
       Ei_z_plus + Ei_z_minus) /
      V_new;
  eta_rho_vr_m[idx] =
      (eta_rho_vr_m_old[idx] * V_old - mom_r_r_plus + mom_r_r_minus -
       mom_r_z_plus + mom_r_z_minus) /
      V_new;
  eta_rho_vz_m[idx] =
      (eta_rho_vz_m_old[idx] * V_old - mom_z_r_plus + mom_z_r_minus -
       mom_z_z_plus + mom_z_z_minus) /
      V_new;
}

inline bool plic_reconstruction_successful(
    const plic::PlicRemapStatus& status) {
  return status.reconstruction_attempts > 0 &&
         status.reconstruction_successes + status.axis_exempt_cells ==
             status.reconstruction_attempts &&
         status.class_d_events == 0;
}

__global__ void recover_primitive_kernel_per_material(
    double* __restrict__ rho,
    double* __restrict__ mass,
    double* __restrict__ ee,
    double* __restrict__ ei,
    double* __restrict__ v_r_cell,
    double* __restrict__ v_z_cell,
    double* __restrict__ volfrac,
    double* __restrict__ mass_per_material,
    double* __restrict__ Ee_per_material,
    double* __restrict__ Ei_per_material,
    const double* __restrict__ eta_rho_m,
    const double* __restrict__ eta_rho_ee_m,
    const double* __restrict__ eta_rho_ei_m,
    const double* __restrict__ eta_rho_vr_m,
    const double* __restrict__ eta_rho_vz_m,
    const double* __restrict__ volfrac_mat,
    const double* __restrict__ vol,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const int n_mat,
    const double rho_floor,
    double* __restrict__ dm_floor,
    double* __restrict__ E_floor,
    double* __restrict__ dm_floor_per_material,
    int* __restrict__ repair_count) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end || n_mat <= 0) {
    return;
  }

  const double V = vol[c];
  if (!(V > 0.0) || !isfinite(V)) {
    if (repair_count != nullptr) {
      atomicAdd(repair_count, 1);
    }
    return;
  }

  double M = 0.0;
  double Ee_sum = 0.0;
  double Ei_sum = 0.0;
  double rho_density_sum = 0.0;
  double ee_density_sum = 0.0;
  double ei_density_sum = 0.0;
  double mom_r_density_sum = 0.0;
  double mom_z_density_sum = 0.0;
  int dominant = 0;
  double dominant_mass = -1.0;

  for (int mat = 0; mat < n_mat; ++mat) {
    const int idx_cm = c * n_mat + mat;
    const int idx_mc = mat * n_cells + c;

    const double rho_m_raw = eta_rho_m[idx_mc];
    const bool repair_mass = !isfinite(rho_m_raw) || rho_m_raw < 0.0;
    const double rho_m_density = repair_mass ? 0.0 : rho_m_raw;
    const double mass_m = rho_m_density * V;
    if (repair_mass && repair_count != nullptr) {
      atomicAdd(repair_count, 1);
    }
    if (repair_mass && dm_floor != nullptr && isfinite(rho_m_raw) && rho_m_raw < 0.0) {
      atomic_add_double(dm_floor, -rho_m_raw * V);
    }

    const double ee_density_raw = eta_rho_ee_m[idx_mc];
    const double ei_density_raw = eta_rho_ei_m[idx_mc];
    const bool repair_Ee = !isfinite(ee_density_raw) || ee_density_raw < 0.0;
    const bool repair_Ei = !isfinite(ei_density_raw) || ei_density_raw < 0.0;
    const double ee_m_density = repair_Ee ? 0.0 : ee_density_raw;
    const double ei_m_density = repair_Ei ? 0.0 : ei_density_raw;
    const double Ee_m = ee_m_density * V;
    const double Ei_m = ei_m_density * V;
    if ((repair_Ee || repair_Ei) && repair_count != nullptr) {
      atomicAdd(repair_count, 1);
    }
    if (E_floor != nullptr) {
      double dE = 0.0;
      if (repair_Ee && isfinite(ee_density_raw) && ee_density_raw < 0.0) {
        dE += -ee_density_raw * V;
      }
      if (repair_Ei && isfinite(ei_density_raw) && ei_density_raw < 0.0) {
        dE += -ei_density_raw * V;
      }
      if (dE > 0.0) {
        atomic_add_double(E_floor, dE);
      }
    }

    mass_per_material[idx_cm] = mass_m;
    Ee_per_material[idx_cm] = Ee_m;
    Ei_per_material[idx_cm] = Ei_m;
    M += mass_m;
    Ee_sum += Ee_m;
    Ei_sum += Ei_m;
    rho_density_sum += rho_m_density;
    ee_density_sum += ee_m_density;
    ei_density_sum += ei_m_density;
    if (mass_m > dominant_mass) {
      dominant_mass = mass_m;
      dominant = mat;
    }

    const double mom_r_m = eta_rho_vr_m[idx_mc];
    const double mom_z_m = eta_rho_vz_m[idx_mc];
    if (mass_m > 0.0) {
      mom_r_density_sum += isfinite(mom_r_m) ? mom_r_m : 0.0;
      mom_z_density_sum += isfinite(mom_z_m) ? mom_z_m : 0.0;
    }

    const double vf_m = volfrac_mat[idx_mc];
    volfrac[idx_cm] = (isfinite(vf_m) && vf_m > 0.0) ? vf_m : 0.0;
  }

  const double mass_floor = rho_floor * V;
  const bool floor_applied = M < mass_floor;
  if (floor_applied) {
    const double dM = mass_floor - M;
    const int idx_dom = c * n_mat + dominant;
    mass_per_material[idx_dom] += dM;
    M = mass_floor;
    if (dm_floor != nullptr) {
      atomic_add_double(dm_floor, dM);
    }
    if (dm_floor_per_material != nullptr) {
      atomic_add_double(dm_floor_per_material + dominant, dM);
    }
  }

  mass[c] = M;
  const double rho_div = floor_applied ? (M / V) : rho_density_sum;
  rho[c] = rho_div;
  if (rho_div > 0.0) {
    v_r_cell[c] = mom_r_density_sum / rho_div;
    v_z_cell[c] = mom_z_density_sum / rho_div;
    ee[c] = floor_applied ? (Ee_sum / M) : (ee_density_sum / rho_div);
    ei[c] = floor_applied ? (Ei_sum / M) : (ei_density_sum / rho_div);
  } else {
    v_r_cell[c] = 0.0;
    v_z_cell[c] = 0.0;
    ee[c] = 0.0;
    ei[c] = 0.0;
  }
}

__global__ void compute_current_corner_mass_kernel(double* __restrict__ corner_mass,
                                                   const double* __restrict__ mass,
                                                   const double* __restrict__ x_r,
                                                   const double* __restrict__ x_z,
                                                   const std::uint8_t* __restrict__ cell_nverts,
                                                   rz::CornerMassFallbackRecorder*
                                                       fallback_recorder,
                                                   const int fallback_stage,
                                                   const int corner_mass_convention,
                                                   const int nr,
                                                   const int nz,
                                                   const int corner_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  const double m_cell = fmax(mass[c], 0.0);
  rz::CornerMassFallbackProbe probe{};
  rz::compute_rz_corner_masses_for_cell(c, nz, m_cell, x_r, x_z, cell_nverts,
                                        corner_mass, corner_stride, &probe,
                                        corner_mass_convention);
  if (probe.fired == 1) {
    rz::record_corner_mass_fallback(
        fallback_recorder, probe, true, c, fallback_stage, -2);
  }
}

__device__ inline double corner_kinetic_for_cell(const double* __restrict__ corner_mass,
                                                 const double* __restrict__ v_r_node,
                                                 const double* __restrict__ v_z_node,
                                                 const std::uint8_t* __restrict__ cell_nverts,
                                                 const int c,
                                                 const int i,
                                                 const int j,
                                                 const int stride) {
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int base = c * 4;
  const double m00 = fmax(corner_mass[base + 0], 0.0);
  const double m10 = fmax(corner_mass[base + 1], 0.0);
  const double m11 = fmax(corner_mass[base + 2], 0.0);
  const int active_nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  const double vr00 = v_r_node[n00];
  const double vz00 = v_z_node[n00];
  const double vr10 = v_r_node[n10];
  const double vz10 = v_z_node[n10];
  const double vr11 = v_r_node[n11];
  const double vz11 = v_z_node[n11];
  if (active_nverts == 3) {
    return 0.5 * (m00 * (vr00 * vr00 + vz00 * vz00) +
                  m10 * (vr10 * vr10 + vz10 * vz10) +
                  m11 * (vr11 * vr11 + vz11 * vz11));
  }
  const int n01 = i * stride + (j + 1);
  const double m01 = fmax(corner_mass[base + 3], 0.0);
  const double vr01 = v_r_node[n01];
  const double vz01 = v_z_node[n01];
  return 0.5 * (m00 * (vr00 * vr00 + vz00 * vz00) +
                m10 * (vr10 * vr10 + vz10 * vz10) +
                m11 * (vr11 * vr11 + vz11 * vz11) +
                m01 * (vr01 * vr01 + vz01 * vz01));
}

__global__ void compute_corner_kinetic_density_kernel(
    double* __restrict__ kinetic_density,
    const double* __restrict__ corner_mass,
    const double* __restrict__ vol,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const std::uint8_t* __restrict__ cell_nverts,
    const int nr,
    const int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const double K = corner_kinetic_for_cell(corner_mass, v_r_node, v_z_node,
                                           cell_nverts, c, i, j, stride);
  const double V = vol[c];
  kinetic_density[c] = (V > 0.0 && isfinite(V)) ? (K / V) : 0.0;
}

__global__ void compute_cell_velocity_from_corner_momentum_kernel(
    double* __restrict__ v_r_cell,
    double* __restrict__ v_z_cell,
    const double* __restrict__ corner_mass,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const std::uint8_t* __restrict__ cell_nverts,
    const int nr,
    const int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int base = c * 4;
  const double m00 = fmax(corner_mass[base + 0], 0.0);
  const double m10 = fmax(corner_mass[base + 1], 0.0);
  const double m11 = fmax(corner_mass[base + 2], 0.0);
  const int active_nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  if (active_nverts == 3) {
    const double m_sum = m00 + m10 + m11;
    if (!(m_sum > 0.0)) {
      v_r_cell[c] = 0.0;
      v_z_cell[c] = 0.0;
      return;
    }
    v_r_cell[c] =
        (m00 * v_r_node[n00] + m10 * v_r_node[n10] +
         m11 * v_r_node[n11]) /
        m_sum;
    v_z_cell[c] =
        (m00 * v_z_node[n00] + m10 * v_z_node[n10] +
         m11 * v_z_node[n11]) /
        m_sum;
    return;
  }
  const int n01 = i * stride + (j + 1);
  const double m01 = fmax(corner_mass[base + 3], 0.0);
  const double m_sum = m00 + m10 + m11 + m01;
  if (!(m_sum > 0.0)) {
    v_r_cell[c] = 0.0;
    v_z_cell[c] = 0.0;
    return;
  }

  v_r_cell[c] = (m00 * v_r_node[n00] + m10 * v_r_node[n10] +
                 m11 * v_r_node[n11] + m01 * v_r_node[n01]) /
                m_sum;
  v_z_cell[c] = (m00 * v_z_node[n00] + m10 * v_z_node[n10] +
                 m11 * v_z_node[n11] + m01 * v_z_node[n01]) /
                m_sum;
}

__global__ void project_cell_velocity_to_nodes_corner_mass_kernel(
    double* __restrict__ v_r_node,
    double* __restrict__ v_z_node,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const double* __restrict__ corner_mass,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ node_flags,
    const int n_begin,
    const int n_end,
    const int nr,
    const int nz,
    const int r_outer_bc_mode,
    const int z_bottom_bc_mode,
    const int z_top_bc_mode) {
  const int n = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_nodes = (nr + 1) * (nz + 1);
  if (n >= n_end) {
    return;
  }

  const int stride = nz + 1;
  const int i = n / stride;
  const int j = n - i * stride;

  double m_sum = 0.0;
  double pr_sum = 0.0;
  double pz_sum = 0.0;

  for (int di = -1; di <= 0; ++di) {
    for (int dj = -1; dj <= 0; ++dj) {
      const int ic = i + di;
      const int jc = j + dj;
      if (ic < 0 || ic >= nr || jc < 0 || jc >= nz) {
        continue;
      }
      const int c = ic * nz + jc;
      int corner = 3;
      if (di == 0 && dj == 0) {
        corner = 0;
      } else if (di == -1 && dj == 0) {
        corner = 1;
      } else if (di == -1 && dj == -1) {
        corner = 2;
      }
      const int active_nverts =
          mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
      if (corner >= active_nverts) {
        continue;
      }
      const double m = fmax(corner_mass[c * 4 + corner], 0.0);
      m_sum += m;
      pr_sum += m * v_r_cell[c];
      pz_sum += m * v_z_cell[c];
    }
  }

  if (m_sum > 0.0) {
    v_r_node[n] = pr_sum / m_sum;
    v_z_node[n] = pz_sum / m_sum;
  } else {
    v_r_node[n] = 0.0;
    v_z_node[n] = 0.0;
  }

  pole_axis::apply_2d_boundary_vector_constraints(
      v_r_node[n],
      v_z_node[n],
      node_flags,
      n,
      i,
      j,
      nr,
      nz,
      r_outer_bc_mode,
      z_bottom_bc_mode,
      z_top_bc_mode,
      false);
}

__global__ void project_cell_velocity_to_nodes_corner_mass_no_bc_kernel(
    double* __restrict__ v_r_node,
    double* __restrict__ v_z_node,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const double* __restrict__ corner_mass,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_begin,
    const int n_end,
    const int nr,
    const int nz) {
  const int n = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_nodes = (nr + 1) * (nz + 1);
  if (n >= n_end) {
    return;
  }

  const int stride = nz + 1;
  const int i = n / stride;
  const int j = n - i * stride;

  double m_sum = 0.0;
  double pr_sum = 0.0;
  double pz_sum = 0.0;

  for (int di = -1; di <= 0; ++di) {
    for (int dj = -1; dj <= 0; ++dj) {
      const int ic = i + di;
      const int jc = j + dj;
      if (ic < 0 || ic >= nr || jc < 0 || jc >= nz) {
        continue;
      }
      const int c = ic * nz + jc;
      int corner = 3;
      if (di == 0 && dj == 0) {
        corner = 0;
      } else if (di == -1 && dj == 0) {
        corner = 1;
      } else if (di == -1 && dj == -1) {
        corner = 2;
      }
      const int active_nverts =
          mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
      if (corner >= active_nverts) {
        continue;
      }
      const double m = fmax(corner_mass[c * 4 + corner], 0.0);
      m_sum += m;
      pr_sum += m * v_r_cell[c];
      pz_sum += m * v_z_cell[c];
    }
  }

  if (m_sum > 0.0) {
    v_r_node[n] = pr_sum / m_sum;
    v_z_node[n] = pz_sum / m_sum;
  } else {
    v_r_node[n] = 0.0;
    v_z_node[n] = 0.0;
  }
}

constexpr int kAuditPreCols = 2;
constexpr int kAuditPostRemapCols = 3;
constexpr int kAuditDepositCols = 7;

__global__ void audit_pre_ale_cell_kernel(
    double* __restrict__ contrib,
    double* __restrict__ node_mass,
    const double* __restrict__ corner_mass,
    const double* __restrict__ mass,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const std::uint8_t* __restrict__ cell_nverts,
    const int c_begin,
    const int c_end,
    const int nr,
    const int nz) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= c_end) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int base = c * 4;
  const double m00 = fmax(corner_mass[base + 0], 0.0);
  const double m10 = fmax(corner_mass[base + 1], 0.0);
  const double m11 = fmax(corner_mass[base + 2], 0.0);
  const int active_nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  atomic_add_double(node_mass + n00, m00);
  atomic_add_double(node_mass + n10, m10);
  atomic_add_double(node_mass + n11, m11);
  if (active_nverts != 3) {
    const int n01 = i * stride + (j + 1);
    const double m01 = fmax(corner_mass[base + 3], 0.0);
    atomic_add_double(node_mass + n01, m01);
  }

  const int o = c * kAuditPreCols;
  contrib[o + 0] = corner_kinetic_for_cell(corner_mass,
                                           v_r_node,
                                           v_z_node,
                                           cell_nverts,
                                           c,
                                           i,
                                           j,
                                           stride);
  contrib[o + 1] = mass[c] * (ee[c] + ei[c]);
}

__global__ void audit_node_kinetic_kernel(double* __restrict__ contrib,
                                          const double* __restrict__ node_mass,
                                          const double* __restrict__ v_r_node,
                                          const double* __restrict__ v_z_node,
                                          const int n_begin,
                                          const int n_end,
                                          const int n_nodes) {
  const int n = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_end) {
    return;
  }

  const double m = node_mass[n];
  const double vr = v_r_node[n];
  const double vz = v_z_node[n];
  contrib[n] = 0.5 * m * (vr * vr + vz * vz);
}

__global__ void audit_scalar_volume_kernel(double* __restrict__ contrib,
                                           const double* __restrict__ field,
                                           const double* __restrict__ vol,
                                           const int c_begin,
                                           const int c_end,
                                           const int n_cells) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  contrib[c] = field[c] * vol[c];
}

__global__ void audit_post_remap_kernel(double* __restrict__ contrib,
                                        const double* __restrict__ kinetic_density_remap,
                                        const double* __restrict__ e_e_cons,
                                        const double* __restrict__ e_i_cons,
                                        const double* __restrict__ mom_r,
                                        const double* __restrict__ mom_z,
                                        const double* __restrict__ rho,
                                        const double* __restrict__ vol,
                                        const int c_begin,
                                        const int c_end,
                                        const int n_cells) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }

  const double V = vol[c];
  const double rho_c = rho[c];
  const int o = c * kAuditPostRemapCols;
  contrib[o + 0] = kinetic_density_remap[c] * V;
  contrib[o + 1] = (e_e_cons[c] + e_i_cons[c]) * V;
  contrib[o + 2] = (rho_c > 0.0 && isfinite(rho_c))
                       ? (0.5 * (mom_r[c] * mom_r[c] + mom_z[c] * mom_z[c]) * V / rho_c)
                       : 0.0;
}

__global__ void audit_corner_kinetic_total_kernel(
    double* __restrict__ contrib,
    const double* __restrict__ corner_mass,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const std::uint8_t* __restrict__ cell_nverts,
    const int nr,
    const int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  contrib[c] = corner_kinetic_for_cell(corner_mass,
                                       v_r_node,
                                       v_z_node,
                                       cell_nverts,
                                       c,
                                       i,
                                       j,
                                       stride);
}

__global__ void deposit_kinetic_closure_audit_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    double* __restrict__ contrib,
    const double* __restrict__ kinetic_density_remap,
    const double* __restrict__ corner_mass,
    const double* __restrict__ mass,
    const double* __restrict__ vol,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const std::uint8_t* __restrict__ cell_nverts,
    const int c_begin,
    const int c_end,
    const int nr,
    const int nz,
    const int two_temperature) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= c_end) {
    return;
  }

  double dI_raw = 0.0;
  double dI_after_floor = 0.0;
  double tentative_e_e = ee[c];
  double tentative_e_i = ei[c];
  int negative_dI = 0;
  int floor_e = 0;
  int floor_i = 0;

  const double m = fmax(mass[c], 0.0);
  if (m > 0.0) {
    const int i = c / nz;
    const int j = c - i * nz;
    const int stride = nz + 1;
    const double K_node =
        corner_kinetic_for_cell(corner_mass, v_r_node, v_z_node, cell_nverts,
                                c, i, j, stride);
    const double K_remap = kinetic_density_remap[c] * fmax(vol[c], 0.0);
    double dE_raw = K_remap - K_node;
    if (isfinite(dE_raw)) {
      dI_raw = dE_raw;
      negative_dI = (dE_raw < 0.0) ? 1 : 0;
    }

    if (isfinite(dE_raw) && dE_raw != 0.0) {
      const double ee_before = ee[c];
      const double ei_before = ei[c];
      const double e_e_raw = fmax(ee_before, 0.0);
      const double e_i_raw = fmax(ei_before, 0.0);
      const double floor_ee = fmin(0.0, ee_before);
      const double floor_ei = fmin(0.0, ei_before);
      const double e_sum = e_e_raw + (two_temperature != 0 ? e_i_raw : 0.0);
      bool apply = true;
      if (dE_raw < 0.0) {
        const double removable = m * e_sum;
        if (!(removable > 0.0)) {
          apply = false;
        } else {
          dE_raw = fmax(dE_raw, -removable);
        }
      }

      if (apply) {
        const double de = dE_raw / m;
        if (two_temperature == 0) {
          tentative_e_e = ee_before + de;
          tentative_e_i = ei_before;
          const double ee_after = fmax(tentative_e_e, floor_ee);
          const double ei_after = ei_before;
          floor_e = (tentative_e_e < floor_ee) ? 1 : 0;
          floor_i = 0;
          ee[c] = ee_after;
          ei[c] = ei_after;
          dI_after_floor = m * ((ee_after - ee_before) + (ei_after - ei_before));
        } else {
          const double f_e = (e_sum > 0.0 && isfinite(e_sum)) ? (e_e_raw / e_sum) : 0.5;
          tentative_e_e = ee_before + f_e * de;
          tentative_e_i = ei_before + (1.0 - f_e) * de;
          const double ee_after = fmax(tentative_e_e, floor_ee);
          const double ei_after = fmax(tentative_e_i, floor_ei);
          floor_e = (tentative_e_e < floor_ee) ? 1 : 0;
          floor_i = (tentative_e_i < floor_ei) ? 1 : 0;
          ee[c] = ee_after;
          ei[c] = ei_after;
          dI_after_floor = m * ((ee_after - ee_before) + (ei_after - ei_before));
        }
      }
    }
  }

  const int o = c * kAuditDepositCols;
  contrib[o + 0] = dI_raw;
  contrib[o + 1] = dI_after_floor;
  contrib[o + 2] = tentative_e_e;
  contrib[o + 3] = tentative_e_i;
  contrib[o + 4] = static_cast<double>(negative_dI);
  contrib[o + 5] = static_cast<double>(floor_e);
  contrib[o + 6] = static_cast<double>(floor_i);
}

__global__ void deposit_kinetic_closure_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ kinetic_density_remap,
    const double* __restrict__ corner_mass,
    const double* __restrict__ mass,
    const double* __restrict__ vol,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const std::uint8_t* __restrict__ cell_nverts,
    const int c_begin,
    const int c_end,
    const int nr,
    const int nz,
    const int two_temperature) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= c_end) {
    return;
  }

  const double m = fmax(mass[c], 0.0);
  if (!(m > 0.0)) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const double K_node =
      corner_kinetic_for_cell(corner_mass, v_r_node, v_z_node, cell_nverts,
                              c, i, j, stride);
  const double K_remap = kinetic_density_remap[c] * fmax(vol[c], 0.0);
  double dE = K_remap - K_node;
  if (!isfinite(dE) || dE == 0.0) {
    return;
  }

  const double ee_before = ee[c];
  const double ei_before = ei[c];
  const double e_e_raw = fmax(ee_before, 0.0);
  const double e_i_raw = fmax(ei_before, 0.0);
  const double floor_ee = fmin(0.0, ee_before);
  const double floor_ei = fmin(0.0, ei_before);
  const double e_sum = e_e_raw + (two_temperature != 0 ? e_i_raw : 0.0);
  if (dE < 0.0) {
    const double removable = m * e_sum;
    if (!(removable > 0.0)) {
      return;
    }
    dE = fmax(dE, -removable);
  }

  const double de = dE / m;
  if (two_temperature == 0) {
    ee[c] = fmax(ee_before + de, floor_ee);
    return;
  }

  const double f_e = (e_sum > 0.0 && isfinite(e_sum)) ? (e_e_raw / e_sum) : 0.5;
  ee[c] = fmax(ee_before + f_e * de, floor_ee);
  ei[c] = fmax(ei_before + (1.0 - f_e) * de, floor_ei);
}

__global__ void compute_kinetic_closure_deficit_capacity_kernel(
    double* __restrict__ deficit,
    double* __restrict__ capacity,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ kinetic_density_remap,
    const double* __restrict__ corner_mass,
    const double* __restrict__ mass,
    const double* __restrict__ vol,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const std::uint8_t* __restrict__ cell_nverts,
    const double e_floor,
    const int c_begin,
    const int c_end,
    const int nr,
    const int nz,
    const int two_temperature) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= c_end) {
    return;
  }

  deficit[c] = 0.0;
  capacity[c] = 0.0;
  const double m = fmax(mass[c], 0.0);
  if (!(m > 0.0)) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const double K_node =
      corner_kinetic_for_cell(corner_mass, v_r_node, v_z_node, cell_nverts,
                              c, i, j, stride);
  const double K_remap = kinetic_density_remap[c] * fmax(vol[c], 0.0);
  double dE = K_remap - K_node;
  if (!isfinite(dE)) {
    dE = 0.0;
  }

  const double ee_before = ee[c];
  const double ei_before = ei[c];
  const double e_e_raw = fmax(ee_before, 0.0);
  const double e_i_raw = fmax(ei_before, 0.0);
  const double floor_e = fmax(e_floor, 0.0);
  const double floor_ee = fmin(floor_e, ee_before);
  const double floor_ei = fmin(floor_e, ei_before);
  const double de = dE / m;
  if (two_temperature == 0) {
    const double ee_tent = ee_before + de;
    deficit[c] = fmax(0.0, floor_ee - ee_tent) * m;
    capacity[c] = fmax(0.0, ee_tent - floor_ee) * m;
    return;
  }

  const double e_sum = e_e_raw + e_i_raw;
  const double f_e = (e_sum > 0.0 && isfinite(e_sum)) ? (e_e_raw / e_sum) : 0.5;
  const double ee_tent = ee_before + f_e * de;
  const double ei_tent = ei_before + (1.0 - f_e) * de;
  const double D_e = fmax(0.0, floor_ee - ee_tent) * m;
  const double D_i = fmax(0.0, floor_ei - ei_tent) * m;
  const double C_e = fmax(0.0, ee_tent - floor_ee) * m;
  const double C_i = fmax(0.0, ei_tent - floor_ei) * m;
  deficit[c] = D_e + D_i;
  capacity[c] = C_e + C_i;
}

__global__ void apply_kinetic_closure_redistribution_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    double* __restrict__ contrib,
    const double* __restrict__ kinetic_density_remap,
    const double* __restrict__ corner_mass,
    const double* __restrict__ mass,
    const double* __restrict__ vol,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const std::uint8_t* __restrict__ cell_nverts,
    const double e_floor,
    const double absorption_factor,
    const int c_begin,
    const int c_end,
    const int nr,
    const int nz,
    const int two_temperature) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= c_end) {
    return;
  }

  double dI_raw = 0.0;
  double dI_after_floor = 0.0;
  double tentative_e_e = ee[c];
  double tentative_e_i = ei[c];
  int negative_dI = 0;
  int floor_e_flag = 0;
  int floor_i_flag = 0;

  const double m = fmax(mass[c], 0.0);
  if (m > 0.0) {
    const int i = c / nz;
    const int j = c - i * nz;
    const int stride = nz + 1;
    const double K_node =
        corner_kinetic_for_cell(corner_mass, v_r_node, v_z_node, cell_nverts,
                                c, i, j, stride);
    const double K_remap = kinetic_density_remap[c] * fmax(vol[c], 0.0);
    double dE = K_remap - K_node;
    if (!isfinite(dE)) {
      dE = 0.0;
    }
    dI_raw = dE;
    negative_dI = (dE < 0.0) ? 1 : 0;

    const double ee_before = ee[c];
    const double ei_before = ei[c];
    const double e_e_raw = fmax(ee_before, 0.0);
    const double e_i_raw = fmax(ei_before, 0.0);
    const double floor_e = fmax(e_floor, 0.0);
    const double floor_ee = fmin(floor_e, ee_before);
    const double floor_ei = fmin(floor_e, ei_before);
    const double de = dE / m;
    double ee_after = ee_before;
    double ei_after = ei_before;

    if (two_temperature == 0) {
      tentative_e_e = ee_before + de;
      tentative_e_i = ei_before;
      const double D_e = fmax(0.0, floor_ee - tentative_e_e) * m;
      const double C_e = fmax(0.0, tentative_e_e - floor_ee) * m;
      floor_e_flag = (D_e > 0.0) ? 1 : 0;
      floor_i_flag = 0;
      ee_after = (D_e > 0.0) ? floor_ee
                             : fmax(floor_ee, tentative_e_e - absorption_factor * C_e / m);
      ei_after = ei_before;
    } else {
      const double e_sum = e_e_raw + e_i_raw;
      const double f_e =
          (e_sum > 0.0 && isfinite(e_sum)) ? (e_e_raw / e_sum) : 0.5;
      tentative_e_e = ee_before + f_e * de;
      tentative_e_i = ei_before + (1.0 - f_e) * de;
      const double D_e = fmax(0.0, floor_ee - tentative_e_e) * m;
      const double D_i = fmax(0.0, floor_ei - tentative_e_i) * m;
      const double C_e = fmax(0.0, tentative_e_e - floor_ee) * m;
      const double C_i = fmax(0.0, tentative_e_i - floor_ei) * m;
      floor_e_flag = (D_e > 0.0) ? 1 : 0;
      floor_i_flag = (D_i > 0.0) ? 1 : 0;
      ee_after = (D_e > 0.0) ? floor_ee
                             : fmax(floor_ee, tentative_e_e - absorption_factor * C_e / m);
      ei_after = (D_i > 0.0) ? floor_ei
                             : fmax(floor_ei, tentative_e_i - absorption_factor * C_i / m);
    }

    ee[c] = ee_after;
    ei[c] = ei_after;
    dI_after_floor = m * ((ee_after - ee_before) + (ei_after - ei_before));
  }

  if (contrib != nullptr) {
    const int o = c * kAuditDepositCols;
    contrib[o + 0] = dI_raw;
    contrib[o + 1] = dI_after_floor;
    contrib[o + 2] = tentative_e_e;
    contrib[o + 3] = tentative_e_i;
    contrib[o + 4] = static_cast<double>(negative_dI);
    contrib[o + 5] = static_cast<double>(floor_e_flag);
    contrib[o + 6] = static_cast<double>(floor_i_flag);
  }
}

__global__ void gather_group_field_kernel(double* __restrict__ out,
                                          const double* __restrict__ in,
                                          const int n_cells,
                                          const int n_groups,
                                          const int g) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  out[c] = in[c * n_groups + g];
}

__global__ void scatter_group_field_kernel(double* __restrict__ out,
                                           const double* __restrict__ in,
                                           const int n_cells,
                                           const int n_groups,
                                           const int g) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  out[c * n_groups + g] = in[c];
}

__global__ void interpolate_rezone_candidate_kernel(
    double* __restrict__ x_r,
    double* __restrict__ x_z,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_cand,
    const double* __restrict__ x_z_cand,
    const double lambda,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }

  x_r[n] = x_r_old[n] + lambda * (x_r_cand[n] - x_r_old[n]);
  x_z[n] = x_z_old[n] + lambda * (x_z_cand[n] - x_z_old[n]);
}

struct RemapAdmissibilityResult {
  bool admissible = true;
  int nonpositive_count = 0;
  AleMinQualityCell first_fail_cell;
};

enum class FailureClass {
  None,
  AxisMargin,
  CellVolume,
};

struct PredictiveAcceptanceResult {
  bool feasible = true;
  FailureClass failure_class = FailureClass::None;
  int axis_failure_count = 0;
  int cell_vol_failure_count = 0;
  int first_axis_failing_j = -1;
  int first_vol_failing_c = -1;
  double candidate_axis_margin_min = 1.0e300;
  double trial_axis_margin_min = 1.0e300;
  double candidate_cell_vol_min = 1.0e300;
  double trial_cell_vol_min = 1.0e300;
};

const char* failure_class_name(const FailureClass failure_class) {
  switch (failure_class) {
    case FailureClass::None:
      return "none";
    case FailureClass::AxisMargin:
      return "axis_margin";
    case FailureClass::CellVolume:
      return "cell_volume";
  }
  return "unknown";
}

template <bool FixedSign>
__global__ void remap_admissibility_check_kernel(
    double* __restrict__ vol_mid,
    int* __restrict__ nonpositive_count,
    int* __restrict__ first_bad_cell,
    const double* __restrict__ vol_old,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int sweep_direction,
    const int c_begin,
    const int c_end,
    const int nr,
    const int nz) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= c_end) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;

  double dV_plus = 0.0;
  double dV_minus = 0.0;
  if (sweep_direction == 0) {
    dV_plus = swept_volume_r_face_t<FixedSign>(
        x_r_old, x_z_old, x_r_new, x_z_new, i + 1, j, nz);
    dV_minus = swept_volume_r_face_t<FixedSign>(
        x_r_old, x_z_old, x_r_new, x_z_new, i, j, nz);
  } else {
    dV_plus = swept_volume_z_face_t<FixedSign>(
        x_r_old, x_z_old, x_r_new, x_z_new, i, j + 1, nz);
    dV_minus = swept_volume_z_face_t<FixedSign>(
        x_r_old, x_z_old, x_r_new, x_z_new, i, j, nz);
  }

  const double vol = vol_old[c] - dV_plus + dV_minus;
  const bool invalid_vol = (!isfinite(vol)) || !(vol > 0.0);
  vol_mid[c] = invalid_vol ? 0.0 : vol;
  if (invalid_vol) {
    atomicAdd(nonpositive_count, 1);
    atomicMin(first_bad_cell, c);
  }
}

RemapAdmissibilityResult check_first_sweep_admissibility(
    double* d_vol_mid,
    const double* d_vol_old,
    const double* d_xr_old,
    const double* d_xz_old,
    const double* d_xr_new,
    const double* d_xz_new,
    const int c_begin,
    const int c_end,
    const int nr,
    const int nz,
    const int step_number,
    const bool donor_sign_fixed) {
  RemapAdmissibilityResult out;
  const int n_cells = nr * nz;
  if (n_cells <= 0) {
    return out;
  }

  int* d_nonpositive_count = nullptr;
  int* d_first_bad_cell = nullptr;
  d_nonpositive_count = static_cast<int*>(
      core::device_scratch_acquire("ale_driver:first_sweep:d_nonpositive_count", sizeof(int)));
  d_first_bad_cell = static_cast<int*>(
      core::device_scratch_acquire("ale_driver:first_sweep:d_first_bad_cell", sizeof(int)));
  CUDA_CHECK(cudaMemset(d_nonpositive_count, 0, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_first_bad_cell, &n_cells, sizeof(int), cudaMemcpyHostToDevice));

  const int first_dir = (step_number % 2 == 0) ? 0 : 1;
  const int blocks = ((c_end - c_begin) + 255) / 256;
  if (donor_sign_fixed) {
    remap_admissibility_check_kernel<true><<<blocks, 256>>>(d_vol_mid,
                                                            d_nonpositive_count,
                                                            d_first_bad_cell,
                                                            d_vol_old,
                                                            d_xr_old,
                                                            d_xz_old,
                                                            d_xr_new,
                                                            d_xz_new,
                                                            first_dir,
                                                            c_begin,
                                                            c_end,
                                                            nr,
                                                            nz);
  } else {
    remap_admissibility_check_kernel<false><<<blocks, 256>>>(d_vol_mid,
                                                             d_nonpositive_count,
                                                             d_first_bad_cell,
                                                             d_vol_old,
                                                             d_xr_old,
                                                             d_xz_old,
                                                             d_xr_new,
                                                             d_xz_new,
                                                             first_dir,
                                                             c_begin,
                                                             c_end,
                                                             nr,
                                                             nz);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(core::debug_kernel_sync());

  int first_bad_cell = n_cells;
  CUDA_CHECK(cudaMemcpy(&out.nonpositive_count,
                        d_nonpositive_count,
                        sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&first_bad_cell,
                        d_first_bad_cell,
                        sizeof(int),
                        cudaMemcpyDeviceToHost));

  out.admissible = (out.nonpositive_count == 0);
  if (!out.admissible && first_bad_cell >= 0 && first_bad_cell < n_cells) {
    out.first_fail_cell.c = first_bad_cell;
    out.first_fail_cell.i = first_bad_cell / nz;
    out.first_fail_cell.j = first_bad_cell - out.first_fail_cell.i * nz;
  }
  return out;
}

__device__ inline void atomic_min_double(double* address, const double value) {
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  while (value < __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(value)));
    if (assumed == old) {
      break;
    }
  }
}

__device__ inline void atomic_min_double_with_index(double* address,
                                                    int* index_address,
                                                    const double value,
                                                    const int index) {
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  while (value < __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(value)));
    if (assumed == old) {
      atomicExch(index_address, index);
      break;
    }
  }
}

__device__ inline double predictive_node_r(const double* __restrict__ x_r,
                                           const double* __restrict__ v_r,
                                           const int n,
                                           const double dt) {
  return x_r[n] + dt * v_r[n];
}

__device__ inline double predictive_node_z(const double* __restrict__ x_z,
                                           const double* __restrict__ v_z,
                                           const int n,
                                           const double dt) {
  return x_z[n] + dt * v_z[n];
}

__device__ inline double predictive_cell_volume(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int i,
    const int j,
    const int nz,
    const double dt) {
  const int n00 = node_index(i, j, nz);
  const int n10 = node_index(i + 1, j, nz);
  const int n11 = node_index(i + 1, j + 1, nz);
  const int n01 = node_index(i, j + 1, nz);
  return rz_signed_quad_volume(predictive_node_r(x_r, v_r, n00, dt),
                               predictive_node_z(x_z, v_z, n00, dt),
                               predictive_node_r(x_r, v_r, n10, dt),
                               predictive_node_z(x_z, v_z, n10, dt),
                               predictive_node_r(x_r, v_r, n11, dt),
                               predictive_node_z(x_z, v_z, n11, dt),
                               predictive_node_r(x_r, v_r, n01, dt),
                               predictive_node_z(x_z, v_z, n01, dt));
}

__global__ void predictive_axis_margin_collect_kernel(
    double* __restrict__ d_candidate_min_margin,
    double* __restrict__ d_trial_min_margin,
    int* __restrict__ d_trial_min_j,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int nz,
    const double dt_next) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nz) {
    return;
  }

  const int stride = nz + 1;
  const int n_axis_j = j;
  const int n_axis_jp1 = j + 1;
  const int n_outer_j = stride + j;
  const int n_outer_jp1 = stride + j + 1;

  double candidate_margin = axis_cell_margin(x_r[n_outer_j],
                                             x_z[n_outer_j],
                                             x_r[n_outer_jp1],
                                             x_z[n_outer_jp1],
                                             x_z[n_axis_j],
                                             x_z[n_axis_jp1]);
  if (!isfinite(candidate_margin)) {
    candidate_margin = -1.0e300;
  }
  atomic_min_double(d_candidate_min_margin, candidate_margin);

  double trial_margin =
      axis_cell_margin(x_r[n_outer_j] + dt_next * v_r[n_outer_j],
                       x_z[n_outer_j] + dt_next * v_z[n_outer_j],
                       x_r[n_outer_jp1] + dt_next * v_r[n_outer_jp1],
                       x_z[n_outer_jp1] + dt_next * v_z[n_outer_jp1],
                       x_z[n_axis_j] + dt_next * v_z[n_axis_j],
                       x_z[n_axis_jp1] + dt_next * v_z[n_axis_jp1]);
  if (!isfinite(trial_margin)) {
    trial_margin = -1.0e300;
  }
  atomic_min_double_with_index(d_trial_min_margin, d_trial_min_j, trial_margin, j);
}

__global__ void predictive_axis_failure_kernel(
    int* __restrict__ d_failure_count,
    int* __restrict__ d_first_fail_j,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int nz,
    const double dt_next,
    const double axis_margin_floor) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nz) {
    return;
  }

  const int stride = nz + 1;
  const int n_axis_j = j;
  const int n_axis_jp1 = j + 1;
  const int n_outer_j = stride + j;
  const int n_outer_jp1 = stride + j + 1;

  const double candidate_margin = axis_cell_margin(x_r[n_outer_j],
                                                   x_z[n_outer_j],
                                                   x_r[n_outer_jp1],
                                                   x_z[n_outer_jp1],
                                                   x_z[n_axis_j],
                                                   x_z[n_axis_jp1]);
  const double trial_margin =
      axis_cell_margin(x_r[n_outer_j] + dt_next * v_r[n_outer_j],
                       x_z[n_outer_j] + dt_next * v_z[n_outer_j],
                       x_r[n_outer_jp1] + dt_next * v_r[n_outer_jp1],
                       x_z[n_outer_jp1] + dt_next * v_z[n_outer_jp1],
                       x_z[n_axis_j] + dt_next * v_z[n_axis_j],
                       x_z[n_axis_jp1] + dt_next * v_z[n_axis_jp1]);
  const bool invalid =
      (!isfinite(candidate_margin)) || !(candidate_margin > 0.0) ||
      (!isfinite(trial_margin)) || !(trial_margin > axis_margin_floor);
  if (invalid) {
    atomicAdd(d_failure_count, 1);
    atomicMin(d_first_fail_j, j);
  }
}

__global__ void predictive_cell_volume_check_kernel(
    double* __restrict__ d_candidate_min_vol,
    double* __restrict__ d_trial_min_vol,
    int* __restrict__ d_failure_count,
    int* __restrict__ d_first_fail_cell,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int c_begin,
    const int c_end,
    const int nr,
    const int nz,
    const double dt_next,
    const double cell_vol_floor_fraction) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= c_end) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  double candidate_vol = predictive_cell_volume(x_r, x_z, v_r, v_z, i, j, nz, 0.0);
  double trial_vol = predictive_cell_volume(x_r, x_z, v_r, v_z, i, j, nz, dt_next);
  if (!isfinite(candidate_vol)) {
    candidate_vol = -1.0e300;
  }
  if (!isfinite(trial_vol)) {
    trial_vol = -1.0e300;
  }

  atomic_min_double(d_candidate_min_vol, candidate_vol);
  atomic_min_double(d_trial_min_vol, trial_vol);

  const double floor = cell_vol_floor_fraction * candidate_vol;
  const bool invalid =
      !(candidate_vol > 0.0) || !(trial_vol > floor);
  if (invalid) {
    atomicAdd(d_failure_count, 1);
    atomicMin(d_first_fail_cell, c);
  }
}

void log_ale_predictive_acceptance_stats(const int axis_rejects,
                                         const int cell_vol_rejects,
                                         const double dt_next) {
  if (axis_rejects <= 0 && cell_vol_rejects <= 0) {
    return;
  }
  static int log_count = 0;
  ++log_count;
  const bool emit = log_count <= 8 || ((log_count & (log_count - 1)) == 0);
  if (!emit) {
    return;
  }
  core::log_warning("[ale-stats] predictive_acceptance_rejects axis=" +
                    std::to_string(axis_rejects) +
                    " cell_vol=" + std::to_string(cell_vol_rejects) +
                    " dt_next=" + std::to_string(dt_next));
}

PredictiveAcceptanceResult check_next_step_feasibility(
    const core::State& state,
    const double* d_xr_cand,
    const double* d_xz_cand,
    const double dt_next,
    const double axis_floor_fraction,
    const double cell_vol_floor_fraction,
    const bool has_physical_rz_axis,
    const parallel::Reduction* reduction) {
  PredictiveAcceptanceResult out;
  if (state.mesh.dim != 2 || !(dt_next > 0.0) || !std::isfinite(dt_next)) {
    return out;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_nodes = state.mesh.topo.n_nodes;
  const int n_cells = state.mesh.topo.n_cells;
  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  if (nr <= 0 || nz <= 0 || n_nodes <= 0 || n_cells <= 0) {
    return out;
  }
  TENRYU_ASSERT(static_cast<int>(state.v_r.size()) == n_nodes,
                "ALE predictive acceptance requires v_r size to match n_nodes");
  TENRYU_ASSERT(static_cast<int>(state.v_z.size()) == n_nodes,
                "ALE predictive acceptance requires v_z size to match n_nodes");
  const bool axis_check_active =
      has_physical_rz_axis && axis_floor_fraction > 0.0;

  double* d_candidate_axis_min = nullptr;
  double* d_trial_axis_min = nullptr;
  double* d_candidate_cell_min = nullptr;
  double* d_trial_cell_min = nullptr;
  int* d_trial_axis_min_j = nullptr;
  int* d_axis_failure_count = nullptr;
  int* d_axis_first_fail_j = nullptr;
  int* d_cell_failure_count = nullptr;
  int* d_cell_first_fail_c = nullptr;

  d_candidate_axis_min = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:next_step:d_candidate_axis_min", sizeof(double)));
  d_trial_axis_min = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:next_step:d_trial_axis_min", sizeof(double)));
  d_candidate_cell_min = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:next_step:d_candidate_cell_min", sizeof(double)));
  d_trial_cell_min = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:next_step:d_trial_cell_min", sizeof(double)));
  d_trial_axis_min_j = static_cast<int*>(
      core::device_scratch_acquire("ale_driver:next_step:d_trial_axis_min_j", sizeof(int)));
  d_axis_failure_count = static_cast<int*>(
      core::device_scratch_acquire("ale_driver:next_step:d_axis_failure_count", sizeof(int)));
  d_axis_first_fail_j = static_cast<int*>(
      core::device_scratch_acquire("ale_driver:next_step:d_axis_first_fail_j", sizeof(int)));
  d_cell_failure_count = static_cast<int*>(
      core::device_scratch_acquire("ale_driver:next_step:d_cell_failure_count", sizeof(int)));
  d_cell_first_fail_c = static_cast<int*>(
      core::device_scratch_acquire("ale_driver:next_step:d_cell_first_fail_c", sizeof(int)));

  const double init_min = 1.0e300;
  const int zero = 0;
  const int no_axis_fail = nz + 1;
  const int no_cell_fail = n_cells;
  CUDA_CHECK(cudaMemcpy(d_candidate_axis_min,
                        &init_min,
                        sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_trial_axis_min, &init_min, sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_candidate_cell_min,
                        &init_min,
                        sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_trial_cell_min, &init_min, sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_trial_axis_min_j,
                        &no_axis_fail,
                        sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cell_failure_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cell_first_fail_c,
                        &no_cell_fail,
                        sizeof(int),
                        cudaMemcpyHostToDevice));

  const int blocks_axis = (nz + 255) / 256;
  if (axis_check_active) {
    predictive_axis_margin_collect_kernel<<<blocks_axis, 256>>>(
        d_candidate_axis_min,
        d_trial_axis_min,
        d_trial_axis_min_j,
        d_xr_cand,
        d_xz_cand,
        state.v_r.data(),
        state.v_z.data(),
        nz,
        dt_next);
    CUDA_CHECK(cudaGetLastError());
  }

  predictive_cell_volume_check_kernel<<<cw.blocks(), 256>>>(
      d_candidate_cell_min,
      d_trial_cell_min,
      d_cell_failure_count,
      d_cell_first_fail_c,
      d_xr_cand,
      d_xz_cand,
      state.v_r.data(),
      state.v_z.data(),
      cw.begin,
      cw.end,
      nr,
      nz,
      dt_next,
      cell_vol_floor_fraction);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(core::debug_kernel_sync());

  int trial_axis_min_j = no_axis_fail;
  CUDA_CHECK(cudaMemcpy(&out.candidate_axis_margin_min,
                        d_candidate_axis_min,
                        sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&out.trial_axis_margin_min,
                        d_trial_axis_min,
                        sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&trial_axis_min_j,
                        d_trial_axis_min_j,
                        sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&out.candidate_cell_vol_min,
                        d_candidate_cell_min,
                        sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&out.trial_cell_vol_min,
                        d_trial_cell_min,
                        sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&out.cell_vol_failure_count,
                        d_cell_failure_count,
                        sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&out.first_vol_failing_c,
                        d_cell_first_fail_c,
                        sizeof(int),
                        cudaMemcpyDeviceToHost));

  if (reduction != nullptr) {
    double mins[4] = {out.candidate_axis_margin_min,
                      out.trial_axis_margin_min,
                      out.candidate_cell_vol_min,
                      out.trial_cell_vol_min};
    reduction->allreduce_min(mins, 4);
    out.candidate_axis_margin_min = mins[0];
    out.trial_axis_margin_min = mins[1];
    out.candidate_cell_vol_min = mins[2];
    out.trial_cell_vol_min = mins[3];

    double cell_fail[2] = {
        static_cast<double>(out.cell_vol_failure_count),
        (out.first_vol_failing_c >= 0 && out.first_vol_failing_c < n_cells)
            ? static_cast<double>(out.first_vol_failing_c)
            : 1.0e300};
    cell_fail[0] = reduction->allreduce_sum(cell_fail[0]);
    cell_fail[1] = reduction->allreduce_min(cell_fail[1]);
    out.cell_vol_failure_count = static_cast<int>(cell_fail[0]);
    out.first_vol_failing_c =
        (cell_fail[1] < 1.0e299) ? static_cast<int>(cell_fail[1]) : -1;
  } else if (out.first_vol_failing_c < 0 || out.first_vol_failing_c >= n_cells) {
    out.first_vol_failing_c = -1;
  }

  const double axis_margin_floor =
      axis_floor_fraction * out.candidate_axis_margin_min;
  CUDA_CHECK(cudaMemcpy(d_axis_failure_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_axis_first_fail_j,
                        &no_axis_fail,
                        sizeof(int),
                        cudaMemcpyHostToDevice));
  if (axis_check_active) {
    predictive_axis_failure_kernel<<<blocks_axis, 256>>>(
        d_axis_failure_count,
        d_axis_first_fail_j,
        d_xr_cand,
        d_xz_cand,
        state.v_r.data(),
        state.v_z.data(),
        nz,
        dt_next,
        axis_margin_floor);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(core::debug_kernel_sync());
  }

  CUDA_CHECK(cudaMemcpy(&out.axis_failure_count,
                        d_axis_failure_count,
                        sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&out.first_axis_failing_j,
                        d_axis_first_fail_j,
                        sizeof(int),
                        cudaMemcpyDeviceToHost));

  if (reduction != nullptr) {
    double axis_fail[2] = {
        static_cast<double>(out.axis_failure_count),
        (out.first_axis_failing_j >= 0 && out.first_axis_failing_j < nz)
            ? static_cast<double>(out.first_axis_failing_j)
            : 1.0e300};
    axis_fail[0] = reduction->allreduce_sum(axis_fail[0]);
    axis_fail[1] = reduction->allreduce_min(axis_fail[1]);
    out.axis_failure_count = static_cast<int>(axis_fail[0]);
    out.first_axis_failing_j =
        (axis_fail[1] < 1.0e299) ? static_cast<int>(axis_fail[1]) : -1;
  } else if (out.first_axis_failing_j < 0 || out.first_axis_failing_j >= nz) {
    out.first_axis_failing_j = -1;
  }

  out.feasible = (out.axis_failure_count == 0 && out.cell_vol_failure_count == 0);
  if (!out.feasible) {
    out.failure_class =
        (out.axis_failure_count > 0) ? FailureClass::AxisMargin : FailureClass::CellVolume;
  }
  return out;
}

__global__ void eos_reclosure_kernel(double* __restrict__ Te,
                                     double* __restrict__ Ti,
                                     double* __restrict__ Pe,
                                     double* __restrict__ Pi,
                                     double* __restrict__ ee,
                                     double* __restrict__ ei,
                                     double* __restrict__ cv_e_out,
                                     double* __restrict__ cv_i_out,
                                     const double* __restrict__ mass,
                                     const double* __restrict__ rho,
                                     const double* __restrict__ zbar,
                                     const int n_cells,
                                     const double gamma,
                                     const double A,
                                     const double rho_floor,
                                     const double te_floor,
                                     const double ti_floor,
                                     const tenryu::materials::DeviceEOSTableView tab_ion,
                                     const tenryu::materials::DeviceEOSTableView tab_ele,
                                     const bool low_density_extrapolation,
                                     double* __restrict__ E_floor) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  const double z = fmax(zbar[c], 0.0);
  const double cv_i = kEvToErg / (A * kProtonMass * (gamma - 1.0));
  const double cv_e = z * kEvToErg / (A * kProtonMass * (gamma - 1.0));

  const double rho_c = fmax(rho[c], rho_floor);
  double e_i = fmax(ei[c], 0.0);
  double e_e = fmax(ee[c], 0.0);

  const bool use_ion_table =
      tab_ion.n_rho > 0 && tab_ion.n_T > 0 && tab_ion.e_table != nullptr &&
      tab_ion.P_table != nullptr && tab_ion.cv_table != nullptr &&
      tab_ion.supports_rho_e_reclosure != 0u;
  const bool use_ele_table =
      tab_ele.n_rho > 0 && tab_ele.n_T > 0 && tab_ele.e_table != nullptr &&
      tab_ele.P_table != nullptr && tab_ele.cv_table != nullptr &&
      tab_ele.supports_rho_e_reclosure != 0u;
  if (!use_ion_table && !use_ele_table) {
    double Ti_c = ti_floor;
    double Te_c = te_floor;
    double de_floor_specific = 0.0;
    if (cv_i > 0.0) {
      const double Ti_old = e_i / cv_i;
      Ti_c = fmax(Ti_old, ti_floor);
      if (Ti_c > Ti_old) {
        de_floor_specific += cv_i * (Ti_c - Ti_old);
      }
      e_i = cv_i * Ti_c;
    } else {
      e_i = 0.0;
    }
    if (cv_e > 0.0) {
      const double Te_old = e_e / cv_e;
      Te_c = fmax(Te_old, te_floor);
      if (Te_c > Te_old) {
        de_floor_specific += cv_e * (Te_c - Te_old);
      }
      e_e = cv_e * Te_c;
    } else {
      e_e = 0.0;
    }

    if (E_floor != nullptr && de_floor_specific > 0.0) {
      const double m = fmax(mass[c], 0.0);
      atomic_add_double(E_floor, de_floor_specific * m);
    }

    Ti[c] = Ti_c;
    Te[c] = Te_c;
    ei[c] = e_i;
    ee[c] = e_e;
    Pe[c] = (gamma - 1.0) * rho_c * e_e;
    Pi[c] = (gamma - 1.0) * rho_c * e_i;
    if (cv_i_out != nullptr) {
      cv_i_out[c] = cv_i;
    }
    if (cv_e_out != nullptr) {
      cv_e_out[c] = cv_e;
    }
    return;
  }

  double Ti_c = ti_floor;
  double Te_c = te_floor;
  double Pi_c = 0.0;
  double Pe_c = 0.0;
  double cv_i_c = cv_i;
  double cv_e_c = cv_e;
  double de_floor_specific = 0.0;
  if (use_ion_table) {
    const auto inv = tenryu::materials::device_inverse_reclose_with_low_density_extrap(
        tab_ion, rho_c, e_i, ti_floor, 1.0, A, low_density_extrapolation);
    Ti_c = inv.T;
    Pi_c = inv.pressure;
    if (inv.T <= fmax(ti_floor, 1.0e-30) && inv.energy > e_i) {
      de_floor_specific += inv.energy - e_i;
    }
    e_i = inv.energy;
    cv_i_c = inv.cv;
  } else if (cv_i > 0.0) {
    const double Ti_old = e_i / cv_i;
    Ti_c = fmax(Ti_old, ti_floor);
    if (Ti_c > Ti_old) {
      de_floor_specific += cv_i * (Ti_c - Ti_old);
    }
    e_i = cv_i * Ti_c;
    Pi_c = (gamma - 1.0) * rho_c * e_i;
  } else {
    e_i = 0.0;
    cv_i_c = 0.0;
    Pi_c = 0.0;
  }
  if (use_ele_table) {
    const auto inv = tenryu::materials::device_inverse_reclose_with_low_density_extrap(
        tab_ele, rho_c, e_e, te_floor, z, A, low_density_extrapolation);
    Te_c = inv.T;
    Pe_c = inv.pressure;
    if (inv.T <= fmax(te_floor, 1.0e-30) && inv.energy > e_e) {
      de_floor_specific += inv.energy - e_e;
    }
    e_e = inv.energy;
    cv_e_c = inv.cv;
  } else if (cv_e > 0.0) {
    const double Te_old = e_e / cv_e;
    Te_c = fmax(Te_old, te_floor);
    if (Te_c > Te_old) {
      de_floor_specific += cv_e * (Te_c - Te_old);
    }
    e_e = cv_e * Te_c;
    Pe_c = (gamma - 1.0) * rho_c * e_e;
  } else {
    e_e = 0.0;
    cv_e_c = 0.0;
    Pe_c = 0.0;
  }

  if (E_floor != nullptr && de_floor_specific > 0.0) {
    const double m = fmax(mass[c], 0.0);
    atomic_add_double(E_floor, de_floor_specific * m);
  }

  Ti[c] = Ti_c;
  Te[c] = Te_c;
  ei[c] = e_i;
  ee[c] = e_e;
  Pe[c] = Pe_c;
  Pi[c] = Pi_c;
  if (cv_i_out != nullptr) {
    cv_i_out[c] = cv_i_c;
  }
  if (cv_e_out != nullptr) {
    cv_e_out[c] = cv_e_c;
  }
}

std::vector<double> reduce_device_column_sums(const double* d_values,
                                              const int n_rows,
                                              const int n_cols) {
  std::vector<double> sums(static_cast<std::size_t>(std::max(n_cols, 0)), 0.0);
  if (d_values == nullptr || n_rows <= 0 || n_cols <= 0) {
    return sums;
  }

  std::vector<double> host(static_cast<std::size_t>(n_rows) *
                               static_cast<std::size_t>(n_cols),
                           0.0);
  CUDA_CHECK(cudaMemcpy(host.data(),
                        d_values,
                        host.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
  std::vector<long double> accum(static_cast<std::size_t>(n_cols), 0.0L);
  for (int r = 0; r < n_rows; ++r) {
    const int base = r * n_cols;
    for (int col = 0; col < n_cols; ++col) {
      accum[static_cast<std::size_t>(col)] +=
          static_cast<long double>(host[static_cast<std::size_t>(base + col)]);
    }
  }
  for (int col = 0; col < n_cols; ++col) {
    sums[static_cast<std::size_t>(col)] =
        static_cast<double>(accum[static_cast<std::size_t>(col)]);
  }
  return sums;
}

double reduce_device_sum(const double* d_values, const int n_values) {
  return reduce_device_column_sums(d_values, n_values, 1)[0];
}

double reduce_global_sum(const double local, const parallel::Reduction* reduction) {
  double value = local;
  if (reduction != nullptr) {
    reduction->allreduce_sum(&value, 1);
  }
  return value;
}

double reduce_global_max(const double local, const parallel::Reduction* reduction) {
  double value = local;
  if (reduction != nullptr) {
    value = reduction->allreduce_max(value);
  }
  return value;
}

double total_energy(const diagnostics::EnergyTotals& totals) {
  return totals.E_int_e + totals.E_int_i + totals.E_kin;
}

tenryu::coupling::EscapeValveEvent::CellState capture_escape_valve_cell_state(
    const core::State& state,
    const int cell_id) {
  tenryu::coupling::EscapeValveEvent::CellState out;
  if (cell_id < 0) {
    return out;
  }
  const std::size_t c = static_cast<std::size_t>(cell_id);
  const std::size_t n_cells = state.rho.size();
  if (c >= n_cells) {
    return out;
  }

  std::vector<double> rho;
  std::vector<double> Pe;
  std::vector<double> Pi;
  std::vector<double> Te;
  std::vector<double> Ti;
  state.rho.copy_to_host(rho);
  state.Pe.copy_to_host(Pe);
  state.Pi.copy_to_host(Pi);
  state.Te.copy_to_host(Te);
  state.Ti.copy_to_host(Ti);
  if (rho.size() == n_cells) {
    out.rho = rho[c];
  }
  if (Pe.size() == n_cells) {
    out.p = Pe[c] + (Pi.size() == n_cells ? Pi[c] : 0.0);
  }
  if (Te.size() == n_cells) {
    out.Te = Te[c];
  }
  if (Ti.size() == n_cells) {
    out.Ti = Ti[c];
  }
  return out;
}

diagnostics::EnergyTotals reduce_energy_totals_global(
    diagnostics::EnergyTotals totals,
    const parallel::Reduction* reduction) {
  if (reduction != nullptr) {
    double values[3] = {totals.E_int_e, totals.E_int_i, totals.E_kin};
    reduction->allreduce_sum(values, 3);
    totals.E_int_e = values[0];
    totals.E_int_i = values[1];
    totals.E_kin = values[2];
  }
  return totals;
}

tenryu::coupling::EscapeValveEvent make_escape_valve_event(
    const core::State& state,
    const char* split_phase,
    const char* operator_inserted,
    const char* flag_name,
    const char* reason,
    const int cell_id,
    const tenryu::coupling::EscapeValveEvent::CellState& before_state,
    const tenryu::coupling::EscapeValveEvent::CellState& after_state,
    const double mass_delta,
    const double momentum_delta_r,
    const double momentum_delta_z,
    const diagnostics::EnergyTotals& before,
    const diagnostics::EnergyTotals& after) {
  tenryu::coupling::EscapeValveEvent event;
  event.split_phase = split_phase != nullptr ? split_phase : "post_step_ale";
  event.operator_inserted =
      operator_inserted != nullptr ? operator_inserted : "unknown";
  event.order_degraded = true;
  event.E_thermal_before = before.E_int_e + before.E_int_i;
  event.E_thermal_after = after.E_int_e + after.E_int_i;
  event.E_kinetic_before = before.E_kin;
  event.E_kinetic_after = after.E_kin;
  event.step = state.step;
  event.time = state.t;
  event.cell_id = cell_id;
  if (cell_id >= 0 && state.mesh.topo.nz > 0) {
    event.cell_i = cell_id / state.mesh.topo.nz;
    event.cell_j = cell_id - event.cell_i * state.mesh.topo.nz;
  }
  event.flag_name = flag_name != nullptr ? flag_name : "";
  event.reason = reason != nullptr ? reason : "";
  event.mass_delta = mass_delta;
  event.momentum_delta_r = momentum_delta_r;
  event.momentum_delta_z = momentum_delta_z;
  event.energy_delta = total_energy(after) - total_energy(before);
  event.before_state = before_state;
  event.after_state = after_state;
  return event;
}

double relative_delta(const double delta, const double reference) {
  return std::abs(delta) / std::max(std::abs(reference), 1.0e-300);
}

std::string format_scientific(const double value) {
  std::ostringstream oss;
  oss.setf(std::ios::scientific);
  oss.precision(16);
  oss << value;
  return oss.str();
}

std::string format_scientific17(const double value) {
  std::ostringstream oss;
  oss.setf(std::ios::scientific);
  oss.precision(17);
  oss << value;
  return oss.str();
}

tenryu::mesh::MeshGeometryResult recompute_geometry_for_ale(
    core::State& state,
    const core::Config& cfg,
    tenryu::coupling::ProfileObservability* observability) {
  if (!tenryu::core::effective_mesh_geometry_soft_fail(cfg)) {
    state.mesh.recompute_geometry();
    return {};
  }
  tenryu::mesh::MeshGeometryCheckOptions opts{};
  opts.policy = tenryu::mesh::MeshGeometryFailurePolicy::SoftReturn;
  const tenryu::mesh::MeshGeometryResult result =
      state.mesh.recompute_geometry_checked(opts);
  if (!result.admissible && observability != nullptr) {
    observability->note_mesh_geometry_failure(result);
  }
  return result;
}

double compute_total_mass(const core::State& state, const parallel::Reduction* reduction) {
  const double local = reduce_device_sum(
      state.mass.data(), static_cast<int>(state.mass.size()));
  return reduce_global_sum(local, reduction);
}

std::uint8_t* upload_cell_nverts_if_tri(const core::State& state) {
  const auto& cell_nverts = state.mesh.cell_nverts;
  if (cell_nverts.size() != state.mass.size()) {
    return nullptr;
  }
  const bool has_tri = std::any_of(cell_nverts.begin(), cell_nverts.end(),
                                   [](const std::uint8_t nverts) {
                                     return nverts == 3U;
                                   });
  if (!has_tri) {
    return nullptr;
  }
  std::uint8_t* d_cell_nverts = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cell_nverts),
                        cell_nverts.size() * sizeof(std::uint8_t)));
  CUDA_CHECK(cudaMemcpy(d_cell_nverts,
                        cell_nverts.data(),
                        cell_nverts.size() * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice));
  return d_cell_nverts;
}

std::uint8_t* upload_node_flags_if_constraints(const core::State& state) {
  const auto& node_flags = state.mesh.topo.node_flags;
  if (node_flags.size() != state.x_r.size()) {
    return nullptr;
  }
  const bool has_constraints = std::any_of(
      node_flags.begin(), node_flags.end(), [](const std::uint8_t flags) {
        return (flags & (pole_axis::kNodeCenterFlag |
                         pole_axis::kNodePoleAxisFlag)) != 0U;
      });
  if (!has_constraints) {
    return nullptr;
  }
  std::uint8_t* d_node_flags = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_node_flags),
                        node_flags.size() * sizeof(std::uint8_t)));
  CUDA_CHECK(cudaMemcpy(d_node_flags,
                        node_flags.data(),
                        node_flags.size() * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice));
  return d_node_flags;
}

double compute_axis_uR_max(const core::State& state,
                           const int nz,
                           const parallel::Reduction* reduction) {
  const int axis_nodes = nz + 1;
  if (axis_nodes <= 0 || state.v_r.size() < static_cast<std::size_t>(axis_nodes)) {
    return 0.0;
  }
  std::vector<double> axis_vr(static_cast<std::size_t>(axis_nodes), 0.0);
  CUDA_CHECK(cudaMemcpy(axis_vr.data(),
                        state.v_r.data(),
                        static_cast<std::size_t>(axis_nodes) * sizeof(double),
                        cudaMemcpyDeviceToHost));
  double local_max = 0.0;
  for (const double value : axis_vr) {
    local_max = std::max(local_max, std::abs(value));
  }
  return reduce_global_max(local_max, reduction);
}

struct AxisRezoneMetrics {
  double min_edge = std::numeric_limits<double>::infinity();
  double min_altitude = std::numeric_limits<double>::infinity();
  bool sampled_incident_cell = false;
  std::vector<double> edge_length;
  std::vector<double> adjacent_cell_area;
};

struct AxisRezoneCache {
  bool initialized = false;
  int n_nodes = -1;
  int n_cells = -1;
  int last_fire_step = -1;
  int fire_count = 0;
  // Convergence-rezone time cooldown (wedge lesson: a step-based cadence
  // alone lets fires-per-unit-time explode when dt collapses).
  double conv_last_fire_t = -1.0e300;
  // Pole-sector early-arm time cooldown (dyncore2 lesson: an uncooled
  // per-step arm opens a global remap transaction every step).
  double pole_last_fire_t = -1.0e300;
  // TMOP patch-rezone cooldown (same rationale) + emergency-untangle latch
  // (waives the cooldown while an inverted-vs-target corner persists).
  double tmop_last_fire_t = -1.0e300;
  bool tmop_untangle_mode = false;
  bool convergent_locality_active = false;
  std::uint64_t convergent_locality_engaged_steps = 0;
  std::vector<int> node_ids;
  std::vector<double> initial_edge_length;
  std::vector<double> adjacent_cell_area;
  double initial_min_edge = std::numeric_limits<double>::infinity();
  double initial_min_altitude = std::numeric_limits<double>::infinity();
  double compatible_initial_total = 0.0;
  double compatible_E0 = 1.0;
  // Pristine per-cell hourglass shear captured at cache init: the
  // core-quality trigger compares DEGRADATION against this baseline
  // (rounded/D-shape seam cells are intrinsically keystone-shaped).
  axis_ale::AxisAleCoreQualityBaseline core_quality_baseline;
};

struct AxisRezoneConvergencePredicate {
  bool active = false;
  double alpha = 1.0;
  double s_dot = 0.0;
  int sampled_nodes = 0;
};

bool axis_rezone_cell_dormant(const core::State& state,
                              const int c,
                              const int n_cells) {
  if (c < 0 || c >= n_cells) {
    return false;
  }
  const std::size_t cu = static_cast<std::size_t>(c);
  const std::size_t n_cells_u = static_cast<std::size_t>(n_cells);
  const bool void_cell =
      state.cell_is_void.size() == n_cells_u && state.cell_is_void[cu] != 0U;
  const bool inactive_cell =
      state.hydro_active.size() == n_cells_u && state.hydro_active[cu] == 0;
  return void_cell || inactive_cell;
}

std::vector<double> sanitize_axis_chain_corner_mass(
    const core::State& state,
    std::vector<double> corner_mass) {
  const int n_cells = state.mesh.topo.n_cells;
  TENRYU_ASSERT(n_cells >= 0,
                "axis rezone corner mass sanitize requires non-negative cell count");
  TENRYU_ASSERT(corner_mass.size() == static_cast<std::size_t>(n_cells) * 4U,
                "axis rezone corner mass sanitize requires four corners per cell");

  double scale = 0.0;
  for (const double m : corner_mass) {
    if (std::isfinite(m)) {
      scale = std::max(scale, std::abs(m));
    }
  }
  const double tiny_negative =
      64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, scale);

  for (int c = 0; c < n_cells; ++c) {
    const bool dormant = axis_rezone_cell_dormant(state, c, n_cells);
    const int active_nverts = state_cell_active_nverts(state, c, n_cells);
    const std::size_t base = static_cast<std::size_t>(c) * 4U;
    for (int k = 0; k < 4; ++k) {
      double& m = corner_mass[base + static_cast<std::size_t>(k)];
      if (dormant) {
        m = 0.0;
        continue;
      }
      if (k >= active_nverts) {
        m = 0.0;
        continue;
      }
      TENRYU_ASSERT(std::isfinite(m),
                    "axis rezone active corner mass must be finite");
      if (m < 0.0) {
        TENRYU_ASSERT(std::abs(m) <= tiny_negative,
                      "axis rezone active corner mass must be non-negative");
        m = 0.0;
      }
    }
  }
  return corner_mass;
}

constexpr double kAxisRezoneConvergentAlphaTol = 1.0e-8;
constexpr double kAxisRezoneConvergentSdotTol = 1.0e-12;

std::unordered_map<const core::State*, AxisRezoneCache>& axis_rezone_caches() {
  static std::unordered_map<const core::State*, AxisRezoneCache> caches;
  return caches;
}

double quad_planar_area(const std::array<double, 4>& r,
                        const std::array<double, 4>& z) {
  double twice_area = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int kp = (k + 1) & 3;
    twice_area += r[static_cast<std::size_t>(k)] *
                      z[static_cast<std::size_t>(kp)] -
                  r[static_cast<std::size_t>(kp)] *
                      z[static_cast<std::size_t>(k)];
  }
  return 0.5 * std::abs(twice_area);
}

double polygon_planar_area(const std::array<double, 4>& r,
                           const std::array<double, 4>& z,
                           const int active_nverts) {
  if (active_nverts == 3) {
    double twice_area = 0.0;
    for (int k = 0; k < 3; ++k) {
      const int kp = (k + 1) % 3;
      twice_area += r[static_cast<std::size_t>(k)] *
                        z[static_cast<std::size_t>(kp)] -
                    r[static_cast<std::size_t>(kp)] *
                        z[static_cast<std::size_t>(k)];
    }
    return 0.5 * std::abs(twice_area);
  }
  return quad_planar_area(r, z);
}

double quad_min_altitude(const std::array<double, 4>& r,
                         const std::array<double, 4>& z) {
  double min_altitude = std::numeric_limits<double>::infinity();
  for (int k = 0; k < 4; ++k) {
    const int e0 = (k + 1) & 3;
    const int e1 = (k + 2) & 3;
    const double er = r[static_cast<std::size_t>(e1)] -
                      r[static_cast<std::size_t>(e0)];
    const double ez = z[static_cast<std::size_t>(e1)] -
                      z[static_cast<std::size_t>(e0)];
    const double len = std::hypot(er, ez);
    if (!(len > 0.0) || !std::isfinite(len)) {
      return 0.0;
    }
    const double pr = r[static_cast<std::size_t>(k)] -
                      r[static_cast<std::size_t>(e0)];
    const double pz = z[static_cast<std::size_t>(k)] -
                      z[static_cast<std::size_t>(e0)];
    const double altitude = std::abs(pr * ez - pz * er) / len;
    if (!std::isfinite(altitude)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    min_altitude = std::min(min_altitude, altitude);
  }
  return min_altitude;
}

double polygon_min_altitude(const std::array<double, 4>& r,
                            const std::array<double, 4>& z,
                            const int active_nverts) {
  if (active_nverts != 3) {
    return quad_min_altitude(r, z);
  }
  double min_altitude = std::numeric_limits<double>::infinity();
  for (int k = 0; k < 3; ++k) {
    const int e0 = (k + 1) % 3;
    const int e1 = (k + 2) % 3;
    const double er = r[static_cast<std::size_t>(e1)] -
                      r[static_cast<std::size_t>(e0)];
    const double ez = z[static_cast<std::size_t>(e1)] -
                      z[static_cast<std::size_t>(e0)];
    const double len = std::hypot(er, ez);
    if (!(len > 0.0) || !std::isfinite(len)) {
      return 0.0;
    }
    const double pr = r[static_cast<std::size_t>(k)] -
                      r[static_cast<std::size_t>(e0)];
    const double pz = z[static_cast<std::size_t>(k)] -
                      z[static_cast<std::size_t>(e0)];
    const double altitude = std::abs(pr * ez - pz * er) / len;
    if (!std::isfinite(altitude)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    min_altitude = std::min(min_altitude, altitude);
  }
  return min_altitude;
}

bool axis_rezone_convergent_locality_diag_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_AXIS_REZONE_CONVERGENT_LOCALITY_DIAG");
    if (raw == nullptr || *raw == '\0') {
      return false;
    }
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" ||
           value == "on" || value == "ON";
  }();
  return enabled;
}

AxisRezoneConvergencePredicate evaluate_axis_rezone_convergence_predicate(
    const core::State& state,
    AxisRezoneCache& cache,
    const parallel::Reduction* reduction) {
  AxisRezoneConvergencePredicate predicate;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (state.mesh.topo.node_flags.size() != static_cast<std::size_t>(n_nodes) ||
      state.x_r_initial.size() != static_cast<std::size_t>(n_nodes) ||
      state.x_z_initial.size() != static_cast<std::size_t>(n_nodes) ||
      state.v_r.size() != static_cast<std::size_t>(n_nodes) ||
      state.v_z.size() != static_cast<std::size_t>(n_nodes)) {
    cache.convergent_locality_active = false;
    return predicate;
  }

  std::vector<double> x_r;
  std::vector<double> x_z;
  std::vector<double> x_r_initial;
  std::vector<double> x_z_initial;
  std::vector<double> v_r;
  std::vector<double> v_z;
  state.x_r.copy_to_host(x_r);
  state.x_z.copy_to_host(x_z);
  state.x_r_initial.copy_to_host(x_r_initial);
  state.x_z_initial.copy_to_host(x_z_initial);
  state.v_r.copy_to_host(v_r);
  state.v_z.copy_to_host(v_z);

  double alpha_sum = 0.0;
  double s_dot_sum = 0.0;
  double count = 0.0;
  for (int n = 0; n < n_nodes; ++n) {
    const std::uint8_t flags = state.mesh.topo.node_flags[static_cast<std::size_t>(n)];
    if ((flags & mesh::NODE_OUTER_PHYSICAL_BOUNDARY) == 0U) {
      continue;
    }
    const double r0 = x_r_initial[static_cast<std::size_t>(n)];
    const double z0 = x_z_initial[static_cast<std::size_t>(n)];
    const double r = x_r[static_cast<std::size_t>(n)];
    const double z = x_z[static_cast<std::size_t>(n)];
    const double s0 = std::hypot(r0, z0);
    const double s = std::hypot(r, z);
    if (!(s0 > 0.0) || !(s > 0.0) || !std::isfinite(s0) ||
        !std::isfinite(s)) {
      continue;
    }
    const double vr = v_r[static_cast<std::size_t>(n)];
    const double vz = v_z[static_cast<std::size_t>(n)];
    if (!std::isfinite(vr) || !std::isfinite(vz)) {
      continue;
    }
    alpha_sum += s / s0;
    s_dot_sum += (r * vr + z * vz) / (s * s0);
    count += 1.0;
  }

  double sums[3] = {alpha_sum, s_dot_sum, count};
  if (reduction != nullptr) {
    reduction->allreduce_sum(sums, 3);
  }
  if (!(sums[2] > 0.0)) {
    cache.convergent_locality_active = false;
    return predicate;
  }
  predicate.sampled_nodes = static_cast<int>(std::llround(sums[2]));
  predicate.alpha = sums[0] / sums[2];
  predicate.s_dot = sums[1] / sums[2];
  // Rebound-scope verdict PR2: the sign-only convergence gate switches the
  // quality-driven axis rezone OFF exactly when the rebound needs it (the
  // outer boundary moves outward). Opt-in ACTIVITY predicate: gate on
  // |alpha-1| and |s_dot| magnitudes instead of their signs — a superset
  // of the convergent condition during implosion (spurious early firing is
  // bounded because the downstream core-quality trigger must ALSO fire).
  // Default OFF: the verified contract window does not need it; it is the
  // out-of-contract survival tool (stress mode).
  static const bool activity_predicate = [] {
    const char* raw =
        std::getenv("TENRYU_I1B_AXIS_REZONE_ACTIVITY_PREDICATE");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  if (activity_predicate) {
    const bool entering =
        std::abs(predicate.alpha - 1.0) > kAxisRezoneConvergentAlphaTol &&
        std::abs(predicate.s_dot) > kAxisRezoneConvergentSdotTol;
    predicate.active =
        cache.convergent_locality_active
            ? std::abs(predicate.alpha - 1.0) >
                  kAxisRezoneConvergentAlphaTol
            : entering;
    cache.convergent_locality_active = predicate.active;
    return predicate;
  }
  const bool entering =
      predicate.alpha < 1.0 - kAxisRezoneConvergentAlphaTol &&
      predicate.s_dot < -kAxisRezoneConvergentSdotTol;
  predicate.active =
      cache.convergent_locality_active
          ? predicate.alpha < 1.0 - kAxisRezoneConvergentAlphaTol
          : entering;
  cache.convergent_locality_active = predicate.active;
  return predicate;
}

AxisRezoneMetrics compute_axis_rezone_metrics(
    const core::State& state,
    const std::vector<int>& axis_node_ids,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  AxisRezoneMetrics metrics;
  const int n_nodes = state.mesh.topo.n_nodes;
  const int n_cells = state.mesh.topo.n_cells;
  if (axis_node_ids.empty()) {
    return metrics;
  }
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "axis rezone metrics require multiblock topology");
  TENRYU_ASSERT(node_r.size() == static_cast<std::size_t>(n_nodes) &&
                    node_z.size() == static_cast<std::size_t>(n_nodes),
                "axis rezone metrics require one coordinate per node");
  const auto& mb = *state.mesh.topo.multiblock;
  TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells + 1),
                "axis rezone metrics require cell-node CSR offsets");
  TENRYU_ASSERT(mb.cell_node_csr_indices.size() ==
                    static_cast<std::size_t>(n_cells) * 4U,
                "axis rezone metrics require cell-node CSR indices");

  std::vector<int> node_to_chain(static_cast<std::size_t>(n_nodes), -1);
  for (std::size_t i = 0; i < axis_node_ids.size(); ++i) {
    const int n = axis_node_ids[i];
    TENRYU_ASSERT(n >= 0 && n < n_nodes,
                  "axis rezone chain node id out of range");
    node_to_chain[static_cast<std::size_t>(n)] = static_cast<int>(i);
  }

  const std::size_t n_edges =
      axis_node_ids.size() > 0U ? axis_node_ids.size() - 1U : 0U;
  metrics.edge_length.assign(n_edges, 0.0);
  metrics.adjacent_cell_area.assign(
      n_edges, std::numeric_limits<double>::infinity());
  for (std::size_t e = 0; e < n_edges; ++e) {
    const int n0 = axis_node_ids[e];
    const int n1 = axis_node_ids[e + 1U];
    const double length =
        std::hypot(node_r[static_cast<std::size_t>(n1)] -
                       node_r[static_cast<std::size_t>(n0)],
                   node_z[static_cast<std::size_t>(n1)] -
                       node_z[static_cast<std::size_t>(n0)]);
    metrics.edge_length[e] = length;
    if (std::isfinite(length)) {
      metrics.min_edge = std::min(metrics.min_edge, length);
    }
  }

  for (int c = 0; c < n_cells; ++c) {
    if (axis_rezone_cell_dormant(state, c, n_cells)) {
      continue;
    }
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int active_nverts = state_cell_active_nverts(state, c, n_cells);
    std::array<int, 4> nodes{};
    std::array<double, 4> cr{};
    std::array<double, 4> cz{};
    std::array<int, 4> chain_idx{};
    bool touches_chain = false;
    for (int k = 0; k < active_nverts; ++k) {
      const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      TENRYU_ASSERT(n >= 0 && n < n_nodes,
                    "axis rezone metrics cell node out of range");
      nodes[static_cast<std::size_t>(k)] = n;
      cr[static_cast<std::size_t>(k)] = node_r[static_cast<std::size_t>(n)];
      cz[static_cast<std::size_t>(k)] = node_z[static_cast<std::size_t>(n)];
      chain_idx[static_cast<std::size_t>(k)] =
          node_to_chain[static_cast<std::size_t>(n)];
      touches_chain = touches_chain ||
                      chain_idx[static_cast<std::size_t>(k)] >= 0;
    }
    if (!touches_chain) {
      continue;
    }

    const double altitude = polygon_min_altitude(cr, cz, active_nverts);
    if (std::isfinite(altitude)) {
      metrics.min_altitude = std::min(metrics.min_altitude, altitude);
      metrics.sampled_incident_cell = true;
    }
    const double area = polygon_planar_area(cr, cz, active_nverts);
    if (!(std::isfinite(area) && area > 0.0)) {
      continue;
    }
    for (int a = 0; a < active_nverts; ++a) {
      const int ia = chain_idx[static_cast<std::size_t>(a)];
      if (ia < 0) {
        continue;
      }
      for (int b = a + 1; b < active_nverts; ++b) {
        const int ib = chain_idx[static_cast<std::size_t>(b)];
        if (ib < 0 || std::abs(ia - ib) != 1) {
          continue;
        }
        const int edge = std::min(ia, ib);
        metrics.adjacent_cell_area[static_cast<std::size_t>(edge)] =
            std::min(metrics.adjacent_cell_area[static_cast<std::size_t>(edge)],
                     area);
      }
    }
  }
  for (std::size_t e = 0; e < n_edges; ++e) {
    if (std::isfinite(metrics.adjacent_cell_area[e]) &&
        metrics.adjacent_cell_area[e] > 0.0) {
      continue;
    }
    const double length = metrics.edge_length[e];
    const double fallback_area = length * length;
    if (std::isfinite(fallback_area) && fallback_area > 0.0) {
      metrics.adjacent_cell_area[e] = fallback_area;
    }
  }
  return metrics;
}

__global__ void install_axis_rezone_target_kernel(
    double* __restrict__ target_r,
    double* __restrict__ target_z,
    const int* __restrict__ node_ids,
    const double* __restrict__ z_target,
    const int n_axis_nodes) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_axis_nodes) {
    return;
  }
  const int node = node_ids[tid];
  target_r[node] = 0.0;
  target_z[node] = z_target[tid];
}

void install_axis_rezone_target(double* d_target_r,
                                double* d_target_z,
                                const std::vector<int>& node_ids,
                                const std::vector<double>& z_target) {
  TENRYU_ASSERT(d_target_r != nullptr && d_target_z != nullptr,
                "axis rezone target install requires target buffers");
  TENRYU_ASSERT(node_ids.size() == z_target.size(),
                "axis rezone target install requires paired node/z arrays");
  const int n_axis_nodes = static_cast<int>(node_ids.size());
  if (n_axis_nodes <= 0) {
    return;
  }
  int* d_node_ids = nullptr;
  double* d_z_target = nullptr;
  d_node_ids = static_cast<int*>(
      core::device_scratch_acquire("ale_driver:axis_target:d_node_ids",
                                   node_ids.size() * sizeof(int)));
  d_z_target = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:axis_target:d_z_target",
                                   z_target.size() * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_node_ids,
                        node_ids.data(),
                        node_ids.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_z_target,
                        z_target.data(),
                        z_target.size() * sizeof(double),
                        cudaMemcpyHostToDevice));
  const int blocks = (n_axis_nodes + 255) / 256;
  install_axis_rezone_target_kernel<<<blocks, 256>>>(
      d_target_r, d_target_z, d_node_ids, d_z_target, n_axis_nodes);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(core::debug_kernel_sync());
}

__global__ void install_axis_rezone_patch_target_kernel(
    double* __restrict__ target_r,
    double* __restrict__ target_z,
    const int* __restrict__ node_ids,
    const double* __restrict__ r_target,
    const double* __restrict__ z_target,
    const int n_nodes) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_nodes) {
    return;
  }
  const int node = node_ids[tid];
  target_r[node] = r_target[tid];
  target_z[node] = z_target[tid];
}

void install_axis_rezone_patch_target(
    double* d_target_r,
    double* d_target_z,
    const axis_ale::AxisAlePatchTarget& target) {
  TENRYU_ASSERT(d_target_r != nullptr && d_target_z != nullptr,
                "axis rezone patch target install requires target buffers");
  TENRYU_ASSERT(target.node_ids.size() == target.r_target.size() &&
                    target.node_ids.size() == target.z_target.size(),
                "axis rezone patch target install requires paired node/r/z arrays");
  const int n_patch_nodes = static_cast<int>(target.node_ids.size());
  if (n_patch_nodes <= 0) {
    return;
  }
  int* d_node_ids = nullptr;
  double* d_r_target = nullptr;
  double* d_z_target = nullptr;
  d_node_ids = static_cast<int*>(
      core::device_scratch_acquire("ale_driver:axis_patch:d_node_ids",
                                   target.node_ids.size() * sizeof(int)));
  d_r_target = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:axis_patch:d_r_target",
                                   target.r_target.size() * sizeof(double)));
  d_z_target = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:axis_patch:d_z_target",
                                   target.z_target.size() * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_node_ids,
                        target.node_ids.data(),
                        target.node_ids.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_r_target,
                        target.r_target.data(),
                        target.r_target.size() * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_z_target,
                        target.z_target.data(),
                        target.z_target.size() * sizeof(double),
                        cudaMemcpyHostToDevice));
  const int blocks = (n_patch_nodes + 255) / 256;
  install_axis_rezone_patch_target_kernel<<<blocks, 256>>>(
      d_target_r, d_target_z, d_node_ids, d_r_target, d_z_target, n_patch_nodes);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(core::debug_kernel_sync());
}

// Equal-mu pole-sector rezone target (pole-shear verdict, step 3).
// Redistributes the polar angles of the first M off-axis node columns at
// each pole onto the uniform axisymmetric-volume-fraction ladder
//   delta_ref(a) = arccos(1 - (a/M)(1 - cos delta_M)),  a = 1..M-1,
// anchored per row at the a=M column and preserving each node's spherical
// radius (an ANGULAR rezone: the tangential null mode the compression AV
// cannot see). Rows interior to the central macro cell are skipped (their
// nodes are virtual). The returned target rides the same transactional
// guard and conservative remap as the axis-chain target.
// Namelist-first with environment override: the historical
// TENRYU_I1B_POLE_REZONE* variables win when SET non-empty (experimental
// decks), otherwise the Numerics.ale.pole_sector_rezone_* keys apply.
bool pole_sector_rezone_enabled(const core::Config& cfg) {
  static const int env_state = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_REZONE");
    if (raw == nullptr || raw[0] == '\0') {
      return -1;
    }
    return raw[0] != '0' ? 1 : 0;
  }();
  if (env_state >= 0) {
    return env_state == 1;
  }
  return cfg.numerics.ale.pole_sector_rezone_enabled;
}

int pole_sector_rezone_m_theta(const core::Config& cfg) {
  static const int env_value = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_REZONE_M");
    const int v = raw != nullptr ? std::atoi(raw) : 0;
    return v > 1 ? v : -1;
  }();
  return env_value > 1 ? env_value
                       : cfg.numerics.ale.pole_sector_rezone_m_theta;
}

double pole_sector_rezone_lambda(const core::Config& cfg) {
  static const double env_value = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_REZONE_LAMBDA");
    const double v = raw != nullptr ? std::atof(raw) : 0.0;
    return (std::isfinite(v) && v > 0.0 && v <= 1.0) ? v : -1.0;
  }();
  return env_value > 0.0 ? env_value
                         : cfg.numerics.ale.pole_sector_rezone_lambda;
}

// Reference ladder: "uniform" (default) restores the initial uniform-theta
// pole zoning, delta_ref = (a/M) delta_M -- a pure de-shearing with ZERO
// displacement on a healthy mesh; "equal_mu" re-zones onto the
// axisymmetric-volume-fraction ladder (a soft angular merge; a large
// restructuring relative to the initial zoning, so it churns the remap
// when applied continuously).
bool pole_sector_rezone_equal_mu_mode(const core::Config& cfg) {
  static const int env_state = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_REZONE_MODE");
    if (raw == nullptr || raw[0] == '\0') {
      return -1;
    }
    return std::strcmp(raw, "equal_mu") == 0 ? 1 : 0;
  }();
  if (env_state >= 0) {
    return env_state == 1;
  }
  return cfg.numerics.ale.pole_sector_rezone_mode == "equal_mu";
}

// Per-node deadband: nodes within frac*delta_M of their reference angle are
// left untouched, so a healthy pole sector produces an identity target and
// no remap churn.
double pole_sector_rezone_deadband_frac(const core::Config& cfg) {
  // NOTE: 0 is a valid explicit env value ("no deadband"), so only an
  // unset/empty/out-of-range env falls through to the namelist key. The
  // historical env-unset behavior is 0.0 (the namelist default).
  static const double env_value = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_REZONE_DEADBAND_FRAC");
    if (raw == nullptr || raw[0] == '\0') {
      return -1.0;
    }
    const double v = std::atof(raw);
    return (std::isfinite(v) && v >= 0.0 && v < 1.0) ? v : -1.0;
  }();
  return env_value >= 0.0
             ? env_value
             : cfg.numerics.ale.pole_sector_rezone_deadband_frac;
}

// PR2 macro-band tapered rezone (basis-contract verdict interim #1).
bool macro_band_rezone_taper_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_MACRO_BAND_REZONE_TAPER");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  return enabled;
}

int macro_band_rezone_taper_hold() {
  static const int hold = [] {
    const char* raw = std::getenv("TENRYU_I1B_MACRO_BAND_REZONE_TAPER_HOLD");
    const int v = raw != nullptr ? std::atoi(raw) : -1;
    return v >= 0 ? v : 2;
  }();
  return hold;
}

int macro_band_rezone_taper_ramp() {
  static const int ramp = [] {
    const char* raw = std::getenv("TENRYU_I1B_MACRO_BAND_REZONE_TAPER_RAMP");
    const int v = raw != nullptr ? std::atoi(raw) : 0;
    return v > 0 ? v : 4;
  }();
  return ramp;
}

// Outer-freeze discriminator (pole-shear verdict Q2): when > 0, restrict
// pole-sector rezone targets to shell node rows q >= Q_MIN so the inner
// (dense-shell) rows stay Lagrangian. Default 0 = unrestricted.
int pole_sector_rezone_q_min() {
  static const int q_min = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_REZONE_Q_MIN");
    const int v = raw != nullptr ? std::atoi(raw) : 0;
    return v > 0 ? v : 0;
  }();
  return q_min;
}

double pole_sector_rezone_cooldown_t() {
  static const double v = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_REZONE_COOLDOWN_T");
    const double d = raw != nullptr ? std::atof(raw) : 1.0e-12;
    return std::max(d, 0.0);
  }();
  return v;
}

// Early-onset arm (pole-shear verdict: "turn on early and gently"): when
// TENRYU_I1B_POLE_REZONE_T0_NS is a non-negative time in ns, a non-empty
// pole-sector target opens the rezone transaction on every evaluation step
// with t >= T0, independent of the axis triggers (the axis chain rides
// along with an identity target; same transactional guard and conservative
// remap). Unset or negative = disabled. Returned value is in seconds.
double pole_sector_rezone_t0_seconds() {
  static const double t0 = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_REZONE_T0_NS");
    if (raw == nullptr || raw[0] == '\0') {
      return -1.0;
    }
    const double v = std::atof(raw);
    return (std::isfinite(v) && v >= 0.0) ? v * 1.0e-9 : -1.0;
  }();
  return t0;
}

axis_ale::AxisAlePatchTarget compute_pole_sector_equal_mu_target(
    const core::State& state,
    const core::Config& cfg,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  axis_ale::AxisAlePatchTarget out;
  if (!state.mesh.topo.multiblock.has_value()) {
    return out;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const mesh::BlockInfo* shell = nullptr;
  for (const auto& block : mb.blocks) {
    if (block.role == mesh::BlockRole::POLAR_SHELL) {
      shell = &block;
      break;
    }
  }
  if (shell == nullptr || shell->n_j_cells < 8 || shell->n_i_cells < 1) {
    return out;
  }
  const int ntheta = shell->n_j_cells;
  const int m_theta = std::min(pole_sector_rezone_m_theta(cfg), ntheta / 4);
  if (m_theta < 2) {
    return out;
  }
  const double lambda = pole_sector_rezone_lambda(cfg);
  // Skip node rows interior to the central macro cell (virtual members);
  // the boundary row itself is real and participates.
  int q_begin = 0;
  if (state.central_pseudo_core.built && mb.has_trifan_cap) {
    const auto& north =
        mesh::mesh_topo_trifan_fan_block(mb, mesh::BlockRole::NORTH_FAN);
    const int shell_member_rows =
        std::max(0,
                 state.central_pseudo_core.member_ring_count - mb.n_cap -
                     north.n_i_cells);
    q_begin = shell_member_rows;
  }
  const int n_nodes = static_cast<int>(node_r.size());
  // Conservation contract (Inc3a boundary-freeze lesson, measured live on
  // dyncore2: dM_rel=-1.05e-6 per fire with the old bounds): the macro
  // boundary row must stay pinned (a moved node there sweeps against
  // excluded member donors) and the outer physical boundary row must stay
  // pinned (donor-less sweep at the domain surface). Start strictly outside
  // the macro boundary and stop strictly inside the outer boundary.
  const int q_start = std::max(q_begin + 1, pole_sector_rezone_q_min());
  for (int q = q_start; q <= shell->n_i_cells - 1; ++q) {
    const int row_base = shell->owned_node_begin + q * (ntheta + 1);
    // pole 0: north (k = a, delta = theta); pole 1: south (k = ntheta - a,
    // delta = pi - theta).
    for (int pole = 0; pole < 2; ++pole) {
      const int k_anchor = pole == 0 ? m_theta : ntheta - m_theta;
      const int anchor = row_base + k_anchor;
      if (anchor < 0 || anchor >= n_nodes) {
        continue;
      }
      const double ra = node_r[static_cast<std::size_t>(anchor)];
      const double za = pole == 0 ? node_z[static_cast<std::size_t>(anchor)]
                                  : -node_z[static_cast<std::size_t>(anchor)];
      const double delta_m = std::atan2(ra, za);
      if (!std::isfinite(delta_m) || delta_m <= 0.0 ||
          delta_m >= 0.5 * 3.14159265358979323846) {
        continue;
      }
      const double one_minus_cos_m = 1.0 - std::cos(delta_m);
      const bool equal_mu = pole_sector_rezone_equal_mu_mode(cfg);
      const double deadband = pole_sector_rezone_deadband_frac(cfg) * delta_m;
      for (int a = 1; a < m_theta; ++a) {
        const int k = pole == 0 ? a : ntheta - a;
        const int node = row_base + k;
        if (node < 0 || node >= n_nodes) {
          continue;
        }
        const double r0 = node_r[static_cast<std::size_t>(node)];
        const double z0 = node_z[static_cast<std::size_t>(node)];
        const double s = std::hypot(r0, z0);
        if (!(s > 0.0) || !std::isfinite(s)) {
          continue;
        }
        const double frac =
            static_cast<double>(a) / static_cast<double>(m_theta);
        double delta_t;
        if (equal_mu) {
          const double mu = 1.0 - frac * one_minus_cos_m;
          delta_t = std::acos(std::min(1.0, std::max(-1.0, mu)));
        } else {
          delta_t = frac * delta_m;
        }
        const double z_signed = pole == 0 ? z0 : -z0;
        const double delta_0 = std::atan2(r0, z_signed);
        if (std::isfinite(delta_0) &&
            std::abs(delta_t - delta_0) < deadband) {
          continue;
        }
        if (state.mesh.topo.node_flags.size() ==
            static_cast<std::size_t>(n_nodes)) {
          const auto flags =
              state.mesh.topo.node_flags[static_cast<std::size_t>(node)];
          if ((flags & (mesh::NODE_AXIS | mesh::NODE_CENTER |
                        mesh::NODE_OUTER_PHYSICAL_BOUNDARY)) != 0U) {
            continue;
          }
        }
        const double r_t = s * std::sin(delta_t);
        const double z_abs = s * std::cos(delta_t);
        const double z_t = pole == 0 ? z_abs : -z_abs;
        out.node_ids.push_back(node);
        out.r_target.push_back(r0 + lambda * (r_t - r0));
        out.z_target.push_back(z0 + lambda * (z_t - z0));
        out.beta.push_back(1.0);
      }
    }
  }
  out.active = !out.node_ids.empty();
  out.patch_nodes = static_cast<int>(out.node_ids.size());
  return out;
}

// Convergence-following global rezone (verdict #5 Q1 Rank-1, implementation
// order #5): periodically de-noise the ACTIVE POLAR_SHELL interior so mesh
// quality survives deep convergence and the failure-triggered absorption
// walk never has to swallow the stagnating gas. Per node row (radial index
// q) the nodes are decomposed as (rho, delta) = (hypot(r,z), atan2(r,z))
// about the capsule center; the target pulls delta toward the design
// equiangular distribution and rho toward a low-mode cosine LSQ fit of the
// current row (physical low-m content is retained, grid-scale angular noise
// is removed). The target rides the UNCHANGED axis-rezone fire transaction:
// transactional path guard with lambda halving, macro-band taper, staged
// conservation ledger, TER + OptionB coherent remap, persistent-reference
// restore. Env-gated, default off (TENRYU_I1B_CONV_REZONE).
bool conv_rezone_enabled(const core::Config& cfg) {
  static const char* raw = std::getenv("TENRYU_I1B_CONV_REZONE");
  if (raw != nullptr && raw[0] != '\0') {
    return raw[0] != '0';
  }
  return cfg.numerics.ale.conv_rezone_enabled;
}

int conv_rezone_every() {
  static const int every = [] {
    const char* raw = std::getenv("TENRYU_I1B_CONV_REZONE_EVERY");
    const int v = raw != nullptr ? std::atoi(raw) : 50;
    return v > 0 ? v : 50;
  }();
  return every;
}

double conv_rezone_start_t() {
  static const double t0 = [] {
    const char* raw = std::getenv("TENRYU_I1B_CONV_REZONE_START_T");
    return raw != nullptr ? std::atof(raw) : 0.0;
  }();
  return t0;
}

double conv_rezone_alpha() {
  static const double v = [] {
    const char* raw = std::getenv("TENRYU_I1B_CONV_REZONE_ALPHA");
    const double a = raw != nullptr ? std::atof(raw) : 0.3;
    return std::min(std::max(a, 0.0), 1.0);
  }();
  return v;
}

double conv_rezone_alpha_theta() {
  static const double v = [] {
    const char* raw = std::getenv("TENRYU_I1B_CONV_REZONE_ALPHA_THETA");
    const double a = raw != nullptr ? std::atof(raw) : 0.3;
    return std::min(std::max(a, 0.0), 1.0);
  }();
  return v;
}

int conv_rezone_modes() {
  static const int modes = [] {
    const char* raw = std::getenv("TENRYU_I1B_CONV_REZONE_MODES");
    const int v = raw != nullptr ? std::atoi(raw) : 2;
    return std::min(std::max(v, 0), 4);
  }();
  return modes;
}

double conv_rezone_rough_trig() {
  static const double v = [] {
    const char* raw = std::getenv("TENRYU_I1B_CONV_REZONE_ROUGH_TRIG");
    return raw != nullptr ? std::atof(raw) : 0.05;
  }();
  return v;
}

double conv_rezone_shear_trig() {
  static const double v = [] {
    const char* raw = std::getenv("TENRYU_I1B_CONV_REZONE_SHEAR_TRIG");
    return raw != nullptr ? std::atof(raw) : 0.35;
  }();
  return v;
}

double conv_rezone_deadband_frac() {
  static const double v = [] {
    const char* raw = std::getenv("TENRYU_I1B_CONV_REZONE_DEADBAND");
    const double d = raw != nullptr ? std::atof(raw) : 0.02;
    return std::max(d, 0.0);
  }();
  return v;
}

bool conv_rezone_force() {
  static const bool force = [] {
    const char* raw = std::getenv("TENRYU_I1B_CONV_REZONE_FORCE");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  return force;
}

double conv_rezone_cooldown_t() {
  static const double v = [] {
    const char* raw = std::getenv("TENRYU_I1B_CONV_REZONE_COOLDOWN_T");
    const double d = raw != nullptr ? std::atof(raw) : 1.0e-12;
    return std::max(d, 0.0);
  }();
  return v;
}

struct ConvergenceRezoneMetrics {
  bool sampled = false;
  double max_rough = 0.0;
  double max_shear = 0.0;
  int rough_row = -1;
  int shear_row = -1;
  int rows_sampled = 0;
};

// Least-squares fit of rho(delta) onto cos(m*delta), m = 0..n_modes, via
// normal equations with partial-pivot Gaussian elimination. Returns false
// on a singular system (caller falls back to the row mean).
bool conv_rezone_fit_cos_modes(const std::vector<double>& delta,
                               const std::vector<double>& rho,
                               const int n_modes,
                               double* coeff) {
  const int n = n_modes + 1;
  const std::size_t n_pts = delta.size();
  double ata[5][5] = {};
  double atb[5] = {};
  for (std::size_t p = 0; p < n_pts; ++p) {
    double basis[5];
    for (int m = 0; m < n; ++m) {
      basis[m] = std::cos(static_cast<double>(m) * delta[p]);
    }
    for (int a = 0; a < n; ++a) {
      for (int b = 0; b < n; ++b) {
        ata[a][b] += basis[a] * basis[b];
      }
      atb[a] += basis[a] * rho[p];
    }
  }
  for (int col = 0; col < n; ++col) {
    int pivot = col;
    for (int row = col + 1; row < n; ++row) {
      if (std::abs(ata[row][col]) > std::abs(ata[pivot][col])) {
        pivot = row;
      }
    }
    if (std::abs(ata[pivot][col]) < 1.0e-300) {
      return false;
    }
    if (pivot != col) {
      for (int k = 0; k < n; ++k) {
        std::swap(ata[col][k], ata[pivot][k]);
      }
      std::swap(atb[col], atb[pivot]);
    }
    for (int row = col + 1; row < n; ++row) {
      const double f = ata[row][col] / ata[col][col];
      for (int k = col; k < n; ++k) {
        ata[row][k] -= f * ata[col][k];
      }
      atb[row] -= f * atb[col];
    }
  }
  for (int row = n - 1; row >= 0; --row) {
    double s = atb[row];
    for (int k = row + 1; k < n; ++k) {
      s -= ata[row][k] * coeff[k];
    }
    coeff[row] = s / ata[row][row];
  }
  return true;
}

axis_ale::AxisAlePatchTarget compute_convergence_rezone_target(
    const core::State& state,
    const core::Config& cfg,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    ConvergenceRezoneMetrics& metrics) {
  (void)cfg;
  axis_ale::AxisAlePatchTarget out;
  if (!state.mesh.topo.multiblock.has_value()) {
    return out;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const mesh::BlockInfo* shell = nullptr;
  for (const auto& block : mb.blocks) {
    if (block.role == mesh::BlockRole::POLAR_SHELL) {
      shell = &block;
      break;
    }
  }
  if (shell == nullptr || shell->n_j_cells < 8 || shell->n_i_cells < 2) {
    return out;
  }
  const int ntheta = shell->n_j_cells;
  const int n_rows = shell->n_i_cells + 1;
  const int n_nodes = static_cast<int>(node_r.size());
  // Rows interior to the central macro cell hold stale virtual-member
  // geometry; the macro boundary row itself must stay pinned so the global
  // reference remap never sweeps a face against excluded member donors
  // (Inc3a boundary-freeze contract). Start strictly outside it.
  int q_begin = 0;
  if (state.central_pseudo_core.built && mb.has_trifan_cap) {
    const auto& north =
        mesh::mesh_topo_trifan_fan_block(mb, mesh::BlockRole::NORTH_FAN);
    q_begin = std::max(
        0, state.central_pseudo_core.member_ring_count - mb.n_cap -
               north.n_i_cells);
  }
  // Row 0 is shared with the fan/butterfly seam and row n_rows-1 with the
  // outer physical boundary: both stay pinned.
  const int q_start = std::max(q_begin + 1, 1);
  const int q_end = n_rows - 2;
  if (q_start > q_end) {
    return out;
  }
  const double pi = 3.14159265358979323846;
  const double dtheta = pi / static_cast<double>(ntheta);
  const double alpha_rho = conv_rezone_alpha();
  const double alpha_theta = conv_rezone_alpha_theta();
  const int n_modes = conv_rezone_modes();
  const double deadband_frac = conv_rezone_deadband_frac();

  const int n_cols = ntheta + 1;
  std::vector<double> rho_cur(static_cast<std::size_t>(n_rows) * n_cols,
                              std::numeric_limits<double>::quiet_NaN());
  std::vector<double> delta_cur(static_cast<std::size_t>(n_rows) * n_cols,
                                std::numeric_limits<double>::quiet_NaN());
  std::vector<double> rho_tgt = rho_cur;
  std::vector<double> delta_tgt = delta_cur;
  std::vector<double> row_mean(static_cast<std::size_t>(n_rows), 0.0);
  const auto at = [n_cols](const int q, const int k) {
    return static_cast<std::size_t>(q) * n_cols + k;
  };
  for (int q = 0; q < n_rows; ++q) {
    double mean = 0.0;
    for (int k = 0; k < n_cols; ++k) {
      const int node = shell->owned_node_begin + q * n_cols + k;
      if (node < 0 || node >= n_nodes) {
        return out;
      }
      const double r = node_r[static_cast<std::size_t>(node)];
      const double z = node_z[static_cast<std::size_t>(node)];
      const double rho = std::hypot(r, z);
      rho_cur[at(q, k)] = rho;
      delta_cur[at(q, k)] = std::atan2(std::max(r, 0.0), z);
      mean += rho;
    }
    row_mean[static_cast<std::size_t>(q)] = mean / n_cols;
  }

  metrics.sampled = true;
  std::vector<double> delta_row(static_cast<std::size_t>(n_cols));
  std::vector<double> rho_row(static_cast<std::size_t>(n_cols));
  for (int q = q_start; q <= q_end; ++q) {
    for (int k = 0; k < n_cols; ++k) {
      delta_row[static_cast<std::size_t>(k)] = delta_cur[at(q, k)];
      rho_row[static_cast<std::size_t>(k)] = rho_cur[at(q, k)];
    }
    double coeff[5] = {};
    if (!conv_rezone_fit_cos_modes(delta_row, rho_row, n_modes, coeff)) {
      coeff[0] = row_mean[static_cast<std::size_t>(q)];
      for (int m = 1; m <= n_modes && m < 5; ++m) {
        coeff[m] = 0.0;
      }
    }
    // Local radial spacing normalizes both the roughness metric and the
    // deadband: mean gap to the adjacent rows' mean radii.
    const double gap_lo = row_mean[static_cast<std::size_t>(q)] -
                          row_mean[static_cast<std::size_t>(q - 1)];
    const double gap_hi = row_mean[static_cast<std::size_t>(q + 1)] -
                          row_mean[static_cast<std::size_t>(q)];
    const double h_row =
        std::max(0.5 * (std::abs(gap_lo) + std::abs(gap_hi)), 1.0e-300);
    ++metrics.rows_sampled;
    for (int k = 1; k < ntheta; ++k) {
      const double delta_0 = delta_cur[at(q, k)];
      const double rho_0 = rho_cur[at(q, k)];
      if (!std::isfinite(delta_0) || !std::isfinite(rho_0) || rho_0 <= 0.0) {
        continue;
      }
      double fit_0 = 0.0;
      for (int m = 0; m <= n_modes; ++m) {
        fit_0 += coeff[m] * std::cos(static_cast<double>(m) * delta_0);
      }
      const double rough = std::abs(rho_0 - fit_0) / h_row;
      if (rough > metrics.max_rough) {
        metrics.max_rough = rough;
        metrics.rough_row = q;
      }
      const double delta_design = static_cast<double>(k) * dtheta;
      const double shear = std::abs(delta_0 - delta_design) / dtheta;
      if (shear > metrics.max_shear) {
        metrics.max_shear = shear;
        metrics.shear_row = q;
      }
      const double delta_t = delta_0 + alpha_theta * (delta_design - delta_0);
      double fit_t = 0.0;
      for (int m = 0; m <= n_modes; ++m) {
        fit_t += coeff[m] * std::cos(static_cast<double>(m) * delta_t);
      }
      const double rho_t = rho_0 + alpha_rho * (fit_t - rho_0);
      if (!(rho_t > 0.0) || !std::isfinite(rho_t) || !std::isfinite(delta_t)) {
        continue;
      }
      rho_tgt[at(q, k)] = rho_t;
      delta_tgt[at(q, k)] =
          std::min(std::max(delta_t, 1.0e-6), pi - 1.0e-6);
    }
  }

  // Radial ordering clamp per column: targets must preserve strictly
  // increasing rho across rows, anchored on the pinned rows' CURRENT radii.
  // A node whose target cannot fit inside its neighbors' band reverts to
  // its current position; the transactional path guard downstream remains
  // the final safety net.
  for (int k = 1; k < ntheta; ++k) {
    double prev = rho_cur[at(q_start - 1, k)];
    for (int q = q_start; q <= q_end; ++q) {
      const std::size_t idx = at(q, k);
      const double cur = rho_cur[idx];
      double tgt = rho_tgt[idx];
      if (!std::isfinite(tgt)) {
        prev = cur;
        continue;
      }
      const double gap_min = 0.05 * std::max(cur - prev, 0.0) + 1.0e-300;
      if (tgt < prev + gap_min) {
        tgt = std::min(cur, prev + gap_min);
      }
      if (q == q_end) {
        const double outer_anchor = rho_cur[at(q_end + 1, k)];
        if (tgt > outer_anchor - gap_min) {
          tgt = std::max(cur, outer_anchor - gap_min);
        }
      }
      if (!(tgt > prev) || !std::isfinite(tgt)) {
        rho_tgt[idx] = std::numeric_limits<double>::quiet_NaN();
        prev = cur;
        continue;
      }
      rho_tgt[idx] = tgt;
      prev = tgt;
    }
  }

  for (int q = q_start; q <= q_end; ++q) {
    const double gap_lo = std::abs(row_mean[static_cast<std::size_t>(q)] -
                                   row_mean[static_cast<std::size_t>(q - 1)]);
    const double gap_hi = std::abs(row_mean[static_cast<std::size_t>(q + 1)] -
                                   row_mean[static_cast<std::size_t>(q)]);
    const double h_row = std::max(0.5 * (gap_lo + gap_hi), 1.0e-300);
    for (int k = 1; k < ntheta; ++k) {
      const std::size_t idx = at(q, k);
      if (!std::isfinite(rho_tgt[idx]) || !std::isfinite(delta_tgt[idx])) {
        continue;
      }
      const int node = shell->owned_node_begin + q * n_cols + k;
      if (state.mesh.topo.node_flags.size() ==
          static_cast<std::size_t>(n_nodes)) {
        const auto flags =
            state.mesh.topo.node_flags[static_cast<std::size_t>(node)];
        if ((flags & (mesh::NODE_AXIS | mesh::NODE_CENTER |
                      mesh::NODE_OUTER_PHYSICAL_BOUNDARY)) != 0U) {
          continue;
        }
      }
      const double r_t = rho_tgt[idx] * std::sin(delta_tgt[idx]);
      const double z_t = rho_tgt[idx] * std::cos(delta_tgt[idx]);
      const double r_0 = node_r[static_cast<std::size_t>(node)];
      const double z_0 = node_z[static_cast<std::size_t>(node)];
      if (std::hypot(r_t - r_0, z_t - z_0) < deadband_frac * h_row) {
        continue;
      }
      out.node_ids.push_back(node);
      out.r_target.push_back(r_t);
      out.z_target.push_back(z_t);
      out.beta.push_back(1.0);
    }
  }
  out.active = !out.node_ids.empty();
  out.patch_nodes = static_cast<int>(out.node_ids.size());
  return out;
}

// TMOP-like polar patch rezone (verdict #6 B1): local target-matrix
// barrier-quality optimization for the POLAR_SHELL pole/mid-latitude cells
// whose meridional shape shears while the volume monitor stays quiet
// (J_RZ = 2*pi*r*J_A: near the pole r is small, so q_J/q_edge degrade first).
// Per cell the ideal local geometry is a spherical-annular sector
// W = [h_R e_R | h_theta e_theta] built from the current mean radius and the
// design angular pitch; corner Jacobians A_k give T = A W^{-1} and the
// barrier metrics q_shape = 2 det T / tr(T^T T), q_J = det A / det W,
// q_edge = min|e|/max|e|. A projected-gradient host optimizer minimizes a
// log-barrier objective over the patch's free nodes (Inc3a freeze contract:
// macro-boundary row, outer physical boundary, axis columns and
// non-POLAR_SHELL-incident nodes stay pinned in v1). The target rides the
// same conservative fire transaction as conv_rezone. Env-gated, default
// off (TENRYU_I1B_TMOP_PATCH). References: Dobrev et al. 2020 (TMOP),
// Knupp 2001 (algebraic quality), Escobar et al. 2003 (untangling
// barriers) — docs/Papers/メッシュ品質/01_最適化_untangling.
bool tmop_patch_enabled() {
  static const bool on = [] {
    const char* raw = std::getenv("TENRYU_I1B_TMOP_PATCH");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  return on;
}

int tmop_patch_every() {
  static const int v = [] {
    const char* raw = std::getenv("TENRYU_I1B_TMOP_PATCH_EVERY");
    const int e = raw != nullptr ? std::atoi(raw) : 25;
    return e > 0 ? e : 25;
  }();
  return v;
}

double tmop_env_double(const char* name, const double fallback) {
  const char* raw = std::getenv(name);
  return (raw != nullptr && raw[0] != '\0') ? std::atof(raw) : fallback;
}

double tmop_q_shape_warn() {
  static const double v =
      tmop_env_double("TENRYU_I1B_TMOP_QSHAPE_WARN", 0.25);
  return v;
}
double tmop_q_j_warn() {
  static const double v = tmop_env_double("TENRYU_I1B_TMOP_QJ_WARN", 0.25);
  return v;
}
double tmop_q_edge_warn() {
  static const double v =
      tmop_env_double("TENRYU_I1B_TMOP_QEDGE_WARN", 0.12);
  return v;
}
double tmop_q_shape_min() {
  static const double v =
      tmop_env_double("TENRYU_I1B_TMOP_QSHAPE_MIN", 0.15);
  return v;
}
double tmop_q_j_min() {
  static const double v = tmop_env_double("TENRYU_I1B_TMOP_QJ_MIN", 0.12);
  return v;
}
double tmop_q_edge_min() {
  static const double v =
      tmop_env_double("TENRYU_I1B_TMOP_QEDGE_MIN", 0.08);
  return v;
}
int tmop_iters() {
  static const int v = [] {
    const char* raw = std::getenv("TENRYU_I1B_TMOP_ITERS");
    const int e = raw != nullptr ? std::atoi(raw) : 80;
    return e > 0 ? e : 80;
  }();
  return v;
}
double tmop_cooldown_t() {
  static const double v =
      tmop_env_double("TENRYU_I1B_TMOP_COOLDOWN_T", 1.0e-12);
  return v;
}

// Activation time gate (dyncore28: TMOP fires at 0.51 ns near the fragile
// early core triggered a full-capsule absorption cascade — the third
// early-intervention destabilization after equal-mu and the pole arm).
double tmop_start_t() {
  static const double v = tmop_env_double("TENRYU_I1B_TMOP_START_T", 0.0);
  return v;
}

struct TmopMetrics {
  double q_shape = 1.0;
  double q_j = 1.0;
  double q_edge = 1.0;
};

// Corner-sampled quality of a quad against the spherical-annular target.
// nodes: 4 corner positions in CCW logical order; W columns are
// (h_R e_R, h_theta e_theta) evaluated at the cell center.
TmopMetrics tmop_cell_quality(const double* xr,
                              const double* xz,
                              const double h_R,
                              const double h_t) {
  TmopMetrics out;
  const double rc = 0.25 * (xr[0] + xr[1] + xr[2] + xr[3]);
  const double zc = 0.25 * (xz[0] + xz[1] + xz[2] + xz[3]);
  const double s = std::hypot(rc, zc);
  const double er_r = s > 0.0 ? rc / s : 0.0;
  const double er_z = s > 0.0 ? zc / s : 1.0;
  // e_theta = rotate e_R by +90deg in the meridional plane
  const double et_r = er_z;
  const double et_z = -er_r;
  // Corner Jacobians with edges CLASSIFIED by logical direction (radial
  // edge oriented +i, angular edge oriented +j) and projected onto the
  // orthonormal (e_R, e_theta) pair. This makes the ideal spherical-annular
  // cell map to T = I with det T = +1 REGARDLESS of the pair's handedness
  // in (r,z) — the naive cyclic-corner T = A W^{-1} evaluated det T < 0 on
  // every healthy cell because (e_R, e_theta) is LEFT-handed when theta is
  // measured from +z (measured: q_shape identically 0 across two runs).
  // Node order: 0=(i,j) 1=(i,j+1) 2=(i+1,j+1) 3=(i+1,j).
  const double Rer[2] = {xr[3] - xr[0], xr[2] - xr[1]};
  const double Rez[2] = {xz[3] - xz[0], xz[2] - xz[1]};
  const double Ter[2] = {xr[1] - xr[0], xr[2] - xr[3]};
  const double Tez[2] = {xz[1] - xz[0], xz[2] - xz[3]};
  double min_edge = std::numeric_limits<double>::infinity();
  double max_edge = 0.0;
  for (int a = 0; a < 2; ++a) {
    const double elR = std::hypot(Rer[a], Rez[a]);
    const double elT = std::hypot(Ter[a], Tez[a]);
    min_edge = std::min({min_edge, elR, elT});
    max_edge = std::max({max_edge, elR, elT});
  }
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      const double t11 = (Rer[a] * er_r + Rez[a] * er_z) / h_R;
      const double t21 = (Rer[a] * et_r + Rez[a] * et_z) / h_t;
      const double t12 = (Ter[b] * er_r + Tez[b] * er_z) / h_R;
      const double t22 = (Ter[b] * et_r + Tez[b] * et_z) / h_t;
      const double detT = t11 * t22 - t12 * t21;
      out.q_j = std::min(out.q_j, detT);
      const double frob2 = t11 * t11 + t21 * t21 + t12 * t12 + t22 * t22;
      const double q_s =
          frob2 > 0.0 ? std::abs(2.0 * detT) / frob2 : 0.0;
      out.q_shape = std::min(out.q_shape, detT > 0.0 ? q_s : 0.0);
    }
  }
  out.q_edge = max_edge > 0.0 ? min_edge / max_edge : 0.0;
  return out;
}

double tmop_barrier(const double q, const double q_min) {
  if (!(q > q_min)) {
    return 1.0e6 * (1.0 + q_min - q);
  }
  return -std::log((q - q_min) / (1.0 - q_min));
}

struct TmopPatchResult {
  bool sampled = false;
  bool triggered = false;
  int worst_cell = -1;
  double worst_q_shape = 1.0;
  double worst_q_j = 1.0;
  double worst_q_edge = 1.0;
  int patch_cells = 0;
  int free_nodes = 0;
  int iters_used = 0;
  double phi0 = 0.0;
  double phi1 = 0.0;
};

// Scan the POLAR_SHELL for warn-level cells, build a BFS-window patch
// around the worst cell, optimize its free nodes and return a blended
// patch target. Metrics/decisions logged via the result struct.
axis_ale::AxisAlePatchTarget compute_tmop_patch_target(
    const core::State& state,
    const core::Config& cfg,
    const std::vector<double>& node_r_now,
    const std::vector<double>& node_z_now,
    const std::vector<double>& node_vr,
    const std::vector<double>& node_vz,
    const double dt_pred,
    TmopPatchResult& res) {
  (void)cfg;
  axis_ale::AxisAlePatchTarget out;
  if (!state.mesh.topo.multiblock.has_value()) {
    return out;
  }
  // Predicted-metric mode (verdict #6 B1 original spec; dyncore31: the
  // rebound fold lives in the VELOCITY field — the static scan reads the
  // failing cell as healthy while its straight-line path inverts). With
  // velocities supplied, the scan and the barrier optimization run on the
  // PREDICTED coordinates x + dt*v, and the resulting displacement is
  // applied to the CURRENT positions (v unchanged => the same displacement
  // un-folds the predicted state, to linear order).
  const bool predicted =
      dt_pred > 0.0 && node_vr.size() == node_r_now.size() &&
      node_vz.size() == node_z_now.size();
  std::vector<double> pred_r;
  std::vector<double> pred_z;
  if (predicted) {
    pred_r.resize(node_r_now.size());
    pred_z.resize(node_z_now.size());
    for (std::size_t i = 0; i < node_r_now.size(); ++i) {
      pred_r[i] = node_r_now[i] + dt_pred * node_vr[i];
      pred_z[i] = node_z_now[i] + dt_pred * node_vz[i];
    }
  }
  const std::vector<double>& node_r = predicted ? pred_r : node_r_now;
  const std::vector<double>& node_z = predicted ? pred_z : node_z_now;
  const auto& mb = *state.mesh.topo.multiblock;
  const mesh::BlockInfo* shell = nullptr;
  for (const auto& block : mb.blocks) {
    if (block.role == mesh::BlockRole::POLAR_SHELL) {
      shell = &block;
      break;
    }
  }
  if (shell == nullptr || shell->n_i_cells < 3 || shell->n_j_cells < 8) {
    return out;
  }
  const int n_i = shell->n_i_cells;
  const int n_j = shell->n_j_cells;
  const int n_cols = n_j + 1;
  const int n_nodes = static_cast<int>(node_r.size());
  const double dtheta = 3.14159265358979323846 / n_j;
  int q_begin = 0;
  if (state.central_pseudo_core.built && mb.has_trifan_cap) {
    const auto& north =
        mesh::mesh_topo_trifan_fan_block(mb, mesh::BlockRole::NORTH_FAN);
    q_begin = std::max(
        0, state.central_pseudo_core.member_ring_count - mb.n_cap -
               north.n_i_cells);
  }
  const auto node_id = [&](const int q, const int k) {
    return shell->owned_node_begin + q * n_cols + k;
  };
  const auto cell_nodes4 = [&](const int i, const int j, int* ids) {
    ids[0] = node_id(i, j);
    ids[1] = node_id(i, j + 1);
    ids[2] = node_id(i + 1, j + 1);
    ids[3] = node_id(i + 1, j);
  };
  // mean radius per node row for h_R
  std::vector<double> row_mean(static_cast<std::size_t>(n_i + 1), 0.0);
  for (int q = 0; q <= n_i; ++q) {
    double acc = 0.0;
    for (int k = 0; k < n_cols; ++k) {
      const int n = node_id(q, k);
      if (n < 0 || n >= n_nodes) {
        return out;
      }
      acc += std::hypot(node_r[static_cast<std::size_t>(n)],
                        node_z[static_cast<std::size_t>(n)]);
    }
    row_mean[static_cast<std::size_t>(q)] = acc / n_cols;
  }
  const auto cell_targets = [&](const int i, double* h_R, double* h_t) {
    const double rlo = row_mean[static_cast<std::size_t>(i)];
    const double rhi = row_mean[static_cast<std::size_t>(i + 1)];
    *h_R = std::max(rhi - rlo, 1.0e-30);
    *h_t = std::max(0.5 * (rlo + rhi) * dtheta, 1.0e-30);
  };
  // scan for the worst cell (active rings only)
  res.sampled = true;
  int worst_i = -1, worst_j = -1;
  double worst_score = std::numeric_limits<double>::infinity();
  // Pole-adjacent cell columns (j = 0, n_j-1) are structurally degenerate
  // quads (an axis edge) for which the annular W target is meaningless
  // (measured: q_shape = 0 on the HEALTHY initial mesh at cell 384, which
  // locked the scan onto the pole forever). Those columns belong to the
  // axis/BBSW machinery; scan and objective skip them.
  for (int i = std::max(q_begin, 0); i < n_i; ++i) {
    for (int j = 1; j < n_j - 1; ++j) {
      int ids[4];
      cell_nodes4(i, j, ids);
      double xr[4], xz[4];
      for (int k = 0; k < 4; ++k) {
        xr[k] = node_r[static_cast<std::size_t>(ids[k])];
        xz[k] = node_z[static_cast<std::size_t>(ids[k])];
      }
      double h_R, h_t;
      cell_targets(i, &h_R, &h_t);
      const TmopMetrics m = tmop_cell_quality(xr, xz, h_R, h_t);
      const double score =
          std::min({m.q_shape / tmop_q_shape_warn(),
                    m.q_j / tmop_q_j_warn(), m.q_edge / tmop_q_edge_warn()});
      if (score < worst_score) {
        worst_score = score;
        worst_i = i;
        worst_j = j;
        res.worst_cell = shell->cell_begin + i * n_j + j;
        res.worst_q_shape = m.q_shape;
        res.worst_q_j = m.q_j;
        res.worst_q_edge = m.q_edge;
      }
    }
  }
  if (worst_i < 0 || worst_score >= 1.0) {
    return out;  // res carries the measured worst for the force probe
  }
  res.triggered = true;
  // BFS-window patch |di|<=2, |dj|<=3 clipped to active rings
  const int i_lo = std::max(worst_i - 2, std::max(q_begin, 0));
  const int i_hi = std::min(worst_i + 2, n_i - 1);
  const int j_lo = std::max(worst_j - 3, 1);
  const int j_hi = std::min(worst_j + 3, n_j - 2);
  res.patch_cells = (i_hi - i_lo + 1) * (j_hi - j_lo + 1);
  // free nodes: interior of the patch, minus pinned classes
  std::vector<int> free_nodes;
  std::vector<std::uint8_t> is_free(static_cast<std::size_t>(n_nodes), 0U);
  for (int q = i_lo; q <= i_hi + 1; ++q) {
    if (q <= q_begin || q >= n_i) {
      continue;  // macro-boundary row and outer boundary row pinned
    }
    for (int k = j_lo; k <= j_hi + 1; ++k) {
      if (k <= 0 || k >= n_j) {
        continue;  // axis columns pinned
      }
      const int n = node_id(q, k);
      if (state.mesh.topo.node_flags.size() ==
          static_cast<std::size_t>(n_nodes)) {
        const auto flags =
            state.mesh.topo.node_flags[static_cast<std::size_t>(n)];
        if ((flags & (mesh::NODE_AXIS | mesh::NODE_CENTER |
                      mesh::NODE_OUTER_PHYSICAL_BOUNDARY)) != 0U) {
          continue;
        }
      }
      free_nodes.push_back(n);
      is_free[static_cast<std::size_t>(n)] = 1U;
    }
  }
  if (free_nodes.empty()) {
    return out;
  }
  res.free_nodes = static_cast<int>(free_nodes.size());
  // working copies
  std::vector<double> wr = node_r;
  std::vector<double> wz = node_z;
  const double a_J = tmop_env_double("TENRYU_I1B_TMOP_ALPHA_J", 10.0);
  const double a_s = tmop_env_double("TENRYU_I1B_TMOP_ALPHA_S", 3.0);
  const double a_e = tmop_env_double("TENRYU_I1B_TMOP_ALPHA_E", 3.0);
  const double a_M = tmop_env_double("TENRYU_I1B_TMOP_ALPHA_M", 0.05);
  const auto phi = [&]() {
    double acc = 0.0;
    for (int i = std::max(i_lo - 1, std::max(q_begin, 0));
         i <= std::min(i_hi + 1, n_i - 1); ++i) {
      for (int j = std::max(j_lo - 1, 1); j <= std::min(j_hi + 1, n_j - 2);
           ++j) {
        int ids[4];
        cell_nodes4(i, j, ids);
        double xr[4], xz[4];
        for (int k = 0; k < 4; ++k) {
          xr[k] = wr[static_cast<std::size_t>(ids[k])];
          xz[k] = wz[static_cast<std::size_t>(ids[k])];
        }
        double h_R, h_t;
        cell_targets(i, &h_R, &h_t);
        const TmopMetrics m = tmop_cell_quality(xr, xz, h_R, h_t);
        acc += a_s * tmop_barrier(m.q_shape, 0.02) +
               a_J * tmop_barrier(m.q_j, 0.02) +
               a_e * tmop_barrier(m.q_edge, 0.02);
      }
    }
    for (const int n : free_nodes) {
      const std::size_t ni = static_cast<std::size_t>(n);
      const double dr = wr[ni] - node_r[ni];
      const double dz = wz[ni] - node_z[ni];
      const double h = std::max(row_mean[1] - row_mean[0], 1.0e-30);
      acc += a_M * (dr * dr + dz * dz) / (h * h);
    }
    return acc;
  };
  res.phi0 = phi();
  double step = 0.02 * (row_mean[1] - row_mean[0]);
  double phi_cur = res.phi0;
  for (int it = 0; it < tmop_iters(); ++it) {
    bool improved = false;
    for (const int n : free_nodes) {
      const std::size_t ni = static_cast<std::size_t>(n);
      const double eps = 1.0e-7 * std::max(row_mean[1], 1.0e-30);
      const double r0 = wr[ni], z0 = wz[ni];
      wr[ni] = r0 + eps;
      const double fpr = phi();
      wr[ni] = r0 - eps;
      const double fmr = phi();
      wr[ni] = r0;
      wz[ni] = z0 + eps;
      const double fpz = phi();
      wz[ni] = z0 - eps;
      const double fmz = phi();
      wz[ni] = z0;
      double gr = (fpr - fmr) / (2.0 * eps);
      double gz = (fpz - fmz) / (2.0 * eps);
      const double gn = std::hypot(gr, gz);
      if (!(gn > 0.0) || !std::isfinite(gn)) {
        continue;
      }
      gr /= gn;
      gz /= gn;
      double local_step = step;
      for (int ls = 0; ls < 6; ++ls) {
        wr[ni] = std::max(r0 - local_step * gr, 0.0);
        wz[ni] = z0 - local_step * gz;
        const double f_new = phi();
        if (f_new < phi_cur) {
          phi_cur = f_new;
          improved = true;
          break;
        }
        wr[ni] = r0;
        wz[ni] = z0;
        local_step *= 0.5;
      }
    }
    res.iters_used = it + 1;
    if (!improved) {
      break;
    }
  }
  res.phi1 = phi_cur;
  if (!(phi_cur < res.phi0)) {
    return out;
  }
  for (const int n : free_nodes) {
    const std::size_t ni = static_cast<std::size_t>(n);
    if (wr[ni] == node_r[ni] && wz[ni] == node_z[ni]) {
      continue;
    }
    out.node_ids.push_back(n);
    // In predicted mode the optimizer displacement (computed on x + dt*v)
    // is applied to the CURRENT positions.
    out.r_target.push_back(
        std::max(node_r_now[ni] + (wr[ni] - node_r[ni]), 0.0));
    out.z_target.push_back(node_z_now[ni] + (wz[ni] - node_z[ni]));
    out.beta.push_back(1.0);
  }
  out.active = !out.node_ids.empty();
  out.patch_nodes = static_cast<int>(out.node_ids.size());
  return out;
}

// Pole-shear radial mode-amplitude instrumentation (pole-shear verdict Q2):
// every N steps log, per pole, the first-off-axis-column pole angle
// deviation B_q = (delta_{q,1} - dtheta)/dtheta and its radial second
// difference H_q = (delta_{q+1,1} - 2 delta_{q,1} + delta_{q-1,1})/dtheta
// over all shell node rows q (H spans q = 1..nq-2). Rows q < q_begin are
// central-macro-cell members whose printed angles are stale. Consumes the
// host node download the axis-rezone path already performs; inert unless
// TENRYU_I1B_POLE_SHEAR_DIAG_EVERY > 0.
int pole_shear_diag_every() {
  static const int every = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_SHEAR_DIAG_EVERY");
    const int v = raw != nullptr ? std::atoi(raw) : 0;
    return v > 0 ? v : 0;
  }();
  return every;
}

// Stage-resolved sampling (macro-boundary endgame verdict Q2 #1 / Exp 3):
// with TENRYU_I1B_POLE_SHEAR_DIAG_STAGES=1 the H_q diagnostic is also
// sampled after the rezone target is finalized (stage R, reference mesh)
// and after the conservative remap commits (stage A), in addition to the
// existing post-Lagrange sample (stage L). Which stage the dense-band H_q
// jumps at discriminates remap-pumped vs Lagrangian growth.
bool pole_shear_diag_stages_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_SHEAR_DIAG_STAGES");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  return enabled;
}

void log_pole_shear_diag(const core::State& state,
                         const std::vector<double>& node_r,
                         const std::vector<double>& node_z,
                         const double t_post,
                         const char stage) {
  const int every = pole_shear_diag_every();
  if (every <= 0 || ((state.step + 1) % every) != 0) {
    return;
  }
  // Same-step dt retries re-enter each sampling site; log once per step per
  // stage (L=0, R=1, A=2).
  const int stage_idx = stage == 'R' ? 1 : (stage == 'A' ? 2 : 0);
  static int last_logged_step[3] = {-1, -1, -1};
  if (state.step == last_logged_step[stage_idx]) {
    return;
  }
  if (!state.mesh.topo.multiblock.has_value()) {
    return;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const mesh::BlockInfo* shell = nullptr;
  for (const auto& block : mb.blocks) {
    if (block.role == mesh::BlockRole::POLAR_SHELL) {
      shell = &block;
      break;
    }
  }
  if (shell == nullptr || shell->n_j_cells < 4 || shell->n_i_cells < 2) {
    return;
  }
  const int ntheta = shell->n_j_cells;
  const double dtheta = 3.14159265358979323846 / static_cast<double>(ntheta);
  int q_begin = 0;
  if (state.central_pseudo_core.built && mb.has_trifan_cap) {
    const auto& north =
        mesh::mesh_topo_trifan_fan_block(mb, mesh::BlockRole::NORTH_FAN);
    q_begin = std::max(0,
                       state.central_pseudo_core.member_ring_count - mb.n_cap -
                           north.n_i_cells);
  }
  const int n_nodes = static_cast<int>(node_r.size());
  const int n_rows = shell->n_i_cells + 1;
  last_logged_step[stage_idx] = state.step;
  for (int pole = 0; pole < 2; ++pole) {
    std::vector<double> delta(static_cast<std::size_t>(n_rows),
                              std::numeric_limits<double>::quiet_NaN());
    const int k1 = pole == 0 ? 1 : ntheta - 1;
    for (int q = 0; q < n_rows; ++q) {
      const int node = shell->owned_node_begin + q * (ntheta + 1) + k1;
      if (node < 0 || node >= n_nodes) {
        continue;
      }
      const double r = node_r[static_cast<std::size_t>(node)];
      const double z_signed = pole == 0
                                  ? node_z[static_cast<std::size_t>(node)]
                                  : -node_z[static_cast<std::size_t>(node)];
      delta[static_cast<std::size_t>(q)] = std::atan2(r, z_signed);
    }
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(3);
    oss << "[pole_shear_diag] stage=" << stage << " step=" << (state.step + 1)
        << " t=" << t_post
        << " pole=" << (pole == 0 ? 'N' : 'S') << " q_begin=" << q_begin
        << " nq=" << n_rows << " B=";
    for (int q = 0; q < n_rows; ++q) {
      oss << ' ' << (delta[static_cast<std::size_t>(q)] - dtheta) / dtheta;
    }
    oss << " H=";
    for (int q = 1; q + 1 < n_rows; ++q) {
      oss << ' '
          << (delta[static_cast<std::size_t>(q + 1)] -
              2.0 * delta[static_cast<std::size_t>(q)] +
              delta[static_cast<std::size_t>(q - 1)]) /
                 dtheta;
    }
    core::log_info(oss.str());
  }
}

// Stage R/A wrapper: downloads the requested node coordinates only when the
// stages mode is armed AND this is a sampled step, then forwards to
// log_pole_shear_diag. Stage R samples the finalized rezone reference
// (x_*_reference); stage A samples the committed post-remap mesh (x_*).
void log_pole_shear_stage_sample(const core::State& state,
                                 const bool use_reference,
                                 const char stage,
                                 const double t_post) {
  if (!pole_shear_diag_stages_enabled()) {
    return;
  }
  const int every = pole_shear_diag_every();
  if (every <= 0 || ((state.step + 1) % every) != 0) {
    return;
  }
  std::vector<double> r_h;
  std::vector<double> z_h;
  if (use_reference) {
    state.x_r_reference.copy_to_host(r_h);
    state.x_z_reference.copy_to_host(z_h);
  } else {
    state.x_r.copy_to_host(r_h);
    state.x_z.copy_to_host(z_h);
  }
  log_pole_shear_diag(state, r_h, z_h, t_post, stage);
}

// Macro-boundary loop diagnostics (macro-boundary endgame verdict Q4 #3/#4):
// every N steps (TENRYU_I1B_BOUNDARY_DIAG_EVERY, default 0 = off) log
//  (1) the boundary loop's own high-frequency metric: theta_i = atan2(r,z)
//      along boundary_nodes_ordered, H_i = second difference of theta
//      normalized by the mean spacing, plus a DCT split of the
//      endpoint-detrended theta sequence into low (l<=4) and high (l>4)
//      spectral energy;
//  (2) health of the active cells immediately outside the macro (cells not
//      in member_mask touching a boundary node): min cell volume, min
//      orientation-normalized corner-Jacobian ratio
//      min_k(s*J_k)/max_k|J_k| with J_k = (x_{k+1}-x_k) x (x_{k+3}-x_k) and
//      s the sign of sum_k J_k, max sound speed, min angular width.
// Host-only; the per-cell volume and sound-speed downloads happen only on
// sampled steps. Inert unless the env is set.
int boundary_diag_every() {
  static const int every = [] {
    const char* raw = std::getenv("TENRYU_I1B_BOUNDARY_DIAG_EVERY");
    const int v = raw != nullptr ? std::atoi(raw) : 0;
    return v > 0 ? v : 0;
  }();
  return every;
}

void log_macro_boundary_diag(const core::State& state,
                             const core::Config& cfg,
                             const std::vector<double>& node_r,
                             const std::vector<double>& node_z,
                             const double t_post) {
  const int every = boundary_diag_every();
  if (every <= 0 || ((state.step + 1) % every) != 0) {
    return;
  }
  static int last_logged_step = -1;
  if (state.step == last_logged_step) {
    return;
  }
  const auto& pc = state.central_pseudo_core;
  const int nb = static_cast<int>(pc.boundary_nodes_ordered.size());
  const int n_nodes = static_cast<int>(node_r.size());
  if (!pc.built || nb < 5) {
    return;
  }
  last_logged_step = state.step;

  std::vector<double> theta(static_cast<std::size_t>(nb));
  for (int i = 0; i < nb; ++i) {
    const int n = pc.boundary_nodes_ordered[static_cast<std::size_t>(i)];
    if (n < 0 || n >= n_nodes) {
      theta[static_cast<std::size_t>(i)] =
          std::numeric_limits<double>::quiet_NaN();
      continue;
    }
    theta[static_cast<std::size_t>(i)] =
        std::atan2(node_r[static_cast<std::size_t>(n)],
                   node_z[static_cast<std::size_t>(n)]);
  }
  const double span = theta[static_cast<std::size_t>(nb - 1)] - theta[0];
  const double dtheta_mean =
      std::abs(span) / static_cast<double>(nb - 1);
  double h_max = 0.0;
  int h_max_i = -1;
  if (dtheta_mean > 0.0 && std::isfinite(dtheta_mean)) {
    for (int i = 1; i + 1 < nb; ++i) {
      const double h = (theta[static_cast<std::size_t>(i + 1)] -
                        2.0 * theta[static_cast<std::size_t>(i)] +
                        theta[static_cast<std::size_t>(i - 1)]) /
                       dtheta_mean;
      if (std::isfinite(h) && std::abs(h) > std::abs(h_max)) {
        h_max = h;
        h_max_i = i;
      }
    }
  }
  // DCT-II spectral split of the endpoint-detrended theta sequence.
  double e_low = 0.0;
  double e_hi = 0.0;
  {
    std::vector<double> resid(static_cast<std::size_t>(nb));
    for (int i = 0; i < nb; ++i) {
      const double ramp =
          theta[0] + span * static_cast<double>(i) /
                         static_cast<double>(nb - 1);
      const double v = theta[static_cast<std::size_t>(i)] - ramp;
      resid[static_cast<std::size_t>(i)] = std::isfinite(v) ? v : 0.0;
    }
    const double pi = 3.14159265358979323846;
    for (int l = 1; l < nb; ++l) {
      double c = 0.0;
      for (int i = 0; i < nb; ++i) {
        c += resid[static_cast<std::size_t>(i)] *
             std::cos(pi * static_cast<double>(l) *
                      (static_cast<double>(i) + 0.5) /
                      static_cast<double>(nb));
      }
      const double e = c * c;
      if (l <= 4) {
        e_low += e;
      } else {
        e_hi += e;
      }
    }
  }
  {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(3);
    oss << "[macro_boundary_diag] step=" << (state.step + 1)
        << " t=" << t_post << " nb=" << nb << " Vc=" << pc.V_c
        << " dtheta_mean=" << dtheta_mean << " Hmax=" << h_max
        << " Hmax_i=" << h_max_i << " Hmax_theta="
        << (h_max_i >= 0 ? theta[static_cast<std::size_t>(h_max_i)]
                         : std::numeric_limits<double>::quiet_NaN())
        << " E_low=" << e_low << " E_hi=" << e_hi;
    core::log_info(oss.str());
  }

  // Boundary-adjacent active cell health.
  const int n_cells = state.mesh.topo.n_cells;
  if (static_cast<int>(pc.member_mask.size()) != n_cells ||
      static_cast<int>(pc.boundary_node_mask.size()) != n_nodes) {
    return;
  }
  std::vector<double> vol_h;
  std::vector<double> cs_h;
  state.vol.copy_to_host(vol_h);
  state.cs.copy_to_host(cs_h);
  int n_adj = 0;
  double min_v = std::numeric_limits<double>::infinity();
  int min_v_cell = -1;
  double min_jrel = std::numeric_limits<double>::infinity();
  int min_jrel_cell = -1;
  double max_cs = 0.0;
  int max_cs_cell = -1;
  double min_dth = std::numeric_limits<double>::infinity();
  for (int c = 0; c < n_cells; ++c) {
    if (pc.member_mask[static_cast<std::size_t>(c)] != 0) {
      continue;
    }
    const auto nodes = mesh::mesh_topo_cell_corner_nodes_n(
        state.mesh.topo, state.mesh.cell_nverts, c, cfg.mesh);
    bool touches = false;
    for (int k = 0; k < nodes.count; ++k) {
      const int n = nodes.values[static_cast<std::size_t>(k)];
      if (n >= 0 && n < n_nodes &&
          pc.boundary_node_mask[static_cast<std::size_t>(n)] != 0) {
        touches = true;
        break;
      }
    }
    if (!touches) {
      continue;
    }
    ++n_adj;
    const double v = vol_h[static_cast<std::size_t>(c)];
    if (v < min_v) {
      min_v = v;
      min_v_cell = c;
    }
    const double cs_c = cs_h[static_cast<std::size_t>(c)];
    if (cs_c > max_cs) {
      max_cs = cs_c;
      max_cs_cell = c;
    }
    double j[4];
    bool corner_real[4];
    double j_sum = 0.0;
    double j_absmax = 0.0;
    double th_min = std::numeric_limits<double>::infinity();
    double th_max = -std::numeric_limits<double>::infinity();
    bool geom_ok = true;
    for (int k = 0; k < nodes.count; ++k) {
      const int na = nodes.values[static_cast<std::size_t>(k)];
      const int nbn =
          nodes.values[static_cast<std::size_t>((k + 1) % nodes.count)];
      const int nd = nodes.values[static_cast<std::size_t>(
          (k + nodes.count - 1) % nodes.count)];
      if (na < 0 || na >= n_nodes || nbn < 0 || nbn >= n_nodes || nd < 0 ||
          nd >= n_nodes) {
        geom_ok = false;
        break;
      }
      const double ra = node_r[static_cast<std::size_t>(na)];
      const double za = node_z[static_cast<std::size_t>(na)];
      const double er = node_r[static_cast<std::size_t>(nbn)] - ra;
      const double ez = node_z[static_cast<std::size_t>(nbn)] - za;
      const double fr = node_r[static_cast<std::size_t>(nd)] - ra;
      const double fz = node_z[static_cast<std::size_t>(nd)] - za;
      j[k] = er * fz - ez * fr;
      // Trifan cells arrive as degenerate quads (one duplicated corner);
      // a corner with a zero-length incident edge is topological, not a
      // fold -- exclude it from the min.
      corner_real[k] =
          (er != 0.0 || ez != 0.0) && (fr != 0.0 || fz != 0.0);
      j_sum += j[k];
      if (corner_real[k]) {
        j_absmax = std::max(j_absmax, std::abs(j[k]));
      }
      const double th = std::atan2(ra, za);
      th_min = std::min(th_min, th);
      th_max = std::max(th_max, th);
    }
    if (geom_ok && j_absmax > 0.0) {
      const double s = j_sum >= 0.0 ? 1.0 : -1.0;
      double jr = std::numeric_limits<double>::infinity();
      for (int k = 0; k < 4; ++k) {
        if (corner_real[k]) {
          jr = std::min(jr, s * j[k] / j_absmax);
        }
      }
      if (jr < min_jrel) {
        min_jrel = jr;
        min_jrel_cell = c;
      }
    }
    if (geom_ok) {
      min_dth = std::min(min_dth, th_max - th_min);
    }
  }
  {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(3);
    oss << "[macro_boundary_adjacent] step=" << (state.step + 1)
        << " t=" << t_post << " n_adj=" << n_adj << " minV=" << min_v
        << " minV_cell=" << min_v_cell << " minJrel=" << min_jrel
        << " minJrel_cell=" << min_jrel_cell << " maxcs=" << max_cs
        << " maxcs_cell=" << max_cs_cell << " min_dtheta=" << min_dth;
    core::log_info(oss.str());
  }
}

struct DepositAuditReduction {
  double sum_dI_raw = 0.0;
  double sum_dI_after_floor = 0.0;
  double min_tentative_e_e = 0.0;
  double min_tentative_e_i = 0.0;
  int n_cells_negative_dI = 0;
  int n_cells_floor_e = 0;
  int n_cells_floor_i = 0;
};

DepositAuditReduction reduce_deposit_audit(const double* d_values, const int n_cells) {
  DepositAuditReduction out;
  if (d_values == nullptr || n_cells <= 0) {
    return out;
  }

  std::vector<double> host(static_cast<std::size_t>(n_cells) *
                               static_cast<std::size_t>(kAuditDepositCols),
                           0.0);
  CUDA_CHECK(cudaMemcpy(host.data(),
                        d_values,
                        host.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
  long double sum_raw = 0.0L;
  long double sum_after = 0.0L;
  double min_ee = std::numeric_limits<double>::infinity();
  double min_ei = std::numeric_limits<double>::infinity();
  long double n_neg = 0.0L;
  long double n_floor_e = 0.0L;
  long double n_floor_i = 0.0L;
  for (int c = 0; c < n_cells; ++c) {
    const int o = c * kAuditDepositCols;
    sum_raw += static_cast<long double>(host[static_cast<std::size_t>(o + 0)]);
    sum_after += static_cast<long double>(host[static_cast<std::size_t>(o + 1)]);
    min_ee = std::min(min_ee, host[static_cast<std::size_t>(o + 2)]);
    min_ei = std::min(min_ei, host[static_cast<std::size_t>(o + 3)]);
    n_neg += static_cast<long double>(host[static_cast<std::size_t>(o + 4)]);
    n_floor_e += static_cast<long double>(host[static_cast<std::size_t>(o + 5)]);
    n_floor_i += static_cast<long double>(host[static_cast<std::size_t>(o + 6)]);
  }

  out.sum_dI_raw = static_cast<double>(sum_raw);
  out.sum_dI_after_floor = static_cast<double>(sum_after);
  out.min_tentative_e_e = std::isfinite(min_ee) ? min_ee : 0.0;
  out.min_tentative_e_i = std::isfinite(min_ei) ? min_ei : 0.0;
  out.n_cells_negative_dI = static_cast<int>(std::llround(n_neg));
  out.n_cells_floor_e = static_cast<int>(std::llround(n_floor_e));
  out.n_cells_floor_i = static_cast<int>(std::llround(n_floor_i));
  return out;
}

void reduce_closure_audit_global(diagnostics::AleClosureAuditDiagnostics& audit,
                                 const parallel::Reduction* reduction) {
  if (reduction == nullptr || !audit.valid) {
    return;
  }

  double sums[17] = {
      audit.K0_cellcorner,
      audit.K0_node_from_corner,
      audit.K0_budget,
      audit.I0,
      audit.K0_scalar_total,
      audit.K_remap_total,
      audit.I_raw,
      audit.K_cellmom,
      audit.K_node_preBC,
      audit.K_node_postBC,
      audit.K_post_budget,
      audit.sum_dI_raw,
      audit.sum_dI_after_floor,
      audit.dE_ale_total,
      static_cast<double>(audit.n_cells_negative_dI),
      static_cast<double>(audit.n_cells_floor_e),
      static_cast<double>(audit.n_cells_floor_i)};
  reduction->allreduce_sum(sums, 17);
  audit.K0_cellcorner = sums[0];
  audit.K0_node_from_corner = sums[1];
  audit.K0_budget = sums[2];
  audit.I0 = sums[3];
  audit.K0_scalar_total = sums[4];
  audit.K_remap_total = sums[5];
  audit.I_raw = sums[6];
  audit.K_cellmom = sums[7];
  audit.K_node_preBC = sums[8];
  audit.K_node_postBC = sums[9];
  audit.K_post_budget = sums[10];
  audit.sum_dI_raw = sums[11];
  audit.sum_dI_after_floor = sums[12];
  audit.dE_ale_total = sums[13];
  audit.n_cells_negative_dI = static_cast<int>(std::llround(sums[14]));
  audit.n_cells_floor_e = static_cast<int>(std::llround(sums[15]));
  audit.n_cells_floor_i = static_cast<int>(std::llround(sums[16]));
  audit.min_tentative_e_e = reduction->allreduce_min(audit.min_tentative_e_e);
  audit.min_tentative_e_i = reduction->allreduce_min(audit.min_tentative_e_i);
}

}  // namespace detail

namespace detail {

void reduce_rezone_stats(AleRezoneIterStats& stats, const parallel::Reduction* reduction) {
  if (reduction == nullptr) {
    return;
  }
  double values[8] = {
      static_cast<double>(stats.j_floor_i1_fallback_hits),
      static_cast<double>(stats.j_floor_ix_skip_hits),
      static_cast<double>(stats.regular_cap_i1_hits),
      static_cast<double>(stats.regular_cap_ix_hits),
      static_cast<double>(stats.fallback_cap_hits),
      static_cast<double>(stats.zero_or_tiny_motion_hits),
      static_cast<double>(stats.local_linesearch_rejects),
      static_cast<double>(stats.weighted_laplacian_fallback_hits)};
  reduction->allreduce_sum(values, 8);
  stats.j_floor_i1_fallback_hits = static_cast<int>(values[0]);
  stats.j_floor_ix_skip_hits = static_cast<int>(values[1]);
  stats.regular_cap_i1_hits = static_cast<int>(values[2]);
  stats.regular_cap_ix_hits = static_cast<int>(values[3]);
  stats.fallback_cap_hits = static_cast<int>(values[4]);
  stats.zero_or_tiny_motion_hits = static_cast<int>(values[5]);
  stats.local_linesearch_rejects = static_cast<int>(values[6]);
  stats.weighted_laplacian_fallback_hits = static_cast<int>(values[7]);
  double max_values[3] = {
      stats.max_proposed_displacement,
      stats.max_displacement_cap,
      stats.max_proposed_over_cap};
  reduction->allreduce_max(max_values, 3);
  stats.max_proposed_displacement = max_values[0];
  stats.max_displacement_cap = max_values[1];
  stats.max_proposed_over_cap = max_values[2];
}

}  // namespace detail

void reset_safe_backtrack_lambda_distribution() {
  for (auto& count : detail::g_safe_backtrack_lambda_distribution) {
    count.store(0, std::memory_order_relaxed);
  }
  detail::g_safe_backtrack_distribution_max_exp.store(0, std::memory_order_relaxed);
}

void log_safe_backtrack_lambda_distribution(const int max_exp) {
  const int requested_max = detail::clamp_safe_backtrack_max_exp(max_exp);
  const int observed_max =
      detail::g_safe_backtrack_distribution_max_exp.load(std::memory_order_relaxed);
  const int max_bin = std::max(requested_max, observed_max);
  std::uint64_t total = 0;
  std::ostringstream os;
  os << "[ale-stats] safe_backtrack_lambda_distribution total=";
  for (int bin = 0; bin <= max_bin; ++bin) {
    total += detail::g_safe_backtrack_lambda_distribution[static_cast<std::size_t>(bin)]
                 .load(std::memory_order_relaxed);
  }
  os << total << " max_exp=" << max_bin;
  for (int bin = 0; bin <= max_bin; ++bin) {
    os << " bin" << bin << "="
       << detail::g_safe_backtrack_lambda_distribution[static_cast<std::size_t>(bin)]
              .load(std::memory_order_relaxed);
  }
  core::log_warning(os.str());
}

namespace detail {

bool shell_rezone_env_enabled(const char* const name) {
  const char* const raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return false;
  }
  const std::string value(raw);
  return value != "0" && value != "false" && value != "FALSE" &&
         value != "off" && value != "OFF" && value != "no" &&
         value != "NO";
}

int shell_rezone_env_int(const char* const name, const int fallback) {
  const char* const raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const long value = std::strtol(raw, &end, 10);
  return end == raw ? fallback : static_cast<int>(value);
}

double shell_rezone_env_double(const char* const name,
                               const double fallback) {
  const char* const raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const double value = std::strtod(raw, &end);
  return end == raw || !std::isfinite(value) ? fallback : value;
}

std::string shell_rezone_env_string(const char* const name,
                                    const char* const fallback) {
  const char* const raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  return std::string(raw);
}

int shell_rezone_patch_layers() {
  return std::clamp(
      shell_rezone_env_int("TENRYU_I1B_SHELL_REZONE_PATCH_LAYERS", 3), 2, 4);
}

double shell_rezone_q_warn_default() {
  const double mesh_warn = shell_rezone_env_double(
      "TENRYU_I1B_MESH_FORECAST_Q_WARN", 5.0e-2);
  return std::max(1.0e-12,
                  shell_rezone_env_double("TENRYU_I1B_SHELL_REZONE_Q_WARN",
                                          mesh_warn));
}

double shell_rezone_q_release(const double q_warn, const double q_floor) {
  const double fallback = std::max(0.10, 2.0 * q_warn);
  return std::max(q_floor, shell_rezone_env_double(
                               "TENRYU_I1B_SHELL_REZONE_Q_RELEASE", fallback));
}

double shell_rezone_q_stop(const double q_release) {
  const double reserve = std::clamp(
      shell_rezone_env_double("TENRYU_I1B_SHELL_REZONE_Q_RESERVE", 1.25),
      1.0,
      2.0);
  return std::max(q_release, std::min(1.0, q_release * reserve));
}

bool shell_rezone_replay_stage_selected(const char* const stage) {
  const std::string want = shell_rezone_env_string(
      "TENRYU_I1B_SHELL_REZONE_REPLAY_STAGE", "post_corrector_commit");
  return want == "*" || want == (stage != nullptr ? stage : "");
}

template <typename Field>
std::vector<double> copy_double_field_to_host(const Field& field) {
  std::vector<double> host(field.size(), 0.0);
  if (!host.empty()) {
    field.copy_to_host(host.data());
  }
  return host;
}

struct ShellSubcyclePostRezoneAudit {
  bool passed = false;
  std::string reason = "not_run";
  double cfl_dt = std::numeric_limits<double>::infinity();
  double cfl_dt_before = std::numeric_limits<double>::infinity();
  double max_node_speed = 0.0;
  double max_node_speed_before = 0.0;
  double min_node_mass = std::numeric_limits<double>::infinity();
  int first_bad_cell = -1;
  int first_bad_node = -1;
};

bool shell_subcycle_cell_active(const core::State& state, const int c) {
  const std::size_t idx = static_cast<std::size_t>(c);
  if (idx < state.cell_is_void.size() && state.cell_is_void[idx] != 0U) {
    return false;
  }
  if (idx < state.hydro_active.size() && state.hydro_active[idx] == 0) {
    return false;
  }
  return true;
}

ShellSubcyclePostRezoneAudit audit_shell_subcycle_post_rezone(
    const core::State& state,
    const core::Config& cfg,
    const parallel::Reduction* reduction,
    const ShellSubcyclePostRezoneAudit* before = nullptr) {
  ShellSubcyclePostRezoneAudit audit{};
  audit.passed = true;
  audit.reason = "ok";
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  const auto rho = copy_double_field_to_host(state.rho);
  const auto mass = copy_double_field_to_host(state.mass);
  const auto vol = copy_double_field_to_host(state.vol);
  const auto corner_mass = copy_double_field_to_host(state.corner_mass);
  const auto v_r = copy_double_field_to_host(state.v_r);
  const auto v_z = copy_double_field_to_host(state.v_z);
  const bool bad_sizes =
      n_cells < 0 || n_nodes < 0 ||
      rho.size() < static_cast<std::size_t>(n_cells) ||
      mass.size() < static_cast<std::size_t>(n_cells) ||
      vol.size() < static_cast<std::size_t>(n_cells) ||
      corner_mass.size() < static_cast<std::size_t>(4 * n_cells) ||
      v_r.size() < static_cast<std::size_t>(n_nodes) ||
      v_z.size() < static_cast<std::size_t>(n_nodes);
  bool local_failed = bad_sizes;
  if (bad_sizes) {
    audit.reason = "bad_field_sizes";
  }
  for (int c = 0; !bad_sizes && c < n_cells; ++c) {
    if (!shell_subcycle_cell_active(state, c)) {
      continue;
    }
    const std::size_t idx = static_cast<std::size_t>(c);
    if (!std::isfinite(mass[idx]) || mass[idx] <= 0.0 ||
        !std::isfinite(rho[idx]) || rho[idx] <= 0.0 ||
        !std::isfinite(vol[idx]) || vol[idx] <= 0.0) {
      local_failed = true;
      audit.reason = "cell_mass_positivity";
      audit.first_bad_cell = c;
      break;
    }
  }

  std::vector<double> node_mass(static_cast<std::size_t>(std::max(n_nodes, 0)),
                                0.0);
  if (!bad_sizes) {
    if (state.mesh.topo.multiblock.has_value()) {
      const auto& mb = *state.mesh.topo.multiblock;
      const bool csr_ok =
          mb.cell_node_csr_offsets.size() >=
          static_cast<std::size_t>(n_cells + 1);
      if (!csr_ok) {
        local_failed = true;
        audit.reason = "bad_cell_node_csr";
      } else {
        for (int c = 0; c < n_cells; ++c) {
          if (!shell_subcycle_cell_active(state, c)) {
            continue;
          }
          const int nverts =
              tenryu::mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts,
                                                         c);
          const int off =
              mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
          const int end =
              mb.cell_node_csr_offsets[static_cast<std::size_t>(c + 1)];
          for (int k = 0; k < nverts && off + k < end && k < 4; ++k) {
            const int node =
                mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
            if (node >= 0 && node < n_nodes) {
              node_mass[static_cast<std::size_t>(node)] +=
                  corner_mass[static_cast<std::size_t>(4 * c + k)];
            }
          }
        }
      }
    } else if (state.mesh.topo.nr > 0 && state.mesh.topo.nz > 0) {
      for (int i = 0; i < state.mesh.topo.nr; ++i) {
        for (int j = 0; j < state.mesh.topo.nz; ++j) {
          const int c = state.mesh.topo.cell_index(i, j);
          if (!shell_subcycle_cell_active(state, c)) {
            continue;
          }
          const int nodes[4] = {
              state.mesh.topo.node_index(i, j),
              state.mesh.topo.node_index(i + 1, j),
              state.mesh.topo.node_index(i + 1, j + 1),
              state.mesh.topo.node_index(i, j + 1),
          };
          for (int k = 0; k < 4; ++k) {
            if (nodes[k] >= 0 && nodes[k] < n_nodes) {
              node_mass[static_cast<std::size_t>(nodes[k])] +=
                  corner_mass[static_cast<std::size_t>(4 * c + k)];
            }
          }
        }
      }
    }
  }

  const double node_mass_floor = std::max(
      0.0, shell_rezone_env_double(
               "TENRYU_I1B_SHELL_SUBCYCLE_NODE_MASS_FLOOR_G", 0.0));
  for (int n = 0; !bad_sizes && n < n_nodes; ++n) {
    const std::size_t idx = static_cast<std::size_t>(n);
    const double vr = v_r[idx];
    const double vz = v_z[idx];
    const double speed = std::hypot(vr, vz);
    audit.max_node_speed = std::max(audit.max_node_speed, speed);
    audit.min_node_mass = std::min(audit.min_node_mass, node_mass[idx]);
    if (!std::isfinite(speed) || !std::isfinite(node_mass[idx]) ||
        node_mass[idx] < 0.0 ||
        (node_mass_floor > 0.0 && node_mass[idx] < node_mass_floor)) {
      local_failed = true;
      if (!std::isfinite(speed)) {
        audit.reason = "node_velocity";
      } else if (!std::isfinite(node_mass[idx])) {
        audit.reason = "node_mass_nonfinite";
      } else {
        audit.reason =
            node_mass[idx] < 0.0 ? "node_mass_negative" : "node_mass_floor";
      }
      audit.first_bad_node = n;
      break;
    }
  }

  const tenryu::hydro::HydroDtDiagnostics cfl =
      tenryu::hydro::compute_dt_hydro_diagnostics(state, cfg);
  audit.cfl_dt = cfl.dt;
  if (!std::isfinite(audit.cfl_dt)) {
    local_failed = true;
    audit.reason = "cfl_nonfinite";
  }
  if (reduction != nullptr) {
    audit.cfl_dt = reduction->allreduce_min(audit.cfl_dt);
    audit.max_node_speed = reduction->allreduce_max(audit.max_node_speed);
    audit.min_node_mass = reduction->allreduce_min(audit.min_node_mass);
    local_failed =
        reduction->allreduce_max(local_failed ? 1.0 : 0.0) > 0.5;
  }
  if (before != nullptr && !local_failed) {
    audit.cfl_dt_before = before->cfl_dt;
    audit.max_node_speed_before = before->max_node_speed;
    const double tol = std::max(
        0.0, shell_rezone_env_double(
                 "TENRYU_I1B_SHELL_SUBCYCLE_POST_REZONE_REL_TOL", 3.0e-2));
    if (std::isfinite(before->cfl_dt) && std::isfinite(audit.cfl_dt) &&
        audit.cfl_dt < (1.0 - tol) * before->cfl_dt) {
      local_failed = true;
      audit.reason = "cfl_worsened";
    } else if (std::isfinite(before->max_node_speed) &&
               std::isfinite(audit.max_node_speed) &&
               audit.max_node_speed > (1.0 + tol) * before->max_node_speed) {
      local_failed = true;
      audit.reason = "node_velocity_worsened";
    }
  }
  audit.passed = !local_failed;
  return audit;
}

void refresh_shell_subcycle_void_mask_after_rezone(core::State& state,
                                                   const core::Config& cfg) {
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_mat = cfg.materials.materials.size();
  TENRYU_ASSERT(state.cell_is_void.size() == n_cells,
                "Shell subcycle void mask update requires cell_is_void/rho size match");
  TENRYU_ASSERT(n_mat > 0,
                "Shell subcycle void mask update requires at least one material");
  const std::size_t expected = n_cells * n_mat;
  TENRYU_ASSERT(state.volFrac.size() == expected,
                "Shell subcycle void mask update requires volFrac size == n_cells*n_materials");
  std::vector<double> host_vf(expected, 0.0);
  state.volFrac.copy_to_host(host_vf.data());
  constexpr double kFracTol = 1.0e-12;
  int void_mat = -1;
  for (std::size_t m = 0; m < n_mat; ++m) {
    if (cfg.materials.materials[m].is_void) {
      void_mat = static_cast<int>(m);
      break;
    }
  }
  bool volfrac_changed = false;
  for (std::size_t c = 0; c < n_cells; ++c) {
    const std::size_t base = c * n_mat;
    if (state.cell_is_void[c] != 0U) {
      for (std::size_t m = 0; m < n_mat; ++m) {
        const double vf = (static_cast<int>(m) == void_mat) ? 1.0 : 0.0;
        if (host_vf[base + m] != vf) {
          host_vf[base + m] = vf;
          volfrac_changed = true;
        }
      }
      state.cell_is_void[c] = 1U;
      if (!state.hydro_active.empty()) {
        TENRYU_ASSERT(state.hydro_active.size() == n_cells,
                      "Shell subcycle void mask update requires hydro_active/rho size match");
        state.hydro_active[c] = 0;
      }
      continue;
    }

    double nonvoid_sum = 0.0;
    for (std::size_t m = 0; m < n_mat; ++m) {
      if (!cfg.materials.materials[m].is_void) {
        nonvoid_sum += host_vf[base + m];
      }
    }
    state.cell_is_void[c] =
        static_cast<std::uint8_t>((nonvoid_sum <= kFracTol) ? 1U : 0U);
    if (state.cell_is_void[c] != 0U && !state.hydro_active.empty()) {
      TENRYU_ASSERT(state.hydro_active.size() == n_cells,
                    "Shell subcycle void mask update requires hydro_active/rho size match");
      state.hydro_active[c] = 0;
    }
  }
  if (volfrac_changed) {
    state.volFrac.copy_from_host(host_vf.data());
  }
}

struct ShellProtectedPatch {
  bool applicable = false;
  std::string status;
  std::vector<std::uint8_t> cell_in_patch;
  std::vector<std::uint8_t> cell_support;
  std::vector<std::uint8_t> node_affected;
  std::vector<std::uint8_t> cell_energy_closure;
  std::vector<std::uint8_t> node_rezone_active;
  std::vector<std::uint8_t> node_patch_boundary;
  std::vector<std::uint8_t> frozen_node;
  std::vector<std::uint8_t> inactive_cell;
  int seed_cells = 0;
  int shell_seed_cells = 0;
  int n_patch_cells = 0;
  int n_support_cells = 0;
  int n_affected_nodes = 0;
  int n_energy_closure_cells = 0;
  int n_energy_collar_cells = 0;
  int first_energy_collar_cell = -1;
  std::array<int, 8> energy_collar_sample{{-1, -1, -1, -1, -1, -1, -1, -1}};
  int n_closure_hard_frozen_nodes = 0;
  int n_closure_hsrc_axis_pole_center = 0;
  int n_closure_hsrc_core = 0;
  int n_closure_hsrc_deref = 0;
  int shrink_iterations = 0;
  int M_size_initial = 0;
  int M_size_final = 0;
  int n_M_removed_for_closure = 0;
  std::array<int, 4> closure_deref_node_id{{-1, -1, -1, -1}};
  std::array<int, 4> closure_deref_node_block{{-1, -1, -1, -1}};
  std::array<int, 4> closure_deref_node_local_i{{-1, -1, -1, -1}};
  std::array<int, 4> closure_deref_node_local_j{{-1, -1, -1, -1}};
  std::array<int, 4> closure_deref_node_subtype{{0, 0, 0, 0}};
  std::array<int, 8> closure_removed_M_node_id{{-1, -1, -1, -1, -1, -1, -1, -1}};
  std::array<int, 8> closure_removed_M_node_block{{-1, -1, -1, -1, -1, -1, -1, -1}};
  std::array<int, 8> closure_removed_M_node_local_i{{-1, -1, -1, -1, -1, -1, -1, -1}};
  std::array<int, 8> closure_removed_M_node_local_j{{-1, -1, -1, -1, -1, -1, -1, -1}};
  int n_affected_boundary_outer_nodes = 0;
  int n_closure_core_cells = 0;
  int n_active_nodes = 0;
  int n_boundary_nodes = 0;
  int n_frozen_nodes = 0;
};

struct ShellMomentumTransportSupport {
  std::vector<std::uint8_t> cell_mask;
  int n_cells = 0;
  int n_collar_cells = 0;
  int n_iterations = 0;
};

bool shell_rezone_cell_is_shell(const mesh::MultiBlockTopology& mb,
                                const int cell) {
  if (cell < 0 ||
      static_cast<std::size_t>(cell) >= mb.cell_block_id.size()) {
    return false;
  }
  const int block_id = mb.cell_block_id[static_cast<std::size_t>(cell)];
  return block_id >= 0 && block_id < static_cast<int>(mb.blocks.size()) &&
         mb.blocks[static_cast<std::size_t>(block_id)].role ==
             mesh::BlockRole::POLAR_SHELL;
}

void mark_shell_rezone_cell_nodes(const core::State& state,
                                  const int cell,
                                  std::vector<std::uint8_t>& node_mask) {
  const auto& mb = *state.mesh.topo.multiblock;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (cell < 0 || cell >= n_cells) {
    return;
  }
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  const int nverts =
      state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)
          ? mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, cell)
          : mesh::kMeshTopoCellStorageSlots;
  for (int k = 0; k < nverts; ++k) {
    const int node = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
    if (node >= 0 && node < n_nodes) {
      node_mask[static_cast<std::size_t>(node)] = 1U;
    }
  }
}

std::vector<std::uint8_t> shell_rezone_cell_vertices_mask(
    const core::State& state,
    const std::vector<std::uint8_t>& cell_mask) {
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  std::vector<std::uint8_t> node_mask(static_cast<std::size_t>(n_nodes), 0U);
  if (n_cells <= 0 || n_nodes <= 0 ||
      cell_mask.size() != static_cast<std::size_t>(n_cells) ||
      !state.mesh.topo.multiblock.has_value()) {
    return node_mask;
  }
  for (int c = 0; c < n_cells; ++c) {
    if (cell_mask[static_cast<std::size_t>(c)] == 0U) {
      continue;
    }
    mark_shell_rezone_cell_nodes(state, c, node_mask);
  }
  return node_mask;
}

ShellMomentumTransportSupport build_shell_momentum_transport_support(
    const core::State& state,
    const ShellProtectedPatch& patch,
    const std::vector<double>& r_old,
    const std::vector<double>& z_old,
    const std::vector<double>& r_new,
    const std::vector<double>& z_new) {
  ShellMomentumTransportSupport support;
  const int n_cells = state.mesh.topo.n_cells;
  if (n_cells <= 0 ||
      patch.cell_support.size() != static_cast<std::size_t>(n_cells) ||
      patch.cell_energy_closure.size() != static_cast<std::size_t>(n_cells) ||
      patch.inactive_cell.size() != static_cast<std::size_t>(n_cells) ||
      r_old.size() != static_cast<std::size_t>(state.mesh.topo.n_nodes) ||
      z_old.size() != static_cast<std::size_t>(state.mesh.topo.n_nodes) ||
      r_new.size() != static_cast<std::size_t>(state.mesh.topo.n_nodes) ||
      z_new.size() != static_cast<std::size_t>(state.mesh.topo.n_nodes) ||
      !state.mesh.topo.multiblock.has_value()) {
    support.cell_mask = patch.cell_support;
    return support;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const int* const offsets = mb.cell_node_csr_offsets.data();
  const int* const indices = mb.cell_node_csr_indices.data();
  const int* const orientation = mb.cell_orientation_sign.data();
  const std::uint8_t* const cell_nverts =
      state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)
          ? state.mesh.cell_nverts.data()
          : nullptr;
  support.cell_mask = patch.cell_support;
  std::vector<std::uint8_t> eligible(static_cast<std::size_t>(n_cells), 0U);
  for (int c = 0; c < n_cells; ++c) {
    const auto idx = static_cast<std::size_t>(c);
    if (patch.cell_energy_closure[idx] != 0U &&
        patch.inactive_cell[idx] == 0U) {
      eligible[idx] = 1U;
    }
  }

  for (;;) {
    bool added = false;
    for (const auto& face : mb.unique_internal_faces) {
      const int cell_a = face.cell_a;
      const int cell_b = face.cell_b;
      if (cell_a < 0 || cell_a >= n_cells || cell_b < 0 ||
          cell_b >= n_cells || face.local_a < 0 || face.local_b < 0) {
        continue;
      }
      const double dV_a = csr_face_swept_volume_outward(
          r_old.data(),
          z_old.data(),
          r_new.data(),
          z_new.data(),
          offsets,
          indices,
          orientation,
          cell_a,
          face.local_a,
          cell_nverts);
      if (!std::isfinite(dV_a) || dV_a == 0.0) {
        continue;
      }
      const auto ia = static_cast<std::size_t>(cell_a);
      const auto ib = static_cast<std::size_t>(cell_b);
      const bool in_a = support.cell_mask[ia] != 0U;
      const bool in_b = support.cell_mask[ib] != 0U;
      if (in_a == in_b) {
        continue;
      }
      if (in_a && eligible[ib] != 0U) {
        support.cell_mask[ib] = 1U;
        added = true;
      } else if (in_b && eligible[ia] != 0U) {
        support.cell_mask[ia] = 1U;
        added = true;
      }
    }
    if (!added) {
      break;
    }
    ++support.n_iterations;
  }

  for (int c = 0; c < n_cells; ++c) {
    const auto idx = static_cast<std::size_t>(c);
    if (support.cell_mask[idx] == 0U) {
      continue;
    }
    ++support.n_cells;
    if (patch.cell_support[idx] == 0U) {
      ++support.n_collar_cells;
    }
  }
  return support;
}

ShellProtectedPatch build_shell_protected_patch(core::State& state,
                                                const core::Config& cfg,
                                                const std::vector<std::uint8_t>& seed_mask,
                                                const int layers) {
  ShellProtectedPatch patch;
  if (cfg.main.dim != 2 || cfg.main.dimension != "2D_RZ" ||
      !mesh::mesh_topo_is_multiblock(cfg.mesh) ||
      !state.mesh.topo.multiblock.has_value()) {
    patch.status = "not_2d_rz_multiblock";
    return patch;
  }
  const auto& topo = state.mesh.topo;
  const auto& mb = *topo.multiblock;
  const int n_cells = topo.n_cells;
  const int n_nodes = topo.n_nodes;
  if (n_cells <= 0 || n_nodes <= 0 ||
      seed_mask.size() != static_cast<std::size_t>(n_cells)) {
    patch.status = "bad_seed_mask";
    return patch;
  }
  if (central_pseudo_core::configured(cfg)) {
    central_pseudo_core::ensure_built(state, cfg);
  }
  pole_angular_derefine::ensure_built(state, cfg);

  patch.cell_in_patch.assign(static_cast<std::size_t>(n_cells), 0U);
  patch.inactive_cell.assign(static_cast<std::size_t>(n_cells), 0U);
  const auto& pc = state.central_pseudo_core;
  if (pc.inactive_member_mask.size() == static_cast<std::size_t>(n_cells)) {
    for (int c = 0; c < n_cells; ++c) {
      if (pc.inactive_member_mask[static_cast<std::size_t>(c)] != 0U) {
        patch.inactive_cell[static_cast<std::size_t>(c)] = 1U;
      }
    }
  }
  if (state.pole_angular_derefine.inactive_member_mask.size() ==
      static_cast<std::size_t>(n_cells)) {
    for (int c = 0; c < n_cells; ++c) {
      if (state.pole_angular_derefine
              .inactive_member_mask[static_cast<std::size_t>(c)] != 0U) {
        patch.inactive_cell[static_cast<std::size_t>(c)] = 1U;
      }
    }
  }

  for (int c = 0; c < n_cells; ++c) {
    if (seed_mask[static_cast<std::size_t>(c)] == 0U) {
      continue;
    }
    ++patch.seed_cells;
    if (shell_rezone_cell_is_shell(mb, c) &&
        patch.inactive_cell[static_cast<std::size_t>(c)] == 0U) {
      patch.cell_in_patch[static_cast<std::size_t>(c)] = 1U;
      ++patch.shell_seed_cells;
    }
  }
  if (patch.shell_seed_cells == 0) {
    patch.status = "no_shell_seed";
    return patch;
  }

  for (int layer = 0; layer < layers; ++layer) {
    std::vector<std::uint8_t> next = patch.cell_in_patch;
    multiblock_center_patch_detail::dilate_multiblock_cells(
        state, patch.cell_in_patch, next);
    for (int c = 0; c < n_cells; ++c) {
      if (!shell_rezone_cell_is_shell(mb, c) ||
          patch.inactive_cell[static_cast<std::size_t>(c)] != 0U) {
        next[static_cast<std::size_t>(c)] = 0U;
      }
    }
    patch.cell_in_patch.swap(next);
  }

  for (const std::uint8_t in_patch : patch.cell_in_patch) {
    if (in_patch != 0U) {
      ++patch.n_patch_cells;
    }
  }
  if (patch.n_patch_cells == 0) {
    patch.status = "empty_patch";
    return patch;
  }

  patch.frozen_node.assign(static_cast<std::size_t>(n_nodes), 0U);
  std::vector<std::uint8_t> hard_frozen_node(
      static_cast<std::size_t>(n_nodes), 0U);
  std::vector<std::uint8_t> hard_frozen_axis_pole_center(
      static_cast<std::size_t>(n_nodes), 0U);
  std::vector<std::uint8_t> hard_frozen_core(
      static_cast<std::size_t>(n_nodes), 0U);
  std::vector<std::uint8_t> hard_frozen_deref(
      static_cast<std::size_t>(n_nodes), 0U);
  std::vector<std::uint8_t> core_owned_cell(
      static_cast<std::size_t>(n_cells), 0U);
  if (topo.node_flags.size() == static_cast<std::size_t>(n_nodes)) {
    for (int n = 0; n < n_nodes; ++n) {
      const std::uint8_t flags = topo.node_flags[static_cast<std::size_t>(n)];
      if ((flags & (mesh::NODE_BOUNDARY | mesh::NODE_AXIS |
                    mesh::NODE_CENTER | mesh::NODE_POLE_AXIS |
                    mesh::NODE_OUTER_PHYSICAL_BOUNDARY)) != 0U) {
        patch.frozen_node[static_cast<std::size_t>(n)] = 1U;
      }
      if ((flags & (mesh::NODE_AXIS | mesh::NODE_CENTER |
                    mesh::NODE_POLE_AXIS)) != 0U) {
        hard_frozen_node[static_cast<std::size_t>(n)] = 1U;
        hard_frozen_axis_pole_center[static_cast<std::size_t>(n)] = 1U;
      }
    }
  }
  if (mb.has_trifan_cap) {
    const int apex = mesh::mesh_topo_cap_apex_node_id(mb);
    if (apex >= 0 && apex < n_nodes) {
      patch.frozen_node[static_cast<std::size_t>(apex)] = 1U;
      hard_frozen_node[static_cast<std::size_t>(apex)] = 1U;
      hard_frozen_axis_pole_center[static_cast<std::size_t>(apex)] = 1U;
    }
  }
  if (central_pseudo_core::active(state) &&
      pc.member_mask.size() == static_cast<std::size_t>(n_cells)) {
    for (int c = 0; c < n_cells; ++c) {
      if (pc.member_mask[static_cast<std::size_t>(c)] != 0U) {
        mark_shell_rezone_cell_nodes(state, c, patch.frozen_node);
        mark_shell_rezone_cell_nodes(state, c, hard_frozen_node);
        mark_shell_rezone_cell_nodes(state, c, hard_frozen_core);
        core_owned_cell[static_cast<std::size_t>(c)] = 1U;
      }
    }
  }
  if (pc.inactive_member_mask.size() == static_cast<std::size_t>(n_cells)) {
    for (int c = 0; c < n_cells; ++c) {
      if (pc.inactive_member_mask[static_cast<std::size_t>(c)] != 0U) {
        mark_shell_rezone_cell_nodes(state, c, patch.frozen_node);
        mark_shell_rezone_cell_nodes(state, c, hard_frozen_node);
        mark_shell_rezone_cell_nodes(state, c, hard_frozen_core);
        core_owned_cell[static_cast<std::size_t>(c)] = 1U;
      }
    }
  }
  if (pc.boundary_node_mask.size() == static_cast<std::size_t>(n_nodes)) {
    for (int n = 0; n < n_nodes; ++n) {
      if (pc.boundary_node_mask[static_cast<std::size_t>(n)] != 0U) {
        patch.frozen_node[static_cast<std::size_t>(n)] = 1U;
        hard_frozen_node[static_cast<std::size_t>(n)] = 1U;
        hard_frozen_core[static_cast<std::size_t>(n)] = 1U;
      }
    }
  }
  if (state.pole_angular_derefine.inactive_member_mask.size() ==
      static_cast<std::size_t>(n_cells)) {
    for (int c = 0; c < n_cells; ++c) {
      if (state.pole_angular_derefine
              .inactive_member_mask[static_cast<std::size_t>(c)] != 0U) {
        mark_shell_rezone_cell_nodes(state, c, patch.frozen_node);
        mark_shell_rezone_cell_nodes(state, c, hard_frozen_node);
        mark_shell_rezone_cell_nodes(state, c, hard_frozen_deref);
      }
    }
  }
  if (state.pole_angular_derefine.boundary_node_mask.size() ==
      static_cast<std::size_t>(n_nodes)) {
    for (int n = 0; n < n_nodes; ++n) {
      if (state.pole_angular_derefine
              .boundary_node_mask[static_cast<std::size_t>(n)] != 0U) {
        patch.frozen_node[static_cast<std::size_t>(n)] = 1U;
        hard_frozen_node[static_cast<std::size_t>(n)] = 1U;
        hard_frozen_deref[static_cast<std::size_t>(n)] = 1U;
      }
    }
  }

  const std::vector<std::uint8_t>* cell_nverts =
      state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)
          ? &state.mesh.cell_nverts
          : nullptr;
  const ReverseCellNodeCSR reverse_csr =
      build_reverse_cell_node_csr(
          mb, n_nodes, cell_nverts, state.mesh.corner_stride);
  patch.node_rezone_active.assign(static_cast<std::size_t>(n_nodes), 0U);
  patch.node_patch_boundary.assign(static_cast<std::size_t>(n_nodes), 0U);
  for (int n = 0; n < n_nodes; ++n) {
    const int begin = reverse_csr.node_offsets[static_cast<std::size_t>(n)];
    const int end = reverse_csr.node_offsets[static_cast<std::size_t>(n) + 1U];
    bool touches_patch = false;
    bool touches_outside = false;
    for (int p = begin; p < end; ++p) {
      const int c = reverse_csr.node_cells[static_cast<std::size_t>(p)];
      if (c < 0 || c >= n_cells) {
        continue;
      }
      if (patch.cell_in_patch[static_cast<std::size_t>(c)] != 0U) {
        touches_patch = true;
      } else {
        touches_outside = true;
      }
    }
    const auto idx = static_cast<std::size_t>(n);
    if (!touches_patch) {
      continue;
    }
    if (touches_outside || patch.frozen_node[idx] != 0U ||
        multiblock_center_patch_detail::node_pinned(topo, mb, n)) {
      patch.node_patch_boundary[idx] = 1U;
    } else {
      patch.node_rezone_active[idx] = 1U;
    }
  }

  const auto locate_shell_node = [&](const int node,
                                     int& block_id,
                                     int& local_i,
                                     int& local_j) {
    block_id = -1;
    local_i = -1;
    local_j = -1;
    const auto& dc = state.pole_angular_derefine;
    if (dc.block_id < 0 || dc.n_j_cells < 0 || dc.owned_node_begin < 0) {
      return;
    }
    const int stride = dc.n_j_cells + 1;
    const int rel = node - dc.owned_node_begin;
    const int n_shell_nodes = (dc.n_i_cells + 1) * stride;
    if (stride <= 0 || rel < 0 || rel >= n_shell_nodes) {
      return;
    }
    block_id = dc.block_id;
    local_i = rel / stride;
    local_j = rel - local_i * stride;
  };
  const auto deref_node_subtype = [&](const int node) {
    const auto& dc = state.pole_angular_derefine;
    if (node < 0 || node >= n_nodes) {
      return 0;
    }
    const auto idx = static_cast<std::size_t>(node);
    if (dc.boundary_node_mask.size() == static_cast<std::size_t>(n_nodes) &&
        dc.boundary_node_mask[idx] != 0U) {
      return 1;
    }
    if (dc.inactive_member_mask.size() == static_cast<std::size_t>(n_cells)) {
      const int begin = reverse_csr.node_offsets[idx];
      const int end = reverse_csr.node_offsets[idx + 1U];
      for (int p = begin; p < end; ++p) {
        const int c = reverse_csr.node_cells[static_cast<std::size_t>(p)];
        if (c >= 0 && c < n_cells &&
            dc.inactive_member_mask[static_cast<std::size_t>(c)] != 0U) {
          return 2;
        }
      }
    }
    return 0;
  };
  const auto record_removed_M = [&](const int node) {
    for (std::size_t i = 0; i < patch.closure_removed_M_node_id.size(); ++i) {
      if (patch.closure_removed_M_node_id[i] == node) {
        return;
      }
      if (patch.closure_removed_M_node_id[i] < 0) {
        int block_id = -1;
        int local_i = -1;
        int local_j = -1;
        locate_shell_node(node, block_id, local_i, local_j);
        patch.closure_removed_M_node_id[i] = node;
        patch.closure_removed_M_node_block[i] = block_id;
        patch.closure_removed_M_node_local_i[i] = local_i;
        patch.closure_removed_M_node_local_j[i] = local_j;
        return;
      }
    }
  };
  const auto recompute_closure = [&]() {
    patch.n_support_cells = 0;
    patch.n_affected_nodes = 0;
    patch.n_energy_closure_cells = 0;
    patch.n_energy_collar_cells = 0;
    patch.first_energy_collar_cell = -1;
    patch.energy_collar_sample.fill(-1);
    patch.n_closure_hard_frozen_nodes = 0;
    patch.n_closure_hsrc_axis_pole_center = 0;
    patch.n_closure_hsrc_core = 0;
    patch.n_closure_hsrc_deref = 0;
    patch.n_affected_boundary_outer_nodes = 0;
    patch.n_closure_core_cells = 0;
    patch.n_active_nodes = 0;
    patch.n_boundary_nodes = 0;
    patch.n_frozen_nodes = 0;
    patch.closure_deref_node_id.fill(-1);
    patch.closure_deref_node_block.fill(-1);
    patch.closure_deref_node_local_i.fill(-1);
    patch.closure_deref_node_local_j.fill(-1);
    patch.closure_deref_node_subtype.fill(0);
    patch.cell_support.assign(static_cast<std::size_t>(n_cells), 0U);
    patch.node_affected.assign(static_cast<std::size_t>(n_nodes), 0U);
    patch.cell_energy_closure.assign(static_cast<std::size_t>(n_cells), 0U);
    for (int n = 0; n < n_nodes; ++n) {
      const auto idx = static_cast<std::size_t>(n);
      if (patch.frozen_node[idx] != 0U) {
        ++patch.n_frozen_nodes;
      }
      if (patch.node_patch_boundary[idx] != 0U) {
        ++patch.n_boundary_nodes;
      }
      if (patch.node_rezone_active[idx] == 0U) {
        continue;
      }
      ++patch.n_active_nodes;
      const int begin = reverse_csr.node_offsets[idx];
      const int end = reverse_csr.node_offsets[idx + 1U];
      for (int p = begin; p < end; ++p) {
        const int c = reverse_csr.node_cells[static_cast<std::size_t>(p)];
        if (c >= 0 && c < n_cells &&
            patch.inactive_cell[static_cast<std::size_t>(c)] == 0U) {
          patch.cell_support[static_cast<std::size_t>(c)] = 1U;
        }
      }
    }
    for (int c = 0; c < n_cells; ++c) {
      if (patch.cell_support[static_cast<std::size_t>(c)] == 0U) {
        continue;
      }
      ++patch.n_support_cells;
      const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
      const int nverts =
          state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)
              ? mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c)
              : mesh::kMeshTopoCellStorageSlots;
      for (int k = 0; k < nverts; ++k) {
        const int node =
            mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        if (node >= 0 && node < n_nodes) {
          patch.node_affected[static_cast<std::size_t>(node)] = 1U;
        }
      }
    }
    for (int n = 0; n < n_nodes; ++n) {
      const auto idx = static_cast<std::size_t>(n);
      if (patch.node_affected[idx] == 0U) {
        continue;
      }
      ++patch.n_affected_nodes;
      if (hard_frozen_node[idx] != 0U) {
        ++patch.n_closure_hard_frozen_nodes;
      }
      if (hard_frozen_axis_pole_center[idx] != 0U) {
        ++patch.n_closure_hsrc_axis_pole_center;
      }
      if (hard_frozen_core[idx] != 0U) {
        ++patch.n_closure_hsrc_core;
      }
      if (hard_frozen_deref[idx] != 0U) {
        ++patch.n_closure_hsrc_deref;
        for (std::size_t s = 0; s < patch.closure_deref_node_id.size(); ++s) {
          if (patch.closure_deref_node_id[s] < 0) {
            int block_id = -1;
            int local_i = -1;
            int local_j = -1;
            locate_shell_node(n, block_id, local_i, local_j);
            patch.closure_deref_node_id[s] = n;
            patch.closure_deref_node_block[s] = block_id;
            patch.closure_deref_node_local_i[s] = local_i;
            patch.closure_deref_node_local_j[s] = local_j;
            patch.closure_deref_node_subtype[s] = deref_node_subtype(n);
            break;
          }
        }
      }
      if (topo.node_flags.size() == static_cast<std::size_t>(n_nodes) &&
          (topo.node_flags[idx] &
           (mesh::NODE_BOUNDARY | mesh::NODE_OUTER_PHYSICAL_BOUNDARY)) != 0U) {
        ++patch.n_affected_boundary_outer_nodes;
      }
      const int begin = reverse_csr.node_offsets[idx];
      const int end = reverse_csr.node_offsets[idx + 1U];
      for (int p = begin; p < end; ++p) {
        const int c = reverse_csr.node_cells[static_cast<std::size_t>(p)];
        if (c < 0 || c >= n_cells) {
          continue;
        }
        if (core_owned_cell[static_cast<std::size_t>(c)] != 0U) {
          patch.cell_energy_closure[static_cast<std::size_t>(c)] = 1U;
        } else if (patch.inactive_cell[static_cast<std::size_t>(c)] == 0U) {
          patch.cell_energy_closure[static_cast<std::size_t>(c)] = 1U;
        }
      }
    }
    for (int c = 0; c < n_cells; ++c) {
      if (patch.cell_energy_closure[static_cast<std::size_t>(c)] == 0U) {
        continue;
      }
      ++patch.n_energy_closure_cells;
      if (core_owned_cell[static_cast<std::size_t>(c)] != 0U) {
        ++patch.n_closure_core_cells;
      }
      if (patch.cell_support[static_cast<std::size_t>(c)] == 0U &&
          patch.inactive_cell[static_cast<std::size_t>(c)] == 0U) {
        ++patch.n_energy_collar_cells;
        if (patch.first_energy_collar_cell < 0) {
          patch.first_energy_collar_cell = c;
        }
        for (std::size_t i = 0; i < patch.energy_collar_sample.size(); ++i) {
          if (patch.energy_collar_sample[i] < 0) {
            patch.energy_collar_sample[i] = c;
            break;
          }
        }
      }
    }
  };
  recompute_closure();
  patch.M_size_initial = patch.n_active_nodes;
  constexpr int kShrinkMMinMovableNodes = 3;
  while (patch.n_closure_hard_frozen_nodes > 0) {
    if (patch.n_active_nodes < kShrinkMMinMovableNodes) {
      patch.M_size_final = patch.n_active_nodes;
      patch.status = "closure_unshrinkable";
      return patch;
    }
    std::vector<std::uint8_t> remove_M(static_cast<std::size_t>(n_nodes), 0U);
    int n_remove = 0;
    for (int m = 0; m < n_nodes; ++m) {
      const auto midx = static_cast<std::size_t>(m);
      if (patch.node_rezone_active[midx] == 0U) {
        continue;
      }
      const int begin = reverse_csr.node_offsets[midx];
      const int end = reverse_csr.node_offsets[midx + 1U];
      for (int p = begin; p < end && remove_M[midx] == 0U; ++p) {
        const int c = reverse_csr.node_cells[static_cast<std::size_t>(p)];
        if (c < 0 || c >= n_cells ||
            patch.inactive_cell[static_cast<std::size_t>(c)] != 0U) {
          continue;
        }
        const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
        const int nverts =
            state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)
                ? mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c)
                : mesh::kMeshTopoCellStorageSlots;
        for (int k = 0; k < nverts; ++k) {
          const int h =
              mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
          if (h >= 0 && h < n_nodes &&
              patch.node_affected[static_cast<std::size_t>(h)] != 0U &&
              hard_frozen_node[static_cast<std::size_t>(h)] != 0U) {
            remove_M[midx] = 1U;
            ++n_remove;
            record_removed_M(m);
            break;
          }
        }
      }
    }
    if (n_remove == 0) {
      patch.M_size_final = patch.n_active_nodes;
      patch.status = "closure_unshrinkable";
      return patch;
    }
    for (int n = 0; n < n_nodes; ++n) {
      if (remove_M[static_cast<std::size_t>(n)] != 0U) {
        patch.node_rezone_active[static_cast<std::size_t>(n)] = 0U;
        patch.node_patch_boundary[static_cast<std::size_t>(n)] = 1U;
        ++patch.n_M_removed_for_closure;
      }
    }
    ++patch.shrink_iterations;
    recompute_closure();
  }
  patch.M_size_final = patch.n_active_nodes;
  if (patch.n_support_cells == 0 || patch.n_affected_nodes == 0 ||
      patch.n_energy_closure_cells == 0) {
    patch.status = "empty_support_closure";
    return patch;
  }
  if (patch.n_closure_hard_frozen_nodes > 0 ||
      patch.n_closure_core_cells > 0) {
    patch.status = "closure_violates_frozen";
    return patch;
  }
  if (patch.n_active_nodes == 0) {
    patch.status = "no_active_nodes";
    return patch;
  }
  patch.applicable = true;
  patch.status = "ok";
  return patch;
}

double shell_patch_q_min_at(const core::State& state,
                            const std::vector<double>& r_old,
                            const std::vector<double>& z_old,
                            const std::vector<double>& r_new,
                            const std::vector<double>& z_new,
                            const std::vector<std::uint8_t>& cell_mask,
                            const double tau,
                            const double q_floor,
                            int* worst_cell = nullptr) {
  double q_min = std::numeric_limits<double>::infinity();
  int cell_min = -1;
  const int n_cells = state.mesh.topo.n_cells;
  for (int c = 0; c < n_cells; ++c) {
    if (cell_mask[static_cast<std::size_t>(c)] == 0U) {
      continue;
    }
    const mesh::MeshForecastComponents q =
        mesh::path_admissibility_detail::mesh_forecast_cell_components(
            state, r_old, z_old, r_new, z_new, c, tau, q_floor);
    const double q_cell =
        mesh::path_admissibility_detail::mesh_forecast_q_min(q);
    if (q_cell < q_min) {
      q_min = q_cell;
      cell_min = c;
    }
  }
  if (worst_cell != nullptr) {
    *worst_cell = cell_min;
  }
  return q_min;
}

double shell_patch_q_min_path(const core::State& state,
                              const std::vector<double>& r_old,
                              const std::vector<double>& z_old,
                              const std::vector<double>& r_new,
                              const std::vector<double>& z_new,
                              const std::vector<std::uint8_t>& cell_mask,
                              const double q_floor) {
  double q_min = std::numeric_limits<double>::infinity();
  constexpr int kSamples = 64;
  for (int s = 0; s <= kSamples; ++s) {
    const double tau = static_cast<double>(s) / static_cast<double>(kSamples);
    q_min = std::min(q_min, shell_patch_q_min_at(
                                state, r_old, z_old, r_new, z_new,
                                cell_mask, tau, q_floor, nullptr));
  }
  return q_min;
}

double shell_rezone_max_masked_delta(const std::vector<double>& r0,
                                     const std::vector<double>& z0,
                                     const std::vector<double>& r1,
                                     const std::vector<double>& z1,
                                     const std::vector<std::uint8_t>& mask) {
  double max_delta = 0.0;
  const std::size_t n = std::min({r0.size(), z0.size(), r1.size(), z1.size(),
                                  mask.size()});
  for (std::size_t i = 0; i < n; ++i) {
    if (mask[i] == 0U) {
      continue;
    }
    max_delta = std::max(max_delta,
                         std::hypot(r1[i] - r0[i], z1[i] - z0[i]));
  }
  return max_delta;
}

double shell_rezone_source_current_delta(const core::State& state,
                                         const double* const d_xr_source,
                                         const double* const d_xz_source) {
  if (d_xr_source == nullptr || d_xz_source == nullptr) {
    return std::numeric_limits<double>::infinity();
  }
  const int n_nodes = state.mesh.topo.n_nodes;
  std::vector<double> src_r(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> src_z(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> cur_r;
  std::vector<double> cur_z;
  CUDA_CHECK(cudaMemcpy(src_r.data(),
                        d_xr_source,
                        src_r.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(src_z.data(),
                        d_xz_source,
                        src_z.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
  state.x_r.copy_to_host(cur_r);
  state.x_z.copy_to_host(cur_z);
  double max_delta = 0.0;
  for (int n = 0; n < n_nodes; ++n) {
    const auto idx = static_cast<std::size_t>(n);
    max_delta =
        std::max(max_delta, std::hypot(cur_r[idx] - src_r[idx],
                                       cur_z[idx] - src_z[idx]));
  }
  return max_delta;
}

struct ShellNodeKinematicSnapshot {
  bool valid = false;
  std::vector<double> r;
  std::vector<double> z;
  std::vector<double> vr;
  std::vector<double> vz;
  std::vector<double> nodal_mass;
  std::vector<double> corner_mass;
};

struct ShellClassMomentum {
  double pr = 0.0;
  double pz = 0.0;
};

ShellNodeKinematicSnapshot shell_rezone_capture_node_kinematics(
    const core::State& state) {
  ShellNodeKinematicSnapshot out;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (n_cells <= 0 || n_nodes <= 0 ||
      !state.mesh.topo.multiblock.has_value() ||
      !state.corner_mass_initialized ||
      state.corner_mass.size() != static_cast<std::size_t>(n_cells) * 4U ||
      state.v_r.size() != static_cast<std::size_t>(n_nodes) ||
      state.v_z.size() != static_cast<std::size_t>(n_nodes)) {
    return out;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  if (mb.cell_node_csr_offsets.size() != static_cast<std::size_t>(n_cells + 1) ||
      mb.cell_node_csr_indices.empty()) {
    return out;
  }

  state.x_r.copy_to_host(out.r);
  state.x_z.copy_to_host(out.z);
  state.v_r.copy_to_host(out.vr);
  state.v_z.copy_to_host(out.vz);
  if (out.r.size() != static_cast<std::size_t>(n_nodes) ||
      out.z.size() != static_cast<std::size_t>(n_nodes) ||
      out.vr.size() != static_cast<std::size_t>(n_nodes) ||
      out.vz.size() != static_cast<std::size_t>(n_nodes)) {
    return out;
  }

  state.corner_mass.copy_to_host(out.corner_mass);
  if (out.corner_mass.size() != static_cast<std::size_t>(n_cells) * 4U) {
    return out;
  }
  out.nodal_mass.assign(static_cast<std::size_t>(n_nodes), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int end = mb.cell_node_csr_offsets[static_cast<std::size_t>(c + 1)];
    int active_nverts =
        state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)
            ? mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c)
            : mesh::kMeshTopoCellStorageSlots;
    active_nverts = std::min({active_nverts, end - off,
                              mesh::kMeshTopoCellStorageSlots});
    for (int k = 0; k < active_nverts; ++k) {
      const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      if (n < 0 || n >= n_nodes) {
        continue;
      }
      const double m =
          out.corner_mass[static_cast<std::size_t>(c) * 4U +
                          static_cast<std::size_t>(k)];
      if (std::isfinite(m)) {
        out.nodal_mass[static_cast<std::size_t>(n)] += m;
      }
    }
  }
  out.valid = true;
  return out;
}

ShellClassMomentum shell_rezone_masked_node_momentum(
    const ShellNodeKinematicSnapshot& snapshot,
    const std::vector<std::uint8_t>& mask) {
  ShellClassMomentum out;
  if (!snapshot.valid) {
    return out;
  }
  const std::size_t n =
      std::min({snapshot.nodal_mass.size(), snapshot.vr.size(),
                snapshot.vz.size(), mask.size()});
  long double pr = 0.0L;
  long double pz = 0.0L;
  for (std::size_t i = 0; i < n; ++i) {
    if (mask[i] == 0U) {
      continue;
    }
    const double m = snapshot.nodal_mass[i];
    if (!std::isfinite(m)) {
      continue;
    }
    pr += static_cast<long double>(m) *
          static_cast<long double>(snapshot.vr[i]);
    pz += static_cast<long double>(m) *
          static_cast<long double>(snapshot.vz[i]);
  }
  out.pr = static_cast<double>(pr);
  out.pz = static_cast<double>(pz);
  return out;
}

constexpr int kShellNodeClassOther = 0;
constexpr int kShellNodeClassActive = 1;
constexpr int kShellNodeClassBoundary = 2;

const char* shell_rezone_node_class_name(const int node_class) {
  switch (node_class) {
    case kShellNodeClassActive:
      return "active";
    case kShellNodeClassBoundary:
      return "boundary";
    default:
      return "other";
  }
}

bool shell_rezone_roundoff_changed(const double before,
                                   const double after) {
  if (!std::isfinite(before) || !std::isfinite(after)) {
    return before != after;
  }
  const double scale =
      std::max({std::abs(before), std::abs(after), 1.0e-300});
  return std::abs(after - before) >
         64.0 * std::numeric_limits<double>::epsilon() * scale;
}

int shell_rezone_node_class(const ShellProtectedPatch& patch,
                            const std::size_t i) {
  if (i < patch.node_rezone_active.size() &&
      patch.node_rezone_active[i] != 0U) {
    return kShellNodeClassActive;
  }
  if (i < patch.node_affected.size() &&
      i < patch.frozen_node.size() &&
      patch.node_affected[i] != 0U &&
      patch.frozen_node[i] == 0U) {
    return kShellNodeClassBoundary;
  }
  if (i < patch.node_patch_boundary.size() &&
      i < patch.frozen_node.size() &&
      patch.node_patch_boundary[i] != 0U &&
      patch.frozen_node[i] == 0U &&
      (i >= patch.node_rezone_active.size() ||
       patch.node_rezone_active[i] == 0U)) {
    return kShellNodeClassBoundary;
  }
  return kShellNodeClassOther;
}

void shell_rezone_insert_top_momentum_node(
    std::array<ShellMomentumTopNode, 5>& top,
    const ShellMomentumTopNode& candidate) {
  for (std::size_t i = 0; i < top.size(); ++i) {
    if (candidate.abs_dPr <= top[i].abs_dPr) {
      continue;
    }
    for (std::size_t j = top.size() - 1U; j > i; --j) {
      top[j] = top[j - 1U];
    }
    top[i] = candidate;
    return;
  }
}

void shell_rezone_insert_velchk_node(
    std::array<ShellVelocityCheckNode, 4>& top,
    const ShellVelocityCheckNode& candidate) {
  for (std::size_t i = 0; i < top.size(); ++i) {
    if (candidate.dv <= top[i].dv) {
      continue;
    }
    for (std::size_t j = top.size() - 1U; j > i; --j) {
      top[j] = top[j - 1U];
    }
    top[i] = candidate;
    return;
  }
}

void shell_rezone_apply_boundary_node_diagnostics(
    const ShellProtectedPatch& patch,
    const ShellNodeKinematicSnapshot& before,
    const ShellNodeKinematicSnapshot& after,
    const std::vector<std::uint8_t>& velocity_mask,
    const std::vector<std::uint8_t>& node_flags,
    ShellProtectedRezoneResult& result) {
  if (!before.valid || !after.valid) {
    return;
  }
  const bool velchk_diag =
      shell_rezone_env_int("TENRYU_I1B_SHELL_REZONE_VELCHK_DIAG", 0) != 0;
  const std::size_t n =
      std::min({patch.node_patch_boundary.size(),
                patch.node_rezone_active.size(),
                patch.node_affected.size(),
                patch.frozen_node.size(),
                velocity_mask.size(),
                before.r.size(),
                before.z.size(),
                before.vr.size(),
                before.vz.size(),
                before.nodal_mass.size(),
                after.r.size(),
                after.z.size(),
                after.vr.size(),
                after.vz.size(),
                after.nodal_mass.size()});
  std::vector<std::uint8_t> boundary_mask(n, 0U);
  std::vector<std::uint8_t> active_mask(n, 0U);
  long double p_scale = 0.0L;
  for (std::size_t i = 0; i < n; ++i) {
    if (patch.node_rezone_active[i] != 0U) {
      active_mask[i] = 1U;
    }
    if (patch.node_affected[i] != 0U &&
        patch.node_rezone_active[i] == 0U &&
        patch.frozen_node[i] == 0U) {
      boundary_mask[i] = 1U;
      result.max_boundary_dx =
          std::max(result.max_boundary_dx,
                   std::hypot(after.r[i] - before.r[i],
                              after.z[i] - before.z[i]));
      result.max_boundary_dvr =
          std::max(result.max_boundary_dvr,
                   std::abs(after.vr[i] - before.vr[i]));
      result.max_boundary_dvz =
          std::max(result.max_boundary_dvz,
                   std::abs(after.vz[i] - before.vz[i]));
      result.max_boundary_dm =
          std::max(result.max_boundary_dm,
                   std::abs(after.nodal_mass[i] - before.nodal_mass[i]));
    }
    const double m_before = before.nodal_mass[i];
    const double vr_before = before.vr[i];
    const double vz_before = before.vz[i];
    const double m_after = after.nodal_mass[i];
    const double vr_after = after.vr[i];
    const double vz_after = after.vz[i];
    if (std::isfinite(m_before) && std::isfinite(vr_before) &&
        std::isfinite(vz_before)) {
      p_scale += static_cast<long double>(m_before) *
                 static_cast<long double>(
                     std::hypot(vr_before, vz_before));
    }
    const int node_class = shell_rezone_node_class(patch, i);
    const double dpr =
        (std::isfinite(m_after) && std::isfinite(vr_after)
             ? m_after * vr_after
             : 0.0) -
        (std::isfinite(m_before) && std::isfinite(vr_before)
             ? m_before * vr_before
             : 0.0);
    const double dm = m_after - m_before;
    const double dvr = vr_after - vr_before;
    const double dvz = vz_after - vz_before;
    const bool mass_changed =
        shell_rezone_roundoff_changed(m_before, m_after);
    const bool velocity_changed =
        shell_rezone_roundoff_changed(vr_before, vr_after) ||
        shell_rezone_roundoff_changed(vz_before, vz_after);
    ShellMomentumTopNode candidate;
    candidate.node = static_cast<int>(i);
    candidate.node_class = node_class;
    candidate.abs_dPr = std::abs(dpr);
    candidate.dPr = dpr;
    candidate.dm = dm;
    candidate.dvr = dvr;
    candidate.dvz = dvz;
    candidate.mass_changed = mass_changed;
    candidate.velocity_changed = velocity_changed;
    shell_rezone_insert_top_momentum_node(result.top_momentum_nodes,
                                          candidate);
    if (node_class == kShellNodeClassOther && velocity_changed) {
      ++result.n_ext_node_vel_changed;
      if (result.first_ext_node_vel_changed < 0) {
        result.first_ext_node_vel_changed = static_cast<int>(i);
      }
      const double dv = std::hypot(dvr, dvz);
      if (dv > result.max_ext_node_vel_delta) {
        result.max_ext_node_vel_delta = dv;
        result.max_ext_node_vel_changed_node = static_cast<int>(i);
      }
      if (velchk_diag) {
        ShellVelocityCheckNode velchk;
        velchk.node = static_cast<int>(i);
        velchk.flags =
            i < node_flags.size() ? static_cast<int>(node_flags[i]) : 0;
        velchk.dv = dv;
        velchk.m_before = m_before;
        velchk.m_after = m_after;
        velchk.vr_before = vr_before;
        velchk.vr_after = vr_after;
        velchk.vz_before = vz_before;
        velchk.vz_after = vz_after;
        shell_rezone_insert_velchk_node(result.velchk_nodes, velchk);
      }
    }
    if (velocity_changed && velocity_mask[i] == 0U) {
      ++result.n_vel_changed_outside_velmask;
    }
  }
  result.p_scale = static_cast<double>(p_scale);
  const ShellClassMomentum boundary_before =
      shell_rezone_masked_node_momentum(before, boundary_mask);
  const ShellClassMomentum boundary_after =
      shell_rezone_masked_node_momentum(after, boundary_mask);
  const ShellClassMomentum active_before =
      shell_rezone_masked_node_momentum(before, active_mask);
  const ShellClassMomentum active_after =
      shell_rezone_masked_node_momentum(after, active_mask);
  result.dPr_boundary = boundary_after.pr - boundary_before.pr;
  result.dPz_boundary = boundary_after.pz - boundary_before.pz;
  result.dPr_active = active_after.pr - active_before.pr;
  result.dPz_active = active_after.pz - active_before.pz;

  const std::size_t n_cells =
      std::min({patch.cell_in_patch.size(),
                before.corner_mass.size() / 4U,
                after.corner_mass.size() / 4U});
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (patch.cell_in_patch[c] != 0U) {
      continue;
    }
    bool changed = false;
    double max_delta = 0.0;
    for (std::size_t k = 0; k < 4U; ++k) {
      const std::size_t idx = 4U * c + k;
      const double before_m = before.corner_mass[idx];
      const double after_m = after.corner_mass[idx];
      if (shell_rezone_roundoff_changed(before_m, after_m)) {
        changed = true;
        max_delta = std::max(max_delta, std::abs(after_m - before_m));
      }
    }
    if (!changed) {
      continue;
    }
    ++result.n_ext_cell_cmass_changed;
    if (result.first_ext_cell_cmass_changed < 0) {
      result.first_ext_cell_cmass_changed = static_cast<int>(c);
    }
    if (max_delta > result.max_ext_cell_cmass_delta) {
      result.max_ext_cell_cmass_delta = max_delta;
      result.max_ext_cell_cmass_changed_cell = static_cast<int>(c);
    }
  }
}

struct ReplayTotals {
  double mass = 0.0;
  double momentum_r = 0.0;
  double momentum_z = 0.0;
  double energy_internal = 0.0;
  double energy_kin = 0.0;
  double energy_total = 0.0;
};

bool shell_rezone_capture_corner_momentum(const core::State& state,
                                          double* const momentum_r,
                                          double* const momentum_z) {
  if (momentum_r == nullptr || momentum_z == nullptr) {
    return false;
  }
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (n_cells <= 0 || n_nodes <= 0 ||
      !state.mesh.topo.multiblock.has_value() ||
      !state.corner_mass_initialized ||
      state.corner_mass.size() != static_cast<std::size_t>(n_cells) * 4U ||
      state.v_r.size() != static_cast<std::size_t>(n_nodes) ||
      state.v_z.size() != static_cast<std::size_t>(n_nodes) ||
      state.mesh.multiblock_cell_node_csr_offsets.size() !=
          static_cast<std::size_t>(n_cells + 1) ||
      state.mesh.multiblock_cell_node_csr_indices.empty()) {
    return false;
  }

  std::vector<double> corner_mass;
  std::vector<double> v_r;
  std::vector<double> v_z;
  std::vector<int> offsets;
  std::vector<int> indices;
  state.corner_mass.copy_to_host(corner_mass);
  state.v_r.copy_to_host(v_r);
  state.v_z.copy_to_host(v_z);
  state.mesh.multiblock_cell_node_csr_offsets.copy_to_host(offsets);
  state.mesh.multiblock_cell_node_csr_indices.copy_to_host(indices);
  if (offsets.size() < static_cast<std::size_t>(n_cells + 1)) {
    return false;
  }

  long double pr = 0.0L;
  long double pz = 0.0L;
  for (int c = 0; c < n_cells; ++c) {
    const int off = offsets[static_cast<std::size_t>(c)];
    const int end = offsets[static_cast<std::size_t>(c + 1)];
    int active_nverts =
        state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)
            ? mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c)
            : mesh::kMeshTopoCellStorageSlots;
    active_nverts = std::min({active_nverts, end - off,
                              mesh::kMeshTopoCellStorageSlots});
    for (int k = 0; k < active_nverts; ++k) {
      const int n = indices[static_cast<std::size_t>(off + k)];
      if (n < 0 || n >= n_nodes) {
        continue;
      }
      const double m =
          corner_mass[static_cast<std::size_t>(c) * 4U +
                      static_cast<std::size_t>(k)];
      if (!std::isfinite(m)) {
        continue;
      }
      pr += static_cast<long double>(m) *
            static_cast<long double>(v_r[static_cast<std::size_t>(n)]);
      pz += static_cast<long double>(m) *
            static_cast<long double>(v_z[static_cast<std::size_t>(n)]);
    }
  }
  *momentum_r = static_cast<double>(pr);
  *momentum_z = static_cast<double>(pz);
  return true;
}

ReplayTotals shell_rezone_capture_totals(const core::State& state,
                                         const parallel::Reduction* reduction) {
  ReplayTotals out;
  out.mass = compute_total_mass(state, reduction);
  diagnostics::EnergyTotals energy =
      reduce_energy_totals_global(diagnostics::compute_energy_totals_2d(state),
                                  reduction);
  out.energy_internal = energy.E_int_e + energy.E_int_i;
  out.energy_kin = energy.E_kin;
  out.energy_total = total_energy(energy);
  if (!shell_rezone_capture_corner_momentum(
          state, &out.momentum_r, &out.momentum_z)) {
    AleVolClosureReference vol_closure =
        capture_ale_vol_closure_reference(state);
    out.momentum_r = vol_closure.total_r_momentum;
    out.momentum_z = vol_closure.total_z_momentum;
  }
  if (reduction != nullptr) {
    double values[2] = {out.momentum_r, out.momentum_z};
    reduction->allreduce_sum(values, 2);
    out.momentum_r = values[0];
    out.momentum_z = values[1];
  }
  return out;
}

}  // namespace detail

ShellProtectedRezoneResult shell_protected_rezone(
    core::State& state,
    const core::Config& cfg,
    const std::vector<std::uint8_t>& seed_mask,
    const HydroEOSContext* eos_ctx,
    const double dt_hydro_used,
    const parallel::Reduction* reduction) {
  ShellProtectedRezoneResult result;
  result.attempted = true;
  result.status = "started";
  result.q_floor = std::max(0.0, cfg.numerics.ale.path_admissibility_floor);
  result.q_warn = detail::shell_rezone_q_warn_default();
  result.q_release =
      detail::shell_rezone_q_release(result.q_warn, result.q_floor);
  if (cfg.main.dim != 2 || cfg.main.dimension != "2D_RZ" ||
      !mesh::mesh_topo_is_multiblock(cfg.mesh) ||
      !state.mesh.topo.multiblock.has_value()) {
    result.status = "not_2d_rz_multiblock";
    return result;
  }
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (n_cells <= 0 || n_nodes <= 0 ||
      seed_mask.size() != static_cast<std::size_t>(n_cells)) {
    result.status = "bad_seed_mask";
    return result;
  }
  const int blocks_nodes = (n_nodes + 255) / 256;
  if (state.x_r_reference.size() != static_cast<std::size_t>(n_nodes) ||
      state.x_z_reference.size() != static_cast<std::size_t>(n_nodes) ||
      state.cell_vol_initial.size() != static_cast<std::size_t>(n_cells)) {
    result.status = "missing_reference_state";
    return result;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const std::size_t node_bytes = static_cast<std::size_t>(n_nodes) * sizeof(double);
  const std::size_t cell_bytes = static_cast<std::size_t>(n_cells) * sizeof(double);

  core::DeviceArray<double> d_xr_old(static_cast<std::size_t>(n_nodes));
  core::DeviceArray<double> d_xz_old(static_cast<std::size_t>(n_nodes));
  core::DeviceArray<double> d_vol_old(static_cast<std::size_t>(n_cells));
  core::DeviceArray<double> d_ref_r_old(static_cast<std::size_t>(n_nodes));
  core::DeviceArray<double> d_ref_z_old(static_cast<std::size_t>(n_nodes));
  core::DeviceArray<double> d_cell_vol_initial_old(
      static_cast<std::size_t>(n_cells));
  CUDA_CHECK(cudaMemcpy(d_xr_old.data(), state.x_r.data(), node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_xz_old.data(), state.x_z.data(), node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_vol_old.data(), state.vol.data(), cell_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_ref_r_old.data(), state.x_r_reference.data(),
                        node_bytes, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_ref_z_old.data(), state.x_z_reference.data(),
                        node_bytes, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_cell_vol_initial_old.data(),
                        state.cell_vol_initial.data(), cell_bytes,
                        cudaMemcpyDeviceToDevice));

  const auto restore_source_and_reference = [&]() {
    CUDA_CHECK(cudaMemcpy(state.x_r.data(), d_xr_old.data(), node_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.x_z.data(), d_xz_old.data(), node_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.vol.data(), d_vol_old.data(), cell_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.x_r_reference.data(), d_ref_r_old.data(),
                          node_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.x_z_reference.data(), d_ref_z_old.data(),
                          node_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.cell_vol_initial.data(),
                          d_cell_vol_initial_old.data(), cell_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    state.mesh.node_r = state.x_r.data();
    state.mesh.node_z = state.x_z.data();
    state.mesh.recompute_geometry();
  };
  const auto restore_persistent_reference = [&]() {
    CUDA_CHECK(cudaMemcpy(state.x_r_reference.data(), d_ref_r_old.data(),
                          node_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.x_z_reference.data(), d_ref_z_old.data(),
                          node_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.cell_vol_initial.data(),
                          d_cell_vol_initial_old.data(), cell_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
  };

  const detail::ShellProtectedPatch patch = detail::build_shell_protected_patch(
      state, cfg, seed_mask, detail::shell_rezone_patch_layers());
  result.seed_cells = patch.seed_cells;
  result.shell_seed_cells = patch.shell_seed_cells;
  result.patch_cells = patch.n_patch_cells;
  result.support_cells = patch.n_support_cells;
  result.affected_nodes = patch.n_affected_nodes;
  result.energy_closure_cells = patch.n_energy_closure_cells;
  result.energy_collar_cells = patch.n_energy_collar_cells;
  result.first_energy_collar_cell = patch.first_energy_collar_cell;
  result.energy_collar_sample = patch.energy_collar_sample;
  result.closure_hard_frozen_nodes = patch.n_closure_hard_frozen_nodes;
  result.closure_hsrc_axis_pole_center =
      patch.n_closure_hsrc_axis_pole_center;
  result.closure_hsrc_core = patch.n_closure_hsrc_core;
  result.closure_hsrc_deref = patch.n_closure_hsrc_deref;
  result.shrink_iterations = patch.shrink_iterations;
  result.M_size_initial = patch.M_size_initial;
  result.M_size_final = patch.M_size_final;
  result.n_M_removed_for_closure = patch.n_M_removed_for_closure;
  result.closure_deref_node_id = patch.closure_deref_node_id;
  result.closure_deref_node_block = patch.closure_deref_node_block;
  result.closure_deref_node_local_i = patch.closure_deref_node_local_i;
  result.closure_deref_node_local_j = patch.closure_deref_node_local_j;
  result.closure_deref_node_subtype = patch.closure_deref_node_subtype;
  result.closure_removed_M_node_id = patch.closure_removed_M_node_id;
  result.closure_removed_M_node_block = patch.closure_removed_M_node_block;
  result.closure_removed_M_node_local_i =
      patch.closure_removed_M_node_local_i;
  result.closure_removed_M_node_local_j =
      patch.closure_removed_M_node_local_j;
  result.affected_boundary_outer_nodes = patch.n_affected_boundary_outer_nodes;
  result.closure_core_cells = patch.n_closure_core_cells;
  result.active_nodes = patch.n_active_nodes;
  result.boundary_nodes = patch.n_boundary_nodes;
  result.frozen_nodes = patch.n_frozen_nodes;
  if (!patch.applicable) {
    result.status = patch.status;
    restore_source_and_reference();
    return result;
  }

  std::vector<double> r_lag;
  std::vector<double> z_lag;
  state.x_r.copy_to_host(r_lag);
  state.x_z.copy_to_host(z_lag);
  result.q_min_before = detail::shell_patch_q_min_at(
      state, r_lag, z_lag, r_lag, z_lag, patch.cell_in_patch, 0.0,
      result.q_floor, nullptr);

  core::DeviceArray<std::uint8_t> d_rezone_active_node_mask(
      patch.node_rezone_active.size());
  d_rezone_active_node_mask.copy_from_host(patch.node_rezone_active);
  core::DeviceArray<std::uint8_t> d_support_active_cell_mask(
      patch.cell_support.size());
  d_support_active_cell_mask.copy_from_host(patch.cell_support);
  core::DeviceArray<std::uint8_t> d_energy_closure_cell_mask(
      patch.cell_energy_closure.size());
  d_energy_closure_cell_mask.copy_from_host(patch.cell_energy_closure);
  core::DeviceArray<std::uint8_t> d_patch_boundary_node_mask(
      patch.node_patch_boundary.size());
  d_patch_boundary_node_mask.copy_from_host(patch.node_patch_boundary);
  core::DeviceArray<std::uint8_t> d_frozen_node_mask(patch.frozen_node.size());
  d_frozen_node_mask.copy_from_host(patch.frozen_node);
  core::DeviceArray<std::uint8_t> d_inactive_cell_mask;
  const std::uint8_t* d_inactive_cell_mask_ptr = nullptr;
  if (std::any_of(patch.inactive_cell.begin(), patch.inactive_cell.end(),
                  [](const std::uint8_t v) { return v != 0U; })) {
    d_inactive_cell_mask.reset(patch.inactive_cell.size());
    d_inactive_cell_mask.copy_from_host(patch.inactive_cell);
    d_inactive_cell_mask_ptr = d_inactive_cell_mask.data();
  }

  const double omega_initial =
      std::clamp(cfg.numerics.ale.relaxation, 0.05, 1.0);
  const int max_winslow_iterations =
      std::max(0, cfg.numerics.ale.max_iterations);
  const double q_stop = detail::shell_rezone_q_stop(result.q_release);
  for (int iter = 0; iter < max_winslow_iterations;) {
    const MultiblockWinslowSmoothStats stats =
        cfg.numerics.ale.multiblock_cross_seam_rezone_enabled
            ? multiblock_winslow_smooth_with_seams_stats(
                  state, cfg, 1, omega_initial,
                  d_rezone_active_node_mask.data())
            : multiblock_winslow_smooth_with_stats(
                  state, cfg, 1, omega_initial,
                  d_rezone_active_node_mask.data());
    if (stats.accepted_iterations <= 0) {
      break;
    }
    result.winslow_iterations += stats.accepted_iterations;
    iter += stats.accepted_iterations;

    std::vector<double> r_iter;
    std::vector<double> z_iter;
    state.x_r.copy_to_host(r_iter);
    state.x_z.copy_to_host(z_iter);
    const double q_iter = detail::shell_patch_q_min_at(
        state, r_lag, z_lag, r_iter, z_iter, patch.cell_in_patch, 1.0,
        result.q_floor, nullptr);
    if (std::isfinite(q_iter) && q_iter >= q_stop) {
      break;
    }
  }

  std::vector<double> r_smooth;
  std::vector<double> z_smooth;
  state.x_r.copy_to_host(r_smooth);
  state.x_z.copy_to_host(z_smooth);

  TENRYU_ASSERT(mb.cell_orientation_sign.size() ==
                    static_cast<std::size_t>(n_cells),
                "shell protected rezone requires cell orientation signs");
  core::DeviceArray<int> d_cell_orientation_sign(mb.cell_orientation_sign.size());
  d_cell_orientation_sign.copy_from_host(mb.cell_orientation_sign);
  core::DeviceArray<std::uint8_t> d_cell_nverts;
  const std::uint8_t* d_cell_nverts_ptr = nullptr;
  if (tracking_reference_detail::multiblock_has_tri_cell_nverts(state)) {
    d_cell_nverts.reset(state.mesh.cell_nverts.size());
    d_cell_nverts.copy_from_host(state.mesh.cell_nverts);
    d_cell_nverts_ptr = d_cell_nverts.data();
  }
  if (state.mesh.multiblock_reverse_csr_node_offsets.size() !=
          static_cast<std::size_t>(n_nodes + 1) ||
      state.mesh.multiblock_reverse_csr_node_cells.empty() ||
      state.mesh.multiblock_reverse_csr_node_cells.size() !=
          state.mesh.multiblock_reverse_csr_node_corners.size()) {
    result.status = "missing_reverse_csr";
    restore_source_and_reference();
    return result;
  }
  core::DeviceArray<std::uint8_t> d_node_flags(state.mesh.topo.node_flags.size());
  d_node_flags.copy_from_host(state.mesh.topo.node_flags);
  core::DeviceArray<double> d_barrier_r(static_cast<std::size_t>(n_nodes));
  core::DeviceArray<double> d_barrier_z(static_cast<std::size_t>(n_nodes));
  center_patch_barrier::CenterPatchBarrierParams barrier_params;
  barrier_params.volume_floor_rel =
      std::max(cfg.numerics.ale.reference_volume_floor_rel, result.q_floor);
  barrier_params.jacobian_floor_rel =
      std::max(cfg.numerics.ale.reference_corner_j_floor_rel, result.q_floor);
  barrier_params.max_sweeps = std::max(
      1, detail::shell_rezone_env_int(
             "TENRYU_I1B_SHELL_REZONE_BARRIER_SWEEPS",
             barrier_params.max_sweeps));
  barrier_params.constrain_cap_apex = mb.has_trifan_cap;
  if (mb.has_trifan_cap) {
    barrier_params.cap_apex_node_id = mesh::mesh_topo_cap_apex_node_id(mb);
  }
  center_patch_barrier::CenterPatchBarrierResult barrier_result =
      center_patch_barrier::optimize_center_patch_phi_barrier(
          d_barrier_r.data(),
          d_barrier_z.data(),
          state.x_r.data(),
          state.x_z.data(),
          d_xr_old.data(),
          d_xz_old.data(),
          d_ref_r_old.data(),
          d_ref_z_old.data(),
          state.mesh.multiblock_reverse_csr_node_offsets.data(),
          state.mesh.multiblock_reverse_csr_node_cells.data(),
          state.mesh.multiblock_reverse_csr_node_corners.data(),
          state.mesh.multiblock_cell_node_csr_offsets.data(),
          state.mesh.multiblock_cell_node_csr_indices.data(),
          d_cell_orientation_sign.data(),
          d_cell_nverts_ptr,
          d_node_flags.data(),
          d_rezone_active_node_mask.data(),
          d_patch_boundary_node_mask.data(),
          n_nodes,
          barrier_params,
          d_frozen_node_mask.data());
  result.barrier_sweeps = barrier_result.sweeps;
  CUDA_CHECK(cudaMemcpy(state.x_r.data(), d_barrier_r.data(), node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(state.x_z.data(), d_barrier_z.data(), node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  state.x_r.copy_to_host(r_smooth);
  state.x_z.copy_to_host(z_smooth);

  std::vector<double> delta_r(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> delta_z(static_cast<std::size_t>(n_nodes), 0.0);
  for (int n = 0; n < n_nodes; ++n) {
    const auto idx = static_cast<std::size_t>(n);
    delta_r[idx] = r_smooth[idx] - r_lag[idx];
    delta_z[idx] = z_smooth[idx] - z_lag[idx];
    const double d = std::hypot(delta_r[idx], delta_z[idx]);
    if (patch.node_rezone_active[idx] != 0U) {
      result.max_active_delta = std::max(result.max_active_delta, d);
    }
  }
  result.max_frozen_delta = detail::shell_rezone_max_masked_delta(
      r_lag, z_lag, r_smooth, z_smooth, patch.frozen_node);
  if (!(result.max_active_delta > 0.0)) {
    result.status = "zero_active_motion";
    restore_source_and_reference();
    return result;
  }
  const double frozen_tol =
      detail::shell_rezone_env_double("TENRYU_I1B_SHELL_REZONE_FROZEN_TOL",
                                      0.0);
  if (result.max_frozen_delta > frozen_tol) {
    result.status = "frozen_node_moved";
    restore_source_and_reference();
    return result;
  }

  core::DeviceArray<double> d_delta_r(static_cast<std::size_t>(n_nodes));
  core::DeviceArray<double> d_delta_z(static_cast<std::size_t>(n_nodes));
  d_delta_r.copy_from_host(delta_r);
  d_delta_z.copy_from_host(delta_z);
  core::DeviceArray<int> d_cell_id_stable(mb.cell_id_stable.size());
  d_cell_id_stable.copy_from_host(mb.cell_id_stable);
  mesh::CandidateMeshAdmissibilityFloors floors;
  floors.volume_rel =
      std::max(cfg.numerics.ale.reference_volume_floor_rel, result.q_floor);
  floors.corner_j_rel =
      std::max(cfg.numerics.ale.reference_corner_j_floor_rel, result.q_floor);
  floors.gauss_j_rel =
      std::max(cfg.numerics.ale.reference_gauss_j_floor_rel, result.q_floor);
  const auto& pc = state.central_pseudo_core;
  const int* d_macro_boundary_nodes =
      central_pseudo_core::active(state) && !pc.d_boundary_nodes_ordered.empty()
          ? pc.d_boundary_nodes_ordered.data()
          : nullptr;
  const int n_macro_boundary_nodes =
      central_pseudo_core::active(state)
          ? static_cast<int>(pc.boundary_nodes_ordered.size())
          : 0;
  const mesh::LineSearchResult ls =
      mesh::linesearch_largest_admissible_sigma_csr(
          d_xr_old.data(),
          d_xz_old.data(),
          d_delta_r.data(),
          d_delta_z.data(),
          1.0,
          0.0,
          cfg.numerics.ale.reference_linesearch_max_iters,
          n_cells,
          state.mesh.corner_stride,
          state.mesh.multiblock_cell_node_csr_offsets.data(),
          state.mesh.multiblock_cell_node_csr_indices.data(),
          d_cell_id_stable.data(),
          d_cell_orientation_sign.data(),
          floors,
          d_cell_nverts_ptr,
          d_inactive_cell_mask_ptr,
          d_macro_boundary_nodes,
          n_macro_boundary_nodes,
          state.x_r_reference.size() == state.x_r.size() &&
                  state.x_z_reference.size() == state.x_z.size()
              ? state.x_r_reference.data()
              : nullptr,
          state.x_r_reference.size() == state.x_r.size() &&
                  state.x_z_reference.size() == state.x_z.size()
              ? state.x_z_reference.data()
              : nullptr);
  result.sigma_accepted = ls.sigma_accepted;
  result.line_search_iters = ls.iters_used;
  result.first_bad_cell = ls.quality.first_bad_cell;
  if (!(result.sigma_accepted > 0.0)) {
    result.status = "line_search_zero";
    restore_source_and_reference();
    return result;
  }
  const double q_stop_endpoint = detail::shell_rezone_q_stop(result.q_release);
  const double q_at_sigma = detail::shell_patch_q_min_at(
      state, r_lag, z_lag, r_smooth, z_smooth, patch.cell_in_patch,
      result.sigma_accepted, result.q_floor, nullptr);
  if (std::isfinite(q_at_sigma) && q_at_sigma > q_stop_endpoint) {
    double lo = 0.0;
    double hi = result.sigma_accepted;
    for (int it = 0; it < 40; ++it) {
      const double mid = 0.5 * (lo + hi);
      const double q_mid = detail::shell_patch_q_min_at(
          state, r_lag, z_lag, r_smooth, z_smooth, patch.cell_in_patch, mid,
          result.q_floor, nullptr);
      if (std::isfinite(q_mid) && q_mid >= q_stop_endpoint) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    result.sigma_accepted = hi;
  }

  std::vector<double> r_ref = r_lag;
  std::vector<double> z_ref = z_lag;
  for (int n = 0; n < n_nodes; ++n) {
    const auto idx = static_cast<std::size_t>(n);
    r_ref[idx] = r_lag[idx] + result.sigma_accepted * delta_r[idx];
    z_ref[idx] = z_lag[idx] + result.sigma_accepted * delta_z[idx];
  }
  core::DeviceArray<double> d_target_r(static_cast<std::size_t>(n_nodes));
  core::DeviceArray<double> d_target_z(static_cast<std::size_t>(n_nodes));
  d_target_r.copy_from_host(r_ref);
  d_target_z.copy_from_host(z_ref);
  core::DeviceArray<std::uint8_t> core_freeze_frozen_nodes;
  const auto core_freeze_diag =
      core_freeze::restore_target_if_enabled(state,
                                             cfg,
                                             d_target_r.data(),
                                             d_target_z.data(),
                                             d_xr_old.data(),
                                             d_xz_old.data(),
                                             false,
                                             "shell_protected_rezone",
                                             cfg.numerics.ale
                                                     .core_freeze_skip_velocity_projection
                                                 ? &core_freeze_frozen_nodes
                                                 : nullptr);
  if (core_freeze_diag.enabled && core_freeze_diag.frozen_node_count > 0) {
    d_target_r.copy_to_host(r_ref);
    d_target_z.copy_to_host(z_ref);
  }
  result.q_min_after = detail::shell_patch_q_min_at(
      state, r_lag, z_lag, r_ref, z_ref, patch.cell_in_patch, 1.0,
      result.q_floor, &result.first_bad_cell);
  result.q_min_path = detail::shell_patch_q_min_path(
      state, r_lag, z_lag, r_ref, z_ref, patch.cell_in_patch,
      result.q_floor);
  result.q_release_restored =
      std::isfinite(result.q_min_after) &&
      result.q_min_after >= result.q_release;

  const mesh::PathAdmissibilityResult rezone_path =
      mesh::evaluate_path_admissibility(state, cfg, d_xr_old.data(),
                                        d_xz_old.data(), d_target_r.data(),
                                        d_target_z.data(), result.q_floor,
                                        nullptr,
                                        state.x_r_reference.size() ==
                                                state.x_r.size()
                                            ? state.x_r_reference.data()
                                            : nullptr,
                                        state.x_z_reference.size() ==
                                                state.x_z.size()
                                            ? state.x_z_reference.data()
                                            : nullptr);
  result.rezone_path_valid =
      rezone_path.first_failing_cell == -1 ||
      !(rezone_path.first_failing_lambda < 1.0);
  result.endpoint_valid_after = rezone_path.first_failing_cell == -1;
  if (!result.rezone_path_valid) {
    result.status = "rezone_path_invalid";
    result.first_bad_cell = rezone_path.first_failing_cell;
    restore_source_and_reference();
    return result;
  }
  const detail::ShellMomentumTransportSupport momentum_support =
      detail::build_shell_momentum_transport_support(
          state, patch, r_lag, z_lag, r_ref, z_ref);
  if (momentum_support.cell_mask.size() != static_cast<std::size_t>(n_cells)) {
    result.status = "bad_momentum_transport_support";
    restore_source_and_reference();
    return result;
  }
  result.momentum_transport_cells = momentum_support.n_cells;
  result.momentum_transport_collar_cells = momentum_support.n_collar_cells;
  result.momentum_transport_iterations = momentum_support.n_iterations;
  core::DeviceArray<std::uint8_t> d_corner_transport_cell_mask(
      momentum_support.cell_mask.size());
  d_corner_transport_cell_mask.copy_from_host(momentum_support.cell_mask);
  const std::vector<std::uint8_t> allowed_velocity_nodes =
      detail::shell_rezone_cell_vertices_mask(state, momentum_support.cell_mask);
  if (allowed_velocity_nodes.size() != static_cast<std::size_t>(n_nodes)) {
    result.status = "bad_velocity_restore_mask";
    restore_source_and_reference();
    return result;
  }
  core::DeviceArray<std::uint8_t> d_allowed_velocity_node_mask(
      allowed_velocity_nodes.size());
  d_allowed_velocity_node_mask.copy_from_host(allowed_velocity_nodes);

  const std::vector<double> vol_ref =
      tracking_reference_detail::compute_multiblock_reference_volumes(
          state, cfg, r_ref, z_ref);
  state.x_r_reference.copy_from_host(r_ref);
  state.x_z_reference.copy_from_host(z_ref);
  state.cell_vol_initial.copy_from_host(vol_ref);
  CUDA_CHECK(cudaMemcpy(state.x_r.data(), state.x_r_reference.data(),
                        node_bytes, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(state.x_z.data(), state.x_z_reference.data(),
                        node_bytes, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  state.mesh.node_r = state.x_r.data();
  state.mesh.node_z = state.x_z.data();
  state.mesh.recompute_geometry();
  state.vol = state.mesh.cell_vol;

  CUDA_CHECK(cudaMemcpy(state.x_r.data(), d_xr_old.data(), node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(state.x_z.data(), d_xz_old.data(), node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(state.vol.data(), d_vol_old.data(), cell_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  state.mesh.node_r = state.x_r.data();
  state.mesh.node_z = state.x_z.data();

  core::Config remap_cfg = cfg;
  remap_cfg.numerics.ale.conservative_remap_enabled = true;
  remap_cfg.numerics.ale.conservative_remap_target = "reference";
  remap_cfg.numerics.ale.swept_volume_sign_fixed = true;
  if (!state.corner_mass_initialized ||
      state.corner_mass.size() != static_cast<std::size_t>(n_cells) * 4U) {
    result.status = "missing_corner_mass_basis";
    restore_source_and_reference();
    return result;
  }
  core::DeviceArray<double> d_vr_replay_old(static_cast<std::size_t>(n_nodes));
  core::DeviceArray<double> d_vz_replay_old(static_cast<std::size_t>(n_nodes));
  CUDA_CHECK(cudaMemcpy(d_vr_replay_old.data(), state.v_r.data(), node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_vz_replay_old.data(), state.v_z.data(), node_bytes,
                        cudaMemcpyDeviceToDevice));
  const detail::ShellNodeKinematicSnapshot node_diag_before =
      detail::shell_rezone_capture_node_kinematics(state);
  AleRemap2DRZOverrides remap_overrides;
  remap_overrides.force_total_energy_remap = true;
  remap_overrides.force_optionb_velocity_authority = true;
  remap_overrides.force_optionb_coherent = true;
  remap_overrides.force_optionb_coherent_transport = true;
  remap_overrides.force_optionb_coherent_rerecover = true;
  remap_overrides.allow_polar_shell_derefine = true;
  remap_overrides.force_optionb_corner_mass_install = true;
  remap_overrides.collect_replay_diagnostics = true;
  remap_overrides.near_massless_velocity_mass_floor = std::max(
      1.0e-300,
      detail::shell_rezone_env_double(
          "TENRYU_I1B_SHELL_REZONE_MASSLESS_NODE_FLOOR_G", 1.0e-300));
  remap_overrides.active_cell_mask = d_support_active_cell_mask.data();
  remap_overrides.corner_transport_cell_mask =
      d_corner_transport_cell_mask.data();
  remap_overrides.energy_closure_cell_mask =
      d_energy_closure_cell_mask.data();
  remap_overrides.active_node_velocity_mask =
      d_allowed_velocity_node_mask.data();
  const std::uint8_t* core_freeze_velocity_mask =
      core_freeze_frozen_nodes.size() == static_cast<std::size_t>(n_nodes)
          ? core_freeze_frozen_nodes.data()
          : nullptr;
  RollbackGuard remap_snapshot;
  remap_snapshot.capture(state, nullptr);
  const AleRemap2DRZResult remap_result =
      ale_remap_2d_rz(state,
                      remap_cfg,
                      eos_ctx,
                      dt_hydro_used,
                      core_freeze_velocity_mask,
                      remap_overrides);
  result.remap_applied = remap_result.applied;
  result.succeeded = remap_result.applied;
  result.dE_floor_deposit = remap_result.E_eint_floor_deposit;
  result.dE_active_floor = remap_result.E_active_floor;
  result.active_floor_mass_delta = remap_result.mass_floor_delta;
  result.n_eint_floor_hits = remap_result.n_eint_floor_hits;
  result.n_active_floor_hits = remap_result.n_active_floor_hits;
  if (remap_result.applied &&
      (remap_result.n_eint_floor_hits > 0 ||
       remap_result.E_eint_floor_deposit > 0.0)) {
    result.succeeded = false;
    result.remap_applied = false;
    result.status = "closure_floor_violation";
    remap_snapshot.restore(state, nullptr);
    restore_source_and_reference();
    return result;
  }
  result.transport_momentum_valid =
      remap_result.replay_transport_momentum_valid;
  result.transport_pr = remap_result.replay_transport_pr;
  result.transport_pz = remap_result.replay_transport_pz;
  result.R_pi_r =
      remap_result.replay_raw_pi1_r - remap_result.replay_raw_pi0_r;
  result.R_pi_z =
      remap_result.replay_raw_pi1_z - remap_result.replay_raw_pi0_z;
  result.R_assm_r = remap_result.replay_assm_residual_r;
  result.R_assm_z = remap_result.replay_assm_residual_z;
  result.R_rec_r = remap_result.replay_rec_residual_r;
  result.R_rec_z = remap_result.replay_rec_residual_z;
  result.discarded_dual_faces = remap_result.replay_discarded_dual_faces;
  result.discarded_dual_mass = remap_result.replay_discarded_dual_mass;
  result.discarded_dual_pi_r = remap_result.replay_discarded_dual_pi_r;
  result.discarded_dual_pi_z = remap_result.replay_discarded_dual_pi_z;
  if (result.succeeded) {
    detail::restore_velocity_outside_mask_kernel<<<blocks_nodes, 256>>>(
        state.v_r.data(),
        state.v_z.data(),
        d_vr_replay_old.data(),
        d_vz_replay_old.data(),
        d_allowed_velocity_node_mask.data(),
        n_nodes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const detail::ShellNodeKinematicSnapshot node_diag_after =
        detail::shell_rezone_capture_node_kinematics(state);
    detail::shell_rezone_apply_boundary_node_diagnostics(
        patch, node_diag_before, node_diag_after, allowed_velocity_nodes,
        state.mesh.topo.node_flags, result);
    state.ale_rezoned = true;
    ++state.ale_rezone_invocations;
    state.ale_last_applied_step = state.step;
    state.holo_ale_invalidated = true;
    result.status = "ok";
  } else {
    result.status = "remap_not_applied";
    restore_source_and_reference();
    return result;
  }
  if (reduction != nullptr) {
    double sums[26] = {
        result.dPr_boundary,
        result.dPz_boundary,
        result.dPr_active,
        result.dPz_active,
        result.dE_floor_deposit,
        result.dE_active_floor,
        result.active_floor_mass_delta,
        static_cast<double>(result.n_eint_floor_hits),
        static_cast<double>(result.n_active_floor_hits),
        result.p_scale,
        static_cast<double>(result.n_ext_cell_cmass_changed),
        static_cast<double>(result.n_ext_node_vel_changed),
        static_cast<double>(result.n_vel_changed_outside_velmask),
        result.transport_momentum_valid ? result.transport_pr : 0.0,
        result.transport_momentum_valid ? result.transport_pz : 0.0,
        result.transport_momentum_valid ? 1.0 : 0.0,
        result.R_pi_r,
        result.R_pi_z,
        result.R_assm_r,
        result.R_assm_z,
        result.R_rec_r,
        result.R_rec_z,
        static_cast<double>(result.discarded_dual_faces),
        result.discarded_dual_mass,
        result.discarded_dual_pi_r,
        result.discarded_dual_pi_z};
    reduction->allreduce_sum(sums, 26);
    result.dPr_boundary = sums[0];
    result.dPz_boundary = sums[1];
    result.dPr_active = sums[2];
    result.dPz_active = sums[3];
    result.dE_floor_deposit = sums[4];
    result.dE_active_floor = sums[5];
    result.active_floor_mass_delta = sums[6];
    result.n_eint_floor_hits = static_cast<int>(std::llround(sums[7]));
    result.n_active_floor_hits = static_cast<int>(std::llround(sums[8]));
    result.p_scale = sums[9];
    result.n_ext_cell_cmass_changed =
        static_cast<int>(std::llround(sums[10]));
    result.n_ext_node_vel_changed =
        static_cast<int>(std::llround(sums[11]));
    result.n_vel_changed_outside_velmask =
        static_cast<int>(std::llround(sums[12]));
    result.transport_pr = sums[13];
    result.transport_pz = sums[14];
    result.transport_momentum_valid = sums[15] > 0.0;
    result.R_pi_r = sums[16];
    result.R_pi_z = sums[17];
    result.R_assm_r = sums[18];
    result.R_assm_z = sums[19];
    result.R_rec_r = sums[20];
    result.R_rec_z = sums[21];
    result.discarded_dual_faces = static_cast<int>(std::llround(sums[22]));
    result.discarded_dual_mass = sums[23];
    result.discarded_dual_pi_r = sums[24];
    result.discarded_dual_pi_z = sums[25];
    double maxima[6] = {result.max_boundary_dx,
                        result.max_boundary_dvr,
                        result.max_boundary_dvz,
                        result.max_boundary_dm,
                        result.max_ext_cell_cmass_delta,
                        result.max_ext_node_vel_delta};
    reduction->allreduce_max(maxima, 6);
    result.max_boundary_dx = maxima[0];
    result.max_boundary_dvr = maxima[1];
    result.max_boundary_dvz = maxima[2];
    result.max_boundary_dm = maxima[3];
    result.max_ext_cell_cmass_delta = maxima[4];
    result.max_ext_node_vel_delta = maxima[5];
  }
  restore_persistent_reference();
  return result;
}

bool maybe_run_shell_rezone_replay_probe(
    core::State& state,
    const core::Config& cfg,
    const mesh::MeshForecast& forecast,
    const double* d_xr_source,
    const double* d_xz_source,
    const double* d_v_r,
    const double* d_v_z,
    const double path_dt,
    const char* stage,
    const HydroEOSContext* eos_ctx,
    const parallel::Reduction* reduction) {
  static bool replay_fired = false;
  const bool post_stage =
      stage != nullptr && std::string(stage) == "post_corrector_commit";
  const bool replay_env =
      detail::shell_rezone_env_enabled("TENRYU_I1B_SHELL_REZONE_REPLAY");
  const bool replay_mode =
      replay_env && !replay_fired &&
      detail::shell_rezone_replay_stage_selected(stage);
  const bool commit_mode =
      !replay_env &&
      detail::shell_rezone_env_enabled("TENRYU_I1B_SHELL_SUBCYCLE") &&
      post_stage;
  if (!replay_mode && !commit_mode) {
    return false;
  }
  const double q_floor =
      std::max(0.0, cfg.numerics.ale.path_admissibility_floor);
  const double q_current_forecast =
      post_stage ? forecast.q_min_end : forecast.q_min_now;
  if (forecast.endpoint_valid == 0 || forecast.seed_count <= 0 ||
      !(q_current_forecast >= q_floor &&
        q_current_forecast <= forecast.q_warn)) {
    return false;
  }
  const double source_delta =
      detail::shell_rezone_source_current_delta(state, d_xr_source, d_xz_source);
  const double source_tol =
      detail::shell_rezone_env_double(
          "TENRYU_I1B_SHELL_REZONE_REPLAY_SOURCE_TOL", 0.0);
  if (source_delta > source_tol) {
    if (commit_mode) {
      core::log_info(
          std::string("[shell_subcycle] step=") + std::to_string(state.step) +
          " q_min_before=" +
          detail::format_scientific(std::numeric_limits<double>::quiet_NaN()) +
          " q_min_after=" +
          detail::format_scientific(std::numeric_limits<double>::quiet_NaN()) +
          " committed=0 gate=rollback dM_rel=" +
          detail::format_scientific(std::numeric_limits<double>::quiet_NaN()) +
          " dPr=" +
          detail::format_scientific(std::numeric_limits<double>::quiet_NaN()) +
          " dPz=" +
          detail::format_scientific(std::numeric_limits<double>::quiet_NaN()) +
          " dE_tot_rel=" +
          detail::format_scientific(std::numeric_limits<double>::quiet_NaN()) +
          " cfl_dt=" +
          detail::format_scientific(std::numeric_limits<double>::quiet_NaN()) +
          " max_node_speed=" +
          detail::format_scientific(std::numeric_limits<double>::quiet_NaN()) +
          " min_node_mass=" +
          detail::format_scientific(std::numeric_limits<double>::quiet_NaN()) +
          " reason=source_current_mismatch source_current_delta=" +
          detail::format_scientific(source_delta) +
          " source_tol=" + detail::format_scientific(source_tol));
    } else {
      core::log_warning(
          std::string("[i1b_shell_rezone_replay_skip] stage=") +
          (stage != nullptr ? stage : "") +
          " reason=source_current_mismatch max_delta=" +
          detail::format_scientific(source_delta) +
          " tol=" + detail::format_scientific(source_tol));
    }
    return false;
  }
  if (replay_mode) {
    replay_fired = true;
  }

  RollbackGuard snapshot;
  snapshot.capture(state, nullptr);
  detail::ShellSubcyclePostRezoneAudit pre_rezone_audit{};
  if (commit_mode) {
    pre_rezone_audit =
        detail::audit_shell_subcycle_post_rezone(state, cfg, reduction);
  }
  const detail::ReplayTotals before =
      detail::shell_rezone_capture_totals(state, reduction);
  ShellProtectedRezoneResult result = shell_protected_rezone(
      state, cfg, forecast.seed_mask, eos_ctx, path_dt, reduction);
  if (result.remap_applied && d_v_r != nullptr && d_v_z != nullptr &&
      path_dt > 0.0 && std::isfinite(path_dt)) {
    const mesh::PathAdmissibilityResult path_after =
        mesh::evaluate_path_admissibility(state, cfg, state.x_r.data(),
                                          state.x_z.data(), d_v_r, d_v_z,
                                          path_dt, q_floor, nullptr);
    const mesh::MeshForecast forecast_after =
        mesh::evaluate_mesh_forecast(state, cfg, state.x_r.data(),
                                     state.x_z.data(), d_v_r, d_v_z,
                                     path_dt, q_floor, forecast.q_warn,
                                     path_after, nullptr);
    result.tau_zero_after = forecast_after.tau_zero;
    result.endpoint_valid_after = forecast_after.endpoint_valid != 0;
  }
  const detail::ReplayTotals after =
      detail::shell_rezone_capture_totals(state, reduction);
  result.dM = after.mass - before.mass;
  result.dPr = after.momentum_r - before.momentum_r;
  result.dPz = after.momentum_z - before.momentum_z;
  result.dE_internal = after.energy_internal - before.energy_internal;
  result.dKE = after.energy_kin - before.energy_kin;
  result.dE = after.energy_total - before.energy_total;
  result.dE_int_transport_residual =
      result.dE_internal - result.dE_floor_deposit - result.dE_active_floor;
  result.dKE_reconstruction_residual = result.dKE;
  result.dE_unexplained =
      result.dE - result.dE_int_transport_residual -
      result.dKE_reconstruction_residual - result.dE_floor_deposit -
      result.dE_active_floor;
  result.R_E = result.dE;
  const double dPr_other =
      result.dPr - result.dPr_boundary - result.dPr_active;
  const double dPz_other =
      result.dPz - result.dPz_boundary - result.dPz_active;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double dPr_transport_only =
      result.transport_momentum_valid ? result.transport_pr - before.momentum_r
                                      : nan;
  const double dPz_transport_only =
      result.transport_momentum_valid ? result.transport_pz - before.momentum_z
                                      : nan;
  const double p_scale = result.p_scale;
  const double dPr_over_p_scale =
      p_scale > 0.0 ? result.dPr / p_scale : nan;
  const double dPz_over_p_scale =
      p_scale > 0.0 ? result.dPz / p_scale : nan;

  if (commit_mode) {
    bool rollback = true;
    std::string gate = "rollback";
    std::string reason = "shell_rezone_status=" + result.status;
    detail::ShellSubcyclePostRezoneAudit audit{};
    const double dM_rel = detail::relative_delta(result.dM, before.mass);
    const double dE_rel =
        detail::relative_delta(result.dE, before.energy_total);
    const double conservation_tol = std::max(
        0.0, detail::shell_rezone_env_double(
                 "TENRYU_I1B_SHELL_SUBCYCLE_CONSERVATION_TOL", 1.0e-10));
    const bool q_improved =
        std::isfinite(result.q_min_before) &&
        std::isfinite(result.q_min_after) &&
        result.q_min_after > result.q_min_before && result.q_min_after > 0.0;
    const bool conservation_ok =
        std::isfinite(dM_rel) && std::abs(dM_rel) <= conservation_tol &&
        std::isfinite(dE_rel) && std::abs(dE_rel) <= conservation_tol &&
        std::isfinite(dPr_over_p_scale) &&
        std::abs(dPr_over_p_scale) <= conservation_tol &&
        std::isfinite(dPz_over_p_scale) &&
        std::abs(dPz_over_p_scale) <= conservation_tol;
    if (!pre_rezone_audit.passed) {
      reason = "pre_rezone_audit=" + pre_rezone_audit.reason;
    } else if (!(result.status == "ok" && result.succeeded &&
                 result.remap_applied && result.rezone_path_valid)) {
      reason = "shell_rezone_status=" + result.status;
    } else if (!q_improved) {
      reason = "q_not_improved";
    } else if (!conservation_ok) {
      reason = "conservation_not_roundoff";
    } else {
      int rho_clamp_count = 0;
      const tenryu::coupling::HydroStepResult refresh_result =
          tenryu::hydro::refresh_geometry_and_density(
              state,
              cfg,
              tenryu::coupling::HydroFailureStage::PostCorrector,
              &rho_clamp_count,
              nullptr);
      if (refresh_result.retry_required) {
        reason = "post_rezone_geometry_refresh";
      } else {
        tenryu::hydro::ensure_hourglass_subzonal_masses_2d(state, cfg, false);
        csr_optionb_canonicalize_corner_mass_basis(state, cfg);
        tenryu::hydro::reset_volume_rate_cfl_history_after_ale(state);
        detail::refresh_shell_subcycle_void_mask_after_rezone(state, cfg);
        audit = detail::audit_shell_subcycle_post_rezone(
            state, cfg, reduction, &pre_rezone_audit);
        if (audit.passed) {
          rollback = false;
          gate = "pass";
          reason = "ok";
        } else {
          reason = "post_rezone_audit=" + audit.reason;
        }
      }
    }
    if (rollback) {
      snapshot.restore(state, nullptr);
    }
    core::log_info(
        std::string("[shell_subcycle] step=") + std::to_string(state.step) +
        " q_min_before=" + detail::format_scientific(result.q_min_before) +
        " q_min_after=" + detail::format_scientific(result.q_min_after) +
        " committed=" + (rollback ? std::string("0") : std::string("1")) +
        " gate=" + gate +
        " dM_rel=" +
        detail::format_scientific(dM_rel) +
        " dPr=" + detail::format_scientific(result.dPr) +
        " dPz=" + detail::format_scientific(result.dPz) +
        " dE_tot_rel=" +
        detail::format_scientific(dE_rel) +
        " dPr_over_p_scale=" +
        detail::format_scientific(dPr_over_p_scale) +
        " dPz_over_p_scale=" +
        detail::format_scientific(dPz_over_p_scale) +
        " cfl_dt_before=" +
        detail::format_scientific(pre_rezone_audit.cfl_dt) +
        " cfl_dt=" + detail::format_scientific(audit.cfl_dt) +
        " max_node_speed_before=" +
        detail::format_scientific(pre_rezone_audit.max_node_speed) +
        " max_node_speed=" + detail::format_scientific(audit.max_node_speed) +
        " min_node_mass=" + detail::format_scientific(audit.min_node_mass) +
        " reason=" + reason +
        " status=" + result.status +
        " q_min_forecast_current=" +
        detail::format_scientific(q_current_forecast) +
        " q_warn=" + detail::format_scientific(forecast.q_warn) +
        " restored_q_release=" + (result.q_release_restored ? "1" : "0") +
        " rezone_path_valid=" + (result.rezone_path_valid ? "1" : "0") +
        " first_bad_cell=" + std::to_string(audit.first_bad_cell) +
        " first_bad_node=" + std::to_string(audit.first_bad_node));
    return !rollback;
  }

  std::ostringstream oss;
  oss << "[i1b_shell_rezone_replay]"
      << " step=" << state.step
      << " t=" << detail::format_scientific(state.t)
      << " stage=" << (stage != nullptr ? stage : "")
      << " path_dt=" << detail::format_scientific(path_dt)
      << " status=" << result.status
      << " attempted=" << (result.attempted ? 1 : 0)
      << " remap_applied=" << (result.remap_applied ? 1 : 0)
      << " q_floor=" << detail::format_scientific(result.q_floor)
      << " q_warn=" << detail::format_scientific(forecast.q_warn)
      << " q_release=" << detail::format_scientific(result.q_release)
      << " q_min_forecast_current="
      << detail::format_scientific(q_current_forecast)
      << " q_min_before=" << detail::format_scientific(result.q_min_before)
      << " q_min_after=" << detail::format_scientific(result.q_min_after)
      << " q_min_path=" << detail::format_scientific(result.q_min_path)
      << " restored_q_release=" << (result.q_release_restored ? 1 : 0)
      << " tau_zero_before=" << detail::format_scientific(forecast.tau_zero)
      << " tau_zero_after=" << detail::format_scientific(result.tau_zero_after)
      << " endpoint_valid_before=" << forecast.endpoint_valid
      << " endpoint_valid_after=" << (result.endpoint_valid_after ? 1 : 0)
      << " rezone_path_valid=" << (result.rezone_path_valid ? 1 : 0)
      << " seed_cells=" << result.seed_cells
      << " shell_seed_cells=" << result.shell_seed_cells
      << " patch_cells=" << result.patch_cells
      << " support_cells_G=" << result.support_cells
      << " support_cells_Cpi=" << result.momentum_transport_cells
      << " momentum_collar_cells="
      << result.momentum_transport_collar_cells
      << " momentum_support_iterations="
      << result.momentum_transport_iterations
      << " affected_nodes_A=" << result.affected_nodes
      << " energy_closure_cells_CE=" << result.energy_closure_cells
      << " energy_collar_cells=" << result.energy_collar_cells
      << " first_energy_collar_cell=" << result.first_energy_collar_cell
      << " energy_collar_sample0=" << result.energy_collar_sample[0]
      << " energy_collar_sample1=" << result.energy_collar_sample[1]
      << " energy_collar_sample2=" << result.energy_collar_sample[2]
      << " energy_collar_sample3=" << result.energy_collar_sample[3]
      << " energy_collar_sample4=" << result.energy_collar_sample[4]
      << " energy_collar_sample5=" << result.energy_collar_sample[5]
      << " energy_collar_sample6=" << result.energy_collar_sample[6]
      << " energy_collar_sample7=" << result.energy_collar_sample[7]
      << " closure_hard_frozen_nodes="
      << result.closure_hard_frozen_nodes
      << " n_Hsrc_axis_pole_center="
      << result.closure_hsrc_axis_pole_center
      << " n_Hsrc_core=" << result.closure_hsrc_core
      << " n_Hsrc_deref=" << result.closure_hsrc_deref
      << " shrink_iterations=" << result.shrink_iterations
      << " M_size_initial=" << result.M_size_initial
      << " M_size_final=" << result.M_size_final
      << " n_M_removed_for_closure=" << result.n_M_removed_for_closure
      << " deref_H0_node=" << result.closure_deref_node_id[0]
      << " deref_H0_block=" << result.closure_deref_node_block[0]
      << " deref_H0_i=" << result.closure_deref_node_local_i[0]
      << " deref_H0_j=" << result.closure_deref_node_local_j[0]
      << " deref_H0_subtype="
      << (result.closure_deref_node_subtype[0] == 1
              ? "boundary_node"
              : (result.closure_deref_node_subtype[0] == 2
                     ? "inactive_member"
                     : "none"))
      << " deref_H1_node=" << result.closure_deref_node_id[1]
      << " deref_H1_block=" << result.closure_deref_node_block[1]
      << " deref_H1_i=" << result.closure_deref_node_local_i[1]
      << " deref_H1_j=" << result.closure_deref_node_local_j[1]
      << " deref_H1_subtype="
      << (result.closure_deref_node_subtype[1] == 1
              ? "boundary_node"
              : (result.closure_deref_node_subtype[1] == 2
                     ? "inactive_member"
                     : "none"))
      << " deref_H2_node=" << result.closure_deref_node_id[2]
      << " deref_H2_block=" << result.closure_deref_node_block[2]
      << " deref_H2_i=" << result.closure_deref_node_local_i[2]
      << " deref_H2_j=" << result.closure_deref_node_local_j[2]
      << " deref_H2_subtype="
      << (result.closure_deref_node_subtype[2] == 1
              ? "boundary_node"
              : (result.closure_deref_node_subtype[2] == 2
                     ? "inactive_member"
                     : "none"))
      << " deref_H3_node=" << result.closure_deref_node_id[3]
      << " deref_H3_block=" << result.closure_deref_node_block[3]
      << " deref_H3_i=" << result.closure_deref_node_local_i[3]
      << " deref_H3_j=" << result.closure_deref_node_local_j[3]
      << " deref_H3_subtype="
      << (result.closure_deref_node_subtype[3] == 1
              ? "boundary_node"
              : (result.closure_deref_node_subtype[3] == 2
                     ? "inactive_member"
                     : "none"))
      << " removed_M0_node=" << result.closure_removed_M_node_id[0]
      << " removed_M0_block=" << result.closure_removed_M_node_block[0]
      << " removed_M0_i=" << result.closure_removed_M_node_local_i[0]
      << " removed_M0_j=" << result.closure_removed_M_node_local_j[0]
      << " removed_M1_node=" << result.closure_removed_M_node_id[1]
      << " removed_M1_block=" << result.closure_removed_M_node_block[1]
      << " removed_M1_i=" << result.closure_removed_M_node_local_i[1]
      << " removed_M1_j=" << result.closure_removed_M_node_local_j[1]
      << " removed_M2_node=" << result.closure_removed_M_node_id[2]
      << " removed_M2_block=" << result.closure_removed_M_node_block[2]
      << " removed_M2_i=" << result.closure_removed_M_node_local_i[2]
      << " removed_M2_j=" << result.closure_removed_M_node_local_j[2]
      << " removed_M3_node=" << result.closure_removed_M_node_id[3]
      << " removed_M3_block=" << result.closure_removed_M_node_block[3]
      << " removed_M3_i=" << result.closure_removed_M_node_local_i[3]
      << " removed_M3_j=" << result.closure_removed_M_node_local_j[3]
      << " removed_M4_node=" << result.closure_removed_M_node_id[4]
      << " removed_M4_block=" << result.closure_removed_M_node_block[4]
      << " removed_M4_i=" << result.closure_removed_M_node_local_i[4]
      << " removed_M4_j=" << result.closure_removed_M_node_local_j[4]
      << " removed_M5_node=" << result.closure_removed_M_node_id[5]
      << " removed_M5_block=" << result.closure_removed_M_node_block[5]
      << " removed_M5_i=" << result.closure_removed_M_node_local_i[5]
      << " removed_M5_j=" << result.closure_removed_M_node_local_j[5]
      << " removed_M6_node=" << result.closure_removed_M_node_id[6]
      << " removed_M6_block=" << result.closure_removed_M_node_block[6]
      << " removed_M6_i=" << result.closure_removed_M_node_local_i[6]
      << " removed_M6_j=" << result.closure_removed_M_node_local_j[6]
      << " removed_M7_node=" << result.closure_removed_M_node_id[7]
      << " removed_M7_block=" << result.closure_removed_M_node_block[7]
      << " removed_M7_i=" << result.closure_removed_M_node_local_i[7]
      << " removed_M7_j=" << result.closure_removed_M_node_local_j[7]
      << " n_Aboundary_outer=" << result.affected_boundary_outer_nodes
      << " closure_core_cells=" << result.closure_core_cells
      << " active_nodes=" << result.active_nodes
      << " boundary_nodes=" << result.boundary_nodes
      << " frozen_nodes=" << result.frozen_nodes
      << " sigma_accepted=" << detail::format_scientific(result.sigma_accepted)
      << " winslow_iterations=" << result.winslow_iterations
      << " barrier_sweeps=" << result.barrier_sweeps
      << " line_search_iters=" << result.line_search_iters
      << " first_bad_cell=" << result.first_bad_cell
      << " max_active_delta="
      << detail::format_scientific(result.max_active_delta)
      << " max_frozen_delta="
      << detail::format_scientific(result.max_frozen_delta)
      << " max_boundary_dx="
      << detail::format_scientific(result.max_boundary_dx)
      << " max_boundary_dvr="
      << detail::format_scientific(result.max_boundary_dvr)
      << " max_boundary_dvz="
      << detail::format_scientific(result.max_boundary_dvz)
      << " max_boundary_dm="
      << detail::format_scientific(result.max_boundary_dm)
      << " source_current_delta="
      << detail::format_scientific(source_delta)
      << " dM=" << detail::format_scientific(result.dM)
      << " dM_rel="
      << detail::format_scientific(
             detail::relative_delta(result.dM, before.mass))
      << " dPr=" << detail::format_scientific(result.dPr)
      << " dPz=" << detail::format_scientific(result.dPz)
      << " p_scale=" << detail::format_scientific(p_scale)
      << " dPr_over_p_scale="
      << detail::format_scientific(dPr_over_p_scale)
      << " dPz_over_p_scale="
      << detail::format_scientific(dPz_over_p_scale)
      << " transport_momentum_valid="
      << (result.transport_momentum_valid ? 1 : 0)
      << " dPr_transport_only="
      << detail::format_scientific(dPr_transport_only)
      << " dPz_transport_only="
      << detail::format_scientific(dPz_transport_only)
      << " dPr_after_override=" << detail::format_scientific(result.dPr)
      << " dPz_after_override=" << detail::format_scientific(result.dPz)
      << " R_pi_r=" << detail::format_scientific(result.R_pi_r)
      << " R_pi_z=" << detail::format_scientific(result.R_pi_z)
      << " R_assm_r=" << detail::format_scientific(result.R_assm_r)
      << " R_assm_z=" << detail::format_scientific(result.R_assm_z)
      << " R_rec_r=" << detail::format_scientific(result.R_rec_r)
      << " R_rec_z=" << detail::format_scientific(result.R_rec_z)
      << " discarded_dual_faces=" << result.discarded_dual_faces
      << " discarded_dual_mass="
      << detail::format_scientific(result.discarded_dual_mass)
      << " discarded_dual_pi_r="
      << detail::format_scientific(result.discarded_dual_pi_r)
      << " discarded_dual_pi_z="
      << detail::format_scientific(result.discarded_dual_pi_z)
      << " dPr_boundary=" << detail::format_scientific(result.dPr_boundary)
      << " dPz_boundary=" << detail::format_scientific(result.dPz_boundary)
      << " dPr_active=" << detail::format_scientific(result.dPr_active)
      << " dPz_active=" << detail::format_scientific(result.dPz_active)
      << " dPr_other=" << detail::format_scientific(dPr_other)
      << " dPz_other=" << detail::format_scientific(dPz_other)
      << " n_ext_cell_cmass_changed="
      << result.n_ext_cell_cmass_changed
      << " first_ext_cell_cmass_changed="
      << result.first_ext_cell_cmass_changed
      << " max_ext_cell_cmass_changed_cell="
      << result.max_ext_cell_cmass_changed_cell
      << " max_ext_cell_cmass_delta="
      << detail::format_scientific(result.max_ext_cell_cmass_delta)
      << " n_ext_node_vel_changed=" << result.n_ext_node_vel_changed
      << " n_vel_changed_outside_velmask="
      << result.n_vel_changed_outside_velmask
      << " first_ext_node_vel_changed="
      << result.first_ext_node_vel_changed
      << " max_ext_node_vel_changed_node="
      << result.max_ext_node_vel_changed_node
      << " max_ext_node_vel_delta="
      << detail::format_scientific(result.max_ext_node_vel_delta)
      << " dE_tot=" << detail::format_scientific(result.dE)
      << " R_E=" << detail::format_scientific(result.R_E)
      << " dE_tot_rel="
      << detail::format_scientific(
             detail::relative_delta(result.dE, before.energy_total))
      << " dE_internal_delta="
      << detail::format_scientific(result.dE_internal)
      << " dKE_delta=" << detail::format_scientific(result.dKE)
      << " n_eint_floor_hits=" << result.n_eint_floor_hits
      << " dE_floor_deposit="
      << detail::format_scientific(result.dE_floor_deposit)
      << " n_active_floor_hits=" << result.n_active_floor_hits
      << " active_floor_mass_delta="
      << detail::format_scientific(result.active_floor_mass_delta)
      << " dE_active_floor="
      << detail::format_scientific(result.dE_active_floor)
      << " dE_int_transport_residual="
      << detail::format_scientific(result.dE_int_transport_residual)
      << " dKE_reconstruction_residual="
      << detail::format_scientific(result.dKE_reconstruction_residual)
      << " dE_unexplained="
      << detail::format_scientific(result.dE_unexplained);
  for (std::size_t i = 0; i < result.top_momentum_nodes.size(); ++i) {
    const ShellMomentumTopNode& top = result.top_momentum_nodes[i];
    oss << " top" << i << "_node=" << top.node
        << " top" << i << "_class="
        << detail::shell_rezone_node_class_name(top.node_class)
        << " top" << i << "_abs_dPr="
        << detail::format_scientific(top.abs_dPr)
        << " top" << i << "_dPr=" << detail::format_scientific(top.dPr)
        << " top" << i << "_dm=" << detail::format_scientific(top.dm)
        << " top" << i << "_dvr=" << detail::format_scientific(top.dvr)
        << " top" << i << "_dvz=" << detail::format_scientific(top.dvz)
        << " top" << i << "_mass_changed="
        << (top.mass_changed ? 1 : 0)
        << " top" << i << "_velocity_changed="
        << (top.velocity_changed ? 1 : 0);
  }
  for (std::size_t i = 0; i < result.velchk_nodes.size(); ++i) {
    const ShellVelocityCheckNode& chk = result.velchk_nodes[i];
    if (chk.node < 0) {
      continue;
    }
    oss << " velchk" << i << "_node=" << chk.node
        << " velchk" << i << "_flags=" << chk.flags
        << " velchk" << i << "_is_boundary="
        << (((chk.flags & mesh::NODE_BOUNDARY) != 0) ? 1 : 0)
        << " velchk" << i << "_is_outer_physical="
        << (((chk.flags & mesh::NODE_OUTER_PHYSICAL_BOUNDARY) != 0) ? 1 : 0)
        << " velchk" << i << "_dv=" << detail::format_scientific(chk.dv)
        << " velchk" << i << "_m_before="
        << detail::format_scientific(chk.m_before)
        << " velchk" << i << "_m_after="
        << detail::format_scientific(chk.m_after)
        << " velchk" << i << "_vr_before="
        << detail::format_scientific(chk.vr_before)
        << " velchk" << i << "_vr_after="
        << detail::format_scientific(chk.vr_after)
        << " velchk" << i << "_vz_before="
        << detail::format_scientific(chk.vz_before)
        << " velchk" << i << "_vz_after="
        << detail::format_scientific(chk.vz_after);
  }
  core::log_warning(oss.str());
  snapshot.restore(state, nullptr);
  return false;
}

namespace {

struct Cell113TraceConfig {
  bool enabled = false;
  int cell = 113;
  int step_begin = 0;
  int step_end = std::numeric_limits<int>::max();
};

struct Cell113TraceSample {
  int cell = -1;
  std::array<int, 4> nodes{{-1, -1, -1, -1}};
  double rho = std::numeric_limits<double>::quiet_NaN();
  double mass = std::numeric_limits<double>::quiet_NaN();
  double vol = std::numeric_limits<double>::quiet_NaN();
  double area = std::numeric_limits<double>::quiet_NaN();
  double ee = std::numeric_limits<double>::quiet_NaN();
  double ei = std::numeric_limits<double>::quiet_NaN();
  double e_spec = std::numeric_limits<double>::quiet_NaN();
  double cs = std::numeric_limits<double>::quiet_NaN();
  double Pe = std::numeric_limits<double>::quiet_NaN();
  double Pi = std::numeric_limits<double>::quiet_NaN();
  double avg_uz = std::numeric_limits<double>::quiet_NaN();
  double avg_vmag = std::numeric_limits<double>::quiet_NaN();
  double max_abs_uz = std::numeric_limits<double>::quiet_NaN();
  double max_vmag = std::numeric_limits<double>::quiet_NaN();
  std::array<double, 4> r{};
  std::array<double, 4> z{};
  std::array<double, 4> ur{};
  std::array<double, 4> uz{};
  std::array<double, 4> vmag{};
};

int parse_cell113_trace_env_int(const char* name, const int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const long value = std::strtol(raw, &end, 10);
  if (end == raw) {
    return fallback;
  }
  return static_cast<int>(value);
}

bool parse_cell113_trace_env_bool(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return false;
  }
  const std::string value(raw);
  return value == "1" || value == "true" || value == "TRUE" ||
         value == "on" || value == "ON";
}

const Cell113TraceConfig& cell113_trace_config() {
  static const Cell113TraceConfig cfg = [] {
    Cell113TraceConfig out;
    out.enabled = parse_cell113_trace_env_bool("TENRYU_CELL113_TRACE");
    out.cell = parse_cell113_trace_env_int("TENRYU_CELL113_TRACE_CELL", 113);
    out.step_begin =
        parse_cell113_trace_env_int("TENRYU_CELL113_TRACE_STEP_BEGIN", 0);
    out.step_end = parse_cell113_trace_env_int(
        "TENRYU_CELL113_TRACE_STEP_END", std::numeric_limits<int>::max());
    return out;
  }();
  return cfg;
}

template <typename Field>
double read_cell113_trace_field_value(const Field& field, const int idx) {
  if (idx < 0 || static_cast<std::size_t>(idx) >= field.size()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double value = std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(cudaMemcpy(&value,
                        field.data() + idx,
                        sizeof(double),
                        cudaMemcpyDeviceToHost));
  return value;
}

double cell113_trace_area_from_nodes(const std::array<double, 4>& r,
                                     const std::array<double, 4>& z) {
  double sum = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int kp = (k + 1) & 3;
    sum += r[static_cast<std::size_t>(k)] * z[static_cast<std::size_t>(kp)] -
           r[static_cast<std::size_t>(kp)] * z[static_cast<std::size_t>(k)];
  }
  return 0.5 * std::abs(sum);
}

Cell113TraceSample capture_cell113_trace_sample(const core::State& state,
                                                const core::Config& cfg,
                                                const int cell) {
  Cell113TraceSample s;
  s.cell = cell;
  s.nodes = mesh::mesh_topo_cell_corner_nodes(state.mesh.topo, cell, cfg.mesh);
  s.rho = read_cell113_trace_field_value(state.rho, cell);
  s.mass = read_cell113_trace_field_value(state.mass, cell);
  s.vol = read_cell113_trace_field_value(state.vol, cell);
  s.ee = read_cell113_trace_field_value(state.ee, cell);
  s.ei = read_cell113_trace_field_value(state.ei, cell);
  s.e_spec = s.ee + s.ei;
  s.cs = read_cell113_trace_field_value(state.cs, cell);
  s.Pe = read_cell113_trace_field_value(state.Pe, cell);
  s.Pi = read_cell113_trace_field_value(state.Pi, cell);

  double sum_uz = 0.0;
  double sum_vmag = 0.0;
  s.max_abs_uz = 0.0;
  s.max_vmag = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int n = s.nodes[static_cast<std::size_t>(k)];
    s.r[static_cast<std::size_t>(k)] =
        read_cell113_trace_field_value(state.x_r, n);
    s.z[static_cast<std::size_t>(k)] =
        read_cell113_trace_field_value(state.x_z, n);
    s.ur[static_cast<std::size_t>(k)] =
        read_cell113_trace_field_value(state.v_r, n);
    s.uz[static_cast<std::size_t>(k)] =
        read_cell113_trace_field_value(state.v_z, n);
    s.vmag[static_cast<std::size_t>(k)] =
        std::hypot(s.ur[static_cast<std::size_t>(k)],
                   s.uz[static_cast<std::size_t>(k)]);
    sum_uz += s.uz[static_cast<std::size_t>(k)];
    sum_vmag += s.vmag[static_cast<std::size_t>(k)];
    s.max_abs_uz =
        std::max(s.max_abs_uz, std::abs(s.uz[static_cast<std::size_t>(k)]));
    s.max_vmag = std::max(s.max_vmag, s.vmag[static_cast<std::size_t>(k)]);
  }
  s.avg_uz = 0.25 * sum_uz;
  s.avg_vmag = 0.25 * sum_vmag;
  s.area = cell113_trace_area_from_nodes(s.r, s.z);
  return s;
}

std::string format_cell113_trace_delta(const char* name,
                                       const double value,
                                       const double previous) {
  return std::string(" d_") + name + "=" +
         detail::format_scientific(value - previous);
}

bool cell113_trace_active_for_step(const core::State& state,
                                   const parallel::PartitionInfo& part) {
  const Cell113TraceConfig& trace = cell113_trace_config();
  if (!trace.enabled || part.rank != 0) {
    return false;
  }
  const int step = state.step + 1;
  return step >= trace.step_begin && step <= trace.step_end;
}

std::string format_cell113_axis_target_extra(
    const core::State& state,
    const core::Config& cfg,
    const axis_ale::AxisAleRezoneInput& input,
    const axis_ale::AxisAleRezoneResult& target) {
  const Cell113TraceConfig& trace = cell113_trace_config();
  if (!trace.enabled || trace.cell < 0 ||
      trace.cell >= state.mesh.topo.n_cells || !target.active ||
      target.node_ids.size() != target.z_target.size() ||
      target.node_ids.size() != input.z_tilde.size()) {
    return {};
  }
  double max_abs_dz = 0.0;
  double sum_dz2 = 0.0;
  for (std::size_t i = 0; i < target.node_ids.size(); ++i) {
    const double dz = target.z_target[i] - input.z_tilde[i];
    max_abs_dz = std::max(max_abs_dz, std::abs(dz));
    sum_dz2 += dz * dz;
  }
  const std::array<int, 4> cell_nodes =
      mesh::mesh_topo_cell_corner_nodes(state.mesh.topo, trace.cell, cfg.mesh);
  std::array<double, 4> cell_node_dz{};
  std::array<int, 4> cell_node_targeted{{0, 0, 0, 0}};
  for (int k = 0; k < 4; ++k) {
    cell_node_dz[static_cast<std::size_t>(k)] =
        std::numeric_limits<double>::quiet_NaN();
    for (std::size_t i = 0; i < target.node_ids.size(); ++i) {
      if (target.node_ids[i] == cell_nodes[static_cast<std::size_t>(k)]) {
        cell_node_targeted[static_cast<std::size_t>(k)] = 1;
        cell_node_dz[static_cast<std::size_t>(k)] =
            target.z_target[i] - input.z_tilde[i];
        break;
      }
    }
  }
  const double rms_dz =
      target.node_ids.empty()
          ? 0.0
          : std::sqrt(sum_dz2 / static_cast<double>(target.node_ids.size()));
  std::ostringstream oss;
  oss << "axis_target_nodes=" << target.node_ids.size()
      << " axis_target_max_abs_dz=" << detail::format_scientific(max_abs_dz)
      << " axis_target_rms_dz=" << detail::format_scientific(rms_dz);
  for (int k = 0; k < 4; ++k) {
    oss << " cell_node" << k
        << "_axis_targeted=" << cell_node_targeted[static_cast<std::size_t>(k)]
        << " cell_node" << k
        << "_axis_target_dz="
        << detail::format_scientific(cell_node_dz[static_cast<std::size_t>(k)]);
  }
  return oss.str();
}

struct AleEstepEnergySample {
  bool valid = false;
  long double E_internal = 0.0L;
  long double E_kin = 0.0L;
  long double E_total = 0.0L;
  long double Ediag_internal = 0.0L;
  long double Ediag_kin = 0.0L;
  long double Ediag_total = 0.0L;
};

struct AleEstepTraceRecord {
  int step = -1;
  double dt_s = 0.0;
  double t_s = 0.0;
  bool have_step_start = false;
  bool have_post_lagrangian = false;
  bool have_post_rezone = false;
  bool have_post_remap = false;
  bool have_step_end = false;
  AleEstepEnergySample step_start;
  AleEstepEnergySample post_lagrangian;
  AleEstepEnergySample post_rezone;
  AleEstepEnergySample post_remap;
  AleEstepEnergySample step_end;
};

std::unordered_map<const core::State*, AleEstepTraceRecord>&
ale_estep_trace_records() {
  static std::unordered_map<const core::State*, AleEstepTraceRecord> records;
  return records;
}

bool ale_estep_trace_env_enabled() {
  static const bool enabled =
      parse_cell113_trace_env_bool("TENRYU_I1B_REMAP_ENERGY_AUDIT");
  return enabled;
}

bool ale_physical_ke_remap_trace_env_enabled() {
  static const bool enabled =
      parse_cell113_trace_env_bool("TENRYU_ALE_PHYSICAL_KE_REMAP");
  return enabled;
}

int ale_estep_trace_every_n() {
  static const int every = []() {
    const int parsed =
        parse_cell113_trace_env_int("TENRYU_I1B_REMAP_ENERGY_AUDIT_EVERY", 32);
    return (parsed > 0 && parsed <= 1000000) ? parsed : 32;
  }();
  return every;
}

bool ale_estep_trace_active_for_step(const core::State& state,
                                     const parallel::PartitionInfo& part) {
  if (!ale_estep_trace_env_enabled() || state.mesh.dim != 2) {
    return false;
  }
  if (part.rank < 0) {
    return false;
  }
  if (state.step < 8) {
    return true;
  }
  const int every = ale_estep_trace_every_n();
  return every > 0 && ((state.step + 1) % every) == 0;
}

std::string format_scientific_long(const long double value) {
  std::ostringstream oss;
  oss.setf(std::ios::scientific);
  oss.precision(18);
  oss << value;
  return oss.str();
}

long double ale_estep_trace_reference_energy(const long double E_step_start) {
  static long double E_ref = 0.0L;
  if (!(std::fabs(E_ref) > 0.0L) || !std::isfinite(E_ref)) {
    E_ref = (std::fabs(E_step_start) > 0.0L && std::isfinite(E_step_start))
                ? std::fabs(E_step_start)
                : 1.0L;
  }
  return E_ref;
}

long double ale_estep_trace_relative(const long double delta,
                                     const long double reference) {
  if (!(std::fabs(reference) > 0.0L) || !std::isfinite(reference)) {
    return 0.0L;
  }
  return delta / reference;
}

int ale_estep_trace_active_nverts(const core::State& state, const int c) {
  return mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c);
}

void ale_estep_trace_accumulate_node_mass_structured(
    std::vector<long double>& node_mass,
    const core::State& state,
    const std::vector<double>& corner_mass) {
  const int n_cells = state.mesh.topo.n_cells;
  const int nz = state.mesh.topo.nz;
  const int stride = nz + 1;
  for (int c = 0; c < n_cells; ++c) {
    const int i = (nz > 0) ? c / nz : 0;
    const int j = (nz > 0) ? c - i * nz : 0;
    const int active_nverts = ale_estep_trace_active_nverts(state, c);
    const int nodes[4] = {
        i * stride + j,
        (i + 1) * stride + j,
        (i + 1) * stride + (j + 1),
        i * stride + (j + 1)};
    for (int k = 0; k < active_nverts && k < 4; ++k) {
      const int n = nodes[k];
      if (n >= 0 && static_cast<std::size_t>(n) < node_mass.size()) {
        node_mass[static_cast<std::size_t>(n)] +=
            static_cast<long double>(
                corner_mass[static_cast<std::size_t>(c) * 4U +
                            static_cast<std::size_t>(k)]);
      }
    }
  }
}

void ale_estep_trace_accumulate_node_mass_multiblock(
    std::vector<long double>& node_mass,
    const core::State& state,
    const std::vector<double>& corner_mass) {
  const int n_cells = state.mesh.topo.n_cells;
  std::vector<int> offsets;
  std::vector<int> indices;
  state.mesh.multiblock_cell_node_csr_offsets.copy_to_host(offsets);
  state.mesh.multiblock_cell_node_csr_indices.copy_to_host(indices);
  if (offsets.size() < static_cast<std::size_t>(n_cells + 1)) {
    return;
  }
  for (int c = 0; c < n_cells; ++c) {
    const int off = offsets[static_cast<std::size_t>(c)];
    const int active_nverts = ale_estep_trace_active_nverts(state, c);
    for (int k = 0; k < active_nverts && k < 4; ++k) {
      const std::size_t idx = static_cast<std::size_t>(off + k);
      if (idx >= indices.size()) {
        continue;
      }
      const int n = indices[idx];
      if (n >= 0 && static_cast<std::size_t>(n) < node_mass.size()) {
        node_mass[static_cast<std::size_t>(n)] +=
            static_cast<long double>(
                corner_mass[static_cast<std::size_t>(c) * 4U +
                            static_cast<std::size_t>(k)]);
      }
    }
  }
}

AleEstepEnergySample compute_ale_estep_energy_sample(
    const core::State& state,
    const core::Config& cfg,
    const parallel::Reduction* reduction) {
  (void)cfg;
  AleEstepEnergySample out;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (n_cells <= 0 || n_nodes <= 0) {
    return out;
  }
  if (state.mass.size() != static_cast<std::size_t>(n_cells) ||
      state.ee.size() != static_cast<std::size_t>(n_cells) ||
      state.ei.size() != static_cast<std::size_t>(n_cells) ||
      state.v_r.size() != static_cast<std::size_t>(n_nodes) ||
      state.v_z.size() != static_cast<std::size_t>(n_nodes) ||
      state.corner_mass.size() <
          static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(4)) {
    return out;
  }

  std::vector<double> mass;
  std::vector<double> ee;
  std::vector<double> ei;
  std::vector<double> v_r;
  std::vector<double> v_z;
  std::vector<double> corner_mass;
  state.mass.copy_to_host(mass);
  state.ee.copy_to_host(ee);
  state.ei.copy_to_host(ei);
  state.v_r.copy_to_host(v_r);
  state.v_z.copy_to_host(v_z);
  state.corner_mass.copy_to_host(corner_mass);

  long double E_internal = 0.0L;
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t idx = static_cast<std::size_t>(c);
    E_internal += static_cast<long double>(mass[idx]) *
                  (static_cast<long double>(ee[idx]) +
                   static_cast<long double>(ei[idx]));
  }

  std::vector<long double> node_mass(static_cast<std::size_t>(n_nodes), 0.0L);
  if (state.mesh.topo.multiblock.has_value()) {
    ale_estep_trace_accumulate_node_mass_multiblock(
        node_mass, state, corner_mass);
  } else {
    ale_estep_trace_accumulate_node_mass_structured(
        node_mass, state, corner_mass);
  }

  long double E_kin = 0.0L;
  for (int n = 0; n < n_nodes; ++n) {
    const std::size_t idx = static_cast<std::size_t>(n);
    const long double vr = static_cast<long double>(v_r[idx]);
    const long double vz = static_cast<long double>(v_z[idx]);
    E_kin += 0.5L * node_mass[idx] * (vr * vr + vz * vz);
  }

  double local_values[2] = {static_cast<double>(E_internal),
                            static_cast<double>(E_kin)};
  if (reduction != nullptr) {
    reduction->allreduce_sum(local_values, 2);
  }
  out.E_internal = static_cast<long double>(local_values[0]);
  out.E_kin = static_cast<long double>(local_values[1]);
  out.E_total = out.E_internal + out.E_kin;

  const diagnostics::EnergyTotals diag =
      detail::reduce_energy_totals_global(
          diagnostics::compute_energy_totals_2d(state), reduction);
  out.Ediag_internal =
      static_cast<long double>(diag.E_int_e + diag.E_int_i);
  out.Ediag_kin = static_cast<long double>(diag.E_kin);
  out.Ediag_total = out.Ediag_internal + out.Ediag_kin;
  if (ale_physical_ke_remap_trace_env_enabled()) {
    out.E_internal = out.Ediag_internal;
    out.E_kin = out.Ediag_kin;
    out.E_total = out.Ediag_total;
  }
  out.valid = true;
  return out;
}

AleEstepTraceRecord& ale_estep_trace_record_for_state(const core::State& state,
                                                      const double dt_s,
                                                      const double t_s) {
  AleEstepTraceRecord& record = ale_estep_trace_records()[&state];
  if (record.step != state.step) {
    record = AleEstepTraceRecord{};
    record.step = state.step;
  }
  if (std::isfinite(dt_s) && dt_s > 0.0) {
    record.dt_s = dt_s;
  }
  if (std::isfinite(t_s)) {
    record.t_s = t_s;
  }
  return record;
}

const AleEstepEnergySample& ale_estep_trace_sample_or_nan(
    const AleEstepEnergySample& sample,
    const bool have_sample) {
  static const AleEstepEnergySample missing = []() {
    AleEstepEnergySample out;
    const long double nan = std::numeric_limits<long double>::quiet_NaN();
    out.E_internal = nan;
    out.E_kin = nan;
    out.E_total = nan;
    out.Ediag_internal = nan;
    out.Ediag_kin = nan;
    out.Ediag_total = nan;
    return out;
  }();
  return have_sample ? sample : missing;
}

void emit_ale_estep_trace(const core::State& state,
                          const parallel::PartitionInfo& part) {
  auto& records = ale_estep_trace_records();
  auto it = records.find(&state);
  if (it == records.end()) {
    return;
  }
  const AleEstepTraceRecord& record = it->second;
  if (part.rank == 0 && record.have_step_start &&
      record.have_post_lagrangian && record.have_step_end) {
    const AleEstepEnergySample& a =
        ale_estep_trace_sample_or_nan(record.step_start,
                                      record.have_step_start);
    const AleEstepEnergySample& b =
        ale_estep_trace_sample_or_nan(record.post_lagrangian,
                                      record.have_post_lagrangian);
    const AleEstepEnergySample& c =
        ale_estep_trace_sample_or_nan(record.post_rezone,
                                      record.have_post_rezone);
    const AleEstepEnergySample& d =
        ale_estep_trace_sample_or_nan(record.post_remap,
                                      record.have_post_remap);
    const AleEstepEnergySample& e =
        ale_estep_trace_sample_or_nan(record.step_end,
                                      record.have_step_end);
    const long double E_ref = ale_estep_trace_reference_energy(a.E_total);
    std::ostringstream oss;
    oss << "[ale_estep_trace]"
        << " step=" << record.step
        << " physical_ke="
        << (ale_physical_ke_remap_trace_env_enabled() ? 1 : 0)
        << " time_s=" << format_scientific_long(record.t_s)
        << " dt_s=" << format_scientific_long(record.dt_s)
        << " E_ref=" << format_scientific_long(E_ref)
        << " E_step_start=" << format_scientific_long(a.E_total)
        << " E_post_lagrangian=" << format_scientific_long(b.E_total)
        << " E_post_rezone_pre_remap=" << format_scientific_long(c.E_total)
        << " E_post_remap=" << format_scientific_long(d.E_total)
        << " E_step_end=" << format_scientific_long(e.E_total)
        << " K_step_start=" << format_scientific_long(a.E_kin)
        << " K_post_lagrangian=" << format_scientific_long(b.E_kin)
        << " K_post_rezone_pre_remap=" << format_scientific_long(c.E_kin)
        << " K_post_remap=" << format_scientific_long(d.E_kin)
        << " K_step_end=" << format_scientific_long(e.E_kin)
        << " dE_lagrangian_rel="
        << format_scientific_long(
               ale_estep_trace_relative(b.E_total - a.E_total, E_ref))
        << " dE_rezone_rel="
        << format_scientific_long(
               ale_estep_trace_relative(c.E_total - b.E_total, E_ref))
        << " dE_remap_rel="
        << format_scientific_long(
               ale_estep_trace_relative(d.E_total - c.E_total, E_ref))
        << " dE_finish_rel="
        << format_scientific_long(
               ale_estep_trace_relative(e.E_total - d.E_total, E_ref))
        << " Ediag_step_start=" << format_scientific_long(a.Ediag_total)
        << " Ediag_post_lagrangian=" << format_scientific_long(b.Ediag_total)
        << " Ediag_post_rezone_pre_remap="
        << format_scientific_long(c.Ediag_total)
        << " Ediag_post_remap=" << format_scientific_long(d.Ediag_total)
        << " Ediag_step_end=" << format_scientific_long(e.Ediag_total)
        << " Kdiag_step_start=" << format_scientific_long(a.Ediag_kin)
        << " Kdiag_post_lagrangian=" << format_scientific_long(b.Ediag_kin)
        << " Kdiag_post_rezone_pre_remap="
        << format_scientific_long(c.Ediag_kin)
        << " Kdiag_post_remap=" << format_scientific_long(d.Ediag_kin)
        << " Kdiag_step_end=" << format_scientific_long(e.Ediag_kin)
        << " dEdiag_lagrangian_rel="
        << format_scientific_long(
               ale_estep_trace_relative(b.Ediag_total - a.Ediag_total, E_ref))
        << " dEdiag_rezone_rel="
        << format_scientific_long(
               ale_estep_trace_relative(c.Ediag_total - b.Ediag_total, E_ref))
        << " dEdiag_remap_rel="
        << format_scientific_long(
               ale_estep_trace_relative(d.Ediag_total - c.Ediag_total, E_ref))
        << " dEdiag_finish_rel="
        << format_scientific_long(
               ale_estep_trace_relative(e.Ediag_total - d.Ediag_total, E_ref));
    core::log_info(oss.str());
  }
  records.erase(it);
}

}  // namespace

void record_estep_trace_boundary(const core::State& state,
                                  const core::Config& cfg,
                                  const parallel::PartitionInfo& part,
                                  const parallel::Reduction* reduction,
                                  const char* stage,
                                  const double dt_s,
                                  const double t_s) {
  // Stage-resolved pole-shear sampling rides these stage boundaries on every
  // ALE path (fire/barrier/Winslow/no-op alike); armed only by
  // TENRYU_I1B_POLE_SHEAR_DIAG_STAGES + _EVERY, ahead of the estep-trace
  // enable gate below. Per-step-per-stage dedup lives in the sampler.
  if (stage != nullptr && part.rank == 0) {
    if (std::strcmp(stage, "post_rezone_pre_remap") == 0) {
      detail::log_pole_shear_stage_sample(state, true, 'R', t_s);
    } else if (std::strcmp(stage, "post_remap") == 0) {
      detail::log_pole_shear_stage_sample(state, false, 'A', t_s);
    }
  }
  if (!ale_estep_trace_active_for_step(state, part) || stage == nullptr) {
    return;
  }
  AleEstepTraceRecord& record =
      ale_estep_trace_record_for_state(state, dt_s, t_s);
  const AleEstepEnergySample sample =
      compute_ale_estep_energy_sample(state, cfg, reduction);
  const std::string stage_name(stage);
  if (stage_name == "step_start") {
    record.step_start = sample;
    record.have_step_start = sample.valid;
  } else if (stage_name == "post_lagrangian") {
    record.post_lagrangian = sample;
    record.have_post_lagrangian = sample.valid;
  } else if (stage_name == "post_rezone_pre_remap") {
    record.post_rezone = sample;
    record.have_post_rezone = sample.valid;
  } else if (stage_name == "post_remap") {
    record.post_remap = sample;
    record.have_post_remap = sample.valid;
  } else if (stage_name == "step_end") {
    record.step_end = sample;
    record.have_step_end = sample.valid;
    emit_ale_estep_trace(state, part);
  }
}

void log_cell113_substage_trace(const core::State& state,
                                const core::Config& cfg,
                                const parallel::PartitionInfo& part,
                                const char* stage,
                                const double dt_s,
                                const double t_s,
                                const char* extra) {
  if (!cell113_trace_active_for_step(state, part)) {
    return;
  }
  const Cell113TraceConfig& trace = cell113_trace_config();
  if (trace.cell < 0 || trace.cell >= state.mesh.topo.n_cells) {
    return;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  const Cell113TraceSample sample =
      capture_cell113_trace_sample(state, cfg, trace.cell);
  static bool have_previous = false;
  static Cell113TraceSample previous;
  const bool comparable =
      have_previous && previous.cell == sample.cell && previous.nodes == sample.nodes;

  std::ostringstream oss;
  oss << "[cell113_substage_trace]"
      << " step=" << (state.step + 1)
      << " state_step=" << state.step
      << " t_s=" << detail::format_scientific(t_s)
      << " dt_s=" << detail::format_scientific(dt_s)
      << " stage=" << (stage != nullptr ? stage : "unknown")
      << " cell=" << sample.cell
      << " nodes=" << sample.nodes[0] << "," << sample.nodes[1] << ","
      << sample.nodes[2] << "," << sample.nodes[3]
      << " rho=" << detail::format_scientific(sample.rho)
      << " mass=" << detail::format_scientific(sample.mass)
      << " vol=" << detail::format_scientific(sample.vol)
      << " area=" << detail::format_scientific(sample.area)
      << " ee=" << detail::format_scientific(sample.ee)
      << " ei=" << detail::format_scientific(sample.ei)
      << " e_spec=" << detail::format_scientific(sample.e_spec)
      << " cs=" << detail::format_scientific(sample.cs)
      << " Pe=" << detail::format_scientific(sample.Pe)
      << " Pi=" << detail::format_scientific(sample.Pi)
      << " avg_uz=" << detail::format_scientific(sample.avg_uz)
      << " avg_vmag=" << detail::format_scientific(sample.avg_vmag)
      << " max_abs_uz=" << detail::format_scientific(sample.max_abs_uz)
      << " max_vmag=" << detail::format_scientific(sample.max_vmag);
  if (comparable) {
    oss << format_cell113_trace_delta("e_spec", sample.e_spec, previous.e_spec)
        << format_cell113_trace_delta("cs", sample.cs, previous.cs)
        << format_cell113_trace_delta("avg_uz", sample.avg_uz, previous.avg_uz)
        << format_cell113_trace_delta("max_abs_uz",
                                      sample.max_abs_uz,
                                      previous.max_abs_uz)
        << format_cell113_trace_delta("max_vmag",
                                      sample.max_vmag,
                                      previous.max_vmag)
        << format_cell113_trace_delta("area", sample.area, previous.area)
        << format_cell113_trace_delta("vol", sample.vol, previous.vol);
  }
  for (int k = 0; k < 4; ++k) {
    const std::size_t ku = static_cast<std::size_t>(k);
    oss << " node" << k << "_id=" << sample.nodes[ku]
        << " node" << k << "_r=" << detail::format_scientific(sample.r[ku])
        << " node" << k << "_z=" << detail::format_scientific(sample.z[ku])
        << " node" << k << "_ur=" << detail::format_scientific(sample.ur[ku])
        << " node" << k << "_uz=" << detail::format_scientific(sample.uz[ku])
        << " node" << k << "_vmag="
        << detail::format_scientific(sample.vmag[ku]);
    if (comparable) {
      oss << " d_node" << k
          << "_r=" << detail::format_scientific(sample.r[ku] - previous.r[ku])
          << " d_node" << k
          << "_z=" << detail::format_scientific(sample.z[ku] - previous.z[ku])
          << " d_node" << k
          << "_ur=" << detail::format_scientific(sample.ur[ku] - previous.ur[ku])
          << " d_node" << k
          << "_uz=" << detail::format_scientific(sample.uz[ku] - previous.uz[ku])
          << " d_node" << k
          << "_vmag="
          << detail::format_scientific(sample.vmag[ku] - previous.vmag[ku]);
    }
  }
  if (extra != nullptr && *extra != '\0') {
    oss << ' ' << extra;
  }
  core::log_info(oss.str());
  previous = sample;
  have_previous = true;
}

namespace detail {

mesh::MultiBlockTopology m1_forward_topology(const core::State& state) {
  mesh::MultiBlockTopology topology;
  const bool is_multiblock = state.mesh.topo.multiblock.has_value();
  if (is_multiblock) {
    topology = *state.mesh.topo.multiblock;
  } else {
    const int nr = state.mesh.topo.nr;
    const int nz = state.mesh.topo.nz;
    const int n_cells = state.mesh.topo.n_cells;
    // Structured polar cells use the same CW n00,n10,n11,n01 cycle encoded
    // by polar_orientation_sign=-1 in the polar tri-fan geometry kernel and
    // ORIENT=-1 in the polar-in-box recompute kernel.
    const int orientation_sign =
        state.mesh.logical == mesh::LogicalMesh2D::SphericalPolarHalfplane ||
                state.mesh.logical == mesh::LogicalMesh2D::PolarInBox
            ? -1
            : 1;
    topology.cell_id_stable.resize(static_cast<std::size_t>(n_cells));
    topology.cell_orientation_sign.assign(
        static_cast<std::size_t>(n_cells), orientation_sign);
    topology.cell_node_csr_offsets.resize(
        static_cast<std::size_t>(n_cells) + 1U);
    topology.cell_node_csr_indices.resize(
        static_cast<std::size_t>(4 * n_cells));
    for (int i = 0; i < nr; ++i) {
      for (int j = 0; j < nz; ++j) {
        const int cell = state.mesh.topo.cell_index(i, j);
        const int begin = 4 * cell;
        topology.cell_id_stable[static_cast<std::size_t>(cell)] = cell;
        topology.cell_node_csr_offsets[static_cast<std::size_t>(cell)] =
            begin;
        topology.cell_node_csr_indices[static_cast<std::size_t>(begin)] =
            state.mesh.topo.node_index(i, j);
        topology.cell_node_csr_indices[static_cast<std::size_t>(begin + 1)] =
            state.mesh.topo.node_index(i + 1, j);
        topology.cell_node_csr_indices[static_cast<std::size_t>(begin + 2)] =
            state.mesh.topo.node_index(i + 1, j + 1);
        topology.cell_node_csr_indices[static_cast<std::size_t>(begin + 3)] =
            state.mesh.topo.node_index(i, j + 1);
      }
    }
    topology.cell_node_csr_offsets[static_cast<std::size_t>(n_cells)] =
        4 * n_cells;
  }

  const int n_cells = state.mesh.topo.n_cells;
  TENRYU_ASSERT(
      topology.cell_orientation_sign.size() ==
              static_cast<std::size_t>(n_cells) &&
          topology.cell_node_csr_offsets.size() ==
              static_cast<std::size_t>(n_cells) + 1U,
      "M1 cell orientation metadata size mismatch");
  for (int cell = 0; cell < n_cells; ++cell) {
    const int begin =
        topology.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
    const int end =
        topology.cell_node_csr_offsets[static_cast<std::size_t>(cell) + 1U];
    int nverts = 4;
    if (is_multiblock) {
      const int corner_stride = state.mesh.corner_stride;
      TENRYU_ASSERT(corner_stride == 4 || corner_stride == 8,
                    "M1 multiblock topology requires stride 4 or 8");
      TENRYU_ASSERT(end - begin == corner_stride,
                    "M1 multiblock CSR width must match corner stride");
      nverts =
          state.mesh.cell_nverts.empty()
              ? 4
              : static_cast<int>(
                    state.mesh.cell_nverts[static_cast<std::size_t>(cell)]);
      TENRYU_ASSERT(nverts >= 3 && nverts <= end - begin,
                    "M1 multiblock cell vertex count out of range");
      for (int corner = 0; corner < nverts; ++corner) {
        TENRYU_ASSERT(
            topology.cell_node_csr_indices[
                static_cast<std::size_t>(begin + corner)] >= 0,
            "M1 multiblock occupied cell-node slot must be non-negative");
      }
    } else {
      TENRYU_ASSERT(end - begin == 4,
                    "M1 v1 runtime supports QUAD meshes only");
    }
    const int orientation_sign =
        topology.cell_orientation_sign[static_cast<std::size_t>(cell)];
    TENRYU_ASSERT(orientation_sign == -1 || orientation_sign == 1,
                  "M1 cell orientation sign must be +/-1");
    if (orientation_sign == -1) {
      // Normalize only the M1-local cell cycle. Node ids remain unchanged,
      // so reverse CSR gradients and sweep updates stay keyed to real nodes.
      // Retain the sign: candidate quality pairs it with the original CSR.
      std::reverse(
          topology.cell_node_csr_indices.begin() + begin + 1,
          topology.cell_node_csr_indices.begin() + begin + nverts);
    }
  }
  return topology;
}

}  // namespace detail

namespace {

std::vector<double> m1_reference_tether_scale(
    const mesh::MultiBlockTopology& topology,
    const std::vector<std::uint8_t>& cell_nverts,
    const int topology_stride,
    const std::vector<double>& reference_r,
    const std::vector<double>& reference_z,
    const int n_nodes) {
  std::vector<std::vector<int>> neighbors(
      static_cast<std::size_t>(n_nodes));
  const int n_cells =
      static_cast<int>(topology.cell_node_csr_offsets.size()) - 1;
  TENRYU_ASSERT(
      cell_nverts.size() == static_cast<std::size_t>(n_cells),
      "M1 tether scale requires per-cell vertex counts");
  for (int cell = 0; cell < n_cells; ++cell) {
    const int begin =
        topology.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
    const int end =
        topology.cell_node_csr_offsets[static_cast<std::size_t>(cell) + 1U];
    TENRYU_ASSERT(end - begin == topology_stride,
                  "M1 tether CSR width must match topology stride");
    const int nverts =
        static_cast<int>(
            cell_nverts[static_cast<std::size_t>(cell)]);
    TENRYU_ASSERT(nverts >= 3 && nverts <= topology_stride,
                  "M1 tether cell vertex count out of range");
    for (int corner = 0; corner < nverts; ++corner) {
      const int node =
          topology.cell_node_csr_indices[
              static_cast<std::size_t>(begin + corner)];
      const int next =
          topology.cell_node_csr_indices[
              static_cast<std::size_t>(
                  begin + (corner + 1) % nverts)];
      TENRYU_ASSERT(node >= 0 && node < n_nodes &&
                        next >= 0 && next < n_nodes,
                    "M1 reference edge node id out of range");
      neighbors[static_cast<std::size_t>(node)].push_back(next);
      neighbors[static_cast<std::size_t>(next)].push_back(node);
    }
  }

  // standard tether scale: RMS of the incident reference edge lengths.
  std::vector<double> h(static_cast<std::size_t>(n_nodes), 0.0);
  for (int node = 0; node < n_nodes; ++node) {
    auto& incident = neighbors[static_cast<std::size_t>(node)];
    std::sort(incident.begin(), incident.end());
    incident.erase(std::unique(incident.begin(), incident.end()),
                   incident.end());
    TENRYU_ASSERT(!incident.empty(),
                  "M1 tether scale requires an incident reference edge");
    double sum_squared = 0.0;
    for (const int neighbor : incident) {
      const double dr =
          reference_r[static_cast<std::size_t>(neighbor)] -
          reference_r[static_cast<std::size_t>(node)];
      const double dz =
          reference_z[static_cast<std::size_t>(neighbor)] -
          reference_z[static_cast<std::size_t>(node)];
      sum_squared += dr * dr + dz * dz;
    }
    h[static_cast<std::size_t>(node)] =
        std::sqrt(sum_squared / static_cast<double>(incident.size()));
    TENRYU_ASSERT(std::isfinite(h[static_cast<std::size_t>(node)]) &&
                      h[static_cast<std::size_t>(node)] > 0.0,
                  "M1 tether scale must be finite and positive");
  }
  return h;
}

RezoneResult run_m1_tmop_rezone_transaction(
    core::State& state,
    const core::Config& cfg,
    const parallel::PartitionInfo& part,
    const parallel::Reduction* reduction,
    const bool force_rezone) {
  (void)force_rezone;
  RezoneResult out;
  out.min_quality =
      compute_min_quality(
          state, cfg, &out.mesh_tangle, &out.min_quality_cell_pre);
  if (reduction != nullptr) {
    out.min_quality = reduction->allreduce_min(out.min_quality);
    out.mesh_tangle =
        reduction->allreduce_max(out.mesh_tangle ? 1.0 : 0.0) > 0.5;
  }

  TENRYU_ASSERT(state.mesh.dim == 2,
                "M1 v1 runtime requires a 2D mesh");
  TENRYU_ASSERT(
      state.corner_stride == state.mesh.corner_stride &&
          (state.corner_stride == 4 || state.corner_stride == 8),
      "M1 runtime requires matching stride-4 or stride-8 corner storage");
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  TENRYU_ASSERT(n_cells > 0 && n_nodes > 0,
                "M1 v1 runtime requires a non-empty mesh");
  TENRYU_ASSERT(state.x_r_reference.size() == state.x_r.size() &&
                    state.x_z_reference.size() == state.x_z.size(),
                "M1 requires frozen reference coordinates");
  if (!state.mesh.cell_nverts.empty()) {
    TENRYU_ASSERT(
        state.mesh.cell_nverts.size() ==
            static_cast<std::size_t>(n_cells),
        "M1 cell_nverts size mismatch");
    for (const std::uint8_t nverts : state.mesh.cell_nverts) {
      TENRYU_ASSERT(
          nverts >= 3U && nverts <= 8U &&
              static_cast<int>(nverts) <= state.corner_stride,
          "M1 cell vertex count must be in [3,8] and fit corner stride");
    }
  }

  RollbackGuard tx;
  tx.capture(state, nullptr);

  std::vector<double> current_r;
  std::vector<double> current_z;
  std::vector<double> reference_r;
  std::vector<double> reference_z;
  state.x_r.copy_to_host(current_r);
  state.x_z.copy_to_host(current_z);
  state.x_r_reference.copy_to_host(reference_r);
  state.x_z_reference.copy_to_host(reference_z);
  const std::vector<double> lagrangian_r = current_r;
  const std::vector<double> lagrangian_z = current_z;

  const mesh::MultiBlockTopology topology =
      detail::m1_forward_topology(state);
  TENRYU_ASSERT(
      topology.cell_node_csr_offsets.size() ==
          static_cast<std::size_t>(n_cells) + 1U &&
          topology.cell_node_csr_indices.size() ==
              static_cast<std::size_t>(
                  state.corner_stride * n_cells),
      "M1 runtime requires fixed-width cell-node CSR");
  const std::vector<std::uint8_t> cell_nverts =
      state.mesh.cell_nverts.empty()
          ? std::vector<std::uint8_t>(
                static_cast<std::size_t>(n_cells), 4U)
          : state.mesh.cell_nverts;
  const ReverseCellNodeCSR reverse =
      build_reverse_cell_node_csr(
          topology, n_nodes, &cell_nverts, state.corner_stride);
  const std::vector<double> h =
      m1_reference_tether_scale(
          topology,
          cell_nverts,
          state.corner_stride,
          reference_r,
          reference_z,
          n_nodes);

  std::vector<m1::RezoneQuadratureTarget> targets(
      static_cast<std::size_t>(
          n_cells * m1::kRezoneMaxQuadraturePoints));
  for (int cell = 0; cell < n_cells; ++cell) {
    double cell_reference_r[8]{};
    double cell_reference_z[8]{};
    const int begin =
        topology.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts =
        static_cast<int>(
            cell_nverts[static_cast<std::size_t>(cell)]);
    for (int corner = 0; corner < nverts; ++corner) {
      const int node =
          topology.cell_node_csr_indices[
              static_cast<std::size_t>(begin + corner)];
      cell_reference_r[corner] =
          reference_r[static_cast<std::size_t>(node)];
      cell_reference_z[corner] =
          reference_z[static_cast<std::size_t>(node)];
    }
    TENRYU_ASSERT(
        m1::rezone_build_cell_quadrature_targets(
            cell_reference_r,
            cell_reference_z,
            nverts,
            nullptr,
            targets.data() +
                static_cast<std::size_t>(
                    cell * m1::kRezoneMaxQuadraturePoints)),
        "M1 failed to build a frozen reference quadrature target");
  }

  std::vector<std::uint8_t> axis_mask(
      static_cast<std::size_t>(n_nodes), 0U);
  std::vector<std::uint8_t> node_mask(
      static_cast<std::size_t>(n_nodes), 1U);
  TENRYU_ASSERT(state.mesh.topo.node_flags.size() ==
                    static_cast<std::size_t>(n_nodes),
                "M1 requires per-node topology flags");
  for (int node = 0; node < n_nodes; ++node) {
    const std::uint8_t flags =
        state.mesh.topo.node_flags[static_cast<std::size_t>(node)];
    const bool pole_axis = (flags & mesh::NODE_POLE_AXIS) != 0U;
    axis_mask[static_cast<std::size_t>(node)] =
        ((flags & (mesh::NODE_AXIS | mesh::NODE_POLE_AXIS)) != 0U)
            ? 1U
            : 0U;
    if ((flags &
         (mesh::NODE_OUTER_PHYSICAL_BOUNDARY | mesh::NODE_CENTER)) != 0U ||
        (!pole_axis && (flags & mesh::NODE_AXIS) != 0U)) {
      node_mask[static_cast<std::size_t>(node)] = 0U;
    }
  }

  const m1::RezoneRingNeighborPairs ring_pairs =
      m1::rezone_build_ring_neighbor_pairs(
          state.mesh.topo, cell_nverts, state.corner_stride);
  m1::RezoneMeshView view{};
  view.n_nodes = n_nodes;
  view.n_cells = n_cells;
  view.topology_stride = state.corner_stride;
  view.quadrature_stride = m1::kRezoneMaxQuadraturePoints;
  view.cell_node_offsets = topology.cell_node_csr_offsets.data();
  view.cell_node_indices = topology.cell_node_csr_indices.data();
  view.cell_nverts = cell_nverts.data();
  view.node_cell_offsets = reverse.node_offsets.data();
  view.node_cells = reverse.node_cells.data();
  view.node_corners = reverse.node_corners.data();
  view.quadrature_targets = targets.data();
  view.h = h.data();
  view.x_lagrangian_r = lagrangian_r.data();
  view.x_lagrangian_z = lagrangian_z.data();
  view.axis_mask = axis_mask.data();
  view.theta_left = ring_pairs.left.data();
  view.theta_right = ring_pairs.right.data();
  view.theta_touch_offsets = ring_pairs.touch_offsets.data();
  view.theta_touch_centers =
      ring_pairs.touch_centers.empty()
          ? nullptr
          : ring_pairs.touch_centers.data();

  m1::RezoneObjectiveParams params{};
  params.gamma_align = cfg.numerics.ale.m1_gamma_align;
  params.lambda_m = cfg.numerics.ale.m1_lambda_tether;
  params.theta_reg = cfg.numerics.ale.m1_theta_reg;
  params.beta = cfg.numerics.ale.m1_barrier_beta;
  params.objective_node_count = n_nodes;

  const m1::RezoneLocalObjective objective_before =
      m1::rezone_global_objective(
          view, current_r.data(), current_z.data(), params);
  double objective_after = objective_before.value;
  for (int sweep = 0; sweep < cfg.numerics.ale.m1_sweeps; ++sweep) {
    const m1::RezoneSweepResult sweep_result =
        m1::rezone_sweep(
            view,
            current_r.data(),
            current_z.data(),
            node_mask.data(),
            params);
    objective_after = sweep_result.J_after;
  }

  int moved_nodes = 0;
  std::vector<double> delta_r(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> delta_z(static_cast<std::size_t>(n_nodes), 0.0);
  for (int node = 0; node < n_nodes; ++node) {
    const auto index = static_cast<std::size_t>(node);
    delta_r[index] = current_r[index] - lagrangian_r[index];
    delta_z[index] = current_z[index] - lagrangian_z[index];
    if (delta_r[index] != 0.0 || delta_z[index] != 0.0) {
      ++moved_nodes;
    }
  }

  core::DeviceArray<double> d_delta_r(static_cast<std::size_t>(n_nodes));
  core::DeviceArray<double> d_delta_z(static_cast<std::size_t>(n_nodes));
  d_delta_r.copy_from_host(delta_r);
  d_delta_z.copy_from_host(delta_z);
  mesh::CandidateMeshAdmissibilityFloors floors;
  floors.volume_rel = cfg.numerics.ale.reference_volume_floor_rel;
  floors.corner_j_rel = cfg.numerics.ale.reference_corner_j_floor_rel;
  floors.gauss_j_rel = cfg.numerics.ale.reference_gauss_j_floor_rel;

  mesh::CandidateMeshQuality candidate_quality{};
  if (state.mesh.topo.multiblock.has_value()) {
    core::DeviceArray<int> d_cell_id_stable(
        topology.cell_id_stable.size());
    core::DeviceArray<int> d_cell_orientation_sign(
        topology.cell_orientation_sign.size());
    d_cell_id_stable.copy_from_host(topology.cell_id_stable);
    d_cell_orientation_sign.copy_from_host(
        topology.cell_orientation_sign);
    core::DeviceArray<std::uint8_t> d_cell_nverts_eval(cell_nverts.size());
    d_cell_nverts_eval.copy_from_host(cell_nverts);
    candidate_quality = mesh::evaluate_candidate_mesh_quality_csr(
        state.x_r.data(),
        state.x_z.data(),
        d_delta_r.data(),
        d_delta_z.data(),
        1.0,
        n_cells,
        state.corner_stride,
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        d_cell_id_stable.data(),
        d_cell_orientation_sign.data(),
        floors,
        d_cell_nverts_eval.data(),
        nullptr,
        nullptr,
        0,
        state.x_r_reference.data(),
        state.x_z_reference.data());
  } else {
    candidate_quality = mesh::evaluate_candidate_mesh_quality(
        state.mesh,
        state.x_r.data(),
        state.x_z.data(),
        d_delta_r.data(),
        d_delta_z.data(),
        1.0,
        floors,
        state.x_r_reference.data(),
        state.x_z_reference.data());
  }

  const double j_dec_gate = cfg.numerics.ale.m1_min_j_dec_rel;
  const double j_dec_rel =
      (objective_before.value - objective_after) /
      std::max(std::abs(objective_before.value), 1.0e-300);
  bool accepted =
      objective_before.feasible &&
      std::isfinite(objective_after) &&
      (objective_before.value - objective_after) >
          j_dec_gate * std::max(std::abs(objective_before.value), 1.0e-300) &&
      candidate_quality.admissible();
  if (reduction != nullptr) {
    accepted =
        reduction->allreduce_min(
            static_cast<std::uint64_t>(accepted ? 1U : 0U)) != 0U;
  }

  out.iterations = cfg.numerics.ale.m1_sweeps;
  out.residual = objective_before.value - objective_after;
  if (accepted) {
    state.x_r.copy_from_host(current_r);
    state.x_z.copy_from_host(current_z);
    tx.telemetry_increment("m1_accepted");
    tx.accept();
    out.triggered = true;
    out.converged = true;
    out.accepted_lambda = 1.0;
    out.min_quality =
        std::min({candidate_quality.min_rz_volume_rel,
                  candidate_quality.min_corner_j_rel,
                  candidate_quality.min_gauss_j_rel});
  } else {
    tx.telemetry_increment("m1_rejected");
    tx.restore(state, nullptr);
    out.triggered = false;
    out.converged = false;
    out.accepted_lambda = 0.0;
  }

  if (part.rank == 0) {
    std::ostringstream log;
    log << "[m1-tmop]"
        << " step=" << (state.step + 1)
        << " accepted=" << (accepted ? "true" : "false")
        << " J_before="
        << detail::format_scientific(objective_before.value)
        << " J_after=" << detail::format_scientific(objective_after)
        << " feasible=" << (objective_before.feasible ? "1" : "0")
        << " J_dec="
        << (objective_after < objective_before.value ? "1" : "0")
        << " j_dec_rel=" << detail::format_scientific(j_dec_rel)
        << " j_gate=" << detail::format_scientific(j_dec_gate)
        << " admissible=" << (candidate_quality.admissible() ? "1" : "0")
        << " min_vol_rel="
        << detail::format_scientific(candidate_quality.min_rz_volume_rel)
        << " min_cj_rel="
        << detail::format_scientific(candidate_quality.min_corner_j_rel)
        << " min_gj_rel="
        << detail::format_scientific(candidate_quality.min_gauss_j_rel)
        << " kind=" << static_cast<int>(candidate_quality.kind)
        << " moved_nodes=" << moved_nodes;
    core::log_info(log.str());
  }
  return out;
}

}  // namespace

RezoneResult run_winslow_rezone_with_parallel(
    core::State& state, const core::Config& cfg,
    const parallel::PartitionInfo& part,
    parallel::CommBuffers* bufs,
    const parallel::Reduction* reduction,
    const bool force_rezone) {
  if (cfg.numerics.ale.rezone_solver == "m1_tmop") {
    return run_m1_tmop_rezone_transaction(
        state, cfg, part, reduction, force_rezone);
  }
  const bool do_node_exchange = (part.n_ranks > 1 && bufs != nullptr);
  if (!do_node_exchange && reduction == nullptr) {
    return run_winslow_rezone(state, cfg, force_rezone);
  }

  RezoneResult out;
  out.min_quality =
      compute_min_quality(state, cfg, &out.mesh_tangle, &out.min_quality_cell_pre);
  if (reduction != nullptr) {
    out.min_quality = reduction->allreduce_min(out.min_quality);
    out.mesh_tangle = (reduction->allreduce_max(out.mesh_tangle ? 1.0 : 0.0) > 0.5);
  }

  const bool corner_floor_violated =
      detail::corner_cell_aspect_floor_violated(state, cfg, reduction);
  if (!force_rezone && !corner_floor_violated &&
      out.min_quality >= cfg.numerics.ale.quality_threshold) {
    return out;
  }

  out.triggered = true;
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_nodes = state.mesh.topo.n_nodes;
  const int node_size = n_nodes;

  std::vector<std::uint8_t> node_flags = state.mesh.topo.node_flags;
  std::uint8_t* d_node_flags = nullptr;
  d_node_flags = static_cast<std::uint8_t*>(
      core::device_scratch_acquire("ale_driver:winslow_par:d_node_flags",
                                   static_cast<std::size_t>(n_nodes) * sizeof(std::uint8_t)));
  CUDA_CHECK(cudaMemcpy(d_node_flags,
                        node_flags.data(),
                        static_cast<std::size_t>(n_nodes) * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice));

  double* d_xr_a = nullptr;
  double* d_xz_a = nullptr;
  double* d_xr_b = nullptr;
  double* d_xz_b = nullptr;
  d_xr_a = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:winslow_par:d_xr_a",
                                   static_cast<std::size_t>(n_nodes) * sizeof(double)));
  d_xz_a = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:winslow_par:d_xz_a",
                                   static_cast<std::size_t>(n_nodes) * sizeof(double)));
  d_xr_b = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:winslow_par:d_xr_b",
                                   static_cast<std::size_t>(n_nodes) * sizeof(double)));
  d_xz_b = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:winslow_par:d_xz_b",
                                   static_cast<std::size_t>(n_nodes) * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_xr_a,
                        state.x_r.data(),
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_xz_a,
                        state.x_z.data(),
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice));

  double dl_min = 1.0;
  if (!state.mesh.cell_area.empty()) {
    dl_min = 1.0e300;
    for (double area : state.mesh.cell_area) {
      if (area > 0.0) {
        dl_min = std::min(dl_min, std::sqrt(area));
      }
    }
    if (!std::isfinite(dl_min) || !(dl_min > 0.0)) {
      dl_min = 1.0;
    }
  }
  if (reduction != nullptr) {
    dl_min = reduction->allreduce_min(dl_min);
    if (!std::isfinite(dl_min) || !(dl_min > 0.0)) {
      dl_min = 1.0;
    }
  }
  const double tol = cfg.numerics.ale.convergence_tol * dl_min;
  const double dr_init =
      (cfg.mesh.r_max - cfg.mesh.r_min) / static_cast<double>(cfg.mesh.nr);
  const double dz_init =
      (cfg.mesh.z_max - cfg.mesh.z_min) / static_cast<double>(cfg.mesh.nz);
  const double reference_lmin = std::max(std::min(dr_init, dz_init), 1.0e-30);
  const bool axis_z_motion_winslow =
      cfg.numerics.has_physical_rz_axis &&
      (cfg.numerics.ale.axis_z_motion == "winslow");
  const bool use_rz_full_metric_winslow =
      (cfg.numerics.ale.rezone_solver == "rz_full_metric_winslow");
  const tenryu::hydro::BC2DRZConfig bc_config =
      cfg.numerics.hydro.boundary_2d.make_bc_config();
  const int axis_nodes = nz + 1;
  std::vector<double> z_axis_old_h(static_cast<std::size_t>(axis_nodes), 0.0);
  std::vector<double> z_axis_target_h(static_cast<std::size_t>(axis_nodes), 0.0);
  std::vector<std::uint8_t> axis_flags_h(static_cast<std::size_t>(axis_nodes),
                                         mesh::NODE_NONE);
  double* d_z_axis_target = nullptr;
  if (axis_z_motion_winslow) {
    for (int j = 0; j <= nz; ++j) {
      axis_flags_h[static_cast<std::size_t>(j)] =
          node_flags[static_cast<std::size_t>(j)];
    }
    CUDA_CHECK(cudaMemcpy(z_axis_old_h.data(),
                          d_xz_a,
                          static_cast<std::size_t>(axis_nodes) * sizeof(double),
                          cudaMemcpyDeviceToHost));
    d_z_axis_target = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:winslow_par:d_z_axis_target",
                                     static_cast<std::size_t>(axis_nodes) * sizeof(double)));
  }

  double* d_max_delta = nullptr;
  AleRezoneIterStats* d_stats = nullptr;
  d_max_delta = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:winslow_par:d_max_delta", sizeof(double)));
  d_stats = static_cast<AleRezoneIterStats*>(
      core::device_scratch_acquire("ale_driver:winslow_par:d_stats",
                                   sizeof(AleRezoneIterStats)));

  double* cur_r = d_xr_a;
  double* cur_z = d_xz_a;
  double* nxt_r = d_xr_b;
  double* nxt_z = d_xz_b;

  const int blocks = (n_nodes + 255) / 256;
  out.converged = false;
  for (int iter = 0; iter < cfg.numerics.ale.max_iterations; ++iter) {
    CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(AleRezoneIterStats)));
    if (use_rz_full_metric_winslow) {
      detail::winslow_jacobi_step_rz_full_metric_kernel<<<blocks, 256>>>(
          nxt_r,
          nxt_z,
          cur_r,
          cur_z,
          d_node_flags,
          nr,
          nz,
          cfg.numerics.ale.max_displacement_fraction,
          reference_lmin,
          d_stats,
          axis_z_motion_winslow,
          cfg.numerics.has_physical_rz_axis,
          cfg.numerics.ale.rezone_local_admissibility_linesearch,
          cfg.numerics.ale.rezone_local_j_floor_rel,
          cfg.numerics.ale.rezone_local_linesearch_max_halves,
          state.x_r_initial.size() == state.x_r.size()
              ? state.x_r_initial.data()
              : nullptr,
          state.x_z_initial.size() == state.x_z.size()
              ? state.x_z_initial.data()
              : nullptr,
          state.x_r_reference.size() == state.x_r.size()
              ? state.x_r_reference.data()
              : nullptr,
          state.x_z_reference.size() == state.x_z.size()
              ? state.x_z_reference.data()
              : nullptr,
          bc_config,
          cfg.numerics.ale.corner_cell_aspect_protection_enabled,
          cfg.numerics.ale.corner_cell_aspect_eta,
          cfg.numerics.hydro.boundary_2d.z_bottom_cfg.is_state_supply(),
          cfg.numerics.hydro.boundary_2d.z_top_cfg.is_state_supply());
    } else {
      detail::winslow_jacobi_step_kernel<<<blocks, 256>>>(
          nxt_r,
          nxt_z,
          cur_r,
          cur_z,
          d_node_flags,
          nr,
          nz,
          cfg.numerics.ale.max_displacement_fraction,
          reference_lmin,
          d_stats,
          axis_z_motion_winslow,
          cfg.numerics.has_physical_rz_axis,
          state.x_r_initial.size() == state.x_r.size()
              ? state.x_r_initial.data()
              : nullptr,
          state.x_z_initial.size() == state.x_z.size()
              ? state.x_z_initial.data()
              : nullptr,
          state.x_r_reference.size() == state.x_r.size()
              ? state.x_r_reference.data()
              : nullptr,
          state.x_z_reference.size() == state.x_z.size()
              ? state.x_z_reference.data()
              : nullptr,
          bc_config,
          cfg.numerics.ale.corner_cell_aspect_protection_enabled,
          cfg.numerics.ale.corner_cell_aspect_eta,
          cfg.numerics.hydro.boundary_2d.z_bottom_cfg.is_state_supply(),
          cfg.numerics.hydro.boundary_2d.z_top_cfg.is_state_supply());
    }
    CUDA_CHECK(cudaGetLastError());

    if (axis_z_motion_winslow && !use_rz_full_metric_winslow) {
      detail::apply_axis_spine_projection(nxt_r,
                                          nxt_z,
                                          d_z_axis_target,
                                          z_axis_target_h,
                                          z_axis_old_h,
                                          axis_flags_h,
                                          nz,
                                          dz_init,
                                          cfg.numerics.has_physical_rz_axis);
    }

    if (do_node_exchange) {
      double* node_ptrs[2] = {nxt_r, nxt_z};
      parallel::exchange_node_fields(
          part, *bufs, node_ptrs, 2, node_size, nullptr, 6);
    }

    if (axis_z_motion_winslow) {
      const AxisRadialBoundResult bound = check_axis_radial_bound(
          nxt_r, d_xr_a, nr, nz, cfg.numerics.ale.winslow_axis_kappa, reduction);
      if (!bound.passed) {
        out.axis_radial_bound_failed = true;
        if (out.axis_radial_bound_fail_j < 0) {
          out.axis_radial_bound_fail_j = bound.fail_j;
        }
      }
    }

    const double zero = 0.0;
    CUDA_CHECK(cudaMemcpy(d_max_delta, &zero, sizeof(double), cudaMemcpyHostToDevice));
    detail::max_node_delta_kernel<<<blocks, 256>>>(nxt_r,
                                                   nxt_z,
                                                   cur_r,
                                                   cur_z,
                                                   d_max_delta,
                                                   n_nodes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(core::debug_kernel_sync());

    CUDA_CHECK(cudaMemcpy(&out.residual, d_max_delta, sizeof(double), cudaMemcpyDeviceToHost));
    if (reduction != nullptr) {
      out.residual = reduction->allreduce_max(out.residual);
    }
    AleRezoneIterStats iter_stats;
    CUDA_CHECK(cudaMemcpy(&iter_stats,
                          d_stats,
                          sizeof(AleRezoneIterStats),
                          cudaMemcpyDeviceToHost));
    detail::reduce_rezone_stats(iter_stats, reduction);
    detail::accumulate_rezone_stats(out.stats, iter_stats);
    out.iterations = iter + 1;

    double* old_cur_r = cur_r;
    double* old_cur_z = cur_z;
    cur_r = nxt_r;
    cur_z = nxt_z;
    nxt_r = old_cur_r;
    nxt_z = old_cur_z;

    bool converged_iter = (out.residual < tol);
    if (reduction != nullptr) {
      const double converged_local = converged_iter ? 1.0 : 0.0;
      converged_iter = (reduction->allreduce_min(converged_local) > 0.5);
    }
    if (converged_iter) {
      out.converged = true;
      break;
    }
  }

  detail::enforce_corner_cell_aspect_floor(cur_r, cur_z, state, cfg);

  CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                        cur_r,
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                        cur_z,
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice));

  bool post_tangle = false;
  (void)compute_min_quality(state, cfg, &post_tangle, &out.min_quality_cell_post);
  out.final_axis_margin =
      compute_axis_margin_min(state, cfg.numerics.has_physical_rz_axis).min_margin;
  if (reduction != nullptr) {
    out.final_axis_margin = reduction->allreduce_min(out.final_axis_margin);
  }

  if (!out.converged) {
    core::log_warning("ALE rezone did not converge within max_iterations: iter=" +
                      std::to_string(out.iterations) +
                      ", residual=" + std::to_string(out.residual));
    detail::log_ale_rezone_stats(out.iterations,
                                 out.stats,
                                 out.min_quality_cell_pre,
                                 out.min_quality_cell_post,
                                 out.final_axis_margin,
                                 0,
                                 "non-convergence");
  }

  return out;
}

RezoneResult run_axis_spine_rezone_with_parallel(
    core::State& state, const core::Config& cfg,
    const parallel::PartitionInfo& part,
    parallel::CommBuffers* bufs,
    const parallel::Reduction* reduction,
    const bool force_rezone) {
  RezoneResult out;
  out.min_quality =
      compute_min_quality(state, cfg, &out.mesh_tangle, &out.min_quality_cell_pre);
  if (reduction != nullptr) {
    out.min_quality = reduction->allreduce_min(out.min_quality);
    out.mesh_tangle = (reduction->allreduce_max(out.mesh_tangle ? 1.0 : 0.0) > 0.5);
  }

  if (!force_rezone && out.min_quality >= cfg.numerics.ale.quality_threshold) {
    return out;
  }
  if (!cfg.numerics.has_physical_rz_axis) {
    return out;
  }

  out.triggered = true;
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_nodes = state.mesh.topo.n_nodes;
  constexpr double kAxisSpineRepairAlpha = 0.5;
  constexpr int kAxisSpineRepairMaxSweeps = 10;
  constexpr double kAxisSpineRepairTolZ = 1.0e-12;

  double max_dz = 0.0;
  for (int sweep = 0; sweep < kAxisSpineRepairMaxSweeps; ++sweep) {
    max_dz = minimal_axis_z_repair(state.x_r,
                                   state.x_z,
                                   nr,
                                   nz,
                                   kAxisSpineRepairAlpha,
                                   1,
                                   kAxisSpineRepairTolZ);
    out.iterations = sweep + 1;
    if (max_dz < kAxisSpineRepairTolZ) {
      break;
    }
  }
  out.residual = max_dz;
  out.converged = (max_dz < kAxisSpineRepairTolZ);

  if (part.n_ranks > 1 && bufs != nullptr) {
    double* node_ptrs[2] = {state.x_r.data(), state.x_z.data()};
    parallel::exchange_node_fields(part, *bufs, node_ptrs, 2, n_nodes, nullptr, 6);
  }

  bool post_tangle = false;
  (void)compute_min_quality(state, cfg, &post_tangle, &out.min_quality_cell_post);
  out.final_axis_margin =
      compute_axis_margin_min(state, cfg.numerics.has_physical_rz_axis).min_margin;
  if (reduction != nullptr) {
    out.final_axis_margin = reduction->allreduce_min(out.final_axis_margin);
  }

  core::log_info("[ale] Phase 10 axis-spine-only repair: max|dz|=" +
                 std::to_string(max_dz));

  return out;
}

bool axis_rezone_topology_enabled(const core::State& state,
                                  const core::Config& cfg) {
  return cfg.numerics.ale.axis_rezone_enabled &&
         (cfg.mesh.topology_scheme ==
              core::TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_5BLOCK ||
          mesh::mesh_topo_has_trifan_cap(cfg.mesh)) &&
         state.mesh.topo.multiblock.has_value() &&
         state.mesh.topo.multiblock->block_count == 5;
}

bool axis_rezone_cache_needs_init(const detail::AxisRezoneCache& cache,
                                  const int n_nodes,
                                  const int n_cells) {
  return !cache.initialized || cache.n_nodes != n_nodes ||
         cache.n_cells != n_cells || cache.node_ids.empty();
}

// --- axis-rezone conservation ledger (env TENRYU_I1B_AXIS_REZONE_LEDGER,
// default OFF = zero cost). Stages the FIRE path (pre -> post-target-install
// -> post-remap -> post-apply) with global sums of stored mass, rho*V, V and
// corner mass plus the top-8 per-cell mass deltas, to pin WHERE the known
// per-fire mass injection (~1e-5 rel) enters. Global sums (640 cells) are
// cheap; patch scoping is unnecessary at this size.
bool axis_rezone_ledger_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_AXIS_REZONE_LEDGER");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  return enabled;
}

struct AxisRezoneLedgerSample {
  bool valid = false;
  double sum_mass = 0.0;
  double sum_rho_vol = 0.0;
  double sum_vol = 0.0;
  double sum_corner_mass = 0.0;
  double sum_U_member = 0.0;
  double sum_U_resolved = 0.0;
  double sum_K_member = 0.0;
  double sum_K_resolved = 0.0;
  std::vector<double> cell_mass;
};

AxisRezoneLedgerSample take_axis_rezone_ledger_sample(core::State& state) {
  AxisRezoneLedgerSample s;
  if (!axis_rezone_ledger_enabled()) {
    return s;
  }
  std::vector<double> mass;
  std::vector<double> rho;
  std::vector<double> vol;
  std::vector<double> corner;
  std::vector<double> ee_h;
  std::vector<double> ei_h;
  state.mass.copy_to_host(mass);
  state.rho.copy_to_host(rho);
  state.vol.copy_to_host(vol);
  state.corner_mass.copy_to_host(corner);
  state.ee.copy_to_host(ee_h);
  if (!state.ei.empty()) {
    state.ei.copy_to_host(ei_h);
  }
  const auto& pc_led = state.central_pseudo_core;
  for (std::size_t c = 0; c < mass.size(); ++c) {
    s.sum_mass += mass[c];
    if (c < rho.size() && c < vol.size()) {
      s.sum_rho_vol += rho[c] * vol[c];
    }
    if (c < vol.size()) {
      s.sum_vol += vol[c];
    }
    double u_cell = 0.0;
    if (c < ee_h.size()) {
      u_cell += mass[c] * ee_h[c];
    }
    if (c < ei_h.size()) {
      u_cell += mass[c] * ei_h[c];
    }
    const bool member =
        (c < pc_led.member_mask.size() && pc_led.member_mask[c] != 0U) ||
        (c < pc_led.passive_mask.size() && pc_led.passive_mask[c] != 0U) ||
        (c < pc_led.inactive_member_mask.size() &&
         pc_led.inactive_member_mask[c] != 0U);
    if (member) {
      s.sum_U_member += u_cell;
    } else {
      s.sum_U_resolved += u_cell;
    }
  }
  // Kinetic partition: direct corner sum 1/2 * m_corner * |v(node)|^2 (node
  // mass split by incident corners — equivalent to the nodal sum).
  if (state.mesh.topo.multiblock.has_value()) {
    std::vector<double> vr_h;
    std::vector<double> vz_h;
    state.v_r.copy_to_host(vr_h);
    state.v_z.copy_to_host(vz_h);
    const auto& mb_led = *state.mesh.topo.multiblock;
    const int n_nodes_led = state.mesh.topo.n_nodes;
    for (std::size_t c = 0; c < mass.size(); ++c) {
      const int off =
          mb_led.cell_node_csr_offsets[c];
      const int end = mb_led.cell_node_csr_offsets[c + 1U];
      const bool member =
          (c < pc_led.member_mask.size() && pc_led.member_mask[c] != 0U) ||
          (c < pc_led.passive_mask.size() && pc_led.passive_mask[c] != 0U) ||
          (c < pc_led.inactive_member_mask.size() &&
           pc_led.inactive_member_mask[c] != 0U);
      for (int k = off; k < end; ++k) {
        const int node = mb_led.cell_node_csr_indices[
            static_cast<std::size_t>(k)];
        const std::size_t slot = c * 4U + static_cast<std::size_t>(k - off);
        if (node < 0 || node >= n_nodes_led || slot >= corner.size()) {
          continue;
        }
        const double v2 =
            vr_h[static_cast<std::size_t>(node)] *
                vr_h[static_cast<std::size_t>(node)] +
            vz_h[static_cast<std::size_t>(node)] *
                vz_h[static_cast<std::size_t>(node)];
        const double ke = 0.5 * corner[slot] * v2;
        if (member) {
          s.sum_K_member += ke;
        } else {
          s.sum_K_resolved += ke;
        }
      }
    }
  }
  for (const double m : corner) {
    s.sum_corner_mass += m;
  }
  s.cell_mass = std::move(mass);
  s.valid = true;
  return s;
}

void report_axis_rezone_ledger_stage(const char* stage,
                                     const int step,
                                     const AxisRezoneLedgerSample& base,
                                     const AxisRezoneLedgerSample& cur) {
  if (!base.valid || !cur.valid) {
    return;
  }
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(9);
  const double denom = std::max(std::abs(base.sum_mass), 1.0e-300);
  oss << "[axis_rezone_ledger] step=" << step << " stage=" << stage
      << " sum_mass=" << cur.sum_mass
      << " dM=" << (cur.sum_mass - base.sum_mass)
      << " dM_rel=" << (cur.sum_mass - base.sum_mass) / denom
      << " d_rhoV=" << (cur.sum_rho_vol - base.sum_rho_vol)
      << " dV=" << (cur.sum_vol - base.sum_vol)
      << " d_corner_mass=" << (cur.sum_corner_mass - base.sum_corner_mass)
      << " dU_member=" << (cur.sum_U_member - base.sum_U_member)
      << " dU_resolved=" << (cur.sum_U_resolved - base.sum_U_resolved)
      << " dK_member=" << (cur.sum_K_member - base.sum_K_member)
      << " dK_resolved=" << (cur.sum_K_resolved - base.sum_K_resolved)
      << " U_member=" << cur.sum_U_member;
  if (cur.cell_mass.size() == base.cell_mass.size() &&
      !cur.cell_mass.empty()) {
    std::vector<std::pair<double, int>> dm;
    dm.reserve(cur.cell_mass.size());
    for (std::size_t c = 0; c < cur.cell_mass.size(); ++c) {
      const double d = cur.cell_mass[c] - base.cell_mass[c];
      if (d != 0.0) {
        dm.emplace_back(std::abs(d), static_cast<int>(c));
      }
    }
    const std::size_t k = std::min<std::size_t>(8, dm.size());
    std::partial_sort(dm.begin(), dm.begin() + k, dm.end(),
                      [](const std::pair<double, int>& a,
                         const std::pair<double, int>& b) {
                        return a.first > b.first;
                      });
    oss << " nonzero_cells=" << dm.size() << " top_dM=";
    for (std::size_t i = 0; i < k; ++i) {
      const int c = dm[i].second;
      oss << (i ? ";" : "") << c << ":"
          << (cur.cell_mass[static_cast<std::size_t>(c)] -
              base.cell_mass[static_cast<std::size_t>(c)]);
    }
  }
  core::log_warning(oss.str());
}

bool try_apply_axis_rezone(core::State& state,
                           const core::Config& cfg,
                           const parallel::PartitionInfo& part,
                           const parallel::Reduction* reduction,
                           const HydroEOSContext* eos_ctx,
                           const double dt_hydro_used,
                           AleStepResult& out,
                           const bool force_radial_order_repair = false) {
  if (!axis_rezone_topology_enabled(state, cfg)) {
    return false;
  }
  // Closure cooldown: after a closure-violating rezone commit was rejected,
  // scheduled fires stand down; the FORCED repair route stays available (its
  // own remap remains covered by the driver's closure gate).
  if (!force_radial_order_repair &&
      rezone_closure_cooldown_active(state.step)) {
    return false;
  }
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (n_cells <= 0 || n_nodes <= 0) {
    return false;
  }
  TENRYU_ASSERT(std::isfinite(dt_hydro_used) && dt_hydro_used > 0.0,
                "axis rezone requires a positive hydro dt");
  TENRYU_ASSERT(state.corner_mass_initialized &&
                    state.corner_mass.size() == static_cast<std::size_t>(n_cells) * 4U,
                "axis rezone requires initialized corner masses");

  std::vector<double> corner_mass;
  state.corner_mass.copy_to_host(corner_mass);
  corner_mass =
      detail::sanitize_axis_chain_corner_mass(state, std::move(corner_mass));
  const mesh::FullAxisNodeChain current_chain =
      mesh::build_full_axis_node_chain(state.mesh, corner_mass);
  if (!current_chain.active() || current_chain.node_ids.size() < 2U) {
    return false;
  }

  std::vector<double> node_r;
  std::vector<double> node_z;
  state.x_r.copy_to_host(node_r);
  state.x_z.copy_to_host(node_z);

  if (part.rank == 0) {
    detail::log_pole_shear_diag(state, node_r, node_z,
                                state.t + dt_hydro_used, 'L');
    detail::log_macro_boundary_diag(state, cfg, node_r, node_z,
                                    state.t + dt_hydro_used);
  }

  detail::AxisRezoneCache& cache = detail::axis_rezone_caches()[&state];
  if (axis_rezone_cache_needs_init(cache, n_nodes, n_cells) ||
      cache.node_ids != current_chain.node_ids) {
    const detail::AxisRezoneMetrics initial_metrics =
        detail::compute_axis_rezone_metrics(
            state, current_chain.node_ids, node_r, node_z);
    TENRYU_ASSERT(std::isfinite(initial_metrics.min_edge) &&
                      initial_metrics.min_edge > 0.0,
                  "axis rezone initial min edge must be positive");
    TENRYU_ASSERT(initial_metrics.sampled_incident_cell &&
                      std::isfinite(initial_metrics.min_altitude) &&
                      initial_metrics.min_altitude > 0.0,
                  "axis rezone initial min incident-cell altitude must be positive");
    for (std::size_t e = 0; e < initial_metrics.edge_length.size(); ++e) {
      TENRYU_ASSERT(std::isfinite(initial_metrics.edge_length[e]) &&
                        initial_metrics.edge_length[e] > 0.0,
                    "axis rezone initial edge length must be positive");
      TENRYU_ASSERT(std::isfinite(initial_metrics.adjacent_cell_area[e]) &&
                        initial_metrics.adjacent_cell_area[e] > 0.0,
                    "axis rezone initial adjacent cell area must be positive");
    }
    cache = detail::AxisRezoneCache{};
    cache.initialized = true;
    cache.n_nodes = n_nodes;
    cache.n_cells = n_cells;
    cache.core_quality_baseline =
        axis_ale::collect_axis_ale_core_quality_baseline(state.mesh);
    cache.node_ids = current_chain.node_ids;
    cache.initial_edge_length = initial_metrics.edge_length;
    cache.adjacent_cell_area = initial_metrics.adjacent_cell_area;
    cache.initial_min_edge = initial_metrics.min_edge;
    cache.initial_min_altitude = initial_metrics.min_altitude;
    const diagnostics::EnergyTotals initial_energy =
        detail::reduce_energy_totals_global(
            diagnostics::compute_energy_totals_2d(state), reduction);
    cache.compatible_initial_total = detail::total_energy(initial_energy);
    cache.compatible_E0 =
        std::max(std::abs(cache.compatible_initial_total), 1.0e-300);
  }

  const detail::AxisRezoneMetrics metrics =
      detail::compute_axis_rezone_metrics(state, cache.node_ids, node_r, node_z);
  TENRYU_ASSERT(std::isfinite(metrics.min_edge),
                "axis rezone current min edge must be finite");
  TENRYU_ASSERT(metrics.sampled_incident_cell &&
                    std::isfinite(metrics.min_altitude),
                "axis rezone current min incident-cell altitude must be finite");

  if (cache.last_fire_step == state.step && !force_radial_order_repair) {
    return false;
  }
  const double edge_ratio = metrics.min_edge / cache.initial_min_edge;
  const double altitude_ratio = metrics.min_altitude / cache.initial_min_altitude;
  const bool edge_trigger =
      edge_ratio < cfg.numerics.ale.axis_rezone_trigger_edge_fraction;
  const bool altitude_trigger =
      altitude_ratio <
      cfg.numerics.ale.axis_rezone_trigger_min_altitude_fraction;
  const bool axis_trigger = edge_trigger || altitude_trigger;

  const detail::AxisRezoneConvergencePredicate convergence =
      detail::evaluate_axis_rezone_convergence_predicate(
          state, cache, reduction);
  axis_ale::AxisAleCoreQualityTrigger core_quality_trigger;
  if (convergence.active) {
    core_quality_trigger = axis_ale::compute_axis_ale_core_quality_trigger(
        state.mesh,
        axis_ale::kAxisAleCoreShearTriggerThreshold,
        cache.core_quality_baseline.valid ? &cache.core_quality_baseline
                                          : nullptr);
    if (reduction != nullptr) {
      double max_values[2] = {core_quality_trigger.active ? 1.0 : 0.0,
                              core_quality_trigger.max_shear};
      reduction->allreduce_max(max_values, 2);
      double sampled_cells =
          static_cast<double>(core_quality_trigger.sampled_cells);
      reduction->allreduce_sum(&sampled_cells, 1);
      double min_corner_j_value =
          std::isfinite(core_quality_trigger.min_corner_j)
              ? -core_quality_trigger.min_corner_j
              : -std::numeric_limits<double>::infinity();
      reduction->allreduce_max(&min_corner_j_value, 1);
      core_quality_trigger.active = max_values[0] > 0.5;
      core_quality_trigger.max_shear = max_values[1];
      core_quality_trigger.sampled_cells =
          static_cast<int>(std::llround(sampled_cells));
      core_quality_trigger.sampled = core_quality_trigger.sampled_cells > 0;
      if (std::isfinite(min_corner_j_value)) {
        core_quality_trigger.min_corner_j = -min_corner_j_value;
      }
    }
  }
  const bool use_convergent_locality_target =
      convergence.active && core_quality_trigger.active;
  // Early-onset pole-sector arm: past T0 a non-empty pole target alone
  // opens the rezone transaction (identity axis target below). Computed
  // here, ahead of the gate, from the same node download the triggers use;
  // x_r/x_z are not mutated between here and the install site.
  axis_ale::AxisAlePatchTarget pole_target;
  if (detail::pole_sector_rezone_enabled(cfg)) {
    pole_target = detail::compute_pole_sector_equal_mu_target(state, cfg,
                                                              node_r, node_z);
  }
  const double pole_rezone_t0 = detail::pole_sector_rezone_t0_seconds();
  const bool pole_early_trigger =
      pole_target.active && pole_rezone_t0 >= 0.0 &&
      state.t >= pole_rezone_t0 &&
      state.t - cache.pole_last_fire_t >=
          detail::pole_sector_rezone_cooldown_t();
  // Arm provenance probe: inert unless the T0 env is set.
  if (pole_rezone_t0 >= 0.0 && part.rank == 0 &&
      ((state.step + 1) % 100) == 0) {
    core::log_info(
        "[pole_early_probe] step=" + std::to_string(state.step + 1) +
        " t=" + detail::format_scientific(state.t) +
        " t0=" + detail::format_scientific(pole_rezone_t0) +
        " enabled=" + (detail::pole_sector_rezone_enabled(cfg) ? "1" : "0") +
        " active=" + (pole_target.active ? "1" : "0") +
        " nodes=" + std::to_string(pole_target.patch_nodes) +
        " early=" + (pole_early_trigger ? "1" : "0") +
        " axis=" + (axis_trigger ? "1" : "0"));
  }
  // Convergence-following global rezone arm (verdict #5 Q1 Rank-1): on its
  // cadence, sample the POLAR_SHELL row metrics from the same node download
  // and open the rezone transaction alone (identity axis target) when the
  // angular roughness or shear metric trips.
  axis_ale::AxisAlePatchTarget conv_target;
  detail::ConvergenceRezoneMetrics conv_metrics;
  bool conv_trigger = false;
  if (detail::conv_rezone_enabled(cfg) &&
      state.t >= detail::conv_rezone_start_t() &&
      state.t - cache.conv_last_fire_t >= detail::conv_rezone_cooldown_t() &&
      ((state.step + 1) % detail::conv_rezone_every()) == 0) {
    conv_target = detail::compute_convergence_rezone_target(
        state, cfg, node_r, node_z, conv_metrics);
    conv_trigger =
        conv_target.active &&
        (detail::conv_rezone_force() ||
         conv_metrics.max_rough >= detail::conv_rezone_rough_trig() ||
         conv_metrics.max_shear >= detail::conv_rezone_shear_trig());
    if (part.rank == 0 && conv_metrics.sampled) {
      core::log_info(
          "[ale] conv_rezone_probe step=" + std::to_string(state.step + 1) +
          " t=" + detail::format_scientific(state.t) +
          " rough=" + detail::format_scientific(conv_metrics.max_rough) +
          "@q" + std::to_string(conv_metrics.rough_row) +
          " shear=" + detail::format_scientific(conv_metrics.max_shear) +
          "@q" + std::to_string(conv_metrics.shear_row) +
          " rows=" + std::to_string(conv_metrics.rows_sampled) +
          " nodes=" + std::to_string(conv_target.patch_nodes) +
          " trigger=" + (conv_trigger ? "1" : "0"));
    }
  }
  // TMOP polar patch arm (verdict #6 B1): on its cadence, scan the
  // POLAR_SHELL for warn-level q_shape/q_J/q_edge cells and open the rezone
  // transaction with a barrier-optimized local patch target.
  axis_ale::AxisAlePatchTarget tmop_target;
  detail::TmopPatchResult tmop_res;
  bool tmop_trigger = false;
  // Emergency-untangle mode: while an inverted-vs-target corner exists
  // (q_J < 0 seen on the last scan), the cooldown is waived — the fold is
  // flow-driven and the transactional guard already lambda-halves each
  // correction, so throttling fires on top loses the race outright.
  // Failure-path arm (I1-B-R candidate a): a forced repair invocation
  // (driver retry ladder) rides the same transaction, so the TMOP patch is
  // evaluated immediately at the failure instead of waiting for the
  // cadence — the rebound fold completes within one step's retry chain,
  // faster than any sampled trigger (measured, dyncore29). START_T still
  // gates it away from the fragile convergence era (dyncore28 lesson).
  if (detail::tmop_patch_enabled() &&
      state.t >= detail::tmop_start_t() &&
      (force_radial_order_repair ||
       ((cache.tmop_untangle_mode ||
         state.t - cache.tmop_last_fire_t >= detail::tmop_cooldown_t()) &&
        ((state.step + 1) % detail::tmop_patch_every()) == 0))) {
    std::vector<double> tmop_vr;
    std::vector<double> tmop_vz;
    state.v_r.copy_to_host(tmop_vr);
    state.v_z.copy_to_host(tmop_vz);
    tmop_target = detail::compute_tmop_patch_target(
        state, cfg, node_r, node_z, tmop_vr, tmop_vz, dt_hydro_used,
        tmop_res);
    tmop_trigger = tmop_target.active;
    cache.tmop_untangle_mode = tmop_res.triggered && tmop_res.worst_q_j < 0.0;
    // Un-blind the force path: on a ladder-invoked repair, log the measured
    // worst predicted-quality values even when nothing triggered, so the
    // thresholds/window/dt can be tuned against data instead of reruns.
    if (part.rank == 0 && force_radial_order_repair && tmop_res.sampled &&
        !tmop_res.triggered) {
      core::log_info(
          "[ale] tmop_force_probe step=" + std::to_string(state.step + 1) +
          " t=" + detail::format_scientific(state.t) +
          " dt=" + detail::format_scientific(dt_hydro_used) +
          " worst_cell=" + std::to_string(tmop_res.worst_cell) +
          " q_shape=" + detail::format_scientific(tmop_res.worst_q_shape) +
          " q_J=" + detail::format_scientific(tmop_res.worst_q_j) +
          " q_edge=" + detail::format_scientific(tmop_res.worst_q_edge));
    }
    if (part.rank == 0 && tmop_res.triggered) {
      core::log_info(
          "[ale] tmop_patch_probe step=" + std::to_string(state.step + 1) +
          " t=" + detail::format_scientific(state.t) +
          " worst_cell=" + std::to_string(tmop_res.worst_cell) +
          " q_shape=" + detail::format_scientific(tmop_res.worst_q_shape) +
          " q_J=" + detail::format_scientific(tmop_res.worst_q_j) +
          " q_edge=" + detail::format_scientific(tmop_res.worst_q_edge) +
          " cells=" + std::to_string(tmop_res.patch_cells) +
          " free=" + std::to_string(tmop_res.free_nodes) +
          " iters=" + std::to_string(tmop_res.iters_used) +
          " phi=" + detail::format_scientific(tmop_res.phi0) + "->" +
          detail::format_scientific(tmop_res.phi1) +
          " fire=" + (tmop_trigger ? "1" : "0"));
    }
  }
  if (!axis_trigger && !use_convergent_locality_target &&
      !force_radial_order_repair && !pole_early_trigger && !conv_trigger &&
      !tmop_trigger) {
    return false;
  }

  std::vector<double> lumped_mass_by_node(
      static_cast<std::size_t>(n_nodes),
      std::numeric_limits<double>::quiet_NaN());
  for (std::size_t i = 0; i < current_chain.node_ids.size(); ++i) {
    const int n = current_chain.node_ids[i];
    TENRYU_ASSERT(n >= 0 && n < n_nodes,
                  "axis rezone current chain node id out of range");
    lumped_mass_by_node[static_cast<std::size_t>(n)] =
        current_chain.lumped_mass[i];
  }

  axis_ale::AxisAleRezoneInput input;
  input.node_ids = cache.node_ids;
  input.z_tilde.reserve(cache.node_ids.size());
  input.lumped_mass.reserve(cache.node_ids.size());
  for (const int n : cache.node_ids) {
    input.z_tilde.push_back(node_z[static_cast<std::size_t>(n)]);
    const double mass = lumped_mass_by_node[static_cast<std::size_t>(n)];
    TENRYU_ASSERT(std::isfinite(mass) && mass >= 0.0,
                  "axis rezone current lumped node mass must be finite non-negative");
    input.lumped_mass.push_back(mass);
  }
  input.initial_edge_length = cache.initial_edge_length;
  input.adjacent_cell_area = cache.adjacent_cell_area;
  input.eta_floor = cfg.numerics.ale.axis_rezone_eta_floor;
  input.dt = dt_hydro_used;
  input.patch_winslow_iterations = cfg.numerics.ale.max_iterations;
  input.patch_winslow_relaxation =
      std::min(std::max(cfg.numerics.ale.relaxation, 0.0), 1.0);
  axis_ale::AxisAleRezoneResult target;
  if (axis_trigger || force_radial_order_repair) {
    target = axis_ale::compute_axis_ale_rezone_target(state.mesh, input);
  } else {
    target.active = true;
    target.node_ids = input.node_ids;
    target.z_target = input.z_tilde;
    if (state.mesh.topo.multiblock.has_value() &&
        state.mesh.topo.multiblock->has_trifan_cap &&
        state.mesh.topo.node_flags.size() == static_cast<std::size_t>(n_nodes)) {
      for (std::size_t i = 0; i < target.node_ids.size(); ++i) {
        const int n = target.node_ids[i];
        if (n >= 0 && n < n_nodes &&
            (state.mesh.topo.node_flags[static_cast<std::size_t>(n)] &
             mesh::NODE_CENTER) != 0U) {
          target.z_target[i] = 0.0;
        }
      }
    }
    target.first_off_axis_ring =
        axis_ale::compute_first_off_axis_ring_diagnostics(
            state.mesh, input.node_ids);
  }
  axis_ale::project_polar_shell_axis_radial_order_target(
      state.mesh, input, target);
  // Conservation contract (verdict #5 Q2 / Inc3a boundary-freeze lesson): the
  // axis-chain target must NOT move nodes on the outer PHYSICAL boundary (the
  // pole tips). A moved boundary node makes the global reference remap sweep
  // volume through donor-less domain-boundary faces, materializing mass at the
  // pole cells (measured: +9.4e-6 rel per fire, localized at cells 639/607).
  // Pin their target to the current position AFTER all target transforms.
  // SAME contract for the CORE-INTERIOR chain segment (measured 2026-07-02:
  // the first axis fire DOUBLES the pseudo-core internal energy pre-shock —
  // PAVA moves chain nodes inside the aggregated core, the global reference
  // remap then pumps energy into the member mirrors; the core's virtual
  // geometry is owned by rebuild_virtual_member_geometry, not the axis
  // rezone). Pin every chain node incident to a central-excluded cell.
  if (state.mesh.topo.node_flags.size() ==
      static_cast<std::size_t>(n_nodes)) {
    std::vector<std::uint8_t> core_node_mask;
    if (state.mesh.topo.multiblock.has_value() &&
        state.central_pseudo_core.built) {
      const auto& mb_pin = *state.mesh.topo.multiblock;
      const auto& pc_pin = state.central_pseudo_core;
      const int n_cells_pin = state.mesh.topo.n_cells;
      core_node_mask.assign(static_cast<std::size_t>(n_nodes), 0U);
      for (int c = 0; c < n_cells_pin; ++c) {
        const std::size_t ci = static_cast<std::size_t>(c);
        const bool excluded =
            (ci < pc_pin.member_mask.size() &&
             pc_pin.member_mask[ci] != 0U) ||
            (ci < pc_pin.passive_mask.size() &&
             pc_pin.passive_mask[ci] != 0U) ||
            (ci < pc_pin.inactive_member_mask.size() &&
             pc_pin.inactive_member_mask[ci] != 0U);
        if (!excluded) {
          continue;
        }
        const int off =
            mb_pin.cell_node_csr_offsets[static_cast<std::size_t>(c)];
        const int end =
            mb_pin.cell_node_csr_offsets[static_cast<std::size_t>(c) + 1U];
        for (int k = off; k < end; ++k) {
          const int node =
              mb_pin.cell_node_csr_indices[static_cast<std::size_t>(k)];
          if (node >= 0 && node < n_nodes) {
            core_node_mask[static_cast<std::size_t>(node)] = 1U;
          }
        }
      }
    }
    for (std::size_t i = 0; i < target.node_ids.size(); ++i) {
      const int n = target.node_ids[i];
      if (n < 0 || n >= n_nodes || i >= target.z_target.size() ||
          i >= input.z_tilde.size()) {
        continue;
      }
      const bool outer_boundary =
          (state.mesh.topo.node_flags[static_cast<std::size_t>(n)] &
           mesh::NODE_OUTER_PHYSICAL_BOUNDARY) != 0U;
      // Core-incident pinning REFUTED for the energy influx (2026-07-02:
      // 6 early Ue jumps unchanged, axis fires exploded 2k->9236 from the
      // lost relief) — energy path is elsewhere; keep only the proven
      // outer-boundary pin. core_node_mask retained for diagnostics.
      static_cast<void>(core_node_mask);
      if (outer_boundary) {
        target.z_target[i] = input.z_tilde[i];
      }
    }
  }
  if (!target.active) {
    return false;
  }
  bool install_convergent_locality_target = use_convergent_locality_target;
  std::string convergent_locality_fallback_reason =
      convergence.active && !core_quality_trigger.active
          ? "core_quality_inactive"
          : "none";
  axis_ale::AxisAlePatchTarget patch_target;
  axis_ale::AxisAleCellTargetAudit patch_cell_audit;
  if (use_convergent_locality_target) {
    patch_target =
        axis_ale::compute_axis_ale_patch_winslow_target(state.mesh, input, target);
    if (!patch_target.active || patch_target.node_ids.empty()) {
      install_convergent_locality_target = false;
      convergent_locality_fallback_reason = "patch_inactive";
      out.energy_audit_ale_fallback = true;
      if (part.rank == 0) {
        std::ostringstream oss;
        oss << "[ale] axis_rezone_patch_target_fallback"
            << " step=" << (state.step + 1)
            << " active=" << (patch_target.active ? "true" : "false")
            << " patch_install_nodes=" << patch_target.node_ids.size();
        core::log_warning(oss.str());
      }
    } else {
      patch_cell_audit =
          axis_ale::audit_axis_ale_patch_target_cell(
              state.mesh, input, target, patch_target, 113);
    }
    if ((!patch_target.volume_guard.sampled ||
         !patch_target.volume_guard.passed) &&
        install_convergent_locality_target) {
      install_convergent_locality_target = false;
      convergent_locality_fallback_reason = "volume_guard";
      out.energy_audit_ale_fallback = true;
      if (part.rank == 0) {
        std::ostringstream oss;
        oss << "[ale] axis_rezone_patch_volume_guard_fallback"
            << " step=" << (state.step + 1)
            << " sampled="
            << (patch_target.volume_guard.sampled ? "true" : "false")
            << " min_ratio_cell=" << patch_target.volume_guard.min_ratio_cell
            << " min_ratio="
            << detail::format_scientific(patch_target.volume_guard.min_ratio)
            << " current_volume="
            << detail::format_scientific(
                   patch_target.volume_guard.min_current_volume)
            << " target_volume="
            << detail::format_scientific(
                   patch_target.volume_guard.min_target_volume)
            << " contraction_floor_frac="
            << detail::format_scientific(
                   patch_target.volume_guard.contraction_floor_frac)
            << " aggregate_ratio="
            << detail::format_scientific(
                   patch_target.volume_guard.aggregate_ratio)
            << " aggregate_rel_tol="
            << detail::format_scientific(
                   patch_target.volume_guard.aggregate_rel_tol);
        core::log_warning(oss.str());
      }
    }
    if (patch_cell_audit.sampled && !patch_cell_audit.passed) {
      install_convergent_locality_target = false;
      if (convergent_locality_fallback_reason == "none") {
        convergent_locality_fallback_reason = "cell113_audit";
      }
      out.energy_audit_ale_fallback = true;
      if (part.rank == 0) {
        std::ostringstream oss;
        oss << "[ale] axis_rezone_cell113_audit_fallback"
            << " step=" << (state.step + 1)
            << " cell=" << patch_cell_audit.cell
            << " nodes=" << patch_cell_audit.nodes[0] << ","
            << patch_cell_audit.nodes[1] << ","
            << patch_cell_audit.nodes[2] << ","
            << patch_cell_audit.nodes[3]
            << " meaningful_shear="
            << (patch_cell_audit.meaningful_shear ? "true" : "false")
            << " shear_threshold="
            << detail::format_scientific(patch_cell_audit.shear_threshold)
            << " shear_reduced="
            << (patch_cell_audit.shear_reduced ? "true" : "false")
            << " corner_j_ok="
            << (patch_cell_audit.corner_j_ok ? "true" : "false")
            << " axis_only_shear="
            << detail::format_scientific(patch_cell_audit.axis_only.shear)
            << " patch_shear="
            << detail::format_scientific(patch_cell_audit.patch.shear)
            << " axis_only_aspect="
            << detail::format_scientific(
                   patch_cell_audit.axis_only.aspect_ratio)
            << " patch_aspect="
            << detail::format_scientific(patch_cell_audit.patch.aspect_ratio)
            << " axis_only_min_corner_j="
            << detail::format_scientific(
                   patch_cell_audit.axis_only.min_corner_j)
            << " patch_min_corner_j="
            << detail::format_scientific(patch_cell_audit.patch.min_corner_j);
        core::log_warning(oss.str());
      }
    }
  }
  if (!axis_trigger && !install_convergent_locality_target &&
      !force_radial_order_repair && !pole_early_trigger) {
    return false;
  }
  ale_velcoherence::sample(state, cfg, "s0_post_hydro");
  const std::string axis_target_trace_extra =
      format_cell113_axis_target_extra(state, cfg, input, target);
  log_cell113_substage_trace(state,
                             cfg,
                             part,
                             "axis_rezone_target",
                             dt_hydro_used,
                             state.t + dt_hydro_used,
                             axis_target_trace_extra.c_str());
  TENRYU_ASSERT(state.x_r_reference.size() == static_cast<std::size_t>(n_nodes) &&
                    state.x_z_reference.size() == static_cast<std::size_t>(n_nodes),
                "axis rezone requires reference node storage");
  TENRYU_ASSERT(state.cell_vol_initial.size() == static_cast<std::size_t>(n_cells),
                "axis rezone requires reference volume storage");

  cache.last_fire_step = state.step;
  ++cache.fire_count;
  if (install_convergent_locality_target) {
    ++cache.convergent_locality_engaged_steps;
  }

  const diagnostics::EnergyTotals energy_before =
      detail::reduce_energy_totals_global(
          diagnostics::compute_energy_totals_2d(state), reduction);
  const double total_before = detail::total_energy(energy_before);

  const std::size_t node_bytes = static_cast<std::size_t>(n_nodes) * sizeof(double);
  const std::size_t cell_bytes = static_cast<std::size_t>(n_cells) * sizeof(double);
  double* d_xr_old = nullptr;
  double* d_xz_old = nullptr;
  double* d_vol_old = nullptr;
  double* d_ref_r_old = nullptr;
  double* d_ref_z_old = nullptr;
  double* d_cell_vol_initial_old = nullptr;
  d_xr_old = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:axis_rezone:d_xr_old", node_bytes));
  d_xz_old = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:axis_rezone:d_xz_old", node_bytes));
  d_vol_old = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:axis_rezone:d_vol_old", cell_bytes));
  d_ref_r_old = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:axis_rezone:d_ref_r_old", node_bytes));
  d_ref_z_old = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:axis_rezone:d_ref_z_old", node_bytes));
  d_cell_vol_initial_old = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:axis_rezone:d_cell_vol_initial_old", cell_bytes));
  const auto cleanup = [&]() {
  };
  const auto restore_persistent_reference = [&]() {
    CUDA_CHECK(cudaMemcpy(state.x_r_reference.data(),
                          d_ref_r_old,
                          node_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.x_z_reference.data(),
                          d_ref_z_old,
                          node_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.cell_vol_initial.data(),
                          d_cell_vol_initial_old,
                          cell_bytes,
                          cudaMemcpyDeviceToDevice));
  };
  CUDA_CHECK(cudaMemcpy(d_xr_old,
                        state.x_r.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_xz_old,
                        state.x_z.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_vol_old,
                        state.vol.data(),
                        cell_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_ref_r_old,
                        state.x_r_reference.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_ref_z_old,
                        state.x_z_reference.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_cell_vol_initial_old,
                        state.cell_vol_initial.data(),
                        cell_bytes,
                        cudaMemcpyDeviceToDevice));

  CUDA_CHECK(cudaMemcpy(state.x_r_reference.data(),
                        state.x_r.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(state.x_z_reference.data(),
                        state.x_z.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice));
  detail::install_axis_rezone_target(state.x_r_reference.data(),
                                     state.x_z_reference.data(),
                                     target.node_ids,
                                     target.z_target);
  if (install_convergent_locality_target) {
    detail::install_axis_rezone_patch_target(state.x_r_reference.data(),
                                             state.x_z_reference.data(),
                                             patch_target);
  }
  // Convergence-following global target installs before the pole-sector
  // target so the specialized equal-mu pole treatment wins on any
  // overlapping pole-sector nodes when both are enabled.
  if (tmop_trigger && tmop_target.active) {
    detail::install_axis_rezone_patch_target(state.x_r_reference.data(),
                                             state.x_z_reference.data(),
                                             tmop_target);
    cache.tmop_last_fire_t = state.t;
    if (part.rank == 0) {
      core::log_info(
          "[ale] tmop_patch_fire step=" + std::to_string(state.step + 1) +
          " t=" + detail::format_scientific(state.t) +
          " nodes=" + std::to_string(tmop_target.patch_nodes) +
          " worst_cell=" + std::to_string(tmop_res.worst_cell));
    }
  }
  if (conv_trigger && conv_target.active) {
    detail::install_axis_rezone_patch_target(state.x_r_reference.data(),
                                             state.x_z_reference.data(),
                                             conv_target);
    cache.conv_last_fire_t = state.t;
    if (part.rank == 0) {
      core::log_info(
          "[ale] conv_rezone_fire step=" + std::to_string(state.step + 1) +
          " t=" + detail::format_scientific(state.t) +
          " nodes=" + std::to_string(conv_target.patch_nodes) +
          " rough=" + detail::format_scientific(conv_metrics.max_rough) +
          " shear=" + detail::format_scientific(conv_metrics.max_shear));
    }
  }
  if (detail::pole_sector_rezone_enabled(cfg)) {
    if (pole_target.active) {
      detail::install_axis_rezone_patch_target(state.x_r_reference.data(),
                                               state.x_z_reference.data(),
                                               pole_target);
      cache.pole_last_fire_t = state.t;
      if (part.rank == 0 && (cache.fire_count % 500 == 1)) {
        core::log_info(
            "[ale] pole_sector_equal_mu step=" +
            std::to_string(state.step + 1) +
            " nodes=" + std::to_string(pole_target.patch_nodes) +
            " m_theta=" +
            std::to_string(detail::pole_sector_rezone_m_theta(cfg)) +
            " q_min=" +
            std::to_string(detail::pole_sector_rezone_q_min()) +
            " lambda=" +
            detail::format_scientific(
                detail::pole_sector_rezone_lambda(cfg)));
      }
    }
  }
  core::DeviceArray<std::uint8_t> core_freeze_frozen_nodes;
  core_freeze::restore_target_if_enabled(
      state,
      cfg,
      state.x_r_reference.data(),
      state.x_z_reference.data(),
      d_xr_old,
      d_xz_old,
      true,
      "axis_rezone",
      cfg.numerics.ale.core_freeze_skip_velocity_projection
          ? &core_freeze_frozen_nodes
          : nullptr);

  // PR2 of the corner-mass basis-contract verdict (interim stabilization
  // rank 1): macro-adjacent TAPERED rezone. The PR1 basis-defect audit
  // quantitatively closed the mixed-core budget drift (Sum R ~ dE) and
  // localized the dominant share to the macro boundary band
  // (band_dW = -1.4e8 erg of a +3.6e8 budget; net OUTWARD spurious
  // impulse band_dI_r = +9.9 g cm/s ~ 30% of the rim velocity scale =
  // implicates the rebound under-compression). Remap activity in that
  // band is the source term, so the rezone reference is blended back to
  // the Lagrangian positions there: lambda = 0 for the boundary shell row
  // and HOLD rows beyond it, ramping linearly to 1 over RAMP further rows.
  // The tapered target rides the existing transactional guard unchanged.
  // Env-gated, default off (TENRYU_I1B_MACRO_BAND_REZONE_TAPER, _HOLD=2,
  // _RAMP=4).
  if (detail::macro_band_rezone_taper_enabled() &&
      state.step > state.central_pseudo_core.taper_lift_until_step &&
      state.central_pseudo_core.built &&
      state.mesh.topo.multiblock.has_value() &&
      state.mesh.topo.multiblock->has_trifan_cap) {
    const auto& mb_taper = *state.mesh.topo.multiblock;
    const mesh::BlockInfo* shell_tp = nullptr;
    int north_rows_tp = 0;
    for (const auto& block : mb_taper.blocks) {
      if (block.role == mesh::BlockRole::POLAR_SHELL) {
        shell_tp = &block;
      } else if (block.role == mesh::BlockRole::NORTH_FAN) {
        north_rows_tp = block.n_i_cells;
      }
    }
    if (shell_tp != nullptr && shell_tp->n_j_cells >= 2) {
      const int q_boundary =
          state.central_pseudo_core.member_ring_count - mb_taper.n_cap -
          north_rows_tp;
      const int hold = detail::macro_band_rezone_taper_hold();
      const int ramp = detail::macro_band_rezone_taper_ramp();
      const int ntheta_tp = shell_tp->n_j_cells;
      const int n_rows_tp = shell_tp->n_i_cells + 1;
      std::vector<double> ref_r_tp(static_cast<std::size_t>(n_nodes));
      std::vector<double> ref_z_tp(static_cast<std::size_t>(n_nodes));
      CUDA_CHECK(cudaMemcpy(ref_r_tp.data(), state.x_r_reference.data(),
                            node_bytes, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(ref_z_tp.data(), state.x_z_reference.data(),
                            node_bytes, cudaMemcpyDeviceToHost));
      int tapered_nodes = 0;
      for (int q = std::max(q_boundary, 0); q < n_rows_tp; ++q) {
        const int dq = q - std::max(q_boundary, 0);
        double lam = 1.0;
        if (dq <= hold) {
          lam = 0.0;
        } else if (dq <= hold + ramp) {
          lam = static_cast<double>(dq - hold) / static_cast<double>(ramp);
        } else {
          break;
        }
        const int row_base =
            shell_tp->owned_node_begin + q * (ntheta_tp + 1);
        for (int k = 0; k <= ntheta_tp; ++k) {
          const int n = row_base + k;
          if (n < 0 || n >= n_nodes) {
            continue;
          }
          const std::size_t nn = static_cast<std::size_t>(n);
          ref_r_tp[nn] = node_r[nn] + lam * (ref_r_tp[nn] - node_r[nn]);
          ref_z_tp[nn] = node_z[nn] + lam * (ref_z_tp[nn] - node_z[nn]);
          ++tapered_nodes;
        }
      }
      if (tapered_nodes > 0) {
        CUDA_CHECK(cudaMemcpy(state.x_r_reference.data(), ref_r_tp.data(),
                              node_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state.x_z_reference.data(), ref_z_tp.data(),
                              node_bytes, cudaMemcpyHostToDevice));
        if (part.rank == 0 && (cache.fire_count % 500 == 1)) {
          core::log_info(
              "[ale] macro_band_rezone_taper q_boundary=" +
              std::to_string(q_boundary) + " hold=" + std::to_string(hold) +
              " ramp=" + std::to_string(ramp) +
              " nodes=" + std::to_string(tapered_nodes));
        }
      }
    }
  }

  // Transactional rezone guard: never commit a rezone whose straight-line
  // node paths fold a cell or break the macro boundary loop. The SAME
  // evaluator gates Lagrangian trials (all cells, inactive-member aware,
  // macro-boundary volume + simple-loop). On failure the target is blended
  // toward the pre-rezone positions (lambda halving); if no fraction is
  // admissible the rezone is skipped for this step.
  {
    const auto rezone_path_failed =
        [](const mesh::PathAdmissibilityResult& p) {
          return p.first_failing_cell != -1 && p.first_failing_lambda >= 0.0 &&
                 p.first_failing_lambda < 1.0;
        };
    int rezone_path_halvings = 0;
    auto rezone_path = mesh::evaluate_path_admissibility(
        state,
        cfg,
        d_xr_old,
        d_xz_old,
        state.x_r_reference.data(),
        state.x_z_reference.data(),
        cfg.numerics.ale.path_admissibility_floor,
        nullptr,
        state.x_r_reference.size() == state.x_r.size()
            ? state.x_r_reference.data()
            : nullptr,
        state.x_z_reference.size() == state.x_z.size()
            ? state.x_z_reference.data()
            : nullptr);
    while (rezone_path_failed(rezone_path) && rezone_path_halvings < 4) {
      ++rezone_path_halvings;
      std::vector<double> ref_r_h(static_cast<std::size_t>(n_nodes));
      std::vector<double> ref_z_h(static_cast<std::size_t>(n_nodes));
      std::vector<double> old_r_h(static_cast<std::size_t>(n_nodes));
      std::vector<double> old_z_h(static_cast<std::size_t>(n_nodes));
      CUDA_CHECK(cudaMemcpy(ref_r_h.data(),
                            state.x_r_reference.data(),
                            node_bytes,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(ref_z_h.data(),
                            state.x_z_reference.data(),
                            node_bytes,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(old_r_h.data(), d_xr_old, node_bytes,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(old_z_h.data(), d_xz_old, node_bytes,
                            cudaMemcpyDeviceToHost));
      for (int n = 0; n < n_nodes; ++n) {
        const std::size_t i = static_cast<std::size_t>(n);
        ref_r_h[i] = old_r_h[i] + 0.5 * (ref_r_h[i] - old_r_h[i]);
        ref_z_h[i] = old_z_h[i] + 0.5 * (ref_z_h[i] - old_z_h[i]);
      }
      CUDA_CHECK(cudaMemcpy(state.x_r_reference.data(),
                            ref_r_h.data(),
                            node_bytes,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(state.x_z_reference.data(),
                            ref_z_h.data(),
                            node_bytes,
                            cudaMemcpyHostToDevice));
      rezone_path = mesh::evaluate_path_admissibility(
          state,
          cfg,
          d_xr_old,
          d_xz_old,
          state.x_r_reference.data(),
          state.x_z_reference.data(),
          cfg.numerics.ale.path_admissibility_floor,
          nullptr,
          state.x_r_reference.size() == state.x_r.size()
              ? state.x_r_reference.data()
              : nullptr,
          state.x_z_reference.size() == state.x_z.size()
              ? state.x_z_reference.data()
              : nullptr);
    }
    if (rezone_path_failed(rezone_path)) {
      if (part.rank == 0) {
        std::ostringstream oss;
        oss << "[ale] axis_rezone_path_inadmissible_skip step="
            << (state.step + 1)
            << " first_cell=" << rezone_path.first_failing_cell
            << " lambda="
            << detail::format_scientific(rezone_path.first_failing_lambda)
            << " min_margin="
            << detail::format_scientific(rezone_path.min_margin)
            << " halvings=" << rezone_path_halvings;
        core::log_warning(oss.str());
      }
      out.energy_audit_ale_fallback = true;
      restore_persistent_reference();
      cleanup();
      return false;
    }
    if (rezone_path_halvings > 0 && part.rank == 0) {
      std::ostringstream oss;
      oss << "[ale] axis_rezone_path_blend step=" << (state.step + 1)
          << " halvings=" << rezone_path_halvings
          << " min_margin="
          << detail::format_scientific(rezone_path.min_margin);
      core::log_info(oss.str());
    }
  }

  const AxisRezoneLedgerSample axis_ledger_pre =
      take_axis_rezone_ledger_sample(state);
  CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                        state.x_r_reference.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                        state.x_z_reference.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  state.mesh.recompute_geometry();
  state.cell_vol_initial.copy_from_host(state.mesh.cell_vol);
  ale_velcoherence::sample(state, cfg, "s1_post_rezone");
  record_estep_trace_boundary(state,
                              cfg,
                              part,
                              reduction,
                              "post_rezone_pre_remap",
                              dt_hydro_used,
                              state.t + dt_hydro_used);

  CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                        d_xr_old,
                        node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                        d_xz_old,
                        node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(state.vol.data(),
                        d_vol_old,
                        cell_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());

  report_axis_rezone_ledger_stage("post_target_install_pre_remap",
                                  state.step + 1, axis_ledger_pre,
                                  take_axis_rezone_ledger_sample(state));
  core::Config remap_cfg = cfg;
  remap_cfg.numerics.ale.conservative_remap_enabled = true;
  remap_cfg.numerics.ale.conservative_remap_target = "reference";
  remap_cfg.numerics.ale.multiblock_scaled_reference_enabled = false;
  const std::uint8_t* core_freeze_velocity_mask =
      core_freeze_frozen_nodes.size() == static_cast<std::size_t>(n_nodes)
          ? core_freeze_frozen_nodes.data()
          : nullptr;
  const auto remap_result =
      ale_remap_2d_rz(
          state, remap_cfg, eos_ctx, dt_hydro_used, core_freeze_velocity_mask);
  record_estep_trace_boundary(state,
                              cfg,
                              part,
                              reduction,
                              "post_remap",
                              dt_hydro_used,
                              state.t + dt_hydro_used);
  out.rezone_triggered = true;
  out.rezone_converged = remap_result.applied;
  out.rezone_iterations = 1;
  out.rezone_residual =
      std::max(std::abs(1.0 - edge_ratio), std::abs(1.0 - altitude_ratio));
  out.applied = remap_result.applied;
  out.mass_floor_delta += remap_result.mass_floor_delta;
  out.E_floor_injected += remap_result.E_floor_injected;
  out.E_redistribution_unresolved += remap_result.E_redistribution_unresolved;
  out.cap_energy_audit_D_K += remap_result.cap_energy_audit_D_K;
  out.eta_contact_step = remap_result.eta_contact_step;
  out.eta_contact_cumulative += remap_result.eta_contact_step;
  out.i1b_ale_ke_sensor = remap_result.i1b_ale_ke_sensor;
  out.accepted_remap_count = state.ale_remaps_applied;
  report_axis_rezone_ledger_stage("post_remap", state.step + 1,
                                  axis_ledger_pre,
                                  take_axis_rezone_ledger_sample(state));
  restore_persistent_reference();

  if (out.applied) {
    state.ale_rezoned = true;
    ++state.ale_rezone_invocations;
    state.ale_last_applied_step = state.step;
    state.holo_ale_invalidated = true;
  } else {
    CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                          d_xr_old,
                          node_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                          d_xz_old,
                          node_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.vol.data(),
                          d_vol_old,
                          cell_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    state.mesh.recompute_geometry();
    state.vol = state.mesh.cell_vol;
  }

  report_axis_rezone_ledger_stage("post_apply", state.step + 1,
                                  axis_ledger_pre,
                                  take_axis_rezone_ledger_sample(state));
  std::ostringstream axis_post_trace_extra;
  axis_post_trace_extra << "remap_applied=" << (out.applied ? "true" : "false")
                        << " axis_trigger="
                        << (axis_trigger ? "true" : "false")
                        << " core_quality_trigger="
                        << (use_convergent_locality_target ? "true" : "false")
                        << " radial_order_repair="
                        << (force_radial_order_repair ? "true" : "false")
                        << " edge_ratio="
                        << detail::format_scientific(edge_ratio)
                        << " altitude_ratio="
                        << detail::format_scientific(altitude_ratio)
                        << " axis_rezone_count=" << cache.fire_count;
  const std::string axis_post_trace_extra_text = axis_post_trace_extra.str();
  log_cell113_substage_trace(state,
                             cfg,
                             part,
                             "axis_rezone_post_remap",
                             dt_hydro_used,
                             state.t + dt_hydro_used,
                             axis_post_trace_extra_text.c_str());

  const diagnostics::EnergyTotals energy_after =
      detail::reduce_energy_totals_global(
          diagnostics::compute_energy_totals_2d(state), reduction);
  const double total_after = detail::total_energy(energy_after);
  const double compatible_step_residual_rel =
      std::abs(total_after - total_before) / cache.compatible_E0;
  const double compatible_cumulative_residual_rel =
      std::abs(total_after - cache.compatible_initial_total) /
      cache.compatible_E0;
  if (part.rank == 0) {
    std::ostringstream oss;
    oss << "[ale] axis_rezone_fire step=" << (state.step + 1)
        << " min_edge=" << detail::format_scientific(metrics.min_edge)
        << " min_altitude=" << detail::format_scientific(metrics.min_altitude)
        << " edge_ratio=" << detail::format_scientific(edge_ratio)
        << " altitude_ratio=" << detail::format_scientific(altitude_ratio)
        << " off_axis_ring_min_edge="
        << detail::format_scientific(target.first_off_axis_ring.min_edge_length)
        << " off_axis_ring_min_altitude="
        << detail::format_scientific(target.first_off_axis_ring.min_altitude)
        << " axis_rezone_count=" << cache.fire_count
        << " remap_applied=" << (out.applied ? "true" : "false")
        << " compatible_energy_step_residual_rel="
        << detail::format_scientific(compatible_step_residual_rel)
        << " compatible_energy_cumulative_residual_rel="
        << detail::format_scientific(compatible_cumulative_residual_rel)
        << " axis_trigger=" << (axis_trigger ? "true" : "false")
        << " core_quality_trigger="
        << (use_convergent_locality_target ? "true" : "false")
        << " radial_order_repair="
        << (force_radial_order_repair ? "true" : "false");
    core::log_info(oss.str());
  }
  if (part.rank == 0 &&
      detail::axis_rezone_convergent_locality_diag_enabled()) {
    std::ostringstream oss;
    oss << "[ale] axis_rezone_convergent_locality"
        << " step=" << (state.step + 1)
        << " active=" << (convergence.active ? "true" : "false")
        << " core_quality_sampled="
        << (core_quality_trigger.sampled ? "true" : "false")
        << " core_quality_trigger="
        << (use_convergent_locality_target ? "true" : "false")
        << " core_quality_sampled_cells="
        << core_quality_trigger.sampled_cells
        << " core_quality_max_shear="
        << detail::format_scientific(core_quality_trigger.max_shear)
        << " core_quality_max_shear_cell="
        << core_quality_trigger.max_shear_cell
        << " core_quality_shear_threshold="
        << detail::format_scientific(core_quality_trigger.shear_threshold)
        << " core_quality_min_corner_j="
        << detail::format_scientific(core_quality_trigger.min_corner_j)
        << " core_quality_min_corner_j_cell="
        << core_quality_trigger.min_corner_j_cell
        << " patch_installed="
        << (install_convergent_locality_target ? "true" : "false")
        << " fallback_reason=" << convergent_locality_fallback_reason
        << " alpha=" << detail::format_scientific(convergence.alpha)
        << " s_dot=" << detail::format_scientific(convergence.s_dot)
        << " sampled_outer_nodes=" << convergence.sampled_nodes
        << " patch_nodes=" << patch_target.patch_nodes
        << " core_nodes=" << patch_target.core_nodes
        << " seam_nodes=" << patch_target.seam_nodes
        << " fan_transition_nodes=" << patch_target.fan_transition_nodes
        << " patch_install_nodes=" << patch_target.node_ids.size()
        << " convergent_locality_engaged_steps="
        << cache.convergent_locality_engaged_steps;
    if (use_convergent_locality_target) {
      oss << " volume_guard_passed="
          << (patch_target.volume_guard.passed ? "true" : "false")
          << " volume_guard_min_ratio="
          << detail::format_scientific(patch_target.volume_guard.min_ratio)
          << " volume_guard_cell="
          << patch_target.volume_guard.min_ratio_cell
          << " volume_guard_aggregate_ratio="
          << detail::format_scientific(
                 patch_target.volume_guard.aggregate_ratio)
          << " volume_guard_contraction_floor_frac="
          << detail::format_scientific(
                 patch_target.volume_guard.contraction_floor_frac);
    }
    if (use_convergent_locality_target && patch_cell_audit.sampled) {
      oss << " audit_cell=" << patch_cell_audit.cell
          << " audit_nodes=" << patch_cell_audit.nodes[0] << ","
          << patch_cell_audit.nodes[1] << ","
          << patch_cell_audit.nodes[2] << ","
          << patch_cell_audit.nodes[3]
          << " axis_only_shear="
          << detail::format_scientific(patch_cell_audit.axis_only.shear)
          << " patch_shear="
          << detail::format_scientific(patch_cell_audit.patch.shear)
          << " axis_only_aspect="
          << detail::format_scientific(
                 patch_cell_audit.axis_only.aspect_ratio)
          << " patch_aspect="
          << detail::format_scientific(patch_cell_audit.patch.aspect_ratio)
          << " axis_only_min_corner_j="
          << detail::format_scientific(
                 patch_cell_audit.axis_only.min_corner_j)
          << " patch_min_corner_j="
          << detail::format_scientific(patch_cell_audit.patch.min_corner_j)
          << " audit_meaningful_shear="
          << (patch_cell_audit.meaningful_shear ? "true" : "false")
          << " audit_shear_threshold="
          << detail::format_scientific(patch_cell_audit.shear_threshold)
          << " audit_shear_reduced="
          << (patch_cell_audit.shear_reduced ? "true" : "false")
          << " audit_corner_j_ok="
          << (patch_cell_audit.corner_j_ok ? "true" : "false")
          << " audit_passed="
          << (patch_cell_audit.passed ? "true" : "false");
    }
    core::log_info(oss.str());
  }
  record_estep_trace_boundary(state,
                              cfg,
                              part,
                              reduction,
                              "step_end",
                              dt_hydro_used,
                              state.t + dt_hydro_used);
  cleanup();
  return true;
}

AleStepResult apply_pole_axis_radial_order_repair(
    core::State& state,
    const core::Config& cfg,
    const parallel::PartitionInfo& part,
    const parallel::Reduction* reduction,
    const HydroEOSContext* eos_ctx,
    const double dt_hydro_used) {
  AleStepResult out;
  (void)try_apply_axis_rezone(state,
                              cfg,
                              part,
                              reduction,
                              eos_ctx,
                              dt_hydro_used,
                              out,
                              true);
  return out;
}

namespace {
int g_rezone_closure_cooldown_until_step = -1;

tenryu::hydro::ReferenceBarrierAleResult
apply_reference_barrier_rezone_transaction(
    core::State& state,
    const core::Config& cfg,
    const double* d_target_r,
    const double* d_target_z) {
  // TransactionClientKind::kReferenceBarrierRezone maps to
  // MeshEventKind::kRSameTopology; formal client-kind attachment arrives with M1.
  RollbackGuard tx;
  tx.capture(state, nullptr);
  auto result = tenryu::hydro::apply_reference_barrier_rezone(
      state, cfg, d_target_r, d_target_z);
  const bool inject_precommit_failure =
      cfg.numerics.ale.transaction_failure_inject_point == 10;
  if (result.succeeded && !inject_precommit_failure) {
    tx.telemetry_increment("reference_barrier_accepted");
    tx.accept();
  } else {
    // Intentional failure-path atomicity improvement: every rejected attempt
    // restores coordinates, state, and geometry to their pre-attempt bytes.
    tx.telemetry_increment("reference_barrier_rejected");
    tx.restore(state, nullptr);
    result.succeeded = false;
  }
  return result;
}
}  // namespace

void arm_rezone_closure_cooldown(const int until_step) {
  if (until_step > g_rezone_closure_cooldown_until_step) {
    g_rezone_closure_cooldown_until_step = until_step;
    core::log_warning(
        "[ale] rezone closure cooldown armed until step=" +
        std::to_string(until_step) +
        " (a committed rezone's remap violated total-mass closure)");
  }
}

bool rezone_closure_cooldown_active(const int step) {
  return step <= g_rezone_closure_cooldown_until_step;
}

AleStepResult apply_multiblock_csr_ale_step(
    core::State& state,
    const core::Config& cfg,
    const parallel::PartitionInfo& part,
    const parallel::Reduction* reduction,
    const HydroEOSContext* eos_ctx,
    const double dt_hydro_used,
    const bool force_rezone,
    const char* force_reason) {
  (void)force_reason;
  AleStepResult out;
  TENRYU_ASSERT(
      state.corner_stride == 4 ||
          ((cfg.numerics.ale.rezone_solver == "m1_tmop" ||
            cfg.numerics.ale.euler_window.enabled) &&
           state.corner_stride == 8),
      "corner_stride must be 4, except stride 8 for belt M1 or "
      "Eulerian-window ALE");
  if (!mesh::mesh_topo_is_multiblock(cfg.mesh) || state.mesh.dim != 2 ||
      !cfg.numerics.ale.enabled) {
    return out;
  }
  if (ale_identity_mode_enabled(cfg)) {
    return out;
  }
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (n_cells <= 0 || n_nodes <= 0) {
    return out;
  }
  if (try_apply_axis_rezone(
          state, cfg, part, reduction, eos_ctx, dt_hydro_used, out)) {
    return out;
  }
  {
    const auto morph = tenryu::hydro::button_morph::apply_button_morph_if_due(
        state, cfg, state.t + dt_hydro_used, state.step);
    if (morph.attempted) {
      out.rezone_triggered = true;
      out.rezone_converged = morph.applied;
      out.applied = morph.applied;
      if (morph.applied) {
        state.ale_rezoned = true;
        state.ale_last_applied_step = state.step;
        state.holo_ale_invalidated = true;
      }
      record_estep_trace_boundary(state, cfg, part, reduction,
                                  "post_rezone_pre_remap", dt_hydro_used,
                                  state.t + dt_hydro_used);
      record_estep_trace_boundary(state, cfg, part, reduction, "post_remap",
                                  dt_hydro_used, state.t + dt_hydro_used);
      record_estep_trace_boundary(state, cfg, part, reduction, "step_end",
                                  dt_hydro_used, state.t + dt_hydro_used);
      return out;
    }
  }
  const bool ale_cadence_due =
      cfg.numerics.ale.every_n_steps <= 1 ||
      (state.step % cfg.numerics.ale.every_n_steps) == 0;
  const std::uint64_t reference_barrier_trigger_mask =
      tenryu::hydro::evaluate_reference_barrier_trigger(state, cfg);
  if (!force_rezone && !ale_cadence_due && reference_barrier_trigger_mask == 0) {
    record_estep_trace_boundary(state,
                                cfg,
                                part,
                                reduction,
                                "post_rezone_pre_remap",
                                dt_hydro_used,
                                state.t + dt_hydro_used);
    record_estep_trace_boundary(state,
                                cfg,
                                part,
                                reduction,
                                "post_remap",
                                dt_hydro_used,
                                state.t + dt_hydro_used);
    record_estep_trace_boundary(state,
                                cfg,
                                part,
                                reduction,
                                "step_end",
                                dt_hydro_used,
                                state.t + dt_hydro_used);
    return out;
  }

  if (reference_barrier_trigger_mask != 0) {
    double* d_target_r = nullptr;
    double* d_target_z = nullptr;
    const std::size_t node_bytes =
        static_cast<std::size_t>(n_nodes) * sizeof(double);
    d_target_r = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:mb_step:d_target_r", node_bytes));
    d_target_z = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:mb_step:d_target_z", node_bytes));
    tenryu::hydro::build_reference_target_mesh(state, cfg, d_target_r, d_target_z);
    auto ref_result = apply_reference_barrier_rezone_transaction(
        state, cfg, d_target_r, d_target_z);
    ref_result.trigger_reason_mask = reference_barrier_trigger_mask;

    ++state.reference_barrier_engaged_count;
    state.reference_barrier_last_lambda = ref_result.lambda_accepted;
    if (ref_result.succeeded) {
      ++state.reference_barrier_succeeded_count;
    }
    out.rezone_triggered = ref_result.engaged;
    out.rezone_converged = ref_result.succeeded;
    out.rezone_iterations = ref_result.linesearch_iters;
    out.rezone_residual = 1.0 - ref_result.lambda_accepted;
    out.quality_min = std::min({ref_result.final_quality.min_rz_volume_rel,
                                ref_result.final_quality.min_corner_j_rel,
                                ref_result.final_quality.min_gauss_j_rel});
    out.applied = ref_result.succeeded && ref_result.lambda_accepted > 0.0;
    out.energy_audit_ale_fallback =
        out.energy_audit_ale_fallback ||
        (ref_result.engaged && !ref_result.succeeded);
    out.mass_floor_delta += ref_result.mass_floor_delta;
    out.E_floor_injected += ref_result.E_floor_injected;
    out.E_redistribution_unresolved += ref_result.E_redistribution_unresolved;
    out.accepted_remap_count = state.ale_remaps_applied;
    if (out.applied) {
      state.ale_rezoned = true;
      state.ale_last_applied_step = state.step;
      state.holo_ale_invalidated = true;
    }
    record_estep_trace_boundary(state,
                                cfg,
                                part,
                                reduction,
                                "post_rezone_pre_remap",
                                dt_hydro_used,
                                state.t + dt_hydro_used);
    record_estep_trace_boundary(state,
                                cfg,
                                part,
                                reduction,
                                "post_remap",
                                dt_hydro_used,
                                state.t + dt_hydro_used);
    record_estep_trace_boundary(state,
                                cfg,
                                part,
                                reduction,
                                "step_end",
                                dt_hydro_used,
                                state.t + dt_hydro_used);
    return out;
  }

  if (cfg.numerics.ale.euler_window.enabled) {
    TENRYU_ASSERT(
        state.mesh.topo.multiblock.has_value(),
        "Eulerian-window ALE requires multiblock topology");
    TENRYU_ASSERT(
        state.x_r_reference.size() == static_cast<std::size_t>(n_nodes) &&
            state.x_z_reference.size() ==
                static_cast<std::size_t>(n_nodes),
        "Eulerian-window ALE requires reference node storage");
    TENRYU_ASSERT(
        state.cell_vol_initial.size() ==
            static_cast<std::size_t>(n_cells),
        "Eulerian-window ALE requires reference volume storage");

    const auto& topology = *state.mesh.topo.multiblock;
    TENRYU_ASSERT(
        topology.cell_node_csr_offsets.size() ==
                static_cast<std::size_t>(n_cells) + 1U &&
            topology.cell_id_stable.size() ==
                static_cast<std::size_t>(n_cells) &&
            topology.cell_orientation_sign.size() ==
                static_cast<std::size_t>(n_cells),
        "Eulerian-window ALE requires complete multiblock CSR metadata");
    TENRYU_ASSERT(
        state.mesh.cell_nverts.empty() ||
            state.mesh.cell_nverts.size() ==
                static_cast<std::size_t>(n_cells),
        "Eulerian-window ALE cell_nverts size mismatch");

    std::vector<double> lagrangian_r;
    std::vector<double> lagrangian_z;
    std::vector<double> euler_r;
    std::vector<double> euler_z;
    std::vector<double> old_vol;
    std::vector<double> persistent_reference_vol;
    state.x_r.copy_to_host(lagrangian_r);
    state.x_z.copy_to_host(lagrangian_z);
    state.x_r_reference.copy_to_host(euler_r);
    state.x_z_reference.copy_to_host(euler_z);
    state.vol.copy_to_host(old_vol);
    state.cell_vol_initial.copy_to_host(persistent_reference_vol);

    const std::vector<std::uint8_t> cell_nverts =
        state.mesh.cell_nverts.empty()
            ? std::vector<std::uint8_t>(
                  static_cast<std::size_t>(n_cells), 4U)
            : state.mesh.cell_nverts;
    std::vector<double> cell_centroid_r(
        static_cast<std::size_t>(n_cells), 0.0);
    std::vector<double> cell_centroid_z(
        static_cast<std::size_t>(n_cells), 0.0);
    for (int cell = 0; cell < n_cells; ++cell) {
      const auto cell_index = static_cast<std::size_t>(cell);
      const int begin =
          topology.cell_node_csr_offsets[cell_index];
      const int end =
          topology.cell_node_csr_offsets[cell_index + 1U];
      const int nverts =
          static_cast<int>(cell_nverts[cell_index]);
      TENRYU_ASSERT(
          nverts >= 3 && nverts <= state.corner_stride &&
              begin >= 0 && begin + nverts <= end &&
              static_cast<std::size_t>(end) <=
                  topology.cell_node_csr_indices.size(),
          "Eulerian-window ALE cell cycle is invalid");
      for (int local = 0; local < nverts; ++local) {
        const int node =
            topology.cell_node_csr_indices[
                static_cast<std::size_t>(begin + local)];
        TENRYU_ASSERT(
            node >= 0 && node < n_nodes,
            "Eulerian-window ALE cell cycle node is out of range");
        cell_centroid_r[cell_index] +=
            lagrangian_r[static_cast<std::size_t>(node)];
        cell_centroid_z[cell_index] +=
            lagrangian_z[static_cast<std::size_t>(node)];
      }
      cell_centroid_r[cell_index] /= static_cast<double>(nverts);
      cell_centroid_z[cell_index] /= static_cast<double>(nverts);
    }

    const auto& euler_window = cfg.numerics.ale.euler_window;
    EulerWindowSpec spec{
        euler_window.shape == "annulus"
            ? EulerWindowSpec::Shape::Annulus
            : EulerWindowSpec::Shape::Rectangle,
        euler_window.r0,
        euler_window.r1,
        euler_window.z0,
        euler_window.z1,
        euler_window.cr,
        euler_window.cz,
        euler_window.rad_in,
        euler_window.rad_out,
        euler_window.transition_width};
    std::vector<double> cell_weights(
        static_cast<std::size_t>(n_cells), 0.0);
    std::vector<double> node_weights(
        static_cast<std::size_t>(n_nodes), 0.0);
    euler_window_cell_weights(
        spec,
        cell_centroid_r.data(),
        cell_centroid_z.data(),
        n_cells,
        cell_weights.data());
    euler_window_node_weights(
        topology.cell_node_csr_offsets.data(),
        topology.cell_node_csr_indices.data(),
        cell_nverts.data(),
        n_cells,
        state.corner_stride,
        n_nodes,
        cell_weights.data(),
        node_weights.data());

    std::vector<double> target_r(static_cast<std::size_t>(n_nodes), 0.0);
    std::vector<double> target_z(static_cast<std::size_t>(n_nodes), 0.0);
    euler_window_blend_targets(
        node_weights.data(),
        n_nodes,
        lagrangian_r.data(),
        lagrangian_z.data(),
        euler_r.data(),
        euler_z.data(),
        target_r.data(),
        target_z.data());

    std::vector<double> delta_r(static_cast<std::size_t>(n_nodes), 0.0);
    std::vector<double> delta_z(static_cast<std::size_t>(n_nodes), 0.0);
    int moved_nodes = 0;
    for (int node = 0; node < n_nodes; ++node) {
      const auto node_index = static_cast<std::size_t>(node);
      delta_r[node_index] =
          target_r[node_index] - lagrangian_r[node_index];
      delta_z[node_index] =
          target_z[node_index] - lagrangian_z[node_index];
      if (delta_r[node_index] != 0.0 || delta_z[node_index] != 0.0) {
        ++moved_nodes;
      }
    }

    core::DeviceArray<double> d_delta_r(
        static_cast<std::size_t>(n_nodes));
    core::DeviceArray<double> d_delta_z(
        static_cast<std::size_t>(n_nodes));
    core::DeviceArray<int> d_cell_id_stable(
        topology.cell_id_stable.size());
    core::DeviceArray<int> d_cell_orientation_sign(
        topology.cell_orientation_sign.size());
    core::DeviceArray<std::uint8_t> d_cell_nverts(
        cell_nverts.size());
    d_delta_r.copy_from_host(delta_r);
    d_delta_z.copy_from_host(delta_z);
    d_cell_id_stable.copy_from_host(topology.cell_id_stable);
    d_cell_orientation_sign.copy_from_host(
        topology.cell_orientation_sign);
    d_cell_nverts.copy_from_host(cell_nverts);

    mesh::CandidateMeshAdmissibilityFloors floors;
    floors.volume_rel = cfg.numerics.ale.reference_volume_floor_rel;
    floors.corner_j_rel =
        cfg.numerics.ale.reference_corner_j_floor_rel;
    floors.gauss_j_rel = cfg.numerics.ale.reference_gauss_j_floor_rel;
    const mesh::CandidateMeshQuality candidate_quality =
        mesh::evaluate_candidate_mesh_quality_csr(
            state.x_r.data(),
            state.x_z.data(),
            d_delta_r.data(),
            d_delta_z.data(),
            1.0,
            n_cells,
            state.corner_stride,
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            d_cell_id_stable.data(),
            d_cell_orientation_sign.data(),
            floors,
            d_cell_nverts.data(),
            nullptr,
            nullptr,
            0,
            state.x_r_reference.data(),
            state.x_z_reference.data());
    bool accepted = candidate_quality.admissible();
    if (reduction != nullptr) {
      accepted =
          reduction->allreduce_min(
              static_cast<std::uint64_t>(accepted ? 1U : 0U)) != 0U;
    }

    out.rezone_triggered = accepted;
    out.rezone_converged = accepted;
    out.quality_min =
        std::min({candidate_quality.min_rz_volume_rel,
                  candidate_quality.min_corner_j_rel,
                  candidate_quality.min_gauss_j_rel});
    out.energy_audit_ale_fallback = !accepted;
    if (!accepted) {
      if (part.rank == 0) {
        core::log_warning(
            "[euler-window] step=" + std::to_string(state.step + 1) +
            " inadmissible target skipped");
      }
      record_estep_trace_boundary(state,
                                  cfg,
                                  part,
                                  reduction,
                                  "post_rezone_pre_remap",
                                  dt_hydro_used,
                                  state.t + dt_hydro_used);
      record_estep_trace_boundary(state,
                                  cfg,
                                  part,
                                  reduction,
                                  "post_remap",
                                  dt_hydro_used,
                                  state.t + dt_hydro_used);
      record_estep_trace_boundary(state,
                                  cfg,
                                  part,
                                  reduction,
                                  "step_end",
                                  dt_hydro_used,
                                  state.t + dt_hydro_used);
      return out;
    }

    state.x_r.copy_from_host(target_r);
    state.x_z.copy_from_host(target_z);
    state.x_r_reference.copy_from_host(target_r);
    state.x_z_reference.copy_from_host(target_z);
    state.mesh.recompute_geometry();
    state.vol = state.mesh.cell_vol;
    state.cell_vol_initial.copy_from_host(state.mesh.cell_vol);
    ale_velcoherence::sample(state, cfg, "s1_post_rezone");
    record_estep_trace_boundary(state,
                                cfg,
                                part,
                                reduction,
                                "post_rezone_pre_remap",
                                dt_hydro_used,
                                state.t + dt_hydro_used);

    state.x_r.copy_from_host(lagrangian_r);
    state.x_z.copy_from_host(lagrangian_z);
    state.vol.copy_from_host(old_vol);
    core::Config remap_cfg = cfg;
    remap_cfg.numerics.ale.conservative_remap_enabled = true;
    remap_cfg.numerics.ale.conservative_remap_target = "reference";
    const auto remap_result =
        ale_remap_2d_rz(
            state, remap_cfg, eos_ctx, dt_hydro_used, nullptr);
    record_estep_trace_boundary(state,
                                cfg,
                                part,
                                reduction,
                                "post_remap",
                                dt_hydro_used,
                                state.t + dt_hydro_used);
    out.applied = remap_result.applied;
    out.mass_floor_delta += remap_result.mass_floor_delta;
    out.E_floor_injected += remap_result.E_floor_injected;
    out.E_redistribution_unresolved +=
        remap_result.E_redistribution_unresolved;
    out.cap_energy_audit_D_K += remap_result.cap_energy_audit_D_K;
    out.eta_contact_step = remap_result.eta_contact_step;
    out.eta_contact_cumulative += remap_result.eta_contact_step;
    out.i1b_ale_ke_sensor = remap_result.i1b_ale_ke_sensor;
    out.accepted_remap_count = state.ale_remaps_applied;
    state.x_r_reference.copy_from_host(euler_r);
    state.x_z_reference.copy_from_host(euler_z);
    state.cell_vol_initial.copy_from_host(persistent_reference_vol);
    if (out.applied) {
      state.ale_rezoned = true;
      ++state.ale_rezone_invocations;
      state.ale_last_applied_step = state.step;
      state.holo_ale_invalidated = true;
      if (part.rank == 0) {
        const auto weight_bounds =
            std::minmax_element(
                node_weights.begin(), node_weights.end());
        std::ostringstream log;
        log << "[euler-window] step=" << (state.step + 1)
            << " applied w_max="
            << detail::format_scientific(*weight_bounds.second)
            << " w_min="
            << detail::format_scientific(*weight_bounds.first)
            << " moved_nodes=" << moved_nodes;
        core::log_info(log.str());
      }
    } else {
      state.x_r.copy_from_host(lagrangian_r);
      state.x_z.copy_from_host(lagrangian_z);
      state.vol.copy_from_host(old_vol);
      state.mesh.recompute_geometry();
      state.vol = state.mesh.cell_vol;
    }
    record_estep_trace_boundary(state,
                                cfg,
                                part,
                                reduction,
                                "step_end",
                                dt_hydro_used,
                                state.t + dt_hydro_used);
    return out;
  }

  if (cfg.numerics.ale.rezone_solver == "m1_tmop") {
    TENRYU_ASSERT(
        state.x_r_reference.size() == static_cast<std::size_t>(n_nodes) &&
            state.x_z_reference.size() ==
                static_cast<std::size_t>(n_nodes),
        "multiblock M1 ALE requires reference node storage");
    TENRYU_ASSERT(
        state.cell_vol_initial.size() ==
            static_cast<std::size_t>(n_cells),
        "multiblock M1 ALE requires reference volume storage");
    std::vector<double> old_r;
    std::vector<double> old_z;
    std::vector<double> old_vol;
    std::vector<double> persistent_reference_r;
    std::vector<double> persistent_reference_z;
    std::vector<double> persistent_reference_vol;
    state.x_r.copy_to_host(old_r);
    state.x_z.copy_to_host(old_z);
    state.vol.copy_to_host(old_vol);
    state.x_r_reference.copy_to_host(persistent_reference_r);
    state.x_z_reference.copy_to_host(persistent_reference_z);
    state.cell_vol_initial.copy_to_host(persistent_reference_vol);

    const RezoneResult m1_result =
        run_m1_tmop_rezone_transaction(
            state, cfg, part, reduction, force_rezone);
    out.rezone_triggered = m1_result.triggered;
    out.rezone_converged = m1_result.converged;
    out.rezone_iterations = m1_result.iterations;
    out.rezone_residual = m1_result.residual;
    out.quality_min = m1_result.min_quality;
    out.energy_audit_ale_fallback =
        m1_result.accepted_lambda == 0.0;
    if (!m1_result.triggered) {
      record_estep_trace_boundary(state,
                                  cfg,
                                  part,
                                  reduction,
                                  "post_rezone_pre_remap",
                                  dt_hydro_used,
                                  state.t + dt_hydro_used);
      record_estep_trace_boundary(state,
                                  cfg,
                                  part,
                                  reduction,
                                  "post_remap",
                                  dt_hydro_used,
                                  state.t + dt_hydro_used);
      record_estep_trace_boundary(state,
                                  cfg,
                                  part,
                                  reduction,
                                  "step_end",
                                  dt_hydro_used,
                                  state.t + dt_hydro_used);
      return out;
    }

    std::vector<double> target_r;
    std::vector<double> target_z;
    state.x_r.copy_to_host(target_r);
    state.x_z.copy_to_host(target_z);
    state.x_r_reference.copy_from_host(target_r);
    state.x_z_reference.copy_from_host(target_z);
    state.mesh.recompute_geometry();
    state.vol = state.mesh.cell_vol;
    state.cell_vol_initial.copy_from_host(state.mesh.cell_vol);
    ale_velcoherence::sample(state, cfg, "s1_post_rezone");
    record_estep_trace_boundary(state,
                                cfg,
                                part,
                                reduction,
                                "post_rezone_pre_remap",
                                dt_hydro_used,
                                state.t + dt_hydro_used);

    state.x_r.copy_from_host(old_r);
    state.x_z.copy_from_host(old_z);
    state.vol.copy_from_host(old_vol);
    core::Config remap_cfg = cfg;
    remap_cfg.numerics.ale.conservative_remap_enabled = true;
    remap_cfg.numerics.ale.conservative_remap_target = "reference";
    const auto remap_result =
        ale_remap_2d_rz(
            state, remap_cfg, eos_ctx, dt_hydro_used, nullptr);
    record_estep_trace_boundary(state,
                                cfg,
                                part,
                                reduction,
                                "post_remap",
                                dt_hydro_used,
                                state.t + dt_hydro_used);
    out.applied = remap_result.applied;
    out.mass_floor_delta += remap_result.mass_floor_delta;
    out.E_floor_injected += remap_result.E_floor_injected;
    out.E_redistribution_unresolved +=
        remap_result.E_redistribution_unresolved;
    out.cap_energy_audit_D_K += remap_result.cap_energy_audit_D_K;
    out.eta_contact_step = remap_result.eta_contact_step;
    out.eta_contact_cumulative += remap_result.eta_contact_step;
    out.i1b_ale_ke_sensor = remap_result.i1b_ale_ke_sensor;
    out.accepted_remap_count = state.ale_remaps_applied;
    state.x_r_reference.copy_from_host(persistent_reference_r);
    state.x_z_reference.copy_from_host(persistent_reference_z);
    state.cell_vol_initial.copy_from_host(persistent_reference_vol);
    if (out.applied) {
      state.ale_rezoned = true;
      ++state.ale_rezone_invocations;
      state.ale_last_applied_step = state.step;
      state.holo_ale_invalidated = true;
    } else {
      state.x_r.copy_from_host(old_r);
      state.x_z.copy_from_host(old_z);
      state.vol.copy_from_host(old_vol);
      state.mesh.recompute_geometry();
      state.vol = state.mesh.cell_vol;
    }
    record_estep_trace_boundary(state,
                                cfg,
                                part,
                                reduction,
                                "step_end",
                                dt_hydro_used,
                                state.t + dt_hydro_used);
    return out;
  }

  if (!(cfg.numerics.ale.relaxation > 0.0)) {
    return out;
  }

  ale_velcoherence::sample(state, cfg, "s0_post_hydro");

  const double omega_initial = std::min(cfg.numerics.ale.relaxation, 1.0);
  const std::size_t node_bytes = static_cast<std::size_t>(n_nodes) * sizeof(double);
  const std::size_t cell_bytes = static_cast<std::size_t>(n_cells) * sizeof(double);
  TENRYU_ASSERT(state.x_r_reference.size() == static_cast<std::size_t>(n_nodes) &&
                    state.x_z_reference.size() == static_cast<std::size_t>(n_nodes),
                "multiblock CSR ALE requires reference node storage");
  TENRYU_ASSERT(state.cell_vol_initial.size() == static_cast<std::size_t>(n_cells),
                "multiblock CSR ALE requires reference volume storage");

  double* d_xr_old = nullptr;
  double* d_xz_old = nullptr;
  double* d_vol_old = nullptr;
  double* d_ref_r_old = nullptr;
  double* d_ref_z_old = nullptr;
  double* d_cell_vol_initial_old = nullptr;
  d_xr_old = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:mb_step:d_xr_old", node_bytes));
  d_xz_old = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:mb_step:d_xz_old", node_bytes));
  d_vol_old = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:mb_step:d_vol_old", cell_bytes));
  d_ref_r_old = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:mb_step:d_ref_r_old", node_bytes));
  d_ref_z_old = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:mb_step:d_ref_z_old", node_bytes));
  d_cell_vol_initial_old = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:mb_step:d_cell_vol_initial_old", cell_bytes));
  const auto cleanup = [&]() {
  };
  const auto restore_persistent_reference = [&]() {
    CUDA_CHECK(cudaMemcpy(state.x_r_reference.data(),
                          d_ref_r_old,
                          node_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.x_z_reference.data(),
                          d_ref_z_old,
                          node_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.cell_vol_initial.data(),
                          d_cell_vol_initial_old,
                          cell_bytes,
                          cudaMemcpyDeviceToDevice));
  };
  CUDA_CHECK(cudaMemcpy(d_xr_old,
                        state.x_r.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_xz_old,
                        state.x_z.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_vol_old,
                        state.vol.data(),
                        cell_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_ref_r_old,
                        state.x_r_reference.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_ref_z_old,
                        state.x_z_reference.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_cell_vol_initial_old,
                        state.cell_vol_initial.data(),
                        cell_bytes,
                        cudaMemcpyDeviceToDevice));

  if (cfg.numerics.ale.multiblock_lagrangian_bulk_center_patch_reference_enabled &&
      cfg.numerics.ale.conservative_remap_target == "reference" &&
      !rezone_closure_cooldown_active(state.step)) {
    const MultiblockCenterPatchResult patch =
        build_multiblock_center_quality_patch(state, cfg);
    if (patch.applicable) {
      TENRYU_ASSERT(patch.node_rezone_active.size() ==
                        static_cast<std::size_t>(n_nodes),
                    "center patch active-node mask size mismatch");
      TENRYU_ASSERT(patch.cell_in_patch.size() ==
                        static_cast<std::size_t>(n_cells),
                    "center patch cell mask size mismatch");
      const auto& mb = *state.mesh.topo.multiblock;
      core::State& mutable_state = state;
      if (central_pseudo_core::configured(cfg)) {
        central_pseudo_core::ensure_built(mutable_state, cfg);
      }
      pole_angular_derefine::ensure_built(mutable_state, cfg);
      const bool csr_cons_audit = csr_cons_audit_env_enabled();
      const auto csr_cons_state_energy = [&]() {
        diagnostics::EnergyTotals energy = detail::reduce_energy_totals_global(
            diagnostics::compute_energy_totals_2d(state), reduction);
        return detail::total_energy(energy);
      };
      const double csr_cons_rejected_E_before =
          csr_cons_audit ? csr_cons_state_energy() : 0.0;
      const auto emit_csr_cons_rejected = [&](const char* reason) {
        if (!csr_cons_audit) {
          return;
        }
        const double E_after = csr_cons_state_energy();
        core::log_warning(
            std::string("[csr_cons_audit] rejected step=") +
            std::to_string(state.step) +
            " t=" + detail::format_scientific17(state.t + dt_hydro_used) +
            " dt=" + detail::format_scientific17(dt_hydro_used) +
            " reason=" + (reason != nullptr ? reason : "unknown") +
            " dE_rejected=" +
            detail::format_scientific17(E_after - csr_cons_rejected_E_before) +
            " dW_rejected=" + detail::format_scientific17(0.0));
      };
      const bool central_macro_active = central_pseudo_core::active(state);
      const auto& pc = state.central_pseudo_core;
      core::DeviceArray<std::uint8_t> d_combined_inactive_cell_mask;
      const std::uint8_t* d_inactive_cell_mask =
          pole_angular_derefine::combined_inactive_mask_device(
              mutable_state, d_combined_inactive_cell_mask);
      const int* d_macro_boundary_nodes =
          central_macro_active && !pc.d_boundary_nodes_ordered.empty()
              ? pc.d_boundary_nodes_ordered.data()
              : nullptr;
      const int n_macro_boundary_nodes =
          central_macro_active
              ? static_cast<int>(pc.boundary_nodes_ordered.size())
              : 0;
      std::vector<std::uint8_t> center_patch_node_rezone_active =
          patch.node_rezone_active;
      std::vector<std::uint8_t> center_patch_macro_excluded_node(
          static_cast<std::size_t>(n_nodes), 0U);
      if (central_macro_active &&
          pc.inactive_member_mask.size() == static_cast<std::size_t>(n_cells)) {
        TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                          static_cast<std::size_t>(n_cells + 1),
                      "center patch macro exclusion requires CSR offsets");
        TENRYU_ASSERT(mb.cell_node_csr_indices.size() ==
                          static_cast<std::size_t>(mesh::kMeshTopoCellStorageSlots *
                                                   n_cells),
                      "center patch macro exclusion requires CSR indices");
        for (int c = 0; c < n_cells; ++c) {
          if (pc.inactive_member_mask[static_cast<std::size_t>(c)] == 0U) {
            continue;
          }
          const int off =
              mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
          const int nverts =
              state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)
                  ? mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c)
                  : mesh::kMeshTopoCellStorageSlots;
          for (int k = 0; k < nverts; ++k) {
            const int node =
                mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
            if (node >= 0 && node < n_nodes) {
              center_patch_macro_excluded_node[static_cast<std::size_t>(node)] =
                  1U;
            }
          }
        }
        if (pc.boundary_node_mask.size() == static_cast<std::size_t>(n_nodes)) {
          for (int n = 0; n < n_nodes; ++n) {
            if (pc.boundary_node_mask[static_cast<std::size_t>(n)] != 0U) {
              center_patch_macro_excluded_node[static_cast<std::size_t>(n)] =
                  1U;
            }
          }
        }
        for (int n = 0; n < n_nodes; ++n) {
          if (center_patch_macro_excluded_node[static_cast<std::size_t>(n)] !=
              0U) {
            center_patch_node_rezone_active[static_cast<std::size_t>(n)] = 0U;
          }
        }
      }
      std::vector<double> r_lag;
      std::vector<double> z_lag;
      state.x_r.copy_to_host(r_lag);
      state.x_z.copy_to_host(z_lag);

      core::DeviceArray<std::uint8_t> d_rezone_active_node_mask(
          center_patch_node_rezone_active.size());
      d_rezone_active_node_mask.copy_from_host(center_patch_node_rezone_active);
      core::DeviceArray<std::uint8_t> d_macro_excluded_node_mask(
          center_patch_macro_excluded_node.size());
      d_macro_excluded_node_mask.copy_from_host(center_patch_macro_excluded_node);
      const MultiblockWinslowSmoothStats stats =
          cfg.numerics.ale.multiblock_cross_seam_rezone_enabled
              ? multiblock_winslow_smooth_with_seams_stats(
                    state,
                    cfg,
                    cfg.numerics.ale.max_iterations,
                    omega_initial,
                    d_rezone_active_node_mask.data())
              : multiblock_winslow_smooth_with_stats(
                    state,
                    cfg,
                    cfg.numerics.ale.max_iterations,
                    omega_initial,
                    d_rezone_active_node_mask.data());
      out.rezone_triggered = stats.accepted_iterations > 0;
      out.rezone_converged = true;
      out.rezone_iterations = stats.accepted_iterations;
      out.rezone_residual = stats.max_displacement;
      out.applied = false;
      if (out.rezone_triggered) {
        out.quality_min =
            std::min({stats.final_quality.min_rz_volume_rel,
                      stats.final_quality.min_corner_j_rel,
                      stats.final_quality.min_gauss_j_rel});
      }
      if (!out.rezone_triggered || !(stats.max_displacement > 0.0)) {
        CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                              d_xr_old,
                              node_bytes,
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                              d_xz_old,
                              node_bytes,
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state.vol.data(),
                              d_vol_old,
                              cell_bytes,
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        ale_velcoherence::sample(state, cfg, "s1_post_rezone");
        ale_velcoherence::sample(state, cfg, "s2_post_remap");
        ale_velcoherence::sample(state, cfg, "s3_post_velproj");
        record_estep_trace_boundary(state,
                                    cfg,
                                    part,
                                    reduction,
                                    "post_rezone_pre_remap",
                                    dt_hydro_used,
                                    state.t + dt_hydro_used);
        record_estep_trace_boundary(state,
                                    cfg,
                                    part,
                                    reduction,
                                    "post_remap",
                                    dt_hydro_used,
                                    state.t + dt_hydro_used);
        record_estep_trace_boundary(state,
                                    cfg,
                                    part,
                                    reduction,
                                    "step_end",
                                    dt_hydro_used,
                                    state.t + dt_hydro_used);
        emit_csr_cons_rejected("center_patch_no_rezone");
        cleanup();
        return out;
      }

      std::vector<double> r_smooth;
      std::vector<double> z_smooth;
      state.x_r.copy_to_host(r_smooth);
      state.x_z.copy_to_host(z_smooth);
      const std::vector<double> unit_volume_ref(static_cast<std::size_t>(n_cells),
                                                1.0);
      double seed_min_corner_j = std::numeric_limits<double>::infinity();
      for (int c = 0; c < n_cells; ++c) {
        const auto idx = static_cast<std::size_t>(c);
        if (patch.cell_in_patch[idx] == 0U) {
          continue;
        }
        const multiblock_center_patch_detail::CellQuality q =
            multiblock_center_patch_detail::evaluate_cell_quality(
                state, r_smooth, z_smooth, unit_volume_ref, c);
        seed_min_corner_j = std::min(seed_min_corner_j, q.corner_j_ratio);
      }
      if (!std::isfinite(seed_min_corner_j)) {
        seed_min_corner_j = -std::numeric_limits<double>::infinity();
      }
      ale_motion::AleMotionTriggerMetrics barrier_trigger_metrics;
      barrier_trigger_metrics.min_corner_j_rel = seed_min_corner_j;
      ale_motion::AleMotionTriggerParams barrier_trigger_params;
      barrier_trigger_params.enabled = true;
      barrier_trigger_params.min_corner_j_on =
          cfg.numerics.ale.multiblock_center_patch_cornerj_off;
      barrier_trigger_params.min_corner_j_off =
          std::min(1.0, std::max(1.5 * barrier_trigger_params.min_corner_j_on,
                                 barrier_trigger_params.min_corner_j_on +
                                     1.0e-12));
      barrier_trigger_params.min_corner_j_critical =
          cfg.numerics.ale.multiblock_center_patch_cornerj_on;
      barrier_trigger_params.min_gauss_j_on = 0.0;
      barrier_trigger_params.min_gauss_j_off = 0.0;
      barrier_trigger_params.min_cell_volume_ratio_on = 0.0;
      barrier_trigger_params.min_cell_volume_ratio_off = 0.0;
      barrier_trigger_params.max_aspect_ratio_on = 0.0;
      barrier_trigger_params.max_aspect_ratio_off = 0.0;
      barrier_trigger_params.dt_ratio_on = 0.0;
      barrier_trigger_params.dt_ratio_off = 0.0;
      const ale_motion::AleMotionTriggerDecision barrier_trigger =
          ale_motion::evaluate_ale_motion_trigger(
              barrier_trigger_metrics,
              barrier_trigger_params,
              ale_motion::AleMotionTriggerState{},
              0);
      const bool barrier_applied = barrier_trigger.should_rezone;
      center_patch_barrier::CenterPatchBarrierResult barrier_result;

      TENRYU_ASSERT(mb.cell_orientation_sign.size() ==
                        static_cast<std::size_t>(n_cells),
                    "center patch requires cell orientation signs");
      core::DeviceArray<int> d_cell_orientation_sign(
          mb.cell_orientation_sign.size());
      d_cell_orientation_sign.copy_from_host(mb.cell_orientation_sign);
      core::DeviceArray<std::uint8_t> d_cell_nverts;
      const std::uint8_t* d_cell_nverts_ptr = nullptr;
      if (tracking_reference_detail::multiblock_has_tri_cell_nverts(state)) {
        d_cell_nverts.reset(state.mesh.cell_nverts.size());
        d_cell_nverts.copy_from_host(state.mesh.cell_nverts);
        d_cell_nverts_ptr = d_cell_nverts.data();
      }

      if (barrier_applied) {
        TENRYU_ASSERT(patch.node_patch_boundary.size() ==
                          static_cast<std::size_t>(n_nodes),
                      "center patch boundary-node mask size mismatch");
        TENRYU_ASSERT(state.mesh.topo.node_flags.size() ==
                          static_cast<std::size_t>(n_nodes),
                      "center patch barrier requires node flags");
        TENRYU_ASSERT(state.mesh.multiblock_reverse_csr_node_offsets.size() ==
                          static_cast<std::size_t>(n_nodes + 1),
                      "center patch barrier requires reverse CSR node offsets");
        TENRYU_ASSERT(state.mesh.multiblock_reverse_csr_node_cells.size() ==
                          state.mesh.multiblock_reverse_csr_node_corners.size(),
                      "center patch barrier reverse CSR cell/corner mismatch");
        core::DeviceArray<std::uint8_t> d_patch_boundary_node_mask(
            patch.node_patch_boundary.size());
        d_patch_boundary_node_mask.copy_from_host(patch.node_patch_boundary);
        core::DeviceArray<std::uint8_t> d_node_flags(
            state.mesh.topo.node_flags.size());
        d_node_flags.copy_from_host(state.mesh.topo.node_flags);
        core::DeviceArray<double> d_barrier_r(static_cast<std::size_t>(n_nodes));
        core::DeviceArray<double> d_barrier_z(static_cast<std::size_t>(n_nodes));
        center_patch_barrier::CenterPatchBarrierParams barrier_params;
        barrier_params.volume_floor_rel =
            cfg.numerics.ale.multiblock_center_patch_vol_on;
        barrier_params.jacobian_floor_rel =
            cfg.numerics.ale.multiblock_center_patch_cornerj_on;
        barrier_params.constrain_cap_apex = mb.has_trifan_cap;
        if (mb.has_trifan_cap) {
          barrier_params.cap_apex_node_id =
              mesh::mesh_topo_cap_apex_node_id(mb);
        }
        barrier_result = center_patch_barrier::optimize_center_patch_phi_barrier(
            d_barrier_r.data(),
            d_barrier_z.data(),
            state.x_r.data(),
            state.x_z.data(),
            d_xr_old,
            d_xz_old,
            d_ref_r_old,
            d_ref_z_old,
            state.mesh.multiblock_reverse_csr_node_offsets.data(),
            state.mesh.multiblock_reverse_csr_node_cells.data(),
            state.mesh.multiblock_reverse_csr_node_corners.data(),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            d_cell_orientation_sign.data(),
            d_cell_nverts_ptr,
            d_node_flags.data(),
            d_rezone_active_node_mask.data(),
            d_patch_boundary_node_mask.data(),
            n_nodes,
            barrier_params,
            d_macro_excluded_node_mask.data());
        if (barrier_result.invalid_nodes > 0) {
          barrier_result =
              center_patch_barrier::optimize_center_patch_phi_barrier(
                  d_barrier_r.data(),
                  d_barrier_z.data(),
                  d_ref_r_old,
                  d_ref_z_old,
                  d_xr_old,
                  d_xz_old,
                  d_ref_r_old,
                  d_ref_z_old,
                  state.mesh.multiblock_reverse_csr_node_offsets.data(),
                  state.mesh.multiblock_reverse_csr_node_cells.data(),
                  state.mesh.multiblock_reverse_csr_node_corners.data(),
                  state.mesh.multiblock_cell_node_csr_offsets.data(),
                  state.mesh.multiblock_cell_node_csr_indices.data(),
                  d_cell_orientation_sign.data(),
                  d_cell_nverts_ptr,
                  d_node_flags.data(),
                  d_rezone_active_node_mask.data(),
                  d_patch_boundary_node_mask.data(),
                  n_nodes,
                  barrier_params,
                  d_macro_excluded_node_mask.data());
        }
        if (barrier_result.invalid_nodes > 0 ||
            barrier_result.reverted_nodes > 0) {
          core::log_warning(
              std::string("[ale] center-patch barrier optimizer: ") +
              std::to_string(barrier_result.invalid_nodes) +
              " invalid node-sweep events; " +
              std::to_string(barrier_result.reverted_nodes) +
              " still-inadmissible nodes REVERTED to Lagrangian (identity "
              "target); " +
              std::to_string(barrier_result.invalid_nodes_final) +
              " inadmissible even at Lagrangian (mesh already degenerate "
              "there)");
        }
        CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                              d_barrier_r.data(),
                              node_bytes,
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                              d_barrier_z.data(),
                              node_bytes,
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        state.x_r.copy_to_host(r_smooth);
        state.x_z.copy_to_host(z_smooth);
      }
      std::vector<double> delta_r(static_cast<std::size_t>(n_nodes), 0.0);
      std::vector<double> delta_z(static_cast<std::size_t>(n_nodes), 0.0);
      double max_active_delta = 0.0;
      double max_frozen_delta = 0.0;
      for (int n = 0; n < n_nodes; ++n) {
        const auto idx = static_cast<std::size_t>(n);
        delta_r[idx] = r_smooth[idx] - r_lag[idx];
        delta_z[idx] = z_smooth[idx] - z_lag[idx];
        const double d = std::hypot(delta_r[idx], delta_z[idx]);
        TENRYU_ASSERT(std::isfinite(d),
                      "center patch candidate displacement is non-finite");
        if (center_patch_node_rezone_active[idx] != 0U) {
          max_active_delta = std::max(max_active_delta, d);
        } else {
          max_frozen_delta = std::max(max_frozen_delta, d);
        }
      }
      TENRYU_ASSERT(max_frozen_delta <= 0.0,
                    "center patch candidate moved a frozen node");
      if (!(max_active_delta > 0.0)) {
        // A zero-motion candidate is legitimate since the barrier revert
        // post-pass (every proposed update negligible or reverted). Fall
        // through: the identity rezone+remap is exact (zero swept volumes)
        // and keeps the fire's downstream reference handling identical to
        // the historical path.
        core::log_warning(
            "[ale] center-patch candidate has no active-node motion "
            "(updates negligible or reverted); committing identity fire "
            "step=" +
            std::to_string(state.step + 1));
      }

      core::DeviceArray<double> d_delta_r(static_cast<std::size_t>(n_nodes));
      core::DeviceArray<double> d_delta_z(static_cast<std::size_t>(n_nodes));
      d_delta_r.copy_from_host(delta_r);
      d_delta_z.copy_from_host(delta_z);
      core::DeviceArray<int> d_cell_id_stable(mb.cell_id_stable.size());
      d_cell_id_stable.copy_from_host(mb.cell_id_stable);

      mesh::CandidateMeshAdmissibilityFloors floors;
      floors.volume_rel = cfg.numerics.ale.reference_volume_floor_rel;
      floors.corner_j_rel = cfg.numerics.ale.reference_corner_j_floor_rel;
      floors.gauss_j_rel = cfg.numerics.ale.reference_gauss_j_floor_rel;
      const mesh::LineSearchResult ls =
          mesh::linesearch_largest_admissible_sigma_csr(
              d_xr_old,
              d_xz_old,
              d_delta_r.data(),
              d_delta_z.data(),
              1.0,
              0.0,
              cfg.numerics.ale.reference_linesearch_max_iters,
              n_cells,
              state.mesh.corner_stride,
              state.mesh.multiblock_cell_node_csr_offsets.data(),
              state.mesh.multiblock_cell_node_csr_indices.data(),
              d_cell_id_stable.data(),
              d_cell_orientation_sign.data(),
              floors,
              d_cell_nverts_ptr,
              d_inactive_cell_mask,
              d_macro_boundary_nodes,
              n_macro_boundary_nodes,
              state.x_r_reference.size() == state.x_r.size() &&
                      state.x_z_reference.size() == state.x_z.size()
                  ? state.x_r_reference.data()
                  : nullptr,
              state.x_r_reference.size() == state.x_r.size() &&
                      state.x_z_reference.size() == state.x_z.size()
                  ? state.x_z_reference.data()
                  : nullptr);
      double sigma_accepted = ls.sigma_accepted;
      out.center_patch_sigma_accepted = sigma_accepted;
      mesh::CandidateMeshQuality final_quality = ls.quality;
      if (sigma_accepted == 0.0 && !final_quality.admissible()) {
        final_quality = mesh::evaluate_candidate_mesh_quality_csr(
            d_xr_old,
            d_xz_old,
            d_delta_r.data(),
            d_delta_z.data(),
            0.0,
            n_cells,
            state.mesh.corner_stride,
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            d_cell_id_stable.data(),
            d_cell_orientation_sign.data(),
            floors,
            d_cell_nverts_ptr,
            d_inactive_cell_mask,
            d_macro_boundary_nodes,
            n_macro_boundary_nodes,
            state.x_r_reference.size() == state.x_r.size() &&
                    state.x_z_reference.size() == state.x_z.size()
                ? state.x_r_reference.data()
                : nullptr,
            state.x_r_reference.size() == state.x_r.size() &&
                    state.x_z_reference.size() == state.x_z.size()
                ? state.x_z_reference.data()
                : nullptr);
        const int first_bad_cell = final_quality.first_bad_cell;
        const char* is_pseudo_core_member = "unknown";
        const char* is_pseudo_core_representative = "unknown";
        const char* is_pseudo_core_passive = "unknown";
        if (central_pseudo_core::configured(cfg) &&
            !state.central_pseudo_core.member_mask.empty() &&
            first_bad_cell >= 0 && first_bad_cell < n_cells &&
            static_cast<std::size_t>(first_bad_cell) <
                state.central_pseudo_core.member_mask.size()) {
          const auto first_bad_idx = static_cast<std::size_t>(first_bad_cell);
          const bool pseudo_core_member =
              state.central_pseudo_core.member_mask[first_bad_idx] != 0U;
          const bool pseudo_core_representative =
              first_bad_cell == state.central_pseudo_core.representative_cell;
          bool pseudo_core_passive = false;
          if (first_bad_idx < state.central_pseudo_core.passive_mask.size()) {
            pseudo_core_passive =
                state.central_pseudo_core.passive_mask[first_bad_idx] != 0U;
          }
          is_pseudo_core_member = pseudo_core_member ? "1" : "0";
          is_pseudo_core_representative =
              pseudo_core_representative ? "1" : "0";
          is_pseudo_core_passive = pseudo_core_passive ? "1" : "0";
        }
        std::ostringstream death_diag;
        death_diag << "[i1b_centerpatch_death] step=" << state.step
                   << " time=" << state.t
                   << " first_bad_cell=" << final_quality.first_bad_cell
                   << " first_bad_stable_cell="
                   << final_quality.first_bad_stable_cell
                   << " first_bad_corner=" << final_quality.first_bad_corner
                   << " min_rz_volume_rel="
                   << final_quality.min_rz_volume_rel
                   << " min_corner_j_rel=" << final_quality.min_corner_j_rel
                   << " min_gauss_j_rel=" << final_quality.min_gauss_j_rel
                   << " kind=" << static_cast<int>(final_quality.kind)
                   << " kind_label="
                   << mesh::mesh_geometry_failure_kind_name(
                          final_quality.kind)
                   << " sigma_accepted=" << sigma_accepted
                   << " is_pseudo_core_member=" << is_pseudo_core_member
                   << " is_pseudo_core_representative="
                   << is_pseudo_core_representative
                   << " is_pseudo_core_passive="
                   << is_pseudo_core_passive;
        core::log_warning(death_diag.str());
        if (!final_quality.admissible() &&
            central_pseudo_core::request_ring_absorption(
                state, cfg, final_quality.first_bad_cell)) {
          core::log_warning(
              "[i1b_centerpatch_absorb_retry] center-patch inadmissible at "
              "first_bad_cell=" +
              std::to_string(final_quality.first_bad_cell) +
              "; ring-absorption requested, aborting ALE step for driver "
              "full-step retry");
          out.status = AleStatus::CenterPatchInadmissibleAbsorbRetry;
          emit_csr_cons_rejected("center_patch_absorb_retry");
          return out;
        }
        if (!final_quality.admissible()) {
          // No legal absorption repair remains (e.g. the failing cell is in
          // the material shell, which absorption must never eat). The
          // linesearch already accepted sigma=0, so the reference built
          // below is the identity (x_L): reject the patch for this step
          // instead of dying -- an inadmissible rezone is rejected, never
          // fatal. The hydro-side gates (path admissibility, dt floors,
          // absorption triggers) respond on the evolved state.
          core::log_warning(
              "[i1b_centerpatch_skip] center-patch quality floors violated "
              "at first_bad_cell=" +
              std::to_string(final_quality.first_bad_cell) +
              " with sigma=0 and no legal absorption repair; proceeding "
              "with the identity reference this step");
          out.energy_audit_identity_skip = true;
        }
      }

      std::vector<double> r_ref = r_lag;
      std::vector<double> z_ref = z_lag;
      for (int n = 0; n < n_nodes; ++n) {
        const auto idx = static_cast<std::size_t>(n);
        r_ref[idx] = r_lag[idx] + sigma_accepted * delta_r[idx];
        z_ref[idx] = z_lag[idx] + sigma_accepted * delta_z[idx];
      }
      const std::vector<double> vol_ref =
          tracking_reference_detail::compute_multiblock_reference_volumes(
              state, cfg, r_ref, z_ref);
      double patch_min_corner_j =
          std::numeric_limits<double>::infinity();
      for (int c = 0; c < n_cells; ++c) {
        const auto idx = static_cast<std::size_t>(c);
        if (patch.cell_in_patch[idx] == 0U) {
          continue;
        }
        const multiblock_center_patch_detail::CellQuality q =
            multiblock_center_patch_detail::evaluate_cell_quality(
                state, r_ref, z_ref, vol_ref, c);
        patch_min_corner_j =
            std::min(patch_min_corner_j, q.corner_j_ratio);
      }
      if (!std::isfinite(patch_min_corner_j)) {
        patch_min_corner_j = 1.0;
      }

      state.x_r_reference.copy_from_host(r_ref);
      state.x_z_reference.copy_from_host(z_ref);
      state.cell_vol_initial.copy_from_host(vol_ref);
      CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                            state.x_r_reference.data(),
                            node_bytes,
                            cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                            state.x_z_reference.data(),
                            node_bytes,
                            cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaDeviceSynchronize());
      state.mesh.recompute_geometry();
      state.vol = state.mesh.cell_vol;
      ale_velcoherence::sample(state, cfg, "s1_post_rezone");
      record_estep_trace_boundary(state,
                                  cfg,
                                  part,
                                  reduction,
                                  "post_rezone_pre_remap",
                                  dt_hydro_used,
                                  state.t + dt_hydro_used);

      CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                            d_xr_old,
                            node_bytes,
                            cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                            d_xz_old,
                            node_bytes,
                            cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemcpy(state.vol.data(),
                            d_vol_old,
                            cell_bytes,
                            cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaDeviceSynchronize());

      core::Config remap_cfg = cfg;
      remap_cfg.numerics.ale.conservative_remap_enabled = true;
      remap_cfg.numerics.ale.conservative_remap_target = "reference";
      remap_cfg.numerics.ale.swept_volume_sign_fixed = true;
      if (csr_cons_audit) {
        CsrConsAuditContext context;
        context.enabled = true;
        context.reduction = reduction;
        context.step = state.step;
        context.t = state.t + dt_hydro_used;
        context.dt = dt_hydro_used;
        context.remap_id = state.ale_remaps_applied + 1;
        context.trigger_cell = final_quality.first_bad_cell;
        context.min_path_margin =
            std::min({final_quality.min_rz_volume_rel,
                      final_quality.min_corner_j_rel,
                      final_quality.min_gauss_j_rel});
        set_csr_cons_audit_context(context);
      }
      const auto remap_result =
          ale_remap_2d_rz(state, remap_cfg, eos_ctx, dt_hydro_used, nullptr);
      if (csr_cons_audit) {
        clear_csr_cons_audit_context();
      }
      record_estep_trace_boundary(state,
                                  cfg,
                                  part,
                                  reduction,
                                  "post_remap",
                                  dt_hydro_used,
                                  state.t + dt_hydro_used);
      out.applied = remap_result.applied;
      out.mass_floor_delta += remap_result.mass_floor_delta;
      out.E_floor_injected += remap_result.E_floor_injected;
      out.E_redistribution_unresolved +=
          remap_result.E_redistribution_unresolved;
      out.cap_energy_audit_D_K += remap_result.cap_energy_audit_D_K;
      out.eta_contact_step = remap_result.eta_contact_step;
      out.eta_contact_cumulative += remap_result.eta_contact_step;
      out.i1b_ale_ke_sensor = remap_result.i1b_ale_ke_sensor;
      out.accepted_remap_count = state.ale_remaps_applied;
      out.quality_min =
          std::min({final_quality.min_rz_volume_rel,
                    final_quality.min_corner_j_rel,
                    final_quality.min_gauss_j_rel});
      restore_persistent_reference();
      if (out.applied) {
        state.ale_rezoned = true;
        ++state.ale_rezone_invocations;
        state.ale_last_applied_step = state.step;
        state.holo_ale_invalidated = true;
        core::log_info(
            std::string("[ale] multiblock CSR center-patch Winslow "
                        "rezone+remap applied: iterations=") +
            std::to_string(stats.accepted_iterations) +
            ", max_displacement=" +
            detail::format_scientific(stats.max_displacement) +
            ", barrier_applied=" +
            std::string(barrier_applied ? "true" : "false") +
            ", barrier_trigger_min_corner_j=" +
            detail::format_scientific(seed_min_corner_j) +
            ", barrier_trigger_reason_mask=" +
            std::to_string(barrier_trigger.reason_mask) +
            ", barrier_converged=" +
            std::string(barrier_result.converged ? "true" : "false") +
            ", barrier_sweeps=" + std::to_string(barrier_result.sweeps) +
            ", barrier_invalid_nodes=" +
            std::to_string(barrier_result.invalid_nodes) +
            ", barrier_max_step=" +
            detail::format_scientific(barrier_result.max_step) +
            ", sigma_accepted=" +
            detail::format_scientific(sigma_accepted) +
            ", patch_min_corner_j=" +
            detail::format_scientific(patch_min_corner_j) +
            ", max_active_delta=" +
            detail::format_scientific(max_active_delta) +
            ", max_frozen_delta=" +
            detail::format_scientific(max_frozen_delta) +
            ", active_nodes=" + std::to_string(patch.n_active_nodes) +
            ", boundary_nodes=" + std::to_string(patch.n_boundary_nodes) +
            ", patch_cells=" + std::to_string(patch.n_patch_cells));
      } else {
        CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                              d_xr_old,
                              node_bytes,
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                              d_xz_old,
                              node_bytes,
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state.vol.data(),
                              d_vol_old,
                              cell_bytes,
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        state.mesh.recompute_geometry();
        state.vol = state.mesh.cell_vol;
        emit_csr_cons_rejected("center_patch_remap_not_applied");
      }
      record_estep_trace_boundary(state,
                                  cfg,
                                  part,
                                  reduction,
                                  "step_end",
                                  dt_hydro_used,
                                  state.t + dt_hydro_used);
      cleanup();
      return out;
    }
  }

  const MultiblockWinslowSmoothStats stats =
      rezone_closure_cooldown_active(state.step)
          ? MultiblockWinslowSmoothStats{}
          : (cfg.numerics.ale.multiblock_cross_seam_rezone_enabled
                 ? multiblock_winslow_smooth_with_seams_stats(
                       state, cfg, cfg.numerics.ale.max_iterations,
                       omega_initial)
                 : multiblock_winslow_smooth_with_stats(
                       state, cfg, cfg.numerics.ale.max_iterations,
                       omega_initial));
  out.rezone_triggered = stats.accepted_iterations > 0;
  out.rezone_converged = true;
  out.rezone_iterations = stats.accepted_iterations;
  out.rezone_residual = stats.max_displacement;
  out.applied = false;
  if (out.rezone_triggered) {
    out.quality_min = std::min({stats.final_quality.min_rz_volume_rel,
                                stats.final_quality.min_corner_j_rel,
                                stats.final_quality.min_gauss_j_rel});
  }
  if (!out.rezone_triggered || !(stats.max_displacement > 0.0)) {
    ale_velcoherence::sample(state, cfg, "s1_post_rezone");
    ale_velcoherence::sample(state, cfg, "s2_post_remap");
    ale_velcoherence::sample(state, cfg, "s3_post_velproj");
    record_estep_trace_boundary(state,
                                cfg,
                                part,
                                reduction,
                                "post_rezone_pre_remap",
                                dt_hydro_used,
                                state.t + dt_hydro_used);
    record_estep_trace_boundary(state,
                                cfg,
                                part,
                                reduction,
                                "post_remap",
                                dt_hydro_used,
                                state.t + dt_hydro_used);
    record_estep_trace_boundary(state,
                                cfg,
                                part,
                                reduction,
                                "step_end",
                                dt_hydro_used,
                                state.t + dt_hydro_used);
    cleanup();
    return out;
  }

  CUDA_CHECK(cudaMemcpy(state.x_r_reference.data(),
                        state.x_r.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(state.x_z_reference.data(),
                        state.x_z.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice));
  core::DeviceArray<std::uint8_t> core_freeze_frozen_nodes;
  const auto core_freeze_diag =
      core_freeze::restore_target_if_enabled(state,
                                             cfg,
                                             state.x_r_reference.data(),
                                             state.x_z_reference.data(),
                                             d_xr_old,
                                             d_xz_old,
                                             false,
                                             cfg.numerics.ale
                                                     .multiblock_cross_seam_rezone_enabled
                                                 ? "multiblock_cross_seam_winslow"
                                                 : "multiblock_per_block_winslow",
                                             cfg.numerics.ale
                                                     .core_freeze_skip_velocity_projection
                                                 ? &core_freeze_frozen_nodes
                                                 : nullptr);
  if (core_freeze_diag.enabled && core_freeze_diag.frozen_node_count > 0) {
    CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                          state.x_r_reference.data(),
                          node_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                          state.x_z_reference.data(),
                          node_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    state.mesh.recompute_geometry();
    state.cell_vol_initial.copy_from_host(state.mesh.cell_vol);
  } else {
    CUDA_CHECK(cudaMemcpy(state.cell_vol_initial.data(),
                          state.vol.data(),
                          cell_bytes,
                          cudaMemcpyDeviceToDevice));
  }
  ale_velcoherence::sample(state, cfg, "s1_post_rezone");
  record_estep_trace_boundary(state,
                              cfg,
                              part,
                              reduction,
                              "post_rezone_pre_remap",
                              dt_hydro_used,
                              state.t + dt_hydro_used);
  CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                        d_xr_old,
                        node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                        d_xz_old,
                        node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(state.vol.data(),
                        d_vol_old,
                        cell_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());

  core::Config remap_cfg = cfg;
  remap_cfg.numerics.ale.conservative_remap_enabled = true;
  remap_cfg.numerics.ale.conservative_remap_target = "reference";
  const std::uint8_t* core_freeze_velocity_mask =
      core_freeze_frozen_nodes.size() == static_cast<std::size_t>(n_nodes)
          ? core_freeze_frozen_nodes.data()
          : nullptr;
  const auto remap_result =
      ale_remap_2d_rz(
          state, remap_cfg, eos_ctx, dt_hydro_used, core_freeze_velocity_mask);
  record_estep_trace_boundary(state,
                              cfg,
                              part,
                              reduction,
                              "post_remap",
                              dt_hydro_used,
                              state.t + dt_hydro_used);
  out.applied = remap_result.applied;
  out.mass_floor_delta += remap_result.mass_floor_delta;
  out.E_floor_injected += remap_result.E_floor_injected;
  out.E_redistribution_unresolved += remap_result.E_redistribution_unresolved;
  out.cap_energy_audit_D_K += remap_result.cap_energy_audit_D_K;
  out.eta_contact_step = remap_result.eta_contact_step;
  out.eta_contact_cumulative += remap_result.eta_contact_step;
  out.i1b_ale_ke_sensor = remap_result.i1b_ale_ke_sensor;
  out.accepted_remap_count = state.ale_remaps_applied;
  restore_persistent_reference();
  if (out.applied) {
    state.ale_rezoned = true;
    ++state.ale_rezone_invocations;
    state.ale_last_applied_step = state.step;
    state.holo_ale_invalidated = true;
    core::log_info(std::string("[ale] multiblock CSR ") +
                   (cfg.numerics.ale.multiblock_cross_seam_rezone_enabled
                        ? "cross-seam"
                        : "per-block") +
                   " Winslow rezone+remap applied: "
                   "iterations=" +
                   std::to_string(stats.accepted_iterations) +
                   ", max_displacement=" +
                   detail::format_scientific(stats.max_displacement) +
                   ", omega_final=" + std::to_string(stats.omega_final));
  } else {
    CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                          d_xr_old,
                          node_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                          d_xz_old,
                          node_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.vol.data(),
                          d_vol_old,
                          cell_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    state.mesh.recompute_geometry();
    state.vol = state.mesh.cell_vol;
  }
  record_estep_trace_boundary(state,
                              cfg,
                              part,
                              reduction,
                              "step_end",
                              dt_hydro_used,
                              state.t + dt_hydro_used);
  cleanup();
  return out;
}

AleStepResult apply_ale_with_request(
    core::State& state, const core::Config& cfg,
    const AleRequest* request,
    const CellRegime* d_cell_regime,
    const parallel::PartitionInfo& part,
    parallel::CommBuffers* bufs,
    const parallel::Reduction* reduction,
    const HydroEOSContext* eos_ctx,
    const double dt_hydro_used,
    const bool force_rezone,
	    const char* force_reason,
	    tenryu::coupling::ProfileObservability* observability,
	    const AleDriverRetryContext* retry_context) {
  AleStepResult out;
  TENRYU_ASSERT(
      state.corner_stride == 4,
      "corner_stride != 4: runtime-ALE corner path is staged (ALE P2-1c/P2-4+)");
  if (ale_identity_mode_enabled(cfg)) {
    return out;
  }
  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    const bool request_forced = request != nullptr && request->force_rezone;
    return apply_multiblock_csr_ale_step(
        state,
        cfg,
        part,
        reduction,
        eos_ctx,
        dt_hydro_used,
        force_rezone || request_forced,
        force_reason);
  }

  if (state.mesh.dim != 2) {
    return out;
  }
  if (!cfg.numerics.ale.enabled) {
    return out;
  }
  if (cfg.numerics.ale.conservative_remap_enabled && request == nullptr &&
      !force_rezone) {
    return out;
  }
  if (cfg.numerics.plic.enabled && part.n_ranks > 1) {
    throw core::namelist::ConfigError(
        "Wave C PLIC remap not validated under MPI; deferred to Stage 31");
  }
  if (observability != nullptr && state.plic_remap_sticky_fallback) {
    observability->plic_remap_fallback_engaged = true;
  }

  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  const core::State::LaunchWindow nw = state.owned_node_window(n_nodes);
  const core::State::LaunchWindow pw =
      state.owned_cell_window_ghost(n_cells, state.mesh.topo.nz);
  const int blocks_cells = (n_cells + 255) / 256;
  const bool donor_sign_fixed = cfg.numerics.ale.swept_volume_sign_fixed;
  if (n_cells <= 0 || n_nodes <= 0) {
    return out;
  }
  TENRYU_ASSERT(static_cast<int>(state.x_r.size()) == n_nodes,
                "ALE requires x_r size to match mesh topology n_nodes");
  TENRYU_ASSERT(static_cast<int>(state.x_z.size()) == n_nodes,
                "ALE requires x_z size to match mesh topology n_nodes");
  const AleDriverRetryContext default_retry_context{};
  const AleDriverRetryContext& retry_ctx =
      retry_context != nullptr ? *retry_context : default_retry_context;

  const bool ale_cadence_due =
      cfg.numerics.ale.every_n_steps <= 1 ||
      (state.step % cfg.numerics.ale.every_n_steps) == 0;
  AxisMarginResult axis_guard_margin;
  double axis_guard_threshold = 0.0;
  const bool axis_guard_trigger =
      axis_feasibility_guard_trigger(state,
                                     cfg,
                                     reduction,
                                     &axis_guard_margin,
                                     &axis_guard_threshold);
  const std::uint64_t reference_barrier_trigger_mask =
      tenryu::hydro::evaluate_reference_barrier_trigger(state, cfg);
  const auto dump_i1b_polar_pole_diag =
      [&](const char* phase,
          const bool rezone_triggered,
          const bool escape_valve_or_repair_fired) {
        diagnostics::mesh_diag::I1BPolarPoleAleCounters counters;
        counters.phase = phase;
        counters.force_reason = force_reason != nullptr ? force_reason : "";
        counters.axis_guard_trigger = axis_guard_trigger;
        counters.force_rezone_input = force_rezone;
        counters.forced_rezone_fired =
            rezone_triggered && (axis_guard_trigger || force_rezone);
        counters.rezone_triggered = rezone_triggered;
        counters.retry_request_path = (request != nullptr) || retry_ctx.active;
        counters.escape_valve_or_repair_fired = escape_valve_or_repair_fired;
        diagnostics::mesh_diag::dump_i1b_polar_pole_diag(state, cfg, counters);
      };
  if (!ale_cadence_due && !axis_guard_trigger && !force_rezone &&
      reference_barrier_trigger_mask == 0) {
    if (plic::plic_runtime_active(state, cfg)) {
      const plic::PlicRemapStatus drift_status =
          plic::launch_plic_drift_sensor(state,
                                         cfg,
                                         state.x_r.data(),
                                         state.x_z.data(),
                                         state.vol.data(),
                                         nr,
                                         nz,
                                         observability);
      plic::apply_plic_fallback_policy(state, cfg, drift_status, observability);
      out.plic_drift_triggered = drift_status.drift_triggered;
      out.plic_remap_fallback_engaged = state.plic_remap_sticky_fallback;
      out.plic_max_interface_centroid_drift_relative =
          drift_status.max_interface_centroid_drift_relative;
    }
    dump_i1b_polar_pole_diag("ale_skip", false, false);
    return out;
  }
  if (axis_guard_trigger && !ale_cadence_due) {
    core::log_warning("[ale-stats] preventive axis guard triggered ALE at step " +
                      std::to_string(state.step) +
                      " (axis_margin_min=" +
                      std::to_string(axis_guard_margin.min_margin) +
                      ", initial=" + std::to_string(state.axis_margin_initial) +
                      ", threshold=" + std::to_string(axis_guard_threshold) +
                      ", threshold_fraction=" +
                      std::to_string(cfg.numerics.ale.preventive_axis_guard_fraction) +
                      ")");
  }
  if (force_rezone && !ale_cadence_due && !axis_guard_trigger) {
    core::log_warning(std::string("[ale-stats] forced ALE at step ") +
                      std::to_string(state.step) +
                      " (reason=" +
                      (force_reason != nullptr ? force_reason : "unspecified") + ")");
  }
  TENRYU_ASSERT(static_cast<int>(state.v_r.size()) == n_nodes,
                "ALE requires v_r size to match mesh topology n_nodes");
  TENRYU_ASSERT(static_cast<int>(state.v_z.size()) == n_nodes,
                "ALE requires v_z size to match mesh topology n_nodes");
  TENRYU_ASSERT(static_cast<int>(state.rho.size()) == n_cells,
                "ALE requires rho size to match mesh topology n_cells");
  TENRYU_ASSERT(static_cast<int>(state.mass.size()) == n_cells,
                "ALE requires mass size to match mesh topology n_cells");
  TENRYU_ASSERT(static_cast<int>(state.vol.size()) == n_cells,
                "ALE requires vol size to match mesh topology n_cells");
  TENRYU_ASSERT(static_cast<int>(state.ee.size()) == n_cells,
                "ALE requires ee size to match mesh topology n_cells");
  TENRYU_ASSERT(static_cast<int>(state.ei.size()) == n_cells,
                "ALE requires ei size to match mesh topology n_cells");
  TENRYU_ASSERT(static_cast<int>(state.Te.size()) == n_cells,
                "ALE requires Te size to match mesh topology n_cells");
  TENRYU_ASSERT(static_cast<int>(state.Ti.size()) == n_cells,
                "ALE requires Ti size to match mesh topology n_cells");
  TENRYU_ASSERT(static_cast<int>(state.Pe.size()) == n_cells,
                "ALE requires Pe size to match mesh topology n_cells");
  TENRYU_ASSERT(static_cast<int>(state.Pi.size()) == n_cells,
                "ALE requires Pi size to match mesh topology n_cells");
  TENRYU_ASSERT(static_cast<int>(state.zbar.size()) == n_cells,
                "ALE requires zbar size to match mesh topology n_cells");

  double* d_xr_old = nullptr;
  double* d_xz_old = nullptr;
  double* d_xr_cand = nullptr;
  double* d_xz_cand = nullptr;
  double* d_vol_old = nullptr;
  double* d_vr_cell = nullptr;
  double* d_vz_cell = nullptr;
  double* d_tmp = nullptr;
  double* d_vol_mid = nullptr;
  double* d_mom_r = nullptr;
  double* d_mom_z = nullptr;
  double* d_e_e = nullptr;
  double* d_e_i = nullptr;
  double* d_eta_rho_m = nullptr;
  double* d_eta_rho_ee_m = nullptr;
  double* d_eta_rho_ei_m = nullptr;
  double* d_eta_rho_vr_m = nullptr;
  double* d_eta_rho_vz_m = nullptr;
  double* d_eta_rho_m_old = nullptr;
  double* d_eta_rho_ee_m_old = nullptr;
  double* d_eta_rho_ei_m_old = nullptr;
  double* d_eta_rho_vr_m_old = nullptr;
  double* d_eta_rho_vz_m_old = nullptr;
  double* d_volfrac_mat_old = nullptr;
#ifndef NDEBUG
  int per_material_old_snapshot_step = -1;
#endif
  double* d_ke_remap = nullptr;
  double* d_ke_closure_deficit = nullptr;
  double* d_ke_closure_capacity = nullptr;
  double* d_volfrac_mat = nullptr;
  double* d_rad_e_group = nullptr;
  double* d_dm_floor = nullptr;
  double* d_dm_floor_per_material = nullptr;
  double* d_e_floor = nullptr;
  double* d_audit_cell = nullptr;
  double* d_audit_node = nullptr;
  double* d_audit_node_mass = nullptr;
  double* d_audit_vr_node = nullptr;
  double* d_audit_vz_node = nullptr;
  std::uint8_t* d_cell_nverts = detail::upload_cell_nverts_if_tri(state);
  std::uint8_t* d_node_flags = detail::upload_node_flags_if_constraints(state);
  int* d_vf_deg = nullptr;
  int* d_repair_count = nullptr;

  const auto cleanup = [&]() {
    if (d_node_flags != nullptr) {
      CUDA_CHECK(cudaFree(d_node_flags));
    }
    if (d_cell_nverts != nullptr) {
      CUDA_CHECK(cudaFree(d_cell_nverts));
    }
  };

  d_xr_old = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:ale_req:d_xr_old",
                                   static_cast<std::size_t>(n_nodes) * sizeof(double)));
  d_xz_old = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:ale_req:d_xz_old",
                                   static_cast<std::size_t>(n_nodes) * sizeof(double)));
  d_vol_old = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:ale_req:d_vol_old",
                                   static_cast<std::size_t>(n_cells) * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_xr_old,
                        state.x_r.data(),
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_xz_old,
                        state.x_z.data(),
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_vol_old,
                        state.vol.data(),
                        static_cast<std::size_t>(n_cells) * sizeof(double),
                        cudaMemcpyDeviceToDevice));

  const bool ke_closure_enabled = cfg.numerics.ale.ke_conservation_closure;
  const bool ke_closure_redistribute_floor =
      ke_closure_enabled && cfg.numerics.ale.ke_closure_redistribute_floor;
  const bool closure_audit_enabled =
      ke_closure_enabled && cfg.numerics.ale.ke_conservation_closure_audit;
  const std::size_t expected_corner_mass_size =
      static_cast<std::size_t>(n_cells) * 4U;
  if (ke_closure_enabled &&
      (state.corner_mass.size() != expected_corner_mass_size ||
       d_cell_nverts != nullptr)) {
    if (corner_mass_lagrangian_invariant_enabled(cfg)) {
      detail::warn_invariant_corner_mass_ke_reinit_once();
    }
    if (state.corner_mass.size() != expected_corner_mass_size) {
      state.corner_mass.reset(expected_corner_mass_size);
      state.corner_mass_initialized = false;
    }
    detail::compute_current_corner_mass_kernel<<<blocks_cells, 256>>>(
        state.corner_mass.data(), state.mass.data(), state.x_r.data(),
        state.x_z.data(), d_cell_nverts,
        rz::corner_mass_fallback_device_recorder(),
        rz::kCornerMassFallbackStageAleDriverPre,
        static_cast<int>(
            cfg.numerics.hydro.corner_mass_convention),
        nr, nz, state.corner_stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(core::debug_kernel_sync());
    state.corner_mass_initialized = true;
  }
  diagnostics::EnergyTotals closure_audit_pre_budget{};
  if (closure_audit_enabled) {
    closure_audit_pre_budget = diagnostics::compute_energy_totals_2d(state);
  }
  const bool debug_per_remap_log = cfg.numerics.ale.debug_per_remap_log;
  double remap_mass_pre = 0.0;
  double remap_energy_pre = 0.0;
  if (debug_per_remap_log) {
    remap_mass_pre = detail::compute_total_mass(state, reduction);
    remap_energy_pre = detail::total_energy(detail::reduce_energy_totals_global(
        diagnostics::compute_energy_totals_2d(state), reduction));
  }
  conservation_audit::emit_stage(state, "ale_pre_rezone_remap");
  const auto log_axis_uR_if_nonzero = [&]() {
    const double axis_uR_max = detail::compute_axis_uR_max(state, nz, reduction);
    if (axis_uR_max > 0.0) {
      core::log_warning("[ale-stats] axis_uR_max=" +
                        detail::format_scientific(axis_uR_max) +
                        " step=" + std::to_string(state.step));
    }
  };

  if (cfg.numerics.diagnostics.production_audit.gcl.enabled ||
      conservation_audit::enabled()) {
    capture_ale_vol_closure_reference_if_needed(state);
  }

  double pre_rezone_axis_margin = 1.0e300;
  if (cfg.numerics.ale.remap_damage_gate_enabled) {
    pre_rezone_axis_margin =
        compute_axis_margin_min(
            state, reduction, cfg.numerics.has_physical_rz_axis).min_margin;
  }
  const bool phase9_axis_emergency_bypass =
      cfg.numerics.ale.remap_damage_gate_enabled && state.axis_margin_initial > 0.0 &&
      pre_rezone_axis_margin <
          detail::kPhase9AxisEmergencyFraction * state.axis_margin_initial;
  bool phase9_axis_emergency_bypass_logged = false;

  const std::string& repair_mode = cfg.numerics.ale.axis_repair_mode;
  const bool axis_z_motion_winslow =
      cfg.numerics.has_physical_rz_axis &&
      (cfg.numerics.ale.axis_z_motion == "winslow");
  RezoneResult rezone;
  AleMode effective_mode = AleMode::ScheduledDefault;
  const bool request_forced = request != nullptr && request->force_rezone;
  diagnostics::EnergyTotals escape_event_pre_budget{};
  tenryu::coupling::EscapeValveEvent::CellState escape_event_pre_state{};
  bool have_escape_event_pre_budget = false;
  int escape_event_cell_id = request != nullptr ? request->focus_cell : -1;
  if (reference_barrier_trigger_mask != 0) {
    double* d_target_r = nullptr;
    double* d_target_z = nullptr;
    d_target_r = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:ale_req:d_target_r",
                                     static_cast<std::size_t>(n_nodes) * sizeof(double)));
    d_target_z = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:ale_req:d_target_z",
                                     static_cast<std::size_t>(n_nodes) * sizeof(double)));
    tenryu::hydro::build_reference_target_mesh(state, cfg, d_target_r, d_target_z);
    auto ref_result = apply_reference_barrier_rezone_transaction(
        state, cfg, d_target_r, d_target_z);
    ref_result.trigger_reason_mask = reference_barrier_trigger_mask;

    ++state.reference_barrier_engaged_count;
    state.reference_barrier_last_lambda = ref_result.lambda_accepted;
    if (ref_result.succeeded) {
      ++state.reference_barrier_succeeded_count;
    }
    if (cfg.main.verbosity == "verbose") {
      core::log_info("[ref-barrier-ale] step " + std::to_string(state.step) +
                     ": engaged=" + std::to_string(ref_result.engaged) +
                     ", succeeded=" + std::to_string(ref_result.succeeded) +
                     ", lambda=" +
                     detail::format_scientific(ref_result.lambda_accepted) +
                     ", iters=" + std::to_string(ref_result.linesearch_iters) +
                     ", trigger_mask=" +
                     std::to_string(reference_barrier_trigger_mask));
    }
    rezone.triggered = ref_result.engaged;
    rezone.converged = ref_result.succeeded;
    rezone.iterations = ref_result.linesearch_iters;
    rezone.residual = 1.0 - ref_result.lambda_accepted;
    rezone.min_quality = std::min({ref_result.final_quality.min_rz_volume_rel,
                                   ref_result.final_quality.min_corner_j_rel,
                                   ref_result.final_quality.min_gauss_j_rel});
    rezone.accepted_lambda = ref_result.lambda_accepted;
    rezone.final_axis_margin =
        compute_axis_margin_min(
            state, reduction, cfg.numerics.has_physical_rz_axis).min_margin;
    out.energy_audit_ale_fallback =
        out.energy_audit_ale_fallback ||
        (ref_result.engaged && !ref_result.succeeded);
    out.mass_floor_delta += ref_result.mass_floor_delta;
    out.E_floor_injected += ref_result.E_floor_injected;
    out.E_redistribution_unresolved += ref_result.E_redistribution_unresolved;
  } else if (request_forced && request->mode != AleMode::ScheduledDefault) {
    const bool request_uses_local_repair =
        request->mode == AleMode::AxisSpinePlusLocal ||
        request->mode == AleMode::BoundaryPatchProjection ||
        request->mode == AleMode::CdLocalWinslow ||
        request->mode == AleMode::InteriorMultiNodeProjection ||
        request->mode == AleMode::AxisVariationalProjection;
    if (observability != nullptr && request_uses_local_repair) {
      escape_event_pre_budget = detail::reduce_energy_totals_global(
          diagnostics::compute_energy_totals_2d(state), reduction);
      escape_event_pre_state =
          detail::capture_escape_valve_cell_state(state, escape_event_cell_id);
      have_escape_event_pre_budget = true;
    }
    std::vector<double> local_repair_r_original;
    std::vector<double> local_repair_z_original;
    if (request_uses_local_repair) {
      state.x_r.copy_to_host(local_repair_r_original);
      state.x_z.copy_to_host(local_repair_z_original);
    }
    const bool request_eligible_for_interior_escalation =
        request->mode == AleMode::AxisSpinePlusLocal ||
        request->mode == AleMode::BoundaryPatchProjection;
    double pre_request_min_quality = 1.0;
    if (request_uses_local_repair) {
      bool pre_request_tangle = false;
      AleMinQualityCell pre_request_min_cell;
      pre_request_min_quality =
          compute_min_quality(state, cfg, &pre_request_tangle, &pre_request_min_cell);
      if (reduction != nullptr) {
        pre_request_min_quality = reduction->allreduce_min(pre_request_min_quality);
      }
    }
    const auto restore_local_repair_original = [&]() {
      if (!local_repair_r_original.empty() && !local_repair_z_original.empty()) {
        state.x_r.copy_from_host(local_repair_r_original);
        state.x_z.copy_from_host(local_repair_z_original);
      }
    };
    const auto emit_axis_projection_attempt =
        [&](const RezoneResult& av_attempt,
            const char* phase_tag,
            const bool include_post_failure) {
          if (!cfg.numerics.diagnostics.mesh_attribution.enabled) {
            return;
          }
          diagnostics::mesh_attribution::AxisProjectionRecord record;
          record.step = static_cast<std::uint64_t>(std::max(state.step, 0));
          record.t_simulation = state.t;
          record.phase_tag = phase_tag;
          const int nr_cells = state.mesh.topo.nr;
          const int nz_cells = state.mesh.topo.nz;
          int ci = (nr_cells > 0) ? nr_cells / 2 : 0;
          int cj = (nz_cells > 0) ? nz_cells / 2 : 0;
          if (request->focus_cell >= 0 && nz_cells > 0) {
            ci = request->focus_cell / nz_cells;
            cj = request->focus_cell - ci * nz_cells;
          }
          ci = std::clamp(ci, 0, std::max(nr_cells - 1, 0));
          cj = std::clamp(cj, 0, std::max(nz_cells - 1, 0));
          const int radius_j = std::max(request->patch_radius_j, 0);
          record.patch_i0 = 0;
          record.patch_i1 = std::min(
              nr_cells - 1,
              std::max(0, cfg.numerics.hydro.axis_guard_band_cells));
          record.patch_j0 =
              std::clamp(cj - radius_j, 0, std::max(nz_cells - 1, 0));
          record.patch_j1 =
              std::clamp(cj + radius_j, 0, std::max(nz_cells - 1, 0));
          record.min_local_j = av_attempt.residual;
          record.pre_request_min_quality = pre_request_min_quality;
          record.post_min_quality = av_attempt.min_quality;
          if (include_post_failure && av_attempt.triggered) {
            AleMinQualityCell post_fail_cell;
            double min_corner_j = 1.0;
            int post_fail_corner = -1;
            const bool post_corner_tangle = compute_corner_post_tangle(
                state, cfg, &post_fail_cell, &min_corner_j, &post_fail_corner);
            record.post_failing_cell =
                post_corner_tangle ? post_fail_cell.c
                                   : av_attempt.min_quality_cell_post.c;
            record.post_failing_corner = post_corner_tangle ? post_fail_corner : -1;
          }
          diagnostics::mesh_attribution::emit_axis_projection_record(
              state, cfg, record);
        };
    switch (request->mode) {
      case AleMode::FullWinslow:
        rezone = run_winslow_rezone_with_parallel(
            state, cfg, part, bufs, reduction, true);
        effective_mode = AleMode::FullWinslow;
        break;
      case AleMode::AxisSpineOnly:
        if (cfg.numerics.has_physical_rz_axis) {
          rezone = run_axis_spine_rezone_with_parallel(
              state, cfg, part, bufs, reduction, true);
          effective_mode = AleMode::AxisSpineOnly;
        } else {
          rezone = run_winslow_rezone_with_parallel(
              state, cfg, part, bufs, reduction, true);
          effective_mode = AleMode::FullWinslow;
        }
        break;
      case AleMode::AxisSpinePlusLocal:
        if (cfg.numerics.has_physical_rz_axis) {
          rezone = run_axis_spine_plus_local_rezone(
              state, cfg, part, bufs, reduction, *request, d_cell_regime);
          effective_mode = AleMode::AxisSpinePlusLocal;
        } else {
          rezone = run_winslow_rezone_with_parallel(
              state, cfg, part, bufs, reduction, true);
          effective_mode = AleMode::FullWinslow;
        }
        break;
      case AleMode::BoundaryPatchProjection:
        rezone = run_boundary_patch_projection(
            state, cfg, part, bufs, reduction, *request);
        effective_mode = AleMode::BoundaryPatchProjection;
        break;
      case AleMode::CdLocalWinslow:
        rezone = run_cd_local_winslow_rezone(
            state, cfg, part, bufs, reduction, *request, d_cell_regime);
        effective_mode = AleMode::CdLocalWinslow;
        break;
      case AleMode::InteriorMultiNodeProjection:
        rezone = cfg.numerics.ale.multi_node_interior_repair_enabled
                     ? run_interior_multi_node_projection(
                           state, cfg, part, bufs, reduction, *request)
                     : run_winslow_rezone_with_parallel(
                           state, cfg, part, bufs, reduction, true);
        effective_mode = cfg.numerics.ale.multi_node_interior_repair_enabled
                             ? AleMode::InteriorMultiNodeProjection
                             : AleMode::FullWinslow;
        break;
      case AleMode::AxisVariationalProjection:
        rezone = cfg.numerics.has_physical_rz_axis &&
                         cfg.numerics.ale.axis_variational_projection_enabled
                     ? run_axis_variational_projection(
                           state, cfg, part, bufs, reduction, *request, d_cell_regime)
                     : run_winslow_rezone_with_parallel(
                           state, cfg, part, bufs, reduction, true);
        effective_mode = cfg.numerics.has_physical_rz_axis &&
                                 cfg.numerics.ale.axis_variational_projection_enabled
                             ? AleMode::AxisVariationalProjection
                             : AleMode::FullWinslow;
        break;
      case AleMode::ScheduledDefault:
        break;
    }
    if (!rezone.triggered || !rezone.converged) {
      effective_mode = AleMode::ScheduledDefault;
    }
    if ((!rezone.triggered || !rezone.converged) &&
        request->mode != AleMode::FullWinslow &&
        request->mode != AleMode::AxisSpineOnly) {
      if (request->mode == AleMode::AxisSpinePlusLocal &&
          cfg.numerics.has_physical_rz_axis &&
          cfg.numerics.ale.axis_variational_projection_enabled) {
        restore_local_repair_original();
        RezoneResult av_attempt = run_axis_variational_projection(
            state, cfg, part, bufs, reduction, *request, d_cell_regime);
        if (ale_escalation_quality_gate_accepts(
                av_attempt, pre_request_min_quality)) {
          core::log_warning(
              "[ale-stats] AxisVariationalProjection escalation accepted");
          emit_axis_projection_attempt(av_attempt, "accepted", false);
          rezone = av_attempt;
          effective_mode = AleMode::AxisVariationalProjection;
        } else {
          core::log_warning(
              "[ale-stats] AxisVariationalProjection escalation rejected by quality gate; continuing escalation");
          emit_axis_projection_attempt(
              av_attempt,
              av_attempt.triggered ? "post_global_quality_failed"
                                   : "local_projection_failed",
              av_attempt.triggered);
          restore_local_repair_original();
        }
      }
      if ((!rezone.triggered || !rezone.converged) &&
          request_eligible_for_interior_escalation &&
          cfg.numerics.ale.multi_node_interior_repair_enabled) {
        restore_local_repair_original();
        RezoneResult interior = run_interior_multi_node_projection(
            state, cfg, part, bufs, reduction, *request);
        if (ale_escalation_quality_gate_accepts(
                interior, pre_request_min_quality)) {
          core::log_warning(
              "[ale-stats] interior escalation accepted; skipping FullWinslow");
          rezone = interior;
          effective_mode = AleMode::InteriorMultiNodeProjection;
        } else {
          restore_local_repair_original();
          core::log_warning(
              "[ale-stats] interior escalation rejected by quality gate; falling back to FullWinslow");
          rezone = run_winslow_rezone_with_parallel(
              state, cfg, part, bufs, reduction, true);
          if (rezone.triggered && rezone.converged) {
            effective_mode = AleMode::FullWinslow;
          }
        }
      } else if (!rezone.triggered || !rezone.converged) {
        restore_local_repair_original();
        core::log_warning(
            "[ale-stats] requested local repair did not converge; falling back to full Winslow");
        rezone = run_winslow_rezone_with_parallel(
            state, cfg, part, bufs, reduction, true);
        if (rezone.triggered && rezone.converged) {
          effective_mode = AleMode::FullWinslow;
        }
      }
    }
  } else if (repair_mode == "axis_spine_only" &&
             cfg.numerics.has_physical_rz_axis) {
    rezone = run_axis_spine_rezone_with_parallel(state,
                                                 cfg,
                                                 part,
                                                 bufs,
                                                 reduction,
                                                 axis_guard_trigger || force_rezone);
    const double phase10_fallback_threshold =
        detail::kPhase10SpineFallbackFraction * state.axis_margin_initial;
    if (rezone.triggered && state.axis_margin_initial > 0.0 &&
        rezone.final_axis_margin < phase10_fallback_threshold) {
      core::log_warning(
          "[ale-stats] Phase 10 spine-only insufficient; falling back to full Winslow "
          "(axis_margin_min=" +
          std::to_string(rezone.final_axis_margin) +
          ", threshold=" + std::to_string(phase10_fallback_threshold) + ")");
      rezone = run_winslow_rezone_with_parallel(state, cfg, part, bufs, reduction, true);
    }
  } else {
    if (repair_mode == "axis_z_winslow") {
      static bool warned_axis_z_winslow_fallback = false;
      if (!warned_axis_z_winslow_fallback) {
        core::log_warning("[ale] axis_z_winslow not yet stable; falling back to full_winslow");
        warned_axis_z_winslow_fallback = true;
      }
    }
    rezone = run_winslow_rezone_with_parallel(state,
                                              cfg,
                                              part,
                                              bufs,
                                              reduction,
                                              axis_guard_trigger || force_rezone);
  }
  out.rezone_triggered = rezone.triggered;
  out.rezone_converged = rezone.converged;
  out.quality_min = rezone.min_quality;
  out.rezone_iterations = rezone.iterations;
  out.rezone_residual = rezone.residual;
  out.effective_mode_executed = effective_mode;
  if (observability != nullptr && rezone.triggered && rezone.converged) {
    const diagnostics::EnergyTotals escape_event_post_budget =
        detail::reduce_energy_totals_global(
            diagnostics::compute_energy_totals_2d(state), reduction);
    const auto escape_event_post_state =
        detail::capture_escape_valve_cell_state(state, escape_event_cell_id);
    const char* event_reason =
        force_reason != nullptr ? force_reason : "requested ALE repair";
    if (effective_mode == AleMode::BoundaryPatchProjection) {
      observability->note_class_c_fire("ale_multi_node_boundary_repair");
      if (have_escape_event_pre_budget) {
        observability->note_escape_valve_event(
            detail::make_escape_valve_event(
                state,
                "post_step_ale",
                "MultiNodeBoundaryRepair",
                "multi_node_boundary_repair_count",
                event_reason,
                escape_event_cell_id,
                escape_event_pre_state,
                escape_event_post_state,
                0.0,
                0.0,
                0.0,
                escape_event_pre_budget,
                escape_event_post_budget));
      }
    } else if (effective_mode == AleMode::InteriorMultiNodeProjection) {
      observability->note_class_c_fire("ale_multi_node_interior_repair");
      if (have_escape_event_pre_budget) {
        observability->note_escape_valve_event(
            detail::make_escape_valve_event(
                state,
                "post_step_ale",
                "InteriorMultiNodeProjection",
                "multi_node_interior_repair_count",
                event_reason,
                escape_event_cell_id,
                escape_event_pre_state,
                escape_event_post_state,
                0.0,
                0.0,
                0.0,
                escape_event_pre_budget,
                escape_event_post_budget));
      }
    } else if (effective_mode == AleMode::AxisVariationalProjection) {
      observability->note_class_c_fire("ale_axis_variational_projection");
      if (have_escape_event_pre_budget) {
        observability->note_escape_valve_event(
            detail::make_escape_valve_event(
                state,
                "post_step_ale",
                "AxisVariationalProjection",
                "axis_variational_projection_count",
                event_reason,
                escape_event_cell_id,
                escape_event_pre_state,
                escape_event_post_state,
                0.0,
                0.0,
                0.0,
                escape_event_pre_budget,
                escape_event_post_budget));
      }
    }
  }

  if (!rezone.triggered) {
    log_axis_uR_if_nonzero();
    cleanup();
    return out;
  }

  d_xr_cand = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:ale_req:d_xr_cand",
                                   static_cast<std::size_t>(n_nodes) * sizeof(double)));
  d_xz_cand = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:ale_req:d_xz_cand",
                                   static_cast<std::size_t>(n_nodes) * sizeof(double)));
  d_vol_mid = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:ale_req:d_vol_mid",
                                   static_cast<std::size_t>(n_cells) * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_xr_cand,
                        state.x_r.data(),
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_xz_cand,
                        state.x_z.data(),
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice));

  bool post_tangle = false;
  bool post_corner_tangle = false;
  AleMinQualityCell post_min_quality_cell;
  AleMinQualityCell post_corner_fail_cell;
  int post_corner_fail_corner = -1;
  double post_min_corner_j = 1.0;
  AleMinQualityCell first_backtrack_fail_cell;
  bool first_backtrack_fail_recorded = false;
  AleMinQualityCell first_corner_backtrack_fail_cell;
  int first_corner_backtrack_fail_corner = -1;
  bool first_corner_backtrack_fail_recorded = false;
  bool accepted_backtrack = false;
  int predictive_axis_rejects = 0;
  int predictive_cell_vol_rejects = 0;
  bool phase9_candidate_passed = false;
  bool have_phase9_emergency = false;
  double phase9_emergency_lambda = 0.0;
  double phase9_emergency_quality = 0.0;
  double phase9_emergency_axis_margin = 0.0;
  AleMinQualityCell phase9_emergency_post_cell;
  detail::RemapDamageResult phase9_emergency_damage;
  std::string backtrack_last_reason = "backtrack exhausted";
  bool safe_backtrack_terminal_failure = false;
  const int blocks_nodes = (n_nodes + 255) / 256;
  std::vector<double> accepted_axis_inflow_this_event;
  const bool remap_damage_budget_enabled =
      cfg.numerics.ale.remap_damage_gate_enabled &&
      cfg.numerics.ale.remap_damage_axis_budget_enabled;
  const bool predictive_acceptance_active =
      cfg.numerics.ale.predictive_acceptance_enabled &&
      dt_hydro_used > 0.0 && std::isfinite(dt_hydro_used);
  const int lambda_sweep_target_cell_c = [&]() {
    if (cfg.numerics.ale.lambda_sweep_target_cell_c >= 0) {
      return cfg.numerics.ale.lambda_sweep_target_cell_c;
    }
    const int i = cfg.numerics.ale.lambda_sweep_target_cell_i;
    const int j = cfg.numerics.ale.lambda_sweep_target_cell_j;
    if (i >= 0 && j >= 0 && i < nr && j < nz) {
      return i * nz + j;
    }
    return -1;
  }();
  bool lambda_sweep_dumped = false;

  const auto maybe_dump_lambda_sweep = [&](const AleMinQualityCell& fail_cell) {
    if (!cfg.numerics.ale.lambda_sweep_diagnostic_enabled ||
        lambda_sweep_dumped ||
        lambda_sweep_target_cell_c < 0 ||
        fail_cell.c != lambda_sweep_target_cell_c) {
      return;
    }

    const LambdaSweepResult sweep =
        evaluate_lambda_sweep(state,
                              d_xr_old,
                              d_xz_old,
                              d_xr_cand,
                              d_xz_cand,
                              lambda_sweep_target_cell_c,
                              cfg.numerics.ale.lambda_sweep_max_exp);
    lambda_sweep_dumped = true;
    const char* classification = classify_lambda_sweep(sweep);

    state.ale_lambda_sweep_target_cell_c = sweep.target_cell_c;
    state.ale_lambda_sweep_target_cell_i = sweep.target_cell_i;
    state.ale_lambda_sweep_target_cell_j = sweep.target_cell_j;
    state.ale_lambda_sweep_classification = classification;
    state.ale_lambda_sweep_lambda.clear();
    state.ale_lambda_sweep_min_gauss_j.clear();
    state.ale_lambda_sweep_min_corner_j.clear();
    state.ale_lambda_sweep_min_v_rz.clear();
    state.ale_lambda_sweep_admissible.clear();
    state.ale_lambda_sweep_lambda.reserve(sweep.points.size());
    state.ale_lambda_sweep_min_gauss_j.reserve(sweep.points.size());
    state.ale_lambda_sweep_min_corner_j.reserve(sweep.points.size());
    state.ale_lambda_sweep_min_v_rz.reserve(sweep.points.size());
    state.ale_lambda_sweep_admissible.reserve(sweep.points.size());
    for (const LambdaSweepPoint& point : sweep.points) {
      state.ale_lambda_sweep_lambda.push_back(point.lambda);
      state.ale_lambda_sweep_min_gauss_j.push_back(point.min_gauss_j);
      state.ale_lambda_sweep_min_corner_j.push_back(point.min_corner_j);
      state.ale_lambda_sweep_min_v_rz.push_back(point.min_v_rz);
      state.ale_lambda_sweep_admissible.push_back(
          static_cast<std::uint8_t>(point.admissible ? 1u : 0u));
    }

    std::ostringstream os;
    os << "[ale_lambda_sweep] target=(c=" << sweep.target_cell_c
       << ",i=" << sweep.target_cell_i << ",j=" << sweep.target_cell_j
       << ") max_exp=" << sweep.max_exp
       << " classification=" << classification << '\n';
    os << "[ale_lambda_sweep] lambda min_gauss_J_cm2 min_corner_J_cm2 "
          "min_V_RZ_cm3 admissible";
    for (const LambdaSweepPoint& point : sweep.points) {
      os << '\n'
         << "[ale_lambda_sweep] " << std::scientific << std::setprecision(12)
         << point.lambda << ' ' << point.min_gauss_j << ' '
         << point.min_corner_j << ' ' << point.min_v_rz << ' '
         << (point.admissible ? 1 : 0);
    }
    core::log_info(os.str());
  };

  const auto ensure_remap_damage_budget_state = [&]() {
    if (!remap_damage_budget_enabled) {
      return;
    }
    if (state.axis_mass_initial.empty()) {
      std::vector<double> mass_h;
      state.mass.copy_to_host(mass_h);
      state.axis_mass_initial.assign(static_cast<std::size_t>(nz), 0.0);
      for (int j = 0; j < nz; ++j) {
        state.axis_mass_initial[static_cast<std::size_t>(j)] =
            mass_h[static_cast<std::size_t>(j)];
      }
    }
    if (state.axis_inflow_budget.empty()) {
      state.axis_inflow_budget.assign(static_cast<std::size_t>(nz), 0.0);
    }
    TENRYU_ASSERT(static_cast<int>(state.axis_mass_initial.size()) == nz,
                  "ALE remap damage axis mass initial size mismatch");
    TENRYU_ASSERT(static_cast<int>(state.axis_inflow_budget.size()) == nz,
                  "ALE remap damage axis inflow budget size mismatch");
  };

  const auto check_remap_damage_gate = [&](detail::RemapDamageResult& dmg) -> bool {
    if (!cfg.numerics.ale.remap_damage_gate_enabled) {
      dmg = detail::RemapDamageResult{};
      return true;
    }
    if (phase9_axis_emergency_bypass) {
      dmg = detail::RemapDamageResult{};
      if (!phase9_axis_emergency_bypass_logged) {
        core::log_warning(
            "[ale-stats] Phase 9 axis-emergency bypass: damage gate disabled this ALE "
            "invoke (axis_margin_min=" +
            std::to_string(pre_rezone_axis_margin) + ")");
        phase9_axis_emergency_bypass_logged = true;
      }
      return true;
    }

    ensure_remap_damage_budget_state();
    const std::vector<double>* axis_mass_initial =
        state.axis_mass_initial.empty() ? nullptr : &state.axis_mass_initial;
    dmg = detail::compute_remap_damage(state.rho.data(),
                                       d_vol_old,
                                       d_xr_old,
                                       d_xz_old,
                                       state.x_r.data(),
                                       state.x_z.data(),
                                       nr,
                                       nz,
                                       cfg.numerics.floors.rho,
                                       axis_mass_initial,
                                       donor_sign_fixed);

    detail::RemapDamageResult log_dmg = dmg;
    double D_rho_max = dmg.D_rho_max;
    if (reduction != nullptr) {
      D_rho_max = reduction->allreduce_max(D_rho_max);
    }
    log_dmg.D_rho_max = D_rho_max;
    if (D_rho_max > cfg.numerics.ale.remap_damage_dmax) {
      backtrack_last_reason = "remap damage D_rho_max exceeds threshold";
      detail::log_ale_damage_stats(log_dmg);
      if (!first_backtrack_fail_recorded) {
        first_backtrack_fail_cell.c = -1;
        first_backtrack_fail_cell.i = -1;
        first_backtrack_fail_cell.j = -1;
        first_backtrack_fail_recorded = true;
      }
      return false;
    }

    double A_axis_max = dmg.A_axis_max;
    if (reduction != nullptr) {
      A_axis_max = reduction->allreduce_max(A_axis_max);
    }
    log_dmg.A_axis_max = A_axis_max;
    if (A_axis_max > cfg.numerics.ale.remap_damage_axis_eta) {
      backtrack_last_reason = "axis remap damage A_0j exceeds eta threshold";
      detail::log_ale_damage_stats(log_dmg);
      if (!first_backtrack_fail_recorded) {
        first_backtrack_fail_cell.c = dmg.axis_max_j;
        first_backtrack_fail_cell.i = 0;
        first_backtrack_fail_cell.j = dmg.axis_max_j;
        first_backtrack_fail_recorded = true;
      }
      return false;
    }

    if (remap_damage_budget_enabled && !state.axis_mass_initial.empty()) {
      int budget_fail_j = -1;
      bool budget_exceeded = detail::remap_damage_axis_budget_exceeded(
          dmg,
          state.axis_inflow_budget,
          state.axis_mass_initial,
          cfg.numerics.ale.remap_damage_axis_budget_factor,
          &budget_fail_j);
      if (reduction != nullptr) {
        budget_exceeded =
            (reduction->allreduce_max(budget_exceeded ? 1.0 : 0.0) > 0.5);
      }
      if (budget_exceeded) {
        backtrack_last_reason = "axis cumulative inflow budget exceeded";
        detail::log_ale_damage_stats(log_dmg);
        if (!first_backtrack_fail_recorded) {
          first_backtrack_fail_cell.c = budget_fail_j;
          first_backtrack_fail_cell.i = 0;
          first_backtrack_fail_cell.j = budget_fail_j;
          first_backtrack_fail_recorded = true;
        }
        return false;
      }
    }

    core::log_warning("[ale-stats] damage_accepted D_rho_max=" +
                      std::to_string(log_dmg.D_rho_max) +
                      " A_axis_max=" + std::to_string(log_dmg.A_axis_max) +
                      " axis_max_j=" + std::to_string(log_dmg.axis_max_j));
    return true;
  };

  const auto check_predictive_acceptance_gate =
      [&](const double* d_xr_trial, const double* d_xz_trial) -> bool {
    if (!predictive_acceptance_active) {
      return true;
    }
    const detail::PredictiveAcceptanceResult pred =
        detail::check_next_step_feasibility(
            state,
            d_xr_trial,
            d_xz_trial,
            dt_hydro_used,
            cfg.numerics.ale.predictive_acceptance_axis_floor_fraction,
            cfg.numerics.ale.predictive_acceptance_cell_vol_floor_fraction,
            cfg.numerics.has_physical_rz_axis,
            reduction);
    diagnostics::mesh_diag::dump_ale_predictive_acceptance(
        state,
        cfg,
        predictive_acceptance_active,
        pred.feasible,
        detail::failure_class_name(pred.failure_class),
        pred.axis_failure_count,
        pred.cell_vol_failure_count,
        pred.first_axis_failing_j,
        pred.first_vol_failing_c,
        pred.candidate_axis_margin_min,
        pred.trial_axis_margin_min,
        pred.candidate_cell_vol_min,
        pred.trial_cell_vol_min);
    if (pred.feasible) {
      return true;
    }
    if (pred.failure_class == detail::FailureClass::AxisMargin) {
      ++predictive_axis_rejects;
      backtrack_last_reason = "predictive acceptance: axis margin would invert";
      if (!first_backtrack_fail_recorded) {
        first_backtrack_fail_cell.c = pred.first_axis_failing_j;
        first_backtrack_fail_cell.i = 0;
        first_backtrack_fail_cell.j = pred.first_axis_failing_j;
        first_backtrack_fail_recorded = true;
      }
    } else {
      ++predictive_cell_vol_rejects;
      backtrack_last_reason = "predictive acceptance: cell volume would invert";
      if (!first_backtrack_fail_recorded) {
        first_backtrack_fail_cell.c = pred.first_vol_failing_c;
        first_backtrack_fail_cell.i =
            (pred.first_vol_failing_c >= 0) ? pred.first_vol_failing_c / nz : -1;
        first_backtrack_fail_cell.j =
            (pred.first_vol_failing_c >= 0) ? pred.first_vol_failing_c % nz : -1;
        first_backtrack_fail_recorded = true;
      }
    }
    return false;
  };

  const auto log_predictive_rejects_if_any = [&]() {
    detail::log_ale_predictive_acceptance_stats(predictive_axis_rejects,
                                                predictive_cell_vol_rejects,
                                                dt_hydro_used);
  };

  if (!cfg.numerics.ale.safe_backtrack_enabled) {
    for (int attempt = 0; attempt < detail::kBacktrackMaxAttempts; ++attempt) {
      const double lambda = detail::kBacktrackLambdaSchedule[attempt];
      rezone.backtrack_attempts = attempt + 1;
      if (attempt > 0) {
        detail::interpolate_rezone_candidate_kernel<<<blocks_nodes, 256>>>(
            state.x_r.data(),
            state.x_z.data(),
            d_xr_old,
            d_xz_old,
            d_xr_cand,
            d_xz_cand,
            lambda,
            n_nodes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(core::debug_kernel_sync());
      }

      post_tangle = false;
      post_min_quality_cell = AleMinQualityCell{};
      post_corner_tangle = false;
      post_corner_fail_cell = AleMinQualityCell{};
      post_corner_fail_corner = -1;
      post_min_corner_j = 1.0;
      double trial_quality =
          compute_min_quality(state, cfg, &post_tangle, &post_min_quality_cell);
      if (reduction != nullptr) {
        trial_quality = reduction->allreduce_min(trial_quality);
        post_tangle = (reduction->allreduce_max(post_tangle ? 1.0 : 0.0) > 0.5);
      }
      if (cfg.numerics.ale.corner_jacobian_post_tangle_enabled) {
        post_corner_tangle = compute_corner_post_tangle(state,
                                                        cfg,
                                                        &post_corner_fail_cell,
                                                        &post_min_corner_j,
                                                        &post_corner_fail_corner);
        if (reduction != nullptr) {
          post_min_corner_j = reduction->allreduce_min(post_min_corner_j);
          post_corner_tangle =
              (reduction->allreduce_max(post_corner_tangle ? 1.0 : 0.0) > 0.5);
        }
      }
      const auto dump_backtrack_iter = [&](const bool accepted) {
        diagnostics::mesh_diag::dump_ale_backtrack_iter(state,
                                                        cfg,
                                                        attempt,
                                                        lambda,
                                                        post_tangle,
                                                        post_corner_tangle,
                                                        trial_quality,
                                                        post_min_corner_j,
                                                        post_min_quality_cell.c,
                                                        post_min_quality_cell.i,
                                                        post_min_quality_cell.j,
                                                        post_corner_fail_cell.c,
                                                        post_corner_fail_cell.i,
                                                        post_corner_fail_cell.j,
                                                        post_corner_fail_corner,
                                                        accepted,
                                                        backtrack_last_reason.c_str());
      };
      if (post_tangle || post_corner_tangle) {
        const AleMinQualityCell current_fail_cell =
            post_tangle ? post_min_quality_cell : post_corner_fail_cell;
        if (post_tangle) {
          ++rezone.gauss_tangle_rejection_count;
          backtrack_last_reason = "post-rezone Gauss tangle";
        } else {
          ++rezone.corner_tangle_rejection_count;
          backtrack_last_reason = "post-rezone corner-J tangle";
        }
        if (!first_backtrack_fail_recorded) {
          first_backtrack_fail_cell = current_fail_cell;
          first_backtrack_fail_recorded = true;
          maybe_dump_lambda_sweep(first_backtrack_fail_cell);
        }
        if (post_corner_tangle && !first_corner_backtrack_fail_recorded) {
          first_corner_backtrack_fail_cell = post_corner_fail_cell;
          first_corner_backtrack_fail_corner = post_corner_fail_corner;
          first_corner_backtrack_fail_recorded = true;
        }
        dump_backtrack_iter(false);
        continue;
      }

      AxisMarginResult axis_margin =
          compute_axis_margin_min(state, cfg.numerics.has_physical_rz_axis);
      bool axis_admissible = (axis_margin.min_margin > 0.0);
      rezone.final_axis_margin = axis_margin.min_margin;
      if (reduction != nullptr) {
        rezone.final_axis_margin = reduction->allreduce_min(rezone.final_axis_margin);
        axis_admissible =
            (reduction->allreduce_max(axis_admissible ? 0.0 : 1.0) < 0.5);
      }
      if (!axis_admissible) {
        backtrack_last_reason = "axis-cell analytic margin negative";
        ++rezone.other_rejection_count;
        if (!first_backtrack_fail_recorded) {
          first_backtrack_fail_cell.c = axis_margin.min_j;
          first_backtrack_fail_cell.i = 0;
          first_backtrack_fail_cell.j = axis_margin.min_j;
          first_backtrack_fail_recorded = true;
        }
        dump_backtrack_iter(false);
        continue;
      }

      if (axis_z_motion_winslow) {
        const AxisRadialBoundResult radial_bound =
            check_axis_radial_bound(state.x_r.data(),
                                    d_xr_old,
                                    nr,
                                    nz,
                                    cfg.numerics.ale.winslow_axis_kappa,
                                    reduction);
        if (!radial_bound.passed) {
          backtrack_last_reason = "axis-adjacent radial bound violated";
          ++rezone.other_rejection_count;
          if (!first_backtrack_fail_recorded) {
            first_backtrack_fail_cell.c = radial_bound.fail_j;
            first_backtrack_fail_cell.i = 1;
            first_backtrack_fail_cell.j = radial_bound.fail_j;
            first_backtrack_fail_recorded = true;
          }
          dump_backtrack_iter(false);
          continue;
        }
      }

      const bool capture_phase9_emergency =
          cfg.numerics.ale.remap_damage_gate_enabled && !have_phase9_emergency;
      if (capture_phase9_emergency) {
        phase9_emergency_lambda = lambda;
        phase9_emergency_quality = trial_quality;
        phase9_emergency_axis_margin = rezone.final_axis_margin;
        phase9_emergency_post_cell = post_min_quality_cell;
        have_phase9_emergency = true;
      }

      detail::RemapDamageResult trial_damage;
      if (!check_remap_damage_gate(trial_damage)) {
        ++rezone.other_rejection_count;
        if (capture_phase9_emergency) {
          phase9_emergency_damage = trial_damage;
        }
        dump_backtrack_iter(false);
        continue;
      }
      if (cfg.numerics.ale.remap_damage_gate_enabled) {
        phase9_candidate_passed = true;
      }

      detail::RemapAdmissibilityResult remap_admissibility =
          detail::check_first_sweep_admissibility(d_vol_mid,
                                                  d_vol_old,
                                                  d_xr_old,
                                                  d_xz_old,
                                                  state.x_r.data(),
                                                  state.x_z.data(),
                                                  cw.begin,
                                                  cw.end,
                                                  nr,
                                                  nz,
                                                  state.step,
                                                  donor_sign_fixed);
      bool remap_admissible = remap_admissibility.admissible;
      if (reduction != nullptr) {
        remap_admissible =
            (reduction->allreduce_max(remap_admissible ? 0.0 : 1.0) < 0.5);
      }
      if (!remap_admissible) {
        backtrack_last_reason = "remap non-positive intermediate volume";
        ++rezone.other_rejection_count;
        if (!first_backtrack_fail_recorded) {
          first_backtrack_fail_cell = remap_admissibility.first_fail_cell;
          first_backtrack_fail_recorded = true;
        }
        dump_backtrack_iter(false);
        continue;
      }

      if (!check_predictive_acceptance_gate(state.x_r.data(), state.x_z.data())) {
        ++rezone.other_rejection_count;
        dump_backtrack_iter(false);
        continue;
      }

      accepted_axis_inflow_this_event = trial_damage.axis_inflow_this_event;
      accepted_backtrack = true;
      rezone.accepted_lambda = lambda;
      out.quality_min = trial_quality;
      backtrack_last_reason = "accepted";
      dump_backtrack_iter(true);
      break;
    }
  } else {
    struct SafeBacktrackTrial {
      bool admissible = false;
      double trial_quality = 0.0;
      detail::RemapDamageResult damage;
    };

    const auto mesh_admissible_at_lambda =
        [&](const double lambda,
            const int attempt,
            const bool count_rejections,
            const bool dump_iter,
            const bool allow_phase9_emergency_capture) -> SafeBacktrackTrial {
      detail::interpolate_rezone_candidate_kernel<<<blocks_nodes, 256>>>(
          state.x_r.data(),
          state.x_z.data(),
          d_xr_old,
          d_xz_old,
          d_xr_cand,
          d_xz_cand,
          lambda,
          n_nodes);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(core::debug_kernel_sync());

      SafeBacktrackTrial result;
      post_tangle = false;
      post_min_quality_cell = AleMinQualityCell{};
      post_corner_tangle = false;
      post_corner_fail_cell = AleMinQualityCell{};
      post_corner_fail_corner = -1;
      post_min_corner_j = 1.0;
      double trial_quality =
          compute_min_quality(state, cfg, &post_tangle, &post_min_quality_cell);
      if (reduction != nullptr) {
        trial_quality = reduction->allreduce_min(trial_quality);
        post_tangle = (reduction->allreduce_max(post_tangle ? 1.0 : 0.0) > 0.5);
      }
      if (cfg.numerics.ale.corner_jacobian_post_tangle_enabled) {
        post_corner_tangle = compute_corner_post_tangle(state,
                                                        cfg,
                                                        &post_corner_fail_cell,
                                                        &post_min_corner_j,
                                                        &post_corner_fail_corner);
        if (reduction != nullptr) {
          post_min_corner_j = reduction->allreduce_min(post_min_corner_j);
          post_corner_tangle =
              (reduction->allreduce_max(post_corner_tangle ? 1.0 : 0.0) > 0.5);
        }
      }
      result.trial_quality = trial_quality;

      const auto dump_backtrack_iter = [&](const bool accepted) {
        if (!dump_iter) {
          return;
        }
        diagnostics::mesh_diag::dump_ale_backtrack_iter(state,
                                                        cfg,
                                                        attempt,
                                                        lambda,
                                                        post_tangle,
                                                        post_corner_tangle,
                                                        trial_quality,
                                                        post_min_corner_j,
                                                        post_min_quality_cell.c,
                                                        post_min_quality_cell.i,
                                                        post_min_quality_cell.j,
                                                        post_corner_fail_cell.c,
                                                        post_corner_fail_cell.i,
                                                        post_corner_fail_cell.j,
                                                        post_corner_fail_corner,
                                                        accepted,
                                                        backtrack_last_reason.c_str());
      };

      if (post_tangle || post_corner_tangle) {
        const AleMinQualityCell current_fail_cell =
            post_tangle ? post_min_quality_cell : post_corner_fail_cell;
        if (post_tangle) {
          if (count_rejections) {
            ++rezone.gauss_tangle_rejection_count;
          }
          backtrack_last_reason = "post-rezone Gauss tangle";
        } else {
          if (count_rejections) {
            ++rezone.corner_tangle_rejection_count;
          }
          backtrack_last_reason = "post-rezone corner-J tangle";
        }
        if (count_rejections && !first_backtrack_fail_recorded) {
          first_backtrack_fail_cell = current_fail_cell;
          first_backtrack_fail_recorded = true;
          maybe_dump_lambda_sweep(first_backtrack_fail_cell);
        }
        if (count_rejections && post_corner_tangle &&
            !first_corner_backtrack_fail_recorded) {
          first_corner_backtrack_fail_cell = post_corner_fail_cell;
          first_corner_backtrack_fail_corner = post_corner_fail_corner;
          first_corner_backtrack_fail_recorded = true;
        }
        dump_backtrack_iter(false);
        return result;
      }

      AxisMarginResult axis_margin =
          compute_axis_margin_min(state, cfg.numerics.has_physical_rz_axis);
      bool axis_admissible = (axis_margin.min_margin > 0.0);
      rezone.final_axis_margin = axis_margin.min_margin;
      if (reduction != nullptr) {
        rezone.final_axis_margin = reduction->allreduce_min(rezone.final_axis_margin);
        axis_admissible =
            (reduction->allreduce_max(axis_admissible ? 0.0 : 1.0) < 0.5);
      }
      if (!axis_admissible) {
        backtrack_last_reason = "axis-cell analytic margin negative";
        if (count_rejections) {
          ++rezone.other_rejection_count;
          if (!first_backtrack_fail_recorded) {
            first_backtrack_fail_cell.c = axis_margin.min_j;
            first_backtrack_fail_cell.i = 0;
            first_backtrack_fail_cell.j = axis_margin.min_j;
            first_backtrack_fail_recorded = true;
          }
        }
        dump_backtrack_iter(false);
        return result;
      }

      if (axis_z_motion_winslow) {
        const AxisRadialBoundResult radial_bound =
            check_axis_radial_bound(state.x_r.data(),
                                    d_xr_old,
                                    nr,
                                    nz,
                                    cfg.numerics.ale.winslow_axis_kappa,
                                    reduction);
        if (!radial_bound.passed) {
          backtrack_last_reason = "axis-adjacent radial bound violated";
          if (count_rejections) {
            ++rezone.other_rejection_count;
            if (!first_backtrack_fail_recorded) {
              first_backtrack_fail_cell.c = radial_bound.fail_j;
              first_backtrack_fail_cell.i = 1;
              first_backtrack_fail_cell.j = radial_bound.fail_j;
              first_backtrack_fail_recorded = true;
            }
          }
          dump_backtrack_iter(false);
          return result;
        }
      }

      const bool capture_phase9_emergency =
          allow_phase9_emergency_capture && cfg.numerics.ale.remap_damage_gate_enabled &&
          !have_phase9_emergency;
      if (capture_phase9_emergency) {
        phase9_emergency_lambda = lambda;
        phase9_emergency_quality = trial_quality;
        phase9_emergency_axis_margin = rezone.final_axis_margin;
        phase9_emergency_post_cell = post_min_quality_cell;
        have_phase9_emergency = true;
      }

      detail::RemapDamageResult trial_damage;
      if (!check_remap_damage_gate(trial_damage)) {
        if (count_rejections) {
          ++rezone.other_rejection_count;
        }
        if (capture_phase9_emergency) {
          phase9_emergency_damage = trial_damage;
        }
        dump_backtrack_iter(false);
        return result;
      }
      if (cfg.numerics.ale.remap_damage_gate_enabled) {
        phase9_candidate_passed = true;
      }

      detail::RemapAdmissibilityResult remap_admissibility =
          detail::check_first_sweep_admissibility(d_vol_mid,
                                                  d_vol_old,
                                                  d_xr_old,
                                                  d_xz_old,
                                                  state.x_r.data(),
                                                  state.x_z.data(),
                                                  cw.begin,
                                                  cw.end,
                                                  nr,
                                                  nz,
                                                  state.step,
                                                  donor_sign_fixed);
      bool remap_admissible = remap_admissibility.admissible;
      if (reduction != nullptr) {
        remap_admissible =
            (reduction->allreduce_max(remap_admissible ? 0.0 : 1.0) < 0.5);
      }
      if (!remap_admissible) {
        backtrack_last_reason = "remap non-positive intermediate volume";
        if (count_rejections) {
          ++rezone.other_rejection_count;
          if (!first_backtrack_fail_recorded) {
            first_backtrack_fail_cell = remap_admissibility.first_fail_cell;
            first_backtrack_fail_recorded = true;
          }
        }
        dump_backtrack_iter(false);
        return result;
      }

      if (!check_predictive_acceptance_gate(state.x_r.data(), state.x_z.data())) {
        if (count_rejections) {
          ++rezone.other_rejection_count;
        }
        dump_backtrack_iter(false);
        return result;
      }

      result.admissible = true;
      result.damage = trial_damage;
      backtrack_last_reason = "accepted";
      dump_backtrack_iter(true);
      return result;
    };

    const auto admissible_for_search = [&](const double lambda) -> bool {
      const int attempt = rezone.backtrack_attempts;
      rezone.backtrack_attempts += 1;
      return mesh_admissible_at_lambda(
                 lambda, attempt, true, true, true)
          .admissible;
    };

    const auto search = detail::select_safe_backtrack_lambda(
        cfg.numerics.ale.safe_backtrack_min_exp,
        cfg.numerics.ale.safe_backtrack_binary_iters,
        admissible_for_search);
    if (search.status == AleStatus::PreRezoneInvalid) {
      out.status = AleStatus::PreRezoneInvalid;
      rezone.accepted_lambda = 0.0;
      backtrack_last_reason = "pre-rezone mesh already inadmissible";
      safe_backtrack_terminal_failure = true;
    } else if (search.status == AleStatus::NoLambdaAdmissible) {
      out.status = AleStatus::NoLambdaAdmissible;
      rezone.accepted_lambda = 0.0;
      backtrack_last_reason = "no positive admissible lambda above 2^-" +
                              std::to_string(
                                  cfg.numerics.ale.safe_backtrack_min_exp);
      safe_backtrack_terminal_failure = true;
    } else {
      const SafeBacktrackTrial final_trial =
          mesh_admissible_at_lambda(search.accepted_lambda,
                                    rezone.backtrack_attempts,
                                    false,
                                    false,
                                    false);
      if (final_trial.admissible) {
        accepted_axis_inflow_this_event = final_trial.damage.axis_inflow_this_event;
        accepted_backtrack = true;
        rezone.accepted_lambda = search.accepted_lambda;
        out.quality_min = final_trial.trial_quality;
        backtrack_last_reason = "accepted";
        detail::record_safe_backtrack_lambda(
            rezone.accepted_lambda, cfg.numerics.ale.safe_backtrack_min_exp);
      } else {
        out.status = AleStatus::NoLambdaAdmissible;
        rezone.accepted_lambda = 0.0;
        backtrack_last_reason = "safe backtrack accepted lambda failed revalidation";
        safe_backtrack_terminal_failure = true;
      }
    }
  }

  if (!safe_backtrack_terminal_failure && !accepted_backtrack &&
      cfg.numerics.ale.remap_damage_gate_enabled &&
      have_phase9_emergency && !phase9_candidate_passed) {
    detail::interpolate_rezone_candidate_kernel<<<blocks_nodes, 256>>>(
        state.x_r.data(),
        state.x_z.data(),
        d_xr_old,
        d_xz_old,
        d_xr_cand,
        d_xz_cand,
        phase9_emergency_lambda,
        n_nodes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(core::debug_kernel_sync());

    accepted_axis_inflow_this_event = phase9_emergency_damage.axis_inflow_this_event;
    accepted_backtrack = true;
    rezone.accepted_lambda = phase9_emergency_lambda;
    rezone.final_axis_margin = phase9_emergency_axis_margin;
    post_min_quality_cell = phase9_emergency_post_cell;
    out.quality_min = phase9_emergency_quality;
    backtrack_last_reason = "phase9 emergency bypass accepted";
    core::log_warning(
        "[ale-stats] Phase 9 emergency bypass: damage gate rejected all candidates; "
        "falling back to legacy criterion at lambda=" +
        std::to_string(phase9_emergency_lambda));
  }

  const bool corner_backtrack_exhausted =
      rezone.backtrack_attempts == detail::kBacktrackMaxAttempts &&
      rezone.corner_tangle_rejection_count == rezone.backtrack_attempts;
  const bool cascade_due_to_backtrack =
      !accepted_backtrack &&
      corner_backtrack_exhausted &&
      first_corner_backtrack_fail_recorded;
  const bool cascade_due_to_hydro_retry =
      cfg.numerics.hydro.cascade_on_hydro_retry_enabled &&
      accepted_backtrack &&
      retry_ctx.active &&
      retry_ctx.reason == AleDriverRetryContext::Reason::corner_j &&
      retry_ctx.post_ale_corner_balance_bad &&
      retry_ctx.first_cell >= 0 &&
      retry_ctx.first_corner >= 0;
  const bool should_try_corner_cascade =
      !safe_backtrack_terminal_failure &&
      cfg.numerics.ale.local_boundary_repair_enabled &&
      cfg.numerics.ale.corner_jacobian_post_tangle_enabled &&
      (cascade_due_to_backtrack || cascade_due_to_hydro_retry);
  if (should_try_corner_cascade) {
    const AleMinQualityCell target_cell =
        cascade_due_to_hydro_retry
            ? AleMinQualityCell{retry_ctx.first_cell, -1, -1}
            : first_corner_backtrack_fail_cell;
    const int target_corner =
        cascade_due_to_hydro_retry ? retry_ctx.first_corner
                                   : first_corner_backtrack_fail_corner;
    const char* gate_reason =
        cascade_due_to_hydro_retry ? "hydro_retry_bad_corner_balance"
                                   : "ale_backtrack_exhausted";
    diagnostics::EnergyTotals local_repair_pre_budget{};
    tenryu::coupling::EscapeValveEvent::CellState local_repair_pre_state{};
    bool have_local_repair_pre_budget = false;
    const int local_repair_cell_id = target_cell.c;
    if (observability != nullptr) {
      local_repair_pre_budget = detail::reduce_energy_totals_global(
          diagnostics::compute_energy_totals_2d(state), reduction);
      local_repair_pre_state =
          detail::capture_escape_valve_cell_state(state, local_repair_cell_id);
      have_local_repair_pre_budget = true;
    }
    LocalBoundaryRepairResult repair = try_local_boundary_repair(
        state, cfg, target_cell, target_corner, gate_reason);
    detail::log_local_boundary_repair_stats(repair, gate_reason);
    const std::string repair_reason(repair.reason);
    const bool multi_node_boundary_repair =
        repair_reason.rfind("multi-node", 0) == 0 ||
        repair.repaired_node_count > 1 ||
        std::string(repair.variable).rfind("multi_", 0) == 0;
    if (observability != nullptr && repair.fired) {
      observability->note_class_c_fire(
          multi_node_boundary_repair ? "ale_multi_node_boundary_repair"
                                     : "ale_local_boundary_repair");
      if (have_local_repair_pre_budget) {
        const diagnostics::EnergyTotals local_repair_post_budget =
            detail::reduce_energy_totals_global(
                diagnostics::compute_energy_totals_2d(state), reduction);
        const auto local_repair_post_state =
            detail::capture_escape_valve_cell_state(state, local_repair_cell_id);
        observability->note_escape_valve_event(
            detail::make_escape_valve_event(
                state,
                "post_step_ale",
                multi_node_boundary_repair ? "MultiNodeBoundaryRepair"
                                           : "LocalBoundaryRepair",
                multi_node_boundary_repair
                    ? "multi_node_boundary_repair_count"
                    : "local_boundary_repair_count",
                repair.reason,
                local_repair_cell_id,
                local_repair_pre_state,
                local_repair_post_state,
                0.0,
                0.0,
                0.0,
                local_repair_pre_budget,
                local_repair_post_budget));
      }
    }
    const bool multi_node_infeasible =
        repair_reason == "multi-node minimal ring infeasible" ||
        repair_reason == "multi-node expanded ring infeasible";
    if (!repair.applied && cfg.numerics.ale.emergency_cell_deactivation_enabled &&
        cfg.numerics.ale.multi_node_boundary_repair_enabled && multi_node_infeasible) {
      CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                            d_xr_old,
                            static_cast<std::size_t>(n_nodes) * sizeof(double),
                            cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                            d_xz_old,
                            static_cast<std::size_t>(n_nodes) * sizeof(double),
                            cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaDeviceSynchronize());
      diagnostics::EnergyTotals deactivation_pre_budget{};
      bool have_deactivation_pre_budget = false;
      if (observability != nullptr) {
        deactivation_pre_budget = detail::reduce_energy_totals_global(
            diagnostics::compute_energy_totals_2d(state), reduction);
        have_deactivation_pre_budget = true;
      }
      EmergencyCellDeactivationResult deactivation =
          try_emergency_cell_deactivation(state,
                                          cfg,
                                          target_cell.c,
                                          target_corner);
      detail::log_emergency_cell_deactivation_stats(deactivation, gate_reason);
      if (observability != nullptr && deactivation.applied) {
        observability->note_emergency_cell_deactivation();
        if (have_deactivation_pre_budget) {
          const diagnostics::EnergyTotals deactivation_post_budget =
              detail::reduce_energy_totals_global(
                  diagnostics::compute_energy_totals_2d(state), reduction);
          tenryu::coupling::EscapeValveEvent::CellState before_state;
          before_state.rho = deactivation.before_rho;
          before_state.p = deactivation.before_p;
          before_state.Te = deactivation.before_Te;
          before_state.Ti = deactivation.before_Ti;
          tenryu::coupling::EscapeValveEvent::CellState after_state;
          after_state.rho = deactivation.after_rho;
          after_state.p = deactivation.after_p;
          after_state.Te = deactivation.after_Te;
          after_state.Ti = deactivation.after_Ti;
          auto event = detail::make_escape_valve_event(
              state,
              "post_step_ale",
              "EmergencyCellDeactivation",
              "emergency_cell_deactivation_count",
              deactivation.reason,
              deactivation.target_cell_c,
              before_state,
              after_state,
              deactivation.mass_floor_delta,
              deactivation.momentum_delta_r,
              deactivation.momentum_delta_z,
              deactivation_pre_budget,
              deactivation_post_budget);
          event.energy_delta = deactivation.energy_delta;
          observability->note_escape_valve_event(event);
        }
      }
      if (deactivation.applied) {
        out.mass_floor_delta += deactivation.mass_floor_delta;
        out.E_floor_injected += deactivation.E_floor_injected;
        out.applied = false;
        out.quality_rollback = false;
        state.holo_ale_invalidated = true;
        rezone.accepted_lambda = 0.0;
        backtrack_last_reason = "emergency cell deactivation accepted";
        AxisMarginResult axis_margin =
            compute_axis_margin_min(state, cfg.numerics.has_physical_rz_axis);
        rezone.final_axis_margin = axis_margin.min_margin;
        if (reduction != nullptr) {
          rezone.final_axis_margin =
              reduction->allreduce_min(rezone.final_axis_margin);
        }
        log_predictive_rejects_if_any();
        detail::log_ale_backtrack_stats(rezone.accepted_lambda,
                                        rezone.backtrack_attempts,
                                        first_backtrack_fail_cell,
                                        rezone.gauss_tangle_rejection_count,
                                        rezone.corner_tangle_rejection_count,
                                        rezone.other_rejection_count);
        detail::log_ale_rezone_stats(rezone.iterations,
                                     rezone.stats,
                                     rezone.min_quality_cell_pre,
                                     post_min_quality_cell,
                                     rezone.final_axis_margin,
                                     1,
                                     backtrack_last_reason);
        const double accepted_axis_inflow_max =
            accepted_axis_inflow_this_event.empty()
                ? 0.0
                : *std::max_element(accepted_axis_inflow_this_event.begin(),
                                    accepted_axis_inflow_this_event.end());
        diagnostics::mesh_diag::dump_ale_rezone_end(
            state,
            cfg,
            0,
            backtrack_last_reason.c_str(),
            rezone.final_axis_margin,
            !accepted_axis_inflow_this_event.empty(),
            static_cast<int>(accepted_axis_inflow_this_event.size()),
            accepted_axis_inflow_max,
            rezone.iterations,
            rezone.min_quality_cell_pre.c,
            rezone.min_quality_cell_pre.i,
            rezone.min_quality_cell_pre.j,
            deactivation.target_cell.c,
            deactivation.target_cell.i,
            deactivation.target_cell.j);
        core::log_warning(
            "ALE emergency cell deactivation accepted after local boundary repair "
            "infeasibility; mesh coordinates were rolled back before deactivation.");
        log_axis_uR_if_nonzero();
        cleanup();
        return out;
      }
    }
    if (repair.applied) {
      post_tangle = false;
      post_min_quality_cell = AleMinQualityCell{};
      post_corner_tangle = false;
      post_corner_fail_cell = AleMinQualityCell{};
      post_corner_fail_corner = -1;
      post_min_corner_j = 1.0;
      double trial_quality =
          compute_min_quality(state, cfg, &post_tangle, &post_min_quality_cell);
      if (reduction != nullptr) {
        trial_quality = reduction->allreduce_min(trial_quality);
        post_tangle =
            (reduction->allreduce_max(post_tangle ? 1.0 : 0.0) > 0.5);
      }
      post_corner_tangle = compute_corner_post_tangle(state,
                                                      cfg,
                                                      &post_corner_fail_cell,
                                                      &post_min_corner_j,
                                                      &post_corner_fail_corner);
      if (reduction != nullptr) {
        post_min_corner_j = reduction->allreduce_min(post_min_corner_j);
        post_corner_tangle =
            (reduction->allreduce_max(post_corner_tangle ? 1.0 : 0.0) > 0.5);
      }
      if (post_tangle || post_corner_tangle) {
        backtrack_last_reason =
            post_tangle ? "post-repair Gauss tangle" : "post-repair corner-J tangle";
      } else {
        AxisMarginResult axis_margin =
            compute_axis_margin_min(state, cfg.numerics.has_physical_rz_axis);
        bool axis_admissible = (axis_margin.min_margin > 0.0);
        rezone.final_axis_margin = axis_margin.min_margin;
        if (reduction != nullptr) {
          rezone.final_axis_margin =
              reduction->allreduce_min(rezone.final_axis_margin);
          axis_admissible =
              (reduction->allreduce_max(axis_admissible ? 0.0 : 1.0) < 0.5);
        }
        if (!axis_admissible) {
          backtrack_last_reason = "post-repair axis-cell analytic margin negative";
        } else {
          bool radial_admissible = true;
          if (axis_z_motion_winslow) {
            const AxisRadialBoundResult radial_bound =
                check_axis_radial_bound(state.x_r.data(),
                                        d_xr_old,
                                        nr,
                                        nz,
                                        cfg.numerics.ale.winslow_axis_kappa,
                                        reduction);
            radial_admissible = radial_bound.passed;
            if (!radial_admissible) {
              backtrack_last_reason = "post-repair axis-adjacent radial bound violated";
            }
          }
          if (radial_admissible) {
            detail::RemapDamageResult trial_damage;
            if (!check_remap_damage_gate(trial_damage)) {
              ++rezone.other_rejection_count;
            } else {
              detail::RemapAdmissibilityResult remap_admissibility =
                  detail::check_first_sweep_admissibility(d_vol_mid,
                                                          d_vol_old,
                                                          d_xr_old,
                                                          d_xz_old,
                                                          state.x_r.data(),
                                                          state.x_z.data(),
                                                          cw.begin,
                                                          cw.end,
                                                          nr,
                                                          nz,
                                                          state.step,
                                                          donor_sign_fixed);
              bool remap_admissible = remap_admissibility.admissible;
              if (reduction != nullptr) {
                remap_admissible =
                    (reduction->allreduce_max(remap_admissible ? 0.0 : 1.0) < 0.5);
              }
              if (!remap_admissible) {
                backtrack_last_reason = "post-repair remap non-positive intermediate volume";
                ++rezone.other_rejection_count;
                if (!first_backtrack_fail_recorded) {
                  first_backtrack_fail_cell = remap_admissibility.first_fail_cell;
                  first_backtrack_fail_recorded = true;
                }
              } else if (!check_predictive_acceptance_gate(state.x_r.data(),
                                                           state.x_z.data())) {
                ++rezone.other_rejection_count;
              } else {
                accepted_axis_inflow_this_event = trial_damage.axis_inflow_this_event;
                accepted_backtrack = true;
                rezone.accepted_lambda = 1.0;
                out.quality_min = trial_quality;
                backtrack_last_reason = "local boundary repair accepted";
              }
            }
          }
        }
      }
    }
  }

  if (!accepted_backtrack) {
    bool sliding_axis_accepted = false;
    double accepted_omega = 0.0;
    if (!safe_backtrack_terminal_failure) {
      const int axis_nodes = nz + 1;
      const int blocks_axis = (axis_nodes + 255) / 256;
      const double dz_init =
          std::fabs((cfg.mesh.z_max - cfg.mesh.z_min) / static_cast<double>(cfg.mesh.nz));

      std::vector<double> z_axis_old_h(static_cast<std::size_t>(axis_nodes), 0.0);
      std::vector<double> z_axis_target_h(static_cast<std::size_t>(axis_nodes), 0.0);
      std::vector<std::uint8_t> axis_flags_h(static_cast<std::size_t>(axis_nodes),
                                             mesh::NODE_NONE);
      TENRYU_ASSERT(static_cast<int>(state.mesh.topo.node_flags.size()) == n_nodes,
                    "ALE sliding-axis fallback requires node_flags size to match n_nodes");
      bool has_full_axis = cfg.numerics.has_physical_rz_axis;
      for (int j = 0; j <= nz; ++j) {
        axis_flags_h[static_cast<std::size_t>(j)] =
            state.mesh.topo.node_flags[static_cast<std::size_t>(j)];
        if ((axis_flags_h[static_cast<std::size_t>(j)] & mesh::NODE_AXIS) == 0u) {
          has_full_axis = false;
        }
      }
      if (has_full_axis) {
        CUDA_CHECK(cudaMemcpy(z_axis_old_h.data(),
                              d_xz_old,
                              static_cast<std::size_t>(axis_nodes) * sizeof(double),
                              cudaMemcpyDeviceToHost));

        double* d_z_axis_target = nullptr;
        std::uint8_t* d_node_flags = nullptr;
        d_z_axis_target = static_cast<double*>(
            core::device_scratch_acquire("ale_driver:ale_req:d_z_axis_target",
                                         static_cast<std::size_t>(axis_nodes) * sizeof(double)));
        d_node_flags = static_cast<std::uint8_t*>(
            core::device_scratch_acquire("ale_driver:ale_req:d_node_flags",
                                         static_cast<std::size_t>(n_nodes) * sizeof(std::uint8_t)));
        CUDA_CHECK(cudaMemcpy(d_node_flags,
                              state.mesh.topo.node_flags.data(),
                              static_cast<std::size_t>(n_nodes) * sizeof(std::uint8_t),
                              cudaMemcpyHostToDevice));

        const double omega_schedule[4] = {1.0, 0.5, 0.25, 0.125};
        for (const double omega : omega_schedule) {
          detail::axis_spine_target_kernel<<<blocks_axis, 256>>>(d_z_axis_target,
                                                                 d_xz_old,
                                                                 d_xz_cand,
                                                                 d_node_flags,
                                                                 nz,
                                                                 omega,
                                                                 cfg.numerics.has_physical_rz_axis);
          CUDA_CHECK(cudaGetLastError());
          CUDA_CHECK(core::debug_kernel_sync());
          CUDA_CHECK(cudaMemcpy(z_axis_target_h.data(),
                                d_z_axis_target,
                                static_cast<std::size_t>(axis_nodes) * sizeof(double),
                                cudaMemcpyDeviceToHost));
          detail::apply_axis_spine_constraints(z_axis_target_h,
                                               z_axis_old_h,
                                               axis_flags_h,
                                               dz_init);
          CUDA_CHECK(cudaMemcpy(d_z_axis_target,
                                z_axis_target_h.data(),
                                static_cast<std::size_t>(axis_nodes) * sizeof(double),
                                cudaMemcpyHostToDevice));

          CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                                d_xr_cand,
                                static_cast<std::size_t>(n_nodes) * sizeof(double),
                                cudaMemcpyDeviceToDevice));
          CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                                d_xz_cand,
                                static_cast<std::size_t>(n_nodes) * sizeof(double),
                                cudaMemcpyDeviceToDevice));
          detail::apply_axis_spine_kernel<<<blocks_axis, 256>>>(
              state.x_r.data(), state.x_z.data(), d_z_axis_target, nz,
              cfg.numerics.has_physical_rz_axis);
          CUDA_CHECK(cudaGetLastError());
          CUDA_CHECK(core::debug_kernel_sync());

          post_tangle = false;
          post_min_quality_cell = AleMinQualityCell{};
          post_corner_tangle = false;
          post_corner_fail_cell = AleMinQualityCell{};
          post_corner_fail_corner = -1;
          post_min_corner_j = 1.0;
          double trial_quality =
              compute_min_quality(state, cfg, &post_tangle, &post_min_quality_cell);
          if (reduction != nullptr) {
            trial_quality = reduction->allreduce_min(trial_quality);
            post_tangle = (reduction->allreduce_max(post_tangle ? 1.0 : 0.0) > 0.5);
          }
          if (cfg.numerics.ale.corner_jacobian_post_tangle_enabled) {
            post_corner_tangle = compute_corner_post_tangle(state,
                                                            cfg,
                                                            &post_corner_fail_cell,
                                                            &post_min_corner_j,
                                                            &post_corner_fail_corner);
            if (reduction != nullptr) {
              post_min_corner_j = reduction->allreduce_min(post_min_corner_j);
              post_corner_tangle =
                  (reduction->allreduce_max(post_corner_tangle ? 1.0 : 0.0) > 0.5);
            }
          }
          if (post_tangle || post_corner_tangle) {
            continue;
          }

          AxisMarginResult axis_margin =
              compute_axis_margin_min(state, cfg.numerics.has_physical_rz_axis);
          bool axis_admissible = (axis_margin.min_margin > 0.0);
          rezone.final_axis_margin = axis_margin.min_margin;
          if (reduction != nullptr) {
            rezone.final_axis_margin =
                reduction->allreduce_min(rezone.final_axis_margin);
            axis_admissible =
                (reduction->allreduce_max(axis_admissible ? 0.0 : 1.0) < 0.5);
          }
          if (!axis_admissible) {
            continue;
          }

          if (axis_z_motion_winslow) {
            const AxisRadialBoundResult radial_bound =
                check_axis_radial_bound(state.x_r.data(),
                                        d_xr_old,
                                        nr,
                                        nz,
                                        cfg.numerics.ale.winslow_axis_kappa,
                                        reduction);
            if (!radial_bound.passed) {
              continue;
            }
          }

          detail::RemapDamageResult trial_damage;
          if (!check_remap_damage_gate(trial_damage)) {
            continue;
          }

          detail::RemapAdmissibilityResult remap_admissibility =
              detail::check_first_sweep_admissibility(d_vol_mid,
                                                      d_vol_old,
                                                      d_xr_old,
                                                      d_xz_old,
                                                      state.x_r.data(),
                                                      state.x_z.data(),
                                                      cw.begin,
                                                      cw.end,
                                                      nr,
                                                      nz,
                                                      state.step,
                                                      donor_sign_fixed);
          bool remap_admissible = remap_admissibility.admissible;
          if (reduction != nullptr) {
            remap_admissible =
                (reduction->allreduce_max(remap_admissible ? 0.0 : 1.0) < 0.5);
          }
          if (!remap_admissible) {
            continue;
          }

          if (!check_predictive_acceptance_gate(state.x_r.data(), state.x_z.data())) {
            continue;
          }

          accepted_axis_inflow_this_event = trial_damage.axis_inflow_this_event;
          sliding_axis_accepted = true;
          accepted_omega = omega;
          rezone.accepted_lambda = 1.0;
          out.quality_min = trial_quality;
          break;
        }

      }
    }

    if (sliding_axis_accepted) {
      accepted_backtrack = true;
      backtrack_last_reason = "sliding-axis fallback accepted";
      core::log_warning("ALE sliding-axis fallback accepted (omega=" +
                        std::to_string(accepted_omega) + ")");
    }

    if (!accepted_backtrack) {
      rezone.accepted_lambda = 0.0;
      AxisMarginResult axis_margin =
          compute_axis_margin_min(state, cfg.numerics.has_physical_rz_axis);
      rezone.final_axis_margin = axis_margin.min_margin;
      if (reduction != nullptr) {
        rezone.final_axis_margin =
            reduction->allreduce_min(rezone.final_axis_margin);
      }
      log_predictive_rejects_if_any();
      detail::log_ale_backtrack_stats(rezone.accepted_lambda,
                                      rezone.backtrack_attempts,
                                      first_backtrack_fail_cell,
                                      rezone.gauss_tangle_rejection_count,
                                      rezone.corner_tangle_rejection_count,
                                      rezone.other_rejection_count);
      detail::log_ale_rezone_stats(rezone.iterations,
                                   rezone.stats,
                                   rezone.min_quality_cell_pre,
                                   post_min_quality_cell,
                                   rezone.final_axis_margin,
                                   1,
                                   backtrack_last_reason);
      const double accepted_axis_inflow_max =
          accepted_axis_inflow_this_event.empty()
              ? 0.0
              : *std::max_element(accepted_axis_inflow_this_event.begin(),
                                  accepted_axis_inflow_this_event.end());
      diagnostics::mesh_diag::dump_ale_rezone_end(
          state,
          cfg,
          1,
          backtrack_last_reason.c_str(),
          rezone.final_axis_margin,
          !accepted_axis_inflow_this_event.empty(),
          static_cast<int>(accepted_axis_inflow_this_event.size()),
          accepted_axis_inflow_max,
          rezone.iterations,
          rezone.min_quality_cell_pre.c,
          rezone.min_quality_cell_pre.i,
          rezone.min_quality_cell_pre.j,
          post_min_quality_cell.c,
          post_min_quality_cell.i,
          post_min_quality_cell.j);
      CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                            d_xr_old,
                            static_cast<std::size_t>(n_nodes) * sizeof(double),
                            cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                            d_xz_old,
                            static_cast<std::size_t>(n_nodes) * sizeof(double),
                            cudaMemcpyDeviceToDevice));
      out.applied = false;
      out.quality_rollback = (backtrack_last_reason == "post-rezone Gauss tangle" ||
                              backtrack_last_reason == "post-rezone corner-J tangle" ||
                              backtrack_last_reason ==
                                  "pre-rezone mesh already inadmissible");
      if (out.quality_rollback) {
        core::log_warning(
            "ALE rezone produced inadmissible mesh; rolling back to pre-rezone "
            "coordinates.");
      } else if (backtrack_last_reason == "no positive admissible lambda above 2^-" +
                                             std::to_string(
                                                 cfg.numerics.ale.safe_backtrack_min_exp)) {
        core::log_warning(
            "ALE safe backtracking found no positive admissible displacement above "
            "configured lambda_min; rolled back to pre-rezone mesh.");
      } else if (backtrack_last_reason ==
                 "safe backtrack accepted lambda failed revalidation") {
        core::log_warning(
            "ALE safe backtracking accepted lambda failed revalidation; rolled back "
            "to pre-rezone mesh.");
      } else if (backtrack_last_reason == "axis-cell analytic margin negative") {
        core::log_warning(
            "ALE analytic axis-cell check rejected all rezone displacements; rolled back "
            "to pre-rezone mesh.");
      } else if (backtrack_last_reason == "axis-adjacent radial bound violated") {
        core::log_warning(
            "ALE axis-adjacent radial bound rejected all rezone displacements; rolled back "
            "to pre-rezone mesh.");
      } else {
        core::log_warning(
            "ALE remap preflight rejected all rezone displacements; rolled back to "
            "pre-rezone mesh.");
      }
      log_axis_uR_if_nonzero();
      cleanup();
      return out;
    }
  }

  log_predictive_rejects_if_any();
  detail::log_ale_backtrack_stats(rezone.accepted_lambda,
                                  rezone.backtrack_attempts,
                                  first_backtrack_fail_cell,
                                  rezone.gauss_tangle_rejection_count,
                                  rezone.corner_tangle_rejection_count,
                                  rezone.other_rejection_count);
  const double accepted_axis_inflow_max =
      accepted_axis_inflow_this_event.empty()
          ? 0.0
          : *std::max_element(accepted_axis_inflow_this_event.begin(),
                              accepted_axis_inflow_this_event.end());
  diagnostics::mesh_diag::dump_ale_rezone_end(state,
                                              cfg,
                                              0,
                                              backtrack_last_reason.c_str(),
                                              rezone.final_axis_margin,
                                              !accepted_axis_inflow_this_event.empty(),
                                              static_cast<int>(
                                                  accepted_axis_inflow_this_event.size()),
                                              accepted_axis_inflow_max,
                                              rezone.iterations,
                                              rezone.min_quality_cell_pre.c,
                                              rezone.min_quality_cell_pre.i,
                                              rezone.min_quality_cell_pre.j,
                                              post_min_quality_cell.c,
                                              post_min_quality_cell.i,
                                              post_min_quality_cell.j);

  out.applied = true;
  detail::enforce_corner_cell_aspect_floor(state.x_r.data(), state.x_z.data(), state, cfg);

  const tenryu::mesh::MeshGeometryResult post_rezone_geometry =
      detail::recompute_geometry_for_ale(state, cfg, observability);
  if (!post_rezone_geometry.admissible) {
    CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                          d_xr_old,
                          static_cast<std::size_t>(n_nodes) * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                          d_xz_old,
                          static_cast<std::size_t>(n_nodes) * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    (void)detail::recompute_geometry_for_ale(state, cfg, observability);
    state.vol = state.mesh.cell_vol;
    out.applied = false;
    out.quality_rollback = true;
    core::log_warning(
        "ALE rezone geometry refresh returned typed soft failure; rolled back "
        "to pre-rezone mesh.");
    cleanup();
    return out;
  }
  state.vol = state.mesh.cell_vol;
  state.holo_ale_invalidated = true;
  csr_optionb_canonicalize_corner_mass_basis(state, cfg);
  record_estep_trace_boundary(state,
                              cfg,
                              part,
                              reduction,
                              "post_rezone_pre_remap",
                              dt_hydro_used,
                              state.t + dt_hydro_used);
  if (cfg.numerics.diagnostics.mesh_attribution.enabled &&
      cfg.numerics.diagnostics.mesh_attribution.record_node_displacements) {
    auto& attr = tenryu::diagnostics::mesh_attribution::global_workspace();
    if (attr.active()) {
      attr.record_direct_displacement(
          tenryu::diagnostics::mesh_attribution::MeshDeformSource::ALERezone,
          d_xr_old,
          d_xz_old,
          state.x_r.data(),
          state.x_z.data(),
          n_nodes);
    }
  }

  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  const int n_groups = std::max(cfg.radiation.groups, 1);
  const bool per_material_conservation_enabled =
      cfg.numerics.materials.per_material_conservation_enabled;
  const std::size_t expected_rad_size =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  const std::size_t n_cell_mat =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_mat);
  const bool remap_rad_E = (state.rad_E.size() == expected_rad_size);
  TENRYU_ASSERT(n_mat > 0, "ALE requires at least one material");
  TENRYU_ASSERT(static_cast<int>(state.volFrac.size()) == n_cells * n_mat,
                "ALE requires volFrac size to match n_cells*n_materials");
  if (per_material_conservation_enabled) {
    TENRYU_ASSERT(state.mass_per_material.size() == n_cell_mat,
                  "ALE per-material mode requires mass_per_material size n_cells*n_materials");
    TENRYU_ASSERT(state.Ee_per_material.size() == n_cell_mat,
                  "ALE per-material mode requires Ee_per_material size n_cells*n_materials");
    TENRYU_ASSERT(state.Ei_per_material.size() == n_cell_mat,
                  "ALE per-material mode requires Ei_per_material size n_cells*n_materials");
  }
  if (!remap_rad_E && !state.rad_E.empty()) {
    core::log_warning("ALE: skipping rad_E remap because rad_E size does not match n_cells*n_groups "
                      "(rad_E=" + std::to_string(state.rad_E.size()) +
                      ", expected=" + std::to_string(expected_rad_size) + ").");
  }

  d_vr_cell = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:ale_req:d_vr_cell",
                                   static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_vz_cell = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:ale_req:d_vz_cell",
                                   static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_tmp = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:ale_req:d_tmp",
                                   static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_mom_r = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:ale_req:d_mom_r",
                                   static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_mom_z = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:ale_req:d_mom_z",
                                   static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_e_e = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:ale_req:d_e_e",
                                   static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_e_i = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:ale_req:d_e_i",
                                   static_cast<std::size_t>(n_cells) * sizeof(double)));
  if (per_material_conservation_enabled) {
    d_eta_rho_m = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:ale_req:d_eta_rho_m",
                                     n_cell_mat * sizeof(double)));
    d_eta_rho_ee_m = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:ale_req:d_eta_rho_ee_m",
                                     n_cell_mat * sizeof(double)));
    d_eta_rho_ei_m = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:ale_req:d_eta_rho_ei_m",
                                     n_cell_mat * sizeof(double)));
    d_eta_rho_vr_m = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:ale_req:d_eta_rho_vr_m",
                                     n_cell_mat * sizeof(double)));
    d_eta_rho_vz_m = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:ale_req:d_eta_rho_vz_m",
                                     n_cell_mat * sizeof(double)));
    d_eta_rho_m_old = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:ale_req:d_eta_rho_m_old",
                                     n_cell_mat * sizeof(double)));
    d_eta_rho_ee_m_old = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:ale_req:d_eta_rho_ee_m_old",
                                     n_cell_mat * sizeof(double)));
    d_eta_rho_ei_m_old = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:ale_req:d_eta_rho_ei_m_old",
                                     n_cell_mat * sizeof(double)));
    d_eta_rho_vr_m_old = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:ale_req:d_eta_rho_vr_m_old",
                                     n_cell_mat * sizeof(double)));
    d_eta_rho_vz_m_old = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:ale_req:d_eta_rho_vz_m_old",
                                     n_cell_mat * sizeof(double)));
    d_volfrac_mat_old = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:ale_req:d_volfrac_mat_old",
                                     n_cell_mat * sizeof(double)));
  }
  if (ke_closure_enabled) {
    d_ke_remap = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:ale_req:d_ke_remap",
                                     static_cast<std::size_t>(n_cells) * sizeof(double)));
    if (ke_closure_redistribute_floor) {
      d_ke_closure_deficit = static_cast<double*>(
          core::device_scratch_acquire("ale_driver:ale_req:d_ke_closure_deficit",
                                       static_cast<std::size_t>(n_cells) * sizeof(double)));
      d_ke_closure_capacity = static_cast<double*>(
          core::device_scratch_acquire("ale_driver:ale_req:d_ke_closure_capacity",
                                       static_cast<std::size_t>(n_cells) * sizeof(double)));
    }
  }
  d_volfrac_mat = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:ale_req:d_volfrac_mat",
                                   n_cell_mat * sizeof(double)));
  if (remap_rad_E) {
    d_rad_e_group = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:ale_req:d_rad_e_group",
                                     static_cast<std::size_t>(n_cells) * sizeof(double)));
  }
  d_dm_floor = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:ale_req:d_dm_floor", sizeof(double)));
  if (per_material_conservation_enabled) {
    d_dm_floor_per_material = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:ale_req:d_dm_floor_per_material",
                                     static_cast<std::size_t>(n_mat) * sizeof(double)));
    d_repair_count = static_cast<int*>(
        core::device_scratch_acquire("ale_driver:ale_req:d_repair_count", sizeof(int)));
  }
  d_e_floor = static_cast<double*>(
      core::device_scratch_acquire("ale_driver:ale_req:d_e_floor", sizeof(double)));
  const double zero_d = 0.0;
  CUDA_CHECK(cudaMemcpy(d_dm_floor, &zero_d, sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_e_floor, &zero_d, sizeof(double), cudaMemcpyHostToDevice));
  if (per_material_conservation_enabled) {
    const int zero_i = 0;
    CUDA_CHECK(cudaMemset(d_dm_floor_per_material,
                          0,
                          static_cast<std::size_t>(n_mat) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_repair_count, &zero_i, sizeof(int), cudaMemcpyHostToDevice));
  }
  if (closure_audit_enabled) {
    d_audit_cell = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:ale_req:d_audit_cell",
                                     static_cast<std::size_t>(n_cells) *
                                         static_cast<std::size_t>(detail::kAuditDepositCols) *
                                         sizeof(double)));
    d_audit_node = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:ale_req:d_audit_node",
                                     static_cast<std::size_t>(n_nodes) * sizeof(double)));
    d_audit_node_mass = static_cast<double*>(
        core::device_scratch_acquire("ale_driver:ale_req:d_audit_node_mass",
                                     static_cast<std::size_t>(n_nodes) * sizeof(double)));
  }

  if (ke_closure_enabled) {
    detail::compute_cell_velocity_from_corner_momentum_kernel<<<blocks_cells, 256>>>(
        d_vr_cell,
        d_vz_cell,
        state.corner_mass.data(),
        state.v_r.data(),
        state.v_z.data(),
        d_cell_nverts,
        nr,
        nz);
  } else {
    compute_cell_velocity_from_nodes_kernel<<<blocks_cells, 256>>>(
        d_vr_cell,
        d_vz_cell,
        state.v_r.data(),
        state.v_z.data(),
        d_cell_nverts,
        nr,
        nz);
  }
  if (per_material_conservation_enabled) {
    detail::pack_conserved_kernel_per_material<<<cw.blocks(), 256>>>(
        d_eta_rho_m,
        d_eta_rho_ee_m,
        d_eta_rho_ei_m,
        d_eta_rho_vr_m,
        d_eta_rho_vz_m,
        d_volfrac_mat,
        state.mass_per_material.data(),
        state.Ee_per_material.data(),
        state.Ei_per_material.data(),
        state.volFrac.data(),
        d_vol_old,
        d_vr_cell,
        d_vz_cell,
        cw.begin,
        cw.end,
        n_cells,
        n_mat,
        d_dm_floor,
        d_e_floor,
        d_repair_count);
    state.dispatch_counters.per_material_kernel_call_count.fetch_add(
        1, std::memory_order_relaxed);
  } else {
    detail::pack_conserved_kernel<<<cw.blocks(), 256>>>(d_mom_r,
                                                          d_mom_z,
                                                          d_e_e,
                                                          d_e_i,
                                                          d_volfrac_mat,
                                                          state.rho.data(),
                                                          state.ee.data(),
                                                          state.ei.data(),
                                                          d_vr_cell,
                                                          d_vz_cell,
                                                          state.volFrac.data(),
                                                          cw.begin,
                                                          cw.end,
                                                          n_cells,
                                                          n_mat);
  }
  CUDA_CHECK(cudaGetLastError());
  if (per_material_conservation_enabled) {
    CUDA_CHECK(cudaMemcpy(d_eta_rho_m_old,
                          d_eta_rho_m,
                          n_cell_mat * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_eta_rho_ee_m_old,
                          d_eta_rho_ee_m,
                          n_cell_mat * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_eta_rho_ei_m_old,
                          d_eta_rho_ei_m,
                          n_cell_mat * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_eta_rho_vr_m_old,
                          d_eta_rho_vr_m,
                          n_cell_mat * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_eta_rho_vz_m_old,
                          d_eta_rho_vz_m,
                          n_cell_mat * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_volfrac_mat_old,
                          d_volfrac_mat,
                          n_cell_mat * sizeof(double),
                          cudaMemcpyDeviceToDevice));
#ifndef NDEBUG
    per_material_old_snapshot_step = state.step;
#endif
  }
  if (ke_closure_enabled) {
    detail::compute_corner_kinetic_density_kernel<<<blocks_cells, 256>>>(
        d_ke_remap,
        state.corner_mass.data(),
        d_vol_old,
        state.v_r.data(),
        state.v_z.data(),
        d_cell_nverts,
        nr,
        nz);
    CUDA_CHECK(cudaGetLastError());
  }
  if (closure_audit_enabled) {
    out.closure_audit.K0_budget = closure_audit_pre_budget.E_kin;
    CUDA_CHECK(cudaMemset(d_audit_node_mass,
                          0,
                          static_cast<std::size_t>(n_nodes) * sizeof(double)));
    detail::audit_pre_ale_cell_kernel<<<cw.blocks(), 256>>>(
        d_audit_cell,
        d_audit_node_mass,
        state.corner_mass.data(),
        state.mass.data(),
        state.ee.data(),
        state.ei.data(),
        state.v_r.data(),
        state.v_z.data(),
        d_cell_nverts,
        cw.begin,
        cw.end,
        nr,
        nz);
    CUDA_CHECK(cudaGetLastError());
    detail::audit_node_kinetic_kernel<<<nw.blocks(), 256>>>(
        d_audit_node,
        d_audit_node_mass,
        state.v_r.data(),
        state.v_z.data(),
        nw.begin,
        nw.end,
        n_nodes);
    CUDA_CHECK(cudaGetLastError());
    detail::audit_scalar_volume_kernel<<<cw.blocks(), 256>>>(
        d_audit_cell + static_cast<std::size_t>(n_cells) * detail::kAuditPreCols,
        d_ke_remap,
        d_vol_old,
        cw.begin,
        cw.end,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(core::debug_kernel_sync());
    const auto pre_sums =
        detail::reduce_device_column_sums(d_audit_cell, n_cells, detail::kAuditPreCols);
    out.closure_audit.K0_cellcorner = pre_sums[0];
    out.closure_audit.I0 = pre_sums[1];
    out.closure_audit.K0_node_from_corner =
        detail::reduce_device_sum(d_audit_node, n_nodes);
    out.closure_audit.K0_scalar_total = detail::reduce_device_sum(
        d_audit_cell + static_cast<std::size_t>(n_cells) * detail::kAuditPreCols,
        n_cells);
  }

  const auto rollback_after_remap_failure = [&]() {
    detail::log_ale_rezone_stats(rezone.iterations,
                                 rezone.stats,
                                 rezone.min_quality_cell_pre,
                                 post_min_quality_cell,
                                 rezone.final_axis_margin,
                                 1,
                                 "remap non-positive intermediate volume");
    CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                          d_xr_old,
                          static_cast<std::size_t>(n_nodes) * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                          d_xz_old,
                          static_cast<std::size_t>(n_nodes) * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    const tenryu::mesh::MeshGeometryResult rollback_geometry =
        detail::recompute_geometry_for_ale(state, cfg, observability);
    if (rollback_geometry.admissible) {
      state.vol = state.mesh.cell_vol;
    }
    out.applied = false;
    core::log_warning(
        "ALE remap aborted (non-positive intermediate volume); rolled back to pre-rezone "
        "mesh.");
    log_axis_uR_if_nonzero();
  };

  const bool use_ms2_remap = (cfg.numerics.ale.remap_scheme == "ms2_moments");
  const RemapMs2Limiter ms2_limiter =
      (cfg.numerics.ale.remap_ms2_limiter == "barth_jespersen")
          ? RemapMs2Limiter::BarthJespersen
          : RemapMs2Limiter::VanLeer;
  const auto launch_scalar_remap = [&](double* field) {
    if (use_ms2_remap) {
      return launch_remap_strang_ms2(field,
                                     d_tmp,
                                     d_vol_mid,
                                     d_vol_old,
                                     state.vol.data(),
                                     d_xr_old,
                                     d_xz_old,
                                     state.x_r.data(),
                                     state.x_z.data(),
                                     nr,
                                     nz,
                                     state.step,
                                     ms2_limiter,
                                     donor_sign_fixed);
    }
    return launch_remap_strang(field,
                               d_tmp,
                               d_vol_mid,
                               d_vol_old,
                               state.vol.data(),
                               d_xr_old,
                               d_xz_old,
                               state.x_r.data(),
                               state.x_z.data(),
                               nr,
                               nz,
                               state.step,
                               donor_sign_fixed);
  };

  const bool cf6_plic_rho_remap_requested =
      !per_material_conservation_enabled &&
      cfg.numerics.plic.rho_material_aware_donor &&
      plic::plic_runtime_active(state, cfg);
  bool rho_remapped = false;
  bool plic_volfrac_remap_used = false;
  plic::PlicRemapStatus plic_status;
  const auto publish_plic_status = [&](const plic::PlicRemapStatus& status) {
    out.plic_remap_used = plic_volfrac_remap_used;
    out.plic_remap_fallback_engaged =
        state.plic_remap_sticky_fallback || status.fallback_engaged;
    out.plic_drift_triggered = status.drift_triggered;
    out.plic_cf6_density_remap_used = status.cf6_density_remap_used;
    out.plic_interface_cells = status.interface_cells;
    out.plic_reconstruction_attempts = status.reconstruction_attempts;
    out.plic_reconstruction_successes = status.reconstruction_successes;
    out.plic_axis_exempt_cells = status.axis_exempt_cells;
    out.plic_class_d_events = status.class_d_events;
    out.plic_repair_events = status.repair_events;
    out.plic_max_volume_fraction_residual =
        status.max_volume_fraction_residual;
    out.plic_max_interface_centroid_drift_relative =
        status.max_interface_centroid_drift_relative;
    out.plic_max_swept_fraction = status.max_swept_fraction;
    if (status.fallback_engaged &&
        (status.class_d_events > 0 || status.repair_events > 0) &&
        observability != nullptr) {
      observability->plic_remap_fallback_engaged = true;
      observability->class_d_runtime_fires_matrix[2][1] += 1;
      core::log_warning(
          "[plic_degradation] PLIC remap scalar fallback: plic_remap_soft_fallback");
    }
  };
  if (per_material_conservation_enabled) {
    bool plic_unified_remap_used = false;
    if (plic::plic_runtime_active(state, cfg)) {
      plic_status = plic::launch_plic_material_volume_remap(state,
                                                            cfg,
                                                            d_volfrac_mat,
                                                            d_vol_old,
                                                            state.vol.data(),
                                                            d_xr_old,
                                                            d_xz_old,
                                                            state.x_r.data(),
                                                            state.x_z.data(),
                                                            nr,
                                                            nz,
                                                            observability,
                                                            eos_ctx);
      if (plic_status.active &&
          !detail::plic_reconstruction_successful(plic_status)) {
        plic_status.fallback_engaged = true;
      }
      plic::apply_plic_fallback_policy(state, cfg, plic_status, observability);
      plic_unified_remap_used =
          plic_status.active &&
          !plic_status.fallback_engaged &&
          !plic_status.drift_triggered &&
          !state.plic_remap_sticky_fallback &&
          detail::plic_reconstruction_successful(plic_status);
      plic_volfrac_remap_used = plic_unified_remap_used;
      publish_plic_status(plic_status);
      if (!plic_unified_remap_used) {
        CUDA_CHECK(cudaMemcpy(d_volfrac_mat,
                              d_volfrac_mat_old,
                              n_cell_mat * sizeof(double),
                              cudaMemcpyDeviceToDevice));
      }
    }

    const auto launch_per_material_scalar_remap = [&](double* field) {
      state.dispatch_counters.per_material_kernel_call_count.fetch_add(
          1, std::memory_order_relaxed);
      return launch_scalar_remap(field);
    };
    if (plic_unified_remap_used) {
#ifndef NDEBUG
      TENRYU_ASSERT(
          per_material_old_snapshot_step == state.step,
          "ALE per-material PLIC remap requires _old snapshot from current ALE call");
#endif
      detail::RemapAdmissibilityResult remap_admissibility =
          detail::check_first_sweep_admissibility(d_vol_mid,
                                                  d_vol_old,
                                                  d_xr_old,
                                                  d_xz_old,
                                                  state.x_r.data(),
                                                  state.x_z.data(),
                                                  cw.begin,
                                                  cw.end,
                                                  nr,
                                                  nz,
                                                  state.step,
                                                  donor_sign_fixed);
      bool remap_admissible = remap_admissibility.admissible;
      if (reduction != nullptr) {
        remap_admissible =
            (reduction->allreduce_max(remap_admissible ? 0.0 : 1.0) < 0.5);
      }
      if (!remap_admissible) {
        rollback_after_remap_failure();
        cleanup();
        return out;
      }
      const int blocks_cm = (pw.count() * n_mat + 255) / 256;
      detail::plic_unified_per_material_remap_kernel<<<blocks_cm, 256>>>(
          d_eta_rho_m,
          d_eta_rho_ee_m,
          d_eta_rho_ei_m,
          d_eta_rho_vr_m,
          d_eta_rho_vz_m,
          d_eta_rho_m_old,
          d_eta_rho_ee_m_old,
          d_eta_rho_ei_m_old,
          d_eta_rho_vr_m_old,
          d_eta_rho_vz_m_old,
          d_volfrac_mat_old,
          d_vr_cell,
          d_vz_cell,
          state.plic_face_flux_r.data(),
          state.plic_face_flux_z.data(),
          d_vol_old,
          state.vol.data(),
          pw.begin,
          pw.end,
          nr,
          nz,
          n_mat,
          cfg.numerics.materials.presence_threshold_volfrac,
          cfg.numerics.materials.presence_threshold_mass_density_g_per_cc);
      state.dispatch_counters.per_material_kernel_call_count.fetch_add(
          1, std::memory_order_relaxed);
      CUDA_CHECK(cudaGetLastError());
    } else {
      for (int m = 0; m < n_mat; ++m) {
        const std::size_t off =
            static_cast<std::size_t>(m) * static_cast<std::size_t>(n_cells);
        if (!launch_per_material_scalar_remap(d_eta_rho_m + off)) {
          rollback_after_remap_failure();
          cleanup();
          return out;
        }
        if (!launch_per_material_scalar_remap(d_eta_rho_ee_m + off)) {
          rollback_after_remap_failure();
          cleanup();
          return out;
        }
        if (!launch_per_material_scalar_remap(d_eta_rho_ei_m + off)) {
          rollback_after_remap_failure();
          cleanup();
          return out;
        }
        if (!launch_per_material_scalar_remap(d_eta_rho_vr_m + off)) {
          rollback_after_remap_failure();
          cleanup();
          return out;
        }
        if (!launch_per_material_scalar_remap(d_eta_rho_vz_m + off)) {
          rollback_after_remap_failure();
          cleanup();
          return out;
        }
      }
    }
    rho_remapped = true;
  } else {
    if (!cf6_plic_rho_remap_requested && !launch_scalar_remap(state.rho.data())) {
      rollback_after_remap_failure();
      cleanup();
      return out;
    }
    rho_remapped = !cf6_plic_rho_remap_requested;
    if (!launch_scalar_remap(d_mom_r)) {
      rollback_after_remap_failure();
      cleanup();
      return out;
    }
    if (!launch_scalar_remap(d_mom_z)) {
      rollback_after_remap_failure();
      cleanup();
      return out;
    }
    if (!launch_scalar_remap(d_e_e)) {
      rollback_after_remap_failure();
      cleanup();
      return out;
    }
    if (!launch_scalar_remap(d_e_i)) {
      rollback_after_remap_failure();
      cleanup();
      return out;
    }
  }
  if (ke_closure_enabled && !launch_scalar_remap(d_ke_remap)) {
    rollback_after_remap_failure();
    cleanup();
    return out;
  }
  if (remap_rad_E) {
    for (int g = 0; g < n_groups; ++g) {
      detail::gather_group_field_kernel<<<blocks_cells, 256>>>(
          d_rad_e_group, state.rad_E.data(), n_cells, n_groups, g);
      CUDA_CHECK(cudaGetLastError());
      if (!launch_scalar_remap(d_rad_e_group)) {
        rollback_after_remap_failure();
        cleanup();
        return out;
      }
      detail::scatter_group_field_kernel<<<blocks_cells, 256>>>(
          state.rad_E.data(), d_rad_e_group, n_cells, n_groups, g);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  if (!per_material_conservation_enabled && plic::plic_runtime_active(state, cfg)) {
    plic_status = plic::launch_plic_material_volume_remap(state,
                                                          cfg,
                                                          d_volfrac_mat,
                                                          d_vol_old,
                                                          state.vol.data(),
                                                          d_xr_old,
                                                          d_xz_old,
                                                          state.x_r.data(),
                                                          state.x_z.data(),
                                                          nr,
                                                          nz,
                                                          observability,
                                                          eos_ctx);
    plic::apply_plic_fallback_policy(state, cfg, plic_status, observability);
    plic_volfrac_remap_used = plic_status.active && !plic_status.fallback_engaged;
    if (plic_volfrac_remap_used && cf6_plic_rho_remap_requested) {
      rho_remapped = true;
    }
    publish_plic_status(plic_status);
  }
  if (!rho_remapped && !launch_scalar_remap(state.rho.data())) {
    rollback_after_remap_failure();
    cleanup();
    return out;
  }
  if (!plic_volfrac_remap_used) {
    for (int m = 0; m < n_mat; ++m) {
      if (!launch_scalar_remap(d_volfrac_mat + static_cast<std::size_t>(m) * n_cells)) {
        rollback_after_remap_failure();
        cleanup();
        return out;
      }
    }
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(core::debug_kernel_sync());

  if (per_material_conservation_enabled && closure_audit_enabled) {
    detail::reduce_per_material_conserved_to_cell_kernel<<<blocks_cells, 256>>>(
        state.rho.data(),
        d_mom_r,
        d_mom_z,
        d_e_e,
        d_e_i,
        d_eta_rho_m,
        d_eta_rho_ee_m,
        d_eta_rho_ei_m,
        d_eta_rho_vr_m,
        d_eta_rho_vz_m,
        n_cells,
        n_mat);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(core::debug_kernel_sync());
  }

  if (closure_audit_enabled) {
    detail::audit_post_remap_kernel<<<cw.blocks(), 256>>>(
        d_audit_cell,
        d_ke_remap,
        d_e_e,
        d_e_i,
        d_mom_r,
        d_mom_z,
        state.rho.data(),
        state.vol.data(),
        cw.begin,
        cw.end,
        n_cells);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(core::debug_kernel_sync());
    const auto post_remap_sums = detail::reduce_device_column_sums(
        d_audit_cell, n_cells, detail::kAuditPostRemapCols);
    out.closure_audit.K_remap_total = post_remap_sums[0];
    out.closure_audit.I_raw = post_remap_sums[1];
    out.closure_audit.K_cellmom = post_remap_sums[2];
  }

  if (per_material_conservation_enabled) {
    detail::recover_primitive_kernel_per_material<<<cw.blocks(), 256>>>(
        state.rho.data(),
        state.mass.data(),
        state.ee.data(),
        state.ei.data(),
        d_vr_cell,
        d_vz_cell,
        state.volFrac.data(),
        state.mass_per_material.data(),
        state.Ee_per_material.data(),
        state.Ei_per_material.data(),
        d_eta_rho_m,
        d_eta_rho_ee_m,
        d_eta_rho_ei_m,
        d_eta_rho_vr_m,
        d_eta_rho_vz_m,
        d_volfrac_mat,
        state.vol.data(),
        cw.begin,
        cw.end,
        n_cells,
        n_mat,
        cfg.numerics.floors.rho,
        d_dm_floor,
        d_e_floor,
        d_dm_floor_per_material,
        d_repair_count);
    state.dispatch_counters.per_material_kernel_call_count.fetch_add(
        1, std::memory_order_relaxed);
  } else {
    detail::recover_primitive_kernel<<<cw.blocks(), 256>>>(state.rho.data(),
                                                             state.mass.data(),
                                                             state.ee.data(),
                                                             state.ei.data(),
                                                             d_vr_cell,
                                                             d_vz_cell,
                                                             state.volFrac.data(),
                                                             d_mom_r,
                                                             d_mom_z,
                                                             d_e_e,
                                                             d_e_i,
                                                             d_volfrac_mat,
                                                             state.vol.data(),
                                                             cw.begin,
                                                             cw.end,
                                                             n_cells,
                                                             n_mat,
                                                             cfg.numerics.floors.rho,
                                                             d_dm_floor,
                                                             d_e_floor);
  }
  CUDA_CHECK(cudaGetLastError());

  d_vf_deg = static_cast<int*>(
      core::device_scratch_acquire("ale_driver:ale_req:d_vf_deg", sizeof(int)));
  const int zero_i = 0;
  CUDA_CHECK(cudaMemcpy(d_vf_deg, &zero_i, sizeof(int), cudaMemcpyHostToDevice));
  if (n_mat > 0 && !plic_volfrac_remap_used) {
    normalize_volfrac_kernel<<<blocks_cells, 256>>>(state.volFrac.data(), d_vf_deg, n_cells,
                                                     n_mat);
  }

  const auto r_outer_type =
      parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_outer);
  const auto z_bottom_type =
      parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_bottom);
  const auto z_top_type =
      parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_top);
  const int r_outer_bc_mode = detail::velocity_bc_mode(r_outer_type);
  const int z_bottom_bc_mode = detail::velocity_bc_mode(z_bottom_type);
  const int z_top_bc_mode = detail::velocity_bc_mode(z_top_type);

  if (ke_closure_enabled) {
    if (corner_mass_lagrangian_invariant_enabled(cfg)) {
      detail::warn_invariant_corner_mass_ke_reinit_once();
    }
    if (state.corner_mass.size() != expected_corner_mass_size) {
      state.corner_mass.reset(expected_corner_mass_size);
      state.corner_mass_initialized = false;
    }
    detail::compute_current_corner_mass_kernel<<<blocks_cells, 256>>>(
        state.corner_mass.data(), state.mass.data(), state.x_r.data(),
        state.x_z.data(), d_cell_nverts,
        rz::corner_mass_fallback_device_recorder(),
        rz::kCornerMassFallbackStageAleDriverPost,
        static_cast<int>(
            cfg.numerics.hydro.corner_mass_convention),
        nr, nz, state.corner_stride);
    CUDA_CHECK(cudaGetLastError());
    state.corner_mass_initialized = true;
  }

  if (ke_closure_enabled) {
    if (closure_audit_enabled) {
      d_audit_vr_node = static_cast<double*>(
          core::device_scratch_acquire("ale_driver:ale_req:d_audit_vr_node",
                                       static_cast<std::size_t>(n_nodes) * sizeof(double)));
      d_audit_vz_node = static_cast<double*>(
          core::device_scratch_acquire("ale_driver:ale_req:d_audit_vz_node",
                                       static_cast<std::size_t>(n_nodes) * sizeof(double)));
      detail::project_cell_velocity_to_nodes_corner_mass_no_bc_kernel<<<nw.blocks(), 256>>>(
          d_audit_vr_node,
          d_audit_vz_node,
          d_vr_cell,
          d_vz_cell,
          state.corner_mass.data(),
          d_cell_nverts,
          nw.begin,
          nw.end,
          nr,
          nz);
      CUDA_CHECK(cudaGetLastError());
      detail::audit_corner_kinetic_total_kernel<<<blocks_cells, 256>>>(
          d_audit_cell,
          state.corner_mass.data(),
          d_audit_vr_node,
          d_audit_vz_node,
          d_cell_nverts,
          nr,
          nz);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(core::debug_kernel_sync());
      out.closure_audit.K_node_preBC =
          detail::reduce_device_sum(d_audit_cell, n_cells);
    }
    detail::project_cell_velocity_to_nodes_corner_mass_kernel<<<nw.blocks(), 256>>>(
        state.v_r.data(),
        state.v_z.data(),
        d_vr_cell,
        d_vz_cell,
        state.corner_mass.data(),
        d_cell_nverts,
        d_node_flags,
        nw.begin,
        nw.end,
        nr,
        nz,
        r_outer_bc_mode,
        z_bottom_bc_mode,
        z_top_bc_mode);
    CUDA_CHECK(cudaGetLastError());
    if (closure_audit_enabled) {
      detail::audit_corner_kinetic_total_kernel<<<blocks_cells, 256>>>(
          d_audit_cell,
          state.corner_mass.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_cell_nverts,
          nr,
          nz);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(core::debug_kernel_sync());
      out.closure_audit.K_node_postBC =
          detail::reduce_device_sum(d_audit_cell, n_cells);
    }
    if (ke_closure_redistribute_floor) {
      constexpr double kClosureSpecificEnergyFloor = 0.0;
      detail::compute_kinetic_closure_deficit_capacity_kernel<<<cw.blocks(), 256>>>(
          d_ke_closure_deficit,
          d_ke_closure_capacity,
          state.ee.data(),
          state.ei.data(),
          d_ke_remap,
          state.corner_mass.data(),
          state.mass.data(),
          state.vol.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_cell_nverts,
          kClosureSpecificEnergyFloor,
          cw.begin,
          cw.end,
          nr,
          nz,
          cfg.main.two_temperature ? 1 : 0);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(core::debug_kernel_sync());

      const double local_deficit =
          detail::reduce_device_sum(d_ke_closure_deficit, n_cells);
      const double local_capacity =
          detail::reduce_device_sum(d_ke_closure_capacity, n_cells);
      double global_deficit = local_deficit;
      double global_capacity = local_capacity;
      if (reduction != nullptr) {
        double sums[2] = {global_deficit, global_capacity};
        reduction->allreduce_sum(sums, 2);
        global_deficit = sums[0];
        global_capacity = sums[1];
      }

      double absorption_factor = 0.0;
      if (global_deficit > 0.0 && global_capacity > 0.0) {
        absorption_factor = std::min(1.0, global_deficit / global_capacity);
      }
      if (global_deficit > global_capacity && global_deficit > 0.0) {
        out.E_redistribution_unresolved =
            (global_deficit - global_capacity) * (local_deficit / global_deficit);
      }

      detail::apply_kinetic_closure_redistribution_kernel<<<cw.blocks(), 256>>>(
          state.ee.data(),
          state.ei.data(),
          closure_audit_enabled ? d_audit_cell : nullptr,
          d_ke_remap,
          state.corner_mass.data(),
          state.mass.data(),
          state.vol.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_cell_nverts,
          kClosureSpecificEnergyFloor,
          absorption_factor,
          cw.begin,
          cw.end,
          nr,
          nz,
          cfg.main.two_temperature ? 1 : 0);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(core::debug_kernel_sync());
      if (closure_audit_enabled) {
        const detail::DepositAuditReduction deposit_audit =
            detail::reduce_deposit_audit(d_audit_cell, n_cells);
        out.closure_audit.sum_dI_raw = deposit_audit.sum_dI_raw;
        out.closure_audit.sum_dI_after_floor = deposit_audit.sum_dI_after_floor;
        out.closure_audit.min_tentative_e_e = deposit_audit.min_tentative_e_e;
        out.closure_audit.min_tentative_e_i = deposit_audit.min_tentative_e_i;
        out.closure_audit.n_cells_negative_dI = deposit_audit.n_cells_negative_dI;
        out.closure_audit.n_cells_floor_e = deposit_audit.n_cells_floor_e;
        out.closure_audit.n_cells_floor_i = deposit_audit.n_cells_floor_i;
      }
    } else if (closure_audit_enabled) {
      detail::deposit_kinetic_closure_audit_kernel<<<cw.blocks(), 256>>>(
          state.ee.data(),
          state.ei.data(),
          d_audit_cell,
          d_ke_remap,
          state.corner_mass.data(),
          state.mass.data(),
          state.vol.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_cell_nverts,
          cw.begin,
          cw.end,
          nr,
          nz,
          cfg.main.two_temperature ? 1 : 0);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(core::debug_kernel_sync());
      const detail::DepositAuditReduction deposit_audit =
          detail::reduce_deposit_audit(d_audit_cell, n_cells);
      out.closure_audit.sum_dI_raw = deposit_audit.sum_dI_raw;
      out.closure_audit.sum_dI_after_floor = deposit_audit.sum_dI_after_floor;
      out.closure_audit.min_tentative_e_e = deposit_audit.min_tentative_e_e;
      out.closure_audit.min_tentative_e_i = deposit_audit.min_tentative_e_i;
      out.closure_audit.n_cells_negative_dI = deposit_audit.n_cells_negative_dI;
      out.closure_audit.n_cells_floor_e = deposit_audit.n_cells_floor_e;
      out.closure_audit.n_cells_floor_i = deposit_audit.n_cells_floor_i;
    } else {
      detail::deposit_kinetic_closure_kernel<<<cw.blocks(), 256>>>(
          state.ee.data(),
          state.ei.data(),
          d_ke_remap,
          state.corner_mass.data(),
          state.mass.data(),
          state.vol.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_cell_nverts,
          cw.begin,
          cw.end,
          nr,
          nz,
          cfg.main.two_temperature ? 1 : 0);
      CUDA_CHECK(cudaGetLastError());
    }
  } else {
    project_cell_velocity_to_nodes_kernel<<<blocks_nodes, 256>>>(
        state.v_r.data(), state.v_z.data(), d_vr_cell, d_vz_cell, state.rho.data(),
        state.vol.data(), d_node_flags, nr, nz, r_outer_bc_mode, z_bottom_bc_mode,
        z_top_bc_mode);
  }

  if (per_material_conservation_enabled) {
    if (eos_ctx == nullptr) {
      bool any_table = false;
      for (const auto& material : cfg.materials.materials) {
        if (material.eos_tables != nullptr || material.eos_model != "ideal_gas") {
          any_table = true;
          break;
        }
      }
      if (any_table) {
        TENRYU_ASSERT(false,
                      "per-material EOS requires HydroEOSContext for table-backed materials");
      }
    }
    per_material::refresh_per_material_derived_cell_fields(state, cfg, eos_ctx);
  } else {
    const auto& mat = cfg.materials.materials.front();
    tenryu::materials::DeviceEOSTableView ion_eos{};
    tenryu::materials::DeviceEOSTableView ele_eos{};
    if (eos_ctx != nullptr && eos_ctx->n_materials > 0) {
      ion_eos = eos_ctx->ion_view(0);
      ele_eos = eos_ctx->electron_view(0);
    }
    detail::eos_reclosure_kernel<<<blocks_cells, 256>>>(state.Te.data(),
                                                         state.Ti.data(),
                                                         state.Pe.data(),
                                                         state.Pi.data(),
                                                         state.ee.data(),
                                                         state.ei.data(),
                                                         state.cv_e.empty() ? nullptr
                                                                            : state.cv_e.data(),
                                                         state.cv_i.empty() ? nullptr
                                                                            : state.cv_i.data(),
                                                         state.mass.data(),
                                                         state.rho.data(),
                                                         state.zbar.data(),
                                                         n_cells,
                                                         mat.ideal_gas_gamma,
                                                         mat.A,
                                                         cfg.numerics.floors.rho,
                                                         cfg.numerics.floors.Te,
                                                         cfg.numerics.floors.Ti,
                                                         ion_eos,
                                                         ele_eos,
                                                         cfg.materials.low_density_extrapolation,
                                                         d_e_floor);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(core::debug_kernel_sync());
  }
  record_estep_trace_boundary(state,
                              cfg,
                              part,
                              reduction,
                              "post_remap",
                              dt_hydro_used,
                              state.t + dt_hydro_used);

  if (part.n_ranks > 1 && bufs != nullptr) {
    const int cell_size = n_cells;
    double* cell_ptrs[5] = {
        state.rho.data(), state.Te.data(), state.Ti.data(), state.Pe.data(), state.Pi.data()};
    parallel::exchange_cell_fields(part, *bufs, cell_ptrs, 5, cell_size, nullptr, 6);
  }

  CUDA_CHECK(cudaMemcpy(&out.mass_floor_delta,
                        d_dm_floor,
                        sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&out.E_floor_injected,
                        d_e_floor,
                        sizeof(double),
                        cudaMemcpyDeviceToHost));
  if (per_material_conservation_enabled) {
    out.mass_floor_delta_per_material.assign(static_cast<std::size_t>(n_mat), 0.0);
    CUDA_CHECK(cudaMemcpy(out.mass_floor_delta_per_material.data(),
                          d_dm_floor_per_material,
                          static_cast<std::size_t>(n_mat) * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&out.per_material_repair_count,
                          d_repair_count,
                          sizeof(int),
                          cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaMemcpy(&out.volfrac_degenerate,
                        d_vf_deg,
                        sizeof(int),
                        cudaMemcpyDeviceToHost));

  if (closure_audit_enabled) {
    const diagnostics::EnergyTotals closure_audit_post_budget =
        diagnostics::compute_energy_totals_2d(state);
    const double pre_total = closure_audit_pre_budget.E_int_e +
                             closure_audit_pre_budget.E_int_i +
                             closure_audit_pre_budget.E_kin;
    const double post_total = closure_audit_post_budget.E_int_e +
                              closure_audit_post_budget.E_int_i +
                              closure_audit_post_budget.E_kin;
    out.closure_audit.K_post_budget = closure_audit_post_budget.E_kin;
    out.closure_audit.dE_ale_total = post_total - pre_total;
    out.closure_audit.valid = true;
    detail::reduce_closure_audit_global(out.closure_audit, reduction);
    diagnostics::finalize_ale_closure_audit(out.closure_audit);
  }

  if (out.volfrac_degenerate > 0) {
    core::log_warning("ALE normalize_volFrac encountered degenerate cells: " +
                      std::to_string(out.volfrac_degenerate));
  }

  if (remap_damage_budget_enabled && !state.axis_inflow_budget.empty() &&
      accepted_axis_inflow_this_event.size() == static_cast<std::size_t>(nz)) {
    for (int j = 0; j < nz; ++j) {
      state.axis_inflow_budget[static_cast<std::size_t>(j)] +=
          accepted_axis_inflow_this_event[static_cast<std::size_t>(j)];
    }
  }

  state.ale_remaps_applied += 1;
  out.accepted_remap_count = state.ale_remaps_applied;
  tenryu::hydro::ensure_hourglass_subzonal_masses_2d(
      state, cfg, !csr_optionb_velocity_authority_enabled(cfg));
  if (cfg.numerics.diagnostics.mesh_attribution.enabled &&
      cfg.numerics.diagnostics.mesh_attribution.record_node_displacements) {
    tenryu::diagnostics::mesh_attribution::record_zero_source(
        tenryu::diagnostics::mesh_attribution::MeshDeformSource::Remap);
  }
  if (debug_per_remap_log) {
    const double remap_mass_post = detail::compute_total_mass(state, reduction);
    const double remap_energy_post =
        detail::total_energy(detail::reduce_energy_totals_global(
            diagnostics::compute_energy_totals_2d(state), reduction));
    const double dM = remap_mass_post - remap_mass_pre;
    const double dE = remap_energy_post - remap_energy_pre;
    double conservation_values[3] = {out.E_floor_injected,
                                     out.E_redistribution_unresolved,
                                     out.cap_energy_audit_D_K};
    if (reduction != nullptr) {
      reduction->allreduce_sum(conservation_values, 3);
    }
    core::log_warning("[ale-stats] remap_delta step=" + std::to_string(state.step) +
                      " n_applied=" + std::to_string(state.ale_remaps_applied) +
                      " dM=" + detail::format_scientific(dM) +
                      " dM_rel=" +
                      detail::format_scientific(detail::relative_delta(dM, remap_mass_pre)) +
                      " dE=" + detail::format_scientific(dE) +
                      " dE_rel=" +
                      detail::format_scientific(detail::relative_delta(dE, remap_energy_pre)));
    core::log_warning(
        "[ale_per_remap] step=" + std::to_string(state.step) +
        " t=" + detail::format_scientific17(state.t + dt_hydro_used) +
        " n_applied=" + std::to_string(state.ale_remaps_applied) +
        " dM=" + detail::format_scientific17(dM) +
        " dM_rel=" +
        detail::format_scientific17(detail::relative_delta(dM, remap_mass_pre)) +
        " dE=" + detail::format_scientific17(dE) +
        " dE_rel=" +
        detail::format_scientific17(detail::relative_delta(dE, remap_energy_pre)) +
        " E_floor_injected=" +
        detail::format_scientific17(conservation_values[0]) +
        " E_redistribution_unresolved=" +
        detail::format_scientific17(conservation_values[1]) +
        " cap_energy_audit_D_K=" +
        detail::format_scientific17(conservation_values[2]));
  }
  log_axis_uR_if_nonzero();
  pole_axis_diag::emit_post_remap(state, "post_remap");

  // Reset volume-rate CFL Lagrangian baseline after accepted ALE rezone+remap.
  // ALE displacement is non-Lagrangian; without this reset, the next
  // compute_dt_hydro's volume-rate clamp interprets it as one-step hydro flow.
  // Phase 2d-ext v3 root cause for L2 256x512 dt-collapse cascade.
  hydro::reset_volume_rate_cfl_history_after_ale(state);
  central_pseudo_core::aggregate_state(state, cfg, "post_ale_remap", false);
  pole_angular_derefine::maintain_pole_spans(state, cfg, "post_ale_remap");
  pole_angular_derefine::aggregate_state(state, cfg, "post_ale_remap", true);
  conservation_audit::emit_stage(state, "ale_post_remap_after_aggregate");
  record_estep_trace_boundary(state,
                              cfg,
                              part,
                              reduction,
                              "step_end",
                              dt_hydro_used,
                              state.t + dt_hydro_used);

  if (cfg.numerics.diagnostics.production_audit.gcl.enabled ||
      conservation_audit::enabled()) {
    (void)log_ale_vol_closure_residual(state, reduction);
  }

  const bool repair_or_escape_fired =
      out.per_material_repair_count > 0 || out.plic_repair_events > 0 ||
      out.effective_mode_executed == AleMode::AxisSpinePlusLocal ||
      out.effective_mode_executed == AleMode::BoundaryPatchProjection ||
      out.effective_mode_executed == AleMode::CdLocalWinslow ||
      out.effective_mode_executed == AleMode::InteriorMultiNodeProjection ||
      out.effective_mode_executed == AleMode::AxisVariationalProjection;
  dump_i1b_polar_pole_diag("ale_end", out.rezone_triggered, repair_or_escape_fired);

  cleanup();

  return out;
}

AleStepResult apply_ale(core::State& state, const core::Config& cfg,
                        const parallel::PartitionInfo& part,
                        parallel::CommBuffers* bufs,
                        const parallel::Reduction* reduction,
                        const HydroEOSContext* eos_ctx,
                        const double dt_hydro_used,
	                        const bool force_rezone,
	                        const char* force_reason,
	                        tenryu::coupling::ProfileObservability* observability,
	                        const AleDriverRetryContext* retry_context) {
  ale_align::maybe_log_post_lagrange(state, cfg, dt_hydro_used, part.rank);
  if (ale_identity_mode_enabled(cfg)) {
    return AleStepResult{};
  }
  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    reset_remap_mass_closure_step_max();
    AleStepResult multiblock_result = apply_multiblock_csr_ale_step(
        state, cfg, part, reduction, eos_ctx, dt_hydro_used, force_rezone, force_reason);
    multiblock_result.remap_mass_closure_rel = remap_mass_closure_step_max();
    return multiblock_result;
  }
  return apply_ale_with_request(state,
                                cfg,
                                nullptr,
                                nullptr,
                                part,
                                bufs,
                                reduction,
                                eos_ctx,
                                dt_hydro_used,
                                force_rezone,
                                force_reason,
                                observability,
                                retry_context);
}

}  // namespace tenryu::hydro::ale
