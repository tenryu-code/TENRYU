#include "coupling/persistent_loop.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "core/constants.hpp"
#include "core/config.hpp"
#include "core/device_block_primitives.cuh"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/namelist/frozen_table_device.cuh"
#include "core/state.hpp"
#include "coupling/dt_controller_device.cuh"
#include "coupling/rad_gamma_coupling.cuh"
#include "coupling/rad_gamma_coupling_bodies.cuh"
#include "hydro/boundary.hpp"
#include "hydro/central_pseudo_core.cuh"
#include "hydro/conduction_bodies.cuh"
#include "hydro/hydro_1d_bodies.cuh"
#include "materials/opacity_eval.cuh"
#include "mesh/mesh.hpp"
#include "parallel/partition.hpp"
#include "laser/beams.cuh"
#include "laser/ib_absorption.cuh"
#include "laser/laser_mesh.cuh"
#include "laser/laser_mesh_bodies.cuh"
#include "laser/laser_phys_ext.cuh"
#include "laser/ray_trace_bodies.cuh"
#include "laser/refraction.cuh"
#include "radiation/fld_1d_bodies.cuh"
#include "radiation/planck_table.cuh"

namespace tenryu::coupling {
namespace {

constexpr int kBlockSize = 512;
constexpr int kExitChunkBudget = 0;
constexpr int kExitTEnd = 1;
constexpr int kExitOutputBoundary = 2;
constexpr int kExitRetryOrError = 3;
constexpr int kExitMaxSteps = 4;
constexpr int kUpdateMatterWarps =
    kBlockSize / radiation::fld_1d_bodies::kUpdateMatterWarpSize;
constexpr int kPkWatchProgressInts = 8;
constexpr int kPkPhaseProfSlots = 32;
constexpr int kPkPhaseProfFldOuterIterations = 29;
constexpr int kPkPhaseProfStepCount = 30;
constexpr int kPkPhaseProfGridSyncCount = 31;

struct PersistentParams {
  int n_cells = 0;
  int n_nodes = 0;
  int n_materials = 0;
  int first_void_cell = 0;
  int geom_code = 0;
  int boundary_type = 0;
  int apply_outer_boundary = 0;
  int chunk_steps = 0;
  int max_steps_remaining = 0;
  int use_cooperative = 0;
  int cooperative_grid_blocks = 1;
  int phase_trace = 0;
  int phase_prof = 0;
  int laser_trace = 0;
  volatile int* progress = nullptr;
  double* phase_prof_slab = nullptr;
  int renorm_active = 0;
  int hydro_enabled = 0;
  int compatible_energy = 0;
  int energy_authoritative = 0;
  int two_temperature = 0;
  int q_heat_to_electron = 0;
  int conduction_enabled = 0;
  int conduction_geom_code = 0;
  int conduction_face_kappa_policy = 0;
  int conduction_kappa_power_active = 0;
  int conduction_sts_max_stages = 0;
  int radiation_enabled = 0;
  int rad_gamma43_enabled = 0;
  int n_groups = 1;
  int fld_outer_bc = 0;
  int fld_flux_limiter = 0;
  int fld_use_nlte_table = 0;
  int fld_use_fleck = 0;
  int fld_use_fleck_blend = 0;
  int fld_fleck_form_exp = 0;
  int fld_has_cv_e = 0;
  int fld_max_outer_iterations = 0;
  int laser_enabled = 0;
  int laser_n_folded = 0;
  int laser_mode_raytrace_1d = 0;
  int laser_rays_per_beam = 0;
  int laser_profile_kind = 0;
  int laser_profile_m = 2;
  int laser_lmesh_nr_capacity = 0;
  int laser_lmesh_nz_capacity = 0;
  int laser_lmesh_n_nodes_r_capacity = 0;
  int laser_lmesh_n_nodes_z_capacity = 0;
  int laser_lmesh_nodes_capacity = 0;
  int laser_lmesh_nr_max = 0;
  int laser_lmesh_critical_clip = 0;
  int laser_lmesh_ghost_corona_enabled = 0;
  int laser_lmesh_ghost_n_out = 0;
  int laser_lmesh_ghost_handoff_cells = 4;
  int laser_raytrace_max_steps = laser::ray_trace_bodies::kMaxRayStepsGuard;
  radiation::PlanckTableDeviceView planck{};
  materials::IonmixOpacityDeviceView nlte_opacity{};
  core::namelist::FrozenTable1DDeviceView laser_power_table{};
  double gamma = 5.0 / 3.0;
  double fallback_z = 1.0;
  double material_A = 1.0;
  double material_gm1 = 2.0 / 3.0;
  double cv_e_override = -1.0;
  double eos_T_ref_eV = -1.0;
  double Te_floor = 1.0e-3;
  double Ti_floor = 1.0e-3;
  double rho_floor = 1.0e-10;
  double av_linear = 0.1;
  double av_quadratic = 1.5;
  double av_limiter_J = 1.0;
  double cfl_hydro = 0.3;
  double cfl_cond = 0.25;
  double conduction_f_lim = 0.06;
  double conduction_mfp_limiter_C = 0.0;
  double conduction_sts_damping = 0.01;
  double conduction_sts_subcycle_eta = 0.9;
  double conduction_test_kappa = -1.0;
  double conduction_kappa_power = 0.0;
  double conduction_kappa_rho_power = 0.0;
  double fld_kappa_a = 0.0;
  double fld_opacity_floor = 1.0e-100;
  double fld_opacity_cap = 1.0e20;
  double fld_outer_tol = 1.0e-6;
  double fld_marshak_Tr_eV = 0.0;
  double fld_marshak_flux = 0.0;
  double fld_marshak_pulse_duration = -1.0;
  double fld_volume_source_rate = 0.0;
  double fld_volume_source_r_max = 0.0;
  double laser_n_crit = 0.0;
  double laser_lambda_cm = 0.0;
  double laser_eps_n = 1.0e-4;
  double laser_eps_crit = 1.0e-4;
  double laser_test_kappa = -1.0;
  double laser_intensity_cutoff = 1.0e-6;
  double laser_coulomb_log_floor = 2.0;
  double laser_raytrace_cfl_ray = 0.8;
  double laser_raytrace_ds_adapt_g_target = 0.05;
  double laser_raytrace_ds_adapt_tau_target = 0.05;
  double laser_raytrace_ds_adapt_theta_target = 0.02;
  double laser_raytrace_ds_adapt_max_factor = 4.0;
  double laser_lmesh_mesh_factor = 0.5;
  double laser_lmesh_rmax_n_hat_threshold = 0.001;
  double laser_lmesh_r_max_factor = 1.5;
  double laser_lmesh_target_radius = 0.0;
  double laser_lmesh_n_hat_margin = 0.9999;
  double laser_lmesh_ghost_ne_min_frac = 0.03;
  double laser_lmesh_ghost_ne_max_frac = 1.05;
  double laser_lmesh_ghost_Te_min_eV = 50.0;
  double laser_lmesh_ghost_zbar_min = 1.0;
  double laser_lmesh_ghost_zbar_max = 4.0;
  double laser_lmesh_ghost_handoff_decay = 1.5;
  double laser_beam_f_number = 8.0;
  double laser_beam_focus_lab_z = 0.0;
  double laser_beam_profile_w0_cm = 0.0;
  double fld_alpha = 1.0;
  double fld_f_min = 0.01;
  double growth_factor = 1.2;
  double dt_max = 1.0e-9;
  double r_min = 0.0;
  double r_max = 0.0;
  double t_end = 0.0;
  double t_next_output = 0.0;
  double void_rho = 0.0;
};

struct PersistentMaterialParams {
  double A = 1.0;
  double Zbar = 1.0;
  double gamma = 5.0 / 3.0;
  double cv_e_override = -1.0;
  double eos_T_ref_eV = -1.0;
  double kappa_a_constant = 0.0;
  int is_void = 0;
};

struct PersistentDeviceBuffers {
  double* rho = nullptr;
  double* mass = nullptr;
  double* vol = nullptr;
  const double* volFrac = nullptr;
  const PersistentMaterialParams* material_params = nullptr;
  double* zbar = nullptr;
  double* A_eff = nullptr;
  double* gamma_eff = nullptr;
  double* Te = nullptr;
  double* Ti = nullptr;
  double* ee = nullptr;
  double* ei = nullptr;
  double* Pe = nullptr;
  double* Pi = nullptr;
  double* Qvisc = nullptr;
  double* cv_e = nullptr;
  double* cv_i = nullptr;
  double* cs = nullptr;
  double* x_r = nullptr;
  double* v_r = nullptr;
  std::int8_t* hydro_active = nullptr;
  const std::uint8_t* cell_is_void = nullptr;

  double* cell_centroid_r = nullptr;
  double* cell_centroid_z = nullptr;
  double* sigma = nullptr;
  double* r_old = nullptr;
  double* u_old = nullptr;
  double* V_old = nullptr;
  double* e_old = nullptr;
  double* ei_old = nullptr;
  double* Pe_old = nullptr;
  double* Pi_old = nullptr;
  double* Q_old = nullptr;
  std::uint8_t* node_active = nullptr;
  double* pq_n = nullptr;
  double* a_n = nullptr;
  double* u_half = nullptr;
  double* cs_half = nullptr;
  double* P_half = nullptr;
  double* Pi_half = nullptr;
  double* Q_half = nullptr;
  double* rho_half = nullptr;
  double* Te_half = nullptr;
  double* Ti_half = nullptr;
  double* pq_half = nullptr;
  double* a_half = nullptr;
  double* floors_pack = nullptr;
  double* E_floor = nullptr;
  int* clamp_count = nullptr;
  int* rho_clamp_count = nullptr;
  int* failing_cell = nullptr;

  double* cond_kappa_eff = nullptr;
  double* cond_rho_cv_e = nullptr;
  double* cond_te_tmp = nullptr;
  double* cond_flux_limiter_faces = nullptr;
  double* cond_diag3 = nullptr;
  double* cond_A_eff = nullptr;
  double* cond_gamma_eff = nullptr;

  double* rad_E = nullptr;
  double* rad_E_old = nullptr;
  double* rad_gamma_p_r = nullptr;
  double* rad_gamma_W_r = nullptr;
  double* rad_gamma_vol_before = nullptr;
  double* rad_gamma_r_half = nullptr;
  double* rad_dep = nullptr;
  double* rad_emit = nullptr;
  double* fld_sigma_a = nullptr;
  double* fld_sigma_pe = nullptr;
  double* fld_sigma_R = nullptr;
  double* fld_eta = nullptr;
  double* fld_lower = nullptr;
  double* fld_diag = nullptr;
  double* fld_upper = nullptr;
  double* fld_rhs = nullptr;
  double* fld_Te_old = nullptr;
  double* fld_delta_T = nullptr;
  double* fld_cp_work = nullptr;
  double* fld_pcr_dl_work = nullptr;
  double* fld_pcr_d_work = nullptr;
  double* fld_pcr_du_work = nullptr;
  double* fld_pcr_rhs_work = nullptr;
  double* fld_fleck = nullptr;
  double* fld_nlte_sigma_eff_work = nullptr;
  double* fld_nlte_sigma_s_eff_work = nullptr;
  double* fld_nlte_eta_cdf_work = nullptr;
  double* fld_nlte_lambda_work = nullptr;
  double* fld_marshak_finc = nullptr;
  double* laser_dep = nullptr;
  double* laser_node_R = nullptr;
  double* laser_node_Z = nullptr;
  double* laser_n_e_hat = nullptr;
  double* laser_n_e_hat_raw = nullptr;
  double* laser_T_e = nullptr;
  double* laser_Zbar = nullptr;
  double* laser_smooth_kappa_factor = nullptr;
  double* laser_grad_n_hat_R = nullptr;
  double* laser_grad_n_hat_Z = nullptr;
  double* laser_radial_node_r = nullptr;
  double* laser_radial_n_hat = nullptr;
  double* laser_radial_n_hat_raw = nullptr;
  double* laser_radial_smooth_kappa = nullptr;
  double* laser_radial_dn_dr = nullptr;
  void* laser_step_tally_slab = nullptr;
  double* laser_unabsorbed = nullptr;
  unsigned long long* laser_tail_closure_count = nullptr;
  double* laser_tail_closure_absorbed_power = nullptr;
  unsigned long long* laser_critical_surface_hit_count = nullptr;
  core::DeviceErrorFlags* laser_error_flags = nullptr;
  double* laser_ray_R0 = nullptr;
  double* laser_ray_Z0 = nullptr;
  double* laser_ray_vR0 = nullptr;
  double* laser_ray_vZ0 = nullptr;
  double* laser_ray_power = nullptr;
  double* laser_ray_power0 = nullptr;
  double* laser_ray_weights = nullptr;
  double* laser_per_ray_deposit = nullptr;
  double* laser_per_ray_unabsorbed = nullptr;
  double* laser_per_ray_tail_power = nullptr;
  double* laser_cell_n_hat = nullptr;
  double* laser_widths = nullptr;
  double* laser_ema_state = nullptr;
  int* laser_ema_valid = nullptr;
  std::uint8_t* laser_cell_is_void = nullptr;

  double* grid_reduce_partials = nullptr;
  int* grid_reduce_indices = nullptr;
  double* grid_reduce_scalar = nullptr;
  int* grid_reduce_index_scalar = nullptr;
  double* grid_broadcast_doubles = nullptr;
  int* grid_broadcast_ints = nullptr;
  int* grid_error = nullptr;
};

struct PersistentDiagRecord {
  double t = 0.0;
  double dt = 0.0;
  double cand_hydro = 0.0;
  double cand_cond = 0.0;
  double cand_rad = 0.0;
  double checksum_u = 0.0;
  double checksum_e = 0.0;
  double fld_outer_residual = 0.0;
  double fld_escaped = 0.0;
  double fld_marshak_in = 0.0;
  double fld_volume_source = 0.0;
  double laser_input = 0.0;
  double laser_escaped = 0.0;
  double laser_floor = 0.0;
  double laser_skipped = 0.0;
  int step = 0;
  int limiter = 0;
  int fld_outer_iterations = 0;
  int use_cooperative = 0;
  int cooperative_grid_blocks = 1;
  int pad = 0;
};

static constexpr int kPersistentDiagRecordVersion = 5;

// PersistentChunkResult::error_code packs:
// low 8 bits reason enum:
// 1=laser mesh prep failed, 2=laser ray error flags,
// 3=hydro non-positive volume, 4=FLD divergence, 5=conduction.
// For reason 2, bits 8..15 hold laser flag bits:
// nan_particle/invalid_cell/invalid_boundary/pool_overflow/
// opacity_out_of_range/infinite_loop/ddmc_sigma_tot_zero/roulette_kill.
constexpr int kPkErrorReasonMask = 0xff;
constexpr int kPkErrorLaserFlagShift = 8;

enum PkErrorReason {
  kPkErrorLaserMeshPrep = 1,
  kPkErrorLaserRayFlags = 2,
  kPkErrorHydroNonPositiveVolume = 3,
  kPkErrorFldDivergence = 4,
  kPkErrorConduction = 5,
};

enum PkLaserErrorFlagBits {
  kPkLaserFlagNanParticle = 1 << 0,
  kPkLaserFlagInvalidCell = 1 << 1,
  kPkLaserFlagInvalidBoundary = 1 << 2,
  kPkLaserFlagPoolOverflow = 1 << 3,
  kPkLaserFlagOpacityOutOfRange = 1 << 4,
  kPkLaserFlagInfiniteLoop = 1 << 5,
  kPkLaserFlagDdmcSigmaTotZero = 1 << 6,
  kPkLaserFlagRouletteKill = 1 << 7,
};

__host__ __device__ inline int pk_encode_error_code(const int reason,
                                                    const int laser_flags = 0) {
  return (reason & kPkErrorReasonMask) |
         ((laser_flags & kPkErrorReasonMask) << kPkErrorLaserFlagShift);
}

// pk_watch phase mapping: 1=controller 2=laser 3=hydro1 4=conduction
// 5=fld_pre 6=fld_opacity 7=fld_eta 8=fld_fleck 9=fld_assemble
// 10=fld_thomas 11=fld_publish 12=fld_update_matter 13=fld_reduce
// 14=fld_post 15=hydro2 16=ring 17=exit. Phase-prof laser substages:
// 20=power_eval+entry 21=mesh_prep 22=map+prep chain 23=ray_init
// 24=trace march 25=tally_reduce 26=fold_replay+deposit 27=inject_sources.
enum PkWatchPhase {
  kPkWatchPhaseController = 1,
  kPkWatchPhaseLaser = 2,
  kPkWatchPhaseHydro1 = 3,
  kPkWatchPhaseConduction = 4,
  kPkWatchPhaseFldPre = 5,
  kPkWatchPhaseFldOpacity = 6,
  kPkWatchPhaseFldEta = 7,
  kPkWatchPhaseFldFleck = 8,
  kPkWatchPhaseFldAssemble = 9,
  kPkWatchPhaseFldThomas = 10,
  kPkWatchPhaseFldPublish = 11,
  kPkWatchPhaseFldUpdateMatter = 12,
  kPkWatchPhaseFldReduce = 13,
  kPkWatchPhaseFldPost = 14,
  kPkWatchPhaseHydro2 = 15,
  kPkWatchPhaseRing = 16,
  kPkWatchPhaseExit = 17,
  kPkWatchPhaseLaserPowerEntry = 20,
  kPkWatchPhaseLaserMeshPrep = 21,
  kPkWatchPhaseLaserMapPrep = 22,
  kPkWatchPhaseLaserRayInit = 23,
  kPkWatchPhaseLaserTraceMarch = 24,
  kPkWatchPhaseLaserTallyReduce = 25,
  kPkWatchPhaseLaserFoldReplayDeposit = 26,
  kPkWatchPhaseLaserInjectSources = 27,
};

__device__ inline int pk_thread_id(const PersistentParams& p) {
  return (p.use_cooperative != 0) ? (blockIdx.x * blockDim.x + threadIdx.x)
                                  : threadIdx.x;
}

__device__ inline int pk_thread_stride(const PersistentParams& p) {
  return (p.use_cooperative != 0) ? (gridDim.x * blockDim.x) : blockDim.x;
}

__device__ inline bool pk_global_leader(const PersistentParams& p) {
  return threadIdx.x == 0 && (p.use_cooperative == 0 || blockIdx.x == 0);
}

__device__ inline void pk_phase_prof_mark(const PersistentParams& p,
                                          const int phase_id,
                                          int* last_phase,
                                          unsigned long long* last_clock) {
  if (p.phase_prof == 0 || p.phase_prof_slab == nullptr ||
      last_phase == nullptr || last_clock == nullptr ||
      !pk_global_leader(p)) {
    return;
  }
  const unsigned long long now = clock64();
  const int prev_phase = *last_phase;
  if (prev_phase >= 0 && prev_phase < kPkPhaseProfFldOuterIterations) {
    p.phase_prof_slab[prev_phase] +=
        static_cast<double>(now - *last_clock);
  }
  *last_phase = phase_id;
  *last_clock = now;
}

__device__ inline void pk_watch_mark(const PersistentParams& p,
                                     const int phase_id,
                                     const int local_step,
                                     const int fld_iter,
                                     const int aux) {
  if (p.progress != nullptr && pk_global_leader(p)) {
    p.progress[0] = phase_id;
    p.progress[1] = local_step;
    p.progress[2] = fld_iter;
    p.progress[3] = aux;
    __threadfence_system();
  }
}

__device__ inline void pk_sync(const PersistentParams& p) {
  if (p.use_cooperative != 0) {
    if (p.phase_prof != 0 && p.phase_prof_slab != nullptr &&
        pk_global_leader(p)) {
      p.phase_prof_slab[kPkPhaseProfGridSyncCount] += 1.0;
    }
    cooperative_groups::this_grid().sync();
  } else {
    __syncthreads();
  }
}

__device__ inline int pk_laser_error_mask_from_flags(
    const core::DeviceErrorFlags* flags) {
  if (flags == nullptr) {
    return 0;
  }
  int mask = 0;
  if (*(volatile const std::int32_t*)&flags->nan_particle != 0) {
    mask |= kPkLaserFlagNanParticle;
  }
  if (*(volatile const std::int32_t*)&flags->invalid_cell != 0) {
    mask |= kPkLaserFlagInvalidCell;
  }
  if (*(volatile const std::int32_t*)&flags->invalid_boundary != 0) {
    mask |= kPkLaserFlagInvalidBoundary;
  }
  if (*(volatile const std::int32_t*)&flags->pool_overflow != 0) {
    mask |= kPkLaserFlagPoolOverflow;
  }
  if (*(volatile const std::int32_t*)&flags->opacity_out_of_range != 0) {
    mask |= kPkLaserFlagOpacityOutOfRange;
  }
  if (*(volatile const std::int32_t*)&flags->infinite_loop != 0) {
    mask |= kPkLaserFlagInfiniteLoop;
  }
  if (*(volatile const std::int32_t*)&flags->ddmc_sigma_tot_zero != 0) {
    mask |= kPkLaserFlagDdmcSigmaTotZero;
  }
  if (*(volatile const std::int32_t*)&flags->roulette_kill != 0) {
    mask |= kPkLaserFlagRouletteKill;
  }
  return mask;
}

__device__ inline void pk_raise_error(int* error_flag,
                                      const int encoded_error) {
  atomicCAS(error_flag, 0, encoded_error);
  __threadfence();
}

__device__ inline double pk_reduce_sum(const PersistentParams& p,
                                       const PersistentDeviceBuffers& b,
                                       double v,
                                       double* smem) {
  if (p.use_cooperative != 0) {
    return core::grid_reduce_sum_fixed_order<kBlockSize>(
        v, smem, b.grid_reduce_partials, b.grid_reduce_scalar,
        p.cooperative_grid_blocks);
  }
  return core::block_reduce_sum_fixed_order<kBlockSize>(v, smem);
}

__device__ inline double pk_reduce_max(const PersistentParams& p,
                                       const PersistentDeviceBuffers& b,
                                       double v,
                                       double* smem) {
  if (p.use_cooperative != 0) {
    return core::grid_reduce_max_fixed_order<kBlockSize>(
        v, smem, b.grid_reduce_partials, b.grid_reduce_scalar,
        p.cooperative_grid_blocks);
  }
  return core::block_reduce_max_fixed_order<kBlockSize>(v, smem);
}

__device__ inline void pk_reduce_argmin(const PersistentParams& p,
                                        const PersistentDeviceBuffers& b,
                                        double v,
                                        int idx,
                                        double* smem_v,
                                        int* smem_i,
                                        double* out_v,
                                        int* out_i) {
  if (p.use_cooperative != 0) {
    core::grid_reduce_argmin_fixed_order<kBlockSize>(
        v, idx, smem_v, smem_i, b.grid_reduce_partials,
        b.grid_reduce_indices, b.grid_reduce_scalar,
        b.grid_reduce_index_scalar, p.cooperative_grid_blocks, out_v, out_i);
  } else {
    core::block_reduce_argmin_fixed_order<kBlockSize>(
        v, idx, smem_v, smem_i, out_v, out_i);
  }
}

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

const char* pk_error_reason_name(const int reason) {
  switch (reason) {
    case kPkErrorLaserMeshPrep:
      return "laser mesh prep failed";
    case kPkErrorLaserRayFlags:
      return "laser ray error flags";
    case kPkErrorHydroNonPositiveVolume:
      return "hydro non-positive volume";
    case kPkErrorFldDivergence:
      return "FLD divergence";
    case kPkErrorConduction:
      return "conduction";
    default:
      return "unknown";
  }
}

void pk_append_laser_flag_name(std::string& out,
                               const int mask,
                               const int bit,
                               const char* name) {
  if ((mask & bit) == 0) {
    return;
  }
  if (!out.empty()) {
    out += "|";
  }
  out += name;
}

std::string pk_laser_flag_names(const int mask) {
  std::string out;
  pk_append_laser_flag_name(out, mask, kPkLaserFlagNanParticle,
                            "nan_particle");
  pk_append_laser_flag_name(out, mask, kPkLaserFlagInvalidCell,
                            "invalid_cell");
  pk_append_laser_flag_name(out, mask, kPkLaserFlagInvalidBoundary,
                            "invalid_boundary");
  pk_append_laser_flag_name(out, mask, kPkLaserFlagPoolOverflow,
                            "pool_overflow");
  pk_append_laser_flag_name(out, mask, kPkLaserFlagOpacityOutOfRange,
                            "opacity_out_of_range");
  pk_append_laser_flag_name(out, mask, kPkLaserFlagInfiniteLoop,
                            "infinite_loop");
  pk_append_laser_flag_name(out, mask, kPkLaserFlagDdmcSigmaTotZero,
                            "ddmc_sigma_tot_zero");
  pk_append_laser_flag_name(out, mask, kPkLaserFlagRouletteKill,
                            "roulette_kill");
  return out.empty() ? std::string("none") : out;
}

std::string pk_error_description(const int error_code) {
  const int reason = error_code & kPkErrorReasonMask;
  const int laser_flags = error_code >> kPkErrorLaserFlagShift;
  std::string out = std::string("reason=") + pk_error_reason_name(reason) +
                    " code=" + std::to_string(reason);
  if (reason == kPkErrorLaserRayFlags) {
    out += " laser_flags=" + pk_laser_flag_names(laser_flags);
    out += " mask=0x";
    char hex[16];
    std::snprintf(hex, sizeof(hex), "%02x", laser_flags & 0xff);
    out += hex;
  }
  out += " packed_error_code=" + std::to_string(error_code);
  return out;
}

struct PkWatchProgressSlab {
  int* host = nullptr;
  volatile int* device = nullptr;
};

bool pk_watch_enabled() {
  return std::getenv("TENRYU_PK_WATCH") != nullptr;
}

PkWatchProgressSlab& pk_watch_progress_slab() {
  static PkWatchProgressSlab slab;
  if (slab.host == nullptr) {
    void* host_ptr = nullptr;
    cuda_check(cudaHostAlloc(&host_ptr,
                             kPkWatchProgressInts * sizeof(int),
                             cudaHostAllocMapped),
               "persistent_loop: pk_watch progress cudaHostAlloc failed");
    slab.host = static_cast<int*>(host_ptr);
    void* device_ptr = nullptr;
    cuda_check(cudaHostGetDevicePointer(&device_ptr, slab.host, 0),
               "persistent_loop: pk_watch progress cudaHostGetDevicePointer "
               "failed");
    slab.device = static_cast<volatile int*>(device_ptr);
  }
  std::fill(slab.host, slab.host + kPkWatchProgressInts, 0);
  return slab;
}

void pk_watch_wait_for_kernel(const volatile int* progress) {
  int poll_iterations = 0;
  while (true) {
    const cudaError_t query = cudaStreamQuery(nullptr);
    if (query == cudaSuccess) {
      break;
    }
    if (query != cudaErrorNotReady) {
      cuda_check(query,
                 "persistent_loop: persistent_chunk_kernel execution failed");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ++poll_iterations;
    if (poll_iterations % 10 == 0) {
      char buffer[128];
      std::snprintf(buffer,
                    sizeof(buffer),
                    "pk_watch phase=%d step=%d fld_iter=%d aux=%d",
                    progress[0],
                    progress[1],
                    progress[2],
                    progress[3]);
      core::log_info(buffer);
    }
  }
}

const char* pk_phase_prof_name(const int phase_id) {
  switch (phase_id) {
    case kPkWatchPhaseController:
      return "controller";
    case kPkWatchPhaseLaser:
      return "laser";
    case kPkWatchPhaseHydro1:
      return "hydro_half_1";
    case kPkWatchPhaseConduction:
      return "conduction";
    case kPkWatchPhaseFldPre:
      return "fld_pre";
    case kPkWatchPhaseFldOpacity:
      return "fld_opacity";
    case kPkWatchPhaseFldEta:
      return "fld_eta";
    case kPkWatchPhaseFldFleck:
      return "fld_fleck";
    case kPkWatchPhaseFldAssemble:
      return "fld_assemble";
    case kPkWatchPhaseFldThomas:
      return "fld_thomas";
    case kPkWatchPhaseFldPublish:
      return "fld_publish";
    case kPkWatchPhaseFldUpdateMatter:
      return "fld_update_matter";
    case kPkWatchPhaseFldReduce:
      return "fld_reduce";
    case kPkWatchPhaseFldPost:
      return "fld_post";
    case kPkWatchPhaseHydro2:
      return "hydro_half_2";
    case kPkWatchPhaseRing:
      return "ring";
    case kPkWatchPhaseExit:
      return "exit";
    case kPkWatchPhaseLaserPowerEntry:
      return "laser_power_eval_entry";
    case kPkWatchPhaseLaserMeshPrep:
      return "laser_mesh_prep";
    case kPkWatchPhaseLaserMapPrep:
      return "laser_map_prep_chain";
    case kPkWatchPhaseLaserRayInit:
      return "laser_ray_init";
    case kPkWatchPhaseLaserTraceMarch:
      return "laser_trace_march";
    case kPkWatchPhaseLaserTallyReduce:
      return "laser_tally_reduce";
    case kPkWatchPhaseLaserFoldReplayDeposit:
      return "laser_fold_replay_deposit";
    case kPkWatchPhaseLaserInjectSources:
      return "laser_inject_sources";
    default:
      return "unknown";
  }
}

void pk_phase_prof_log_phase(const double* slab,
                             const int phase,
                             const double total_cycles) {
  const double cycles = slab[phase];
  const double percent =
      (total_cycles > 0.0) ? (100.0 * cycles / total_cycles) : 0.0;
  char line[192];
  std::snprintf(line,
                sizeof(line),
                "pk_phase_prof phase=%s cycles=%.0f pct=%.3f",
                pk_phase_prof_name(phase),
                cycles,
                percent);
  core::log_info(line);
}

void pk_phase_prof_log(const double* slab) {
  double total_cycles = 0.0;
  for (int phase = 1; phase < kPkPhaseProfFldOuterIterations; ++phase) {
    total_cycles += slab[phase];
  }
  const double steps = slab[kPkPhaseProfStepCount];
  const double outers = slab[kPkPhaseProfFldOuterIterations];
  const double outers_per_step = (steps > 0.0) ? (outers / steps) : 0.0;
  char header[224];
  std::snprintf(header,
                sizeof(header),
                "pk_phase_prof total_cycles=%.0f steps=%.0f "
                "fld_outer_iterations=%.0f outers_per_step=%.3f "
                "grid_sync=%.0f",
                total_cycles,
                steps,
                outers,
                outers_per_step,
                slab[kPkPhaseProfGridSyncCount]);
  core::log_info(header);
  for (int phase = kPkWatchPhaseController; phase <= kPkWatchPhaseExit;
       ++phase) {
    pk_phase_prof_log_phase(slab, phase, total_cycles);
  }
  for (int phase = kPkWatchPhaseLaserPowerEntry;
       phase <= kPkWatchPhaseLaserInjectSources; ++phase) {
    pk_phase_prof_log_phase(slab, phase, total_cycles);
  }
}

bool warn_unsupported_once(const std::string& reason) {
  static std::vector<std::string> warned;
  if (std::find(warned.begin(), warned.end(), reason) == warned.end()) {
    warned.push_back(reason);
    core::log_warning("persistent_loop C1/C2b/C3 unsupported: " + reason);
  }
  return false;
}

bool brag_env_enabled() {
  const char* value = std::getenv("TENRYU_BRAG_ENABLE");
  return value != nullptr && std::atoi(value) != 0;
}

bool pk_chunk_trace_enabled() {
  static const bool trace = [] {
    const char* v = std::getenv("TENRYU_PK_CHUNK_TRACE");
    return v && std::string(v) == "1";
  }();
  return trace;
}

bool frozen_tables_equal(const core::namelist::FrozenTable1D& a,
                         const core::namelist::FrozenTable1D& b) {
  return a.n_points == b.n_points &&
         a.x_min == b.x_min &&
         a.x_max == b.x_max &&
         a.zero_outside == b.zero_outside &&
         a.x == b.x &&
         a.y == b.y;
}

bool persistent_materials_supported(const core::State& state,
                                    const core::Config& cfg,
                                    const bool radiation_enabled) {
  const auto& materials = cfg.materials.materials;
  if (materials.empty()) {
    return warn_unsupported_once("material count == 0");
  }
  if (cfg.numerics.materials.per_material_conservation_enabled) {
    return warn_unsupported_once("per-material conservation enabled");
  }
  if (materials.size() > 1U) {
    const std::size_t expected = state.rho.size() * materials.size();
    if (state.volFrac.size() != expected) {
      return warn_unsupported_once("multi-material volFrac size mismatch");
    }
  }

  const auto& ref = materials.front();
  for (const auto& mat : materials) {
    if (mat.is_void) {
      continue;
    }
    if (mat.eos_model != "ideal_gas" || mat.hydro_eos_backend != "legacy") {
      return warn_unsupported_once("non-legacy ideal-gas EOS");
    }
    if (materials.size() > 1U &&
        (mat.cv_e_override != ref.cv_e_override ||
         mat.eos_T_ref_eV != ref.eos_T_ref_eV)) {
      return warn_unsupported_once(
          "multi-material cv_e_override/eos_T_ref_eV differ");
    }
    if (radiation_enabled && materials.size() > 1U) {
      if (mat.opacity_model != "constant") {
        return warn_unsupported_once(
            "multi-material non-constant opacity not plumbed");
      }
      if (!(mat.kappa_a_constant > 0.0)) {
        return warn_unsupported_once(
            "multi-material constant opacity kappa_a_constant <= 0");
      }
    }
  }
  return true;
}

int persistent_fld_outer_bc_id(const std::string& value) {
  if (value == "vacuum") {
    return radiation::fld_1d_bodies::kFld1dOuterVacuum;
  }
  if (value == "reflect" || value == "reflective") {
    return radiation::fld_1d_bodies::kFld1dOuterReflect;
  }
  if (value == "marshak") {
    return radiation::fld_1d_bodies::kFld1dOuterMarshak;
  }
  TENRYU_ASSERT(false, "persistent_loop: unsupported FLD 1D outer boundary");
  return radiation::fld_1d_bodies::kFld1dOuterVacuum;
}

int first_trailing_void_cell(const core::State& state) {
  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0 ||
      state.cell_is_void.size() != static_cast<std::size_t>(n_cells)) {
    return n_cells;
  }
  int first_void = n_cells;
  for (int c = n_cells - 1; c >= 0; --c) {
    if (state.cell_is_void[static_cast<std::size_t>(c)] != 0U) {
      first_void = c;
    } else {
      break;
    }
  }
  return first_void;
}

int persistent_fld_limiter_id(const std::string& limiter) {
  if (limiter == "larsen") {
    return 1;
  }
  if (limiter == "none") {
    return 2;
  }
  return 0;
}

template <typename T>
T* acquire_device_buffer(const char* tag, const std::size_t n) {
  if (n == 0) {
    return nullptr;
  }
  return static_cast<T*>(core::device_scratch_acquire(tag, n * sizeof(T)));
}

std::vector<PersistentMaterialParams> make_persistent_material_params(
    const core::Config& cfg) {
  std::vector<PersistentMaterialParams> params(cfg.materials.materials.size());
  for (std::size_t m = 0; m < cfg.materials.materials.size(); ++m) {
    const auto& mat = cfg.materials.materials[m];
    PersistentMaterialParams p{};
    p.A = std::max(mat.A, 1.0e-12);
    p.gamma = std::max(mat.ideal_gas_gamma, 1.0 + 1.0e-12);
    if (cfg.materials.zbar.model == "fixed" &&
        cfg.materials.zbar.fixed_value >= 0.0) {
      p.Zbar = cfg.materials.zbar.fixed_value;
    } else {
      p.Zbar = (mat.Z > 0.0) ? mat.Z : 1.0;
    }
    p.cv_e_override = mat.cv_e_override;
    p.eos_T_ref_eV = mat.eos_T_ref_eV;
    p.kappa_a_constant = std::max(0.0, mat.kappa_a_constant);
    p.is_void = mat.is_void ? 1 : 0;
    if (p.is_void != 0) {
      p.Zbar = 0.0;
      p.kappa_a_constant = 0.0;
    }
    params[m] = p;
  }
  return params;
}

__device__ double persistent_constant_opacity_kappa_a(
    const PersistentParams& p,
    const PersistentDeviceBuffers& b,
    const int c) {
  if (b.cell_is_void != nullptr && b.cell_is_void[c] != 0U) {
    return 0.0;
  }
  if (p.n_materials <= 1 || b.volFrac == nullptr ||
      b.material_params == nullptr) {
    return fmax(p.fld_kappa_a, 0.0);
  }
  const int base = c * p.n_materials;
  double frac_sum = 0.0;
  double kappa = 0.0;
  for (int m = 0; m < p.n_materials; ++m) {
    const PersistentMaterialParams mp = b.material_params[m];
    if (mp.is_void != 0) {
      continue;
    }
    const double frac_raw = b.volFrac[base + m];
    const double frac =
        (isfinite(frac_raw) && frac_raw > 0.0) ? frac_raw : 0.0;
    frac_sum += frac;
    kappa += frac * fmax(mp.kappa_a_constant, 0.0);
  }
  if (frac_sum > 1.0e-30 && isfinite(kappa)) {
    return fmax(kappa / frac_sum, 0.0);
  }
  return fmax(p.fld_kappa_a, 0.0);
}

__device__ void persistent_eval_constant_opacity_cell(
    const PersistentParams& p,
    const PersistentDeviceBuffers& b,
    const int c) {
  const double rho_c = b.rho[c];
  const double rho_safe = (rho_c >= 0.0) ? rho_c : 0.0;
  const double kappa = persistent_constant_opacity_kappa_a(p, b, c);
  const double sigma_min = rho_safe * fmax(p.fld_opacity_floor, 0.0);
  const double sigma_max = rho_safe * fmax(p.fld_opacity_cap, 0.0);
  const double sigma =
      materials::apply_sigma_bounds(rho_safe * kappa, sigma_min, sigma_max);
  const int base = c * p.n_groups;
  for (int g = 0; g < p.n_groups; ++g) {
    b.fld_sigma_a[base + g] = sigma;
    b.fld_sigma_R[base + g] = sigma;
  }
}

__device__ void refresh_geometry_density_closure(const PersistentParams& p,
                                                 const PersistentDeviceBuffers& b) {
  using namespace tenryu::hydro::persistent_1d;
  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    recompute_geometry_1d_kernel_body(c, b.vol, b.cell_centroid_r,
                                      b.cell_centroid_z, b.x_r, p.n_cells,
                                      p.geom_code);
  }
  pk_sync(p);

  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    compute_density_kernel_body(c, b.rho, b.mass, b.vol, b.cell_is_void,
                                p.n_cells, p.rho_floor, b.rho_clamp_count);
  }
  pk_sync(p);

  if (p.two_temperature != 0) {
    const tenryu::materials::DeviceEOSTableView no_table{};
    const tenryu::materials::EOSRhoEDeviceView no_rho_e_table{};
    const tenryu::materials::HelmholtzSplineDeviceView no_spline{};
    const tenryu::materials::HelmholtzJetDeviceView no_jet{};
    const tenryu::materials::MieGruneisenDeviceView no_mie_gruneisen{};
    for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
      enforce_2t_closure_kernel_body(
          c, b.ee, b.ei, b.Te, b.Ti, b.Pe, b.Pi, b.rho, b.zbar, p.n_cells,
          b.gamma_eff, b.A_eff, p.fallback_z, p.cv_e_override, p.Te_floor,
          p.Ti_floor, no_table, no_table, no_rho_e_table, no_spline, no_jet,
          no_mie_gruneisen, false, false, false, false, false, false, false,
          p.energy_authoritative != 0, nullptr, nullptr, kExactOverrideNone,
          b.cv_e, b.cv_i);
    }
  } else {
    for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
      enforce_1t_closure_ideal_kernel_body(
          c, b.ee, b.ei, b.Te, b.Ti, b.Pe, b.Pi, b.rho, b.zbar, b.gamma_eff,
          b.A_eff, p.fallback_z, p.cv_e_override, p.Te_floor, b.cv_e, b.cv_i);
    }
  }
  pk_sync(p);
}

__device__ void compute_sound_speed_into(const PersistentParams& p,
                                         const PersistentDeviceBuffers& b,
                                         double* out_cs) {
  using namespace tenryu::hydro::persistent_1d;
  if (p.two_temperature != 0) {
    const tenryu::materials::DeviceEOSTableView no_table{};
    const tenryu::materials::EOSRhoEDeviceView no_rho_e_table{};
    const tenryu::materials::HelmholtzSplineDeviceView no_spline{};
    const tenryu::materials::HelmholtzJetDeviceView no_jet{};
    const tenryu::materials::MieGruneisenDeviceView no_mie_gruneisen{};
    for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
      compute_sound_speed_2t_kernel_body(
          c, out_cs, b.rho, b.ee, b.ei, b.Pe, b.Pi, b.Te, b.Ti, no_table,
          no_table, no_rho_e_table, no_spline, no_jet, no_mie_gruneisen,
          b.cv_i, b.cv_e, p.n_cells, false, false, false, false, false,
          kExactOverrideNone, b.gamma_eff);
    }
  } else {
    for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
      compute_sound_speed_1t_ideal_kernel_body(c, out_cs, b.ee, b.gamma_eff);
    }
  }
  pk_sync(p);
}

__device__ void compute_q_vnr_into(const PersistentParams& p,
                                   const PersistentDeviceBuffers& b,
                                   const double* cs,
                                   double* out_q) {
  using namespace tenryu::hydro::persistent_1d;
  for (int j = pk_thread_id(p); j < p.n_nodes; j += pk_thread_stride(p)) {
    compute_node_sigma_1d_kernel_body(j, b.sigma, b.x_r, b.v_r, p.n_nodes,
                                      p.av_limiter_J);
  }
  pk_sync(p);

  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    switch (p.geom_code) {
      case 1:
        compute_q_1d_kernel_body<1>(
            c, out_q, b.rho, b.vol, b.x_r, b.v_r, b.Pe, b.Pi, cs, b.sigma,
            nullptr, nullptr, nullptr, b.hydro_active, p.n_cells, p.av_linear,
            p.av_quadratic, nullptr, nullptr, false, p.gamma, 1.0);
        break;
      case 2:
        compute_q_1d_kernel_body<2>(
            c, out_q, b.rho, b.vol, b.x_r, b.v_r, b.Pe, b.Pi, cs, b.sigma,
            nullptr, nullptr, nullptr, b.hydro_active, p.n_cells, p.av_linear,
            p.av_quadratic, nullptr, nullptr, false, p.gamma, 1.0);
        break;
      default:
        compute_q_1d_kernel_body<0>(
            c, out_q, b.rho, b.vol, b.x_r, b.v_r, b.Pe, b.Pi, cs, b.sigma,
            nullptr, nullptr, nullptr, b.hydro_active, p.n_cells, p.av_linear,
            p.av_quadratic, nullptr, nullptr, false, p.gamma, 1.0);
        break;
    }
  }
  pk_sync(p);
}

__device__ double ghost_pq_device(const PersistentParams& p,
                                  const PersistentDeviceBuffers& b) {
  const int last = p.n_cells - 1;
  if (last < 0) {
    return 0.0;
  }
  if (p.boundary_type == static_cast<int>(hydro::HydroBoundaryType::FREE)) {
    return b.Qvisc[last];
  }
  return b.Pe[last] + b.Pi[last] + b.Qvisc[last];
}

__device__ double block_total_energy_1d(const PersistentParams& p,
                                        const PersistentDeviceBuffers& b,
                                        double* smem) {
  double local = 0.0;
  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    const double u = 0.5 * (b.v_r[c] + b.v_r[c + 1]);
    local += b.rho[c] * (b.ee[c] + b.ei[c]) * b.vol[c] +
             0.5 * b.mass[c] * u * u;
  }
  return pk_reduce_sum(p, b, local, smem);
}

__device__ double block_active_mass_1d(const PersistentParams& p,
                                       const PersistentDeviceBuffers& b,
                                       double* smem) {
  double local = 0.0;
  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    const bool active =
        (b.hydro_active == nullptr) || (b.hydro_active[c] != 0);
    if (active) {
      local += b.mass[c];
    }
  }
  return pk_reduce_sum(p, b, local, smem);
}

__device__ double block_active_internal_1d(const PersistentParams& p,
                                           const PersistentDeviceBuffers& b,
                                           double* smem) {
  double local = 0.0;
  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    const bool active =
        (b.hydro_active == nullptr) || (b.hydro_active[c] != 0);
    if (active) {
      local += b.mass[c] * fmax(b.ee[c], 0.0);
    }
  }
  return pk_reduce_sum(p, b, local, smem);
}

__device__ void scan_persistent_error(const PersistentParams& p,
                                      const PersistentDeviceBuffers& b,
                                      int* error_flag) {
  using namespace tenryu::hydro::persistent_1d;
  if (pk_global_leader(p)) {
    *b.failing_cell = p.n_cells;
  }
  __threadfence();
  pk_sync(p);

  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    find_nonpositive_volume_1d_kernel_body(c, b.vol, b.failing_cell,
                                           p.n_cells);
  }
  __threadfence();
  pk_sync(p);
  if (threadIdx.x == 0) {
    const int failing_cell = *(volatile const int*)b.failing_cell;
    if (failing_cell < p.n_cells) {
      pk_raise_error(
          error_flag,
          pk_encode_error_code(kPkErrorHydroNonPositiveVolume));
    }
  }
  pk_sync(p);
}

__device__ double persistent_laser_cell_nhat_raw(
    const PersistentParams& p,
    const PersistentDeviceBuffers& b,
    const int c) {
  const double rho = fmax(b.rho[c], 0.0);
  const double z = (b.zbar != nullptr) ? fmax(b.zbar[c], 0.0)
                                       : fmax(p.fallback_z, 0.0);
  const double A = (b.A_eff != nullptr) ? fmax(b.A_eff[c], 1.0e-30)
                                        : p.material_A;
  if (!(p.laser_n_crit > 0.0) || !(A > 0.0)) {
    return 0.0;
  }
  return rho * z /
         (A * core::constants::proton_mass * p.laser_n_crit);
}

__device__ void persistent_laser_radial_absorption_single_trace(
    const PersistentParams& p,
    const PersistentDeviceBuffers& b,
    const double P_beam,
    double* single_unabsorbed,
    unsigned long long* critical_hits,
    int* error_flag) {
  if (!pk_global_leader(p)) {
    return;
  }

  *single_unabsorbed = 0.0;
  *critical_hits = 0ULL;
  if (!isfinite(P_beam) || !(P_beam > 0.0)) {
    if (!isfinite(P_beam)) {
      pk_raise_error(
          error_flag,
          pk_encode_error_code(kPkErrorLaserRayFlags,
                               kPkLaserFlagNanParticle));
    }
    return;
  }
  if (b.x_r == nullptr || b.laser_dep == nullptr || p.n_cells <= 0) {
    *single_unabsorbed = P_beam;
    return;
  }

  double P = P_beam;
  const double nh_crit = 1.0 - p.laser_eps_crit;
  const double cutoff_power =
      (isfinite(p.laser_intensity_cutoff) && p.laser_intensity_cutoff > 0.0)
          ? p.laser_intensity_cutoff * P_beam
          : -1.0;

  for (int c = p.n_cells - 1; c >= 0; --c) {
    if (cutoff_power > 0.0 && P < cutoff_power) {
      *single_unabsorbed += P;
      return;
    }

    const double r0 = b.x_r[c];
    const double r1 = b.x_r[c + 1];
    const double dr = r1 - r0;
    if (!isfinite(r0) || !isfinite(r1) || !(dr > 0.0)) {
      *single_unabsorbed += P;
      return;
    }

    const double nh_raw = persistent_laser_cell_nhat_raw(p, b, c);
    const double nh = fmin(1.0, fmax(0.0, nh_raw));
    if (!isfinite(nh) || !isfinite(nh_raw)) {
      *single_unabsorbed += P;
      return;
    }

    if (nh_raw >= nh_crit) {
      ++(*critical_hits);
      *single_unabsorbed += P;
      return;
    }

    double kappa = 0.0;
    if (p.laser_test_kappa > 0.0) {
      kappa = p.laser_test_kappa;
    } else {
      const double Te = (b.Te != nullptr) ? b.Te[c] : p.Te_floor;
      const double z = (b.zbar != nullptr) ? fmax(b.zbar[c], 0.0)
                                           : fmax(p.fallback_z, 0.0);
      const double smooth_factor = laser::compute_kappa_smooth_factor(
          nh, Te, z, p.laser_lambda_cm, p.laser_eps_n,
          p.laser_coulomb_log_floor);
      kappa = laser::compute_kappa_from_smooth(smooth_factor, nh,
                                               p.laser_eps_n);
    }
    if (!isfinite(kappa) || kappa < 0.0) {
      *single_unabsorbed += P;
      return;
    }

    const double tau = laser::compute_optical_depth(kappa, kappa, dr);
    if (!isfinite(tau) || tau < 0.0) {
      *single_unabsorbed += P;
      return;
    }

    double P_next = P;
    const double dP = laser::absorbed_power_expm1(P, tau, P_next);
    if (!isfinite(dP) || !isfinite(P_next) || dP < 0.0 || P_next < 0.0) {
      *single_unabsorbed += P;
      pk_raise_error(
          error_flag,
          pk_encode_error_code(kPkErrorLaserRayFlags,
                               kPkLaserFlagNanParticle));
      return;
    }

    if (dP > 0.0) {
      b.laser_dep[c] += dP;
    }
    P = P_next;
  }

  if (P > 0.0) {
    *single_unabsorbed += P;
  }
}

__device__ double persistent_laser_replay_deposit_to_energy(
    const PersistentParams& p,
    const PersistentDeviceBuffers& b,
    const double dt,
    double* smem) {
  double local_dep = 0.0;
  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    const double single_power = b.laser_dep[c];
    double total_power = 0.0;
    for (int k = 0; k < p.laser_n_folded; ++k) {
      total_power += single_power;
    }
    const double energy = total_power * dt;
    b.laser_dep[c] = energy;
    local_dep += fmax(energy, 0.0);
  }
  return pk_reduce_sum(p, b, local_dep, smem);
}

struct PersistentCriticalSurfaceEstimate1D {
  int fcrit = -1;
  double r_face = -1.0;
  double r_interp = -1.0;
};

__device__ double persistent_laser_average_outer_cells(
    const double* __restrict__ values,
    const std::uint8_t* __restrict__ cell_is_void,
    const int outer_surface_cell,
    const int span,
    const int n_cells,
    const double fallback) {
  if (values == nullptr || cell_is_void == nullptr || outer_surface_cell < 0 ||
      span <= 0) {
    return fallback;
  }
  double sum = 0.0;
  double w_sum = 0.0;
  for (int k = 0; k < span; ++k) {
    const int c = outer_surface_cell - k;
    if (c < 0 || c >= n_cells) {
      break;
    }
    if (cell_is_void[c] != 0U) {
      continue;
    }
    const double w = 1.0 / static_cast<double>(k + 1);
    sum += w * values[c];
    w_sum += w;
  }
  return (w_sum > 0.0) ? (sum / w_sum) : fallback;
}

__device__ double persistent_laser_ghost_sound_speed_cm_s(
    const double Te_eV,
    const double zbar_eff,
    const double A_eff) {
  const double Te_safe = fmax(Te_eV, 0.0);
  const double z_safe = fmax(zbar_eff, 1.0);
  const double A_safe = fmax(A_eff, 1.0e-30);
  return sqrt(z_safe * core::constants::eV_to_erg * Te_safe /
              (A_safe * core::constants::proton_mass));
}

struct PersistentAllowedSupercriticalCell1D {
  int allowed_cell = -1;
  int critical_adjacent_subcritical_cell = -1;
  double r_crit = -1.0;
  int fallback_only = 0;
};

__device__ double persistent_laser_profile_value(const PersistentParams& p,
                                                 const double R_cm) {
  const double w0 = fmax(p.laser_beam_profile_w0_cm, 1.0e-30);
  const double x = R_cm / w0;
  if (p.laser_profile_kind == 1) {
    const int m = (p.laser_profile_m > 1) ? p.laser_profile_m : 1;
    const double q = pow(fabs(x), 2.0 * static_cast<double>(m));
    return exp(-2.0 * q);
  }
  if (p.laser_profile_kind == 2) {
    return (R_cm <= w0) ? 1.0 : 0.0;
  }
  return exp(-2.0 * x * x);
}

__device__ void persistent_laser_initialize_rays_1d(
    const PersistentParams& p,
    const PersistentDeviceBuffers& b,
    const int nr,
    const double beam_power,
    double* sum_w_out) {
  constexpr double kPi = 3.14159265358979323846;
  const int n_rays = p.laser_rays_per_beam;
  if (pk_global_leader(p)) {
    b.grid_broadcast_doubles[0] = 0.0;
    b.grid_broadcast_ints[0] = 0;
  }
  if (n_rays > 0 && beam_power > 0.0) {
    const double Z_init = b.laser_node_R[nr];
    const double R_beam =
        fabs(Z_init - p.laser_beam_focus_lab_z) /
        (2.0 * fmax(p.laser_beam_f_number, 1.0e-12));
    const double dR = R_beam / static_cast<double>(n_rays);
    for (int k = pk_thread_id(p); k < n_rays; k += pk_thread_stride(p)) {
      const double Rk = dR * static_cast<double>(k);
      const double dRk = (k == 0) ? (0.5 * dR) : dR;
      const double area =
          (k == 0) ? (kPi * 0.25 * dR * dR) : (2.0 * kPi * Rk * dRk);
      b.laser_ray_weights[k] =
          persistent_laser_profile_value(p, Rk) * fmax(area, 0.0);
    }
  }
  __threadfence();
  pk_sync(p);
  if (pk_global_leader(p)) {
    if (n_rays > 0 && beam_power > 0.0) {
      double sum_w = 0.0;
      // The parallel fill uses the old per-k formula for the same weights;
      // summing cached weights in ascending k keeps the result bit-identical.
      for (int k = 0; k < n_rays; ++k) {
        sum_w += b.laser_ray_weights[k];
      }
      const int use_uniform_weights = (!(sum_w > 0.0)) ? 1 : 0;
      b.grid_broadcast_ints[0] = use_uniform_weights;
      if (use_uniform_weights != 0) {
        sum_w = static_cast<double>(n_rays);
      }
      b.grid_broadcast_doubles[0] = sum_w;
    }
  }
  __threadfence();
  pk_sync(p);
  if (*(volatile const int*)(b.grid_broadcast_ints + 0) != 0) {
    for (int k = pk_thread_id(p); k < n_rays; k += pk_thread_stride(p)) {
      b.laser_ray_weights[k] = 1.0;
    }
  }
  __threadfence();
  pk_sync(p);
  if (threadIdx.x == 0) {
    *sum_w_out = *(volatile const double*)(b.grid_broadcast_doubles + 0);
  }
  pk_sync(p);

  const double sum_w = *sum_w_out;
  if (n_rays <= 0 || !(beam_power > 0.0) || !(sum_w > 0.0)) {
    return;
  }
  const double Z_init = b.laser_node_R[nr];
  const double R_beam =
      fabs(Z_init - p.laser_beam_focus_lab_z) /
      (2.0 * fmax(p.laser_beam_f_number, 1.0e-12));
  const double dR = R_beam / static_cast<double>(n_rays);
  for (int k = pk_thread_id(p); k < n_rays; k += pk_thread_stride(p)) {
    const double Rk = dR * static_cast<double>(k);
    const double dz = Z_init - p.laser_beam_focus_lab_z;
    const double L = sqrt(Rk * Rk + dz * dz);
    const double vR = (L > 0.0) ? (-Rk / L) : 0.0;
    const double sgn = (dz >= 0.0) ? -1.0 : 1.0;
    const double vR2 = vR * vR;
    const double I = beam_power * (b.laser_ray_weights[k] / sum_w);
    b.laser_ray_R0[k] = Rk;
    b.laser_ray_Z0[k] = Z_init;
    b.laser_ray_vR0[k] = vR;
    b.laser_ray_vZ0[k] = sgn * sqrt(fmax(0.0, 1.0 - vR2));
    b.laser_ray_power[k] = I;
    b.laser_ray_power0[k] = I;
  }
  pk_sync(p);
}

__device__ int persistent_laser_geometric_width_count(double L,
                                                      double d0,
                                                      double g) {
  if (!(L > 0.0)) {
    return 0;
  }
  d0 = fmax(d0, 1.0e-12);
  g = fmax(g, 1.0001);
  int n = 0;
  double used = 0.0;
  double d = d0;
  double last = 0.0;
  while (used + d < L) {
    last = d;
    used += d;
    d *= g;
    ++n;
  }
  const double tail = L - used;
  if (tail > 0.0) {
    if (n > 0 && tail < 0.35 * last) {
      return n;
    }
    return n + 1;
  }
  return n;
}

__device__ int persistent_laser_uniform_width_count(const double L,
                                                    const double d_target) {
  if (!(L > 0.0)) {
    return 0;
  }
  const int n = static_cast<int>(ceil(L / fmax(d_target, 1.0e-12)));
  return (n > 1) ? n : 1;
}

__device__ int persistent_laser_graded_count(const double R_crit,
                                             const double dR_fine,
                                             const double R_max,
                                             const double scale) {
  constexpr int kFineSideTarget = 64;
  constexpr double kGradedCoreRatio = 1.08;
  constexpr double kGradedCoronaRatio = 1.05;
  const double R_max_s = fmax(R_max, 1.0e-12);
  const double R_crit_s = fmin(R_max_s, fmax(0.0, R_crit));
  const double dF = fmax(dR_fine, 1.0e-12) * fmax(scale, 1.0);
  const double delta_target = static_cast<double>(kFineSideTarget) * dF;
  const double delta_in = fmin(delta_target, R_crit_s);
  const double delta_out = fmin(delta_target, R_max_s - R_crit_s);
  const double R_left = R_crit_s - delta_in;
  const double R_right = R_crit_s + delta_out;
  int count = 0;
  count += persistent_laser_geometric_width_count(R_left, dF, kGradedCoreRatio);
  count += persistent_laser_uniform_width_count(R_crit_s - R_left, dF);
  count += persistent_laser_uniform_width_count(R_right - R_crit_s, dF);
  count += persistent_laser_geometric_width_count(R_max_s - R_right, dF,
                                                  kGradedCoronaRatio);
  return (count > 4) ? count : 4;
}

__device__ int persistent_laser_append_geometric_widths(double* widths,
                                                        int offset,
                                                        const double L,
                                                        double d0,
                                                        double g) {
  if (!(L > 0.0)) {
    return offset;
  }
  d0 = fmax(d0, 1.0e-12);
  g = fmax(g, 1.0001);
  double used = 0.0;
  double d = d0;
  while (used + d < L) {
    widths[offset++] = d;
    used += d;
    d *= g;
  }
  const double tail = L - used;
  if (tail > 0.0) {
    if (offset > 0 && tail < 0.35 * widths[offset - 1]) {
      widths[offset - 1] += tail;
    } else {
      widths[offset++] = tail;
    }
  }
  return offset;
}

__device__ int persistent_laser_append_uniform_widths(double* widths,
                                                      int offset,
                                                      const double L,
                                                      const double d_target) {
  if (!(L > 0.0)) {
    return offset;
  }
  const int n_raw = static_cast<int>(ceil(L / fmax(d_target, 1.0e-12)));
  const int n = (n_raw > 1) ? n_raw : 1;
  const double d = L / static_cast<double>(n);
  for (int k = 0; k < n; ++k) {
    widths[offset++] = d;
  }
  return offset;
}

__device__ int persistent_laser_build_graded_nodes_1d(
    const double R_crit,
    const double dR_fine,
    const double R_max,
    const int nr_max,
    double* node_R,
    double* node_Z,
    double* widths) {
  constexpr int kFineSideTarget = 64;
  constexpr double kGradedCoreRatio = 1.08;
  constexpr double kGradedCoronaRatio = 1.05;
  const double R_max_s = fmax(R_max, 1.0e-12);
  const double R_crit_s = fmin(R_max_s, fmax(0.0, R_crit));
  const double dF_base = fmax(dR_fine, 1.0e-12);
  const int nr_max_s = (nr_max > 4) ? nr_max : 4;
  double scale = 1.0;
  if (persistent_laser_graded_count(R_crit_s, dF_base, R_max_s, scale) >
      nr_max_s) {
    double lo = 1.0;
    double hi = 2.0;
    while (persistent_laser_graded_count(R_crit_s, dF_base, R_max_s, hi) >
           nr_max_s) {
      hi *= 2.0;
    }
    for (int it = 0; it < 40; ++it) {
      const double mid = 0.5 * (lo + hi);
      if (persistent_laser_graded_count(R_crit_s, dF_base, R_max_s, mid) >
          nr_max_s) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    scale = hi;
  }

  const double dF = dF_base * fmax(scale, 1.0);
  const double delta_target = static_cast<double>(kFineSideTarget) * dF;
  const double delta_in = fmin(delta_target, R_crit_s);
  const double delta_out = fmin(delta_target, R_max_s - R_crit_s);
  const double R_left = R_crit_s - delta_in;
  const double R_right = R_crit_s + delta_out;
  int n = 0;
  n = persistent_laser_append_geometric_widths(widths, n, R_left, dF,
                                               kGradedCoreRatio);
  for (int a = 0, z = n - 1; a < z; ++a, --z) {
    const double tmp = widths[a];
    widths[a] = widths[z];
    widths[z] = tmp;
  }
  n = persistent_laser_append_uniform_widths(widths, n, R_crit_s - R_left,
                                             dF);
  n = persistent_laser_append_uniform_widths(widths, n, R_right - R_crit_s,
                                             dF);
  n = persistent_laser_append_geometric_widths(widths, n, R_max_s - R_right,
                                               dF, kGradedCoronaRatio);
  if (n < 4) {
    n = 4;
    for (int i = 0; i <= n; ++i) {
      node_R[i] = R_max_s * static_cast<double>(i) / static_cast<double>(n);
    }
  } else {
    node_R[0] = 0.0;
    for (int i = 0; i < n; ++i) {
      node_R[i + 1] = node_R[i] + fmax(widths[i], 1.0e-12);
    }
    node_R[n] = R_max_s;
  }
  for (int k = 0; k <= n; ++k) {
    node_Z[k] = -node_R[n - k];
  }
  for (int k = 1; k <= n; ++k) {
    node_Z[n + k] = node_R[k];
  }
  return n;
}

__device__ PersistentCriticalSurfaceEstimate1D
persistent_laser_estimate_critical_surface_mesh(
    const double* n_hat_cell,
    const double* r_edges,
    const int n_cells) {
  PersistentCriticalSurfaceEstimate1D out;
  if (n_cells <= 0) {
    return out;
  }
  int outermost_critical = -1;
  for (int c = 0; c < n_cells; ++c) {
    if (n_hat_cell[c] >= 1.0) {
      outermost_critical = c;
    }
  }
  if (outermost_critical < 0) {
    return out;
  }
  int fcrit = -1;
  for (int c = outermost_critical; c < n_cells; ++c) {
    const bool this_supercritical = (n_hat_cell[c] >= 1.0);
    const bool next_subcritical =
        (c + 1 >= n_cells) || (n_hat_cell[c + 1] < 1.0);
    if (this_supercritical && next_subcritical) {
      fcrit = c + 1;
      break;
    }
  }
  if (fcrit < 0 || fcrit > n_cells) {
    return out;
  }
  out.fcrit = fcrit;
  out.r_face = r_edges[fcrit];
  out.r_interp = out.r_face;
  if (fcrit > 0 && fcrit < n_cells) {
    const int c_hi = fcrit - 1;
    const int c_lo = fcrit;
    const double n_hi = n_hat_cell[c_hi];
    const double n_lo = n_hat_cell[c_lo];
    const double r_hi = 0.5 * (r_edges[c_hi] + r_edges[c_hi + 1]);
    const double r_lo = 0.5 * (r_edges[c_lo] + r_edges[c_lo + 1]);
    if (n_hi > 1.0 && n_lo > 0.0 && n_lo < 1.0 && r_lo > r_hi) {
      const double denom = log(n_lo / n_hi);
      if (isfinite(denom) && fabs(denom) > 1.0e-12) {
        const double theta =
            fmin(1.0, fmax(0.0, log(1.0 / n_hi) / denom));
        out.r_interp = fmin(r_lo, fmax(r_hi, r_hi + theta * (r_lo - r_hi)));
      }
    }
  }
  return out;
}

__device__ bool persistent_laser_prepare_mesh_1d(
    const PersistentParams& p,
    const PersistentDeviceBuffers& b,
    int* nr_out,
    int* n_nodes_r_out,
    int* n_nodes_z_out,
    int* n_nodes_total_out,
    int* fcrit_cell_out,
    int* outer_surface_cell_out,
    int* use_ghost_corona_out,
    double* r_crit_interp_out,
    double* ne_fcrit_center_out,
    double* r_fcrit_center_out,
    double* r_surface_outer_out,
    double* ghost_ne_inner_out,
    double* ghost_scale_length_out,
    double* ghost_ne_min_out,
    double* r_ghost_outer_out,
    double* Te_anchor_out,
    double* zbar_anchor_out,
    const double t_op,
    int* error_flag) {
  if (!pk_global_leader(p)) {
    return true;
  }
  const double n_crit_safe = fmax(p.laser_n_crit, 1.0e-30);
  for (int c = 0; c < p.n_cells; ++c) {
    if (b.laser_cell_is_void != nullptr && b.laser_cell_is_void[c] != 0U) {
      b.laser_cell_n_hat[c] = 0.0;
      continue;
    }
    const double A_eff = (b.A_eff != nullptr) ? fmax(b.A_eff[c], 1.0e-30)
                                              : fmax(p.material_A, 1.0e-30);
    const double ne = fmax(0.0, b.rho[c]) * fmax(0.0, b.zbar[c]) /
                      (A_eff * core::constants::proton_mass);
    b.laser_cell_n_hat[c] = fmax(0.0, ne / n_crit_safe);
  }
  const PersistentCriticalSurfaceEstimate1D crit_est =
      persistent_laser_estimate_critical_surface_mesh(
          b.laser_cell_n_hat, b.x_r, p.n_cells);
  const int fcrit = crit_est.fcrit;
  const double fallback_r_max =
      p.laser_lmesh_r_max_factor * fmax(p.laser_lmesh_target_radius, 1.0e-12);
  int outermost_threshold = -1;
  for (int c = 0; c < p.n_cells; ++c) {
    if (b.laser_cell_n_hat[c] >= p.laser_lmesh_rmax_n_hat_threshold) {
      outermost_threshold = c;
    }
  }
  int outer_surface_cell = -1;
  for (int c = p.n_cells - 1; c >= 0; --c) {
    if (b.laser_cell_is_void != nullptr && b.laser_cell_is_void[c] != 0U) {
      continue;
    }
    if (c + 1 >= p.n_cells ||
        (b.laser_cell_is_void != nullptr &&
         b.laser_cell_is_void[c + 1] != 0U)) {
      outer_surface_cell = c;
      break;
    }
  }
  double R_max = (outermost_threshold >= 0)
                     ? p.laser_lmesh_r_max_factor * b.x_r[outermost_threshold + 1]
                     : fallback_r_max;
  const double R_crit_raw =
      (fcrit >= 0 && fcrit <= p.n_cells) ? b.x_r[fcrit] : (0.5 * R_max);
  double min_dr_crit = CUDART_INF;
  double min_dr_global = CUDART_INF;
  const double r_window = 0.05 * fmax(R_crit_raw, 1.0e-12);
  for (int c = 0; c < p.n_cells; ++c) {
    const double r_l = b.x_r[c];
    const double r_r = b.x_r[c + 1];
    const double dr = r_r - r_l;
    if (!(dr > 0.0) || !isfinite(dr)) {
      continue;
    }
    min_dr_global = fmin(min_dr_global, dr);
    const double r_c = 0.5 * (r_l + r_r);
    const int dc = c - fcrit;
    const bool near_fcrit =
        (fcrit >= 0) ? ((dc < 0 ? -dc : dc) <= 10) : false;
    const bool near_rcrit = fabs(r_c - R_crit_raw) <= r_window;
    if (near_fcrit || near_rcrit) {
      min_dr_crit = fmin(min_dr_crit, dr);
    }
  }
  if (!(min_dr_crit > 0.0) || !isfinite(min_dr_crit)) {
    min_dr_crit = min_dr_global;
  }
  if (!(min_dr_crit > 0.0) || !isfinite(min_dr_crit)) {
    min_dr_crit = fmax(R_max / 4.0, 1.0e-12);
  }
  R_max = fmax(R_max, 4.0 * min_dr_crit);
  const double R_crit = fmin(R_max, fmax(0.0, R_crit_raw));
  const double dR_fine = fmax(p.laser_lmesh_mesh_factor * min_dr_crit,
                              1.0e-12);
  const int nr = persistent_laser_build_graded_nodes_1d(
      R_crit, dR_fine, R_max, p.laser_lmesh_nr_max, b.laser_node_R,
      b.laser_node_Z, b.laser_widths);
  const int nz = 2 * nr;
  if (nr > p.laser_lmesh_nr_capacity || nz > p.laser_lmesh_nz_capacity) {
    pk_raise_error(error_flag,
                   pk_encode_error_code(kPkErrorLaserMeshPrep));
    return false;
  }
  constexpr int kGhostAnchorSpan = 3;
  const int use_ghost_corona =
      (p.laser_lmesh_ghost_corona_enabled != 0 && outer_surface_cell >= 0 &&
       p.laser_lmesh_ghost_n_out > 0)
          ? 1
          : 0;
  const double r_surface_outer =
      (outer_surface_cell >= 0) ? b.x_r[outer_surface_cell + 1] : 0.0;
  const double Te_anchor_raw =
      (use_ghost_corona != 0)
          ? persistent_laser_average_outer_cells(
                b.Te, b.laser_cell_is_void, outer_surface_cell, kGhostAnchorSpan,
                p.n_cells, p.laser_lmesh_ghost_Te_min_eV)
          : p.laser_lmesh_ghost_Te_min_eV;
  const double Te_anchor =
      fmax(Te_anchor_raw, p.laser_lmesh_ghost_Te_min_eV);
  const double zbar_anchor_raw = persistent_laser_average_outer_cells(
      b.zbar, b.laser_cell_is_void, outer_surface_cell, kGhostAnchorSpan,
      p.n_cells, p.laser_lmesh_ghost_zbar_max);
  const double zbar_anchor =
      fmin(fmax(fmax(zbar_anchor_raw, p.laser_lmesh_ghost_zbar_min),
                p.laser_lmesh_ghost_zbar_min),
           fmax(p.laser_lmesh_ghost_zbar_max,
                p.laser_lmesh_ghost_zbar_min));
  const double A_anchor =
      fmax(persistent_laser_average_outer_cells(
               b.A_eff, b.laser_cell_is_void, outer_surface_cell,
               kGhostAnchorSpan, p.n_cells, p.material_A),
           1.0e-30);
  const double base_ghost_width =
      (use_ghost_corona != 0)
          ? fmax(static_cast<double>(p.laser_lmesh_ghost_n_out) * dR_fine,
                 1.0e-30)
          : 0.0;
  const double ghost_cs =
      persistent_laser_ghost_sound_speed_cm_s(Te_anchor, zbar_anchor, A_anchor);
  const double ghost_width =
      (use_ghost_corona != 0)
          ? fmax(base_ghost_width, ghost_cs * fmax(t_op, 0.0))
          : 0.0;
  const double r_ghost_outer = r_surface_outer + ghost_width;
  const double ghost_ne_min = fmax(p.laser_lmesh_ghost_ne_min_frac, 1.0e-12);
  const double ghost_ne_max =
      fmax(p.laser_lmesh_ghost_ne_max_frac, ghost_ne_min * 1.0001);
  const double outer_n_hat =
      (outer_surface_cell >= 0) ? b.laser_cell_n_hat[outer_surface_cell] : 0.0;
  const double ghost_ne_inner =
      (outer_surface_cell >= 0 && outer_n_hat < 1.0)
          ? fmin(ghost_ne_max,
                 fmax(fmax(outer_n_hat, ghost_ne_min * 1.0001),
                      ghost_ne_min * 1.0001))
          : ghost_ne_max;
  const double ghost_log_span =
      log(fmax(ghost_ne_inner, ghost_ne_min * 1.0001) / ghost_ne_min);
  const double ghost_scale_length =
      (ghost_log_span > 0.0) ? fmax(ghost_width / ghost_log_span, 1.0e-30)
                             : ghost_width;
  *nr_out = nr;
  *n_nodes_r_out = nr + 1;
  *n_nodes_z_out = nz + 1;
  *n_nodes_total_out = (nr + 1) * (nz + 1);
  *fcrit_cell_out = fcrit;
  *outer_surface_cell_out = outer_surface_cell;
  *use_ghost_corona_out = use_ghost_corona;
  *r_crit_interp_out = crit_est.r_interp;
  *ne_fcrit_center_out =
      (fcrit >= 0 && fcrit < p.n_cells) ? b.laser_cell_n_hat[fcrit] : 0.0;
  *r_fcrit_center_out =
      (fcrit >= 0 && fcrit < p.n_cells)
          ? 0.5 * (b.x_r[fcrit] + b.x_r[fcrit + 1])
          : 0.0;
  *r_surface_outer_out = r_surface_outer;
  *ghost_ne_inner_out = ghost_ne_inner;
  *ghost_scale_length_out = ghost_scale_length;
  *ghost_ne_min_out = ghost_ne_min;
  *r_ghost_outer_out = r_ghost_outer;
  *Te_anchor_out = Te_anchor;
  *zbar_anchor_out = zbar_anchor;
  return true;
}

__device__ PersistentCriticalSurfaceEstimate1D
persistent_laser_estimate_critical_surface_transfer(
    const double* n_hat_cell,
    const double* r_edges,
    const int n_cells) {
  PersistentCriticalSurfaceEstimate1D out;
  int outermost_critical = -1;
  for (int c = 0; c < n_cells; ++c) {
    if (n_hat_cell[c] >= 1.0) {
      outermost_critical = c;
    }
  }
  if (outermost_critical < 0) {
    return out;
  }
  int fcrit = -1;
  for (int c = outermost_critical; c < n_cells; ++c) {
    const bool this_supercritical = (n_hat_cell[c] >= 1.0);
    const bool next_subcritical =
        (c + 1 >= n_cells) || (n_hat_cell[c + 1] < 1.0);
    if (this_supercritical && next_subcritical) {
      fcrit = c + 1;
      break;
    }
  }
  if (fcrit <= 0 || fcrit >= n_cells) {
    return out;
  }
  out.fcrit = fcrit;
  out.r_face = r_edges[fcrit];
  out.r_interp = out.r_face;
  const int c_hi = fcrit - 1;
  const int c_lo = fcrit;
  const double n_hi = n_hat_cell[c_hi];
  const double n_lo = n_hat_cell[c_lo];
  const double r_hi = 0.5 * (r_edges[c_hi] + r_edges[c_hi + 1]);
  const double r_lo = 0.5 * (r_edges[c_lo] + r_edges[c_lo + 1]);
  if (n_hi > 1.0 && n_lo > 0.0 && n_lo < 1.0 && r_lo > r_hi) {
    const double denom = log(n_lo / n_hi);
    if (isfinite(denom) && fabs(denom) > 1.0e-12) {
      const double theta = fmin(1.0, fmax(0.0, log(1.0 / n_hi) / denom));
      out.r_interp = fmin(r_lo, fmax(r_hi, r_hi + theta * (r_lo - r_hi)));
    }
  }
  return out;
}

__device__ PersistentAllowedSupercriticalCell1D
persistent_laser_find_allowed_supercritical_1d(
    const PersistentParams& p,
    const PersistentDeviceBuffers& b) {
  PersistentAllowedSupercriticalCell1D out;
  int outermost_real = -1;
  bool any_subcritical_real = false;
  for (int c = 0; c < p.n_cells; ++c) {
    if (b.laser_cell_is_void != nullptr && b.laser_cell_is_void[c] != 0U) {
      continue;
    }
    outermost_real = c;
    any_subcritical_real = any_subcritical_real || (b.laser_cell_n_hat[c] < 1.0);
  }
  if (outermost_real < 0) {
    return out;
  }
  const PersistentCriticalSurfaceEstimate1D crit_est =
      persistent_laser_estimate_critical_surface_transfer(
          b.laser_cell_n_hat, b.x_r, p.n_cells);
  if (crit_est.fcrit > 0 && crit_est.fcrit < p.n_cells &&
      (b.laser_cell_is_void == nullptr ||
       b.laser_cell_is_void[crit_est.fcrit] == 0U) &&
      (b.laser_cell_is_void == nullptr ||
       b.laser_cell_is_void[crit_est.fcrit - 1] == 0U) &&
      b.laser_cell_n_hat[crit_est.fcrit - 1] >= 1.0 &&
      b.laser_cell_n_hat[crit_est.fcrit] < 1.0) {
    out.allowed_cell = crit_est.fcrit - 1;
    out.critical_adjacent_subcritical_cell = crit_est.fcrit;
    out.r_crit = crit_est.r_interp;
    out.fallback_only = 0;
    return out;
  }
  if (any_subcritical_real) {
    return out;
  }
  int fallback = -1;
  for (int c = outermost_real; c >= 0; --c) {
    if (b.laser_cell_is_void != nullptr && b.laser_cell_is_void[c] != 0U) {
      continue;
    }
    fallback = c;
    break;
  }
  if (fallback >= 0) {
    out.allowed_cell = fallback;
    out.r_crit = crit_est.r_interp;
    out.fallback_only = 1;
  }
  return out;
}

__device__ bool persistent_laser_transfer_blocked(const PersistentDeviceBuffers& b,
                                                  const int c,
                                                  const int allowed_cell) {
  if (b.laser_cell_is_void != nullptr && b.laser_cell_is_void[c] != 0U) {
    return true;
  }
  return b.laser_cell_n_hat[c] >= 1.0 && c != allowed_cell;
}

__device__ bool persistent_laser_transfer_subcritical_receiver(
    const PersistentDeviceBuffers& b,
    const int c) {
  if (b.laser_cell_is_void != nullptr && b.laser_cell_is_void[c] != 0U) {
    return true;
  }
  return b.laser_cell_n_hat[c] >= 1.0;
}

__device__ bool persistent_laser_anchor_mask_blocked(
    const PersistentDeviceBuffers& b,
    const PersistentAllowedSupercriticalCell1D& allowed,
    const int c) {
  return (allowed.fallback_only != 0)
             ? persistent_laser_transfer_blocked(b, c, allowed.allowed_cell)
             : persistent_laser_transfer_subcritical_receiver(b, c);
}

__device__ int persistent_laser_search_active_anchor_1d(
    const PersistentDeviceBuffers& b,
    const PersistentAllowedSupercriticalCell1D& allowed,
    const int begin,
    const int end,
    const int step,
    const int dir,
    int* direction) {
  for (int d = begin; d != end; d += step) {
    if (!persistent_laser_anchor_mask_blocked(b, allowed, d)) {
      *direction = dir;
      return d;
    }
  }
  return -1;
}

__device__ int persistent_laser_find_active_anchor_1d(
    const PersistentParams& p,
    const PersistentDeviceBuffers& b,
    const PersistentAllowedSupercriticalCell1D& allowed,
    const int c,
    const bool prefer_outward,
    const bool allow_opposite_fallback,
    int* direction) {
  *direction = 0;
  if (prefer_outward) {
    const int outward = persistent_laser_search_active_anchor_1d(
        b, allowed, c + 1, p.n_cells, 1, 1, direction);
    if (outward >= 0 || !allow_opposite_fallback) {
      return outward;
    }
    return persistent_laser_search_active_anchor_1d(
        b, allowed, c - 1, -1, -1, -1, direction);
  }
  const int inward = persistent_laser_search_active_anchor_1d(
      b, allowed, c - 1, -1, -1, -1, direction);
  if (inward >= 0 || !allow_opposite_fallback) {
    return inward;
  }
  return persistent_laser_search_active_anchor_1d(
      b, allowed, c + 1, p.n_cells, 1, 1, direction);
}

__device__ void persistent_laser_distribute_to_stencil_1d(
    const PersistentParams& p,
    const PersistentDeviceBuffers& b,
    const PersistentAllowedSupercriticalCell1D& allowed,
    const int anchor,
    const int direction,
    const double power) {
  if (!(power > 0.0) || anchor < 0 || anchor >= p.n_cells || direction == 0) {
    return;
  }
  const int span = (p.laser_lmesh_ghost_handoff_cells > 1)
                       ? p.laser_lmesh_ghost_handoff_cells
                       : 1;
  const double decay = fmax(p.laser_lmesh_ghost_handoff_decay, 1.0e-12);
  double weight_sum = 0.0;
  for (int k = 0; k < span; ++k) {
    const int idx = anchor + direction * k;
    if (idx < 0 || idx >= p.n_cells) {
      break;
    }
    if (persistent_laser_transfer_blocked(b, idx, allowed.allowed_cell)) {
      continue;
    }
    weight_sum += exp(-static_cast<double>(k) / decay);
  }
  if (!(weight_sum > 0.0)) {
    b.laser_dep[anchor] += power;
    return;
  }
  for (int k = 0; k < span; ++k) {
    const int idx = anchor + direction * k;
    if (idx < 0 || idx >= p.n_cells) {
      break;
    }
    if (persistent_laser_transfer_blocked(b, idx, allowed.allowed_cell)) {
      continue;
    }
    const double w = exp(-static_cast<double>(k) / decay);
    b.laser_dep[idx] += power * (w / weight_sum);
  }
}

__device__ double persistent_laser_apply_deposit_transfer_1d(
    const PersistentParams& p,
    const PersistentDeviceBuffers& b,
    const double dt) {
  if (!pk_global_leader(p)) {
    return 0.0;
  }
  const PersistentAllowedSupercriticalCell1D allowed =
      persistent_laser_find_allowed_supercritical_1d(p, b);
  for (int c = 0; c < p.n_cells; ++c) {
    if (!persistent_laser_transfer_blocked(b, c, allowed.allowed_cell)) {
      continue;
    }
    const double power = b.laser_dep[c];
    if (!(power > 0.0)) {
      continue;
    }
    const bool is_void_source =
        b.laser_cell_is_void != nullptr && b.laser_cell_is_void[c] != 0U;
    int direction = 0;
    const int target = persistent_laser_find_active_anchor_1d(
        p, b, allowed, c, !is_void_source, is_void_source, &direction);
    if (target >= 0) {
      if (is_void_source && p.laser_lmesh_ghost_corona_enabled != 0) {
        persistent_laser_distribute_to_stencil_1d(
            p, b, allowed, target, direction, power);
      } else {
        b.laser_dep[target] += power;
      }
    }
    b.laser_dep[c] = 0.0;
  }
  double dep_energy_sum = 0.0;
  for (int c = 0; c < p.n_cells; ++c) {
    const double energy = b.laser_dep[c] * dt;
    b.laser_dep[c] = energy;
    dep_energy_sum += fmax(energy, 0.0);
  }
  return dep_energy_sum;
}

__device__ inline void persistent_laser_trace_substage(
    const PersistentParams& p,
    const int local_step,
    const double beam_power,
    const char* name) {
  (void)beam_power;
  if (p.laser_trace && local_step < 70 && pk_global_leader(p)) {
    printf("pk_laser step=%d sub=%s done\n", local_step, name);
  }
}

__device__ void persistent_laser_raytrace_1d_folded(
    const PersistentParams& p,
    const PersistentDeviceBuffers& b,
    const double beam_power,
    const double total_power,
    const double dt,
    const double t_op,
    const int local_step,
    int* phase_prof_last_phase,
    unsigned long long* phase_prof_last_clock,
    int* error_flag,
    double* out_input,
    double* out_escaped) {
  __shared__ int sh_nr;
  __shared__ int sh_n_nodes_r;
  __shared__ int sh_n_nodes_z;
  __shared__ int sh_n_nodes_total;
  __shared__ int sh_fcrit_cell;
  __shared__ int sh_outer_surface_cell;
  __shared__ int sh_use_ghost_corona;
  __shared__ double sh_r_crit_interp;
  __shared__ double sh_ne_fcrit_center;
  __shared__ double sh_r_fcrit_center;
  __shared__ double sh_r_surface_outer;
  __shared__ double sh_ghost_ne_inner;
  __shared__ double sh_ghost_scale_length;
  __shared__ double sh_ghost_ne_min;
  __shared__ double sh_r_ghost_outer;
  __shared__ double sh_Te_anchor;
  __shared__ double sh_zbar_anchor;
  __shared__ double sh_ray_sum_w;
  __shared__ double sh_dep_energy_sum;

  if (threadIdx.x == 0) {
    *out_input = total_power * dt;
    *out_escaped = 0.0;
    sh_dep_energy_sum = 0.0;
  }
  if (pk_global_leader(p)) {
    if (b.laser_error_flags != nullptr) {
      b.laser_error_flags->nan_particle = 0;
      b.laser_error_flags->invalid_cell = 0;
      b.laser_error_flags->invalid_boundary = 0;
      b.laser_error_flags->pool_overflow = 0;
      b.laser_error_flags->opacity_out_of_range = 0;
      b.laser_error_flags->infinite_loop = 0;
      b.laser_error_flags->ddmc_sigma_tot_zero = 0;
      b.laser_error_flags->roulette_kill = 0;
    }
    if (b.laser_unabsorbed != nullptr) {
      *b.laser_unabsorbed = 0.0;
    }
    if (b.laser_tail_closure_count != nullptr) {
      *b.laser_tail_closure_count = 0ULL;
    }
    if (b.laser_tail_closure_absorbed_power != nullptr) {
      *b.laser_tail_closure_absorbed_power = 0.0;
    }
    if (b.laser_critical_surface_hit_count != nullptr) {
      *b.laser_critical_surface_hit_count = 0ULL;
    }
  }
  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    b.laser_dep[c] = 0.0;
  }
  const int n_rays = p.laser_rays_per_beam;
  const int n_ray_cells = n_rays * p.n_cells;
  for (int i = pk_thread_id(p); i < n_ray_cells; i += pk_thread_stride(p)) {
    b.laser_per_ray_deposit[i] = 0.0;
  }
  for (int r = pk_thread_id(p); r < n_rays; r += pk_thread_stride(p)) {
    b.laser_per_ray_unabsorbed[r] = 0.0;
    b.laser_per_ray_tail_power[r] = 0.0;
  }
  __threadfence();
  pk_sync(p);

  if (!(beam_power > 0.0) || n_rays <= 0) {
    if (pk_global_leader(p)) {
      *out_escaped = total_power * dt;
    }
    pk_sync(p);
    return;
  }

  pk_phase_prof_mark(p, kPkWatchPhaseLaserMeshPrep, phase_prof_last_phase,
                     phase_prof_last_clock);
  if (pk_global_leader(p)) {
    const bool mesh_ok = persistent_laser_prepare_mesh_1d(
        p, b, b.grid_broadcast_ints + 0, b.grid_broadcast_ints + 1,
        b.grid_broadcast_ints + 2, b.grid_broadcast_ints + 3,
        b.grid_broadcast_ints + 4, b.grid_broadcast_ints + 6,
        b.grid_broadcast_ints + 7, b.grid_broadcast_doubles + 0,
        b.grid_broadcast_doubles + 1, b.grid_broadcast_doubles + 2,
        b.grid_broadcast_doubles + 3, b.grid_broadcast_doubles + 4,
        b.grid_broadcast_doubles + 5, b.grid_broadcast_doubles + 6,
        b.grid_broadcast_doubles + 7, b.grid_broadcast_doubles + 8,
        b.grid_broadcast_doubles + 9, t_op, error_flag);
    b.grid_broadcast_ints[5] = mesh_ok ? 1 : 0;
  }
  __threadfence();
  pk_sync(p);
  if (threadIdx.x == 0) {
    sh_nr = *(volatile const int*)(b.grid_broadcast_ints + 0);
    sh_n_nodes_r = *(volatile const int*)(b.grid_broadcast_ints + 1);
    sh_n_nodes_z = *(volatile const int*)(b.grid_broadcast_ints + 2);
    sh_n_nodes_total = *(volatile const int*)(b.grid_broadcast_ints + 3);
    sh_fcrit_cell = *(volatile const int*)(b.grid_broadcast_ints + 4);
    sh_outer_surface_cell = *(volatile const int*)(b.grid_broadcast_ints + 6);
    sh_use_ghost_corona = *(volatile const int*)(b.grid_broadcast_ints + 7);
    sh_r_crit_interp =
        *(volatile const double*)(b.grid_broadcast_doubles + 0);
    sh_ne_fcrit_center =
        *(volatile const double*)(b.grid_broadcast_doubles + 1);
    sh_r_fcrit_center =
        *(volatile const double*)(b.grid_broadcast_doubles + 2);
    sh_r_surface_outer =
        *(volatile const double*)(b.grid_broadcast_doubles + 3);
    sh_ghost_ne_inner =
        *(volatile const double*)(b.grid_broadcast_doubles + 4);
    sh_ghost_scale_length =
        *(volatile const double*)(b.grid_broadcast_doubles + 5);
    sh_ghost_ne_min =
        *(volatile const double*)(b.grid_broadcast_doubles + 6);
    sh_r_ghost_outer =
        *(volatile const double*)(b.grid_broadcast_doubles + 7);
    sh_Te_anchor =
        *(volatile const double*)(b.grid_broadcast_doubles + 8);
    sh_zbar_anchor =
        *(volatile const double*)(b.grid_broadcast_doubles + 9);
    if (*(volatile const int*)(b.grid_broadcast_ints + 5) == 0) {
      pk_raise_error(error_flag,
                     pk_encode_error_code(kPkErrorLaserMeshPrep));
    }
  }
  pk_sync(p);
  if (*(volatile const int*)error_flag != 0) {
    return;
  }

  pk_phase_prof_mark(p, kPkWatchPhaseLaserMapPrep, phase_prof_last_phase,
                     phase_prof_last_clock);
  for (int idx = pk_thread_id(p); idx < sh_n_nodes_total; idx += pk_thread_stride(p)) {
    laser::laser_mesh_bodies::map_hydro_to_laser_1d_kernel_body(
        idx, b.laser_node_R, b.laser_node_Z, b.rho, b.Te, b.zbar, b.A_eff,
        b.laser_cell_is_void, b.x_r, b.laser_n_e_hat, b.laser_n_e_hat_raw,
        b.laser_T_e, b.laser_Zbar, sh_n_nodes_total, sh_n_nodes_z, p.n_cells,
        fmax(p.laser_n_crit, 1.0e-30), sh_use_ghost_corona,
        sh_outer_surface_cell, sh_ghost_ne_inner, sh_ghost_scale_length,
        sh_ghost_ne_min, sh_r_surface_outer, sh_r_ghost_outer, sh_Te_anchor,
        sh_zbar_anchor, p.laser_lmesh_ghost_zbar_min,
        p.laser_lmesh_ghost_zbar_max,
        p.laser_lmesh_ghost_Te_min_eV, p.laser_lmesh_critical_clip,
        p.laser_lmesh_n_hat_margin, sh_fcrit_cell, sh_r_crit_interp,
        sh_ne_fcrit_center, sh_r_fcrit_center);
  }
  pk_sync(p);
  persistent_laser_trace_substage(p, local_step, beam_power, "prep_map");

  if (b.laser_ema_valid != nullptr && b.laser_ema_state != nullptr) {
    if (*(volatile const int*)b.laser_ema_valid == sh_n_nodes_total) {
      for (int idx = pk_thread_id(p); idx < sh_n_nodes_total; idx += pk_thread_stride(p)) {
        laser::laser_mesh_bodies::ema_smooth_n_hat_kernel_body(
            idx, b.laser_n_e_hat, b.laser_ema_state, sh_n_nodes_total);
      }
      pk_sync(p);
    }
    for (int idx = pk_thread_id(p); idx < sh_n_nodes_total; idx += pk_thread_stride(p)) {
      b.laser_ema_state[idx] = b.laser_n_e_hat[idx];
    }
    pk_sync(p);
    if (pk_global_leader(p)) {
      *b.laser_ema_valid = sh_n_nodes_total;
    }
    __threadfence();
    pk_sync(p);
  }
  persistent_laser_trace_substage(p, local_step, beam_power, "prep_ema");

  const int j_center = sh_n_nodes_z / 2;
  for (int i = pk_thread_id(p); i < sh_n_nodes_r; i += pk_thread_stride(p)) {
    laser::laser_mesh_bodies::extract_radial_profile_kernel_body(
        i, b.laser_radial_node_r, b.laser_radial_n_hat,
        b.laser_radial_n_hat_raw, b.laser_radial_smooth_kappa,
        b.laser_node_R, b.laser_n_e_hat, b.laser_n_e_hat_raw, nullptr,
        sh_n_nodes_r, sh_n_nodes_z, j_center, 0);
  }
  pk_sync(p);
  for (int i = pk_thread_id(p); i < sh_n_nodes_r; i += pk_thread_stride(p)) {
    laser::laser_mesh_bodies::compute_radial_gradient_kernel_body(
        i, b.laser_radial_dn_dr, b.laser_radial_node_r,
        b.laser_radial_n_hat, sh_n_nodes_r);
  }
  for (int idx = pk_thread_id(p); idx < sh_n_nodes_total; idx += pk_thread_stride(p)) {
    laser::laser_mesh_bodies::compute_gradient_kernel_body(
        idx, b.laser_grad_n_hat_R, b.laser_grad_n_hat_Z, b.laser_n_e_hat,
        b.laser_node_R, b.laser_node_Z, sh_n_nodes_r, sh_n_nodes_z);
  }
  pk_sync(p);
  for (int idx = pk_thread_id(p); idx < sh_n_nodes_total; idx += pk_thread_stride(p)) {
    laser::laser_mesh_bodies::compute_smooth_kappa_kernel_body(
        idx, b.laser_smooth_kappa_factor, b.laser_n_e_hat, b.laser_T_e,
        b.laser_Zbar, p.laser_lambda_cm, p.laser_eps_n,
        p.laser_coulomb_log_floor, sh_n_nodes_total);
  }
  pk_sync(p);
  persistent_laser_trace_substage(p, local_step, beam_power, "prep_kappa");
  for (int i = pk_thread_id(p); i < sh_n_nodes_r; i += pk_thread_stride(p)) {
    laser::laser_mesh_bodies::extract_radial_profile_kernel_body(
        i, b.laser_radial_node_r, b.laser_radial_n_hat,
        b.laser_radial_n_hat_raw, b.laser_radial_smooth_kappa,
        b.laser_node_R, b.laser_n_e_hat, b.laser_n_e_hat_raw,
        b.laser_smooth_kappa_factor, sh_n_nodes_r, sh_n_nodes_z, j_center, 1);
  }
  pk_sync(p);
  persistent_laser_trace_substage(p, local_step, beam_power, "prep_profile");
  for (int i = pk_thread_id(p); i < sh_n_nodes_r; i += pk_thread_stride(p)) {
    laser::laser_mesh_bodies::compute_radial_gradient_kernel_body(
        i, b.laser_radial_dn_dr, b.laser_radial_node_r,
        b.laser_radial_n_hat, sh_n_nodes_r);
  }
  pk_sync(p);
  persistent_laser_trace_substage(p, local_step, beam_power, "prep_gradients");

  const PersistentAllowedSupercriticalCell1D allowed =
      persistent_laser_find_allowed_supercritical_1d(p, b);
  pk_phase_prof_mark(p, kPkWatchPhaseLaserRayInit, phase_prof_last_phase,
                     phase_prof_last_clock);
  persistent_laser_initialize_rays_1d(p, b, sh_nr, beam_power,
                                      &sh_ray_sum_w);
  pk_sync(p);
  persistent_laser_trace_substage(p, local_step, beam_power, "ray_init");

  pk_phase_prof_mark(p, kPkWatchPhaseLaserTraceMarch, phase_prof_last_phase,
                     phase_prof_last_clock);
  for (int ray = pk_thread_id(p); ray < n_rays; ray += pk_thread_stride(p)) {
    laser::ray_trace_bodies::ray_trace_1d_sph_body<false, false, false>(
        ray, b.laser_dep, b.laser_per_ray_deposit,
        b.laser_per_ray_unabsorbed, b.laser_per_ray_tail_power,
        b.laser_radial_node_r, b.laser_radial_n_hat,
        b.laser_radial_n_hat_raw, b.laser_radial_smooth_kappa,
        b.laser_radial_dn_dr, b.x_r, allowed.allowed_cell,
        allowed.critical_adjacent_subcritical_cell, allowed.r_crit,
        b.laser_ray_R0, b.laser_ray_Z0, b.laser_ray_vR0, b.laser_ray_vZ0,
        b.laser_ray_power, b.laser_ray_power0, p.laser_raytrace_cfl_ray,
        p.laser_raytrace_ds_adapt_g_target,
        p.laser_raytrace_ds_adapt_tau_target,
        p.laser_raytrace_ds_adapt_theta_target,
        p.laser_raytrace_ds_adapt_max_factor, p.laser_eps_n,
        p.laser_eps_crit, p.laser_lambda_cm, p.laser_coulomb_log_floor,
        p.laser_test_kappa, p.laser_intensity_cutoff,
        p.laser_raytrace_max_steps, sh_n_nodes_r, p.n_cells, n_rays, nullptr,
        nullptr, nullptr, nullptr, 0, 1, 0, nullptr, nullptr, nullptr,
        b.laser_unabsorbed, b.laser_tail_closure_count,
        b.laser_tail_closure_absorbed_power,
        b.laser_critical_surface_hit_count, b.laser_error_flags,
        laser::CbetRecordDeviceArgs{}, laser::HotECaptureParams{}, nullptr,
        laser::LaserPhysExtOptions{}, nullptr, nullptr, nullptr);
  }
  __threadfence();
  pk_sync(p);
  persistent_laser_trace_substage(p, local_step, beam_power, "trace");

  pk_phase_prof_mark(p, kPkWatchPhaseLaserTallyReduce, phase_prof_last_phase,
                     phase_prof_last_clock);
  for (int c = pk_thread_id(p); c < p.n_cells + 2; c += pk_thread_stride(p)) {
    laser::ray_trace_bodies::reduce_per_ray_tallies_1d_body(
        c, b.laser_per_ray_deposit, b.laser_per_ray_unabsorbed,
        b.laser_per_ray_tail_power, b.laser_dep, b.laser_unabsorbed,
        b.laser_tail_closure_absorbed_power, n_rays, p.n_cells, nullptr,
        nullptr);
  }
  pk_sync(p);
  persistent_laser_trace_substage(p, local_step, beam_power, "tally_reduce");

  pk_phase_prof_mark(p, kPkWatchPhaseLaserFoldReplayDeposit,
                     phase_prof_last_phase, phase_prof_last_clock);
  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    const double single = b.laser_dep[c];
    double total = 0.0;
    for (int k = 0; k < p.laser_n_folded; ++k) {
      total += single;
    }
    b.laser_dep[c] = total;
  }
  if (pk_global_leader(p)) {
    const double single_unabsorbed =
        (b.laser_unabsorbed != nullptr) ? *b.laser_unabsorbed : 0.0;
    const double single_tail_power =
        (b.laser_tail_closure_absorbed_power != nullptr)
            ? *b.laser_tail_closure_absorbed_power
            : 0.0;
    const unsigned long long single_tail_count =
        (b.laser_tail_closure_count != nullptr) ? *b.laser_tail_closure_count
                                                : 0ULL;
    const unsigned long long single_crit_hits =
        (b.laser_critical_surface_hit_count != nullptr)
            ? *b.laser_critical_surface_hit_count
            : 0ULL;
    double u = 0.0;
    double tp = 0.0;
    unsigned long long tc = 0ULL;
    unsigned long long ch = 0ULL;
    for (int k = 0; k < p.laser_n_folded; ++k) {
      u += single_unabsorbed;
      tp += single_tail_power;
      tc += single_tail_count;
      ch += single_crit_hits;
    }
    if (b.laser_unabsorbed != nullptr) {
      *b.laser_unabsorbed = u;
    }
    if (b.laser_tail_closure_absorbed_power != nullptr) {
      *b.laser_tail_closure_absorbed_power = tp;
    }
    if (b.laser_tail_closure_count != nullptr) {
      *b.laser_tail_closure_count = tc;
    }
    if (b.laser_critical_surface_hit_count != nullptr) {
      *b.laser_critical_surface_hit_count = ch;
    }
  }
  __threadfence();
  pk_sync(p);
  persistent_laser_trace_substage(p, local_step, beam_power, "fold_replay");

  if (pk_global_leader(p)) {
    sh_dep_energy_sum = persistent_laser_apply_deposit_transfer_1d(p, b, dt);
    const double dep_power =
        (dt > 0.0) ? (sh_dep_energy_sum / dt) : 0.0;
    const double trace_unabsorbed =
        (b.laser_unabsorbed != nullptr) ? *b.laser_unabsorbed : 0.0;
    const double escaped_power =
        fmax(trace_unabsorbed, fmax(0.0, total_power - dep_power));
    *out_escaped = escaped_power * dt;
    if (b.laser_error_flags != nullptr) {
      const core::DeviceErrorFlags* const flags = b.laser_error_flags;
      const int laser_mask = pk_laser_error_mask_from_flags(flags);
      if ((laser_mask & kPkLaserFlagNanParticle) != 0) {
        pk_raise_error(
            error_flag,
            pk_encode_error_code(kPkErrorLaserRayFlags, laser_mask));
      }
    }
  }
  pk_sync(p);
  persistent_laser_trace_substage(p, local_step, beam_power, "deposit_apply");
}

__device__ double persistent_laser_close_electron(
    const PersistentParams& p,
    const PersistentDeviceBuffers& b,
    const int c,
    const double rho_safe,
    const double cv_mass_i,
    const double ee_before_floor,
    int* local_clamp_count) {
  const double A_c = (b.A_eff != nullptr) ? fmax(b.A_eff[c], 1.0e-30)
                                          : p.material_A;
  const double gm1_c = (b.gamma_eff != nullptr)
                           ? fmax(b.gamma_eff[c] - 1.0, 1.0e-30)
                           : p.material_gm1;
  double Te_raw = 0.0;
  double Te_new = p.Te_floor;
  if (p.cv_e_override > 0.0 && p.eos_T_ref_eV > 0.0) {
    const double T_ref = p.eos_T_ref_eV;
    const double T_ref3 = T_ref * T_ref * T_ref;
    const double alpha0 = p.cv_e_override / (4.0 * T_ref3);
    const double arg = b.ee[c] * rho_safe / alpha0;
    Te_raw = (arg > 0.0) ? ::pow(arg, 0.25) : 0.0;
    if (!isfinite(Te_raw)) {
      Te_raw = 0.0;
    }
    Te_new = fmax(Te_raw, p.Te_floor);
    const double Te2 = Te_new * Te_new;
    const double Te4 = Te2 * Te2;
    b.ee[c] = alpha0 * Te4 / rho_safe;
  } else {
    const double z = (b.zbar != nullptr) ? fmax(b.zbar[c], 0.0)
                                         : fmax(p.fallback_z, 0.0);
    double cv_mass_e = 0.0;
    if (p.cv_e_override > 0.0) {
      cv_mass_e = p.cv_e_override / rho_safe;
    } else if (b.cv_e != nullptr && b.cv_e[c] > 0.0) {
      cv_mass_e = b.cv_e[c];
    } else {
      cv_mass_e = z * core::constants::eV_to_erg /
                  (A_c * core::constants::proton_mass * gm1_c);
    }
    cv_mass_e = fmax(cv_mass_e, 1.0e-30);
    const double cv_mass_total =
        (p.two_temperature != 0)
            ? cv_mass_e
            : ((p.cv_e_override > 0.0)
                   ? cv_mass_e
                   : fmax(cv_mass_e + cv_mass_i, 1.0e-30));
    Te_raw = b.ee[c] / cv_mass_total;
    if (!isfinite(Te_raw)) {
      Te_raw = 0.0;
    }
    Te_new = fmax(Te_raw, p.Te_floor);
    b.ee[c] = cv_mass_total * Te_new;
  }
  if (Te_new > Te_raw) {
    ++(*local_clamp_count);
  }
  b.Te[c] = Te_new;
  b.Pe[c] = gm1_c * b.rho[c] * b.ee[c];
  return fmax(b.ee[c] - ee_before_floor, 0.0);
}

__device__ void persistent_laser_inject_source_terms(
    const PersistentParams& p,
    const PersistentDeviceBuffers& b,
    double* smem,
    double* out_floor,
    double* out_skipped) {
  double local_floor = 0.0;
  double local_skipped = 0.0;
  int local_clamp_count = 0;

  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    double dep = b.laser_dep[c];
    if (dep < 0.0) {
      dep = 0.0;
      b.laser_dep[c] = 0.0;
    }
    const double rho_c = b.rho[c];
    const double vol_c = b.vol[c];
    const double denom = rho_c * vol_c;
    if (!(denom > 1.0e-30)) {
      local_skipped += dep;
      continue;
    }

    const double A_c = (b.A_eff != nullptr) ? fmax(b.A_eff[c], 1.0e-30)
                                            : p.material_A;
    const double gm1_c = (b.gamma_eff != nullptr)
                             ? fmax(b.gamma_eff[c] - 1.0, 1.0e-30)
                             : p.material_gm1;
    const double cv_mass_i =
        fmax(core::constants::eV_to_erg /
                 (A_c * core::constants::proton_mass * gm1_c),
             1.0e-30);
    if (p.two_temperature == 0) {
      b.ee[c] += b.ei[c];
    }
    b.ee[c] += dep / denom;
    const double ee_before_floor = b.ee[c];

    const double rho_safe = fmax(rho_c, 1.0e-30);
    const double de_floor_e = persistent_laser_close_electron(
        p, b, c, rho_safe, cv_mass_i, ee_before_floor, &local_clamp_count);
    if (de_floor_e > 0.0) {
      local_floor += rho_c * vol_c * de_floor_e;
    }

    if (p.two_temperature != 0) {
      const double Ti_prev = isfinite(b.Ti[c]) ? b.Ti[c] : 0.0;
      if (Ti_prev < p.Ti_floor) {
        ++local_clamp_count;
        const double ei_before_floor = b.ei[c];
        b.Ti[c] = p.Ti_floor;
        b.ei[c] = cv_mass_i * p.Ti_floor;
        b.Pi[c] = gm1_c * rho_c * b.ei[c];
        const double de_floor_i = b.ei[c] - ei_before_floor;
        if (de_floor_i > 0.0) {
          local_floor += rho_c * vol_c * de_floor_i;
        }
      }
    } else {
      b.Ti[c] = b.Te[c];
      b.ei[c] = 0.0;
      b.Pi[c] = 0.0;
    }
  }

  const double floor_sum = pk_reduce_sum(p, b, local_floor, smem);
  const double skipped_sum = pk_reduce_sum(p, b, local_skipped, smem);
  const double clamp_sum =
      pk_reduce_sum(p, b, static_cast<double>(local_clamp_count), smem);
  if (pk_global_leader(p)) {
    *out_floor = floor_sum;
    *out_skipped = skipped_sum;
    if (b.clamp_count != nullptr) {
      *b.clamp_count += static_cast<int>(clamp_sum);
    }
    if (b.E_floor != nullptr) {
      b.E_floor[0] += floor_sum;
    }
  }
  pk_sync(p);
}

__device__ void persistent_laser_step(const PersistentParams& p,
                                      const PersistentDeviceBuffers& b,
                                      const double dt,
                                      const double t_op,
                                      const int local_step,
                                      int* phase_prof_last_phase,
                                      unsigned long long* phase_prof_last_clock,
                                      int* error_flag,
                                      double* smem,
                                      double* out_input,
                                      double* out_escaped,
                                      double* out_floor,
                                      double* out_skipped) {
  if (p.laser_mode_raytrace_1d != 0) {
    pk_phase_prof_mark(p, kPkWatchPhaseLaserPowerEntry,
                       phase_prof_last_phase, phase_prof_last_clock);
  }
  if (threadIdx.x == 0) {
    *out_input = 0.0;
    *out_escaped = 0.0;
    *out_floor = 0.0;
    *out_skipped = 0.0;
  }
  pk_sync(p);
  if (p.laser_enabled == 0 || !(dt > 0.0) || p.laser_n_folded <= 0 ||
      b.laser_dep == nullptr) {
    return;
  }

  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    b.laser_dep[c] = 0.0;
  }
  pk_sync(p);

  __shared__ double sh_single_unabsorbed;
  __shared__ unsigned long long sh_critical_hits;
  __shared__ double sh_beam_power;
  __shared__ double sh_total_power;
  if (threadIdx.x == 0) {
    sh_beam_power = fmax(0.0, p.laser_power_table.eval(t_op) * 1.0e7);
    sh_total_power = 0.0;
    for (int k = 0; k < p.laser_n_folded; ++k) {
      sh_total_power += sh_beam_power;
    }
    *out_input = sh_total_power * dt;
  }
  pk_sync(p);

  if (p.laser_mode_raytrace_1d != 0) {
    if (p.laser_trace && local_step < 70 && pk_global_leader(p)) {
      printf("pk_laser step=%d sub=power_eval P=%.6e\n", local_step,
             sh_beam_power);
    }
    persistent_laser_raytrace_1d_folded(
        p, b, sh_beam_power, sh_total_power, dt, t_op, local_step,
        phase_prof_last_phase, phase_prof_last_clock, error_flag, out_input,
        out_escaped);
    pk_sync(p);
  } else {
    persistent_laser_radial_absorption_single_trace(
        p, b, sh_beam_power, &sh_single_unabsorbed, &sh_critical_hits,
        error_flag);
    pk_sync(p);

    double replay_unabsorbed = 0.0;
    if (pk_global_leader(p)) {
      for (int k = 0; k < p.laser_n_folded; ++k) {
        replay_unabsorbed += sh_single_unabsorbed;
      }
      *out_escaped = replay_unabsorbed * dt;
    }
    pk_sync(p);

    (void)persistent_laser_replay_deposit_to_energy(p, b, dt, smem);
    pk_sync(p);
  }

  if (p.laser_mode_raytrace_1d != 0) {
    pk_phase_prof_mark(p, kPkWatchPhaseLaserInjectSources,
                       phase_prof_last_phase, phase_prof_last_clock);
  }
  persistent_laser_inject_source_terms(p, b, smem, out_floor, out_skipped);
  pk_sync(p);
}

__device__ void apply_boundary_device(const PersistentParams& p,
                                      const PersistentDeviceBuffers& b) {
  using namespace tenryu::hydro::persistent_1d;
  for (int i = pk_thread_id(p); i < 2; i += pk_thread_stride(p)) {
    apply_boundary_1d_kernel_body(i, b.x_r, b.v_r, p.n_nodes,
                                  p.apply_outer_boundary, p.r_min, p.r_max);
  }
  pk_sync(p);
}

__device__ void persistent_follow_void_region_nodes_1d(
    const PersistentParams& p,
    const PersistentDeviceBuffers& b) {
  using namespace tenryu::hydro::persistent_1d;
  const int first_void = p.first_void_cell;
  const int n_span = p.n_cells - first_void;
  if (first_void >= p.n_cells || n_span <= 1) {
    return;
  }
  for (int j = pk_thread_id(p); j < n_span; j += pk_thread_stride(p)) {
    switch (p.geom_code) {
      case 1:
        follow_void_nodes_1d_body_persistent<1>(
            j, b.x_r, b.v_r, b.mass, first_void, p.n_cells, p.void_rho);
        break;
      case 2:
        follow_void_nodes_1d_body_persistent<2>(
            j, b.x_r, b.v_r, b.mass, first_void, p.n_cells, p.void_rho);
        break;
      default:
        follow_void_nodes_1d_body_persistent<0>(
            j, b.x_r, b.v_r, b.mass, first_void, p.n_cells, p.void_rho);
        break;
    }
  }
}

__device__ void persistent_conduction_operator(const PersistentParams& p,
                                               const PersistentDeviceBuffers& b) {
  using namespace tenryu::hydro::conduction_bodies;
  if (p.conduction_enabled == 0) {
    return;
  }
  if (pk_global_leader(p)) {
    b.cond_diag3[0] = CUDART_INF;
    b.cond_diag3[1] = CUDART_INF;
    b.cond_diag3[2] = 0.0;
  }
  __threadfence();
  pk_sync(p);

  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    if (p.conduction_kappa_power_active != 0) {
      compute_powerlaw_test_kappa_deff_1d_kernel_body(
          c, b.cond_kappa_eff, b.cond_rho_cv_e, b.cond_diag3 + 0,
          b.cond_diag3 + 1, b.cond_diag3 + 2, b.Te, b.rho, b.zbar, b.x_r,
          p.n_cells, b.cond_gamma_eff, b.cond_A_eff, b.cell_is_void,
          p.conduction_test_kappa, p.conduction_kappa_power,
          p.conduction_kappa_rho_power, b.cv_e);
    } else {
      compute_spitzer_deff_1d_kernel_body<false>(
          c, b.cond_kappa_eff, b.cond_rho_cv_e, b.cond_diag3 + 0,
          b.cond_diag3 + 1, b.cond_diag3 + 2, b.Te, b.rho, b.zbar, b.vol,
          b.x_r, p.n_cells, b.cond_gamma_eff, b.cond_A_eff, b.cell_is_void,
          p.conduction_f_lim, p.conduction_test_kappa,
          p.conduction_mfp_limiter_C, b.cv_e, nullptr);
    }
  }
  __threadfence();
  pk_sync(p);

  const bool kirchhoff =
      (p.conduction_face_kappa_policy != 0 &&
       p.conduction_test_kappa <= 0.0 &&
       p.conduction_kappa_power_active == 0);
  if (kirchhoff) {
    for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
      switch (p.conduction_geom_code) {
        case 1:
          kirchhoff_dt_ratio_1d_kernel_body<1>(
              c, b.cond_diag3 + 0, b.cond_kappa_eff, b.cond_rho_cv_e, b.Te,
              b.cell_is_void, b.vol, b.x_r, p.n_cells);
          break;
        case 2:
          kirchhoff_dt_ratio_1d_kernel_body<2>(
              c, b.cond_diag3 + 0, b.cond_kappa_eff, b.cond_rho_cv_e, b.Te,
              b.cell_is_void, b.vol, b.x_r, p.n_cells);
          break;
        default:
          kirchhoff_dt_ratio_1d_kernel_body<0>(
              c, b.cond_diag3 + 0, b.cond_kappa_eff, b.cond_rho_cv_e, b.Te,
              b.cell_is_void, b.vol, b.x_r, p.n_cells);
          break;
      }
    }
    __threadfence();
    pk_sync(p);
  }
}

__device__ double persistent_conduction_dt_exp(const PersistentParams& p,
                                               const PersistentDeviceBuffers& b) {
  if (p.conduction_enabled == 0) {
    return CUDART_INF;
  }
  const double min_ratio = *(volatile const double*)(b.cond_diag3 + 0);
  if (isfinite(min_ratio) && min_ratio > 0.0) {
    const double dt_exp = p.cfl_cond * min_ratio;
    return (isfinite(dt_exp) && dt_exp > 0.0) ? dt_exp : CUDART_INF;
  }
  return CUDART_INF;
}

__device__ void persistent_conduction_flux_limiter_faces(
    const PersistentParams& p,
    const PersistentDeviceBuffers& b) {
  using namespace tenryu::hydro::conduction_bodies;
  if (p.conduction_enabled == 0 || p.conduction_test_kappa > 0.0 ||
      p.n_cells <= 1) {
    return;
  }
  // Face-kappa-policy consistency fix: limiter estimate follows the policy of the consuming
  // stage kernels (same guard as persistent_conduction_stage_geom).
  const bool limiter_face_kirchhoff =
      (p.conduction_face_kappa_policy != 0 && p.conduction_test_kappa <= 0.0);
  for (int face = pk_thread_id(p); face + 1 < p.n_cells; face += pk_thread_stride(p)) {
    compute_1d_flux_limiter_faces_kernel_body(
        face, b.cond_flux_limiter_faces, b.Te, b.cond_kappa_eff, b.rho,
        b.zbar, b.cond_A_eff, b.x_r, p.n_cells, p.conduction_f_lim,
        limiter_face_kirchhoff);
  }
  pk_sync(p);
}

template <int GEOM>
__device__ void persistent_conduction_stage_geom(const PersistentParams& p,
                                                 const PersistentDeviceBuffers& b,
                                                 const double* te_curr,
                                                 double* te_next,
                                                 const double tau) {
  using namespace tenryu::hydro::conduction_bodies;
  const bool power = (p.conduction_kappa_power_active != 0);
  const bool kirchhoff =
      (p.conduction_face_kappa_policy != 0 &&
       p.conduction_test_kappa <= 0.0 && !power);
  const double* flux_limiter =
      (p.conduction_test_kappa > 0.0) ? nullptr : b.cond_flux_limiter_faces;
  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    if (power) {
      conduction_1d_sts_stage_secant_legacy_inline_alpha_kernel_body<GEOM>(
          c, te_curr, te_next, b.cond_rho_cv_e, b.cell_is_void, b.vol, b.x_r,
          b.rho, p.n_cells, tau, p.Te_floor, p.conduction_test_kappa,
          p.conduction_kappa_power, p.conduction_kappa_rho_power,
          b.clamp_count, b.E_floor);
    } else if (kirchhoff) {
      conduction_1d_sts_stage_kirchhoff_legacy_inline_alpha_kernel_body<GEOM>(
          c, te_curr, te_next, b.cond_kappa_eff, flux_limiter,
          b.cond_rho_cv_e, b.cell_is_void, b.vol, b.x_r, p.n_cells, tau,
          p.Te_floor, b.clamp_count, b.E_floor);
    } else {
      conduction_1d_sts_stage_legacy_inline_alpha_kernel_body<GEOM>(
          c, te_curr, te_next, b.cond_kappa_eff, flux_limiter,
          b.cond_rho_cv_e, b.cell_is_void, b.vol, b.x_r, p.n_cells, tau,
          p.Te_floor, b.clamp_count, b.E_floor);
    }
  }
  pk_sync(p);
}

__device__ void persistent_conduction_stage(const PersistentParams& p,
                                            const PersistentDeviceBuffers& b,
                                            const double* te_curr,
                                            double* te_next,
                                            const double tau) {
  switch (p.conduction_geom_code) {
    case 1:
      persistent_conduction_stage_geom<1>(p, b, te_curr, te_next, tau);
      break;
    case 2:
      persistent_conduction_stage_geom<2>(p, b, te_curr, te_next, tau);
      break;
    default:
      persistent_conduction_stage_geom<0>(p, b, te_curr, te_next, tau);
      break;
  }
}

__device__ void persistent_conduction_sync_eos(const PersistentParams& p,
                                               const PersistentDeviceBuffers& b) {
  using namespace tenryu::hydro::conduction_bodies;
  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    eos_sync_electron_kernel_body(c, b.Te, b.ee, b.Pe, b.rho, b.zbar,
                                  b.cond_gamma_eff, b.cond_A_eff, p.n_cells,
                                  p.Te_floor, b.cv_e);
  }
  pk_sync(p);
}

__device__ void persistent_conduction_step(const PersistentParams& p,
                                           const PersistentDeviceBuffers& b,
                                           const double dt,
                                           int* error_flag) {
  using namespace tenryu::hydro::conduction_bodies;
  if (p.conduction_enabled == 0 || !(dt > 0.0)) {
    return;
  }
  if (pk_global_leader(p)) {
    b.E_floor[0] = 0.0;
    *b.clamp_count = 0;
    *b.failing_cell = p.n_cells;
  }
  __threadfence();
  pk_sync(p);

  persistent_conduction_operator(p, b);
  const double dt_exp = persistent_conduction_dt_exp(p, b);
  if (!isfinite(dt_exp) || !(dt_exp > 0.0)) {
    return;
  }
  persistent_conduction_flux_limiter_faces(p, b);

  const int smax = (p.conduction_sts_max_stages > 0)
                       ? ((p.conduction_sts_max_stages > 1)
                              ? p.conduction_sts_max_stages
                              : 1)
                       : 100000;
  int n_sub = 1;
  double dt_sub = dt;
  if (p.conduction_sts_max_stages > 0) {
    const double eta = fmin(1.0, fmax(p.conduction_sts_subcycle_eta, 1.0e-6));
    const double smax_factor =
        0.5 * static_cast<double>(smax) * static_cast<double>(smax + 1);
    const double dt_sts_max = smax_factor * dt_exp * eta;
    if (isfinite(dt_sts_max) && dt_sts_max > 0.0 && dt > dt_sts_max) {
      n_sub = ceil_to_int_clamped(dt / dt_sts_max);
      dt_sub = dt / static_cast<double>(n_sub);
    }
  }
  const int stages =
      sts_stage_count(dt_sub, dt_exp, p.conduction_sts_max_stages);

  double* te_curr = b.Te;
  double* te_next = b.cond_te_tmp;
  for (int sub = 0; sub < n_sub; ++sub) {
    for (int stage = 0; stage < stages; ++stage) {
      const double tau = sts_tau_for_stage(dt_sub, dt_exp, stages,
                                           p.conduction_sts_damping, stage);
      persistent_conduction_stage(p, b, te_curr, te_next, tau);
      double* tmp = te_curr;
      te_curr = te_next;
      te_next = tmp;
      pk_sync(p);
    }
  }

  if (te_curr != b.Te) {
    for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
      b.Te[c] = te_curr[c];
    }
    pk_sync(p);
  }

  persistent_conduction_sync_eos(p, b);
  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    if (!isfinite(b.Te[c]) || !isfinite(b.ee[c]) || !isfinite(b.Pe[c])) {
      atomicMin(b.failing_cell, c);
      pk_raise_error(error_flag,
                     pk_encode_error_code(kPkErrorConduction));
    }
  }
  pk_sync(p);
}

__device__ double persistent_fld_dt_rad(const PersistentParams& p,
                                        const PersistentDeviceBuffers& b,
                                        double* smem_v,
                                        int* smem_i) {
  if (p.radiation_enabled == 0) {
    return CUDART_INF;
  }
  double local_min = CUDART_INF;
  int local_idx = INT_MAX;
  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    if (b.cell_is_void != nullptr && b.cell_is_void[c] != 0U) {
      continue;
    }
    const double rho_raw = b.rho[c];
    if (!(rho_raw >= p.rho_floor)) {
      continue;
    }
    const double sigma_P =
        fmax(rho_raw, 0.0) * persistent_constant_opacity_kappa_a(p, b, c);
    if (!(sigma_P > 0.0)) {
      continue;
    }
    const double Te_c = fmax(b.Te[c], p.Te_floor);
    const bool has_cv_e_cell =
        p.fld_has_cv_e != 0 && b.cv_e != nullptr && b.cv_e[c] > 0.0;
    double Cv_e = -1.0;
    if (p.cv_e_override > 0.0) {
      Cv_e = p.cv_e_override;
    } else if (has_cv_e_cell) {
      Cv_e = fmax(rho_raw, 0.0) * b.cv_e[c];
    } else {
      const double z = fmax(b.zbar[c], 0.0);
      const double A_c = (b.A_eff != nullptr) ? fmax(b.A_eff[c], 1.0e-12)
                                              : fmax(p.material_A, 1.0e-12);
      const double gm1_c =
          (b.gamma_eff != nullptr)
              ? fmax(b.gamma_eff[c] - 1.0, 1.0e-12)
              : fmax(p.material_gm1, 1.0e-12);
      const double cv_mass_e =
          z * core::constants::eV_to_erg /
          (A_c * core::constants::proton_mass * gm1_c);
      Cv_e = fmax(rho_raw, 0.0) * cv_mass_e;
    }
    Cv_e = fmax(Cv_e, 1.0e-30);
    double beta =
        4.0 * core::constants::a_eV * Te_c * Te_c * Te_c / Cv_e;
    if (p.cv_e_override <= 0.0 && !has_cv_e_cell) {
      beta = fmin(beta, 1.0);
    }
    const double denom =
        p.fld_f_min * p.fld_alpha * core::constants::c_light * beta * sigma_P;
    if (!(denom > 0.0)) {
      continue;
    }
    const double candidate = (1.0 - p.fld_f_min) / denom;
    if (candidate < local_min ||
        (candidate == local_min && c < local_idx)) {
      local_min = candidate;
      local_idx = c;
    }
  }
  double min_local = CUDART_INF;
  int min_idx = INT_MAX;
  pk_reduce_argmin(p, b, local_min, local_idx, smem_v, smem_i, &min_local,
                   &min_idx);
  (void)min_idx;
  return min_local;
}

template <int GEOM>
__device__ void persistent_fld_assemble_geom(const PersistentParams& p,
                                             const PersistentDeviceBuffers& b,
                                             const double dt) {
  using namespace tenryu::radiation::fld_1d_bodies;
  const int total = p.n_cells * p.n_groups;
  for (int idx = pk_thread_id(p); idx < total; idx += pk_thread_stride(p)) {
    assemble_fld_tridiag_kernel_body<GEOM>(
        idx, b.x_r, b.vol, b.fld_sigma_a,
        (p.fld_use_fleck_blend != 0) ? b.fld_sigma_a : nullptr,
        (p.fld_use_fleck_blend != 0) ? b.fld_fleck : nullptr, b.fld_eta,
        b.rad_E_old, b.rad_E, b.fld_sigma_R, b.fld_lower, b.fld_diag,
        b.fld_upper, b.fld_rhs, p.n_cells, p.n_groups, dt, p.fld_outer_bc,
        b.fld_marshak_finc, p.fld_volume_source_rate,
        p.fld_volume_source_r_max, p.fld_opacity_floor, p.fld_flux_limiter);
  }
  pk_sync(p);
}

__device__ void persistent_fld_assemble(const PersistentParams& p,
                                        const PersistentDeviceBuffers& b,
                                        const double dt) {
  switch (p.geom_code) {
    case 1:
      persistent_fld_assemble_geom<1>(p, b, dt);
      break;
    case 2:
      persistent_fld_assemble_geom<2>(p, b, dt);
      break;
    default:
      persistent_fld_assemble_geom<0>(p, b, dt);
      break;
  }
}

__device__ double persistent_fld_outer_area(const PersistentParams& p,
                                            const PersistentDeviceBuffers& b) {
  const double r = b.x_r[p.n_cells];
  if (p.geom_code == 0) {
    return 4.0 * radiation::fld_1d_bodies::kPi * r * r;
  }
  return mesh::geometry_1d_face_area(p.geom_code, r);
}

template <bool kGrid>
__device__ void persistent_fld_step(const PersistentParams& p,
                                    const PersistentDeviceBuffers& b,
                                    const double dt,
                                    const double t_op,
                                    const int local_step,
                                    int* phase_prof_last_phase,
                                    unsigned long long* phase_prof_last_clock,
                                    int* error_flag,
                                    double* reduce_smem,
                                    double* fld_update_smem,
                                    double* out_residual,
                                    int* out_iterations,
                                    double* out_escaped,
                                    double* out_marshak_in,
                                    double* out_volume_source) {
  using namespace tenryu::radiation::fld_1d_bodies;
  if (p.radiation_enabled == 0 || !(dt > 0.0)) {
    if (threadIdx.x == 0) {
      *out_residual = 0.0;
      *out_iterations = 0;
      *out_escaped = 0.0;
      *out_marshak_in = 0.0;
      *out_volume_source = 0.0;
    }
    pk_sync(p);
    return;
  }

  const int total = p.n_cells * p.n_groups;
  for (int idx = pk_thread_id(p); idx < total; idx += pk_thread_stride(p)) {
    b.rad_E_old[idx] = b.rad_E[idx];
  }
  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    snapshot_Te_kernel_body(c, b.Te, b.fld_Te_old, p.n_cells);
  }
  for (int g = pk_thread_id(p); g < p.n_groups; g += pk_thread_stride(p)) {
    double const_flux = p.fld_marshak_flux;
    if (p.fld_marshak_pulse_duration >= 0.0 &&
        t_op >= p.fld_marshak_pulse_duration) {
      const_flux = 0.0;
    }
    compute_marshak_finc_kernel_body(g, p.planck, p.fld_marshak_Tr_eV,
                                     const_flux, b.fld_marshak_finc,
                                     p.n_groups);
  }
  pk_sync(p);

  __shared__ int sh_converged;
  __shared__ double sh_residual;
  __shared__ int sh_iterations;
  double* const s_F_all = fld_update_smem;
  double* const s_dF_all = fld_update_smem + kUpdateMatterWarps * p.n_groups;
  if (pk_global_leader(p)) {
    b.grid_broadcast_ints[0] = 0;
    b.grid_broadcast_ints[1] = 0;
    b.grid_broadcast_doubles[0] = CUDART_INF;
  }
  __threadfence();
  pk_sync(p);
  if (threadIdx.x == 0) {
    sh_converged = *(volatile const int*)(b.grid_broadcast_ints + 0);
    sh_iterations = *(volatile const int*)(b.grid_broadcast_ints + 1);
    sh_residual = *(volatile const double*)(b.grid_broadcast_doubles + 0);
  }
  pk_sync(p);

  const int max_iter =
      (p.fld_max_outer_iterations > 1) ? p.fld_max_outer_iterations : 1;
  for (int iter = 0; iter < max_iter; ++iter) {
    if (sh_converged != 0) {
      break;
    }
    pk_phase_prof_mark(p, kPkWatchPhaseFldOpacity, phase_prof_last_phase,
                       phase_prof_last_clock);
    pk_watch_mark(p, kPkWatchPhaseFldOpacity, local_step, iter, 0);
    if (p.phase_trace && pk_global_leader(p)) {
      printf("pk_phase step=%d fld_iter %d opacity\n", local_step, iter);
    }
    if (p.fld_use_nlte_table != 0) {
      for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
        eval_nlte_opacity_emission_kernel_body(
            c, b.rho, b.Te, b.zbar, (p.fld_has_cv_e != 0) ? b.cv_e : nullptr,
            b.cell_is_void, p.nlte_opacity, p.planck, b.fld_fleck,
            b.fld_sigma_a, b.fld_sigma_pe, b.fld_sigma_R,
            b.fld_nlte_sigma_eff_work, b.fld_nlte_sigma_s_eff_work,
            b.fld_nlte_eta_cdf_work, b.fld_eta, b.fld_nlte_lambda_work,
            p.n_cells, p.n_groups, dt, p.material_A, p.fld_alpha,
            p.fld_opacity_cap, p.cv_e_override, p.Te_floor, p.material_gm1);
      }
      pk_sync(p);
    } else {
      for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
        persistent_eval_constant_opacity_cell(p, b, c);
      }
      pk_sync(p);

      pk_phase_prof_mark(p, kPkWatchPhaseFldEta, phase_prof_last_phase,
                         phase_prof_last_clock);
      pk_watch_mark(p, kPkWatchPhaseFldEta, local_step, iter, 0);
      if (p.phase_trace && pk_global_leader(p)) {
        printf("pk_phase step=%d fld_iter %d eta\n", local_step, iter);
      }
      for (int idx = pk_thread_id(p); idx < total; idx += pk_thread_stride(p)) {
        b.fld_sigma_pe[idx] = b.fld_sigma_a[idx];
        build_eta_from_planck_kernel_body(idx, b.Te, b.fld_sigma_a, p.planck,
                                          b.fld_eta, p.n_cells, p.n_groups,
                                          p.Te_floor);
      }
      pk_sync(p);

      pk_phase_prof_mark(p, kPkWatchPhaseFldFleck, phase_prof_last_phase,
                         phase_prof_last_clock);
      pk_watch_mark(p, kPkWatchPhaseFldFleck, local_step, iter, 0);
      if (p.phase_trace && pk_global_leader(p)) {
        printf("pk_phase step=%d fld_iter %d fleck\n", local_step, iter);
      }
      if (p.fld_use_fleck != 0) {
        for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
          // Persistent FLD path does not support table EOS yet: its
          // update_matter stage passes an empty DeviceEOSTableView (see the
          // update_matter_body_persistent call below), so the Fleck factor must
          // stay on the legacy cv chain (use_table_cv=0) until both stages gain
          // table support together (fleck_cv_default_flip_20260711.md section 3).
          // fleck_beta secant is likewise inert here (needs the table EOS
          // branch); pass the tangent-only args explicitly. fleck_form is
          // live in this path (phi_1 needs no table).
          compute_fleck_for_fld_kernel_body(
              c, b.rho, b.Te, b.zbar, b.cell_is_void, b.fld_sigma_a,
              (p.fld_has_cv_e != 0) ? b.cv_e : nullptr, b.fld_fleck,
              p.n_cells, p.n_groups, dt, 1.0, p.cv_e_override, b.gamma_eff,
              b.A_eff, materials::DeviceEOSTableView{}, 0, p.Te_floor,
              nullptr, radiation::PlanckTableDeviceView{}, 0,
              p.fld_fleck_form_exp);
        }
        pk_sync(p);
      }
    }

    pk_phase_prof_mark(p, kPkWatchPhaseFldAssemble, phase_prof_last_phase,
                       phase_prof_last_clock);
    pk_watch_mark(p, kPkWatchPhaseFldAssemble, local_step, iter, 0);
    if (p.phase_trace && pk_global_leader(p)) {
      printf("pk_phase step=%d fld_iter %d assemble\n", local_step, iter);
    }
    persistent_fld_assemble(p, b, dt);

    pk_phase_prof_mark(p, kPkWatchPhaseFldThomas, phase_prof_last_phase,
                       phase_prof_last_clock);
    pk_watch_mark(p, kPkWatchPhaseFldThomas, local_step, iter, 0);
    if (p.phase_trace && pk_global_leader(p)) {
      printf("pk_phase step=%d fld_iter %d thomas\n", local_step, iter);
    }
    if constexpr (kGrid) {
      if (p.n_cells >= 64) {
        core::pcr_solve_strided<true>(
            b.fld_lower, b.fld_diag, b.fld_upper, b.fld_rhs,
            b.fld_pcr_dl_work, b.fld_pcr_d_work, b.fld_pcr_du_work,
            b.fld_pcr_rhs_work, p.n_cells, p.n_cells, p.n_groups,
            pk_thread_id(p), pk_thread_stride(p));
      } else {
        for (int g = pk_thread_id(p); g < p.n_groups;
             g += pk_thread_stride(p)) {
          core::thomas_solve_strided(b.fld_lower, b.fld_diag, b.fld_upper,
                                     b.fld_rhs, b.fld_cp_work, p.n_cells,
                                     p.n_cells, g);
        }
        pk_sync(p);
      }
    } else {
      for (int g = pk_thread_id(p); g < p.n_groups; g += pk_thread_stride(p)) {
        core::thomas_solve_strided(b.fld_lower, b.fld_diag, b.fld_upper,
                                   b.fld_rhs, b.fld_cp_work, p.n_cells,
                                   p.n_cells, g);
      }
      pk_sync(p);
    }

    pk_phase_prof_mark(p, kPkWatchPhaseFldPublish, phase_prof_last_phase,
                       phase_prof_last_clock);
    pk_watch_mark(p, kPkWatchPhaseFldPublish, local_step, iter, 0);
    if (p.phase_trace && pk_global_leader(p)) {
      printf("pk_phase step=%d fld_iter %d publish\n", local_step, iter);
    }
    for (int idx = pk_thread_id(p); idx < total; idx += pk_thread_stride(p)) {
      publish_solution_kernel_body(idx, b.fld_rhs, b.rad_E, p.n_cells,
                                   p.n_groups);
    }
    pk_sync(p);

    pk_phase_prof_mark(p, kPkWatchPhaseFldUpdateMatter, phase_prof_last_phase,
                       phase_prof_last_clock);
    pk_watch_mark(p, kPkWatchPhaseFldUpdateMatter, local_step, iter, 0);
    if (p.phase_trace && pk_global_leader(p)) {
      printf("pk_phase step=%d fld_iter %d update_matter\n", local_step, iter);
    }
    const int local_warp = threadIdx.x / kUpdateMatterWarpSize;
    const int lane = threadIdx.x - local_warp * kUpdateMatterWarpSize;
    const int global_warp = pk_thread_id(p) / kUpdateMatterWarpSize;
    const int warp_stride = pk_thread_stride(p) / kUpdateMatterWarpSize;
    for (int c = global_warp; c < p.n_cells; c += warp_stride) {
      if (c < p.n_cells) {
        update_matter_body_persistent(
            c, lane, s_F_all + local_warp * p.n_groups,
            s_dF_all + local_warp * p.n_groups, b.rho, b.vol, b.zbar,
            (p.fld_has_cv_e != 0) ? b.cv_e : nullptr, b.fld_Te_old,
            b.fld_sigma_a, (p.fld_use_fleck != 0) ? b.fld_sigma_pe : nullptr,
            b.rad_E,
            (p.fld_use_fleck_blend != 0) ? b.fld_fleck : nullptr, b.rad_E_old,
            p.planck, materials::DeviceEOSTableView{}, b.Te, b.ee, b.Pe,
            b.rad_dep, b.rad_emit, b.fld_delta_T, p.n_cells, p.n_groups, dt,
            b.A_eff, b.gamma_eff, p.cv_e_override, p.Te_floor,
            p.fld_has_cv_e, p.fld_max_outer_iterations, p.fld_outer_tol);
      }
    }
    pk_sync(p);

    pk_phase_prof_mark(p, kPkWatchPhaseFldReduce, phase_prof_last_phase,
                       phase_prof_last_clock);
    pk_watch_mark(p, kPkWatchPhaseFldReduce, local_step, iter, 0);
    if (p.phase_trace && pk_global_leader(p)) {
      printf("pk_phase step=%d fld_iter %d reduce\n", local_step, iter);
    }
    double local_residual = 0.0;
    for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
      local_residual = fmax(
          local_residual,
          fmax(radiation::fld_1d_bodies::finite_or_zero(b.fld_delta_T[c]),
               0.0));
    }
    const double residual = pk_reduce_max(p, b, local_residual, reduce_smem);
    if (pk_global_leader(p)) {
      b.grid_broadcast_doubles[0] = residual;
      b.grid_broadcast_ints[0] = (residual < p.fld_outer_tol) ? 1 : 0;
      b.grid_broadcast_ints[1] = iter + 1;
    }
    __threadfence();
    pk_sync(p);
    if (threadIdx.x == 0) {
      sh_converged = *(volatile const int*)(b.grid_broadcast_ints + 0);
      sh_iterations = *(volatile const int*)(b.grid_broadcast_ints + 1);
      sh_residual = *(volatile const double*)(b.grid_broadcast_doubles + 0);
    }
    pk_sync(p);
  }

  double local_escaped = 0.0;
  for (int g = pk_thread_id(p); g < p.n_groups; g += pk_thread_stride(p)) {
    const double E =
        fmax(finite_or_zero(b.rad_E[(p.n_cells - 1) * p.n_groups + g]), 0.0);
    local_escaped +=
        dt * persistent_fld_outer_area(p, b) *
        fld_1d_outer_leak_coeff(p.fld_outer_bc) * E;
  }
  const double escaped = pk_reduce_sum(p, b, local_escaped, reduce_smem);

  double local_volume_source = 0.0;
  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    if (p.fld_volume_source_rate > 0.0 && p.fld_volume_source_r_max > 0.0) {
      const double r_c = 0.5 * (b.x_r[c] + b.x_r[c + 1]);
      if (r_c <= p.fld_volume_source_r_max) {
        local_volume_source += dt * fmax(finite_or_zero(b.vol[c]), 0.0) *
                               p.fld_volume_source_rate;
      }
    }
  }
  const double volume_source =
      pk_reduce_sum(p, b, local_volume_source, reduce_smem);

  if (threadIdx.x == 0) {
    double marshak_flux_total = 0.0;
    double const_flux = p.fld_marshak_flux;
    if (p.fld_marshak_pulse_duration >= 0.0 &&
        t_op >= p.fld_marshak_pulse_duration) {
      const_flux = 0.0;
    }
    if (p.fld_outer_bc == kFld1dOuterMarshak) {
      if (p.fld_marshak_Tr_eV > 0.0) {
        const double T2 = p.fld_marshak_Tr_eV * p.fld_marshak_Tr_eV;
        marshak_flux_total = 0.25 * core::constants::c_light *
                             core::constants::a_eV * T2 * T2;
      } else {
        marshak_flux_total = fmax(const_flux, 0.0);
      }
    }
    *out_residual = sh_residual;
    *out_iterations = sh_iterations;
    *out_escaped = escaped;
    *out_marshak_in = dt * persistent_fld_outer_area(p, b) *
                      marshak_flux_total;
    *out_volume_source = volume_source;
    if (!isfinite(sh_residual)) {
      pk_raise_error(error_flag,
                     pk_encode_error_code(kPkErrorFldDivergence));
    }
  }
  pk_sync(p);
}

__device__ void persistent_gamma_r_pre_hydro(const PersistentParams& p,
                                             const PersistentDeviceBuffers& b) {
  if (p.rad_gamma43_enabled == 0) {
    return;
  }
  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    b.rad_gamma_vol_before[c] = b.vol[c];
    rad_gamma::radiation_pressure_field_kernel_body(
        c, b.rad_gamma_p_r, b.rad_E, p.n_cells, p.n_groups);
  }
  pk_sync(p);
}

__device__ void persistent_hydro_half_step(const PersistentParams& p,
                                           const PersistentDeviceBuffers& b,
                                           const double dt,
                                           const double t_op,
                                           int* error_flag,
                                           double* smem) {
  using namespace tenryu::hydro::persistent_1d;
  (void)t_op;

  if (pk_global_leader(p)) {
    b.E_floor[0] = 0.0;
    *b.clamp_count = 0;
    *b.rho_clamp_count = 0;
  }
  pk_sync(p);

  refresh_geometry_density_closure(p, b);
  const double E_before =
      (p.renorm_active != 0 && p.two_temperature == 0)
          ? block_total_energy_1d(p, b, smem)
          : 0.0;

  compute_sound_speed_into(p, b, b.cs);
  compute_q_vnr_into(p, b, b.cs, b.Qvisc);

  // C1 predicate refuses adaptive AV, bulk viscosity, shock-history heat,
  // odd-even velocity damping, high-k damping, Braginskii, MPI halo exchange,
  // compatible-energy, and all verbose probes.

  for (int j = pk_thread_id(p); j < p.n_nodes; j += pk_thread_stride(p)) {
    copy_array_kernel_body(j, b.r_old, b.x_r, p.n_nodes);
    copy_array_kernel_body(j, b.u_old, b.v_r, p.n_nodes);
  }
  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    copy_array_kernel_body(c, b.V_old, b.vol, p.n_cells);
    copy_array_kernel_body(c, b.e_old, b.ee, p.n_cells);
    if (p.two_temperature != 0) {
      copy_array_kernel_body(c, b.ei_old, b.ei, p.n_cells);
    }
    copy_array_kernel_body(c, b.Pe_old, b.Pe, p.n_cells);
    copy_array_kernel_body(c, b.Pi_old, b.Pi, p.n_cells);
    copy_array_kernel_body(c, b.Q_old, b.Qvisc, p.n_cells);
  }
  pk_sync(p);

  for (int j = pk_thread_id(p); j < p.n_nodes; j += pk_thread_stride(p)) {
    b.node_active[j] = 0u;
  }
  pk_sync(p);
  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    compute_node_activity_kernel_body(c, b.node_active, b.hydro_active,
                                      p.n_cells);
  }
  pk_sync(p);
  if (pk_global_leader(p)) {
    zero_center_node_kernel_body(0, b.node_active, p.n_nodes);
  }
  pk_sync(p);

  if (p.rad_gamma43_enabled != 0) {
    for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
      build_cell_pq_kernel_body<true>(c, b.pq_n, b.Pe, b.Pi, b.Qvisc,
                                      b.rad_gamma_p_r, b.hydro_active,
                                      p.n_cells);
    }
  } else {
    for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
      build_cell_pq_kernel_body<false>(c, b.pq_n, b.Pe, b.Pi, b.Qvisc,
                                       nullptr, b.hydro_active, p.n_cells);
    }
  }
  pk_sync(p);
  // C1 predicate refuses odd-even damping, so checkerboard pq filtering is off.

  const double ghost_pq_n = ghost_pq_device(p, b);
  for (int j = pk_thread_id(p); j < p.n_nodes; j += pk_thread_stride(p)) {
    switch (p.geom_code) {
      case 1:
        compute_acceleration_1d_kernel_body<1>(
            j, b.a_n, b.mass, b.x_r, b.pq_n, b.node_active, p.n_cells,
            ghost_pq_n, p.boundary_type);
        break;
      case 2:
        compute_acceleration_1d_kernel_body<2>(
            j, b.a_n, b.mass, b.x_r, b.pq_n, b.node_active, p.n_cells,
            ghost_pq_n, p.boundary_type);
        break;
      default:
        compute_acceleration_1d_kernel_body<0>(
            j, b.a_n, b.mass, b.x_r, b.pq_n, b.node_active, p.n_cells,
            ghost_pq_n, p.boundary_type);
        break;
    }
  }
  pk_sync(p);

  for (int j = pk_thread_id(p); j < p.n_nodes; j += pk_thread_stride(p)) {
    predictor_update_kernel_body(j, b.v_r, b.x_r, b.u_half, b.u_old,
                                 b.r_old, b.a_n, b.node_active, p.n_nodes, dt);
  }
  pk_sync(p);
  if (pk_global_leader(p)) {
    propagate_void_node_displacement_kernel_body(0, b.x_r, b.v_r, b.r_old,
                                                b.node_active, p.n_nodes);
  }
  pk_sync(p);
  apply_boundary_device(p, b);
  persistent_follow_void_region_nodes_1d(p, b);
  pk_sync(p);
  for (int j = pk_thread_id(p); j < p.n_nodes; j += pk_thread_stride(p)) {
    copy_array_kernel_body(j, b.u_half, b.v_r, p.n_nodes);
  }
  pk_sync(p);

  refresh_geometry_density_closure(p, b);
  compute_sound_speed_into(p, b, b.cs_half);
  compute_q_vnr_into(p, b, b.cs_half, b.Qvisc);

  for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
    average_arrays_kernel_body(c, b.P_half, b.Pe_old, b.Pe, p.n_cells);
    average_arrays_kernel_body(c, b.Pi_half, b.Pi_old, b.Pi, p.n_cells);
    average_arrays_kernel_body(c, b.Q_half, b.Q_old, b.Qvisc, p.n_cells);
  }
  pk_sync(p);
  // C1 predicate refuses artificial AV heat and ion artificial heat.

  if (p.two_temperature != 0) {
    for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
      copy_array_kernel_body(c, b.rho_half, b.rho, p.n_cells);
      copy_array_kernel_body(c, b.Te_half, b.Te, p.n_cells);
      copy_array_kernel_body(c, b.Ti_half, b.Ti, p.n_cells);
    }
    pk_sync(p);
  }

  if (p.rad_gamma43_enabled != 0) {
    for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
      build_cell_pq_kernel_body<true>(c, b.pq_half, b.Pe, b.Pi, b.Qvisc,
                                      b.rad_gamma_p_r, b.hydro_active,
                                      p.n_cells);
    }
  } else {
    for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
      build_cell_pq_kernel_body<false>(c, b.pq_half, b.Pe, b.Pi, b.Qvisc,
                                       nullptr, b.hydro_active, p.n_cells);
    }
  }
  pk_sync(p);
  // C1 predicate refuses odd-even damping, so checkerboard pq filtering is off.

  const double ghost_pq_half = ghost_pq_device(p, b);
  for (int j = pk_thread_id(p); j < p.n_nodes; j += pk_thread_stride(p)) {
    switch (p.geom_code) {
      case 1:
        compute_acceleration_1d_kernel_body<1>(
            j, b.a_half, b.mass, b.x_r, b.pq_half, b.node_active, p.n_cells,
            ghost_pq_half, p.boundary_type);
        break;
      case 2:
        compute_acceleration_1d_kernel_body<2>(
            j, b.a_half, b.mass, b.x_r, b.pq_half, b.node_active, p.n_cells,
            ghost_pq_half, p.boundary_type);
        break;
      default:
        compute_acceleration_1d_kernel_body<0>(
            j, b.a_half, b.mass, b.x_r, b.pq_half, b.node_active, p.n_cells,
            ghost_pq_half, p.boundary_type);
        break;
    }
  }
  pk_sync(p);

  if (p.rad_gamma43_enabled != 0) {
    for (int j = pk_thread_id(p); j < p.n_nodes; j += pk_thread_stride(p)) {
      copy_array_kernel_body(j, b.rad_gamma_r_half, b.x_r, p.n_nodes);
    }
  }
  pk_sync(p);

  for (int j = pk_thread_id(p); j < p.n_nodes; j += pk_thread_stride(p)) {
    corrector_update_kernel_body(j, b.v_r, b.x_r, b.u_old, b.r_old,
                                 b.u_half, b.a_half, b.node_active, p.n_nodes,
                                 dt);
  }
  pk_sync(p);
  if (p.rad_gamma43_enabled != 0) {
    for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
      switch (p.geom_code) {
        case 1:
          gamma_r_force_work_1d_kernel_body<1>(
              c, b.rad_gamma_W_r, b.rad_gamma_p_r, b.rad_gamma_r_half,
              b.u_old, b.v_r, b.node_active, p.n_cells, dt);
          break;
        case 2:
          gamma_r_force_work_1d_kernel_body<2>(
              c, b.rad_gamma_W_r, b.rad_gamma_p_r, b.rad_gamma_r_half,
              b.u_old, b.v_r, b.node_active, p.n_cells, dt);
          break;
        default:
          gamma_r_force_work_1d_kernel_body<0>(
              c, b.rad_gamma_W_r, b.rad_gamma_p_r, b.rad_gamma_r_half,
              b.u_old, b.v_r, b.node_active, p.n_cells, dt);
          break;
      }
    }
  }
  pk_sync(p);
  if (pk_global_leader(p)) {
    propagate_void_node_displacement_kernel_body(0, b.x_r, b.v_r, b.r_old,
                                                b.node_active, p.n_nodes);
  }
  pk_sync(p);
  apply_boundary_device(p, b);
  persistent_follow_void_region_nodes_1d(p, b);
  pk_sync(p);

  refresh_geometry_density_closure(p, b);

  if (p.two_temperature != 0) {
    const tenryu::materials::DeviceEOSTableView no_table{};
    for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
      switch (p.geom_code) {
        case 1:
          energy_update_with_old_volume_2t_kernel_body<1>(
              c, b.ee, b.ei, b.e_old, b.ei_old, b.rho_half, b.Te_half,
              b.Ti_half, b.x_r, b.r_old, b.u_half, b.vol, b.V_old, b.mass,
              b.P_half, b.Pi_half, b.Q_half, b.zbar, b.hydro_active,
              p.n_cells, dt, b.gamma_eff, b.A_eff, p.fallback_z, no_table,
              no_table, nullptr, nullptr, p.q_heat_to_electron,
              0, nullptr, b.E_floor, b.clamp_count);
          break;
        case 2:
          energy_update_with_old_volume_2t_kernel_body<2>(
              c, b.ee, b.ei, b.e_old, b.ei_old, b.rho_half, b.Te_half,
              b.Ti_half, b.x_r, b.r_old, b.u_half, b.vol, b.V_old, b.mass,
              b.P_half, b.Pi_half, b.Q_half, b.zbar, b.hydro_active,
              p.n_cells, dt, b.gamma_eff, b.A_eff, p.fallback_z, no_table,
              no_table, nullptr, nullptr, p.q_heat_to_electron,
              0, nullptr, b.E_floor, b.clamp_count);
          break;
        default:
          energy_update_with_old_volume_2t_kernel_body<0>(
              c, b.ee, b.ei, b.e_old, b.ei_old, b.rho_half, b.Te_half,
              b.Ti_half, b.x_r, b.r_old, b.u_half, b.vol, b.V_old, b.mass,
              b.P_half, b.Pi_half, b.Q_half, b.zbar, b.hydro_active,
              p.n_cells, dt, b.gamma_eff, b.A_eff, p.fallback_z, no_table,
              no_table, nullptr, nullptr, p.q_heat_to_electron,
              0, nullptr, b.E_floor, b.clamp_count);
          break;
      }
    }
  } else {
    for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
      switch (p.geom_code) {
        case 1:
          energy_update_with_old_volume_kernel_body<1>(
              c, b.ee, b.e_old, b.x_r, b.r_old, b.u_half, b.vol, b.V_old,
              b.mass, b.P_half, b.Q_half, b.hydro_active, p.n_cells, dt,
              p.compatible_energy, nullptr, b.E_floor, b.clamp_count);
          break;
        case 2:
          energy_update_with_old_volume_kernel_body<2>(
              c, b.ee, b.e_old, b.x_r, b.r_old, b.u_half, b.vol, b.V_old,
              b.mass, b.P_half, b.Q_half, b.hydro_active, p.n_cells, dt,
              p.compatible_energy, nullptr, b.E_floor, b.clamp_count);
          break;
        default:
          energy_update_with_old_volume_kernel_body<0>(
              c, b.ee, b.e_old, b.x_r, b.r_old, b.u_half, b.vol, b.V_old,
              b.mass, b.P_half, b.Q_half, b.hydro_active, p.n_cells, dt,
              p.compatible_energy, nullptr, b.E_floor, b.clamp_count);
          break;
      }
    }
  }
  pk_sync(p);
  // Boundary-PdV/budget ledgers and positivity history wiring are C3 concerns.

  if (p.renorm_active != 0 && p.two_temperature == 0) {
    const double E_after = block_total_energy_1d(p, b, smem);
    const double active_mass = block_active_mass_1d(p, b, smem);
    const double active_internal = block_active_internal_1d(p, b, smem);
    __shared__ int renorm_mode;
    __shared__ double renorm_value;
    if (threadIdx.x == 0) {
      renorm_mode = 0;
      renorm_value = 0.0;
      const double delta_E = E_before - E_after;
      if (fabs(delta_E) > 0.0) {
        if (active_internal > 0.0) {
          renorm_mode = 1;
          renorm_value = (active_internal + delta_E) / active_internal;
        } else if (active_mass > 0.0) {
          renorm_mode = 2;
          renorm_value = delta_E / active_mass;
        }
      }
    }
    pk_sync(p);
    for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
      if (renorm_mode == 1) {
        scale_active_energy_kernel_body(c, b.ee, b.hydro_active, renorm_value);
      } else if (renorm_mode == 2) {
        shift_active_energy_kernel_body(c, b.ee, b.hydro_active, renorm_value);
      }
    }
    pk_sync(p);

    const double E_corrected = block_total_energy_1d(p, b, smem);
    if (pk_global_leader(p)) {
      const double residual = E_before - E_corrected;
      if (fabs(residual) > 0.0) {
        add_first_active_residual_kernel_body(b.ee, b.mass, 0, residual);
      }
    }
    pk_sync(p);
  }

  refresh_geometry_density_closure(p, b);
  // The multi-kernel path writes cs_new then moves it into state.cs. C1 writes
  // directly into state.cs because Qvisc reads cs only after the full cs loop sync.
  compute_sound_speed_into(p, b, b.cs);
  compute_q_vnr_into(p, b, b.cs, b.Qvisc);
  scan_persistent_error(p, b, error_flag);
  pk_sync(p);
  if (p.rad_gamma43_enabled != 0 &&
      *(volatile const int*)error_flag == 0) {
    for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
      rad_gamma::gamma_r_43_work_update_kernel_body(
          c, b.rad_E, b.rad_gamma_W_r, b.rad_gamma_vol_before, b.vol,
          p.n_cells, p.n_groups, b.E_floor);
    }
  }
  pk_sync(p);
}

template <bool kGrid>
__launch_bounds__(kBlockSize, 1) __global__ void
persistent_chunk_kernel(PersistentParams p,
                        PersistentDeviceBuffers b,
                        PersistentDiagRecord* ring,
                        int* d_exit,
                        double* d_t_dt) {
#define PK_PHASE_TRACE(phase_id, name)                                       \
  do {                                                                       \
    pk_phase_prof_mark(p, phase_id, &phase_prof_last_phase,                  \
                       &phase_prof_last_clock);                              \
    pk_watch_mark(p, phase_id, local_step, -1, 0);                           \
    if (p.phase_trace && pk_global_leader(p)) {                               \
      printf("pk_phase step=%d %s\n", local_step, name);                    \
    }                                                                        \
  } while (0)
  extern __shared__ double fld_update_smem[];
  __shared__ double smem[kBlockSize];
  __shared__ double smem_arg[kBlockSize];
  __shared__ int smem_i[kBlockSize];
  __shared__ double sh_t;
  __shared__ double sh_dt;
  __shared__ double sh_cand_hydro;
  __shared__ double sh_cand_cond;
  __shared__ double sh_cand_rad;
  __shared__ double sh_fld_residual;
  __shared__ double sh_fld_escaped;
  __shared__ double sh_fld_marshak_in;
  __shared__ double sh_fld_volume_source;
  __shared__ double sh_laser_input;
  __shared__ double sh_laser_escaped;
  __shared__ double sh_laser_floor;
  __shared__ double sh_laser_skipped;
  __shared__ int sh_limiter;
  __shared__ int sh_exit;
  __shared__ int sh_error;
  __shared__ int sh_fld_iterations;

  int steps_done = 0;
  int phase_prof_last_phase = -1;
  unsigned long long phase_prof_last_clock = 0;
  if (threadIdx.x == 0) {
    sh_t = d_t_dt[0];
    sh_dt = d_t_dt[1];
    sh_cand_hydro = CUDART_INF;
    sh_cand_cond = CUDART_INF;
    sh_cand_rad = CUDART_INF;
    sh_fld_residual = 0.0;
    sh_fld_escaped = 0.0;
    sh_fld_marshak_in = 0.0;
    sh_fld_volume_source = 0.0;
    sh_laser_input = 0.0;
    sh_laser_escaped = 0.0;
    sh_laser_floor = 0.0;
    sh_laser_skipped = 0.0;
    sh_limiter = kDtLimiterUnknown;
    sh_exit = kExitChunkBudget;
    sh_error = 0;
    sh_fld_iterations = 0;
  }
  if (pk_global_leader(p)) {
    *b.grid_error = 0;
  }
  __threadfence();
  pk_sync(p);
  int* const error_flag = kGrid ? b.grid_error : &sh_error;

  for (int local_step = 0; local_step < p.chunk_steps; ++local_step) {
    if (threadIdx.x == 0) {
      const double remaining_out = p.t_next_output - sh_t;
      const double remaining_end = p.t_end - sh_t;
      const double output_eps =
          1.0e-14 * fmax(fabs(sh_t), fabs(p.t_next_output));
      const double end_eps = 1.0e-14 * fmax(fabs(sh_t), fabs(p.t_end));
      sh_exit = kExitChunkBudget;
      if (remaining_end <= end_eps) {
        sh_exit = kExitTEnd;
      } else if (remaining_out <= output_eps) {
        sh_exit = kExitOutputBoundary;
      } else if (steps_done >= p.max_steps_remaining) {
        sh_exit = kExitMaxSteps;
      }
    }
    pk_sync(p);
    if (sh_exit != kExitChunkBudget) {
      break;
    }

    PK_PHASE_TRACE(kPkWatchPhaseController, "controller");
    double local_min = CUDART_INF;
    int local_idx = INT_MAX;
    int local_have = 0;
    if (p.hydro_enabled != 0) {
      for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
        double post_dt = CUDART_INF;
        int have = 0;
        const double candidate =
            tenryu::hydro::persistent_1d::cfl_1d_candidate_body(
                c, b.x_r, b.v_r, b.ee, b.ei, b.cs, nullptr, b.hydro_active,
                p.n_cells, sh_t, p.gamma, p.av_linear, p.av_quadratic, 0, 0,
                0.0, 0.0, p.av_limiter_J, &post_dt, &have);
        if (candidate < local_min ||
            (candidate == local_min && c < local_idx)) {
          local_min = candidate;
          local_idx = c;
        }
        (void)post_dt;
        local_have |= have;
      }
    }
    double min_local = CUDART_INF;
    int min_idx = INT_MAX;
    pk_reduce_argmin(p, b, local_min, local_idx, smem_arg, smem_i, &min_local,
                     &min_idx);
    const double have_sum =
        pk_reduce_sum(p, b, static_cast<double>(local_have), smem);
    (void)min_idx;

    persistent_conduction_operator(p, b);
    const double dt_rad = persistent_fld_dt_rad(p, b, smem_arg, smem_i);

    if (threadIdx.x == 0) {
      const double dt_hydro =
          (p.hydro_enabled != 0)
              ? dt_hydro_post_1d(min_local, CUDART_INF, p.cfl_hydro,
                                 have_sum > 0.0 ? 1 : 0)
              : CUDART_INF;
      const double cond_min_ratio =
          (p.conduction_enabled != 0 && b.cond_diag3 != nullptr)
              ? *(volatile const double*)(b.cond_diag3 + 0)
              : CUDART_INF;
      const double remaining_out = p.t_next_output - sh_t;
      const double remaining_end = p.t_end - sh_t;
      DtLadderIn in{};
      in.dt_hydro = dt_hydro;
      in.dt_cond = (p.conduction_enabled != 0)
                       ? dt_cond_post(cond_min_ratio,
                                      p.cfl_cond,
                                      p.conduction_sts_max_stages)
                       : CUDART_INF;
      in.dt_visc = CUDART_INF;
      in.dt_rad = dt_rad;
      in.dt_prev = sh_dt;
      in.growth_factor = p.growth_factor;
      in.dt_max = p.dt_max;
      in.dt_output = remaining_out;
      in.dt_remaining = remaining_end;
      sh_cand_hydro = in.dt_hydro;
      sh_cand_cond = in.dt_cond;
      sh_cand_rad = in.dt_rad;
      const DtLadderOut out = dt_ladder_eval(in);
      sh_dt = out.dt_chosen;
      sh_limiter = out.limiter;
    }
    pk_sync(p);
    if (sh_exit != kExitChunkBudget) {
      break;
    }

    PK_PHASE_TRACE(kPkWatchPhaseLaser, "laser");
    persistent_laser_step(p, b, sh_dt, sh_t, local_step,
                          &phase_prof_last_phase, &phase_prof_last_clock,
                          error_flag, smem, &sh_laser_input,
                          &sh_laser_escaped, &sh_laser_floor,
                          &sh_laser_skipped);
    if (p.hydro_enabled != 0) {
      PK_PHASE_TRACE(kPkWatchPhaseHydro1, "hydro_half_1");
      persistent_gamma_r_pre_hydro(p, b);
      persistent_hydro_half_step(p, b, 0.5 * sh_dt, sh_t, error_flag, smem);
    }
    PK_PHASE_TRACE(kPkWatchPhaseConduction, "conduction");
    persistent_conduction_step(p, b, sh_dt, error_flag);
    PK_PHASE_TRACE(kPkWatchPhaseFldPre, "fld_pre");
    persistent_fld_step<kGrid>(
        p, b, sh_dt, sh_t + 0.5 * sh_dt, local_step,
        &phase_prof_last_phase, &phase_prof_last_clock, error_flag, smem,
        fld_update_smem, &sh_fld_residual, &sh_fld_iterations,
        &sh_fld_escaped, &sh_fld_marshak_in, &sh_fld_volume_source);
    if (p.phase_prof != 0 && p.phase_prof_slab != nullptr &&
        pk_global_leader(p)) {
      p.phase_prof_slab[kPkPhaseProfFldOuterIterations] +=
          static_cast<double>(sh_fld_iterations);
    }
    PK_PHASE_TRACE(kPkWatchPhaseFldPost, "fld_post");
    if (p.hydro_enabled != 0) {
      PK_PHASE_TRACE(kPkWatchPhaseHydro2, "hydro_half_2");
      persistent_gamma_r_pre_hydro(p, b);
      persistent_hydro_half_step(p, b, 0.5 * sh_dt, sh_t + 0.5 * sh_dt,
                                 error_flag, smem);
    }

    PK_PHASE_TRACE(kPkWatchPhaseRing, "ring");
    double local_checksum_u = 0.0;
    for (int j = pk_thread_id(p); j < p.n_nodes; j += pk_thread_stride(p)) {
      local_checksum_u += b.v_r[j];
    }
    const double checksum_u = pk_reduce_sum(p, b, local_checksum_u, smem);
    double local_checksum_e = 0.0;
    for (int c = pk_thread_id(p); c < p.n_cells; c += pk_thread_stride(p)) {
      local_checksum_e += b.ee[c];
    }
    const double checksum_e = pk_reduce_sum(p, b, local_checksum_e, smem);

    if (threadIdx.x == 0) {
      if (pk_global_leader(p)) {
        ring[local_step] = PersistentDiagRecord{sh_t,
                                                sh_dt,
                                                sh_cand_hydro,
                                                sh_cand_cond,
                                                sh_cand_rad,
                                                checksum_u,
                                                checksum_e,
                                                sh_fld_residual,
                                                sh_fld_escaped,
                                                sh_fld_marshak_in,
                                                sh_fld_volume_source,
                                                sh_laser_input,
                                                sh_laser_escaped,
                                                sh_laser_floor,
                                                sh_laser_skipped,
                                                steps_done,
                                                sh_limiter,
                                                sh_fld_iterations,
                                                p.use_cooperative,
                                                p.cooperative_grid_blocks,
                                                0};
      }
      sh_t += sh_dt;
      ++steps_done;
      if (p.phase_prof != 0 && p.phase_prof_slab != nullptr &&
          pk_global_leader(p)) {
        p.phase_prof_slab[kPkPhaseProfStepCount] += 1.0;
      }
      if (*(volatile const int*)error_flag != 0) {
        sh_exit = kExitRetryOrError;
      }
    }
    pk_sync(p);
    if (sh_exit == kExitRetryOrError) {
      break;
    }
  }

  if (pk_global_leader(p)) {
    pk_phase_prof_mark(p, kPkWatchPhaseExit, &phase_prof_last_phase,
                       &phase_prof_last_clock);
    pk_watch_mark(p, kPkWatchPhaseExit, steps_done, -1, 0);
    d_t_dt[0] = sh_t;
    d_t_dt[1] = sh_dt;
    d_exit[0] = sh_exit;
    d_exit[1] = steps_done;
    d_exit[2] = *(volatile const int*)error_flag;
  }
#undef PK_PHASE_TRACE
}

struct PersistentLaunchConfig {
  int use_cooperative = 0;
  int blocks = 1;
  int active_blocks_per_sm = 1;
  int sm_count = 1;
  int reduce_padded_count = 1;
};

int next_power_of_two_int(const int n) {
  int padded = 1;
  while (padded < n) {
    padded <<= 1;
  }
  return padded;
}

PersistentLaunchConfig choose_persistent_launch_config(
    const int n_cells,
    const std::size_t dynamic_shared_bytes) {
  PersistentLaunchConfig config{};
  int device = 0;
  cuda_check(cudaGetDevice(&device),
             "persistent_loop: cudaGetDevice failed");

  int cooperative_supported = 0;
  cuda_check(cudaDeviceGetAttribute(&cooperative_supported,
                                    cudaDevAttrCooperativeLaunch,
                                    device),
             "persistent_loop: query cooperative launch support failed");
  if (cooperative_supported == 0 || n_cells <= 64) {
    config.reduce_padded_count = next_power_of_two_int(config.blocks);
    return config;
  }

  int sm_count = 0;
  cuda_check(cudaDeviceGetAttribute(&sm_count,
                                    cudaDevAttrMultiProcessorCount,
                                    device),
             "persistent_loop: query SM count failed");

  int active_blocks_per_sm = 0;
  const cudaError_t occupancy_err =
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks_per_sm,
          persistent_chunk_kernel<true>,
          kBlockSize,
          dynamic_shared_bytes);
  if (occupancy_err != cudaSuccess || active_blocks_per_sm <= 0 ||
      sm_count <= 0) {
    config.reduce_padded_count = next_power_of_two_int(config.blocks);
    return config;
  }

  config.use_cooperative = 1;
  config.active_blocks_per_sm = active_blocks_per_sm;
  config.sm_count = sm_count;
  config.blocks = active_blocks_per_sm * sm_count;
  const char* grid_cap_env = std::getenv("TENRYU_PK_GRID_CAP");
  if (grid_cap_env != nullptr) {
    const int grid_cap = std::atoi(grid_cap_env);
    if (grid_cap > 0) {
      config.blocks = std::min(config.blocks, grid_cap);
    }
  }
  config.reduce_padded_count = next_power_of_two_int(config.blocks);
  return config;
}

PersistentParams make_params(const core::State& state,
                             const core::Config& cfg,
                             const laser::LaserMesh& laser_mesh,
                             const radiation::PlanckTableDeviceView planck,
                             const materials::IonmixOpacityDeviceView nlte_opacity,
                             const core::namelist::FrozenTable1DDeviceView
                                 laser_waveform,
                             const double t_end,
                             const double t_next_output,
                             const int max_steps_remaining,
                             const int use_cooperative,
                             const int cooperative_grid_blocks) {
  PersistentParams p{};
  const auto& mat = cfg.materials.materials.front();
  p.n_cells = static_cast<int>(state.rho.size());
  p.n_nodes = static_cast<int>(state.x_r.size());
  p.n_materials = static_cast<int>(cfg.materials.materials.size());
  p.first_void_cell = first_trailing_void_cell(state);
  p.geom_code = state.mesh.geometry_code;
  p.boundary_type = static_cast<int>(hydro::parse_boundary_type_1d(cfg));
  p.apply_outer_boundary =
      (p.boundary_type == static_cast<int>(hydro::HydroBoundaryType::FIXED) ||
       p.boundary_type == static_cast<int>(hydro::HydroBoundaryType::REFLECT))
          ? 1
          : 0;
  p.chunk_steps = cfg.numerics.persistent_loop.chunk_steps;
  p.max_steps_remaining = max_steps_remaining;
  p.use_cooperative = use_cooperative;
  p.cooperative_grid_blocks = cooperative_grid_blocks;
  const char* phase_trace_env = std::getenv("TENRYU_PK_PHASE_TRACE");
  p.phase_trace =
      (phase_trace_env != nullptr && std::strcmp(phase_trace_env, "1") == 0)
          ? 1
          : 0;
  const char* phase_prof_env = std::getenv("TENRYU_PK_PHASE_PROF");
  p.phase_prof =
      (phase_prof_env != nullptr && std::strcmp(phase_prof_env, "1") == 0)
          ? 1
          : 0;
  const char* laser_trace_env = std::getenv("TENRYU_PK_LASER_TRACE");
  p.laser_trace =
      (laser_trace_env != nullptr && std::strcmp(laser_trace_env, "1") == 0)
          ? 1
          : 0;
  p.renorm_active =
      (!cfg.main.two_temperature &&
       cfg.numerics.hydro.enabled &&
       cfg.numerics.hydro.boundary_1d != "pressure" &&
       !cfg.numerics.hydro.compatible_energy)
          ? 1
          : 0;
  p.hydro_enabled = cfg.numerics.hydro.enabled ? 1 : 0;
  p.compatible_energy = cfg.numerics.hydro.compatible_energy ? 1 : 0;
  p.energy_authoritative =
      (cfg.numerics.hydro.eos_closure_mode == "energy_authoritative") ? 1 : 0;
  p.two_temperature = cfg.main.two_temperature ? 1 : 0;
  p.q_heat_to_electron =
      (cfg.numerics.hydro.av_heat_to == "electron") ? 1 : 0;
  p.conduction_enabled = cfg.numerics.conduction.enabled ? 1 : 0;
  p.conduction_geom_code = cfg.numerics.conduction.test_planar
                               ? 2
                               : state.mesh.geometry_code;
  const char* face_policy_env =
      std::getenv("TENRYU_CONDUCTION_FACE_KAPPA_POLICY");
  const std::string face_policy =
      (face_policy_env != nullptr) ? std::string(face_policy_env)
                                   : cfg.numerics.conduction.face_kappa_policy;
  p.conduction_face_kappa_policy =
      (face_policy == "kirchhoff_same_material") ? 1 : 0;
  const char* kappa_power_env =
      std::getenv("TENRYU_CONDUCTION_TEST_KAPPA_POWER");
  p.conduction_kappa_power_active =
      (kappa_power_env != nullptr && cfg.numerics.conduction.test_kappa > 0.0)
          ? 1
          : 0;
  p.conduction_kappa_power =
      (p.conduction_kappa_power_active != 0) ? std::atof(kappa_power_env) : 0.0;
  const char* kappa_rho_power_env =
      std::getenv("TENRYU_CONDUCTION_TEST_KAPPA_RHO_POWER");
  p.conduction_kappa_rho_power =
      (p.conduction_kappa_power_active != 0 && kappa_rho_power_env != nullptr)
          ? std::atof(kappa_rho_power_env)
          : 0.0;
  p.conduction_sts_max_stages = cfg.numerics.conduction.sts_max_stages;
  p.radiation_enabled = cfg.radiation.enabled ? 1 : 0;
  p.rad_gamma43_enabled =
      (cfg.radiation.enabled && cfg.numerics.hydro.enabled &&
       cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion &&
       state.mesh.dim == 1 &&
       (cfg.radiation.multigroup_diffusion.hydro_coupling == "gamma_r_43" ||
        rad_gamma::gamma_r_43_enabled_from_env()))
          ? 1
          : 0;
  p.n_groups = std::max(cfg.radiation.groups, 1);
  p.fld_outer_bc =
      persistent_fld_outer_bc_id(cfg.radiation.multigroup_diffusion.boundary.outer_r);
  p.fld_flux_limiter =
      persistent_fld_limiter_id(cfg.radiation.multigroup_diffusion.flux_limiter);
  p.fld_use_nlte_table =
      (mat.opacity_model == "table_nlte" || mat.opacity_model == "tmat") ? 1 : 0;
  p.fld_use_fleck =
      (mat.opacity_model == "constant" || p.fld_use_nlte_table != 0) ? 1 : 0;
  p.fld_use_fleck_blend =
      (p.fld_use_fleck != 0 &&
       cfg.radiation.multigroup_diffusion.fleck_mode != "afi")
          ? 1
          : 0;
  p.fld_fleck_form_exp =
      (cfg.radiation.multigroup_diffusion.fleck_form == "exp_phi1") ? 1 : 0;
  p.fld_has_cv_e =
      (state.cv_e.size() == state.rho.size()) ? 1 : 0;
  p.fld_max_outer_iterations =
      std::max(cfg.radiation.multigroup_diffusion.max_outer_iterations, 1);
  p.laser_enabled = cfg.laser.enabled ? 1 : 0;
  p.laser_n_folded =
      cfg.laser.enabled ? static_cast<int>(cfg.laser.beams.size()) : 0;
  p.laser_mode_raytrace_1d =
      (cfg.laser.enabled && state.mesh.dim == 1 &&
       cfg.laser.mode == "raytrace_2d")
          ? 1
          : 0;
  p.laser_rays_per_beam = cfg.laser.rays_per_beam;
  p.laser_lmesh_nr_capacity = laser_mesh.nr_capacity;
  p.laser_lmesh_nz_capacity = laser_mesh.nz_capacity;
  p.laser_lmesh_n_nodes_r_capacity = laser_mesh.n_nodes_r_capacity;
  p.laser_lmesh_n_nodes_z_capacity = laser_mesh.n_nodes_z_capacity;
  p.laser_lmesh_nodes_capacity =
      laser_mesh.n_nodes_r_capacity * laser_mesh.n_nodes_z_capacity;
  p.laser_lmesh_nr_max = cfg.laser.lasermesh.nr_max;
  p.laser_lmesh_critical_clip =
      cfg.laser.lasermesh.critical_clip ? 1 : 0;
  p.laser_lmesh_ghost_corona_enabled =
      laser_mesh.ghost_corona_enabled ? 1 : 0;
  p.laser_lmesh_ghost_n_out = laser_mesh.ghost_n_out;
  p.laser_lmesh_ghost_handoff_cells = laser_mesh.ghost_handoff_cells;
  p.laser_raytrace_max_steps =
      std::min(std::max(cfg.laser.raytrace.max_steps, 1),
               laser::ray_trace_bodies::kMaxRayStepsGuard);
  if (p.laser_mode_raytrace_1d != 0) {
    const double z_center = 0.5 * (laser_mesh.Z_min + laser_mesh.Z_max);
    const laser::Beams beams =
        laser::create_from_config(cfg.laser, state, laser_mesh.target_radius,
                                  z_center);
    if (!beams.items.empty()) {
      const laser::Beam& beam = beams.items.front();
      p.laser_beam_f_number = beam.f_number;
      p.laser_beam_focus_lab_z = beam.focus_lab_z;
      p.laser_beam_profile_w0_cm = beam.profile_w0_cm;
      p.laser_profile_m = beam.profile_m;
      if (beam.profile_model == "super_gaussian") {
        p.laser_profile_kind = 1;
      } else if (beam.profile_model == "flat_top") {
        p.laser_profile_kind = 2;
      } else {
        p.laser_profile_kind = 0;
      }
    }
  }
  p.planck = planck;
  p.nlte_opacity = nlte_opacity;
  p.laser_power_table = laser_waveform;
  p.gamma = mat.ideal_gas_gamma;
  p.fallback_z = mat.Z;
  p.material_A = std::max(mat.A, 1.0e-12);
  p.material_gm1 = std::max(mat.ideal_gas_gamma - 1.0, 1.0e-12);
  p.cv_e_override = mat.cv_e_override;
  p.eos_T_ref_eV = mat.eos_T_ref_eV;
  p.Te_floor = cfg.numerics.floors.Te;
  p.Ti_floor = cfg.numerics.floors.Ti;
  p.rho_floor = cfg.numerics.floors.rho;
  p.void_rho = cfg.materials.void_config.rho;
  p.av_linear = cfg.numerics.hydro.av_linear;
  p.av_quadratic = cfg.numerics.hydro.av_quadratic;
  p.av_limiter_J = cfg.numerics.hydro.av_limiter_J;
  p.cfl_hydro = cfg.numerics.dt.cfl_hydro;
  p.cfl_cond = cfg.numerics.dt.cfl_cond;
  p.conduction_f_lim = cfg.numerics.conduction.f_lim;
  p.conduction_mfp_limiter_C = cfg.numerics.conduction.mfp_limiter_C;
  p.conduction_sts_damping = cfg.numerics.conduction.sts_damping;
  p.conduction_sts_subcycle_eta = cfg.numerics.conduction.sts_subcycle_eta;
  p.conduction_test_kappa = cfg.numerics.conduction.test_kappa;
  p.fld_kappa_a = std::max(0.0, mat.kappa_a_constant);
  p.fld_opacity_floor = cfg.radiation.multigroup_diffusion.opacity_floor;
  p.fld_opacity_cap = cfg.radiation.multigroup_diffusion.opacity_cap;
  p.fld_outer_tol = cfg.radiation.multigroup_diffusion.outer_tol;
  p.fld_marshak_Tr_eV = cfg.radiation.boundary.marshak_Tr_eV;
  p.fld_marshak_flux =
      cfg.radiation.multigroup_diffusion.marshak.flux_erg_per_cm2_s;
  p.fld_marshak_pulse_duration =
      cfg.radiation.multigroup_diffusion.marshak.flux_pulse_duration_s;
  p.fld_volume_source_rate = cfg.radiation.volume_source_rate;
  p.fld_volume_source_r_max = cfg.radiation.volume_source_x_max;
  p.laser_lambda_cm = cfg.laser.wavelength_nm * 1.0e-7;
  p.laser_n_crit = laser_mesh.n_crit;
  p.laser_eps_n = cfg.laser.absorption.eps_n;
  p.laser_eps_crit = cfg.laser.raytrace.eps_crit;
  p.laser_test_kappa = cfg.laser.raytrace.test_kappa;
  p.laser_intensity_cutoff = cfg.laser.raytrace.intensity_cutoff;
  p.laser_coulomb_log_floor = cfg.laser.absorption.coulomb_log_floor;
  p.laser_raytrace_cfl_ray = cfg.laser.raytrace.cfl_ray;
  p.laser_raytrace_ds_adapt_g_target =
      cfg.laser.raytrace.ds_adapt_g_target;
  p.laser_raytrace_ds_adapt_tau_target =
      cfg.laser.raytrace.ds_adapt_tau_target;
  p.laser_raytrace_ds_adapt_theta_target =
      cfg.laser.raytrace.ds_adapt_theta_target;
  p.laser_raytrace_ds_adapt_max_factor =
      cfg.laser.raytrace.ds_adapt_max_factor;
  p.laser_lmesh_mesh_factor = cfg.laser.lasermesh.mesh_factor;
  p.laser_lmesh_rmax_n_hat_threshold =
      cfg.laser.lasermesh.rmax_n_hat_threshold;
  p.laser_lmesh_r_max_factor = cfg.laser.lasermesh.r_max_factor;
  p.laser_lmesh_target_radius = laser_mesh.target_radius;
  p.laser_lmesh_n_hat_margin = laser_mesh.n_hat_margin;
  p.laser_lmesh_ghost_ne_min_frac = laser_mesh.ghost_ne_min_frac;
  p.laser_lmesh_ghost_ne_max_frac = laser_mesh.ghost_ne_max_frac;
  p.laser_lmesh_ghost_Te_min_eV = laser_mesh.ghost_Te_min_eV;
  p.laser_lmesh_ghost_zbar_min = laser_mesh.ghost_zbar_min;
  p.laser_lmesh_ghost_zbar_max = laser_mesh.ghost_zbar_max;
  p.laser_lmesh_ghost_handoff_decay = laser_mesh.ghost_handoff_decay;
  p.fld_alpha = cfg.radiation.imc.alpha;
  p.fld_f_min = cfg.numerics.dt.f_min_fleck;
  p.growth_factor = cfg.numerics.dt.growth_factor;
  p.dt_max = cfg.numerics.dt.max_s;
  p.r_min = cfg.mesh.r_min;
  p.r_max = cfg.mesh.r_max;
  p.t_end = t_end;
  p.t_next_output = t_next_output;
  return p;
}

PersistentDeviceBuffers make_buffers(core::State& state,
                                     const core::Config& cfg,
                                     laser::LaserMesh& laser_mesh,
                                     const int grid_reduce_padded_count,
                                     const bool allocate_fld_pcr_work) {
  const bool trace = pk_chunk_trace_enabled();
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_nodes = state.x_r.size();
  PersistentDeviceBuffers b{};
  b.rho = state.rho.data();
  b.mass = state.mass.data();
  b.vol = state.vol.data();
  b.volFrac = state.volFrac.empty() ? nullptr : state.volFrac.data();
  b.zbar = state.zbar.empty() ? nullptr : state.zbar.data();
  b.A_eff = state.A_eff.empty() ? nullptr : state.A_eff.data();
  b.gamma_eff = state.gamma_eff.empty() ? nullptr : state.gamma_eff.data();
  b.Te = state.Te.data();
  b.Ti = state.Ti.data();
  b.ee = state.ee.data();
  b.ei = state.ei.data();
  b.Pe = state.Pe.data();
  b.Pi = state.Pi.data();
  b.Qvisc = state.Qvisc.data();
  b.cv_e = state.cv_e.empty() ? nullptr : state.cv_e.data();
  b.cv_i = state.cv_i.empty() ? nullptr : state.cv_i.data();
  b.cs = state.cs.data();
  b.x_r = state.x_r.data();
  b.v_r = state.v_r.data();
  b.hydro_active = const_cast<std::int8_t*>(state.hydro_active_device_ptr());
  const std::size_t reduce_count =
      static_cast<std::size_t>(std::max(grid_reduce_padded_count, 1));
  b.grid_reduce_partials = acquire_device_buffer<double>(
      "persistent_loop:grid_reduce_partials", reduce_count);
  b.grid_reduce_indices = acquire_device_buffer<int>(
      "persistent_loop:grid_reduce_indices", reduce_count);
  b.grid_reduce_scalar = acquire_device_buffer<double>(
      "persistent_loop:grid_reduce_scalar", 1);
  b.grid_reduce_index_scalar = acquire_device_buffer<int>(
      "persistent_loop:grid_reduce_index_scalar", 1);
  b.grid_broadcast_doubles = acquire_device_buffer<double>(
      "persistent_loop:grid_broadcast_doubles", 12);
  b.grid_broadcast_ints = acquire_device_buffer<int>(
      "persistent_loop:grid_broadcast_ints", 8);
  b.grid_error = acquire_device_buffer<int>("persistent_loop:grid_error", 1);
  if (state.cell_is_void.size() == n_cells) {
    if (trace) {
      core::log_info("pk_host: setup cell_is_void upload begin");
    }
    auto* d_cell_is_void = acquire_device_buffer<std::uint8_t>(
        "persistent_loop:cell_is_void", n_cells);
    b.cell_is_void = d_cell_is_void;
    cuda_check(cudaMemcpy(d_cell_is_void,
                          state.cell_is_void.data(),
                          n_cells * sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice),
               "persistent_loop: cell_is_void copy failed");
    if (trace) {
      core::log_info("pk_host: setup cell_is_void upload end");
    }
  }

  if (trace) {
    core::log_info("pk_host: setup material params upload begin");
  }
  const std::vector<PersistentMaterialParams> h_material_params =
      make_persistent_material_params(cfg);
  if (!h_material_params.empty()) {
    auto* d_material_params = acquire_device_buffer<PersistentMaterialParams>(
        "persistent_loop:material_params", h_material_params.size());
    b.material_params = d_material_params;
    cuda_check(cudaMemcpy(d_material_params,
                          h_material_params.data(),
                          h_material_params.size() *
                              sizeof(PersistentMaterialParams),
                          cudaMemcpyHostToDevice),
               "persistent_loop: material params copy failed");
  }
  if (trace) {
    core::log_info("pk_host: setup material params upload end");
  }

  b.cell_centroid_r =
      acquire_device_buffer<double>("persistent_loop:cell_centroid_r", n_cells);
  b.cell_centroid_z =
      acquire_device_buffer<double>("persistent_loop:cell_centroid_z", n_cells);
  b.sigma = acquire_device_buffer<double>("persistent_loop:av_sigma", n_nodes);
  b.r_old = acquire_device_buffer<double>("hydro1d_r_old", n_nodes);
  b.u_old = acquire_device_buffer<double>("hydro1d_u_old", n_nodes);
  b.V_old = acquire_device_buffer<double>("hydro1d_V_old", n_cells);
  b.e_old = acquire_device_buffer<double>("hydro1d_e_old", n_cells);
  if (cfg.main.two_temperature) {
    b.ei_old = acquire_device_buffer<double>("hydro1d_ei_old", n_cells);
  }
  b.Pe_old = acquire_device_buffer<double>("hydro1d_Pe_old", n_cells);
  b.Pi_old = acquire_device_buffer<double>("hydro1d_Pi_old", n_cells);
  b.Q_old = acquire_device_buffer<double>("hydro1d_Q_old", n_cells);
  b.node_active = acquire_device_buffer<std::uint8_t>(
      "hydro_1d:lagrangian_step:d_node_active", n_nodes);
  b.pq_n = acquire_device_buffer<double>("hydro1d_pq_n", n_cells);
  b.a_n = acquire_device_buffer<double>("hydro1d_a_n", n_nodes);
  b.u_half = acquire_device_buffer<double>("hydro1d_u_half", n_nodes);
  b.cs_half = acquire_device_buffer<double>("hydro1d_cs_half", n_cells);
  b.P_half = acquire_device_buffer<double>("hydro1d_P_half", n_cells);
  b.Pi_half = acquire_device_buffer<double>("hydro1d_Pi_half", n_cells);
  b.Q_half = acquire_device_buffer<double>("hydro1d_Q_half", n_cells);
  if (cfg.main.two_temperature) {
    b.rho_half = acquire_device_buffer<double>("hydro1d_rho_half", n_cells);
    b.Te_half = acquire_device_buffer<double>("hydro1d_Te_half", n_cells);
    b.Ti_half = acquire_device_buffer<double>("hydro1d_Ti_half", n_cells);
  }
  b.pq_half = acquire_device_buffer<double>("hydro1d_pq_half", n_cells);
  b.a_half = acquire_device_buffer<double>("hydro1d_a_half", n_nodes);
  b.floors_pack = acquire_device_buffer<double>(
      "hydro_1d:lagrangian_step:floors_pack", 2);
  b.E_floor = b.floors_pack;
  auto* floors_bytes = reinterpret_cast<unsigned char*>(b.floors_pack);
  b.clamp_count = reinterpret_cast<int*>(floors_bytes + 8);
  b.rho_clamp_count = reinterpret_cast<int*>(floors_bytes + 12);
  b.failing_cell = acquire_device_buffer<int>(
      "hydro_1d:lagrangian_step:d_failing_cell", 1);
  cuda_check(cudaMemset(b.floors_pack, 0, 2 * sizeof(double)),
             "persistent_loop: floors_pack memset failed");

  if (cfg.numerics.conduction.enabled) {
    b.cond_kappa_eff =
        acquire_device_buffer<double>("persistent_loop:cond_kappa_eff", n_cells);
    b.cond_rho_cv_e =
        acquire_device_buffer<double>("persistent_loop:cond_rho_cv_e", n_cells);
    b.cond_te_tmp =
        acquire_device_buffer<double>("persistent_loop:cond_te_tmp", n_cells);
    b.cond_flux_limiter_faces = acquire_device_buffer<double>(
        "persistent_loop:cond_flux_limiter_faces", std::max<std::size_t>(n_cells, 1));
    b.cond_diag3 =
        acquire_device_buffer<double>("persistent_loop:cond_diag3", 3);
    if (state.A_eff.size() == n_cells && state.gamma_eff.size() == n_cells) {
      b.cond_A_eff = state.A_eff.data();
      b.cond_gamma_eff = state.gamma_eff.data();
    } else {
      b.cond_A_eff =
          acquire_device_buffer<double>("persistent_loop:cond_A_eff", n_cells);
      b.cond_gamma_eff =
          acquire_device_buffer<double>("persistent_loop:cond_gamma_eff", n_cells);
      const auto& mat = cfg.materials.materials.front();
      std::vector<double> A_eff(n_cells, std::max(mat.A, 1.0e-12));
      std::vector<double> gamma_eff(n_cells,
                                    std::max(mat.ideal_gas_gamma,
                                             1.0 + 1.0e-12));
      cuda_check(cudaMemcpy(b.cond_A_eff,
                            A_eff.data(),
                            n_cells * sizeof(double),
                            cudaMemcpyHostToDevice),
                 "persistent_loop: cond_A_eff copy failed");
      cuda_check(cudaMemcpy(b.cond_gamma_eff,
                            gamma_eff.data(),
                            n_cells * sizeof(double),
                            cudaMemcpyHostToDevice),
                 "persistent_loop: cond_gamma_eff copy failed");
    }
  }
  if (cfg.radiation.enabled) {
    const std::size_t n_groups =
        static_cast<std::size_t>(std::max(cfg.radiation.groups, 1));
    const std::size_t n_total = n_cells * n_groups;
    const bool rad_gamma43_active =
        cfg.numerics.hydro.enabled &&
        cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion &&
        state.mesh.dim == 1 &&
        (cfg.radiation.multigroup_diffusion.hydro_coupling == "gamma_r_43" ||
         rad_gamma::gamma_r_43_enabled_from_env());
    state.rad_E.reset(n_total);
    state.rad_dep.reset(n_total);
    state.rad_emit.reset(n_total);
    state.rad_E_old.reset(n_total);
    state.fld_sigma_a.reset(n_total);
    state.fld_sigma_pe.reset(n_total);
    state.fld_sigma_R.reset(n_total);
    state.fld_eta.reset(n_total);
    state.fld_lower.reset(n_total);
    state.fld_diag.reset(n_total);
    state.fld_upper.reset(n_total);
    state.fld_rhs.reset(n_total);
    state.fld_Te_old.reset(n_cells);
    state.fld_delta_T.reset(n_cells);
    state.fld_cusparse_buffer.reset(n_total);
    state.fld_nlte_f_work.reset(n_total);
    state.fld_nlte_sigma_eff_work.reset(n_total);
    state.fld_nlte_sigma_s_eff_work.reset(n_total);
    state.fld_nlte_eta_cdf_work.reset(n_total);
    state.fld_nlte_lambda_work.reset(n_total);
    b.rad_E = state.rad_E.data();
    b.rad_E_old = state.rad_E_old.data();
    if (rad_gamma43_active) {
      b.rad_gamma_p_r =
          acquire_device_buffer<double>("persistent_loop:rad_gamma43_p_r",
                                        n_cells);
      b.rad_gamma_W_r =
          acquire_device_buffer<double>("persistent_loop:rad_gamma43_W_r",
                                        n_cells);
      b.rad_gamma_vol_before = acquire_device_buffer<double>(
          "persistent_loop:rad_gamma43_vol_before", n_cells);
      b.rad_gamma_r_half =
          acquire_device_buffer<double>("hydro1d_r_half_gamma", n_nodes);
    }
    b.rad_dep = state.rad_dep.data();
    b.rad_emit = state.rad_emit.data();
    b.fld_sigma_a = state.fld_sigma_a.data();
    b.fld_sigma_pe = state.fld_sigma_pe.data();
    b.fld_sigma_R = state.fld_sigma_R.data();
    b.fld_eta = state.fld_eta.data();
    b.fld_lower = state.fld_lower.data();
    b.fld_diag = state.fld_diag.data();
    b.fld_upper = state.fld_upper.data();
    b.fld_rhs = state.fld_rhs.data();
    b.fld_Te_old = state.fld_Te_old.data();
    b.fld_delta_T = state.fld_delta_T.data();
    b.fld_cp_work = state.fld_cusparse_buffer.data();
    if (allocate_fld_pcr_work) {
      b.fld_pcr_dl_work =
          acquire_device_buffer<double>("persistent_loop:fld_pcr_dl_work",
                                        n_total);
      b.fld_pcr_d_work =
          acquire_device_buffer<double>("persistent_loop:fld_pcr_d_work",
                                        n_total);
      b.fld_pcr_du_work =
          acquire_device_buffer<double>("persistent_loop:fld_pcr_du_work",
                                        n_total);
      b.fld_pcr_rhs_work =
          acquire_device_buffer<double>("persistent_loop:fld_pcr_rhs_work",
                                        n_total);
    }
    b.fld_fleck = state.fld_nlte_f_work.data();
    b.fld_nlte_sigma_eff_work = state.fld_nlte_sigma_eff_work.data();
    b.fld_nlte_sigma_s_eff_work = state.fld_nlte_sigma_s_eff_work.data();
    b.fld_nlte_eta_cdf_work = state.fld_nlte_eta_cdf_work.data();
    b.fld_nlte_lambda_work = state.fld_nlte_lambda_work.data();
    b.fld_marshak_finc =
        acquire_device_buffer<double>("persistent_loop:fld_marshak_finc",
                                      n_groups);
  }
  if (cfg.laser.enabled) {
    b.laser_dep = state.laser_dep.empty() ? nullptr : state.laser_dep.data();
    if (cfg.laser.mode == "raytrace_2d" && state.mesh.dim == 1) {
      const std::size_t n_rays =
          static_cast<std::size_t>(std::max(cfg.laser.rays_per_beam, 1));
      laser_mesh.ensure_step_scratch();
      b.laser_node_R = laser_mesh.node_R;
      b.laser_node_Z = laser_mesh.node_Z;
      b.laser_n_e_hat = laser_mesh.n_e_hat;
      b.laser_n_e_hat_raw = laser_mesh.n_e_hat_raw;
      b.laser_T_e = laser_mesh.T_e;
      b.laser_Zbar = laser_mesh.Zbar;
      b.laser_smooth_kappa_factor = laser_mesh.smooth_kappa_factor;
      b.laser_grad_n_hat_R = laser_mesh.grad_n_hat_R;
      b.laser_grad_n_hat_Z = laser_mesh.grad_n_hat_Z;
      b.laser_radial_node_r = laser_mesh.radial_node_r;
      b.laser_radial_n_hat = laser_mesh.radial_n_hat;
      b.laser_radial_n_hat_raw = laser_mesh.radial_n_hat_raw;
      b.laser_radial_smooth_kappa = laser_mesh.radial_smooth_kappa;
      b.laser_radial_dn_dr = laser_mesh.radial_dn_dr;
      b.laser_step_tally_slab = laser_mesh.scratch_step_tally_slab;
      b.laser_unabsorbed = laser_mesh.scratch_unabsorbed;
      b.laser_tail_closure_count = laser_mesh.scratch_tail_closure_count;
      b.laser_tail_closure_absorbed_power =
          laser_mesh.scratch_tail_closure_absorbed_power;
      b.laser_critical_surface_hit_count =
          laser_mesh.scratch_critical_surface_hit_count;
      b.laser_error_flags = laser_mesh.scratch_error_flags;

      double* ray_slab = acquire_device_buffer<double>(
          "persistent_laser:ray_slab", 6ULL * n_rays);
      b.laser_ray_R0 = ray_slab + 0 * n_rays;
      b.laser_ray_Z0 = ray_slab + 1 * n_rays;
      b.laser_ray_vR0 = ray_slab + 2 * n_rays;
      b.laser_ray_vZ0 = ray_slab + 3 * n_rays;
      b.laser_ray_power = ray_slab + 4 * n_rays;
      b.laser_ray_power0 = ray_slab + 5 * n_rays;
      b.laser_ray_weights = acquire_device_buffer<double>(
          "persistent_laser:ray_weights", n_rays);

      const std::size_t per_ray_deposit =
          n_rays * std::max<std::size_t>(n_cells, 1);
      double* per_ray_slab = acquire_device_buffer<double>(
          "persistent_laser:per_ray_tallies",
          per_ray_deposit + 2ULL * n_rays);
      b.laser_per_ray_deposit = per_ray_slab;
      b.laser_per_ray_unabsorbed = per_ray_slab + per_ray_deposit;
      b.laser_per_ray_tail_power = b.laser_per_ray_unabsorbed + n_rays;
      b.laser_cell_n_hat = acquire_device_buffer<double>(
          "persistent_laser:cell_n_hat", n_cells);
      b.laser_widths = acquire_device_buffer<double>(
          "persistent_laser:graded_widths",
          static_cast<std::size_t>(std::max(cfg.laser.lasermesh.nr_max, 4)) +
              1U);
      b.laser_ema_state = acquire_device_buffer<double>(
          "persistent_laser:ema_state",
          static_cast<std::size_t>(
              std::max(laser_mesh.n_nodes_r_capacity *
                           laser_mesh.n_nodes_z_capacity,
                       1)));
      b.laser_ema_valid =
          acquire_device_buffer<int>("persistent_laser:ema_valid", 1);
      b.laser_cell_is_void = acquire_device_buffer<std::uint8_t>(
          "persistent_laser:cell_is_void", n_cells);
      std::vector<std::uint8_t> cell_is_void_host(n_cells, 0U);
      if (state.cell_is_void.size() == n_cells) {
        std::copy(state.cell_is_void.begin(), state.cell_is_void.end(),
                  cell_is_void_host.begin());
      }
      if (trace) {
        core::log_info("pk_host: setup cell_is_void upload begin");
      }
      cuda_check(cudaMemcpy(b.laser_cell_is_void,
                            cell_is_void_host.data(),
                            n_cells * sizeof(std::uint8_t),
                            cudaMemcpyHostToDevice),
                 "persistent_loop: laser cell_is_void copy failed");
      if (trace) {
        core::log_info("pk_host: setup cell_is_void upload end");
      }
    }
  }
  return b;
}

struct PersistentLaserMeshCapacityBound {
  int nr = 0;
  int nz = 0;
  int entry_nr = 0;
  int entry_nz = 0;
  int ratio_bound_nr = 0;
  int nr_config_bound = 0;
  double radius_bound = 0.0;
  double min_dr_entry = 0.0;
  double radius_ratio_bound = 0.0;
};

int ceil_to_int_saturated(const double value) {
  if (!(value > 0.0) || !std::isfinite(value)) {
    return INT_MAX;
  }
  if (value >= static_cast<double>(INT_MAX)) {
    return INT_MAX;
  }
  return static_cast<int>(std::ceil(value));
}

int double_int_saturated(const int value) {
  if (value > INT_MAX / 2) {
    return INT_MAX;
  }
  return 2 * value;
}

PersistentLaserMeshCapacityBound persistent_laser_mesh_capacity_bound_1d(
    const laser::HydroMirror1D& hydro,
    const core::Config& cfg,
    const laser::LaserMesh& laser_mesh,
    const int entry_nr,
    const int entry_nz) {
  PersistentLaserMeshCapacityBound out;
  out.entry_nr = entry_nr;
  out.entry_nz = entry_nz;
  out.nr_config_bound = std::max(cfg.laser.lasermesh.nr_max, 4);

  double r_outer_entry = 0.0;
  for (const double r : hydro.r_edges) {
    if (std::isfinite(r)) {
      r_outer_entry = std::max(r_outer_entry, r);
    }
  }

  double min_dr_entry = std::numeric_limits<double>::max();
  for (std::size_t i = 1; i < hydro.r_edges.size(); ++i) {
    const double dr = hydro.r_edges[i] - hydro.r_edges[i - 1];
    if (dr > 0.0 && std::isfinite(dr)) {
      min_dr_entry = std::min(min_dr_entry, dr);
    }
  }
  const bool has_min_dr =
      (min_dr_entry > 0.0 && std::isfinite(min_dr_entry) &&
       min_dr_entry < std::numeric_limits<double>::max());
  out.min_dr_entry = has_min_dr ? min_dr_entry : 0.0;

  const double target_radius = std::max(laser_mesh.target_radius, 1.0e-12);
  out.radius_bound = cfg.laser.lasermesh.r_max_factor *
                     std::max(target_radius, r_outer_entry);
  if (!has_min_dr || !(cfg.laser.lasermesh.mesh_factor > 0.0) ||
      !std::isfinite(cfg.laser.lasermesh.mesh_factor)) {
    out.ratio_bound_nr = out.nr_config_bound;
    out.nr = std::max(entry_nr, out.nr_config_bound);
    out.nz = std::max(entry_nz, double_int_saturated(out.nr));
    return out;
  }

  const double rmax_formula_bound =
      std::max(out.radius_bound, 4.0 * min_dr_entry);
  const double dR_fine_lower =
      std::max(cfg.laser.lasermesh.mesh_factor * min_dr_entry, 1.0e-12);
  out.radius_ratio_bound = rmax_formula_bound / dR_fine_lower;
  out.ratio_bound_nr =
      ceil_to_int_saturated(2.0 * (out.radius_ratio_bound + 4.0));
  out.nr = std::max(entry_nr,
                    std::min(out.nr_config_bound,
                             std::max(out.ratio_bound_nr, 4)));
  out.nz = std::max(entry_nz, double_int_saturated(out.nr));
  return out;
}

void log_persistent_laser_mesh_capacity_bound(
    const PersistentLaserMeshCapacityBound& bound,
    const core::Config& cfg) {
  char buffer[512];
  std::snprintf(
      buffer,
      sizeof(buffer),
      "persistent_loop: laser lmesh capacity bound entry_nr=%d entry_nz=%d "
      "bound_nr=%d bound_nz=%d ratio_bound_nr=%d nr_max=%d "
      "radius_bound=%.6e min_dr_entry=%.6e mesh_factor=%.6e "
      "ratio=%.6e formula=min(nr_max,ceil(2*(R_bound/"
      "(mesh_factor*min_dr_entry)+4)))",
      bound.entry_nr,
      bound.entry_nz,
      bound.nr,
      bound.nz,
      bound.ratio_bound_nr,
      bound.nr_config_bound,
      bound.radius_bound,
      bound.min_dr_entry,
      cfg.laser.lasermesh.mesh_factor,
      bound.radius_ratio_bound);
  core::log_info(buffer);
}

}  // namespace

void prepare_persistent_laser_entry(core::State& state,
                                    const core::Config& cfg,
                                    laser::LaserMesh& laser_mesh) {
  if (!cfg.laser.enabled || state.mesh.dim != 1 ||
      cfg.laser.mode != "raytrace_2d") {
    return;
  }
  state.ensure_cell_material_props(cfg);
  laser::HydroMirror1D hydro;
  laser::build_hydro_mirror_1d(laser_mesh, state, hydro);
  laser::map_from_hydro_1d(laser_mesh, state, cfg.laser, hydro, nullptr);
  const PersistentLaserMeshCapacityBound capacity_bound =
      persistent_laser_mesh_capacity_bound_1d(hydro, cfg, laser_mesh,
                                              laser_mesh.nr, laser_mesh.nz);
  log_persistent_laser_mesh_capacity_bound(capacity_bound, cfg);
  if (capacity_bound.nr > laser_mesh.nr_capacity ||
      capacity_bound.nz > laser_mesh.nz_capacity) {
    laser_mesh.ensure_capacity(capacity_bound.nr, capacity_bound.nz);
    laser::map_from_hydro_1d(laser_mesh, state, cfg.laser, hydro, nullptr);
  }
  laser_mesh.ensure_step_scratch();
  int* ema_valid =
      acquire_device_buffer<int>("persistent_laser:ema_valid", 1);
  cuda_check(cudaMemset(ema_valid, 0, sizeof(int)),
             "persistent_loop: persistent_laser ema_valid memset failed");
}

bool persistent_loop_supported_c1(const core::State& state,
                                  const core::Config& cfg,
                                  const laser::LaserMesh* laser_mesh) {
  const bool hydro_enabled = cfg.numerics.hydro.enabled;
  const bool radiation_enabled = cfg.radiation.enabled;
  const bool laser_enabled = cfg.laser.enabled;
  if (cfg.main.dimension != "1D_SPH") {
    return warn_unsupported_once("dimension != 1D_SPH");
  }
  if (state.mesh.dim != 1) {
    return warn_unsupported_once("state.mesh.dim != 1");
  }
  if (cfg.mesh.motion == "ale" || cfg.numerics.ale.enabled) {
    return warn_unsupported_once("ALE enabled");
  }
  if (!hydro_enabled && !radiation_enabled && !laser_enabled &&
      !cfg.numerics.conduction.enabled) {
    return warn_unsupported_once("no C2b physics enabled");
  }
  if (cfg.numerics.conduction.enabled) {
    if (!hydro_enabled) {
      return warn_unsupported_once("conduction without hydro");
    }
    if (cfg.numerics.conduction.solver != "sts") {
      return warn_unsupported_once("non-STS conduction solver");
    }
    if (cfg.numerics.conduction.ion_conduction) {
      return warn_unsupported_once("ion conduction enabled");
    }
    if (cfg.numerics.conduction.sts_max_stages <= 0) {
      return warn_unsupported_once("conduction sts_max_stages <= 0");
    }
    if (cfg.numerics.materials.per_material_conservation_enabled) {
      return warn_unsupported_once("per-material conservation enabled");
    }
  }
  if (!laser_enabled && !cfg.laser.beams.empty()) {
    return warn_unsupported_once("laser beams configured while laser disabled");
  }
  if (laser_enabled) {
    if (!hydro_enabled) {
      return warn_unsupported_once("laser without hydro");
    }
    const bool laser_radial_absorption =
        cfg.laser.mode == "radial_absorption_1d";
    const bool laser_raytrace_1d = cfg.laser.mode == "raytrace_2d";
    if (!laser_radial_absorption && !laser_raytrace_1d) {
      return warn_unsupported_once(
          "laser mode != radial_absorption_1d or raytrace_2d");
    }
    if (cfg.laser.cbet.enable) {
      return warn_unsupported_once("CBET enabled");
    }
    if (cfg.laser.beams.empty()) {
      return warn_unsupported_once("laser enabled without beams");
    }
    if (cfg.laser.absorption.model != "inverse_bremsstrahlung") {
      return warn_unsupported_once("non-IB laser absorption model");
    }
    if (cfg.laser.absorption.debug_dump_lasermesh) {
      return warn_unsupported_once("laser debug lasermesh dump enabled");
    }
    if (cfg.laser.ray_output_count > 0 ||
        cfg.laser.ray_output_trajectory) {
      return warn_unsupported_once("laser ray output configured");
    }
    if (laser_raytrace_1d) {
      if (laser_mesh == nullptr || !laser_mesh->is_allocated()) {
        return warn_unsupported_once("laser mesh unavailable for raytrace");
      }
      if (cfg.main.verbosity == "verbose") {
        return warn_unsupported_once("laser raytrace verbose fold disabled");
      }
      const char* no_fold = std::getenv("TENRYU_LASER_NO_BEAM_FOLD");
      if (no_fold != nullptr && no_fold[0] != '\0' && no_fold[0] != '0') {
        return warn_unsupported_once("laser beam folding disabled");
      }
      if (cfg.laser.rays_per_beam <= 0) {
        return warn_unsupported_once("laser rays_per_beam <= 0");
      }
      // v1.1: raytrace_skip cache/replay is intentionally out of scope for
      // persistent raytrace.
      if (cfg.laser.raytrace_skip_config.enabled ||
          cfg.laser.raytrace_skip > 0.0) {
        return warn_unsupported_once("laser raytrace_skip enabled");
      }
      const double z_center = 0.5 * (laser_mesh->Z_min + laser_mesh->Z_max);
      const laser::Beams beams =
          laser::create_from_config(cfg.laser, state, laser_mesh->target_radius,
                                    z_center);
      if (beams.items.empty()) {
        return warn_unsupported_once("laser enabled without beams");
      }
      const laser::Beam& first = beams.items.front();
      for (std::size_t i = 1; i < beams.items.size(); ++i) {
        const laser::Beam& beam = beams.items[i];
        if (beam.f_number != first.f_number ||
            beam.focus_lab_z != first.focus_lab_z ||
            beam.delta_lambda_nm != first.delta_lambda_nm ||
            beam.profile_m != first.profile_m ||
            beam.profile_w0_cm != first.profile_w0_cm ||
            beam.profile_model != first.profile_model) {
          return warn_unsupported_once("laser beams not fold-identical");
        }
      }
    }
    if (cfg.laser.lasermesh.ghost_corona.enabled &&
        cfg.laser.lasermesh.ghost_corona.transition_enabled) {
      return warn_unsupported_once("laser ghost_corona transition enabled");
    }
    if (cfg.laser.deposit.deposit_smooth_passes > 0 &&
        cfg.laser.deposit.deposit_smooth_alpha > 0.0) {
      return warn_unsupported_once("laser deposit smoothing configured");
    }
    if (!(cfg.laser.wavelength_nm > 0.0)) {
      return warn_unsupported_once("laser wavelength_nm <= 0");
    }
    if (state.laser_dep.size() != state.rho.size()) {
      return warn_unsupported_once("laser_dep size mismatch");
    }
    if (state.zbar.size() != state.rho.size()) {
      return warn_unsupported_once("laser zbar size mismatch");
    }
    if (state.laser_waveforms.size() != cfg.laser.beams.size()) {
      return warn_unsupported_once("laser waveform count mismatch");
    }
    if (state.laser_waveforms.empty()) {
      return warn_unsupported_once("laser waveform table missing");
    }
    const auto& first_waveform = state.laser_waveforms.front();
    if (first_waveform.n_points <= 0 ||
        first_waveform.n_points != static_cast<int>(first_waveform.x.size()) ||
        first_waveform.n_points != static_cast<int>(first_waveform.y.size())) {
      return warn_unsupported_once("laser waveform table invalid");
    }
    if (!first_waveform.zero_outside) {
      return warn_unsupported_once("laser waveform zero_outside=false");
    }
    for (const auto& waveform : state.laser_waveforms) {
      if (waveform.n_points <= 0 ||
          waveform.n_points != static_cast<int>(waveform.x.size()) ||
          waveform.n_points != static_cast<int>(waveform.y.size())) {
        return warn_unsupported_once("laser waveform table invalid");
      }
      if (!waveform.zero_outside) {
        return warn_unsupported_once("laser waveform zero_outside=false");
      }
      if (!frozen_tables_equal(waveform, first_waveform)) {
        return warn_unsupported_once("laser waveforms not fold-identical");
      }
    }
  }
  if (cfg.laser.cbet.enable) {
    return warn_unsupported_once("CBET enabled");
  }
  if (cfg.laser.enabled && cfg.laser.hot_electron.enable) {
    return warn_unsupported_once("hot_electron");
  }
  if (cfg.burn.enabled) {
    return warn_unsupported_once("burn");
  }
  if (cfg.radiation.multigroup_diffusion.source_integrator ==
      "exp_rosenbrock") {
    return warn_unsupported_once("source_integrator=exp_rosenbrock");
  }
  const parallel::PartitionInfo part = parallel::Partition::build(
      state.mesh.topo.nr, std::max(state.mesh.topo.nz, 1), cfg.main.dim,
      cfg.parallel.decomposition.method, cfg.parallel.decomposition.dims,
      cfg.parallel.decomposition.min_cells_per_rank);
  if (part.n_ranks != 1) {
    return warn_unsupported_once("MPI n_ranks != 1");
  }
  if (cfg.output.history_every > 0) {
    return warn_unsupported_once("step-based history cadence configured");
  }
  if (cfg.numerics.diagnostics.dt_breakdown_history_enabled) {
    return warn_unsupported_once("per-step dt-breakdown history enabled");
  }
  if (cfg.output.plot_every > 0) {
    return warn_unsupported_once("step-based plot cadence configured");
  }
  if (cfg.output.checkpoint_every > 0) {
    return warn_unsupported_once("step-based checkpoint cadence configured");
  }
  if (!(cfg.numerics.dt.initial_s > 0.0) ||
      !(cfg.numerics.dt.growth_factor > 0.0)) {
    return warn_unsupported_once("unsupported initial dt handling");
  }
  if (cfg.numerics.persistent_loop.chunk_steps < 1) {
    return warn_unsupported_once("chunk_steps < 1");
  }
  if (!persistent_materials_supported(state, cfg, radiation_enabled)) {
    return false;
  }
  const auto& mat = cfg.materials.materials.front();
  if (radiation_enabled) {
    if ((cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion &&
         cfg.radiation.multigroup_diffusion.boundary.outer_r == "marshak") ||
        (cfg.radiation.mode == core::RadiationMode::SnTransport &&
         cfg.radiation.sn_transport.boundary.outer_r == "marshak")) {
      return warn_unsupported_once("marshak_outer_boundary");
    }
    if (cfg.radiation.mode != core::RadiationMode::MultigroupDiffusion) {
      return warn_unsupported_once("non-FLD radiation mode");
    }
    if (cfg.radiation.groups < 1) {
      return warn_unsupported_once("radiation groups < 1");
    }
    if (cfg.numerics.radiation_thermal_subcycle) {
      return warn_unsupported_once("radiation thermal subcycle enabled");
    }
    if (cfg.radiation.imc.two_stage) {
      return warn_unsupported_once("two-stage radiation enabled");
    }
    if (cfg.radiation.imc.difference.enabled) {
      return warn_unsupported_once("radiation difference formulation enabled");
    }
    if (cfg.radiation.imc.linearized_planck) {
      return warn_unsupported_once("linearized Planck dt_rad enabled");
    }
    if (cfg.radiation.holo.enabled) {
      return warn_unsupported_once("HOLO radiation enabled");
    }
    const bool use_nlte_table =
        mat.opacity_model == "table_nlte" || mat.opacity_model == "tmat";
    if (mat.opacity_model != "constant" && !use_nlte_table) {
      return warn_unsupported_once("non-constant opacity path: C2c");
    }
    if (!use_nlte_table && !(mat.kappa_a_constant > 0.0)) {
      return warn_unsupported_once("constant opacity kappa_a_constant <= 0");
    }
    if (cfg.radiation.multigroup_diffusion.linear_solver_1d !=
        "cusparse_tridiag") {
      return warn_unsupported_once("non-tridiag 1D FLD solver");
    }
    if (cfg.radiation.boundary.marshak_Tr.detected ||
        !cfg.radiation.boundary.marshak_Tr_map.empty()) {
      return warn_unsupported_once("Marshak callable boundary: C2c");
    }
    if (cfg.radiation.multigroup_diffusion.boundary.inner_r != "reflect") {
      return warn_unsupported_once("unsupported FLD inner_r boundary");
    }
    const std::string& outer_r =
        cfg.radiation.multigroup_diffusion.boundary.outer_r;
    if (outer_r != "vacuum" && outer_r != "reflect" &&
        outer_r != "reflective" && outer_r != "marshak") {
      return warn_unsupported_once("unsupported FLD outer_r boundary");
    }
  }
  if (hydro_enabled) {
    if (cfg.numerics.hydro.volume_rate_cfl_enabled) {
      return warn_unsupported_once("volume-rate CFL enabled");
    }
    if (hydro::central_pseudo_core::configured(cfg)) {
      return warn_unsupported_once("central pseudo-core configured");
    }
    if (cfg.numerics.hydro.hk_velocity_damper_C > 0.0) {
      return warn_unsupported_once("hk velocity damper enabled");
    }
    if (cfg.numerics.hydro.ee_odd_even_C > 0.0) {
      return warn_unsupported_once("electron odd-even flux enabled");
    }
    if (cfg.numerics.hydro.plasma_viscosity.enabled || brag_env_enabled()) {
      return warn_unsupported_once("Braginskii viscosity enabled");
    }
    if (cfg.numerics.hydro.compatible_energy) {
      return warn_unsupported_once("compatible_energy enabled");
    }
    if (cfg.numerics.hydro.exact_override != "none") {
      return warn_unsupported_once("exact ideal-gas override enabled");
    }
    if (cfg.numerics.hydro.boundary_1d == "pressure") {
      return warn_unsupported_once("pressure boundary configured");
    }
    if (cfg.numerics.hydro.boundary_1d != "free" &&
        cfg.numerics.hydro.boundary_1d != "fixed" &&
        cfg.numerics.hydro.boundary_1d != "reflect") {
      return warn_unsupported_once("unsupported 1D boundary");
    }
    if (cfg.numerics.hydro.av_type != "vnr") {
      return warn_unsupported_once("non-VNR artificial viscosity");
    }
    if (cfg.numerics.hydro.av_model != core::AvModel::ScalarVnrLegacy) {
      return warn_unsupported_once("non-scalar VNR AV model");
    }
    if (cfg.numerics.hydro.adaptive_av.enabled) {
      return warn_unsupported_once("adaptive AV enabled");
    }
    if (cfg.numerics.hydro.av_eos_aware) {
      return warn_unsupported_once("EOS-aware AV enabled");
    }
    if (cfg.numerics.hydro.post_shock_heat ||
        cfg.numerics.hydro.post_shock_velocity_damping_C > 0.0) {
      return warn_unsupported_once("post-shock hydro block enabled");
    }
    if (cfg.numerics.hydro.odd_even_damping_C > 0.0) {
      return warn_unsupported_once("odd-even velocity damping enabled");
    }
    if (cfg.numerics.hydro.av_heat_C > 0.0) {
      return warn_unsupported_once("AV heat enabled");
    }
    if (cfg.numerics.hydro.ion_art_heat_C > 0.0) {
      return warn_unsupported_once("ion artificial heat enabled");
    }
    if (cfg.numerics.hydro.bulk_viscosity_C > 0.0) {
      return warn_unsupported_once("bulk viscosity enabled");
    }
    if (cfg.numerics.debug.trace_mesh_motion) {
      return warn_unsupported_once("mesh motion trace enabled");
    }
    // v1.1 retry sketch: chunk-entry device snapshot via pool slabs, on exit-3 restore + re-run k steps + forced dt/2 at the failing step.
  }
  if (state.mesh.button_center && state.mesh.button_center->enabled) {
    return warn_unsupported_once("button-center mesh configured");
  }
  if (hydro_enabled && state.hydro_t_start_eV > 0.0) {
    return warn_unsupported_once("hydro T_start_eV gating enabled");
  }
  const bool has_cell_void_mask =
      state.cell_is_void.size() == state.rho.size();
  if (!has_cell_void_mask && !state.cell_is_void.empty()) {
    return warn_unsupported_once("cell_is_void size mismatch");
  }
  if (hydro_enabled && !state.hydro_active.empty()) {
    if (state.hydro_active.size() != state.rho.size()) {
      return warn_unsupported_once("hydro_active size mismatch");
    }
    for (std::size_t i = 0; i < state.hydro_active.size(); ++i) {
      const bool void_cell =
          has_cell_void_mask && state.cell_is_void[i] != 0U;
      if (state.hydro_active[i] == 0 && !void_cell) {
        return warn_unsupported_once("inactive hydro cells present");
      }
    }
  }
  if (state.rho.empty() || state.x_r.empty()) {
    return warn_unsupported_once("empty 1D hydro mesh");
  }
  if (state.x_r.size() != state.rho.size() + 1U ||
      state.v_r.size() != state.x_r.size()) {
    return warn_unsupported_once("1D node/cell size mismatch");
  }
  if (state.vol.size() != state.rho.size() ||
      state.ee.size() != state.rho.size() ||
      state.Pe.size() != state.rho.size() ||
      (!state.A_eff.empty() && state.A_eff.size() != state.rho.size()) ||
      (!state.gamma_eff.empty() && state.gamma_eff.size() != state.rho.size())) {
    return warn_unsupported_once("required cell field size mismatch");
  }
  if (hydro_enabled &&
      (state.mass.size() != state.rho.size() ||
       state.ei.size() != state.rho.size() ||
       state.Pi.size() != state.rho.size() ||
       state.Qvisc.size() != state.rho.size())) {
    return warn_unsupported_once("required hydro field size mismatch");
  }
  if (radiation_enabled) {
    const std::size_t n_groups =
        static_cast<std::size_t>(std::max(cfg.radiation.groups, 1));
    const std::size_t n_total = state.rho.size() * n_groups;
    if (state.rad_E.size() != n_total) {
      return warn_unsupported_once("radiation rad_E size mismatch");
    }
    if (state.zbar.size() != state.rho.size()) {
      return warn_unsupported_once("radiation zbar size mismatch");
    }
    if (!state.cv_e.empty() && state.cv_e.size() != state.rho.size()) {
      return warn_unsupported_once("radiation cv_e size mismatch");
    }
  }
  if (cfg.numerics.conduction.enabled) {
    if (state.zbar.size() != state.rho.size()) {
      return warn_unsupported_once("conduction zbar size mismatch");
    }
    if (!state.cv_e.empty() && state.cv_e.size() != state.rho.size()) {
      return warn_unsupported_once("conduction cv_e size mismatch");
    }
  }
  if (laser_enabled && !state.cv_e.empty() &&
      state.cv_e.size() != state.rho.size()) {
    return warn_unsupported_once("laser cv_e size mismatch");
  }
  return true;
}

PersistentChunkResult run_persistent_chunk(core::State& state,
                                           const core::Config& cfg,
                                           laser::LaserMesh& laser_mesh,
                                           radiation::PlanckTableDeviceView planck,
                                           materials::IonmixOpacityDeviceView
                                               nlte_opacity,
                                           core::namelist::FrozenTable1DDeviceView
                                               laser_waveform,
                                           const double t_end,
                                           const double t_next_output,
                                           const int max_steps_remaining) {
  const bool trace = pk_chunk_trace_enabled();
  TENRYU_ASSERT(persistent_loop_supported_c1(state, cfg, &laser_mesh),
                "persistent_loop: unsupported C1/C2b/C3 config reached runner");
  if (cfg.numerics.hydro.enabled) {
    TENRYU_ASSERT(state.cs.size() == state.rho.size(),
                  "persistent_loop: state.cs must be initialized before launch");
  }
  if (cfg.radiation.enabled) {
    TENRYU_ASSERT(planck.n_groups == std::max(cfg.radiation.groups, 1),
                  "persistent_loop: Planck table group count mismatch");
    TENRYU_ASSERT(planck.n_groups <= 1 || planck.b_g != nullptr,
                  "persistent_loop: Planck table device view is empty");
    const auto& mat = cfg.materials.materials.front();
    const bool use_nlte_table =
        mat.opacity_model == "table_nlte" || mat.opacity_model == "tmat";
    if (use_nlte_table) {
      TENRYU_ASSERT(nlte_opacity.ngroups == std::max(cfg.radiation.groups, 1),
                    "persistent_loop: NLTE opacity table group count mismatch");
      TENRYU_ASSERT(nlte_opacity.kappa_PA != nullptr &&
                        nlte_opacity.kappa_PE != nullptr &&
                        nlte_opacity.kappa_R != nullptr,
                    "persistent_loop: NLTE opacity table device view is empty");
    }
  }
  if (cfg.laser.enabled) {
    TENRYU_ASSERT(laser_waveform.n > 0 && laser_waveform.x != nullptr &&
                      laser_waveform.y != nullptr,
                  "persistent_loop: laser waveform device view is empty");
  }
  state.ensure_cell_material_props(cfg);

  const int chunk_steps = cfg.numerics.persistent_loop.chunk_steps;
  std::size_t dynamic_shared_bytes = 0;
  if (cfg.radiation.enabled) {
    const std::size_t n_groups =
        static_cast<std::size_t>(std::max(cfg.radiation.groups, 1));
    dynamic_shared_bytes =
        2U * static_cast<std::size_t>(kUpdateMatterWarps) * n_groups *
        sizeof(double);
    TENRYU_ASSERT(
        dynamic_shared_bytes <= 32U * 1024U,
        "persistent_loop: FLD update_matter dynamic shared memory exceeds "
        "32 KiB safety cap; reduce n_groups");
  }

  const PersistentLaunchConfig launch = choose_persistent_launch_config(
      static_cast<int>(state.rho.size()), dynamic_shared_bytes);
  PersistentParams params =
      make_params(state, cfg, laser_mesh, planck, nlte_opacity, laser_waveform,
                  t_end, t_next_output, max_steps_remaining,
                  launch.use_cooperative, launch.blocks);
  const bool watch = pk_watch_enabled();
  PkWatchProgressSlab* watch_progress = nullptr;
  if (watch) {
    watch_progress = &pk_watch_progress_slab();
    params.progress = watch_progress->device;
  }
  double* d_phase_prof = nullptr;
  if (params.phase_prof != 0) {
    d_phase_prof = acquire_device_buffer<double>("persistent:phase_prof",
                                                 kPkPhaseProfSlots);
    params.phase_prof_slab = d_phase_prof;
    cuda_check(cudaMemset(d_phase_prof,
                          0,
                          kPkPhaseProfSlots * sizeof(double)),
               "persistent_loop: phase_prof memset failed");
    static bool phase_prof_scalars_logged = false;
    if (!phase_prof_scalars_logged) {
      char phase_prof_scalars[192];
      std::snprintf(phase_prof_scalars,
                    sizeof(phase_prof_scalars),
                    "pk_phase_prof scalars n_groups=%d rays_per_beam=%d "
                    "n_folded=%d grid_blocks=%d",
                    params.n_groups,
                    params.laser_rays_per_beam,
                    params.laser_n_folded,
                    launch.blocks);
      core::log_info(phase_prof_scalars);
      phase_prof_scalars_logged = true;
    }
  }
  if (trace) {
    core::log_info("pk_host: setup pool acquires block begin");
  }
  PersistentDeviceBuffers buffers =
      make_buffers(state, cfg, laser_mesh, launch.reduce_padded_count,
                   launch.use_cooperative != 0 && state.rho.size() >= 64U);
  auto* ring = acquire_device_buffer<PersistentDiagRecord>(
      "persistent_loop:diag_ring", static_cast<std::size_t>(chunk_steps));
  int* d_exit = acquire_device_buffer<int>("persistent_loop:exit", 3);
  double* d_t_dt = acquire_device_buffer<double>("persistent_loop:t_dt", 2);
  if (trace) {
    core::log_info("pk_host: setup pool acquires block end");
  }

  const double h_t_dt[2] = {state.t, state.dt};
  cuda_check(cudaMemcpy(d_t_dt, h_t_dt, sizeof(h_t_dt), cudaMemcpyHostToDevice),
             "persistent_loop: copy t/dt H2D failed");
  cuda_check(cudaMemset(d_exit, 0, 3 * sizeof(int)),
             "persistent_loop: exit memset failed");

  if (trace || params.phase_trace != 0 || params.laser_trace != 0) {
    static bool printf_fifo_limit_set = false;
    if (!printf_fifo_limit_set) {
      cuda_check(cudaDeviceSetLimit(cudaLimitPrintfFifoSize,
                                    64ull * 1024ull * 1024ull),
                 "persistent_loop: set device printf FIFO limit failed");
      core::log_info(
          "persistent_loop: cudaLimitPrintfFifoSize set to 67108864 bytes");
      printf_fifo_limit_set = true;
    }
  }

  cudaError_t err = cudaSuccess;
  if (launch.use_cooperative != 0) {
    void* kernel_args[] = {
        &params,
        &buffers,
        &ring,
        &d_exit,
        &d_t_dt,
    };
    if (trace) {
      char launch_buffer[96];
      std::snprintf(launch_buffer,
                    sizeof(launch_buffer),
                    "pk_host: launch grid=%d coop=%d",
                    launch.blocks,
                    launch.use_cooperative);
      core::log_info(launch_buffer);
    }
    const cudaError_t launch_err = cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(persistent_chunk_kernel<true>),
        dim3(launch.blocks),
        dim3(kBlockSize),
        kernel_args,
        dynamic_shared_bytes,
        nullptr);
    err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
      err = launch_err;
    }
    if (trace) {
      core::log_info(std::string("pk_host: launch rc=") +
                     cudaGetErrorString(err));
    }
  } else {
    if (trace) {
      char launch_buffer[96];
      std::snprintf(launch_buffer,
                    sizeof(launch_buffer),
                    "pk_host: launch grid=%d coop=%d",
                    launch.blocks,
                    launch.use_cooperative);
      core::log_info(launch_buffer);
    }
    persistent_chunk_kernel<false><<<1, kBlockSize, dynamic_shared_bytes>>>(
        params, buffers, ring, d_exit, d_t_dt);
    err = cudaGetLastError();
    if (trace) {
      core::log_info(std::string("pk_host: launch rc=") +
                     cudaGetErrorString(err));
    }
  }
  const std::string message =
      std::string("persistent_loop: persistent_chunk_kernel launch failed: ") +
      cudaGetErrorString(err);
  cuda_check(err, message.c_str());
  if (watch) {
    pk_watch_wait_for_kernel(watch_progress->host);
  } else {
    cuda_check(cudaDeviceSynchronize(),
               "persistent_loop: persistent_chunk_kernel execution failed");
  }
  if (trace) {
    core::log_info("pk_host: sync done");
  }
  if (params.phase_prof != 0) {
    double h_phase_prof[kPkPhaseProfSlots] = {};
    cuda_check(cudaMemcpy(h_phase_prof,
                          d_phase_prof,
                          sizeof(h_phase_prof),
                          cudaMemcpyDeviceToHost),
               "persistent_loop: phase_prof copy D2H failed");
    pk_phase_prof_log(h_phase_prof);
  }

  double h_after[2] = {0.0, 0.0};
  int h_exit[3] = {0, 0, 0};
  cuda_check(cudaMemcpy(h_after, d_t_dt, sizeof(h_after), cudaMemcpyDeviceToHost),
             "persistent_loop: copy t/dt D2H failed");
  cuda_check(cudaMemcpy(h_exit, d_exit, sizeof(h_exit), cudaMemcpyDeviceToHost),
             "persistent_loop: copy exit D2H failed");

  std::vector<PersistentDiagRecord> ring_host(static_cast<std::size_t>(chunk_steps));
  if (chunk_steps > 0) {
    cuda_check(cudaMemcpy(ring_host.data(), ring,
                          ring_host.size() * sizeof(PersistentDiagRecord),
                          cudaMemcpyDeviceToHost),
               "persistent_loop: copy diag ring D2H failed");
  }

  state.t = h_after[0];
  state.dt = h_after[1];
  state.step += h_exit[1];
  for (int i = 0; i < h_exit[1] && i < chunk_steps; ++i) {
    const PersistentDiagRecord& rec =
        ring_host[static_cast<std::size_t>(i)];
    state.E_laser_deposited +=
        std::max(rec.laser_input - rec.laser_escaped, 0.0);
    state.E_laser_escaped += std::max(rec.laser_escaped, 0.0);
    state.E_laser_incident += std::max(rec.laser_input, 0.0);
    state.E_numerical_loss += std::max(rec.laser_skipped, 0.0);
    state.E_floor_injected += std::max(rec.laser_floor, 0.0);
    state.E_safety += std::max(rec.laser_floor, 0.0);
  }
  state.mesh.recompute_geometry_device_only(state.vol.data());
  cuda_check(cudaDeviceSynchronize(),
             "persistent_loop: post-chunk geometry recompute failed");
  state.mesh.sync_device_geometry_to_host(state.vol.data());

  PersistentChunkResult result{};
  result.steps_advanced = h_exit[1];
  result.t_after = state.t;
  result.dt_after = state.dt;
  result.exit_reason = h_exit[0];
  result.error_code = h_exit[2];
  if (result.error_code != 0) {
    result.exit_reason = kExitRetryOrError;
  }
  if (result.exit_reason == kExitRetryOrError) {
    const std::string error_detail = pk_error_description(result.error_code);
    TENRYU_ASSERT(
        false,
        "persistent_loop v1: device-side step failure (exit 3, " +
        error_detail + ") — the "
        "multi-kernel path would enter driver-retry here; retry inside the "
        "persistent loop is a v1.1 item (device-side chunk snapshot + "
        "dt-halved re-run). Re-run this deck with persistent_loop.enabled=False.");
  }
  (void)ring_host;
  if (trace) {
    char buffer[192];
    std::snprintf(buffer,
                  sizeof(buffer),
                  "pk_chunk: steps=%d exit=%d t=%.3e dt=%.3e",
                  result.steps_advanced,
                  result.exit_reason,
                  result.t_after,
                  result.dt_after);
    core::log_info(buffer);
    if (result.steps_advanced > 0) {
      const auto log_dt_record = [](const PersistentDiagRecord& rec) {
        char dt_buffer[192];
        std::snprintf(dt_buffer,
                      sizeof(dt_buffer),
                      "pk_dt s=%d dt=%.3e lim=%d h=%.3e c=%.3e r=%.3e",
                      rec.step,
                      rec.dt,
                      rec.limiter,
                      rec.cand_hydro,
                      rec.cand_cond,
                      rec.cand_rad);
        core::log_info(dt_buffer);
      };
      log_dt_record(ring_host[0]);
      log_dt_record(
          ring_host[static_cast<std::size_t>(result.steps_advanced - 1)]);
    }
  }
  return result;
}

}  // namespace tenryu::coupling
