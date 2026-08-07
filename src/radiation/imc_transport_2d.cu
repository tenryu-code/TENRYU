#include "radiation/imc_transport_2d.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "core/constants.hpp"
#include "core/error.hpp"
#include "parallel/partition.hpp"
#include "radiation/boundary.cuh"
#include "radiation/boundary_distance_2d.cuh"
#include "radiation/interface.hpp"

namespace tenryu::radiation {
namespace {

constexpr unsigned kFullMask = 0xFFFFFFFFu;
constexpr double kGeomEps = 1.0e-12;
constexpr double kInf = INFINITY;
constexpr int kMaxEvents = 10000;
constexpr double kRngEps = 1.0e-16;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

struct RankBoundaryParams2D {
  int ghost_layers = 0;
  int nr_local = 0;
  int nz_local = 0;
  std::int8_t has_left_boundary = 0;
  std::int8_t has_right_boundary = 0;
  std::int8_t has_bottom_boundary = 0;
  std::int8_t has_top_boundary = 0;
};

inline RankBoundaryParams2D make_rank_boundary_params_2d(
    const tenryu::parallel::PartitionInfo* part) {
  RankBoundaryParams2D out{};
  if (part == nullptr) {
    return out;
  }
  out.ghost_layers = std::max(part->ghost_layers, 0);
  out.nr_local = std::max(part->nr_local, 0);
  out.nz_local = std::max(part->nz_local, 0);
  out.has_left_boundary = part->has_left_boundary() ? static_cast<std::int8_t>(1)
                                                     : static_cast<std::int8_t>(0);
  out.has_right_boundary = part->has_right_boundary() ? static_cast<std::int8_t>(1)
                                                       : static_cast<std::int8_t>(0);
  out.has_bottom_boundary = part->has_bottom_boundary() ? static_cast<std::int8_t>(1)
                                                         : static_cast<std::int8_t>(0);
  out.has_top_boundary = part->has_top_boundary() ? static_cast<std::int8_t>(1)
                                                   : static_cast<std::int8_t>(0);
  return out;
}

__device__ inline double rng_uniform(curandStatePhilox4_32_10_t* rng,
                                     std::uint32_t* counter) {
  *counter += 1U;
  return fmin(fmax(curand_uniform_double(rng), kRngEps), 1.0 - kRngEps);
}

__device__ inline double compute_tau_saturated(const double sigma,
                                               const double length) {
  const double sigma_pos = fmax(sigma, 0.0);
  const double length_pos = fmax(length, 0.0);
  if (sigma_pos <= 0.0 || length_pos <= 0.0) {
    return 0.0;
  }
  return fmin(sigma_pos * length_pos, DBL_MAX);
}

__device__ inline double atomic_add_double(double* address, const double value) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, value);
#else
  auto* address_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_ull;
  unsigned long long int assumed = 0ULL;
  do {
    assumed = old;
    old = atomicCAS(address_ull,
                    assumed,
                    __double_as_longlong(
                        value + __longlong_as_double(static_cast<long long>(assumed))));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
#endif
}

__device__ inline void atomic_inc_counter(unsigned long long* counter) {
  if (counter != nullptr) {
    atomicAdd(counter, 1ULL);
  }
}

__device__ inline void sample_half_space_normal_2d(curandStatePhilox4_32_10_t* rng,
                                                    std::uint32_t* counter,
                                                    const double nr,
                                                    const double nz,
                                                    double* dir_r,
                                                    double* dir_z,
                                                    double* dir_phi) {
  const double xi_mu = rng_uniform(rng, counter);
  // NUMERICS §7.7.2 (v1.0 default): cosine-weighted half-space
  // sampling at DDMC->IMC interfaces, p(mu)=2*mu with mu=sqrt(xi).
  const double mu = sqrt(fmax(fmin(xi_mu, 1.0), 0.0));
  const double xi_phi = rng_uniform(rng, counter);
  const double phi = 2.0 * 3.14159265358979323846 * xi_phi;
  const double sin_theta = sqrt(fmax(0.0, 1.0 - mu * mu));
  const double tangent = sin_theta * cos(phi);
  *dir_r = mu * nr - tangent * nz;
  *dir_z = mu * nz + tangent * nr;
  *dir_phi = sin_theta * sin(phi);
}

__device__ inline int opposite_face_2d(const int face) {
  if (face == 0) {
    return 1;
  }
  if (face == 1) {
    return 0;
  }
  if (face == 2) {
    return 3;
  }
  return 2;
}

__device__ inline FaceGeom2D hit_face_geom_2d(const int cell,
                                              const int face,
                                              const int nz,
                                              const double* __restrict__ node_r,
                                              const double* __restrict__ node_z) {
  const int i = cell / nz;
  const int j = cell - i * nz;
  const int n00 = node_index_2d_rz(i, j, nz);
  const int n10 = node_index_2d_rz(i + 1, j, nz);
  const int n11 = node_index_2d_rz(i + 1, j + 1, nz);
  const int n01 = node_index_2d_rz(i, j + 1, nz);
  return face_geom_2d_compat(face,
                             node_r[n00],
                             node_z[n00],
                             node_r[n10],
                             node_z[n10],
                             node_r[n11],
                             node_z[n11],
                             node_r[n01],
                             node_z[n01]);
}

__device__ inline void push_from_face_2d(double& r,
                                         double& z,
                                         const FaceGeom2D& geom,
                                         const double normal_sign,
                                         const double eps,
                                         const bool enforce_axis_floor) {
  push_off_face_2d_compat(
      r, z, normal_sign * geom.nr, normal_sign * geom.nz, eps);
  r = enforce_axis_floor ? fmax(r, eps) : fmax(r, 0.0);
}

__device__ inline bool is_axis_aligned_face_2d(const FaceGeom2D& geom) {
  constexpr double kAlignTol = 1.0e-14;
  const bool r_face = (fabs(fabs(geom.nr) - 1.0) <= kAlignTol) && (fabs(geom.nz) <= kAlignTol);
  const bool z_face = (fabs(geom.nr) <= kAlignTol) && (fabs(fabs(geom.nz) - 1.0) <= kAlignTol);
  return r_face || z_face;
}

__device__ inline void warp_tally(double* rad_dep,
                                  double* rad_E_tally,
                                  const int key,
                                  const double dep,
                                  const double tl) {
#if __CUDA_ARCH__ >= 700
  const unsigned mask = __activemask();
  const int lane = threadIdx.x & 31;
  const unsigned peers = __match_any_sync(mask, key);
  const unsigned lane_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
  const int peer_rank = __popc(peers & lane_mask);
  const int peer_count = __popc(peers);

  double sum_dep = dep;
  double sum_tl = tl;
  // Reduce by peer rank, not physical lane distance, so sparse peer masks are safe.
  for (int offset = 1; offset < peer_count; offset <<= 1) {
    const int src_rank = peer_rank + offset;
    const bool has_src = src_rank < peer_count;
    unsigned src_lane = static_cast<unsigned>(lane);
    if (has_src) {
      const unsigned mapped = __fns(peers, 0u, src_rank + 1);
      src_lane = (mapped < 32u) ? mapped : static_cast<unsigned>(lane);
    }
    const double tmp_dep = __shfl_sync(peers, sum_dep, static_cast<int>(src_lane));
    const double tmp_tl = __shfl_sync(peers, sum_tl, static_cast<int>(src_lane));
    if (has_src) {
      sum_dep += tmp_dep;
      sum_tl += tmp_tl;
    }
  }

  if (peer_rank == 0) {
    if (sum_dep != 0.0) {
      atomic_add_double(&rad_dep[key], sum_dep);
    }
    if (sum_tl != 0.0) {
      atomic_add_double(&rad_E_tally[key], sum_tl);
    }
  }
#else
  if (dep != 0.0) {
    atomic_add_double(&rad_dep[key], dep);
  }
  if (tl != 0.0) {
    atomic_add_double(&rad_E_tally[key], tl);
  }
#endif
}

__device__ inline bool has_non_finite_imc_state_2d(const double r,
                                                    const double z,
                                                    const double dir_r,
                                                    const double dir_z,
                                                    const double dir_phi,
                                                    const double E) {
  return !isfinite(r) || !isfinite(z) || !isfinite(dir_r) || !isfinite(dir_z) ||
         !isfinite(dir_phi) || !isfinite(E);
}

__device__ inline void mark_nan_particle(tenryu::core::DeviceErrorFlags* error_flags,
                                         double* E_numerical_loss,
                                         double* E) {
  if (error_flags != nullptr) {
    atomicExch(&error_flags->nan_particle, 1);
  }
  if (E_numerical_loss != nullptr && isfinite(*E) && *E > 0.0) {
    atomic_add_double(E_numerical_loss, *E);
  }
  *E = 0.0;
}

__global__ __launch_bounds__(128, 8) void imc_transport_2d_kernel(
    double* __restrict__ pos_r_arr,
    double* __restrict__ pos_z_arr,
    double* __restrict__ dir_r_arr,
    double* __restrict__ dir_z_arr,
    double* __restrict__ dir_phi_arr,
    double* __restrict__ energy_arr,
    double* __restrict__ birth_energy_arr,
    double* __restrict__ time_remain_arr,
    std::uint64_t* __restrict__ global_id_arr,
    std::uint32_t* __restrict__ rng_counter_arr,
    std::int32_t* __restrict__ cell_id_arr,
    std::uint16_t* __restrict__ group_id_arr,
    std::uint8_t* __restrict__ mode_arr,
    std::uint8_t* __restrict__ alive_arr,
    const double* __restrict__ sigma_a_eff,
    const double* __restrict__ sigma_s_eff,
    const double* __restrict__ Te,
    const double* __restrict__ vol,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const TransportMode* __restrict__ ddmc_mode,
    const int ddmc_zero_flux_interfaces,
    const int n_groups_for_mode,
    const int emissivity_preserving,
    const double* __restrict__ sigma_R,
    const double* __restrict__ fleck_f,
    const double* __restrict__ sigma_a_raw,
    const double* __restrict__ eta_cdf,
    const PlanckTableDeviceView planck,
    const int inelastic_scatter,
    double* __restrict__ rad_dep,
    double* __restrict__ rad_E_tally,
    double* __restrict__ E_escape,
    double* __restrict__ E_numerical_loss,
    unsigned long long* __restrict__ imc_absorbed,
    unsigned long long* __restrict__ imc_escaped,
    unsigned long long* __restrict__ interface_transitions,
    unsigned long long* __restrict__ interface_reflections,
    unsigned long long* __restrict__ conversion_prob_violations,
    unsigned long long* __restrict__ cnt_boundary,
    unsigned long long* __restrict__ cnt_scatter,
    unsigned long long* __restrict__ cnt_census,
    unsigned long long* __restrict__ cnt_absorb_kill,
    unsigned long long* __restrict__ cnt_absorb_survive,
    unsigned long long* __restrict__ cnt_roulette_kill,
    int* __restrict__ global_counter,
    const int nr,
    const int nz,
    const int n_cells,
    const int n_groups,
    const int n_imc,
    const double dt,
    const double E_avg,
    const double w_cutoff,
    const double p_survival,
    const double f_cutoff,
    const int tail_pass,
    const int bc_inner,
    const int bc_outer,
    const int bc_bottom_z,
    const int bc_top_z,
    const std::uint64_t step_number,
    const std::uint64_t user_seed,
    const int ghost_layers,
    const int nr_local,
    const int nz_local,
    const std::int8_t has_left_boundary,
    const std::int8_t has_right_boundary,
    const std::int8_t has_bottom_boundary,
    const std::int8_t has_top_boundary,
    tenryu::core::DeviceErrorFlags* __restrict__ error_flags) {
  const int lane = threadIdx.x & 31;

  int my_idx = -1;
  bool active = false;
  int events = 0;

  double r_l = 0.0;
  double z_l = 0.0;
  double dir_r_l = 0.0;
  double dir_z_l = 0.0;
  double dir_phi_l = 0.0;
  double E_l = 0.0;
  double birth_E_l = 0.0;
  double t_remain_l = 0.0;
  double tau_scatter_remain = 0.0;
  std::int32_t cell_l = -1;
  std::uint16_t group_l = 0;
  std::uint8_t alive_l = kAlive;
  std::uint64_t gid_l = 0;
  std::uint32_t rng_counter_l = 0;
  std::uint32_t local_cnt_boundary = 0;
  std::uint32_t local_cnt_scatter = 0;
  std::uint32_t local_cnt_census = 0;
  std::uint32_t local_cnt_absorb_kill = 0;
  std::uint32_t local_cnt_absorb_survive = 0;
  std::uint32_t local_cnt_roulette_kill = 0;
  curandStatePhilox4_32_10_t rng;

  while (true) {
    const unsigned need = __ballot_sync(kFullMask, !active);
    const int n_need = __popc(need);

    int base = 0;
    if (lane == 0 && n_need > 0) {
      base = atomicAdd(global_counter, n_need);
    }
    base = __shfl_sync(kFullMask, base, 0);

    if (!active) {
      const unsigned lane_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
      const int offset = __popc(need & lane_mask);
      const int idx = base + offset;
      if (offset < n_need && idx < n_imc) {
        my_idx = idx;
        local_cnt_boundary = 0;
        local_cnt_scatter = 0;
        local_cnt_census = 0;
        local_cnt_absorb_kill = 0;
        local_cnt_absorb_survive = 0;
        local_cnt_roulette_kill = 0;
        r_l = pos_r_arr[my_idx];
        z_l = pos_z_arr[my_idx];
        dir_r_l = dir_r_arr[my_idx];
        dir_z_l = dir_z_arr[my_idx];
        dir_phi_l = dir_phi_arr[my_idx];
        E_l = energy_arr[my_idx];
        birth_E_l = birth_energy_arr[my_idx];
        t_remain_l = time_remain_arr[my_idx];
        if (tail_pass != 0 && t_remain_l <= 0.0) {
          alive_l = alive_arr[my_idx];
          active = false;
        } else if (t_remain_l <= 0.0) {
          t_remain_l = dt;
        }
        gid_l = global_id_arr[my_idx];
        rng_counter_l = rng_counter_arr[my_idx];
        cell_l = cell_id_arr[my_idx];
        group_l = group_id_arr[my_idx];
        alive_l = alive_arr[my_idx];
        if (alive_l != kAlive || (tail_pass != 0 && t_remain_l <= 0.0)) {
          alive_l = kDead;
          active = false;
        } else {
          curand_init(gid_l ^ user_seed,
                      static_cast<unsigned long long>(step_number),
                      static_cast<unsigned long long>(rng_counter_l),
                      &rng);
          tau_scatter_remain = -log(rng_uniform(&rng, &rng_counter_l));
          events = 0;
          active = true;
        }
      }
    }

    const bool any_active = __any_sync(kFullMask, active);
    if (!any_active) {
      const int consumed = atomicAdd(global_counter, 0);
      if (consumed >= n_imc) {
        break;
      }
    }

    bool killed_before_transport = false;

    if (active) {
      events += 1;
      if (events >= kMaxEvents) {
        if (error_flags != nullptr) {
          atomicAdd(&error_flags->infinite_loop, 1);
        }
        if (E_l > 0.0 && E_numerical_loss != nullptr) {
          atomic_add_double(E_numerical_loss, E_l);
        }
        E_l = 0.0;
        alive_l = kDead;
        active = false;
        killed_before_transport = true;
      }
    }

    if (active) {
      if (cell_l < 0 || cell_l >= n_cells || group_l >= static_cast<unsigned>(n_groups)) {
        if (error_flags != nullptr) {
          atomicExch(&error_flags->invalid_cell, 1);
        }
        if (E_l > 0.0 && E_numerical_loss != nullptr) {
          atomic_add_double(E_numerical_loss, E_l);
        }
        E_l = 0.0;
        alive_l = kDead;
        active = false;
        killed_before_transport = true;
      }
    }

    if (killed_before_transport && my_idx >= 0) {
      if (cnt_boundary != nullptr && local_cnt_boundary > 0) {
        atomicAdd(cnt_boundary, static_cast<unsigned long long>(local_cnt_boundary));
      }
      if (cnt_scatter != nullptr && local_cnt_scatter > 0) {
        atomicAdd(cnt_scatter, static_cast<unsigned long long>(local_cnt_scatter));
      }
      if (cnt_census != nullptr && local_cnt_census > 0) {
        atomicAdd(cnt_census, static_cast<unsigned long long>(local_cnt_census));
      }
      if (cnt_absorb_kill != nullptr && local_cnt_absorb_kill > 0) {
        atomicAdd(cnt_absorb_kill, static_cast<unsigned long long>(local_cnt_absorb_kill));
      }
      if (cnt_absorb_survive != nullptr && local_cnt_absorb_survive > 0) {
        atomicAdd(cnt_absorb_survive,
                  static_cast<unsigned long long>(local_cnt_absorb_survive));
      }
      if (cnt_roulette_kill != nullptr && local_cnt_roulette_kill > 0) {
        atomicAdd(cnt_roulette_kill, static_cast<unsigned long long>(local_cnt_roulette_kill));
      }
      pos_r_arr[my_idx] = r_l;
      pos_z_arr[my_idx] = z_l;
      dir_r_arr[my_idx] = dir_r_l;
      dir_z_arr[my_idx] = dir_z_l;
      dir_phi_arr[my_idx] = dir_phi_l;
      energy_arr[my_idx] = E_l;
      time_remain_arr[my_idx] = t_remain_l;
      birth_energy_arr[my_idx] = birth_E_l;
      cell_id_arr[my_idx] = cell_l;
      group_id_arr[my_idx] = group_l;
      alive_arr[my_idx] = alive_l;
      if (alive_l == kAlive) {
        if (mode_arr[my_idx] != kModeDDMC) {
          mode_arr[my_idx] = kModeIMC;
        }
      }
      rng_counter_arr[my_idx] = rng_counter_l;
    }

    if (active) {
      const int key = cell_l * n_groups + static_cast<int>(group_l);
      const double sigma_a = sigma_a_eff[key];
      const double sigma_s = sigma_s_eff[key];

      const BoundaryHit2D hit = boundary_distance_2d_rz(r_l,
                                                        z_l,
                                                        dir_r_l,
                                                        dir_z_l,
                                                        dir_phi_l,
                                                        cell_l,
                                                        nr,
                                                        nz,
                                                        node_r,
                                                        node_z);
      const double s_bdry = (hit.face >= 0) ? hit.s : kInf;
      const double s_cen = tenryu::core::constants::c_light * t_remain_l;
      const double s_scatter = (sigma_s > 0.0)
                                   ? ((tau_scatter_remain > 1.0e-14)
                                          ? (tau_scatter_remain / sigma_s)
                                          : 0.0)
                                   : kInf;

      const double s_min = fmin(s_cen, fmin(s_bdry, s_scatter));

      const double E_old = E_l;
      const double tau = compute_tau_saturated(sigma_a, s_min);
      double dep = 0.0;
      if (tau < 1.0e-6) {
        dep = E_old * tau * (1.0 - 0.5 * tau);
      } else {
        dep = -E_old * expm1(-tau);
      }
      dep = fmin(fmax(dep, 0.0), E_old);
      E_l = E_old - dep;
      if (dep > 0.0) {
        ++local_cnt_absorb_survive;
      }

      const double tl = (sigma_a > 0.0) ? (dep / sigma_a) : (E_old * s_min);
      if (dep != 0.0 || tl != 0.0) {
        warp_tally(rad_dep, rad_E_tally, key, dep, tl);
      }

      const double r_old = r_l;
      const double r2_new =
          fmax(0.0,
               r_old * r_old + 2.0 * r_old * dir_r_l * s_min +
                   (dir_r_l * dir_r_l + dir_phi_l * dir_phi_l) * s_min * s_min);
      const double r_new = sqrt(r2_new);
      z_l += dir_z_l * s_min;
      if (r_new > 1.0e-30) {
        const double inv_r_new = 1.0 / r_new;
        const double dir_r_new =
            (dir_r_l * r_old + (dir_r_l * dir_r_l + dir_phi_l * dir_phi_l) * s_min) *
            inv_r_new;
        const double dir_phi_new = (dir_phi_l * r_old) * inv_r_new;
        r_l = r_new;
        dir_r_l = dir_r_new;
        dir_phi_l = dir_phi_new;
      } else {
        r_l = 0.0;
      }
      t_remain_l -= s_min / tenryu::core::constants::c_light;
      if (sigma_s > 0.0) {
        tau_scatter_remain -= sigma_s * s_min;
      }

      if (E_l <= 0.0) {
        E_l = 0.0;
        alive_l = kDead;
        active = false;
        ++local_cnt_absorb_kill;
        atomic_inc_counter(imc_absorbed);
      }

      if (active) {
        const bool is_census =
            (t_remain_l <= 1.0e-10 * dt) ||
            (s_cen <= s_bdry && s_cen <= s_scatter);
        if (is_census) {
          ++local_cnt_census;
          t_remain_l = 0.0;
          alive_l = kAlive;
          active = false;
        } else {
          const bool is_boundary = (s_bdry <= s_scatter);
          if (is_boundary) {
            ++local_cnt_boundary;
            if (hit.face < 0) {
              if (error_flags != nullptr) {
                atomicExch(&error_flags->invalid_boundary, 1);
              }
              if (E_l > 0.0 && E_numerical_loss != nullptr) {
                atomic_add_double(E_numerical_loss, E_l);
              }
              E_l = 0.0;
              alive_l = kDead;
              active = false;
            } else if (!hit.is_boundary) {
              bool treat_as_internal = true;
              bool mark_emigrant = false;
              if ((ghost_layers > 0) && (nr_local > 0) && (nz_local > 0) &&
                  hit.neighbor >= 0 && hit.neighbor < n_cells) {
                const int ni_ghost = hit.neighbor / nz;
                const int nj_ghost = hit.neighbor - ni_ghost * nz;
                bool enters_ghost = false;
                bool has_rank_boundary = true;
                if (hit.face == 0) {
                  enters_ghost = (ni_ghost < ghost_layers);
                  has_rank_boundary = (has_left_boundary != 0);
                } else if (hit.face == 1) {
                  enters_ghost = (ni_ghost >= (ghost_layers + nr_local));
                  has_rank_boundary = (has_right_boundary != 0);
                } else if (hit.face == 2) {
                  enters_ghost = (nj_ghost < ghost_layers);
                  has_rank_boundary = (has_bottom_boundary != 0);
                } else if (hit.face == 3) {
                  enters_ghost = (nj_ghost >= (ghost_layers + nz_local));
                  has_rank_boundary = (has_top_boundary != 0);
                }

                if (enters_ghost) {
                  if (!has_rank_boundary) {
                    mark_emigrant = true;
                  }
                  // Physical boundary at rank edge: reflect/escape path below.
                  treat_as_internal = false;
                }
              }

              if (mark_emigrant) {
                cell_l = -(100 + hit.face);
                alive_l = kAlive;
                active = false;
              } else if (treat_as_internal) {
              const FaceGeom2D hit_geom =
                  hit_face_geom_2d(cell_l, hit.face, nz, node_r, node_z);
              const int neighbor_cell = hit.neighbor;
              bool neighbor_is_ddmc = false;
              if (ddmc_mode != nullptr && neighbor_cell >= 0 && neighbor_cell < n_cells &&
                  static_cast<int>(group_l) < n_groups_for_mode) {
                const int mode_idx =
                    neighbor_cell * n_groups_for_mode + static_cast<int>(group_l);
                neighbor_is_ddmc = (ddmc_mode[mode_idx] == TransportMode::DDMC);
              }

              if (neighbor_is_ddmc && ddmc_zero_flux_interfaces != 0) {
                sample_half_space_normal_2d(&rng,
                                            &rng_counter_l,
                                            -hit_geom.nr,
                                            -hit_geom.nz,
                                            &dir_r_l,
                                            &dir_z_l,
                                            &dir_phi_l);
                push_from_face_2d(
                    r_l, z_l, hit_geom, -1.0, kGeomEps, false);
                atomic_inc_counter(interface_reflections);
              } else if (neighbor_is_ddmc) {
                const int sigma_idx =
                    neighbor_cell * n_groups + static_cast<int>(group_l);
                const double sigma_r =
                    (sigma_R != nullptr) ? fmax(sigma_R[sigma_idx], 0.0) : 0.0;

                const int ni = neighbor_cell / nz;
                const int nj = neighbor_cell - ni * nz;
                const int enter_face = opposite_face_2d(hit.face);
                const int n00 = node_index_2d_rz(ni, nj, nz);
                const int n10 = node_index_2d_rz(ni + 1, nj, nz);
                const int n11 = node_index_2d_rz(ni + 1, nj + 1, nz);
                const int n01 = node_index_2d_rz(ni, nj + 1, nz);
                const double r0 = node_r[n00];
                const double z0 = node_z[n00];
                const double r1 = node_r[n10];
                const double z1 = node_z[n10];
                const double r2 = node_r[n11];
                const double z2 = node_z[n11];
                const double r3 = node_r[n01];
                const double z3 = node_z[n01];
                const double r_max = fmax(fmax(r0, r1), fmax(r2, r3));

                double ra = 0.0;
                double za = 0.0;
                double rb = 0.0;
                double zb = 0.0;
                if (enter_face == 0) {
                  ra = r0;
                  za = z0;
                  rb = r3;
                  zb = z3;
                } else if (enter_face == 1) {
                  ra = r1;
                  za = z1;
                  rb = r2;
                  zb = z2;
                } else if (enter_face == 2) {
                  ra = r0;
                  za = z0;
                  rb = r1;
                  zb = z1;
                } else {
                  ra = r3;
                  za = z3;
                  rb = r2;
                  zb = z2;
                }
                const double L_m = hypot(rb - ra, zb - za);
                const double R_bar = 0.5 * (ra + rb);
                const double V_i = (vol != nullptr) ? fmax(vol[neighbor_cell], 0.0) : 0.0;
                const double delta_x = compute_delta_x_m_2d_rz(V_i, R_bar, L_m, r_max);

                double omega = 0.0;
                if (fleck_f != nullptr) {
                  omega = clamp01(1.0 - fleck_f[neighbor_cell]);
                } else if (sigma_R != nullptr && sigma_a_raw != nullptr && sigma_r > 0.0) {
                  const double sigma_a_c = fmax(sigma_a_raw[sigma_idx], 0.0);
                  omega = clamp01(1.0 - sigma_a_c / sigma_r);
                }

                const double mu_face = dir_r_l * hit_geom.nr + dir_z_l * hit_geom.nz;
                InterfaceProbabilityResult p_result{};
                if (emissivity_preserving != 0) {
                  p_result = compute_P_hat_emissivity(sigma_r, delta_x, omega, mu_face);
                } else {
                  p_result.probability = compute_standard_p_mu(sigma_r, delta_x, mu_face);
                }
                if (p_result.used_standard_fallback) {
                  atomic_inc_counter(conversion_prob_violations);
                }

                if (rng_uniform(&rng, &rng_counter_l) < p_result.probability) {
                  cell_l = neighbor_cell;
                  if (cell_l < 0 || cell_l >= n_cells) {
                    if (error_flags != nullptr) {
                      atomicExch(&error_flags->invalid_cell, 1);
                    }
                    if (E_l > 0.0 && E_numerical_loss != nullptr) {
                      atomic_add_double(E_numerical_loss, E_l);
                    }
                    E_l = 0.0;
                    alive_l = kDead;
                    active = false;
                  } else {
                    mode_arr[my_idx] = kModeDDMC;
                    t_remain_l = 0.0;
                    const double nan = NAN;
                    r_l = nan;
                    z_l = nan;
                    dir_r_l = nan;
                    dir_z_l = nan;
                    dir_phi_l = nan;
                    atomic_inc_counter(interface_transitions);
                    alive_l = kAlive;
                    active = false;
                  }
                } else {
                  sample_half_space_normal_2d(&rng,
                                              &rng_counter_l,
                                              -hit_geom.nr,
                                              -hit_geom.nz,
                                              &dir_r_l,
                                              &dir_z_l,
                                              &dir_phi_l);
                  push_from_face_2d(
                      r_l, z_l, hit_geom, -1.0, kGeomEps, false);
                  atomic_inc_counter(interface_reflections);
                }
              } else {
                cell_l = neighbor_cell;
                if (cell_l < 0 || cell_l >= n_cells) {
                  if (error_flags != nullptr) {
                    atomicExch(&error_flags->invalid_cell, 1);
                  }
                  if (E_l > 0.0 && E_numerical_loss != nullptr) {
                    atomic_add_double(E_numerical_loss, E_l);
                  }
                  E_l = 0.0;
                  alive_l = kDead;
                  active = false;
                } else {
                  push_from_face_2d(
                      r_l, z_l, hit_geom, +1.0, kGeomEps, false);
                }
              }  // close inner if/else (push_from_face_2d)
              } else {
                // enters_ghost && has_rank_boundary: physical boundary at rank edge.
                int bc = kBoundaryReflect;
                if (hit.face == 0) {
                  bc = bc_inner;
                } else if (hit.face == 1) {
                  bc = bc_outer;
                } else if (hit.face == 2) {
                  bc = bc_bottom_z;
                } else if (hit.face == 3) {
                  bc = bc_top_z;
                }
                const FaceGeom2D hit_geom =
                    hit_face_geom_2d(cell_l, hit.face, nz, node_r, node_z);
                if (is_escape_boundary(bc)) {
                  atomic_add_double(&E_escape[group_l], E_l);
                  E_l = 0.0;
                  alive_l = kDead;
                  active = false;
                  atomic_inc_counter(imc_escaped);
                } else if (bc == kBoundaryReflect) {
                  if (is_axis_aligned_face_2d(hit_geom)) {
                    if (hit.face == 0 || hit.face == 1) {
                      dir_r_l = -dir_r_l;
                      if (hit.face == 1) {
                        r_l = fmax(r_l - kGeomEps, 0.0);
                      } else {
                        r_l = fmax(r_l, kGeomEps);
                      }
                    } else {
                      dir_z_l = -dir_z_l;
                      if (hit.face == 2) {
                        z_l += kGeomEps;
                      } else {
                        z_l -= kGeomEps;
                      }
                    }
                  } else {
                    const double dot = dir_r_l * hit_geom.nr + dir_z_l * hit_geom.nz;
                    dir_r_l -= 2.0 * dot * hit_geom.nr;
                    dir_z_l -= 2.0 * dot * hit_geom.nz;
                    push_from_face_2d(
                        r_l, z_l, hit_geom, -1.0, kGeomEps, false);
                  }
                }
              }  // close else-if treat_as_internal block
            } else {  // boundary handling branch
              int bc = kBoundaryReflect;
              if (hit.face == 0) {
                bc = bc_inner;
              } else if (hit.face == 1) {
                bc = bc_outer;
              } else if (hit.face == 2) {
                bc = bc_bottom_z;
              } else if (hit.face == 3) {
                bc = bc_top_z;
              }
              const FaceGeom2D hit_geom =
                  hit_face_geom_2d(cell_l, hit.face, nz, node_r, node_z);

              bool axis_face = false;
              const int axis_i = ((ghost_layers > 0) && (nr_local > 0))
                                     ? ghost_layers
                                     : 0;
              if ((hit.face == 0) && ((cell_l / nz) == axis_i) &&
                  (bc_inner == kBoundaryReflect)) {
                const int ci = cell_l / nz;
                const int cj = cell_l - ci * nz;
                const int n00 = node_index_2d_rz(ci, cj, nz);
                const int n01 = node_index_2d_rz(ci, cj + 1, nz);
                const int n_outer0 = node_index_2d_rz(nr, cj, nz);
                const int n_outer1 = node_index_2d_rz(nr, cj + 1, nz);
                const double r_face = 0.5 * (node_r[n00] + node_r[n01]);
                const double r_max_local =
                    fmax(fabs(node_r[n_outer0]), fabs(node_r[n_outer1]));
                axis_face = (r_face <= 1.0e-10 * fmax(r_max_local, 1.0));
              }
              if (axis_face) {
                dir_r_l = -dir_r_l;
                dir_phi_l = -dir_phi_l;
                // Legacy-compatible axis handling: avoid z-shift and keep strict R-floor.
                r_l = fmax(r_l, kGeomEps);
              } else if (is_escape_boundary(bc)) {
                atomic_add_double(&E_escape[group_l], E_l);
                E_l = 0.0;
                alive_l = kDead;
                active = false;
                atomic_inc_counter(imc_escaped);
              } else if (bc == kBoundaryReflect) {
                if (is_axis_aligned_face_2d(hit_geom)) {
                  // Keep exact legacy behavior on rectangular faces.
                  if (hit.face == 0 || hit.face == 1) {
                    dir_r_l = -dir_r_l;
                    if (hit.face == 1) {
                      r_l = fmax(r_l - kGeomEps, 0.0);
                    } else {
                      r_l = fmax(r_l, kGeomEps);
                    }
                  } else {
                    dir_z_l = -dir_z_l;
                    if (hit.face == 2) {
                      z_l += kGeomEps;
                    } else {
                      z_l -= kGeomEps;
                    }
                  }
                } else {
                  const double dot = dir_r_l * hit_geom.nr + dir_z_l * hit_geom.nz;
                  dir_r_l -= 2.0 * dot * hit_geom.nr;
                  dir_z_l -= 2.0 * dot * hit_geom.nz;
                  push_from_face_2d(
                      r_l, z_l, hit_geom, -1.0, kGeomEps, false);
                }
              } else {
                if (error_flags != nullptr) {
                  atomicExch(&error_flags->invalid_boundary, 1);
                }
                if (E_l > 0.0 && E_numerical_loss != nullptr) {
                  atomic_add_double(E_numerical_loss, E_l);
                }
                E_l = 0.0;
                alive_l = kDead;
                active = false;
              }
            }
          } else {
            ++local_cnt_scatter;
            tau_scatter_remain = -log(rng_uniform(&rng, &rng_counter_l));
            const double xi_mu = rng_uniform(&rng, &rng_counter_l);
            const double xi_phi = rng_uniform(&rng, &rng_counter_l);
            const double mu_z_new = 2.0 * xi_mu - 1.0;
            const double phi = 2.0 * 3.14159265358979323846 * xi_phi;
            const double sin_theta = sqrt(fmax(0.0, 1.0 - mu_z_new * mu_z_new));
            dir_r_l = sin_theta * cos(phi);
            dir_z_l = mu_z_new;
            dir_phi_l = sin_theta * sin(phi);

            if (inelastic_scatter != 0 && n_groups > 1) {
              const int cell_base = cell_l * n_groups;
              if (eta_cdf != nullptr) {
                const double xi_g = rng_uniform(&rng, &rng_counter_l);
                int g_new = n_groups - 1;
                // eta_cdf is assumed to be a valid CDF: monotonically non-decreasing,
                // cdf[n_groups-1] == 1.0. This is guaranteed by compute_nlte_coefficients().
                // No runtime validation is performed here for performance.
                for (int g = 0; g < n_groups; ++g) {
                  const double cdf_g =
                      fmin(fmax(eta_cdf[cell_base + g], 0.0), 1.0);
                  if (xi_g <= cdf_g) {
                    g_new = g;
                    break;
                  }
                }
                group_l = static_cast<std::uint16_t>(g_new);
              } else {
                const double Te_c = fmax(Te[cell_l], 1.0e-12);

                double cdf_total = 0.0;
                for (int g = 0; g < n_groups; ++g) {
                  const double sigma_g = fmax(sigma_a_eff[cell_base + g], 0.0);
                  const double b_g = fmax(planck.interpolate_b(g, Te_c), 0.0);
                  cdf_total += sigma_g * b_g;
                }

                if (cdf_total > 0.0) {
                  const double xi_g = rng_uniform(&rng, &rng_counter_l) * cdf_total;
                  double cdf = 0.0;
                  int g_new = static_cast<int>(group_l);
                  for (int g = 0; g < n_groups; ++g) {
                    const double sigma_g = fmax(sigma_a_eff[cell_base + g], 0.0);
                    const double b_g = fmax(planck.interpolate_b(g, Te_c), 0.0);
                    cdf += sigma_g * b_g;
                    if (xi_g <= cdf) {
                      g_new = g;
                      break;
                    }
                  }
                  group_l = static_cast<std::uint16_t>(g_new);
                }
              }
            }
          }
        }
      }

      if (active && f_cutoff > 0.0 && E_l < f_cutoff * birth_E_l) {
        const int key_cut = cell_l * n_groups + static_cast<int>(group_l);
        warp_tally(rad_dep, rad_E_tally, key_cut, E_l, 0.0);
        E_l = 0.0;
        alive_l = kDead;
        active = false;
        atomic_inc_counter(imc_absorbed);
      }

      if (active && E_l < w_cutoff * fmax(E_avg, 1.0e-300)) {
        if (p_survival > 0.0 && rng_uniform(&rng, &rng_counter_l) < p_survival) {
          // Russian roulette is unbiased in expectation; per-realization energy
          // has variance because survivors are up-weighted by 1/p_survival.
          E_l /= p_survival;
        } else {
          const int key_rr = cell_l * n_groups + static_cast<int>(group_l);
          warp_tally(rad_dep, rad_E_tally, key_rr, E_l, 0.0);
          E_l = 0.0;
          alive_l = kDead;
          active = false;
          ++local_cnt_roulette_kill;
          atomic_inc_counter(imc_absorbed);
          if (error_flags != nullptr) {
            atomicExch(&error_flags->roulette_kill, 1);
          }
        }
      }

      if (active &&
          has_non_finite_imc_state_2d(r_l, z_l, dir_r_l, dir_z_l, dir_phi_l, E_l)) {
        mark_nan_particle(error_flags, E_numerical_loss, &E_l);
        r_l = 0.0;
        z_l = 0.0;
        dir_r_l = 1.0;
        dir_z_l = 0.0;
        dir_phi_l = 0.0;
        t_remain_l = 0.0;
        alive_l = kDead;
        active = false;
      }

      if (!active) {
        if (cnt_boundary != nullptr && local_cnt_boundary > 0) {
          atomicAdd(cnt_boundary, static_cast<unsigned long long>(local_cnt_boundary));
        }
        if (cnt_scatter != nullptr && local_cnt_scatter > 0) {
          atomicAdd(cnt_scatter, static_cast<unsigned long long>(local_cnt_scatter));
        }
        if (cnt_census != nullptr && local_cnt_census > 0) {
          atomicAdd(cnt_census, static_cast<unsigned long long>(local_cnt_census));
        }
        if (cnt_absorb_kill != nullptr && local_cnt_absorb_kill > 0) {
          atomicAdd(cnt_absorb_kill, static_cast<unsigned long long>(local_cnt_absorb_kill));
        }
        if (cnt_absorb_survive != nullptr && local_cnt_absorb_survive > 0) {
          atomicAdd(cnt_absorb_survive,
                    static_cast<unsigned long long>(local_cnt_absorb_survive));
        }
        if (cnt_roulette_kill != nullptr && local_cnt_roulette_kill > 0) {
          atomicAdd(cnt_roulette_kill, static_cast<unsigned long long>(local_cnt_roulette_kill));
        }
        const bool is_ddmc_sentinel_state =
            (alive_l == kAlive) && (mode_arr[my_idx] == kModeDDMC);
        if (!is_ddmc_sentinel_state &&
            has_non_finite_imc_state_2d(r_l, z_l, dir_r_l, dir_z_l, dir_phi_l, E_l)) {
          mark_nan_particle(error_flags, E_numerical_loss, &E_l);
          r_l = 0.0;
          z_l = 0.0;
          dir_r_l = 1.0;
          dir_z_l = 0.0;
          dir_phi_l = 0.0;
          t_remain_l = 0.0;
          alive_l = kDead;
        }

        pos_r_arr[my_idx] = r_l;
        pos_z_arr[my_idx] = z_l;
        dir_r_arr[my_idx] = dir_r_l;
        dir_z_arr[my_idx] = dir_z_l;
        dir_phi_arr[my_idx] = dir_phi_l;
        energy_arr[my_idx] = E_l;
        time_remain_arr[my_idx] = t_remain_l;
        birth_energy_arr[my_idx] = birth_E_l;
        cell_id_arr[my_idx] = cell_l;
        group_id_arr[my_idx] = group_l;
        alive_arr[my_idx] = alive_l;
        if (alive_l == kAlive) {
          if (mode_arr[my_idx] != kModeDDMC) {
            mode_arr[my_idx] = kModeIMC;
          }
        }
        rng_counter_arr[my_idx] = rng_counter_l;
      }
    }
  }

  if (active && my_idx >= 0) {
    const bool is_ddmc_sentinel_state =
        (alive_l == kAlive) && (mode_arr[my_idx] == kModeDDMC);
    if (!is_ddmc_sentinel_state &&
        has_non_finite_imc_state_2d(r_l, z_l, dir_r_l, dir_z_l, dir_phi_l, E_l)) {
      mark_nan_particle(error_flags, E_numerical_loss, &E_l);
      r_l = 0.0;
      z_l = 0.0;
      dir_r_l = 1.0;
      dir_z_l = 0.0;
      dir_phi_l = 0.0;
      t_remain_l = 0.0;
      alive_l = kDead;
    }

    pos_r_arr[my_idx] = r_l;
    pos_z_arr[my_idx] = z_l;
    dir_r_arr[my_idx] = dir_r_l;
    dir_z_arr[my_idx] = dir_z_l;
    dir_phi_arr[my_idx] = dir_phi_l;
    energy_arr[my_idx] = E_l;
    time_remain_arr[my_idx] = t_remain_l;
    birth_energy_arr[my_idx] = birth_E_l;
    cell_id_arr[my_idx] = cell_l;
    group_id_arr[my_idx] = group_l;
    alive_arr[my_idx] = alive_l;
    if (alive_l == kAlive) {
      if (mode_arr[my_idx] != kModeDDMC) {
        mode_arr[my_idx] = kModeIMC;
      }
    }
    rng_counter_arr[my_idx] = rng_counter_l;
  }
}

__global__ void update_imc_weight_2d_kernel(const int n_imc,
                                            const double* __restrict__ energy_arr,
                                            const double* __restrict__ birth_energy_arr,
                                            double* __restrict__ weight_arr) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_imc) {
    return;
  }
  const double birth = birth_energy_arr[idx];
  const double energy = fmax(energy_arr[idx], 0.0);
  weight_arr[idx] = (birth > 0.0) ? (energy / birth) : 0.0;
}

}  // namespace

void imc_transport_2d_persistent_cuda(const TransportInputs& in,
                                      const tenryu::parallel::PartitionInfo* part) {
  TENRYU_ASSERT(in.pool != nullptr, "imc_transport_2d requires pool");
  TENRYU_ASSERT(in.sigma_a_eff != nullptr, "imc_transport_2d requires sigma_a_eff");
  TENRYU_ASSERT(in.sigma_s_eff != nullptr, "imc_transport_2d requires sigma_s_eff");
  TENRYU_ASSERT(in.Te != nullptr, "imc_transport_2d requires Te");
  TENRYU_ASSERT(in.vol != nullptr, "imc_transport_2d requires vol");
  TENRYU_ASSERT(in.node_r != nullptr, "imc_transport_2d requires node_r");
  TENRYU_ASSERT(in.node_z != nullptr, "imc_transport_2d requires node_z");
  TENRYU_ASSERT(in.rad_dep != nullptr, "imc_transport_2d requires rad_dep");
  TENRYU_ASSERT(in.rad_E_tally != nullptr, "imc_transport_2d requires rad_E_tally");
  TENRYU_ASSERT(in.E_escape != nullptr, "imc_transport_2d requires E_escape");
  TENRYU_ASSERT(in.E_numerical_loss != nullptr,
                "imc_transport_2d requires E_numerical_loss");
  TENRYU_ASSERT(in.imc_absorbed != nullptr, "imc_transport_2d requires imc_absorbed");
  TENRYU_ASSERT(in.imc_escaped != nullptr, "imc_transport_2d requires imc_escaped");
  TENRYU_ASSERT(in.mesh_dim == 2, "imc_transport_2d requires mesh_dim=2");
  TENRYU_ASSERT(in.nr > 0 && in.nz > 0, "imc_transport_2d requires positive nr/nz");
  TENRYU_ASSERT(in.n_cells >= 0, "imc_transport_2d requires n_cells >= 0");
  TENRYU_ASSERT(in.n_groups >= 1, "imc_transport_2d requires n_groups >= 1");
  TENRYU_ASSERT(in.n_imc >= 0, "imc_transport_2d requires n_imc >= 0");
  if (in.ddmc_mode != nullptr) {
    TENRYU_ASSERT(in.n_groups_for_mode >= 1,
                  "imc_transport_2d requires n_groups_for_mode >= 1 when ddmc_mode is set");
    TENRYU_ASSERT(in.sigma_R != nullptr,
                  "imc_transport_2d requires sigma_R when ddmc_mode is set");
    TENRYU_ASSERT(in.fleck_f != nullptr || in.sigma_a != nullptr,
                  "imc_transport_2d requires fleck_f or sigma_a when ddmc_mode is set");
  }

  if (in.n_imc == 0) {
    return;
  }

  TENRYU_ASSERT(in.pool->pos_r != nullptr, "imc_transport_2d requires pool->pos_r");
  TENRYU_ASSERT(in.pool->pos_z != nullptr, "imc_transport_2d requires pool->pos_z");
  TENRYU_ASSERT(in.pool->dir_r != nullptr, "imc_transport_2d requires pool->dir_r");
  TENRYU_ASSERT(in.pool->dir_z != nullptr, "imc_transport_2d requires pool->dir_z");
  TENRYU_ASSERT(in.pool->dir_phi != nullptr, "imc_transport_2d requires pool->dir_phi");
  TENRYU_ASSERT(in.pool->energy != nullptr, "imc_transport_2d requires pool->energy");
  TENRYU_ASSERT(in.pool->birth_energy != nullptr,
                "imc_transport_2d requires pool->birth_energy");
  TENRYU_ASSERT(in.pool->time_remain != nullptr,
                "imc_transport_2d requires pool->time_remain");
  TENRYU_ASSERT(in.pool->global_id != nullptr,
                "imc_transport_2d requires pool->global_id");
  TENRYU_ASSERT(in.pool->rng_counter != nullptr,
                "imc_transport_2d requires pool->rng_counter");
  TENRYU_ASSERT(in.pool->cell_id != nullptr, "imc_transport_2d requires pool->cell_id");
  TENRYU_ASSERT(in.pool->group_id != nullptr,
                "imc_transport_2d requires pool->group_id");
  TENRYU_ASSERT(in.pool->mode != nullptr, "imc_transport_2d requires pool->mode");
  TENRYU_ASSERT(in.pool->alive != nullptr, "imc_transport_2d requires pool->alive");
  TENRYU_ASSERT(in.pool->weight != nullptr, "imc_transport_2d requires pool->weight");

  int* d_counter = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_counter), sizeof(int)),
             "imc_transport_2d cudaMalloc counter failed");
  cuda_check(cudaMemset(d_counter, 0, sizeof(int)),
             "imc_transport_2d cudaMemset counter failed");

  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, 0),
             "imc_transport_2d cudaGetDeviceProperties failed");

  constexpr int kBlock = 128;
  constexpr int kBlocksPerSm = 8;
  const int grid = std::max(1, prop.multiProcessorCount * kBlocksPerSm);
  const RankBoundaryParams2D rank_boundary = make_rank_boundary_params_2d(part);

  imc_transport_2d_kernel<<<grid, kBlock>>>(in.pool->pos_r,
                                            in.pool->pos_z,
                                            in.pool->dir_r,
                                            in.pool->dir_z,
                                            in.pool->dir_phi,
                                            in.pool->energy,
                                            in.pool->birth_energy,
                                            in.pool->time_remain,
                                            in.pool->global_id,
                                            in.pool->rng_counter,
                                            in.pool->cell_id,
                                            in.pool->group_id,
                                            in.pool->mode,
                                            in.pool->alive,
                                            in.sigma_a_eff,
                                            in.sigma_s_eff,
                                            in.Te,
                                            in.vol,
                                            in.node_r,
                                            in.node_z,
                                            in.ddmc_mode,
                                            in.ddmc_zero_flux_interfaces ? 1 : 0,
                                            in.n_groups_for_mode,
                                            in.emissivity_preserving ? 1 : 0,
                                            in.sigma_R,
                                            in.fleck_f,
                                            in.sigma_a,
                                            in.eta_cdf,
                                            in.planck,
                                            in.inelastic_scatter ? 1 : 0,
                                            in.rad_dep,
                                            in.rad_E_tally,
                                            in.E_escape,
                                            in.E_numerical_loss,
                                            in.imc_absorbed,
                                            in.imc_escaped,
                                            in.interface_transitions,
                                            in.interface_reflections,
                                            in.conversion_prob_violations,
                                            in.cnt_boundary,
                                            in.cnt_scatter,
                                            in.cnt_census,
                                            in.cnt_absorb_kill,
                                            in.cnt_absorb_survive,
                                            in.cnt_roulette_kill,
                                            d_counter,
                                            in.nr,
                                            in.nz,
                                            in.n_cells,
                                            in.n_groups,
                                            in.n_imc,
                                            in.dt,
                                            in.E_avg,
                                            in.w_cutoff,
                                            in.p_survival,
                                            in.f_cutoff,
                                            in.tail_pass ? 1 : 0,
                                            in.bc_inner,
                                            in.bc_outer,
                                            in.bc_bottom_z,
                                            in.bc_top_z,
                                            in.step_number,
                                            in.user_seed,
                                            rank_boundary.ghost_layers,
                                            rank_boundary.nr_local,
                                            rank_boundary.nz_local,
                                            rank_boundary.has_left_boundary,
                                            rank_boundary.has_right_boundary,
                                            rank_boundary.has_bottom_boundary,
                                            rank_boundary.has_top_boundary,
                                            in.error_flags);

  cuda_check(cudaGetLastError(), "imc_transport_2d kernel launch failed");
  const int weight_grid = (in.n_imc + kBlock - 1) / kBlock;
  update_imc_weight_2d_kernel<<<weight_grid, kBlock>>>(in.n_imc,
                                                       in.pool->energy,
                                                       in.pool->birth_energy,
                                                       in.pool->weight);
  cuda_check(cudaGetLastError(), "imc_transport_2d weight kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "imc_transport_2d kernel execution failed");
  cuda_check(cudaFree(d_counter), "imc_transport_2d cudaFree counter failed");
}

void imc_transport_2d_persistent_cuda(const TransportInputs& in) {
  imc_transport_2d_persistent_cuda(in, nullptr);
}

void imc_transport_2d_persistent_cuda(const TransportInputs& in,
                                      const tenryu::parallel::PartitionInfo& part) {
  imc_transport_2d_persistent_cuda(in, &part);
}

}  // namespace tenryu::radiation
