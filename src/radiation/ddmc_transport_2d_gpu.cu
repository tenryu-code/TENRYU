#include "radiation/ddmc_transport_2d_gpu.cuh"

#include <cmath>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "core/constants.hpp"
#include "core/error.hpp"
#include "radiation/ddmc_coefficients.hpp"
#include "radiation/ddmc_event.hpp"
#include "radiation/face_geometry_2d.cuh"

namespace tenryu::radiation {
namespace {

constexpr double kSigmaTotFloor = 1.0e-30;
constexpr double kLeakTol = 1.0e-14;
constexpr double kRngEps = 1.0e-16;  // Clamp for uniform RNG to avoid log(0) and log(1)
constexpr double kTwoPi = 6.28318530717958647692;
constexpr double kGeomEps = 1.0e-12;
constexpr int kMaxEventsDdmc = 100000;

constexpr std::uint8_t kBcInternal =
    static_cast<std::uint8_t>(DDMCBoundaryType::Internal);
constexpr std::uint8_t kBcVacuum =
    static_cast<std::uint8_t>(DDMCBoundaryType::Vacuum);
constexpr std::uint8_t kBcReflective =
    static_cast<std::uint8_t>(DDMCBoundaryType::Reflective);
constexpr std::uint8_t kBcInterface =
    static_cast<std::uint8_t>(DDMCBoundaryType::Interface);
constexpr std::uint8_t kInterfaceExitCosine = 0U;
constexpr std::uint8_t kInterfaceExitHalfIsotropic = 1U;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
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

__device__ inline double rng_uniform(curandStatePhilox4_32_10_t* rng,
                                     std::uint32_t* counter) {
  *counter += 1U;
  const double xi = curand_uniform_double(rng);
  return fmin(fmax(xi, kRngEps), 1.0 - kRngEps);
}

__device__ inline double sample_ddmc_event_time_device(const double sigma_tot,
                                                       const double xi) {
  if (sigma_tot <= 0.0) {
    return 1.0e300;  // Infinity surrogate when sigma_tot <= 0 (matches CPU ddmc_event.hpp)
  }
  return -log(fmin(fmax(xi, kRngEps), 1.0 - kRngEps)) /
         (core::constants::c_light * sigma_tot);
}

__device__ inline void sample_isotropic_direction_2d_device(
    curandStatePhilox4_32_10_t* rng,
    std::uint32_t* counter,
    double* dir_r,
    double* dir_z,
    double* dir_phi) {
  const double mu_z = 2.0 * rng_uniform(rng, counter) - 1.0;
  const double phi = kTwoPi * rng_uniform(rng, counter);
  const double sin_theta = sqrt(fmax(0.0, 1.0 - mu_z * mu_z));
  *dir_r = sin_theta * cos(phi);
  *dir_z = mu_z;
  *dir_phi = sin_theta * sin(phi);
}

__device__ inline void bilinear_map_device(const double eta,
                                           const double zeta,
                                           const double r00,
                                           const double z00,
                                           const double r10,
                                           const double z10,
                                           const double r11,
                                           const double z11,
                                           const double r01,
                                           const double z01,
                                           double* r_out,
                                           double* z_out) {
  const double n00 = (1.0 - eta) * (1.0 - zeta);
  const double n10 = eta * (1.0 - zeta);
  const double n11 = eta * zeta;
  const double n01 = (1.0 - eta) * zeta;
  *r_out = n00 * r00 + n10 * r10 + n11 * r11 + n01 * r01;
  *z_out = n00 * z00 + n10 * z10 + n11 * z11 + n01 * z01;
}

__device__ inline int node_index_2d_rz_device(const int i, const int j, const int nz) {
  return i * (nz + 1) + j;
}

__device__ inline int neighbor_from_face_device(const int cell,
                                                const int face,
                                                const int nr,
                                                const int nz) {
  const int i = cell / nz;
  const int j = cell - i * nz;
  if (face == 0) {
    return (i > 0) ? ((i - 1) * nz + j) : -1;
  }
  if (face == 1) {
    return (i + 1 < nr) ? ((i + 1) * nz + j) : -1;
  }
  if (face == 2) {
    return (j > 0) ? (i * nz + (j - 1)) : -1;
  }
  if (face == 3) {
    return (j + 1 < nz) ? (i * nz + (j + 1)) : -1;
  }
  return -1;
}

struct CellVertices2DDevice {
  double r00 = 0.0;
  double z00 = 0.0;
  double r10 = 0.0;
  double z10 = 0.0;
  double r11 = 0.0;
  double z11 = 0.0;
  double r01 = 0.0;
  double z01 = 0.0;
};

__device__ inline CellVertices2DDevice get_cell_vertices_2d_device(
    const int cell,
    const int nz,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z) {
  const int i = cell / nz;
  const int j = cell - i * nz;
  const int n00 = node_index_2d_rz_device(i, j, nz);
  const int n10 = node_index_2d_rz_device(i + 1, j, nz);
  const int n11 = node_index_2d_rz_device(i + 1, j + 1, nz);
  const int n01 = node_index_2d_rz_device(i, j + 1, nz);

  CellVertices2DDevice vertices{};
  vertices.r00 = node_r[n00];
  vertices.z00 = node_z[n00];
  vertices.r10 = node_r[n10];
  vertices.z10 = node_z[n10];
  vertices.r11 = node_r[n11];
  vertices.z11 = node_z[n11];
  vertices.r01 = node_r[n01];
  vertices.z01 = node_z[n01];
  return vertices;
}

__device__ inline void cell_center_2d_device(const CellVertices2DDevice& vertices,
                                             double* const r,
                                             double* const z) {
  *r = 0.25 * (vertices.r00 + vertices.r10 + vertices.r11 + vertices.r01);
  *z = 0.25 * (vertices.z00 + vertices.z10 + vertices.z11 + vertices.z01);
}

__device__ inline void sample_volume_uniform_position_2d_device(
    const CellVertices2DDevice& vertices,
    curandStatePhilox4_32_10_t* rng,
    std::uint32_t* counter,
    double* pos_r,
    double* pos_z) {
  const double r_max_cell = fmax(fmax(vertices.r00, vertices.r10), fmax(vertices.r11, vertices.r01));
  double r_p = 0.25 * (vertices.r00 + vertices.r10 + vertices.r11 + vertices.r01);
  double z_p = 0.25 * (vertices.z00 + vertices.z10 + vertices.z11 + vertices.z01);
  bool accepted = false;
  for (int n_try = 0; n_try < 64; ++n_try) {
    const double xi_eta = rng_uniform(rng, counter);
    const double xi_zeta = rng_uniform(rng, counter);
    const double xi_reject = rng_uniform(rng, counter);
    bilinear_map_device(xi_eta,
                        xi_zeta,
                        vertices.r00,
                        vertices.z00,
                        vertices.r10,
                        vertices.z10,
                        vertices.r11,
                        vertices.z11,
                        vertices.r01,
                        vertices.z01,
                        &r_p,
                        &z_p);
    const double w = (r_max_cell > 0.0) ? fmin(fmax(r_p / r_max_cell, 0.0), 1.0) : 1.0;
    if (xi_reject <= w) {
      accepted = true;
      break;
    }
  }
  if (!accepted) {
    r_p = fmax(r_p, 0.0);
  }
  *pos_r = fmax(r_p, 0.0);
  *pos_z = z_p;
}

__device__ inline double normal_sign_toward_point_device(const FaceGeom2D& geom,
                                                         const double target_r,
                                                         const double target_z) {
  const double z_mid = 0.5 * (geom.z1 + geom.z2);
  const double dot =
      (target_r - geom.r_mid) * geom.nr + (target_z - z_mid) * geom.nz;
  return (dot >= 0.0) ? 1.0 : -1.0;
}

__device__ inline double sample_face_param_r_weighted_device(const FaceGeom2D& geom,
                                                             const double xi) {
  const double R1 = geom.r1;
  const double R2 = geom.r2;
  const double eps_R = 1.0e-10 * fmax(fmax(fabs(R1), fabs(R2)), 1.0e-20);
  const double xi_clamped = fmin(fmax(xi, 0.0), 1.0);
  if (fabs(R2 - R1) < eps_R) {
    return xi_clamped;
  }

  const double R1_sq = R1 * R1;
  const double R2_sq = R2 * R2;
  const double term = fmax(R1_sq + xi_clamped * (R2_sq - R1_sq), 0.0);
  const double t = (-R1 + sqrt(term)) / (R2 - R1);
  return fmin(fmax(t, 0.0), 1.0);
}

__device__ inline void sample_ddmc_to_imc_direction_device(
    const double xi_mu,
    const double xi_phi,
    const double nr,
    const double nz,
    const double tr,
    const double tz,
    const bool interface_exit_half_isotropic,
    double* dir_r,
    double* dir_z,
    double* dir_phi) {
  const double mu = interface_exit_half_isotropic ? fmin(fmax(xi_mu, 0.0), 1.0)
                                                  : sqrt(fmax(xi_mu, 0.0));
  const double phi = kTwoPi * xi_phi;
  const double sin_theta = sqrt(fmax(0.0, 1.0 - mu * mu));
  const double tangent = sin_theta * cos(phi);
  *dir_r = mu * nr + tangent * tr;
  *dir_z = mu * nz + tangent * tz;
  *dir_phi = sin_theta * sin(phi);
}

__global__ __launch_bounds__(128, 8) void ddmc_event_loop_2d(
    double* __restrict__ pos_r_arr,
    double* __restrict__ pos_z_arr,
    double* __restrict__ dir_r_arr,
    double* __restrict__ dir_z_arr,
    double* __restrict__ dir_phi_arr,
    double* __restrict__ energy_arr,
    double* __restrict__ time_remain_arr,
    const std::int8_t* __restrict__ sign_arr,
    const std::uint64_t* __restrict__ global_id_arr,
    std::uint32_t* __restrict__ rng_counter_arr,
    std::int32_t* __restrict__ cell_id_arr,
    std::uint16_t* __restrict__ group_id_arr,
    std::uint8_t* __restrict__ mode_arr,
    std::uint8_t* __restrict__ alive_arr,
    const double* __restrict__ sigma_a_eff,
    const double* __restrict__ sigma_s_eff,
    const double* __restrict__ sigma_leak_face,
    const std::uint8_t* __restrict__ bc_face,
    const int* __restrict__ neighbor_face,
    const double* __restrict__ eta_cdf,
    const TransportMode* __restrict__ ddmc_mode,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    double* __restrict__ rad_dep,
    double* __restrict__ rad_E_tally,
    double* __restrict__ E_escape,
    double* __restrict__ E_numerical_loss,
    unsigned long long* __restrict__ ddmc_absorbed,
    unsigned long long* __restrict__ ddmc_census,
    unsigned long long* __restrict__ ddmc_leak_face0,
    unsigned long long* __restrict__ ddmc_leak_face1,
    unsigned long long* __restrict__ ddmc_leak_face2,
    unsigned long long* __restrict__ ddmc_leak_face3,
    unsigned long long* __restrict__ ddmc_leak_boundary,
    unsigned long long* __restrict__ ddmc_converted_to_imc,
    unsigned long long* __restrict__ ddmc_sigma_tot_zero,
    unsigned long long* __restrict__ ddmc_max_events_reached,
    const int n_cells,
    const int n_groups,
    const int nr,
    const int nz,
    const int ghost_layers,
    const int nr_local,
    const int nz_local,
    const bool has_r_inner_boundary,
    const bool has_r_outer_boundary,
    const bool has_z_bottom_boundary,
    const bool has_z_top_boundary,
    const int n_ddmc,
    const int ddmc_start,
    const int pool_capacity,
    const std::uint8_t interface_exit_distribution,
    const double dt,
    const std::uint64_t step_number,
    const std::uint64_t user_seed,
    core::DeviceErrorFlags* __restrict__ error_flags) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_ddmc) {
    return;
  }

  const int p = ddmc_start + tid;
  if (p < 0 || p >= pool_capacity) {
    return;
  }
  std::uint8_t alive_l = alive_arr[p];
  std::uint8_t mode_l = mode_arr[p];
  if (alive_l != kAlive || mode_l != kModeDDMC) {
    return;
  }

  std::int32_t cell_l = cell_id_arr[p];
  int group_l = static_cast<int>(group_id_arr[p]);
  double E_l = energy_arr[p];
  double t_remain_l = time_remain_arr[p];
  const std::int8_t sign_l = sign_arr[p];
  const double sign_d = static_cast<double>(sign_l);
  if (!isfinite(E_l) || !isfinite(t_remain_l)) {
    if (error_flags != nullptr) {
      atomicExch(&error_flags->nan_particle, 1);
    }
    if (isfinite(E_l) && E_l > 0.0 && E_numerical_loss != nullptr) {
      atomic_add_double(E_numerical_loss, sign_d * E_l);
    }
    energy_arr[p] = 0.0;
    time_remain_arr[p] = 0.0;
    alive_arr[p] = kDead;
    return;
  }
  if (E_l <= 0.0) {
    if (E_l < 0.0 && E_numerical_loss != nullptr && isfinite(E_l)) {
      atomic_add_double(E_numerical_loss, sign_d * (-E_l));
    }
    energy_arr[p] = 0.0;
    time_remain_arr[p] = 0.0;
    alive_arr[p] = kDead;
    return;
  }
  if (t_remain_l <= 0.0) {
    t_remain_l = dt;
  }

  std::uint32_t rng_counter_l = rng_counter_arr[p];
  curandStatePhilox4_32_10_t rng;
  curand_init(global_id_arr[p] ^ user_seed,
              static_cast<unsigned long long>(step_number),
              static_cast<unsigned long long>(rng_counter_l),
              &rng);

  int events = 0;
  bool preserved_census = false;
  while (alive_l == kAlive && mode_l == kModeDDMC &&
         t_remain_l > 0.0 && events < kMaxEventsDdmc) {
    ++events;

    if (!isfinite(E_l) || !isfinite(t_remain_l)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->nan_particle, 1);
      }
      if (isfinite(E_l) && E_l > 0.0 && E_numerical_loss != nullptr) {
        atomic_add_double(E_numerical_loss, sign_d * E_l);
      }
      E_l = 0.0;
      t_remain_l = 0.0;
      alive_l = kDead;
      break;
    }
    if (E_l <= 0.0) {
      if (E_l < 0.0 && E_numerical_loss != nullptr && isfinite(E_l)) {
        atomic_add_double(E_numerical_loss, sign_d * (-E_l));
      }
      E_l = 0.0;
      t_remain_l = 0.0;
      alive_l = kDead;
      break;
    }

    if (cell_l < 0 || cell_l >= n_cells ||
        group_l < 0 || group_l >= n_groups) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      if (E_l > 0.0 && E_numerical_loss != nullptr) {
        atomic_add_double(E_numerical_loss, sign_d * E_l);
      }
      E_l = 0.0;
      alive_l = kDead;
      break;
    }

    const int key = cell_l * n_groups + group_l;
    const int face_group_base = cell_l * 4 * n_groups + group_l;
    const int neighbor_base = cell_l * 4;
    const double sigma_a = fmax(sigma_a_eff[key], 0.0);
    const double sigma_s =
        (sigma_s_eff != nullptr && eta_cdf != nullptr) ? fmax(sigma_s_eff[key], 0.0) : 0.0;
    double sigma_tot = sigma_a + sigma_s;
    double sigma_leak_face_pos[4] = {0.0, 0.0, 0.0, 0.0};
    bool invalid_leak = false;
    for (int face = 0; face < 4; ++face) {
      const double sigma_leak_raw = sigma_leak_face[face_group_base + face * n_groups];
      if (sigma_leak_raw < -kLeakTol) {
        invalid_leak = true;
        break;
      }
      sigma_leak_face_pos[face] = fmax(sigma_leak_raw, 0.0);
      sigma_tot += sigma_leak_face_pos[face];
    }
    if (invalid_leak) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_boundary, 1);
      }
      atomic_add_double(&rad_dep[key], sign_d * E_l);
      E_l = 0.0;
      alive_l = kDead;
      atomic_inc_counter(ddmc_absorbed);
      break;
    }

    if (sigma_tot <= kSigmaTotFloor) {
      atomic_add_double(&rad_E_tally[key],
                        sign_d * core::constants::c_light * E_l * t_remain_l);
      t_remain_l = 0.0;
      atomic_inc_counter(ddmc_sigma_tot_zero);
      if (error_flags != nullptr) {
        atomicExch(&error_flags->ddmc_sigma_tot_zero, 1);
      }
      break;
    }

    const double dt_evt =
        sample_ddmc_event_time_device(sigma_tot, rng_uniform(&rng, &rng_counter_l));
    const double dt_res = fmin(dt_evt, t_remain_l);
    atomic_add_double(&rad_E_tally[key],
                      sign_d * core::constants::c_light * E_l * dt_res);

    if (dt_evt >= t_remain_l) {
      t_remain_l = 0.0;
      atomic_inc_counter(ddmc_census);
      preserved_census = true;
      break;
    }

    t_remain_l -= dt_evt;

    const double threshold = rng_uniform(&rng, &rng_counter_l) * sigma_tot;
    if (threshold < sigma_a) {
      atomic_add_double(&rad_dep[key], sign_d * E_l);
      E_l = 0.0;
      alive_l = kDead;
      atomic_inc_counter(ddmc_absorbed);
      break;
    }
    if (sigma_s > 0.0 && threshold < sigma_a + sigma_s) {
      const int g_new = sample_group_from_cdf(
          eta_cdf + cell_l * n_groups,
          n_groups,
          rng_uniform(&rng, &rng_counter_l));
      if (ddmc_mode != nullptr &&
          ddmc_mode[cell_l * n_groups + g_new] != TransportMode::DDMC) {
        const CellVertices2DDevice sample_vertices =
            get_cell_vertices_2d_device(cell_l, nz, node_r, node_z);
        mode_l = kModeIMC;
        group_l = static_cast<std::uint16_t>(g_new);
        atomic_inc_counter(ddmc_converted_to_imc);
        sample_volume_uniform_position_2d_device(
            sample_vertices, &rng, &rng_counter_l, &pos_r_arr[p], &pos_z_arr[p]);
        sample_isotropic_direction_2d_device(
            &rng, &rng_counter_l, &dir_r_arr[p], &dir_z_arr[p], &dir_phi_arr[p]);
        break;
      }
      group_l = static_cast<std::uint16_t>(g_new);
      continue;
    }
    double cdf = sigma_a + sigma_s;
    int selected_face = -1;
    for (int face = 0; face < 4; ++face) {
      cdf += sigma_leak_face_pos[face];
      if (threshold <= cdf + 1.0e-15) {  // Rounding tolerance for CDF
        selected_face = face;
        break;
      }
    }

    if (selected_face < 0) {
      atomic_add_double(&rad_dep[key], sign_d * E_l);
      E_l = 0.0;
      alive_l = kDead;
      atomic_inc_counter(ddmc_absorbed);
      break;
    }

    if (selected_face == 0) {
      atomic_inc_counter(ddmc_leak_face0);
    } else if (selected_face == 1) {
      atomic_inc_counter(ddmc_leak_face1);
    } else if (selected_face == 2) {
      atomic_inc_counter(ddmc_leak_face2);
    } else {
      atomic_inc_counter(ddmc_leak_face3);
    }

    const std::uint8_t bc = bc_face[face_group_base + selected_face * n_groups];
    if (bc == kBcVacuum) {
      if (group_l >= 0 && group_l < n_groups && E_l > 0.0) {
        atomic_add_double(&E_escape[group_l], sign_d * E_l);
      }
      E_l = 0.0;
      alive_l = kDead;
      atomic_inc_counter(ddmc_leak_boundary);
      break;
    }

    if (bc == kBcReflective) {
      const CellVertices2DDevice reflect_vertices =
          get_cell_vertices_2d_device(cell_l, nz, node_r, node_z);
      const FaceGeom2D reflect_geom =
          compute_face_geom(selected_face,
                            reflect_vertices.r00,
                            reflect_vertices.z00,
                            reflect_vertices.r10,
                            reflect_vertices.z10,
                            reflect_vertices.r11,
                            reflect_vertices.z11,
                            reflect_vertices.r01,
                            reflect_vertices.z01);
      if (!(reflect_geom.length > 0.0)) {
        if (error_flags != nullptr) {
          atomicExch(&error_flags->invalid_boundary, 1);
        }
        atomic_add_double(&rad_dep[key], sign_d * E_l);
        E_l = 0.0;
        alive_l = kDead;
        atomic_inc_counter(ddmc_absorbed);
        break;
      }
      reflect_direction_2d(dir_r_arr[p], dir_z_arr[p], reflect_geom.nr, reflect_geom.nz);
      continue;
    }

    int next = neighbor_face[neighbor_base + selected_face];
    if (next < 0 || next >= n_cells) {
      next = neighbor_from_face_device(cell_l, selected_face, nr, nz);
    }
    if (next < 0 || next >= n_cells) {
      if (group_l >= 0 && group_l < n_groups && E_l > 0.0) {
        atomic_add_double(&E_escape[group_l], sign_d * E_l);
      }
      E_l = 0.0;
      alive_l = kDead;
      atomic_inc_counter(ddmc_leak_boundary);
      break;
    }

    if (bc == kBcInternal) {
      cell_l = next;
      if (ghost_layers > 0 && nr_local > 0 && nz_local > 0) {
        const int i_new = cell_l / nz;
        const int j_new = cell_l - i_new * nz;
        bool enters_ghost = false;
        bool has_rank_boundary = false;
        if (selected_face == 0) {
          enters_ghost = (i_new < ghost_layers);
          has_rank_boundary = has_r_inner_boundary;
        } else if (selected_face == 1) {
          enters_ghost = (i_new >= (ghost_layers + nr_local));
          has_rank_boundary = has_r_outer_boundary;
        } else if (selected_face == 2) {
          enters_ghost = (j_new < ghost_layers);
          has_rank_boundary = has_z_bottom_boundary;
        } else if (selected_face == 3) {
          enters_ghost = (j_new >= (ghost_layers + nz_local));
          has_rank_boundary = has_z_top_boundary;
        }
        if (enters_ghost) {
          if (has_rank_boundary) {
            // Safety: physical boundary reached via internal BC - should be
            // unreachable (sigma_leak=0 at reflective). Absorb as fallback.
            atomic_add_double(&rad_dep[key], sign_d * E_l);
            E_l = 0.0;
            alive_l = kDead;
            atomic_inc_counter(ddmc_absorbed);
            break;
          }
          cell_l = -(100 + selected_face);
          alive_l = kAlive;
          break;
        }
      }
      continue;
    }

    if (bc != kBcInterface) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_boundary, 1);
      }
      atomic_add_double(&rad_dep[key], sign_d * E_l);
      E_l = 0.0;
      alive_l = kDead;
      atomic_inc_counter(ddmc_absorbed);
      break;
    }

    const CellVertices2DDevice leak_vertices =
        get_cell_vertices_2d_device(cell_l, nz, node_r, node_z);
    const FaceGeom2D leak_geom =
        compute_face_geom(selected_face,
                          leak_vertices.r00,
                          leak_vertices.z00,
                          leak_vertices.r10,
                          leak_vertices.z10,
                          leak_vertices.r11,
                          leak_vertices.z11,
                          leak_vertices.r01,
                          leak_vertices.z01);
    if (!(leak_geom.length > 0.0)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_boundary, 1);
      }
      atomic_add_double(&rad_dep[key], sign_d * E_l);
      E_l = 0.0;
      alive_l = kDead;
      atomic_inc_counter(ddmc_absorbed);
      break;
    }

    mode_l = kModeIMC;
    atomic_inc_counter(ddmc_converted_to_imc);

    const CellVertices2DDevice imc_vertices =
        get_cell_vertices_2d_device(next, nz, node_r, node_z);
    double imc_r_center = 0.0;
    double imc_z_center = 0.0;
    cell_center_2d_device(imc_vertices, &imc_r_center, &imc_z_center);
    const double normal_sign =
        normal_sign_toward_point_device(leak_geom, imc_r_center, imc_z_center);
    const double nr_to_imc = normal_sign * leak_geom.nr;
    const double nz_to_imc = normal_sign * leak_geom.nz;

    const double t_face = sample_face_param_r_weighted_device(
        leak_geom, rng_uniform(&rng, &rng_counter_l));
    pos_r_arr[p] = leak_geom.r1 + t_face * (leak_geom.r2 - leak_geom.r1);
    pos_z_arr[p] = leak_geom.z1 + t_face * (leak_geom.z2 - leak_geom.z1);
    push_off_face_2d(pos_r_arr[p], pos_z_arr[p], nr_to_imc, nz_to_imc, kGeomEps);
    pos_r_arr[p] = fmax(pos_r_arr[p], 0.0);

    sample_ddmc_to_imc_direction_device(
        rng_uniform(&rng, &rng_counter_l),
        rng_uniform(&rng, &rng_counter_l),
        nr_to_imc,
        nz_to_imc,
        leak_geom.tr,
        leak_geom.tz,
        interface_exit_distribution == kInterfaceExitHalfIsotropic,
        &dir_r_arr[p],
        &dir_z_arr[p],
        &dir_phi_arr[p]);
    cell_l = next;
    break;
  }

  if (events >= kMaxEventsDdmc && alive_l == kAlive &&
      mode_l == kModeDDMC && !preserved_census) {
    if (cell_l >= 0 && cell_l < n_cells &&
        group_l >= 0 && group_l < n_groups) {
      const int key = cell_l * n_groups + group_l;
      atomic_add_double(&rad_dep[key], sign_d * E_l);
    } else if (E_l > 0.0 && E_numerical_loss != nullptr) {
      atomic_add_double(E_numerical_loss, sign_d * E_l);
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
    }
    E_l = 0.0;
    alive_l = kDead;
    atomic_inc_counter(ddmc_max_events_reached);
    if (error_flags != nullptr) {
      atomicAdd(&error_flags->infinite_loop, 1);
    }
  }

  energy_arr[p] = E_l;
  time_remain_arr[p] = fmax(t_remain_l, 0.0);
  cell_id_arr[p] = cell_l;
  group_id_arr[p] = static_cast<std::uint16_t>(group_l);
  mode_arr[p] = mode_l;
  alive_arr[p] = alive_l;
  rng_counter_arr[p] = rng_counter_l;
}

}  // namespace

void ddmc_transport_2d_gpu_cuda(const DDMCTransport2DGPUInputs& in) {
  TENRYU_ASSERT(in.pool != nullptr, "ddmc_transport_2d_gpu requires pool");
  TENRYU_ASSERT(in.sigma_a_eff != nullptr, "ddmc_transport_2d_gpu requires sigma_a_eff");
  TENRYU_ASSERT(in.sigma_leak_face != nullptr,
                "ddmc_transport_2d_gpu requires sigma_leak_face");
  TENRYU_ASSERT(in.bc_face != nullptr, "ddmc_transport_2d_gpu requires bc_face");
  TENRYU_ASSERT(in.neighbor_face != nullptr,
                "ddmc_transport_2d_gpu requires neighbor_face");
  TENRYU_ASSERT(in.node_r != nullptr, "ddmc_transport_2d_gpu requires node_r");
  TENRYU_ASSERT(in.node_z != nullptr, "ddmc_transport_2d_gpu requires node_z");
  TENRYU_ASSERT(in.rad_dep != nullptr, "ddmc_transport_2d_gpu requires rad_dep");
  TENRYU_ASSERT(in.rad_E_tally != nullptr, "ddmc_transport_2d_gpu requires rad_E_tally");
  TENRYU_ASSERT(in.E_escape != nullptr, "ddmc_transport_2d_gpu requires E_escape");
  TENRYU_ASSERT(in.E_numerical_loss != nullptr,
                "ddmc_transport_2d_gpu requires E_numerical_loss");
  TENRYU_ASSERT(in.n_cells >= 0, "ddmc_transport_2d_gpu requires n_cells >= 0");
  TENRYU_ASSERT(in.n_groups >= 1, "ddmc_transport_2d_gpu requires n_groups >= 1");
  TENRYU_ASSERT(in.nr >= 1, "ddmc_transport_2d_gpu requires nr >= 1");
  TENRYU_ASSERT(in.nz >= 1, "ddmc_transport_2d_gpu requires nz >= 1");
  TENRYU_ASSERT(in.n_ddmc >= 0, "ddmc_transport_2d_gpu requires n_ddmc >= 0");
  TENRYU_ASSERT(in.ddmc_start >= 0, "ddmc_transport_2d_gpu requires ddmc_start >= 0");
  TENRYU_ASSERT(in.interface_exit_distribution == kInterfaceExitCosine ||
                    in.interface_exit_distribution == kInterfaceExitHalfIsotropic,
                "ddmc_transport_2d_gpu requires interface_exit_distribution in {0,1}");
  const auto ddmc_end =
      static_cast<long long>(in.ddmc_start) + static_cast<long long>(in.n_ddmc);
  TENRYU_ASSERT(ddmc_end <= static_cast<long long>(in.pool->capacity),
                "DDMC2D slice out of bounds: start=" + std::to_string(in.ddmc_start) +
                    " n=" + std::to_string(in.n_ddmc) +
                    " capacity=" + std::to_string(in.pool->capacity));

  if (in.n_ddmc == 0) {
    return;
  }

  TENRYU_ASSERT(in.pool->pos_r != nullptr, "ddmc_transport_2d_gpu requires pool->pos_r");
  TENRYU_ASSERT(in.pool->pos_z != nullptr, "ddmc_transport_2d_gpu requires pool->pos_z");
  TENRYU_ASSERT(in.pool->dir_r != nullptr, "ddmc_transport_2d_gpu requires pool->dir_r");
  TENRYU_ASSERT(in.pool->dir_z != nullptr, "ddmc_transport_2d_gpu requires pool->dir_z");
  TENRYU_ASSERT(in.pool->dir_phi != nullptr, "ddmc_transport_2d_gpu requires pool->dir_phi");
  TENRYU_ASSERT(in.pool->energy != nullptr, "ddmc_transport_2d_gpu requires pool->energy");
  TENRYU_ASSERT(in.pool->time_remain != nullptr,
                "ddmc_transport_2d_gpu requires pool->time_remain");
  TENRYU_ASSERT(in.pool->sign != nullptr, "ddmc_transport_2d_gpu requires pool->sign");
  TENRYU_ASSERT(in.pool->global_id != nullptr,
                "ddmc_transport_2d_gpu requires pool->global_id");
  TENRYU_ASSERT(in.pool->rng_counter != nullptr,
                "ddmc_transport_2d_gpu requires pool->rng_counter");
  TENRYU_ASSERT(in.pool->cell_id != nullptr, "ddmc_transport_2d_gpu requires pool->cell_id");
  TENRYU_ASSERT(in.pool->group_id != nullptr,
                "ddmc_transport_2d_gpu requires pool->group_id");
  TENRYU_ASSERT(in.pool->mode != nullptr, "ddmc_transport_2d_gpu requires pool->mode");
  TENRYU_ASSERT(in.pool->alive != nullptr, "ddmc_transport_2d_gpu requires pool->alive");

  constexpr int kBlock = 128;
  const int grid = (in.n_ddmc + kBlock - 1) / kBlock;

  ddmc_event_loop_2d<<<grid, kBlock>>>(in.pool->pos_r,
                                       in.pool->pos_z,
                                       in.pool->dir_r,
                                       in.pool->dir_z,
                                       in.pool->dir_phi,
                                       in.pool->energy,
                                       in.pool->time_remain,
                                       in.pool->sign,
                                       in.pool->global_id,
                                       in.pool->rng_counter,
                                       in.pool->cell_id,
                                       in.pool->group_id,
                                       in.pool->mode,
                                       in.pool->alive,
                                       in.sigma_a_eff,
                                       in.sigma_s_eff,
                                       in.sigma_leak_face,
      in.bc_face,
      in.neighbor_face,
      in.eta_cdf,
      in.ddmc_mode,
      in.node_r,
                                       in.node_z,
                                       in.rad_dep,
                                       in.rad_E_tally,
                                       in.E_escape,
                                       in.E_numerical_loss,
                                       in.ddmc_absorbed,
                                       in.ddmc_census,
                                       in.ddmc_leak_face0,
                                       in.ddmc_leak_face1,
                                       in.ddmc_leak_face2,
                                       in.ddmc_leak_face3,
                                       in.ddmc_leak_boundary,
                                       in.ddmc_converted_to_imc,
                                       in.ddmc_sigma_tot_zero,
                                       in.ddmc_max_events_reached,
                                       in.n_cells,
                                       in.n_groups,
                                       in.nr,
                                       in.nz,
                                       in.ghost_layers,
                                       in.nr_local,
                                       in.nz_local,
                                       in.has_r_inner_boundary,
                                       in.has_r_outer_boundary,
                                       in.has_z_bottom_boundary,
                                       in.has_z_top_boundary,
                                       in.n_ddmc,
                                       in.ddmc_start,
                                       in.pool->capacity,
                                       in.interface_exit_distribution,
                                       in.dt,
                                       in.step_number,
                                       in.user_seed,
                                       in.error_flags);

  cuda_check(cudaGetLastError(), "ddmc_transport_2d_gpu kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "ddmc_transport_2d_gpu kernel execution failed");
}

}  // namespace tenryu::radiation
