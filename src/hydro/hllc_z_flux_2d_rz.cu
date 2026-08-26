#include "hydro/hllc_z_flux_2d_rz.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/kernel_guard.hpp"
#include "hydro/anti_hourglass.cuh"
#include "hydro/boundary_2d.hpp"
#include "hydro/rz_corner_kinetic.cuh"
#include "materials/eos_device.cuh"

namespace tenryu::hydro {
namespace {

constexpr double kRhoTiny = 1.0e-300;
constexpr double kPressureTiny = 1.0e-300;

inline void cuda_check_local(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, std::string(message) + ": " + cudaGetErrorString(err));
}

inline void sync_kernel_local(const char* message) {
  cuda_check_local(cudaGetLastError(), message);
  if (core::sync_every_kernel_enabled()) {
    cuda_check_local(cudaDeviceSynchronize(), message);
  }
}

void warn_hllc_invariant_corner_mass_once() {
  static bool warned = false;
  if (warned) {
    return;
  }
  warned = true;
  core::log_warning(
      "HLLC z flux is preserving Lagrangian-invariant corner_mass; "
      "conservative subzonal mass flux is deferred to Stage F");
}

__device__ inline double atomic_add_double_local(double* address, const double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed = 0ULL;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif
}

struct Primitive {
  double rho = 0.0;
  double u = 0.0;
  double p = 0.0;
  double E = 0.0;
  double y = 0.5;
};

struct HllcFlux {
  double mass = 0.0;
  double mom = 0.0;
  double energy = 0.0;
  double ye = 0.0;
};

struct HllcDeviceStats {
  unsigned long long hlle_fallback_count = 0ULL;
  unsigned long long invalid_state_count = 0ULL;
  double min_star_rho = 1.0e300;
  double min_star_p = 1.0e300;
};

__device__ inline void atomic_min_double(double* addr, const double value) {
  auto* addr_ull = reinterpret_cast<unsigned long long int*>(addr);
  unsigned long long int old = *addr_ull;
  unsigned long long int assumed = 0ULL;
  while (value < __longlong_as_double(static_cast<long long>(old))) {
    assumed = old;
    old = atomicCAS(addr_ull, assumed, __double_as_longlong(value));
    if (assumed == old) {
      break;
    }
  }
}

__device__ inline HllcFlux physical_flux(const Primitive& q) {
  HllcFlux f{};
  f.mass = q.rho * q.u;
  f.mom = q.rho * q.u * q.u + q.p;
  f.energy = q.u * (q.E + q.p);
  f.ye = f.mass * q.y;
  return f;
}

__device__ inline bool valid_primitive(const Primitive& q) {
  return isfinite(q.rho) && isfinite(q.u) && isfinite(q.p) && isfinite(q.E) &&
         q.rho > 0.0 && q.p > 0.0 && q.E > 0.0;
}

__device__ inline HllcFlux hlle_flux(const Primitive& L,
                                     const Primitive& R,
                                     const double gamma) {
  const double cL = sqrt(fmax(gamma * L.p / fmax(L.rho, kRhoTiny), 0.0));
  const double cR = sqrt(fmax(gamma * R.p / fmax(R.rho, kRhoTiny), 0.0));
  const double SL = fmin(L.u - cL, R.u - cR);
  const double SR = fmax(L.u + cL, R.u + cR);
  const HllcFlux FL = physical_flux(L);
  const HllcFlux FR = physical_flux(R);
  if (SL >= 0.0) {
    return FL;
  }
  if (SR <= 0.0) {
    return FR;
  }
  const double inv = 1.0 / fmax(SR - SL, 1.0e-300);
  HllcFlux out{};
  out.mass = (SR * FL.mass - SL * FR.mass + SL * SR * (R.rho - L.rho)) * inv;
  out.mom = (SR * FL.mom - SL * FR.mom +
             SL * SR * (R.rho * R.u - L.rho * L.u)) *
            inv;
  out.energy = (SR * FL.energy - SL * FR.energy + SL * SR * (R.E - L.E)) * inv;
  out.ye = (SR * FL.ye - SL * FR.ye +
            SL * SR * (R.rho * R.y - L.rho * L.y)) *
           inv;
  return out;
}

__device__ inline HllcFlux hllc_flux(const Primitive& L,
                                     const Primitive& R,
                                     const double gamma,
                                     const bool allow_hlle,
                                     HllcDeviceStats* stats) {
  if (!valid_primitive(L) || !valid_primitive(R)) {
    if (stats != nullptr) {
      atomicAdd(&stats->invalid_state_count, 1ULL);
      atomicAdd(&stats->hlle_fallback_count, 1ULL);
    }
    return hlle_flux(L, R, gamma);
  }

  const double cL = sqrt(fmax(gamma * L.p / L.rho, 0.0));
  const double cR = sqrt(fmax(gamma * R.p / R.rho, 0.0));
  const double SL = fmin(L.u - cL, R.u - cR);
  const double SR = fmax(L.u + cL, R.u + cR);
  const HllcFlux FL = physical_flux(L);
  const HllcFlux FR = physical_flux(R);
  if (SL >= 0.0) {
    return FL;
  }
  if (SR <= 0.0) {
    return FR;
  }

  const double denom = L.rho * (SL - L.u) - R.rho * (SR - R.u);
  if (!isfinite(denom) || fabs(denom) <= 1.0e-300) {
    if (stats != nullptr) {
      atomicAdd(&stats->hlle_fallback_count, 1ULL);
    }
    return allow_hlle ? hlle_flux(L, R, gamma) : FL;
  }
  const double SM =
      (R.p - L.p + L.rho * L.u * (SL - L.u) -
       R.rho * R.u * (SR - R.u)) /
      denom;
  const double pStarL = L.p + L.rho * (SL - L.u) * (SM - L.u);
  const double pStarR = R.p + R.rho * (SR - R.u) * (SM - R.u);
  const double pStar = 0.5 * (pStarL + pStarR);
  const double denomL = SL - SM;
  const double denomR = SR - SM;
  const double rhoStarL =
      (fabs(denomL) > 1.0e-300) ? L.rho * (SL - L.u) / denomL : -1.0;
  const double rhoStarR =
      (fabs(denomR) > 1.0e-300) ? R.rho * (SR - R.u) / denomR : -1.0;
  if (stats != nullptr) {
    atomic_min_double(&stats->min_star_rho, fmin(rhoStarL, rhoStarR));
    atomic_min_double(&stats->min_star_p, pStar);
  }
  const bool invalid_star =
      !isfinite(SM) || !isfinite(pStar) || !isfinite(rhoStarL) ||
      !isfinite(rhoStarR) || rhoStarL <= 0.0 || rhoStarR <= 0.0 ||
      pStar <= 0.0;
  if (invalid_star) {
    if (stats != nullptr) {
      atomicAdd(&stats->invalid_state_count, 1ULL);
      atomicAdd(&stats->hlle_fallback_count, 1ULL);
    }
    return allow_hlle ? hlle_flux(L, R, gamma) : FL;
  }

  if (SM >= 0.0) {
    const double inv = 1.0 / denomL;
    const double rhoS = rhoStarL;
    const double ES = ((SL - L.u) * L.E - L.p * L.u + pStar * SM) * inv;
    HllcFlux out = FL;
    out.mass += SL * (rhoS - L.rho);
    out.mom += SL * (rhoS * SM - L.rho * L.u);
    out.energy += SL * (ES - L.E);
    out.ye += SL * (rhoS * L.y - L.rho * L.y);
    return out;
  }

  const double inv = 1.0 / denomR;
  const double rhoS = rhoStarR;
  const double ES = ((SR - R.u) * R.E - R.p * R.u + pStar * SM) * inv;
  HllcFlux out = FR;
  out.mass += SR * (rhoS - R.rho);
  out.mom += SR * (rhoS * SM - R.rho * R.u);
  out.energy += SR * (ES - R.E);
  out.ye += SR * (rhoS * R.y - R.rho * R.y);
  return out;
}

__global__ void init_hllc_mom_from_nodes_kernel(double* __restrict__ mom_z_cell,
                                                const double* __restrict__ mass,
                                                const double* __restrict__ corner_mass,
                                                const double* __restrict__ v_z,
                                                const int nr,
                                                const int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int n00 = rz::node_index(i, j, nz);
  const int n10 = rz::node_index(i + 1, j, nz);
  const int n11 = rz::node_index(i + 1, j + 1, nz);
  const int n01 = rz::node_index(i, j + 1, nz);
  const int nodes[4] = {n00, n10, n11, n01};
  double pz = 0.0;
  for (int q = 0; q < 4; ++q) {
    pz += corner_mass[c * 4 + q] * v_z[nodes[q]];
  }
  if (!(mass[c] > 0.0) || !isfinite(pz)) {
    pz = 0.0;
  }
  mom_z_cell[c] = pz;
}

__global__ void compute_hllc_corner_mass_kernel(double* __restrict__ corner_mass,
                                                const double* __restrict__ mass,
                                                const double* __restrict__ x_r,
                                                const double* __restrict__ x_z,
                                                rz::CornerMassFallbackRecorder*
                                                    fallback_recorder,
                                                const int fallback_stage,
                                                const int corner_mass_convention,
                                                const int nr,
                                                const int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  rz::CornerMassFallbackProbe probe{};
  rz::compute_rz_corner_masses_for_cell(
      c, nz, mass[c], x_r, x_z, nullptr, corner_mass, 4, &probe,
      corner_mass_convention);
  if (probe.fired == 1) {
    rz::record_corner_mass_fallback(
        fallback_recorder, probe, true, c, fallback_stage, -2);
  }
}

__device__ inline Primitive cell_primitive(const int c,
                                           const double* __restrict__ rho,
                                           const double* __restrict__ mass,
                                           const double* __restrict__ vol,
                                           const double* __restrict__ mom_z,
                                           const double* __restrict__ ee,
                                           const double* __restrict__ ei,
                                           const double* __restrict__ Pe,
                                           const double* __restrict__ Pi,
                                           const double* __restrict__ corner_mass,
                                           const double* __restrict__ v_r,
                                           const double* __restrict__ v_z,
                                           const int nr,
                                           const int nz,
                                           const double ye_default) {
  Primitive q{};
  q.rho = fmax(rho[c], kRhoTiny);
  const double m = fmax(mass[c], kRhoTiny);
  q.u = mom_z[c] / m;
  q.p = fmax(Pe[c] + Pi[c], kPressureTiny);
  const double k = rz::corner_nodal_kinetic_for_cell(c, nr, nz, corner_mass, v_r, v_z);
  q.E = (m * fmax(ee[c] + ei[c], 0.0) + fmax(k, 0.0)) / fmax(vol[c], 1.0e-300);
  const double eint = ee[c] + ei[c];
  q.y = (eint > 0.0 && isfinite(eint)) ? fmin(fmax(ee[c] / eint, 0.0), 1.0)
                                       : ye_default;
  return q;
}

__device__ inline Primitive ghost_primitive(const Primitive& interior,
                                            const int side,
                                            const int bc_kind,
                                            const int is_state_supply,
                                            const double supply_rho,
                                            const double supply_u,
                                            const double supply_eint,
                                            const double supply_ye,
                                            const double gamma) {
  Primitive g = interior;
  if (is_state_supply != 0) {
    g.rho = fmax(supply_rho, kRhoTiny);
    g.u = supply_u;
    const double e = fmax(supply_eint, 0.0);
    g.p = fmax((gamma - 1.0) * g.rho * e, kPressureTiny);
    g.E = g.rho * (e + 0.5 * g.u * g.u);
    g.y = fmin(fmax(supply_ye, 0.0), 1.0);
    return g;
  }
  if (bc_kind == 2) {  // Boundary2DType::REFLECT
    g.u = -interior.u;
  } else {
    (void)side;
  }
  return g;
}

__global__ void compute_hllc_z_face_flux_kernel(
    double* __restrict__ flux_m,
    double* __restrict__ flux_pz,
    double* __restrict__ flux_E,
    double* __restrict__ flux_y,
    const double* __restrict__ rho,
    const double* __restrict__ mass,
    const double* __restrict__ vol,
    const double* __restrict__ mom_z,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const double* __restrict__ corner_mass,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ x_r,
    const std::int8_t* __restrict__ hydro_active,
    HllcDeviceStats* __restrict__ stats,
    const int nr,
    const int nz,
    const double gamma,
    const double ye_default,
    const int allow_hlle,
    const int z_bottom_kind,
    const int z_top_kind,
    const int bottom_state_supply,
    const int top_state_supply,
    const double bottom_rho,
    const double bottom_u,
    const double bottom_e,
    const double top_rho,
    const double top_u,
    const double top_e) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_faces = nr * (nz + 1);
  if (f >= n_faces) {
    return;
  }
  const int i = f / (nz + 1);
  const int jf = f - i * (nz + 1);
  const int n_inner = rz::node_index(i, jf, nz);
  const int n_outer = rz::node_index(i + 1, jf, nz);
  const double r0 = x_r[n_inner];
  const double r1 = x_r[n_outer];
  const double area = M_PI * fabs(r1 * r1 - r0 * r0);

  Primitive L{};
  Primitive R{};
  if (jf == 0) {
    const int cR = i * nz;
    R = cell_primitive(cR, rho, mass, vol, mom_z, ee, ei, Pe, Pi, corner_mass, v_r,
                       v_z, nr, nz, ye_default);
    L = ghost_primitive(R, -1, z_bottom_kind, bottom_state_supply, bottom_rho,
                        bottom_u, bottom_e, ye_default, gamma);
  } else if (jf == nz) {
    const int cL = i * nz + (nz - 1);
    L = cell_primitive(cL, rho, mass, vol, mom_z, ee, ei, Pe, Pi, corner_mass, v_r,
                       v_z, nr, nz, ye_default);
    R = ghost_primitive(L, 1, z_top_kind, top_state_supply, top_rho, top_u,
                        top_e, ye_default, gamma);
  } else {
    const int cL = i * nz + (jf - 1);
    const int cR = i * nz + jf;
    const bool activeL = (hydro_active == nullptr) || (hydro_active[cL] != 0);
    const bool activeR = (hydro_active == nullptr) || (hydro_active[cR] != 0);
    if (!activeL || !activeR) {
      flux_m[f] = 0.0;
      flux_pz[f] = 0.0;
      flux_E[f] = 0.0;
      flux_y[f] = 0.0;
      return;
    }
    L = cell_primitive(cL, rho, mass, vol, mom_z, ee, ei, Pe, Pi, corner_mass, v_r,
                       v_z, nr, nz, ye_default);
    R = cell_primitive(cR, rho, mass, vol, mom_z, ee, ei, Pe, Pi, corner_mass, v_r,
                       v_z, nr, nz, ye_default);
  }

  const HllcFlux F = hllc_flux(L, R, gamma, allow_hlle != 0, stats);
  flux_m[f] = F.mass * area;
  flux_pz[f] = F.mom * area;
  flux_E[f] = F.energy * area;
  flux_y[f] = F.ye * area;
}

__global__ void apply_hllc_z_update_kernel(
    double* __restrict__ mass,
    double* __restrict__ rho,
    double* __restrict__ mom_z,
    double* __restrict__ E_cell,
    double* __restrict__ y_mass,
    const double* __restrict__ vol,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ corner_mass,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ flux_m,
    const double* __restrict__ flux_pz,
    const double* __restrict__ flux_E,
    const double* __restrict__ flux_y,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const double dt,
    const double rho_floor,
    const double ye_default,
    double* __restrict__ floor_E,
    int* __restrict__ rho_clamp_count) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const bool active = (hydro_active == nullptr) || (hydro_active[c] != 0);
  const int i = c / nz;
  const int j = c - i * nz;
  const int f_bot = i * (nz + 1) + j;
  const int f_top = f_bot + 1;
  const double k_old = rz::corner_nodal_kinetic_for_cell(c, nr, nz, corner_mass, v_r, v_z);
  const double eint = fmax(ee[c] + ei[c], 0.0);
  const double y = (eint > 0.0 && isfinite(eint)) ? fmin(fmax(ee[c] / eint, 0.0), 1.0)
                                                  : ye_default;
  double m_new = mass[c];
  double p_new = mom_z[c];
  double E_new = mass[c] * eint + fmax(k_old, 0.0);
  double y_new = mass[c] * y;
  if (active) {
    m_new -= dt * (flux_m[f_top] - flux_m[f_bot]);
    p_new -= dt * (flux_pz[f_top] - flux_pz[f_bot]);
    E_new -= dt * (flux_E[f_top] - flux_E[f_bot]);
    y_new -= dt * (flux_y[f_top] - flux_y[f_bot]);
  }
  const double m_floor = fmax(rho_floor * fmax(vol[c], 0.0), 0.0);
  if (!(m_new > m_floor) || !isfinite(m_new)) {
    const double repaired = fmax(m_floor, 1.0e-300);
    if (floor_E != nullptr && isfinite(m_new) && m_new > 0.0) {
      atomic_add_double_local(floor_E, 0.0);
    }
    if (rho_clamp_count != nullptr) {
      atomicAdd(rho_clamp_count, 1);
    }
    m_new = repaired;
    p_new = 0.0;
  }
  if (!isfinite(E_new)) {
    E_new = mass[c] * eint + fmax(k_old, 0.0);
  }
  mass[c] = m_new;
  rho[c] = m_new / fmax(vol[c], 1.0e-300);
  mom_z[c] = p_new;
  E_cell[c] = E_new;
  y_mass[c] = fmin(fmax(y_new, 0.0), m_new);
}

__global__ void project_hllc_cell_momentum_to_nodes_kernel(
    double* __restrict__ node_mom_z,
    double* __restrict__ node_mass,
    const double* __restrict__ mom_z,
    const double* __restrict__ mass,
    const double* __restrict__ corner_mass,
    const int nr,
    const int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int nodes[4] = {rz::node_index(i, j, nz), rz::node_index(i + 1, j, nz),
                        rz::node_index(i + 1, j + 1, nz),
                        rz::node_index(i, j + 1, nz)};
  const double u = (mass[c] > 0.0) ? mom_z[c] / mass[c] : 0.0;
  for (int q = 0; q < 4; ++q) {
    const double cm = corner_mass[c * 4 + q];
    atomic_add_double_local(node_mass + nodes[q], cm);
    atomic_add_double_local(node_mom_z + nodes[q], cm * u);
  }
}

__global__ void finish_hllc_node_projection_kernel(double* __restrict__ v_z,
                                                   const double* __restrict__ node_mom_z,
                                                   const double* __restrict__ node_mass,
                                                   const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  if (node_mass[n] > 0.0) {
    v_z[n] = node_mom_z[n] / node_mass[n];
  }
}

__global__ void recover_hllc_internal_energy_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ mass,
    const double* __restrict__ rho,
    const double* __restrict__ E_cell,
    const double* __restrict__ y_mass,
    const double* __restrict__ corner_mass,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ zbar,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const double dt,
    const double gamma,
    const double A,
    const double fallback_z,
    const double cv_e_default,
    const double cv_i_default,
    const double qei_multiplier,
    double* __restrict__ floor_E,
    int* __restrict__ clamp_count) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const bool active = (hydro_active == nullptr) || (hydro_active[c] != 0);
  if (!active || !(mass[c] > 0.0)) {
    return;
  }
  const double k = rz::corner_nodal_kinetic_for_cell(c, nr, nz, corner_mass, v_r, v_z);
  double e_int = (E_cell[c] - k) / fmax(mass[c], 1.0e-300);
  if (!isfinite(e_int) || e_int < 0.0) {
    const double repaired = fmax(e_int, 0.0);
    if (floor_E != nullptr && isfinite(e_int)) {
      atomic_add_double_local(floor_E, mass[c] * (repaired - e_int));
    }
    if (clamp_count != nullptr) {
      atomicAdd(clamp_count, 1);
    }
    e_int = repaired;
  }
  const double y = fmin(fmax(y_mass[c] / fmax(mass[c], 1.0e-300), 0.0), 1.0);
  double ee_new = y * e_int;
  double ei_new = (1.0 - y) * e_int;
  const double cv_e = fmax(cv_e_default, 0.0);
  const double cv_i = fmax(cv_i_default, 0.0);
  const double Te = (cv_e > 0.0) ? ee_new / cv_e : 0.0;
  const double Ti = (cv_i > 0.0) ? ei_new / cv_i : 0.0;
  const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : fallback_z;
  const double qei = tenryu::materials::compute_qei_term_analytical(
      fmax(rho[c], 0.0), fmax(Te, 0.0), fmax(Ti, 0.0), z, A, gamma, dt,
      qei_multiplier);
  ee_new -= qei;
  ei_new += qei;
  if (ee_new < 0.0 || !isfinite(ee_new)) {
    const double repaired = fmax(ee_new, 0.0);
    if (floor_E != nullptr && isfinite(ee_new)) {
      atomic_add_double_local(floor_E, mass[c] * (repaired - ee_new));
    }
    if (clamp_count != nullptr) {
      atomicAdd(clamp_count, 1);
    }
    ee_new = repaired;
  }
  if (ei_new < 0.0 || !isfinite(ei_new)) {
    const double repaired = fmax(ei_new, 0.0);
    if (floor_E != nullptr && isfinite(ei_new)) {
      atomic_add_double_local(floor_E, mass[c] * (repaired - ei_new));
    }
    if (clamp_count != nullptr) {
      atomicAdd(clamp_count, 1);
    }
    ei_new = repaired;
  }
  ee[c] = ee_new;
  ei[c] = ei_new;
}

template <typename Field>
std::vector<double> copy_to_host_vec(const Field& field) {
  std::vector<double> host(field.size());
  if (!host.empty()) {
    cuda_check_local(cudaMemcpy(host.data(),
                                field.data(),
                                host.size() * sizeof(double),
                                cudaMemcpyDeviceToHost),
                     "HLLC z flux copy field to host failed");
  }
  return host;
}

std::vector<double> copy_device_vec(const double* ptr, const std::size_t n) {
  std::vector<double> host(n);
  if (n > 0) {
    cuda_check_local(cudaMemcpy(host.data(), ptr, n * sizeof(double),
                                cudaMemcpyDeviceToHost),
                     "HLLC z flux copy device vec failed");
  }
  return host;
}

int width_between(const std::vector<double>& q,
                  const int nr,
                  const int nz,
                  const int row,
                  const double lo,
                  const double hi) {
  if (row < 0 || row >= nr) {
    return 0;
  }
  int first = -1;
  int last = -1;
  for (int j = 0; j < nz; ++j) {
    const double v = q[static_cast<std::size_t>(row * nz + j)];
    if (v >= lo && v <= hi) {
      if (first < 0) {
        first = j;
      }
      last = j;
    }
  }
  return (first >= 0 && last >= first) ? (last - first + 1) : 0;
}

void append_hllc_audit(const core::State& state,
                       const core::Config& cfg,
                       const HllcZFlux2DRZResult& result,
                       const double* d_flux_m,
                       const double* d_flux_pz,
                       const double* d_flux_E,
                       const double* d_E_cell,
                       const double* d_y_mass,
                       const double dt,
                       const int rank,
                       const char* hydro_half) {
  if (!cfg.numerics.hydro.hllc_z_flux_audit_2d_rz) {
    return;
  }
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int row = std::max(0, std::min(nr - 1, nr / 2));
  const auto Te = copy_to_host_vec(state.Te);
  const auto rho = copy_to_host_vec(state.rho);
  const auto mass = copy_to_host_vec(state.mass);
  const auto mom = copy_to_host_vec(state.hllc_mom_z_cell);
  const auto E_cell = copy_device_vec(d_E_cell, mass.size());
  const auto y_mass = copy_device_vec(d_y_mass, mass.size());
  const auto vz = copy_to_host_vec(state.v_z);
  const auto corner_mass = copy_to_host_vec(state.corner_mass);
  const std::size_t n_faces = static_cast<std::size_t>(nr) * (nz + 1U);
  const auto flux_m = copy_device_vec(d_flux_m, n_faces);
  const auto flux_pz = copy_device_vec(d_flux_pz, n_faces);
  const auto flux_E = copy_device_vec(d_flux_E, n_faces);

  long double m_sum = 0.0L;
  long double pz_sum = 0.0L;
  long double pz_resid_abs = 0.0L;
  long double pz_norm_abs = 0.0L;
  for (std::size_t c = 0; c < mass.size(); ++c) {
    m_sum += static_cast<long double>(mass[c]);
    pz_sum += static_cast<long double>(mom[c]);
    const int ci = static_cast<int>(c) / nz;
    const int cj = static_cast<int>(c) - ci * nz;
    const int nodes[4] = {rz::node_index(ci, cj, nz), rz::node_index(ci + 1, cj, nz),
                          rz::node_index(ci + 1, cj + 1, nz),
                          rz::node_index(ci, cj + 1, nz)};
    double pz_node = 0.0;
    for (int q = 0; q < 4; ++q) {
      pz_node += corner_mass[c * 4U + static_cast<std::size_t>(q)] *
                 vz[static_cast<std::size_t>(nodes[q])];
    }
    pz_resid_abs +=
        static_cast<long double>(std::abs(pz_node - mom[c]));
    pz_norm_abs += static_cast<long double>(std::abs(mom[c]));
  }
  const double cell_node_mom_residual_rel =
      pz_norm_abs > 0.0L ? static_cast<double>(pz_resid_abs / pz_norm_abs) : 0.0;
  const int te_width = width_between(Te, nr, nz, row, 32.0, 60.0);
  const auto& mat = cfg.materials.materials.front();
  const double gamma = mat.ideal_gas_gamma;
  const double gm1 = std::max(gamma - 1.0, 1.0e-30);
  const double A = mat.A > 0.0 ? mat.A : 1.0;
  const double z_bc = mat.Z > 0.0 ? mat.Z : 1.0;
  const double cv_e =
      z_bc * core::constants::eV_to_erg / (A * core::constants::proton_mass * gm1);
  std::vector<double> Te_cell_ke(Te.size(), 0.0);
  for (std::size_t c = 0; c < mass.size(); ++c) {
    const double m = mass[c];
    const double u = (m > 0.0) ? mom[c] / m : 0.0;
    const double k_cell = 0.5 * m * u * u;
    const double e_int = (m > 0.0) ? (E_cell[c] - k_cell) / m : 0.0;
    const double y = (m > 0.0) ? std::min(std::max(y_mass[c] / m, 0.0), 1.0) : 0.5;
    Te_cell_ke[c] = (cv_e > 0.0) ? y * std::max(e_int, 0.0) / cv_e : 0.0;
  }
  const int te_width_cell_ke = width_between(Te_cell_ke, nr, nz, row, 32.0, 60.0);
  const int projection_width_delta = te_width - te_width_cell_ke;
  int rho_first = -1;
  int rho_last = -1;
  double rho_min = std::numeric_limits<double>::infinity();
  double rho_max = 0.0;
  for (int j = 0; j < nz; ++j) {
    const double v = rho[static_cast<std::size_t>(row * nz + j)];
    rho_min = std::min(rho_min, v);
    rho_max = std::max(rho_max, v);
  }
  const double rho10 = rho_min + 0.1 * (rho_max - rho_min);
  const double rho90 = rho_min + 0.9 * (rho_max - rho_min);
  for (int j = 0; j < nz; ++j) {
    const double v = rho[static_cast<std::size_t>(row * nz + j)];
    if (v >= rho10 && v <= rho90) {
      if (rho_first < 0) {
        rho_first = j;
      }
      rho_last = j;
    }
  }
  const int rho_width = (rho_first >= 0 && rho_last >= rho_first)
                            ? (rho_last - rho_first + 1)
                            : 0;

  std::filesystem::path diag_dir(cfg.output.directory);
  diag_dir /= "diag";
  std::filesystem::create_directories(diag_dir);
  std::ostringstream name;
  name << "hllc_z_flux_2d_rz_rank" << std::setw(4) << std::setfill('0') << rank
       << ".csv";
  const std::filesystem::path path = diag_dir / name.str();
  const bool write_header = !std::filesystem::exists(path);
  std::ofstream out(path, std::ios::app);
  if (!out) {
    core::log_warning("HLLC z flux audit: failed to open " + path.string());
    return;
  }
  if (write_header) {
    out << "step,t,dt,half,row,te_width_cells,rho_width_cells,total_mass_g,"
           "total_mom_z_gcm_s,hlle_fallback_count,invalid_state_count,"
           "min_star_rho,min_star_p,bottom_mass_flux,top_mass_flux,"
           "bottom_mom_flux,top_mom_flux,bottom_E_flux,top_E_flux,"
           "projection_width_delta_cells,cell_node_mom_residual_rel,"
           "E_floor_injected,clamp_count,rho_clamp_count,reconstructed_momentum\n";
  }
  const int f_bot = row * (nz + 1);
  const int f_top = row * (nz + 1) + nz;
  out << state.step << ',' << std::setprecision(17) << state.t << ',' << dt << ','
      << (hydro_half != nullptr ? hydro_half : "") << ',' << row << ','
      << te_width << ',' << rho_width << ','
      << static_cast<double>(m_sum) << ',' << static_cast<double>(pz_sum) << ','
      << result.hlle_fallback_count << ',' << result.invalid_state_count << ','
      << result.min_star_rho << ',' << result.min_star_p << ','
      << flux_m[static_cast<std::size_t>(f_bot)] << ','
      << flux_m[static_cast<std::size_t>(f_top)] << ','
      << flux_pz[static_cast<std::size_t>(f_bot)] << ','
      << flux_pz[static_cast<std::size_t>(f_top)] << ','
      << flux_E[static_cast<std::size_t>(f_bot)] << ','
      << flux_E[static_cast<std::size_t>(f_top)] << ','
      << projection_width_delta << ',' << cell_node_mom_residual_rel << ','
      << result.E_floor_injected << ',' << result.clamp_count << ','
      << result.rho_clamp_count << ','
      << (result.reconstructed_momentum ? 1 : 0) << '\n';
}

}  // namespace

HllcZFlux2DRZResult apply_hllc_z_flux_2d_rz(
    core::State& state,
    const core::Config& cfg,
    const double dt,
    const std::int8_t* d_hydro_active,
    const int rank,
    const char* hydro_half) {
  HllcZFlux2DRZResult result;
  result.applied = true;
  TENRYU_ASSERT(
      state.corner_stride == 4,
      "corner_stride != 4: HLLC z-flux corner path is staged for a later revision");
  TENRYU_ASSERT(cfg.main.dimension == "2D_RZ",
                "HLLC z flux is supported only in 2D_RZ");
  TENRYU_ASSERT(cfg.numerics.hydro.total_energy_remap_2d_rz,
                "HLLC z flux requires total_energy_remap_2d_rz=true");
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "HLLC z flux requires a material definition");
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = nr * nz;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (n_cells <= 0 || n_nodes <= 0) {
    return result;
  }
  if (state.hllc_mom_z_cell.size() != static_cast<std::size_t>(n_cells)) {
    state.hllc_mom_z_cell.reset(static_cast<std::size_t>(n_cells));
    state.hllc_mom_z_cell_initialized = false;
  }

  const int blocks_cells = (n_cells + 255) / 256;
  const int blocks_nodes = (n_nodes + 255) / 256;
  const int n_faces = nr * (nz + 1);
  const int blocks_faces = (n_faces + 255) / 256;
  const auto& mat = cfg.materials.materials.front();
  const double gamma = mat.ideal_gas_gamma;
  const double gm1 = std::max(gamma - 1.0, 1.0e-30);
  const double A = mat.A > 0.0 ? mat.A : 1.0;
  const double z_bc = mat.Z > 0.0 ? mat.Z : 1.0;
  const double cv_i =
      core::constants::eV_to_erg / (A * core::constants::proton_mass * gm1);
  const double cv_e =
      z_bc * core::constants::eV_to_erg / (A * core::constants::proton_mass * gm1);
  const double ye_default = (cv_i + cv_e) > 0.0 ? cv_e / (cv_i + cv_e) : 0.5;
  const bool invariant_corner_mass = corner_mass_lagrangian_invariant_enabled(cfg);

  if (invariant_corner_mass) {
    warn_hllc_invariant_corner_mass_once();
    TENRYU_ASSERT(state.corner_mass_initialized &&
                      state.corner_mass.size() == static_cast<std::size_t>(4 * n_cells),
                  "HLLC z flux requires initialized invariant corner_mass");
  } else {
    if (state.corner_mass.size() != static_cast<std::size_t>(4 * n_cells)) {
      state.corner_mass.reset(static_cast<std::size_t>(4 * n_cells));
      state.corner_mass_initialized = false;
    }
    compute_hllc_corner_mass_kernel<<<blocks_cells, 256>>>(
        state.corner_mass.data(),
        state.mass.data(),
        state.x_r.data(),
        state.x_z.data(),
        rz::corner_mass_fallback_device_recorder(),
        rz::kCornerMassFallbackStageHllcPre,
        static_cast<int>(
            cfg.numerics.hydro.corner_mass_convention),
        nr,
        nz);
    sync_kernel_local("HLLC z flux corner mass precompute failed");
    state.corner_mass_initialized = true;
  }

  if (!state.hllc_mom_z_cell_initialized) {
    init_hllc_mom_from_nodes_kernel<<<blocks_cells, 256>>>(
        state.hllc_mom_z_cell.data(), state.mass.data(), state.corner_mass.data(),
        state.v_z.data(), nr, nz);
    sync_kernel_local("HLLC z flux initial cell momentum projection failed");
    state.hllc_mom_z_cell_initialized = true;
    result.reconstructed_momentum = true;
  }

  double* d_flux_m = nullptr;
  double* d_flux_pz = nullptr;
  double* d_flux_E = nullptr;
  double* d_flux_y = nullptr;
  double* d_E_cell = nullptr;
  double* d_y_mass = nullptr;
  double* d_node_mom = nullptr;
  double* d_node_mass = nullptr;
  double* d_floor_E = nullptr;
  int* d_clamp = nullptr;
  int* d_rho_clamp = nullptr;
  HllcDeviceStats* d_stats = nullptr;
  const std::size_t face_bytes = static_cast<std::size_t>(n_faces) * sizeof(double);
  const std::size_t cell_bytes = static_cast<std::size_t>(n_cells) * sizeof(double);
  const std::size_t node_bytes = static_cast<std::size_t>(n_nodes) * sizeof(double);
  d_flux_m = static_cast<double*>(core::device_scratch_acquire("hllc:flux_m", face_bytes));
  d_flux_pz = static_cast<double*>(core::device_scratch_acquire("hllc:flux_pz", face_bytes));
  d_flux_E = static_cast<double*>(core::device_scratch_acquire("hllc:flux_E", face_bytes));
  d_flux_y = static_cast<double*>(core::device_scratch_acquire("hllc:flux_y", face_bytes));
  d_E_cell = static_cast<double*>(core::device_scratch_acquire("hllc:E_cell", cell_bytes));
  d_y_mass = static_cast<double*>(core::device_scratch_acquire("hllc:y_mass", cell_bytes));
  d_node_mom = static_cast<double*>(core::device_scratch_acquire("hllc:node_mom", node_bytes));
  d_node_mass =
      static_cast<double*>(core::device_scratch_acquire("hllc:node_mass", node_bytes));
  d_floor_E = static_cast<double*>(core::device_scratch_acquire("hllc:floor_E", sizeof(double)));
  d_clamp = static_cast<int*>(core::device_scratch_acquire("hllc:clamp", sizeof(int)));
  d_rho_clamp =
      static_cast<int*>(core::device_scratch_acquire("hllc:rho_clamp", sizeof(int)));
  d_stats = static_cast<HllcDeviceStats*>(
      core::device_scratch_acquire("hllc:stats", sizeof(HllcDeviceStats)));
  cuda_check_local(cudaMemset(d_floor_E, 0, sizeof(double)),
                   "HLLC z flux memset floor failed");
  cuda_check_local(cudaMemset(d_clamp, 0, sizeof(int)),
                   "HLLC z flux memset clamp failed");
  cuda_check_local(cudaMemset(d_rho_clamp, 0, sizeof(int)),
                   "HLLC z flux memset rho clamp failed");
  HllcDeviceStats stats_init{};
  cuda_check_local(cudaMemcpy(d_stats, &stats_init, sizeof(HllcDeviceStats),
                              cudaMemcpyHostToDevice),
                   "HLLC z flux init stats failed");

  const auto& bc = cfg.numerics.hydro.boundary_2d;
  const auto bottom_kind = static_cast<int>(parse_boundary_2d_type(bc.z_bottom));
  const auto top_kind = static_cast<int>(parse_boundary_2d_type(bc.z_top));
  const int bottom_supply = bc.z_bottom_cfg.is_state_supply() ? 1 : 0;
  const int top_supply = bc.z_top_cfg.is_state_supply() ? 1 : 0;
  const double e_bottom = (cv_i + cv_e) * std::max(bc.z_bottom_cfg.supply_T_eV, 0.0);
  const double e_top = (cv_i + cv_e) * std::max(bc.z_top_cfg.supply_T_eV, 0.0);

  compute_hllc_z_face_flux_kernel<<<blocks_faces, 256>>>(
      d_flux_m, d_flux_pz, d_flux_E, d_flux_y, state.rho.data(), state.mass.data(),
      state.vol.data(), state.hllc_mom_z_cell.data(), state.ee.data(),
      state.ei.data(), state.Pe.data(), state.Pi.data(), state.corner_mass.data(),
      state.v_r.data(), state.v_z.data(), state.x_r.data(), d_hydro_active, d_stats,
      nr, nz, gamma, ye_default, cfg.numerics.hydro.hllc_z_flux_hlle_fallback ? 1 : 0,
      bottom_kind, top_kind, bottom_supply, top_supply,
      bc.z_bottom_cfg.supply_rho_g_per_cc, bc.z_bottom_cfg.supply_u_z_cm_per_s,
      e_bottom, bc.z_top_cfg.supply_rho_g_per_cc,
      bc.z_top_cfg.supply_u_z_cm_per_s, e_top);
  sync_kernel_local("HLLC z flux face flux kernel failed");

  apply_hllc_z_update_kernel<<<blocks_cells, 256>>>(
      state.mass.data(), state.rho.data(), state.hllc_mom_z_cell.data(), d_E_cell,
      d_y_mass, state.vol.data(), state.ee.data(), state.ei.data(),
      state.corner_mass.data(), state.v_r.data(), state.v_z.data(), d_flux_m,
      d_flux_pz, d_flux_E, d_flux_y, d_hydro_active, nr, nz, dt,
      cfg.numerics.floors.rho, ye_default, d_floor_E, d_rho_clamp);
  sync_kernel_local("HLLC z flux conservative update failed");

  if (!invariant_corner_mass) {
    compute_hllc_corner_mass_kernel<<<blocks_cells, 256>>>(
        state.corner_mass.data(),
        state.mass.data(),
        state.x_r.data(),
        state.x_z.data(),
        rz::corner_mass_fallback_device_recorder(),
        rz::kCornerMassFallbackStageHllcPost,
        static_cast<int>(
            cfg.numerics.hydro.corner_mass_convention),
        nr,
        nz);
    sync_kernel_local("HLLC z flux corner mass postcompute failed");
    state.corner_mass_initialized = true;
  }

  cuda_check_local(cudaMemset(d_node_mom, 0, node_bytes),
                   "HLLC z flux memset node momentum failed");
  cuda_check_local(cudaMemset(d_node_mass, 0, node_bytes),
                   "HLLC z flux memset node mass failed");
  project_hllc_cell_momentum_to_nodes_kernel<<<blocks_cells, 256>>>(
      d_node_mom, d_node_mass, state.hllc_mom_z_cell.data(), state.mass.data(),
      state.corner_mass.data(), nr, nz);
  finish_hllc_node_projection_kernel<<<blocks_nodes, 256>>>(
      state.v_z.data(), d_node_mom, d_node_mass, n_nodes);
  sync_kernel_local("HLLC z flux node projection failed");

  recover_hllc_internal_energy_kernel<<<blocks_cells, 256>>>(
      state.ee.data(), state.ei.data(), state.mass.data(), state.rho.data(), d_E_cell,
      d_y_mass, state.corner_mass.data(), state.v_r.data(), state.v_z.data(),
      state.zbar.data(), d_hydro_active, nr, nz, dt, gamma, A, z_bc, cv_e, cv_i,
      cfg.numerics.hydro.qei_multiplier, d_floor_E, d_clamp);
  sync_kernel_local("HLLC z flux internal energy recovery failed");

  cuda_check_local(cudaMemcpy(&result.E_floor_injected, d_floor_E, sizeof(double),
                              cudaMemcpyDeviceToHost),
                   "HLLC z flux copy floor failed");
  cuda_check_local(cudaMemcpy(&result.clamp_count, d_clamp, sizeof(int),
                              cudaMemcpyDeviceToHost),
                   "HLLC z flux copy clamp failed");
  cuda_check_local(cudaMemcpy(&result.rho_clamp_count, d_rho_clamp, sizeof(int),
                              cudaMemcpyDeviceToHost),
                   "HLLC z flux copy rho clamp failed");
  HllcDeviceStats stats{};
  cuda_check_local(cudaMemcpy(&stats, d_stats, sizeof(HllcDeviceStats),
                              cudaMemcpyDeviceToHost),
                   "HLLC z flux copy stats failed");
  result.hlle_fallback_count = stats.hlle_fallback_count;
  result.invalid_state_count = stats.invalid_state_count;
  result.min_star_rho = stats.min_star_rho;
  result.min_star_p = stats.min_star_p;

  append_hllc_audit(state, cfg, result, d_flux_m, d_flux_pz, d_flux_E, d_E_cell,
                    d_y_mass, dt, rank, hydro_half);
  return result;
}

}  // namespace tenryu::hydro
