#include "hydro/reference_barrier_ale.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>

#include "core/error.hpp"
#include "hydro/ale_remap.cuh"
#include "hydro/ale_remap_2d_rz.hpp"
#include "hydro/ale_scaled_reference.cuh"
#include "hydro/boundary_2d.hpp"
#include "hydro/core_freeze_ale.cuh"
#include "hydro/mesh_motion_trace.hpp"
#include "hydro/pole_axis_constraints.cuh"
#include "mesh/mesh.hpp"

namespace tenryu::hydro {
namespace {

constexpr double kPi = 3.14159265358979323846;

void cuda_check(const cudaError_t err, const char* msg) {
  TENRYU_ASSERT(err == cudaSuccess, msg);
}

std::uint8_t* upload_node_flags_if_constraints(const tenryu::core::State& state) {
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
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_node_flags),
                        node_flags.size() * sizeof(std::uint8_t)),
             "reference barrier cudaMalloc node_flags failed");
  cuda_check(cudaMemcpy(d_node_flags,
                        node_flags.data(),
                        node_flags.size() * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             "reference barrier cudaMemcpy node_flags failed");
  return d_node_flags;
}

int node_index(const int i, const int j, const int nz) {
  return i * (nz + 1) + j;
}

double cross2(const double ar, const double az, const double br, const double bz) {
  return ar * bz - az * br;
}

void corner_jacobians(const double* r, const double* z, double* j) {
  j[0] = cross2(r[1] - r[0], z[1] - z[0], r[3] - r[0], z[3] - z[0]);
  j[1] = cross2(r[1] - r[0], z[1] - z[0], r[2] - r[1], z[2] - z[1]);
  j[2] = cross2(r[2] - r[3], z[2] - z[3], r[2] - r[1], z[2] - z[1]);
  j[3] = cross2(r[2] - r[3], z[2] - z[3], r[3] - r[0], z[3] - z[0]);
}

void triangle_corner_jacobians(const double* r, const double* z, double* j) {
  for (int k = 0; k < 3; ++k) {
    const int kp = (k + 1) % 3;
    const int km = (k + 2) % 3;
    j[k] = cross2(r[kp] - r[k], z[kp] - z[k],
                  r[km] - r[k], z[km] - z[k]);
  }
}

double diagonal_corner_b_eff(const double* r, const double* z) {
  return 0.5 * (std::hypot(r[1] - r[0], z[1] - z[0]) +
                std::hypot(r[3] - r[0], z[3] - z[0]));
}

int state_cell_active_nverts(const tenryu::core::State& state,
                             const int c,
                             const int n_cells) {
  return state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)
             ? mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c)
             : mesh::kMeshTopoCellStorageSlots;
}

bool state_has_tri_cell_nverts(const tenryu::core::State& state) {
  const int n_cells = state.mesh.topo.n_cells;
  if (state.mesh.cell_nverts.size() != static_cast<std::size_t>(n_cells)) {
    return false;
  }
  return std::any_of(state.mesh.cell_nverts.begin(),
                     state.mesh.cell_nverts.end(),
                     [](const std::uint8_t nverts) { return nverts == 3U; });
}

std::uint8_t* upload_cell_nverts_if_active(const tenryu::core::State& state) {
  if (!state_has_tri_cell_nverts(state)) {
    return nullptr;
  }
  std::uint8_t* d_cell_nverts = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cell_nverts),
                        state.mesh.cell_nverts.size() * sizeof(std::uint8_t)),
             "reference barrier cudaMalloc cell_nverts failed");
  cuda_check(cudaMemcpy(d_cell_nverts,
                        state.mesh.cell_nverts.data(),
                        state.mesh.cell_nverts.size() * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             "reference barrier cudaMemcpy cell_nverts failed");
  return d_cell_nverts;
}

double compute_axis_margin_min_host(const tenryu::core::State& state) {
  if (state.mesh.topo.multiblock.has_value()) {
    const auto& mb = *state.mesh.topo.multiblock;
    const int n_cells = state.mesh.topo.n_cells;
    if (n_cells <= 0 ||
        mb.cell_node_csr_offsets.size() != static_cast<std::size_t>(n_cells) + 1U ||
        mb.cell_node_csr_indices.size() != static_cast<std::size_t>(n_cells) * 4U) {
      return std::numeric_limits<double>::infinity();
    }
    std::vector<double> r;
    std::vector<double> z;
    state.x_r.copy_to_host(r);
    state.x_z.copy_to_host(z);

    double out = std::numeric_limits<double>::infinity();
    for (int c = 0; c < n_cells; ++c) {
      const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
      const int active_nverts = state_cell_active_nverts(state, c, n_cells);
      double rc = 0.0;
      double zc = 0.0;
      for (int k = 0; k < active_nverts; ++k) {
        const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        rc += r[static_cast<std::size_t>(n)];
        zc += z[static_cast<std::size_t>(n)];
      }
      if (active_nverts == 3) {
        rc *= (1.0 / 3.0);
        zc *= (1.0 / 3.0);
      } else {
        rc *= 0.25;
        zc *= 0.25;
      }
      const double s = std::hypot(rc, zc);
      if (s > 0.0 && std::isfinite(s)) {
        out = std::min(out, std::abs(rc) / s);
      }
    }
    return out;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  if (nr <= 0 || nz <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  std::vector<double> r;
  std::vector<double> z;
  state.x_r.copy_to_host(r);
  state.x_z.copy_to_host(z);

  double out = std::numeric_limits<double>::infinity();
  const int i_max = std::min(nr - 1, 2);
  for (int i = 0; i <= i_max; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int n0 = node_index(i, j, nz);
      const int n1 = node_index(i + 1, j, nz);
      const int n2 = node_index(i + 1, j + 1, nz);
      const int n3 = node_index(i, j + 1, nz);
      const double rc = 0.25 * (r[n0] + r[n1] + r[n2] + r[n3]);
      const double zc = 0.25 * (z[n0] + z[n1] + z[n2] + z[n3]);
      const double s = std::hypot(rc, zc);
      if (s > 0.0 && std::isfinite(s)) {
        out = std::min(out, std::abs(rc) / s);
      }
    }
  }
  return out;
}

double compute_corner_j_ratio_min_host(const tenryu::core::State& state) {
  if (state.mesh.topo.multiblock.has_value()) {
    const auto& mb = *state.mesh.topo.multiblock;
    const int n_cells = state.mesh.topo.n_cells;
    if (n_cells <= 0 ||
        mb.cell_node_csr_offsets.size() != static_cast<std::size_t>(n_cells) + 1U ||
        mb.cell_node_csr_indices.size() != static_cast<std::size_t>(n_cells) * 4U ||
        state.x_r_initial.size() != state.x_r.size() ||
        state.x_z_initial.size() != state.x_z.size()) {
      return std::numeric_limits<double>::infinity();
    }
    std::vector<double> r_old;
    std::vector<double> z_old;
    std::vector<double> r_new;
    std::vector<double> z_new;
    state.x_r_initial.copy_to_host(r_old);
    state.x_z_initial.copy_to_host(z_old);
    state.x_r.copy_to_host(r_new);
    state.x_z.copy_to_host(z_new);

    double out = std::numeric_limits<double>::infinity();
    for (int c = 0; c < n_cells; ++c) {
      const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
      const int active_nverts = state_cell_active_nverts(state, c, n_cells);
      double r0[4]{};
      double z0[4]{};
      double r1[4]{};
      double z1[4]{};
      for (int k = 0; k < active_nverts; ++k) {
        const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        r0[k] = r_old[static_cast<std::size_t>(n)];
        z0[k] = z_old[static_cast<std::size_t>(n)];
        r1[k] = r_new[static_cast<std::size_t>(n)];
        z1[k] = z_new[static_cast<std::size_t>(n)];
      }
      double j0[4]{};
      double j1[4]{};
      if (active_nverts == 3) {
        triangle_corner_jacobians(r0, z0, j0);
        triangle_corner_jacobians(r1, z1, j1);
      } else {
        corner_jacobians(r0, z0, j0);
        corner_jacobians(r1, z1, j1);
      }
      for (int k = 0; k < active_nverts; ++k) {
        if (j0[k] != 0.0 && std::isfinite(j0[k]) && std::isfinite(j1[k])) {
          out = std::min(out, j1[k] / j0[k]);
        }
      }
      if (active_nverts != 3) {
        const double b0 = diagonal_corner_b_eff(r0, z0);
        const double b1 = diagonal_corner_b_eff(r1, z1);
        if (b0 > 0.0 && std::isfinite(b0) && std::isfinite(b1)) {
          out = std::min(out, b1 / b0);
        }
      }
    }
    return out;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  if (nr <= 0 || nz <= 0 || state.x_r_initial.size() != state.x_r.size() ||
      state.x_z_initial.size() != state.x_z.size()) {
    return std::numeric_limits<double>::infinity();
  }
  std::vector<double> r_old;
  std::vector<double> z_old;
  std::vector<double> r_new;
  std::vector<double> z_new;
  state.x_r_initial.copy_to_host(r_old);
  state.x_z_initial.copy_to_host(z_old);
  state.x_r.copy_to_host(r_new);
  state.x_z.copy_to_host(z_new);

  double out = std::numeric_limits<double>::infinity();
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int nodes[4] = {
          node_index(i, j, nz),
          node_index(i + 1, j, nz),
          node_index(i + 1, j + 1, nz),
          node_index(i, j + 1, nz),
      };
      double r0[4]{};
      double z0[4]{};
      double r1[4]{};
      double z1[4]{};
      for (int k = 0; k < 4; ++k) {
        r0[k] = r_old[nodes[k]];
        z0[k] = z_old[nodes[k]];
        r1[k] = r_new[nodes[k]];
        z1[k] = z_new[nodes[k]];
      }
      double j0[4]{};
      double j1[4]{};
      corner_jacobians(r0, z0, j0);
      corner_jacobians(r1, z1, j1);
      for (int k = 0; k < 4; ++k) {
        if (j0[k] != 0.0 && std::isfinite(j0[k]) && std::isfinite(j1[k])) {
          out = std::min(out, j1[k] / j0[k]);
        }
      }
      const double b0 = diagonal_corner_b_eff(r0, z0);
      const double b1 = diagonal_corner_b_eff(r1, z1);
      if (b0 > 0.0 && std::isfinite(b0) && std::isfinite(b1)) {
        out = std::min(out, b1 / b0);
      }
    }
  }
  return out;
}

__global__ void build_spherical_equal_angle_kernel(
    const double* __restrict__ r0,
    const double* __restrict__ z0,
    double* __restrict__ target_r,
    double* __restrict__ target_z,
    const int nr,
    const int nz) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_nodes = (nr + 1) * (nz + 1);
  if (n >= n_nodes) {
    return;
  }
  const int i = n / (nz + 1);
  const int j = n - i * (nz + 1);
  const int shell = i * (nz + 1);
  const double s = hypot(r0[shell], z0[shell]);
  const double theta = (nz > 0) ? kPi * static_cast<double>(j) / static_cast<double>(nz)
                                : 0.0;
  target_r[n] = s * sin(theta);
  target_z[n] = s * cos(theta);
}

__global__ void delta_kernel(const double* __restrict__ x_r,
                             const double* __restrict__ x_z,
                             const double* __restrict__ target_r,
                             const double* __restrict__ target_z,
                             double* __restrict__ delta_r,
                             double* __restrict__ delta_z,
                             const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  delta_r[n] = target_r[n] - x_r[n];
  delta_z[n] = target_z[n] - x_z[n];
}

__global__ void blend_kernel(double* __restrict__ x_r,
                             double* __restrict__ x_z,
                             const double* __restrict__ delta_r,
                             const double* __restrict__ delta_z,
                             const double sigma,
                             const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  x_r[n] += sigma * delta_r[n];
  x_z[n] += sigma * delta_z[n];
}

__global__ void blend_to_output_kernel(double* __restrict__ out_r,
                                       double* __restrict__ out_z,
                                       const double* __restrict__ x_r_old,
                                       const double* __restrict__ x_z_old,
                                       const double* __restrict__ delta_r,
                                       const double* __restrict__ delta_z,
                                       const double sigma,
                                       const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  out_r[n] = x_r_old[n] + sigma * delta_r[n];
  out_z[n] = x_z_old[n] + sigma * delta_z[n];
}

__global__ void pack_cell_conserved_kernel(double* __restrict__ mom_r,
                                           double* __restrict__ mom_z,
                                           double* __restrict__ e_e,
                                           double* __restrict__ e_i,
                                           double* __restrict__ v_r_cell,
                                           double* __restrict__ v_z_cell,
                                           const double* __restrict__ rho,
                                           const double* __restrict__ ee,
                                           const double* __restrict__ ei,
                                           const double* __restrict__ v_r_node,
                                           const double* __restrict__ v_z_node,
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
  const int n01 = i * stride + (j + 1);
  const double vr = 0.25 * (v_r_node[n00] + v_r_node[n10] + v_r_node[n11] + v_r_node[n01]);
  const double vz = 0.25 * (v_z_node[n00] + v_z_node[n10] + v_z_node[n11] + v_z_node[n01]);
  const double rho_c = fmax(rho[c], 0.0);
  v_r_cell[c] = vr;
  v_z_cell[c] = vz;
  mom_r[c] = rho_c * vr;
  mom_z[c] = rho_c * vz;
  e_e[c] = rho_c * fmax(ee[c], 0.0);
  e_i[c] = rho_c * fmax(ei[c], 0.0);
}

__global__ void recover_cell_primitives_kernel(double* __restrict__ rho,
                                               double* __restrict__ mass,
                                               double* __restrict__ ee,
                                               double* __restrict__ ei,
                                               double* __restrict__ v_r_cell,
                                               double* __restrict__ v_z_cell,
                                               const double* __restrict__ mom_r,
                                               const double* __restrict__ mom_z,
                                               const double* __restrict__ e_e,
                                               const double* __restrict__ e_i,
                                               const double* __restrict__ vol,
                                               const double rho_floor,
                                               const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const double V = fmax(vol[c], 1.0e-30);
  const double rho_c = fmax(rho[c], rho_floor);
  rho[c] = rho_c;
  mass[c] = rho_c * V;
  if (rho_c > 0.0) {
    v_r_cell[c] = mom_r[c] / rho_c;
    v_z_cell[c] = mom_z[c] / rho_c;
    ee[c] = fmax(e_e[c], 0.0) / rho_c;
    ei[c] = fmax(e_i[c], 0.0) / rho_c;
  } else {
    v_r_cell[c] = 0.0;
    v_z_cell[c] = 0.0;
    ee[c] = 0.0;
    ei[c] = 0.0;
  }
}

__global__ void gather_material_field_kernel(double* __restrict__ out,
                                             const double* __restrict__ in,
                                             const int n_cells,
                                             const int n_mat,
                                             const int mat) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  out[c] = in[c * n_mat + mat];
}

__global__ void scatter_material_field_kernel(double* __restrict__ out,
                                              const double* __restrict__ in,
                                              const int n_cells,
                                              const int n_mat,
                                              const int mat) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  out[c * n_mat + mat] = fmax(in[c], 0.0);
}

__global__ void gather_group_field_kernel(double* __restrict__ out,
                                          const double* __restrict__ in,
                                          const int n_cells,
                                          const int n_groups,
                                          const int group) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  out[c] = in[c * n_groups + group];
}

__global__ void scatter_group_field_kernel(double* __restrict__ out,
                                           const double* __restrict__ in,
                                           const int n_cells,
                                           const int n_groups,
                                           const int group) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  out[c * n_groups + group] = fmax(in[c], 0.0);
}

__global__ void normalize_reference_volfrac_kernel(double* __restrict__ volfrac,
                                                   const int n_cells,
                                                   const int n_mat) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells || n_mat <= 0) {
    return;
  }
  if (n_mat == 1) {
    volfrac[c * n_mat] = 1.0;
    return;
  }
  double sum = 0.0;
  int imax = 0;
  double vmax = volfrac[c * n_mat];
  for (int m = 0; m < n_mat; ++m) {
    const int idx = c * n_mat + m;
    const double clamped = fmax(volfrac[idx], 0.0);
    volfrac[idx] = clamped;
    sum += clamped;
    if (clamped > vmax) {
      vmax = clamped;
      imax = m;
    }
  }
  if (sum > 1.0e-30) {
    for (int m = 0; m < n_mat; ++m) {
      volfrac[c * n_mat + m] /= sum;
    }
  } else {
    for (int m = 0; m < n_mat; ++m) {
      volfrac[c * n_mat + m] = (m == imax) ? 1.0 : 0.0;
    }
  }
}

__global__ void project_reference_cell_velocity_to_nodes_kernel(
    double* __restrict__ v_r_node,
    double* __restrict__ v_z_node,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const std::uint8_t* __restrict__ node_flags,
    const int nr,
    const int nz,
    const int r_outer_bc_mode,
    const int z_bottom_bc_mode,
    const int z_top_bc_mode) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_nodes = (nr + 1) * (nz + 1);
  if (n >= n_nodes) {
    return;
  }
  const int stride = nz + 1;
  const int i = n / stride;
  const int j = n - i * stride;
  double w_sum = 0.0;
  double vr_sum = 0.0;
  double vz_sum = 0.0;
  for (int di = -1; di <= 0; ++di) {
    for (int dj = -1; dj <= 0; ++dj) {
      const int ic = i + di;
      const int jc = j + dj;
      if (ic < 0 || ic >= nr || jc < 0 || jc >= nz) {
        continue;
      }
      const int c = ic * nz + jc;
      const double w = fmax(rho[c] * vol[c], 0.0);
      w_sum += w;
      vr_sum += w * v_r_cell[c];
      vz_sum += w * v_z_cell[c];
    }
  }
  if (w_sum > 0.0) {
    v_r_node[n] = vr_sum / w_sum;
    v_z_node[n] = vz_sum / w_sum;
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

double relative_residual(const double before, const double after) {
  const double denom = std::max(std::abs(before), 1.0);
  return std::abs(after - before) / denom;
}

struct ConservedSums {
  double mass = 0.0;
  double internal = 0.0;
  double kinetic = 0.0;
};

ConservedSums compute_conserved_sums_host(const tenryu::core::State& state) {
  std::vector<double> rho;
  std::vector<double> mass_field;
  std::vector<double> ee;
  std::vector<double> ei;
  std::vector<double> vol;
  std::vector<double> vr;
  std::vector<double> vz;
  state.rho.copy_to_host(rho);
  state.mass.copy_to_host(mass_field);
  state.ee.copy_to_host(ee);
  state.ei.copy_to_host(ei);
  state.vol.copy_to_host(vol);
  state.v_r.copy_to_host(vr);
  state.v_z.copy_to_host(vz);

  ConservedSums sums;
  if (state.mesh.topo.multiblock.has_value()) {
    const auto& mb = *state.mesh.topo.multiblock;
    const int n_cells = state.mesh.topo.n_cells;
    for (int c = 0; c < n_cells; ++c) {
      const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
      const int active_nverts = state_cell_active_nverts(state, c, n_cells);
      double vr_cell = 0.0;
      double vz_cell = 0.0;
      for (int k = 0; k < active_nverts; ++k) {
        const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        vr_cell += vr[static_cast<std::size_t>(n)];
        vz_cell += vz[static_cast<std::size_t>(n)];
      }
      if (active_nverts == 3) {
        vr_cell *= (1.0 / 3.0);
        vz_cell *= (1.0 / 3.0);
      } else {
        vr_cell *= 0.25;
        vz_cell *= 0.25;
      }
      const double mass =
          mass_field.empty()
              ? rho[static_cast<std::size_t>(c)] * vol[static_cast<std::size_t>(c)]
              : mass_field[static_cast<std::size_t>(c)];
      sums.mass += mass;
      sums.internal += mass * (ee[static_cast<std::size_t>(c)] +
                               ei[static_cast<std::size_t>(c)]);
      sums.kinetic += 0.5 * mass * (vr_cell * vr_cell + vz_cell * vz_cell);
    }
    return sums;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int c = i * nz + j;
      const int n00 = node_index(i, j, nz);
      const int n10 = node_index(i + 1, j, nz);
      const int n11 = node_index(i + 1, j + 1, nz);
      const int n01 = node_index(i, j + 1, nz);
      const double mass = rho[c] * vol[c];
      const double vr_cell = 0.25 * (vr[n00] + vr[n10] + vr[n11] + vr[n01]);
      const double vz_cell = 0.25 * (vz[n00] + vz[n10] + vz[n11] + vz[n01]);
      sums.mass += mass;
      sums.internal += mass * (ee[c] + ei[c]);
      sums.kinetic += 0.5 * mass * (vr_cell * vr_cell + vz_cell * vz_cell);
    }
  }
  return sums;
}

void log_reference_conservation(const int step,
                                const ConservedSums& before,
                                const ConservedSums& after) {
  const double mass_residual = relative_residual(before.mass, after.mass);
  const double energy_residual =
      relative_residual(before.internal + before.kinetic, after.internal + after.kinetic);
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(6)
      << "[ref-barrier-ale-conservation] step " << step
      << ": mass_residual=" << mass_residual
      << " energy_residual=" << energy_residual;
  if (mass_residual > 1.0e-10 || energy_residual > 1.0e-10) {
    core::log_warning(oss.str());
  } else {
    core::log_info(oss.str());
  }
}

int velocity_bc_mode(const Boundary2DType bc) {
  if (bc == Boundary2DType::FIXED) {
    return 2;
  }
  if (bc == Boundary2DType::REFLECT || bc == Boundary2DType::STATE_SUPPLY) {
    return 1;
  }
  return 0;
}

int* upload_multiblock_stable_cell_ids(const tenryu::core::State& state) {
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "CSR reference barrier requires multiblock topology");
  const auto& stable = state.mesh.topo.multiblock->cell_id_stable;
  TENRYU_ASSERT(stable.size() == static_cast<std::size_t>(state.mesh.topo.n_cells),
                "CSR reference barrier requires stable cell ids");
  int* d_stable = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_stable),
                        stable.size() * sizeof(int)),
             "reference barrier stable-cell allocation failed");
  cuda_check(cudaMemcpy(d_stable,
                        stable.data(),
                        stable.size() * sizeof(int),
                        cudaMemcpyHostToDevice),
             "reference barrier stable-cell upload failed");
  return d_stable;
}

ale::AleRemap2DRZResult apply_multiblock_csr_reference_remap(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const double* d_xr_old,
    const double* d_xz_old,
    const double* d_vol_old,
    const double* d_delta_r,
    const double* d_delta_z,
    const double sigma,
    const ReferenceBarrierScope* scope = nullptr) {
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "CSR reference barrier remap requires multiblock topology");
  const int n_nodes = state.mesh.topo.n_nodes;
  const int n_cells = state.mesh.topo.n_cells;
  const std::size_t node_bytes = static_cast<std::size_t>(n_nodes) * sizeof(double);
  const std::size_t cell_bytes = static_cast<std::size_t>(n_cells) * sizeof(double);
  TENRYU_ASSERT(state.x_r_reference.size() == static_cast<std::size_t>(n_nodes) &&
                    state.x_z_reference.size() == static_cast<std::size_t>(n_nodes),
                "CSR reference barrier remap requires reference node storage");
  TENRYU_ASSERT(state.cell_vol_initial.size() == static_cast<std::size_t>(n_cells),
                "CSR reference barrier remap requires reference volume storage");

  double* d_ref_r_old = nullptr;
  double* d_ref_z_old = nullptr;
  double* d_cell_vol_initial_old = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ref_r_old), node_bytes),
             "reference barrier old reference r allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ref_z_old), node_bytes),
             "reference barrier old reference z allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cell_vol_initial_old), cell_bytes),
             "reference barrier old reference volume allocation failed");
  cuda_check(cudaMemcpy(d_ref_r_old,
                        state.x_r_reference.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice),
             "reference barrier old reference r copy failed");
  cuda_check(cudaMemcpy(d_ref_z_old,
                        state.x_z_reference.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice),
             "reference barrier old reference z copy failed");
  cuda_check(cudaMemcpy(d_cell_vol_initial_old,
                        state.cell_vol_initial.data(),
                        cell_bytes,
                        cudaMemcpyDeviceToDevice),
             "reference barrier old reference volume copy failed");

  const auto restore_reference = [&]() {
    cuda_check(cudaMemcpy(state.x_r_reference.data(),
                          d_ref_r_old,
                          node_bytes,
                          cudaMemcpyDeviceToDevice),
               "reference barrier restore reference r failed");
    cuda_check(cudaMemcpy(state.x_z_reference.data(),
                          d_ref_z_old,
                          node_bytes,
                          cudaMemcpyDeviceToDevice),
               "reference barrier restore reference z failed");
    cuda_check(cudaMemcpy(state.cell_vol_initial.data(),
                          d_cell_vol_initial_old,
                          cell_bytes,
                          cudaMemcpyDeviceToDevice),
               "reference barrier restore reference volume failed");
  };

  const int blocks_nodes = (n_nodes + 255) / 256;
  mesh_trace::trace_cell0_geometry(state, cfg, "refbar_pre_csr_remap");
  blend_to_output_kernel<<<blocks_nodes, 256>>>(state.x_r_reference.data(),
                                                state.x_z_reference.data(),
                                                d_xr_old,
                                                d_xz_old,
                                                d_delta_r,
                                                d_delta_z,
                                                sigma,
                                                n_nodes);
  cuda_check(cudaGetLastError(),
             "reference barrier CSR candidate blend kernel launch failed");
  tenryu::core::DeviceArray<std::uint8_t> core_freeze_frozen_nodes;
  tenryu::hydro::ale::core_freeze::restore_target_if_enabled(
      state,
      cfg,
      state.x_r_reference.data(),
      state.x_z_reference.data(),
      d_xr_old,
      d_xz_old,
      false,
      "reference_barrier_csr",
      cfg.numerics.ale.core_freeze_skip_velocity_projection
          ? &core_freeze_frozen_nodes
          : nullptr);

  cuda_check(cudaMemcpy(state.x_r.data(),
                        state.x_r_reference.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice),
             "reference barrier CSR candidate r copy failed");
  cuda_check(cudaMemcpy(state.x_z.data(),
                        state.x_z_reference.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice),
             "reference barrier CSR candidate z copy failed");
  cuda_check(cudaDeviceSynchronize(),
             "reference barrier CSR candidate synchronize failed");
  state.mesh.node_r = state.x_r.data();
  state.mesh.node_z = state.x_z.data();
  state.mesh.recompute_geometry();
  state.cell_vol_initial = state.mesh.cell_vol;
  mesh_trace::trace_cell0_geometry(state, cfg, "refbar_candidate");
  tenryu::hydro::ale::ale_velcoherence::sample(
      state, cfg, "s1_post_rezone");

  cuda_check(cudaMemcpy(state.x_r.data(),
                        d_xr_old,
                        node_bytes,
                        cudaMemcpyDeviceToDevice),
             "reference barrier restore old r before CSR remap failed");
  cuda_check(cudaMemcpy(state.x_z.data(),
                        d_xz_old,
                        node_bytes,
                        cudaMemcpyDeviceToDevice),
             "reference barrier restore old z before CSR remap failed");
  cuda_check(cudaMemcpy(state.vol.data(),
                        d_vol_old,
                        cell_bytes,
                        cudaMemcpyDeviceToDevice),
             "reference barrier restore old volume before CSR remap failed");
  cuda_check(cudaDeviceSynchronize(),
             "reference barrier restore old mesh before CSR remap failed");

  tenryu::core::Config remap_cfg = cfg;
  remap_cfg.numerics.ale.conservative_remap_enabled = true;
  remap_cfg.numerics.ale.conservative_remap_target = "reference";
  remap_cfg.numerics.ale.multiblock_scaled_reference_enabled = false;
  const std::uint8_t* core_freeze_velocity_mask =
      core_freeze_frozen_nodes.size() == static_cast<std::size_t>(n_nodes)
          ? core_freeze_frozen_nodes.data()
          : nullptr;
  ale::AleRemap2DRZOverrides overrides;
  if (scope != nullptr) {
    overrides.active_cell_mask = scope->d_active_cell_mask;
    overrides.velocity_projection_frozen_node_mask =
        scope->d_frozen_velocity_node_mask;
  }
  const auto remap_result =
      tenryu::hydro::ale::ale_remap_2d_rz(
          state,
          remap_cfg,
          nullptr,
          0.0,
          core_freeze_velocity_mask,
          overrides);
  restore_reference();
  mesh_trace::trace_cell0_geometry(state, cfg, "refbar_post_remap");
  cuda_check(cudaFree(d_cell_vol_initial_old),
             "reference barrier old reference volume free failed");
  cuda_check(cudaFree(d_ref_z_old), "reference barrier old reference z free failed");
  cuda_check(cudaFree(d_ref_r_old), "reference barrier old reference r free failed");
  return remap_result;
}

}  // namespace

std::uint64_t evaluate_reference_barrier_trigger(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg) {
  std::uint64_t mask = 0;
  if (!cfg.numerics.ale.reference_barrier_enabled) {
    return mask;
  }
  if (cfg.numerics.ale.reference_force_engage_every_step) {
    return static_cast<std::uint64_t>(
        ReferenceBarrierAleResult::TriggerReason::AxisMargin);
  }
  if (cfg.numerics.ale.reference_trigger_axis_margin_enabled) {
    const double margin = cfg.numerics.has_physical_rz_axis
                              ? compute_axis_margin_min_host(state)
                              : std::numeric_limits<double>::infinity();
    if (margin < cfg.numerics.ale.reference_trigger_axis_margin_threshold) {
      mask |= static_cast<std::uint64_t>(
          ReferenceBarrierAleResult::TriggerReason::AxisMargin);
    }
  }
  if (cfg.numerics.ale.reference_trigger_corner_j_ratio_enabled) {
    const double ratio = compute_corner_j_ratio_min_host(state);
    if (ratio < cfg.numerics.ale.reference_trigger_corner_j_ratio_threshold) {
      mask |= static_cast<std::uint64_t>(
          ReferenceBarrierAleResult::TriggerReason::CornerJRatio);
    }
  }
  return mask;
}

void build_reference_target_mesh(const tenryu::core::State& state,
                                 const tenryu::core::Config& cfg,
                                 double* d_target_r,
                                 double* d_target_z) {
  TENRYU_ASSERT(d_target_r != nullptr, "reference target r buffer is null");
  TENRYU_ASSERT(d_target_z != nullptr, "reference target z buffer is null");
  const int n_nodes = state.mesh.topo.n_nodes;
  if (cfg.numerics.ale.multiblock_scaled_reference_enabled &&
      tenryu::mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    const double alpha =
        tenryu::hydro::ale::compute_scale_factor_alpha(state, cfg);
    mesh_trace::trace_alpha_source(state, cfg, alpha);
    tenryu::hydro::ale::build_scaled_gamma_mvp_target(
        state, cfg, alpha, d_target_r, d_target_z);
    mesh_trace::trace_post_target(state, cfg, d_target_r, d_target_z);
    mesh_trace::trace_cell0_target(
        state, cfg, "refbar_scaled_reference", d_target_r, d_target_z);
    return;
  }
  if (cfg.numerics.ale.reference_target == "eulerian_initial") {
    TENRYU_ASSERT(state.x_r_initial.size() == state.x_r.size(),
                  "reference target requires x_r_initial");
    TENRYU_ASSERT(state.x_z_initial.size() == state.x_z.size(),
                  "reference target requires x_z_initial");
    cuda_check(cudaMemcpy(d_target_r,
                          state.x_r_initial.data(),
                          static_cast<std::size_t>(n_nodes) * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "reference target r copy failed");
    cuda_check(cudaMemcpy(d_target_z,
                          state.x_z_initial.data(),
                          static_cast<std::size_t>(n_nodes) * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "reference target z copy failed");
    return;
  }
  if (cfg.numerics.ale.reference_target == "spherical_equal_angle") {
    const int blocks = (n_nodes + 255) / 256;
    build_spherical_equal_angle_kernel<<<blocks, 256>>>(state.x_r_initial.data(),
                                                        state.x_z_initial.data(),
                                                        d_target_r,
                                                        d_target_z,
                                                        state.mesh.topo.nr,
                                                        state.mesh.topo.nz);
    cuda_check(cudaGetLastError(), "reference spherical target kernel launch failed");
    cuda_check(cudaDeviceSynchronize(),
               "reference spherical target kernel synchronize failed");
    return;
  }
  cuda_check(cudaMemcpy(d_target_r,
                        state.x_r.data(),
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "reference none target r copy failed");
  cuda_check(cudaMemcpy(d_target_z,
                        state.x_z.data(),
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "reference none target z copy failed");
}

ReferenceBarrierAleResult apply_reference_barrier_rezone(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const double* d_target_r,
    const double* d_target_z,
    const ReferenceBarrierScope* scope) {
  ReferenceBarrierAleResult result;
  // The trigger path gates on reference_barrier_enabled alone; the apply path also serves the button morph (S-C), which supplies its own targets.
  if (!cfg.numerics.ale.reference_barrier_enabled &&
      !cfg.numerics.ale.button_morph.enabled) {
    return result;
  }
  result.engaged = true;
  tenryu::hydro::ale::ale_velcoherence::sample(
      state, cfg, "s0_post_hydro");

  const bool is_multiblock = mesh::mesh_topo_is_multiblock(cfg.mesh);
  const int n_nodes = state.mesh.topo.n_nodes;
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = is_multiblock ? state.mesh.topo.n_cells : nr * nz;
  const int blocks_cells = (n_cells + 255) / 256;
  const int blocks_nodes = (n_nodes + 255) / 256;
  const std::size_t cell_bytes = static_cast<std::size_t>(n_cells) * sizeof(double);
  const std::size_t node_bytes = static_cast<std::size_t>(n_nodes) * sizeof(double);
  double* d_delta_r = nullptr;
  double* d_delta_z = nullptr;
  double* d_xr_old = nullptr;
  double* d_xz_old = nullptr;
  double* d_vol_old = nullptr;
  double* d_tmp = nullptr;
  double* d_vol_mid = nullptr;
  double* d_mom_r = nullptr;
  double* d_mom_z = nullptr;
  double* d_e_e = nullptr;
  double* d_e_i = nullptr;
  double* d_vr_cell = nullptr;
  double* d_vz_cell = nullptr;
  double* d_group = nullptr;
  std::uint8_t* d_node_flags = upload_node_flags_if_constraints(state);
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_delta_r),
                        node_bytes),
             "reference barrier delta_r allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_delta_z),
                        node_bytes),
             "reference barrier delta_z allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_xr_old), node_bytes),
             "reference barrier old r allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_xz_old), node_bytes),
             "reference barrier old z allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_vol_old), cell_bytes),
             "reference barrier old volume allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_tmp), cell_bytes),
             "reference barrier remap tmp allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_vol_mid), cell_bytes),
             "reference barrier remap volume tmp allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mom_r), cell_bytes),
             "reference barrier mom_r allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mom_z), cell_bytes),
             "reference barrier mom_z allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_e_e), cell_bytes),
             "reference barrier e_e allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_e_i), cell_bytes),
             "reference barrier e_i allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_vr_cell), cell_bytes),
             "reference barrier v_r_cell allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_vz_cell), cell_bytes),
             "reference barrier v_z_cell allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_group), cell_bytes),
             "reference barrier group scratch allocation failed");
  cuda_check(cudaMemcpy(d_xr_old, state.x_r.data(), node_bytes, cudaMemcpyDeviceToDevice),
             "reference barrier old r copy failed");
  cuda_check(cudaMemcpy(d_xz_old, state.x_z.data(), node_bytes, cudaMemcpyDeviceToDevice),
             "reference barrier old z copy failed");
  cuda_check(cudaMemcpy(d_vol_old, state.vol.data(), cell_bytes, cudaMemcpyDeviceToDevice),
             "reference barrier old volume copy failed");
  const ConservedSums sums_before = compute_conserved_sums_host(state);
  mesh_trace::trace_cell0_geometry(state, cfg, "refbar_pre_rezone");

  delta_kernel<<<blocks_nodes, 256>>>(state.x_r.data(),
                                      state.x_z.data(),
                                      d_target_r,
                                      d_target_z,
                                      d_delta_r,
                                      d_delta_z,
                                      n_nodes);
  cuda_check(cudaGetLastError(), "reference barrier delta kernel launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "reference barrier delta kernel synchronize failed");

  tenryu::mesh::CandidateMeshAdmissibilityFloors floors;
  floors.volume_rel = cfg.numerics.ale.reference_volume_floor_rel;
  floors.corner_j_rel = cfg.numerics.ale.reference_corner_j_floor_rel;
  floors.gauss_j_rel = cfg.numerics.ale.reference_gauss_j_floor_rel;
  int* d_cell_id_stable = nullptr;
  std::uint8_t* d_cell_nverts = nullptr;
  tenryu::mesh::LineSearchResult ls;
  if (is_multiblock) {
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "CSR reference barrier requires device cell-node CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                      static_cast<std::size_t>(n_cells) *
                          static_cast<std::size_t>(state.mesh.corner_stride),
                  "CSR reference barrier requires device cell-node CSR indices");
    d_cell_id_stable = upload_multiblock_stable_cell_ids(state);
    d_cell_nverts = upload_cell_nverts_if_active(state);
    ls = tenryu::mesh::linesearch_largest_admissible_sigma_csr(
        state.x_r.data(),
        state.x_z.data(),
        d_delta_r,
        d_delta_z,
        cfg.numerics.ale.reference_blend_default,
        0.0,
        cfg.numerics.ale.reference_linesearch_max_iters,
        n_cells,
        state.mesh.corner_stride,
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        d_cell_id_stable,
        floors,
        d_cell_nverts,
        nullptr,
        nullptr,
        0,
        state.x_r_reference.size() == state.x_r.size() &&
                state.x_z_reference.size() == state.x_z.size()
            ? state.x_r_reference.data()
            : nullptr,
        state.x_r_reference.size() == state.x_r.size() &&
                state.x_z_reference.size() == state.x_z.size()
            ? state.x_z_reference.data()
            : nullptr);
  } else {
    ls = tenryu::mesh::linesearch_largest_admissible_sigma(
        state.x_r.data(),
        state.x_z.data(),
        d_delta_r,
        d_delta_z,
        cfg.numerics.ale.reference_blend_default,
        0.0,
        cfg.numerics.ale.reference_linesearch_max_iters,
        nr,
        nz,
        floors,
        nullptr,
        state.x_r_reference.size() == state.x_r.size() &&
                state.x_z_reference.size() == state.x_z.size()
            ? state.x_r_reference.data()
            : nullptr,
        state.x_r_reference.size() == state.x_r.size() &&
                state.x_z_reference.size() == state.x_z.size()
            ? state.x_z_reference.data()
            : nullptr);
  }
  result.lambda_accepted = ls.sigma_accepted;
  result.linesearch_iters = ls.iters_used;
  result.final_quality = ls.quality;

  if (ls.sigma_accepted > 0.0) {
    if (is_multiblock) {
      const auto remap_result =
          apply_multiblock_csr_reference_remap(state,
                                               cfg,
                                               d_xr_old,
                                               d_xz_old,
                                               d_vol_old,
                                               d_delta_r,
                                               d_delta_z,
                                               ls.sigma_accepted,
                                               scope);
      result.mass_floor_delta += remap_result.mass_floor_delta;
      result.E_floor_injected += remap_result.E_floor_injected;
      result.E_redistribution_unresolved +=
          remap_result.E_redistribution_unresolved;
      const bool remap_ok = remap_result.applied;
      result.succeeded = remap_ok;
      if (remap_ok) {
        const ConservedSums sums_after = compute_conserved_sums_host(state);
        log_reference_conservation(state.step, sums_before, sums_after);
      }
    } else {
    pack_cell_conserved_kernel<<<blocks_cells, 256>>>(d_mom_r,
                                                       d_mom_z,
                                                       d_e_e,
                                                       d_e_i,
                                                       d_vr_cell,
                                                       d_vz_cell,
                                                       state.rho.data(),
                                                       state.ee.data(),
                                                       state.ei.data(),
                                                       state.v_r.data(),
                                                       state.v_z.data(),
                                                       nr,
                                                       nz);
    cuda_check(cudaGetLastError(), "reference barrier pack conserved kernel launch failed");

    blend_kernel<<<blocks_nodes, 256>>>(state.x_r.data(),
                                        state.x_z.data(),
                                        d_delta_r,
                                        d_delta_z,
                                        ls.sigma_accepted,
                                        n_nodes);
    cuda_check(cudaGetLastError(), "reference barrier blend kernel launch failed");
    cuda_check(cudaDeviceSynchronize(),
               "reference barrier blend kernel synchronize failed");
    tenryu::hydro::ale::core_freeze::restore_target_if_enabled(
        state,
        cfg,
        state.x_r.data(),
        state.x_z.data(),
        d_xr_old,
        d_xz_old,
        false,
        "reference_barrier_structured");
    state.mesh.recompute_geometry();
    state.vol = state.mesh.cell_vol;

    const bool donor_sign_fixed = cfg.numerics.ale.swept_volume_sign_fixed;
    const auto remap_scalar = [&](double* field) {
      return ale::launch_remap_strang(field,
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

    bool remap_ok = remap_scalar(state.rho.data());
    remap_ok = remap_ok && remap_scalar(d_mom_r);
    remap_ok = remap_ok && remap_scalar(d_mom_z);
    remap_ok = remap_ok && remap_scalar(d_e_e);
    remap_ok = remap_ok && remap_scalar(d_e_i);

    const int n_mat = static_cast<int>(cfg.materials.materials.size());
    if (remap_ok && n_mat > 0 &&
        state.volFrac.size() == static_cast<std::size_t>(n_cells * n_mat)) {
      for (int m = 0; m < n_mat && remap_ok; ++m) {
        gather_material_field_kernel<<<blocks_cells, 256>>>(
            d_group, state.volFrac.data(), n_cells, n_mat, m);
        cuda_check(cudaGetLastError(),
                   "reference barrier gather volFrac kernel launch failed");
        remap_ok = remap_scalar(d_group);
        if (remap_ok) {
          scatter_material_field_kernel<<<blocks_cells, 256>>>(
              state.volFrac.data(), d_group, n_cells, n_mat, m);
          cuda_check(cudaGetLastError(),
                     "reference barrier scatter volFrac kernel launch failed");
        }
      }
      if (remap_ok) {
        normalize_reference_volfrac_kernel<<<blocks_cells, 256>>>(
            state.volFrac.data(), n_cells, n_mat);
        cuda_check(cudaGetLastError(),
                   "reference barrier normalize volFrac kernel launch failed");
      }
    }

    const int n_groups = cfg.radiation.groups;
    if (remap_ok && n_groups > 0 &&
        state.rad_E.size() == static_cast<std::size_t>(n_cells * n_groups)) {
      for (int g = 0; g < n_groups && remap_ok; ++g) {
        gather_group_field_kernel<<<blocks_cells, 256>>>(
            d_group, state.rad_E.data(), n_cells, n_groups, g);
        cuda_check(cudaGetLastError(),
                   "reference barrier gather rad_E kernel launch failed");
        remap_ok = remap_scalar(d_group);
        if (remap_ok) {
          scatter_group_field_kernel<<<blocks_cells, 256>>>(
              state.rad_E.data(), d_group, n_cells, n_groups, g);
          cuda_check(cudaGetLastError(),
                     "reference barrier scatter rad_E kernel launch failed");
        }
      }
    }

    if (remap_ok) {
      recover_cell_primitives_kernel<<<blocks_cells, 256>>>(state.rho.data(),
                                                             state.mass.data(),
                                                             state.ee.data(),
                                                             state.ei.data(),
                                                             d_vr_cell,
                                                             d_vz_cell,
                                                             d_mom_r,
                                                             d_mom_z,
                                                             d_e_e,
                                                             d_e_i,
                                                             state.vol.data(),
                                                             cfg.numerics.floors.rho,
                                                             n_cells);
      cuda_check(cudaGetLastError(),
                 "reference barrier recover primitive kernel launch failed");

      const auto r_outer_type =
          parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_outer);
      const auto z_bottom_type =
          parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_bottom);
      const auto z_top_type =
          parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_top);
      project_reference_cell_velocity_to_nodes_kernel<<<blocks_nodes, 256>>>(
          state.v_r.data(),
          state.v_z.data(),
          d_vr_cell,
          d_vz_cell,
          state.rho.data(),
          state.vol.data(),
          d_node_flags,
          nr,
          nz,
          velocity_bc_mode(r_outer_type),
          velocity_bc_mode(z_bottom_type),
          velocity_bc_mode(z_top_type));
      cuda_check(cudaGetLastError(),
                 "reference barrier velocity projection kernel launch failed");
      cuda_check(cudaDeviceSynchronize(),
                 "reference barrier remap synchronize failed");
      const ConservedSums sums_after = compute_conserved_sums_host(state);
      log_reference_conservation(state.step, sums_before, sums_after);
    } else {
      core::log_warning(
          "[ref-barrier-ale] remap aborted: non-positive intermediate volume");
    }
    result.succeeded = remap_ok;
    }
  } else if (is_multiblock) {
    tenryu::hydro::ale::ale_velcoherence::sample(
        state, cfg, "s1_post_rezone");
    tenryu::hydro::ale::ale_velcoherence::sample(
        state, cfg, "s2_post_remap");
    tenryu::hydro::ale::ale_velcoherence::sample(
        state, cfg, "s3_post_velproj");
  }

  if (d_cell_id_stable != nullptr) {
    cuda_check(cudaFree(d_cell_id_stable),
               "reference barrier stable-cell free failed");
  }
  if (d_cell_nverts != nullptr) {
    cuda_check(cudaFree(d_cell_nverts),
               "reference barrier cell_nverts free failed");
  }
  cuda_check(cudaFree(d_group), "reference barrier group scratch free failed");
  if (d_node_flags != nullptr) {
    cuda_check(cudaFree(d_node_flags), "reference barrier node_flags free failed");
  }
  cuda_check(cudaFree(d_vz_cell), "reference barrier v_z_cell free failed");
  cuda_check(cudaFree(d_vr_cell), "reference barrier v_r_cell free failed");
  cuda_check(cudaFree(d_e_i), "reference barrier e_i free failed");
  cuda_check(cudaFree(d_e_e), "reference barrier e_e free failed");
  cuda_check(cudaFree(d_mom_z), "reference barrier mom_z free failed");
  cuda_check(cudaFree(d_mom_r), "reference barrier mom_r free failed");
  cuda_check(cudaFree(d_vol_mid), "reference barrier remap volume tmp free failed");
  cuda_check(cudaFree(d_tmp), "reference barrier remap tmp free failed");
  cuda_check(cudaFree(d_vol_old), "reference barrier old volume free failed");
  cuda_check(cudaFree(d_xz_old), "reference barrier old z free failed");
  cuda_check(cudaFree(d_xr_old), "reference barrier old r free failed");
  cuda_check(cudaFree(d_delta_z), "reference barrier delta_z free failed");
  cuda_check(cudaFree(d_delta_r), "reference barrier delta_r free failed");
  return result;
}

}  // namespace tenryu::hydro
