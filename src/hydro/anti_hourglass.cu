#include "hydro/anti_hourglass.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "core/error.hpp"
#include "hydro/boundary_2d.hpp"
#include "hydro/central_pseudo_core.cuh"
#include "hydro/corner_jacobian_quality.cuh"
#include "hydro/pentagon_geometry.cuh"
#include "hydro/pole_angular_derefine.cuh"
#include "hydro/rz_corner_mass.cuh"
#include "mesh/mesh.hpp"

// Default-off 2D_RZ subzonal pressure anti-hourglass force.
// References:
// - Caramana, Burton, Shashkov, Whalen, JCP 146, 227-262 (1998).
// - Caramana and Shashkov, JCP 142, 521-561 (1998).
namespace tenryu::hydro {
namespace {

constexpr double kPiOverThree = 1.0471975511965977461542144610931676280657;
constexpr double kTiny = 1.0e-300;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline void sync_kernel(const char* message) {
  cuda_check(cudaGetLastError(), message);
  cuda_check(cudaDeviceSynchronize(), message);
}

bool has_tri_cell_nverts(const std::vector<std::uint8_t>& cell_nverts,
                         const int n_cells) {
  if (cell_nverts.size() != static_cast<std::size_t>(n_cells)) {
    return false;
  }
  return std::any_of(cell_nverts.begin(), cell_nverts.end(),
                     [](const std::uint8_t nverts) {
                       return nverts == 3U;
                     });
}

const std::uint8_t* upload_cell_nverts_if_tri(
    core::DeviceArray<std::uint8_t>& d_cell_nverts,
    const core::State& state,
    const int n_cells) {
  if (!has_tri_cell_nverts(state.mesh.cell_nverts, n_cells)) {
    return nullptr;
  }
  d_cell_nverts.reset(static_cast<std::size_t>(n_cells));
  d_cell_nverts.copy_from_host(state.mesh.cell_nverts);
  return d_cell_nverts.data();
}

const std::uint8_t* upload_cell_nverts_if_nonquad(
    core::DeviceArray<std::uint8_t>& d_cell_nverts,
    const core::State& state,
    const int n_cells) {
  if (state.mesh.cell_nverts.size() !=
          static_cast<std::size_t>(n_cells) ||
      std::none_of(state.mesh.cell_nverts.begin(),
                   state.mesh.cell_nverts.end(),
                   [](const std::uint8_t nverts) {
                     return nverts != 4U;
                   })) {
    return nullptr;
  }
  d_cell_nverts.reset(static_cast<std::size_t>(n_cells));
  d_cell_nverts.copy_from_host(state.mesh.cell_nverts);
  return d_cell_nverts.data();
}

// Returns the per-cell bridge-band weight for
// subzonal_band_mode="bridge_feather": w=1 for cells with multiblock block id
// 1; for other cells, w = quintic_smoothstep(1 -
// layer/(feather_layers+1)) where layer is the face-adjacency BFS distance (in
// cells) to the nearest bridge cell, w=0 for layer > feather_layers. Uses the
// host CSR (state.mesh.topo.multiblock->face_adj_csr_offsets/indices for
// neighbor cells and ->cell_block_id). BFS is deterministic (ascending cell
// order).
std::vector<double> build_bridge_band_weights(const core::State& state,
                                              const int feather_layers) {
  const int n_cells = static_cast<int>(state.rho.size());
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "bridge_feather subzonal band requires multiblock topology "
                "with cell block ids");
  const auto& mb = *state.mesh.topo.multiblock;
  TENRYU_ASSERT(mb.cell_block_id.size() == static_cast<std::size_t>(n_cells),
                "bridge_feather subzonal band requires one multiblock cell "
                "block id per cell");
  TENRYU_ASSERT(mb.face_adj_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U,
                "bridge_feather subzonal band requires face-adjacency CSR "
                "offsets");

  std::vector<int> layer(static_cast<std::size_t>(n_cells), -1);
  std::vector<double> weights(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    if (mb.cell_block_id[static_cast<std::size_t>(c)] == 1) {
      layer[static_cast<std::size_t>(c)] = 0;
      weights[static_cast<std::size_t>(c)] = 1.0;
    }
  }

  for (int next_layer = 1; next_layer <= feather_layers; ++next_layer) {
    for (int c = 0; c < n_cells; ++c) {
      if (layer[static_cast<std::size_t>(c)] >= 0) {
        continue;
      }
      const int off = mb.face_adj_csr_offsets[static_cast<std::size_t>(c)];
      const int end =
          mb.face_adj_csr_offsets[static_cast<std::size_t>(c) + 1U];
      for (int p = off; p < end; ++p) {
        const int neighbor =
            mb.face_adj_csr_indices[static_cast<std::size_t>(p)];
        if (neighbor >= 0 &&
            layer[static_cast<std::size_t>(neighbor)] == next_layer - 1) {
          layer[static_cast<std::size_t>(c)] = next_layer;
          break;
        }
      }
    }
  }

  for (int c = 0; c < n_cells; ++c) {
    const int cell_layer = layer[static_cast<std::size_t>(c)];
    if (cell_layer <= 0 || cell_layer > feather_layers) {
      continue;
    }
    const double u =
        1.0 - static_cast<double>(cell_layer) /
                  static_cast<double>(feather_layers + 1);
    const double u2 = u * u;
    const double u3 = u2 * u;
    const double u4 = u3 * u;
    const double u5 = u4 * u;
    weights[static_cast<std::size_t>(c)] =
        6.0 * u5 - 15.0 * u4 + 10.0 * u3;
  }
  return weights;
}

__device__ inline double atomic_add_double(double* address, const double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long>(__double_as_longlong(
                        val + __longlong_as_double(static_cast<long long>(assumed)))));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
#endif
}

__host__ __device__ inline int node_index_2d(const int i,
                                             const int j,
                                             const int nz) {
  return i * (nz + 1) + j;
}

__device__ inline bool button_or_dormant_center_cell(
    const int c,
    const int nz,
    const int button_outer_node_ring) {
  if (button_outer_node_ring <= 0 || nz <= 0) {
    return false;
  }
  if (c == 0) {
    return true;
  }
  const int i = c / nz;
  return i >= 0 && i < button_outer_node_ring;
}

__device__ inline void bilinear_hourglass_amplitude(const double* r,
                                                    const double* z,
                                                    double& ar,
                                                    double& az) {
  ar = 0.25 * (r[0] - r[1] + r[2] - r[3]);
  az = 0.25 * (z[0] - z[1] + z[2] - z[3]);
}

__device__ inline double planar_area(const double* r, const double* z) {
  double cross_sum = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int kp = (k + 1) & 3;
    cross_sum += r[k] * z[kp] - r[kp] * z[k];
  }
  return 0.5 * fabs(cross_sum);
}

__device__ inline double rz_polygon_volume(const double* r,
                                           const double* z,
                                           const int n) {
  double rc = 0.0;
  double zc = 0.0;
  for (int k = 0; k < n; ++k) {
    rc += r[k];
    zc += z[k];
  }
  rc /= static_cast<double>(n);
  zc /= static_cast<double>(n);

  double vol_sum = 0.0;
  for (int k = 0; k < n; ++k) {
    const int kp = (k + 1) % n;
    const double rk = r[k] - rc;
    const double rkp = r[kp] - rc;
    const double zk = z[k] - zc;
    const double zkp = z[kp] - zc;
    const double cross_local = rk * zkp - rkp * zk;
    const double weight = rk + rkp + 3.0 * rc;
    vol_sum += cross_local * weight;
  }
  return kPiOverThree * vol_sum;
}

__device__ inline void rz_quad_corner_svec(const double* r,
                                           const double* z,
                                           double& sr,
                                           double& sz) {
  const double rc = 0.25 * (r[0] + r[1] + r[2] + r[3]);
  const double zc = 0.25 * (z[0] + z[1] + z[2] + z[3]);

  const double trkm1 = r[3] - rc;
  const double trk = r[0] - rc;
  const double trkp1 = r[1] - rc;
  const double tzkm1 = z[3] - zc;
  const double tzk = z[0] - zc;
  const double tzkp1 = z[1] - zc;

  sr = kPiOverThree *
       (2.0 * trk * (tzkp1 - tzkm1) + trkp1 * (tzkp1 - tzk) +
        trkm1 * (tzk - tzkm1) + 3.0 * rc * (tzkp1 - tzkm1));
  sz = kPiOverThree *
       (trkm1 * (trkm1 + trk) - trkp1 * (trk + trkp1) +
        3.0 * rc * (trkm1 - trkp1));
}

__device__ inline void subzone_geometry(const double* r,
                                        const double* z,
                                        const int corner,
                                        double& volume,
                                        double& sr,
                                        double& sz) {
  const int kp = (corner + 1) & 3;
  const int km = (corner + 3) & 3;
  const double rc = 0.25 * (r[0] + r[1] + r[2] + r[3]);
  const double zc = 0.25 * (z[0] + z[1] + z[2] + z[3]);
  double rq[4] = {
      r[corner],
      0.5 * (r[corner] + r[kp]),
      rc,
      0.5 * (r[km] + r[corner]),
  };
  double zq[4] = {
      z[corner],
      0.5 * (z[corner] + z[kp]),
      zc,
      0.5 * (z[km] + z[corner]),
  };
  volume = rz_polygon_volume(rq, zq, 4);
  rz_quad_corner_svec(rq, zq, sr, sz);
}

__device__ inline void triangle_corner_volumes(const double* r,
                                               const double* z,
                                               double* v_corner) {
  const double rc = (r[0] + r[1] + r[2]) / 3.0;
  const double zc = (z[0] + z[1] + z[2]) / 3.0;
  for (int k = 0; k < 3; ++k) {
    const int kp = (k + 1) % 3;
    const int km = (k + 2) % 3;
    const double rq[4] = {
        r[k],
        0.5 * (r[k] + r[kp]),
        rc,
        0.5 * (r[km] + r[k]),
    };
    const double zq[4] = {
        z[k],
        0.5 * (z[k] + z[kp]),
        zc,
        0.5 * (z[km] + z[k]),
    };
    v_corner[k] = fabs(rz_polygon_volume(rq, zq, 4));
  }
  v_corner[3] = 0.0;
}

__device__ inline void validate_pentagon_corner_volumes(
    const double* v_corner) {
  for (int k = 0; k < 5; ++k) {
    if (!(v_corner[k] > 0.0) || !isfinite(v_corner[k])) {
      printf("pentagon subzonal corner volume invalid\n");
      __trap();
    }
  }
}

__global__ void initialize_subzonal_masses_kernel(
    double* __restrict__ m0,
    double* __restrict__ m1,
    double* __restrict__ m2,
    double* __restrict__ m3,
    double* __restrict__ corner_mass,
    double* __restrict__ corner_volume,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int nr,
    const int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int n00 = node_index_2d(i, j, nz);
  const int n10 = node_index_2d(i + 1, j, nz);
  const int n11 = node_index_2d(i + 1, j + 1, nz);
  const int n01 = node_index_2d(i, j + 1, nz);

  const double m_cell = fmax(mass[c], 0.0);
  double v_corner[4] = {0.0, 0.0, 0.0, 0.0};
  rz::compute_quad_corner_volumes_exact_subpolygon(x_r[n00],
                                                   x_z[n00],
                                                   x_r[n10],
                                                   x_z[n10],
                                                   x_r[n11],
                                                   x_z[n11],
                                                   x_r[n01],
                                                   x_z[n01],
                                                   v_corner);
  const double v_sum = v_corner[0] + v_corner[1] + v_corner[2] + v_corner[3];
  double cm[4] = {0.0, 0.0, 0.0, 0.0};
  if (!(isfinite(v_sum) && v_sum > 0.0)) {
    const double q = 0.25 * m_cell;
    cm[0] = q;
    cm[1] = q;
    cm[2] = q;
    cm[3] = q;
  } else {
    cm[0] = m_cell * (v_corner[0] / v_sum);
    cm[1] = m_cell * (v_corner[1] / v_sum);
    cm[2] = m_cell * (v_corner[2] / v_sum);
    cm[3] = m_cell * (v_corner[3] / v_sum);
  }
  m0[c] = cm[0];
  m1[c] = cm[1];
  m2[c] = cm[2];
  m3[c] = cm[3];
  if (corner_mass != nullptr) {
    corner_mass[c * 4 + 0] = cm[0];
    corner_mass[c * 4 + 1] = cm[1];
    corner_mass[c * 4 + 2] = cm[2];
    corner_mass[c * 4 + 3] = cm[3];
  }
  if (corner_volume != nullptr) {
    corner_volume[c * 4 + 0] = v_corner[0];
    corner_volume[c * 4 + 1] = v_corner[1];
    corner_volume[c * 4 + 2] = v_corner[2];
    corner_volume[c * 4 + 3] = v_corner[3];
  }
}

__global__ void initialize_subzonal_masses_multiblock_kernel(
    double* __restrict__ m0,
    double* __restrict__ m1,
    double* __restrict__ m2,
    double* __restrict__ m3,
    double* __restrict__ corner_mass,
    double* __restrict__ corner_volume,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int corner_stride,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  const int off = cell_node_csr_offsets[c];
  const double m_cell = fmax(mass[c], 0.0);
  double v_corner[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  double cm[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  if (active_nverts == 3) {
    const int n0 = cell_node_csr_indices[off + 0];
    const int n1 = cell_node_csr_indices[off + 1];
    const int n2 = cell_node_csr_indices[off + 2];
    const double r[4] = {x_r[n0], x_r[n1], x_r[n2], 0.0};
    const double z[4] = {x_z[n0], x_z[n1], x_z[n2], 0.0};
    triangle_corner_volumes(r, z, v_corner);
    const double v_sum = v_corner[0] + v_corner[1] + v_corner[2];
    if (isfinite(v_sum) && v_sum > 0.0) {
      cm[0] = m_cell * (v_corner[0] / v_sum);
      cm[1] = m_cell * (v_corner[1] / v_sum);
      cm[2] = m_cell * (v_corner[2] / v_sum);
    } else {
      const double third = m_cell / 3.0;
      cm[0] = third;
      cm[1] = third;
      cm[2] = third;
    }
  } else if (active_nverts == 5) {
    PentagonPoint x[5];
    for (int k = 0; k < 5; ++k) {
      const int n = cell_node_csr_indices[off + k];
      x[k] = {x_r[n], x_z[n]};
    }
    pentagon_corner_rz_volumes(x, v_corner);
    validate_pentagon_corner_volumes(v_corner);
    double v_sum = 0.0;
    for (int k = 0; k < 5; ++k) {
      v_sum += v_corner[k];
    }
    for (int k = 0; k < 5; ++k) {
      cm[k] = m_cell * (v_corner[k] / v_sum);
    }
    cm[2] =
        m_cell - ((cm[0] + cm[1]) + (cm[3] + cm[4]));
  } else {
    if (active_nverts != 4) {
#ifdef __CUDA_ARCH__
      __trap();
#else
      ::tenryu::core::tenryu_abort(
          "active_nverts == 4",
          "hourglass subzonal mass: general-valence cell reached the quad branch",
          __FILE__, __LINE__);
#endif
    }
    const int n00 = cell_node_csr_indices[off + 0];
    const int n10 = cell_node_csr_indices[off + 1];
    const int n11 = cell_node_csr_indices[off + 2];
    const int n01 = cell_node_csr_indices[off + 3];
    rz::compute_quad_corner_volumes_exact_subpolygon(x_r[n00],
                                                     x_z[n00],
                                                     x_r[n10],
                                                     x_z[n10],
                                                     x_r[n11],
                                                     x_z[n11],
                                                     x_r[n01],
                                                     x_z[n01],
                                                     v_corner);
    const double v_sum = v_corner[0] + v_corner[1] + v_corner[2] + v_corner[3];
    cm[0] = 0.25 * m_cell;
    cm[1] = 0.25 * m_cell;
    cm[2] = 0.25 * m_cell;
    cm[3] = 0.25 * m_cell;
    if (isfinite(v_sum) && v_sum > 0.0) {
      cm[0] = m_cell * (v_corner[0] / v_sum);
      cm[1] = m_cell * (v_corner[1] / v_sum);
      cm[2] = m_cell * (v_corner[2] / v_sum);
      cm[3] = m_cell * (v_corner[3] / v_sum);
    }
  }
  m0[c] = cm[0];
  m1[c] = cm[1];
  m2[c] = cm[2];
  m3[c] = cm[3];
  const int base = c * corner_stride;
  if (corner_mass != nullptr) {
    for (int k = 0; k < active_nverts; ++k) {
      corner_mass[base + k] = cm[k];
    }
    for (int k = active_nverts; k < corner_stride; ++k) {
      corner_mass[base + k] = 0.0;
    }
  }
  if (corner_volume != nullptr) {
    for (int k = 0; k < active_nverts; ++k) {
      corner_volume[base + k] = v_corner[k];
    }
    for (int k = active_nverts; k < corner_stride; ++k) {
      corner_volume[base + k] = 0.0;
    }
  }
}

__global__ void copy_corner_mass_to_subzonal_kernel(
    double* __restrict__ m0,
    double* __restrict__ m1,
    double* __restrict__ m2,
    double* __restrict__ m3,
    const double* __restrict__ corner_mass,
    const std::uint8_t* __restrict__ cell_nverts,
    const int corner_stride,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const int base = c * corner_stride;
  m0[c] = corner_mass[base + 0];
  m1[c] = corner_mass[base + 1];
  m2[c] = corner_mass[base + 2];
  if (mesh::mesh_topo_cell_active_nverts(cell_nverts, c) == 3) {
    m3[c] = 0.0;
  } else {
    m3[c] = corner_mass[base + 3];
  }
}

__global__ void update_corner_volumes_kernel(
    double* __restrict__ corner_volume,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int nr,
    const int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int n00 = node_index_2d(i, j, nz);
  const int n10 = node_index_2d(i + 1, j, nz);
  const int n11 = node_index_2d(i + 1, j + 1, nz);
  const int n01 = node_index_2d(i, j + 1, nz);
  double v_corner[4] = {0.0, 0.0, 0.0, 0.0};
  rz::compute_quad_corner_volumes_exact_subpolygon(x_r[n00],
                                                   x_z[n00],
                                                   x_r[n10],
                                                   x_z[n10],
                                                   x_r[n11],
                                                   x_z[n11],
                                                   x_r[n01],
                                                   x_z[n01],
                                                   v_corner);
  corner_volume[c * 4 + 0] = v_corner[0];
  corner_volume[c * 4 + 1] = v_corner[1];
  corner_volume[c * 4 + 2] = v_corner[2];
  corner_volume[c * 4 + 3] = v_corner[3];
}

__global__ void update_corner_volumes_multiblock_kernel(
    double* __restrict__ corner_volume,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int corner_stride,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const int off = cell_node_csr_offsets[c];
  double v_corner[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  if (active_nverts == 3) {
    const int n0 = cell_node_csr_indices[off + 0];
    const int n1 = cell_node_csr_indices[off + 1];
    const int n2 = cell_node_csr_indices[off + 2];
    const double r[4] = {x_r[n0], x_r[n1], x_r[n2], 0.0};
    const double z[4] = {x_z[n0], x_z[n1], x_z[n2], 0.0};
    triangle_corner_volumes(r, z, v_corner);
  } else if (active_nverts == 5) {
    PentagonPoint x[5];
    for (int k = 0; k < 5; ++k) {
      const int n = cell_node_csr_indices[off + k];
      x[k] = {x_r[n], x_z[n]};
    }
    pentagon_corner_rz_volumes(x, v_corner);
    validate_pentagon_corner_volumes(v_corner);
  } else {
    if (active_nverts != 4) {
#ifdef __CUDA_ARCH__
      __trap();
#else
      ::tenryu::core::tenryu_abort(
          "active_nverts == 4",
          "hourglass corner volume: general-valence cell reached the quad branch",
          __FILE__, __LINE__);
#endif
    }
    const int n00 = cell_node_csr_indices[off + 0];
    const int n10 = cell_node_csr_indices[off + 1];
    const int n11 = cell_node_csr_indices[off + 2];
    const int n01 = cell_node_csr_indices[off + 3];
    rz::compute_quad_corner_volumes_exact_subpolygon(x_r[n00],
                                                     x_z[n00],
                                                     x_r[n10],
                                                     x_z[n10],
                                                     x_r[n11],
                                                     x_z[n11],
                                                     x_r[n01],
                                                     x_z[n01],
                                                     v_corner);
  }
  const int base = c * corner_stride;
  for (int k = 0; k < active_nverts; ++k) {
    corner_volume[base + k] = v_corner[k];
  }
  for (int k = active_nverts; k < corner_stride; ++k) {
    corner_volume[base + k] = 0.0;
  }
}

void warn_invariant_remap_reinit_once() {
  static bool warned = false;
  if (warned) {
    return;
  }
  warned = true;
  core::log_warning(
      "Lagrangian-invariant corner_mass is being reinitialized after ALE remap; "
      "conservative subzonal mass remap is deferred to Stage F");
}

void update_corner_volumes(core::State& state,
                           const int blocks,
                           const std::size_t expected,
                           const std::size_t expected_corner) {
  if (state.mesh.topo.multiblock.has_value()) {
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                      expected + 1U,
                  "Hydro2D hourglass corner volumes require cell-node CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                      expected_corner,
                  "Hydro2D hourglass corner volumes require cell-node CSR indices");
    core::DeviceArray<std::uint8_t> d_cell_nverts;
    const std::uint8_t* d_cell_nverts_ptr = upload_cell_nverts_if_nonquad(
        d_cell_nverts, state, static_cast<int>(expected));
    update_corner_volumes_multiblock_kernel<<<blocks, 256>>>(
        state.corner_volume.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        d_cell_nverts_ptr,
        state.corner_stride,
        static_cast<int>(expected));
  } else {
    update_corner_volumes_kernel<<<blocks, 256>>>(
        state.corner_volume.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.mesh.topo.nr,
        state.mesh.topo.nz);
  }
  sync_kernel("Hydro2D update hourglass corner volumes failed");
}

__global__ void compute_anti_hourglass_force_kernel(
    double* __restrict__ force_r_hg,
    double* __restrict__ force_z_hg,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ rho_cell,
    const double* __restrict__ p_cell,
    const double* __restrict__ subzonal_m0,
    const double* __restrict__ subzonal_m1,
    const double* __restrict__ subzonal_m2,
    const double* __restrict__ subzonal_m3,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const double scale,
    const double activation_corner_j_ratio_threshold,
    const double activation_hourglass_amplitude_threshold,
    const int stiffness_mode,
    const int button_outer_node_ring,
    const double dt) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }
  if (button_or_dormant_center_cell(c, nz, button_outer_node_ring)) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int nodes[4] = {
      node_index_2d(i, j, nz),
      node_index_2d(i + 1, j, nz),
      node_index_2d(i + 1, j + 1, nz),
      node_index_2d(i, j + 1, nz),
  };
  double r[4] = {x_r[nodes[0]], x_r[nodes[1]], x_r[nodes[2]], x_r[nodes[3]]};
  double z[4] = {x_z[nodes[0]], x_z[nodes[1]], x_z[nodes[2]], x_z[nodes[3]]};

  double j_min = corner_jacobian_from_quad(r, z, 0);
  double j_max = j_min;
  for (int k = 1; k < 4; ++k) {
    const double jk = corner_jacobian_from_quad(r, z, k);
    j_min = fmin(j_min, jk);
    j_max = fmax(j_max, jk);
  }
  if (!(isfinite(j_min) && isfinite(j_max) && j_max > kTiny)) {
    return;
  }
  if (j_min / j_max >= activation_corner_j_ratio_threshold) {
    return;
  }

  double a_r = 0.0;
  double a_z = 0.0;
  bilinear_hourglass_amplitude(r, z, a_r, a_z);
  const double cell_size = sqrt(fmax(planar_area(r, z), 0.0));
  const double a_norm = hypot(a_r, a_z);
  if (!(cell_size > kTiny) ||
      a_norm / cell_size < activation_hourglass_amplitude_threshold) {
    return;
  }

  const double rho = fmax(rho_cell[c], kTiny);
  const double p = fmax(p_cell[c], 0.0);
  const double dp_drho_e = p / rho;
  if (stiffness_mode == 0 && !(dp_drho_e > 0.0)) {
    return;
  }

  const double sub_m[4] = {
      fmax(subzonal_m0[c], 0.0),
      fmax(subzonal_m1[c], 0.0),
      fmax(subzonal_m2[c], 0.0),
      fmax(subzonal_m3[c], 0.0),
  };
  double fr[4] = {};
  double fz[4] = {};
  if (stiffness_mode != 0) {
    if (!(dt > 0.0)) {
      return;
    }
    const double inv_dt2 = 1.0 / (dt * dt);
    for (int k = 0; k < 4; ++k) {
      const double sign = (k == 0 || k == 2) ? 1.0 : -1.0;
      fr[k] = -scale * sub_m[k] * sign * a_r * inv_dt2;
      fz[k] = -scale * sub_m[k] * sign * a_z * inv_dt2;
    }
  } else {
    for (int k = 0; k < 4; ++k) {
      double v_sub = 0.0;
      double sr = 0.0;
      double sz = 0.0;
      subzone_geometry(r, z, k, v_sub, sr, sz);
      if (!(isfinite(v_sub) && fabs(v_sub) > kTiny)) {
        return;
      }
      const double rho_sub = sub_m[k] / fmax(fabs(v_sub), kTiny);
      const double dp_sub = dp_drho_e * (rho_sub - rho);
      fr[k] = scale * dp_sub * sr;
      fz[k] = scale * dp_sub * sz;
    }
  }

  const double mean_fr = 0.25 * (fr[0] + fr[1] + fr[2] + fr[3]);
  const double mean_fz = 0.25 * (fz[0] + fz[1] + fz[2] + fz[3]);
  double modal_work = 0.0;
  for (int k = 0; k < 4; ++k) {
    fr[k] -= mean_fr;
    fz[k] -= mean_fz;
    const double sign = (k == 0 || k == 2) ? 1.0 : -1.0;
    modal_work += fr[k] * (sign * a_r) + fz[k] * (sign * a_z);
  }
  // Caramana-Shashkov subzonal pressures should remove, not feed, the
  // checkerboard mode. The sign guard protects the implementation convention
  // for RZ area vectors while preserving zero response for inactive cells.
  const double orient = (modal_work > 0.0) ? -1.0 : 1.0;
  for (int k = 0; k < 4; ++k) {
    atomic_add_double(force_r_hg + nodes[k], orient * fr[k]);
    atomic_add_double(force_z_hg + nodes[k], orient * fz[k]);
  }
}

__global__ void compute_anti_hourglass_force_multiblock_kernel(
    double* __restrict__ force_r_hg,
    double* __restrict__ force_z_hg,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ rho_cell,
    const double* __restrict__ p_cell,
    const double* __restrict__ corner_mass,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells,
    const double scale,
    const double activation_corner_j_ratio_threshold,
    const double activation_hourglass_amplitude_threshold,
    const int stiffness_mode,
    const double dt) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  if (active_nverts == 3) {
    // Triangles have no bilinear hourglass mode.
    return;
  }

  const int off = cell_node_csr_offsets[c];
  const int nodes[4] = {
      cell_node_csr_indices[off + 0],
      cell_node_csr_indices[off + 1],
      cell_node_csr_indices[off + 2],
      cell_node_csr_indices[off + 3],
  };
  double r[4] = {x_r[nodes[0]], x_r[nodes[1]], x_r[nodes[2]], x_r[nodes[3]]};
  double z[4] = {x_z[nodes[0]], x_z[nodes[1]], x_z[nodes[2]], x_z[nodes[3]]};

  double j_min = fabs(corner_jacobian_from_quad(r, z, 0));
  double j_max = j_min;
  for (int k = 1; k < 4; ++k) {
    const double jk = fabs(corner_jacobian_from_quad(r, z, k));
    j_min = fmin(j_min, jk);
    j_max = fmax(j_max, jk);
  }
  if (!(isfinite(j_min) && isfinite(j_max) && j_max > kTiny)) {
    return;
  }
  if (j_min / j_max >= activation_corner_j_ratio_threshold) {
    return;
  }

  double a_r = 0.0;
  double a_z = 0.0;
  bilinear_hourglass_amplitude(r, z, a_r, a_z);
  const double cell_size = sqrt(fmax(planar_area(r, z), 0.0));
  const double a_norm = hypot(a_r, a_z);
  if (!(cell_size > kTiny) ||
      a_norm / cell_size < activation_hourglass_amplitude_threshold) {
    return;
  }

  const double rho = fmax(rho_cell[c], kTiny);
  const double p = fmax(p_cell[c], 0.0);
  const double dp_drho_e = p / rho;
  if (stiffness_mode == 0 && !(dp_drho_e > 0.0)) {
    return;
  }

  const double sub_m[4] = {
      fmax(corner_mass[c * 4 + 0], 0.0),
      fmax(corner_mass[c * 4 + 1], 0.0),
      fmax(corner_mass[c * 4 + 2], 0.0),
      fmax(corner_mass[c * 4 + 3], 0.0),
  };
  double fr[4] = {};
  double fz[4] = {};
  if (stiffness_mode != 0) {
    if (!(dt > 0.0)) {
      return;
    }
    const double inv_dt2 = 1.0 / (dt * dt);
    for (int k = 0; k < 4; ++k) {
      const double sign = (k == 0 || k == 2) ? 1.0 : -1.0;
      fr[k] = -scale * sub_m[k] * sign * a_r * inv_dt2;
      fz[k] = -scale * sub_m[k] * sign * a_z * inv_dt2;
    }
  } else {
    for (int k = 0; k < 4; ++k) {
      double v_sub = 0.0;
      double sr = 0.0;
      double sz = 0.0;
      subzone_geometry(r, z, k, v_sub, sr, sz);
      if (!(isfinite(v_sub) && fabs(v_sub) > kTiny)) {
        return;
      }
      const double rho_sub = sub_m[k] / fmax(fabs(v_sub), kTiny);
      const double dp_sub = dp_drho_e * (rho_sub - rho);
      fr[k] = scale * dp_sub * sr;
      fz[k] = scale * dp_sub * sz;
    }
  }

  const double mean_fr = 0.25 * (fr[0] + fr[1] + fr[2] + fr[3]);
  const double mean_fz = 0.25 * (fz[0] + fz[1] + fz[2] + fz[3]);
  double modal_work = 0.0;
  for (int k = 0; k < 4; ++k) {
    fr[k] -= mean_fr;
    fz[k] -= mean_fz;
    const double sign = (k == 0 || k == 2) ? 1.0 : -1.0;
    modal_work += fr[k] * (sign * a_r) + fz[k] * (sign * a_z);
  }
  const double orient = (modal_work > 0.0) ? -1.0 : 1.0;
  for (int k = 0; k < 4; ++k) {
    atomic_add_double(force_r_hg + nodes[k], orient * fr[k]);
    atomic_add_double(force_z_hg + nodes[k], orient * fz[k]);
  }
}

__global__ void limit_force_to_accel_kernel(
    double* __restrict__ accel_r_hg,
    double* __restrict__ accel_z_hg,
    const double* __restrict__ force_r_hg,
    const double* __restrict__ force_z_hg,
    const double* __restrict__ node_mass,
    const double* __restrict__ pressure_accel_r,
    const double* __restrict__ pressure_accel_z,
    const int nr,
    const int nz,
    const int r_outer_type,
    const int z_bottom_type,
    const int z_top_type,
    const double max_force_per_node_fraction) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_nodes = (nr + 1) * (nz + 1);
  if (n >= n_nodes) {
    return;
  }
  const double m = node_mass[n];
  if (!(m > 0.0)) {
    accel_r_hg[n] = 0.0;
    accel_z_hg[n] = 0.0;
    return;
  }
  double fr = force_r_hg[n];
  double fz = force_z_hg[n];
  const double f_mag = hypot(fr, fz);
  const double a_pressure = hypot(pressure_accel_r[n], pressure_accel_z[n]);
  const double f_cap = max_force_per_node_fraction * m * a_pressure;
  if (f_cap > 0.0 && f_mag > f_cap) {
    const double s = f_cap / f_mag;
    fr *= s;
    fz *= s;
  }

  const int stride = nz + 1;
  const int i = n / stride;
  const int j = n - i * stride;
  if (i == 0) {
    fr = 0.0;
  }
  if (i == nr) {
    if (r_outer_type == static_cast<int>(Boundary2DType::FIXED)) {
      fr = 0.0;
      fz = 0.0;
    } else if (r_outer_type == static_cast<int>(Boundary2DType::REFLECT)) {
      fr = 0.0;
    }
  }
  if (j == 0) {
    if (z_bottom_type == static_cast<int>(Boundary2DType::FIXED)) {
      fr = 0.0;
      fz = 0.0;
    } else if (z_bottom_type == static_cast<int>(Boundary2DType::REFLECT) ||
               z_bottom_type == static_cast<int>(Boundary2DType::STATE_SUPPLY)) {
      fz = 0.0;
    }
  }
  if (j == nz) {
    if (z_top_type == static_cast<int>(Boundary2DType::FIXED)) {
      fr = 0.0;
      fz = 0.0;
    } else if (z_top_type == static_cast<int>(Boundary2DType::REFLECT) ||
               z_top_type == static_cast<int>(Boundary2DType::STATE_SUPPLY)) {
      fz = 0.0;
    }
  }

  accel_r_hg[n] = fr / m;
  accel_z_hg[n] = fz / m;
}

__device__ inline void remove_radial_normal_component(double& fr,
                                                      double& fz,
                                                      const double r,
                                                      const double z) {
  const double s = sqrt(r * r + z * z);
  if (s > 0.0) {
    const double inv_s = 1.0 / s;
    const double nr = r * inv_s;
    const double nz = z * inv_s;
    const double fn = fr * nr + fz * nz;
    fr -= fn * nr;
    fz -= fn * nz;
  }
}

__global__ void limit_force_to_accel_multiblock_kernel(
    double* __restrict__ accel_r_hg,
    double* __restrict__ accel_z_hg,
    const double* __restrict__ force_r_hg,
    const double* __restrict__ force_z_hg,
    const double* __restrict__ node_mass,
    const double* __restrict__ pressure_accel_r,
    const double* __restrict__ pressure_accel_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const std::uint8_t* __restrict__ node_flags,
    const int n_nodes,
    const int r_outer_type,
    const double max_force_per_node_fraction) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  const double m = node_mass[n];
  if (!(m > 0.0)) {
    accel_r_hg[n] = 0.0;
    accel_z_hg[n] = 0.0;
    return;
  }
  double fr = force_r_hg[n];
  double fz = force_z_hg[n];
  const double f_mag = hypot(fr, fz);
  const double a_pressure = hypot(pressure_accel_r[n], pressure_accel_z[n]);
  const double f_cap = max_force_per_node_fraction * m * a_pressure;
  if (f_cap > 0.0 && f_mag > f_cap) {
    const double s = f_cap / f_mag;
    fr *= s;
    fz *= s;
  }
  if (node_flags != nullptr) {
    const std::uint8_t flags = node_flags[n];
    if ((flags & mesh::NODE_CENTER) != 0U) {
      fr = 0.0;
      fz = 0.0;
    } else {
      if ((flags & (mesh::NODE_AXIS | mesh::NODE_POLE_AXIS)) != 0U) {
        fr = 0.0;
      }
      if ((flags & mesh::NODE_OUTER_PHYSICAL_BOUNDARY) != 0U) {
        if (r_outer_type == static_cast<int>(Boundary2DType::FIXED)) {
          fr = 0.0;
          fz = 0.0;
        } else if (r_outer_type != static_cast<int>(Boundary2DType::FREE) &&
                   r_outer_type != static_cast<int>(Boundary2DType::PRESSURE)) {
          remove_radial_normal_component(fr, fz, x_r[n], x_z[n]);
        }
      } else if ((flags & mesh::NODE_BOUNDARY) != 0U) {
        remove_radial_normal_component(fr, fz, x_r[n], x_z[n]);
      }
    }
  }

  accel_r_hg[n] = fr / m;
  accel_z_hg[n] = fz / m;
}

__global__ void compute_hourglass_work_kernel(
    double* __restrict__ work_erg,
    const double* __restrict__ accel_r_hg,
    const double* __restrict__ accel_z_hg,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ subzonal_m0,
    const double* __restrict__ subzonal_m1,
    const double* __restrict__ subzonal_m2,
    const double* __restrict__ subzonal_m3,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const int button_outer_node_ring,
    const double dt) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    work_erg[c] = 0.0;
    return;
  }
  if (button_or_dormant_center_cell(c, nz, button_outer_node_ring)) {
    work_erg[c] = 0.0;
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int nodes[4] = {
      node_index_2d(i, j, nz),
      node_index_2d(i + 1, j, nz),
      node_index_2d(i + 1, j + 1, nz),
      node_index_2d(i, j + 1, nz),
  };
  const double sub_m[4] = {
      fmax(subzonal_m0[c], 0.0),
      fmax(subzonal_m1[c], 0.0),
      fmax(subzonal_m2[c], 0.0),
      fmax(subzonal_m3[c], 0.0),
  };
  double mechanical_work = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int n = nodes[k];
    mechanical_work += sub_m[k] * (accel_r_hg[n] * v_r[n] +
                                   accel_z_hg[n] * v_z[n]) * dt;
  }
  work_erg[c] = -mechanical_work;
}

__global__ void compute_hourglass_work_multiblock_kernel(
    double* __restrict__ work_erg,
    const double* __restrict__ accel_r_hg,
    const double* __restrict__ accel_z_hg,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ corner_mass,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells,
    const double dt) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    work_erg[c] = 0.0;
    return;
  }
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  if (active_nverts == 3) {
    // Triangles have no bilinear hourglass mode.
    work_erg[c] = 0.0;
    return;
  }
  const int off = cell_node_csr_offsets[c];
  double mechanical_work = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int n = cell_node_csr_indices[off + k];
    const double sub_m = fmax(corner_mass[c * 4 + k], 0.0);
    mechanical_work += sub_m * (accel_r_hg[n] * v_r[n] +
                                accel_z_hg[n] * v_z[n]) * dt;
  }
  work_erg[c] = -mechanical_work;
}

__global__ void add_accel_kernel(double* __restrict__ accel_r,
                                 double* __restrict__ accel_z,
                                 const double* __restrict__ accel_r_hg,
                                 const double* __restrict__ accel_z_hg,
                                 const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  accel_r[n] += accel_r_hg[n];
  accel_z[n] += accel_z_hg[n];
}

__global__ void apply_hourglass_work_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ mass,
    const double* __restrict__ work_erg,
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ central_member_mask,
    const std::uint8_t* __restrict__ central_passive_mask,
    const std::uint8_t* __restrict__ pole_member_mask,
    const int n_cells,
    const int use_two_temp,
    const int heat_to_electron,
    double* __restrict__ E_floor_injected,
    int* __restrict__ clamp_count) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if ((central_member_mask != nullptr && central_member_mask[c] != 0U) ||
      (central_passive_mask != nullptr && central_passive_mask[c] != 0U) ||
      (pole_member_mask != nullptr && pole_member_mask[c] != 0U)) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }
  const double m = mass[c];
  if (!(m > 0.0)) {
    return;
  }
  const double de = work_erg[c] / m;
  if (use_two_temp != 0) {
    double* target = (heat_to_electron != 0) ? ee : ei;
    const double raw = target[c] + de;
    const double out = fmax(raw, 0.0);
    target[c] = out;
    if (raw < 0.0) {
      if (E_floor_injected != nullptr) {
        atomic_add_double(E_floor_injected, m * (out - raw));
      }
      if (clamp_count != nullptr) {
        atomicAdd(clamp_count, 1);
      }
    }
    return;
  }
  const double raw = ee[c] + de;
  const double out = fmax(raw, 0.0);
  ee[c] = out;
  if (raw < 0.0) {
    if (E_floor_injected != nullptr) {
      atomic_add_double(E_floor_injected, m * (out - raw));
    }
    if (clamp_count != nullptr) {
      atomicAdd(clamp_count, 1);
    }
  }
}

constexpr int kI1BSpuriousSensorBlockSize = 256;

struct I1BSpuriousSensorBlockStats {
  int sampled_cells = 0;
  double residual_ke_sum = 0.0;
  double residual_ke_max = 0.0;
};

__host__ __device__ inline I1BSpuriousSensorCell empty_sensor_cell() {
  I1BSpuriousSensorCell cell{};
  cell.rank = -1;
  cell.cell_id = -1;
  cell.active_nverts = 0;
  cell.residual_ke = 0.0;
  cell.affine_ke = 0.0;
  cell.eta2 = 0.0;
  cell.score = -1.0;
  return cell;
}

__host__ __device__ inline void insert_sensor_top(
    I1BSpuriousSensorCell* const top,
    const int top_k,
    const I1BSpuriousSensorCell candidate) {
  if (top == nullptr || top_k <= 0 || candidate.cell_id < 0 ||
      !(candidate.score > 0.0) || !isfinite(candidate.score)) {
    return;
  }
  for (int slot = 0; slot < top_k; ++slot) {
    if (candidate.score > top[slot].score) {
      for (int dst = top_k - 1; dst > slot; --dst) {
        top[dst] = top[dst - 1];
      }
      top[slot] = candidate;
      return;
    }
  }
}

__device__ inline int sensor_cell_node_id(
    const int c,
    const int i,
    const int j,
    const int k,
    const int nz,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices) {
  if (cell_node_csr_offsets != nullptr && cell_node_csr_indices != nullptr) {
    return cell_node_csr_indices[cell_node_csr_offsets[c] + k];
  }
  if (k == 0) {
    return node_index_2d(i, j, nz);
  }
  if (k == 1) {
    return node_index_2d(i + 1, j, nz);
  }
  if (k == 2) {
    return node_index_2d(i + 1, j + 1, nz);
  }
  return node_index_2d(i, j + 1, nz);
}

__global__ void hydro_nonaffine_ke_sensor_kernel(
    I1BSpuriousSensorCell* __restrict__ block_top,
    I1BSpuriousSensorBlockStats* __restrict__ block_stats,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ corner_mass,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const int n_cells,
    const int top_k) {
  __shared__ I1BSpuriousSensorCell entries[kI1BSpuriousSensorBlockSize];
  const int tid = threadIdx.x;
  const int c = blockIdx.x * blockDim.x + tid;
  I1BSpuriousSensorCell entry = empty_sensor_cell();

  if (c < n_cells && ((hydro_active == nullptr) || hydro_active[c] != 0)) {
    const int active_nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
    if (active_nverts == 4) {
      const int i = (nz > 0) ? (c / nz) : 0;
      const int j = (nz > 0) ? (c - i * nz) : 0;
      int nodes[4] = {0, 0, 0, 0};
      double m[4] = {0.0, 0.0, 0.0, 0.0};
      double r[4] = {0.0, 0.0, 0.0, 0.0};
      double z[4] = {0.0, 0.0, 0.0, 0.0};
      double ur[4] = {0.0, 0.0, 0.0, 0.0};
      double uz[4] = {0.0, 0.0, 0.0, 0.0};
      bool duplicate_node = false;
      double m_sum = 0.0;
      bool finite_cell = true;
      for (int k = 0; k < 4; ++k) {
        nodes[k] = sensor_cell_node_id(
            c, i, j, k, nz, cell_node_csr_offsets, cell_node_csr_indices);
        for (int q = 0; q < k; ++q) {
          duplicate_node = duplicate_node || (nodes[k] == nodes[q]);
        }
        m[k] = fmax(corner_mass[4 * c + k], 0.0);
        r[k] = x_r[nodes[k]];
        z[k] = x_z[nodes[k]];
        ur[k] = v_r[nodes[k]];
        uz[k] = v_z[nodes[k]];
        finite_cell = finite_cell && isfinite(m[k]) && isfinite(r[k]) &&
                      isfinite(z[k]) && isfinite(ur[k]) && isfinite(uz[k]);
        m_sum += m[k];
      }
      if (finite_cell && !duplicate_node && m_sum > 0.0 && isfinite(m_sum)) {
        double rc = 0.0;
        double zc = 0.0;
        double urc = 0.0;
        double uzc = 0.0;
        for (int k = 0; k < 4; ++k) {
          rc += m[k] * r[k];
          zc += m[k] * z[k];
          urc += m[k] * ur[k];
          uzc += m[k] * uz[k];
        }
        const double inv_m = 1.0 / m_sum;
        rc *= inv_m;
        zc *= inv_m;
        urc *= inv_m;
        uzc *= inv_m;

        double c00 = 0.0;
        double c01 = 0.0;
        double c11 = 0.0;
        double d00 = 0.0;
        double d01 = 0.0;
        double d10 = 0.0;
        double d11 = 0.0;
        double xrel_r[4] = {0.0, 0.0, 0.0, 0.0};
        double xrel_z[4] = {0.0, 0.0, 0.0, 0.0};
        double urel_r[4] = {0.0, 0.0, 0.0, 0.0};
        double urel_z[4] = {0.0, 0.0, 0.0, 0.0};
        double total_norm = 0.0;
        for (int k = 0; k < 4; ++k) {
          xrel_r[k] = r[k] - rc;
          xrel_z[k] = z[k] - zc;
          urel_r[k] = ur[k] - urc;
          urel_z[k] = uz[k] - uzc;
          c00 += m[k] * xrel_r[k] * xrel_r[k];
          c01 += m[k] * xrel_r[k] * xrel_z[k];
          c11 += m[k] * xrel_z[k] * xrel_z[k];
          d00 += m[k] * urel_r[k] * xrel_r[k];
          d01 += m[k] * urel_r[k] * xrel_z[k];
          d10 += m[k] * urel_z[k] * xrel_r[k];
          d11 += m[k] * urel_z[k] * xrel_z[k];
          total_norm += m[k] * (urel_r[k] * urel_r[k] +
                                urel_z[k] * urel_z[k]);
        }

        const double trace = c00 + c11;
        const double lambda = 1.0e-12 * fmax(trace, 1.0e-300);
        const double a00 = c00 + lambda;
        const double a11 = c11 + lambda;
        const double det = a00 * a11 - c01 * c01;
        if (isfinite(det) && fabs(det) > 1.0e-300) {
          const double inv00 = a11 / det;
          const double inv01 = -c01 / det;
          const double inv11 = a00 / det;
          const double b00 = d00 * inv00 + d01 * inv01;
          const double b01 = d00 * inv01 + d01 * inv11;
          const double b10 = d10 * inv00 + d11 * inv01;
          const double b11 = d10 * inv01 + d11 * inv11;
          double residual_norm = 0.0;
          double affine_norm = 0.0;
          for (int k = 0; k < 4; ++k) {
            const double ar = b00 * xrel_r[k] + b01 * xrel_z[k];
            const double az = b10 * xrel_r[k] + b11 * xrel_z[k];
            const double hr = urel_r[k] - ar;
            const double hz = urel_z[k] - az;
            residual_norm += m[k] * (hr * hr + hz * hz);
            affine_norm += m[k] * (ar * ar + az * az);
          }
          if (isfinite(residual_norm) && isfinite(affine_norm)) {
            entry.cell_id = c;
            entry.active_nverts = active_nverts;
            entry.residual_ke = 0.5 * fmax(residual_norm, 0.0);
            entry.affine_ke = 0.5 * fmax(affine_norm, 0.0);
            entry.eta2 =
                residual_norm / (fmax(total_norm, 0.0) + 1.0e-300);
            entry.score = entry.residual_ke;
          }
        }
      }
    }
  }

  entries[tid] = entry;
  __syncthreads();

  if (tid == 0) {
    I1BSpuriousSensorCell local_top[kI1BSpuriousSensorTopKMax];
    for (int k = 0; k < kI1BSpuriousSensorTopKMax; ++k) {
      local_top[k] = empty_sensor_cell();
    }
    I1BSpuriousSensorBlockStats stats{};
    for (int t = 0; t < blockDim.x; ++t) {
      const I1BSpuriousSensorCell candidate = entries[t];
      if (candidate.cell_id < 0) {
        continue;
      }
      ++stats.sampled_cells;
      stats.residual_ke_sum += candidate.residual_ke;
      stats.residual_ke_max =
          fmax(stats.residual_ke_max, candidate.residual_ke);
      insert_sensor_top(local_top, top_k, candidate);
    }
    const int base = blockIdx.x * kI1BSpuriousSensorTopKMax;
    for (int k = 0; k < kI1BSpuriousSensorTopKMax; ++k) {
      block_top[base + k] = local_top[k];
    }
    block_stats[blockIdx.x] = stats;
  }
}

}  // namespace

const double* ensure_bridge_band_subzonal_weights_2d(
    core::State& state,
    const core::Config& cfg) {
  if (cfg.numerics.hydro.subzonal_band_mode == "off") {
    return nullptr;
  }

  TENRYU_ASSERT(cfg.numerics.hydro.subzonal_band_mode == "bridge_feather",
                "unsupported subzonal band mode");
  const std::size_t expected = state.rho.size();
  bool rebuild = !state.subzonal_band_weight_initialized;
  if (state.subzonal_band_weight.size() != expected) {
    state.subzonal_band_weight.reset(expected);
    rebuild = true;
  }
  if (rebuild) {
    const std::vector<double> weights = build_bridge_band_weights(
        state, cfg.numerics.hydro.subzonal_band_feather_layers);
    state.subzonal_band_weight.copy_from_host(weights);
    state.subzonal_band_weight_initialized = true;
  }
  return state.subzonal_band_weight.data();
}

void ensure_hourglass_subzonal_masses_2d(core::State& state,
                                         const core::Config& cfg,
                                         const bool force_recompute) {
  if (!subzonal_hourglass_enabled(cfg) || cfg.main.dim != 2) {
    return;
  }
  const int n_cells = static_cast<int>(state.mass.size());
  if (n_cells <= 0) {
    return;
  }
  static const int diag_cell = [] {
    const char* s = std::getenv("TENRYU_SUBZ_ENSURE_DIAG");
    return s != nullptr ? std::atoi(s) : -1;
  }();
  static int diag_call_count = 0;
  ++diag_call_count;
  const bool invariant = corner_mass_lagrangian_invariant_enabled(cfg);
  auto dump = [&](const char* tag) {
    if (diag_cell < 0 || diag_cell >= n_cells ||
        state.corner_mass.size() <
            static_cast<std::size_t>(diag_cell + 1) * state.corner_stride) {
      return;
    }
    double corner_mass[4];
    cuda_check(cudaMemcpy(corner_mass,
                          state.corner_mass.data() +
                              diag_cell * state.corner_stride,
                          sizeof(corner_mass), cudaMemcpyDeviceToHost),
               "Hydro2D subzonal ensure diagnostic copy failed");
    char line[512];
    std::snprintf(
        line, sizeof(line),
        "[subz_ensure] call=%d tag=%s force=%d invariant=%d initialized=%d "
        "cm0=%.17e cm1=%.17e cm2=%.17e cm3=%.17e",
        diag_call_count, tag, force_recompute ? 1 : 0, invariant ? 1 : 0,
        state.corner_mass_initialized ? 1 : 0, corner_mass[0], corner_mass[1],
        corner_mass[2], corner_mass[3]);
    core::log_info(line);
  };
  dump("entry");
  const std::size_t expected = static_cast<std::size_t>(n_cells);
  state.corner_mass_is_lagrangian_invariant = invariant;
  bool recompute = force_recompute;
  auto ensure = [&](core::CellField1D& field) {
    if (field.size() != expected) {
      field.reset(expected);
      recompute = true;
    }
  };
  ensure(state.subzonal_mass_corner0);
  ensure(state.subzonal_mass_corner1);
  ensure(state.subzonal_mass_corner2);
  ensure(state.subzonal_mass_corner3);
  const std::size_t expected_corner =
      expected * static_cast<std::size_t>(state.corner_stride);
  if (state.corner_mass.size() != expected_corner) {
    state.corner_mass.reset(expected_corner);
    state.corner_mass_initialized = false;
    recompute = true;
  }
  bool update_volume = force_recompute || invariant;
  if (state.corner_volume.size() != expected_corner) {
    state.corner_volume.reset(expected_corner);
    update_volume = true;
  }
  if (invariant && !state.corner_mass_initialized) {
    recompute = true;
  }
  core::DeviceArray<std::uint8_t> d_cell_nverts;
  const std::uint8_t* d_cell_nverts_ptr =
      upload_cell_nverts_if_nonquad(d_cell_nverts, state, n_cells);
  if (invariant && state.corner_mass_initialized && !force_recompute) {
    const int blocks = (n_cells + 255) / 256;
    copy_corner_mass_to_subzonal_kernel<<<blocks, 256>>>(
        state.subzonal_mass_corner0.data(),
        state.subzonal_mass_corner1.data(),
        state.subzonal_mass_corner2.data(),
        state.subzonal_mass_corner3.data(),
        state.corner_mass.data(),
        d_cell_nverts_ptr,
        state.corner_stride,
        n_cells);
    sync_kernel("Hydro2D copy invariant corner masses to subzonal SoA failed");
    if (update_volume) {
      update_corner_volumes(state, blocks, expected, expected_corner);
    }
    dump("exit");
    return;
  }
  if (invariant && force_recompute && state.corner_mass_initialized) {
    warn_invariant_remap_reinit_once();
  }
  if (!recompute && !update_volume) {
    dump("exit");
    return;
  }
  const int blocks = (n_cells + 255) / 256;
  if (!recompute) {
    update_corner_volumes(state, blocks, expected, expected_corner);
    dump("exit");
    return;
  }
  if (state.mesh.topo.multiblock.has_value()) {
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                      expected + 1U,
                  "Hydro2D hourglass subzonal masses require cell-node CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                      expected_corner,
                  "Hydro2D hourglass subzonal masses require cell-node CSR indices");
    initialize_subzonal_masses_multiblock_kernel<<<blocks, 256>>>(
        state.subzonal_mass_corner0.data(),
        state.subzonal_mass_corner1.data(),
        state.subzonal_mass_corner2.data(),
        state.subzonal_mass_corner3.data(),
        state.corner_mass.data(),
        state.corner_volume.data(),
        state.mass.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        d_cell_nverts_ptr,
        state.corner_stride,
        n_cells);
  } else {
    initialize_subzonal_masses_kernel<<<blocks, 256>>>(
        state.subzonal_mass_corner0.data(),
        state.subzonal_mass_corner1.data(),
        state.subzonal_mass_corner2.data(),
        state.subzonal_mass_corner3.data(),
        state.corner_mass.data(),
        state.corner_volume.data(),
        state.mass.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.mesh.topo.nr,
        state.mesh.topo.nz);
  }
  sync_kernel("Hydro2D initialize hourglass subzonal masses failed");
  state.corner_mass_initialized = true;
  dump("exit");
}

I1BSpuriousSensorSummary compute_hydro_nonaffine_ke_sensor_2d(
    const core::State& state,
    const std::int8_t* const d_hydro_active,
    const int top_k) {
  I1BSpuriousSensorSummary out{};
  const int n_cells = static_cast<int>(state.mass.size());
  if (n_cells <= 0 || state.corner_mass.size() !=
                          static_cast<std::size_t>(n_cells) * 4U) {
    return out;
  }
  const int clamped_top_k =
      std::clamp(top_k, 1, kI1BSpuriousSensorTopKMax);
  const int blocks =
      (n_cells + kI1BSpuriousSensorBlockSize - 1) /
      kI1BSpuriousSensorBlockSize;
  core::DeviceArray<I1BSpuriousSensorCell> d_block_top;
  core::DeviceArray<I1BSpuriousSensorBlockStats> d_block_stats;
  d_block_top.reset(static_cast<std::size_t>(blocks) *
                    static_cast<std::size_t>(kI1BSpuriousSensorTopKMax));
  d_block_stats.reset(static_cast<std::size_t>(blocks));

  core::DeviceArray<std::uint8_t> d_cell_nverts;
  const std::uint8_t* const d_cell_nverts_ptr =
      upload_cell_nverts_if_tri(d_cell_nverts, state, n_cells);
  const bool multiblock = state.mesh.topo.multiblock.has_value();
  const int* const d_cell_node_offsets =
      multiblock ? state.mesh.multiblock_cell_node_csr_offsets.data()
                 : nullptr;
  const int* const d_cell_node_indices =
      multiblock ? state.mesh.multiblock_cell_node_csr_indices.data()
                 : nullptr;
  if (multiblock) {
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "Hydro2D sensor requires multiblock cell-node CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                      static_cast<std::size_t>(n_cells) * 4U,
                  "Hydro2D sensor requires multiblock cell-node CSR indices");
  }

  hydro_nonaffine_ke_sensor_kernel<<<blocks, kI1BSpuriousSensorBlockSize>>>(
      d_block_top.data(),
      d_block_stats.data(),
      state.x_r.data(),
      state.x_z.data(),
      state.v_r.data(),
      state.v_z.data(),
      state.corner_mass.data(),
      d_cell_node_offsets,
      d_cell_node_indices,
      d_cell_nverts_ptr,
      d_hydro_active,
      state.mesh.topo.nr,
      state.mesh.topo.nz,
      n_cells,
      clamped_top_k);
  sync_kernel("Hydro2D I1B non-affine KE sensor failed");

  std::vector<I1BSpuriousSensorBlockStats> stats;
  std::vector<I1BSpuriousSensorCell> block_top;
  d_block_stats.copy_to_host(stats);
  d_block_top.copy_to_host(block_top);

  out.valid = true;
  for (const auto& block : stats) {
    out.sampled_cells += block.sampled_cells;
    out.residual_ke_sum += block.residual_ke_sum;
    out.residual_ke_max = std::max(out.residual_ke_max, block.residual_ke_max);
  }
  for (const I1BSpuriousSensorCell& candidate : block_top) {
    insert_sensor_top(out.top.data(), clamped_top_k, candidate);
  }
  out.top_count = 0;
  for (int k = 0; k < clamped_top_k; ++k) {
    if (out.top[static_cast<std::size_t>(k)].cell_id >= 0) {
      ++out.top_count;
    }
  }
  return out;
}

void compute_anti_hourglass_acceleration_2d(
    core::NodeField1D& accel_r_hg,
    core::NodeField1D& accel_z_hg,
    core::CellField1D* compatible_work_erg,
    const core::State& state,
    const core::Config& cfg,
    const core::CellField1D& cell_pressure,
    const core::NodeField1D& node_mass,
    const core::NodeField1D& pressure_accel_r,
    const core::NodeField1D& pressure_accel_z,
    const std::int8_t* d_hydro_active,
    const double dt,
    const bool compute_compatible_work,
    const std::uint8_t* d_node_flags,
    const double* work_velocity_r,
    const double* work_velocity_z,
    const double compatible_work_dt) {
  const int n_nodes = static_cast<int>(state.x_r.size());
  const int n_cells = static_cast<int>(state.rho.size());
  accel_r_hg.reset(static_cast<std::size_t>(n_nodes));
  accel_z_hg.reset(static_cast<std::size_t>(n_nodes));
  if (cfg.numerics.hydro.subzonal_pressure_enabled) {
    return;
  }
  if (!subzonal_hourglass_enabled(cfg) || n_nodes <= 0 || n_cells <= 0) {
    return;
  }
  TENRYU_ASSERT(state.subzonal_mass_corner0.size() == static_cast<std::size_t>(n_cells),
                "hourglass subzonal_mass_corner0 is not initialized");
  TENRYU_ASSERT(state.subzonal_mass_corner1.size() == static_cast<std::size_t>(n_cells),
                "hourglass subzonal_mass_corner1 is not initialized");
  TENRYU_ASSERT(state.subzonal_mass_corner2.size() == static_cast<std::size_t>(n_cells),
                "hourglass subzonal_mass_corner2 is not initialized");
  TENRYU_ASSERT(state.subzonal_mass_corner3.size() == static_cast<std::size_t>(n_cells),
                "hourglass subzonal_mass_corner3 is not initialized");

  core::NodeField1D force_r;
  core::NodeField1D force_z;
  force_r.reset(static_cast<std::size_t>(n_nodes));
  force_z.reset(static_cast<std::size_t>(n_nodes));
  const int blocks_cells = (n_cells + 255) / 256;
  const int blocks_nodes = (n_nodes + 255) / 256;
  const auto& hg = cfg.numerics.hydro.hourglass;
  const double scale = anti_hourglass_scale(cfg);
  const int stiffness_mode = cfg.numerics.hydro.subzonal_mass_enabled ? 1 : 0;
  const int button_outer_node_ring =
      (state.mesh.button_center && state.mesh.button_center->enabled)
          ? state.mesh.button_center->outer_node_ring
          : 0;
  core::DeviceArray<std::uint8_t> d_cell_nverts;
  const std::uint8_t* d_cell_nverts_ptr =
      upload_cell_nverts_if_tri(d_cell_nverts, state, n_cells);
  if (state.mesh.topo.multiblock.has_value()) {
    TENRYU_ASSERT(state.corner_mass.size() ==
                      static_cast<std::size_t>(n_cells) * 4U,
                  "Hydro2D multiblock hourglass requires flat corner_mass");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "Hydro2D multiblock hourglass requires cell-node CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                      static_cast<std::size_t>(n_cells) * 4U,
                  "Hydro2D multiblock hourglass requires cell-node CSR indices");
    compute_anti_hourglass_force_multiblock_kernel<<<blocks_cells, 256>>>(
        force_r.data(),
        force_z.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.rho.data(),
        cell_pressure.data(),
        state.corner_mass.data(),
        d_hydro_active,
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        d_cell_nverts_ptr,
        n_cells,
        scale,
        hg.activation_corner_j_ratio_threshold,
        hg.activation_hourglass_amplitude_threshold,
        stiffness_mode,
        dt);
  } else {
    compute_anti_hourglass_force_kernel<<<blocks_cells, 256>>>(
        force_r.data(),
        force_z.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.rho.data(),
        cell_pressure.data(),
        state.subzonal_mass_corner0.data(),
        state.subzonal_mass_corner1.data(),
        state.subzonal_mass_corner2.data(),
        state.subzonal_mass_corner3.data(),
        d_hydro_active,
        state.mesh.topo.nr,
        state.mesh.topo.nz,
        scale,
        hg.activation_corner_j_ratio_threshold,
        hg.activation_hourglass_amplitude_threshold,
        stiffness_mode,
        button_outer_node_ring,
        dt);
  }

  const auto r_outer = parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_outer);
  const auto z_bottom =
      parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_bottom);
  const auto z_top = parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_top);
  if (state.mesh.topo.multiblock.has_value()) {
    limit_force_to_accel_multiblock_kernel<<<blocks_nodes, 256>>>(
        accel_r_hg.data(),
        accel_z_hg.data(),
        force_r.data(),
        force_z.data(),
        node_mass.data(),
        pressure_accel_r.data(),
        pressure_accel_z.data(),
        state.x_r.data(),
        state.x_z.data(),
        d_node_flags,
        n_nodes,
        static_cast<int>(r_outer),
        hg.max_force_per_node_fraction);
  } else {
    limit_force_to_accel_kernel<<<blocks_nodes, 256>>>(
        accel_r_hg.data(),
        accel_z_hg.data(),
        force_r.data(),
        force_z.data(),
        node_mass.data(),
        pressure_accel_r.data(),
        pressure_accel_z.data(),
        state.mesh.topo.nr,
        state.mesh.topo.nz,
        static_cast<int>(r_outer),
        static_cast<int>(z_bottom),
        static_cast<int>(z_top),
        hg.max_force_per_node_fraction);
  }
  sync_kernel("Hydro2D anti-hourglass acceleration failed");

  if (compatible_work_erg != nullptr) {
    compatible_work_erg->reset(static_cast<std::size_t>(n_cells));
    if (compute_compatible_work && hg.compatible_work_enabled && dt > 0.0) {
      const double work_dt =
          compatible_work_dt >= 0.0 ? compatible_work_dt : dt;
      const double* work_v_r =
          work_velocity_r != nullptr ? work_velocity_r : state.v_r.data();
      const double* work_v_z =
          work_velocity_z != nullptr ? work_velocity_z : state.v_z.data();
      if (state.mesh.topo.multiblock.has_value()) {
        compute_hourglass_work_multiblock_kernel<<<blocks_cells, 256>>>(
            compatible_work_erg->data(),
            accel_r_hg.data(),
            accel_z_hg.data(),
            work_v_r,
            work_v_z,
            state.corner_mass.data(),
            d_hydro_active,
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            d_cell_nverts_ptr,
            n_cells,
            work_dt);
      } else {
        compute_hourglass_work_kernel<<<blocks_cells, 256>>>(
            compatible_work_erg->data(),
            accel_r_hg.data(),
            accel_z_hg.data(),
            work_v_r,
            work_v_z,
            state.subzonal_mass_corner0.data(),
            state.subzonal_mass_corner1.data(),
            state.subzonal_mass_corner2.data(),
            state.subzonal_mass_corner3.data(),
            d_hydro_active,
            state.mesh.topo.nr,
            state.mesh.topo.nz,
            button_outer_node_ring,
            work_dt);
      }
      sync_kernel("Hydro2D anti-hourglass compatible work failed");
    }
  }
}

void add_hourglass_acceleration_2d(core::NodeField1D& accel_r,
                                   core::NodeField1D& accel_z,
                                   const core::NodeField1D& accel_r_hg,
                                   const core::NodeField1D& accel_z_hg) {
  const int n_nodes = static_cast<int>(accel_r.size());
  if (n_nodes <= 0 || accel_r_hg.empty() || accel_z_hg.empty()) {
    return;
  }
  TENRYU_ASSERT(accel_z.size() == accel_r.size(),
                "add_hourglass_acceleration_2d acceleration size mismatch");
  TENRYU_ASSERT(accel_r_hg.size() == accel_r.size() &&
                    accel_z_hg.size() == accel_r.size(),
                "add_hourglass_acceleration_2d hourglass size mismatch");
  const int blocks = (n_nodes + 255) / 256;
  add_accel_kernel<<<blocks, 256>>>(
      accel_r.data(), accel_z.data(), accel_r_hg.data(), accel_z_hg.data(),
      n_nodes);
  sync_kernel("Hydro2D add anti-hourglass acceleration failed");
}

void apply_hourglass_work_2d(core::State& state,
                             const core::Config& cfg,
                             const core::CellField1D& compatible_work_erg,
                             const bool use_two_temp,
                             const std::int8_t* d_hydro_active,
                             double* E_floor_injected,
                             int* clamp_count) {
  if (!subzonal_hourglass_enabled(cfg) ||
      !cfg.numerics.hydro.hourglass.compatible_work_enabled ||
      compatible_work_erg.empty()) {
    return;
  }
  const int n_cells = static_cast<int>(state.mass.size());
  TENRYU_ASSERT(compatible_work_erg.size() == state.mass.size(),
                "apply_hourglass_work_2d work size mismatch");
  const int blocks = (n_cells + 255) / 256;
  const int heat_to_electron = (cfg.numerics.hydro.av_heat_to == "electron") ? 1 : 0;
  const bool central_overlay_active = central_pseudo_core::active(state);
  const bool pole_overlay_active = pole_angular_derefine::active(state);
  apply_hourglass_work_kernel<<<blocks, 256>>>(
      state.ee.data(),
      state.ei.data(),
      state.mass.data(),
      compatible_work_erg.data(),
      d_hydro_active,
      central_overlay_active ? state.central_pseudo_core.d_member_mask.data()
                             : nullptr,
      central_overlay_active ? state.central_pseudo_core.d_passive_mask.data()
                             : nullptr,
      pole_overlay_active ? state.pole_angular_derefine.d_member_mask.data()
                          : nullptr,
      n_cells,
      use_two_temp ? 1 : 0,
      heat_to_electron,
      E_floor_injected,
      clamp_count);
  sync_kernel("Hydro2D apply anti-hourglass compatible work failed");
}

}  // namespace tenryu::hydro
