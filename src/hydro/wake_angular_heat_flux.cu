#include "hydro/wake_angular_heat_flux.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/euler_window_blend.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::wake_angular_heat_flux {
namespace {

constexpr int kThreads = 256;
constexpr double kTiny = 1.0e-300;
constexpr double kPi = 3.141592653589793238462643383279502884;

void cuda_check(const cudaError_t error, const char* message) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " +
                             cudaGetErrorString(error));
  }
}

void sync_kernel(const char* message) {
  cuda_check(cudaGetLastError(), message);
  cuda_check(cudaDeviceSynchronize(), message);
}

struct FaceRecord {
  int cell_a = -1;
  int cell_b = -1;
  int local_a = -1;
};

struct UnionFind {
  explicit UnionFind(const int n) : parent(static_cast<std::size_t>(n)) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int find(const int value) {
    int root = value;
    while (parent[static_cast<std::size_t>(root)] != root) {
      root = parent[static_cast<std::size_t>(root)];
    }
    int current = value;
    while (parent[static_cast<std::size_t>(current)] != current) {
      const int next = parent[static_cast<std::size_t>(current)];
      parent[static_cast<std::size_t>(current)] = root;
      current = next;
    }
    return root;
  }

  void unite(const int a, const int b) {
    const int root_a = find(a);
    const int root_b = find(b);
    if (root_a == root_b) {
      return;
    }
    const int low = std::min(root_a, root_b);
    const int high = std::max(root_a, root_b);
    parent[static_cast<std::size_t>(high)] = low;
  }

  std::vector<int> parent;
};

bool side_is_angular(const mesh::BlockSide side) {
  return side == mesh::BlockSide::J_MINUS ||
         side == mesh::BlockSide::J_PLUS;
}

bool angular_face_from_topology(const mesh::MultiBlockTopology& topology,
                                const mesh::UniqueOrientedFace& face) {
  const int block_a =
      topology.cell_block_id[static_cast<std::size_t>(face.cell_a)];
  const int block_b =
      topology.cell_block_id[static_cast<std::size_t>(face.cell_b)];
  if (block_a < 0 || block_b < 0 ||
      block_a >= static_cast<int>(topology.blocks.size()) ||
      block_b >= static_cast<int>(topology.blocks.size())) {
    return false;
  }
  const auto& a = topology.blocks[static_cast<std::size_t>(block_a)];
  const auto& b = topology.blocks[static_cast<std::size_t>(block_b)];
  if (a.role == mesh::BlockRole::CENTRAL_CORE ||
      b.role == mesh::BlockRole::CENTRAL_CORE) {
    return false;
  }
  if (block_a == block_b) {
    if (a.n_j_cells <= 0) {
      return false;
    }
    const int local_a = face.cell_a - a.cell_begin;
    const int local_b = face.cell_b - a.cell_begin;
    return local_a >= 0 && local_b >= 0 &&
           local_a / a.n_j_cells == local_b / a.n_j_cells;
  }
  for (const auto& seam : topology.seams) {
    if (seam.block_a == block_a && seam.block_b == block_b) {
      return side_is_angular(seam.side_a) && side_is_angular(seam.side_b);
    }
    if (seam.block_a == block_b && seam.block_b == block_a) {
      return side_is_angular(seam.side_a) && side_is_angular(seam.side_b);
    }
  }
  return false;
}

template <typename T>
void upload(core::DeviceArray<T>& device, const std::vector<T>& host) {
  device.reset(host.size());
  if (!host.empty()) {
    device.copy_from_host(host);
  }
}

void append_face_adjacency(const std::vector<FaceRecord>& faces,
                           const int n_cells,
                           std::vector<int>& offsets,
                           std::vector<int>& indices) {
  std::vector<std::vector<int>> per_cell(static_cast<std::size_t>(n_cells));
  for (int f = 0; f < static_cast<int>(faces.size()); ++f) {
    const auto& face = faces[static_cast<std::size_t>(f)];
    per_cell[static_cast<std::size_t>(face.cell_a)].push_back(f + 1);
    per_cell[static_cast<std::size_t>(face.cell_b)].push_back(-(f + 1));
  }
  offsets.assign(static_cast<std::size_t>(n_cells) + 1U, 0);
  indices.clear();
  indices.reserve(2U * faces.size());
  for (int c = 0; c < n_cells; ++c) {
    const auto& adjacency = per_cell[static_cast<std::size_t>(c)];
    indices.insert(indices.end(), adjacency.begin(), adjacency.end());
    offsets[static_cast<std::size_t>(c) + 1U] =
        static_cast<int>(indices.size());
  }
}

struct Runtime {
  const mesh::MultiBlockTopology* topology = nullptr;
  int n_cells = 0;
  int n_chains = 0;
  int n_angular_faces = 0;
  int n_radial_faces = 0;

  core::DeviceArray<int> angular_cell_a;
  core::DeviceArray<int> angular_cell_b;
  core::DeviceArray<int> angular_local_a;
  core::DeviceArray<int> angular_adj_offsets;
  core::DeviceArray<int> angular_adj_indices;
  core::DeviceArray<int> radial_cell_a;
  core::DeviceArray<int> radial_cell_b;
  core::DeviceArray<int> radial_local_a;
  core::DeviceArray<int> radial_adj_offsets;
  core::DeviceArray<int> radial_adj_indices;
  core::DeviceArray<int> chain_offsets;
  core::DeviceArray<int> chain_cells;
  core::DeviceArray<int> cell_chain;
  core::DeviceArray<std::uint8_t> cell_nverts;
  core::DeviceArray<std::uint8_t> excluded;
  core::DeviceArray<int> dominant_material;

  core::DeviceArray<double> centroid_r;
  core::DeviceArray<double> centroid_z;
  core::DeviceArray<double> velocity_r;
  core::DeviceArray<double> velocity_z;
  core::DeviceArray<double> radial_length;
  core::DeviceArray<double> av_rate;
  core::DeviceArray<double> chain_max;
  core::DeviceArray<double> sensor;
  core::DeviceArray<std::uint8_t> front_mask;
  core::DeviceArray<double> gate;
  core::DeviceArray<double> raw_conductance;
  core::DeviceArray<double> cell_scale;
  core::DeviceArray<double> face_delta_u;

  Runtime(core::State& state, const core::Config& cfg) {
    TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                  "wake heat flux requires multiblock topology metadata");
    topology = &*state.mesh.topo.multiblock;
    const auto& mb = *topology;
    n_cells = state.mesh.topo.n_cells;
    TENRYU_ASSERT(n_cells > 0 &&
                      mb.cell_block_id.size() == static_cast<std::size_t>(n_cells),
                  "wake heat flux requires complete cell-block metadata");

    std::vector<std::uint8_t> supported(static_cast<std::size_t>(n_cells), 0U);
    for (int c = 0; c < n_cells; ++c) {
      const int block = mb.cell_block_id[static_cast<std::size_t>(c)];
      if (block >= 0 && block < static_cast<int>(mb.blocks.size()) &&
          mb.blocks[static_cast<std::size_t>(block)].role !=
              mesh::BlockRole::CENTRAL_CORE) {
        supported[static_cast<std::size_t>(c)] = 1U;
      }
    }

    std::vector<FaceRecord> angular_faces;
    std::vector<FaceRecord> radial_faces;
    angular_faces.reserve(mb.unique_internal_faces.size());
    radial_faces.reserve(mb.unique_internal_faces.size());
    for (const auto& face : mb.unique_internal_faces) {
      if (face.cell_a < 0 || face.cell_a >= n_cells || face.cell_b < 0 ||
          face.cell_b >= n_cells || face.local_a < 0 ||
          supported[static_cast<std::size_t>(face.cell_a)] == 0U ||
          supported[static_cast<std::size_t>(face.cell_b)] == 0U) {
        continue;
      }
      const FaceRecord record{face.cell_a, face.cell_b, face.local_a};
      if (angular_face_from_topology(mb, face)) {
        angular_faces.push_back(record);
      } else {
        radial_faces.push_back(record);
      }
    }
    n_angular_faces = static_cast<int>(angular_faces.size());
    n_radial_faces = static_cast<int>(radial_faces.size());

    const auto upload_faces = [](const std::vector<FaceRecord>& faces,
                                 core::DeviceArray<int>& cell_a,
                                 core::DeviceArray<int>& cell_b,
                                 core::DeviceArray<int>& local_a) {
      std::vector<int> host_a;
      std::vector<int> host_b;
      std::vector<int> host_local;
      host_a.reserve(faces.size());
      host_b.reserve(faces.size());
      host_local.reserve(faces.size());
      for (const auto& face : faces) {
        host_a.push_back(face.cell_a);
        host_b.push_back(face.cell_b);
        host_local.push_back(face.local_a);
      }
      upload(cell_a, host_a);
      upload(cell_b, host_b);
      upload(local_a, host_local);
    };
    upload_faces(angular_faces, angular_cell_a, angular_cell_b,
                 angular_local_a);
    upload_faces(radial_faces, radial_cell_a, radial_cell_b, radial_local_a);

    std::vector<int> host_offsets;
    std::vector<int> host_indices;
    append_face_adjacency(angular_faces, n_cells, host_offsets, host_indices);
    upload(angular_adj_offsets, host_offsets);
    upload(angular_adj_indices, host_indices);
    append_face_adjacency(radial_faces, n_cells, host_offsets, host_indices);
    upload(radial_adj_offsets, host_offsets);
    upload(radial_adj_indices, host_indices);

    UnionFind unions(n_cells);
    for (const auto& face : radial_faces) {
      unions.unite(face.cell_a, face.cell_b);
    }
    std::vector<int> root_to_chain(static_cast<std::size_t>(n_cells), -1);
    std::vector<std::vector<int>> chains;
    std::vector<int> host_cell_chain(static_cast<std::size_t>(n_cells), -1);
    for (int c = 0; c < n_cells; ++c) {
      if (supported[static_cast<std::size_t>(c)] == 0U) {
        continue;
      }
      const int root = unions.find(c);
      int& chain = root_to_chain[static_cast<std::size_t>(root)];
      if (chain < 0) {
        chain = static_cast<int>(chains.size());
        chains.emplace_back();
      }
      chains[static_cast<std::size_t>(chain)].push_back(c);
      host_cell_chain[static_cast<std::size_t>(c)] = chain;
    }
    n_chains = static_cast<int>(chains.size());
    std::vector<int> host_chain_offsets(static_cast<std::size_t>(n_chains) + 1U,
                                        0);
    std::vector<int> host_chain_cells;
    for (int chain = 0; chain < n_chains; ++chain) {
      const auto& cells = chains[static_cast<std::size_t>(chain)];
      host_chain_cells.insert(host_chain_cells.end(), cells.begin(), cells.end());
      host_chain_offsets[static_cast<std::size_t>(chain) + 1U] =
          static_cast<int>(host_chain_cells.size());
    }
    upload(chain_offsets, host_chain_offsets);
    upload(chain_cells, host_chain_cells);
    upload(cell_chain, host_cell_chain);

    std::vector<std::uint8_t> host_nverts = state.mesh.cell_nverts;
    if (host_nverts.empty()) {
      host_nverts.assign(static_cast<std::size_t>(n_cells), 4U);
    }
    TENRYU_ASSERT(host_nverts.size() == static_cast<std::size_t>(n_cells),
                  "wake heat flux requires complete cell vertex counts");
    upload(cell_nverts, host_nverts);

    std::vector<std::uint8_t> host_excluded(static_cast<std::size_t>(n_cells),
                                            0U);
    if (cfg.numerics.ale.euler_window.enabled &&
        cfg.numerics.ale.euler_window.role == "axis_survival_core") {
      const auto observation = axis_core_disk_observe(state);
      for (const int c : observation.plateau_cells) {
        if (c >= 0 && c < n_cells) {
          host_excluded[static_cast<std::size_t>(c)] = 1U;
        }
      }
    }
    upload(excluded, host_excluded);

    dominant_material.reset(static_cast<std::size_t>(n_cells));
    centroid_r.reset(static_cast<std::size_t>(n_cells));
    centroid_z.reset(static_cast<std::size_t>(n_cells));
    velocity_r.reset(static_cast<std::size_t>(n_cells));
    velocity_z.reset(static_cast<std::size_t>(n_cells));
    radial_length.reset(static_cast<std::size_t>(n_cells));
    av_rate.reset(static_cast<std::size_t>(n_cells));
    chain_max.reset(static_cast<std::size_t>(n_chains));
    sensor.reset(static_cast<std::size_t>(n_cells));
    front_mask.reset(static_cast<std::size_t>(n_cells));
    gate.reset(static_cast<std::size_t>(n_cells));
    raw_conductance.reset(static_cast<std::size_t>(n_angular_faces));
    cell_scale.reset(static_cast<std::size_t>(n_cells));
    face_delta_u.reset(static_cast<std::size_t>(n_angular_faces));
  }

  void update_dominant_material(const core::State& state,
                                const core::Config& cfg) {
    const int n_materials = static_cast<int>(cfg.materials.materials.size());
    std::vector<int> host(static_cast<std::size_t>(n_cells), 0);
    if (n_materials > 1) {
      const std::size_t expected = static_cast<std::size_t>(n_cells) *
                                   static_cast<std::size_t>(n_materials);
      TENRYU_ASSERT(state.volFrac.size() == expected,
                    "wake heat flux requires volFrac size consistency");
      std::vector<double> volfrac;
      state.volFrac.copy_to_host(volfrac);
      for (int c = 0; c < n_cells; ++c) {
        double best = -1.0;
        int best_material = 0;
        const std::size_t base = static_cast<std::size_t>(c) *
                                 static_cast<std::size_t>(n_materials);
        for (int material = 0; material < n_materials; ++material) {
          const double raw = volfrac[base + static_cast<std::size_t>(material)];
          const double fraction =
              std::isfinite(raw) && raw > 0.0 ? raw : 0.0;
          if (fraction > best) {
            best = fraction;
            best_material = material;
          }
        }
        host[static_cast<std::size_t>(c)] = best_material;
      }
    }
    dominant_material.copy_from_host(host);
  }
};

std::unique_ptr<Runtime> g_runtime;

__global__ void compute_cell_kinematics_kernel(
    double* __restrict__ centroid_r,
    double* __restrict__ centroid_z,
    double* __restrict__ velocity_r,
    double* __restrict__ velocity_z,
    double* __restrict__ radial_length,
    double* __restrict__ av_rate,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const double* __restrict__ node_velocity_r,
    const double* __restrict__ node_velocity_z,
    const double* __restrict__ mass,
    const double* __restrict__ work_av_power,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double dt,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const int nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  const int offset = cell_node_offsets[c];
  double twice_area = 0.0;
  double centroid_numer_r = 0.0;
  double centroid_numer_z = 0.0;
  double mean_r = 0.0;
  double mean_z = 0.0;
  double mean_vr = 0.0;
  double mean_vz = 0.0;
  double radius_min = 1.0e300;
  double radius_max = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int node0 = cell_node_indices[offset + k];
    const int node1 = cell_node_indices[offset + (k + 1) % nverts];
    const double r0 = node_r[node0];
    const double z0 = node_z[node0];
    const double r1 = node_r[node1];
    const double z1 = node_z[node1];
    const double cross = r0 * z1 - r1 * z0;
    twice_area += cross;
    centroid_numer_r += (r0 + r1) * cross;
    centroid_numer_z += (z0 + z1) * cross;
    mean_r += r0;
    mean_z += z0;
    mean_vr += node_velocity_r[node0];
    mean_vz += node_velocity_z[node0];
    const double radius = hypot(r0, z0);
    radius_min = fmin(radius_min, radius);
    radius_max = fmax(radius_max, radius);
  }
  const double inv_nverts = 1.0 / static_cast<double>(nverts);
  if (fabs(twice_area) > kTiny) {
    centroid_r[c] = centroid_numer_r / (3.0 * twice_area);
    centroid_z[c] = centroid_numer_z / (3.0 * twice_area);
  } else {
    centroid_r[c] = mean_r * inv_nverts;
    centroid_z[c] = mean_z * inv_nverts;
  }
  velocity_r[c] = mean_vr * inv_nverts;
  velocity_z[c] = mean_vz * inv_nverts;
  radial_length[c] = fmax(radius_max - radius_min, 0.0);
  const double deposited_work = fmax(work_av_power[c], 0.0) * dt;
  av_rate[c] = deposited_work /
               fmax(fmax(mass[c], 0.0) * dt, kTiny);
}

__global__ void reduce_chain_max_kernel(
    double* __restrict__ chain_max,
    const double* __restrict__ av_rate,
    const int* __restrict__ chain_offsets,
    const int* __restrict__ chain_cells,
    const int n_chains) {
  const int chain = blockIdx.x;
  if (chain >= n_chains) {
    return;
  }
  __shared__ double values[kThreads];
  __shared__ int cell_ids[kThreads];
  const int begin = chain_offsets[chain];
  const int end = chain_offsets[chain + 1];
  double best = -1.0;
  int best_cell = 0x7fffffff;
  for (int pos = begin + threadIdx.x; pos < end; pos += blockDim.x) {
    const int cell = chain_cells[pos];
    const double value = av_rate[cell];
    if (value > best || (value == best && cell < best_cell)) {
      best = value;
      best_cell = cell;
    }
  }
  values[threadIdx.x] = best;
  cell_ids[threadIdx.x] = best_cell;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      const double other_value = values[threadIdx.x + stride];
      const int other_cell = cell_ids[threadIdx.x + stride];
      if (other_value > values[threadIdx.x] ||
          (other_value == values[threadIdx.x] &&
           other_cell < cell_ids[threadIdx.x])) {
        values[threadIdx.x] = other_value;
        cell_ids[threadIdx.x] = other_cell;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    chain_max[chain] = fmax(values[0], 0.0);
  }
}

__device__ bool face_nodes(const int cell,
                           const int local_face,
                           const int* __restrict__ cell_node_offsets,
                           const int* __restrict__ cell_node_indices,
                           const std::uint8_t* __restrict__ cell_nverts,
                           int* node0,
                           int* node1) {
  const int nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, cell);
  int corner0 = 0;
  int corner1 = 0;
  if (!mesh::mesh_topo_active_local_face_corners(
          nverts, local_face, &corner0, &corner1)) {
    return false;
  }
  const int offset = cell_node_offsets[cell];
  *node0 = cell_node_indices[offset + corner0];
  *node1 = cell_node_indices[offset + corner1];
  return true;
}

__global__ void compute_sensor_and_history_kernel(
    double* __restrict__ sensor,
    std::uint8_t* __restrict__ front_mask,
    double* __restrict__ gate,
    double* __restrict__ eta,
    double* __restrict__ zeta,
    const double* __restrict__ av_rate,
    const double* __restrict__ chain_max,
    const int* __restrict__ cell_chain,
    const int* __restrict__ radial_adj_offsets,
    const int* __restrict__ radial_adj_indices,
    const int* __restrict__ radial_cell_a,
    const int* __restrict__ radial_cell_b,
    const int* __restrict__ radial_local_a,
    const double* __restrict__ centroid_r,
    const double* __restrict__ centroid_z,
    const double* __restrict__ velocity_r,
    const double* __restrict__ velocity_z,
    const double* __restrict__ radial_length,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const double* __restrict__ pressure_e,
    const double* __restrict__ pressure_i,
    const double* __restrict__ sound_speed,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ excluded,
    const std::int8_t* __restrict__ hydro_active,
    const double theta_a,
    const double theta_b,
    const int global_theta,
    const double dt,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (excluded[c] != 0U ||
      (hydro_active != nullptr && hydro_active[c] == 0)) {
    sensor[c] = 0.0;
    front_mask[c] = 0U;
    gate[c] = 0.0;
    return;
  }

  const int chain = cell_chain[c];
  const double maximum = chain >= 0 ? chain_max[chain] : 0.0;
  const double activity = av_rate[c] / (maximum + kTiny);
  const double av_sensor = fmin(fmax((activity - 0.01) / 0.09, 0.0), 1.0);

  double compression = 0.0;
  double pressure_jump = 0.0;
  const double pressure_c = fmax(pressure_e[c] + pressure_i[c], 0.0);
  for (int pos = radial_adj_offsets[c]; pos < radial_adj_offsets[c + 1]; ++pos) {
    const int encoded = radial_adj_indices[pos];
    const int face = encoded > 0 ? encoded - 1 : -encoded - 1;
    const int a = radial_cell_a[face];
    const int b = radial_cell_b[face];
    const int d = c == a ? b : a;
    int node0 = -1;
    int node1 = -1;
    if (!face_nodes(a, radial_local_a[face], cell_node_offsets,
                    cell_node_indices, cell_nverts, &node0, &node1)) {
      continue;
    }
    const double dr = node_r[node1] - node_r[node0];
    const double dz = node_z[node1] - node_z[node0];
    const double length = hypot(dr, dz);
    if (!(length > kTiny)) {
      continue;
    }
    double normal_r = dz / length;
    double normal_z = -dr / length;
    const double delta_centroid_r = centroid_r[d] - centroid_r[c];
    const double delta_centroid_z = centroid_z[d] - centroid_z[c];
    if (delta_centroid_r * normal_r + delta_centroid_z * normal_z < 0.0) {
      normal_r = -normal_r;
      normal_z = -normal_z;
    }
    const double relative_normal_velocity =
        (velocity_r[d] - velocity_r[c]) * normal_r +
        (velocity_z[d] - velocity_z[c]) * normal_z;
    const double mach = fmax(-relative_normal_velocity, 0.0) /
                        fmax(sound_speed[c] + sound_speed[d], kTiny);
    compression = fmax(compression, mach);
    const double pressure_d = fmax(pressure_e[d] + pressure_i[d], 0.0);
    const double jump = fabs(pressure_d - pressure_c) /
                        fmax(pressure_d + pressure_c + 1.0e-30, 1.0e-30);
    pressure_jump = fmax(pressure_jump, jump);
  }

  const double kinematic_sensor =
      fmin(fmax((compression - 0.03) / 0.07, 0.0), 1.0) *
      fmin(fmax((pressure_jump - 0.05) / 0.15, 0.0), 1.0);
  const double hit = fmax(av_sensor, kinematic_sensor);
  const bool at_front = activity > 0.01 ||
                        (compression > 0.05 && pressure_jump > 0.10);
  sensor[c] = hit;
  front_mask[c] = at_front ? 1U : 0U;

  double pole_weight = 1.0;
  if (global_theta == 0) {
    const double theta = atan2(fmax(centroid_r[c], 0.0), centroid_z[c]);
    const double polar_angle = fmin(theta, kPi - theta);
    if (polar_angle >= theta_b) {
      pole_weight = 0.0;
    } else if (polar_angle > theta_a) {
      const double xi = (polar_angle - theta_a) / (theta_b - theta_a);
      pole_weight = 1.0 - 3.0 * xi * xi + 2.0 * xi * xi * xi;
    }
  }
  const double old_eta = fmin(fmax(eta[c], 0.0), 1.0);
  const double old_zeta = fmin(fmax(zeta[c], 0.0), 1.0);
  const double one_minus_hit = 1.0 - hit;
  gate[c] = pole_weight * old_zeta * one_minus_hit * one_minus_hit *
            (at_front ? 0.0 : 1.0);

  const double crossing_time =
      radial_length[c] / fmax(sound_speed[c], 1.0e-30);
  const double eta_decay = exp(-dt / fmax(3.0 * crossing_time, kTiny));
  const double zeta_decay = exp(-dt / fmax(3.0 * crossing_time, kTiny));
  const double lag_gain =
      1.0 - exp(-dt / fmax(0.5 * crossing_time, kTiny));
  eta[c] = fmin(fmax(fmax(hit, old_eta * eta_decay), 0.0), 1.0);
  zeta[c] = fmin(fmax(old_zeta * zeta_decay + lag_gain * old_eta, 0.0), 1.0);
}

__device__ double cell_heat_capacity(const double* __restrict__ cv_e,
                                     const double* __restrict__ cv_i,
                                     const int c,
                                     const int two_temperature) {
  const double electron = fmax(cv_e[c], 0.0);
  return two_temperature != 0 ? electron + fmax(cv_i[c], 0.0) : electron;
}

__device__ double cell_temperature(const double* __restrict__ temperature_e,
                                   const double* __restrict__ temperature_i,
                                   const double* __restrict__ cv_e,
                                   const double* __restrict__ cv_i,
                                   const int c,
                                   const int two_temperature) {
  if (two_temperature == 0) {
    return temperature_e[c];
  }
  const double electron = fmax(cv_e[c], 0.0);
  const double ion = fmax(cv_i[c], 0.0);
  const double total = electron + ion;
  return total > kTiny
             ? (electron * temperature_e[c] + ion * temperature_i[c]) / total
             : 0.5 * (temperature_e[c] + temperature_i[c]);
}

__global__ void compute_raw_conductance_kernel(
    double* __restrict__ raw_conductance,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const double* __restrict__ centroid_r,
    const double* __restrict__ centroid_z,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const double* __restrict__ volume,
    const double* __restrict__ density,
    const double* __restrict__ cv_e,
    const double* __restrict__ cv_i,
    const double* __restrict__ sound_speed,
    const double* __restrict__ gate,
    const int* __restrict__ dominant_material,
    const double* __restrict__ plic_interface_mask,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::int8_t* __restrict__ hydro_active,
    const double coefficient,
    const int two_temperature,
    const int n_faces) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int c = face_cell_a[f];
  const int d = face_cell_b[f];
  raw_conductance[f] = 0.0;
  if ((hydro_active != nullptr &&
       (hydro_active[c] == 0 || hydro_active[d] == 0)) ||
      dominant_material[c] != dominant_material[d] ||
      (plic_interface_mask != nullptr &&
       (plic_interface_mask[c] > 0.0 || plic_interface_mask[d] > 0.0))) {
    return;
  }
  int node0 = -1;
  int node1 = -1;
  if (!face_nodes(c, face_local_a[f], cell_node_offsets, cell_node_indices,
                  cell_nverts, &node0, &node1)) {
    return;
  }
  const double r0 = node_r[node0];
  const double z0 = node_z[node0];
  const double r1 = node_r[node1];
  const double z1 = node_z[node1];
  const double dr = r1 - r0;
  const double dz = z1 - z0;
  const double length = hypot(dr, dz);
  if (!(length > kTiny)) {
    return;
  }
  const double area = kPi * fmax(r0 + r1, 0.0) * length;
  if (!(area > kTiny)) {
    return;
  }
  const double normal_r = dz / length;
  const double normal_z = -dr / length;
  const double distance =
      fmax(fabs((centroid_r[d] - centroid_r[c]) * normal_r +
                (centroid_z[d] - centroid_z[c]) * normal_z),
           kTiny);
  const double h_c = fmax(volume[c], 0.0) / area;
  const double h_d = fmax(volume[d], 0.0) / area;
  const double face_length = 2.0 * h_c * h_d / (h_c + h_d + kTiny);
  const double rho_c = fmax(density[c], 0.0);
  const double rho_d = fmax(density[d], 0.0);
  const double face_density =
      2.0 * rho_c * rho_d / (rho_c + rho_d + kTiny);
  const double capacity_c = cell_heat_capacity(cv_e, cv_i, c, two_temperature);
  const double capacity_d = cell_heat_capacity(cv_e, cv_i, d, two_temperature);
  const double face_capacity =
      2.0 * capacity_c * capacity_d /
      (capacity_c + capacity_d + kTiny);
  const double face_sound_speed =
      fmax(fmin(sound_speed[c], sound_speed[d]), 0.0);
  const double face_gate = fmax(fmin(gate[c], gate[d]), 0.0);
  const double conductivity = coefficient * face_density * face_capacity *
                              face_sound_speed * face_length * face_gate;
  raw_conductance[f] = area * conductivity / distance;
}

__global__ void compute_cell_scale_kernel(
    double* __restrict__ cell_scale,
    const double* __restrict__ raw_conductance,
    const int* __restrict__ adjacency_offsets,
    const int* __restrict__ adjacency_indices,
    const double* __restrict__ mass,
    const double* __restrict__ cv_e,
    const double* __restrict__ cv_i,
    const std::uint8_t* __restrict__ excluded,
    const std::int8_t* __restrict__ hydro_active,
    const double dt,
    const int two_temperature,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (excluded[c] != 0U ||
      (hydro_active != nullptr && hydro_active[c] == 0)) {
    cell_scale[c] = 0.0;
    return;
  }
  double sum = 0.0;
  for (int pos = adjacency_offsets[c]; pos < adjacency_offsets[c + 1]; ++pos) {
    const int encoded = adjacency_indices[pos];
    const int face = encoded > 0 ? encoded - 1 : -encoded - 1;
    sum += raw_conductance[face];
  }
  const double capacity = cell_heat_capacity(cv_e, cv_i, c, two_temperature);
  const double lambda = dt * sum /
                        fmax(fmax(mass[c], 0.0) * capacity, kTiny);
  cell_scale[c] = fmin(1.0, 0.45 / fmax(lambda, kTiny));
}

__global__ void compute_face_exchange_kernel(
    double* __restrict__ face_delta_u,
    const double* __restrict__ raw_conductance,
    const double* __restrict__ cell_scale,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const double* __restrict__ temperature_e,
    const double* __restrict__ temperature_i,
    const double* __restrict__ cv_e,
    const double* __restrict__ cv_i,
    const double dt,
    const int two_temperature,
    const int n_faces) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int c = face_cell_a[f];
  const int d = face_cell_b[f];
  const double conductance =
      fmin(cell_scale[c], cell_scale[d]) * raw_conductance[f];
  const double temperature_c =
      cell_temperature(temperature_e, temperature_i, cv_e, cv_i, c,
                       two_temperature);
  const double temperature_d =
      cell_temperature(temperature_e, temperature_i, cv_e, cv_i, d,
                       two_temperature);
  face_delta_u[f] = dt * conductance * (temperature_d - temperature_c);
}

__global__ void gather_energy_kernel(
    double* __restrict__ energy_e,
    double* __restrict__ energy_i,
    const double* __restrict__ face_delta_u,
    const int* __restrict__ adjacency_offsets,
    const int* __restrict__ adjacency_indices,
    const double* __restrict__ mass,
    const double* __restrict__ pressure_e,
    const double* __restrict__ pressure_i,
    const double* __restrict__ cv_e,
    const double* __restrict__ cv_i,
    const std::uint8_t* __restrict__ excluded,
    const std::int8_t* __restrict__ hydro_active,
    const int two_temperature,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells || excluded[c] != 0U ||
      (hydro_active != nullptr && hydro_active[c] == 0)) {
    return;
  }
  double delta_u = 0.0;
  for (int pos = adjacency_offsets[c]; pos < adjacency_offsets[c + 1]; ++pos) {
    const int encoded = adjacency_indices[pos];
    const int face = encoded > 0 ? encoded - 1 : -encoded - 1;
    const double face_value = face_delta_u[face];
    delta_u += encoded > 0 ? face_value : -face_value;
  }
  const double inv_mass = 1.0 / fmax(mass[c], kTiny);
  if (two_temperature == 0) {
    energy_e[c] += delta_u * inv_mass;
    return;
  }
  const double pe = fmax(pressure_e[c], 0.0);
  const double pi = fmax(pressure_i[c], 0.0);
  const double pressure_sum = pe + pi;
  double electron_fraction = 0.5;
  if (pressure_sum > 1.0e-30) {
    electron_fraction = pe / pressure_sum;
  } else {
    const double electron_capacity = fmax(cv_e[c], 0.0);
    const double ion_capacity = fmax(cv_i[c], 0.0);
    const double capacity_sum = electron_capacity + ion_capacity;
    if (capacity_sum > kTiny) {
      electron_fraction = electron_capacity / capacity_sum;
    }
  }
  energy_e[c] += electron_fraction * delta_u * inv_mass;
  energy_i[c] += (1.0 - electron_fraction) * delta_u * inv_mass;
}

}  // namespace

bool apply(core::State& state, const core::Config& cfg, const double dt) {
  if (!cfg.numerics.hydro.wake_heat_flux_enabled) {
    return false;
  }
  TENRYU_ASSERT(dt > 0.0, "wake heat flux requires positive dt");
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "wake heat flux requires multiblock topology");
  const int n_cells = state.mesh.topo.n_cells;
  TENRYU_ASSERT(state.wake_heat_flux_eta.size() ==
                        static_cast<std::size_t>(n_cells) &&
                    state.wake_heat_flux_zeta.size() ==
                        static_cast<std::size_t>(n_cells) &&
                    state.work_av_per_cell.size() ==
                        static_cast<std::size_t>(n_cells) &&
                    state.cv_e.size() == static_cast<std::size_t>(n_cells) &&
                    state.cv_i.size() == static_cast<std::size_t>(n_cells) &&
                    state.cs.size() == static_cast<std::size_t>(n_cells),
                "wake heat flux state arrays are not allocated");
  TENRYU_ASSERT(
      state.mesh.multiblock_cell_node_csr_offsets.size() ==
              static_cast<std::size_t>(n_cells) + 1U &&
          !state.mesh.multiblock_cell_node_csr_indices.empty(),
      "wake heat flux requires device cell-node CSR metadata");
  if (g_runtime == nullptr ||
      g_runtime->topology != &*state.mesh.topo.multiblock ||
      g_runtime->n_cells != n_cells) {
    g_runtime = std::make_unique<Runtime>(state, cfg);
  }
  Runtime& runtime = *g_runtime;
  runtime.update_dominant_material(state, cfg);

  const int cell_blocks = (n_cells + kThreads - 1) / kThreads;
  compute_cell_kinematics_kernel<<<cell_blocks, kThreads>>>(
      runtime.centroid_r.data(), runtime.centroid_z.data(),
      runtime.velocity_r.data(), runtime.velocity_z.data(),
      runtime.radial_length.data(), runtime.av_rate.data(), state.x_r.data(),
      state.x_z.data(), state.v_r.data(), state.v_z.data(), state.mass.data(),
      state.work_av_per_cell.data(),
      state.mesh.multiblock_cell_node_csr_offsets.data(),
      state.mesh.multiblock_cell_node_csr_indices.data(),
      runtime.cell_nverts.data(), dt, n_cells);
  sync_kernel("wake heat flux cell kinematics failed");

  if (runtime.n_chains > 0) {
    reduce_chain_max_kernel<<<runtime.n_chains, kThreads>>>(
        runtime.chain_max.data(), runtime.av_rate.data(),
        runtime.chain_offsets.data(), runtime.chain_cells.data(),
        runtime.n_chains);
    sync_kernel("wake heat flux chain reduction failed");
  }

  const std::int8_t* hydro_active = state.hydro_active_device_ptr();
  compute_sensor_and_history_kernel<<<cell_blocks, kThreads>>>(
      runtime.sensor.data(), runtime.front_mask.data(), runtime.gate.data(),
      state.wake_heat_flux_eta.data(), state.wake_heat_flux_zeta.data(),
      runtime.av_rate.data(), runtime.chain_max.data(), runtime.cell_chain.data(),
      runtime.radial_adj_offsets.data(), runtime.radial_adj_indices.data(),
      runtime.radial_cell_a.data(), runtime.radial_cell_b.data(),
      runtime.radial_local_a.data(), runtime.centroid_r.data(),
      runtime.centroid_z.data(), runtime.velocity_r.data(),
      runtime.velocity_z.data(), runtime.radial_length.data(), state.x_r.data(),
      state.x_z.data(), state.Pe.data(), state.Pi.data(), state.cs.data(),
      state.mesh.multiblock_cell_node_csr_offsets.data(),
      state.mesh.multiblock_cell_node_csr_indices.data(),
      runtime.cell_nverts.data(), runtime.excluded.data(), hydro_active,
      cfg.numerics.hydro.wake_heat_flux_theta_a_rad,
      cfg.numerics.hydro.wake_heat_flux_theta_b_rad,
      cfg.numerics.hydro.wake_heat_flux_global_theta ? 1 : 0, dt, n_cells);
  sync_kernel("wake heat flux sensor/history update failed");

  if (!(cfg.numerics.hydro.wake_heat_flux_CE > 0.0) ||
      runtime.n_angular_faces == 0) {
    return false;
  }

  const int face_blocks =
      (runtime.n_angular_faces + kThreads - 1) / kThreads;
  const double* plic_interface_mask =
      state.plic_interface_mask.size() == static_cast<std::size_t>(n_cells)
          ? state.plic_interface_mask.data()
          : nullptr;
  compute_raw_conductance_kernel<<<face_blocks, kThreads>>>(
      runtime.raw_conductance.data(), runtime.angular_cell_a.data(),
      runtime.angular_cell_b.data(), runtime.angular_local_a.data(),
      runtime.centroid_r.data(), runtime.centroid_z.data(), state.x_r.data(),
      state.x_z.data(), state.vol.data(), state.rho.data(), state.cv_e.data(),
      state.cv_i.data(), state.cs.data(), runtime.gate.data(),
      runtime.dominant_material.data(), plic_interface_mask,
      state.mesh.multiblock_cell_node_csr_offsets.data(),
      state.mesh.multiblock_cell_node_csr_indices.data(),
      runtime.cell_nverts.data(), hydro_active,
      cfg.numerics.hydro.wake_heat_flux_CE,
      cfg.main.two_temperature ? 1 : 0, runtime.n_angular_faces);
  sync_kernel("wake heat flux raw conductance failed");

  compute_cell_scale_kernel<<<cell_blocks, kThreads>>>(
      runtime.cell_scale.data(), runtime.raw_conductance.data(),
      runtime.angular_adj_offsets.data(), runtime.angular_adj_indices.data(),
      state.mass.data(), state.cv_e.data(), state.cv_i.data(),
      runtime.excluded.data(), hydro_active, dt,
      cfg.main.two_temperature ? 1 : 0, n_cells);
  sync_kernel("wake heat flux diffusion cap failed");

  compute_face_exchange_kernel<<<face_blocks, kThreads>>>(
      runtime.face_delta_u.data(), runtime.raw_conductance.data(),
      runtime.cell_scale.data(), runtime.angular_cell_a.data(),
      runtime.angular_cell_b.data(), state.Te.data(), state.Ti.data(),
      state.cv_e.data(), state.cv_i.data(), dt,
      cfg.main.two_temperature ? 1 : 0, runtime.n_angular_faces);
  sync_kernel("wake heat flux face exchange failed");

  gather_energy_kernel<<<cell_blocks, kThreads>>>(
      state.ee.data(), state.ei.data(), runtime.face_delta_u.data(),
      runtime.angular_adj_offsets.data(), runtime.angular_adj_indices.data(),
      state.mass.data(), state.Pe.data(), state.Pi.data(), state.cv_e.data(),
      state.cv_i.data(), runtime.excluded.data(), hydro_active,
      cfg.main.two_temperature ? 1 : 0, n_cells);
  sync_kernel("wake heat flux deterministic energy gather failed");
  return true;
}

void destroy() {
  g_runtime.reset();
}

}  // namespace tenryu::hydro::wake_angular_heat_flux
