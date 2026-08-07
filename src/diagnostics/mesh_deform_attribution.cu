#include "diagnostics/mesh_deform_attribution.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

#include <cuda_runtime.h>

#include "core/error.hpp"

namespace tenryu::diagnostics::mesh_attribution {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline bool stage24_telemetry_enabled() {
  static const bool enabled = []() {
    const char* raw = std::getenv("TENRYU_STAGE24_TELEMETRY");
    if (raw == nullptr || raw[0] == '\0') {
      return false;
    }
    return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0 &&
           std::strcmp(raw, "FALSE") != 0;
  }();
  return enabled;
}

int source_index(const MeshDeformSource source) {
  return static_cast<int>(source);
}

__global__ void add_delta_kernel(double* __restrict__ dst_r,
                                 double* __restrict__ dst_z,
                                 const double* __restrict__ delta_r,
                                 const double* __restrict__ delta_z,
                                 const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  dst_r[n] += delta_r[n];
  dst_z[n] += delta_z[n];
}

__global__ void add_scaled_node_vector_kernel(double* __restrict__ dst_r,
                                              double* __restrict__ dst_z,
                                              const double* __restrict__ v_r,
                                              const double* __restrict__ v_z,
                                              const std::uint8_t* __restrict__ node_active,
                                              const int n_nodes,
                                              const double scale) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  if (node_active != nullptr && node_active[n] == 0u) {
    return;
  }
  dst_r[n] += scale * v_r[n];
  dst_z[n] += scale * v_z[n];
}

__global__ void add_direct_displacement_kernel(double* __restrict__ dst_r,
                                               double* __restrict__ dst_z,
                                               const double* __restrict__ r_before,
                                               const double* __restrict__ z_before,
                                               const double* __restrict__ r_after,
                                               const double* __restrict__ z_after,
                                               const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  dst_r[n] += r_after[n] - r_before[n];
  dst_z[n] += z_after[n] - z_before[n];
}

__device__ __forceinline__ void cell_nodes(const int c,
                                           const int nz,
                                           int nodes[4]) {
  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  nodes[0] = i * stride + j;
  nodes[1] = (i + 1) * stride + j;
  nodes[2] = (i + 1) * stride + (j + 1);
  nodes[3] = i * stride + (j + 1);
}

__device__ __forceinline__ double corner_j_from_nodes(
    const double* __restrict__ r,
    const double* __restrict__ z,
    const int nodes[4],
    const int corner) {
  const int kp = (corner + 1) & 3;
  const int km = (corner + 3) & 3;
  const int nc = nodes[corner];
  const int np = nodes[kp];
  const int nm = nodes[km];
  return device::cross2(r[np] - r[nc], z[np] - z[nc],
                        r[nm] - r[nc], z[nm] - z[nc]);
}

__device__ __forceinline__ double source_corner_dj(
    const double* __restrict__ start_r,
    const double* __restrict__ start_z,
    const double* __restrict__ delta_r,
    const double* __restrict__ delta_z,
    const int nodes[4],
    const int corner) {
  const int kp = (corner + 1) & 3;
  const int km = (corner + 3) & 3;
  const int nc = nodes[corner];
  const int np = nodes[kp];
  const int nm = nodes[km];
  const double a0_r = start_r[np] - start_r[nc];
  const double a0_z = start_z[np] - start_z[nc];
  const double b0_r = start_r[nm] - start_r[nc];
  const double b0_z = start_z[nm] - start_z[nc];
  const double da_r = delta_r[np] - delta_r[nc];
  const double da_z = delta_z[np] - delta_z[nc];
  const double db_r = delta_r[nm] - delta_r[nc];
  const double db_z = delta_z[nm] - delta_z[nc];
  return device::cross2(da_r, da_z, b0_r, b0_z) +
         device::cross2(a0_r, a0_z, db_r, db_z);
}

__global__ void compute_failure_dj_kernel(
    double* __restrict__ out_dj,
    double* __restrict__ out_actual_delta_j,
    const double* __restrict__ start_r,
    const double* __restrict__ start_z,
    const double* __restrict__ candidate_r,
    const double* __restrict__ candidate_z,
    const double* __restrict__ delta_r,
    const double* __restrict__ delta_z,
    const int nr,
    const int nz,
    const int failing_cell,
    const int failing_corner,
    const int n_nodes) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  for (int s = 0; s < kMeshDeformSourceCount; ++s) {
    out_dj[s] = 0.0;
  }
  out_actual_delta_j[0] = 0.0;
  const int n_cells = nr * nz;
  if (failing_cell < 0 || failing_cell >= n_cells ||
      failing_corner < 0 || failing_corner >= 4) {
    return;
  }
  int nodes[4];
  cell_nodes(failing_cell, nz, nodes);
  for (int k = 0; k < 4; ++k) {
    if (nodes[k] < 0 || nodes[k] >= n_nodes) {
      return;
    }
  }
  const double j0 = corner_j_from_nodes(start_r, start_z, nodes, failing_corner);
  const double j1 = corner_j_from_nodes(candidate_r, candidate_z, nodes, failing_corner);
  out_actual_delta_j[0] = j1 - j0;
  for (int s = 0; s < kMeshDeformSourceCount; ++s) {
    out_dj[s] = source_corner_dj(start_r,
                                 start_z,
                                 delta_r + static_cast<std::size_t>(s) * n_nodes,
                                 delta_z + static_cast<std::size_t>(s) * n_nodes,
                                 nodes,
                                 failing_corner);
  }
}

__global__ void compute_aggregate_dj_kernel(
    double* __restrict__ out_dj,
    const double* __restrict__ start_r,
    const double* __restrict__ start_z,
    const double* __restrict__ delta_r,
    const double* __restrict__ delta_z,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const int n_nodes) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }
  int nodes[4];
  cell_nodes(c, nz, nodes);
  for (int k = 0; k < 4; ++k) {
    if (nodes[k] < 0 || nodes[k] >= n_nodes) {
      return;
    }
  }
  double best_degrade = -1.0;
  int best_corner = 0;
  for (int corner = 0; corner < 4; ++corner) {
    double total_dj = 0.0;
    for (int s = 0; s < kMeshDeformSourceCount; ++s) {
      total_dj += source_corner_dj(start_r,
                                   start_z,
                                   delta_r + static_cast<std::size_t>(s) * n_nodes,
                                   delta_z + static_cast<std::size_t>(s) * n_nodes,
                                   nodes,
                                   corner);
    }
    const double degrade = fmax(0.0, -total_dj);
    if (degrade > best_degrade) {
      best_degrade = degrade;
      best_corner = corner;
    }
  }
  for (int s = 0; s < kMeshDeformSourceCount; ++s) {
    const double dj = source_corner_dj(
        start_r,
        start_z,
        delta_r + static_cast<std::size_t>(s) * n_nodes,
        delta_z + static_cast<std::size_t>(s) * n_nodes,
        nodes,
        best_corner);
    atomicAdd(out_dj + s, fmax(0.0, -dj));
  }
}

std::filesystem::path attribution_path() {
  if (const char* explicit_outdir = std::getenv("TENRYU_DT_LINEAGE_OUTDIR");
      explicit_outdir != nullptr && explicit_outdir[0] != '\0') {
    return std::filesystem::path(explicit_outdir) / "mesh_failure_attribution.jsonl";
  }
  if (const char* tenryu_outdir = std::getenv("TENRYU_OUTDIR");
      tenryu_outdir != nullptr && tenryu_outdir[0] != '\0') {
    return std::filesystem::path(tenryu_outdir) / "diag" /
           "mesh_failure_attribution.jsonl";
  }
  return std::filesystem::path("mesh_failure_attribution.jsonl");
}

void write_json_number(std::ostream& os, const char* key, const double value) {
  os << ",\"" << key << "\":";
  if (std::isfinite(value)) {
    os << value;
  } else {
    os << "null";
  }
}

void finalize_record(MeshFailureAttributionRecord& record,
                     const bool aggregate,
                     const double actual_delta_j) {
  double degradation_sum = 0.0;
  double dominant = 0.0;
  int dominant_source = source_index(MeshDeformSource::Other);
  double linear_sum = 0.0;
  for (int s = 0; s < kMeshDeformSourceCount; ++s) {
    const double value = record.dJ_source[s];
    linear_sum += value;
    const double degradation = aggregate ? fmax(0.0, value) : fmax(0.0, -value);
    degradation_sum += degradation;
    if (degradation > dominant) {
      dominant = degradation;
      dominant_source = s;
    }
  }
  record.dominant_source = static_cast<std::uint8_t>(dominant_source);
  record.dominant_fraction =
      (degradation_sum > 0.0) ? (dominant / degradation_sum) : 0.0;
  record.nonlinear_residual = aggregate ? 0.0 : (linear_sum - actual_delta_j);
}

void write_record_jsonl(const MeshFailureAttributionRecord& record) {
  const std::filesystem::path path = attribution_path();
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream os(path, std::ios::out | std::ios::app);
  TENRYU_ASSERT(os.good(), "mesh attribution diagnostics failed to open JSONL output");
  os << std::scientific << std::setprecision(17);
  os << "{\"step\":" << record.step;
  write_json_number(os, "t_simulation", record.t_simulation);
  os << ",\"mesh_failure_id\":" << record.mesh_failure_id;
  os << ",\"failing_cell\":" << record.failing_cell;
  os << ",\"failing_corner\":" << record.failing_corner;
  os << ",\"dJ_source\":[";
  for (int s = 0; s < kMeshDeformSourceCount; ++s) {
    if (s > 0) {
      os << ",";
    }
    if (std::isfinite(record.dJ_source[s])) {
      os << record.dJ_source[s];
    } else {
      os << "null";
    }
  }
  os << "]";
  os << ",\"dominant_source\":" << static_cast<int>(record.dominant_source);
  write_json_number(os, "dominant_fraction", record.dominant_fraction);
  write_json_number(os, "nonlinear_residual", record.nonlinear_residual);
  if (stage24_telemetry_enabled()) {
    auto write_source_array = [&os](const char* key, const double* values) {
      os << ",\"" << key << "\":[";
      for (int s = 0; s < kMeshDeformSourceCount; ++s) {
        if (s > 0) {
          os << ",";
        }
        if (std::isfinite(values[s])) {
          os << values[s];
        } else {
          os << "null";
        }
      }
      os << "]";
    };
    write_source_array("d_area_dt_by_source", record.d_area_dt_by_source);
    write_source_array("d_rzvol_dt_by_source", record.d_rzvol_dt_by_source);
    write_source_array("d_radial_gap_dt_by_source",
                       record.d_radial_gap_dt_by_source);
    write_source_array("d_z_order_dt_by_source", record.d_z_order_dt_by_source);
    write_source_array("d_remap_intermediate_dt_by_source",
                       record.d_remap_intermediate_dt_by_source);
    write_json_number(os, "dt_star_area", record.dt_star_area);
    write_json_number(os, "dt_star_rzvol", record.dt_star_rzvol);
    write_json_number(os, "dt_star_radial_gap", record.dt_star_radial_gap);
    write_json_number(os, "dt_star_z_order", record.dt_star_z_order);
    write_json_number(os,
                      "dt_star_remap_intermediate",
                      record.dt_star_remap_intermediate);
  }
  os << "}\n";
}

void write_axis_projection_record_jsonl(const AxisProjectionRecord& record) {
  const std::filesystem::path path = attribution_path();
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream os(path, std::ios::out | std::ios::app);
  TENRYU_ASSERT(os.good(), "mesh attribution diagnostics failed to open JSONL output");
  os << std::scientific << std::setprecision(17);
  os << "{\"record_kind\":\"axis_projection_attempt\"";
  os << ",\"step\":" << record.step;
  write_json_number(os, "t_simulation", record.t_simulation);
  os << ",\"operator_name\":\"" << record.operator_name << "\"";
  os << ",\"phase_tag\":\"" << record.phase_tag << "\"";
  os << ",\"patch_i0\":" << record.patch_i0;
  os << ",\"patch_i1\":" << record.patch_i1;
  os << ",\"patch_j0\":" << record.patch_j0;
  os << ",\"patch_j1\":" << record.patch_j1;
  write_json_number(os, "min_local_j", record.min_local_j);
  os << ",\"post_failing_cell\":" << record.post_failing_cell;
  os << ",\"post_failing_corner\":" << record.post_failing_corner;
  write_json_number(os, "pre_request_min_quality", record.pre_request_min_quality);
  write_json_number(os, "post_min_quality", record.post_min_quality);
  os << "}\n";
}

std::uint64_t next_failure_id() {
  static std::uint64_t id = 0;
  return id++;
}

}  // namespace

MeshDeformAttributionWorkspace::~MeshDeformAttributionWorkspace() {
  release();
}

void MeshDeformAttributionWorkspace::release() {
  const auto free_if_needed = [](auto*& ptr, const char* message) {
    if (ptr != nullptr) {
      cuda_check(cudaFree(ptr), message);
      ptr = nullptr;
    }
  };
  free_if_needed(d_hydro_active_, "mesh attribution cudaFree hydro_active failed");
  free_if_needed(d_actual_delta_j_, "mesh attribution cudaFree actual_dj failed");
  free_if_needed(d_dj_source_, "mesh attribution cudaFree dj_source failed");
  free_if_needed(d_delta_z_, "mesh attribution cudaFree delta_z failed");
  free_if_needed(d_delta_r_, "mesh attribution cudaFree delta_r failed");
  free_if_needed(d_start_z_, "mesh attribution cudaFree start_z failed");
  free_if_needed(d_start_r_, "mesh attribution cudaFree start_r failed");
  n_nodes_ = 0;
  n_cells_ = 0;
  active_ = false;
}

void MeshDeformAttributionWorkspace::ensure_capacity(const std::size_t n_nodes,
                                                     const std::size_t n_cells) {
  if (n_nodes == n_nodes_ && n_cells == n_cells_ && d_start_r_ != nullptr) {
    return;
  }
  release();
  n_nodes_ = n_nodes;
  n_cells_ = n_cells;
  const std::size_t node_bytes = n_nodes * sizeof(double);
  const std::size_t source_bytes =
      static_cast<std::size_t>(kMeshDeformSourceCount) * node_bytes;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_start_r_), node_bytes),
             "mesh attribution cudaMalloc start_r failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_start_z_), node_bytes),
             "mesh attribution cudaMalloc start_z failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_delta_r_), source_bytes),
             "mesh attribution cudaMalloc delta_r failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_delta_z_), source_bytes),
             "mesh attribution cudaMalloc delta_z failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_dj_source_),
                        kMeshDeformSourceCount * sizeof(double)),
             "mesh attribution cudaMalloc dj_source failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_actual_delta_j_), sizeof(double)),
             "mesh attribution cudaMalloc actual_dj failed");
  if (n_cells > 0) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_hydro_active_),
                          n_cells * sizeof(std::int8_t)),
               "mesh attribution cudaMalloc hydro_active failed");
  }
}

bool MeshDeformAttributionWorkspace::begin_step(const core::State& state,
                                                const core::Config& cfg,
                                                const double t_simulation,
                                                const char* hydro_half) {
  (void)hydro_half;
  active_ = false;
  if (!cfg.numerics.diagnostics.mesh_attribution.enabled ||
      !cfg.numerics.diagnostics.mesh_attribution.record_node_displacements ||
      state.mesh.dim != 2 || state.x_r.empty() || state.x_z.empty()) {
    release();
    return false;
  }
  const std::size_t n_nodes = state.x_r.size();
  const std::size_t n_cells = state.rho.size();
  TENRYU_ASSERT(state.x_z.size() == n_nodes,
                "mesh attribution requires matching node coordinate sizes");
  ensure_capacity(n_nodes, n_cells);
  const std::size_t node_bytes = n_nodes * sizeof(double);
  const std::size_t source_bytes =
      static_cast<std::size_t>(kMeshDeformSourceCount) * node_bytes;
  cuda_check(cudaMemcpy(d_start_r_, state.x_r.data(), node_bytes, cudaMemcpyDeviceToDevice),
             "mesh attribution copy start_r failed");
  cuda_check(cudaMemcpy(d_start_z_, state.x_z.data(), node_bytes, cudaMemcpyDeviceToDevice),
             "mesh attribution copy start_z failed");
  cuda_check(cudaMemset(d_delta_r_, 0, source_bytes),
             "mesh attribution clear delta_r failed");
  cuda_check(cudaMemset(d_delta_z_, 0, source_bytes),
             "mesh attribution clear delta_z failed");
  if (d_hydro_active_ != nullptr) {
    if (state.hydro_active.size() == n_cells) {
      cuda_check(cudaMemcpy(d_hydro_active_,
                            state.hydro_active.data(),
                            n_cells * sizeof(std::int8_t),
                            cudaMemcpyHostToDevice),
                 "mesh attribution copy hydro_active failed");
    } else {
      cuda_check(cudaMemset(d_hydro_active_, 1, n_cells * sizeof(std::int8_t)),
                 "mesh attribution default hydro_active failed");
    }
  }
  step_ = static_cast<std::uint64_t>(std::max(state.step, 0));
  t_simulation_ = t_simulation;
  dump_on_failure_only_ =
      cfg.numerics.diagnostics.mesh_attribution.dump_on_failure_only;
  active_ = true;
  return true;
}

void MeshDeformAttributionWorkspace::discard() {
  release();
}

double* MeshDeformAttributionWorkspace::source_delta_r(const MeshDeformSource source) {
  return d_delta_r_ + static_cast<std::size_t>(source_index(source)) * n_nodes_;
}

double* MeshDeformAttributionWorkspace::source_delta_z(const MeshDeformSource source) {
  return d_delta_z_ + static_cast<std::size_t>(source_index(source)) * n_nodes_;
}

const double* MeshDeformAttributionWorkspace::source_delta_r(
    const MeshDeformSource source) const {
  return d_delta_r_ + static_cast<std::size_t>(source_index(source)) * n_nodes_;
}

const double* MeshDeformAttributionWorkspace::source_delta_z(
    const MeshDeformSource source) const {
  return d_delta_z_ + static_cast<std::size_t>(source_index(source)) * n_nodes_;
}

void MeshDeformAttributionWorkspace::clear_lagrangian_sources() {
  if (!active_) {
    return;
  }
  const std::size_t bytes = n_nodes_ * sizeof(double);
  const MeshDeformSource sources[3] = {MeshDeformSource::KinematicCarry,
                                      MeshDeformSource::HydroPressure,
                                      MeshDeformSource::ArtificialViscosity};
  for (const MeshDeformSource source : sources) {
    cuda_check(cudaMemset(source_delta_r(source), 0, bytes),
               "mesh attribution clear source delta_r failed");
    cuda_check(cudaMemset(source_delta_z(source), 0, bytes),
               "mesh attribution clear source delta_z failed");
  }
}

void MeshDeformAttributionWorkspace::record_displacement_delta(
    const MeshDeformSource source,
    const double* d_delta_r,
    const double* d_delta_z,
    const int n_nodes) {
  if (!active_) {
    return;
  }
  TENRYU_ASSERT(n_nodes == static_cast<int>(n_nodes_),
                "mesh attribution displacement size mismatch");
  const int blocks = (n_nodes + 255) / 256;
  add_delta_kernel<<<blocks, 256>>>(source_delta_r(source),
                                    source_delta_z(source),
                                    d_delta_r,
                                    d_delta_z,
                                    n_nodes);
  cuda_check(cudaGetLastError(), "mesh attribution add delta kernel failed");
}

void MeshDeformAttributionWorkspace::record_kinematic_carry(
    const double* d_v_r,
    const double* d_v_z,
    const std::uint8_t* d_node_active,
    const int n_nodes,
    const double dt_scale) {
  record_acceleration_displacement(MeshDeformSource::KinematicCarry,
                                   d_v_r,
                                   d_v_z,
                                   d_node_active,
                                   n_nodes,
                                   dt_scale);
}

void MeshDeformAttributionWorkspace::record_acceleration_displacement(
    const MeshDeformSource source,
    const double* d_a_r,
    const double* d_a_z,
    const std::uint8_t* d_node_active,
    const int n_nodes,
    const double coefficient) {
  if (!active_) {
    return;
  }
  TENRYU_ASSERT(n_nodes == static_cast<int>(n_nodes_),
                "mesh attribution acceleration size mismatch");
  const int blocks = (n_nodes + 255) / 256;
  add_scaled_node_vector_kernel<<<blocks, 256>>>(source_delta_r(source),
                                                 source_delta_z(source),
                                                 d_a_r,
                                                 d_a_z,
                                                 d_node_active,
                                                 n_nodes,
                                                 coefficient);
  cuda_check(cudaGetLastError(), "mesh attribution add scaled vector kernel failed");
}

void MeshDeformAttributionWorkspace::record_direct_displacement(
    const MeshDeformSource source,
    const double* d_r_before,
    const double* d_z_before,
    const double* d_r_after,
    const double* d_z_after,
    const int n_nodes) {
  if (!active_) {
    return;
  }
  TENRYU_ASSERT(n_nodes == static_cast<int>(n_nodes_),
                "mesh attribution direct displacement size mismatch");
  const int blocks = (n_nodes + 255) / 256;
  add_direct_displacement_kernel<<<blocks, 256>>>(source_delta_r(source),
                                                  source_delta_z(source),
                                                  d_r_before,
                                                  d_z_before,
                                                  d_r_after,
                                                  d_z_after,
                                                  n_nodes);
  cuda_check(cudaGetLastError(), "mesh attribution direct delta kernel failed");
}

MeshFailureAttributionRecord MeshDeformAttributionWorkspace::compute_failure_record(
    const core::State& state,
    const core::Config& cfg,
    const int failing_cell,
    const int failing_corner,
    const double* d_candidate_r,
    const double* d_candidate_z) {
  MeshFailureAttributionRecord record;
  record.step = step_;
  record.t_simulation = t_simulation_;
  record.failing_cell = failing_cell;
  record.failing_corner = failing_corner;
  if (!active_) {
    return record;
  }
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_nodes = static_cast<int>(n_nodes_);
  cuda_check(cudaMemset(d_dj_source_, 0, kMeshDeformSourceCount * sizeof(double)),
             "mesh attribution clear failure dj failed");
  const double zero = 0.0;
  cuda_check(cudaMemcpy(d_actual_delta_j_, &zero, sizeof(double), cudaMemcpyHostToDevice),
             "mesh attribution clear actual dj failed");
  compute_failure_dj_kernel<<<1, 1>>>(d_dj_source_,
                                      d_actual_delta_j_,
                                      d_start_r_,
                                      d_start_z_,
                                      d_candidate_r != nullptr ? d_candidate_r
                                                               : state.x_r.data(),
                                      d_candidate_z != nullptr ? d_candidate_z
                                                               : state.x_z.data(),
                                      d_delta_r_,
                                      d_delta_z_,
                                      nr,
                                      nz,
                                      failing_cell,
                                      failing_corner,
                                      n_nodes);
  cuda_check(cudaGetLastError(), "mesh attribution failure kernel failed");
  cuda_check(cudaDeviceSynchronize(), "mesh attribution failure kernel sync failed");
  cuda_check(cudaMemcpy(record.dJ_source,
                        d_dj_source_,
                        kMeshDeformSourceCount * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "mesh attribution copy failure dj failed");
  double actual_delta_j = 0.0;
  cuda_check(cudaMemcpy(&actual_delta_j,
                        d_actual_delta_j_,
                        sizeof(double),
                        cudaMemcpyDeviceToHost),
             "mesh attribution copy actual dj failed");
  finalize_record(record, false, actual_delta_j);
  (void)cfg;
  return record;
}

MeshFailureAttributionRecord MeshDeformAttributionWorkspace::compute_aggregate_record(
    const core::State& state,
    const core::Config& cfg) {
  MeshFailureAttributionRecord record;
  record.step = step_;
  record.t_simulation = t_simulation_;
  record.failing_cell = -1;
  record.failing_corner = -1;
  if (!active_) {
    return record;
  }
  cuda_check(cudaMemset(d_dj_source_, 0, kMeshDeformSourceCount * sizeof(double)),
             "mesh attribution clear aggregate dj failed");
  const int n_cells = static_cast<int>(n_cells_);
  if (n_cells <= 0) {
    finalize_record(record, true, 0.0);
    (void)cfg;
    return record;
  }
  const int blocks = (n_cells + 255) / 256;
  compute_aggregate_dj_kernel<<<blocks, 256>>>(d_dj_source_,
                                               d_start_r_,
                                               d_start_z_,
                                               d_delta_r_,
                                               d_delta_z_,
                                               d_hydro_active_,
                                               state.mesh.topo.nr,
                                               state.mesh.topo.nz,
                                               static_cast<int>(n_nodes_));
  cuda_check(cudaGetLastError(), "mesh attribution aggregate kernel failed");
  cuda_check(cudaDeviceSynchronize(), "mesh attribution aggregate kernel sync failed");
  cuda_check(cudaMemcpy(record.dJ_source,
                        d_dj_source_,
                        kMeshDeformSourceCount * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "mesh attribution copy aggregate dj failed");
  finalize_record(record, true, 0.0);
  (void)cfg;
  return record;
}

void MeshDeformAttributionWorkspace::emit_failure_record(
    const core::State& state,
    const core::Config& cfg,
    const int failing_cell,
    const int failing_corner,
    const double* d_candidate_r,
    const double* d_candidate_z) {
  if (!active_) {
    return;
  }
  MeshFailureAttributionRecord record =
      compute_failure_record(state, cfg, failing_cell, failing_corner,
                             d_candidate_r, d_candidate_z);
  record.mesh_failure_id = next_failure_id();
  write_record_jsonl(record);
  release();
}

void MeshDeformAttributionWorkspace::end_step_success(const core::State& state,
                                                      const core::Config& cfg) {
  if (!active_) {
    return;
  }
  if (!dump_on_failure_only_) {
    MeshFailureAttributionRecord record = compute_aggregate_record(state, cfg);
    record.mesh_failure_id = next_failure_id();
    write_record_jsonl(record);
  }
  release();
}

LeaveOneOutReplayResult MeshDeformAttributionWorkspace::leave_one_out_replay_stub(
    const MeshDeformSource source) const {
  (void)source;
  return LeaveOneOutReplayResult{false, "not_run"};
}

MeshDeformAttributionWorkspace& global_workspace() {
  static auto* workspace = new MeshDeformAttributionWorkspace();
  return *workspace;
}

bool begin_step(const core::State& state,
                const core::Config& cfg,
                const double t_simulation,
                const char* hydro_half) {
  if (!cfg.numerics.diagnostics.mesh_attribution.enabled ||
      !cfg.numerics.diagnostics.mesh_attribution.record_node_displacements) {
    return false;
  }
  return global_workspace().begin_step(state, cfg, t_simulation, hydro_half);
}

void end_step_success(const core::State& state, const core::Config& cfg) {
  global_workspace().end_step_success(state, cfg);
}

void discard() {
  global_workspace().discard();
}

void emit_failure_record(const core::State& state,
                         const core::Config& cfg,
                         const int failing_cell,
                         const int failing_corner,
                         const double* d_candidate_r,
                         const double* d_candidate_z) {
  global_workspace().emit_failure_record(
      state, cfg, failing_cell, failing_corner, d_candidate_r, d_candidate_z);
}

void emit_axis_projection_record(const core::State& state,
                                 const core::Config& cfg,
                                 const AxisProjectionRecord& record) {
  if (!cfg.numerics.diagnostics.mesh_attribution.enabled) {
    return;
  }
  (void)state;
  write_axis_projection_record_jsonl(record);
}

void record_zero_source(const MeshDeformSource source) {
  (void)source;
}

}  // namespace tenryu::diagnostics::mesh_attribution
