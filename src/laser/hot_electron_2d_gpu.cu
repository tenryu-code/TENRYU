#include "laser/hot_electron_2d_gpu.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/constants.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "hydro/ale_remap.cuh"
#include "mesh/mesh.hpp"

namespace tenryu::laser::hot_electron {
namespace {

constexpr int kHotE2dBlock = 128;
constexpr int kMaxRecordedSegments = 1024;

inline void hot_e2d_cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

int default_max_segments_2d_device(const int n_cells) {
  const int n_side =
      static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n_cells))));
  return 8 * std::max(64, n_side);
}

__global__ __launch_bounds__(kHotE2dBlock, 8) void hot_e_cone_2d_chord_kernel(
    const double* __restrict__ chords,
    const int* __restrict__ job_start_cell,
    const int n_jobs,
    const double* __restrict__ groups,
    const int n_groups,
    const double* __restrict__ rho,
    const double* __restrict__ zbar,
    const double* __restrict__ A_eff,
    const double* __restrict__ Te_eV,
    const std::uint8_t* __restrict__ cell_is_void,
    const DeviceMeshView2D dview,
    const double mesh_scale,
    const int seg_cap,
    const double T_h_erg,
    const double proton_mass,
    const double ev_to_erg,
    int* __restrict__ seg_cell,
    double* __restrict__ seg_ds,
    double* __restrict__ seg_dP,
    int* __restrict__ seg_count,
    double* __restrict__ job_escaped,
    int* __restrict__ job_stuck,
    int* __restrict__ job_relocs,
    int* __restrict__ job_cap_hits) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n_jobs) {
    return;
  }

  const std::size_t base = static_cast<std::size_t>(j) * static_cast<std::size_t>(seg_cap);
  const double* const job = chords + static_cast<std::size_t>(j) * 6U;
  double om[3] = {job[2], job[3], job[4]};
  const Chord3D chord(job[0], job[1], om);
  const double P_chord = job[5];
  const MeshView2D view = dview.as_mesh_view();
  const int start_cell = job_start_cell[j];

  int count = 0;
  int terminal_cell = start_cell;
  bool exited = false;
  bool stuck = false;
  int cap_hits = 0;
  ChordWalkerRZ walker(view, chord, start_cell, mesh_scale, seg_cap + 1);
  for (;;) {
    double ds = 0.0;
    int cell = -1;
    const WalkStatus status = walker.next(ds, cell);
    if (status == WalkStatus::Segment) {
      if (cell >= 0 && cell < view.n_cells && ds > 0.0) {
        if (count >= seg_cap) {
          stuck = true;
          ++cap_hits;
          terminal_cell =
              (count > 0) ? seg_cell[base + static_cast<std::size_t>(count - 1)] : cell;
          break;
        }
        seg_cell[base + static_cast<std::size_t>(count)] = cell;
        seg_ds[base + static_cast<std::size_t>(count)] = ds;
        terminal_cell = cell;
        ++count;
      }
      if (walker.exited_boundary) {
        exited = true;
        break;
      }
      continue;
    }
    if (status == WalkStatus::Stuck) {
      stuck = true;
      terminal_cell =
          (count > 0) ? seg_cell[base + static_cast<std::size_t>(count - 1)] : cell;
    } else {
      exited = true;
    }
    break;
  }

  if (stuck && count == 0 && terminal_cell >= 0 && terminal_cell < view.n_cells &&
      seg_cap > 0) {
    seg_cell[base] = terminal_cell;
    seg_ds[base] = 0.0;
    count = 1;
  }
  for (int iseg = 0; iseg < count; ++iseg) {
    seg_dP[base + static_cast<std::size_t>(iseg)] = 0.0;
  }
  double escaped = 0.0;

  for (int g = 0; g < n_groups; ++g) {
    const double E_rep = groups[static_cast<std::size_t>(g) * 2U + 0U];
    const double weight = groups[static_cast<std::size_t>(g) * 2U + 1U];
    if (!(weight > 0.0) || !(E_rep > 0.0)) {
      continue;
    }
    const double Ndot = P_chord * weight / E_rep;
    double E = E_rep;
    for (int iseg = 0; iseg < count; ++iseg) {
      const int cell = seg_cell[base + static_cast<std::size_t>(iseg)];
      const std::size_t c = static_cast<std::size_t>(cell);
      if (cell_is_void[c]) {
        continue;
      }
      const double rho_c = rho[c];
      const double dSigma = rho_c * seg_ds[base + static_cast<std::size_t>(iseg)];
      const double Te_erg = ((Te_eV[c] > 0.0) ? Te_eV[c] : 0.0) * ev_to_erg;
      double ne = 0.0;
      if (A_eff[c] > 0.0 && rho_c > 0.0) {
        ne = rho_c * ((zbar[c] > 0.0) ? zbar[c] : 0.0) / (A_eff[c] * proton_mass);
      }
      const double E_floor =
          ((2.0 * Te_erg) > (1.0e-3 * T_h_erg)) ? (2.0 * Te_erg) : (1.0e-3 * T_h_erg);
      const auto stopping = [=](const double Ee) -> double {
        return stopping_power_erg_cm2_per_g(Ee, ne, Te_erg, rho_c);
      };
      const double E_out =
          march_cell(E, dSigma, stopping, E_floor, kSubstepEnergyFraction,
                     kMaxSubstepsPerCell, &cap_hits);
      seg_dP[base + static_cast<std::size_t>(iseg)] += Ndot * (E - E_out);
      E = E_out;
      if (!(E > 0.0)) {
        break;
      }
    }
    if (E > 0.0) {
      const double residual = Ndot * E;
      if (stuck && count > 0) {
        seg_dP[base + static_cast<std::size_t>(count - 1)] += residual;
      } else if (exited) {
        escaped += residual;
      }
    }
  }

  seg_count[j] = count;
  job_escaped[j] = escaped;
  job_stuck[j] = stuck ? 1 : 0;
  job_relocs[j] = walker.relocate_attempts;
  job_cap_hits[j] = cap_hits;
}

}  // namespace

DeviceMeshView2D upload_mesh_view_2d(
    const MeshView2DStorage& storage,
    const std::vector<double>& node_R_host,
    const std::vector<double>& node_Z_host,
    const int n_cells,
    cudaStream_t stream,
    DeviceMeshView2DScratch& scratch) {
  TENRYU_ASSERT(n_cells > 0, "hot_electron 2d device mesh upload requires cells");
  TENRYU_ASSERT(storage.cell_nodes.size() == static_cast<std::size_t>(n_cells) * 4U,
                "hot_electron 2d device mesh upload cell_nodes size mismatch");
  TENRYU_ASSERT(storage.cell_neighbor.size() == static_cast<std::size_t>(n_cells) * 4U,
                "hot_electron 2d device mesh upload cell_neighbor size mismatch");
  TENRYU_ASSERT(node_R_host.size() == node_Z_host.size(),
                "hot_electron 2d device mesh upload node size mismatch");
  TENRYU_ASSERT(!node_R_host.empty(), "hot_electron 2d device mesh upload requires nodes");

  const std::size_t topo_bytes = static_cast<std::size_t>(n_cells) * 4U * sizeof(int);
  const std::size_t node_bytes = node_R_host.size() * sizeof(double);
  scratch.cell_nodes = static_cast<int*>(
      core::device_scratch_acquire("hot_electron_2d:mesh_cell_nodes", topo_bytes));
  scratch.cell_neighbor = static_cast<int*>(
      core::device_scratch_acquire("hot_electron_2d:mesh_cell_neighbor", topo_bytes));
  scratch.node_R = static_cast<double*>(
      core::device_scratch_acquire("hot_electron_2d:mesh_node_R", node_bytes));
  scratch.node_Z = static_cast<double*>(
      core::device_scratch_acquire("hot_electron_2d:mesh_node_Z", node_bytes));

  hot_e2d_cuda_check(cudaMemcpyAsync(scratch.cell_nodes, storage.cell_nodes.data(),
                                     topo_bytes, cudaMemcpyHostToDevice, stream),
                     "hot_electron 2d device mesh cell_nodes H2D failed");
  hot_e2d_cuda_check(cudaMemcpyAsync(scratch.cell_neighbor, storage.cell_neighbor.data(),
                                     topo_bytes, cudaMemcpyHostToDevice, stream),
                     "hot_electron 2d device mesh cell_neighbor H2D failed");
  hot_e2d_cuda_check(cudaMemcpyAsync(scratch.node_R, node_R_host.data(), node_bytes,
                                     cudaMemcpyHostToDevice, stream),
                     "hot_electron 2d device mesh node_R H2D failed");
  hot_e2d_cuda_check(cudaMemcpyAsync(scratch.node_Z, node_Z_host.data(), node_bytes,
                                     cudaMemcpyHostToDevice, stream),
                     "hot_electron 2d device mesh node_Z H2D failed");

  DeviceMeshView2D out;
  out.n_cells = n_cells;
  out.cell_nodes = scratch.cell_nodes;
  out.cell_neighbor = scratch.cell_neighbor;
  out.node_R = scratch.node_R;
  out.node_Z = scratch.node_Z;
  return out;
}

DepositResult2D deposit_hot_electrons_cone_2d_device(
    const HotEChannelSpec& spec,
    const std::vector<HotESource2D>& sources,
    const double* d_rho,
    const double* d_zbar,
    const double* d_A_eff,
    const double* d_Te_eV,
    const std::vector<std::uint8_t>& cell_is_void_host,
    const DeviceMeshView2D& dview,
    const double mesh_scale,
    int max_segments,
    cudaStream_t stream,
    std::vector<double>& dep_power_cell) {
  DepositResult2D result;
  TENRYU_ASSERT(dview.n_cells > 0, "hot_electron 2d cone device requires cells");
  TENRYU_ASSERT(dview.cell_nodes != nullptr, "hot_electron 2d cone device missing cell nodes");
  TENRYU_ASSERT(dview.cell_neighbor != nullptr, "hot_electron 2d cone device missing neighbors");
  TENRYU_ASSERT(dview.node_R != nullptr, "hot_electron 2d cone device missing node R");
  TENRYU_ASSERT(dview.node_Z != nullptr, "hot_electron 2d cone device missing node Z");
  TENRYU_ASSERT(d_rho != nullptr, "hot_electron 2d cone device missing rho");
  TENRYU_ASSERT(d_zbar != nullptr, "hot_electron 2d cone device missing zbar");
  TENRYU_ASSERT(d_A_eff != nullptr, "hot_electron 2d cone device missing A_eff");
  TENRYU_ASSERT(d_Te_eV != nullptr, "hot_electron 2d cone device missing Te_eV");
  const std::size_t n = static_cast<std::size_t>(dview.n_cells);
  TENRYU_ASSERT(cell_is_void_host.size() == n,
                "hot_electron 2d cone device cell_is_void size mismatch");
  TENRYU_ASSERT(dep_power_cell.size() == n,
                "hot_electron 2d cone device dep_power_cell size mismatch");

  double PR = 0.0;
  for (const HotESource2D& source : sources) {
    result.P_hot += source.P_hot;
    PR += source.P_hot * source.R_s;
  }
  if (sources.empty() || !(result.P_hot > 0.0)) {
    return result;
  }
  result.n_sources = static_cast<int>(sources.size());
  result.r_source_mean = PR / result.P_hot;

  const double T_h_erg = spec.T_hot_erg;
  const std::vector<GroupSpec> groups =
      build_groups(T_h_erg, spec.n_energy_groups, spec.E_min_over_Th,
                   spec.E_max_over_Th);

  std::vector<double> chords;
  std::vector<int> job_start_cell;
  for (const HotESource2D& source : sources) {
    TENRYU_ASSERT(source.cell >= 0 && source.cell < dview.n_cells,
                  "hot_electron 2d device source cell out of range");
    const std::vector<DirNode> dirs =
        build_band_dirs_3d(source.k, spec.mu_lo, spec.mu_hi, spec.n_mu, spec.n_phi);
    for (const DirNode& dir : dirs) {
      const double P_chord = source.P_hot * dir.weight;
      if (!(P_chord > 0.0)) {
        continue;
      }
      chords.push_back(source.R_s);
      chords.push_back(source.Z_s);
      chords.push_back(dir.om[0]);
      chords.push_back(dir.om[1]);
      chords.push_back(dir.om[2]);
      chords.push_back(P_chord);
      job_start_cell.push_back(source.cell);
    }
  }

  if (chords.empty() || groups.empty()) {
    result.conservation_resid =
        std::abs(result.P_deposited + result.P_escaped - result.P_hot) /
        std::max(result.P_hot, 1.0e-300);
    if (result.conservation_resid > 1.0e-8) {
      core::log_warning("hot_electron 2d cone conservation check failed");
    }
    result.active = true;
    return result;
  }

  if (max_segments <= 0) {
    max_segments = default_max_segments_2d_device(dview.n_cells);
  }
  const int seg_cap = std::min(max_segments, kMaxRecordedSegments);
  TENRYU_ASSERT(seg_cap > 0, "hot_electron 2d cone device segment cap must be positive");

  std::vector<double> group_flat;
  group_flat.reserve(groups.size() * 2U);
  for (const GroupSpec& group : groups) {
    group_flat.push_back(group.E_rep);
    group_flat.push_back(group.weight);
  }

  const int n_jobs = static_cast<int>(chords.size() / 6U);
  const int n_groups = static_cast<int>(groups.size());
  const std::size_t seg_slots =
      static_cast<std::size_t>(n_jobs) * static_cast<std::size_t>(seg_cap);
  const std::size_t chords_bytes = chords.size() * sizeof(double);
  const std::size_t job_start_cell_bytes = job_start_cell.size() * sizeof(int);
  const std::size_t groups_bytes = group_flat.size() * sizeof(double);
  const std::size_t void_bytes = n * sizeof(std::uint8_t);
  const std::size_t seg_cell_bytes = seg_slots * sizeof(int);
  const std::size_t seg_double_bytes = seg_slots * sizeof(double);
  const std::size_t job_int_bytes = static_cast<std::size_t>(n_jobs) * sizeof(int);
  const std::size_t job_double_bytes = static_cast<std::size_t>(n_jobs) * sizeof(double);

  double* const d_chords = static_cast<double*>(
      core::device_scratch_acquire("hot_electron_2d:chords", chords_bytes));
  int* const d_job_start_cell = static_cast<int*>(
      core::device_scratch_acquire("hot_electron_2d:job_start_cell",
                                   job_start_cell_bytes));
  double* const d_groups = static_cast<double*>(
      core::device_scratch_acquire("hot_electron_2d:groups", groups_bytes));
  std::uint8_t* const d_void = static_cast<std::uint8_t*>(
      core::device_scratch_acquire("hot_electron_2d:void", void_bytes));
  int* const d_seg_cell = static_cast<int*>(
      core::device_scratch_acquire("hot_electron_2d:seg_cell", seg_cell_bytes));
  double* const d_seg_ds = static_cast<double*>(
      core::device_scratch_acquire("hot_electron_2d:seg_ds", seg_double_bytes));
  double* const d_seg_dP = static_cast<double*>(
      core::device_scratch_acquire("hot_electron_2d:seg_dP", seg_double_bytes));
  int* const d_seg_count = static_cast<int*>(
      core::device_scratch_acquire("hot_electron_2d:seg_count", job_int_bytes));
  double* const d_job_escaped = static_cast<double*>(
      core::device_scratch_acquire("hot_electron_2d:job_escaped", job_double_bytes));
  int* const d_job_stuck = static_cast<int*>(
      core::device_scratch_acquire("hot_electron_2d:job_stuck", job_int_bytes));
  int* const d_job_relocs = static_cast<int*>(
      core::device_scratch_acquire("hot_electron_2d:job_relocs", job_int_bytes));
  int* const d_job_cap_hits = static_cast<int*>(
      core::device_scratch_acquire("hot_electron_2d:job_cap_hits", job_int_bytes));

  hot_e2d_cuda_check(cudaMemcpyAsync(d_chords, chords.data(), chords_bytes,
                                     cudaMemcpyHostToDevice, stream),
                     "hot_electron 2d cone device chords H2D failed");
  hot_e2d_cuda_check(cudaMemcpyAsync(d_job_start_cell, job_start_cell.data(),
                                     job_start_cell_bytes, cudaMemcpyHostToDevice,
                                     stream),
                     "hot_electron 2d cone device start cells H2D failed");
  hot_e2d_cuda_check(cudaMemcpyAsync(d_groups, group_flat.data(), groups_bytes,
                                     cudaMemcpyHostToDevice, stream),
                     "hot_electron 2d cone device groups H2D failed");
  hot_e2d_cuda_check(cudaMemcpyAsync(d_void, cell_is_void_host.data(), void_bytes,
                                     cudaMemcpyHostToDevice, stream),
                     "hot_electron 2d cone device void H2D failed");

  const int grid = (n_jobs + kHotE2dBlock - 1) / kHotE2dBlock;
  hot_e_cone_2d_chord_kernel<<<grid, kHotE2dBlock, 0, stream>>>(
      d_chords, d_job_start_cell, n_jobs, d_groups, n_groups, d_rho, d_zbar,
      d_A_eff, d_Te_eV, d_void, dview, mesh_scale, seg_cap, T_h_erg,
      core::constants::proton_mass, core::constants::eV_to_erg, d_seg_cell,
      d_seg_ds, d_seg_dP, d_seg_count, d_job_escaped, d_job_stuck,
      d_job_relocs, d_job_cap_hits);
  hot_e2d_cuda_check(cudaGetLastError(),
                     "hot_electron 2d cone chord kernel launch failed");

  std::vector<int> seg_cell(seg_slots, -1);
  std::vector<double> seg_dP(seg_slots, 0.0);
  std::vector<int> seg_count(static_cast<std::size_t>(n_jobs), 0);
  std::vector<double> job_escaped(static_cast<std::size_t>(n_jobs), 0.0);
  std::vector<int> job_stuck(static_cast<std::size_t>(n_jobs), 0);
  std::vector<int> job_relocs(static_cast<std::size_t>(n_jobs), 0);
  std::vector<int> job_cap_hits(static_cast<std::size_t>(n_jobs), 0);
  hot_e2d_cuda_check(cudaMemcpyAsync(seg_cell.data(), d_seg_cell, seg_cell_bytes,
                                     cudaMemcpyDeviceToHost, stream),
                     "hot_electron 2d cone device seg_cell D2H failed");
  hot_e2d_cuda_check(cudaMemcpyAsync(seg_dP.data(), d_seg_dP, seg_double_bytes,
                                     cudaMemcpyDeviceToHost, stream),
                     "hot_electron 2d cone device seg_dP D2H failed");
  hot_e2d_cuda_check(cudaMemcpyAsync(seg_count.data(), d_seg_count, job_int_bytes,
                                     cudaMemcpyDeviceToHost, stream),
                     "hot_electron 2d cone device seg_count D2H failed");
  hot_e2d_cuda_check(cudaMemcpyAsync(job_escaped.data(), d_job_escaped,
                                     job_double_bytes, cudaMemcpyDeviceToHost, stream),
                     "hot_electron 2d cone device escaped D2H failed");
  hot_e2d_cuda_check(cudaMemcpyAsync(job_stuck.data(), d_job_stuck, job_int_bytes,
                                     cudaMemcpyDeviceToHost, stream),
                     "hot_electron 2d cone device stuck D2H failed");
  hot_e2d_cuda_check(cudaMemcpyAsync(job_relocs.data(), d_job_relocs, job_int_bytes,
                                     cudaMemcpyDeviceToHost, stream),
                     "hot_electron 2d cone device reloc D2H failed");
  hot_e2d_cuda_check(cudaMemcpyAsync(job_cap_hits.data(), d_job_cap_hits,
                                     job_int_bytes, cudaMemcpyDeviceToHost, stream),
                     "hot_electron 2d cone device cap D2H failed");
  hot_e2d_cuda_check(cudaStreamSynchronize(stream),
                     "hot_electron 2d cone device stream synchronize failed");

  for (int j = 0; j < n_jobs; ++j) {
    const std::size_t base =
        static_cast<std::size_t>(j) * static_cast<std::size_t>(seg_cap);
    for (int iseg = 0; iseg < seg_count[static_cast<std::size_t>(j)]; ++iseg) {
      const int cell = seg_cell[base + static_cast<std::size_t>(iseg)];
      TENRYU_ASSERT(cell >= 0 && cell < dview.n_cells,
                    "hot_electron 2d cone device folded segment cell out of range");
      const double deposited = seg_dP[base + static_cast<std::size_t>(iseg)];
      dep_power_cell[static_cast<std::size_t>(cell)] += deposited;
      result.P_deposited += deposited;
    }
    result.P_escaped += job_escaped[static_cast<std::size_t>(j)];
    result.walker_stuck_terminations += job_stuck[static_cast<std::size_t>(j)];
    result.walker_relocations += job_relocs[static_cast<std::size_t>(j)];
    result.substep_cap_hits += job_cap_hits[static_cast<std::size_t>(j)];
  }

  result.conservation_resid =
      std::abs(result.P_deposited + result.P_escaped - result.P_hot) /
      std::max(result.P_hot, 1.0e-300);
  if (result.conservation_resid > 1.0e-8) {
    core::log_warning("hot_electron 2d cone conservation check failed");
  }
  result.active = true;
  return result;
}

MeshView2DStorage build_multiblock_view(
    const tenryu::mesh::MeshTopology& topo, const double* node_R,
    const double* node_Z, const std::uint8_t* cell_nverts) {
  TENRYU_ASSERT(topo.multiblock.has_value(),
                "hot_electron build_multiblock_view requires multiblock topology");
  const auto& mb = *topo.multiblock;
  const int n_cells = topo.n_cells;
  TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U,
                "hot_electron multiblock view: cell_node_csr_offsets size");
  TENRYU_ASSERT(mb.cell_node_csr_indices.size() ==
                    static_cast<std::size_t>(n_cells) * 4U,
                "hot_electron multiblock view: cell_node_csr_indices size");
  for (int c = 0; c <= n_cells; ++c) {
    TENRYU_ASSERT(mb.cell_node_csr_offsets[static_cast<std::size_t>(c)] ==
                      4 * c,
                  "hot_electron multiblock view: non-uniform node CSR stride");
  }
  // Active vertex counts come from the mesh cell_nverts channel; the
  // authoritative accessor tolerates a null pointer (=> 4 verts).

  MeshView2DStorage storage;
  storage.cell_nodes.assign(static_cast<std::size_t>(n_cells) * 4U, -1);
  storage.cell_neighbor.assign(static_cast<std::size_t>(n_cells) * 4U, -1);

  // Pass 1: node slots, ccw-normalized on the ACTIVE corners.
  for (int c = 0; c < n_cells; ++c) {
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int nv = tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
    int nodes[4];
    for (int k = 0; k < nv; ++k) {
      nodes[k] = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
    }
    double area2 = 0.0;
    for (int k = 0; k < nv; ++k) {
      const int a = nodes[k];
      const int b = nodes[(k + 1) % nv];
      area2 += node_R[a] * node_Z[b] - node_R[b] * node_Z[a];
    }
    if (area2 < 0.0) {
      int reversed[4];
      reversed[0] = nodes[0];
      for (int k = 1; k < nv; ++k) {
        reversed[k] = nodes[nv - k];
      }
      for (int k = 0; k < nv; ++k) {
        nodes[k] = reversed[k];
      }
    }
    for (int k = 0; k < nv; ++k) {
      storage.cell_nodes[static_cast<std::size_t>(c) * 4U +
                         static_cast<std::size_t>(k)] = nodes[k];
    }
    if (nv == 3) {
      storage.cell_nodes[static_cast<std::size_t>(c) * 4U + 3U] = nodes[0];
    }
  }

  // Pass 2: neighbors from the authoritative face tables. For each recorded
  // (cell, local_face) the canonical node pair comes from the mesh helper;
  // the matching view face is found by unordered node-id pair within that
  // cell (id spaces are consistent per cell even across seams).
  const auto find_view_face = [&storage](const int cell, const int na,
                                         const int nb) {
    for (int f = 0; f < 4; ++f) {
      const int va =
          storage.cell_nodes[static_cast<std::size_t>(cell) * 4U +
                             static_cast<std::size_t>(f)];
      const int vb =
          storage.cell_nodes[static_cast<std::size_t>(cell) * 4U +
                             static_cast<std::size_t>((f + 1) & 3)];
      if (va == vb) {
        continue;
      }
      if ((va == na && vb == nb) || (va == nb && vb == na)) {
        return f;
      }
    }
    return -1;
  };
  const auto face_nodes = [&mb, cell_nverts](const int cell,
                                            const int local_face,
                                            int* na, int* nb) {
    return tenryu::hydro::ale::detail::csr_face_swept_node_indices(
        mb.cell_node_csr_offsets.data(),
        mb.cell_node_csr_indices.data(),
        cell_nverts,
        cell,
        local_face,
        na,
        nb);
  };

  for (const auto& face : mb.unique_internal_faces) {
    int na = -1;
    int nb = -1;
    TENRYU_ASSERT(face_nodes(face.cell_a, face.local_a, &na, &nb) && na != nb,
                  "hot_electron multiblock view: internal face A node pair");
    const int fa = find_view_face(face.cell_a, na, nb);
    TENRYU_ASSERT(fa >= 0,
                  "hot_electron multiblock view: internal face A unmatched");
    TENRYU_ASSERT(storage.cell_neighbor[static_cast<std::size_t>(face.cell_a) *
                                            4U +
                                        static_cast<std::size_t>(fa)] == -1,
                  "hot_electron multiblock view: internal face A double link");
    storage.cell_neighbor[static_cast<std::size_t>(face.cell_a) * 4U +
                          static_cast<std::size_t>(fa)] = face.cell_b;

    TENRYU_ASSERT(face_nodes(face.cell_b, face.local_b, &na, &nb) && na != nb,
                  "hot_electron multiblock view: internal face B node pair");
    const int fb = find_view_face(face.cell_b, na, nb);
    TENRYU_ASSERT(fb >= 0,
                  "hot_electron multiblock view: internal face B unmatched");
    TENRYU_ASSERT(storage.cell_neighbor[static_cast<std::size_t>(face.cell_b) *
                                            4U +
                                        static_cast<std::size_t>(fb)] == -1,
                  "hot_electron multiblock view: internal face B double link");
    storage.cell_neighbor[static_cast<std::size_t>(face.cell_b) * 4U +
                          static_cast<std::size_t>(fb)] = face.cell_a;
  }

  // Pass 3: completeness — every non-degenerate view face must be linked or
  // tagged as a domain boundary; anything else would silently leak chords.
  std::vector<std::uint8_t> boundary_seen(
      static_cast<std::size_t>(n_cells) * 4U, 0U);
  for (const auto& face : mb.boundary_faces) {
    int na = -1;
    int nb = -1;
    if (!face_nodes(face.cell_a, face.local_a, &na, &nb) || na == nb) {
      continue;
    }
    const int f = find_view_face(face.cell_a, na, nb);
    TENRYU_ASSERT(f >= 0,
                  "hot_electron multiblock view: boundary face unmatched");
    boundary_seen[static_cast<std::size_t>(face.cell_a) * 4U +
                  static_cast<std::size_t>(f)] = 1U;
  }
  for (int c = 0; c < n_cells; ++c) {
    for (int f = 0; f < 4; ++f) {
      const std::size_t s =
          static_cast<std::size_t>(c) * 4U + static_cast<std::size_t>(f);
      const int va = storage.cell_nodes[s];
      const int vb = storage.cell_nodes[static_cast<std::size_t>(c) * 4U +
                                        static_cast<std::size_t>((f + 1) & 3)];
      if (va == vb) {
        continue;
      }
      TENRYU_ASSERT(storage.cell_neighbor[s] != -1 || boundary_seen[s] != 0U,
                    "hot_electron multiblock view: face neither linked nor "
                    "boundary-tagged (chord leak)");
    }
  }
  return storage;
}

}  // namespace tenryu::laser::hot_electron
