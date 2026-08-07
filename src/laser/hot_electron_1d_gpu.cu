#include "laser/hot_electron_1d.cuh"
#include "core/device_scratch.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/constants.hpp"
#include "core/error.hpp"
#include "laser/hot_electron_1d_gpu.cuh"

namespace tenryu::laser::hot_electron {
namespace {

struct ChordJob {
  double r_s;      // launch radius / planar x
  double mu_dir;   // direction cosine to radial/slab axis
  double P_chord;  // erg/s carried by this chord (source P_hot * node weight)
};

inline void hot_e_cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__global__ void hot_e_cone_chord_kernel(
    const ChordJob* __restrict__ jobs, const int n_jobs,
    const GroupSpec* __restrict__ groups, const int n_groups,
    const double* __restrict__ rho, const double* __restrict__ zbar,
    const double* __restrict__ A_eff, const double* __restrict__ Te_eV,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ r_nodes, const int n_nodes,
    const int geom_planar,
    const double T_h_erg, const double proton_mass, const double ev_to_erg,
    double* __restrict__ rows,
    double* __restrict__ chord_escaped,
    int* __restrict__ chord_cap_hits) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_rows = n_jobs * n_groups;
  if (tid >= n_rows) {
    return;
  }

  const int j = tid / n_groups;
  const int g = tid % n_groups;
  const int n_cells = n_nodes - 1;
  const ChordJob job = jobs[j];
  double* const row = rows + static_cast<std::size_t>(tid) * n_cells;
  int cap_local = 0;

  const GroupSpec group = groups[g];
  if (group.weight > 0.0 && group.E_rep > 0.0) {
    const double Ndot = job.P_chord * group.weight / group.E_rep;
    double E = group.E_rep;
    if (geom_planar == 0) {
      ChordWalkerSph walker(job.r_s, job.r_s * job.mu_dir, r_nodes, n_nodes);
      double ds = 0.0;
      int cell = -1;
      while (walker.next(ds, cell)) {
        const int c = cell;
        if (cell_is_void[c]) {
          continue;
        }
        const double dSigma = rho[c] * ds;
        const double Te_erg = ((Te_eV[c] > 0.0) ? Te_eV[c] : 0.0) * ev_to_erg;
        double ne = 0.0;
        if (A_eff[c] > 0.0 && rho[c] > 0.0) {
          ne = rho[c] * ((zbar[c] > 0.0) ? zbar[c] : 0.0) / (A_eff[c] * proton_mass);
        }
        const double E_floor =
            ((2.0 * Te_erg) > (1.0e-3 * T_h_erg)) ? (2.0 * Te_erg) : (1.0e-3 * T_h_erg);
        const double rho_c = rho[c];
        const auto stopping = [=](const double Ee) -> double {
          return stopping_power_erg_cm2_per_g(Ee, ne, Te_erg, rho_c);
        };
        const double E_out = march_cell(E, dSigma, stopping, E_floor,
                                        kSubstepEnergyFraction,
                                        kMaxSubstepsPerCell,
                                        &cap_local);
        row[c] += Ndot * (E - E_out);
        E = E_out;
        if (!(E > 0.0)) {
          break;
        }
      }
    } else {
      ChordWalkerPln walker(job.r_s, job.mu_dir, r_nodes, n_nodes);
      double ds = 0.0;
      int cell = -1;
      while (walker.next(ds, cell)) {
        const int c = cell;
        if (cell_is_void[c]) {
          continue;
        }
        const double dSigma = rho[c] * ds;
        const double Te_erg = ((Te_eV[c] > 0.0) ? Te_eV[c] : 0.0) * ev_to_erg;
        double ne = 0.0;
        if (A_eff[c] > 0.0 && rho[c] > 0.0) {
          ne = rho[c] * ((zbar[c] > 0.0) ? zbar[c] : 0.0) / (A_eff[c] * proton_mass);
        }
        const double E_floor =
            ((2.0 * Te_erg) > (1.0e-3 * T_h_erg)) ? (2.0 * Te_erg) : (1.0e-3 * T_h_erg);
        const double rho_c = rho[c];
        const auto stopping = [=](const double Ee) -> double {
          return stopping_power_erg_cm2_per_g(Ee, ne, Te_erg, rho_c);
        };
        const double E_out = march_cell(E, dSigma, stopping, E_floor,
                                        kSubstepEnergyFraction,
                                        kMaxSubstepsPerCell,
                                        &cap_local);
        row[c] += Ndot * (E - E_out);
        E = E_out;
        if (!(E > 0.0)) {
          break;
        }
      }
    }
    if (E > 0.0) {
      chord_escaped[tid] += Ndot * E;
    }
  }
  chord_cap_hits[tid] = cap_local;
}

__global__ void hot_e_reduce_rows_kernel(
    const double* __restrict__ rows, const int n_rows, const int n_cells,
    double* __restrict__ out_cell) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  double acc = 0.0;
  for (int row = 0; row < n_rows; ++row) {
    acc += rows[static_cast<std::size_t>(row) * n_cells + c];
  }
  out_cell[c] = acc;
}

}  // namespace

DepositResult deposit_hot_electrons_cone_1d_device(
    const HotEChannelSpec& spec,
    const Geometry1D geom,
    const std::vector<HotESource>& sources,
    const double* d_rho, const double* d_zbar, const double* d_A_eff,
    const double* d_Te_eV, const std::vector<std::uint8_t>& cell_is_void_host,
    const double* d_r_nodes, const int n_nodes,
    cudaStream_t stream,
    std::vector<double>& dep_power_cell) {
  DepositResult result;
  TENRYU_ASSERT(n_nodes >= 2, "hot_electron cone device requires at least two nodes");
  const int n_cells = n_nodes - 1;
  TENRYU_ASSERT(cell_is_void_host.size() == static_cast<std::size_t>(n_cells),
                "hot_electron cone device cell_is_void size mismatch");
  TENRYU_ASSERT(dep_power_cell.size() == static_cast<std::size_t>(n_cells),
                "hot_electron cone device dep_power_cell size mismatch");

  double Pr = 0.0;
  for (const HotESource& source : sources) {
    result.P_hot += source.P_hot;
    Pr += source.P_hot * source.r_s;
  }
  if (sources.empty() || !(result.P_hot > 0.0)) {
    return result;
  }
  result.n_sources = static_cast<int>(sources.size());
  result.r_source_mean = Pr / result.P_hot;

  const double T_h_erg = spec.T_hot_erg;
  const std::vector<GroupSpec> groups =
      build_groups(T_h_erg, spec.n_energy_groups, spec.E_min_over_Th, spec.E_max_over_Th);

  std::vector<ChordJob> jobs;
  for (const HotESource& source : sources) {
    const std::vector<AngleNode> nodes =
        build_band_nodes(source.mu_axis, spec.mu_lo, spec.mu_hi, spec.n_mu, spec.n_phi);
    for (const AngleNode& node : nodes) {
      const double P_chord = source.P_hot * node.weight;
      if (!(P_chord > 0.0)) {
        continue;
      }
      if (geom == Geometry1D::planar &&
          std::abs(node.mu_dir) < kMuxEpsilon) {
        // In-plane chord: thermalizes in the source cell (mirrors the host
        // pipeline; the walker emits no segments for |mu| < kMuxEpsilon).
        const std::size_t sc = static_cast<std::size_t>(source.cell);
        if (source.cell >= 0 && source.cell < n_cells &&
            !cell_is_void_host[sc]) {
          dep_power_cell[sc] += P_chord;
          result.P_deposited += P_chord;
        } else {
          result.P_escaped += P_chord;
        }
        continue;
      }
      jobs.push_back(ChordJob{source.r_s, node.mu_dir, P_chord});
    }
  }

  if (jobs.empty() || groups.empty()) {
    result.conservation_resid =
        std::abs(result.P_deposited + result.P_escaped - result.P_hot) /
        std::max(result.P_hot, 1.0e-300);
    if (result.conservation_resid > 1.0e-8) {
      core::log_warning("hot_electron cone conservation check failed");
    }
    result.active = true;
    return result;
  }

  const int n_jobs = static_cast<int>(jobs.size());
  const int n_groups = static_cast<int>(groups.size());
  const int n_rows = n_jobs * n_groups;
  const std::size_t jobs_bytes = jobs.size() * sizeof(ChordJob);
  const std::size_t groups_bytes = groups.size() * sizeof(GroupSpec);
  const std::size_t void_bytes = static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t);
  const std::size_t rows_bytes =
      static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(n_cells) * sizeof(double);
  // The dense per-(job,group) row scratch is O(n_jobs * n_groups * n_cells)
  // and its theoretical worst case grows as O(n_cells^2) (AI review k10-5.6).
  // Physical capture bands are narrow so real sizes stay in the MB range;
  // fail loudly before an OOM-by-design allocation instead of crashing.
  TENRYU_ASSERT(rows_bytes <= (4ULL << 30),
                "hot_electron device row scratch exceeds 4 GiB "
                "(n_jobs*n_groups*n_cells); reduce hot_electron n_mu/n_phi/"
                "n_energy_groups or the capture spread");
  const std::size_t out_bytes = static_cast<std::size_t>(n_cells) * sizeof(double);
  const std::size_t escaped_bytes = static_cast<std::size_t>(n_rows) * sizeof(double);
  const std::size_t caps_bytes = static_cast<std::size_t>(n_rows) * sizeof(int);

  ChordJob* const d_jobs =
      static_cast<ChordJob*>(core::device_scratch_acquire("hot_electron:jobs", jobs_bytes));
  GroupSpec* const d_groups =
      static_cast<GroupSpec*>(core::device_scratch_acquire("hot_electron:groups", groups_bytes));
  std::uint8_t* const d_void = static_cast<std::uint8_t*>(
      core::device_scratch_acquire("hot_electron:void", void_bytes));
  double* const d_rows =
      static_cast<double*>(core::device_scratch_acquire("hot_electron:rows", rows_bytes));
  double* const d_out =
      static_cast<double*>(core::device_scratch_acquire("hot_electron:scalars", out_bytes));
  double* const d_escaped =
      static_cast<double*>(core::device_scratch_acquire("hot_electron:escaped", escaped_bytes));
  int* const d_caps =
      static_cast<int*>(core::device_scratch_acquire("hot_electron:caps", caps_bytes));

  hot_e_cuda_check(cudaMemcpyAsync(d_jobs, jobs.data(), jobs_bytes,
                                   cudaMemcpyHostToDevice, stream),
                   "hot_electron cone device jobs H2D failed");
  hot_e_cuda_check(cudaMemcpyAsync(d_groups, groups.data(), groups_bytes,
                                   cudaMemcpyHostToDevice, stream),
                   "hot_electron cone device groups H2D failed");
  hot_e_cuda_check(cudaMemcpyAsync(d_void, cell_is_void_host.data(), void_bytes,
                                   cudaMemcpyHostToDevice, stream),
                   "hot_electron cone device void H2D failed");
  hot_e_cuda_check(cudaMemsetAsync(d_rows, 0, rows_bytes, stream),
                   "hot_electron cone device rows memset failed");
  hot_e_cuda_check(cudaMemsetAsync(d_escaped, 0, escaped_bytes, stream),
                   "hot_electron cone device escaped memset failed");
  hot_e_cuda_check(cudaMemsetAsync(d_caps, 0, caps_bytes, stream),
                   "hot_electron cone device caps memset failed");

  constexpr int kBlock = 128;
  const int chord_grid = (n_rows + kBlock - 1) / kBlock;
  hot_e_cone_chord_kernel<<<chord_grid, kBlock, 0, stream>>>(
      d_jobs, n_jobs, d_groups, n_groups, d_rho, d_zbar, d_A_eff, d_Te_eV,
      d_void, d_r_nodes, n_nodes, (geom == Geometry1D::planar) ? 1 : 0,
      T_h_erg, core::constants::proton_mass, core::constants::eV_to_erg,
      d_rows, d_escaped, d_caps);
  hot_e_cuda_check(cudaGetLastError(), "hot_electron cone chord kernel launch failed");

  const int reduce_grid = (n_cells + kBlock - 1) / kBlock;
  hot_e_reduce_rows_kernel<<<reduce_grid, kBlock, 0, stream>>>(
      d_rows, n_rows, n_cells, d_out);
  hot_e_cuda_check(cudaGetLastError(), "hot_electron cone reduce kernel launch failed");

  std::vector<double> out_cell(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> escaped(static_cast<std::size_t>(n_rows), 0.0);
  std::vector<int> caps(static_cast<std::size_t>(n_rows), 0);
  hot_e_cuda_check(cudaMemcpyAsync(out_cell.data(), d_out, out_bytes,
                                   cudaMemcpyDeviceToHost, stream),
                   "hot_electron cone device out_cell D2H failed");
  hot_e_cuda_check(cudaMemcpyAsync(escaped.data(), d_escaped, escaped_bytes,
                                   cudaMemcpyDeviceToHost, stream),
                   "hot_electron cone device escaped D2H failed");
  hot_e_cuda_check(cudaMemcpyAsync(caps.data(), d_caps, caps_bytes,
                                   cudaMemcpyDeviceToHost, stream),
                   "hot_electron cone device caps D2H failed");
  hot_e_cuda_check(cudaStreamSynchronize(stream),
                   "hot_electron cone device stream synchronize failed");

  for (int c = 0; c < n_cells; ++c) {
    const double deposited = out_cell[static_cast<std::size_t>(c)];
    dep_power_cell[static_cast<std::size_t>(c)] += deposited;
    result.P_deposited += deposited;
  }
  for (int row = 0; row < n_rows; ++row) {
    result.P_escaped += escaped[static_cast<std::size_t>(row)];
    result.substep_cap_hits += caps[static_cast<std::size_t>(row)];
  }

  result.conservation_resid =
      std::abs(result.P_deposited + result.P_escaped - result.P_hot) /
      std::max(result.P_hot, 1.0e-300);
  if (result.conservation_resid > 1.0e-8) {
    core::log_warning("hot_electron cone conservation check failed");
  }
  result.active = true;
  return result;
}

DepositResult deposit_hot_electrons_cone_1d_device(
    const core::Config::LaserConfig::HotElectronConfig& cfg,
    const Geometry1D geom,
    const std::vector<HotESource>& sources,
    const double* d_rho, const double* d_zbar, const double* d_A_eff,
    const double* d_Te_eV, const std::vector<std::uint8_t>& cell_is_void_host,
    const double* d_r_nodes, const int n_nodes,
    cudaStream_t stream,
    std::vector<double>& dep_power_cell) {
  return deposit_hot_electrons_cone_1d_device(make_channel_spec_from_shorthand(cfg),
                                              geom, sources, d_rho, d_zbar,
                                              d_A_eff, d_Te_eV, cell_is_void_host,
                                              d_r_nodes, n_nodes, stream,
                                              dep_power_cell);
}

}  // namespace tenryu::laser::hot_electron
