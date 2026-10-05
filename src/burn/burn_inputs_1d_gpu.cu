#include "burn/burn_inputs_1d_gpu.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <limits>

#include "burn/burn_constants.hpp"
#include "core/constants.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/device_ordered_sum.cuh"

// Compiled with -fmad=false (src/burn/CMakeLists.txt): the sums of products in the field-ion means
// and the products in the bookkeeping after the transport are rounded before they are added, as
// in the host loops.

namespace tenryu::burn {
namespace {

constexpr int kBlock = 128;

int grid_for(const int n) { return (n + kBlock - 1) / kBlock; }

// The kinds of one cell's field-ion mixture in cell_field_ions' order: the inventory species, then
// the materials. A material whose ions do not join (not a field material, A <= 0 or vf <= 0) has
// zero weight, which field_ions_from_kind_source passes over as cell_field_ions leaves it out.
struct CellIonKinds {
  const double* y = nullptr;          // the cell's kNumSpecies inventories
  const double* vf = nullptr;         // the cell's volume fractions
  const double* materials = nullptr;  // kFieldMaterialValues per material
  int n_species = 0;                  // kNumSpecies when y is given, else 0

  TENRYU_HOST_DEVICE IonKind operator()(const int k) const {
    if (k < n_species) {
      return IonKind{y[k] * core::constants::proton_mass, species_A(k), species_Z(k)};
    }
    const int m = k - n_species;
    const double* const mat = materials + kFieldMaterialValues * m;
    const double A = mat[0];
    const double vf_m = vf[m];
    if (mat[2] != 0.0 && A > 0.0 && vf_m > 0.0) {
      return IonKind{vf_m / A, A, mat[1]};
    }
    return IonKind{0.0, A, mat[1]};
  }
};

__global__ void cell_velocities_kernel(const double* __restrict__ v_node, const int n_cells,
                                       double* __restrict__ v_cell) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < n_cells) {
    v_cell[c] = 0.5 * (v_node[c] + v_node[c + 1]);
  }
}

__global__ void range_medium_kernel(const double* __restrict__ burn_y,
                                    const double* __restrict__ volFrac, const FieldIonSetup setup,
                                    const int n_cells, double* __restrict__ fe,
                                    double* __restrict__ fi, double* __restrict__ field_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  CellIonKinds kinds;
  int n_kinds = 0;
  if (burn_y != nullptr) {
    kinds.y = burn_y + static_cast<std::size_t>(c) * kNumSpecies;
    kinds.n_species = kNumSpecies;
    n_kinds = kNumSpecies;
  }
  if (volFrac != nullptr && setup.n_mat > 0) {
    kinds.vf = volFrac + static_cast<std::size_t>(c) * setup.n_mat;
    kinds.materials = setup.materials;
    n_kinds += setup.n_mat;
  }
  const FieldIons ions = field_ions_from_kind_source(kinds, n_kinds, setup.fallback);
  const FraleyRangeMedium medium = fraley_range_medium(ions);
  fe[c] = medium.fe;
  fi[c] = medium.fi;
  if (field_cells != nullptr) {
    double* const v = field_cells + static_cast<std::size_t>(c) * kFieldIonCellValues;
    v[0] = ions.A_bar;
    v[1] = ions.z2_bar;
    v[2] = ions.z2_over_A_bar;
  }
}

__global__ void burn_cumulative_kernel(const double* __restrict__ rho,
                                       const double* __restrict__ vol,
                                       const double* __restrict__ dE_e,
                                       const double* __restrict__ dE_i,
                                       const double* __restrict__ births, const int n_cells,
                                       double* __restrict__ eps_cum,
                                       double* __restrict__ neutron_cum) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (eps_cum != nullptr) {
    const double denom = rho[c] * vol[c];
    if (denom > 1.0e-30) {
      eps_cum[c] += (dE_e[c] + dE_i[c]) / denom;
    }
  }
  if (births != nullptr && neutron_cum != nullptr) {
    neutron_cum[c] += births[c];
  }
}

__global__ void electron_density_kernel(const double* __restrict__ zbar,
                                        const double* __restrict__ rho,
                                        const double* __restrict__ A_eff, const int n_cells,
                                        double* __restrict__ ne) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const double denom = A_eff[c] * core::constants::proton_mass;
  ne[c] = (denom > 0.0) ? zbar[c] * rho[c] / denom : 0.0;
}

constexpr int kSlotFlags = 12;

__global__ void slot_nonzero_flags_kernel(const double* __restrict__ S_birth,
                                          const double* __restrict__ N, const int n_cells,
                                          const int n_groups, int* __restrict__ flags) {
  __shared__ int block_flags[kSlotFlags];
  if (threadIdx.x < kSlotFlags) {
    block_flags[threadIdx.x] = 0;
  }
  __syncthreads();
  const long long slot_cells = n_cells;
  const long long slot_spectra = static_cast<long long>(n_groups) * n_cells;
  const long long n_sources = 6LL * slot_cells;
  const long long n_total = n_sources + ((N != nullptr) ? 6LL * slot_spectra : 0LL);
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n_total;
       i += stride) {
    if (i < n_sources) {
      if (S_birth[i] != 0.0) {
        block_flags[i / slot_cells] = 1;
      }
    } else {
      const long long j = i - n_sources;
      if (N[j] != 0.0) {
        block_flags[6 + j / slot_spectra] = 1;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < kSlotFlags && block_flags[threadIdx.x] != 0) {
    atomicOr(&flags[threadIdx.x], 1);
  }
}

__global__ void transport_finish_cells_kernel(
    double* __restrict__ dep_e, double* __restrict__ dep_i, const double* __restrict__ nh_e,
    const double* __restrict__ nh_i, const double* __restrict__ rho,
    const double* __restrict__ vol, const double* __restrict__ ee, const double* __restrict__ ei,
    const int n_cells, const double dt, const double explicit_source_limit,
    double* __restrict__ eps_cum, double* __restrict__ Q_e, double* __restrict__ Q_i,
    double* __restrict__ candidate) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  double dE_e = dep_e[c];
  double dE_i = dep_i[c];
  if (nh_e != nullptr) {
    dE_e += nh_e[c];
    dE_i += nh_i[c];
    dep_e[c] = dE_e;
    dep_i[c] = dE_i;
  }
  const double denom = rho[c] * vol[c];
  if (denom > 1.0e-30) {
    eps_cum[c] += (dE_e + dE_i) / denom;
  }
  if (vol[c] > 0.0 && dt > 0.0) {
    Q_e[c] = dE_e / (vol[c] * dt);
    Q_i[c] = dE_i / (vol[c] * dt);
  } else {
    Q_e[c] = 0.0;
    Q_i[c] = 0.0;
  }
  // The cell's candidate of the explicit-source limit (infinity: none). std::max(x, 0.0) of the
  // host loop returns x unless x < 0 (a NaN stays).
  double limit = CUDART_INF;
  const double P_dep = (dE_e + dE_i) / dt;
  if (P_dep > 0.0) {
    const double e_sum = ee[c] + ei[c];
    const double e_cell = rho[c] * vol[c] * ((e_sum < 0.0) ? 0.0 : e_sum);
    if (e_cell > 0.0) {
      limit = explicit_source_limit * e_cell / P_dep;
    }
  }
  candidate[c] = limit;
}

constexpr int kFinishBlock = 256;
constexpr int kFinishPerThread = 4;

// The sums of dep_e and dep_i in cell order (block_ordered_sum_nonzero: zero entries skipped,
// which leaves a sum from +0 unchanged) and the minimum of the candidates as the host loop's
// std::min(limit, candidate) forms it (a NaN candidate is never taken).
__global__ void transport_finish_totals_kernel(const double* __restrict__ dep_e,
                                               const double* __restrict__ dep_i,
                                               const double* __restrict__ candidate,
                                               const int n_cells, double* __restrict__ totals) {
  __shared__ double sh_values[kFinishBlock * kFinishPerThread];
  __shared__ int sh_scan[kFinishBlock];
  __shared__ double sh_min[kFinishBlock];
  const double sum_e = core::device_ordered::block_ordered_sum_nonzero<kFinishBlock,
                                                                         kFinishPerThread>(
      dep_e, n_cells, 0.0, sh_values, sh_scan);
  const double sum_i = core::device_ordered::block_ordered_sum_nonzero<kFinishBlock,
                                                                         kFinishPerThread>(
      dep_i, n_cells, 0.0, sh_values, sh_scan);
  double m = CUDART_INF;
  for (int c = threadIdx.x; c < n_cells; c += kFinishBlock) {
    const double v = candidate[c];
    m = (v < m) ? v : m;
  }
  sh_min[threadIdx.x] = m;
  __syncthreads();
  for (int offset = kFinishBlock / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      const double v = sh_min[threadIdx.x + offset];
      sh_min[threadIdx.x] = (v < sh_min[threadIdx.x]) ? v : sh_min[threadIdx.x];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    totals[0] = sum_e;
    totals[1] = sum_i;
    totals[2] = sh_min[0];
  }
}

}  // namespace

void pack_field_ion_materials(const std::vector<FieldIonMaterial>& materials,
                              std::vector<double>& packed) {
  packed.assign(materials.size() * static_cast<std::size_t>(kFieldMaterialValues), 0.0);
  for (std::size_t m = 0; m < materials.size(); ++m) {
    double* const v = packed.data() + m * static_cast<std::size_t>(kFieldMaterialValues);
    v[0] = materials[m].A;
    v[1] = materials[m].Z;
    v[2] = materials[m].field ? 1.0 : 0.0;
  }
}

void cell_velocities_1d_device(const double* v_node, const int n_cells, double* v_cell) {
  if (n_cells <= 0) {
    return;
  }
  cell_velocities_kernel<<<grid_for(n_cells), kBlock>>>(v_node, n_cells, v_cell);
  TENRYU_ASSERT(cudaGetLastError() == cudaSuccess, "burn cell velocities launch failed");
}

void range_medium_1d_device(const double* burn_y, const double* volFrac,
                            const FieldIonSetup& setup, const int n_cells, double* fe,
                            double* fi, double* field_cells) {
  if (n_cells <= 0) {
    return;
  }
  TENRYU_ASSERT(setup.n_mat >= 0 && (setup.n_mat == 0 || setup.materials != nullptr),
                "burn range medium: the field-ion materials are missing");
  range_medium_kernel<<<grid_for(n_cells), kBlock>>>(burn_y, volFrac, setup, n_cells, fe, fi,
                                                     field_cells);
  TENRYU_ASSERT(cudaGetLastError() == cudaSuccess, "burn range medium launch failed");
}

void burn_cumulative_1d_device(const double* rho, const double* vol, const double* dE_e,
                               const double* dE_i, const double* births, const int n_cells,
                               double* eps_cum, double* neutron_cum) {
  if (n_cells <= 0) {
    return;
  }
  burn_cumulative_kernel<<<grid_for(n_cells), kBlock>>>(rho, vol, dE_e, dE_i, births, n_cells,
                                                        eps_cum, neutron_cum);
  TENRYU_ASSERT(cudaGetLastError() == cudaSuccess, "burn cumulative update launch failed");
}

void electron_density_1d_device(const double* zbar, const double* rho, const double* A_eff,
                                const int n_cells, double* ne) {
  if (n_cells <= 0) {
    return;
  }
  electron_density_kernel<<<grid_for(n_cells), kBlock>>>(zbar, rho, A_eff, n_cells, ne);
  TENRYU_ASSERT(cudaGetLastError() == cudaSuccess, "burn electron density launch failed");
}

void slot_nonzero_flags_1d_device(const double* S_birth, const double* N, const int n_cells,
                                  const int n_groups, int flags[12]) {
  for (int s = 0; s < kSlotFlags; ++s) {
    flags[s] = 0;
  }
  if (n_cells <= 0) {
    return;
  }
  auto* const d_flags = static_cast<int*>(
      core::device_scratch_acquire("burn:slot_nonzero_flags", kSlotFlags * sizeof(int)));
  TENRYU_ASSERT(cudaMemset(d_flags, 0, kSlotFlags * sizeof(int)) == cudaSuccess,
                "burn slot flags reset failed");
  const long long n_total =
      6LL * n_cells + ((N != nullptr) ? 6LL * static_cast<long long>(n_groups) * n_cells : 0LL);
  const long long blocks = (n_total + kBlock - 1) / kBlock;
  const int grid = static_cast<int>(blocks < 1024 ? blocks : 1024);
  slot_nonzero_flags_kernel<<<grid, kBlock>>>(S_birth, N, n_cells, n_groups, d_flags);
  TENRYU_ASSERT(cudaGetLastError() == cudaSuccess, "burn slot flags launch failed");
  TENRYU_ASSERT(cudaMemcpy(flags, d_flags, kSlotFlags * sizeof(int), cudaMemcpyDeviceToHost) ==
                    cudaSuccess,
                "burn slot flags copy failed");
}

BurnTransportTotals burn_transport_finish_1d_device(
    double* dep_e, double* dep_i, const double* nh_e, const double* nh_i, const double* rho,
    const double* vol, const double* ee, const double* ei, const int n_cells, const double dt,
    const double explicit_source_limit, double* eps_cum, double* Q_e, double* Q_i) {
  BurnTransportTotals totals;
  totals.dt_limit = std::numeric_limits<double>::infinity();
  if (n_cells <= 0) {
    return totals;
  }
  TENRYU_ASSERT((nh_e == nullptr) == (nh_i == nullptr),
                "burn transport finish: both neutron-heating deposits or neither");
  auto* const scratch = static_cast<double*>(core::device_scratch_acquire(
      "burn:transport_finish", (static_cast<std::size_t>(n_cells) + 3U) * sizeof(double)));
  double* const candidate = scratch;
  double* const d_totals = scratch + n_cells;
  transport_finish_cells_kernel<<<grid_for(n_cells), kBlock>>>(
      dep_e, dep_i, nh_e, nh_i, rho, vol, ee, ei, n_cells, dt, explicit_source_limit, eps_cum,
      Q_e, Q_i, candidate);
  TENRYU_ASSERT(cudaGetLastError() == cudaSuccess, "burn transport finish launch failed");
  transport_finish_totals_kernel<<<1, kFinishBlock>>>(dep_e, dep_i, candidate, n_cells,
                                                      d_totals);
  TENRYU_ASSERT(cudaGetLastError() == cudaSuccess, "burn transport totals launch failed");
  double host_totals[3] = {0.0, 0.0, 0.0};
  TENRYU_ASSERT(cudaMemcpy(host_totals, d_totals, sizeof(host_totals), cudaMemcpyDeviceToHost) ==
                    cudaSuccess,
                "burn transport totals copy failed");
  totals.dep_e = host_totals[0];
  totals.dep_i = host_totals[1];
  totals.dt_limit = host_totals[2];
  return totals;
}

}  // namespace tenryu::burn
