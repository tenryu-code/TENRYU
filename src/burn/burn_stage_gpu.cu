#include "burn/burn_stage_gpu.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>

#include "burn/deposition.cuh"
#include "burn/neutron_heating_device.cuh"
#include "burn/partition_device.cuh"
#include "burn/screening_device.cuh"
#include "core/device_scratch.hpp"
#include "core/error.hpp"

namespace tenryu::burn {
namespace {

constexpr double kProtonMassG = 1.6726219e-24;
constexpr double kBarnToCm2 = 1.0e-24;
constexpr int kReductionBlock = 256;

struct CellTallies {
  double w_dt;
  double w_dd;
  double wTi_dt;
  double wTi_dd;
  double wvr2_dt;
  double wvr2_dd;
  double released_charged;
  double released_neutron;
  double charged_dep_e;
  double charged_dep_i;
  double n_neutrons_dt;
  double n_neutrons_dd;
  double dt_limit_s;
  double dt_limit_subcycle;
  int max_substeps;
  int max_substeps_required;
  int subcycle_saturated_cells;
  unsigned int screening_warning_flags;
};

struct NeutronHeatingTallies {
  double dep_e;
  double dep_i;
  double degraded;
  double escaped;
  double emitted;
};

struct ResultPacket {
  double released_charged;
  double released_neutron;
  double charged_dep_e;
  double charged_dep_i;
  double dep_e;
  double dep_i;
  double n_neutrons_dt;
  double n_neutrons_dd;
  double w_dt;
  double w_dd;
  double wTi_dt;
  double wTi_dd;
  double wvr2_dt;
  double wvr2_dd;
  double dt_limit_s;
  double dt_limit_subcycle;
  int max_substeps;
  int max_substeps_required;
  int subcycle_saturated_cells;
  unsigned int screening_warning_flags;
  double nh_dep_e;
  double nh_dep_i;
  double nh_degraded;
  double nh_escaped;
  double nh_conservation_resid;
  int neutron_heating_invalid_mu;
};

std::size_t align_up(const std::size_t value, const std::size_t alignment) {
  return (value + alignment - 1U) & ~(alignment - 1U);
}

struct PackedLayout {
  std::size_t bytes = 0U;
  std::size_t output_bytes = 0U;
  std::size_t burn_y = 0U;
  std::size_t dE_e = 0U;
  std::size_t dE_i = 0U;
  std::size_t rate_diag = 0U;
  std::size_t Qe_diag = 0U;
  std::size_t Qi_diag = 0U;
  std::size_t S_birth = 0U;
  std::size_t nh_emit = 0U;
  std::size_t result = 0U;
  std::size_t rho = 0U;
  std::size_t Ti_eV = 0U;
  std::size_t Te_eV = 0U;
  std::size_t zbar = 0U;
  std::size_t A_eff = 0U;
  std::size_t vol = 0U;
  std::size_t r_node = 0U;
  std::size_t ee = 0U;
  std::size_t ei = 0U;
  std::size_t v_r = 0U;
  std::size_t mu = 0U;
  std::size_t mu_weight = 0U;
  std::size_t S_out = 0U;
  std::size_t cell_tallies = 0U;
  std::size_t neutron_nD = 0U;
  std::size_t neutron_nT = 0U;
  std::size_t neutron_dep_e_by_source = 0U;
  std::size_t neutron_dep_i_by_source = 0U;
  std::size_t neutron_tallies = 0U;

  template <typename T>
  std::size_t take(const std::size_t count) {
    bytes = align_up(bytes, alignof(T));
    const std::size_t offset = bytes;
    bytes += count * sizeof(T);
    return offset;
  }

  PackedLayout(const std::size_t n, const bool have_birth,
               const bool have_neutron, const bool have_velocity,
               const std::size_t n_mu) {
    burn_y = take<double>(n * static_cast<std::size_t>(kNumSpecies));
    dE_e = take<double>(n);
    dE_i = take<double>(n);
    rate_diag = take<double>(n);
    Qe_diag = take<double>(n);
    Qi_diag = take<double>(n);
    if (have_birth) {
      S_birth = take<double>(6U * n);
    }
    if (have_neutron) {
      nh_emit = take<double>(
          n * static_cast<std::size_t>(kNumNeutronLines));
    }
    result = take<ResultPacket>(1U);
    output_bytes = bytes;

    rho = take<double>(n);
    Ti_eV = take<double>(n);
    Te_eV = take<double>(n);
    zbar = take<double>(n);
    A_eff = take<double>(n);
    vol = take<double>(n);
    r_node = take<double>(n + 1U);
    ee = take<double>(n);
    ei = take<double>(n);
    if (have_velocity) {
      v_r = take<double>(n);
    }
    if (have_neutron) {
      mu = take<double>(n_mu);
      mu_weight = take<double>(n_mu);
    }
    S_out = take<double>(n);
    cell_tallies = take<CellTallies>(n);
    if (have_neutron) {
      neutron_nD = take<double>(n);
      neutron_nT = take<double>(n);
      const std::size_t contribution_count =
          static_cast<std::size_t>(kNumNeutronLines) * n * n;
      neutron_dep_e_by_source = take<double>(contribution_count);
      neutron_dep_i_by_source = take<double>(contribution_count);
      neutron_tallies = take<NeutronHeatingTallies>(n);
    }
  }
};

PartitionTableDeviceView partition_table_device_view(
    const PartitionTable& table, const bool upload_values) {
  PartitionTableDeviceView view{
      table.n_te,
      table.n_ti,
      table.n_ne,
      table.te_min_keV,
      table.te_max_keV,
      table.ti_min_keV,
      table.ti_max_keV,
      table.ne_min,
      table.ne_max,
      nullptr};
  if (!upload_values) {
    return view;
  }

  const std::size_t value_count =
      static_cast<std::size_t>(PartitionTable::kNumProductSlots) *
      static_cast<std::size_t>(table.n_te) *
      static_cast<std::size_t>(table.n_ti) *
      static_cast<std::size_t>(table.n_ne);
  TENRYU_ASSERT(table.f_ion.size() == value_count,
                "burn partition table size mismatch");

  struct Cache {
    const PartitionTable* key = nullptr;
    const double* host_values = nullptr;
    const double* device_values = nullptr;
    std::size_t value_count = 0U;
    double x_D = 0.0;
    double x_T = 0.0;
    double x_He3 = 0.0;
  };
  static Cache cache;
  if (cache.key != &table || cache.host_values != table.f_ion.data() ||
      cache.value_count != value_count || cache.x_D != table.x_D ||
      cache.x_T != table.x_T || cache.x_He3 != table.x_He3) {
    auto* const device_values = static_cast<double*>(
        core::device_scratch_acquire("burn:stage1d:partition", value_count *
                                                                    sizeof(double)));
    CUDA_CHECK(cudaMemcpy(device_values, table.f_ion.data(),
                          value_count * sizeof(double),
                          cudaMemcpyHostToDevice));
    cache.key = &table;
    cache.host_values = table.f_ion.data();
    cache.device_values = device_values;
    cache.value_count = value_count;
    cache.x_D = table.x_D;
    cache.x_T = table.x_T;
    cache.x_He3 = table.x_He3;
  }
  view.f_ion = cache.device_values;
  return view;
}

template <typename T>
T* packed_ptr(unsigned char* const base, const std::size_t offset) {
  return reinterpret_cast<T*>(base + offset);
}

template <typename T>
const T* packed_ptr(const unsigned char* const base,
                    const std::size_t offset) {
  return reinterpret_cast<const T*>(base + offset);
}

void pack_doubles(unsigned char* const base, const std::size_t offset,
                  const double* const source, const std::size_t count) {
  std::memcpy(base + offset, source, count * sizeof(double));
}

__host__ __device__ inline int product_slot(const int reaction,
                                            const int product_index) {
  constexpr int slots[kNumReactions][2] = {
      {0, -1}, {1, 2}, {3, -1}, {4, 5}};
  return slots[reaction][product_index];
}

__host__ __device__ inline double clamp01(const double x) {
  return (x < 0.0) ? 0.0 : ((x > 1.0) ? 1.0 : x);
}

__device__ inline double min_like_std(const double a, const double b) {
  return (b < a) ? b : a;
}

__device__ inline CellTallies empty_cell_tallies() {
  CellTallies out{};
  out.dt_limit_s = CUDART_INF;
  out.dt_limit_subcycle = CUDART_INF;
  return out;
}

__global__ void burn_suffix_sum_1d_kernel(
    const int first, const int last, const double* __restrict__ r_node,
    const double* __restrict__ rho, double* __restrict__ S_out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const double dr_last = r_node[last + 1] - r_node[last];
  S_out[last] = 0.5 * rho[last] * dr_last;
  for (int c = last - 1; c >= first; --c) {
    const double dr_c = r_node[c + 1] - r_node[c];
    const double dr_cp1 = r_node[c + 2] - r_node[c + 1];
    S_out[c] = S_out[c + 1] + 0.5 * rho[c] * dr_c +
               0.5 * rho[c + 1] * dr_cp1;
  }
}

__global__ void burn_stage_main_1d_kernel(
    const int n_cells, const int first, const int last,
    const BurnStageParams p, const PartitionTableDeviceView partition_table,
    const double* __restrict__ rho,
    const double* __restrict__ Ti_eV, const double* __restrict__ Te_eV,
    const double* __restrict__ zbar, const double* __restrict__ A_eff,
    const double* __restrict__ vol, const double* __restrict__ r_node,
    const double* __restrict__ ee, const double* __restrict__ ei,
    const double* __restrict__ v_r, const double* __restrict__ S_out,
    double* __restrict__ burn_y, double* __restrict__ dE_e,
    double* __restrict__ dE_i, double* __restrict__ rate_diag,
    double* __restrict__ Qe_diag, double* __restrict__ Qi_diag,
    double* __restrict__ S_birth, double* __restrict__ nh_emit,
    CellTallies* __restrict__ cell_tallies) {
  const int c = first + blockIdx.x * blockDim.x + threadIdx.x;
  if (c > last || c >= n_cells) {
    return;
  }

  CellTallies tallies = empty_cell_tallies();
  if (rho[c] <= 0.0) {
    cell_tallies[c] = tallies;
    return;
  }

  const double T_keV = Ti_eV[c] * 1.0e-3;
  if (T_keV < p.T_floor_keV) {
    cell_tallies[c] = tallies;
    return;
  }

  double local_n[kNumSpecies];
  for (int s = 0; s < kNumSpecies; ++s) {
    local_n[s] = burn_y[c * kNumSpecies + s] * rho[c];
  }

  const double ne_c = zbar[c] * rho[c] / (A_eff[c] * kProtonMassG);
  double F[kNumReactions];
  const double* screen = nullptr;
  const ScreeningMode screening_mode =
      static_cast<ScreeningMode>(p.screening_mode);
  if (screening_mode != ScreeningMode::kNone) {
    burn_screening_factors_core(screening_mode, Ti_eV[c], Te_eV[c], ne_c,
                                local_n, F,
                                &tallies.screening_warning_flags);
    screen = F;
  }

  double counts[kNumReactions];
  int M_req = 0;
  const int M = burn_network_step(local_n, T_keV, p.dt_s, p.channels,
                                  p.eps_deplete, p.subcycle_max, counts,
                                  screen, &M_req);
  tallies.max_substeps_required = M_req;
  if (M_req > p.subcycle_max) {
    tallies.subcycle_saturated_cells = 1;
    const double dt_shrink =
        0.9 * p.dt_s * static_cast<double>(p.subcycle_max) /
        static_cast<double>(M_req);
    if (dt_shrink > 0.0) {
      tallies.dt_limit_subcycle = dt_shrink;
      tallies.dt_limit_s = dt_shrink;
    }
  }
  tallies.max_substeps = M;

  for (int s = 0; s < kNumSpecies; ++s) {
    burn_y[c * kNumSpecies + s] = local_n[s] / rho[c];
  }

  bool any_counts = false;
  for (int k = 0; k < kNumReactions; ++k) {
    any_counts = any_counts || (counts[k] != 0.0);
  }
  if (!any_counts) {
    cell_tallies[c] = tallies;
    return;
  }

  const double rate_dt = counts[kDT] / p.dt_s;
  const double rate_dd = counts[kDDn] / p.dt_s;
  const double vol_c = vol[c];
  const double vr_c = (v_r != nullptr) ? v_r[c] : 0.0;
  const double vr2_c = vr_c * vr_c;
  tallies.w_dt += rate_dt * vol_c;
  tallies.w_dd += rate_dd * vol_c;
  tallies.wTi_dt += rate_dt * vol_c * Ti_eV[c];
  tallies.wTi_dd += rate_dd * vol_c * Ti_eV[c];
  tallies.wvr2_dt += rate_dt * vol_c * vr2_c;
  tallies.wvr2_dd += rate_dd * vol_c * vr2_c;

  const double rc = 0.5 * (r_node[c] + r_node[c + 1]);
  const double R_b = r_node[last + 1];
  const double u = clamp01(rc / R_b);
  const double span = R_b - rc;
  const double rho_bar =
      (span > 1.0e-9 * R_b) ? (S_out[c] / span) : rho[c];
  const double Te_keV_c = Te_eV[c] * 1.0e-3;
  const double Ti_keV_c = ((Ti_eV[c] > 0.0) ? Ti_eV[c] : 0.0) * 1.0e-3;
  const double rho_lam_alpha = alpha_rho_lambda(Te_keV_c, rho[c]);

  for (int k = 0; k < kNumReactions; ++k) {
    if (!(counts[k] > 0.0)) {
      continue;
    }

    const double E_rel_neutron =
        counts[k] * vol[c] * neutron_MeV(k) * kMeVToErg;
    tallies.released_neutron += E_rel_neutron;
    const double n_reactions = counts[k] * vol[c];
    if (k == kDT) {
      tallies.n_neutrons_dt += n_reactions;
    } else if (k == kDDn) {
      tallies.n_neutrons_dd += n_reactions;
    }
    if (p.neutron_heating && E_rel_neutron > 0.0) {
      const int line = (k == kDT) ? 0 : 1;
      nh_emit[c * kNumNeutronLines + line] += E_rel_neutron;
    }

    for (int i = 0; i < 2; ++i) {
      const int sp = charged_species(k, i);
      if (sp < 0) {
        continue;
      }

      const double E_MeV = charged_MeV(k, i);
      const double E_rel = counts[k] * vol[c] * E_MeV * kMeVToErg;
      tallies.released_charged += E_rel;
      if (p.scheme == 1) {
        if (S_birth != nullptr) {
          const int slot = product_slot(k, i);
          if (slot >= 0) {
            S_birth[slot * n_cells + c] += counts[k] / p.dt_s;
          }
        }
        continue;
      }
      const double rho_lam_s =
          rho_lam_alpha * range_scale_factor(sp, E_MeV);
      const double tau =
          (rho_lam_s > 0.0) ? (R_b * rho_bar / rho_lam_s) : 1.0e30;
      const double f_dep = point_sphere_deposited_fraction(u, tau);
      const double f_ion =
          p.use_fraley_partition
              ? fraley_ion_fraction_device(Te_keV_c)
              : partition_f_ion_device(partition_table, product_slot(k, i),
                                       Te_keV_c, Ti_keV_c, ne_c);

      dE_i[c] += f_ion * f_dep * E_rel;
      dE_e[c] += (1.0 - f_ion) * f_dep * E_rel;
    }
  }

  const double rate_total =
      (counts[0] + counts[1] + counts[2] + counts[3]) / p.dt_s;
  rate_diag[c] = rate_total;
  Qe_diag[c] = dE_e[c] / (vol[c] * p.dt_s);
  Qi_diag[c] = dE_i[c] / (vol[c] * p.dt_s);

  const double P_dep_c = (dE_e[c] + dE_i[c]) / p.dt_s;
  if (P_dep_c > 0.0) {
    const double internal_energy = ee[c] + ei[c];
    const double e_cell =
        rho[c] * vol[c] * ((internal_energy < 0.0) ? 0.0 : internal_energy);
    if (e_cell > 0.0) {
      const double cand = p.explicit_source_limit * e_cell / P_dep_c;
      tallies.dt_limit_s = min_like_std(tallies.dt_limit_s, cand);
    }
  }
  tallies.charged_dep_e = dE_e[c];
  tallies.charged_dep_i = dE_i[c];
  cell_tallies[c] = tallies;
}

__global__ void burn_neutron_targets_1d_kernel(
    const int n_cells, const double* __restrict__ rho,
    const double* __restrict__ burn_y, double* __restrict__ nD,
    double* __restrict__ nT) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  nD[c] = 0.0;
  nT[c] = 0.0;
  if (rho[c] > 0.0) {
    nD[c] = burn_y[c * kNumSpecies + kD] * rho[c];
    nT[c] = burn_y[c * kNumSpecies + kT] * rho[c];
  }
}

struct NeutronSegmentAccumulator {
  double E_ch;
  double T_rem;
  double sD;
  double sT;
  double fD;
  double fT;
  int slotD;
  int slotT;
  const double* nD;
  const double* nT;
  const double* rho;
  const double* Te_eV;
  const double* Ti_eV;
  const double* zbar;
  const double* A_eff;
  bool use_fraley_partition;
  PartitionTableDeviceView partition_table;
  double* dep_e;
  double* dep_i;
  NeutronHeatingTallies* tallies;

  TENRYU_HOST_DEVICE void operator()(const double ds, const int c) {
    const neutron_heating_device_detail::SegmentDeposit dep =
        neutron_heating_device_detail::deposit_segment(
            ds, c, E_ch, T_rem, sD, sT, fD, fT, slotD, slotT, nD, nT,
            rho, Te_eV, Ti_eV, zbar, A_eff, use_fraley_partition,
            partition_table);
    dep_i[c] += dep.dep_i_D;
    dep_e[c] += dep.dep_e_D;
    tallies->dep_i += dep.dep_i_D;
    tallies->dep_e += dep.dep_e_D;
    tallies->degraded += dep.degraded_D;
    dep_i[c] += dep.dep_i_T;
    dep_e[c] += dep.dep_e_T;
    tallies->dep_i += dep.dep_i_T;
    tallies->dep_e += dep.dep_e_T;
    tallies->degraded += dep.degraded_T;
    T_rem = dep.transmission;
  }
};

__global__ void burn_neutron_transport_1d_kernel(
    const int n_cells, const int n_mu, const bool use_fraley_partition,
    const PartitionTableDeviceView partition_table,
    const double* __restrict__ r_node, const double* __restrict__ rho,
    const double* __restrict__ Te_eV, const double* __restrict__ Ti_eV,
    const double* __restrict__ zbar, const double* __restrict__ A_eff,
    const double* __restrict__ nD, const double* __restrict__ nT,
    const double* __restrict__ emit_erg, const double* __restrict__ mu,
    const double* __restrict__ mu_weight,
    double* __restrict__ dep_e_by_source,
    double* __restrict__ dep_i_by_source,
    NeutronHeatingTallies* __restrict__ neutron_tallies) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n_cells) {
    return;
  }

  NeutronHeatingTallies tallies{};
  for (int l = 0; l < kNumNeutronLines; ++l) {
    const double E_cell = emit_erg[j * kNumNeutronLines + l];
    if (!(E_cell > 0.0)) {
      continue;
    }
    tallies.emitted += E_cell;
    const double sD = neutron_sigma_el_barn(l, 0) * kBarnToCm2;
    const double sT = neutron_sigma_el_barn(l, 1) * kBarnToCm2;
    const double fD = neutron_f_transfer(l, 0);
    const double fT = neutron_f_transfer(l, 1);
    const int slotD = neutron_recoil_slot(l, 0);
    const int slotT = neutron_recoil_slot(l, 1);
    const double r0 = 0.5 * (r_node[j] + r_node[j + 1]);
    const std::size_t row =
        (static_cast<std::size_t>(l) * static_cast<std::size_t>(n_cells) +
         static_cast<std::size_t>(j)) *
        static_cast<std::size_t>(n_cells);
    double* const dep_e = dep_e_by_source + row;
    double* const dep_i = dep_i_by_source + row;
    for (int q = 0; q < n_mu; ++q) {
      const double E_ch = E_cell * 0.5 * mu_weight[q];
      NeutronSegmentAccumulator accumulator{
          E_ch,
          1.0,
          sD,
          sT,
          fD,
          fT,
          slotD,
          slotT,
          nD,
          nT,
          rho,
          Te_eV,
          Ti_eV,
          zbar,
          A_eff,
          use_fraley_partition,
          partition_table,
          dep_e,
          dep_i,
          &tallies};
      neutron_heating_device_detail::walk_spherical_chord(
          r0, mu[q], r_node, n_cells, accumulator);
      tallies.escaped += E_ch * accumulator.T_rem;
    }
  }
  neutron_tallies[j] = tallies;
}

__global__ void burn_neutron_finalize_1d_kernel(
    const int n_cells, const BurnStageParams p,
    const double* __restrict__ rho, const double* __restrict__ vol,
    const double* __restrict__ ee, const double* __restrict__ ei,
    const double* __restrict__ dep_e_by_source,
    const double* __restrict__ dep_i_by_source, double* __restrict__ dE_e,
    double* __restrict__ dE_i, double* __restrict__ Qe_diag,
    double* __restrict__ Qi_diag,
    CellTallies* __restrict__ cell_tallies) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  double dep_e = dE_e[c];
  double dep_i = dE_i[c];
  for (int l = 0; l < kNumNeutronLines; ++l) {
    for (int j = 0; j < n_cells; ++j) {
      const std::size_t index =
          (static_cast<std::size_t>(l) *
               static_cast<std::size_t>(n_cells) +
           static_cast<std::size_t>(j)) *
              static_cast<std::size_t>(n_cells) +
          static_cast<std::size_t>(c);
      dep_e += dep_e_by_source[index];
      dep_i += dep_i_by_source[index];
    }
  }
  dE_e[c] = dep_e;
  dE_i[c] = dep_i;

  CellTallies tallies = cell_tallies[c];
  tallies.dt_limit_s = CUDART_INF;
  if (vol[c] > 0.0) {
    Qe_diag[c] = dep_e / (vol[c] * p.dt_s);
    Qi_diag[c] = dep_i / (vol[c] * p.dt_s);
    const double P_dep_c = (dep_e + dep_i) / p.dt_s;
    if (P_dep_c > 0.0) {
      const double internal_energy = ee[c] + ei[c];
      const double e_cell =
          rho[c] * vol[c] * ((internal_energy < 0.0) ? 0.0 : internal_energy);
      if (e_cell > 0.0) {
        tallies.dt_limit_s = p.explicit_source_limit * e_cell / P_dep_c;
      }
    }
  }
  cell_tallies[c] = tallies;
}

__device__ CellTallies combine_tallies(const CellTallies& a,
                                       const CellTallies& b) {
  CellTallies out;
  out.w_dt = a.w_dt + b.w_dt;
  out.w_dd = a.w_dd + b.w_dd;
  out.wTi_dt = a.wTi_dt + b.wTi_dt;
  out.wTi_dd = a.wTi_dd + b.wTi_dd;
  out.wvr2_dt = a.wvr2_dt + b.wvr2_dt;
  out.wvr2_dd = a.wvr2_dd + b.wvr2_dd;
  out.released_charged = a.released_charged + b.released_charged;
  out.released_neutron = a.released_neutron + b.released_neutron;
  out.charged_dep_e = a.charged_dep_e + b.charged_dep_e;
  out.charged_dep_i = a.charged_dep_i + b.charged_dep_i;
  out.n_neutrons_dt = a.n_neutrons_dt + b.n_neutrons_dt;
  out.n_neutrons_dd = a.n_neutrons_dd + b.n_neutrons_dd;
  out.dt_limit_s = min_like_std(a.dt_limit_s, b.dt_limit_s);
  out.dt_limit_subcycle =
      min_like_std(a.dt_limit_subcycle, b.dt_limit_subcycle);
  out.max_substeps =
      (a.max_substeps < b.max_substeps) ? b.max_substeps : a.max_substeps;
  out.max_substeps_required =
      (a.max_substeps_required < b.max_substeps_required)
          ? b.max_substeps_required
          : a.max_substeps_required;
  out.subcycle_saturated_cells =
      a.subcycle_saturated_cells + b.subcycle_saturated_cells;
  out.screening_warning_flags =
      a.screening_warning_flags | b.screening_warning_flags;
  return out;
}

__device__ NeutronHeatingTallies combine_neutron_tallies(
    const NeutronHeatingTallies& a, const NeutronHeatingTallies& b) {
  NeutronHeatingTallies out;
  out.dep_e = a.dep_e + b.dep_e;
  out.dep_i = a.dep_i + b.dep_i;
  out.degraded = a.degraded + b.degraded;
  out.escaped = a.escaped + b.escaped;
  out.emitted = a.emitted + b.emitted;
  return out;
}

__global__ void burn_stage_reduce_1d_kernel(
    const int n_cells, const double* __restrict__ dE_e,
    const double* __restrict__ dE_i,
    const CellTallies* __restrict__ cell_tallies,
    const NeutronHeatingTallies* __restrict__ neutron_tallies,
    const bool have_neutron, const bool neutron_mu_valid,
    ResultPacket* __restrict__ result) {
  __shared__ CellTallies shared_tallies[kReductionBlock];
  __shared__ NeutronHeatingTallies shared_neutron[kReductionBlock];
  __shared__ double shared_dep_e[kReductionBlock];
  __shared__ double shared_dep_i[kReductionBlock];

  const int tid = threadIdx.x;
  CellTallies local = empty_cell_tallies();
  NeutronHeatingTallies local_neutron{};
  double local_dep_e = 0.0;
  double local_dep_i = 0.0;
  for (int c = tid; c < n_cells; c += blockDim.x) {
    local = combine_tallies(local, cell_tallies[c]);
    if (have_neutron && neutron_mu_valid) {
      local_neutron =
          combine_neutron_tallies(local_neutron, neutron_tallies[c]);
    }
    local_dep_e += dE_e[c];
    local_dep_i += dE_i[c];
  }
  shared_tallies[tid] = local;
  shared_neutron[tid] = local_neutron;
  shared_dep_e[tid] = local_dep_e;
  shared_dep_i[tid] = local_dep_i;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_tallies[tid] =
          combine_tallies(shared_tallies[tid], shared_tallies[tid + stride]);
      shared_neutron[tid] = combine_neutron_tallies(
          shared_neutron[tid], shared_neutron[tid + stride]);
      shared_dep_e[tid] += shared_dep_e[tid + stride];
      shared_dep_i[tid] += shared_dep_i[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const CellTallies& t = shared_tallies[0];
    result->released_charged = t.released_charged;
    result->released_neutron = t.released_neutron;
    result->charged_dep_e = t.charged_dep_e;
    result->charged_dep_i = t.charged_dep_i;
    result->dep_e = shared_dep_e[0];
    result->dep_i = shared_dep_i[0];
    result->n_neutrons_dt = t.n_neutrons_dt;
    result->n_neutrons_dd = t.n_neutrons_dd;
    result->w_dt = t.w_dt;
    result->w_dd = t.w_dd;
    result->wTi_dt = t.wTi_dt;
    result->wTi_dd = t.wTi_dd;
    result->wvr2_dt = t.wvr2_dt;
    result->wvr2_dd = t.wvr2_dd;
    result->dt_limit_s = min_like_std(t.dt_limit_s, t.dt_limit_subcycle);
    result->dt_limit_subcycle = t.dt_limit_subcycle;
    result->max_substeps = t.max_substeps;
    result->max_substeps_required = t.max_substeps_required;
    result->subcycle_saturated_cells = t.subcycle_saturated_cells;
    result->screening_warning_flags = t.screening_warning_flags;
    const NeutronHeatingTallies& nh = shared_neutron[0];
    result->nh_dep_e = nh.dep_e;
    result->nh_dep_i = nh.dep_i;
    result->nh_degraded = nh.degraded;
    result->nh_escaped = nh.escaped;
    const double conservation_delta =
        nh.emitted - (nh.dep_e + nh.dep_i + nh.degraded + nh.escaped);
    const double conservation_abs =
        (conservation_delta < 0.0) ? -conservation_delta : conservation_delta;
    const double conservation_scale =
        (nh.emitted > 1.0e-300) ? nh.emitted : 1.0e-300;
    result->nh_conservation_resid = conservation_abs / conservation_scale;
    result->neutron_heating_invalid_mu =
        (have_neutron && !neutron_mu_valid && t.released_neutron > 0.0) ? 1
                                                                       : 0;
  }
}

}  // namespace

BurnStageResult compute_burn_step_1d_device_stage(
    const BurnStageInputs& in, const BurnStageParams& p,
    const PartitionTable& table,
    std::vector<double>& burn_y, std::vector<double>& dE_e,
    std::vector<double>& dE_i, std::vector<double>& rate_diag,
    std::vector<double>& Qe_diag, std::vector<double>& Qi_diag,
    std::vector<double>* S_birth, std::vector<double>& nh_emit,
    double& dt_limit_subcycle, unsigned int& screening_warning_flags) {
  const int n_cells = in.n_cells;
  const std::size_t n = static_cast<std::size_t>(n_cells);
  dE_e.assign(n, 0.0);
  dE_i.assign(n, 0.0);
  rate_diag.assign(n, 0.0);
  Qe_diag.assign(n, 0.0);
  Qi_diag.assign(n, 0.0);
  const bool have_birth = p.scheme == 1 && S_birth != nullptr;
  if (have_birth) {
    S_birth->assign(6U * n, 0.0);
  }
  const bool have_neutron = p.neutron_heating;
  const bool neutron_mu_valid =
      p.neutron_heating_n_mu >= 2 && (p.neutron_heating_n_mu % 2) == 0;
  std::vector<double> neutron_mu;
  std::vector<double> neutron_mu_weight;
  if (have_neutron) {
    nh_emit.assign(n * static_cast<std::size_t>(kNumNeutronLines), 0.0);
    if (neutron_mu_valid) {
      neutron_mu.assign(static_cast<std::size_t>(p.neutron_heating_n_mu),
                        0.0);
      neutron_mu_weight.assign(
          static_cast<std::size_t>(p.neutron_heating_n_mu), 0.0);
      const int half = p.neutron_heating_n_mu / 2;
      for (int k = 0; k < half; ++k) {
        double weight = 0.0;
        neutron_heating_device_detail::gauss_legendre_node_weight(
            p.neutron_heating_n_mu, k,
            &neutron_mu[static_cast<std::size_t>(k)],
            &neutron_mu[static_cast<std::size_t>(p.neutron_heating_n_mu - 1 -
                                                k)],
            &weight);
        neutron_mu_weight[static_cast<std::size_t>(k)] = weight;
        neutron_mu_weight[static_cast<std::size_t>(
            p.neutron_heating_n_mu - 1 - k)] = weight;
      }
    }
  } else {
    nh_emit.clear();
  }

  BurnStageResult result;
  result.dt_limit_s = std::numeric_limits<double>::infinity();
  dt_limit_subcycle = std::numeric_limits<double>::infinity();
  screening_warning_flags = 0U;

  int first = -1;
  int last = -1;
  for (int c = 0; c < n_cells; ++c) {
    double fuel_frac = 0.0;
    for (int i = 0; i < in.n_fuel_mat; ++i) {
      const int m = in.fuel_mat[i];
      fuel_frac += in.volFrac[c * in.n_mat + m];
    }
    if (fuel_frac > p.vf_threshold) {
      if (first < 0) {
        first = c;
      }
      last = c;
    }
  }

  if (first < 0) {
    return result;
  }
  result.burn_region_first = first;
  result.burn_region_last = last;

  const bool need_partition_table =
      !p.use_fraley_partition && (p.scheme != 1 || have_neutron);
  const PartitionTableDeviceView partition_table =
      partition_table_device_view(table, need_partition_table);
  const std::size_t packed_n_mu =
      (have_neutron && neutron_mu_valid)
          ? static_cast<std::size_t>(p.neutron_heating_n_mu)
          : 0U;
  const PackedLayout layout(n, have_birth, have_neutron, in.v_r != nullptr,
                            packed_n_mu);
  std::vector<unsigned char> host_pack(layout.bytes, 0U);
  unsigned char* const host_base = host_pack.data();
  pack_doubles(host_base, layout.burn_y, burn_y.data(),
               n * static_cast<std::size_t>(kNumSpecies));
  pack_doubles(host_base, layout.rho, in.rho, n);
  pack_doubles(host_base, layout.Ti_eV, in.Ti_eV, n);
  pack_doubles(host_base, layout.Te_eV, in.Te_eV, n);
  pack_doubles(host_base, layout.zbar, in.zbar, n);
  pack_doubles(host_base, layout.A_eff, in.A_eff, n);
  pack_doubles(host_base, layout.vol, in.vol, n);
  pack_doubles(host_base, layout.r_node, in.r_node, n + 1U);
  pack_doubles(host_base, layout.ee, in.ee, n);
  pack_doubles(host_base, layout.ei, in.ei, n);
  if (in.v_r != nullptr) {
    pack_doubles(host_base, layout.v_r, in.v_r, n);
  }
  if (have_neutron && neutron_mu_valid) {
    pack_doubles(host_base, layout.mu, neutron_mu.data(), packed_n_mu);
    pack_doubles(host_base, layout.mu_weight, neutron_mu_weight.data(),
                 packed_n_mu);
  }

  std::vector<CellTallies> initial_tallies(n);
  for (CellTallies& t : initial_tallies) {
    t.dt_limit_s = std::numeric_limits<double>::infinity();
    t.dt_limit_subcycle = std::numeric_limits<double>::infinity();
  }
  std::memcpy(host_base + layout.cell_tallies, initial_tallies.data(),
              n * sizeof(CellTallies));

  auto* const device_base = static_cast<unsigned char*>(
      core::device_scratch_acquire("burn:stage1d:arena", layout.bytes));
  CUDA_CHECK(cudaMemcpy(device_base, host_base, layout.bytes,
                        cudaMemcpyHostToDevice));

  double* const d_burn_y = packed_ptr<double>(device_base, layout.burn_y);
  double* const d_dE_e = packed_ptr<double>(device_base, layout.dE_e);
  double* const d_dE_i = packed_ptr<double>(device_base, layout.dE_i);
  double* const d_rate_diag =
      packed_ptr<double>(device_base, layout.rate_diag);
  double* const d_Qe_diag = packed_ptr<double>(device_base, layout.Qe_diag);
  double* const d_Qi_diag = packed_ptr<double>(device_base, layout.Qi_diag);
  double* const d_S_birth =
      have_birth ? packed_ptr<double>(device_base, layout.S_birth) : nullptr;
  double* const d_nh_emit =
      have_neutron ? packed_ptr<double>(device_base, layout.nh_emit) : nullptr;
  const double* const d_rho = packed_ptr<double>(device_base, layout.rho);
  const double* const d_Ti_eV = packed_ptr<double>(device_base, layout.Ti_eV);
  const double* const d_Te_eV = packed_ptr<double>(device_base, layout.Te_eV);
  const double* const d_zbar = packed_ptr<double>(device_base, layout.zbar);
  const double* const d_A_eff = packed_ptr<double>(device_base, layout.A_eff);
  const double* const d_vol = packed_ptr<double>(device_base, layout.vol);
  const double* const d_r_node =
      packed_ptr<double>(device_base, layout.r_node);
  const double* const d_ee = packed_ptr<double>(device_base, layout.ee);
  const double* const d_ei = packed_ptr<double>(device_base, layout.ei);
  const double* const d_v_r =
      (in.v_r != nullptr) ? packed_ptr<double>(device_base, layout.v_r)
                          : nullptr;
  const double* const d_mu =
      (have_neutron && neutron_mu_valid)
          ? packed_ptr<double>(device_base, layout.mu)
          : nullptr;
  const double* const d_mu_weight =
      (have_neutron && neutron_mu_valid)
          ? packed_ptr<double>(device_base, layout.mu_weight)
          : nullptr;
  double* const d_S_out = packed_ptr<double>(device_base, layout.S_out);
  CellTallies* const d_cell_tallies =
      packed_ptr<CellTallies>(device_base, layout.cell_tallies);
  double* const d_neutron_nD =
      have_neutron ? packed_ptr<double>(device_base, layout.neutron_nD)
                   : nullptr;
  double* const d_neutron_nT =
      have_neutron ? packed_ptr<double>(device_base, layout.neutron_nT)
                   : nullptr;
  double* const d_neutron_dep_e_by_source =
      have_neutron
          ? packed_ptr<double>(device_base,
                               layout.neutron_dep_e_by_source)
          : nullptr;
  double* const d_neutron_dep_i_by_source =
      have_neutron
          ? packed_ptr<double>(device_base,
                               layout.neutron_dep_i_by_source)
          : nullptr;
  NeutronHeatingTallies* const d_neutron_tallies =
      have_neutron
          ? packed_ptr<NeutronHeatingTallies>(device_base,
                                              layout.neutron_tallies)
          : nullptr;
  ResultPacket* const d_result =
      packed_ptr<ResultPacket>(device_base, layout.result);

  burn_suffix_sum_1d_kernel<<<1, 1>>>(first, last, d_r_node, d_rho, d_S_out);
  CUDA_CHECK(cudaGetLastError());
  constexpr int kBlock = 128;
  const int active_cells = last - first + 1;
  const int active_grid = (active_cells + kBlock - 1) / kBlock;
  burn_stage_main_1d_kernel<<<active_grid, kBlock>>>(
      n_cells, first, last, p, partition_table, d_rho, d_Ti_eV, d_Te_eV,
      d_zbar, d_A_eff, d_vol, d_r_node, d_ee, d_ei, d_v_r, d_S_out,
      d_burn_y, d_dE_e, d_dE_i, d_rate_diag, d_Qe_diag, d_Qi_diag,
      d_S_birth, d_nh_emit, d_cell_tallies);
  CUDA_CHECK(cudaGetLastError());
  if (have_neutron && neutron_mu_valid) {
    const int cell_grid = (n_cells + kBlock - 1) / kBlock;
    burn_neutron_targets_1d_kernel<<<cell_grid, kBlock>>>(
        n_cells, d_rho, d_burn_y, d_neutron_nD, d_neutron_nT);
    CUDA_CHECK(cudaGetLastError());
    burn_neutron_transport_1d_kernel<<<cell_grid, kBlock>>>(
        n_cells, p.neutron_heating_n_mu, p.use_fraley_partition,
        partition_table, d_r_node, d_rho, d_Te_eV, d_Ti_eV, d_zbar,
        d_A_eff, d_neutron_nD, d_neutron_nT, d_nh_emit, d_mu,
        d_mu_weight, d_neutron_dep_e_by_source,
        d_neutron_dep_i_by_source, d_neutron_tallies);
    CUDA_CHECK(cudaGetLastError());
    burn_neutron_finalize_1d_kernel<<<cell_grid, kBlock>>>(
        n_cells, p, d_rho, d_vol, d_ee, d_ei,
        d_neutron_dep_e_by_source, d_neutron_dep_i_by_source, d_dE_e,
        d_dE_i, d_Qe_diag, d_Qi_diag, d_cell_tallies);
    CUDA_CHECK(cudaGetLastError());
  }
  burn_stage_reduce_1d_kernel<<<1, kReductionBlock>>>(
      n_cells, d_dE_e, d_dE_i, d_cell_tallies, d_neutron_tallies,
      have_neutron, neutron_mu_valid, d_result);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(host_base, device_base, layout.output_bytes,
                        cudaMemcpyDeviceToHost));

  std::memcpy(burn_y.data(), host_base + layout.burn_y,
              n * static_cast<std::size_t>(kNumSpecies) * sizeof(double));
  std::memcpy(dE_e.data(), host_base + layout.dE_e, n * sizeof(double));
  std::memcpy(dE_i.data(), host_base + layout.dE_i, n * sizeof(double));
  std::memcpy(rate_diag.data(), host_base + layout.rate_diag,
              n * sizeof(double));
  std::memcpy(Qe_diag.data(), host_base + layout.Qe_diag, n * sizeof(double));
  std::memcpy(Qi_diag.data(), host_base + layout.Qi_diag, n * sizeof(double));
  if (have_birth) {
    std::memcpy(S_birth->data(), host_base + layout.S_birth,
                6U * n * sizeof(double));
  }
  if (have_neutron) {
    std::memcpy(nh_emit.data(), host_base + layout.nh_emit,
                n * static_cast<std::size_t>(kNumNeutronLines) *
                    sizeof(double));
  }

  ResultPacket packet;
  std::memcpy(&packet, host_base + layout.result, sizeof(packet));
  result.released_charged = packet.released_charged;
  result.released_neutron = packet.released_neutron;
  result.dep_e = packet.dep_e;
  result.dep_i = packet.dep_i;
  result.n_neutrons_dt = packet.n_neutrons_dt;
  result.n_neutrons_dd = packet.n_neutrons_dd;
  result.w_dt = packet.w_dt;
  result.w_dd = packet.w_dd;
  result.wTi_dt = packet.wTi_dt;
  result.wTi_dd = packet.wTi_dd;
  result.wvr2_dt = packet.wvr2_dt;
  result.wvr2_dd = packet.wvr2_dd;
  result.dt_limit_s = packet.dt_limit_s;
  result.max_substeps = packet.max_substeps;
  result.max_substeps_required = packet.max_substeps_required;
  result.subcycle_saturated_cells = packet.subcycle_saturated_cells;
  result.nh_dep_e = packet.nh_dep_e;
  result.nh_dep_i = packet.nh_dep_i;
  result.nh_degraded = packet.nh_degraded;
  result.nh_escaped = packet.nh_escaped;
  result.nh_conservation_resid = packet.nh_conservation_resid;
  result.esc_charged =
      (p.scheme == 1)
          ? 0.0
          : (packet.released_charged - packet.charged_dep_e -
             packet.charged_dep_i);
  result.esc_neutron =
      packet.released_neutron - packet.nh_dep_e - packet.nh_dep_i;
  TENRYU_ASSERT(packet.neutron_heating_invalid_mu == 0,
                "neutron_heating requires an even n_mu >= 2");
  dt_limit_subcycle = packet.dt_limit_subcycle;
  screening_warning_flags = packet.screening_warning_flags;
  return result;
}

}  // namespace tenryu::burn
