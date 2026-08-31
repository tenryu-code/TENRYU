#include "radiation/imc.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "core/device_error_flags.cuh"
#include "core/error.hpp"
#include "materials/ionmix_reader.hpp"
#include "mesh/cell_geometry_2d.cuh"
#include "materials/opacity.cuh"
#include "materials/tmat_reader.hpp"
#include "radiation/cell_radiation_coeffs.hpp"
#include "radiation/composite_sort.cuh"
#include "radiation/census_comb.cuh"
#include "radiation/ddmc.hpp"
#include "radiation/ddmc_diffusion_1d.cuh"
#include "radiation/deterministic_diffusion_1d.cuh"
#include "radiation/difference_residualization.cuh"
#include "radiation/ddmc_transport_2d_gpu.cuh"
#include "radiation/ddmc_transport_gpu.cuh"
#include "radiation/diffusion_conversion.cuh"
#include "radiation/diffusion_interface.cuh"
#include "radiation/diffusion_source_solve.cuh"
#include "radiation/fleck.cuh"
#include "radiation/group_structure.hpp"
#include "radiation/groups.cuh"
#include "radiation/holo_lo_solver.hpp"
#include "radiation/holo_lo_state.cuh"
#include "radiation/holo_selector.hpp"
#include "radiation/imc_transport_2d.cuh"
#include "radiation/imc_transport_persistent.cuh"
#include "radiation/fld_1d_gpu.cuh"
#include "radiation/fld_2d_rz_gpu.cuh"
#include "radiation/nlte_coeffs.cuh"
#include "radiation/nlte_coeffs.hpp"
#include "radiation/particle_reid.hpp"
#include "radiation/planck_table.cuh"
#include "radiation/pool_stats.cuh"
#include "radiation/rw_transport_gpu.cuh"
#include "radiation/sn_transport_1d.hpp"
#include "radiation/sn_transport_1d_gpu.cuh"
#include "radiation/sn_transport_2d_gpu.cuh"
#include "radiation/sn_transport_gpu.cuh"
#include "radiation/source.cuh"
#include "radiation/tally.cuh"
#include "radiation/rad_lite_mesh.hpp"

namespace tenryu::radiation {

double difference_reference_weight(const double W_max,
                                   const double tau,
                                   const double tau0,
                                   const double chi,
                                   const double chi0) {
  const double W_max_cfg = std::clamp(W_max, 0.0, 1.0);
  const double tau0_sq = tau0 * tau0;
  const double tau_sq = tau * tau;
  const double w_tau = (tau_sq > 0.0) ? (tau_sq / (tau_sq + tau0_sq)) : 0.0;
  const double chi_arg = chi / std::max(chi0, 1.0e-300);
  const double w_lte =
      (chi_arg > 1.0e75) ? 0.0 : (1.0 / (1.0 + chi_arg * chi_arg * chi_arg * chi_arg));
  return std::clamp(W_max_cfg * w_tau * w_lte, 0.0, W_max_cfg);
}

double difference_reference_cell_sigma(const double weight_sum,
                                       const double inverse_sigma_weight_sum,
                                       const double sigma_max) {
  return (weight_sum > 0.0 && inverse_sigma_weight_sum > 0.0)
             ? (weight_sum / inverse_sigma_weight_sum)
             : sigma_max;
}

namespace {

constexpr int kInfiniteLoopFatalThreshold = 10;
constexpr double kMaxTemperatureForT4 = 1.0e6;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kPgrwSeriesTerms = 100;
constexpr double kPgrwThetaMin = 1.0e-6;
constexpr double kPgrwThetaMaxDefault = 3.0;
constexpr double kPgrwGroupTauFactor = 20.0;
constexpr double kDiffusionRadiationEnergyThreshold = 1.0e-20;
constexpr double kDiffusionBalanceRelTol = 1.0e-8;
constexpr double kDiffusionBalanceFloor = 1.0e-30;
constexpr double kDiffusionEmergencyGrowthFactor = 10.0;
constexpr double kClosureChiMin = 1.0 / 3.0;
constexpr double kClosureChiMax = 1.0;
constexpr std::uint64_t kStepLocalIdBits = 40ULL;
constexpr std::uint64_t kStepLocalIdMask = (1ULL << kStepLocalIdBits) - 1ULL;
constexpr std::uint64_t kCensusResidualLocalIdBase = 1ULL << 38;
constexpr std::uint64_t kCensusResidualReservedRange = 1ULL << 38;
constexpr std::uint64_t kDiffusionExitLocalIdBase = 1ULL << 39;
constexpr std::uint64_t kDiffusionReservedHalfRange = 1ULL << 38;
constexpr std::uint64_t kDiffusionInterfaceLocalIdBase =
    kDiffusionExitLocalIdBase + kDiffusionReservedHalfRange;

// flip_negative_signs kernel is in source.cu, called via source.cuh wrapper

[[nodiscard]] bool equal_double_bits(const std::vector<double>& lhs,
                                     const std::vector<double>& rhs) {
  return lhs.size() == rhs.size() &&
         (lhs.empty() ||
          std::memcmp(lhs.data(), rhs.data(), sizeof(double) * lhs.size()) == 0);
}

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

double finite_or_zero_host(double value);

void copy_u8_to_device(const std::vector<std::uint8_t>& host,
                       parallel::DeviceArray& device,
                       const char* message) {
  const std::size_t bytes = sizeof(std::uint8_t) * host.size();
  device.resize(bytes);
  if (bytes > 0U) {
    cuda_check(cudaMemcpy(device.ptr, host.data(), bytes, cudaMemcpyHostToDevice),
               message);
  }
}

std::uint64_t compute_reserved_source_gid_base(const std::uint64_t step_base_gid,
                                               const std::int64_t n_new_local,
                                               const int rank,
                                               const int n_ranks,
                                               const std::uint64_t local_base,
                                               const std::uint64_t range_size,
                                               const char* label) {
  TENRYU_ASSERT(n_new_local >= 0,
                "reserved source gid base requires non-negative n_new_local");
  const int ranks = std::max(n_ranks, 1);
  TENRYU_ASSERT(rank >= 0 && rank < ranks,
                "reserved source gid base requires valid rank");
  const std::uint64_t step_prefix = step_base_gid & ~kStepLocalIdMask;
  const std::uint64_t rank_stride = range_size / static_cast<std::uint64_t>(ranks);
  TENRYU_ASSERT(rank_stride > 0ULL, "reserved source gid base has too many ranks");
  const std::uint64_t rank_offset =
      static_cast<std::uint64_t>(rank) * rank_stride;
  const std::uint64_t local_start = local_base + rank_offset;
  TENRYU_ASSERT(static_cast<std::uint64_t>(n_new_local) <= rank_stride,
                label);
  return step_prefix + local_start;
}

std::uint64_t compute_diffusion_exit_gid_base(const std::uint64_t step_base_gid,
                                              const std::int64_t n_new_local,
                                              const int rank,
                                              const int n_ranks) {
  return compute_reserved_source_gid_base(
      step_base_gid,
      n_new_local,
      rank,
      n_ranks,
      kDiffusionExitLocalIdBase,
      kDiffusionReservedHalfRange,
      "diffusion exit gid base local count exceeds reserved range");
}

std::uint64_t compute_diffusion_interface_gid_base(const std::uint64_t step_base_gid,
                                                   const std::int64_t n_new_local_max,
                                                   const int rank,
                                                   const int n_ranks) {
  return compute_reserved_source_gid_base(
      step_base_gid,
      n_new_local_max,
      rank,
      n_ranks,
      kDiffusionInterfaceLocalIdBase,
      kDiffusionReservedHalfRange,
      "diffusion interface gid base local count exceeds reserved range");
}

std::uint64_t compute_census_residual_gid_base(const std::uint64_t step_base_gid,
                                               const std::int64_t n_new_local,
                                               const int rank,
                                               const int n_ranks) {
  return compute_reserved_source_gid_base(
      step_base_gid,
      n_new_local,
      rank,
      n_ranks,
      kCensusResidualLocalIdBase,
      kCensusResidualReservedRange,
      "census residual gid base local count exceeds reserved range");
}

template <typename T>
void copy_pool_field_to_host(std::vector<T>& host,
                             const T* device,
                             const int n,
                             const char* message) {
  host.assign(static_cast<std::size_t>(std::max(n, 0)), T{});
  if (n <= 0) {
    return;
  }
  TENRYU_ASSERT(device != nullptr, message);
  cuda_check(cudaMemcpy(host.data(),
                        device,
                        sizeof(T) * host.size(),
                        cudaMemcpyDeviceToHost),
             message);
}

template <typename T>
void upload_pool_field_from_host(T* device,
                                 const std::vector<T>& host,
                                 const char* message) {
  if (host.empty()) {
    return;
  }
  TENRYU_ASSERT(device != nullptr, message);
  cuda_check(cudaMemcpy(device,
                        host.data(),
                        sizeof(T) * host.size(),
                        cudaMemcpyHostToDevice),
             message);
}

struct DifferenceCensusSnapshot {
  int n = 0;
  std::vector<double> pos_r;
  std::vector<double> pos_z;
  std::vector<double> dir_r;
  std::vector<double> dir_z;
  std::vector<double> dir_phi;
  std::vector<double> energy;
  std::vector<double> weight;
  std::vector<double> time_remain;
  std::vector<double> birth_energy;
  std::vector<std::int8_t> sign;
  std::vector<std::uint64_t> global_id;
  std::vector<std::uint32_t> rng_counter;
  std::vector<std::int32_t> cell_id;
  std::vector<std::uint16_t> group_id;
  std::vector<std::uint8_t> mode;
  std::vector<std::uint8_t> alive;
};

struct DifferenceBinData {
  std::vector<double> signed_U;
  std::vector<double> abs_U;
  std::vector<int> head;
  std::vector<int> next;
  std::vector<int> count;
  int valid_particles = 0;
};

struct DifferenceResidualOutput {
  std::vector<double> pos_r;
  std::vector<double> pos_z;
  std::vector<double> dir_r;
  std::vector<double> dir_z;
  std::vector<double> dir_phi;
  std::vector<double> energy;
  std::vector<double> weight;
  std::vector<double> time_remain;
  std::vector<double> birth_energy;
  std::vector<std::int8_t> sign;
  std::vector<std::uint64_t> global_id;
  std::vector<std::uint32_t> rng_counter;
  std::vector<std::int32_t> cell_id;
  std::vector<std::uint16_t> group_id;
  std::vector<std::uint8_t> mode;
  std::vector<std::int32_t> empty_cell_id;
  std::vector<std::uint16_t> empty_group_id;
  std::vector<double> empty_energy;
  std::vector<std::int8_t> empty_sign;
};

struct DifferenceResidualizationStats {
  int scaled_bins = 0;
  int rebuilt_bins = 0;
  int killed_bins = 0;
  int empty_created = 0;
  int n_before = 0;
  int n_after = 0;
};

std::int8_t normalized_particle_sign(const std::int8_t sign) {
  return (sign < 0) ? static_cast<std::int8_t>(-1) : static_cast<std::int8_t>(1);
}

double finite_nonnegative_energy(const double value) {
  return (std::isfinite(value) && value > 0.0) ? value : 0.0;
}

DifferenceCensusSnapshot copy_difference_census_snapshot_host(const PhotonPool& pool,
                                                              const int n_alive) {
  DifferenceCensusSnapshot snapshot{};
  snapshot.n = std::max(n_alive, 0);
  if (snapshot.n <= 0) {
    return snapshot;
  }
  TENRYU_ASSERT(snapshot.n <= pool.capacity,
                "copy_difference_census_snapshot_host n_alive exceeds capacity");
  copy_pool_field_to_host(snapshot.pos_r,
                          pool.pos_r,
                          snapshot.n,
                          "difference census copy pos_r failed");
  copy_pool_field_to_host(snapshot.pos_z,
                          pool.pos_z,
                          snapshot.n,
                          "difference census copy pos_z failed");
  copy_pool_field_to_host(snapshot.dir_r,
                          pool.dir_r,
                          snapshot.n,
                          "difference census copy dir_r failed");
  copy_pool_field_to_host(snapshot.dir_z,
                          pool.dir_z,
                          snapshot.n,
                          "difference census copy dir_z failed");
  copy_pool_field_to_host(snapshot.dir_phi,
                          pool.dir_phi,
                          snapshot.n,
                          "difference census copy dir_phi failed");
  copy_pool_field_to_host(snapshot.energy,
                          pool.energy,
                          snapshot.n,
                          "difference census copy energy failed");
  copy_pool_field_to_host(snapshot.weight,
                          pool.weight,
                          snapshot.n,
                          "difference census copy weight failed");
  copy_pool_field_to_host(snapshot.time_remain,
                          pool.time_remain,
                          snapshot.n,
                          "difference census copy time_remain failed");
  copy_pool_field_to_host(snapshot.birth_energy,
                          pool.birth_energy,
                          snapshot.n,
                          "difference census copy birth_energy failed");
  copy_pool_field_to_host(snapshot.sign,
                          pool.sign,
                          snapshot.n,
                          "difference census copy sign failed");
  copy_pool_field_to_host(snapshot.global_id,
                          pool.global_id,
                          snapshot.n,
                          "difference census copy global_id failed");
  copy_pool_field_to_host(snapshot.rng_counter,
                          pool.rng_counter,
                          snapshot.n,
                          "difference census copy rng_counter failed");
  copy_pool_field_to_host(snapshot.cell_id,
                          pool.cell_id,
                          snapshot.n,
                          "difference census copy cell_id failed");
  copy_pool_field_to_host(snapshot.group_id,
                          pool.group_id,
                          snapshot.n,
                          "difference census copy group_id failed");
  copy_pool_field_to_host(snapshot.mode,
                          pool.mode,
                          snapshot.n,
                          "difference census copy mode failed");
  copy_pool_field_to_host(snapshot.alive,
                          pool.alive,
                          snapshot.n,
                          "difference census copy alive failed");
  return snapshot;
}

DifferenceBinData build_difference_bin_data(const DifferenceCensusSnapshot& snapshot,
                                            const int n_cells,
                                            const int n_groups) {
  const std::size_t n_bins =
      static_cast<std::size_t>(std::max(n_cells, 0)) *
      static_cast<std::size_t>(std::max(n_groups, 0));
  DifferenceBinData bins{};
  bins.signed_U.assign(n_bins, 0.0);
  bins.abs_U.assign(n_bins, 0.0);
  bins.head.assign(n_bins, -1);
  bins.next.assign(static_cast<std::size_t>(snapshot.n), -1);
  bins.count.assign(n_bins, 0);
  if (n_cells <= 0 || n_groups <= 0) {
    return bins;
  }

  for (int p = 0; p < snapshot.n; ++p) {
    const std::size_t p_us = static_cast<std::size_t>(p);
    if (snapshot.alive[p_us] != kAlive) {
      continue;
    }
    const int cell = snapshot.cell_id[p_us];
    const int group = static_cast<int>(snapshot.group_id[p_us]);
    if (cell < 0 || cell >= n_cells || group < 0 || group >= n_groups) {
      continue;
    }
    const double E = finite_nonnegative_energy(snapshot.energy[p_us]);
    const std::int8_t s = normalized_particle_sign(snapshot.sign[p_us]);
    const std::size_t key =
        static_cast<std::size_t>(cell) * static_cast<std::size_t>(n_groups) +
        static_cast<std::size_t>(group);
    bins.signed_U[key] += static_cast<double>(s) * E;
    bins.abs_U[key] += E;
    bins.next[p_us] = bins.head[key];
    bins.head[key] = p;
    ++bins.count[key];
    ++bins.valid_particles;
  }
  return bins;
}

std::vector<double> make_reference_U_from_density(const std::vector<double>& E_ref,
                                                  const std::vector<double>& vol,
                                                  const int n_cells,
                                                  const int n_groups) {
  const std::size_t n_bins =
      static_cast<std::size_t>(std::max(n_cells, 0)) *
      static_cast<std::size_t>(std::max(n_groups, 0));
  TENRYU_ASSERT(E_ref.size() == n_bins,
                "make_reference_U_from_density E_ref size mismatch");
  TENRYU_ASSERT(vol.size() == static_cast<std::size_t>(std::max(n_cells, 0)),
                "make_reference_U_from_density volume size mismatch");
  std::vector<double> U_ref(n_bins, 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const double V = std::max(vol[static_cast<std::size_t>(c)], 0.0);
    const std::size_t base =
        static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups);
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx = base + static_cast<std::size_t>(g);
      U_ref[idx] = finite_or_zero_host(E_ref[idx]) * V;
    }
  }
  return U_ref;
}

std::vector<double> build_difference_physical_U_old(
    const core::State& state,
    const std::vector<double>& host_vol,
    const std::vector<double>& previous_reference_U,
    const DifferenceBinData& bins,
    const int n_cells,
    const int n_groups) {
  const std::size_t n_bins =
      static_cast<std::size_t>(std::max(n_cells, 0)) *
      static_cast<std::size_t>(std::max(n_groups, 0));
  TENRYU_ASSERT(host_vol.size() == static_cast<std::size_t>(std::max(n_cells, 0)),
                "build_difference_physical_U_old volume size mismatch");
  TENRYU_ASSERT(bins.signed_U.size() == n_bins,
                "build_difference_physical_U_old bin size mismatch");
  std::vector<double> U_phys(n_bins, 0.0);
  const bool have_reference = previous_reference_U.size() == n_bins;
  if (have_reference || bins.valid_particles > 0) {
    for (std::size_t i = 0; i < n_bins; ++i) {
      U_phys[i] = bins.signed_U[i] +
                  (have_reference ? finite_or_zero_host(previous_reference_U[i]) : 0.0);
    }
    return U_phys;
  }

  if (state.rad_E.size() == n_bins && n_bins > 0U) {
    std::vector<double> host_rad_E(n_bins, 0.0);
    state.rad_E.copy_to_host(host_rad_E.data());
    for (int c = 0; c < n_cells; ++c) {
      const double V = std::max(host_vol[static_cast<std::size_t>(c)], 0.0);
      const std::size_t base =
          static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups);
      for (int g = 0; g < n_groups; ++g) {
        const std::size_t idx = base + static_cast<std::size_t>(g);
        U_phys[idx] = finite_or_zero_host(host_rad_E[idx]) * V;
      }
    }
  }
  return U_phys;
}

std::vector<double> density_from_U(const std::vector<double>& U,
                                   const std::vector<double>& vol,
                                   const int n_cells,
                                   const int n_groups) {
  const std::size_t n_bins =
      static_cast<std::size_t>(std::max(n_cells, 0)) *
      static_cast<std::size_t>(std::max(n_groups, 0));
  TENRYU_ASSERT(U.size() == n_bins, "density_from_U U size mismatch");
  TENRYU_ASSERT(vol.size() == static_cast<std::size_t>(std::max(n_cells, 0)),
                "density_from_U volume size mismatch");
  std::vector<double> E(n_bins, 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const double V = std::max(vol[static_cast<std::size_t>(c)], 0.0);
    const std::size_t base =
        static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups);
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx = base + static_cast<std::size_t>(g);
      E[idx] = (V > 0.0) ? (finite_or_zero_host(U[idx]) / V) : 0.0;
    }
  }
  return E;
}

void reserve_difference_output(DifferenceResidualOutput& out, const std::size_t n) {
  out.pos_r.reserve(n);
  out.pos_z.reserve(n);
  out.dir_r.reserve(n);
  out.dir_z.reserve(n);
  out.dir_phi.reserve(n);
  out.energy.reserve(n);
  out.weight.reserve(n);
  out.time_remain.reserve(n);
  out.birth_energy.reserve(n);
  out.sign.reserve(n);
  out.global_id.reserve(n);
  out.rng_counter.reserve(n);
  out.cell_id.reserve(n);
  out.group_id.reserve(n);
  out.mode.reserve(n);
}

void append_difference_particle(DifferenceResidualOutput& out,
                                const DifferenceCensusSnapshot& snapshot,
                                const int p,
                                const double energy,
                                const std::int8_t sign,
                                const double birth_energy,
                                const double weight) {
  const double E = finite_nonnegative_energy(energy);
  if (!(E > 0.0)) {
    return;
  }
  const std::size_t p_us = static_cast<std::size_t>(p);
  out.pos_r.push_back(snapshot.pos_r[p_us]);
  out.pos_z.push_back(snapshot.pos_z[p_us]);
  out.dir_r.push_back(snapshot.dir_r[p_us]);
  out.dir_z.push_back(snapshot.dir_z[p_us]);
  out.dir_phi.push_back(snapshot.dir_phi[p_us]);
  out.energy.push_back(E);
  const double birth = (std::isfinite(birth_energy) && birth_energy > 0.0)
                           ? birth_energy
                           : E;
  out.birth_energy.push_back(birth);
  (void)weight;
  out.weight.push_back((birth > 0.0) ? (E / birth) : 0.0);
  out.time_remain.push_back(std::max(finite_or_zero_host(snapshot.time_remain[p_us]), 0.0));
  out.sign.push_back(normalized_particle_sign(sign));
  out.global_id.push_back(snapshot.global_id[p_us]);
  out.rng_counter.push_back(snapshot.rng_counter[p_us]);
  out.cell_id.push_back(snapshot.cell_id[p_us]);
  out.group_id.push_back(snapshot.group_id[p_us]);
  out.mode.push_back(snapshot.mode[p_us]);
}

void write_difference_output_to_pool(PhotonPool& pool,
                                     const DifferenceResidualOutput& out,
                                     const int max_pool_size) {
  const int n_out = static_cast<int>(out.energy.size());
  if (n_out <= 0) {
    pool.n_alive = 0;
    pool.n_census = 0;
    return;
  }
  pool.reserve(n_out, max_pool_size);
  upload_pool_field_from_host(pool.pos_r, out.pos_r, "difference census upload pos_r failed");
  upload_pool_field_from_host(pool.pos_z, out.pos_z, "difference census upload pos_z failed");
  upload_pool_field_from_host(pool.dir_r, out.dir_r, "difference census upload dir_r failed");
  upload_pool_field_from_host(pool.dir_z, out.dir_z, "difference census upload dir_z failed");
  upload_pool_field_from_host(pool.dir_phi,
                              out.dir_phi,
                              "difference census upload dir_phi failed");
  upload_pool_field_from_host(pool.energy,
                              out.energy,
                              "difference census upload energy failed");
  upload_pool_field_from_host(pool.weight,
                              out.weight,
                              "difference census upload weight failed");
  upload_pool_field_from_host(pool.time_remain,
                              out.time_remain,
                              "difference census upload time_remain failed");
  upload_pool_field_from_host(pool.birth_energy,
                              out.birth_energy,
                              "difference census upload birth_energy failed");
  upload_pool_field_from_host(pool.sign, out.sign, "difference census upload sign failed");
  upload_pool_field_from_host(pool.global_id,
                              out.global_id,
                              "difference census upload global_id failed");
  upload_pool_field_from_host(pool.rng_counter,
                              out.rng_counter,
                              "difference census upload rng_counter failed");
  upload_pool_field_from_host(pool.cell_id,
                              out.cell_id,
                              "difference census upload cell_id failed");
  upload_pool_field_from_host(pool.group_id,
                              out.group_id,
                              "difference census upload group_id failed");
  upload_pool_field_from_host(pool.mode, out.mode, "difference census upload mode failed");
  std::vector<std::uint8_t> alive(static_cast<std::size_t>(n_out), kAlive);
  upload_pool_field_from_host(pool.alive, alive, "difference census upload alive failed");
  pool.n_alive = n_out;
  pool.n_census = n_out;
}

DifferenceResidualizationStats residualize_census_against_reference(
    core::State& state,
    PhotonPool& pool,
    const int max_pool_size,
    const double dt,
    const std::uint64_t step_number,
    const std::uint64_t user_seed,
    const std::uint64_t empty_gid_base,
    const DifferenceCensusSnapshot& snapshot,
    const DifferenceBinData& bins,
    const std::vector<double>& target_U,
    const int n_cells,
    const int n_groups) {
  const std::size_t n_bins =
      static_cast<std::size_t>(std::max(n_cells, 0)) *
      static_cast<std::size_t>(std::max(n_groups, 0));
  TENRYU_ASSERT(target_U.size() == n_bins,
                "residualize_census_against_reference target size mismatch");
  TENRYU_ASSERT(bins.signed_U.size() == n_bins,
                "residualize_census_against_reference bin size mismatch");

  DifferenceResidualizationStats stats{};
  stats.n_before = snapshot.n;
  DifferenceResidualOutput out;
  reserve_difference_output(out, static_cast<std::size_t>(snapshot.n));

  constexpr double kConditionFloor = 1.0e-300;
  constexpr double kConditionRel = 1.0e-12;
  for (std::size_t b = 0; b < n_bins; ++b) {
    double target = finite_or_zero_host(target_U[b]);
    const int n_bin = bins.count[b];
    if (n_bin <= 0) {
      if (target != 0.0) {
        const int cell = static_cast<int>(b / static_cast<std::size_t>(n_groups));
        const int group = static_cast<int>(b % static_cast<std::size_t>(n_groups));
        out.empty_cell_id.push_back(static_cast<std::int32_t>(cell));
        out.empty_group_id.push_back(static_cast<std::uint16_t>(group));
        out.empty_energy.push_back(std::abs(target));
        out.empty_sign.push_back((target < 0.0) ? static_cast<std::int8_t>(-1)
                                                : static_cast<std::int8_t>(1));
        ++stats.empty_created;
      }
      continue;
    }

    if (target == 0.0) {
      ++stats.killed_bins;
      continue;
    }

    const double R_old = finite_or_zero_host(bins.signed_U[b]);
    const double scale =
        std::max({std::abs(target), std::abs(R_old), bins.abs_U[b], kConditionFloor});
    const bool well_conditioned =
        std::abs(R_old) > kConditionRel * scale;
    if (well_conditioned) {
      const double ratio = target / R_old;
      if (std::isfinite(ratio)) {
        const double abs_ratio = std::abs(ratio);
        for (int p = bins.head[b]; p >= 0; p = bins.next[static_cast<std::size_t>(p)]) {
          const std::size_t p_us = static_cast<std::size_t>(p);
          const double E_new = finite_nonnegative_energy(snapshot.energy[p_us]) * abs_ratio;
          const double birth_new =
              finite_nonnegative_energy(snapshot.birth_energy[p_us]) * abs_ratio;
          const std::int8_t old_sign = normalized_particle_sign(snapshot.sign[p_us]);
          const std::int8_t new_sign =
              (ratio < 0.0) ? static_cast<std::int8_t>(-old_sign) : old_sign;
          const double weight_new =
              (birth_new > 0.0) ? (E_new / birth_new) : finite_nonnegative_energy(snapshot.weight[p_us]);
          append_difference_particle(out,
                                     snapshot,
                                     p,
                                     E_new,
                                     new_sign,
                                     birth_new,
                                     weight_new);
        }
        ++stats.scaled_bins;
        continue;
      }
    }

    const int template_p = bins.head[b];
    append_difference_particle(out,
                               snapshot,
                               template_p,
                               std::abs(target),
                               (target < 0.0) ? static_cast<std::int8_t>(-1)
                                              : static_cast<std::int8_t>(1),
                               std::abs(target),
                               1.0);
    ++stats.rebuilt_bins;
  }

  write_difference_output_to_pool(pool, out, max_pool_size);
  if (!out.empty_energy.empty()) {
    const SourceStats empty_stats =
        IMCSource::append_census_residual_1d(state,
                                             pool,
                                             max_pool_size,
                                             dt,
                                             step_number,
                                             user_seed,
                                             empty_gid_base,
                                             out.empty_cell_id,
                                             out.empty_group_id,
                                             out.empty_energy,
                                             out.empty_sign);
    TENRYU_ASSERT(empty_stats.pool_overflow == 0,
                  "difference census empty-bin particle creation overflowed pool");
  }
  stats.n_after = pool.n_alive;
  pool.n_census = pool.n_alive;
  return stats;
}

void append_diffusion_exit_particles(PhotonPool& pool,
                                     const int start,
                                     const std::vector<std::int32_t>& cell_id,
                                     const std::vector<std::uint16_t>& group_id,
                                     const std::vector<double>& energy,
                                     const double dt,
                                     const std::uint64_t gid_base) {
  const int n_new = static_cast<int>(energy.size());
  TENRYU_ASSERT(static_cast<int>(cell_id.size()) == n_new,
                "append_diffusion_exit_particles cell size mismatch");
  TENRYU_ASSERT(static_cast<int>(group_id.size()) == n_new,
                "append_diffusion_exit_particles group size mismatch");
  if (n_new <= 0) {
    return;
  }

  std::vector<double> weight(static_cast<std::size_t>(n_new), 1.0);
  std::vector<double> time_remain(static_cast<std::size_t>(n_new), dt);
  std::vector<double> birth_energy = energy;
  std::vector<std::int8_t> sign(static_cast<std::size_t>(n_new), 1);
  std::vector<std::uint64_t> global_id(static_cast<std::size_t>(n_new), 0);
  std::vector<std::uint32_t> rng_counter(static_cast<std::size_t>(n_new), 0U);
  std::vector<std::uint8_t> mode(static_cast<std::size_t>(n_new), kModeIMC);
  std::vector<std::uint8_t> alive(static_cast<std::size_t>(n_new), kAlive);

  for (int i = 0; i < n_new; ++i) {
    global_id[static_cast<std::size_t>(i)] = gid_base + static_cast<std::uint64_t>(i);
  }

  const std::size_t bytes_d = sizeof(double) * static_cast<std::size_t>(n_new);
  const std::size_t bytes_i32 = sizeof(std::int32_t) * static_cast<std::size_t>(n_new);
  const std::size_t bytes_u16 = sizeof(std::uint16_t) * static_cast<std::size_t>(n_new);
  const std::size_t bytes_u64 = sizeof(std::uint64_t) * static_cast<std::size_t>(n_new);
  const std::size_t bytes_u32 = sizeof(std::uint32_t) * static_cast<std::size_t>(n_new);
  const std::size_t bytes_i8 = sizeof(std::int8_t) * static_cast<std::size_t>(n_new);
  const std::size_t bytes_u8 = sizeof(std::uint8_t) * static_cast<std::size_t>(n_new);

  cuda_check(cudaMemcpy(pool.energy + start, energy.data(), bytes_d, cudaMemcpyHostToDevice),
             "append_diffusion_exit_particles copy energy failed");
  cuda_check(cudaMemcpy(pool.weight + start, weight.data(), bytes_d, cudaMemcpyHostToDevice),
             "append_diffusion_exit_particles copy weight failed");
  cuda_check(cudaMemcpy(pool.time_remain + start,
                        time_remain.data(),
                        bytes_d,
                        cudaMemcpyHostToDevice),
             "append_diffusion_exit_particles copy time_remain failed");
  cuda_check(cudaMemcpy(pool.birth_energy + start,
                        birth_energy.data(),
                        bytes_d,
                        cudaMemcpyHostToDevice),
             "append_diffusion_exit_particles copy birth_energy failed");
  cuda_check(cudaMemcpy(pool.sign + start, sign.data(), bytes_i8, cudaMemcpyHostToDevice),
             "append_diffusion_exit_particles copy sign failed");
  cuda_check(cudaMemcpy(pool.cell_id + start, cell_id.data(), bytes_i32, cudaMemcpyHostToDevice),
             "append_diffusion_exit_particles copy cell_id failed");
  cuda_check(cudaMemcpy(pool.group_id + start, group_id.data(), bytes_u16, cudaMemcpyHostToDevice),
             "append_diffusion_exit_particles copy group_id failed");
  cuda_check(cudaMemcpy(pool.global_id + start,
                        global_id.data(),
                        bytes_u64,
                        cudaMemcpyHostToDevice),
             "append_diffusion_exit_particles copy global_id failed");
  cuda_check(cudaMemcpy(pool.rng_counter + start,
                        rng_counter.data(),
                        bytes_u32,
                        cudaMemcpyHostToDevice),
             "append_diffusion_exit_particles copy rng_counter failed");
  cuda_check(cudaMemcpy(pool.mode + start, mode.data(), bytes_u8, cudaMemcpyHostToDevice),
             "append_diffusion_exit_particles copy mode failed");
  cuda_check(cudaMemcpy(pool.alive + start, alive.data(), bytes_u8, cudaMemcpyHostToDevice),
             "append_diffusion_exit_particles copy alive failed");
}

DiffusionConversionStats spawn_exiting_diffusion_particles_1d(
    core::State& state,
    const core::Config& cfg,
    PhotonPool& pool,
    const int max_pool_size,
    const double dt,
    const std::uint64_t step_base_gid,
    const std::uint64_t gid_local_offset,
    const std::vector<std::uint8_t>& diff_cell,
    const std::vector<std::uint8_t>& diff_cell_prev,
    const std::vector<double>& host_diff_E,
    const std::vector<double>& host_vol,
    const int n_cells,
    const int n_groups,
    const int rank,
    const int n_ranks) {
  TENRYU_ASSERT(static_cast<int>(diff_cell.size()) == n_cells,
                "spawn_exiting_diffusion_particles_1d diff_cell size mismatch");
  TENRYU_ASSERT(static_cast<int>(diff_cell_prev.size()) == n_cells,
                "spawn_exiting_diffusion_particles_1d diff_cell_prev size mismatch");
  TENRYU_ASSERT(static_cast<int>(host_vol.size()) == n_cells,
                "spawn_exiting_diffusion_particles_1d vol size mismatch");
  TENRYU_ASSERT(static_cast<int>(host_diff_E.size()) == n_cells * n_groups,
                "spawn_exiting_diffusion_particles_1d diff_E size mismatch");
  TENRYU_ASSERT(state.mesh.dim == 1,
                "spawn_exiting_diffusion_particles_1d requires 1D mesh");

  const int n_per_group =
      std::max(1, cfg.radiation.diffusion.exit_particles_per_cell_group);
  std::vector<std::int32_t> out_cell;
  std::vector<std::uint16_t> out_group;
  std::vector<double> out_energy;
  double total_energy = 0.0;

  for (int c = 0; c < n_cells; ++c) {
    if (diff_cell[c] != 0U || diff_cell_prev[c] == 0U) {
      continue;
    }
    const double V = host_vol[static_cast<std::size_t>(c)];
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t key =
          static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups) +
          static_cast<std::size_t>(g);
      const double E_density = host_diff_E[key];
      if (!(E_density > 0.0)) {
        continue;
      }
      TENRYU_ASSERT(V > 0.0,
                    "diffusion exit conversion requires positive cell volume");
      const double E_total = E_density * V;
      if (!(E_total > 0.0)) {
        continue;
      }
      const double E_per_particle = E_total / static_cast<double>(n_per_group);
      for (int k = 0; k < n_per_group; ++k) {
        out_cell.push_back(c);
        out_group.push_back(static_cast<std::uint16_t>(g));
        if (k + 1 == n_per_group) {
          out_energy.push_back(E_total - E_per_particle * static_cast<double>(n_per_group - 1));
        } else {
          out_energy.push_back(E_per_particle);
        }
      }
      total_energy += E_total;
    }
  }

  const auto n_new_i64 = static_cast<std::int64_t>(out_energy.size());
  TENRYU_ASSERT(gid_local_offset <=
                    static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max() - n_new_i64),
                "diffusion exit gid offset exceeds int64 range");
  const std::uint64_t gid_base =
      compute_diffusion_exit_gid_base(
          step_base_gid,
          n_new_i64 + static_cast<std::int64_t>(gid_local_offset),
          rank,
          n_ranks) +
      gid_local_offset;
  TENRYU_ASSERT(n_new_i64 <= static_cast<std::int64_t>(std::numeric_limits<int>::max()),
                "diffusion exit particle count exceeds int range");
  const int n_new = static_cast<int>(n_new_i64);
  if (n_new == 0) {
    return {};
  }

  const std::int64_t required_i64 =
      static_cast<std::int64_t>(pool.n_alive) + n_new_i64;
  TENRYU_ASSERT(required_i64 <= static_cast<std::int64_t>(max_pool_size),
                "diffusion exit particle creation exceeds max_pool_size");
  const int required_capacity = static_cast<int>(required_i64);
  pool.reserve(required_capacity, max_pool_size);
  const int start = pool.n_alive;
  append_diffusion_exit_particles(pool, start, out_cell, out_group, out_energy, dt, gid_base);
  fill_diffusion_exit_phase_space_1d_cuda(pool,
                                          start,
                                          n_new,
                                          state.x_r.data(),
                                          n_cells,
                                          cfg.main.seed,
                                          static_cast<std::uint64_t>(state.step),
                                          dt);
  pool.n_alive += n_new;

  DiffusionConversionStats stats;
  stats.n_particles = static_cast<std::uint64_t>(n_new);
  stats.energy = total_energy;
  return stats;
}

double pgrw_survival_probability(const double theta) {
  if (!(theta > 0.0)) {
    return 1.0;
  }

  long double sum = 0.0L;
  const long double theta_ld = static_cast<long double>(theta);
  const long double pi2 = static_cast<long double>(kPi * kPi);
  for (int n = 1; n <= kPgrwSeriesTerms; ++n) {
    const long double n_ld = static_cast<long double>(n);
    const long double term =
        std::exp(-n_ld * n_ld * pi2 * theta_ld);
    sum += ((n & 1) != 0 ? 1.0L : -1.0L) * term;
  }
  const double survival = static_cast<double>(2.0L * sum);
  return std::clamp(survival, 0.0, 1.0);
}

double pgrw_leak_cdf(const double theta) {
  if (!(theta > 0.0)) {
    return 0.0;
  }
  return std::clamp(1.0 - pgrw_survival_probability(theta), 0.0, 1.0);
}

double pgrw_position_cdf(const double rho_frac, const double theta) {
  if (!(rho_frac > 0.0)) {
    return 0.0;
  }
  if (rho_frac >= 1.0) {
    return 1.0;
  }
  if (!(theta > 0.0)) {
    return 0.0;
  }

  long double numer = 0.0L;
  long double denom = 0.0L;
  const long double x = static_cast<long double>(rho_frac);
  const long double theta_ld = static_cast<long double>(theta);
  const long double pi_ld = static_cast<long double>(kPi);
  const long double pi2_ld = pi_ld * pi_ld;
  for (int n = 1; n <= kPgrwSeriesTerms; ++n) {
    const long double n_ld = static_cast<long double>(n);
    const long double decay = std::exp(-n_ld * n_ld * pi2_ld * theta_ld);
    const long double sign = ((n & 1) != 0) ? 1.0L : -1.0L;
    numer += sign * decay *
             (std::sin(n_ld * pi_ld * x) / (n_ld * pi2_ld) -
              x * std::cos(n_ld * pi_ld * x) / pi_ld);
    denom += decay / pi_ld;
  }

  if (!(denom > 0.0L) || !std::isfinite(static_cast<double>(denom))) {
    return 1.0;
  }
  const double cdf = static_cast<double>(numer / denom);
  return std::clamp(cdf, 0.0, 1.0);
}

void enforce_monotone_unit_cdf(std::vector<double>& values) {
  double prev = 0.0;
  for (double& value : values) {
    value = std::clamp(value, prev, 1.0);
    prev = value;
  }
  if (!values.empty()) {
    values.back() = 1.0;
  }
}

double clamped_temperature_for_t4_host(double temperature_eV);

void write_uniform_unit_cdf_row(double* const row, const int n_groups) {
  TENRYU_ASSERT(row != nullptr, "write_uniform_unit_cdf_row requires row");
  TENRYU_ASSERT(n_groups > 0, "write_uniform_unit_cdf_row requires positive n_groups");
  for (int g = 0; g < n_groups; ++g) {
    row[g] = static_cast<double>(g + 1) / static_cast<double>(n_groups);
  }
}

template <typename T>
void assign_scatter_bias_cdf_if_supported(T& in, const double* const scatter_bias_cdf) {
  if constexpr (requires { in.scatter_bias_cdf; }) {
    in.scatter_bias_cdf = scatter_bias_cdf;
  }
}

std::vector<double> compute_rosseland_importance(const core::State& state,
                                                 const core::Config& cfg,
                                                 const PlanckTable& planck,
                                                 const std::vector<double>& sigma_t) {
  const int n_cells = static_cast<int>(state.Te.size());
  const int n_groups = std::max(cfg.radiation.groups, 1);
  const std::size_t n_cell_groups =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  TENRYU_ASSERT(sigma_t.size() == n_cell_groups,
                "compute_rosseland_importance sigma_t size mismatch");
  TENRYU_ASSERT(planck.n_groups() == n_groups,
                "compute_rosseland_importance group count mismatch");

  std::vector<double> importance(n_cell_groups, 0.0);
  if (n_cells <= 0 || n_groups <= 0) {
    return importance;
  }

  std::vector<double> host_Te(static_cast<std::size_t>(n_cells), 0.0);
  state.Te.copy_to_host(host_Te.data());

  constexpr double kSigmaFloor = 1.0e-30;
  for (int c = 0; c < n_cells; ++c) {
    if (state.cell_is_void[c] != 0U) {
      continue;
    }

    const std::size_t base =
        static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups);
    const double T =
        clamped_temperature_for_t4_host(fmax(host_Te[static_cast<std::size_t>(c)],
                                             cfg.numerics.floors.Te));
    const double dT = fmax(T * 0.01, 1.0e-3);
    const double T_hi = T + dT;
    const double T_lo = fmax(T - dT, 1.0e-6);
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx = base + static_cast<std::size_t>(g);
      const double b_g_hi = planck.interpolate_b_host(g, T_hi);
      const double b_g_lo = planck.interpolate_b_host(g, T_lo);
      const double dBdT_g = (b_g_hi - b_g_lo) / (2.0 * dT);
      importance[idx] =
          fmax(dBdT_g, 0.0) / fmax(fmax(sigma_t[idx], 0.0), kSigmaFloor);
    }
  }

  return importance;
}

std::vector<double> build_spectral_emission_bias_cdf(const core::State& state,
                                                     const core::Config& cfg,
                                                     const PlanckTable& planck,
                                                     const std::vector<double>& sigma_t) {
  const int n_cells = static_cast<int>(state.Te.size());
  const int n_groups = std::max(cfg.radiation.groups, 1);
  const std::size_t n_cell_groups =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  TENRYU_ASSERT(sigma_t.size() == n_cell_groups,
                "build_spectral_emission_bias_cdf sigma_t size mismatch");
  TENRYU_ASSERT(planck.n_groups() == n_groups,
                "build_spectral_emission_bias_cdf group count mismatch");

  std::vector<double> cdf(n_cell_groups, 0.0);
  if (n_cells <= 0 || n_groups <= 0) {
    return cdf;
  }

  std::vector<double> host_Te(static_cast<std::size_t>(n_cells), 0.0);
  state.Te.copy_to_host(host_Te.data());
  const std::vector<double> importance =
      compute_rosseland_importance(state, cfg, planck, sigma_t);

  const double eta = std::clamp(cfg.radiation.imc.spectral_bias_eta, 0.0, 1.0);
  constexpr double kPdfFloor = 1.0e-300;
  std::vector<double> p_phys(static_cast<std::size_t>(n_groups), 0.0);
  std::vector<double> row(static_cast<std::size_t>(n_groups), 0.0);

  for (int c = 0; c < n_cells; ++c) {
    const std::size_t base =
        static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups);
    double* const row_out = cdf.data() + base;
    if (state.cell_is_void[c] != 0U) {
      write_uniform_unit_cdf_row(row_out, n_groups);
      continue;
    }

    const double T =
        clamped_temperature_for_t4_host(fmax(host_Te[static_cast<std::size_t>(c)],
                                             cfg.numerics.floors.Te));
    double p_sum = 0.0;
    double importance_sum = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx = base + static_cast<std::size_t>(g);
      const double b_g =
          (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b_host(g, T), 0.0);
      p_phys[static_cast<std::size_t>(g)] = b_g;
      p_sum += b_g;
      importance_sum += importance[idx];
    }

    if (!(p_sum > kPdfFloor)) {
      write_uniform_unit_cdf_row(row_out, n_groups);
      continue;
    }

    double running = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      const double p_g_phys = p_phys[static_cast<std::size_t>(g)] / p_sum;
      const double p_g_ross =
          (importance_sum > kPdfFloor)
              ? (importance[base + static_cast<std::size_t>(g)] / importance_sum)
              : p_g_phys;
      const double q_g = (1.0 - eta) * p_g_phys + eta * p_g_ross;
      running += fmax(q_g, 0.0);
      row[static_cast<std::size_t>(g)] = running;
    }

    if (!(running > kPdfFloor)) {
      write_uniform_unit_cdf_row(row_out, n_groups);
      continue;
    }

    for (double& value : row) {
      value /= running;
    }
    enforce_monotone_unit_cdf(row);
    std::copy(row.begin(), row.end(), row_out);
  }

  return cdf;
}

std::int64_t saturating_i64_from_u64(const unsigned long long value) {
  constexpr unsigned long long kI64Max =
      static_cast<unsigned long long>(std::numeric_limits<std::int64_t>::max());
  return (value > kI64Max) ? std::numeric_limits<std::int64_t>::max()
                           : static_cast<std::int64_t>(value);
}

std::int64_t saturating_add_i64(const std::int64_t a, const std::int64_t b) {
  if (b > 0 && a > (std::numeric_limits<std::int64_t>::max() - b)) {
    return std::numeric_limits<std::int64_t>::max();
  }
  if (b < 0 && a < (std::numeric_limits<std::int64_t>::min() - b)) {
    return std::numeric_limits<std::int64_t>::min();
  }
  return a + b;
}

double clamped_temperature_for_t4_host(const double temperature_eV) {
  if (!std::isfinite(temperature_eV) || temperature_eV <= 0.0) {
    return 0.0;
  }
  return std::min(temperature_eV, kMaxTemperatureForT4);
}

double safe_temperature_pow4_host(const double temperature_eV) {
  const double t = clamped_temperature_for_t4_host(temperature_eV);
  return t * t * t * t;
}

double finite_or_zero_host(const double value) {
  return std::isfinite(value) ? value : 0.0;
}

[[nodiscard]] std::size_t cell_group_index_host(const int cell,
                                                const int group,
                                                const int n_groups) {
  return static_cast<std::size_t>(cell) * static_cast<std::size_t>(n_groups) +
         static_cast<std::size_t>(group);
}

[[nodiscard]] double sanitize_holo_closure_chi(const double value) {
  if (!std::isfinite(value) || !(value > 0.0)) {
    return kClosureChiMin;
  }
  return std::clamp(value, kClosureChiMin, kClosureChiMax);
}

void sanitize_holo_closure_chi(std::vector<double>& chi) {
  for (double& value : chi) {
    value = sanitize_holo_closure_chi(value);
  }
}

[[nodiscard]] std::vector<int> compute_holo_closure_material_ids(
    const core::State& state,
    const core::Config& cfg,
    const int n_cells) {
  std::vector<int> material_id(static_cast<std::size_t>(std::max(n_cells, 0)), -1);
  const int first_nonvoid = cfg.materials.first_nonvoid_material_index();
  if (first_nonvoid < 0 || n_cells <= 0) {
    return material_id;
  }

  for (int c = 0; c < n_cells; ++c) {
    const std::size_t c_idx = static_cast<std::size_t>(c);
    material_id[c_idx] =
        (state.cell_is_void[c_idx] == 0U) ? first_nonvoid : -1;
  }

  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  if (n_mat <= 1) {
    return material_id;
  }
  const std::size_t expected =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_mat);
  if (state.volFrac.size() != expected) {
    return material_id;
  }

  std::vector<double> volfrac(expected, 0.0);
  state.volFrac.copy_to_host(volfrac.data());
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t c_idx = static_cast<std::size_t>(c);
    if (material_id[c_idx] < 0) {
      continue;
    }
    const std::size_t base = c_idx * static_cast<std::size_t>(n_mat);
    double best_frac = -1.0;
    int best_mat = first_nonvoid;
    for (int m = 0; m < n_mat; ++m) {
      const std::size_t m_idx = static_cast<std::size_t>(m);
      if (cfg.materials.materials[m_idx].is_void) {
        continue;
      }
      const double frac = volfrac[base + m_idx];
      if (std::isfinite(frac) && frac > best_frac) {
        best_frac = frac;
        best_mat = m;
      }
    }
    material_id[c_idx] = best_mat;
  }
  return material_id;
}

[[nodiscard]] bool same_holo_closure_material(const std::vector<int>& material_id,
                                              const int left,
                                              const int right) {
  const int left_mat = material_id[static_cast<std::size_t>(left)];
  return left_mat >= 0 &&
         left_mat == material_id[static_cast<std::size_t>(right)];
}

void spatial_smooth_holo_qd_chi(std::vector<double>& chi,
                                const std::vector<int>& material_id,
                                const int n_cells,
                                const int n_groups,
                                const int passes,
                                const double alpha) {
  const double a = std::clamp(alpha, 0.0, 1.0);
  if (n_cells < 2 || n_groups <= 0 || passes <= 0 || !(a > 0.0)) {
    return;
  }

  std::vector<double> next = chi;
  for (int pass = 0; pass < passes; ++pass) {
    for (int c = 0; c < n_cells; ++c) {
      const bool has_left =
          c > 0 && same_holo_closure_material(material_id, c - 1, c);
      const bool has_right =
          c + 1 < n_cells && same_holo_closure_material(material_id, c, c + 1);
      for (int g = 0; g < n_groups; ++g) {
        const std::size_t key = cell_group_index_host(c, g, n_groups);
        const double center = chi[key];
        const double left =
            has_left ? chi[cell_group_index_host(c - 1, g, n_groups)] : center;
        const double right =
            has_right ? chi[cell_group_index_host(c + 1, g, n_groups)] : center;
        next[key] = sanitize_holo_closure_chi(
            (1.0 - a) * center + 0.5 * a * (left + right));
      }
    }
    chi.swap(next);
  }
}

void apply_holo_qd_closure_history(core::State& state,
                                   const core::Config& cfg,
                                   const std::size_t n_total,
                                   const bool update_history,
                                   std::vector<double>& chi_host) {
  if (state.holo_chi_filtered.size() != n_total) {
    return;
  }

  std::vector<double> old_chi(n_total, 0.0);
  state.holo_chi_filtered.copy_to_host(old_chi.data());
  std::vector<std::uint8_t> old_valid(n_total, 0U);
  bool any_old_valid = false;
  for (std::size_t i = 0; i < n_total; ++i) {
    if (std::isfinite(old_chi[i]) && old_chi[i] > 0.0) {
      old_valid[i] = 1U;
      any_old_valid = true;
    }
  }
  if (!any_old_valid) {
    return;
  }

  if (!update_history) {
    for (std::size_t i = 0; i < n_total; ++i) {
      if (old_valid[i] != 0U) {
        chi_host[i] = sanitize_holo_closure_chi(old_chi[i]);
      }
    }
    return;
  }

  const double w = std::clamp(cfg.radiation.holo.closure_relax, 0.0, 1.0);
  for (std::size_t i = 0; i < n_total; ++i) {
    if (old_valid[i] == 0U) {
      continue;
    }
    const double old_value = sanitize_holo_closure_chi(old_chi[i]);
    chi_host[i] = sanitize_holo_closure_chi((1.0 - w) * old_value +
                                            w * chi_host[i]);
  }
}

void regularize_holo_qd_closure_chi(core::State& state,
                                    const core::Config& cfg,
                                    const int n_cells,
                                    const int n_groups,
                                    const bool update_history,
                                    std::vector<double>& chi_host) {
  const std::size_t n_cells_us = static_cast<std::size_t>(std::max(n_cells, 0));
  const std::size_t n_groups_us = static_cast<std::size_t>(std::max(n_groups, 0));
  const std::size_t n_total = n_cells_us * n_groups_us;
  if (n_total == 0U || chi_host.empty()) {
    return;
  }
  TENRYU_ASSERT(chi_host.size() == n_total,
                "regularize_holo_qd_closure_chi size mismatch");

  sanitize_holo_closure_chi(chi_host);

  const int passes = cfg.radiation.holo.closure_smooth_passes;
  const double alpha = cfg.radiation.holo.closure_smooth_alpha;
  if (passes > 0 && alpha > 0.0) {
    const std::vector<int> material_id =
        compute_holo_closure_material_ids(state, cfg, n_cells);
    spatial_smooth_holo_qd_chi(
        chi_host, material_id, n_cells, n_groups, passes, alpha);
  }

  apply_holo_qd_closure_history(
      state, cfg, n_total, update_history, chi_host);
  sanitize_holo_closure_chi(chi_host);
}

[[nodiscard]] bool prepare_holo_qd_closure_chi(core::State& state,
                                               const core::Config& cfg,
                                               const int n_cells,
                                               const int n_groups,
                                               const bool update_history,
                                               std::vector<double>& chi_host) {
  const std::size_t n_cells_us = static_cast<std::size_t>(std::max(n_cells, 0));
  const std::size_t n_groups_us = static_cast<std::size_t>(std::max(n_groups, 0));
  const std::size_t n_total = n_cells_us * n_groups_us;
  chi_host.clear();
  if (!cfg.radiation.holo.p_rr_tally || n_total == 0U ||
      state.holo_chi.size() != n_total) {
    return false;
  }

  chi_host.assign(n_total, kClosureChiMin);
  state.holo_chi.copy_to_host(chi_host.data());
  regularize_holo_qd_closure_chi(
      state, cfg, n_cells, n_groups, update_history, chi_host);
  return true;
}

[[nodiscard]] bool prepare_holo_sn_closure_chi(core::State& state,
                                               const core::Config& cfg,
                                               const PlanckTable& planck,
                                               const std::vector<double>& sigma_P,
                                               const std::vector<double>& fleck_f,
                                               const std::vector<double>& Te,
                                               const std::vector<double>& node_r,
                                               const std::vector<double>& vol,
                                               const int n_cells,
                                               const int n_groups,
                                               const bool update_history,
                                               std::vector<double>& chi_host) {
  const std::size_t n_cells_us = static_cast<std::size_t>(std::max(n_cells, 0));
  const std::size_t n_groups_us = static_cast<std::size_t>(std::max(n_groups, 0));
  const std::size_t n_total = n_cells_us * n_groups_us;
  chi_host.clear();
  if (n_cells <= 0 || n_groups <= 0 || n_total == 0U) {
    return false;
  }
  TENRYU_ASSERT(planck.n_groups() == n_groups,
                "prepare_holo_sn_closure_chi group count mismatch");
  TENRYU_ASSERT(sigma_P.size() == n_total,
                "prepare_holo_sn_closure_chi sigma_P size mismatch");
  TENRYU_ASSERT(Te.size() == n_cells_us,
                "prepare_holo_sn_closure_chi Te size mismatch");
  TENRYU_ASSERT(node_r.size() == n_cells_us + 1U,
                "prepare_holo_sn_closure_chi node_r size mismatch");
  TENRYU_ASSERT(vol.size() == n_cells_us,
                "prepare_holo_sn_closure_chi vol size mismatch");
  TENRYU_ASSERT(fleck_f.empty() || fleck_f.size() == n_cells_us,
                "prepare_holo_sn_closure_chi fleck size mismatch");

  std::vector<double> sigma_a_eff(n_total, 0.0);
  std::vector<double> sigma_s_eff(n_total, 0.0);
  std::vector<double> planck_source(n_total, 0.0);
  const double te_floor = std::max(cfg.numerics.floors.Te, 1.0e-12);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    const double f_raw = fleck_f.empty() ? 1.0 : finite_or_zero_host(fleck_f[c_us]);
    const double fleck = std::clamp(f_raw, 0.0, 1.0);
    const double T =
        clamped_temperature_for_t4_host(std::max(finite_or_zero_host(Te[c_us]), te_floor));
    const double T4 = T * T * T * T;
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t key = cell_group_index_host(c, g, n_groups);
      const double sigma_a = std::max(finite_or_zero_host(sigma_P[key]), 0.0);
      const double sigma_a_fleck = fleck * sigma_a;
      sigma_a_eff[key] = sigma_a_fleck;
      sigma_s_eff[key] = (1.0 - fleck) * sigma_a;
      const double b_g =
          (n_groups == 1) ? 1.0 : std::max(planck.interpolate_b_host(g, T), 0.0);
      planck_source[key] =
          core::constants::c_light * sigma_a_fleck * core::constants::a_eV * T4 * b_g;
    }
  }

  SNTransport1DConfig sn_cfg{};
  sn_cfg.n_angles = cfg.radiation.holo.sn_n_angles;
  sn_cfg.origin_parity_only = cfg.radiation.origin_parity_only;
  const SNTransport1DResult sn =
      solve_sn_transport_1d(sigma_a_eff.data(),
                            sigma_s_eff.data(),
                            planck_source.data(),
                            node_r.data(),
                            vol.data(),
                            n_cells,
                            n_groups,
                            sn_cfg);
  chi_host = sn.chi;
  regularize_holo_qd_closure_chi(
      state, cfg, n_cells, n_groups, update_history, chi_host);

  if (!sn.converged && cfg.main.verbosity != "quiet") {
    std::ostringstream oss;
    oss << "[holo_sn] source iteration did not converge"
        << " iterations=" << sn.iterations
        << " error=" << sn.convergence_error
        << " tol=" << sn_cfg.convergence_tol;
    core::log_warning(oss.str());
  }
  return true;
}

HoloLOResult solve_holo_lo_source_ownership(
    core::State& state,
    const core::Config& cfg,
    const PlanckTable& planck,
    const core::Config::MaterialsConfig::MatDef& mat,
    const double* d_sigma_P,
    const double* d_sigma_R,
    const double* d_fleck,
    const int n_cells,
    const int n_groups,
    const double dt,
    const double* d_E_initial,
    const std::size_t E_initial_bytes,
    const double* d_consistency_source,
    const std::size_t consistency_source_bytes,
    const double* d_face_current_step,
    const std::size_t face_current_step_bytes,
    const double* d_reference_face_current,
    const std::size_t reference_face_current_bytes,
    const bool commit_material,
    const bool publish_source_diagnostics,
    const bool use_full_mesh_lo_solve,
    const bool has_physical_outer_vacuum,
    const char* solve_phase,
    const std::vector<double>* precomputed_chi = nullptr);

SNTransportGPUResult solve_holo_sn_material_coupling(core::State& state,
                                                     const core::Config& cfg,
                                                     const PlanckTable& planck,
                                                     const core::Config::MaterialsConfig::MatDef& mat,
                                                     const double* d_sigma_P,
                                                     const double* d_sigma_a_eff,
                                                     const double* d_sigma_s,
                                                     const double* d_sigma_R,
                                                     const double* d_fleck,
                                                     const int n_cells,
                                                     const int n_groups,
                                                     const double dt) {
  const std::size_t n_cells_us = static_cast<std::size_t>(std::max(n_cells, 0));
  const std::size_t n_groups_us = static_cast<std::size_t>(std::max(n_groups, 0));
  const std::size_t n_total = n_cells_us * n_groups_us;
  const bool use_qd_lo =
      state.mesh.dim == 1 && cfg.radiation.holo.solver == "quasidiffusion_1d";
  SNTransportGPUResult result{};
  if (n_cells <= 0 || n_groups <= 0 || !(dt > 0.0)) {
    result.converged = true;
    return result;
  }

  TENRYU_ASSERT(d_sigma_P != nullptr,
                "SN material coupling requires sigma_P");
  TENRYU_ASSERT(d_sigma_a_eff != nullptr,
                "SN material coupling requires sigma_a_eff");
  TENRYU_ASSERT(d_sigma_s != nullptr, "SN material coupling requires sigma_s");
  TENRYU_ASSERT(d_sigma_R != nullptr, "SN material coupling requires sigma_R");
  TENRYU_ASSERT(state.Te.size() == n_cells_us,
                "SN material coupling requires Te size match");
  TENRYU_ASSERT(state.ee.size() == n_cells_us,
                "SN material coupling requires ee size match");
  TENRYU_ASSERT(state.rho.size() == n_cells_us,
                "SN material coupling requires rho size match");
  TENRYU_ASSERT(state.vol.size() == n_cells_us,
                "SN material coupling requires vol size match");
  TENRYU_ASSERT(state.cv_e.empty() || state.cv_e.size() == n_cells_us,
                "SN material coupling requires cv_e size match");
  TENRYU_ASSERT(state.x_r.size() > 0U,
                "SN material coupling requires node_r");
  TENRYU_ASSERT(planck.n_groups() == n_groups,
                "SN material coupling planck group count mismatch");

  const bool holo_ale_invalidated = state.holo_ale_invalidated;
  const bool holo_E_LO_resized = state.holo_E_LO.size() != n_total;
  if (holo_E_LO_resized) {
    state.holo_E_LO.reset(n_total);
  }
  if (use_qd_lo && (holo_E_LO_resized || state.step == 0 || holo_ale_invalidated)) {
    if (state.rad_E.size() == n_total && n_total > 0U) {
      cuda_check(cudaMemcpy(state.holo_E_LO.data(),
                            state.rad_E.data(),
                            sizeof(double) * n_total,
                            cudaMemcpyDeviceToDevice),
                 "SN material coupling initialize holo_E_LO from rad_E failed");
    }
    initialize_holo_lo_state_cuda(state.holo_E_LO.data(), n_cells, n_groups);
    initialize_holo_lo_from_lte_cuda(state.holo_E_LO.data(),
                                     state.Te.data(),
                                     n_cells,
                                     n_groups);
  }
  if (state.holo_rad_dep.size() != n_total) {
    state.holo_rad_dep.reset(n_total);
  }
  if (state.holo_rad_emit.size() != n_total) {
    state.holo_rad_emit.reset(n_total);
  }
  if (state.holo_Prr.size() != n_total) {
    state.holo_Prr.reset(n_total);
  }
  if (state.holo_chi.size() != n_total) {
    state.holo_chi.reset(n_total);
  }
  if (state.holo_Prr_coverage.size() != n_total) {
    state.holo_Prr_coverage.reset(n_total);
  }
  if (state.holo_consistency_source.size() != n_total) {
    state.holo_consistency_source.reset(n_total);
  }
  state.holo_consistency_source.fill(0.0);

  state.holo_core_mask.assign(n_cells_us, static_cast<std::uint8_t>(1U));
  state.holo_patch_mask.assign(n_cells_us, static_cast<std::uint8_t>(0U));
  state.holo_core_prev_mask.assign(n_cells_us, static_cast<std::uint8_t>(1U));
  state.holo_hold_count.assign(n_cells_us, 0);
  state.holo_dwell_count.assign(n_cells_us, 0);
  state.holo_tau_R.assign(n_cells_us, 0.0);
  state.holo_reduced_flux.assign(n_cells_us, 0.0);
  state.holo_mass_q.assign(n_cells_us, 0.0);
  state.holo_lo_weight.assign(n_cells_us, 1.0);
  state.holo_core_mask_valid = true;
  state.holo_lo_source_valid = false;

  SNTransportGPUConfig sn_cfg{};
  sn_cfg.n_angles = cfg.radiation.holo.sn_n_angles;
  sn_cfg.temperature_floor_eV = std::max(cfg.numerics.floors.Te, 1.0e-12);
  sn_cfg.origin_parity_only = cfg.radiation.origin_parity_only;

  parallel::DeviceArray d_Te_old;
  d_Te_old.resize(sizeof(double) * n_cells_us);
  cuda_check(cudaMemcpy(d_Te_old.as<double>(),
                        state.Te.data(),
                        sizeof(double) * n_cells_us,
                        cudaMemcpyDeviceToDevice),
             "SN material coupling copy Te_old failed");

  double cv_e_const = 0.0;
  double Cv_e_const = 0.0;
  const int first_nonvoid = cfg.materials.first_nonvoid_material_index();
  if (first_nonvoid >= 0) {
    const auto& mat = cfg.materials.materials[static_cast<std::size_t>(first_nonvoid)];
    if (mat.cv_e_override > 0.0) {
      Cv_e_const = mat.cv_e_override;
    }
    const double gm1 = std::max(mat.ideal_gas_gamma - 1.0, 1.0e-12);
    const double A = std::max(mat.A, 1.0e-12);
    cv_e_const =
        core::constants::eV_to_erg / (A * core::constants::proton_mass * gm1);
  }

  parallel::DeviceArray d_sn_E_out;
  double* sn_E_out = state.holo_E_LO.data();
  if (use_qd_lo) {
    d_sn_E_out.resize(sizeof(double) * n_total);
    sn_E_out = d_sn_E_out.as<double>();
  }

  SNMaterialCouplingGPUInputs sn_in{};
  sn_in.sigma_a = d_sigma_a_eff;
  sn_in.sigma_s = d_sigma_s;
  sn_in.Te = state.Te.data();
  sn_in.ee = state.ee.data();
  sn_in.node_r = state.x_r.data();
  sn_in.node_z = (state.mesh.dim == 2) ? state.x_z.data() : nullptr;
  sn_in.vol = state.vol.data();
  sn_in.rho = state.rho.data();
  sn_in.cv_e = state.cv_e.empty() ? nullptr : state.cv_e.data();
  sn_in.sigma_R = d_sigma_R;
  sn_in.Te_old = d_Te_old.as<double>();
  sn_in.planck = planck.device_view();
  sn_in.planck_table_cpu = &planck;
  sn_in.E_out = sn_E_out;
  sn_in.P_rr_out = state.holo_Prr.data();
  sn_in.chi_out = state.holo_chi.data();
  sn_in.rad_dep = state.holo_rad_dep.data();
  sn_in.rad_emit = state.holo_rad_emit.data();
  sn_in.coverage = state.holo_Prr_coverage.data();
  sn_in.dim = state.mesh.dim;
  if (state.mesh.dim == 2) {
    sn_in.nr = state.mesh.topo.nr;
    sn_in.nz = state.mesh.topo.nz;
    TENRYU_ASSERT(sn_in.nr * sn_in.nz == n_cells,
                  "SN material coupling 2D mesh size mismatch");
    TENRYU_ASSERT(state.x_r.size() ==
                      static_cast<std::size_t>((sn_in.nr + 1) * (sn_in.nz + 1)),
                  "SN material coupling 2D node_r size mismatch");
    TENRYU_ASSERT(state.x_z.size() == state.x_r.size(),
                  "SN material coupling 2D node_z size mismatch");
  } else {
    sn_in.nr = n_cells;
    sn_in.nz = 1;
    TENRYU_ASSERT(state.x_r.size() == n_cells_us + 1U,
                  "SN material coupling 1D node_r size mismatch");
  }
  sn_in.n_groups = n_groups;
  sn_in.dt = dt;
  sn_in.update_material = !use_qd_lo;
  sn_in.cv_e_const = cv_e_const;
  sn_in.Cv_e_const = Cv_e_const;

  result = solve_sn_material_coupling_gpu(sn_in, sn_cfg);
  if (!result.converged && cfg.main.verbosity != "quiet") {
    std::ostringstream oss;
    oss << "[holo_sn_gpu] source iteration did not converge"
        << " iterations=" << result.iterations
        << " error=" << result.convergence_error
        << " tol=" << sn_cfg.convergence_tol;
    core::log_warning(oss.str());
  }
  if (use_qd_lo) {
    std::vector<double> chi_host(n_total, 0.0);
    state.holo_chi.copy_to_host(chi_host.data());

    const std::size_t E_initial_bytes = sizeof(double) * n_total;
    const HoloLOResult lo_result =
        solve_holo_lo_source_ownership(state,
                                       cfg,
                                       planck,
                                       mat,
                                       d_sigma_P,
                                       d_sigma_R,
                                       d_fleck,
                                       n_cells,
                                       n_groups,
                                       dt,
                                       state.holo_E_LO.data(),
                                       E_initial_bytes,
                                       nullptr,
                                       0U,
                                       nullptr,
                                       0U,
                                       nullptr,
                                       0U,
                                       true,
                                       true,
                                       true,
                                       cfg.radiation.boundary.outer_r == "vacuum",
                                       "sn_material_qd",
                                       &chi_host);
    result.lo_solver_iterations = lo_result.solver_iterations;
    result.lo_failures = lo_result.failures;
  } else {
    state.holo_lo_source_valid = true;
  }
  state.holo_ale_invalidated = false;
  return result;
}

double harmonic_pair_host(const double a, const double b, const double floor) {
  const double aa = std::max(a, floor);
  const double bb = std::max(b, floor);
  return 2.0 * aa * bb / std::max(aa + bb, floor);
}

double reference_derivative_weight_host(const PlanckTable& planck,
                                        const int g,
                                        const double T) {
  const double dT = std::max(T * 0.01, 1.0e-3);
  const double T_hi = T + dT;
  const double T_lo = std::max(T - dT, 1.0e-6);
  const double t4_hi = T_hi * T_hi * T_hi * T_hi;
  const double t4_lo = T_lo * T_lo * T_lo * T_lo;
  const double b_hi = std::max(planck.interpolate_b_host(g, T_hi), 0.0);
  const double b_lo = std::max(planck.interpolate_b_host(g, T_lo), 0.0);
  return std::max((t4_hi * b_hi - t4_lo * b_lo) /
                      std::max(T_hi - T_lo, 1.0e-30),
                  0.0);
}

ReferenceFieldDiagnostics compute_reference_field_diagnostics(
    const core::State& state,
    const core::Config& cfg,
    const PlanckTable& planck,
    const std::vector<double>& sigma_R,
    const std::vector<double>& node_r,
    const std::vector<double>* node_z,
    const std::vector<std::uint8_t>* diffusion_cell,
    const int n_cells,
    const int n_groups,
    const std::vector<double>* physical_E_density = nullptr,
    std::vector<double>* E_ref_start = nullptr,
    std::vector<double>* W_ref_cell = nullptr) {
  ReferenceFieldDiagnostics diag{};
  if (!cfg.radiation.imc.difference.enabled) {
    return diag;
  }
  const bool is_1d = (state.mesh.dim == 1);
  const bool is_2d = (state.mesh.dim == 2);
  TENRYU_ASSERT(is_1d || is_2d,
                "difference reference diagnostics require 1D_SPH or 2D_RZ");
  const std::size_t n_cells_us = static_cast<std::size_t>(std::max(n_cells, 0));
  const std::size_t n_groups_us = static_cast<std::size_t>(std::max(n_groups, 0));
  const std::size_t n_cell_groups_us = n_cells_us * n_groups_us;
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  TENRYU_ASSERT(sigma_R.size() == n_cell_groups_us,
                "compute_reference_field_diagnostics sigma_R size mismatch");
  if (is_1d) {
    TENRYU_ASSERT(node_r.size() == n_cells_us + 1U,
                  "compute_reference_field_diagnostics node_r size mismatch");
  } else {
    TENRYU_ASSERT(nr > 0 && nz > 0,
                  "compute_reference_field_diagnostics requires valid 2D topology");
    TENRYU_ASSERT(static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz) ==
                      n_cells_us,
                  "compute_reference_field_diagnostics requires nr*nz cells");
    const std::size_t n_nodes =
        static_cast<std::size_t>(nr + 1) * static_cast<std::size_t>(nz + 1);
    TENRYU_ASSERT(node_r.size() == n_nodes,
                  "compute_reference_field_diagnostics node_r size mismatch");
    TENRYU_ASSERT(node_z != nullptr && node_z->size() == n_nodes,
                  "compute_reference_field_diagnostics node_z size mismatch");
  }
  TENRYU_ASSERT(planck.n_groups() == n_groups,
                "compute_reference_field_diagnostics group count mismatch");
  TENRYU_ASSERT(state.cell_is_void.size() == n_cells_us,
                "compute_reference_field_diagnostics void mask size mismatch");

  diag.valid = true;
  if (E_ref_start != nullptr) {
    E_ref_start->assign(n_cell_groups_us, 0.0);
  }
  if (W_ref_cell != nullptr) {
    W_ref_cell->assign(n_cells_us, 0.0);
  }
  if (n_cells <= 0 || n_groups <= 0) {
    return diag;
  }

  std::vector<double> host_Te(n_cells_us, 0.0);
  std::vector<double> host_rho(n_cells_us, 0.0);
  std::vector<double> host_vol(n_cells_us, 0.0);
  state.Te.copy_to_host(host_Te.data());
  state.rho.copy_to_host(host_rho.data());
  state.vol.copy_to_host(host_vol.data());

  std::vector<double> host_rad_E(n_cell_groups_us, 0.0);
  if (physical_E_density != nullptr) {
    TENRYU_ASSERT(physical_E_density->size() == n_cell_groups_us,
                  "compute_reference_field_diagnostics physical_E_density size mismatch");
    host_rad_E = *physical_E_density;
  } else if (state.rad_E.size() == n_cell_groups_us && n_cell_groups_us > 0U) {
    state.rad_E.copy_to_host(host_rad_E.data());
  }

  constexpr double kSigmaFloor = 1.0e-30;
  constexpr double kLogFloor = 1.0e-300;
  const auto& dc = cfg.radiation.imc.difference;
  const double e_floor =
      std::max(core::constants::a_eV *
                   safe_temperature_pow4_host(std::max(cfg.numerics.floors.Te, 1.0e-12)),
               1.0e-300);

  std::vector<double> sigma_cell(n_cells_us, 0.0);
  std::vector<double> E_total(n_cells_us, 0.0);
  std::vector<double> centers(n_cells_us, 0.0);
  std::vector<double> centers_z(n_cells_us, 0.0);
  double W_sum = 0.0;
  double tau_sum = 0.0;
  double chi_sum = 0.0;
  double W_min = std::numeric_limits<double>::infinity();
  double W_max_seen = 0.0;
  double tau_min = std::numeric_limits<double>::infinity();
  double tau_max = 0.0;
  double chi_max = 0.0;

  for (int c = 0; c < n_cells; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    double h_eff = 0.0;
    if (is_1d) {
      centers[c_us] = 0.5 * (node_r[c_us] + node_r[c_us + 1U]);
      h_eff = std::max(node_r[c_us + 1U] - node_r[c_us], 0.0);
    } else {
      const int i = c / nz;
      const int j = c - i * nz;
      const int stride = nz + 1;
      const int n00 = i * stride + j;
      const int n10 = (i + 1) * stride + j;
      const int n11 = (i + 1) * stride + (j + 1);
      const int n01 = i * stride + (j + 1);
      centers[c_us] =
          0.25 * (node_r[static_cast<std::size_t>(n00)] +
                  node_r[static_cast<std::size_t>(n10)] +
                  node_r[static_cast<std::size_t>(n11)] +
                  node_r[static_cast<std::size_t>(n01)]);
      centers_z[c_us] =
          0.25 * ((*node_z)[static_cast<std::size_t>(n00)] +
                  (*node_z)[static_cast<std::size_t>(n10)] +
                  (*node_z)[static_cast<std::size_t>(n11)] +
                  (*node_z)[static_cast<std::size_t>(n01)]);
      const tenryu::mesh::CellWidths2D widths =
          tenryu::mesh::compute_cell_widths_2d(node_r.data(),
                                               node_z->data(),
                                               host_vol.data(),
                                               nr,
                                               nz,
                                               c);
      h_eff = std::max(std::min(widths.h_R, widths.h_Z), 0.0);
    }
    if (state.cell_is_void[c_us] != 0U) {
      continue;
    }

    ++diag.eligible_cells;
    const std::size_t base = c_us * n_groups_us;
    const double T =
        clamped_temperature_for_t4_host(std::max(host_Te[c_us], cfg.numerics.floors.Te));
    const double T4 = safe_temperature_pow4_host(T);

    double weight_sum = 0.0;
    double denom_sigma = 0.0;
    double sigma_max = 0.0;
    double chi_num = 0.0;
    double chi_den = e_floor;
    double B_total = 0.0;
    double E_phys_total = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx = base + static_cast<std::size_t>(g);
      const double sigma_g = std::max(finite_or_zero_host(sigma_R[idx]), 0.0);
      sigma_max = std::max(sigma_max, sigma_g);
      const double w_g =
          (n_groups > 1) ? reference_derivative_weight_host(planck, g, T) : 1.0;
      weight_sum += w_g;
      denom_sigma += w_g / std::max(sigma_g, kSigmaFloor);

      const double b_g =
          (n_groups == 1) ? 1.0 : std::max(planck.interpolate_b_host(g, T), 0.0);
      const double B_g = core::constants::a_eV * T4 * b_g;
      const double E_g = finite_or_zero_host(host_rad_E[idx]);
      B_total += B_g;
      E_phys_total += std::max(E_g, 0.0);
      chi_num += std::abs(E_g - B_g);
      chi_den += std::max(E_g, 0.0) + B_g;
    }

    const double sigma_mean =
        difference_reference_cell_sigma(weight_sum, denom_sigma, sigma_max);
    sigma_cell[c_us] = sigma_mean;
    E_total[c_us] = E_phys_total;
    const double tau = sigma_mean * h_eff;
    const double chi = (chi_den > 0.0) ? (chi_num / chi_den) : 0.0;
    double W = difference_reference_weight(dc.W_max, tau, dc.tau0, chi, dc.chi0);

    if (diffusion_cell != nullptr && c_us < diffusion_cell->size() &&
        (*diffusion_cell)[c_us] != 0U) {
      W = 0.0;
      ++diag.hybrid_suppressed_cells;
    }

    W_sum += W;
    tau_sum += tau;
    chi_sum += chi;
    W_min = std::min(W_min, W);
    W_max_seen = std::max(W_max_seen, W);
    tau_min = std::min(tau_min, tau);
    tau_max = std::max(tau_max, tau);
    chi_max = std::max(chi_max, chi);
    if (W > 0.0) {
      ++diag.active_cells;
    }
    if (W >= 0.5) {
      ++diag.strong_cells;
    }
    if (W_ref_cell != nullptr) {
      (*W_ref_cell)[c_us] = W;
    }
    diag.E_ref_total += W * B_total * std::max(host_vol[c_us], 0.0);
    if (E_ref_start != nullptr && W != 0.0) {
      for (int g = 0; g < n_groups; ++g) {
        const std::size_t idx = base + static_cast<std::size_t>(g);
        const double b_g =
            (n_groups == 1) ? 1.0 : std::max(planck.interpolate_b_host(g, T), 0.0);
        (*E_ref_start)[idx] = W * core::constants::a_eV * T4 * b_g;
      }
    }
  }

  if (diag.eligible_cells > 0) {
    const double inv_count = 1.0 / static_cast<double>(diag.eligible_cells);
    diag.W_min = std::isfinite(W_min) ? W_min : 0.0;
    diag.W_mean = W_sum * inv_count;
    diag.W_max = W_max_seen;
    diag.tau_min = std::isfinite(tau_min) ? tau_min : 0.0;
    diag.tau_mean = tau_sum * inv_count;
    diag.tau_max = tau_max;
    diag.chi_mean = chi_sum * inv_count;
    diag.chi_max = chi_max;
  }

  const auto accumulate_face_diag = [&](const int left, const int right) {
    const std::size_t l = static_cast<std::size_t>(left);
    const std::size_t r = static_cast<std::size_t>(right);
    if (state.cell_is_void[l] != 0U || state.cell_is_void[r] != 0U) {
      return;
    }
    const double dr = centers[r] - centers[l];
    const double dz = centers_z[r] - centers_z[l];
    const double ds = std::max(std::sqrt(dr * dr + dz * dz), 1.0e-30);
    const double sigma_face = harmonic_pair_host(sigma_cell[l], sigma_cell[r], kSigmaFloor);
    const double E_face = std::max(0.5 * (E_total[l] + E_total[r]), e_floor);
    const double knudsen =
        std::abs(E_total[r] - E_total[l]) / (std::max(sigma_face, kSigmaFloor) * E_face * ds);
    if (std::isfinite(knudsen)) {
      diag.knudsen_max = std::max(diag.knudsen_max, knudsen);
      diag.reduced_flux_max = std::max(diag.reduced_flux_max, knudsen / 3.0);
    }

    const double dln_Te =
        std::abs(std::log(std::max(host_Te[r], kLogFloor) /
                          std::max(host_Te[l], kLogFloor)));
    const double dln_rho =
        std::abs(std::log(std::max(host_rho[r], kLogFloor) /
                          std::max(host_rho[l], kLogFloor)));
    if (std::isfinite(dln_Te)) {
      diag.front_grad_Te_max = std::max(diag.front_grad_Te_max, dln_Te);
    }
    if (std::isfinite(dln_rho)) {
      diag.front_grad_rho_max = std::max(diag.front_grad_rho_max, dln_rho);
    }
  };

  if (is_1d) {
    for (int c = 0; c + 1 < n_cells; ++c) {
      accumulate_face_diag(c, c + 1);
    }
  } else {
    for (int c = 0; c < n_cells; ++c) {
      const int i = c / nz;
      const int j = c - i * nz;
      if (i + 1 < nr) {
        accumulate_face_diag(c, c + nz);
      }
      if (j + 1 < nz) {
        accumulate_face_diag(c, c + 1);
      }
    }
  }

  return diag;
}

std::vector<double> compute_imc_source_tilt_1d(const core::State& state,
                                               const core::Config& cfg) {
  const int n_cells = static_cast<int>(state.Te.size());
  std::vector<double> tilt(static_cast<std::size_t>(std::max(n_cells, 0)), 0.0);
  if (!cfg.radiation.imc.source_tilting || state.mesh.dim != 1 || n_cells < 3) {
    return tilt;
  }

  TENRYU_ASSERT(static_cast<int>(state.cell_is_void.size()) == n_cells,
                "compute_imc_source_tilt_1d requires cell_is_void size to match n_cells");
  TENRYU_ASSERT(state.x_r.size() >= static_cast<std::size_t>(n_cells + 1),
                "compute_imc_source_tilt_1d requires 1D node-centered x_r");

  std::vector<double> host_Te(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> host_node_r(static_cast<std::size_t>(n_cells + 1), 0.0);
  state.Te.copy_to_host(host_Te.data());
  state.x_r.copy_to_host(host_node_r.data());

  std::vector<double> centers(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> U(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    centers[c_us] = 0.5 * (host_node_r[c_us] + host_node_r[c_us + 1]);
    U[c_us] = core::constants::a_eV * safe_temperature_pow4_host(host_Te[c_us]);
  }

  const double U_floor =
      std::max(core::constants::a_eV * safe_temperature_pow4_host(cfg.numerics.floors.Te),
               std::numeric_limits<double>::min());

  for (int c = 1; c + 1 < n_cells; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    if (state.cell_is_void[c_us] != 0U || state.cell_is_void[c_us - 1] != 0U ||
        state.cell_is_void[c_us + 1] != 0U) {
      continue;
    }

    const double dr_cell = host_node_r[c_us + 1] - host_node_r[c_us];
    const double center_span = centers[c_us + 1] - centers[c_us - 1];
    if (!(dr_cell > 0.0) || !(center_span > 0.0)) {
      continue;
    }

    const double grad = (U[c_us + 1] - U[c_us - 1]) / center_span;
    const double delta = grad * (0.5 * dr_cell) / std::max(U[c_us], U_floor);
    if (std::isfinite(delta)) {
      tilt[c_us] = std::clamp(delta, -1.0, 1.0);
    }
  }

  return tilt;
}

std::vector<double> compute_imc_source_tilt_2d(const core::State& state,
                                               const core::Config& cfg) {
  const int n_cells = static_cast<int>(state.Te.size());
  std::vector<double> tilt(2U * static_cast<std::size_t>(std::max(n_cells, 0)), 0.0);
  if (!cfg.radiation.imc.source_tilting || state.mesh.dim != 2 || n_cells < 4) {
    return tilt;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  TENRYU_ASSERT(nr > 0 && nz > 0,
                "compute_imc_source_tilt_2d requires valid 2D topology");
  TENRYU_ASSERT(static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz) ==
                    static_cast<std::size_t>(n_cells),
                "compute_imc_source_tilt_2d requires nr*nz cells");
  TENRYU_ASSERT(static_cast<int>(state.cell_is_void.size()) == n_cells,
                "compute_imc_source_tilt_2d requires cell_is_void size to match n_cells");
  const std::size_t n_nodes =
      static_cast<std::size_t>(nr + 1) * static_cast<std::size_t>(nz + 1);
  TENRYU_ASSERT(state.x_r.size() == n_nodes,
                "compute_imc_source_tilt_2d requires x_r node count to match topology");
  TENRYU_ASSERT(state.x_z.size() == n_nodes,
                "compute_imc_source_tilt_2d requires x_z node count to match topology");

  std::vector<double> host_Te(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> host_node_r(n_nodes, 0.0);
  std::vector<double> host_node_z(n_nodes, 0.0);
  state.Te.copy_to_host(host_Te.data());
  state.x_r.copy_to_host(host_node_r.data());
  state.x_z.copy_to_host(host_node_z.data());

  std::vector<double> center_r(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> center_z(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> U(static_cast<std::size_t>(n_cells), 0.0);
  const int stride = nz + 1;
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int c = i * nz + j;
      const std::size_t c_us = static_cast<std::size_t>(c);
      const int n00 = i * stride + j;
      const int n10 = (i + 1) * stride + j;
      const int n11 = (i + 1) * stride + (j + 1);
      const int n01 = i * stride + (j + 1);
      center_r[c_us] = 0.25 * (host_node_r[static_cast<std::size_t>(n00)] +
                               host_node_r[static_cast<std::size_t>(n10)] +
                               host_node_r[static_cast<std::size_t>(n11)] +
                               host_node_r[static_cast<std::size_t>(n01)]);
      center_z[c_us] = 0.25 * (host_node_z[static_cast<std::size_t>(n00)] +
                               host_node_z[static_cast<std::size_t>(n10)] +
                               host_node_z[static_cast<std::size_t>(n11)] +
                               host_node_z[static_cast<std::size_t>(n01)]);
      U[c_us] = core::constants::a_eV * safe_temperature_pow4_host(host_Te[c_us]);
    }
  }

  const double U_floor =
      std::max(core::constants::a_eV * safe_temperature_pow4_host(cfg.numerics.floors.Te),
               std::numeric_limits<double>::min());

  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int c = i * nz + j;
      const std::size_t c_us = static_cast<std::size_t>(c);
      if (state.cell_is_void[c_us] != 0U) {
        continue;
      }

      const int n00 = i * stride + j;
      const int n10 = (i + 1) * stride + j;
      const int n11 = (i + 1) * stride + (j + 1);
      const int n01 = i * stride + (j + 1);
      const double r_min =
          std::min(std::min(host_node_r[static_cast<std::size_t>(n00)],
                            host_node_r[static_cast<std::size_t>(n10)]),
                   std::min(host_node_r[static_cast<std::size_t>(n11)],
                            host_node_r[static_cast<std::size_t>(n01)]));
      const double r_max =
          std::max(std::max(host_node_r[static_cast<std::size_t>(n00)],
                            host_node_r[static_cast<std::size_t>(n10)]),
                   std::max(host_node_r[static_cast<std::size_t>(n11)],
                            host_node_r[static_cast<std::size_t>(n01)]));
      const double z_min =
          std::min(std::min(host_node_z[static_cast<std::size_t>(n00)],
                            host_node_z[static_cast<std::size_t>(n10)]),
                   std::min(host_node_z[static_cast<std::size_t>(n11)],
                            host_node_z[static_cast<std::size_t>(n01)]));
      const double z_max =
          std::max(std::max(host_node_z[static_cast<std::size_t>(n00)],
                            host_node_z[static_cast<std::size_t>(n10)]),
                   std::max(host_node_z[static_cast<std::size_t>(n11)],
                            host_node_z[static_cast<std::size_t>(n01)]));

      double tilt_r = 0.0;
      if (i > 0 && i + 1 < nr) {
        const int cm = c - nz;
        const int cp = c + nz;
        const std::size_t cm_us = static_cast<std::size_t>(cm);
        const std::size_t cp_us = static_cast<std::size_t>(cp);
        const double dr_cell = r_max - r_min;
        const double center_span = center_r[cp_us] - center_r[cm_us];
        if (state.cell_is_void[cm_us] == 0U && state.cell_is_void[cp_us] == 0U &&
            dr_cell > 0.0 && center_span > 0.0) {
          const double grad = (U[cp_us] - U[cm_us]) / center_span;
          const double delta = grad * (0.5 * dr_cell) / std::max(U[c_us], U_floor);
          if (std::isfinite(delta)) {
            tilt_r = delta;
          }
        }
      }

      double tilt_z = 0.0;
      if (j > 0 && j + 1 < nz) {
        const int cm = c - 1;
        const int cp = c + 1;
        const std::size_t cm_us = static_cast<std::size_t>(cm);
        const std::size_t cp_us = static_cast<std::size_t>(cp);
        const double dz_cell = z_max - z_min;
        const double center_span = center_z[cp_us] - center_z[cm_us];
        if (state.cell_is_void[cm_us] == 0U && state.cell_is_void[cp_us] == 0U &&
            dz_cell > 0.0 && center_span > 0.0) {
          const double grad = (U[cp_us] - U[cm_us]) / center_span;
          const double delta = grad * (0.5 * dz_cell) / std::max(U[c_us], U_floor);
          if (std::isfinite(delta)) {
            tilt_z = delta;
          }
        }
      }

      const double l1 = std::abs(tilt_r) + std::abs(tilt_z);
      if (l1 > 1.0) {
        tilt_r /= l1;
        tilt_z /= l1;
      }
      tilt[2U * c_us] = std::clamp(tilt_r, -1.0, 1.0);
      tilt[2U * c_us + 1U] = std::clamp(tilt_z, -1.0, 1.0);
    }
  }

  return tilt;
}

std::int64_t fold_ddmc_census_from_imc_partition_into_tally(PhotonPool& pool,
                                                            const int n_particles,
                                                            const int n_cells,
                                                            const int n_groups,
                                                            const double dt,
                                                            double* d_rad_E_tally) {
  TENRYU_ASSERT(d_rad_E_tally != nullptr,
                "fold_ddmc_census_from_imc_partition_into_tally requires rad_E_tally");
  if (n_particles <= 0 || n_cells <= 0 || n_groups <= 0 || !(dt > 0.0)) {
    return 0;
  }

  const std::size_t n_particles_us = static_cast<std::size_t>(n_particles);
  const std::size_t n_cell_groups =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  std::vector<std::uint8_t> host_alive(n_particles_us, kDead);
  std::vector<std::uint8_t> host_mode(n_particles_us, kModeIMC);
  std::vector<std::int32_t> host_cell(n_particles_us, -1);
  std::vector<std::uint16_t> host_group(n_particles_us, 0);
  std::vector<double> host_energy(n_particles_us, 0.0);

  cuda_check(cudaMemcpy(host_alive.data(),
                        pool.alive,
                        sizeof(std::uint8_t) * n_particles_us,
                        cudaMemcpyDeviceToHost),
             "IMC::transport_step copy alive for DDMC fold failed");
  cuda_check(cudaMemcpy(host_mode.data(),
                        pool.mode,
                        sizeof(std::uint8_t) * n_particles_us,
                        cudaMemcpyDeviceToHost),
             "IMC::transport_step copy mode for DDMC fold failed");
  cuda_check(cudaMemcpy(host_cell.data(),
                        pool.cell_id,
                        sizeof(std::int32_t) * n_particles_us,
                        cudaMemcpyDeviceToHost),
             "IMC::transport_step copy cell_id for DDMC fold failed");
  cuda_check(cudaMemcpy(host_group.data(),
                        pool.group_id,
                        sizeof(std::uint16_t) * n_particles_us,
                        cudaMemcpyDeviceToHost),
             "IMC::transport_step copy group_id for DDMC fold failed");
  cuda_check(cudaMemcpy(host_energy.data(),
                        pool.energy,
                        sizeof(double) * n_particles_us,
                        cudaMemcpyDeviceToHost),
             "IMC::transport_step copy energy for DDMC fold failed");

  std::vector<double> host_rad_E_tally;
  std::int64_t folded_particles = 0;
  for (std::size_t i = 0; i < n_particles_us; ++i) {
    if (host_alive[i] != kAlive || host_mode[i] != kModeDDMC) {
      continue;
    }

    const int cell = static_cast<int>(host_cell[i]);
    const int group = static_cast<int>(host_group[i]);
    TENRYU_ASSERT(cell >= 0 && cell < n_cells,
                  "IMC::transport_step DDMC fold encountered invalid cell");
    TENRYU_ASSERT(group >= 0 && group < n_groups,
                  "IMC::transport_step DDMC fold encountered invalid group");
    if (host_rad_E_tally.empty()) {
      host_rad_E_tally.assign(n_cell_groups, 0.0);
      cuda_check(cudaMemcpy(host_rad_E_tally.data(),
                            d_rad_E_tally,
                            sizeof(double) * n_cell_groups,
                            cudaMemcpyDeviceToHost),
                 "IMC::transport_step copy rad_E_tally for DDMC fold failed");
    }

    const std::size_t key =
        static_cast<std::size_t>(cell) * static_cast<std::size_t>(n_groups) +
        static_cast<std::size_t>(group);
    host_rad_E_tally[key] +=
        core::constants::c_light * std::max(host_energy[i], 0.0) * dt;
    host_alive[i] = kDead;
    ++folded_particles;
  }

  if (folded_particles <= 0) {
    return 0;
  }

  cuda_check(cudaMemcpy(d_rad_E_tally,
                        host_rad_E_tally.data(),
                        sizeof(double) * n_cell_groups,
                        cudaMemcpyHostToDevice),
             "IMC::transport_step upload rad_E_tally for DDMC fold failed");
  cuda_check(cudaMemcpy(pool.alive,
                        host_alive.data(),
                        sizeof(std::uint8_t) * n_particles_us,
                        cudaMemcpyHostToDevice),
             "IMC::transport_step upload alive for DDMC fold failed");
  return folded_particles;
}

void kill_particles_with_mode(PhotonPool& pool,
                              const int n_particles,
                              const std::uint8_t mode_to_kill) {
  if (n_particles <= 0) {
    return;
  }

  const std::size_t n_particles_us = static_cast<std::size_t>(n_particles);
  std::vector<std::uint8_t> host_alive(n_particles_us, kDead);
  std::vector<std::uint8_t> host_mode(n_particles_us, kModeIMC);
  cuda_check(cudaMemcpy(host_alive.data(),
                        pool.alive,
                        sizeof(std::uint8_t) * n_particles_us,
                        cudaMemcpyDeviceToHost),
             "IMC::transport_step copy alive for mode kill failed");
  cuda_check(cudaMemcpy(host_mode.data(),
                        pool.mode,
                        sizeof(std::uint8_t) * n_particles_us,
                        cudaMemcpyDeviceToHost),
             "IMC::transport_step copy mode for mode kill failed");
  for (std::size_t i = 0; i < n_particles_us; ++i) {
    if (host_alive[i] == kAlive && host_mode[i] == mode_to_kill) {
      host_alive[i] = kDead;
    }
  }
  cuda_check(cudaMemcpy(pool.alive,
                        host_alive.data(),
                        sizeof(std::uint8_t) * n_particles_us,
                        cudaMemcpyHostToDevice),
             "IMC::transport_step upload alive for mode kill failed");
}

struct RankBoundaryParams1D {
  int ghost_layers = 0;
  int nr_local = 0;
  bool has_left_boundary = false;
  bool has_right_boundary = false;
};

struct RankBoundaryParams2D {
  int ghost_layers = 0;
  int nr_local = 0;
  int nz_local = 0;
  bool has_r_inner_boundary = false;
  bool has_r_outer_boundary = false;
  bool has_z_bottom_boundary = false;
  bool has_z_top_boundary = false;
};

RankBoundaryParams1D make_rank_boundary_params_1d(
    const parallel::PartitionInfo& part) {
  RankBoundaryParams1D out{};
  out.ghost_layers = std::max(part.ghost_layers, 0);
  out.nr_local = std::max(part.nr_local, 0);
  out.has_left_boundary = part.has_left_boundary();
  out.has_right_boundary = part.has_right_boundary();
  return out;
}

RankBoundaryParams2D make_rank_boundary_params_2d(
    const parallel::PartitionInfo& part) {
  RankBoundaryParams2D out{};
  out.ghost_layers = std::max(part.ghost_layers, 0);
  out.nr_local = std::max(part.nr_local, 0);
  out.nz_local = std::max(part.nz_local, 0);
  out.has_r_inner_boundary = part.has_left_boundary();
  out.has_r_outer_boundary = part.has_right_boundary();
  out.has_z_bottom_boundary = part.has_bottom_boundary();
  out.has_z_top_boundary = part.has_top_boundary();
  return out;
}

double compute_alive_particle_energy_host(const PhotonPool& pool) {
  if (pool.n_alive <= 0 || pool.energy == nullptr || pool.alive == nullptr) {
    return 0.0;
  }
  TENRYU_ASSERT(pool.n_alive <= pool.capacity,
                "compute_alive_particle_energy_host pool.n_alive exceeds pool.capacity");

  std::vector<double> host_energy(static_cast<std::size_t>(pool.n_alive), 0.0);
  std::vector<std::uint8_t> host_alive(static_cast<std::size_t>(pool.n_alive), kDead);
  cuda_check(cudaMemcpy(host_energy.data(),
                        pool.energy,
                        sizeof(double) * host_energy.size(),
                        cudaMemcpyDeviceToHost),
             "compute_alive_particle_energy_host copy energy failed");
  cuda_check(cudaMemcpy(host_alive.data(),
                        pool.alive,
                        sizeof(std::uint8_t) * host_alive.size(),
                        cudaMemcpyDeviceToHost),
             "compute_alive_particle_energy_host copy alive failed");

  long double sum = 0.0L;
  for (std::size_t i = 0; i < host_energy.size(); ++i) {
    if (host_alive[i] == kAlive) {
      sum += static_cast<long double>(std::max(host_energy[i], 0.0));
    }
  }
  return static_cast<double>(sum);
}

double compute_diffusion_energy_host(const parallel::DeviceArray& diff_E,
                                     const std::vector<std::uint8_t>& diff_cell,
                                     const std::vector<double>& vol) {
  const std::size_t n_cells = diff_cell.size();
  if (n_cells == 0U || diff_E.ptr == nullptr || diff_E.size == 0U) {
    return 0.0;
  }
  TENRYU_ASSERT(vol.size() == n_cells,
                "compute_diffusion_energy_host volume size mismatch");
  const std::size_t n_values = diff_E.size / sizeof(double);
  TENRYU_ASSERT(n_values % n_cells == 0U,
                "compute_diffusion_energy_host diffusion energy size mismatch");
  const std::size_t n_groups = n_values / n_cells;

  std::vector<double> host_diff_E(n_values, 0.0);
  cuda_check(cudaMemcpy(host_diff_E.data(),
                        diff_E.as<double>(),
                        sizeof(double) * host_diff_E.size(),
                        cudaMemcpyDeviceToHost),
             "compute_diffusion_energy_host copy diff_E failed");

  long double sum = 0.0L;
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (diff_cell[c] == 0U) {
      continue;
    }
    const double V = std::max(vol[c], 0.0);
    for (std::size_t g = 0; g < n_groups; ++g) {
      sum += static_cast<long double>(std::max(host_diff_E[c * n_groups + g], 0.0)) *
             static_cast<long double>(V);
    }
  }
  return static_cast<double>(sum);
}

HoloLOResult solve_holo_lo_source_ownership(core::State& state,
                                            const core::Config& cfg,
                                            const PlanckTable& planck,
                                            const core::Config::MaterialsConfig::MatDef& mat,
                                            const double* d_sigma_P,
                                            const double* d_sigma_R,
                                            const double* d_fleck,
                                            const int n_cells,
                                            const int n_groups,
                                            const double dt,
                                            const double* d_E_initial,
                                            const std::size_t E_initial_bytes,
                                            const double* d_consistency_source,
                                            const std::size_t consistency_source_bytes,
                                            const double* d_face_current_step,
                                            const std::size_t face_current_step_bytes,
                                            const double* d_reference_face_current,
                                            const std::size_t reference_face_current_bytes,
                                            const bool commit_material,
                                            const bool publish_source_diagnostics,
                                            const bool use_full_mesh_lo_solve,
                                            const bool has_physical_outer_vacuum,
                                            const char* solve_phase,
                                            const std::vector<double>* precomputed_chi) {
  HoloLOResult result{};
  const std::size_t n_cells_us = static_cast<std::size_t>(std::max(n_cells, 0));
  const std::size_t n_groups_us = static_cast<std::size_t>(std::max(n_groups, 0));
  const std::size_t n_total = n_cells_us * n_groups_us;
  const std::size_t n_faces = n_cells_us + 1U;
  const std::size_t n_face_groups = n_faces * n_groups_us;
  const bool use_qd_solver = (cfg.radiation.holo.solver == "quasidiffusion_1d");
  state.holo_lo_source_valid = false;
  if (state.holo_rad_dep.size() != n_total) {
    state.holo_rad_dep.reset(n_total);
  }
  if (state.holo_rad_emit.size() != n_total) {
    state.holo_rad_emit.reset(n_total);
  }
  state.holo_rad_dep.fill(0.0);
  state.holo_rad_emit.fill(0.0);

  if (!cfg.radiation.holo.enabled || state.mesh.dim != 1 ||
      !state.holo_core_mask_valid || n_cells <= 0 || n_groups <= 0 || !(dt > 0.0)) {
    return result;
  }
  TENRYU_ASSERT(state.holo_core_mask.size() == n_cells_us,
                "solve_holo_lo_source_ownership requires LO coupling mask size match");
  TENRYU_ASSERT(state.holo_patch_mask.size() == n_cells_us,
                "solve_holo_lo_source_ownership requires LO patch mask size match");
  TENRYU_ASSERT(state.holo_lo_weight.size() == n_cells_us,
                "solve_holo_lo_source_ownership requires LO weight size match");

  TENRYU_ASSERT(d_sigma_P != nullptr,
                "solve_holo_lo_source_ownership requires Planck opacity");
  TENRYU_ASSERT(d_sigma_R != nullptr,
                "solve_holo_lo_source_ownership requires Rosseland opacity");
  TENRYU_ASSERT(state.holo_E_LO.size() == n_total,
                "solve_holo_lo_source_ownership requires holo_E_LO size match");
  TENRYU_ASSERT(state.holo_consistency_source.size() == n_total,
                "solve_holo_lo_source_ownership requires holo_consistency_source size match");
  TENRYU_ASSERT(state.ee.size() == n_cells_us,
                "solve_holo_lo_source_ownership requires ee size match");
  TENRYU_ASSERT(state.Te.size() == n_cells_us,
                "solve_holo_lo_source_ownership requires Te size match");
  TENRYU_ASSERT(state.Pe.size() == n_cells_us,
                "solve_holo_lo_source_ownership requires Pe size match");
  TENRYU_ASSERT(state.rho.size() == n_cells_us,
                "solve_holo_lo_source_ownership requires rho size match");
  TENRYU_ASSERT(state.mass.size() == n_cells_us,
                "solve_holo_lo_source_ownership requires mass size match");
  TENRYU_ASSERT(state.vol.size() == n_cells_us,
                "solve_holo_lo_source_ownership requires vol size match");
  TENRYU_ASSERT(state.x_r.size() == n_cells_us + 1U,
                "solve_holo_lo_source_ownership requires 1D node size match");
  TENRYU_ASSERT(state.cell_is_void.size() == n_cells_us,
                "solve_holo_lo_source_ownership requires cell_is_void size match");

  std::vector<double> E_lo(n_total, 0.0);
  std::vector<double> ee(n_cells_us, 0.0);
  std::vector<double> Te(n_cells_us, 0.0);
  std::vector<double> Pe(n_cells_us, 0.0);
  std::vector<double> sigma_P(n_total, 0.0);
  std::vector<double> sigma_R(n_total, 0.0);
  std::vector<double> rho(n_cells_us, 0.0);
  std::vector<double> mass(n_cells_us, 0.0);
  std::vector<double> vol(n_cells_us, 0.0);
  std::vector<double> node_r(n_cells_us + 1U, 0.0);
  std::vector<double> rad_dep_lo(n_total, 0.0);
  std::vector<double> rad_emit_lo(n_total, 0.0);
  std::vector<double> matter_delta_lo_cell(n_cells_us, 0.0);
  std::vector<double> consistency_source;
  std::vector<double> face_current;
  std::vector<double> fleck_f;
  std::vector<double> chi_host;
  std::vector<double> F_lo;
  std::vector<double> cv_e;

  const std::size_t expected_E_bytes = sizeof(double) * n_total;
  if (d_E_initial != nullptr) {
    TENRYU_ASSERT(E_initial_bytes == expected_E_bytes,
                  "solve_holo_lo_source_ownership E_initial size mismatch");
    cuda_check(cudaMemcpy(E_lo.data(),
                          d_E_initial,
                          expected_E_bytes,
                          cudaMemcpyDeviceToHost),
               "solve_holo_lo_source_ownership copy E_initial failed");
  } else if (state.rad_E.size() == n_total) {
    state.rad_E.copy_to_host(E_lo.data());
  } else {
    state.holo_E_LO.copy_to_host(E_lo.data());
  }
  state.ee.copy_to_host(ee.data());
  state.Te.copy_to_host(Te.data());
  state.Pe.copy_to_host(Pe.data());
  state.rho.copy_to_host(rho.data());
  state.mass.copy_to_host(mass.data());
  state.vol.copy_to_host(vol.data());
  state.x_r.copy_to_host(node_r.data());
  cuda_check(cudaMemcpy(sigma_P.data(),
                        d_sigma_P,
                        sizeof(double) * n_total,
                        cudaMemcpyDeviceToHost),
             "solve_holo_lo_source_ownership copy sigma_P failed");
  cuda_check(cudaMemcpy(sigma_R.data(),
                        d_sigma_R,
                        sizeof(double) * n_total,
                        cudaMemcpyDeviceToHost),
             "solve_holo_lo_source_ownership copy sigma_R failed");
  if (d_consistency_source != nullptr) {
    TENRYU_ASSERT(consistency_source_bytes == expected_E_bytes,
                  "solve_holo_lo_source_ownership consistency source size mismatch");
    consistency_source.assign(n_total, 0.0);
    cuda_check(cudaMemcpy(consistency_source.data(),
                          d_consistency_source,
                          expected_E_bytes,
                          cudaMemcpyDeviceToHost),
               "solve_holo_lo_source_ownership copy consistency_source failed");
  }
  const std::size_t expected_face_current_bytes =
      sizeof(double) * (n_cells_us + 1U) * n_groups_us;
  if (d_face_current_step != nullptr) {
    TENRYU_ASSERT(face_current_step_bytes == expected_face_current_bytes,
                  "solve_holo_lo_source_ownership face current size mismatch");
    face_current.assign((n_cells_us + 1U) * n_groups_us, 0.0);
    cuda_check(cudaMemcpy(face_current.data(),
                          d_face_current_step,
                          expected_face_current_bytes,
                          cudaMemcpyDeviceToHost),
               "solve_holo_lo_source_ownership copy face_current_step failed");
  }
  if (d_reference_face_current != nullptr) {
    TENRYU_ASSERT(reference_face_current_bytes == expected_face_current_bytes,
                  "solve_holo_lo_source_ownership reference face current size mismatch");
    if (face_current.empty()) {
      face_current.assign((n_cells_us + 1U) * n_groups_us, 0.0);
    }
    std::vector<double> reference_face_current((n_cells_us + 1U) * n_groups_us,
                                               0.0);
    cuda_check(cudaMemcpy(reference_face_current.data(),
                          d_reference_face_current,
                          expected_face_current_bytes,
                          cudaMemcpyDeviceToHost),
               "solve_holo_lo_source_ownership copy reference_face_current failed");
    for (std::size_t i = 0; i < reference_face_current.size(); ++i) {
      if (std::isfinite(reference_face_current[i])) {
        face_current[i] += reference_face_current[i];
      }
    }
  }
  if (use_qd_solver) {
    if (precomputed_chi != nullptr) {
      TENRYU_ASSERT(precomputed_chi->size() == n_total,
                    "solve_holo_lo_source_ownership chi override size mismatch");
      chi_host = *precomputed_chi;
      regularize_holo_qd_closure_chi(
          state, cfg, n_cells, n_groups, commit_material, chi_host);
    } else if (cfg.radiation.holo.sn_closure) {
      fleck_f.assign(n_cells_us, 1.0);
      if (d_fleck != nullptr && n_cells_us > 0U) {
        cuda_check(cudaMemcpy(fleck_f.data(),
                              d_fleck,
                              sizeof(double) * n_cells_us,
                              cudaMemcpyDeviceToHost),
                   "solve_holo_lo_source_ownership copy fleck factor failed");
      }
      prepare_holo_sn_closure_chi(state,
                                  cfg,
                                  planck,
                                  sigma_P,
                                  fleck_f,
                                  Te,
                                  node_r,
                                  vol,
                                  n_cells,
                                  n_groups,
                                  commit_material,
                                  chi_host);
    } else {
      prepare_holo_qd_closure_chi(
          state, cfg, n_cells, n_groups, commit_material, chi_host);
    }
    F_lo.assign(n_face_groups, 0.0);
    if (state.holo_F_LO.size() == n_face_groups) {
      state.holo_F_LO.copy_to_host(F_lo.data());
    }
  }
  if (!state.cv_e.empty()) {
    TENRYU_ASSERT(state.cv_e.size() == n_cells_us,
                  "solve_holo_lo_source_ownership requires cv_e size match");
    cv_e.assign(n_cells_us, 0.0);
    state.cv_e.copy_to_host(cv_e.data());
  }

  const double A = std::max(mat.A, 1.0e-12);
  const double gm1 = std::max(mat.ideal_gas_gamma - 1.0, 1.0e-12);
  HoloLOInputs in{};
  in.dimension = HoloGeometryDimension::Spherical1D;
  in.E_lo = E_lo.data();
  in.ee = ee.data();
  in.Te = Te.data();
  in.Pe = Pe.data();
  in.sigma_P = sigma_P.data();
  in.sigma_R = sigma_R.data();
  in.chi = chi_host.empty() ? nullptr : chi_host.data();
  in.F_lo = F_lo.empty() ? nullptr : F_lo.data();
  in.rho = rho.data();
  in.mass = mass.data();
  in.vol = vol.data();
  in.node_r = node_r.data();
  in.lo_coupled = commit_material ? state.holo_core_mask.data() : nullptr;
  in.cell_active = use_full_mesh_lo_solve ? nullptr : state.holo_patch_mask.data();
  in.lo_weight = state.holo_lo_weight.data();
  in.consistency_source =
      consistency_source.empty() ? nullptr : consistency_source.data();
  in.consistency_alpha = cfg.radiation.holo.consistency_alpha;
  in.face_current = face_current.empty() ? nullptr : face_current.data();
  in.face_current_dt = dt;
  in.has_physical_outer_vacuum = has_physical_outer_vacuum;
  in.rad_dep_lo = rad_dep_lo.data();
  in.rad_emit_lo = rad_emit_lo.data();
  in.matter_delta_lo_cell = matter_delta_lo_cell.data();
  in.cv_e = cv_e.empty() ? nullptr : cv_e.data();
  in.planck = &planck;
  in.n_cells = n_cells;
  in.n_groups = n_groups;
  in.dt = dt;
  in.cv_e_const =
      core::constants::eV_to_erg / (A * core::constants::proton_mass * gm1);
  in.pressure_gamma_minus_one = gm1;
  in.temperature_floor_eV = cfg.numerics.floors.Te;
  const bool use_table_eos =
      mat.eos_tables != nullptr && mat.hydro_eos_backend != "exact_ideal_gas";
  if (use_table_eos) {
    in.material_model = HoloLOMaterialModel::TableEOS;
    in.electron_eos = &mat.eos_tables->electron;
  }

  result = solve_holo_lo_1d_cpu(in, use_qd_solver);
  if (result.failures != 0) {
    std::ostringstream oss;
    oss << "[holo_lo] step=" << state.step
        << " phase=" << ((solve_phase != nullptr) ? solve_phase : "solve")
        << " LO solve failed failures=" << result.failures
        << " stage=" << holo_lo_failure_stage_name(result.failure_stage)
        << " cell=" << result.failure_cell
        << " group=" << result.failure_group
        << " E_sum=" << result.failure_E_sum
        << " T=" << result.failure_T
        << " residual=" << result.failure_residual
        << "; falling back to particle material coupling for LO-coupled cells";
    core::log_warning(oss.str());
    const HoloLOResult failed_result = result;
    result = HoloLOResult{};
    result.failures = failed_result.failures;
    result.solver_iterations = failed_result.solver_iterations;
    result.failure_stage = failed_result.failure_stage;
    result.failure_cell = failed_result.failure_cell;
    result.failure_group = failed_result.failure_group;
    result.failure_E_sum = failed_result.failure_E_sum;
    result.failure_T = failed_result.failure_T;
    result.failure_residual = failed_result.failure_residual;
    return result;
  }

  if (use_qd_solver && commit_material && !chi_host.empty()) {
    if (state.holo_chi_filtered.size() != n_total) {
      state.holo_chi_filtered.reset(n_total);
    }
    state.holo_chi_filtered.copy_from_host(chi_host.data());
  }
  state.holo_E_LO.copy_from_host(E_lo.data());
  if (use_qd_solver) {
    if (state.holo_F_LO.size() != n_face_groups) {
      state.holo_F_LO.reset(n_face_groups);
    }
    state.holo_F_LO.copy_from_host(F_lo.data());
  }
  if (commit_material) {
    state.ee.copy_from_host(ee.data());
    state.Te.copy_from_host(Te.data());
    state.Pe.copy_from_host(Pe.data());
  }
  if (publish_source_diagnostics) {
    state.holo_rad_dep.copy_from_host(rad_dep_lo.data());
    state.holo_rad_emit.copy_from_host(rad_emit_lo.data());
    state.holo_lo_source_valid = true;
  }
  return result;
}

void log_diffusion_step_balance_if_bad(const std::int64_t step,
                                       const double E_before,
                                       const double E_face_in,
                                       const double E_face_out,
                                       const double E_vacuum,
                                       const double E_source_exchange,
                                       const double E_after) {
  const double expected =
      E_before + E_face_in - E_face_out - E_vacuum - E_source_exchange;
  const double residual = expected - E_after;
  const double scale = std::max({std::fabs(E_before),
                                 std::fabs(E_face_in),
                                 std::fabs(E_face_out),
                                 std::fabs(E_vacuum),
                                 std::fabs(E_source_exchange),
                                 std::fabs(E_after),
                                 kDiffusionBalanceFloor});
  if (std::isfinite(residual) && std::isfinite(scale) &&
      std::fabs(residual) <= kDiffusionBalanceRelTol * scale) {
    return;
  }
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(6);
  oss << "[diffusion_balance] critical step=" << step
      << " E_before=" << E_before
      << " E_face_in=" << E_face_in
      << " E_face_out=" << E_face_out
      << " E_vacuum=" << E_vacuum
      << " E_source_exchange=" << E_source_exchange
      << " E_after=" << E_after
      << " expected=" << expected
      << " residual=" << residual
      << " rel=" << (std::fabs(residual) / scale);
  core::log_fatal(oss.str());
}

void log_diffusion_substep_energy(const std::int64_t step,
                                  const char* name,
                                  const double E_before,
                                  const double E_after,
                                  const double E_face_in,
                                  const double E_face_out,
                                  const double E_vacuum,
                                  const double E_source_exchange,
                                  const int failures) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(6);
  oss << "[diffusion_substep] step=" << step
      << " name=" << name
      << " E_before=" << E_before
      << " E_after=" << E_after
      << " dE=" << (E_after - E_before)
      << " E_face_in=" << E_face_in
      << " E_face_out=" << E_face_out
      << " E_vacuum=" << E_vacuum
      << " E_source_exchange=" << E_source_exchange
      << " failures=" << failures;
  core::log_info(oss.str());
}

double diffusion_growth_reference(const double E_before,
                                  const double E_face_in,
                                  const double E_face_out,
                                  const double E_vacuum,
                                  const double E_source_exchange) {
  const double expected =
      E_before + E_face_in - E_face_out - E_vacuum - E_source_exchange;
  return std::max({std::fabs(E_before),
                   std::fabs(E_face_in),
                   std::fabs(E_face_out),
                   std::fabs(E_vacuum),
                   std::fabs(E_source_exchange),
                   std::fabs(expected),
                   kDiffusionRadiationEnergyThreshold});
}

bool diffusion_energy_growth_exceeded(const double reference,
                                      const double E_after) {
  return std::isfinite(reference) && std::isfinite(E_after) &&
         E_after > kDiffusionEmergencyGrowthFactor *
                       std::max(reference, kDiffusionBalanceFloor);
}

int boundary_code_from_string(const std::string& mode) {
  if (mode == "vacuum") {
    return kBoundaryVacuum;
  }
  if (mode == "marshak") {
    return kBoundaryMarshak;
  }
  return kBoundaryReflect;
}

int diffusion_boundary_code_from_string(const std::string& mode) {
  return (mode == "vacuum" || mode == "marshak") ? 1 : 0;
}

DDMCBoundaryType ddmc_boundary_type_from_string(const std::string& mode) {
  if (mode == "reflect") {
    return DDMCBoundaryType::Reflective;
  }
  if (mode == "vacuum" || mode == "marshak") {
    return DDMCBoundaryType::Vacuum;
  }
  return DDMCBoundaryType::Reflective;
}

bool is_supported_2d_bc_outer(const std::string& mode) {
  return mode == "vacuum" || mode == "reflect";
}

bool is_supported_2d_bc_z(const std::string& mode) {
  return mode == "vacuum" || mode == "reflect" || mode == "marshak";
}

bool has_marshak_boundary(const core::Config::RadiationConfig::BoundaryConfig& boundary,
                          const int mesh_dim) {
  if (mesh_dim == 1) {
    return boundary.inner_r == "marshak" || boundary.outer_r == "marshak";
  }
  return boundary.inner_r == "marshak" || boundary.outer_r == "marshak" ||
         boundary.bottom_z == "marshak" || boundary.top_z == "marshak";
}

std::string implicit_ddmc_diffusion_unsupported_reason(const core::Config& cfg,
                                                       const int mesh_dim,
                                                       const bool is_nlte_mode) {
  if (mesh_dim != 1) {
    return "implicit DDMC diffusion is 1D-only in Phase-1";
  }
  if (is_nlte_mode) {
    return "implicit DDMC diffusion is limited to LTE opacity mode";
  }
  if (has_marshak_boundary(cfg.radiation.boundary, mesh_dim)) {
    return "implicit DDMC diffusion does not support Marshak boundaries";
  }
  if (cfg.radiation.volume_source_rate > 0.0) {
    return "implicit DDMC diffusion does not support volume_source_rate > 0";
  }
  return {};
}

void log_imc_device_flags(const core::DeviceErrorFlags& flags) {
  if (flags.infinite_loop != 0) {
    core::log_warning("IMC transport: infinite-loop guard triggered (count=" +
                      std::to_string(flags.infinite_loop) + ")");
  }
  if (flags.invalid_cell != 0) {
    core::log_warning("IMC transport: invalid cell encountered");
  }
  if (flags.nan_particle != 0) {
    core::log_warning("IMC transport: non-finite particle state encountered");
  }
  if (flags.invalid_boundary != 0) {
    core::log_warning("IMC transport: invalid boundary encountered");
  }
  if (flags.pool_overflow != 0) {
    core::log_warning("IMC transport: particle-pool overflow reported");
  }
  if (flags.opacity_out_of_range != 0) {
    core::log_warning("IMC transport: opacity out-of-range reported");
  }
  if (flags.ddmc_sigma_tot_zero != 0) {
    core::log_warning("IMC transport: DDMC sigma_tot was zero");
  }
  if (flags.roulette_kill != 0) {
    core::log_debug("IMC transport: roulette kill occurred");
  }
}

void fatal_on_excess_infinite_loop(const core::DeviceErrorFlags& flags,
                                   const char* context) {
  if (flags.infinite_loop <= kInfiniteLoopFatalThreshold) {
    return;
  }
  std::ostringstream oss;
  oss << "IMC transport: infinite-loop guard exceeded fatal threshold at " << context
      << " (count=" << flags.infinite_loop
      << ", threshold=" << kInfiniteLoopFatalThreshold << ")";
  TENRYU_ASSERT(false, oss.str());
}

void check_device_flags_after_stage(const core::DeviceErrorFlags* d_error_flags,
                                    const char* stage_name) {
  if (d_error_flags == nullptr) {
    return;
  }
  core::DeviceErrorFlags stage_flags{};
  cuda_check(cudaMemcpy(&stage_flags,
                        d_error_flags,
                        sizeof(stage_flags),
                        cudaMemcpyDeviceToHost),
             "IMC::transport_step copy stage error flags failed");
  const bool has_serious_flags =
      stage_flags.infinite_loop != 0 || stage_flags.invalid_cell != 0 ||
      stage_flags.nan_particle != 0 || stage_flags.invalid_boundary != 0 ||
      stage_flags.pool_overflow != 0 || stage_flags.opacity_out_of_range != 0 ||
      stage_flags.ddmc_sigma_tot_zero != 0;
  if (has_serious_flags) {
    core::log_warning(std::string("IMC transport stage '") + stage_name +
                      "' reported device flags");
    log_imc_device_flags(stage_flags);
  } else if (stage_flags.roulette_kill != 0) {
    core::log_debug(std::string("IMC transport stage '") + stage_name +
                    "' reported roulette kill");
  }
  fatal_on_excess_infinite_loop(stage_flags, stage_name);
  if (stage_flags.invalid_cell != 0 || stage_flags.nan_particle != 0) {
    TENRYU_ASSERT(
        false,
        "IMC transport: fatal device error flag set after kernel stage");
  }
}

}  // namespace

void IMC::ensure_diffusion_face_current_buffers(const int n_cells, const int n_groups) {
  const std::size_t n_faces =
      (n_cells >= 0) ? (static_cast<std::size_t>(n_cells) + 1U) : 0U;
  const std::size_t n_values = n_faces * static_cast<std::size_t>(std::max(n_groups, 0));
  const std::size_t bytes = sizeof(double) * n_values;
  const bool resize_prev = (face_current_prev_.size != bytes);
  const bool resize_step = (face_current_step_.size != bytes);
  const bool resize_in = (face_current_in_.size != bytes);
  const bool resize_out = (face_current_out_.size != bytes);
  face_current_prev_.resize(bytes);
  face_current_step_.resize(bytes);
  face_current_in_.resize(bytes);
  face_current_out_.resize(bytes);
  if (resize_prev && bytes > 0U) {
    cuda_check(cudaMemset(face_current_prev_.ptr, 0, bytes),
               "IMC::ensure_diffusion_face_current_buffers zero prev failed");
    face_current_prev_dt_ = 0.0;
  }
  if (resize_step && bytes > 0U) {
    cuda_check(cudaMemset(face_current_step_.ptr, 0, bytes),
               "IMC::ensure_diffusion_face_current_buffers zero step failed");
  }
  if (resize_in && bytes > 0U) {
    cuda_check(cudaMemset(face_current_in_.ptr, 0, bytes),
               "IMC::ensure_diffusion_face_current_buffers zero in failed");
  }
  if (resize_out && bytes > 0U) {
    cuda_check(cudaMemset(face_current_out_.ptr, 0, bytes),
               "IMC::ensure_diffusion_face_current_buffers zero out failed");
  }
}

void IMC::ensure_diffusion_energy_buffer(const int n_cells, const int n_groups) {
  TENRYU_ASSERT(n_cells >= 0, "IMC::ensure_diffusion_energy_buffer n_cells >= 0");
  TENRYU_ASSERT(n_groups > 0, "IMC::ensure_diffusion_energy_buffer n_groups > 0");
  const std::size_t n_total = static_cast<std::size_t>(n_cells) *
                              static_cast<std::size_t>(n_groups);
  const std::size_t bytes = sizeof(double) * n_total;
  const bool resize = (diff_E_.size != bytes);
  diff_E_.resize(bytes);
  if (resize && bytes > 0U) {
    cuda_check(cudaMemset(diff_E_.ptr, 0, bytes),
               "IMC::ensure_diffusion_energy_buffer zero diff_E failed");
  }
}

void IMC::classify_diffusion_cells(core::State& state,
                                   const core::Config& cfg,
                                   const PlanckTable& planck,
                                   const std::vector<double>& sigma_R,
                                   const std::vector<double>& node_r,
                                   const int n_cells,
                                   const int n_groups,
                                   const double dt) {
  const std::size_t n_cells_us = static_cast<std::size_t>(std::max(n_cells, 0));
  const std::size_t n_groups_us = static_cast<std::size_t>(std::max(n_groups, 0));
  const std::size_t n_cell_groups_us = n_cells_us * n_groups_us;
  TENRYU_ASSERT(sigma_R.size() == n_cell_groups_us,
                "IMC::classify_diffusion_cells sigma_R size mismatch");
  TENRYU_ASSERT(node_r.size() == n_cells_us + 1U,
                "IMC::classify_diffusion_cells node_r size mismatch");
  TENRYU_ASSERT(planck.n_groups() == n_groups,
                "IMC::classify_diffusion_cells group count mismatch");

  const bool had_previous_mask =
      diff_cell_.size() == n_cells_us && diff_cell_prev_.size() == n_cells_us;
  if (diff_cell_.size() != n_cells_us) {
    diff_cell_.assign(n_cells_us, 0U);
    diff_cell_prev_.assign(n_cells_us, 0U);
    diff_hold_.assign(n_cells_us, 0U);
    diff_guard_cell_.assign(n_cells_us, 0U);
    diff_tau_R_.assign(n_cells_us, 0.0);
    diff_reduced_flux_.assign(n_cells_us, 0.0);
  } else {
    diff_cell_.assign(n_cells_us, 0U);
    diff_guard_cell_.assign(n_cells_us, 0U);
  }

  std::vector<double> host_Te(n_cells_us, 0.0);
  if (n_cells > 0) {
    state.Te.copy_to_host(host_Te.data());
  }
  std::vector<double> host_rad_E(n_cell_groups_us, 0.0);
  if (state.rad_E.size() == n_cell_groups_us && n_cell_groups_us > 0U) {
    state.rad_E.copy_to_host(host_rad_E.data());
  }

  const std::size_t n_face_current = (n_cells_us + 1U) * n_groups_us;
  const std::size_t current_bytes = sizeof(double) * n_face_current;
  std::vector<double> host_face_current(n_face_current, 0.0);
  const bool have_prev_current =
      face_current_prev_.ptr != nullptr && face_current_prev_.size == current_bytes &&
      face_current_prev_dt_ > 0.0;
  if (have_prev_current && current_bytes > 0U) {
    cuda_check(cudaMemcpy(host_face_current.data(),
                          face_current_prev_.ptr,
                          current_bytes,
                          cudaMemcpyDeviceToHost),
               "IMC::classify_diffusion_cells copy face current failed");
  }

  constexpr double kSigmaFloor = 1.0e-30;
  const auto& dc = cfg.radiation.diffusion;
  const int hold_required = std::max(dc.mode_hold, 0);
  const double rate_max = std::max(dc.rate_max, 0.0);
  const int mode_update_interval = std::max(dc.mode_update_interval, 1);
  const bool allow_new_entry =
      !had_previous_mask || state.step == 0 ||
      (state.step % mode_update_interval == 0);
  const int min_island_cells = std::max(dc.min_diffusion_island_cells, 1);
  const double te_floor = std::max(cfg.numerics.floors.Te, 1.0e-12);
  const double e_floor =
      core::constants::a_eV * te_floor * te_floor * te_floor * te_floor;
  const double flux_dt = have_prev_current ? face_current_prev_dt_ : dt;
  const bool use_face_current = have_prev_current && flux_dt > 0.0;

  int n_diff = 0;
  int n_guard = 0;
  bool have_tau_diag = false;
  int n_rad_energy_empty = 0;
  int n_island_rejected = 0;
  double tau_min = std::numeric_limits<double>::infinity();
  double tau_max = 0.0;
  double rf_max = 0.0;

  for (int c = 0; c < n_cells; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    const std::size_t base = c_us * n_groups_us;
    const bool is_void = (c_us < state.cell_is_void.size() &&
                          state.cell_is_void[c_us] != 0U);

    const double dx = std::max(node_r[c_us + 1U] - node_r[c_us], 0.0);
    const double sigma_mean =
        temperature_weighted_rosseland_sigma_R(
            planck, sigma_R, base, n_groups, host_Te[c_us], cfg.numerics.floors.Te);
    const double tau_R = sigma_mean * dx;

    double E_rad_total = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      E_rad_total += std::max(host_rad_E[base + static_cast<std::size_t>(g)], 0.0);
    }
    const double E_for_flux = std::max(E_rad_total, e_floor);

    double reduced_flux = 0.0;
    if (use_face_current) {
      double current_left = 0.0;
      double current_right = 0.0;
      const std::size_t left_face = c_us;
      const std::size_t right_face = c_us + 1U;
      for (int g = 0; g < n_groups; ++g) {
        current_left += host_face_current[left_face * n_groups_us +
                                          static_cast<std::size_t>(g)];
        current_right += host_face_current[right_face * n_groups_us +
                                           static_cast<std::size_t>(g)];
      }
      const double r_left = std::max(node_r[left_face], 0.0);
      const double r_right = std::max(node_r[right_face], 0.0);
      const double area_left = 4.0 * kPi * r_left * r_left;
      const double area_right = 4.0 * kPi * r_right * r_right;
      const double F_left =
          (area_left > 0.0) ? (current_left / (area_left * flux_dt)) : 0.0;
      const double F_right =
          (area_right > 0.0) ? (current_right / (area_right * flux_dt)) : 0.0;
      reduced_flux =
          std::max(std::abs(F_left), std::abs(F_right)) /
          (core::constants::c_light * E_for_flux);
    }

    const double prev_tau = diff_tau_R_[c_us];
    diff_tau_R_[c_us] = tau_R;
    diff_reduced_flux_[c_us] = reduced_flux;
    if (is_void) {
      diff_cell_[c_us] = 0U;
      diff_hold_[c_us] = 0U;
      continue;
    }

    have_tau_diag = true;
    tau_min = std::min(tau_min, tau_R);
    tau_max = std::max(tau_max, tau_R);
    rf_max = std::max(rf_max, reduced_flux);

    if (!(E_rad_total > kDiffusionRadiationEnergyThreshold)) {
      diff_cell_[c_us] = 0U;
      diff_hold_[c_us] = 0U;
      ++n_rad_energy_empty;
      continue;
    }

    const bool was_diff = (diff_cell_prev_[c_us] != 0U);
    const bool hard_exit =
        tau_R < dc.tau_off || reduced_flux > dc.reduced_flux_off ||
        tau_R < 0.5 * dc.tau_off;
    if (was_diff && hard_exit) {
      diff_cell_[c_us] = 0U;
      diff_hold_[c_us] = 0U;
      continue;
    }
    if (!allow_new_entry && !was_diff) {
      diff_cell_[c_us] = 0U;
      continue;
    }
    if (!was_diff) {
      const double tau_rate =
          (prev_tau > kSigmaFloor) ? (std::abs(tau_R - prev_tau) / prev_tau) : 0.0;
      const bool rate_ok = (rate_max <= 0.0) || (tau_rate <= rate_max);
      if (tau_R >= dc.tau_on && reduced_flux <= dc.reduced_flux_on && rate_ok) {
        if (diff_hold_[c_us] >= static_cast<std::uint8_t>(std::min(hold_required, 255))) {
          diff_cell_[c_us] = 1U;
          diff_hold_[c_us] = 0U;
        } else {
          diff_hold_[c_us] =
              static_cast<std::uint8_t>(std::min<int>(diff_hold_[c_us] + 1, 255));
        }
      } else {
        diff_hold_[c_us] = 0U;
      }
    } else {
      diff_cell_[c_us] = 1U;
      diff_hold_[c_us] =
          static_cast<std::uint8_t>(std::min<int>(diff_hold_[c_us] + 1, 255));
    }
  }

  if (allow_new_entry && min_island_cells > 1) {
    int run_start = -1;
    for (int c = 0; c <= n_cells; ++c) {
      const bool is_diff =
          (c < n_cells) && diff_cell_[static_cast<std::size_t>(c)] != 0U;
      if (is_diff && run_start < 0) {
        run_start = c;
      }
      if (!is_diff && run_start >= 0) {
        const int run_len = c - run_start;
        if (run_len < min_island_cells) {
          for (int j = run_start; j < c; ++j) {
            const std::size_t j_us = static_cast<std::size_t>(j);
            if (diff_cell_[j_us] != 0U) {
              diff_cell_[j_us] = 0U;
              diff_hold_[j_us] = 0U;
              ++n_island_rejected;
            }
          }
        }
        run_start = -1;
      }
    }
  }

  int n_entered = 0;
  int n_exited = 0;
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    if (diff_cell_[c_us] != 0U && diff_cell_prev_[c_us] == 0U) {
      ++n_entered;
    } else if (diff_cell_[c_us] == 0U && diff_cell_prev_[c_us] != 0U) {
      ++n_exited;
    }
  }

  const int guard_cells = std::max(dc.imc_guard_cells, 1);
  for (int c = 0; c < n_cells; ++c) {
    if (diff_cell_[static_cast<std::size_t>(c)] == 0U) {
      continue;
    }
    ++n_diff;
    const int j_min = std::max(0, c - guard_cells);
    const int j_max = std::min(n_cells - 1, c + guard_cells);
    for (int j = j_min; j <= j_max; ++j) {
      const std::size_t j_us = static_cast<std::size_t>(j);
      if (j == c || diff_cell_[j_us] != 0U || diff_guard_cell_[j_us] != 0U) {
        continue;
      }
      if (j_us < state.cell_is_void.size() && state.cell_is_void[j_us] != 0U) {
        continue;
      }
      diff_guard_cell_[j_us] = 1U;
      ++n_guard;
    }
  }

  if (!have_tau_diag) {
    tau_min = 0.0;
  }
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(6);
  oss << "[diffusion_classify] step=" << state.step
      << " n_diff=" << n_diff
      << " n_guard=" << n_guard
      << " n_entered=" << n_entered
      << " n_exited=" << n_exited
      << " n_island_rejected=" << n_island_rejected
      << " mode_update=" << (allow_new_entry ? 1 : 0)
      << " n_cells=" << n_cells
      << " n_E_empty=" << n_rad_energy_empty
      << " tau_R_min=" << tau_min
      << " tau_R_max=" << tau_max
      << " RF_max=" << rf_max;
  core::log_info(oss.str());
}

IMC::DeviceOpacityTable::DeviceOpacityTable(DeviceOpacityTable&& other) noexcept {
  *this = std::move(other);
}

IMC::DeviceOpacityTable& IMC::DeviceOpacityTable::operator=(
    DeviceOpacityTable&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  d_log_temps = std::exchange(other.d_log_temps, nullptr);
  d_log_numdens = std::exchange(other.d_log_numdens, nullptr);
  d_kappa_PA = std::exchange(other.d_kappa_PA, nullptr);
  d_kappa_PE = std::exchange(other.d_kappa_PE, nullptr);
  d_kappa_R = std::exchange(other.d_kappa_R, nullptr);
  ntemp = std::exchange(other.ntemp, 0);
  ndens = std::exchange(other.ndens, 0);
  ngroups = std::exchange(other.ngroups, 0);
  log_T_min = std::exchange(other.log_T_min, 0.0);
  log_T_max = std::exchange(other.log_T_max, 0.0);
  log_ni_min = std::exchange(other.log_ni_min, 0.0);
  log_ni_max = std::exchange(other.log_ni_max, 0.0);
  T_min = std::exchange(other.T_min, 0.0);
  T_max = std::exchange(other.T_max, 0.0);
  host_generation = std::exchange(other.host_generation, 0);
  return *this;
}

void IMC::DeviceOpacityTable::upload(const materials::IonmixOpacityData& host) {
  TENRYU_ASSERT(host.ntemp > 0, "IMC::DeviceOpacityTable::upload requires host.ntemp > 0");
  TENRYU_ASSERT(host.ndens > 0, "IMC::DeviceOpacityTable::upload requires host.ndens > 0");
  TENRYU_ASSERT(host.ngroups > 0,
                "IMC::DeviceOpacityTable::upload requires host.ngroups > 0");
  TENRYU_ASSERT(host.log_temps.size() == static_cast<std::size_t>(host.ntemp),
                "IMC::DeviceOpacityTable::upload log_temps size mismatch");
  TENRYU_ASSERT(host.log_numdens.size() == static_cast<std::size_t>(host.ndens),
                "IMC::DeviceOpacityTable::upload log_numdens size mismatch");
  const std::size_t n_table =
      static_cast<std::size_t>(host.ngroups) * static_cast<std::size_t>(host.ndens) *
      static_cast<std::size_t>(host.ntemp);
  TENRYU_ASSERT(host.kappa_PA.size() == n_table,
                "IMC::DeviceOpacityTable::upload kappa_PA size mismatch");
  TENRYU_ASSERT(host.kappa_PE.size() == n_table,
                "IMC::DeviceOpacityTable::upload kappa_PE size mismatch");
  TENRYU_ASSERT(host.kappa_R.size() == n_table,
                "IMC::DeviceOpacityTable::upload kappa_R size mismatch");
  TENRYU_ASSERT(!host.temps_eV.empty(),
                "IMC::DeviceOpacityTable::upload requires non-empty temperatures");
  TENRYU_ASSERT(!host.numdens_cm3.empty(),
                "IMC::DeviceOpacityTable::upload requires non-empty number densities");

  release();

  const auto upload_vector = [](double** dst,
                                const std::vector<double>& src,
                                const char* malloc_message,
                                const char* copy_message) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(dst), sizeof(double) * src.size()),
               malloc_message);
    cuda_check(cudaMemcpy(*dst,
                          src.data(),
                          sizeof(double) * src.size(),
                          cudaMemcpyHostToDevice),
               copy_message);
  };

  upload_vector(&d_log_temps,
                host.log_temps,
                "IMC::DeviceOpacityTable cudaMalloc log_temps failed",
                "IMC::DeviceOpacityTable copy log_temps failed");
  upload_vector(&d_log_numdens,
                host.log_numdens,
                "IMC::DeviceOpacityTable cudaMalloc log_numdens failed",
                "IMC::DeviceOpacityTable copy log_numdens failed");
  upload_vector(&d_kappa_PA,
                host.kappa_PA,
                "IMC::DeviceOpacityTable cudaMalloc kappa_PA failed",
                "IMC::DeviceOpacityTable copy kappa_PA failed");
  upload_vector(&d_kappa_PE,
                host.kappa_PE,
                "IMC::DeviceOpacityTable cudaMalloc kappa_PE failed",
                "IMC::DeviceOpacityTable copy kappa_PE failed");
  upload_vector(&d_kappa_R,
                host.kappa_R,
                "IMC::DeviceOpacityTable cudaMalloc kappa_R failed",
                "IMC::DeviceOpacityTable copy kappa_R failed");

  ntemp = host.ntemp;
  ndens = host.ndens;
  ngroups = host.ngroups;
  log_T_min = host.log_temps.front();
  log_T_max = host.log_temps.back();
  log_ni_min = host.log_numdens.front();
  log_ni_max = host.log_numdens.back();
  T_min = host.temps_eV.front();
  T_max = host.temps_eV.back();
}

void IMC::DeviceOpacityTable::release() {
  if (d_kappa_R != nullptr) {
    cuda_check(cudaFree(d_kappa_R), "IMC::DeviceOpacityTable cudaFree kappa_R failed");
    d_kappa_R = nullptr;
  }
  if (d_kappa_PE != nullptr) {
    cuda_check(cudaFree(d_kappa_PE), "IMC::DeviceOpacityTable cudaFree kappa_PE failed");
    d_kappa_PE = nullptr;
  }
  if (d_kappa_PA != nullptr) {
    cuda_check(cudaFree(d_kappa_PA), "IMC::DeviceOpacityTable cudaFree kappa_PA failed");
    d_kappa_PA = nullptr;
  }
  if (d_log_numdens != nullptr) {
    cuda_check(cudaFree(d_log_numdens),
               "IMC::DeviceOpacityTable cudaFree log_numdens failed");
    d_log_numdens = nullptr;
  }
  if (d_log_temps != nullptr) {
    cuda_check(cudaFree(d_log_temps), "IMC::DeviceOpacityTable cudaFree log_temps failed");
    d_log_temps = nullptr;
  }
  ntemp = 0;
  ndens = 0;
  ngroups = 0;
  log_T_min = 0.0;
  log_T_max = 0.0;
  log_ni_min = 0.0;
  log_ni_max = 0.0;
  T_min = 0.0;
  T_max = 0.0;
  host_generation = 0;
}

materials::IonmixOpacityDeviceView IMC::DeviceOpacityTable::view() const {
  materials::IonmixOpacityDeviceView out{};
  out.log_temps = d_log_temps;
  out.log_numdens = d_log_numdens;
  out.kappa_PA = d_kappa_PA;
  out.kappa_PE = d_kappa_PE;
  out.kappa_R = d_kappa_R;
  out.ntemp = ntemp;
  out.ndens = ndens;
  out.ngroups = ngroups;
  out.log_T_min = log_T_min;
  out.log_T_max = log_T_max;
  out.log_ni_min = log_ni_min;
  out.log_ni_max = log_ni_max;
  out.T_min = T_min;
  out.T_max = T_max;
  return out;
}

IMC::PGRWTables::PGRWTables(PGRWTables&& other) noexcept {
  *this = std::move(other);
}

IMC::PGRWTables& IMC::PGRWTables::operator=(PGRWTables&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  d_leak_inv_cdf = std::exchange(other.d_leak_inv_cdf, nullptr);
  d_leak_cdf_xi = std::exchange(other.d_leak_cdf_xi, nullptr);
  d_pos_cdf = std::exchange(other.d_pos_cdf, nullptr);
  leak_table_size = std::exchange(other.leak_table_size, 1024);
  pos_theta_bins = std::exchange(other.pos_theta_bins, 64);
  pos_rho_bins = std::exchange(other.pos_rho_bins, 128);
  theta_max = std::exchange(other.theta_max, kPgrwThetaMaxDefault);
  initialized = std::exchange(other.initialized, false);
  return *this;
}

void IMC::PGRWTables::initialize() {
  if (initialized) {
    return;
  }

  std::vector<double> host_theta(static_cast<std::size_t>(leak_table_size), 0.0);
  std::vector<double> host_xi(static_cast<std::size_t>(leak_table_size), 0.0);
  const double theta_span = theta_max - kPgrwThetaMin;
  const double theta_denom =
      static_cast<double>(std::max(leak_table_size - 1, 1));
  for (int i = 0; i < leak_table_size; ++i) {
    const double alpha = static_cast<double>(i) / theta_denom;
    const double theta = kPgrwThetaMin + theta_span * alpha;
    host_theta[static_cast<std::size_t>(i)] = theta;
    host_xi[static_cast<std::size_t>(i)] = pgrw_leak_cdf(theta);
    if (i > 0) {
      host_xi[static_cast<std::size_t>(i)] = std::max(
          host_xi[static_cast<std::size_t>(i)],
          host_xi[static_cast<std::size_t>(i - 1)]);
    }
  }

  std::vector<double> host_pos_cdf(
      static_cast<std::size_t>(pos_theta_bins) * static_cast<std::size_t>(pos_rho_bins), 0.0);
  const double theta_step = theta_max / static_cast<double>(std::max(pos_theta_bins - 1, 1));
  const double rho_step = 1.0 / static_cast<double>(std::max(pos_rho_bins - 1, 1));
  for (int it = 1; it < pos_theta_bins; ++it) {
    const double theta = theta_step * static_cast<double>(it);
    std::vector<double> row(static_cast<std::size_t>(pos_rho_bins), 0.0);
    for (int ir = 0; ir < pos_rho_bins; ++ir) {
      const double rho_frac = rho_step * static_cast<double>(ir);
      row[static_cast<std::size_t>(ir)] = pgrw_position_cdf(rho_frac, theta);
    }
    enforce_monotone_unit_cdf(row);
    const std::size_t row_offset =
        static_cast<std::size_t>(it) * static_cast<std::size_t>(pos_rho_bins);
    std::copy(row.begin(), row.end(), host_pos_cdf.begin() + row_offset);
  }

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_leak_inv_cdf),
                        sizeof(double) * host_theta.size()),
             "IMC::PGRWTables cudaMalloc leak_inv_cdf failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_leak_cdf_xi),
                        sizeof(double) * host_xi.size()),
             "IMC::PGRWTables cudaMalloc leak_cdf_xi failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_pos_cdf),
                        sizeof(double) * host_pos_cdf.size()),
             "IMC::PGRWTables cudaMalloc pos_cdf failed");
  cuda_check(cudaMemcpy(d_leak_inv_cdf,
                        host_theta.data(),
                        sizeof(double) * host_theta.size(),
                        cudaMemcpyHostToDevice),
             "IMC::PGRWTables copy leak_inv_cdf failed");
  cuda_check(cudaMemcpy(d_leak_cdf_xi,
                        host_xi.data(),
                        sizeof(double) * host_xi.size(),
                        cudaMemcpyHostToDevice),
             "IMC::PGRWTables copy leak_cdf_xi failed");
  cuda_check(cudaMemcpy(d_pos_cdf,
                        host_pos_cdf.data(),
                        sizeof(double) * host_pos_cdf.size(),
                        cudaMemcpyHostToDevice),
             "IMC::PGRWTables copy pos_cdf failed");
  initialized = true;
}

void IMC::PGRWTables::release() {
  if (d_pos_cdf != nullptr) {
    cuda_check(cudaFree(d_pos_cdf), "IMC::PGRWTables cudaFree pos_cdf failed");
    d_pos_cdf = nullptr;
  }
  if (d_leak_cdf_xi != nullptr) {
    cuda_check(cudaFree(d_leak_cdf_xi), "IMC::PGRWTables cudaFree leak_cdf_xi failed");
    d_leak_cdf_xi = nullptr;
  }
  if (d_leak_inv_cdf != nullptr) {
    cuda_check(cudaFree(d_leak_inv_cdf), "IMC::PGRWTables cudaFree leak_inv_cdf failed");
    d_leak_inv_cdf = nullptr;
  }
  initialized = false;
}

IMC::PersistentCoeffBuffers::PersistentCoeffBuffers(PersistentCoeffBuffers&& other) noexcept {
  *this = std::move(other);
}

IMC::PersistentCoeffBuffers& IMC::PersistentCoeffBuffers::operator=(
    PersistentCoeffBuffers&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  d_sigma_a = std::exchange(other.d_sigma_a, nullptr);
  d_sigma_R = std::exchange(other.d_sigma_R, nullptr);
  d_f = std::exchange(other.d_f, nullptr);
  d_sigma_a_eff = std::exchange(other.d_sigma_a_eff, nullptr);
  d_sigma_s_eff = std::exchange(other.d_sigma_s_eff, nullptr);
  d_eta = std::exchange(other.d_eta, nullptr);
  d_eta_cdf = std::exchange(other.d_eta_cdf, nullptr);
  d_emission_bias_cdf = std::exchange(other.d_emission_bias_cdf, nullptr);
  d_lambda_raw = std::exchange(other.d_lambda_raw, nullptr);
  d_g_diff_end = std::exchange(other.d_g_diff_end, nullptr);
  d_sigma_a_bar = std::exchange(other.d_sigma_a_bar, nullptr);
  d_sigma_t_bar = std::exchange(other.d_sigma_t_bar, nullptr);
  d_D_pgrw = std::exchange(other.d_D_pgrw, nullptr);
  d_gamma_pgrw = std::exchange(other.d_gamma_pgrw, nullptr);
  d_rad_E_tally = std::exchange(other.d_rad_E_tally, nullptr);
  d_E_escape = std::exchange(other.d_E_escape, nullptr);
  d_E_numerical_loss = std::exchange(other.d_E_numerical_loss, nullptr);
  d_group_bounds = std::exchange(other.d_group_bounds, nullptr);
  d_cell_dx = std::exchange(other.d_cell_dx, nullptr);
  d_error_flags = std::exchange(other.d_error_flags, nullptr);
  d_imc_absorbed = std::exchange(other.d_imc_absorbed, nullptr);
  d_imc_escaped = std::exchange(other.d_imc_escaped, nullptr);
  d_cnt_boundary = std::exchange(other.d_cnt_boundary, nullptr);
  d_cnt_scatter = std::exchange(other.d_cnt_scatter, nullptr);
  d_cnt_census = std::exchange(other.d_cnt_census, nullptr);
  d_cnt_absorb_kill = std::exchange(other.d_cnt_absorb_kill, nullptr);
  d_cnt_absorb_survive = std::exchange(other.d_cnt_absorb_survive, nullptr);
  d_cnt_roulette_kill = std::exchange(other.d_cnt_roulette_kill, nullptr);
  d_diffusion_interface_kills =
      std::exchange(other.d_diffusion_interface_kills, nullptr);
  d_interface_transitions = std::exchange(other.d_interface_transitions, nullptr);
  d_interface_reflections = std::exchange(other.d_interface_reflections, nullptr);
  d_conversion_prob_violations =
      std::exchange(other.d_conversion_prob_violations, nullptr);
  n_cells = std::exchange(other.n_cells, 0);
  n_groups = std::exchange(other.n_groups, 0);
  return *this;
}

void IMC::PersistentCoeffBuffers::ensure(const int nc, const int ng) {
  if (n_cells == nc && n_groups == ng) {
    return;
  }
  release();

  const std::size_t n_cell_groups =
      static_cast<std::size_t>(nc) * static_cast<std::size_t>(ng);
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_a), sizeof(double) * n_cell_groups),
             "IMC::PersistentCoeffBuffers cudaMalloc sigma_a failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_R), sizeof(double) * n_cell_groups),
             "IMC::PersistentCoeffBuffers cudaMalloc sigma_R failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_f), sizeof(double) * n_cell_groups),
             "IMC::PersistentCoeffBuffers cudaMalloc f failed");
  cuda_check(
      cudaMalloc(reinterpret_cast<void**>(&d_sigma_a_eff), sizeof(double) * n_cell_groups),
      "IMC::PersistentCoeffBuffers cudaMalloc sigma_a_eff failed");
  cuda_check(
      cudaMalloc(reinterpret_cast<void**>(&d_sigma_s_eff), sizeof(double) * n_cell_groups),
      "IMC::PersistentCoeffBuffers cudaMalloc sigma_s_eff failed");
  cuda_check(
      cudaMalloc(reinterpret_cast<void**>(&d_eta_cdf), sizeof(double) * n_cell_groups),
      "IMC::PersistentCoeffBuffers cudaMalloc eta_cdf failed");
  cuda_check(
      cudaMalloc(reinterpret_cast<void**>(&d_emission_bias_cdf), sizeof(double) * n_cell_groups),
      "IMC::PersistentCoeffBuffers cudaMalloc emission_bias_cdf failed");
  cuda_check(
      cudaMalloc(reinterpret_cast<void**>(&d_rad_E_tally), sizeof(double) * n_cell_groups),
      "IMC::PersistentCoeffBuffers cudaMalloc rad_E_tally failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_E_escape), sizeof(double) * static_cast<std::size_t>(ng)),
             "IMC::PersistentCoeffBuffers cudaMalloc E_escape failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_E_numerical_loss), sizeof(double)),
             "IMC::PersistentCoeffBuffers cudaMalloc E_numerical_loss failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_group_bounds),
                        sizeof(double) * static_cast<std::size_t>(ng + 1)),
             "IMC::PersistentCoeffBuffers cudaMalloc group bounds failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cell_dx), sizeof(double) * static_cast<std::size_t>(nc)),
             "IMC::PersistentCoeffBuffers cudaMalloc cell_dx failed");
  cuda_check(cudaMalloc(&d_error_flags, sizeof(core::DeviceErrorFlags)),
             "IMC::PersistentCoeffBuffers cudaMalloc error_flags failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_imc_absorbed), sizeof(unsigned long long)),
             "IMC::PersistentCoeffBuffers cudaMalloc imc_absorbed failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_imc_escaped), sizeof(unsigned long long)),
             "IMC::PersistentCoeffBuffers cudaMalloc imc_escaped failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cnt_boundary), sizeof(unsigned long long)),
             "IMC::PersistentCoeffBuffers cudaMalloc cnt_boundary failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cnt_scatter), sizeof(unsigned long long)),
             "IMC::PersistentCoeffBuffers cudaMalloc cnt_scatter failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cnt_census), sizeof(unsigned long long)),
             "IMC::PersistentCoeffBuffers cudaMalloc cnt_census failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cnt_absorb_kill), sizeof(unsigned long long)),
             "IMC::PersistentCoeffBuffers cudaMalloc cnt_absorb_kill failed");
  cuda_check(
      cudaMalloc(reinterpret_cast<void**>(&d_cnt_absorb_survive), sizeof(unsigned long long)),
      "IMC::PersistentCoeffBuffers cudaMalloc cnt_absorb_survive failed");
  cuda_check(
      cudaMalloc(reinterpret_cast<void**>(&d_cnt_roulette_kill), sizeof(unsigned long long)),
      "IMC::PersistentCoeffBuffers cudaMalloc cnt_roulette_kill failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_diffusion_interface_kills),
                        sizeof(unsigned long long)),
             "IMC::PersistentCoeffBuffers cudaMalloc diffusion_interface_kills failed");
  cuda_check(
      cudaMalloc(reinterpret_cast<void**>(&d_interface_transitions), sizeof(unsigned long long)),
      "IMC::PersistentCoeffBuffers cudaMalloc interface_transitions failed");
  cuda_check(
      cudaMalloc(reinterpret_cast<void**>(&d_interface_reflections), sizeof(unsigned long long)),
      "IMC::PersistentCoeffBuffers cudaMalloc interface_reflections failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_conversion_prob_violations),
                        sizeof(unsigned long long)),
             "IMC::PersistentCoeffBuffers cudaMalloc conversion_prob_violations failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_lambda_raw), sizeof(double) * static_cast<std::size_t>(nc)),
             "IMC::PersistentCoeffBuffers cudaMalloc lambda_raw failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_eta),
                        sizeof(double) * n_cell_groups),
             "IMC::PersistentCoeffBuffers cudaMalloc eta failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_g_diff_end),
                        sizeof(int) * static_cast<std::size_t>(nc)),
             "IMC::PersistentCoeffBuffers cudaMalloc g_diff_end failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_a_bar),
                        sizeof(double) * static_cast<std::size_t>(nc)),
             "IMC::PersistentCoeffBuffers cudaMalloc sigma_a_bar failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_t_bar),
                        sizeof(double) * static_cast<std::size_t>(nc)),
             "IMC::PersistentCoeffBuffers cudaMalloc sigma_t_bar failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_D_pgrw),
                        sizeof(double) * static_cast<std::size_t>(nc)),
             "IMC::PersistentCoeffBuffers cudaMalloc D_pgrw failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_gamma_pgrw),
                        sizeof(double) * static_cast<std::size_t>(nc)),
             "IMC::PersistentCoeffBuffers cudaMalloc gamma_pgrw failed");
  n_cells = nc;
  n_groups = ng;
}

void IMC::PersistentCoeffBuffers::release() {
  if (d_gamma_pgrw != nullptr) {
    cuda_check(cudaFree(d_gamma_pgrw),
               "IMC::PersistentCoeffBuffers cudaFree gamma_pgrw failed");
    d_gamma_pgrw = nullptr;
  }
  if (d_D_pgrw != nullptr) {
    cuda_check(cudaFree(d_D_pgrw), "IMC::PersistentCoeffBuffers cudaFree D_pgrw failed");
    d_D_pgrw = nullptr;
  }
  if (d_sigma_t_bar != nullptr) {
    cuda_check(cudaFree(d_sigma_t_bar),
               "IMC::PersistentCoeffBuffers cudaFree sigma_t_bar failed");
    d_sigma_t_bar = nullptr;
  }
  if (d_sigma_a_bar != nullptr) {
    cuda_check(cudaFree(d_sigma_a_bar),
               "IMC::PersistentCoeffBuffers cudaFree sigma_a_bar failed");
    d_sigma_a_bar = nullptr;
  }
  if (d_g_diff_end != nullptr) {
    cuda_check(cudaFree(d_g_diff_end),
               "IMC::PersistentCoeffBuffers cudaFree g_diff_end failed");
    d_g_diff_end = nullptr;
  }
  if (d_conversion_prob_violations != nullptr) {
    cuda_check(cudaFree(d_conversion_prob_violations),
               "IMC::PersistentCoeffBuffers cudaFree conversion_prob_violations failed");
    d_conversion_prob_violations = nullptr;
  }
  if (d_lambda_raw != nullptr) {
    cuda_check(cudaFree(d_lambda_raw),
               "IMC::PersistentCoeffBuffers cudaFree lambda_raw failed");
    d_lambda_raw = nullptr;
  }
  if (d_eta != nullptr) {
    cuda_check(cudaFree(d_eta),
               "IMC::PersistentCoeffBuffers cudaFree eta failed");
    d_eta = nullptr;
  }
  if (d_interface_reflections != nullptr) {
    cuda_check(cudaFree(d_interface_reflections),
               "IMC::PersistentCoeffBuffers cudaFree interface_reflections failed");
    d_interface_reflections = nullptr;
  }
  if (d_interface_transitions != nullptr) {
    cuda_check(cudaFree(d_interface_transitions),
               "IMC::PersistentCoeffBuffers cudaFree interface_transitions failed");
    d_interface_transitions = nullptr;
  }
  if (d_diffusion_interface_kills != nullptr) {
    cuda_check(cudaFree(d_diffusion_interface_kills),
               "IMC::PersistentCoeffBuffers cudaFree diffusion_interface_kills failed");
    d_diffusion_interface_kills = nullptr;
  }
  if (d_cnt_roulette_kill != nullptr) {
    cuda_check(cudaFree(d_cnt_roulette_kill),
               "IMC::PersistentCoeffBuffers cudaFree cnt_roulette_kill failed");
    d_cnt_roulette_kill = nullptr;
  }
  if (d_cnt_absorb_survive != nullptr) {
    cuda_check(cudaFree(d_cnt_absorb_survive),
               "IMC::PersistentCoeffBuffers cudaFree cnt_absorb_survive failed");
    d_cnt_absorb_survive = nullptr;
  }
  if (d_cnt_absorb_kill != nullptr) {
    cuda_check(cudaFree(d_cnt_absorb_kill),
               "IMC::PersistentCoeffBuffers cudaFree cnt_absorb_kill failed");
    d_cnt_absorb_kill = nullptr;
  }
  if (d_cnt_census != nullptr) {
    cuda_check(cudaFree(d_cnt_census),
               "IMC::PersistentCoeffBuffers cudaFree cnt_census failed");
    d_cnt_census = nullptr;
  }
  if (d_cnt_scatter != nullptr) {
    cuda_check(cudaFree(d_cnt_scatter),
               "IMC::PersistentCoeffBuffers cudaFree cnt_scatter failed");
    d_cnt_scatter = nullptr;
  }
  if (d_cnt_boundary != nullptr) {
    cuda_check(cudaFree(d_cnt_boundary),
               "IMC::PersistentCoeffBuffers cudaFree cnt_boundary failed");
    d_cnt_boundary = nullptr;
  }
  if (d_imc_escaped != nullptr) {
    cuda_check(cudaFree(d_imc_escaped),
               "IMC::PersistentCoeffBuffers cudaFree imc_escaped failed");
    d_imc_escaped = nullptr;
  }
  if (d_imc_absorbed != nullptr) {
    cuda_check(cudaFree(d_imc_absorbed),
               "IMC::PersistentCoeffBuffers cudaFree imc_absorbed failed");
    d_imc_absorbed = nullptr;
  }
  if (d_error_flags != nullptr) {
    cuda_check(cudaFree(d_error_flags),
               "IMC::PersistentCoeffBuffers cudaFree error_flags failed");
    d_error_flags = nullptr;
  }
  if (d_cell_dx != nullptr) {
    cuda_check(cudaFree(d_cell_dx), "IMC::PersistentCoeffBuffers cudaFree cell_dx failed");
    d_cell_dx = nullptr;
  }
  if (d_group_bounds != nullptr) {
    cuda_check(cudaFree(d_group_bounds),
               "IMC::PersistentCoeffBuffers cudaFree group bounds failed");
    d_group_bounds = nullptr;
  }
  if (d_E_numerical_loss != nullptr) {
    cuda_check(cudaFree(d_E_numerical_loss),
               "IMC::PersistentCoeffBuffers cudaFree E_numerical_loss failed");
    d_E_numerical_loss = nullptr;
  }
  if (d_E_escape != nullptr) {
    cuda_check(cudaFree(d_E_escape), "IMC::PersistentCoeffBuffers cudaFree E_escape failed");
    d_E_escape = nullptr;
  }
  if (d_rad_E_tally != nullptr) {
    cuda_check(cudaFree(d_rad_E_tally),
               "IMC::PersistentCoeffBuffers cudaFree rad_E_tally failed");
    d_rad_E_tally = nullptr;
  }
  if (d_eta_cdf != nullptr) {
    cuda_check(cudaFree(d_eta_cdf), "IMC::PersistentCoeffBuffers cudaFree eta_cdf failed");
    d_eta_cdf = nullptr;
  }
  if (d_emission_bias_cdf != nullptr) {
    cuda_check(cudaFree(d_emission_bias_cdf),
               "IMC::PersistentCoeffBuffers cudaFree emission_bias_cdf failed");
    d_emission_bias_cdf = nullptr;
  }
  if (d_sigma_s_eff != nullptr) {
    cuda_check(cudaFree(d_sigma_s_eff),
               "IMC::PersistentCoeffBuffers cudaFree sigma_s_eff failed");
    d_sigma_s_eff = nullptr;
  }
  if (d_sigma_a_eff != nullptr) {
    cuda_check(cudaFree(d_sigma_a_eff),
               "IMC::PersistentCoeffBuffers cudaFree sigma_a_eff failed");
    d_sigma_a_eff = nullptr;
  }
  if (d_f != nullptr) {
    cuda_check(cudaFree(d_f), "IMC::PersistentCoeffBuffers cudaFree f failed");
    d_f = nullptr;
  }
  if (d_sigma_R != nullptr) {
    cuda_check(cudaFree(d_sigma_R), "IMC::PersistentCoeffBuffers cudaFree sigma_R failed");
    d_sigma_R = nullptr;
  }
  if (d_sigma_a != nullptr) {
    cuda_check(cudaFree(d_sigma_a), "IMC::PersistentCoeffBuffers cudaFree sigma_a failed");
    d_sigma_a = nullptr;
  }
  n_cells = 0;
  n_groups = 0;
}

IMC::~IMC() {
  release_compact_alive_scratch();
  release_pool_stats_scratch();
  coeff_buf_.release();
  pgrw_tables_.release();
  device_opacity_table_.release();
}

double IMC::compute_dt_rad(const core::State& state, const core::Config& cfg) {
  return compute_dt_rad(state, cfg, nullptr);
}

double IMC::compute_dt_rad(const core::State& state,
                           const core::Config& cfg,
                           DtRadDiagnostics* diag) {
  return compute_dt_rad_limit(state, cfg, diag);
}

void IMC::transport_step(core::State& state,
                         const core::Config& cfg,
                         const double dt,
                         const parallel::PartitionInfo& part,
                         parallel::CommBuffers* bufs) {
  using Clock = std::chrono::steady_clock;
  const bool verbose_imc_timing = cfg.main.verbosity == "verbose";
  const auto elapsed_ms = [](const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  const auto t_transport_start =
      verbose_imc_timing ? Clock::now() : Clock::time_point{};
  auto t_pre_subphase = t_transport_start;
  double pre_setup_ms = 0.0;
  double pre_te_pred_ms = 0.0;
  double pre_nlte_coeffs_ms = 0.0;
  double pre_nlte_h2d_ms = 0.0;
  double pre_difference_setup_ms = 0.0;
  double pre_fleck_build_ms = 0.0;
  double pre_other_A_ms = 0.0;
  double pre_other_B_ms = 0.0;
  double pre_other_C_ms = 0.0;
  double pre_other_D_ms = 0.0;
  const auto mark_pre_subphase = [&](double& value) {
    if (verbose_imc_timing) {
      const auto t_now = Clock::now();
      value += elapsed_ms(t_pre_subphase, t_now);
      t_pre_subphase = t_now;
    }
  };

  TENRYU_ASSERT(bufs != nullptr || part.n_ranks <= 1,
                "IMC::transport_step requires CommBuffers when n_ranks > 1");
  last_interface_transitions_ = 0;
  last_interface_reflections_ = 0;
  last_conversion_prob_violations_ = 0;
  last_cnt_boundary_ = 0;
  last_cnt_scatter_ = 0;
  last_cnt_census_ = 0;
  last_cnt_absorb_kill_ = 0;
  last_cnt_absorb_survive_ = 0;
  last_cnt_roulette_kill_ = 0;
  last_f_min_ = 1.0;
  last_f_mean_ = 1.0;
  last_f_p95_ = 1.0;
  last_tau_gt1_ = 0;
  last_tau_gt2_ = 0;
  last_tau_gt3_ = 0;
  last_tau_gt4_ = 0;
  last_numerical_loss_step_ = 0.0;
  last_thermal_lost_step_ = 0.0;
  last_marshak_in_step_ = 0.0;
  last_volume_source_step_ = 0.0;
  last_ddmc_mode_count_ = 0;
  last_imc_mode_count_ = 0;
  last_n_total_ = 0;
  last_n_imc_particles_ = 0;
  last_n_ddmc_particles_ = 0;
  last_n_census_ = 0;
  last_n_absorbed_ = 0;
  last_n_escaped_ = 0;
  last_ddmc_fraction_ = 0.0;
  last_weight_min_ = 0.0;
  last_weight_mean_ = 0.0;
  last_weight_max_ = 0.0;
  last_overshoot_count_ = 0;
  last_overshoot_max_ = 0.0;
  last_mmatrix_violations_ = 0;
  last_mmatrix_fallback_count_ = 0;
  last_omega_below_threshold_ = 0;
  last_ddmc_to_imc_conversions_ = 0;
  last_switches_imc_to_ddmc_ = 0;
  last_switches_ddmc_to_imc_ = 0;
  last_rad_momentum_deposition_ = 0.0;
  last_device_error_flags_ = core::DeviceErrorFlags{};
  last_cell_radiation_coeffs_valid_ = false;
  last_reference_field_diagnostics_ = ReferenceFieldDiagnostics{};
  last_sigma_R_max_.clear();
  if (!cfg.radiation.enabled || !cfg.radiation.imc.difference.enabled) {
    previous_reference_U_.clear();
    previous_reference_U_device_valid_ = false;
    state.difference_W.reset(0);
    state.difference_E_ref.reset(0);
    state.difference_residual_E.reset(0);
  }
  if (!cfg.radiation.enabled || dt <= 0.0 || state.rho.empty()) {
    return;
  }
  if (cfg.materials.materials.empty()) {
    return;
  }
  if (state.mesh.dim == 2) {
    TENRYU_ASSERT(cfg.radiation.boundary.inner_r == "reflect",
                  "2D_RZ radiation boundary inner_r must be \"reflect\"");
    TENRYU_ASSERT(is_supported_2d_bc_outer(cfg.radiation.boundary.outer_r),
                  "2D_RZ radiation boundary outer_r must be \"vacuum\" or \"reflect\"");
    TENRYU_ASSERT(is_supported_2d_bc_z(cfg.radiation.boundary.bottom_z),
                  "2D_RZ radiation boundary bottom_z must be \"vacuum\", \"reflect\", or \"marshak\"");
    TENRYU_ASSERT(is_supported_2d_bc_z(cfg.radiation.boundary.top_z),
                  "2D_RZ radiation boundary top_z must be \"vacuum\", \"reflect\", or \"marshak\"");
  }

  TENRYU_ASSERT(state.rho.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "IMC::transport_step n_cells exceeds int range");
  const int n_cells = static_cast<int>(state.rho.size());
  TENRYU_ASSERT(static_cast<int>(state.cell_is_void.size()) == n_cells,
                "IMC::transport_step requires cell_is_void size to match n_cells");
  const int n_groups = std::max(cfg.radiation.groups, 1);
  TENRYU_ASSERT(
      n_groups <= static_cast<int>(std::numeric_limits<std::uint16_t>::max()),
      "IMC::transport_step supports at most 65535 groups (PhotonPool::group_id is uint16_t)");
  const std::size_t n_cells_us = static_cast<std::size_t>(n_cells);
  const std::size_t n_groups_us = static_cast<std::size_t>(n_groups);
  TENRYU_ASSERT(n_cells == 0 || n_groups_us <=
                                  (std::numeric_limits<std::size_t>::max() / n_cells_us),
                "IMC::transport_step n_cells*n_groups overflow");
  const std::size_t n_cell_groups_us = n_cells_us * n_groups_us;
  TENRYU_ASSERT(
      n_cell_groups_us <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
      "IMC::transport_step n_cells*n_groups exceeds int range");
  const int n_cell_groups = static_cast<int>(n_cell_groups_us);
  const bool diffusion_enabled_1d =
      cfg.radiation.diffusion.enabled && state.mesh.dim == 1;
  const bool sn_material_coupling_active =
      cfg.radiation.holo.enabled &&
      cfg.radiation.holo.sn_closure &&
      cfg.radiation.holo.sn_material_coupling &&
      (state.mesh.dim == 1 || state.mesh.dim == 2);
  const bool holo_enabled_1d =
      cfg.radiation.holo.enabled && state.mesh.dim == 1 &&
      !sn_material_coupling_active;
  const bool holo_qd_solver =
      holo_enabled_1d && cfg.radiation.holo.solver == "quasidiffusion_1d";
  const bool holo_prr_enabled =
      holo_enabled_1d && cfg.radiation.holo.p_rr_tally;
  const bool face_current_tracking_enabled_1d =
      diffusion_enabled_1d || holo_enabled_1d;
  const bool need_source_smoothing_sigma_R =
      state.mesh.dim == 1 &&
      (cfg.radiation.imc.net_e_source_smoothing.enabled ||
       cfg.numerics.hydro.hk_velocity_damper_C > 0.0);
  last_holo_selector_diagnostics_ = HoloSelectorDiagnostics{};
  last_holo_lo_result_ = HoloLOResult{};

  const int mat_idx = cfg.materials.first_nonvoid_material_index();
  TENRYU_ASSERT(mat_idx >= 0, "IMC::transport_step requires at least one non-void material");
  const auto& mat = cfg.materials.materials[static_cast<std::size_t>(mat_idx)];
  const bool use_freq_dep_marshak = (mat.opacity_model == "freq_dep_marshak");
  const bool use_power_law = (mat.opacity_model == "power_law");
  const auto nonvoid_count = std::count_if(
      cfg.materials.materials.begin(), cfg.materials.materials.end(),
      [](const auto& m) { return !m.is_void; });
  // P2a (2026-08-30): multi-material multigroup-diffusion decks reach here only
  // with constant/none/LTE-tmat materials (namelist guard); their per-material
  // tables are evaluated inside evaluate_fld_opacity_and_emission, so the
  // single-table NLTE routing and its Phase-1 assert do not apply.
  const bool multimat_fld_tables =
      cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion &&
      nonvoid_count > 1;
  const bool use_nlte_table =
      !multimat_fld_tables &&
      (mat.opacity_model == "table_nlte" || mat.opacity_model == "tmat");
  if (use_nlte_table) {
    TENRYU_ASSERT(nonvoid_count == 1,
                  "Phase-1 NLTE: single non-void material contract violated");
  }
  const int runtime_opacity_model =
      use_nlte_table
          ? materials::kOpacityModelTableNLTE
          : (use_freq_dep_marshak ? materials::kOpacityModelFreqDepMarshak
             : (use_power_law ? materials::kOpacityModelPowerLaw
                              : materials::kOpacityModelConstant));
  if (cfg.radiation.mode != core::RadiationMode::MultigroupDiffusion &&
      cfg.radiation.mode != core::RadiationMode::SnTransport &&
      !use_freq_dep_marshak && !use_power_law && !use_nlte_table &&
      mat.kappa_a_constant <= 0.0) {
    // No absorbing opacity: radiation operator is inert in current milestone implementation.
    previous_reference_U_.clear();
    previous_reference_U_device_valid_ = false;
    state.rad_dep.fill(0.0);
    state.rad_E.fill(0.0);
    state.rad_emit.fill(0.0);
    state.holo_Prr.fill(0.0);
    state.holo_chi.fill(0.0);
    state.holo_chi_filtered.fill(0.0);
    state.holo_Prr_coverage.fill(0.0);
    state.difference_W.reset(0);
    state.difference_E_ref.reset(0);
    state.difference_residual_E.reset(0);
    if (state.holo_core_mask_valid) {
      std::fill(state.holo_core_mask.begin(), state.holo_core_mask.end(), 0U);
      std::fill(state.holo_patch_mask.begin(), state.holo_patch_mask.end(), 0U);
      std::fill(state.holo_core_prev_mask.begin(), state.holo_core_prev_mask.end(), 0U);
      std::fill(state.holo_hold_count.begin(), state.holo_hold_count.end(), 0);
      std::fill(state.holo_dwell_count.begin(), state.holo_dwell_count.end(), 0);
      std::fill(state.holo_lo_weight.begin(), state.holo_lo_weight.end(), 0.0);
      state.holo_core_mask_valid = false;
    }
    state.holo_lo_source_valid = false;
    state.holo_ale_invalidated = false;
    if (need_source_smoothing_sigma_R) {
      last_sigma_R_max_.assign(n_cells_us, 0.0);
    }
    return;
  }

  coeff_buf_.ensure(n_cells, n_groups);

  std::vector<double> bounds = cfg.radiation.group_bounds_eV;
  if (bounds.empty() || static_cast<int>(bounds.size()) != n_groups + 1) {
    const std::vector<double> range = resolve_compute_T_range_eV(cfg, false);
    const double Tmin = std::max(range[0], 1.0e-3);
    const double Tmax = std::max(range[1], Tmin * 1.001);
    bounds = Groups::make_log_uniform_bounds(n_groups, Tmin, Tmax);
  }
  Groups groups(bounds);
  const auto& planck_cfg = cfg.radiation.planck_fraction;
  if (planck_cfg.method == "tabulate") {
    core::log_warning("Radiation.groups.planck_fraction.method=\"tabulate\" is stored but not "
                      "implemented yet; falling back to computed Planck fractions");
  }
  const int planck_n_T = std::max(planck_cfg.compute_N_T, 2);
  const std::vector<double> planck_range =
      resolve_compute_T_range_eV(cfg, false);
  const double planck_T_min = planck_range[0];
  const double planck_T_max = planck_range[1];
  const int planck_n_groups = groups.num_groups();
  if (!planck_cache_valid_ ||
      planck_cache_n_groups_ != planck_n_groups ||
      planck_cache_n_T_ != planck_n_T ||
      planck_cache_T_min_eV_ != planck_T_min ||
      planck_cache_T_max_eV_ != planck_T_max ||
      !equal_double_bits(planck_cache_bounds_eV_, bounds)) {
    planck_cache_.build(groups, planck_n_T, planck_T_min, planck_T_max);
    planck_cache_bounds_eV_ = bounds;
    planck_cache_n_groups_ = planck_n_groups;
    planck_cache_n_T_ = planck_n_T;
    planck_cache_T_min_eV_ = planck_T_min;
    planck_cache_T_max_eV_ = planck_T_max;
    planck_cache_valid_ = true;
  }
  const PlanckTable& planck = planck_cache_;

  if (cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion) {
    if (state.mesh.dim == 1) {
      advance_radiation_step_fld_1d(state, cfg, planck, mat, dt);
      return;
    }
    advance_radiation_step_fld_2d_rz(state, cfg, planck, mat, dt, part, bufs);
    return;
  }
  if (cfg.radiation.mode == core::RadiationMode::SnTransport) {
    if (state.mesh.dim == 1) {
      advance_radiation_step_sn_1d(state, cfg, planck, mat, dt);
      return;
    }
    advance_radiation_step_sn_2d_rz(state, cfg, planck, mat, dt, part,
                                    bufs);
    return;
  }

  const std::int64_t particles_per_cell_group =
      std::max<std::int64_t>(cfg.radiation.imc.particles_per_cell_group, 0);
  std::int64_t initial_requested = 0;
  if (particles_per_cell_group > 0 && n_cells > 0) {
    TENRYU_ASSERT(
        particles_per_cell_group <=
            (std::numeric_limits<std::int64_t>::max() /
             static_cast<std::int64_t>(n_cells)),
        "IMC::transport_step initial particle capacity overflow (ppcg*n_cells)");
    const std::int64_t per_group =
        particles_per_cell_group * static_cast<std::int64_t>(n_cells);
    TENRYU_ASSERT(
        per_group <=
            (std::numeric_limits<std::int64_t>::max() /
             static_cast<std::int64_t>(n_groups)),
        "IMC::transport_step initial particle capacity overflow (ppcg*n_cells*n_groups)");
    initial_requested = per_group * static_cast<std::int64_t>(n_groups);
  }
  const std::int64_t initial_capacity_ll = std::max<std::int64_t>(1024LL, initial_requested);
  TENRYU_ASSERT(
      initial_capacity_ll <= static_cast<std::int64_t>(std::numeric_limits<int>::max()),
      "IMC::transport_step initial particle capacity exceeds int range");
  const int initial_capacity = static_cast<int>(initial_capacity_ll);
  if (!pool_initialized_) {
    pool_.allocate(initial_capacity);
    pool_initialized_ = true;
  }
  reserve_compact_alive_scratch(pool_.capacity);
  reserve_pool_stats_scratch(pool_.capacity);
  pool_.n_census = std::max(pool_.n_alive, 0);

  TENRYU_ASSERT(
      initial_capacity_ll <= (std::numeric_limits<std::int64_t>::max() / 128LL),
      "IMC::transport_step max_pool_size overflow");
  const std::int64_t max_pool_size_ll =
      std::max<std::int64_t>(initial_capacity_ll * 128LL, 20000000LL);
  TENRYU_ASSERT(
      max_pool_size_ll <= static_cast<std::int64_t>(std::numeric_limits<int>::max()),
      "IMC::transport_step max_pool_size exceeds int range");
  const int max_pool_size = static_cast<int>(max_pool_size_ll);

  CellRadiationCoeffs nlte_coeffs;
  const auto& fleck_diag_cfg = cfg.diagnostics.fleck_diag;
  const bool fleck_diag_has_radius =
      fleck_diag_cfg.r_min_cm >= 0.0 && fleck_diag_cfg.r_max_cm >= 0.0;
  const int fleck_diag_step = state.step + 1;
  const bool fleck_diag_due =
      fleck_diag_step > 0 && (fleck_diag_step % std::max(fleck_diag_cfg.every, 1)) == 0;
  const bool capture_nlte_host_coeffs =
      state.mesh.dim == 1 &&
      fleck_diag_due &&
      (cfg.main.verbosity == "verbose" ||
       (cfg.diagnostics.enabled && fleck_diag_cfg.enabled)) &&
      (!fleck_diag_cfg.cells.empty() || fleck_diag_has_radius);
  if (runtime_opacity_model == materials::kOpacityModelTableNLTE) {
    if (!nlte_table_loaded_ || nlte_table_ == nullptr ||
        nlte_table_file_ != mat.opacity_file) {
      if (nlte_table_loaded_ && nlte_table_file_ != mat.opacity_file) {
        core::log_warning("IMC::transport_step reloading NLTE opacity table due to "
                          "opacity.file change from \"" +
                          nlte_table_file_ + "\" to \"" + mat.opacity_file + "\"");
      }
      if (mat.opacity_model == "tmat") {
        const materials::TmatFile tmat = materials::load_tmat(mat.opacity_file);
        TENRYU_ASSERT(tmat.opacity.has_value(),
                      "IMC::transport_step tmat opacity.model requires /opacity payload");
        nlte_table_ = std::make_unique<materials::IonmixOpacityData>(
            materials::tmat_to_ionmix_opacity(*tmat.opacity));
      } else {
        nlte_table_ = std::make_unique<materials::IonmixOpacityData>(
            materials::load_ionmix_opacity(mat.opacity_file));
      }
      if (cfg.radiation.group_repack_hard_xray) {
        std::vector<double> target_bounds = cfg.radiation.group_bounds_eV;
        if (target_bounds.empty()) {
          const std::vector<double> range = resolve_compute_T_range_eV(cfg, false);
          target_bounds = repack_group_bounds_for_hard_xray(
              nlte_table_->ngroups, range);
        }
        *nlte_table_ = resample_opacity_groups_to_bounds(*nlte_table_, target_bounds);
        core::log_info("NLTE opacity table resampled onto hard-X-ray group bounds: " +
                       std::to_string(count_groups_inside_energy_band(
                           nlte_table_->bounds_eV, 2000.0, 5000.0)) +
                       " groups in 2-5 keV");
      }
      nlte_table_loaded_ = true;
      nlte_table_file_ = mat.opacity_file;
      ++nlte_table_generation_;
    }
    TENRYU_ASSERT(nlte_table_ != nullptr,
                  "IMC::transport_step NLTE table cache must be initialized");
    if (device_opacity_table_.needs_upload(nlte_table_generation_)) {
      device_opacity_table_.upload(*nlte_table_);
      device_opacity_table_.host_generation = nlte_table_generation_;
    }
  }

  double* d_sigma_a = coeff_buf_.d_sigma_a;
  double* d_sigma_R = coeff_buf_.d_sigma_R;
  double* d_f = coeff_buf_.d_f;
  double* d_sigma_a_eff = coeff_buf_.d_sigma_a_eff;
  double* d_sigma_s_eff = coeff_buf_.d_sigma_s_eff;
  double* d_rad_E_tally = coeff_buf_.d_rad_E_tally;
  if (holo_enabled_1d) {
    if (state.holo_Prr.size() != n_cell_groups_us) {
      state.holo_Prr.reset(n_cell_groups_us);
    }
    if (state.holo_chi.size() != n_cell_groups_us) {
      state.holo_chi.reset(n_cell_groups_us);
    }
    if (holo_qd_solver && state.holo_chi_filtered.size() != n_cell_groups_us) {
      state.holo_chi_filtered.reset(n_cell_groups_us);
    }
    if (state.holo_Prr_coverage.size() != n_cell_groups_us) {
      state.holo_Prr_coverage.reset(n_cell_groups_us);
    }
    if (!holo_prr_enabled) {
      state.holo_Prr.fill(0.0);
      state.holo_chi.fill(0.0);
      state.holo_chi_filtered.fill(0.0);
      state.holo_Prr_coverage.fill(0.0);
    }
  }
  double* d_holo_Prr_tally = holo_prr_enabled ? state.holo_Prr.data() : nullptr;
  double* d_holo_Prr_coverage_tally =
      holo_prr_enabled ? state.holo_Prr_coverage.data() : nullptr;
  double* d_E_escape = coeff_buf_.d_E_escape;
  double* d_E_numerical_loss = coeff_buf_.d_E_numerical_loss;
  double* d_group_bounds = coeff_buf_.d_group_bounds;
  double* d_cell_dx = nullptr;
  double* d_eta_cdf = nullptr;
  int* d_g_diff_end = coeff_buf_.d_g_diff_end;
  double* d_sigma_a_bar = coeff_buf_.d_sigma_a_bar;
  double* d_sigma_t_bar = coeff_buf_.d_sigma_t_bar;
  double* d_D_pgrw = coeff_buf_.d_D_pgrw;
  double* d_gamma_pgrw = coeff_buf_.d_gamma_pgrw;
  TransportMode* d_ddmc_mode = nullptr;
  unsigned long long* d_imc_absorbed = coeff_buf_.d_imc_absorbed;
  unsigned long long* d_imc_escaped = coeff_buf_.d_imc_escaped;
  unsigned long long* d_cnt_boundary = coeff_buf_.d_cnt_boundary;
  unsigned long long* d_cnt_scatter = coeff_buf_.d_cnt_scatter;
  unsigned long long* d_cnt_census = coeff_buf_.d_cnt_census;
  unsigned long long* d_cnt_absorb_kill = coeff_buf_.d_cnt_absorb_kill;
  unsigned long long* d_cnt_absorb_survive = coeff_buf_.d_cnt_absorb_survive;
  unsigned long long* d_cnt_roulette_kill = coeff_buf_.d_cnt_roulette_kill;
  unsigned long long* d_diffusion_interface_kills =
      coeff_buf_.d_diffusion_interface_kills;
  unsigned long long* d_interface_transitions = nullptr;
  unsigned long long* d_interface_reflections = nullptr;
  unsigned long long* d_conversion_prob_violations = nullptr;
  auto* d_error_flags = static_cast<core::DeviceErrorFlags*>(coeff_buf_.d_error_flags);

  cuda_check(cudaMemset(d_E_numerical_loss, 0, sizeof(double)),
             "IMC::transport_step cudaMemset E_numerical_loss failed");
  cuda_check(cudaMemset(d_imc_absorbed, 0, sizeof(unsigned long long)),
             "IMC::transport_step cudaMemset imc_absorbed failed");
  cuda_check(cudaMemset(d_imc_escaped, 0, sizeof(unsigned long long)),
             "IMC::transport_step cudaMemset imc_escaped failed");
  cuda_check(cudaMemset(d_cnt_boundary, 0, sizeof(unsigned long long)),
             "IMC::transport_step cudaMemset cnt_boundary failed");
  cuda_check(cudaMemset(d_cnt_scatter, 0, sizeof(unsigned long long)),
             "IMC::transport_step cudaMemset cnt_scatter failed");
  cuda_check(cudaMemset(d_cnt_census, 0, sizeof(unsigned long long)),
             "IMC::transport_step cudaMemset cnt_census failed");
  cuda_check(cudaMemset(d_cnt_absorb_kill, 0, sizeof(unsigned long long)),
             "IMC::transport_step cudaMemset cnt_absorb_kill failed");
  cuda_check(cudaMemset(d_cnt_absorb_survive, 0, sizeof(unsigned long long)),
             "IMC::transport_step cudaMemset cnt_absorb_survive failed");
  cuda_check(cudaMemset(d_cnt_roulette_kill, 0, sizeof(unsigned long long)),
             "IMC::transport_step cudaMemset cnt_roulette_kill failed");
  cuda_check(cudaMemset(d_diffusion_interface_kills, 0, sizeof(unsigned long long)),
             "IMC::transport_step cudaMemset diffusion_interface_kills failed");
  cuda_check(cudaMemcpy(d_group_bounds,
                        bounds.data(),
                        sizeof(double) * static_cast<std::size_t>(n_groups + 1),
                        cudaMemcpyHostToDevice),
             "IMC::transport_step copy group bounds failed");
  cuda_check(cudaMemset(d_error_flags, 0, sizeof(*d_error_flags)),
             "IMC::transport_step cudaMemset error_flags failed");

  mark_pre_subphase(pre_setup_ms);

  core::DeviceErrorFlags opacity_flags{};
  bool is_nlte_mode = false;
  if (runtime_opacity_model == materials::kOpacityModelTableNLTE) {
    is_nlte_mode = true;
    TENRYU_ASSERT(static_cast<int>(nlte_table_->bounds_eV.size()) == n_groups + 1,
                  "IMC::transport_step NLTE table group boundary count mismatch");
    std::vector<double> host_Te_restore;
    bool te_overridden = false;
    if (cfg.radiation.imc.opacity_predictor && !state.delta_E_rad_prev.empty()) {
      TENRYU_ASSERT(state.delta_E_rad_prev.size() == n_cells_us,
                    "IMC::transport_step delta_E_rad_prev size mismatch");
      std::vector<double> host_Te(n_cells_us, 0.0);
      std::vector<double> host_Te_pred(n_cells_us, 0.0);
      std::vector<double> host_delta_E_prev(n_cells_us, 0.0);
      std::vector<double> host_rho(n_cells_us, 0.0);
      std::vector<double> host_vol(n_cells_us, 0.0);
      std::vector<double> host_zbar(n_cells_us, 0.0);
      std::vector<double> host_cv_e;
      state.Te.copy_to_host(host_Te.data());
      state.delta_E_rad_prev.copy_to_host(host_delta_E_prev.data());
      state.rho.copy_to_host(host_rho.data());
      state.vol.copy_to_host(host_vol.data());
      state.zbar.copy_to_host(host_zbar.data());
      const bool has_state_cv_e = !state.cv_e.empty();
      if (has_state_cv_e) {
        TENRYU_ASSERT(state.cv_e.size() == n_cells_us,
                      "IMC::transport_step state cv_e size mismatch");
        host_cv_e.resize(n_cells_us, 0.0);
        state.cv_e.copy_to_host(host_cv_e.data());
      }

      constexpr double kOpacityPredictorTheta = 0.5;
      const double te_floor = cfg.numerics.floors.Te;
      const double A = std::max(mat.A, 1.0e-12);
      const double gm1 = std::max(mat.ideal_gas_gamma - 1.0, 1.0e-12);
      for (int c = 0; c < n_cells; ++c) {
        const std::size_t c_us = static_cast<std::size_t>(c);
        if (state.cell_is_void[c_us] != 0U) {
          host_Te_pred[c_us] = host_Te[c_us];
          continue;
        }

        const double Te_curr =
            (std::isfinite(host_Te[c_us]) && host_Te[c_us] > te_floor) ? host_Te[c_us] : te_floor;
        const double rho_c =
            (std::isfinite(host_rho[c_us]) && host_rho[c_us] > 0.0) ? host_rho[c_us] : 1.0e-30;
        const double vol_c =
            (std::isfinite(host_vol[c_us]) && host_vol[c_us] > 0.0) ? host_vol[c_us] : 0.0;
        double cv_mass_e = 0.0;
        if (mat.cv_e_override > 0.0) {
          cv_mass_e = mat.cv_e_override / rho_c;
        } else if (has_state_cv_e && host_cv_e[c_us] > 0.0) {
          cv_mass_e = host_cv_e[c_us];
        } else if (mat.eos_tables != nullptr) {
          cv_mass_e = std::max(mat.eos_tables->electron.cv(rho_c, Te_curr), 0.0);
        } else {
          const double zbar_c =
              (std::isfinite(host_zbar[c_us]) && host_zbar[c_us] > 0.0) ? host_zbar[c_us] : 0.0;
          cv_mass_e = zbar_c * core::constants::eV_to_erg /
                      (A * core::constants::proton_mass * gm1);
        }

        const double heat_capacity =
            rho_c * (std::isfinite(cv_mass_e) ? std::max(cv_mass_e, 0.0) : 0.0) * vol_c;
        double Te_pred = Te_curr;
        if (heat_capacity > 1.0e-30 && std::isfinite(host_delta_E_prev[c_us])) {
          Te_pred += kOpacityPredictorTheta * host_delta_E_prev[c_us] / heat_capacity;
        }
        const double dTe_lim = 0.5 * Te_curr;
        Te_pred = std::clamp(Te_pred, Te_curr - dTe_lim, Te_curr + dTe_lim);
        host_Te_pred[c_us] = std::max(Te_pred, te_floor);
      }

      host_Te_restore = host_Te;
      state.Te.copy_from_host(host_Te_pred.data());
      te_overridden = true;
    }
    mark_pre_subphase(pre_te_pred_ms);
    const auto nlte_device_result = radiation::compute_nlte_coefficients_cuda(
        state.rho.data(),
        state.Te.data(),
        state.zbar.data(),
        state.cv_e.empty() ? nullptr : state.cv_e.data(),
        state.cell_is_void.data(),
        state.cell_is_void.size(),
        device_opacity_table_.view(),
        planck.device_view(),
        n_cells,
        n_groups,
        dt,
        std::max(mat.A, 1.0e-12),
        cfg.radiation.imc.alpha,
        mat.nlte_f_min,
        cfg.radiation.imc.f_max,
        mat.lambda_fd_delta_rel,
        mat.lambda_fd_abs_min,
        std::max(cfg.numerics.safety.opacity_cap, 0.0),
        mat.cv_e_override,
        std::max(cfg.numerics.floors.Te, 0.0),
        mat.ideal_gas_gamma - 1.0,
        cfg.radiation.imc.linearized_planck,
        mat.lambda_method == "freeze_opacity",
        cfg.radiation.imc.corrected_fleck,
        d_f,
        d_sigma_a,
        d_sigma_R,
        d_sigma_a_eff,
        d_sigma_s_eff,
        coeff_buf_.d_eta_cdf,
        coeff_buf_.d_eta,
        coeff_buf_.d_lambda_raw,
        0);
    mark_pre_subphase(pre_nlte_coeffs_ms);
    if (nlte_device_result.negative_alpha_clamp_count > 0) {
      static int warn_count = 0;
      ++warn_count;
      if (warn_count == 1 || warn_count % 100 == 0) {
        core::log_warning("NLTE: clamped " +
                          std::to_string(nlte_device_result.negative_alpha_clamp_count) +
                          " negative/non-finite alpha_g values to zero");
      }
    }
    if (nlte_device_result.negative_eta_clamp_count > 0) {
      static int warn_count = 0;
      ++warn_count;
      if (warn_count == 1 || warn_count % 100 == 0) {
        core::log_warning("NLTE: clamped " +
                          std::to_string(nlte_device_result.negative_eta_clamp_count) +
                          " negative/non-finite eta_g values to zero");
      }
    }
    if (nlte_device_result.nan_inf_count > 0) {
      static int warn_count = 0;
      ++warn_count;
      if (warn_count == 1 || warn_count % 100 == 0) {
        core::log_warning("NLTE: encountered " +
                          std::to_string(nlte_device_result.nan_inf_count) +
                          " non-finite coefficient intermediates");
      }
    }
    if (capture_nlte_host_coeffs) {
      nlte_coeffs = radiation::compute_nlte_coefficients(
          state, cfg, *nlte_table_, planck, n_cells, n_groups, dt);
    }
    mark_pre_subphase(pre_other_A_ms);
    if (te_overridden) {
      state.Te.copy_from_host(host_Te_restore.data());
    }
    d_eta_cdf = coeff_buf_.d_eta_cdf;
    mark_pre_subphase(pre_nlte_h2d_ms);
  } else {
    materials::OpacityEvalView opacity_view;
    opacity_view.rho = state.rho.data();
    opacity_view.Te = state.Te.data();
    opacity_view.sigma_a = d_sigma_a;
    opacity_view.sigma_R = d_sigma_R;
    opacity_view.n_cells = n_cells;
    opacity_view.n_groups = n_groups;
    opacity_view.group_bounds_eV = d_group_bounds;
    opacity_view.opacity_model = runtime_opacity_model;
    opacity_view.kappa_planck_const = std::max(0.0, mat.kappa_a_constant);
    opacity_view.kappa_rosseland_const = std::max(0.0, mat.kappa_a_constant);
    opacity_view.kappa_floor = std::max(cfg.numerics.safety.opacity_floor, 0.0);
    opacity_view.kappa_cap = std::max(cfg.numerics.safety.opacity_cap, 0.0);
    opacity_view.power_law_kappa0 = mat.opacity_power_law_kappa0_cm2_g;
    opacity_view.power_law_alpha_T = mat.opacity_power_law_alpha_T;
    opacity_view.power_law_lambda_rho = mat.opacity_power_law_lambda_rho;
    opacity_view.power_law_T_ref_eV = mat.opacity_power_law_T_ref_eV;
    opacity_view.power_law_rho_ref = mat.opacity_power_law_rho_ref_g_cc;
    materials::evaluate_opacity_cuda(opacity_view, &opacity_flags);
    state.radiation_device_flags.opacity_out_of_range =
        std::max(state.radiation_device_flags.opacity_out_of_range,
                 opacity_flags.opacity_out_of_range);

    FleckView fleck_view;
    fleck_view.rho = state.rho.data();
    fleck_view.Te = state.Te.data();
    fleck_view.zbar = state.zbar.data();
    fleck_view.state_cv_e = state.cv_e.empty() ? nullptr : state.cv_e.data();
    fleck_view.cell_is_void = state.cell_is_void.data();
    fleck_view.sigma_a = d_sigma_a;
    fleck_view.planck = planck.device_view();
    fleck_view.f_fleck = d_f;
    fleck_view.sigma_a_eff = d_sigma_a_eff;
    fleck_view.sigma_s_eff = d_sigma_s_eff;
    fleck_view.n_cells = n_cells;
    fleck_view.n_groups = n_groups;
    fleck_view.dt = dt;
    compute_fleck_and_sigma_eff_cuda(fleck_view, cfg);
    mark_pre_subphase(pre_fleck_build_ms);
  }

  const double spectral_bias_eta = std::clamp(cfg.radiation.imc.spectral_bias_eta, 0.0, 1.0);
  const double* d_emission_bias_cdf = nullptr;
  std::vector<double> host_sigma_t_bias;
  if (spectral_bias_eta > 0.0) {
    host_sigma_t_bias.resize(n_cell_groups_us, 0.0);
    cuda_check(cudaMemcpy(host_sigma_t_bias.data(),
                          d_sigma_R,
                          sizeof(double) * n_cell_groups_us,
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy sigma_R for spectral bias failed");
    const std::vector<double> host_emission_bias_cdf =
        build_spectral_emission_bias_cdf(state, cfg, planck, host_sigma_t_bias);
    cuda_check(cudaMemcpy(coeff_buf_.d_emission_bias_cdf,
                          host_emission_bias_cdf.data(),
                          sizeof(double) * n_cell_groups_us,
                          cudaMemcpyHostToDevice),
               "IMC::transport_step H2D emission_bias_cdf copy failed");
    d_emission_bias_cdf = coeff_buf_.d_emission_bias_cdf;
  }

  constexpr std::uint64_t kStepPrefixLimit = (1ULL << 24);
  TENRYU_ASSERT(state.step >= 0, "IMC step must be non-negative");
  TENRYU_ASSERT(static_cast<std::uint64_t>(state.step) < kStepPrefixLimit,
                "IMC global_id prefix overflow: step must be <= 2^24-1 (16777215)");
  const std::uint64_t step_base_gid = static_cast<std::uint64_t>(state.step) << 40;
  // TODO(MPI): Source emission still owns the low local-index range. Diffusion
  // exit/interface particles split the high reserved range with rank offsets.
  const std::uint64_t rank_particle_offset = 0ULL;
  const std::uint64_t global_id_base = step_base_gid | rank_particle_offset;
  parallel::DeviceArray d_diff_cell;
  parallel::DeviceArray d_diff_cell_prev;
  parallel::DeviceArray d_diff_entry_E;
  std::uint64_t diffusion_exit_gid_local_used = 0ULL;
  bool have_diffusion_device_masks = false;
  bool has_diffusion_cells = false;
  int n_diffusion_cells_step = 0;
  int n_diffusion_guard_cells_step = 0;

  if (face_current_tracking_enabled_1d) {
    ensure_diffusion_face_current_buffers(n_cells, n_groups);
  }

  if (diffusion_enabled_1d) {
    ensure_diffusion_energy_buffer(n_cells, n_groups);
    std::vector<double> host_sigma_R_diff(n_cell_groups_us, 0.0);
    cuda_check(cudaMemcpy(host_sigma_R_diff.data(),
                          d_sigma_R,
                          sizeof(double) * host_sigma_R_diff.size(),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy sigma_R for diffusion classification failed");
    std::vector<double> host_node_r_diff(state.x_r.size(), 0.0);
    state.x_r.copy_to_host(host_node_r_diff.data());
    classify_diffusion_cells(state,
                             cfg,
                             planck,
                             host_sigma_R_diff,
                             host_node_r_diff,
                             n_cells,
                             n_groups,
                             dt);
    n_diffusion_cells_step = static_cast<int>(
        std::count(diff_cell_.begin(), diff_cell_.end(), static_cast<std::uint8_t>(1U)));
    n_diffusion_guard_cells_step = static_cast<int>(
        std::count(diff_guard_cell_.begin(),
                   diff_guard_cell_.end(),
                   static_cast<std::uint8_t>(1U)));
    has_diffusion_cells = n_diffusion_cells_step > 0;
    diff_vol_.assign(n_cells_us, 0.0);
    state.vol.copy_to_host(diff_vol_.data());
    copy_u8_to_device(diff_cell_,
                      d_diff_cell,
                      "IMC::transport_step copy diffusion cell mask failed");
    copy_u8_to_device(diff_cell_prev_,
                      d_diff_cell_prev,
                      "IMC::transport_step copy previous diffusion cell mask failed");
    have_diffusion_device_masks = true;

    if (has_diffusion_cells) {
      DiffusionStepInputs diff_plan_in{};
      diff_plan_in.sigma_R = d_sigma_R;
      diff_plan_in.vol = state.vol.data();
      diff_plan_in.node_r = state.x_r.data();
      diff_plan_in.diff_cell = d_diff_cell.as<std::uint8_t>();
      diff_plan_in.n_cells = n_cells;
      diff_plan_in.n_groups = n_groups;
      diff_plan_in.dt = dt;
      diff_plan_in.bc_inner =
          diffusion_boundary_code_from_string(cfg.radiation.boundary.inner_r);
      diff_plan_in.bc_outer =
          diffusion_boundary_code_from_string(cfg.radiation.boundary.outer_r);
      const DiffusionStepPlan diff_plan =
          deterministic_diffusion_plan_1d(diff_plan_in,
                                          cfg.radiation.diffusion.sts_max_stages,
                                          cfg.radiation.diffusion.sts_subcycle_eta);
      if (diff_plan.skip) {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(6);
        oss << "[diffusion] step=" << state.step
            << " reverting diffusion mask to IMC: rkl2_stages="
            << diff_plan.rkl2_stages
            << " subcycles=" << diff_plan.rkl2_subcycles
            << " dt_explicit=" << diff_plan.dt_explicit;
        core::log_warning(oss.str());
        std::fill(diff_cell_.begin(), diff_cell_.end(), 0U);
        std::fill(diff_guard_cell_.begin(), diff_guard_cell_.end(), 0U);
        n_diffusion_cells_step = 0;
        n_diffusion_guard_cells_step = 0;
        has_diffusion_cells = false;
        copy_u8_to_device(diff_cell_,
                          d_diff_cell,
                          "IMC::transport_step copy reverted diffusion cell mask failed");
      }
    }

    int n_entering_cells = 0;
    int n_exiting_cells = 0;
    for (std::size_t c = 0; c < n_cells_us; ++c) {
      if (diff_cell_[c] != 0U && diff_cell_prev_[c] == 0U) {
        ++n_entering_cells;
      } else if (diff_cell_[c] == 0U && diff_cell_prev_[c] != 0U) {
        ++n_exiting_cells;
      }
    }

    if (n_entering_cells > 0) {
      d_diff_entry_E.resize(sizeof(double) * n_cell_groups_us);
      const DiffusionConversionStats entry_stats =
          fold_entering_diffusion_particles_cuda(pool_,
                                                 pool_.n_alive,
                                                 d_diff_cell.as<std::uint8_t>(),
                                                 d_diff_cell_prev.as<std::uint8_t>(),
                                                 state.vol.data(),
                                                 diff_E_.as<double>(),
                                                 d_diff_entry_E.as<double>(),
                                                 n_cells,
                                                 n_groups);
      if (entry_stats.n_particles > 0U) {
        const CompositeSortResult compacted =
            compact_alive_only(pool_, pool_.n_alive, n_cells, n_groups, nullptr);
        pool_.n_alive = compacted.n_alive;
        pool_.n_census = pool_.n_alive;
      }
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(6);
      oss << "[diffusion:entry] step=" << state.step
          << " cells=" << n_entering_cells
          << " particles=" << entry_stats.n_particles
          << " E_folded=" << entry_stats.energy;
      core::log_info(oss.str());
    }

    if (n_exiting_cells > 0) {
      std::vector<double> host_diff_E(n_cell_groups_us, 0.0);
      cuda_check(cudaMemcpy(host_diff_E.data(),
                            diff_E_.as<double>(),
                            sizeof(double) * host_diff_E.size(),
                            cudaMemcpyDeviceToHost),
                 "IMC::transport_step copy diff_E for exit conversion failed");
      const DiffusionConversionStats exit_stats =
          spawn_exiting_diffusion_particles_1d(state,
                                               cfg,
                                               pool_,
                                               max_pool_size,
                                               dt,
                                               step_base_gid,
                                               diffusion_exit_gid_local_used,
                                               diff_cell_,
                                               diff_cell_prev_,
                                               host_diff_E,
                                               diff_vol_,
                                               n_cells,
                                               n_groups,
                                               part.rank,
                                               part.n_ranks);
      diffusion_exit_gid_local_used += exit_stats.n_particles;
      zero_exiting_diffusion_energy_cuda(diff_E_.as<double>(),
                                         d_diff_cell.as<std::uint8_t>(),
                                         d_diff_cell_prev.as<std::uint8_t>(),
                                         n_cells,
                                         n_groups);
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(6);
      oss << "[diffusion:exit] step=" << state.step
          << " cells=" << n_exiting_cells
          << " particles=" << exit_stats.n_particles
          << " E_spawned=" << exit_stats.energy;
      core::log_info(oss.str());
    }
  }

  if (holo_enabled_1d) {
    std::vector<double> host_sigma_R_holo(n_cell_groups_us, 0.0);
    cuda_check(cudaMemcpy(host_sigma_R_holo.data(),
                          d_sigma_R,
                          sizeof(double) * host_sigma_R_holo.size(),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy sigma_R for HOLO selector failed");
    std::vector<double> host_node_r_holo(state.x_r.size(), 0.0);
    std::vector<double> host_mass_holo(state.mass.size(), 0.0);
    std::vector<double> host_Te_holo(state.Te.size(), 0.0);
    state.x_r.copy_to_host(host_node_r_holo.data());
    state.mass.copy_to_host(host_mass_holo.data());
    state.Te.copy_to_host(host_Te_holo.data());

    HoloSelectorConfig holo_cfg{};
    holo_cfg.enabled = true;
    holo_cfg.coupling_tau = cfg.radiation.holo.coupling_tau;
    holo_cfg.guard_cells = cfg.radiation.holo.guard_cells;
    holo_cfg.blend_cells = cfg.radiation.holo.blend_cells;
    holo_cfg.min_lo_cells = cfg.radiation.holo.min_lo_cells;
    holo_cfg.tau_on = cfg.radiation.holo.tau_on;
    holo_cfg.tau_off = cfg.radiation.holo.tau_off;
    holo_cfg.min_dwell_steps = cfg.radiation.holo.min_dwell_steps;
    holo_cfg.temperature_floor = cfg.numerics.floors.Te;

    HoloSelectorInputs holo_in{};
    holo_in.dimension = HoloGeometryDimension::Spherical1D;
    holo_in.n_cells = n_cells;
    holo_in.n_groups = n_groups;
    holo_in.step = state.step;
    holo_in.node_r = &host_node_r_holo;
    holo_in.mass = &host_mass_holo;
    holo_in.Te = &host_Te_holo;
    holo_in.sigma_R = &host_sigma_R_holo;
    holo_in.planck = &planck;
    holo_in.cell_is_void = &state.cell_is_void;

    HoloSelectorStateView holo_state{state.holo_core_mask,
                                     state.holo_patch_mask,
                                     state.holo_core_prev_mask,
                                     state.holo_hold_count,
                                     state.holo_dwell_count,
                                     state.holo_tau_R,
                                     state.holo_reduced_flux,
                                     state.holo_mass_q,
                                     state.holo_lo_weight,
                                     state.holo_core_mask_valid};
    last_holo_selector_diagnostics_ =
        update_holo_core_mask(holo_cfg, holo_in, holo_state);
    const bool holo_ale_invalidated = state.holo_ale_invalidated;
    const bool holo_E_LO_resized = state.holo_E_LO.size() != n_cell_groups_us;
    if (holo_E_LO_resized) {
      state.holo_E_LO.reset(n_cell_groups_us);
    }
    if (holo_E_LO_resized || state.step == 0 || holo_ale_invalidated) {
      if (state.rad_E.size() == n_cell_groups_us && n_cell_groups_us > 0U) {
        cuda_check(cudaMemcpy(state.holo_E_LO.data(),
                              state.rad_E.data(),
                              sizeof(double) * n_cell_groups_us,
                              cudaMemcpyDeviceToDevice),
                   "IMC::transport_step initialize holo_E_LO from rad_E failed");
      }
      initialize_holo_lo_state_cuda(state.holo_E_LO.data(), n_cells, n_groups);
      initialize_holo_lo_from_lte_cuda(state.holo_E_LO.data(),
                                       state.Te.data(),
                                       n_cells,
                                       n_groups);
    }
    const std::size_t n_holo_face_groups_us = (n_cells_us + 1U) * n_groups_us;
    const bool holo_F_LO_resized = state.holo_F_LO.size() != n_holo_face_groups_us;
    if (holo_F_LO_resized) {
      state.holo_F_LO.reset(n_holo_face_groups_us);
    }
    if (holo_F_LO_resized || state.step == 0 || holo_ale_invalidated) {
      state.holo_F_LO.fill(0.0);
    }
    if (state.holo_consistency_source.size() != n_cell_groups_us) {
      state.holo_consistency_source.reset(n_cell_groups_us);
    }
    if (cfg.main.verbosity != "quiet") {
      const auto& hd = last_holo_selector_diagnostics_;
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(6);
      oss << "[holo_selector] step=" << state.step
          << " n_lo_coupled=" << hd.n_core_cells
          << " n_entered=" << hd.n_entered
          << " n_exited=" << hd.n_exited
          << " n_hard_exited=" << hd.n_hard_exited
          << " n_island_rejected=" << hd.n_island_rejected
          << " tau_R_min=" << hd.tau_R_min
          << " tau_R_max=" << hd.tau_R_max
          << " RF_max=" << hd.reduced_flux_max;
      core::log_info(oss.str());
    }
    if (holo_ale_invalidated) {
      state.holo_ale_invalidated = false;
    }
  } else if (!sn_material_coupling_active && state.holo_core_mask_valid) {
    std::fill(state.holo_core_mask.begin(), state.holo_core_mask.end(), 0U);
    std::fill(state.holo_patch_mask.begin(), state.holo_patch_mask.end(), 0U);
    std::fill(state.holo_core_prev_mask.begin(), state.holo_core_prev_mask.end(), 0U);
    std::fill(state.holo_hold_count.begin(), state.holo_hold_count.end(), 0);
    std::fill(state.holo_dwell_count.begin(), state.holo_dwell_count.end(), 0);
    std::fill(state.holo_lo_weight.begin(), state.holo_lo_weight.end(), 0.0);
    state.holo_consistency_source.fill(0.0);
    state.holo_chi_filtered.fill(0.0);
    state.holo_core_mask_valid = false;
  }
  if (!sn_material_coupling_active && state.holo_ale_invalidated &&
      (!holo_enabled_1d || !holo_qd_solver)) {
    state.holo_ale_invalidated = false;
  }

  if (sn_material_coupling_active) {
    state.holo_core_mask.assign(n_cells_us, static_cast<std::uint8_t>(1U));
    state.holo_patch_mask.assign(n_cells_us, static_cast<std::uint8_t>(0U));
    state.holo_lo_weight.assign(n_cells_us, 1.0);
    state.holo_core_mask_valid = true;

    // HYDRA IMEX material coupling uses Fleck effective absorption/scattering.
    SNTransportGPUResult sn_result =
        solve_holo_sn_material_coupling(state,
                                        cfg,
                                        planck,
                                        mat,
                                        d_sigma_a,
                                        d_sigma_a_eff,
                                        d_sigma_s_eff,
                                        d_sigma_R,
                                        d_f,
                                        n_cells,
                                        n_groups,
                                        dt);
    last_holo_selector_diagnostics_ = HoloSelectorDiagnostics{};
    last_holo_selector_diagnostics_.active = true;
    last_holo_selector_diagnostics_.n_core_cells = n_cells;
    last_holo_lo_result_ = HoloLOResult{};
    last_holo_lo_result_.solver_iterations =
        (sn_result.lo_solver_iterations > 0 || sn_result.lo_failures != 0)
            ? sn_result.lo_solver_iterations
            : sn_result.iterations;
    last_holo_lo_result_.failures = sn_result.lo_failures;
    if (cfg.main.verbosity != "quiet") {
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(6);
      oss << "[holo_sn_gpu] step=" << state.step
          << " dim=" << state.mesh.dim
          << " cells=" << n_cells
          << " groups=" << n_groups
          << " dirs=" << sn_result.n_directions
          << " iterations=" << sn_result.iterations
          << " error=" << sn_result.convergence_error
          << " converged=" << (sn_result.converged ? 1 : 0)
          << " lo_iterations=" << sn_result.lo_solver_iterations
          << " lo_failures=" << sn_result.lo_failures;
      core::log_info(oss.str());
    }
  }

  const bool implicit_ddmc_diffusion_requested =
      cfg.radiation.ddmc.enabled && cfg.radiation.ddmc.implicit_diffusion;
  const std::string implicit_ddmc_diffusion_reason =
      implicit_ddmc_diffusion_requested
          ? implicit_ddmc_diffusion_unsupported_reason(cfg, state.mesh.dim, is_nlte_mode)
          : std::string{};
  const bool implicit_ddmc_diffusion_available =
      implicit_ddmc_diffusion_requested && implicit_ddmc_diffusion_reason.empty();
  if (implicit_ddmc_diffusion_requested && !implicit_ddmc_diffusion_available &&
      state.step == 0) {
    core::log_warning("DDMC implicit diffusion requested but unavailable: " +
                      implicit_ddmc_diffusion_reason +
                      "; falling back to particle DDMC transport");
  }
  std::vector<double> host_rad_E_prev;
  if (implicit_ddmc_diffusion_available) {
    host_rad_E_prev.assign(n_cell_groups_us, 0.0);
    if (state.rad_E.size() == n_cell_groups_us) {
      state.rad_E.copy_to_host(host_rad_E_prev.data());
    }
  }

  if (cfg.main.verbosity == "verbose") {
    std::vector<double> host_f(static_cast<std::size_t>(n_cells), 1.0);
    {
      const std::size_t fleck_groups =
          n_cell_groups_us / static_cast<std::size_t>(n_cells);
      cuda_check(cudaMemcpy2D(host_f.data(), sizeof(double),
                              d_f, sizeof(double) * fleck_groups,
                              sizeof(double), static_cast<std::size_t>(n_cells),
                              cudaMemcpyDeviceToHost),
                 "IMC::transport_step copy f for stats failed");
    }

    double f_min_val = 1.0;
    double f_sum = 0.0;
    int f_count = 0;
    std::vector<double> f_valid;
    f_valid.reserve(host_f.size());
    for (std::size_t i = 0; i < host_f.size(); ++i) {
      if (state.cell_is_void.empty() || !state.cell_is_void[i]) {
        const double fi = host_f[i];
        f_min_val = std::min(f_min_val, fi);
        f_sum += fi;
        ++f_count;
        f_valid.push_back(fi);
      }
    }
    last_f_min_ = f_min_val;
    last_f_mean_ = (f_count > 0) ? (f_sum / static_cast<double>(f_count)) : 1.0;
    if (!f_valid.empty()) {
      std::sort(f_valid.begin(), f_valid.end());
      const std::size_t idx_p95 = std::min(
          static_cast<std::size_t>(0.95 * static_cast<double>(f_valid.size())),
          f_valid.size() - 1);
      last_f_p95_ = f_valid[idx_p95];
    } else {
      last_f_p95_ = 1.0;
    }

    std::vector<double> host_sigma_R_stats(static_cast<std::size_t>(n_cell_groups), 0.0);
    cuda_check(cudaMemcpy(host_sigma_R_stats.data(),
                          d_sigma_R,
                          sizeof(double) * host_sigma_R_stats.size(),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy sigma_R for tau stats failed");

    std::vector<double> host_node_r_stats(state.x_r.size(), 0.0);
    state.x_r.copy_to_host(host_node_r_stats.data());

    int tau_gt1 = 0;
    int tau_gt2 = 0;
    int tau_gt3 = 0;
    int tau_gt4 = 0;
    for (int c = 0; c < n_cells; ++c) {
      double dx_c = 0.0;
      if (state.mesh.dim == 1) {
        dx_c = host_node_r_stats[static_cast<std::size_t>(c + 1)] -
               host_node_r_stats[static_cast<std::size_t>(c)];
      } else {
        const int ir = c / state.mesh.topo.nz;
        dx_c = host_node_r_stats[static_cast<std::size_t>(ir + 1)] -
               host_node_r_stats[static_cast<std::size_t>(ir)];
      }
      for (int g = 0; g < n_groups; ++g) {
        const double tau = host_sigma_R_stats[static_cast<std::size_t>(c) *
                                                  static_cast<std::size_t>(n_groups) +
                                              static_cast<std::size_t>(g)] *
                           dx_c;
        if (tau > 1.0) {
          ++tau_gt1;
        }
        if (tau > 2.0) {
          ++tau_gt2;
        }
        if (tau > 3.0) {
          ++tau_gt3;
        }
        if (tau > 4.0) {
          ++tau_gt4;
        }
      }
    }
    last_tau_gt1_ = tau_gt1;
    last_tau_gt2_ = tau_gt2;
    last_tau_gt3_ = tau_gt3;
    last_tau_gt4_ = tau_gt4;
  }

  mark_pre_subphase(pre_other_B_ms);

  bool have_reference_field = false;
  if (cfg.radiation.imc.difference.enabled) {
    const std::size_t E_ref_bytes = sizeof(double) * n_cell_groups_us;
    state.difference_W.reset(n_cells_us);
    state.difference_E_ref.reset(n_cell_groups_us);
    state.difference_residual_E.reset(n_cell_groups_us);
    const bool have_previous_reference =
        previous_reference_U_device_valid_ &&
        previous_reference_U_device_.size == E_ref_bytes;
    if (n_cell_groups_us > 0U) {
      prepare_difference_census_reference_cuda(
          pool_,
          (state.rad_E.size() == n_cell_groups_us) ? state.rad_E.data() : nullptr,
          state.vol.data(),
          have_previous_reference ? previous_reference_U_device_.as<double>() : nullptr,
          have_previous_reference,
          n_cells,
          n_groups,
          state.difference_residual_E.data(),
          difference_residual_workspace_);
    }
    const std::uint8_t* d_diffusion_cell =
        (diffusion_enabled_1d && have_diffusion_device_masks &&
         diff_cell_.size() == n_cells_us)
            ? d_diff_cell.as<std::uint8_t>()
            : nullptr;
    last_reference_field_diagnostics_ =
        compute_reference_field_diagnostics_cuda(state,
                                                 cfg,
                                                 planck.device_view(),
                                                 d_sigma_R,
                                                 (n_cell_groups_us > 0U &&
                                                  state.rad_E.size() == n_cell_groups_us)
                                                     ? state.rad_E.data()
                                                     : nullptr,
                                                 d_diffusion_cell,
                                                 n_cells,
                                                 n_groups,
                                                 (n_cell_groups_us > 0U)
                                                     ? state.difference_E_ref.data()
                                                     : nullptr,
                                                 (n_cells_us > 0U)
                                                     ? state.difference_W.data()
                                                     : nullptr,
                                                 diff_ref_face_delta_U_,
                                                 diff_U_ref_end_,
                                                 diff_E_ref_avg_);
    have_reference_field = state.difference_E_ref.size() == n_cell_groups_us;
    if (n_cell_groups_us > 0U) {
      cuda_check(cudaMemset(state.difference_residual_E.data(),
                            0,
                            sizeof(double) * n_cell_groups_us),
                 "IMC::transport_step zero difference residual scratch failed");
    }
    if (last_reference_field_diagnostics_.valid && cfg.main.verbosity == "verbose") {
      const auto& rd = last_reference_field_diagnostics_;
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(6);
      oss << "[difference_ref] step=" << state.step
          << " eligible=" << rd.eligible_cells
          << " active=" << rd.active_cells
          << " strong=" << rd.strong_cells
          << " hybrid_suppressed=" << rd.hybrid_suppressed_cells
          << " W_min=" << rd.W_min
          << " W_mean=" << rd.W_mean
          << " W_max=" << rd.W_max
          << " tau_mean=" << rd.tau_mean
          << " tau_max=" << rd.tau_max
          << " chi_mean=" << rd.chi_mean
          << " chi_max=" << rd.chi_max
          << " reduced_flux_max=" << rd.reduced_flux_max
          << " knudsen_max=" << rd.knudsen_max
          << " front_grad_Te_max=" << rd.front_grad_Te_max
          << " front_grad_rho_max=" << rd.front_grad_rho_max
          << " E_ref_total=" << rd.E_ref_total;
      core::log_info(oss.str());
    }
  }
  if (cfg.radiation.imc.difference.enabled) {
    mark_pre_subphase(pre_difference_setup_ms);
  } else {
    mark_pre_subphase(pre_other_B_ms);
  }

  state.rad_dep.reset(static_cast<std::size_t>(n_cell_groups));
  state.rad_E.reset(static_cast<std::size_t>(n_cell_groups));
  state.rad_emit.reset(static_cast<std::size_t>(n_cell_groups));
  state.ddmc_mode_map.assign(static_cast<std::size_t>(n_cell_groups), static_cast<std::int8_t>(0));
  state.ddmc_mode_map_valid = true;
  const auto mark_diffusion_cells_in_state_mode_map = [&]() {
    if (!diffusion_enabled_1d || diff_cell_.size() != n_cells_us ||
        state.ddmc_mode_map.size() != n_cell_groups_us) {
      return;
    }
    for (int c = 0; c < n_cells; ++c) {
      if (diff_cell_[static_cast<std::size_t>(c)] == 0U) {
        continue;
      }
      const std::size_t base =
          static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups);
      for (int g = 0; g < n_groups; ++g) {
        state.ddmc_mode_map[base + static_cast<std::size_t>(g)] =
            static_cast<std::int8_t>(TransportMode::Diffusion);
      }
    }
  };
  mark_diffusion_cells_in_state_mode_map();
  const bool difference_source_supported =
      cfg.radiation.imc.difference.enabled &&
      !cfg.radiation.imc.linearized_planck &&
      (!is_nlte_mode || nlte_table_ != nullptr) &&
      have_reference_field &&
      n_cell_groups_us > 0U;
  if (cfg.radiation.imc.difference.enabled && !difference_source_supported &&
      cfg.radiation.imc.linearized_planck) {
    static bool warned_difference_source_mode = false;
    if (!warned_difference_source_mode) {
      core::log_warning(
          "Radiation.imc.difference PR4/PR5 source and census residualization currently "
          "do not support the linearized Planck source path; using diagnostics-only "
          "difference behavior for this mode");
      warned_difference_source_mode = true;
    }
  }
  if (difference_source_supported) {
    const std::size_t E_ref_bytes = sizeof(double) * n_cell_groups_us;
    previous_reference_U_device_.resize(E_ref_bytes);
    const std::uint64_t empty_gid_base =
        compute_census_residual_gid_base(step_base_gid,
                                         static_cast<std::int64_t>(n_cell_groups_us),
                                         part.rank,
                                         part.n_ranks);
    const DifferenceResidualizationDeviceStats residual_stats =
        residualize_census_against_reference_cuda(
            state,
            pool_,
            max_pool_size,
            dt,
            static_cast<std::uint64_t>(state.step),
            cfg.main.seed,
            empty_gid_base,
            state.difference_E_ref.data(),
            state.vol.data(),
            previous_reference_U_device_.as<double>(),
            n_cells,
            n_groups,
            difference_residual_pool_,
            difference_residual_workspace_,
            difference_residual_scan_workspace_);
    previous_reference_U_device_valid_ = previous_reference_U_device_.size == E_ref_bytes;
    previous_reference_U_.clear();
    if (cfg.main.verbosity == "verbose") {
      core::log_info("[difference_census] step=" + std::to_string(state.step) +
                     " before=" + std::to_string(residual_stats.n_before) +
                     " after=" + std::to_string(residual_stats.n_after) +
                     " scaled_bins=" + std::to_string(residual_stats.scaled_bins) +
                     " rebuilt_bins=" + std::to_string(residual_stats.rebuilt_bins) +
                     " killed_bins=" + std::to_string(residual_stats.killed_bins) +
                     " empty_created=" + std::to_string(residual_stats.empty_created));
    }
  } else if (cfg.radiation.imc.difference.enabled) {
    previous_reference_U_.clear();
    previous_reference_U_device_valid_ = false;
    state.difference_residual_E.reset(0);
  }
  state.rad_emit.fill(0.0);
  zero_tallies_cuda(state.rad_dep.data(),
                    d_rad_E_tally,
                    d_E_escape,
                    n_cells,
                    n_groups,
                    d_holo_Prr_tally,
                    d_holo_Prr_coverage_tally);
  mark_pre_subphase(pre_other_C_ms);
  const double* d_E_ref_start = nullptr;
  const double* d_E_ref_avg_for_finalize = nullptr;
  double reference_boundary_source_step = 0.0;
  if (difference_source_supported) {
    const std::size_t E_ref_bytes = sizeof(double) * n_cell_groups_us;
    TENRYU_ASSERT(state.difference_E_ref.size() == n_cell_groups_us,
                  "IMC::transport_step difference E_ref_start size mismatch");
    d_E_ref_start = state.difference_E_ref.data();
    d_E_ref_avg_for_finalize = d_E_ref_start;
    preseed_reference_absorption_cuda(state.rad_dep.data(),
                                      d_sigma_a_eff,
                                      d_E_ref_start,
                                      state.vol.data(),
                                      n_cells,
                                      n_groups,
                                      dt);
    if (cfg.radiation.imc.difference.face_transport && state.mesh.dim == 1) {
      const std::size_t ref_face_current_bytes =
          sizeof(double) * (n_cells_us + 1U) * n_groups_us;
      diff_ref_face_delta_U_.resize(E_ref_bytes);
      diff_ref_face_current_.resize(ref_face_current_bytes);
      diff_U_ref_end_.resize(E_ref_bytes);
      diff_E_ref_avg_.resize(E_ref_bytes);
      const ReferenceFaceTransportResult ref_transport =
          reference_face_transport_1d_cuda(diff_U_ref_end_.as<double>(),
                                           diff_ref_face_delta_U_.as<double>(),
                                           diff_E_ref_avg_.as<double>(),
                                           d_E_ref_start,
                                           d_sigma_R,
                                           state.x_r.data(),
                                           state.vol.data(),
                                           nullptr,
                                           nullptr,
                                           n_cells,
                                           n_groups,
                                           dt,
                                           boundary_code_from_string(cfg.radiation.boundary.inner_r),
                                           boundary_code_from_string(cfg.radiation.boundary.outer_r),
                                           &diff_ref_boundary_workspace_,
                                           diff_ref_face_current_.as<double>());
      d_E_ref_avg_for_finalize = diff_E_ref_avg_.as<double>();
      if (state.difference_E_ref.size() == n_cell_groups_us && n_cell_groups_us > 0U) {
        cuda_check(cudaMemcpy(state.difference_E_ref.data(),
                              d_E_ref_avg_for_finalize,
                              E_ref_bytes,
                              cudaMemcpyDeviceToDevice),
                   "IMC::transport_step copy difference E_ref_avg failed");
      }
      escaped_energy_total_ += std::max(ref_transport.E_escape, 0.0);
      reference_boundary_source_step = std::max(ref_transport.E_source, 0.0);
      previous_reference_U_device_.resize(E_ref_bytes);
      cuda_check(cudaMemcpy(previous_reference_U_device_.as<double>(),
                            diff_U_ref_end_.as<double>(),
                            E_ref_bytes,
                            cudaMemcpyDeviceToDevice),
                 "IMC::transport_step copy U_ref_end failed");
      previous_reference_U_device_valid_ = true;
      previous_reference_U_.clear();
      if (cfg.main.verbosity == "verbose") {
        core::log_info("[difference_face] step=" + std::to_string(state.step) +
                       " E_escape=" + std::to_string(ref_transport.E_escape) +
                       " E_source=" + std::to_string(ref_transport.E_source));
      }
    }
  }
  if (cfg.radiation.imc.difference.enabled) {
    mark_pre_subphase(pre_difference_setup_ms);
  } else {
    mark_pre_subphase(pre_other_C_ms);
  }
  const std::size_t face_current_bytes =
      sizeof(double) * (n_cells_us + 1U) * n_groups_us;
  if (diffusion_enabled_1d && face_current_in_.size == face_current_bytes &&
      face_current_out_.size == face_current_bytes && face_current_bytes > 0U) {
    cuda_check(cudaMemset(face_current_in_.ptr, 0, face_current_bytes),
               "IMC::transport_step zero diffusion face current in failed");
    cuda_check(cudaMemset(face_current_out_.ptr, 0, face_current_bytes),
               "IMC::transport_step zero diffusion face current out failed");
  }

  bool diffusion_emergency_reverted = false;
  std::uint64_t diffusion_emergency_revert_particles = 0ULL;
  double diffusion_emergency_revert_E_out = 0.0;
  const auto force_diffusion_cells_to_imc =
      [&](const char* reason, const double E_reference, const double E_current) {
        if (!has_diffusion_cells || !have_diffusion_device_masks ||
            diff_cell_.size() != n_cells_us) {
          return;
        }

        const std::vector<std::uint8_t> revert_prev = diff_cell_;
        std::vector<std::uint8_t> all_imc(n_cells_us, 0U);
        std::vector<double> host_diff_E(n_cell_groups_us, 0.0);
        cuda_check(cudaMemcpy(host_diff_E.data(),
                              diff_E_.as<double>(),
                              sizeof(double) * host_diff_E.size(),
                              cudaMemcpyDeviceToHost),
                   "IMC::transport_step copy diff_E for emergency diffusion revert failed");

        const DiffusionConversionStats revert_stats =
            spawn_exiting_diffusion_particles_1d(state,
                                                 cfg,
                                                 pool_,
                                                 max_pool_size,
                                                 dt,
                                                 step_base_gid,
                                                 diffusion_exit_gid_local_used,
                                                 all_imc,
                                                 revert_prev,
                                                 host_diff_E,
                                                 diff_vol_,
                                                 n_cells,
                                                 n_groups,
                                                 part.rank,
                                                 part.n_ranks);
        diffusion_exit_gid_local_used += revert_stats.n_particles;
        diffusion_emergency_revert_particles += revert_stats.n_particles;
        diffusion_emergency_revert_E_out += revert_stats.energy;

        parallel::DeviceArray d_revert_cell;
        parallel::DeviceArray d_revert_prev;
        copy_u8_to_device(all_imc,
                          d_revert_cell,
                          "IMC::transport_step copy emergency diffusion revert mask failed");
        copy_u8_to_device(revert_prev,
                          d_revert_prev,
                          "IMC::transport_step copy emergency diffusion revert prev mask failed");
        zero_exiting_diffusion_energy_cuda(diff_E_.as<double>(),
                                           d_revert_cell.as<std::uint8_t>(),
                                           d_revert_prev.as<std::uint8_t>(),
                                           n_cells,
                                           n_groups);

        diff_cell_ = all_imc;
        std::fill(diff_guard_cell_.begin(), diff_guard_cell_.end(), 0U);
        n_diffusion_cells_step = 0;
        n_diffusion_guard_cells_step = 0;
        has_diffusion_cells = false;
        diffusion_emergency_reverted = true;
        copy_u8_to_device(diff_cell_,
                          d_diff_cell,
                          "IMC::transport_step copy emergency reverted diffusion mask failed");
        for (int c = 0; c < n_cells; ++c) {
          if (revert_prev[static_cast<std::size_t>(c)] == 0U) {
            continue;
          }
          const std::size_t base =
              static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups);
          for (int g = 0; g < n_groups; ++g) {
            const std::size_t idx = base + static_cast<std::size_t>(g);
            if (idx < state.ddmc_mode_map.size() &&
                state.ddmc_mode_map[idx] ==
                    static_cast<std::int8_t>(TransportMode::Diffusion)) {
              state.ddmc_mode_map[idx] = static_cast<std::int8_t>(0);
            }
          }
        }

        std::ostringstream oss;
        oss << std::scientific << std::setprecision(6);
        oss << "[diffusion_safety] step=" << state.step
            << " action=revert_to_imc"
            << " reason=" << reason
            << " E_reference=" << E_reference
            << " E_current=" << E_current
            << " growth_factor=" << kDiffusionEmergencyGrowthFactor
            << " particles_spawned=" << revert_stats.n_particles
            << " E_spawned=" << revert_stats.energy;
        core::log_warning(oss.str());
      };

  mark_pre_subphase(pre_other_D_ms);
  const auto t_step_start = verbose_imc_timing ? t_pre_subphase : Clock::time_point{};
  auto t_phase = t_step_start;
  (void)t_phase;
  const int N_start = std::max(pool_.n_alive, 0);  // census count at step start (before emission)
  if (state.mesh.dim == 1 && N_start > 0) {
    const auto reid = reidentify_finite_position_particles_1d_cuda(
        pool_, state.x_r.data(), n_cells, N_start);
    if (reid.updated > 0) {
      state.particle_sort_cache_invalidated = true;
    }
  }
  // Phase A: Source-side particle budget controller
  int ppcg_override = -1;
  if (cfg.radiation.imc.particle_budget > 0) {
    const std::int64_t budget = static_cast<std::int64_t>(cfg.radiation.imc.particle_budget);
    // pool_.n_alive holds the census count from the previous step (before emission)
    const std::int64_t n_census = static_cast<std::int64_t>(std::max(pool_.n_alive, 0));
    const std::int64_t n_cell_group_bins = static_cast<std::int64_t>(n_cell_groups);
    const std::int64_t n_emit_max =
        static_cast<std::int64_t>(cfg.radiation.imc.particles_per_cell_group) * n_cell_group_bins;
    // Minimum ppcg is 1/5 of nominal to preserve statistical accuracy
    const int ppcg_min = std::max(1, cfg.radiation.imc.particles_per_cell_group / 5);
    const std::int64_t n_emit_min = static_cast<std::int64_t>(ppcg_min) * n_cell_group_bins;
    const std::int64_t n_emit_target = std::max(budget - n_census, n_emit_min);
    const std::int64_t n_emit_clamped = std::min(n_emit_target, n_emit_max);
    ppcg_override = std::max(ppcg_min, static_cast<int>(n_emit_clamped / n_cell_group_bins));
    if (cfg.main.verbosity == "verbose") {
      core::log_info("[budget] n_census=" + std::to_string(n_census) +
                     " budget=" + std::to_string(budget) +
                     " n_emit_target=" + std::to_string(n_emit_target) +
                     " ppcg_eff=" + std::to_string(ppcg_override) +
                     " ppcg_nominal=" +
                     std::to_string(cfg.radiation.imc.particles_per_cell_group));
    }
  } else if (cfg.radiation.imc.census_comb.enabled && pop_ctrl_ppcg_applied_ >= 0) {
    // Predictive population controller with affine emission model.
    // Model: S_pred = (1 - rem) * (N_start + b_scaled * ppcg + E_fixed)
    // where E_fixed = marshak + volume (independent of ppcg).
    const auto& cc = cfg.radiation.imc.census_comb;
    const int N_max_eff = std::max(1, std::min(pool_.capacity, cc.max_particles));
    constexpr double kSoftFraction = 0.92;
    const double N_soft = kSoftFraction * N_max_eff;

    const double rem = std::clamp(pop_ctrl_rem_ema_, 1e-6, 0.95);
    const double rho = 1.0 - rem;
    const double b_scaled = std::max(pop_ctrl_b_scaled_ema_, 1.0);

    // Deterministic prediction of fixed (non-ppcg) emissions
    const int E_fixed_pred = cfg.radiation.boundary.marshak_particles;
    // Note: volume source also uses ppcg_nom (not ppcg_override) but is rarely
    // active in implosion problems. If needed, add its expected count here.

    // Solve affine model: rho * (N_start + b_scaled * ppcg + E_fixed) <= N_soft
    const double ppcg_cont =
        (N_soft / rho - static_cast<double>(N_start) - static_cast<double>(E_fixed_pred)) / b_scaled;

    const int ppcg_nom = cfg.radiation.imc.particles_per_cell_group;
    const int ppcg_min_quality = std::max(1, ppcg_nom / 5);

    // Adaptive floor relaxation: allow ppcg=1 in high-retention, near-cap regimes
    int ppcg_min = ppcg_min_quality;
    if (rem < 0.01 && N_start > static_cast<int>(0.85 * N_max_eff)) {
      // Streaming regime: equilibrium ppcg is tiny, quality floor would cause
      // repeated combing. Relax to 1.
      ppcg_min = 1;
    }

    // Deterministic dithering: realize fractional ppcg via Bresenham accumulator
    const double ppcg_clamped = std::clamp(ppcg_cont, static_cast<double>(ppcg_min),
                                           static_cast<double>(ppcg_nom));
    const int ppcg_floor = static_cast<int>(std::floor(ppcg_clamped));
    const double frac = ppcg_clamped - static_cast<double>(ppcg_floor);
    pop_ctrl_ppcg_accum_ += frac;
    int ppcg_desired = ppcg_floor;
    if (pop_ctrl_ppcg_accum_ >= 1.0) {
      ppcg_desired += 1;
      pop_ctrl_ppcg_accum_ -= 1.0;
    }
    ppcg_desired = std::clamp(ppcg_desired, ppcg_min, ppcg_nom);

    // Rate limit: fast decrease (safety), slow increase (stability)
    if (ppcg_desired < pop_ctrl_ppcg_applied_) {
      ppcg_override = std::max(ppcg_desired, pop_ctrl_ppcg_applied_ - 4);
    } else {
      ppcg_override = std::min(ppcg_desired, pop_ctrl_ppcg_applied_ + 1);
    }
    ppcg_override = std::clamp(ppcg_override, ppcg_min, ppcg_nom);
    pop_ctrl_ppcg_applied_ = ppcg_override;

    if (cfg.main.verbosity == "verbose") {
      core::log_info("[pop_ctrl] N_start=" + std::to_string(N_start) +
                     " rem=" + std::to_string(rem) +
                     " b_scaled=" + std::to_string(b_scaled) +
                     " E_fixed=" + std::to_string(E_fixed_pred) +
                     " ppcg_cont=" + std::to_string(ppcg_cont) +
                     " ppcg_desired=" + std::to_string(ppcg_desired) +
                     " ppcg_eff=" + std::to_string(ppcg_override) +
                     " ppcg_min=" + std::to_string(ppcg_min) +
                     " ppcg_nom=" + std::to_string(ppcg_nom));
    }
  }
  DiffusionSourceSolveResult diffusion_source_result{};
  const auto accumulate_diffusion_source_result =
      [](DiffusionSourceSolveResult* total,
         const DiffusionSourceSolveResult& step) {
        total->max_newton_iter =
            std::max(total->max_newton_iter, step.max_newton_iter);
        total->n_failures += step.n_failures;
        total->matter_delta += step.matter_delta;
        total->rad_delta += step.rad_delta;
      };
  const auto run_diffusion_source_solve = [&](const double dt_s) {
    DiffusionSourceSolveResult result{};
    if (!has_diffusion_cells || !have_diffusion_device_masks || !(dt_s > 0.0)) {
      return result;
    }
    const bool use_table_eos_for_diffusion_source =
        mat.eos_tables != nullptr && mat.hydro_eos_backend != "exact_ideal_gas";
    TENRYU_ASSERT(!use_table_eos_for_diffusion_source,
                  "Radiation.diffusion local source solve currently supports "
                  "ideal-gas EOS only");

    const double A = std::max(mat.A, 1.0e-12);
    const double gm1 = std::max(mat.ideal_gas_gamma - 1.0, 1.0e-12);
    DiffusionSourceSolveInputs source_in{};
    source_in.diff_E = diff_E_.as<double>();
    source_in.ee = state.ee.data();
    source_in.Te = state.Te.data();
    source_in.Pe = state.Pe.data();
    source_in.sigma_P = d_sigma_a;
    source_in.vol = state.vol.data();
    source_in.rho = state.rho.data();
    source_in.mass = state.mass.data();
    source_in.cv_e = state.cv_e.empty() ? nullptr : state.cv_e.data();
    source_in.diff_cell = d_diff_cell.as<std::uint8_t>();
    source_in.rad_dep = state.rad_dep.data();
    source_in.rad_emit = state.rad_emit.data();
    source_in.n_cells = n_cells;
    source_in.n_groups = n_groups;
    source_in.dt_s = dt_s;
    source_in.planck = planck.device_view();
    source_in.cv_e_const =
        core::constants::eV_to_erg / (A * core::constants::proton_mass * gm1);
    source_in.pressure_gamma_minus_one = gm1;
    source_in.temperature_floor_eV = cfg.numerics.floors.Te;
    source_in.use_table_eos = false;
    return diffusion_source_solve_cuda(source_in);
  };
  const double diffusion_E_balance_start =
      has_diffusion_cells
          ? compute_diffusion_energy_host(diff_E_, diff_cell_, diff_vol_)
          : 0.0;
  double diffusion_E_after_source1 = diffusion_E_balance_start;
  double diffusion_E_before_source2 = 0.0;
  double diffusion_E_after_source2 = 0.0;
  if (has_diffusion_cells && have_diffusion_device_masks) {
    const double E_before_source1 =
        compute_diffusion_energy_host(diff_E_, diff_cell_, diff_vol_);
    const DiffusionSourceSolveResult source1_result =
        run_diffusion_source_solve(0.5 * dt);
    diffusion_E_after_source1 =
        compute_diffusion_energy_host(diff_E_, diff_cell_, diff_vol_);
    log_diffusion_substep_energy(state.step,
                                 "source1",
                                 E_before_source1,
                                 diffusion_E_after_source1,
                                 0.0,
                                 0.0,
                                 0.0,
                                 source1_result.matter_delta,
                                 source1_result.n_failures);
    accumulate_diffusion_source_result(&diffusion_source_result, source1_result);
    const double growth_reference =
        diffusion_growth_reference(E_before_source1,
                                   0.0,
                                   0.0,
                                   0.0,
                                   source1_result.matter_delta);
    if (diffusion_energy_growth_exceeded(growth_reference,
                                         diffusion_E_after_source1)) {
      force_diffusion_cells_to_imc("source1_growth",
                                   growth_reference,
                                   diffusion_E_after_source1);
    }
  }
  SourceStats thermal_stats{};
  std::vector<double> source_tilt;
  const std::vector<double>* source_tilt_ptr = nullptr;
  const bool source_localization_requested = cfg.radiation.imc.source_localization;
  const bool source_localization_supported =
      source_localization_requested && state.mesh.dim == 1;
  if (source_localization_requested && !source_localization_supported) {
    static bool warned_source_localization_dim = false;
    if (!warned_source_localization_dim) {
      core::log_warning(
          "Radiation.imc.source_localization currently supports 1D_SPH only; "
          "falling back to legacy thermal source sampling");
      warned_source_localization_dim = true;
    }
  }
  if (source_localization_supported) {
    sloc_abs_wr_.reset(n_cells_us);
    sloc_abs_wr2_.reset(n_cells_us);
    sloc_abs_E_.reset(n_cells_us);
    sloc_mean_r_.reset(n_cells_us);
    sloc_sigma_.reset(n_cells_us);
    sloc_alpha_.reset(n_cells_us);
    sloc_prev_E_.reset(n_cells_us);
    cuda_check(cudaMemset(sloc_abs_wr_.data(), 0, sizeof(double) * n_cells_us),
               "IMC::transport_step cudaMemset sloc_abs_wr failed");
    cuda_check(cudaMemset(sloc_abs_wr2_.data(), 0, sizeof(double) * n_cells_us),
               "IMC::transport_step cudaMemset sloc_abs_wr2 failed");
    cuda_check(cudaMemset(sloc_abs_E_.data(), 0, sizeof(double) * n_cells_us),
               "IMC::transport_step cudaMemset sloc_abs_E failed");
  }
  if (cfg.radiation.imc.source_tilting) {
    if (state.mesh.dim == 1) {
      source_tilt = compute_imc_source_tilt_1d(state, cfg);
      source_tilt_ptr = &source_tilt;
    } else if (state.mesh.dim == 2) {
      source_tilt = compute_imc_source_tilt_2d(state, cfg);
      source_tilt_ptr = &source_tilt;
    }
  }
  const std::vector<double>* nlte_eta_ptr = nullptr;
  const std::vector<double>* nlte_f_ptr = nullptr;
  const double* d_nlte_eta_source = is_nlte_mode ? coeff_buf_.d_eta : nullptr;
  const double* d_nlte_f_source = is_nlte_mode ? d_f : nullptr;
  const std::vector<std::uint8_t>* source_skip_cell_ptr =
      (diffusion_enabled_1d && have_diffusion_device_masks) ? &diff_cell_ : nullptr;
  thermal_stats = IMCSource::emit_thermal(state,
                                          cfg,
                                          planck,
                                          d_sigma_a_eff,
                                          pool_,
                                          max_pool_size,
                                          dt,
                                          static_cast<std::uint64_t>(state.step),
                                          cfg.main.seed,
                                          global_id_base,
                                          nlte_eta_ptr,
                                          nlte_f_ptr,
                                          source_tilt_ptr,
                                          source_localization_supported ? sloc_mean_r_.data()
                                                                        : nullptr,
                                          source_localization_supported ? sloc_sigma_.data()
                                                                        : nullptr,
                                          source_localization_supported ? sloc_alpha_.data()
                                                                        : nullptr,
                                          source_localization_supported ? sloc_prev_E_.data()
                                                                        : nullptr,
                                          ppcg_override,
                                          d_emission_bias_cdf,
                                          source_skip_cell_ptr,
                                          d_E_ref_start,
                                          d_nlte_eta_source,
                                          d_nlte_f_source);
  const SourceStats marshak_stats = IMCSource::emit_marshak(state,
                                                            cfg,
                                                            planck.device_view(),
                                                            pool_,
                                                            max_pool_size,
                                                            dt,
                                                            static_cast<std::uint64_t>(state.step),
                                                            cfg.main.seed,
                                                            global_id_base + static_cast<std::uint64_t>(thermal_stats.n_thermal));
  last_marshak_in_step_ =
      std::max(marshak_stats.E_marshak, 0.0) + reference_boundary_source_step;
  SourceStats volume_stats{};
  if (cfg.radiation.volume_source_rate > 0.0 &&
      cfg.radiation.volume_source_x_max > 0.0) {
    volume_stats = IMCSource::emit_volume_source(
        state,
        cfg,
        pool_,
        max_pool_size,
        dt,
        static_cast<std::uint64_t>(state.step),
        cfg.main.seed,
        global_id_base + static_cast<std::uint64_t>(thermal_stats.n_thermal +
                                                    marshak_stats.n_marshak));
    last_volume_source_step_ = std::max(volume_stats.E_thermal, 0.0);
  }
  if (cfg.main.verbosity == "verbose") {
    core::log_info("[diag:imc] E_thermal=" + std::to_string(thermal_stats.E_thermal) +
                   " E_thermal_lost=" +
                   std::to_string(thermal_stats.E_thermal_lost) +
                   " E_marshak=" + std::to_string(marshak_stats.E_marshak) +
                   " n_alive=" + std::to_string(pool_.n_alive));
  }
  const double emitted_energy =
      thermal_stats.E_thermal + marshak_stats.E_marshak + volume_stats.E_thermal;
  const int emitted_count = thermal_stats.n_thermal + marshak_stats.n_marshak +
                            volume_stats.n_thermal;
  const int n_alive_before_emit = pool_.n_alive - emitted_count;
  const int n_alive_after_emit = pool_.n_alive;
  last_n_total_ = static_cast<std::int64_t>(std::max(n_alive_after_emit, 0));
  const double E_avg = (emitted_count > 0)
                           ? (emitted_energy / static_cast<double>(emitted_count))
                           : (cfg.numerics.floors.Te * core::constants::eV_to_erg);
  const auto t_emit_end = verbose_imc_timing ? Clock::now() : Clock::time_point{};
  auto t_prep_subphase = t_emit_end;
  double prep_radlite_check_ms = 0.0;
  double prep_other_ms = 0.0;
  double prep_pgrw_prep_ms = 0.0;
  double prep_ddmc_input_ms = 0.0;
  double prep_ddmc_prep_ms = 0.0;
  double prep_sloc_prep_ms = 0.0;
  double prep_source_smooth_ms = 0.0;
  double prep_face_alloc_ms = 0.0;
  const auto mark_prep_subphase = [&](double& value) {
    if (verbose_imc_timing) {
      const auto t_now = Clock::now();
      value += elapsed_ms(t_prep_subphase, t_now);
      t_prep_subphase = t_now;
    }
  };

  const bool allow_ddmc = true;
  const bool allow_rad_lite = true;
  const bool rad_lite_nlte_auto =
      cfg.radiation.imc.rad_lite_mesh.nlte_auto && is_nlte_mode;
  const bool rad_lite_enabled =
      (cfg.radiation.imc.rad_lite_mesh.enabled || rad_lite_nlte_auto) &&
      !holo_enabled_1d;
  if (holo_enabled_1d &&
      (cfg.radiation.imc.rad_lite_mesh.enabled || rad_lite_nlte_auto) &&
      cfg.main.verbosity != "quiet") {
    static bool warned_holo_rad_lite = false;
    if (!warned_holo_rad_lite) {
      core::log_warning("HOLO global LO currently uses the hydro mesh; disabling "
                        "RadLite mesh while Radiation.holo.enabled=True");
      warned_holo_rad_lite = true;
    }
  }
  const double rad_lite_sigma_ratio_max =
      rad_lite_nlte_auto
          ? std::max(cfg.radiation.imc.rad_lite_mesh.sigma_ratio_max, 3.0)
          : cfg.radiation.imc.rad_lite_mesh.sigma_ratio_max;
  mark_prep_subphase(prep_radlite_check_ms);
  last_ddmc_mode_count_ = 0;
  last_imc_mode_count_ = 0;
  last_mmatrix_violations_ = 0;
  last_mmatrix_fallback_count_ = 0;
  last_omega_below_threshold_ = 0;

  DDMCPreparation ddmc_preparation{};
  DDMCDiagnostics ddmc_diag{};
  DDMCDiffusionSolveResult ddmc_diffusion_result{};
  DiffusionStepResult deterministic_diffusion_result{};
  DiffusionInterfaceResult diffusion_interface_result{};
  double diffusion_interface_E_in = 0.0;
  double diffusion_interface_E_in_tail = 0.0;
  std::int64_t folded_ddmc_census = 0;
  std::int64_t rw_converted_to_imc = 0;
  std::int64_t rw_converted_to_ddmc = 0;
  std::int64_t rw_census = 0;
  std::int64_t rw_leak_left = 0;
  std::int64_t rw_leak_right = 0;
  std::int64_t rw_escaped = 0;
  std::int64_t rw_processed = 0;
  std::vector<double> host_node_r;
  std::vector<double> host_node_z;
  std::vector<double> host_cell_vol;
  std::vector<double> host_rho;
  std::vector<double> host_Te;
  std::vector<double> host_cell_heat_capacity;
  std::vector<double> host_sigma_R;
  std::vector<double> host_sigma_a;
  std::vector<double> host_sigma_a_eff;
  std::vector<double> host_sigma_s_eff;
  std::vector<double> host_fleck;
  const bool pgrw_requested =
      (state.mesh.dim == 1) && (cfg.radiation.ddmc.tau_rw > 0.0) && !rad_lite_enabled;
  mark_prep_subphase(prep_other_ms);

  if (pgrw_requested) {
    pgrw_tables_.initialize();

    host_node_r.resize(state.x_r.size(), 0.0);
    host_cell_vol.resize(state.vol.size(), 0.0);
    host_rho.resize(state.rho.size(), 0.0);
    host_Te.resize(state.Te.size(), 0.0);
    host_sigma_a.resize(static_cast<std::size_t>(n_cell_groups), 0.0);
    host_sigma_s_eff.resize(static_cast<std::size_t>(n_cell_groups), 0.0);
    state.x_r.copy_to_host(host_node_r.data());
    state.vol.copy_to_host(host_cell_vol.data());
    state.rho.copy_to_host(host_rho.data());
    state.Te.copy_to_host(host_Te.data());
    cuda_check(cudaMemcpy(host_sigma_a.data(),
                          d_sigma_a,
                          sizeof(double) * host_sigma_a.size(),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy sigma_a to host for PGRW failed");
    cuda_check(cudaMemcpy(host_sigma_s_eff.data(),
                          d_sigma_s_eff,
                          sizeof(double) * host_sigma_s_eff.size(),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy sigma_s_eff to host for PGRW failed");

    std::vector<int> host_g_diff_end(n_cells_us, 0);
    std::vector<double> host_sigma_a_bar(n_cells_us, 0.0);
    std::vector<double> host_sigma_t_bar(n_cells_us, 0.0);
    std::vector<double> host_D_pgrw(n_cells_us, 0.0);
    std::vector<double> host_gamma_pgrw(n_cells_us, 0.0);
    const double tau_min = kPgrwGroupTauFactor * cfg.radiation.ddmc.tau_rw;
    const bool emit_pgrw_diag = (state.step == 0) || ((state.step % 100) == 0);
    int diag_cell = -1;
    double diag_rho = -1.0;

    for (int c = 0; c < n_cells; ++c) {
      const std::size_t c_us = static_cast<std::size_t>(c);
      if (state.cell_is_void[c_us] != 0U) {
        continue;
      }

      const double r_lo = host_node_r[c_us];
      const double r_hi = host_node_r[c_us + 1];
      const double face_area =
          4.0 * kPi * (std::max(r_hi * r_hi, 0.0) + std::max(r_lo * r_lo, 0.0));
      const double vol_c = std::max(host_cell_vol[c_us], 0.0);
      const double s_bar = (face_area > 0.0) ? (4.0 * vol_c / face_area) : 0.0;
      if (!(s_bar > 0.0)) {
        continue;
      }

      const double Te_c = host_Te[c_us];
      double b_sum_diff = 0.0;
      double sigma_a_b_sum = 0.0;
      double sigma_t_b_sum = 0.0;
      double b_over_sigma_t_sum = 0.0;
      double sigma_p_total = 0.0;
      int g_cut = 0;
      for (int g = 0; g < n_groups; ++g) {
        const std::size_t idx =
            static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups) +
            static_cast<std::size_t>(g);
        const double bg = (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b_host(g, Te_c), 0.0);
        const double sigma_a_g = std::max(host_sigma_a[idx], 0.0);
        sigma_p_total += sigma_a_g * bg;

        const double sigma_t_g =
            std::max(host_sigma_a[idx] + host_sigma_s_eff[idx], 0.0);
        const double tau_g = sigma_t_g * s_bar;
        if (tau_g < tau_min) {
          break;
        }

        g_cut = g + 1;
        b_sum_diff += bg;
        sigma_a_b_sum += sigma_a_g * bg;
        sigma_t_b_sum += sigma_t_g * bg;
        if (sigma_t_g > 0.0) {
          b_over_sigma_t_sum += bg / sigma_t_g;
        }
      }

      if (g_cut <= 0 || !(b_sum_diff > 0.0) || !(sigma_p_total > 0.0)) {
        continue;
      }

      host_g_diff_end[c_us] = g_cut;
      host_sigma_a_bar[c_us] = sigma_a_b_sum / b_sum_diff;
      host_sigma_t_bar[c_us] = sigma_t_b_sum / b_sum_diff;
      host_D_pgrw[c_us] = (1.0 / 3.0) * b_over_sigma_t_sum / b_sum_diff;
      host_gamma_pgrw[c_us] = std::clamp(sigma_a_b_sum / sigma_p_total, 0.0, 1.0);

      if (emit_pgrw_diag) {
        const double rho_c =
            (std::isfinite(host_rho[c_us]) && host_rho[c_us] > 0.0) ? host_rho[c_us] : 0.0;
        if (diag_cell < 0 || rho_c > diag_rho) {
          diag_cell = c;
          diag_rho = rho_c;
        }
      }
    }

    if (emit_pgrw_diag && diag_cell >= 0) {
      const std::size_t c_us = static_cast<std::size_t>(diag_cell);
      const double R0_diag = 0.5 * std::max(host_node_r[c_us + 1] - host_node_r[c_us], 0.0);
      const double tau_sphere_diag = host_sigma_t_bar[c_us] * R0_diag;
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(6) << "[pgrw] step=" << state.step
          << ", cell=" << diag_cell << ", g_cut=" << host_g_diff_end[c_us]
          << ", sigma_a_bar=" << host_sigma_a_bar[c_us]
          << ", sigma_t_bar=" << host_sigma_t_bar[c_us] << ", D=" << host_D_pgrw[c_us]
          << ", gamma=" << host_gamma_pgrw[c_us] << ", tau_sphere=" << tau_sphere_diag;
      core::log_info(oss.str());
    }

    cuda_check(cudaMemcpy(d_g_diff_end,
                          host_g_diff_end.data(),
                          sizeof(int) * host_g_diff_end.size(),
                          cudaMemcpyHostToDevice),
               "IMC::transport_step copy g_diff_end to device failed");
    cuda_check(cudaMemcpy(d_sigma_a_bar,
                          host_sigma_a_bar.data(),
                          sizeof(double) * host_sigma_a_bar.size(),
                          cudaMemcpyHostToDevice),
               "IMC::transport_step copy sigma_a_bar to device failed");
    cuda_check(cudaMemcpy(d_sigma_t_bar,
                          host_sigma_t_bar.data(),
                          sizeof(double) * host_sigma_t_bar.size(),
                          cudaMemcpyHostToDevice),
               "IMC::transport_step copy sigma_t_bar to device failed");
    cuda_check(cudaMemcpy(d_D_pgrw,
                          host_D_pgrw.data(),
                          sizeof(double) * host_D_pgrw.size(),
                          cudaMemcpyHostToDevice),
               "IMC::transport_step copy D_pgrw to device failed");
    cuda_check(cudaMemcpy(d_gamma_pgrw,
                          host_gamma_pgrw.data(),
                          sizeof(double) * host_gamma_pgrw.size(),
                          cudaMemcpyHostToDevice),
               "IMC::transport_step copy gamma_pgrw to device failed");
  }
  mark_prep_subphase(prep_pgrw_prep_ms);

  if (cfg.radiation.ddmc.enabled && allow_ddmc) {
    host_node_r.resize(state.x_r.size(), 0.0);
    host_node_z.resize((state.mesh.dim == 2) ? state.x_z.size() : 0, 0.0);
    state.x_r.copy_to_host(host_node_r.data());
    if (state.mesh.dim == 2) {
      state.x_z.copy_to_host(host_node_z.data());
    }

    host_sigma_R.resize(static_cast<std::size_t>(n_cell_groups), 0.0);
    host_sigma_a.resize(static_cast<std::size_t>(n_cell_groups), 0.0);
    host_fleck.resize(static_cast<std::size_t>(n_cells), 0.0);
    cuda_check(cudaMemcpy(host_sigma_R.data(),
                          d_sigma_R,
                          sizeof(double) * host_sigma_R.size(),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy sigma_R to host failed");
    cuda_check(cudaMemcpy(host_sigma_a.data(),
                          d_sigma_a,
                          sizeof(double) * host_sigma_a.size(),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy sigma_a to host failed");
    {
      const std::size_t fleck_groups =
          n_cell_groups / static_cast<std::size_t>(n_cells);
      cuda_check(cudaMemcpy2D(host_fleck.data(), sizeof(double),
                              d_f, sizeof(double) * fleck_groups,
                              sizeof(double), static_cast<std::size_t>(n_cells),
                              cudaMemcpyDeviceToHost),
                 "IMC::transport_step copy fleck factor to host failed");
    }

    HysteresisState hysteresis;
    const bool use_hysteresis = !prev_ddmc_mode_.empty();
    if (use_hysteresis) {
      hysteresis.prev_mode = &prev_ddmc_mode_;
      hysteresis.prev_tau = &prev_ddmc_tau_;
      hysteresis.hold_count = &hysteresis_hold_count_;
    }
    std::vector<std::uint8_t> diffusion_force_imc_cells;
    const std::vector<std::uint8_t>* diffusion_force_imc_ptr = nullptr;
    if (diffusion_enabled_1d && diff_cell_.size() == n_cells_us &&
        diff_guard_cell_.size() == n_cells_us) {
      diffusion_force_imc_cells.assign(n_cells_us, 0U);
      for (std::size_t c = 0; c < n_cells_us; ++c) {
        diffusion_force_imc_cells[c] =
            (diff_cell_[c] != 0U || diff_guard_cell_[c] != 0U) ? 1U : 0U;
      }
      diffusion_force_imc_ptr = &diffusion_force_imc_cells;
    }
    mark_prep_subphase(prep_ddmc_input_ms);

    bool used_ddmc_all_imc_fast_path = false;
    const std::string& ddmc_leak_stencil = cfg.radiation.ddmc.leak_stencil;
    const bool ddmc_prep_supported =
        (state.mesh.dim == 1 &&
         (ddmc_leak_stencil == "4" || ddmc_leak_stencil == "9_kershaw")) ||
        (state.mesh.dim == 2 && ddmc_leak_stencil == "4");
    const bool previous_modes_all_imc =
        ddmc_prep_supported &&
        prev_ddmc_mode_.size() == n_cell_groups_us &&
        prev_ddmc_tau_.size() == n_cell_groups_us &&
        std::all_of(prev_ddmc_mode_.begin(),
                    prev_ddmc_mode_.end(),
                    [](const TransportMode mode) { return mode == TransportMode::IMC; });
    if (previous_modes_all_imc) {
      ModeSelectorConfig selector_cfg;
      selector_cfg.tau_ddmc = cfg.radiation.ddmc.tau_ddmc;
      selector_cfg.tau_rw = cfg.radiation.ddmc.tau_rw;
      selector_cfg.omega_ddmc = cfg.radiation.ddmc.omega_ddmc;
      if (cfg.radiation.ddmc.implicit_diffusion) {
        selector_cfg.tau_ddmc = std::min(selector_cfg.tau_ddmc, 1.0);
        selector_cfg.omega_ddmc = 0.0;
      }
      selector_cfg.tau_ddmc_off = cfg.radiation.ddmc.tau_ddmc_off;
      selector_cfg.omega_ddmc_off = cfg.radiation.ddmc.omega_ddmc_off;
      selector_cfg.mode_hold = cfg.radiation.ddmc.mode_hold;
      selector_cfg.rate_max = cfg.radiation.ddmc.rate_max;
      selector_cfg.emissivity_preserving = cfg.radiation.ddmc.emissivity_preserving;
      selector_cfg.sigma_floor = cfg.numerics.safety.opacity_floor;
      const double alpha_imc =
          (cfg.radiation.imc.alpha > 0.0) ? cfg.radiation.imc.alpha : 1.0;
      const CellRadiationCoeffs* ddmc_coeffs =
          (is_nlte_mode && nlte_coeffs.is_nlte) ? &nlte_coeffs : nullptr;

      ModeSelector mode_selector(n_cells, n_groups, selector_cfg);
      if (state.mesh.dim == 2) {
        mode_selector.compute_modes_2d_rz(host_node_r,
                                          host_node_z,
                                          state.mesh.topo.nr,
                                          state.mesh.topo.nz,
                                          host_sigma_R,
                                          host_fleck,
                                          host_sigma_a,
                                          {},
                                          ddmc_coeffs,
                                          dt,
                                          alpha_imc);
      } else {
        mode_selector.compute_modes(host_node_r,
                                    host_sigma_R,
                                    host_fleck,
                                    host_sigma_a,
                                    {},
                                    ddmc_coeffs,
                                    dt,
                                    alpha_imc);
      }

      double max_tau = 0.0;
      for (std::int64_t c = 0; c < n_cells; ++c) {
        for (int g = 0; g < n_groups; ++g) {
          max_tau = std::max(max_tau, mode_selector.get_tau(c, g));
        }
      }
      if (max_tau < selector_cfg.tau_ddmc) {
        const auto result = mode_selector.apply_hysteresis(prev_ddmc_mode_,
                                                           prev_ddmc_tau_,
                                                           hysteresis_hold_count_);
        hysteresis.switches_imc_to_ddmc = result.switches_imc_to_ddmc;
        hysteresis.switches_ddmc_to_imc = result.switches_ddmc_to_imc;
        if (diffusion_force_imc_ptr != nullptr) {
          for (std::int64_t c = 0; c < n_cells; ++c) {
            if ((*diffusion_force_imc_ptr)[static_cast<std::size_t>(c)] == 0U) {
              continue;
            }
            for (int g = 0; g < n_groups; ++g) {
              mode_selector.force_imc(c, g);
            }
          }
        }

        const std::int64_t ddmc_count = mode_selector.count_ddmc();
        const std::int64_t rw_count = mode_selector.count_rw();
        if (ddmc_count == 0 && rw_count == 0) {
          ddmc_preparation.mode_selector = std::move(mode_selector);
          ddmc_preparation.ddmc_mode_count = 0;
          ddmc_preparation.rw_mode_count = 0;
          ddmc_preparation.imc_mode_count = ddmc_preparation.mode_selector.count_imc();
          ddmc_preparation.omega_below_threshold =
              ddmc_preparation.mode_selector.count_omega_below_threshold();
          ddmc_preparation.active = true;
          used_ddmc_all_imc_fast_path = true;
        }
      }
    }

    if (!used_ddmc_all_imc_fast_path) {
      host_cell_vol.resize(state.vol.size(), 0.0);
      host_rho.resize(state.rho.size(), 0.0);
      host_Te.resize(state.Te.size(), 0.0);
      state.vol.copy_to_host(host_cell_vol.data());
      state.rho.copy_to_host(host_rho.data());
      state.Te.copy_to_host(host_Te.data());
      host_cell_heat_capacity.resize(state.vol.size(), 0.0);

      host_sigma_a_eff.resize(static_cast<std::size_t>(n_cell_groups), 0.0);
      cuda_check(cudaMemcpy(host_sigma_a_eff.data(),
                            d_sigma_a_eff,
                            sizeof(double) * host_sigma_a_eff.size(),
                            cudaMemcpyDeviceToHost),
                 "IMC::transport_step copy sigma_a_eff to host failed");

      std::vector<double> host_cv_e;
      const bool has_state_cv_e = !state.cv_e.empty();
      if (has_state_cv_e) {
        TENRYU_ASSERT(state.cv_e.size() == host_cell_heat_capacity.size(),
                      "IMC::transport_step state cv_e size mismatch for DDMC Picard");
        host_cv_e.resize(state.cv_e.size(), 0.0);
        state.cv_e.copy_to_host(host_cv_e.data());
      }

      std::vector<double> host_zbar;
      const bool need_zbar_fallback =
          mat.cv_e_override <= 0.0 && mat.eos_tables == nullptr;
      if (need_zbar_fallback) {
        host_zbar.resize(state.zbar.size(), 0.0);
        state.zbar.copy_to_host(host_zbar.data());
      }

      const double te_floor = cfg.numerics.floors.Te;
      const double A = std::max(mat.A, 1.0e-12);
      const double gm1 = std::max(mat.ideal_gas_gamma - 1.0, 1.0e-12);
      for (int c = 0; c < n_cells; ++c) {
        const std::size_t c_us = static_cast<std::size_t>(c);
        if (state.cell_is_void[c_us] != 0U) {
          continue;
        }

        const double rho_c =
            (std::isfinite(host_rho[c_us]) && host_rho[c_us] > 0.0) ? host_rho[c_us] : 1.0e-30;
        const double vol_c =
            (std::isfinite(host_cell_vol[c_us]) && host_cell_vol[c_us] > 0.0)
                ? host_cell_vol[c_us]
                : 0.0;
        const double Te_c =
            (std::isfinite(host_Te[c_us]) && host_Te[c_us] > te_floor) ? host_Te[c_us] : te_floor;

        double cv_mass_e = 0.0;
        if (mat.cv_e_override > 0.0) {
          cv_mass_e = mat.cv_e_override / rho_c;
        } else if (has_state_cv_e && host_cv_e[c_us] > 0.0) {
          cv_mass_e = host_cv_e[c_us];
        } else if (mat.eos_tables != nullptr) {
          cv_mass_e = std::max(mat.eos_tables->electron.cv(rho_c, Te_c), 0.0);
        } else {
          const double zbar_c =
              (c_us < host_zbar.size() && std::isfinite(host_zbar[c_us]) && host_zbar[c_us] > 0.0)
                  ? host_zbar[c_us]
                  : 0.0;
          cv_mass_e = zbar_c * core::constants::eV_to_erg /
                      (A * core::constants::proton_mass * gm1);
        }

        host_cell_heat_capacity[c_us] =
            rho_c * std::max(cv_mass_e, 0.0) * vol_c;
      }
      mark_prep_subphase(prep_ddmc_input_ms);

      ddmc_preparation = prepare_ddmc_step(
          pool_,
          cfg,
          static_cast<std::uint64_t>(state.step),
          dt,
          state.mesh.dim,
          n_cells,
          n_groups,
          host_node_r,
          host_node_z,
          state.mesh.topo.nr,
          state.mesh.topo.nz,
          host_cell_vol,
          host_rho,
          host_Te,
          host_sigma_R,
          host_sigma_a,
          host_sigma_a_eff,
          host_fleck,
          (is_nlte_mode && nlte_coeffs.is_nlte) ? &nlte_coeffs : nullptr,
          ddmc_boundary_type_from_string(cfg.radiation.boundary.inner_r),
          ddmc_boundary_type_from_string(cfg.radiation.boundary.outer_r),
          ddmc_boundary_type_from_string(cfg.radiation.boundary.bottom_z),
          ddmc_boundary_type_from_string(cfg.radiation.boundary.top_z),
          use_hysteresis ? &hysteresis : nullptr,
          diffusion_force_imc_ptr);
    }

    last_ddmc_mode_count_ = ddmc_preparation.ddmc_mode_count;
    last_imc_mode_count_ = ddmc_preparation.imc_mode_count;
    last_mmatrix_violations_ = ddmc_preparation.mmatrix.total_violations;
    // Current fallback path demotes violating DDMC cells to IMC.
    last_mmatrix_fallback_count_ = ddmc_preparation.mmatrix.total_violations;
    last_omega_below_threshold_ = ddmc_preparation.omega_below_threshold;
    if (ddmc_preparation.active &&
        (cfg.main.verbosity == "verbose" || state.step == 0)) {
      core::log_info("[ddmc_prep] step=" + std::to_string(state.step) +
                     " mode_ddmc=" + std::to_string(ddmc_preparation.ddmc_mode_count) +
                     " mode_rw=" + std::to_string(ddmc_preparation.rw_mode_count) +
                     " mode_imc=" + std::to_string(ddmc_preparation.imc_mode_count) +
                     " omega_below_threshold=" +
                     std::to_string(ddmc_preparation.omega_below_threshold) +
                     " mmatrix_violations=" +
                     std::to_string(ddmc_preparation.mmatrix.total_violations));
    }
    if (ddmc_preparation.active) {
      const auto& ms = ddmc_preparation.mode_selector;
      const std::size_t total = static_cast<std::size_t>(ms.n_cells()) *
                                static_cast<std::size_t>(ms.n_groups());
      prev_ddmc_mode_.assign(ms.modes().begin(), ms.modes().end());
      prev_ddmc_tau_.resize(total, 0.0);
      for (std::int64_t c = 0; c < ms.n_cells(); ++c) {
        for (int g = 0; g < ms.n_groups(); ++g) {
          prev_ddmc_tau_[static_cast<std::size_t>(c) * static_cast<std::size_t>(ms.n_groups()) +
                         static_cast<std::size_t>(g)] = ms.get_tau(c, g);
        }
      }
      if (hysteresis_hold_count_.size() != total) {
        hysteresis_hold_count_.assign(total, 0U);
      }
      last_switches_imc_to_ddmc_ = hysteresis.switches_imc_to_ddmc;
      last_switches_ddmc_to_imc_ = hysteresis.switches_ddmc_to_imc;
    }

    if (ddmc_preparation.active && ddmc_preparation.mmatrix.total_violations > 0) {
      core::log_warning("DDMC M-matrix fallback activated: violations=" +
                        std::to_string(ddmc_preparation.mmatrix.total_violations));
    }
  }
  mark_prep_subphase(prep_ddmc_prep_ms);
  if (need_source_smoothing_sigma_R) {
    if (host_sigma_R.empty()) {
      host_sigma_R.resize(static_cast<std::size_t>(n_cell_groups), 0.0);
      cuda_check(cudaMemcpy(host_sigma_R.data(),
                            d_sigma_R,
                            sizeof(double) * host_sigma_R.size(),
                            cudaMemcpyDeviceToHost),
                 "IMC::transport_step copy sigma_R to host for source smoothing failed");
    }
    last_sigma_R_max_.assign(n_cells_us, 0.0);
    for (int c = 0; c < n_cells; ++c) {
      const std::size_t base =
          static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups);
      double sigma_max = 0.0;
      for (int g = 0; g < n_groups; ++g) {
        sigma_max =
            std::max(sigma_max, std::max(host_sigma_R[base + static_cast<std::size_t>(g)], 0.0));
      }
      last_sigma_R_max_[static_cast<std::size_t>(c)] = sigma_max;
    }
  }
  mark_prep_subphase(prep_source_smooth_ms);
  const auto t_prep_end = verbose_imc_timing ? t_prep_subphase : Clock::time_point{};
  const bool use_implicit_ddmc_diffusion =
      implicit_ddmc_diffusion_available && ddmc_preparation.active &&
      ddmc_preparation.ddmc_mode_count > 0;

  CompositeSortResult sorted{};
  const bool need_transport_partition =
      ddmc_preparation.active &&
      (ddmc_preparation.ddmc_mode_count > 0 || ddmc_preparation.rw_mode_count > 0);
  if (need_transport_partition) {
    TENRYU_ASSERT(pool_.n_alive >= 0 && pool_.n_alive <= pool_.capacity,
                  "IMC::transport_step pool invariant violated before composite sort");
    sorted = composite_sort_and_partition(pool_,
                                          pool_.n_alive,
                                          n_cells,
                                          n_groups,
                                          d_E_numerical_loss);
    pool_.n_alive = sorted.n_alive;

    const auto& host_mode = ddmc_preparation.mode_selector.modes();
    TENRYU_ASSERT(static_cast<int>(host_mode.size()) == n_cell_groups,
                  "IMC::transport_step ddmc mode size mismatch");
    TENRYU_ASSERT(state.ddmc_mode_map.size() == host_mode.size(),
                  "IMC::transport_step state ddmc_mode_map size mismatch");
    for (std::size_t idx = 0; idx < host_mode.size(); ++idx) {
      state.ddmc_mode_map[idx] = static_cast<std::int8_t>(host_mode[idx]);
    }
    mark_diffusion_cells_in_state_mode_map();
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_mode),
                          sizeof(TransportMode) * host_mode.size()),
               "IMC::transport_step cudaMalloc ddmc_mode failed");
    cuda_check(cudaMemcpy(d_ddmc_mode,
                          host_mode.data(),
                          sizeof(TransportMode) * host_mode.size(),
                          cudaMemcpyHostToDevice),
               "IMC::transport_step copy ddmc_mode to device failed");

    if (state.mesh.dim == 1) {
      std::vector<double> host_cell_dx(static_cast<std::size_t>(n_cells), 0.0);
      TENRYU_ASSERT(static_cast<int>(host_node_r.size()) == n_cells + 1,
                    "IMC::transport_step node_r size mismatch for cell_dx");
      for (int c = 0; c < n_cells; ++c) {
        host_cell_dx[static_cast<std::size_t>(c)] =
            std::max(host_node_r[static_cast<std::size_t>(c + 1)] -
                         host_node_r[static_cast<std::size_t>(c)],
                     0.0);
      }
      d_cell_dx = coeff_buf_.d_cell_dx;
      cuda_check(cudaMemcpy(d_cell_dx,
                            host_cell_dx.data(),
                            sizeof(double) * host_cell_dx.size(),
                            cudaMemcpyHostToDevice),
                 "IMC::transport_step copy cell_dx to device failed");
    }

    d_interface_transitions = coeff_buf_.d_interface_transitions;
    d_interface_reflections = coeff_buf_.d_interface_reflections;
    d_conversion_prob_violations = coeff_buf_.d_conversion_prob_violations;
    cuda_check(cudaMemset(d_interface_transitions, 0, sizeof(unsigned long long)),
               "IMC::transport_step cudaMemset interface_transitions failed");
    cuda_check(cudaMemset(d_interface_reflections, 0, sizeof(unsigned long long)),
               "IMC::transport_step cudaMemset interface_reflections failed");
    cuda_check(cudaMemset(d_conversion_prob_violations, 0, sizeof(unsigned long long)),
               "IMC::transport_step cudaMemset conversion_prob_violations failed");
  } else {
    sorted.n_alive = pool_.n_alive;
    sorted.n_imc = pool_.n_alive;
    sorted.n_ddmc = 0;
  }
  const auto t_sort1_end = verbose_imc_timing ? Clock::now() : Clock::time_point{};

  last_n_imc_particles_ = static_cast<std::int64_t>(std::max(sorted.n_imc, 0));
  last_n_ddmc_particles_ = static_cast<std::int64_t>(std::max(sorted.n_ddmc, 0));
  if (last_n_total_ > 0) {
    last_ddmc_fraction_ =
        static_cast<double>(last_n_ddmc_particles_) / static_cast<double>(last_n_total_);
  } else {
    last_ddmc_fraction_ = 0.0;
  }
  // --- RadLite radiation mesh overlay (1D only) ---
  RadLiteMesh1D rad_lite;
  RadDeviceData rad_device_data;
  bool use_rad_lite = false;
  if (allow_rad_lite && state.mesh.dim == 1 && n_cells > 1 && rad_lite_enabled &&
      ddmc_preparation.rw_mode_count == 0) {
    // Get host copies of sigma_a_eff and sigma_s_eff
    std::vector<double> h_sigma_a_eff_rl(static_cast<std::size_t>(n_cell_groups), 0.0);
    std::vector<double> h_sigma_s_eff_rl(static_cast<std::size_t>(n_cell_groups), 0.0);
    std::vector<double> h_Te_rl(static_cast<std::size_t>(n_cells), 0.0);
    std::vector<double> h_vol_rl(static_cast<std::size_t>(n_cells), 0.0);
    std::vector<double> h_node_r_rl(static_cast<std::size_t>(n_cells + 1), 0.0);
    cuda_check(cudaMemcpy(h_sigma_a_eff_rl.data(),
                          d_sigma_a_eff,
                          sizeof(double) * h_sigma_a_eff_rl.size(),
                          cudaMemcpyDeviceToHost),
               "IMC RadLite copy sigma_a_eff failed");
    cuda_check(cudaMemcpy(h_sigma_s_eff_rl.data(),
                          d_sigma_s_eff,
                          sizeof(double) * h_sigma_s_eff_rl.size(),
                          cudaMemcpyDeviceToHost),
               "IMC RadLite copy sigma_s_eff failed");
    state.Te.copy_to_host(h_Te_rl.data());
    state.vol.copy_to_host(h_vol_rl.data());
    state.x_r.copy_to_host(h_node_r_rl.data());

    // DDMC mode (as int8_t for merge constraint)
    const int8_t* ddmc_mode_ptr = nullptr;
    std::vector<int8_t> h_ddmc_mode_rl;
    if (ddmc_preparation.active) {
      const auto& host_mode = ddmc_preparation.mode_selector.modes();
      h_ddmc_mode_rl.resize(host_mode.size());
      for (std::size_t idx = 0; idx < host_mode.size(); ++idx) {
        h_ddmc_mode_rl[idx] =
            (host_mode[idx] == TransportMode::DDMC) ? static_cast<int8_t>(1) : static_cast<int8_t>(0);
      }
      ddmc_mode_ptr = h_ddmc_mode_rl.data();
    }

    rad_lite = build_rad_lite_mesh(n_cells,
                                   n_groups,
                                   h_node_r_rl.data(),
                                   h_vol_rl.data(),
                                   h_sigma_a_eff_rl.data(),
                                   h_sigma_s_eff_rl.data(),
                                   h_Te_rl.data(),
                                   ddmc_mode_ptr,
                                   nullptr,  // material_id (single material -> nullptr allows all merges)
                                   rad_lite_sigma_ratio_max);

    if (rad_lite.enabled) {
      // Get additional host data for RadDeviceData
      std::vector<double> h_sigma_R_rl(static_cast<std::size_t>(n_cell_groups), 0.0);
      std::vector<double> h_sigma_a_rl(static_cast<std::size_t>(n_cell_groups), 0.0);
      std::vector<double> h_fleck_rl(static_cast<std::size_t>(n_cells), 0.0);
      cuda_check(cudaMemcpy(h_sigma_R_rl.data(),
                            d_sigma_R,
                            sizeof(double) * h_sigma_R_rl.size(),
                            cudaMemcpyDeviceToHost),
                 "IMC RadLite copy sigma_R failed");
      cuda_check(cudaMemcpy(h_sigma_a_rl.data(),
                            d_sigma_a,
                            sizeof(double) * h_sigma_a_rl.size(),
                            cudaMemcpyDeviceToHost),
                 "IMC RadLite copy sigma_a failed");
      cuda_check(cudaMemcpy(h_fleck_rl.data(),
                            d_f,
                            sizeof(double) * h_fleck_rl.size(),
                            cudaMemcpyDeviceToHost),
                 "IMC RadLite copy fleck failed");

      rad_device_data = prepare_rad_device_data(rad_lite,
                                                h_node_r_rl.data(),
                                                h_vol_rl.data(),
                                                h_sigma_R_rl.data(),
                                                h_sigma_a_rl.data(),
                                                h_fleck_rl.data(),
                                                ddmc_mode_ptr,
                                                nullptr);
      use_rad_lite = (rad_lite.enabled && rad_device_data.active && state.mesh.dim == 1);

      // Remap IMC particle cell_ids from hydro to rad
      if (sorted.n_imc > 0) {
        remap_cell_ids_to_rad_cuda(pool_.cell_id, rad_device_data.hydro_to_rad, sorted.n_imc);
        cuda_check(cudaDeviceSynchronize(), "IMC RadLite remap to rad sync failed");
      }

      core::log_info("[rad_lite] n_hydro=" + std::to_string(rad_lite.n_hydro) +
                     " n_rad=" + std::to_string(rad_lite.n_rad) +
                     " ratio=" + std::to_string(static_cast<double>(rad_lite.n_hydro) /
                                                std::max(rad_lite.n_rad, 1)));
    }
  } else if (allow_rad_lite && state.mesh.dim == 1 && n_cells > 1 && rad_lite_enabled &&
             ddmc_preparation.rw_mode_count > 0) {
    static bool warned_rad_lite_rw = false;
    if (!warned_rad_lite_rw) {
      core::log_warning(
          "RadLite mesh is disabled when RW transport is active; using hydro mesh for IMC");
      warned_rad_lite_rw = true;
    }
  }

  unsigned long long host_interface_transitions = 0ULL;
  unsigned long long host_interface_reflections = 0ULL;
  unsigned long long host_conversion_prob_violations = 0ULL;
  unsigned long long host_imc_absorbed = 0ULL;
  unsigned long long host_imc_escaped = 0ULL;
  unsigned long long host_diffusion_interface_kills = 0ULL;
  unsigned long long host_cnt_boundary = 0ULL;
  unsigned long long host_cnt_scatter = 0ULL;
  unsigned long long host_cnt_census = 0ULL;
  unsigned long long host_cnt_absorb_kill = 0ULL;
  unsigned long long host_cnt_absorb_survive = 0ULL;
  unsigned long long host_cnt_roulette_kill = 0ULL;
  const int n_imc_transport = sorted.n_imc;
  const bool enable_source_localization_transport = source_localization_supported;
  const bool face_current_tracking_active =
      face_current_tracking_enabled_1d && !use_rad_lite &&
      face_current_step_.ptr != nullptr &&
      face_current_step_.size == face_current_bytes;
  const bool diffusion_face_current_active =
      diffusion_enabled_1d && !use_rad_lite &&
      face_current_step_.ptr != nullptr &&
      face_current_step_.size == face_current_bytes &&
      face_current_in_.ptr != nullptr &&
      face_current_in_.size == face_current_bytes &&
      face_current_out_.ptr != nullptr &&
      face_current_out_.size == face_current_bytes;

  TransportInputs t_in;
  t_in.pool = &pool_;
  if (use_rad_lite) {
    t_in.sigma_a_eff = rad_device_data.sigma_a_eff;
    t_in.sigma_s_eff = rad_device_data.sigma_s_eff;
    t_in.Te = rad_device_data.Te;
    t_in.vol = rad_device_data.vol;
    t_in.node_r = rad_device_data.node_r;
  } else {
    t_in.sigma_a_eff = d_sigma_a_eff;
    t_in.sigma_s_eff = d_sigma_s_eff;
    t_in.Te = state.Te.data();
    t_in.vol = state.vol.data();
    t_in.node_r = state.x_r.data();
  }
  t_in.sloc_abs_wr = enable_source_localization_transport ? sloc_abs_wr_.data() : nullptr;
  t_in.sloc_abs_wr2 = enable_source_localization_transport ? sloc_abs_wr2_.data() : nullptr;
  t_in.sloc_abs_E = enable_source_localization_transport ? sloc_abs_E_.data() : nullptr;
  t_in.node_z = (state.mesh.dim == 2) ? state.x_z.data() : nullptr;
  t_in.nr = state.mesh.topo.nr;
  t_in.nz = state.mesh.topo.nz;
  t_in.mesh_dim = state.mesh.dim;
  t_in.ddmc_mode = use_rad_lite ? reinterpret_cast<const TransportMode*>(rad_device_data.ddmc_mode)
                                : d_ddmc_mode;
  t_in.ddmc_zero_flux_interfaces = false;
  t_in.n_groups_for_mode = n_groups;
  t_in.emissivity_preserving = cfg.radiation.ddmc.emissivity_preserving;
  t_in.sigma_R = use_rad_lite ? rad_device_data.sigma_R : d_sigma_R;
  t_in.cell_dx = use_rad_lite ? rad_device_data.cell_dx : d_cell_dx;
  t_in.fleck_f = use_rad_lite ? rad_device_data.fleck_f : d_f;
  t_in.sigma_a = use_rad_lite ? rad_device_data.sigma_a : d_sigma_a;
  if (use_rad_lite) {
    t_in.hydro_node_r = rad_device_data.hydro_node_r;
    t_in.rad_h_begin = rad_device_data.rad_h_begin;
    t_in.rad_h_end = rad_device_data.rad_h_end;
  }
  if (use_rad_lite && d_eta_cdf != nullptr) {
    t_in.eta_cdf = nullptr;
    t_in.eta_cdf_hydro = d_eta_cdf;
  } else {
    t_in.eta_cdf = use_rad_lite ? rad_device_data.eta_cdf : d_eta_cdf;
  }
  if (pgrw_requested && !use_rad_lite) {
    t_in.g_diff_end = d_g_diff_end;
    t_in.sigma_a_bar = d_sigma_a_bar;
    t_in.sigma_t_bar = d_sigma_t_bar;
    t_in.D_pgrw = d_D_pgrw;
    t_in.gamma_pgrw = d_gamma_pgrw;
    t_in.pgrw_leak_inv_cdf = pgrw_tables_.d_leak_inv_cdf;
    t_in.pgrw_leak_cdf_xi = pgrw_tables_.d_leak_cdf_xi;
    t_in.pgrw_pos_cdf = pgrw_tables_.d_pos_cdf;
    t_in.pgrw_leak_table_size = pgrw_tables_.leak_table_size;
    t_in.pgrw_pos_theta_bins = pgrw_tables_.pos_theta_bins;
    t_in.pgrw_pos_rho_bins = pgrw_tables_.pos_rho_bins;
    t_in.pgrw_theta_max = pgrw_tables_.theta_max;
    t_in.pgrw_tau_rw = cfg.radiation.ddmc.tau_rw;
  }
  t_in.planck = planck.device_view();
  t_in.inelastic_scatter = cfg.radiation.imc.inelastic_scatter;
  assign_scatter_bias_cdf_if_supported(t_in, d_emission_bias_cdf);
  t_in.rad_dep = use_rad_lite ? rad_device_data.rad_dep : state.rad_dep.data();
  t_in.rad_E_tally = use_rad_lite ? rad_device_data.rad_E_tally : d_rad_E_tally;
  t_in.holo_Prr_tally = use_rad_lite ? nullptr : d_holo_Prr_tally;
  t_in.holo_Prr_coverage_tally = use_rad_lite ? nullptr : d_holo_Prr_coverage_tally;
  t_in.face_current_step =
      face_current_tracking_active ? face_current_step_.as<double>() : nullptr;
  t_in.diff_cell =
      diffusion_face_current_active ? d_diff_cell.as<std::uint8_t>() : nullptr;
  t_in.diff_face_current_in =
      diffusion_face_current_active ? face_current_in_.as<double>() : nullptr;
  t_in.E_escape = d_E_escape;
  t_in.E_numerical_loss = d_E_numerical_loss;
  t_in.imc_absorbed = d_imc_absorbed;
  t_in.imc_escaped = d_imc_escaped;
  t_in.diffusion_interface_kills =
      diffusion_face_current_active ? d_diffusion_interface_kills : nullptr;
  t_in.cnt_boundary = d_cnt_boundary;
  t_in.cnt_scatter = d_cnt_scatter;
  t_in.cnt_census = d_cnt_census;
  t_in.cnt_absorb_kill = d_cnt_absorb_kill;
  t_in.cnt_absorb_survive = d_cnt_absorb_survive;
  t_in.cnt_roulette_kill = d_cnt_roulette_kill;
  t_in.n_cells = use_rad_lite ? rad_lite.n_rad : n_cells;
  t_in.n_groups = n_groups;
  t_in.n_imc = n_imc_transport;
  t_in.dt = dt;
  t_in.E_avg = E_avg;
  t_in.w_cutoff = cfg.radiation.imc.weight_cutoff;
  t_in.p_survival = std::clamp(cfg.radiation.imc.roulette_survival, 0.0, 1.0);
  t_in.f_cutoff = cfg.radiation.imc.cutoff_fraction;
  t_in.bc_inner = boundary_code_from_string(cfg.radiation.boundary.inner_r);
  t_in.bc_outer = boundary_code_from_string(cfg.radiation.boundary.outer_r);
  t_in.bc_bottom_z = boundary_code_from_string(cfg.radiation.boundary.bottom_z);
  t_in.bc_top_z = boundary_code_from_string(cfg.radiation.boundary.top_z);
  t_in.step_number = static_cast<std::uint64_t>(state.step);
  t_in.user_seed = cfg.main.seed;
  t_in.error_flags = d_error_flags;
  t_in.interface_transitions = d_interface_transitions;
  t_in.interface_reflections = d_interface_reflections;
  t_in.conversion_prob_violations = d_conversion_prob_violations;
  t_in.tail_pass = false;
  if (state.mesh.dim == 2) {
    imc_transport_2d_persistent_cuda(t_in, part);
  } else {
    imc_transport_persistent_cuda(t_in, part);
  }
  // --- RadLite: remap cell_ids and redistribute tallies ---
  if (use_rad_lite) {
    if (sorted.n_imc > 0) {
      remap_cell_ids_to_hydro_cuda(pool_.cell_id,
                                   pool_.pos_r,
                                   rad_device_data.hydro_node_r,
                                   rad_device_data.rad_h_begin,
                                   rad_device_data.rad_h_end,
                                   sorted.n_imc,
                                   rad_lite.n_rad);
      cuda_check(cudaDeviceSynchronize(), "IMC RadLite remap to hydro sync failed");
    }
    // Redistribute rad-indexed tallies to hydro-indexed arrays
    redistribute_tallies_cuda(state.rad_dep.data(),
                              d_rad_E_tally,
                              rad_device_data.rad_dep,
                              rad_device_data.rad_E_tally,
                              rad_device_data.hydro_to_rad,
                              rad_device_data.w_dep,
                              rad_device_data.w_tl,
                              n_cells,
                              n_groups);
    cuda_check(cudaDeviceSynchronize(), "IMC RadLite redistribute sync failed");
  }
  check_device_flags_after_stage(d_error_flags, "imc_transport");
  cuda_check(cudaMemcpy(&host_imc_absorbed,
                        d_imc_absorbed,
                        sizeof(host_imc_absorbed),
                        cudaMemcpyDeviceToHost),
             "IMC::transport_step copy imc_absorbed failed");
  cuda_check(cudaMemcpy(&host_imc_escaped,
                        d_imc_escaped,
                        sizeof(host_imc_escaped),
                        cudaMemcpyDeviceToHost),
             "IMC::transport_step copy imc_escaped failed");
  cuda_check(cudaMemcpy(&host_diffusion_interface_kills,
                        d_diffusion_interface_kills,
                        sizeof(host_diffusion_interface_kills),
                        cudaMemcpyDeviceToHost),
             "IMC::transport_step copy diffusion_interface_kills failed");
  cuda_check(cudaMemcpy(&host_cnt_boundary,
                        d_cnt_boundary,
                        sizeof(host_cnt_boundary),
                        cudaMemcpyDeviceToHost),
             "IMC::transport_step copy cnt_boundary failed");
  cuda_check(cudaMemcpy(&host_cnt_scatter,
                        d_cnt_scatter,
                        sizeof(host_cnt_scatter),
                        cudaMemcpyDeviceToHost),
             "IMC::transport_step copy cnt_scatter failed");
  cuda_check(cudaMemcpy(&host_cnt_census,
                        d_cnt_census,
                        sizeof(host_cnt_census),
                        cudaMemcpyDeviceToHost),
             "IMC::transport_step copy cnt_census failed");
  cuda_check(cudaMemcpy(&host_cnt_absorb_kill,
                        d_cnt_absorb_kill,
                        sizeof(host_cnt_absorb_kill),
                        cudaMemcpyDeviceToHost),
             "IMC::transport_step copy cnt_absorb_kill failed");
  cuda_check(cudaMemcpy(&host_cnt_absorb_survive,
                        d_cnt_absorb_survive,
                        sizeof(host_cnt_absorb_survive),
                        cudaMemcpyDeviceToHost),
             "IMC::transport_step copy cnt_absorb_survive failed");
  cuda_check(cudaMemcpy(&host_cnt_roulette_kill,
                        d_cnt_roulette_kill,
                        sizeof(host_cnt_roulette_kill),
                        cudaMemcpyDeviceToHost),
             "IMC::transport_step copy cnt_roulette_kill failed");

  last_thermal_lost_step_ = std::max(thermal_stats.E_thermal_lost, 0.0);

  if (d_interface_transitions != nullptr) {
    cuda_check(cudaMemcpy(&host_interface_transitions,
                          d_interface_transitions,
                          sizeof(host_interface_transitions),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy interface_transitions failed");
  }
  if (d_interface_reflections != nullptr) {
    cuda_check(cudaMemcpy(&host_interface_reflections,
                          d_interface_reflections,
                          sizeof(host_interface_reflections),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy interface_reflections failed");
  }
  if (d_conversion_prob_violations != nullptr) {
    cuda_check(cudaMemcpy(&host_conversion_prob_violations,
                          d_conversion_prob_violations,
                          sizeof(host_conversion_prob_violations),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy conversion_prob_violations failed");
  }
  last_interface_transitions_ =
      static_cast<std::uint64_t>(host_interface_transitions);
  last_interface_reflections_ =
      static_cast<std::uint64_t>(host_interface_reflections);
  last_conversion_prob_violations_ =
      static_cast<std::uint64_t>(host_conversion_prob_violations);
  last_cnt_boundary_ = static_cast<std::uint64_t>(host_cnt_boundary);
  last_cnt_scatter_ = static_cast<std::uint64_t>(host_cnt_scatter);
  last_cnt_census_ = static_cast<std::uint64_t>(host_cnt_census);
  last_cnt_absorb_kill_ = static_cast<std::uint64_t>(host_cnt_absorb_kill);
  last_cnt_absorb_survive_ = static_cast<std::uint64_t>(host_cnt_absorb_survive);
  last_cnt_roulette_kill_ = static_cast<std::uint64_t>(host_cnt_roulette_kill);
  const auto t_imc_end = verbose_imc_timing ? Clock::now() : Clock::time_point{};

  if (use_implicit_ddmc_diffusion) {
    ddmc_diffusion_result = solve_ddmc_diffusion_1d(ddmc_preparation.mode_selector,
                                                    ddmc_preparation.coefficients,
                                                    planck,
                                                    host_node_r,
                                                    host_cell_vol,
                                                    host_sigma_a_eff,
                                                    host_Te,
                                                    host_cell_heat_capacity,
                                                    host_rad_E_prev,
                                                    state.rad_dep.data(),
                                                    state.rad_emit.data(),
                                                    d_rad_E_tally,
                                                    d_E_escape,
                                                    dt);
    folded_ddmc_census = fold_ddmc_census_from_imc_partition_into_tally(pool_,
                                                                        n_imc_transport,
                                                                        n_cells,
                                                                        n_groups,
                                                                        dt,
                                                                        d_rad_E_tally);
    kill_particles_with_mode(pool_, sorted.n_alive, kModeDDMC);
    sorted.n_ddmc = 0;
    last_n_imc_particles_ = static_cast<std::int64_t>(std::max(sorted.n_imc, 0));
    last_n_ddmc_particles_ = 0;
    last_ddmc_fraction_ = 0.0;
    last_ddmc_to_imc_conversions_ = 0;
  } else if (ddmc_preparation.active && sorted.n_ddmc > 0) {
    const RankBoundaryParams1D rank_boundary_1d = make_rank_boundary_params_1d(part);
    const RankBoundaryParams2D rank_boundary_2d = make_rank_boundary_params_2d(part);
    if (state.mesh.dim == 1) {
      std::vector<double> host_sigma_leak_left(static_cast<std::size_t>(n_cell_groups), 0.0);
      std::vector<double> host_sigma_leak_right(static_cast<std::size_t>(n_cell_groups),
                                                0.0);
      std::vector<std::uint8_t> host_bc_left(static_cast<std::size_t>(n_cell_groups), 0U);
      std::vector<std::uint8_t> host_bc_right(static_cast<std::size_t>(n_cell_groups), 0U);
      for (int c = 0; c < n_cells; ++c) {
        for (int g = 0; g < n_groups; ++g) {
          const std::size_t key =
              static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups) +
              static_cast<std::size_t>(g);
          const auto& cell_data = ddmc_preparation.coefficients.get_cell_data(c, g);
          host_sigma_leak_left[key] = cell_data.sigma_leak_left;
          host_sigma_leak_right[key] = cell_data.sigma_leak_right;
          host_bc_left[key] = static_cast<std::uint8_t>(cell_data.bc_left);
          host_bc_right[key] = static_cast<std::uint8_t>(cell_data.bc_right);
        }
      }

      double* d_sigma_leak_left = nullptr;
      double* d_sigma_leak_right = nullptr;
      std::uint8_t* d_bc_left = nullptr;
      std::uint8_t* d_bc_right = nullptr;
      unsigned long long* d_ddmc_absorbed = nullptr;
      unsigned long long* d_ddmc_census = nullptr;
      unsigned long long* d_ddmc_leak_left = nullptr;
      unsigned long long* d_ddmc_leak_right = nullptr;
      unsigned long long* d_ddmc_leak_boundary = nullptr;
      unsigned long long* d_ddmc_vacuum_leak_left = nullptr;
      unsigned long long* d_ddmc_vacuum_leak_right = nullptr;
      unsigned long long* d_ddmc_converted_to_imc = nullptr;
      unsigned long long* d_ddmc_converted_to_rw = nullptr;
      unsigned long long* d_ddmc_sigma_tot_zero = nullptr;
      unsigned long long* d_ddmc_max_events_reached = nullptr;

    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_leak_left),
                          sizeof(double) * host_sigma_leak_left.size()),
               "IMC::transport_step cudaMalloc ddmc sigma_leak_left failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_leak_right),
                          sizeof(double) * host_sigma_leak_right.size()),
               "IMC::transport_step cudaMalloc ddmc sigma_leak_right failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_bc_left),
                          sizeof(std::uint8_t) * host_bc_left.size()),
               "IMC::transport_step cudaMalloc ddmc bc_left failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_bc_right),
                          sizeof(std::uint8_t) * host_bc_right.size()),
               "IMC::transport_step cudaMalloc ddmc bc_right failed");
    cuda_check(cudaMemcpy(d_sigma_leak_left,
                          host_sigma_leak_left.data(),
                          sizeof(double) * host_sigma_leak_left.size(),
                          cudaMemcpyHostToDevice),
               "IMC::transport_step copy ddmc sigma_leak_left failed");
    cuda_check(cudaMemcpy(d_sigma_leak_right,
                          host_sigma_leak_right.data(),
                          sizeof(double) * host_sigma_leak_right.size(),
                          cudaMemcpyHostToDevice),
               "IMC::transport_step copy ddmc sigma_leak_right failed");
    cuda_check(cudaMemcpy(d_bc_left,
                          host_bc_left.data(),
                          sizeof(std::uint8_t) * host_bc_left.size(),
                          cudaMemcpyHostToDevice),
               "IMC::transport_step copy ddmc bc_left failed");
    cuda_check(cudaMemcpy(d_bc_right,
                          host_bc_right.data(),
                          sizeof(std::uint8_t) * host_bc_right.size(),
                          cudaMemcpyHostToDevice),
               "IMC::transport_step copy ddmc bc_right failed");

    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_absorbed),
                          sizeof(unsigned long long)),
               "IMC::transport_step cudaMalloc ddmc_absorbed failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_census),
                          sizeof(unsigned long long)),
               "IMC::transport_step cudaMalloc ddmc_census failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_leak_left),
                          sizeof(unsigned long long)),
               "IMC::transport_step cudaMalloc ddmc_leak_left failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_leak_right),
                          sizeof(unsigned long long)),
               "IMC::transport_step cudaMalloc ddmc_leak_right failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_leak_boundary),
                          sizeof(unsigned long long)),
               "IMC::transport_step cudaMalloc ddmc_leak_boundary failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_vacuum_leak_left),
                          sizeof(unsigned long long)),
               "IMC::transport_step cudaMalloc ddmc_vacuum_leak_left failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_vacuum_leak_right),
                          sizeof(unsigned long long)),
               "IMC::transport_step cudaMalloc ddmc_vacuum_leak_right failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_converted_to_imc),
                          sizeof(unsigned long long)),
               "IMC::transport_step cudaMalloc ddmc_converted_to_imc failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_converted_to_rw),
                          sizeof(unsigned long long)),
               "IMC::transport_step cudaMalloc ddmc_converted_to_rw failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_sigma_tot_zero),
                          sizeof(unsigned long long)),
               "IMC::transport_step cudaMalloc ddmc_sigma_tot_zero failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_max_events_reached),
                          sizeof(unsigned long long)),
               "IMC::transport_step cudaMalloc ddmc_max_events_reached failed");

    cuda_check(cudaMemset(d_ddmc_absorbed, 0, sizeof(unsigned long long)),
               "IMC::transport_step cudaMemset ddmc_absorbed failed");
    cuda_check(cudaMemset(d_ddmc_census, 0, sizeof(unsigned long long)),
               "IMC::transport_step cudaMemset ddmc_census failed");
    cuda_check(cudaMemset(d_ddmc_leak_left, 0, sizeof(unsigned long long)),
               "IMC::transport_step cudaMemset ddmc_leak_left failed");
    cuda_check(cudaMemset(d_ddmc_leak_right, 0, sizeof(unsigned long long)),
               "IMC::transport_step cudaMemset ddmc_leak_right failed");
    cuda_check(cudaMemset(d_ddmc_leak_boundary, 0, sizeof(unsigned long long)),
               "IMC::transport_step cudaMemset ddmc_leak_boundary failed");
    cuda_check(cudaMemset(d_ddmc_vacuum_leak_left, 0, sizeof(unsigned long long)),
               "IMC::transport_step cudaMemset ddmc_vacuum_leak_left failed");
    cuda_check(cudaMemset(d_ddmc_vacuum_leak_right, 0, sizeof(unsigned long long)),
               "IMC::transport_step cudaMemset ddmc_vacuum_leak_right failed");
    cuda_check(cudaMemset(d_ddmc_converted_to_imc, 0, sizeof(unsigned long long)),
               "IMC::transport_step cudaMemset ddmc_converted_to_imc failed");
    cuda_check(cudaMemset(d_ddmc_converted_to_rw, 0, sizeof(unsigned long long)),
               "IMC::transport_step cudaMemset ddmc_converted_to_rw failed");
    cuda_check(cudaMemset(d_ddmc_sigma_tot_zero, 0, sizeof(unsigned long long)),
               "IMC::transport_step cudaMemset ddmc_sigma_tot_zero failed");
    cuda_check(cudaMemset(d_ddmc_max_events_reached, 0, sizeof(unsigned long long)),
               "IMC::transport_step cudaMemset ddmc_max_events_reached failed");

    DDMCTransportGPUInputs ddmc_in{};
    ddmc_in.pool = &pool_;
    ddmc_in.sigma_a_eff = d_sigma_a_eff;
    ddmc_in.sigma_s_eff = d_eta_cdf != nullptr ? d_sigma_s_eff : nullptr;
    ddmc_in.sigma_leak_left = d_sigma_leak_left;
    ddmc_in.sigma_leak_right = d_sigma_leak_right;
    ddmc_in.bc_left = d_bc_left;
    ddmc_in.bc_right = d_bc_right;
    ddmc_in.eta_cdf = d_eta_cdf;
    ddmc_in.ddmc_mode = d_ddmc_mode;
    ddmc_in.node_r = state.x_r.data();
    ddmc_in.rad_dep = state.rad_dep.data();
    ddmc_in.rad_E_tally = d_rad_E_tally;
    ddmc_in.E_escape = d_E_escape;
    ddmc_in.E_numerical_loss = d_E_numerical_loss;
    ddmc_in.ddmc_absorbed = d_ddmc_absorbed;
    ddmc_in.ddmc_census = d_ddmc_census;
    ddmc_in.ddmc_leak_left = d_ddmc_leak_left;
    ddmc_in.ddmc_leak_right = d_ddmc_leak_right;
    ddmc_in.ddmc_leak_boundary = d_ddmc_leak_boundary;
    ddmc_in.ddmc_vacuum_leak_left = d_ddmc_vacuum_leak_left;
    ddmc_in.ddmc_vacuum_leak_right = d_ddmc_vacuum_leak_right;
    ddmc_in.ddmc_converted_to_imc = d_ddmc_converted_to_imc;
    ddmc_in.ddmc_converted_to_rw = d_ddmc_converted_to_rw;
    ddmc_in.ddmc_sigma_tot_zero = d_ddmc_sigma_tot_zero;
    ddmc_in.ddmc_max_events_reached = d_ddmc_max_events_reached;
    ddmc_in.n_cells = n_cells;
    ddmc_in.n_groups = n_groups;
    ddmc_in.n_ddmc = sorted.n_ddmc;
    ddmc_in.ddmc_start = sorted.n_imc;
    ddmc_in.ghost_layers = rank_boundary_1d.ghost_layers;
    ddmc_in.nr_local = rank_boundary_1d.nr_local;
    ddmc_in.has_left_boundary = rank_boundary_1d.has_left_boundary;
    ddmc_in.has_right_boundary = rank_boundary_1d.has_right_boundary;
    ddmc_in.interface_exit_distribution =
        (cfg.radiation.ddmc.interface_exit_distribution == "half_isotropic")
            ? static_cast<std::uint8_t>(1U)
            : static_cast<std::uint8_t>(0U);
    ddmc_in.dt = dt;
    ddmc_in.step_number = static_cast<std::uint64_t>(state.step);
    ddmc_in.user_seed = cfg.main.seed;
    ddmc_in.error_flags = d_error_flags;
    ddmc_transport_gpu_cuda(ddmc_in);
    check_device_flags_after_stage(d_error_flags, "ddmc_transport");

    unsigned long long host_ddmc_absorbed = 0ULL;
    unsigned long long host_ddmc_census = 0ULL;
    unsigned long long host_ddmc_leak_left = 0ULL;
    unsigned long long host_ddmc_leak_right = 0ULL;
    unsigned long long host_ddmc_leak_boundary = 0ULL;
    unsigned long long host_ddmc_vacuum_leak_left = 0ULL;
    unsigned long long host_ddmc_vacuum_leak_right = 0ULL;
    unsigned long long host_ddmc_converted_to_imc = 0ULL;
    unsigned long long host_ddmc_converted_to_rw = 0ULL;
    unsigned long long host_ddmc_sigma_tot_zero = 0ULL;
    unsigned long long host_ddmc_max_events_reached = 0ULL;
    cuda_check(cudaMemcpy(&host_ddmc_absorbed,
                          d_ddmc_absorbed,
                          sizeof(host_ddmc_absorbed),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy ddmc_absorbed failed");
    cuda_check(cudaMemcpy(&host_ddmc_census,
                          d_ddmc_census,
                          sizeof(host_ddmc_census),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy ddmc_census failed");
    cuda_check(cudaMemcpy(&host_ddmc_leak_left,
                          d_ddmc_leak_left,
                          sizeof(host_ddmc_leak_left),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy ddmc_leak_left failed");
    cuda_check(cudaMemcpy(&host_ddmc_leak_right,
                          d_ddmc_leak_right,
                          sizeof(host_ddmc_leak_right),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy ddmc_leak_right failed");
    cuda_check(cudaMemcpy(&host_ddmc_leak_boundary,
                          d_ddmc_leak_boundary,
                          sizeof(host_ddmc_leak_boundary),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy ddmc_leak_boundary failed");
    cuda_check(cudaMemcpy(&host_ddmc_vacuum_leak_left,
                          d_ddmc_vacuum_leak_left,
                          sizeof(host_ddmc_vacuum_leak_left),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy ddmc_vacuum_leak_left failed");
    cuda_check(cudaMemcpy(&host_ddmc_vacuum_leak_right,
                          d_ddmc_vacuum_leak_right,
                          sizeof(host_ddmc_vacuum_leak_right),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy ddmc_vacuum_leak_right failed");
    cuda_check(cudaMemcpy(&host_ddmc_converted_to_imc,
                          d_ddmc_converted_to_imc,
                          sizeof(host_ddmc_converted_to_imc),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy ddmc_converted_to_imc failed");
    cuda_check(cudaMemcpy(&host_ddmc_converted_to_rw,
                          d_ddmc_converted_to_rw,
                          sizeof(host_ddmc_converted_to_rw),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy ddmc_converted_to_rw failed");
    cuda_check(cudaMemcpy(&host_ddmc_sigma_tot_zero,
                          d_ddmc_sigma_tot_zero,
                          sizeof(host_ddmc_sigma_tot_zero),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy ddmc_sigma_tot_zero failed");
    cuda_check(cudaMemcpy(&host_ddmc_max_events_reached,
                          d_ddmc_max_events_reached,
                          sizeof(host_ddmc_max_events_reached),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy ddmc_max_events_reached failed");

    ddmc_diag.processed = static_cast<std::int64_t>(sorted.n_ddmc);
    ddmc_diag.absorbed = static_cast<std::int64_t>(host_ddmc_absorbed);
    ddmc_diag.census = static_cast<std::int64_t>(host_ddmc_census);
    ddmc_diag.leak_left = static_cast<std::int64_t>(host_ddmc_leak_left);
    ddmc_diag.leak_right = static_cast<std::int64_t>(host_ddmc_leak_right);
    ddmc_diag.leak_boundary = static_cast<std::int64_t>(host_ddmc_leak_boundary);
    ddmc_diag.converted_to_imc =
        static_cast<std::int64_t>(host_ddmc_converted_to_imc);
    ddmc_diag.converted_to_rw =
        static_cast<std::int64_t>(host_ddmc_converted_to_rw);
    ddmc_diag.sigma_tot_zero = static_cast<std::int64_t>(host_ddmc_sigma_tot_zero);
    ddmc_diag.max_events_reached =
        static_cast<std::int64_t>(host_ddmc_max_events_reached);
    last_ddmc_to_imc_conversions_ = ddmc_diag.converted_to_imc;
    if (cfg.main.verbosity == "verbose" || cfg.main.verbosity == "debug") {
      core::log_info("[ddmc:gpu] vacuum_leak_left=" +
                     std::to_string(host_ddmc_vacuum_leak_left) +
                     " vacuum_leak_right=" +
                     std::to_string(host_ddmc_vacuum_leak_right));
    }

    cuda_check(cudaFree(d_ddmc_max_events_reached),
               "IMC::transport_step cudaFree ddmc_max_events_reached failed");
    cuda_check(cudaFree(d_ddmc_sigma_tot_zero),
               "IMC::transport_step cudaFree ddmc_sigma_tot_zero failed");
    cuda_check(cudaFree(d_ddmc_converted_to_imc),
               "IMC::transport_step cudaFree ddmc_converted_to_imc failed");
    cuda_check(cudaFree(d_ddmc_converted_to_rw),
               "IMC::transport_step cudaFree ddmc_converted_to_rw failed");
    cuda_check(cudaFree(d_ddmc_leak_right),
               "IMC::transport_step cudaFree ddmc_leak_right failed");
    cuda_check(cudaFree(d_ddmc_leak_left),
               "IMC::transport_step cudaFree ddmc_leak_left failed");
    cuda_check(cudaFree(d_ddmc_leak_boundary),
               "IMC::transport_step cudaFree ddmc_leak_boundary failed");
    cuda_check(cudaFree(d_ddmc_vacuum_leak_right),
               "IMC::transport_step cudaFree ddmc_vacuum_leak_right failed");
    cuda_check(cudaFree(d_ddmc_vacuum_leak_left),
               "IMC::transport_step cudaFree ddmc_vacuum_leak_left failed");
    cuda_check(cudaFree(d_ddmc_census),
               "IMC::transport_step cudaFree ddmc_census failed");
    cuda_check(cudaFree(d_ddmc_absorbed),
               "IMC::transport_step cudaFree ddmc_absorbed failed");
    cuda_check(cudaFree(d_bc_right), "IMC::transport_step cudaFree ddmc bc_right failed");
    cuda_check(cudaFree(d_bc_left), "IMC::transport_step cudaFree ddmc bc_left failed");
    cuda_check(cudaFree(d_sigma_leak_right),
               "IMC::transport_step cudaFree ddmc sigma_leak_right failed");
    cuda_check(cudaFree(d_sigma_leak_left),
               "IMC::transport_step cudaFree ddmc sigma_leak_left failed");
    } else {
      constexpr int kFaceCount = 4;
      const std::size_t n_face_cell_groups =
          static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(kFaceCount) *
          static_cast<std::size_t>(n_groups);
      const std::size_t n_face_cells =
          static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(kFaceCount);
      std::vector<double> host_sigma_leak_face(n_face_cell_groups, 0.0);
      std::vector<std::uint8_t> host_bc_face(n_face_cell_groups, 0U);
      std::vector<int> host_neighbor_face(n_face_cells, -1);
      for (int c = 0; c < n_cells; ++c) {
        for (int g = 0; g < n_groups; ++g) {
          const auto& cell_data = ddmc_preparation.coefficients.get_cell_data(c, g);
          const std::size_t face_group_base =
              static_cast<std::size_t>(c) * static_cast<std::size_t>(kFaceCount) *
                  static_cast<std::size_t>(n_groups) +
              static_cast<std::size_t>(g);
          for (int face = 0; face < kFaceCount; ++face) {
            const std::size_t face_group_index =
                face_group_base +
                static_cast<std::size_t>(face) * static_cast<std::size_t>(n_groups);
            host_sigma_leak_face[face_group_index] = cell_data.sigma_leak_face[face];
            host_bc_face[face_group_index] =
                static_cast<std::uint8_t>(cell_data.bc_face[face]);
            if (g == 0) {
              host_neighbor_face[static_cast<std::size_t>(c) *
                                     static_cast<std::size_t>(kFaceCount) +
                                 static_cast<std::size_t>(face)] =
                  cell_data.neighbor_face[face];
            }
          }
        }
      }

      double* d_sigma_leak_face = nullptr;
      std::uint8_t* d_bc_face = nullptr;
      int* d_neighbor_face = nullptr;
      unsigned long long* d_ddmc_absorbed = nullptr;
      unsigned long long* d_ddmc_census = nullptr;
      unsigned long long* d_ddmc_leak_face0 = nullptr;
      unsigned long long* d_ddmc_leak_face1 = nullptr;
      unsigned long long* d_ddmc_leak_face2 = nullptr;
      unsigned long long* d_ddmc_leak_face3 = nullptr;
      unsigned long long* d_ddmc_leak_boundary = nullptr;
      unsigned long long* d_ddmc_converted_to_imc = nullptr;
      unsigned long long* d_ddmc_sigma_tot_zero = nullptr;
      unsigned long long* d_ddmc_max_events_reached = nullptr;

      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_leak_face),
                            sizeof(double) * host_sigma_leak_face.size()),
                 "IMC::transport_step cudaMalloc ddmc sigma_leak_face failed");
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_bc_face),
                            sizeof(std::uint8_t) * host_bc_face.size()),
                 "IMC::transport_step cudaMalloc ddmc bc_face failed");
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_neighbor_face),
                            sizeof(int) * host_neighbor_face.size()),
                 "IMC::transport_step cudaMalloc ddmc neighbor_face failed");
      cuda_check(cudaMemcpy(d_sigma_leak_face,
                            host_sigma_leak_face.data(),
                            sizeof(double) * host_sigma_leak_face.size(),
                            cudaMemcpyHostToDevice),
                 "IMC::transport_step copy ddmc sigma_leak_face failed");
      cuda_check(cudaMemcpy(d_bc_face,
                            host_bc_face.data(),
                            sizeof(std::uint8_t) * host_bc_face.size(),
                            cudaMemcpyHostToDevice),
                 "IMC::transport_step copy ddmc bc_face failed");
      cuda_check(cudaMemcpy(d_neighbor_face,
                            host_neighbor_face.data(),
                            sizeof(int) * host_neighbor_face.size(),
                            cudaMemcpyHostToDevice),
                 "IMC::transport_step copy ddmc neighbor_face failed");

      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_absorbed),
                            sizeof(unsigned long long)),
                 "IMC::transport_step cudaMalloc ddmc_absorbed failed");
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_census),
                            sizeof(unsigned long long)),
                 "IMC::transport_step cudaMalloc ddmc_census failed");
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_leak_face0),
                            sizeof(unsigned long long)),
                 "IMC::transport_step cudaMalloc ddmc_leak_face0 failed");
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_leak_face1),
                            sizeof(unsigned long long)),
                 "IMC::transport_step cudaMalloc ddmc_leak_face1 failed");
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_leak_face2),
                            sizeof(unsigned long long)),
                 "IMC::transport_step cudaMalloc ddmc_leak_face2 failed");
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_leak_face3),
                            sizeof(unsigned long long)),
                 "IMC::transport_step cudaMalloc ddmc_leak_face3 failed");
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_leak_boundary),
                            sizeof(unsigned long long)),
                 "IMC::transport_step cudaMalloc ddmc_leak_boundary failed");
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_converted_to_imc),
                            sizeof(unsigned long long)),
                 "IMC::transport_step cudaMalloc ddmc_converted_to_imc failed");
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_sigma_tot_zero),
                            sizeof(unsigned long long)),
                 "IMC::transport_step cudaMalloc ddmc_sigma_tot_zero failed");
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_max_events_reached),
                            sizeof(unsigned long long)),
                 "IMC::transport_step cudaMalloc ddmc_max_events_reached failed");

      cuda_check(cudaMemset(d_ddmc_absorbed, 0, sizeof(unsigned long long)),
                 "IMC::transport_step cudaMemset ddmc_absorbed failed");
      cuda_check(cudaMemset(d_ddmc_census, 0, sizeof(unsigned long long)),
                 "IMC::transport_step cudaMemset ddmc_census failed");
      cuda_check(cudaMemset(d_ddmc_leak_face0, 0, sizeof(unsigned long long)),
                 "IMC::transport_step cudaMemset ddmc_leak_face0 failed");
      cuda_check(cudaMemset(d_ddmc_leak_face1, 0, sizeof(unsigned long long)),
                 "IMC::transport_step cudaMemset ddmc_leak_face1 failed");
      cuda_check(cudaMemset(d_ddmc_leak_face2, 0, sizeof(unsigned long long)),
                 "IMC::transport_step cudaMemset ddmc_leak_face2 failed");
      cuda_check(cudaMemset(d_ddmc_leak_face3, 0, sizeof(unsigned long long)),
                 "IMC::transport_step cudaMemset ddmc_leak_face3 failed");
      cuda_check(cudaMemset(d_ddmc_leak_boundary, 0, sizeof(unsigned long long)),
                 "IMC::transport_step cudaMemset ddmc_leak_boundary failed");
      cuda_check(cudaMemset(d_ddmc_converted_to_imc, 0, sizeof(unsigned long long)),
                 "IMC::transport_step cudaMemset ddmc_converted_to_imc failed");
      cuda_check(cudaMemset(d_ddmc_sigma_tot_zero, 0, sizeof(unsigned long long)),
                 "IMC::transport_step cudaMemset ddmc_sigma_tot_zero failed");
      cuda_check(cudaMemset(d_ddmc_max_events_reached, 0, sizeof(unsigned long long)),
                 "IMC::transport_step cudaMemset ddmc_max_events_reached failed");

      DDMCTransport2DGPUInputs ddmc_in{};
      ddmc_in.pool = &pool_;
      ddmc_in.sigma_a_eff = d_sigma_a_eff;
      ddmc_in.sigma_s_eff = d_eta_cdf != nullptr ? d_sigma_s_eff : nullptr;
      ddmc_in.sigma_leak_face = d_sigma_leak_face;
      ddmc_in.bc_face = d_bc_face;
      ddmc_in.neighbor_face = d_neighbor_face;
      ddmc_in.eta_cdf = d_eta_cdf;
      ddmc_in.ddmc_mode = d_ddmc_mode;
      ddmc_in.node_r = state.x_r.data();
      ddmc_in.node_z = state.x_z.data();
      ddmc_in.rad_dep = state.rad_dep.data();
      ddmc_in.rad_E_tally = d_rad_E_tally;
      ddmc_in.E_escape = d_E_escape;
      ddmc_in.E_numerical_loss = d_E_numerical_loss;
      ddmc_in.ddmc_absorbed = d_ddmc_absorbed;
      ddmc_in.ddmc_census = d_ddmc_census;
      ddmc_in.ddmc_leak_face0 = d_ddmc_leak_face0;
      ddmc_in.ddmc_leak_face1 = d_ddmc_leak_face1;
      ddmc_in.ddmc_leak_face2 = d_ddmc_leak_face2;
      ddmc_in.ddmc_leak_face3 = d_ddmc_leak_face3;
      ddmc_in.ddmc_leak_boundary = d_ddmc_leak_boundary;
      ddmc_in.ddmc_converted_to_imc = d_ddmc_converted_to_imc;
      ddmc_in.ddmc_sigma_tot_zero = d_ddmc_sigma_tot_zero;
      ddmc_in.ddmc_max_events_reached = d_ddmc_max_events_reached;
      ddmc_in.n_cells = n_cells;
      ddmc_in.n_groups = n_groups;
      ddmc_in.nr = state.mesh.topo.nr;
      ddmc_in.nz = state.mesh.topo.nz;
      ddmc_in.n_ddmc = sorted.n_ddmc;
      ddmc_in.ddmc_start = sorted.n_imc;
      ddmc_in.ghost_layers = rank_boundary_2d.ghost_layers;
      ddmc_in.nr_local = rank_boundary_2d.nr_local;
      ddmc_in.nz_local = rank_boundary_2d.nz_local;
      ddmc_in.has_r_inner_boundary = rank_boundary_2d.has_r_inner_boundary;
      ddmc_in.has_r_outer_boundary = rank_boundary_2d.has_r_outer_boundary;
      ddmc_in.has_z_bottom_boundary = rank_boundary_2d.has_z_bottom_boundary;
      ddmc_in.has_z_top_boundary = rank_boundary_2d.has_z_top_boundary;
      ddmc_in.interface_exit_distribution =
          (cfg.radiation.ddmc.interface_exit_distribution == "half_isotropic")
              ? static_cast<std::uint8_t>(1U)
              : static_cast<std::uint8_t>(0U);
      ddmc_in.dt = dt;
      ddmc_in.step_number = static_cast<std::uint64_t>(state.step);
      ddmc_in.user_seed = cfg.main.seed;
      ddmc_in.error_flags = d_error_flags;
      ddmc_transport_2d_gpu_cuda(ddmc_in);
      check_device_flags_after_stage(d_error_flags, "ddmc_transport");

      unsigned long long host_ddmc_absorbed = 0ULL;
      unsigned long long host_ddmc_census = 0ULL;
      unsigned long long host_ddmc_leak_face0 = 0ULL;
      unsigned long long host_ddmc_leak_face1 = 0ULL;
      unsigned long long host_ddmc_leak_face2 = 0ULL;
      unsigned long long host_ddmc_leak_face3 = 0ULL;
      unsigned long long host_ddmc_leak_boundary = 0ULL;
      unsigned long long host_ddmc_converted_to_imc = 0ULL;
      unsigned long long host_ddmc_sigma_tot_zero = 0ULL;
      unsigned long long host_ddmc_max_events_reached = 0ULL;
      cuda_check(cudaMemcpy(&host_ddmc_absorbed,
                            d_ddmc_absorbed,
                            sizeof(host_ddmc_absorbed),
                            cudaMemcpyDeviceToHost),
                 "IMC::transport_step copy ddmc_absorbed failed");
      cuda_check(cudaMemcpy(&host_ddmc_census,
                            d_ddmc_census,
                            sizeof(host_ddmc_census),
                            cudaMemcpyDeviceToHost),
                 "IMC::transport_step copy ddmc_census failed");
      cuda_check(cudaMemcpy(&host_ddmc_leak_face0,
                            d_ddmc_leak_face0,
                            sizeof(host_ddmc_leak_face0),
                            cudaMemcpyDeviceToHost),
                 "IMC::transport_step copy ddmc_leak_face0 failed");
      cuda_check(cudaMemcpy(&host_ddmc_leak_face1,
                            d_ddmc_leak_face1,
                            sizeof(host_ddmc_leak_face1),
                            cudaMemcpyDeviceToHost),
                 "IMC::transport_step copy ddmc_leak_face1 failed");
      cuda_check(cudaMemcpy(&host_ddmc_leak_face2,
                            d_ddmc_leak_face2,
                            sizeof(host_ddmc_leak_face2),
                            cudaMemcpyDeviceToHost),
                 "IMC::transport_step copy ddmc_leak_face2 failed");
      cuda_check(cudaMemcpy(&host_ddmc_leak_face3,
                            d_ddmc_leak_face3,
                            sizeof(host_ddmc_leak_face3),
                            cudaMemcpyDeviceToHost),
                 "IMC::transport_step copy ddmc_leak_face3 failed");
      cuda_check(cudaMemcpy(&host_ddmc_leak_boundary,
                            d_ddmc_leak_boundary,
                            sizeof(host_ddmc_leak_boundary),
                            cudaMemcpyDeviceToHost),
                 "IMC::transport_step copy ddmc_leak_boundary failed");
      cuda_check(cudaMemcpy(&host_ddmc_converted_to_imc,
                            d_ddmc_converted_to_imc,
                            sizeof(host_ddmc_converted_to_imc),
                            cudaMemcpyDeviceToHost),
                 "IMC::transport_step copy ddmc_converted_to_imc failed");
      cuda_check(cudaMemcpy(&host_ddmc_sigma_tot_zero,
                            d_ddmc_sigma_tot_zero,
                            sizeof(host_ddmc_sigma_tot_zero),
                            cudaMemcpyDeviceToHost),
                 "IMC::transport_step copy ddmc_sigma_tot_zero failed");
      cuda_check(cudaMemcpy(&host_ddmc_max_events_reached,
                            d_ddmc_max_events_reached,
                            sizeof(host_ddmc_max_events_reached),
                            cudaMemcpyDeviceToHost),
                 "IMC::transport_step copy ddmc_max_events_reached failed");

      ddmc_diag.processed = static_cast<std::int64_t>(sorted.n_ddmc);
      ddmc_diag.absorbed = static_cast<std::int64_t>(host_ddmc_absorbed);
      ddmc_diag.census = static_cast<std::int64_t>(host_ddmc_census);
      ddmc_diag.leak_left = static_cast<std::int64_t>(host_ddmc_leak_face0);
      ddmc_diag.leak_right = static_cast<std::int64_t>(host_ddmc_leak_face1);
      ddmc_diag.leak_face2 = static_cast<std::int64_t>(host_ddmc_leak_face2);
      ddmc_diag.leak_face3 = static_cast<std::int64_t>(host_ddmc_leak_face3);
      ddmc_diag.leak_boundary = static_cast<std::int64_t>(host_ddmc_leak_boundary);
      ddmc_diag.converted_to_imc =
          static_cast<std::int64_t>(host_ddmc_converted_to_imc);
      ddmc_diag.sigma_tot_zero = static_cast<std::int64_t>(host_ddmc_sigma_tot_zero);
      ddmc_diag.max_events_reached =
          static_cast<std::int64_t>(host_ddmc_max_events_reached);
      last_ddmc_to_imc_conversions_ = ddmc_diag.converted_to_imc;

      cuda_check(cudaFree(d_ddmc_max_events_reached),
                 "IMC::transport_step cudaFree ddmc_max_events_reached failed");
      cuda_check(cudaFree(d_ddmc_sigma_tot_zero),
                 "IMC::transport_step cudaFree ddmc_sigma_tot_zero failed");
      cuda_check(cudaFree(d_ddmc_converted_to_imc),
                 "IMC::transport_step cudaFree ddmc_converted_to_imc failed");
      cuda_check(cudaFree(d_ddmc_leak_boundary),
                 "IMC::transport_step cudaFree ddmc_leak_boundary failed");
      cuda_check(cudaFree(d_ddmc_leak_face3),
                 "IMC::transport_step cudaFree ddmc_leak_face3 failed");
      cuda_check(cudaFree(d_ddmc_leak_face2),
                 "IMC::transport_step cudaFree ddmc_leak_face2 failed");
      cuda_check(cudaFree(d_ddmc_leak_face1),
                 "IMC::transport_step cudaFree ddmc_leak_face1 failed");
      cuda_check(cudaFree(d_ddmc_leak_face0),
                 "IMC::transport_step cudaFree ddmc_leak_face0 failed");
      cuda_check(cudaFree(d_ddmc_census), "IMC::transport_step cudaFree ddmc_census failed");
      cuda_check(cudaFree(d_ddmc_absorbed),
                 "IMC::transport_step cudaFree ddmc_absorbed failed");
      cuda_check(cudaFree(d_neighbor_face),
                 "IMC::transport_step cudaFree ddmc neighbor_face failed");
      cuda_check(cudaFree(d_bc_face), "IMC::transport_step cudaFree ddmc bc_face failed");
      cuda_check(cudaFree(d_sigma_leak_face),
                 "IMC::transport_step cudaFree ddmc sigma_leak_face failed");
    }
  }
  const auto t_ddmc_end = verbose_imc_timing ? Clock::now() : Clock::time_point{};
  auto t_rw_end = t_ddmc_end;
  auto t_imc_tail_end = t_rw_end;

  if (state.mesh.dim == 1 && (sorted.n_rw > 0 || ddmc_diag.converted_to_rw > 0)) {
    if (use_implicit_ddmc_diffusion || ddmc_diag.converted_to_rw > 0) {
      TENRYU_ASSERT(pool_.n_alive >= 0 && pool_.n_alive <= pool_.capacity,
                    "IMC::transport_step pool invariant violated before RW sort");
      sorted = composite_sort_and_partition(pool_,
                                            pool_.n_alive,
                                            n_cells,
                                            n_groups,
                                            d_E_numerical_loss);
      pool_.n_alive = sorted.n_alive;
    }
    rw_processed = static_cast<std::int64_t>(sorted.n_rw);

    unsigned long long* d_rw_census = nullptr;
    unsigned long long* d_rw_leak_left = nullptr;
    unsigned long long* d_rw_leak_right = nullptr;
    unsigned long long* d_rw_escaped = nullptr;
    unsigned long long* d_rw_converted_to_imc = nullptr;
    unsigned long long* d_rw_converted_to_ddmc = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rw_census), sizeof(unsigned long long)),
               "IMC::transport_step cudaMalloc rw_census failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rw_leak_left), sizeof(unsigned long long)),
               "IMC::transport_step cudaMalloc rw_leak_left failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rw_leak_right), sizeof(unsigned long long)),
               "IMC::transport_step cudaMalloc rw_leak_right failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rw_escaped), sizeof(unsigned long long)),
               "IMC::transport_step cudaMalloc rw_escaped failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rw_converted_to_imc),
                          sizeof(unsigned long long)),
               "IMC::transport_step cudaMalloc rw_converted_to_imc failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rw_converted_to_ddmc),
                          sizeof(unsigned long long)),
               "IMC::transport_step cudaMalloc rw_converted_to_ddmc failed");
    cuda_check(cudaMemset(d_rw_census, 0, sizeof(unsigned long long)),
               "IMC::transport_step cudaMemset rw_census failed");
    cuda_check(cudaMemset(d_rw_leak_left, 0, sizeof(unsigned long long)),
               "IMC::transport_step cudaMemset rw_leak_left failed");
    cuda_check(cudaMemset(d_rw_leak_right, 0, sizeof(unsigned long long)),
               "IMC::transport_step cudaMemset rw_leak_right failed");
    cuda_check(cudaMemset(d_rw_escaped, 0, sizeof(unsigned long long)),
               "IMC::transport_step cudaMemset rw_escaped failed");
    cuda_check(cudaMemset(d_rw_converted_to_imc, 0, sizeof(unsigned long long)),
               "IMC::transport_step cudaMemset rw_converted_to_imc failed");
    cuda_check(cudaMemset(d_rw_converted_to_ddmc, 0, sizeof(unsigned long long)),
               "IMC::transport_step cudaMemset rw_converted_to_ddmc failed");

    RWTransportGPUInputs rw_in{};
    rw_in.pool = &pool_;
    rw_in.sigma_R = d_sigma_R;
    rw_in.mode_map = d_ddmc_mode;
    rw_in.node_r = state.x_r.data();
    rw_in.rad_E_tally = d_rad_E_tally;
    rw_in.E_escape = d_E_escape;
    rw_in.E_numerical_loss = d_E_numerical_loss;
    rw_in.rw_census = d_rw_census;
    rw_in.rw_leak_left = d_rw_leak_left;
    rw_in.rw_leak_right = d_rw_leak_right;
    rw_in.rw_escaped = d_rw_escaped;
    rw_in.rw_converted_to_imc = d_rw_converted_to_imc;
    rw_in.rw_converted_to_ddmc = d_rw_converted_to_ddmc;
    rw_in.n_cells = n_cells;
    rw_in.n_groups = n_groups;
    rw_in.n_rw = sorted.n_rw;
    rw_in.rw_start = sorted.n_imc + sorted.n_ddmc;
    rw_in.bc_inner = boundary_code_from_string(cfg.radiation.boundary.inner_r);
    rw_in.bc_outer = boundary_code_from_string(cfg.radiation.boundary.outer_r);
    rw_in.dt = dt;
    rw_in.step_number = static_cast<std::uint64_t>(state.step);
    rw_in.user_seed = cfg.main.seed;
    rw_in.error_flags = d_error_flags;
    rw_transport_1d_gpu_cuda(rw_in);
    check_device_flags_after_stage(d_error_flags, "rw_transport");

    unsigned long long host_rw_census = 0ULL;
    unsigned long long host_rw_leak_left = 0ULL;
    unsigned long long host_rw_leak_right = 0ULL;
    unsigned long long host_rw_escaped = 0ULL;
    unsigned long long host_rw_converted_to_imc = 0ULL;
    unsigned long long host_rw_converted_to_ddmc = 0ULL;
    cuda_check(cudaMemcpy(&host_rw_census,
                          d_rw_census,
                          sizeof(host_rw_census),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy rw_census failed");
    cuda_check(cudaMemcpy(&host_rw_leak_left,
                          d_rw_leak_left,
                          sizeof(host_rw_leak_left),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy rw_leak_left failed");
    cuda_check(cudaMemcpy(&host_rw_leak_right,
                          d_rw_leak_right,
                          sizeof(host_rw_leak_right),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy rw_leak_right failed");
    cuda_check(cudaMemcpy(&host_rw_escaped,
                          d_rw_escaped,
                          sizeof(host_rw_escaped),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy rw_escaped failed");
    cuda_check(cudaMemcpy(&host_rw_converted_to_imc,
                          d_rw_converted_to_imc,
                          sizeof(host_rw_converted_to_imc),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy rw_converted_to_imc failed");
    cuda_check(cudaMemcpy(&host_rw_converted_to_ddmc,
                          d_rw_converted_to_ddmc,
                          sizeof(host_rw_converted_to_ddmc),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy rw_converted_to_ddmc failed");
    rw_census = static_cast<std::int64_t>(host_rw_census);
    rw_leak_left = static_cast<std::int64_t>(host_rw_leak_left);
    rw_leak_right = static_cast<std::int64_t>(host_rw_leak_right);
    rw_escaped = static_cast<std::int64_t>(host_rw_escaped);
    rw_converted_to_imc = static_cast<std::int64_t>(host_rw_converted_to_imc);
    rw_converted_to_ddmc = static_cast<std::int64_t>(host_rw_converted_to_ddmc);

    cuda_check(cudaFree(d_rw_converted_to_ddmc),
               "IMC::transport_step cudaFree rw_converted_to_ddmc failed");
    cuda_check(cudaFree(d_rw_converted_to_imc),
               "IMC::transport_step cudaFree rw_converted_to_imc failed");
    cuda_check(cudaFree(d_rw_escaped),
               "IMC::transport_step cudaFree rw_escaped failed");
    cuda_check(cudaFree(d_rw_leak_right),
               "IMC::transport_step cudaFree rw_leak_right failed");
    cuda_check(cudaFree(d_rw_leak_left),
               "IMC::transport_step cudaFree rw_leak_left failed");
    cuda_check(cudaFree(d_rw_census),
               "IMC::transport_step cudaFree rw_census failed");

    t_rw_end = verbose_imc_timing ? Clock::now() : Clock::time_point{};
  }

  if (has_diffusion_cells && have_diffusion_device_masks) {
    const double diffusion_E_before_rkl2 =
        compute_diffusion_energy_host(diff_E_, diff_cell_, diff_vol_);
    DiffusionStepInputs diff_in{};
    diff_in.diff_E = diff_E_.as<double>();
    diff_in.sigma_R = d_sigma_R;
    diff_in.vol = state.vol.data();
    diff_in.node_r = state.x_r.data();
    diff_in.diff_cell = d_diff_cell.as<std::uint8_t>();
    diff_in.face_current_in =
        diffusion_face_current_active ? face_current_in_.as<double>() : nullptr;
    diff_in.face_current_dt = dt;
    diff_in.n_cells = n_cells;
    diff_in.n_groups = n_groups;
    diff_in.dt = dt;
    diff_in.bc_inner = diffusion_boundary_code_from_string(cfg.radiation.boundary.inner_r);
    diff_in.bc_outer = diffusion_boundary_code_from_string(cfg.radiation.boundary.outer_r);
    if (diffusion_face_current_active) {
      diffusion_interface_E_in =
          diffusion_face_current_in_energy(face_current_in_.as<double>(),
                                           d_diff_cell.as<std::uint8_t>(),
                                           n_cells,
                                           n_groups);
    }
    deterministic_diffusion_result =
        deterministic_diffusion_step_1d(diff_in,
                                        cfg.radiation.diffusion.sts_max_stages,
                                        cfg.radiation.diffusion.sts_damping,
                                        cfg.radiation.diffusion.sts_subcycle_eta);
    escaped_energy_total_ += std::max(deterministic_diffusion_result.E_leaked, 0.0);
    const double diffusion_E_after_rkl2 =
        compute_diffusion_energy_host(diff_E_, diff_cell_, diff_vol_);
    log_diffusion_substep_energy(state.step,
                                 "rkl2",
                                 diffusion_E_before_rkl2,
                                 diffusion_E_after_rkl2,
                                 diffusion_interface_E_in,
                                 0.0,
                                 deterministic_diffusion_result.E_leaked,
                                 0.0,
                                 0);
    const double growth_reference =
        diffusion_growth_reference(diffusion_E_before_rkl2,
                                   diffusion_interface_E_in,
                                   0.0,
                                   deterministic_diffusion_result.E_leaked,
                                   0.0);
    if (diffusion_energy_growth_exceeded(growth_reference,
                                         diffusion_E_after_rkl2)) {
      force_diffusion_cells_to_imc("rkl2_growth",
                                   growth_reference,
                                   diffusion_E_after_rkl2);
    }

    if (diffusion_face_current_active && has_diffusion_cells) {
      int n_diffusion_interface_faces = 0;
      if (diff_cell_.size() == n_cells_us) {
        for (int face = 1; face < n_cells; ++face) {
          const bool left_diff =
              diff_cell_[static_cast<std::size_t>(face - 1)] != 0U;
          const bool right_diff =
              diff_cell_[static_cast<std::size_t>(face)] != 0U;
          if (left_diff != right_diff) {
            ++n_diffusion_interface_faces;
          }
        }
      }
      const std::int64_t max_interface_particles =
          static_cast<std::int64_t>(n_diffusion_interface_faces) *
          static_cast<std::int64_t>(n_groups) *
          static_cast<std::int64_t>(
              std::max(1, cfg.radiation.diffusion.interface_particles_per_face_group));
      const std::uint64_t interface_gid_base =
          compute_diffusion_interface_gid_base(step_base_gid,
                                               max_interface_particles,
                                               part.rank,
                                               part.n_ranks);
      diffusion_interface_result =
          spawn_imc_from_diffusion_faces(
              pool_,
              diff_E_.as<double>(),
              d_sigma_R,
              state.vol.data(),
              state.x_r.data(),
              d_diff_cell.as<std::uint8_t>(),
              face_current_out_.as<double>(),
              n_cells,
              n_groups,
              dt,
              std::max(1, cfg.radiation.diffusion.interface_particles_per_face_group),
              interface_gid_base,
              cfg.main.seed,
              static_cast<std::uint64_t>(state.step),
              max_pool_size);
      const double diffusion_E_after_interface_out =
          compute_diffusion_energy_host(diff_E_, diff_cell_, diff_vol_);
      log_diffusion_substep_energy(state.step,
                                   "interface_out",
                                   diffusion_E_after_rkl2,
                                   diffusion_E_after_interface_out,
                                   0.0,
                                   diffusion_interface_result.E_spawned,
                                   diffusion_interface_result.E_leaked_vacuum,
                                   0.0,
                                   0);
      if (face_current_bytes > 0U) {
        cuda_check(cudaMemset(face_current_in_.ptr, 0, face_current_bytes),
                   "IMC::transport_step zero consumed diffusion face current in failed");
      }
    }
  }

  if (ddmc_diag.converted_to_imc > 0 || rw_converted_to_imc > 0 ||
      diffusion_interface_result.n_spawned > 0 ||
      diffusion_emergency_revert_particles > 0ULL) {
    TENRYU_ASSERT(pool_.n_alive >= 0 && pool_.n_alive <= pool_.capacity,
                  "IMC::transport_step pool invariant violated before tail IMC sort");
    sorted = composite_sort_and_partition(pool_,
                                          pool_.n_alive,
                                          n_cells,
                                          n_groups,
                                          d_E_numerical_loss);
    pool_.n_alive = sorted.n_alive;

    TransportInputs tail_in;
    tail_in.pool = &pool_;
    tail_in.sigma_a_eff = d_sigma_a_eff;
    tail_in.sigma_s_eff = d_sigma_s_eff;
    tail_in.Te = state.Te.data();
    tail_in.vol = state.vol.data();
    tail_in.sloc_abs_wr =
        enable_source_localization_transport ? sloc_abs_wr_.data() : nullptr;
    tail_in.sloc_abs_wr2 =
        enable_source_localization_transport ? sloc_abs_wr2_.data() : nullptr;
    tail_in.sloc_abs_E =
        enable_source_localization_transport ? sloc_abs_E_.data() : nullptr;
    tail_in.node_r = state.x_r.data();
    tail_in.node_z = (state.mesh.dim == 2) ? state.x_z.data() : nullptr;
    tail_in.nr = state.mesh.topo.nr;
    tail_in.nz = state.mesh.topo.nz;
    tail_in.mesh_dim = state.mesh.dim;
    tail_in.ddmc_mode = nullptr;
    tail_in.n_groups_for_mode = n_groups;
    tail_in.emissivity_preserving = cfg.radiation.ddmc.emissivity_preserving;
    tail_in.sigma_R = d_sigma_R;
    tail_in.cell_dx = d_cell_dx;
    tail_in.fleck_f = d_f;
    tail_in.sigma_a = d_sigma_a;
    tail_in.eta_cdf = d_eta_cdf;
    if (pgrw_requested) {
      tail_in.g_diff_end = d_g_diff_end;
      tail_in.sigma_a_bar = d_sigma_a_bar;
      tail_in.sigma_t_bar = d_sigma_t_bar;
      tail_in.D_pgrw = d_D_pgrw;
      tail_in.gamma_pgrw = d_gamma_pgrw;
      tail_in.pgrw_leak_inv_cdf = pgrw_tables_.d_leak_inv_cdf;
      tail_in.pgrw_leak_cdf_xi = pgrw_tables_.d_leak_cdf_xi;
      tail_in.pgrw_pos_cdf = pgrw_tables_.d_pos_cdf;
      tail_in.pgrw_leak_table_size = pgrw_tables_.leak_table_size;
      tail_in.pgrw_pos_theta_bins = pgrw_tables_.pos_theta_bins;
      tail_in.pgrw_pos_rho_bins = pgrw_tables_.pos_rho_bins;
      tail_in.pgrw_theta_max = pgrw_tables_.theta_max;
      tail_in.pgrw_tau_rw = cfg.radiation.ddmc.tau_rw;
    }
    tail_in.planck = planck.device_view();
    tail_in.inelastic_scatter = cfg.radiation.imc.inelastic_scatter;
    assign_scatter_bias_cdf_if_supported(tail_in, d_emission_bias_cdf);
    tail_in.rad_dep = state.rad_dep.data();
    tail_in.rad_E_tally = d_rad_E_tally;
    tail_in.holo_Prr_tally = d_holo_Prr_tally;
    tail_in.holo_Prr_coverage_tally = d_holo_Prr_coverage_tally;
    tail_in.face_current_step =
        face_current_tracking_active ? face_current_step_.as<double>() : nullptr;
    tail_in.diff_cell =
        diffusion_face_current_active ? d_diff_cell.as<std::uint8_t>() : nullptr;
    tail_in.diff_face_current_in =
        diffusion_face_current_active ? face_current_in_.as<double>() : nullptr;
    tail_in.E_escape = d_E_escape;
    tail_in.E_numerical_loss = d_E_numerical_loss;
    tail_in.imc_absorbed = d_imc_absorbed;
    tail_in.imc_escaped = d_imc_escaped;
    tail_in.diffusion_interface_kills =
        diffusion_face_current_active ? d_diffusion_interface_kills : nullptr;
    tail_in.cnt_boundary = d_cnt_boundary;
    tail_in.cnt_scatter = d_cnt_scatter;
    tail_in.cnt_census = d_cnt_census;
    tail_in.cnt_absorb_kill = d_cnt_absorb_kill;
    tail_in.cnt_absorb_survive = d_cnt_absorb_survive;
    tail_in.cnt_roulette_kill = d_cnt_roulette_kill;
    tail_in.n_cells = n_cells;
    tail_in.n_groups = n_groups;
    tail_in.n_imc = sorted.n_imc;
    tail_in.dt = dt;
    tail_in.E_avg = E_avg;
    tail_in.w_cutoff = cfg.radiation.imc.weight_cutoff;
    tail_in.p_survival = std::clamp(cfg.radiation.imc.roulette_survival, 0.0, 1.0);
    tail_in.f_cutoff = cfg.radiation.imc.cutoff_fraction;
    tail_in.tail_pass = true;
    tail_in.bc_inner = boundary_code_from_string(cfg.radiation.boundary.inner_r);
    tail_in.bc_outer = boundary_code_from_string(cfg.radiation.boundary.outer_r);
    tail_in.bc_bottom_z = boundary_code_from_string(cfg.radiation.boundary.bottom_z);
    tail_in.bc_top_z = boundary_code_from_string(cfg.radiation.boundary.top_z);
    tail_in.step_number = static_cast<std::uint64_t>(state.step);
    tail_in.user_seed = cfg.main.seed;
    tail_in.error_flags = d_error_flags;

    if (state.mesh.dim == 2) {
      imc_transport_2d_persistent_cuda(tail_in, part);
    } else {
      imc_transport_persistent_cuda(tail_in, part);
    }
    check_device_flags_after_stage(d_error_flags, "imc_transport_tail");

    cuda_check(cudaMemcpy(&host_imc_absorbed,
                          d_imc_absorbed,
                          sizeof(host_imc_absorbed),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy imc_absorbed(tail) failed");
    cuda_check(cudaMemcpy(&host_imc_escaped,
                          d_imc_escaped,
                          sizeof(host_imc_escaped),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy imc_escaped(tail) failed");
    cuda_check(cudaMemcpy(&host_diffusion_interface_kills,
                          d_diffusion_interface_kills,
                          sizeof(host_diffusion_interface_kills),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy diffusion_interface_kills(tail) failed");
    cuda_check(cudaMemcpy(&host_cnt_boundary,
                          d_cnt_boundary,
                          sizeof(host_cnt_boundary),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy cnt_boundary(tail) failed");
    cuda_check(cudaMemcpy(&host_cnt_scatter,
                          d_cnt_scatter,
                          sizeof(host_cnt_scatter),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy cnt_scatter(tail) failed");
    cuda_check(cudaMemcpy(&host_cnt_census,
                          d_cnt_census,
                          sizeof(host_cnt_census),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy cnt_census(tail) failed");
    cuda_check(cudaMemcpy(&host_cnt_absorb_kill,
                          d_cnt_absorb_kill,
                          sizeof(host_cnt_absorb_kill),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy cnt_absorb_kill(tail) failed");
    cuda_check(cudaMemcpy(&host_cnt_absorb_survive,
                          d_cnt_absorb_survive,
                          sizeof(host_cnt_absorb_survive),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy cnt_absorb_survive(tail) failed");
    cuda_check(cudaMemcpy(&host_cnt_roulette_kill,
                          d_cnt_roulette_kill,
                          sizeof(host_cnt_roulette_kill),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy cnt_roulette_kill(tail) failed");
    t_imc_tail_end = verbose_imc_timing ? Clock::now() : Clock::time_point{};
  }

  if (has_diffusion_cells && have_diffusion_device_masks) {
    if (diffusion_face_current_active && face_current_bytes > 0U) {
      const double diffusion_E_before_tail_deposit =
          compute_diffusion_energy_host(diff_E_, diff_cell_, diff_vol_);
      diffusion_interface_E_in_tail =
          deposit_diffusion_face_current_in(diff_E_.as<double>(),
                                            face_current_in_.as<double>(),
                                            state.vol.data(),
                                            d_diff_cell.as<std::uint8_t>(),
                                            n_cells,
                                            n_groups);
      cuda_check(cudaMemset(face_current_in_.ptr, 0, face_current_bytes),
                 "IMC::transport_step zero diffusion tail face current in failed");
      const double diffusion_E_after_tail_deposit =
          compute_diffusion_energy_host(diff_E_, diff_cell_, diff_vol_);
      log_diffusion_substep_energy(state.step,
                                   "tail_deposit",
                                   diffusion_E_before_tail_deposit,
                                   diffusion_E_after_tail_deposit,
                                   diffusion_interface_E_in_tail,
                                   0.0,
                                   0.0,
                                   0.0,
                                   0);
    }
    diffusion_E_before_source2 =
        compute_diffusion_energy_host(diff_E_, diff_cell_, diff_vol_);
    const DiffusionSourceSolveResult source2_result =
        run_diffusion_source_solve(0.5 * dt);
    diffusion_E_after_source2 =
        compute_diffusion_energy_host(diff_E_, diff_cell_, diff_vol_);
    log_diffusion_substep_energy(state.step,
                                 "source2",
                                 diffusion_E_before_source2,
                                 diffusion_E_after_source2,
                                 0.0,
                                 0.0,
                                 0.0,
                                 source2_result.matter_delta,
                                 source2_result.n_failures);
    accumulate_diffusion_source_result(&diffusion_source_result, source2_result);
    const double growth_reference =
        diffusion_growth_reference(diffusion_E_before_source2,
                                   0.0,
                                   0.0,
                                   0.0,
                                   source2_result.matter_delta);
    if (diffusion_energy_growth_exceeded(growth_reference,
                                         diffusion_E_after_source2)) {
      force_diffusion_cells_to_imc("source2_growth",
                                   growth_reference,
                                   diffusion_E_after_source2);
    }
  }

  parallel::DeviceArray d_holo_E_old;
  bool holo_predictor_succeeded = false;
  const auto log_holo_lo_result = [&](const char* phase,
                                      const HoloLOResult& result) {
    if (cfg.main.verbosity != "quiet" &&
        (result.matter_delta != 0.0 ||
         result.rad_delta != 0.0 ||
         result.boundary_E_in != 0.0 ||
         result.boundary_E_out != 0.0 ||
         result.boundary_limited_E != 0.0 ||
         result.failures != 0)) {
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(6);
      oss << "[holo_lo] step=" << state.step
          << " phase=" << ((phase != nullptr) ? phase : "solve")
          << " matter_delta=" << result.matter_delta
          << " rad_delta=" << result.rad_delta
          << " boundary_in=" << result.boundary_E_in
          << " boundary_out=" << result.boundary_E_out
          << " boundary_limited=" << result.boundary_limited_E
          << " balance_error=" << result.conservation_error
          << " source_iter=" << result.solver_iterations
          << " failures=" << result.failures;
      if (result.failures != 0) {
        oss << " failure_stage="
            << holo_lo_failure_stage_name(result.failure_stage)
            << " failure_cell=" << result.failure_cell
            << " failure_group=" << result.failure_group
            << " failure_E_sum=" << result.failure_E_sum
            << " failure_T=" << result.failure_T
            << " failure_residual=" << result.failure_residual;
      }
      core::log_info(oss.str());
    }
  };
  if (holo_enabled_1d) {
    last_holo_lo_result_ = HoloLOResult{};
    const std::size_t E_old_bytes = sizeof(double) * n_cell_groups_us;
    TENRYU_ASSERT(state.rad_E.size() == n_cell_groups_us,
                  "IMC::transport_step HOLO predictor requires rad_E size match");
    d_holo_E_old.resize(E_old_bytes);
    if (E_old_bytes > 0U) {
      cuda_check(cudaMemcpy(d_holo_E_old.as<double>(),
                            state.rad_E.data(),
                            E_old_bytes,
                            cudaMemcpyDeviceToDevice),
                 "IMC::transport_step copy HOLO E_old failed");
    }
    const bool holo_face_current_physical =
        face_current_tracking_active && !difference_source_supported;
    const HoloLOResult predictor_result =
        solve_holo_lo_source_ownership(state,
                                       cfg,
                                       planck,
                                       mat,
                                       d_sigma_a,
                                       d_sigma_R,
                                       d_f,
                                       n_cells,
                                       n_groups,
                                       dt,
                                       d_holo_E_old.as<double>(),
                                       E_old_bytes,
                                       nullptr,
                                       0U,
                                       holo_face_current_physical
                                           ? face_current_step_.as<double>()
                                           : nullptr,
                                       holo_face_current_physical ? face_current_bytes : 0U,
                                       nullptr,
                                       0U,
                                       false,
                                       false,
                                       difference_source_supported,
                                       cfg.radiation.boundary.outer_r == "vacuum",
                                       "predictor");
    holo_predictor_succeeded = predictor_result.failures == 0;
    if (!holo_predictor_succeeded) {
      last_holo_lo_result_ = predictor_result;
    }
    log_holo_lo_result("predictor", predictor_result);
  }

  double host_E_numerical_loss_step = 0.0;
  cuda_check(cudaMemcpy(&host_E_numerical_loss_step,
                        d_E_numerical_loss,
                        sizeof(host_E_numerical_loss_step),
                        cudaMemcpyDeviceToHost),
             "IMC::transport_step copy E_numerical_loss(final) failed");
  const double transport_numerical_loss_step = std::max(host_E_numerical_loss_step, 0.0);
  const double thermal_lost_step = std::max(last_thermal_lost_step_, 0.0);
  // Keep source discretization loss in the step energy budget, but do not
  // report it as transport numerical loss.
  last_numerical_loss_step_ = transport_numerical_loss_step + thermal_lost_step;
  // Only warn when numerical loss is significant relative to the step's energy scale.
  // Reference: total estimated particle energy = n_particles * E_avg.
  // Threshold: 0.1% - below this is normal MC noise (O(1/sqrt(N)) ~ 3% for ~1000 ppcg).
  {
    const double E_ref = static_cast<double>(std::max(n_alive_after_emit, 1)) * E_avg;
    if (transport_numerical_loss_step > 0.0 && E_ref > 0.0 &&
        transport_numerical_loss_step / E_ref > 1e-3) {
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(6)
          << "IMC transport: numerical energy loss=" << transport_numerical_loss_step
          << " erg (" << (transport_numerical_loss_step / E_ref * 100.0)
          << "% of E_ref="
          << E_ref << ")";
      core::log_warning(oss.str());
    }
  }

  // M15.5/M16.2: DDMC momentum estimation is diagnostic-only and currently disabled
  // in both GPU (1D and 2D_RZ) transport paths.
  const double total_momentum_dep = 0.0;
  last_rad_momentum_deposition_ = total_momentum_dep;

  TENRYU_ASSERT(pool_.n_alive >= 0 && pool_.n_alive <= pool_.capacity,
                "IMC::transport_step pool invariant violated before final composite sort");
  // Sort only when transport partitions need it. Census combing sorts lazily
  // inside the hard-threshold branch before GPU bin detection.
  const bool need_sort =
      sorted.n_ddmc > 0 || sorted.n_rw > 0 || state.particle_sort_cache_invalidated;
  bool census_bins_sorted = false;
  if (need_sort) {
    sorted = composite_sort_and_partition(pool_,
                                          pool_.n_alive,
                                          n_cells,
                                          n_groups,
                                          d_E_numerical_loss);
    census_bins_sorted = true;
  } else {
    sorted = compact_alive_only(pool_,
                                pool_.n_alive,
                                n_cells,
                                n_groups,
                                d_E_numerical_loss);
  }
  pool_.n_alive = sorted.n_alive;
  pool_.n_census = pool_.n_alive;
  state.particle_sort_cache_invalidated = false;
  last_n_census_ = static_cast<std::int64_t>(std::max(pool_.n_census, 0));
  const auto t_sort2_end = verbose_imc_timing ? Clock::now() : Clock::time_point{};

  // Population controller: update EMA estimators + safety-valve combing
  if (cfg.radiation.imc.census_comb.enabled) {
    const auto& cc = cfg.radiation.imc.census_comb;
    const int S_t = pool_.n_alive;  // post-transport census count

    // Split emissions: controlled (thermal, scales with ppcg) vs fixed (marshak, volume)
    const int E_ctrl = thermal_stats.n_thermal;
    const int E_fixed = marshak_stats.n_marshak + volume_stats.n_thermal;
    const int E_total = E_ctrl + E_fixed;

    const double I_t = static_cast<double>(std::max(N_start + E_total, 1));
    // Track removal fraction (1-rho) for numerical stability when rho ≈ 1
    const double rem_t = static_cast<double>(N_start + E_total - S_t) / I_t;
    const int ppcg_t = (ppcg_override > 0) ? ppcg_override
                                           : cfg.radiation.imc.particles_per_cell_group;
    // b_scaled uses only the ppcg-proportional emission
    const double b_scaled_t = static_cast<double>(E_ctrl) / static_cast<double>(std::max(ppcg_t, 1));

    constexpr double kAlpha = 0.1;
    if (pop_ctrl_ppcg_applied_ < 0) {
      // First step: initialize EMA state
      pop_ctrl_rem_ema_ = std::max(rem_t, 1e-6);
      pop_ctrl_b_scaled_ema_ = b_scaled_t;
      pop_ctrl_ppcg_applied_ = cfg.radiation.imc.particles_per_cell_group;
    } else {
      pop_ctrl_rem_ema_ += kAlpha * (rem_t - pop_ctrl_rem_ema_);
      pop_ctrl_rem_ema_ = std::max(pop_ctrl_rem_ema_, 1e-6);  // prevent exactly 0
      pop_ctrl_b_scaled_ema_ += kAlpha * (b_scaled_t - pop_ctrl_b_scaled_ema_);
    }

    // Safety-valve combing: only when hard threshold is exceeded
    const int N_max_eff = std::max(1, std::min(pool_.capacity, cc.max_particles));
    constexpr double kHardFraction = 0.98;
    constexpr double kSoftFraction = 0.92;
    const int N_hard = static_cast<int>(kHardFraction * N_max_eff);
    if (pool_.n_census > N_hard) {
      if (!census_bins_sorted) {
        sorted = composite_sort_and_partition(pool_,
                                              pool_.n_alive,
                                              n_cells,
                                              n_groups,
                                              d_E_numerical_loss);
        pool_.n_alive = sorted.n_alive;
        pool_.n_census = pool_.n_alive;
        last_n_census_ = static_cast<std::int64_t>(std::max(pool_.n_census, 0));
        census_bins_sorted = true;
      }
      const int n_alive_before = pool_.n_alive;
      const std::uint64_t n_emit_total_u64 =
          static_cast<std::uint64_t>(thermal_stats.n_thermal) +
          static_cast<std::uint64_t>(marshak_stats.n_marshak) +
          static_cast<std::uint64_t>(volume_stats.n_thermal);
      // Use a runtime copy of config with soft-cap target
      auto cc_runtime = cc;
      cc_runtime.max_particles = N_max_eff;
      cc_runtime.target_fraction = kSoftFraction;
      std::uint64_t comb_n_emit_total_u64 = n_emit_total_u64;
      if (cc_runtime.ess_floor_enabled && spectral_bias_eta > 0.0) {
        const std::vector<double> host_importance =
            compute_rosseland_importance(state, cfg, planck, host_sigma_t_bias);
        auto ess_result = census_ess_floor_gpu(pool_,
                                               pool_.n_alive,
                                               n_cells,
                                               n_groups,
                                               cc_runtime,
                                               host_importance,
                                               step_base_gid,
                                               n_emit_total_u64,
                                               cfg.main.seed,
                                               state.step);
        pool_.n_alive = ess_result.n_alive_out;
        pool_.n_census = pool_.n_alive;
        const int ess_added = std::max(pool_.n_alive - n_alive_before, 0);
        comb_n_emit_total_u64 += static_cast<std::uint64_t>(ess_added);
        if (cfg.main.verbosity == "verbose" || cfg.main.verbosity == "normal") {
          core::log_info("[ess_floor] step=" + std::to_string(state.step) +
                         " split_bins=" + std::to_string(ess_result.n_split_bins) +
                         " n_alive=" + std::to_string(pool_.n_alive));
        }
      }
      auto comb_result = census_comb_gpu(pool_,
                                     pool_.n_alive,
                                     n_cells,
                                     n_groups,
                                     cc_runtime,
                                     step_base_gid,
                                     comb_n_emit_total_u64,
                                     cfg.main.seed,
                                     state.step,
                                     d_E_numerical_loss);
      pool_.n_alive = comb_result.n_alive_out;
      pool_.n_census = pool_.n_alive;
      last_n_census_ = static_cast<std::int64_t>(pool_.n_census);
      core::log_debug("[census_comb] step=" + std::to_string(state.step) +
                     " S_t=" + std::to_string(S_t) +
                     " N_hard=" + std::to_string(N_hard) +
                     " before=" + std::to_string(n_alive_before) +
                     " after=" + std::to_string(pool_.n_alive) +
                     " target=" + std::to_string(comb_result.target_count) +
                     " bins=" + std::to_string(comb_result.n_combed_bins) +
                     " rem_ema=" + std::to_string(pop_ctrl_rem_ema_) +
                     " b_scaled_ema=" + std::to_string(pop_ctrl_b_scaled_ema_) +
                     " emergency=" +
                     std::to_string(comb_result.emergency ? 1 : 0) +
                     " E_killed=" + std::to_string(comb_result.E_killed_bins));
    }
  }
  if (source_localization_supported) {
    std::vector<double> host_sloc_abs_wr(n_cells_us, 0.0);
    std::vector<double> host_sloc_abs_wr2(n_cells_us, 0.0);
    std::vector<double> host_sloc_abs_E(n_cells_us, 0.0);
    std::vector<double> host_sloc_prev_mean_r(n_cells_us, 0.0);
    std::vector<double> host_sloc_prev_E(n_cells_us, 0.0);
    std::vector<double> host_sloc_mean_r(n_cells_us, 0.0);
    std::vector<double> host_sloc_sigma(n_cells_us, 0.0);
    std::vector<double> host_sloc_alpha(n_cells_us, 0.0);
    std::vector<double> host_sloc_sigma_R(static_cast<std::size_t>(n_cell_groups), 0.0);
    std::vector<double> host_node_r(state.x_r.size(), 0.0);
    sloc_abs_wr_.copy_to_host(host_sloc_abs_wr.data());
    sloc_abs_wr2_.copy_to_host(host_sloc_abs_wr2.data());
    sloc_abs_E_.copy_to_host(host_sloc_abs_E.data());
    sloc_mean_r_.copy_to_host(host_sloc_prev_mean_r.data());
    sloc_prev_E_.copy_to_host(host_sloc_prev_E.data());
    cuda_check(cudaMemcpy(host_sloc_sigma_R.data(),
                          d_sigma_R,
                          sizeof(double) * host_sloc_sigma_R.size(),
                          cudaMemcpyDeviceToHost),
               "IMC::transport_step copy sigma_R to host for source localization failed");
    state.x_r.copy_to_host(host_node_r.data());

    const double sloc_beta = cfg.radiation.imc.sloc_ema_beta;
    const double sloc_sigma_floor_frac = cfg.radiation.imc.sloc_sigma_floor;
    const double sloc_sigma_cap_frac = cfg.radiation.imc.sloc_sigma_cap;
    const double sloc_tau_ref = cfg.radiation.imc.sloc_tau_ref;
    const double sloc_E_gate =
        std::max(thermal_stats.E_thermal / static_cast<double>(std::max(n_cells, 1)), 1.0e-300);
    const double sloc_E_gate_threshold = 0.1 * sloc_E_gate;
    double sloc_abs_total = 0.0;
    int sloc_localized_cells = 0;
    int sloc_tau_gated_cells = 0;
    for (int c = 0; c < n_cells; ++c) {
      const std::size_t c_us = static_cast<std::size_t>(c);
      const double r_lo = host_node_r[c_us];
      const double r_hi = host_node_r[c_us + 1];
      const double dr = std::max(r_hi - r_lo, 0.0);
      const double r_center = 0.5 * (r_lo + r_hi);
      const double sigma_floor = std::max(sloc_sigma_floor_frac * dr, 0.0);
      const double sigma_cap = std::max(sloc_sigma_cap_frac * dr, sigma_floor);
      const double sigma_default = std::clamp(0.25 * dr, sigma_floor, sigma_cap);
      const double abs_E = fmax(host_sloc_abs_E[c_us], 0.0);
      host_sloc_abs_E[c_us] = abs_E;
      sloc_abs_total += abs_E;
      host_sloc_mean_r[c_us] = r_center;
      host_sloc_sigma[c_us] = sigma_default;
      host_sloc_alpha[c_us] = 0.0;
      if (abs_E > sloc_E_gate_threshold) {
        const double mu_raw = std::clamp(host_sloc_abs_wr[c_us] / std::max(abs_E, 1.0e-300),
                                         r_lo,
                                         r_hi);
        const double second_moment = host_sloc_abs_wr2[c_us] / std::max(abs_E, 1.0e-300);
        const double variance = std::max(second_moment - mu_raw * mu_raw, 0.0);
        const double sigma_raw = std::sqrt(variance);
        const double mu_filtered =
            (host_sloc_prev_E[c_us] > 0.0)
                ? (sloc_beta * mu_raw + (1.0 - sloc_beta) * host_sloc_prev_mean_r[c_us])
                : mu_raw;
        const double sigma_loc =
            std::clamp(std::max(sigma_raw, sigma_floor), sigma_floor, sigma_cap);
        double tau_max = 0.0;
        const std::size_t group_base = c_us * static_cast<std::size_t>(n_groups);
        for (int g = 0; g < n_groups; ++g) {
          const std::size_t cg = group_base + static_cast<std::size_t>(g);
          tau_max = std::max(tau_max, std::max(host_sloc_sigma_R[cg], 0.0) * dr);
        }
        const double w_tau = tau_max / (tau_max + sloc_tau_ref);
        const double alpha_E = std::min(1.0, abs_E / (abs_E + sloc_E_gate));
        host_sloc_mean_r[c_us] =
            std::clamp(r_center + w_tau * (mu_filtered - r_center), r_lo, r_hi);
        host_sloc_sigma[c_us] = std::clamp(sigma_default + w_tau * (sigma_loc - sigma_default),
                                           sigma_floor,
                                           sigma_cap);
        host_sloc_alpha[c_us] = std::clamp(alpha_E * w_tau, 0.0, 1.0);
        if (host_sloc_alpha[c_us] > 0.0) {
          ++sloc_localized_cells;
        }
        if (w_tau < 0.5) {
          ++sloc_tau_gated_cells;
        }
      }
    }

    sloc_mean_r_.copy_from_host(host_sloc_mean_r.data());
    sloc_sigma_.copy_from_host(host_sloc_sigma.data());
    sloc_alpha_.copy_from_host(host_sloc_alpha.data());
    sloc_prev_E_.copy_from_host(host_sloc_abs_E.data());
    core::log_info("[source_localization] step=" + std::to_string(state.step) +
                   " localized_cells=" + std::to_string(sloc_localized_cells) + "/" +
                   std::to_string(n_cells) +
                   " tau_gated_cells=" + std::to_string(sloc_tau_gated_cells) +
                   " absorbed_E=" + std::to_string(sloc_abs_total));
  }
  const auto t_comb_end = verbose_imc_timing ? Clock::now() : Clock::time_point{};
  auto t_post_subphase = t_comb_end;
  double post_tally_finalize_ms = 0.0;
  double post_diffusion_final_ms = 0.0;
  double post_diagnostics_log_ms = 0.0;
  double post_other_ms = 0.0;
  const auto mark_post_subphase = [&](double& value) {
    if (verbose_imc_timing) {
      const auto t_now = Clock::now();
      value += elapsed_ms(t_post_subphase, t_now);
      t_post_subphase = t_now;
    }
  };

  if (verbose_imc_timing) {
    core::log_info("[imc_timing] step=" + std::to_string(state.step) +
                   " emit=" + std::to_string(elapsed_ms(t_step_start, t_emit_end)) +
                   " prep=" + std::to_string(elapsed_ms(t_emit_end, t_prep_end)) +
                   " sort1=" + std::to_string(elapsed_ms(t_prep_end, t_sort1_end)) +
                   " imc=" + std::to_string(elapsed_ms(t_sort1_end, t_imc_end)) +
                   " ddmc=" + std::to_string(elapsed_ms(t_imc_end, t_ddmc_end)) +
                   " rw=" + std::to_string(elapsed_ms(t_ddmc_end, t_rw_end)) +
                   " imc_tail=" + std::to_string(elapsed_ms(t_rw_end, t_imc_tail_end)) +
                   " sort2=" + std::to_string(elapsed_ms(t_imc_tail_end, t_sort2_end)) +
                   " comb=" + std::to_string(elapsed_ms(t_sort2_end, t_comb_end)) +
                   " total=" + std::to_string(elapsed_ms(t_step_start, t_comb_end)));
    const double prep_subphase_total_ms =
        prep_radlite_check_ms + prep_other_ms + prep_pgrw_prep_ms +
        prep_ddmc_input_ms + prep_ddmc_prep_ms + prep_sloc_prep_ms +
        prep_source_smooth_ms + prep_face_alloc_ms;
    core::log_info("[prep_subphase] step=" + std::to_string(state.step) +
                   " ddmc_input=" + std::to_string(prep_ddmc_input_ms) +
                   " ddmc_prep=" + std::to_string(prep_ddmc_prep_ms) +
                   " pgrw_prep=" + std::to_string(prep_pgrw_prep_ms) +
                   " sloc_prep=" + std::to_string(prep_sloc_prep_ms) +
                   " radlite_check=" + std::to_string(prep_radlite_check_ms) +
                   " source_smooth=" + std::to_string(prep_source_smooth_ms) +
                   " face_alloc=" + std::to_string(prep_face_alloc_ms) +
                   " other=" + std::to_string(prep_other_ms) +
                   " total=" + std::to_string(prep_subphase_total_ms));
  }
  mark_post_subphase(post_other_ms);

  const std::int64_t imc_absorbed_i64 = saturating_i64_from_u64(host_imc_absorbed);
  const std::int64_t imc_escaped_i64 = saturating_i64_from_u64(host_imc_escaped);
  const std::int64_t ddmc_absorbed_i64 = std::max(ddmc_diag.absorbed, static_cast<std::int64_t>(0));
  const std::int64_t ddmc_escaped_i64 =
      std::max(ddmc_diag.leak_boundary, static_cast<std::int64_t>(0));
  last_n_absorbed_ = saturating_add_i64(imc_absorbed_i64, ddmc_absorbed_i64);
  last_n_escaped_ = saturating_add_i64(
      saturating_add_i64(imc_escaped_i64, ddmc_escaped_i64),
      std::max(rw_escaped, static_cast<std::int64_t>(0)));

  compute_weight_stats_device(pool_, &last_weight_min_, &last_weight_mean_, &last_weight_max_);
  mark_post_subphase(post_tally_finalize_ms);

  if (diffusion_enabled_1d) {
    double diffusion_E_final =
        has_diffusion_cells
            ? compute_diffusion_energy_host(diff_E_, diff_cell_, diff_vol_)
            : 0.0;
    const double diffusion_E_in_total =
        diffusion_interface_E_in + diffusion_interface_E_in_tail;
    const double diffusion_E_vacuum =
        std::max(deterministic_diffusion_result.E_leaked, 0.0) +
        std::max(diffusion_interface_result.E_leaked_vacuum, 0.0);
    double diffusion_E_out_total =
        diffusion_interface_result.E_spawned + diffusion_emergency_revert_E_out;
    const double final_growth_reference =
        diffusion_growth_reference(diffusion_E_balance_start,
                                   diffusion_E_in_total,
                                   diffusion_E_out_total,
                                   diffusion_E_vacuum,
                                   diffusion_source_result.matter_delta);
    if (has_diffusion_cells &&
        diffusion_energy_growth_exceeded(final_growth_reference,
                                         diffusion_E_final)) {
      force_diffusion_cells_to_imc("step_growth",
                                   final_growth_reference,
                                   diffusion_E_final);
      diffusion_E_final =
          has_diffusion_cells
              ? compute_diffusion_energy_host(diff_E_, diff_cell_, diff_vol_)
              : 0.0;
      diffusion_E_out_total =
          diffusion_interface_result.E_spawned + diffusion_emergency_revert_E_out;
    }
    if (diffusion_emergency_reverted) {
      compute_weight_stats_device(pool_, &last_weight_min_, &last_weight_mean_, &last_weight_max_);
    }
    log_diffusion_step_balance_if_bad(state.step,
                                      diffusion_E_balance_start,
                                      diffusion_E_in_total,
                                      diffusion_E_out_total,
                                      diffusion_E_vacuum,
                                      diffusion_source_result.matter_delta,
                                      diffusion_E_final);
    const double particle_E_final = compute_alive_particle_energy_host(pool_);
    std::ostringstream summary;
    summary << std::scientific << std::setprecision(6);
    summary << "[diffusion] step=" << state.step
            << " n_diff=" << n_diffusion_cells_step
            << " n_guard=" << n_diffusion_guard_cells_step
            << " rkl2_stages=" << deterministic_diffusion_result.rkl2_stages
            << " rkl2_subcycles=" << deterministic_diffusion_result.rkl2_subcycles
            << " dt_explicit=" << deterministic_diffusion_result.dt_explicit
            << " rkl2_skipped=" << (deterministic_diffusion_result.rkl2_skipped ? 1 : 0)
            << " source_iter=" << diffusion_source_result.max_newton_iter
            << " E_diff=" << diffusion_E_final
            << " E_in=" << diffusion_E_in_total
            << " E_out=" << diffusion_E_out_total
            << " E_particle=" << particle_E_final
            << " safety_reverted=" << (diffusion_emergency_reverted ? 1 : 0);
    core::log_info(summary.str());

    std::ostringstream oss;
    oss << std::scientific << std::setprecision(6);
    oss << "[diffusion_interface] step=" << state.step
        << " packets_killed=" << host_diffusion_interface_kills
        << " E_in=" << diffusion_E_in_total
        << " packets_spawned="
        << (static_cast<std::int64_t>(diffusion_interface_result.n_spawned) +
            static_cast<std::int64_t>(diffusion_emergency_revert_particles))
        << " E_out=" << diffusion_E_out_total
        << " E_vacuum=" << diffusion_E_vacuum;
    core::log_info(oss.str());
  }
  mark_post_subphase(post_diffusion_final_ms);

  if (ddmc_preparation.active && cfg.main.verbosity == "verbose") {
    if (use_implicit_ddmc_diffusion) {
      core::log_info("[ddmc_diffusion] mode_ddmc=" +
                     std::to_string(ddmc_preparation.ddmc_mode_count) +
                     " mode_rw=" +
                     std::to_string(ddmc_preparation.rw_mode_count) +
                     " mode_imc=" + std::to_string(ddmc_preparation.imc_mode_count) +
                     " solved_cell_groups=" +
                     std::to_string(ddmc_diffusion_result.solved_cell_groups) +
                     " folded_interface_census=" +
                     std::to_string(folded_ddmc_census) +
                     " interface_reflections=" +
                     std::to_string(host_interface_reflections) +
                     " escaped_energy=" +
                     std::to_string(ddmc_diffusion_result.escaped_energy));
    } else {
      core::log_info("[ddmc] mode_ddmc=" + std::to_string(ddmc_preparation.ddmc_mode_count) +
                     " mode_rw=" + std::to_string(ddmc_preparation.rw_mode_count) +
                     " mode_imc=" + std::to_string(ddmc_preparation.imc_mode_count) +
                     " mmatrix_violations=" +
                     std::to_string(ddmc_preparation.mmatrix.total_violations) +
                     " processed=" + std::to_string(ddmc_diag.processed) +
                     " converted_to_imc=" +
                     std::to_string(ddmc_diag.converted_to_imc) +
                     " converted_to_rw=" +
                     std::to_string(ddmc_diag.converted_to_rw) +
                     " interface_to_ddmc=" +
                     std::to_string(host_interface_transitions) +
                     " interface_reflections=" +
                     std::to_string(host_interface_reflections) +
                     " prob_fallbacks=" +
                     std::to_string(host_conversion_prob_violations) +
                     " mom_dep=" + std::to_string(total_momentum_dep));
    }
  }
  if (diffusion_enabled_1d && cfg.main.verbosity == "verbose") {
    core::log_info("[diffusion_source] max_newton_iter=" +
                   std::to_string(diffusion_source_result.max_newton_iter) +
                   " failures=" +
                   std::to_string(diffusion_source_result.n_failures));
    core::log_info("[diffusion_rkl2] stages=" +
                   std::to_string(deterministic_diffusion_result.rkl2_stages) +
                   " subcycles=" +
                   std::to_string(deterministic_diffusion_result.rkl2_subcycles) +
                   " dt_explicit=" +
                   std::to_string(deterministic_diffusion_result.dt_explicit) +
                   " skipped=" +
                   std::to_string(deterministic_diffusion_result.rkl2_skipped ? 1 : 0) +
                   " E_before=" +
                   std::to_string(deterministic_diffusion_result.E_before) +
                   " E_after=" +
                   std::to_string(deterministic_diffusion_result.E_after) +
                   " E_leaked=" +
                   std::to_string(deterministic_diffusion_result.E_leaked) +
                   " positivity_limited=" +
                   std::to_string(
                       deterministic_diffusion_result.positivity_limited ? 1 : 0) +
                   " E_negative_before=" +
                   std::to_string(deterministic_diffusion_result.E_negative_before) +
                   " E_limiter_removed=" +
                   std::to_string(deterministic_diffusion_result.E_limiter_removed));
  }

  if (cfg.main.verbosity == "verbose" && (rw_processed > 0 || rw_converted_to_imc > 0 ||
                                          rw_converted_to_ddmc > 0)) {
    core::log_info("[rw] processed=" + std::to_string(rw_processed) +
                   " census=" + std::to_string(rw_census) +
                   " leak_left=" + std::to_string(rw_leak_left) +
                   " leak_right=" + std::to_string(rw_leak_right) +
                   " escaped=" + std::to_string(rw_escaped) +
                   " converted_to_imc=" + std::to_string(rw_converted_to_imc) +
                   " converted_to_ddmc=" + std::to_string(rw_converted_to_ddmc));
  }

  if (cfg.main.verbosity == "verbose") {
    const unsigned long long total_events = host_cnt_boundary + host_cnt_scatter +
                                            host_cnt_census + host_cnt_absorb_kill +
                                            host_cnt_roulette_kill;
    const double events_per_particle =
        (n_imc_transport > 0)
            ? (static_cast<double>(total_events) /
               static_cast<double>(n_imc_transport))
            : 0.0;
    core::log_info("[imc_events] boundary=" + std::to_string(host_cnt_boundary) +
                   " scatter=" + std::to_string(host_cnt_scatter) +
                   " census=" + std::to_string(host_cnt_census) +
                   " absorb_kill=" + std::to_string(host_cnt_absorb_kill) +
                   " absorb_survive=" + std::to_string(host_cnt_absorb_survive) +
                   " roulette_kill=" + std::to_string(host_cnt_roulette_kill) +
                   " total=" + std::to_string(total_events) +
                   " events_per_particle=" + std::to_string(events_per_particle));
    core::log_info("[imc_fleck] f_min=" + std::to_string(last_f_min_) +
                   " f_mean=" + std::to_string(last_f_mean_) +
                   " f_p95=" + std::to_string(last_f_p95_));
    core::log_info("[imc_tau] gt1=" + std::to_string(last_tau_gt1_) +
                   " gt2=" + std::to_string(last_tau_gt2_) +
                   " gt3=" + std::to_string(last_tau_gt3_) +
                   " gt4=" + std::to_string(last_tau_gt4_) +
                   " total_cell_groups=" + std::to_string(n_cell_groups));
    core::log_info("[imc_hysteresis] I->D=" + std::to_string(last_switches_imc_to_ddmc_) +
                   " D->nonD=" + std::to_string(last_switches_ddmc_to_imc_));
  }

  if (use_freq_dep_marshak && cfg.main.verbosity != "quiet") {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(6);
    oss << "[imc:pool] step=" << state.step << ", dt=" << dt
        << ", alive_before_emit=" << std::max(n_alive_before_emit, 0)
        << ", alive_after_emit=" << n_alive_after_emit
        << ", alive_after_sort=" << pool_.n_alive;
    core::log_info(oss.str());
  }
  mark_post_subphase(post_diagnostics_log_ms);

  if (holo_prr_enabled) {
    parallel::DeviceArray d_holo_core_for_prr;
    const std::uint8_t* d_holo_core_ptr = nullptr;
    if (state.holo_core_mask_valid && state.holo_core_mask.size() == n_cells_us) {
      copy_u8_to_device(state.holo_core_mask,
                        d_holo_core_for_prr,
                        "IMC::transport_step copy HOLO core mask for Prr failed");
      d_holo_core_ptr = d_holo_core_for_prr.as<std::uint8_t>();
    }
    const double te_floor = std::max(cfg.numerics.floors.Te, 1.0e-12);
    const double E_floor =
        core::constants::a_eV * te_floor * te_floor * te_floor * te_floor;
    holo_prr_finalize_cuda(state.holo_Prr.data(),
                           state.holo_chi.data(),
                           state.holo_Prr_coverage.data(),
                           state.holo_Prr.data(),
                           state.holo_Prr_coverage.data(),
                           d_rad_E_tally,
                           state.vol.data(),
                           d_holo_core_ptr,
                           n_cells,
                           n_groups,
                           dt,
                           E_floor);
  }

  tally_finalize_cuda(state.rad_E.data(),
                      d_rad_E_tally,
                      state.vol.data(),
                      n_cells,
                      n_groups,
                      dt,
                      (diffusion_enabled_1d && have_diffusion_device_masks)
                          ? diff_E_.as<double>()
                          : nullptr,
                      (diffusion_enabled_1d && have_diffusion_device_masks)
                          ? d_diff_cell.as<std::uint8_t>()
                          : nullptr,
                      d_E_ref_avg_for_finalize,
                      (difference_source_supported &&
                       state.difference_residual_E.size() == n_cell_groups_us)
                          ? state.difference_residual_E.data()
                          : nullptr);

  if (holo_enabled_1d && state.holo_core_mask_valid && holo_predictor_succeeded &&
      n_cell_groups_us > 0U) {
    TENRYU_ASSERT(state.holo_E_LO.size() == n_cell_groups_us,
                  "IMC::transport_step HOLO acceptance requires holo_E_LO size match");
    TENRYU_ASSERT(state.holo_consistency_source.size() == n_cell_groups_us,
                  "IMC::transport_step HOLO consistency source size match");
    TENRYU_ASSERT(state.holo_lo_weight.size() == n_cells_us,
                  "IMC::transport_step HOLO acceptance requires holo_lo_weight size match");
    TENRYU_ASSERT(state.holo_core_mask.size() == n_cells_us,
                  "IMC::transport_step HOLO acceptance requires holo_core_mask size match");
    const std::size_t E_old_bytes = sizeof(double) * n_cell_groups_us;
    TENRYU_ASSERT(d_holo_E_old.size == E_old_bytes,
                  "IMC::transport_step HOLO E_old size mismatch");
    parallel::DeviceArray d_holo_lo_weight;
    const std::size_t weight_bytes = sizeof(double) * n_cells_us;
    d_holo_lo_weight.resize(weight_bytes);
    if (weight_bytes > 0U) {
      cuda_check(cudaMemcpy(d_holo_lo_weight.as<double>(),
                            state.holo_lo_weight.data(),
                            weight_bytes,
                            cudaMemcpyHostToDevice),
                 "IMC::transport_step copy HOLO LO weights failed");
    }
    const double* d_holo_face_current =
        face_current_tracking_active ? face_current_step_.as<double>() : nullptr;
    const double* d_holo_reference_face_current =
        (difference_source_supported && diff_ref_face_current_.size == face_current_bytes)
            ? diff_ref_face_current_.as<double>()
            : nullptr;
    compute_holo_consistency_cuda(state.holo_consistency_source.data(),
                                  state.rad_E.data(),
                                  state.holo_E_LO.data(),
                                  d_holo_E_old.as<double>(),
                                  d_holo_face_current,
                                  d_holo_reference_face_current,
                                  d_sigma_R,
                                  state.vol.data(),
                                  state.x_r.data(),
                                  d_holo_lo_weight.as<double>(),
                                  n_cells,
                                  n_groups,
                                  dt);
    const bool holo_face_current_physical =
        face_current_tracking_active && !difference_source_supported;
    last_holo_lo_result_ =
        solve_holo_lo_source_ownership(state,
                                       cfg,
                                       planck,
                                       mat,
                                       d_sigma_a,
                                       d_sigma_R,
                                       d_f,
                                       n_cells,
                                       n_groups,
                                       dt,
                                       d_holo_E_old.as<double>(),
                                       E_old_bytes,
                                       state.holo_consistency_source.data(),
                                       E_old_bytes,
                                       holo_face_current_physical
                                           ? face_current_step_.as<double>()
                                           : nullptr,
                                       holo_face_current_physical ? face_current_bytes : 0U,
                                       nullptr,
                                       0U,
                                       true,
                                       true,
                                       difference_source_supported,
                                       cfg.radiation.boundary.outer_r == "vacuum",
                                       "corrector");
    log_holo_lo_result("corrector", last_holo_lo_result_);
    if (state.holo_lo_source_valid) {
      parallel::DeviceArray d_holo_core_for_accept;
      copy_u8_to_device(state.holo_core_mask,
                        d_holo_core_for_accept,
                        "IMC::transport_step copy HOLO core mask for acceptance failed");
      const std::uint8_t* d_holo_core_accept =
          d_holo_core_for_accept.as<std::uint8_t>();
      holo_accept_radiation_cuda(state.rad_E.data(),
                                 state.holo_E_LO.data(),
                                 d_holo_core_accept,
                                 n_cells,
                                 n_groups,
                                 nullptr,
                                 nullptr);
      if (difference_source_supported) {
        const std::size_t E_ref_bytes = sizeof(double) * n_cell_groups_us;
        previous_reference_U_device_.resize(E_ref_bytes);
        holo_set_reference_U_cuda(previous_reference_U_device_.as<double>(),
                                  state.rad_E.data(),
                                  state.vol.data(),
                                  d_holo_core_accept,
                                  n_cells,
                                  n_groups);
        previous_reference_U_device_valid_ =
            previous_reference_U_device_.size == E_ref_bytes;
        previous_reference_U_.clear();
        if (state.difference_residual_E.size() == n_cell_groups_us &&
            d_E_ref_avg_for_finalize != nullptr) {
          difference_reproject_residual_cuda(state.difference_residual_E.data(),
                                             state.rad_E.data(),
                                             d_E_ref_avg_for_finalize,
                                             n_cells,
                                             n_groups,
                                             d_holo_core_accept);
        }

        const int killed_holo_core_census = kill_difference_census_in_holo_core_cuda(
            pool_,
            d_holo_core_accept,
            pool_.n_alive,
            n_cells);
        if (killed_holo_core_census > 0) {
          const CompositeSortResult compacted =
              compact_alive_only(pool_, pool_.n_alive, n_cells, n_groups, nullptr);
          pool_.n_alive = compacted.n_alive;
          pool_.n_census = pool_.n_alive;
          last_n_census_ = static_cast<std::int64_t>(std::max(pool_.n_census, 0));
          if (cfg.main.verbosity == "verbose") {
            core::log_info("[holo_df_recenter] step=" + std::to_string(state.step) +
                           " killed_core_census=" +
                           std::to_string(killed_holo_core_census) +
                           " n_alive=" + std::to_string(pool_.n_alive));
          }
        }
      }
    }
  } else if (holo_enabled_1d &&
             state.holo_consistency_source.size() == n_cell_groups_us) {
    state.holo_consistency_source.fill(0.0);
  }

  core::DeviceErrorFlags host_flags{};
  cuda_check(cudaMemcpy(&host_flags,
                        d_error_flags,
                        sizeof(host_flags),
                        cudaMemcpyDeviceToHost),
             "IMC::transport_step copy error flags failed");
  host_flags.opacity_out_of_range =
      std::max(host_flags.opacity_out_of_range, opacity_flags.opacity_out_of_range);
  host_flags.pool_overflow =
      std::max(host_flags.pool_overflow, state.radiation_device_flags.pool_overflow);
  last_device_error_flags_ = host_flags;
  log_imc_device_flags(host_flags);
  fatal_on_excess_infinite_loop(host_flags, "transport_step");
  if (host_flags.invalid_cell != 0 || host_flags.nan_particle != 0) {
    TENRYU_ASSERT(false,
                  "IMC transport: fatal device error flag set (invalid_cell or nan_particle)");
  }

  std::vector<double> host_E_escape_step(static_cast<std::size_t>(n_groups), 0.0);
  cuda_check(cudaMemcpy(host_E_escape_step.data(),
                        d_E_escape,
                        sizeof(double) * host_E_escape_step.size(),
                        cudaMemcpyDeviceToHost),
             "IMC::transport_step copy E_escape(step) failed");
  for (const double e : host_E_escape_step) {
    escaped_energy_total_ += std::max(e, 0.0);
  }
  mark_post_subphase(post_tally_finalize_ms);

  if (face_current_tracking_enabled_1d) {
    const std::size_t current_bytes = face_current_bytes;
    if (face_current_prev_.size == current_bytes &&
        face_current_step_.size == current_bytes) {
      std::swap(face_current_prev_, face_current_step_);
      face_current_prev_dt_ = dt;
      if (current_bytes > 0U) {
        cuda_check(cudaMemset(face_current_step_.ptr, 0, current_bytes),
                   "IMC::transport_step zero diffusion face current step failed");
      }
    }
    if (face_current_in_.size == current_bytes && current_bytes > 0U) {
      cuda_check(cudaMemset(face_current_in_.ptr, 0, current_bytes),
                 "IMC::transport_step zero diffusion face current in end failed");
    }
  }
  if (diffusion_enabled_1d) {
    diff_cell_prev_ = diff_cell_;
  }
  mark_post_subphase(post_diffusion_final_ms);

  if (d_ddmc_mode != nullptr) {
    cuda_check(cudaFree(d_ddmc_mode), "IMC::transport_step cudaFree ddmc_mode failed");
  }
  if (rad_device_data.active) {
    free_rad_device_data(rad_device_data);
  }
  if (is_nlte_mode && nlte_coeffs.is_nlte) {
    last_cell_radiation_coeffs_ = std::move(nlte_coeffs);
    last_cell_radiation_coeffs_valid_ = true;
  }
  if (verbose_imc_timing) {
    const double pre_total_ms = pre_setup_ms + pre_te_pred_ms + pre_nlte_coeffs_ms +
                                pre_nlte_h2d_ms + pre_difference_setup_ms +
                                pre_fleck_build_ms + pre_other_A_ms +
                                pre_other_B_ms + pre_other_C_ms + pre_other_D_ms;
    core::log_info("[imc_presub] step=" + std::to_string(state.step) +
                   " setup=" + std::to_string(pre_setup_ms) +
                   " te_pred=" + std::to_string(pre_te_pred_ms) +
                   " nlte_coeffs=" + std::to_string(pre_nlte_coeffs_ms) +
                   " nlte_h2d=" + std::to_string(pre_nlte_h2d_ms) +
                   " diff_setup=" + std::to_string(pre_difference_setup_ms) +
                   " fleck_build=" + std::to_string(pre_fleck_build_ms) +
                   " other_A=" + std::to_string(pre_other_A_ms) +
                   " other_B=" + std::to_string(pre_other_B_ms) +
                   " other_C=" + std::to_string(pre_other_C_ms) +
                   " other_D=" + std::to_string(pre_other_D_ms) +
                   " total=" + std::to_string(pre_total_ms));
  }
  mark_post_subphase(post_other_ms);
  if (verbose_imc_timing) {
    const double post_total_ms = post_tally_finalize_ms + post_diffusion_final_ms +
                                 post_diagnostics_log_ms + post_other_ms;
    core::log_info("[imc_postsub] step=" + std::to_string(state.step) +
                   " tally_final=" + std::to_string(post_tally_finalize_ms) +
                   " diff_final=" + std::to_string(post_diffusion_final_ms) +
                   " diag_log=" + std::to_string(post_diagnostics_log_ms) +
                   " post_other=" + std::to_string(post_other_ms) +
                   " total=" + std::to_string(post_total_ms));
  }
}

double IMC::census_energy() const {
  long double sum = 0.0L;
  std::size_t clamped_negative = 0;

  if (previous_reference_U_device_valid_ && previous_reference_U_device_.size > 0U) {
    const std::size_t n_ref = previous_reference_U_device_.size / sizeof(double);
    std::vector<double> host_reference_U(n_ref, 0.0);
    cuda_check(cudaMemcpy(host_reference_U.data(),
                          previous_reference_U_device_.as<double>(),
                          sizeof(double) * host_reference_U.size(),
                          cudaMemcpyDeviceToHost),
               "IMC::census_energy copy previous reference U failed");
    for (const double U_ref : host_reference_U) {
      if (std::isfinite(U_ref)) {
        sum += static_cast<long double>(U_ref);
      }
    }
  } else {
    for (const double U_ref : previous_reference_U_) {
      if (std::isfinite(U_ref)) {
        sum += static_cast<long double>(U_ref);
      }
    }
  }

  if (pool_.n_alive > 0 && pool_.energy != nullptr && pool_.alive != nullptr) {
    TENRYU_ASSERT(pool_.n_alive <= pool_.capacity,
                  "IMC::census_energy pool.n_alive exceeds pool.capacity");

    std::vector<double> host_energy(static_cast<std::size_t>(pool_.n_alive), 0.0);
    std::vector<std::int8_t> host_sign(static_cast<std::size_t>(pool_.n_alive), 1);
    std::vector<std::uint8_t> host_alive(static_cast<std::size_t>(pool_.n_alive), kDead);
    cuda_check(cudaMemcpy(host_energy.data(),
                          pool_.energy,
                          sizeof(double) * host_energy.size(),
                          cudaMemcpyDeviceToHost),
               "IMC::census_energy copy energy failed");
    if (pool_.sign != nullptr) {
      cuda_check(cudaMemcpy(host_sign.data(),
                            pool_.sign,
                            sizeof(std::int8_t) * host_sign.size(),
                            cudaMemcpyDeviceToHost),
                 "IMC::census_energy copy sign failed");
    }
    cuda_check(cudaMemcpy(host_alive.data(),
                          pool_.alive,
                          sizeof(std::uint8_t) * host_alive.size(),
                          cudaMemcpyDeviceToHost),
               "IMC::census_energy copy alive failed");

    for (std::size_t i = 0; i < host_energy.size(); ++i) {
      if (host_alive[i] == kAlive) {
        const double E = host_energy[i];
        if (E < 0.0) {
          ++clamped_negative;
          continue;
        }
        const std::int8_t s = (host_sign[i] < 0) ? static_cast<std::int8_t>(-1)
                                                 : static_cast<std::int8_t>(1);
        sum += static_cast<long double>(s) * static_cast<long double>(E);
      }
    }
  }

  const std::size_t n_cells = diff_cell_.size();
  if (n_cells > 0U && diff_E_.ptr != nullptr && diff_E_.size > 0U) {
    TENRYU_ASSERT(diff_vol_.size() == n_cells,
                  "IMC::census_energy diffusion volume size mismatch");
    const std::size_t n_values = diff_E_.size / sizeof(double);
    TENRYU_ASSERT(n_values % n_cells == 0U,
                  "IMC::census_energy diffusion energy size mismatch");
    const std::size_t n_groups = n_values / n_cells;
    std::vector<double> host_diff_E(n_values, 0.0);
    cuda_check(cudaMemcpy(host_diff_E.data(),
                          diff_E_.as<double>(),
                          sizeof(double) * host_diff_E.size(),
                          cudaMemcpyDeviceToHost),
               "IMC::census_energy copy diff_E failed");
    for (std::size_t c = 0; c < n_cells; ++c) {
      if (diff_cell_[c] == 0U) {
        continue;
      }
      const double V = diff_vol_[c];
      for (std::size_t g = 0; g < n_groups; ++g) {
        sum += static_cast<long double>(host_diff_E[c * n_groups + g]) *
               static_cast<long double>(V);
      }
    }
  }

  if (clamped_negative > 0) {
    core::log_warning("IMC::census_energy: clamped negative alive-particle energy to 0 (count=" +
                      std::to_string(clamped_negative) + ")");
  }
  return static_cast<double>(sum);
}

void IMC::set_last_overshoot_metrics(const std::int64_t count, const double max_ratio) {
  last_overshoot_count_ = std::max<std::int64_t>(count, 0);
  last_overshoot_max_ = std::max(max_ratio, 0.0);
}

void IMC::restore_photon_pool(PhotonPool&& pool) {
  pool_ = std::move(pool);
  pool_initialized_ = (pool_.capacity > 0);
  previous_reference_U_.clear();
  previous_reference_U_device_valid_ = false;
}

void IMC::invalidate_difference_reference() {
  previous_reference_U_.clear();
  previous_reference_U_device_valid_ = false;
  if (pool_.n_alive > 0 && pool_.sign != nullptr) {
    constexpr int kBlock = 256;
    const int grid = (pool_.n_alive + kBlock - 1) / kBlock;
    flip_negative_census_signs_cuda(pool_.sign, pool_.n_alive);
  }
}

void IMC::transport_step(const double dt) {
  // Legacy single-argument API kept for compatibility with early milestones.
  // Production transport uses the state+config overload above.
  (void)dt;
}

}  // namespace tenryu::radiation
