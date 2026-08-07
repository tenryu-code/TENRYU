#include "laser/laser.cuh"

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include "core/constants.hpp"
#include "core/device_error_flags.cuh"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "laser/beams.cuh"
#include "laser/bilinear_interpolation.cuh"
#include "laser/cbet_lm_fields.cuh"
#include "laser/deposit_transfer.cuh"
#include "laser/cbet.cuh"
#include "laser/fast_trace_1d.cuh"
#include "laser/hot_e_eta_model.hpp"
#include "laser/hot_electron_1d.cuh"
#include "laser/hot_electron_1d_gpu.cuh"
#include "laser/hot_electron_2d.cuh"
#include "laser/hot_electron_2d_gpu.cuh"
#include "laser/laser_phys_ext.cuh"
#include "laser/port_geometry.hpp"
#include "laser/port_section_chi.hpp"
#include "laser/port_section_overlap.hpp"
#include "laser/ray_init.cuh"
#include "laser/ray_trace.cuh"
#include "laser/raytrace_skip.cuh"
#include "laser/sector_adapter.hpp"
#include "laser/sector_phase_space.hpp"

namespace tenryu::laser {
namespace {

struct PortSectionState {
  port_geom::PortTable ports;
  sector_ps::PhaseSpaceTable table;
  std::vector<sector_ps::RayAnnotation> annotations;
  sector_ps::ExclusionLedger ledger{};
  std::vector<std::int32_t> ray_bin;
  std::unique_ptr<::tenryu::laser::port_section::ChiDeviceWorkspace> chi_ws;
  std::vector<std::int16_t> pair_p;
  std::vector<std::int16_t> pair_q;
  std::vector<std::int32_t> pair_index;
  std::vector<double> port_weight;
  std::vector<double> ps_capture_thresh;
  std::vector<std::int32_t> ps_capture_order;
  bool ps_static_built = false;
  int build_count = 0;
  sector_adapter::S1Audit audit{};
};

struct IncidentAttenuationContext {
  const std::vector<double>* rec_S = nullptr;
};

static double g_ps_timing_sum[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
static long g_ps_timing_calls = 0;

double apply_incident_attenuation(const double power_in,
                                  const std::int64_t record_index,
                                  const void* context) {
  const auto* attenuation =
      static_cast<const IncidentAttenuationContext*>(context);
  const double S_half =
      0.5 * (*attenuation->rec_S)[static_cast<std::size_t>(record_index)];
  double power = power_in;
  const double dP_first = -power * std::expm1(-S_half);
  power -= dP_first;
  const double dP_second = -power * std::expm1(-S_half);
  power -= dP_second;
  return power;
}

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

LaserPhysExtOptions build_phys_ext_options(
    const core::Config::LaserConfig& laser) {
  LaserPhysExtOptions options;
  if (laser.ib.zeff_model == "sequential_strip") {
    options.zeff_model = 1;
  } else if (laser.ib.zeff_model == "table") {
    options.zeff_model = 2;
    options.zt_nd = laser.ib.zeff_table.ndens;
    options.zt_nt = laser.ib.zeff_table.ntemp;
    options.zt_l10d0 = std::log10(laser.ib.zeff_table.ni_grid.front());
    if (laser.ib.zeff_table.ni_grid.size() >= 2U) {
      options.zt_dl10d =
          std::log10(laser.ib.zeff_table.ni_grid[1]) - options.zt_l10d0;
    }
    options.zt_l10t0 =
        std::log10(laser.ib.zeff_table.T_grid_eV.front());
    if (laser.ib.zeff_table.T_grid_eV.size() >= 2U) {
      options.zt_dl10t =
          std::log10(laser.ib.zeff_table.T_grid_eV[1]) - options.zt_l10t0;
    }
  }
  options.n_species = static_cast<int>(
      std::min<std::size_t>(
          std::min(laser.ib.species_z.size(), laser.ib.species_x.size()),
          static_cast<std::size_t>(kIBExtMaxSpecies)));
  double zcoll_num = 0.0;
  double zcoll_den = 0.0;
  for (int s = 0; s < options.n_species; ++s) {
    const std::size_t i = static_cast<std::size_t>(s);
    options.z_nuc[s] = laser.ib.species_z[i];
    options.x_frac[s] = laser.ib.species_x[i];
    zcoll_num += options.x_frac[s] * options.z_nuc[s] * options.z_nuc[s];
    zcoll_den += options.x_frac[s] * options.z_nuc[s];
  }
  options.coulomb_log_model =
      (laser.ib.coulomb_log_model == "laser_frequency") ? 1 : 0;
  options.langdon_model =
      (laser.ib.langdon_model == "legacy_vacuum_map") ? 1 : 0;
  options.langdon_zcoll =
      (zcoll_den > 0.0) ? (zcoll_num / zcoll_den) : 1.0;
  options.langdon_te_min_eV = laser.ib.langdon_te_min_eV;
  options.ra_enable = laser.ra.enable ? 1 : 0;
  options.ra_chi_p = laser.ra.chi_p;
  options.ra_c = laser.ra.c_ra;
  options.crit_terminate_deposit =
      (laser.absorption.terminate_mode == "deposit") ? 1 : 0;
  constexpr double kPi = 3.14159265358979323846;
  const double lambda_cm = laser.wavelength_nm * 1.0e-7;
  options.k0_cm_inv = 2.0 * kPi / lambda_cm;
  return options;
}

bool beam_fold_disabled() {
  static const bool disabled = [] {
    const char* v = std::getenv("TENRYU_LASER_NO_BEAM_FOLD");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  return disabled;
}

bool ray_sort_disabled() {
  static const bool disabled = [] {
    const char* v = std::getenv("TENRYU_LASER_NO_RAY_SORT");
    return v != nullptr && std::strcmp(v, "1") == 0;
  }();
  return disabled;
}

bool hot_e_2d_host_pipeline_enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("TENRYU_HOTE2D_HOST_PIPELINE");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  return enabled;
}

const std::string& hot_e_2d_dump_path() {
  static const std::string path = [] {
    const char* v = std::getenv("TENRYU_HOTE2D_DUMP");
    return (v != nullptr && v[0] != '\0') ? std::string(v) : std::string{};
  }();
  return path;
}

bool trace_tau_diag_enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("TENRYU_TRACE_TAU_DIAG");
    return v != nullptr && std::strcmp(v, "1") == 0;
  }();
  return enabled;
}

int trace_tau_diag_max_step() {
  static const int max_step = [] {
    const char* v = std::getenv("TENRYU_TRACE_TAU_DIAG_MAXSTEP");
    if (v == nullptr || v[0] == '\0') {
      return 3;
    }
    char* end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end == v || *end != '\0' ||
        parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
      return 3;
    }
    return static_cast<int>(parsed);
  }();
  return max_step;
}

void write_tau_diag_dump(const std::string& output_dir,
                         const char* trace_kind,
                         const int step,
                         const std::size_t beam_index,
                         const int n_rays,
                         const int n_intervals,
                         const std::vector<double>& tau_shell,
                         const bool legs_only) {
  const std::filesystem::path base_dir =
      output_dir.empty() ? std::filesystem::path(".")
                         : std::filesystem::path(output_dir);
  const std::filesystem::path diag_dir = base_dir / "tau_diag";
  std::error_code mkdir_error;
  std::filesystem::create_directories(diag_dir, mkdir_error);
  TENRYU_ASSERT(!mkdir_error,
                "TENRYU_TRACE_TAU_DIAG failed to create output directory");

  const std::string stem = std::string(trace_kind) + "_step" +
                           std::to_string(step) + "_beam" +
                           std::to_string(beam_index);
  const std::filesystem::path binary_path = diag_dir / (stem + ".bin");
  std::ofstream binary(binary_path, std::ios::binary | std::ios::trunc);
  TENRYU_ASSERT(binary,
                "TENRYU_TRACE_TAU_DIAG failed to open binary output");
  binary.write(reinterpret_cast<const char*>(tau_shell.data()),
               static_cast<std::streamsize>(tau_shell.size() * sizeof(double)));
  TENRYU_ASSERT(binary,
                "TENRYU_TRACE_TAU_DIAG failed to write binary output");

  const std::filesystem::path header_path = diag_dir / (stem + ".txt");
  std::ofstream header(header_path, std::ios::trunc);
  TENRYU_ASSERT(header,
                "TENRYU_TRACE_TAU_DIAG failed to open text header");
  header << "n_rays " << n_rays << '\n'
         << "n_intervals " << n_intervals << '\n'
         << "step " << step << '\n'
         << "beam " << beam_index << '\n';
  if (legs_only) {
    header << "legs_only\n";
  }
  TENRYU_ASSERT(header,
                "TENRYU_TRACE_TAU_DIAG failed to write text header");
}

void write_pabs_diag_dump(const std::string& output_dir,
                          const char* trace_kind,
                          const int step,
                          const std::size_t beam_index,
                          const std::vector<double>& pabs_per_ray) {
  const std::filesystem::path base_dir =
      output_dir.empty() ? std::filesystem::path(".")
                         : std::filesystem::path(output_dir);
  const std::filesystem::path diag_dir = base_dir / "tau_diag";
  std::error_code mkdir_error;
  std::filesystem::create_directories(diag_dir, mkdir_error);
  TENRYU_ASSERT(!mkdir_error,
                "TENRYU_TRACE_TAU_DIAG failed to create output directory");

  const std::string stem = std::string(trace_kind) + "_step" +
                           std::to_string(step) + "_beam" +
                           std::to_string(beam_index) + "_pabs.bin";
  const std::filesystem::path binary_path = diag_dir / stem;
  std::ofstream binary(binary_path, std::ios::binary | std::ios::trunc);
  TENRYU_ASSERT(binary,
                "TENRYU_TRACE_TAU_DIAG failed to open pabs binary output");
  binary.write(reinterpret_cast<const char*>(pabs_per_ray.data()),
               static_cast<std::streamsize>(pabs_per_ray.size() * sizeof(double)));
  TENRYU_ASSERT(binary,
                "TENRYU_TRACE_TAU_DIAG failed to write pabs binary output");
}

struct FoldKey {
  double p_beam = 0.0;
  double f_number = 0.0;
  double focus_lab_z = 0.0;
  double delta_lambda_nm = 0.0;
  int profile_m = 0;
  double profile_w0_cm = 0.0;
  std::string profile_model;
};

FoldKey make_fold_key(const Beam& beam, const double p_beam) {
  return FoldKey{p_beam, beam.f_number, beam.focus_lab_z, beam.delta_lambda_nm,
                 beam.profile_m, beam.profile_w0_cm, beam.profile_model};
}

bool fold_keys_equal(const FoldKey& lhs, const FoldKey& rhs) {
  return lhs.p_beam == rhs.p_beam && lhs.f_number == rhs.f_number &&
         lhs.focus_lab_z == rhs.focus_lab_z &&
         lhs.delta_lambda_nm == rhs.delta_lambda_nm && lhs.profile_m == rhs.profile_m &&
         lhs.profile_w0_cm == rhs.profile_w0_cm && lhs.profile_model == rhs.profile_model;
}

__global__ void per_warp_step_reduce_kernel(const int* d_step_count,
                                            int n_rays,
                                            int* d_warp_max,
                                            unsigned long long* d_warp_sum) {
  const int lane = static_cast<int>(threadIdx.x);
  const int idx = static_cast<int>(blockIdx.x) * 32 + lane;
  const bool active = (idx < n_rays);
  int step_max = active ? d_step_count[idx] : 0;
  unsigned long long step_sum =
      active ? static_cast<unsigned long long>(d_step_count[idx]) : 0ULL;

  for (int offset = 16; offset > 0; offset >>= 1) {
    const int other_max = __shfl_xor_sync(0xffffffffU, step_max, offset);
    step_max = (step_max > other_max) ? step_max : other_max;
    step_sum += __shfl_xor_sync(0xffffffffU, step_sum, offset);
  }

  if (lane == 0) {
    d_warp_max[blockIdx.x] = step_max;
    d_warp_sum[blockIdx.x] = step_sum;
  }
}

__global__ void replay_fold_tallies_kernel(void* tally_slab,
                                           const unsigned char* fold_snapshot,
                                           const int replay_count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  unsigned char* const tally = static_cast<unsigned char*>(tally_slab);
  double* const unabsorbed = reinterpret_cast<double*>(tally + 0);
  auto* const tail_count =
      reinterpret_cast<unsigned long long*>(tally + 8);
  double* const tail_power = reinterpret_cast<double*>(tally + 16);
  auto* const critical_hits =
      reinterpret_cast<unsigned long long*>(tally + 24);
  double* const ra_power = reinterpret_cast<double*>(tally + 32);
  const double fold_unabsorbed =
      *reinterpret_cast<const double*>(fold_snapshot + 0);
  const unsigned long long fold_tail_count =
      *reinterpret_cast<const unsigned long long*>(fold_snapshot + 8);
  const double fold_tail_power =
      *reinterpret_cast<const double*>(fold_snapshot + 16);
  const unsigned long long fold_critical_hits =
      *reinterpret_cast<const unsigned long long*>(fold_snapshot + 24);
  const double fold_ra_power =
      *reinterpret_cast<const double*>(fold_snapshot + 32);
  for (int k = 0; k < replay_count; ++k) {
    *unabsorbed += fold_unabsorbed;
    *ra_power += fold_ra_power;
    *tail_power += fold_tail_power;
    *tail_count += fold_tail_count;
    *critical_hits += fold_critical_hits;
  }
}

int locate_cell_1d(const std::vector<double>& edges, double r);

int nearest_shell_index(const std::vector<double>& shell_r,
                        const double radius) {
  TENRYU_ASSERT(!shell_r.empty(),
                "nearest_shell_index requires a non-empty shell grid");
  const auto upper =
      std::lower_bound(shell_r.begin(), shell_r.end(), radius);
  if (upper == shell_r.begin()) {
    return 0;
  }
  if (upper == shell_r.end()) {
    return static_cast<int>(shell_r.size() - 1U);
  }
  const auto lower = upper - 1;
  // Exact midpoint ties choose the lower shell.
  return (radius - *lower <= *upper - radius)
             ? static_cast<int>(lower - shell_r.begin())
             : static_cast<int>(upper - shell_r.begin());
}

void fill_cbet_viz_fields(core::State& state,
                          const CbetSolveResult& result,
                          const CbetWorkspace& workspace,
                          sector_adapter::S1Audit* audit) {
  TENRYU_ASSERT(result.dq_G > 0,
                "CBET visualization requires a positive dQ state dimension");
  TENRYU_ASSERT(
      result.dq_n_bins > 0 && result.dq_n_branches > 0,
      "CBET visualization requires positive branch/bin dimensions");
  const std::size_t n_cells =
      result.dq_host.size() / static_cast<std::size_t>(result.dq_G);
  TENRYU_ASSERT(
      result.dq_host.size() ==
          n_cells * static_cast<std::size_t>(result.dq_G) &&
          n_cells == state.rho.size() &&
          n_cells == static_cast<std::size_t>(workspace.n_cells),
      "CBET visualization dQ/cell dimensions are inconsistent");

  std::vector<double> cell_vol(n_cells, 0.0);
  std::vector<double> iaw_cell(n_cells, 0.0);
  cuda_check(cudaMemcpy(cell_vol.data(), workspace.cell_vol,
                        cell_vol.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "CBET visualization cell_vol D2H failed");
  if (result.dq_n_ports > 0) {
    cuda_check(cudaMemcpy(iaw_cell.data(), workspace.iaw_cell,
                          iaw_cell.size() * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "CBET visualization iaw_cell D2H failed");
  }
  state.cbet_gross_exchange.assign(n_cells, 0.0);
  state.cbet_net_to_inbound.assign(n_cells, 0.0);
  if (audit != nullptr) {
    audit->dq_abs_max = 0.0;
    audit->dq_abs_sum = 0.0;
    // No host L copy is available here without an additional D2H transfer.
    audit->L_ps_abs_max = -1.0;
  }
  constexpr double kClosureTiny = 1.0e-300;
  for (std::size_t cell = 0; cell < n_cells; ++cell) {
    double sum = 0.0;
    double sum_abs = 0.0;
    double sum_inbound = 0.0;
    const std::size_t row =
        cell * static_cast<std::size_t>(result.dq_G);
    for (int g = 0; g < result.dq_G; ++g) {
      const double dq = result.dq_host[row + static_cast<std::size_t>(g)];
      const double dq_abs = std::abs(dq);
      if (audit != nullptr) {
        audit->dq_abs_max = std::max(audit->dq_abs_max, dq_abs);
        audit->dq_abs_sum += dq_abs;
      }
      sum += dq;
      sum_abs += dq_abs;
      const int branch =
          (g % (result.dq_n_branches * result.dq_n_bins)) /
          result.dq_n_bins;
      if (branch == 0) {
        sum_inbound += dq;
      }
    }
    TENRYU_ASSERT(
        std::abs(sum + iaw_cell[cell]) <=
            1.0e-9 *
                std::max(sum_abs + std::abs(iaw_cell[cell]), kClosureTiny),
        "CBET final-iteration dQ+IAW violates per-cell action closure");
    TENRYU_ASSERT(
        std::isfinite(cell_vol[cell]) && cell_vol[cell] > 0.0,
        "CBET visualization requires a positive finite cell volume");
    state.cbet_gross_exchange[cell] =
        0.5 * sum_abs / cell_vol[cell];
    state.cbet_net_to_inbound[cell] =
        sum_inbound / cell_vol[cell];
  }
}

void fill_port_section_ray_map(
    core::State& state,
    const sector_ps::PhaseSpaceTable& table,
    const std::vector<double>& shell_r) {
  constexpr int kThetaBins = 64;
  constexpr double kPi = 3.14159265358979323846;
  constexpr double kErgPerSToW = 1.0e-7;
  const std::size_t n_shells = shell_r.size();
  state.ps_ray_map.assign(
      n_shells * static_cast<std::size_t>(kThetaBins) * 2U, 0.0);
  state.ps_ray_map_shell_r = shell_r;
  for (std::size_t shell = 0; shell < n_shells; ++shell) {
    for (int sheet = 0; sheet < 2; ++sheet) {
      std::array<double, kThetaBins> weighted_intensity{};
      std::array<double, kThetaBins> power{};
      for (const sector_ps::CrossingView& crossing :
           sector_ps::crossings(table, static_cast<int>(shell), sheet)) {
        const double scaled =
            std::clamp(crossing.theta, 0.0, kPi) *
            static_cast<double>(kThetaBins) / kPi;
        const int bin = std::clamp(
            static_cast<int>(std::floor(scaled)), 0, kThetaBins - 1);
        const double intensity = crossing.P / crossing.area;
        weighted_intensity[static_cast<std::size_t>(bin)] +=
            crossing.P * intensity;
        power[static_cast<std::size_t>(bin)] += crossing.P;
      }
      for (int bin = 0; bin < kThetaBins; ++bin) {
        const std::size_t index =
            (shell * static_cast<std::size_t>(kThetaBins) +
             static_cast<std::size_t>(bin)) *
                2U +
            static_cast<std::size_t>(sheet);
        if (power[static_cast<std::size_t>(bin)] > 0.0) {
          state.ps_ray_map[index] =
              weighted_intensity[static_cast<std::size_t>(bin)] /
              power[static_cast<std::size_t>(bin)] * kErgPerSToW;
        }
      }
    }
  }
}

void fill_port_section_sky_map(
    core::State& state,
    const port_section::CommonWaveDriveGridOutput& grid) {
  constexpr double kErgPerSToW = 1.0e-7;
  state.ps_sky_mu = grid.mu;
  state.ps_sky_phi = grid.phi;
  state.ps_sky_I_tot.resize(grid.I_tot.size());
  state.ps_sky_I_cw.resize(grid.I_cw.size());
  state.ps_sky_n_sigma = grid.n_sigma;
  for (std::size_t index = 0; index < grid.I_tot.size(); ++index) {
    state.ps_sky_I_tot[index] = grid.I_tot[index] * kErgPerSToW;
    state.ps_sky_I_cw[index] = grid.I_cw[index] * kErgPerSToW;
  }
}

void fill_port_section_outgoing_power(core::State& state,
                                      const CbetWorkspace& workspace) {
  const int n_rays = workspace.n_rays_total;
  std::vector<std::int32_t> rec_count(static_cast<std::size_t>(n_rays));
  std::vector<std::int64_t> ray_rec_offset(
      static_cast<std::size_t>(n_rays) + 1U);
  cuda_check(cudaMemcpy(rec_count.data(), workspace.rec_count,
                        rec_count.size() * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost),
             "port_section outgoing rec_count D2H failed");
  cuda_check(cudaMemcpy(ray_rec_offset.data(), workspace.ray_rec_offset,
                        ray_rec_offset.size() * sizeof(std::int64_t),
                        cudaMemcpyDeviceToHost),
             "port_section outgoing ray_rec_offset D2H failed");

  const std::size_t records_per_port =
      static_cast<std::size_t>(n_rays) *
      static_cast<std::size_t>(workspace.cap_per_ray);
  std::vector<double> rec_w_port(records_per_port, 0.0);
  state.ps_port_outgoing_power.assign(
      static_cast<std::size_t>(workspace.n_ports), 0.0);
  for (int port = 0; port < workspace.n_ports; ++port) {
    cuda_check(
        cudaMemcpy(
            rec_w_port.data(),
            workspace.rec_w_ps +
                static_cast<std::size_t>(port) * records_per_port,
            rec_w_port.size() * sizeof(double), cudaMemcpyDeviceToHost),
        "port_section final rec_w_ps D2H failed");
    double sum = 0.0;
    for (int ray = 0; ray < n_rays; ++ray) {
      const std::int64_t offset_count =
          ray_rec_offset[static_cast<std::size_t>(ray) + 1U] -
          ray_rec_offset[static_cast<std::size_t>(ray)];
      TENRYU_ASSERT(
          offset_count ==
              static_cast<std::int64_t>(
                  rec_count[static_cast<std::size_t>(ray)]),
          "port_section outgoing record offset/count mismatch");
      const int count = rec_count[static_cast<std::size_t>(ray)];
      if (count <= 0) {
        continue;
      }
      const std::size_t final_record =
          static_cast<std::size_t>(ray) *
              static_cast<std::size_t>(workspace.cap_per_ray) +
          static_cast<std::size_t>(count - 1);
      sum += rec_w_port[final_record];
    }
    state.ps_port_outgoing_power[static_cast<std::size_t>(port)] = sum;
  }
}

template <typename CheckFn, typename MsFn>
void emit_per_ray_step_stats(const LaserMesh& lmesh,
                             const int n_rays,
                             const int step,
                             const std::size_t beam_index,
                             cudaStream_t stream,
                             CheckFn&& check,
                             MsFn&& ms,
                             double& transfer_ms) {
  const int n_warps = (n_rays + 31) / 32;
  int* const d_step_count = lmesh.scratch_per_ray_step_count;
  int* const d_sorted = lmesh.scratch_sorted_step_count;
  int* const d_warp_max = lmesh.scratch_per_warp_step_max;
  unsigned long long* const d_warp_sum = lmesh.scratch_per_warp_step_sum;

  thrust::copy(thrust::cuda::par.on(stream), d_step_count, d_step_count + n_rays, d_sorted);
  check(cudaGetLastError(), "laser_step thrust copy per-ray step counts failed");
  thrust::sort(thrust::cuda::par.on(stream), d_sorted, d_sorted + n_rays);
  check(cudaGetLastError(), "laser_step thrust sort per-ray step counts failed");

  per_warp_step_reduce_kernel<<<n_warps, 32, 0, stream>>>(
      d_step_count, n_rays, d_warp_max, d_warp_sum);
  check(cudaGetLastError(), "laser_step per_warp_step_reduce_kernel launch failed");
  check(cudaStreamSynchronize(stream),
        "laser_step stream synchronize failed after per-ray step stats device work");

  std::vector<int> h_warp_max(static_cast<std::size_t>(n_warps), 0);
  std::vector<unsigned long long> h_warp_sum(static_cast<std::size_t>(n_warps), 0ULL);
  int p50 = 0;
  int p90 = 0;
  const int p50_idx = (n_rays - 1) / 2;
  // Lower-index nearest-rank percentile rule on the ascending sorted counts.
  const int p90_idx = ((n_rays - 1) * 9) / 10;

  const auto t_stats_transfer_start = std::chrono::steady_clock::now();
  check(cudaMemcpyAsync(h_warp_max.data(), d_warp_max,
                        static_cast<std::size_t>(n_warps) * sizeof(int),
                        cudaMemcpyDeviceToHost, stream),
        "laser_step memcpy per-warp step max D2H failed");
  check(cudaMemcpyAsync(h_warp_sum.data(), d_warp_sum,
                        static_cast<std::size_t>(n_warps) * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost, stream),
        "laser_step memcpy per-warp step sum D2H failed");
  check(cudaMemcpyAsync(&p50, d_sorted + p50_idx, sizeof(int), cudaMemcpyDeviceToHost, stream),
        "laser_step memcpy p50 step count D2H failed");
  check(cudaMemcpyAsync(&p90, d_sorted + p90_idx, sizeof(int), cudaMemcpyDeviceToHost, stream),
        "laser_step memcpy p90 step count D2H failed");
  check(cudaStreamSynchronize(stream),
        "laser_step stream synchronize failed after per-ray step stats D2H");
  transfer_ms += ms(t_stats_transfer_start, std::chrono::steady_clock::now());

  unsigned long long total_steps = 0ULL;
  double warp_max_sum = 0.0;
  double warp_mean_sum = 0.0;
  int global_max = 0;
  for (int w = 0; w < n_warps; ++w) {
    const int active_lanes =
        (w == n_warps - 1) ? (n_rays - (n_warps - 1) * 32) : 32;
    const int warp_max = h_warp_max[static_cast<std::size_t>(w)];
    const unsigned long long warp_sum = h_warp_sum[static_cast<std::size_t>(w)];
    total_steps += warp_sum;
    global_max = std::max(global_max, warp_max);
    warp_max_sum += static_cast<double>(warp_max);
    warp_mean_sum += static_cast<double>(warp_sum) / static_cast<double>(active_lanes);
  }

  const double mean = static_cast<double>(total_steps) / static_cast<double>(n_rays);
  const double mean_per_warp_max = warp_max_sum / static_cast<double>(n_warps);
  const double mean_per_warp_mean = warp_mean_sum / static_cast<double>(n_warps);

  core::log_info("[laser_per_ray_steps] step=" + std::to_string(step) +
                 " beam=" + std::to_string(beam_index) +
                 " n_rays=" + std::to_string(n_rays) +
                 " max=" + std::to_string(global_max) +
                 " p90=" + std::to_string(p90) +
                 " p50=" + std::to_string(p50) +
                 " mean=" + std::to_string(mean) +
                 " mean_per_warp_max=" + std::to_string(mean_per_warp_max) +
                 " mean_per_warp_mean=" + std::to_string(mean_per_warp_mean));
}

struct PackedStepHistogram {
  std::size_t offset = 0;
  int step = 0;
  int beam_id = 0;
  int n_rays = 0;
};

struct PackedPerRayStepStats {
  std::size_t warp_max_offset = 0;
  std::size_t warp_sum_offset = 0;
  std::size_t p50_offset = 0;
  std::size_t p90_offset = 0;
  int n_rays = 0;
  int step = 0;
  std::size_t beam_index = 0;
};

std::size_t reserve_step_pack_slot(std::size_t& cursor,
                                   const std::size_t bytes,
                                   const std::size_t alignment,
                                   const std::size_t capacity) {
  cursor = (cursor + alignment - 1U) & ~(alignment - 1U);
  TENRYU_ASSERT(cursor <= capacity && bytes <= capacity - cursor,
                "laser_step scalar pack capacity exceeded");
  const std::size_t offset = cursor;
  cursor += bytes;
  return offset;
}

template <typename CheckFn>
PackedPerRayStepStats stage_per_ray_step_stats(
    const LaserMesh& lmesh,
    const int n_rays,
    const int step,
    const std::size_t beam_index,
    cudaStream_t stream,
    unsigned char* pack_device,
    std::size_t& pack_cursor,
    const std::size_t pack_capacity,
    CheckFn&& check) {
  const int n_warps = (n_rays + 31) / 32;
  int* const d_step_count = lmesh.scratch_per_ray_step_count;
  int* const d_sorted = lmesh.scratch_sorted_step_count;
  int* const d_warp_max = lmesh.scratch_per_warp_step_max;
  unsigned long long* const d_warp_sum = lmesh.scratch_per_warp_step_sum;

  thrust::copy(thrust::cuda::par.on(stream), d_step_count,
               d_step_count + n_rays, d_sorted);
  check(cudaGetLastError(), "laser_step thrust copy per-ray step counts failed");
  thrust::sort(thrust::cuda::par.on(stream), d_sorted, d_sorted + n_rays);
  check(cudaGetLastError(), "laser_step thrust sort per-ray step counts failed");
  per_warp_step_reduce_kernel<<<n_warps, 32, 0, stream>>>(
      d_step_count, n_rays, d_warp_max, d_warp_sum);
  check(cudaGetLastError(),
        "laser_step per_warp_step_reduce_kernel launch failed");

  PackedPerRayStepStats packed;
  packed.n_rays = n_rays;
  packed.step = step;
  packed.beam_index = beam_index;
  packed.warp_max_offset = reserve_step_pack_slot(
      pack_cursor, static_cast<std::size_t>(n_warps) * sizeof(int),
      alignof(int), pack_capacity);
  packed.warp_sum_offset = reserve_step_pack_slot(
      pack_cursor,
      static_cast<std::size_t>(n_warps) * sizeof(unsigned long long),
      alignof(unsigned long long), pack_capacity);
  packed.p50_offset = reserve_step_pack_slot(
      pack_cursor, sizeof(int), alignof(int), pack_capacity);
  packed.p90_offset = reserve_step_pack_slot(
      pack_cursor, sizeof(int), alignof(int), pack_capacity);

  check(cudaMemcpyAsync(pack_device + packed.warp_max_offset, d_warp_max,
                        static_cast<std::size_t>(n_warps) * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream),
        "laser_step pack per-warp step max D2D failed");
  check(cudaMemcpyAsync(pack_device + packed.warp_sum_offset, d_warp_sum,
                        static_cast<std::size_t>(n_warps) *
                            sizeof(unsigned long long),
                        cudaMemcpyDeviceToDevice, stream),
        "laser_step pack per-warp step sum D2D failed");
  const int p50_idx = (n_rays - 1) / 2;
  const int p90_idx = ((n_rays - 1) * 9) / 10;
  check(cudaMemcpyAsync(pack_device + packed.p50_offset,
                        d_sorted + p50_idx, sizeof(int),
                        cudaMemcpyDeviceToDevice, stream),
        "laser_step pack p50 step count D2D failed");
  check(cudaMemcpyAsync(pack_device + packed.p90_offset,
                        d_sorted + p90_idx, sizeof(int),
                        cudaMemcpyDeviceToDevice, stream),
        "laser_step pack p90 step count D2D failed");
  return packed;
}

void emit_packed_step_histogram(const unsigned char* pack_host,
                                const PackedStepHistogram& packed) {
  std::array<int, LaserMesh::kTraceStepHistSize> histogram{};
  std::memcpy(histogram.data(), pack_host + packed.offset,
              histogram.size() * sizeof(int));
  const double mean_steps =
      (packed.n_rays > 0)
          ? (static_cast<double>(histogram[0]) /
             static_cast<double>(packed.n_rays))
          : 0.0;
  core::log_info(
      "[laser_ray_stats] step=" + std::to_string(packed.step) +
      " beam=" + std::to_string(packed.beam_id) +
      " n_rays=" + std::to_string(packed.n_rays) +
      " total_steps=" + std::to_string(histogram[0]) +
      " mean=" + std::to_string(mean_steps) +
      " b100=" + std::to_string(histogram[1]) +
      " b1k=" + std::to_string(histogram[2]) +
      " b10k=" + std::to_string(histogram[3]) +
      " bmax=" + std::to_string(histogram[4]));
}

void emit_packed_per_ray_step_stat(const unsigned char* pack_host,
                                   const PackedPerRayStepStats& packed) {
  const int n_warps = (packed.n_rays + 31) / 32;
  const int* const warp_max =
      reinterpret_cast<const int*>(pack_host + packed.warp_max_offset);
  const unsigned long long* const warp_sum =
      reinterpret_cast<const unsigned long long*>(
          pack_host + packed.warp_sum_offset);
  int p50 = 0;
  int p90 = 0;
  std::memcpy(&p50, pack_host + packed.p50_offset, sizeof(int));
  std::memcpy(&p90, pack_host + packed.p90_offset, sizeof(int));

  unsigned long long total_steps = 0ULL;
  double warp_max_sum = 0.0;
  double warp_mean_sum = 0.0;
  int global_max = 0;
  for (int w = 0; w < n_warps; ++w) {
    const int active_lanes =
        (w == n_warps - 1)
            ? (packed.n_rays - (n_warps - 1) * 32)
            : 32;
    total_steps += warp_sum[w];
    global_max = std::max(global_max, warp_max[w]);
    warp_max_sum += static_cast<double>(warp_max[w]);
    warp_mean_sum += static_cast<double>(warp_sum[w]) /
                     static_cast<double>(active_lanes);
  }
  const double mean = static_cast<double>(total_steps) /
                      static_cast<double>(packed.n_rays);
  const double mean_per_warp_max =
      warp_max_sum / static_cast<double>(n_warps);
  const double mean_per_warp_mean =
      warp_mean_sum / static_cast<double>(n_warps);
  core::log_info(
      "[laser_per_ray_steps] step=" + std::to_string(packed.step) +
      " beam=" + std::to_string(packed.beam_index) +
      " n_rays=" + std::to_string(packed.n_rays) +
      " max=" + std::to_string(global_max) +
      " p90=" + std::to_string(p90) +
      " p50=" + std::to_string(p50) +
      " mean=" + std::to_string(mean) +
      " mean_per_warp_max=" + std::to_string(mean_per_warp_max) +
      " mean_per_warp_mean=" + std::to_string(mean_per_warp_mean));
}

struct LaserMeshDebugStats {
  double min = 0.0;
  double max = 0.0;
  double mean = 0.0;
};

LaserMeshDebugStats summarize_lasermesh_field(const std::vector<double>& values) {
  LaserMeshDebugStats stats;
  if (values.empty()) {
    return stats;
  }
  const auto minmax = std::minmax_element(values.begin(), values.end());
  stats.min = *minmax.first;
  stats.max = *minmax.second;
  stats.mean = std::accumulate(values.begin(), values.end(), 0.0) /
               static_cast<double>(values.size());
  return stats;
}

std::size_t count_lasermesh_nonpositive(const std::vector<double>& values) {
  return static_cast<std::size_t>(
      std::count_if(values.begin(), values.end(), [](const double x) { return x <= 0.0; }));
}

void append_lasermesh_field_stats(std::ostringstream& oss,
                                  const char* name,
                                  const std::vector<double>& values) {
  const LaserMeshDebugStats stats = summarize_lasermesh_field(values);
  oss << ' ' << name << "_min=" << stats.min << ' ' << name << "_max=" << stats.max << ' '
      << name << "_mean=" << stats.mean;
}

void emit_lasermesh_debug_dump(const LaserMesh& mesh,
                               const int step,
                               const int rank,
                               cudaStream_t stream) {
  const int n_nodes_total = mesh.n_nodes();
  if (n_nodes_total <= 0) {
    core::log_warning("[laser_lasermesh_debug] step=" + std::to_string(step) +
                      " rank=" + std::to_string(rank) + " n_nodes=0");
    return;
  }

  const std::size_t n_nodes = static_cast<std::size_t>(n_nodes_total);
  const std::size_t bytes = n_nodes * sizeof(double);
  std::vector<double> h_Te(n_nodes, 0.0);
  std::vector<double> h_Zbar(n_nodes, 0.0);
  std::vector<double> h_nhat_raw(n_nodes, 0.0);
  std::vector<double> h_nhat(n_nodes, 0.0);
  std::vector<double> h_smooth(n_nodes, 0.0);

  cuda_check(cudaStreamSynchronize(stream),
             "laser_step debug_dump_lasermesh stream synchronize failed");
  cuda_check(cudaMemcpy(h_Te.data(), mesh.T_e, bytes, cudaMemcpyDeviceToHost),
             "laser_step debug_dump_lasermesh T_e D2H failed");
  cuda_check(cudaMemcpy(h_Zbar.data(), mesh.Zbar, bytes, cudaMemcpyDeviceToHost),
             "laser_step debug_dump_lasermesh Zbar D2H failed");
  cuda_check(cudaMemcpy(h_nhat_raw.data(), mesh.n_e_hat_raw, bytes, cudaMemcpyDeviceToHost),
             "laser_step debug_dump_lasermesh n_hat_raw D2H failed");
  cuda_check(cudaMemcpy(h_nhat.data(), mesh.n_e_hat, bytes, cudaMemcpyDeviceToHost),
             "laser_step debug_dump_lasermesh n_hat D2H failed");
  cuda_check(cudaMemcpy(h_smooth.data(), mesh.smooth_kappa_factor, bytes,
                        cudaMemcpyDeviceToHost),
             "laser_step debug_dump_lasermesh smooth_kappa_factor D2H failed");

  std::ostringstream summary;
  summary << std::setprecision(17)
          << "[laser_lasermesh_debug] step=" << step << " rank=" << rank
          << " n_nodes=" << n_nodes_total << " n_nodes_r=" << mesh.n_nodes_r
          << " n_nodes_z=" << mesh.n_nodes_z;
  append_lasermesh_field_stats(summary, "Te", h_Te);
  append_lasermesh_field_stats(summary, "Zbar", h_Zbar);
  append_lasermesh_field_stats(summary, "n_hat_raw", h_nhat_raw);
  append_lasermesh_field_stats(summary, "n_hat", h_nhat);
  append_lasermesh_field_stats(summary, "smooth_kappa_factor", h_smooth);
  summary << " count_Zbar_le_0=" << count_lasermesh_nonpositive(h_Zbar)
          << " count_n_hat_raw_le_0=" << count_lasermesh_nonpositive(h_nhat_raw)
          << " count_smooth_kappa_factor_le_0=" << count_lasermesh_nonpositive(h_smooth);
  core::log_warning(summary.str());

  const int n_axis = std::min(mesh.n_nodes_z, n_nodes_total);
  for (int j = 0; j < n_axis; ++j) {
    const std::size_t idx = static_cast<std::size_t>(j);
    std::ostringstream axis;
    axis << std::setprecision(17) << "[laser_lasermesh_debug_axis] step=" << step
         << " rank=" << rank << " i=0 j=" << j << " Te=" << h_Te[idx]
         << " Zbar=" << h_Zbar[idx] << " n_hat_raw=" << h_nhat_raw[idx]
         << " smooth_kappa_factor=" << h_smooth[idx];
    core::log_warning(axis.str());
  }
}

std::vector<double> copy_lm_deposit_to_host(const LaserMesh& lmesh, cudaStream_t stream) {
  std::vector<double> dep(static_cast<std::size_t>(lmesh.n_nodes()), 0.0);
  cuda_check(cudaMemcpyAsync(dep.data(), lmesh.deposit, dep.size() * sizeof(double),
                             cudaMemcpyDeviceToHost, stream),
             "laser_step memcpyAsync LaserMesh deposit D2H failed");
  cuda_check(cudaStreamSynchronize(stream), "laser_step stream synchronize failed");
  return dep;
}

std::vector<double> copy_cell_deposit_to_host(const core::CellField1D& field, cudaStream_t stream) {
  std::vector<double> dep(field.size(), 0.0);
  cuda_check(cudaMemcpyAsync(dep.data(), field.data(), dep.size() * sizeof(double),
                             cudaMemcpyDeviceToHost, stream),
             "laser_step memcpyAsync 1D deposit D2H failed");
  cuda_check(cudaStreamSynchronize(stream), "laser_step stream synchronize failed");
  return dep;
}

void copy_lm_nodes_to_host(const LaserMesh& lmesh,
                           std::vector<double>& node_R,
                           std::vector<double>& node_Z) {
  node_R.assign(static_cast<std::size_t>(lmesh.n_nodes_r), 0.0);
  node_Z.assign(static_cast<std::size_t>(lmesh.n_nodes_z), 0.0);
  cuda_check(cudaMemcpy(node_R.data(), lmesh.node_R, node_R.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "laser_step memcpy node_R D2H failed");
  cuda_check(cudaMemcpy(node_Z.data(), lmesh.node_Z, node_Z.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "laser_step memcpy node_Z D2H failed");
}

std::vector<double> project_lm_power_to_hydro_2d(const core::State& state,
                                                 const LaserMesh& lmesh,
                                                 const std::vector<double>& dep_lm,
                                                 const std::vector<double>& node_R,
                                                 const std::vector<double>& node_Z) {
  std::vector<double> dep_cell(static_cast<std::size_t>(state.mesh.topo.n_cells), 0.0);
  const double r_min = node_R.front();
  const double r_max = node_R.back();
  const double z_min = node_Z.front();
  const double z_max = node_Z.back();
  const double scale = std::max({1.0, std::abs(r_max), std::abs(z_min), std::abs(z_max)});
  const double tol = 1.0e-12 * scale;
  for (int c = 0; c < state.mesh.topo.n_cells; ++c) {
    const double rc = state.mesh.cell_centroid_r[static_cast<std::size_t>(c)];
    const double zc = state.mesh.cell_centroid_z[static_cast<std::size_t>(c)];
    if (rc < r_min - tol || rc > r_max + tol || zc < z_min - tol || zc > z_max + tol) {
      dep_cell[static_cast<std::size_t>(c)] = 0.0;
      continue;
    }
    const BilinearCell cell = BilinearInterp::locate_cell(
        node_R.data(), node_Z.data(), lmesh.n_nodes_r, lmesh.n_nodes_z, rc, zc);
    const BilinearWeights w = BilinearInterp::compute_weights(cell.xi, cell.eta);
    dep_cell[static_cast<std::size_t>(c)] =
        BilinearInterp::interpolate(dep_lm.data(), lmesh.n_nodes_z, cell, w);
  }
  return dep_cell;
}

int locate_cell_1d(const std::vector<double>& edges, const double r) {
  const int n_cells = static_cast<int>(edges.size()) - 1;
  if (n_cells <= 0) {
    return -1;
  }
  if (r <= edges.front()) {
    return 0;
  }
  if (r >= edges.back()) {
    return n_cells - 1;
  }
  auto it = std::upper_bound(edges.begin(), edges.end(), r);
  const int idx = static_cast<int>(std::distance(edges.begin(), it)) - 1;
  return std::max(0, std::min(n_cells - 1, idx));
}

// Option C layout: arrays are global-size on every rank, so ownership is a
// pure global-index range check (see PartitionInfo in parallel/partition.hpp).
bool is_owned_cell_1d(const int c,
                      const int n_cells,
                      const core::State& state,
                      const parallel::PartitionInfo& part) {
  (void)n_cells;
  (void)state;
  if (part.n_ranks <= 1) {
    return true;
  }
  const int i_global_begin = part.local_cell_range[0][0];
  const int i_global_end = part.local_cell_range[0][1];
  return c >= i_global_begin && c < i_global_end;
}

bool is_owned_cell_2d(const int c,
                      const int n_cells,
                      const core::State& state,
                      const parallel::PartitionInfo& part) {
  (void)n_cells;
  if (part.n_ranks <= 1) {
    return true;
  }
  const int nz_global = std::max(state.mesh.topo.nz, 1);
  return part.owns_cell(c / nz_global, c % nz_global);
}

void mask_non_owned_skip_deposit(core::State& state, const parallel::PartitionInfo& part) {
  if (part.n_ranks <= 1 || state.laser_dep.empty()) {
    return;
  }

  std::vector<double> dep(state.laser_dep.size(), 0.0);
  state.laser_dep.copy_to_host(dep.data());
  const int n_cells = static_cast<int>(dep.size());
  for (int c = 0; c < n_cells; ++c) {
    bool owned = true;
    if (state.mesh.dim == 1) {
      owned = is_owned_cell_1d(c, n_cells, state, part);
    } else if (state.mesh.dim == 2) {
      owned = is_owned_cell_2d(c, n_cells, state, part);
    }
    if (!owned) {
      dep[static_cast<std::size_t>(c)] = 0.0;
    }
  }
  state.laser_dep.copy_from_host(dep.data());
}

int locate_interval_centers(const std::vector<double>& centers, const double x) {
  const int n = static_cast<int>(centers.size());
  if (n <= 1) {
    return 0;
  }
  if (n == 2) {
    return 0;
  }
  if (x <= centers.front()) {
    return 0;
  }
  if (x >= centers.back()) {
    return n - 2;
  }
  auto it = std::upper_bound(centers.begin(), centers.end(), x);
  const int idx = static_cast<int>(std::distance(centers.begin(), it)) - 1;
  return std::max(0, std::min(n - 2, idx));
}

struct HydroCellLocator2D {
  int nr_h = 0;
  int nz_h = 0;
  const std::vector<double>* hydro_r = nullptr;
  const std::vector<double>* hydro_z = nullptr;
  std::vector<double> r_centers;
  std::vector<double> z_centers;
  bool tensor_like_centroids = true;

  explicit HydroCellLocator2D(const core::State& state)
      : nr_h(state.mesh.topo.nr),
        nz_h(state.mesh.topo.nz),
        hydro_r(&state.mesh.cell_centroid_r),
        hydro_z(&state.mesh.cell_centroid_z) {
    if (nr_h <= 0 || nz_h <= 0 ||
        hydro_r->size() != static_cast<std::size_t>(nr_h * nz_h) ||
        hydro_z->size() != static_cast<std::size_t>(nr_h * nz_h)) {
      nr_h = 0;
      nz_h = 0;
      return;
    }
    r_centers.assign(static_cast<std::size_t>(nr_h), 0.0);
    z_centers.assign(static_cast<std::size_t>(nz_h), 0.0);
    for (int i = 0; i < nr_h; ++i) {
      long double r_sum = 0.0L;
      for (int j = 0; j < nz_h; ++j) {
        const int c = i * nz_h + j;
        r_sum += static_cast<long double>((*hydro_r)[static_cast<std::size_t>(c)]);
      }
      r_centers[static_cast<std::size_t>(i)] =
          static_cast<double>(r_sum / std::max(1, nz_h));
    }
    for (int j = 0; j < nz_h; ++j) {
      long double z_sum = 0.0L;
      for (int i = 0; i < nr_h; ++i) {
        const int c = i * nz_h + j;
        z_sum += static_cast<long double>((*hydro_z)[static_cast<std::size_t>(c)]);
      }
      z_centers[static_cast<std::size_t>(j)] =
          static_cast<double>(z_sum / std::max(1, nr_h));
    }

    const double r_scale = std::max(1.0, std::abs(r_centers.back() - r_centers.front()));
    const double z_scale = std::max(1.0, std::abs(z_centers.back() - z_centers.front()));
    const double tol_r = 1.0e-10 * r_scale;
    const double tol_z = 1.0e-10 * z_scale;
    tensor_like_centroids = true;
    for (int i = 0; i < nr_h && tensor_like_centroids; ++i) {
      for (int j = 0; j < nz_h; ++j) {
        const int c = i * nz_h + j;
        if (std::abs((*hydro_r)[static_cast<std::size_t>(c)] -
                     r_centers[static_cast<std::size_t>(i)]) > tol_r ||
            std::abs((*hydro_z)[static_cast<std::size_t>(c)] -
                     z_centers[static_cast<std::size_t>(j)]) > tol_z) {
          tensor_like_centroids = false;
          break;
        }
      }
    }
  }

  [[nodiscard]] int locate(const double R, const double Z) const {
    if (nr_h <= 0 || nz_h <= 0) {
      return -1;
    }
    if (nr_h == 1 || nz_h == 1) {
      return 0;
    }
    const int i = locate_interval_centers(r_centers, R);
    const int j = locate_interval_centers(z_centers, Z);
    const int c00 = i * nz_h + j;
    const int c10 = (i + 1) * nz_h + j;
    const int c01 = i * nz_h + (j + 1);
    const int c11 = (i + 1) * nz_h + (j + 1);
    if (tensor_like_centroids) {
      return c00;
    }

    auto dist2 = [&](const int c) {
      const double dr = R - (*hydro_r)[static_cast<std::size_t>(c)];
      const double dz = Z - (*hydro_z)[static_cast<std::size_t>(c)];
      return dr * dr + dz * dz;
    };
    int best = c00;
    double best_d2 = dist2(c00);
    const int candidates[3] = {c10, c01, c11};
    for (const int c : candidates) {
      const double d2 = dist2(c);
      if (d2 < best_d2) {
        best_d2 = d2;
        best = c;
      }
    }
    return best;
  }
};

void capture_ray_output_1d(const RayArray1D& rays,
                           const int n_copy,
                           const int stride,
                           const int beam_id,
                           std::vector<RayOutputData>* ray_output,
                           cudaStream_t stream) {
  if (ray_output == nullptr || n_copy <= 0 || rays.n_rays <= 0 || stride <= 0) {
    return;
  }
  RayOutputData data;
  data.beam_id = beam_id;
  data.rays_2d.resize(static_cast<std::size_t>(n_copy));
  const int n_total = rays.n_rays;
  std::vector<double> R0(static_cast<std::size_t>(n_total), 0.0);
  std::vector<double> Z0(static_cast<std::size_t>(n_total), 0.0);
  std::vector<double> vR0(static_cast<std::size_t>(n_total), 0.0);
  std::vector<double> vZ0(static_cast<std::size_t>(n_total), 0.0);
  std::vector<double> power0(static_cast<std::size_t>(n_total), 0.0);
  const std::size_t bytes = static_cast<std::size_t>(n_total) * sizeof(double);
  cuda_check(cudaMemcpyAsync(R0.data(), rays.R0, bytes, cudaMemcpyDeviceToHost, stream),
             "laser_step memcpyAsync ray R0 D2H failed");
  cuda_check(cudaMemcpyAsync(Z0.data(), rays.Z0, bytes, cudaMemcpyDeviceToHost, stream),
             "laser_step memcpyAsync ray Z0 D2H failed");
  cuda_check(cudaMemcpyAsync(vR0.data(), rays.vR0, bytes, cudaMemcpyDeviceToHost, stream),
             "laser_step memcpyAsync ray vR0 D2H failed");
  cuda_check(cudaMemcpyAsync(vZ0.data(), rays.vZ0, bytes, cudaMemcpyDeviceToHost, stream),
             "laser_step memcpyAsync ray vZ0 D2H failed");
  cuda_check(cudaMemcpyAsync(power0.data(), rays.power0, bytes, cudaMemcpyDeviceToHost, stream),
             "laser_step memcpyAsync ray power0 D2H failed");
  cuda_check(cudaStreamSynchronize(stream), "laser_step stream synchronize failed");
  for (int i = 0; i < n_copy; ++i) {
    const int src = i * stride;
    if (src >= n_total) {
      break;
    }
    auto& rec = data.rays_2d[static_cast<std::size_t>(i)];
    rec.R0 = R0[static_cast<std::size_t>(src)];
    rec.Z0 = Z0[static_cast<std::size_t>(src)];
    rec.vR0 = vR0[static_cast<std::size_t>(src)];
    rec.vZ0 = vZ0[static_cast<std::size_t>(src)];
    rec.power0 = power0[static_cast<std::size_t>(src)];
  }
  ray_output->push_back(std::move(data));
}

void capture_ray_output_3d(const RayArray2D& rays,
                           const int n_copy,
                           const int stride,
                           const int beam_id,
                           std::vector<RayOutputData>* ray_output,
                           cudaStream_t stream) {
  if (ray_output == nullptr || n_copy <= 0 || rays.n_rays <= 0 || stride <= 0) {
    return;
  }
  RayOutputData data;
  data.beam_id = beam_id;
  data.rays_3d.resize(static_cast<std::size_t>(n_copy));
  const int n_total = rays.n_rays;
  std::vector<double> x0(static_cast<std::size_t>(n_total), 0.0);
  std::vector<double> y0(static_cast<std::size_t>(n_total), 0.0);
  std::vector<double> z0(static_cast<std::size_t>(n_total), 0.0);
  std::vector<double> vx0(static_cast<std::size_t>(n_total), 0.0);
  std::vector<double> vy0(static_cast<std::size_t>(n_total), 0.0);
  std::vector<double> vz0(static_cast<std::size_t>(n_total), 0.0);
  std::vector<double> power0(static_cast<std::size_t>(n_total), 0.0);
  const std::size_t bytes = static_cast<std::size_t>(n_total) * sizeof(double);
  cuda_check(cudaMemcpyAsync(x0.data(), rays.x0, bytes, cudaMemcpyDeviceToHost, stream),
             "laser_step memcpyAsync ray x0 D2H failed");
  cuda_check(cudaMemcpyAsync(y0.data(), rays.y0, bytes, cudaMemcpyDeviceToHost, stream),
             "laser_step memcpyAsync ray y0 D2H failed");
  cuda_check(cudaMemcpyAsync(z0.data(), rays.z0, bytes, cudaMemcpyDeviceToHost, stream),
             "laser_step memcpyAsync ray z0 D2H failed");
  cuda_check(cudaMemcpyAsync(vx0.data(), rays.vx0, bytes, cudaMemcpyDeviceToHost, stream),
             "laser_step memcpyAsync ray vx0 D2H failed");
  cuda_check(cudaMemcpyAsync(vy0.data(), rays.vy0, bytes, cudaMemcpyDeviceToHost, stream),
             "laser_step memcpyAsync ray vy0 D2H failed");
  cuda_check(cudaMemcpyAsync(vz0.data(), rays.vz0, bytes, cudaMemcpyDeviceToHost, stream),
             "laser_step memcpyAsync ray vz0 D2H failed");
  cuda_check(cudaMemcpyAsync(power0.data(), rays.power0, bytes, cudaMemcpyDeviceToHost, stream),
             "laser_step memcpyAsync ray power0 D2H failed");
  cuda_check(cudaStreamSynchronize(stream), "laser_step stream synchronize failed");
  for (int i = 0; i < n_copy; ++i) {
    const int src = i * stride;
    if (src >= n_total) {
      break;
    }
    auto& rec = data.rays_3d[static_cast<std::size_t>(i)];
    rec.x0 = x0[static_cast<std::size_t>(src)];
    rec.y0 = y0[static_cast<std::size_t>(src)];
    rec.z0 = z0[static_cast<std::size_t>(src)];
    rec.vx0 = vx0[static_cast<std::size_t>(src)];
    rec.vy0 = vy0[static_cast<std::size_t>(src)];
    rec.vz0 = vz0[static_cast<std::size_t>(src)];
    rec.power0 = power0[static_cast<std::size_t>(src)];
  }
  ray_output->push_back(std::move(data));
}

void accumulate_ray_density_1d(const RayArray1D& rays,
                               const std::vector<double>& r_edges,
                               std::vector<double>& ray_counts,
                               cudaStream_t stream) {
  if (rays.n_rays <= 0 || ray_counts.empty() || r_edges.size() < 2) {
    return;
  }
  std::vector<double> R0(static_cast<std::size_t>(rays.n_rays), 0.0);
  std::vector<double> Z0(static_cast<std::size_t>(rays.n_rays), 0.0);
  const std::size_t bytes = static_cast<std::size_t>(rays.n_rays) * sizeof(double);
  cuda_check(cudaMemcpyAsync(R0.data(), rays.R0, bytes, cudaMemcpyDeviceToHost, stream),
             "laser_step memcpyAsync ray-density R0 D2H failed");
  cuda_check(cudaMemcpyAsync(Z0.data(), rays.Z0, bytes, cudaMemcpyDeviceToHost, stream),
             "laser_step memcpyAsync ray-density Z0 D2H failed");
  cuda_check(cudaStreamSynchronize(stream), "laser_step stream synchronize failed");
  for (int i = 0; i < rays.n_rays; ++i) {
    const double r = std::sqrt(R0[static_cast<std::size_t>(i)] * R0[static_cast<std::size_t>(i)] +
                               Z0[static_cast<std::size_t>(i)] * Z0[static_cast<std::size_t>(i)]);
    const int c = locate_cell_1d(r_edges, r);
    if (c >= 0 && c < static_cast<int>(ray_counts.size())) {
      ray_counts[static_cast<std::size_t>(c)] += 1.0;
    }
  }
}

void accumulate_ray_density_2d(const RayArray2D& rays,
                               const HydroCellLocator2D& locator,
                               std::vector<double>& ray_counts,
                               cudaStream_t stream) {
  if (rays.n_rays <= 0 || ray_counts.empty() || locator.nr_h <= 0 || locator.nz_h <= 0) {
    return;
  }
  std::vector<double> x0(static_cast<std::size_t>(rays.n_rays), 0.0);
  std::vector<double> y0(static_cast<std::size_t>(rays.n_rays), 0.0);
  std::vector<double> z0(static_cast<std::size_t>(rays.n_rays), 0.0);
  const std::size_t bytes = static_cast<std::size_t>(rays.n_rays) * sizeof(double);
  cuda_check(cudaMemcpyAsync(x0.data(), rays.x0, bytes, cudaMemcpyDeviceToHost, stream),
             "laser_step memcpyAsync ray-density x0 D2H failed");
  cuda_check(cudaMemcpyAsync(y0.data(), rays.y0, bytes, cudaMemcpyDeviceToHost, stream),
             "laser_step memcpyAsync ray-density y0 D2H failed");
  cuda_check(cudaMemcpyAsync(z0.data(), rays.z0, bytes, cudaMemcpyDeviceToHost, stream),
             "laser_step memcpyAsync ray-density z0 D2H failed");
  cuda_check(cudaStreamSynchronize(stream), "laser_step stream synchronize failed");
  for (int i = 0; i < rays.n_rays; ++i) {
    const double R = std::hypot(x0[static_cast<std::size_t>(i)], y0[static_cast<std::size_t>(i)]);
    const double Z = z0[static_cast<std::size_t>(i)];
    const int c = locator.locate(R, Z);
    if (c >= 0 && c < static_cast<int>(ray_counts.size())) {
      ray_counts[static_cast<std::size_t>(c)] += 1.0;
    }
  }
}

void finalize_ray_density(core::State& state, const std::vector<double>& ray_counts) {
  if (state.ray_density.size() != ray_counts.size()) {
    return;
  }
  std::vector<double> vol(state.vol.size(), 0.0);
  state.vol.copy_to_host(vol.data());
  std::vector<double> density(ray_counts.size(), 0.0);
  for (std::size_t c = 0; c < ray_counts.size(); ++c) {
    const double v = (c < vol.size()) ? std::max(vol[c], 1.0e-30) : 1.0;
    density[c] = ray_counts[c] / v;
  }
  state.ray_density.copy_from_host(density.data());
}

double sum_field_energy(const core::CellField1D& field) {
  std::vector<double> host(field.size(), 0.0);
  field.copy_to_host(host.data());
  long double sum = 0.0L;
  for (const double v : host) {
    sum += static_cast<long double>(v);
  }
  return static_cast<double>(sum);
}

void log_laser_flags(const core::DeviceErrorFlags& flags) {
  if (flags.infinite_loop != 0) {
    core::log_warning("Laser ray trace hit MAX_RAY_STEPS guard");
  }
  if (flags.nan_particle != 0) {
    core::log_warning("Laser ray trace encountered non-finite ray state");
  }
  if (flags.invalid_cell != 0) {
    core::log_warning(
        "Laser ray trace encountered invalid interpolation state; affected rays were "
        "treated as unabsorbed");
  }
  if (flags.nan_particle != 0) {
    TENRYU_ASSERT(
        false, "Laser ray trace: fatal device error flag set (nan_particle)");
  }
}

void synchronize_1d_deposit_allgatherv(std::vector<double>& total_dep_lm,
                                       const parallel::Reduction& reduction,
                                       const int n_ranks) {
  if (n_ranks <= 1 || total_dep_lm.empty()) {
    return;
  }
  TENRYU_ASSERT(total_dep_lm.size() <=
                    static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "laser_step 1D allgatherv sendcount exceeds MPI int range");
  TENRYU_ASSERT(
      total_dep_lm.size() <=
          (std::numeric_limits<std::size_t>::max() /
           static_cast<std::size_t>(std::max(1, n_ranks))),
      "laser_step 1D allgatherv receive buffer size overflow");

  const int sendcount = static_cast<int>(total_dep_lm.size());
  TENRYU_ASSERT(sendcount <=
                    (std::numeric_limits<int>::max() / std::max(1, n_ranks)),
                "laser_step 1D allgatherv displacements exceed MPI int range");
  std::vector<int> recvcounts(static_cast<std::size_t>(n_ranks), sendcount);
  std::vector<int> displs(static_cast<std::size_t>(n_ranks), 0);
  for (int r = 1; r < n_ranks; ++r) {
    displs[static_cast<std::size_t>(r)] =
        displs[static_cast<std::size_t>(r - 1)] +
        recvcounts[static_cast<std::size_t>(r - 1)];
  }
  std::vector<double> gathered(
      total_dep_lm.size() * static_cast<std::size_t>(n_ranks), 0.0);
  reduction.allgatherv(total_dep_lm.data(), sendcount, gathered.data(),
                       recvcounts.data(), displs.data());

  std::fill(total_dep_lm.begin(), total_dep_lm.end(), 0.0);
  for (int r = 0; r < n_ranks; ++r) {
    const std::size_t base = static_cast<std::size_t>(r) * total_dep_lm.size();
    for (std::size_t i = 0; i < total_dep_lm.size(); ++i) {
      total_dep_lm[i] += gathered[base + i];
    }
  }
}

void build_port_section_diagnostics(
    LaserMesh& lmesh,
    const Beam& beam,
    const HydroMirror1D& hydro,
    const CbetWorkspace& cbet_ws,
    const int n_rays,
    const bool verbose) {
  auto* const port_state =
      static_cast<PortSectionState*>(lmesh.port_section_state.get());
  TENRYU_ASSERT(port_state != nullptr,
                "port_section diagnostics require initialized host state");
  if (n_rays <= 0) {
    std::memset(&port_state->audit, 0, sizeof(port_state->audit));
    return;
  }

  const int cap_per_ray = cbet_ws.cap_per_ray;
  const std::size_t stripe_size =
      static_cast<std::size_t>(n_rays) *
      static_cast<std::size_t>(cap_per_ray);
  std::vector<std::int32_t> rec_count(static_cast<std::size_t>(n_rays));
  std::vector<std::int32_t> stripe_cell(stripe_size);
  std::vector<float> stripe_mu(stripe_size);
  std::vector<double> stripe_ds(stripe_size);
  std::vector<double> stripe_S(stripe_size);
  std::vector<double> ray_P0(static_cast<std::size_t>(n_rays));
  std::vector<std::int32_t> ray_group_base(
      static_cast<std::size_t>(n_rays));

  cuda_check(cudaMemcpy(rec_count.data(), cbet_ws.rec_count,
                        rec_count.size() * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost),
             "port_section rec_count D2H failed");
  cuda_check(cudaMemcpy(stripe_cell.data(), cbet_ws.rec_cell,
                        stripe_cell.size() * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost),
             "port_section rec_cell D2H failed");
  cuda_check(cudaMemcpy(stripe_mu.data(), cbet_ws.rec_mu,
                        stripe_mu.size() * sizeof(float),
                        cudaMemcpyDeviceToHost),
             "port_section rec_mu D2H failed");
  cuda_check(cudaMemcpy(stripe_ds.data(), cbet_ws.rec_ds,
                        stripe_ds.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "port_section rec_ds D2H failed");
  cuda_check(cudaMemcpy(stripe_S.data(), cbet_ws.rec_S,
                        stripe_S.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "port_section rec_S D2H failed");
  cuda_check(cudaMemcpy(ray_P0.data(), cbet_ws.ray_P0,
                        ray_P0.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "port_section ray_P0 D2H failed");
  cuda_check(cudaMemcpy(ray_group_base.data(), cbet_ws.ray_group_base,
                        ray_group_base.size() * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost),
             "port_section ray_group_base D2H failed");

  std::vector<std::int64_t> ray_rec_offset(
      static_cast<std::size_t>(n_rays) + 1, 0);
  for (int ray = 0; ray < n_rays; ++ray) {
    const int count =
        std::clamp(static_cast<int>(rec_count[static_cast<std::size_t>(ray)]),
                   0, cap_per_ray);
    ray_rec_offset[static_cast<std::size_t>(ray) + 1] =
        ray_rec_offset[static_cast<std::size_t>(ray)] + count;
  }
  const std::size_t n_records =
      static_cast<std::size_t>(ray_rec_offset.back());
  std::vector<std::int32_t> rec_cell(n_records);
  std::vector<float> rec_mu(n_records);
  std::vector<double> rec_ds(n_records);
  std::vector<double> rec_S(n_records);
  for (int ray = 0; ray < n_rays; ++ray) {
    const std::size_t source =
        static_cast<std::size_t>(ray) *
        static_cast<std::size_t>(cap_per_ray);
    const std::size_t destination = static_cast<std::size_t>(
        ray_rec_offset[static_cast<std::size_t>(ray)]);
    const std::size_t count = static_cast<std::size_t>(
        ray_rec_offset[static_cast<std::size_t>(ray) + 1] -
        ray_rec_offset[static_cast<std::size_t>(ray)]);
    if (count == 0) {
      continue;
    }
    std::copy_n(stripe_cell.data() + source, count,
                rec_cell.data() + destination);
    std::copy_n(stripe_mu.data() + source, count,
                rec_mu.data() + destination);
    std::copy_n(stripe_ds.data() + source, count,
                rec_ds.data() + destination);
    std::copy_n(stripe_S.data() + source, count,
                rec_S.data() + destination);
  }

  const int n_cells = static_cast<int>(hydro.rho.size());
  std::vector<double> cell_r_center(static_cast<std::size_t>(n_cells));
  std::vector<double> cell_eps(static_cast<std::size_t>(n_cells));
  for (int cell = 0; cell < n_cells; ++cell) {
    const std::size_t index = static_cast<std::size_t>(cell);
    cell_r_center[index] =
        0.5 * (hydro.r_edges[index] + hydro.r_edges[index + 1]);
    const double zbar = std::max(hydro.zbar[index], 0.0);
    const double A_eff = std::max(hydro.A_eff[index], 1.0e-30);
    const double ne =
        hydro.rho[index] * zbar /
        (A_eff * core::constants::proton_mass);
    cell_eps[index] = std::max(0.0, 1.0 - ne / lmesh.n_crit);
  }

  std::vector<double> impact_parameter(static_cast<std::size_t>(n_rays));
  const double R_beam =
      std::abs(lmesh.Z_max - beam.focus_lab_z) /
      (2.0 * std::max(beam.f_number, 1.0e-12));
  const double dR = R_beam / static_cast<double>(n_rays);
  for (int ray = 0; ray < n_rays; ++ray) {
    impact_parameter[static_cast<std::size_t>(ray)] =
        dR * (static_cast<double>(ray) + 0.5);
  }

  const IncidentAttenuationContext attenuation_context{&rec_S};
  sector_adapter::AdapterInput adapter_input{};
  adapter_input.n_rays = n_rays;
  adapter_input.ray_rec_offset = ray_rec_offset.data();
  adapter_input.rec_cell = rec_cell.data();
  adapter_input.rec_mu = rec_mu.data();
  adapter_input.rec_ds = rec_ds.data();
  adapter_input.ray_P0 = ray_P0.data();
  adapter_input.ray_impact_parameter = impact_parameter.data();
  adapter_input.cell_r_center = cell_r_center.data();
  adapter_input.r_edges = hydro.r_edges.data();
  adapter_input.cell_eps = cell_eps.data();
  adapter_input.n_cells = n_cells;
  adapter_input.attenuate = apply_incident_attenuation;
  adapter_input.attenuation_context = &attenuation_context;
  std::vector<int> source_ray_indices;
  const std::vector<sector_ps::RayPath> ray_paths =
      sector_adapter::build_ray_paths(adapter_input, &source_ray_indices);
  port_state->ray_bin.resize(source_ray_indices.size());
  for (std::size_t ray = 0; ray < source_ray_indices.size(); ++ray) {
    const int source_ray = source_ray_indices[ray];
    // build_keys adds n_bins only for mu>0 records. Phase-space sheet 0 is
    // the incoming (mu<0) leg, so beam-0 group_base carries the bin only.
    const int bin =
        ray_group_base[static_cast<std::size_t>(source_ray)] % cbet_ws.n_bins;
    TENRYU_ASSERT(bin >= 0 && bin < cbet_ws.n_bins,
                  "port_section ray impact bin out of range");
    port_state->ray_bin[ray] = bin;
  }
  int crossings_shell0 = 0;
  const double shell0 = hydro.r_edges.front();
  for (const sector_ps::RayPath& path : ray_paths) {
    for (std::size_t node = 0; node + 1 < path.r.size(); ++node) {
      if (path.r[node] != path.r[node + 1] &&
          shell0 >= std::min(path.r[node], path.r[node + 1]) &&
          shell0 <= std::max(path.r[node], path.r[node + 1])) {
        ++crossings_shell0;
      }
    }
  }

  std::vector<double> shell_eps(hydro.r_edges.size(), 0.0);
  for (std::size_t shell = 0; shell < hydro.r_edges.size(); ++shell) {
    const double radius = hydro.r_edges[shell];
    if (radius <= cell_r_center.front()) {
      shell_eps[shell] = cell_eps.front();
      continue;
    }
    if (radius >= cell_r_center.back()) {
      shell_eps[shell] = cell_eps.back();
      continue;
    }
    const auto upper =
        std::upper_bound(cell_r_center.begin(), cell_r_center.end(), radius);
    const std::size_t hi =
        static_cast<std::size_t>(upper - cell_r_center.begin());
    const std::size_t lo = hi - 1;
    const double weight =
        (radius - cell_r_center[lo]) /
        (cell_r_center[hi] - cell_r_center[lo]);
    shell_eps[shell] =
        cell_eps[lo] + weight * (cell_eps[hi] - cell_eps[lo]);
  }

  // Record-granularity audit tolerances (2026-07-31): per-record mu +
  // cell-edge radii give a coarse B reconstruction (floor ~0.3 on the
  // S1 fixture); the library defaults are precision-fixture numbers and
  // would trip the internal audit assert in Debug builds. 0.5 matches
  // the external catastrophic-breakage gate.
  sector_ps::PhaseSpaceParams s1_audit_params{};
  s1_audit_params.bouguer_tol = 0.5;
  s1_audit_params.bouguer_tol_fd = 0.5;
  port_state->table =
      sector_ps::build_table(ray_paths, hydro.r_edges, shell_eps,
                             s1_audit_params, port_state->annotations);
  port_state->ledger = sector_ps::exclusion_ledger(port_state->table);
  long long total_nodes = 0;
  for (const sector_ps::RayPath& path : ray_paths) {
    total_nodes += static_cast<long long>(path.r.size());
  }
  long long total_crossings = 0;
  for (std::size_t shell = 0; shell < hydro.r_edges.size(); ++shell) {
    for (int sheet = 0; sheet < 2; ++sheet) {
      total_crossings += static_cast<long long>(
          sector_ps::crossings(port_state->table,
                               static_cast<int>(shell), sheet).size());
    }
  }
  double bouguer_drift_max = 0.0;
  for (const sector_ps::RayAnnotation& annotation :
       port_state->annotations) {
    bouguer_drift_max =
        std::max(bouguer_drift_max, annotation.bouguer_max_drift);
  }
  std::memset(&port_state->audit, 0, sizeof(port_state->audit));
  port_state->audit.n_rays = n_rays;
  port_state->audit.total_nodes = total_nodes;
  port_state->audit.total_crossings = total_crossings;
  port_state->audit.excluded_frac =
      port_state->ledger.excluded_power_fraction;
  port_state->audit.bouguer_drift_max = bouguer_drift_max;
  ++port_state->build_count;

  if (verbose) {
    std::ostringstream oss;
    oss.setf(std::ios::scientific);
    oss << std::setprecision(3)
        << "port_section_s1: rays=" << port_state->audit.n_rays
        << " excluded_frac=" << port_state->audit.excluded_frac
        << " bouguer_drift=" << port_state->audit.bouguer_drift_max
        << " crossings_shell0=" << crossings_shell0
        << " ports=" << port_state->ports.ports.size()
        << " pairs=" << port_state->ports.pairs.size();
    core::log_info(oss.str());
  }
}

RaytraceSkipCache& global_skip_cache() {
  // Process-lifetime singleton to avoid static-destruction-order teardown crashes
  // when CUDA runtime has already begun unloading.
  static RaytraceSkipCache* cache = new RaytraceSkipCache();
  return *cache;
}

}  // namespace

namespace sector_adapter {

const S1Audit* last_s1_audit(const LaserMesh& mesh) {
  if (mesh.port_section_state == nullptr) {
    return nullptr;
  }
  const auto* const state =
      static_cast<const PortSectionState*>(mesh.port_section_state.get());
  if (state->build_count == 0) {
    return nullptr;
  }
  return &state->audit;
}

}  // namespace sector_adapter

void laser_step(core::State& state,
                LaserMesh& lmesh,
                const core::Config::LaserConfig& laser,
                const double dt,
                const double t,
                const parallel::PartitionInfo& part,
                cudaStream_t stream,
                const double rho_floor,
                const double Te_floor,
                bool* used_skip,
                std::vector<RayOutputData>* ray_output,
                const parallel::Reduction* reduction,
                const bool collect_trajectory,
                const bool verbose,
                const bool collect_density_diag,
                const std::string& output_dir) {
  const bool phys_ext_active = laser.laser_phys_ext_active();
  TENRYU_ASSERT(!phys_ext_active || state.mesh.dim == 1,
                "laser ib/ra extensions are 1D_SPH-only in v1");
  LaserPhysExtOptions phys_ext_options =
      build_phys_ext_options(laser);
  if (laser.raytrace.test_kappa > 0.0) {
    static bool warned_test_kappa = false;
    if (!warned_test_kappa) {
      core::log_warning(
          "Laser raytrace.test_kappa > 0 overrides the inverse-bremsstrahlung "
          "opacity with a constant — TEST HOOK, results are not physical.");
      warned_test_kappa = true;
    }
  }
  using Clock = std::chrono::steady_clock;
  const auto t_step_start = Clock::now();
  auto t_setup_end = t_step_start;
  auto t_skip_end = t_step_start;
  auto t_map_end = t_step_start;
  auto t_trace_end = t_step_start;
  auto t_transfer_end = t_step_start;
  auto t_finalize_end = t_step_start;
  double init_ms = 0.0;
  double density_diag_ms = 0.0;
  double capture_ms = 0.0;
  double memset_ms = 0.0;
  double kernel_ms = 0.0;
  double transfer_ms = 0.0;
  double ps_s1_ms = 0.0;
  double ps_chi_d2h_ms = 0.0;
  double ps_chi_build_ms = 0.0;
  double ps_chi_post_ms = 0.0;
  double ps_solve_ms = 0.0;
  double ps_capture_ms = 0.0;
  const bool port_section = laser.cbet.enable &&
                            laser.cbet.geometry_mode == "port_section";
  auto ms = [](const auto& a, const auto& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  auto emit_timing = [&]() {
    if (!verbose) {
      return;
    }
    const double trace_total_ms = ms(t_map_end, t_trace_end);
    core::log_info("[laser_timing] step=" + std::to_string(state.step) +
                   " setup=" + std::to_string(ms(t_step_start, t_setup_end)) +
                   " skip=" + std::to_string(ms(t_setup_end, t_skip_end)) +
                   " map=" + std::to_string(ms(t_skip_end, t_map_end)) +
                   " trace=" + std::to_string(trace_total_ms) +
                   " transfer=" + std::to_string(ms(t_trace_end, t_transfer_end)) +
                   " finalize=" + std::to_string(ms(t_transfer_end, t_finalize_end)) +
                   " total=" + std::to_string(ms(t_step_start, t_finalize_end)) + "ms");
    if (laser.mode != "radial_absorption_1d") {
      const double finalize_ms = std::max(
          0.0, trace_total_ms - (init_ms + density_diag_ms + capture_ms + memset_ms +
                                 kernel_ms + transfer_ms));
      core::log_info("[laser_subtrace] init=" + std::to_string(init_ms) +
                     " density_diag=" + std::to_string(density_diag_ms) +
                     " capture=" + std::to_string(capture_ms) +
                     " memset=" + std::to_string(memset_ms) +
                     " kernel=" + std::to_string(kernel_ms) +
                     " transfer=" + std::to_string(transfer_ms) +
                     " finalize=" + std::to_string(finalize_ms) +
                     " total=" + std::to_string(trace_total_ms) + "ms");
    }
    if (port_section) {
      core::log_info("[ps_timing] s1=" + std::to_string(ps_s1_ms) +
                     " chi_d2h=" + std::to_string(ps_chi_d2h_ms) +
                     " chi_build=" + std::to_string(ps_chi_build_ms) +
                     " chi_post=" + std::to_string(ps_chi_post_ms) +
                     " solve=" + std::to_string(ps_solve_ms) +
                     " capture=" + std::to_string(ps_capture_ms) + "ms");
    }
  };
  auto emit_setup_only_timing = [&]() {
    if (!verbose) {
      return;
    }
    const auto t_now = Clock::now();
    t_setup_end = t_now;
    t_skip_end = t_now;
    t_map_end = t_now;
    t_trace_end = t_now;
    t_transfer_end = t_now;
    t_finalize_end = t_now;
    emit_timing();
  };

  if (used_skip != nullptr) {
    *used_skip = false;
  }
  if (ray_output != nullptr) {
    ray_output->clear();
  }
  state.hot_e_in_step = 0.0;
  state.hot_e_deposited_step = 0.0;
  state.hot_e_residual_step = 0.0;
  state.hot_e_escaped_step = 0.0;
  state.E_cbet_iaw_step = 0.0;
  state.hot_e_conservation_resid = 0.0;
  state.hot_e_dt_limit_s = std::numeric_limits<double>::infinity();
  if (!state.hot_e_Q_host.empty()) {
    std::fill(state.hot_e_Q_host.begin(), state.hot_e_Q_host.end(), 0.0);
  }
  std::fill(state.hot_e_ch_in_step.begin(), state.hot_e_ch_in_step.end(), 0.0);
  std::fill(state.hot_e_ch_deposited_step.begin(), state.hot_e_ch_deposited_step.end(), 0.0);
  std::fill(state.hot_e_ch_escaped_step.begin(), state.hot_e_ch_escaped_step.end(), 0.0);
  state.ray_density.fill(0.0);
  if (collect_trajectory) {
    state.ray_traj_offsets.clear();
    state.ray_traj_step_counts.clear();
    state.ray_traj_beam_ids.clear();
    state.ray_traj_pos1.clear();
    state.ray_traj_pos2.clear();
    state.ray_traj_pos3.clear();
    state.ray_traj_power.clear();
    state.ray_traj_is_3d = (state.mesh.dim == 2);
    state.laser_mesh_n_nodes_r = 0;
    state.laser_mesh_n_nodes_z = 0;
  }
  lmesh.last_commanded_energy = 0.0;
  lmesh.last_trace_unabsorbed_power = 0.0;
  lmesh.last_transfer_blocked_power = 0.0;
  lmesh.last_unabsorbed_power = 0.0;
  lmesh.last_ra_power = 0.0;
  lmesh.last_tail_closure_count = 0;
  lmesh.last_tail_closure_absorbed_power = 0.0;
  lmesh.last_critical_surface_hit_count = 0;
  lmesh.last_cbet_exchanged_power = 0.0;
  lmesh.last_cbet_ledger_residual = 0.0;
  lmesh.last_cbet_conv_final = 0.0;
  lmesh.last_cbet_clamp_count = 0;
  lmesh.last_cbet_overflow_rays = 0;
  lmesh.last_cbet_iterations = 0;
  lmesh.last_cbet_converged = true;
  const bool radial_absorption_1d =
      (state.mesh.dim == 1 && laser.mode == "radial_absorption_1d");
  if (radial_absorption_1d) {
    global_skip_cache().invalidate();
  }

  if (!laser.enabled) {
    state.laser_dep.fill(0.0);
    lmesh.clear_deposit(stream);
    emit_setup_only_timing();
    return;
  }

  if (laser.mode != "raytrace_2d" && laser.mode != "raytrace_3d" &&
      laser.mode != "radial_absorption_1d") {
    core::log_warning("Laser mode '" + laser.mode + "' is unsupported; skipping");
    state.laser_dep.fill(0.0);
    lmesh.clear_deposit(stream);
    lmesh.last_trace_unabsorbed_power = 0.0;
    lmesh.last_transfer_blocked_power = 0.0;
    lmesh.last_unabsorbed_power = 0.0;
    emit_setup_only_timing();
    return;
  }

  if (state.mesh.dim == 1 && laser.mode != "raytrace_2d" &&
      laser.mode != "radial_absorption_1d") {
    core::log_warning(
        "laser_step requires mode=raytrace_2d or radial_absorption_1d for 1D_SPH; skipping");
    state.laser_dep.fill(0.0);
    lmesh.clear_deposit(stream);
    lmesh.last_trace_unabsorbed_power = 0.0;
    lmesh.last_transfer_blocked_power = 0.0;
    lmesh.last_unabsorbed_power = 0.0;
    emit_setup_only_timing();
    return;
  }
  if (state.mesh.dim == 2 && laser.mode != "raytrace_3d") {
    core::log_warning("laser_step requires mode=raytrace_3d for 2D_RZ; skipping");
    state.laser_dep.fill(0.0);
    lmesh.clear_deposit(stream);
    lmesh.last_trace_unabsorbed_power = 0.0;
    lmesh.last_transfer_blocked_power = 0.0;
    lmesh.last_unabsorbed_power = 0.0;
    emit_setup_only_timing();
    return;
  }
  TENRYU_ASSERT(lmesh.is_allocated(), "laser_step requires allocated LaserMesh");

  if (!(dt > 0.0)) {
    core::log_warning("laser_step received non-positive dt; skipping laser deposition");
    state.laser_dep.fill(0.0);
    lmesh.clear_deposit(stream);
    lmesh.last_trace_unabsorbed_power = 0.0;
    lmesh.last_transfer_blocked_power = 0.0;
    lmesh.last_unabsorbed_power = 0.0;
    emit_setup_only_timing();
    return;
  }

  const bool cbet_on = laser.enabled && laser.cbet.enable && state.mesh.dim == 1 &&
                       laser.mode == "raytrace_2d";
  const bool hot_e_on_1d = laser.enabled && laser.hot_electron.enable && state.mesh.dim == 1;
  const bool cbet_on_2d = laser.enabled && laser.cbet.enable && state.mesh.dim == 2 &&
                          laser.mode == "raytrace_3d";
  double cbet_iaw_power = 0.0;
  TENRYU_ASSERT(!(cbet_on || cbet_on_2d) || part.n_ranks == 1,
                "Laser.cbet v1 requires a single MPI rank");
  const bool hot_e_on_2d =
      laser.enabled && laser.hot_electron.enable && state.mesh.dim == 2;
  const bool hot_e_on = hot_e_on_1d || hot_e_on_2d;
  const bool hot_e_model =
      hot_e_on && (laser.hot_electron.eta_mode == "model");
  if (hot_e_on && part.n_ranks > 1) {
    TENRYU_ASSERT(false, "Laser.hot_electron requires a single rank");
  }
  if (port_section && lmesh.port_section_state == nullptr) {
    auto port_state = std::make_shared<PortSectionState>();
    std::vector<port_geom::Port> ports;
    ports.reserve(laser.port_configuration.ports.size());
    for (const auto& port : laser.port_configuration.ports) {
      ports.push_back(port_geom::Port{
          port.port_id,
          {port.direction[0], port.direction[1], port.direction[2]},
          port.roll_deg,
          port.power_weight,
          port.delta_lambda_nm,
          port.beam_class});
    }
    port_state->ports =
        port_geom::build_port_table(std::move(ports), laser.wavelength_nm);
    lmesh.port_section_state = std::move(port_state);
  }

  const double z_center = 0.5 * (lmesh.Z_min + lmesh.Z_max);
  const Beams beams = create_from_config(laser, state, lmesh.target_radius, z_center);
  if (laser.ib.langdon_model != "off") {
    TENRYU_ASSERT(beams.items.size() == 1,
                  "langdon legacy_vacuum_map requires a single beam");
    TENRYU_ASSERT(beams.items.front().profile_model == "gaussian" ||
                      beams.items.front().profile_model == "flat_top",
                  "langdon legacy_vacuum_map requires a gaussian or flat_top beam "
                  "profile (2026-08-02: flat_top uniform-disc vacuum map added for "
                  "HELIOS-parity decks)");
  }
  RaytraceSkipCache* skip_cache = nullptr;
  if (!radial_absorption_1d && !hot_e_on) {
    skip_cache = &global_skip_cache();
  }
  if (beams.items.empty()) {
    if (skip_cache != nullptr) {
      skip_cache->invalidate();
    }
    state.laser_dep.fill(0.0);
    lmesh.clear_deposit(stream);
    lmesh.last_trace_unabsorbed_power = 0.0;
    lmesh.last_transfer_blocked_power = 0.0;
    lmesh.last_unabsorbed_power = 0.0;
    emit_setup_only_timing();
    return;
  }
  std::vector<Vec3> beam_dirs;
  beam_dirs.reserve(beams.items.size());
  std::vector<Vec3> beam_focuses;
  beam_focuses.reserve(beams.items.size());
  std::vector<double> beam_defocus;
  beam_defocus.reserve(beams.items.size());
  for (const Beam& beam : beams.items) {
    beam_dirs.push_back(Vec3{beam.dir_x, beam.dir_y, beam.dir_z});
    beam_focuses.push_back(Vec3{beam.focus_x, beam.focus_y, beam.focus_lab_z});
    beam_defocus.push_back(beam.defocus_DR);
  }

  const double total_power = beams.total_power(t);
  lmesh.last_commanded_energy = total_power * dt;
  if (!(total_power > 0.0)) {
    // Keep skip cache transitions consistent when power goes to zero.
    if (skip_cache != nullptr) {
      skip_cache->invalidate();
    }
    state.laser_dep.fill(0.0);
    lmesh.clear_deposit(stream);
    lmesh.last_trace_unabsorbed_power = 0.0;
    lmesh.last_transfer_blocked_power = 0.0;
    lmesh.last_unabsorbed_power = 0.0;
    emit_setup_only_timing();
    if (hot_e_model && !state.hot_e_eta_prev_valid.empty()) {
      std::fill(state.hot_e_eta_prev_Pcross.begin(),
                state.hot_e_eta_prev_Pcross.end(), 0.0);
      std::fill(state.hot_e_eta_prev_valid.begin(),
                state.hot_e_eta_prev_valid.end(),
                static_cast<std::uint8_t>(0));
    }
    return;
  }

  if (!radial_absorption_1d && laser.rays_per_beam <= 0) {
    core::log_warning("laser_step rays_per_beam <= 0; skipping ray trace and marking all beam "
                      "power as unabsorbed");
    state.laser_dep.fill(0.0);
    lmesh.clear_deposit(stream);
    lmesh.last_trace_unabsorbed_power = total_power;
    lmesh.last_transfer_blocked_power = 0.0;
    lmesh.last_unabsorbed_power = total_power;
    emit_setup_only_timing();
    if (hot_e_model && !state.hot_e_eta_prev_valid.empty()) {
      std::fill(state.hot_e_eta_prev_Pcross.begin(),
                state.hot_e_eta_prev_Pcross.end(), 0.0);
      std::fill(state.hot_e_eta_prev_valid.begin(),
                state.hot_e_eta_prev_valid.end(),
                static_cast<std::uint8_t>(0));
    }
    return;
  }

  struct ResolvedHotEChannel {
    int config_index;
    double f_s;
    double eta_eff;
  };
  static_assert(HotECaptureParams::kMaxChannels >=
                    core::Config::LaserConfig::HotElectronConfig::kMaxSources,
                "hot-electron kernel channel capacity must cover the config channel cap");
  std::vector<ResolvedHotEChannel> hot_e_channels;
  int hot_e_n_config_channels = 0;
  bool hot_e_capture_on = false;
  bool ps_hot_e_capture_on = false;
  std::vector<double> hot_e_model_cell_nhat;
  std::vector<double> hot_e_model_cell_r;
  if (hot_e_on && total_power > 0.0) {
    const auto& he_cfg = laser.hot_electron;
    const auto clamp_eta = [](const double eta_raw) {
      if (eta_raw < 0.0 || eta_raw >= 1.0) {
        static bool warned_eta_range = false;
        if (!warned_eta_range) {
          warned_eta_range = true;
          core::log_warning("Laser.hot_electron eta table value outside [0,1); clamping to [0,0.95]");
        }
      }
      return std::min(std::max(eta_raw, 0.0), 0.95);
    };
    if (hot_e_model) {
      TENRYU_ASSERT(
          part.n_ranks <= 1,
          "hot_electron eta_mode=\"model\" does not support MPI in v1");
      hot_e_n_config_channels = static_cast<int>(he_cfg.sources.size());
      const std::size_t n_ch =
          static_cast<std::size_t>(hot_e_n_config_channels);
      if (state.hot_e_eta_state_eta.size() != n_ch) {
        state.hot_e_eta_state_eta.assign(n_ch, 0.0);
        state.hot_e_eta_state_kappa_bar.assign(n_ch, 0.0);
        state.hot_e_eta_prev_Pcross.assign(n_ch, 0.0);
        state.hot_e_eta_prev_rbar.assign(n_ch, 0.0);
        state.hot_e_eta_prev_valid.assign(n_ch, static_cast<std::uint8_t>(0));
        state.hot_e_eta_diag_g.assign(n_ch, 0.0);
        state.hot_e_eta_diag_eta_eq.assign(n_ch, 0.0);
        state.hot_e_eta_diag_tau_s.assign(n_ch, 0.0);
        state.hot_e_eta_diag_I14.assign(n_ch, 0.0);
        state.hot_e_eta_diag_I14_lower.assign(n_ch, 0.0);
        state.hot_e_eta_diag_I14_upper.assign(n_ch, 0.0);
        state.hot_e_eta_diag_n_sigma.assign(n_ch, 0.0);
        state.hot_e_eta_diag_Te_keV.assign(n_ch, 0.0);
        state.hot_e_eta_diag_Ln_um.assign(n_ch, 0.0);
        state.hot_e_eta_diag_clamped.assign(n_ch, 0.0);
      }
      if (state.hot_e_eta_diag_I14_lower.size() != n_ch) {
        state.hot_e_eta_diag_I14_lower.assign(n_ch, 0.0);
      }
      if (state.hot_e_eta_diag_I14_upper.size() != n_ch) {
        state.hot_e_eta_diag_I14_upper.assign(n_ch, 0.0);
      }
      if (state.hot_e_eta_diag_n_sigma.size() != n_ch) {
        state.hot_e_eta_diag_n_sigma.assign(n_ch, 0.0);
      }

      HydroMirror1D em;
      build_hydro_mirror_1d(lmesh, state, em);
      std::vector<double> h_Te(state.Te.size());
      state.Te.copy_to_host(h_Te.data());
      const int n_cells = static_cast<int>(em.rho.size());
      hot_e_model_cell_r.assign(static_cast<std::size_t>(n_cells), 0.0);
      hot_e_model_cell_nhat.assign(static_cast<std::size_t>(n_cells), 0.0);
      std::vector<double> n_e(static_cast<std::size_t>(n_cells), 0.0);
      for (int c = 0; c < n_cells; ++c) {
        const std::size_t s = static_cast<std::size_t>(c);
        hot_e_model_cell_r[s] =
            0.5 * (em.r_edges[s] + em.r_edges[s + 1U]);
        if (em.cell_is_void[s] != 0U) {
          n_e[s] = 0.0;
        } else {
          n_e[s] =
              std::max(0.0, em.rho[s]) * std::max(0.0, em.zbar[s]) /
              (std::max(em.A_eff[s], 1.0e-30) *
               core::constants::proton_mass);
        }
      }
      const double n_crit_safe = std::max(lmesh.n_crit, 1.0e-30);
      for (int c = 0; c < n_cells; ++c) {
        const std::size_t s = static_cast<std::size_t>(c);
        hot_e_model_cell_nhat[s] = n_e[s] / n_crit_safe;
      }

      hot_e_eta::ModelParams model_params;
      model_params.ln_filter_tau_s = he_cfg.eta_model.ln_filter_tau_s;
      model_params.eta_total_cap = he_cfg.eta_model.eta_total_cap;
      std::vector<hot_e_eta::ChannelState> model_states(n_ch);
      int sky_eval_channel = 0;
      for (int ci = 0; ci < hot_e_n_config_channels; ++ci) {
        if (he_cfg.sources[static_cast<std::size_t>(ci)].mechanism ==
            "tpd") {
          sky_eval_channel = ci;
          break;
        }
      }
      for (int ci = 0; ci < hot_e_n_config_channels; ++ci) {
        const std::size_t s = static_cast<std::size_t>(ci);
        const auto& source = he_cfg.sources[s];
        state.hot_e_eta_diag_I14_lower[s] = 0.0;
        state.hot_e_eta_diag_I14_upper[s] = 0.0;
        state.hot_e_eta_diag_n_sigma[s] = 0.0;
        hot_e_eta::ChannelInputs inputs;
        const double n_tgt =
            source.eval_nc_fraction * n_crit_safe;
        int eval_c = -1;
        double alpha = 0.0;
        for (int c = n_cells - 1; c >= 1; --c) {
          const std::size_t outer = static_cast<std::size_t>(c);
          const std::size_t inner = static_cast<std::size_t>(c - 1);
          if (em.cell_is_void[outer] != 0U ||
              em.cell_is_void[inner] != 0U ||
              !(n_e[outer] > 0.0) || !(n_e[inner] > 0.0)) {
            continue;
          }
          if (n_e[outer] < n_tgt && n_e[inner] >= n_tgt) {
            eval_c = c;
            alpha =
                (n_tgt - n_e[outer]) / (n_e[inner] - n_e[outer]);
            break;
          }
        }

        double Te_s_eV = 0.0;
        double kappa_um = 0.0;
        double eval_radius_cm = 0.0;
        int eval_shell = -1;
        bool valid = (eval_c >= 1);
        if (valid) {
          const std::size_t outer = static_cast<std::size_t>(eval_c);
          const std::size_t inner = static_cast<std::size_t>(eval_c - 1);
          Te_s_eV = (1.0 - alpha) * h_Te[outer] + alpha * h_Te[inner];
          eval_radius_cm =
              hot_e_model_cell_r[outer] +
              alpha * (hot_e_model_cell_r[inner] -
                       hot_e_model_cell_r[outer]);
          eval_shell = nearest_shell_index(em.r_edges, eval_radius_cm);
          const double r_s_um =
              eval_radius_cm * 1.0e4;
          const int fit_begin = std::max(0, eval_c - 4);
          const int fit_end = std::min(n_cells - 1, eval_c + 2);
          std::vector<double> fit_r_um;
          std::vector<double> fit_ne;
          fit_r_um.reserve(static_cast<std::size_t>(fit_end - fit_begin + 1));
          fit_ne.reserve(static_cast<std::size_t>(fit_end - fit_begin + 1));
          for (int c = fit_begin; c <= fit_end; ++c) {
            const std::size_t fit_s = static_cast<std::size_t>(c);
            if (em.cell_is_void[fit_s] != 0U || !(n_e[fit_s] > 0.0)) {
              continue;
            }
            fit_r_um.push_back(hot_e_model_cell_r[fit_s] * 1.0e4);
            fit_ne.push_back(n_e[fit_s]);
          }
          kappa_um = hot_e_eta::fit_kappa_abs_um(
              fit_r_um.data(), fit_ne.data(), fit_r_um.size(), r_s_um);
          if (!(kappa_um > 0.0)) {
            valid = false;
          }
        }

        double f_illum = 1.0;
        const auto* const port_state =
            port_section
                ? static_cast<const PortSectionState*>(
                      lmesh.port_section_state.get())
                : nullptr;
        const bool collect_sky_grid =
            port_section && ci == sky_eval_channel &&
            port_state != nullptr && port_state->build_count > 0 &&
            eval_shell >= 0;
        if (port_section &&
            he_cfg.illumination_metric == "equivalent_area") {
          if (port_state != nullptr && port_state->build_count > 0 &&
              eval_shell >= 0) {
            const ::tenryu::laser::port_section::IlluminationInput
                illumination_input{
                    &port_state->ports, &port_state->table, eval_shell};
            const port_geom::IlluminationResult illumination =
                ::tenryu::laser::port_section::illumination_at_shell(
                    illumination_input);
            f_illum = illumination.f_illum2;
            if (ci == sky_eval_channel) {
              state.ps_f_illum2 = illumination.f_illum2;
              state.ps_f_union = illumination.f_union;
            }
            if (!(std::isfinite(f_illum) && f_illum > 0.0)) {
              valid = false;
              f_illum = 1.0;
            }
          } else {
            valid = false;
          }
        }

        double I14 = 0.0;
        const double P = state.hot_e_eta_prev_Pcross[s];
        const double rbar = state.hot_e_eta_prev_rbar[s];
        if (state.hot_e_eta_prev_valid[s] != 0U &&
            P > 0.0 && rbar > 0.0) {
          I14 = P / (4.0 * M_PI * rbar * rbar * f_illum) /
                1.0e7 / 1.0e14;
        } else {
          valid = false;
        }
        if (port_section && source.mechanism == "tpd" &&
            he_cfg.tpd_overlap_mode == "common_wave_cluster") {
          ::tenryu::laser::port_section::CommonWaveDriveResult drive{};
          ::tenryu::laser::port_section::CommonWaveDriveGridOutput
              grid_output;
          if (port_state != nullptr && port_state->build_count > 0 &&
              eval_shell >= 0) {
            const ::tenryu::laser::port_section::CommonWaveDriveInput
                drive_input{
                    &port_state->ports,
                    &port_state->table,
                    eval_shell,
                    he_cfg.common_wave_delta_theta_deg};
            drive =
                ::tenryu::laser::port_section::common_wave_drive(
                    drive_input,
                    collect_sky_grid ? &grid_output : nullptr);
            if (collect_sky_grid) {
              fill_port_section_sky_map(state, grid_output);
            }
            if (std::isfinite(drive.I_drive) && drive.I_drive > 0.0) {
              I14 = drive.I_drive / 1.0e7 / 1.0e14;
            } else {
              valid = false;
            }
          } else {
            valid = false;
          }
          std::ostringstream drive_oss;
          drive_oss.setf(std::ios::scientific);
          drive_oss << std::setprecision(6)
                    << "hot_e_tpd_common_wave: ch=" << ci
                    << " lower=" << drive.I_lower
                    << " drive=" << drive.I_drive
                    << " upper=" << drive.I_upper;
          core::log_info(drive_oss.str());
          state.hot_e_eta_diag_I14_lower[s] =
              drive.I_lower / 1.0e7 / 1.0e14;
          state.hot_e_eta_diag_I14_upper[s] =
              drive.I_upper / 1.0e7 / 1.0e14;
          state.hot_e_eta_diag_n_sigma[s] =
              static_cast<double>(drive.n_sigma_mode);
        } else if (collect_sky_grid) {
          const ::tenryu::laser::port_section::CommonWaveDriveInput
              drive_input{
                  &port_state->ports,
                  &port_state->table,
                  eval_shell,
                  he_cfg.common_wave_delta_theta_deg};
          ::tenryu::laser::port_section::CommonWaveDriveGridOutput
              grid_output;
          static_cast<void>(
              ::tenryu::laser::port_section::common_wave_drive(
                  drive_input, &grid_output));
          fill_port_section_sky_map(state, grid_output);
        }
        if (!(Te_s_eV > 0.0)) {
          valid = false;
        }
        inputs.I14 = I14;
        inputs.Te_keV = Te_s_eV / 1000.0;
        inputs.kappa_um = kappa_um;
        inputs.valid = valid;

        hot_e_eta::ChannelParams channel_params;
        channel_params.mechanism =
            (source.mechanism == "tpd")
                ? hot_e_eta::Mechanism::kTpd
                : hot_e_eta::Mechanism::kSrs;
        channel_params.threshold_multiplier = source.threshold_multiplier;
        channel_params.eta_inf = source.eta_inf;
        channel_params.eta_hard_cap = source.eta_hard_cap;
        channel_params.shape_coefficient = source.shape_coefficient;
        channel_params.tau_vu2012 =
            (source.relaxation_model == "vu2012");
        channel_params.relaxation_tau_s = source.relaxation_tau_s;
        channel_params.relaxation_tau_min_s =
            source.relaxation_tau_min_s;
        channel_params.relaxation_tau_max_s =
            source.relaxation_tau_max_s;

        hot_e_eta::ChannelState channel_state{
            state.hot_e_eta_state_eta[s],
            state.hot_e_eta_state_kappa_bar[s]};
        hot_e_eta::ChannelDiagnostics diag;
        hot_e_eta::update_channel(
            channel_params, model_params, channel_state, inputs, dt,
            laser.wavelength_nm * 1.0e-3, diag);
        model_states[s] = channel_state;
        state.hot_e_eta_state_eta[s] = channel_state.eta;
        state.hot_e_eta_state_kappa_bar[s] = channel_state.kappa_bar;
        state.hot_e_eta_diag_g[s] = diag.g;
        state.hot_e_eta_diag_eta_eq[s] = diag.eta_eq;
        state.hot_e_eta_diag_tau_s[s] = diag.tau_s;
        state.hot_e_eta_diag_I14[s] = inputs.I14;
        state.hot_e_eta_diag_Te_keV[s] = inputs.Te_keV;
        state.hot_e_eta_diag_Ln_um[s] = diag.L_n_eff_um;
        state.hot_e_eta_diag_clamped[s] =
            static_cast<double>(diag.clamped);
      }

      hot_e_eta::apply_total_cap(
          model_params, model_states.data(), model_states.size());
      for (int ci = 0; ci < hot_e_n_config_channels; ++ci) {
        const std::size_t s = static_cast<std::size_t>(ci);
        state.hot_e_eta_state_eta[s] = model_states[s].eta;
      }
      for (int ci = 0; ci < hot_e_n_config_channels; ++ci) {
        hot_e_channels.push_back(ResolvedHotEChannel{
            ci,
            he_cfg.sources[static_cast<std::size_t>(ci)]
                .capture_nc_fraction,
            state.hot_e_eta_state_eta[static_cast<std::size_t>(ci)]});
      }
      if (verbose) {
        std::ostringstream oss;
        oss.setf(std::ios::scientific);
        oss << std::setprecision(3) << "hot_e_eta_model:";
        for (int ci = 0; ci < hot_e_n_config_channels; ++ci) {
          const std::size_t s = static_cast<std::size_t>(ci);
          oss << " ch" << ci << "["
              << he_cfg.sources[s].mechanism << "] eta="
              << state.hot_e_eta_state_eta[s] << " g="
              << state.hot_e_eta_diag_g[s] << " eta_eq="
              << state.hot_e_eta_diag_eta_eq[s] << " I14="
              << state.hot_e_eta_diag_I14[s] << " Te_keV="
              << state.hot_e_eta_diag_Te_keV[s] << " Ln_um="
              << state.hot_e_eta_diag_Ln_um[s] << " tau_ps="
              << state.hot_e_eta_diag_tau_s[s] * 1.0e12 << " clamp="
              << static_cast<int>(state.hot_e_eta_diag_clamped[s]);
          if (ci + 1 < hot_e_n_config_channels) {
            oss << ";";
          }
        }
        core::log_info(oss.str());
      }
    } else if (he_cfg.sources_specified) {
      hot_e_n_config_channels = static_cast<int>(he_cfg.sources.size());
      for (int ci = 0; ci < hot_e_n_config_channels; ++ci) {
        const auto& channel = he_cfg.sources[static_cast<std::size_t>(ci)];
        double eta_eff = channel.eta;
        if (channel.eta_table.detected &&
            static_cast<std::size_t>(ci) < state.hot_e_eta_ch_1d.size() &&
            state.hot_e_eta_ch_1d[static_cast<std::size_t>(ci)].has_value()) {
          eta_eff = clamp_eta(state.hot_e_eta_ch_1d[static_cast<std::size_t>(ci)]->eval(t));
        }
        if (eta_eff > 0.0) {
          hot_e_channels.push_back(
              ResolvedHotEChannel{ci, channel.capture_nc_fraction, eta_eff});
        }
      }
    } else {
      hot_e_n_config_channels = 1;
      double eta_eff = he_cfg.eta_hot;
      if (he_cfg.eta_hot_table.detected && state.hot_e_eta_1d.has_value()) {
        eta_eff = clamp_eta(state.hot_e_eta_1d->eval(t));
      }
      if (eta_eff > 0.0) {
        hot_e_channels.push_back(
            ResolvedHotEChannel{0, he_cfg.source_nc_fraction, eta_eff});
      }
    }
    std::stable_sort(hot_e_channels.begin(), hot_e_channels.end(),
                     [](const ResolvedHotEChannel& a, const ResolvedHotEChannel& b) {
                       return a.f_s < b.f_s;
                     });
    ps_hot_e_capture_on =
        port_section && hot_e_model && !hot_e_channels.empty();
    hot_e_capture_on = !port_section && !hot_e_channels.empty();
  }
  const bool hot_e_transport_on =
      hot_e_capture_on || ps_hot_e_capture_on;
  tenryu::laser::HotECaptureParams hot_e_params;
  if (hot_e_capture_on) {
    hot_e_params.n_channels = static_cast<int>(hot_e_channels.size());
    for (int k = 0; k < hot_e_params.n_channels; ++k) {
      hot_e_params.threshold_nhat[k] = hot_e_channels[static_cast<std::size_t>(k)].f_s;
      hot_e_params.one_minus_eta[k] =
          laser.hot_electron.subtract_from_laser
              ? (1.0 - hot_e_channels[static_cast<std::size_t>(k)].eta_eff)
              : 1.0;
    }
  }
  std::vector<std::vector<tenryu::laser::hot_electron::RayCapture>>
      hot_e_captures_by_channel(static_cast<std::size_t>(hot_e_n_config_channels));
  std::vector<double> hot_e_capture_stage;
  int hot_e_capture_beams = 0;
  std::vector<std::vector<tenryu::laser::hot_electron::RayCapture2D>>
      hot_e_captures_by_channel_2d(static_cast<std::size_t>(hot_e_n_config_channels));
  std::vector<double> hot_e_model_sum_P;
  std::vector<double> hot_e_model_sum_Pr;
  double ps_banked_hot_e_power = 0.0;
  if (hot_e_model) {
    hot_e_model_sum_P.assign(
        static_cast<std::size_t>(hot_e_n_config_channels), 0.0);
    hot_e_model_sum_Pr.assign(
        static_cast<std::size_t>(hot_e_n_config_channels), 0.0);
  }

  std::vector<BeamGroup> beam_groups;
  std::vector<double> group_powers;
  std::vector<std::vector<double>> f_hat_groups;
  std::vector<double> skip_group_powers;
  if (state.mesh.dim == 1) {
    group_powers.assign(beams.items.size(), 0.0);
    for (std::size_t b = 0; b < beams.items.size(); ++b) {
      group_powers[b] = std::max(0.0, beams.items[b].get_power(t));
    }
    f_hat_groups.assign(group_powers.size(), std::vector<double>(state.laser_dep.size(), 0.0));
    skip_group_powers = group_powers;
  } else {
    beam_groups = group_beams_by_theta(beams.items, laser.cbet.enable);
    group_powers.assign(beam_groups.size(), 0.0);
    for (std::size_t g = 0; g < beam_groups.size(); ++g) {
      double Pg = 0.0;
      for (const int bi : beam_groups[g].beam_indices) {
        Pg += std::max(0.0, beams.items[static_cast<std::size_t>(bi)].get_power(t));
      }
      group_powers[g] = Pg;
    }
    f_hat_groups.assign(group_powers.size(), std::vector<double>(state.laser_dep.size(), 0.0));
    const double total_group_power =
        std::accumulate(group_powers.begin(), group_powers.end(), 0.0, [](double a, double b) {
          return a + std::max(0.0, b);
        });
    skip_group_powers.assign(1, total_group_power);
  }
  t_setup_end = Clock::now();
  // SS16.6 invariant dump (design doc §6p): the env forces FULL ray-record
  // collection regardless of the namelist count so the launch set can be
  // compared bitwise across ranks. Read-only observer; no physics effect.
  const bool lm_invariant_collect_all =
      std::getenv("TENRYU_LM_INVARIANT_DUMP") != nullptr;
  const int ray_record_cap = lm_invariant_collect_all
                                 ? std::numeric_limits<int>::max()
                                 : laser.ray_output_count;
  const bool collect_ray_output =
      (ray_record_cap > 0) && (ray_output != nullptr);
  std::vector<double> ray_counts(state.ray_density.size(), 0.0);
  HydroMirror1D hydro_mirror;
  HydroCellLocator2D hydro_locator_2d(state);

  bool skip_raytrace = false;
  if (!radial_absorption_1d && !hot_e_on && skip_cache != nullptr && !cbet_on_2d) {
    skip_cache->ensure_capacity(static_cast<int>(state.laser_dep.size()),
                                std::max(1, static_cast<int>(skip_group_powers.size())));

    const double A_eff_uniform = lmesh.material_A;
    const bool use_global_skip_sync =
        (reduction != nullptr && part.n_ranks > 1);
    double local_skip_metric = std::numeric_limits<double>::infinity();
    bool local_skip_eligible = false;
    bool local_crit_hit = false;
    const bool local_skip = skip_cache->should_skip(
        state, laser, lmesh.n_crit, lmesh.n_hat_margin, lmesh.material_A_list,
        A_eff_uniform, skip_group_powers, beam_dirs, beam_focuses, beam_defocus,
        state.ale_rezoned, stream, rho_floor, Te_floor, &local_skip_metric,
        &local_skip_eligible, &local_crit_hit, cbet_on);
    if (use_global_skip_sync) {
      const double all_skip_eligible =
          reduction->allreduce_min(local_skip_eligible ? 1.0 : 0.0);
      if (all_skip_eligible > 0.5) {
        const double global_metric =
            reduction->allreduce_max(local_crit_hit
                                         ? std::numeric_limits<double>::infinity()
                                         : local_skip_metric);
        skip_raytrace = std::isfinite(global_metric) &&
                        (global_metric < laser.raytrace_skip_config.threshold);
      } else {
        skip_raytrace = false;
      }
      if (!skip_raytrace) {
        skip_cache->consecutive_skip_count = 0;
      }
    } else {
      skip_raytrace = local_skip;
    }
  }

  if (skip_raytrace) {
    if (used_skip != nullptr) {
      *used_skip = true;
    }
    skip_cache->scale_deposit(state, skip_group_powers, dt, stream);
    if (part.n_ranks > 1) {
      // Keep skip-path deposition ownership consistent with transfer_to_1d/transfer_to_2d.
      mask_non_owned_skip_deposit(state, part);
    }
    const double dep_power_local = sum_field_energy(state.laser_dep) / dt;
    const double dep_power =
        (reduction != nullptr && part.n_ranks > 1)
            ? reduction->allreduce_sum(dep_power_local)
            : dep_power_local;
    lmesh.last_trace_unabsorbed_power = std::max(0.0, total_power - dep_power);
    lmesh.last_unabsorbed_power = std::max(0.0, total_power - dep_power);
    lmesh.last_tail_closure_count = 0;
    lmesh.last_tail_closure_absorbed_power = 0.0;
    lmesh.last_critical_surface_hit_count = 0;
    lmesh.clear_deposit(stream);
    cuda_check(cudaStreamSynchronize(stream),
               "laser_step stream synchronize failed after skip scaling");
    const auto t_now = Clock::now();
    t_skip_end = t_now;
    t_map_end = t_now;
    t_trace_end = t_now;
    t_transfer_end = t_now;
    t_finalize_end = t_now;
    emit_timing();
    return;
  }
  t_skip_end = Clock::now();

  const double lambda_cm = laser.wavelength_nm * 1.0e-7;
  CbetLmFields cbet_lm;
  if (state.mesh.dim == 1) {
    build_hydro_mirror_1d(lmesh, state, hydro_mirror);
    map_from_hydro_1d(lmesh, state, laser, hydro_mirror, stream);
  } else {
    cbet_on_2d ? map_from_hydro_2d_cbet(lmesh, state, laser, cbet_lm, stream)
               : map_from_hydro_2d(lmesh, state, laser, stream, &part,
                                   reduction);
  }
  if (phys_ext_options.zeff_model == 2) {
    if (lmesh.zeff_table_dev == nullptr) {
      upload_zeff_table(
          lmesh, laser.ib.zeff_table.ratio.data(),
          laser.ib.zeff_table.ndens * laser.ib.zeff_table.ntemp);
    }
    phys_ext_options.zeff_table = lmesh.zeff_table_dev;
  }
  phys_ext_options.n_crit_cm3 = lmesh.n_crit;
  compute_gradients(lmesh, stream);
  compute_smooth_kappa(lmesh, lambda_cm, laser.absorption.eps_n,
                       laser.absorption.coulomb_log_floor, stream,
                       phys_ext_active ? &phys_ext_options : nullptr);
  static bool debug_dump_lasermesh_emitted = false;
  if (state.mesh.dim == 2 && laser.absorption.debug_dump_lasermesh &&
      !debug_dump_lasermesh_emitted) {
    emit_lasermesh_debug_dump(lmesh, state.step, part.rank, stream);
    debug_dump_lasermesh_emitted = true;
  }
  if (collect_trajectory) {
    const int nn = lmesh.n_nodes();
    state.laser_mesh_n_nodes_r = lmesh.n_nodes_r;
    state.laser_mesh_n_nodes_z = lmesh.n_nodes_z;
    state.laser_mesh_n_crit = lmesh.n_crit;
    state.laser_mesh_node_R.resize(static_cast<std::size_t>(lmesh.n_nodes_r));
    state.laser_mesh_node_Z.resize(static_cast<std::size_t>(lmesh.n_nodes_z));
    state.laser_mesh_n_e_hat.resize(static_cast<std::size_t>(nn));
    state.laser_mesh_T_e.resize(static_cast<std::size_t>(nn));
    state.laser_mesh_Zbar.resize(static_cast<std::size_t>(nn));
    state.laser_mesh_grad_R.resize(static_cast<std::size_t>(nn));
    state.laser_mesh_grad_Z.resize(static_cast<std::size_t>(nn));
    cuda_check(cudaMemcpy(state.laser_mesh_node_R.data(), lmesh.node_R,
                          static_cast<std::size_t>(lmesh.n_nodes_r) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "laser_step memcpy laser mesh node_R D2H failed");
    cuda_check(cudaMemcpy(state.laser_mesh_node_Z.data(), lmesh.node_Z,
                          static_cast<std::size_t>(lmesh.n_nodes_z) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "laser_step memcpy laser mesh node_Z D2H failed");
    cuda_check(cudaMemcpy(state.laser_mesh_n_e_hat.data(), lmesh.n_e_hat,
                          static_cast<std::size_t>(nn) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "laser_step memcpy laser mesh n_e_hat D2H failed");
    cuda_check(cudaMemcpy(state.laser_mesh_T_e.data(), lmesh.T_e,
                          static_cast<std::size_t>(nn) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "laser_step memcpy laser mesh T_e D2H failed");
    cuda_check(cudaMemcpy(state.laser_mesh_Zbar.data(), lmesh.Zbar,
                          static_cast<std::size_t>(nn) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "laser_step memcpy laser mesh Zbar D2H failed");
    cuda_check(cudaMemcpy(state.laser_mesh_grad_R.data(), lmesh.grad_n_hat_R,
                          static_cast<std::size_t>(nn) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "laser_step memcpy laser mesh grad_R D2H failed");
    cuda_check(cudaMemcpy(state.laser_mesh_grad_Z.data(), lmesh.grad_n_hat_Z,
                          static_cast<std::size_t>(nn) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "laser_step memcpy laser mesh grad_Z D2H failed");
  }
  lmesh.clear_deposit(stream);
  t_map_end = Clock::now();

  constexpr int kStepHistSize = LaserMesh::kTraceStepHistSize;
  lmesh.ensure_step_scratch();
  lmesh.clear_step_scratch(stream);
  int* d_step_histogram = verbose ? lmesh.scratch_step_histogram : nullptr;
  double* d_unabsorbed = lmesh.scratch_unabsorbed;
  double* d_ra_power_total = lmesh.scratch_ra_power_total;
  unsigned long long* d_tail_closure_count = lmesh.scratch_tail_closure_count;
  double* d_tail_closure_absorbed_power = lmesh.scratch_tail_closure_absorbed_power;
  unsigned long long* d_critical_surface_hit_count =
      lmesh.scratch_critical_surface_hit_count;
  core::DeviceErrorFlags* d_error_flags = lmesh.scratch_error_flags;
  auto check_or_fail = [&](const cudaError_t err, const char* message) {
    TENRYU_ASSERT(err == cudaSuccess, message);
  };

  static const bool laser_pack_disabled = [] {
    const char* value = std::getenv("TENRYU_DISABLE_LASER_PACK");
    return value != nullptr && std::atoi(value) != 0;
  }();
  const bool laser_pack_enabled = !laser_pack_disabled;
  std::size_t step_pack_cursor = 0;
  std::size_t step_tally_pack_offset = 0;
  std::size_t error_flags_pack_offset = 0;
  std::size_t fold_tally_pack_offset = 0;
  std::size_t radial_capture_pack_offset = 0;
  std::size_t radial_capture_pack_bytes = 0;
  std::vector<PackedStepHistogram> packed_step_histograms;
  std::vector<PackedPerRayStepStats> packed_per_ray_stats;
  std::vector<std::uint8_t> ray_steps_pending(beams.items.size(), 0U);
  bool laser_pack_drained = false;
  if (laser_pack_enabled) {
    const std::size_t trace_slot_count = group_powers.size();
    const std::size_t max_trace_rays =
        (state.mesh.dim == 1)
            ? static_cast<std::size_t>(std::max(laser.rays_per_beam, 1))
            : static_cast<std::size_t>(std::max(laser.rays_per_beam, 2)) *
                  static_cast<std::size_t>(std::max(laser.rays_per_beam, 2));
    const std::size_t max_trace_warps = (max_trace_rays + 31U) / 32U;
    const std::size_t fixed_bytes =
        40U + sizeof(core::DeviceErrorFlags) + 40U +
        3U * static_cast<std::size_t>(HotECaptureParams::kMaxChannels) *
            sizeof(double) +
        64U;
    const std::size_t per_trace_bytes =
        static_cast<std::size_t>(kStepHistSize) * sizeof(int) +
        max_trace_warps *
            (sizeof(int) + sizeof(unsigned long long)) +
        2U * sizeof(int) + 64U;
    const std::size_t pack_capacity =
        fixed_bytes + trace_slot_count * per_trace_bytes;
    lmesh.ensure_step_pack(pack_capacity);
    step_tally_pack_offset = reserve_step_pack_slot(
        step_pack_cursor, 40U, alignof(double),
        lmesh.scratch_step_pack_capacity);
    error_flags_pack_offset = reserve_step_pack_slot(
        step_pack_cursor, sizeof(core::DeviceErrorFlags),
        alignof(core::DeviceErrorFlags), lmesh.scratch_step_pack_capacity);
    fold_tally_pack_offset = reserve_step_pack_slot(
        step_pack_cursor, 40U, alignof(double),
        lmesh.scratch_step_pack_capacity);
    radial_capture_pack_bytes =
        3U * static_cast<std::size_t>(HotECaptureParams::kMaxChannels) *
        sizeof(double);
    radial_capture_pack_offset = reserve_step_pack_slot(
        step_pack_cursor, radial_capture_pack_bytes, alignof(double),
        lmesh.scratch_step_pack_capacity);
  }
  auto drain_laser_pack = [&]() {
    if (!laser_pack_enabled || laser_pack_drained) {
      return;
    }
    const auto t_pack_transfer_start =
        verbose ? Clock::now() : Clock::time_point{};
    check_or_fail(
        cudaMemcpyAsync(lmesh.scratch_step_pack_device +
                            step_tally_pack_offset,
                        lmesh.scratch_step_tally_slab, 40U,
                        cudaMemcpyDeviceToDevice, stream),
        "laser_step pack step tally slab D2D failed");
    check_or_fail(
        cudaMemcpyAsync(lmesh.scratch_step_pack_device +
                            error_flags_pack_offset,
                        d_error_flags, sizeof(core::DeviceErrorFlags),
                        cudaMemcpyDeviceToDevice, stream),
        "laser_step pack error flags D2D failed");
    check_or_fail(
        cudaMemcpyAsync(lmesh.scratch_step_pack_host,
                        lmesh.scratch_step_pack_device, step_pack_cursor,
                        cudaMemcpyDeviceToHost, stream),
        "laser_step scalar pack D2H failed");
    check_or_fail(cudaStreamSynchronize(stream),
                  "laser_step scalar pack stream synchronize failed");
    if (verbose) {
      transfer_ms += ms(t_pack_transfer_start, Clock::now());
    }
    TENRYU_ASSERT(packed_step_histograms.size() ==
                      packed_per_ray_stats.size(),
                  "laser_step packed ray-stat record count mismatch");
    for (std::size_t i = 0; i < packed_step_histograms.size(); ++i) {
      emit_packed_step_histogram(lmesh.scratch_step_pack_host,
                                 packed_step_histograms[i]);
      emit_packed_per_ray_step_stat(lmesh.scratch_step_pack_host,
                                    packed_per_ray_stats[i]);
    }
    laser_pack_drained = true;
  };

  std::vector<double> node_R;
  std::vector<double> node_Z;

  double skipped_unabsorbed_power = 0.0;
  if (state.mesh.dim == 1) {
    const std::vector<std::uint8_t>& cell_is_void =
        hydro_mirror.cell_is_void.empty() ? state.cell_is_void : hydro_mirror.cell_is_void;
    const AllowedSupercriticalCell1D allowed_supercritical =
        (hydro_mirror.rho.size() == state.laser_dep.size() &&
         hydro_mirror.zbar.size() == state.laser_dep.size() &&
         hydro_mirror.A_eff.size() == state.laser_dep.size())
            ? find_allowed_supercritical_cell_1d(cell_is_void, hydro_mirror.rho,
                                                 hydro_mirror.zbar, hydro_mirror.A_eff,
                                                 hydro_mirror.r_edges, lmesh.n_crit)
            : AllowedSupercriticalCell1D{};
    std::vector<double> total_dep_1d(state.laser_dep.size(), 0.0);
    if (radial_absorption_1d) {
      check_or_fail(
          cudaMemsetAsync(state.laser_dep.data(), 0,
                          state.laser_dep.size() * sizeof(double), stream),
          "laser_step memset d_deposit_1d failed before radial_absorption_1d");
      double* d_hot_e_capture_radial = nullptr;
      if (part.n_ranks <= 1 || part.rank == 0) {
        check_or_fail(
            launch_radial_absorption_1d(
                total_power, lmesh, laser, state.x_r.data(),
                static_cast<int>(state.laser_dep.size()), state.laser_dep.data(),
                d_unabsorbed, d_error_flags, d_critical_surface_hit_count, stream,
                hot_e_params, &d_hot_e_capture_radial),
            "laser_step radial_absorption_1d launch failed");
      }
      const auto dep_power_cell = copy_cell_deposit_to_host(state.laser_dep, stream);
      if (hot_e_capture_on && d_hot_e_capture_radial != nullptr) {
        std::vector<double> cap(3 * static_cast<std::size_t>(hot_e_params.n_channels), 0.0);
        if (laser_pack_enabled) {
          const std::size_t capture_bytes = cap.size() * sizeof(double);
          TENRYU_ASSERT(capture_bytes <= radial_capture_pack_bytes,
                        "laser_step radial capture exceeds scalar pack slot");
          check_or_fail(
              cudaMemcpyAsync(lmesh.scratch_step_pack_device +
                                  radial_capture_pack_offset,
                              d_hot_e_capture_radial, capture_bytes,
                              cudaMemcpyDeviceToDevice, stream),
              "laser_step pack hot_e radial capture D2D failed");
          drain_laser_pack();
          std::memcpy(cap.data(),
                      lmesh.scratch_step_pack_host +
                          radial_capture_pack_offset,
                      capture_bytes);
        } else {
          check_or_fail(
              cudaMemcpy(cap.data(), d_hot_e_capture_radial,
                         cap.size() * sizeof(double), cudaMemcpyDeviceToHost),
              "laser_step hot_e radial capture D2H failed");
        }
        for (int k = 0; k < hot_e_params.n_channels; ++k) {
          const double* slot = cap.data() + static_cast<std::size_t>(k) * 3;
          if (slot[0] > 0.5 && slot[2] > 0.0) {
            const ResolvedHotEChannel& rch = hot_e_channels[static_cast<std::size_t>(k)];
            if (hot_e_model) {
              hot_e_model_sum_P[static_cast<std::size_t>(rch.config_index)] +=
                  slot[2];
              hot_e_model_sum_Pr[static_cast<std::size_t>(rch.config_index)] +=
                  slot[1] * slot[2];
            }
            if (rch.eta_eff > 0.0) {
              hot_e_captures_by_channel[static_cast<std::size_t>(rch.config_index)]
                  .push_back(tenryu::laser::hot_electron::RayCapture{
                      slot[1], -1.0, rch.eta_eff * slot[2]});
            }
          }
        }
      }
      for (std::size_t c = 0; c < dep_power_cell.size(); ++c) {
        total_dep_1d[c] += dep_power_cell[c];
      }
    } else {
      CbetWorkspace* cbet_ws = nullptr;
      int cbet_ray_cursor = 0;
      std::vector<int> cbet_beam_offsets(beams.items.size(), 0);
      std::vector<int> cbet_beam_counts(beams.items.size(), 0);
      if (cbet_on) {
        cbet_ws = &global_cbet_workspace();
        const int n_cells_1d = static_cast<int>(state.laser_dep.size());
        // Turning-arc allowance (2026-07-31): the ds_adapt_theta_target
        // limiter resolves turning arcs in ~pi/theta steps; a grazing
        // arc can re-cross one radial face on O(pi/theta) consecutive
        // micro-steps, each producing a (correct) record. The theta
        // floor caps the allowance for tiny user thetas.
        const double theta_t = laser.raytrace.ds_adapt_theta_target;
        const int theta_arc_allowance =
            (theta_t > 0.0)
                ? static_cast<int>(2.0 * M_PI / std::max(theta_t, 1.0e-3))
                : 0;
        const int cap_per_ray = (laser.cbet.max_segments_per_ray > 0)
                                    ? laser.cbet.max_segments_per_ray
                                    : (2 * n_cells_1d + 64 +
                                       theta_arc_allowance);
        const int n_rays_capacity =
            static_cast<int>(beams.items.size()) * std::max(laser.rays_per_beam, 1);
        const auto* const port_state =
            port_section
                ? static_cast<const PortSectionState*>(
                      lmesh.port_section_state.get())
                : nullptr;
        cbet_workspace_prepare(*cbet_ws, n_rays_capacity, cap_per_ray, n_cells_1d,
                               static_cast<int>(beams.items.size()),
                               laser.cbet.n_impact_bins, stream, 2, false, 0, 0,
                               0,
                               port_state != nullptr
                                   ? static_cast<int>(
                                         port_state->ports.ports.size())
                                   : 0,
                               ps_hot_e_capture_on
                                   ? static_cast<int>(hot_e_channels.size())
                                   : 0);
      }
      const bool fold_eligible_config =
          !verbose && !collect_ray_output && !collect_density_diag &&
          !collect_trajectory && !cbet_on && beam_fold_disabled() == false;
      bool fold_enabled = false;
      if (fold_eligible_config) {
        bool have_fold_key = false;
        FoldKey prepass_key{};
        fold_enabled = true;
        for (std::size_t b = 0; b < beams.items.size(); ++b) {
          const double P_beam = group_powers[b];
          if (!(P_beam > 0.0)) {
            continue;
          }
          const FoldKey key = make_fold_key(beams.items[b], P_beam);
          if (!have_fold_key) {
            prepass_key = key;
            have_fold_key = true;
          } else if (!fold_keys_equal(key, prepass_key)) {
            fold_enabled = false;
            break;
          }
        }
        fold_enabled = fold_enabled && have_fold_key;
      }

      bool fold_armed = false;
      FoldKey fold_key{};
      std::vector<double> fold_dep;
      double fold_d_unabsorbed = 0.0;
      double fold_d_ra_power = 0.0;
      unsigned long long fold_d_tail_count = 0ULL;
      double fold_d_tail_power = 0.0;
      unsigned long long fold_d_crit_hits = 0ULL;
      int fold_replays = 0;
      lmesh.ray_steps_previous.resize(beams.items.size());
      lmesh.ray_steps_output.resize(beams.items.size());
      lmesh.ray_order.resize(beams.items.size());
      hot_e_capture_stage.assign(static_cast<std::size_t>(beams.items.size()) *
                                     static_cast<std::size_t>(laser.rays_per_beam) * 4 *
                                     static_cast<std::size_t>(hot_e_params.n_channels),
                                 0.0);
      double* d_traj_pos1 = nullptr;
      double* d_traj_pos2 = nullptr;
      double* d_traj_power = nullptr;
      std::int32_t* d_traj_rec_idx = nullptr;
      int* d_traj_step_count = nullptr;
      int n_output_rays_traj = 0;
      int traj_output_stride = 1;
      const int traj_max_steps = laser.ray_output_max_steps;
      const bool tau_diag_this_step =
          trace_tau_diag_enabled() && part.rank == 0 &&
          state.step <= trace_tau_diag_max_step();
      long long traj_ray_offset = 0;
      int traj_beam_id = 0;
      auto release_traj_buffers = [&]() {
        if (d_traj_pos1 != nullptr) {
          static_cast<void>(cudaFree(d_traj_pos1));
          d_traj_pos1 = nullptr;
        }
        if (d_traj_pos2 != nullptr) {
          static_cast<void>(cudaFree(d_traj_pos2));
          d_traj_pos2 = nullptr;
        }
        if (d_traj_power != nullptr) {
          static_cast<void>(cudaFree(d_traj_power));
          d_traj_power = nullptr;
        }
        if (d_traj_rec_idx != nullptr) {
          static_cast<void>(cudaFree(d_traj_rec_idx));
          d_traj_rec_idx = nullptr;
        }
        if (d_traj_step_count != nullptr) {
          static_cast<void>(cudaFree(d_traj_step_count));
          d_traj_step_count = nullptr;
        }
      };
      auto check_traj_or_cleanup = [&](const cudaError_t err, const char* message) {
        if (err == cudaSuccess) {
          return;
        }
        release_traj_buffers();
        check_or_fail(err, message);
      };
    for (std::size_t b = 0; b < beams.items.size(); ++b) {
      const Beam& beam = beams.items[b];
      const double P_beam = group_powers[b];
      if (!(P_beam > 0.0)) {
        continue;
      }
      if (fold_enabled && fold_armed &&
          fold_keys_equal(make_fold_key(beam, P_beam), fold_key)) {
        for (std::size_t c = 0; c < fold_dep.size(); ++c) {
          total_dep_1d[c] += fold_dep[c];
        }
        for (std::size_t c = 0; c < fold_dep.size(); ++c) {
          f_hat_groups[b][c] = fold_dep[c] / std::max(P_beam, 1.0e-30);
        }
        ++fold_replays;
        continue;
      }
      lmesh.clear_deposit(stream);
      const auto t_init_start = verbose ? Clock::now() : Clock::time_point{};
      RayArray1D rays = initialize_rays_1d(beam, lmesh, laser.rays_per_beam, P_beam, stream);
      if (verbose) {
        init_ms += ms(t_init_start, Clock::now());
      }
      if (rays.empty()) {
        skipped_unabsorbed_power += P_beam;
        continue;
      }
      const int n_radial_intervals = lmesh.radial_n_nodes - 1;
      double* d_tau_shell_out = nullptr;
      double* d_pabs_per_ray_out = nullptr;
      std::size_t tau_diag_doubles = 0;
      if (tau_diag_this_step) {
        tau_diag_doubles = static_cast<std::size_t>(rays.n_rays) *
                           static_cast<std::size_t>(n_radial_intervals);
        d_tau_shell_out = static_cast<double*>(core::device_scratch_acquire(
            "laser:tau_shell_diag", tau_diag_doubles * sizeof(double)));
        check_traj_or_cleanup(
            cudaMemsetAsync(d_tau_shell_out, 0,
                            tau_diag_doubles * sizeof(double), stream),
            "laser_step memset tau-shell diagnostic failed");
        d_pabs_per_ray_out = static_cast<double*>(core::device_scratch_acquire(
            "laser:pabs_per_ray_diag",
            static_cast<std::size_t>(rays.n_rays) * sizeof(double)));
        check_traj_or_cleanup(
            cudaMemsetAsync(d_pabs_per_ray_out, 0,
                            static_cast<std::size_t>(rays.n_rays) * sizeof(double),
                            stream),
            "laser_step memset pabs-per-ray diagnostic failed");
      }
      if (collect_density_diag) {
        const auto t_density_diag_start = verbose ? Clock::now() : Clock::time_point{};
        accumulate_ray_density_1d(rays, hydro_mirror.r_edges, ray_counts, stream);
        if (verbose) {
          density_diag_ms += ms(t_density_diag_start, Clock::now());
        }
      }
      if (collect_ray_output) {
        const auto t_capture_start = verbose ? Clock::now() : Clock::time_point{};
        const int n_copy = std::min(ray_record_cap, rays.n_rays);
        const int output_stride = std::max(1, rays.n_rays / std::max(1, n_copy));
        capture_ray_output_1d(rays, n_copy, output_stride, beam.wave_id, ray_output, stream);
        if (verbose) {
          capture_ms += ms(t_capture_start, Clock::now());
        }
      }
      n_output_rays_traj = collect_trajectory
          ? std::min(ray_record_cap, rays.n_rays) : 0;
      traj_output_stride = (n_output_rays_traj > 0)
          ? std::max(1, rays.n_rays / std::max(1, n_output_rays_traj)) : 1;
      traj_beam_id = beam.wave_id;
      if (n_output_rays_traj > 0) {
        const std::size_t traj_buf_size =
            static_cast<std::size_t>(n_output_rays_traj) * static_cast<std::size_t>(traj_max_steps);
        check_traj_or_cleanup(cudaMalloc(reinterpret_cast<void**>(&d_traj_pos1),
                                         traj_buf_size * sizeof(double)),
                              "laser_step cudaMalloc traj_pos1 failed");
        check_traj_or_cleanup(cudaMalloc(reinterpret_cast<void**>(&d_traj_pos2),
                                         traj_buf_size * sizeof(double)),
                              "laser_step cudaMalloc traj_pos2 failed");
        check_traj_or_cleanup(cudaMalloc(reinterpret_cast<void**>(&d_traj_power),
                                         traj_buf_size * sizeof(double)),
                              "laser_step cudaMalloc traj_power failed");
        if (port_section) {
          check_traj_or_cleanup(
              cudaMalloc(reinterpret_cast<void**>(&d_traj_rec_idx),
                         traj_buf_size * sizeof(std::int32_t)),
              "laser_step cudaMalloc traj_rec_idx failed");
        }
        check_traj_or_cleanup(cudaMalloc(reinterpret_cast<void**>(&d_traj_step_count),
                                         static_cast<std::size_t>(n_output_rays_traj) * sizeof(int)),
                              "laser_step cudaMalloc traj_step_count failed");
      }
      const auto t_memset_start = verbose ? Clock::now() : Clock::time_point{};
      if (d_step_histogram != nullptr) {
        check_traj_or_cleanup(cudaMemsetAsync(
                                  d_step_histogram, 0,
                                  static_cast<std::size_t>(kStepHistSize) * sizeof(int), stream),
                              "laser_step memset d_step_histogram failed before ray_trace_1d_sph");
      }
      check_traj_or_cleanup(
          cudaMemsetAsync(state.laser_dep.data(), 0,
                          state.laser_dep.size() * sizeof(double), stream),
          "laser_step memset d_deposit_1d failed before ray_trace_1d_sph");
      if (verbose) {
        memset_ms += ms(t_memset_start, Clock::now());
      }

      if (verbose && laser.mode != "radial_absorption_1d" && rays.n_rays > 0) {
        lmesh.ensure_per_ray_step_scratch(rays.n_rays);
      }
      CbetRecordDeviceArgs cbet_rec_args;
      if (cbet_on) {
        constexpr double kPiCbet = 3.14159265358979323846;
        const double lambda_b_cm =
            (laser.wavelength_nm + beam.delta_lambda_nm) * 1.0e-7;
        const double omega_b =
            2.0 * kPiCbet * core::constants::c_light / lambda_b_cm;
        cbet_stage_ray_meta(*cbet_ws, static_cast<int>(b), cbet_ray_cursor,
                            rays.n_rays, omega_b, rays.power, stream);
        cbet_beam_offsets[b] = cbet_ray_cursor;
        cbet_beam_counts[b] = rays.n_rays;
        cbet_rec_args.rec_cell = cbet_ws->rec_cell;
        cbet_rec_args.rec_mu = cbet_ws->rec_mu;
        cbet_rec_args.rec_ds = cbet_ws->rec_ds;
        cbet_rec_args.rec_S = cbet_ws->rec_S;
        cbet_rec_args.rec_w = cbet_ws->rec_w;
        cbet_rec_args.traj_rec_idx = d_traj_rec_idx;
        cbet_rec_args.rec_count = cbet_ws->rec_count;
        cbet_rec_args.ray_overflow = cbet_ws->ray_overflow;
        cbet_rec_args.ray_offset = cbet_ray_cursor;
        cbet_rec_args.cap_per_ray = cbet_ws->cap_per_ray;
        if (port_section && n_output_rays_traj > 0) {
          traj_ray_offset = cbet_ray_cursor;
        }
        cbet_ray_cursor += rays.n_rays;
      }
      if (phys_ext_options.ra_enable != 0) {
        phys_ext_options.ra_r_crit_cm = -1.0;
        phys_ext_options.ra_ln_cm = -1.0;
        const int n_cells =
            static_cast<int>(hydro_mirror.rho.size());
        std::vector<double> n_hat_cell(
            static_cast<std::size_t>(n_cells), 0.0);
        const double n_crit_safe = std::max(lmesh.n_crit, 1.0e-30);
        for (int c = 0; c < n_cells; ++c) {
          const std::size_t s = static_cast<std::size_t>(c);
          if (cell_is_void[s] != 0U) {
            continue;
          }
          const double A_eff =
              std::max(hydro_mirror.A_eff[s], 1.0e-30);
          const double n_e =
              std::max(0.0, hydro_mirror.rho[s]) *
              std::max(0.0, hydro_mirror.zbar[s]) /
              (A_eff * core::constants::proton_mass);
          n_hat_cell[s] = std::max(0.0, n_e / n_crit_safe);
        }
        int c_out = -1;
        for (int c = n_cells - 2; c >= 0; --c) {
          if (n_hat_cell[static_cast<std::size_t>(c)] >= 1.0 &&
              n_hat_cell[static_cast<std::size_t>(c + 1)] < 1.0) {
            c_out = c;
            break;
          }
        }
        if (c_out >= 0) {
          const std::size_t inner = static_cast<std::size_t>(c_out);
          const std::size_t outer = static_cast<std::size_t>(c_out + 1);
          const double r_inner =
              0.5 * (hydro_mirror.r_edges[inner] +
                     hydro_mirror.r_edges[inner + 1U]);
          const double r_outer =
              0.5 * (hydro_mirror.r_edges[outer] +
                     hydro_mirror.r_edges[outer + 1U]);
          const double n_inner = n_hat_cell[inner];
          const double n_outer = n_hat_cell[outer];
          const double alpha =
              (1.0 - n_inner) / (n_outer - n_inner);
          phys_ext_options.ra_r_crit_cm =
              r_inner + alpha * (r_outer - r_inner);
          const double dln =
              std::log(std::max(n_outer, 1.0e-12)) -
              std::log(std::max(n_inner, 1.0e-12));
          phys_ext_options.ra_ln_cm =
              std::clamp(std::abs((r_outer - r_inner) / dln),
                         1.0e-5, 1.0);
        }
      }
      const double beam_P_w =
          (phys_ext_options.langdon_model != 0) ? P_beam * 1.0e-7 : 0.0;
      const double beam_w_cm =
          (phys_ext_options.langdon_model != 0)
              ? beam.profile_w0_cm / std::sqrt(2.0)
              : 0.0;
      const int beam_profile_flat =
          (beam.profile_model == "flat_top") ? 1 : 0;
      // Opt-in until the S1.1 physics-accuracy pass closes the march parity gap;
      // see perf wave-3 notes.
      static const bool fast_trace_enabled = [] {
        const char* s = std::getenv("TENRYU_FAST_TRACE");
        return s != nullptr && std::strcmp(s, "1") == 0;
      }();
      const bool use_fast_trace_1d =
          fast_trace_enabled && state.mesh.dim == 1 && laser.mode == "raytrace_2d" &&
          !laser.cbet.enable && beam.profile_model == "flat_top" &&
          !hot_e_capture_on && !collect_trajectory && !collect_ray_output;
      static const bool logged_fast_gate = [&] {
        core::log_info(std::string("[fast_trace_gate] use=") +
            (use_fast_trace_1d ? "1" : "0") +
            " dim1=" + (state.mesh.dim == 1 ? "1" : "0") +
            " mode_rt2d=" + (laser.mode == "raytrace_2d" ? "1" : "0") +
            " cbet=" + (laser.cbet.enable ? "1" : "0") +
            " flat=" + (beam.profile_model == "flat_top" ? "1" : "0") +
            " hote=" + (hot_e_capture_on ? "1" : "0") +
            " traj=" + (collect_trajectory ? "1" : "0") +
            " rayout=" + (collect_ray_output ? "1" : "0") +
            " fastenv=" + (fast_trace_enabled ? "1" : "0"));
        return true;
      }();
      (void)logged_fast_gate;
      const int* h_ray_order = nullptr;
      int* h_ray_steps_out = nullptr;
      int max_ray_steps_override = 0;
      if (!use_fast_trace_1d) {
        auto& previous_steps = lmesh.ray_steps_previous[b];
        auto& ray_order = lmesh.ray_order[b];
        auto& steps_output = lmesh.ray_steps_output[b];
        static const bool spike_cap_disabled = [] {
          const char* s = std::getenv("TENRYU_LASER_NO_SPIKE_CAP");
          return s != nullptr && std::strcmp(s, "1") == 0;
        }();
        if (!spike_cap_disabled && !previous_steps.empty() &&
            previous_steps.size() == static_cast<std::size_t>(rays.n_rays)) {
          std::vector<int> step_counts_scratch(previous_steps);
          const std::size_t p90_idx =
              ((step_counts_scratch.size() - 1) * 9) / 10;
          std::nth_element(step_counts_scratch.begin(),
                           step_counts_scratch.begin() +
                               static_cast<std::ptrdiff_t>(p90_idx),
                           step_counts_scratch.end());
          const int p90 = step_counts_scratch[p90_idx];
          const int dynamic_cap = std::max(20000, 10 * p90);
          max_ray_steps_override =
              std::min(laser.raytrace.max_steps, dynamic_cap);
        }
        ray_order.resize(static_cast<std::size_t>(rays.n_rays));
        std::iota(ray_order.begin(), ray_order.end(), 0);
        if (!ray_sort_disabled() &&
            previous_steps.size() == static_cast<std::size_t>(rays.n_rays)) {
          std::stable_sort(
              ray_order.begin(), ray_order.end(),
              [&](const int lhs, const int rhs) {
                return previous_steps[static_cast<std::size_t>(lhs)] >
                       previous_steps[static_cast<std::size_t>(rhs)];
              });
        }
        steps_output.resize(static_cast<std::size_t>(rays.n_rays));
        h_ray_order = ray_order.data();
        h_ray_steps_out = steps_output.data();
      }
      const auto t_kernel_start = verbose ? Clock::now() : Clock::time_point{};
      double* d_hot_e_capture_rays = nullptr;
      const cudaError_t trace_status = use_fast_trace_1d
          ? launch_fast_trace_1d(
              rays, lmesh, laser, lambda_cm, state.x_r.data(),
              static_cast<int>(state.laser_dep.size()), allowed_supercritical.allowed_cell,
              allowed_supercritical.critical_adjacent_subcritical_cell,
              allowed_supercritical.r_crit, state.laser_dep.data(), d_unabsorbed,
              d_error_flags, d_critical_surface_hit_count, stream, d_step_histogram,
              verbose ? lmesh.scratch_per_ray_step_count : nullptr,
              d_traj_step_count, n_output_rays_traj,
              phys_ext_active ? &phys_ext_options : nullptr,
              phys_ext_active ? lmesh.radial_T_e : nullptr,
              beam_P_w, beam_w_cm,
              phys_ext_active && phys_ext_options.ra_enable != 0
                  ? d_ra_power_total
                  : nullptr,
              d_tau_shell_out, d_pabs_per_ray_out)
          : launch_ray_trace_1d_sph(
              rays, lmesh, laser, lambda_cm, state.x_r.data(),
              static_cast<int>(state.laser_dep.size()), allowed_supercritical.allowed_cell,
              allowed_supercritical.critical_adjacent_subcritical_cell,
              allowed_supercritical.r_crit, state.laser_dep.data(), d_traj_pos1, d_traj_pos2,
              nullptr, d_traj_power, d_traj_step_count, n_output_rays_traj, traj_output_stride,
              traj_max_steps, cbet_on ? (cbet_ws->d_scalars + 6) : d_unabsorbed, d_error_flags,
              d_tail_closure_count,
              cbet_on ? (cbet_ws->d_scalars + 7) : d_tail_closure_absorbed_power,
              d_critical_surface_hit_count, stream, d_step_histogram,
              verbose ? lmesh.scratch_per_ray_step_count : nullptr,
              cbet_on ? &cbet_rec_args : nullptr, hot_e_params, &d_hot_e_capture_rays,
              phys_ext_active ? &phys_ext_options : nullptr,
              phys_ext_active ? lmesh.radial_T_e : nullptr,
              beam_P_w, beam_w_cm, beam_profile_flat,
              phys_ext_active && phys_ext_options.ra_enable != 0
                  ? d_ra_power_total
                  : nullptr,
              h_ray_order, h_ray_steps_out, max_ray_steps_override,
              d_tau_shell_out, d_pabs_per_ray_out);
      check_traj_or_cleanup(trace_status,
                            "laser_step 1D spherical trace launch failed");
      if (d_tau_shell_out != nullptr) {
        std::vector<double> tau_shell(tau_diag_doubles, 0.0);
        check_traj_or_cleanup(
            cudaMemcpyAsync(tau_shell.data(), d_tau_shell_out,
                            tau_diag_doubles * sizeof(double),
                            cudaMemcpyDeviceToHost, stream),
            "laser_step tau-shell diagnostic D2H failed");
        check_traj_or_cleanup(
            cudaStreamSynchronize(stream),
            "laser_step tau-shell diagnostic synchronize failed");
        write_tau_diag_dump(output_dir,
                            use_fast_trace_1d ? "fast" : "march",
                            state.step, b, rays.n_rays, n_radial_intervals,
                            tau_shell, use_fast_trace_1d);
      }
      if (d_pabs_per_ray_out != nullptr) {
        std::vector<double> pabs_per_ray(static_cast<std::size_t>(rays.n_rays),
                                         0.0);
        check_traj_or_cleanup(
            cudaMemcpyAsync(pabs_per_ray.data(), d_pabs_per_ray_out,
                            pabs_per_ray.size() * sizeof(double),
                            cudaMemcpyDeviceToHost, stream),
            "laser_step pabs-per-ray diagnostic D2H failed");
        check_traj_or_cleanup(
            cudaStreamSynchronize(stream),
            "laser_step pabs-per-ray diagnostic synchronize failed");
        write_pabs_diag_dump(output_dir,
                             use_fast_trace_1d ? "fast" : "march",
                             state.step, b, pabs_per_ray);
      }
      if (!use_fast_trace_1d) {
        ray_steps_pending[b] = 1U;
      }
      if (hot_e_capture_on && d_hot_e_capture_rays != nullptr) {
        const std::size_t stage_row_doubles =
            static_cast<std::size_t>(laser.rays_per_beam) * 4 *
            static_cast<std::size_t>(hot_e_params.n_channels);
        const std::size_t copy_doubles =
            static_cast<std::size_t>(rays.n_rays) * 4 *
            static_cast<std::size_t>(hot_e_params.n_channels);
        const std::size_t off = static_cast<std::size_t>(b) * stage_row_doubles;
        check_or_fail(cudaMemcpyAsync(hot_e_capture_stage.data() + off,
                                      d_hot_e_capture_rays,
                                      copy_doubles * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream),
                      "laser_step hot_e capture async D2H failed");
        ++hot_e_capture_beams;
      }
      if (verbose) {
        check_traj_or_cleanup(cudaStreamSynchronize(stream),
                              "laser_step stream synchronize failed after ray_trace_1d_sph timing");
        kernel_ms += ms(t_kernel_start, Clock::now());
      }
      const auto t_trace_transfer_start = verbose ? Clock::now() : Clock::time_point{};
      if (verbose && d_step_histogram != nullptr) {
        if (laser_pack_enabled) {
          PackedStepHistogram packed;
          packed.offset = reserve_step_pack_slot(
              step_pack_cursor,
              static_cast<std::size_t>(kStepHistSize) * sizeof(int),
              alignof(int), lmesh.scratch_step_pack_capacity);
          packed.step = state.step;
          packed.beam_id = beam.wave_id;
          packed.n_rays = rays.n_rays;
          check_traj_or_cleanup(
              cudaMemcpyAsync(
                  lmesh.scratch_step_pack_device + packed.offset,
                  d_step_histogram,
                  static_cast<std::size_t>(kStepHistSize) * sizeof(int),
                  cudaMemcpyDeviceToDevice, stream),
              "laser_step pack step_histogram D2D failed after ray_trace_1d_sph");
          packed_step_histograms.push_back(packed);
        } else {
          std::array<int, kStepHistSize> h_step_hist{};
          check_traj_or_cleanup(
              cudaMemcpyAsync(
                  h_step_hist.data(), d_step_histogram,
                  static_cast<std::size_t>(kStepHistSize) * sizeof(int),
                  cudaMemcpyDeviceToHost, stream),
              "laser_step memcpy step_histogram D2H failed after ray_trace_1d_sph");
          check_traj_or_cleanup(
              cudaStreamSynchronize(stream),
              "laser_step stream synchronize failed after ray_trace_1d_sph stats");
          const double mean_steps =
              (rays.n_rays > 0)
                  ? (static_cast<double>(h_step_hist[0]) /
                     static_cast<double>(rays.n_rays))
                  : 0.0;
          core::log_info(
              "[laser_ray_stats] step=" + std::to_string(state.step) +
              " beam=" + std::to_string(beam.wave_id) +
              " n_rays=" + std::to_string(rays.n_rays) +
              " total_steps=" + std::to_string(h_step_hist[0]) +
              " mean=" + std::to_string(mean_steps) +
              " b100=" + std::to_string(h_step_hist[1]) +
              " b1k=" + std::to_string(h_step_hist[2]) +
              " b10k=" + std::to_string(h_step_hist[3]) +
              " bmax=" + std::to_string(h_step_hist[4]));
        }
      }
      if (verbose) {
        transfer_ms += ms(t_trace_transfer_start, Clock::now());
      }

      if (verbose && laser.mode != "radial_absorption_1d" && rays.n_rays > 0) {
        if (laser_pack_enabled) {
          packed_per_ray_stats.push_back(stage_per_ray_step_stats(
              lmesh, rays.n_rays, state.step, b, stream,
              lmesh.scratch_step_pack_device, step_pack_cursor,
              lmesh.scratch_step_pack_capacity, check_traj_or_cleanup));
        } else {
          emit_per_ray_step_stats(lmesh, rays.n_rays, state.step, b, stream,
                                  check_traj_or_cleanup, ms, transfer_ms);
        }
      }
      const auto t_payload_transfer_start = verbose ? Clock::now() : Clock::time_point{};
      if (!port_section && n_output_rays_traj > 0) {
        std::vector<int> h_step_counts(static_cast<std::size_t>(n_output_rays_traj));
        check_traj_or_cleanup(cudaMemcpy(h_step_counts.data(), d_traj_step_count,
                                         static_cast<std::size_t>(n_output_rays_traj) * sizeof(int),
                                         cudaMemcpyDeviceToHost),
                              "laser_step memcpy traj_step_count D2H failed");

        std::int64_t beam_total = 0;
        for (int i = 0; i < n_output_rays_traj; ++i) {
          beam_total += static_cast<std::int64_t>(h_step_counts[static_cast<std::size_t>(i)]);
        }

        if (beam_total > 0) {
          const std::size_t traj_buf_size =
              static_cast<std::size_t>(n_output_rays_traj) * static_cast<std::size_t>(traj_max_steps);
          std::vector<double> h_pos1(traj_buf_size);
          std::vector<double> h_pos2(traj_buf_size);
          std::vector<double> h_power(traj_buf_size);
          check_traj_or_cleanup(cudaMemcpy(h_pos1.data(), d_traj_pos1,
                                           traj_buf_size * sizeof(double), cudaMemcpyDeviceToHost),
                                "laser_step memcpy traj_pos1 D2H failed");
          check_traj_or_cleanup(cudaMemcpy(h_pos2.data(), d_traj_pos2,
                                           traj_buf_size * sizeof(double), cudaMemcpyDeviceToHost),
                                "laser_step memcpy traj_pos2 D2H failed");
          check_traj_or_cleanup(cudaMemcpy(h_power.data(), d_traj_power,
                                           traj_buf_size * sizeof(double), cudaMemcpyDeviceToHost),
                                "laser_step memcpy traj_power D2H failed");

          const bool first_beam = state.ray_traj_offsets.empty();
          if (first_beam) {
            state.ray_traj_offsets.push_back(0);
          }
          for (int i = 0; i < n_output_rays_traj; ++i) {
            const int sc = h_step_counts[static_cast<std::size_t>(i)];
            const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(traj_max_steps);
            if (collect_ray_output && ray_output != nullptr && sc > 0) {
              ray_output->back().rays_2d[static_cast<std::size_t>(i)].power =
                  h_power[base + static_cast<std::size_t>(sc - 1)];
            }
            for (int s = 0; s < sc; ++s) {
              state.ray_traj_pos1.push_back(h_pos1[base + static_cast<std::size_t>(s)]);
              state.ray_traj_pos2.push_back(h_pos2[base + static_cast<std::size_t>(s)]);
              state.ray_traj_power.push_back(h_power[base + static_cast<std::size_t>(s)]);
            }
            state.ray_traj_step_counts.push_back(static_cast<std::int32_t>(sc));
            state.ray_traj_beam_ids.push_back(static_cast<std::int32_t>(beam.wave_id));
            state.ray_traj_offsets.push_back(
                static_cast<std::int64_t>(state.ray_traj_pos1.size()));
          }
        }
      }
      if (!port_section) {
        release_traj_buffers();
      }
      if (!cbet_on) {
        const auto dep_power_cell = copy_cell_deposit_to_host(state.laser_dep, stream);
        if (verbose) {
          transfer_ms += ms(t_payload_transfer_start, Clock::now());
        }
        for (std::size_t c = 0; c < dep_power_cell.size(); ++c) {
          total_dep_1d[c] += dep_power_cell[c];
        }
        for (std::size_t c = 0; c < dep_power_cell.size(); ++c) {
          f_hat_groups[b][c] = dep_power_cell[c] / std::max(P_beam, 1.0e-30);
        }
        if (fold_enabled && !fold_armed) {
          fold_dep = dep_power_cell;
          if (laser_pack_enabled) {
            check_or_fail(
                cudaMemcpyAsync(lmesh.scratch_step_pack_device +
                                    fold_tally_pack_offset,
                                lmesh.scratch_step_tally_slab, 40U,
                                cudaMemcpyDeviceToDevice, stream),
                "laser_step fold cache pack step tally slab D2D failed");
          } else {
            alignas(8) unsigned char step_tally_staging[40];
            check_or_fail(
                cudaMemcpyAsync(step_tally_staging,
                                lmesh.scratch_step_tally_slab, 40,
                                cudaMemcpyDeviceToHost, stream),
                "laser_step fold cache read step tally slab failed");
            check_or_fail(cudaStreamSynchronize(stream),
                          "laser_step fold cache sync failed");
            std::memcpy(&fold_d_unabsorbed, step_tally_staging + 0,
                        sizeof(double));
            std::memcpy(&fold_d_tail_count, step_tally_staging + 8,
                        sizeof(unsigned long long));
            std::memcpy(&fold_d_tail_power, step_tally_staging + 16,
                        sizeof(double));
            std::memcpy(&fold_d_crit_hits, step_tally_staging + 24,
                        sizeof(unsigned long long));
            std::memcpy(&fold_d_ra_power, step_tally_staging + 32,
                        sizeof(double));
          }
          fold_key = make_fold_key(beam, P_beam);
          fold_armed = true;
        }
      } else if (verbose) {
        transfer_ms += ms(t_payload_transfer_start, Clock::now());
      }
    }
    if (cbet_on) {
      cbet_stage_cell_fields(*cbet_ws, state, hydro_mirror, laser, lambda_cm, stream);
      if (port_section) {
        const auto t_ps_s1_start = Clock::now();
        build_port_section_diagnostics(
            lmesh, beams.items.front(), hydro_mirror, *cbet_ws,
            cbet_ray_cursor, verbose);
        const auto* const port_state =
            static_cast<const PortSectionState*>(
                lmesh.port_section_state.get());
        TENRYU_ASSERT(port_state != nullptr,
                      "port_section ray map requires initialized host state");
        fill_port_section_ray_map(
            state, port_state->table, hydro_mirror.r_edges);
        ps_s1_ms += ms(t_ps_s1_start, Clock::now());
      }
      CbetSolveResult cbet_res;
      if (port_section) {
        auto* const port_state =
            static_cast<PortSectionState*>(
                lmesh.port_section_state.get());
        TENRYU_ASSERT(port_state != nullptr,
                      "port_section solve requires initialized host state");
        const int n_cells = cbet_ws->n_cells;
        const auto t_ps_chi_d2h_start = Clock::now();
        ps_chi_d2h_ms += ms(t_ps_chi_d2h_start, Clock::now());

        const ::tenryu::laser::port_section::ChiBuildInput chi_input{
            &port_state->ports,
            &port_state->table,
            laser.cbet.n_impact_bins,
            port_state->ray_bin.data(),
            static_cast<int>(port_state->ray_bin.size()),
            n_cells,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            laser.cbet.f_cbet,
            laser.cbet.alpha_iaw,
            laser.cbet.k_a_floor,
            laser.cbet.n_section_phi,
            laser.wavelength_nm};
        const ::tenryu::laser::port_section::ChiDeviceCellFields dev_fields{
            cbet_ws->cell_chi_pref,
            cbet_ws->cell_c_a,
            cbet_ws->cell_u_r,
            cbet_ws->cell_k_bar,
            cbet_ws->cell_mask};
        if (!port_state->chi_ws) {
          port_state->chi_ws = std::make_unique<
              ::tenryu::laser::port_section::ChiDeviceWorkspace>();
        }
        const auto t_ps_chi_build_start = Clock::now();
        const auto chi_view =
            ::tenryu::laser::port_section::build_chi_ps_device_ws(
                chi_input, dev_fields, *port_state->chi_ws, stream);
        ps_chi_build_ms += ms(t_ps_chi_build_start, Clock::now());
        const auto t_ps_chi_post_start = Clock::now();
        port_state->audit.chi_abs_max = chi_view.audit.chi_abs_max;
        port_state->audit.chi_nonzero =
            static_cast<long long>(chi_view.audit.chi_nonzero);
        port_state->audit.pairs_with_seed =
            chi_view.pairs_with_seed;
        port_state->audit.pairs_with_pump =
            chi_view.pairs_with_pump;
        TENRYU_ASSERT(chi_view.G_ps == cbet_ws->n_groups &&
                          chi_view.n_pairs == cbet_ws->n_pairs,
                      "port_section chi/workspace size mismatch");

        if (!port_state->ps_static_built) {
          port_state->port_weight.reserve(
              port_state->ports.ports.size());
          for (const port_geom::Port& port :
               port_state->ports.ports) {
            port_state->port_weight.push_back(port.power_weight);
          }
          port_state->pair_p.reserve(
              static_cast<std::size_t>(chi_view.n_pairs));
          port_state->pair_q.reserve(
              static_cast<std::size_t>(chi_view.n_pairs));
          port_state->pair_index.assign(
              static_cast<std::size_t>(chi_view.G_ps) * chi_view.G_ps,
              -1);
          int pair = 0;
          for (int p = 0; p < chi_view.G_ps; ++p) {
            for (int q = p + 1; q < chi_view.G_ps; ++q) {
              port_state->pair_p.push_back(static_cast<std::int16_t>(p));
              port_state->pair_q.push_back(static_cast<std::int16_t>(q));
              port_state->pair_index[
                  static_cast<std::size_t>(p) * chi_view.G_ps + q] =
                  pair;
              port_state->pair_index[
                  static_cast<std::size_t>(q) * chi_view.G_ps + p] =
                  pair;
              ++pair;
            }
          }
          TENRYU_ASSERT(pair == chi_view.n_pairs,
                        "port_section pair table construction mismatch");
          if (ps_hot_e_capture_on) {
            port_state->ps_capture_thresh.assign(
                hot_e_channels.size(), 0.0);
            port_state->ps_capture_order.resize(hot_e_channels.size());
            for (std::size_t k = 0; k < hot_e_channels.size(); ++k) {
              const ResolvedHotEChannel& channel = hot_e_channels[k];
              port_state->ps_capture_thresh[k] = channel.f_s;
              port_state->ps_capture_order[k] =
                  static_cast<std::int32_t>(channel.config_index);
            }
          }
          port_state->ps_static_built = true;
        }
        std::vector<double> ps_one_minus_eta;
        if (ps_hot_e_capture_on) {
          TENRYU_ASSERT(
              cbet_ws->ps_n_channels ==
                  static_cast<int>(hot_e_channels.size()),
              "port_section hot-e channel/workspace size mismatch");
          TENRYU_ASSERT(
              hot_e_model_cell_nhat.size() ==
                  static_cast<std::size_t>(n_cells),
              "port_section hot-e cell n_hat size mismatch");
          ps_one_minus_eta.resize(hot_e_channels.size(), 1.0);
          for (std::size_t k = 0; k < hot_e_channels.size(); ++k) {
            const ResolvedHotEChannel& channel = hot_e_channels[k];
            ps_one_minus_eta[k] =
                laser.hot_electron.subtract_from_laser
                    ? (1.0 - channel.eta_eff)
                    : 1.0;
          }
          cuda_check(cudaMemcpyAsync(
                         cbet_ws->ps_cell_nhat,
                         hot_e_model_cell_nhat.data(),
                         hot_e_model_cell_nhat.size() * sizeof(double),
                         cudaMemcpyHostToDevice, stream),
                     "port_section hot-e cell n_hat H2D failed");
          cuda_check(cudaMemcpyAsync(
                         cbet_ws->ps_capture_thresh,
                         port_state->ps_capture_thresh.data(),
                         port_state->ps_capture_thresh.size() *
                             sizeof(double),
                         cudaMemcpyHostToDevice, stream),
                     "port_section hot-e capture threshold H2D failed");
          cuda_check(cudaMemcpyAsync(
                         cbet_ws->ps_capture_order,
                         port_state->ps_capture_order.data(),
                         port_state->ps_capture_order.size() *
                             sizeof(std::int32_t),
                         cudaMemcpyHostToDevice, stream),
                     "port_section hot-e capture order H2D failed");
          cuda_check(cudaMemcpyAsync(
                         cbet_ws->ps_one_minus_eta,
                         ps_one_minus_eta.data(),
                         ps_one_minus_eta.size() * sizeof(double),
                         cudaMemcpyHostToDevice, stream),
                     "port_section hot-e one-minus-eta H2D failed");
          cuda_check(
              cudaStreamSynchronize(stream),
              "port_section hot-e capture parameter stage sync failed");
        }
        double* d_traj_rec_ratio = nullptr;
        if (n_output_rays_traj > 0 && d_traj_power != nullptr &&
            d_traj_rec_idx != nullptr && d_traj_step_count != nullptr) {
          const std::size_t ratio_count =
              static_cast<std::size_t>(n_output_rays_traj) *
              static_cast<std::size_t>(cbet_ws->cap_per_ray);
          d_traj_rec_ratio = static_cast<double*>(
              core::device_scratch_acquire(
                  "laser:traj_rec_ratio", ratio_count * sizeof(double)));
        }
        const CbetPortSectionArgs ps_args{
            static_cast<int>(port_state->port_weight.size()),
            nullptr,
            chi_view.d_chi,
            chi_view.omega_state,
            port_state->port_weight.data(),
            port_state->pair_p.data(),
            port_state->pair_q.data(),
            port_state->pair_index.data(),
            d_traj_power,
            d_traj_rec_idx,
            d_traj_step_count,
            d_traj_rec_ratio,
            traj_ray_offset,
            n_output_rays_traj,
            traj_output_stride,
            traj_max_steps,
            0};
        ps_chi_post_ms += ms(t_ps_chi_post_start, Clock::now());
        const auto t_ps_solve_start = Clock::now();
        cbet_res = cbet_solve_and_deposit(
            *cbet_ws, laser.cbet, stream, &ps_args, true);
        ps_solve_ms += ms(t_ps_solve_start, Clock::now());
        const auto t_ps_traj_transfer_start =
            verbose ? Clock::now() : Clock::time_point{};
        if (n_output_rays_traj > 0) {
          std::vector<int> h_step_counts(
              static_cast<std::size_t>(n_output_rays_traj));
          check_traj_or_cleanup(
              cudaMemcpy(h_step_counts.data(), d_traj_step_count,
                         static_cast<std::size_t>(n_output_rays_traj) *
                             sizeof(int),
                         cudaMemcpyDeviceToHost),
              "laser_step memcpy traj_step_count D2H failed");

          std::int64_t beam_total = 0;
          for (int i = 0; i < n_output_rays_traj; ++i) {
            beam_total += static_cast<std::int64_t>(
                h_step_counts[static_cast<std::size_t>(i)]);
          }

          if (beam_total > 0) {
            const std::size_t traj_buf_size =
                static_cast<std::size_t>(n_output_rays_traj) *
                static_cast<std::size_t>(traj_max_steps);
            std::vector<double> h_pos1(traj_buf_size);
            std::vector<double> h_pos2(traj_buf_size);
            std::vector<double> h_power(traj_buf_size);
            check_traj_or_cleanup(
                cudaMemcpy(h_pos1.data(), d_traj_pos1,
                           traj_buf_size * sizeof(double),
                           cudaMemcpyDeviceToHost),
                "laser_step memcpy traj_pos1 D2H failed");
            check_traj_or_cleanup(
                cudaMemcpy(h_pos2.data(), d_traj_pos2,
                           traj_buf_size * sizeof(double),
                           cudaMemcpyDeviceToHost),
                "laser_step memcpy traj_pos2 D2H failed");
            check_traj_or_cleanup(
                cudaMemcpy(h_power.data(), d_traj_power,
                           traj_buf_size * sizeof(double),
                           cudaMemcpyDeviceToHost),
                "laser_step memcpy traj_power D2H failed");

            const bool first_beam = state.ray_traj_offsets.empty();
            if (first_beam) {
              state.ray_traj_offsets.push_back(0);
            }
            for (int i = 0; i < n_output_rays_traj; ++i) {
              const int sc =
                  h_step_counts[static_cast<std::size_t>(i)];
              const std::size_t base =
                  static_cast<std::size_t>(i) *
                  static_cast<std::size_t>(traj_max_steps);
              if (collect_ray_output && ray_output != nullptr && sc > 0) {
                ray_output->back().rays_2d[static_cast<std::size_t>(i)].power =
                    h_power[base + static_cast<std::size_t>(sc - 1)];
              }
              for (int s = 0; s < sc; ++s) {
                state.ray_traj_pos1.push_back(
                    h_pos1[base + static_cast<std::size_t>(s)]);
                state.ray_traj_pos2.push_back(
                    h_pos2[base + static_cast<std::size_t>(s)]);
                state.ray_traj_power.push_back(
                    h_power[base + static_cast<std::size_t>(s)]);
              }
              state.ray_traj_step_counts.push_back(
                  static_cast<std::int32_t>(sc));
              state.ray_traj_beam_ids.push_back(
                  static_cast<std::int32_t>(traj_beam_id));
              state.ray_traj_offsets.push_back(
                  static_cast<std::int64_t>(
                      state.ray_traj_pos1.size()));
            }
          }
        }
        release_traj_buffers();
        if (verbose) {
          transfer_ms +=
              ms(t_ps_traj_transfer_start, Clock::now());
        }
        const auto t_ps_capture_start = Clock::now();
        fill_port_section_outgoing_power(state, *cbet_ws);
        state.ps_port_capture_pcross.assign(
            static_cast<std::size_t>(cbet_ws->n_ports) *
                static_cast<std::size_t>(hot_e_n_config_channels),
            0.0);
        if (ps_hot_e_capture_on) {
          const std::size_t capture_doubles =
              static_cast<std::size_t>(cbet_ws->n_ports) *
              static_cast<std::size_t>(cbet_ws->n_rays_total) *
              static_cast<std::size_t>(cbet_ws->ps_n_channels) * 4U;
          std::vector<double> capture_stage(capture_doubles, 0.0);
          cuda_check(cudaMemcpy(
                         capture_stage.data(), cbet_ws->ps_capture_stage,
                         capture_stage.size() * sizeof(double),
                         cudaMemcpyDeviceToHost),
                     "port_section hot-e capture stage D2H failed");
          std::vector<double> sum_Pmu(
              static_cast<std::size_t>(hot_e_n_config_channels), 0.0);
          for (int port = 0; port < cbet_ws->n_ports; ++port) {
            for (int ray = 0; ray < cbet_ws->n_rays_total; ++ray) {
              for (int k = 0; k < cbet_ws->ps_n_channels; ++k) {
                const std::size_t row =
                    ((static_cast<std::size_t>(port) *
                          static_cast<std::size_t>(
                              cbet_ws->n_rays_total) +
                      static_cast<std::size_t>(ray)) *
                         static_cast<std::size_t>(
                             cbet_ws->ps_n_channels) +
                     static_cast<std::size_t>(k)) *
                    4U;
                const double* const capture =
                    capture_stage.data() + row;
                if (!(capture[0] > 0.5) || !(capture[1] > 0.0)) {
                  continue;
                }
                const int config_index =
                    port_state->ps_capture_order[
                        static_cast<std::size_t>(k)];
                const int cell = static_cast<int>(capture[3]);
                TENRYU_ASSERT(
                    config_index >= 0 &&
                        config_index < hot_e_n_config_channels,
                    "port_section hot-e capture config index out of range");
                TENRYU_ASSERT(
                    cell >= 0 &&
                        static_cast<std::size_t>(cell) <
                            hot_e_model_cell_r.size(),
                    "port_section hot-e capture cell hint out of range");
                const std::size_t channel =
                    static_cast<std::size_t>(config_index);
                state.ps_port_capture_pcross[
                    static_cast<std::size_t>(port) *
                            static_cast<std::size_t>(
                                hot_e_n_config_channels) +
                        channel] += capture[1];
                hot_e_model_sum_P[channel] += capture[1];
                hot_e_model_sum_Pr[channel] +=
                    hot_e_model_cell_r[static_cast<std::size_t>(cell)] *
                    capture[1];
                sum_Pmu[channel] += capture[2] * capture[1];
                ps_banked_hot_e_power +=
                    capture[1] *
                    (1.0 -
                     ps_one_minus_eta[static_cast<std::size_t>(k)]);
              }
            }
          }
          for (int config_index = 0;
               config_index < hot_e_n_config_channels;
               ++config_index) {
            const std::size_t channel =
                static_cast<std::size_t>(config_index);
            const double P_cross = hot_e_model_sum_P[channel];
            const double eta = state.hot_e_eta_state_eta[channel];
            if (eta > 0.0 && P_cross > 0.0) {
              hot_e_captures_by_channel[channel].push_back(
                  tenryu::laser::hot_electron::RayCapture{
                      hot_e_model_sum_Pr[channel] / P_cross,
                      sum_Pmu[channel] / P_cross,
                      eta * P_cross});
            }
          }
        }
        ps_capture_ms += ms(t_ps_capture_start, Clock::now());
        g_ps_timing_sum[0] += ps_s1_ms;
        g_ps_timing_sum[1] += ps_chi_d2h_ms;
        g_ps_timing_sum[2] += ps_chi_build_ms;
        g_ps_timing_sum[3] += ps_chi_post_ms;
        g_ps_timing_sum[4] += ps_solve_ms;
        g_ps_timing_sum[5] += ps_capture_ms;
        ++g_ps_timing_calls;
        if (g_ps_timing_calls % 500 == 0) {
          core::log_info(
              "[ps_timing_sum] calls=" +
              std::to_string(g_ps_timing_calls) +
              " s1=" + std::to_string(g_ps_timing_sum[0]) +
              " chi_d2h=" + std::to_string(g_ps_timing_sum[1]) +
              " chi_build=" + std::to_string(g_ps_timing_sum[2]) +
              " chi_post=" + std::to_string(g_ps_timing_sum[3]) +
              " solve=" + std::to_string(g_ps_timing_sum[4]) +
              " capture=" + std::to_string(g_ps_timing_sum[5]) +
              " ms_total");
        }
      } else {
        cbet_res =
            cbet_solve_and_deposit(
                *cbet_ws, laser.cbet, stream, nullptr, true);
      }
      fill_cbet_viz_fields(
          state, cbet_res, *cbet_ws,
          port_section
              ? &static_cast<PortSectionState*>(
                     lmesh.port_section_state.get())
                     ->audit
              : nullptr);
      cbet_iaw_power = cbet_res.E_iaw_rate;
      state.E_cbet_iaw_step += cbet_iaw_power * dt;
      lmesh.last_cbet_exchanged_power = cbet_res.exchanged_power;
      lmesh.last_cbet_ledger_residual = cbet_res.ledger_residual_rel;
      lmesh.last_cbet_conv_final = cbet_res.conv_final;
      lmesh.last_cbet_clamp_count = static_cast<std::int64_t>(cbet_res.clamp_count);
      lmesh.last_cbet_overflow_rays = cbet_res.overflow_rays;
      lmesh.last_cbet_iterations = cbet_res.iterations;
      lmesh.last_cbet_converged = cbet_res.converged;
      // overflow_rays > 0 is a hard error inside cbet_solve_and_deposit
      // (C-02): the diagnostic here can only ever record zero.
      for (std::size_t b = 0; b < beams.items.size(); ++b) {
        const int count = cbet_beam_counts[b];
        if (count <= 0) {
          continue;
        }
        const int offset = cbet_beam_offsets[b];
        check_or_fail(
            cudaMemsetAsync(state.laser_dep.data(), 0,
                            state.laser_dep.size() * sizeof(double), stream),
            "laser_step memset laser_dep failed before cbet beam reduce");
        check_or_fail(
            launch_reduce_per_ray_tallies_1d(
                cbet_ws->dep_rows +
                    static_cast<std::size_t>(offset) * state.laser_dep.size(),
                cbet_ws->unabs_rows + offset, nullptr, state.laser_dep.data(),
                d_unabsorbed, nullptr, count,
                static_cast<int>(state.laser_dep.size()), stream),
            "laser_step cbet per-beam reduce failed");
        const auto dep_power_cell = copy_cell_deposit_to_host(state.laser_dep, stream);
        const double P_beam_b = std::max(0.0, beams.items[b].get_power(t));
        for (std::size_t c = 0; c < dep_power_cell.size(); ++c) {
          total_dep_1d[c] += dep_power_cell[c];
          f_hat_groups[b][c] = dep_power_cell[c] / std::max(P_beam_b, 1.0e-30);
        }
      }
      std::ostringstream cbet_oss;
      cbet_oss.setf(std::ios::scientific);
      cbet_oss << std::setprecision(3) << "[cbet] iters="
               << cbet_res.iterations << " converged=" << cbet_res.converged
               << " exchanged=" << cbet_res.exchanged_power
               << " ledger_rel=" << cbet_res.ledger_residual_rel
               << " clamps=" << cbet_res.clamp_count
               << " capped_pairs=" << cbet_res.capped_pairs;
      core::log_debug(cbet_oss.str());
    }
      if (fold_replays > 0) {
        if (laser_pack_enabled) {
          replay_fold_tallies_kernel<<<1, 1, 0, stream>>>(
              lmesh.scratch_step_tally_slab,
              lmesh.scratch_step_pack_device + fold_tally_pack_offset,
              fold_replays);
          check_or_fail(cudaGetLastError(),
                        "laser_step fold replay tally kernel failed");
        } else {
          double u = 0.0;
          double ra = 0.0;
          double tp = 0.0;
          unsigned long long tc = 0ULL;
          unsigned long long ch = 0ULL;
          check_or_fail(cudaMemcpyAsync(&u, d_unabsorbed, sizeof(double),
                                        cudaMemcpyDeviceToHost, stream),
                        "laser_step fold read d_unabsorbed failed");
          check_or_fail(
              cudaMemcpyAsync(&tc, d_tail_closure_count,
                              sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost, stream),
              "laser_step fold read d_tail_closure_count failed");
          check_or_fail(
              cudaMemcpyAsync(&tp, d_tail_closure_absorbed_power,
                              sizeof(double), cudaMemcpyDeviceToHost, stream),
              "laser_step fold read d_tail_closure_absorbed_power failed");
          check_or_fail(
              cudaMemcpyAsync(&ch, d_critical_surface_hit_count,
                              sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost, stream),
              "laser_step fold read d_critical_surface_hit_count failed");
          check_or_fail(cudaMemcpyAsync(&ra, d_ra_power_total,
                                        sizeof(double),
                                        cudaMemcpyDeviceToHost, stream),
                        "laser_step fold read d_ra_power_total failed");
          check_or_fail(cudaStreamSynchronize(stream),
                        "laser_step fold sync failed");
          for (int k = 0; k < fold_replays; ++k) {
            u += fold_d_unabsorbed;
            ra += fold_d_ra_power;
            tp += fold_d_tail_power;
            tc += fold_d_tail_count;
            ch += fold_d_crit_hits;
          }
          check_or_fail(cudaMemcpyAsync(d_unabsorbed, &u, sizeof(double),
                                        cudaMemcpyHostToDevice, stream),
                        "laser_step fold write d_unabsorbed failed");
          check_or_fail(
              cudaMemcpyAsync(d_tail_closure_count, &tc,
                              sizeof(unsigned long long),
                              cudaMemcpyHostToDevice, stream),
              "laser_step fold write d_tail_closure_count failed");
          check_or_fail(
              cudaMemcpyAsync(d_tail_closure_absorbed_power, &tp,
                              sizeof(double), cudaMemcpyHostToDevice, stream),
              "laser_step fold write d_tail_closure_absorbed_power failed");
          check_or_fail(
              cudaMemcpyAsync(d_critical_surface_hit_count, &ch,
                              sizeof(unsigned long long),
                              cudaMemcpyHostToDevice, stream),
              "laser_step fold write d_critical_surface_hit_count failed");
          check_or_fail(cudaMemcpyAsync(d_ra_power_total, &ra,
                                        sizeof(double),
                                        cudaMemcpyHostToDevice, stream),
                        "laser_step fold write d_ra_power_total failed");
          check_or_fail(cudaStreamSynchronize(stream),
                        "laser_step fold writeback sync failed");
        }
      }
    }

    // OPEN-GXII-LASER (design doc §6o.2): NO deposit sum-assembly here. The
    // 1D trace is REPLICATED — every rank already holds the identical
    // full-line total_dep_1d (inputs are line-gathered at the driver), so
    // the former synchronize_1d_deposit_allgatherv added every rank's full
    // copy and multiplied the deposit by n_ranks (measured exactly 2x at
    // the r_outer absorption cell under P2; the 1D twin of the 2D phantom
    // Allreduce removed in M18d). Ownership masking downstream
    // (transfer_to_1d) keeps the owned windows.

    if (hot_e_capture_on && hot_e_capture_beams > 0) {
      check_or_fail(cudaStreamSynchronize(stream),
                    "laser_step hot_e capture stage sync failed");
      const std::size_t row_doubles =
          static_cast<std::size_t>(laser.rays_per_beam) * 4 *
          static_cast<std::size_t>(hot_e_params.n_channels);
      for (int b = 0; b < static_cast<int>(beams.items.size()); ++b) {
        const double* base = hot_e_capture_stage.data() +
                             static_cast<std::size_t>(b) * row_doubles;
        for (int rr = 0; rr < laser.rays_per_beam; ++rr) {
          for (int k = 0; k < hot_e_params.n_channels; ++k) {
            const double* cap = base +
                (static_cast<std::size_t>(rr) * hot_e_params.n_channels +
                 static_cast<std::size_t>(k)) * 4;
            if (cap[0] > 0.5 && cap[3] > 0.0) {
              const ResolvedHotEChannel& rch = hot_e_channels[static_cast<std::size_t>(k)];
              if (hot_e_model) {
                hot_e_model_sum_P[static_cast<std::size_t>(rch.config_index)] +=
                    cap[3];
                hot_e_model_sum_Pr[static_cast<std::size_t>(rch.config_index)] +=
                    cap[1] * cap[3];
              }
              if (rch.eta_eff > 0.0) {
                hot_e_captures_by_channel[static_cast<std::size_t>(rch.config_index)]
                    .push_back(tenryu::laser::hot_electron::RayCapture{
                        cap[1], cap[2], rch.eta_eff * cap[3]});
              }
            }
          }
        }
      }
    }

    if (hot_e_model) {
      for (int ci = 0; ci < hot_e_n_config_channels; ++ci) {
        const std::size_t s = static_cast<std::size_t>(ci);
        const double sp = hot_e_model_sum_P[s];
        state.hot_e_eta_prev_Pcross[s] = sp;
        state.hot_e_eta_prev_rbar[s] =
            (sp > 0.0) ? (hot_e_model_sum_Pr[s] / sp) : 0.0;
        state.hot_e_eta_prev_valid[s] = (sp > 0.0) ? 1U : 0U;
      }
    }

    t_trace_end = Clock::now();
    std::vector<double> hot_e_power_cell;
    bool hot_e_any_captures = false;
    for (const auto& channel_caps : hot_e_captures_by_channel) {
      if (!channel_caps.empty()) {
        hot_e_any_captures = true;
        break;
      }
    }
    if (hot_e_transport_on && hot_e_any_captures) {
      namespace he = tenryu::laser::hot_electron;
      const std::size_t n_c = hydro_mirror.rho.size();
      hot_e_power_cell.assign(n_c, 0.0);
      state.hot_e_ch_in_step.assign(static_cast<std::size_t>(hot_e_n_config_channels), 0.0);
      state.hot_e_ch_deposited_step.assign(static_cast<std::size_t>(hot_e_n_config_channels), 0.0);
      state.hot_e_ch_escaped_step.assign(static_cast<std::size_t>(hot_e_n_config_channels), 0.0);
      double total_P_hot = 0.0;
      double total_deposited = 0.0;
      double total_residual = 0.0;
      double total_escaped = 0.0;
      double total_Pr = 0.0;
      double max_conservation_resid = 0.0;
      int total_substep_cap_hits = 0;
      bool hot_e_any_active = false;
      std::vector<double> hot_e_Te_host;  // radial mode only, fetched once
      for (int ci = 0; ci < hot_e_n_config_channels; ++ci) {
        const auto& channel_caps = hot_e_captures_by_channel[static_cast<std::size_t>(ci)];
        if (channel_caps.empty()) {
          continue;
        }
        const std::vector<he::HotESource> hot_e_sources =
            he::reduce_captures(channel_caps, hydro_mirror.r_edges);
        if (hot_e_sources.empty()) {
          continue;
        }
        const he::HotEChannelSpec spec =
            laser.hot_electron.sources_specified
                ? he::make_channel_spec(laser.hot_electron,
                                        laser.hot_electron.sources[static_cast<std::size_t>(ci)])
                : he::make_channel_spec_from_shorthand(laser.hot_electron);
        he::DepositResult hot_e_res;
        if (laser.hot_electron.angular_model == "cone") {
          TENRYU_ASSERT(state.mesh.geometry_code != 1,
                        "hot_electron cone with cylindrical geometry (validation gap)");
          const he::Geometry1D geom = (state.mesh.geometry_code == 2)
                                          ? he::Geometry1D::planar
                                          : he::Geometry1D::spherical;
          hot_e_res = he::deposit_hot_electrons_cone_1d_device(
              spec, geom, hot_e_sources, state.rho.data(),
              state.zbar.data(), state.A_eff.data(), state.Te.data(),
              hydro_mirror.cell_is_void, state.x_r.data(),
              static_cast<int>(state.x_r.size()), stream, hot_e_power_cell);
        } else {
          if (hot_e_Te_host.empty()) {
            hot_e_Te_host = copy_cell_deposit_to_host(state.Te, stream);
          }
          hot_e_res = he::deposit_hot_electrons_radial_1d(
              spec, hot_e_sources, hydro_mirror.rho,
              hydro_mirror.zbar, hydro_mirror.A_eff, hot_e_Te_host,
              hydro_mirror.cell_is_void, hydro_mirror.r_edges, hot_e_power_cell);
        }
        if (!hot_e_res.active) {
          continue;
        }
        hot_e_any_active = true;
        state.hot_e_ch_in_step[static_cast<std::size_t>(ci)] = hot_e_res.P_hot * dt;
        state.hot_e_ch_deposited_step[static_cast<std::size_t>(ci)] =
            hot_e_res.P_deposited * dt;
        state.hot_e_ch_escaped_step[static_cast<std::size_t>(ci)] = hot_e_res.P_escaped * dt;
        total_P_hot += hot_e_res.P_hot;
        total_deposited += hot_e_res.P_deposited;
        total_residual += hot_e_res.P_residual_inner;
        total_escaped += hot_e_res.P_escaped;
        total_Pr += hot_e_res.P_hot * hot_e_res.r_source_mean;
        total_substep_cap_hits += hot_e_res.substep_cap_hits;
        if (hot_e_res.conservation_resid > max_conservation_resid) {
          max_conservation_resid = hot_e_res.conservation_resid;
        }
      }
      if (total_substep_cap_hits > 0) {
        // The RK substep cap deposits the chord's remaining energy in the
        // current cell (a thermalization-in-place fallback). Surface it —
        // the per-channel counter was previously computed but dropped
        // (AI review k10-3.4).
        core::log_warning("hot_electron: " +
                          std::to_string(total_substep_cap_hits) +
                          " chord-cell march(es) hit the RK substep cap and "
                          "deposited their remaining energy locally");
      }
      if (hot_e_any_active) {
        state.hot_e_enabled_any = true;
        state.hot_e_in_step = total_P_hot * dt;
        state.hot_e_deposited_step = total_deposited * dt;
        state.hot_e_residual_step = total_residual * dt;
        state.hot_e_escaped_step = total_escaped * dt;
        state.hot_e_source_r = (total_P_hot > 0.0) ? (total_Pr / total_P_hot) : 0.0;
        state.hot_e_conservation_resid = max_conservation_resid;
        // per-cell diagnostics + explicit-source dt limit
        const auto vol_host = copy_cell_deposit_to_host(state.vol, stream);
        const auto ee_host = copy_cell_deposit_to_host(state.ee, stream);
        if (state.hot_e_Q_host.size() != n_c) {
          state.hot_e_Q_host.assign(n_c, 0.0);
        }
        if (state.hot_e_eps_cum_host.size() != n_c) {
          state.hot_e_eps_cum_host.assign(n_c, 0.0);
        }
        double dt_lim = std::numeric_limits<double>::infinity();
        for (std::size_t c = 0; c < n_c; ++c) {
          const double P_cell = hot_e_power_cell[c];
          if (!(P_cell > 0.0)) {
            state.hot_e_Q_host[c] = 0.0;
            continue;
          }
          const double vol_c = vol_host[c];
          const double rho_c = hydro_mirror.rho[c];
          if (vol_c > 0.0) {
            state.hot_e_Q_host[c] = P_cell / vol_c;
            if (rho_c > 0.0) {
              state.hot_e_eps_cum_host[c] += P_cell * dt / (rho_c * vol_c);
              const double e_cell = rho_c * vol_c * std::max(ee_host[c], 0.0);
              if (e_cell > 0.0) {
                const double cand =
                    laser.hot_electron.explicit_source_limit * e_cell / P_cell;
                if (cand < dt_lim) {
                  dt_lim = cand;
                }
              }
            }
          }
        }
        state.hot_e_dt_limit_s = dt_lim;
      }
    }
    apply_deposit_redistribution_1d(state, lmesh, hydro_mirror, total_dep_1d, dt,
                                    laser.deposit.conservation_tol, part,
                                    laser.deposit.deposit_smooth_passes,
                                    laser.deposit.deposit_smooth_alpha,
                                    (hot_e_transport_on &&
                                     !hot_e_power_cell.empty())
                                        ? &hot_e_power_cell
                                        : nullptr);
    t_transfer_end = Clock::now();
  } else {
    std::vector<double> total_dep_lm(static_cast<std::size_t>(lmesh.n_nodes()), 0.0);
    std::vector<double> hot_e_power_cell(static_cast<std::size_t>(state.mesh.topo.n_cells),
                                         0.0);
    std::vector<double> hot_e_ch_in_power(
        static_cast<std::size_t>(hot_e_n_config_channels), 0.0);
    double hot_e_in_power = 0.0;
    double hot_e_source_Pr_power = 0.0;
    copy_lm_nodes_to_host(lmesh, node_R, node_Z);
    CbetWorkspace* cbet_ws = nullptr;
    int cbet_ray_cursor = 0;
    std::vector<int> cbet_group_offsets;
    std::vector<int> cbet_group_counts;
    if (cbet_on_2d) {
      cbet_ws = &global_cbet_workspace();
      cbet_group_offsets.assign(beam_groups.size(), 0);
      cbet_group_counts.assign(beam_groups.size(), 0);
      const int n_cells = (lmesh.n_nodes_r - 1) * (lmesh.n_nodes_z - 1);
      const int nz_cells = lmesh.n_nodes_z - 1;
      const int n_nodes = lmesh.n_nodes();
      // Turning-arc allowance (2026-07-31): the ds_adapt_theta_target
      // limiter resolves turning arcs in ~pi/theta steps; a grazing
      // arc can re-cross one radial face on O(pi/theta) consecutive
      // micro-steps, each producing a (correct) record. The theta
      // floor caps the allowance for tiny user thetas.
      const double theta_t = laser.raytrace.ds_adapt_theta_target;
      const int theta_arc_allowance =
          (theta_t > 0.0)
              ? static_cast<int>(2.0 * M_PI / std::max(theta_t, 1.0e-3))
              : 0;
      const int cap_per_ray = (laser.cbet.max_segments_per_ray > 0)
                                  ? laser.cbet.max_segments_per_ray
                                  : 4 * ((lmesh.n_nodes_r - 1) +
                                         (lmesh.n_nodes_z - 1)) +
                                        64 + theta_arc_allowance;
      const int rays_axis = std::max(2, laser.rays_per_beam);
      const int n_rays_capacity =
          static_cast<int>(beam_groups.size()) * rays_axis * rays_axis;
      cbet_workspace_prepare(*cbet_ws, n_rays_capacity, cap_per_ray, n_cells,
                             static_cast<int>(beam_groups.size()),
                             laser.cbet.n_impact_bins, stream, 4, true,
                             n_nodes, nz_cells, lmesh.n_nodes_z);
    }
    bool debug_one_ray_launched = false;
    for (std::size_t g = 0; g < beam_groups.size(); ++g) {
      const double P_group = group_powers[g];
      if (!(P_group > 0.0)) {
        continue;
      }

      lmesh.clear_deposit(stream);
      const auto& beam_indices = beam_groups[g].beam_indices;
      if (beam_indices.empty()) {
        continue;
      }
      // 2D_RZ shortcut: trace one representative beam per equivalence group and
      // scale by group power. Exactness assumes axisymmetric plasma/optics so
      // azimuthal rotations are physically equivalent.
      const Beam& rep = beams.items[static_cast<std::size_t>(beam_indices.front())];
      const auto t_init_start = verbose ? Clock::now() : Clock::time_point{};
      RayArray2D rays = initialize_rays_2d(rep, lmesh, laser.rays_per_beam, P_group, stream);
      if (verbose) {
        init_ms += ms(t_init_start, Clock::now());
      }
      if (rays.empty()) {
        skipped_unabsorbed_power += P_group;
        continue;
      }
      if (collect_density_diag) {
        const auto t_density_diag_start = verbose ? Clock::now() : Clock::time_point{};
        accumulate_ray_density_2d(rays, hydro_locator_2d, ray_counts, stream);
        if (verbose) {
          density_diag_ms += ms(t_density_diag_start, Clock::now());
        }
      }
      if (collect_ray_output) {
        const auto t_capture_start = verbose ? Clock::now() : Clock::time_point{};
        const int n_copy = std::min(ray_record_cap, rays.n_rays);
        const int output_stride = std::max(1, rays.n_rays / std::max(1, n_copy));
        capture_ray_output_3d(rays, n_copy, output_stride, rep.wave_id, ray_output, stream);
        if (verbose) {
          capture_ms += ms(t_capture_start, Clock::now());
        }
      }
      double* d_traj_pos1 = nullptr;
      double* d_traj_pos2 = nullptr;
      double* d_traj_pos3 = nullptr;
      double* d_traj_power = nullptr;
      int* d_traj_step_count = nullptr;
      const int n_output_rays_traj = collect_trajectory
          ? std::min(ray_record_cap, rays.n_rays) : 0;
      const int traj_output_stride = (n_output_rays_traj > 0)
          ? std::max(1, rays.n_rays / std::max(1, n_output_rays_traj)) : 1;
      const int traj_max_steps = laser.ray_output_max_steps;
      auto release_traj_buffers = [&]() {
        if (d_traj_pos1 != nullptr) {
          static_cast<void>(cudaFree(d_traj_pos1));
          d_traj_pos1 = nullptr;
        }
        if (d_traj_pos2 != nullptr) {
          static_cast<void>(cudaFree(d_traj_pos2));
          d_traj_pos2 = nullptr;
        }
        if (d_traj_pos3 != nullptr) {
          static_cast<void>(cudaFree(d_traj_pos3));
          d_traj_pos3 = nullptr;
        }
        if (d_traj_power != nullptr) {
          static_cast<void>(cudaFree(d_traj_power));
          d_traj_power = nullptr;
        }
        if (d_traj_step_count != nullptr) {
          static_cast<void>(cudaFree(d_traj_step_count));
          d_traj_step_count = nullptr;
        }
      };
      auto check_traj_or_cleanup = [&](const cudaError_t err, const char* message) {
        if (err == cudaSuccess) {
          return;
        }
        release_traj_buffers();
        check_or_fail(err, message);
      };
      if (n_output_rays_traj > 0) {
        const std::size_t traj_buf_size =
            static_cast<std::size_t>(n_output_rays_traj) * static_cast<std::size_t>(traj_max_steps);
        check_traj_or_cleanup(cudaMalloc(reinterpret_cast<void**>(&d_traj_pos1),
                                         traj_buf_size * sizeof(double)),
                              "laser_step cudaMalloc traj_pos1 failed");
        check_traj_or_cleanup(cudaMalloc(reinterpret_cast<void**>(&d_traj_pos2),
                                         traj_buf_size * sizeof(double)),
                              "laser_step cudaMalloc traj_pos2 failed");
        check_traj_or_cleanup(cudaMalloc(reinterpret_cast<void**>(&d_traj_pos3),
                                         traj_buf_size * sizeof(double)),
                              "laser_step cudaMalloc traj_pos3 failed");
        check_traj_or_cleanup(cudaMalloc(reinterpret_cast<void**>(&d_traj_power),
                                         traj_buf_size * sizeof(double)),
                              "laser_step cudaMalloc traj_power failed");
        check_traj_or_cleanup(cudaMalloc(reinterpret_cast<void**>(&d_traj_step_count),
                                         static_cast<std::size_t>(n_output_rays_traj) * sizeof(int)),
                              "laser_step cudaMalloc traj_step_count failed");
      }
      const auto t_memset_start = verbose ? Clock::now() : Clock::time_point{};
      if (d_step_histogram != nullptr) {
        check_traj_or_cleanup(cudaMemsetAsync(
                                  d_step_histogram, 0,
                                  static_cast<std::size_t>(kStepHistSize) * sizeof(int), stream),
                              "laser_step memset d_step_histogram failed before ray_trace_3d");
      }
      if (verbose) {
        memset_ms += ms(t_memset_start, Clock::now());
      }
      if (verbose && laser.mode != "radial_absorption_1d" && rays.n_rays > 0) {
        lmesh.ensure_per_ray_step_scratch(rays.n_rays);
      }
      CbetRecordDeviceArgs cbet_rec_args;
      if (cbet_on_2d) {
        constexpr double kPiCbet = 3.14159265358979323846;
        const double lambda_g_cm =
            (laser.wavelength_nm + rep.delta_lambda_nm) * 1.0e-7;
        const double omega_g =
            2.0 * kPiCbet * core::constants::c_light / lambda_g_cm;
        cbet_stage_ray_meta(*cbet_ws, static_cast<int>(g), cbet_ray_cursor,
                            rays.n_rays, omega_g, rays.power, stream);
        cbet_group_offsets[g] = cbet_ray_cursor;
        cbet_group_counts[g] = rays.n_rays;
        cbet_rec_args.rec_cell = cbet_ws->rec_cell;
        cbet_rec_args.rec_mu = cbet_ws->rec_mu;
        cbet_rec_args.rec_c = cbet_ws->rec_c;
        cbet_rec_args.rec_w00 = cbet_ws->rec_w00;
        cbet_rec_args.rec_w10 = cbet_ws->rec_w10;
        cbet_rec_args.rec_w01 = cbet_ws->rec_w01;
        cbet_rec_args.rec_ds = cbet_ws->rec_ds;
        cbet_rec_args.rec_S = cbet_ws->rec_S;
        cbet_rec_args.rec_w = cbet_ws->rec_w;
        cbet_rec_args.rec_count = cbet_ws->rec_count;
        cbet_rec_args.ray_overflow = cbet_ws->ray_overflow;
        cbet_rec_args.ray_offset = cbet_ray_cursor;
        cbet_rec_args.cap_per_ray = cbet_ws->cap_per_ray;
        cbet_ray_cursor += rays.n_rays;
      }
      const auto t_kernel_start = verbose ? Clock::now() : Clock::time_point{};
      const bool debug_one_ray_this_launch =
          laser.raytrace.debug_one_ray && !debug_one_ray_launched;
      if (hot_e_capture_on) {
        const std::size_t hot_e_capture_bytes =
            static_cast<std::size_t>(rays.n_rays) *
            static_cast<std::size_t>(hot_e_params.n_channels) * 8 * sizeof(double);
        lmesh.ensure_hot_e_capture(rays.n_rays, hot_e_params.n_channels);
        check_traj_or_cleanup(cudaMemsetAsync(lmesh.hot_e_capture, 0, hot_e_capture_bytes,
                                              stream),
                              "laser_step hot_e capture memset failed before ray_trace_3d");
      }
      check_traj_or_cleanup(
          launch_ray_trace_3d(rays, lmesh, laser, lambda_cm,
                              d_traj_pos1, d_traj_pos2, d_traj_pos3, d_traj_power,
                              d_traj_step_count, n_output_rays_traj, traj_output_stride,
                              traj_max_steps,
                              cbet_on_2d ? (cbet_ws->d_scalars + 6) : d_unabsorbed,
                              d_error_flags,
                              hot_e_capture_on ? hot_e_params : HotECaptureParams{},
                              hot_e_capture_on ? lmesh.hot_e_capture : nullptr,
                              d_tail_closure_count,
                              cbet_on_2d ? (cbet_ws->d_scalars + 7)
                                         : d_tail_closure_absorbed_power,
                              d_critical_surface_hit_count,
                              stream, d_step_histogram,
                              lmesh.smooth_kappa_factor,
                              verbose ? lmesh.scratch_per_ray_step_count : nullptr,
                              cbet_on_2d ? false : debug_one_ray_this_launch,
                              cbet_on_2d ? &cbet_rec_args : nullptr),
          "laser_step ray_trace_3d launch failed");
      if (debug_one_ray_this_launch) {
        debug_one_ray_launched = true;
      }
      if (verbose) {
        check_traj_or_cleanup(cudaStreamSynchronize(stream),
                              "laser_step stream synchronize failed after ray_trace_3d timing");
        kernel_ms += ms(t_kernel_start, Clock::now());
      }
      const auto t_trace_transfer_start = verbose ? Clock::now() : Clock::time_point{};
      if (verbose && d_step_histogram != nullptr) {
        if (laser_pack_enabled) {
          PackedStepHistogram packed;
          packed.offset = reserve_step_pack_slot(
              step_pack_cursor,
              static_cast<std::size_t>(kStepHistSize) * sizeof(int),
              alignof(int), lmesh.scratch_step_pack_capacity);
          packed.step = state.step;
          packed.beam_id = rep.wave_id;
          packed.n_rays = rays.n_rays;
          check_traj_or_cleanup(
              cudaMemcpyAsync(
                  lmesh.scratch_step_pack_device + packed.offset,
                  d_step_histogram,
                  static_cast<std::size_t>(kStepHistSize) * sizeof(int),
                  cudaMemcpyDeviceToDevice, stream),
              "laser_step pack step_histogram D2D failed after ray_trace_3d");
          packed_step_histograms.push_back(packed);
        } else {
          std::array<int, kStepHistSize> h_step_hist{};
          check_traj_or_cleanup(
              cudaMemcpyAsync(
                  h_step_hist.data(), d_step_histogram,
                  static_cast<std::size_t>(kStepHistSize) * sizeof(int),
                  cudaMemcpyDeviceToHost, stream),
              "laser_step memcpy step_histogram D2H failed after ray_trace_3d");
          check_traj_or_cleanup(
              cudaStreamSynchronize(stream),
              "laser_step stream synchronize failed after ray_trace_3d stats");
          const double mean_steps =
              (rays.n_rays > 0)
                  ? (static_cast<double>(h_step_hist[0]) /
                     static_cast<double>(rays.n_rays))
                  : 0.0;
          core::log_info(
              "[laser_ray_stats] step=" + std::to_string(state.step) +
              " beam=" + std::to_string(rep.wave_id) +
              " n_rays=" + std::to_string(rays.n_rays) +
              " total_steps=" + std::to_string(h_step_hist[0]) +
              " mean=" + std::to_string(mean_steps) +
              " b100=" + std::to_string(h_step_hist[1]) +
              " b1k=" + std::to_string(h_step_hist[2]) +
              " b10k=" + std::to_string(h_step_hist[3]) +
              " bmax=" + std::to_string(h_step_hist[4]));
        }
      }
      if (verbose) {
        transfer_ms += ms(t_trace_transfer_start, Clock::now());
      }

      if (verbose && laser.mode != "radial_absorption_1d" && rays.n_rays > 0) {
        if (laser_pack_enabled) {
          packed_per_ray_stats.push_back(stage_per_ray_step_stats(
              lmesh, rays.n_rays, state.step, g, stream,
              lmesh.scratch_step_pack_device, step_pack_cursor,
              lmesh.scratch_step_pack_capacity, check_traj_or_cleanup));
        } else {
          emit_per_ray_step_stats(lmesh, rays.n_rays, state.step, g, stream,
                                  check_traj_or_cleanup, ms, transfer_ms);
        }
      }
      const auto t_payload_transfer_start = verbose ? Clock::now() : Clock::time_point{};
      if (n_output_rays_traj > 0) {
        std::vector<int> h_step_counts(static_cast<std::size_t>(n_output_rays_traj));
        check_traj_or_cleanup(cudaMemcpy(h_step_counts.data(), d_traj_step_count,
                                         static_cast<std::size_t>(n_output_rays_traj) * sizeof(int),
                                         cudaMemcpyDeviceToHost),
                              "laser_step memcpy traj_step_count D2H failed");
        std::int64_t beam_total = 0;
        for (int i = 0; i < n_output_rays_traj; ++i) {
          beam_total += static_cast<std::int64_t>(h_step_counts[static_cast<std::size_t>(i)]);
        }

        if (beam_total > 0) {
          const std::size_t traj_buf_size =
              static_cast<std::size_t>(n_output_rays_traj) * static_cast<std::size_t>(traj_max_steps);
          std::vector<double> h_pos1(traj_buf_size);
          std::vector<double> h_pos2(traj_buf_size);
          std::vector<double> h_pos3(traj_buf_size);
          std::vector<double> h_power(traj_buf_size);
          check_traj_or_cleanup(cudaMemcpy(h_pos1.data(), d_traj_pos1,
                                           traj_buf_size * sizeof(double), cudaMemcpyDeviceToHost),
                                "laser_step memcpy traj_pos1 D2H failed");
          check_traj_or_cleanup(cudaMemcpy(h_pos2.data(), d_traj_pos2,
                                           traj_buf_size * sizeof(double), cudaMemcpyDeviceToHost),
                                "laser_step memcpy traj_pos2 D2H failed");
          check_traj_or_cleanup(cudaMemcpy(h_pos3.data(), d_traj_pos3,
                                           traj_buf_size * sizeof(double), cudaMemcpyDeviceToHost),
                                "laser_step memcpy traj_pos3 D2H failed");
          check_traj_or_cleanup(cudaMemcpy(h_power.data(), d_traj_power,
                                           traj_buf_size * sizeof(double), cudaMemcpyDeviceToHost),
                                "laser_step memcpy traj_power D2H failed");

          const bool first_beam = state.ray_traj_offsets.empty();
          if (first_beam) {
            state.ray_traj_offsets.push_back(0);
          }
          for (int i = 0; i < n_output_rays_traj; ++i) {
            const int sc = h_step_counts[static_cast<std::size_t>(i)];
            const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(traj_max_steps);
            for (int s = 0; s < sc; ++s) {
              state.ray_traj_pos1.push_back(h_pos1[base + static_cast<std::size_t>(s)]);
              state.ray_traj_pos2.push_back(h_pos2[base + static_cast<std::size_t>(s)]);
              state.ray_traj_pos3.push_back(h_pos3[base + static_cast<std::size_t>(s)]);
              state.ray_traj_power.push_back(h_power[base + static_cast<std::size_t>(s)]);
            }
            state.ray_traj_step_counts.push_back(static_cast<std::int32_t>(sc));
            state.ray_traj_beam_ids.push_back(static_cast<std::int32_t>(rep.wave_id));
            state.ray_traj_offsets.push_back(
                static_cast<std::int64_t>(state.ray_traj_pos1.size()));
          }
        }
      }
      release_traj_buffers();

      if (!cbet_on_2d) {
        const auto dep_group = copy_lm_deposit_to_host(lmesh, stream);
        if (verbose) {
          transfer_ms += ms(t_payload_transfer_start, Clock::now());
        }
        for (std::size_t n = 0; n < dep_group.size(); ++n) {
          total_dep_lm[n] += dep_group[n];
        }
        const auto dep_power_cell =
            project_lm_power_to_hydro_2d(state, lmesh, dep_group, node_R, node_Z);
        for (std::size_t c = 0; c < dep_power_cell.size(); ++c) {
          f_hat_groups[g][c] = dep_power_cell[c] / std::max(P_group, 1.0e-30);
        }
        if (hot_e_capture_on) {
          const std::size_t capture_doubles =
              static_cast<std::size_t>(rays.n_rays) *
              static_cast<std::size_t>(hot_e_params.n_channels) * 8;
          std::vector<double> hot_e_capture_host(capture_doubles, 0.0);
          check_or_fail(cudaMemcpyAsync(hot_e_capture_host.data(), lmesh.hot_e_capture,
                                        capture_doubles * sizeof(double),
                                        cudaMemcpyDeviceToHost, stream),
                        "laser_step hot_e capture D2H failed");
          check_or_fail(cudaStreamSynchronize(stream),
                        "laser_step hot_e capture sync failed");
          for (int rr = 0; rr < rays.n_rays; ++rr) {
            for (int k = 0; k < hot_e_params.n_channels; ++k) {
              const double* row =
                  hot_e_capture_host.data() +
                  (static_cast<std::size_t>(rr) *
                       static_cast<std::size_t>(hot_e_params.n_channels) +
                   static_cast<std::size_t>(k)) *
                      8;
              const bool valid = row[0] == 1.0;
              if (!valid) {
                continue;
              }
              const ResolvedHotEChannel& rch = hot_e_channels[static_cast<std::size_t>(k)];
              const double P_hot = rch.eta_eff * row[6];
              hot_e_captures_by_channel_2d[static_cast<std::size_t>(rch.config_index)].push_back(
                  tenryu::laser::hot_electron::RayCapture2D{row[1], row[2], row[3],
                                                            row[4], row[5], P_hot});
              hot_e_in_power += P_hot;
              hot_e_source_Pr_power += P_hot * row[1];
              hot_e_ch_in_power[static_cast<std::size_t>(rch.config_index)] += P_hot;
            }
          }
        }
      } else if (verbose) {
        transfer_ms += ms(t_payload_transfer_start, Clock::now());
      }
    }

    if (cbet_on_2d) {
      cbet_stage_cell_fields_2d(*cbet_ws, cbet_lm, laser, lambda_cm, stream);
      const CbetSolveResult cbet_res =
          cbet_solve_and_deposit(*cbet_ws, laser.cbet, stream);
      lmesh.last_cbet_exchanged_power = cbet_res.exchanged_power;
      lmesh.last_cbet_ledger_residual = cbet_res.ledger_residual_rel;
      lmesh.last_cbet_conv_final = cbet_res.conv_final;
      lmesh.last_cbet_clamp_count = static_cast<std::int64_t>(cbet_res.clamp_count);
      lmesh.last_cbet_overflow_rays = cbet_res.overflow_rays;
      lmesh.last_cbet_iterations = cbet_res.iterations;
      lmesh.last_cbet_converged = cbet_res.converged;
      // overflow_rays > 0 is a hard error inside cbet_solve_and_deposit
      // (C-02): the diagnostic here can only ever record zero.
      const std::size_t n_nodes = total_dep_lm.size();
      std::vector<double> row(n_nodes, 0.0);
      for (std::size_t g = 0; g < beam_groups.size(); ++g) {
        const int count = cbet_group_counts[g];
        if (count <= 0) {
          continue;
        }
        check_or_fail(cudaMemcpyAsync(
                          row.data(),
                          cbet_ws->dep_nodes + g * n_nodes,
                          n_nodes * sizeof(double), cudaMemcpyDeviceToHost, stream),
                      "laser_step cbet 2D dep_nodes row D2H failed");
        check_or_fail(cudaStreamSynchronize(stream),
                      "laser_step cbet 2D dep_nodes row sync failed");
        for (std::size_t n = 0; n < n_nodes; ++n) {
          total_dep_lm[n] += row[n];
        }
        const auto dep_power_cell =
            project_lm_power_to_hydro_2d(state, lmesh, row, node_R, node_Z);
        const double P_group = group_powers[g];
        for (std::size_t c = 0; c < dep_power_cell.size(); ++c) {
          f_hat_groups[g][c] = dep_power_cell[c] / std::max(P_group, 1.0e-30);
        }
      }
      for (std::size_t g = 0; g < beam_groups.size(); ++g) {
        const int count = cbet_group_counts[g];
        if (count <= 0) {
          continue;
        }
        cbet_sum_rows_add(cbet_ws->unabs_rows, cbet_group_offsets[g], count,
                          d_unabsorbed, stream);
      }
      std::ostringstream cbet_oss;
      cbet_oss.setf(std::ios::scientific);
      cbet_oss << std::setprecision(3) << "[cbet] iters="
               << cbet_res.iterations << " converged=" << cbet_res.converged
               << " exchanged=" << cbet_res.exchanged_power
               << " ledger_rel=" << cbet_res.ledger_residual_rel
               << " clamps=" << cbet_res.clamp_count
               << " clamped=" << cbet_res.clamped_power
               << " capped_pairs=" << cbet_res.capped_pairs;
      core::log_debug(cbet_oss.str());
    }

    // NO deposit Allreduce here (M18d, Option C): rays are NOT split
    // across ranks — every rank traces the FULL beam set through the
    // identical partition-of-unity LaserMesh, so total_dep_lm is already
    // complete and identical (modulo device-atomic LSBs) on every rank.
    // The historic Allreduce(SUM) assumed a ray split that does not
    // exist and multiplied the deposit by n_ranks (measured x6.5 after
    // the conservation rescale/blocked-filter interplay at the axis
    // critical-surface cell; design doc §6j). Ownership is enforced
    // downstream by mask_non_owned_skip_deposit, and the budget's
    // dep_power Allreduce sums those owned partials correctly. The CBET
    // contribution path is validate-FATAL under MPI.

    if (hot_e_capture_on) {
      state.hot_e_enabled_any = true;
      bool hot_e_any_captures = false;
      for (const auto& channel_caps : hot_e_captures_by_channel_2d) {
        if (!channel_caps.empty()) {
          hot_e_any_captures = true;
          break;
        }
      }
      if (hot_e_any_captures) {
        namespace he = tenryu::laser::hot_electron;
        const std::size_t n_c = hot_e_power_cell.size();
        state.hot_e_ch_in_step.assign(static_cast<std::size_t>(hot_e_n_config_channels), 0.0);
        state.hot_e_ch_deposited_step.assign(static_cast<std::size_t>(hot_e_n_config_channels),
                                             0.0);
        state.hot_e_ch_escaped_step.assign(static_cast<std::size_t>(hot_e_n_config_channels),
                                           0.0);

        std::vector<double> node_r_host(static_cast<std::size_t>(state.mesh.topo.n_nodes), 0.0);
        std::vector<double> node_z_host(static_cast<std::size_t>(state.mesh.topo.n_nodes), 0.0);
        state.x_r.copy_to_host(node_r_host.data());
        state.x_z.copy_to_host(node_z_host.data());
        auto storage =
            state.mesh.topo.multiblock.has_value()
                ? he::build_multiblock_view(state.mesh.topo,
                                            node_r_host.data(),
                                            node_z_host.data(),
                                            state.mesh.cell_nverts.empty()
                                                ? nullptr
                                                : state.mesh.cell_nverts.data())
                : he::build_single_block_view(state.mesh.topo.nr,
                                              state.mesh.topo.nz);
        const auto view =
            storage.view(node_r_host.data(), node_z_host.data(), state.mesh.topo.n_cells);
        he::DeviceMeshView2DScratch dview_scratch;
        const bool hot_e_use_host_pipeline = hot_e_2d_host_pipeline_enabled();
        const he::DeviceMeshView2D dview =
            hot_e_use_host_pipeline
                ? he::DeviceMeshView2D{}
                : he::upload_mesh_view_2d(storage, node_r_host, node_z_host,
                                          state.mesh.topo.n_cells, stream,
                                          dview_scratch);

        double mesh_scale = 0.0;
        for (const double r : node_r_host) {
          mesh_scale = std::max(mesh_scale, std::abs(r));
        }
        for (const double z : node_z_host) {
          mesh_scale = std::max(mesh_scale, std::abs(z));
        }

        std::vector<double> rho_h(n_c, 0.0);
        std::vector<double> zbar_h(n_c, 0.0);
        std::vector<double> A_h(n_c, 0.0);
        std::vector<double> Te_h(n_c, 0.0);
        std::vector<double> ee_h(n_c, 0.0);
        std::vector<double> vol_h(n_c, 0.0);
        state.rho.copy_to_host(rho_h.data());
        state.zbar.copy_to_host(zbar_h.data());
        state.A_eff.copy_to_host(A_h.data());
        state.Te.copy_to_host(Te_h.data());
        state.ee.copy_to_host(ee_h.data());
        state.vol.copy_to_host(vol_h.data());
        const std::vector<std::uint8_t>& void_h = state.cell_is_void;

        double total_deposited = 0.0;
        double total_residual = 0.0;
        double total_escaped = 0.0;
        double max_conservation_resid = 0.0;
        bool hot_e_any_active = false;
        int hot_e_locate_failed = 0;
        static bool dumped = false;
        const std::string& hot_e_dump_path = hot_e_2d_dump_path();
        const bool hot_e_dump_this_step = !dumped && !hot_e_dump_path.empty();
        std::ostringstream hot_e_dump;
        if (hot_e_dump_this_step) {
          hot_e_dump << std::scientific << std::setprecision(17);
          hot_e_dump << "TOPO " << state.mesh.topo.nr << ' ' << state.mesh.topo.nz << ' '
                     << state.mesh.topo.n_cells << ' ' << state.mesh.topo.n_nodes << ' '
                     << mesh_scale << '\n';
          for (std::size_t n = 0; n < node_r_host.size(); ++n) {
            hot_e_dump << "N " << n << ' ' << node_r_host[n] << ' ' << node_z_host[n]
                       << '\n';
          }
          for (std::size_t c = 0; c < n_c; ++c) {
            const int void01 = (c < void_h.size() && void_h[c] != 0U) ? 1 : 0;
            hot_e_dump << "F " << c << ' ' << rho_h[c] << ' ' << zbar_h[c] << ' '
                       << A_h[c] << ' ' << Te_h[c] << ' ' << void01 << '\n';
          }
        }
        for (int ci = 0; ci < hot_e_n_config_channels; ++ci) {
          const auto& channel_caps = hot_e_captures_by_channel_2d[static_cast<std::size_t>(ci)];
          if (channel_caps.empty()) {
            continue;
          }
          const he::HotEChannelSpec spec =
              laser.hot_electron.sources_specified
                  ? he::make_channel_spec(laser.hot_electron,
                                          laser.hot_electron.sources[static_cast<std::size_t>(ci)])
                  : he::make_channel_spec_from_shorthand(laser.hot_electron);
          const auto red = he::reduce_captures_2d(channel_caps, view);
          hot_e_locate_failed += red.n_locate_failed;
          if (hot_e_dump_this_step) {
            hot_e_dump << "CH " << ci << ' ' << spec.T_hot_erg << ' '
                       << spec.n_energy_groups << ' ' << spec.E_min_over_Th << ' '
                       << spec.E_max_over_Th << ' ' << spec.mu_lo << ' ' << spec.mu_hi
                       << ' ' << spec.n_mu << ' ' << spec.n_phi << '\n';
            for (const he::HotESource2D& src : red.sources) {
              hot_e_dump << "S " << ci << ' ' << src.cell << ' ' << src.R_s << ' '
                         << src.Z_s << ' ' << src.k[0] << ' ' << src.k[1] << ' '
                         << src.k[2] << ' ' << src.P_hot << '\n';
            }
          }
          const auto res =
              hot_e_use_host_pipeline
                  ? he::deposit_hot_electrons_cone_2d(
                        spec, red.sources, rho_h, zbar_h, A_h, Te_h, void_h, view,
                        mesh_scale, hot_e_power_cell)
                  : he::deposit_hot_electrons_cone_2d_device(
                        spec, red.sources, state.rho.data(), state.zbar.data(),
                        state.A_eff.data(), state.Te.data(), void_h, dview,
                        mesh_scale, 0, stream, hot_e_power_cell);
          const double P_escaped_ci = red.P_locate_failed + res.P_escaped;
          if (hot_e_ch_in_power[static_cast<std::size_t>(ci)] > 0.0 ||
              res.P_deposited > 0.0 || res.P_residual_inner > 0.0 ||
              P_escaped_ci > 0.0) {
            hot_e_any_active = true;
          }
          state.hot_e_ch_in_step[static_cast<std::size_t>(ci)] =
              hot_e_ch_in_power[static_cast<std::size_t>(ci)] * dt;
          state.hot_e_ch_deposited_step[static_cast<std::size_t>(ci)] =
              res.P_deposited * dt;
          state.hot_e_ch_escaped_step[static_cast<std::size_t>(ci)] = P_escaped_ci * dt;
          total_deposited += res.P_deposited;
          total_residual += res.P_residual_inner;
          total_escaped += P_escaped_ci;
          if (res.conservation_resid > max_conservation_resid) {
            max_conservation_resid = res.conservation_resid;
          }
        }
        if (hot_e_dump_this_step) {
          for (std::size_t c = 0; c < hot_e_power_cell.size(); ++c) {
            const double P_cell = hot_e_power_cell[c];
            if (P_cell != 0.0) {
              hot_e_dump << "D " << c << ' ' << P_cell << '\n';
            }
          }
          hot_e_dump << "SUM " << hot_e_in_power << ' ' << total_deposited << ' '
                     << total_escaped << '\n';
          std::ofstream out(hot_e_dump_path);
          if (out) {
            out << hot_e_dump.str();
          } else {
            core::log_warning("TENRYU_HOTE2D_DUMP open failed: " + hot_e_dump_path);
          }
          dumped = true;
        }
        static_cast<void>(hot_e_locate_failed);
        if (hot_e_any_active) {
          state.hot_e_in_step = hot_e_in_power * dt;
          state.hot_e_deposited_step = total_deposited * dt;
          state.hot_e_residual_step = total_residual * dt;
          state.hot_e_escaped_step = total_escaped * dt;
          state.hot_e_source_r =
              (hot_e_in_power > 0.0) ? (hot_e_source_Pr_power / hot_e_in_power) : 0.0;
          state.hot_e_conservation_resid = max_conservation_resid;
          if (state.hot_e_Q_host.size() != n_c) {
            state.hot_e_Q_host.assign(n_c, 0.0);
          }
          if (state.hot_e_eps_cum_host.size() != n_c) {
            state.hot_e_eps_cum_host.resize(n_c, 0.0);
          }
          double dt_lim = std::numeric_limits<double>::infinity();
          constexpr double tiny = 1.0e-300;
          for (std::size_t c = 0; c < n_c; ++c) {
            const double P_cell = hot_e_power_cell[c];
            if (c < void_h.size() && void_h[c] != 0U) {
              state.hot_e_Q_host[c] = 0.0;
              continue;
            }
            if (!(P_cell > 0.0)) {
              state.hot_e_Q_host[c] = 0.0;
              continue;
            }
            const double vol_c = vol_h[c];
            const double rho_c = rho_h[c];
            if (vol_c > 0.0) {
              state.hot_e_Q_host[c] = P_cell / vol_c;
              if (rho_c > 0.0) {
                state.hot_e_eps_cum_host[c] +=
                    state.hot_e_Q_host[c] / std::max(rho_c, tiny) * dt;
                const double cand =
                    laser.hot_electron.explicit_source_limit *
                    (rho_c * std::max(ee_h[c], 0.0) / state.hot_e_Q_host[c]);
                if (cand < dt_lim) {
                  dt_lim = cand;
                }
              }
            }
          }
          state.hot_e_dt_limit_s = dt_lim;
        }
      }
    }

    const auto t_deposit_transfer_start = verbose ? Clock::now() : Clock::time_point{};
    check_or_fail(cudaMemcpyAsync(lmesh.deposit, total_dep_lm.data(),
                                  total_dep_lm.size() * sizeof(double),
                                  cudaMemcpyHostToDevice, stream),
                  "laser_step memcpyAsync total_dep_lm H2D failed");
    if (verbose) {
      transfer_ms += ms(t_deposit_transfer_start, Clock::now());
    }
    double transfer_scale = 1.0;
    t_trace_end = Clock::now();
    transfer_to_2d(state, lmesh, dt, laser.deposit.conservation_tol, stream, &transfer_scale,
                   part, laser.deposit.deposit_smooth_passes,
                   laser.deposit.deposit_smooth_alpha,
                   hot_e_capture_on ? &hot_e_power_cell : nullptr,
                   reduction);
    t_transfer_end = Clock::now();
    if (std::isfinite(transfer_scale) && transfer_scale > 0.0) {
      for (auto& fg : f_hat_groups) {
        for (double& v : fg) {
          v *= transfer_scale;
        }
      }
    }
  }
  if (collect_density_diag) {
    finalize_ray_density(state, ray_counts);
  }

  double P_unabsorbed_trace = 0.0;
  double ra_power_total = 0.0;
  unsigned long long tail_closure_count = 0;
  double tail_closure_absorbed_power = 0.0;
  unsigned long long critical_surface_hit_count = 0;
  if (laser_pack_enabled) {
    drain_laser_pack();
    const unsigned char* const step_tally_staging =
        lmesh.scratch_step_pack_host + step_tally_pack_offset;
    std::memcpy(&P_unabsorbed_trace, step_tally_staging + 0,
                sizeof(double));
    std::memcpy(&tail_closure_count, step_tally_staging + 8,
                sizeof(unsigned long long));
    std::memcpy(&tail_closure_absorbed_power, step_tally_staging + 16,
                sizeof(double));
    std::memcpy(&critical_surface_hit_count, step_tally_staging + 24,
                sizeof(unsigned long long));
    std::memcpy(&ra_power_total, step_tally_staging + 32,
                sizeof(double));
  } else {
    alignas(8) unsigned char step_tally_staging[40];
    check_or_fail(
        cudaMemcpyAsync(step_tally_staging, lmesh.scratch_step_tally_slab,
                        40, cudaMemcpyDeviceToHost, stream),
        "laser_step memcpy step tally slab failed");
    check_or_fail(cudaStreamSynchronize(stream),
                  "laser_step stream synchronize failed");
    std::memcpy(&P_unabsorbed_trace, step_tally_staging + 0,
                sizeof(double));
    std::memcpy(&tail_closure_count, step_tally_staging + 8,
                sizeof(unsigned long long));
    std::memcpy(&tail_closure_absorbed_power, step_tally_staging + 16,
                sizeof(double));
    std::memcpy(&critical_surface_hit_count, step_tally_staging + 24,
                sizeof(unsigned long long));
    std::memcpy(&ra_power_total, step_tally_staging + 32,
                sizeof(double));
  }
  for (std::size_t b = 0; b < ray_steps_pending.size(); ++b) {
    if (ray_steps_pending[b] != 0U) {
      lmesh.ray_steps_previous[b].swap(lmesh.ray_steps_output[b]);
    }
  }
  P_unabsorbed_trace = std::max(0.0, P_unabsorbed_trace + skipped_unabsorbed_power);
  if (radial_absorption_1d && reduction != nullptr && part.n_ranks > 1) {
    P_unabsorbed_trace = std::max(0.0, reduction->allreduce_sum(P_unabsorbed_trace));
  }
  lmesh.last_trace_unabsorbed_power = P_unabsorbed_trace;
  lmesh.last_ra_power =
      (phys_ext_active && phys_ext_options.ra_enable != 0)
          ? std::max(0.0, ra_power_total)
          : 0.0;
  lmesh.last_tail_closure_count = static_cast<std::int64_t>(std::min<unsigned long long>(
      tail_closure_count,
      static_cast<unsigned long long>(std::numeric_limits<std::int64_t>::max())));
  lmesh.last_tail_closure_absorbed_power = std::max(0.0, tail_closure_absorbed_power);
  lmesh.last_critical_surface_hit_count =
      static_cast<std::int64_t>(std::min<unsigned long long>(
          critical_surface_hit_count,
          static_cast<unsigned long long>(std::numeric_limits<std::int64_t>::max())));
  if (part.n_ranks > 1) {
    mask_non_owned_skip_deposit(state, part);
  }
  const double dep_power_local = sum_field_energy(state.laser_dep) / dt;
  const double dep_power =
      (reduction != nullptr && part.n_ranks > 1)
          ? reduction->allreduce_sum(dep_power_local)
          : dep_power_local;
  if (radial_absorption_1d) {
    lmesh.last_transfer_blocked_power = 0.0;
    lmesh.last_unabsorbed_power = P_unabsorbed_trace;
  } else if (ps_hot_e_capture_on) {
    lmesh.last_unabsorbed_power = P_unabsorbed_trace;
  } else {
    const double cbet_iaw_sink = port_section ? cbet_iaw_power : 0.0;
    const double P_unabsorbed =
        std::max(P_unabsorbed_trace,
                 std::max(0.0, total_power - dep_power - cbet_iaw_sink));
    lmesh.last_unabsorbed_power = P_unabsorbed;
  }
  if (ps_hot_e_capture_on) {
    const double hot_e_deposited_power =
        state.hot_e_deposited_step / dt;
    const double optical_deposited_power =
        dep_power - hot_e_deposited_power;
    const double ledger_residual =
        optical_deposited_power + lmesh.last_unabsorbed_power +
        ps_banked_hot_e_power + cbet_iaw_power - total_power;
    const double ledger_rel =
        std::abs(ledger_residual) / std::max(total_power, 1.0e-300);
    std::ostringstream ps_ledger_oss;
    ps_ledger_oss.setf(std::ios::scientific);
    ps_ledger_oss << std::setprecision(17)
                  << "port_section_s3: optical_dep="
                  << optical_deposited_power
                  << " unabsorbed=" << lmesh.last_unabsorbed_power
                  << " banked_hot_e=" << ps_banked_hot_e_power
                  << " cbet_iaw=" << cbet_iaw_power
                  << " input=" << total_power
                  << " ledger_rel=" << ledger_rel;
    core::log_debug(ps_ledger_oss.str());
  }

  core::DeviceErrorFlags h_flags{};
  if (laser_pack_enabled) {
    std::memcpy(&h_flags,
                lmesh.scratch_step_pack_host + error_flags_pack_offset,
                sizeof(core::DeviceErrorFlags));
  } else {
    check_or_fail(
        cudaMemcpy(&h_flags, d_error_flags, sizeof(core::DeviceErrorFlags),
                   cudaMemcpyDeviceToHost),
        "laser_step memcpy d_error_flags failed");
  }
  log_laser_flags(h_flags);

  if (!radial_absorption_1d && !hot_e_on && skip_cache != nullptr && !cbet_on_2d) {
    if (state.mesh.dim == 2) {
      std::vector<std::vector<double>> f_hat_total(
          1, std::vector<double>(state.laser_dep.size(), 0.0));
      auto& f_hat0 = f_hat_total.front();
      const double total_group_power =
          skip_group_powers.empty() ? 0.0 : skip_group_powers.front();
      if (total_group_power > 0.0) {
        for (std::size_t g = 0; g < f_hat_groups.size(); ++g) {
          const double Pg = (g < group_powers.size()) ? std::max(0.0, group_powers[g]) : 0.0;
          if (!(Pg > 0.0)) {
            continue;
          }
          const double w = Pg / total_group_power;
          for (std::size_t c = 0; c < f_hat0.size(); ++c) {
            f_hat0[c] += f_hat_groups[g][c] * w;
          }
        }
      }
      skip_cache->update_cache(state, f_hat_total, skip_group_powers, beam_dirs, beam_focuses,
                               beam_defocus, stream);
    } else {
      skip_cache->update_cache(state, f_hat_groups, skip_group_powers, beam_dirs, beam_focuses,
                               beam_defocus, stream);
    }
  }

  const double P_absorbed =
      std::max(0.0, total_power - lmesh.last_unabsorbed_power -
                        (port_section ? cbet_iaw_power : 0.0));
  const double absorb_eff = (total_power > 0.0)
                                ? std::clamp(100.0 * P_absorbed / total_power, 0.0, 100.0)
                                : 0.0;
  std::ostringstream oss;
  oss.setf(std::ios::scientific);
  oss << std::setprecision(6) << "[laser] input=" << total_power << " erg/s, absorbed="
      << P_absorbed << " erg/s, unabsorbed=" << lmesh.last_unabsorbed_power
      << " erg/s, efficiency=" << absorb_eff << " %";
  core::log_debug(oss.str());
  t_finalize_end = Clock::now();
  emit_timing();
}

void invalidate_global_skip_cache() {
  global_skip_cache().invalidate();
}

}  // namespace tenryu::laser
