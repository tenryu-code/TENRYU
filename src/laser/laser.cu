#include "laser/laser.cuh"
#include "core/nvtx_range.hpp"

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
#include "core/field.hpp"
#include "laser/beams.cuh"
#include "laser/bilinear_interpolation.cuh"
#include "laser/cbet_lm_fields.cuh"
#include "laser/deposit_transfer.cuh"
#include "laser/cbet.cuh"
#include "laser/cbet_stage_gpu.cuh"
#include "laser/fast_trace_1d.cuh"
#include "laser/hot_e_eta_model.hpp"
#include "laser/deposit_1d_gpu.cuh"
#include "laser/hot_e_inputs_gpu.cuh"
#include "laser/hot_e_transport_1d_gpu.cuh"
#include "laser/hot_electron_1d.cuh"
#include "laser/hot_electron_1d_gpu.cuh"
#include "laser/hot_electron_2d.cuh"
#include "laser/hot_electron_2d_gpu.cuh"
#include "laser/laser_phys_ext.cuh"
#include "laser/port_geometry.hpp"
#include "laser/port_section_chi.hpp"
#include "laser/port_section_overlap.hpp"
#include "laser/port_section_s1_gpu.cuh"
#include "laser/ray_init.cuh"
#include "laser/ray_trace.cuh"
#include "laser/raytrace_skip.cuh"
#include "laser/sector_adapter.hpp"
#include "laser/sector_phase_space.hpp"
#include "mesh/geometry_1d.cuh"

namespace tenryu::laser {
namespace {

struct PortSectionState {
  port_geom::PortTable ports;
  // The phase-space table of the last step, built on the device (port_section_s1_gpu.cuh).
  std::unique_ptr<::tenryu::laser::port_section::S1DeviceWorkspace> s1_ws;
  ::tenryu::laser::port_section::S1DeviceTable device_table{};
  ::tenryu::laser::port_section::S1DeviceInput device_input{};  // the input of the last build
  bool device_table_valid = false;
  core::DeviceArray<double> ray_map_device;  // [n_shells * 64 * 2] snapshot intensity map
  core::DeviceArray<double> shell_r_device;  // [n_shells] the table's shell radii (build time)
  bool ray_map_valid = false;
  // The hot-electron model's sky map of the last step that computed one (hot_e_inputs), kept on
  // the device until a snapshot: I_tot, I_cw [erg/s/cm^2] and n_sigma on the mu-major grid.
  core::DeviceArray<double> sky_I_tot_device;
  core::DeviceArray<double> sky_I_cw_device;
  core::DeviceArray<double> sky_n_sigma_device;
  std::vector<double> sky_mu;
  std::vector<double> sky_phi;
  bool sky_valid = false;
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

static double g_ps_timing_sum[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
static long g_ps_timing_calls = 0;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

// Per-material collision charge of a 1D multi-material deck (Config
// material_zeff, 2026-09-24): device descriptors and tables, uploaded once,
// and the per-node arrays of the laser mesh, sized for its node count.
struct LaserMaterialZeffState {
  std::vector<std::string> signature;
  core::DeviceArray<LaserZeffMaterial> descriptors;
  std::vector<core::DeviceArray<double>> tables;
  LaserNodeMaterial1D nodes;
};

LaserMaterialZeffState& laser_material_zeff_state() {
  static LaserMaterialZeffState state;
  return state;
}

// Uploads the descriptors when the materials' models change; returns false
// when the deck has none (single material or 2D).
bool prepare_material_zeff(const core::Config::LaserConfig& laser) {
  auto& st = laser_material_zeff_state();
  if (laser.ib.material_zeff.empty()) {
    return false;
  }
  std::vector<std::string> signature;
  for (const auto& mz : laser.ib.material_zeff) {
    std::ostringstream key;
    key.precision(17);
    key << mz.model << ";" << mz.langdon_zcoll << ";" << mz.species_z.size() << ";"
        << mz.zeff_table.ndens << "x" << mz.zeff_table.ntemp;
    for (std::size_t e = 0; e < mz.species_z.size(); ++e) {
      key << ";" << mz.species_z[e] << ":" << mz.species_x[e];
    }
    signature.push_back(key.str());
  }
  if (signature == st.signature && st.descriptors.size() == laser.ib.material_zeff.size()) {
    return true;
  }
  const std::size_t n_mat = laser.ib.material_zeff.size();
  std::vector<LaserZeffMaterial> host(n_mat);
  st.tables.clear();
  st.tables.resize(n_mat);
  for (std::size_t m = 0; m < n_mat; ++m) {
    const auto& mz = laser.ib.material_zeff[m];
    LaserZeffMaterial& d = host[m];
    d.langdon_zcoll = mz.langdon_zcoll;
    if (mz.model == "sequential_strip") {
      d.model = 1;
      d.n_species = static_cast<int>(std::min<std::size_t>(
          std::min(mz.species_z.size(), mz.species_x.size()),
          static_cast<std::size_t>(kIBExtMaxSpecies)));
      for (int e = 0; e < d.n_species; ++e) {
        d.z_nuc[e] = mz.species_z[static_cast<std::size_t>(e)];
        d.x_frac[e] = mz.species_x[static_cast<std::size_t>(e)];
      }
    } else if (mz.model == "table" && mz.zeff_table.ndens > 0 && mz.zeff_table.ntemp > 0) {
      const auto& zt = mz.zeff_table;
      d.model = 2;
      st.tables[m].reset(zt.ratio.size());
      st.tables[m].copy_from_host(zt.ratio);
      d.zeff_table = st.tables[m].data();
      d.zt_nd = zt.ndens;
      d.zt_nt = zt.ntemp;
      d.zt_l10d0 = std::log10(zt.ni_grid.front());
      if (zt.ni_grid.size() >= 2U) {
        d.zt_dl10d = std::log10(zt.ni_grid[1]) - d.zt_l10d0;
      }
      d.zt_l10t0 = std::log10(zt.T_grid_eV.front());
      if (zt.T_grid_eV.size() >= 2U) {
        d.zt_dl10t = std::log10(zt.T_grid_eV[1]) - d.zt_l10t0;
      }
    }
  }
  st.descriptors.reset(n_mat);
  st.descriptors.copy_from_host(host);
  st.signature = signature;
  return true;
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

int trace_tau_diag_min_step() {
  static const int min_step = [] {
    const char* v = std::getenv("TENRYU_TRACE_TAU_DIAG_MINSTEP");
    if (v == nullptr || v[0] == '\0') {
      return 0;
    }
    char* end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end == v || *end != '\0' ||
        parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
      return 0;
    }
    return static_cast<int>(parsed);
  }();
  return min_step;
}

bool trace_tau_diag_pabs_only() {
  static const bool enabled = [] {
    const char* v = std::getenv("TENRYU_TRACE_TAU_DIAG_PABS_ONLY");
    return v != nullptr && std::strcmp(v, "1") == 0;
  }();
  return enabled;
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

void write_profile_diag_dump(
    const std::string& output_dir,
    const char* trace_kind,
    const int step,
    const int n_nodes,
    const std::vector<double>& radial_node_r,
    const std::vector<double>& radial_n_hat,
    const std::vector<double>& radial_n_hat_raw,
    const std::vector<double>& radial_smooth_kappa,
    const std::vector<double>& radial_dn_dr) {
  const std::filesystem::path base_dir =
      output_dir.empty() ? std::filesystem::path(".")
                         : std::filesystem::path(output_dir);
  const std::filesystem::path diag_dir = base_dir / "tau_diag";
  std::error_code mkdir_error;
  std::filesystem::create_directories(diag_dir, mkdir_error);
  TENRYU_ASSERT(!mkdir_error,
                "TENRYU_TRACE_TAU_DIAG failed to create output directory");

  const std::string stem = std::string(trace_kind) + "_step" +
                           std::to_string(step) + "_profile";
  const std::filesystem::path binary_path = diag_dir / (stem + ".bin");
  std::ofstream binary(binary_path, std::ios::binary | std::ios::trunc);
  TENRYU_ASSERT(binary,
                "TENRYU_TRACE_TAU_DIAG failed to open profile binary output");
  const auto write_array = [&](const std::vector<double>& values) {
    binary.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(double)));
  };
  write_array(radial_node_r);
  write_array(radial_n_hat);
  write_array(radial_n_hat_raw);
  write_array(radial_smooth_kappa);
  write_array(radial_dn_dr);
  TENRYU_ASSERT(binary,
                "TENRYU_TRACE_TAU_DIAG failed to write profile binary output");

  const std::filesystem::path header_path = diag_dir / (stem + ".txt");
  std::ofstream header(header_path, std::ios::trunc);
  TENRYU_ASSERT(header,
                "TENRYU_TRACE_TAU_DIAG failed to open profile text header");
  header << "n_nodes " << n_nodes << '\n'
         << "radial_node_r " << radial_node_r.size() << '\n'
         << "radial_n_hat " << radial_n_hat.size() << '\n'
         << "radial_n_hat_raw " << radial_n_hat_raw.size() << '\n'
         << "radial_smooth_kappa " << radial_smooth_kappa.size() << '\n'
         << "radial_dn_dr " << radial_dn_dr.size() << '\n';
  TENRYU_ASSERT(header,
                "TENRYU_TRACE_TAU_DIAG failed to write profile text header");
}

// Appends the recorded trajectories of the output rays of one beam (ray i: step_counts[i]
// points at the start of its device row of traj_max_steps) to the state's flattened trajectory
// arrays. The rows are read into compact host rows of the largest point count: the trace writes
// a row only up to the ray's stored point count, and a row's remainder (uninitialised device
// memory; compute-sanitizer initcheck, 2026-09-24) is copied but never read. The last point's
// power of each ray goes to its output record when one is given. Nothing is appended when no
// ray recorded a point.
static cudaError_t append_trajectory_rows_1d(core::State& state,
                                             const std::vector<int>& step_counts,
                                             const int traj_max_steps,
                                             const double* d_pos1,
                                             const double* d_pos2,
                                             const double* d_power,
                                             const int beam_id,
                                             RayOutputData* record) {
  const int n_rays = static_cast<int>(step_counts.size());
  int points = 0;
  std::size_t total = 0;
  for (const int count : step_counts) {
    const int stored = std::clamp(count, 0, traj_max_steps);
    points = std::max(points, stored);
    total += static_cast<std::size_t>(stored);
  }
  if (total == 0) {
    return cudaSuccess;
  }
  const std::size_t row = static_cast<std::size_t>(points);
  std::vector<double> h_pos1(static_cast<std::size_t>(n_rays) * row);
  std::vector<double> h_pos2(h_pos1.size());
  std::vector<double> h_power(h_pos1.size());
  const std::size_t device_pitch = static_cast<std::size_t>(traj_max_steps) * sizeof(double);
  const std::size_t host_pitch = row * sizeof(double);
  const double* sources[3] = {d_pos1, d_pos2, d_power};
  double* targets[3] = {h_pos1.data(), h_pos2.data(), h_power.data()};
  for (int k = 0; k < 3; ++k) {
    const cudaError_t err =
        cudaMemcpy2D(targets[k], host_pitch, sources[k], device_pitch, host_pitch,
                     static_cast<std::size_t>(n_rays), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      return err;
    }
  }
  if (state.ray_traj_offsets.empty()) {
    state.ray_traj_offsets.push_back(0);
  }
  for (int i = 0; i < n_rays; ++i) {
    const int stored = std::clamp(step_counts[static_cast<std::size_t>(i)], 0, traj_max_steps);
    const std::size_t base = static_cast<std::size_t>(i) * row;
    const auto first = static_cast<std::ptrdiff_t>(base);
    const auto last = static_cast<std::ptrdiff_t>(base + static_cast<std::size_t>(stored));
    if (record != nullptr && stored > 0) {
      record->rays_2d[static_cast<std::size_t>(i)].power =
          h_power[base + static_cast<std::size_t>(stored - 1)];
    }
    state.ray_traj_pos1.insert(state.ray_traj_pos1.end(), h_pos1.begin() + first,
                               h_pos1.begin() + last);
    state.ray_traj_pos2.insert(state.ray_traj_pos2.end(), h_pos2.begin() + first,
                               h_pos2.begin() + last);
    state.ray_traj_power.insert(state.ray_traj_power.end(), h_power.begin() + first,
                                h_power.begin() + last);
    state.ray_traj_step_counts.push_back(static_cast<std::int32_t>(stored));
    state.ray_traj_beam_ids.push_back(static_cast<std::int32_t>(beam_id));
    state.ray_traj_offsets.push_back(static_cast<std::int64_t>(state.ray_traj_pos1.size()));
  }
  return cudaSuccess;
}

static void write_ray0_diag_dump(const std::string& output_dir,
                                 const char* trace_kind,
                                 const int step,
                                 const int n_rays,
                                 const std::vector<double>& ray_R0,
                                 const std::vector<double>& ray_Z0,
                                 const std::vector<double>& ray_vR0,
                                 const std::vector<double>& ray_vZ0) {
  const std::filesystem::path base_dir =
      output_dir.empty() ? std::filesystem::path(".")
                         : std::filesystem::path(output_dir);
  const std::filesystem::path diag_dir = base_dir / "tau_diag";
  std::error_code mkdir_error;
  std::filesystem::create_directories(diag_dir, mkdir_error);
  TENRYU_ASSERT(!mkdir_error,
                "TENRYU_TRACE_TAU_DIAG failed to create output directory");

  const std::string stem = std::string(trace_kind) + "_step" +
                           std::to_string(step) + "_ray0";
  const std::filesystem::path binary_path = diag_dir / (stem + ".bin");
  std::ofstream binary(binary_path, std::ios::binary | std::ios::trunc);
  TENRYU_ASSERT(binary,
                "TENRYU_TRACE_TAU_DIAG failed to open ray0 binary output");
  const auto write_array = [&](const std::vector<double>& values) {
    binary.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(double)));
  };
  write_array(ray_R0);
  write_array(ray_Z0);
  write_array(ray_vR0);
  write_array(ray_vZ0);
  TENRYU_ASSERT(binary,
                "TENRYU_TRACE_TAU_DIAG failed to write ray0 binary output");

  const std::filesystem::path header_path = diag_dir / (stem + ".txt");
  std::ofstream header(header_path, std::ios::trunc);
  TENRYU_ASSERT(header,
                "TENRYU_TRACE_TAU_DIAG failed to open ray0 text header");
  header << "n_rays " << n_rays << '\n';
  TENRYU_ASSERT(header,
                "TENRYU_TRACE_TAU_DIAG failed to write ray0 text header");
}

// 1D beams with equal keys trace the same rays (initialize_rays_1d), so one
// trace serves them all. The rays depend on the focus through its position on
// the beam axis (Beam::axial_focus_1d), not the lab z alone.
struct FoldKey {
  double p_beam = 0.0;
  double f_number = 0.0;
  double axial_focus = 0.0;
  double delta_lambda_nm = 0.0;
  int profile_m = 0;
  double profile_w0_cm = 0.0;
  std::string profile_model;
  // "table" profiles: the ring weights come from these samples, so two table
  // beams fold only when their tables are equal.
  std::vector<double> profile_r_cm;
  std::vector<double> profile_I;
};

FoldKey make_fold_key(const Beam& beam, const double p_beam) {
  return FoldKey{p_beam,
                 beam.f_number,
                 beam.axial_focus_1d(),
                 beam.delta_lambda_nm,
                 beam.profile_m,
                 beam.profile_w0_cm,
                 beam.profile_model,
                 beam.profile_r_cm,
                 beam.profile_I};
}

bool fold_keys_equal(const FoldKey& lhs, const FoldKey& rhs) {
  return lhs.p_beam == rhs.p_beam && lhs.f_number == rhs.f_number &&
         lhs.axial_focus == rhs.axial_focus &&
         lhs.delta_lambda_nm == rhs.delta_lambda_nm && lhs.profile_m == rhs.profile_m &&
         lhs.profile_w0_cm == rhs.profile_w0_cm && lhs.profile_model == rhs.profile_model &&
         lhs.profile_r_cm == rhs.profile_r_cm && lhs.profile_I == rhs.profile_I;
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

// The checks of the device CBET output stages (cbet_stage_gpu.cuh), with the messages of the host
// loops they replace.
void check_cbet_step_flags(const int flags) {
  TENRYU_ASSERT((flags & kCbetFlagClosure) == 0,
                "CBET final-iteration dQ+IAW violates per-cell action closure");
  TENRYU_ASSERT((flags & kCbetFlagCellVolume) == 0,
                "CBET visualization requires a positive finite cell volume");
  TENRYU_ASSERT((flags & kCbetFlagRecordOffsets) == 0,
                "port_section outgoing record offset/count mismatch");
  TENRYU_ASSERT((flags & kCbetFlagCaptureChannel) == 0,
                "port_section hot-e capture config index out of range");
  TENRYU_ASSERT((flags & kCbetFlagCaptureCell) == 0,
                "port_section hot-e capture cell hint out of range");
}

// The device workspace of the hot-electron model inputs (one per process, as the CBET workspace;
// deliberately leaked: CUDA teardown order at exit makes destructor-time frees unsafe).
hot_e_inputs::Workspace& hot_e_inputs_workspace() {
  static auto* ws = new hot_e_inputs::Workspace();
  return *ws;
}

// The device workspaces of the 1D laser mesh map (laser_map_1d_gpu.cuh) and of the step's cell
// deposit (deposit_1d_gpu.cuh), one per process and deliberately leaked as the others.
laser_map_1d::Workspace& laser_map_workspace() {
  static auto* ws = new laser_map_1d::Workspace();
  return *ws;
}

deposit_1d::Workspace& deposit_workspace() {
  static auto* ws = new deposit_1d::Workspace();
  return *ws;
}

// The device workspace of the 1D hot-electron transport (hot_e_transport_1d_gpu.cuh), one per
// process and deliberately leaked as the others.
hot_e_transport_1d::Workspace& hot_e_transport_workspace() {
  static auto* ws = new hot_e_transport_1d::Workspace();
  return *ws;
}

// apply_deposit_redistribution_1d on the device for the deposit accumulated in
// deposit_workspace() (overwritten), with the host function's bookkeeping of the laser mesh and
// its warnings; m is this step's map of the same state. Returns the sum of the energies written to
// state.laser_dep (zero when dt is not positive).
double redistribute_deposit_1d(core::State& state,
                               LaserMesh& mesh,
                               const laser_map_1d::MapScalars& m,
                               const double* hot_e_extra_device,
                               const double dt,
                               const double conservation_tol,
                               const parallel::PartitionInfo& part,
                               const int smooth_passes,
                               const double smooth_alpha,
                               cudaStream_t stream) {
  TENRYU_ASSERT(smooth_passes >= 0,
                "apply_deposit_redistribution_1d smooth_passes must be >= 0");
  TENRYU_ASSERT(smooth_alpha >= 0.0 && smooth_alpha <= 0.5,
                "apply_deposit_redistribution_1d smooth_alpha must be in [0, 0.5]");
  mesh.last_ghost_transition_blend = 0.0;
  mesh.last_ghost_transition_resolved_cells = 0;
  mesh.last_transfer_blocked_power = 0.0;
  if (!(dt > 0.0)) {
    state.laser_dep.fill(0.0);
    return 0.0;
  }
  const int n_cells = static_cast<int>(state.laser_dep.size());
  deposit_1d::Inputs in;
  in.dt = dt;
  in.conservation_tol = conservation_tol;
  in.smooth_passes = smooth_passes;
  in.smooth_alpha = smooth_alpha;
  in.ghost_enabled = mesh.ghost_corona_enabled ? 1 : 0;
  in.transition_enabled = mesh.ghost_transition_enabled ? 1 : 0;
  in.handoff_cells = mesh.ghost_handoff_cells;
  in.handoff_decay = mesh.ghost_handoff_decay;
  in.transition_resolved_nhat = mesh.ghost_transition_resolved_nhat;
  in.transition_resolved_cells = mesh.ghost_transition_resolved_cells;
  in.transition_density_exponent = mesh.ghost_transition_density_exponent;
  in.n_crit = mesh.n_crit;
  in.owned_begin = (part.n_ranks > 1) ? part.local_cell_range[0][0] : 0;
  in.owned_end = (part.n_ranks > 1) ? part.local_cell_range[0][1] : n_cells;
  const deposit_1d::Result r = deposit_1d::redistribute(
      deposit_workspace(), state, laser_map_workspace(), m, hot_e_extra_device, in, stream);
  mesh.last_ghost_transition_blend = r.transition_blend;
  mesh.last_ghost_transition_resolved_cells = r.resolved_cells;
  mesh.last_transfer_blocked_power = r.blocked_power;
  if (r.smoothing_ran != 0 && std::abs(r.smoothing_sum_before) > 1.0e-20 &&
      r.smoothing_rel > conservation_tol) {
    core::log_warning("Laser deposit smoothing conservation check failed: rel=" +
                      std::to_string(r.smoothing_rel));
  }
  if (std::abs(r.sum_input) > 1.0e-20 && r.conservation_rel > conservation_tol) {
    core::log_warning("Laser transfer conservation check failed: rel=" +
                      std::to_string(r.conservation_rel));
  }
  return r.energy_sum;
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

// Rays i * stride (i < n_valid) of the five launch arrays, packed as five
// consecutive runs of n_valid values.
__global__ void gather_ray_output_1d_kernel(const double* __restrict__ R0,
                                            const double* __restrict__ Z0,
                                            const double* __restrict__ vR0,
                                            const double* __restrict__ vZ0,
                                            const double* __restrict__ power0,
                                            const int n_valid,
                                            const int stride,
                                            double* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_valid) {
    return;
  }
  const std::size_t src = static_cast<std::size_t>(i) * static_cast<std::size_t>(stride);
  out[0 * n_valid + i] = R0[src];
  out[1 * n_valid + i] = Z0[src];
  out[2 * n_valid + i] = vR0[src];
  out[3 * n_valid + i] = vZ0[src];
  out[4 * n_valid + i] = power0[src];
}

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
  // Only the recorded rays come back (i * stride < n_total), gathered on the
  // device into one buffer and read with one copy.
  const long long n_reachable =
      (static_cast<long long>(n_total) + stride - 1) / static_cast<long long>(stride);
  const int n_valid = static_cast<int>(std::min<long long>(n_copy, n_reachable));
  if (n_valid > 0) {
    auto* d_out = static_cast<double*>(core::device_scratch_acquire(
        "laser:capture_ray_output_1d", 5 * static_cast<std::size_t>(n_valid) * sizeof(double)));
    gather_ray_output_1d_kernel<<<(n_valid + 127) / 128, 128, 0, stream>>>(
        rays.R0, rays.Z0, rays.vR0, rays.vZ0, rays.power0, n_valid, stride, d_out);
    cuda_check(cudaGetLastError(), "laser_step ray output gather launch failed");
    std::vector<double> h_out(5 * static_cast<std::size_t>(n_valid), 0.0);
    cuda_check(cudaMemcpyAsync(h_out.data(), d_out, h_out.size() * sizeof(double),
                               cudaMemcpyDeviceToHost, stream),
               "laser_step memcpyAsync ray output D2H failed");
    cuda_check(cudaStreamSynchronize(stream), "laser_step stream synchronize failed");
    const std::size_t n = static_cast<std::size_t>(n_valid);
    for (std::size_t i = 0; i < n; ++i) {
      auto& rec = data.rays_2d[i];
      rec.R0 = h_out[0 * n + i];
      rec.Z0 = h_out[1 * n + i];
      rec.vR0 = h_out[2 * n + i];
      rec.vZ0 = h_out[3 * n + i];
      rec.power0 = h_out[4 * n + i];
    }
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
  if (flags.unresolved_quadrature != 0) {
    static long long unresolved_warnings = 0;
    if (++unresolved_warnings <= 10 || unresolved_warnings % 1000 == 0) {
      core::log_warning("Laser ray trace (characteristic): " +
                        std::to_string(flags.unresolved_quadrature) +
                        " pieces accepted at the quadrature panel cap (error estimate above the "
                        "tolerance), warning #" + std::to_string(unresolved_warnings));
    }
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

// The phase-space table of the step's reference trace, built on the device from the CBET ray
// records (port_section_s1_gpu.cuh; NUMERICS §5.10.8). Only the S1 audit and ledger scalars come
// back to the host; the snapshot's intensity map stays on the device until a snapshot is written.
void build_port_section_table_device(LaserMesh& lmesh,
                                     const Beam& beam,
                                     const core::State& state,
                                     const CbetWorkspace& cbet_ws,
                                     const int n_rays,
                                     const bool verbose,
                                     cudaStream_t stream) {
  auto* const port_state =
      static_cast<PortSectionState*>(lmesh.port_section_state.get());
  TENRYU_ASSERT(port_state != nullptr,
                "port_section table build requires initialized host state");
  TENRYU_ASSERT(n_rays >= 0, "port_section table build: negative ray count");
  const int n_cells = static_cast<int>(state.rho.size());
  // The effective mass number of the cells, staged on the device for the CBET cell fields of this
  // step (cbet_stage_cell_A_eff_device).
  TENRYU_ASSERT(cbet_ws.cell_A_eff != nullptr &&
                    cbet_ws.cap_cell_outputs >= static_cast<std::size_t>(n_cells),
                "port_section table build: cell A_eff not staged");
  if (!port_state->s1_ws) {
    port_state->s1_ws =
        std::make_unique<::tenryu::laser::port_section::S1DeviceWorkspace>();
  }

  ::tenryu::laser::port_section::S1DeviceInput input;
  input.n_rays = n_rays;
  input.cap_per_ray = cbet_ws.cap_per_ray;
  input.n_cells = n_cells;
  input.n_bins = cbet_ws.n_bins;
  input.rec_count = cbet_ws.rec_count;
  input.rec_cell = cbet_ws.rec_cell;
  input.rec_mu = cbet_ws.rec_mu;
  input.rec_ds = cbet_ws.rec_ds;
  input.rec_S = cbet_ws.rec_S;
  input.ray_P0 = cbet_ws.ray_P0;
  input.ray_group_base = cbet_ws.ray_group_base;
  input.r_edges = state.x_r.data();
  input.rho = state.rho.data();
  input.zbar = state.zbar.data();
  input.A_eff = cbet_ws.cell_A_eff;
  input.n_crit = lmesh.n_crit;
  const double R_beam =
      std::abs(lmesh.Z_max - beam.axial_focus_1d()) /
      (2.0 * std::max(beam.f_number, 1.0e-12));
  input.impact_spacing = n_rays > 0 ? R_beam / static_cast<double>(n_rays) : 0.0;

  const ::tenryu::laser::port_section::S1DeviceSummary summary =
      ::tenryu::laser::port_section::build_s1_table_device(
          input, *port_state->s1_ws, stream, &port_state->device_table);
  TENRYU_ASSERT((summary.error_flags & 1) == 0,
                "port_section ray impact bin out of range");
  // The table keeps the radii it was built on (the hydro nodes move before it is read again).
  if (port_state->shell_r_device.size() != state.x_r.size()) {
    port_state->shell_r_device.reset(state.x_r.size());
  }
  cuda_check(cudaMemcpyAsync(port_state->shell_r_device.data(), state.x_r.data(),
                             state.x_r.size() * sizeof(double),
                             cudaMemcpyDeviceToDevice, stream),
             "port_section shell radii copy failed");
  port_state->device_table.shell_r = port_state->shell_r_device.data();
  port_state->device_input = input;
  port_state->device_table_valid = true;

  const std::size_t map_size =
      static_cast<std::size_t>(port_state->device_table.n_shells) *
      static_cast<std::size_t>(::tenryu::laser::port_section::kRayMapThetaBins) * 2U;
  if (port_state->ray_map_device.size() != map_size) {
    port_state->ray_map_device.reset(map_size);
  }
  ::tenryu::laser::port_section::build_ray_map_device(
      port_state->device_table, port_state->ray_map_device.data(), stream);
  port_state->ray_map_valid = true;

  std::memset(&port_state->audit, 0, sizeof(port_state->audit));
  if (n_rays == 0) {
    // A step without rays leaves an empty table and a zero intensity map; the audit of the
    // former host build stayed zero and was not counted as a build.
    return;
  }
  port_state->audit.n_rays = n_rays;
  port_state->audit.total_nodes = summary.total_nodes;
  port_state->audit.total_crossings = summary.total_crossings;
  port_state->audit.excluded_frac = summary.excluded_power_fraction;
  port_state->audit.bouguer_drift_max = summary.bouguer_drift_max;
  ++port_state->build_count;

  if (verbose) {
    std::ostringstream oss;
    oss.setf(std::ios::scientific);
    oss << std::setprecision(3)
        << "port_section_s1: rays=" << port_state->audit.n_rays
        << " paths=" << summary.n_paths
        << " excluded_frac=" << port_state->audit.excluded_frac
        << " bouguer_drift=" << port_state->audit.bouguer_drift_max
        << " crossings=" << summary.total_crossings
        << " ports=" << port_state->ports.ports.size()
        << " pairs=" << port_state->ports.pairs.size();
    core::log_info(oss.str());
  }
}

int laser_trace_compare_every() {
  static const int every = [] {
    const char* v = std::getenv("TENRYU_LASER_TRACE_COMPARE_EVERY");
    if (v == nullptr || v[0] == '\0') {
      return 0;
    }
    char* end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end == v || *end != '\0' || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max()) {
      return 0;
    }
    return static_cast<int>(parsed);
  }();
  return every;
}

// Measure-only comparison of the 1D spherical integrators on one beam's
// frozen laser state (TENRYU_LASER_TRACE_COMPARE_EVERY=N: every N-th step,
// traces without CBET records or hot-electron capture). The beam's rays are
// traced again into scratch buffers with the configured settings, with the
// characteristic integrator and with the leapfrog march at 4 and 16 times
// finer steps (cfl_ray and the ds_adapt targets divided, ds_adapt_max_factor
// 1; these and characteristic_closure trace with the analytic critical-layer
// closure, the march having no reflection at the critical radius), and with
// the characteristic integrator on two other radial profiles of the same
// state: the hydro-anchored profile with every half cell split into
// TENRYU_LASER_TRACE_COMPARE_PARTS parts (default 8; profile_refined) and the
// Z = 0 column of the 2D laser mesh, the profile before 2026-09-24
// (axis_column); the hydro-anchored profile is rebuilt afterwards. One log line per variant: absorbed and unabsorbed power, the
// absorbed power relative to the characteristic trace, the L1 distance of the
// deposit profile from the characteristic trace relative to its absorbed
// power, the rays stopped by the step cap, the profile's node count and the
// kernel time. The run's deposit, ledgers and error flags are not touched;
// *phys_ext's radial collision-charge pointer is refreshed (the rebuilt arrays
// may move).
void compare_1d_trace_integrators(const RayArray1D& rays,
                                  LaserMesh& lmesh,
                                  const core::State& state,
                                  const core::Config::LaserConfig& laser,
                                  const double lambda_cm,
                                  const double* d_hydro_r_edges,
                                  const int n_cells,
                                  const AllowedSupercriticalCell1D& allowed,
                                  LaserPhysExtOptions* phys_ext,
                                  LaserNodeMaterial1D* node_material,
                                  const int step,
                                  const std::size_t beam,
                                  cudaStream_t stream) {
  if (rays.n_rays <= 0 || n_cells <= 0) {
    return;
  }
  enum ProfileKind : int { kProfileRun = 0, kProfileRefined = 1, kProfileAxisColumn = 2 };
  struct Variant {
    const char* name;
    const char* integrator;
    double refine;
    int profile;
  };
  // The march has no reflection at the critical radius: its variants and
  // characteristic_closure trace with critical_handling.terminate = True
  // (the analytic critical-layer closure).
  const Variant variants[] = {{"configured", nullptr, 1.0, kProfileRun},
                              {"characteristic", "characteristic", 1.0, kProfileRun},
                              {"characteristic_closure", "characteristic", 1.0, kProfileRun},
                              {"leapfrog_x4", "leapfrog", 4.0, kProfileRun},
                              {"leapfrog_x16", "leapfrog", 16.0, kProfileRun},
                              {"profile_refined", "characteristic", 1.0, kProfileRefined},
                              {"axis_column", "characteristic", 1.0, kProfileAxisColumn}};
  constexpr int kVariants = 7;
  constexpr int kReference = 1;  // characteristic
  // Parts per half cell of the refined profile (TENRYU_LASER_TRACE_COMPARE_PARTS,
  // default 8).
  static const int kRefinedParts = [] {
    const char* v = std::getenv("TENRYU_LASER_TRACE_COMPARE_PARTS");
    const long parsed = (v != nullptr) ? std::strtol(v, nullptr, 10) : 0;
    return (parsed >= 2 && parsed <= 256) ? static_cast<int>(parsed) : 8;
  }();
  const bool zcoll_radial_on =
      phys_ext != nullptr && phys_ext->langdon_zcoll_radial != nullptr && node_material != nullptr;
  const auto refresh_zcoll = [&]() {
    if (zcoll_radial_on) {
      phys_ext->langdon_zcoll_radial = node_material->radial_zcoll.data();
    }
  };
  std::array<int, kVariants> profile_nodes{};
  double* d_dep = nullptr;
  double* d_scalars = nullptr;  // unabsorbed, tail power, RA power
  unsigned long long* d_counts = nullptr;  // tail closures, critical hits
  core::DeviceErrorFlags* d_flags = nullptr;
  const auto check = [](const cudaError_t err, const char* what) {
    TENRYU_ASSERT(err == cudaSuccess, what);
  };
  check(cudaMalloc(reinterpret_cast<void**>(&d_dep),
                   static_cast<std::size_t>(n_cells) * sizeof(double)),
        "laser trace compare: deposit allocation failed");
  check(cudaMalloc(reinterpret_cast<void**>(&d_scalars), 3 * sizeof(double)),
        "laser trace compare: scalar allocation failed");
  check(cudaMalloc(reinterpret_cast<void**>(&d_counts), 2 * sizeof(unsigned long long)),
        "laser trace compare: counter allocation failed");
  check(cudaMalloc(reinterpret_cast<void**>(&d_flags), sizeof(core::DeviceErrorFlags)),
        "laser trace compare: flag allocation failed");
  cudaEvent_t ev0 = nullptr;
  cudaEvent_t ev1 = nullptr;
  check(cudaEventCreate(&ev0), "laser trace compare: event creation failed");
  check(cudaEventCreate(&ev1), "laser trace compare: event creation failed");
  std::array<std::vector<double>, kVariants> deposits;
  std::array<std::array<double, 3>, kVariants> scalars{};
  std::array<std::array<unsigned long long, 2>, kVariants> counts{};
  std::array<core::DeviceErrorFlags, kVariants> flags{};
  std::array<float, kVariants> kernel_ms{};
  const bool ra_on = phys_ext != nullptr && phys_ext->ra_enable != 0;
  int built_profile = kProfileRun;
  for (int v = 0; v < kVariants; ++v) {
    if (variants[v].profile != built_profile) {
      if (variants[v].profile == kProfileRefined) {
        map_trace_profile_1d(lmesh, state, kRefinedParts, stream, node_material);
        compute_trace_profile_kappa_1d(lmesh, lambda_cm, laser.absorption.eps_n,
                                       laser.absorption.coulomb_log_floor, stream, phys_ext,
                                       node_material);
      } else if (variants[v].profile == kProfileAxisColumn) {
        extract_axis_column_profile_1d(lmesh, stream, node_material);
      }
      refresh_zcoll();
      built_profile = variants[v].profile;
    }
    profile_nodes[static_cast<std::size_t>(v)] = lmesh.radial_n_nodes;
    const double* d_radial_T_e = (phys_ext != nullptr) ? lmesh.radial_T_e : nullptr;
    core::Config::LaserConfig cfg = laser;
    if (variants[v].integrator != nullptr) {
      cfg.raytrace.integrator = variants[v].integrator;
    }
    if (std::string(variants[v].name) == "characteristic_closure" ||
        std::string(variants[v].integrator != nullptr ? variants[v].integrator : "") ==
            "leapfrog") {
      cfg.absorption.terminate = true;
    }
    const double refine = variants[v].refine;
    if (refine > 1.0) {
      cfg.raytrace.cfl_ray /= refine;
      cfg.raytrace.ds_adapt_g_target /= refine;
      cfg.raytrace.ds_adapt_tau_target /= refine;
      if (cfg.raytrace.ds_adapt_theta_target > 0.0) {
        cfg.raytrace.ds_adapt_theta_target /= refine;
      }
      cfg.raytrace.ds_adapt_max_factor = 1.0;
      cfg.raytrace.max_steps = 100000;
    }
    check(cudaMemsetAsync(d_dep, 0, static_cast<std::size_t>(n_cells) * sizeof(double), stream),
          "laser trace compare: memset failed");
    check(cudaMemsetAsync(d_scalars, 0, 3 * sizeof(double), stream),
          "laser trace compare: memset failed");
    check(cudaMemsetAsync(d_counts, 0, 2 * sizeof(unsigned long long), stream),
          "laser trace compare: memset failed");
    check(cudaMemsetAsync(d_flags, 0, sizeof(core::DeviceErrorFlags), stream),
          "laser trace compare: memset failed");
    check(cudaEventRecord(ev0, stream), "laser trace compare: event record failed");
    check(launch_ray_trace_1d_sph(
              rays, lmesh, cfg, lambda_cm, d_hydro_r_edges, n_cells, allowed.allowed_cell,
              allowed.critical_adjacent_subcritical_cell, allowed.r_crit, d_dep, nullptr,
              nullptr, nullptr, nullptr, nullptr, 0, 0, 0, d_scalars, d_flags, d_counts,
              d_scalars + 1, d_counts + 1, stream, nullptr, nullptr, nullptr,
              HotECaptureParams{}, nullptr, phys_ext, d_radial_T_e,
              ra_on ? d_scalars + 2 : nullptr),
          "laser trace compare: trace launch failed");
    check(cudaEventRecord(ev1, stream), "laser trace compare: event record failed");
    deposits[static_cast<std::size_t>(v)].assign(static_cast<std::size_t>(n_cells), 0.0);
    check(cudaMemcpyAsync(deposits[static_cast<std::size_t>(v)].data(), d_dep,
                          static_cast<std::size_t>(n_cells) * sizeof(double),
                          cudaMemcpyDeviceToHost, stream),
          "laser trace compare: deposit copy failed");
    check(cudaMemcpyAsync(scalars[static_cast<std::size_t>(v)].data(), d_scalars,
                          3 * sizeof(double), cudaMemcpyDeviceToHost, stream),
          "laser trace compare: scalar copy failed");
    check(cudaMemcpyAsync(counts[static_cast<std::size_t>(v)].data(), d_counts,
                          2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream),
          "laser trace compare: counter copy failed");
    check(cudaMemcpyAsync(&flags[static_cast<std::size_t>(v)], d_flags,
                          sizeof(core::DeviceErrorFlags), cudaMemcpyDeviceToHost, stream),
          "laser trace compare: flag copy failed");
    check(cudaStreamSynchronize(stream), "laser trace compare: synchronize failed");
    check(cudaEventElapsedTime(&kernel_ms[static_cast<std::size_t>(v)], ev0, ev1),
          "laser trace compare: event timing failed");
  }
  if (built_profile != kProfileRun) {
    map_trace_profile_1d(lmesh, state, 1, stream, node_material);
    compute_trace_profile_kappa_1d(lmesh, lambda_cm, laser.absorption.eps_n,
                                   laser.absorption.coulomb_log_floor, stream, phys_ext,
                                   node_material);
    refresh_zcoll();
  }
  const auto& ref = deposits[kReference];
  double ref_absorbed = 0.0;
  for (const double x : ref) {
    ref_absorbed += x;
  }
  for (int v = 0; v < kVariants; ++v) {
    const auto& dep = deposits[static_cast<std::size_t>(v)];
    double absorbed = 0.0;
    double l1 = 0.0;
    for (std::size_t c = 0; c < dep.size(); ++c) {
      absorbed += dep[c];
      l1 += std::abs(dep[c] - ref[c]);
    }
    const auto& f = flags[static_cast<std::size_t>(v)];
    std::ostringstream oss;
    oss << std::setprecision(10) << "[laser_trace_compare] step=" << step << " beam=" << beam
        << " variant=" << variants[v].name << " absorbed=" << absorbed
        << " unabsorbed=" << scalars[static_cast<std::size_t>(v)][0]
        << " tail_power=" << scalars[static_cast<std::size_t>(v)][1]
        << " ra_power=" << scalars[static_cast<std::size_t>(v)][2]
        << " tail_closures=" << counts[static_cast<std::size_t>(v)][0]
        << " critical_hits=" << counts[static_cast<std::size_t>(v)][1]
        << " absorbed_rel_characteristic="
        << ((ref_absorbed > 0.0) ? absorbed / ref_absorbed - 1.0 : 0.0)
        << " deposit_l1_rel_characteristic=" << ((ref_absorbed > 0.0) ? l1 / ref_absorbed : 0.0)
        << " step_capped_rays=" << f.infinite_loop
        << " unresolved_pieces=" << f.unresolved_quadrature << " nan=" << f.nan_particle
        << " invalid=" << f.invalid_cell
        << " profile_nodes=" << profile_nodes[static_cast<std::size_t>(v)]
        << " kernel_ms=" << kernel_ms[static_cast<std::size_t>(v)];
    core::log_info(oss.str());
  }
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  cudaFree(d_flags);
  cudaFree(d_counts);
  cudaFree(d_scalars);
  cudaFree(d_dep);
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

bool last_port_section_table_build(const LaserMesh& lmesh,
                                   port_section::S1DeviceInput* input,
                                   port_section::S1DeviceTable* table) {
  const auto* const port_state =
      static_cast<const PortSectionState*>(lmesh.port_section_state.get());
  if (port_state == nullptr || !port_state->device_table_valid) {
    return false;
  }
  *input = port_state->device_input;
  *table = port_state->device_table;
  return true;
}

void sync_laser_snapshot_fields(core::State& state, const LaserMesh& lmesh) {
  // The 1D CBET exchange maps and the port_section outgoing power and capture per port.
  cbet_sync_output_fields(state, global_cbet_workspace());
  const auto* const port_state =
      static_cast<const PortSectionState*>(lmesh.port_section_state.get());
  if (port_state == nullptr) {
    return;
  }
  if (port_state->sky_valid) {
    // The hot-electron model's sky map, in W/cm^2 as the host's fill_port_section_sky_map wrote it.
    constexpr double kErgPerSToW = 1.0e-7;
    state.ps_sky_mu = port_state->sky_mu;
    state.ps_sky_phi = port_state->sky_phi;
    port_state->sky_I_tot_device.copy_to_host(state.ps_sky_I_tot);
    port_state->sky_I_cw_device.copy_to_host(state.ps_sky_I_cw);
    port_state->sky_n_sigma_device.copy_to_host(state.ps_sky_n_sigma);
    for (std::size_t i = 0; i < state.ps_sky_I_tot.size(); ++i) {
      state.ps_sky_I_tot[i] *= kErgPerSToW;
      state.ps_sky_I_cw[i] *= kErgPerSToW;
    }
  }
  if (!port_state->ray_map_valid) {
    return;
  }
  state.ps_ray_map.resize(port_state->ray_map_device.size());
  port_state->ray_map_device.copy_to_host(state.ps_ray_map);
  state.ps_ray_map_shell_r.resize(port_state->shell_r_device.size());
  port_state->shell_r_device.copy_to_host(state.ps_ray_map_shell_r);
}

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
  const core::NvtxRange nvtx_range("laser.step");
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
  // Zeroed on the device in the legacy default stream, ordered like the
  // copy Field1D::fill made (from a host array of zeros, which waited for the
  // device every step) before the readers: finalize_ray_density and the
  // output copies, all on that stream.
  if (!state.ray_density.empty()) {
    cuda_check(cudaMemsetAsync(state.ray_density.data(), 0,
                               state.ray_density.size() * sizeof(double), nullptr),
               "laser_step ray_density zero failed");
  }
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
  if (laser.ib.langdon_model == "legacy_vacuum_map" && !beams.items.empty()) {
    const Beam& common_profile = beams.items.front();
    TENRYU_ASSERT(common_profile.profile_model == "gaussian" ||
                      common_profile.profile_model == "super_gaussian" ||
                      common_profile.profile_model == "flat_top",
                  "langdon legacy_vacuum_map requires gaussian, super_gaussian, "
                  "or flat_top beam profiles");
    for (const Beam& beam : beams.items) {
      TENRYU_ASSERT(beam.profile_model == common_profile.profile_model &&
                        beam.profile_w0_cm == common_profile.profile_w0_cm &&
                        beam.profile_m == common_profile.profile_m,
                    "langdon legacy_vacuum_map requires all beams to share one "
                    "effective (model, w0, m) profile");
    }
  }
  RaytraceSkipCache* skip_cache = nullptr;
  if (!radial_absorption_1d && !hot_e_on) {
    auto& cache = global_skip_cache();
    if (state.mesh.dim != 1 || laser.raytrace_skip_config.enabled) {
      skip_cache = &cache;
    } else if (cache.valid || cache.consecutive_skip_count != 0) {
      cache.invalidate();
    }
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
  std::vector<Vec3> beam_focuses;
  std::vector<double> beam_defocus;
  if (state.mesh.dim != 1 || skip_cache != nullptr) {
    beam_dirs.reserve(beams.items.size());
    beam_focuses.reserve(beams.items.size());
    beam_defocus.reserve(beams.items.size());
    for (const Beam& beam : beams.items) {
      beam_dirs.push_back(Vec3{beam.dir_x, beam.dir_y, beam.dir_z});
      beam_focuses.push_back(Vec3{beam.focus_x, beam.focus_y, beam.focus_lab_z});
      beam_defocus.push_back(beam.defocus_DR);
    }
  }

  // 1D: the laser operator deposits over [t, t + dt], so the commanded power is
  // the waveform's exact average over the step (step energies then sum to the
  // pulse integral; a step straddling a sharp turn-on receives its share).
  // 2D keeps the historic value at the step start.
  const double total_power = (state.mesh.dim == 1)
                                 ? beams.total_average_power(t, t + dt)
                                 : beams.total_power(t);
  if (phys_ext_options.langdon_model != 0) {
    constexpr double kPiLangdon = 3.14159265358979323846;
    const Beam& common_profile = beams.items.front();
    const double P_total_w = total_power * 1.0e-7;
    if (common_profile.profile_model == "flat_top") {
      phys_ext_options.langdon_profile_kind = 1;
      phys_ext_options.langdon_w_cm = common_profile.profile_w0_cm;
      phys_ext_options.langdon_I0_wcm2 =
          P_total_w /
          (kPiLangdon * common_profile.profile_w0_cm *
           common_profile.profile_w0_cm);
    } else if (common_profile.profile_model == "super_gaussian" &&
               common_profile.profile_m >= 2) {
      const double m = static_cast<double>(common_profile.profile_m);
      phys_ext_options.langdon_profile_kind = 2;
      phys_ext_options.langdon_w_cm = common_profile.profile_w0_cm;
      phys_ext_options.langdon_sg_two_m = 2.0 * m;
      phys_ext_options.langdon_I0_wcm2 =
          P_total_w * m * std::pow(2.0, 1.0 / m) /
          (kPiLangdon * common_profile.profile_w0_cm *
           common_profile.profile_w0_cm * std::tgamma(1.0 / m));
    } else {
      phys_ext_options.langdon_profile_kind = 0;
      phys_ext_options.langdon_w_cm =
          common_profile.profile_w0_cm / std::sqrt(2.0);
      phys_ext_options.langdon_I0_wcm2 =
          P_total_w /
          (kPiLangdon * phys_ext_options.langdon_w_cm *
           phys_ext_options.langdon_w_cm);
    }
  }
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
  // n_e / n_c of the cells for the port_section capture thresholds (device, hot_e_inputs).
  const double* hot_e_model_cell_nhat_device = nullptr;
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

      // The model's inputs on the device (hot_e_inputs_gpu.cuh): the cells' n_e / n_c and centres,
      // and per channel the evaluation surface, its temperature, nearest shell and fitted density
      // scale; in port_section, with the previous step's phase-space table, the illumination metric
      // and the common-wave drive with the sky map.
      auto& he_inputs = hot_e_inputs_workspace();
      auto& cbet_cells = global_cbet_workspace();
      cbet_stage_cell_A_eff_device(cbet_cells, state, lmesh.material_A,
                                   lmesh.material_A_list, stream);
      hot_e_inputs::stage_cells(he_inputs, state, cbet_cells.cell_A_eff,
                                lmesh.n_crit, stream);
      hot_e_model_cell_nhat_device = hot_e_inputs::cell_n_hat(he_inputs);

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
      auto* const port_state =
          port_section
              ? static_cast<PortSectionState*>(lmesh.port_section_state.get())
              : nullptr;
      // The previous step's table, as the host consumers read it (a table exists once a step
      // with rays has built one).
      const bool have_table = port_state != nullptr &&
                              port_state->build_count > 0 &&
                              port_state->device_table_valid;
      std::vector<hot_e_inputs::ChannelSpec> specs(n_ch);
      for (int ci = 0; ci < hot_e_n_config_channels; ++ci) {
        const auto& source = he_cfg.sources[static_cast<std::size_t>(ci)];
        auto& spec = specs[static_cast<std::size_t>(ci)];
        spec.eval_nc_fraction = source.eval_nc_fraction;
        spec.illumination =
            port_section && he_cfg.illumination_metric == "equivalent_area";
        spec.common_wave = port_section && source.mechanism == "tpd" &&
                           he_cfg.tpd_overlap_mode == "common_wave_cluster";
        spec.sky_map = port_section && ci == sky_eval_channel;
        spec.delta_theta_deg = he_cfg.common_wave_delta_theta_deg;
      }
      if (have_table) {
        hot_e_inputs::stage_ports(he_inputs, port_state->ports);
      }
      const std::vector<hot_e_inputs::ChannelResult> he_results =
          hot_e_inputs::evaluate_channels(
              he_inputs, state, specs,
              have_table ? &port_state->device_table : nullptr, lmesh.n_crit,
              stream);
      if (port_state != nullptr) {
        const std::size_t n_sky = static_cast<std::size_t>(hot_e_inputs::kCommonWaveMuGrid) *
                                  static_cast<std::size_t>(hot_e_inputs::kCommonWavePhiGrid);
        if (port_state->sky_I_tot_device.size() != n_sky) {
          port_state->sky_I_tot_device.reset(n_sky);
          port_state->sky_I_cw_device.reset(n_sky);
          port_state->sky_n_sigma_device.reset(n_sky);
        }
        if (hot_e_inputs::copy_sky_map(he_inputs, port_state->sky_I_tot_device.data(),
                                       port_state->sky_I_cw_device.data(),
                                       port_state->sky_n_sigma_device.data(), stream)) {
          port_state->sky_mu = hot_e_inputs::sky_mu(he_inputs);
          port_state->sky_phi = hot_e_inputs::sky_phi(he_inputs);
          port_state->sky_valid = true;
        }
      }
      for (int ci = 0; ci < hot_e_n_config_channels; ++ci) {
        const std::size_t s = static_cast<std::size_t>(ci);
        const auto& source = he_cfg.sources[s];
        const hot_e_inputs::ChannelResult& he_result = he_results[s];
        state.hot_e_eta_diag_I14_lower[s] = 0.0;
        state.hot_e_eta_diag_I14_upper[s] = 0.0;
        state.hot_e_eta_diag_n_sigma[s] = 0.0;
        hot_e_eta::ChannelInputs inputs;
        // As the host search left them: set once an evaluation surface exists (zero otherwise),
        // whether or not the fitted scale is positive.
        const double Te_s_eV = he_result.Te_s_eV;
        const double kappa_um = he_result.kappa_um;
        const int eval_shell = he_result.eval_shell;
        bool valid = he_result.valid;

        double f_illum = 1.0;
        if (port_section &&
            he_cfg.illumination_metric == "equivalent_area") {
          if (have_table && eval_shell >= 0 && he_result.illumination_valid) {
            f_illum = he_result.f_illum2;
            if (ci == sky_eval_channel) {
              state.ps_f_illum2 = he_result.f_illum2;
              state.ps_f_union = he_result.f_union;
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
          // Intensity through the capture shell: 4 pi rbar^2 (sphere),
          // 2 pi rbar per unit length (cylinder), 1 per unit area (planar).
          I14 = P /
                (mesh::geometry_1d_face_area(state.mesh.geometry_code, rbar) *
                 f_illum) /
                1.0e7 / 1.0e14;
        } else {
          valid = false;
        }
        if (port_section && source.mechanism == "tpd" &&
            he_cfg.tpd_overlap_mode == "common_wave_cluster") {
          double I_lower = 0.0;
          double I_drive = 0.0;
          double I_upper = 0.0;
          int n_sigma_mode = 0;
          if (have_table && eval_shell >= 0 && he_result.drive_computed) {
            I_drive = he_result.I_drive;
            I_lower = he_result.I_lower;
            I_upper = he_result.I_upper;
            n_sigma_mode = he_result.n_sigma_mode;
            if (std::isfinite(I_drive) && I_drive > 0.0) {
              I14 = I_drive / 1.0e7 / 1.0e14;
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
                    << " lower=" << I_lower
                    << " drive=" << I_drive
                    << " upper=" << I_upper;
          core::log_info(drive_oss.str());
          state.hot_e_eta_diag_I14_lower[s] =
              I_lower / 1.0e7 / 1.0e14;
          state.hot_e_eta_diag_I14_upper[s] =
              I_upper / 1.0e7 / 1.0e14;
          state.hot_e_eta_diag_n_sigma[s] =
              static_cast<double>(n_sigma_mode);
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
  // The few captures made on the host (radial_absorption_1d, port_section). The 1D transport runs
  // on the device (hot_e_transport_1d_gpu.cuh), where the trace's capture rows are staged.
  std::vector<std::vector<tenryu::laser::hot_electron::RayCapture>>
      hot_e_captures_by_channel(static_cast<std::size_t>(hot_e_n_config_channels));
  if (hot_e_transport_on && state.mesh.dim == 1) {
    hot_e_transport_1d::begin_step(hot_e_transport_workspace(), hot_e_n_config_channels, stream);
  }
  int hot_e_capture_beams = 0;
  // 1D beam folding: beams that replay the anchor's trace also replay its
  // hot-electron capture rows (identical keys give identical rays and captures).
  std::size_t fold_anchor_beam = 0;
  std::vector<std::size_t> fold_replayed_beams;
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
      group_powers[b] = std::max(0.0, beams.items[b].get_average_power(t, t + dt));
    }
    if (skip_cache != nullptr) {
      // the normalised deposits are kept on the device (RaytraceSkipCache::f_hat_step)
      skip_group_powers = group_powers;
    }
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
  double skip_energy_sum_1d = 0.0;
  bool skip_energy_sum_valid = false;
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
        &local_skip_eligible, &local_crit_hit, cbet_on, state.mesh.dim == 1);
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
    if (state.mesh.dim == 1) {
      // The 1D cache holds the per-beam ray deposition before the
      // redistribution (blocked receivers, ghost-corona handoff, smoothing),
      // so the scaled deposit goes through the same redistribution with the
      // current hydro state as a traced step (it used to be written to
      // laser_dep as it was, void and supercritical cells included).
      // Map and redistribution on the device (laser_map_1d, deposit_1d).
      auto& skip_cells = global_cbet_workspace();
      cbet_stage_cell_A_eff_device(skip_cells, state, lmesh.material_A,
                                   lmesh.material_A_list, stream);
      const laser_map_1d::MapScalars skip_map = laser_map_1d::map_scalars(
          laser_map_workspace(), state, skip_cells.cell_A_eff,
          map_inputs_1d(lmesh, state, laser), stream);
      auto& skip_deposit = deposit_workspace();
      deposit_1d::begin(skip_deposit, static_cast<int>(state.laser_dep.size()), stream);
      deposit_1d::assign_divided(skip_deposit, state.laser_dep.data(), dt, stream);
      skip_energy_sum_1d = redistribute_deposit_1d(
          state, lmesh, skip_map, nullptr, dt, laser.deposit.conservation_tol, part,
          laser.deposit.deposit_smooth_passes, laser.deposit.deposit_smooth_alpha, stream);
      skip_energy_sum_valid = true;
    }
    if (part.n_ranks > 1) {
      // Keep skip-path deposition ownership consistent with transfer_to_1d/transfer_to_2d.
      mask_non_owned_skip_deposit(state, part);
    }
    // 1D: the redistribution's sum of the written energies (the owned cells, as the mask above).
    const double dep_power_local =
        (skip_energy_sum_valid ? skip_energy_sum_1d : sum_field_energy(state.laser_dep)) / dt;
    const double dep_power =
        (reduction != nullptr && part.n_ranks > 1)
            ? reduction->allreduce_sum(dep_power_local)
            : dep_power_local;
    // The driver books the blocked power (no receiver cell) as a numerical loss, so the unabsorbed power of a
    // skipped step leaves it out (2026-09-29; it was counted in both). 1D: the redistribution above recomputed it
    // for this step. 2D: no transfer runs here and the scaled deposit already lacks it, so the value left from the
    // last traced step is cleared instead of being booked again.
    if (state.mesh.dim != 1) {
      lmesh.last_transfer_blocked_power = 0.0;
    }
    const double skip_blocked_power = std::max(lmesh.last_transfer_blocked_power, 0.0);
    lmesh.last_trace_unabsorbed_power = std::max(0.0, total_power - dep_power - skip_blocked_power);
    lmesh.last_unabsorbed_power = std::max(0.0, total_power - dep_power - skip_blocked_power);
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
  if (state.mesh.dim == 1 && skip_cache != nullptr) {
    skip_cache->begin_fhat_1d(stream);
  }

  const double lambda_cm = laser.wavelength_nm * 1.0e-7;
  CbetLmFields cbet_lm;
  // Multi-material 1D decks: each laser-mesh node's material's collision
  // charge, and the Langdon charge per radial node (2026-09-24).
  LaserNodeMaterial1D* node_material_args = nullptr;
  const bool material_zeff_on =
      state.mesh.dim == 1 && phys_ext_active && prepare_material_zeff(laser) &&
      state.cell_material_index.size() == state.rho.size();
  if (material_zeff_on) {
    auto& st = laser_material_zeff_state();
    st.nodes.cell_material_index = state.cell_material_index.data();
    st.nodes.zeff_materials = st.descriptors.data();
    st.nodes.n_materials = static_cast<int>(st.descriptors.size());
    node_material_args = &st.nodes;
    phys_ext_options.zeff_materials = st.descriptors.data();
    phys_ext_options.n_zeff_materials = static_cast<int>(st.descriptors.size());
  }
  laser_map_1d::MapScalars map_1d;
  if (state.mesh.dim == 1) {
    // The map's scalar part on the device (laser_map_1d_gpu.cuh), from the device A_eff.
    auto& map_cells = global_cbet_workspace();
    cbet_stage_cell_A_eff_device(map_cells, state, lmesh.material_A, lmesh.material_A_list,
                                 stream);
    map_1d = map_from_hydro_1d_device(lmesh, state, laser, map_cells.cell_A_eff,
                                      laser_map_workspace(), stream, node_material_args);
    // The ray-density diagnostic still reads a host mirror.
    if (collect_density_diag) {
      build_hydro_mirror_1d(lmesh, state, hydro_mirror);
    }
    if (phys_ext_options.langdon_model != 0 &&
        phys_ext_options.n_species == 0 && !material_zeff_on) {
      // Zbar of the outermost real cell
      if (map_1d.outer_zbar_cell >= 0) {
        phys_ext_options.langdon_zcoll = map_1d.outer_zbar;
      }
      static bool logged_langdon_zcoll_fallback = false;
      if (!logged_langdon_zcoll_fallback) {
        core::log_info(
            "langdon Z_coll fallback: outermost-cell Zbar = " +
            std::to_string(phys_ext_options.langdon_zcoll) +
            " (no Laser.ib.species)");
        logged_langdon_zcoll_fallback = true;
      }
    }
    if (node_material_args != nullptr && phys_ext_options.langdon_model != 0) {
      phys_ext_options.langdon_zcoll_radial = node_material_args->radial_zcoll.data();
    }
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
                       phys_ext_active ? &phys_ext_options : nullptr, node_material_args);
  // Rays per 1D beam: rings, times azimuths on a cylinder or slab
  // (initialize_rays_1d).
  const int rays_1d_per_beam =
      max_rays_1d_per_beam(lmesh, laser.rays_per_beam, laser.raytrace.azimuthal_rays);
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
    if (err != cudaSuccess) {
      // Name the error: a failed launch or copy looks the same otherwise,
      // whether the device ran out of a resource or a kernel faulted.
      const std::string detail = std::string(message) + ": " + cudaGetErrorName(err) + " (" +
                                 cudaGetErrorString(err) + ")";
      ::tenryu::core::tenryu_abort("err == cudaSuccess", detail, __FILE__, __LINE__);
    }
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
            ? static_cast<std::size_t>(std::max(rays_1d_per_beam, 1))
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
  // 1D: the sum of the energies the device redistribution wrote to laser_dep.
  double step_energy_sum_1d = 0.0;
  bool step_energy_sum_valid = false;
  if (state.mesh.dim == 1) {
    // find_allowed_supercritical_cell_1d, from the device map
    const AllowedSupercriticalCell1D allowed_supercritical{
        map_1d.allowed_cell, map_1d.critical_adjacent_subcritical_cell, map_1d.r_crit_allowed,
        map_1d.fallback_only != 0};
    // The step's cell deposit, accumulated on the device (the beams in order).
    auto& step_deposit = deposit_workspace();
    deposit_1d::begin(step_deposit, static_cast<int>(state.laser_dep.size()), stream);
    if (radial_absorption_1d) {
      check_or_fail(
          cudaMemsetAsync(state.laser_dep.data(), 0,
                          state.laser_dep.size() * sizeof(double), stream),
          "laser_step memset d_deposit_1d failed before radial_absorption_1d");
      double* d_hot_e_capture_radial = nullptr;
      // Every rank evaluates the whole line, as the 1D ray trace does (its inputs are line-gathered at the
      // driver); the ownership masking of the redistribution keeps each rank's cells. Until 2026-09-29 only rank 0
      // launched it, so under MPI the cells of the other ranks received no laser energy.
      check_or_fail(
          launch_radial_absorption_1d(
              total_power, lmesh, laser, state.x_r.data(),
              static_cast<int>(state.laser_dep.size()), state.laser_dep.data(),
              d_unabsorbed, d_error_flags, d_critical_surface_hit_count, stream,
              hot_e_params, &d_hot_e_capture_radial),
          "laser_step radial_absorption_1d launch failed");
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
      deposit_1d::add(step_deposit, state.laser_dep.data(), stream);
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
      double fold_d_unabsorbed = 0.0;
      double fold_d_ra_power = 0.0;
      unsigned long long fold_d_tail_count = 0ULL;
      double fold_d_tail_power = 0.0;
      unsigned long long fold_d_crit_hits = 0ULL;
      int fold_replays = 0;
      lmesh.ray_steps_previous.resize(beams.items.size());
      lmesh.ray_steps_output.resize(beams.items.size());
      lmesh.ray_order.resize(beams.items.size());
      if (hot_e_capture_on) {
        hot_e_transport_1d::begin_trace(hot_e_transport_workspace(), hot_e_params.n_channels,
                                        static_cast<int>(beams.items.size()), rays_1d_per_beam,
                                        stream);
      }
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
          state.step >= trace_tau_diag_min_step() &&
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
        deposit_1d::add_fold(step_deposit, stream);
        if (skip_cache != nullptr) {
          skip_cache->set_fhat_1d(static_cast<int>(b), deposit_1d::fold(step_deposit), P_beam,
                                  stream);
        }
        ++fold_replays;
        fold_replayed_beams.push_back(b);
        continue;
      }
      lmesh.clear_deposit(stream);
      const auto t_init_start = verbose ? Clock::now() : Clock::time_point{};
      RayArray1D rays = initialize_rays_1d(beam, lmesh, laser.rays_per_beam, P_beam, stream,
                                           laser.raytrace.azimuthal_rays);
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
        if (!trace_tau_diag_pabs_only()) {
          tau_diag_doubles = static_cast<std::size_t>(rays.n_rays) *
                             static_cast<std::size_t>(n_radial_intervals);
          d_tau_shell_out = static_cast<double*>(core::device_scratch_acquire(
              "laser:tau_shell_diag", tau_diag_doubles * sizeof(double)));
          check_traj_or_cleanup(
              cudaMemsetAsync(d_tau_shell_out, 0,
                              tau_diag_doubles * sizeof(double), stream),
              "laser_step memset tau-shell diagnostic failed");
        }
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
        // The outermost downward crossing of n_c, from the device map (-1 when none).
        phys_ext_options.ra_r_crit_cm = map_1d.ra_r_crit_cm;
        phys_ext_options.ra_ln_cm = map_1d.ra_ln_cm;
      }
      // Opt-in until the S1.1 physics-accuracy pass closes the march parity gap;
      // see NUMERICS §5.3.5 (2026-08-04 Bouguer fast path).
      static const bool fast_trace_enabled = [] {
        const char* s = std::getenv("TENRYU_FAST_TRACE");
        return s != nullptr && std::strcmp(s, "1") == 0;
      }();
      const bool use_fast_trace_1d =
          fast_trace_enabled && state.mesh.dim == 1 && laser.mode == "raytrace_2d" &&
          state.mesh.geometry_code == 0 && !laser.cbet.enable &&
          beam.profile_model == "flat_top" && !hot_e_capture_on && !collect_trajectory &&
          !collect_ray_output && laser.absorption.terminate;
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
      // The characteristic integrator's cost per ray is bounded by its pieces
      // (radial nodes, faces, events): no previous-step ordering or step cap.
      const bool characteristic_trace_1d =
          !use_fast_trace_1d && (laser.raytrace.integrator == "characteristic" ||
                                 laser.raytrace.integrator == "auto");
      const int* d_ray_order = nullptr;
      int* d_ray_steps_out = nullptr;
      int max_ray_steps_override = 0;
      if (!use_fast_trace_1d && !characteristic_trace_1d) {
        const core::NvtxRange nvtx_order_range("laser.ray_ordering");
        auto& previous_steps = lmesh.ray_steps_previous[b];
        auto& ray_order = lmesh.ray_order[b];
        auto& steps_output = lmesh.ray_steps_output[b];
        static const bool spike_cap_disabled = [] {
          const char* s = std::getenv("TENRYU_LASER_NO_SPIKE_CAP");
          return s != nullptr && std::strcmp(s, "1") == 0;
        }();
        const std::size_t n_rays_sz = static_cast<std::size_t>(rays.n_rays);
        const bool have_previous = previous_steps.size() == n_rays_sz && n_rays_sz > 0U;
        const bool cap_steps = !spike_cap_disabled && have_previous;
        // The longest-first order and the 90th percentile of the previous step counts on the
        // device (the host stable sort and nth_element's results; one integer comes back).
        ray_order.reset(n_rays_sz);
        int p90 = 0;
        check_traj_or_cleanup(
            order_rays_by_previous_steps(previous_steps.data(), rays.n_rays,
                                         !ray_sort_disabled() && have_previous,
                                         ray_order.data(), cap_steps ? &p90 : nullptr, stream),
            "laser_step ray ordering failed");
        if (cap_steps) {
          const int dynamic_cap = std::max(20000, 10 * p90);
          max_ray_steps_override = std::min(laser.raytrace.max_steps, dynamic_cap);
        }
        steps_output.reset(n_rays_sz);
        d_ray_order = ray_order.data();
        d_ray_steps_out = steps_output.data();
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
              phys_ext_active && phys_ext_options.ra_enable != 0
                  ? d_ra_power_total
                  : nullptr,
              d_ray_order, d_ray_steps_out, max_ray_steps_override,
              d_tau_shell_out, d_pabs_per_ray_out);
      check_traj_or_cleanup(trace_status,
                            "laser_step 1D spherical trace launch failed");
      if (laser_trace_compare_every() > 0 && !use_fast_trace_1d && !cbet_on &&
          !hot_e_capture_on && state.step % laser_trace_compare_every() == 0) {
        compare_1d_trace_integrators(
            rays, lmesh, state, laser, lambda_cm, state.x_r.data(),
            static_cast<int>(state.laser_dep.size()), allowed_supercritical,
            phys_ext_active ? &phys_ext_options : nullptr, node_material_args, state.step, b,
            stream);
      }
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
      if (tau_diag_this_step && b == 0) {
        const std::size_t n_profile_nodes =
            static_cast<std::size_t>(lmesh.radial_n_nodes);
        std::vector<double> radial_node_r(n_profile_nodes, 0.0);
        std::vector<double> radial_n_hat(n_profile_nodes, 0.0);
        std::vector<double> radial_n_hat_raw(n_profile_nodes, 0.0);
        std::vector<double> radial_smooth_kappa(n_profile_nodes, 0.0);
        std::vector<double> radial_dn_dr(n_profile_nodes, 0.0);
        check_traj_or_cleanup(
            cudaMemcpyAsync(radial_node_r.data(), lmesh.radial_node_r,
                            n_profile_nodes * sizeof(double),
                            cudaMemcpyDeviceToHost, stream),
            "laser_step radial_node_r diagnostic D2H failed");
        check_traj_or_cleanup(
            cudaMemcpyAsync(radial_n_hat.data(), lmesh.radial_n_hat,
                            n_profile_nodes * sizeof(double),
                            cudaMemcpyDeviceToHost, stream),
            "laser_step radial_n_hat diagnostic D2H failed");
        check_traj_or_cleanup(
            cudaMemcpyAsync(radial_n_hat_raw.data(), lmesh.radial_n_hat_raw,
                            n_profile_nodes * sizeof(double),
                            cudaMemcpyDeviceToHost, stream),
            "laser_step radial_n_hat_raw diagnostic D2H failed");
        check_traj_or_cleanup(
            cudaMemcpyAsync(radial_smooth_kappa.data(),
                            lmesh.radial_smooth_kappa,
                            n_profile_nodes * sizeof(double),
                            cudaMemcpyDeviceToHost, stream),
            "laser_step radial_smooth_kappa diagnostic D2H failed");
        check_traj_or_cleanup(
            cudaMemcpyAsync(radial_dn_dr.data(), lmesh.radial_dn_dr,
                            n_profile_nodes * sizeof(double),
                            cudaMemcpyDeviceToHost, stream),
            "laser_step radial_dn_dr diagnostic D2H failed");
        check_traj_or_cleanup(
            cudaStreamSynchronize(stream),
            "laser_step radial profile diagnostic synchronize failed");
        write_profile_diag_dump(output_dir,
                                use_fast_trace_1d ? "fast" : "march",
                                state.step, lmesh.radial_n_nodes,
                                radial_node_r, radial_n_hat, radial_n_hat_raw,
                                radial_smooth_kappa, radial_dn_dr);
        const std::size_t n_rays = static_cast<std::size_t>(rays.n_rays);
        std::vector<double> ray_R0(n_rays, 0.0);
        std::vector<double> ray_Z0(n_rays, 0.0);
        std::vector<double> ray_vR0(n_rays, 0.0);
        std::vector<double> ray_vZ0(n_rays, 0.0);
        check_traj_or_cleanup(
            cudaMemcpyAsync(ray_R0.data(), rays.R0,
                            n_rays * sizeof(double),
                            cudaMemcpyDeviceToHost, stream),
            "laser_step ray_R0 diagnostic D2H failed");
        check_traj_or_cleanup(
            cudaMemcpyAsync(ray_Z0.data(), rays.Z0,
                            n_rays * sizeof(double),
                            cudaMemcpyDeviceToHost, stream),
            "laser_step ray_Z0 diagnostic D2H failed");
        check_traj_or_cleanup(
            cudaMemcpyAsync(ray_vR0.data(), rays.vR0,
                            n_rays * sizeof(double),
                            cudaMemcpyDeviceToHost, stream),
            "laser_step ray_vR0 diagnostic D2H failed");
        check_traj_or_cleanup(
            cudaMemcpyAsync(ray_vZ0.data(), rays.vZ0,
                            n_rays * sizeof(double),
                            cudaMemcpyDeviceToHost, stream),
            "laser_step ray_vZ0 diagnostic D2H failed");
        check_traj_or_cleanup(
            cudaStreamSynchronize(stream),
            "laser_step ray0 diagnostic synchronize failed");
        write_ray0_diag_dump(output_dir,
                             use_fast_trace_1d
                                 ? "fast"
                                 : (characteristic_trace_1d ? "characteristic" : "march"),
                             state.step, rays.n_rays,
                             ray_R0, ray_Z0, ray_vR0, ray_vZ0);
      }
      if (!use_fast_trace_1d && !characteristic_trace_1d) {
        ray_steps_pending[b] = 1U;
      }
      if (hot_e_capture_on && d_hot_e_capture_rays != nullptr) {
        hot_e_transport_1d::stage_beam(hot_e_transport_workspace(), static_cast<int>(b),
                                       d_hot_e_capture_rays, rays.n_rays, stream);
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
        check_traj_or_cleanup(
            append_trajectory_rows_1d(
                state, h_step_counts, traj_max_steps, d_traj_pos1, d_traj_pos2, d_traj_power,
                beam.wave_id,
                (collect_ray_output && ray_output != nullptr) ? &ray_output->back() : nullptr),
            "laser_step trajectory rows D2H failed");
      }
      if (!port_section) {
        release_traj_buffers();
      }
      if (!cbet_on) {
        deposit_1d::add(step_deposit, state.laser_dep.data(), stream);
        if (skip_cache != nullptr) {
          skip_cache->set_fhat_1d(static_cast<int>(b), state.laser_dep.data(), P_beam, stream);
        }
        if (verbose) {
          transfer_ms += ms(t_payload_transfer_start, Clock::now());
        }
        if (fold_enabled && !fold_armed) {
          deposit_1d::keep_fold(step_deposit, state.laser_dep.data(), stream);
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
          fold_anchor_beam = b;
          fold_armed = true;
        }
      } else if (verbose) {
        transfer_ms += ms(t_payload_transfer_start, Clock::now());
      }
    }
    if (cbet_on) {
      cbet_stage_cell_A_eff_device(*cbet_ws, state, lmesh.material_A,
                                   lmesh.material_A_list, stream);
      cbet_stage_cell_fields_device(*cbet_ws, state, laser, lambda_cm, stream);
      if (port_section) {
        const auto t_ps_s1_start = Clock::now();
        build_port_section_table_device(
            lmesh, beams.items.front(), state, *cbet_ws, cbet_ray_cursor,
            verbose, stream);
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

        TENRYU_ASSERT(port_state->device_table_valid,
                      "port_section chi build requires the step's phase-space table");
        const ::tenryu::laser::port_section::ChiBuildInput chi_input{
            &port_state->ports,
            nullptr,
            laser.cbet.n_impact_bins,
            nullptr,
            port_state->device_table.n_paths,
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
            laser.wavelength_nm,
            &port_state->device_table};
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
          TENRYU_ASSERT(hot_e_model_cell_nhat_device != nullptr,
                        "port_section hot-e cell n_hat not staged");
          ps_one_minus_eta.resize(hot_e_channels.size(), 1.0);
          for (std::size_t k = 0; k < hot_e_channels.size(); ++k) {
            const ResolvedHotEChannel& channel = hot_e_channels[k];
            ps_one_minus_eta[k] =
                laser.hot_electron.subtract_from_laser
                    ? (1.0 - channel.eta_eff)
                    : 1.0;
          }
          cuda_check(cudaMemcpyAsync(
                         cbet_ws->ps_cell_nhat, hot_e_model_cell_nhat_device,
                         static_cast<std::size_t>(n_cells) * sizeof(double),
                         cudaMemcpyDeviceToDevice, stream),
                     "port_section hot-e cell n_hat D2D failed");
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
            *cbet_ws, laser.cbet, stream, &ps_args, false);
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
          check_traj_or_cleanup(
              append_trajectory_rows_1d(
                  state, h_step_counts, traj_max_steps, d_traj_pos1, d_traj_pos2,
                  d_traj_power, traj_beam_id,
                  (collect_ray_output && ray_output != nullptr) ? &ray_output->back()
                                                                : nullptr),
              "laser_step trajectory rows D2H failed");
        }
        release_traj_buffers();
        if (verbose) {
          transfer_ms +=
              ms(t_ps_traj_transfer_start, Clock::now());
        }
        const auto t_ps_capture_start = Clock::now();
        // The exchange maps, the outgoing power of each port and the hot-electron capture sums,
        // on the device (cbet_stage_gpu.cuh); the capture sums come back for the hot-electron
        // source below, with the stages' checks.
        cbet_begin_step_outputs(*cbet_ws, hot_e_n_config_channels, stream);
        cbet_viz_fields_device(*cbet_ws, cbet_res.dq_G, cbet_res.dq_n_branches,
                               cbet_res.dq_n_bins, true, stream);
        ps_outputs_device(*cbet_ws, state.x_r.data(), n_cells, hot_e_n_config_channels,
                          ps_hot_e_capture_on, stream);
        const CbetStepReadback step_out =
            cbet_read_step_outputs(*cbet_ws, hot_e_n_config_channels, stream);
        check_cbet_step_flags(step_out.flags);
        port_state->audit.dq_abs_max = step_out.dq_abs_max;
        port_state->audit.dq_abs_sum = step_out.dq_abs_sum;
        // No L copy is read back for the audit.
        port_state->audit.L_ps_abs_max = -1.0;
        if (ps_hot_e_capture_on) {
          for (int config_index = 0;
               config_index < hot_e_n_config_channels;
               ++config_index) {
            const std::size_t channel =
                static_cast<std::size_t>(config_index);
            // port_section has no trace-side capture: these sums start from zero here.
            TENRYU_ASSERT(hot_e_model_sum_P[channel] == 0.0 &&
                              hot_e_model_sum_Pr[channel] == 0.0,
                          "port_section hot-e capture sums not empty");
            hot_e_model_sum_P[channel] = step_out.capture_sum_P[channel];
            hot_e_model_sum_Pr[channel] = step_out.capture_sum_Pr[channel];
            const double P_cross = hot_e_model_sum_P[channel];
            const double eta = state.hot_e_eta_state_eta[channel];
            if (eta > 0.0 && P_cross > 0.0) {
              hot_e_captures_by_channel[channel].push_back(
                  tenryu::laser::hot_electron::RayCapture{
                      hot_e_model_sum_Pr[channel] / P_cross,
                      step_out.capture_sum_Pmu[channel] / P_cross,
                      eta * P_cross});
            }
          }
          TENRYU_ASSERT(ps_banked_hot_e_power == 0.0,
                        "port_section banked hot-e power not empty");
          ps_banked_hot_e_power = step_out.capture_banked_power;
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
                *cbet_ws, laser.cbet, stream, nullptr, false);
        // The exchange maps on the device (cbet_stage_gpu.cuh), with their checks.
        cbet_begin_step_outputs(*cbet_ws, 0, stream);
        cbet_viz_fields_device(*cbet_ws, cbet_res.dq_G, cbet_res.dq_n_branches,
                               cbet_res.dq_n_bins, false, stream);
        check_cbet_step_flags(cbet_read_step_outputs(*cbet_ws, 0, stream).flags);
      }
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
        deposit_1d::add(step_deposit, state.laser_dep.data(), stream);
        if (skip_cache != nullptr) {
          // Same step-averaged power the 1D rays were launched with.
          const double P_beam_b = std::max(0.0, beams.items[b].get_average_power(t, t + dt));
          skip_cache->set_fhat_1d(static_cast<int>(b), state.laser_dep.data(), P_beam_b, stream);
        }
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
    // the former deposit Allgatherv (removed with its helper 2026-09-29) added every rank's full
    // copy and multiplied the deposit by n_ranks (measured exactly 2x at
    // the r_outer absorption cell under P2; the 1D twin of the 2D phantom
    // Allreduce removed in M18d). Ownership masking downstream
    // (transfer_to_1d) keeps the owned windows.

    bool hot_e_any_captures = false;
    if (hot_e_transport_on) {
      auto& he_ws = hot_e_transport_workspace();
      const bool trace_captures = hot_e_capture_on && hot_e_capture_beams > 0;
      if (trace_captures) {
        // A folded beam replays the anchor's deposit without tracing; its
        // captures are the anchor's (they used to be left empty, losing those
        // beams' hot-electron source).
        for (const std::size_t replayed : fold_replayed_beams) {
          hot_e_transport_1d::replay_beam(he_ws, static_cast<int>(fold_anchor_beam),
                                          static_cast<int>(replayed), stream);
        }
        std::vector<int> capture_config_index(hot_e_channels.size());
        std::vector<double> capture_eta_eff(hot_e_channels.size());
        for (std::size_t k = 0; k < hot_e_channels.size(); ++k) {
          capture_config_index[k] = hot_e_channels[k].config_index;
          capture_eta_eff[k] = hot_e_channels[k].eta_eff;
        }
        hot_e_transport_1d::collect_trace_captures(he_ws, capture_config_index, capture_eta_eff,
                                                   hot_e_model, stream);
      }
      for (int ci = 0; ci < hot_e_n_config_channels; ++ci) {
        const auto& host_caps = hot_e_captures_by_channel[static_cast<std::size_t>(ci)];
        if (!host_caps.empty()) {
          hot_e_transport_1d::add_host_captures(he_ws, ci, host_caps);
        }
      }
      std::vector<double> trace_sum_P;
      std::vector<double> trace_sum_Pr;
      hot_e_transport_1d::read_capture_state(he_ws, &trace_sum_P, &trace_sum_Pr,
                                             &hot_e_any_captures, stream);
      if (trace_captures && hot_e_model) {
        for (int ci = 0; ci < hot_e_n_config_channels; ++ci) {
          const std::size_t s = static_cast<std::size_t>(ci);
          hot_e_model_sum_P[s] += trace_sum_P[s];
          hot_e_model_sum_Pr[s] += trace_sum_Pr[s];
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
    const double* hot_e_power_device = nullptr;
    if (hot_e_transport_on && hot_e_any_captures) {
      namespace he = tenryu::laser::hot_electron;
      auto& he_ws = hot_e_transport_workspace();
      state.hot_e_ch_in_step.assign(static_cast<std::size_t>(hot_e_n_config_channels), 0.0);
      state.hot_e_ch_deposited_step.assign(static_cast<std::size_t>(hot_e_n_config_channels), 0.0);
      state.hot_e_ch_escaped_step.assign(static_cast<std::size_t>(hot_e_n_config_channels), 0.0);
      std::vector<he::HotEChannelSpec> hot_e_specs(
          static_cast<std::size_t>(hot_e_n_config_channels));
      for (int ci = 0; ci < hot_e_n_config_channels; ++ci) {
        hot_e_specs[static_cast<std::size_t>(ci)] =
            laser.hot_electron.sources_specified
                ? he::make_channel_spec(laser.hot_electron,
                                        laser.hot_electron.sources[static_cast<std::size_t>(ci)])
                : he::make_channel_spec_from_shorthand(laser.hot_electron);
      }
      const bool hot_e_cone = laser.hot_electron.angular_model == "cone";
      // The radial march reads the laser's effective mass number of the cells; the cone chords
      // read the State's.
      const double* hot_e_A_eff_radial = nullptr;
      if (!hot_e_cone) {
        auto& radial_cells = global_cbet_workspace();
        cbet_stage_cell_A_eff_device(radial_cells, state, lmesh.material_A,
                                     lmesh.material_A_list, stream);
        hot_e_A_eff_radial = radial_cells.cell_A_eff;
      }
      const std::vector<hot_e_transport_1d::ChannelResult> hot_e_results =
          hot_e_transport_1d::transport(he_ws, state, hot_e_specs, state.mesh.geometry_code,
                                        hot_e_cone, state.A_eff.data(), hot_e_A_eff_radial,
                                        stream);
      hot_e_power_device = hot_e_transport_1d::power_cell(he_ws);
      double total_P_hot = 0.0;
      double total_deposited = 0.0;
      double total_residual = 0.0;
      double total_escaped = 0.0;
      double total_Pr = 0.0;
      double max_conservation_resid = 0.0;
      int total_substep_cap_hits = 0;
      bool hot_e_any_active = false;
      for (int ci = 0; ci < hot_e_n_config_channels; ++ci) {
        const hot_e_transport_1d::ChannelResult& hot_e_res =
            hot_e_results[static_cast<std::size_t>(ci)];
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
        // (2026-07-26 review).
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
        // per-cell diagnostics + explicit-source dt limit, on the device
        state.hot_e_dt_limit_s = hot_e_transport_1d::diagnostics(
            he_ws, state, dt, laser.hot_electron.explicit_source_limit, stream);
      }
    }
    static const bool transfer_audit_enabled = [] {
      const char* value = std::getenv("TENRYU_LASER_TRANSFER_AUDIT");
      return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    long double audit_in = 0.0L;
    if (transfer_audit_enabled) {
      std::vector<double> total_dep_host(state.laser_dep.size(), 0.0);
      check_or_fail(cudaMemcpyAsync(total_dep_host.data(), deposit_1d::total(step_deposit),
                                    total_dep_host.size() * sizeof(double),
                                    cudaMemcpyDeviceToHost, stream),
                    "laser_step transfer audit D2H failed");
      check_or_fail(cudaStreamSynchronize(stream), "laser_step transfer audit sync failed");
      for (const double deposit : total_dep_host) {
        audit_in += static_cast<long double>(deposit);
      }
    }
    // The hot-electron power per cell of the transport above (device).
    TENRYU_ASSERT(hot_e_power_device == nullptr || state.rho.size() == state.laser_dep.size(),
                  "apply_deposit_redistribution_1d hot_e_extra_power size mismatch");
    step_energy_sum_1d = redistribute_deposit_1d(
        state, lmesh, map_1d, hot_e_power_device, dt, laser.deposit.conservation_tol, part,
        laser.deposit.deposit_smooth_passes, laser.deposit.deposit_smooth_alpha, stream);
    step_energy_sum_valid = true;
    if (transfer_audit_enabled) {
      const auto audit_dep_host =
          copy_cell_deposit_to_host(state.laser_dep, stream);
      long double audit_out = 0.0L;
      for (const double deposit : audit_dep_host) {
        audit_out += static_cast<long double>(deposit);
      }
      std::ostringstream audit;
      audit << std::setprecision(17) << "[laser:transfer-audit] in="
            << static_cast<double>(audit_in) << " out="
            << static_cast<double>(audit_out);
      core::log_warning(audit.str());
    }
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
      std::swap(lmesh.ray_steps_previous[b], lmesh.ray_steps_output[b]);
    }
  }
  P_unabsorbed_trace = std::max(0.0, P_unabsorbed_trace + skipped_unabsorbed_power);
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
  // 1D: the redistribution's sum of the written energies (the owned cells, as the mask above).
  const double dep_power_local =
      (step_energy_sum_valid ? step_energy_sum_1d : sum_field_energy(state.laser_dep)) / dt;
  const double dep_power =
      (reduction != nullptr && part.n_ranks > 1)
          ? reduction->allreduce_sum(dep_power_local)
          : dep_power_local;
  // Laser power ledger: input = deposited (incl. the hot-electron deposit) +
  // unabsorbed (rays leaving the profile, or stopped by the intensity cutoff,
  // the step guard or an invalid state; skipped and folded beams) +
  // hot-electron escape + power the transfer could not place (no receiver
  // cell) + CBET ion-acoustic sink. In 1D every path books its unabsorbed
  // power explicitly, and the remainder of the ledger is checked, not
  // classified as unabsorbed (that hid leaks, and counted the hot-electron
  // escape twice in the driver's escaped energy). In 2D the difference still
  // bounds the trace tally from below.
  const double hot_e_escaped_power =
      (dt > 0.0) ? std::max(state.hot_e_escaped_step, 0.0) / dt : 0.0;
  const double cbet_iaw_sink = port_section ? cbet_iaw_power : 0.0;
  // radial_absorption_1d books like the 1D trace (2026-09-29): its deposit goes through the same redistribution,
  // whose blocked power (no receiver cell) is a numerical loss, not unabsorbed power; it used to be reset to 0
  // here, which dropped it from the ledger, and the ledger check skipped the radial mode.
  if (ps_hot_e_capture_on || state.mesh.dim == 1) {
    lmesh.last_unabsorbed_power = P_unabsorbed_trace;
  } else {
    const double P_unabsorbed = std::max(
        P_unabsorbed_trace,
        std::max(0.0, total_power - dep_power - cbet_iaw_sink - hot_e_escaped_power -
                          std::max(lmesh.last_transfer_blocked_power, 0.0)));
    lmesh.last_unabsorbed_power = P_unabsorbed;
  }
  if (state.mesh.dim == 1 && !ps_hot_e_capture_on) {
    const double residual = total_power - dep_power - P_unabsorbed_trace -
                            hot_e_escaped_power -
                            std::max(lmesh.last_transfer_blocked_power, 0.0) - cbet_iaw_sink;
    const double tol = std::max(laser.deposit.conservation_tol, 1.0e-10);
    static long long ledger_warnings = 0;
    if (total_power > 0.0 && std::abs(residual) > tol * total_power &&
        (++ledger_warnings <= 10 || ledger_warnings % 1000 == 0)) {
      std::ostringstream ledger_oss;
      ledger_oss.setf(std::ios::scientific);
      ledger_oss << std::setprecision(6)
                 << "Laser power ledger does not close: input=" << total_power
                 << " deposited=" << dep_power << " unabsorbed=" << P_unabsorbed_trace
                 << " hot_e_escaped=" << hot_e_escaped_power
                 << " transfer_blocked=" << lmesh.last_transfer_blocked_power
                 << " cbet_iaw=" << cbet_iaw_sink << " residual=" << residual << " ("
                 << residual / total_power << " of the input, warning #" << ledger_warnings
                 << ")";
      core::log_warning(ledger_oss.str());
    }
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
      skip_cache->update_cache_1d(state, skip_group_powers, beam_dirs, beam_focuses, beam_defocus,
                                  stream);
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
