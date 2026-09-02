#include "drivers/cmd_checkpoint_swap_center.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "burn/burn_constants.hpp"
#include "core/config.hpp"
#include "core/constants.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "core/state_per_material_init.hpp"
#include "core/version.hpp"
#include "coupling/driver.hpp"
#include "hydro/eos_context.hpp"
#include "hydro/euler_window_blend.hpp"
#include "hydro/hydro_2d.hpp"
#include "io/hdf5_reader.hpp"
#include "io/hdf5_writer.hpp"
#include "mesh/mesh.hpp"
#include "radiation/particle_pool.cuh"

#if TENRYU_ENABLE_HDF5
#include <hdf5.h>

#include "io/hdf5_utils.hpp"
#endif

#if TENRYU_ENABLE_PYTHON
#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include "core/namelist/freeze.hpp"
#include "core/namelist/geometry_eval.hpp"
#include "core/namelist/runtime.hpp"
#endif

namespace tenryu::drivers {
namespace {

constexpr std::string_view kAuditGroup = "metadata/epoch_swap/v1";
constexpr std::string_view kToolVersion =
    "tenryu-checkpoint-swap-center/v1";
constexpr double kFreshVelocityNormalizationLimitCmS = 1.0e-3;

double relative_difference(const double a, const double b) {
  if (!std::isfinite(a) || !std::isfinite(b)) {
    return std::numeric_limits<double>::infinity();
  }
  const double scale = std::max({std::abs(a), std::abs(b),
                                 std::numeric_limits<double>::min()});
  return std::abs(a - b) / scale;
}

double signed_relative_difference_with_floor(const double a,
                                             const double b,
                                             const double sigma,
                                             const double floor) {
  if (!std::isfinite(a) || !std::isfinite(b)) {
    return std::numeric_limits<double>::infinity();
  }
  const double scale = std::max(std::abs(a), std::abs(b));
  if (scale <= floor) {
    return 0.0;
  }
  return std::abs(a - sigma * b) / scale;
}

int pairwise_depth(const std::size_t terms) {
  if (terms <= 1U) {
    return 1;
  }
  int depth = 0;
  std::size_t covered = 1U;
  while (covered < terms) {
    covered <<= 1U;
    ++depth;
  }
  return depth + 1;
}

double gamma_bound(const std::size_t terms) {
  const double d = static_cast<double>(pairwise_depth(terms));
  const double du = d * std::numeric_limits<double>::epsilon();
  if (!(du < 1.0)) {
    throw std::runtime_error(
        "checkpoint-swap-center conservation reduction depth overflow");
  }
  return du / (1.0 - du);
}

double legendre_p(const int ell, const double mu) {
  if (ell == 0) {
    return 1.0;
  }
  if (ell == 1) {
    return mu;
  }
  double p_nm2 = 1.0;
  double p_nm1 = mu;
  for (int n = 2; n <= ell; ++n) {
    const double p_n =
        ((2.0 * static_cast<double>(n) - 1.0) * mu * p_nm1 -
         (static_cast<double>(n) - 1.0) * p_nm2) /
        static_cast<double>(n);
    p_nm2 = p_nm1;
    p_nm1 = p_n;
  }
  return p_nm1;
}

bool files_byte_equal(const std::filesystem::path& lhs,
                      const std::filesystem::path& rhs) {
  std::error_code ec_lhs;
  std::error_code ec_rhs;
  const auto lhs_size = std::filesystem::file_size(lhs, ec_lhs);
  const auto rhs_size = std::filesystem::file_size(rhs, ec_rhs);
  if (ec_lhs || ec_rhs || lhs_size != rhs_size) {
    return false;
  }
  std::ifstream left(lhs, std::ios::binary);
  std::ifstream right(rhs, std::ios::binary);
  if (!left || !right) {
    return false;
  }
  std::array<char, 1U << 16U> left_buffer{};
  std::array<char, 1U << 16U> right_buffer{};
  while (left && right) {
    left.read(left_buffer.data(),
              static_cast<std::streamsize>(left_buffer.size()));
    right.read(right_buffer.data(),
               static_cast<std::streamsize>(right_buffer.size()));
    const std::streamsize left_count = left.gcount();
    const std::streamsize right_count = right.gcount();
    if (left_count != right_count ||
        !std::equal(left_buffer.begin(),
                    left_buffer.begin() + left_count,
                    right_buffer.begin())) {
      return false;
    }
  }
  return left.eof() && right.eof();
}

std::filesystem::path temporary_sibling_path(
    const std::filesystem::path& output) {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path parent =
      output.parent_path().empty() ? std::filesystem::path(".")
                                   : output.parent_path();
  for (int attempt = 0; attempt < 1000; ++attempt) {
    const auto name = output.filename().string() + ".swap." +
                      std::to_string(stamp) + "." +
                      std::to_string(attempt) + ".tmp";
    const auto candidate = parent / name;
    if (!std::filesystem::exists(candidate)) {
      return candidate;
    }
  }
  throw std::runtime_error(
      "checkpoint-swap-center could not allocate a temporary output name");
}

void require_publish_target_available(const std::filesystem::path& output) {
  if (output.empty()) {
    throw std::runtime_error(
        "checkpoint-swap-center output checkpoint path is empty");
  }
  if (output.extension() != ".h5") {
    throw std::runtime_error(
        "checkpoint-swap-center output checkpoint must end in .h5");
  }
  if (std::filesystem::exists(output)) {
    throw std::runtime_error(
        "checkpoint-swap-center refuses to overwrite existing output: " +
        output.string());
  }
  const auto parent = output.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
}

void atomic_publish(const std::filesystem::path& temporary,
                    const std::filesystem::path& output) {
  if (std::filesystem::exists(output)) {
    throw std::runtime_error(
        "checkpoint-swap-center refuses to overwrite output at publish time: " +
        output.string());
  }
  std::error_code rename_error;
  std::filesystem::rename(temporary, output, rename_error);
  if (rename_error) {
    throw std::runtime_error(
        "checkpoint-swap-center failed to publish '" + output.string() +
        "': " + rename_error.message());
  }
}

template <typename Tag>
std::vector<double> field_to_host(const core::Field1D<Tag>& field) {
  std::vector<double> host(field.size(), 0.0);
  if (!host.empty()) {
    field.copy_to_host(host.data());
  }
  return host;
}

template <typename Tag>
void copy_field_prefix(const core::Field1D<Tag>& source,
                       core::Field1D<Tag>& target,
                       const std::size_t count,
                       const std::string_view name) {
  if (source.size() < count || target.size() < count) {
    throw std::runtime_error(
        "checkpoint-swap-center field prefix extent mismatch for " +
        std::string(name));
  }
  auto source_host = field_to_host(source);
  auto target_host = field_to_host(target);
  std::copy_n(source_host.begin(), count, target_host.begin());
  if (!target_host.empty()) {
    target.copy_from_host(target_host.data());
  }
}

template <typename T>
void copy_vector_prefix(const std::vector<T>& source,
                        std::vector<T>& target,
                        const std::size_t count,
                        const std::string_view name) {
  if (source.size() < count || target.size() < count) {
    throw std::runtime_error(
        "checkpoint-swap-center vector prefix extent mismatch for " +
        std::string(name));
  }
  std::copy_n(source.begin(), count, target.begin());
}

void copy_restart_cell_prefix(const core::State& parent,
                              core::State& target,
                              const std::size_t kept_cells,
                              const std::size_t n_materials,
                              const bool per_material_enabled) {
  copy_field_prefix(parent.rho, target.rho, kept_cells, "hydro/rho");
  copy_field_prefix(parent.Te, target.Te, kept_cells, "hydro/Te");
  copy_field_prefix(parent.Ti, target.Ti, kept_cells, "hydro/Ti");
  copy_field_prefix(parent.ee, target.ee, kept_cells, "hydro/ee");
  copy_field_prefix(parent.ei, target.ei, kept_cells, "hydro/ei");
  copy_field_prefix(parent.Pe, target.Pe, kept_cells, "hydro/Pe");
  copy_field_prefix(parent.Pi, target.Pi, kept_cells, "hydro/Pi");
  copy_field_prefix(parent.mass, target.mass, kept_cells, "hydro/mass");
  copy_field_prefix(parent.vol, target.vol, kept_cells, "hydro/vol");
  copy_field_prefix(parent.Qvisc, target.Qvisc, kept_cells, "hydro/Qvisc");
  copy_field_prefix(parent.zbar, target.zbar, kept_cells, "hydro/zbar");
  copy_field_prefix(parent.volFrac,
                    target.volFrac,
                    kept_cells * n_materials,
                    "hydro/volFrac");

  if (!parent.shock_time.empty() && !target.shock_time.empty()) {
    copy_field_prefix(parent.shock_time,
                      target.shock_time,
                      kept_cells,
                      "hydro/shock_time");
  }
  if (!parent.adaptive_av_gate.empty() &&
      !target.adaptive_av_gate.empty()) {
    copy_field_prefix(parent.adaptive_av_gate,
                      target.adaptive_av_gate,
                      kept_cells,
                      "hydro/adaptive_av_gate");
  }
  if (!parent.eta_compatible.empty() && !target.eta_compatible.empty()) {
    copy_field_prefix(parent.eta_compatible,
                      target.eta_compatible,
                      kept_cells,
                      "hydro/eta_compatible");
  }
  if (!parent.hllc_mom_z_cell.empty() &&
      !target.hllc_mom_z_cell.empty()) {
    copy_field_prefix(parent.hllc_mom_z_cell,
                      target.hllc_mom_z_cell,
                      kept_cells,
                      "hydro/hllc_mom_z_cell");
    target.hllc_mom_z_cell_initialized =
        parent.hllc_mom_z_cell_initialized;
  }
  if (!parent.cv_e.empty() && !target.cv_e.empty()) {
    copy_field_prefix(parent.cv_e, target.cv_e, kept_cells, "hydro/cv_e");
    copy_field_prefix(parent.cv_i, target.cv_i, kept_cells, "hydro/cv_i");
    copy_field_prefix(parent.cs, target.cs, kept_cells, "hydro/cs");
  }

  if (parent.corner_stride != target.corner_stride) {
    throw std::runtime_error(
        "checkpoint-swap-center parent/target corner stride mismatch");
  }
  const std::size_t kept_corners =
      kept_cells * static_cast<std::size_t>(target.corner_stride);
  if (parent.corner_mass.size() < kept_corners ||
      target.corner_mass.size() < kept_corners ||
      !parent.corner_mass_initialized) {
    throw std::runtime_error(
        "checkpoint-swap-center requires authoritative hydro/corner_mass in checkpoint A");
  }
  copy_field_prefix(parent.corner_mass,
                    target.corner_mass,
                    kept_corners,
                    "hydro/corner_mass");
  target.corner_mass_initialized = true;
  target.corner_mass_is_lagrangian_invariant =
      parent.corner_mass_is_lagrangian_invariant;

  if (per_material_enabled) {
    const std::size_t kept_material_entries = kept_cells * n_materials;
    copy_field_prefix(parent.mass_per_material,
                      target.mass_per_material,
                      kept_material_entries,
                      "hydro/per_material/v1/mass");
    copy_field_prefix(parent.Ee_per_material,
                      target.Ee_per_material,
                      kept_material_entries,
                      "hydro/per_material/v1/Ee");
    copy_field_prefix(parent.Ei_per_material,
                      target.Ei_per_material,
                      kept_material_entries,
                      "hydro/per_material/v1/Ei");
    target.Te_per_material_valid.assign(
        target.Te_per_material_valid.size(), static_cast<std::uint8_t>(0));
    target.Ti_per_material_valid.assign(
        target.Ti_per_material_valid.size(), static_cast<std::uint8_t>(0));
  }

  if (target.hydro_active.size() < target.rho.size()) {
    target.hydro_active.assign(target.rho.size(), static_cast<std::int8_t>(1));
  }
  if (!parent.hydro_active.empty()) {
    copy_vector_prefix(parent.hydro_active,
                       target.hydro_active,
                       kept_cells,
                       "hydro_flags/hydro_active");
    target.note_hydro_active_host_write();
  }
  if (!parent.merge_tombstone.empty()) {
    target.merge_tombstone.assign(target.rho.size(),
                                  static_cast<std::uint8_t>(0));
    copy_vector_prefix(parent.merge_tombstone,
                       target.merge_tombstone,
                       kept_cells,
                       "metadata/merge_tombstone");
  }
}

void initialize_and_copy_restart_burn_state(
    const core::State& parent,
    core::State& target,
    const core::Config& cfg,
    const std::size_t kept_cells) {
  if (!cfg.burn.enabled) {
    return;
  }

  constexpr std::size_t kSpecies =
      static_cast<std::size_t>(burn::kNumSpecies);
  const std::size_t parent_cells = parent.rho.size();
  const std::size_t target_cells = target.rho.size();
  if (parent.burn_n_host.size() != parent_cells * kSpecies) {
    throw std::runtime_error(
        "checkpoint-swap-center burn-enabled checkpoint A is missing complete hydro/burn_n_* state");
  }

  std::vector<int> fuel_materials;
  fuel_materials.reserve(cfg.burn.fuel_materials.size());
  for (const auto& fuel_name : cfg.burn.fuel_materials) {
    const auto material = std::find_if(
        cfg.materials.materials.begin(),
        cfg.materials.materials.end(),
        [&](const auto& candidate) { return candidate.name == fuel_name; });
    if (material == cfg.materials.materials.end()) {
      throw std::runtime_error(
          "checkpoint-swap-center burn fuel material is absent from the segment-2 deck: " +
          fuel_name);
    }
    fuel_materials.push_back(static_cast<int>(
        std::distance(cfg.materials.materials.begin(), material)));
  }

  const double Abar = cfg.burn.x_D * burn::species_A(burn::kD) +
                      cfg.burn.x_T * burn::species_A(burn::kT) +
                      cfg.burn.x_He3 * burn::species_A(burn::kHe3);
  if (!(std::isfinite(Abar) && Abar > 0.0)) {
    throw std::runtime_error(
        "checkpoint-swap-center cannot initialize fresh burn inventory from the segment-2 deck");
  }
  const std::size_t n_materials = cfg.materials.materials.size();
  const auto volume_fraction = field_to_host(target.volFrac);
  target.burn_n_host.assign(target_cells * kSpecies, 0.0);
  for (std::size_t c = kept_cells; c < target_cells; ++c) {
    double fuel_fraction = 0.0;
    for (const int material : fuel_materials) {
      fuel_fraction += volume_fraction[
          c * n_materials + static_cast<std::size_t>(material)];
    }
    if (fuel_fraction <= 0.0) {
      continue;
    }
    target.burn_n_host[c * kSpecies + burn::kD] =
        cfg.burn.x_D * fuel_fraction /
        (Abar * core::constants::proton_mass);
    target.burn_n_host[c * kSpecies + burn::kT] =
        cfg.burn.x_T * fuel_fraction /
        (Abar * core::constants::proton_mass);
    target.burn_n_host[c * kSpecies + burn::kHe3] =
        cfg.burn.x_He3 * fuel_fraction /
        (Abar * core::constants::proton_mass);
  }
  std::copy_n(parent.burn_n_host.begin(),
              kept_cells * kSpecies,
              target.burn_n_host.begin());

  const auto copy_cell_vector = [&](const std::vector<double>& source,
                                    std::vector<double>& destination,
                                    const std::string_view name) {
    destination.assign(target_cells, 0.0);
    if (!source.empty()) {
      copy_vector_prefix(source, destination, kept_cells, name);
    }
  };
  copy_cell_vector(parent.burn_rate_host,
                   target.burn_rate_host,
                   "hydro/burn_rate");
  copy_cell_vector(parent.burn_Q_e_host,
                   target.burn_Q_e_host,
                   "hydro/burn_Q_e");
  copy_cell_vector(parent.burn_Q_i_host,
                   target.burn_Q_i_host,
                   "hydro/burn_Q_i");
  copy_cell_vector(parent.burn_eps_cum_host,
                   target.burn_eps_cum_host,
                   "hydro/burn_eps_cum");
  target.burn_enabled_any = true;

  target.burn_diffusion_any = false;
  target.burn_Ng.reset(0);
  if (!parent.burn_Ng.empty()) {
    const std::size_t groups =
        static_cast<std::size_t>(cfg.burn.diffusion_groups);
    const std::size_t parent_slot = groups * parent_cells;
    const std::size_t target_slot = groups * target_cells;
    if (parent.burn_Ng.size() != 6U * parent_slot) {
      throw std::runtime_error(
          "checkpoint-swap-center hydro/burn_Ng_slot* extent does not match checkpoint A");
    }
    std::vector<double> parent_spectrum;
    parent.burn_Ng.copy_to_host(parent_spectrum);
    std::vector<double> target_spectrum(6U * target_slot, 0.0);
    for (std::size_t slot = 0; slot < 6U; ++slot) {
      for (std::size_t group = 0; group < groups; ++group) {
        const std::size_t parent_offset =
            slot * parent_slot + group * parent_cells;
        const std::size_t target_offset =
            slot * target_slot + group * target_cells;
        std::copy_n(parent_spectrum.begin() +
                        static_cast<std::ptrdiff_t>(parent_offset),
                    kept_cells,
                    target_spectrum.begin() +
                        static_cast<std::ptrdiff_t>(target_offset));
      }
    }
    target.burn_Ng.reset(target_spectrum.size());
    target.burn_Ng.copy_from_host(target_spectrum);
    target.burn_diffusion_any = true;
  }
}

void copy_restart_node_prefix(const core::State& parent,
                              core::State& target,
                              const std::size_t kept_nodes) {
  copy_field_prefix(parent.x_r, target.x_r, kept_nodes, "mesh/x_r");
  copy_field_prefix(parent.x_z, target.x_z, kept_nodes, "mesh/x_z");
  copy_field_prefix(parent.v_r, target.v_r, kept_nodes, "mesh/v_r");
  copy_field_prefix(parent.v_z, target.v_z, kept_nodes, "mesh/v_z");
  copy_field_prefix(parent.x_r_reference,
                    target.x_r_reference,
                    kept_nodes,
                    "mesh/ale_reference/v1/x_r_reference");
  copy_field_prefix(parent.x_z_reference,
                    target.x_z_reference,
                    kept_nodes,
                    "mesh/ale_reference/v1/x_z_reference");
  copy_field_prefix(parent.x_r_shock_target,
                    target.x_r_shock_target,
                    kept_nodes,
                    "mesh/ale_reference/v1/x_r_shock_target");
  copy_field_prefix(parent.x_z_shock_target,
                    target.x_z_shock_target,
                    kept_nodes,
                    "mesh/ale_reference/v1/x_z_shock_target");
}

void copy_restart_ledgers(const core::State& parent,
                          core::State& target,
                          const core::Config& cfg) {
  target.t = parent.t;
  target.step = parent.step;
  target.dt = parent.dt;
  target.adaptive_av_r0 = parent.adaptive_av_r0;
  target.adaptive_av_last_rs = parent.adaptive_av_last_rs;
  target.adaptive_av_last_us = parent.adaptive_av_last_us;
  target.adaptive_av_rs_min = parent.adaptive_av_rs_min;
  target.adaptive_av_tracker_steps = parent.adaptive_av_tracker_steps;
  target.adaptive_av_mode = parent.adaptive_av_mode;
  target.adaptive_av_tracker_valid = parent.adaptive_av_tracker_valid;
  target.adaptive_av_bounce_seen = parent.adaptive_av_bounce_seen;

  target.E_safety = parent.E_safety;
  target.E_numerical_loss = parent.E_numerical_loss;
  target.E_laser_deposited = parent.E_laser_deposited;
  target.E_cbet_iaw_step = parent.E_cbet_iaw_step;
  target.E_cbet_iaw = parent.E_cbet_iaw;
  target.E_burn_released = parent.E_burn_released;
  target.E_burn_dep_e = parent.E_burn_dep_e;
  target.E_burn_dep_i = parent.E_burn_dep_i;
  target.E_burn_esc_charged = parent.E_burn_esc_charged;
  target.E_burn_esc_neutron = parent.E_burn_esc_neutron;
  target.N_burn_neutrons_dt = parent.N_burn_neutrons_dt;
  target.N_burn_neutrons_dd = parent.N_burn_neutrons_dd;
  target.E_burn_inflight = parent.E_burn_inflight;
  target.E_laser_escaped = parent.E_laser_escaped;
  target.E_rad_escaped = parent.E_rad_escaped;
  target.E_floor_injected = parent.E_floor_injected;
  target.E_pdV_bdry = parent.E_pdV_bdry;
  target.E_Marshak_in = parent.E_Marshak_in;
  target.E_solver = parent.E_solver;

  // The swap begins a new mesh epoch. Empty budget vectors are the reader/
  // ALE fresh-start sentinel; axis_margin_initial=-1 is re-baselined on the
  // next applicable ALE observation. Counting the swap as the last applied
  // rezone suppresses an immediate cadence rezone at the restart boundary.
  target.ale_last_applied_step = target.step;
  target.axis_margin_initial = -1.0;
  target.axis_mass_initial.clear();
  target.axis_inflow_budget.clear();

  target.t_next_plot =
      cfg.output.plot_every_s > 0.0 ? target.t : -1.0;
  target.t_next_history =
      cfg.output.history_every_s > 0.0
          ? target.t + cfg.output.history_every_s
          : -1.0;
  target.t_next_checkpoint =
      cfg.output.checkpoint_every_s > 0.0
          ? target.t + cfg.output.checkpoint_every_s
          : -1.0;
}

std::vector<std::uint8_t> effective_cell_nverts(
    const mesh::Mesh& mesh_value) {
  if (!mesh_value.cell_nverts.empty()) {
    return mesh_value.cell_nverts;
  }
  return std::vector<std::uint8_t>(
      static_cast<std::size_t>(mesh_value.topo.n_cells),
      static_cast<std::uint8_t>(4));
}

std::pair<double, double> cell_centroid(
    const mesh::MultiBlockTopology& mb,
    const std::vector<std::uint8_t>& nverts,
    const std::vector<double>& x_r_host,
    const std::vector<double>& x_z_host,
    const std::size_t c) {
  const int begin = mb.cell_node_csr_offsets[c];
  const int count = static_cast<int>(nverts[c]);
  std::set<int> active_nodes;
  for (int k = 0; k < count; ++k) {
    const int node = mb.cell_node_csr_indices[
        static_cast<std::size_t>(begin + k)];
    if (node >= 0) {
      active_nodes.insert(node);
    }
  }
  double r = 0.0;
  double z = 0.0;
  for (const int node : active_nodes) {
    r += x_r_host[static_cast<std::size_t>(node)];
    z += x_z_host[static_cast<std::size_t>(node)];
  }
  const double inverse_count =
      1.0 / static_cast<double>(active_nodes.size());
  return {r * inverse_count, z * inverse_count};
}

struct ThermodynamicReference {
  double rho = 0.0;
  double Te = 0.0;
  double Ti = 0.0;
};

struct UniformReference : ThermodynamicReference {
  double cs = 0.0;
};

ThermodynamicReference replaced_region_means(
    const core::State& state,
    const std::size_t first_cell) {
  const auto rho = field_to_host(state.rho);
  const auto Te = field_to_host(state.Te);
  const auto Ti = field_to_host(state.Ti);
  const auto vol = field_to_host(state.vol);
  if (first_cell >= rho.size()) {
    throw std::runtime_error(
        "checkpoint-swap-center reference region has no cells");
  }
  long double weight_sum = 0.0L;
  long double rho_sum = 0.0L;
  long double Te_sum = 0.0L;
  long double Ti_sum = 0.0L;
  for (std::size_t c = first_cell; c < rho.size(); ++c) {
    const double weight = vol[c];
    if (!(std::isfinite(weight) && weight > 0.0)) {
      throw std::runtime_error(
          "checkpoint-swap-center reference region has non-positive volume");
    }
    if (!(std::isfinite(rho[c]) && rho[c] > 0.0 &&
          std::isfinite(Te[c]) && Te[c] > 0.0 &&
          std::isfinite(Ti[c]) && Ti[c] > 0.0)) {
      throw std::runtime_error(
          "checkpoint-swap-center quiescence gate found non-finite rho/Te/Ti");
    }
    weight_sum += static_cast<long double>(weight);
    rho_sum += static_cast<long double>(weight) * rho[c];
    Te_sum += static_cast<long double>(weight) * Te[c];
    Ti_sum += static_cast<long double>(weight) * Ti[c];
  }
  if (!(weight_sum > 0.0L)) {
    throw std::runtime_error(
        "checkpoint-swap-center reference must have positive rho/Te/Ti");
  }
  ThermodynamicReference out;
  out.rho = static_cast<double>(rho_sum / weight_sum);
  out.Te = static_cast<double>(Te_sum / weight_sum);
  out.Ti = static_cast<double>(Ti_sum / weight_sum);
  if (!(out.rho > 0.0 && out.Te > 0.0 && out.Ti > 0.0)) {
    throw std::runtime_error(
        "checkpoint-swap-center reference must have positive rho/Te/Ti");
  }
  return out;
}

UniformReference uniform_reference(const core::State& state,
                                   const std::size_t first_cell) {
  const ThermodynamicReference means =
      replaced_region_means(state, first_cell);
  const auto cs = field_to_host(state.cs);
  if (cs.size() != state.rho.size()) {
    throw std::runtime_error(
        "checkpoint-swap-center reference region has invalid rho/Te/Ti/cs");
  }
  double cs_min = std::numeric_limits<double>::infinity();
  for (std::size_t c = first_cell; c < cs.size(); ++c) {
    if (!(std::isfinite(cs[c]) && cs[c] > 0.0)) {
      throw std::runtime_error(
          "checkpoint-swap-center reference region has invalid rho/Te/Ti/cs");
    }
    cs_min = std::min(cs_min, cs[c]);
  }
  if (!std::isfinite(cs_min)) {
    throw std::runtime_error(
        "checkpoint-swap-center cannot derive positive rho/T/cs reference");
  }
  UniformReference out;
  out.rho = means.rho;
  out.Te = means.Te;
  out.Ti = means.Ti;
  out.cs = cs_min;
  return out;
}

std::set<int> incident_nodes(const mesh::MultiBlockTopology& topology,
                             const std::size_t first_cell,
                             const std::size_t n_cells) {
  std::set<int> nodes;
  for (std::size_t c = first_cell; c < n_cells; ++c) {
    const int begin = topology.cell_node_csr_offsets[c];
    const int end = topology.cell_node_csr_offsets[c + 1U];
    for (int entry = begin; entry < end; ++entry) {
      const int node = topology.cell_node_csr_indices[
          static_cast<std::size_t>(entry)];
      if (node >= 0) {
        nodes.insert(node);
      }
    }
  }
  return nodes;
}

double replaced_region_mirror_margin(const core::State& state,
                                     const std::size_t first_cell,
                                     const char* label,
                                     const CheckpointSwapCenterThresholds& thresholds,
                                     const bool dump_offenders,
                                     const bool include_internal_energy) {
  if (!state.mesh.topo.multiblock.has_value()) {
    throw std::runtime_error(
        "checkpoint-swap-center mirror gate requires multiblock topology");
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const std::size_t n_cells = state.rho.size();
  const auto nverts = effective_cell_nverts(state.mesh);
  const auto rho = field_to_host(state.rho);
  const auto Te = field_to_host(state.Te);
  const auto Ti = field_to_host(state.Ti);
  const auto mass = field_to_host(state.mass);
  const auto ee = include_internal_energy ? field_to_host(state.ee)
                                          : std::vector<double>{};
  const auto ei = include_internal_energy ? field_to_host(state.ei)
                                          : std::vector<double>{};
  const auto vol = field_to_host(state.vol);
  const auto x_r = field_to_host(state.x_r);
  const auto x_z = field_to_host(state.x_z);
  const auto v_r = field_to_host(state.v_r);
  const auto v_z = field_to_host(state.v_z);

  if (mb.south_node_of.size() != x_r.size()) {
    throw std::runtime_error(
        "checkpoint-swap-center parent mirror map is unavailable");
  }
  struct MirrorOffender {
    double margin;
    const char* kind;
    std::size_t id;
    std::size_t south_id;
    double value;
    double south_value;
    bool is_node;
    double x_r;
    double x_z;
    double south_x_r;
    double south_x_z;
    int block;
  };
  std::vector<MirrorOffender> offenders;
  auto record_offender = [&offenders](const MirrorOffender& offender) {
    if (!(offender.margin > 1.0e-6)) {
      return;
    }
    offenders.push_back(offender);
    std::stable_sort(
        offenders.begin(), offenders.end(),
        [](const MirrorOffender& lhs, const MirrorOffender& rhs) {
          return lhs.margin > rhs.margin;
        });
    if (offenders.size() > 8U) {
      offenders.pop_back();
    }
  };

  std::map<std::vector<int>, std::size_t> cell_by_nodes;
  std::map<int, int> incident_node_block;
  for (std::size_t c = first_cell; c < n_cells; ++c) {
    const int begin = mb.cell_node_csr_offsets[c];
    const int count = static_cast<int>(nverts[c]);
    std::vector<int> key;
    key.reserve(static_cast<std::size_t>(count));
    for (int k = 0; k < count; ++k) {
      const int node = mb.cell_node_csr_indices[
          static_cast<std::size_t>(begin + k)];
      key.push_back(node);
      if (node >= 0 &&
          incident_node_block.find(node) == incident_node_block.end()) {
        const int block =
            c < mb.cell_block_id.size() ? mb.cell_block_id[c] : -1;
        incident_node_block.emplace(node, block);
      }
    }
    std::sort(key.begin(), key.end());
    if (!cell_by_nodes.emplace(std::move(key), c).second) {
      throw std::runtime_error(
          "checkpoint-swap-center duplicate cell vertex set in mirror gate");
    }
  }

  double margin = 0.0;
  for (std::size_t c = first_cell; c < n_cells; ++c) {
    const int begin = mb.cell_node_csr_offsets[c];
    const int count = static_cast<int>(nverts[c]);
    std::vector<int> mirrored;
    mirrored.reserve(static_cast<std::size_t>(count));
    for (int k = 0; k < count; ++k) {
      const int node = mb.cell_node_csr_indices[
          static_cast<std::size_t>(begin + k)];
      if (node < 0 || static_cast<std::size_t>(node) >= mb.south_node_of.size()) {
        throw std::runtime_error(
            "checkpoint-swap-center invalid node in mirror gate");
      }
      const int south = mb.south_node_of[static_cast<std::size_t>(node)];
      if (south < 0) {
        throw std::runtime_error(
            "checkpoint-swap-center incomplete node mirror map");
      }
      mirrored.push_back(south);
    }
    std::sort(mirrored.begin(), mirrored.end());
    const auto match = cell_by_nodes.find(mirrored);
    if (match == cell_by_nodes.end()) {
      throw std::runtime_error(
          "checkpoint-swap-center replaced cell has no mirrored partner");
    }
    const std::size_t south_cell = match->second;
    auto check_cell_field = [&](const char* kind, const double value,
                                const double south_value) {
      const double field_margin = relative_difference(value, south_value);
      margin = std::max(margin, field_margin);
      record_offender(MirrorOffender{field_margin,
                                     kind,
                                     c,
                                     south_cell,
                                     value,
                                     south_value,
                                     false,
                                     0.0,
                                     0.0,
                                     0.0,
                                     0.0,
                                     -1});
    };
    check_cell_field("rho", rho[c], rho[south_cell]);
    check_cell_field("Te", Te[c], Te[south_cell]);
    check_cell_field("Ti", Ti[c], Ti[south_cell]);
    check_cell_field("mass", mass[c], mass[south_cell]);
    if (include_internal_energy) {
      check_cell_field("ee", ee[c], ee[south_cell]);
      check_cell_field("ei", ei[c], ei[south_cell]);
    }
    check_cell_field("vol", vol[c], vol[south_cell]);
  }

  const auto nodes = incident_nodes(mb, first_cell, n_cells);
  for (const int node : nodes) {
    const int south = mb.south_node_of[static_cast<std::size_t>(node)];
    if (south < 0 || static_cast<std::size_t>(south) >= x_r.size()) {
      throw std::runtime_error(
          "checkpoint-swap-center incomplete incident-node mirror map");
    }
    const std::size_t node_id = static_cast<std::size_t>(node);
    const std::size_t south_id = static_cast<std::size_t>(south);
    const auto block_it = incident_node_block.find(node);
    const int block =
        block_it != incident_node_block.end() ? block_it->second : -1;
    auto check_node_field = [&](const char* kind, const double value,
                                const double south_value,
                                const double field_margin) {
      margin = std::max(margin, field_margin);
      record_offender(MirrorOffender{field_margin,
                                     kind,
                                     node_id,
                                     south_id,
                                     value,
                                     south_value,
                                     true,
                                     x_r[node_id],
                                     x_z[node_id],
                                     x_r[south_id],
                                     x_z[south_id],
                                     block});
    };
    check_node_field("node-x_r",
                     x_r[node_id],
                     x_r[south_id],
                     relative_difference(x_r[node_id], x_r[south_id]));
    check_node_field(
        "node-x_z",
        x_z[node_id],
        x_z[south_id],
        signed_relative_difference_with_floor(x_z[node_id],
                                              x_z[south_id],
                                              -1.0,
                                              thresholds.mirror_z_floor_cm));
    check_node_field(
        "node-v_r",
        v_r[node_id],
        v_r[south_id],
        signed_relative_difference_with_floor(v_r[node_id],
                                              v_r[south_id],
                                              1.0,
                                              thresholds.mirror_v_floor_cm_s));
    check_node_field(
        "node-v_z",
        v_z[node_id],
        v_z[south_id],
        signed_relative_difference_with_floor(v_z[node_id],
                                              v_z[south_id],
                                              -1.0,
                                              thresholds.mirror_v_floor_cm_s));
  }
  if (dump_offenders) {
    for (const auto& offender : offenders) {
      std::ostringstream out;
      out << std::setprecision(17)
          << "checkpoint-swap-center mirror offender label=" << label
          << " kind=" << offender.kind;
      if (offender.is_node) {
        out << " node=" << offender.id << " south=" << offender.south_id;
      } else {
        out << " c=" << offender.id << " south_cell=" << offender.south_id;
      }
      out << " value=" << offender.value
          << " south_value=" << offender.south_value;
      if (offender.is_node) {
        out << " x_r=" << offender.x_r << " x_z=" << offender.x_z
            << " south_x_r=" << offender.south_x_r
            << " south_x_z=" << offender.south_x_z
            << " block=" << offender.block;
      }
      out << " margin=" << offender.margin << '\n';
      std::cerr << out.str();
    }
  }
  return margin;
}

void print_mirror_map_summary(const core::State& state, const char* label) {
  const auto x_z = field_to_host(state.x_z);
  const std::size_t n_nodes = x_z.size();
  if (!state.mesh.topo.multiblock.has_value()) {
    std::cerr << "checkpoint-swap-center mirror info label=" << label
              << " self_mirror_equator_nodes=0 max_z_odd_residual_cm=0\n";
    return;
  }
  const auto& south_node_of = state.mesh.topo.multiblock->south_node_of;
  std::size_t self = 0;
  double max_z_odd_residual = 0.0;
  for (std::size_t node = 0; node < south_node_of.size(); ++node) {
    const int south = south_node_of[node];
    if (south < 0) {
      continue;
    }
    if (static_cast<std::size_t>(south) == node) {
      ++self;
    }
    if (node < n_nodes && static_cast<std::size_t>(south) < n_nodes) {
      max_z_odd_residual = std::max(
          max_z_odd_residual,
          std::abs(x_z[node] + x_z[static_cast<std::size_t>(south)]));
    }
  }
  std::ostringstream out;
  out << std::setprecision(17)
      << "checkpoint-swap-center mirror info label=" << label
      << " self_mirror_equator_nodes=" << self
      << " max_z_odd_residual_cm=" << max_z_odd_residual << '\n';
  std::cerr << out.str();
}

constexpr double kPi = 3.141592653589793238462643383279502884;

struct SeamPoint {
  double r = 0.0;
  double z = 0.0;
};

SeamPoint constructed_seam_point(
    const double radius,
    const int k,
    const int columns,
    const int master_columns,
    const int pole_cap_m,
    const double pole_cap_alpha,
    const std::vector<int>& master_labels) {
  const bool south = k > columns / 2;
  const int north_k = south ? columns - k : k;
  if (north_k == 0) {
    return SeamPoint{0.0, south ? -radius : radius};
  }
  if (north_k == columns / 2) {
    return SeamPoint{radius, 0.0};
  }

  double theta = 0.0;
  if (!master_labels.empty()) {
    theta = static_cast<double>(
                master_labels[static_cast<std::size_t>(north_k)]) *
            kPi / static_cast<double>(master_columns);
  } else {
    const bool cap_active = pole_cap_m > 0 && pole_cap_alpha > 0.0;
    if (!cap_active) {
      theta = static_cast<double>(north_k) * kPi /
              static_cast<double>(columns);
    } else {
      const double x =
          static_cast<double>(north_k) *
          (static_cast<double>(master_columns) /
           static_cast<double>(columns));
      if (x >= static_cast<double>(pole_cap_m)) {
        theta = static_cast<double>(north_k) * kPi /
                static_cast<double>(columns);
      } else {
        const double theta_u =
            x * kPi / static_cast<double>(master_columns);
        const double t = x / static_cast<double>(pole_cap_m);
        const double theta_mu = std::acos(
            1.0 -
            t * (1.0 -
                 std::cos(static_cast<double>(pole_cap_m) * kPi /
                          static_cast<double>(master_columns))));
        const double smoothstep =
            t * t * t * (10.0 + t * (-15.0 + 6.0 * t));
        theta = theta_u +
                pole_cap_alpha * (1.0 - smoothstep) *
                    (theta_mu - theta_u);
      }
    }
  }
  return SeamPoint{radius * std::sin(theta),
                   (south ? -1.0 : 1.0) * radius * std::cos(theta)};
}

std::vector<int> seam_master_labels(
    const core::Config& cfg,
    const core::PolarTierLayout& truncated) {
  if (!cfg.mesh.polar_tier_dendrite_enabled) {
    return {};
  }
  if (truncated.schedule.empty() ||
      truncated.schedule.front().kind != core::PolarTierEntityKind::TIER) {
    throw std::runtime_error(
        "checkpoint-swap-center cannot locate the truncated seam tier");
  }
  constexpr std::array<int, 6> tier_ring_indices = {{0, 1, 3, 4, 5, 6}};
  const int tier = truncated.schedule.front().index;
  if (tier < 0 ||
      static_cast<std::size_t>(tier) >= tier_ring_indices.size()) {
    throw std::runtime_error(
        "checkpoint-swap-center truncated seam tier index is invalid");
  }
  const int ring_index =
      tier_ring_indices[static_cast<std::size_t>(tier)];
  if (ring_index < 0 ||
      static_cast<std::size_t>(ring_index) >=
          truncated.dendrite_master_theta_node_labels.size()) {
    throw std::runtime_error(
        "checkpoint-swap-center truncated seam angular labels are unavailable");
  }
  return truncated.dendrite_master_theta_node_labels[
      static_cast<std::size_t>(ring_index)];
}

double measure_seam_displacement(
    const core::State& parent,
    const core::Config& cfg,
    const core::PolarTierLayout& truncated,
    const core::PolarTierTruncation& truncation) {
  const std::size_t kept_nodes =
      static_cast<std::size_t>(truncated.n_nodes);
  const int columns = truncation.n_theta_cut;
  if (columns <= 0 || columns % 2 != 0 || cfg.mesh.nz <= 0 ||
      kept_nodes < static_cast<std::size_t>(columns) + 1U) {
    throw std::runtime_error(
        "checkpoint-swap-center seam-displacement gate received an invalid truncated layout");
  }
  const auto x_r = field_to_host(parent.x_r);
  const auto x_z = field_to_host(parent.x_z);
  if (x_r.size() < kept_nodes || x_z.size() < kept_nodes) {
    throw std::runtime_error(
        "checkpoint-swap-center seam-displacement gate node extent mismatch");
  }
  const std::vector<int> master_labels =
      seam_master_labels(cfg, truncated);
  if (!master_labels.empty() &&
      master_labels.size() != static_cast<std::size_t>(columns) + 1U) {
    throw std::runtime_error(
        "checkpoint-swap-center seam-displacement gate angular-label extent mismatch");
  }
  const std::size_t seam_begin =
      kept_nodes - (static_cast<std::size_t>(columns) + 1U);
  double max_displacement = 0.0;
  for (int k = 0; k <= columns; ++k) {
    const SeamPoint expected = constructed_seam_point(
        truncation.r_cut,
        k,
        columns,
        cfg.mesh.nz,
        cfg.mesh.polar_tier_pole_cap_m,
        cfg.mesh.polar_tier_pole_cap_alpha,
        master_labels);
    const std::size_t node = seam_begin + static_cast<std::size_t>(k);
    const double displacement =
        std::hypot(x_r[node] - expected.r, x_z[node] - expected.z);
    if (!std::isfinite(displacement)) {
      return std::numeric_limits<double>::infinity();
    }
    max_displacement = std::max(max_displacement, displacement);
  }
  return max_displacement;
}

CheckpointSwapCenterGateMeasurements measure_a_side_quiescence(
    const core::State& parent,
    const core::Config& cfg,
    const core::PolarTierLayout& truncated,
    const core::PolarTierTruncation& truncation,
    const int lmax,
    const CheckpointSwapCenterThresholds& thresholds) {
  print_mirror_map_summary(parent, "A");
  CheckpointSwapCenterGateMeasurements measured;
  const auto parent_rho = field_to_host(parent.rho);
  const auto parent_Te = field_to_host(parent.Te);
  const auto parent_Ti = field_to_host(parent.Ti);
  const auto parent_vol = field_to_host(parent.vol);
  const auto parent_v_r = field_to_host(parent.v_r);
  const auto parent_v_z = field_to_host(parent.v_z);
  const auto parent_x_r = field_to_host(parent.x_r);
  const auto parent_x_z = field_to_host(parent.x_z);
  const auto parent_nverts = effective_cell_nverts(parent.mesh);
  const std::size_t first_parent_cell =
      static_cast<std::size_t>(truncated.n_cells);

  if (!parent.mesh.topo.multiblock.has_value()) {
    throw std::runtime_error(
        "checkpoint-swap-center quiescence gate requires parent multiblock topology");
  }
  const auto& mb = *parent.mesh.topo.multiblock;
  for (const int node :
       incident_nodes(mb, first_parent_cell, parent.rho.size())) {
    const double speed = std::hypot(
        parent_v_r[static_cast<std::size_t>(node)],
        parent_v_z[static_cast<std::size_t>(node)]);
    if (!std::isfinite(speed)) {
      throw std::runtime_error(
          "checkpoint-swap-center quiescence gate found non-finite velocity");
    }
    measured.max_speed_cm_s = std::max(measured.max_speed_cm_s, speed);
  }
  measured.max_seam_displacement_cm =
      measure_seam_displacement(parent, cfg, truncated, truncation);
  measured.max_mirror_relative =
      replaced_region_mirror_margin(
          parent, first_parent_cell, "A", thresholds, false, false);

  const ThermodynamicReference reference =
      replaced_region_means(parent, first_parent_cell);

  std::vector<CheckpointSwapCenterAngularSample> samples;
  samples.reserve(parent.rho.size() - first_parent_cell);
  for (std::size_t c = first_parent_cell; c < parent.rho.size(); ++c) {
    const double rho_delta = parent_rho[c] / reference.rho - 1.0;
    const double Te_delta = parent_Te[c] / reference.Te - 1.0;
    const double Ti_delta = parent_Ti[c] / reference.Ti - 1.0;
    if (!std::isfinite(rho_delta) || !std::isfinite(Te_delta) ||
        !std::isfinite(Ti_delta)) {
      throw std::runtime_error(
          "checkpoint-swap-center quiescence gate found non-finite rho/Te/Ti");
    }
    const auto [r, z] = cell_centroid(
        mb, parent_nverts, parent_x_r, parent_x_z, c);
    const double floored_z =
        std::abs(z) <= thresholds.mirror_z_floor_cm ? 0.0 : z;
    const double radius = std::hypot(r, floored_z);
    samples.push_back(CheckpointSwapCenterAngularSample{
        parent_vol[c],
        radius > 0.0 ? floored_z / radius : 0.0,
        rho_delta,
        Te_delta,
        Ti_delta});
  }
  measured.max_legendre_relative =
      checkpoint_swap_legendre_margin(samples, lmax);
  return measured;
}

CheckpointSwapCenterGateMeasurements measure_b_side_quiescence(
    const core::State& parent,
    const core::State& fresh_target,
    const std::size_t first_parent_cell,
    const std::size_t first_target_cell,
    const CheckpointSwapCenterGateMeasurements& a_side,
    const CheckpointSwapCenterThresholds& thresholds) {
  print_mirror_map_summary(fresh_target, "B");
  CheckpointSwapCenterGateMeasurements measured = a_side;
  const UniformReference reference =
      uniform_reference(fresh_target, first_target_cell);
  measured.max_mach = measured.max_speed_cm_s / reference.cs;
  const auto parent_rho = field_to_host(parent.rho);
  const auto parent_Te = field_to_host(parent.Te);
  const auto parent_Ti = field_to_host(parent.Ti);
  const auto fresh_rho = field_to_host(fresh_target.rho);
  const auto fresh_Te = field_to_host(fresh_target.Te);
  const auto fresh_Ti = field_to_host(fresh_target.Ti);
  for (std::size_t c = first_parent_cell; c < parent_rho.size(); ++c) {
    if (!std::isfinite(parent_rho[c]) || !std::isfinite(parent_Te[c]) ||
        !std::isfinite(parent_Ti[c])) {
      throw std::runtime_error(
          "checkpoint-swap-center quiescence gate found non-finite rho/Te/Ti");
    }
    measured.max_rho_relative = std::max(
        measured.max_rho_relative,
        std::abs(parent_rho[c] / reference.rho - 1.0));
    measured.max_temperature_relative = std::max(
        measured.max_temperature_relative,
        std::max(std::abs(parent_Te[c] / reference.Te - 1.0),
                 std::abs(parent_Ti[c] / reference.Ti - 1.0)));
  }
  for (std::size_t c = first_target_cell; c < fresh_rho.size(); ++c) {
    if (!std::isfinite(fresh_rho[c]) || !std::isfinite(fresh_Te[c]) ||
        !std::isfinite(fresh_Ti[c])) {
      throw std::runtime_error(
          "checkpoint-swap-center fresh center contains non-finite rho/Te/Ti");
    }
    measured.max_rho_relative = std::max(
        measured.max_rho_relative,
        std::abs(fresh_rho[c] / reference.rho - 1.0));
    measured.max_temperature_relative = std::max(
        measured.max_temperature_relative,
        std::max(std::abs(fresh_Te[c] / reference.Te - 1.0),
                 std::abs(fresh_Ti[c] / reference.Ti - 1.0)));
  }
  (void)replaced_region_mirror_margin(
      fresh_target, first_target_cell, "B", thresholds, false, true);
  return measured;
}

CheckpointSwapCenterConservationTotals replacement_totals(
    const core::State& state,
    const std::size_t first_cell) {
  if (!state.mesh.topo.multiblock.has_value()) {
    throw std::runtime_error(
        "checkpoint-swap-center conservation gate requires multiblock topology");
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const auto nverts = effective_cell_nverts(state.mesh);
  const auto mass = field_to_host(state.mass);
  const auto ee = field_to_host(state.ee);
  const auto ei = field_to_host(state.ei);
  const auto corner_mass = field_to_host(state.corner_mass);
  const auto v_r = field_to_host(state.v_r);
  const auto v_z = field_to_host(state.v_z);
  CheckpointSwapCenterConservationTotals totals;
  for (std::size_t c = first_cell; c < mass.size(); ++c) {
    const long double cell_mass = mass[c];
    const long double internal =
        cell_mass * static_cast<long double>(ee[c] + ei[c]);
    long double kinetic = 0.0L;
    const int csr_begin = mb.cell_node_csr_offsets[c];
    const int count = static_cast<int>(nverts[c]);
    for (int k = 0; k < count; ++k) {
      const std::size_t corner =
          c * static_cast<std::size_t>(state.corner_stride) +
          static_cast<std::size_t>(k);
      const int node = mb.cell_node_csr_indices[
          static_cast<std::size_t>(csr_begin + k)];
      const long double corner_m = corner_mass[corner];
      const long double vr = v_r[static_cast<std::size_t>(node)];
      const long double vz = v_z[static_cast<std::size_t>(node)];
      const long double kinetic_term =
          0.5L * corner_m * (vr * vr + vz * vz);
      const long double momentum_r_term = corner_m * vr;
      const long double momentum_z_term = corner_m * vz;
      kinetic += kinetic_term;
      totals.momentum_r += momentum_r_term;
      totals.momentum_z += momentum_z_term;
      totals.abs_energy_terms += std::abs(kinetic_term);
      totals.abs_momentum_r_terms += std::abs(momentum_r_term);
      totals.abs_momentum_z_terms += std::abs(momentum_z_term);
      ++totals.energy_terms;
      ++totals.momentum_terms;
    }
    totals.mass += cell_mass;
    totals.energy += internal + kinetic;
    totals.abs_mass_terms += std::abs(cell_mass);
    totals.abs_energy_terms += std::abs(internal);
    ++totals.mass_terms;
    ++totals.energy_terms;
  }
  return totals;
}

std::string format_gate_failure(
    const CheckpointSwapCenterGateMeasurements& measured,
    const CheckpointSwapCenterThresholds& limits) {
  std::ostringstream message;
  message << std::setprecision(17)
          << "checkpoint-swap-center quiescence/symmetry gate refused: "
          << "speed=" << measured.max_speed_cm_s << "/"
          << limits.max_speed_cm_s << " cm/s, Mach=" << measured.max_mach
          << "/" << limits.max_mach << ", rho_rel="
          << measured.max_rho_relative << "/" << limits.max_rho_relative
          << ", T_rel=" << measured.max_temperature_relative << "/"
          << limits.max_temperature_relative << ", seam_displacement_cm="
          << measured.max_seam_displacement_cm << "/"
          << limits.seam_displacement_floor_cm << ", mirror_rel="
          << measured.max_mirror_relative << "/" << limits.mirror_tolerance
          << ", Legendre_rel=" << measured.max_legendre_relative << "/"
          << limits.legendre_tolerance;
  return message.str();
}

std::string format_seam_displacement_failure(
    const CheckpointSwapCenterGateMeasurements& measured,
    const CheckpointSwapCenterThresholds& limits) {
  std::ostringstream message;
  message << std::setprecision(17)
          << "checkpoint-swap-center seam-displacement gate refused: "
          << "max_displacement_cm=" << measured.max_seam_displacement_cm
          << " exceeds seam_displacement_floor_cm="
          << limits.seam_displacement_floor_cm
          << "; the replaced region has been reached by the flow; "
             "swap the checkpoint earlier";
  return message.str();
}

std::string format_conservation_failure(
    const CheckpointSwapCenterConservationComparison& comparison) {
  std::ostringstream message;
  message << std::setprecision(17)
          << "checkpoint-swap-center gamma_d conservation gate refused: "
          << "dM=" << comparison.delta_mass << " tol="
          << comparison.tolerance_mass << ", dE="
          << comparison.delta_energy << " tol="
          << comparison.tolerance_energy << ", dPr="
          << comparison.delta_momentum_r << " tol="
          << comparison.tolerance_momentum_r << ", dPz="
          << comparison.delta_momentum_z << " tol="
          << comparison.tolerance_momentum_z << ", momentum_floor="
          << comparison.momentum_floor;
  return message.str();
}

void validate_thresholds(const CheckpointSwapCenterThresholds& thresholds) {
  const std::array<double, 11> values = {
      thresholds.max_speed_cm_s,
      thresholds.max_mach,
      thresholds.max_rho_relative,
      thresholds.max_temperature_relative,
      thresholds.seam_displacement_floor_cm,
      thresholds.mirror_z_floor_cm,
      thresholds.mirror_v_floor_cm_s,
      thresholds.mirror_tolerance,
      thresholds.legendre_tolerance,
      thresholds.ledger_kappa,
      thresholds.conservation_v_floor_cm_s};
  if (std::any_of(values.begin(), values.end(), [](const double value) {
        return !std::isfinite(value) || value < 0.0;
      }) ||
      !(thresholds.mirror_z_floor_cm > 0.0) ||
      !(thresholds.mirror_v_floor_cm_s > 0.0) ||
      !(thresholds.seam_displacement_floor_cm > 0.0) ||
      !(thresholds.ledger_kappa > 0.0) ||
      !(thresholds.conservation_v_floor_cm_s > 0.0)) {
    throw std::runtime_error(
        "checkpoint-swap-center gate thresholds must be finite and non-negative, with seam-displacement-floor-cm, mirror floors, ledger-kappa, and conservation-v-floor-cm-s > 0");
  }
}

#if TENRYU_ENABLE_HDF5

class H5File final {
 public:
  explicit H5File(const std::filesystem::path& path,
                  const unsigned flags) {
    id_ = H5Fopen(path.string().c_str(), flags, H5P_DEFAULT);
    if (id_ < 0) {
      throw std::runtime_error(
          "checkpoint-swap-center failed to open HDF5 file: " +
          path.string());
    }
  }
  ~H5File() {
    if (id_ >= 0) {
      H5Fclose(id_);
    }
  }
  H5File(const H5File&) = delete;
  H5File& operator=(const H5File&) = delete;
  hid_t get() const { return id_; }

 private:
  hid_t id_ = -1;
};

bool h5_exists(const hid_t file, const std::string& path) {
  return io::h5_link_exists(file, path.c_str()) > 0;
}

template <typename T>
T h5_read_scalar(const hid_t file,
                 const std::string& path,
                 const hid_t type,
                 const T fallback) {
  if (!h5_exists(file, path)) {
    return fallback;
  }
  const hid_t dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  if (dataset < 0) {
    throw std::runtime_error(
        "checkpoint-swap-center failed to open dataset: " + path);
  }
  T value{};
  const herr_t status =
      H5Dread(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
  H5Dclose(dataset);
  if (status < 0) {
    throw std::runtime_error(
        "checkpoint-swap-center failed to read dataset: " + path);
  }
  return value;
}

std::int64_t h5_read_i64_attribute(const hid_t object,
                                   const std::string& name,
                                   const std::int64_t fallback) {
  if (H5Aexists(object, name.c_str()) <= 0) {
    return fallback;
  }
  const hid_t attribute = H5Aopen(object, name.c_str(), H5P_DEFAULT);
  if (attribute < 0) {
    throw std::runtime_error(
        "checkpoint-swap-center failed to open attribute: " + name);
  }
  std::int64_t value = fallback;
  const herr_t status = H5Aread(attribute, H5T_NATIVE_INT64, &value);
  H5Aclose(attribute);
  if (status < 0) {
    throw std::runtime_error(
        "checkpoint-swap-center failed to read attribute: " + name);
  }
  return value;
}

std::string h5_read_string(const hid_t file, const std::string& path) {
  if (!h5_exists(file, path)) {
    throw std::runtime_error(
        "checkpoint-swap-center missing dataset: " + path);
  }
  const hid_t dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  const hid_t type = dataset >= 0 ? H5Dget_type(dataset) : -1;
  if (dataset < 0 || type < 0) {
    if (type >= 0) {
      H5Tclose(type);
    }
    if (dataset >= 0) {
      H5Dclose(dataset);
    }
    throw std::runtime_error(
        "checkpoint-swap-center failed to open string dataset: " + path);
  }
  std::string value;
  herr_t status = -1;
  if (H5Tis_variable_str(type) > 0) {
    char* raw = nullptr;
    status = H5Dread(dataset,
                     type,
                     H5S_ALL,
                     H5S_ALL,
                     H5P_DEFAULT,
                     &raw);
    if (status >= 0 && raw != nullptr) {
      value = raw;
      H5free_memory(raw);
    }
  } else {
    const std::size_t size = H5Tget_size(type);
    value.assign(size, '\0');
    status = H5Dread(dataset,
                     type,
                     H5S_ALL,
                     H5S_ALL,
                     H5P_DEFAULT,
                     value.data());
    while (!value.empty() && value.back() == '\0') {
      value.pop_back();
    }
  }
  H5Tclose(type);
  H5Dclose(dataset);
  if (status < 0) {
    throw std::runtime_error(
        "checkpoint-swap-center failed to read string dataset: " + path);
  }
  return value;
}

struct CheckpointPreflight {
  std::filesystem::path resolved_path;
  std::string frozen_config;
  int axis_core_released_units = 0;
  std::int64_t n_particles = 0;
  int burn_mc_live = 0;
  double burn_inflight = 0.0;
  bool has_carrier_group = false;
};

std::filesystem::path resolve_checkpoint_path(
    const std::string& checkpoint) {
  const std::filesystem::path path(checkpoint);
  if (path.extension() == ".h5" && std::filesystem::exists(path)) {
    return path;
  }
  if (path.extension().empty()) {
    const std::filesystem::path direct = checkpoint + ".h5";
    if (std::filesystem::exists(direct)) {
      return direct;
    }
  }
  const std::filesystem::path ranked = checkpoint + "_r0000.h5";
  if (std::filesystem::exists(ranked)) {
    return ranked;
  }
  throw std::runtime_error(
      "checkpoint-swap-center checkpoint file not found: " + checkpoint);
}

CheckpointPreflight read_checkpoint_preflight(
    const std::string& checkpoint) {
  CheckpointPreflight out;
  out.resolved_path = resolve_checkpoint_path(checkpoint);
  H5File file(out.resolved_path, H5F_ACC_RDONLY);
  out.frozen_config = h5_read_string(file.get(), "metadata/frozen_config");
  out.axis_core_released_units = h5_read_scalar<std::int32_t>(
      file.get(),
      "metadata/axis_core_released_units",
      H5T_NATIVE_INT32,
      0);
  out.burn_mc_live = h5_read_scalar<std::int32_t>(
      file.get(), "time_state/burn_mc_live", H5T_NATIVE_INT32, 0);
  out.burn_inflight = h5_read_scalar<double>(
      file.get(), "time_state/E_burn_inflight", H5T_NATIVE_DOUBLE, 0.0);
  out.has_carrier_group = h5_exists(file.get(), "mesh/carrier/v1");
  if (h5_exists(file.get(), "particles")) {
    const hid_t particles = H5Gopen2(file.get(), "particles", H5P_DEFAULT);
    if (particles < 0) {
      throw std::runtime_error(
          "checkpoint-swap-center failed to open particles group");
    }
    out.n_particles =
        h5_read_i64_attribute(particles, "n_particles", 0);
    H5Gclose(particles);
  }
  return out;
}

hid_t ensure_h5_group(const hid_t file, const std::string& path) {
  hid_t current = file;
  hid_t owned = -1;
  std::size_t begin = 0;
  while (begin < path.size()) {
    const std::size_t slash = path.find('/', begin);
    const std::string component =
        path.substr(begin,
                    slash == std::string::npos ? std::string::npos
                                               : slash - begin);
    if (!component.empty()) {
      hid_t next = -1;
      if (io::h5_link_exists(current, component.c_str()) > 0) {
        next = H5Gopen2(current, component.c_str(), H5P_DEFAULT);
      } else {
        next = H5Gcreate2(current,
                          component.c_str(),
                          H5P_DEFAULT,
                          H5P_DEFAULT,
                          H5P_DEFAULT);
      }
      if (next < 0) {
        if (owned >= 0) {
          H5Gclose(owned);
        }
        throw std::runtime_error(
            "checkpoint-swap-center failed to create audit group: " + path);
      }
      if (owned >= 0) {
        H5Gclose(owned);
      }
      owned = next;
      current = next;
    }
    if (slash == std::string::npos) {
      break;
    }
    begin = slash + 1U;
  }
  return owned;
}

template <typename T>
void h5_write_scalar(const hid_t group,
                     const std::string& name,
                     const hid_t type,
                     const T value) {
  const hid_t space = H5Screate(H5S_SCALAR);
  const hid_t dataset =
      space >= 0
          ? H5Dcreate2(group,
                       name.c_str(),
                       type,
                       space,
                       H5P_DEFAULT,
                       H5P_DEFAULT,
                       H5P_DEFAULT)
          : -1;
  const herr_t status =
      dataset >= 0
          ? H5Dwrite(dataset,
                     type,
                     H5S_ALL,
                     H5S_ALL,
                     H5P_DEFAULT,
                     &value)
          : -1;
  if (dataset >= 0) {
    H5Dclose(dataset);
  }
  if (space >= 0) {
    H5Sclose(space);
  }
  if (status < 0) {
    throw std::runtime_error(
        "checkpoint-swap-center failed to write audit dataset: " + name);
  }
}

void h5_write_string(const hid_t group,
                     const std::string& name,
                     const std::string& value) {
  const hid_t space = H5Screate(H5S_SCALAR);
  const hid_t type = H5Tcopy(H5T_C_S1);
  if (space < 0 || type < 0 ||
      H5Tset_size(type, std::max<std::size_t>(1U, value.size())) < 0) {
    if (type >= 0) {
      H5Tclose(type);
    }
    if (space >= 0) {
      H5Sclose(space);
    }
    throw std::runtime_error(
        "checkpoint-swap-center failed to create audit string type");
  }
  const hid_t dataset = H5Dcreate2(group,
                                   name.c_str(),
                                   type,
                                   space,
                                   H5P_DEFAULT,
                                   H5P_DEFAULT,
                                   H5P_DEFAULT);
  const char* data = value.empty() ? "" : value.data();
  const herr_t status =
      dataset >= 0
          ? H5Dwrite(dataset,
                     type,
                     H5S_ALL,
                     H5S_ALL,
                     H5P_DEFAULT,
                     data)
          : -1;
  if (dataset >= 0) {
    H5Dclose(dataset);
  }
  H5Tclose(type);
  H5Sclose(space);
  if (status < 0) {
    throw std::runtime_error(
        "checkpoint-swap-center failed to write audit string: " + name);
  }
}

std::string transition_value(const double value) {
  std::array<char, 64> buffer{};
  std::snprintf(buffer.data(), buffer.size(), "%.17g", value);
  return buffer.data();
}

std::string transition_value(const int value) {
  std::array<char, 32> buffer{};
  std::snprintf(buffer.data(), buffer.size(), "%d", value);
  return buffer.data();
}

std::string transition_value(const long long value) {
  return std::to_string(value);
}

std::string transition_value(const bool value) {
  return value ? "true" : "false";
}

std::string transition_value(const std::string& value) {
  return value;
}

std::string canonical_pressure_drive_modes(
    const core::Config::NumericsConfig::HydroConfig::
        PressureDrivePerturbationConfig& config) {
  std::string result;
  for (std::size_t k = 0; k < config.mode_l.size(); ++k) {
    if (!result.empty()) {
      result += ',';
    }
    result += std::to_string(config.mode_l[k]);
    result += ':';
    result += transition_value(config.mode_a[k]);
  }
  return result;
}

std::string canonical_pressure_drive_spots(
    const core::Config::NumericsConfig::HydroConfig::
        PressureDrivePerturbationConfig& config) {
  std::string result;
  for (std::size_t m = 0; m < config.spot_theta0.size(); ++m) {
    if (!result.empty()) {
      result += ',';
    }
    result += transition_value(config.spot_theta0[m]);
    result += ':';
    result += transition_value(config.spot_sigma[m]);
    result += ':';
    result += transition_value(config.spot_amp[m]);
  }
  return result;
}

std::string transition_value(const core::TopologyScheme value) {
  switch (value) {
    case core::TopologyScheme::MULTIBLOCK_POLAR_TIER:
      return "multiblock_polar_tier";
    case core::TopologyScheme::MULTIBLOCK_POLAR_TIER_CART_CENTER:
      return "multiblock_polar_tier_cart_center";
    default:
      throw std::runtime_error(
          "checkpoint-swap-center audit received an unsupported topology transition");
  }
}

template <typename T>
void record_transition(std::map<std::string, std::string>* transitions,
                       const std::string& key,
                       const T& parent,
                       const T& segment2) {
  if (parent != segment2) {
    transitions->emplace(
        key,
        transition_value(parent) + " -> " + transition_value(segment2));
  }
}

std::map<std::string, std::string> collect_config_transitions(
    const core::Config& parent,
    const core::Config& segment2) {
  std::map<std::string, std::string> transitions;
  record_transition(&transitions,
                    "mesh.topology_scheme",
                    parent.mesh.topology_scheme,
                    segment2.mesh.topology_scheme);
  record_transition(&transitions,
                    "mesh.polar_tier_cart_cut_ring",
                    parent.mesh.polar_tier_cart_cut_ring,
                    segment2.mesh.polar_tier_cart_cut_ring);
  record_transition(&transitions,
                    "mesh.polar_tier_center_kind",
                    parent.mesh.polar_tier_center_kind,
                    segment2.mesh.polar_tier_center_kind);
  record_transition(&transitions,
                    "mesh.multiblock_cart_core_r_c",
                    parent.mesh.multiblock_cart_core_r_c,
                    segment2.mesh.multiblock_cart_core_r_c);
  record_transition(&transitions,
                    "mesh.multiblock_cart_core_n_c",
                    parent.mesh.multiblock_cart_core_n_c,
                    segment2.mesh.multiblock_cart_core_n_c);
  record_transition(&transitions,
                    "mesh.multiblock_cart_core_bridge_layers",
                    parent.mesh.multiblock_cart_core_bridge_layers,
                    segment2.mesh.multiblock_cart_core_bridge_layers);
  record_transition(&transitions,
                    "mesh.multiblock_cart_core_bridge_grading",
                    parent.mesh.multiblock_cart_core_bridge_grading,
                    segment2.mesh.multiblock_cart_core_bridge_grading);
  record_transition(&transitions,
                    "mesh.multiblock_cart_core_bridge_spacing_floor",
                    parent.mesh.multiblock_cart_core_bridge_spacing_floor,
                    segment2.mesh.multiblock_cart_core_bridge_spacing_floor);
  record_transition(&transitions,
                    "mesh.multiblock_cart_core_bridge_ratio_max",
                    parent.mesh.multiblock_cart_core_bridge_ratio_max,
                    segment2.mesh.multiblock_cart_core_bridge_ratio_max);

  record_transition(&transitions,
                    "numerics.z_reflection.mode",
                    parent.numerics.z_reflection.mode,
                    segment2.numerics.z_reflection.mode);

  const auto& parent_ale = parent.numerics.ale;
  const auto& segment2_ale = segment2.numerics.ale;
  record_transition(
      &transitions,
      "numerics.ale.band_ale.respace_move_cap_frac",
      parent_ale.band_ale.respace_move_cap_frac,
      segment2_ale.band_ale.respace_move_cap_frac);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.estimator_band_hold_mach",
      parent_ale.band_ale.estimator_band_hold_mach,
      segment2_ale.band_ale.estimator_band_hold_mach);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.estimator_band_per_column",
      parent_ale.band_ale.estimator_band_per_column,
      segment2_ale.band_ale.estimator_band_per_column);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.estimator_band_pc_filter_halfwidth",
      parent_ale.band_ale.estimator_band_pc_filter_halfwidth,
      segment2_ale.band_ale.estimator_band_pc_filter_halfwidth);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.estimator_band_pc_slope_limit",
      parent_ale.band_ale.estimator_band_pc_slope_limit,
      segment2_ale.band_ale.estimator_band_pc_slope_limit);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.estimator_band_pc_slope_reject",
      parent_ale.band_ale.estimator_band_pc_slope_reject,
      segment2_ale.band_ale.estimator_band_pc_slope_reject);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.estimator_band_pc_curvature_limit",
      parent_ale.band_ale.estimator_band_pc_curvature_limit,
      segment2_ale.band_ale.estimator_band_pc_curvature_limit);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.estimator_band_pc_chi_max",
      parent_ale.band_ale.estimator_band_pc_chi_max,
      segment2_ale.band_ale.estimator_band_pc_chi_max);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.estimator_band_pc_chi_step",
      parent_ale.band_ale.estimator_band_pc_chi_step,
      segment2_ale.band_ale.estimator_band_pc_chi_step);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.estimator_band_pc_sigma_floor",
      parent_ale.band_ale.estimator_band_pc_sigma_floor,
      segment2_ale.band_ale.estimator_band_pc_sigma_floor);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.estimator_band_pc_coverage_full",
      parent_ale.band_ale.estimator_band_pc_coverage_full,
      segment2_ale.band_ale.estimator_band_pc_coverage_full);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.estimator_band_pc_coverage_min",
      parent_ale.band_ale.estimator_band_pc_coverage_min,
      segment2_ale.band_ale.estimator_band_pc_coverage_min);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.estimator_band_pc_cooldown_events",
      parent_ale.band_ale.estimator_band_pc_cooldown_events,
      segment2_ale.band_ale.estimator_band_pc_cooldown_events);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.estimator_band_pc_phase_b",
      parent_ale.band_ale.estimator_band_pc_phase_b,
      segment2_ale.band_ale.estimator_band_pc_phase_b);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.closure_catchment_enabled",
      parent_ale.band_ale.closure_catchment_enabled,
      segment2_ale.band_ale.closure_catchment_enabled);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.closure_catchment_forced_active",
      parent_ale.band_ale.closure_catchment_forced_active,
      segment2_ale.band_ale.closure_catchment_forced_active);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.closure_catchment_s_catch_cm",
      parent_ale.band_ale.closure_catchment_s_catch_cm,
      segment2_ale.band_ale.closure_catchment_s_catch_cm);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.closure_catchment_s_protect_cm",
      parent_ale.band_ale.closure_catchment_s_protect_cm,
      segment2_ale.band_ale.closure_catchment_s_protect_cm);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.closure_catchment_spacing_floor_cm",
      parent_ale.band_ale.closure_catchment_spacing_floor_cm,
      segment2_ale.band_ale.closure_catchment_spacing_floor_cm);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.closure_catchment_ratio_max",
      parent_ale.band_ale.closure_catchment_ratio_max,
      segment2_ale.band_ale.closure_catchment_ratio_max);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.closure_catchment_nu_max",
      parent_ale.band_ale.closure_catchment_nu_max,
      segment2_ale.band_ale.closure_catchment_nu_max);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.closure_catchment_max_bites",
      parent_ale.band_ale.closure_catchment_max_bites,
      segment2_ale.band_ale.closure_catchment_max_bites);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.closure_catchment_shock_hold",
      parent_ale.band_ale.closure_catchment_shock_hold,
      segment2_ale.band_ale.closure_catchment_shock_hold);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.closure_catchment_eta_h_arm",
      parent_ale.band_ale.closure_catchment_eta_h_arm,
      segment2_ale.band_ale.closure_catchment_eta_h_arm);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.closure_catchment_eta_h_full",
      parent_ale.band_ale.closure_catchment_eta_h_full,
      segment2_ale.band_ale.closure_catchment_eta_h_full);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.closure_catchment_eta_m_arm",
      parent_ale.band_ale.closure_catchment_eta_m_arm,
      segment2_ale.band_ale.closure_catchment_eta_m_arm);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.closure_catchment_eta_m_full",
      parent_ale.band_ale.closure_catchment_eta_m_full,
      segment2_ale.band_ale.closure_catchment_eta_m_full);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.closure_catchment_reset_eta",
      parent_ale.band_ale.closure_catchment_reset_eta,
      segment2_ale.band_ale.closure_catchment_reset_eta);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.closure_catchment_support_core_rows",
      parent_ale.band_ale.closure_catchment_support_core_rows,
      segment2_ale.band_ale.closure_catchment_support_core_rows);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.closure_catchment_support_taper_rows",
      parent_ale.band_ale.closure_catchment_support_taper_rows,
      segment2_ale.band_ale.closure_catchment_support_taper_rows);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.closure_catchment_accum_frac",
      parent_ale.band_ale.closure_catchment_accum_frac,
      segment2_ale.band_ale.closure_catchment_accum_frac);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.closure_catchment_rearm_drop",
      parent_ale.band_ale.closure_catchment_rearm_drop,
      segment2_ale.band_ale.closure_catchment_rearm_drop);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_enabled",
      parent_ale.band_ale.pole_theta_enabled,
      segment2_ale.band_ale.pole_theta_enabled);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_routine_enabled",
      parent_ale.band_ale.pole_theta_routine_enabled,
      segment2_ale.band_ale.pole_theta_routine_enabled);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_phys_lp",
      parent_ale.band_ale.pole_theta_phys_lp,
      segment2_ale.band_ale.pole_theta_phys_lp);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_phys_lc",
      parent_ale.band_ale.pole_theta_phys_lc,
      segment2_ale.band_ale.pole_theta_phys_lc);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_noise_floor",
      parent_ale.band_ale.pole_theta_noise_floor,
      segment2_ale.band_ale.pole_theta_noise_floor);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_noise_ceiling",
      parent_ale.band_ale.pole_theta_noise_ceiling,
      segment2_ale.band_ale.pole_theta_noise_ceiling);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_h_arm",
      parent_ale.band_ale.pole_theta_h_arm,
      segment2_ale.band_ale.pole_theta_h_arm);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_h_fire",
      parent_ale.band_ale.pole_theta_h_fire,
      segment2_ale.band_ale.pole_theta_h_fire);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_h_hard",
      parent_ale.band_ale.pole_theta_h_hard,
      segment2_ale.band_ale.pole_theta_h_hard);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_h_release",
      parent_ale.band_ale.pole_theta_h_release,
      segment2_ale.band_ale.pole_theta_h_release);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_kappa_arm",
      parent_ale.band_ale.pole_theta_kappa_arm,
      segment2_ale.band_ale.pole_theta_kappa_arm);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_kappa_fire",
      parent_ale.band_ale.pole_theta_kappa_fire,
      segment2_ale.band_ale.pole_theta_kappa_fire);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_kappa_hard",
      parent_ale.band_ale.pole_theta_kappa_hard,
      segment2_ale.band_ale.pole_theta_kappa_hard);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_alpha",
      parent_ale.band_ale.pole_theta_alpha,
      segment2_ale.band_ale.pole_theta_alpha);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_alpha_hard",
      parent_ale.band_ale.pole_theta_alpha_hard,
      segment2_ale.band_ale.pole_theta_alpha_hard);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_deadband_frac",
      parent_ale.band_ale.pole_theta_deadband_frac,
      segment2_ale.band_ale.pole_theta_deadband_frac);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_move_limit_frac",
      parent_ale.band_ale.pole_theta_move_limit_frac,
      segment2_ale.band_ale.pole_theta_move_limit_frac);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_move_limit_hard_frac",
      parent_ale.band_ale.pole_theta_move_limit_hard_frac,
      segment2_ale.band_ale.pole_theta_move_limit_hard_frac);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_cooldown_s",
      parent_ale.band_ale.pole_theta_cooldown_s,
      segment2_ale.band_ale.pole_theta_cooldown_s);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_cooldown_base_s",
      parent_ale.band_ale.pole_theta_cooldown_base_s,
      segment2_ale.band_ale.pole_theta_cooldown_base_s);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_predict_window_s",
      parent_ale.band_ale.pole_theta_predict_window_s,
      segment2_ale.band_ale.pole_theta_predict_window_s);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_predict_horizon_s",
      parent_ale.band_ale.pole_theta_predict_horizon_s,
      segment2_ale.band_ale.pole_theta_predict_horizon_s);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_halo_columns",
      parent_ale.band_ale.pole_theta_halo_columns,
      segment2_ale.band_ale.pole_theta_halo_columns);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_halo_rows",
      parent_ale.band_ale.pole_theta_halo_rows,
      segment2_ale.band_ale.pole_theta_halo_rows);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_post_h_floor",
      parent_ale.band_ale.pole_theta_post_h_floor,
      segment2_ale.band_ale.pole_theta_post_h_floor);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_curve_preserving",
      parent_ale.band_ale.pole_theta_curve_preserving,
      segment2_ale.band_ale.pole_theta_curve_preserving);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_protected_modes",
      parent_ale.band_ale.pole_theta_protected_modes,
      segment2_ale.band_ale.pole_theta_protected_modes);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_fit_order",
      parent_ale.band_ale.pole_theta_fit_order,
      segment2_ale.band_ale.pole_theta_fit_order);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.pole_theta_shock_hold",
      parent_ale.band_ale.pole_theta_shock_hold,
      segment2_ale.band_ale.pole_theta_shock_hold);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.estimator_band_pc_tube_dilate_rows",
      parent_ale.band_ale.estimator_band_pc_tube_dilate_rows,
      segment2_ale.band_ale.estimator_band_pc_tube_dilate_rows);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.estimator_band_pc_tube_dilate_cols_extra",
      parent_ale.band_ale.estimator_band_pc_tube_dilate_cols_extra,
      segment2_ale.band_ale.estimator_band_pc_tube_dilate_cols_extra);
  record_transition(
      &transitions,
      "numerics.ale.band_ale.estimator_band_pc_ambiguous_hold_fraction",
      parent_ale.band_ale.estimator_band_pc_ambiguous_hold_fraction,
      segment2_ale.band_ale.estimator_band_pc_ambiguous_hold_fraction);
  record_transition(&transitions,
                    "numerics.ale.central_pseudo_core_enabled",
                    parent_ale.central_pseudo_core_enabled,
                    segment2_ale.central_pseudo_core_enabled);
  record_transition(&transitions,
                    "numerics.ale.central_pseudo_core_s_c",
                    parent_ale.central_pseudo_core_s_c,
                    segment2_ale.central_pseudo_core_s_c);
  record_transition(
      &transitions,
      "numerics.ale.central_pseudo_core_activation_time_s",
      parent_ale.central_pseudo_core_activation_time_s,
      segment2_ale.central_pseudo_core_activation_time_s);
  record_transition(
      &transitions,
      "numerics.ale.central_pseudo_core_ring_absorption_enabled",
      parent_ale.central_pseudo_core_ring_absorption_enabled,
      segment2_ale.central_pseudo_core_ring_absorption_enabled);
  record_transition(&transitions,
                    "numerics.ale.central_pseudo_core_ring_absorption_tau",
                    parent_ale.central_pseudo_core_ring_absorption_tau,
                    segment2_ale.central_pseudo_core_ring_absorption_tau);
  record_transition(&transitions,
                    "numerics.ale.central_pseudo_core_core1d_enabled",
                    parent_ale.central_pseudo_core_core1d_enabled,
                    segment2_ale.central_pseudo_core_core1d_enabled);
  record_transition(&transitions,
                    "numerics.ale.central_pseudo_core_core1d_build_shells",
                    parent_ale.central_pseudo_core_core1d_build_shells,
                    segment2_ale.central_pseudo_core_core1d_build_shells);
  record_transition(&transitions,
                    "numerics.ale.central_pseudo_core_core1d_split_append",
                    parent_ale.central_pseudo_core_core1d_split_append,
                    segment2_ale.central_pseudo_core_core1d_split_append);
  record_transition(&transitions,
                    "numerics.ale.central_pseudo_core_core1d_av_c1",
                    parent_ale.central_pseudo_core_core1d_av_c1,
                    segment2_ale.central_pseudo_core_core1d_av_c1);
  record_transition(&transitions,
                    "numerics.ale.central_pseudo_core_core1d_av_c2",
                    parent_ale.central_pseudo_core_core1d_av_c2,
                    segment2_ale.central_pseudo_core_core1d_av_c2);
  record_transition(&transitions,
                    "numerics.ale.central_pseudo_core_core1d_cfl",
                    parent_ale.central_pseudo_core_core1d_cfl,
                    segment2_ale.central_pseudo_core_core1d_cfl);
  record_transition(&transitions,
                    "numerics.ale.central_pseudo_core_core1d_piston_cap",
                    parent_ale.central_pseudo_core_core1d_piston_cap,
                    segment2_ale.central_pseudo_core_core1d_piston_cap);
  record_transition(&transitions,
                    "numerics.ale.central_pseudo_core_core1d_max_substeps",
                    parent_ale.central_pseudo_core_core1d_max_substeps,
                    segment2_ale.central_pseudo_core_core1d_max_substeps);
  record_transition(&transitions,
                    "numerics.ale.central_pseudo_core_core1d_dist_append",
                    parent_ale.central_pseudo_core_core1d_dist_append,
                    segment2_ale.central_pseudo_core_core1d_dist_append);
  record_transition(
      &transitions,
      "numerics.ale.central_pseudo_core_spherical_absorb_gasfront",
      parent_ale.central_pseudo_core_spherical_absorb_gasfront,
      segment2_ale.central_pseudo_core_spherical_absorb_gasfront);
  record_transition(&transitions,
                    "numerics.ale.central_pseudo_core_spherical_absorb_alpha",
                    parent_ale.central_pseudo_core_spherical_absorb_alpha,
                    segment2_ale.central_pseudo_core_spherical_absorb_alpha);
  record_transition(&transitions,
                    "numerics.ale.central_pseudo_core_spherical_absorb_pjump",
                    parent_ale.central_pseudo_core_spherical_absorb_pjump,
                    segment2_ale.central_pseudo_core_spherical_absorb_pjump);
  record_transition(&transitions,
                    "numerics.ale.central_pseudo_core_mixed_absorb_enabled",
                    parent_ale.central_pseudo_core_mixed_absorb_enabled,
                    segment2_ale.central_pseudo_core_mixed_absorb_enabled);
  record_transition(&transitions,
                    "numerics.ale.central_pseudo_core_absorb_watch_rows",
                    parent_ale.central_pseudo_core_absorb_watch_rows,
                    segment2_ale.central_pseudo_core_absorb_watch_rows);
  record_transition(
      &transitions,
      "numerics.ale.central_pseudo_core_ring_absorption_max_rings",
      parent_ale.central_pseudo_core_ring_absorption_max_rings,
      segment2_ale.central_pseudo_core_ring_absorption_max_rings);
  record_transition(
      &transitions,
      "numerics.ale.central_pseudo_core_ring_absorption_gas_tracer_min",
      parent_ale.central_pseudo_core_ring_absorption_gas_tracer_min,
      segment2_ale.central_pseudo_core_ring_absorption_gas_tracer_min);
  record_transition(
      &transitions,
      "numerics.ale.central_pseudo_core_ring_absorption_gas_tracer_cell_min",
      parent_ale.central_pseudo_core_ring_absorption_gas_tracer_cell_min,
      segment2_ale.central_pseudo_core_ring_absorption_gas_tracer_cell_min);
  const auto& parent_pressure_perturbation =
      parent.numerics.hydro.pressure_drive_perturbation;
  const auto& segment2_pressure_perturbation =
      segment2.numerics.hydro.pressure_drive_perturbation;
  record_transition(
      &transitions,
      "numerics.hydro.pressure_drive_perturbation.enabled",
      parent_pressure_perturbation.enabled,
      segment2_pressure_perturbation.enabled);
  record_transition(
      &transitions,
      "numerics.hydro.pressure_drive_perturbation.legendre_modes",
      canonical_pressure_drive_modes(parent_pressure_perturbation),
      canonical_pressure_drive_modes(segment2_pressure_perturbation));
  record_transition(
      &transitions,
      "numerics.hydro.pressure_drive_perturbation.ring_spots",
      canonical_pressure_drive_spots(parent_pressure_perturbation),
      canonical_pressure_drive_spots(segment2_pressure_perturbation));
  record_transition(
      &transitions,
      "numerics.hydro.pressure_drive_perturbation.random_seed",
      parent_pressure_perturbation.random_seed,
      segment2_pressure_perturbation.random_seed);
  record_transition(
      &transitions,
      "numerics.hydro.pressure_drive_perturbation.random_l_min",
      parent_pressure_perturbation.random_l_min,
      segment2_pressure_perturbation.random_l_min);
  record_transition(
      &transitions,
      "numerics.hydro.pressure_drive_perturbation.random_l_max",
      parent_pressure_perturbation.random_l_max,
      segment2_pressure_perturbation.random_l_max);
  record_transition(
      &transitions,
      "numerics.hydro.pressure_drive_perturbation.random_rms",
      parent_pressure_perturbation.random_rms,
      segment2_pressure_perturbation.random_rms);
  record_transition(&transitions,
                    "numerics.diagnostics.refinement_estimator.enabled",
                    parent.numerics.diagnostics.refinement_estimator.enabled,
                    segment2.numerics.diagnostics.refinement_estimator.enabled);
  record_transition(&transitions,
                    "numerics.diagnostics.refinement_estimator.every",
                    parent.numerics.diagnostics.refinement_estimator.every,
                    segment2.numerics.diagnostics.refinement_estimator.every);
  record_transition(
      &transitions,
      "numerics.diagnostics.conduction_energy_rate_export.enabled",
      parent.numerics.diagnostics.conduction_energy_rate_export.enabled,
      segment2.numerics.diagnostics.conduction_energy_rate_export.enabled);
  const auto& parent_autopilot =
      parent.numerics.diagnostics.refinement_autopilot;
  const auto& segment2_autopilot =
      segment2.numerics.diagnostics.refinement_autopilot;
  record_transition(&transitions,
                    "numerics.diagnostics.refinement_autopilot.enabled",
                    parent_autopilot.enabled,
                    segment2_autopilot.enabled);
  record_transition(&transitions,
                    "numerics.diagnostics.refinement_autopilot.mode",
                    parent_autopilot.mode,
                    segment2_autopilot.mode);
  record_transition(&transitions,
                    "numerics.diagnostics.refinement_autopilot.ckpt_lead_h",
                    parent_autopilot.ckpt_lead_h,
                    segment2_autopilot.ckpt_lead_h);
  record_transition(&transitions,
                    "numerics.diagnostics.refinement_autopilot.e_on",
                    parent_autopilot.e_on,
                    segment2_autopilot.e_on);
  record_transition(&transitions,
                    "numerics.diagnostics.refinement_autopilot.e_off",
                    parent_autopilot.e_off,
                    segment2_autopilot.e_off);
  record_transition(&transitions,
                    "numerics.diagnostics.refinement_autopilot.assoc_cut",
                    parent_autopilot.assoc_cut,
                    segment2_autopilot.assoc_cut);
  record_transition(&transitions,
                    "numerics.diagnostics.refinement_autopilot.strong_cut",
                    parent_autopilot.strong_cut,
                    segment2_autopilot.strong_cut);
  record_transition(&transitions,
                    "numerics.diagnostics.refinement_autopilot.gap_bridge",
                    parent_autopilot.gap_bridge,
                    segment2_autopilot.gap_bridge);
  record_transition(&transitions,
                    "numerics.diagnostics.refinement_autopilot.persist",
                    parent_autopilot.persist,
                    segment2_autopilot.persist);
  record_transition(&transitions,
                    "numerics.diagnostics.refinement_autopilot.n_q_plan",
                    parent_autopilot.n_q_plan,
                    segment2_autopilot.n_q_plan);
  record_transition(&transitions,
                    "numerics.diagnostics.refinement_autopilot.chi_design",
                    parent_autopilot.chi_design,
                    segment2_autopilot.chi_design);
  record_transition(&transitions,
                    "numerics.diagnostics.refinement_autopilot.s_rep_cm",
                    parent_autopilot.s_rep_cm,
                    segment2_autopilot.s_rep_cm);
  record_transition(&transitions,
                    "numerics.diagnostics.refinement_autopilot.handoff_cm",
                    parent_autopilot.handoff_cm,
                    segment2_autopilot.handoff_cm);
  record_transition(&transitions,
                    "numerics.diagnostics.refinement_autopilot.window_lo_h",
                    parent_autopilot.window_lo_h,
                    segment2_autopilot.window_lo_h);
  record_transition(&transitions,
                    "numerics.diagnostics.refinement_autopilot.window_hi_h",
                    parent_autopilot.window_hi_h,
                    segment2_autopilot.window_hi_h);
  record_transition(&transitions,
                    "numerics.diagnostics.refinement_autopilot.history",
                    parent_autopilot.history,
                    segment2_autopilot.history);
  record_transition(&transitions,
                    "numerics.diagnostics.refinement_autopilot.cov_min",
                    parent_autopilot.cov_min,
                    segment2_autopilot.cov_min);

  record_transition(&transitions,
                    "main.t_end",
                    parent.main.t_end,
                    segment2.main.t_end);
  record_transition(&transitions,
                    "output.plot_every",
                    parent.output.plot_every,
                    segment2.output.plot_every);
  record_transition(&transitions,
                    "output.history_every",
                    parent.output.history_every,
                    segment2.output.history_every);
  record_transition(&transitions,
                    "output.checkpoint_every",
                    parent.output.checkpoint_every,
                    segment2.output.checkpoint_every);
  record_transition(&transitions,
                    "output.plot_every_s",
                    parent.output.plot_every_s,
                    segment2.output.plot_every_s);
  record_transition(&transitions,
                    "output.history_every_s",
                    parent.output.history_every_s,
                    segment2.output.history_every_s);
  record_transition(&transitions,
                    "output.checkpoint_every_s",
                    parent.output.checkpoint_every_s,
                    segment2.output.checkpoint_every_s);
  record_transition(&transitions,
                    "numerics.diagnostics_every",
                    parent.numerics.diagnostics_every,
                    segment2.numerics.diagnostics_every);
  return transitions;
}

struct AuditRecord {
  std::string input_file_hash;
  std::string segment2_config_hash;
  std::string center_kind;
  int cut_ring = -1;
  std::int64_t kept_cells = 0;
  std::int64_t kept_nodes = 0;
  double r_cut_cm = 0.0;
  int n_c = 0;
  int n_b = 0;
  CheckpointSwapCenterConservationComparison conservation;
  CheckpointSwapCenterThresholds thresholds;
  CheckpointSwapCenterGateMeasurements measured;
  double fresh_v_prenorm_max_cm_s = 0.0;
  int axis_core_released_units_a = 0;
  std::map<std::string, std::string> transitions;
};

void append_audit_group(const std::filesystem::path& checkpoint,
                        const AuditRecord& audit) {
  H5File file(checkpoint, H5F_ACC_RDWR);
  if (h5_exists(file.get(), std::string(kAuditGroup))) {
    throw std::runtime_error(
        "checkpoint-swap-center staging checkpoint already contains epoch-swap audit group");
  }
  const hid_t group = ensure_h5_group(file.get(), std::string(kAuditGroup));
  if (group < 0) {
    throw std::runtime_error(
        "checkpoint-swap-center failed to create epoch-swap audit group");
  }
  h5_write_string(group, "checkpoint_a_sha256", audit.input_file_hash);
  h5_write_string(group,
                  "segment2_frozen_config_sha256",
                  audit.segment2_config_hash);
  h5_write_string(group,
                  "tool_version",
                  std::string(kToolVersion) + " tenryu/" +
                      core::tenryu_version_string());
  h5_write_string(group, "center_kind", audit.center_kind);
  h5_write_scalar(group, "cut_ring", H5T_NATIVE_INT32, audit.cut_ring);
  h5_write_scalar(group,
                  "kept_cells_C",
                  H5T_NATIVE_INT64,
                  audit.kept_cells);
  h5_write_scalar(group,
                  "kept_nodes_N",
                  H5T_NATIVE_INT64,
                  audit.kept_nodes);
  h5_write_scalar(group, "r_cut_cm", H5T_NATIVE_DOUBLE, audit.r_cut_cm);
  h5_write_scalar(group, "n_c", H5T_NATIVE_INT32, audit.n_c);
  h5_write_scalar(group, "n_b", H5T_NATIVE_INT32, audit.n_b);
  h5_write_scalar(group,
                  "delta_mass_g",
                  H5T_NATIVE_DOUBLE,
                  audit.conservation.delta_mass);
  h5_write_scalar(group,
                  "delta_energy_erg",
                  H5T_NATIVE_DOUBLE,
                  audit.conservation.delta_energy);
  h5_write_scalar(group,
                  "delta_momentum_r_g_cm_s",
                  H5T_NATIVE_DOUBLE,
                  audit.conservation.delta_momentum_r);
  h5_write_scalar(group,
                  "delta_momentum_z_g_cm_s",
                  H5T_NATIVE_DOUBLE,
                  audit.conservation.delta_momentum_z);
  h5_write_scalar(group,
                  "mass_tolerance_g",
                  H5T_NATIVE_DOUBLE,
                  audit.conservation.tolerance_mass);
  h5_write_scalar(group,
                  "energy_tolerance_erg",
                  H5T_NATIVE_DOUBLE,
                  audit.conservation.tolerance_energy);
  h5_write_scalar(group,
                  "momentum_r_tolerance_g_cm_s",
                  H5T_NATIVE_DOUBLE,
                  audit.conservation.tolerance_momentum_r);
  h5_write_scalar(group,
                  "momentum_z_tolerance_g_cm_s",
                  H5T_NATIVE_DOUBLE,
                  audit.conservation.tolerance_momentum_z);
  h5_write_scalar(group,
                  "momentum_floor_g_cm_s",
                  H5T_NATIVE_DOUBLE,
                  audit.conservation.momentum_floor);
  h5_write_scalar(group,
                  "max_speed_cm_s_threshold",
                  H5T_NATIVE_DOUBLE,
                  audit.thresholds.max_speed_cm_s);
  h5_write_scalar(group,
                  "max_mach_threshold",
                  H5T_NATIVE_DOUBLE,
                  audit.thresholds.max_mach);
  h5_write_scalar(group,
                  "max_rho_relative_threshold",
                  H5T_NATIVE_DOUBLE,
                  audit.thresholds.max_rho_relative);
  h5_write_scalar(group,
                  "max_temperature_relative_threshold",
                  H5T_NATIVE_DOUBLE,
                  audit.thresholds.max_temperature_relative);
  h5_write_scalar(group,
                  "mirror_tolerance",
                  H5T_NATIVE_DOUBLE,
                  audit.thresholds.mirror_tolerance);
  h5_write_scalar(group,
                  "legendre_tolerance",
                  H5T_NATIVE_DOUBLE,
                  audit.thresholds.legendre_tolerance);
  h5_write_scalar(group,
                  "ledger_kappa",
                  H5T_NATIVE_DOUBLE,
                  audit.thresholds.ledger_kappa);
  h5_write_scalar(group,
                  "measured_max_speed_cm_s",
                  H5T_NATIVE_DOUBLE,
                  audit.measured.max_speed_cm_s);
  h5_write_scalar(group,
                  "measured_max_mach",
                  H5T_NATIVE_DOUBLE,
                  audit.measured.max_mach);
  h5_write_scalar(group,
                  "measured_max_rho_relative",
                  H5T_NATIVE_DOUBLE,
                  audit.measured.max_rho_relative);
  h5_write_scalar(group,
                  "measured_max_temperature_relative",
                  H5T_NATIVE_DOUBLE,
                  audit.measured.max_temperature_relative);
  h5_write_scalar(group,
                  "measured_max_mirror_relative",
                  H5T_NATIVE_DOUBLE,
                  audit.measured.max_mirror_relative);
  h5_write_scalar(group,
                  "measured_max_legendre_relative",
                  H5T_NATIVE_DOUBLE,
                  audit.measured.max_legendre_relative);
  h5_write_scalar(group,
                  "fresh_v_prenorm_max_cm_s",
                  H5T_NATIVE_DOUBLE,
                  audit.fresh_v_prenorm_max_cm_s);
  h5_write_scalar(group,
                  "axis_core_released_units_a",
                  H5T_NATIVE_INT32,
                  audit.axis_core_released_units_a);
  if (!audit.transitions.empty()) {
    const hid_t transition_group = ensure_h5_group(group, "transition");
    for (const auto& [key, value] : audit.transitions) {
      h5_write_string(transition_group, key, value);
    }
    H5Gclose(transition_group);
  }
  H5Gclose(group);
}

bool checkpoint_has_audit_group(const std::filesystem::path& checkpoint) {
  H5File file(checkpoint, H5F_ACC_RDONLY);
  return h5_exists(file.get(), std::string(kAuditGroup));
}

#endif  // TENRYU_ENABLE_HDF5

#if TENRYU_ENABLE_PYTHON

namespace py = pybind11;

py::dict parse_frozen_config(const std::string& frozen_config) {
  py::object parsed;
  try {
    parsed = py::module_::import("json").attr("loads")(frozen_config);
  } catch (const py::error_already_set& error) {
    throw std::runtime_error(
        "checkpoint-swap-center requires canonical JSON metadata/frozen_config: " +
        std::string(error.what()));
  }
  if (!py::isinstance<py::dict>(parsed)) {
    throw std::runtime_error(
        "checkpoint-swap-center requires object-valued metadata/frozen_config");
  }
  return py::reinterpret_borrow<py::dict>(parsed);
}

template <typename T>
void inherit_frozen_value_if_present(const py::dict& root,
                                     const char* section,
                                     const char* key,
                                     T* value) {
  const py::str section_key(section);
  if (!root.contains(section_key)) {
    return;
  }
  const py::object child = root[section_key];
  if (!py::isinstance<py::dict>(child)) {
    throw std::runtime_error(
        std::string("checkpoint-swap-center frozen_config.") + section +
        " must be an object");
  }
  const py::dict values = py::reinterpret_borrow<py::dict>(child);
  const py::str value_key(key);
  if (values.contains(value_key)) {
    *value = py::cast<T>(values[value_key]);
  }
}

template <typename T>
void inherit_frozen_path_value_if_present(
    const py::dict& root,
    const std::vector<const char*>& sections,
    const char* key,
    T* value) {
  py::dict current = root;
  for (const char* section : sections) {
    const py::str section_key(section);
    if (!current.contains(section_key)) {
      return;
    }
    const py::object child = current[section_key];
    if (!py::isinstance<py::dict>(child)) {
      throw std::runtime_error(
          std::string("checkpoint-swap-center frozen_config.") + section +
          " must be an object");
    }
    current = py::reinterpret_borrow<py::dict>(child);
  }
  const py::str value_key(key);
  if (current.contains(value_key)) {
    *value = py::cast<T>(current[value_key]);
  }
}

void inherit_frozen_pressure_drive_composites_if_present(
    const py::dict& root,
    core::Config::NumericsConfig::HydroConfig::
        PressureDrivePerturbationConfig* config) {
  py::dict current = root;
  for (const char* section :
       std::vector<const char*>{"numerics",
                                "hydro",
                                "pressure_drive_perturbation"}) {
    const py::str section_key(section);
    if (!current.contains(section_key)) {
      return;
    }
    const py::object child = current[section_key];
    if (!py::isinstance<py::dict>(child)) {
      throw std::runtime_error(
          std::string("checkpoint-swap-center frozen_config.") + section +
          " must be an object");
    }
    current = py::reinterpret_borrow<py::dict>(child);
  }

  const py::str modes_key("legendre_modes");
  if (current.contains(modes_key)) {
    const py::object modes_obj = current[modes_key];
    if (!py::isinstance<py::sequence>(modes_obj) ||
        py::isinstance<py::str>(modes_obj)) {
      throw std::runtime_error(
          "checkpoint-swap-center frozen_config legendre_modes must be a "
          "list");
    }
    config->mode_l.clear();
    config->mode_a.clear();
    const py::sequence modes =
        py::reinterpret_borrow<py::sequence>(modes_obj);
    for (std::size_t k = 0; k < modes.size(); ++k) {
      const py::object mode_obj = modes[k];
      if (!py::isinstance<py::sequence>(mode_obj) ||
          py::isinstance<py::str>(mode_obj)) {
        throw std::runtime_error(
            "checkpoint-swap-center frozen_config legendre_modes entry must "
            "be a pair");
      }
      const py::sequence mode =
          py::reinterpret_borrow<py::sequence>(mode_obj);
      if (mode.size() != 2) {
        throw std::runtime_error(
            "checkpoint-swap-center frozen_config legendre_modes entry must "
            "contain two values");
      }
      config->mode_l.push_back(py::cast<int>(mode[0]));
      config->mode_a.push_back(py::cast<double>(mode[1]));
    }
  }

  const py::str spots_key("ring_spots");
  if (current.contains(spots_key)) {
    const py::object spots_obj = current[spots_key];
    if (!py::isinstance<py::sequence>(spots_obj) ||
        py::isinstance<py::str>(spots_obj)) {
      throw std::runtime_error(
          "checkpoint-swap-center frozen_config ring_spots must be a list");
    }
    config->spot_theta0.clear();
    config->spot_sigma.clear();
    config->spot_amp.clear();
    const py::sequence spots =
        py::reinterpret_borrow<py::sequence>(spots_obj);
    for (std::size_t m = 0; m < spots.size(); ++m) {
      const py::object spot_obj = spots[m];
      if (!py::isinstance<py::sequence>(spot_obj) ||
          py::isinstance<py::str>(spot_obj)) {
        throw std::runtime_error(
            "checkpoint-swap-center frozen_config ring_spots entry must be a "
            "triple");
      }
      const py::sequence spot =
          py::reinterpret_borrow<py::sequence>(spot_obj);
      if (spot.size() != 3) {
        throw std::runtime_error(
            "checkpoint-swap-center frozen_config ring_spots entry must "
            "contain three values");
      }
      config->spot_theta0.push_back(py::cast<double>(spot[0]));
      config->spot_sigma.push_back(py::cast<double>(spot[1]));
      config->spot_amp.push_back(py::cast<double>(spot[2]));
    }
  }
}

std::string sha256_bytes(const std::string& bytes) {
  const py::object digest =
      py::module_::import("hashlib").attr("sha256")(py::bytes(bytes));
  return "sha256:" +
         py::str(digest.attr("hexdigest")()).cast<std::string>();
}

std::string sha256_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "checkpoint-swap-center failed to hash input file: " +
        path.string());
  }
  py::object digest = py::module_::import("hashlib").attr("sha256")();
  std::array<char, 1U << 20U> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0) {
      digest.attr("update")(
          py::bytes(buffer.data(), static_cast<std::size_t>(count)));
    }
  }
  if (!input.eof()) {
    throw std::runtime_error(
        "checkpoint-swap-center failed while hashing input file: " +
        path.string());
  }
  return "sha256:" +
         py::str(digest.attr("hexdigest")()).cast<std::string>();
}

bool dict_enabled(const py::dict& root, const char* key) {
  const py::str py_key(key);
  if (!root.contains(py_key)) {
    return false;
  }
  const py::object child = root[py_key];
  if (!py::isinstance<py::dict>(child)) {
    throw std::runtime_error(
        std::string("checkpoint-swap-center frozen_config.") + key +
        " must be an object");
  }
  const py::dict dict = py::reinterpret_borrow<py::dict>(child);
  const py::str enabled_key("enabled");
  return dict.contains(enabled_key) && py::cast<bool>(dict[enabled_key]);
}

bool frozen_burn_mc_active(const py::dict& root) {
  const py::str burn_key("burn");
  if (!root.contains(burn_key)) {
    return false;
  }
  const py::object burn_child = root[burn_key];
  if (!py::isinstance<py::dict>(burn_child)) {
    throw std::runtime_error(
        "checkpoint-swap-center frozen_config.burn must be an object");
  }
  const py::dict burn = py::reinterpret_borrow<py::dict>(burn_child);
  const py::str enabled_key("enabled");
  const py::str scheme_key("scheme");
  const bool enabled =
      burn.contains(enabled_key) && py::cast<bool>(burn[enabled_key]);
  const std::string scheme =
      burn.contains(scheme_key) ? py::cast<std::string>(burn[scheme_key])
                                : std::string{};
  return enabled && scheme == "mc";
}

void reject_unsupported_frozen_physics(const std::string& frozen_config) {
  const py::dict root = parse_frozen_config(frozen_config);
  if (dict_enabled(root, "radiation")) {
    throw std::runtime_error(
        "checkpoint-swap-center v1 refuses checkpoint A because frozen_config Radiation.enabled is true");
  }
  if (dict_enabled(root, "laser")) {
    throw std::runtime_error(
        "checkpoint-swap-center v1 refuses checkpoint A because frozen_config Laser.enabled is true");
  }
  if (frozen_burn_mc_active(root)) {
    throw std::runtime_error(
        "checkpoint-swap-center v1 refuses checkpoint A because frozen_config burn MC is active");
  }
}

std::string frozen_namelist_source_hash(const py::dict& root) {
  const py::str key("_namelist_source_hash");
  if (!root.contains(key)) {
    throw std::runtime_error(
        "checkpoint-swap-center checkpoint A frozen_config is missing _namelist_source_hash");
  }
  const py::object source_hash = root[key];
  if (!py::isinstance<py::str>(source_hash)) {
    throw std::runtime_error(
        "checkpoint-swap-center checkpoint A frozen_config _namelist_source_hash must be a string");
  }
  return py::cast<std::string>(source_hash);
}

core::Config make_parent_config_impl(
    const core::Config& segment2,
    const std::string& parent_frozen_config) {
  core::Config parent = segment2;
  const py::dict frozen = parse_frozen_config(parent_frozen_config);
  const core::Config::MeshConfig mesh_defaults;
  const core::Config::NumericsConfig::AleConfig ale_defaults;
  parent.mesh.topology_scheme =
      core::TopologyScheme::MULTIBLOCK_POLAR_TIER;
  parent.mesh.polar_tier_cart_cut_ring =
      mesh_defaults.polar_tier_cart_cut_ring;
  parent.mesh.polar_tier_center_kind =
      mesh_defaults.polar_tier_center_kind;
  parent.mesh.multiblock_cart_core_r_c =
      mesh_defaults.multiblock_cart_core_r_c;
  parent.mesh.multiblock_cart_core_n_c =
      mesh_defaults.multiblock_cart_core_n_c;
  parent.mesh.multiblock_cart_core_bridge_layers =
      mesh_defaults.multiblock_cart_core_bridge_layers;
  parent.mesh.multiblock_cart_core_bridge_grading =
      mesh_defaults.multiblock_cart_core_bridge_grading;
  parent.mesh.multiblock_cart_core_bridge_spacing_floor =
      mesh_defaults.multiblock_cart_core_bridge_spacing_floor;
  parent.mesh.multiblock_cart_core_bridge_ratio_max =
      mesh_defaults.multiblock_cart_core_bridge_ratio_max;

  auto& ale = parent.numerics.ale;
  ale.central_pseudo_core_enabled =
      ale_defaults.central_pseudo_core_enabled;
  ale.central_pseudo_core_s_c = ale_defaults.central_pseudo_core_s_c;
  ale.central_pseudo_core_activation_time_s =
      ale_defaults.central_pseudo_core_activation_time_s;
  ale.central_pseudo_core_ring_absorption_enabled =
      ale_defaults.central_pseudo_core_ring_absorption_enabled;
  ale.central_pseudo_core_ring_absorption_tau =
      ale_defaults.central_pseudo_core_ring_absorption_tau;
  ale.central_pseudo_core_core1d_enabled =
      ale_defaults.central_pseudo_core_core1d_enabled;
  ale.central_pseudo_core_core1d_build_shells =
      ale_defaults.central_pseudo_core_core1d_build_shells;
  ale.central_pseudo_core_core1d_split_append =
      ale_defaults.central_pseudo_core_core1d_split_append;
  ale.central_pseudo_core_core1d_av_c1 =
      ale_defaults.central_pseudo_core_core1d_av_c1;
  ale.central_pseudo_core_core1d_av_c2 =
      ale_defaults.central_pseudo_core_core1d_av_c2;
  ale.central_pseudo_core_core1d_cfl =
      ale_defaults.central_pseudo_core_core1d_cfl;
  ale.central_pseudo_core_core1d_piston_cap =
      ale_defaults.central_pseudo_core_core1d_piston_cap;
  ale.central_pseudo_core_core1d_max_substeps =
      ale_defaults.central_pseudo_core_core1d_max_substeps;
  ale.central_pseudo_core_core1d_dist_append =
      ale_defaults.central_pseudo_core_core1d_dist_append;
  ale.central_pseudo_core_spherical_absorb_gasfront =
      ale_defaults.central_pseudo_core_spherical_absorb_gasfront;
  ale.central_pseudo_core_spherical_absorb_alpha =
      ale_defaults.central_pseudo_core_spherical_absorb_alpha;
  ale.central_pseudo_core_spherical_absorb_pjump =
      ale_defaults.central_pseudo_core_spherical_absorb_pjump;
  ale.central_pseudo_core_mixed_absorb_enabled =
      ale_defaults.central_pseudo_core_mixed_absorb_enabled;
  ale.central_pseudo_core_absorb_watch_rows =
      ale_defaults.central_pseudo_core_absorb_watch_rows;
  ale.central_pseudo_core_ring_absorption_max_rings =
      ale_defaults.central_pseudo_core_ring_absorption_max_rings;
  ale.central_pseudo_core_ring_absorption_gas_tracer_min =
      ale_defaults.central_pseudo_core_ring_absorption_gas_tracer_min;
  ale.central_pseudo_core_ring_absorption_gas_tracer_cell_min =
      ale_defaults.central_pseudo_core_ring_absorption_gas_tracer_cell_min;
  ale.band_ale.respace_move_cap_frac =
      ale_defaults.band_ale.respace_move_cap_frac;
  ale.band_ale.estimator_band_hold_mach =
      ale_defaults.band_ale.estimator_band_hold_mach;
  ale.band_ale.estimator_band_per_column =
      ale_defaults.band_ale.estimator_band_per_column;
  ale.band_ale.estimator_band_pc_filter_halfwidth =
      ale_defaults.band_ale.estimator_band_pc_filter_halfwidth;
  ale.band_ale.estimator_band_pc_slope_limit =
      ale_defaults.band_ale.estimator_band_pc_slope_limit;
  ale.band_ale.estimator_band_pc_slope_reject =
      ale_defaults.band_ale.estimator_band_pc_slope_reject;
  ale.band_ale.estimator_band_pc_curvature_limit =
      ale_defaults.band_ale.estimator_band_pc_curvature_limit;
  ale.band_ale.estimator_band_pc_chi_max =
      ale_defaults.band_ale.estimator_band_pc_chi_max;
  ale.band_ale.estimator_band_pc_chi_step =
      ale_defaults.band_ale.estimator_band_pc_chi_step;
  ale.band_ale.estimator_band_pc_sigma_floor =
      ale_defaults.band_ale.estimator_band_pc_sigma_floor;
  ale.band_ale.estimator_band_pc_coverage_full =
      ale_defaults.band_ale.estimator_band_pc_coverage_full;
  ale.band_ale.estimator_band_pc_coverage_min =
      ale_defaults.band_ale.estimator_band_pc_coverage_min;
  ale.band_ale.estimator_band_pc_cooldown_events =
      ale_defaults.band_ale.estimator_band_pc_cooldown_events;
  ale.band_ale.estimator_band_pc_phase_b =
      ale_defaults.band_ale.estimator_band_pc_phase_b;
  ale.band_ale.closure_catchment_enabled =
      ale_defaults.band_ale.closure_catchment_enabled;
  ale.band_ale.closure_catchment_forced_active =
      ale_defaults.band_ale.closure_catchment_forced_active;
  ale.band_ale.closure_catchment_s_catch_cm =
      ale_defaults.band_ale.closure_catchment_s_catch_cm;
  ale.band_ale.closure_catchment_s_protect_cm =
      ale_defaults.band_ale.closure_catchment_s_protect_cm;
  ale.band_ale.closure_catchment_spacing_floor_cm =
      ale_defaults.band_ale.closure_catchment_spacing_floor_cm;
  ale.band_ale.closure_catchment_ratio_max =
      ale_defaults.band_ale.closure_catchment_ratio_max;
  ale.band_ale.closure_catchment_nu_max =
      ale_defaults.band_ale.closure_catchment_nu_max;
  ale.band_ale.closure_catchment_max_bites =
      ale_defaults.band_ale.closure_catchment_max_bites;
  ale.band_ale.closure_catchment_shock_hold =
      ale_defaults.band_ale.closure_catchment_shock_hold;
  ale.band_ale.closure_catchment_eta_h_arm =
      ale_defaults.band_ale.closure_catchment_eta_h_arm;
  ale.band_ale.closure_catchment_eta_h_full =
      ale_defaults.band_ale.closure_catchment_eta_h_full;
  ale.band_ale.closure_catchment_eta_m_arm =
      ale_defaults.band_ale.closure_catchment_eta_m_arm;
  ale.band_ale.closure_catchment_eta_m_full =
      ale_defaults.band_ale.closure_catchment_eta_m_full;
  ale.band_ale.closure_catchment_reset_eta =
      ale_defaults.band_ale.closure_catchment_reset_eta;
  ale.band_ale.closure_catchment_support_core_rows =
      ale_defaults.band_ale.closure_catchment_support_core_rows;
  ale.band_ale.closure_catchment_support_taper_rows =
      ale_defaults.band_ale.closure_catchment_support_taper_rows;
  ale.band_ale.closure_catchment_accum_frac =
      ale_defaults.band_ale.closure_catchment_accum_frac;
  ale.band_ale.closure_catchment_rearm_drop =
      ale_defaults.band_ale.closure_catchment_rearm_drop;
  ale.band_ale.pole_theta_enabled =
      ale_defaults.band_ale.pole_theta_enabled;
  ale.band_ale.pole_theta_routine_enabled =
      ale_defaults.band_ale.pole_theta_routine_enabled;
  ale.band_ale.pole_theta_phys_lp =
      ale_defaults.band_ale.pole_theta_phys_lp;
  ale.band_ale.pole_theta_phys_lc =
      ale_defaults.band_ale.pole_theta_phys_lc;
  ale.band_ale.pole_theta_noise_floor =
      ale_defaults.band_ale.pole_theta_noise_floor;
  ale.band_ale.pole_theta_noise_ceiling =
      ale_defaults.band_ale.pole_theta_noise_ceiling;
  ale.band_ale.pole_theta_h_arm =
      ale_defaults.band_ale.pole_theta_h_arm;
  ale.band_ale.pole_theta_h_fire =
      ale_defaults.band_ale.pole_theta_h_fire;
  ale.band_ale.pole_theta_h_hard =
      ale_defaults.band_ale.pole_theta_h_hard;
  ale.band_ale.pole_theta_h_release =
      ale_defaults.band_ale.pole_theta_h_release;
  ale.band_ale.pole_theta_kappa_arm =
      ale_defaults.band_ale.pole_theta_kappa_arm;
  ale.band_ale.pole_theta_kappa_fire =
      ale_defaults.band_ale.pole_theta_kappa_fire;
  ale.band_ale.pole_theta_kappa_hard =
      ale_defaults.band_ale.pole_theta_kappa_hard;
  ale.band_ale.pole_theta_alpha =
      ale_defaults.band_ale.pole_theta_alpha;
  ale.band_ale.pole_theta_alpha_hard =
      ale_defaults.band_ale.pole_theta_alpha_hard;
  ale.band_ale.pole_theta_deadband_frac =
      ale_defaults.band_ale.pole_theta_deadband_frac;
  ale.band_ale.pole_theta_move_limit_frac =
      ale_defaults.band_ale.pole_theta_move_limit_frac;
  ale.band_ale.pole_theta_move_limit_hard_frac =
      ale_defaults.band_ale.pole_theta_move_limit_hard_frac;
  ale.band_ale.pole_theta_cooldown_s =
      ale_defaults.band_ale.pole_theta_cooldown_s;
  ale.band_ale.pole_theta_cooldown_base_s =
      ale_defaults.band_ale.pole_theta_cooldown_base_s;
  ale.band_ale.pole_theta_predict_window_s =
      ale_defaults.band_ale.pole_theta_predict_window_s;
  ale.band_ale.pole_theta_predict_horizon_s =
      ale_defaults.band_ale.pole_theta_predict_horizon_s;
  ale.band_ale.pole_theta_halo_columns =
      ale_defaults.band_ale.pole_theta_halo_columns;
  ale.band_ale.pole_theta_halo_rows =
      ale_defaults.band_ale.pole_theta_halo_rows;
  ale.band_ale.pole_theta_post_h_floor =
      ale_defaults.band_ale.pole_theta_post_h_floor;
  ale.band_ale.pole_theta_curve_preserving =
      ale_defaults.band_ale.pole_theta_curve_preserving;
  ale.band_ale.pole_theta_protected_modes =
      ale_defaults.band_ale.pole_theta_protected_modes;
  ale.band_ale.pole_theta_fit_order =
      ale_defaults.band_ale.pole_theta_fit_order;
  ale.band_ale.pole_theta_shock_hold =
      ale_defaults.band_ale.pole_theta_shock_hold;
  ale.band_ale.estimator_band_pc_tube_dilate_rows =
      ale_defaults.band_ale.estimator_band_pc_tube_dilate_rows;
  ale.band_ale.estimator_band_pc_tube_dilate_cols_extra =
      ale_defaults.band_ale.estimator_band_pc_tube_dilate_cols_extra;
  ale.band_ale.estimator_band_pc_ambiguous_hold_fraction =
      ale_defaults.band_ale.estimator_band_pc_ambiguous_hold_fraction;
  parent.numerics.diagnostics.refinement_estimator =
      core::Config::NumericsConfig::DiagnosticsConfig::RefinementEstimator{};
  parent.numerics.diagnostics.refinement_autopilot =
      core::Config::NumericsConfig::DiagnosticsConfig::RefinementAutopilot{};
  parent.numerics.diagnostics.conduction_energy_rate_export =
      core::Config::NumericsConfig::DiagnosticsConfig::
          ConductionEnergyRateExportConfig{};
  parent.numerics.z_reflection =
      core::Config::NumericsConfig::ZReflectionConfig{};
  parent.numerics.hydro.pressure_drive_perturbation =
      core::Config::NumericsConfig::HydroConfig::
          PressureDrivePerturbationConfig{};
  // Inherit the frozen parent values so reconstruction matches the checkpoint
  // config byte-exactly. Old checkpoints lack the keys and keep the reset
  // defaults (legacy default completion equivalence).
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "z_reflection"}, "mode",
      &parent.numerics.z_reflection.mode);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "hydro", "pressure_drive_perturbation"},
      "enabled", &parent.numerics.hydro.pressure_drive_perturbation.enabled);
  inherit_frozen_pressure_drive_composites_if_present(
      frozen, &parent.numerics.hydro.pressure_drive_perturbation);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "hydro", "pressure_drive_perturbation"},
      "random_seed",
      &parent.numerics.hydro.pressure_drive_perturbation.random_seed);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "hydro", "pressure_drive_perturbation"},
      "random_l_min",
      &parent.numerics.hydro.pressure_drive_perturbation.random_l_min);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "hydro", "pressure_drive_perturbation"},
      "random_l_max",
      &parent.numerics.hydro.pressure_drive_perturbation.random_l_max);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "hydro", "pressure_drive_perturbation"},
      "random_rms",
      &parent.numerics.hydro.pressure_drive_perturbation.random_rms);
  parent.numerics.hydro.pressure_drive_perturbation.random_enabled =
      parent.numerics.hydro.pressure_drive_perturbation.random_rms > 0.0;
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "respace_move_cap_frac",
      &parent.numerics.ale.band_ale.respace_move_cap_frac);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "estimator_band_hold_mach",
      &parent.numerics.ale.band_ale.estimator_band_hold_mach);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "estimator_band_per_column",
      &parent.numerics.ale.band_ale.estimator_band_per_column);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "estimator_band_pc_filter_halfwidth",
      &parent.numerics.ale.band_ale.estimator_band_pc_filter_halfwidth);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "estimator_band_pc_slope_limit",
      &parent.numerics.ale.band_ale.estimator_band_pc_slope_limit);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "estimator_band_pc_slope_reject",
      &parent.numerics.ale.band_ale.estimator_band_pc_slope_reject);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "estimator_band_pc_curvature_limit",
      &parent.numerics.ale.band_ale.estimator_band_pc_curvature_limit);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "estimator_band_pc_chi_max",
      &parent.numerics.ale.band_ale.estimator_band_pc_chi_max);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "estimator_band_pc_chi_step",
      &parent.numerics.ale.band_ale.estimator_band_pc_chi_step);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "estimator_band_pc_sigma_floor",
      &parent.numerics.ale.band_ale.estimator_band_pc_sigma_floor);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "estimator_band_pc_coverage_full",
      &parent.numerics.ale.band_ale.estimator_band_pc_coverage_full);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "estimator_band_pc_coverage_min",
      &parent.numerics.ale.band_ale.estimator_band_pc_coverage_min);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "estimator_band_pc_cooldown_events",
      &parent.numerics.ale.band_ale.estimator_band_pc_cooldown_events);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "estimator_band_pc_phase_b",
      &parent.numerics.ale.band_ale.estimator_band_pc_phase_b);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "closure_catchment_enabled",
      &parent.numerics.ale.band_ale.closure_catchment_enabled);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "closure_catchment_forced_active",
      &parent.numerics.ale.band_ale.closure_catchment_forced_active);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "closure_catchment_s_catch_cm",
      &parent.numerics.ale.band_ale.closure_catchment_s_catch_cm);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "closure_catchment_s_protect_cm",
      &parent.numerics.ale.band_ale.closure_catchment_s_protect_cm);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "closure_catchment_spacing_floor_cm",
      &parent.numerics.ale.band_ale.closure_catchment_spacing_floor_cm);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "closure_catchment_ratio_max",
      &parent.numerics.ale.band_ale.closure_catchment_ratio_max);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "closure_catchment_nu_max",
      &parent.numerics.ale.band_ale.closure_catchment_nu_max);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "closure_catchment_max_bites",
      &parent.numerics.ale.band_ale.closure_catchment_max_bites);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "closure_catchment_shock_hold",
      &parent.numerics.ale.band_ale.closure_catchment_shock_hold);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "closure_catchment_eta_h_arm",
      &parent.numerics.ale.band_ale.closure_catchment_eta_h_arm);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "closure_catchment_eta_h_full",
      &parent.numerics.ale.band_ale.closure_catchment_eta_h_full);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "closure_catchment_eta_m_arm",
      &parent.numerics.ale.band_ale.closure_catchment_eta_m_arm);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "closure_catchment_eta_m_full",
      &parent.numerics.ale.band_ale.closure_catchment_eta_m_full);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "closure_catchment_reset_eta",
      &parent.numerics.ale.band_ale.closure_catchment_reset_eta);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "closure_catchment_support_core_rows",
      &parent.numerics.ale.band_ale.closure_catchment_support_core_rows);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "closure_catchment_support_taper_rows",
      &parent.numerics.ale.band_ale.closure_catchment_support_taper_rows);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "closure_catchment_accum_frac",
      &parent.numerics.ale.band_ale.closure_catchment_accum_frac);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "closure_catchment_rearm_drop",
      &parent.numerics.ale.band_ale.closure_catchment_rearm_drop);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_enabled",
      &parent.numerics.ale.band_ale.pole_theta_enabled);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_routine_enabled",
      &parent.numerics.ale.band_ale.pole_theta_routine_enabled);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_phys_lp",
      &parent.numerics.ale.band_ale.pole_theta_phys_lp);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_phys_lc",
      &parent.numerics.ale.band_ale.pole_theta_phys_lc);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_noise_floor",
      &parent.numerics.ale.band_ale.pole_theta_noise_floor);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_noise_ceiling",
      &parent.numerics.ale.band_ale.pole_theta_noise_ceiling);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_h_arm",
      &parent.numerics.ale.band_ale.pole_theta_h_arm);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_h_fire",
      &parent.numerics.ale.band_ale.pole_theta_h_fire);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_h_hard",
      &parent.numerics.ale.band_ale.pole_theta_h_hard);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_h_release",
      &parent.numerics.ale.band_ale.pole_theta_h_release);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_kappa_arm",
      &parent.numerics.ale.band_ale.pole_theta_kappa_arm);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_kappa_fire",
      &parent.numerics.ale.band_ale.pole_theta_kappa_fire);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_kappa_hard",
      &parent.numerics.ale.band_ale.pole_theta_kappa_hard);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_alpha",
      &parent.numerics.ale.band_ale.pole_theta_alpha);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_alpha_hard",
      &parent.numerics.ale.band_ale.pole_theta_alpha_hard);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_deadband_frac",
      &parent.numerics.ale.band_ale.pole_theta_deadband_frac);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_move_limit_frac",
      &parent.numerics.ale.band_ale.pole_theta_move_limit_frac);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_move_limit_hard_frac",
      &parent.numerics.ale.band_ale.pole_theta_move_limit_hard_frac);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_cooldown_s",
      &parent.numerics.ale.band_ale.pole_theta_cooldown_s);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_cooldown_base_s",
      &parent.numerics.ale.band_ale.pole_theta_cooldown_base_s);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_predict_window_s",
      &parent.numerics.ale.band_ale.pole_theta_predict_window_s);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_predict_horizon_s",
      &parent.numerics.ale.band_ale.pole_theta_predict_horizon_s);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_halo_columns",
      &parent.numerics.ale.band_ale.pole_theta_halo_columns);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_halo_rows",
      &parent.numerics.ale.band_ale.pole_theta_halo_rows);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_post_h_floor",
      &parent.numerics.ale.band_ale.pole_theta_post_h_floor);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_curve_preserving",
      &parent.numerics.ale.band_ale.pole_theta_curve_preserving);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_protected_modes",
      &parent.numerics.ale.band_ale.pole_theta_protected_modes);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_fit_order",
      &parent.numerics.ale.band_ale.pole_theta_fit_order);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "pole_theta_shock_hold",
      &parent.numerics.ale.band_ale.pole_theta_shock_hold);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "estimator_band_pc_tube_dilate_rows",
      &parent.numerics.ale.band_ale.estimator_band_pc_tube_dilate_rows);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "estimator_band_pc_tube_dilate_cols_extra",
      &parent.numerics.ale.band_ale.estimator_band_pc_tube_dilate_cols_extra);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "ale", "band_ale"},
      "estimator_band_pc_ambiguous_hold_fraction",
      &parent.numerics.ale.band_ale.estimator_band_pc_ambiguous_hold_fraction);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_estimator"},
      "enabled", &parent.numerics.diagnostics.refinement_estimator.enabled);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_estimator"},
      "every", &parent.numerics.diagnostics.refinement_estimator.every);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_estimator"},
      "filter_eps",
      &parent.numerics.diagnostics.refinement_estimator.filter_eps);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_estimator"},
      "detect_cutoff",
      &parent.numerics.diagnostics.refinement_estimator.detect_cutoff);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "conduction_energy_rate_export"},
      "enabled",
      &parent.numerics.diagnostics.conduction_energy_rate_export.enabled);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_autopilot"},
      "enabled", &parent.numerics.diagnostics.refinement_autopilot.enabled);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_autopilot"},
      "e_on", &parent.numerics.diagnostics.refinement_autopilot.e_on);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_autopilot"},
      "e_off", &parent.numerics.diagnostics.refinement_autopilot.e_off);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_autopilot"},
      "assoc_cut",
      &parent.numerics.diagnostics.refinement_autopilot.assoc_cut);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_autopilot"},
      "strong_cut",
      &parent.numerics.diagnostics.refinement_autopilot.strong_cut);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_autopilot"},
      "gap_bridge",
      &parent.numerics.diagnostics.refinement_autopilot.gap_bridge);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_autopilot"},
      "persist", &parent.numerics.diagnostics.refinement_autopilot.persist);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_autopilot"},
      "n_q_plan", &parent.numerics.diagnostics.refinement_autopilot.n_q_plan);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_autopilot"},
      "chi_design",
      &parent.numerics.diagnostics.refinement_autopilot.chi_design);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_autopilot"},
      "s_rep_cm", &parent.numerics.diagnostics.refinement_autopilot.s_rep_cm);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_autopilot"},
      "handoff_cm",
      &parent.numerics.diagnostics.refinement_autopilot.handoff_cm);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_autopilot"},
      "window_lo_h",
      &parent.numerics.diagnostics.refinement_autopilot.window_lo_h);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_autopilot"},
      "window_hi_h",
      &parent.numerics.diagnostics.refinement_autopilot.window_hi_h);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_autopilot"},
      "history", &parent.numerics.diagnostics.refinement_autopilot.history);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_autopilot"},
      "cov_min", &parent.numerics.diagnostics.refinement_autopilot.cov_min);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_autopilot"},
      "mode", &parent.numerics.diagnostics.refinement_autopilot.mode);
  inherit_frozen_path_value_if_present(
      frozen, {"numerics", "diagnostics", "refinement_autopilot"},
      "ckpt_lead_h",
      &parent.numerics.diagnostics.refinement_autopilot.ckpt_lead_h);

  inherit_frozen_value_if_present(
      frozen, "main", "t_end", &parent.main.t_end);
  // Checkpoint freeze always emits all six cadence values and never emits the
  // Builder-only *_explicit flags, so restoring the values is byte-exact.
  inherit_frozen_value_if_present(
      frozen, "output", "plot_every", &parent.output.plot_every);
  inherit_frozen_value_if_present(
      frozen, "output", "history_every", &parent.output.history_every);
  inherit_frozen_value_if_present(
      frozen, "output", "checkpoint_every", &parent.output.checkpoint_every);
  inherit_frozen_value_if_present(
      frozen, "output", "plot_every_s", &parent.output.plot_every_s);
  inherit_frozen_value_if_present(
      frozen, "output", "history_every_s", &parent.output.history_every_s);
  inherit_frozen_value_if_present(frozen,
                                  "output",
                                  "checkpoint_every_s",
                                  &parent.output.checkpoint_every_s);
  inherit_frozen_value_if_present(frozen,
                                  "numerics",
                                  "diagnostics_every",
                                  &parent.numerics.diagnostics_every);
  // Frozen-config equivalence includes the source hash. Preserve A's hash so
  // the normal reader still compares every derived parent setting while the
  // segment-2 deck remains a distinct source file.
  parent.meta.namelist_source_hash = frozen_namelist_source_hash(frozen);
  parent.meta.frozen_config_json =
      core::namelist::Freeze::to_checkpoint_json(parent);
  return parent;
}

void validate_segment2_config(const core::Config& cfg) {
  if (cfg.main.dimension != "2D_RZ" || cfg.main.dim != 2) {
    throw std::runtime_error(
        "checkpoint-swap-center requires a 2D_RZ segment-2 deck");
  }
  if (cfg.mesh.topology_scheme !=
      core::TopologyScheme::MULTIBLOCK_POLAR_TIER_CART_CENTER) {
    throw std::runtime_error(
        "checkpoint-swap-center requires Mesh.topology_scheme=multiblock_polar_tier_cart_center");
  }
  if (cfg.numerics.ale.mesh_mode == "reale_v2") {
    throw std::runtime_error(
        "checkpoint-swap-center v1 refuses a carrier/ReALE segment-2 deck; fixed mesh only");
  }
  if (cfg.radiation.enabled) {
    throw std::runtime_error(
        "checkpoint-swap-center v1 refuses a segment-2 deck with Radiation.enabled=true");
  }
  if (cfg.laser.enabled) {
    throw std::runtime_error(
        "checkpoint-swap-center v1 refuses a segment-2 deck with Laser.enabled=true");
  }
  if (cfg.burn.enabled && cfg.burn.scheme == "mc") {
    throw std::runtime_error(
        "checkpoint-swap-center v1 refuses a segment-2 deck with burn MC active");
  }
}

void assert_optional_parameters(
    const CheckpointSwapCenterOptions& options,
    const core::Config& cfg,
    const core::PolarTierTruncation& truncation) {
  const int derived_n_c = truncation.n_theta_cut / 4;
  if (options.assert_cut_ring.has_value() &&
      *options.assert_cut_ring != cfg.mesh.polar_tier_cart_cut_ring) {
    throw std::runtime_error(
        "checkpoint-swap-center --cut-ring assertion does not match the segment-2 deck");
  }
  if (options.assert_core_cells_per_half.has_value() &&
      *options.assert_core_cells_per_half != derived_n_c) {
    throw std::runtime_error(
        "checkpoint-swap-center --core-cells assertion does not match derived n_c");
  }
  if (options.assert_bridge_layers.has_value() &&
      *options.assert_bridge_layers !=
          cfg.mesh.multiblock_cart_core_bridge_layers) {
    throw std::runtime_error(
        "checkpoint-swap-center --bridge-layers assertion does not match the segment-2 deck");
  }
  if (options.assert_r_c_cm.has_value()) {
    if (!std::isfinite(*options.assert_r_c_cm)) {
      throw std::runtime_error(
          "checkpoint-swap-center --r-c assertion must be finite");
    }
    const double expected = cfg.mesh.multiblock_cart_core_r_c;
    const double tolerance =
        64.0 * std::numeric_limits<double>::epsilon() *
        std::max({1.0, std::abs(expected),
                  std::abs(*options.assert_r_c_cm)});
    if (std::abs(*options.assert_r_c_cm - expected) > tolerance) {
      throw std::runtime_error(
          "checkpoint-swap-center --r-c assertion does not match the segment-2 deck");
    }
  }
}

core::State build_fresh_target(
    core::Config& cfg,
    const core::namelist::Builder& builder,
    const core::State& parent,
    const std::size_t kept_nodes) {
  core::State target =
      core::State::allocate(cfg, builder.hydro_t_start_eV);
  target.mesh = mesh::create_mesh(cfg, target);

  // The current seam is checkpoint A's deformed seam. Install it before
  // evaluating fresh cell-centered deck data so bridge volumes and centroids
  // tessellate the actual seam-bounded polygon at t_swap.
  copy_field_prefix(parent.x_r, target.x_r, kept_nodes, "mesh/x_r");
  copy_field_prefix(parent.x_z, target.x_z, kept_nodes, "mesh/x_z");
  target.mesh.recompute_geometry();
  target.vol = target.mesh.cell_vol;
  core::namelist::evaluate_geometry(cfg, builder, target);
  // The replacement epoch is quiescent by definition. Geometry supplies the
  // deck-authoritative material state, but any optional launch velocity is not
  // part of the fresh center; the kept node prefix is restored from A below.
  target.v_r.fill(0.0);
  target.v_z.fill(0.0);
  coupling::initialize_eos_fields_if_needed(target, cfg);
  if (cfg.numerics.materials.per_material_conservation_enabled) {
    core::populate_per_material_conserved_from_cell_mean(target, cfg);
  }
  if (!parent.cv_e.empty()) {
    target.cv_e.reset(target.rho.size());
    target.cv_i.reset(target.rho.size());
  }
  hydro::HydroEOSContext eos_context;
  eos_context.initialize(cfg);
  hydro::Hydro2D hydro;
  hydro.prepare_initial_sound_speed(target, cfg, &eos_context);
  hydro.prepare_initial_corner_mass(target, cfg);
  return target;
}

double normalize_fresh_tail_velocities(core::State& target,
                                       const std::size_t kept_nodes) {
  if (target.v_r.size() != target.v_z.size() ||
      kept_nodes > target.v_r.size()) {
    throw std::runtime_error(
        "checkpoint-swap-center fresh velocity tail extent mismatch");
  }
  auto v_r = field_to_host(target.v_r);
  auto v_z = field_to_host(target.v_z);
  double prenorm_max = 0.0;
  for (std::size_t node = kept_nodes; node < v_r.size(); ++node) {
    const double speed = std::hypot(v_r[node], v_z[node]);
    if (!std::isfinite(speed)) {
      throw std::runtime_error(
          "checkpoint-swap-center fresh velocity tail contains a non-finite value");
    }
    prenorm_max = std::max(prenorm_max, speed);
  }
  if (prenorm_max > kFreshVelocityNormalizationLimitCmS) {
    std::ostringstream message;
    message << std::setprecision(17)
            << "checkpoint-swap-center fresh velocity normalization refused: "
            << "fresh_v_prenorm_max_cm_s=" << prenorm_max
            << " exceeds " << kFreshVelocityNormalizationLimitCmS;
    throw std::runtime_error(message.str());
  }
  std::fill(v_r.begin() + static_cast<std::ptrdiff_t>(kept_nodes),
            v_r.end(),
            0.0);
  std::fill(v_z.begin() + static_cast<std::ptrdiff_t>(kept_nodes),
            v_z.end(),
            0.0);
  if (!v_r.empty()) {
    target.v_r.copy_from_host(v_r.data());
    target.v_z.copy_from_host(v_z.data());
  }
  return prenorm_max;
}

#endif  // TENRYU_ENABLE_PYTHON

}  // namespace

#if TENRYU_ENABLE_PYTHON
core::Config make_parent_config(
    const core::Config& segment2,
    const std::string& parent_frozen_config) {
  return make_parent_config_impl(segment2, parent_frozen_config);
}
#endif

CheckpointSwapCenterGateMeasurements measure_checkpoint_swap_a_side(
    const core::State& parent,
    const core::Config& cfg,
    const core::PolarTierLayout& truncated,
    const core::PolarTierTruncation& truncation,
    const int lmax,
    const CheckpointSwapCenterThresholds& thresholds) {
  return measure_a_side_quiescence(
      parent, cfg, truncated, truncation, lmax, thresholds);
}

CheckpointSwapCenterGateMeasurements measure_checkpoint_swap_b_side(
    const core::State& parent,
    const core::State& fresh_target,
    const std::size_t first_parent_cell,
    const std::size_t first_target_cell,
    const CheckpointSwapCenterGateMeasurements& a_side,
    const CheckpointSwapCenterThresholds& thresholds) {
  return measure_b_side_quiescence(
      parent,
      fresh_target,
      first_parent_cell,
      first_target_cell,
      a_side,
      thresholds);
}

void assert_checkpoint_swap_a_side_gates(
    const CheckpointSwapCenterGateMeasurements& measured,
    const CheckpointSwapCenterThresholds& thresholds) {
  if (!(std::isfinite(measured.max_speed_cm_s) &&
        measured.max_speed_cm_s <= thresholds.max_speed_cm_s)) {
    throw std::runtime_error(format_gate_failure(measured, thresholds));
  }
  if (!(measured.max_seam_displacement_cm <=
        thresholds.seam_displacement_floor_cm)) {
    throw std::runtime_error(
        format_seam_displacement_failure(measured, thresholds));
  }
  if (!(std::isfinite(measured.max_mirror_relative) &&
        measured.max_mirror_relative <= thresholds.mirror_tolerance)) {
    throw std::runtime_error(format_gate_failure(measured, thresholds));
  }
  if (!(std::isfinite(measured.max_legendre_relative) &&
        measured.max_legendre_relative <= thresholds.legendre_tolerance)) {
    throw std::runtime_error(format_gate_failure(measured, thresholds));
  }
}

void assert_checkpoint_swap_topology_prefix(
    const std::vector<int>& parent_cell_node_csr_offsets,
    const std::vector<int>& parent_cell_node_csr_indices,
    const std::vector<unsigned char>& parent_cell_nverts,
    const std::vector<int>& target_cell_node_csr_offsets,
    const std::vector<int>& target_cell_node_csr_indices,
    const std::vector<unsigned char>& target_cell_nverts,
    const std::size_t kept_cells) {
  if (parent_cell_node_csr_offsets.size() < kept_cells + 1U ||
      target_cell_node_csr_offsets.size() < kept_cells + 1U ||
      parent_cell_nverts.size() < kept_cells ||
      target_cell_nverts.size() < kept_cells) {
    throw std::runtime_error(
        "checkpoint-swap-center topology-prefix gate extent mismatch");
  }
  for (std::size_t c = 0; c <= kept_cells; ++c) {
    if (parent_cell_node_csr_offsets[c] !=
        target_cell_node_csr_offsets[c]) {
      throw std::runtime_error(
          "checkpoint-swap-center topology-prefix gate: first cell_node CSR offset divergence at " +
          std::to_string(c));
    }
  }
  const int prefix_entries = parent_cell_node_csr_offsets[kept_cells];
  if (prefix_entries < 0 ||
      parent_cell_node_csr_indices.size() <
          static_cast<std::size_t>(prefix_entries) ||
      target_cell_node_csr_indices.size() <
          static_cast<std::size_t>(prefix_entries)) {
    throw std::runtime_error(
        "checkpoint-swap-center topology-prefix gate CSR entry extent mismatch");
  }
  for (int entry = 0; entry < prefix_entries; ++entry) {
    if (parent_cell_node_csr_indices[static_cast<std::size_t>(entry)] !=
        target_cell_node_csr_indices[static_cast<std::size_t>(entry)]) {
      throw std::runtime_error(
          "checkpoint-swap-center topology-prefix gate: first cell_node CSR content divergence at entry " +
          std::to_string(entry));
    }
  }
  for (std::size_t c = 0; c < kept_cells; ++c) {
    if (parent_cell_nverts[c] != target_cell_nverts[c]) {
      throw std::runtime_error(
          "checkpoint-swap-center topology-prefix gate: first cell_nverts divergence at cell " +
          std::to_string(c));
    }
  }
}

bool checkpoint_swap_gates_pass(
    const CheckpointSwapCenterGateMeasurements& measured,
    const CheckpointSwapCenterThresholds& thresholds) {
  return std::isfinite(measured.max_speed_cm_s) &&
         std::isfinite(measured.max_mach) &&
         std::isfinite(measured.max_rho_relative) &&
         std::isfinite(measured.max_temperature_relative) &&
         std::isfinite(measured.max_seam_displacement_cm) &&
         std::isfinite(measured.max_mirror_relative) &&
         std::isfinite(measured.max_legendre_relative) &&
         measured.max_speed_cm_s <= thresholds.max_speed_cm_s &&
         measured.max_mach <= thresholds.max_mach &&
         measured.max_rho_relative <= thresholds.max_rho_relative &&
         measured.max_temperature_relative <=
             thresholds.max_temperature_relative &&
         measured.max_seam_displacement_cm <=
             thresholds.seam_displacement_floor_cm &&
         measured.max_mirror_relative <= thresholds.mirror_tolerance &&
         measured.max_legendre_relative <= thresholds.legendre_tolerance;
}

double checkpoint_swap_legendre_margin(
    const std::vector<CheckpointSwapCenterAngularSample>& samples,
    const int lmax) {
  if (samples.empty() || lmax < 1) {
    return 0.0;
  }
  long double total_weight = 0.0L;
  for (const auto& sample : samples) {
    if (!std::isfinite(sample.weight) || sample.weight < 0.0 ||
        !std::isfinite(sample.mu) || sample.mu < -1.0 || sample.mu > 1.0 ||
        !std::isfinite(sample.rho_relative_delta) ||
        !std::isfinite(sample.Te_relative_delta) ||
        !std::isfinite(sample.Ti_relative_delta)) {
      throw std::runtime_error(
          "checkpoint-swap-center Legendre gate received invalid sample");
    }
    total_weight += sample.weight;
  }
  if (!(total_weight > 0.0L)) {
    throw std::runtime_error(
        "checkpoint-swap-center Legendre gate has zero total weight");
  }
  double margin = 0.0;
  for (int ell = 1; ell <= lmax; ++ell) {
    std::array<long double, 3> moment{};
    for (const auto& sample : samples) {
      const long double factor =
          static_cast<long double>(sample.weight) *
          legendre_p(ell, sample.mu);
      moment[0] += factor * sample.rho_relative_delta;
      moment[1] += factor * sample.Te_relative_delta;
      moment[2] += factor * sample.Ti_relative_delta;
    }
    for (const long double value : moment) {
      margin = std::max(
          margin,
          std::abs(static_cast<double>(value / total_weight)));
    }
  }
  return margin;
}

CheckpointSwapCenterConservationComparison
compare_checkpoint_swap_conservation(
    const CheckpointSwapCenterConservationTotals& dropped,
    const CheckpointSwapCenterConservationTotals& fresh,
    const double kappa,
    const double conservation_v_floor_cm_s) {
  if (!(std::isfinite(kappa) && kappa > 0.0) ||
      !(std::isfinite(conservation_v_floor_cm_s) &&
        conservation_v_floor_cm_s > 0.0)) {
    throw std::runtime_error(
        "checkpoint-swap-center ledger kappa and conservation velocity floor must be finite and positive");
  }
  CheckpointSwapCenterConservationComparison out;
  out.delta_mass = static_cast<double>(fresh.mass - dropped.mass);
  out.delta_energy = static_cast<double>(fresh.energy - dropped.energy);
  out.delta_momentum_r =
      static_cast<double>(fresh.momentum_r - dropped.momentum_r);
  out.delta_momentum_z =
      static_cast<double>(fresh.momentum_z - dropped.momentum_z);

  const auto tolerance = [kappa](const long double scale,
                                 const std::size_t terms) {
    return kappa * gamma_bound(terms) * static_cast<double>(scale);
  };
  out.tolerance_mass = tolerance(
      dropped.abs_mass_terms + fresh.abs_mass_terms,
      dropped.mass_terms + fresh.mass_terms);
  out.tolerance_energy = tolerance(
      dropped.abs_energy_terms + fresh.abs_energy_terms,
      dropped.energy_terms + fresh.energy_terms);
  out.momentum_floor = static_cast<double>(
      0.5L * (dropped.mass + fresh.mass) * conservation_v_floor_cm_s);
  out.tolerance_momentum_r = std::max(
      tolerance(dropped.abs_momentum_r_terms + fresh.abs_momentum_r_terms,
                dropped.momentum_terms + fresh.momentum_terms),
      out.momentum_floor);
  out.tolerance_momentum_z = std::max(
      tolerance(dropped.abs_momentum_z_terms + fresh.abs_momentum_z_terms,
                dropped.momentum_terms + fresh.momentum_terms),
      out.momentum_floor);
  out.passed = std::isfinite(out.delta_mass) &&
               std::isfinite(out.delta_energy) &&
               std::isfinite(out.delta_momentum_r) &&
               std::isfinite(out.delta_momentum_z) &&
               std::isfinite(out.tolerance_mass) &&
               std::isfinite(out.tolerance_energy) &&
               std::isfinite(out.tolerance_momentum_r) &&
               std::isfinite(out.tolerance_momentum_z) &&
               std::isfinite(out.momentum_floor) &&
               std::abs(out.delta_mass) <= out.tolerance_mass &&
               std::abs(out.delta_energy) <= out.tolerance_energy &&
               std::abs(out.delta_momentum_r) <=
                   out.tolerance_momentum_r &&
               std::abs(out.delta_momentum_z) <=
                   out.tolerance_momentum_z;
  return out;
}

void publish_checkpoint_identity_noop(
    const std::filesystem::path& input,
    const std::filesystem::path& output) {
  require_publish_target_available(output);
  const auto temporary = temporary_sibling_path(output);
  try {
    std::filesystem::copy_file(input,
                               temporary,
                               std::filesystem::copy_options::none);
    if (!files_byte_equal(input, temporary)) {
      throw std::runtime_error(
          "checkpoint-swap-center identity-noop byte comparison failed");
    }
    atomic_publish(temporary, output);
  } catch (...) {
    std::error_code cleanup_error;
    std::filesystem::remove(temporary, cleanup_error);
    throw;
  }
}

int cmd_checkpoint_swap_center(
    const CheckpointSwapCenterOptions& options) {
#if TENRYU_ENABLE_PYTHON && TENRYU_ENABLE_HDF5
  try {
    validate_thresholds(options.thresholds);
    core::namelist::PythonGuard python_guard;
    core::namelist::Runtime runtime;
    runtime.execute(options.namelist_path);
    core::Config cfg = runtime.config();
    validate_segment2_config(cfg);

    const CheckpointPreflight preflight =
        read_checkpoint_preflight(options.input_checkpoint);
    // These state families have adopted future transfer policies but no v1
    // implementation. Refuse them before constructing or publishing B.
    if (preflight.has_carrier_group) {
      throw std::runtime_error(
          "checkpoint-swap-center v1 refuses checkpoint A with mesh/carrier/v1 (carrier/ReALE checkpoint)");
    }
    if (preflight.n_particles > 0) {
      throw std::runtime_error(
          "checkpoint-swap-center v1 refuses checkpoint A with live photon particles (particles/@n_particles > 0)");
    }
    if (preflight.burn_mc_live > 0 || preflight.burn_inflight != 0.0) {
      throw std::runtime_error(
          "checkpoint-swap-center v1 refuses checkpoint A with live burn-MC state (burn_mc_live > 0 or E_burn_inflight != 0)");
    }
    reject_unsupported_frozen_physics(preflight.frozen_config);

    const std::filesystem::path output(options.output_checkpoint);
    if (!options.dry_run) {
      require_publish_target_available(output);
    }

    if (options.identity_noop) {
      io::HDF5Reader reader;
      hydro::axis_core_set_released_units(0);
      (void)reader.read_checkpoint(cfg, preflight.resolved_path.string());
      hydro::axis_core_set_released_units(0);
      if (!options.dry_run) {
        const auto temporary = temporary_sibling_path(output);
        try {
          std::filesystem::copy_file(preflight.resolved_path,
                                     temporary,
                                     std::filesystem::copy_options::none);
          (void)reader.read_checkpoint(cfg, temporary.string());
          hydro::axis_core_set_released_units(0);
          if (!files_byte_equal(preflight.resolved_path, temporary)) {
            throw std::runtime_error(
                "checkpoint-swap-center identity-noop byte comparison failed");
          }
          atomic_publish(temporary, output);
        } catch (...) {
          hydro::axis_core_set_released_units(0);
          std::error_code cleanup_error;
          std::filesystem::remove(temporary, cleanup_error);
          throw;
        }
      }
      std::cout
          << "checkpoint-swap-center identity-noop: byte-identical checkpoint validated"
          << (options.dry_run ? " (dry-run)\n" : "\n");
      return 0;
    }

    core::PolarTierTruncation truncation;
    const core::PolarTierLayout truncated =
        core::make_polar_tier_layout_truncated(
            cfg.mesh,
            cfg.mesh.polar_tier_cart_cut_ring,
            &truncation);
    assert_optional_parameters(options, cfg, truncation);
    const std::size_t kept_cells =
        static_cast<std::size_t>(truncated.n_cells);
    const std::size_t kept_nodes =
        static_cast<std::size_t>(truncated.n_nodes);
    const int n_c = truncation.n_theta_cut / 4;
    const int lmax = std::min(16, cfg.mesh.nz / 2);

    core::Config parent_cfg =
        make_parent_config(cfg, preflight.frozen_config);
    io::HDF5Reader reader;
    hydro::axis_core_set_released_units(0);
    io::CheckpointData checkpoint_a =
        reader.read_checkpoint(parent_cfg,
                               preflight.resolved_path.string());
    hydro::axis_core_set_released_units(0);
    if (cfg.numerics.materials.per_material_conservation_enabled &&
        checkpoint_a.per_material_checkpoint_status !=
            io::PerMaterialCheckpointReadStatus::PresentEnabled) {
      throw std::runtime_error(
          "checkpoint-swap-center per-material segment-2 deck requires complete hydro/per_material/v1 state in checkpoint A");
    }

    const CheckpointSwapCenterGateMeasurements a_side =
        measure_checkpoint_swap_a_side(checkpoint_a.state,
                                       cfg,
                                       truncated,
                                       truncation,
                                       lmax,
                                       options.thresholds);
    try {
      assert_checkpoint_swap_a_side_gates(
          a_side, options.thresholds);
    } catch (...) {
      if (a_side.max_seam_displacement_cm <=
          options.thresholds.seam_displacement_floor_cm) {
        (void)replaced_region_mirror_margin(checkpoint_a.state,
                                            kept_cells,
                                            "A",
                                            options.thresholds,
                                            true,
                                            false);
      }
      throw;
    }

    core::State checkpoint_b = build_fresh_target(
        cfg, runtime.builder(), checkpoint_a.state, kept_nodes);
    const double fresh_v_prenorm_max_cm_s =
        normalize_fresh_tail_velocities(checkpoint_b, kept_nodes);
    std::cout << std::setprecision(17)
              << "checkpoint-swap-center fresh velocity normalization: "
              << "N_kept=" << kept_nodes
              << " N_B=" << checkpoint_b.v_r.size()
              << " fresh_v_prenorm_max_cm_s="
              << fresh_v_prenorm_max_cm_s << '\n';
    copy_restart_node_prefix(
        checkpoint_a.state, checkpoint_b, kept_nodes);
    if (!checkpoint_a.state.mesh.topo.multiblock.has_value() ||
        !checkpoint_b.mesh.topo.multiblock.has_value()) {
      throw std::runtime_error(
          "checkpoint-swap-center requires parent and target v3 multiblock topology");
    }
    const auto parent_nverts =
        effective_cell_nverts(checkpoint_a.state.mesh);
    const auto target_nverts = effective_cell_nverts(checkpoint_b.mesh);
    const auto& parent_mb = *checkpoint_a.state.mesh.topo.multiblock;
    const auto& target_mb = *checkpoint_b.mesh.topo.multiblock;
    // face_adj is intentionally excluded: bridge attachment changes the
    // neighbor across the seam. State attachment is guarded by the cell-node
    // CSR and cell_nverts authority only.
    assert_checkpoint_swap_topology_prefix(
        parent_mb.cell_node_csr_offsets,
        parent_mb.cell_node_csr_indices,
        parent_nverts,
        target_mb.cell_node_csr_offsets,
        target_mb.cell_node_csr_indices,
        target_nverts,
        kept_cells);

    const CheckpointSwapCenterGateMeasurements measured =
        measure_checkpoint_swap_b_side(checkpoint_a.state,
                                       checkpoint_b,
                                       kept_cells,
                                       kept_cells,
                                       a_side,
                                       options.thresholds);
    if (!checkpoint_swap_gates_pass(measured, options.thresholds)) {
      (void)replaced_region_mirror_margin(checkpoint_b,
                                          kept_cells,
                                          "B",
                                          options.thresholds,
                                          true,
                                          true);
      throw std::runtime_error(
          format_gate_failure(measured, options.thresholds));
    }

    const auto dropped_totals =
        replacement_totals(checkpoint_a.state, kept_cells);
    const auto fresh_totals = replacement_totals(checkpoint_b, kept_cells);
    const auto conservation = compare_checkpoint_swap_conservation(
        dropped_totals,
        fresh_totals,
        options.thresholds.ledger_kappa,
        options.thresholds.conservation_v_floor_cm_s);
    // The two regions tessellate the same seam-bounded polygon. At uniform
    // rho0/T0 their totals must agree to summation roundoff; a failure is a
    // construction or transfer defect and must never be absorbed into a
    // cumulative energy-ledger baseline.
    if (!conservation.passed) {
      throw std::runtime_error(format_conservation_failure(conservation));
    }

    copy_restart_cell_prefix(
        checkpoint_a.state,
        checkpoint_b,
        kept_cells,
        cfg.materials.materials.size(),
        cfg.numerics.materials.per_material_conservation_enabled);
    initialize_and_copy_restart_burn_state(
        checkpoint_a.state, checkpoint_b, cfg, kept_cells);
    copy_restart_ledgers(checkpoint_a.state, checkpoint_b, cfg);
    if (checkpoint_a.state.reference_epoch ==
        std::numeric_limits<std::uint64_t>::max()) {
      throw std::runtime_error(
          "checkpoint-swap-center ALE reference_epoch overflow");
    }
    checkpoint_b.reference_epoch = checkpoint_a.state.reference_epoch + 1U;
    checkpoint_b.mesh.recompute_geometry();

    std::cout << std::setprecision(17)
              << "checkpoint-swap-center gates passed: C=" << kept_cells
              << " N=" << kept_nodes << " cut_ring="
              << truncation.cut_ring << " r_cut_cm=" << truncation.r_cut
              << " n_c=" << n_c << " n_b="
              << cfg.mesh.multiblock_cart_core_bridge_layers
              << " max_speed=" << measured.max_speed_cm_s
              << " max_mach=" << measured.max_mach
              << " max_rho_rel=" << measured.max_rho_relative
              << " max_T_rel=" << measured.max_temperature_relative
              << " seam_displacement_cm="
              << measured.max_seam_displacement_cm
              << " mirror_rel=" << measured.max_mirror_relative
              << " Legendre_rel=" << measured.max_legendre_relative
              << " dM=" << conservation.delta_mass
              << " dE=" << conservation.delta_energy
              << " dPr=" << conservation.delta_momentum_r
              << " dPz=" << conservation.delta_momentum_z << '\n';
    if (options.dry_run) {
      return 0;
    }

    AuditRecord audit;
    audit.input_file_hash = sha256_file(preflight.resolved_path);
    audit.segment2_config_hash =
        sha256_bytes(cfg.meta.frozen_config_json);
    audit.center_kind = cfg.mesh.polar_tier_center_kind;
    audit.cut_ring = truncation.cut_ring;
    audit.kept_cells = static_cast<std::int64_t>(kept_cells);
    audit.kept_nodes = static_cast<std::int64_t>(kept_nodes);
    audit.r_cut_cm = truncation.r_cut;
    audit.n_c = n_c;
    audit.n_b = cfg.mesh.multiblock_cart_core_bridge_layers;
    audit.conservation = conservation;
    audit.thresholds = options.thresholds;
    audit.measured = measured;
    audit.fresh_v_prenorm_max_cm_s = fresh_v_prenorm_max_cm_s;
    audit.axis_core_released_units_a =
        preflight.axis_core_released_units;
    audit.transitions = collect_config_transitions(parent_cfg, cfg);

    const std::filesystem::path output_parent =
        output.parent_path().empty() ? std::filesystem::path(".")
                                     : output.parent_path();
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string stage_case =
        ".tenryu_epoch_swap_stage_" + std::to_string(stamp);
    const std::filesystem::path staging_checkpoint =
        output_parent / (stage_case + "_ckpt_0000.h5");
    if (std::filesystem::exists(staging_checkpoint)) {
      throw std::runtime_error(
          "checkpoint-swap-center staging path collision: " +
          staging_checkpoint.string());
    }

    try {
      hydro::axis_core_set_released_units(0);
      io::HDF5Writer writer;
      radiation::PhotonPool empty_photon_pool;
      writer.write_checkpoint(checkpoint_b,
                              cfg,
                              empty_photon_pool,
                              0,
                              checkpoint_b.step,
                              checkpoint_b.t,
                              output_parent.string(),
                              stage_case);
      append_audit_group(staging_checkpoint, audit);

      hydro::axis_core_set_released_units(0);
      io::CheckpointData verified =
          reader.read_checkpoint(cfg, staging_checkpoint.string());
      hydro::axis_core_set_released_units(0);
      const CheckpointPreflight verified_preflight =
          read_checkpoint_preflight(staging_checkpoint.string());
      if (verified.state.rho.size() != checkpoint_b.rho.size() ||
          verified.state.x_r.size() != checkpoint_b.x_r.size() ||
          verified.state.step != checkpoint_b.step ||
          verified.state.reference_epoch != checkpoint_b.reference_epoch ||
          verified.state.ale_last_applied_step != checkpoint_b.step ||
          verified.state.axis_margin_initial != -1.0 ||
          !verified.state.axis_mass_initial.empty() ||
          !verified.state.axis_inflow_budget.empty() ||
          verified_preflight.axis_core_released_units != 0 ||
          !verified.state.mesh.topo.multiblock.has_value() ||
          (cfg.numerics.materials.per_material_conservation_enabled &&
           verified.per_material_checkpoint_status !=
               io::PerMaterialCheckpointReadStatus::PresentEnabled) ||
          !checkpoint_has_audit_group(staging_checkpoint)) {
        throw std::runtime_error(
            "checkpoint-swap-center self-read verification failed");
      }
      const auto& verified_mb =
          *verified.state.mesh.topo.multiblock;
      assert_checkpoint_swap_topology_prefix(
          parent_mb.cell_node_csr_offsets,
          parent_mb.cell_node_csr_indices,
          parent_nverts,
          verified_mb.cell_node_csr_offsets,
          verified_mb.cell_node_csr_indices,
          effective_cell_nverts(verified.state.mesh),
          kept_cells);
      atomic_publish(staging_checkpoint, output);
    } catch (...) {
      hydro::axis_core_set_released_units(0);
      std::error_code cleanup_error;
      std::filesystem::remove(staging_checkpoint, cleanup_error);
      std::filesystem::remove(staging_checkpoint.string() + ".tmp",
                              cleanup_error);
      throw;
    }

    std::cout << "checkpoint-swap-center wrote restartable checkpoint B: "
              << output.string() << '\n';
    return 0;
  } catch (const std::exception& error) {
    hydro::axis_core_set_released_units(0);
    std::cerr << "TENRYU ERROR [checkpoint-swap-center]: " << error.what()
              << '\n';
    return 1;
  }
#else
  (void)options;
  core::log_error(
      "checkpoint-swap-center requires TENRYU_ENABLE_PYTHON=ON and TENRYU_ENABLE_HDF5=ON");
  return 1;
#endif
}

}  // namespace tenryu::drivers
