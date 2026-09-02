#include "io/hdf5_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "core/config_validate.hpp"
#include "core/error.hpp"
#include "core/namelist/errors.hpp"
#include "core/namelist/freeze.hpp"
#include "hydro/axis_edge_collapse.hpp"
#include "hydro/euler_window_blend.hpp"
#include "hydro/evacuated_cell_shadow.hpp"
#include "hydro/oriented_swept_volume.cuh"
#include "hydro/pole_axis_bbsw.hpp"
#include "io/frozen_config_probe.hpp"
#include "mesh/mesh.hpp"

#if TENRYU_ENABLE_HDF5
#include "io/hdf5_utils.hpp"
#endif

namespace tenryu::io {
namespace {

constexpr int kSchemaVersion = 1;
constexpr const char* kMeshTopologyV2Path = "mesh/topology/v2";
constexpr const char* kMeshTopologyV3Path = "mesh/topology/v3";

#if TENRYU_ENABLE_HDF5
bool pole_axis_bbsw_enabled_for_restart(const core::Config& cfg) {
  static const char* const env =
      std::getenv(hydro::pole_axis_bbsw::kEnvEnable);
  static const bool env_present = env != nullptr && env[0] != '\0';
  static const bool env_enabled =
      env_present && std::string(env) != "0";
  return env_present ? env_enabled
                     : cfg.numerics.ale.pole_axis_bbsw_enabled;
}
#endif

std::string format_rank(const int rank) {
  std::ostringstream oss;
  oss << std::setw(4) << std::setfill('0') << rank;
  return oss.str();
}

void cuda_check(const cudaError_t err, const std::string_view message) {
  TENRYU_ASSERT(err == cudaSuccess, std::string(message));
}

const char* topology_scheme_name(const core::TopologyScheme scheme) {
  switch (scheme) {
    case core::TopologyScheme::SINGLE_BLOCK:
      return "single_block";
    case core::TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL:
      return "multiblock_cart_core_polar_shell";
    case core::TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_5BLOCK:
      return "multiblock_half_butterfly_5block";
    case core::TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_TRIFAN_CAP_5BLOCK:
      return "multiblock_half_butterfly_trifan_cap_5block";
    case core::TopologyScheme::MULTIBLOCK_POLAR_TIER:
      return "multiblock_polar_tier";
    case core::TopologyScheme::MULTIBLOCK_POLAR_TIER_CART_CENTER:
      return "multiblock_polar_tier_cart_center";
    case core::TopologyScheme::CONE_SHELL_SPINE:
      return "cone_shell_spine";
    case core::TopologyScheme::PENTAGON_BELT_SHELL:
      return "pentagon_belt_shell";
  }
  return "unknown";
}

int expected_topology_v2_block_count(
    const core::Config::MeshConfig& mesh_cfg) {
  switch (mesh_cfg.topology_scheme) {
    case core::TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL:
      return 3;
    case core::TopologyScheme::CONE_SHELL_SPINE:
      return mesh::kConeShellSpineBlockCount;
    case core::TopologyScheme::PENTAGON_BELT_SHELL:
      return mesh::mesh_topo_checked_int_count(
          2LL * static_cast<long long>(mesh_cfg.pentagon_belt_layers.size()) +
              1LL,
          "pentagon-belt topology block count");
    case core::TopologyScheme::SINGLE_BLOCK:
    case core::TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_5BLOCK:
    case core::TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_TRIFAN_CAP_5BLOCK:
    case core::TopologyScheme::MULTIBLOCK_POLAR_TIER:
    case core::TopologyScheme::MULTIBLOCK_POLAR_TIER_CART_CENTER:
      TENRYU_ASSERT(false,
                    "ConfigError: checkpoint /mesh/topology/v2 is not "
                    "supported for topology_scheme=" +
                        std::string(
                            topology_scheme_name(mesh_cfg.topology_scheme)));
      return -1;
  }
  TENRYU_ASSERT(false,
                "ConfigError: checkpoint /mesh/topology/v2 has an unknown "
                "topology scheme");
  return -1;
}

int expected_topology_v3_block_count(
    const core::Config::MeshConfig& mesh_cfg) {
  switch (mesh_cfg.topology_scheme) {
    case core::TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_5BLOCK:
    case core::TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_TRIFAN_CAP_5BLOCK:
      return 5;
    case core::TopologyScheme::MULTIBLOCK_POLAR_TIER:
      return core::make_polar_tier_layout(mesh_cfg).block_count;
    case core::TopologyScheme::MULTIBLOCK_POLAR_TIER_CART_CENTER:
      return core::make_polar_tier_layout_truncated(
                 mesh_cfg, mesh_cfg.polar_tier_cart_cut_ring, nullptr)
                 .block_count +
             2;
    case core::TopologyScheme::SINGLE_BLOCK:
    case core::TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL:
    case core::TopologyScheme::CONE_SHELL_SPINE:
      TENRYU_ASSERT(
          false,
          "ConfigError: checkpoint /mesh/topology/v3 is not supported for "
          "topology_scheme=" +
              std::string(topology_scheme_name(mesh_cfg.topology_scheme)));
      return -1;
  }
  TENRYU_ASSERT(false,
                "ConfigError: checkpoint /mesh/topology/v3 has an unknown "
                "topology scheme");
  return -1;
}

std::size_t checked_mul_size(const std::size_t a,
                             const std::size_t b,
                             const std::string& context) {
  TENRYU_ASSERT(b == 0 || a <= std::numeric_limits<std::size_t>::max() / b,
                "HDF5 size overflow while computing " + context);
  return a * b;
}

std::size_t checked_i64_to_size(const std::int64_t value, const std::string& context) {
  TENRYU_ASSERT(value >= 0, context + " must be non-negative");
  const auto uvalue = static_cast<std::uint64_t>(value);
  TENRYU_ASSERT(uvalue <= std::numeric_limits<std::size_t>::max(),
                context + " exceeds size_t range");
  return static_cast<std::size_t>(uvalue);
}

int checked_size_to_int(const std::size_t value, const std::string& context) {
  TENRYU_ASSERT(value <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                context + " exceeds INT_MAX");
  return static_cast<int>(value);
}

template <typename T>
std::size_t checked_bytes_for_count(const std::size_t count, const std::string& context) {
  return checked_mul_size(count, sizeof(T), context);
}

void initialize_output_timing(core::State& state, const core::Config& cfg) {
  state.t_next_plot = (cfg.output.plot_every_s > 0.0) ? (state.t + cfg.output.plot_every_s) : -1.0;
  state.t_next_history =
      (cfg.output.history_every_s > 0.0) ? (state.t + cfg.output.history_every_s) : -1.0;
  state.t_next_checkpoint =
      (cfg.output.checkpoint_every_s > 0.0) ? (state.t + cfg.output.checkpoint_every_s) : -1.0;
}

std::string checkpoint_config_signature(const core::Config& cfg) {
  const int dim = (cfg.main.dimension == "2D_RZ") ? 2 : 1;
  const int nr = cfg.mesh.nr;
  const int nz = (dim == 2) ? cfg.mesh.nz : 1;
  const std::int64_t n_cells =
      static_cast<std::int64_t>(nr) * static_cast<std::int64_t>(nz);
  const std::int64_t n_nodes =
      static_cast<std::int64_t>(nr + 1) *
      static_cast<std::int64_t>((dim == 2) ? (nz + 1) : 1);

  std::ostringstream oss;
  oss << "dimension=" << cfg.main.dimension;
  oss << ";temperature_model=" << cfg.main.temperature_model;
  oss << ";nr=" << nr;
  oss << ";nz=" << nz;
  oss << ";n_cells=" << n_cells;
  oss << ";n_nodes=" << n_nodes;
  oss << ";n_groups=" << std::max(cfg.radiation.groups, 1);
  oss << ";n_materials=" << cfg.materials.materials.size();
  oss << ";radiation_enabled=" << (cfg.radiation.enabled ? 1 : 0);
  oss << ";ddmc_enabled=" << (cfg.radiation.ddmc.enabled ? 1 : 0);
  oss << ";laser_enabled=" << (cfg.laser.enabled ? 1 : 0);
  oss << ";hydro_enabled=" << (cfg.numerics.hydro.enabled ? 1 : 0);
  oss << ";compatible_energy=" << (cfg.numerics.hydro.compatible_energy ? 1 : 0);
  oss << ";conduction_enabled=" << (cfg.numerics.conduction.enabled ? 1 : 0);
  oss << ";av_type=" << cfg.numerics.hydro.av_type;
  oss << ";av_heat_to=" << cfg.numerics.hydro.av_heat_to;
  oss << ";hk_velocity_damper_C=" << cfg.numerics.hydro.hk_velocity_damper_C;
  oss << ";hk_velocity_damper_tau_min=" << cfg.numerics.hydro.hk_velocity_damper_tau_min;
  oss << ";hk_velocity_damper_grad_Te_max="
      << cfg.numerics.hydro.hk_velocity_damper_grad_Te_max;
  oss << ";hk_velocity_damper_grad_rho_max="
      << cfg.numerics.hydro.hk_velocity_damper_grad_rho_max;
  oss << ";hk_velocity_damper_guard_cells="
      << cfg.numerics.hydro.hk_velocity_damper_guard_cells;
  oss << ";ion_art_heat_C=" << cfg.numerics.hydro.ion_art_heat_C;
  oss << ";two_temperature=1";
  oss << ";seed=" << cfg.main.seed;
  oss << ";output_format=" << cfg.output.format;
  oss << ";output_compression=" << cfg.output.compression;
  return oss.str();
}

std::string trim_ascii_whitespace(std::string text) {
  auto is_space = [](const char ch) {
    return ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t' || ch == '\f' ||
           ch == '\v';
  };
  while (!text.empty() && is_space(text.front())) {
    text.erase(text.begin());
  }
  while (!text.empty() && is_space(text.back())) {
    text.pop_back();
  }
  return text;
}

bool looks_like_json_object(const std::string& text) {
  const std::string trimmed = trim_ascii_whitespace(text);
  return !trimmed.empty() && trimmed.front() == '{';
}

std::string abbreviated_preview(const std::string& text, const std::size_t max_chars) {
  if (text.size() <= max_chars) {
    return text;
  }
  return text.substr(0, max_chars) + "...";
}

std::vector<double> expected_group_bounds_from_config(const core::Config& cfg) {
  if (!cfg.radiation.group_bounds_eV.empty()) {
    return cfg.radiation.group_bounds_eV;
  }
  const int groups = std::max(cfg.radiation.groups, 1);
  std::vector<double> bounds(static_cast<std::size_t>(groups + 1), 0.0);

  double e_min = 1.0e-2;
  double e_max = 1.0e2;
  if (cfg.radiation.compute_T_range_eV.size() == 2) {
    const double cfg_min = cfg.radiation.compute_T_range_eV[0];
    const double cfg_max = cfg.radiation.compute_T_range_eV[1];
    if (cfg_min > 0.0 && cfg_max > cfg_min && std::isfinite(cfg_min) &&
        std::isfinite(cfg_max)) {
      e_min = cfg_min;
      e_max = cfg_max;
    }
  }

  if (groups == 1) {
    bounds[0] = e_min;
    bounds[1] = e_max;
    return bounds;
  }
  const double ln_min = std::log(e_min);
  const double ln_max = std::log(e_max);
  for (int g = 0; g <= groups; ++g) {
    const double u = static_cast<double>(g) / static_cast<double>(groups);
    bounds[static_cast<std::size_t>(g)] = std::exp((1.0 - u) * ln_min + u * ln_max);
  }
  bounds.back() = e_max;
  return bounds;
}

std::vector<std::uint64_t> eos_signatures_from_config(const core::Config& cfg) {
  std::vector<std::uint64_t> signatures;
  signatures.reserve(cfg.materials.materials.size());
  for (const auto& mat : cfg.materials.materials) {
    signatures.push_back(mat.eos_signature);
  }
  return signatures;
}

std::string format_hash_hex(const std::uint64_t value) {
  std::ostringstream oss;
  oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
  return oss.str();
}

std::filesystem::path resolve_checkpoint_path(const std::string& checkpoint_prefix,
                                              const int rank) {
  const std::filesystem::path prefix(checkpoint_prefix);
  if (prefix.extension() == ".h5" && std::filesystem::exists(prefix)) {
    return prefix;
  }
  if (prefix.extension().empty()) {
    const std::filesystem::path direct = checkpoint_prefix + ".h5";
    if (std::filesystem::exists(direct)) {
      return direct;
    }
  }

  const int safe_rank = std::max(rank, 0);
  const std::filesystem::path ranked =
      std::filesystem::path(checkpoint_prefix + std::string("_r") +
                            format_rank(safe_rank) + ".h5");
  if (std::filesystem::exists(ranked)) {
    return ranked;
  }

  const std::filesystem::path dir = prefix.parent_path().empty() ? std::filesystem::path(".")
                                                                  : prefix.parent_path();
  const std::string stem = prefix.filename().string();

  std::vector<std::filesystem::path> candidates;
  if (std::filesystem::exists(dir)) {
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      const auto& p = entry.path();
      const std::string filename = p.filename().string();
      if (filename.rfind(stem + "_r", 0) != 0) {
        continue;
      }
      if (p.extension() != ".h5") {
        continue;
      }
      candidates.push_back(p);
    }
  }

  std::sort(candidates.begin(), candidates.end());
  TENRYU_ASSERT(!candidates.empty(),
                "checkpoint files not found: " + checkpoint_prefix + "_r*.h5");
  if (candidates.size() > 1) {
    core::log_warning("Multiple checkpoint files matched prefix '" + checkpoint_prefix +
                      "'; using '" + candidates.front().string() +
                      "'. Please pass an exact checkpoint prefix to avoid ambiguity.");
  }
  return candidates.front();
}

#if TENRYU_ENABLE_HDF5

void warn_h5_close_failure(const herr_t status, const char* op, const char* context) {
  if (status < 0) {
    core::log_warning(std::string("[WARN] ") + op + " failed in " + context);
  }
}

bool link_exists(const hid_t file, const std::string& path) {
  return tenryu::io::h5_link_exists(file, path.c_str(), H5P_DEFAULT) > 0;
}

std::string swept_volume_convention_value(const std::uint8_t value) {
  switch (static_cast<hydro::SweptVolumeConvention>(value)) {
    case hydro::SweptVolumeConvention::LegacyRawV0:
      return "LegacyRawV0 (0)";
    case hydro::SweptVolumeConvention::OrientedLowToHighV1:
      return "OrientedLowToHighV1 (1)";
  }
  return "unknown (" + std::to_string(value) + ")";
}

void validate_swept_volume_convention(
    const char* operator_name,
    const std::uint8_t restart_value,
    const hydro::SweptVolumeConvention config_value) {
  const std::uint8_t config_raw = static_cast<std::uint8_t>(config_value);
  TENRYU_ASSERT(
      restart_value == config_raw,
      "ConfigError: swept-volume contract mismatch for operator " +
          std::string(operator_name) + " (restart=" +
          swept_volume_convention_value(restart_value) + ", config=" +
          swept_volume_convention_value(config_raw) +
          "); restart contract is authoritative; adjust the namelist or fork the lineage");
}

void validate_swept_volume_contract(
    const std::uint8_t plain_csr,
    const std::uint8_t conservative_csr,
    const std::uint8_t option_b,
    const std::uint8_t axis_band,
    const std::uint8_t plic,
    const hydro::SweptVolumeResolvedContract& config_contract) {
  validate_swept_volume_convention(
      "plain_csr", plain_csr, config_contract.plain_csr);
  validate_swept_volume_convention(
      "conservative_csr", conservative_csr, config_contract.conservative_csr);
  validate_swept_volume_convention(
      "option_b", option_b, config_contract.option_b);
  validate_swept_volume_convention(
      "axis_band", axis_band, config_contract.axis_band);
  validate_swept_volume_convention("plic", plic, config_contract.plic);
}

std::optional<std::int32_t> read_root_attr_i32(const hid_t file, const std::string& name) {
  if (H5Aexists(file, name.c_str()) <= 0) {
    return std::nullopt;
  }
  const hid_t attr = H5Aopen(file, name.c_str(), H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HDF5 failed to open root attribute: " + name);
  std::int32_t value = 0;
  TENRYU_ASSERT(H5Aread(attr, H5T_NATIVE_INT32, &value) >= 0,
                "HDF5 failed to read i32 root attribute: " + name);
  H5Aclose(attr);
  return value;
}

std::optional<std::int64_t> read_root_attr_i64(const hid_t file, const std::string& name) {
  if (H5Aexists(file, name.c_str()) <= 0) {
    return std::nullopt;
  }
  const hid_t attr = H5Aopen(file, name.c_str(), H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HDF5 failed to open root attribute: " + name);
  std::int64_t value = 0;
  TENRYU_ASSERT(H5Aread(attr, H5T_NATIVE_INT64, &value) >= 0,
                "HDF5 failed to read i64 root attribute: " + name);
  H5Aclose(attr);
  return value;
}

std::optional<std::int64_t> read_group_attr_i64(const hid_t file,
                                                const std::string& group_path,
                                                const std::string& name) {
  if (!link_exists(file, group_path)) {
    return std::nullopt;
  }

  const hid_t group = H5Gopen2(file, group_path.c_str(), H5P_DEFAULT);
  TENRYU_ASSERT(group >= 0, "HDF5 failed to open group for attribute read: " + group_path);
  if (H5Aexists(group, name.c_str()) <= 0) {
    warn_h5_close_failure(H5Gclose(group), "H5Gclose",
                          "HDF5Reader::read_group_attr_i64(group missing attr)");
    return std::nullopt;
  }

  const hid_t attr = H5Aopen(group, name.c_str(), H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HDF5 failed to open group attribute: " + group_path + "/" + name);
  std::int64_t value = 0;
  TENRYU_ASSERT(H5Aread(attr, H5T_NATIVE_INT64, &value) >= 0,
                "HDF5 failed to read i64 group attribute: " + group_path + "/" + name);
  H5Aclose(attr);
  warn_h5_close_failure(H5Gclose(group), "H5Gclose",
                        "HDF5Reader::read_group_attr_i64(group close)");
  return value;
}

std::optional<std::int8_t> read_group_attr_i8(const hid_t file,
                                              const std::string& group_path,
                                              const std::string& name) {
  if (!link_exists(file, group_path)) {
    return std::nullopt;
  }

  const hid_t group = H5Gopen2(file, group_path.c_str(), H5P_DEFAULT);
  TENRYU_ASSERT(group >= 0,
                "HDF5 failed to open group for attribute read: " +
                    group_path);
  if (H5Aexists(group, name.c_str()) <= 0) {
    warn_h5_close_failure(
        H5Gclose(group), "H5Gclose",
        "HDF5Reader::read_group_attr_i8(group missing attr)");
    return std::nullopt;
  }

  const hid_t attr = H5Aopen(group, name.c_str(), H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0,
                "HDF5 failed to open group attribute: " + group_path + "/" +
                    name);
  std::int8_t value = 0;
  TENRYU_ASSERT(H5Aread(attr, H5T_NATIVE_INT8, &value) >= 0,
                "HDF5 failed to read i8 group attribute: " + group_path +
                    "/" + name);
  H5Aclose(attr);
  warn_h5_close_failure(H5Gclose(group), "H5Gclose",
                        "HDF5Reader::read_group_attr_i8(group close)");
  return value;
}

int checkpoint_corner_stride(const hid_t file) {
  if (!link_exists(file, "mesh/topology/v4")) {
    return mesh::kMeshTopoCellStorageSlots;
  }
  const auto corner_stride =
      read_group_attr_i64(file, "mesh/topology/v4", "corner_stride");
  TENRYU_ASSERT(
      corner_stride.has_value(),
      "Restart checkpoint missing attribute: mesh/topology/v4/corner_stride");
  TENRYU_ASSERT(
      *corner_stride > 0 &&
          *corner_stride <=
              static_cast<std::int64_t>(std::numeric_limits<int>::max()),
      "Restart checkpoint mesh/topology/v4 corner_stride must be a positive int");
  return static_cast<int>(*corner_stride);
}

std::optional<std::uint8_t> read_group_attr_u8(const hid_t file,
                                               const std::string& group_path,
                                               const std::string& name) {
  if (!link_exists(file, group_path)) {
    return std::nullopt;
  }

  const hid_t group = H5Gopen2(file, group_path.c_str(), H5P_DEFAULT);
  TENRYU_ASSERT(group >= 0, "HDF5 failed to open group for attribute read: " + group_path);
  if (H5Aexists(group, name.c_str()) <= 0) {
    warn_h5_close_failure(H5Gclose(group), "H5Gclose",
                          "HDF5Reader::read_group_attr_u8(group missing attr)");
    return std::nullopt;
  }

  const hid_t attr = H5Aopen(group, name.c_str(), H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HDF5 failed to open group attribute: " + group_path + "/" + name);
  std::uint8_t value = 0;
  TENRYU_ASSERT(H5Aread(attr, H5T_NATIVE_UINT8, &value) >= 0,
                "HDF5 failed to read u8 group attribute: " + group_path + "/" + name);
  H5Aclose(attr);
  warn_h5_close_failure(H5Gclose(group), "H5Gclose",
                        "HDF5Reader::read_group_attr_u8(group close)");
  return value;
}

std::optional<std::uint64_t> read_group_attr_u64(
    const hid_t file,
    const std::string& group_path,
    const std::string& name) {
  if (!link_exists(file, group_path)) {
    return std::nullopt;
  }

  const hid_t group = H5Gopen2(file, group_path.c_str(), H5P_DEFAULT);
  TENRYU_ASSERT(
      group >= 0,
      "HDF5 failed to open group for attribute read: " + group_path);
  if (H5Aexists(group, name.c_str()) <= 0) {
    warn_h5_close_failure(
        H5Gclose(group),
        "H5Gclose",
        "HDF5Reader::read_group_attr_u64(group missing attr)");
    return std::nullopt;
  }

  const hid_t attr = H5Aopen(group, name.c_str(), H5P_DEFAULT);
  TENRYU_ASSERT(
      attr >= 0,
      "HDF5 failed to open group attribute: " + group_path + "/" + name);
  std::uint64_t value = 0;
  TENRYU_ASSERT(
      H5Aread(attr, H5T_NATIVE_UINT64, &value) >= 0,
      "HDF5 failed to read u64 group attribute: " + group_path + "/" +
          name);
  H5Aclose(attr);
  warn_h5_close_failure(
      H5Gclose(group),
      "H5Gclose",
      "HDF5Reader::read_group_attr_u64(group close)");
  return value;
}

std::optional<std::string> read_root_attr_string(const hid_t file, const std::string& name) {
  if (H5Aexists(file, name.c_str()) <= 0) {
    return std::nullopt;
  }
  const hid_t attr = H5Aopen(file, name.c_str(), H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HDF5 failed to open root string attribute: " + name);
  const hid_t type = H5Aget_type(attr);
  TENRYU_ASSERT(type >= 0, "HDF5 failed to get root string attribute type: " + name);
  const std::size_t size = H5Tget_size(type);
  std::string value(size, '\0');
  TENRYU_ASSERT(H5Aread(attr, type, value.data()) >= 0,
                "HDF5 failed to read root string attribute: " + name);
  while (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  warn_h5_close_failure(H5Tclose(type), "H5Tclose",
                        "HDF5Reader::read_root_attr_string(type)");
  H5Aclose(attr);
  return value;
}

std::size_t dataset_size(const hid_t file, const std::string& path) {
  const hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  TENRYU_ASSERT(dset >= 0, "HDF5 failed to open dataset: " + path);
  const hid_t space = H5Dget_space(dset);
  TENRYU_ASSERT(space >= 0, "HDF5 failed to get dataspace: " + path);

  const int rank = H5Sget_simple_extent_ndims(space);
  TENRYU_ASSERT(rank >= 0, "HDF5 failed to get dataset rank: " + path);
  std::size_t total = 1;
  if (rank > 0) {
    std::vector<hsize_t> dims(static_cast<std::size_t>(rank), 0);
    const int got_dims = H5Sget_simple_extent_dims(space, dims.data(), nullptr);
    TENRYU_ASSERT(got_dims == rank, "HDF5 failed to get dataset dimensions: " + path);
    for (const hsize_t d : dims) {
      const std::size_t dim = static_cast<std::size_t>(d);
      TENRYU_ASSERT(static_cast<hsize_t>(dim) == d,
                    "HDF5 dataset dimension exceeds size_t range: " + path);
      total = checked_mul_size(total, dim, "dataset element count for " + path);
    }
  }
  warn_h5_close_failure(H5Sclose(space), "H5Sclose", "HDF5Reader::dataset_size(space)");
  warn_h5_close_failure(H5Dclose(dset), "H5Dclose", "HDF5Reader::dataset_size(dataset)");
  return total;
}

std::string read_string_dataset_checked(const hid_t file, const std::string& path) {
  TENRYU_ASSERT(link_exists(file, path), "Restart checkpoint missing dataset: " + path);
  const hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  TENRYU_ASSERT(dset >= 0, "HDF5 failed to open string dataset: " + path);
  const hid_t type = H5Dget_type(dset);
  TENRYU_ASSERT(type >= 0, "HDF5 failed to get string dataset type: " + path);
  const std::size_t size = H5Tget_size(type);
  TENRYU_ASSERT(size > 0, "HDF5 string dataset has invalid size: " + path);
  std::string value(size, '\0');
  TENRYU_ASSERT(H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.data()) >= 0,
                "HDF5 failed to read string dataset: " + path);
  while (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  warn_h5_close_failure(H5Tclose(type), "H5Tclose",
                        "HDF5Reader::read_string_dataset_checked(type)");
  warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                        "HDF5Reader::read_string_dataset_checked(dataset)");
  return value;
}

template <typename T>
T read_scalar_dataset(const hid_t file,
                      const std::string& path,
                      const hid_t type,
                      const T default_value) {
  if (!link_exists(file, path)) {
    return default_value;
  }
  const hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  TENRYU_ASSERT(dset >= 0, "HDF5 failed to open scalar dataset: " + path);
  T value = default_value;
  TENRYU_ASSERT(H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value) >= 0,
                "HDF5 failed to read scalar dataset: " + path);
  warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                        "HDF5Reader::read_scalar_dataset(dataset)");
  return value;
}

template <typename T>
std::vector<T> read_vector_dataset(const hid_t file,
                                   const std::string& path,
                                   const hid_t type) {
  const hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  TENRYU_ASSERT(dset >= 0, "HDF5 failed to open dataset: " + path);
  const hid_t space = H5Dget_space(dset);
  TENRYU_ASSERT(space >= 0, "HDF5 failed to get dataspace: " + path);

  const int rank = H5Sget_simple_extent_ndims(space);
  TENRYU_ASSERT(rank >= 0, "HDF5 failed to get dataset rank: " + path);
  std::size_t total = 1;
  if (rank > 0) {
    std::vector<hsize_t> dims(static_cast<std::size_t>(rank), 0);
    const int got_dims = H5Sget_simple_extent_dims(space, dims.data(), nullptr);
    TENRYU_ASSERT(got_dims == rank, "HDF5 failed to get dataset dimensions: " + path);
    for (const hsize_t d : dims) {
      const std::size_t dim = static_cast<std::size_t>(d);
      TENRYU_ASSERT(static_cast<hsize_t>(dim) == d,
                    "HDF5 dataset dimension exceeds size_t range: " + path);
      total = checked_mul_size(total, dim, "dataset element count for " + path);
    }
  }

  std::vector<T> values(total);
  if (!values.empty()) {
    TENRYU_ASSERT(H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0,
                  "HDF5 failed to read dataset: " + path);
  }

  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HDF5Reader::read_vector_dataset(space)");
  warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                        "HDF5Reader::read_vector_dataset(dataset)");
  return values;
}

template <typename T>
std::vector<T> read_vector_dataset_checked(const hid_t file,
                                           const std::string& path,
                                           const hid_t type,
                                           const std::size_t expected_size) {
  TENRYU_ASSERT(link_exists(file, path), "Restart checkpoint missing dataset: " + path);
  const std::size_t actual_size = dataset_size(file, path);
  TENRYU_ASSERT(actual_size == expected_size,
                "Restart checkpoint dataset size mismatch for " + path +
                    " (expected " + std::to_string(expected_size) +
                    ", got " + std::to_string(actual_size) + ")");
  auto values = read_vector_dataset<T>(file, path, type);
  values.resize(expected_size);
  return values;
}

template <typename Tag>
void copy_vector_to_field(const std::vector<double>& src,
                          core::Field1D<Tag>& dst,
                          const std::string_view dataset_path) {
  if (src.size() != dst.size()) {
    core::log_warning("Restart checkpoint dataset size mismatch for " +
                      std::string(dataset_path) + " (checkpoint=" +
                      std::to_string(src.size()) + ", state=" +
                      std::to_string(dst.size()) +
                      "); truncating/padding with zeros.");
  }
  std::vector<double> resized(dst.size(), 0.0);
  const std::size_t n = std::min(resized.size(), src.size());
  std::copy_n(src.begin(), n, resized.begin());
  if (!resized.empty()) {
    dst.copy_from_host(resized.data());
  }
}

template <typename Tag>
void copy_required_dataset_to_field(const hid_t file,
                                    const std::string& path,
                                    core::Field1D<Tag>& dst) {
  const auto values =
      read_vector_dataset_checked<double>(file, path, H5T_NATIVE_DOUBLE, dst.size());
  if (!values.empty()) {
    dst.copy_from_host(values.data());
  }
}

void load_ale_reference_checkpoint_group(const hid_t file,
                                         core::State& state) {
  state.x_r_reference.fill(0.0);
  state.x_z_reference.fill(0.0);
  state.x_r_shock_target.fill(0.0);
  state.x_z_shock_target.fill(0.0);
  state.reference_epoch = 0;

  constexpr const char* group_path = "mesh/ale_reference/v1";
  if (!link_exists(file, group_path)) {
    return;
  }

  const auto reference_epoch =
      read_group_attr_u64(file, group_path, "reference_epoch");
  TENRYU_ASSERT(
      reference_epoch.has_value(),
      "Restart checkpoint missing attribute: "
      "mesh/ale_reference/v1/reference_epoch");
  copy_required_dataset_to_field(
      file,
      "mesh/ale_reference/v1/x_r_reference",
      state.x_r_reference);
  copy_required_dataset_to_field(
      file,
      "mesh/ale_reference/v1/x_z_reference",
      state.x_z_reference);
  copy_required_dataset_to_field(
      file,
      "mesh/ale_reference/v1/x_r_shock_target",
      state.x_r_shock_target);
  copy_required_dataset_to_field(
      file,
      "mesh/ale_reference/v1/x_z_shock_target",
      state.x_z_shock_target);
  state.reference_epoch = *reference_epoch;
}

void load_carrier_checkpoint_group(const hid_t file,
                                   core::State& state) {
  constexpr const char* group_path = "mesh/carrier/v1";
  if (!link_exists(file, group_path)) {
    return;
  }

  const auto valid_attr = read_group_attr_i8(file, group_path, "valid");
  const auto domain_active_attr =
      read_group_attr_i8(file, group_path, "domain_active");
  const auto n_masters_attr =
      read_group_attr_i64(file, group_path, "n_masters");
  TENRYU_ASSERT(valid_attr.has_value(),
                "Restart checkpoint missing attribute: mesh/carrier/v1/valid");
  TENRYU_ASSERT(
      domain_active_attr.has_value(),
      "Restart checkpoint missing attribute: mesh/carrier/v1/domain_active");
  TENRYU_ASSERT(
      n_masters_attr.has_value(),
      "Restart checkpoint missing attribute: mesh/carrier/v1/n_masters");
  TENRYU_ASSERT(*valid_attr == 0 || *valid_attr == 1,
                "Restart checkpoint mesh/carrier/v1 valid must be 0 or 1");
  TENRYU_ASSERT(
      *domain_active_attr == 0 || *domain_active_attr == 1,
      "Restart checkpoint mesh/carrier/v1 domain_active must be 0 or 1");

  const std::size_t n_masters = checked_i64_to_size(
      *n_masters_attr, "Restart checkpoint mesh/carrier/v1 n_masters");
  const std::size_t n_nodes = state.x_r.size();
  const auto master_stable_id = read_vector_dataset_checked<std::uint64_t>(
      file, "mesh/carrier/v1/master_stable_id", H5T_NATIVE_UINT64,
      n_masters);
  const auto master_mesh_node = read_vector_dataset_checked<std::int32_t>(
      file, "mesh/carrier/v1/master_mesh_node", H5T_NATIVE_INT32,
      n_masters);
  const auto master_bc_class = read_vector_dataset_checked<std::uint8_t>(
      file, "mesh/carrier/v1/master_bc_class", H5T_NATIVE_UINT8,
      n_masters);
  const auto node_class = read_vector_dataset_checked<std::uint8_t>(
      file, "mesh/carrier/v1/node_class", H5T_NATIVE_UINT8, n_nodes);
  const auto node_edge = read_vector_dataset_checked<std::int32_t>(
      file, "mesh/carrier/v1/node_edge", H5T_NATIVE_INT32, n_nodes);
  const auto node_lambda = read_vector_dataset_checked<double>(
      file, "mesh/carrier/v1/node_lambda", H5T_NATIVE_DOUBLE, n_nodes);

  const bool valid = *valid_attr != 0;
  TENRYU_ASSERT(!valid || n_masters >= 3U,
                "Restart checkpoint valid carrier must have at least three masters");
  for (std::size_t master_index = 0; master_index < n_masters;
       ++master_index) {
    TENRYU_ASSERT(
        master_mesh_node[master_index] >= 0 &&
            static_cast<std::size_t>(master_mesh_node[master_index]) < n_nodes,
        "Restart checkpoint carrier master_mesh_node entry out of range");
    TENRYU_ASSERT(
        master_bc_class[master_index] <=
            static_cast<std::uint8_t>(mesh::CarrierBcClass::kAxis),
        "Restart checkpoint carrier master_bc_class entry out of range");
  }
  for (std::size_t node = 0; node < n_nodes; ++node) {
    TENRYU_ASSERT(node_class[node] <= 2U,
                  "Restart checkpoint carrier node_class entry out of range");
    TENRYU_ASSERT(node_lambda[node] >= 0.0 && node_lambda[node] <= 1.0,
                  "Restart checkpoint carrier node_lambda entry out of range");
  }

  state.boundary_carrier.masters.clear();
  state.boundary_carrier.masters.reserve(n_masters);
  for (std::size_t master_index = 0; master_index < n_masters;
       ++master_index) {
    state.boundary_carrier.masters.push_back(
        {master_stable_id[master_index],
         static_cast<int>(master_mesh_node[master_index]),
         static_cast<mesh::CarrierBcClass>(master_bc_class[master_index])});
  }
  state.boundary_carrier.valid = valid;
  state.carrier_domain_active = *domain_active_attr != 0;
  state.carrier_node_class = node_class;
  state.carrier_node_edge.assign(node_edge.begin(), node_edge.end());
  state.carrier_node_lambda = node_lambda;
  state.carrier_ring_pages = {};
  state.mesh.ring_pages = nullptr;
}

void load_evacuated_cells_checkpoint_group(const hid_t file,
                                           core::State& state) {
  constexpr const char* group_path = "evacuated_cells";
  if (!link_exists(file, group_path)) {
    // Compatibility: old checkpoints leave the state empty so the first
    // controller evaluation retains the legacy reference-capture behavior.
    return;
  }

  auto& evacuated = state.evacuated_cells;
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_nodes = state.x_r.size();
  evacuated.inactive_member_mask =
      read_vector_dataset_checked<std::uint8_t>(
          file, "evacuated_cells/inactive_member_mask", H5T_NATIVE_UINT8,
          n_cells);
  evacuated.closure_lineage = read_vector_dataset_checked<std::uint8_t>(
      file, "evacuated_cells/closure_lineage", H5T_NATIVE_UINT8, n_cells);
  evacuated.contact_active_mask =
      read_vector_dataset_checked<std::uint8_t>(
          file, "evacuated_cells/contact_active_mask", H5T_NATIVE_UINT8,
          n_cells);
  evacuated.mass_ref = read_vector_dataset_checked<double>(
      file, "evacuated_cells/mass_ref", H5T_NATIVE_DOUBLE, n_cells);
  evacuated.vol_ref = read_vector_dataset_checked<double>(
      file, "evacuated_cells/vol_ref", H5T_NATIVE_DOUBLE, n_cells);
  evacuated.x_r_ref = read_vector_dataset_checked<double>(
      file, "evacuated_cells/x_r_ref", H5T_NATIVE_DOUBLE, n_nodes);
  evacuated.x_z_ref = read_vector_dataset_checked<double>(
      file, "evacuated_cells/x_z_ref", H5T_NATIVE_DOUBLE, n_nodes);
  evacuated.controller_previous_volume =
      read_vector_dataset_checked<double>(
          file, "evacuated_cells/controller_previous_volume",
          H5T_NATIVE_DOUBLE, n_cells);
  const auto controller_previous_contact_gap =
      read_vector_dataset_checked<double>(
          file, "evacuated_cells/controller_previous_contact_gap",
          H5T_NATIVE_DOUBLE,
          checked_mul_size(n_cells, 2U,
                           "evacuated-cell previous contact gap size"));
  evacuated.controller_previous_contact_gap.resize(n_cells);
  for (std::size_t cell_index = 0; cell_index < n_cells; ++cell_index) {
    evacuated.controller_previous_contact_gap[cell_index] = {
        controller_previous_contact_gap[2U * cell_index],
        controller_previous_contact_gap[2U * cell_index + 1U]};
  }
  evacuated.off_streak = read_vector_dataset_checked<int>(
      file, "evacuated_cells/off_streak", H5T_NATIVE_INT, n_cells);
  evacuated.controller_dwell_remaining = read_vector_dataset_checked<int>(
      file, "evacuated_cells/controller_dwell_remaining", H5T_NATIVE_INT,
      n_cells);
  evacuated.controller_evaluations_since_conversion =
      read_vector_dataset_checked<int>(
          file,
          "evacuated_cells/controller_evaluations_since_conversion",
          H5T_NATIVE_INT, n_cells);
  if (link_exists(file, "evacuated_cells/cell_axis_edge_collapsed")) {
    evacuated.cell_axis_edge_collapsed =
        read_vector_dataset_checked<std::uint8_t>(
            file, "evacuated_cells/cell_axis_edge_collapsed",
            H5T_NATIVE_UINT8, n_cells);
  }
  if (link_exists(file, "evacuated_cells/node_axis_alias")) {
    evacuated.node_axis_alias = read_vector_dataset_checked<int>(
        file, "evacuated_cells/node_axis_alias", H5T_NATIVE_INT, n_nodes);
  }
  hydro::rebuild_axis_edge_collapse_device_state(state);
  TENRYU_ASSERT(link_exists(file, "evacuated_cells/conversions_total"),
                "Restart checkpoint missing dataset: "
                "evacuated_cells/conversions_total");
  evacuated.conversions_total = read_scalar_dataset<int>(
      file, "evacuated_cells/conversions_total", H5T_NATIVE_INT, 0);
  TENRYU_ASSERT(link_exists(file, "evacuated_cells/n_slots"),
                "Restart checkpoint missing dataset: evacuated_cells/n_slots");
  const int n_slots_value = read_scalar_dataset<int>(
      file, "evacuated_cells/n_slots", H5T_NATIVE_INT, -1);
  TENRYU_ASSERT(n_slots_value >= 0,
                "Restart checkpoint evacuated_cells/n_slots must be >= 0");
  const std::size_t n_slots = static_cast<std::size_t>(n_slots_value);

  const auto cell = read_vector_dataset_checked<int>(
      file, "evacuated_cells/cell", H5T_NATIVE_INT, n_slots);
  const auto axis = read_vector_dataset_checked<int>(
      file, "evacuated_cells/axis", H5T_NATIVE_INT, n_slots);
  const auto node_a0 = read_vector_dataset_checked<int>(
      file, "evacuated_cells/node_a0", H5T_NATIVE_INT, n_slots);
  const auto node_a1 = read_vector_dataset_checked<int>(
      file, "evacuated_cells/node_a1", H5T_NATIVE_INT, n_slots);
  const auto node_b0 = read_vector_dataset_checked<int>(
      file, "evacuated_cells/node_b0", H5T_NATIVE_INT, n_slots);
  const auto node_b1 = read_vector_dataset_checked<int>(
      file, "evacuated_cells/node_b1", H5T_NATIVE_INT, n_slots);
  std::vector<double> normal_pair_r0(n_slots, 0.0);
  std::vector<double> normal_pair_z0(n_slots, 1.0);
  std::vector<double> normal_pair_r1(n_slots, 0.0);
  std::vector<double> normal_pair_z1(n_slots, 1.0);
  const bool has_per_pair_normals =
      link_exists(file, "evacuated_cells/normal_pair_r0") &&
      link_exists(file, "evacuated_cells/normal_pair_z0") &&
      link_exists(file, "evacuated_cells/normal_pair_r1") &&
      link_exists(file, "evacuated_cells/normal_pair_z1");
  if (has_per_pair_normals) {
    normal_pair_r0 = read_vector_dataset_checked<double>(
        file, "evacuated_cells/normal_pair_r0", H5T_NATIVE_DOUBLE, n_slots);
    normal_pair_z0 = read_vector_dataset_checked<double>(
        file, "evacuated_cells/normal_pair_z0", H5T_NATIVE_DOUBLE, n_slots);
    normal_pair_r1 = read_vector_dataset_checked<double>(
        file, "evacuated_cells/normal_pair_r1", H5T_NATIVE_DOUBLE, n_slots);
    normal_pair_z1 = read_vector_dataset_checked<double>(
        file, "evacuated_cells/normal_pair_z1", H5T_NATIVE_DOUBLE, n_slots);
  } else if (link_exists(file, "evacuated_cells/normal_r") &&
             link_exists(file, "evacuated_cells/normal_z")) {
    const auto normal_r = read_vector_dataset_checked<double>(
        file, "evacuated_cells/normal_r", H5T_NATIVE_DOUBLE, n_slots);
    const auto normal_z = read_vector_dataset_checked<double>(
        file, "evacuated_cells/normal_z", H5T_NATIVE_DOUBLE, n_slots);
    // Pre-C11 checkpoints used one normal for both pairs, so duplicate it.
    normal_pair_r0 = normal_r;
    normal_pair_z0 = normal_z;
    normal_pair_r1 = normal_r;
    normal_pair_z1 = normal_z;
  }
  std::vector<double> gap_at_engagement_pair0(n_slots, 0.0);
  std::vector<double> gap_at_engagement_pair1(n_slots, 0.0);
  const bool has_gap_at_engagement_pairs =
      link_exists(file, "evacuated_cells/gap_at_engagement_pair0") &&
      link_exists(file, "evacuated_cells/gap_at_engagement_pair1");
  if (has_gap_at_engagement_pairs) {
    gap_at_engagement_pair0 = read_vector_dataset_checked<double>(
        file, "evacuated_cells/gap_at_engagement_pair0", H5T_NATIVE_DOUBLE,
        n_slots);
    gap_at_engagement_pair1 = read_vector_dataset_checked<double>(
        file, "evacuated_cells/gap_at_engagement_pair1", H5T_NATIVE_DOUBLE,
        n_slots);
  }
  const auto h_perp_ref = read_vector_dataset_checked<double>(
      file, "evacuated_cells/h_perp_ref", H5T_NATIVE_DOUBLE, n_slots);
  const auto g0 = read_vector_dataset_checked<double>(
      file, "evacuated_cells/g0", H5T_NATIVE_DOUBLE, n_slots);
  const auto g_arm = read_vector_dataset_checked<double>(
      file, "evacuated_cells/g_arm", H5T_NATIVE_DOUBLE, n_slots);
  const auto gap_pair0 = read_vector_dataset_checked<double>(
      file, "evacuated_cells/gap_pair0", H5T_NATIVE_DOUBLE, n_slots);
  const auto gap_pair1 = read_vector_dataset_checked<double>(
      file, "evacuated_cells/gap_pair1", H5T_NATIVE_DOUBLE, n_slots);
  const auto gap_prev_pair0 = read_vector_dataset_checked<double>(
      file, "evacuated_cells/gap_prev_pair0", H5T_NATIVE_DOUBLE, n_slots);
  const auto gap_prev_pair1 = read_vector_dataset_checked<double>(
      file, "evacuated_cells/gap_prev_pair1", H5T_NATIVE_DOUBLE, n_slots);
  const auto pair_engaged0 = read_vector_dataset_checked<std::uint8_t>(
      file, "evacuated_cells/pair_engaged0", H5T_NATIVE_UINT8, n_slots);
  const auto pair_engaged1 = read_vector_dataset_checked<std::uint8_t>(
      file, "evacuated_cells/pair_engaged1", H5T_NATIVE_UINT8, n_slots);
  const auto tensile_streak_pair0 = read_vector_dataset_checked<int>(
      file, "evacuated_cells/tensile_streak_pair0", H5T_NATIVE_INT,
      n_slots);
  const auto tensile_streak_pair1 = read_vector_dataset_checked<int>(
      file, "evacuated_cells/tensile_streak_pair1", H5T_NATIVE_INT,
      n_slots);
  const auto gap = read_vector_dataset_checked<double>(
      file, "evacuated_cells/gap", H5T_NATIVE_DOUBLE, n_slots);
  const auto gap_prev = read_vector_dataset_checked<double>(
      file, "evacuated_cells/gap_prev", H5T_NATIVE_DOUBLE, n_slots);
  const auto slot_state = read_vector_dataset_checked<std::uint8_t>(
      file, "evacuated_cells/state", H5T_NATIVE_UINT8, n_slots);
  const auto mass_at_engagement = read_vector_dataset_checked<double>(
      file, "evacuated_cells/mass_at_engagement", H5T_NATIVE_DOUBLE,
      n_slots);
  std::vector<double> vol_at_engagement(n_slots, 0.0);
  if (link_exists(file, "evacuated_cells/vol_at_engagement")) {
    vol_at_engagement = read_vector_dataset_checked<double>(
        file, "evacuated_cells/vol_at_engagement", H5T_NATIVE_DOUBLE,
        n_slots);
  }
  std::vector<int> devolumized(n_slots, 0);
  if (link_exists(file, "evacuated_cells/devolumized")) {
    devolumized = read_vector_dataset_checked<int>(
        file, "evacuated_cells/devolumized", H5T_NATIVE_INT, n_slots);
  }
  const auto lambda_last = read_vector_dataset_checked<double>(
      file, "evacuated_cells/lambda_last", H5T_NATIVE_DOUBLE, n_slots);
  const auto impact_heat_total = read_vector_dataset_checked<double>(
      file, "evacuated_cells/impact_heat_total", H5T_NATIVE_DOUBLE,
      n_slots);
  const auto engage_count = read_vector_dataset_checked<int>(
      file, "evacuated_cells/engage_count", H5T_NATIVE_INT, n_slots);
  const auto release_count = read_vector_dataset_checked<int>(
      file, "evacuated_cells/release_count", H5T_NATIVE_INT, n_slots);
  const auto reproject_count = read_vector_dataset_checked<int>(
      file, "evacuated_cells/reproject_count", H5T_NATIVE_INT, n_slots);
  const auto reproject_heat_total = read_vector_dataset_checked<double>(
      file, "evacuated_cells/reproject_heat_total", H5T_NATIVE_DOUBLE,
      n_slots);
  const auto reproject_count_last_logged =
      read_vector_dataset_checked<int>(
          file, "evacuated_cells/reproject_count_last_logged",
          H5T_NATIVE_INT, n_slots);
  const auto reproject_heat_last_logged =
      read_vector_dataset_checked<double>(
          file, "evacuated_cells/reproject_heat_last_logged",
          H5T_NATIVE_DOUBLE, n_slots);

  state.contact_graph.records.resize(n_slots);
  for (std::size_t index = 0; index < n_slots; ++index) {
    auto& slot = state.contact_graph.records[index];
    slot.cell = cell[index];
    slot.axis = axis[index];
    slot.node_a[0] = node_a0[index];
    slot.node_a[1] = node_a1[index];
    slot.node_b[0] = node_b0[index];
    slot.node_b[1] = node_b1[index];
    slot.normal_pair_r[0] = normal_pair_r0[index];
    slot.normal_pair_z[0] = normal_pair_z0[index];
    slot.normal_pair_r[1] = normal_pair_r1[index];
    slot.normal_pair_z[1] = normal_pair_z1[index];
    slot.h_perp_ref = h_perp_ref[index];
    slot.g0 = g0[index];
    slot.g_arm = g_arm[index];
    slot.gap_pair[0] = gap_pair0[index];
    slot.gap_pair[1] = gap_pair1[index];
    slot.gap_prev_pair[0] = gap_prev_pair0[index];
    slot.gap_prev_pair[1] = gap_prev_pair1[index];
    slot.pair_engaged[0] = pair_engaged0[index];
    slot.pair_engaged[1] = pair_engaged1[index];
    slot.gap_at_engagement_pair[0] =
        has_gap_at_engagement_pairs
            ? gap_at_engagement_pair0[index]
            : (slot.pair_engaged[0] != 0U ? slot.g0 : 0.0);
    slot.gap_at_engagement_pair[1] =
        has_gap_at_engagement_pairs
            ? gap_at_engagement_pair1[index]
            : (slot.pair_engaged[1] != 0U ? slot.g0 : 0.0);
    slot.tensile_streak_pair[0] = tensile_streak_pair0[index];
    slot.tensile_streak_pair[1] = tensile_streak_pair1[index];
    slot.gap = gap[index];
    slot.gap_prev = gap_prev[index];
    slot.state = static_cast<core::EvacContactState>(slot_state[index]);
    slot.mass_at_engagement = mass_at_engagement[index];
    slot.vol_at_engagement = vol_at_engagement[index];
    slot.devolumized = static_cast<std::uint8_t>(devolumized[index]);
    slot.lambda_last = lambda_last[index];
    slot.impact_heat_total = impact_heat_total[index];
    slot.engage_count = engage_count[index];
    slot.release_count = release_count[index];
    slot.reproject_count = reproject_count[index];
    slot.reproject_heat_total = reproject_heat_total[index];
    slot.reproject_count_last_logged = reproject_count_last_logged[index];
    slot.reproject_heat_last_logged = reproject_heat_last_logged[index];
  }

  evacuated.d_inactive_member_mask.reset(
      evacuated.inactive_member_mask.size());
  evacuated.d_inactive_member_mask.copy_from_host(
      evacuated.inactive_member_mask);
  for (const core::EvacContactSlot& slot : state.contact_graph.records) {
    if (slot.devolumized == 0U) {
      continue;
    }
    if (state.mesh.geometry_exempt_cells.empty()) {
      state.mesh.geometry_exempt_cells.assign(n_cells, 0U);
    }
    state.mesh.geometry_exempt_cells[static_cast<std::size_t>(slot.cell)] =
        1U;
  }
  hydro::rebuild_geometry_policy_exempt(state);
  hydro::evacuated_cell_upload_contact_device_state(state);
  evacuated.contact_dt_cap_s = std::numeric_limits<double>::infinity();
}

void prepare_evolved_reale_node_state(core::State& state) {
  std::vector<double> node_r;
  std::vector<double> node_z;
  state.x_r.copy_to_host(node_r);
  state.x_z.copy_to_host(node_z);

  // Match coupling/reale_mode.cpp::install_node_fields for runtime-only
  // node fields. ALE reference fields are restored from their checkpoint group.
  state.x_r_initial = node_r;
  state.x_z_initial = node_z;
  state.node_planar_mass.reset(node_r.size());
  state.node_planar_mass.fill(0.0);
  state.ring7_last_failed_v_r.reset(0);
  state.ring7_last_failed_v_z.reset(0);
  state.ring7_last_failed_velocity_valid = false;
  state.ref_xi_node.reset(0);
  state.ref_xi_cell.reset(0);
  state.ref_dir0_r.reset(0);
  state.ref_dir0_z.reset(0);
  state.ale_monitor_ref_x_r.reset(0);
  state.ale_monitor_ref_x_z.reset(0);
  state.ale_monitor_ref_corner_fractions.reset(0);
  state.ale_monitor_ref_valid = false;
  state.ale_monitor_ref_post_morph = false;
  state.trace_mesh_outer_node = -1;
  state.trace_mesh_outer_cell = -1;

  state.mesh.node_r = state.x_r.data();
  state.mesh.node_z = state.x_z.data();
}

std::size_t topology_dataset_size(const hid_t file,
                                  const std::string& group_path,
                                  const std::string& name) {
  const std::string path = group_path + "/" + name;
  TENRYU_ASSERT(link_exists(file, path),
                "ConfigError: checkpoint missing topology dataset: /" + path);
  return dataset_size(file, path);
}

std::vector<int> read_topology_int_dataset(const hid_t file,
                                           const std::string& group_path,
                                           const std::string& name,
                                           const std::size_t expected_size) {
  const std::string path = group_path + "/" + name;
  TENRYU_ASSERT(link_exists(file, path),
                "ConfigError: checkpoint missing topology dataset: /" + path);
  const std::size_t actual_size = dataset_size(file, path);
  TENRYU_ASSERT(actual_size == expected_size,
                "ConfigError: checkpoint topology dataset size mismatch for /" + path +
                    " (expected " + std::to_string(expected_size) +
                    ", got " + std::to_string(actual_size) + ")");
  return read_vector_dataset<int>(file, path, H5T_NATIVE_INT);
}

void validate_orientation_signs(const std::vector<int>& values,
                                const std::string& path) {
  for (const int value : values) {
    TENRYU_ASSERT(value == -1 || value == 1,
                  "ConfigError: checkpoint topology orientation sign must be +/-1 in /" +
                      path);
  }
}

void load_topology_v3_block_table(const hid_t file,
                                  const std::string& group_path,
                                  const int block_count,
                                  const int n_cells,
                                  const int n_nodes,
                                  mesh::MultiBlockTopology& mb) {
  const std::size_t n_blocks = static_cast<std::size_t>(block_count);
  const auto block_id =
      read_topology_int_dataset(file, group_path, "block_id", n_blocks);
  const auto block_role =
      read_topology_int_dataset(file, group_path, "block_role", n_blocks);
  const auto block_n_i_cells =
      read_topology_int_dataset(file, group_path, "block_n_i_cells", n_blocks);
  const auto block_n_j_cells =
      read_topology_int_dataset(file, group_path, "block_n_j_cells", n_blocks);
  const auto block_cell_begin =
      read_topology_int_dataset(file, group_path, "block_cell_begin", n_blocks);
  const auto block_cell_count =
      read_topology_int_dataset(file, group_path, "block_cell_count", n_blocks);
  const auto block_owned_node_begin =
      read_topology_int_dataset(file, group_path, "block_owned_node_begin", n_blocks);
  const auto block_owned_node_count =
      read_topology_int_dataset(file, group_path, "block_owned_node_count", n_blocks);

  mb.blocks.resize(n_blocks);
  std::int64_t cell_sum = 0;
  std::int64_t owned_node_sum = 0;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    TENRYU_ASSERT(block_id[b] == static_cast<int>(b),
                  "ConfigError: checkpoint topology block_id table must be dense "
                  "and zero-based for /" +
                      group_path);
    TENRYU_ASSERT(block_role[b] >= static_cast<int>(mesh::BlockRole::CENTRAL_CORE) &&
                      block_role[b] <=
                          static_cast<int>(mesh::BlockRole::PENTAGON_BELT),
                  "ConfigError: checkpoint topology block_role out of range in /" +
                      group_path);
    TENRYU_ASSERT(block_n_i_cells[b] >= 0 && block_n_j_cells[b] >= 0 &&
                      block_cell_begin[b] >= 0 && block_cell_count[b] >= 0 &&
                      block_owned_node_begin[b] >= 0 &&
                      block_owned_node_count[b] >= 0,
                  "ConfigError: checkpoint topology block table contains negative "
                  "extent in /" +
                      group_path);
    TENRYU_ASSERT(block_cell_begin[b] <= n_cells &&
                      block_cell_count[b] <= n_cells - block_cell_begin[b],
                  "ConfigError: checkpoint topology block cell extent out of range "
                  "in /" +
                      group_path);
    TENRYU_ASSERT(block_owned_node_begin[b] <= n_nodes &&
                      block_owned_node_count[b] <= n_nodes - block_owned_node_begin[b],
                  "ConfigError: checkpoint topology block owned-node extent out of "
                  "range in /" +
                      group_path);
    TENRYU_ASSERT(block_cell_count[b] ==
                      block_n_i_cells[b] * block_n_j_cells[b],
                  "ConfigError: checkpoint topology block cell count does not match "
                  "block dimensions in /" +
                      group_path);

    mb.blocks[b] = {static_cast<mesh::BlockRole>(block_role[b]),
                    block_n_i_cells[b],
                    block_n_j_cells[b],
                    block_cell_begin[b],
                    block_cell_count[b],
                    block_owned_node_begin[b],
                    block_owned_node_count[b]};
    cell_sum += block_cell_count[b];
    owned_node_sum += block_owned_node_count[b];
  }

  TENRYU_ASSERT(cell_sum == n_cells,
                "ConfigError: checkpoint topology block cell counts do not sum to "
                "N_cell for /" +
                    group_path);
  TENRYU_ASSERT(owned_node_sum == n_nodes,
                "ConfigError: checkpoint topology owned-node counts do not sum to "
                "N_node for /" +
                    group_path);
}

void load_topology_v3_seam_table(const hid_t file,
                                 const std::string& group_path,
                                 const int block_count,
                                 mesh::MultiBlockTopology& mb) {
  const std::size_t n_seams =
      topology_dataset_size(file, group_path, "seam_block_a");
  const auto seam_block_a =
      read_topology_int_dataset(file, group_path, "seam_block_a", n_seams);
  const auto seam_side_a =
      read_topology_int_dataset(file, group_path, "seam_side_a", n_seams);
  const auto seam_block_b =
      read_topology_int_dataset(file, group_path, "seam_block_b", n_seams);
  const auto seam_side_b =
      read_topology_int_dataset(file, group_path, "seam_side_b", n_seams);
  const auto seam_orientation =
      read_topology_int_dataset(file, group_path, "seam_orientation", n_seams);
  const auto seam_index_begin =
      read_topology_int_dataset(file, group_path, "seam_index_begin", n_seams);
  const auto seam_index_count =
      read_topology_int_dataset(file, group_path, "seam_index_count", n_seams);

  mb.seams.resize(n_seams);
  for (std::size_t s = 0; s < n_seams; ++s) {
    TENRYU_ASSERT(seam_block_a[s] >= 0 && seam_block_a[s] < block_count &&
                      seam_block_b[s] >= 0 && seam_block_b[s] < block_count,
                  "ConfigError: checkpoint topology seam block id out of range "
                  "in /" +
                      group_path);
    TENRYU_ASSERT(seam_side_a[s] >= static_cast<int>(mesh::BlockSide::I_MINUS) &&
                      seam_side_a[s] <=
                          static_cast<int>(mesh::BlockSide::COMPOSITE_OUTER) &&
                      seam_side_b[s] >= static_cast<int>(mesh::BlockSide::I_MINUS) &&
                      seam_side_b[s] <=
                          static_cast<int>(mesh::BlockSide::COMPOSITE_OUTER),
                  "ConfigError: checkpoint topology seam side out of range in /" +
                      group_path);
    TENRYU_ASSERT(seam_orientation[s] == -1 || seam_orientation[s] == 1,
                  "ConfigError: checkpoint topology seam orientation must be +/-1 "
                  "in /" +
                      group_path);
    TENRYU_ASSERT(seam_index_begin[s] >= 0 && seam_index_count[s] >= 0,
                  "ConfigError: checkpoint topology seam index extent is negative "
                  "in /" +
                      group_path);
    mb.seams[s] = {seam_block_a[s],
                   static_cast<mesh::BlockSide>(seam_side_a[s]),
                   seam_block_b[s],
                   static_cast<mesh::BlockSide>(seam_side_b[s]),
                   seam_orientation[s],
                   seam_index_begin[s],
                   seam_index_count[s]};
  }
}

void validate_csr_offsets(const std::vector<int>& offsets,
                          const std::size_t expected_cells,
                          const std::size_t expected_entries,
                          const int expected_stride,
                          const std::string& path) {
  TENRYU_ASSERT(offsets.size() == expected_cells + 1U,
                "ConfigError: checkpoint topology CSR offset size mismatch for /" +
                    path);
  TENRYU_ASSERT(!offsets.empty() && offsets.front() == 0,
                "ConfigError: checkpoint topology CSR offsets must start at zero for /" +
                    path);
  for (std::size_t c = 0; c < expected_cells; ++c) {
    const int off = offsets[c];
    const int next = offsets[c + 1U];
    TENRYU_ASSERT(next >= off,
                  "ConfigError: checkpoint topology CSR offsets are not monotone for /" +
                      path);
    TENRYU_ASSERT(next - off == expected_stride,
                  "ConfigError: checkpoint topology CSR offsets must encode the "
                  "checkpoint corner_stride entries "
                  "per cell for /" +
                      path);
  }
  TENRYU_ASSERT(offsets.back() >= 0 &&
                    static_cast<std::size_t>(offsets.back()) == expected_entries,
                "ConfigError: checkpoint topology CSR terminal offset mismatch for /" +
                    path);
}

void validate_index_range(const std::vector<int>& values,
                          const int lower,
                          const int upper,
                          const bool allow_minus_one,
                          const std::string& path) {
  for (const int value : values) {
    if (allow_minus_one && value == -1) {
      continue;
    }
    TENRYU_ASSERT(value >= lower && value < upper,
                  "ConfigError: checkpoint topology index out of range in /" + path);
  }
}

void load_mesh_topology_v2_group(const hid_t file,
                                 const core::Config& cfg,
                                 const int corner_stride,
                                 mesh::MeshTopology& topo) {
  if (!link_exists(file, kMeshTopologyV2Path)) {
    return;
  }

  const int n_cells_i = mesh::mesh_topo_n_cells_total(cfg.mesh);
  const int n_nodes_i = mesh::mesh_topo_n_nodes_total(cfg.mesh);
  const std::size_t n_cells = static_cast<std::size_t>(n_cells_i);
  const std::size_t n_csr_entries = checked_mul_size(
      n_cells, static_cast<std::size_t>(corner_stride),
      "mesh topology v2 CSR entries");

  mesh::MultiBlockTopology mb =
      topo.multiblock.has_value()
          ? std::move(*topo.multiblock)
          : mesh::mesh_topo_make_empty_multiblock_topology(cfg.mesh);
  const auto block_count =
      read_topology_int_dataset(file, kMeshTopologyV2Path, "block_count", 1U);
  mb.block_count = block_count[0];
  // This is a constraint generalization, NOT a format change.
  // v2 validates the stored block count against the expected scheme count.
  const int expected_block_count =
      expected_topology_v2_block_count(cfg.mesh);
  TENRYU_ASSERT(mb.block_count == expected_block_count,
                "ConfigError: checkpoint /mesh/topology/v2/block_count mismatch "
                "(expected " + std::to_string(expected_block_count) +
                    ", got " +
                    std::to_string(mb.block_count) + ")");

  mb.cell_block_id =
      read_topology_int_dataset(file, kMeshTopologyV2Path, "cell_block_id", n_cells);
  mb.cell_node_csr_offsets =
      read_topology_int_dataset(file, kMeshTopologyV2Path, "cell_node_csr_offsets",
                                n_cells + 1U);
  mb.cell_node_csr_indices =
      read_topology_int_dataset(file, kMeshTopologyV2Path, "cell_node_csr_indices",
                                n_csr_entries);
  mb.face_adj_csr_offsets =
      read_topology_int_dataset(file, kMeshTopologyV2Path, "face_adj_csr_offsets",
                                n_cells + 1U);
  mb.face_adj_csr_indices =
      read_topology_int_dataset(file, kMeshTopologyV2Path, "face_adj_csr_indices",
                                n_csr_entries);
  mb.face_bc_tags =
      read_topology_int_dataset(file, kMeshTopologyV2Path, "face_bc_tags",
                                n_csr_entries);
  mb.cell_id_stable =
      read_topology_int_dataset(file, kMeshTopologyV2Path, "cell_id_stable", n_cells);

  validate_csr_offsets(mb.cell_node_csr_offsets,
                       n_cells,
                       mb.cell_node_csr_indices.size(),
                       corner_stride,
                       std::string(kMeshTopologyV2Path) + "/cell_node_csr_offsets");
  validate_csr_offsets(mb.face_adj_csr_offsets,
                       n_cells,
                       mb.face_adj_csr_indices.size(),
                       corner_stride,
                       std::string(kMeshTopologyV2Path) + "/face_adj_csr_offsets");
  validate_index_range(mb.cell_block_id,
                       0,
                       mb.block_count,
                       false,
                       std::string(kMeshTopologyV2Path) + "/cell_block_id");
  validate_index_range(mb.cell_node_csr_indices,
                       0,
                       n_nodes_i,
                       false,
                       std::string(kMeshTopologyV2Path) + "/cell_node_csr_indices");
  validate_index_range(mb.face_adj_csr_indices,
                       0,
                       n_cells_i,
                       true,
                       std::string(kMeshTopologyV2Path) + "/face_adj_csr_indices");
  validate_index_range(mb.cell_id_stable,
                       0,
                       n_cells_i,
                       false,
                       std::string(kMeshTopologyV2Path) + "/cell_id_stable");

  topo.nr = cfg.mesh.nr;
  topo.nz = cfg.mesh.nz;
  topo.n_cells = n_cells_i;
  topo.n_nodes = n_nodes_i;
  topo.multiblock = std::move(mb);
}

void load_mesh_topology_v3_group(const hid_t file,
                                 const core::Config& cfg,
                                 const int corner_stride,
                                 const int checkpoint_n_nodes,
                                 mesh::MeshTopology& topo) {
  if (!link_exists(file, kMeshTopologyV3Path)) {
    return;
  }

  const int n_cells_i = mesh::mesh_topo_n_cells_total(cfg.mesh);
  const int initial_n_nodes_i = mesh::mesh_topo_n_nodes_total(cfg.mesh);
  const std::size_t n_cells = static_cast<std::size_t>(n_cells_i);
  const std::size_t n_csr_entries = checked_mul_size(
      n_cells, static_cast<std::size_t>(corner_stride),
      "mesh topology v3 CSR entries");

  mesh::MultiBlockTopology mb =
      topo.multiblock.has_value()
          ? std::move(*topo.multiblock)
          : mesh::mesh_topo_make_empty_multiblock_topology(cfg.mesh);
  const auto block_count =
      read_topology_int_dataset(file, kMeshTopologyV3Path, "block_count", 1U);
  mb.block_count = block_count[0];
  if (cfg.mesh.topology_scheme ==
      core::TopologyScheme::PENTAGON_BELT_SHELL) {
    const int expected_block_count =
        mesh::mesh_topo_make_empty_multiblock_topology(cfg.mesh).block_count;
    TENRYU_ASSERT(
        mb.block_count == expected_block_count,
        "ConfigError: checkpoint /mesh/topology/v3/block_count mismatch "
        "(expected " +
            std::to_string(expected_block_count) + ", got " +
            std::to_string(mb.block_count) + ")");
  } else {
    const int expected_block_count =
        expected_topology_v3_block_count(cfg.mesh);
    TENRYU_ASSERT(
        mb.block_count == expected_block_count,
        "ConfigError: checkpoint /mesh/topology/v3/block_count mismatch "
        "(expected " +
            std::to_string(expected_block_count) + ", got " +
            std::to_string(mb.block_count) + ")");
  }

  load_topology_v3_block_table(file,
                               kMeshTopologyV3Path,
                               mb.block_count,
                               n_cells_i,
                               initial_n_nodes_i,
                               mb);
  load_topology_v3_seam_table(file, kMeshTopologyV3Path, mb.block_count, mb);

  mb.cell_block_id =
      read_topology_int_dataset(file, kMeshTopologyV3Path, "cell_block_id", n_cells);
  mb.cell_id_stable =
      read_topology_int_dataset(file, kMeshTopologyV3Path, "cell_id_stable", n_cells);
  mb.cell_orientation_sign =
      read_topology_int_dataset(file, kMeshTopologyV3Path, "cell_orientation_sign",
                                n_cells);
  mb.cell_node_csr_offsets =
      read_topology_int_dataset(file, kMeshTopologyV3Path, "cell_node_csr_offsets",
                                n_cells + 1U);
  mb.cell_node_csr_indices =
      read_topology_int_dataset(file, kMeshTopologyV3Path, "cell_node_csr_indices",
                                n_csr_entries);
  mb.face_adj_csr_offsets =
      read_topology_int_dataset(file, kMeshTopologyV3Path, "face_adj_csr_offsets",
                                n_cells + 1U);
  mb.face_adj_csr_indices =
      read_topology_int_dataset(file, kMeshTopologyV3Path, "face_adj_csr_indices",
                                n_csr_entries);
  mb.face_bc_tags =
      read_topology_int_dataset(file, kMeshTopologyV3Path, "face_bc_tags",
                                n_csr_entries);

  validate_csr_offsets(mb.cell_node_csr_offsets,
                       n_cells,
                       mb.cell_node_csr_indices.size(),
                       corner_stride,
                       std::string(kMeshTopologyV3Path) + "/cell_node_csr_offsets");
  validate_csr_offsets(mb.face_adj_csr_offsets,
                       n_cells,
                       mb.face_adj_csr_indices.size(),
                       corner_stride,
                       std::string(kMeshTopologyV3Path) + "/face_adj_csr_offsets");
  validate_index_range(mb.cell_block_id,
                       0,
                       mb.block_count,
                       false,
                       std::string(kMeshTopologyV3Path) + "/cell_block_id");
  if (std::getenv("TENRYU_RESTART_DEBUG") != nullptr) {
    const int max_idx = mb.cell_node_csr_indices.empty()
                            ? -1
                            : *std::max_element(
                                  mb.cell_node_csr_indices.begin(),
                                  mb.cell_node_csr_indices.end());
    std::fprintf(stderr,
                 "[restart_dbg] v3 idx validate: upper=%d initial=%d max_idx=%d\n",
                 checkpoint_n_nodes,
                 initial_n_nodes_i,
                 max_idx);
  }
  // Staged-CSR convention uses -1 padding for inactive slots.
  validate_index_range(mb.cell_node_csr_indices,
                       0,
                       checkpoint_n_nodes,
                       true,
                       std::string(kMeshTopologyV3Path) + "/cell_node_csr_indices");
  validate_index_range(mb.face_adj_csr_indices,
                       0,
                       n_cells_i,
                       true,
                       std::string(kMeshTopologyV3Path) + "/face_adj_csr_indices");
  validate_index_range(mb.cell_id_stable,
                       0,
                       n_cells_i,
                       false,
                       std::string(kMeshTopologyV3Path) + "/cell_id_stable");
  validate_orientation_signs(mb.cell_orientation_sign,
                             std::string(kMeshTopologyV3Path) +
                                 "/cell_orientation_sign");

  topo.nr = cfg.mesh.nr;
  topo.nz = cfg.mesh.nz;
  topo.n_cells = n_cells_i;
  topo.n_nodes = checkpoint_n_nodes;
  topo.multiblock = std::move(mb);
}

void load_mesh_topology_group(const hid_t file,
                              const core::Config& cfg,
                              const int checkpoint_n_nodes,
                              mesh::MeshTopology& topo) {
  const int corner_stride = checkpoint_corner_stride(file);
  if (link_exists(file, kMeshTopologyV3Path)) {
    load_mesh_topology_v3_group(
        file, cfg, corner_stride, checkpoint_n_nodes, topo);
    return;
  }
  load_mesh_topology_v2_group(file, cfg, corner_stride, topo);
}

template <typename Tag>
std::vector<double> copy_field_to_host(const core::Field1D<Tag>& field) {
  std::vector<double> host(field.size(), 0.0);
  if (!host.empty()) {
    field.copy_to_host(host.data());
  }
  return host;
}

PerMaterialCheckpointReadStatus detect_per_material_checkpoint_status(const hid_t file,
                                                                      const bool per_material_enabled) {
  constexpr const char* group = "hydro/per_material/v1";
  if (!link_exists(file, group)) {
    return per_material_enabled ? PerMaterialCheckpointReadStatus::MissingGroupEnabled
                                : PerMaterialCheckpointReadStatus::MissingGroupDisabled;
  }
  const bool enabled = read_group_attr_u8(file, group, "enabled").value_or(0U) != 0U;
  if (!enabled) {
    return PerMaterialCheckpointReadStatus::PresentDisabled;
  }
  const bool complete = link_exists(file, "hydro/per_material/v1/mass") &&
                        link_exists(file, "hydro/per_material/v1/Ee") &&
                        link_exists(file, "hydro/per_material/v1/Ei");
  return complete ? PerMaterialCheckpointReadStatus::PresentEnabled
                  : PerMaterialCheckpointReadStatus::PresentEnabledIncomplete;
}

double material_zbar_for_restart(const core::Config& cfg, const std::size_t m) {
  if (cfg.materials.zbar.model == "fixed" && cfg.materials.zbar.fixed_value >= 0.0) {
    return cfg.materials.zbar.fixed_value;
  }
  if (m >= cfg.materials.materials.size()) {
    return 1.0;
  }
  const auto& mat = cfg.materials.materials[m];
  if (mat.is_void) {
    return 0.0;
  }
  return (mat.Z > 0.0) ? mat.Z : 1.0;
}

double ideal_species_cv_for_restart(const double zbar,
                                    const double A_amu,
                                    const double gamma) {
  constexpr double kProtonMass = 1.6726219e-24;
  constexpr double kEvToErg = 1.6022e-12;
  const double gm1 = gamma - 1.0;
  if (!(A_amu > 0.0) || !(gm1 > 0.0) || !(zbar >= 0.0)) {
    return 0.0;
  }
  return zbar * kEvToErg / (A_amu * kProtonMass * gm1);
}

bool restart_can_rebuild_per_material_derived_on_cpu(const core::Config& cfg) {
  return std::all_of(cfg.materials.materials.begin(),
                     cfg.materials.materials.end(),
                     [](const auto& mat) { return mat.eos_model == "ideal_gas"; });
}

void rebuild_ideal_per_material_derived_after_restart(core::State& state,
                                                      const core::Config& cfg) {
  if (!cfg.numerics.materials.per_material_conservation_enabled ||
      !restart_can_rebuild_per_material_derived_on_cpu(cfg)) {
    if (cfg.numerics.materials.per_material_conservation_enabled) {
      core::log_warning(
          "Restart loaded V22 per-material conserved arrays; table-backed derived "
          "cell fields remain from checkpoint until the next HydroEOSContext refresh.");
    }
    return;
  }
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_mat = cfg.materials.materials.size();
  const std::size_t n_cell_mat = checked_mul_size(n_cells, n_mat, "V22 restart per-material size");
  if (n_cells == 0 || n_mat == 0 || state.mass_per_material.size() != n_cell_mat ||
      state.Ee_per_material.size() != n_cell_mat ||
      state.Ei_per_material.size() != n_cell_mat || state.volFrac.size() != n_cell_mat ||
      state.vol.size() != n_cells) {
    return;
  }

  const auto mass_m = copy_field_to_host(state.mass_per_material);
  const auto Ee_m = copy_field_to_host(state.Ee_per_material);
  const auto Ei_m = copy_field_to_host(state.Ei_per_material);
  const auto volfrac = copy_field_to_host(state.volFrac);
  const auto vol = copy_field_to_host(state.vol);

  std::vector<double> rho(n_cells, 0.0);
  std::vector<double> mass(n_cells, 0.0);
  std::vector<double> ee(n_cells, 0.0);
  std::vector<double> ei(n_cells, 0.0);
  std::vector<double> Te(n_cells, 0.0);
  std::vector<double> Ti(n_cells, 0.0);
  std::vector<double> Pe(n_cells, 0.0);
  std::vector<double> Pi(n_cells, 0.0);
  std::vector<double> zbar(n_cells, 0.0);
  std::vector<double> cv_e(n_cells, 0.0);
  std::vector<double> cv_i(n_cells, 0.0);
  std::vector<double> cs(n_cells, 0.0);

  for (std::size_t c = 0; c < n_cells; ++c) {
    double Te_sum = 0.0;
    double Ti_sum = 0.0;
    double zbar_sum = 0.0;
    double cv_e_sum = 0.0;
    double cv_i_sum = 0.0;
    for (std::size_t m = 0; m < n_mat; ++m) {
      const std::size_t idx = c * n_mat + m;
      const auto& mat = cfg.materials.materials[m];
      const double M = std::max(0.0, mass_m[idx]);
      const double Ee = std::max(0.0, Ee_m[idx]);
      const double Ei = std::max(0.0, Ei_m[idx]);
      const double vf = volfrac[idx];
      const double V = vol[c];
      const double rho_m =
          (M > 0.0 && vf > 0.0 && V > 0.0) ? (M / (vf * V)) : 0.0;
      const double gamma = (mat.ideal_gas_gamma > 1.0) ? mat.ideal_gas_gamma
                                                       : (5.0 / 3.0);
      const double A = (mat.A > 0.0) ? mat.A : 1.0;
      const double z_m = material_zbar_for_restart(cfg, m);
      const double cv_e_m = ideal_species_cv_for_restart(z_m, A, gamma);
      const double cv_i_m = ideal_species_cv_for_restart(1.0, A, gamma);
      const double ee_m = (M > 0.0) ? (Ee / M) : 0.0;
      const double ei_m = (M > 0.0) ? (Ei / M) : 0.0;
      const double Te_m = (cv_e_m > 0.0) ? (ee_m / cv_e_m) : 0.0;
      const double Ti_m = (cv_i_m > 0.0) ? (ei_m / cv_i_m) : 0.0;
      const double Pe_m = (gamma - 1.0) * rho_m * ee_m;
      const double Pi_m = (gamma - 1.0) * rho_m * ei_m;
      const double cs_m =
          (rho_m > 0.0 && (Pe_m + Pi_m) > 0.0)
              ? std::sqrt(std::max(0.0, gamma * (Pe_m + Pi_m) / rho_m))
              : 0.0;
      mass[c] += M;
      ee[c] += Ee;
      ei[c] += Ei;
      Te_sum += M * Te_m;
      Ti_sum += M * Ti_m;
      zbar_sum += M * z_m;
      cv_e_sum += M * cv_e_m;
      cv_i_sum += M * cv_i_m;
      Pe[c] += vf * Pe_m;
      Pi[c] += vf * Pi_m;
      cs[c] = std::max(cs[c], cs_m);
    }
    if (mass[c] > 0.0) {
      rho[c] = (vol[c] > 0.0) ? (mass[c] / vol[c]) : 0.0;
      ee[c] /= mass[c];
      ei[c] /= mass[c];
      Te[c] = Te_sum / mass[c];
      Ti[c] = Ti_sum / mass[c];
      zbar[c] = zbar_sum / mass[c];
      cv_e[c] = cv_e_sum / mass[c];
      cv_i[c] = cv_i_sum / mass[c];
    } else {
      ee[c] = 0.0;
      ei[c] = 0.0;
    }
  }

  state.rho.copy_from_host(rho);
  state.mass.copy_from_host(mass);
  state.ee.copy_from_host(ee);
  state.ei.copy_from_host(ei);
  state.Te.copy_from_host(Te);
  state.Ti.copy_from_host(Ti);
  state.Pe.copy_from_host(Pe);
  state.Pi.copy_from_host(Pi);
  state.zbar.copy_from_host(zbar);
  if (state.cv_e.size() != n_cells) {
    state.cv_e.reset(n_cells);
  }
  if (state.cv_i.size() != n_cells) {
    state.cv_i.reset(n_cells);
  }
  if (state.cs.size() != n_cells) {
    state.cs.reset(n_cells);
  }
  state.cv_e.copy_from_host(cv_e);
  state.cv_i.copy_from_host(cv_i);
  state.cs.copy_from_host(cs);
}

int validate_schema_and_frozen_config(const hid_t file,
                                      const core::Config& cfg) {
  const std::int32_t schema = read_root_attr_i32(file, "schema_version").value_or(0);
  if (schema > kSchemaVersion) {
    TENRYU_ASSERT(false,
                  "checkpoint schema version " + std::to_string(schema) +
                      " is newer than code version " + std::to_string(kSchemaVersion));
  }
  if (schema < kSchemaVersion) {
    core::log_warning("checkpoint schema is older than current reader; using compatibility defaults");
  }

  const auto geometry = read_root_attr_string(file, "geometry");
  if (geometry.has_value()) {
    TENRYU_ASSERT(*geometry == cfg.main.dimension,
                  "ConfigError: checkpoint geometry mismatch (checkpoint=" + *geometry +
                      ", namelist=" + cfg.main.dimension + ")");
  }

  const bool has_topology_v3 = link_exists(file, kMeshTopologyV3Path);
  const bool has_topology_v2 = link_exists(file, kMeshTopologyV2Path);
  const bool has_multiblock_topology = has_topology_v3 || has_topology_v2;
  const bool cfg_multiblock = mesh::mesh_topo_is_multiblock(cfg.mesh);
  if (has_multiblock_topology) {
    const char* topology_path =
        has_topology_v3 ? kMeshTopologyV3Path : kMeshTopologyV2Path;
    TENRYU_ASSERT(cfg_multiblock,
                  "ConfigError: checkpoint contains /" +
                      std::string(topology_path) +
                      " multiblock topology, but namelist topology_scheme is " +
                      topology_scheme_name(cfg.mesh.topology_scheme));
    if (has_topology_v3) {
      TENRYU_ASSERT(
          cfg.mesh.topology_scheme ==
                  core::TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_5BLOCK ||
              cfg.mesh.topology_scheme ==
                  core::TopologyScheme::
                      MULTIBLOCK_HALF_BUTTERFLY_TRIFAN_CAP_5BLOCK ||
              cfg.mesh.topology_scheme ==
                  core::TopologyScheme::MULTIBLOCK_POLAR_TIER ||
              cfg.mesh.topology_scheme == core::TopologyScheme::
                                              MULTIBLOCK_POLAR_TIER_CART_CENTER ||
              cfg.mesh.topology_scheme ==
                  core::TopologyScheme::PENTAGON_BELT_SHELL,
          "ConfigError: checkpoint contains /mesh/topology/v3 "
          "variable-block topology, but namelist topology_scheme is " +
              std::string(topology_scheme_name(cfg.mesh.topology_scheme)));
    } else {
      TENRYU_ASSERT(
          cfg.mesh.topology_scheme ==
                  core::TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL ||
              cfg.mesh.topology_scheme ==
                  core::TopologyScheme::CONE_SHELL_SPINE ||
              cfg.mesh.topology_scheme ==
                  core::TopologyScheme::PENTAGON_BELT_SHELL,
          "ConfigError: checkpoint contains /mesh/topology/v2 topology, but "
          "namelist topology_scheme is " +
              std::string(topology_scheme_name(cfg.mesh.topology_scheme)));
    }
  } else {
    TENRYU_ASSERT(
        !cfg.mesh.polar_tier_native_pentagon,
        "ConfigError: restart support pending for v1 checkpoints with "
        "Mesh.polar_tier_native_pentagon=true");
    TENRYU_ASSERT(mesh::mesh_topo_is_single_block(cfg.mesh),
                  "ConfigError: checkpoint has no /mesh/topology/v2 or "
                  "/mesh/topology/v3 group and is "
                  "treated as v1 single_block, but namelist topology_scheme is " +
                      std::string(topology_scheme_name(cfg.mesh.topology_scheme)));
  }

  const auto n_cells_attr = read_root_attr_i64(file, "n_cells");
  const auto n_nodes_attr = read_root_attr_i64(file, "n_nodes");
  const auto n_groups_attr = read_root_attr_i64(file, "n_groups");
  const auto n_mat_attr = read_root_attr_i64(file, "n_materials");
  const auto nr_attr = read_root_attr_i64(file, "nr");
  const auto nz_attr = read_root_attr_i64(file, "nz");
  const auto two_temperature_attr = read_root_attr_i32(file, "two_temperature");
  const std::int64_t expected_nr = static_cast<std::int64_t>(cfg.mesh.nr);
  const std::int64_t expected_nz =
      static_cast<std::int64_t>((cfg.main.dim == 2) ? cfg.mesh.nz : 1);
  const std::int64_t expected_cells =
      has_multiblock_topology
          ? static_cast<std::int64_t>(mesh::mesh_topo_n_cells_total(cfg.mesh))
          : static_cast<std::int64_t>(cfg.mesh.nr) *
                static_cast<std::int64_t>((cfg.main.dim == 2) ? cfg.mesh.nz : 1);
  const std::int64_t expected_nodes =
      has_multiblock_topology
          ? static_cast<std::int64_t>(mesh::mesh_topo_n_nodes_total(cfg.mesh))
          : static_cast<std::int64_t>(cfg.mesh.nr + 1) *
                static_cast<std::int64_t>((cfg.main.dim == 2) ? (cfg.mesh.nz + 1) : 1);
  if (n_cells_attr.has_value()) {
    TENRYU_ASSERT(*n_cells_attr == expected_cells,
                  "ConfigError: checkpoint mesh size mismatch");
  }
  const bool evolved_reale_v3 =
      cfg.numerics.ale.mesh_mode == "reale_v2" && has_topology_v3;
  TENRYU_ASSERT(
      !evolved_reale_v3 || n_nodes_attr.has_value(),
      "ConfigError: evolved ReALE checkpoint missing root n_nodes attribute");
  const std::int64_t checkpoint_nodes =
      n_nodes_attr.value_or(expected_nodes);
  TENRYU_ASSERT(
      checkpoint_nodes > 0 &&
          checkpoint_nodes <=
              static_cast<std::int64_t>(std::numeric_limits<int>::max()),
      "ConfigError: checkpoint root n_nodes must be a positive int");
  if (!evolved_reale_v3) {
    TENRYU_ASSERT(checkpoint_nodes == expected_nodes,
                  "ConfigError: checkpoint node count mismatch");
  }
  TENRYU_ASSERT(link_exists(file, "mesh/x_r"),
                "Restart checkpoint missing dataset: mesh/x_r");
  const std::size_t x_r_extent = dataset_size(file, "mesh/x_r");
  TENRYU_ASSERT(
      x_r_extent == checked_i64_to_size(
                          checkpoint_nodes,
                          "checkpoint root n_nodes"),
      "ConfigError: checkpoint root n_nodes does not match mesh/x_r extent "
      "(n_nodes=" +
          std::to_string(checkpoint_nodes) + ", mesh/x_r=" +
          std::to_string(x_r_extent) + ")");
  if (n_groups_attr.has_value()) {
    TENRYU_ASSERT(*n_groups_attr == static_cast<std::int64_t>(std::max(cfg.radiation.groups, 1)),
                  "ConfigError: checkpoint n_groups mismatch");
  }
  if (n_mat_attr.has_value()) {
    TENRYU_ASSERT(*n_mat_attr == static_cast<std::int64_t>(cfg.materials.materials.size()),
                  "ConfigError: checkpoint material count mismatch");
  }
  if (nr_attr.has_value()) {
    TENRYU_ASSERT(*nr_attr == expected_nr,
                  "ConfigError: checkpoint nr mismatch (checkpoint=" +
                      std::to_string(*nr_attr) + ", namelist=" + std::to_string(expected_nr) +
                      ")");
  }
  if (nz_attr.has_value()) {
    TENRYU_ASSERT(*nz_attr == expected_nz,
                  "ConfigError: checkpoint nz mismatch (checkpoint=" +
                      std::to_string(*nz_attr) + ", namelist=" + std::to_string(expected_nz) +
                      ")");
  }
  if (two_temperature_attr.has_value() && *two_temperature_attr != 1) {
    core::log_warning("checkpoint two_temperature flag mismatch: checkpoint=" +
                      std::to_string(*two_temperature_attr) + ", expected=1");
  }

  if (link_exists(file, "metadata/group_bounds_eV")) {
    const auto old_bounds =
        read_vector_dataset<double>(file, "metadata/group_bounds_eV", H5T_NATIVE_DOUBLE);
    const auto new_bounds = expected_group_bounds_from_config(cfg);
    TENRYU_ASSERT(old_bounds.size() == new_bounds.size(),
                  "ConfigError: group_bounds_eV size mismatch");
    for (std::size_t i = 0; i < old_bounds.size(); ++i) {
      const double old_v = old_bounds[i];
      const double new_v = new_bounds[i];
      const double tol = std::max(1.0e-12 * std::max(std::abs(old_v), std::abs(new_v)), 1.0e-14);
      TENRYU_ASSERT(std::abs(old_v - new_v) <= tol,
                    "group_bounds_eV mismatch: checkpoint has " + std::to_string(old_v) +
                        ", namelist has " + std::to_string(new_v));
    }
  }

  const auto current_eos_signatures = eos_signatures_from_config(cfg);
  const bool current_has_table_eos = std::any_of(current_eos_signatures.begin(),
                                                 current_eos_signatures.end(),
                                                 [](const std::uint64_t sig) {
                                                   return sig != 0;
                                                 });
  // Schema extension: /metadata/eos/eos_signature stores per-material uint64
  // EOS signatures. Missing path is accepted for legacy checkpoints.
  if (!link_exists(file, "metadata/eos/eos_signature")) {
    core::log_warning(
        "Restart checkpoint missing metadata/eos/eos_signature; skipping EOS signature "
        "validation for legacy compatibility.");
  } else {
    const auto checkpoint_eos_signatures = read_vector_dataset<std::uint64_t>(
        file, "metadata/eos/eos_signature", H5T_NATIVE_UINT64);
    TENRYU_ASSERT(checkpoint_eos_signatures.size() == current_eos_signatures.size(),
                  "ConfigError: checkpoint EOS signature count mismatch (checkpoint=" +
                      std::to_string(checkpoint_eos_signatures.size()) +
                      ", namelist=" + std::to_string(current_eos_signatures.size()) + ")");
    const bool checkpoint_has_table_eos =
        std::any_of(checkpoint_eos_signatures.begin(),
                    checkpoint_eos_signatures.end(),
                    [](const std::uint64_t sig) { return sig != 0; });
    if (current_has_table_eos || checkpoint_has_table_eos) {
      for (std::size_t i = 0; i < current_eos_signatures.size(); ++i) {
        TENRYU_ASSERT(
            checkpoint_eos_signatures[i] == current_eos_signatures[i],
            "ConfigError: EOS signature mismatch for material \"" +
                cfg.materials.materials[i].name + "\" (checkpoint=" +
                format_hash_hex(checkpoint_eos_signatures[i]) +
                ", namelist=" + format_hash_hex(current_eos_signatures[i]) + ")");
      }
    }
  }

  constexpr std::int32_t kPentagonCornerMassPartitionVersion = 2;
  const bool has_pentagon_corner_mass_partition_version =
      link_exists(file, "metadata/pentagon_corner_mass_partition_version");
  const std::int32_t pentagon_corner_mass_partition_version =
      read_scalar_dataset<std::int32_t>(
          file,
          "metadata/pentagon_corner_mass_partition_version",
          H5T_NATIVE_INT32,
          0);
  if (!has_pentagon_corner_mass_partition_version ||
      pentagon_corner_mass_partition_version !=
          kPentagonCornerMassPartitionVersion) {
    const char* forensic =
        std::getenv("TENRYU_I1B_RESTART_FORENSIC");
    const bool forensic_mode =
        forensic != nullptr && std::string_view(forensic) == "1";
    if (forensic_mode) {
      core::log_warning(
          "[restart] FORENSIC MODE: artifact-substrate checkpoint loaded; "
          "results are NOT production-comparable");
    } else {
      throw core::namelist::ConfigError(
          "checkpoint predates the pentagon corner-mass partition "
          "fix (physics epoch clean_pentagon_v2); it cannot continue under this "
          "executable. Set TENRYU_I1B_RESTART_FORENSIC=1 to load it for forensic "
          "analysis only.");
    }
  }
  constexpr std::int32_t kAleSweptSignEpoch = 2;
  bool ale_swept_sign_epoch_forensic = false;
  const bool has_ale_swept_sign_epoch =
      link_exists(file, "metadata/ale_swept_sign_epoch");
  const std::int32_t ale_swept_sign_epoch =
      read_scalar_dataset<std::int32_t>(file,
                                        "metadata/ale_swept_sign_epoch",
                                        H5T_NATIVE_INT32,
                                        0);
  if (!has_ale_swept_sign_epoch ||
      ale_swept_sign_epoch != kAleSweptSignEpoch) {
    const char* forensic = std::getenv("TENRYU_I1B_RESTART_FORENSIC");
    const bool forensic_mode =
        forensic != nullptr && std::string_view(forensic) == "1";
    if (forensic_mode) {
      ale_swept_sign_epoch_forensic = true;
      core::log_warning(
          "[restart] FORENSIC MODE: artifact-substrate checkpoint loaded; "
          "results are NOT production-comparable");
    } else {
      throw core::namelist::ConfigError(
          "legacy swept-sign epoch checkpoint rejected; regenerate from "
          "scratch (consult-17 Q3, 2026-08-05)");
    }
  }
  const bool has_axis_core_released_units =
      link_exists(file, "metadata/axis_core_released_units");
  const std::int32_t axis_core_released_units =
      read_scalar_dataset<std::int32_t>(file,
                                        "metadata/axis_core_released_units",
                                        H5T_NATIVE_INT32,
                                        0);
  if (has_axis_core_released_units) {
    TENRYU_ASSERT(
        axis_core_released_units >= 0,
        "ConfigError: AXIS_CORE_RELEASED_UNITS=" +
            std::to_string(axis_core_released_units) +
            " must be non-negative");
  }
  if (axis_core_released_units > 0) {
    if (!hydro::axis_core_ring_release_enabled(cfg)) {
      const char* forensic = std::getenv("TENRYU_I1B_RESTART_FORENSIC");
      const bool forensic_mode =
          forensic != nullptr && std::string_view(forensic) == "1";
      if (forensic_mode) {
        core::log_warning(
            "[restart] FORENSIC MODE: AXIS_CORE_RELEASED_UNITS=" +
            std::to_string(axis_core_released_units) +
            " loaded while TENRYU_I1B_RING_RELEASE and "
            "Numerics.ale.euler_window.axis_core_ring_release_enabled are "
            "disabled; results are NOT production-comparable");
      } else {
        TENRYU_ASSERT(
            false,
            "ConfigError: checkpoint AXIS_CORE_RELEASED_UNITS=" +
                std::to_string(axis_core_released_units) +
                " requires TENRYU_I1B_RING_RELEASE=1 or "
                "Numerics.ale.euler_window.axis_core_ring_release_enabled=true; "
                "set TENRYU_I1B_RESTART_FORENSIC=1 to load it for forensic "
                "analysis only");
      }
    }
    hydro::axis_core_set_released_units(axis_core_released_units);
  }
  TENRYU_ASSERT(link_exists(file, "metadata/frozen_config"),
                "ConfigError: checkpoint missing metadata/frozen_config");
  const std::string frozen_config = read_string_dataset_checked(file, "metadata/frozen_config");
  TENRYU_ASSERT(!frozen_config.empty(), "ConfigError: checkpoint metadata/frozen_config is empty");
  if (tenryu::io::plic_enabled_from_frozen_config(frozen_config) == true &&
      tenryu::core::is_cart_core_parameterized_topology(
          cfg.mesh.topology_scheme)) {
    TENRYU_ASSERT(
        false,
        "restart carries PLIC-enabled state; multiblock topologies do not "
        "support PLIC (PLIC restart guard); restart on a "
        "structured topology or fork the lineage");
  }
  const auto config_swept_volume_contract =
      hydro::resolve_swept_volume_contract(
          cfg.numerics.ale.swept_volume_sign_fixed);
  constexpr std::uint8_t kSweptVolumeContractAbsent = 255;
  if (link_exists(file, "metadata/swept_volume_contract")) {
    const std::uint8_t plain_csr = read_scalar_dataset<std::uint8_t>(
        file,
        "metadata/swept_volume_contract/plain_csr",
        H5T_NATIVE_UINT8,
        kSweptVolumeContractAbsent);
    const std::uint8_t conservative_csr =
        read_scalar_dataset<std::uint8_t>(
            file,
            "metadata/swept_volume_contract/conservative_csr",
            H5T_NATIVE_UINT8,
            kSweptVolumeContractAbsent);
    const std::uint8_t option_b = read_scalar_dataset<std::uint8_t>(
        file,
        "metadata/swept_volume_contract/option_b",
        H5T_NATIVE_UINT8,
        kSweptVolumeContractAbsent);
    const std::uint8_t axis_band = read_scalar_dataset<std::uint8_t>(
        file,
        "metadata/swept_volume_contract/axis_band",
        H5T_NATIVE_UINT8,
        kSweptVolumeContractAbsent);
    const std::uint8_t plic = read_scalar_dataset<std::uint8_t>(
        file,
        "metadata/swept_volume_contract/plic",
        H5T_NATIVE_UINT8,
        kSweptVolumeContractAbsent);
    if (!ale_swept_sign_epoch_forensic) {
      validate_swept_volume_contract(plain_csr,
                                     conservative_csr,
                                     option_b,
                                     axis_band,
                                     plic,
                                     config_swept_volume_contract);
    }
  } else {
    const auto frozen_swept_volume_sign_fixed =
        tenryu::io::swept_volume_flag_from_frozen_config(frozen_config);
    if (frozen_swept_volume_sign_fixed.has_value()) {
      const auto restart_swept_volume_contract =
          hydro::resolve_swept_volume_contract(
              *frozen_swept_volume_sign_fixed);
      if (!ale_swept_sign_epoch_forensic) {
        validate_swept_volume_contract(
            static_cast<std::uint8_t>(restart_swept_volume_contract.plain_csr),
            static_cast<std::uint8_t>(
                restart_swept_volume_contract.conservative_csr),
            static_cast<std::uint8_t>(restart_swept_volume_contract.option_b),
            static_cast<std::uint8_t>(restart_swept_volume_contract.axis_band),
            static_cast<std::uint8_t>(restart_swept_volume_contract.plic),
            config_swept_volume_contract);
      }
    } else {
      core::log_warning(
          "restart predates the swept-volume contract; resolved LegacyRawV0 from era "
          "(migration rule)");
      const auto era_contract = hydro::resolve_swept_volume_contract(false);
      if (!ale_swept_sign_epoch_forensic) {
        validate_swept_volume_contract(
            static_cast<std::uint8_t>(era_contract.plain_csr),
            static_cast<std::uint8_t>(era_contract.conservative_csr),
            static_cast<std::uint8_t>(era_contract.option_b),
            static_cast<std::uint8_t>(era_contract.axis_band),
            static_cast<std::uint8_t>(era_contract.plic),
            config_swept_volume_contract);
      }
    }
  }
  if (looks_like_json_object(frozen_config)) {
    if (!cfg.meta.frozen_config_json.empty()) {
      const char* allow_drift = std::getenv("TENRYU_I1B_RESTART_ALLOW_CONFIG_DRIFT");
      const bool drift_ok = (allow_drift != nullptr && std::string_view(allow_drift) == "1");
      const bool equivalent = core::namelist::Freeze::configs_equivalent(
          frozen_config, cfg.meta.frozen_config_json);
      if (!equivalent && drift_ok) {
        core::log_warning("[restart] frozen-config mismatch OVERRIDDEN by TENRYU_I1B_RESTART_ALLOW_CONFIG_DRIFT=1 (diagnostic use only; results are NOT production-comparable)");
      } else {
        TENRYU_ASSERT(equivalent,
            "ConfigError: checkpoint frozen_config JSON mismatch between checkpoint and namelist");
      }
    } else {
      core::log_warning(
          "Restart config warning: checkpoint metadata/frozen_config is JSON, but "
          "current run has no comparable canonical JSON. Fixed-item checks were applied.");
    }
  } else {
    const std::string current_signature = checkpoint_config_signature(cfg);
    if (frozen_config != current_signature) {
      core::log_warning(
          "Restart config warning: legacy metadata/frozen_config signature differs "
          "from current namelist signature. checkpoint=\"" +
          abbreviated_preview(frozen_config, 200) + "\", current=\"" +
          abbreviated_preview(current_signature, 200) + "\"");
    }
  }

  const std::uint64_t user_seed =
      read_scalar_dataset<std::uint64_t>(file, "time_state/user_seed", H5T_NATIVE_UINT64, cfg.main.seed);
  TENRYU_ASSERT(user_seed == cfg.main.seed,
                "ConfigError: Main.seed mismatch between checkpoint and namelist");

  core::log_info("Restart config check: fixed items validated");
  if (std::getenv("TENRYU_RESTART_DEBUG") != nullptr) {
    std::fprintf(stderr,
                 "[restart_dbg] schema: checkpoint_nodes=%lld expected=%lld evolved=%d\n",
                 static_cast<long long>(checkpoint_nodes),
                 static_cast<long long>(expected_nodes),
                 evolved_reale_v3 ? 1 : 0);
  }
  return static_cast<int>(checkpoint_nodes);
}

void restore_particles(const hid_t file,
                       radiation::PhotonPool& pool,
                       std::vector<std::int8_t>* ddmc_mode_map,
                       const std::size_t n_cells,
                       const std::size_t n_groups) {
  if (ddmc_mode_map != nullptr) {
    ddmc_mode_map->assign(checked_mul_size(n_cells, n_groups, "ddmc mode-map size"), 0);
  }
  if (!link_exists(file, "particles")) {
    return;
  }

  const auto n_particles_attr = read_group_attr_i64(file, "particles", "n_particles");
  if (n_particles_attr.has_value()) {
    TENRYU_ASSERT(*n_particles_attr >= 0,
                  "Restart checkpoint has negative particles/n_particles attribute");
  }

  const bool has_energy_dataset = link_exists(file, "particles/energy");
  const std::size_t energy_dataset_size =
      has_energy_dataset ? dataset_size(file, "particles/energy") : 0;

  const std::size_t n_p =
      n_particles_attr.has_value() ? checked_i64_to_size(*n_particles_attr, "particles/n_particles")
                                   : energy_dataset_size;
  if (n_particles_attr.has_value() && has_energy_dataset) {
    TENRYU_ASSERT(
        energy_dataset_size == n_p,
        "Restart checkpoint particle count mismatch: particles/n_particles=" +
            std::to_string(n_p) + ", particles/energy size=" + std::to_string(energy_dataset_size));
  }

  TENRYU_ASSERT(n_p <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "Restart checkpoint particle count exceeds INT_MAX");
  const std::int64_t pool_capacity_attr =
      read_scalar_dataset<std::int64_t>(file, "particles/pool_capacity", H5T_NATIVE_INT64, 0);
  const std::size_t pool_capacity_hint =
      (pool_capacity_attr > 0) ? checked_i64_to_size(pool_capacity_attr, "particles/pool_capacity")
                               : 0;
  const std::size_t pool_capacity_size = std::max(n_p, pool_capacity_hint);
  const int pool_capacity =
      checked_size_to_int(pool_capacity_size, "Restart checkpoint particles/pool_capacity");

  if (pool_capacity <= 0) {
    return;
  }

  pool.allocate(pool_capacity);

  if (n_p == 0) {
    pool.n_alive = 0;
    pool.n_census = 0;
    return;
  }

  auto pos_r = read_vector_dataset_checked<double>(file, "particles/pos_r", H5T_NATIVE_DOUBLE, n_p);
  auto pos_z = read_vector_dataset_checked<double>(file, "particles/pos_z", H5T_NATIVE_DOUBLE, n_p);
  auto dir_r = read_vector_dataset_checked<double>(file, "particles/dir_r", H5T_NATIVE_DOUBLE, n_p);
  auto dir_z = read_vector_dataset_checked<double>(file, "particles/dir_z", H5T_NATIVE_DOUBLE, n_p);
  auto dir_phi =
      read_vector_dataset_checked<double>(file, "particles/dir_phi", H5T_NATIVE_DOUBLE, n_p);
  auto energy =
      read_vector_dataset_checked<double>(file, "particles/energy", H5T_NATIVE_DOUBLE, n_p);
  auto birth_energy =
      read_vector_dataset_checked<double>(file, "particles/birth_energy", H5T_NATIVE_DOUBLE, n_p);
  const bool has_sign = link_exists(file, "particles/sign");
  auto sign = has_sign
                  ? read_vector_dataset_checked<std::int8_t>(
                        file, "particles/sign", H5T_NATIVE_INT8, n_p)
                  : std::vector<std::int8_t>(n_p, 1);
  if (!has_sign) {
    core::log_warning(
        "Restart checkpoint missing particles/sign; defaulting all particle signs to +1.");
  }
  auto group_id =
      read_vector_dataset_checked<std::uint16_t>(file, "particles/group_id", H5T_NATIVE_UINT16, n_p);
  auto cell_id =
      read_vector_dataset_checked<std::int32_t>(file, "particles/cell_id", H5T_NATIVE_INT32, n_p);
  auto mode =
      read_vector_dataset_checked<std::uint8_t>(file, "particles/mode", H5T_NATIVE_UINT8, n_p);
  auto global_id = read_vector_dataset_checked<std::uint64_t>(
      file, "particles/global_id", H5T_NATIVE_UINT64, n_p);
  auto weight =
      read_vector_dataset_checked<double>(file, "particles/weight", H5T_NATIVE_DOUBLE, n_p);
  auto time_remain =
      read_vector_dataset_checked<double>(file, "particles/time_remain", H5T_NATIVE_DOUBLE, n_p);

  const bool has_rng_counter = link_exists(file, "rng/rng_counter");
  auto rng_counter = has_rng_counter
                         ? read_vector_dataset_checked<std::uint32_t>(
                               file, "rng/rng_counter", H5T_NATIVE_UINT32, n_p)
                         : std::vector<std::uint32_t>(n_p, 0u);
  if (!has_rng_counter) {
    core::log_warning(
        "Restart checkpoint missing rng/rng_counter; defaulting counters to zero "
        "(reproducibility may change).");
  }
  if (link_exists(file, "rng/global_id")) {
    global_id =
        read_vector_dataset_checked<std::uint64_t>(file, "rng/global_id", H5T_NATIVE_UINT64, n_p);
  }
  const bool has_alive = link_exists(file, "particles/alive");
  auto alive = has_alive
                   ? read_vector_dataset_checked<std::uint8_t>(
                         file, "particles/alive", H5T_NATIVE_UINT8, n_p)
                   : std::vector<std::uint8_t>(n_p, radiation::kAlive);
  if (!has_alive) {
    core::log_warning(
        "Restart checkpoint missing particles/alive; defaulting all particles to alive=1.");
  }

  const std::size_t n = n_p;
  pos_r.resize(n, std::numeric_limits<double>::quiet_NaN());
  pos_z.resize(n, std::numeric_limits<double>::quiet_NaN());
  dir_r.resize(n, std::numeric_limits<double>::quiet_NaN());
  dir_z.resize(n, std::numeric_limits<double>::quiet_NaN());
  dir_phi.resize(n, std::numeric_limits<double>::quiet_NaN());
  group_id.resize(n, 0);
  cell_id.resize(n, 0);
  mode.resize(n, radiation::kModeIMC);
  global_id.resize(n, 0);
  energy.resize(n, 0.0);
  birth_energy.resize(n, 0.0);
  sign.resize(n, 1);
  weight.resize(n, 1.0);
  time_remain.resize(n, 0.0);
  rng_counter.resize(n, 0);
  alive.resize(n, radiation::kAlive);

  std::size_t invalid_group_id = 0;
  std::size_t invalid_cell_id = 0;
  std::size_t invalid_energy = 0;
  std::size_t invalid_sign = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (sign[i] != -1 && sign[i] != 1) {
      sign[i] = 1;
      ++invalid_sign;
    }
    const bool bad_group = static_cast<std::size_t>(group_id[i]) >= n_groups;
    const bool bad_cell = (cell_id[i] < 0) || (static_cast<std::size_t>(cell_id[i]) >= n_cells);
    const bool bad_energy = energy[i] < 0.0;
    if (!(bad_group || bad_cell || bad_energy)) {
      continue;
    }
    alive[i] = radiation::kDead;
    if (bad_group) {
      group_id[i] = 0;
      ++invalid_group_id;
    }
    if (bad_cell) {
      cell_id[i] = 0;
      ++invalid_cell_id;
    }
    if (bad_energy) {
      energy[i] = 0.0;
      birth_energy[i] = std::max(birth_energy[i], 0.0);
      ++invalid_energy;
    }
  }
  if (invalid_group_id > 0 || invalid_cell_id > 0 || invalid_energy > 0) {
    core::log_warning("Restart: invalid particle fields detected; marked particles dead "
                      "(invalid group_id=" + std::to_string(invalid_group_id) +
                      ", invalid cell_id=" + std::to_string(invalid_cell_id) +
                      ", negative energy=" + std::to_string(invalid_energy) + ").");
  }
  if (invalid_sign > 0) {
    core::log_warning("Restart: invalid particle signs detected; reset to +1 "
                      "(invalid sign=" + std::to_string(invalid_sign) + ").");
  }

  if (ddmc_mode_map != nullptr && !ddmc_mode_map->empty()) {
    for (std::size_t i = 0; i < n; ++i) {
      if (alive[i] != radiation::kAlive) {
        continue;
      }
      if (mode[i] != radiation::kModeDDMC && mode[i] != radiation::kModeRW) {
        continue;
      }
      const int cell = cell_id[i];
      const int group = static_cast<int>(group_id[i]);
      if (cell < 0 || static_cast<std::size_t>(cell) >= n_cells || group < 0 ||
          static_cast<std::size_t>(group) >= n_groups) {
        continue;
      }
      const std::size_t idx =
          static_cast<std::size_t>(cell) * n_groups + static_cast<std::size_t>(group);
      (*ddmc_mode_map)[idx] = static_cast<std::int8_t>(mode[i]);
    }
  }

  bool warned_ddmc_nan = false;
  for (std::size_t i = 0; i < n; ++i) {
    if (mode[i] != radiation::kModeDDMC) {
      continue;
    }
    const bool is_nan = std::isnan(pos_r[i]) && std::isnan(pos_z[i]) && std::isnan(dir_r[i]) &&
                        std::isnan(dir_z[i]) && std::isnan(dir_phi[i]);
    if (!is_nan) {
      pos_r[i] = std::numeric_limits<double>::quiet_NaN();
      pos_z[i] = std::numeric_limits<double>::quiet_NaN();
      dir_r[i] = std::numeric_limits<double>::quiet_NaN();
      dir_z[i] = std::numeric_limits<double>::quiet_NaN();
      dir_phi[i] = std::numeric_limits<double>::quiet_NaN();
      warned_ddmc_nan = true;
    }
  }
  if (warned_ddmc_nan) {
    core::log_warning("Restart: repaired DDMC NaN sentinel on legacy checkpoint particles");
  }

  const std::size_t bytes_double =
      checked_bytes_for_count<double>(n, "restart particle double-array copy");
  const std::size_t bytes_group_id =
      checked_bytes_for_count<std::uint16_t>(n, "restart particles/group_id copy");
  const std::size_t bytes_cell_id =
      checked_bytes_for_count<std::int32_t>(n, "restart particles/cell_id copy");
  const std::size_t bytes_sign =
      checked_bytes_for_count<std::int8_t>(n, "restart particles/sign copy");
  const std::size_t bytes_mode =
      checked_bytes_for_count<std::uint8_t>(n, "restart particles/mode copy");
  const std::size_t bytes_global_id =
      checked_bytes_for_count<std::uint64_t>(n, "restart particles/global_id copy");
  const std::size_t bytes_rng_counter =
      checked_bytes_for_count<std::uint32_t>(n, "restart rng/rng_counter copy");

  cuda_check(cudaMemcpy(pool.pos_r, pos_r.data(), bytes_double, cudaMemcpyHostToDevice),
             "restart copy pos_r failed");
  cuda_check(cudaMemcpy(pool.pos_z, pos_z.data(), bytes_double, cudaMemcpyHostToDevice),
             "restart copy pos_z failed");
  cuda_check(cudaMemcpy(pool.dir_r, dir_r.data(), bytes_double, cudaMemcpyHostToDevice),
             "restart copy dir_r failed");
  cuda_check(cudaMemcpy(pool.dir_z, dir_z.data(), bytes_double, cudaMemcpyHostToDevice),
             "restart copy dir_z failed");
  cuda_check(cudaMemcpy(pool.dir_phi, dir_phi.data(), bytes_double, cudaMemcpyHostToDevice),
             "restart copy dir_phi failed");
  cuda_check(cudaMemcpy(pool.energy, energy.data(), bytes_double, cudaMemcpyHostToDevice),
             "restart copy energy failed");
  cuda_check(cudaMemcpy(pool.birth_energy, birth_energy.data(), bytes_double, cudaMemcpyHostToDevice),
             "restart copy birth_energy failed");
  cuda_check(cudaMemcpy(pool.sign, sign.data(), bytes_sign, cudaMemcpyHostToDevice),
             "restart copy sign failed");
  cuda_check(cudaMemcpy(pool.group_id, group_id.data(), bytes_group_id, cudaMemcpyHostToDevice),
             "restart copy group_id failed");
  cuda_check(cudaMemcpy(pool.cell_id, cell_id.data(), bytes_cell_id, cudaMemcpyHostToDevice),
             "restart copy cell_id failed");
  cuda_check(cudaMemcpy(pool.mode, mode.data(), bytes_mode, cudaMemcpyHostToDevice),
             "restart copy mode failed");
  cuda_check(cudaMemcpy(pool.global_id, global_id.data(), bytes_global_id, cudaMemcpyHostToDevice),
             "restart copy global_id failed");
  cuda_check(cudaMemcpy(pool.weight, weight.data(), bytes_double, cudaMemcpyHostToDevice),
             "restart copy weight failed");
  cuda_check(cudaMemcpy(pool.time_remain, time_remain.data(), bytes_double, cudaMemcpyHostToDevice),
             "restart copy time_remain failed");
  cuda_check(cudaMemcpy(pool.rng_counter,
                        rng_counter.data(),
                        bytes_rng_counter,
                        cudaMemcpyHostToDevice),
             "restart copy rng_counter failed");

  cuda_check(cudaMemcpy(pool.alive, alive.data(), bytes_mode, cudaMemcpyHostToDevice),
             "restart copy alive failed");

  pool.n_alive = checked_size_to_int(n, "Restart checkpoint particle count");
  pool.n_census = pool.n_alive;
}

#endif  // TENRYU_ENABLE_HDF5

}  // namespace

PerMaterialCheckpointReadStatus read_per_material_checkpoint_status(
    const std::string& h5_file_path,
    const bool per_material_enabled) {
#if TENRYU_ENABLE_HDF5
  const hid_t file = H5Fopen(h5_file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    return per_material_enabled ? PerMaterialCheckpointReadStatus::MissingGroupEnabled
                                : PerMaterialCheckpointReadStatus::MissingGroupDisabled;
  }
  const auto status = detect_per_material_checkpoint_status(file, per_material_enabled);
  warn_h5_close_failure(H5Fclose(file), "H5Fclose",
                        "HDF5Reader::read_per_material_checkpoint_status");
  return status;
#else
  (void)h5_file_path;
  (void)per_material_enabled;
  return per_material_enabled ? PerMaterialCheckpointReadStatus::MissingGroupEnabled
                              : PerMaterialCheckpointReadStatus::MissingGroupDisabled;
#endif
}

PerMaterialCheckpointReadStatus read_per_material_checkpoint_status(
    const std::string& h5_file_path) {
  return read_per_material_checkpoint_status(h5_file_path, false);
}

CheckpointData HDF5Reader::read_checkpoint(const core::Config& cfg,
                                           const std::string& checkpoint_prefix,
                                           const int rank) const {
  if (std::getenv("TENRYU_RESTART_DEBUG") != nullptr) {
    std::fprintf(stderr,
                 "[restart_dbg] reading checkpoint: %s\n",
                 checkpoint_prefix.c_str());
  }
  CheckpointData out{};

#if TENRYU_ENABLE_HDF5
  const std::filesystem::path checkpoint_path = resolve_checkpoint_path(checkpoint_prefix, rank);

  const hid_t file = H5Fopen(checkpoint_path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  TENRYU_ASSERT(file >= 0, "Failed to open checkpoint file: " + checkpoint_path.string());

  if (link_exists(file, "metadata/epoch_swap/v1") &&
      pole_axis_bbsw_enabled_for_restart(cfg)) {
    TENRYU_ASSERT(
        false,
        "ConfigError: epoch-swap §15 contract violation: swapped articles must never run with pole-axis BBSW");
  }

  constexpr const char* carrier_group_path = "mesh/carrier/v1";
  if (cfg.numerics.ale.mesh_mode == "reale_v2" &&
      !link_exists(file, carrier_group_path)) {
    warn_h5_close_failure(H5Fclose(file), "H5Fclose",
                          "HDF5Reader::read_checkpoint(carrier absence)");
    throw core::namelist::ConfigError(
        "checkpoint predates carrier IO; restart of a carrier run from this "
        "file is not representable — rerun from the deck or restart a gen-0 "
        "checkpoint");
  }

  const int config_stride = core::corner_stride_for_config(cfg);
  if (link_exists(file, "mesh/topology/v4")) {
    const int checkpoint_stride = checkpoint_corner_stride(file);
    TENRYU_ASSERT(
        checkpoint_stride == config_stride,
        "ConfigError: restart checkpoint v4 corner_stride mismatch "
        "(checkpoint=" +
            std::to_string(checkpoint_stride) + ", current config=" +
            std::to_string(config_stride) +
            "); refusing silent stride mixing");
  } else {
    TENRYU_ASSERT(
        config_stride == 4,
        "ConfigError: checkpoint predates v4 corner-stride metadata and cannot "
        "restart a stride-8 (native pentagon / pentagon_belt_shell) "
        "configuration");
  }

  const int checkpoint_n_nodes =
      validate_schema_and_frozen_config(file, cfg);

  out.state = core::State::allocate(cfg);
  out.state.mesh = mesh::create_mesh(cfg, out.state);
  out.state.vol = out.state.mesh.cell_vol;

  const int initial_n_nodes = out.state.mesh.topo.n_nodes;
  std::vector<int> initial_csr_offsets;
  std::vector<int> initial_csr_indices;
  if (out.state.mesh.topo.multiblock.has_value()) {
    initial_csr_offsets =
        out.state.mesh.topo.multiblock->cell_node_csr_offsets;
    initial_csr_indices =
        out.state.mesh.topo.multiblock->cell_node_csr_indices;
  }
  if (checkpoint_n_nodes != initial_n_nodes) {
    const std::size_t n_nodes =
        static_cast<std::size_t>(checkpoint_n_nodes);
    out.state.x_r.reset(n_nodes);
    out.state.x_z.reset(n_nodes);
    out.state.x_r_initial.reset(n_nodes);
    out.state.x_z_initial.reset(n_nodes);
    out.state.x_r_reference.reset(n_nodes);
    out.state.x_z_reference.reset(n_nodes);
    out.state.x_r_shock_target.reset(n_nodes);
    out.state.x_z_shock_target.reset(n_nodes);
    out.state.v_r.reset(n_nodes);
    out.state.v_z.reset(n_nodes);
  }

  load_mesh_topology_group(
      file, cfg, checkpoint_n_nodes, out.state.mesh.topo);

  bool cell_node_csr_changed = false;
  if (out.state.mesh.topo.multiblock.has_value()) {
    const auto& restored_mb = *out.state.mesh.topo.multiblock;
    cell_node_csr_changed =
        restored_mb.cell_node_csr_offsets != initial_csr_offsets ||
        restored_mb.cell_node_csr_indices != initial_csr_indices;
  }

  copy_required_dataset_to_field(file, "mesh/x_r", out.state.x_r);
  copy_required_dataset_to_field(file, "mesh/v_r", out.state.v_r);
  if (cfg.main.dimension == "2D_RZ") {
    copy_required_dataset_to_field(file, "mesh/x_z", out.state.x_z);
    copy_required_dataset_to_field(file, "mesh/v_z", out.state.v_z);
  }
  load_ale_reference_checkpoint_group(file, out.state);
  load_carrier_checkpoint_group(file, out.state);
  load_evacuated_cells_checkpoint_group(file, out.state);
  const bool restored_evolved_reale_topology =
      cfg.numerics.ale.mesh_mode == "reale_v2" &&
      (checkpoint_n_nodes != initial_n_nodes || cell_node_csr_changed ||
       (out.state.boundary_carrier.valid &&
        out.state.carrier_domain_active));
  if (restored_evolved_reale_topology) {
    prepare_evolved_reale_node_state(out.state);
  }

  copy_required_dataset_to_field(file, "hydro/rho", out.state.rho);
  copy_required_dataset_to_field(file, "hydro/Te", out.state.Te);
  copy_required_dataset_to_field(file, "hydro/Ti", out.state.Ti);
  copy_required_dataset_to_field(file, "hydro/ee", out.state.ee);
  copy_required_dataset_to_field(file, "hydro/ei", out.state.ei);
  copy_required_dataset_to_field(file, "hydro/Pe", out.state.Pe);
  copy_required_dataset_to_field(file, "hydro/Pi", out.state.Pi);
  copy_required_dataset_to_field(file, "hydro/mass", out.state.mass);
  out.state.corner_stride = checkpoint_corner_stride(file);
  if (link_exists(file, "mesh/topology/v4")) {
    if (link_exists(file, "mesh/topology/v4/cell_nverts")) {
      out.state.cell_nverts_host = read_vector_dataset<std::int8_t>(
          file, "mesh/topology/v4/cell_nverts", H5T_NATIVE_INT8);
      TENRYU_ASSERT(
          out.state.cell_nverts_host.size() == out.state.rho.size(),
          "Restart checkpoint mesh/topology/v4/cell_nverts size mismatch");
      out.state.mesh.cell_nverts.resize(
          out.state.cell_nverts_host.size());
      for (std::size_t cell = 0;
           cell < out.state.cell_nverts_host.size();
           ++cell) {
        const std::int8_t nverts = out.state.cell_nverts_host[cell];
        TENRYU_ASSERT(
            nverts >= 3 && nverts <= out.state.corner_stride,
            "Restart checkpoint mesh/topology/v4/cell_nverts entry must be "
            "in [3, corner_stride]");
        out.state.mesh.cell_nverts[cell] =
            static_cast<std::uint8_t>(nverts);
      }
    }
  }
  if (restored_evolved_reale_topology) {
    // ReALE install invokes this same helper after replacing cell-node CSR
    // (coupling/reale_mode.cpp::rezone_step). It rebuilds node flags, reverse
    // CSR, face topology, and their device mirrors from the restored authority.
    mesh::rebuild_runtime_topology_caches(out.state, cfg, true);
  }
  if (link_exists(file, "hydro/corner_mass")) {
    const std::size_t expected_corner_mass_size = checked_mul_size(
        out.state.mass.size(),
        static_cast<std::size_t>(out.state.corner_stride),
        "restart hydro/corner_mass size");
    const std::size_t actual_corner_mass_size =
        dataset_size(file, "hydro/corner_mass");
    TENRYU_ASSERT(
        actual_corner_mass_size == expected_corner_mass_size,
        "Restart checkpoint dataset size mismatch for hydro/corner_mass "
        "(expected " +
            std::to_string(expected_corner_mass_size) + ", got " +
            std::to_string(actual_corner_mass_size) + ")");
    out.state.corner_mass.reset(expected_corner_mass_size);
    copy_required_dataset_to_field(file, "hydro/corner_mass", out.state.corner_mass);
    out.state.corner_mass_initialized = true;
  }
  if (link_exists(file, "hydro/hllc_mom_z_cell")) {
    out.state.hllc_mom_z_cell.reset(out.state.mass.size());
    copy_required_dataset_to_field(file,
                                   "hydro/hllc_mom_z_cell",
                                   out.state.hllc_mom_z_cell);
    out.state.hllc_mom_z_cell_initialized = true;
  } else {
    out.state.hllc_mom_z_cell_initialized = false;
  }
  copy_required_dataset_to_field(file, "hydro/vol", out.state.vol);

  if (link_exists(file, "hydro/Qvisc")) {
    copy_vector_to_field(read_vector_dataset<double>(file, "hydro/Qvisc", H5T_NATIVE_DOUBLE),
                         out.state.Qvisc,
                         "hydro/Qvisc");
  }
  if (link_exists(file, "hydro/hot_e_eps_cum")) {
    out.state.hot_e_eps_cum_host = read_vector_dataset_checked<double>(
        file, "hydro/hot_e_eps_cum", H5T_NATIVE_DOUBLE, out.state.rho.size());
  }
  if (link_exists(file, "hydro/burn_rate")) {
    out.state.burn_rate_host = read_vector_dataset_checked<double>(
        file, "hydro/burn_rate", H5T_NATIVE_DOUBLE, out.state.rho.size());
  }
  if (link_exists(file, "hydro/burn_Q_e")) {
    out.state.burn_Q_e_host = read_vector_dataset_checked<double>(
        file, "hydro/burn_Q_e", H5T_NATIVE_DOUBLE, out.state.rho.size());
  }
  if (link_exists(file, "hydro/burn_Q_i")) {
    out.state.burn_Q_i_host = read_vector_dataset_checked<double>(
        file, "hydro/burn_Q_i", H5T_NATIVE_DOUBLE, out.state.rho.size());
  }
  if (link_exists(file, "hydro/burn_eps_cum")) {
    out.state.burn_eps_cum_host = read_vector_dataset_checked<double>(
        file, "hydro/burn_eps_cum", H5T_NATIVE_DOUBLE, out.state.rho.size());
  }
  if (link_exists(file, "hydro/burn_n_D")) {
    constexpr std::size_t kBurnSpeciesCount = 5;
    const char* burn_species_paths[kBurnSpeciesCount] = {
        "hydro/burn_n_D",
        "hydro/burn_n_T",
        "hydro/burn_n_He3",
        "hydro/burn_n_He4",
        "hydro/burn_n_p"};
    std::vector<double> burn_species[kBurnSpeciesCount];
    for (std::size_t s = 0; s < kBurnSpeciesCount; ++s) {
      burn_species[s] = read_vector_dataset_checked<double>(
          file, burn_species_paths[s], H5T_NATIVE_DOUBLE, out.state.rho.size());
    }
    std::vector<double> rho(out.state.rho.size(), 0.0);
    out.state.rho.copy_to_host(rho.data());
    out.state.burn_n_host.assign(out.state.rho.size() * kBurnSpeciesCount, 0.0);
    for (std::size_t c = 0; c < out.state.rho.size(); ++c) {
      const double rho_safe = std::max(rho[c], 1.0e-30);
      for (std::size_t s = 0; s < kBurnSpeciesCount; ++s) {
        out.state.burn_n_host[c * kBurnSpeciesCount + s] =
            burn_species[s][c] / rho_safe;
      }
    }
  }
  // hydro/burn_Ng_slot* stores persistent specific spectra Y_g [1/g].
  if (link_exists(file, "hydro/burn_Ng_slot0")) {
    constexpr std::size_t kBurnSlotCount = 6;
    const std::size_t G = static_cast<std::size_t>(cfg.burn.diffusion_groups);
    const std::size_t slot_size = G * out.state.rho.size();
    std::vector<double> burn_Ng(kBurnSlotCount * slot_size, 0.0);
    for (std::size_t slot = 0; slot < kBurnSlotCount; ++slot) {
      const std::string path =
          "hydro/burn_Ng_slot" + std::to_string(static_cast<long long>(slot));
      const auto slot_values = read_vector_dataset_checked<double>(
          file, path, H5T_NATIVE_DOUBLE, slot_size);
      std::copy(slot_values.begin(),
                slot_values.end(),
                burn_Ng.begin() + slot * slot_size);
    }
    out.state.burn_Ng.reset(burn_Ng.size());
    out.state.burn_Ng.copy_from_host(burn_Ng);
    out.state.burn_diffusion_any = true;
  }
  const bool burn_mc_checkpoint_present =
      link_exists(file, "time_state/burn_mc_live") ||
      link_exists(file, "hydro/burn_mc_r") ||
      link_exists(file, "hydro/burn_mc_mu") ||
      link_exists(file, "hydro/burn_mc_E") ||
      link_exists(file, "hydro/burn_mc_w") ||
      link_exists(file, "hydro/burn_mc_slot") ||
      link_exists(file, "hydro/burn_mc_alive");
  if (burn_mc_checkpoint_present) {
    TENRYU_ASSERT(link_exists(file, "time_state/burn_mc_live"),
                  "Restart checkpoint missing dataset: time_state/burn_mc_live");
    const std::int32_t burn_mc_live_i32 =
        read_scalar_dataset<std::int32_t>(
            file, "time_state/burn_mc_live", H5T_NATIVE_INT32, 0);
    TENRYU_ASSERT(burn_mc_live_i32 >= 0,
                  "Restart checkpoint time_state/burn_mc_live is negative");
    const std::size_t live = static_cast<std::size_t>(burn_mc_live_i32);
    auto mc_r = read_vector_dataset_checked<double>(
        file, "hydro/burn_mc_r", H5T_NATIVE_DOUBLE, live);
    auto mc_mu = read_vector_dataset_checked<double>(
        file, "hydro/burn_mc_mu", H5T_NATIVE_DOUBLE, live);
    auto mc_E = read_vector_dataset_checked<double>(
        file, "hydro/burn_mc_E", H5T_NATIVE_DOUBLE, live);
    auto mc_w = read_vector_dataset_checked<double>(
        file, "hydro/burn_mc_w", H5T_NATIVE_DOUBLE, live);
    auto mc_slot = read_vector_dataset_checked<int>(
        file, "hydro/burn_mc_slot", H5T_NATIVE_INT, live);
    auto mc_alive = read_vector_dataset_checked<unsigned char>(
        file, "hydro/burn_mc_alive", H5T_NATIVE_UCHAR, live);
    out.state.burn_mc_r.reset(live);
    out.state.burn_mc_mu.reset(live);
    out.state.burn_mc_E.reset(live);
    out.state.burn_mc_w.reset(live);
    out.state.burn_mc_slot.reset(live);
    out.state.burn_mc_alive.reset(live);
    out.state.burn_mc_r.copy_from_host(mc_r);
    out.state.burn_mc_mu.copy_from_host(mc_mu);
    out.state.burn_mc_E.copy_from_host(mc_E);
    out.state.burn_mc_w.copy_from_host(mc_w);
    out.state.burn_mc_slot.copy_from_host(mc_slot);
    out.state.burn_mc_alive.copy_from_host(mc_alive);
    out.state.burn_mc_live = static_cast<int>(burn_mc_live_i32);
    out.state.burn_mc_any = true;
  }
  if (!out.state.shock_time.empty() && link_exists(file, "hydro/shock_time")) {
    copy_vector_to_field(
        read_vector_dataset<double>(file, "hydro/shock_time", H5T_NATIVE_DOUBLE),
        out.state.shock_time,
        "hydro/shock_time");
  }
  if (!out.state.adaptive_av_gate.empty() && link_exists(file, "hydro/adaptive_av_gate")) {
    copy_vector_to_field(
        read_vector_dataset<double>(file, "hydro/adaptive_av_gate", H5T_NATIVE_DOUBLE),
        out.state.adaptive_av_gate,
        "hydro/adaptive_av_gate");
  }
  if (link_exists(file, "hydro/zbar")) {
    copy_vector_to_field(read_vector_dataset<double>(file, "hydro/zbar", H5T_NATIVE_DOUBLE),
                         out.state.zbar,
                         "hydro/zbar");
  }
  if (link_exists(file, "hydro/cv_e")) {
    if (out.state.cv_e.size() != out.state.rho.size()) {
      out.state.cv_e.reset(out.state.rho.size());
    }
    copy_vector_to_field(read_vector_dataset<double>(file, "hydro/cv_e", H5T_NATIVE_DOUBLE),
                         out.state.cv_e,
                         "hydro/cv_e");
  }
  if (link_exists(file, "hydro/cv_i")) {
    if (out.state.cv_i.size() != out.state.rho.size()) {
      out.state.cv_i.reset(out.state.rho.size());
    }
    copy_vector_to_field(read_vector_dataset<double>(file, "hydro/cv_i", H5T_NATIVE_DOUBLE),
                         out.state.cv_i,
                         "hydro/cv_i");
  }
  if (link_exists(file, "hydro/cs")) {
    if (out.state.cs.size() != out.state.rho.size()) {
      out.state.cs.reset(out.state.rho.size());
    }
    copy_vector_to_field(read_vector_dataset<double>(file, "hydro/cs", H5T_NATIVE_DOUBLE),
                         out.state.cs,
                         "hydro/cs");
  }
  if (link_exists(file, "hydro/eta_compatible")) {
    copy_vector_to_field(
        read_vector_dataset<double>(file, "hydro/eta_compatible", H5T_NATIVE_DOUBLE),
        out.state.eta_compatible,
        "hydro/eta_compatible");
  }
  if (link_exists(file, "hydro/volFrac")) {
    copy_vector_to_field(read_vector_dataset<double>(file, "hydro/volFrac", H5T_NATIVE_DOUBLE),
                         out.state.volFrac,
                         "hydro/volFrac");
  }
  const auto per_material_status =
      detect_per_material_checkpoint_status(
          file, cfg.numerics.materials.per_material_conservation_enabled);
  out.per_material_checkpoint_status = per_material_status;
  if (per_material_status == PerMaterialCheckpointReadStatus::PresentEnabled) {
    const std::size_t n_cell_mat =
        checked_mul_size(out.state.rho.size(),
                         cfg.materials.materials.size(),
                         "restart V22 per-material conserved fields");
    if (out.state.mass_per_material.size() != n_cell_mat) {
      out.state.mass_per_material.reset(n_cell_mat);
    }
    if (out.state.Ee_per_material.size() != n_cell_mat) {
      out.state.Ee_per_material.reset(n_cell_mat);
    }
    if (out.state.Ei_per_material.size() != n_cell_mat) {
      out.state.Ei_per_material.reset(n_cell_mat);
    }
    copy_required_dataset_to_field(file,
                                   "hydro/per_material/v1/mass",
                                   out.state.mass_per_material);
    copy_required_dataset_to_field(file,
                                   "hydro/per_material/v1/Ee",
                                   out.state.Ee_per_material);
    copy_required_dataset_to_field(file,
                                   "hydro/per_material/v1/Ei",
                                   out.state.Ei_per_material);
    if (cfg.numerics.materials.lazy_cache_te_m_enabled) {
      if (out.state.Te_per_material.size() != n_cell_mat) {
        out.state.Te_per_material.reset(n_cell_mat);
      }
      if (out.state.Ti_per_material.size() != n_cell_mat) {
        out.state.Ti_per_material.reset(n_cell_mat);
      }
      out.state.Te_per_material_valid.assign(n_cell_mat, static_cast<std::uint8_t>(0));
      out.state.Ti_per_material_valid.assign(n_cell_mat, static_cast<std::uint8_t>(0));
    } else {
      out.state.Te_per_material.reset(0);
      out.state.Ti_per_material.reset(0);
      out.state.Te_per_material_valid.clear();
      out.state.Ti_per_material_valid.clear();
    }
    rebuild_ideal_per_material_derived_after_restart(out.state, cfg);
  } else {
    out.state.mass_per_material.reset(0);
    out.state.Ee_per_material.reset(0);
    out.state.Ei_per_material.reset(0);
    out.state.Te_per_material.reset(0);
    out.state.Ti_per_material.reset(0);
    out.state.Te_per_material_valid.clear();
    out.state.Ti_per_material_valid.clear();
  }

  if (link_exists(file, "radiation/energy_density")) {
    copy_vector_to_field(
        read_vector_dataset<double>(file, "radiation/energy_density", H5T_NATIVE_DOUBLE),
        out.state.rad_E,
        "radiation/energy_density");
  }
  if (link_exists(file, "radiation/rad_dep")) {
    copy_vector_to_field(read_vector_dataset<double>(file, "radiation/rad_dep", H5T_NATIVE_DOUBLE),
                         out.state.rad_dep,
                         "radiation/rad_dep");
  }
  if (link_exists(file, "radiation/rad_emit")) {
    copy_vector_to_field(
        read_vector_dataset<double>(file, "radiation/rad_emit", H5T_NATIVE_DOUBLE),
        out.state.rad_emit,
        "radiation/rad_emit");
  } else {
    out.state.rad_emit.fill(0.0);
  }
  if (link_exists(file, "radiation/delta_E_rad_prev")) {
    auto delta_E_rad_prev =
        read_vector_dataset<double>(file, "radiation/delta_E_rad_prev", H5T_NATIVE_DOUBLE);
    if (delta_E_rad_prev.size() != out.state.rho.size()) {
      core::log_warning("Restart checkpoint dataset size mismatch for radiation/delta_E_rad_prev "
                        "(checkpoint=" +
                        std::to_string(delta_E_rad_prev.size()) + ", state=" +
                        std::to_string(out.state.rho.size()) +
                        "); truncating/padding with zeros.");
    }
    if (!delta_E_rad_prev.empty()) {
      delta_E_rad_prev.resize(out.state.rho.size(), 0.0);
      out.state.delta_E_rad_prev.reset(out.state.rho.size());
      out.state.delta_E_rad_prev.copy_from_host(delta_E_rad_prev.data());
    } else {
      out.state.delta_E_rad_prev.reset(0);
    }
  } else {
    out.state.delta_E_rad_prev.reset(0);
  }
  const std::size_t n_groups = static_cast<std::size_t>(std::max(cfg.radiation.groups, 1));
  const std::size_t n_cell_groups =
      checked_mul_size(out.state.rho.size(), n_groups, "restart radiation/ddmc_flag size");
  bool has_checkpoint_ddmc_map = false;
  if (link_exists(file, "radiation/ddmc_flag")) {
    out.state.ddmc_mode_map = read_vector_dataset_checked<std::int8_t>(
        file, "radiation/ddmc_flag", H5T_NATIVE_INT8, n_cell_groups);
    has_checkpoint_ddmc_map = true;
    out.state.ddmc_mode_map_valid = true;
  } else {
    out.state.ddmc_mode_map.assign(n_cell_groups, static_cast<std::int8_t>(0));
    out.state.ddmc_mode_map_valid = false;
  }

  const std::size_t n_cells = out.state.rho.size();
  if (link_exists(file, "holo/E_LO")) {
    copy_vector_to_field(read_vector_dataset<double>(file, "holo/E_LO", H5T_NATIVE_DOUBLE),
                         out.state.holo_E_LO,
                         "holo/E_LO");
  } else {
    out.state.holo_E_LO.fill(0.0);
  }
  if (link_exists(file, "holo/consistency_source")) {
    copy_vector_to_field(
        read_vector_dataset<double>(file, "holo/consistency_source", H5T_NATIVE_DOUBLE),
        out.state.holo_consistency_source,
        "holo/consistency_source");
  } else {
    out.state.holo_consistency_source.fill(0.0);
  }
  if (link_exists(file, "holo/rad_dep_LO")) {
    copy_vector_to_field(read_vector_dataset<double>(file, "holo/rad_dep_LO", H5T_NATIVE_DOUBLE),
                         out.state.holo_rad_dep,
                         "holo/rad_dep_LO");
  } else {
    out.state.holo_rad_dep.fill(0.0);
  }
  if (link_exists(file, "holo/rad_emit_LO")) {
    copy_vector_to_field(read_vector_dataset<double>(file, "holo/rad_emit_LO", H5T_NATIVE_DOUBLE),
                         out.state.holo_rad_emit,
                         "holo/rad_emit_LO");
  } else {
    out.state.holo_rad_emit.fill(0.0);
  }
  if (link_exists(file, "holo/Prr_HO")) {
    copy_vector_to_field(read_vector_dataset<double>(file, "holo/Prr_HO", H5T_NATIVE_DOUBLE),
                         out.state.holo_Prr,
                         "holo/Prr_HO");
  } else {
    out.state.holo_Prr.fill(0.0);
  }
  if (link_exists(file, "holo/chi")) {
    copy_vector_to_field(read_vector_dataset<double>(file, "holo/chi", H5T_NATIVE_DOUBLE),
                         out.state.holo_chi,
                         "holo/chi");
  } else {
    out.state.holo_chi.fill(0.0);
  }
  if (link_exists(file, "holo/Prr_coverage")) {
    copy_vector_to_field(
        read_vector_dataset<double>(file, "holo/Prr_coverage", H5T_NATIVE_DOUBLE),
        out.state.holo_Prr_coverage,
        "holo/Prr_coverage");
  } else {
    out.state.holo_Prr_coverage.fill(0.0);
  }
  if (link_exists(file, "holo/core_mask")) {
    out.state.holo_core_mask = read_vector_dataset_checked<std::uint8_t>(
        file, "holo/core_mask", H5T_NATIVE_UINT8, n_cells);
    out.state.holo_core_mask_valid = true;
  } else {
    out.state.holo_core_mask.assign(n_cells, static_cast<std::uint8_t>(0));
    out.state.holo_core_mask_valid = false;
  }
  if (link_exists(file, "holo/prev_core_mask")) {
    out.state.holo_core_prev_mask = read_vector_dataset_checked<std::uint8_t>(
        file, "holo/prev_core_mask", H5T_NATIVE_UINT8, n_cells);
  } else {
    out.state.holo_core_prev_mask = out.state.holo_core_mask;
  }
  if (link_exists(file, "holo/hold_count")) {
    out.state.holo_hold_count = read_vector_dataset_checked<std::int32_t>(
        file, "holo/hold_count", H5T_NATIVE_INT32, n_cells);
  } else {
    out.state.holo_hold_count.assign(n_cells, 0);
  }
  if (link_exists(file, "holo/dwell_count")) {
    out.state.holo_dwell_count = read_vector_dataset_checked<std::int32_t>(
        file, "holo/dwell_count", H5T_NATIVE_INT32, n_cells);
  } else {
    out.state.holo_dwell_count.assign(n_cells, 0);
  }
  if (link_exists(file, "holo/tau_R")) {
    out.state.holo_tau_R = read_vector_dataset_checked<double>(
        file, "holo/tau_R", H5T_NATIVE_DOUBLE, n_cells);
  } else {
    out.state.holo_tau_R.assign(n_cells, 0.0);
  }
  if (link_exists(file, "holo/reduced_flux")) {
    out.state.holo_reduced_flux = read_vector_dataset_checked<double>(
        file, "holo/reduced_flux", H5T_NATIVE_DOUBLE, n_cells);
  } else {
    out.state.holo_reduced_flux.assign(n_cells, 0.0);
  }
  if (link_exists(file, "holo/mass_q")) {
    out.state.holo_mass_q = read_vector_dataset_checked<double>(
        file, "holo/mass_q", H5T_NATIVE_DOUBLE, n_cells);
  } else {
    out.state.holo_mass_q.assign(n_cells, 0.0);
  }

  if (cfg.laser.enabled && link_exists(file, "laser/deposited_power")) {
    auto deposited_power =
        read_vector_dataset<double>(file, "laser/deposited_power", H5T_NATIVE_DOUBLE);
    deposited_power.resize(out.state.laser_dep.size(), 0.0);
    auto vol = read_vector_dataset<double>(file, "hydro/vol", H5T_NATIVE_DOUBLE);
    vol.resize(out.state.laser_dep.size(), 1.0);
    const double dt = read_scalar_dataset<double>(file, "time_state/dt", H5T_NATIVE_DOUBLE, 0.0);
    const double dt_safe = (dt > 0.0) ? dt : 1.0;
    for (std::size_t i = 0; i < deposited_power.size(); ++i) {
      deposited_power[i] *= std::max(vol[i], 1.0e-30) * dt_safe;
    }
    copy_vector_to_field(deposited_power, out.state.laser_dep, "laser/deposited_power");
  }

  if (link_exists(file, "hydro_flags/hydro_active")) {
    out.state.hydro_active =
        read_vector_dataset<std::int8_t>(file, "hydro_flags/hydro_active", H5T_NATIVE_INT8);
    out.state.note_hydro_active_host_write();
  }
  if (link_exists(file, "metadata/merge_tombstone")) {
    out.state.merge_tombstone = read_vector_dataset_checked<std::uint8_t>(
        file, "metadata/merge_tombstone", H5T_NATIVE_UINT8, n_cells);
    const std::size_t tombstone_count = static_cast<std::size_t>(std::count_if(
        out.state.merge_tombstone.begin(),
        out.state.merge_tombstone.end(),
        [](const std::uint8_t value) { return value != 0U; }));
    if (tombstone_count != 0U) {
      core::log_info("[restart] loaded merge_tombstone mask: tombstones=" +
                     std::to_string(tombstone_count));
    }
  }

  out.state.t = read_scalar_dataset<double>(file, "time_state/t", H5T_NATIVE_DOUBLE, 0.0);
  out.state.step = static_cast<int>(
      read_scalar_dataset<std::int32_t>(file, "time_state/step", H5T_NATIVE_INT32, 0));
  out.state.dt = read_scalar_dataset<double>(file, "time_state/dt", H5T_NATIVE_DOUBLE, 0.0);
  out.state.dt_growth_ref =
      link_exists(file, "time_state/dt_growth_ref")
          ? read_scalar_dataset<double>(file,
                                        "time_state/dt_growth_ref",
                                        H5T_NATIVE_DOUBLE,
                                        out.state.dt)
          : out.state.dt;
  out.state.ale_last_applied_step =
      static_cast<int>(read_scalar_dataset<std::int32_t>(
          file, "time_state/ale_last_applied_step", H5T_NATIVE_INT32, -1));
  out.state.axis_margin_initial = read_scalar_dataset<double>(
      file, "time_state/axis_margin_initial", H5T_NATIVE_DOUBLE, -1.0);
  if (link_exists(file, "time_state/axis_mass_initial")) {
    out.state.axis_mass_initial =
        read_vector_dataset<double>(file, "time_state/axis_mass_initial", H5T_NATIVE_DOUBLE);
  }
  if (link_exists(file, "time_state/axis_inflow_budget")) {
    out.state.axis_inflow_budget =
        read_vector_dataset<double>(file, "time_state/axis_inflow_budget", H5T_NATIVE_DOUBLE);
  }
  out.state.adaptive_av_r0 =
      read_scalar_dataset<double>(file, "time_state/adaptive_av_r0", H5T_NATIVE_DOUBLE, 0.0);
  out.state.adaptive_av_last_rs = read_scalar_dataset<double>(
      file, "time_state/adaptive_av_last_rs", H5T_NATIVE_DOUBLE, 0.0);
  out.state.adaptive_av_last_us = read_scalar_dataset<double>(
      file, "time_state/adaptive_av_last_us", H5T_NATIVE_DOUBLE, 0.0);
  out.state.adaptive_av_rs_min = read_scalar_dataset<double>(
      file,
      "time_state/adaptive_av_rs_min",
      H5T_NATIVE_DOUBLE,
      std::numeric_limits<double>::infinity());
  out.state.adaptive_av_tracker_steps =
      static_cast<int>(read_scalar_dataset<std::int32_t>(
          file, "time_state/adaptive_av_tracker_steps", H5T_NATIVE_INT32, 0));
  out.state.adaptive_av_mode =
      static_cast<int>(read_scalar_dataset<std::int32_t>(
          file, "time_state/adaptive_av_mode", H5T_NATIVE_INT32, 0));
  out.state.adaptive_av_tracker_valid =
      read_scalar_dataset<std::int32_t>(
          file, "time_state/adaptive_av_tracker_valid", H5T_NATIVE_INT32, 0) != 0;
  out.state.adaptive_av_bounce_seen =
      read_scalar_dataset<std::int32_t>(
          file, "time_state/adaptive_av_bounce_seen", H5T_NATIVE_INT32, 0) != 0;

  out.state.E_safety =
      read_scalar_dataset<double>(file, "time_state/E_safety", H5T_NATIVE_DOUBLE, 0.0);
  out.state.E_numerical_loss =
      read_scalar_dataset<double>(file, "time_state/E_numerical_loss", H5T_NATIVE_DOUBLE, 0.0);
  out.state.E_laser_deposited =
      read_scalar_dataset<double>(file, "time_state/E_laser_deposited", H5T_NATIVE_DOUBLE, 0.0);
  out.state.E_cbet_iaw_step =
      read_scalar_dataset<double>(file, "time_state/E_cbet_iaw_step", H5T_NATIVE_DOUBLE, 0.0);
  out.state.E_cbet_iaw =
      read_scalar_dataset<double>(file, "time_state/E_cbet_iaw", H5T_NATIVE_DOUBLE, 0.0);
  out.state.E_burn_released =
      read_scalar_dataset<double>(file, "time_state/E_burn_released", H5T_NATIVE_DOUBLE, 0.0);
  out.state.E_burn_dep_e =
      read_scalar_dataset<double>(file, "time_state/E_burn_dep_e", H5T_NATIVE_DOUBLE, 0.0);
  out.state.E_burn_dep_i =
      read_scalar_dataset<double>(file, "time_state/E_burn_dep_i", H5T_NATIVE_DOUBLE, 0.0);
  out.state.E_burn_esc_charged =
      read_scalar_dataset<double>(file, "time_state/E_burn_esc_charged", H5T_NATIVE_DOUBLE, 0.0);
  out.state.E_burn_esc_neutron =
      read_scalar_dataset<double>(file, "time_state/E_burn_esc_neutron", H5T_NATIVE_DOUBLE, 0.0);
  out.state.N_burn_neutrons_dt =
      read_scalar_dataset<double>(file, "time_state/N_burn_neutrons_dt", H5T_NATIVE_DOUBLE, 0.0);
  out.state.N_burn_neutrons_dd =
      read_scalar_dataset<double>(file, "time_state/N_burn_neutrons_dd", H5T_NATIVE_DOUBLE, 0.0);
  out.state.E_burn_inflight =
      read_scalar_dataset<double>(file, "time_state/E_burn_inflight", H5T_NATIVE_DOUBLE, 0.0);
  out.state.burn_mc_live =
      static_cast<int>(read_scalar_dataset<std::int32_t>(
          file,
          "time_state/burn_mc_live",
          H5T_NATIVE_INT32,
          static_cast<std::int32_t>(out.state.burn_mc_live)));
  out.state.E_laser_escaped =
      read_scalar_dataset<double>(file, "time_state/E_laser_escaped", H5T_NATIVE_DOUBLE, 0.0);
  out.state.E_laser_incident =
      read_scalar_dataset<double>(file, "time_state/E_laser_incident", H5T_NATIVE_DOUBLE, 0.0);
  out.state.E_ra_deposited =
      read_scalar_dataset<double>(file, "time_state/E_ra_deposited", H5T_NATIVE_DOUBLE, 0.0);
  out.state.E_rad_escaped =
      read_scalar_dataset<double>(file, "time_state/E_rad_escaped", H5T_NATIVE_DOUBLE, 0.0);
  out.state.E_floor_injected =
      read_scalar_dataset<double>(file, "time_state/E_floor_injected", H5T_NATIVE_DOUBLE, 0.0);
  out.state.E_pdV_bdry =
      read_scalar_dataset<double>(file, "time_state/E_pdV_bdry", H5T_NATIVE_DOUBLE, 0.0);
  out.state.E_Marshak_in =
      read_scalar_dataset<double>(file, "time_state/E_Marshak_in", H5T_NATIVE_DOUBLE, 0.0);
  out.state.E_solver =
      read_scalar_dataset<double>(file, "time_state/E_solver", H5T_NATIVE_DOUBLE, 0.0);

  initialize_output_timing(out.state, cfg);
  out.state.t_next_plot = read_scalar_dataset<double>(
      file, "output_state/t_next_plot", H5T_NATIVE_DOUBLE, out.state.t_next_plot);
  out.state.t_next_history = read_scalar_dataset<double>(
      file, "output_state/t_next_history", H5T_NATIVE_DOUBLE, out.state.t_next_history);
  out.state.t_next_checkpoint = read_scalar_dataset<double>(
      file, "output_state/t_next_checkpoint", H5T_NATIVE_DOUBLE, out.state.t_next_checkpoint);

  std::vector<std::int8_t> ddmc_map_from_particles;
  restore_particles(file, out.photon_pool, &ddmc_map_from_particles, out.state.rho.size(), n_groups);
  if (!has_checkpoint_ddmc_map) {
    if (!ddmc_map_from_particles.empty() && out.photon_pool.n_alive > 0) {
      out.state.ddmc_mode_map = std::move(ddmc_map_from_particles);
      out.state.ddmc_mode_map_valid = true;
      core::log_warning(
          "Restart checkpoint missing radiation/ddmc_flag; reconstructed mode map from particle "
          "states.");
    } else {
      out.state.ddmc_mode_map_valid = false;
      core::log_warning(
          "Restart checkpoint missing radiation/ddmc_flag; defaulting mode map to IMC "
          "(recomputed during transport setup).");
    }
  }

  warn_h5_close_failure(H5Fclose(file), "H5Fclose", "HDF5Reader::read_checkpoint");

  out.state.mesh.recompute_geometry();
#else
  (void)cfg;
  (void)checkpoint_prefix;
  (void)rank;
  TENRYU_ASSERT(false, "Restart requires TENRYU_ENABLE_HDF5=ON");
#endif

  return out;
}

}  // namespace tenryu::io
