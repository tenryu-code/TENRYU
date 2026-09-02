#include "io/hdf5_writer.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "core/error.hpp"
#include "diagnostics/diagnostics.hpp"
#include "hydro/euler_window_blend.hpp"
#include "hydro/oriented_swept_volume.cuh"
#include "hydro/rz_corner_mass.cuh"
#include "mesh/mesh.hpp"

#if TENRYU_ENABLE_HDF5
#include "io/hdf5_utils.hpp"
#endif

namespace tenryu::io {
namespace {

constexpr int kSchemaVersion = 1;

std::string format_step(const int step) {
  std::ostringstream oss;
  oss << std::setw(4) << std::setfill('0') << step;
  return oss.str();
}

std::string format_rank(const int rank) {
  std::ostringstream oss;
  oss << std::setw(4) << std::setfill('0') << rank;
  return oss.str();
}

std::string safe_case_name(const std::string& case_name) {
  std::string safe = case_name.empty() ? std::string("unnamed") : case_name;
  std::replace(safe.begin(), safe.end(), '/', '_');
  std::replace(safe.begin(), safe.end(), '\\', '_');
  return safe;
}

std::string read_text_if_exists(const std::filesystem::path& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.good()) {
    return {};
  }
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

std::string detect_build_type() {
#if defined(NDEBUG)
  return "Release";
#else
  return "Debug";
#endif
}

std::string detect_gpu_name() {
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    return "unknown";
  }
  return prop.name;
}

std::string checkpoint_config_signature(const core::Config& cfg) {
  // Legacy compact signature fallback used only when canonical JSON is unavailable.
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

std::string solver_requested(const core::Config& cfg) {
  const auto& fld = cfg.radiation.multigroup_diffusion;
  return fld.linear_solver_2d_requested.empty() ? fld.linear_solver_2d
                                                : fld.linear_solver_2d_requested;
}

std::string solver_resolved(const core::Config& cfg) {
  const auto& fld = cfg.radiation.multigroup_diffusion;
  return fld.linear_solver_2d_resolved.empty() ? fld.linear_solver_2d
                                               : fld.linear_solver_2d_resolved;
}

void cuda_check(const cudaError_t err, const std::string_view message) {
  TENRYU_ASSERT(err == cudaSuccess, std::string(message));
}

#if TENRYU_ENABLE_HDF5
void warn_h5_close_failure(herr_t status,
                           const char* op,
                           std::string_view context);
hid_t ensure_group(hid_t file, const std::string& path);
void write_string_attribute(hid_t object,
                            const std::string& name,
                            const std::string& value);
void write_scalar_attribute_i32(hid_t object,
                                const std::string& name,
                                std::int32_t value);
void write_scalar_attribute_i8(hid_t object,
                               const std::string& name,
                               std::int8_t value);
void write_scalar_attribute_u8(hid_t object,
                               const std::string& name,
                               std::uint8_t value);
void write_i32_vector_attribute(hid_t object,
                                const std::string& name,
                                const std::vector<std::int32_t>& values);
void write_string_vector_attribute(hid_t object,
                                   const std::string& name,
                                   const std::vector<std::string>& values);
void write_string_dataset(const hid_t file,
                          const std::string& group_path,
                          const std::string& name,
                          const std::string& value);
template <typename T>
void write_numeric_dataset(hid_t file,
                           const std::string& group_path,
                           const std::string& name,
                           hid_t h5_type,
                           const std::vector<hsize_t>& dims,
                           const T* data,
                           const std::string& units,
                           const core::Config& cfg);

std::optional<std::string> read_string_attribute_if_exists(const hid_t object,
                                                           const char* name) {
  if (H5Aexists(object, name) <= 0) {
    return std::nullopt;
  }
  const hid_t attr = H5Aopen(object, name, H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HDF5 H5Aopen string attribute failed");
  const hid_t type = H5Aget_type(attr);
  TENRYU_ASSERT(type >= 0, "HDF5 H5Aget_type string attribute failed");
  const std::size_t size = H5Tget_size(type);
  std::string value(size, '\0');
  TENRYU_ASSERT(H5Aread(attr, type, value.data()) >= 0,
                "HDF5 H5Aread string attribute failed");
  warn_h5_close_failure(H5Tclose(type), "H5Tclose",
                        "HDF5Writer::read_string_attribute_if_exists(type)");
  warn_h5_close_failure(H5Aclose(attr), "H5Aclose",
                        "HDF5Writer::read_string_attribute_if_exists(attr)");
  if (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  return value;
}

std::optional<std::uint8_t> read_u8_attribute_if_exists(const hid_t object,
                                                        const char* name) {
  if (H5Aexists(object, name) <= 0) {
    return std::nullopt;
  }
  const hid_t attr = H5Aopen(object, name, H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HDF5 H5Aopen u8 attribute failed");
  std::uint8_t value = 0;
  TENRYU_ASSERT(H5Aread(attr, H5T_NATIVE_UINT8, &value) >= 0,
                "HDF5 H5Aread u8 attribute failed");
  warn_h5_close_failure(H5Aclose(attr), "H5Aclose",
                        "HDF5Writer::read_u8_attribute_if_exists");
  return value;
}

std::optional<std::int32_t> read_i32_attribute_if_exists(const hid_t object,
                                                         const char* name) {
  if (H5Aexists(object, name) <= 0) {
    return std::nullopt;
  }
  const hid_t attr = H5Aopen(object, name, H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HDF5 H5Aopen i32 attribute failed");
  std::int32_t value = 0;
  TENRYU_ASSERT(H5Aread(attr, H5T_NATIVE_INT32, &value) >= 0,
                "HDF5 H5Aread i32 attribute failed");
  warn_h5_close_failure(H5Aclose(attr), "H5Aclose",
                        "HDF5Writer::read_i32_attribute_if_exists");
  return value;
}
#endif

template <typename Tag>
std::vector<double> copy_field_to_host(const core::Field1D<Tag>& field) {
  std::vector<double> host(field.size(), 0.0);
  if (!host.empty()) {
    field.copy_to_host(host.data());
  }
  return host;
}

bool recompute_corner_mass_for_output(const core::Config& cfg) {
  const bool invariant_corner_mass =
      cfg.numerics.hydro.subzonal_mass_lagrangian_invariant_enabled ||
      cfg.numerics.hydro.subzonal_mass_enabled ||
      cfg.numerics.hydro.hourglass.enabled;
  return cfg.mesh.motion == "ale" && cfg.numerics.ale.enabled &&
         !invariant_corner_mass;
}

void accumulate_node_mass_corner(std::vector<double>& node_mass,
                                 const int c,
                                 const double m00,
                                 const double m10,
                                 const double m11,
                                 const double m01,
                                 const int nz) {
  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);
  node_mass[static_cast<std::size_t>(n00)] += m00;
  node_mass[static_cast<std::size_t>(n10)] += m10;
  node_mass[static_cast<std::size_t>(n11)] += m11;
  node_mass[static_cast<std::size_t>(n01)] += m01;
}

std::vector<double> compute_node_mass_for_output(const core::State& state,
                                                 const core::Config& cfg) {
  if (cfg.main.dimension != "2D_RZ") {
    return {};
  }
  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    const std::size_t n_cells = static_cast<std::size_t>(state.mesh.topo.n_cells);
    const std::size_t n_nodes = static_cast<std::size_t>(state.mesh.topo.n_nodes);
    const std::size_t corner_stride =
        static_cast<std::size_t>(state.corner_stride);
    const std::size_t expected_corner_entries = corner_stride * n_cells;
    if (!state.mesh.topo.multiblock.has_value() ||
        state.corner_mass.size() != expected_corner_entries ||
        state.mesh.topo.multiblock->cell_node_csr_offsets.size() != n_cells + 1U ||
        state.mesh.topo.multiblock->cell_node_csr_indices.size() !=
            expected_corner_entries) {
      core::log_warning(
          "HDF5 write: hydro/node_mass multiblock size unexpected -- got "
          + std::to_string(state.corner_mass.size()) + " corner entries, expected "
          + std::to_string(expected_corner_entries));
      return {};
    }

    const auto& mb = *state.mesh.topo.multiblock;
    const auto corner_mass = copy_field_to_host(state.corner_mass);
    std::vector<double> node_mass(n_nodes, 0.0);
    for (std::size_t c = 0; c < n_cells; ++c) {
      const int off = mb.cell_node_csr_offsets[c];
      const int active_nverts = mesh::mesh_topo_cell_active_nverts(
          state.mesh.cell_nverts, static_cast<int>(c));
      const int corner_count = active_nverts >= 5 ? active_nverts : 4;
      for (int q = 0; q < corner_count; ++q) {
        const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + q)];
        node_mass[static_cast<std::size_t>(n)] +=
            corner_mass[c * corner_stride + static_cast<std::size_t>(q)];
      }
    }
    return node_mass;
  }

  const int nr = cfg.mesh.nr;
  const int nz = cfg.mesh.nz;
  if (nr <= 0 || nz <= 0) {
    return {};
  }
  const std::size_t n_cells = static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz);
  const std::size_t n_nodes =
      static_cast<std::size_t>(nr + 1) * static_cast<std::size_t>(nz + 1);
  if (state.mass.size() != n_cells ||
      state.x_r.size() != n_nodes ||
      state.x_z.size() != n_nodes) {
    core::log_warning("HDF5 write: hydro/node_mass skipped due to mesh/state size mismatch.");
    return {};
  }

  std::vector<double> node_mass(n_nodes, 0.0);
  const int button_outer_node_ring =
      (state.mesh.button_center && state.mesh.button_center->enabled)
          ? state.mesh.button_center->outer_node_ring
          : 0;
  const bool use_cached_corner_mass =
      button_outer_node_ring <= 0 &&
      state.corner_mass.size() == 4U * n_cells &&
      !recompute_corner_mass_for_output(cfg);
  if (use_cached_corner_mass) {
    const auto corner_mass = copy_field_to_host(state.corner_mass);
    for (std::size_t c = 0; c < n_cells; ++c) {
      accumulate_node_mass_corner(node_mass,
                                  static_cast<int>(c),
                                  corner_mass[c * 4U + 0U],
                                  corner_mass[c * 4U + 1U],
                                  corner_mass[c * 4U + 2U],
                                  corner_mass[c * 4U + 3U],
                                  nz);
    }
    return node_mass;
  }

  const auto mass = copy_field_to_host(state.mass);
  const auto x_r = copy_field_to_host(state.x_r);
  const auto x_z = copy_field_to_host(state.x_z);
  const std::uint8_t* cell_nverts =
      state.mesh.cell_nverts.size() == n_cells ? state.mesh.cell_nverts.data() : nullptr;
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int c = i * nz + j;
      const double m_cell = mass[static_cast<std::size_t>(c)];
      if (button_outer_node_ring > 0 && c == 0) {
        const double volume = hydro::rz::button_polygon_volume_from_nodes(
            x_r.data(), x_z.data(), button_outer_node_ring, nz);
        if (volume > 0.0 && std::isfinite(volume)) {
          double centroid_r = 0.0;
          double centroid_z = 0.0;
          hydro::rz::button_polygon_area_centroid_from_nodes(
              x_r.data(), x_z.data(), button_outer_node_ring, nz,
              &centroid_r, &centroid_z);
          const double rho_button = std::max(m_cell, 0.0) / volume;
          for (int k = 0; k <= nz; ++k) {
            const int n = hydro::rz::button_seam_node_index(
                button_outer_node_ring, k, nz);
            const double m_node =
                hydro::rz::button_corner_mass_exact_subpolygon(
                    rho_button, x_r.data(), x_z.data(),
                    button_outer_node_ring, nz, k, centroid_r, centroid_z);
            node_mass[static_cast<std::size_t>(n)] +=
                std::max(m_node, 0.0);
          }
        }
        continue;
      }
      double m_corner[4] = {0.0, 0.0, 0.0, 0.0};
      hydro::rz::compute_rz_corner_masses_from_nodes(
          c, nz, m_cell, x_r.data(), x_z.data(), cell_nverts, m_corner,
          nullptr,
          static_cast<int>(cfg.numerics.hydro.corner_mass_convention));
      accumulate_node_mass_corner(node_mass, c, m_corner[0], m_corner[1],
                                  m_corner[2], m_corner[3], nz);
    }
  }
  return node_mass;
}

template <typename T>
std::vector<T> copy_device_array(const T* ptr,
                                 const std::size_t count,
                                 const std::string_view label) {
  std::vector<T> host(count);
  if (count == 0) {
    return host;
  }
  TENRYU_ASSERT(ptr != nullptr, std::string(label) + " pointer is null");
  cuda_check(cudaMemcpy(host.data(),
                        ptr,
                        sizeof(T) * count,
                        cudaMemcpyDeviceToHost),
             std::string(label) + " cudaMemcpy D2H failed");
  return host;
}

std::vector<double> group_bounds_from_config(const core::Config& cfg) {
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

std::vector<std::int32_t> compute_cell_material_id(const core::State& state,
                                                   const core::Config& cfg) {
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_mat = cfg.materials.materials.size();
  std::vector<std::int32_t> ids(n_cells, 0);
  if (n_cells == 0 || n_mat == 0 || state.volFrac.size() != n_cells * n_mat) {
    if (n_cells > 0 && n_mat > 0 && state.volFrac.size() != n_cells * n_mat) {
      static bool warned_volfrac_size_mismatch = false;
      if (!warned_volfrac_size_mismatch) {
        core::log_warning("HDF5 writer: volFrac size mismatch in cell_material_id export "
                          "(volFrac=" + std::to_string(state.volFrac.size()) +
                          ", expected=" + std::to_string(n_cells * n_mat) +
                          "); writing fallback material IDs.");
        warned_volfrac_size_mismatch = true;
      }
    }
    return ids;
  }

  const auto vol_frac = copy_field_to_host(state.volFrac);
  for (std::size_t c = 0; c < n_cells; ++c) {
    std::size_t best = 0;
    double best_val = vol_frac[c * n_mat];
    for (std::size_t m = 1; m < n_mat; ++m) {
      const double value = vol_frac[c * n_mat + m];
      if (value > best_val) {
        best_val = value;
        best = m;
      }
    }
    ids[c] = static_cast<std::int32_t>(best);
  }
  return ids;
}

bool is_table_eos_model(const std::string& eos_model) {
  return eos_model == "sesame" || eos_model == "ionmix" || eos_model == "tmat";
}

std::size_t dominant_material_index_for_output(const core::State& state,
                                               const core::Config& cfg) {
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_mat = cfg.materials.materials.size();
  if (n_cells == 0 || n_mat == 0 || state.volFrac.size() != n_cells * n_mat) {
    return 0;
  }
  const auto vol_frac = copy_field_to_host(state.volFrac);
  std::vector<double> sums(n_mat, 0.0);
  for (std::size_t c = 0; c < n_cells; ++c) {
    for (std::size_t m = 0; m < n_mat; ++m) {
      sums[m] += vol_frac[c * n_mat + m];
    }
  }
  std::size_t best = 0;
  for (std::size_t m = 1; m < n_mat; ++m) {
    if (sums[m] > sums[best]) {
      best = m;
    }
  }
  return best;
}

std::string per_material_schema_eos_method(const core::State& state,
                                           const core::Config& cfg) {
  if (cfg.materials.materials.empty()) {
    return "ideal_gas";
  }
  const std::size_t dominant =
      std::min(dominant_material_index_for_output(state, cfg),
               cfg.materials.materials.size() - 1U);
  return is_table_eos_model(cfg.materials.materials[dominant].eos_model)
             ? "table"
             : "ideal_gas";
}

double ideal_species_cv_for_output(const double zbar,
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

double material_zbar_for_output(const core::Config& cfg, const std::size_t m) {
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

std::vector<double> resize_payload(std::vector<double> values,
                                   const std::size_t expected,
                                   const std::string& label,
                                   const bool warn = true) {
  if (values.size() != expected) {
    if (warn) {
      core::log_warning("HDF5 writer: " + label + " size mismatch (state=" +
                        std::to_string(values.size()) + ", expected=" +
                        std::to_string(expected) + "); resizing with zero fill.");
    }
    values.resize(expected, 0.0);
  }
  return values;
}

bool should_mark_derived_projection_when_wave_f_enabled(
    const std::string& group_path,
    const std::string& name,
    const core::Config& cfg) {
  if (!cfg.numerics.materials.per_material_conservation_enabled ||
      group_path != "hydro") {
    return false;
  }
  return name == "rho" || name == "ee" || name == "ei" || name == "Te" ||
         name == "Ti" || name == "Pe" || name == "Pi" || name == "zbar" ||
         name == "cv_e" || name == "cv_i" || name == "cs";
}

#if TENRYU_ENABLE_HDF5
void write_per_material_schema_group(const hid_t file,
                                     const core::State& state,
                                     const core::Config& cfg) {
  if (!cfg.numerics.materials.per_material_conservation_enabled) {
    return;
  }

  const hid_t group = ensure_group(file, "hydro/per_material/v1");
  TENRYU_ASSERT(group >= 0, "HDF5 failed to open hydro/per_material/v1 group");

  const std::size_t n_mat = cfg.materials.materials.size();
  std::vector<std::string> material_names;
  std::vector<std::int32_t> material_ids;
  material_names.reserve(n_mat);
  material_ids.reserve(n_mat);
  for (std::size_t m = 0; m < n_mat; ++m) {
    material_names.push_back(cfg.materials.materials[m].name);
    material_ids.push_back(static_cast<std::int32_t>(m));
  }

  write_scalar_attribute_u8(group, "enabled", static_cast<std::uint8_t>(1));
  write_scalar_attribute_i32(group, "n_mat", static_cast<std::int32_t>(n_mat));
  write_string_attribute(group, "eos_method", per_material_schema_eos_method(state, cfg));
  write_scalar_attribute_i32(group, "schema_version", 1);
  write_string_attribute(group, "conserved_basis", "extensive");
  write_string_attribute(group, "layout", "cell_major_ncells_nmat");
  write_string_vector_attribute(group, "material_names", material_names);
  write_i32_vector_attribute(group, "material_ids", material_ids);

  warn_h5_close_failure(H5Gclose(group), "H5Gclose",
                        "HDF5Writer::write_per_material_schema_group(group)");

  const std::size_t n_cells = state.rho.size();
  const std::size_t n_cell_mat = n_cells * n_mat;
  const std::vector<hsize_t> cm_dims = {
      static_cast<hsize_t>(n_cells), static_cast<hsize_t>(n_mat)};
  const double zero = 0.0;

  auto mass_m = resize_payload(copy_field_to_host(state.mass_per_material),
                               n_cell_mat,
                               "mass_per_material");
  auto Ee_m = resize_payload(copy_field_to_host(state.Ee_per_material),
                             n_cell_mat,
                             "Ee_per_material");
  auto Ei_m = resize_payload(copy_field_to_host(state.Ei_per_material),
                             n_cell_mat,
                             "Ei_per_material");
  const double* mass_ptr = mass_m.empty() ? &zero : mass_m.data();
  const double* Ee_ptr = Ee_m.empty() ? &zero : Ee_m.data();
  const double* Ei_ptr = Ei_m.empty() ? &zero : Ei_m.data();
  write_numeric_dataset(file,
                        "hydro/per_material/v1",
                        "mass",
                        H5T_NATIVE_DOUBLE,
                        cm_dims,
                        mass_ptr,
                        "g",
                        cfg);
  write_numeric_dataset(file,
                        "hydro/per_material/v1",
                        "Ee",
                        H5T_NATIVE_DOUBLE,
                        cm_dims,
                        Ee_ptr,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "hydro/per_material/v1",
                        "Ei",
                        H5T_NATIVE_DOUBLE,
                        cm_dims,
                        Ei_ptr,
                        "erg",
                        cfg);

  if (!cfg.numerics.materials.hdf5_emit_derived_per_material) {
    return;
  }

  const auto vol = resize_payload(copy_field_to_host(state.vol), n_cells, "vol");
  const auto volfrac =
      resize_payload(copy_field_to_host(state.volFrac), n_cell_mat, "volFrac");
  const auto cell_Te = resize_payload(copy_field_to_host(state.Te), n_cells, "Te");
  const auto cell_Ti = resize_payload(copy_field_to_host(state.Ti), n_cells, "Ti");
  const auto cache_Te = copy_field_to_host(state.Te_per_material);
  const auto cache_Ti = copy_field_to_host(state.Ti_per_material);

  std::vector<double> rho_derived(n_cell_mat, 0.0);
  std::vector<double> ee_derived(n_cell_mat, 0.0);
  std::vector<double> ei_derived(n_cell_mat, 0.0);
  std::vector<double> Te_derived(n_cell_mat, 0.0);
  std::vector<double> Ti_derived(n_cell_mat, 0.0);
  std::vector<double> Pe_derived(n_cell_mat, 0.0);
  const bool have_cache_Te = cache_Te.size() == n_cell_mat;
  const bool have_cache_Ti = cache_Ti.size() == n_cell_mat;
  for (std::size_t c = 0; c < n_cells; ++c) {
    const double V = vol[c];
    for (std::size_t m = 0; m < n_mat; ++m) {
      const std::size_t idx = c * n_mat + m;
      const double vf = volfrac[idx];
      const double M = mass_m[idx];
      if (vf > 0.0 && V > 0.0 && M > 0.0 && std::isfinite(vf) &&
          std::isfinite(V) && std::isfinite(M)) {
        rho_derived[idx] = M / (vf * V);
      }
      if (M > 0.0 && std::isfinite(M)) {
        ee_derived[idx] = Ee_m[idx] / M;
        ei_derived[idx] = Ei_m[idx] / M;
      }
      const auto& mat = cfg.materials.materials[m];
      const double gamma = (mat.ideal_gas_gamma > 1.0) ? mat.ideal_gas_gamma
                                                       : (5.0 / 3.0);
      const double A = (mat.A > 0.0) ? mat.A : 1.0;
      const double zbar = material_zbar_for_output(cfg, m);
      const double cv_e = ideal_species_cv_for_output(zbar, A, gamma);
      const double cv_i = ideal_species_cv_for_output(1.0, A, gamma);
      Te_derived[idx] =
          have_cache_Te ? cache_Te[idx]
                        : ((cv_e > 0.0) ? (ee_derived[idx] / cv_e) : cell_Te[c]);
      Ti_derived[idx] =
          have_cache_Ti ? cache_Ti[idx]
                        : ((cv_i > 0.0) ? (ei_derived[idx] / cv_i) : cell_Ti[c]);
      Pe_derived[idx] =
          (rho_derived[idx] > 0.0) ? ((gamma - 1.0) * rho_derived[idx] * ee_derived[idx])
                                   : 0.0;
    }
  }

  write_numeric_dataset(file,
                        "hydro/per_material/v1",
                        "rho_derived",
                        H5T_NATIVE_DOUBLE,
                        cm_dims,
                        rho_derived.empty() ? &zero : rho_derived.data(),
                        "g/cm3",
                        cfg);
  write_numeric_dataset(file,
                        "hydro/per_material/v1",
                        "ee_derived",
                        H5T_NATIVE_DOUBLE,
                        cm_dims,
                        ee_derived.empty() ? &zero : ee_derived.data(),
                        "erg/g",
                        cfg);
  write_numeric_dataset(file,
                        "hydro/per_material/v1",
                        "ei_derived",
                        H5T_NATIVE_DOUBLE,
                        cm_dims,
                        ei_derived.empty() ? &zero : ei_derived.data(),
                        "erg/g",
                        cfg);
  write_numeric_dataset(file,
                        "hydro/per_material/v1",
                        "Te_derived",
                        H5T_NATIVE_DOUBLE,
                        cm_dims,
                        Te_derived.empty() ? &zero : Te_derived.data(),
                        "eV",
                        cfg);
  write_numeric_dataset(file,
                        "hydro/per_material/v1",
                        "Ti_derived",
                        H5T_NATIVE_DOUBLE,
                        cm_dims,
                        Ti_derived.empty() ? &zero : Ti_derived.data(),
                        "eV",
                        cfg);
  write_numeric_dataset(file,
                        "hydro/per_material/v1",
                        "Pe_derived",
                        H5T_NATIVE_DOUBLE,
                        cm_dims,
                        Pe_derived.empty() ? &zero : Pe_derived.data(),
                        "dyne/cm2",
                        cfg);
}
#endif

std::vector<double> compute_power_density(const std::vector<double>& energy,
                                          const std::vector<double>& vol,
                                          const std::size_t cells,
                                          const std::size_t groups,
                                          const double dt) {
  std::vector<double> out(energy.size(), 0.0);
  const double dt_safe = (dt > 0.0) ? dt : 1.0;
  for (std::size_t c = 0; c < cells; ++c) {
    const double v = (c < vol.size()) ? std::max(vol[c], 1.0e-30) : 1.0;
    for (std::size_t g = 0; g < groups; ++g) {
      const std::size_t idx = c * groups + g;
      if (idx >= energy.size()) {
        break;
      }
      out[idx] = energy[idx] / (v * dt_safe);
    }
  }
  return out;
}

std::vector<double> compute_laser_power_density(const std::vector<double>& dep,
                                                const std::vector<double>& vol,
                                                const double dt) {
  std::vector<double> out(dep.size(), 0.0);
  const double dt_safe = (dt > 0.0) ? dt : 1.0;
  for (std::size_t c = 0; c < dep.size(); ++c) {
    const double v = (c < vol.size()) ? std::max(vol[c], 1.0e-30) : 1.0;
    out[c] = dep[c] / (v * dt_safe);
  }
  return out;
}

void compute_aq_shells_2d_rz(const std::vector<double>& dep,
                             const int nr,
                             const int nz,
                             std::vector<double>& aq_r_shells,
                             std::vector<double>& aq_z_shells) {
  aq_r_shells.assign(static_cast<std::size_t>(std::max(nr, 0)), 0.0);
  aq_z_shells.assign(static_cast<std::size_t>(std::max(nz, 0)), 0.0);
  if (nr <= 0 || nz <= 0 ||
      dep.size() < static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz)) {
    return;
  }

  for (int i = 0; i < nr; ++i) {
    double q_min = std::numeric_limits<double>::infinity();
    double q_max = 0.0;
    for (int j = 0; j < nz; ++j) {
      const double raw = dep[static_cast<std::size_t>(i * nz + j)];
      const double q = (std::isfinite(raw) && raw > 0.0) ? raw : 0.0;
      q_min = std::min(q_min, q);
      q_max = std::max(q_max, q);
    }
    const double denom = q_max + q_min;
    aq_r_shells[static_cast<std::size_t>(i)] =
        (denom > 0.0) ? ((q_max - q_min) / denom) : 0.0;
  }

  for (int j = 0; j < nz; ++j) {
    double q_min = std::numeric_limits<double>::infinity();
    double q_max = 0.0;
    for (int i = 0; i < nr; ++i) {
      const double raw = dep[static_cast<std::size_t>(i * nz + j)];
      const double q = (std::isfinite(raw) && raw > 0.0) ? raw : 0.0;
      q_min = std::min(q_min, q);
      q_max = std::max(q_max, q);
    }
    const double denom = q_max + q_min;
    aq_z_shells[static_cast<std::size_t>(j)] =
        (denom > 0.0) ? ((q_max - q_min) / denom) : 0.0;
  }
}

#if TENRYU_ENABLE_HDF5

void warn_h5_close_failure(const herr_t status,
                           const char* op,
                           const std::string_view context) {
  if (status < 0) {
    core::log_warning(std::string("[WARN] ") + op + " failed in " +
                      std::string(context));
  }
}

hid_t ensure_group(const hid_t file, const std::string& path) {
  if (path.empty() || path == "/") {
    return H5Gopen2(file, "/", H5P_DEFAULT);
  }

  hid_t current = H5Gopen2(file, "/", H5P_DEFAULT);
  TENRYU_ASSERT(current >= 0, "HDF5 ensure_group failed to open root group");

  std::size_t pos = 0;
  while (pos < path.size()) {
    while (pos < path.size() && path[pos] == '/') {
      ++pos;
    }
    if (pos >= path.size()) {
      break;
    }
    std::size_t next = path.find('/', pos);
    if (next == std::string::npos) {
      next = path.size();
    }
    const std::string part = path.substr(pos, next - pos);

    hid_t child = -1;
    if (tenryu::io::h5_link_exists(current, part.c_str(), H5P_DEFAULT) > 0) {
      child = H5Gopen2(current, part.c_str(), H5P_DEFAULT);
    } else {
      child = H5Gcreate2(current, part.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    }
    TENRYU_ASSERT(child >= 0, "HDF5 ensure_group failed for '" + part + "'");
    warn_h5_close_failure(H5Gclose(current), "H5Gclose", "HDF5Writer::ensure_group(current)");
    current = child;
    pos = next;
  }

  return current;
}

void write_string_attribute(const hid_t object,
                            const std::string& name,
                            const std::string& value) {
  const hid_t type = H5Tcopy(H5T_C_S1);
  TENRYU_ASSERT(type >= 0, "HDF5 H5Tcopy for string attribute failed");
  const std::size_t len = std::max<std::size_t>(1, value.size());
  TENRYU_ASSERT(H5Tset_size(type, len) >= 0,
                "HDF5 H5Tset_size for string attribute failed");
  const hid_t space = H5Screate(H5S_SCALAR);
  TENRYU_ASSERT(space >= 0, "HDF5 H5Screate scalar attribute failed");
  const hid_t attr = H5Acreate2(object, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HDF5 H5Acreate2 string attribute failed: " + name);
  TENRYU_ASSERT(H5Awrite(attr, type, value.c_str()) >= 0,
                "HDF5 H5Awrite string attribute failed: " + name);
  H5Aclose(attr);
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HDF5Writer::write_string_attribute(space)");
  warn_h5_close_failure(H5Tclose(type), "H5Tclose",
                        "HDF5Writer::write_string_attribute(type)");
}

void write_scalar_attribute_i64(const hid_t object,
                                const std::string& name,
                                const std::int64_t value) {
  const hid_t space = H5Screate(H5S_SCALAR);
  TENRYU_ASSERT(space >= 0, "HDF5 H5Screate scalar i64 attribute failed");
  const hid_t attr =
      H5Acreate2(object, name.c_str(), H5T_NATIVE_INT64, space, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HDF5 H5Acreate2 i64 attribute failed: " + name);
  TENRYU_ASSERT(H5Awrite(attr, H5T_NATIVE_INT64, &value) >= 0,
                "HDF5 H5Awrite i64 attribute failed: " + name);
  H5Aclose(attr);
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HDF5Writer::write_scalar_attribute_i64(space)");
}

void write_scalar_attribute_i32(const hid_t object,
                                const std::string& name,
                                const std::int32_t value) {
  const hid_t space = H5Screate(H5S_SCALAR);
  TENRYU_ASSERT(space >= 0, "HDF5 H5Screate scalar i32 attribute failed");
  const hid_t attr =
      H5Acreate2(object, name.c_str(), H5T_NATIVE_INT32, space, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HDF5 H5Acreate2 i32 attribute failed: " + name);
  TENRYU_ASSERT(H5Awrite(attr, H5T_NATIVE_INT32, &value) >= 0,
                "HDF5 H5Awrite i32 attribute failed: " + name);
  H5Aclose(attr);
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HDF5Writer::write_scalar_attribute_i32(space)");
}

void write_scalar_attribute_i8(const hid_t object,
                               const std::string& name,
                               const std::int8_t value) {
  const hid_t space = H5Screate(H5S_SCALAR);
  TENRYU_ASSERT(space >= 0, "HDF5 H5Screate scalar i8 attribute failed");
  const hid_t attr =
      H5Acreate2(object, name.c_str(), H5T_NATIVE_INT8, space, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HDF5 H5Acreate2 i8 attribute failed: " + name);
  TENRYU_ASSERT(H5Awrite(attr, H5T_NATIVE_INT8, &value) >= 0,
                "HDF5 H5Awrite i8 attribute failed: " + name);
  H5Aclose(attr);
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HDF5Writer::write_scalar_attribute_i8(space)");
}

void write_scalar_attribute_u64(const hid_t object,
                                const std::string& name,
                                const std::uint64_t value) {
  const hid_t space = H5Screate(H5S_SCALAR);
  TENRYU_ASSERT(space >= 0, "HDF5 H5Screate scalar u64 attribute failed");
  const hid_t attr =
      H5Acreate2(object, name.c_str(), H5T_NATIVE_UINT64, space, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HDF5 H5Acreate2 u64 attribute failed: " + name);
  TENRYU_ASSERT(H5Awrite(attr, H5T_NATIVE_UINT64, &value) >= 0,
                "HDF5 H5Awrite u64 attribute failed: " + name);
  H5Aclose(attr);
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HDF5Writer::write_scalar_attribute_u64(space)");
}

void write_scalar_attribute_u8(const hid_t object,
                               const std::string& name,
                               const std::uint8_t value) {
  const hid_t space = H5Screate(H5S_SCALAR);
  TENRYU_ASSERT(space >= 0, "HDF5 H5Screate scalar u8 attribute failed");
  const hid_t attr =
      H5Acreate2(object, name.c_str(), H5T_NATIVE_UINT8, space, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HDF5 H5Acreate2 u8 attribute failed: " + name);
  TENRYU_ASSERT(H5Awrite(attr, H5T_NATIVE_UINT8, &value) >= 0,
                "HDF5 H5Awrite u8 attribute failed: " + name);
  H5Aclose(attr);
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HDF5Writer::write_scalar_attribute_u8(space)");
}

void write_scalar_attribute_f64(const hid_t object,
                                const std::string& name,
                                const double value) {
  const hid_t space = H5Screate(H5S_SCALAR);
  TENRYU_ASSERT(space >= 0, "HDF5 H5Screate scalar f64 attribute failed");
  const hid_t attr =
      H5Acreate2(object, name.c_str(), H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HDF5 H5Acreate2 f64 attribute failed: " + name);
  TENRYU_ASSERT(H5Awrite(attr, H5T_NATIVE_DOUBLE, &value) >= 0,
                "HDF5 H5Awrite f64 attribute failed: " + name);
  H5Aclose(attr);
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HDF5Writer::write_scalar_attribute_f64(space)");
}

void write_i32_vector_attribute(const hid_t object,
                                const std::string& name,
                                const std::vector<std::int32_t>& values) {
  const hsize_t dim = static_cast<hsize_t>(values.size());
  const hid_t space = H5Screate_simple(1, &dim, nullptr);
  TENRYU_ASSERT(space >= 0, "HDF5 H5Screate i32 vector attribute failed");
  const hid_t attr =
      H5Acreate2(object, name.c_str(), H5T_NATIVE_INT32, space, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HDF5 H5Acreate2 i32 vector attribute failed: " + name);
  TENRYU_ASSERT(H5Awrite(attr, H5T_NATIVE_INT32, values.data()) >= 0,
                "HDF5 H5Awrite i32 vector attribute failed: " + name);
  H5Aclose(attr);
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HDF5Writer::write_i32_vector_attribute(space)");
}

void write_string_vector_attribute(const hid_t object,
                                   const std::string& name,
                                   const std::vector<std::string>& values) {
  std::size_t max_len = 1;
  for (const auto& value : values) {
    max_len = std::max(max_len, value.size() + 1U);
  }
  std::vector<char> payload(values.size() * max_len, '\0');
  for (std::size_t i = 0; i < values.size(); ++i) {
    std::copy(values[i].begin(), values[i].end(), payload.begin() +
              static_cast<std::ptrdiff_t>(i * max_len));
  }

  const hsize_t dim = static_cast<hsize_t>(values.size());
  const hid_t space = H5Screate_simple(1, &dim, nullptr);
  TENRYU_ASSERT(space >= 0, "HDF5 H5Screate string vector attribute failed");
  const hid_t type = H5Tcopy(H5T_C_S1);
  TENRYU_ASSERT(type >= 0, "HDF5 H5Tcopy string vector attribute failed");
  TENRYU_ASSERT(H5Tset_size(type, max_len) >= 0,
                "HDF5 H5Tset_size string vector attribute failed");
  const hid_t attr =
      H5Acreate2(object, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HDF5 H5Acreate2 string vector attribute failed: " + name);
  TENRYU_ASSERT(H5Awrite(attr, type, payload.data()) >= 0,
                "HDF5 H5Awrite string vector attribute failed: " + name);
  H5Aclose(attr);
  warn_h5_close_failure(H5Tclose(type), "H5Tclose",
                        "HDF5Writer::write_string_vector_attribute(type)");
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HDF5Writer::write_string_vector_attribute(space)");
}

void write_units_attribute(const hid_t dset, const std::string& units) {
  if (units.empty()) {
    return;
  }
  write_string_attribute(dset, "units", units);
}

hid_t make_dcpl(const core::Config& cfg,
                const int rank,
                const std::vector<hsize_t>& dims) {
  const hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  TENRYU_ASSERT(dcpl >= 0, "HDF5 H5Pcreate dataset create failed");

  const bool skip_filters_for_empty_dataset =
      (cfg.mesh.topology_scheme ==
           core::TopologyScheme::PENTAGON_BELT_SHELL ||
       cfg.numerics.ale.mesh_mode == "reale_v2") &&
      std::any_of(dims.begin(), dims.end(),
                  [](const hsize_t dim) { return dim == 0; });
  if (rank > 0 && !skip_filters_for_empty_dataset) {
    std::vector<hsize_t> chunk(static_cast<std::size_t>(rank), 1);
    for (int i = 0; i < rank; ++i) {
      chunk[static_cast<std::size_t>(i)] = std::max<hsize_t>(1, dims[static_cast<std::size_t>(i)]);
    }
    TENRYU_ASSERT(H5Pset_chunk(dcpl, rank, chunk.data()) >= 0,
                  "HDF5 H5Pset_chunk failed");
  }

  if (cfg.output.compression == "gzip" &&
      cfg.output.compression_level > 0 && rank > 0 &&
      !skip_filters_for_empty_dataset) {
    const int level = std::clamp(cfg.output.compression_level, 0, 9);
    TENRYU_ASSERT(H5Pset_deflate(dcpl, static_cast<unsigned int>(level)) >= 0,
                  "HDF5 H5Pset_deflate failed");
  }

  return dcpl;
}

template <typename T>
void write_numeric_dataset(const hid_t file,
                           const std::string& group_path,
                           const std::string& name,
                           const hid_t h5_type,
                           const std::vector<hsize_t>& dims,
                           const T* data,
                           const std::string& units,
                           const core::Config& cfg) {
  const hid_t group = ensure_group(file, group_path);
  TENRYU_ASSERT(group >= 0, "HDF5 failed to open group: " + group_path);

  const int rank = static_cast<int>(dims.size());
  const hid_t space = (rank == 0)
                          ? H5Screate(H5S_SCALAR)
                          : H5Screate_simple(rank, dims.data(), nullptr);
  TENRYU_ASSERT(space >= 0, "HDF5 failed to create dataspace for " + group_path + "/" + name);

  const hid_t dcpl = make_dcpl(cfg, rank, dims);
  const hid_t dset =
      H5Dcreate2(group, name.c_str(), h5_type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
  TENRYU_ASSERT(dset >= 0, "HDF5 failed to create dataset: " + group_path + "/" + name);

  TENRYU_ASSERT(H5Dwrite(dset, h5_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) >= 0,
                "HDF5 failed to write dataset: " + group_path + "/" + name);
  write_units_attribute(dset, units);
  if (should_mark_derived_projection_when_wave_f_enabled(group_path, name, cfg)) {
    write_scalar_attribute_u8(
        dset, "derived_projection_when_per_material_enabled", static_cast<std::uint8_t>(1));
  }

  warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                        "HDF5Writer::write_numeric_dataset(dataset)");
  warn_h5_close_failure(H5Pclose(dcpl), "H5Pclose",
                        "HDF5Writer::write_numeric_dataset(dcpl)");
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HDF5Writer::write_numeric_dataset(space)");
  warn_h5_close_failure(H5Gclose(group), "H5Gclose",
                        "HDF5Writer::write_numeric_dataset(group)");
}

void write_string_dataset(const hid_t file,
                          const std::string& group_path,
                          const std::string& name,
                          const std::string& value) {
  const hid_t group = ensure_group(file, group_path);
  TENRYU_ASSERT(group >= 0, "HDF5 failed to open group: " + group_path);

  const hid_t space = H5Screate(H5S_SCALAR);
  TENRYU_ASSERT(space >= 0, "HDF5 failed to create scalar dataspace");

  const hid_t type = H5Tcopy(H5T_C_S1);
  TENRYU_ASSERT(type >= 0, "HDF5 failed to copy string type");
  const std::size_t len = std::max<std::size_t>(1, value.size());
  TENRYU_ASSERT(H5Tset_size(type, len) >= 0, "HDF5 failed to set string size");

  const hid_t dset =
      H5Dcreate2(group, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(dset >= 0, "HDF5 failed to create string dataset: " + name);
  TENRYU_ASSERT(H5Dwrite(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.c_str()) >= 0,
                "HDF5 failed to write string dataset: " + name);

  warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                        "HDF5Writer::write_string_dataset(dataset)");
  warn_h5_close_failure(H5Tclose(type), "H5Tclose",
                        "HDF5Writer::write_string_dataset(type)");
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HDF5Writer::write_string_dataset(space)");
  warn_h5_close_failure(H5Gclose(group), "H5Gclose",
                        "HDF5Writer::write_string_dataset(group)");
}

void write_root_attributes(const hid_t file,
                           const core::State& state,
                           const core::Config& cfg,
                           const int step,
                           const double t,
                           const double dt) {
  const std::int64_t nr_attr = static_cast<std::int64_t>(cfg.mesh.nr);
  const std::int64_t nz_attr =
      static_cast<std::int64_t>((cfg.main.dimension == "2D_RZ") ? cfg.mesh.nz : 1);
  write_scalar_attribute_f64(file, "t", t);
  write_scalar_attribute_f64(file, "dt", dt);
  write_scalar_attribute_i64(file, "cycle", static_cast<std::int64_t>(step));
  write_scalar_attribute_i64(file, "step", static_cast<std::int64_t>(step));
  write_string_attribute(file, "geometry", cfg.main.dimension);
  write_scalar_attribute_i64(file, "nr", nr_attr);
  write_scalar_attribute_i64(file, "nz", nz_attr);
  write_scalar_attribute_i64(file, "n_cells", static_cast<std::int64_t>(state.rho.size()));
  write_scalar_attribute_i64(file, "n_nodes", static_cast<std::int64_t>(state.x_r.size()));
  write_scalar_attribute_i64(file,
                             "n_groups",
                             static_cast<std::int64_t>(std::max(cfg.radiation.groups, 1)));
  write_scalar_attribute_i64(file,
                             "n_materials",
                             static_cast<std::int64_t>(cfg.materials.materials.size()));
  write_scalar_attribute_i32(file, "two_temperature", 1);
  write_scalar_attribute_i32(file, "schema_version", kSchemaVersion);
}

void write_metadata_group(const hid_t file,
                          const core::State& state,
                          const core::Config& cfg,
                          const std::string& output_dir,
                          const std::string& case_name) {
  (void)output_dir;
  (void)case_name;
  const std::string namelist_source =
      cfg.meta.namelist_source_path.empty()
          ? std::string{}
          : read_text_if_exists(std::filesystem::path(cfg.meta.namelist_source_path));
  write_string_dataset(file, "metadata", "namelist_source", namelist_source);

  std::string frozen = cfg.meta.frozen_config_json;
  if (frozen.empty()) {
    frozen = checkpoint_config_signature(cfg);
    static bool warned_missing_canonical_json = false;
    if (!warned_missing_canonical_json) {
      core::log_warning("HDF5 writer: canonical frozen_config JSON is unavailable; "
                        "writing legacy compatibility signature.");
      warned_missing_canonical_json = true;
    }
  }
  write_string_dataset(file, "metadata", "frozen_config", frozen);
  const std::int32_t pentagon_corner_mass_partition_version = 2;
  write_numeric_dataset(file,
                        "metadata",
                        "pentagon_corner_mass_partition_version",
                        H5T_NATIVE_INT32,
                        {},
                        &pentagon_corner_mass_partition_version,
                        "",
                        cfg);
  const std::int32_t ale_swept_sign_epoch = 2;
  write_numeric_dataset(file,
                        "metadata",
                        "ale_swept_sign_epoch",
                        H5T_NATIVE_INT32,
                        {},
                        &ale_swept_sign_epoch,
                        "",
                        cfg);
  const std::int32_t axis_core_released_units =
      hydro::axis_core_released_units();
  write_numeric_dataset(file,
                        "metadata",
                        "axis_core_released_units",
                        H5T_NATIVE_INT32,
                        {},
                        &axis_core_released_units,
                        "",
                        cfg);
  if (std::any_of(state.merge_tombstone.begin(),
                  state.merge_tombstone.end(),
                  [](const std::uint8_t value) { return value != 0U; })) {
    write_numeric_dataset(
        file,
        "metadata",
        "merge_tombstone",
        H5T_NATIVE_UINT8,
        {static_cast<hsize_t>(state.merge_tombstone.size())},
        state.merge_tombstone.data(),
        "",
        cfg);
  }
  const auto swept_volume_contract = hydro::resolve_swept_volume_contract(
      cfg.numerics.ale.swept_volume_sign_fixed);
  const auto write_swept_volume_convention =
      [&](const std::string& name,
          const hydro::SweptVolumeConvention convention) {
        const std::uint8_t value = static_cast<std::uint8_t>(convention);
        write_numeric_dataset(file,
                              "metadata/swept_volume_contract",
                              name,
                              H5T_NATIVE_UINT8,
                              {},
                              &value,
                              "",
                              cfg);
      };
  write_swept_volume_convention("plain_csr", swept_volume_contract.plain_csr);
  write_swept_volume_convention("conservative_csr",
                                swept_volume_contract.conservative_csr);
  write_swept_volume_convention("option_b", swept_volume_contract.option_b);
  write_swept_volume_convention("axis_band", swept_volume_contract.axis_band);
  write_swept_volume_convention("plic", swept_volume_contract.plic);
  write_string_dataset(
      file,
      "metadata/swept_volume_contract",
      "semantic",
      "positive_means_low_index_to_high_index_transfer_when_oriented_v1");
  write_string_dataset(file,
                       "metadata",
                       "solver_requested",
                       solver_requested(cfg));
  write_string_dataset(file,
                       "metadata",
                       "solver_resolved",
                       solver_resolved(cfg));

  const hid_t metadata = ensure_group(file, "metadata");
  TENRYU_ASSERT(metadata >= 0, "HDF5 failed to open metadata group");
  const hid_t frozen_dset = H5Dopen2(metadata, "frozen_config", H5P_DEFAULT);
  TENRYU_ASSERT(frozen_dset >= 0, "HDF5 failed to open metadata/frozen_config");

  write_string_attribute(frozen_dset, "git_hash", "unknown");
  write_string_attribute(frozen_dset, "build_type", detect_build_type());
  write_string_attribute(frozen_dset, "cuda_arch", "unknown");
  write_string_attribute(frozen_dset, "gpu_name", detect_gpu_name());
  write_scalar_attribute_i32(frozen_dset, "n_ranks", 1);
  write_scalar_attribute_u64(frozen_dset, "rng_seed", cfg.main.seed);

  warn_h5_close_failure(H5Dclose(frozen_dset), "H5Dclose",
                        "HDF5Writer::write_metadata_group(frozen_dset)");
  warn_h5_close_failure(H5Gclose(metadata), "H5Gclose",
                        "HDF5Writer::write_metadata_group(metadata)");

  const auto bounds = group_bounds_from_config(cfg);
  write_numeric_dataset(file,
                        "metadata",
                        "group_bounds_eV",
                        H5T_NATIVE_DOUBLE,
                        {static_cast<hsize_t>(bounds.size())},
                        bounds.data(),
                        "eV",
                        cfg);

  const auto eos_signatures = eos_signatures_from_config(cfg);
  // Checkpoint schema: /metadata/eos/eos_signature is a per-material uint64
  // array for EOS table signatures (FNV-1a over model, path, and grid extents).
  write_numeric_dataset(file,
                        "metadata/eos",
                        "eos_signature",
                        H5T_NATIVE_UINT64,
                        {static_cast<hsize_t>(eos_signatures.size())},
                        eos_signatures.data(),
                        "hash64",
                        cfg);
}

std::uint64_t counter_or_zero_if_disabled(const core::State& state,
                                          const core::Config& cfg,
                                          const std::atomic<std::uint64_t>& counter) {
  if (!cfg.numerics.materials.per_material_conservation_enabled) {
    return 0;
  }
  (void)state;
  return counter.load(std::memory_order_relaxed);
}

void write_dispatch_counters_group(const hid_t file,
                                   const core::State& state,
                                   const core::Config& cfg) {
  const std::uint64_t per_material_kernel_call_count =
      counter_or_zero_if_disabled(state,
                                  cfg,
                                  state.dispatch_counters.per_material_kernel_call_count);
  const std::uint64_t eos_inverse_call_count =
      counter_or_zero_if_disabled(state, cfg, state.dispatch_counters.eos_inverse_call_count);
  const std::uint64_t mixture_projection_call_count =
      counter_or_zero_if_disabled(state,
                                  cfg,
                                  state.dispatch_counters.mixture_projection_call_count);
  const std::uint64_t lazy_cache_te_m_hit_count =
      counter_or_zero_if_disabled(state,
                                  cfg,
                                  state.dispatch_counters.lazy_cache_te_m_hit_count);
  const std::uint64_t lazy_cache_te_m_miss_count =
      counter_or_zero_if_disabled(state,
                                  cfg,
                                  state.dispatch_counters.lazy_cache_te_m_miss_count);
  const std::uint64_t hash = dispatch_counters_regression_hash(state, cfg);

  write_numeric_dataset(file,
                        "metadata/dispatch_counters",
                        "per_material_kernel_call_count",
                        H5T_NATIVE_UINT64,
                        {},
                        &per_material_kernel_call_count,
                        "count",
                        cfg);
  write_numeric_dataset(file,
                        "metadata/dispatch_counters",
                        "eos_inverse_call_count",
                        H5T_NATIVE_UINT64,
                        {},
                        &eos_inverse_call_count,
                        "count",
                        cfg);
  write_numeric_dataset(file,
                        "metadata/dispatch_counters",
                        "mixture_projection_call_count",
                        H5T_NATIVE_UINT64,
                        {},
                        &mixture_projection_call_count,
                        "count",
                        cfg);
  write_numeric_dataset(file,
                        "metadata/dispatch_counters",
                        "lazy_cache_te_m_hit_count",
                        H5T_NATIVE_UINT64,
                        {},
                        &lazy_cache_te_m_hit_count,
                        "count",
                        cfg);
  write_numeric_dataset(file,
                        "metadata/dispatch_counters",
                        "lazy_cache_te_m_miss_count",
                        H5T_NATIVE_UINT64,
                        {},
                        &lazy_cache_te_m_miss_count,
                        "count",
                        cfg);
  write_numeric_dataset(file,
                        "metadata/dispatch_counters",
                        "regression_hash_u64",
                        H5T_NATIVE_UINT64,
                        {},
                        &hash,
                        "hash64",
                        cfg);
}

void write_per_material_event_counters_group(const hid_t file,
                                             const core::State& state,
                                             const core::Config& cfg) {
  if (!cfg.numerics.materials.per_material_conservation_enabled) {
    return;
  }
  const std::uint64_t eos_table_validity_violations =
      state.dispatch_counters.eos_table_validity_violations.load(std::memory_order_relaxed);
  const std::uint64_t presence_absent_events =
      state.dispatch_counters.presence_absent_events.load(std::memory_order_relaxed);
  const std::uint64_t conservation_drift_warnings =
      state.dispatch_counters.conservation_drift_warnings.load(std::memory_order_relaxed);
  const std::uint64_t lazy_cache_te_m_invalidation_count =
      state.dispatch_counters.lazy_cache_te_m_invalidation_count.load(std::memory_order_relaxed);
  const std::uint64_t lazy_cache_te_m_hit_count =
      state.dispatch_counters.lazy_cache_te_m_hit_count.load(std::memory_order_relaxed);
  const std::uint64_t lazy_cache_te_m_miss_count =
      state.dispatch_counters.lazy_cache_te_m_miss_count.load(std::memory_order_relaxed);

  write_numeric_dataset(file,
                        "diagnostics/per_material/v1",
                        "eos_table_validity_violations_cumulative",
                        H5T_NATIVE_UINT64,
                        {},
                        &eos_table_validity_violations,
                        "count",
                        cfg);
  write_numeric_dataset(file,
                        "diagnostics/per_material/v1",
                        "presence_absent_events_cumulative",
                        H5T_NATIVE_UINT64,
                        {},
                        &presence_absent_events,
                        "count",
                        cfg);
  write_numeric_dataset(file,
                        "diagnostics/per_material/v1",
                        "conservation_drift_warnings_cumulative",
                        H5T_NATIVE_UINT64,
                        {},
                        &conservation_drift_warnings,
                        "count",
                        cfg);
  write_numeric_dataset(file,
                        "diagnostics/per_material/v1",
                        "lazy_cache_te_m_invalidation_count_cumulative",
                        H5T_NATIVE_UINT64,
                        {},
                        &lazy_cache_te_m_invalidation_count,
                        "count",
                        cfg);
  write_numeric_dataset(file,
                        "diagnostics/per_material/v1",
                        "lazy_cache_te_m_hit_count_cumulative",
                        H5T_NATIVE_UINT64,
                        {},
                        &lazy_cache_te_m_hit_count,
                        "count",
                        cfg);
  write_numeric_dataset(file,
                        "diagnostics/per_material/v1",
                        "lazy_cache_te_m_miss_count_cumulative",
                        H5T_NATIVE_UINT64,
                        {},
                        &lazy_cache_te_m_miss_count,
                        "count",
                        cfg);
}

void write_per_material_conservation_diagnostics_group(const hid_t file,
                                                       const core::State& state,
                                                       const core::Config& cfg) {
  if (!cfg.numerics.materials.per_material_conservation_enabled) {
    return;
  }
  const PerMaterialConservationResiduals residuals =
      compute_per_material_conservation_residuals(state, cfg);
  const double warn_threshold =
      cfg.numerics.materials.conservation_residual_warn_threshold_rel;
  const double hard_threshold =
      cfg.numerics.materials.conservation_residual_hard_warning_threshold_rel;
  const double max_rel = std::max({residuals.mass_max_rel_residual,
                                   residuals.Ee_max_rel_residual,
                                   residuals.Ei_max_rel_residual});
  if (max_rel > warn_threshold) {
    state.dispatch_counters.conservation_drift_warnings.fetch_add(
        1, std::memory_order_relaxed);
    const char* level = (max_rel > hard_threshold) ? "HARD WARNING" : "WARNING";
    core::log_warning(std::string("[per_material_conservation] ") + level +
                      ": max relative residual " + std::to_string(max_rel) +
                      " exceeds threshold " +
                      std::to_string((max_rel > hard_threshold) ? hard_threshold
                                                                : warn_threshold));
  }

  write_numeric_dataset(file,
                        "diagnostics/conservation/v1",
                        "per_material_mass_max_abs_residual",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &residuals.mass_max_abs_residual,
                        "g",
                        cfg);
  write_numeric_dataset(file,
                        "diagnostics/conservation/v1",
                        "per_material_mass_max_rel_residual",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &residuals.mass_max_rel_residual,
                        "dimensionless",
                        cfg);
  write_numeric_dataset(file,
                        "diagnostics/conservation/v1",
                        "per_material_Ee_max_abs_residual",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &residuals.Ee_max_abs_residual,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "diagnostics/conservation/v1",
                        "per_material_Ee_max_rel_residual",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &residuals.Ee_max_rel_residual,
                        "dimensionless",
                        cfg);
  write_numeric_dataset(file,
                        "diagnostics/conservation/v1",
                        "per_material_Ei_max_abs_residual",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &residuals.Ei_max_abs_residual,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "diagnostics/conservation/v1",
                        "per_material_Ei_max_rel_residual",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &residuals.Ei_max_rel_residual,
                        "dimensionless",
                        cfg);
}

void write_multiblock_topology_v2_group(const hid_t file,
                                        const core::State& state,
                                        const core::Config& cfg) {
  if (!mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    return;
  }
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "HDF5 write: multiblock topology requested but state topology is missing");
  const auto& mb = *state.mesh.topo.multiblock;

  const std::size_t n_cells =
      static_cast<std::size_t>(mesh::mesh_topo_n_cells_total(cfg.mesh));
  const std::size_t n_csr_entries =
      n_cells * static_cast<std::size_t>(state.mesh.corner_stride);
  // This is a constraint generalization, NOT a format change.
  // v2 writes the mesh's actual block count while preserving the datasets.
  TENRYU_ASSERT(mb.cell_block_id.size() == n_cells,
                "HDF5 write: multiblock cell_block_id size mismatch");
  TENRYU_ASSERT(mb.cell_id_stable.size() == n_cells,
                "HDF5 write: multiblock cell_id_stable size mismatch");
  TENRYU_ASSERT(mb.cell_node_csr_offsets.size() == n_cells + 1U,
                "HDF5 write: multiblock cell_node_csr_offsets size mismatch");
  TENRYU_ASSERT(mb.cell_node_csr_indices.size() == n_csr_entries,
                "HDF5 write: multiblock cell_node_csr_indices size mismatch");
  TENRYU_ASSERT(mb.face_adj_csr_offsets.size() == n_cells + 1U,
                "HDF5 write: multiblock face_adj_csr_offsets size mismatch");
  TENRYU_ASSERT(mb.face_adj_csr_indices.size() == n_csr_entries,
                "HDF5 write: multiblock face_adj_csr_indices size mismatch");
  TENRYU_ASSERT(mb.face_bc_tags.size() == n_csr_entries,
                "HDF5 write: multiblock face_bc_tags size mismatch");

  const int block_count = mb.block_count;
  write_numeric_dataset(file,
                        "mesh/topology/v2",
                        "block_count",
                        H5T_NATIVE_INT,
                        {},
                        &block_count,
                        "count",
                        cfg);
  write_numeric_dataset(file,
                        "mesh/topology/v2",
                        "cell_block_id",
                        H5T_NATIVE_INT,
                        {static_cast<hsize_t>(mb.cell_block_id.size())},
                        mb.cell_block_id.data(),
                        "index",
                        cfg);
  write_numeric_dataset(file,
                        "mesh/topology/v2",
                        "cell_node_csr_offsets",
                        H5T_NATIVE_INT,
                        {static_cast<hsize_t>(mb.cell_node_csr_offsets.size())},
                        mb.cell_node_csr_offsets.data(),
                        "offset",
                        cfg);
  write_numeric_dataset(file,
                        "mesh/topology/v2",
                        "cell_node_csr_indices",
                        H5T_NATIVE_INT,
                        {static_cast<hsize_t>(mb.cell_node_csr_indices.size())},
                        mb.cell_node_csr_indices.data(),
                        "index",
                        cfg);
  write_numeric_dataset(file,
                        "mesh/topology/v2",
                        "face_adj_csr_offsets",
                        H5T_NATIVE_INT,
                        {static_cast<hsize_t>(mb.face_adj_csr_offsets.size())},
                        mb.face_adj_csr_offsets.data(),
                        "offset",
                        cfg);
  write_numeric_dataset(file,
                        "mesh/topology/v2",
                        "face_adj_csr_indices",
                        H5T_NATIVE_INT,
                        {static_cast<hsize_t>(mb.face_adj_csr_indices.size())},
                        mb.face_adj_csr_indices.data(),
                        "index",
                        cfg);
  write_numeric_dataset(file,
                        "mesh/topology/v2",
                        "face_bc_tags",
                        H5T_NATIVE_INT,
                        {static_cast<hsize_t>(mb.face_bc_tags.size())},
                        mb.face_bc_tags.data(),
                        "tag",
                        cfg);
  write_numeric_dataset(file,
                        "mesh/topology/v2",
                        "cell_id_stable",
                        H5T_NATIVE_INT,
                        {static_cast<hsize_t>(mb.cell_id_stable.size())},
                        mb.cell_id_stable.data(),
                        "index",
                        cfg);
}

void write_multiblock_topology_v3_group(const hid_t file,
                                        const core::State& state,
                                        const core::Config& cfg) {
  if (!mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    return;
  }
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "HDF5 write: multiblock topology requested but state topology is missing");
  const auto& mb = *state.mesh.topo.multiblock;

  const std::size_t n_cells =
      static_cast<std::size_t>(mesh::mesh_topo_n_cells_total(cfg.mesh));
  const std::size_t n_csr_entries =
      n_cells * static_cast<std::size_t>(state.mesh.corner_stride);
  TENRYU_ASSERT(mb.block_count >= 3,
                "HDF5 write: multiblock topology v3 block_count must be at least 3");
  TENRYU_ASSERT(static_cast<int>(mb.blocks.size()) == mb.block_count,
                "HDF5 write: multiblock topology v3 block table size mismatch");
  TENRYU_ASSERT(mb.cell_block_id.size() == n_cells,
                "HDF5 write: multiblock cell_block_id size mismatch");
  TENRYU_ASSERT(mb.cell_id_stable.size() == n_cells,
                "HDF5 write: multiblock cell_id_stable size mismatch");
  TENRYU_ASSERT(mb.cell_orientation_sign.size() == n_cells,
                "HDF5 write: multiblock cell_orientation_sign size mismatch");
  TENRYU_ASSERT(mb.cell_node_csr_offsets.size() == n_cells + 1U,
                "HDF5 write: multiblock cell_node_csr_offsets size mismatch");
  TENRYU_ASSERT(mb.cell_node_csr_indices.size() == n_csr_entries,
                "HDF5 write: multiblock cell_node_csr_indices size mismatch");
  TENRYU_ASSERT(mb.face_adj_csr_offsets.size() == n_cells + 1U,
                "HDF5 write: multiblock face_adj_csr_offsets size mismatch");
  TENRYU_ASSERT(mb.face_adj_csr_indices.size() == n_csr_entries,
                "HDF5 write: multiblock face_adj_csr_indices size mismatch");
  TENRYU_ASSERT(mb.face_bc_tags.size() == n_csr_entries,
                "HDF5 write: multiblock face_bc_tags size mismatch");

  const std::string group_path = "mesh/topology/v3";
  const int block_count = mb.block_count;
  write_numeric_dataset(file,
                        group_path,
                        "block_count",
                        H5T_NATIVE_INT,
                        {},
                        &block_count,
                        "count",
                        cfg);

  std::vector<int> block_id(mb.blocks.size(), 0);
  std::vector<int> block_role(mb.blocks.size(), 0);
  std::vector<int> block_n_i_cells(mb.blocks.size(), 0);
  std::vector<int> block_n_j_cells(mb.blocks.size(), 0);
  std::vector<int> block_cell_begin(mb.blocks.size(), 0);
  std::vector<int> block_cell_count(mb.blocks.size(), 0);
  std::vector<int> block_owned_node_begin(mb.blocks.size(), 0);
  std::vector<int> block_owned_node_count(mb.blocks.size(), 0);
  for (std::size_t b = 0; b < mb.blocks.size(); ++b) {
    const auto& block = mb.blocks[b];
    block_id[b] = static_cast<int>(b);
    block_role[b] = static_cast<int>(block.role);
    block_n_i_cells[b] = block.n_i_cells;
    block_n_j_cells[b] = block.n_j_cells;
    block_cell_begin[b] = block.cell_begin;
    block_cell_count[b] = block.cell_count;
    block_owned_node_begin[b] = block.owned_node_begin;
    block_owned_node_count[b] = block.owned_node_count;
  }
  const std::vector<hsize_t> block_dims = {
      static_cast<hsize_t>(mb.blocks.size())};
  write_numeric_dataset(file, group_path, "block_id", H5T_NATIVE_INT,
                        block_dims, block_id.data(), "index", cfg);
  write_numeric_dataset(file, group_path, "block_role", H5T_NATIVE_INT,
                        block_dims, block_role.data(), "enum", cfg);
  write_numeric_dataset(file, group_path, "block_n_i_cells", H5T_NATIVE_INT,
                        block_dims, block_n_i_cells.data(), "count", cfg);
  write_numeric_dataset(file, group_path, "block_n_j_cells", H5T_NATIVE_INT,
                        block_dims, block_n_j_cells.data(), "count", cfg);
  write_numeric_dataset(file, group_path, "block_cell_begin", H5T_NATIVE_INT,
                        block_dims, block_cell_begin.data(), "index", cfg);
  write_numeric_dataset(file, group_path, "block_cell_count", H5T_NATIVE_INT,
                        block_dims, block_cell_count.data(), "count", cfg);
  write_numeric_dataset(file, group_path, "block_owned_node_begin", H5T_NATIVE_INT,
                        block_dims, block_owned_node_begin.data(), "index", cfg);
  write_numeric_dataset(file, group_path, "block_owned_node_count", H5T_NATIVE_INT,
                        block_dims, block_owned_node_count.data(), "count", cfg);

  std::vector<int> seam_block_a(mb.seams.size(), 0);
  std::vector<int> seam_side_a(mb.seams.size(), 0);
  std::vector<int> seam_block_b(mb.seams.size(), 0);
  std::vector<int> seam_side_b(mb.seams.size(), 0);
  std::vector<int> seam_orientation(mb.seams.size(), 0);
  std::vector<int> seam_index_begin(mb.seams.size(), 0);
  std::vector<int> seam_index_count(mb.seams.size(), 0);
  for (std::size_t s = 0; s < mb.seams.size(); ++s) {
    const auto& seam = mb.seams[s];
    seam_block_a[s] = seam.block_a;
    seam_side_a[s] = static_cast<int>(seam.side_a);
    seam_block_b[s] = seam.block_b;
    seam_side_b[s] = static_cast<int>(seam.side_b);
    seam_orientation[s] = seam.orientation;
    seam_index_begin[s] = seam.index_begin;
    seam_index_count[s] = seam.index_count;
  }
  const std::vector<hsize_t> seam_dims = {
      static_cast<hsize_t>(mb.seams.size())};
  write_numeric_dataset(file, group_path, "seam_block_a", H5T_NATIVE_INT,
                        seam_dims, seam_block_a.data(), "index", cfg);
  write_numeric_dataset(file, group_path, "seam_side_a", H5T_NATIVE_INT,
                        seam_dims, seam_side_a.data(), "enum", cfg);
  write_numeric_dataset(file, group_path, "seam_block_b", H5T_NATIVE_INT,
                        seam_dims, seam_block_b.data(), "index", cfg);
  write_numeric_dataset(file, group_path, "seam_side_b", H5T_NATIVE_INT,
                        seam_dims, seam_side_b.data(), "enum", cfg);
  write_numeric_dataset(file, group_path, "seam_orientation", H5T_NATIVE_INT,
                        seam_dims, seam_orientation.data(), "sign", cfg);
  write_numeric_dataset(file, group_path, "seam_index_begin", H5T_NATIVE_INT,
                        seam_dims, seam_index_begin.data(), "index", cfg);
  write_numeric_dataset(file, group_path, "seam_index_count", H5T_NATIVE_INT,
                        seam_dims, seam_index_count.data(), "count", cfg);

  write_numeric_dataset(file,
                        group_path,
                        "cell_block_id",
                        H5T_NATIVE_INT,
                        {static_cast<hsize_t>(mb.cell_block_id.size())},
                        mb.cell_block_id.data(),
                        "index",
                        cfg);
  write_numeric_dataset(file,
                        group_path,
                        "cell_id_stable",
                        H5T_NATIVE_INT,
                        {static_cast<hsize_t>(mb.cell_id_stable.size())},
                        mb.cell_id_stable.data(),
                        "index",
                        cfg);
  write_numeric_dataset(file,
                        group_path,
                        "cell_orientation_sign",
                        H5T_NATIVE_INT,
                        {static_cast<hsize_t>(mb.cell_orientation_sign.size())},
                        mb.cell_orientation_sign.data(),
                        "sign",
                        cfg);
  write_numeric_dataset(file,
                        group_path,
                        "cell_node_csr_offsets",
                        H5T_NATIVE_INT,
                        {static_cast<hsize_t>(mb.cell_node_csr_offsets.size())},
                        mb.cell_node_csr_offsets.data(),
                        "offset",
                        cfg);
  write_numeric_dataset(file,
                        group_path,
                        "cell_node_csr_indices",
                        H5T_NATIVE_INT,
                        {static_cast<hsize_t>(mb.cell_node_csr_indices.size())},
                        mb.cell_node_csr_indices.data(),
                        "index",
                        cfg);
  write_numeric_dataset(file,
                        group_path,
                        "face_adj_csr_offsets",
                        H5T_NATIVE_INT,
                        {static_cast<hsize_t>(mb.face_adj_csr_offsets.size())},
                        mb.face_adj_csr_offsets.data(),
                        "offset",
                        cfg);
  write_numeric_dataset(file,
                        group_path,
                        "face_adj_csr_indices",
                        H5T_NATIVE_INT,
                        {static_cast<hsize_t>(mb.face_adj_csr_indices.size())},
                        mb.face_adj_csr_indices.data(),
                        "index",
                        cfg);
  write_numeric_dataset(file,
                        group_path,
                        "face_bc_tags",
                        H5T_NATIVE_INT,
                        {static_cast<hsize_t>(mb.face_bc_tags.size())},
                        mb.face_bc_tags.data(),
                        "tag",
                        cfg);
}

void write_multiblock_topology_group(const hid_t file,
                                     const core::State& state,
                                     const core::Config& cfg) {
  if (!mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    return;
  }
  if (cfg.mesh.topology_scheme ==
          core::TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_5BLOCK ||
      mesh::mesh_topo_has_trifan_cap(cfg.mesh) ||
      mesh::mesh_topo_has_polar_tier(cfg.mesh) ||
      cfg.mesh.topology_scheme ==
          core::TopologyScheme::MULTIBLOCK_POLAR_TIER_CART_CENTER ||
      cfg.mesh.topology_scheme ==
          core::TopologyScheme::PENTAGON_BELT_SHELL) {
    write_multiblock_topology_v3_group(file, state, cfg);
    return;
  }
  write_multiblock_topology_v2_group(file, state, cfg);
}

bool has_button_center(const core::State& state) {
  return state.mesh.button_center && state.mesh.button_center->enabled;
}

void write_button_hydro_flags_group(const hid_t file,
                                    const core::State& state,
                                    const core::Config& cfg) {
  if (!has_button_center(state)) {
    return;
  }
  std::vector<std::int8_t> hydro_active = state.hydro_active;
  if (hydro_active.size() != state.rho.size()) {
    hydro_active.assign(state.rho.size(), static_cast<std::int8_t>(1));
  }
  write_numeric_dataset(file,
                        "hydro_flags",
                        "hydro_active",
                        H5T_NATIVE_INT8,
                        {static_cast<hsize_t>(hydro_active.size())},
                        hydro_active.data(),
                        "flag",
                        cfg);

  std::vector<std::uint8_t> cell_is_void = state.cell_is_void;
  if (cell_is_void.size() != state.rho.size()) {
    cell_is_void.assign(state.rho.size(), static_cast<std::uint8_t>(0));
  }
  write_numeric_dataset(file,
                        "hydro_flags",
                        "cell_is_void",
                        H5T_NATIVE_UINT8,
                        {static_cast<hsize_t>(cell_is_void.size())},
                        cell_is_void.data(),
                        "flag",
                        cfg);
}

void write_mesh_group(const hid_t file,
                      const core::State& state,
                      const core::Config& cfg) {
  const auto x_r = copy_field_to_host(state.x_r);
  const auto x_z = copy_field_to_host(state.x_z);
  const auto v_r = copy_field_to_host(state.v_r);
  const auto v_z = copy_field_to_host(state.v_z);
  const auto mat_id = compute_cell_material_id(state, cfg);
  const bool is_2d = cfg.main.dimension == "2D_RZ";
  const bool is_multiblock = mesh::mesh_topo_is_multiblock(cfg.mesh);

  if (is_2d && !is_multiblock) {
    const std::vector<hsize_t> node_dims = {
        static_cast<hsize_t>(cfg.mesh.nr + 1), static_cast<hsize_t>(cfg.mesh.nz + 1)};
    write_numeric_dataset(file,
                          "mesh",
                          "x_r",
                          H5T_NATIVE_DOUBLE,
                          node_dims,
                          x_r.data(),
                          "cm",
                          cfg);
    write_numeric_dataset(file,
                          "mesh",
                          "x_z",
                          H5T_NATIVE_DOUBLE,
                          node_dims,
                          x_z.data(),
                          "cm",
                          cfg);
    write_numeric_dataset(file,
                          "mesh",
                          "v_z",
                          H5T_NATIVE_DOUBLE,
                          {static_cast<hsize_t>(v_z.size())},
                          v_z.data(),
                          "cm/s",
                          cfg);
  } else {
    write_numeric_dataset(file,
                          "mesh",
                          "x_r",
                          H5T_NATIVE_DOUBLE,
                          {static_cast<hsize_t>(x_r.size())},
                          x_r.data(),
                          "cm",
                          cfg);
    if (is_2d) {
      write_numeric_dataset(file,
                            "mesh",
                            "x_z",
                            H5T_NATIVE_DOUBLE,
                            {static_cast<hsize_t>(x_z.size())},
                            x_z.data(),
                            "cm",
                            cfg);
      write_numeric_dataset(file,
                            "mesh",
                            "v_z",
                            H5T_NATIVE_DOUBLE,
                            {static_cast<hsize_t>(v_z.size())},
                            v_z.data(),
                            "cm/s",
                            cfg);
    }
  }

  write_numeric_dataset(file,
                        "mesh",
                        "v_r",
                        H5T_NATIVE_DOUBLE,
                        {static_cast<hsize_t>(v_r.size())},
                        v_r.data(),
                        "cm/s",
                        cfg);
  write_numeric_dataset(file,
                        "mesh",
                        "cell_material_id",
                        H5T_NATIVE_INT32,
                        {static_cast<hsize_t>(mat_id.size())},
                        mat_id.data(),
                        "index",
                        cfg);
  if (has_button_center(state) &&
      state.mesh.cell_nverts.size() == state.rho.size()) {
    std::vector<std::uint8_t> cell_nverts = state.mesh.cell_nverts;
    if (!cell_nverts.empty()) {
      cell_nverts[0] = static_cast<std::uint8_t>(cfg.mesh.nz + 1);
      for (int c = 0; c < state.mesh.topo.n_cells; ++c) {
        if (state.mesh.is_dormant_cell(c)) {
          cell_nverts[static_cast<std::size_t>(c)] = 0U;
        }
      }
    }
    write_numeric_dataset(file,
                          "mesh/topology/v1",
                          "cell_nverts",
                          H5T_NATIVE_UINT8,
                          {static_cast<hsize_t>(cell_nverts.size())},
                          cell_nverts.data(),
                          "count",
                          cfg);
  }
  write_multiblock_topology_group(file, state, cfg);
}

void write_ale_reference_checkpoint_group(const hid_t file,
                                          const core::State& state,
                                          const core::Config& cfg) {
  const std::size_t n_nodes = state.x_r.size();
  TENRYU_ASSERT(state.x_r_reference.size() == n_nodes &&
                    state.x_z_reference.size() == n_nodes &&
                    state.x_r_shock_target.size() == n_nodes &&
                    state.x_z_shock_target.size() == n_nodes,
                "ALE checkpoint reference fields must be node-sized");

  const auto x_r_reference = copy_field_to_host(state.x_r_reference);
  const auto x_z_reference = copy_field_to_host(state.x_z_reference);
  const auto x_r_shock_target = copy_field_to_host(state.x_r_shock_target);
  const auto x_z_shock_target = copy_field_to_host(state.x_z_shock_target);
  constexpr const char* group_path = "mesh/ale_reference/v1";

  const hid_t group = ensure_group(file, group_path);
  TENRYU_ASSERT(group >= 0,
                "HDF5 failed to open mesh/ale_reference/v1 group");
  write_scalar_attribute_u64(
      group, "reference_epoch", state.reference_epoch);
  warn_h5_close_failure(
      H5Gclose(group),
      "H5Gclose",
      "HDF5Writer::write_ale_reference_checkpoint_group(group)");

  const std::vector<hsize_t> node_dims = {
      static_cast<hsize_t>(n_nodes)};
  write_numeric_dataset(file,
                        group_path,
                        "x_r_reference",
                        H5T_NATIVE_DOUBLE,
                        node_dims,
                        x_r_reference.data(),
                        "cm",
                        cfg);
  write_numeric_dataset(file,
                        group_path,
                        "x_z_reference",
                        H5T_NATIVE_DOUBLE,
                        node_dims,
                        x_z_reference.data(),
                        "cm",
                        cfg);
  write_numeric_dataset(file,
                        group_path,
                        "x_r_shock_target",
                        H5T_NATIVE_DOUBLE,
                        node_dims,
                        x_r_shock_target.data(),
                        "cm",
                        cfg);
  write_numeric_dataset(file,
                        group_path,
                        "x_z_shock_target",
                        H5T_NATIVE_DOUBLE,
                        node_dims,
                        x_z_shock_target.data(),
                        "cm",
                        cfg);
}

void write_carrier_checkpoint_group(const hid_t file,
                                    const core::State& state,
                                    const core::Config& cfg) {
  if (cfg.numerics.ale.mesh_mode != "reale_v2") {
    return;
  }

  constexpr const char* group_path = "mesh/carrier/v1";
  const std::size_t n_nodes = state.x_r.size();
  const std::size_t n_masters = state.boundary_carrier.masters.size();
  TENRYU_ASSERT(
      n_masters <=
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()),
      "HDF5 write: carrier master count exceeds int64 range");
  TENRYU_ASSERT(!state.boundary_carrier.valid || n_masters >= 3U,
                "HDF5 write: valid carrier must have at least three masters");

  std::vector<std::uint64_t> master_stable_id(n_masters, 0);
  std::vector<std::int32_t> master_mesh_node(n_masters, -1);
  std::vector<std::uint8_t> master_bc_class(n_masters, 0);
  for (std::size_t master_index = 0; master_index < n_masters;
       ++master_index) {
    const auto& master = state.boundary_carrier.masters[master_index];
    TENRYU_ASSERT(master.mesh_node >= 0 &&
                      static_cast<std::size_t>(master.mesh_node) < n_nodes,
                  "HDF5 write: carrier master mesh node out of range");
    const std::uint8_t bc_class =
        static_cast<std::uint8_t>(master.bc_class);
    TENRYU_ASSERT(
        bc_class <= static_cast<std::uint8_t>(mesh::CarrierBcClass::kAxis),
        "HDF5 write: carrier master boundary class out of range");
    master_stable_id[master_index] = master.stable_id;
    master_mesh_node[master_index] =
        static_cast<std::int32_t>(master.mesh_node);
    master_bc_class[master_index] = bc_class;
  }

  std::vector<std::uint8_t> node_class = state.carrier_node_class;
  std::vector<int> node_edge = state.carrier_node_edge;
  std::vector<double> node_lambda = state.carrier_node_lambda;
  const bool empty_generation_zero = !state.boundary_carrier.valid &&
                                     node_class.empty() && node_edge.empty() &&
                                     node_lambda.empty();
  if (empty_generation_zero) {
    node_class.assign(n_nodes, 0U);
    node_edge.assign(n_nodes, -1);
    node_lambda.assign(n_nodes, 0.0);
  }
  TENRYU_ASSERT(node_class.size() == n_nodes && node_edge.size() == n_nodes &&
                    node_lambda.size() == n_nodes,
                "HDF5 write: carrier node arrays must match node count");

  std::vector<std::int32_t> node_edge_i32(n_nodes, -1);
  for (std::size_t node = 0; node < n_nodes; ++node) {
    TENRYU_ASSERT(node_class[node] <= 2U,
                  "HDF5 write: carrier node class out of range");
    TENRYU_ASSERT(node_lambda[node] >= 0.0 && node_lambda[node] <= 1.0,
                  "HDF5 write: carrier node lambda out of range");
    TENRYU_ASSERT(
        node_edge[node] >= std::numeric_limits<std::int32_t>::min() &&
            node_edge[node] <= std::numeric_limits<std::int32_t>::max(),
        "HDF5 write: carrier node edge exceeds int32 range");
    node_edge_i32[node] = static_cast<std::int32_t>(node_edge[node]);
  }

  const hid_t group = ensure_group(file, group_path);
  TENRYU_ASSERT(group >= 0,
                "HDF5 failed to open mesh/carrier/v1 group");
  write_scalar_attribute_i8(
      group, "valid",
      static_cast<std::int8_t>(state.boundary_carrier.valid ? 1 : 0));
  write_scalar_attribute_i8(
      group, "domain_active",
      static_cast<std::int8_t>(state.carrier_domain_active ? 1 : 0));
  write_scalar_attribute_i64(
      group, "n_masters", static_cast<std::int64_t>(n_masters));
  warn_h5_close_failure(
      H5Gclose(group), "H5Gclose",
      "HDF5Writer::write_carrier_checkpoint_group(group)");

  const std::vector<hsize_t> master_dims = {
      static_cast<hsize_t>(n_masters)};
  const std::vector<hsize_t> node_dims = {static_cast<hsize_t>(n_nodes)};
  write_numeric_dataset(file, group_path, "master_stable_id",
                        H5T_NATIVE_UINT64, master_dims,
                        master_stable_id.data(), "index", cfg);
  write_numeric_dataset(file, group_path, "master_mesh_node",
                        H5T_NATIVE_INT32, master_dims,
                        master_mesh_node.data(), "index", cfg);
  write_numeric_dataset(file, group_path, "master_bc_class",
                        H5T_NATIVE_UINT8, master_dims,
                        master_bc_class.data(), "enum", cfg);
  write_numeric_dataset(file, group_path, "node_class", H5T_NATIVE_UINT8,
                        node_dims, node_class.data(), "enum", cfg);
  write_numeric_dataset(file, group_path, "node_edge", H5T_NATIVE_INT32,
                        node_dims, node_edge_i32.data(), "index", cfg);
  write_numeric_dataset(file, group_path, "node_lambda", H5T_NATIVE_DOUBLE,
                        node_dims, node_lambda.data(), "dimensionless", cfg);
}

void write_evacuated_cells_checkpoint_group(const hid_t file,
                                            const core::State& state,
                                            const core::Config& cfg) {
  const auto& evacuated = state.evacuated_cells;
  if (evacuated.mass_ref.empty()) {
    return;
  }

  constexpr const char* group_path = "evacuated_cells";
  const std::size_t n_cells = evacuated.mass_ref.size();
  TENRYU_ASSERT(
      n_cells == state.rho.size() &&
          evacuated.inactive_member_mask.size() == n_cells &&
          evacuated.closure_lineage.size() == n_cells &&
          evacuated.contact_active_mask.size() == n_cells &&
          evacuated.vol_ref.size() == n_cells &&
          evacuated.x_r_ref.size() == state.x_r.size() &&
          evacuated.x_z_ref.size() == state.x_z.size() &&
          evacuated.off_streak.size() == n_cells &&
          evacuated.controller_dwell_remaining.size() == n_cells &&
          evacuated.controller_evaluations_since_conversion.size() ==
              n_cells &&
          evacuated.controller_previous_volume.size() == n_cells &&
          evacuated.controller_previous_contact_gap.size() == n_cells,
      "HDF5 write: evacuated-cell arrays must match cell count");

  std::vector<double> controller_previous_contact_gap(2U * n_cells, 0.0);
  for (std::size_t cell = 0; cell < n_cells; ++cell) {
    controller_previous_contact_gap[2U * cell] =
        evacuated.controller_previous_contact_gap[cell][0];
    controller_previous_contact_gap[2U * cell + 1U] =
        evacuated.controller_previous_contact_gap[cell][1];
  }

  const std::size_t n_slots_size = state.contact_graph.records.size();
  TENRYU_ASSERT(
      n_slots_size <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
      "HDF5 write: evacuated-cell contact slot count exceeds int range");
  const int n_slots = static_cast<int>(n_slots_size);
  std::vector<int> cell(n_slots_size, -1);
  std::vector<int> axis(n_slots_size, 1);
  std::vector<int> node_a0(n_slots_size, -1);
  std::vector<int> node_a1(n_slots_size, -1);
  std::vector<int> node_b0(n_slots_size, -1);
  std::vector<int> node_b1(n_slots_size, -1);
  std::vector<double> normal_pair_r0(n_slots_size, 0.0);
  std::vector<double> normal_pair_z0(n_slots_size, 0.0);
  std::vector<double> normal_pair_r1(n_slots_size, 0.0);
  std::vector<double> normal_pair_z1(n_slots_size, 0.0);
  std::vector<double> gap_at_engagement_pair0(n_slots_size, 0.0);
  std::vector<double> gap_at_engagement_pair1(n_slots_size, 0.0);
  std::vector<double> h_perp_ref(n_slots_size, 0.0);
  std::vector<double> g0(n_slots_size, 0.0);
  std::vector<double> g_arm(n_slots_size, 0.0);
  std::vector<double> gap_pair0(n_slots_size, 0.0);
  std::vector<double> gap_pair1(n_slots_size, 0.0);
  std::vector<double> gap_prev_pair0(n_slots_size, 0.0);
  std::vector<double> gap_prev_pair1(n_slots_size, 0.0);
  std::vector<std::uint8_t> pair_engaged0(n_slots_size, 0U);
  std::vector<std::uint8_t> pair_engaged1(n_slots_size, 0U);
  std::vector<int> tensile_streak_pair0(n_slots_size, 0);
  std::vector<int> tensile_streak_pair1(n_slots_size, 0);
  std::vector<double> gap(n_slots_size, 0.0);
  std::vector<double> gap_prev(n_slots_size, 0.0);
  std::vector<std::uint8_t> slot_state(n_slots_size, 0U);
  std::vector<double> mass_at_engagement(n_slots_size, 0.0);
  std::vector<double> vol_at_engagement(n_slots_size, 0.0);
  std::vector<int> devolumized(n_slots_size, 0);
  std::vector<double> lambda_last(n_slots_size, 0.0);
  std::vector<double> impact_heat_total(n_slots_size, 0.0);
  std::vector<int> engage_count(n_slots_size, 0);
  std::vector<int> release_count(n_slots_size, 0);
  std::vector<int> reproject_count(n_slots_size, 0);
  std::vector<double> reproject_heat_total(n_slots_size, 0.0);
  std::vector<int> reproject_count_last_logged(n_slots_size, 0);
  std::vector<double> reproject_heat_last_logged(n_slots_size, 0.0);
  for (std::size_t index = 0; index < n_slots_size; ++index) {
    const auto& slot = state.contact_graph.records[index];
    cell[index] = slot.cell;
    axis[index] = slot.axis;
    node_a0[index] = slot.node_a[0];
    node_a1[index] = slot.node_a[1];
    node_b0[index] = slot.node_b[0];
    node_b1[index] = slot.node_b[1];
    normal_pair_r0[index] = slot.normal_pair_r[0];
    normal_pair_z0[index] = slot.normal_pair_z[0];
    normal_pair_r1[index] = slot.normal_pair_r[1];
    normal_pair_z1[index] = slot.normal_pair_z[1];
    gap_at_engagement_pair0[index] = slot.gap_at_engagement_pair[0];
    gap_at_engagement_pair1[index] = slot.gap_at_engagement_pair[1];
    h_perp_ref[index] = slot.h_perp_ref;
    g0[index] = slot.g0;
    g_arm[index] = slot.g_arm;
    gap_pair0[index] = slot.gap_pair[0];
    gap_pair1[index] = slot.gap_pair[1];
    gap_prev_pair0[index] = slot.gap_prev_pair[0];
    gap_prev_pair1[index] = slot.gap_prev_pair[1];
    pair_engaged0[index] = slot.pair_engaged[0];
    pair_engaged1[index] = slot.pair_engaged[1];
    tensile_streak_pair0[index] = slot.tensile_streak_pair[0];
    tensile_streak_pair1[index] = slot.tensile_streak_pair[1];
    gap[index] = slot.gap;
    gap_prev[index] = slot.gap_prev;
    slot_state[index] = static_cast<std::uint8_t>(slot.state);
    mass_at_engagement[index] = slot.mass_at_engagement;
    vol_at_engagement[index] = slot.vol_at_engagement;
    devolumized[index] = static_cast<int>(slot.devolumized);
    lambda_last[index] = slot.lambda_last;
    impact_heat_total[index] = slot.impact_heat_total;
    engage_count[index] = slot.engage_count;
    release_count[index] = slot.release_count;
    reproject_count[index] = slot.reproject_count;
    reproject_heat_total[index] = slot.reproject_heat_total;
    reproject_count_last_logged[index] = slot.reproject_count_last_logged;
    reproject_heat_last_logged[index] = slot.reproject_heat_last_logged;
  }

  const std::vector<hsize_t> cell_dims = {static_cast<hsize_t>(n_cells)};
  const std::vector<hsize_t> contact_gap_dims = {
      static_cast<hsize_t>(n_cells), 2U};
  const std::vector<hsize_t> slot_dims = {
      static_cast<hsize_t>(n_slots_size)};
  const auto write_u8 = [&](const std::string& name,
                            const std::vector<std::uint8_t>& values,
                            const std::string& units) {
    write_numeric_dataset(file, group_path, name, H5T_NATIVE_UINT8, slot_dims,
                          values.data(), units, cfg);
  };
  const auto write_int = [&](const std::string& name,
                             const std::vector<int>& values,
                             const std::string& units) {
    write_numeric_dataset(file, group_path, name, H5T_NATIVE_INT, slot_dims,
                          values.data(), units, cfg);
  };
  const auto write_double = [&](const std::string& name,
                                const std::vector<double>& values,
                                const std::string& units) {
    write_numeric_dataset(file, group_path, name, H5T_NATIVE_DOUBLE,
                          slot_dims, values.data(), units, cfg);
  };

  write_numeric_dataset(file, group_path, "inactive_member_mask",
                        H5T_NATIVE_UINT8, cell_dims,
                        evacuated.inactive_member_mask.data(), "flag", cfg);
  write_numeric_dataset(file, group_path, "closure_lineage",
                        H5T_NATIVE_UINT8, cell_dims,
                        evacuated.closure_lineage.data(), "flag", cfg);
  write_numeric_dataset(file, group_path, "contact_active_mask",
                        H5T_NATIVE_UINT8, cell_dims,
                        evacuated.contact_active_mask.data(), "flag", cfg);
  write_numeric_dataset(file, group_path, "mass_ref", H5T_NATIVE_DOUBLE,
                        cell_dims, evacuated.mass_ref.data(), "g", cfg);
  write_numeric_dataset(file, group_path, "vol_ref", H5T_NATIVE_DOUBLE,
                        cell_dims, evacuated.vol_ref.data(), "cm3", cfg);
  write_numeric_dataset(file, group_path, "x_r_ref", H5T_NATIVE_DOUBLE,
                        {static_cast<hsize_t>(evacuated.x_r_ref.size())},
                        evacuated.x_r_ref.data(), "cm", cfg);
  write_numeric_dataset(file, group_path, "x_z_ref", H5T_NATIVE_DOUBLE,
                        {static_cast<hsize_t>(evacuated.x_z_ref.size())},
                        evacuated.x_z_ref.data(), "cm", cfg);
  write_numeric_dataset(file, group_path, "controller_previous_volume",
                        H5T_NATIVE_DOUBLE, cell_dims,
                        evacuated.controller_previous_volume.data(), "cm3", cfg);
  write_numeric_dataset(file, group_path, "controller_previous_contact_gap",
                        H5T_NATIVE_DOUBLE, contact_gap_dims,
                        controller_previous_contact_gap.data(), "cm", cfg);
  write_numeric_dataset(file, group_path, "off_streak", H5T_NATIVE_INT,
                        cell_dims, evacuated.off_streak.data(), "count", cfg);
  write_numeric_dataset(file, group_path, "controller_dwell_remaining",
                        H5T_NATIVE_INT, cell_dims,
                        evacuated.controller_dwell_remaining.data(), "count", cfg);
  write_numeric_dataset(
      file, group_path, "controller_evaluations_since_conversion",
      H5T_NATIVE_INT, cell_dims,
      evacuated.controller_evaluations_since_conversion.data(), "count", cfg);
  if (!evacuated.cell_axis_edge_collapsed.empty()) {
    write_numeric_dataset(file, group_path, "cell_axis_edge_collapsed",
                          H5T_NATIVE_UINT8, cell_dims,
                          evacuated.cell_axis_edge_collapsed.data(), "flag",
                          cfg);
  }
  if (!evacuated.node_axis_alias.empty()) {
    write_numeric_dataset(file, group_path, "node_axis_alias",
                          H5T_NATIVE_INT,
                          {static_cast<hsize_t>(state.x_r.size())},
                          evacuated.node_axis_alias.data(), "index", cfg);
  }
  write_numeric_dataset(file, group_path, "conversions_total", H5T_NATIVE_INT,
                        {}, &evacuated.conversions_total, "count", cfg);
  write_numeric_dataset(file, group_path, "n_slots", H5T_NATIVE_INT, {},
                        &n_slots, "count", cfg);

  write_int("cell", cell, "index");
  write_int("axis", axis, "enum");
  write_int("node_a0", node_a0, "index");
  write_int("node_a1", node_a1, "index");
  write_int("node_b0", node_b0, "index");
  write_int("node_b1", node_b1, "index");
  write_double("normal_pair_r0", normal_pair_r0, "dimensionless");
  write_double("normal_pair_z0", normal_pair_z0, "dimensionless");
  write_double("normal_pair_r1", normal_pair_r1, "dimensionless");
  write_double("normal_pair_z1", normal_pair_z1, "dimensionless");
  write_double("gap_at_engagement_pair0", gap_at_engagement_pair0, "cm");
  write_double("gap_at_engagement_pair1", gap_at_engagement_pair1, "cm");
  write_double("h_perp_ref", h_perp_ref, "cm");
  write_double("g0", g0, "cm");
  write_double("g_arm", g_arm, "cm");
  write_double("gap_pair0", gap_pair0, "cm");
  write_double("gap_pair1", gap_pair1, "cm");
  write_double("gap_prev_pair0", gap_prev_pair0, "cm");
  write_double("gap_prev_pair1", gap_prev_pair1, "cm");
  write_u8("pair_engaged0", pair_engaged0, "flag");
  write_u8("pair_engaged1", pair_engaged1, "flag");
  write_int("tensile_streak_pair0", tensile_streak_pair0, "count");
  write_int("tensile_streak_pair1", tensile_streak_pair1, "count");
  write_double("gap", gap, "cm");
  write_double("gap_prev", gap_prev, "cm");
  write_u8("state", slot_state, "enum");
  write_double("mass_at_engagement", mass_at_engagement, "g");
  write_double("vol_at_engagement", vol_at_engagement, "cm3");
  write_int("devolumized", devolumized, "flag");
  write_double("lambda_last", lambda_last, "dyn");
  write_double("impact_heat_total", impact_heat_total, "erg");
  write_int("engage_count", engage_count, "count");
  write_int("release_count", release_count, "count");
  write_int("reproject_count", reproject_count, "count");
  write_double("reproject_heat_total", reproject_heat_total, "erg");
  write_int("reproject_count_last_logged", reproject_count_last_logged,
            "count");
  write_double("reproject_heat_last_logged", reproject_heat_last_logged,
               "erg");
}

void write_hydro_group(const hid_t file,
                       const core::State& state,
                       const core::Config& cfg) {
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_mat = cfg.materials.materials.size();

  const auto rho = copy_field_to_host(state.rho);
  const auto Te = copy_field_to_host(state.Te);
  const auto Ti = copy_field_to_host(state.Ti);
  const auto ee = copy_field_to_host(state.ee);
  const auto ei = copy_field_to_host(state.ei);
  const auto Pe = copy_field_to_host(state.Pe);
  const auto Pi = copy_field_to_host(state.Pi);
  const auto Qvisc = copy_field_to_host(state.Qvisc);
  const auto shock_time = copy_field_to_host(state.shock_time);
  const auto adaptive_av_gate = copy_field_to_host(state.adaptive_av_gate);
  const auto mass = copy_field_to_host(state.mass);
  const auto vol = copy_field_to_host(state.vol);
  const auto zbar = copy_field_to_host(state.zbar);
  const auto eta_compatible = copy_field_to_host(state.eta_compatible);
  const auto volFrac = copy_field_to_host(state.volFrac);
  const auto node_mass = compute_node_mass_for_output(state, cfg);

  const std::vector<hsize_t> cdim = {static_cast<hsize_t>(n_cells)};

  write_numeric_dataset(file, "hydro", "rho", H5T_NATIVE_DOUBLE, cdim, rho.data(), "g/cm3", cfg);
  write_numeric_dataset(file, "hydro", "Te", H5T_NATIVE_DOUBLE, cdim, Te.data(), "eV", cfg);
  write_numeric_dataset(file, "hydro", "Ti", H5T_NATIVE_DOUBLE, cdim, Ti.data(), "eV", cfg);
  write_numeric_dataset(file, "hydro", "ee", H5T_NATIVE_DOUBLE, cdim, ee.data(), "erg/g", cfg);
  write_numeric_dataset(file, "hydro", "ei", H5T_NATIVE_DOUBLE, cdim, ei.data(), "erg/g", cfg);
  write_numeric_dataset(file, "hydro", "Pe", H5T_NATIVE_DOUBLE, cdim, Pe.data(), "dyne/cm2", cfg);
  write_numeric_dataset(file, "hydro", "Pi", H5T_NATIVE_DOUBLE, cdim, Pi.data(), "dyne/cm2", cfg);
  if (cfg.numerics.diagnostics.conduction_energy_rate_export.enabled) {
    const auto conduction_e_rate =
        copy_field_to_host(state.conduction_e_rate);
    write_numeric_dataset(file,
                          "hydro",
                          "conduction_e_rate",
                          H5T_NATIVE_DOUBLE,
                          cdim,
                          conduction_e_rate.data(),
                          "erg/cm3/s",
                          cfg);
  }
  if (cfg.numerics.diagnostics.refinement_estimator.enabled &&
      state.refine_error.size() == n_cells) {
    write_numeric_dataset(file,
                          "hydro",
                          "refine_error",
                          H5T_NATIVE_DOUBLE,
                          cdim,
                          state.refine_error.data(),
                          "dimensionless",
                          cfg);
  }
  write_numeric_dataset(file,
                        "hydro",
                        "Qvisc",
                        H5T_NATIVE_DOUBLE,
                        cdim,
                        Qvisc.data(),
                        "dyne/cm2",
                        cfg);
  if (state.hot_e_Q_host.size() == static_cast<std::size_t>(n_cells)) {
    write_numeric_dataset(file, "hydro", "hot_e_Q", H5T_NATIVE_DOUBLE, cdim,
                          state.hot_e_Q_host.data(), "erg/cm3/s", cfg);
  }
  if (state.cbet_gross_exchange.size() ==
      static_cast<std::size_t>(n_cells)) {
    write_numeric_dataset(
        file, "hydro", "cbet_gross_exchange", H5T_NATIVE_DOUBLE, cdim,
        state.cbet_gross_exchange.data(), "erg/cm3/s", cfg);
  }
  if (state.cbet_net_to_inbound.size() ==
      static_cast<std::size_t>(n_cells)) {
    write_numeric_dataset(
        file, "hydro", "cbet_net_to_inbound", H5T_NATIVE_DOUBLE, cdim,
        state.cbet_net_to_inbound.data(), "erg/cm3/s", cfg);
  }
  if (!state.hot_e_eta_state_eta.empty()) {
    const std::vector<hsize_t> dims = {
        static_cast<hsize_t>(state.hot_e_eta_state_eta.size())};
    write_numeric_dataset(file, "hydro", "hot_e_eta_model_eta",
                          H5T_NATIVE_DOUBLE, dims,
                          state.hot_e_eta_state_eta.data(), "1", cfg);
    write_numeric_dataset(file, "hydro", "hot_e_eta_model_g",
                          H5T_NATIVE_DOUBLE, dims,
                          state.hot_e_eta_diag_g.data(), "1", cfg);
    write_numeric_dataset(file, "hydro", "hot_e_eta_model_eta_eq",
                          H5T_NATIVE_DOUBLE, dims,
                          state.hot_e_eta_diag_eta_eq.data(), "1", cfg);
    write_numeric_dataset(file, "hydro", "hot_e_eta_model_tau",
                          H5T_NATIVE_DOUBLE, dims,
                          state.hot_e_eta_diag_tau_s.data(), "s", cfg);
    write_numeric_dataset(file, "hydro", "hot_e_eta_model_I14",
                          H5T_NATIVE_DOUBLE, dims,
                          state.hot_e_eta_diag_I14.data(), "1e14 W/cm2", cfg);
    if (!state.hot_e_eta_diag_I14_lower.empty()) {
      TENRYU_ASSERT(
          state.hot_e_eta_diag_I14_lower.size() ==
              state.hot_e_eta_state_eta.size(),
          "HDF5 hot-e eta I14 lower dimensions are inconsistent");
      write_numeric_dataset(file, "hydro", "hot_e_eta_model_I14_lower",
                            H5T_NATIVE_DOUBLE, dims,
                            state.hot_e_eta_diag_I14_lower.data(),
                            "1e14 W/cm2", cfg);
    }
    if (!state.hot_e_eta_diag_I14_upper.empty()) {
      TENRYU_ASSERT(
          state.hot_e_eta_diag_I14_upper.size() ==
              state.hot_e_eta_state_eta.size(),
          "HDF5 hot-e eta I14 upper dimensions are inconsistent");
      write_numeric_dataset(file, "hydro", "hot_e_eta_model_I14_upper",
                            H5T_NATIVE_DOUBLE, dims,
                            state.hot_e_eta_diag_I14_upper.data(),
                            "1e14 W/cm2", cfg);
    }
    if (!state.hot_e_eta_diag_n_sigma.empty()) {
      TENRYU_ASSERT(
          state.hot_e_eta_diag_n_sigma.size() ==
              state.hot_e_eta_state_eta.size(),
          "HDF5 hot-e eta n_sigma dimensions are inconsistent");
      write_numeric_dataset(file, "hydro", "hot_e_eta_model_n_sigma",
                            H5T_NATIVE_DOUBLE, dims,
                            state.hot_e_eta_diag_n_sigma.data(), "count", cfg);
    }
    write_numeric_dataset(file, "hydro", "hot_e_eta_model_Te",
                          H5T_NATIVE_DOUBLE, dims,
                          state.hot_e_eta_diag_Te_keV.data(), "keV", cfg);
    write_numeric_dataset(file, "hydro", "hot_e_eta_model_Ln",
                          H5T_NATIVE_DOUBLE, dims,
                          state.hot_e_eta_diag_Ln_um.data(), "um", cfg);
    write_numeric_dataset(file, "hydro", "hot_e_eta_model_clamped",
                          H5T_NATIVE_DOUBLE, dims,
                          state.hot_e_eta_diag_clamped.data(), "bitmask", cfg);
    write_numeric_dataset(file, "hydro", "hot_e_eta_model_prev_Pcross",
                          H5T_NATIVE_DOUBLE, dims,
                          state.hot_e_eta_prev_Pcross.data(), "erg/s", cfg);
  }
  if (state.hot_e_eps_cum_host.size() == static_cast<std::size_t>(n_cells)) {
    write_numeric_dataset(file, "hydro", "hot_e_eps_cum", H5T_NATIVE_DOUBLE, cdim,
                          state.hot_e_eps_cum_host.data(), "erg/g", cfg);
  }
  if (state.burn_enabled_any &&
      state.burn_rate_host.size() == static_cast<std::size_t>(n_cells)) {
    write_numeric_dataset(file, "hydro", "burn_rate", H5T_NATIVE_DOUBLE, cdim,
                          state.burn_rate_host.data(), "1/cm3/s", cfg);
  }
  if (state.burn_enabled_any &&
      state.burn_Q_e_host.size() == static_cast<std::size_t>(n_cells)) {
    write_numeric_dataset(file, "hydro", "burn_Q_e", H5T_NATIVE_DOUBLE, cdim,
                          state.burn_Q_e_host.data(), "erg/cm3/s", cfg);
  }
  if (state.burn_enabled_any &&
      state.burn_Q_i_host.size() == static_cast<std::size_t>(n_cells)) {
    write_numeric_dataset(file, "hydro", "burn_Q_i", H5T_NATIVE_DOUBLE, cdim,
                          state.burn_Q_i_host.data(), "erg/cm3/s", cfg);
  }
  if (state.burn_enabled_any &&
      state.burn_eps_cum_host.size() == static_cast<std::size_t>(n_cells)) {
    write_numeric_dataset(file, "hydro", "burn_eps_cum", H5T_NATIVE_DOUBLE, cdim,
                          state.burn_eps_cum_host.data(), "erg/g", cfg);
  }
  constexpr std::size_t kBurnSpeciesCount = 5;
  if (state.burn_enabled_any &&
      state.burn_n_host.size() == static_cast<std::size_t>(n_cells) * kBurnSpeciesCount) {
    const char* burn_species_names[kBurnSpeciesCount] = {
        "burn_n_D", "burn_n_T", "burn_n_He3", "burn_n_He4", "burn_n_p"};
    std::vector<double> species(static_cast<std::size_t>(n_cells), 0.0);
    for (std::size_t s = 0; s < kBurnSpeciesCount; ++s) {
      for (std::size_t c = 0; c < static_cast<std::size_t>(n_cells); ++c) {
        species[c] = state.burn_n_host[c * kBurnSpeciesCount + s] * rho[c];
      }
      write_numeric_dataset(file, "hydro", burn_species_names[s], H5T_NATIVE_DOUBLE,
                            cdim, species.data(), "1/cm3", cfg);
    }
  }
  if (state.burn_diffusion_any) {
    const std::size_t G = static_cast<std::size_t>(cfg.burn.diffusion_groups);
    const std::size_t slot_size = G * static_cast<std::size_t>(n_cells);
    if (slot_size > 0U && state.burn_Ng.size() == 6U * slot_size) {
      std::vector<double> burn_Ng_host;
      state.burn_Ng.copy_to_host(burn_Ng_host);
      const std::vector<hsize_t> Ng_dims = {static_cast<hsize_t>(slot_size)};
      for (int slot = 0; slot < 6; ++slot) {
        const std::string name =
            "burn_Ng_slot" + std::to_string(static_cast<long long>(slot));
        write_numeric_dataset(
            file,
            "hydro",
            name,
            H5T_NATIVE_DOUBLE,
            Ng_dims,
            burn_Ng_host.data() + static_cast<std::size_t>(slot) * slot_size,
            "1/g",
            cfg);
      }
    }
  }
  if (state.burn_mc_any) {
    TENRYU_ASSERT(state.burn_mc_live >= 0,
                  "burn MC checkpoint live count is negative");
    const std::size_t live =
        static_cast<std::size_t>(state.burn_mc_live);
    TENRYU_ASSERT(state.burn_mc_r.size() >= live &&
                      state.burn_mc_mu.size() >= live &&
                      state.burn_mc_E.size() >= live &&
                      state.burn_mc_w.size() >= live &&
                      state.burn_mc_slot.size() >= live &&
                      state.burn_mc_alive.size() >= live,
                  "burn MC checkpoint live count exceeds pool size");
    const auto mc_r =
        copy_device_array(state.burn_mc_r.data(), live, "burn_mc_r");
    const auto mc_mu =
        copy_device_array(state.burn_mc_mu.data(), live, "burn_mc_mu");
    const auto mc_E =
        copy_device_array(state.burn_mc_E.data(), live, "burn_mc_E");
    const auto mc_w =
        copy_device_array(state.burn_mc_w.data(), live, "burn_mc_w");
    const auto mc_slot =
        copy_device_array(state.burn_mc_slot.data(), live, "burn_mc_slot");
    const auto mc_alive =
        copy_device_array(state.burn_mc_alive.data(), live, "burn_mc_alive");
    const std::vector<hsize_t> mc_dims = {static_cast<hsize_t>(live)};
    write_numeric_dataset(file, "hydro", "burn_mc_r", H5T_NATIVE_DOUBLE,
                          mc_dims, mc_r.data(), "cm", cfg);
    write_numeric_dataset(file, "hydro", "burn_mc_mu", H5T_NATIVE_DOUBLE,
                          mc_dims, mc_mu.data(), "dimensionless", cfg);
    write_numeric_dataset(file, "hydro", "burn_mc_E", H5T_NATIVE_DOUBLE,
                          mc_dims, mc_E.data(), "erg", cfg);
    write_numeric_dataset(file, "hydro", "burn_mc_w", H5T_NATIVE_DOUBLE,
                          mc_dims, mc_w.data(), "dimensionless", cfg);
    write_numeric_dataset(file, "hydro", "burn_mc_slot", H5T_NATIVE_INT,
                          mc_dims, mc_slot.data(), "index", cfg);
    write_numeric_dataset(file, "hydro", "burn_mc_alive", H5T_NATIVE_UCHAR,
                          mc_dims, mc_alive.data(), "flag", cfg);
  }
  if (!shock_time.empty()) {
    write_numeric_dataset(file,
                          "hydro",
                          "shock_time",
                          H5T_NATIVE_DOUBLE,
                          cdim,
                          shock_time.data(),
                          "s",
                          cfg);
  }
  if (!adaptive_av_gate.empty()) {
    std::vector<std::int8_t> adaptive_av_mode_cell(adaptive_av_gate.size(), 0);
    for (std::size_t i = 0; i < adaptive_av_gate.size(); ++i) {
      adaptive_av_mode_cell[i] = (adaptive_av_gate[i] > 1.0e-6) ? 1 : 0;
    }
    write_numeric_dataset(file,
                          "hydro",
                          "adaptive_av_gate",
                          H5T_NATIVE_DOUBLE,
                          cdim,
                          adaptive_av_gate.data(),
                          "1",
                          cfg);
    write_numeric_dataset(file,
                          "hydro",
                          "adaptive_av_mode_cell",
                          H5T_NATIVE_INT8,
                          cdim,
                          adaptive_av_mode_cell.data(),
                          "1",
                          cfg);
  }
  write_numeric_dataset(file, "hydro", "mass", H5T_NATIVE_DOUBLE, cdim, mass.data(), "g", cfg);
  if (!node_mass.empty()) {
    const std::vector<hsize_t> node_mass_dims =
        mesh::mesh_topo_is_multiblock(cfg.mesh)
            ? std::vector<hsize_t>{static_cast<hsize_t>(state.mesh.topo.n_nodes)}
            : std::vector<hsize_t>{static_cast<hsize_t>(cfg.mesh.nr + 1),
                                   static_cast<hsize_t>(cfg.mesh.nz + 1)};
    write_numeric_dataset(file,
                          "hydro",
                          "node_mass",
                          H5T_NATIVE_DOUBLE,
                          node_mass_dims,
                          node_mass.data(),
                          "g",
                          cfg);
  }
  if (state.corner_mass.size() ==
      static_cast<std::size_t>(state.corner_stride) * n_cells) {
    const auto corner_mass = copy_field_to_host(state.corner_mass);
    write_numeric_dataset(file,
                          "hydro",
                          "corner_mass",
                          H5T_NATIVE_DOUBLE,
                          {static_cast<hsize_t>(n_cells),
                           static_cast<hsize_t>(state.corner_stride)},
                          corner_mass.data(),
                          "g",
                          cfg);
  }
  if (state.corner_stride != 4) {
    TENRYU_ASSERT(
        state.cell_nverts_host.size() == n_cells,
        "HDF5 write: topology v4 requires per-cell cell_nverts");
    const hid_t topology_v4 = ensure_group(file, "mesh/topology/v4");
    TENRYU_ASSERT(topology_v4 >= 0,
                  "HDF5 failed to open group: mesh/topology/v4");
    write_scalar_attribute_i32(
        topology_v4, "corner_stride",
        static_cast<std::int32_t>(state.corner_stride));
    warn_h5_close_failure(H5Gclose(topology_v4), "H5Gclose",
                          "HDF5Writer::write_hydro_group(topology_v4)");
    write_numeric_dataset(
        file, "mesh/topology/v4", "cell_nverts", H5T_NATIVE_INT8,
        {static_cast<hsize_t>(state.cell_nverts_host.size())},
        state.cell_nverts_host.data(), "count", cfg);
  }
  if (state.hllc_mom_z_cell.size() == n_cells) {
    const auto hllc_mom_z_cell = copy_field_to_host(state.hllc_mom_z_cell);
    write_numeric_dataset(file,
                          "hydro",
                          "hllc_mom_z_cell",
                          H5T_NATIVE_DOUBLE,
                          cdim,
                          hllc_mom_z_cell.data(),
                          "g cm/s",
                          cfg);
  }
  if (state.gas_tracer_initialized &&
      state.gas_tracer_Y.size() == n_cells) {
    const auto gas_tracer_Y = copy_field_to_host(state.gas_tracer_Y);
    write_numeric_dataset(file,
                          "hydro",
                          "gas_tracer_Y",
                          H5T_NATIVE_DOUBLE,
                          cdim,
                          gas_tracer_Y.data(),
                          "1",
                          cfg);
  }
  write_numeric_dataset(file, "hydro", "vol", H5T_NATIVE_DOUBLE, cdim, vol.data(), "cm3", cfg);
  write_numeric_dataset(file,
                        "hydro",
                        "zbar",
                        H5T_NATIVE_DOUBLE,
                        cdim,
                        zbar.data(),
                        "dimensionless",
                        cfg);
  if (cfg.numerics.materials.per_material_conservation_enabled) {
    auto cv_e = resize_payload(copy_field_to_host(state.cv_e), n_cells, "cv_e", false);
    auto cv_i = resize_payload(copy_field_to_host(state.cv_i), n_cells, "cv_i", false);
    auto cs = resize_payload(copy_field_to_host(state.cs), n_cells, "cs", false);
    write_numeric_dataset(file,
                          "hydro",
                          "cv_e",
                          H5T_NATIVE_DOUBLE,
                          cdim,
                          cv_e.data(),
                          "erg/(g*eV)",
                          cfg);
    write_numeric_dataset(file,
                          "hydro",
                          "cv_i",
                          H5T_NATIVE_DOUBLE,
                          cdim,
                          cv_i.data(),
                          "erg/(g*eV)",
                          cfg);
    write_numeric_dataset(file,
                          "hydro",
                          "cs",
                          H5T_NATIVE_DOUBLE,
                          cdim,
                          cs.data(),
                          "cm/s",
                          cfg);
  }
  if (!eta_compatible.empty()) {
    write_numeric_dataset(file,
                          "hydro",
                          "eta_compatible",
                          H5T_NATIVE_DOUBLE,
                          cdim,
                          eta_compatible.data(),
                          "cm3",
                          cfg);
  }

  const std::size_t expected_volfrac = n_cells * n_mat;
  auto volfrac_payload = volFrac;
  if (volfrac_payload.size() != expected_volfrac) {
    core::log_warning("Checkpoint write: hydro/volFrac size mismatch (state=" +
                      std::to_string(volfrac_payload.size()) + ", expected=" +
                      std::to_string(expected_volfrac) + "); resizing with zero fill.");
    volfrac_payload.resize(expected_volfrac, 0.0);
  }
  const double volfrac_fallback = 0.0;
  const double* volfrac_ptr = volfrac_payload.empty() ? &volfrac_fallback : volfrac_payload.data();
  write_numeric_dataset(file,
                        "hydro",
                        "volFrac",
                        H5T_NATIVE_DOUBLE,
                        {static_cast<hsize_t>(n_cells), static_cast<hsize_t>(n_mat)},
                        volfrac_ptr,
                        "fraction",
                        cfg);
  write_per_material_schema_group(file, state, cfg);
}

void write_radiation_group(const hid_t file,
                           const core::State& state,
                           const core::Config& cfg) {
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_groups = static_cast<std::size_t>(std::max(cfg.radiation.groups, 1));
  const std::size_t n_cell_groups = n_cells * n_groups;

  auto rad_E = copy_field_to_host(state.rad_E);
  auto rad_dep = copy_field_to_host(state.rad_dep);
  auto rad_emit = copy_field_to_host(state.rad_emit);
  auto fld_fleck = copy_field_to_host(state.fld_fleck);
  auto diag_rad_E_pre = copy_field_to_host(state.sn_diag_rad_E_pre);
  auto diag_rad_E_post = copy_field_to_host(state.sn_diag_rad_E_post);
  auto diag_rad_emission_at_Tn =
      copy_field_to_host(state.sn_diag_rad_emission_at_Tn);
  auto diag_rad_emission_at_Tnp1 =
      copy_field_to_host(state.sn_diag_rad_emission_at_Tnp1);
  auto diag_rad_absorption = copy_field_to_host(state.sn_diag_rad_absorption);
  auto diag_clip_energy = copy_field_to_host(state.sn_diag_clip_energy);
  auto diag_clip_full_deficit =
      copy_field_to_host(state.sn_diag_clip_full_deficit);
  auto diag_chi_opacity = copy_field_to_host(state.sn_diag_chi_opacity);
  auto diag_F_first_moment =
      copy_field_to_host(state.sn_diag_F_first_moment);
  auto diag_E_star_flux = copy_field_to_host(state.sn_diag_E_star_flux);
  auto diag_stream_theta = copy_field_to_host(state.sn_stream_theta);
  auto diag_ap_alpha_face = copy_field_to_host(state.sn_face_alpha);
  auto sn_tau_R = copy_field_to_host(state.sn_tau_R);
  auto sn_reduced_flux = copy_field_to_host(state.sn_reduced_flux);
  auto sn_ap_alpha = copy_field_to_host(state.sn_ap_alpha);
  auto sn_F_z = copy_field_to_host(state.sn_F_z);
  auto sn_P_zz = copy_field_to_host(state.sn_P_zz);
  auto sn_chi_z = copy_field_to_host(state.sn_chi_z);
  auto sn_radial_fixup_count = copy_field_to_host(state.sn_radial_fixup_count);
  auto sn_radial_fixup_artificial_abs =
      copy_field_to_host(state.sn_radial_fixup_artificial_abs);
  auto sn_angular_fixup_count = copy_field_to_host(state.sn_angular_fixup_count);
  auto sn_angular_fixup_artificial_abs =
      copy_field_to_host(state.sn_angular_fixup_artificial_abs);
  auto holo_E_LO = copy_field_to_host(state.holo_E_LO);
  auto holo_consistency_source =
      copy_field_to_host(state.holo_consistency_source);
  auto holo_rad_dep = copy_field_to_host(state.holo_rad_dep);
  auto holo_rad_emit = copy_field_to_host(state.holo_rad_emit);
  auto holo_Prr = copy_field_to_host(state.holo_Prr);
  auto holo_chi = copy_field_to_host(state.holo_chi);
  auto holo_Prr_coverage = copy_field_to_host(state.holo_Prr_coverage);
  auto difference_W = copy_field_to_host(state.difference_W);
  auto difference_E_ref = copy_field_to_host(state.difference_E_ref);
  auto difference_residual_E = copy_field_to_host(state.difference_residual_E);
  const bool has_delta_E_rad_prev = !state.delta_E_rad_prev.empty();
  auto delta_E_rad_prev = copy_field_to_host(state.delta_E_rad_prev);
  const auto vol = copy_field_to_host(state.vol);
  const bool is_2d_rz = cfg.main.dim == 2 || cfg.main.dimension == "2D_RZ";
  const bool write_fld_diagnostics =
      cfg.radiation.enabled && is_2d_rz &&
      cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion;
  const bool write_sn_diagnostics =
      cfg.radiation.enabled && is_2d_rz &&
      cfg.radiation.mode == core::RadiationMode::SnTransport;

  if (rad_E.size() != n_cell_groups) {
    core::log_warning("Checkpoint write: radiation/energy_density size mismatch (state=" +
                      std::to_string(rad_E.size()) + ", expected=" +
                      std::to_string(n_cell_groups) +
                      "); resizing with zero fill.");
  }
  if (rad_dep.size() != n_cell_groups) {
    core::log_warning("Checkpoint write: radiation/rad_dep size mismatch (state=" +
                      std::to_string(rad_dep.size()) + ", expected=" +
                      std::to_string(n_cell_groups) +
                      "); resizing with zero fill.");
  }
  if (rad_emit.size() != n_cell_groups) {
    core::log_warning("Checkpoint write: radiation/rad_emit size mismatch (state=" +
                      std::to_string(rad_emit.size()) + ", expected=" +
                      std::to_string(n_cell_groups) +
                      "); resizing with zero fill.");
  }
  if (state.holo_core_mask_valid && holo_E_LO.size() != n_cell_groups) {
    core::log_warning("Checkpoint write: holo/E_LO size mismatch (state=" +
                      std::to_string(holo_E_LO.size()) + ", expected=" +
                      std::to_string(n_cell_groups) +
                      "); resizing with zero fill.");
  }
  if (state.holo_core_mask_valid &&
      holo_consistency_source.size() != n_cell_groups) {
    core::log_warning("Checkpoint write: holo/consistency_source size mismatch (state=" +
                      std::to_string(holo_consistency_source.size()) + ", expected=" +
                      std::to_string(n_cell_groups) +
                      "); resizing with zero fill.");
  }
  if (state.holo_core_mask_valid && holo_Prr.size() != n_cell_groups) {
    core::log_warning("Checkpoint write: holo/Prr_HO size mismatch (state=" +
                      std::to_string(holo_Prr.size()) + ", expected=" +
                      std::to_string(n_cell_groups) +
                      "); resizing with zero fill.");
  }
  rad_E.resize(n_cell_groups, 0.0);
  rad_dep.resize(n_cell_groups, 0.0);
  rad_emit.resize(n_cell_groups, 0.0);
  if (write_fld_diagnostics) {
    fld_fleck.resize(n_cells, 0.0);
  }
  diag_rad_E_pre.resize(n_cell_groups, 0.0);
  diag_rad_E_post.resize(n_cell_groups, 0.0);
  diag_rad_emission_at_Tn.resize(n_cell_groups, 0.0);
  diag_rad_emission_at_Tnp1.resize(n_cell_groups, 0.0);
  diag_rad_absorption.resize(n_cell_groups, 0.0);
  diag_clip_energy.resize(n_cell_groups, 0.0);
  diag_clip_full_deficit.resize(n_cell_groups, 0.0);
  diag_chi_opacity.resize(n_cell_groups, 0.0);
  diag_F_first_moment.resize(n_cell_groups, 0.0);
  diag_E_star_flux.resize(n_cell_groups, 0.0);
  diag_stream_theta.resize(n_cell_groups, 0.0);
  const std::size_t n_sn_faces =
      (cfg.main.dim == 1 || cfg.main.dimension == "1D_SPH") ? (n_cells + 1U) : 0U;
  const std::size_t n_face_groups = n_sn_faces * n_groups;
  diag_ap_alpha_face.resize(n_face_groups, 0.0);
  sn_radial_fixup_count.resize(n_cell_groups, 0.0);
  sn_radial_fixup_artificial_abs.resize(n_cell_groups, 0.0);
  sn_angular_fixup_count.resize(n_cell_groups, 0.0);
  sn_angular_fixup_artificial_abs.resize(n_cell_groups, 0.0);
  holo_E_LO.resize(n_cell_groups, 0.0);
  holo_consistency_source.resize(n_cell_groups, 0.0);
  holo_rad_dep.resize(n_cell_groups, 0.0);
  holo_rad_emit.resize(n_cell_groups, 0.0);
  holo_Prr.resize(n_cell_groups, 0.0);
  holo_chi.resize(n_cell_groups, 0.0);
  holo_Prr_coverage.resize(n_cell_groups, 0.0);
  if (has_delta_E_rad_prev) {
    delta_E_rad_prev.resize(n_cells, 0.0);
  }
  if (write_sn_diagnostics) {
    sn_tau_R.resize(n_cells, 0.0);
    sn_reduced_flux.resize(n_cells, 0.0);
    sn_ap_alpha.resize(n_cells, 0.0);
    sn_F_z.resize(n_cell_groups, 0.0);
    sn_P_zz.resize(n_cell_groups, 0.0);
    sn_chi_z.resize(n_cell_groups, 0.0);
  }

  const auto deposited_power = compute_power_density(rad_dep, vol, n_cells, n_groups, state.dt);

  const std::vector<hsize_t> cgdims = {
      static_cast<hsize_t>(n_cells), static_cast<hsize_t>(n_groups)};
  write_numeric_dataset(file,
                        "radiation",
                        "energy_density",
                        H5T_NATIVE_DOUBLE,
                        cgdims,
                        rad_E.data(),
                        "erg/cm3",
                        cfg);
  write_numeric_dataset(file,
                        "radiation",
                        "rad_dep",
                        H5T_NATIVE_DOUBLE,
                        cgdims,
                        rad_dep.data(),
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "radiation",
                        "rad_emit",
                        H5T_NATIVE_DOUBLE,
                        cgdims,
                        rad_emit.data(),
                        "erg",
                        cfg);
  if (has_delta_E_rad_prev) {
    write_numeric_dataset(file,
                          "radiation",
                          "delta_E_rad_prev",
                          H5T_NATIVE_DOUBLE,
                          {static_cast<hsize_t>(n_cells)},
                          delta_E_rad_prev.data(),
                          "erg",
                          cfg);
  }
  write_numeric_dataset(file,
                        "radiation",
                        "deposited_power",
                        H5T_NATIVE_DOUBLE,
                        cgdims,
                        deposited_power.data(),
                        "erg/cm3/s",
                        cfg);
  if (write_fld_diagnostics) {
    write_numeric_dataset(file,
                          "radiation",
                          "fleck_factor",
                          H5T_NATIVE_DOUBLE,
                          {static_cast<hsize_t>(n_cells)},
                          fld_fleck.data(),
                          "dimensionless",
                          cfg);
  }
  if (write_sn_diagnostics) {
    write_numeric_dataset(file,
                          "radiation",
                          "sn_tau_R",
                          H5T_NATIVE_DOUBLE,
                          {static_cast<hsize_t>(n_cells)},
                          sn_tau_R.data(),
                          "dimensionless",
                          cfg);
    write_numeric_dataset(file,
                          "radiation",
                          "sn_reduced_flux",
                          H5T_NATIVE_DOUBLE,
                          {static_cast<hsize_t>(n_cells)},
                          sn_reduced_flux.data(),
                          "dimensionless",
                          cfg);
    write_numeric_dataset(file,
                          "radiation",
                          "sn_ap_alpha",
                          H5T_NATIVE_DOUBLE,
                          {static_cast<hsize_t>(n_cells)},
                          sn_ap_alpha.data(),
                          "dimensionless",
                          cfg);
    write_numeric_dataset(file,
                          "radiation",
                          "sn_F_z",
                          H5T_NATIVE_DOUBLE,
                          cgdims,
                          sn_F_z.data(),
                          "erg/cm2/s",
                          cfg);
    write_numeric_dataset(file,
                          "radiation",
                          "sn_P_zz",
                          H5T_NATIVE_DOUBLE,
                          cgdims,
                          sn_P_zz.data(),
                          "erg/cm3",
                          cfg);
    write_numeric_dataset(file,
                          "radiation",
                          "sn_chi_z",
                          H5T_NATIVE_DOUBLE,
                          cgdims,
                          sn_chi_z.data(),
                          "dimensionless",
                          cfg);
  }
  write_numeric_dataset(file,
                        "radiation",
                        "diag_rad_E_pre",
                        H5T_NATIVE_DOUBLE,
                        cgdims,
                        diag_rad_E_pre.data(),
                        "erg/cm3",
                        cfg);
  write_numeric_dataset(file,
                        "radiation",
                        "diag_rad_E_post",
                        H5T_NATIVE_DOUBLE,
                        cgdims,
                        diag_rad_E_post.data(),
                        "erg/cm3",
                        cfg);
  write_numeric_dataset(file,
                        "radiation",
                        "diag_rad_emission_at_Tn",
                        H5T_NATIVE_DOUBLE,
                        cgdims,
                        diag_rad_emission_at_Tn.data(),
                        "erg/cm3",
                        cfg);
  write_numeric_dataset(file,
                        "radiation",
                        "diag_rad_emission_at_Tnp1",
                        H5T_NATIVE_DOUBLE,
                        cgdims,
                        diag_rad_emission_at_Tnp1.data(),
                        "erg/cm3",
                        cfg);
  write_numeric_dataset(file,
                        "radiation",
                        "diag_rad_absorption",
                        H5T_NATIVE_DOUBLE,
                        cgdims,
                        diag_rad_absorption.data(),
                        "erg/cm3",
                        cfg);
  write_numeric_dataset(file,
                        "radiation",
                        "diag_clip_energy",
                        H5T_NATIVE_DOUBLE,
                        cgdims,
                        diag_clip_energy.data(),
                        "erg/cm3",
                        cfg);
  write_numeric_dataset(file,
                        "radiation",
                        "diag_clip_full_deficit",
                        H5T_NATIVE_DOUBLE,
                        cgdims,
                        diag_clip_full_deficit.data(),
                        "erg/cm3",
                        cfg);
  write_numeric_dataset(file,
                        "radiation",
                        "diag_chi_opacity",
                        H5T_NATIVE_DOUBLE,
                        cgdims,
                        diag_chi_opacity.data(),
                        "dimensionless",
                        cfg);
  write_numeric_dataset(file,
                        "radiation",
                        "diag_F_first_moment",
                        H5T_NATIVE_DOUBLE,
                        cgdims,
                        diag_F_first_moment.data(),
                        "erg/cm2/s",
                        cfg);
  write_numeric_dataset(file,
                        "radiation",
                        "diag_E_star_flux",
                        H5T_NATIVE_DOUBLE,
                        cgdims,
                        diag_E_star_flux.data(),
                        "erg/cm3",
                        cfg);
  write_numeric_dataset(file,
                        "radiation",
                        "diag_stream_theta",
                        H5T_NATIVE_DOUBLE,
                        cgdims,
                        diag_stream_theta.data(),
                        "dimensionless",
                        cfg);
  if (n_face_groups > 0U) {
    write_numeric_dataset(file,
                          "radiation",
                          "diag_ap_alpha_face",
                          H5T_NATIVE_DOUBLE,
                          {static_cast<hsize_t>(n_sn_faces),
                           static_cast<hsize_t>(n_groups)},
                          diag_ap_alpha_face.data(),
                          "dimensionless",
                          cfg);
  }
  write_numeric_dataset(file,
                        "radiation",
                        "sn_radial_fixup_count",
                        H5T_NATIVE_DOUBLE,
                        cgdims,
                        sn_radial_fixup_count.data(),
                        "count",
                        cfg);
  write_numeric_dataset(file,
                        "radiation",
                        "sn_radial_fixup_artificial_abs",
                        H5T_NATIVE_DOUBLE,
                        cgdims,
                        sn_radial_fixup_artificial_abs.data(),
                        "erg/cm2/s",
                        cfg);
  write_numeric_dataset(file,
                        "radiation",
                        "sn_angular_fixup_count",
                        H5T_NATIVE_DOUBLE,
                        cgdims,
                        sn_angular_fixup_count.data(),
                        "count",
                        cfg);
  write_numeric_dataset(file,
                        "radiation",
                        "sn_angular_fixup_artificial_abs",
                        H5T_NATIVE_DOUBLE,
                        cgdims,
                        sn_angular_fixup_artificial_abs.data(),
                        "erg/cm2/s",
                        cfg);

  if (state.ddmc_mode_map_valid && state.ddmc_mode_map.size() == n_cell_groups) {
    write_numeric_dataset(file,
                          "radiation",
                          "ddmc_flag",
                          H5T_NATIVE_INT8,
                          cgdims,
                          state.ddmc_mode_map.data(),
                          "flag",
                          cfg);
  }

  if (state.holo_core_mask_valid && state.holo_core_mask.size() == n_cells) {
    write_numeric_dataset(file,
                          "holo",
                          "E_LO",
                          H5T_NATIVE_DOUBLE,
                          cgdims,
                          holo_E_LO.data(),
                          "erg/cm3",
                          cfg);
    write_numeric_dataset(file,
                          "holo",
                          "consistency_source",
                          H5T_NATIVE_DOUBLE,
                          cgdims,
                          holo_consistency_source.data(),
                          "erg/s",
                          cfg);
    write_numeric_dataset(file,
                          "holo",
                          "rad_dep_LO",
                          H5T_NATIVE_DOUBLE,
                          cgdims,
                          holo_rad_dep.data(),
                          "erg",
                          cfg);
    write_numeric_dataset(file,
                          "holo",
                          "rad_emit_LO",
                          H5T_NATIVE_DOUBLE,
                          cgdims,
                          holo_rad_emit.data(),
                          "erg",
                          cfg);
    write_numeric_dataset(file,
                          "holo",
                          "Prr_HO",
                          H5T_NATIVE_DOUBLE,
                          cgdims,
                          holo_Prr.data(),
                          "erg/cm3",
                          cfg);
    write_numeric_dataset(file,
                          "holo",
                          "chi",
                          H5T_NATIVE_DOUBLE,
                          cgdims,
                          holo_chi.data(),
                          "dimensionless",
                          cfg);
    write_numeric_dataset(file,
                          "holo",
                          "Prr_coverage",
                          H5T_NATIVE_DOUBLE,
                          cgdims,
                          holo_Prr_coverage.data(),
                          "dimensionless",
                          cfg);
    write_numeric_dataset(file,
                          "holo",
                          "core_mask",
                          H5T_NATIVE_UINT8,
                          {static_cast<hsize_t>(n_cells)},
                          state.holo_core_mask.data(),
                          "flag",
                          cfg);
    if (state.holo_core_prev_mask.size() == n_cells) {
      write_numeric_dataset(file,
                            "holo",
                            "prev_core_mask",
                            H5T_NATIVE_UINT8,
                            {static_cast<hsize_t>(n_cells)},
                            state.holo_core_prev_mask.data(),
                            "flag",
                            cfg);
    }
    if (state.holo_hold_count.size() == n_cells) {
      write_numeric_dataset(file,
                            "holo",
                            "hold_count",
                            H5T_NATIVE_INT32,
                            {static_cast<hsize_t>(n_cells)},
                            state.holo_hold_count.data(),
                            "count",
                            cfg);
    }
    if (state.holo_dwell_count.size() == n_cells) {
      write_numeric_dataset(file,
                            "holo",
                            "dwell_count",
                            H5T_NATIVE_INT32,
                            {static_cast<hsize_t>(n_cells)},
                            state.holo_dwell_count.data(),
                            "count",
                            cfg);
    }
    if (state.holo_tau_R.size() == n_cells) {
      write_numeric_dataset(file,
                            "holo",
                            "tau_R",
                            H5T_NATIVE_DOUBLE,
                            {static_cast<hsize_t>(n_cells)},
                            state.holo_tau_R.data(),
                            "dimensionless",
                            cfg);
    }
    if (state.holo_reduced_flux.size() == n_cells) {
      write_numeric_dataset(file,
                            "holo",
                            "reduced_flux",
                            H5T_NATIVE_DOUBLE,
                            {static_cast<hsize_t>(n_cells)},
                            state.holo_reduced_flux.data(),
                            "dimensionless",
                            cfg);
    }
    if (state.holo_mass_q.size() == n_cells) {
      write_numeric_dataset(file,
                            "holo",
                            "mass_q",
                            H5T_NATIVE_DOUBLE,
                            {static_cast<hsize_t>(n_cells)},
                            state.holo_mass_q.data(),
                            "dimensionless",
                            cfg);
    }
  }

  if (cfg.radiation.imc.difference.enabled) {
    if (difference_W.size() == n_cells) {
      write_numeric_dataset(file,
                            "difference",
                            "W",
                            H5T_NATIVE_DOUBLE,
                            {static_cast<hsize_t>(n_cells)},
                            difference_W.data(),
                            "dimensionless",
                            cfg);
    }
    if (difference_E_ref.size() == n_cell_groups) {
      write_numeric_dataset(file,
                            "difference",
                            "E_ref",
                            H5T_NATIVE_DOUBLE,
                            cgdims,
                            difference_E_ref.data(),
                            "erg/cm3",
                            cfg);
    }
    if (difference_residual_E.size() == n_cell_groups) {
      write_numeric_dataset(file,
                            "difference",
                            "residual_energy_density",
                            H5T_NATIVE_DOUBLE,
                            cgdims,
                            difference_residual_E.data(),
                            "erg/cm3",
                            cfg);
    }
  }

}

void write_laser_mesh_group(const hid_t file,
                            const core::State& state,
                            const core::Config& cfg) {
  if (state.laser_mesh_n_nodes_r <= 0 || state.laser_mesh_n_nodes_z <= 0) {
    return;
  }
  const auto nr = static_cast<hsize_t>(state.laser_mesh_n_nodes_r);
  const auto nz = static_cast<hsize_t>(state.laser_mesh_n_nodes_z);

  const std::int32_t nr_i32 = static_cast<std::int32_t>(state.laser_mesh_n_nodes_r);
  const std::int32_t nz_i32 = static_cast<std::int32_t>(state.laser_mesh_n_nodes_z);
  write_numeric_dataset(file, "laser/mesh", "n_nodes_r", H5T_NATIVE_INT32, {}, &nr_i32, "count", cfg);
  write_numeric_dataset(file, "laser/mesh", "n_nodes_z", H5T_NATIVE_INT32, {}, &nz_i32, "count", cfg);
  write_numeric_dataset(file, "laser/mesh", "n_crit", H5T_NATIVE_DOUBLE, {}, &state.laser_mesh_n_crit, "1/cm3", cfg);

  write_numeric_dataset(file, "laser/mesh", "node_R", H5T_NATIVE_DOUBLE, {nr}, state.laser_mesh_node_R.data(), "cm", cfg);
  write_numeric_dataset(file, "laser/mesh", "node_Z", H5T_NATIVE_DOUBLE, {nz}, state.laser_mesh_node_Z.data(), "cm", cfg);

  const hsize_t nn = nr * nz;
  write_numeric_dataset(file, "laser/mesh", "n_e_hat", H5T_NATIVE_DOUBLE, {nn}, state.laser_mesh_n_e_hat.data(), "n/n_crit", cfg);
  write_numeric_dataset(file, "laser/mesh", "T_e", H5T_NATIVE_DOUBLE, {nn}, state.laser_mesh_T_e.data(), "eV", cfg);
  write_numeric_dataset(file, "laser/mesh", "Zbar", H5T_NATIVE_DOUBLE, {nn}, state.laser_mesh_Zbar.data(), "dimensionless", cfg);
  write_numeric_dataset(file, "laser/mesh", "grad_n_hat_R", H5T_NATIVE_DOUBLE, {nn}, state.laser_mesh_grad_R.data(), "1/cm", cfg);
  write_numeric_dataset(file, "laser/mesh", "grad_n_hat_Z", H5T_NATIVE_DOUBLE, {nn}, state.laser_mesh_grad_Z.data(), "1/cm", cfg);
}

void write_ray_trajectory_group(const hid_t file,
                                const core::State& state,
                                const core::Config& cfg) {
  if (!cfg.laser.ray_output_trajectory || state.ray_traj_offsets.empty()) {
    return;
  }
  const std::size_t n_rays = state.ray_traj_step_counts.size();
  if (n_rays == 0) {
    return;
  }
  const auto total_steps = static_cast<hsize_t>(state.ray_traj_offsets.back());
  if (total_steps == 0) {
    return;
  }
  const std::int32_t n_rays_i32 = static_cast<std::int32_t>(n_rays);
  const std::vector<hsize_t> ray_dims = {static_cast<hsize_t>(n_rays)};
  const std::vector<hsize_t> offset_dims = {static_cast<hsize_t>(n_rays + 1)};
  const std::vector<hsize_t> step_dims = {total_steps};

  write_numeric_dataset(file, "laser/rays/trajectory", "n_rays", H5T_NATIVE_INT32, {}, &n_rays_i32, "count", cfg);
  write_numeric_dataset(file, "laser/rays/trajectory", "offsets", H5T_NATIVE_INT64, offset_dims, state.ray_traj_offsets.data(), "index", cfg);
  write_numeric_dataset(file, "laser/rays/trajectory", "step_count", H5T_NATIVE_INT32, ray_dims, state.ray_traj_step_counts.data(), "count", cfg);
  write_numeric_dataset(file, "laser/rays/trajectory", "beam_id", H5T_NATIVE_INT32, ray_dims, state.ray_traj_beam_ids.data(), "id", cfg);

  if (state.ray_traj_is_3d) {
    write_numeric_dataset(file, "laser/rays/trajectory", "pos_x", H5T_NATIVE_DOUBLE, step_dims, state.ray_traj_pos1.data(), "cm", cfg);
    write_numeric_dataset(file, "laser/rays/trajectory", "pos_y", H5T_NATIVE_DOUBLE, step_dims, state.ray_traj_pos2.data(), "cm", cfg);
    write_numeric_dataset(file, "laser/rays/trajectory", "pos_z", H5T_NATIVE_DOUBLE, step_dims, state.ray_traj_pos3.data(), "cm", cfg);
  } else {
    write_numeric_dataset(file, "laser/rays/trajectory", "pos_R", H5T_NATIVE_DOUBLE, step_dims, state.ray_traj_pos1.data(), "cm", cfg);
    write_numeric_dataset(file, "laser/rays/trajectory", "pos_Z", H5T_NATIVE_DOUBLE, step_dims, state.ray_traj_pos2.data(), "cm", cfg);
  }
  write_numeric_dataset(file, "laser/rays/trajectory", "power", H5T_NATIVE_DOUBLE, step_dims, state.ray_traj_power.data(), "erg/s", cfg);
}

void write_laser_group(const hid_t file,
                       const core::State& state,
                       const core::Config& cfg) {
  if (!cfg.laser.enabled) {
    return;
  }

  const auto dep = copy_field_to_host(state.laser_dep);
  const auto vol = copy_field_to_host(state.vol);
  const auto power = compute_laser_power_density(dep, vol, state.dt);
  const std::size_t n_cells = state.rho.size();
  std::vector<double> aq_r_shells;
  std::vector<double> aq_z_shells;
  const bool is_2d_rz = cfg.main.dim == 2 || cfg.main.dimension == "2D_RZ";
  const bool write_aq_shells =
      is_2d_rz && cfg.mesh.nr > 0 && cfg.mesh.nz > 0;
  if (write_aq_shells) {
    compute_aq_shells_2d_rz(dep, cfg.mesh.nr, cfg.mesh.nz, aq_r_shells, aq_z_shells);
  }

  const double absorbed = std::max(state.E_laser_deposited, 0.0);
  const double escaped = std::max(state.E_laser_escaped, 0.0);
  const double denom = absorbed + escaped + state.E_cbet_iaw;
  const double absorption_fraction = (denom > 0.0) ? (absorbed / denom) : 0.0;
  const double E_ra = std::max(state.E_ra_deposited, 0.0);
  auto ray_density = copy_field_to_host(state.ray_density);
  ray_density.resize(n_cells, 0.0);

  write_numeric_dataset(file,
                        "laser",
                        "deposited_power",
                        H5T_NATIVE_DOUBLE,
                        {static_cast<hsize_t>(power.size())},
                        power.data(),
                        "erg/cm3/s",
                        cfg);
  write_numeric_dataset(file,
                        "laser",
                        "absorption_fraction",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &absorption_fraction,
                        "fraction",
                        cfg);
  write_numeric_dataset(file,
                        "laser",
                        "E_ra",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &E_ra,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "laser",
                        "E_cbet_iaw_step",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.E_cbet_iaw_step,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "laser",
                        "E_cbet_iaw",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.E_cbet_iaw,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "laser",
                        "ray_density",
                        H5T_NATIVE_DOUBLE,
                        {static_cast<hsize_t>(ray_density.size())},
                        ray_density.data(),
                        "1/cm3",
                        cfg);
  if (write_aq_shells) {
    write_numeric_dataset(file,
                          "laser",
                          "A_Q_r_shells",
                          H5T_NATIVE_DOUBLE,
                          {static_cast<hsize_t>(aq_r_shells.size())},
                          aq_r_shells.data(),
                          "dimensionless",
                          cfg);
    write_numeric_dataset(file,
                          "laser",
                          "A_Q_z_shells",
                          H5T_NATIVE_DOUBLE,
                          {static_cast<hsize_t>(aq_z_shells.size())},
                          aq_z_shells.data(),
                          "dimensionless",
                          cfg);
  }

  if (!state.ps_sky_I_tot.empty() || !state.ps_ray_map.empty()) {
    constexpr std::size_t kRayMapThetaBins = 64;
    constexpr std::size_t kRayMapSheets = 2;
    if (!state.ps_sky_I_tot.empty()) {
      const std::size_t n_mu = state.ps_sky_mu.size();
      const std::size_t n_phi = state.ps_sky_phi.size();
      TENRYU_ASSERT(
          n_mu > 0 && n_phi > 0 &&
              state.ps_sky_I_tot.size() == n_mu * n_phi &&
              state.ps_sky_I_cw.size() == n_mu * n_phi &&
              state.ps_sky_n_sigma.size() == n_mu * n_phi,
          "HDF5 port_section sky-map dimensions are inconsistent");
      write_numeric_dataset(
          file, "laser/port_section", "grid_mu", H5T_NATIVE_DOUBLE,
          {static_cast<hsize_t>(n_mu)}, state.ps_sky_mu.data(), "1", cfg);
      write_numeric_dataset(
          file, "laser/port_section", "grid_phi", H5T_NATIVE_DOUBLE,
          {static_cast<hsize_t>(n_phi)}, state.ps_sky_phi.data(), "rad", cfg);
      const std::vector<hsize_t> sky_dims{
          static_cast<hsize_t>(n_mu), static_cast<hsize_t>(n_phi)};
      write_numeric_dataset(
          file, "laser/port_section", "I_tot_eval", H5T_NATIVE_DOUBLE,
          sky_dims, state.ps_sky_I_tot.data(), "W/cm2", cfg);
      write_numeric_dataset(
          file, "laser/port_section", "I_cw_eval", H5T_NATIVE_DOUBLE,
          sky_dims, state.ps_sky_I_cw.data(), "W/cm2", cfg);
      write_numeric_dataset(
          file, "laser/port_section", "n_sigma_eval", H5T_NATIVE_DOUBLE,
          sky_dims, state.ps_sky_n_sigma.data(), "count", cfg);
    }
    if (!state.ps_ray_map.empty()) {
      const std::size_t n_shells = state.ps_ray_map_shell_r.size();
      TENRYU_ASSERT(
          n_shells > 0 &&
              state.ps_ray_map.size() ==
                  n_shells * kRayMapThetaBins * kRayMapSheets,
          "HDF5 port_section ray-map dimensions are inconsistent");
      write_numeric_dataset(
          file, "laser/port_section", "ray_map", H5T_NATIVE_DOUBLE,
          {static_cast<hsize_t>(n_shells),
           static_cast<hsize_t>(kRayMapThetaBins),
           static_cast<hsize_t>(kRayMapSheets)},
          state.ps_ray_map.data(), "W/cm2", cfg);
      write_numeric_dataset(
          file, "laser/port_section", "ray_map_shell_r",
          H5T_NATIVE_DOUBLE, {static_cast<hsize_t>(n_shells)},
          state.ps_ray_map_shell_r.data(), "cm", cfg);
    }
    if (!state.ps_port_outgoing_power.empty()) {
      write_numeric_dataset(
          file, "laser/port_section", "port_outgoing_power",
          H5T_NATIVE_DOUBLE,
          {static_cast<hsize_t>(state.ps_port_outgoing_power.size())},
          state.ps_port_outgoing_power.data(), "erg/s", cfg);
    }
    if (!state.ps_port_capture_pcross.empty()) {
      const std::size_t n_ports = state.ps_port_outgoing_power.size();
      TENRYU_ASSERT(
          n_ports > 0 &&
              state.ps_port_capture_pcross.size() % n_ports == 0,
          "HDF5 port_section capture dimensions are inconsistent");
      const std::size_t n_channels =
          state.ps_port_capture_pcross.size() / n_ports;
      write_numeric_dataset(
          file, "laser/port_section", "port_capture_Pcross",
          H5T_NATIVE_DOUBLE,
          {static_cast<hsize_t>(n_ports),
           static_cast<hsize_t>(n_channels)},
          state.ps_port_capture_pcross.data(), "erg/s", cfg);
    }
    write_numeric_dataset(
        file, "laser/port_section", "f_illum2", H5T_NATIVE_DOUBLE, {},
        &state.ps_f_illum2, "1", cfg);
    write_numeric_dataset(
        file, "laser/port_section", "f_union", H5T_NATIVE_DOUBLE, {},
        &state.ps_f_union, "1", cfg);
  }

  if (cfg.laser.ray_output_count <= 0 || state.laser_ray_beam_id.empty()) {
    return;
  }
  std::vector<std::int32_t> beam_id;
  beam_id.reserve(state.laser_ray_beam_id.size());
  for (const int id : state.laser_ray_beam_id) {
    beam_id.push_back(static_cast<std::int32_t>(id));
  }
  const std::size_t n = beam_id.size();
  if (n == 0 || state.laser_ray_power0.size() != n) {
    return;
  }
  const std::int32_t n_rays = static_cast<std::int32_t>(n);
  const std::vector<hsize_t> dims(1, static_cast<hsize_t>(n));
  write_numeric_dataset(file, "laser/rays", "n_rays", H5T_NATIVE_INT32, {}, &n_rays, "count", cfg);
  if (state.laser_ray_is_3d) {
    if (state.laser_ray_x0.size() != n || state.laser_ray_y0.size() != n ||
        state.laser_ray_z0.size() != n || state.laser_ray_vx0.size() != n ||
        state.laser_ray_vy0.size() != n || state.laser_ray_vz0.size() != n) {
      return;
    }
    write_numeric_dataset(file,
                          "laser/rays",
                          "x0",
                          H5T_NATIVE_DOUBLE,
                          dims,
                          state.laser_ray_x0.data(),
                          "cm",
                          cfg);
    write_numeric_dataset(file,
                          "laser/rays",
                          "y0",
                          H5T_NATIVE_DOUBLE,
                          dims,
                          state.laser_ray_y0.data(),
                          "cm",
                          cfg);
    write_numeric_dataset(file,
                          "laser/rays",
                          "z0",
                          H5T_NATIVE_DOUBLE,
                          dims,
                          state.laser_ray_z0.data(),
                          "cm",
                          cfg);
    write_numeric_dataset(file,
                          "laser/rays",
                          "vx0",
                          H5T_NATIVE_DOUBLE,
                          dims,
                          state.laser_ray_vx0.data(),
                          "dimensionless",
                          cfg);
    write_numeric_dataset(file,
                          "laser/rays",
                          "vy0",
                          H5T_NATIVE_DOUBLE,
                          dims,
                          state.laser_ray_vy0.data(),
                          "dimensionless",
                          cfg);
    write_numeric_dataset(file,
                          "laser/rays",
                          "vz0",
                          H5T_NATIVE_DOUBLE,
                          dims,
                          state.laser_ray_vz0.data(),
                          "dimensionless",
                          cfg);
  } else {
    if (state.laser_ray_R0.size() != n || state.laser_ray_Z0.size() != n ||
        state.laser_ray_vR0.size() != n || state.laser_ray_vZ0.size() != n) {
      return;
    }
    write_numeric_dataset(file,
                          "laser/rays",
                          "R0",
                          H5T_NATIVE_DOUBLE,
                          dims,
                          state.laser_ray_R0.data(),
                          "cm",
                          cfg);
    write_numeric_dataset(file,
                          "laser/rays",
                          "Z0",
                          H5T_NATIVE_DOUBLE,
                          dims,
                          state.laser_ray_Z0.data(),
                          "cm",
                          cfg);
    write_numeric_dataset(file,
                          "laser/rays",
                          "vR0",
                          H5T_NATIVE_DOUBLE,
                          dims,
                          state.laser_ray_vR0.data(),
                          "dimensionless",
                          cfg);
    write_numeric_dataset(file,
                          "laser/rays",
                          "vZ0",
                          H5T_NATIVE_DOUBLE,
                          dims,
                          state.laser_ray_vZ0.data(),
                          "dimensionless",
                          cfg);
  }
  write_numeric_dataset(file,
                        "laser/rays",
                        "power0",
                        H5T_NATIVE_DOUBLE,
                        dims,
                        state.laser_ray_power0.data(),
                        "erg/s",
                        cfg);
  write_numeric_dataset(
      file, "laser/rays", "beam_id", H5T_NATIVE_INT32, dims, beam_id.data(), "id", cfg);
  write_laser_mesh_group(file, state, cfg);
  write_ray_trajectory_group(file, state, cfg);
}

void write_ale_lambda_sweep_diagnostics_group(const hid_t file,
                                              const core::State& state,
                                              const core::Config& cfg) {
  const std::size_t n = state.ale_lambda_sweep_lambda.size();
  if (n == 0) {
    return;
  }
  if (state.ale_lambda_sweep_min_gauss_j.size() != n ||
      state.ale_lambda_sweep_min_corner_j.size() != n ||
      state.ale_lambda_sweep_min_v_rz.size() != n ||
      state.ale_lambda_sweep_admissible.size() != n) {
    core::log_warning(
        "HDF5Writer: skipping ALE lambda sweep diagnostics due to inconsistent vector sizes");
    return;
  }

  constexpr const char* group_path = "diagnostics/ale_lambda_sweep/v1";
  const hid_t group = ensure_group(file, group_path);
  TENRYU_ASSERT(group >= 0, "HDF5 failed to open ALE lambda sweep diagnostics group");
  write_scalar_attribute_i32(group, "schema_version", 1);
  write_scalar_attribute_i32(
      group, "target_cell_c", state.ale_lambda_sweep_target_cell_c);
  write_scalar_attribute_i32(
      group, "target_cell_i", state.ale_lambda_sweep_target_cell_i);
  write_scalar_attribute_i32(
      group, "target_cell_j", state.ale_lambda_sweep_target_cell_j);
  write_string_attribute(group,
                         "classification",
                         state.ale_lambda_sweep_classification.empty()
                             ? "no_admissible"
                             : state.ale_lambda_sweep_classification);
  warn_h5_close_failure(H5Gclose(group), "H5Gclose",
                        "HDF5Writer::write_ale_lambda_sweep_diagnostics_group(group)");

  const std::vector<hsize_t> dims = {static_cast<hsize_t>(n)};
  write_numeric_dataset(file,
                        group_path,
                        "lambda",
                        H5T_NATIVE_DOUBLE,
                        dims,
                        state.ale_lambda_sweep_lambda.data(),
                        "dimensionless",
                        cfg);
  write_numeric_dataset(file,
                        group_path,
                        "min_gauss_J",
                        H5T_NATIVE_DOUBLE,
                        dims,
                        state.ale_lambda_sweep_min_gauss_j.data(),
                        "cm2",
                        cfg);
  write_numeric_dataset(file,
                        group_path,
                        "min_corner_J",
                        H5T_NATIVE_DOUBLE,
                        dims,
                        state.ale_lambda_sweep_min_corner_j.data(),
                        "cm2",
                        cfg);
  write_numeric_dataset(file,
                        group_path,
                        "min_V_RZ",
                        H5T_NATIVE_DOUBLE,
                        dims,
                        state.ale_lambda_sweep_min_v_rz.data(),
                        "cm3",
                        cfg);
  write_numeric_dataset(file,
                        group_path,
                        "admissible",
                        H5T_NATIVE_UINT8,
                        dims,
                        state.ale_lambda_sweep_admissible.data(),
                        "bool",
                        cfg);
}

void write_areal_density_diagnostics_group(const hid_t file,
                                           const core::State& state,
                                           const core::Config& cfg) {
  if (!cfg.diagnostics.areal_density.enabled) {
    return;
  }
  const diagnostics::ArealDensityDiagnostics areal =
      diagnostics::compute_areal_density(state, cfg);
  const std::size_t n =
      std::min(areal.angles_deg.size(), areal.rhoR.size());
  if (n == 0) {
    return;
  }
  const std::string group = "diagnostics/areal_density/v1";
  const std::vector<hsize_t> dims = {static_cast<hsize_t>(n)};
  std::vector<double> angles(areal.angles_deg.begin(),
                             areal.angles_deg.begin() + n);
  std::vector<double> rhoR(areal.rhoR.begin(), areal.rhoR.begin() + n);
  write_numeric_dataset(
      file, group, "angles_deg", H5T_NATIVE_DOUBLE, dims, angles.data(), "deg", cfg);
  write_numeric_dataset(
      file, group, "rhoR", H5T_NATIVE_DOUBLE, dims, rhoR.data(), "g/cm2", cfg);

  const std::size_t n_hotspot =
      std::min(areal.angles_deg.size(), areal.rhoR_hotspot_tracer.size());
  if (n_hotspot > 0) {
    const std::vector<hsize_t> hotspot_dims = {static_cast<hsize_t>(n_hotspot)};
    std::vector<double> rhoR_hotspot(
        areal.rhoR_hotspot_tracer.begin(),
        areal.rhoR_hotspot_tracer.begin() + n_hotspot);
    write_numeric_dataset(file,
                          group,
                          "rhoR_hotspot_tracer",
                          H5T_NATIVE_DOUBLE,
                          hotspot_dims,
                          rhoR_hotspot.data(),
                          "g/cm2",
                          cfg);
  }

  const std::size_t n_fuel =
      std::min(areal.angles_deg.size(), areal.rhoR_fuel_tracer.size());
  if (n_fuel > 0) {
    const std::vector<hsize_t> fuel_dims = {static_cast<hsize_t>(n_fuel)};
    std::vector<double> rhoR_fuel(areal.rhoR_fuel_tracer.begin(),
                                  areal.rhoR_fuel_tracer.begin() + n_fuel);
    write_numeric_dataset(file,
                          group,
                          "rhoR_fuel_tracer",
                          H5T_NATIVE_DOUBLE,
                          fuel_dims,
                          rhoR_fuel.data(),
                          "g/cm2",
                          cfg);
  }
}

void write_hotspot_gas_diagnostics_group(const hid_t file,
                                         const core::State& state,
                                         const core::Config& cfg) {
  if (!cfg.numerics.diagnostics.hotspot_gas.enabled) {
    return;
  }
  const diagnostics::HotspotGasDiagnostics hot =
      diagnostics::compute_hotspot_gas_diagnostics(state, cfg);
  if (!hot.valid) {
    return;
  }
  const std::string group = "diagnostics/hotspot_gas/v1";
  write_numeric_dataset(
      file, group, "hotspot_Te_mean_eV", H5T_NATIVE_DOUBLE, {}, &hot.Te_mean_eV, "eV", cfg);
  write_numeric_dataset(
      file, group, "hotspot_Te_p10_eV", H5T_NATIVE_DOUBLE, {}, &hot.Te_p10_eV, "eV", cfg);
  write_numeric_dataset(
      file, group, "hotspot_Te_p50_eV", H5T_NATIVE_DOUBLE, {}, &hot.Te_p50_eV, "eV", cfg);
  write_numeric_dataset(
      file, group, "hotspot_Te_p90_eV", H5T_NATIVE_DOUBLE, {}, &hot.Te_p90_eV, "eV", cfg);
  const std::int32_t Ti_valid = hot.Ti_valid ? 1 : 0;
  write_numeric_dataset(file,
                        group,
                        "hotspot_Ti_valid",
                        H5T_NATIVE_INT32,
                        {},
                        &Ti_valid,
                        "dimensionless",
                        cfg);
  if (hot.Ti_valid) {
    write_numeric_dataset(
        file, group, "hotspot_Ti_mean_eV", H5T_NATIVE_DOUBLE, {}, &hot.Ti_mean_eV, "eV", cfg);
    write_numeric_dataset(
        file, group, "hotspot_Ti_p10_eV", H5T_NATIVE_DOUBLE, {}, &hot.Ti_p10_eV, "eV", cfg);
    write_numeric_dataset(
        file, group, "hotspot_Ti_p50_eV", H5T_NATIVE_DOUBLE, {}, &hot.Ti_p50_eV, "eV", cfg);
    write_numeric_dataset(
        file, group, "hotspot_Ti_p90_eV", H5T_NATIVE_DOUBLE, {}, &hot.Ti_p90_eV, "eV", cfg);
  }
  write_numeric_dataset(file,
                        group,
                        "hotspot_energy_internal_erg",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &hot.hotspot_internal_energy_erg,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        group,
                        "hotspot_energy_kinetic_erg",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &hot.hotspot_kinetic_energy_erg,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        group,
                        "hotspot_energy_total_erg",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &hot.hotspot_total_energy_erg,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        group,
                        "hotspot_energy_internal_initial_erg",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &hot.hotspot_internal_energy_initial_erg,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        group,
                        "hotspot_energy_kinetic_initial_erg",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &hot.hotspot_kinetic_energy_initial_erg,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        group,
                        "hotspot_energy_total_initial_erg",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &hot.hotspot_total_energy_initial_erg,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        group,
                        "hotspot_work_proxy_internal_erg",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &hot.hotspot_work_proxy_internal_erg,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        group,
                        "hotspot_work_proxy_kinetic_erg",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &hot.hotspot_work_proxy_kinetic_erg,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        group,
                        "hotspot_work_proxy_total_erg",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &hot.hotspot_work_proxy_total_erg,
                        "erg",
                        cfg);
  write_string_dataset(
      file,
      group,
      "hotspot_work_definition",
      "proxy_total_energy_change=sum(Y_g*m*(ee+ei+0.5*|u_cell|^2))-initial; "
      "not a tracer-boundary pressure-work integral");
}

void write_common_snapshot_content(const hid_t file,
                                   const core::State& state,
                                   const core::Config& cfg,
                                   const int step,
                                   const double t,
                                   const std::string& output_dir,
                                   const std::string& case_name) {
  write_root_attributes(file, state, cfg, step, t, state.dt);
  write_metadata_group(file, state, cfg, output_dir, case_name);
  write_mesh_group(file, state, cfg);
  write_hydro_group(file, state, cfg);
  write_button_hydro_flags_group(file, state, cfg);
  write_per_material_conservation_diagnostics_group(file, state, cfg);
  write_per_material_event_counters_group(file, state, cfg);
  write_dispatch_counters_group(file, state, cfg);
  write_ale_lambda_sweep_diagnostics_group(file, state, cfg);
  write_areal_density_diagnostics_group(file, state, cfg);
  write_hotspot_gas_diagnostics_group(file, state, cfg);
  write_radiation_group(file, state, cfg);
  write_laser_group(file, state, cfg);
  // time_state group (step, t, dt) for post-processing convenience
  const std::int32_t step_i32 = static_cast<std::int32_t>(step);
  write_numeric_dataset(file, "time_state", "step", H5T_NATIVE_INT32, {}, &step_i32, "count", cfg);
  write_numeric_dataset(file, "time_state", "t", H5T_NATIVE_DOUBLE, {}, &t, "s", cfg);
  write_numeric_dataset(file, "time_state", "dt", H5T_NATIVE_DOUBLE, {}, &state.dt, "s", cfg);
  write_numeric_dataset(file, "time_state", "dt_growth_ref", H5T_NATIVE_DOUBLE,
                        {}, &state.dt_growth_ref, "s", cfg);
  if (cfg.numerics.hydro.adaptive_av.enabled) {
    write_numeric_dataset(file,
                          "time_state",
                          "adaptive_av_r0",
                          H5T_NATIVE_DOUBLE,
                          {},
                          &state.adaptive_av_r0,
                          "cm",
                          cfg);
    write_numeric_dataset(file,
                          "time_state",
                          "adaptive_av_last_rs",
                          H5T_NATIVE_DOUBLE,
                          {},
                          &state.adaptive_av_last_rs,
                          "cm",
                          cfg);
    write_numeric_dataset(file,
                          "time_state",
                          "adaptive_av_last_us",
                          H5T_NATIVE_DOUBLE,
                          {},
                          &state.adaptive_av_last_us,
                          "cm/s",
                          cfg);
    write_numeric_dataset(file,
                          "time_state",
                          "adaptive_av_rs_min",
                          H5T_NATIVE_DOUBLE,
                          {},
                          &state.adaptive_av_rs_min,
                          "cm",
                          cfg);
    const std::int32_t adaptive_av_tracker_valid =
        state.adaptive_av_tracker_valid ? 1 : 0;
    const std::int32_t adaptive_av_bounce_seen =
        state.adaptive_av_bounce_seen ? 1 : 0;
    const std::int32_t adaptive_av_tracker_steps =
        static_cast<std::int32_t>(state.adaptive_av_tracker_steps);
    const std::int32_t adaptive_av_mode =
        static_cast<std::int32_t>(state.adaptive_av_mode);
    write_numeric_dataset(file,
                          "time_state",
                          "adaptive_av_tracker_steps",
                          H5T_NATIVE_INT32,
                          {},
                          &adaptive_av_tracker_steps,
                          "count",
                          cfg);
    write_numeric_dataset(file,
                          "time_state",
                          "adaptive_av_mode",
                          H5T_NATIVE_INT32,
                          {},
                          &adaptive_av_mode,
                          "1",
                          cfg);
    write_numeric_dataset(file,
                          "time_state",
                          "adaptive_av_tracker_valid",
                          H5T_NATIVE_INT32,
                          {},
                          &adaptive_av_tracker_valid,
                          "1",
                          cfg);
    write_numeric_dataset(file,
                          "time_state",
                          "adaptive_av_bounce_seen",
                          H5T_NATIVE_INT32,
                          {},
                          &adaptive_av_bounce_seen,
                          "1",
                          cfg);
  }
  auto energy_budget = (state.mesh.dim == 2) ? diagnostics::compute_energy_budget_2d(state)
                                             : diagnostics::compute_energy_budget_1d(state);
  write_numeric_dataset(file,
                        "time_state",
                        "E_total",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &energy_budget.E_total,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "time_state",
                        "E_internal",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &energy_budget.E_internal,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "time_state",
                        "E_kinetic",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &energy_budget.E_kinetic,
                        "erg",
                        cfg);
}

void write_checkpoint_extras(const hid_t file,
                             const core::State& state,
                             const core::Config& cfg,
                             const radiation::PhotonPool& pool,
                             const int step,
                             const double t) {
  write_ale_reference_checkpoint_group(file, state, cfg);
  write_carrier_checkpoint_group(file, state, cfg);
  write_evacuated_cells_checkpoint_group(file, state, cfg);

  std::vector<std::int8_t> hydro_active = state.hydro_active;
  if (!has_button_center(state)) {
    if (hydro_active.size() != state.rho.size()) {
      hydro_active.assign(state.rho.size(), static_cast<std::int8_t>(1));
    }
    write_numeric_dataset(file,
                          "hydro_flags",
                          "hydro_active",
                          H5T_NATIVE_INT8,
                          {static_cast<hsize_t>(hydro_active.size())},
                          hydro_active.data(),
                          "flag",
                          cfg);
  }

  const std::size_t n_p = static_cast<std::size_t>(std::max(pool.n_alive, 0));
  const std::int64_t n_particles_i64 = static_cast<std::int64_t>(n_p);
  const hid_t particles_group = ensure_group(file, "particles");
  TENRYU_ASSERT(particles_group >= 0, "HDF5 failed to open particles group");
  write_scalar_attribute_i64(particles_group, "n_particles", n_particles_i64);
  warn_h5_close_failure(H5Gclose(particles_group), "H5Gclose",
                        "HDF5Writer::write_checkpoint_extras(particles_group)");

  const std::int64_t capacity_i64 = static_cast<std::int64_t>(std::max(pool.capacity, 0));
  write_numeric_dataset(file,
                        "particles",
                        "pool_capacity",
                        H5T_NATIVE_INT64,
                        {},
                        &capacity_i64,
                        "count",
                        cfg);

  if (n_p > 0) {
    // Known overhead: checkpoint write materializes many full-size host mirrors
    // of PhotonPool SoA fields. This favors simple schema writes today; a future
    // streaming/chunked path can reduce peak transient host memory.
    const auto pos_r = copy_device_array(pool.pos_r, n_p, "PhotonPool.pos_r");
    const auto pos_z = copy_device_array(pool.pos_z, n_p, "PhotonPool.pos_z");
    const auto dir_r = copy_device_array(pool.dir_r, n_p, "PhotonPool.dir_r");
    const auto dir_z = copy_device_array(pool.dir_z, n_p, "PhotonPool.dir_z");
    const auto dir_phi = copy_device_array(pool.dir_phi, n_p, "PhotonPool.dir_phi");
    const auto energy = copy_device_array(pool.energy, n_p, "PhotonPool.energy");
    const auto birth_energy =
        copy_device_array(pool.birth_energy, n_p, "PhotonPool.birth_energy");
    const auto sign = copy_device_array(pool.sign, n_p, "PhotonPool.sign");
    const auto group_id = copy_device_array(pool.group_id, n_p, "PhotonPool.group_id");
    const auto cell_id = copy_device_array(pool.cell_id, n_p, "PhotonPool.cell_id");
    const auto mode = copy_device_array(pool.mode, n_p, "PhotonPool.mode");
    const auto alive = copy_device_array(pool.alive, n_p, "PhotonPool.alive");
    const auto global_id = copy_device_array(pool.global_id, n_p, "PhotonPool.global_id");
    const auto weight = copy_device_array(pool.weight, n_p, "PhotonPool.weight");
    const auto time_remain =
        copy_device_array(pool.time_remain, n_p, "PhotonPool.time_remain");
    const auto rng_counter =
        copy_device_array(pool.rng_counter, n_p, "PhotonPool.rng_counter");

    const std::vector<hsize_t> pdims = {static_cast<hsize_t>(n_p)};

    write_numeric_dataset(file, "particles", "pos_r", H5T_NATIVE_DOUBLE, pdims, pos_r.data(), "cm", cfg);
    write_numeric_dataset(file, "particles", "pos_z", H5T_NATIVE_DOUBLE, pdims, pos_z.data(), "cm", cfg);
    write_numeric_dataset(file, "particles", "dir_r", H5T_NATIVE_DOUBLE, pdims, dir_r.data(), "dimensionless", cfg);
    write_numeric_dataset(file, "particles", "dir_z", H5T_NATIVE_DOUBLE, pdims, dir_z.data(), "dimensionless", cfg);
    write_numeric_dataset(file,
                          "particles",
                          "dir_phi",
                          H5T_NATIVE_DOUBLE,
                          pdims,
                          dir_phi.data(),
                          "dimensionless",
                          cfg);
    write_numeric_dataset(file,
                          "particles",
                          "energy",
                          H5T_NATIVE_DOUBLE,
                          pdims,
                          energy.data(),
                          "erg",
                          cfg);
    write_numeric_dataset(file,
                          "particles",
                          "birth_energy",
                          H5T_NATIVE_DOUBLE,
                          pdims,
                          birth_energy.data(),
                          "erg",
                          cfg);
    write_numeric_dataset(file,
                          "particles",
                          "sign",
                          H5T_NATIVE_INT8,
                          pdims,
                          sign.data(),
                          "dimensionless",
                          cfg);
    write_numeric_dataset(file,
                          "particles",
                          "group_id",
                          H5T_NATIVE_UINT16,
                          pdims,
                          group_id.data(),
                          "index",
                          cfg);
    write_numeric_dataset(file,
                          "particles",
                          "cell_id",
                          H5T_NATIVE_INT32,
                          pdims,
                          cell_id.data(),
                          "index",
                          cfg);
    write_numeric_dataset(file,
                          "particles",
                          "mode",
                          H5T_NATIVE_UINT8,
                          pdims,
                          mode.data(),
                          "flag",
                          cfg);
    write_numeric_dataset(file,
                          "particles",
                          "alive",
                          H5T_NATIVE_UINT8,
                          pdims,
                          alive.data(),
                          "flag",
                          cfg);
    write_numeric_dataset(file,
                          "particles",
                          "global_id",
                          H5T_NATIVE_UINT64,
                          pdims,
                          global_id.data(),
                          "id",
                          cfg);
    write_numeric_dataset(file,
                          "particles",
                          "weight",
                          H5T_NATIVE_DOUBLE,
                          pdims,
                          weight.data(),
                          "dimensionless",
                          cfg);
    write_numeric_dataset(file,
                          "particles",
                          "time_remain",
                          H5T_NATIVE_DOUBLE,
                          pdims,
                          time_remain.data(),
                          "s",
                          cfg);

    write_numeric_dataset(file,
                          "rng",
                          "rng_counter",
                          H5T_NATIVE_UINT32,
                          pdims,
                          rng_counter.data(),
                          "count",
                          cfg);
    write_numeric_dataset(file,
                          "rng",
                          "global_id",
                          H5T_NATIVE_UINT64,
                          pdims,
                          global_id.data(),
                          "id",
                          cfg);
  }

  write_numeric_dataset(file,
                        "output_state",
                        "t_next_plot",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.t_next_plot,
                        "s",
                        cfg);
  write_numeric_dataset(file,
                        "output_state",
                        "t_next_history",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.t_next_history,
                        "s",
                        cfg);
  write_numeric_dataset(file,
                        "output_state",
                        "t_next_checkpoint",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.t_next_checkpoint,
                        "s",
                        cfg);

  // time_state/step, t, dt are now written by write_common_snapshot_content
  (void)step;
  (void)t;
  const std::int32_t ale_last_applied_step =
      static_cast<std::int32_t>(state.ale_last_applied_step);
  write_numeric_dataset(file,
                        "time_state",
                        "ale_last_applied_step",
                        H5T_NATIVE_INT32,
                        {},
                        &ale_last_applied_step,
                        "count",
                        cfg);
  write_numeric_dataset(file,
                        "time_state",
                        "axis_margin_initial",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.axis_margin_initial,
                        "cm^2",
                        cfg);
  if (!state.axis_mass_initial.empty()) {
    const std::vector<hsize_t> dims = {static_cast<hsize_t>(state.axis_mass_initial.size())};
    write_numeric_dataset(file,
                          "time_state",
                          "axis_mass_initial",
                          H5T_NATIVE_DOUBLE,
                          dims,
                          state.axis_mass_initial.data(),
                          "g",
                          cfg);
  }
  if (!state.axis_inflow_budget.empty()) {
    const std::vector<hsize_t> dims = {static_cast<hsize_t>(state.axis_inflow_budget.size())};
    write_numeric_dataset(file,
                          "time_state",
                          "axis_inflow_budget",
                          H5T_NATIVE_DOUBLE,
                          dims,
                          state.axis_inflow_budget.data(),
                          "g",
                          cfg);
  }
  write_numeric_dataset(file,
                        "time_state",
                        "E_safety",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.E_safety,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "time_state",
                        "E_numerical_loss",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.E_numerical_loss,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "time_state",
                        "E_laser_deposited",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.E_laser_deposited,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "time_state",
                        "E_cbet_iaw_step",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.E_cbet_iaw_step,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "time_state",
                        "E_cbet_iaw",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.E_cbet_iaw,
                        "erg",
                        cfg);
  if (state.burn_enabled_any) {
    write_numeric_dataset(file,
                          "time_state",
                          "E_burn_released",
                          H5T_NATIVE_DOUBLE,
                          {},
                          &state.E_burn_released,
                          "erg",
                          cfg);
    write_numeric_dataset(file,
                          "time_state",
                          "E_burn_dep_e",
                          H5T_NATIVE_DOUBLE,
                          {},
                          &state.E_burn_dep_e,
                          "erg",
                          cfg);
    write_numeric_dataset(file,
                          "time_state",
                          "E_burn_dep_i",
                          H5T_NATIVE_DOUBLE,
                          {},
                          &state.E_burn_dep_i,
                          "erg",
                          cfg);
    write_numeric_dataset(file,
                          "time_state",
                          "E_burn_esc_charged",
                          H5T_NATIVE_DOUBLE,
                          {},
                          &state.E_burn_esc_charged,
                          "erg",
                          cfg);
    write_numeric_dataset(file,
                          "time_state",
                          "E_burn_esc_neutron",
                          H5T_NATIVE_DOUBLE,
                          {},
                          &state.E_burn_esc_neutron,
                          "erg",
                          cfg);
    write_numeric_dataset(file,
                          "time_state",
                          "N_burn_neutrons_dt",
                          H5T_NATIVE_DOUBLE,
                          {},
                          &state.N_burn_neutrons_dt,
                          "1",
                          cfg);
    write_numeric_dataset(file,
                          "time_state",
                          "N_burn_neutrons_dd",
                          H5T_NATIVE_DOUBLE,
                          {},
                          &state.N_burn_neutrons_dd,
                          "1",
                          cfg);
  }
  if (state.burn_diffusion_any || state.burn_mc_any) {
    write_numeric_dataset(file,
                          "time_state",
                          "E_burn_inflight",
                          H5T_NATIVE_DOUBLE,
                          {},
                          &state.E_burn_inflight,
                          "erg",
                          cfg);
  }
  if (state.burn_mc_any) {
    const std::int32_t burn_mc_live =
        static_cast<std::int32_t>(state.burn_mc_live);
    write_numeric_dataset(file,
                          "time_state",
                          "burn_mc_live",
                          H5T_NATIVE_INT32,
                          {},
                          &burn_mc_live,
                          "count",
                          cfg);
  }
  write_numeric_dataset(file,
                        "time_state",
                        "E_laser_escaped",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.E_laser_escaped,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "time_state",
                        "E_laser_incident",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.E_laser_incident,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "time_state",
                        "E_ra_deposited",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.E_ra_deposited,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "time_state",
                        "E_rad_escaped",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.E_rad_escaped,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "time_state",
                        "E_floor_injected",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.E_floor_injected,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "time_state",
                        "E_pdV_bdry",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.E_pdV_bdry,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "time_state",
                        "E_Marshak_in",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.E_Marshak_in,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "time_state",
                        "E_solver",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.E_solver,
                        "erg",
                        cfg);
  write_numeric_dataset(file,
                        "time_state",
                        "sn_ap_alpha_max",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.sn_ap_alpha_max,
                        "dimensionless",
                        cfg);
  write_numeric_dataset(file,
                        "time_state",
                        "sn_ap_alpha_active_faces",
                        H5T_NATIVE_DOUBLE,
                        {},
                        &state.sn_ap_alpha_active_faces,
                        "count",
                        cfg);
  write_numeric_dataset(file,
                        "time_state",
                        "user_seed",
                        H5T_NATIVE_UINT64,
                        {},
                        &cfg.main.seed,
                        "seed",
                        cfg);
}

#endif  // TENRYU_ENABLE_HDF5

}  // namespace

MaterialInterfaceReadStatus read_material_interface_status(
    const std::string& h5_file_path) {
#if TENRYU_ENABLE_HDF5
  const hid_t file = H5Fopen(h5_file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    return MaterialInterfaceReadStatus::MissingGroupPlicDisabled;
  }
  if (tenryu::io::h5_link_exists(
          file, "/diagnostics/material_interface/v1", H5P_DEFAULT) > 0) {
    warn_h5_close_failure(H5Fclose(file), "H5Fclose",
                          "HDF5Writer::read_material_interface_status(file)");
    return MaterialInterfaceReadStatus::Present;
  }

  MaterialInterfaceReadStatus status =
      MaterialInterfaceReadStatus::MissingGroupPlicDisabled;
  if (tenryu::io::h5_link_exists(
          file, "/diagnostics/ale_provenance/v1", H5P_DEFAULT) > 0) {
    const hid_t group =
        H5Gopen2(file, "/diagnostics/ale_provenance/v1", H5P_DEFAULT);
    TENRYU_ASSERT(group >= 0,
                  "HDF5 H5Gopen2(/diagnostics/ale_provenance/v1) failed");
    const std::string final_claim =
        read_string_attribute_if_exists(group, "final_claim_level")
            .value_or(std::string());
    if (final_claim == "production_comparable") {
      status =
          MaterialInterfaceReadStatus::
              InconsistentMissingGroupForProductionComparable;
    }
    warn_h5_close_failure(H5Gclose(group), "H5Gclose",
                          "HDF5Writer::read_material_interface_status(group)");
  }
  warn_h5_close_failure(H5Fclose(file), "H5Fclose",
                        "HDF5Writer::read_material_interface_status(file)");
  return status;
#else
  (void)h5_file_path;
  return MaterialInterfaceReadStatus::MissingGroupPlicDisabled;
#endif
}

std::optional<MaterialInterfaceAttributes> read_material_interface_attributes(
    const std::string& h5_file_path) {
#if TENRYU_ENABLE_HDF5
  const hid_t file = H5Fopen(h5_file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    return std::nullopt;
  }
  if (tenryu::io::h5_link_exists(
          file, "/diagnostics/material_interface/v1", H5P_DEFAULT) <= 0) {
    warn_h5_close_failure(H5Fclose(file), "H5Fclose",
                          "HDF5Writer::read_material_interface_attributes(file)");
    return std::nullopt;
  }
  const hid_t group =
      H5Gopen2(file, "/diagnostics/material_interface/v1", H5P_DEFAULT);
  TENRYU_ASSERT(group >= 0,
                "HDF5 H5Gopen2(/diagnostics/material_interface/v1) failed");
  MaterialInterfaceAttributes attrs;
  attrs.plic_enabled =
      read_u8_attribute_if_exists(group, "plic_enabled").value_or(0U) == 1U;
  attrs.plic_schema_version =
      read_i32_attribute_if_exists(group, "plic_schema_version").value_or(0);
  attrs.plic_reconstruction_engine_version =
      read_string_attribute_if_exists(group, "plic_reconstruction_engine_version")
          .value_or(std::string());
  attrs.plic_normal_estimator =
      read_string_attribute_if_exists(group, "plic_normal_estimator")
          .value_or(std::string());
  attrs.t0_volume_cut_method =
      read_string_attribute_if_exists(group, "t0_volume_cut_method")
          .value_or(std::string());
  attrs.plic_reconstruction_method =
      read_string_attribute_if_exists(group, "plic_reconstruction_method")
          .value_or(std::string());
  warn_h5_close_failure(H5Gclose(group), "H5Gclose",
                        "HDF5Writer::read_material_interface_attributes(group)");
  warn_h5_close_failure(H5Fclose(file), "H5Fclose",
                        "HDF5Writer::read_material_interface_attributes(file)");
  return attrs;
#else
  (void)h5_file_path;
  return std::nullopt;
#endif
}

PerMaterialConservationResiduals compute_per_material_conservation_residuals(
    const core::State& state,
    const core::Config& cfg) {
  PerMaterialConservationResiduals out{};
  if (!cfg.numerics.materials.per_material_conservation_enabled) {
    return out;
  }

  const std::size_t n_cells = state.rho.size();
  const std::size_t n_mat = cfg.materials.materials.size();
  const std::size_t n_cell_mat = n_cells * n_mat;
  if (n_cells == 0 || n_mat == 0 ||
      state.mass_per_material.size() != n_cell_mat ||
      state.Ee_per_material.size() != n_cell_mat ||
      state.Ei_per_material.size() != n_cell_mat ||
      state.mass.size() != n_cells || state.ee.size() != n_cells ||
      state.ei.size() != n_cells) {
    return out;
  }

  const auto mass = copy_field_to_host(state.mass);
  const auto ee = copy_field_to_host(state.ee);
  const auto ei = copy_field_to_host(state.ei);
  const auto mass_m = copy_field_to_host(state.mass_per_material);
  const auto Ee_m = copy_field_to_host(state.Ee_per_material);
  const auto Ei_m = copy_field_to_host(state.Ei_per_material);

  auto update = [](double& max_abs, double& max_rel, const double actual,
                   const double expected) {
    const double abs_res = std::abs(actual - expected);
    const double denom = (std::abs(expected) > 0.0) ? std::abs(expected) : 1.0;
    const double rel_res = abs_res / denom;
    if (std::isfinite(abs_res)) {
      max_abs = std::max(max_abs, abs_res);
    }
    if (std::isfinite(rel_res)) {
      max_rel = std::max(max_rel, rel_res);
    }
  };

  for (std::size_t c = 0; c < n_cells; ++c) {
    double mass_sum = 0.0;
    double Ee_sum = 0.0;
    double Ei_sum = 0.0;
    for (std::size_t m = 0; m < n_mat; ++m) {
      const std::size_t idx = c * n_mat + m;
      mass_sum += mass_m[idx];
      Ee_sum += Ee_m[idx];
      Ei_sum += Ei_m[idx];
    }
    update(out.mass_max_abs_residual, out.mass_max_rel_residual, mass_sum, mass[c]);
    update(out.Ee_max_abs_residual,
           out.Ee_max_rel_residual,
           Ee_sum,
           mass[c] * ee[c]);
    update(out.Ei_max_abs_residual,
           out.Ei_max_rel_residual,
           Ei_sum,
           mass[c] * ei[c]);
  }
  return out;
}

std::uint64_t dispatch_counters_regression_hash(const core::State& state,
                                                const core::Config& cfg) {
  std::uint64_t values[5] = {};
  if (cfg.numerics.materials.per_material_conservation_enabled) {
    values[0] =
        state.dispatch_counters.per_material_kernel_call_count.load(std::memory_order_relaxed);
    values[1] =
        state.dispatch_counters.eos_inverse_call_count.load(std::memory_order_relaxed);
    values[2] =
        state.dispatch_counters.mixture_projection_call_count.load(std::memory_order_relaxed);
    values[3] =
        state.dispatch_counters.lazy_cache_te_m_hit_count.load(std::memory_order_relaxed);
    values[4] =
        state.dispatch_counters.lazy_cache_te_m_miss_count.load(std::memory_order_relaxed);
  }

  std::uint64_t hash = 14695981039346656037ULL;
  constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
  for (const std::uint64_t value : values) {
    for (int byte = 0; byte < 8; ++byte) {
      const auto b = static_cast<std::uint8_t>((value >> (8 * byte)) & 0xffU);
      hash ^= static_cast<std::uint64_t>(b);
      hash *= kFnvPrime;
    }
  }
  return hash;
}

void HDF5Writer::write_snapshot(const core::State& state,
                                const core::Config& cfg,
                                const int file_index,
                                const int step,
                                const double t,
                                const std::string& output_dir,
                                const std::string& case_name,
                                const int rank) const {
#if TENRYU_ENABLE_HDF5
  if (rank != 0) {
    return;
  }

  std::filesystem::create_directories(output_dir);
  const std::filesystem::path path =
      std::filesystem::path(output_dir) /
      (safe_case_name(case_name) + "_" + format_step(file_index) + ".h5");
  const std::filesystem::path tmp_path = path.string() + ".tmp";

  const hid_t file =
      H5Fcreate(tmp_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(file >= 0, "Failed to create snapshot HDF5 file: " + tmp_path.string());

  write_common_snapshot_content(file, state, cfg, step, t, output_dir, case_name);
  const std::string snapshot_close_context =
      "HDF5Writer::write_snapshot(" + tmp_path.string() + ")";
  const herr_t snapshot_close_status = H5Fclose(file);
  if (snapshot_close_status < 0) {
    core::log_warning("[WARN] H5Fclose failed in " + snapshot_close_context +
                      "; continuing publish because data may already be durable.");
  }

  std::error_code rename_ec;
  std::filesystem::rename(tmp_path, path, rename_ec);
  TENRYU_ASSERT(!rename_ec,
                "Failed to atomically publish snapshot HDF5 file '" + path.string() +
                    "': " + rename_ec.message());
#else
  (void)state;
  (void)cfg;
  (void)file_index;
  (void)step;
  (void)t;
  (void)output_dir;
  (void)case_name;
  (void)rank;
#endif
}

std::string HDF5Writer::write_checkpoint(
    const core::State& state,
    const core::Config& cfg,
    const radiation::PhotonPool& photon_pool,
    const int file_index,
    const int step,
    const double t,
    const std::string& output_dir,
    const std::string& case_name) const {
#if TENRYU_ENABLE_HDF5
  std::filesystem::create_directories(output_dir);
  const std::filesystem::path path =
      std::filesystem::path(output_dir) /
      (safe_case_name(case_name) + "_ckpt_" + format_step(file_index) + ".h5");
  const std::filesystem::path tmp_path = path.string() + ".tmp";

  const hid_t file =
      H5Fcreate(tmp_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(file >= 0, "Failed to create checkpoint HDF5 file: " + tmp_path.string());

  write_common_snapshot_content(file, state, cfg, step, t, output_dir, case_name);
  write_checkpoint_extras(file, state, cfg, photon_pool, step, t);

  const std::string checkpoint_close_context =
      "HDF5Writer::write_checkpoint(" + tmp_path.string() + ")";
  const herr_t checkpoint_close_status = H5Fclose(file);
  if (checkpoint_close_status < 0) {
    core::log_warning("[WARN] H5Fclose failed in " + checkpoint_close_context +
                      "; continuing publish because data may already be durable.");
  }

  std::error_code rename_ec;
  std::filesystem::rename(tmp_path, path, rename_ec);
  TENRYU_ASSERT(!rename_ec,
                "Failed to atomically publish checkpoint HDF5 file '" + path.string() +
                    "': " + rename_ec.message());
  return path.string();
#else
  (void)state;
  (void)cfg;
  (void)photon_pool;
  (void)file_index;
  (void)step;
  (void)t;
  (void)output_dir;
  (void)case_name;
  return {};
#endif
}

}  // namespace tenryu::io
