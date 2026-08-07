#include "diagnostics/escape_valve_history.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <string>

#include "core/error.hpp"

#if TENRYU_ENABLE_HDF5
#include <hdf5.h>

#include "io/hdf5_utils.hpp"
#endif

namespace tenryu::diagnostics {

#if TENRYU_ENABLE_HDF5
namespace {

void warn_h5_close_failure(const herr_t status, const char* op, const char* context) {
  if (status < 0) {
    tenryu::core::log_warning(std::string("[WARN] ") + op + " failed in " +
                              context);
  }
}

void ensure_parent_groups(const hid_t file, const std::string& dataset_path) {
  std::size_t pos = 0;
  while (true) {
    pos = dataset_path.find('/', pos);
    if (pos == std::string::npos) {
      break;
    }
    const std::string group = dataset_path.substr(0, pos);
    if (!group.empty() &&
        tenryu::io::h5_link_exists(file, group.c_str(), H5P_DEFAULT) <= 0) {
      const hid_t gid =
          H5Gcreate2(file, group.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      TENRYU_ASSERT(gid >= 0,
                    "EscapeValveHistory H5Gcreate2 failed: " + group);
      warn_h5_close_failure(H5Gclose(gid),
                            "H5Gclose",
                            "EscapeValveHistory::ensure_parent_groups");
    }
    ++pos;
  }
}

hid_t open_or_create_group(const hid_t file, const std::string& path) {
  ensure_parent_groups(file, path + "/_");
  if (tenryu::io::h5_link_exists(file, path.c_str(), H5P_DEFAULT) > 0) {
    const hid_t gid = H5Gopen2(file, path.c_str(), H5P_DEFAULT);
    TENRYU_ASSERT(gid >= 0, "EscapeValveHistory H5Gopen2 failed: " + path);
    return gid;
  }
  const hid_t gid =
      H5Gcreate2(file, path.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(gid >= 0, "EscapeValveHistory H5Gcreate2 failed: " + path);
  return gid;
}

void write_string_attribute_if_missing(const hid_t object,
                                       const char* name,
                                       const char* value) {
  if (H5Aexists(object, name) > 0) {
    return;
  }
  const hid_t attr_space = H5Screate(H5S_SCALAR);
  TENRYU_ASSERT(attr_space >= 0,
                "EscapeValveHistory H5Screate(string attr) failed");
  const hid_t attr_type = H5Tcopy(H5T_C_S1);
  TENRYU_ASSERT(attr_type >= 0,
                "EscapeValveHistory H5Tcopy(string attr) failed");
  TENRYU_ASSERT(H5Tset_size(attr_type, std::char_traits<char>::length(value) + 1) >= 0,
                "EscapeValveHistory H5Tset_size(string attr) failed");
  TENRYU_ASSERT(H5Tset_strpad(attr_type, H5T_STR_NULLTERM) >= 0,
                "EscapeValveHistory H5Tset_strpad(string attr) failed");
  const hid_t attr =
      H5Acreate2(object, name, attr_type, attr_space, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0,
                "EscapeValveHistory H5Acreate2(string attr) failed");
  TENRYU_ASSERT(H5Awrite(attr, attr_type, value) >= 0,
                "EscapeValveHistory H5Awrite(string attr) failed");
  warn_h5_close_failure(H5Aclose(attr),
                        "H5Aclose",
                        "EscapeValveHistory::write_string_attribute_if_missing");
  warn_h5_close_failure(H5Tclose(attr_type),
                        "H5Tclose",
                        "EscapeValveHistory::write_string_attribute_if_missing");
  warn_h5_close_failure(H5Sclose(attr_space),
                        "H5Sclose",
                        "EscapeValveHistory::write_string_attribute_if_missing");
}

template <typename T>
void append_scalar_native(const hid_t file,
                          const std::string& path,
                          const T value,
                          const hid_t h5_type,
                          const char* label) {
  ensure_parent_groups(file, path);
  if (tenryu::io::h5_link_exists(file, path.c_str(), H5P_DEFAULT) > 0) {
    const hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    TENRYU_ASSERT(dset >= 0,
                  std::string("EscapeValveHistory H5Dopen2(") + label +
                      ") failed");
    const hid_t fspace = H5Dget_space(dset);
    TENRYU_ASSERT(fspace >= 0,
                  std::string("EscapeValveHistory H5Dget_space(") + label +
                      ") failed");
    hsize_t dims[1] = {0};
    const int rank = H5Sget_simple_extent_dims(fspace, dims, nullptr);
    TENRYU_ASSERT(rank == 1,
                  std::string("EscapeValveHistory rank mismatch for ") + label);
    warn_h5_close_failure(H5Sclose(fspace),
                          "H5Sclose",
                          "EscapeValveHistory::append_scalar_native(extent)");
    const hsize_t new_dims[1] = {dims[0] + 1};
    TENRYU_ASSERT(H5Dset_extent(dset, new_dims) >= 0,
                  std::string("EscapeValveHistory H5Dset_extent(") + label +
                      ") failed");
    const hid_t file_space = H5Dget_space(dset);
    TENRYU_ASSERT(file_space >= 0,
                  "EscapeValveHistory H5Dget_space(post-extend) failed");
    const hsize_t start[1] = {dims[0]};
    const hsize_t count[1] = {1};
    TENRYU_ASSERT(H5Sselect_hyperslab(file_space,
                                      H5S_SELECT_SET,
                                      start,
                                      nullptr,
                                      count,
                                      nullptr) >= 0,
                  "EscapeValveHistory H5Sselect_hyperslab failed");
    const hid_t mem_space = H5Screate_simple(1, count, nullptr);
    TENRYU_ASSERT(mem_space >= 0,
                  "EscapeValveHistory H5Screate_simple append failed");
    TENRYU_ASSERT(H5Dwrite(dset,
                           h5_type,
                           mem_space,
                           file_space,
                           H5P_DEFAULT,
                           &value) >= 0,
                  std::string("EscapeValveHistory H5Dwrite(") + label +
                      " append) failed");
    warn_h5_close_failure(H5Sclose(mem_space),
                          "H5Sclose",
                          "EscapeValveHistory::append_scalar_native(mem)");
    warn_h5_close_failure(H5Sclose(file_space),
                          "H5Sclose",
                          "EscapeValveHistory::append_scalar_native(file)");
    warn_h5_close_failure(H5Dclose(dset),
                          "H5Dclose",
                          "EscapeValveHistory::append_scalar_native(dset)");
    return;
  }

  const hsize_t dims[1] = {1};
  const hsize_t max_dims[1] = {H5S_UNLIMITED};
  const hsize_t chunk[1] = {256};
  const hid_t space = H5Screate_simple(1, dims, max_dims);
  TENRYU_ASSERT(space >= 0,
                "EscapeValveHistory H5Screate_simple create failed");
  const hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  TENRYU_ASSERT(dcpl >= 0, "EscapeValveHistory H5Pcreate failed");
  TENRYU_ASSERT(H5Pset_chunk(dcpl, 1, chunk) >= 0,
                "EscapeValveHistory H5Pset_chunk failed");
  const hid_t dset = H5Dcreate2(file,
                                path.c_str(),
                                h5_type,
                                space,
                                H5P_DEFAULT,
                                dcpl,
                                H5P_DEFAULT);
  TENRYU_ASSERT(dset >= 0,
                std::string("EscapeValveHistory H5Dcreate2(") + label +
                    ") failed");
  TENRYU_ASSERT(H5Dwrite(dset,
                         h5_type,
                         H5S_ALL,
                         H5S_ALL,
                         H5P_DEFAULT,
                         &value) >= 0,
                std::string("EscapeValveHistory H5Dwrite(") + label +
                    " create) failed");
  warn_h5_close_failure(H5Dclose(dset),
                        "H5Dclose",
                        "EscapeValveHistory::append_scalar_native(create dset)");
  warn_h5_close_failure(H5Pclose(dcpl),
                        "H5Pclose",
                        "EscapeValveHistory::append_scalar_native(dcpl)");
  warn_h5_close_failure(H5Sclose(space),
                        "H5Sclose",
                        "EscapeValveHistory::append_scalar_native(space)");
}

void append_scalar_i64(const hid_t file,
                       const std::string& path,
                       const std::int64_t value) {
  append_scalar_native(file, path, value, H5T_NATIVE_INT64, "i64");
}

void append_scalar_double(const hid_t file,
                          const std::string& path,
                          const double value) {
  append_scalar_native(file, path, value, H5T_NATIVE_DOUBLE, "double");
}

void append_scalar_fixed_string(const hid_t file,
                                const std::string& path,
                                const std::string& value) {
  constexpr std::size_t kStringBytes = 128;
  ensure_parent_groups(file, path);
  std::array<char, kStringBytes> buffer{};
  const std::size_t n_copy = std::min(value.size(), kStringBytes - 1);
  std::copy_n(value.data(), n_copy, buffer.data());

  const hid_t type = H5Tcopy(H5T_C_S1);
  TENRYU_ASSERT(type >= 0, "EscapeValveHistory H5Tcopy(fixed string) failed");
  TENRYU_ASSERT(H5Tset_size(type, kStringBytes) >= 0,
                "EscapeValveHistory H5Tset_size(fixed string) failed");
  TENRYU_ASSERT(H5Tset_strpad(type, H5T_STR_NULLTERM) >= 0,
                "EscapeValveHistory H5Tset_strpad(fixed string) failed");

  if (tenryu::io::h5_link_exists(file, path.c_str(), H5P_DEFAULT) > 0) {
    const hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    TENRYU_ASSERT(dset >= 0,
                  "EscapeValveHistory H5Dopen2(fixed string) failed");
    const hid_t fspace = H5Dget_space(dset);
    TENRYU_ASSERT(fspace >= 0,
                  "EscapeValveHistory H5Dget_space(fixed string) failed");
    hsize_t dims[1] = {0};
    const int rank = H5Sget_simple_extent_dims(fspace, dims, nullptr);
    TENRYU_ASSERT(rank == 1,
                  "EscapeValveHistory fixed string rank mismatch");
    warn_h5_close_failure(H5Sclose(fspace),
                          "H5Sclose",
                          "EscapeValveHistory::append_fixed_string(extent)");
    const hsize_t new_dims[1] = {dims[0] + 1};
    TENRYU_ASSERT(H5Dset_extent(dset, new_dims) >= 0,
                  "EscapeValveHistory H5Dset_extent(fixed string) failed");
    const hid_t file_space = H5Dget_space(dset);
    TENRYU_ASSERT(file_space >= 0,
                  "EscapeValveHistory H5Dget_space(fixed string append) failed");
    const hsize_t start[1] = {dims[0]};
    const hsize_t count[1] = {1};
    TENRYU_ASSERT(H5Sselect_hyperslab(file_space,
                                      H5S_SELECT_SET,
                                      start,
                                      nullptr,
                                      count,
                                      nullptr) >= 0,
                  "EscapeValveHistory H5Sselect_hyperslab(fixed string) failed");
    const hid_t mem_space = H5Screate_simple(1, count, nullptr);
    TENRYU_ASSERT(mem_space >= 0,
                  "EscapeValveHistory H5Screate_simple(fixed string) failed");
    TENRYU_ASSERT(H5Dwrite(dset,
                           type,
                           mem_space,
                           file_space,
                           H5P_DEFAULT,
                           buffer.data()) >= 0,
                  "EscapeValveHistory H5Dwrite(fixed string append) failed");
    warn_h5_close_failure(H5Sclose(mem_space),
                          "H5Sclose",
                          "EscapeValveHistory::append_fixed_string(mem)");
    warn_h5_close_failure(H5Sclose(file_space),
                          "H5Sclose",
                          "EscapeValveHistory::append_fixed_string(file)");
    warn_h5_close_failure(H5Dclose(dset),
                          "H5Dclose",
                          "EscapeValveHistory::append_fixed_string(dset)");
    warn_h5_close_failure(H5Tclose(type),
                          "H5Tclose",
                          "EscapeValveHistory::append_fixed_string(type)");
    return;
  }

  const hsize_t dims[1] = {1};
  const hsize_t max_dims[1] = {H5S_UNLIMITED};
  const hsize_t chunk[1] = {256};
  const hid_t space = H5Screate_simple(1, dims, max_dims);
  TENRYU_ASSERT(space >= 0,
                "EscapeValveHistory H5Screate_simple(fixed string create) failed");
  const hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  TENRYU_ASSERT(dcpl >= 0, "EscapeValveHistory H5Pcreate(fixed string) failed");
  TENRYU_ASSERT(H5Pset_chunk(dcpl, 1, chunk) >= 0,
                "EscapeValveHistory H5Pset_chunk(fixed string) failed");
  const hid_t dset = H5Dcreate2(file,
                                path.c_str(),
                                type,
                                space,
                                H5P_DEFAULT,
                                dcpl,
                                H5P_DEFAULT);
  TENRYU_ASSERT(dset >= 0,
                "EscapeValveHistory H5Dcreate2(fixed string) failed");
  TENRYU_ASSERT(H5Dwrite(dset,
                         type,
                         H5S_ALL,
                         H5S_ALL,
                         H5P_DEFAULT,
                         buffer.data()) >= 0,
                "EscapeValveHistory H5Dwrite(fixed string create) failed");
  warn_h5_close_failure(H5Dclose(dset),
                        "H5Dclose",
                        "EscapeValveHistory::append_fixed_string(create dset)");
  warn_h5_close_failure(H5Pclose(dcpl),
                        "H5Pclose",
                        "EscapeValveHistory::append_fixed_string(dcpl)");
  warn_h5_close_failure(H5Sclose(space),
                        "H5Sclose",
                        "EscapeValveHistory::append_fixed_string(space)");
  warn_h5_close_failure(H5Tclose(type),
                        "H5Tclose",
                        "EscapeValveHistory::append_fixed_string(type)");
}

}  // namespace
#endif

void write_escape_valve_audit_history_file(
    const std::string& history_path,
    const tenryu::verification::EscapeValveAuditSnapshot& snapshot) {
  if (!snapshot.enabled || history_path.empty()) {
    return;
  }

#if TENRYU_ENABLE_HDF5
  const std::filesystem::path path(history_path);
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  const bool exists = std::filesystem::exists(path);
  const hid_t file =
      exists ? H5Fopen(path.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT)
             : H5Fcreate(path.string().c_str(),
                         H5F_ACC_TRUNC,
                         H5P_DEFAULT,
                         H5P_DEFAULT);
  TENRYU_ASSERT(file >= 0,
                "EscapeValveHistory failed to open/create history file");

  const std::string group_path = "/diagnostics/escape_valve_audit/v1";
  const hid_t group = open_or_create_group(file, group_path);
  write_string_attribute_if_missing(
      group, "schema_version", "tenryu.escape_valve_audit.v1");
  write_string_attribute_if_missing(group, "tier", snapshot.tier.c_str());
  warn_h5_close_failure(H5Gclose(group),
                        "H5Gclose",
                        "EscapeValveHistory::write group");

  const std::string base = group_path + "/";
  append_scalar_double(file, base + "time_s", snapshot.time);
  append_scalar_i64(file, base + "step", static_cast<std::int64_t>(snapshot.step));
  for (std::size_t i = 0; i < tenryu::verification::kEscapeValveFlagCount; ++i) {
    append_scalar_i64(file,
                      base + tenryu::verification::kEscapeValveFlagNames[i],
                      snapshot.counts[i]);
  }
  append_scalar_i64(file, base + "total_count", snapshot.total_count());
  append_scalar_double(
      file, base + "mass_delta_abs_cumulative", snapshot.mass_delta_abs_cumulative);
  append_scalar_double(file,
                       base + "energy_delta_abs_cumulative",
                       snapshot.energy_delta_abs_cumulative);
  append_scalar_fixed_string(file, base + "audit_status", snapshot.audit_status);

  warn_h5_close_failure(H5Fclose(file),
                        "H5Fclose",
                        "EscapeValveHistory::write file");
#else
  (void)history_path;
  (void)snapshot;
#endif
}

}  // namespace tenryu::diagnostics
