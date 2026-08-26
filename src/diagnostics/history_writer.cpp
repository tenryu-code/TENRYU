#include "diagnostics/history_writer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "core/config_validate.hpp"
#include "core/error.hpp"
#include "coupling/profile_observability.hpp"

#if TENRYU_ENABLE_HDF5
#include "io/hdf5_utils.hpp"
#endif

namespace tenryu::diagnostics {
namespace {

std::string angle_tag(const double angle_deg) {
  const long long milli = std::llround(angle_deg * 1000.0);
  return std::to_string(milli);
}

#if TENRYU_ENABLE_HDF5

template <typename Tag>
std::vector<double> copy_field_to_host(const core::Field1D<Tag>& field) {
  std::vector<double> host(field.size(), 0.0);
  field.copy_to_host(host.data());
  return host;
}

void warn_h5_close_failure(const herr_t status, const char* op, const char* context) {
  if (status < 0) {
    core::log_warning(std::string("[WARN] ") + op + " failed in " + context);
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
      TENRYU_ASSERT(gid >= 0, "HistoryWriter H5Gcreate2 failed");
      warn_h5_close_failure(H5Gclose(gid), "H5Gclose", "HistoryWriter::ensure_parent_groups");
    }
    ++pos;
  }
}

void write_units_attribute_if_missing(const hid_t dataset, const char* units) {
  if (units == nullptr || units[0] == '\0') {
    return;
  }
  constexpr const char* kUnitsAttr = "units";
  if (H5Aexists(dataset, kUnitsAttr) > 0) {
    return;
  }
  const hid_t attr_space = H5Screate(H5S_SCALAR);
  TENRYU_ASSERT(attr_space >= 0, "HistoryWriter H5Screate(units attr) failed");
  const hid_t attr_type = H5Tcopy(H5T_C_S1);
  TENRYU_ASSERT(attr_type >= 0, "HistoryWriter H5Tcopy(units attr) failed");
  TENRYU_ASSERT(H5Tset_size(attr_type, std::char_traits<char>::length(units) + 1) >= 0,
                "HistoryWriter H5Tset_size(units attr) failed");
  TENRYU_ASSERT(H5Tset_strpad(attr_type, H5T_STR_NULLTERM) >= 0,
                "HistoryWriter H5Tset_strpad(units attr) failed");
  const hid_t attr =
      H5Acreate2(dataset, kUnitsAttr, attr_type, attr_space, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HistoryWriter H5Acreate2(units attr) failed");
  TENRYU_ASSERT(H5Awrite(attr, attr_type, units) >= 0,
                "HistoryWriter H5Awrite(units attr) failed");
  H5Aclose(attr);
  warn_h5_close_failure(H5Tclose(attr_type), "H5Tclose",
                        "HistoryWriter::write_units_attribute_if_missing");
  warn_h5_close_failure(H5Sclose(attr_space), "H5Sclose",
                        "HistoryWriter::write_units_attribute_if_missing");
}

void write_double_vector_attribute_if_missing(const hid_t dataset,
                                              const char* attr_name,
                                              const std::vector<double>& values) {
  if (attr_name == nullptr || attr_name[0] == '\0' || values.empty()) {
    return;
  }
  if (H5Aexists(dataset, attr_name) > 0) {
    return;
  }
  const hsize_t dims[1] = {static_cast<hsize_t>(values.size())};
  const hid_t attr_space = H5Screate_simple(1, dims, nullptr);
  TENRYU_ASSERT(attr_space >= 0, "HistoryWriter H5Screate_simple(vector attr) failed");
  const hid_t attr =
      H5Acreate2(dataset, attr_name, H5T_NATIVE_DOUBLE, attr_space, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HistoryWriter H5Acreate2(vector attr) failed");
  TENRYU_ASSERT(H5Awrite(attr, H5T_NATIVE_DOUBLE, values.data()) >= 0,
                "HistoryWriter H5Awrite(vector attr) failed");
  warn_h5_close_failure(H5Aclose(attr), "H5Aclose",
                        "HistoryWriter::write_double_vector_attribute_if_missing");
  warn_h5_close_failure(H5Sclose(attr_space), "H5Sclose",
                        "HistoryWriter::write_double_vector_attribute_if_missing");
}

hid_t open_or_create_group(const hid_t file, const std::string& path) {
  ensure_parent_groups(file, path + "/_");
  if (tenryu::io::h5_link_exists(file, path.c_str(), H5P_DEFAULT) > 0) {
    const hid_t gid = H5Gopen2(file, path.c_str(), H5P_DEFAULT);
    TENRYU_ASSERT(gid >= 0, "HistoryWriter H5Gopen2 failed: " + path);
    return gid;
  }
  const hid_t gid =
      H5Gcreate2(file, path.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(gid >= 0, "HistoryWriter H5Gcreate2 failed: " + path);
  return gid;
}

void write_string_attribute_if_missing(const hid_t object,
                                       const char* name,
                                       const char* value) {
  if (value == nullptr) {
    return;
  }
  if (H5Aexists(object, name) > 0) {
    return;
  }
  const hid_t attr_space = H5Screate(H5S_SCALAR);
  TENRYU_ASSERT(attr_space >= 0, "HistoryWriter H5Screate(string attr) failed");
  const hid_t attr_type = H5Tcopy(H5T_C_S1);
  TENRYU_ASSERT(attr_type >= 0, "HistoryWriter H5Tcopy(string attr) failed");
  TENRYU_ASSERT(H5Tset_size(attr_type, std::char_traits<char>::length(value) + 1) >= 0,
                "HistoryWriter H5Tset_size(string attr) failed");
  TENRYU_ASSERT(H5Tset_strpad(attr_type, H5T_STR_NULLTERM) >= 0,
                "HistoryWriter H5Tset_strpad(string attr) failed");
  const hid_t attr =
      H5Acreate2(object, name, attr_type, attr_space, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HistoryWriter H5Acreate2(string attr) failed");
  TENRYU_ASSERT(H5Awrite(attr, attr_type, value) >= 0,
                "HistoryWriter H5Awrite(string attr) failed");
  warn_h5_close_failure(H5Aclose(attr), "H5Aclose",
                        "HistoryWriter::write_string_attribute_if_missing");
  warn_h5_close_failure(H5Tclose(attr_type), "H5Tclose",
                        "HistoryWriter::write_string_attribute_if_missing");
  warn_h5_close_failure(H5Sclose(attr_space), "H5Sclose",
                        "HistoryWriter::write_string_attribute_if_missing");
}

void write_u8_attribute_if_missing(const hid_t object,
                                   const char* name,
                                   const std::uint8_t value) {
  if (H5Aexists(object, name) > 0) {
    return;
  }
  const hid_t attr_space = H5Screate(H5S_SCALAR);
  TENRYU_ASSERT(attr_space >= 0, "HistoryWriter H5Screate(u8 attr) failed");
  const hid_t attr =
      H5Acreate2(object, name, H5T_NATIVE_UINT8, attr_space, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HistoryWriter H5Acreate2(u8 attr) failed");
  TENRYU_ASSERT(H5Awrite(attr, H5T_NATIVE_UINT8, &value) >= 0,
                "HistoryWriter H5Awrite(u8 attr) failed");
  warn_h5_close_failure(H5Aclose(attr), "H5Aclose",
                        "HistoryWriter::write_u8_attribute_if_missing");
  warn_h5_close_failure(H5Sclose(attr_space), "H5Sclose",
                        "HistoryWriter::write_u8_attribute_if_missing");
}

void write_i32_attribute_if_missing(const hid_t object,
                                    const char* name,
                                    const std::int32_t value) {
  if (H5Aexists(object, name) > 0) {
    return;
  }
  const hid_t attr_space = H5Screate(H5S_SCALAR);
  TENRYU_ASSERT(attr_space >= 0, "HistoryWriter H5Screate(i32 attr) failed");
  const hid_t attr =
      H5Acreate2(object, name, H5T_NATIVE_INT32, attr_space, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HistoryWriter H5Acreate2(i32 attr) failed");
  TENRYU_ASSERT(H5Awrite(attr, H5T_NATIVE_INT32, &value) >= 0,
                "HistoryWriter H5Awrite(i32 attr) failed");
  warn_h5_close_failure(H5Aclose(attr), "H5Aclose",
                        "HistoryWriter::write_i32_attribute_if_missing");
  warn_h5_close_failure(H5Sclose(attr_space), "H5Sclose",
                        "HistoryWriter::write_i32_attribute_if_missing");
}

void write_double_attribute_if_missing(const hid_t object,
                                       const char* name,
                                       const double value) {
  if (H5Aexists(object, name) > 0) {
    return;
  }
  const hid_t attr_space = H5Screate(H5S_SCALAR);
  TENRYU_ASSERT(attr_space >= 0, "HistoryWriter H5Screate(double attr) failed");
  const hid_t attr =
      H5Acreate2(object, name, H5T_NATIVE_DOUBLE, attr_space, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HistoryWriter H5Acreate2(double attr) failed");
  TENRYU_ASSERT(H5Awrite(attr, H5T_NATIVE_DOUBLE, &value) >= 0,
                "HistoryWriter H5Awrite(double attr) failed");
  warn_h5_close_failure(H5Aclose(attr), "H5Aclose",
                        "HistoryWriter::write_double_attribute_if_missing");
  warn_h5_close_failure(H5Sclose(attr_space), "H5Sclose",
                        "HistoryWriter::write_double_attribute_if_missing");
}

std::optional<double> read_double_attribute_if_exists(const hid_t object,
                                                      const char* name) {
  if (H5Aexists(object, name) <= 0) {
    return std::nullopt;
  }
  const hid_t attr = H5Aopen(object, name, H5P_DEFAULT);
  TENRYU_ASSERT(attr >= 0, "HistoryWriter H5Aopen(double attr) failed");
  double value = 0.0;
  TENRYU_ASSERT(H5Aread(attr, H5T_NATIVE_DOUBLE, &value) >= 0,
                "HistoryWriter H5Aread(double attr) failed");
  warn_h5_close_failure(H5Aclose(attr), "H5Aclose",
                        "HistoryWriter::read_double_attribute_if_exists");
  return value;
}

void append_scalar_double(const hid_t file,
                          const std::string& path,
                          const double value,
                          const char* units) {
  ensure_parent_groups(file, path);

  if (tenryu::io::h5_link_exists(file, path.c_str(), H5P_DEFAULT) > 0) {
    const hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    TENRYU_ASSERT(dset >= 0, "HistoryWriter H5Dopen2(double) failed");
    write_units_attribute_if_missing(dset, units);

    const hid_t fspace = H5Dget_space(dset);
    TENRYU_ASSERT(fspace >= 0, "HistoryWriter H5Dget_space(double) failed");
    hsize_t dims[1] = {0};
    const int rank = H5Sget_simple_extent_dims(fspace, dims, nullptr);
    TENRYU_ASSERT(rank == 1, "HistoryWriter H5Sget_simple_extent_dims(double) rank mismatch");
    warn_h5_close_failure(H5Sclose(fspace), "H5Sclose",
                          "HistoryWriter::append_scalar_double(extent query)");

    const hsize_t new_dims[1] = {dims[0] + 1};
    TENRYU_ASSERT(H5Dset_extent(dset, new_dims) >= 0,
                  "HistoryWriter H5Dset_extent(double) failed");

    const hid_t file_space = H5Dget_space(dset);
    TENRYU_ASSERT(file_space >= 0, "HistoryWriter H5Dget_space(post-extend) failed");
    const hsize_t start[1] = {dims[0]};
    const hsize_t count[1] = {1};
    TENRYU_ASSERT(H5Sselect_hyperslab(file_space,
                                      H5S_SELECT_SET,
                                      start,
                                      nullptr,
                                      count,
                                      nullptr) >= 0,
                  "HistoryWriter H5Sselect_hyperslab(double) failed");
    const hid_t mem_space = H5Screate_simple(1, count, nullptr);
    TENRYU_ASSERT(mem_space >= 0, "HistoryWriter H5Screate_simple(double) failed");
    TENRYU_ASSERT(H5Dwrite(dset,
                           H5T_NATIVE_DOUBLE,
                           mem_space,
                           file_space,
                           H5P_DEFAULT,
                           &value) >= 0,
                  "HistoryWriter H5Dwrite(double append) failed");
    warn_h5_close_failure(H5Sclose(mem_space), "H5Sclose",
                          "HistoryWriter::append_scalar_double(mem space)");
    warn_h5_close_failure(H5Sclose(file_space), "H5Sclose",
                          "HistoryWriter::append_scalar_double(file space)");
    warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                          "HistoryWriter::append_scalar_double(dataset)");
    return;
  }

  const hsize_t dims[1] = {1};
  const hsize_t max_dims[1] = {H5S_UNLIMITED};
  const hsize_t chunk[1] = {256};
  const hid_t space = H5Screate_simple(1, dims, max_dims);
  TENRYU_ASSERT(space >= 0, "HistoryWriter H5Screate_simple(double create) failed");
  const hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  TENRYU_ASSERT(dcpl >= 0, "HistoryWriter H5Pcreate(double) failed");
  TENRYU_ASSERT(H5Pset_chunk(dcpl, 1, chunk) >= 0,
                "HistoryWriter H5Pset_chunk(double) failed");
  const hid_t dset = H5Dcreate2(file,
                                path.c_str(),
                                H5T_NATIVE_DOUBLE,
                                space,
                                H5P_DEFAULT,
                                dcpl,
                                H5P_DEFAULT);
  TENRYU_ASSERT(dset >= 0, "HistoryWriter H5Dcreate2(double) failed");
  write_units_attribute_if_missing(dset, units);
  TENRYU_ASSERT(H5Dwrite(dset,
                         H5T_NATIVE_DOUBLE,
                         H5S_ALL,
                         H5S_ALL,
                         H5P_DEFAULT,
                         &value) >= 0,
                "HistoryWriter H5Dwrite(double create) failed");
  warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                        "HistoryWriter::append_scalar_double(create dataset)");
  warn_h5_close_failure(H5Pclose(dcpl), "H5Pclose",
                        "HistoryWriter::append_scalar_double(create dcpl)");
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HistoryWriter::append_scalar_double(create space)");
}

void append_scalar_i64(const hid_t file,
                       const std::string& path,
                       const std::int64_t value,
                       const char* units) {
  ensure_parent_groups(file, path);

  if (tenryu::io::h5_link_exists(file, path.c_str(), H5P_DEFAULT) > 0) {
    const hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    TENRYU_ASSERT(dset >= 0, "HistoryWriter H5Dopen2(i64) failed");
    write_units_attribute_if_missing(dset, units);

    const hid_t fspace = H5Dget_space(dset);
    TENRYU_ASSERT(fspace >= 0, "HistoryWriter H5Dget_space(i64) failed");
    hsize_t dims[1] = {0};
    const int rank = H5Sget_simple_extent_dims(fspace, dims, nullptr);
    TENRYU_ASSERT(rank == 1, "HistoryWriter H5Sget_simple_extent_dims(i64) rank mismatch");
    warn_h5_close_failure(H5Sclose(fspace), "H5Sclose",
                          "HistoryWriter::append_scalar_i64(extent query)");

    const hsize_t new_dims[1] = {dims[0] + 1};
    TENRYU_ASSERT(H5Dset_extent(dset, new_dims) >= 0,
                  "HistoryWriter H5Dset_extent(i64) failed");

    const hid_t file_space = H5Dget_space(dset);
    TENRYU_ASSERT(file_space >= 0, "HistoryWriter H5Dget_space(post-extend i64) failed");
    const hsize_t start[1] = {dims[0]};
    const hsize_t count[1] = {1};
    TENRYU_ASSERT(H5Sselect_hyperslab(file_space,
                                      H5S_SELECT_SET,
                                      start,
                                      nullptr,
                                      count,
                                      nullptr) >= 0,
                  "HistoryWriter H5Sselect_hyperslab(i64) failed");
    const hid_t mem_space = H5Screate_simple(1, count, nullptr);
    TENRYU_ASSERT(mem_space >= 0, "HistoryWriter H5Screate_simple(i64) failed");
    TENRYU_ASSERT(H5Dwrite(dset,
                           H5T_NATIVE_INT64,
                           mem_space,
                           file_space,
                           H5P_DEFAULT,
                           &value) >= 0,
                  "HistoryWriter H5Dwrite(i64 append) failed");
    warn_h5_close_failure(H5Sclose(mem_space), "H5Sclose",
                          "HistoryWriter::append_scalar_i64(mem space)");
    warn_h5_close_failure(H5Sclose(file_space), "H5Sclose",
                          "HistoryWriter::append_scalar_i64(file space)");
    warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                          "HistoryWriter::append_scalar_i64(dataset)");
    return;
  }

  const hsize_t dims[1] = {1};
  const hsize_t max_dims[1] = {H5S_UNLIMITED};
  const hsize_t chunk[1] = {256};
  const hid_t space = H5Screate_simple(1, dims, max_dims);
  TENRYU_ASSERT(space >= 0, "HistoryWriter H5Screate_simple(i64 create) failed");
  const hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  TENRYU_ASSERT(dcpl >= 0, "HistoryWriter H5Pcreate(i64) failed");
  TENRYU_ASSERT(H5Pset_chunk(dcpl, 1, chunk) >= 0,
                "HistoryWriter H5Pset_chunk(i64) failed");
  const hid_t dset = H5Dcreate2(file,
                                path.c_str(),
                                H5T_NATIVE_INT64,
                                space,
                                H5P_DEFAULT,
                                dcpl,
                                H5P_DEFAULT);
  TENRYU_ASSERT(dset >= 0, "HistoryWriter H5Dcreate2(i64) failed");
  write_units_attribute_if_missing(dset, units);
  TENRYU_ASSERT(H5Dwrite(dset,
                         H5T_NATIVE_INT64,
                         H5S_ALL,
                         H5S_ALL,
                         H5P_DEFAULT,
                         &value) >= 0,
                "HistoryWriter H5Dwrite(i64 create) failed");
  warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                        "HistoryWriter::append_scalar_i64(create dataset)");
  warn_h5_close_failure(H5Pclose(dcpl), "H5Pclose",
                        "HistoryWriter::append_scalar_i64(create dcpl)");
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HistoryWriter::append_scalar_i64(create space)");
}

template <typename T>
void append_scalar_native(const hid_t file,
                          const std::string& path,
                          const T value,
                          const hid_t h5_type,
                          const char* units,
                          const char* label) {
  ensure_parent_groups(file, path);

  if (tenryu::io::h5_link_exists(file, path.c_str(), H5P_DEFAULT) > 0) {
    const hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    TENRYU_ASSERT(dset >= 0,
                  std::string("HistoryWriter H5Dopen2(") + label + ") failed");
    write_units_attribute_if_missing(dset, units);

    const hid_t fspace = H5Dget_space(dset);
    TENRYU_ASSERT(fspace >= 0,
                  std::string("HistoryWriter H5Dget_space(") + label + ") failed");
    hsize_t dims[1] = {0};
    const int rank = H5Sget_simple_extent_dims(fspace, dims, nullptr);
    TENRYU_ASSERT(rank == 1,
                  std::string("HistoryWriter rank mismatch for ") + label);
    warn_h5_close_failure(H5Sclose(fspace), "H5Sclose",
                          "HistoryWriter::append_scalar_native(extent query)");

    const hsize_t new_dims[1] = {dims[0] + 1};
    TENRYU_ASSERT(H5Dset_extent(dset, new_dims) >= 0,
                  std::string("HistoryWriter H5Dset_extent(") + label +
                      ") failed");

    const hid_t file_space = H5Dget_space(dset);
    TENRYU_ASSERT(file_space >= 0,
                  std::string("HistoryWriter H5Dget_space(post-extend ") +
                      label + ") failed");
    const hsize_t start[1] = {dims[0]};
    const hsize_t count[1] = {1};
    TENRYU_ASSERT(H5Sselect_hyperslab(file_space,
                                      H5S_SELECT_SET,
                                      start,
                                      nullptr,
                                      count,
                                      nullptr) >= 0,
                  std::string("HistoryWriter H5Sselect_hyperslab(") + label +
                      ") failed");
    const hid_t mem_space = H5Screate_simple(1, count, nullptr);
    TENRYU_ASSERT(mem_space >= 0,
                  std::string("HistoryWriter H5Screate_simple(") + label +
                      ") failed");
    TENRYU_ASSERT(H5Dwrite(dset,
                           h5_type,
                           mem_space,
                           file_space,
                           H5P_DEFAULT,
                           &value) >= 0,
                  std::string("HistoryWriter H5Dwrite(") + label +
                      " append) failed");
    warn_h5_close_failure(H5Sclose(mem_space), "H5Sclose",
                          "HistoryWriter::append_scalar_native(mem space)");
    warn_h5_close_failure(H5Sclose(file_space), "H5Sclose",
                          "HistoryWriter::append_scalar_native(file space)");
    warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                          "HistoryWriter::append_scalar_native(dataset)");
    return;
  }

  const hsize_t dims[1] = {1};
  const hsize_t max_dims[1] = {H5S_UNLIMITED};
  const hsize_t chunk[1] = {256};
  const hid_t space = H5Screate_simple(1, dims, max_dims);
  TENRYU_ASSERT(space >= 0,
                std::string("HistoryWriter H5Screate_simple(") + label +
                    " create) failed");
  const hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  TENRYU_ASSERT(dcpl >= 0,
                std::string("HistoryWriter H5Pcreate(") + label + ") failed");
  TENRYU_ASSERT(H5Pset_chunk(dcpl, 1, chunk) >= 0,
                std::string("HistoryWriter H5Pset_chunk(") + label +
                    ") failed");
  const hid_t dset = H5Dcreate2(file,
                                path.c_str(),
                                h5_type,
                                space,
                                H5P_DEFAULT,
                                dcpl,
                                H5P_DEFAULT);
  TENRYU_ASSERT(dset >= 0,
                std::string("HistoryWriter H5Dcreate2(") + label +
                    ") failed");
  write_units_attribute_if_missing(dset, units);
  TENRYU_ASSERT(H5Dwrite(dset,
                         h5_type,
                         H5S_ALL,
                         H5S_ALL,
                         H5P_DEFAULT,
                         &value) >= 0,
                std::string("HistoryWriter H5Dwrite(") + label +
                    " create) failed");
  warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                        "HistoryWriter::append_scalar_native(create dataset)");
  warn_h5_close_failure(H5Pclose(dcpl), "H5Pclose",
                        "HistoryWriter::append_scalar_native(create dcpl)");
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HistoryWriter::append_scalar_native(create space)");
}

void append_scalar_i32(const hid_t file,
                       const std::string& path,
                       const std::int32_t value,
                       const char* units) {
  append_scalar_native(file, path, value, H5T_NATIVE_INT32, units, "i32");
}

void append_scalar_u64(const hid_t file,
                       const std::string& path,
                       const std::uint64_t value,
                       const char* units) {
  append_scalar_native(file, path, value, H5T_NATIVE_UINT64, units, "u64");
}

void append_scalar_u16(const hid_t file,
                       const std::string& path,
                       const std::uint16_t value,
                       const char* units) {
  append_scalar_native(file, path, value, H5T_NATIVE_UINT16, units, "u16");
}

void append_scalar_u8(const hid_t file,
                      const std::string& path,
                      const std::uint8_t value,
                      const char* units) {
  append_scalar_native(file, path, value, H5T_NATIVE_UINT8, units, "u8");
}

void append_scalar_fixed_string(const hid_t file,
                                const std::string& path,
                                const std::string& value) {
  constexpr std::size_t kStringBytes = 128;
  ensure_parent_groups(file, path);
  std::array<char, kStringBytes> buffer{};
  if (value.size() >= kStringBytes) {
    core::log_warning(
        "HistoryWriter: truncating ALE provenance string dataset value for " +
        path);
  }
  const std::size_t n_copy = std::min(value.size(), kStringBytes - 1);
  std::copy_n(value.data(), n_copy, buffer.data());

  const hid_t type = H5Tcopy(H5T_C_S1);
  TENRYU_ASSERT(type >= 0, "HistoryWriter H5Tcopy(fixed string) failed");
  TENRYU_ASSERT(H5Tset_size(type, kStringBytes) >= 0,
                "HistoryWriter H5Tset_size(fixed string) failed");
  TENRYU_ASSERT(H5Tset_strpad(type, H5T_STR_NULLTERM) >= 0,
                "HistoryWriter H5Tset_strpad(fixed string) failed");

  if (tenryu::io::h5_link_exists(file, path.c_str(), H5P_DEFAULT) > 0) {
    const hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    TENRYU_ASSERT(dset >= 0, "HistoryWriter H5Dopen2(fixed string) failed");
    const hid_t fspace = H5Dget_space(dset);
    TENRYU_ASSERT(fspace >= 0, "HistoryWriter H5Dget_space(fixed string) failed");
    hsize_t dims[1] = {0};
    const int rank = H5Sget_simple_extent_dims(fspace, dims, nullptr);
    TENRYU_ASSERT(rank == 1, "HistoryWriter fixed string rank mismatch");
    warn_h5_close_failure(H5Sclose(fspace), "H5Sclose",
                          "HistoryWriter::append_scalar_fixed_string(extent query)");

    const hsize_t new_dims[1] = {dims[0] + 1};
    TENRYU_ASSERT(H5Dset_extent(dset, new_dims) >= 0,
                  "HistoryWriter H5Dset_extent(fixed string) failed");
    const hid_t file_space = H5Dget_space(dset);
    TENRYU_ASSERT(file_space >= 0,
                  "HistoryWriter H5Dget_space(post-extend fixed string) failed");
    const hsize_t start[1] = {dims[0]};
    const hsize_t count[1] = {1};
    TENRYU_ASSERT(H5Sselect_hyperslab(file_space,
                                      H5S_SELECT_SET,
                                      start,
                                      nullptr,
                                      count,
                                      nullptr) >= 0,
                  "HistoryWriter H5Sselect_hyperslab(fixed string) failed");
    const hid_t mem_space = H5Screate_simple(1, count, nullptr);
    TENRYU_ASSERT(mem_space >= 0,
                  "HistoryWriter H5Screate_simple(fixed string) failed");
    TENRYU_ASSERT(H5Dwrite(dset,
                           type,
                           mem_space,
                           file_space,
                           H5P_DEFAULT,
                           buffer.data()) >= 0,
                  "HistoryWriter H5Dwrite(fixed string append) failed");
    warn_h5_close_failure(H5Sclose(mem_space), "H5Sclose",
                          "HistoryWriter::append_scalar_fixed_string(mem space)");
    warn_h5_close_failure(H5Sclose(file_space), "H5Sclose",
                          "HistoryWriter::append_scalar_fixed_string(file space)");
    warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                          "HistoryWriter::append_scalar_fixed_string(dataset)");
    warn_h5_close_failure(H5Tclose(type), "H5Tclose",
                          "HistoryWriter::append_scalar_fixed_string(type)");
    return;
  }

  const hsize_t dims[1] = {1};
  const hsize_t max_dims[1] = {H5S_UNLIMITED};
  const hsize_t chunk[1] = {256};
  const hid_t space = H5Screate_simple(1, dims, max_dims);
  TENRYU_ASSERT(space >= 0,
                "HistoryWriter H5Screate_simple(fixed string create) failed");
  const hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  TENRYU_ASSERT(dcpl >= 0, "HistoryWriter H5Pcreate(fixed string) failed");
  TENRYU_ASSERT(H5Pset_chunk(dcpl, 1, chunk) >= 0,
                "HistoryWriter H5Pset_chunk(fixed string) failed");
  const hid_t dset = H5Dcreate2(file,
                                path.c_str(),
                                type,
                                space,
                                H5P_DEFAULT,
                                dcpl,
                                H5P_DEFAULT);
  TENRYU_ASSERT(dset >= 0, "HistoryWriter H5Dcreate2(fixed string) failed");
  TENRYU_ASSERT(H5Dwrite(dset,
                         type,
                         H5S_ALL,
                         H5S_ALL,
                         H5P_DEFAULT,
                         buffer.data()) >= 0,
                "HistoryWriter H5Dwrite(fixed string create) failed");
  warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                        "HistoryWriter::append_scalar_fixed_string(create dataset)");
  warn_h5_close_failure(H5Pclose(dcpl), "H5Pclose",
                        "HistoryWriter::append_scalar_fixed_string(create dcpl)");
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HistoryWriter::append_scalar_fixed_string(create space)");
  warn_h5_close_failure(H5Tclose(type), "H5Tclose",
                        "HistoryWriter::append_scalar_fixed_string(create type)");
}

void append_scalar_vlen_string(const hid_t file,
                               const std::string& path,
                               const std::string& value) {
  ensure_parent_groups(file, path);
  const char* raw = value.c_str();
  const hid_t type = H5Tcopy(H5T_C_S1);
  TENRYU_ASSERT(type >= 0, "HistoryWriter H5Tcopy(vlen string) failed");
  TENRYU_ASSERT(H5Tset_size(type, H5T_VARIABLE) >= 0,
                "HistoryWriter H5Tset_size(vlen string) failed");

  if (tenryu::io::h5_link_exists(file, path.c_str(), H5P_DEFAULT) > 0) {
    const hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    TENRYU_ASSERT(dset >= 0, "HistoryWriter H5Dopen2(vlen string) failed");
    const hid_t fspace = H5Dget_space(dset);
    TENRYU_ASSERT(fspace >= 0, "HistoryWriter H5Dget_space(vlen string) failed");
    hsize_t dims[1] = {0};
    const int rank = H5Sget_simple_extent_dims(fspace, dims, nullptr);
    TENRYU_ASSERT(rank == 1, "HistoryWriter vlen string rank mismatch");
    warn_h5_close_failure(H5Sclose(fspace), "H5Sclose",
                          "HistoryWriter::append_scalar_vlen_string(extent query)");

    const hsize_t new_dims[1] = {dims[0] + 1};
    TENRYU_ASSERT(H5Dset_extent(dset, new_dims) >= 0,
                  "HistoryWriter H5Dset_extent(vlen string) failed");
    const hid_t file_space = H5Dget_space(dset);
    TENRYU_ASSERT(file_space >= 0,
                  "HistoryWriter H5Dget_space(post-extend vlen string) failed");
    const hsize_t start[1] = {dims[0]};
    const hsize_t count[1] = {1};
    TENRYU_ASSERT(H5Sselect_hyperslab(file_space,
                                      H5S_SELECT_SET,
                                      start,
                                      nullptr,
                                      count,
                                      nullptr) >= 0,
                  "HistoryWriter H5Sselect_hyperslab(vlen string) failed");
    const hid_t mem_space = H5Screate_simple(1, count, nullptr);
    TENRYU_ASSERT(mem_space >= 0,
                  "HistoryWriter H5Screate_simple(vlen string) failed");
    TENRYU_ASSERT(H5Dwrite(dset,
                           type,
                           mem_space,
                           file_space,
                           H5P_DEFAULT,
                           &raw) >= 0,
                  "HistoryWriter H5Dwrite(vlen string append) failed");
    warn_h5_close_failure(H5Sclose(mem_space), "H5Sclose",
                          "HistoryWriter::append_scalar_vlen_string(mem space)");
    warn_h5_close_failure(H5Sclose(file_space), "H5Sclose",
                          "HistoryWriter::append_scalar_vlen_string(file space)");
    warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                          "HistoryWriter::append_scalar_vlen_string(dataset)");
    warn_h5_close_failure(H5Tclose(type), "H5Tclose",
                          "HistoryWriter::append_scalar_vlen_string(type)");
    return;
  }

  const hsize_t dims[1] = {1};
  const hsize_t max_dims[1] = {H5S_UNLIMITED};
  const hsize_t chunk[1] = {256};
  const hid_t space = H5Screate_simple(1, dims, max_dims);
  TENRYU_ASSERT(space >= 0,
                "HistoryWriter H5Screate_simple(vlen string create) failed");
  const hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  TENRYU_ASSERT(dcpl >= 0, "HistoryWriter H5Pcreate(vlen string) failed");
  TENRYU_ASSERT(H5Pset_chunk(dcpl, 1, chunk) >= 0,
                "HistoryWriter H5Pset_chunk(vlen string) failed");
  const hid_t dset = H5Dcreate2(file,
                                path.c_str(),
                                type,
                                space,
                                H5P_DEFAULT,
                                dcpl,
                                H5P_DEFAULT);
  TENRYU_ASSERT(dset >= 0, "HistoryWriter H5Dcreate2(vlen string) failed");
  TENRYU_ASSERT(H5Dwrite(dset,
                         type,
                         H5S_ALL,
                         H5S_ALL,
                         H5P_DEFAULT,
                         &raw) >= 0,
                "HistoryWriter H5Dwrite(vlen string create) failed");
  warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                        "HistoryWriter::append_scalar_vlen_string(create dataset)");
  warn_h5_close_failure(H5Pclose(dcpl), "H5Pclose",
                        "HistoryWriter::append_scalar_vlen_string(create dcpl)");
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HistoryWriter::append_scalar_vlen_string(create space)");
  warn_h5_close_failure(H5Tclose(type), "H5Tclose",
                        "HistoryWriter::append_scalar_vlen_string(create type)");
}

void append_3x3_i32_matrix(const hid_t file,
                           const std::string& path,
                           const int values[3][3],
                           const char* units) {
  ensure_parent_groups(file, path);
  std::int32_t packed[3][3] = {};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      packed[i][j] = static_cast<std::int32_t>(values[i][j]);
    }
  }

  if (tenryu::io::h5_link_exists(file, path.c_str(), H5P_DEFAULT) > 0) {
    const hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    TENRYU_ASSERT(dset >= 0, "HistoryWriter H5Dopen2(3x3 i32) failed");
    write_units_attribute_if_missing(dset, units);
    const hid_t fspace = H5Dget_space(dset);
    TENRYU_ASSERT(fspace >= 0, "HistoryWriter H5Dget_space(3x3 i32) failed");
    hsize_t dims[3] = {0, 0, 0};
    const int rank = H5Sget_simple_extent_dims(fspace, dims, nullptr);
    TENRYU_ASSERT(rank == 3, "HistoryWriter 3x3 matrix rank mismatch for path: " + path);
    TENRYU_ASSERT(dims[1] == 3 && dims[2] == 3,
                  "HistoryWriter 3x3 matrix extent mismatch for path: " + path);
    warn_h5_close_failure(H5Sclose(fspace), "H5Sclose",
                          "HistoryWriter::append_3x3_i32_matrix(extent query)");

    const hsize_t new_dims[3] = {dims[0] + 1, 3, 3};
    TENRYU_ASSERT(H5Dset_extent(dset, new_dims) >= 0,
                  "HistoryWriter H5Dset_extent(3x3 i32) failed");
    const hid_t file_space = H5Dget_space(dset);
    TENRYU_ASSERT(file_space >= 0,
                  "HistoryWriter H5Dget_space(post-extend 3x3 i32) failed");
    const hsize_t start[3] = {dims[0], 0, 0};
    const hsize_t count[3] = {1, 3, 3};
    TENRYU_ASSERT(H5Sselect_hyperslab(file_space,
                                      H5S_SELECT_SET,
                                      start,
                                      nullptr,
                                      count,
                                      nullptr) >= 0,
                  "HistoryWriter H5Sselect_hyperslab(3x3 i32) failed");
    const hid_t mem_space = H5Screate_simple(3, count, nullptr);
    TENRYU_ASSERT(mem_space >= 0,
                  "HistoryWriter H5Screate_simple(3x3 i32) failed");
    TENRYU_ASSERT(H5Dwrite(dset,
                           H5T_NATIVE_INT32,
                           mem_space,
                           file_space,
                           H5P_DEFAULT,
                           packed) >= 0,
                  "HistoryWriter H5Dwrite(3x3 i32 append) failed");
    warn_h5_close_failure(H5Sclose(mem_space), "H5Sclose",
                          "HistoryWriter::append_3x3_i32_matrix(mem space)");
    warn_h5_close_failure(H5Sclose(file_space), "H5Sclose",
                          "HistoryWriter::append_3x3_i32_matrix(file space)");
    warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                          "HistoryWriter::append_3x3_i32_matrix(dataset)");
    return;
  }

  const hsize_t dims[3] = {1, 3, 3};
  const hsize_t max_dims[3] = {H5S_UNLIMITED, 3, 3};
  const hsize_t chunk[3] = {256, 3, 3};
  const hid_t space = H5Screate_simple(3, dims, max_dims);
  TENRYU_ASSERT(space >= 0,
                "HistoryWriter H5Screate_simple(3x3 i32 create) failed");
  const hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  TENRYU_ASSERT(dcpl >= 0, "HistoryWriter H5Pcreate(3x3 i32) failed");
  TENRYU_ASSERT(H5Pset_chunk(dcpl, 3, chunk) >= 0,
                "HistoryWriter H5Pset_chunk(3x3 i32) failed");
  const hid_t dset = H5Dcreate2(file,
                                path.c_str(),
                                H5T_NATIVE_INT32,
                                space,
                                H5P_DEFAULT,
                                dcpl,
                                H5P_DEFAULT);
  TENRYU_ASSERT(dset >= 0, "HistoryWriter H5Dcreate2(3x3 i32) failed");
  write_units_attribute_if_missing(dset, units);
  TENRYU_ASSERT(H5Dwrite(dset,
                         H5T_NATIVE_INT32,
                         H5S_ALL,
                         H5S_ALL,
                         H5P_DEFAULT,
                         packed) >= 0,
                "HistoryWriter H5Dwrite(3x3 i32 create) failed");
  warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                        "HistoryWriter::append_3x3_i32_matrix(create dataset)");
  warn_h5_close_failure(H5Pclose(dcpl), "H5Pclose",
                        "HistoryWriter::append_3x3_i32_matrix(create dcpl)");
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HistoryWriter::append_3x3_i32_matrix(create space)");
}

constexpr std::size_t kPlicEventsPerStepLimit = 10000;

std::uint8_t plic_reconstruction_kind_from_event(
    const tenryu::coupling::ProfileObservability::PlicDegradationEvent& evt) {
  if (evt.case_id == 1) {
    return 2U;
  }
  if (evt.case_id == 3) {
    return 3U;
  }
  if (!evt.fallback_used.empty() && evt.fallback_used != "none") {
    return 4U;
  }
  return 1U;
}

void append_plic_event_row(
    const hid_t file,
    const std::string& base,
    const tenryu::coupling::ProfileObservability::PlicDegradationEvent& evt) {
  append_scalar_u8(file, base + "case_id", evt.case_id, "enum");
  append_scalar_u8(file, base + "severity", evt.severity, "enum");
  append_scalar_i32(file,
                    base + "cell_idx",
                    static_cast<std::int32_t>(evt.cell_idx),
                    "cell");
  append_scalar_i32(file, base + "i", static_cast<std::int32_t>(evt.i), "index");
  append_scalar_i32(file, base + "j", static_cast<std::int32_t>(evt.j), "index");
  append_scalar_double(file, base + "eta_E", evt.eta_E, "dimensionless");
  append_scalar_double(
      file, base + "grad_F_magnitude", evt.grad_F_magnitude, "dimensionless");
  append_scalar_vlen_string(
      file, base + "severity_metric_kind", evt.severity_metric_kind);
  append_scalar_double(file,
                       base + "severity_metric_value",
                       evt.severity_metric_value,
                       "dimensionless");
  append_scalar_vlen_string(file, base + "fallback_used", evt.fallback_used);
  append_scalar_u8(file,
                   base + "prev_normal_invalidated",
                   static_cast<std::uint8_t>(
                       evt.prev_normal_invalidated ? 1U : 0U),
                   "bool");
  append_scalar_i32(file,
                    base + "step",
                    static_cast<std::int32_t>(evt.step),
                    "count");
  append_scalar_double(file, base + "time", evt.time, "s");
}

void append_plic_event_summary_row(const hid_t file,
                                   const std::string& base,
                                   const std::uint8_t case_id,
                                   const std::uint8_t severity,
                                   const std::int32_t count,
                                   const int step,
                                   const double time) {
  append_scalar_u8(file, base + "case_id", case_id, "enum");
  append_scalar_u8(file, base + "severity", severity, "enum");
  append_scalar_i32(file, base + "count", count, "count");
  append_scalar_i32(file, base + "step", static_cast<std::int32_t>(step), "count");
  append_scalar_double(file, base + "time", time, "s");
}

void append_plic_per_cell_state_row(
    const hid_t file,
    const std::string& base,
    const tenryu::coupling::ProfileObservability::PlicDegradationEvent& evt) {
  append_scalar_double(file,
                       base + "volfrac_gradient_magnitude",
                       evt.grad_F_magnitude,
                       "dimensionless");
  append_scalar_double(file, base + "interface_normal_r", 0.0, "dimensionless");
  append_scalar_double(file, base + "interface_normal_z", 0.0, "dimensionless");
  append_scalar_double(file, base + "interface_offset_cm", 0.0, "cm");
  append_scalar_u8(file,
                   base + "plic_reconstruction_kind",
                   plic_reconstruction_kind_from_event(evt),
                   "enum");
}

bool material_interface_per_cell_state_enabled(
    const tenryu::core::Config& cfg) {
  return cfg.numerics.plic.material_interface_per_cell_state ==
             "sparse_on_degradation" ||
         cfg.numerics.plic.material_interface_per_cell_state == "dense_debug";
}

void append_row_double_matrix(const hid_t file,
                              const std::string& path,
                              const std::vector<double>& row_values,
                              const char* units) {
  if (row_values.empty()) {
    return;
  }
  ensure_parent_groups(file, path);
  const hsize_t n_cols = static_cast<hsize_t>(row_values.size());

  if (tenryu::io::h5_link_exists(file, path.c_str(), H5P_DEFAULT) > 0) {
    const hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    TENRYU_ASSERT(dset >= 0, "HistoryWriter H5Dopen2(matrix row) failed");
    write_units_attribute_if_missing(dset, units);

    const hid_t fspace = H5Dget_space(dset);
    TENRYU_ASSERT(fspace >= 0, "HistoryWriter H5Dget_space(matrix row) failed");
    hsize_t dims[2] = {0, 0};
    const int rank = H5Sget_simple_extent_dims(fspace, dims, nullptr);
    TENRYU_ASSERT(rank == 2, "HistoryWriter matrix dataset rank mismatch for path: " + path);
    TENRYU_ASSERT(dims[1] == n_cols,
                  "HistoryWriter matrix column mismatch for path: " + path);
    warn_h5_close_failure(H5Sclose(fspace), "H5Sclose",
                          "HistoryWriter::append_row_double_matrix(extent query)");

    const hsize_t new_dims[2] = {dims[0] + 1, dims[1]};
    TENRYU_ASSERT(H5Dset_extent(dset, new_dims) >= 0,
                  "HistoryWriter H5Dset_extent(matrix row) failed");

    const hid_t file_space = H5Dget_space(dset);
    TENRYU_ASSERT(file_space >= 0, "HistoryWriter H5Dget_space(matrix row post-extend) failed");
    const hsize_t start[2] = {dims[0], 0};
    const hsize_t count[2] = {1, dims[1]};
    TENRYU_ASSERT(H5Sselect_hyperslab(file_space,
                                      H5S_SELECT_SET,
                                      start,
                                      nullptr,
                                      count,
                                      nullptr) >= 0,
                  "HistoryWriter H5Sselect_hyperslab(matrix row) failed");
    const hid_t mem_space = H5Screate_simple(2, count, nullptr);
    TENRYU_ASSERT(mem_space >= 0, "HistoryWriter H5Screate_simple(matrix row) failed");
    TENRYU_ASSERT(H5Dwrite(dset,
                           H5T_NATIVE_DOUBLE,
                           mem_space,
                           file_space,
                           H5P_DEFAULT,
                           row_values.data()) >= 0,
                  "HistoryWriter H5Dwrite(matrix row append) failed");
    warn_h5_close_failure(H5Sclose(mem_space), "H5Sclose",
                          "HistoryWriter::append_row_double_matrix(mem space)");
    warn_h5_close_failure(H5Sclose(file_space), "H5Sclose",
                          "HistoryWriter::append_row_double_matrix(file space)");
    warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                          "HistoryWriter::append_row_double_matrix(dataset)");
    return;
  }

  const hsize_t dims[2] = {1, n_cols};
  const hsize_t max_dims[2] = {H5S_UNLIMITED, n_cols};
  const hsize_t chunk[2] = {256, n_cols};
  const hid_t space = H5Screate_simple(2, dims, max_dims);
  TENRYU_ASSERT(space >= 0, "HistoryWriter H5Screate_simple(matrix create) failed");
  const hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  TENRYU_ASSERT(dcpl >= 0, "HistoryWriter H5Pcreate(matrix create) failed");
  TENRYU_ASSERT(H5Pset_chunk(dcpl, 2, chunk) >= 0,
                "HistoryWriter H5Pset_chunk(matrix create) failed");
  const hid_t dset = H5Dcreate2(file,
                                path.c_str(),
                                H5T_NATIVE_DOUBLE,
                                space,
                                H5P_DEFAULT,
                                dcpl,
                                H5P_DEFAULT);
  TENRYU_ASSERT(dset >= 0, "HistoryWriter H5Dcreate2(matrix create) failed");
  write_units_attribute_if_missing(dset, units);
  TENRYU_ASSERT(H5Dwrite(dset,
                         H5T_NATIVE_DOUBLE,
                         H5S_ALL,
                         H5S_ALL,
                         H5P_DEFAULT,
                         row_values.data()) >= 0,
                "HistoryWriter H5Dwrite(matrix create) failed");
  warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                        "HistoryWriter::append_row_double_matrix(create dataset)");
  warn_h5_close_failure(H5Pclose(dcpl), "H5Pclose",
                        "HistoryWriter::append_row_double_matrix(create dcpl)");
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HistoryWriter::append_row_double_matrix(create space)");
}

void write_i32_vector_dataset_if_missing(const hid_t file,
                                         const std::string& path,
                                         const std::vector<int>& values) {
  if (values.empty()) {
    return;
  }
  ensure_parent_groups(file, path);
  std::vector<std::int32_t> packed(values.size(), 0);
  for (std::size_t i = 0; i < values.size(); ++i) {
    packed[i] = static_cast<std::int32_t>(values[i]);
  }

  if (tenryu::io::h5_link_exists(file, path.c_str(), H5P_DEFAULT) > 0) {
    const hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    TENRYU_ASSERT(dset >= 0, "HistoryWriter H5Dopen2(ell_values) failed");
    const hid_t space = H5Dget_space(dset);
    TENRYU_ASSERT(space >= 0, "HistoryWriter H5Dget_space(ell_values) failed");
    TENRYU_ASSERT(H5Sget_simple_extent_ndims(space) == 1,
                  "HistoryWriter ell_values rank mismatch");
    hsize_t dims[1] = {0};
    TENRYU_ASSERT(H5Sget_simple_extent_dims(space, dims, nullptr) == 1,
                  "HistoryWriter ell_values dims query failed");
    TENRYU_ASSERT(dims[0] == static_cast<hsize_t>(packed.size()),
                  "HistoryWriter ell_values size mismatch");
    std::vector<std::int32_t> existing(packed.size(), 0);
    TENRYU_ASSERT(H5Dread(dset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, existing.data()) >=
                      0,
                  "HistoryWriter H5Dread(ell_values) failed");
    TENRYU_ASSERT(existing == packed, "HistoryWriter ell_values content mismatch");
    warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                          "HistoryWriter::write_i32_vector_dataset_if_missing(space)");
    warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                          "HistoryWriter::write_i32_vector_dataset_if_missing(dataset)");
    return;
  }

  const hsize_t dims[1] = {static_cast<hsize_t>(packed.size())};
  const hid_t space = H5Screate_simple(1, dims, nullptr);
  TENRYU_ASSERT(space >= 0, "HistoryWriter H5Screate_simple(ell_values create) failed");
  const hid_t dset = H5Dcreate2(file,
                                path.c_str(),
                                H5T_NATIVE_INT32,
                                space,
                                H5P_DEFAULT,
                                H5P_DEFAULT,
                                H5P_DEFAULT);
  TENRYU_ASSERT(dset >= 0, "HistoryWriter H5Dcreate2(ell_values create) failed");
  TENRYU_ASSERT(H5Dwrite(dset,
                         H5T_NATIVE_INT32,
                         H5S_ALL,
                         H5S_ALL,
                         H5P_DEFAULT,
                         packed.data()) >= 0,
                "HistoryWriter H5Dwrite(ell_values create) failed");
  warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                        "HistoryWriter::write_i32_vector_dataset_if_missing(create dataset)");
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HistoryWriter::write_i32_vector_dataset_if_missing(create space)");
}

void write_double_vector_dataset_if_missing(const hid_t file,
                                            const std::string& path,
                                            const std::vector<double>& values,
                                            const char* units) {
  if (values.empty()) {
    return;
  }
  ensure_parent_groups(file, path);
  if (tenryu::io::h5_link_exists(file, path.c_str(), H5P_DEFAULT) > 0) {
    const hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    TENRYU_ASSERT(dset >= 0, "HistoryWriter H5Dopen2(double vector) failed");
    write_units_attribute_if_missing(dset, units);
    const hid_t space = H5Dget_space(dset);
    TENRYU_ASSERT(space >= 0, "HistoryWriter H5Dget_space(double vector) failed");
    TENRYU_ASSERT(H5Sget_simple_extent_ndims(space) == 1,
                  "HistoryWriter double vector rank mismatch");
    hsize_t dims[1] = {0};
    TENRYU_ASSERT(H5Sget_simple_extent_dims(space, dims, nullptr) == 1,
                  "HistoryWriter double vector dims query failed");
    TENRYU_ASSERT(dims[0] == static_cast<hsize_t>(values.size()),
                  "HistoryWriter double vector size mismatch");
    warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                          "HistoryWriter::write_double_vector_dataset_if_missing(space)");
    warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                          "HistoryWriter::write_double_vector_dataset_if_missing(dataset)");
    return;
  }

  const hsize_t dims[1] = {static_cast<hsize_t>(values.size())};
  const hid_t space = H5Screate_simple(1, dims, nullptr);
  TENRYU_ASSERT(space >= 0, "HistoryWriter H5Screate_simple(double vector create) failed");
  const hid_t dset = H5Dcreate2(file,
                                path.c_str(),
                                H5T_NATIVE_DOUBLE,
                                space,
                                H5P_DEFAULT,
                                H5P_DEFAULT,
                                H5P_DEFAULT);
  TENRYU_ASSERT(dset >= 0, "HistoryWriter H5Dcreate2(double vector create) failed");
  write_units_attribute_if_missing(dset, units);
  TENRYU_ASSERT(H5Dwrite(dset,
                         H5T_NATIVE_DOUBLE,
                         H5S_ALL,
                         H5S_ALL,
                         H5P_DEFAULT,
                         values.data()) >= 0,
                "HistoryWriter H5Dwrite(double vector create) failed");
  warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                        "HistoryWriter::write_double_vector_dataset_if_missing(create dataset)");
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HistoryWriter::write_double_vector_dataset_if_missing(create space)");
}

void write_dt_breakdown_history(const hid_t file,
                                const DtBreakdownHistoryRecord& record) {
  if (!record.valid) {
    return;
  }
  constexpr const char* base = "/diagnostics/dt_breakdown_history/";
  append_scalar_i64(file, std::string(base) + "cycle", record.cycle, "count");
  append_scalar_double(file, std::string(base) + "t_s", record.t_s, "s");
  append_scalar_double(file, std::string(base) + "dt_chosen", record.dt_chosen, "s");
  append_scalar_fixed_string(file,
                             std::string(base) + "dt_winner",
                             record.dt_winner);
  append_scalar_i32(file,
                    std::string(base) + "dt_winner_code",
                    static_cast<std::int32_t>(record.dt_winner_code),
                    "enum");
  append_scalar_double(file, std::string(base) + "dt_hydro", record.dt_hydro, "s");
  append_scalar_double(file, std::string(base) + "dt_rad", record.dt_rad, "s");
  append_scalar_double(file, std::string(base) + "dt_cond", record.dt_cond, "s");
  append_scalar_double(
      file, std::string(base) + "dt_post_shock", record.dt_post_shock, "s");
  append_scalar_double(
      file, std::string(base) + "dt_growth", record.dt_growth, "s");
  append_scalar_double(file, std::string(base) + "dt_max", record.dt_max, "s");
  append_scalar_double(
      file, std::string(base) + "dt_output", record.dt_output, "s");
  append_scalar_double(
      file, std::string(base) + "dt_remaining", record.dt_remaining, "s");
  append_scalar_double(file,
                       std::string(base) + "dt_hydro_acoustic",
                       record.dt_hydro_acoustic,
                       "s");
  append_scalar_double(file,
                       std::string(base) + "dt_hydro_axis_margin",
                       record.dt_hydro_axis_margin,
                       "s");
  append_scalar_double(file,
                       std::string(base) + "dt_hydro_volume_rate",
                       record.dt_hydro_volume_rate,
                       "s");
  const std::string center_base = std::string(base) + "tri_fan_center_cfl/";
  append_scalar_double(file,
                       center_base + "dt",
                       record.dt_hydro_tri_fan_center,
                       "s");
  append_scalar_i32(file,
                    center_base + "binding_cell_id",
                    static_cast<std::int32_t>(record.tri_fan_center_cfl_cell_id),
                    "index");
  append_scalar_i32(file,
                    center_base + "binding_i",
                    static_cast<std::int32_t>(record.tri_fan_center_cfl_i),
                    "index");
  append_scalar_i32(file,
                    center_base + "binding_j",
                    static_cast<std::int32_t>(record.tri_fan_center_cfl_j),
                    "index");
  append_scalar_double(file,
                       center_base + "h",
                       record.tri_fan_center_cfl_h,
                       "cm");
  append_scalar_double(file,
                       center_base + "c_eff",
                       record.tri_fan_center_cfl_c_eff,
                       "cm/s");
  append_scalar_double(file,
                       center_base + "q_over_p",
                       record.tri_fan_center_cfl_q_over_p,
                       "dimensionless");
  append_scalar_double(
      file,
      center_base + "dt_global_over_dt_center",
      record.tri_fan_center_cfl_dt_global_over_dt_center,
      "dimensionless");
}

void write_plasma_viscosity_history(const hid_t file,
                                    const PlasmaViscosityHistoryRecord& record) {
  if (!record.valid) {
    return;
  }
  constexpr const char* base = "/diagnostics/plasma_viscosity_history/";
  append_scalar_i64(file, std::string(base) + "cycle", record.cycle, "count");
  append_scalar_double(file, std::string(base) + "t_s", record.t_s, "s");
  append_scalar_double(
      file, std::string(base) + "eta_i_max", record.eta_i_max, "poise");
  append_scalar_double(
      file, std::string(base) + "eta_e_max", record.eta_e_max, "poise");
  append_scalar_double(
      file, std::string(base) + "eta_eff_max", record.eta_eff_max, "poise");
  append_scalar_double(
      file, std::string(base) + "ratio_min", record.ratio_min, "ratio");
  append_scalar_double(file,
                       std::string(base) + "ratio_geomean_masswt",
                       record.ratio_geomean_masswt,
                       "ratio");
  append_scalar_double(
      file, std::string(base) + "ratio_max", record.ratio_max, "ratio");
  append_scalar_i32(file,
                    std::string(base) + "n_cells_e_dom",
                    static_cast<std::int32_t>(record.n_cells_e_dom),
                    "count");
  append_scalar_i32(file,
                    std::string(base) + "n_cells_i_dom",
                    static_cast<std::int32_t>(record.n_cells_i_dom),
                    "count");
  append_scalar_i32(file,
                    std::string(base) + "n_cells_mixed",
                    static_cast<std::int32_t>(record.n_cells_mixed),
                    "count");
  append_scalar_i32(file,
                    std::string(base) + "n_cells_active",
                    static_cast<std::int32_t>(record.n_cells_active),
                    "count");
  append_scalar_double(file,
                       std::string(base) + "heat_rate_i_tot",
                       record.heat_rate_i_tot,
                       "erg/s");
  append_scalar_double(file,
                       std::string(base) + "heat_rate_e_tot",
                       record.heat_rate_e_tot,
                       "erg/s");
}

void write_radial_fourier_audit_history(
    const hid_t file,
    const RadialFourierAuditRecord& record) {
  if (!record.valid) {
    return;
  }
  constexpr const char* base = "/diagnostics/radial_fourier_audit/v1/";
  for (int field = 0; field < kRadialFourierFieldCount; ++field) {
    const auto& maximum = record.fields[static_cast<std::size_t>(field)];
    const auto m = static_cast<std::uint16_t>(
        std::min(std::max(maximum.m_max, 0),
                 static_cast<int>(std::numeric_limits<std::uint16_t>::max())));
    const auto j = static_cast<std::uint16_t>(
        std::min(std::max(maximum.j_max, 0),
                 static_cast<int>(std::numeric_limits<std::uint16_t>::max())));
    append_scalar_u64(file, std::string(base) + "cycle", record.cycle, "count");
    append_scalar_double(file, std::string(base) + "t_s", record.t_s, "s");
    append_scalar_u8(file, std::string(base) + "stage_id", record.stage_id, "enum");
    append_scalar_u8(file,
                     std::string(base) + "stage_phase",
                     record.stage_phase,
                     "enum");
    append_scalar_u8(file,
                     std::string(base) + "field_id",
                     static_cast<std::uint8_t>(field),
                     "enum");
    append_scalar_double(file,
                         std::string(base) + "A_max",
                         maximum.A_max,
                         "dimensionless");
    append_scalar_u16(file, std::string(base) + "m_max", m, "index");
    append_scalar_u16(file, std::string(base) + "j_max", j, "index");
  }
}

void write_radial_fourier_complex_audit_history(
    const hid_t file,
    const RadialFourierComplexAuditRecord& record) {
#if !TENRYU_RFA_V2_WRITES_HISTORY
  (void)file;
  (void)record;
  return;
#else
  if (!record.valid) {
    return;
  }
  constexpr const char* base = "/diagnostics/radial_fourier_audit_v2/v1/";
  for (const auto& coeff : record.coeffs) {
    const auto m = static_cast<std::uint16_t>(
        std::min(std::max(coeff.m, 0),
                 static_cast<int>(std::numeric_limits<std::uint16_t>::max())));
    const auto j = static_cast<std::uint16_t>(
        std::min(std::max(coeff.j, 0),
                 static_cast<int>(std::numeric_limits<std::uint16_t>::max())));
    append_scalar_u64(file, std::string(base) + "cycle", record.cycle, "count");
    append_scalar_double(file, std::string(base) + "t_s", record.t_s, "s");
    append_scalar_double(file, std::string(base) + "dt_cycle", record.dt_cycle, "s");
    append_scalar_u8(file, std::string(base) + "stage_id", record.stage_id, "enum");
    append_scalar_u8(file,
                     std::string(base) + "stage_phase",
                     record.stage_phase,
                     "enum");
    append_scalar_u8(file, std::string(base) + "field_id", coeff.field_id, "enum");
    append_scalar_u16(file, std::string(base) + "m", m, "index");
    append_scalar_u16(file, std::string(base) + "j", j, "index");
    append_scalar_double(file, std::string(base) + "mean_unw", coeff.mean_unw, "");
    append_scalar_double(file, std::string(base) + "cre_unw", coeff.cre_unw, "");
    append_scalar_double(file, std::string(base) + "cim_unw", coeff.cim_unw, "");
    append_scalar_double(file, std::string(base) + "amp_unw", coeff.amp_unw, "");
    append_scalar_double(file, std::string(base) + "phase_unw", coeff.phase_unw, "rad");
    append_scalar_double(file, std::string(base) + "mean_vol", coeff.mean_vol, "");
    append_scalar_double(file, std::string(base) + "cre_vol", coeff.cre_vol, "");
    append_scalar_double(file, std::string(base) + "cim_vol", coeff.cim_vol, "");
    append_scalar_double(file, std::string(base) + "amp_vol", coeff.amp_vol, "");
    append_scalar_double(file, std::string(base) + "phase_vol", coeff.phase_vol, "rad");
    append_scalar_double(file, std::string(base) + "q_min_j", coeff.q_min_j, "");
    append_scalar_double(file, std::string(base) + "q_max_j", coeff.q_max_j, "");
    append_scalar_double(file, std::string(base) + "wsum_vol", coeff.wsum_vol, "cm3");
  }
#endif
}

void write_fld_substage_audit_history(
    const hid_t file,
    const std::vector<FldSubstageAuditRecord>& records) {
  constexpr const char* base = "/diagnostics/fld_substage_audit/v1/";
  for (const auto& record : records) {
    if (!record.valid) {
      continue;
    }
    const auto m = static_cast<std::uint16_t>(
        std::min(std::max(record.m, 0),
                 static_cast<int>(std::numeric_limits<std::uint16_t>::max())));
    const auto j = static_cast<std::uint16_t>(
        std::min(std::max(record.j, 0),
                 static_cast<int>(std::numeric_limits<std::uint16_t>::max())));
    append_scalar_u64(file, std::string(base) + "cycle", record.cycle, "count");
    append_scalar_double(file, std::string(base) + "t_s", record.t_s, "s");
    append_scalar_double(file,
                         std::string(base) + "dt_cycle",
                         record.dt_cycle,
                         "s");
    append_scalar_u8(file,
                     std::string(base) + "substage_id",
                     record.substage_id,
                     "enum");
    append_scalar_u8(file, std::string(base) + "field_id", record.field_id, "enum");
    append_scalar_u8(file,
                     std::string(base) + "normalization_kind",
                     record.normalization_kind,
                     "enum");
    append_scalar_u16(file, std::string(base) + "m", m, "index");
    append_scalar_u16(file, std::string(base) + "j", j, "index");
    append_scalar_i32(file,
                      std::string(base) + "group",
                      static_cast<std::int32_t>(record.group),
                      "index");
    append_scalar_i32(file,
                      std::string(base) + "outer_iter",
                      static_cast<std::int32_t>(record.outer_iter),
                      "count");
    append_scalar_i32(file,
                      std::string(base) + "nr",
                      static_cast<std::int32_t>(record.nr),
                      "count");
    append_scalar_i32(file,
                      std::string(base) + "nz",
                      static_cast<std::int32_t>(record.nz),
                      "count");
    append_scalar_double(file, std::string(base) + "Re", record.cre, "");
    append_scalar_double(file, std::string(base) + "Im", record.cim, "");
    append_scalar_double(file, std::string(base) + "amplitude", record.amplitude, "");
    append_scalar_double(file, std::string(base) + "phase", record.phase, "rad");
    append_scalar_double(file, std::string(base) + "mean", record.mean, "");
    append_scalar_double(file, std::string(base) + "min", record.q_min_j, "");
    append_scalar_double(file, std::string(base) + "max", record.q_max_j, "");
    append_scalar_double(file,
                         std::string(base) + "normalization",
                         record.normalization,
                         "count");
    append_scalar_double(file,
                         std::string(base) + "solver_residual_l2_rel",
                         record.solver_residual_l2_rel,
                         "dimensionless");
    append_scalar_double(file,
                         std::string(base) + "solver_residual_max",
                         record.solver_residual_max,
                         "");
  }
}

void write_cfl_winner_history(const hid_t file,
                              const DtBreakdownHistoryRecord& record) {
  if (!record.valid) {
    return;
  }
  constexpr const char* base = "/diagnostics/cfl_winner/";
  append_scalar_i64(file, std::string(base) + "cycle", record.cycle, "count");
  append_scalar_double(file, std::string(base) + "t_s", record.t_s, "s");
  append_scalar_i32(file,
                    std::string(base) + "cell_id",
                    static_cast<std::int32_t>(record.hydro_winner_cell_id),
                    "index");
  append_scalar_i32(file,
                    std::string(base) + "i",
                    static_cast<std::int32_t>(record.hydro_winner_i),
                    "index");
  append_scalar_i32(file,
                    std::string(base) + "j",
                    static_cast<std::int32_t>(record.hydro_winner_j),
                    "index");
  append_scalar_double(file,
                       std::string(base) + "dt_at_cell",
                       record.hydro_winner_dt_at_cell,
                       "s");
  append_scalar_double(file,
                       std::string(base) + "dl_at_cell",
                       record.hydro_winner_dl_at_cell,
                       "cm");
  append_scalar_double(file,
                       std::string(base) + "cs_at_cell",
                       record.hydro_winner_cs_at_cell,
                       "cm/s");
  append_scalar_double(file,
                       std::string(base) + "rho_at_cell",
                       record.hydro_winner_rho_at_cell,
                       "g/cm3");
  append_scalar_double(file,
                       std::string(base) + "u_z_at_cell",
                       record.hydro_winner_u_z_at_cell,
                       "cm/s");
}

double host_cell_center_node_average(const std::vector<double>& node_values,
                                     const int i,
                                     const int j,
                                     const int nr,
                                     const int nz,
                                     const double fallback) {
  const std::size_t expected =
      static_cast<std::size_t>(nr + 1) * static_cast<std::size_t>(nz + 1);
  if (node_values.size() != expected || i < 0 || j < 0 || i >= nr || j >= nz) {
    return fallback;
  }
  const auto node = [nz](const int ii, const int jj) {
    return static_cast<std::size_t>(ii * (nz + 1) + jj);
  };
  return 0.25 * (node_values[node(i, j)] + node_values[node(i + 1, j)] +
                 node_values[node(i, j + 1)] +
                 node_values[node(i + 1, j + 1)]);
}

bool cfl_winner_in_outer_top_corner_halo(const core::State& state,
                                         const DtBreakdownHistoryRecord& record) {
  if (!record.valid || state.mesh.dim != 2) {
    return false;
  }
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  if (nr <= 0 || nz <= 0) {
    return false;
  }
  const int i = record.hydro_winner_i;
  const int j = record.hydro_winner_j;
  if (i < 0 || j < 0 || i >= nr || j >= nz) {
    return false;
  }
  constexpr int kHalo = 3;
  return i >= std::max(0, nr - kHalo) && j >= std::max(0, nz - kHalo);
}

double host_field_value_or_nan(const std::vector<double>& values, const int c) {
  if (c < 0 || static_cast<std::size_t>(c) >= values.size()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return values[static_cast<std::size_t>(c)];
}

double host_radiation_total_or_nan(const core::State& state,
                                   const int c,
                                   const int n_cells) {
  if (c < 0 || n_cells <= 0 || state.rad_E.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (state.rad_E.size() == static_cast<std::size_t>(n_cells)) {
    const auto rad = copy_field_to_host(state.rad_E);
    return host_field_value_or_nan(rad, c);
  }
  if ((state.rad_E.size() % static_cast<std::size_t>(n_cells)) != 0U) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const auto rad = copy_field_to_host(state.rad_E);
  const int n_groups =
      static_cast<int>(state.rad_E.size() / static_cast<std::size_t>(n_cells));
  double total = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    total += rad[static_cast<std::size_t>(c) +
                 static_cast<std::size_t>(g) * static_cast<std::size_t>(n_cells)];
  }
  return total;
}

#ifndef NDEBUG
bool corner_bc_audit_assert_enabled() {
  const char* value = std::getenv("TENRYU_CORNER_BC_AUDIT_ASSERT");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool corner_bc_audit_i1_like_case(const core::Config& cfg) {
  std::string name = cfg.main.name;
  for (char& ch : name) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return name.find("i1") != std::string::npos;
}
#endif

void append_scalar_double_compat(const hid_t file,
                                 const std::string& preferred_path,
                                 const std::string& legacy_path,
                                 const double value,
                                 const char* units) {
  const bool preferred_exists =
      tenryu::io::h5_link_exists(file, preferred_path.c_str(), H5P_DEFAULT) > 0;
  const bool legacy_exists =
      !legacy_path.empty() &&
      (tenryu::io::h5_link_exists(file, legacy_path.c_str(), H5P_DEFAULT) > 0);
  const std::string& selected = (legacy_exists && !preferred_exists) ? legacy_path : preferred_path;
  append_scalar_double(file, selected, value, units);
}

void append_scalar_i64_compat(const hid_t file,
                              const std::string& preferred_path,
                              const std::string& legacy_path,
                              const std::int64_t value,
                              const char* units) {
  const bool preferred_exists =
      tenryu::io::h5_link_exists(file, preferred_path.c_str(), H5P_DEFAULT) > 0;
  const bool legacy_exists =
      !legacy_path.empty() &&
      (tenryu::io::h5_link_exists(file, legacy_path.c_str(), H5P_DEFAULT) > 0);
  const std::string& selected = (legacy_exists && !preferred_exists) ? legacy_path : preferred_path;
  append_scalar_i64(file, selected, value, units);
}

std::optional<hsize_t> dataset_length_if_exists(const hid_t file, const std::string& path) {
  if (tenryu::io::h5_link_exists(file, path.c_str(), H5P_DEFAULT) <= 0) {
    return std::nullopt;
  }
  const hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  TENRYU_ASSERT(dset >= 0, "HistoryWriter H5Dopen2(length check) failed: " + path);
  const hid_t space = H5Dget_space(dset);
  TENRYU_ASSERT(space >= 0, "HistoryWriter H5Dget_space(length check) failed: " + path);
  const int rank = H5Sget_simple_extent_ndims(space);
  TENRYU_ASSERT(rank == 1, "HistoryWriter dataset rank must be 1 for length check: " + path);
  hsize_t dims[1] = {0};
  TENRYU_ASSERT(H5Sget_simple_extent_dims(space, dims, nullptr) == 1,
                "HistoryWriter H5Sget_simple_extent_dims failed: " + path);
  warn_h5_close_failure(H5Sclose(space), "H5Sclose",
                        "HistoryWriter::dataset_length_if_exists(space)");
  warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                        "HistoryWriter::dataset_length_if_exists(dataset)");
  return dims[0];
}

std::optional<double> last_scalar_double_if_exists(const hid_t file,
                                                   const std::string& path) {
  const auto len = dataset_length_if_exists(file, path);
  if (!len.has_value() || *len == 0) {
    return std::nullopt;
  }

  const hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  TENRYU_ASSERT(dset >= 0, "HistoryWriter H5Dopen2(last double) failed: " + path);
  const hid_t file_space = H5Dget_space(dset);
  TENRYU_ASSERT(file_space >= 0, "HistoryWriter H5Dget_space(last double) failed: " + path);
  const hsize_t start[1] = {*len - 1};
  const hsize_t count[1] = {1};
  TENRYU_ASSERT(H5Sselect_hyperslab(file_space,
                                    H5S_SELECT_SET,
                                    start,
                                    nullptr,
                                    count,
                                    nullptr) >= 0,
                "HistoryWriter H5Sselect_hyperslab(last double) failed: " + path);
  const hid_t mem_space = H5Screate_simple(1, count, nullptr);
  TENRYU_ASSERT(mem_space >= 0, "HistoryWriter H5Screate_simple(last double) failed: " + path);
  double value = 0.0;
  TENRYU_ASSERT(H5Dread(dset,
                        H5T_NATIVE_DOUBLE,
                        mem_space,
                        file_space,
                        H5P_DEFAULT,
                        &value) >= 0,
                "HistoryWriter H5Dread(last double) failed: " + path);
  warn_h5_close_failure(H5Sclose(mem_space), "H5Sclose",
                        "HistoryWriter::last_scalar_double_if_exists(mem space)");
  warn_h5_close_failure(H5Sclose(file_space), "H5Sclose",
                        "HistoryWriter::last_scalar_double_if_exists(file space)");
  warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                        "HistoryWriter::last_scalar_double_if_exists(dataset)");
  return value;
}

std::optional<double> first_scalar_double_if_exists(const hid_t file,
                                                    const std::string& path) {
  const auto len = dataset_length_if_exists(file, path);
  if (!len.has_value() || *len == 0) {
    return std::nullopt;
  }

  const hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  TENRYU_ASSERT(dset >= 0, "HistoryWriter H5Dopen2(first double) failed: " + path);
  const hid_t file_space = H5Dget_space(dset);
  TENRYU_ASSERT(file_space >= 0, "HistoryWriter H5Dget_space(first double) failed: " + path);
  const hsize_t start[1] = {0};
  const hsize_t count[1] = {1};
  TENRYU_ASSERT(H5Sselect_hyperslab(file_space,
                                    H5S_SELECT_SET,
                                    start,
                                    nullptr,
                                    count,
                                    nullptr) >= 0,
                "HistoryWriter H5Sselect_hyperslab(first double) failed: " + path);
  const hid_t mem_space = H5Screate_simple(1, count, nullptr);
  TENRYU_ASSERT(mem_space >= 0, "HistoryWriter H5Screate_simple(first double) failed: " + path);
  double value = 0.0;
  TENRYU_ASSERT(H5Dread(dset,
                        H5T_NATIVE_DOUBLE,
                        mem_space,
                        file_space,
                        H5P_DEFAULT,
                        &value) >= 0,
                "HistoryWriter H5Dread(first double) failed: " + path);
  warn_h5_close_failure(H5Sclose(mem_space), "H5Sclose",
                        "HistoryWriter::first_scalar_double_if_exists(mem space)");
  warn_h5_close_failure(H5Sclose(file_space), "H5Sclose",
                        "HistoryWriter::first_scalar_double_if_exists(file space)");
  warn_h5_close_failure(H5Dclose(dset), "H5Dclose",
                        "HistoryWriter::first_scalar_double_if_exists(dataset)");
  return value;
}

std::string sanitize_operator_name(std::string name) {
  for (char& ch : name) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (!std::isalnum(uch) && ch != '_') {
      ch = '_';
    }
  }
  if (name.empty()) {
    name = "unknown";
  }
  return name;
}

void write_icf_shell_history(const hid_t file,
                             const IcfShellDiagnostics& icf,
                             const double time,
                             const int step) {
  if (!icf.valid) {
    return;
  }
  const std::string base = "/diagnostics/icf/v1/";
  append_scalar_double(file, base + "time_s", time, "s");
  append_scalar_i32(file, base + "step", static_cast<std::int32_t>(step), "count");
  append_scalar_double(file, base + "shell_radius_cm", icf.R_shell_cm, "cm");
  append_scalar_double(file,
                       base + "shell_thickness_cm",
                       icf.shell_thickness_cm,
                       "cm");
  append_scalar_double(file, base + "IFAR", icf.IFAR, "dimensionless");
  append_scalar_double(file, base + "CR", icf.CR, "dimensionless");
  const hid_t group = open_or_create_group(file, "/diagnostics/icf/v1");
  write_double_attribute_if_missing(group, "R_initial_cm", icf.R_initial_cm);
  warn_h5_close_failure(H5Gclose(group), "H5Gclose",
                        "HistoryWriter::write_icf_shell_history(group)");
}

void write_hotspot_gas_history(const hid_t file,
                               const HotspotGasDiagnostics& hot,
                               const core::Config& cfg,
                               const double time,
                               const int step) {
  if (!hot.valid) {
    return;
  }
  const std::string base = "/diagnostics/hotspot_gas/v1/";
  append_scalar_double(file, base + "time_s", time, "s");
  append_scalar_i32(file, base + "step", static_cast<std::int32_t>(step), "count");
  append_scalar_double(file, base + "gas_mass_initial_g", hot.gas_mass_initial_g, "g");
  append_scalar_double(file, base + "gas_mass_g", hot.gas_mass_g, "g");
  append_scalar_double(
      file, base + "gas_mass_rel_drift", hot.gas_mass_rel_drift, "dimensionless");
  append_scalar_double(file, base + "R50_cm", hot.R50_cm, "cm");
  append_scalar_double(file, base + "R90_cm", hot.R90_cm, "cm");
  append_scalar_double(file, base + "R95_cm", hot.R95_cm, "cm");
  append_scalar_double(file, base + "R99_cm", hot.R99_cm, "cm");
  append_scalar_double(file, base + "CR50", hot.CR50, "dimensionless");
  append_scalar_double(file, base + "CR90", hot.CR90, "dimensionless");
  append_scalar_double(file, base + "CR95", hot.CR95, "dimensionless");
  append_scalar_double(file, base + "CR99", hot.CR99, "dimensionless");
  append_scalar_double(file, base + "C_R50_norm", hot.C_R50_norm, "dimensionless");
  append_scalar_double(file, base + "Rrms_cm", hot.Rrms_cm, "cm");
  append_scalar_double(file, base + "CRrms", hot.CRrms, "dimensionless");
  append_scalar_double(file, base + "rho_bar_volume_gcc", hot.rho_bar_volume_gcc, "g/cm3");
  append_scalar_double(file, base + "rho_bar_initial", hot.rho_bar_initial_gcc, "g/cm3");
  append_scalar_double(file, base + "CR_V", hot.CR_V, "dimensionless");
  if (hot.macro_core_valid) {
    append_scalar_double(file, base + "M_C_g", hot.macro_core_mass_g, "g");
    append_scalar_double(
        file, base + "M_Y_C_g", hot.macro_core_tracer_mass_g, "g");
    append_scalar_double(
        file, base + "V_C_cm3", hot.macro_core_volume_cm3, "cm3");
    append_scalar_double(file,
                         base + "V_C_initial_cm3",
                         hot.macro_core_volume_initial_cm3,
                         "cm3");
    append_scalar_double(
        file, base + "rho_C_gcc", hot.macro_core_rho_gcc, "g/cm3");
    append_scalar_double(file,
                         base + "rho_C_initial_gcc",
                         hot.macro_core_rho_initial_gcc,
                         "g/cm3");
    append_scalar_double(
        file, base + "CR_V_macro", hot.CR_V_macro, "dimensionless");
    append_scalar_double(file,
                         base + "fgas_C",
                         hot.macro_core_gas_energy_frac,
                         "dimensionless");
  }
  append_scalar_double(file, base + "rho_mean_mass_gcc", hot.rho_mean_mass_gcc, "g/cm3");
  append_scalar_double(file, base + "rho50_gcc", hot.rho50_gcc, "g/cm3");
  append_scalar_double(file, base + "CR_rho50", hot.CR_rho50, "dimensionless");
  append_scalar_double(file, base + "rho90_gcc", hot.rho90_gcc, "g/cm3");
  append_scalar_double(file, base + "rho95_gcc", hot.rho95_gcc, "g/cm3");
  append_scalar_double(file, base + "rho99_gcc", hot.rho99_gcc, "g/cm3");
  append_scalar_double(file, base + "p_mean_dyn_cm2", hot.p_mean_dyn_cm2, "dyn/cm2");
  append_scalar_double(file, base + "p50_dyn_cm2", hot.p50_dyn_cm2, "dyn/cm2");
  append_scalar_double(file, base + "p90_dyn_cm2", hot.p90_dyn_cm2, "dyn/cm2");
  append_scalar_double(file, base + "p95_dyn_cm2", hot.p95_dyn_cm2, "dyn/cm2");
  append_scalar_double(file, base + "p99_dyn_cm2", hot.p99_dyn_cm2, "dyn/cm2");
  append_scalar_double(file, base + "hotspot_Te_mean_eV", hot.Te_mean_eV, "eV");
  append_scalar_double(file, base + "hotspot_Te_p10_eV", hot.Te_p10_eV, "eV");
  append_scalar_double(file, base + "hotspot_Te_p50_eV", hot.Te_p50_eV, "eV");
  append_scalar_double(file, base + "hotspot_Te_p90_eV", hot.Te_p90_eV, "eV");
  append_scalar_i32(file,
                    base + "hotspot_Ti_valid",
                    hot.Ti_valid ? 1 : 0,
                    "dimensionless");
  if (hot.Ti_valid) {
    append_scalar_double(file, base + "hotspot_Ti_mean_eV", hot.Ti_mean_eV, "eV");
    append_scalar_double(file, base + "hotspot_Ti_p10_eV", hot.Ti_p10_eV, "eV");
    append_scalar_double(file, base + "hotspot_Ti_p50_eV", hot.Ti_p50_eV, "eV");
    append_scalar_double(file, base + "hotspot_Ti_p90_eV", hot.Ti_p90_eV, "eV");
  }
  append_scalar_double(file,
                       base + "hotspot_energy_internal_erg",
                       hot.hotspot_internal_energy_erg,
                       "erg");
  append_scalar_double(file,
                       base + "hotspot_energy_kinetic_erg",
                       hot.hotspot_kinetic_energy_erg,
                       "erg");
  append_scalar_double(file,
                       base + "hotspot_energy_total_erg",
                       hot.hotspot_total_energy_erg,
                       "erg");
  append_scalar_double(file,
                       base + "hotspot_energy_internal_initial_erg",
                       hot.hotspot_internal_energy_initial_erg,
                       "erg");
  append_scalar_double(file,
                       base + "hotspot_energy_kinetic_initial_erg",
                       hot.hotspot_kinetic_energy_initial_erg,
                       "erg");
  append_scalar_double(file,
                       base + "hotspot_energy_total_initial_erg",
                       hot.hotspot_total_energy_initial_erg,
                       "erg");
  append_scalar_double(file,
                       base + "hotspot_work_proxy_internal_erg",
                       hot.hotspot_work_proxy_internal_erg,
                       "erg");
  append_scalar_double(file,
                       base + "hotspot_work_proxy_kinetic_erg",
                       hot.hotspot_work_proxy_kinetic_erg,
                       "erg");
  append_scalar_double(file,
                       base + "hotspot_work_proxy_total_erg",
                       hot.hotspot_work_proxy_total_erg,
                       "erg");
  append_scalar_double(file, base + "K_mean", hot.K_mean, "cgs");
  append_scalar_double(file, base + "K50", hot.K50, "cgs");
  append_scalar_double(file, base + "K90", hot.K90, "cgs");
  append_scalar_double(file, base + "K95", hot.K95, "cgs");
  append_scalar_double(file, base + "K99", hot.K99, "cgs");
  const hid_t group = open_or_create_group(file, "/diagnostics/hotspot_gas/v1");
  write_double_attribute_if_missing(group, "R_g_cm", hot.R_g_cm);
  write_string_attribute_if_missing(
      group,
      "hotspot_work_definition",
      "proxy_total_energy_change=sum(Y_g*m*(ee+ei+0.5*|u_cell|^2))-initial; "
      "not a tracer-boundary pressure-work integral");
  warn_h5_close_failure(H5Gclose(group), "H5Gclose",
                        "HistoryWriter::write_hotspot_gas_history(group)");

  std::ostringstream msg;
  msg << "[hotspot_gas] step=" << step
      << " t=" << time
      << " CR50=" << hot.CR50
      << " CR95=" << hot.CR95
      << " CRrms=" << hot.CRrms
      << " CR_V=" << hot.CR_V
      << " C_R50_norm=" << hot.C_R50_norm
      << " CR_rho50=" << hot.CR_rho50
      << " rho_bar=" << hot.rho_bar_volume_gcc
      << " Te_mean=" << hot.Te_mean_eV
      << " W_proxy=" << hot.hotspot_work_proxy_total_erg
      << " K_mean=" << hot.K_mean;
  if (hot.macro_core_valid) {
    msg << " rho_C=" << hot.macro_core_rho_gcc
        << " CR_V_macro=" << hot.CR_V_macro
        << " fgas_C=" << hot.macro_core_gas_energy_frac;
  }
  msg << " M_drift=" << hot.gas_mass_rel_drift;
  core::log_info(msg.str());
  const double warn_rel =
      cfg.numerics.diagnostics.hotspot_gas.mass_drift_warn_rel;
  if (warn_rel >= 0.0 && std::abs(hot.gas_mass_rel_drift) > warn_rel) {
    core::log_warning("[hotspot_gas] tracer mass drift exceeded threshold: rel=" +
                      std::to_string(hot.gas_mass_rel_drift) +
                      " threshold=" + std::to_string(warn_rel));
  }
}

void write_operator_energy_residuals_history(
    const hid_t file,
    const std::vector<OperatorResidualEntry>& entries,
    const double time,
    const int step) {
  if (entries.empty()) {
    return;
  }
  const std::string base = "/diagnostics/conservation/v1/";
  append_scalar_double(file, base + "time_s", time, "s");
  append_scalar_i32(file, base + "step", static_cast<std::int32_t>(step), "count");
  for (const auto& entry : entries) {
    if (entry.operator_name.empty()) {
      continue;
    }
    const std::string op = sanitize_operator_name(entry.operator_name);
    const double residual =
        entry.valid ? entry.residual : std::numeric_limits<double>::quiet_NaN();
    append_scalar_double(file, base + "eps_E_" + op, residual, "dimensionless");
    append_scalar_double(file, base + "E_before_" + op, entry.E_before, "erg");
    append_scalar_double(file, base + "E_after_" + op, entry.E_after, "erg");
    append_scalar_double(
        file, base + "delta_E_ext_" + op, entry.delta_E_ext, "erg");
  }
}

void assert_history_dataset_lengths_consistent(const hid_t file) {
  const auto t_len = dataset_length_if_exists(file, "t");
  if (!t_len.has_value()) {
    return;
  }
  constexpr std::array<const char*, 7> kCorePaths = {"cycle",
                                                      "dt",
                                                      "energy/E_total",
                                                      "energy/dE_total",
                                                      "energy/conservation_error",
                                                      "energy/epsilon_budget",
                                                      "energy/E_volume_in"};
  for (const char* path : kCorePaths) {
    const auto len = dataset_length_if_exists(file, path);
    if (!len.has_value()) {
      continue;
    }
    TENRYU_ASSERT(*len == *t_len,
                  "HistoryWriter dataset length mismatch for " + std::string(path) +
                      " (expected " + std::to_string(static_cast<unsigned long long>(*t_len)) +
                      ", got " + std::to_string(static_cast<unsigned long long>(*len)) + ")");
  }

  constexpr std::array<const char*, 23> kMcPaths = {"mc/n_total",
                                                     "mc/n_imc",
                                                     "mc/n_ddmc",
                                                     "mc/n_census",
                                                     "mc/n_absorbed",
                                                     "mc/n_escaped",
                                                     "mc/n_leaked",
                                                     "mc/ddmc_fraction",
                                                     "mc/weight_min",
                                                     "mc/weight_mean",
                                                     "mc/weight_max",
                                                     "mc/overshoot_count",
                                                     "mc/overshoot_max",
                                                     "mc/ddmc_mode_count",
                                                     "mc/imc_mode_count",
                                                     "mc/mmatrix_violations",
                                                     "mc/mmatrix_fallback_count",
                                                     "mc/omega_below_threshold",
                                                     "mc/interface_transitions",
                                                     "mc/interface_reflections",
                                                     "mc/conversion_prob_violations",
                                                     "mc/ddmc_to_imc_conversions",
                                                     "mc/rad_momentum_deposition"};
  static bool warned_migration = false;
  for (const char* path : kMcPaths) {
    const auto len = dataset_length_if_exists(file, path);
    if (!len.has_value()) {
      continue;
    }
    if (*len != *t_len && !warned_migration) {
      core::log_warning("HistoryWriter: mc/* dataset '" + std::string(path) +
                        "' length mismatch (expected " +
                        std::to_string(static_cast<unsigned long long>(*t_len)) + ", got " +
                        std::to_string(static_cast<unsigned long long>(*len)) +
                        "); history file migration in progress");
      warned_migration = true;
    }
  }

  constexpr std::array<const char*, 18> kDifferencePaths = {
      "difference/reference_valid",
      "difference/eligible_cells",
      "difference/active_cells",
      "difference/strong_cells",
      "difference/hybrid_suppressed_cells",
      "difference/W_min",
      "difference/W_mean",
      "difference/W_max",
      "difference/tau_min",
      "difference/tau_mean",
      "difference/tau_max",
      "difference/chi_mean",
      "difference/chi_max",
      "difference/reduced_flux_max",
      "difference/knudsen_max",
      "difference/front_grad_Te_max",
      "difference/front_grad_rho_max",
      "difference/E_ref_total"};
  static bool warned_difference_migration = false;
  for (const char* path : kDifferencePaths) {
    const auto len = dataset_length_if_exists(file, path);
    if (!len.has_value()) {
      continue;
    }
    if (*len != *t_len && !warned_difference_migration) {
      core::log_warning("HistoryWriter: difference/* dataset '" + std::string(path) +
                        "' length mismatch (expected " +
                        std::to_string(static_cast<unsigned long long>(*t_len)) + ", got " +
                        std::to_string(static_cast<unsigned long long>(*len)) +
                        "); history file migration in progress");
      warned_difference_migration = true;
    }
  }

  constexpr std::array<const char*, 19> kHoloPaths = {
      "holo/n_core_cells",
      "holo/n_entered",
      "holo/n_exited",
      "holo/n_hard_exited",
      "holo/n_island_rejected",
      "holo/tau_R_min",
      "holo/tau_R_max",
      "holo/reduced_flux_max",
      "holo/E_LO_total",
      "holo/E_LO_boundary_in",
      "holo/E_LO_boundary_out",
      "holo/matter_delta",
      "holo/source_balance_error",
      "holo/particle_net_source_core",
      "holo/lo_particle_source_mismatch",
      "holo/Prr_coverage",
      "holo/chi_min",
      "holo/chi_mean",
      "holo/chi_max"};
  static bool warned_holo_migration = false;
  for (const char* path : kHoloPaths) {
    const auto len = dataset_length_if_exists(file, path);
    if (!len.has_value()) {
      continue;
    }
    if (*len != *t_len && !warned_holo_migration) {
      core::log_warning("HistoryWriter: holo/* dataset '" + std::string(path) +
                        "' length mismatch (expected " +
                        std::to_string(static_cast<unsigned long long>(*t_len)) + ", got " +
                        std::to_string(static_cast<unsigned long long>(*len)) +
                        "); history file migration in progress");
      warned_holo_migration = true;
    }
  }
}

#endif

}  // namespace

#if TENRYU_ENABLE_HDF5
std::int64_t saturating_i64_from_u64(const std::uint64_t value) {
  constexpr std::uint64_t kI64Max =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (value > kI64Max) {
    return std::numeric_limits<std::int64_t>::max();
  }
  return static_cast<std::int64_t>(value);
}

std::optional<HistoryWriter::PerRowMassValues>
HistoryWriter::compute_per_row_mass_values(const core::State& state) const {
  if (state.mesh.dim != 2 || state.rho.empty() || state.vol.size() != state.rho.size()) {
    return std::nullopt;
  }
  constexpr int kEvery = 50;
  if (state.step <= 0 || (state.step != 1 && (state.step % kEvery) != 0)) {
    return std::nullopt;
  }
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  if (nr <= 0 || nz <= 0 ||
      static_cast<int>(state.rho.size()) != nr * nz) {
    return std::nullopt;
  }

  std::vector<double> rho = copy_field_to_host(state.rho);
  std::vector<double> vol = copy_field_to_host(state.vol);
  std::vector<double> cs;
  if (state.cs.size() == state.rho.size()) {
    cs = copy_field_to_host(state.cs);
  } else {
    cs.assign(state.rho.size(), 0.0);
  }

  PerRowMassValues out{};
  out.step = state.step;
  out.t = state.t;
  out.z_row_idx.assign(static_cast<std::size_t>(nz), 0);
  out.mass.assign(static_cast<std::size_t>(nz), 0.0);
  out.rho_max.assign(static_cast<std::size_t>(nz), 0.0);
  out.rho_axis.assign(static_cast<std::size_t>(nz), 0.0);
  out.rho_outer.assign(static_cast<std::size_t>(nz), 0.0);
  out.cs_max.assign(static_cast<std::size_t>(nz), 0.0);
  for (int j = 0; j < nz; ++j) {
    out.z_row_idx[static_cast<std::size_t>(j)] = j;
    double row_rho_max = 0.0;
    double row_cs_max = 0.0;
    for (int i = 0; i < nr; ++i) {
      const int c = i * nz + j;
      const std::size_t idx = static_cast<std::size_t>(c);
      const double rho_c = rho[idx];
      const double cs_c = cs[idx];
      out.mass[static_cast<std::size_t>(j)] += rho_c * vol[idx];
      if (i == 0) {
        out.rho_axis[static_cast<std::size_t>(j)] = rho_c;
      }
      if (i == nr - 1) {
        out.rho_outer[static_cast<std::size_t>(j)] = rho_c;
      }
      if (std::isfinite(rho_c)) {
        row_rho_max = std::max(row_rho_max, rho_c);
      }
      if (std::isfinite(cs_c)) {
        row_cs_max = std::max(row_cs_max, cs_c);
      }
    }
    out.rho_max[static_cast<std::size_t>(j)] = row_rho_max;
    out.cs_max[static_cast<std::size_t>(j)] = row_cs_max;
  }
  return out;
}

std::optional<HistoryWriter::CornerBcAuditValues>
HistoryWriter::compute_corner_bc_audit_values(
    const core::State& state,
    const DtBreakdownHistoryRecord& record) const {
  if (!cfl_winner_in_outer_top_corner_halo(state, record)) {
    return std::nullopt;
  }
  const auto& bc = cfg_.numerics.hydro.boundary_2d;
  if (!bc.z_top_cfg.is_state_supply()) {
    return std::nullopt;
  }
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int i = record.hydro_winner_i;
  const int j = record.hydro_winner_j;
  const int c = i * nz + j;
  const int n_cells = nr * nz;
  const double nan = std::numeric_limits<double>::quiet_NaN();

  const auto rho = copy_field_to_host(state.rho);
  const auto ee = copy_field_to_host(state.ee);
  const auto ei = copy_field_to_host(state.ei);
  const auto v_r = copy_field_to_host(state.v_r);
  const auto v_z = copy_field_to_host(state.v_z);

  CornerBcAuditValues out{};
  out.cycle = record.cycle;
  out.t_s = record.t_s;
  out.cell_id = record.hydro_winner_cell_id;
  out.i = i;
  out.j = j;
  out.interior_rho = host_field_value_or_nan(rho, c);
  out.interior_u_r = host_cell_center_node_average(v_r, i, j, nr, nz, 0.0);
  out.interior_u_z = host_cell_center_node_average(
      v_z, i, j, nr, nz, record.hydro_winner_u_z_at_cell);
  out.interior_e = host_field_value_or_nan(ee, c) + host_field_value_or_nan(ei, c);
  out.interior_E_r = host_radiation_total_or_nan(state, c, n_cells);

  out.q_visc_at_cell = nan;
  if (state.Qvisc.size() == static_cast<std::size_t>(n_cells)) {
    const auto q = copy_field_to_host(state.Qvisc);
    out.q_visc_at_cell = host_field_value_or_nan(q, c);
  }
  out.cs_at_cell = record.hydro_winner_cs_at_cell;
  if (state.cs.size() == static_cast<std::size_t>(n_cells)) {
    const auto cs = copy_field_to_host(state.cs);
    out.cs_at_cell = host_field_value_or_nan(cs, c);
  }

  out.r_outer_ghost_rho = nan;
  out.r_outer_ghost_u_r = nan;
  out.r_outer_ghost_u_z = nan;
  if (bc.r_outer == "reflect") {
    out.r_outer_ghost_rho = out.interior_rho;
    out.r_outer_ghost_u_r = -out.interior_u_r;
    out.r_outer_ghost_u_z = out.interior_u_z;
  }

  out.z_top_ghost_rho = bc.z_top_cfg.supply_rho_g_per_cc;
  out.z_top_ghost_u_r = 0.0;
  out.z_top_ghost_u_z = bc.z_top_cfg.supply_u_z_cm_per_s;
  const double z_bottom_ghost_u_z = bc.z_bottom_cfg.supply_u_z_cm_per_s;
  out.diagonal_corner_ghost_rho = out.z_top_ghost_rho;
  out.diagonal_corner_ghost_u_r = 0.0;
  out.diagonal_corner_ghost_u_z = out.z_top_ghost_u_z;
  out.dt_at_cell = record.hydro_winner_dt_at_cell;

#ifndef NDEBUG
  if (corner_bc_audit_assert_enabled() && corner_bc_audit_i1_like_case(cfg_)) {
    if (bc.z_bottom_cfg.is_state_supply()) {
      TENRYU_ASSERT(z_bottom_ghost_u_z > 0.0,
                    "corner_bc_audit: I1 z_bottom state_supply ghost u_z must be positive");
    }
    TENRYU_ASSERT(out.z_top_ghost_u_z > 0.0,
                  "corner_bc_audit: I1 z_top state_supply ghost u_z must be positive");
    TENRYU_ASSERT(out.diagonal_corner_ghost_u_z > 0.0,
                  "corner_bc_audit: I1 diagonal corner ghost u_z must be positive");
    TENRYU_ASSERT(out.r_outer_ghost_u_z == out.interior_u_z,
                  "corner_bc_audit: I1 r_outer reflect ghost must preserve u_z");
  }
#endif

  return out;
}

HistoryWriter::PlasmaHistoryDiagnostics
HistoryWriter::compute_plasma_history_diagnostics(const core::State& state) const {
  PlasmaHistoryDiagnostics out{};
  if (state.zbar.empty()) {
    return out;
  }
  const auto zbar = copy_field_to_host(state.zbar);
  const auto mass = copy_field_to_host(state.mass);
  const bool has_mass = (mass.size() == zbar.size());

  long double weighted_sum = 0.0L;
  long double weight_total = 0.0L;
  double zbar_max = 0.0;
  for (std::size_t c = 0; c < zbar.size(); ++c) {
    const double z = std::max(zbar[c], 0.0);
    const double w = has_mass ? std::max(mass[c], 0.0) : 1.0;
    weighted_sum += static_cast<long double>(w) * static_cast<long double>(z);
    weight_total += static_cast<long double>(w);
    zbar_max = std::max(zbar_max, z);
  }
  if (weight_total <= 0.0L) {
    return out;
  }
  out.valid = true;
  out.zbar_mean = static_cast<double>(weighted_sum / weight_total);
  out.zbar_max = zbar_max;
  return out;
}

HistoryWriter::ImplosionHistoryDiagnostics
HistoryWriter::compute_implosion_history_diagnostics(const core::State& state) const {
  ImplosionHistoryDiagnostics out{};
  if (state.rho.empty()) {
    return out;
  }

  const auto rho = copy_field_to_host(state.rho);
  const auto mass = copy_field_to_host(state.mass);
  const auto Te = copy_field_to_host(state.Te);
  const bool has_mass = (mass.size() == rho.size());
  const bool has_centroid_r = (state.mesh.cell_centroid_r.size() == rho.size());
  const bool has_centroid_z = (state.mesh.cell_centroid_z.size() == rho.size());

  out.rho_peak = *std::max_element(rho.begin(), rho.end());
  const double shell_threshold = 0.1 * out.rho_peak;
  double min_radius = std::numeric_limits<double>::infinity();
  long double weighted_radius_sum = 0.0L;
  long double weight_sum = 0.0L;
  for (std::size_t c = 0; c < rho.size(); ++c) {
    if (rho[c] < shell_threshold) {
      continue;
    }
    if (!has_centroid_r) {
      continue;
    }
    const double r = std::max(state.mesh.cell_centroid_r[c], 0.0);
    min_radius = std::min(min_radius, r);
    const double w = has_mass ? std::max(mass[c], 0.0) : 1.0;
    weighted_radius_sum += static_cast<long double>(w) * static_cast<long double>(r);
    weight_sum += static_cast<long double>(w);
  }
  if (weight_sum > 0.0L) {
    out.shell_radius_mean = static_cast<double>(weighted_radius_sum / weight_sum);
    out.shell_radius_min = std::isfinite(min_radius) ? min_radius : out.shell_radius_mean;
  }
  if (state.mesh.dim == 1) {
    out.shell_radius_min = out.shell_radius_mean;
  }

  if (Te.size() == rho.size() && has_centroid_r) {
    std::size_t center_cell = 0;
    double min_dist2 = std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < rho.size(); ++c) {
      const double r = state.mesh.cell_centroid_r[c];
      double dist2 = r * r;
      if (state.mesh.dim == 2 && has_centroid_z) {
        const double z = state.mesh.cell_centroid_z[c];
        dist2 += z * z;
      }
      if (dist2 < min_dist2) {
        min_dist2 = dist2;
        center_cell = c;
      }
    }
    out.center_temperature = Te[center_cell];
  }
  out.valid = true;
  return out;
}

HistoryWriter::AleProvenanceValues
HistoryWriter::build_ale_provenance_values(
    tenryu::coupling::ProfileObservability& obs,
    const std::int64_t step) const {
  AleProvenanceValues out{};
  out.forbidden_config_violations = obs.forbidden_config_violations;
  out.escape_valve_activations = obs.escape_valve_activations;
  out.class_c_runtime_fires = obs.class_c_runtime_fires;
  out.mesh_geometry_failures_observed = obs.mesh_geometry_failures_observed;
  out.public_baseline_terminal_failures = obs.public_baseline_terminal_failures;
  out.emergency_cell_deactivation_fired = obs.emergency_cell_deactivation_fired;
  out.last_failing_cell = obs.last_failing_cell;
  out.last_failing_i = obs.last_failing_i;
  out.last_failing_j = obs.last_failing_j;
  out.last_min_cell_vol = obs.last_min_cell_vol;
  out.last_min_corner_j = obs.last_min_corner_j;
  out.last_failure_kind = obs.last_failure_kind;
  if (ale_provenance_enabled_) {
    out.escape_valve_events = obs.escape_valve_events;
    obs.escape_valve_events.clear();
  }

  out.interface_cells_observed = obs.interface_cells_observed;
  out.interface_reconstruction_attempt_count =
      obs.interface_reconstruction_attempt_count;
  out.interface_reconstruction_success_count =
      obs.interface_reconstruction_success_count;
  out.plic_max_eta_E_observed = obs.plic_max_eta_E_observed;
  out.plic_max_volume_fraction_residual_observed =
      obs.plic_max_volume_fraction_residual_observed;
  out.plic_min_grad_F_observed = obs.plic_min_grad_F_observed;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out.class_d_runtime_fires_matrix[static_cast<std::size_t>(i)]
                                      [static_cast<std::size_t>(j)] =
          obs.class_d_runtime_fires_matrix[i][j];
    }
  }
  if (ale_provenance_enabled_ && cfg_.numerics.plic.enabled) {
    out.plic_events_emitted_count = obs.plic_events_emitted_count;
    const std::size_t start = obs.plic_events_emitted_count;
    const std::size_t end = obs.plic_events.size();
    if (start < end) {
      out.plic_events.assign(obs.plic_events.begin() + static_cast<std::ptrdiff_t>(start),
                             obs.plic_events.end());
      if (out.plic_events.size() > kPlicEventsPerStepLimit) {
        out.plic_events_summary_mode = true;
        for (const auto& evt : out.plic_events) {
          if (evt.case_id >= 1 && evt.case_id <= 3 && evt.severity <= 2) {
            ++out.plic_event_summary_counts[static_cast<std::size_t>(evt.case_id - 1)]
                                           [static_cast<std::size_t>(evt.severity)];
          }
        }
        std::cout << "[plic_events] event-rate limit exceeded ("
                  << out.plic_events.size() << " events in step " << step
                  << "); switched to summary mode" << std::endl;
      }
    }
    obs.plic_events_emitted_count = end;
  }

  out.mesh_quality_min_observed = obs.mesh_quality_min_observed;
  out.achieved_min_corner_j_rel = obs.achieved_min_corner_j_rel;
  out.achieved_min_gauss_j_rel = obs.achieved_min_gauss_j_rel;
  out.achieved_min_rz_volume_rel = obs.achieved_min_rz_volume_rel;
  out.achieved_min_edge_length_rel = obs.achieved_min_edge_length_rel;
  out.achieved_min_altitude_rel = obs.achieved_min_altitude_rel;
  out.achieved_max_condition_number = obs.achieved_max_condition_number;
  out.negative_rz_volume_count_total = obs.negative_rz_volume_count_total;
  return out;
}

HistoryWriter::PendingHistoryRecord HistoryWriter::build_pending_record(
    const core::State& state,
    const HistorySnapshot& snapshot) {
  PendingHistoryRecord rec{};
  rec.t = state.t;
  rec.dt = state.dt;
  rec.step = state.step;
  rec.mc_group_enabled = mc_group_enabled_;
  rec.mc_particle_counts_enabled = mc_particle_counts_enabled_;
  rec.mc_weight_stats_enabled = mc_weight_stats_enabled_;
  rec.mc_ddmc_fraction_enabled = mc_ddmc_fraction_enabled_;
  rec.phase_resolved_energy_enabled = phase_resolved_energy_enabled_;
  rec.ale_closure_audit_enabled = ale_closure_audit_enabled_;
  rec.icf_enabled = icf_enabled_;
  rec.ale_provenance_enabled = ale_provenance_enabled_;
  rec.hotspot_gas_enabled = hotspot_gas_enabled_;
  rec.mesh_quality_min_enabled = mesh_quality_min_enabled_;
  rec.conservation_enabled = conservation_enabled_;
  rec.dt_breakdown_history_enabled = dt_breakdown_history_enabled_;
  rec.center_perturbation_enabled = center_perturbation_enabled_;
  rec.E_laser_deposited = state.E_laser_deposited;
  rec.E_laser_escaped = state.E_laser_escaped;
  rec.E_laser_incident = state.E_laser_incident;
  rec.E_ra_deposited = state.E_ra_deposited;
  rec.E_cbet_iaw_step = state.E_cbet_iaw_step;
  rec.E_cbet_iaw = state.E_cbet_iaw;
  rec.c1_solver_steps_total = state.c1_solver_steps_total;
  rec.c1_solver_residual_last = state.c1_solver_residual_last;
  rec.c1_solver_residual_max = state.c1_solver_residual_max;
  rec.c1_solver_iter_last = state.c1_solver_iter_last;
  rec.c1_solver_iter_max = state.c1_solver_iter_max;
  rec.c1_solver_cond_number_last = state.c1_solver_cond_number_last;
  rec.c1_solver_cond_number_max = state.c1_solver_cond_number_max;
  rec.c1_bc_heat_flux_integrated = state.c1_bc_heat_flux_integrated;
  rec.burn_enabled_any = state.burn_enabled_any;
  rec.burn_diffusion_any = state.burn_diffusion_any;
  rec.burn_mc_any = state.burn_mc_any;
  rec.burn_released_step = state.burn_released_step;
  rec.burn_dep_e_step = state.burn_dep_e_step;
  rec.burn_dep_i_step = state.burn_dep_i_step;
  rec.burn_esc_charged_step = state.burn_esc_charged_step;
  rec.burn_esc_neutron_step = state.burn_esc_neutron_step;
  rec.burn_nh_dep_e_step = state.burn_nh_dep_e_step;
  rec.burn_nh_dep_i_step = state.burn_nh_dep_i_step;
  rec.burn_nh_degraded_step = state.burn_nh_degraded_step;
  rec.burn_nh_escaped_step = state.burn_nh_escaped_step;
  rec.E_burn_released = state.E_burn_released;
  rec.E_burn_dep_e = state.E_burn_dep_e;
  rec.E_burn_dep_i = state.E_burn_dep_i;
  rec.E_burn_esc_charged = state.E_burn_esc_charged;
  rec.E_burn_esc_neutron = state.E_burn_esc_neutron;
  rec.E_burn_inflight = state.E_burn_inflight;
  rec.N_burn_neutrons_dt = state.N_burn_neutrons_dt;
  rec.N_burn_neutrons_dd = state.N_burn_neutrons_dd;
  rec.burn_Ti_burn_dt_eV = state.burn_Ti_burn_dt_eV;
  rec.burn_Ti_burn_dd_eV = state.burn_Ti_burn_dd_eV;
  rec.burn_neutron_mean_shift_dt_keV = state.burn_neutron_mean_shift_dt_keV;
  rec.burn_neutron_sigma_thermal_dt_keV = state.burn_neutron_sigma_thermal_dt_keV;
  rec.burn_neutron_sigma_total_dt_keV = state.burn_neutron_sigma_total_dt_keV;
  rec.burn_neutron_mean_shift_dd_keV = state.burn_neutron_mean_shift_dd_keV;
  rec.burn_neutron_sigma_thermal_dd_keV = state.burn_neutron_sigma_thermal_dd_keV;
  rec.burn_neutron_sigma_total_dd_keV = state.burn_neutron_sigma_total_dd_keV;
  rec.burn_dt_limit_s = state.burn_dt_limit_s;
  rec.snb_steps_total = state.snb_steps_total;
  rec.snb_picard_iters_last = state.snb_picard_iters_last;
  rec.snb_picard_iters_max = state.snb_picard_iters_max;
  rec.snb_nonconverged_steps = state.snb_nonconverged_steps;
  rec.snb_picard_resid_last = state.snb_picard_resid_last;
  rec.snb_cap_faces_99_last = state.snb_cap_faces_99_last;
  rec.snb_cap_faces_50_last = state.snb_cap_faces_50_last;
  rec.snb_cap_theta_min_last = state.snb_cap_theta_min_last;
  rec.snb_cap_theta_min_run = state.snb_cap_theta_min_run;
  rec.snb_dq_over_qsh_max_last = state.snb_dq_over_qsh_max_last;
  rec.snb_dq_over_qsh_max_run = state.snb_dq_over_qsh_max_run;
  rec.snb_solver_iters = state.snb_solver_iters;
  rec.snb_solver_resid = state.snb_solver_resid;
  rec.hot_e_enabled_any = state.hot_e_enabled_any;
  rec.hot_e_in_step = state.hot_e_in_step;
  rec.hot_e_deposited_step = state.hot_e_deposited_step;
  rec.hot_e_residual_step = state.hot_e_residual_step;
  rec.hot_e_escaped_step = state.hot_e_escaped_step;
  rec.hot_e_source_r = state.hot_e_source_r;
  rec.hot_e_conservation_resid = state.hot_e_conservation_resid;
  rec.E_hot_e_deposited = state.E_hot_e_deposited;
  rec.E_hot_e_escaped = state.E_hot_e_escaped;
  rec.hot_e_ch_in_step = state.hot_e_ch_in_step;
  rec.hot_e_ch_deposited_step = state.hot_e_ch_deposited_step;
  rec.hot_e_ch_escaped_step = state.hot_e_ch_escaped_step;
  rec.ale_rezone_invocations = state.ale_rezone_invocations;
  rec.snapshot = snapshot;
  rec.snapshot.ale_provenance = nullptr;
  if (snapshot.ale_provenance != nullptr) {
    rec.has_ale_provenance = true;
    rec.ale_provenance_values =
        build_ale_provenance_values(*snapshot.ale_provenance, rec.step);
  }
  if (rec.dt_breakdown_history_enabled) {
    rec.per_row_mass_values = compute_per_row_mass_values(state);
    rec.corner_bc_audit_values =
        compute_corner_bc_audit_values(state, snapshot.dt_breakdown);
    if (state.mesh.dim == 2) {
      AvMaxHistoryValues values{};
      values.step = state.step;
      values.t = state.t;
      values.cell_id = state.av_max_cell_id;
      values.i = state.av_max_i;
      values.j = state.av_max_j;
      values.q_visc_max = state.av_q_visc_max;
      values.rho_at_max = state.av_rho_at_max;
      values.cs_at_max = state.av_cs_at_max;
      values.delta_u_at_max = state.av_delta_u_at_max;
      rec.av_max_values = values;
    }
  }
  rec.plasma_diag = compute_plasma_history_diagnostics(state);
  rec.implosion_diag = compute_implosion_history_diagnostics(state);
  rec.tri_fan_center_perturbation_diag = state.tri_fan_center_perturbation_diag;
  return rec;
}

void HistoryWriter::write_per_row_mass_history(
    const hid_t file,
    const PerRowMassValues& values) const {
  constexpr const char* base = "/diagnostics/per_row_mass/";
  append_scalar_i64(file, std::string(base) + "cycle", values.step, "count");
  append_scalar_double(file, std::string(base) + "t_s", values.t, "s");
  write_i32_vector_dataset_if_missing(
      file, std::string(base) + "z_row_idx", values.z_row_idx);
  append_row_double_matrix(
      file, std::string(base) + "mass_per_z_row", values.mass, "g");
  append_row_double_matrix(
      file, std::string(base) + "rho_max_per_z_row", values.rho_max, "g/cm3");
  append_row_double_matrix(file,
                           std::string(base) + "rho_at_axis_per_z_row",
                           values.rho_axis,
                           "g/cm3");
  append_row_double_matrix(file,
                           std::string(base) + "rho_at_outer_per_z_row",
                           values.rho_outer,
                           "g/cm3");
  append_row_double_matrix(
      file, std::string(base) + "cs_max_per_z_row", values.cs_max, "cm/s");

  char cycle_base_buffer[128] = {};
  std::snprintf(cycle_base_buffer,
                sizeof(cycle_base_buffer),
                "%scycle_%05d/",
                base,
                values.step);
  const std::string cycle_base = cycle_base_buffer;
  write_i32_vector_dataset_if_missing(
      file, cycle_base + "z_row_idx", values.z_row_idx);
  write_double_vector_dataset_if_missing(
      file, cycle_base + "mass_per_z_row", values.mass, "g");
  write_double_vector_dataset_if_missing(
      file, cycle_base + "rho_max_per_z_row", values.rho_max, "g/cm3");
  write_double_vector_dataset_if_missing(
      file, cycle_base + "rho_at_axis_per_z_row", values.rho_axis, "g/cm3");
  write_double_vector_dataset_if_missing(
      file, cycle_base + "rho_at_outer_per_z_row", values.rho_outer, "g/cm3");
  write_double_vector_dataset_if_missing(
      file, cycle_base + "cs_max_per_z_row", values.cs_max, "cm/s");
}

void HistoryWriter::write_corner_bc_audit_history(
    const hid_t file,
    const CornerBcAuditValues& values) const {
  constexpr const char* base = "/diagnostics/corner_bc_audit/v1/";
  append_scalar_i64(file, std::string(base) + "cycle", values.cycle, "count");
  append_scalar_double(file, std::string(base) + "t_s", values.t_s, "s");
  append_scalar_i32(file, std::string(base) + "cell_id", values.cell_id, "index");
  append_scalar_i32(file, std::string(base) + "i", values.i, "index");
  append_scalar_i32(file, std::string(base) + "j", values.j, "index");
  append_scalar_double(file, std::string(base) + "interior_rho", values.interior_rho, "g/cm3");
  append_scalar_double(file, std::string(base) + "interior_u_r", values.interior_u_r, "cm/s");
  append_scalar_double(file, std::string(base) + "interior_u_z", values.interior_u_z, "cm/s");
  append_scalar_double(file, std::string(base) + "interior_e", values.interior_e, "erg/g");
  append_scalar_double(file, std::string(base) + "interior_E_r", values.interior_E_r, "erg/cm3");
  append_scalar_double(file,
                       std::string(base) + "r_outer_ghost_rho",
                       values.r_outer_ghost_rho,
                       "g/cm3");
  append_scalar_double(file,
                       std::string(base) + "r_outer_ghost_u_r",
                       values.r_outer_ghost_u_r,
                       "cm/s");
  append_scalar_double(file,
                       std::string(base) + "r_outer_ghost_u_z",
                       values.r_outer_ghost_u_z,
                       "cm/s");
  append_scalar_double(file,
                       std::string(base) + "z_top_ghost_rho",
                       values.z_top_ghost_rho,
                       "g/cm3");
  append_scalar_double(file,
                       std::string(base) + "z_top_ghost_u_r",
                       values.z_top_ghost_u_r,
                       "cm/s");
  append_scalar_double(file,
                       std::string(base) + "z_top_ghost_u_z",
                       values.z_top_ghost_u_z,
                       "cm/s");
  append_scalar_double(file,
                       std::string(base) + "diagonal_corner_ghost_rho",
                       values.diagonal_corner_ghost_rho,
                       "g/cm3");
  append_scalar_double(file,
                       std::string(base) + "diagonal_corner_ghost_u_r",
                       values.diagonal_corner_ghost_u_r,
                       "cm/s");
  append_scalar_double(file,
                       std::string(base) + "diagonal_corner_ghost_u_z",
                       values.diagonal_corner_ghost_u_z,
                       "cm/s");
  append_scalar_double(file,
                       std::string(base) + "dt_at_cell",
                       values.dt_at_cell,
                       "s");
  append_scalar_double(file, std::string(base) + "cs_at_cell", values.cs_at_cell, "cm/s");
  append_scalar_double(file,
                       std::string(base) + "q_visc_at_cell",
                       values.q_visc_at_cell,
                       "dyn/cm2");
}

void HistoryWriter::write_av_max_history(
    const hid_t file,
    const AvMaxHistoryValues& values) const {
  constexpr const char* base = "/diagnostics/av_max/";
  append_scalar_i64(file, std::string(base) + "cycle", values.step, "count");
  append_scalar_double(file, std::string(base) + "t_s", values.t, "s");
  append_scalar_i32(file,
                    std::string(base) + "cell_id",
                    static_cast<std::int32_t>(values.cell_id),
                    "index");
  append_scalar_i32(file,
                    std::string(base) + "i",
                    static_cast<std::int32_t>(values.i),
                    "index");
  append_scalar_i32(file,
                    std::string(base) + "j",
                    static_cast<std::int32_t>(values.j),
                    "index");
  append_scalar_double(
      file, std::string(base) + "q_visc_max", values.q_visc_max, "dyn/cm2");
  append_scalar_double(
      file, std::string(base) + "rho_at_max", values.rho_at_max, "g/cm3");
  append_scalar_double(
      file, std::string(base) + "cs_at_max", values.cs_at_max, "cm/s");
  append_scalar_double(file,
                       std::string(base) + "delta_u_at_max",
                       values.delta_u_at_max,
                       "cm/s");
}

void HistoryWriter::write_tri_fan_center_perturbation_history(
    const hid_t file,
    const core::TriFanCenterPerturbationDiag& diag,
    const double time,
    const std::int64_t step) const {
  if (!diag.valid) {
    return;
  }
  const std::string group_path = "/diagnostics/tri_fan_center_perturbation/v1";
  const hid_t group = open_or_create_group(file, group_path);
  write_u8_attribute_if_missing(group, "valid", static_cast<std::uint8_t>(1));
  warn_h5_close_failure(H5Gclose(group), "H5Gclose",
                        "HistoryWriter::write_tri_fan_center_perturbation_history");

  const std::string base = group_path + "/";
  append_scalar_i64(file, base + "cycle", step, "count");
  append_scalar_double(file, base + "time_s", time, "s");
  append_scalar_u8(file, base + "valid", static_cast<std::uint8_t>(1), "bool");
  append_scalar_double(file, base + "Edot_prime_P", diag.Edot_prime_P, "erg/s");
  append_scalar_double(file, base + "Edot_prime_Q", diag.Edot_prime_Q, "erg/s");
  append_scalar_double(file, base + "Eprime_k", diag.Eprime_k, "erg");
  append_scalar_double(
      file, base + "max_q_over_p", diag.max_q_over_p, "dimensionless");
  append_scalar_double(
      file, base + "min_corner_J", diag.min_corner_J, "dimensionless");
  append_scalar_double(
      file, base + "min_cell_volume", diag.min_cell_volume, "cm3");
}

void HistoryWriter::write_ale_provenance_history(
    const hid_t file,
    const AleProvenanceValues& values,
    const double time,
    const std::int64_t step) const {
  const std::string base = "/diagnostics/ale_provenance/v1/";
  append_scalar_double(file, base + "time_s", time, "s");
  append_scalar_i32(file, base + "step", static_cast<std::int32_t>(step), "count");
  append_scalar_i32(file,
                    base + "forbidden_config_violations",
                    static_cast<std::int32_t>(values.forbidden_config_violations),
                    "count");
  append_scalar_i32(file,
                    base + "escape_valve_activations",
                    static_cast<std::int32_t>(values.escape_valve_activations),
                    "count");
  append_scalar_i32(file,
                    base + "class_c_runtime_fires",
                    static_cast<std::int32_t>(values.class_c_runtime_fires),
                    "count");
  append_scalar_i32(file,
                    base + "mesh_geometry_failures_observed",
                    static_cast<std::int32_t>(values.mesh_geometry_failures_observed),
                    "count");
  append_scalar_i32(file,
                    base + "public_baseline_terminal_failures",
                    static_cast<std::int32_t>(
                        values.public_baseline_terminal_failures),
                    "count");
  append_scalar_u8(file,
                   base + "emergency_cell_deactivation_fired",
                   static_cast<std::uint8_t>(
                       values.emergency_cell_deactivation_fired ? 1U : 0U),
                   "bool");
  append_scalar_i32(file,
                    base + "last_failing_cell",
                    static_cast<std::int32_t>(values.last_failing_cell),
                    "cell");
  append_scalar_i32(file,
                    base + "last_failing_i",
                    static_cast<std::int32_t>(values.last_failing_i),
                    "index");
  append_scalar_i32(file,
                    base + "last_failing_j",
                    static_cast<std::int32_t>(values.last_failing_j),
                    "index");
  append_scalar_double(
      file, base + "last_min_cell_vol", values.last_min_cell_vol, "cm3");
  append_scalar_double(
      file, base + "last_min_corner_j", values.last_min_corner_j, "cm2");
  append_scalar_u8(file,
                   base + "last_failure_kind",
                   static_cast<std::uint8_t>(values.last_failure_kind),
                   "enum");
  const std::string event_base = base + "escape_valve_events/";
  for (const auto& event : values.escape_valve_events) {
    append_scalar_fixed_string(
        file, event_base + "split_phase", event.split_phase);
    append_scalar_fixed_string(
        file, event_base + "operator_inserted", event.operator_inserted);
    append_scalar_u8(file,
                     event_base + "order_degraded",
                     static_cast<std::uint8_t>(event.order_degraded ? 1U : 0U),
                     "bool");
    append_scalar_double(
        file, event_base + "E_thermal_before", event.E_thermal_before, "erg");
    append_scalar_double(
        file, event_base + "E_thermal_after", event.E_thermal_after, "erg");
    append_scalar_double(
        file, event_base + "E_kinetic_before", event.E_kinetic_before, "erg");
    append_scalar_double(
        file, event_base + "E_kinetic_after", event.E_kinetic_after, "erg");
    append_scalar_double(
        file, event_base + "mass_transfer", event.mass_transfer, "g");
    append_scalar_double(file,
                         event_base + "momentum_transfer_R",
                         event.momentum_transfer_R,
                         "g*cm/s");
    append_scalar_double(file,
                         event_base + "momentum_transfer_Z",
                         event.momentum_transfer_Z,
                         "g*cm/s");
    append_scalar_double(file,
                         event_base + "internal_energy_transfer",
                         event.internal_energy_transfer,
                         "erg");
    append_scalar_i32(
        file, event_base + "step", static_cast<std::int32_t>(event.step), "count");
    append_scalar_double(file, event_base + "time", event.time, "s");
  }
}

void HistoryWriter::write_material_interface_history(
    const hid_t file,
    const AleProvenanceValues& values,
    const double time,
    const std::int64_t step) const {
  if (!cfg_.numerics.plic.enabled) {
    return;
  }
  const std::string base = "/diagnostics/material_interface/v1/";
  append_scalar_double(file, base + "time_s", time, "s");
  append_scalar_i32(file, base + "step", static_cast<std::int32_t>(step), "count");
  append_scalar_i32(file,
                    base + "interface_cells_observed",
                    static_cast<std::int32_t>(values.interface_cells_observed),
                    "count");
  append_scalar_i32(
      file,
      base + "interface_reconstruction_attempt_count",
      static_cast<std::int32_t>(values.interface_reconstruction_attempt_count),
      "count");
  append_scalar_i32(
      file,
      base + "interface_reconstruction_success_count",
      static_cast<std::int32_t>(values.interface_reconstruction_success_count),
      "count");
  append_scalar_double(
      file, base + "plic_max_eta_E_observed", values.plic_max_eta_E_observed, "dimensionless");
  append_scalar_double(file,
                       base + "plic_max_volume_fraction_residual_observed",
                       values.plic_max_volume_fraction_residual_observed,
                       "dimensionless");
  append_scalar_double(file,
                       base + "plic_min_grad_F_observed",
                       values.plic_min_grad_F_observed,
                       "dimensionless");
  int class_d_runtime_fires_matrix[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      class_d_runtime_fires_matrix[i][j] =
          values.class_d_runtime_fires_matrix[static_cast<std::size_t>(i)]
                                              [static_cast<std::size_t>(j)];
    }
  }
  append_3x3_i32_matrix(
      file, base + "class_d_runtime_fires_matrix", class_d_runtime_fires_matrix, "count");

  if (values.plic_events.empty()) {
    return;
  }

  if (values.plic_events_summary_mode) {
    const std::string summary_base = base + "plic_events_summary/";
    for (int case_idx = 0; case_idx < 3; ++case_idx) {
      for (int severity = 0; severity < 3; ++severity) {
        const std::int32_t count =
            values.plic_event_summary_counts[static_cast<std::size_t>(case_idx)]
                                            [static_cast<std::size_t>(severity)];
        if (count > 0) {
          append_plic_event_summary_row(
              file,
              summary_base,
              static_cast<std::uint8_t>(case_idx + 1),
              static_cast<std::uint8_t>(severity),
              count,
              static_cast<int>(step),
              time);
        }
      }
    }
    return;
  }

  const std::string event_base = base + "plic_events/";
  const std::string per_cell_base = base + "per_cell_state/";
  const bool write_per_cell = material_interface_per_cell_state_enabled(cfg_);
  for (const auto& evt : values.plic_events) {
    append_plic_event_row(file, event_base, evt);
    if (write_per_cell) {
      append_plic_per_cell_state_row(file, per_cell_base, evt);
    }
  }
}

void HistoryWriter::write_mesh_quality_min_history(
    const hid_t file,
    const AleProvenanceValues& values,
    const double time,
    const std::int64_t step) const {
  if (!values.mesh_quality_min_observed) {
    return;
  }
  const std::string base = "/diagnostics/mesh_quality_min/v1/";
  append_scalar_double(file, base + "time_s", time, "s");
  append_scalar_i32(file, base + "step", static_cast<std::int32_t>(step), "count");
  append_scalar_double(file,
                       base + "achieved_min_corner_j_rel",
                       values.achieved_min_corner_j_rel,
                       "dimensionless");
  append_scalar_double(file,
                       base + "achieved_min_gauss_j_rel",
                       values.achieved_min_gauss_j_rel,
                       "dimensionless");
  append_scalar_double(file,
                       base + "achieved_min_rz_volume_rel",
                       values.achieved_min_rz_volume_rel,
                       "dimensionless");
  append_scalar_i64(
      file,
      base + "negative_rz_volume_count_total",
      saturating_i64_from_u64(values.negative_rz_volume_count_total),
      "count");
  append_scalar_double(file,
                       base + "achieved_min_edge_length_rel",
                       values.achieved_min_edge_length_rel,
                       "dimensionless");
  append_scalar_double(file,
                       base + "achieved_min_altitude_rel",
                       values.achieved_min_altitude_rel,
                       "dimensionless");
  append_scalar_double(file,
                       base + "achieved_max_condition_number",
                       values.achieved_max_condition_number,
                       "dimensionless");
}

void write_mesh_quality_min_history(
    const hid_t file,
    const tenryu::coupling::ProfileObservability& obs,
    const double time,
    const int step) {
  if (!obs.mesh_quality_min_observed) {
    return;
  }
  const std::string base = "/diagnostics/mesh_quality_min/v1/";
  append_scalar_double(file, base + "time_s", time, "s");
  append_scalar_i32(file, base + "step", static_cast<std::int32_t>(step), "count");
  append_scalar_double(file,
                       base + "achieved_min_corner_j_rel",
                       obs.achieved_min_corner_j_rel,
                       "dimensionless");
  append_scalar_double(file,
                       base + "achieved_min_gauss_j_rel",
                       obs.achieved_min_gauss_j_rel,
                       "dimensionless");
  append_scalar_double(file,
                       base + "achieved_min_rz_volume_rel",
                       obs.achieved_min_rz_volume_rel,
                       "dimensionless");
  append_scalar_i64(
      file,
      base + "negative_rz_volume_count_total",
      saturating_i64_from_u64(obs.negative_rz_volume_count_total),
      "count");
  append_scalar_double(file,
                       base + "achieved_min_edge_length_rel",
                       obs.achieved_min_edge_length_rel,
                       "dimensionless");
  append_scalar_double(file,
                       base + "achieved_min_altitude_rel",
                       obs.achieved_min_altitude_rel,
                       "dimensionless");
  append_scalar_double(file,
                       base + "achieved_max_condition_number",
                       obs.achieved_max_condition_number,
                       "dimensionless");
}

void write_ale_provenance_history(
    const hid_t file,
    tenryu::coupling::ProfileObservability& obs,
    const double time,
    const int step) {
  const std::string base = "/diagnostics/ale_provenance/v1/";
  append_scalar_double(file, base + "time_s", time, "s");
  append_scalar_i32(file, base + "step", static_cast<std::int32_t>(step), "count");
  append_scalar_i32(file,
                    base + "forbidden_config_violations",
                    static_cast<std::int32_t>(obs.forbidden_config_violations),
                    "count");
  append_scalar_i32(file,
                    base + "escape_valve_activations",
                    static_cast<std::int32_t>(obs.escape_valve_activations),
                    "count");
  append_scalar_i32(file,
                    base + "class_c_runtime_fires",
                    static_cast<std::int32_t>(obs.class_c_runtime_fires),
                    "count");
  append_scalar_i32(file,
                    base + "mesh_geometry_failures_observed",
                    static_cast<std::int32_t>(obs.mesh_geometry_failures_observed),
                    "count");
  append_scalar_i32(file,
                    base + "public_baseline_terminal_failures",
                    static_cast<std::int32_t>(
                        obs.public_baseline_terminal_failures),
                    "count");
  append_scalar_u8(file,
                   base + "emergency_cell_deactivation_fired",
                   static_cast<std::uint8_t>(
                       obs.emergency_cell_deactivation_fired ? 1U : 0U),
                   "bool");
  append_scalar_i32(file,
                    base + "last_failing_cell",
                    static_cast<std::int32_t>(obs.last_failing_cell),
                    "cell");
  append_scalar_i32(file,
                    base + "last_failing_i",
                    static_cast<std::int32_t>(obs.last_failing_i),
                    "index");
  append_scalar_i32(file,
                    base + "last_failing_j",
                    static_cast<std::int32_t>(obs.last_failing_j),
                    "index");
  append_scalar_double(
      file, base + "last_min_cell_vol", obs.last_min_cell_vol, "cm3");
  append_scalar_double(
      file, base + "last_min_corner_j", obs.last_min_corner_j, "cm2");
  append_scalar_u8(file,
                   base + "last_failure_kind",
                   static_cast<std::uint8_t>(obs.last_failure_kind),
                   "enum");
  const std::string event_base = base + "escape_valve_events/";
  for (const auto& event : obs.escape_valve_events) {
    append_scalar_fixed_string(
        file, event_base + "split_phase", event.split_phase);
    append_scalar_fixed_string(
        file, event_base + "operator_inserted", event.operator_inserted);
    append_scalar_u8(file,
                     event_base + "order_degraded",
                     static_cast<std::uint8_t>(event.order_degraded ? 1U : 0U),
                     "bool");
    append_scalar_double(
        file, event_base + "E_thermal_before", event.E_thermal_before, "erg");
    append_scalar_double(
        file, event_base + "E_thermal_after", event.E_thermal_after, "erg");
    append_scalar_double(
        file, event_base + "E_kinetic_before", event.E_kinetic_before, "erg");
    append_scalar_double(
        file, event_base + "E_kinetic_after", event.E_kinetic_after, "erg");
    append_scalar_double(
        file, event_base + "mass_transfer", event.mass_transfer, "g");
    append_scalar_double(file,
                         event_base + "momentum_transfer_R",
                         event.momentum_transfer_R,
                         "g*cm/s");
    append_scalar_double(file,
                         event_base + "momentum_transfer_Z",
                         event.momentum_transfer_Z,
                         "g*cm/s");
    append_scalar_double(file,
                         event_base + "internal_energy_transfer",
                         event.internal_energy_transfer,
                         "erg");
    append_scalar_i32(
        file, event_base + "step", static_cast<std::int32_t>(event.step), "count");
    append_scalar_double(file, event_base + "time", event.time, "s");
  }
  obs.escape_valve_events.clear();
}

void write_ale_provenance_final_attributes(const hid_t file,
                                           const char* final_provenance,
                                           const char* final_claim_level,
                                           const bool reached_t_end,
                                           const bool profile_enabled,
                                           const char* plic_gate_status) {
  const hid_t group = open_or_create_group(file, "/diagnostics/ale_provenance/v1");
  write_string_attribute_if_missing(group, "final_provenance", final_provenance);
  write_string_attribute_if_missing(group, "final_claim_level", final_claim_level);
  write_u8_attribute_if_missing(
      group, "reached_t_end", static_cast<std::uint8_t>(reached_t_end ? 1U : 0U));
  write_u8_attribute_if_missing(group,
                                "profile_enabled",
                                static_cast<std::uint8_t>(
                                    profile_enabled ? 1U : 0U));
  if (plic_gate_status != nullptr && plic_gate_status[0] != '\0') {
    write_string_attribute_if_missing(group, "plic_gate_status", plic_gate_status);
  }
  warn_h5_close_failure(H5Gclose(group), "H5Gclose",
                        "HistoryWriter::write_ale_provenance_final_attributes");
}

void write_material_interface_history(
    const hid_t file,
    tenryu::coupling::ProfileObservability& obs,
    const tenryu::core::Config& cfg,
    const double time,
    const int step) {
  if (!cfg.numerics.plic.enabled) {
    return;
  }
  const std::string base = "/diagnostics/material_interface/v1/";
  append_scalar_double(file, base + "time_s", time, "s");
  append_scalar_i32(file, base + "step", static_cast<std::int32_t>(step), "count");
  append_scalar_i32(file,
                    base + "interface_cells_observed",
                    static_cast<std::int32_t>(obs.interface_cells_observed),
                    "count");
  append_scalar_i32(
      file,
      base + "interface_reconstruction_attempt_count",
      static_cast<std::int32_t>(obs.interface_reconstruction_attempt_count),
      "count");
  append_scalar_i32(
      file,
      base + "interface_reconstruction_success_count",
      static_cast<std::int32_t>(obs.interface_reconstruction_success_count),
      "count");
  append_scalar_double(
      file, base + "plic_max_eta_E_observed", obs.plic_max_eta_E_observed, "dimensionless");
  append_scalar_double(file,
                       base + "plic_max_volume_fraction_residual_observed",
                       obs.plic_max_volume_fraction_residual_observed,
                       "dimensionless");
  append_scalar_double(file,
                       base + "plic_min_grad_F_observed",
                       obs.plic_min_grad_F_observed,
                       "dimensionless");
  append_3x3_i32_matrix(
      file, base + "class_d_runtime_fires_matrix", obs.class_d_runtime_fires_matrix, "count");

  const std::size_t start = obs.plic_events_emitted_count;
  const std::size_t end = obs.plic_events.size();
  if (start >= end) {
    obs.plic_events_emitted_count = end;
    return;
  }

  const std::size_t n_new = end - start;
  if (n_new > kPlicEventsPerStepLimit) {
    std::array<std::array<std::int32_t, 3>, 3> counts{};
    for (std::size_t idx = start; idx < end; ++idx) {
      const auto& evt = obs.plic_events[idx];
      if (evt.case_id >= 1 && evt.case_id <= 3 && evt.severity <= 2) {
        ++counts[evt.case_id - 1][evt.severity];
      }
    }
    const std::string summary_base = base + "plic_events_summary/";
    for (int case_idx = 0; case_idx < 3; ++case_idx) {
      for (int severity = 0; severity < 3; ++severity) {
        if (counts[case_idx][severity] > 0) {
          append_plic_event_summary_row(
              file,
              summary_base,
              static_cast<std::uint8_t>(case_idx + 1),
              static_cast<std::uint8_t>(severity),
              counts[case_idx][severity],
              step,
              time);
        }
      }
    }
    std::cout << "[plic_events] event-rate limit exceeded (" << n_new
              << " events in step " << step
              << "); switched to summary mode" << std::endl;
    obs.plic_events_emitted_count = end;
    return;
  }

  const std::string event_base = base + "plic_events/";
  const std::string per_cell_base = base + "per_cell_state/";
  const bool write_per_cell = material_interface_per_cell_state_enabled(cfg);
  for (std::size_t idx = start; idx < end; ++idx) {
    const auto& evt = obs.plic_events[idx];
    append_plic_event_row(file, event_base, evt);
    if (write_per_cell) {
      append_plic_per_cell_state_row(file, per_cell_base, evt);
    }
  }
  obs.plic_events_emitted_count = end;
}

void write_material_interface_final_attributes(
    const hid_t file,
    const tenryu::coupling::ProfileObservability& obs,
    const tenryu::core::Config& cfg) {
  (void)obs;
  if (!cfg.numerics.plic.enabled) {
    return;
  }
  const hid_t group =
      open_or_create_group(file, "/diagnostics/material_interface/v1");
  write_string_attribute_if_missing(
      group, "plic_reconstruction_engine_version", "young_plic_v1_skeleton");
  write_string_attribute_if_missing(
      group, "plic_normal_estimator", cfg.numerics.plic.normal_estimator.c_str());
  write_string_attribute_if_missing(
      group, "t0_volume_cut_method", cfg.numerics.plic.t0_volume_cut_method.c_str());
  write_u8_attribute_if_missing(
      group,
      "plic_enabled",
      static_cast<std::uint8_t>(cfg.numerics.plic.enabled ? 1U : 0U));
  write_i32_attribute_if_missing(group, "plic_schema_version", 1);
  write_string_attribute_if_missing(group, "plic_reconstruction_method", "PLIC");
  write_string_attribute_if_missing(
      group,
      "final_class_d_aggregate",
      obs.compute_final_class_d_aggregate().c_str());
  write_u8_attribute_if_missing(
      group,
      "plic_remap_fallback_engaged",
      static_cast<std::uint8_t>(obs.plic_remap_fallback_engaged ? 1U : 0U));
  warn_h5_close_failure(H5Gclose(group), "H5Gclose",
                        "HistoryWriter::write_material_interface_final_attributes");
}

void append_ale_provenance_final_to_history_file(
    const std::string& history_path,
    tenryu::coupling::ProfileObservability& obs,
    const tenryu::core::Config& cfg,
    const char* final_provenance,
    const char* final_claim_level,
    const double time,
    const int step) {
  if (history_path.empty()) {
    return;
  }
  const bool ale_provenance_enabled =
      core::effective_diagnostics_ale_provenance_emission_enabled(cfg);
  const bool mesh_quality_min_enabled =
      cfg.numerics.diagnostics.mesh_quality_min.enabled;
  if (!ale_provenance_enabled && !mesh_quality_min_enabled) {
    return;
  }
  const bool exists = std::filesystem::exists(history_path);
  const hid_t file =
      exists ? H5Fopen(history_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT)
             : H5Fcreate(history_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  TENRYU_ASSERT(file >= 0,
                "HistoryWriter failed to open history file for final ALE provenance");
  if (ale_provenance_enabled) {
    write_ale_provenance_history(file, obs, time, step);
    write_material_interface_history(file, obs, cfg, time, step);
  }
  if (mesh_quality_min_enabled) {
    write_mesh_quality_min_history(file, obs, time, step);
  }
  if (ale_provenance_enabled) {
    const char* plic_gate_status =
        obs.plic_gate_status_recorded ? obs.plic_gate_status.c_str() : nullptr;
    write_ale_provenance_final_attributes(
        file,
        final_provenance,
        final_claim_level,
        obs.reached_t_end,
        obs.profile_enabled,
        plic_gate_status);
    write_material_interface_final_attributes(file, obs, cfg);
  }
  warn_h5_close_failure(H5Fclose(file), "H5Fclose",
                        "HistoryWriter::append_ale_provenance_final_to_history_file");
}

std::optional<double> read_icf_initial_radius_from_history_file(
    const std::string& history_path) {
  if (history_path.empty() || !std::filesystem::exists(history_path)) {
    return std::nullopt;
  }
  const hid_t file = H5Fopen(history_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    core::log_warning("HistoryWriter: failed to open restart history file '" +
                      history_path + "' for ICF initial radius");
    return std::nullopt;
  }

  std::optional<double> result;
  if (tenryu::io::h5_link_exists(file, "/diagnostics/icf/v1", H5P_DEFAULT) > 0) {
    const hid_t group = H5Gopen2(file, "/diagnostics/icf/v1", H5P_DEFAULT);
    TENRYU_ASSERT(group >= 0,
                  "HistoryWriter H5Gopen2(/diagnostics/icf/v1) failed");
    if (const auto attr = read_double_attribute_if_exists(group, "R_initial_cm");
        attr.has_value() && *attr > 0.0 && std::isfinite(*attr)) {
      result = *attr;
    }
    warn_h5_close_failure(H5Gclose(group), "H5Gclose",
                          "HistoryWriter::read_icf_initial_radius_from_history_file(group)");
  }

  if (!result.has_value()) {
    const auto shell =
        first_scalar_double_if_exists(file, "/diagnostics/icf/v1/shell_radius_cm");
    const auto cr = first_scalar_double_if_exists(file, "/diagnostics/icf/v1/CR");
    if (shell.has_value() && cr.has_value() && *shell > 0.0 && *cr > 0.0 &&
        std::isfinite(*shell) && std::isfinite(*cr)) {
      result = (*shell) * (*cr);
    }
  }

  warn_h5_close_failure(H5Fclose(file), "H5Fclose",
                        "HistoryWriter::read_icf_initial_radius_from_history_file(file)");
  return result;
}

#endif

HistoryWriter::~HistoryWriter() noexcept {
  try {
    flush_pending();
  } catch (...) {
    core::log_warning("HistoryWriter: suppressed exception during destructor flush");
  }
}

void HistoryWriter::init(const core::Config& cfg, const std::string& output_dir) {
  cfg_ = cfg;
  append_flush_every_ = 64;
  if (const char* value = std::getenv("TENRYU_HISTORY_APPEND_EVERY");
      value != nullptr && value[0] != '\0') {
    append_flush_every_ = std::max(1, std::atoi(value));
  }
  last_flush_time_ = std::chrono::steady_clock::now();
  pending_.clear();
  history_bootstrapped_ = false;
  mc_group_enabled_ = cfg.diagnostics.mc_stats.enabled;
  mc_particle_counts_enabled_ = cfg.diagnostics.mc_stats.particle_counts;
  mc_weight_stats_enabled_ = cfg.diagnostics.mc_stats.weight_stats;
  mc_ddmc_fraction_enabled_ = cfg.diagnostics.mc_stats.ddmc_fraction;
  phase_resolved_energy_enabled_ = cfg.numerics.diagnostics.phase_resolved_energy;
  ale_closure_audit_enabled_ = cfg.numerics.ale.ke_conservation_closure_audit;
  icf_enabled_ = core::effective_diagnostics_icf_enabled(cfg);
  hotspot_gas_enabled_ = core::effective_diagnostics_hotspot_gas_enabled(cfg);
  ale_provenance_enabled_ =
      core::effective_diagnostics_ale_provenance_emission_enabled(cfg);
  mesh_quality_min_enabled_ = cfg.numerics.diagnostics.mesh_quality_min.enabled;
  conservation_enabled_ = core::effective_diagnostics_conservation_enabled(cfg);
  dt_breakdown_history_enabled_ =
      cfg.numerics.diagnostics.dt_breakdown_history_enabled;
  center_perturbation_enabled_ =
      cfg.numerics.hydro.center_perturbation_diag_scope !=
      core::CenterPerturbationDiagScope::DISABLED;

  const bool history_requested =
      (cfg.output.history_every > 0) || (cfg.output.history_every_s > 0.0) ||
      dt_breakdown_history_enabled_ ||
      mesh_quality_min_enabled_ ||
      center_perturbation_enabled_ ||
      hotspot_gas_enabled_ ||
      cfg.diagnostics.per_operator_radial_fourier_enabled ||
      cfg.diagnostics.per_operator_radial_fourier_complex_enabled ||
      cfg.radiation.multigroup_diffusion.diagnostic_radial_fourier_substage_enabled;
  if (!history_requested) {
    enabled_ = false;
    path_.clear();
    return;
  }

  std::filesystem::create_directories(output_dir);
  const std::string case_name =
      cfg.main.name.empty() ? std::string("unnamed") : cfg.main.name;
  path_ = (std::filesystem::path(output_dir) / (case_name + "_history.h5")).string();

#if TENRYU_ENABLE_HDF5
  enabled_ = true;
#else
  enabled_ = false;
  core::log_info("HistoryWriter: TENRYU_ENABLE_HDF5=OFF, history output disabled");
#endif
}

void HistoryWriter::append_radial_fourier_audit(
    const RadialFourierAuditRecord& record) {
  if (!enabled_ || !record.valid) {
    return;
  }
  flush_pending();

#if TENRYU_ENABLE_HDF5
  const std::filesystem::path history_path(path_);
  const std::filesystem::path temp_path = history_path.string() + ".tmp";
  const bool exists = std::filesystem::exists(history_path);
  std::error_code remove_tmp_ec;
  std::filesystem::remove(temp_path, remove_tmp_ec);
  if (exists) {
    TENRYU_ASSERT(std::filesystem::copy_file(history_path,
                                             temp_path,
                                             std::filesystem::copy_options::overwrite_existing),
                  "HistoryWriter failed to stage history file copy for radial Fourier append");
  }
  const hid_t file =
      exists ? H5Fopen(temp_path.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT)
             : H5Fcreate(temp_path.string().c_str(),
                         H5F_ACC_TRUNC,
                         H5P_DEFAULT,
                         H5P_DEFAULT);
  TENRYU_ASSERT(file >= 0,
                "HistoryWriter failed to open/create history file for radial Fourier audit");

  write_radial_fourier_audit_history(file, record);

  warn_h5_close_failure(H5Fclose(file), "H5Fclose",
                        "HistoryWriter::append_radial_fourier_audit(finalize)");
  try {
    std::filesystem::rename(temp_path, history_path);
  } catch (const std::filesystem::filesystem_error& e) {
    core::log_error("HistoryWriter: transactional rename failed from '" +
                    temp_path.string() + "' to '" + history_path.string() +
                    "': " + e.what());
    std::error_code cleanup_ec;
    std::filesystem::remove(temp_path, cleanup_ec);
  }
#endif
}

void HistoryWriter::append_radial_fourier_complex_audit(
    const RadialFourierComplexAuditRecord& record) {
#if !TENRYU_RFA_V2_WRITES_HISTORY
  (void)record;
  return;
#else
  if (!enabled_ || !record.valid) {
    return;
  }
  flush_pending();

#if TENRYU_ENABLE_HDF5
  const std::filesystem::path history_path(path_);
  const std::filesystem::path temp_path = history_path.string() + ".tmp";
  const bool exists = std::filesystem::exists(history_path);
  std::error_code remove_tmp_ec;
  std::filesystem::remove(temp_path, remove_tmp_ec);
  if (exists) {
    TENRYU_ASSERT(
        std::filesystem::copy_file(
            history_path,
            temp_path,
            std::filesystem::copy_options::overwrite_existing),
        "HistoryWriter failed to stage history file copy for radial Fourier v2 append");
  }
  const hid_t file =
      exists ? H5Fopen(temp_path.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT)
             : H5Fcreate(temp_path.string().c_str(),
                         H5F_ACC_TRUNC,
                         H5P_DEFAULT,
                         H5P_DEFAULT);
  TENRYU_ASSERT(file >= 0,
                "HistoryWriter failed to open/create history file for radial Fourier v2 audit");

  write_radial_fourier_complex_audit_history(file, record);

  warn_h5_close_failure(
      H5Fclose(file),
      "H5Fclose",
      "HistoryWriter::append_radial_fourier_complex_audit(finalize)");
  try {
    std::filesystem::rename(temp_path, history_path);
  } catch (const std::filesystem::filesystem_error& e) {
    core::log_error("HistoryWriter: transactional rename failed from '" +
                    temp_path.string() + "' to '" + history_path.string() +
                    "': " + e.what());
    std::error_code cleanup_ec;
    std::filesystem::remove(temp_path, cleanup_ec);
  }
#endif
#endif
}

void HistoryWriter::append_fld_substage_audit_batch(
    const std::vector<FldSubstageAuditRecord>& records) {
  const bool has_valid =
      std::any_of(records.begin(), records.end(), [](const auto& record) {
        return record.valid;
      });
  if (!enabled_ || !has_valid) {
    return;
  }
  flush_pending();

#if TENRYU_ENABLE_HDF5
  const std::filesystem::path history_path(path_);
  const std::filesystem::path temp_path = history_path.string() + ".tmp";
  const bool exists = std::filesystem::exists(history_path);
  std::error_code remove_tmp_ec;
  std::filesystem::remove(temp_path, remove_tmp_ec);
  if (exists) {
    TENRYU_ASSERT(
        std::filesystem::copy_file(
            history_path,
            temp_path,
            std::filesystem::copy_options::overwrite_existing),
        "HistoryWriter failed to stage history file copy for FLD substage audit append");
  }
  const hid_t file =
      exists ? H5Fopen(temp_path.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT)
             : H5Fcreate(temp_path.string().c_str(),
                         H5F_ACC_TRUNC,
                         H5P_DEFAULT,
                         H5P_DEFAULT);
  TENRYU_ASSERT(file >= 0,
                "HistoryWriter failed to open/create history file for FLD substage audit");

  write_fld_substage_audit_history(file, records);

  warn_h5_close_failure(
      H5Fclose(file),
      "H5Fclose",
      "HistoryWriter::append_fld_substage_audit_batch(finalize)");
  try {
    std::filesystem::rename(temp_path, history_path);
  } catch (const std::filesystem::filesystem_error& e) {
    core::log_error("HistoryWriter: transactional rename failed from '" +
                    temp_path.string() + "' to '" + history_path.string() +
                    "': " + e.what());
    std::error_code cleanup_ec;
    std::filesystem::remove(temp_path, cleanup_ec);
  }
#endif
}

void HistoryWriter::append(const core::State& state, const HistorySnapshot& snapshot) {
  if (!enabled_) {
    (void)state;
    (void)snapshot;
    return;
  }

#if TENRYU_ENABLE_HDF5
  pending_.push_back(build_pending_record(state, snapshot));
  const auto now = std::chrono::steady_clock::now();
  if (!history_bootstrapped_ ||
      static_cast<int>(pending_.size()) >= append_flush_every_ ||
      now - last_flush_time_ >= std::chrono::seconds(30)) {
    flush_pending();
  }
#else
  (void)state;
  (void)snapshot;
#endif
}

void HistoryWriter::flush_pending() {
  last_flush_time_ = std::chrono::steady_clock::now();
  if (pending_.empty() || !enabled_) {
    return;
  }

#if TENRYU_ENABLE_HDF5
  const std::filesystem::path history_path(path_);
  const std::filesystem::path temp_path = history_path.string() + ".tmp";
  const bool exists = std::filesystem::exists(history_path);
  std::error_code remove_tmp_ec;
  std::filesystem::remove(temp_path, remove_tmp_ec);
  if (exists) {
    TENRYU_ASSERT(std::filesystem::copy_file(history_path,
                                             temp_path,
                                             std::filesystem::copy_options::overwrite_existing),
                  "HistoryWriter failed to stage history file copy for transactional append");
  }
  const hid_t file =
      exists ? H5Fopen(temp_path.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT)
             : H5Fcreate(temp_path.string().c_str(),
                         H5F_ACC_TRUNC,
                         H5P_DEFAULT,
                         H5P_DEFAULT);
  TENRYU_ASSERT(file >= 0, "HistoryWriter failed to open/create history file");

  for (const auto& rec : pending_) {
    append_record_to_file(file, rec);
  }

  warn_h5_close_failure(H5Fclose(file), "H5Fclose", "HistoryWriter::append(finalize)");
  try {
    std::filesystem::rename(temp_path, history_path);
    history_bootstrapped_ = true;
  } catch (const std::filesystem::filesystem_error& e) {
    core::log_error("HistoryWriter: transactional rename failed from '" + temp_path.string() +
                    "' to '" + history_path.string() + "': " + e.what());
    std::error_code cleanup_ec;
    std::filesystem::remove(temp_path, cleanup_ec);
  }
  pending_.clear();
#else
  pending_.clear();
#endif
}

#if TENRYU_ENABLE_HDF5
void HistoryWriter::append_record_to_file(
    const hid_t file,
    const PendingHistoryRecord& rec) const {
  const auto last_t = last_scalar_double_if_exists(file, "t");
  if (last_t.has_value() && !(rec.t > *last_t)) {
    core::log_warning("HistoryWriter: skipped non-monotonic append at t=" +
                      std::to_string(rec.t) + " (last_t=" + std::to_string(*last_t) + ")");
    return;
  }

  append_scalar_double(file, "t", rec.t, "s");
  append_scalar_i64(file, "cycle", static_cast<std::int64_t>(rec.step), "count");
  append_scalar_double(file, "dt", rec.dt, "s");
  if (rec.dt_breakdown_history_enabled) {
    write_dt_breakdown_history(file, rec.snapshot.dt_breakdown);
    write_cfl_winner_history(file, rec.snapshot.dt_breakdown);
    if (rec.corner_bc_audit_values.has_value()) {
      write_corner_bc_audit_history(file, *rec.corner_bc_audit_values);
    }
    if (rec.per_row_mass_values.has_value()) {
      write_per_row_mass_history(file, *rec.per_row_mass_values);
    }
    if (rec.av_max_values.has_value()) {
      write_av_max_history(file, *rec.av_max_values);
    }
  }
  write_plasma_viscosity_history(file, rec.snapshot.plasma_viscosity);
  if (rec.center_perturbation_enabled) {
    write_tri_fan_center_perturbation_history(
        file, rec.tri_fan_center_perturbation_diag, rec.t, rec.step);
  }

  // Matches SPECIFICATION.md §7 history keys:
  // - t, cycle, dt
  // - energy/{internal_electron, internal_ion, kinetic, radiation_field, laser_incident,
  //   marshak_in, laser_deposited, laser_escaped, radiation_escaped, numerical_loss,
  //   pdv_boundary, floor_injected, safety_injected, redistribution_unresolved,
  //   solver_residual, conservation_error}
  //   laser_incident, laser_deposited, and laser_escaped are cumulative counters.
  // - plasma/{Zbar_mean, Zbar_max}
  // - implosion/{rho_peak, rho_R, shell_radius_mean, shell_radius_min, center_temperature}
  // - modes/{P_ell, ell_values}
  // - laser/{critical_surface_r, near_critical_r, absorption_weighted_r,
  //   absorbed_total, absorbed_power_total, commanded_power_total,
  //   trace_unabsorbed_power_total, unabsorbed_power_total,
  //   trace_absorption_efficiency_total, absorption_efficiency_total,
  //   transfer_blocked_power_total,
  //   tail_closure_count, tail_closure_absorbed_power_total,
  //   cbet_exchanged_power_total, cbet_ledger_residual_rel, cbet_iterations,
  //   cbet_clamp_count,
  //   critical_surface_hit_count,
  //   absorbed_fraction_beam_<index>}
  // - mc/{n_total, n_imc, n_ddmc, n_census, n_absorbed, n_escaped, n_leaked, ddmc_fraction,
  //   weight_min, weight_mean, weight_max, overshoot_count, overshoot_max, ddmc_mode_count,
  //   imc_mode_count, mmatrix_violations, mmatrix_fallback_count, omega_below_threshold,
  //   interface_transitions, interface_reflections, conversion_prob_violations,
  //   ddmc_to_imc_conversions, rad_momentum_deposition}
  // - holo/{n_core_cells, n_entered, n_exited, n_hard_exited,
  //   n_island_rejected, tau_R_min, tau_R_max, reduced_flux_max,
  //   E_LO_total, E_LO_boundary_in, E_LO_boundary_out, matter_delta,
  //   source_balance_error, particle_net_source_core, lo_particle_source_mismatch,
  //   Prr_coverage, chi_min, chi_mean, chi_max}
  // - mesh/{ale_rezone_invocations}
  // - safety/clamp_count

  // SPECIFICATION §7.3 notes:
  // - Easy 1:1 key-name mismatches are aligned to spec keys below.
  // - Legacy/internal diagnostics (E_total, dE_total, E_denom, mmatrix_*) are
  //   still written as extra datasets.
  append_scalar_double_compat(
      file, "energy/internal_electron", "energy/E_int_e", rec.snapshot.energy.E_int_e, "erg");
  append_scalar_double_compat(
      file, "energy/internal_ion", "energy/E_int_i", rec.snapshot.energy.E_int_i, "erg");
  append_scalar_double_compat(file, "energy/kinetic", "energy/E_kin", rec.snapshot.energy.E_kin, "erg");
  append_scalar_double_compat(
      file, "energy/radiation_field", "energy/E_rad", rec.snapshot.energy.E_rad, "erg");
  append_scalar_double_compat(
      file, "energy/laser_incident", "energy/E_laser_in", rec.E_laser_incident, "erg");
  append_scalar_double_compat(
      file, "energy/marshak_in", "energy/E_Marshak_in", rec.snapshot.energy.E_Marshak_in, "erg");
  append_scalar_double(file, "energy/E_volume_in", rec.snapshot.energy.E_volume_in, "erg");
  append_scalar_double(file, "energy/laser_deposited", rec.E_laser_deposited, "erg");
  append_scalar_double(file, "energy/E_cbet_iaw_step", rec.E_cbet_iaw_step, "erg");
  append_scalar_double(file, "energy/E_cbet_iaw", rec.E_cbet_iaw, "erg");
  append_scalar_double_compat(
      file, "energy/laser_escaped", "energy/E_laser_esc", rec.E_laser_escaped, "erg");
  append_scalar_double(file, "energy/laser_ra_deposited", rec.E_ra_deposited, "erg");
  append_scalar_double_compat(file,
                              "energy/radiation_escaped",
                              "energy/E_rad_esc",
                              rec.snapshot.energy.E_rad_esc,
                              "erg");
  append_scalar_double_compat(file,
                              "energy/numerical_loss",
                              "energy/E_numerical_loss",
                              rec.snapshot.energy.E_numerical_loss,
                              "erg");
  append_scalar_double_compat(
      file, "energy/pdv_boundary", "energy/E_pdV_bdry", rec.snapshot.energy.E_pdV_bdry, "erg");
  append_scalar_double_compat(
      file, "energy/floor_injected", "energy/E_floor", rec.snapshot.energy.E_floor, "erg");
  append_scalar_double_compat(
      file, "energy/safety_injected", "energy/E_safety", rec.snapshot.energy.E_safety, "erg");
  append_scalar_double_compat(file,
                              "energy/redistribution_unresolved",
                              "energy/E_redistribution_unresolved",
                              rec.snapshot.energy.E_redistribution_unresolved,
                              "erg");
  append_scalar_double_compat(
      file, "energy/solver_residual", "energy/E_solver", rec.snapshot.energy.E_solver, "erg");
  append_scalar_double(file, "energy/E_total", rec.snapshot.energy.E_total, "erg");
  append_scalar_double(file, "energy/dE_total", rec.snapshot.energy.dE_total, "erg");
  append_scalar_double_compat(file,
                              "energy/conservation_error",
                              "energy/epsilon_budget",
                              rec.snapshot.energy.epsilon_budget,
                              "dimensionless");
  append_scalar_double(file, "energy/E_denom", rec.snapshot.energy.E_denom, "erg");
  if (rec.snapshot.conservation_residuals.valid) {
    const std::string base = "/diagnostics/conservation/v1/";
    append_scalar_double(file,
                         base + "mass_residual",
                         rec.snapshot.conservation_residuals.mass,
                         "dimensionless");
    append_scalar_double(file,
                         base + "R_momentum_residual",
                         rec.snapshot.conservation_residuals.r_momentum,
                         "dimensionless");
    append_scalar_double(file,
                         base + "Z_momentum_residual",
                         rec.snapshot.conservation_residuals.z_momentum,
                         "dimensionless");
    append_scalar_double(file,
                         base + "gcl_residual",
                         rec.snapshot.conservation_residuals.vol_closure,
                         "dimensionless");
  }
  if (rec.c1_solver_steps_total > 0) {
    const std::string base = "/diagnostics/conduction/v1/";
    append_scalar_double(file, base + "time_s", rec.t, "s");
    append_scalar_i32(file, base + "step", static_cast<std::int32_t>(rec.step), "count");
    append_scalar_double(file,
                         base + "solver_residual",
                         rec.c1_solver_residual_last,
                         "dimensionless");
    append_scalar_double(file,
                         base + "solver_residual_max",
                         rec.c1_solver_residual_max,
                         "dimensionless");
    append_scalar_i32(file,
                      base + "solver_iter",
                      static_cast<std::int32_t>(rec.c1_solver_iter_last),
                      "count");
    append_scalar_i32(file,
                      base + "solver_iter_max",
                      static_cast<std::int32_t>(rec.c1_solver_iter_max),
                      "count");
    append_scalar_double(file,
                         base + "solver_cond_number",
                         rec.c1_solver_cond_number_last,
                         "dimensionless");
    append_scalar_double(file,
                         base + "solver_cond_number_max",
                         rec.c1_solver_cond_number_max,
                         "dimensionless");
    append_scalar_double(file,
                         base + "bc_flux_r_inner",
                         rec.c1_bc_heat_flux_integrated[0],
                         "erg");
    append_scalar_double(file,
                         base + "bc_flux_r_outer",
                         rec.c1_bc_heat_flux_integrated[1],
                         "erg");
    append_scalar_double(file,
                         base + "bc_flux_z_bottom",
                         rec.c1_bc_heat_flux_integrated[2],
                         "erg");
    append_scalar_double(file,
                         base + "bc_flux_z_top",
                         rec.c1_bc_heat_flux_integrated[3],
                         "erg");
  }
  if (rec.burn_enabled_any) {
    append_scalar_double(file, "burn/released", rec.burn_released_step, "erg");
    append_scalar_double(file, "burn/dep_e", rec.burn_dep_e_step, "erg");
    append_scalar_double(file, "burn/dep_i", rec.burn_dep_i_step, "erg");
    append_scalar_double(file, "burn/esc_charged", rec.burn_esc_charged_step, "erg");
    append_scalar_double(file, "burn/esc_neutron", rec.burn_esc_neutron_step, "erg");
    append_scalar_double(file, "burn/neutron_dep_e", rec.burn_nh_dep_e_step, "erg");
    append_scalar_double(file, "burn/neutron_dep_i", rec.burn_nh_dep_i_step, "erg");
    append_scalar_double(file, "burn/neutron_degraded", rec.burn_nh_degraded_step, "erg");
    append_scalar_double(file, "burn/neutron_escaped", rec.burn_nh_escaped_step, "erg");
    append_scalar_double(file, "burn/E_released_cum", rec.E_burn_released, "erg");
    append_scalar_double(file, "burn/E_dep_e_cum", rec.E_burn_dep_e, "erg");
    append_scalar_double(file, "burn/E_dep_i_cum", rec.E_burn_dep_i, "erg");
    append_scalar_double(file, "burn/E_esc_charged_cum", rec.E_burn_esc_charged, "erg");
    append_scalar_double(file, "burn/E_esc_neutron_cum", rec.E_burn_esc_neutron, "erg");
    if (rec.burn_diffusion_any || rec.burn_mc_any) {
      append_scalar_double(file, "burn/E_inflight", rec.E_burn_inflight, "erg");
    }
    append_scalar_double(file, "burn/N_neutrons_dt_cum", rec.N_burn_neutrons_dt, "1");
    append_scalar_double(file, "burn/N_neutrons_dd_cum", rec.N_burn_neutrons_dd, "1");
    append_scalar_double(file, "burn/neutron_Ti_burn_dt",
                         rec.burn_Ti_burn_dt_eV, "eV");
    append_scalar_double(file, "burn/neutron_Ti_burn_dd",
                         rec.burn_Ti_burn_dd_eV, "eV");
    append_scalar_double(file, "burn/neutron_mean_shift_dt",
                         rec.burn_neutron_mean_shift_dt_keV, "keV");
    append_scalar_double(file, "burn/neutron_sigma_thermal_dt",
                         rec.burn_neutron_sigma_thermal_dt_keV, "keV");
    append_scalar_double(file, "burn/neutron_sigma_total_dt",
                         rec.burn_neutron_sigma_total_dt_keV, "keV");
    append_scalar_double(file, "burn/neutron_mean_shift_dd",
                         rec.burn_neutron_mean_shift_dd_keV, "keV");
    append_scalar_double(file, "burn/neutron_sigma_thermal_dd",
                         rec.burn_neutron_sigma_thermal_dd_keV, "keV");
    append_scalar_double(file, "burn/neutron_sigma_total_dd",
                         rec.burn_neutron_sigma_total_dd_keV, "keV");
    append_scalar_double(file, "burn/dt_limit", rec.burn_dt_limit_s, "s");
  }
  if (rec.snb_steps_total > 0) {
    const std::string base = "/diagnostics/conduction/snb/v1/";
    append_scalar_double(file, base + "time_s", rec.t, "s");
    append_scalar_i32(file, base + "step", static_cast<std::int32_t>(rec.step), "count");
    append_scalar_i32(file, base + "picard_iters_last",
                      static_cast<std::int32_t>(rec.snb_picard_iters_last), "count");
    append_scalar_i32(file, base + "picard_iters_max",
                      static_cast<std::int32_t>(rec.snb_picard_iters_max), "count");
    append_scalar_i32(file, base + "nonconverged_steps",
                      static_cast<std::int32_t>(rec.snb_nonconverged_steps), "count");
    // 2D residual is a pair POWER (erg/s), not the 1D face flux; this tree
    // is 2D-only for SNB (1D fail-closed) — per-dimension reconciliation at
    // the feature/1d-brushup merge (2d_snbtr closure).
    append_scalar_double(file, base + "picard_resid_last",
                         rec.snb_picard_resid_last, "erg/s");
    append_scalar_i32(file, base + "cap_faces_99_last",
                      static_cast<std::int32_t>(rec.snb_cap_faces_99_last), "count");
    append_scalar_i32(file, base + "cap_faces_50_last",
                      static_cast<std::int32_t>(rec.snb_cap_faces_50_last), "count");
    append_scalar_double(file, base + "cap_theta_min_last",
                         rec.snb_cap_theta_min_last, "dimensionless");
    append_scalar_double(file, base + "cap_theta_min_run",
                         rec.snb_cap_theta_min_run, "dimensionless");
    append_scalar_double(file, base + "dq_over_qsh_max_last",
                         rec.snb_dq_over_qsh_max_last, "dimensionless");
    append_scalar_double(file, base + "dq_over_qsh_max_run",
                         rec.snb_dq_over_qsh_max_run, "dimensionless");
    append_scalar_i32(file, base + "solver_iters",
                      static_cast<std::int32_t>(rec.snb_solver_iters), "count");
    append_scalar_double(file, base + "solver_resid",
                         rec.snb_solver_resid, "dimensionless");
  }
  if (rec.hot_e_enabled_any) {
    append_scalar_double(file, "laser/hot_e_in", rec.hot_e_in_step, "erg");
    append_scalar_double(file, "laser/hot_e_deposited", rec.hot_e_deposited_step, "erg");
    append_scalar_double(file, "laser/hot_e_residual", rec.hot_e_residual_step, "erg");
    append_scalar_double(file, "laser/hot_e_escaped", rec.hot_e_escaped_step, "erg");
    append_scalar_double(file, "laser/hot_e_source_r", rec.hot_e_source_r, "cm");
    append_scalar_double(file, "laser/hot_e_conservation_resid",
                         rec.hot_e_conservation_resid, "1");
    append_scalar_double(file, "laser/hot_e_E_deposited_cum", rec.E_hot_e_deposited, "erg");
    append_scalar_double(file, "laser/hot_e_E_escaped_cum", rec.E_hot_e_escaped, "erg");
    if (rec.hot_e_ch_in_step.size() > 1) {
      for (std::size_t ch = 0; ch < rec.hot_e_ch_in_step.size(); ++ch) {
        const std::string ch_prefix = "laser/hot_e_ch" + std::to_string(ch) + "_";
        append_scalar_double(file, ch_prefix + "in", rec.hot_e_ch_in_step[ch], "erg");
        append_scalar_double(file, ch_prefix + "deposited",
                             rec.hot_e_ch_deposited_step[ch], "erg");
        append_scalar_double(file, ch_prefix + "escaped",
                             rec.hot_e_ch_escaped_step[ch], "erg");
      }
    }
  }
  if (rec.phase_resolved_energy_enabled && rec.snapshot.phase_energy.valid) {
    const auto& phase = rec.snapshot.phase_energy;
    append_scalar_double(
        file, "energy/phase_diagnostic/E_pre_hydro", phase.E_pre_hydro, "erg");
    append_scalar_double(
        file, "energy/phase_diagnostic/E_post_hydro", phase.E_post_hydro, "erg");
    append_scalar_double(
        file, "energy/phase_diagnostic/E_post_ale", phase.E_post_ale, "erg");
    append_scalar_double(file, "energy/phase_diagnostic/dE_hydro", phase.dE_hydro, "erg");
    append_scalar_double(file, "energy/phase_diagnostic/dE_ale", phase.dE_ale, "erg");
    append_scalar_double(file,
                         "energy/phase_diagnostic/internal_pre_hydro",
                         phase.internal_pre_hydro,
                         "erg");
    append_scalar_double(file,
                         "energy/phase_diagnostic/internal_post_hydro",
                         phase.internal_post_hydro,
                         "erg");
    append_scalar_double(file,
                         "energy/phase_diagnostic/internal_post_ale",
                         phase.internal_post_ale,
                         "erg");
    append_scalar_double(file,
                         "energy/phase_diagnostic/dE_hydro_internal",
                         phase.dE_hydro_internal,
                         "erg");
    append_scalar_double(file,
                         "energy/phase_diagnostic/dE_ale_internal",
                         phase.dE_ale_internal,
                         "erg");
    append_scalar_double(file,
                         "energy/phase_diagnostic/kinetic_pre_hydro",
                         phase.kinetic_pre_hydro,
                         "erg");
    append_scalar_double(file,
                         "energy/phase_diagnostic/kinetic_post_hydro",
                         phase.kinetic_post_hydro,
                         "erg");
    append_scalar_double(file,
                         "energy/phase_diagnostic/kinetic_post_ale",
                         phase.kinetic_post_ale,
                         "erg");
    append_scalar_double(file,
                         "energy/phase_diagnostic/dE_hydro_kinetic",
                         phase.dE_hydro_kinetic,
                         "erg");
    append_scalar_double(file,
                         "energy/phase_diagnostic/dE_ale_kinetic",
                         phase.dE_ale_kinetic,
                         "erg");
  }
  if (rec.ale_closure_audit_enabled && rec.snapshot.ale_closure_audit.valid) {
    const auto& audit = rec.snapshot.ale_closure_audit;
    const std::string base = "energy/ale_closure_audit/";
    append_scalar_i64(file, base + "cycle", static_cast<std::int64_t>(rec.step), "count");
    append_scalar_double(file, base + "t", rec.t, "s");
    append_scalar_double(file, base + "K0_cellcorner", audit.K0_cellcorner, "erg");
    append_scalar_double(
        file, base + "K0_node_from_corner", audit.K0_node_from_corner, "erg");
    append_scalar_double(file, base + "K0_budget", audit.K0_budget, "erg");
    append_scalar_double(file, base + "I0", audit.I0, "erg");
    append_scalar_double(file, base + "K0_scalar_total", audit.K0_scalar_total, "erg");
    append_scalar_double(file, base + "K_remap_total", audit.K_remap_total, "erg");
    append_scalar_double(file, base + "I_raw", audit.I_raw, "erg");
    append_scalar_double(file, base + "K_cellmom", audit.K_cellmom, "erg");
    append_scalar_double(file, base + "K_node_preBC", audit.K_node_preBC, "erg");
    append_scalar_double(file, base + "K_node_postBC", audit.K_node_postBC, "erg");
    append_scalar_double(file, base + "K_post_budget", audit.K_post_budget, "erg");
    append_scalar_double(file, base + "sum_dI_raw", audit.sum_dI_raw, "erg");
    append_scalar_double(
        file, base + "sum_dI_after_floor", audit.sum_dI_after_floor, "erg");
    append_scalar_double(file, base + "F_floor", audit.F_floor, "erg");
    append_scalar_i64(file,
                      base + "n_cells_negative_dI",
                      static_cast<std::int64_t>(audit.n_cells_negative_dI),
                      "cells");
    append_scalar_i64(file,
                      base + "n_cells_floor_e",
                      static_cast<std::int64_t>(audit.n_cells_floor_e),
                      "cells");
    append_scalar_i64(file,
                      base + "n_cells_floor_i",
                      static_cast<std::int64_t>(audit.n_cells_floor_i),
                      "cells");
    append_scalar_double(
        file, base + "min_tentative_e_e", audit.min_tentative_e_e, "erg/g");
    append_scalar_double(
        file, base + "min_tentative_e_i", audit.min_tentative_e_i, "erg/g");
    append_scalar_double(file,
                         base + "residual_K0_cellcorner_node",
                         audit.residual_K0_cellcorner_node,
                         "erg");
    append_scalar_double(file,
                         base + "residual_K0_node_budget",
                         audit.residual_K0_node_budget,
                         "erg");
    append_scalar_double(file, base + "residual_K_remap", audit.residual_K_remap, "erg");
    append_scalar_double(file, base + "residual_I_remap", audit.residual_I_remap, "erg");
    append_scalar_double(file,
                         base + "residual_closure_target",
                         audit.residual_closure_target,
                         "erg");
    append_scalar_double(file, base + "dE_ale_predicted", audit.dE_ale_predicted, "erg");
    append_scalar_double(file, base + "dE_ale_total", audit.dE_ale_total, "erg");
    append_scalar_double(file, base + "residual_dE_ale", audit.residual_dE_ale, "erg");
    append_scalar_i64(file,
                      base + "mechanism_code",
                      static_cast<std::int64_t>(audit.mechanism_code),
                      "enum");
    append_scalar_i64(file,
                      base + "flag_K_remap_defect",
                      audit.k_remap_defect_dominant ? 1 : 0,
                      "bool");
    append_scalar_i64(
        file, base + "flag_floor_dominant", audit.floor_dominant ? 1 : 0, "bool");
    append_scalar_i64(file,
                      base + "flag_diagnostic_mismatch",
                      audit.diagnostic_mismatch_dominant ? 1 : 0,
                      "bool");
  }
  append_scalar_i64(
      file, "safety/clamp_count", static_cast<std::int64_t>(rec.snapshot.clamp_count), "count");
  append_scalar_i64(file,
                    "mesh/ale_rezone_invocations",
                    static_cast<std::int64_t>(rec.ale_rezone_invocations),
                    "count");

  if (rec.plasma_diag.valid) {
    append_scalar_double(
        file, "plasma/Zbar_mean", rec.plasma_diag.zbar_mean, "dimensionless");
    append_scalar_double(
        file, "plasma/Zbar_max", rec.plasma_diag.zbar_max, "dimensionless");
  }

  if (rec.implosion_diag.valid) {
    append_scalar_double(
        file, "implosion/rho_peak", rec.implosion_diag.rho_peak, "g/cm3");
    append_scalar_double(
        file, "implosion/shell_radius_mean", rec.implosion_diag.shell_radius_mean, "cm");
    append_scalar_double(
        file, "implosion/shell_radius_min", rec.implosion_diag.shell_radius_min, "cm");
    append_scalar_double(
        file, "implosion/center_temperature", rec.implosion_diag.center_temperature, "eV");
  }
  if (rec.icf_enabled) {
    write_icf_shell_history(file, rec.snapshot.icf_shell, rec.t, rec.step);
  }
  if (rec.hotspot_gas_enabled) {
    write_hotspot_gas_history(file, rec.snapshot.hotspot_gas, cfg_, rec.t, rec.step);
  }
  if (rec.has_ale_provenance) {
    if (rec.ale_provenance_enabled) {
      write_ale_provenance_history(file,
                                   rec.ale_provenance_values,
                                   rec.t,
                                   rec.step);
      write_material_interface_history(file,
                                       rec.ale_provenance_values,
                                       rec.t,
                                       rec.step);
    }
    if (rec.mesh_quality_min_enabled) {
      write_mesh_quality_min_history(file,
                                     rec.ale_provenance_values,
                                     rec.t,
                                     rec.step);
    }
  }
  if (rec.conservation_enabled) {
    write_operator_energy_residuals_history(file,
                                            rec.snapshot.operator_residuals,
                                            rec.t,
                                            rec.step);
  }

  const std::size_t n_rhoR = std::min(rec.snapshot.areal_density.angles_deg.size(),
                                      rec.snapshot.areal_density.rhoR.size());
  if (n_rhoR > 0) {
    std::vector<double> rhoR_row(rec.snapshot.areal_density.rhoR.begin(),
                                 rec.snapshot.areal_density.rhoR.begin() + n_rhoR);
    std::vector<double> angles(rec.snapshot.areal_density.angles_deg.begin(),
                               rec.snapshot.areal_density.angles_deg.begin() + n_rhoR);
    append_row_double_matrix(file, "implosion/rho_R", rhoR_row, "g/cm2");
    const hid_t rhoR_dset = H5Dopen2(file, "implosion/rho_R", H5P_DEFAULT);
    TENRYU_ASSERT(rhoR_dset >= 0, "HistoryWriter H5Dopen2(implosion/rho_R) failed");
    write_double_vector_attribute_if_missing(rhoR_dset, "angles_deg", angles);
    warn_h5_close_failure(H5Dclose(rhoR_dset), "H5Dclose",
                          "HistoryWriter::append(implosion/rho_R dataset)");
  }
  const std::size_t n_rhoR_hotspot =
      std::min(rec.snapshot.areal_density.angles_deg.size(),
               rec.snapshot.areal_density.rhoR_hotspot_tracer.size());
  if (n_rhoR_hotspot > 0) {
    std::vector<double> rhoR_row(
        rec.snapshot.areal_density.rhoR_hotspot_tracer.begin(),
        rec.snapshot.areal_density.rhoR_hotspot_tracer.begin() + n_rhoR_hotspot);
    std::vector<double> angles(
        rec.snapshot.areal_density.angles_deg.begin(),
        rec.snapshot.areal_density.angles_deg.begin() + n_rhoR_hotspot);
    append_row_double_matrix(
        file, "implosion/rho_R_hotspot_tracer", rhoR_row, "g/cm2");
    const hid_t rhoR_dset =
        H5Dopen2(file, "implosion/rho_R_hotspot_tracer", H5P_DEFAULT);
    TENRYU_ASSERT(rhoR_dset >= 0,
                  "HistoryWriter H5Dopen2(implosion/rho_R_hotspot_tracer) failed");
    write_double_vector_attribute_if_missing(rhoR_dset, "angles_deg", angles);
    warn_h5_close_failure(
        H5Dclose(rhoR_dset),
        "H5Dclose",
        "HistoryWriter::append(implosion/rho_R_hotspot_tracer dataset)");
  }
  const std::size_t n_rhoR_fuel =
      std::min(rec.snapshot.areal_density.angles_deg.size(),
               rec.snapshot.areal_density.rhoR_fuel_tracer.size());
  if (n_rhoR_fuel > 0) {
    std::vector<double> rhoR_row(
        rec.snapshot.areal_density.rhoR_fuel_tracer.begin(),
        rec.snapshot.areal_density.rhoR_fuel_tracer.begin() + n_rhoR_fuel);
    std::vector<double> angles(
        rec.snapshot.areal_density.angles_deg.begin(),
        rec.snapshot.areal_density.angles_deg.begin() + n_rhoR_fuel);
    append_row_double_matrix(file, "implosion/rho_R_fuel_tracer", rhoR_row, "g/cm2");
    const hid_t rhoR_dset =
        H5Dopen2(file, "implosion/rho_R_fuel_tracer", H5P_DEFAULT);
    TENRYU_ASSERT(rhoR_dset >= 0,
                  "HistoryWriter H5Dopen2(implosion/rho_R_fuel_tracer) failed");
    write_double_vector_attribute_if_missing(rhoR_dset, "angles_deg", angles);
    warn_h5_close_failure(
        H5Dclose(rhoR_dset),
        "H5Dclose",
        "HistoryWriter::append(implosion/rho_R_fuel_tracer dataset)");
  }
  // Keep legacy per-angle scalar series for backward compatibility.
  for (std::size_t i = 0; i < n_rhoR; ++i) {
    append_scalar_double(file,
                         "implosion/rhoR_deg_" +
                             angle_tag(rec.snapshot.areal_density.angles_deg[i]),
                         rec.snapshot.areal_density.rhoR[i],
                         "g/cm2");
  }
  for (std::size_t i = 0; i < n_rhoR_hotspot; ++i) {
    append_scalar_double(file,
                         "implosion/rhoR_hotspot_tracer_deg_" +
                             angle_tag(rec.snapshot.areal_density.angles_deg[i]),
                         rec.snapshot.areal_density.rhoR_hotspot_tracer[i],
                         "g/cm2");
  }
  for (std::size_t i = 0; i < n_rhoR_fuel; ++i) {
    append_scalar_double(file,
                         "implosion/rhoR_fuel_tracer_deg_" +
                             angle_tag(rec.snapshot.areal_density.angles_deg[i]),
                         rec.snapshot.areal_density.rhoR_fuel_tracer[i],
                         "g/cm2");
  }

  const std::size_t n_modes = std::min(rec.snapshot.sphericity.modes.size(),
                                       rec.snapshot.sphericity.coefficients.size());
  if (n_modes > 0) {
    std::vector<double> P_ell(rec.snapshot.sphericity.coefficients.begin(),
                              rec.snapshot.sphericity.coefficients.begin() + n_modes);
    std::vector<int> ell_values(rec.snapshot.sphericity.modes.begin(),
                                rec.snapshot.sphericity.modes.begin() + n_modes);
    append_row_double_matrix(file, "modes/P_ell", P_ell, "cm");
    write_i32_vector_dataset_if_missing(file, "modes/ell_values", ell_values);
  }
  // Keep legacy per-mode scalar series for backward compatibility.
  for (std::size_t i = 0; i < n_modes; ++i) {
    append_scalar_double(file,
                         "modes/a" + std::to_string(rec.snapshot.sphericity.modes[i]),
                         rec.snapshot.sphericity.coefficients[i],
                         "cm");
  }

  append_scalar_double(file, "laser/critical_surface_r", rec.snapshot.laser_pattern.critical_surface_r,
                       "cm");
  append_scalar_double(file, "laser/near_critical_r", rec.snapshot.laser_pattern.near_critical_r,
                       "cm");
  append_scalar_double(file,
                       "laser/absorption_weighted_r",
                       rec.snapshot.laser_pattern.absorption_weighted_r,
                       "cm");
  append_scalar_double(file, "laser/absorbed_total", rec.snapshot.laser_pattern.absorbed_total,
                       "erg");
  append_scalar_double(file,
                       "laser/absorbed_power_total",
                       rec.snapshot.laser_pattern.absorbed_power_total,
                       "erg/s");
  append_scalar_double(file,
                       "laser/commanded_power_total",
                       rec.snapshot.laser_pattern.commanded_power_total,
                       "erg/s");
  append_scalar_double(file,
                       "laser/trace_unabsorbed_power_total",
                       rec.snapshot.laser_pattern.trace_unabsorbed_power_total,
                       "erg/s");
  append_scalar_double(file,
                       "laser/unabsorbed_power_total",
                       rec.snapshot.laser_pattern.unabsorbed_power_total,
                       "erg/s");
  append_scalar_double(file,
                       "laser/trace_absorption_efficiency_total",
                       rec.snapshot.laser_pattern.trace_absorption_efficiency_total,
                       "dimensionless");
  append_scalar_double(file,
                       "laser/absorption_efficiency_total",
                       rec.snapshot.laser_pattern.absorption_efficiency_total,
                       "dimensionless");
  append_scalar_double(file,
                       "laser/transfer_blocked_power_total",
                       rec.snapshot.laser_pattern.transfer_blocked_power_total,
                       "erg/s");
  append_scalar_i64(file,
                    "laser/tail_closure_count",
                    rec.snapshot.laser_pattern.tail_closure_count,
                    "count");
  append_scalar_double(file,
                       "laser/tail_closure_absorbed_power_total",
                       rec.snapshot.laser_pattern.tail_closure_absorbed_power_total,
                       "erg/s");
  append_scalar_double(file,
                       "laser/cbet_exchanged_power_total",
                       rec.snapshot.laser_pattern.cbet_exchanged_power_total,
                       "erg/s");
  append_scalar_double(file,
                       "laser/cbet_ledger_residual_rel",
                       rec.snapshot.laser_pattern.cbet_ledger_residual_rel,
                       "dimensionless");
  append_scalar_i64(file,
                    "laser/cbet_iterations",
                    rec.snapshot.laser_pattern.cbet_iterations,
                    "count");
  append_scalar_i64(file,
                    "laser/cbet_clamp_count",
                    rec.snapshot.laser_pattern.cbet_clamp_count,
                    "count");
  append_scalar_i64(file,
                    "laser/critical_surface_hit_count",
                    rec.snapshot.laser_pattern.critical_surface_hit_count,
                    "count");
  append_scalar_double(file,
                       "laser/corona_transition_blend",
                       rec.snapshot.laser_pattern.corona_transition_blend,
                       "dimensionless");
  append_scalar_i64(file,
                    "laser/corona_transition_resolved_cells",
                    rec.snapshot.laser_pattern.corona_transition_resolved_cells,
                    "cells");
  for (std::size_t b = 0; b < rec.snapshot.laser_pattern.absorbed_fraction_per_beam.size();
       ++b) {
    append_scalar_double(file,
                         "laser/absorbed_fraction_beam_" + std::to_string(b),
                         rec.snapshot.laser_pattern.absorbed_fraction_per_beam[b],
                         "dimensionless");
  }

  append_scalar_i64(file,
                    "radiation/fld_outer_iterations",
                    rec.snapshot.fld_solver.outer_iterations,
                    "count");
  append_scalar_double(file,
                       "radiation/fld_outer_residual",
                       rec.snapshot.fld_solver.outer_residual,
                       "dimensionless");
  append_scalar_i64(file,
                    "radiation/fld_outer_converged",
                    static_cast<std::int64_t>(rec.snapshot.fld_solver.outer_converged),
                    "flag");

  // Legacy runs can have mode counts stored in mc/n_imc and mc/n_ddmc.
  const bool imc_mode_uses_legacy_path =
      (tenryu::io::h5_link_exists(file, "mc/n_imc", H5P_DEFAULT) > 0) &&
      (tenryu::io::h5_link_exists(file, "mc/imc_mode_count", H5P_DEFAULT) <= 0);
  const bool ddmc_mode_uses_legacy_path =
      (tenryu::io::h5_link_exists(file, "mc/n_ddmc", H5P_DEFAULT) > 0) &&
      (tenryu::io::h5_link_exists(file, "mc/ddmc_mode_count", H5P_DEFAULT) <= 0);
  if (imc_mode_uses_legacy_path || ddmc_mode_uses_legacy_path) {
    static bool warned_legacy_mode_paths = false;
    if (!warned_legacy_mode_paths) {
      core::log_warning("HistoryWriter: detected legacy mc/n_imc or mc/n_ddmc mode-count "
                        "dataset; particle counts for conflicting paths are skipped to "
                        "preserve append compatibility");
      warned_legacy_mode_paths = true;
    }
  }

  // Mode-map counts (legacy metrics): write to new names, fallback to old names when appending
  // pre-rename history files.
  append_scalar_i64_compat(
      file, "mc/ddmc_mode_count", "mc/n_ddmc", rec.snapshot.mc.ddmc_mode_count, "count");
  append_scalar_i64_compat(
      file, "mc/imc_mode_count", "mc/n_imc", rec.snapshot.mc.imc_mode_count, "count");

  // SPEC fields are always written to preserve fixed schema; disabled channels write zero.
  const bool write_spec_values = rec.mc_group_enabled;
  const bool particle_counts_enabled = write_spec_values && rec.mc_particle_counts_enabled;
  const bool weight_stats_enabled = write_spec_values && rec.mc_weight_stats_enabled;
  const bool ddmc_fraction_enabled = write_spec_values && rec.mc_ddmc_fraction_enabled;

  const std::int64_t n_total = particle_counts_enabled ? rec.snapshot.mc.n_total : 0;
  const std::int64_t n_imc = particle_counts_enabled ? rec.snapshot.mc.n_imc_particles : 0;
  const std::int64_t n_ddmc = particle_counts_enabled ? rec.snapshot.mc.n_ddmc_particles : 0;
  const std::int64_t n_census = particle_counts_enabled ? rec.snapshot.mc.n_census : 0;
  const std::int64_t n_absorbed = particle_counts_enabled ? rec.snapshot.mc.n_absorbed : 0;
  const std::int64_t n_escaped = particle_counts_enabled ? rec.snapshot.mc.n_escaped : 0;
  const std::int64_t n_leaked = particle_counts_enabled ? rec.snapshot.mc.n_leaked : 0;
  const double ddmc_fraction = ddmc_fraction_enabled ? rec.snapshot.mc.ddmc_fraction : 0.0;
  const double weight_min = weight_stats_enabled ? rec.snapshot.mc.weight_min : 0.0;
  const double weight_mean = weight_stats_enabled ? rec.snapshot.mc.weight_mean : 0.0;
  const double weight_max = weight_stats_enabled ? rec.snapshot.mc.weight_max : 0.0;
  const std::int64_t overshoot_count = write_spec_values ? rec.snapshot.mc.overshoot_count : 0;
  const double overshoot_max = write_spec_values ? rec.snapshot.mc.overshoot_max : 0.0;

  append_scalar_i64(file, "mc/n_total", n_total, "count");
  if (!imc_mode_uses_legacy_path) {
    append_scalar_i64(file, "mc/n_imc", n_imc, "count");
  }
  if (!ddmc_mode_uses_legacy_path) {
    append_scalar_i64(file, "mc/n_ddmc", n_ddmc, "count");
  }
  append_scalar_i64(file, "mc/n_census", n_census, "count");
  append_scalar_i64(file, "mc/n_absorbed", n_absorbed, "count");
  append_scalar_i64(file, "mc/n_escaped", n_escaped, "count");
  append_scalar_i64(file, "mc/n_leaked", n_leaked, "count");
  append_scalar_double(file, "mc/ddmc_fraction", ddmc_fraction, "dimensionless");
  append_scalar_double(file, "mc/weight_min", weight_min, "dimensionless");
  append_scalar_double(file, "mc/weight_mean", weight_mean, "dimensionless");
  append_scalar_double(file, "mc/weight_max", weight_max, "dimensionless");
  append_scalar_i64(file, "mc/overshoot_count", overshoot_count, "count");
  append_scalar_double(file, "mc/overshoot_max", overshoot_max, "dimensionless");

  // Legacy diagnostics remain unconditional for backward compatibility.
  append_scalar_i64(file, "mc/mmatrix_violations", rec.snapshot.mc.mmatrix_violations, "count");
  append_scalar_i64(file,
                    "mc/mmatrix_fallback_count",
                    rec.snapshot.mc.mmatrix_fallback_count,
                    "count");
  append_scalar_i64(file,
                    "mc/omega_below_threshold",
                    rec.snapshot.mc.omega_below_threshold,
                    "count");
  append_scalar_i64(file,
                    "mc/interface_transitions",
                    rec.snapshot.mc.interface_transitions,
                    "count");
  append_scalar_i64(file,
                    "mc/interface_reflections",
                    rec.snapshot.mc.interface_reflections,
                    "count");
  append_scalar_i64(file,
                    "mc/conversion_prob_violations",
                    rec.snapshot.mc.conversion_prob_violations,
                    "count");
  append_scalar_i64(file,
                    "mc/ddmc_to_imc_conversions",
                    rec.snapshot.mc.ddmc_to_imc_conversions,
                    "count");
  append_scalar_double(file,
                       "mc/rad_momentum_deposition",
                       rec.snapshot.mc.rad_momentum_deposition,
                       "g*cm/s");

  append_scalar_i64(file,
                    "difference/reference_valid",
                    rec.snapshot.mc.difference_reference_valid,
                    "count");
  append_scalar_i64(file,
                    "difference/eligible_cells",
                    rec.snapshot.mc.difference_eligible_cells,
                    "cells");
  append_scalar_i64(file,
                    "difference/active_cells",
                    rec.snapshot.mc.difference_active_cells,
                    "cells");
  append_scalar_i64(file,
                    "difference/strong_cells",
                    rec.snapshot.mc.difference_strong_cells,
                    "cells");
  append_scalar_i64(file,
                    "difference/hybrid_suppressed_cells",
                    rec.snapshot.mc.difference_hybrid_suppressed_cells,
                    "cells");
  append_scalar_double(file, "difference/W_min", rec.snapshot.mc.difference_W_min, "dimensionless");
  append_scalar_double(file,
                       "difference/W_mean",
                       rec.snapshot.mc.difference_W_mean,
                       "dimensionless");
  append_scalar_double(file, "difference/W_max", rec.snapshot.mc.difference_W_max, "dimensionless");
  append_scalar_double(file,
                       "difference/tau_min",
                       rec.snapshot.mc.difference_tau_min,
                       "dimensionless");
  append_scalar_double(file,
                       "difference/tau_mean",
                       rec.snapshot.mc.difference_tau_mean,
                       "dimensionless");
  append_scalar_double(file,
                       "difference/tau_max",
                       rec.snapshot.mc.difference_tau_max,
                       "dimensionless");
  append_scalar_double(file,
                       "difference/chi_mean",
                       rec.snapshot.mc.difference_chi_mean,
                       "dimensionless");
  append_scalar_double(file,
                       "difference/chi_max",
                       rec.snapshot.mc.difference_chi_max,
                       "dimensionless");
  append_scalar_double(file,
                       "difference/reduced_flux_max",
                       rec.snapshot.mc.difference_reduced_flux_max,
                       "dimensionless");
  append_scalar_double(file,
                       "difference/knudsen_max",
                       rec.snapshot.mc.difference_knudsen_max,
                       "dimensionless");
  append_scalar_double(file,
                       "difference/front_grad_Te_max",
                       rec.snapshot.mc.difference_front_grad_Te_max,
                       "dimensionless");
  append_scalar_double(file,
                       "difference/front_grad_rho_max",
                       rec.snapshot.mc.difference_front_grad_rho_max,
                       "dimensionless");
  append_scalar_double(file,
                       "difference/E_ref_total",
                       rec.snapshot.mc.difference_E_ref_total,
                       "erg");
  append_scalar_i64(file, "holo/n_core_cells", rec.snapshot.mc.holo_n_core_cells, "cells");
  append_scalar_i64(file, "holo/n_entered", rec.snapshot.mc.holo_n_entered, "cells");
  append_scalar_i64(file, "holo/n_exited", rec.snapshot.mc.holo_n_exited, "cells");
  append_scalar_i64(file,
                    "holo/n_hard_exited",
                    rec.snapshot.mc.holo_n_hard_exited,
                    "cells");
  append_scalar_i64(file,
                    "holo/n_island_rejected",
                    rec.snapshot.mc.holo_n_island_rejected,
                    "cells");
  append_scalar_double(file, "holo/tau_R_min", rec.snapshot.mc.holo_tau_R_min, "dimensionless");
  append_scalar_double(file, "holo/tau_R_max", rec.snapshot.mc.holo_tau_R_max, "dimensionless");
  append_scalar_double(file,
                       "holo/reduced_flux_max",
                       rec.snapshot.mc.holo_reduced_flux_max,
                       "dimensionless");
  append_scalar_double(file, "holo/E_LO_total", rec.snapshot.mc.holo_E_LO_total, "erg");
  append_scalar_double(file,
                       "holo/E_LO_boundary_in",
                       rec.snapshot.mc.holo_E_LO_boundary_in,
                       "erg");
  append_scalar_double(file,
                       "holo/E_LO_boundary_out",
                       rec.snapshot.mc.holo_E_LO_boundary_out,
                       "erg");
  append_scalar_double(file, "holo/matter_delta", rec.snapshot.mc.holo_matter_delta, "erg");
  append_scalar_double(file,
                       "holo/source_balance_error",
                       rec.snapshot.mc.holo_source_balance_error,
                       "erg");
  append_scalar_double(file,
                       "holo/particle_net_source_core",
                       rec.snapshot.mc.holo_particle_net_source_core,
                       "erg");
  append_scalar_double(file,
                       "holo/lo_particle_source_mismatch",
                       rec.snapshot.mc.holo_lo_particle_source_mismatch,
                       "erg");
  append_scalar_double(file,
                       "holo/Prr_coverage",
                       rec.snapshot.mc.holo_Prr_coverage,
                       "dimensionless");
  append_scalar_double(file, "holo/chi_min", rec.snapshot.mc.holo_chi_min, "dimensionless");
  append_scalar_double(file,
                       "holo/chi_mean",
                       rec.snapshot.mc.holo_chi_mean,
                       "dimensionless");
  append_scalar_double(file, "holo/chi_max", rec.snapshot.mc.holo_chi_max, "dimensionless");

  assert_history_dataset_lengths_consistent(file);
}
#endif

}  // namespace tenryu::diagnostics
