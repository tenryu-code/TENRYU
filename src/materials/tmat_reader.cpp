#include "materials/tmat_reader.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/constants.hpp"
#include "core/error.hpp"

#if TENRYU_ENABLE_HDF5
#include "io/hdf5_utils.hpp"
#endif

namespace tenryu::materials {
namespace {

[[noreturn]] void tmat_fail(const char* code, const std::string& message) {
  TENRYU_ASSERT(false, std::string(code) + ": " + message);
}

void tmat_require(const bool condition, const char* code, const std::string& message) {
  if (!condition) {
    tmat_fail(code, message);
  }
}

std::size_t checked_mul_size(const std::size_t a,
                             const std::size_t b,
                             const char* code,
                             const std::string& context) {
  tmat_require(b == 0 || a <= std::numeric_limits<std::size_t>::max() / b,
               code,
               "Size overflow while computing " + context);
  return a * b;
}

int checked_size_to_int(const std::size_t value,
                        const char* code,
                        const std::string& context) {
  tmat_require(value <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
               code,
               context + " exceeds INT_MAX");
  return static_cast<int>(value);
}

std::string trim_ascii(std::string text) {
  auto is_space = [](const char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
  };
  while (!text.empty() && is_space(text.front())) {
    text.erase(text.begin());
  }
  while (!text.empty() && is_space(text.back())) {
    text.pop_back();
  }
  return text;
}

bool is_valid_utf8(const std::string_view text) {
  std::size_t i = 0;
  while (i < text.size()) {
    const auto c0 = static_cast<unsigned char>(text[i]);
    if (c0 <= 0x7F) {
      ++i;
      continue;
    }

    std::size_t need = 0;
    std::uint32_t codepoint = 0;
    if ((c0 & 0xE0U) == 0xC0U) {
      need = 1;
      codepoint = c0 & 0x1FU;
    } else if ((c0 & 0xF0U) == 0xE0U) {
      need = 2;
      codepoint = c0 & 0x0FU;
    } else if ((c0 & 0xF8U) == 0xF0U) {
      need = 3;
      codepoint = c0 & 0x07U;
    } else {
      return false;
    }

    if (i + need >= text.size()) {
      return false;
    }
    for (std::size_t k = 1; k <= need; ++k) {
      const auto ck = static_cast<unsigned char>(text[i + k]);
      if ((ck & 0xC0U) != 0x80U) {
        return false;
      }
      codepoint = (codepoint << 6U) | (ck & 0x3FU);
    }

    if ((need == 1 && codepoint < 0x80U) ||
        (need == 2 && codepoint < 0x800U) ||
        (need == 3 && codepoint < 0x10000U)) {
      return false;
    }
    if (codepoint > 0x10FFFFU) {
      return false;
    }
    if (codepoint >= 0xD800U && codepoint <= 0xDFFFU) {
      return false;
    }

    i += need + 1;
  }

  return true;
}

void validate_monotonic_axis(const std::vector<double>& axis,
                             const std::string& path,
                             const bool require_positive,
                             const bool allow_zero_first) {
  tmat_require(!axis.empty(), "TMAT_E009", path + " must be non-empty");
  for (std::size_t i = 0; i < axis.size(); ++i) {
    const double value = axis[i];
    tmat_require(std::isfinite(value),
                 "TMAT_E007",
                 path + " contains non-finite value at index " + std::to_string(i));
    if (require_positive) {
      tmat_require(value > 0.0,
                   "TMAT_E009",
                   path + " must be strictly positive at index " + std::to_string(i));
    } else if (allow_zero_first) {
      if (i == 0) {
        tmat_require(value >= 0.0,
                     "TMAT_E009",
                     path + " first entry must be >= 0");
      } else {
        tmat_require(value > 0.0,
                     "TMAT_E009",
                     path + " entries after index 0 must be > 0");
      }
    } else {
      tmat_require(value > 0.0,
                   "TMAT_E009",
                   path + " must be strictly positive at index " + std::to_string(i));
    }
    if (i > 0) {
      tmat_require(value > axis[i - 1],
                   "TMAT_E008",
                   path + " must be strictly increasing");
    }
  }
}

void validate_finite(const std::vector<double>& values, const std::string& path) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    tmat_require(std::isfinite(values[i]),
                 "TMAT_E007",
                 path + " contains non-finite value at index " + std::to_string(i));
  }
}

void validate_finite_nonnegative(const std::vector<double>& values,
                                 const std::string& path) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double value = values[i];
    tmat_require(std::isfinite(value),
                 "TMAT_E007",
                 path + " contains non-finite value at index " + std::to_string(i));
    tmat_require(value >= 0.0,
                 "TMAT_E009",
                 path + " contains negative value at index " + std::to_string(i));
  }
}

void validate_finite_positive(const std::vector<double>& values,
                              const std::string& path) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double value = values[i];
    tmat_require(std::isfinite(value),
                 "TMAT_E007",
                 path + " contains non-finite value at index " + std::to_string(i));
    tmat_require(value > 0.0,
                 "TMAT_E009",
                 path + " must contain values > 0 at index " + std::to_string(i));
  }
}

std::size_t checked_n2d(const std::size_t n0,
                        const std::size_t n1,
                        const std::string& context) {
  return checked_mul_size(n0, n1, "TMAT_E006", context);
}

std::size_t checked_n3d(const std::size_t n0,
                        const std::size_t n1,
                        const std::size_t n2,
                        const std::string& context) {
  const std::size_t n01 = checked_mul_size(n0, n1, "TMAT_E006", context);
  return checked_mul_size(n01, n2, "TMAT_E006", context);
}

#if TENRYU_ENABLE_HDF5

template <herr_t (*Closer)(hid_t)>
class H5Handle {
 public:
  H5Handle() = default;

  explicit H5Handle(const hid_t id) : id_(id) {}

  ~H5Handle() {
    if (id_ >= 0) {
      Closer(id_);
    }
  }

  H5Handle(const H5Handle&) = delete;
  H5Handle& operator=(const H5Handle&) = delete;

  H5Handle(H5Handle&& other) noexcept : id_(other.id_) {
    other.id_ = -1;
  }

  H5Handle& operator=(H5Handle&& other) noexcept {
    if (this != &other) {
      if (id_ >= 0) {
        Closer(id_);
      }
      id_ = other.id_;
      other.id_ = -1;
    }
    return *this;
  }

  [[nodiscard]] hid_t get() const noexcept {
    return id_;
  }

  [[nodiscard]] bool valid() const noexcept {
    return id_ >= 0;
  }

 private:
  hid_t id_ = -1;
};

bool link_exists(const hid_t loc, const std::string& path) {
  return tenryu::io::h5_link_exists(loc, path.c_str(), H5P_DEFAULT) > 0;
}

void validate_dataset_filters_supported(const hid_t dset, const std::string& path) {
  H5Handle<H5Pclose> dcpl(H5Dget_create_plist(dset));
  tmat_require(dcpl.valid(),
               "TMAT_E012",
               "Failed to query dataset creation property list for " + path);

  const int nfilters = H5Pget_nfilters(dcpl.get());
  tmat_require(nfilters >= 0,
               "TMAT_E012",
               "Failed to query dataset filters for " + path);

  for (int i = 0; i < nfilters; ++i) {
    unsigned int flags = 0;
    std::array<unsigned int, 32> cd_values{};
    std::size_t n_cd_values = cd_values.size();
    unsigned int filter_config = 0;
    const H5Z_filter_t filter = H5Pget_filter2(dcpl.get(),
                                                static_cast<unsigned int>(i),
                                                &flags,
                                                &n_cd_values,
                                                cd_values.data(),
                                                0,
                                                nullptr,
                                                &filter_config);
    tmat_require(filter >= 0,
                 "TMAT_E012",
                 "Failed to inspect filter metadata for " + path);
    tmat_require(H5Zfilter_avail(filter) > 0,
                 "TMAT_E012",
                 "Unsupported HDF5 filter in " + path + " (id=" +
                     std::to_string(static_cast<int>(filter)) + ")");
    unsigned int cfg = 0;
    tmat_require(H5Zget_filter_info(filter, &cfg) >= 0,
                 "TMAT_E012",
                 "Failed to query HDF5 filter configuration for " + path);
    tmat_require((cfg & H5Z_FILTER_CONFIG_DECODE_ENABLED) != 0,
                 "TMAT_E012",
                 "HDF5 filter decode not available for " + path + " (id=" +
                     std::to_string(static_cast<int>(filter)) + ")");
  }
}

void validate_h5_type_float64(const hid_t type, const std::string& path) {
  tmat_require(H5Tget_class(type) == H5T_FLOAT,
               "TMAT_E005",
               path + " must be float64");
  tmat_require(H5Tget_size(type) == sizeof(double),
               "TMAT_E005",
               path + " must be float64");
}

void validate_h5_type_int32(const hid_t type, const std::string& path) {
  tmat_require(H5Tget_class(type) == H5T_INTEGER,
               "TMAT_E005",
               path + " must be int32");
  tmat_require(H5Tget_size(type) == sizeof(std::int32_t),
               "TMAT_E005",
               path + " must be int32");
  tmat_require(H5Tget_sign(type) != H5T_SGN_NONE,
               "TMAT_E005",
               path + " must be signed int32");
}

void validate_h5_type_string(const hid_t type, const std::string& path) {
  tmat_require(H5Tget_class(type) == H5T_STRING,
               "TMAT_E005",
               path + " must be string");
}

std::string read_h5_attr_string(const hid_t attr,
                                const hid_t type,
                                const std::string& path) {
  validate_h5_type_string(type, path);
  std::string out;
  if (H5Tis_variable_str(type) > 0) {
    char* raw = nullptr;
    tmat_require(H5Aread(attr, type, &raw) >= 0,
                 "TMAT_E011",
                 "Failed to read UTF-8 attribute " + path);
    out = (raw != nullptr) ? std::string(raw) : std::string();
    if (raw != nullptr) {
      H5free_memory(raw);
    }
  } else {
    const std::size_t size = H5Tget_size(type);
    tmat_require(size > 0,
                 "TMAT_E005",
                 "Invalid string storage size for attribute " + path);
    std::vector<char> storage(size + 1, '\0');
    tmat_require(H5Aread(attr, type, storage.data()) >= 0,
                 "TMAT_E011",
                 "Failed to read UTF-8 attribute " + path);
    out = std::string(storage.data());
  }
  tmat_require(is_valid_utf8(out),
               "TMAT_E011",
               "Invalid UTF-8 in " + path);
  return out;
}

std::string read_h5_dataset_string_scalar(const hid_t dset,
                                          const hid_t type,
                                          const std::string& path) {
  validate_h5_type_string(type, path);
  std::string out;
  if (H5Tis_variable_str(type) > 0) {
    char* raw = nullptr;
    tmat_require(H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &raw) >= 0,
                 "TMAT_E011",
                 "Failed to read UTF-8 dataset " + path);
    out = (raw != nullptr) ? std::string(raw) : std::string();
    if (raw != nullptr) {
      H5free_memory(raw);
    }
  } else {
    const std::size_t size = H5Tget_size(type);
    tmat_require(size > 0,
                 "TMAT_E005",
                 "Invalid string storage size for dataset " + path);
    std::vector<char> storage(size + 1, '\0');
    tmat_require(H5Dread(dset,
                         type,
                         H5S_ALL,
                         H5S_ALL,
                         H5P_DEFAULT,
                         storage.data()) >= 0,
                 "TMAT_E011",
                 "Failed to read UTF-8 dataset " + path);
    out = std::string(storage.data());
  }
  tmat_require(is_valid_utf8(out),
               "TMAT_E011",
               "Invalid UTF-8 in " + path);
  return out;
}

struct DatasetInfo {
  int rank = 0;
  std::vector<hsize_t> dims;
};

DatasetInfo read_dataset_info(const hid_t dset, const std::string& path) {
  H5Handle<H5Sclose> space(H5Dget_space(dset));
  tmat_require(space.valid(),
               "TMAT_E006",
               "Failed to get dataspace for " + path);

  const int rank = H5Sget_simple_extent_ndims(space.get());
  tmat_require(rank >= 0,
               "TMAT_E006",
               "Failed to get rank for " + path);

  DatasetInfo info;
  info.rank = rank;
  info.dims.resize(static_cast<std::size_t>(rank), 0);
  if (rank > 0) {
    const int got = H5Sget_simple_extent_dims(space.get(), info.dims.data(), nullptr);
    tmat_require(got == rank,
                 "TMAT_E006",
                 "Failed to read dimensions for " + path);
  }
  return info;
}

std::size_t dims_total_size(const std::vector<hsize_t>& dims, const std::string& path) {
  std::size_t total = 1;
  for (const hsize_t d : dims) {
    const std::size_t ds = static_cast<std::size_t>(d);
    tmat_require(static_cast<hsize_t>(ds) == d,
                 "TMAT_E006",
                 "Dimension overflows size_t for " + path);
    total = checked_mul_size(total,
                             ds,
                             "TMAT_E006",
                             "element count for " + path);
  }
  return total;
}

std::optional<std::string> read_optional_attr_string(const hid_t obj,
                                                     const std::string& obj_path,
                                                     const std::string& name) {
  if (H5Aexists(obj, name.c_str()) <= 0) {
    return std::nullopt;
  }

  H5Handle<H5Aclose> attr(H5Aopen(obj, name.c_str(), H5P_DEFAULT));
  tmat_require(attr.valid(),
               "TMAT_E001",
               "Failed to open attribute " + obj_path + "/@" + name);
  H5Handle<H5Tclose> type(H5Aget_type(attr.get()));
  tmat_require(type.valid(),
               "TMAT_E005",
               "Failed to query attribute type " + obj_path + "/@" + name);

  return read_h5_attr_string(attr.get(), type.get(), obj_path + "/@" + name);
}

std::string read_required_attr_string(const hid_t obj,
                                      const std::string& obj_path,
                                      const std::string& name) {
  const auto value = read_optional_attr_string(obj, obj_path, name);
  tmat_require(value.has_value(),
               "TMAT_E001",
               "Required attribute missing: " + obj_path + "/@" + name);
  return *value;
}

std::optional<std::int32_t> read_optional_attr_i32(const hid_t obj,
                                                   const std::string& obj_path,
                                                   const std::string& name) {
  if (H5Aexists(obj, name.c_str()) <= 0) {
    return std::nullopt;
  }

  H5Handle<H5Aclose> attr(H5Aopen(obj, name.c_str(), H5P_DEFAULT));
  tmat_require(attr.valid(),
               "TMAT_E001",
               "Failed to open attribute " + obj_path + "/@" + name);
  H5Handle<H5Tclose> type(H5Aget_type(attr.get()));
  tmat_require(type.valid(),
               "TMAT_E005",
               "Failed to query attribute type " + obj_path + "/@" + name);
  validate_h5_type_int32(type.get(), obj_path + "/@" + name);

  std::int32_t value = 0;
  tmat_require(H5Aread(attr.get(), H5T_NATIVE_INT32, &value) >= 0,
               "TMAT_E005",
               "Failed to read attribute " + obj_path + "/@" + name);
  return value;
}

std::int32_t read_required_attr_i32(const hid_t obj,
                                    const std::string& obj_path,
                                    const std::string& name) {
  const auto value = read_optional_attr_i32(obj, obj_path, name);
  tmat_require(value.has_value(),
               "TMAT_E001",
               "Required attribute missing: " + obj_path + "/@" + name);
  return *value;
}

std::vector<double> read_required_dataset_f64(const hid_t file,
                                              const std::string& path,
                                              const int expected_rank,
                                              std::vector<hsize_t>* dims_out = nullptr) {
  tmat_require(link_exists(file, path),
               "TMAT_E001",
               "Required path missing: " + path);

  H5Handle<H5Dclose> dset(H5Dopen2(file, path.c_str(), H5P_DEFAULT));
  tmat_require(dset.valid(), "TMAT_E001", "Failed to open dataset " + path);
  validate_dataset_filters_supported(dset.get(), path);

  H5Handle<H5Tclose> type(H5Dget_type(dset.get()));
  tmat_require(type.valid(), "TMAT_E005", "Failed to query dataset type for " + path);
  validate_h5_type_float64(type.get(), path);

  const DatasetInfo info = read_dataset_info(dset.get(), path);
  tmat_require(info.rank == expected_rank,
               "TMAT_E006",
               path + " rank mismatch: expected " + std::to_string(expected_rank) +
                   ", got " + std::to_string(info.rank));

  const std::size_t total = dims_total_size(info.dims, path);
  std::vector<double> out(total, 0.0);
  if (total > 0) {
    tmat_require(H5Dread(dset.get(),
                         H5T_NATIVE_DOUBLE,
                         H5S_ALL,
                         H5S_ALL,
                         H5P_DEFAULT,
                         out.data()) >= 0,
                 "TMAT_E005",
                 "Failed to read dataset " + path);
  }

  if (dims_out != nullptr) {
    *dims_out = info.dims;
  }

  return out;
}

std::vector<std::int32_t> read_required_dataset_i32(const hid_t file,
                                                    const std::string& path,
                                                    const int expected_rank,
                                                    std::vector<hsize_t>* dims_out = nullptr) {
  tmat_require(link_exists(file, path),
               "TMAT_E001",
               "Required path missing: " + path);

  H5Handle<H5Dclose> dset(H5Dopen2(file, path.c_str(), H5P_DEFAULT));
  tmat_require(dset.valid(), "TMAT_E001", "Failed to open dataset " + path);
  validate_dataset_filters_supported(dset.get(), path);

  H5Handle<H5Tclose> type(H5Dget_type(dset.get()));
  tmat_require(type.valid(), "TMAT_E005", "Failed to query dataset type for " + path);
  validate_h5_type_int32(type.get(), path);

  const DatasetInfo info = read_dataset_info(dset.get(), path);
  tmat_require(info.rank == expected_rank,
               "TMAT_E006",
               path + " rank mismatch: expected " + std::to_string(expected_rank) +
                   ", got " + std::to_string(info.rank));

  const std::size_t total = dims_total_size(info.dims, path);
  std::vector<std::int32_t> out(total, 0);
  if (total > 0) {
    tmat_require(H5Dread(dset.get(),
                         H5T_NATIVE_INT32,
                         H5S_ALL,
                         H5S_ALL,
                         H5P_DEFAULT,
                         out.data()) >= 0,
                 "TMAT_E005",
                 "Failed to read dataset " + path);
  }

  if (dims_out != nullptr) {
    *dims_out = info.dims;
  }

  return out;
}

double read_required_dataset_f64_scalar(const hid_t file, const std::string& path) {
  tmat_require(link_exists(file, path),
               "TMAT_E001",
               "Required path missing: " + path);

  H5Handle<H5Dclose> dset(H5Dopen2(file, path.c_str(), H5P_DEFAULT));
  tmat_require(dset.valid(), "TMAT_E001", "Failed to open dataset " + path);
  validate_dataset_filters_supported(dset.get(), path);

  H5Handle<H5Tclose> type(H5Dget_type(dset.get()));
  tmat_require(type.valid(), "TMAT_E005", "Failed to query dataset type for " + path);
  validate_h5_type_float64(type.get(), path);

  const DatasetInfo info = read_dataset_info(dset.get(), path);
  const bool scalar_ok =
      info.rank == 0 ||
      (info.rank == 1 && info.dims.size() == 1 && info.dims[0] == 1);
  tmat_require(scalar_ok,
               "TMAT_E006",
               path + " must be scalar");

  double value = 0.0;
  tmat_require(H5Dread(dset.get(),
                       H5T_NATIVE_DOUBLE,
                       H5S_ALL,
                       H5S_ALL,
                       H5P_DEFAULT,
                       &value) >= 0,
               "TMAT_E005",
               "Failed to read scalar dataset " + path);
  return value;
}

std::optional<std::string> read_optional_dataset_string_scalar(const hid_t file,
                                                               const std::string& path) {
  if (!link_exists(file, path)) {
    return std::nullopt;
  }

  H5Handle<H5Dclose> dset(H5Dopen2(file, path.c_str(), H5P_DEFAULT));
  tmat_require(dset.valid(), "TMAT_E001", "Failed to open dataset " + path);
  validate_dataset_filters_supported(dset.get(), path);

  H5Handle<H5Tclose> type(H5Dget_type(dset.get()));
  tmat_require(type.valid(), "TMAT_E005", "Failed to query dataset type for " + path);

  const DatasetInfo info = read_dataset_info(dset.get(), path);
  const bool scalar_ok =
      info.rank == 0 ||
      (info.rank == 1 && info.dims.size() == 1 && info.dims[0] == 1);
  tmat_require(scalar_ok,
               "TMAT_E006",
               path + " must be scalar string");

  return read_h5_dataset_string_scalar(dset.get(), type.get(), path);
}

void validate_optional_dataset_string_1d_shape_and_encoding(const hid_t file,
                                                            const std::string& path,
                                                            const std::size_t expected_count) {
  if (!link_exists(file, path)) {
    return;
  }

  H5Handle<H5Dclose> dset(H5Dopen2(file, path.c_str(), H5P_DEFAULT));
  tmat_require(dset.valid(), "TMAT_E001", "Failed to open dataset " + path);
  validate_dataset_filters_supported(dset.get(), path);

  H5Handle<H5Tclose> type(H5Dget_type(dset.get()));
  tmat_require(type.valid(), "TMAT_E005", "Failed to query dataset type for " + path);
  validate_h5_type_string(type.get(), path);

  const DatasetInfo info = read_dataset_info(dset.get(), path);
  tmat_require(info.rank == 1,
               "TMAT_E006",
               path + " must be rank-1 string array");
  tmat_require(info.dims.size() == 1,
               "TMAT_E006",
               path + " invalid rank metadata");
  tmat_require(static_cast<std::size_t>(info.dims[0]) == expected_count,
               "TMAT_E006",
               path + " length mismatch: expected " + std::to_string(expected_count) +
                   ", got " + std::to_string(static_cast<std::size_t>(info.dims[0])));

  const std::size_t n = expected_count;
  if (H5Tis_variable_str(type.get()) > 0) {
    std::vector<char*> values(n, nullptr);
    tmat_require(H5Dread(dset.get(),
                         type.get(),
                         H5S_ALL,
                         H5S_ALL,
                         H5P_DEFAULT,
                         values.data()) >= 0,
                 "TMAT_E011",
                 "Failed to read UTF-8 string array " + path);
    for (std::size_t i = 0; i < n; ++i) {
      const std::string entry = (values[i] != nullptr) ? std::string(values[i]) : std::string();
      tmat_require(is_valid_utf8(entry),
                   "TMAT_E011",
                   path + " contains invalid UTF-8 at index " + std::to_string(i));
      if (values[i] != nullptr) {
        H5free_memory(values[i]);
      }
    }
    return;
  }

  const std::size_t item_size = H5Tget_size(type.get());
  tmat_require(item_size > 0,
               "TMAT_E005",
               "Invalid string storage size for " + path);
  const std::size_t total_bytes = checked_mul_size(n,
                                                   item_size,
                                                   "TMAT_E006",
                                                   "string dataset storage for " + path);
  std::vector<char> storage(total_bytes, '\0');
  tmat_require(H5Dread(dset.get(),
                       type.get(),
                       H5S_ALL,
                       H5S_ALL,
                       H5P_DEFAULT,
                       storage.data()) >= 0,
               "TMAT_E011",
               "Failed to read UTF-8 string array " + path);

  for (std::size_t i = 0; i < n; ++i) {
    const char* begin = storage.data() + i * item_size;
    std::size_t length = 0;
    while (length < item_size && begin[length] != '\0') {
      ++length;
    }
    const std::string entry(begin, length);
    tmat_require(is_valid_utf8(entry),
                 "TMAT_E011",
                 path + " contains invalid UTF-8 at index " + std::to_string(i));
  }
}

int parse_schema_major(const std::string& semver) {
  const std::string text = trim_ascii(semver);
  const std::size_t dot1 = text.find('.');
  const std::size_t dot2 = (dot1 == std::string::npos) ? std::string::npos
                                                        : text.find('.', dot1 + 1);
  tmat_require(dot1 != std::string::npos && dot2 != std::string::npos,
               "TMAT_E002",
               "Malformed schema_version: " + semver);
  const std::string major = text.substr(0, dot1);
  const std::string minor = text.substr(dot1 + 1, dot2 - dot1 - 1);
  const std::string patch = text.substr(dot2 + 1);
  tmat_require(!major.empty() && !minor.empty() && !patch.empty(),
               "TMAT_E002",
               "Malformed schema_version: " + semver);
  auto digits_only = [](const std::string& token) {
    return std::all_of(token.begin(), token.end(), [](const char c) {
      return std::isdigit(static_cast<unsigned char>(c)) != 0;
    });
  };
  tmat_require(digits_only(major) && digits_only(minor) && digits_only(patch),
               "TMAT_E002",
               "Malformed schema_version: " + semver);

  int major_value = 0;
  for (const char c : major) {
    const int digit = c - '0';
    tmat_require(major_value <= (std::numeric_limits<int>::max() - digit) / 10,
                 "TMAT_E002",
                 "schema_version major overflows int: " + semver);
    major_value = major_value * 10 + digit;
  }
  return major_value;
}

bool is_valid_required_feature_token(const std::string& token) {
  const std::size_t pos = token.rfind("_v");
  if (pos == std::string::npos || pos == 0 || pos + 2 >= token.size()) {
    return false;
  }

  for (std::size_t i = 0; i < pos; ++i) {
    const char c = token[i];
    if (!(std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_')) {
      return false;
    }
  }
  for (std::size_t i = pos + 2; i < token.size(); ++i) {
    if (std::isdigit(static_cast<unsigned char>(token[i])) == 0) {
      return false;
    }
  }
  return true;
}

std::vector<std::string> parse_required_features(const std::string& csv_text) {
  std::vector<std::string> out;
  std::size_t start = 0;
  while (start <= csv_text.size()) {
    std::size_t comma = csv_text.find(',', start);
    if (comma == std::string::npos) {
      comma = csv_text.size();
    }
    std::string token = trim_ascii(csv_text.substr(start, comma - start));
    if (!token.empty()) {
      tmat_require(is_valid_required_feature_token(token),
                   "TMAT_E004",
                   "Malformed required feature token: " + token);
      // TMAT-H5 v1 reader has no extension features enabled in TENRYU yet.
      tmat_fail("TMAT_E004", "Unknown required feature token: " + token);
    }
    if (comma == csv_text.size()) {
      break;
    }
    start = comma + 1;
  }
  return out;
}

TmatMaterialInfo load_material_group(const hid_t file) {
  tmat_require(link_exists(file, "/material"),
               "TMAT_E001",
               "Required path missing: /material");

  TmatMaterialInfo out;
  if (const auto name = read_optional_dataset_string_scalar(file, "/material/name");
      name.has_value()) {
    out.name = *name;
  }

  std::vector<hsize_t> z_dims;
  const std::vector<std::int32_t> z_raw =
      read_required_dataset_i32(file, "/material/Z", 1, &z_dims);
  tmat_require(z_dims.size() == 1, "TMAT_E006", "/material/Z rank metadata invalid");
  const std::size_t n_species = static_cast<std::size_t>(z_dims[0]);
  tmat_require(n_species >= 1,
               "TMAT_E009",
               "/material/Z must contain at least one species");

  out.Z.resize(n_species, 0);
  for (std::size_t i = 0; i < n_species; ++i) {
    tmat_require(z_raw[i] >= 1,
                 "TMAT_E009",
                 "/material/Z must be >= 1 at index " + std::to_string(i));
    out.Z[i] = static_cast<int>(z_raw[i]);
  }

  std::vector<hsize_t> a_dims;
  out.A_amu = read_required_dataset_f64(file, "/material/A_amu", 1, &a_dims);
  tmat_require(a_dims.size() == 1,
               "TMAT_E006",
               "/material/A_amu rank metadata invalid");
  tmat_require(static_cast<std::size_t>(a_dims[0]) == n_species,
               "TMAT_E006",
               "/material/A_amu length mismatch");
  validate_finite_positive(out.A_amu, "/material/A_amu");

  std::vector<hsize_t> mf_dims;
  out.mass_fraction = read_required_dataset_f64(file, "/material/mass_fraction", 1, &mf_dims);
  tmat_require(mf_dims.size() == 1,
               "TMAT_E006",
               "/material/mass_fraction rank metadata invalid");
  tmat_require(static_cast<std::size_t>(mf_dims[0]) == n_species,
               "TMAT_E006",
               "/material/mass_fraction length mismatch");
  validate_finite_nonnegative(out.mass_fraction, "/material/mass_fraction");
  {
    double sum = 0.0;
    for (const double value : out.mass_fraction) {
      sum += value;
    }
    tmat_require(std::abs(sum - 1.0) <= 1.0e-8,
                 "TMAT_E010",
                 "/material/mass_fraction must sum to 1 within 1e-8; got " +
                     std::to_string(sum));
  }

  out.Abar_ion_amu = read_required_dataset_f64_scalar(file, "/material/Abar_ion_amu");
  tmat_require(std::isfinite(out.Abar_ion_amu),
               "TMAT_E007",
               "/material/Abar_ion_amu must be finite");
  tmat_require(out.Abar_ion_amu > 0.0,
               "TMAT_E009",
               "/material/Abar_ion_amu must be > 0");

  if (link_exists(file, "/material/number_fraction")) {
    std::vector<hsize_t> nf_dims;
    out.number_fraction =
        read_required_dataset_f64(file, "/material/number_fraction", 1, &nf_dims);
    tmat_require(nf_dims.size() == 1,
                 "TMAT_E006",
                 "/material/number_fraction rank metadata invalid");
    tmat_require(static_cast<std::size_t>(nf_dims[0]) == n_species,
                 "TMAT_E006",
                 "/material/number_fraction length mismatch");
    validate_finite_nonnegative(out.number_fraction, "/material/number_fraction");
    double sum = 0.0;
    for (const double value : out.number_fraction) {
      sum += value;
    }
    tmat_require(std::abs(sum - 1.0) <= 1.0e-8,
                 "TMAT_E009",
                 "/material/number_fraction must sum to 1 within 1e-8; got " +
                     std::to_string(sum));
  } else {
    out.number_fraction.resize(n_species, 0.0);
    double sum_w_over_a = 0.0;
    for (std::size_t i = 0; i < n_species; ++i) {
      out.number_fraction[i] = out.mass_fraction[i] / out.A_amu[i];
      sum_w_over_a += out.number_fraction[i];
    }
    tmat_require(sum_w_over_a > 0.0 && std::isfinite(sum_w_over_a),
                 "TMAT_E009",
                 "Cannot derive /material/number_fraction from mass_fraction and A_amu");
    for (double& value : out.number_fraction) {
      value /= sum_w_over_a;
    }
  }

  validate_optional_dataset_string_1d_shape_and_encoding(file,
                                                         "/material/species_name",
                                                         n_species);

  return out;
}

TmatEOSData load_eos_group(const hid_t file) {
  TmatEOSData out;

  H5Handle<H5Gclose> eos_group(H5Gopen2(file, "/eos", H5P_DEFAULT));
  tmat_require(eos_group.valid(), "TMAT_E001", "Failed to open /eos group");

  const std::string axis_order = read_required_attr_string(eos_group.get(), "/eos", "axis_order");
  tmat_require(axis_order == "D,T",
               "TMAT_E001",
               "/eos/@axis_order must be \"D,T\", got \"" + axis_order + "\"");
  const std::string density_axis =
      read_required_attr_string(eos_group.get(), "/eos", "primary_density_axis");
  tmat_require(density_axis == "ni_cm3",
               "TMAT_E001",
               "/eos/@primary_density_axis must be \"ni_cm3\", got \"" +
                   density_axis + "\"");

  std::vector<hsize_t> rho_dims;
  out.rho_grid = read_required_dataset_f64(file, "/eos/grid/ni_cm3", 1, &rho_dims);
  std::vector<hsize_t> t_dims;
  out.T_grid_eV =
      read_required_dataset_f64(file, "/eos/grid/temperature_eV", 1, &t_dims);
  tmat_require(rho_dims.size() == 1 && t_dims.size() == 1,
               "TMAT_E006",
               "Invalid /eos grid rank metadata");
  const std::size_t nD = static_cast<std::size_t>(rho_dims[0]);
  const std::size_t nT = static_cast<std::size_t>(t_dims[0]);
  validate_monotonic_axis(out.rho_grid, "/eos/grid/ni_cm3", true, false);
  validate_monotonic_axis(out.T_grid_eV, "/eos/grid/temperature_eV", true, false);

  const auto read_eos_field = [&](const std::string& path,
                                  std::vector<double>* dst,
                                  const bool require_nonnegative) {
    std::vector<hsize_t> dims;
    *dst = read_required_dataset_f64(file, path, 2, &dims);
    tmat_require(dims.size() == 2,
                 "TMAT_E006",
                 path + " rank metadata invalid");
    tmat_require(dims[0] == rho_dims[0] && dims[1] == t_dims[0],
                 "TMAT_E006",
                 path + " shape mismatch; expected [" + std::to_string(nD) + "," +
                     std::to_string(nT) + "]");
    if (require_nonnegative) {
      validate_finite_nonnegative(*dst, path);
    } else {
      validate_finite(*dst, path);
    }
  };

  read_eos_field("/eos/fields/zbar", &out.zbar, true);
  read_eos_field("/eos/fields/P_i", &out.P_i, false);
  read_eos_field("/eos/fields/P_e", &out.P_e, false);
  read_eos_field("/eos/fields/e_i", &out.e_i, false);
  read_eos_field("/eos/fields/e_e", &out.e_e, false);

  const auto read_optional_cv = [&](const std::string& path,
                                    std::optional<std::vector<double>>* dst) {
    if (!link_exists(file, path)) {
      return;
    }
    std::vector<hsize_t> dims;
    std::vector<double> values = read_required_dataset_f64(file, path, 2, &dims);
    tmat_require(dims.size() == 2,
                 "TMAT_E006",
                 path + " rank metadata invalid");
    tmat_require(dims[0] == rho_dims[0] && dims[1] == t_dims[0],
                 "TMAT_E006",
                 path + " shape mismatch; expected [" + std::to_string(nD) + "," +
                     std::to_string(nT) + "]");
    validate_finite_positive(values, path);
    *dst = std::move(values);
  };

  read_optional_cv("/eos/fields/cv_i", &out.cv_i);
  read_optional_cv("/eos/fields/cv_e", &out.cv_e);

  out.ndens = checked_size_to_int(nD, "TMAT_E006", "/eos ndens");
  out.ntemp = checked_size_to_int(nT, "TMAT_E006", "/eos ntemp");
  return out;
}

TmatOpacityData load_opacity_group(const hid_t file, const TmatLoadMode mode) {
  TmatOpacityData out;

  H5Handle<H5Gclose> opacity_group(H5Gopen2(file, "/opacity", H5P_DEFAULT));
  tmat_require(opacity_group.valid(), "TMAT_E001", "Failed to open /opacity group");

  const std::string axis_order =
      read_required_attr_string(opacity_group.get(), "/opacity", "axis_order");
  tmat_require(axis_order == "G,D,T",
               "TMAT_E001",
               "/opacity/@axis_order must be \"G,D,T\", got \"" + axis_order + "\"");
  const std::string density_axis =
      read_required_attr_string(opacity_group.get(), "/opacity", "primary_density_axis");
  tmat_require(density_axis == "ni_cm3",
               "TMAT_E001",
               "/opacity/@primary_density_axis must be \"ni_cm3\", got \"" +
                   density_axis + "\"");

  bool has_is_lte_attr = false;
  if (const auto is_lte = read_optional_attr_i32(opacity_group.get(), "/opacity", "is_lte");
      is_lte.has_value()) {
    has_is_lte_attr = true;
    tmat_require(*is_lte == 0 || *is_lte == 1,
                 "TMAT_E009",
                 "/opacity/@is_lte must be 0 or 1");
    out.is_lte = (*is_lte == 1);
  } else {
    tmat_require(mode == TmatLoadMode::Compatible,
                 "TMAT_E001",
                 "Required attribute missing: /opacity/@is_lte");
  }

  std::vector<hsize_t> rho_dims;
  out.rho_grid = read_required_dataset_f64(file, "/opacity/grid/ni_cm3", 1, &rho_dims);
  std::vector<hsize_t> t_dims;
  out.T_grid_eV =
      read_required_dataset_f64(file, "/opacity/grid/temperature_eV", 1, &t_dims);
  std::vector<hsize_t> b_dims;
  out.bounds_eV =
      read_required_dataset_f64(file, "/opacity/grid/group_bounds_eV", 1, &b_dims);

  tmat_require(rho_dims.size() == 1 && t_dims.size() == 1 && b_dims.size() == 1,
               "TMAT_E006",
               "Invalid /opacity grid rank metadata");
  const std::size_t nD = static_cast<std::size_t>(rho_dims[0]);
  const std::size_t nT = static_cast<std::size_t>(t_dims[0]);
  const std::size_t nBounds = static_cast<std::size_t>(b_dims[0]);
  tmat_require(nBounds >= 2,
               "TMAT_E009",
               "/opacity/grid/group_bounds_eV must have length >= 2");
  const std::size_t nG = nBounds - 1;

  validate_monotonic_axis(out.rho_grid, "/opacity/grid/ni_cm3", true, false);
  validate_monotonic_axis(out.T_grid_eV, "/opacity/grid/temperature_eV", true, false);
  validate_monotonic_axis(out.bounds_eV,
                          "/opacity/grid/group_bounds_eV",
                          false,
                          true);

  const auto read_opacity_field = [&](const std::string& path, std::vector<double>* dst) {
    std::vector<hsize_t> dims;
    *dst = read_required_dataset_f64(file, path, 3, &dims);
    tmat_require(dims.size() == 3,
                 "TMAT_E006",
                 path + " rank metadata invalid");
    tmat_require(dims[0] == b_dims[0] - 1 && dims[1] == rho_dims[0] && dims[2] == t_dims[0],
                 "TMAT_E006",
                 path + " shape mismatch; expected [" + std::to_string(nG) + "," +
                     std::to_string(nD) + "," + std::to_string(nT) + "]");
    validate_finite_nonnegative(*dst, path);
  };

  read_opacity_field("/opacity/fields/kappa_R", &out.kappa_R);
  read_opacity_field("/opacity/fields/kappa_PA", &out.kappa_PA);
  read_opacity_field("/opacity/fields/kappa_PE", &out.kappa_PE);

  out.ngroups = checked_size_to_int(nG, "TMAT_E006", "/opacity ngroups");
  out.ndens = checked_size_to_int(nD, "TMAT_E006", "/opacity ndens");
  out.ntemp = checked_size_to_int(nT, "TMAT_E006", "/opacity ntemp");

  if (!has_is_lte_attr) {
    double max_rel = 0.0;
    for (std::size_t i = 0; i < out.kappa_PA.size(); ++i) {
      const double pa = out.kappa_PA[i];
      const double pe = out.kappa_PE[i];
      const double denom = std::max({pa, pe, 1.0e-30});
      max_rel = std::max(max_rel, std::abs(pa - pe) / denom);
    }
    out.is_lte = (max_rel <= 1.0e-6);

    static std::atomic<bool> warned_missing_is_lte{false};
    bool expected = false;
    if (warned_missing_is_lte.compare_exchange_strong(expected,
                                                      true,
                                                      std::memory_order_relaxed)) {
      core::log_warning("TMAT compatible-mode fallback: /opacity/@is_lte is missing; "
                        "auto-detected LTE flag from kappa_PA/kappa_PE consistency");
    }
  }

  return out;
}

TmatIonizationData load_ionization_group(const hid_t file,
                                         const TmatMaterialInfo& material,
                                         const TmatEOSData* eos) {
  TmatIonizationData out;

  // Optional TMAT-H5 ionization block:
  // /ionization attrs axis_order="S,D,T", primary_density_axis="ni_cm3";
  // grid/ni_cm3 [nD], grid/temperature_eV [nT], stage_element [nS] int32,
  // stage_charge [nS] int32, fields/fractions [nS,nD,nT] float64.
  H5Handle<H5Gclose> ionization_group(H5Gopen2(file, "/ionization", H5P_DEFAULT));
  tmat_require(ionization_group.valid(),
               "TMAT_E001",
               "Failed to open /ionization group");

  const std::string axis_order =
      read_required_attr_string(ionization_group.get(), "/ionization", "axis_order");
  tmat_require(axis_order == "S,D,T",
               "TMAT_E001",
               "/ionization/@axis_order must be \"S,D,T\", got \"" +
                   axis_order + "\"");
  const std::string density_axis =
      read_required_attr_string(ionization_group.get(),
                                "/ionization",
                                "primary_density_axis");
  tmat_require(density_axis == "ni_cm3",
               "TMAT_E001",
               "/ionization/@primary_density_axis must be \"ni_cm3\", got \"" +
                   density_axis + "\"");

  std::vector<hsize_t> rho_dims;
  out.rho_grid =
      read_required_dataset_f64(file, "/ionization/grid/ni_cm3", 1, &rho_dims);
  std::vector<hsize_t> t_dims;
  out.T_grid_eV =
      read_required_dataset_f64(file, "/ionization/grid/temperature_eV", 1, &t_dims);
  std::vector<hsize_t> element_dims;
  const std::vector<std::int32_t> stage_element_raw =
      read_required_dataset_i32(file, "/ionization/stage_element", 1, &element_dims);
  std::vector<hsize_t> charge_dims;
  const std::vector<std::int32_t> stage_charge_raw =
      read_required_dataset_i32(file, "/ionization/stage_charge", 1, &charge_dims);

  tmat_require(rho_dims.size() == 1 && t_dims.size() == 1 &&
                   element_dims.size() == 1 && charge_dims.size() == 1,
               "TMAT_E006",
               "Invalid /ionization rank metadata");
  const std::size_t nD = static_cast<std::size_t>(rho_dims[0]);
  const std::size_t nT = static_cast<std::size_t>(t_dims[0]);
  const std::size_t nS = static_cast<std::size_t>(element_dims[0]);
  tmat_require(charge_dims[0] == element_dims[0],
               "TMAT_E006",
               "/ionization/stage_charge length mismatch");
  validate_monotonic_axis(out.rho_grid,
                          "/ionization/grid/ni_cm3",
                          true,
                          false);
  validate_monotonic_axis(out.T_grid_eV,
                          "/ionization/grid/temperature_eV",
                          true,
                          false);

  if (eos != nullptr) {
    tmat_require(out.rho_grid.size() == eos->rho_grid.size() &&
                     out.T_grid_eV.size() == eos->T_grid_eV.size(),
                 "TMAT_IONIZATION_GRID_MISMATCH",
                 "/ionization grid lengths must match /eos grids");
    const auto require_grid_match = [](const std::vector<double>& ion_grid,
                                       const std::vector<double>& eos_grid,
                                       const std::string& path) {
      for (std::size_t i = 0; i < ion_grid.size(); ++i) {
        const double scale = std::max(std::abs(ion_grid[i]), std::abs(eos_grid[i]));
        tmat_require(std::abs(ion_grid[i] - eos_grid[i]) <= 1.0e-12 * scale,
                     "TMAT_IONIZATION_GRID_MISMATCH",
                     path + " differs from /eos grid at index " + std::to_string(i));
      }
    };
    require_grid_match(out.rho_grid, eos->rho_grid, "/ionization/grid/ni_cm3");
    require_grid_match(out.T_grid_eV,
                       eos->T_grid_eV,
                       "/ionization/grid/temperature_eV");
  }

  const std::size_t n_elements = material.Z.size();
  out.stage_element.resize(nS, 0);
  out.stage_charge.resize(nS, 0);
  std::vector<std::vector<bool>> seen(n_elements);
  for (std::size_t e = 0; e < n_elements; ++e) {
    seen[e].resize(static_cast<std::size_t>(material.Z[e]) + 1, false);
  }
  for (std::size_t s = 0; s < nS; ++s) {
    const std::int32_t element = stage_element_raw[s];
    tmat_require(element >= 0 &&
                     static_cast<std::size_t>(element) < n_elements,
                 "TMAT_E009",
                 "/ionization/stage_element out of range at index " +
                     std::to_string(s));
    const std::int32_t charge = stage_charge_raw[s];
    tmat_require(charge >= 0 && charge <= material.Z[static_cast<std::size_t>(element)],
                 "TMAT_E009",
                 "/ionization/stage_charge out of range at index " +
                     std::to_string(s));
    tmat_require(!seen[static_cast<std::size_t>(element)]
                          [static_cast<std::size_t>(charge)],
                 "TMAT_E010",
                 "Duplicate /ionization (stage_element, stage_charge) pair at index " +
                     std::to_string(s));
    seen[static_cast<std::size_t>(element)][static_cast<std::size_t>(charge)] = true;
    out.stage_element[s] = static_cast<int>(element);
    out.stage_charge[s] = static_cast<int>(charge);
  }

  std::vector<hsize_t> fraction_dims;
  out.fractions =
      read_required_dataset_f64(file, "/ionization/fields/fractions", 3, &fraction_dims);
  tmat_require(fraction_dims.size() == 3,
               "TMAT_E006",
               "/ionization/fields/fractions rank metadata invalid");
  tmat_require(fraction_dims[0] == element_dims[0] &&
                   fraction_dims[1] == rho_dims[0] &&
                   fraction_dims[2] == t_dims[0],
               "TMAT_E006",
               "/ionization/fields/fractions shape mismatch; expected [" +
                   std::to_string(nS) + "," + std::to_string(nD) + "," +
                   std::to_string(nT) + "]");
  for (std::size_t i = 0; i < out.fractions.size(); ++i) {
    const double value = out.fractions[i];
    tmat_require(std::isfinite(value),
                 "TMAT_E007",
                 "/ionization/fields/fractions contains non-finite value at index " +
                     std::to_string(i));
    tmat_require(value >= -1.0e-12 && value <= 1.2,
                 "TMAT_E009",
                 "/ionization/fields/fractions out of range at index " +
                     std::to_string(i));
    if (value < 0.0) {
      out.fractions[i] = 0.0;
    }
  }

  for (std::size_t e = 0; e < n_elements; ++e) {
    for (std::size_t d = 0; d < nD; ++d) {
      for (std::size_t t = 0; t < nT; ++t) {
        double sum = 0.0;
        for (std::size_t s = 0; s < nS; ++s) {
          if (out.stage_element[s] == static_cast<int>(e)) {
            sum += out.fractions[(s * nD + d) * nT + t];
          }
        }
        if (sum <= 0.0) {
          for (std::size_t s = 0; s < nS; ++s) {
            if (out.stage_element[s] == static_cast<int>(e)) {
              out.fractions[(s * nD + d) * nT + t] = 0.0;
            }
          }
          continue;
        }
        tmat_require(std::abs(sum - 1.0) <= 0.2,
                     "TMAT_IONIZATION_NORMALIZATION",
                     "/ionization fractions for element " + std::to_string(e) +
                         " sum to " + std::to_string(sum) + " at grid point (" +
                         std::to_string(d) + "," + std::to_string(t) + ")");
        for (std::size_t s = 0; s < nS; ++s) {
          if (out.stage_element[s] == static_cast<int>(e)) {
            out.fractions[(s * nD + d) * nT + t] /= sum;
          }
        }
      }
    }
  }

  out.ndens = checked_size_to_int(nD, "TMAT_E006", "/ionization ndens");
  out.ntemp = checked_size_to_int(nT, "TMAT_E006", "/ionization ntemp");
  out.n_stages = checked_size_to_int(nS, "TMAT_E006", "/ionization n_stages");
  return out;
}

#endif  // TENRYU_ENABLE_HDF5

std::vector<double> transpose_dt_to_td(const std::vector<double>& src,
                                       const std::size_t nD,
                                       const std::size_t nT) {
  const std::size_t n = checked_n2d(nD, nT, "transpose size");
  tmat_require(src.size() == n,
               "TMAT_E006",
               "Transpose source size mismatch");

  std::vector<double> dst(n, 0.0);
  for (std::size_t d = 0; d < nD; ++d) {
    for (std::size_t t = 0; t < nT; ++t) {
      const std::size_t src_idx = d * nT + t;
      const std::size_t dst_idx = t * nD + d;
      dst[dst_idx] = src[src_idx];
    }
  }
  return dst;
}

}  // namespace

TmatFile load_tmat(const std::string& filepath, const TmatLoadMode mode) {
#if TENRYU_ENABLE_HDF5
  tmat_require(!filepath.empty(),
               "TMAT_E001",
               "TMAT file path must not be empty");

  H5Handle<H5Fclose> file(H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
  tmat_require(file.valid(),
               "TMAT_E001",
               "Failed to open TMAT file: " + filepath);

  TmatFile out;

  const std::string format_id = read_required_attr_string(file.get(), "/", "format_id");
  tmat_require(format_id == "tenryu.material_table.hdf5",
               "TMAT_E001",
               "/@format_id mismatch: expected \"tenryu.material_table.hdf5\", got \"" +
                   format_id + "\"");

  out.schema_version = read_required_attr_string(file.get(), "/", "schema_version");
  const int major = parse_schema_major(out.schema_version);
  tmat_require(major == 1,
               "TMAT_E002",
               "Unsupported schema major version in /@schema_version: " + out.schema_version);

  out.units_system = read_required_attr_string(file.get(), "/", "units_system");
  tmat_require(out.units_system == "cgs_eV",
               "TMAT_E003",
               "Units system mismatch: expected \"cgs_eV\", got \"" + out.units_system +
                   "\"");

  if (const auto req = read_optional_attr_string(file.get(), "/", "required_features");
      req.has_value()) {
    out.required_features = parse_required_features(*req);
  }

  out.material = load_material_group(file.get());

  const bool has_eos = link_exists(file.get(), "/eos");
  const bool has_opacity = link_exists(file.get(), "/opacity");
  tmat_require(has_eos || has_opacity,
               "TMAT_E013",
               "No core payload present: both /eos and /opacity are absent");

  if (has_eos) {
    out.eos = load_eos_group(file.get());
  }
  if (has_opacity) {
    out.opacity = load_opacity_group(file.get(), mode);
  }
  if (link_exists(file.get(), "/ionization")) {
    out.ionization =
        load_ionization_group(file.get(), out.material, out.eos ? &*out.eos : nullptr);
  }

  return out;
#else
  (void)filepath;
  (void)mode;
  TENRYU_ASSERT(false, "TMAT reader requires TENRYU_ENABLE_HDF5=ON");
  return {};
#endif
}

EOSTablePair tmat_eos_to_table_pair(const TmatEOSData& eos) {
  tmat_require(eos.ndens > 0 && eos.ntemp > 0,
               "TMAT_E006",
               "tmat_eos_to_table_pair requires ndens>0 and ntemp>0");

  const std::size_t nD = static_cast<std::size_t>(eos.ndens);
  const std::size_t nT = static_cast<std::size_t>(eos.ntemp);
  const std::size_t n2d = checked_n2d(nD, nT, "TMAT EOS conversion nD*nT");

  tmat_require(eos.rho_grid.size() == nD,
               "TMAT_E006",
               "tmat_eos_to_table_pair rho_grid size mismatch");
  tmat_require(eos.T_grid_eV.size() == nT,
               "TMAT_E006",
               "tmat_eos_to_table_pair T_grid_eV size mismatch");
  tmat_require(eos.zbar.size() == n2d,
               "TMAT_E006",
               "tmat_eos_to_table_pair zbar size mismatch");
  tmat_require(eos.P_i.size() == n2d,
               "TMAT_E006",
               "tmat_eos_to_table_pair P_i size mismatch");
  tmat_require(eos.P_e.size() == n2d,
               "TMAT_E006",
               "tmat_eos_to_table_pair P_e size mismatch");
  tmat_require(eos.e_i.size() == n2d,
               "TMAT_E006",
               "tmat_eos_to_table_pair e_i size mismatch");
  tmat_require(eos.e_e.size() == n2d,
               "TMAT_E006",
               "tmat_eos_to_table_pair e_e size mismatch");

  validate_monotonic_axis(eos.rho_grid, "tmat_eos_to_table_pair rho_grid", true, false);
  validate_monotonic_axis(eos.T_grid_eV, "tmat_eos_to_table_pair T_grid_eV", true, false);
  validate_finite_nonnegative(eos.zbar, "tmat_eos_to_table_pair zbar");
  validate_finite(eos.P_i, "tmat_eos_to_table_pair P_i");
  validate_finite(eos.P_e, "tmat_eos_to_table_pair P_e");
  validate_finite(eos.e_i, "tmat_eos_to_table_pair e_i");
  validate_finite(eos.e_e, "tmat_eos_to_table_pair e_e");

  EOSTable total;
  EOSTable electron;
  total.rho_grid = eos.rho_grid;
  total.T_grid_eV = eos.T_grid_eV;
  electron.rho_grid = eos.rho_grid;
  electron.T_grid_eV = eos.T_grid_eV;

  total.P_table.assign(n2d, 0.0);
  total.e_table.assign(n2d, 0.0);
  electron.P_table.assign(n2d, 0.0);
  electron.e_table.assign(n2d, 0.0);

  for (std::size_t d = 0; d < nD; ++d) {
    for (std::size_t t = 0; t < nT; ++t) {
      const std::size_t src = d * nT + t;
      const std::size_t dst = t * nD + d;
      total.P_table[dst] = eos.P_i[src] + eos.P_e[src];
      total.e_table[dst] = eos.e_i[src] + eos.e_e[src];
      electron.P_table[dst] = eos.P_e[src];
      electron.e_table[dst] = eos.e_e[src];
    }
  }

  total.finalize();
  electron.finalize();

  if (eos.cv_e.has_value()) {
    tmat_require(eos.cv_e->size() == n2d,
                 "TMAT_E006",
                 "tmat_eos_to_table_pair cv_e size mismatch");
    validate_finite_positive(*eos.cv_e, "tmat_eos_to_table_pair cv_e");
    electron.cv_table = transpose_dt_to_td(*eos.cv_e, nD, nT);
  }
  if (eos.cv_i.has_value() && eos.cv_e.has_value()) {
    tmat_require(eos.cv_i->size() == n2d,
                 "TMAT_E006",
                 "tmat_eos_to_table_pair cv_i size mismatch");
    validate_finite_positive(*eos.cv_i, "tmat_eos_to_table_pair cv_i");
    std::vector<double> cv_total_dt(n2d, 0.0);
    for (std::size_t i = 0; i < n2d; ++i) {
      cv_total_dt[i] = (*eos.cv_i)[i] + (*eos.cv_e)[i];
    }
    total.cv_table = transpose_dt_to_td(cv_total_dt, nD, nT);
  }

  return {std::move(total), std::move(electron)};
}

EOSTableTriplet tmat_eos_to_table_triplet(const TmatEOSData& eos, const double A_amu) {
  tmat_require(eos.ndens > 0 && eos.ntemp > 0,
               "TMAT_E006",
               "tmat_eos_to_table_triplet requires ndens>0 and ntemp>0");
  tmat_require(A_amu > 0.0 && std::isfinite(A_amu),
               "TMAT_E009",
               "tmat_eos_to_table_triplet requires finite A_amu>0");

  const std::size_t nD = static_cast<std::size_t>(eos.ndens);
  const std::size_t nT = static_cast<std::size_t>(eos.ntemp);
  const std::size_t n2d = checked_n2d(nD, nT, "TMAT EOS conversion nD*nT");

  tmat_require(eos.rho_grid.size() == nD,
               "TMAT_E006",
               "tmat_eos_to_table_triplet rho_grid size mismatch");
  tmat_require(eos.T_grid_eV.size() == nT,
               "TMAT_E006",
               "tmat_eos_to_table_triplet T_grid_eV size mismatch");
  tmat_require(eos.zbar.size() == n2d,
               "TMAT_E006",
               "tmat_eos_to_table_triplet zbar size mismatch");
  tmat_require(eos.P_i.size() == n2d,
               "TMAT_E006",
               "tmat_eos_to_table_triplet P_i size mismatch");
  tmat_require(eos.P_e.size() == n2d,
               "TMAT_E006",
               "tmat_eos_to_table_triplet P_e size mismatch");
  tmat_require(eos.e_i.size() == n2d,
               "TMAT_E006",
               "tmat_eos_to_table_triplet e_i size mismatch");
  tmat_require(eos.e_e.size() == n2d,
               "TMAT_E006",
               "tmat_eos_to_table_triplet e_e size mismatch");

  validate_monotonic_axis(eos.rho_grid, "tmat_eos_to_table_triplet rho_grid", true, false);
  validate_monotonic_axis(eos.T_grid_eV, "tmat_eos_to_table_triplet T_grid_eV", true, false);
  validate_finite_nonnegative(eos.zbar, "tmat_eos_to_table_triplet zbar");
  validate_finite(eos.P_i, "tmat_eos_to_table_triplet P_i");
  validate_finite(eos.P_e, "tmat_eos_to_table_triplet P_e");
  validate_finite(eos.e_i, "tmat_eos_to_table_triplet e_i");
  validate_finite(eos.e_e, "tmat_eos_to_table_triplet e_e");

  std::vector<double> rho_mass_grid(nD, 0.0);
  for (std::size_t d = 0; d < nD; ++d) {
    const double rho = eos.rho_grid[d] * A_amu * core::constants::proton_mass;
    tmat_require(rho > 0.0 && std::isfinite(rho),
                 "TMAT_E009",
                 "tmat_eos_to_table_triplet converted rho grid must be finite and > 0");
    if (d > 0) {
      tmat_require(rho > rho_mass_grid[d - 1],
                   "TMAT_E008",
                   "tmat_eos_to_table_triplet converted rho grid must be strictly increasing");
    }
    rho_mass_grid[d] = rho;
  }

  EOSTable ion;
  EOSTable electron;
  EOSTable total;
  ion.rho_grid = rho_mass_grid;
  ion.T_grid_eV = eos.T_grid_eV;
  electron.rho_grid = rho_mass_grid;
  electron.T_grid_eV = eos.T_grid_eV;
  total.rho_grid = rho_mass_grid;
  total.T_grid_eV = eos.T_grid_eV;

  ion.P_table.assign(n2d, 0.0);
  ion.e_table.assign(n2d, 0.0);
  electron.P_table.assign(n2d, 0.0);
  electron.e_table.assign(n2d, 0.0);
  total.P_table.assign(n2d, 0.0);
  total.e_table.assign(n2d, 0.0);

  for (std::size_t d = 0; d < nD; ++d) {
    for (std::size_t t = 0; t < nT; ++t) {
      const std::size_t src = d * nT + t;
      const std::size_t dst = t * nD + d;
      ion.P_table[dst] = eos.P_i[src];
      ion.e_table[dst] = eos.e_i[src];
      electron.P_table[dst] = eos.P_e[src];
      electron.e_table[dst] = eos.e_e[src];
      total.P_table[dst] = eos.P_i[src] + eos.P_e[src];
      total.e_table[dst] = eos.e_i[src] + eos.e_e[src];
    }
  }

  ion.finalize();
  electron.finalize();
  total.finalize();

  if (eos.cv_i.has_value()) {
    tmat_require(eos.cv_i->size() == n2d,
                 "TMAT_E006",
                 "tmat_eos_to_table_triplet cv_i size mismatch");
    validate_finite_positive(*eos.cv_i, "tmat_eos_to_table_triplet cv_i");
    ion.cv_table = transpose_dt_to_td(*eos.cv_i, nD, nT);
  }
  if (eos.cv_e.has_value()) {
    tmat_require(eos.cv_e->size() == n2d,
                 "TMAT_E006",
                 "tmat_eos_to_table_triplet cv_e size mismatch");
    validate_finite_positive(*eos.cv_e, "tmat_eos_to_table_triplet cv_e");
    electron.cv_table = transpose_dt_to_td(*eos.cv_e, nD, nT);
  }
  if (eos.cv_i.has_value() && eos.cv_e.has_value()) {
    std::vector<double> cv_total_dt(n2d, 0.0);
    for (std::size_t i = 0; i < n2d; ++i) {
      cv_total_dt[i] = (*eos.cv_i)[i] + (*eos.cv_e)[i];
    }
    total.cv_table = transpose_dt_to_td(cv_total_dt, nD, nT);
  }

  return {std::move(ion), std::move(electron), std::move(total)};
}

IonmixZbarTable tmat_eos_to_zbar_table(const TmatEOSData& eos, const double A_amu) {
  tmat_require(eos.ndens > 0 && eos.ntemp > 0,
               "TMAT_E006",
               "tmat_eos_to_zbar_table requires ndens>0 and ntemp>0");
  tmat_require(A_amu > 0.0 && std::isfinite(A_amu),
               "TMAT_E009",
               "tmat_eos_to_zbar_table requires finite A_amu>0");

  const std::size_t nD = static_cast<std::size_t>(eos.ndens);
  const std::size_t nT = static_cast<std::size_t>(eos.ntemp);
  const std::size_t n2d = checked_n2d(nD, nT, "TMAT EOS Zbar conversion nD*nT");

  tmat_require(eos.rho_grid.size() == nD,
               "TMAT_E006",
               "tmat_eos_to_zbar_table rho_grid size mismatch");
  tmat_require(eos.T_grid_eV.size() == nT,
               "TMAT_E006",
               "tmat_eos_to_zbar_table T_grid_eV size mismatch");
  tmat_require(eos.zbar.size() == n2d,
               "TMAT_E006",
               "tmat_eos_to_zbar_table zbar size mismatch");

  validate_monotonic_axis(eos.rho_grid, "tmat_eos_to_zbar_table rho_grid", true, false);
  validate_monotonic_axis(eos.T_grid_eV, "tmat_eos_to_zbar_table T_grid_eV", true, false);
  validate_finite_nonnegative(eos.zbar, "tmat_eos_to_zbar_table zbar");

  IonmixZbarTable out;
  out.rho_grid.resize(nD, 0.0);
  out.T_grid_eV = eos.T_grid_eV;
  out.zbar_table.assign(n2d, 0.0);

  for (std::size_t d = 0; d < nD; ++d) {
    // TMAT stores /eos/grid/ni_cm3, but tabular Zbar interpolation expects rho [g/cm^3].
    const double rho = eos.rho_grid[d] * A_amu * core::constants::proton_mass;
    tmat_require(rho > 0.0 && std::isfinite(rho),
                 "TMAT_E009",
                 "tmat_eos_to_zbar_table converted rho grid must be finite and > 0");
    if (d > 0) {
      tmat_require(rho > out.rho_grid[d - 1],
                   "TMAT_E008",
                   "tmat_eos_to_zbar_table converted rho grid must be strictly increasing");
    }
    out.rho_grid[d] = rho;
  }

  for (std::size_t d = 0; d < nD; ++d) {
    for (std::size_t t = 0; t < nT; ++t) {
      const std::size_t src = d * nT + t;  // TMAT native layout: [D,T]
      out.zbar_table[out.flat_index(d, t)] = eos.zbar[src];
    }
  }

  out.log_rho_grid.resize(nD, 0.0);
  for (std::size_t d = 0; d < nD; ++d) {
    out.log_rho_grid[d] = std::log(out.rho_grid[d]);
  }

  out.log_T_grid.resize(nT, 0.0);
  for (std::size_t t = 0; t < nT; ++t) {
    out.log_T_grid[t] = std::log(out.T_grid_eV[t]);
  }

  return out;
}

IonmixOpacityData tmat_to_ionmix_opacity(const TmatOpacityData& opacity,
                                         const bool skip_lte_repair) {
  tmat_require(opacity.ngroups > 0 && opacity.ndens > 0 && opacity.ntemp > 0,
               "TMAT_E006",
               "tmat_to_ionmix_opacity requires ngroups>0, ndens>0, ntemp>0");

  const std::size_t nG = static_cast<std::size_t>(opacity.ngroups);
  const std::size_t nD = static_cast<std::size_t>(opacity.ndens);
  const std::size_t nT = static_cast<std::size_t>(opacity.ntemp);
  const std::size_t n3d = checked_n3d(nG, nD, nT, "TMAT opacity conversion nG*nD*nT");

  tmat_require(opacity.rho_grid.size() == nD,
               "TMAT_E006",
               "tmat_to_ionmix_opacity rho_grid size mismatch");
  tmat_require(opacity.T_grid_eV.size() == nT,
               "TMAT_E006",
               "tmat_to_ionmix_opacity T_grid_eV size mismatch");
  tmat_require(opacity.bounds_eV.size() == nG + 1,
               "TMAT_E006",
               "tmat_to_ionmix_opacity bounds_eV size mismatch");
  tmat_require(opacity.kappa_R.size() == n3d,
               "TMAT_E006",
               "tmat_to_ionmix_opacity kappa_R size mismatch");
  tmat_require(opacity.kappa_PA.size() == n3d,
               "TMAT_E006",
               "tmat_to_ionmix_opacity kappa_PA size mismatch");
  tmat_require(opacity.kappa_PE.size() == n3d,
               "TMAT_E006",
               "tmat_to_ionmix_opacity kappa_PE size mismatch");

  validate_monotonic_axis(opacity.rho_grid,
                          "tmat_to_ionmix_opacity rho_grid",
                          true,
                          false);
  validate_monotonic_axis(opacity.T_grid_eV,
                          "tmat_to_ionmix_opacity T_grid_eV",
                          true,
                          false);
  validate_monotonic_axis(opacity.bounds_eV,
                          "tmat_to_ionmix_opacity bounds_eV",
                          false,
                          true);
  validate_finite_nonnegative(opacity.kappa_R, "tmat_to_ionmix_opacity kappa_R");
  validate_finite_nonnegative(opacity.kappa_PA, "tmat_to_ionmix_opacity kappa_PA");
  validate_finite_nonnegative(opacity.kappa_PE, "tmat_to_ionmix_opacity kappa_PE");

  IonmixOpacityData out;
  out.ngroups = opacity.ngroups;
  out.ndens = opacity.ndens;
  out.ntemp = opacity.ntemp;
  out.temps_eV = opacity.T_grid_eV;
  out.bounds_eV = opacity.bounds_eV;
  out.kappa_R = opacity.kappa_R;
  out.kappa_PA = opacity.kappa_PA;
  out.kappa_PE = opacity.kappa_PE;
  out.has_PE = true;
  out.is_lte = opacity.is_lte;
  out.numdens_cm3 = opacity.rho_grid;

  if (!out.is_lte && !skip_lte_repair) {
    constexpr double gamma_floor = 0.9;
    const std::size_t total_nodes = checked_n2d(nD, nT, "lte_repair node count");
    const auto node_index = [nT](const int d, const int t) {
      return static_cast<std::size_t>(d) * nT + static_cast<std::size_t>(t);
    };

    std::vector<unsigned char> repair_flags(total_nodes, 0);
    for (int d = 0; d < out.ndens; ++d) {
      for (int t = 0; t < out.ntemp; ++t) {
        double sigma_pa_node = 0.0;
        double sigma_pe_node = 0.0;
        for (int g = 0; g < out.ngroups; ++g) {
          const std::size_t idx = out.flat_index(g, d, t);
          sigma_pa_node += out.kappa_PA[idx];
          sigma_pe_node += out.kappa_PE[idx];
        }

        const double gamma = sigma_pe_node / std::max(sigma_pa_node, 1.0e-30);
        if (gamma < gamma_floor) {
          repair_flags[node_index(d, t)] = 1;
        }
      }
    }

    std::vector<unsigned char> expanded_flags = repair_flags;
    for (int d = 0; d < out.ndens; ++d) {
      for (int t = 0; t + 1 < out.ntemp; ++t) {
        if (repair_flags[node_index(d, t)] != 0) {
          expanded_flags[node_index(d, t + 1)] = 1;
        }
      }
    }

    std::size_t repaired_nodes = 0;
    for (int d = 0; d < out.ndens; ++d) {
      for (int t = 0; t < out.ntemp; ++t) {
        if (expanded_flags[node_index(d, t)] == 0) {
          continue;
        }
        ++repaired_nodes;
        for (int g = 0; g < out.ngroups; ++g) {
          const std::size_t idx = out.flat_index(g, d, t);
          out.kappa_PE[idx] = out.kappa_PA[idx];
        }
      }
    }

    const double repaired_pct =
        (total_nodes == 0)
            ? 0.0
            : 100.0 * static_cast<double>(repaired_nodes) / static_cast<double>(total_nodes);
    core::log_info("[lte_repair] repaired " + std::to_string(repaired_nodes) +
                   " nodes out of " + std::to_string(total_nodes) + " total (" +
                   std::to_string(repaired_pct) + "%)");
  }

  out.log_temps.resize(nT, 0.0);
  for (std::size_t i = 0; i < nT; ++i) {
    out.log_temps[i] = std::log(out.temps_eV[i]);
  }

  out.log_numdens.resize(nD, 0.0);
  for (std::size_t i = 0; i < nD; ++i) {
    out.log_numdens[i] = std::log(out.numdens_cm3[i]);
  }

  return out;
}

ZeffRatioTable tmat_ionization_to_zeff_ratio(const TmatIonizationData& ion,
                                             const TmatMaterialInfo& material) {
  tmat_require(ion.ndens > 0 && ion.ntemp > 0 && ion.n_stages >= 0,
               "TMAT_E006",
               "tmat_ionization_to_zeff_ratio requires ndens>0, ntemp>0, n_stages>=0");
  const std::size_t nD = static_cast<std::size_t>(ion.ndens);
  const std::size_t nT = static_cast<std::size_t>(ion.ntemp);
  const std::size_t nS = static_cast<std::size_t>(ion.n_stages);
  const std::size_t n2d = checked_n2d(nD, nT, "TMAT ionization reduction nD*nT");
  const std::size_t n3d =
      checked_n3d(nS, nD, nT, "TMAT ionization reduction nS*nD*nT");
  const std::size_t n_elements = material.Z.size();

  tmat_require(ion.rho_grid.size() == nD,
               "TMAT_E006",
               "tmat_ionization_to_zeff_ratio rho_grid size mismatch");
  tmat_require(ion.T_grid_eV.size() == nT,
               "TMAT_E006",
               "tmat_ionization_to_zeff_ratio T_grid_eV size mismatch");
  tmat_require(ion.stage_element.size() == nS && ion.stage_charge.size() == nS,
               "TMAT_E006",
               "tmat_ionization_to_zeff_ratio stage metadata size mismatch");
  tmat_require(ion.fractions.size() == n3d,
               "TMAT_E006",
               "tmat_ionization_to_zeff_ratio fractions size mismatch");
  tmat_require(material.A_amu.size() == n_elements &&
                   material.mass_fraction.size() == n_elements,
               "TMAT_E006",
               "tmat_ionization_to_zeff_ratio material size mismatch");

  std::vector<double> number_fraction = material.number_fraction;
  if (number_fraction.empty()) {
    number_fraction.resize(n_elements, 0.0);
    double sum_w_over_a = 0.0;
    for (std::size_t e = 0; e < n_elements; ++e) {
      tmat_require(material.A_amu[e] > 0.0 && std::isfinite(material.A_amu[e]),
                   "TMAT_E009",
                   "tmat_ionization_to_zeff_ratio requires finite A_amu>0");
      tmat_require(material.mass_fraction[e] >= 0.0 &&
                       std::isfinite(material.mass_fraction[e]),
                   "TMAT_E009",
                   "tmat_ionization_to_zeff_ratio requires finite mass_fraction>=0");
      number_fraction[e] = material.mass_fraction[e] / material.A_amu[e];
      sum_w_over_a += number_fraction[e];
    }
    tmat_require(sum_w_over_a > 0.0 && std::isfinite(sum_w_over_a),
                 "TMAT_E009",
                 "tmat_ionization_to_zeff_ratio cannot derive number fractions");
    for (double& value : number_fraction) {
      value /= sum_w_over_a;
    }
  } else {
    tmat_require(number_fraction.size() == n_elements,
                 "TMAT_E006",
                 "tmat_ionization_to_zeff_ratio number_fraction size mismatch");
  }

  for (std::size_t s = 0; s < nS; ++s) {
    tmat_require(ion.stage_element[s] >= 0 &&
                     static_cast<std::size_t>(ion.stage_element[s]) < n_elements,
                 "TMAT_E009",
                 "tmat_ionization_to_zeff_ratio stage_element out of range");
    const std::size_t e = static_cast<std::size_t>(ion.stage_element[s]);
    tmat_require(ion.stage_charge[s] >= 0 && ion.stage_charge[s] <= material.Z[e],
                 "TMAT_E009",
                 "tmat_ionization_to_zeff_ratio stage_charge out of range");
  }

  ZeffRatioTable out;
  out.ndens = ion.ndens;
  out.ntemp = ion.ntemp;
  out.ni_grid = ion.rho_grid;
  out.T_grid_eV = ion.T_grid_eV;
  out.ratio.assign(n2d, 1.0);
  out.ratio4.assign(n2d, 1.0);
  for (std::size_t d = 0; d < nD; ++d) {
    for (std::size_t t = 0; t < nT; ++t) {
      double zbar_frac = 0.0;
      double z2 = 0.0;
      double z4 = 0.0;
      for (std::size_t s = 0; s < nS; ++s) {
        const std::size_t e = static_cast<std::size_t>(ion.stage_element[s]);
        const double q = static_cast<double>(ion.stage_charge[s]);
        const double fraction = ion.fractions[(s * nD + d) * nT + t];
        zbar_frac += number_fraction[e] * q * fraction;
        z2 += number_fraction[e] * q * q * fraction;
        z4 += number_fraction[e] * q * q * q * q * fraction;
      }
      out.ratio[d * nT + t] =
          (zbar_frac > 1.0e-6)
              ? std::clamp(z2 / (zbar_frac * zbar_frac), 1.0, 10.0)
              : 1.0;
      const double zbar2 = zbar_frac * zbar_frac;
      out.ratio4[d * nT + t] =
          (zbar_frac > 1.0e-6)
              ? std::clamp(z4 / (zbar2 * zbar2), 1.0, 100.0)
              : 1.0;
    }
  }
  return out;
}

ZeffRatioTable resample_zeff_ratio_log_uniform(const ZeffRatioTable& src,
                                               const int nd_out,
                                               const int nt_out) {
  tmat_require(nd_out >= 2,
               "TMAT_E006",
               "resample_zeff_ratio_log_uniform requires nd_out>=2");
  tmat_require(nt_out >= 2,
               "TMAT_E006",
               "resample_zeff_ratio_log_uniform requires nt_out>=2");

  const std::size_t nD = static_cast<std::size_t>(src.ndens);
  const std::size_t nT = static_cast<std::size_t>(src.ntemp);
  std::vector<double> log_ni(nD, 0.0);
  std::vector<double> log_T(nT, 0.0);
  for (std::size_t d = 0; d < nD; ++d) {
    log_ni[d] = std::log10(src.ni_grid[d]);
  }
  for (std::size_t t = 0; t < nT; ++t) {
    log_T[t] = std::log10(src.T_grid_eV[t]);
  }

  ZeffRatioTable out;
  out.ndens = nd_out;
  out.ntemp = nt_out;
  out.ni_grid.resize(static_cast<std::size_t>(nd_out));
  out.T_grid_eV.resize(static_cast<std::size_t>(nt_out));
  out.ratio.resize(static_cast<std::size_t>(nd_out) *
                   static_cast<std::size_t>(nt_out));
  out.ratio4.resize(static_cast<std::size_t>(nd_out) *
                    static_cast<std::size_t>(nt_out));

  const double log_ni_min = log_ni.front();
  const double log_ni_max = log_ni.back();
  const double log_T_min = log_T.front();
  const double log_T_max = log_T.back();
  for (int d = 0; d < nd_out; ++d) {
    const double u =
        static_cast<double>(d) / static_cast<double>(nd_out - 1);
    out.ni_grid[static_cast<std::size_t>(d)] =
        std::pow(10.0, (1.0 - u) * log_ni_min + u * log_ni_max);
  }
  for (int t = 0; t < nt_out; ++t) {
    const double u =
        static_cast<double>(t) / static_cast<double>(nt_out - 1);
    out.T_grid_eV[static_cast<std::size_t>(t)] =
        std::pow(10.0, (1.0 - u) * log_T_min + u * log_T_max);
  }
  out.ni_grid.front() = src.ni_grid.front();
  out.ni_grid.back() = src.ni_grid.back();
  out.T_grid_eV.front() = src.T_grid_eV.front();
  out.T_grid_eV.back() = src.T_grid_eV.back();

  const auto bracket = [](const std::vector<double>& grid, const double x) {
    const auto upper = std::upper_bound(grid.begin(), grid.end(), x);
    if (upper == grid.begin()) {
      return std::array<std::size_t, 2>{0, 0};
    }
    if (upper == grid.end()) {
      return std::array<std::size_t, 2>{grid.size() - 1, grid.size() - 1};
    }
    const std::size_t hi = static_cast<std::size_t>(upper - grid.begin());
    return std::array<std::size_t, 2>{hi - 1, hi};
  };

  for (int d = 0; d < nd_out; ++d) {
    const double x =
        std::log10(out.ni_grid[static_cast<std::size_t>(d)]);
    const auto d_bracket = bracket(log_ni, x);
    const double x0 = log_ni[d_bracket[0]];
    const double x1 = log_ni[d_bracket[1]];
    const double tx = (x1 > x0) ? (x - x0) / (x1 - x0) : 0.0;
    for (int t = 0; t < nt_out; ++t) {
      const double y =
          std::log10(out.T_grid_eV[static_cast<std::size_t>(t)]);
      const auto t_bracket = bracket(log_T, y);
      const double y0 = log_T[t_bracket[0]];
      const double y1 = log_T[t_bracket[1]];
      const double ty = (y1 > y0) ? (y - y0) / (y1 - y0) : 0.0;

      const auto interpolate = [&](const std::vector<double>& values) {
        const auto index = [&](const std::size_t sd, const std::size_t st) {
          return sd * nT + st;
        };
        const double v00 = values[index(d_bracket[0], t_bracket[0])];
        const double v10 = values[index(d_bracket[1], t_bracket[0])];
        const double v01 = values[index(d_bracket[0], t_bracket[1])];
        const double v11 = values[index(d_bracket[1], t_bracket[1])];
        const double vx0 = v00 + tx * (v10 - v00);
        const double vx1 = v01 + tx * (v11 - v01);
        return vx0 + ty * (vx1 - vx0);
      };

      const std::size_t out_index =
          static_cast<std::size_t>(d) * static_cast<std::size_t>(nt_out) +
          static_cast<std::size_t>(t);
      out.ratio[out_index] = interpolate(src.ratio);
      out.ratio4[out_index] = interpolate(src.ratio4);
    }
  }

  return out;
}

}  // namespace tenryu::materials
