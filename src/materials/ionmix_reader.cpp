#include "materials/ionmix_reader.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/constants.hpp"
#include "core/error.hpp"

namespace tenryu::materials {
namespace {

constexpr std::size_t kMaxRecordBytes = 1ULL << 30;  // 1 GB
constexpr std::size_t kMaxTableSize = 1024;          // max points per axis
constexpr std::size_t kMaxOpacityGroups = 300;       // max opacity groups
constexpr std::size_t kMaxTableEntries = 100000000;  // ndens*ntemp*ngroups guard
// Keep interpolation strictly in log-log space. This floor is far below
// physically relevant opacities and prevents log(0) discontinuities.
constexpr double kKappaFloor = 1.0e-100;
constexpr double kFloorAwareThreshold = 1.0e-30;

static_assert(core::constants::proton_mass > 0.0,
              "proton_mass must stay positive for sigma = rho * kappa conversion");

template <typename T>
T byte_swap(const T val) {
  static_assert(std::is_trivially_copyable_v<T>,
                "byte_swap requires trivially copyable type");
  T out{};
  const auto* src = reinterpret_cast<const unsigned char*>(&val);
  auto* dst = reinterpret_cast<unsigned char*>(&out);
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    dst[i] = src[sizeof(T) - 1 - i];
  }
  return out;
}

class FortranSequentialReader {
 public:
  explicit FortranSequentialReader(const std::string& filename)
      : in_(filename, std::ios::binary), filename_(filename) {
    if (!in_.good()) {
      throw std::runtime_error("Failed to open IONMIX file: " + filename);
    }
    in_.seekg(0, std::ios::end);
    const std::streamoff end_pos = in_.tellg();
    file_size_ = (end_pos > 0) ? static_cast<std::size_t>(end_pos) : 0U;
    in_.seekg(0, std::ios::beg);
  }

  [[nodiscard]] bool has_more_records() {
    return in_.peek() != std::ifstream::traits_type::eof();
  }

  [[nodiscard]] std::size_t offset() const noexcept {
    return offset_;
  }

  [[nodiscard]] bool needs_swap() const noexcept {
    return needs_swap_;
  }

  std::vector<char> read_record_bytes(const char* label) {
    const std::size_t record_offset = offset_;

    std::int32_t len_begin_raw = 0;
    if (!in_.read(reinterpret_cast<char*>(&len_begin_raw), sizeof(len_begin_raw))) {
      throw std::runtime_error("Unexpected EOF while reading IONMIX record marker (" +
                               std::string(label) + ") in " + filename_ +
                               " at offset " + std::to_string(record_offset));
    }
    offset_ += sizeof(len_begin_raw);
    const std::int32_t len_begin = decode_record_marker(len_begin_raw);

    if (len_begin < 0) {
      throw std::runtime_error("IONMIX record length marker is negative (" +
                               std::to_string(len_begin) + ") at offset " +
                               std::to_string(record_offset));
    }
    const std::size_t len = static_cast<std::size_t>(len_begin);
    if (len > kMaxRecordBytes) {
      throw std::runtime_error("IONMIX record too large: " + std::to_string(len) +
                               " bytes");
    }

    std::vector<char> payload(len);
    if (!payload.empty() &&
        !in_.read(payload.data(), static_cast<std::streamsize>(payload.size()))) {
      throw std::runtime_error("Unexpected EOF while reading IONMIX payload (" +
                               std::string(label) + ") at offset " +
                               std::to_string(record_offset));
    }
    offset_ += payload.size();

    std::int32_t len_end_raw = 0;
    const std::size_t end_marker_offset = offset_;
    if (!in_.read(reinterpret_cast<char*>(&len_end_raw), sizeof(len_end_raw))) {
      throw std::runtime_error("Unexpected EOF while reading trailing IONMIX marker (" +
                               std::string(label) + ") at offset " +
                               std::to_string(end_marker_offset));
    }
    offset_ += sizeof(len_end_raw);
    const std::int32_t len_end = decode_record_marker(len_end_raw);

    if (len_begin != len_end) {
      throw std::runtime_error("IONMIX binary record marker mismatch at offset " +
                               std::to_string(end_marker_offset));
    }

    return payload;
  }

  void skip_record(const char* label) {
    const std::size_t record_offset = offset_;

    std::int32_t len_begin_raw = 0;
    if (!in_.read(reinterpret_cast<char*>(&len_begin_raw), sizeof(len_begin_raw))) {
      throw std::runtime_error("Unexpected EOF reading IONMIX record marker (" +
                               std::string(label) + ") at offset " +
                               std::to_string(record_offset));
    }
    offset_ += sizeof(len_begin_raw);

    // Route the marker through the same endianness decoder as read_record():
    // on a byte-swapped file the raw marker is garbage (e.g. 8 -> 134217728)
    // and would seek past the payload instead of over it.
    const std::int32_t len_begin = decode_record_marker(len_begin_raw);
    if (len_begin < 0) {
      throw std::runtime_error("IONMIX record length negative (" +
                               std::to_string(len_begin) + ") at offset " +
                               std::to_string(record_offset));
    }
    const std::size_t len = static_cast<std::size_t>(len_begin);
    if (len > kMaxRecordBytes) {
      throw std::runtime_error("IONMIX record too large: " + std::to_string(len) +
                               " bytes");
    }

    if (!in_.seekg(static_cast<std::streamoff>(len), std::ios::cur)) {
      throw std::runtime_error("Unexpected EOF seeking IONMIX payload (" +
                               std::string(label) + ") at offset " +
                               std::to_string(record_offset));
    }
    offset_ += len;

    std::int32_t len_end_raw = 0;
    const std::size_t end_offset = offset_;
    if (!in_.read(reinterpret_cast<char*>(&len_end_raw), sizeof(len_end_raw))) {
      throw std::runtime_error("Unexpected EOF reading trailing IONMIX marker (" +
                               std::string(label) + ") at offset " +
                               std::to_string(end_offset));
    }
    offset_ += sizeof(len_end_raw);

    const std::int32_t len_end = decode_record_marker(len_end_raw);
    if (len_begin != len_end) {
      throw std::runtime_error("IONMIX record marker mismatch at offset " +
                               std::to_string(end_offset));
    }
  }

 private:
  [[nodiscard]] static bool is_plausible_record_marker(const std::int32_t len) {
    return len >= 0 &&
           static_cast<std::size_t>(len) <= kMaxRecordBytes;
  }

  // Full validation of a first-record marker candidate: plausible length,
  // payload + trailing marker fit inside the file, and the trailing marker
  // (interpreted in the candidate's endianness) matches the leading one.
  // Plausibility alone is defeated by small records — byte_swap(8) =
  // 134217728 is under the 1 GB cap, so a big-endian 8-byte first record
  // used to be accepted as a native 128 MB record (2026-07-26 review).
  [[nodiscard]] bool candidate_marker_validates(const std::int32_t len,
                                                const bool swapped) {
    if (!is_plausible_record_marker(len)) {
      return false;
    }
    const std::streamoff pos = in_.tellg();
    if (pos < 0) {
      return false;
    }
    const std::size_t payload_start = static_cast<std::size_t>(pos);
    const std::size_t len_us = static_cast<std::size_t>(len);
    if (payload_start + len_us + sizeof(std::int32_t) > file_size_) {
      return false;
    }
    if (!in_.seekg(static_cast<std::streamoff>(payload_start + len_us),
                   std::ios::beg)) {
      in_.clear();
      in_.seekg(pos, std::ios::beg);
      return false;
    }
    std::int32_t trailing_raw = 0;
    const bool read_ok =
        static_cast<bool>(in_.read(reinterpret_cast<char*>(&trailing_raw),
                                   sizeof(trailing_raw)));
    in_.clear();
    in_.seekg(pos, std::ios::beg);
    if (!read_ok) {
      return false;
    }
    const std::int32_t trailing = swapped ? byte_swap(trailing_raw) : trailing_raw;
    return trailing == len;
  }

  [[nodiscard]] std::int32_t decode_record_marker(const std::int32_t marker_raw) {
    if (!endianness_checked_) {
      endianness_checked_ = true;
      const std::int32_t marker_swapped = byte_swap(marker_raw);
      if (marker_raw == marker_swapped) {
        return marker_raw;  // palindromic marker carries no endianness signal
      }
      if (candidate_marker_validates(marker_raw, /*swapped=*/false)) {
        return marker_raw;  // native interpretation fully validates
      }
      if (candidate_marker_validates(marker_swapped, /*swapped=*/true)) {
        needs_swap_ = true;
        return marker_swapped;
      }
      // Legacy fallback: neither candidate fully validates (e.g. truncated
      // file) — keep the old plausibility-only preference so error messages
      // stay meaningful.
      if (is_plausible_record_marker(marker_raw)) {
        return marker_raw;
      }
      if (is_plausible_record_marker(marker_swapped)) {
        needs_swap_ = true;
        return marker_swapped;
      }
      return marker_raw;
    }
    return needs_swap_ ? byte_swap(marker_raw) : marker_raw;
  }

  std::ifstream in_;
  std::string filename_;
  std::size_t offset_ = 0;
  std::size_t file_size_ = 0;
  bool endianness_checked_ = false;
  bool needs_swap_ = false;
};

std::vector<double> bytes_to_doubles(const std::vector<char>& bytes,
                                     const char* label,
                                     const std::size_t offset,
                                     const bool needs_swap) {
  if (bytes.size() % sizeof(double) != 0) {
    throw std::runtime_error("IONMIX record '" + std::string(label) +
                             "' has invalid byte size " + std::to_string(bytes.size()) +
                             " at offset " + std::to_string(offset));
  }
  std::vector<double> out(bytes.size() / sizeof(double), 0.0);
  if (!bytes.empty()) {
    std::memcpy(out.data(), bytes.data(), bytes.size());
  }
  if (needs_swap) {
    for (double& value : out) {
      value = byte_swap(value);
    }
  }
  return out;
}

std::vector<double> read_record_doubles(FortranSequentialReader& reader,
                                        const char* label) {
  const std::size_t offset = reader.offset();
  const std::vector<char> bytes = reader.read_record_bytes(label);
  return bytes_to_doubles(bytes, label, offset, reader.needs_swap());
}

int read_record_int_as_double(FortranSequentialReader& reader, const char* label) {
  const std::vector<double> values = read_record_doubles(reader, label);
  if (values.size() != 1) {
    throw std::runtime_error("IONMIX record '" + std::string(label) +
                             "' must contain exactly one double, got " +
                             std::to_string(values.size()));
  }

  const double value = values.front();
  if (!std::isfinite(value)) {
    throw std::runtime_error("IONMIX scalar '" + std::string(label) +
                             "' is non-finite");
  }

  const double rounded = std::round(value);
  const double tol = 1.0e-10 * std::max(1.0, std::abs(value));
  if (std::abs(value - rounded) > tol) {
    throw std::runtime_error("IONMIX scalar '" + std::string(label) +
                             "' is not integer-like: " + std::to_string(value));
  }
  if (!(rounded >= 1.0 &&
        rounded <= static_cast<double>(std::numeric_limits<int>::max()))) {
    throw std::runtime_error("IONMIX scalar '" + std::string(label) +
                             "' is out of int range: " + std::to_string(value));
  }

  return static_cast<int>(rounded);
}

std::pair<std::size_t, std::size_t> bracket_index(const std::vector<double>& grid,
                                                  const double x) {
  if (grid.size() < 2) {
    return {0, 0};
  }
  if (x <= grid.front()) {
    return {0, 1};
  }
  if (x >= grid.back()) {
    return {grid.size() - 2, grid.size() - 1};
  }
  const auto it = std::upper_bound(grid.begin(), grid.end(), x);
  const std::size_t hi = static_cast<std::size_t>(it - grid.begin());
  return {hi - 1, hi};
}

double bilinear_lerp(const double tx,
                     const double ty,
                     const double v00,
                     const double v10,
                     const double v01,
                     const double v11) {
  const double vx0 = v00 + tx * (v10 - v00);
  const double vx1 = v01 + tx * (v11 - v01);
  return vx0 + ty * (vx1 - vx0);
}

struct LinearInterpResult {
  double value = 0.0;
  bool valid = false;
};

[[nodiscard]] LinearInterpResult linear_lerp_valid(const double t,
                                                   const bool v0_valid,
                                                   const double v0,
                                                   const bool v1_valid,
                                                   const double v1) {
  if (v0_valid && v1_valid) {
    return {v0 + t * (v1 - v0), true};
  }
  if (v0_valid) {
    return {v0, true};
  }
  if (v1_valid) {
    return {v1, true};
  }
  return {};
}

bool is_strictly_increasing_finite(const std::vector<double>& values,
                                   const bool require_positive,
                                   const bool allow_zero_first) {
  if (values.empty()) {
    return false;
  }
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double v = values[i];
    if (!std::isfinite(v)) {
      return false;
    }
    if (require_positive && !(v > 0.0)) {
      return false;
    }
    if (!require_positive && !allow_zero_first && !(v > 0.0)) {
      return false;
    }
    if (!require_positive && allow_zero_first && i == 0 && !(v >= 0.0)) {
      return false;
    }
    if (!require_positive && allow_zero_first && i > 0 && !(v > 0.0)) {
      return false;
    }
    if (i > 0 && !(v > values[i - 1])) {
      return false;
    }
  }
  return true;
}

void validate_axis_strict(const std::vector<double>& values,
                          const char* name,
                          const bool require_positive,
                          const bool allow_zero_first) {
  if (!is_strictly_increasing_finite(values, require_positive, allow_zero_first)) {
    throw std::runtime_error("IONMIX axis '" + std::string(name) +
                             "' must be finite and strictly increasing");
  }
}

void validate_opacity_table(const IonmixOpacityData& out,
                            const std::vector<double>& table,
                            const char* table_name) {
  const std::size_t expected = static_cast<std::size_t>(out.ngroups) *
                               static_cast<std::size_t>(out.ndens) *
                               static_cast<std::size_t>(out.ntemp);
  if (table.size() != expected) {
    throw std::runtime_error("IONMIX table '" + std::string(table_name) +
                             "' has size mismatch: expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(table.size()));
  }

  for (int g = 0; g < out.ngroups; ++g) {
    for (int d = 0; d < out.ndens; ++d) {
      for (int t = 0; t < out.ntemp; ++t) {
        const double v = table[out.flat_index(g, d, t)];
        if (!std::isfinite(v) || v < 0.0) {
          throw std::runtime_error(
              "Invalid opacity value in IONMIX file at (table=" +
              std::string(table_name) + ", T=" +
              std::to_string(out.temps_eV[static_cast<std::size_t>(t)]) +
              ", n_i=" +
              std::to_string(out.numdens_cm3[static_cast<std::size_t>(d)]) +
              ", g=" + std::to_string(g) + "): " + std::to_string(v));
        }
      }
    }
  }
}

void validate_2d_table_finite(const std::vector<double>& table, const char* table_name) {
  for (std::size_t i = 0; i < table.size(); ++i) {
    const double v = table[i];
    if (!std::isfinite(v)) {
      throw std::runtime_error("IONMIX table '" + std::string(table_name) +
                               "' has non-finite entry at linear index " +
                               std::to_string(i));
    }
  }
}

}  // namespace

double IonmixOpacityData::interpolate_kappa(const std::vector<double>& kappa_table,
                                            const int group,
                                            const double ni_cm3,
                                            const double T_eV) const {
  TENRYU_ASSERT(group >= 0 && group < ngroups,
                "IonmixOpacityData::interpolate_kappa group out of range");
  TENRYU_ASSERT(!temps_eV.empty() && !numdens_cm3.empty(),
                "IonmixOpacityData::interpolate_kappa requires non-empty grids");
  TENRYU_ASSERT(log_temps.size() == temps_eV.size() &&
                    log_numdens.size() == numdens_cm3.size(),
                "IonmixOpacityData::interpolate_kappa log grids are not initialized");
  TENRYU_ASSERT(kappa_table.size() ==
                    static_cast<std::size_t>(ngroups) * static_cast<std::size_t>(ndens) *
                        static_cast<std::size_t>(ntemp),
                "IonmixOpacityData::interpolate_kappa table size mismatch");

  bool clamped_to_bounds = false;
  double ni_safe = ni_cm3;
  double T_safe = T_eV;
  if (!std::isfinite(ni_safe)) {
    ni_safe = numdens_cm3.front();
    clamped_to_bounds = true;
  }
  if (!std::isfinite(T_safe)) {
    T_safe = temps_eV.front();
    clamped_to_bounds = true;
  }
  if (!(ni_safe > 0.0)) {
    ni_safe = numdens_cm3.front();
    clamped_to_bounds = true;
  }
  if (!(T_safe > 0.0)) {
    T_safe = temps_eV.front();
    clamped_to_bounds = true;
  }

  if (ni_safe < numdens_cm3.front() || ni_safe > numdens_cm3.back()) {
    clamped_to_bounds = true;
  }
  if (T_safe < temps_eV.front() || T_safe > temps_eV.back()) {
    clamped_to_bounds = true;
  }
  ni_safe = std::clamp(ni_safe, numdens_cm3.front(), numdens_cm3.back());
  T_safe = std::clamp(T_safe, temps_eV.front(), temps_eV.back());

  if (clamped_to_bounds) {
    static std::atomic<int> clamp_warn_count{0};
    const int warn_count = clamp_warn_count.fetch_add(1, std::memory_order_relaxed);
    if (warn_count < 10) {
      core::log_warning("IONMIX interpolate_kappa clamped inputs to table bounds: "
                        "group=" + std::to_string(group) +
                        ", n_i_in=" + std::to_string(ni_cm3) +
                        ", T_in=" + std::to_string(T_eV) +
                        ", n_i_used=" + std::to_string(ni_safe) +
                        ", T_used=" + std::to_string(T_safe));
    } else if (warn_count == 10) {
      core::log_warning("IONMIX interpolate_kappa clamped inputs to table bounds; "
                        "suppressing further warnings");
    }
  }

  const double x = std::log(ni_safe);
  const double y = std::log(T_safe);
  const auto [d0, d1] = bracket_index(log_numdens, x);
  const auto [t0, t1] = bracket_index(log_temps, y);

  const double x0 = log_numdens[d0];
  const double x1 = log_numdens[d1];
  const double y0 = log_temps[t0];
  const double y1 = log_temps[t1];
  const double tx = (x1 > x0) ? ((x - x0) / (x1 - x0)) : 0.0;
  const double ty = (y1 > y0) ? ((y - y0) / (y1 - y0)) : 0.0;

  const double k00 = kappa_table[flat_index(group,
                                            static_cast<int>(d0),
                                            static_cast<int>(t0))];
  const double k10 = kappa_table[flat_index(group,
                                            static_cast<int>(d1),
                                            static_cast<int>(t0))];
  const double k01 = kappa_table[flat_index(group,
                                            static_cast<int>(d0),
                                            static_cast<int>(t1))];
  const double k11 = kappa_table[flat_index(group,
                                            static_cast<int>(d1),
                                            static_cast<int>(t1))];

  const bool k00_valid = k00 > kFloorAwareThreshold;
  const bool k10_valid = k10 > kFloorAwareThreshold;
  const bool k01_valid = k01 > kFloorAwareThreshold;
  const bool k11_valid = k11 > kFloorAwareThreshold;

  if (!(k00_valid && k10_valid && k01_valid && k11_valid)) {
    const LinearInterpResult kx0 =
        linear_lerp_valid(tx, k00_valid, k00, k10_valid, k10);
    const LinearInterpResult kx1 =
        linear_lerp_valid(tx, k01_valid, k01, k11_valid, k11);
    const LinearInterpResult kxy =
        linear_lerp_valid(ty, kx0.valid, kx0.value, kx1.valid, kx1.value);
    return kxy.valid ? kxy.value : 0.0;
  }

  const double lk00 = std::log(std::max(k00, kKappaFloor));
  const double lk10 = std::log(std::max(k10, kKappaFloor));
  const double lk01 = std::log(std::max(k01, kKappaFloor));
  const double lk11 = std::log(std::max(k11, kKappaFloor));
  return std::exp(bilinear_lerp(tx, ty, lk00, lk10, lk01, lk11));
}

double IonmixOpacityData::sigma_PA(const int group,
                                   const double rho,
                                   const double T_eV,
                                   const double A) const {
  if (!(rho > 0.0) || !(T_eV > 0.0) || !(A > 0.0)) {
    return 0.0;
  }
  const double ni_cm3 = rho / (A * core::constants::proton_mass);
  if (!std::isfinite(ni_cm3) || !(ni_cm3 > 0.0)) {
    return 0.0;
  }
  return rho * interpolate_kappa(kappa_PA, group, ni_cm3, T_eV);
}

double IonmixOpacityData::sigma_PE(const int group,
                                   const double rho,
                                   const double T_eV,
                                   const double A) const {
  if (!(rho > 0.0) || !(T_eV > 0.0) || !(A > 0.0)) {
    return 0.0;
  }
  const double ni_cm3 = rho / (A * core::constants::proton_mass);
  if (!std::isfinite(ni_cm3) || !(ni_cm3 > 0.0)) {
    return 0.0;
  }
  return rho * interpolate_kappa(kappa_PE, group, ni_cm3, T_eV);
}

double IonmixOpacityData::sigma_R(const int group,
                                  const double rho,
                                  const double T_eV,
                                  const double A) const {
  if (!(rho > 0.0) || !(T_eV > 0.0) || !(A > 0.0)) {
    return 0.0;
  }
  const double ni_cm3 = rho / (A * core::constants::proton_mass);
  if (!std::isfinite(ni_cm3) || !(ni_cm3 > 0.0)) {
    return 0.0;
  }
  return rho * interpolate_kappa(kappa_R, group, ni_cm3, T_eV);
}

double IonmixZbarTable::interpolate(const double rho, const double T_eV) const {
  TENRYU_ASSERT(!rho_grid.empty() && !T_grid_eV.empty(),
                "IonmixZbarTable::interpolate requires non-empty grids");
  TENRYU_ASSERT(log_rho_grid.size() == rho_grid.size() &&
                    log_T_grid.size() == T_grid_eV.size(),
                "IonmixZbarTable::interpolate log grids are not initialized");
  TENRYU_ASSERT(zbar_table.size() == n_rho() * n_T(),
                "IonmixZbarTable::interpolate table size mismatch");

  double rho_safe = std::isfinite(rho) ? rho : rho_grid.front();
  double T_safe = std::isfinite(T_eV) ? T_eV : T_grid_eV.front();
  bool clamped_to_bounds = false;
  if (!(rho_safe > 0.0)) {
    rho_safe = rho_grid.front();
    clamped_to_bounds = true;
  }
  if (!(T_safe > 0.0)) {
    T_safe = T_grid_eV.front();
    clamped_to_bounds = true;
  }

  if (rho_safe < rho_grid.front() || rho_safe > rho_grid.back()) {
    clamped_to_bounds = true;
  }
  if (T_safe < T_grid_eV.front() || T_safe > T_grid_eV.back()) {
    clamped_to_bounds = true;
  }
  rho_safe = std::clamp(rho_safe, rho_grid.front(), rho_grid.back());
  T_safe = std::clamp(T_safe, T_grid_eV.front(), T_grid_eV.back());

  if (clamped_to_bounds) {
    static std::atomic<int> clamp_warn_count{0};
    const int warn_count = clamp_warn_count.fetch_add(1, std::memory_order_relaxed);
    if (warn_count < 10) {
      core::log_warning("IONMIX tabular Zbar interpolate clamped inputs to table bounds: "
                        "rho_in=" + std::to_string(rho) +
                        ", T_in=" + std::to_string(T_eV) +
                        ", rho_used=" + std::to_string(rho_safe) +
                        ", T_used=" + std::to_string(T_safe));
    } else if (warn_count == 10) {
      core::log_warning("IONMIX tabular Zbar interpolate clamped inputs to table bounds; "
                        "suppressing further warnings");
    }
  }

  const double x = std::log(rho_safe);
  const double y = std::log(T_safe);
  const auto [i0, i1] = bracket_index(log_rho_grid, x);
  const auto [j0, j1] = bracket_index(log_T_grid, y);

  const double x0 = log_rho_grid[i0];
  const double x1 = log_rho_grid[i1];
  const double y0 = log_T_grid[j0];
  const double y1 = log_T_grid[j1];
  const double tx = (x1 > x0) ? ((x - x0) / (x1 - x0)) : 0.0;
  const double ty = (y1 > y0) ? ((y - y0) / (y1 - y0)) : 0.0;

  const double z00 = zbar_table[flat_index(i0, j0)];
  const double z10 = zbar_table[flat_index(i1, j0)];
  const double z01 = zbar_table[flat_index(i0, j1)];
  const double z11 = zbar_table[flat_index(i1, j1)];

  return std::max(0.0, bilinear_lerp(tx, ty, z00, z10, z01, z11));
}

IonmixOpacityData load_ionmix_opacity(const std::string& filename) {
  FortranSequentialReader reader(filename);

  IonmixOpacityData out;
  out.ntemp = read_record_int_as_double(reader, "ntemp");
  out.ndens = read_record_int_as_double(reader, "ndens");

  // Composition metadata (Z and fraction arrays) are variable-length and
  // ignored by TENRYU in Phase 1.
  reader.skip_record("Z array");
  reader.skip_record("fraction array");

  out.ngroups = read_record_int_as_double(reader, "ngroups");
  if (!(out.ntemp > 0 && out.ndens > 0 && out.ngroups > 0)) {
    throw std::runtime_error("Invalid IONMIX header: ntemp=" +
                             std::to_string(out.ntemp) + ", ndens=" +
                             std::to_string(out.ndens) + ", ngroups=" +
                             std::to_string(out.ngroups));
  }

  out.temps_eV = read_record_doubles(reader, "temperature grid");
  out.numdens_cm3 = read_record_doubles(reader, "ion number density grid");

  if (out.temps_eV.size() != static_cast<std::size_t>(out.ntemp)) {
    throw std::runtime_error("IONMIX temperature grid size mismatch: expected " +
                             std::to_string(out.ntemp) + ", got " +
                             std::to_string(out.temps_eV.size()));
  }
  if (out.numdens_cm3.size() != static_cast<std::size_t>(out.ndens)) {
    throw std::runtime_error("IONMIX density grid size mismatch: expected " +
                             std::to_string(out.ndens) + ", got " +
                             std::to_string(out.numdens_cm3.size()));
  }

  const std::size_t ndens = static_cast<std::size_t>(out.ndens);
  const std::size_t ntemp = static_cast<std::size_t>(out.ntemp);
  const std::size_t ngroups = static_cast<std::size_t>(out.ngroups);
  if (ndens > (std::numeric_limits<std::size_t>::max() / ntemp)) {
    throw std::runtime_error("IONMIX dimensions overflow: ntemp=" +
                             std::to_string(out.ntemp) + " ndens=" +
                             std::to_string(out.ndens) + " ngroups=" +
                             std::to_string(out.ngroups));
  }
  const std::size_t n2d = ndens * ntemp;
  if (ngroups > (std::numeric_limits<std::size_t>::max() / n2d)) {
    throw std::runtime_error("IONMIX dimensions overflow: ntemp=" +
                             std::to_string(out.ntemp) + " ndens=" +
                             std::to_string(out.ndens) + " ngroups=" +
                             std::to_string(out.ngroups));
  }
  const std::size_t n3d = ngroups * n2d;
  if (ntemp > kMaxTableSize || ndens > kMaxTableSize ||
      ngroups > kMaxOpacityGroups) {
    throw std::runtime_error("IONMIX dimensions exceed limits: ntemp=" +
                             std::to_string(out.ntemp) + " ndens=" +
                             std::to_string(out.ndens) + " ngroups=" +
                             std::to_string(out.ngroups) +
                             " (limits: ntemp<=1024, ndens<=1024, ngroups<=300)");
  }
  if (n3d > kMaxTableEntries) {
    throw std::runtime_error("IONMIX opacity table too large: ndens*ntemp*ngroups=" +
                             std::to_string(n3d) + " exceeds limit " +
                             std::to_string(kMaxTableEntries));
  }

  for (int block = 0; block < 12; ++block) {
    const std::size_t block_offset = reader.offset();
    reader.skip_record("EOS 2D block");
    const std::size_t payload_bytes =
        reader.offset() - block_offset - 2 * sizeof(std::int32_t);
    if (payload_bytes % sizeof(double) != 0 ||
        (payload_bytes / sizeof(double)) != n2d) {
      throw std::runtime_error("IONMIX EOS block size mismatch at block " +
                               std::to_string(block + 1) + ": expected " +
                               std::to_string(n2d) + ", got " +
                               std::to_string(payload_bytes / sizeof(double)));
    }
  }

  const std::size_t expected_bounds = static_cast<std::size_t>(out.ngroups + 1);
  // IONMIX6 format: optional entropy block (n2d doubles) may precede group boundaries.
  // When n2d == ngroups+1, the heuristic cannot distinguish entropy from boundaries
  // by size alone; we rely on strict-increasing + finite + non-negative first element.
  // CAUTION: This heuristic is fragile for edge cases where entropy data happens
  // to be strictly increasing.
  std::vector<double> maybe_bounds =
      read_record_doubles(reader, "bounds or optional entropy");
  bool have_bounds = false;
  const bool ambiguous_entropy_or_bounds =
      (n2d == expected_bounds && maybe_bounds.size() == expected_bounds);
  if (ambiguous_entropy_or_bounds) {
    const bool monotonic_bounds =
        is_strictly_increasing_finite(maybe_bounds, false, true);
    core::log_warning("IONMIX ambiguous bounds/entropy record count (n2d == ngroups+1 == " +
                      std::to_string(expected_bounds) +
                      "); interpreting first record as " +
                      (monotonic_bounds ? "group boundaries" : "entropy/metadata"));
    if (monotonic_bounds) {
      have_bounds = true;
    } else {
      maybe_bounds = read_record_doubles(reader, "group boundaries");
      have_bounds = true;
    }
  } else if (maybe_bounds.size() == expected_bounds &&
             is_strictly_increasing_finite(maybe_bounds, false, true)) {
    have_bounds = true;
  } else if (maybe_bounds.size() == n2d) {
    maybe_bounds = read_record_doubles(reader, "group boundaries");
    have_bounds = true;
  }

  if (!have_bounds) {
    throw std::runtime_error("IONMIX group boundary record has invalid size: expected " +
                             std::to_string(expected_bounds) +
                             " doubles (or optional entropy block of size " +
                             std::to_string(n2d) + ")");
  }
  if (maybe_bounds.size() != expected_bounds) {
    throw std::runtime_error("IONMIX group boundary size mismatch: expected " +
                             std::to_string(expected_bounds) + ", got " +
                             std::to_string(maybe_bounds.size()));
  }
  out.bounds_eV = std::move(maybe_bounds);

  // Unit contract: IONMIX kappa tables are mass opacity [cm^2/g].
  // Conversion to macroscopic cross-section sigma [cm^-1] is done at runtime:
  // sigma = rho * kappa (see sigma_PA/sigma_PE/sigma_R).
  out.kappa_R = read_record_doubles(reader, "Rosseland opacity table");
  out.kappa_PA = read_record_doubles(reader, "Planck absorption opacity table");
  if (out.kappa_R.size() != n3d) {
    throw std::runtime_error("IONMIX Rosseland table size mismatch: expected " +
                             std::to_string(n3d) + ", got " +
                             std::to_string(out.kappa_R.size()));
  }
  if (out.kappa_PA.size() != n3d) {
    throw std::runtime_error("IONMIX Planck absorption table size mismatch: expected " +
                             std::to_string(n3d) + ", got " +
                             std::to_string(out.kappa_PA.size()));
  }

  if (reader.has_more_records()) {
    out.kappa_PE = read_record_doubles(reader, "Planck emission opacity table");
    if (out.kappa_PE.size() != n3d) {
      throw std::runtime_error("IONMIX Planck emission table size mismatch: expected " +
                               std::to_string(n3d) + ", got " +
                               std::to_string(out.kappa_PE.size()));
    }
    out.has_PE = true;
  } else {
    out.kappa_PE = out.kappa_PA;
    out.has_PE = false;
    core::log_warning("IONMIX file is missing Planck emission opacity table; "
                      "using LTE fallback kappa_PE = kappa_PA");
  }

  if (reader.has_more_records()) {
    throw std::runtime_error("IONMIX file contains unexpected trailing records");
  }

  validate_axis_strict(out.temps_eV, "temps_eV", true, false);
  validate_axis_strict(out.numdens_cm3, "numDens_cm3", true, false);
  validate_axis_strict(out.bounds_eV, "bounds_eV", false, true);

  out.log_temps.resize(out.temps_eV.size());
  for (std::size_t i = 0; i < out.temps_eV.size(); ++i) {
    out.log_temps[i] = std::log(out.temps_eV[i]);
  }

  out.log_numdens.resize(out.numdens_cm3.size());
  for (std::size_t i = 0; i < out.numdens_cm3.size(); ++i) {
    out.log_numdens[i] = std::log(out.numdens_cm3[i]);
  }

  validate_opacity_table(out, out.kappa_R, "kappa_R");
  validate_opacity_table(out, out.kappa_PA, "kappa_PA");
  validate_opacity_table(out, out.kappa_PE, "kappa_PE");

  double max_rel = 0.0;
  for (std::size_t i = 0; i < out.kappa_PA.size(); ++i) {
    const double pa = out.kappa_PA[i];
    const double pe = out.kappa_PE[i];
    const double denom = std::max({pa, pe, 1.0e-30});
    const double rel = std::abs(pa - pe) / denom;
    max_rel = std::max(max_rel, rel);
  }
  // Contract: runtime transport does not branch on is_lte, but table load DOES
  // snap kappa_PE := kappa_PA when the two agree within 1e-6 relative — this
  // makes near-LTE tables exactly Kirchhoff-consistent so the separate-
  // emissivity path degenerates cleanly to LTE instead of carrying table-
  // generation roundoff as a spurious net NLTE source (2026-07-26 review
  // documented the previous comment's "diagnostic only" claim as false).
  out.is_lte = (max_rel <= 1.0e-6);
  if (out.is_lte && out.has_PE) {
    out.kappa_PE = out.kappa_PA;
  }
  if (out.is_lte) {
    static std::atomic<bool> lte_diag_logged{false};
    bool expected = false;
    if (lte_diag_logged.compare_exchange_strong(expected,
                                                true,
                                                std::memory_order_relaxed)) {
      core::log_info("IONMIX opacity table appears LTE-equivalent "
                     "(max relative |kappa_PA-kappa_PE| = " +
                     std::to_string(max_rel) +
                     "); Phase 1 uses this as diagnostic info only.");
    }
  }

  return out;
}

IonmixEOSData load_ionmix_binary_eos(const std::string& filename) {
  FortranSequentialReader reader(filename);

  IonmixEOSData out;
  out.ntemp = read_record_int_as_double(reader, "ntemp");
  out.ndens = read_record_int_as_double(reader, "ndens");

  // Composition metadata (Z and fraction arrays) are variable-length and
  // ignored by TENRYU in Phase 1 EOS import.
  reader.skip_record("Z array");
  reader.skip_record("fraction array");

  const int ngroups = read_record_int_as_double(reader, "ngroups");
  if (!(out.ntemp > 0 && out.ndens > 0 && ngroups > 0)) {
    throw std::runtime_error("Invalid IONMIX header: ntemp=" +
                             std::to_string(out.ntemp) + ", ndens=" +
                             std::to_string(out.ndens) + ", ngroups=" +
                             std::to_string(ngroups));
  }

  out.temps_eV = read_record_doubles(reader, "temperature grid");
  out.numdens_cm3 = read_record_doubles(reader, "ion number density grid");
  if (out.temps_eV.size() != static_cast<std::size_t>(out.ntemp)) {
    throw std::runtime_error("IONMIX temperature grid size mismatch: expected " +
                             std::to_string(out.ntemp) + ", got " +
                             std::to_string(out.temps_eV.size()));
  }
  if (out.numdens_cm3.size() != static_cast<std::size_t>(out.ndens)) {
    throw std::runtime_error("IONMIX density grid size mismatch: expected " +
                             std::to_string(out.ndens) + ", got " +
                             std::to_string(out.numdens_cm3.size()));
  }
  validate_axis_strict(out.temps_eV, "temps_eV", true, false);
  validate_axis_strict(out.numdens_cm3, "numdens_cm3", true, false);

  const std::size_t ndens = static_cast<std::size_t>(out.ndens);
  const std::size_t ntemp = static_cast<std::size_t>(out.ntemp);
  if (ndens > (std::numeric_limits<std::size_t>::max() / ntemp)) {
    throw std::runtime_error("IONMIX dimensions overflow: ntemp=" +
                             std::to_string(out.ntemp) + " ndens=" +
                             std::to_string(out.ndens));
  }
  const std::size_t n2d = ndens * ntemp;
  if (ntemp > kMaxTableSize || ndens > kMaxTableSize) {
    throw std::runtime_error("IONMIX dimensions exceed limits: ntemp=" +
                             std::to_string(out.ntemp) + " ndens=" +
                             std::to_string(out.ndens) +
                             " (limits: ntemp<=1024, ndens<=1024)");
  }

  out.zbar.assign(n2d, 0.0);
  out.P_i_cgs.assign(n2d, 0.0);
  out.P_e_cgs.assign(n2d, 0.0);
  out.e_i_cgs.assign(n2d, 0.0);
  out.e_e_cgs.assign(n2d, 0.0);

  constexpr double kJToErg = 1.0e7;
  for (int block = 0; block < 12; ++block) {
    const std::vector<double> eos_block =
        read_record_doubles(reader, "EOS 2D block");
    if (eos_block.size() != n2d) {
      throw std::runtime_error("IONMIX EOS block size mismatch at block " +
                               std::to_string(block + 1) + ": expected " +
                               std::to_string(n2d) + ", got " +
                               std::to_string(eos_block.size()));
    }
    for (std::size_t i = 0; i < eos_block.size(); ++i) {
      if (!std::isfinite(eos_block[i])) {
        throw std::runtime_error("IONMIX EOS block has non-finite entry at block " +
                                 std::to_string(block + 1) +
                                 ", linear index " + std::to_string(i));
      }
    }

    switch (block + 1) {
      case 1:  // Zbar
        out.zbar = eos_block;
        break;
      case 3:  // P_i [J/cm^3] -> [dyne/cm^2]
        for (std::size_t i = 0; i < n2d; ++i) {
          out.P_i_cgs[i] = eos_block[i] * kJToErg;
        }
        break;
      case 4:  // P_e [J/cm^3] -> [dyne/cm^2]
        for (std::size_t i = 0; i < n2d; ++i) {
          out.P_e_cgs[i] = eos_block[i] * kJToErg;
        }
        break;
      case 7:  // e_i [J/g] -> [erg/g]
        for (std::size_t i = 0; i < n2d; ++i) {
          out.e_i_cgs[i] = eos_block[i] * kJToErg;
        }
        break;
      case 8:  // e_e [J/g] -> [erg/g]
        for (std::size_t i = 0; i < n2d; ++i) {
          out.e_e_cgs[i] = eos_block[i] * kJToErg;
        }
        break;
      default:  // Unused block for this importer.
        break;
    }
  }

  validate_2d_table_finite(out.zbar, "zbar");
  validate_2d_table_finite(out.P_i_cgs, "P_i_cgs");
  validate_2d_table_finite(out.P_e_cgs, "P_e_cgs");
  validate_2d_table_finite(out.e_i_cgs, "e_i_cgs");
  validate_2d_table_finite(out.e_e_cgs, "e_e_cgs");
  return out;
}

IonmixZbarTable ionmix_eos_to_zbar_table(const IonmixEOSData& eos, const double A) {
  if (!(eos.ntemp > 0 && eos.ndens > 0)) {
    throw std::runtime_error("ionmix_eos_to_zbar_table requires ntemp>0 and ndens>0");
  }
  if (!(A > 0.0) || !std::isfinite(A)) {
    throw std::runtime_error("ionmix_eos_to_zbar_table requires finite A>0");
  }

  const std::size_t n_rho = static_cast<std::size_t>(eos.ndens);
  const std::size_t n_T = static_cast<std::size_t>(eos.ntemp);
  if (n_rho > (std::numeric_limits<std::size_t>::max() / n_T)) {
    throw std::runtime_error("ionmix_eos_to_zbar_table dimensions overflow");
  }
  const std::size_t n_points = n_rho * n_T;
  if (eos.temps_eV.size() != n_T || eos.numdens_cm3.size() != n_rho ||
      eos.zbar.size() != n_points) {
    throw std::runtime_error("ionmix_eos_to_zbar_table input size mismatch");
  }

  IonmixZbarTable out;
  out.rho_grid.resize(n_rho, 0.0);
  out.T_grid_eV = eos.temps_eV;
  out.zbar_table.resize(n_points, 0.0);
  for (std::size_t i_rho = 0; i_rho < n_rho; ++i_rho) {
    out.rho_grid[i_rho] = eos.numdens_cm3[i_rho] * A * core::constants::proton_mass;
  }

  validate_axis_strict(out.rho_grid, "rho_grid", true, false);
  validate_axis_strict(out.T_grid_eV, "T_grid_eV", true, false);

  for (std::size_t i_rho = 0; i_rho < n_rho; ++i_rho) {
    for (std::size_t j_T = 0; j_T < n_T; ++j_T) {
      const std::size_t src = i_rho * n_T + j_T;  // IONMIX: density-major, T-fastest
      out.zbar_table[out.flat_index(i_rho, j_T)] = eos.zbar[src];
    }
  }
  validate_2d_table_finite(out.zbar_table, "zbar_table");

  out.log_rho_grid.resize(out.n_rho(), 0.0);
  for (std::size_t i = 0; i < out.n_rho(); ++i) {
    out.log_rho_grid[i] = std::log(out.rho_grid[i]);
  }
  out.log_T_grid.resize(out.n_T(), 0.0);
  for (std::size_t j = 0; j < out.n_T(); ++j) {
    out.log_T_grid[j] = std::log(out.T_grid_eV[j]);
  }

  return out;
}

}  // namespace tenryu::materials
