#include "materials/sesame_reader.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/namelist/errors.hpp"

namespace tenryu::materials {
namespace {

constexpr double kKelvinToEv = 8.6174e-5;
constexpr double kGPaToDynePerCm2 = 1.0e10;
constexpr double kMJPerKgToErgPerG = 1.0e10;
bool g_warned_unused_opacity_tables = false;

struct RecordHeader {
  int material_id = -1;
  int table_id = -1;
  int n_words = 0;
};

enum class SesameFormat {
  kStandard,
  kXSesame,
};

bool is_blank_line(const std::string& line) {
  return std::all_of(line.begin(), line.end(), [](const unsigned char c) {
    return std::isspace(c) != 0;
  });
}

std::string trim_copy(const std::string& input) {
  const auto first =
      std::find_if_not(input.begin(), input.end(), [](const unsigned char c) {
        return std::isspace(c) != 0;
      });
  if (first == input.end()) {
    return {};
  }
  const auto last =
      std::find_if_not(input.rbegin(), input.rend(), [](const unsigned char c) {
        return std::isspace(c) != 0;
      }).base();
  return std::string(first, last);
}

bool parse_strict_int_token(const std::string& token, int& out) {
  std::size_t parsed = 0;
  long long value = 0;
  try {
    value = std::stoll(token, &parsed, 10);
  } catch (...) {
    return false;
  }
  if (parsed != token.size()) {
    return false;
  }
  if (value < static_cast<long long>(std::numeric_limits<int>::min()) ||
      value > static_cast<long long>(std::numeric_limits<int>::max())) {
    return false;
  }
  out = static_cast<int>(value);
  return true;
}

bool parse_standard_header_line(const std::string& line, RecordHeader& header) {
  std::stringstream ss(line);
  if (!(ss >> header.material_id >> header.table_id >> header.n_words)) {
    return false;
  }

  std::string token_after_header;
  if (!(ss >> token_after_header)) {
    return header.material_id > 0 && header.table_id > 0 && header.n_words >= 0;
  }

  int maybe_int = 0;
  if (parse_strict_int_token(token_after_header, maybe_int)) {
    return false;
  }
  return header.material_id > 0 && header.table_id > 0 && header.n_words >= 0;
}

bool parse_xsesame_header_line(const std::string& line, RecordHeader& header) {
  std::stringstream ss(line);
  int record_index = -1;
  std::string flag;
  if (!(ss >> record_index >> header.material_id >> header.table_id >>
        header.n_words >> flag)) {
    return false;
  }

  int maybe_int = 0;
  if (parse_strict_int_token(flag, maybe_int)) {
    return false;
  }
  return (record_index == 0 || record_index == 1) && header.material_id > 0 &&
         header.table_id > 0 && header.n_words >= 0;
}

bool parse_header_line(const std::string& line,
                       const SesameFormat format,
                       RecordHeader& header) {
  if (format == SesameFormat::kXSesame) {
    return parse_xsesame_header_line(line, header);
  }
  return parse_standard_header_line(line, header);
}

SesameFormat detect_sesame_format(std::istream& in) {
  const std::streampos rewind_pos = in.tellg();
  TENRYU_ASSERT(rewind_pos != std::streampos(-1),
                "SESAME format probe requires seekable stream");

  constexpr int kProbeNonBlankLines = 32;
  int inspected_nonblank = 0;
  std::string line;
  while (inspected_nonblank < kProbeNonBlankLines && std::getline(in, line)) {
    if (is_blank_line(line)) {
      continue;
    }
    ++inspected_nonblank;

    RecordHeader probe_header;
    if (parse_xsesame_header_line(line, probe_header)) {
      in.clear();
      in.seekg(rewind_pos);
      return SesameFormat::kXSesame;
    }
    if (parse_standard_header_line(line, probe_header)) {
      in.clear();
      in.seekg(rewind_pos);
      return SesameFormat::kStandard;
    }
  }

  in.clear();
  in.seekg(rewind_pos);
  throw core::namelist::ConfigError(
      "Unable to detect SESAME header format (expected standard 3-int or "
      "xSESAME 4-int+flag header)");
}

bool is_text_record(const int table_id) {
  return table_id >= 101 && table_id <= 199;
}

void skip_text_record_lines(std::istream& in, const int n_words) {
  TENRYU_ASSERT(n_words >= 0, "SESAME text record must have non-negative nWords");

  constexpr int kTextCharsPerLine = 80;
  const int lines_to_skip = (n_words + kTextCharsPerLine - 1) / kTextCharsPerLine;
  std::string line;
  for (int i = 0; i < lines_to_skip; ++i) {
    TENRYU_ASSERT(static_cast<bool>(std::getline(in, line)),
                  "SESAME text record truncated before nWords payload");
  }
}

void append_fixed_width_values(const std::string& line,
                               std::vector<double>& dst) {
  if (line.empty()) {
    return;
  }

  constexpr std::size_t kWidth = 15;
  std::size_t offset = 0;
  bool parsed_any = false;
  while (offset + kWidth <= line.size()) {
    const std::string token = trim_copy(line.substr(offset, kWidth));
    if (!token.empty()) {
      dst.push_back(std::stod(token));
      parsed_any = true;
    }
    offset += kWidth;
  }

  // Some providers break strict fixed-width formatting. Fall back to
  // whitespace parsing only if fixed-width parsing produced no values.
  if (!parsed_any) {
    std::stringstream ss(line);
    double value = 0.0;
    while (ss >> value) {
      dst.push_back(value);
    }
  }
}

std::vector<double> read_record_words(std::istream& in, const int n_words) {
  TENRYU_ASSERT(n_words >= 0, "xSESAME record must have non-negative nWords");
  std::vector<double> words;
  words.reserve(static_cast<std::size_t>(n_words));

  std::string line;
  while (static_cast<int>(words.size()) < n_words && std::getline(in, line)) {
    if (is_blank_line(line)) {
      continue;
    }
    append_fixed_width_values(line, words);
  }

  TENRYU_ASSERT(static_cast<int>(words.size()) >= n_words,
                "xSESAME record truncated before nWords payload");
  if (static_cast<int>(words.size()) > n_words) {
    words.resize(static_cast<std::size_t>(n_words));
  }
  return words;
}

std::array<int, 2> parse_grid_shape(const std::vector<double>& words) {
  TENRYU_ASSERT(words.size() >= 2, "xSESAME table must contain n_rho and n_T");
  const int n_rho = static_cast<int>(words[0]);
  const int n_T = static_cast<int>(words[1]);
  TENRYU_ASSERT(n_rho > 0 && n_T > 0, "xSESAME table shape must be positive");
  return {n_rho, n_T};
}

SesameEOSTableRaw parse_eos_table(const std::vector<double>& words) {
  const auto [n_rho, n_T] = parse_grid_shape(words);
  const std::size_t n_points =
      static_cast<std::size_t>(n_rho) * static_cast<std::size_t>(n_T);
  const std::size_t expected_words =
      2ULL + static_cast<std::size_t>(n_rho) + static_cast<std::size_t>(n_T) +
      2ULL * n_points;
  TENRYU_ASSERT(words.size() >= expected_words,
                "xSESAME EOS table does not contain enough words");

  SesameEOSTableRaw table;
  table.rho_grid.resize(static_cast<std::size_t>(n_rho));
  table.T_grid_eV.resize(static_cast<std::size_t>(n_T));
  table.pressure_dyne_per_cm2.resize(n_points);
  table.energy_erg_per_g.resize(n_points);

  const std::size_t rho_offset = 2;
  const std::size_t T_offset = rho_offset + static_cast<std::size_t>(n_rho);
  const std::size_t P_offset = T_offset + static_cast<std::size_t>(n_T);
  const std::size_t e_offset = P_offset + n_points;

  for (int i = 0; i < n_rho; ++i) {
    table.rho_grid[static_cast<std::size_t>(i)] =
        words[rho_offset + static_cast<std::size_t>(i)];
  }
  for (int j = 0; j < n_T; ++j) {
    table.T_grid_eV[static_cast<std::size_t>(j)] =
        words[T_offset + static_cast<std::size_t>(j)] * kKelvinToEv;
  }
  for (std::size_t idx = 0; idx < n_points; ++idx) {
    table.pressure_dyne_per_cm2[idx] =
        words[P_offset + idx] * kGPaToDynePerCm2;
    table.energy_erg_per_g[idx] = words[e_offset + idx] * kMJPerKgToErgPerG;
  }

  return table;
}

SesameOpacityTableRaw parse_opacity_table(const std::vector<double>& words) {
  const auto [n_rho, n_T] = parse_grid_shape(words);
  const std::size_t n_points =
      static_cast<std::size_t>(n_rho) * static_cast<std::size_t>(n_T);
  const std::size_t expected_words =
      2ULL + static_cast<std::size_t>(n_rho) + static_cast<std::size_t>(n_T) +
      n_points;
  TENRYU_ASSERT(words.size() >= expected_words,
                "xSESAME opacity table does not contain enough words");

  SesameOpacityTableRaw table;
  table.rho_grid.resize(static_cast<std::size_t>(n_rho));
  table.T_grid_eV.resize(static_cast<std::size_t>(n_T));
  table.values.resize(n_points);

  const std::size_t rho_offset = 2;
  const std::size_t T_offset = rho_offset + static_cast<std::size_t>(n_rho);
  const std::size_t value_offset = T_offset + static_cast<std::size_t>(n_T);

  for (int i = 0; i < n_rho; ++i) {
    table.rho_grid[static_cast<std::size_t>(i)] =
        words[rho_offset + static_cast<std::size_t>(i)];
  }
  for (int j = 0; j < n_T; ++j) {
    table.T_grid_eV[static_cast<std::size_t>(j)] =
        words[T_offset + static_cast<std::size_t>(j)] * kKelvinToEv;
  }
  for (std::size_t idx = 0; idx < n_points; ++idx) {
    table.values[idx] = words[value_offset + idx];
  }

  return table;
}

}  // namespace

SesameData read_xsesame(const std::string& filename, const int material_id) {
  std::ifstream in(filename);
  TENRYU_ASSERT(in.good(), "Failed to open xSESAME file");
  const SesameFormat format = detect_sesame_format(in);

  SesameData out;
  std::string line;
  while (std::getline(in, line)) {
    if (is_blank_line(line)) {
      continue;
    }

    RecordHeader header;
    if (!parse_header_line(line, format, header)) {
      // Skip non-header lines gracefully.
      continue;
    }

    if (is_text_record(header.table_id)) {
      skip_text_record_lines(in, header.n_words);
      continue;
    }

    const std::vector<double> words = read_record_words(in, header.n_words);
    if (header.material_id != material_id) {
      continue;
    }

    switch (header.table_id) {
      case 301:
        out.table_301_total = parse_eos_table(words);
        break;
      case 304:
        out.table_304_electron = parse_eos_table(words);
        break;
      case 502:
        out.table_502_rosseland = parse_opacity_table(words);
        if (!g_warned_unused_opacity_tables) {
          core::log_warning("xSESAME opacity tables (502/505) are parsed, but tabular opacity "
                            "runtime interpolation is not enabled in this build.");
          g_warned_unused_opacity_tables = true;
        }
        break;
      case 505:
        out.table_505_planck = parse_opacity_table(words);
        if (!g_warned_unused_opacity_tables) {
          core::log_warning("xSESAME opacity tables (502/505) are parsed, but tabular opacity "
                            "runtime interpolation is not enabled in this build.");
          g_warned_unused_opacity_tables = true;
        }
        break;
      default:
        break;
    }
  }

  TENRYU_ASSERT(out.table_301_total.has_value(),
                "xSESAME material is missing mandatory table 301");
  return out;
}

}  // namespace tenryu::materials
